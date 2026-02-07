
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Literal

import torch
from minisgl.distributed import get_tp_info
from minisgl.env import ENV
from minisgl.utils import divide_even
from minisgl.utils.logger import init_logger

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData, make_positions

if TYPE_CHECKING:
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
    )
    from minisgl.core import Batch
    from minisgl.kvcache import BaseKVCache
    from minisgl.models import ModelConfig


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << math.ceil(math.log2(n))


logger = init_logger(__name__)


@dataclass
class FICaptureData(BaseCaptureData):
    """用于 FlashInfer CUDA Graph Capture 的数据容器"""
    @property
    def one_tensor(self) -> torch.Tensor:
        # FlashInfer 需要 last_page_len参数，在 Decode 阶段通常全为 1
        return self.seq_lens

    @property
    def indices(self) -> torch.Tensor:
        # 复用 page_table 作为 ragged indices
        return self.page_table


@dataclass
class FIMetadata(BaseAttnMetadata):
    """
    FlashInfer 元数据
    
    FlashInfer 需要特定的参数来规划 (Plan) 和执行注意力计算。
    区分 CPU 和 GPU 上的 tensors。
    """
    # fmt: off
    cu_seqlens_q_cpu:   torch.Tensor  # on cpu (Prefill)
    cu_seqlens_k_cpu:   torch.Tensor  # on cpu
    cu_seqlens_q_gpu:   torch.Tensor  # on gpu (用于 get_last_indices)
    indices:            torch.Tensor  # on gpu (Ragged Paged KV Indices)
    last_page_len_cpu:  torch.Tensor  # on cpu (每个序列最后一页的有效长度)
    num_qo_heads:       int
    num_kv_heads:       int
    head_dim:           int
    page_size:          Literal[1] # currently only support page_size=1
    pos_encoding_mode:  str
    seq_lens_cpu:       torch.Tensor  # on cpu
    dtype:              torch.dtype
    
    # FlashInfer Wrapper (BatchPrefill 或 BatchDecode)
    wrapper:            BatchPrefillWithPagedKVCacheWrapper | BatchDecodeWithPagedKVCacheWrapper
    initialized:        bool = False
    # fmt: on

    def __post_init__(self) -> None:
        assert self.page_size == 1, "Currently only page_size=1 is supported."
        assert (
            self.positions.is_cuda
            and self.cu_seqlens_k_cpu.is_cpu
            and self.cu_seqlens_q_cpu.is_cpu
            and self.cu_seqlens_q_gpu.is_cuda
            and self.indices.is_cuda
            and self.last_page_len_cpu.is_cpu
            and self.seq_lens_cpu.is_cpu
        )

    def get_positions(self) -> torch.Tensor:
        return self.positions

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.cu_seqlens_q_gpu[1 : 1 + bs] - 1


class FlashInferBackend(BaseAttnBackend):
    """
    FlashInfer 后端
    
    主要用于 Decode 阶段，支持 PagedAttention，显存利用率高。
    也支持 Prefill，但通常配合 FlashAttention 使用 Hybrid 模式。
    """
    def __init__(
        self,
        config: ModelConfig,
        kvcache: BaseKVCache,
        page_table: torch.Tensor,
    ) -> None:
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        self.config = config
        self.kvcache = kvcache
        self.device = kvcache.device
        
        # FlashInfer 需要 workspace buffer
        self.float_workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        
        # 初始化 Wrappers
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            backend="fa2",  # flashinfer fa3 is buggy, use fa2 instead
        )
        self.decode_wrappers = BatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            use_tensor_cores=self.use_tensor_cores,
            kv_layout="NHD",
        )

        # NOTE: some hack to reuse the int_workspace_buffer
        # 复用 int workspace 以节省显存
        self.int_workspace_buffer = self.prefill_wrapper._int_workspace_buffer
        self.decode_wrappers._int_workspace_buffer = self.int_workspace_buffer

        # initialize some data members
        tp_size = get_tp_info().size
        self.qo_head_local = divide_even(self.config.num_qo_heads, tp_size)
        self.kv_head_local = divide_even(self.config.num_kv_heads, tp_size)

        self.cached_ones_cpu: torch.Tensor = torch.tensor([], dtype=torch.int32, pin_memory=True)
        # for cuda graph
        self.capture_bs: List[int] = []
        self.max_graph_bs = 0
        self.graph_wrappers: Dict[int, CUDAGraphBatchDecodeWithPagedKVCacheWrapper] = {}
        self.capture: FICaptureData | None = None
        self.page_table = page_table

    @staticmethod
    def _initialize_metadata_once(metadata: FIMetadata) -> None:
        """
        初始化 FlashInfer Metadata (Wrapper Plan)
        
        调用 FlashInfer 的 `plan` 方法，预计算辅助数据结构。
        对于 Decode 阶段，这个开销在 Graph Replay 时可以被消除。
        """
        if metadata.initialized:
            return

        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        metadata.initialized = True
        if isinstance(metadata.wrapper, BatchDecodeWithPagedKVCacheWrapper):
            # Decode Plan
            metadata.wrapper.plan(
                indptr=metadata.cu_seqlens_k_cpu,
                indices=metadata.indices,
                last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                data_type=metadata.dtype,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
            )
        else:
            # Prefill Plan
            metadata.wrapper.plan(
                qo_indptr=metadata.cu_seqlens_q_cpu,
                paged_kv_indptr=metadata.cu_seqlens_k_cpu,
                paged_kv_indices=metadata.indices,
                paged_kv_last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim_qk=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
                causal=True,
            )

    def _get_ones_cpu(self, bs: int) -> torch.Tensor:
        """获取全 1 张量 (cached, pinned memory)"""
        if bs <= len(self.cached_ones_cpu):
            return self.cached_ones_cpu[:bs]
        # padding to next pow of 2
        next_len = _next_power_of_2(bs)
        self.cached_ones_cpu = torch.ones(next_len, dtype=torch.int32, pin_memory=True)
        return self.cached_ones_cpu[:bs]

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        """执行 Attention 计算"""
        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        # 确保 Plan 已执行
        self._initialize_metadata_once(metadata)
        
        # 存 KV Cache
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))
        
        # 执行计算 (BatchDecode 或 BatchPrefill)
        return metadata.wrapper.run(q=q, paged_kv_cache=kv_cache)

    def prepare_metadata(self, batch: Batch) -> None:
        """
        准备 FlashInfer 元数据
        
        主要工作是构建 ragged indices (Pged KV Indices) 和相关的 lengths 信息。
        FlashInfer 使用 ragged array 来表示 Paged KV Cache。
        """
        reqs = batch.padded_reqs

        padded_size = len(reqs)
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_q = max(seqlens_q)
        cpu_kwargs = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        device = self.device
        seq_len_cpu = torch.tensor(seqlens_k, **cpu_kwargs)
        cu_seqlens_k_cpu = torch.tensor([0] + seqlens_k, **cpu_kwargs).cumsum_(dim=0)
        
        if max_seqlen_q == 1:  # decode with all extend_len = 1
            cu_seqlens_q_cpu = torch.arange(0, padded_size + 1, **cpu_kwargs)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit
            cu_seqlens_q_cpu = cu_seqlens_k_cpu
        else:  # normal extend prefill, with partial cache hit
            cu_seqlens_q_cpu = torch.tensor([0] + seqlens_q, **cpu_kwargs).cumsum_(dim=0)
            
        batch.attn_metadata = FIMetadata(
            positions=make_positions(device, reqs),
            cu_seqlens_q_cpu=cu_seqlens_q_cpu,
            cu_seqlens_k_cpu=cu_seqlens_k_cpu,
            cu_seqlens_q_gpu=cu_seqlens_q_cpu.to(device, non_blocking=True),
            # 构建 ragged indices: 将每个请求的 page indices 拼接起来
            indices=torch.cat([self.page_table[req.table_idx, : req.device_len] for req in reqs]),
            last_page_len_cpu=self._get_ones_cpu(padded_size),
            num_qo_heads=self.qo_head_local,
            num_kv_heads=self.kv_head_local,
            head_dim=self.config.head_dim,
            page_size=1,
            pos_encoding_mode="NONE", # RoPE has been applied before attention
            seq_lens_cpu=seq_len_cpu,
            dtype=self.kvcache.dtype,
            wrapper=self.decode_wrappers if batch.is_decode else self.prefill_wrapper,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = FICaptureData.create(max_bs, max_seq_len, self.kvcache.device)
        capture.page_table = capture.page_table.view(-1)  # use 1D as ragged indices
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    @cached_property
    def use_tensor_cores(self) -> bool:
        """检查是否使用 Tensor Cores (通常 GQA >= 4 时启用)"""
        if (overriden_value := ENV.FLASHINFER_USE_TENSOR_CORES.value) is not None:
            logger.warning(f"Overriding FlashInfer tensor core usage to {overriden_value}")
            return overriden_value
        GQA = self.config.num_qo_heads // self.config.num_kv_heads
        return GQA >= 4

    def prepare_for_capture(self, batch: Batch) -> None:
        """
        为 Graph Capture 准备 Metadata
        使用 CUDAGraphBatchDecodeWithPagedKVCacheWrapper 替代普通 Wrapper。
        """
        from flashinfer import CUDAGraphBatchDecodeWithPagedKVCacheWrapper

        bs = batch.size
        assert bs in self.capture_bs and bs not in self.graph_wrappers and self.capture
        batch.padded_reqs = batch.reqs
        capture = self.capture
        
        # 初始化 CUDA Graph Wrapper
        self.graph_wrappers[bs] = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            use_tensor_cores=self.use_tensor_cores,
            indptr_buffer=capture.cu_seqlens_k[: bs + 1],
            indices_buffer=capture.indices,
            last_page_len_buffer=capture.one_tensor[:bs],
        )
        self.graph_wrappers[bs]._int_workspace_buffer = self.int_workspace_buffer
        self.prepare_metadata(batch)
        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        metadata.wrapper = self.graph_wrappers[bs]
        metadata.positions = capture.positions[:bs]
        batch.input_ids = capture.input_ids[:bs]
        batch.out_loc = capture.out_loc[:bs]
        self._initialize_metadata_once(metadata)

    def prepare_for_replay(self, batch: Batch) -> None:
        """
        为 Graph Replay 更新 Metadata
        更新 capture buffers 并重置 wrapper。
        """
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, FIMetadata) and not metadata.initialized
        assert self.capture is not None and bs in self.capture_bs
        
        self.capture.input_ids[:bs].copy_(batch.input_ids)
        self.capture.out_loc[:bs].copy_(batch.out_loc)
        self.capture.positions[:bs].copy_(metadata.positions)
        metadata.wrapper = self.graph_wrappers[bs]
        self._initialize_metadata_once(metadata)
