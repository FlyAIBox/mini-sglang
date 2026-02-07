
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

import torch

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData, make_positions

if TYPE_CHECKING:
    from minisgl.core import Batch
    from minisgl.kvcache import BaseKVCache
    from minisgl.models import ModelConfig


@dataclass
class FACaptureData(BaseCaptureData):
    """用于 FlashAttention CUDA Graph Capture 的数据容器"""
    pass


@dataclass
class FAMetadata(BaseAttnMetadata):
    """
    FlashAttention 元数据
    
    存储 FlashAttention 内核所需的各种索引和长度信息。
    """
    # Cumlative sequence lengths for Key/Value (KV) cache
    # [0, len_0, len_0 + len_1, ...]
    cu_seqlens_k: torch.Tensor
    
    # Cumlative sequence lengths for Query (Q)
    # Prefill: [0, q_len_0, q_len_0 + q_len_1, ...]
    # Decode:  [0, 1, 2, ..., batch_size] (每个请求只有 1 个 query token)
    cu_seqlens_q: torch.Tensor
    
    # 每个请求的 KV Cache 长度
    cache_seqlens: torch.Tensor
    
    max_seqlen_k: int
    max_seqlen_q: int

    # Block Table: 映射逻辑块到物理块 [batch_size, max_blocks]
    # 用于 PagedAttention (虽然 FA 原生不支持 Paged, 
    # 但 mini-sglang 可能通过 sgl-kernel 的修改版 FA 支持或用于其他目的)
    # 注意: 标准 FA 不直接支持 PagedAttention，通常需要 Page Table
    page_table: torch.Tensor

    def get_positions(self) -> torch.Tensor:
        return self.positions

    def get_last_indices(self, bs: int) -> torch.Tensor:
        # 获取每个序列最后一个 token 的索引 (用于 Prefill 阶段提取 logits)
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class FlashAttentionBackend(BaseAttnBackend):
    """
    FlashAttention 后端
    
    主要用于 Prefill 阶段，处理长序列计算高效。
    使用了 `sgl_kernel.flash_attn` 提供的 `flash_attn_with_kvcache` 接口。
    """
    def __init__(self, config: ModelConfig, kvcache: BaseKVCache, page_table: torch.Tensor):
        self.config = config
        self.kvcache = kvcache
        self.capture: FACaptureData | None = None
        self.max_graph_bs = 0
        self.capture_bs: List[int] = []
        self.scale = config.head_dim**-0.5
        self.page_table = page_table

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        """
        执行 Attention 计算
        
        1. 将当前 batch 的 K/V 写入 KV Cache。
        2. 调用 flash_attn_with_kvcache 执行注意力计算。
        """
        metadata = batch.attn_metadata
        assert isinstance(metadata, FAMetadata)
        
        # 将当前 step 的 KV 存入 Cache
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        
        return _fa_sgl_impl(
            q=q,
            k_cache=self.kvcache.k_cache(layer_id),
            v_cache=self.kvcache.v_cache(layer_id),
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k_new=metadata.cu_seqlens_k,
            max_seqlen_q=metadata.max_seqlen_q,
            softmax_scale=self.scale,
        )

    def prepare_metadata(self, batch: Batch) -> None:
        """
        准备元数据
        
        计算序列长度、累积长度 (cu_seqlens)、位置索引等，并拷贝到 GPU。
        """
        reqs = batch.padded_reqs

        padded_size = len(reqs)
        # extend_len: 当前 step 输入的 token 数 (prefill > 1, decode = 1)
        seqlens_q = [req.extend_len for req in reqs]
        # device_len: 当前请求的总 token 数 (cached + extend)
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        max_seqlen_k = max(seqlens_k)
        max_seqlen_q = max(seqlens_q)
        cpu_kwargs = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        device = self.kvcache.device
        
        # 1. cache_seqlens: 每个请求的历史长度
        cache_seqlens = torch.tensor(seqlens_k, **cpu_kwargs)
        cache_seqlens = cache_seqlens.to(device, non_blocking=True)
        
        # 2. cu_seqlens_k: KV 序列的累积长度
        cu_seqlens_k = torch.tensor([0] + seqlens_k, **cpu_kwargs).cumsum_(dim=0)
        cu_seqlens_k = cu_seqlens_k.to(device, non_blocking=True)

        if max_seqlen_q == 1:
            # Decode 阶段: 每个 seq 长度为 1
            cu_seqlens_q = torch.arange(0, padded_size + 1, device=device, dtype=torch.int32)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit
            # 纯 Prefill: Q 长度等于 K 长度
            cu_seqlens_q = cu_seqlens_k
        else:  # normal extend prefill, with partial cache hit
            # 增量 Prefill (Chunked Prefill): Q 长度为 extend_len
            cu_seqlens_q = torch.tensor([0] + seqlens_q, **cpu_kwargs).cumsum_(dim=0)
            cu_seqlens_q = cu_seqlens_q.to(self.kvcache.device, non_blocking=True)

        positions = make_positions(device, reqs)
        page_table = self.page_table
        # 构建当前 batch 的 page table
        new_page_table = torch.stack([page_table[req.table_idx, :max_seqlen_k] for req in reqs])

        # 3. 封装到 Metadata
        batch.attn_metadata = FAMetadata(
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
            positions=positions,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            page_table=new_page_table,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        """初始化 CUDA Graph 捕获所需的数据结构"""
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = FACaptureData.create(max_bs, max_seq_len, self.kvcache.device)
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        """为 Graph Capture 准备 Metadata (使用固定内存地址)"""
        assert (bs := batch.size) in self.capture_bs and self.capture
        capture = self.capture
        metadata = FAMetadata(
            cu_seqlens_k=capture.cu_seqlens_k[: bs + 1],
            cu_seqlens_q=capture.cu_seqlens_q[: bs + 1],
            positions=capture.positions[:bs],
            cache_seqlens=capture.seq_lens[:bs],
            max_seqlen_k=capture.page_table.size(1),
            max_seqlen_q=1,  # decode only
            page_table=capture.page_table[:bs, :],
        )
        batch.attn_metadata = metadata
        batch.input_ids = capture.input_ids[:bs]
        batch.out_loc = capture.out_loc[:bs]

    def prepare_for_replay(self, batch: Batch) -> None:
        """为 Graph Replay 更新 Metadata (拷贝新数据到固定地址)"""
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, FAMetadata)
        assert self.capture is not None and bs in self.capture_bs
        # cu_seqlens_q is always [0, 1, 2, ..., bs] for decode (i.e. no-op)
        
        # 拷贝数据到 capture buffers
        self.capture.input_ids[:bs].copy_(batch.input_ids)
        self.capture.out_loc[:bs].copy_(batch.out_loc)
        self.capture.cu_seqlens_k[: bs + 1].copy_(metadata.cu_seqlens_k)
        self.capture.positions[:bs].copy_(metadata.positions)
        self.capture.seq_lens[:bs].copy_(metadata.cache_seqlens)
        self.capture.page_table[:bs, : metadata.max_seqlen_k].copy_(metadata.page_table)


def _fa_sgl_impl(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k_new: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    sm_margin: int = 0,
    window_size: Tuple[int, int] = (-1, -1),  # -1 means infinite context window
    softcap: float = 0.0,  # 0.0 means deactivated
    num_splits: int = 0,  # Can be tuned for speed
    pack_gqa: bool | None = None,  # Can be tuned for speed
    causal: bool = True,
) -> torch.Tensor:
    """内部实现：调用 sgl-kernel 的 flash_attn"""
    try:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
    except ImportError as e:
        raise ImportError(
            "sgl_kernel.flash_attn is not found. Please install it with `pip install sgl-kernel`.\n"
            "If you're sure it's correctly installed, try `apt update && apt install libnuma1`."
        ) from e

    return flash_attn_with_kvcache(  # type: ignore
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k_new,
        max_seqlen_q=max_seqlen_q,
        softmax_scale=softmax_scale,
        sm_margin=sm_margin,
        window_size=window_size,
        softcap=softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        causal=causal,
        ver=3,  # TODO: support FA4 on blackwell
    )
