
from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Dict, List

import torch
from minisgl.core import Batch, Req, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import init_logger
from tqdm import tqdm

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend
    from minisgl.models import BaseLLMModel

logger = init_logger(__name__)


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    """
    确定需要捕获的 CUDA Graph 批次大小列表
    
    逻辑：
    1. 如果用户显式指定了 bs 列表，直接使用。
    2. 如果没有指定最大 bs，根据显存大小估算 (H200 -> 256, 其他 -> 160)。
    3. 生成一系列 bs: [1, 2, 4] + [8, 16, ..., max_bs]。
    """
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200 (80GB+)
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    # 常用的小 batch size 加上 8 的倍数
    return [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))


def mem_GB(size: int) -> str:
    return f"{size / (1024**3):.2f} GiB"


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


class GraphRunner:
    """
    CUDA Graph 运行器
    
    用于加速 Decoding 阶段的小 Batch 推理。
    
    原理：
    通过预先捕获 (Capture) 特定 Batch Size 的 CUDA Kernel 执行序列（Graph），
    并在推理时重放 (Replay) 这些 Graph，可以显著减少 CPU Launch Kernel 的开销。
    
    主要功能：
    1. 捕获不同 Batch Size 的 CUDA Graphs。
    2. 判断当前 Batch 是否可以使用 Graph 加速。
    3. 对 Batch 进行 Padding 以匹配 Graph 的 Batch Size。
    4. 重放 Graph 执行推理。
    """
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
    ) -> None:
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        self.attn_backend = attn_backend
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.stream = stream
        self.device = device
        self.graph_map = self._capture_graphs(max_seq_len, vocab_size, model)

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        """
        捕获 CUDA Graphs
        
        对每个支持的 Batch Size，执行一次模拟的前向传播并录制 Graph。
        
        Args:
            max_seq_len: 最大序列长度，用于初始化 Attention Backend
            vocab_size: 词表大小，用于预分配 logits 内存
            model: 模型实例
            
        Returns:
            Dict[int, torch.cuda.CUDAGraph]: Batch Size -> CUDAGraph 的映射
        """
        graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        if self.max_graph_bs == 0:
            logger.info_rank0("CUDA graph is disabled.")
            return graph_map

        # 预分配用于存储 CUDA Graph 输出的内存 (Logits)
        # 必须是固定内存地址，因为 Graph 会记录这些地址
        self.logits = torch.empty(
            (self.max_graph_bs, vocab_size),
            dtype=torch.float32,
            device=self.device,
        )
        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        pool = None
        for bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            
            graph = torch.cuda.CUDAGraph()
            # 使用 Dummy Reqs 构建 Batch
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            
            # 准备 Attention Backend (如 FlashInfer 需要预分配一些 buffer)
            self.attn_backend.prepare_for_capture(batch)
            
            with get_global_ctx().forward_batch(batch):
                # Warmup run (必须执行一次以初始化 pytorch cache 等)
                self.logits[:bs] = model.forward()
                
                # Capture run
                # graph pool 共享内存池，减少显存占用
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self.logits[:bs] = model.forward()
                    
            if pool is None:
                pool = graph.pool()
            graph_map[bs] = graph

        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")
        return graph_map

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        """检查当前 Batch 是否可以使用 CUDA Graph 加速"""
        # 仅 Decoding 阶段且 Batch Size 不超过最大捕获 Batch Size 时可用
        return batch.is_decode and batch.size <= self.max_graph_bs

    def replay(self, batch: Batch) -> torch.Tensor:
        """
        重放 CUDA Graph
        
        Args:
            batch: 输入 Batch
            
        Returns:
            torch.Tensor: 输出 Logits
        """
        assert self.can_use_cuda_graph(batch)
        
        # 找到匹配的 Graph (batch.padded_size 必须是 capture 的 bs 之一)
        g = self.graph_map[batch.padded_size]
        
        # 更新 Attention Backend 的 metadata (将当前 batch 的信息 拷贝到 graph 使用的固定内存地址)
        self.attn_backend.prepare_for_replay(batch)
        
        # Replay the graph
        g.replay()
        return self.logits[: batch.size]

    def pad_batch(self, batch: Batch) -> int:
        """
        对 Batch 进行 Padding
        
        因为 CUDA Graph 要求固定的 Tensor Shape，
        所以如果当前 Batch Size 不是捕获的某个大小，
        需要用 Dummy Request 填充到最近的一个可用 Batch Size。
        
        Returns:
            int: 填充的请求数量
        """
        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)
        return batch.padded_size - batch.size

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        """销毁 CUDA Graphs，释放资源"""
        del self.graph_map
        gc.collect()
