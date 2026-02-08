
from __future__ import annotations


from datetime import timedelta
from typing import Dict, NamedTuple, Tuple

import torch
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, Context, Req, set_global_ctx
from minisgl.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info
from minisgl.kvcache import create_kvcache
from minisgl.layers import set_rope_device
from minisgl.models import create_model, load_hf_weight
from minisgl.utils import divide_even, init_logger, torch_dtype

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory, mem_GB
from .sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)


class ForwardOutput(NamedTuple):
    """
    模型Forward的输出结果
    
    包含:
    - next_tokens_gpu: GPU上的下一步预测token
    - next_tokens_cpu: 复制到CPU的下一步预测token（用于Tokenize和Detokenize）
    - copy_done_event: 用于同步GPU到CPU复制完成的CUDA事件
    """
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event


def create_page_table(shape: Tuple[int, int], device: torch.device) -> torch.Tensor:
    """
    创建页表张量，用于 PagedAttention 的逻辑页到物理页映射
    
    PagedAttention 将 KV Cache 组织为固定大小的"页"（类似操作系统的分页内存）。
    页表维护每个请求的逻辑页索引到物理页索引的映射关系。
    
    Args:
        shape: (max_requests, max_seq_len) 页表形状
            - max_requests: 支持的最大并发请求数
            - max_seq_len: 支持的最大序列长度（以 token 为单位）
        device: 页表所在的 GPU 设备
        
    Returns:
        torch.Tensor: 初始化为 0 的页表，shape 为 (max_requests, max_seq_len)，
                     dtype 为 int32。值 0 表示未分配的页。
    
    注意:
        - 页表索引从 1 开始，0 保留作为"未分配"状态
        - 实际使用中，页表会动态更新以反映当前的页映射关系
    """
    return torch.zeros(shape, dtype=torch.int32, device=device)


def _align_up_32(num: int) -> int:
    """将数值向上对齐到32的倍数，用于内存对齐"""
    return (num + 31) // 32 * 32


class Engine:
    """
    推理引擎 (Engine)
    
    负责管理模型执行的核心组件。它协调模型、KV Cache、通信和计算资源。
    
    主要职责:
    1. 初始化模型和分布式环境
    2. 管理显存和 KV Cache 分配
    3. 执行模型的 Forward Pass
    4. 处理采样的后处理
    
    架构角色:
    - 下层: 封装了 Model, KVCache, AttentionBackend 等底层组件
    - 上层: 被 Scheduler 调用，执行具体的计算任务
    """
    def __init__(self, config: EngineConfig):
        self.model_config = config.model_config
        # 设置 Tensor Parallel (TP) 信息
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)

        # 确保 CUDA 未初始化，以便正确设置 device
        assert not torch.cuda.is_initialized()
        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        torch.cuda.set_device(self.device)
        
        # 创建主计算流 (Computation Stream)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype

        # 初始化分布式通信组
        self.tp_cpu_group = self._init_communication(config)
        
        # 记录加载模型前的显存情况
        init_free_memory = self._sync_get_memory()[1]
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # 加载模型
        set_rope_device(self.device)
        with torch.device("meta"), torch_dtype(config.dtype):
            # 在 meta device 上创建模型结构，不占用实际显存
            # 这有助于在大模型初始化时节省 CPU 内存
            self.model = create_model(config.model_path, config.model_config)
        # 加载权重到实际 device
        self.model.load_state_dict(self._load_weight_state_dict(config))
        
        # 确定 KV Cache 的页数
        # 这一步非常关键：它计算模型占用后剩余的显存，并根据比例分配给 KV Cache
        self.num_pages = self.dummy_page = self._determine_num_pages(init_free_memory, config)
        
        # 创建 KV Cache 管理器 (物理显存分配)
        self.kv_cache = create_kvcache(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,  # +1 用于 dummy page（处理越界或无效访问）
            device=self.device,
            dtype=self.dtype,
        )
        
        # 创建页表 (Page Table)
        # 映射关系: [request_index, logical_page_index] -> physical_page_index
        # 128字节对齐优化
        self.max_seq_len = _align_up_32(min(config.max_seq_len, self.num_pages))
        self.page_table = create_page_table(  # + 1 for dummy request
            (config.max_running_req + 1, self.max_seq_len),
            device=self.device,
        )
        
        # 创建注意力后端 (FlashAttention / FlashInfer)
        self.attn_backend = create_attention_backend(
            config.attention_backend,
            config.model_config,
            self.kv_cache,
            self.page_table,
        )
        
        # 初始化全局上下文
        self.ctx = Context(page_size=1, attn_backend=self.attn_backend)
        set_global_ctx(self.ctx)
        
        # 初始化采样器
        self.sampler = Sampler(self.device, self.model_config.vocab_size)

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # 初始化 CUDA Graph 相关
        # Dummy request 用于 CUDA Graph 的图捕获时的占位
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        # 将 dummy request 的页表填满 dummy page
        self.page_table[self.dummy_req.table_idx].fill_(self.dummy_page)
        
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=self.max_seq_len,
            vocab_size=self.model_config.vocab_size,
            dummy_req=self.dummy_req,
        )

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        """初始化多卡通信环境 (NCCL/Gloo)"""
        if config.tp_info.size == 1 or config.use_pynccl:
            # 单卡或使用 PyNCCL 时，使用 Gloo 作为后端 (用于元数据同步)
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            # 计算前向传播可能需要的最大缓冲区大小 (用于 PyNCCL)
            max_bytes = (
                config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
            )
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            # 多卡标准模式，使用 NCCL 作为后端
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            # 使用 Gloo 创建一个 cpu group，用于同步显存信息等元数据
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
        return tp_cpu_group

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        """加载模型权重"""
        if config.use_dummy_weight:
            # 调试模式：使用随机权重
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        else:
            # 正常模式：从 Hugging Face 格式加载
            return {
                k: v.to(self.dtype)
                for k, v in load_hf_weight(config.model_path, self.device).items()
            }

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        """
        确定 KV Cache 可以使用的页数
        
        逻辑:
        1. 计算单个页面 (Page) 的显存占用 (key block + value block)
        2. 如果未指定固定页数:
           - 计算模型加载后的显存增量 (模型权重占用)
           - 根据 memory_ratio 计算允许使用的总显存上限
           - 计算剩余显存可容纳多少页
        
        Args:
            old_free_memory: 加载模型前的空闲显存
            
        Returns:
            int: 允许分配的页数
        """
        new_free_memory = self._sync_get_memory()[1]
        cache_per_page = (
            2  # key + value
            * self.model_config.head_dim
            * divide_even(self.model_config.num_kv_heads, config.tp_info.size)
            * config.page_size
            * self.dtype.itemsize
            * self.model_config.num_layers
        )
        num_pages = config.num_page_override
        if num_pages is None:
            model_memory = old_free_memory - new_free_memory
            # 计算可用显存：(总显存 * 比例) - 模型权重占用
            available_memory = int(config.memory_ratio * old_free_memory) - model_memory
            num_pages = available_memory // cache_per_page

        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-tokens"
        real_kv_size = num_pages * cache_per_page
        logger.info(f"Allocating {num_pages} pages for KV cache, K + V = {mem_GB(real_kv_size)}")
        return num_pages

    def _sync_get_memory(self) -> Tuple[int, int]:
        """
        跨所有 TP rank 同步获取最小和最大空闲显存
        
        确保所有卡上的显存状态一致，防止因显存不均衡导致的 OOM。
        
        Returns:
            Tuple[int, int]: (最小空闲显存, 最大空闲显存)
        """
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        # 使用 tensor 进行集合通信 (All-Reduce)
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:  # 2GB 阈值
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            # 显存严重不均衡可能意味着某些卡上有残留进程或泄漏
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        """
        执行一个批次的前向传播
        
        流程:
        1. 设置各种上下文 (Batch context, Attention Backend context)
        2. 尝试使用 CUDA Graph 重放 (如果适用)
        3. 否则执行常规 PyTorch forward
        4. 更新请求状态
        5. 执行采样 (Sampling)
        6. 异步将结果复制回 CPU
        
        Args:
            batch: 当前批次
            args: 采样参数
            
        Returns:
            ForwardOutput: 包含 GPU 和 CPU 端的输出 Token 以及同步事件
        """
        assert torch.cuda.current_stream() == self.stream
        with self.ctx.forward_batch(batch):
            if self.graph_runner.can_use_cuda_graph(batch):
                # 如果条件允许 (Decode阶段, batch size 匹配等)，使用 CUDA Graph 加速
                logits = self.graph_runner.replay(batch)
            else:
                # 否则使用常规执行路径
                logits = self.model.forward()

        # 更新请求状态 (device_len += 1, cached_len = device_len)
        for req in batch.reqs:
            req.complete_one()

        # 采样：从 logits 生成下一个 token
        next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
        # 异步复制到 CPU
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        
        # 记录事件以便同步
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def shutdown(self) -> None:
        """清理资源，关闭引擎"""
        self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()
