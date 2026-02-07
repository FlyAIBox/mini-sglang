
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from minisgl.utils import init_logger

logger = init_logger(__name__)


def perf_cuda(
    f: Callable[[], Any],
    *,
    init_stream: bool = True,
    repetitions: int = 10,
    cuda_graph_repetitions: int | None = 10,
) -> float:
    """
    测试 CUDA Kernel 性能
    
    精确测量 CUDA 函数执行时间，支持 CUDA Graph 预热。
    
    参数:
        f: 要测试的函数（通常包含 CUDA kernel 调用）
        init_stream: 是否在新的 CUDA Stream 中执行（避免干扰默认流）
        repetitions: 重复运行次数（取平均值）
        cuda_graph_repetitions: CUDA Graph 重放次数（用于消除 CPU launch 开销）
        
    返回:
        float: 平均执行时间（毫秒）
    """
    import torch

    assert repetitions > 0
    tic = torch.cuda.Event(enable_timing=True)
    toc = torch.cuda.Event(enable_timing=True)
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    if init_stream:
        stream = torch.cuda.Stream()
    else:
        stream = torch.cuda.current_stream()

    with torch.cuda.stream(stream):
        # 预热 (Warmup)
        f()
        
        # 使用 CUDA Graph 消除 Python/CPU 开销，测量纯 GPU kernel 时间
        if N := cuda_graph_repetitions:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(N):
                    f()
            replay = g.replay
            del g
        else:
            replay = f
            N = 1

        torch.cuda.synchronize()

        # 开始计时
        tic.record()
        for _ in range(repetitions):
            replay()
        toc.record()
        toc.synchronize()
        
        # 计算平均时间 (ms)
        dur = tic.elapsed_time(toc)
        return dur / (N * repetitions)


def compare_memory_kernel_perf(
    *,
    baseline: Callable[[], Any],
    our_impl: Callable[[], Any],
    memory_footprint: int,  # in bytes
    description: str = " ",
    extra_kwargs: Dict[str, Any] | None = None,
    need_latency: bool = True,
) -> Tuple[float, float]:
    """
    对比两个 Kernel 的内存带宽利用率
    
    计算公式: Bandwidth = Memory Footprint / Latency
    
    参数:
        baseline: 基准实现
        our_impl: 我们的优化实现
        memory_footprint: 读写的总字节数
        
    返回:
        Tuple[float, float]: (Baseline Bandwidth, Our Impl Bandwidth) in GB/s
    """
    extra_kwargs = extra_kwargs or {}

    dur = perf_cuda(baseline, **extra_kwargs)
    bandwidth_0 = memory_footprint / (dur * 1e6)  # GB/s
    latency_msg = f"{dur:8.3f} ms | " if need_latency else ""
    message_0 = f"Baseline: {latency_msg}{bandwidth_0:8.3f} GB/s"

    dur = perf_cuda(our_impl, **extra_kwargs)
    bandwidth_1 = memory_footprint / (dur * 1e6)  # GB/s
    latency_msg = f"{dur:8.3f} ms | " if need_latency else ""
    logger.info(f"{description}{message_0} | Our Impl: {latency_msg}{bandwidth_1:8.3f} GB/s")
    return bandwidth_0, bandwidth_1
