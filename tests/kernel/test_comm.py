"""
测试 NCCL 通信内核（多 GPU 集合通信）

本测试用于验证 PyNCCL（NCCL 的 Python 包装）的正确性和性能，包括：
1. All-Reduce 操作（多 GPU 聚合求和）
2. All-Gather 操作（多 GPU 收集数据）
3. CUDA Graph 优化（减少 kernel 启动开销）
4. 带宽测试（评估通信效率）

测试场景：
- 多进程模拟多 GPU 环境（每个进程对应一个 GPU）
- 使用 PyTorch 分布式组进行 CPU 端同步（Gloo backend）
- 使用 NCCL 进行 GPU 端高速通信

关键概念：
- **All-Reduce**: 每个 GPU 持有一个向量，计算所有向量的和，结果广播回所有 GPU
- **All-Gather**: 每个 GPU 持有一个向量片段，收集所有 GPU 的片段形成完整向量
- **CUDA Graph**: 将一系列 CUDA 操作录制为静态图，通过 replay 减少 CPU 开销
"""

import os
import time
import torch
from minisgl.distributed import set_tp_info
import minisgl.kernel as kernel
from tqdm import tqdm

from minisgl.utils import init_logger


logger = init_logger(__name__)


@torch.no_grad()
def run(tp_size: int, tp_rank: int):
    """
    单个 TP Rank 的测试入口函数
    
    每个 Rank 对应一个 GPU，在独立进程中运行。
    
    Args:
        tp_size: Tensor Parallelism 的总大小（GPU 总数）
        tp_rank: 当前 Rank 的索引（0 到 tp_size-1）
    """
    # ========================================
    # 1. 初始化 CUDA 设备和流
    # ========================================
    torch.cuda.set_device(tp_rank)
    torch.cuda.set_stream(torch.cuda.Stream(tp_rank))  # type: ignore
    stream = torch.cuda.current_stream()
    set_tp_info(tp_rank, tp_size)

    # ========================================
    # 2. 初始化 CPU 通信组（用于 NCCL ID 交换）
    # ========================================
    # 使用 Gloo backend（适用于 CPU），用于在初始化 NCCL 时交换 Unique ID
    torch.distributed.init_process_group(
        world_size=tp_size,
        rank=tp_rank,
        backend="gloo",  # CPU backend
    )

    # 获取默认的 CPU 通信组
    tp_cpu_group = torch.distributed.group.WORLD
    assert tp_cpu_group is not None, "CPU group should not be None"
    dtype = torch.float16

    # ========================================
    # 3. 配置测试参数
    # ========================================
    K = 512  # 每个元素包含 512 个基本单位
    USE_SYMM = 0  # 是否使用对称缓冲区（0=否，1=是）

    # 初始化 NCCL 通信器
    comm = kernel.init_pynccl(
        tp_rank=tp_rank,
        tp_size=tp_size,
        tp_cpu_group=tp_cpu_group,
        # 如果 USE_SYMM=1，分配固定大小的通信缓冲区（可能提升性能）
        max_size_bytes=8192 * K * dtype.itemsize if USE_SYMM else 0,
    )

    def bench_performance(f, use_graph=False):
        """
        性能基准测试函数
        
        测试给定通信操作的平均延迟和带宽。
        
        Args:
            f: 要测试的通信函数（如 lambda x: comm.all_reduce(x, "sum")）
            use_graph: 是否使用 CUDA Graph 优化（减少 kernel 启动开销）
        """
        import gc

        # 禁用垃圾回收，避免干扰性能测试
        gc.collect()
        gc.disable()

        N = 1024  # 内循环迭代次数（用于平均）
        M = 16    # 外循环迭代次数
        # 创建测试数据：8192 * K 个元素
        x = torch.zeros(8192 * K, dtype=dtype, device=f"cuda:{tp_rank}")
        
        # 预热：执行两次通信，确保所有 GPU 已初始化
        f(x)
        f(x)

        # 进度条（仅在 Rank 0 显示）
        pbar = tqdm(list(range(N)), desc="Capturing cuda graph", disable=tp_rank > 0)

        torch.cuda.synchronize()
        if use_graph:
            # ========================================
            # CUDA Graph模式：录制操作序列
            # ========================================
            # 将 N 次通信操作录制为一个静态图
            g = torch.cuda.CUDAGraph()
            graph = torch.cuda.graph(g)
            with graph:
                for _ in pbar:
                    f(x)
            cur_stream = graph.capture_stream
        else:
            # ========================================
            # 普通模式：直接执行
            # ========================================
            nonlocal stream
            f(x)
            f(x)
            cur_stream = stream

        # ========================================
        # 开始计时
        # ========================================
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(cur_stream):
            tic.record(cur_stream)
            if use_graph:
                # CUDA Graph 模式：重放录制的图（M 次）
                for _ in range(M):
                    g.replay()  # type: ignore
            else:
                # 普通模式：直接执行（M * N 次）
                for _ in range(M):
                    for _ in pbar:
                        f(x)
            toc.record(cur_stream)
        
        gc.enable()
        toc.synchronize()
        
        # ========================================
        # 计算性能指标
        # ========================================
        elapsed_time = tic.elapsed_time(toc)  # 总时间（毫秒）
        avg_time = elapsed_time * 1000 / (M * N)  # 平均时间（微秒）
        logger.info(f"Rank {tp_rank} all-reduce avg time: {avg_time: .4f} us")
        
        # 计算带宽（GB/s）
        # All-Reduce 的通信量 = 数据大小（发送和接收都算）
        bandwidth = (8192 * K * dtype.itemsize) / (avg_time * 1e3)  # GB/s
        logger.info(f"Rank {tp_rank} all-reduce bandwidth: {bandwidth:.2f} GB/s")
        
        # 打印显存使用情况
        mem_usage = torch.cuda.memory_allocated() / (1024 * 1024)
        logger.info(f"Rank {tp_rank} memory usage: {mem_usage:.2f} MB")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def test_correctness(f):
        """
        正确性测试函数
        
        验证通信操作的结果是否符合预期。
        
        Args:
            f: 要测试的通信函数
            
        测试用例：
        1. 连续多次 All-Reduce（验证累积效果）
        2. 不同 Rank 的延迟测试（验证同步机制）
        3. 部分元素为 0 的情况（验证边界情况）
        """
        # ========================================
        # 测试 1: 连续 All-Reduce
        # ========================================
        N = 4
        x = torch.ones(8192 * K, dtype=dtype, device=f"cuda:{tp_rank}")
        for _ in range(N):
            f(x)  # 每次 All-Reduce 会将 x 乘以 tp_size
        # 经过 N 次 All-Reduce，每个元素应该是 tp_size^N
        ans = pow(tp_size, N)
        y = torch.full((8192 * K,), ans, dtype=dtype, device=f"cuda:{tp_rank}")

        assert torch.allclose(x, y), f"Rank {tp_rank} failed: {x} != {y}"

        # ========================================
        # 测试 2: Rank 0 延迟测试（验证同步）
        # ========================================
        x = torch.full((8192 * K,), tp_rank, dtype=dtype, device=f"cuda:{tp_rank}")
        # Rank 0 故意延迟 1 秒，测试其他 Rank 是否会等待
        if tp_rank == 0:
            torch.cuda.synchronize()
            time.sleep(1)
        f(x)  # All-Reduce 应该等待所有 Rank
        # 所有 Rank 的和：0 + 1 + 2 + ... + (tp_size-1) = tp_size*(tp_size-1)/2
        ans = (tp_size * (tp_size - 1)) // 2
        y = torch.full((8192 * K,), ans, dtype=dtype, device=f"cuda:{tp_rank}")
        assert torch.allclose(x, y), f"Rank {tp_rank} failed: {x} != {y}"

        # ========================================
        # 测试 3: 部分元素测试
        # ========================================
        # 前半部分为 0，后半部分为 1
        x = torch.cat(
            [
                torch.zeros((8192 * K // 2,), dtype=dtype, device=f"cuda:{tp_rank}"),
                torch.ones((8192 * K // 2,), dtype=dtype, device=f"cuda:{tp_rank}"),
            ]
        )
        f(x)
        # All-Reduce 后：前半部分仍为 0，后半部分为 tp_size
        y = torch.cat(
            [
                torch.zeros((8192 * K // 2,), dtype=dtype, device=f"cuda:{tp_rank}"),
                torch.full((8192 * K // 2,), tp_size, dtype=dtype, device=f"cuda:{tp_rank}"),
            ]
        )
        assert torch.allclose(x, y), f"Rank {tp_rank} failed: {x} != {y}"

        # 额外的奇数次调用（确保状态一致）
        if N % 2 != 0:
            f(x)

        logger.info(f"Correctness check for rank {tp_rank} passed")

    # ========================================
    # 执行测试
    # ========================================
    # 1. All-Reduce 正确性测试
    test_correctness(lambda x: comm.all_reduce(x, "sum"))
    # 2. All-Reduce 性能测试
    bench_performance(lambda x: comm.all_reduce(x, "sum"))
    # 3. 再次正确性测试（确保性能测试没有副作用）
    test_correctness(lambda x: comm.all_reduce(x, "sum"))

    # ========================================
    # All-Gather 测试
    # ========================================
    # 每个 Rank 持有一个值为 tp_rank 的向量
    src = torch.full((K,), tp_rank, dtype=dtype, device=f"cuda:{tp_rank}")
    torch.cuda.synchronize()
    # 目标缓冲区：容纳所有 Rank 的数据
    dst = torch.empty((K * tp_size,), dtype=dtype, device=f"cuda:{tp_rank}")
    comm.all_gather(dst, src)
    torch.cuda.synchronize()
    
    # 验证：dst 应该包含 [0, 0, ..., 1, 1, ..., tp_size-1, tp_size-1, ...]
    expected = torch.arange(tp_size, dtype=dtype, device=f"cuda:{tp_rank}")
    expected = expected.repeat_interleave(K)  # 每个值重复 K 次
    assert torch.allclose(dst, expected), f"Rank {tp_rank} all-gather failed"
    
    # 清理：销毁进程组
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    import multiprocessing as mp

    # ========================================
    # 多进程测试入口
    # ========================================
    tp_size = 4  # 模拟 4 个 GPU
    mp.set_start_method("spawn", force=True)  # 使用 spawn 方式启动子进程
    
    # 设置 PyTorch 分布式环境变量
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # 主节点地址（本地）
    os.environ["MASTER_PORT"] = "12355"       # 通信端口
    
    # 启动多个进程，每个对应一个 Rank
    p_list = []
    for i in range(tp_size):
        p = mp.Process(target=run, args=(tp_size, i))
        p_list.append(p)
    
    try:
        # 启动所有进程
        for p in p_list:
            p.start()
        # 等待所有进程完成
        for p in p_list:
            p.join()
    except BaseException:
        # 异常时终止所有进程
        for p in p_list:
            p.terminate()
        raise
