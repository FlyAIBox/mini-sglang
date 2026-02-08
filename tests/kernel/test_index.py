"""
测试 Indexing 内核（Embedding 查表操作）

本测试用于验证自定义 indexing 内核的正确性和性能，包括：
1. 基本 indexing 操作（类似 torch.embedding）
2. 带 vocab_range 掩码的 indexing（用于 Tensor Parallelism）
3. 性能对比（自定义内核 vs PyTorch 原生实现）

应用场景：
- **Embedding 层**: 将 token IDs 转换为词向量
- **输出层**: 从 logits 矩阵中选择特定行（对应词表中的词）
- **Tensor Parallelism**: 在多 GPU 上切分词表，每个 GPU 负责一部分词汇

关键优化：
- 自定义 CUDA kernel 使用更高效的内存访问模式
- 支持 vocab_range 掩码，避免跨 GPU 访问
- 针对不同 batch size 自动调整并行策略
"""

from __future__ import annotations
from typing import Tuple
import torch
import torch.nn.functional as F

from minisgl.benchmark.perf import compare_memory_kernel_perf
from minisgl.kernel import indexing
from minisgl.utils import call_if_main, init_logger

logger = init_logger(__name__)


def ref_indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    vocab_range: Tuple[int, int] | None = None,  # (start, length)
) -> torch.Tensor:
    """
    参考实现：使用 PyTorch 原生 F.embedding 实现 indexing
    
    用作正确性验证的基准，也用于性能对比。
    
    Args:
        weights: 权重矩阵 [vocab_size, embed_dim]
        indices: 索引张量 [batch_size]，每个值是 0 到 vocab_size-1 的整数
        vocab_range: 可选的词表范围 (start, length)
            - start: 当前 Rank 负责的词表起始索引
            - length: 当前 Rank 负责的词表长度
            用于 Tensor Parallelism，当索引不在当前 Rank 的范围内时，输出为 0
    
    Returns:
        输出张量 [batch_size, embed_dim]
        
    Tensor Parallelism 示例：
        假设词表大小为 128K，4 个 GPU：
        - GPU 0: vocab_range=(0, 32K)
        - GPU 1: vocab_range=(32K, 32K)
        - GPU 2: vocab_range=(64K, 32K)
        - GPU 3: vocab_range=(96K, 32K)
        
        如果 indices=[10, 40000, 70000]：
        - GPU 0 输出：[embedding[10], 0, 0]
        - GPU 1 输出：[0, embedding[40000-32K], 0]
        - GPU 2 输出：[0, 0, embedding[70000-64K]]
        最后通过 All-Reduce 求和得到完整结果
    """
    if vocab_range is not None:
        start, length = vocab_range
        assert length <= weights.shape[0]
        
        # 将全局索引转换为本地索引
        indices = indices - start
        
        # 创建掩码：标记超出范围的索引
        indices_mask = (indices < 0) | (indices >= length)
        
        # 将超出范围的索引置为 0（避免非法访问）
        indices[indices_mask] = 0
        
        # 执行 embedding 查表
        result = F.embedding(indices, weights)
        
        # 将超出范围的结果置为 0（这些位置由其他 GPU 负责）
        result[indices_mask] = 0
        return result
    else:
        # 无范围限制，直接查表
        return F.embedding(indices, weights)


@call_if_main(__name__)
def test_indexing():
    """
    测试基本 indexing 操作（不带 vocab_range）
    
    测试流程：
    1. 对不同的 batch size（1 到 32768）进行测试
    2. 验证自定义内核的输出是否与参考实现一致
    3. 对比性能（延迟和带宽）
    """
    # ========================================
    # 配置参数
    # ========================================
    EMBED_SIZE = 4096      # Embedding 维度（每个 token 的向量大小）
    NUM_TOKENS = 131072    # 词表大小（128K）
    
    # 创建 CUDA 流（用于异步执行）
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    
    # 创建随机权重矩阵（模拟 Embedding 表）
    weights = torch.randn((NUM_TOKENS, EMBED_SIZE), device="cuda", dtype=torch.float16)

    # ========================================
    # 对不同 batch size 进行测试
    # ========================================
    for bs in [2**n for n in range(0, 16)]:  # bs = 1, 2, 4, ..., 32768
        # 生成随机索引
        indices = torch.randint(0, NUM_TOKENS, (bs,), device="cuda", dtype=torch.int32)

        # ========================================
        # 正确性测试
        # ========================================
        result = indexing(
            weights,
            indices,
        )
        expected = ref_indexing(
            weights,
            indices,
        )
        assert torch.all(result == expected), f"Mismatch for BS={bs}"

        # ========================================
        # 性能测试
        # ========================================
        # 计算内存访问量：batch_size * embed_size * sizeof(float16)
        MEM = bs * EMBED_SIZE * weights.element_size()
        
        compare_memory_kernel_perf(
            our_impl=lambda: indexing(weights, indices),
            baseline=lambda: ref_indexing(weights, indices),
            memory_footprint=MEM,  # 用于计算带宽
            description=f"BS={bs:6d} | ",  # 输出描述
        )


@call_if_main(__name__)
def test_indexing_with_mask():
    """
    测试带 vocab_range 的 indexing 操作（用于 Tensor Parallelism）
    
    模拟在 TP=4 的场景下，每个 GPU 负责 1/4 的词表。
    测试流程与 test_indexing 类似，但增加了 vocab_range 参数。
    """
    # ========================================
    # 配置参数
    # ========================================
    EMBED_SIZE = 4096
    NUM_TOKENS = 131072  # 总词表大小
    TP = 4               # 模拟 4 个 GPU
    
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    weights = torch.randn((NUM_TOKENS, EMBED_SIZE), device="cuda", dtype=torch.float16)

    # ========================================
    # 配置 vocab_range（每个 GPU 负责 1/TP 的词表）
    # ========================================
    assert TP > 1
    MASK_LENGTH = NUM_TOKENS // TP  # 每个 Rank 的词表长度 = 32K
    MASK_RANGE = (MASK_LENGTH, MASK_LENGTH)  # (start=32K, length=32K)
    # 这里模拟 GPU 1 的场景（负责 32K-64K 的词表）

    # ========================================
    # 对不同 batch size 进行测试
    # ========================================
    for bs in [2**n for n in range(0, 16)]:
        # 生成随机索引（可能包含不在当前 Rank 范围内的索引）
        indices = torch.randint(0, NUM_TOKENS, (bs,), device="cuda", dtype=torch.int32)

        # ========================================
        # 正确性测试
        # ========================================
        result = indexing(
            weights,
            indices,
            vocab_range=MASK_RANGE,
        )
        expected = ref_indexing(
            weights,
            indices,
            vocab_range=MASK_RANGE,
        )
        assert torch.all(result == expected), f"Mismatch for BS={bs}"

        # ========================================
        # 性能测试
        # ========================================
        MEM = bs * EMBED_SIZE * weights.element_size()
        compare_memory_kernel_perf(
            our_impl=lambda: indexing(weights, indices),
            baseline=lambda: ref_indexing(weights, indices),
            memory_footprint=MEM,
            description=f"BS={bs:6d} | ",
            extra_kwargs={"init_stream": False},  # 不重新初始化 stream
        )
