"""
测试 Store Cache 内核（KV Cache 写入操作）

本测试用于验证自定义 store_cache 内核的正确性和性能，包括：
1. 将计算出的 K/V 张量写入到 KV Cache 的指定位置
2. 支持 Paged Attention 机制（根据 indices 进行分散写入）
3. 性能对比（自定义内核 vs PyTorch 索引赋值）

应用场景：
- **Attention 计算后**: 将新计算的 Key/Value 存储到 KV Cache
- **Paged Attention**: 根据 page table 映射，将数据写入到物理页
- **Continuous Batching**: 不同请求的 KV 可能分散在不同的 cache 位置

关键优化：
- 自定义 CUDA kernel 优化了分散写入的内存访问模式
- 避免 PyTorch 索引操作的开销
- 支持任意的索引顺序（indices 可以是乱序的）
"""

from __future__ import annotations

from minisgl.benchmark.perf import compare_memory_kernel_perf
import torch
from minisgl.kernel import store_cache
from minisgl.utils import call_if_main


@call_if_main(__name__)
def test_store_cache():
    """
    测试 KV Cache 存储内核
    
    测试流程：
    1. 创建大的 KV Cache 缓冲区（模拟 Paged Attention 的 cache pool）
    2. 对不同的 batch size，生成随机的 K/V 数据和写入位置（indices）
    3. 调用 store_cache，验证数据是否正确写入
    4. 对比性能（自定义内核 vs PyTorch 索引赋值 + torch.compile）
    """
    # ========================================
    # 配置参数
    # ========================================
    HEAD_SIZE = 128        # 每个 head 的维度
    NUM_TOKENS = 1048576   # Cache 总容量（1M tokens）
    
    # 创建 CUDA 流
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    
    # ========================================
    # 创建 KV Cache 缓冲区
    # ========================================
    # shape: [NUM_TOKENS, 2, HEAD_SIZE]
    # 维度说明：
    #   - NUM_TOKENS: cache 能存储的最大 token 数
    #   - 2: 分别存储 K 和 V
    #   - HEAD_SIZE: 每个 head 的维度
    kv_cache = torch.randn((NUM_TOKENS, 2, HEAD_SIZE), device="cuda", dtype=torch.float16)
    k_cache = kv_cache[:, 0, :]  # K Cache: [NUM_TOKENS, HEAD_SIZE]
    v_cache = kv_cache[:, 1, :]  # V Cache: [NUM_TOKENS, HEAD_SIZE]

    # ========================================
    # 对不同 batch size 进行测试
    # ========================================
    for bs in [2**n for n in range(0, 16)]:  # bs = 1, 2, 4, ..., 32768
        # ========================================
        # 生成测试数据
        # ========================================
        # 重要：使用 randperm 确保 indices 中没有重复
        # 如果有重复，多次写入同一位置会导致覆盖，使得测试不准确
        indices = torch.randperm(NUM_TOKENS, device="cuda")[:bs].to(torch.int32)
        
        # 生成随机的 QKV 数据（虽然这里只用 K 和 V）
        # shape: [bs, HEAD_SIZE * 4]
        # 通常 QKV 一起计算，这里模拟这个过程
        qkv = torch.randn((bs, HEAD_SIZE * 4), device="cuda", dtype=torch.float16)
        k = qkv[:, :HEAD_SIZE]                    # K: [bs, HEAD_SIZE]
        v = qkv[:, HEAD_SIZE : HEAD_SIZE * 2]     # V: [bs, HEAD_SIZE]
        
        # ========================================
        # 调用 store_cache 写入数据
        # ========================================
        store_cache(
            k_cache,
            v_cache,
            indices,  # 写入位置（可能是乱序的）
            k,
            v,
        )

        # ========================================
        # 正确性验证
        # ========================================
        # 验证 K Cache：indices 位置的数据应该等于 k
        assert torch.all(k_cache[indices] == k), bs
        # 验证 V Cache：indices 位置的数据应该等于 v
        assert torch.all(v_cache[indices] == v), bs

        # ========================================
        # 性能测试
        # ========================================
        # 计算内存访问量：bs * HEAD_SIZE * 2 (K + V) * sizeof(float16)
        MEM = bs * HEAD_SIZE * 2 * kv_cache.element_size()

        # 确保 K 和 V 是连续的（避免额外的内存拷贝开销）
        k = k.contiguous()
        v = v.contiguous()

        # ========================================
        # 基准实现：使用 PyTorch 索引赋值 + torch.compile
        # ========================================
        @torch.compile()
        def baseline():
            """
            使用 PyTorch 原生索引操作写入 cache
            
            torch.compile 会尝试融合操作并优化内存访问，
            但对于分散写入，通常仍不如专门优化的 CUDA kernel
            """
            k_cache[indices] = k
            v_cache[indices] = v

        # 对比性能
        compare_memory_kernel_perf(
            our_impl=lambda: store_cache(k_cache, v_cache, indices, k, v),
            baseline=baseline,
            memory_footprint=MEM,
            description=f"BS={bs:6d} | ",
            extra_kwargs={"init_stream": False},
        )
