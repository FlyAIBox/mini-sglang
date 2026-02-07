
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Tuple

from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module

DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)


@lru_cache(maxsize=None)
def _jit_index_module(
    element_size: int,
    *,
    num_splits: int = 1,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    """
    JIT 编译加载 Index Kernel
    
    Index Kernel 用于从权重矩阵中 gather 指定 indices 的行，通常用于 Embedding 层或输出层。
    
    Args:
        element_size: 每个元素的字节大小 (例如 float16 为 2)
        num_splits: 将 embedding 维度切分为多少份进行并行复制
        config: Kernel 配置 (线程数等)
    """
    args = make_cpp_args(element_size, num_splits, *config)
    return load_jit(
        "index",
        *args,
        cuda_files=["index.cu"],
        cuda_wrappers=[("launch", f"IndexKernel<{args}>::run")],
    )


def indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    vocab_range: Tuple[int, int] | None = None,  # (start, length)
) -> torch.Tensor:
    """
    执行 Indexing 操作 (类似 torch.embedding 或 torch.gather)
    
    Args:
        weights: 权重矩阵 [vocab_size, hidden_size]
        indices: 索引张量 [batch_size]
        output: 输出张量 (可选)
        vocab_range: 如果进行了 Vocabulary Parallelism，当前 Rank 负责的词表范围 (start, length)
    """
    if output is None:
        output = weights.new_empty(indices.shape[0], weights.shape[1])

    element_size = weights.shape[1] * weights.element_size()
    
    # 简单的启发式规则：根据 hidden_size 的字节大小决定并行度
    if element_size % 2048 == 0:
        num_splits = 4
    elif element_size % 1024 == 0:
        num_splits = 2
    else:
        num_splits = 1
        
    module = _jit_index_module(element_size, num_splits=num_splits)
    module.launch(weights, indices, output, vocab_range)
    return output
