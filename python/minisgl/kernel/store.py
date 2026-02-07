
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module

DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)


@lru_cache(maxsize=None)
def _jit_store_module(
    element_size: int,
    *,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    """
    JIT 编译加载 Store Kernel
    
    Store Kernel 用于将计算结果 (k/v tensor) 写入到 KV Cache 中。
    使用 Paged Attention 机制，根据 indices (page mapping) 进行分散写入。
    """
    args = make_cpp_args(element_size, *config)
    return load_jit(
        "store",
        *args,
        cuda_files=["store.cu"],
        cuda_wrappers=[("launch", f"StoreKernel<{args}>::run")],
    )


def store_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """
    将当前 batch 的 K/V 写入到 Cache 中
    
    Args:
        k_cache: Key Cache 张量 [num_blocks, block_size, num_heads, head_dim]
        v_cache: Value Cache 张量 [num_blocks, block_size, num_heads, head_dim]
        indices: 写入位置索引 [num_tokens]，指向 cache 中的绝对位置 (flattened)
        k: 当前 batch 的 Key [num_tokens, num_heads, head_dim]
        v: 当前 batch 的 Value [num_tokens, num_heads, head_dim]
    """
    num_tokens = k_cache.shape[0]
    # Flatten cache for easier indexing in kernel
    k_cache = k_cache.view(num_tokens, -1)
    v_cache = v_cache.view(num_tokens, -1)
    
    element_size = k_cache.shape[1] * k_cache.element_size()
    module = _jit_store_module(element_size)
    module.launch(k_cache, v_cache, indices, k, v)
