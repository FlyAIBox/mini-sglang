
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from .utils import load_aot

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module


@lru_cache(maxsize=None)
def _load_radix_module() -> Module:
    """加载 Radix C++ 模块"""
    return load_aot("radix", cpp_files=["radix.cpp"])


def fast_compare_key(x: torch.Tensor, y: torch.Tensor) -> int:
    """
    快速比较两个 tensor 的前缀匹配长度
    
    使用 C++ 实现，避免 Python 循环开销。
    主要用于 Radix Attention 中查找最长公共前缀。
    
    Args:
        x: 1D int32 cpu tensor
        y: 1D int32 cpu tensor
        
    Returns:
        int: 最长公共前缀的长度
    """
    # compare 2 1-D int cpu tensors for equality
    return _load_radix_module().fast_compare_key(x, y)
