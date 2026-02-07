
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from .utils import load_aot

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module


@lru_cache(maxsize=None)
def _load_test_tensor_module() -> Module:
    """加载测试用的 Tensor C++ 模块"""
    return load_aot("test_tensor", cpp_files=["tensor.cpp"])


def test_tensor(x: torch.Tensor, y: torch.Tensor) -> int:
    """
    测试 Tensor 传递
    
    用于验证 Python 到 C++ 的 tensor 传递是否正常。
    """
    return _load_test_tensor_module().test(x, y)
