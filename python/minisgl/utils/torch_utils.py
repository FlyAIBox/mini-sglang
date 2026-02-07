
from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@contextmanager
def torch_dtype(dtype: torch.dtype):
    """
    上下文管理器：临时切换 PyTorch 默认数据类型
    
    用于在特定代码块中使用不同的精度（例如加载权重时）。
    """
    import torch  # real import when used

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def nvtx_annotate(name: str, layer_id_field: str | None = None):
    """
    装饰器：添加 NVTX 标记
    
    用于 NVIDIA Nsight Systems 性能分析。
    可以在时间轴上显示带有名称的区间。
    
    Args:
        name: 标记名称
        layer_id_field: 可选的层ID字段名，用于动态生成标记名称 (e.g. "layer_{layer_id}")
    """
    import torch.cuda.nvtx as nvtx

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            display_name = name
            if layer_id_field and hasattr(self, layer_id_field):
                display_name = name.format(getattr(self, layer_id_field))
            with nvtx.range(display_name):
                return fn(self, *args, **kwargs)

        return wrapper

    return decorator
