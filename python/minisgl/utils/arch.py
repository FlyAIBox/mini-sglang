
from __future__ import annotations

from functools import lru_cache
from typing import Tuple


@lru_cache(maxsize=None)
def _get_torch_cuda_version() -> Tuple[int, int] | None:
    """
    获取当前 PyTorch 使用的 CUDA 设备 Compute Capability
    
    返回:
        Tuple[int, int] | None: (major, minor) 版本号，如果未检测到 CUDA 则返回 None
    """
    import torch
    import torch.version

    if not torch.cuda.is_available() or not torch.version.cuda:
        return None
    # 返回主设备 (device 0) 的算力版本
    return torch.cuda.get_device_capability()


def is_arch_supported(major: int, minor: int = 0) -> bool:
    """检查当前设备是否支持指定的最低架构版本"""
    arch = _get_torch_cuda_version()
    if arch is None:
        return False
    return arch >= (major, minor)


def is_sm90_supported() -> bool:
    """检查是否支持 SM90 (Hopper 架构, e.g. H100)"""
    return is_arch_supported(9, 0)


def is_sm100_supported() -> bool:
    """检查是否支持 SM100 (Blackwell 架构)"""
    return is_arch_supported(10, 0)
