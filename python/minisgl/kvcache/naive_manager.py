from __future__ import annotations

from typing import Tuple

import torch

from .base import BaseCacheHandle, BaseCacheManager, SizeInfo


class NaiveCacheHandle(BaseCacheHandle):
    pass


class NaiveCacheManager(BaseCacheManager):
    """
    朴素 KV Cache 管理器
    
    一个最简单的 CacheManager 实现，不进行任何缓存复用。
    对于每个请求，match_prefix 总是返回 0 长度的匹配。
    适用于不需要或不希望使用 Radix Cache 的场景（例如调试或极其简单的用例）。
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        super().__init__()

    def match_prefix(self, input_ids: torch.Tensor) -> Tuple[NaiveCacheHandle, torch.Tensor]:
        """由于不进行缓存复用，总是返回 0 匹配"""
        _ = input_ids  # unused
        return NaiveCacheHandle(0), self.empty_tensor

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        pass

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> int:
        """总是提示全部插入（因为没有任何缓存被复用）"""
        assert len(indices) == len(input_ids)
        return len(indices)

    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor
        raise NotImplementedError("NaiveCacheManager does not support eviction.")

    def reset(self) -> None:
        pass

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(evictable_size=0, protected_size=0)

    def check_integrity(self) -> None:
        pass

