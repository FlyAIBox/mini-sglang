from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple, Tuple

import torch


class BaseKVCache(ABC):
    """
    KV Cache 基类
    
    定义了 KV Cache 的通用接口。
    """

    @abstractmethod
    def k_cache(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def v_cache(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...

    @property
    @abstractmethod
    def num_layers(self) -> int: ...


class KVCacheLayout(enum.Enum):
    LayerFirst = enum.auto()
    PageFirst = enum.auto()


@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    cached_len: int


class SizeInfo(NamedTuple):
    evictable_size: int
    protected_size: int

    @property
    def total_size(self) -> int:
        return self.evictable_size + self.protected_size


class BaseCacheManager(ABC):
    @abstractmethod
    def match_prefix(self, input_ids: torch.Tensor) -> Tuple[BaseCacheHandle, torch.Tensor]:
        """
        前缀匹配
        
        查找 cache 中与 input_ids 匹配的最长前缀。
        此操作不会修改 cache。
        返回的 indices 仅在 handle 被锁定时安全使用。
        
        Args:
            input_ids (torch.Tensor): 输入的 token ids. Shape: (seq_len,)
        Returns:
            handle (BaseCacheHandle): 匹配到的前缀的句柄。
            indices (torch.Tensor): cached 中最长匹配前缀的索引。
        """

    @abstractmethod
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """
        锁定或解锁缓存句柄
        
        此操作不会修改 cache 内容，只改变 size info (引用计数)。
        当句柄被锁定时，它不能被驱逐。
        在使用 match_prefix 返回的 tensor 之前，必须先锁定句柄。
        否则可能会被 evict 操作驱逐。
        
        Args:
            handle (BaseCacheHandle): 要锁定/解锁的句柄。
            unlock (bool): 是否解锁。默认为 False。
        """

    @abstractmethod
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> int:
        """
        插入新前缀到 cache
        
        此操作会修改 cache。
        
        Args:
            input_ids (torch.Tensor): 要插入的 token ids. Shape: (seq_len,)
            indices (torch.Tensor): 存储新前缀的索引。Shape: (seq_len,)

        Returns:
            int: 已经在 cache 中的前缀长度。这部分不需要插入，调用者应该释放对应的 indices。
        """

    @abstractmethod
    def evict(self, size: int) -> torch.Tensor:
        """
        驱逐部分前缀以释放空间
        
        此操作会修改 cache。
        特定的 evict 0 总是安全的且不做任何事。
        注意实际驱逐的大小可能大于请求的大小。
        
        Args:
            size (int): 需要驱逐的大小。

        Returns:
            torch.Tensor: 被驱逐的索引。Shape: (evict_size,)
        Raises:
            RuntimeError: 如果请求的大小大于可驱逐的大小。
        """

    @abstractmethod
    def reset(self) -> None:
        """重置 cache manager 和底层的 cache"""

    @property
    @abstractmethod
    def size_info(self) -> SizeInfo:
        """获取 cache 的大小信息"""

    @abstractmethod
    def check_integrity(self) -> None:
        """检查 cache 的完整性。如果损坏则抛出异常。"""

