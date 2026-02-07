
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.kvcache import BaseCacheHandle, create_cache_manager

if TYPE_CHECKING:
    from .utils import PendingReq


class CacheManager:
    """
    KV Cache 管理器
    
    统一管理 CPU/GPU 上的 KV Cache 资源。
    主要职责：
    1. 跟踪空闲的物理页 (free slots)
    2. 管理 Radix Cache (前缀匹配、驱逐、分配)
    3. 为请求分配和释放 Cache
    """
    def __init__(self, device: torch.device, num_pages: int, type: str):
        # TODO: support page_size > 1
        # 初始化空闲页列表 (栈)，存储可用的物理页索引
        self._free_slots = torch.arange(num_pages, dtype=torch.int32, device=device)
        self.device = device
        # 创建底层的 Cache Manager (e.g., RadixCacheManager)
        self.manager = create_cache_manager(device=device, type=type)
        self.num_pages = num_pages

    def _free(self, indices: torch.Tensor) -> None:
        """归还释放的物理页索引"""
        if len(indices) > 0:
            self._free_slots = torch.cat([self._free_slots, indices])

    def match_req(self, req: PendingReq):
        """
        为新请求匹配已存在的前缀 Cache
        
        Args:
            req: 待处理的请求
            
        Returns:
            match_result: 匹配到的前缀信息 (indices, handle 等)
        """
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        # 在缓存中查找匹配的前缀 (不包括最后一个token，因为需要重新计算)
        return self.manager.match_prefix(req.input_ids[: input_len - 1])

    @property
    def available_size(self) -> int:
        """
        当前可用物理页数量
        
        包括：
        1. 可驱逐的页 (evictable_size)
        2. 当前空闲的页 (free_slots)
        """
        return self.manager.size_info.evictable_size + len(self._free_slots)

    def lock(self, handle: BaseCacheHandle) -> None:
        """锁定 Cache 句柄，防止被驱逐"""
        self.manager.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        """解锁 Cache 句柄，允许被驱逐"""
        self.manager.lock_handle(handle, unlock=True)

    def allocate(self, needed_len: int) -> torch.Tensor:
        """
        分配物理页
        
        Args:
            needed_len: 需要的页数/token数的 (目前 page_size=1)
            
        Returns:
            allocated: 分配到的物理页索引
        """
        # 如果空闲页足够，直接分配
        if needed_len <= (free_len := len(self._free_slots)):
            allocated = self._free_slots[:needed_len]
            self._free_slots = self._free_slots[needed_len:]
            return allocated

        # 如果空闲页不足，需要驱逐旧的 Cache
        # NOTE: len(evicted) + free_len >= needed_len
        evicted = self.manager.evict(needed_len - free_len)
        merged = torch.cat([self._free_slots, evicted])
        assert len(merged) >= needed_len, "Eviction did not free enough space."

        allocated = merged[:needed_len]
        self._free_slots = merged[needed_len:]
        return allocated

    def free_and_cache_finished_req(
        self,
        old_handle: BaseCacheHandle,
        input_ids: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        """
        请求完成后，将完整序列插入 Radix Cache 并释放多余资源
        
        Args:
            old_handle: 请求之前的 Cache 句柄
            input_ids: 完整的 token 序列
            indices: 对应的物理页索引
        """
        # 将完整序列插入 Radix Cache（可能会创建新节点或更新引用）
        in_cache_len = self.manager.insert_prefix(input_ids, indices)
        # 释放未被 Radix Cache 采纳的页 (例如重复部分)
        self._free(indices[old_handle.cached_len : in_cache_len])
        # 解锁旧句柄 (引用计数 -1)
        self.unlock(old_handle)

    def check_integrity(self) -> None:
        """检查内存一致性"""
        self.manager.check_integrity()
        if len(self._free_slots) + self.manager.size_info.total_size != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_slots({len(self._free_slots)}) +"
                f" total_size({self.manager.size_info.total_size}) != num_pages({self.num_pages})"
            )
