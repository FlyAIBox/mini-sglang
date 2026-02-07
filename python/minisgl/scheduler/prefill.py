
"""
预填充 (Prefill) 阶段调度器

主要负责将处于 Prefill 阶段的新请求加入到 Batch 中。
支持 Chunked Prefill (分块预填充)，即当一个请求的 Prompt 很长时，
可以将其拆分为多个 Chunk 分多次 Prefill，从而避免单个长请求由于计算量过大阻塞推理系统。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Batch, Req
from minisgl.utils import init_logger

from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl.kvcache import BaseCacheHandle
    from minisgl.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

logger = init_logger(__name__)


class ChunkedReq(Req):
    """
    分块请求 (Chunked Request)
    
    表示一个还未完成 Prefill 的请求。
    它的 input_ids 只包含当前 Chunk 的 tokens。
    这种请求不处于 Decode 阶段，因此不能 append 新 token。
    """
    def append_host(self, next_token: torch.Tensor) -> None:
        raise NotImplementedError("ChunkedReq should be sampled")

    def can_decode(self) -> bool:
        """分块请求还没准备好进入Decode阶段"""
        return False


@dataclass
class PrefillAdder:
    """
    Prefill添加器
    
    辅助类，用于尝试将 pending 队列中的请求加入到当前 Batch 中。
    它会检查显存（KV Cache）容量和 Token 预算（max_extend_tokens）。
    """
    token_budget: int            # 当前Batch剩余可用的Token预算
    reserved_size: int           # 为当前Decode请求预留的空间
    cache_manager: CacheManager  # 显存管理器
    table_manager: TableManager  # 页表管理器

    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        """
        尝试为一个新请求分配资源
        
        资源包括:
        1. 逻辑上的 Table Entry (table_idx)
        2. 物理上的 KV Cache (cache_handle)
        
        Args:
            req: 待处理的挂起请求
            
        Returns:
            Tuple[BaseCacheHandle, int]: (分配的Cache句柄, 分配的Table索引)，如果分配失败则返回None
        """
        if self.table_manager.available_size == 0:
            return None

        # 尝试在Radix Tree中寻找前缀匹配的Cache
        handle, match_indices = self.cache_manager.match_req(req)
        cached_len = handle.cached_len
        
        # 估算所需总显存: 新增部分 + 输出部分
        # TODO: 这里对输出长度的预估可能偏保守，直接使用了 max_new_tokens
        extend_len = req.input_len - cached_len
        estimated_len = extend_len + req.output_len

        # 检查显存是否足够 (需要加上reserved_size，即Decode请求所需的空间)
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None
            
        # 锁定 Cache Handle，防止被回收
        self.cache_manager.lock(handle)
        
        # Double Check: 锁定后可能有其他变化
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return self.cache_manager.unlock(handle)

        # 分配 Table Entry
        table_idx = self.table_manager.allocate()
        
        # 如果有缓存命中，需要将缓存及其元数据通过页表关联起来
        if cached_len > 0:  # NOTE: set the cached part
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            page_entry = self.table_manager.page_table[table_idx][:cached_len]
            # 拷贝 Input IDs (Token IDs) 到设备
            device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
            # 拷贝 Page Mapping (页表项)
            page_entry.copy_(match_indices)

        return handle, table_idx

    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
    ) -> Req:
        """
        将一个请求添加到Batch中
        
        会计算本次能处理的 Chunk Size。
        """
        remain_len = pending_req.input_len - cached_len
        # 计算本次Prefill的长度：受限于剩余Token Budget
        chunk_size = min(self.token_budget, remain_len)
        
        # 判断是否需要分块 (Chunked Prefill)
        is_chunked = chunk_size < remain_len
        CLS = ChunkedReq if is_chunked else Req
        
        # 更新预算
        self.token_budget -= chunk_size
        self.reserved_size += remain_len + pending_req.output_len
        
        # 将本次处理的 Input tokens 拷贝到 Token Pool
        # NOTE: 这里只拷贝 Token ID，KV Cache 的物理页分配在 Scheduler._prepare_batch 中进行
        _slice = slice(cached_len, cached_len + chunk_size)
        device_ids = self.table_manager.token_pool[table_idx][_slice]
        device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)
        
        return CLS(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
        )

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        """
        对外接口：尝试添加一个 Pending 请求
        
        支持处理普通请求和之前的 Chunked 请求续传。
        """
        if self.token_budget <= 0:
            return None

        # 如果这个请求已经是 Chunked 状态 (之前处理了一部分)，则继续处理下一块
        if chunked_req := pending_req.chunked_req:
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
            )

        # 全新请求：尝试分配资源
        if resource := self._try_allocate_one(pending_req):
            cache_handle, table_idx = resource
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
            )

        return None


@dataclass
class PrefillManager:
    """
    Prefill阶段管理器
    
    维护等待队列 (Pending List)，并在每一轮调度时，
    尽可能多地从队列中取出请求放入 Batch 进行 Prefill。
    """
    cache_manager: CacheManager
    table_manager: TableManager
    decode_manager: DecodeManager
    pending_list: List[PendingReq] = field(default_factory=list)

    def add_one_req(self, req: UserMsg) -> None:
        """接收新用户请求"""
        self.pending_list.append(PendingReq(req.uid, req.input_ids, req.sampling_params))

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        """
        调度下一个 Prefill Batch
        
        Args:
            prefill_budget: 本次调度允许的最大Token数量 (max_extend_tokens)
            
        Returns:
            Batch | None: 构建好的 Prefill Batch
        """
        if len(self.pending_list) == 0:
            return None

        # 创建Adder，注入当前资源状态和约束
        # reserved_size: 需要考虑已经在Decode队列中的请求，它们会继续生成Token占用更多显存
        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )
        
        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []
        
        # 遍历等待队列，尝试逐个添加
        for pending_req in self.pending_list:
            if req := adder.try_add_one(pending_req):
                # 添加成功
                pending_req.chunked_req = None
                
                # 如果是 Chunked 请求 (只处理了一部分)，需要放回 chunked_list 等待下一轮继续
                if isinstance(req, ChunkedReq):
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                
                reqs.append(req)
            else:
                # 添加失败（通常是显存或Budget不足），停止添加
                break  # We cannot add more requests
                
        if len(reqs) == 0:
            return None
            
        # 更新等待队列：未完成的Chunked请求 + 未处理的剩余请求
        self.pending_list = chunked_list + self.pending_list[len(reqs) :]
        
        return Batch(reqs=reqs, phase="prefill")

    @property
    def runnable(self) -> bool:
        """是否有等待处理的请求"""
        return len(self.pending_list) > 0
