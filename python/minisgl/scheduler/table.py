
import torch


class TableManager:
    """
    请求表管理器
    
    维护一个预分配的 Tensor (page_table) 用于存储页映射关系。
    管理 "slot" (槽位)，每个请求占用一个 slot，用于在 GPU 上索引其页表。
    类似于操心系统中的进程表管理。
    """
    def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
        self._max_running_reqs = max_running_reqs
        # 初始化空闲槽位列表 (LIFO)
        self._free_slots = list(range(max_running_reqs))
        # 全局页表 Tensor (在 GPU 上), shape usually [max_running_reqs, max_context_len]
        self.page_table = page_table
        # NOTE: dummy request also use this pool to get the input ids, so we need to
        # make sure the token pool is initialized with valid values (token_id = 0).
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)

    @property
    def available_size(self) -> int:
        """当前可用槽位数量"""
        return len(self._free_slots)

    def allocate(self) -> int:
        """分配一个空闲槽位"""
        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        """释放槽位"""
        self._free_slots.append(slot)
