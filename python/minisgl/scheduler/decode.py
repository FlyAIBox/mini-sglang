from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from minisgl.core import Batch, Req


@dataclass
class DecodeManager:
    """
    解码阶段管理器 (Decode Manager)
    
    负责管理处于解码阶段（生成后续token）的请求。
    主要职责包括：
    1. 维护当前正在运行的请求列表 (running_reqs)
    2. 过滤已完成的请求
    3. 移除特定请求
    4. 计算负载 (inflight_tokens)
    5. 调度下一个批次 (schedule_next_batch)
    """
    running_reqs: Set[Req] = field(default_factory=set)

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        """
        过滤并添加请求
        
        将新请求加入到运行队列中，并过滤掉已经完成生成（不能再解码）的请求。
        通常在Prefill阶段完成后，将Prefill的请求转移到这里。
        
        Args:
            reqs: 新的一批请求
        """
        # 将现有请求和新请求合并，保留那些还可以继续解码的请求
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode()}

    def remove_req(self, req: Req) -> None:
        """
        移除指定请求
        
        当请求被中止或出现错误时调用。
        
        Args:
            req: 需要移除的请求对象
        """
        self.running_reqs.discard(req)

    @property
    def inflight_tokens(self) -> int:
        """
        获取当前负载（In-flight tokens）
        
        计算所有正在运行的请求中剩余需要生成的token总长度。
        这可以用来衡量当前的计算负载。
        """
        return sum(req.remain_len for req in self.running_reqs)

    def schedule_next_batch(self) -> Batch | None:
        """
        调度下一个批次
        
        如果当前有可运行的请求，则将它们打包成一个Batch对象返回。
        这个Batch将被送往模型进行推理。
        
        Returns:
            Batch对象，包含当前所有正在解码的请求，标记阶段为 "decode"；
            如果没有请求则返回None。
        """
        if not self.runnable:
            return None
        return Batch(reqs=list(self.running_reqs), phase="decode")

    @property
    def runnable(self) -> bool:
        """
        检查是否有请求可运行
        
        只要有正在运行的请求，就认为是可以调度的。
        """
        return len(self.running_reqs) > 0
