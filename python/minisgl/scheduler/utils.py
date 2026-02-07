
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from minisgl.core import SamplingParams
    from .prefill import ChunkedReq


@dataclass
class PendingReq:
    """
    等待调度的请求
    
    存储请求的基本信息，尚未真正运行。
    
    属性:
        uid: 请求唯一ID
        input_ids: 输入 token ID 序列
        sampling_params: 采样参数
        chunked_req: 如果使用 Chunked Prefill，这里存储切分后的请求状态
    """
    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    chunked_req: ChunkedReq | None = None

    @property
    def input_len(self) -> int:
        """输入序列长度"""
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        """最大输出长度"""
        return self.sampling_params.max_tokens


@dataclass
class ScheduleResult:
    """
    调度结果
    
    包含本次调度选中的请求列表以及相关的输出索引。
    
    属性:
        reqs: 被调度的请求 (PendingReq) 列表
        output_indices: 每个请求在输出 Tensor 中的对应的索引 (用于 copy 结果)
    """
    reqs: List[PendingReq]
    output_indices: List[torch.Tensor]
