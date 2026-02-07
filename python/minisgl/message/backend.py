
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseBackendMsg:
    """
    后端消息基类
    
    定义了发送给后端 (Engine/Scheduler) 的消息格式。
    实现了自定义的序列化/反序列化接口。
    """
    def encoder(self) -> Dict:
        """序列化消息为字典"""
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseBackendMsg:
        """从字典反序列化消息"""
        return deserialize_type(globals(), json)


@dataclass
class BatchBackendMsg(BaseBackendMsg):
    """
    批量后端消息
    
    用于将多个消息打包成一个批次发送，减少 IPC 开销。
    """
    data: List[BaseBackendMsg]


@dataclass
class ExitMsg(BaseBackendMsg):
    """
    退出消息
    
    通知后端进程优雅退出。
    """
    pass


@dataclass
class UserMsg(BaseBackendMsg):
    """
    用户请求消息
    
    包含单个用户请求的所有必要信息，用于 Engine 进行 prefill 或 decode。
    """
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor，包含 Prompt 的 Token IDs
    sampling_params: SamplingParams  # 采样参数
