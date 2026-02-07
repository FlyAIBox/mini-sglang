
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .utils import deserialize_type, serialize_type


@dataclass
class BaseFrontendMsg:
    """
    前端消息基类
    
    定义了发送给前端 (API Server) 的消息格式。
    """
    @staticmethod
    def encoder(msg: BaseFrontendMsg) -> Dict:
        """序列化消息"""
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseFrontendMsg:
        """反序列化消息"""
        return deserialize_type(globals(), json)


@dataclass
class BatchFrontendMsg(BaseFrontendMsg):
    """批量前端消息"""
    data: List[BaseFrontendMsg]


@dataclass
class UserReply(BaseFrontendMsg):
    """
    用户响应消息
    
    Backend/Tokenizer 处理完成后，发送给前端的响应。
    通常包含增量生成的文本。
    """
    uid: int
    incremental_output: str  # 增量输出文本 (本次新生成的字符)
    finished: bool           # 是否已结束生成
