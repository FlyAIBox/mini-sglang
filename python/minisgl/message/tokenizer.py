
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseTokenizerMsg:
    """
    Tokenizer 消息基类
    
    定义了发送给 Tokenizer 进程的消息格式。
    """
    @staticmethod
    def encoder(msg: BaseTokenizerMsg) -> Dict:
        """序列化消息"""
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseTokenizerMsg:
        """反序列化消息"""
        return deserialize_type(globals(), json)


@dataclass
class BatchTokenizerMsg(BaseTokenizerMsg):
    """批量 Tokenizer 消息"""
    data: List[BaseTokenizerMsg]


@dataclass
class DetokenizeMsg(BaseTokenizerMsg):
    """
    Detokenize 请求消息 (Decode)
    
    Backend 生成新的 token ID 后，发送此消息给 Tokenizer 进行解码。
    """
    uid: int
    next_token: int  # 新生成的 token ID
    finished: bool   # 生成是否已结束


@dataclass
class TokenizeMsg(BaseTokenizerMsg):
    """
    Tokenize 请求消息 (Prefill)
    
    Frontend 收到用户请求后，发送此消息给 Tokenizer 将文本转为 token IDs。
    """
    uid: int
    text: str | List[Dict[str, str]] # 用户输入的 Prompt (文本或聊天记录)
    sampling_params: SamplingParams  # 采样参数


@dataclass
class AbortMsg(BaseTokenizerMsg):
    """中止请求消息"""
    uid: int
