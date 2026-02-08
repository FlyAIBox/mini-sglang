"""
测试消息序列化/反序列化模块

本测试用于验证 Mini-SGLang 的消息序列化机制，确保：
1. 自定义数据类（dataclass）可以正确序列化为字节流
2. PyTorch Tensor 可以在序列化过程中正确处理
3. 嵌套数据结构（如 List[A]）可以正确序列化
4. 反序列化后的对象与原对象等价

序列化机制用于：
- ZMQ 消息队列的进程间通信
- 请求和响应的网络传输
- 状态的持久化存储

测试覆盖：
1. 基本数据类型（int, str）
2. 复杂嵌套结构（List, Dict）
3. PyTorch Tensor（需要特殊处理）
4. 实际业务消息类型（UserMsg, BatchBackendMsg）
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List

from minisgl.core import SamplingParams
import torch
from minisgl.message import BatchBackendMsg, UserMsg
from minisgl.message.utils import serialize_type, deserialize_type
from minisgl.utils import call_if_main, init_logger

logger = init_logger(__name__)


@dataclass
class A:
    """
    测试用的数据类
    
    包含多种类型的字段，用于验证序列化机制的完整性：
    - 基本类型（int, str）
    - 递归嵌套（List[A]）
    - PyTorch Tensor（需要特殊处理）
    """
    x: int                  # 整数字段
    y: str                  # 字符串字段
    z: List[A]             # 递归嵌套（列表中包含同类型对象）
    w: torch.Tensor        # PyTorch Tensor（需要序列化为字节）


@call_if_main()
def test_serialize_deserialize():
    """
    测试序列化和反序列化的往返一致性
    
    流程：
    1. 创建包含复杂嵌套结构的对象
    2. 序列化为字节流（或 JSON 兼容的字典）
    3. 反序列化回 Python 对象
    4. 验证内容一致性
    """
    # ========================================
    # 测试 1: 自定义数据类的序列化
    # ========================================
    t = torch.tensor([1, 2, 3], dtype=torch.int32)
    # 创建嵌套结构：x 包含 y，y 包含一个 tensor
    x = A(10, "hello", [A(20, "world", [], t)], t)
    
    # 序列化：将对象转换为可传输的格式
    # 返回的 data 是一个字典，包含类型信息和字段值
    data = serialize_type(x)
    logger.info(data)
    
    # 反序列化：从字典恢复对象
    # 需要提供类型映射 {"类名": 类对象}，以便正确还原类型
    y = deserialize_type({"A": A}, data)
    logger.info(y)
    
    # ========================================
    # 测试 2: 实际业务消息的序列化
    # ========================================
    # UserMsg 和 BatchBackendMsg 是系统中实际使用的消息类型
    # 它们内部包含 SamplingParams 和 Tensor 等复杂字段
    u = BatchBackendMsg([UserMsg(uid=0, input_ids=t, sampling_params=SamplingParams())])
    
    # 使用消息类自带的 encoder/decoder 进行序列化
    # encoder: 对象 -> 字节流
    # decoder: 字节流 -> 对象
    result = u.decoder(u.encoder())
    
    logger.info(u)
    logger.info(result)
    
    # ========================================
    # 验证点：
    # ========================================
    # 1. y 应该与 x 结构相同（内容相等）
    # 2. result 应该与 u 内容一致
    # 3. Tensor 应该保留数据类型和值
    # 4. 嵌套结构应该完整保留
