
from __future__ import annotations

from typing import Any, Dict, Type

import numpy as np
import torch


def _serialize_any(value: Any) -> Any:
    """
    递归序列化任意对象
    
    支持字典、列表、元组以及基本类型。
    对于复杂对象，调用 serialize_type 进行处理。
    """
    if isinstance(value, dict):
        return {k: _serialize_any(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return type(value)(_serialize_any(v) for v in value)
    elif isinstance(value, (int, float, str, type(None), bool, bytes)):
        return value
    else:
        return serialize_type(value)


def serialize_type(self) -> Dict:
    """
    序列化自定义类型对象
    
    将对象转换为字典，包含 "__type__" 字段标识类名。
    特别处理 PyTorch Tensor (目前仅支持 1D Tensor)。
    """
    # find all member variables
    serialized = {}

    if isinstance(self, torch.Tensor):
        assert self.dim() == 1, "we can only serialize 1D tensor for now"
        serialized["__type__"] = "Tensor"
        # 转换为 numpy bytes 存储
        serialized["buffer"] = self.cpu().numpy().tobytes()
        serialized["dtype"] = str(self.dtype)
        return serialized

    # normal type
    serialized["__type__"] = self.__class__.__name__
    for k, v in self.__dict__.items():
        serialized[k] = _serialize_any(v)
    return serialized


def _deserialize_any(cls_map: Dict[str, Type], data: Any) -> Any:
    """
    递归反序列化任意对象
    
    根据 cls_map 查找类定义，重建对象。
    """
    if isinstance(data, dict):
        if "__type__" in data:
            return deserialize_type(cls_map, data)
        else:
            return {k: _deserialize_any(cls_map, v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(_deserialize_any(cls_map, d) for d in data)
    elif isinstance(data, (int, float, str, type(None), bool, bytes)):
        return data
    else:
        raise ValueError(f"Cannot deserialize type {type(data)}")


def deserialize_type(cls_map: Dict[str, Type], data: Dict) -> Any:
    """
    反序列化自定义类型对象
    
    根据 "__type__" 字段重建对象。
    """
    type_name = data["__type__"]
    # we can only serialize 1D tensor for now
    if type_name == "Tensor":
        buffer = data["buffer"]
        dtype_str = data["dtype"].replace("torch.", "")
        # 使用 numpy 还原数据
        np_dtype = getattr(np, dtype_str)
        assert isinstance(buffer, bytes)
        np_tensor = np.frombuffer(buffer, dtype=np_dtype)
        # 转回 torch tensor
        return torch.from_numpy(np_tensor.copy())

    # 重建自定义类实例
    cls = cls_map[type_name]
    kwargs = {}
    for k, v in data.items():
        if k == "__type__":
            continue
        kwargs[k] = _deserialize_any(cls_map, v)
    return cls(**kwargs)
