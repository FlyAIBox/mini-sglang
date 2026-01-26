"""
Mini-SGLang 层基类模块

定义神经网络层的基础类和工具，提供统一的权重管理接口。
所有自定义层（Linear、Attention、Embedding等）都继承自这些基类。

核心概念：
- BaseOP: 所有操作（层）的基类，提供state_dict/load_state_dict
- StateLessOP: 无状态操作（不含可训练参数）的基类
- OPList: 操作列表，用于实现nn.ModuleList的功能

设计目标：
1. 轻量级：不依赖torch.nn.Module，减少开销
2. 类型安全：完整的类型注解
3. 灵活性：支持自定义state_dict行为
4. 兼容性：与PyTorch的state_dict格式兼容
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Generic, List, TypeAlias, TypeVar

import torch

_STATE_DICT: TypeAlias = Dict[str, torch.Tensor]
"""state_dict类型别名：字符串键到Tensor值的字典"""


def _concat_prefix(prefix: str, name: str) -> str:
    """
    拼接权重名称前缀
    
    用于构建嵌套模块的权重名称，例如"model.layers.0.attn.weight"
    
    参数:
        prefix: 父模块的前缀（可以为空）
        name: 当前参数或子模块的名称
    
    返回:
        完整的权重名称
    
    示例:
        _concat_prefix("model.layers", "attn")     → "model.layers.attn"
        _concat_prefix("", "weight")               → "weight"
        _concat_prefix("encoder", "embedding")     → "encoder.embedding"
    """
    return f"{prefix}.{name}" if prefix else name


class BaseOP:
    """
    操作基类（类似torch.nn.Module）
    
    所有神经网络层和操作的基类。提供统一的接口来管理参数。
    
    核心功能：
    1. forward(): 前向传播（子类必须实现）
    2. state_dict(): 收集所有参数到字典
    3. load_state_dict(): 从字典加载参数
    
    设计理念：
        - 简单：不包含torch.nn.Module的复杂特性（hooks、buffers等）
        - 高效：直接操作__dict__，避免额外开销
        - 递归：自动处理嵌套的BaseOP子对象
        - 灵活：子类可以自定义state_dict行为
    
    使用示例：
        class MyLayer(BaseOP):
            def __init__(self):
                self.weight = torch.randn(10, 10)
                self.sublayer = AnotherLayer()
            
            def forward(self, x):
                return x @ self.weight
        
        layer = MyLayer()
        # 保存权重
        state = layer.state_dict()
        # 加载权重  
        layer.load_state_dict(state)
    """
    
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        前向传播（抽象方法，子类必须实现）
        
        参数和返回值由子类定义，取决于具体的层类型。
        """
        ...

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        """
        收集所有参数到字典
        
        递归遍历对象的所有属性，将Tensor参数和子模块的参数收集到字典中。
        
        参数:
            prefix: 参数名称前缀（内部使用，递归时传递）
            result: 结果字典（内部使用，递归时共享）
        
        返回:
            包含所有参数的字典，键为参数的完整路径，值为Tensor
        
        工作原理：
            1. 遍历__dict__中的所有属性
            2. 跳过私有属性（以_开头）
            3. 对于Tensor：直接添加到result
            4. 对于BaseOP子对象：递归调用其state_dict
        
        示例:
            假设有如下结构：
            model = Model()
            model.weight = torch.randn(10, 10)
            model.encoder = Encoder()
            model.encoder.weight = torch.randn(5, 5)
            
            state = model.state_dict()
            # 结果：{
            #   "weight": Tensor(...),
            #   "encoder.weight": Tensor(...)
            # }
        """
        result = result if result is not None else {}

        for name, param in self.__dict__.items():
            # 跳过私有属性（如_cache、_temp等）
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                # Tensor参数：直接添加
                result[_concat_prefix(prefix, name)] = param
            elif isinstance(param, BaseOP):
                # 子模块：递归收集参数
                param.state_dict(prefix=_concat_prefix(prefix, name), result=result)

        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        """
        从字典加载参数
        
        递归遍历对象的所有属性，从state_dict中找到对应的值并加载。
        
        参数:
            state_dict: 参数字典（会被修改，成功加载的键会被移除）
            prefix: 参数名称前缀（内部使用）
            _internal: 是否为内部递归调用（内部使用）
        
        异常:
            RuntimeError: 如果state_dict中有未使用的键（说明参数不匹配）
            AssertionError: 如果参数shape或dtype不匹配
        
        工作原理：
            1. 遍历__dict__中的所有属性
            2. 跳过私有属性
            3. 对于Tensor：从state_dict中pop出对应值，检查shape和dtype，然后替换
            4. 对于BaseOP子对象：递归调用其load_state_dict
            5. 最后检查state_dict是否为空（所有参数都被使用）
        
        示例:
            model = Model()
            state = torch.load("checkpoint.pt")
            model.load_state_dict(state)
        
        注意事项：
            - state_dict会被修改（成功加载的键会被移除）
            - 参数的shape和dtype必须完全匹配
            - 不允许state_dict中有多余的键
        """
        for name, param in self.__dict__.items():
            # 跳过私有属性
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                # Tensor参数：从state_dict中获取并验证
                item = state_dict.pop(_concat_prefix(prefix, name))
                assert isinstance(item, torch.Tensor)
                # 验证shape和dtype必须匹配
                assert param.shape == item.shape and param.dtype == item.dtype
                # 替换参数
                setattr(self, name, item)
            elif isinstance(param, BaseOP):
                # 子模块：递归加载
                param.load_state_dict(
                    state_dict, prefix=_concat_prefix(prefix, name), _internal=True
                )
        # 在顶层调用时，检查是否还有未使用的键
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")


class StateLessOP(BaseOP):
    """
    无状态操作基类
    
    用于没有可训练参数的操作（如激活函数、Dropout等）。
    这些操作不需要加载或保存权重。
    
    与BaseOP的区别：
        - BaseOP: 有参数的层（如Linear、Embedding）
        - StateLessOP: 无参数的层（如ReLU、LayerNorm without learnable params）
    
    使用示例：
        class ReLUOP(StateLessOP):
            def forward(self, x):
                return torch.relu(x)
        
        # 无参数，state_dict为空
        relu = ReLUOP()
        assert relu.state_dict() == {}
    
    注意：
        - state_dict()返回空字典
        - load_state_dict()如果传入非空字典会报错
    """
    
    def __init__(self):
        """初始化无状态操作"""
        super().__init__()

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        """
        加载参数（对于无状态操作应该为空）
        
        参数:
            state_dict: 参数字典
            prefix: 参数前缀
            _internal: 是否为内部调用
        
        异常:
            RuntimeError: 如果state_dict非空（无状态操作不应有参数）
        """
        if not _internal and state_dict:
            _ = prefix  # 标记为已使用
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        """
        返回参数字典（对于无状态操作为空）
        
        参数:
            prefix: 参数前缀
            result: 结果字典
        
        返回:
            空字典或传入的result字典
        """
        _ = prefix  # 标记为已使用
        return result if result is not None else {}


T = TypeVar("T", bound=BaseOP)
"""类型变量，限定为BaseOP的子类"""


class OPList(BaseOP, Generic[T]):
    """
    操作列表容器（类似torch.nn.ModuleList）
    
    用于管理一组相同类型的操作（如多个Transformer层）。
    提供统一的state_dict/load_state_dict接口。
    
    类型参数:
        T: 列表中操作的类型，必须是BaseOP的子类
    
    属性:
        op_list: 操作对象的列表
    
    使用示例：
        # 创建多个Transformer层
        layers = [TransformerLayer() for _ in range(32)]
        layer_list = OPList(layers)
        
        # 使用索引访问
        output = layer_list.op_list[0].forward(input)
        
        # 遍历所有层
        x = input
        for layer in layer_list.op_list:
            x = layer.forward(x)
        
        # 保存和加载所有层的权重
        state = layer_list.state_dict()
        # state包含：{"0.weight": ..., "0.bias": ..., "1.weight": ..., ...}
    
    权重命名规则：
        - 每个操作使用数字索引作为前缀
        - 例如：第0层的权重为"0.weight"，第1层为"1.weight"
        - 递归处理：如果子操作也是OPList，则形成"0.0.weight"等嵌套结构
    """
    
    def __init__(self, ops: List[T]):
        """
        初始化操作列表
        
        参数:
            ops: 操作对象的列表，所有元素必须是BaseOP的子类
        """
        super().__init__()
        self.op_list = ops

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        """
        收集所有操作的参数
        
        遍历列表中的每个操作，递归收集它们的参数。
        每个操作使用数字索引作为前缀。
        
        参数:
            prefix: 参数名称前缀（内部使用）
            result: 结果字典（内部使用）
        
        返回:
            包含所有操作参数的字典
        
        示例:
            假设有2个层，每层有weight参数：
            state = layer_list.state_dict()
            # 结果：{
            #   "0.weight": Tensor(...),
            #   "1.weight": Tensor(...)
            # }
        """
        result = result if result is not None else {}
        # 遍历列表，使用索引作为前缀
        for i, op in enumerate(self.op_list):
            op.state_dict(prefix=_concat_prefix(prefix, str(i)), result=result)
        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        """
        加载所有操作的参数
        
        遍历列表中的每个操作，递归加载它们的参数。
        
        参数:
            state_dict: 参数字典
            prefix: 参数名称前缀（内部使用）
            _internal: 是否为内部递归调用
        
        异常:
            RuntimeError: 如果state_dict中有未使用的键
        
        示例:
            state = torch.load("checkpoint.pt")
            layer_list.load_state_dict(state)
        """
        # 遍历列表，按索引加载参数
        for i, op in enumerate(self.op_list):
            op.load_state_dict(state_dict, prefix=_concat_prefix(prefix, str(i)), _internal=True)
        # 检查是否还有未使用的键
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")
