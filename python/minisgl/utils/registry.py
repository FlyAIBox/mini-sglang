
from typing import Callable, Generic, List, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """
    通用注册表类
    
    用于管理和查找系统中的组件，例如不同的模型架构或 Attention 实现。
    通过装饰器模式注册组件。
    """
    def __init__(self, type: str):
        self._registry = {}
        self._type = type

    def register(self, name: str) -> Callable[[T], None]:
        """
        注册装饰器
        
        Args:
            name: 组件注册名称
            
        Raises:
            KeyError: 如果名称已存在
        """
        if name in self._registry:
            raise KeyError(f"{self._type} '{name}' is already registered.")

        def decorator(item: T) -> None:
            self._registry[name] = item

        return decorator

    def __getitem__(self, name: str) -> T:
        """获取注册的组件"""
        if name not in self._registry:
            raise KeyError(f"Unsupported {self._type}: {name}")
        return self._registry[name]

    def supported_names(self) -> List[str]:
        """返回所有支持的组件名称"""
        return list(self._registry.keys())
