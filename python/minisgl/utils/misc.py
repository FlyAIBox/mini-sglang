
from __future__ import annotations


def call_if_main(name: str = "__main__", discard: bool | None = None):
    """
    装饰器：确保函数只在作为脚本直接运行时调用
    
    用于替代 `if __name__ == "__main__":` 代码块。
    支持可选的返回值丢弃。
    
    Args:
        name: 模块名称，通常传 `__name__`
        discard: 是否丢弃返回值
    """
    if name != "__main__":
        # 如果不是 main 运行，返回一个空函数
        discard = False if discard is None else discard
        if discard:
            return lambda _: None
        else:
            return lambda f: f
    else:
        # 如果是 main 运行，立即执行该函数
        discard = True if discard is None else discard
        if discard:
            return lambda f: (f() or True) and None
        else:
            return lambda f: (f() and None) or f


def divide_even(a: int, b: int) -> int:
    """整除检查，确保 a 能被 b 整除"""
    assert a % b == 0, f"{a = } must be divisible by {b = }"
    return a // b


def divide_up(a: int, b: int) -> int:
    """向上取整除法"""
    return (a + b - 1) // b


def divide_down(a: int, b: int) -> int:
    """向下取整除法"""
    return a // b


class Unset:
    """标识未设置值的哨兵类"""
    pass


UNSET = Unset()
