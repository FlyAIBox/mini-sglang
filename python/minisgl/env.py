"""
Mini-SGLang 环境变量管理模块

本模块提供统一的环境变量管理机制，用于在运行时配置系统行为。
通过环境变量可以控制特性开关、调试选项、性能参数等，无需修改代码。

主要功能：
- 类型安全的环境变量读取（int、float、bool、内存大小）
- 默认值支持
- 单例模式确保全局一致性
- 统一的命名前缀（MINISGL_）

使用示例：
    # 在shell中设置环境变量
    export MINISGL_DISABLE_OVERLAP_SCHEDULING=1
    export MINISGL_SHELL_MAX_TOKENS=4096
    
    # 在Python代码中访问
    from minisgl.env import ENV
    if ENV.DISABLE_OVERLAP_SCHEDULING:
        # 使用简单调度模式
        pass
"""

from __future__ import annotations

import os
from functools import partial
from typing import Callable, Generic, TypeVar


class BaseEnv:
    """
    环境变量基类
    
    定义环境变量类的通用接口。所有具体的环境变量类型
    都继承自这个基类。
    """
    def _init(self, name: str) -> None:
        """
        初始化环境变量
        
        从操作系统环境中读取变量值。子类必须实现此方法。
        
        参数:
            name: 环境变量名称（包括MINISGL_前缀）
        """
        raise NotImplementedError


T = TypeVar("T")


class EnvVar(BaseEnv, Generic[T]):
    """
    泛型环境变量类
    
    支持任意类型的环境变量，通过转换函数将字符串转换为目标类型。
    
    类型参数:
        T: 环境变量的目标类型（如int、bool、float等）
    
    属性:
        value: 当前环境变量的值
        fn: 字符串到目标类型的转换函数
    
    示例:
        # 创建一个整数类型的环境变量，默认值为100
        max_tokens = EnvVar(default_value=100, fn=int)
        max_tokens._init("MINISGL_MAX_TOKENS")
        print(max_tokens.value)  # 输出环境变量值或默认值100
    """
    def __init__(self, default_value: T, fn: Callable[[str], T]):
        """
        初始化环境变量
        
        参数:
            default_value: 默认值（当环境变量未设置时使用）
            fn: 字符串转换函数，将环境变量字符串转换为目标类型
        """
        self.value = default_value
        self.fn = fn
        super().__init__()

    def _init(self, name: str) -> None:
        """
        从操作系统环境中读取变量值
        
        如果环境变量存在，使用转换函数解析其值。
        如果解析失败或变量不存在，保持默认值。
        
        参数:
            name: 环境变量名称
        """
        env_value = os.getenv(name)
        if env_value is not None:
            try:
                self.value = self.fn(env_value)
            except Exception:
                # 解析失败时静默失败，保持默认值
                # 这样可以避免因配置错误导致程序崩溃
                pass

    def __bool__(self):
        """
        布尔转换
        
        允许在if语句中直接使用环境变量：
        if ENV.SOME_FLAG:
            ...
        """
        return self.value

    def __str__(self):
        """
        字符串表示
        
        方便日志输出和调试
        """
        return str(self.value)


# ============================================================================
# 类型转换函数
# ============================================================================

_TO_BOOL = lambda x: x.lower() in ("1", "true", "yes")
"""
布尔值转换函数

支持多种布尔值表示：
- "1", "true", "yes" → True
- 其他任何值 → False

不区分大小写。

示例:
    MINISGL_SOME_FLAG=1      → True
    MINISGL_SOME_FLAG=true   → True
    MINISGL_SOME_FLAG=YES    → True
    MINISGL_SOME_FLAG=0      → False
    MINISGL_SOME_FLAG=false  → False
"""


def _PARSE_MEM_BYTES(mem: str) -> int:
    """
    解析内存大小字符串，转换为字节数
    
    支持的格式：
    - 纯数字：视为字节数
    - K/KB：千字节（1024字节）
    - M/MB：兆字节（1024²字节）
    - G/GB：吉字节（1024³字节）
    
    不区分大小写，支持浮点数。
    
    参数:
        mem: 内存大小字符串
    
    返回:
        int: 字节数
    
    示例:
        _PARSE_MEM_BYTES("1024")     → 1024
        _PARSE_MEM_BYTES("1K")       → 1024
        _PARSE_MEM_BYTES("1.5M")     → 1572864
        _PARSE_MEM_BYTES("2GB")      → 2147483648
        _PARSE_MEM_BYTES("0.5G")     → 536870912
    """
    mem = mem.strip().upper()
    # 如果最后一个字符不是字母，说明是纯数字
    if not mem[-1].isalpha():
        return int(mem)
    # 去掉可选的"B"后缀（如"1GB"中的"B"）
    if mem.endswith("B"):
        mem = mem[:-1]
    # 单位映射表
    UNIT_MAP = {"K": 1024, "M": 1024**2, "G": 1024**3}
    # 提取数字部分和单位，进行转换
    return int(float(mem[:-1]) * UNIT_MAP[mem[-1]])


# ============================================================================
# 环境变量类型别名
# ============================================================================

MINISGL_ENV_PREFIX = "MINISGL_"
"""所有Mini-SGLang环境变量的统一前缀"""

# 创建常用类型的环境变量构造函数
EnvInt = partial(EnvVar[int], fn=int)
"""整数类型环境变量"""

EnvFloat = partial(EnvVar[float], fn=float)
"""浮点数类型环境变量"""

EnvBool = partial(EnvVar[bool], fn=_TO_BOOL)
"""布尔类型环境变量"""

EnvOption = partial(EnvVar[bool | None], fn=_TO_BOOL, default_value=None)
"""可选布尔类型环境变量（默认为None，表示未设置）"""

EnvMem = partial(EnvVar[int], fn=_PARSE_MEM_BYTES)
"""内存大小类型环境变量（支持K/M/G单位）"""


# ============================================================================
# 全局环境变量配置
# ============================================================================

class EnvClassSingleton:
    """
    环境变量单例类
    
    使用单例模式确保全局只有一个环境变量配置实例。
    在首次创建时自动从操作系统环境中读取所有配置。
    
    设计理念：
    - 单例模式：全局唯一实例，避免重复读取环境变量
    - 声明式：所有环境变量集中声明，一目了然
    - 类型安全：每个变量都有明确的类型
    - 默认值：提供合理的默认值，开箱即用
    
    使用方式：
        from minisgl.env import ENV
        
        # 访问环境变量
        max_tokens = ENV.SHELL_MAX_TOKENS.value
        
        # 布尔判断
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            # 禁用重叠调度
            pass
    """
    _instance: EnvClassSingleton | None = None

    # ------------------------------------------------------------------------
    # Shell 交互式命令行配置
    # ------------------------------------------------------------------------
    SHELL_MAX_TOKENS = EnvInt(2048)
    """
    Shell模式下的最大生成token数
    
    环境变量: MINISGL_SHELL_MAX_TOKENS
    默认值: 2048
    说明: 控制交互式shell中单次对话的最大输出长度
    """
    
    SHELL_TOP_K = EnvInt(-1)
    """
    Shell模式下的Top-K采样参数
    
    环境变量: MINISGL_SHELL_TOP_K
    默认值: -1 (不使用top_k)
    说明: 只从概率最高的K个token中采样，-1表示不限制
    """
    
    SHELL_TOP_P = EnvFloat(1.0)
    """
    Shell模式下的Top-P采样参数
    
    环境变量: MINISGL_SHELL_TOP_P
    默认值: 1.0 (不使用top_p)
    说明: Nucleus采样，从累积概率达到p的token集合中采样
    """
    
    SHELL_TEMPERATURE = EnvFloat(0.6)
    """
    Shell模式下的温度参数
    
    环境变量: MINISGL_SHELL_TEMPERATURE
    默认值: 0.6
    说明: 控制输出随机性，越低越确定，越高越随机
    """

    # ------------------------------------------------------------------------
    # 后端运行时配置
    # ------------------------------------------------------------------------
    FLASHINFER_USE_TENSOR_CORES = EnvOption()
    """
    FlashInfer是否使用Tensor Cores
    
    环境变量: MINISGL_FLASHINFER_USE_TENSOR_CORES
    默认值: None (由FlashInfer自动决定)
    说明: 
        - 设置为1/true: 强制使用Tensor Cores（可能更快）
        - 设置为0/false: 禁用Tensor Cores（可能更精确）
        - 不设置: 让FlashInfer根据GPU架构自动选择
    """
    
    DISABLE_OVERLAP_SCHEDULING = EnvBool(False)
    """
    禁用重叠调度优化
    
    环境变量: MINISGL_DISABLE_OVERLAP_SCHEDULING
    默认值: False (启用重叠调度)
    说明: 
        - False: CPU调度与GPU计算重叠执行，提高吞吐量（默认）
        - True: 顺序执行，方便调试
    
    使用场景：
        - 调试调度逻辑时设置为True
        - 性能profiling时可以分别测试两种模式
    """
    
    OVERLAP_EXTRA_SYNC = EnvBool(False)
    """
    重叠调度时添加额外的同步点
    
    环境变量: MINISGL_OVERLAP_EXTRA_SYNC
    默认值: False
    说明: 
        - False: 最小化同步，最高性能（默认）
        - True: 增加同步点，便于调试和验证正确性
    
    使用场景：
        - 怀疑有数据竞争或同步问题时设置为True
    """
    
    PYNCCL_MAX_BUFFER_SIZE = EnvMem(1024**3)
    """
    PyNCCL通信缓冲区的最大大小
    
    环境变量: MINISGL_PYNCCL_MAX_BUFFER_SIZE
    默认值: 1GB (1024^3 字节)
    说明: 限制NCCL通信时单个操作的最大buffer大小
    
    使用示例：
        export MINISGL_PYNCCL_MAX_BUFFER_SIZE=2G    # 2GB
        export MINISGL_PYNCCL_MAX_BUFFER_SIZE=512M  # 512MB
    """

    def __new__(cls):
        """
        实现单例模式
        
        确保全局只有一个EnvClassSingleton实例。
        无论调用多少次，都返回同一个对象。
        """
        # 如果还没有实例，创建新实例
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """
        初始化环境变量
        
        遍历所有声明的环境变量属性，从操作系统环境中读取它们的值。
        这个方法在创建单例时自动调用。
        
        工作流程：
        1. 遍历类的所有属性
        2. 跳过私有属性（以_开头的）
        3. 对每个环境变量，调用_init方法从系统环境读取
        4. 自动添加MINISGL_前缀
        """
        for attr_name in dir(self):
            # 跳过私有属性和方法
            if attr_name.startswith("_"):
                continue
            attr_value = getattr(self, attr_name)
            # 确保是环境变量类型
            assert isinstance(attr_value, BaseEnv)
            # 从系统环境中读取，添加统一前缀
            attr_value._init(f"{MINISGL_ENV_PREFIX}{attr_name}")


# ============================================================================
# 全局环境变量实例
# ============================================================================

ENV = EnvClassSingleton()
"""
全局环境变量实例

这是整个Mini-SGLang系统中唯一的环境变量配置对象。
在模块导入时自动创建并初始化。

使用示例:
    from minisgl.env import ENV
    
    # 读取配置
    if ENV.DISABLE_OVERLAP_SCHEDULING:
        logger.info("Overlap scheduling is disabled")
    
    max_tokens = ENV.SHELL_MAX_TOKENS.value
    print(f"Shell max tokens: {max_tokens}")

可用的环境变量:
    Shell配置:
        - MINISGL_SHELL_MAX_TOKENS: shell最大token数（默认2048）
        - MINISGL_SHELL_TOP_K: shell的top_k参数（默认-1）
        - MINISGL_SHELL_TOP_P: shell的top_p参数（默认1.0）
        - MINISGL_SHELL_TEMPERATURE: shell的temperature（默认0.6）
    
    运行时配置:
        - MINISGL_FLASHINFER_USE_TENSOR_CORES: 是否使用tensor cores
        - MINISGL_DISABLE_OVERLAP_SCHEDULING: 禁用重叠调度（默认False）
        - MINISGL_OVERLAP_EXTRA_SYNC: 添加额外同步点（默认False）
        - MINISGL_PYNCCL_MAX_BUFFER_SIZE: NCCL缓冲区大小（默认1GB）
"""
