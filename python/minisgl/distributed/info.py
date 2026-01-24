"""
分布式信息管理模块

本模块管理张量并行（Tensor Parallelism, TP）的分布式信息，
包括当前进程的rank和world size。

张量并行概念：
    在张量并行中，模型的每一层被切分到多个GPU上。每个GPU运行一个独立的进程，
    称为一个"TP rank"。所有进程共同协作完成模型推理。
    
    例如：使用4个GPU进行TP
    - Rank 0: GPU 0，处理模型参数的第1/4
    - Rank 1: GPU 1，处理模型参数的第2/4
    - Rank 2: GPU 2，处理模型参数的第3/4
    - Rank 3: GPU 3，处理模型参数的第4/4
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedInfo:
    """
    分布式配置信息
    
    存储当前进程在张量并行中的位置信息。这个类是不可变的（frozen=True），
    确保分布式配置在运行时不会被意外修改。
    
    属性:
        rank (int): 当前进程的rank（进程编号）
            - 范围: 0 到 size-1
            - Rank 0 通常是主进程，负责额外的协调工作
            - 例如: 在4-GPU TP中，rank可以是0, 1, 2, 3
            
        size (int): 张量并行的总进程数（world size）
            - 也就是使用的GPU数量
            - 例如: --tp 4 表示size=4
    
    验证:
        初始化后自动验证 rank 在有效范围内 [0, size)
    
    示例:
        # 4-GPU张量并行配置
        tp_info_rank0 = DistributedInfo(rank=0, size=4)  # 主进程
        tp_info_rank1 = DistributedInfo(rank=1, size=4)
        tp_info_rank2 = DistributedInfo(rank=2, size=4)
        tp_info_rank3 = DistributedInfo(rank=3, size=4)
    """
    rank: int
    size: int

    def __post_init__(self):
        """
        初始化后验证
        
        确保rank在有效范围内：0 <= rank < size
        """
        assert 0 <= self.rank < self.size, \
            f"Invalid rank {self.rank} for size {self.size}"

    def is_primary(self) -> bool:
        """
        判断当前进程是否为主进程
        
        主进程（Rank 0）通常负责额外的协调工作：
        - 与Tokenizer/Detokenizer通信
        - 收集和分发请求
        - 日志输出和监控
        
        返回:
            bool: True表示这是主进程（rank==0）
        
        使用场景:
            if tp_info.is_primary():
                # 只有主进程执行
                send_to_tokenizer(request)
        """
        return self.rank == 0


# ============================================================================
# 全局TP信息管理
# ============================================================================

# 全局变量：存储当前进程的TP信息
# 每个进程维护自己的_TP_INFO
_TP_INFO: DistributedInfo | None = None


def set_tp_info(rank: int, size: int) -> None:
    """
    设置当前进程的张量并行信息
    
    在Scheduler Worker进程启动时调用，设置该进程在TP组中的位置。
    每个进程只能设置一次，防止运行时被意外修改。
    
    参数:
        rank (int): 当前进程的rank（0到size-1）
        size (int): 张量并行的总进程数
    
    异常:
        RuntimeError: 如果TP信息已经被设置（防止重复设置）
    
    使用场景:
        # 在Scheduler Worker启动时
        def start_scheduler_worker(rank: int, world_size: int):
            set_tp_info(rank, world_size)
            # 现在可以使用get_tp_info()获取信息
            ...
    
    示例:
        # 启动4个进程进行4-GPU TP
        # 进程0: set_tp_info(0, 4)
        # 进程1: set_tp_info(1, 4)
        # 进程2: set_tp_info(2, 4)
        # 进程3: set_tp_info(3, 4)
    """
    global _TP_INFO
    if _TP_INFO is not None:
        raise RuntimeError("TP info has been set. Cannot set twice.")
    _TP_INFO = DistributedInfo(rank, size)


def get_tp_info() -> DistributedInfo:
    """
    获取当前进程的张量并行信息
    
    在模型和各个模块中调用，获取当前进程的TP配置。
    必须在set_tp_info()之后调用。
    
    返回:
        DistributedInfo: 当前进程的TP信息
    
    异常:
        RuntimeError: 如果TP信息未设置（需要先调用set_tp_info）
    
    使用场景:
        # 在模型中根据TP信息切分参数
        def load_weights(self):
            tp_info = get_tp_info()
            tp_rank = tp_info.rank
            tp_size = tp_info.size
            
            # 只加载属于当前rank的参数
            weight_slice = full_weight[
                tp_rank * slice_size : (tp_rank + 1) * slice_size
            ]
    
    示例:
        tp_info = get_tp_info()
        print(f"I am rank {tp_info.rank} of {tp_info.size}")
        # 输出: "I am rank 2 of 4"
        
        if tp_info.is_primary():
            print("I am the primary rank!")
    """
    if _TP_INFO is None:
        raise RuntimeError(
            "TP info has not been set. Call set_tp_info() first."
        )
    return _TP_INFO


def try_get_tp_info() -> DistributedInfo | None:
    """
    尝试获取张量并行信息（安全版本）
    
    与get_tp_info()不同，这个函数不会在TP信息未设置时抛出异常，
    而是返回None。适用于可选的TP相关逻辑。
    
    返回:
        DistributedInfo | None: 
            - 如果TP信息已设置，返回DistributedInfo对象
            - 如果TP信息未设置，返回None
    
    使用场景:
        # 某些逻辑需要在TP环境下做特殊处理，非TP环境下跳过
        tp_info = try_get_tp_info()
        if tp_info is not None and tp_info.size > 1:
            # 使用张量并行
            result = distributed_forward(input)
        else:
            # 单GPU模式
            result = forward(input)
    
    示例:
        # 初始化前
        info = try_get_tp_info()  # 返回None
        
        # 初始化后
        set_tp_info(0, 4)
        info = try_get_tp_info()  # 返回DistributedInfo(rank=0, size=4)
    """
    return _TP_INFO


__all__ = ["DistributedInfo", "set_tp_info", "get_tp_info", "try_get_tp_info"]
