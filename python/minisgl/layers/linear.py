
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import divide_even

from .base import BaseOP


class _LinearTPImpl(BaseOP):
    """
    张量并行线性层基类 (Tensor Parallel Linear Base)
    
    实现了线性层的基本存储和计算逻辑，但不包含通信操作。
    具体的并行策略（Row/Column）由子类实现。
    """

    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        has_bias: bool,
    ):
        self.full_input_size = full_isize
        self.full_output_size = full_osize
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        # 初始化当前进程负责的权重分片
        self.weight = torch.empty(local_osize, local_isize)
        self.bias = torch.empty(local_osize) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class LinearColParallelMerged(_LinearTPImpl):
    """
    列并行线性层 (Column Parallel Linear)
    
    将权重矩阵按列切分到多个 GPU 上。
    
    Y = X @ W = X @ [W1, W2] = [X @ W1, X @ W2] = [Y1, Y2]
    
    特点：
    - 输入 X 在所有 GPU 上是相同的（或复制的）。
    - 输出 Y 被切分到不同 GPU 上。
    - 通信：通常在 forward 后不需要通信，除非后接 LayerNorm 或 Cross Entropy。
             如果后接 Row Parallel 层，则直接传递分片输出即可。
    
    应用场景：
    - QKV Projection (Merged QKV)
    - MLP 的 Gate/Up Projection
    """
    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        has_bias: bool,
    ):
        # 验证所有输出尺寸都能被 tp_size 整除
        tp_info = get_tp_info()
        tp_output_sizes = [divide_even(size, tp_info.size) for size in output_sizes]
        output_size = sum(output_sizes)
        tp_output_size = sum(tp_output_sizes)
        super().__init__(input_size, output_size, input_size, tp_output_size, has_bias)


class LinearQKVMerged(_LinearTPImpl):
    """
    合并的 QKV 投影层 (Merged QKV Projection)
    
    一种特殊的 Column Parallel Linear，用于同时计算 Query, Key, Value。
    支持 GQA (Grouped Query Attention)，即 KV 头数少于 Q 头数。
    """
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()

        # GQA 比例
        GQA_ratio = divide_even(num_qo_heads, num_kv_heads)
        local_num_kv = divide_even(num_kv_heads, tp_info.size)
        
        full_isize = hidden_size
        # 输出总大小：(Qheads + Kheads + Vheads) * head_dim
        # Qheads = GQA_ratio * Kheads
        full_osize = (GQA_ratio + 2) * num_kv_heads * head_dim
        
        local_isize = hidden_size
        local_osize = (GQA_ratio + 2) * local_num_kv * head_dim
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)


class LinearOProj(_LinearTPImpl):
    """
    输出投影层 (Output Projection) - 行并行
    
    专门用于 Attention 输出的投影。通常采用 Row Parallel。
    """
    def __init__(self, input_size: int, output_size: int, has_bias: bool):
        tp_info = get_tp_info()
        full_isize = input_size
        full_osize = output_size
        local_isize = divide_even(input_size, tp_info.size)
        local_osize = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 本地计算：Y_i = X_i @ W_i
        y = F.linear(x, self.weight, self.bias)
        # 2. All-Reduce：Y = Sum(Y_i)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


class LinearRowParallel(_LinearTPImpl):
    """
    行并行线性层 (Row Parallel Linear)
    
    将权重矩阵按行切分到多个 GPU 上。
    
    Y = X @ W = [X1, X2] @ [W1; W2] = X1 @ W1 + X2 @ W2 = Y1 + Y2
    
    特点：
    - 输入 X 是被切分的（通常来自上一个 Column Parallel 层的输出）。
    - 输出 Y 在所有 GPU 上需要是完整的（Sum 之后）。
    - 通信：forward 后需要进行 All-Reduce 求和。
    
    应用场景：
    - MLP 的 Down Projection
    - Attention 的 Output Projection
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()
        local_input_size = divide_even(input_size, tp_info.size)
        local_output_size = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(input_size, output_size, local_input_size, local_output_size, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 本地计算：Y_i = X_i @ W_i
        y = F.linear(x, self.weight, self.bias)
        # 2. All-Reduce：Y = Sum(Y_i)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y
