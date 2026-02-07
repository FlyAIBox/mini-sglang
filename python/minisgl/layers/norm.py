
from typing import Tuple

import torch

from .base import BaseOP


class RMSNorm(BaseOP):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    
    相比 LayerNorm，RMSNorm省去了均值计算，只计算均方根，
    计算量更小且效果相当，广泛应用于 Llama, Qwen 等现代 LLM。
    
    公式:
    RMS(x) = sqrt(mean(x^2) + eps)
    output = x / RMS(x) * weight
    """
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        """In-place计算，节省显存"""
        self.rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(BaseOP):
    """
    融合残差连接的 RMSNorm (Fused Add + RMSNorm)
    
    将 Residual Add 和 RMSNorm 两个操作融合为一个 Kernel 执行，
    减少显存读写次数，显著提升带宽利用率。
    
    Input: x, residual
    Output:
    1. norm_output = RMSNorm(x + residual)  -> 用于下一层输入
    2. residual = x + residual              -> 用于下一次残差累加
    """
    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import fused_add_rmsnorm, rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 当前层的输出
            residual: 之前的残差累加值
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (normed_output, new_residual)
        """
        if residual is None:
            # 如果没有 residual (通常是第一层)，直接做 RMSNorm
            return self.rmsnorm(x, self.weight, self.eps), x
        
        # Fused kernel: 同时更新 x (作为 normed output) 和 residual
        # 注意：FlashInfer 的实现可能略有不同，这里根据接口推测
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual
