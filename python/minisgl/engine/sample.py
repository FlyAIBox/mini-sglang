from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from minisgl.core import Batch


@dataclass
class BatchSamplingArgs:
    """
    批次采样参数
    
    聚合当前 Batch 中所有请求的采样配置，转换为 Tensor 格式供 Sampling Kernel 使用。
    """
    temperatures: torch.Tensor | None  # 温度参数 [batch_size]，None 表示全为 Greedy
    top_k: torch.Tensor | None = None  # Top-K 参数 [batch_size]
    top_p: torch.Tensor | None = None  # Top-P 参数 [batch_size]


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """辅助函数：创建并移动 Tensor 到指定设备 (non-blocking)"""
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    """
    采样实现 (调用 FlashInfer)
    
    根据 logits 和采样参数生成下一个 Token。
    """
    import flashinfer.sampling as sampling

    # 计算 Softmax 概率
    # enable_pdl: Hopper 架构 (SM90) 支持 Programmatic Dependent Launch 优化
    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    
    # 根据参数组合选择具体的采样 Kernel
    if top_k is None and top_p is None:
        # 纯随机采样 (Temperature only)
        return sampling.sampling_from_probs(probs)

    if top_p is None:
        # Top-K 采样
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)

    if top_k is None:
        # Top-P (Nucleus) 采样
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)

    # 混合 Top-K + Top-P 采样
    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


@dataclass
class Sampler:
    """
    采样器
    
    负责准备采样参数和执行采样操作。
    """
    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        """
        准备采样参数
        
        从 Batch 中提取每个 Req 的采样参数 (SamplingParams)，
        并将其堆叠成 Tensor。
        """
        params = [r.sampling_params for r in batch.reqs]
        
        # 优化：如果所有请求都是 Greedy (temp < epsilon)，则无需传递温度
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)

        MIN_P = MIN_T = 1e-6
        # 处理 Temperature (Greedy -> 0.0)
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        # 处理 Top-K (默认值为 -1 或 0 时表示不限制，这里设为 vocab_size)
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        # 处理 Top-P (限制范围 [MIN_P, 1.0])
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        
        # 仅当参数非默认值时才创建对应的 Tensor
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
            
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p)

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        """执行采样"""
        with torch.cuda.nvtx.range("Sampler"):
            if args.temperatures is None:
                # Greedy Sampling (Argmax) 优化路径
                return torch.argmax(logits, dim=-1)
            # Random Sampling (FlashInfer)
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)

