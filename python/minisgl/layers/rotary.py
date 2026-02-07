
from __future__ import annotations

import math
from functools import lru_cache
from typing import Any, Callable, Dict, Tuple

import torch

from .base import StateLessOP


class RotaryEmbedding(StateLessOP):
    """
    旋转位置编码 (Rotary Position Embedding, RoPE)
    
    这是现代 LLM (如 Llama, Qwen) 中最常用的位置编码方式。
    通过将 query 和 key 向量在复数平面上进行旋转，注入绝对位置信息，
    同时使得 Attention 具有相对位置感知能力。
    
    实现逻辑：
    1. 预计算频率 (inv_freq)：theta_i = 10000 ^ (-2i/d)
    2. 预计算 Cos/Sin 缓存 (_cos_sin_cache)：结合最大位置长度生成位置向量。
    3. 前向传播：调用 FlashInfer 的 kernel 进行 inplace 旋转。
    """
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        post_process: None | Callable[[torch.Tensor], torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        
        # 1. 计算频率 theta
        # inv_freq shape: [rotary_dim / 2]
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        
        # 可选的后处理 (例如 Llama 3 的 scaling)
        if post_process is not None:
            inv_freq = post_process(inv_freq)
            
        # 2. 生成位置索引 t: [0, 1, ..., max_pos-1]
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        
        # 3. 计算 freqs = t * theta
        # outer product: [max_pos] x [dim/2] -> [max_pos, dim/2]
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        
        # 4. 生成 cos/sin 表
        cos = freqs.cos()
        sin = freqs.sin()
        
        # 拼接用于 FlashInfer kernel
        # cache shape: [max_pos, dim] (dim = 2 * dim/2)
        # buffer, so don't load/save
        self._cos_sin_cache = torch.cat((cos, sin), dim=-1)
        assert self.head_size in [64, 128, 256, 512]

        # 引入 FlashInfer 的 RoPE kernel
        from flashinfer import apply_rope_with_cos_sin_cache_inplace

        self.apply_rope_with_cos_sin_cache_inplace = apply_rope_with_cos_sin_cache_inplace

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用 RoPE 旋转
        
        Args:
            positions: Token 的位置索引 [num_tokens]
            query: Query 张量 [num_tokens, num_heads, head_dim]
            key: Key 张量 [num_tokens, num_kv_heads, head_dim]
            
        Returns:
            Tuple[Tensor, Tensor]: 旋转后的 Query 和 Key
        """
        # In-place 修改 query 和 key
        self.apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_size,
            cos_sin_cache=self._cos_sin_cache,
        )
        return query, key


def _get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Dict[str, Any] | None = None,
) -> RotaryEmbedding:
    if rope_scaling is None:
        return RotaryEmbedding(head_dim, rotary_dim, max_position, base)
    
    # RoPE Scaling 处理 (用于长上下文扩展)
    match rope_scaling["rope_type"]:
        case "llama3":
            # Llama 3 特有的 scaling 策略
            scaling_factor: float = rope_scaling["factor"]
            low_freq_factor: float = rope_scaling["low_freq_factor"]
            high_freq_factor: float = rope_scaling["high_freq_factor"]
            original_max_position: int = rope_scaling["original_max_position_embeddings"]

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                # 只有低频部分 (low freq) 需要 scaling，高频部分 (high freq) 保持不变
                # 这样可以保持局部注意力的精确性，同时扩展以前未见过的长距离依赖
                wave_len = 2 * math.pi / inv_freq
                if low_freq_factor == high_freq_factor:
                    return torch.where(
                        wave_len < original_max_position / high_freq_factor,
                        inv_freq,
                        inv_freq / scaling_factor,
                    )

                delta = high_freq_factor - low_freq_factor
                # 平滑过渡区域
                smooth = (original_max_position / wave_len - low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                factor = (1 - smooth) / scaling_factor + smooth
                return factor * inv_freq

            return RotaryEmbedding(head_dim, rotary_dim, max_position, base, post_process)

    raise ValueError(f"Unsupported {rope_scaling = }")


_ROPE_DEVICE: torch.device | None = None


def set_rope_device(device: torch.device):
    """设置 RoPE 缓存所在的设备，通常在 Engine 初始化时调用"""
    global _ROPE_DEVICE
    _ROPE_DEVICE = device


@lru_cache()
def get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Tuple[Tuple[str, Any], ...] | None = None,
) -> RotaryEmbedding:
    """
    获取 RoPE 实例 (带缓存)
    
    因为 RoPE 是无状态的且计算开销较大 (涉及大张量分配)，
    所以使用 lru_cache 复用相同配置的实例。
    """
    rope_map = dict(rope_scaling) if rope_scaling is not None else None
    t = torch.tensor([])
    if t.device == torch.device("meta"):
        # we cannot use meta device for rope
        # 确保在真实设备上创建 RoPE cache，因为 meta device 无法存储数值
        if _ROPE_DEVICE is None:
            raise RuntimeError(
                "We cannot use meta device for rope. Please call set_rope_device() first."
            )
        with torch.device(_ROPE_DEVICE):
            return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)
    return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)


__all__ = ["get_rope", "RotaryEmbedding", "set_rope_device"]
