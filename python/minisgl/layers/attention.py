
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import divide_even

from .base import StateLessOP
from .rotary import get_rope

if TYPE_CHECKING:
    from minisgl.layers import RMSNorm
    from minisgl.models import RotaryConfig


class AttentionLayer(StateLessOP):
    """
    注意力层 (Attention Layer)
    
    实现 Transformer 的注意力机制核心逻辑：
    1. QKV 分割
    2. QK Norm (可选)
    3. Rotary Position Embedding (RoPE)
    4. Attention 计算 (通过 Backend 调用 FlashAttention/FlashInfer)
    
    该层是无状态的 (StateLessOP)，因为它不包含可训练参数（参数在 Model 的 Linear 层中）。
    它只负责计算逻辑。
    """
    def __init__(
        self,
        layer_id: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_config: RotaryConfig,
        q_norm: RMSNorm | None = None,
        k_norm: RMSNorm | None = None,
    ):
        # 确保 Query 头数能被 KV 头数整除，这是 GQA/MQA 机制的要求
        # GQA (Grouped Query Attention): 多个 Query 头共享一组 KV 头，减少 KV Cache 显存占用
        #   例如: 32 个 Q 头共享 8 个 KV 头，即每 4 个 Q 头共享 1 个 KV 头
        # MQA (Multi-Query Attention): 所有 Query 头共享同一个 KV 头，显存占用最小
        #   例如: 32 个 Q 头共享 1 个 KV 头
        # 标准 MHA (Multi-Head Attention): Q 头数等于 KV 头数 (1:1 对应)
        assert num_qo_heads % num_kv_heads == 0
        self.layer_id = layer_id
        self.head_dim = head_dim
        
        # 获取 Tensor Parallel 大小，计算当前进程负责的头数
        tp_size = get_tp_info().size
        self.num_qo_heads = divide_even(num_qo_heads, tp_size)
        self.num_kv_heads = divide_even(num_kv_heads, tp_size)
        
        # 计算本地维度的总大小
        self.qo_attn_dim = self.num_qo_heads * head_dim
        self.kv_attn_dim = self.num_kv_heads * head_dim
        
        # 初始化 RoPE 计算器
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=rotary_config.rotary_dim,
            max_position=rotary_config.max_position,
            base=rotary_config.base,
            rope_scaling=tuple(rotary_config.scaling.items()) if rotary_config.scaling else None,
        )
        # QK Norm (部分模型如 Qwen 需要)
        self.q_norm = q_norm
        self.k_norm = k_norm

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            qkv: 融合的 QKV 张量 [num_tokens, hidden_size + 2 * kv_hidden_size]
        
        Returns:
            torch.Tensor: Attention 输出 [num_tokens, hidden_size]
        """
        # 获取全局上下文
        # ⚠️ 警告: 必须在 Engine.forward_batch 上下文管理器内调用此方法
        # 否则 ctx.batch 会因未设置而抛出 AssertionError
        ctx = get_global_ctx()
        metadata = ctx.batch.attn_metadata
        
        # 1. 拆分 Q, K, V
        # shape: [num_tokens, num_heads * head_dim]
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        
        # 2. QK Norm (如有)
        if self.q_norm is not None:
            self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
        if self.k_norm is not None:
            self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
            
        # 3. 应用 RoPE
        # 如果是 Decode 阶段，RoPE 会使用 cached_len 计算位置
        if self.rotary:
            q, k = self.rotary.forward(metadata.positions, q, k)
            
        # 4. Attention 计算
        # 调用 context 中的 backend (FlashAttention 或 FlashInfer) 执行实际计算
        # 传入 global_context 用于获取 KV Cache 等信息
        q = q.view(-1, self.num_qo_heads, self.head_dim)
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
        
        return o.view(-1, self.qo_attn_dim)
