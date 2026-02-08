from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from transformers import LlamaConfig


@dataclass(frozen=True)
class RotaryConfig:
    """
    RoPE (Rotary Positional Embeddings) 配置
    
    用于配置旋转位置编码的参数。
    """
    head_dim: int
    rotary_dim: int  # 实际进行旋转的维度 (通常等于 head_dim)
    max_position: int  # 最大序列长度 (max_position_embeddings)
    base: float  # RoPE base (default 10000.0)
    scaling: Dict[str, float] | None  # 线性/动态缩放配置


@dataclass(frozen=True)
class ModelConfig:
    """
    模型统一配置类
    
    统一不同 HuggingFace 模型配置 (LlamaConfig, QwenConfig 等) 为 Mini-SGLang
    可识别的格式。
    
    主要用于初始化模型层、Setting up Attention 后端等。
    """
    num_layers: int
    num_qo_heads: int  # Query Head 数量
    num_kv_heads: int  # KV Head 数量 (用于 MQA/GQA)
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int  # MLP 中间层维度
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str  # 激活函数 (如 silu)
    tie_word_embeddings: bool  # 是否共享 output head 和 input embedding 权重

    @classmethod
    def from_hf(cls, config: LlamaConfig) -> ModelConfig:
        """从 HuggingFace Config 创建 ModelConfig"""
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=head_dim,
                max_position=config.max_position_embeddings,
                base=config.rope_theta,
                scaling=getattr(config, "rope_scaling", None),
            ),
        )

