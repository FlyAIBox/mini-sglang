
"""
Llama 模型实现

Llama (Large Language Model Meta AI) 是当前最流行的开源大模型架构之一。
本文件实现了 Llama 模型的推理逻辑，支持张量并行 (Tensor Parallelism)。

主要组件：
1. LlamaDecoderLayer: 单个 Transformer 层
2. LlamaModel: 模型主体（不含 Head）
3. LlamaForCausalLM: 完整的因果语言模型
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .utils import GatedMLP as LlamaMLP
from .utils import RopeAttn as LlamaAttn

if TYPE_CHECKING:
    from .config import ModelConfig


class LlamaDecoderLayer(BaseOP):
    """
    Llama 解码层 (Transformer Block)
    
    标准的 Decoder-only Transformer 层结构。
    
    结构：
    x = x + SelfAttention(RMSNorm(x))
    x = x + MLP(RMSNorm(x))
    
    组件：
    - Input LayerNorm: RMSNorm
    - Attention: Rotating Position Embeddings (RoPE) + FlashAttention
    - Post Attention LayerNorm: RMSNorm
    - Feed Forward: Gated MLP (SwiGLU activation)
    """
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = LlamaAttn(config, layer_id)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        单层前向传播
        
        Args:
            x: 输入张量
            residual: 残差张量。Mini-SGLang 使用 "Fused Add RMSNorm" 优化，
                     因此 residual 会在层间传递并由 Norm 层处理。
                     
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (输出张量, 更新后的残差)
        """
        # 1. Self Attention: x = x + Attention(Norm(x))
        # input_layernorm 会同时执行: output = norm(x + residual); residual = x + residual
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        
        # 2. MLP: x = x + MLP(Norm(x))
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class LlamaModel(BaseOP):
    """
    Llama 模型主体
    
    包含:
    1. Embedding 层: 将 Token ID 转换为向量
    2. N 个 LlamaDecoderLayer: 堆叠的 Transformer 层
    3. Final RMSNorm: 最终的归一化层
    """
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [LlamaDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        模型主体前向传播
        
        Args:
            input_ids: 输入的 Token IDs [batch_size] (展平的)
            
        Returns:
            torch.Tensor: 最后一层的输出特征 [batch_size, hidden_size]
        """
        # 1. Embedding
        x = self.embed_tokens.forward(input_ids)
        
        # 2. Transformer Layers
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
            
        # 3. Final Norm
        # 注意：residual 在这里被最后一次加到 x 上并进行归一化
        return self.norm.forward(x, residual)[0]


class LlamaForCausalLM(BaseLLMModel):
    """
    完整的 Llama 因果语言模型
    
    封装了 LlamaModel 和 LM Head（输出层）。
    继承自 BaseLLMModel，是调度器直接交互的对象。
    """
    def __init__(self, config: ModelConfig):
        self.model = LlamaModel(config)
        # LM Head: 将 hidden_state 映射回 词表大小 的 logits
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        """
        执行模型推理
        
        不接受参数，直接从 get_global_ctx() 获取当前的 batch 和 input_ids。
        这是为了配合 Engine 的工作流。
        """
        # 1. 获取输入
        # 2. 模型主体 Forward
        output = self.model.forward(get_global_ctx().batch.input_ids)
        
        # 3. LM Head Forward -> Generate Logits
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["LlamaForCausalLM"]
