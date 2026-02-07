
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import divide_up, nvtx_annotate

from .base import BaseOP


class VocabParallelEmbedding(BaseOP):
    """
    词表并行 Embedding 层 (Vocabulary Parallel Embedding)
    
    将 Embedding 矩阵按词表维度 (Vocab Dimension) 切分到多个 GPU 上。
    每个 GPU 负责一部分词表的 Embedding 查找。
    
    逻辑：
    1. 输入 X (token ids) 广播到所有 GPU。
    2. 每个 GPU 检查输入 X 中的 id 是否在自己负责的词表范围内。
       - 如果在范围内，通过 lookup 获取 embedding。
       - 如果不在范围内，结果为 0。
    3. All-Reduce (Sum)：将所有 GPU 的结果相加，得到完整的 embedding。
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        tp_info = get_tp_info()
        tp_rank = tp_info.rank
        self.tp_size = tp_info.size
        self.num_embeddings = num_embeddings
        
        # 计算每个 GPU 负责的词表大小
        self.num_embeddings_tp = divide_up(num_embeddings, self.tp_size)
        
        # 计算当前 GPU 负责的词表范围 [start, finish)
        start_idx = self.num_embeddings_tp * tp_rank
        finish_idx = min(start_idx + self.num_embeddings_tp, num_embeddings)
        self.vocab_range = (start_idx, finish_idx - start_idx)
        
        # 初始化权重分片
        self.weight = torch.empty(self.num_embeddings_tp, embedding_dim)
        self._comm = DistributedCommunicator()

    @nvtx_annotate("Embedding")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from minisgl.kernel import indexing

        # 1. 本地查找：只处理范围内的 token id，其他返回 0
        y = indexing(
            weights=self.weight,
            indices=x,
            vocab_range=self.vocab_range if self.tp_size > 1 else None,
        )

        # 2. All-Reduce：将各个 GPU 的查找结果相加
        return self._comm.all_reduce(y) if self.tp_size > 1 else y


class ParallelLMHead(VocabParallelEmbedding):
    """
    并行 LM Head (Output Layer)
    
    用于将 hidden_states 映射回 logits。通常与 Embedding 层共享权重 (Tied Embeddings)。
    同样采用 Vocab Parallel 策略：即按列 (Column) 切分权重矩阵。
    
    Y = X @ W_transposed
    由于 W 按照 Vocab 维度切分，相当于 Linear 层的 Column Parallel。
    
    Forward 逻辑：
    1. 本地计算：Y_i = X @ W_i^T
       - X 是完整的 (Full input from Row Parallel or Gathered)
       - Y_i 是部分的 logits (只包含部分词表的 scores)
    2. All-Gather：收集所有 Y_i，拼成完整的 Logits。
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        tie_word_embeddings: bool = False,
        tied_embedding: VocabParallelEmbedding | None = None,
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.bias = torch.empty(self.num_embeddings_tp) if bias else None
        self.tied_embedding = tied_embedding
        assert (tied_embedding is not None) == tie_word_embeddings

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        # 如果权重共享，不需要加载自己的权重，直接使用 tied_embedding 的权重
        if not self.tied_embedding:
            return super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)
        else:
            # 从 state_dict 中移除相关键值，避免报错
            possible_weight = f"{prefix}.weight"
            possible_bias = f"{prefix}.bias"
            if possible_weight in state_dict:
                state_dict.pop(possible_weight)
            if possible_bias in state_dict:
                state_dict.pop(possible_bias)

    def state_dict(
        self,
        *,
        prefix: str = "",
        result: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        # 如果权重共享，不需要保存自己的权重
        if not self.tied_embedding:
            return super().state_dict(prefix=prefix, result=result)
        return {} if result is None else result

    @nvtx_annotate("LMHead")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        bs = batch.size
        
        # 优化：如果是 Prefill 阶段，只取每条请求的最后一个 token 进行预测
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(bs)
            x = x[indices].contiguous()
            del indices

        # 使用自己的权重或共享的 Embedding 权重
        module = self.tied_embedding or self
        
        # 1. 本地计算 Logits 分片
        logits = F.linear(x, module.weight, self.bias)
        
        if self.tp_size == 1:
            return logits
            
        # 2. All-Gather：收集所有分片，拼成完整 Logits
        # 这里需要注意 Gather 的维度处理
        input_shape = logits.shape
        output_tensor = self._comm.all_gather(logits)

        # 特殊情况处理：bs=1 时的 view 操作
        if bs == 1:
            return output_tensor.view(1, -1)[:, : self.num_embeddings]

        # 重塑张量以匹配 [batch_size, vocab_size]
        # Gather 后通常是 [tp_size * batch_size, partial_vocab] -> 需要 readjust
        # 但这里的逻辑稍微复杂，因为 all_gather 通常沿 dim 0 拼接
        # 这里的 reshape 逻辑是为了处理分布式 gather 后的数据排列
        output_tensor = output_tensor.view((self.tp_size,) + input_shape)
        output_tensor = output_tensor.movedim(0, -1)
        output_tensor = output_tensor.reshape(input_shape[:1] + (self.tp_size * input_shape[1],))
        
        # 裁剪掉 padding 部分 (因为 Vocab Parallel 切分可能导致 pad)
        return output_tensor[:, : self.num_embeddings]
