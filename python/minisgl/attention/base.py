"""
Mini-SGLang 注意力后端基类模块

定义注意力计算的统一接口。Mini-SGLang支持多种注意力后端：
- FlashAttention: 适合prefill阶段，内存高效的注意力实现
- FlashInfer: 针对decode阶段优化，支持PagedAttention
- Hybrid: 组合使用两种后端，prefill用FlashAttention，decode用FlashInfer

设计理念：
- 抽象接口：所有后端实现相同的接口
- 元数据分离：注意力计算所需的索引、偏移等信息封装在metadata中
- CUDA Graph支持：提供capture和replay接口
- 灵活切换：可以在不同阶段使用不同的后端

核心概念：
- Metadata: 存储注意力计算的辅助信息（positions、block table等）
- Prepare: 在forward前准备metadata
- Capture/Replay: 支持CUDA Graph优化
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import torch
    from minisgl.core import Batch


@dataclass
class BaseAttnMetadata(ABC):
    """
    注意力元数据基类
    
    存储注意力计算所需的辅助信息。不同的注意力后端
    有不同的metadata实现。
    
    必选属性:
        positions: token在序列中的位置索引，用于RoPE等位置编码
    
    子类可能包含的额外信息:
        - seq_lens: 每个序列的长度
        - block_table: PagedAttention的page映射表
        - cu_seqlens: 累积序列长度（用于FlashAttention）
        - max_seq_len: 批次中最长序列的长度
        等等
    """
    positions: torch.Tensor
    """Token位置索引，shape为[total_tokens]，值为每个token在序列中的位置（0-based）"""

    @abstractmethod
    def get_last_indices(self, bs: int) -> torch.Tensor:
        """
        获取每个序列最后一个token的索引
        
        在decode阶段，我们只需要最后一个token的输出，
        此方法返回这些token在batch中的位置。
        
        参数:
            bs: 批次大小（实际序列数量）
        
        返回:
            torch.Tensor: 最后一个token的索引，shape为[bs]
        
        示例:
            假设batch中有3个序列：
            - 序列0: 5个token (索引0-4)
            - 序列1: 3个token (索引5-7)
            - 序列2: 4个token (索引8-11)
            
            get_last_indices(3) → tensor([4, 7, 11])
        """
        ...


class BaseAttnBackend(ABC):
    """
    注意力后端抽象基类
    
    定义所有注意力后端必须实现的接口。
    
    核心方法:
        - forward(): 执行注意力计算
        - prepare_metadata(): 准备元数据
        - init_capture_graph(): 初始化CUDA Graph
        - prepare_for_capture(): 为capture准备
        - prepare_for_replay(): 为replay准备
    
    具体实现:
        - FlashAttnBackend (attention/fa.py): 使用FlashAttention
        - FlashInferBackend (attention/fi.py): 使用FlashInfer
        - HybridBackend (本文件): 组合使用两种后端
    """
    
    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        """
        执行注意力计算
        
        计算 softmax(Q @ K^T / sqrt(d)) @ V，并处理KV Cache。
        
        参数:
            q: Query张量，shape为[total_tokens, num_heads, head_dim]
            k: Key张量，shape为[total_tokens, num_kv_heads, head_dim]
            v: Value张量，shape为[total_tokens, num_kv_heads, head_dim]
            layer_id: 当前层的ID（用于索引KV Cache）
            batch: 当前批次信息
        
        返回:
            torch.Tensor: 注意力输出，shape为[total_tokens, num_heads, head_dim]
        
        工作流程:
            1. 根据batch.phase决定是prefill还是decode
            2. 如果是prefill：
               - 计算Q @ K^T（可能包括当前层的K和之前的KV Cache）
               - 存储新的K、V到KV Cache
            3. 如果是decode：
               - 从KV Cache读取所有历史K、V
               - 添加当前token的K、V
               - 计算注意力
            4. 应用causal mask（prefill时）
            5. 返回注意力输出
        """
        ...

    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None:
        """
        准备注意力元数据
        
        在forward之前调用，根据batch信息构建metadata。
        Metadata会被存储到batch.attn_metadata中供forward使用。
        
        参数:
            batch: 要处理的批次
        
        工作内容:
            - 计算序列长度、token位置等信息
            - 构建block table（如果使用PagedAttention）
            - 计算各种索引和偏移
            - 创建特定后端需要的数据结构
        """
        ...

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        """
        初始化CUDA Graph相关的资源
        
        在Engine初始化时调用一次，为不同batch size的CUDA Graph
        分配和初始化所需的内存和数据结构。
        
        参数:
            max_seq_len: 支持的最大序列长度
            bs_list: 要capture的batch size列表（如[1, 2, 4, 8, ...]）
        
        工作内容:
            - 分配workspace内存
            - 预创建索引张量
            - 初始化后端特定的数据结构
        """
        ...

    @abstractmethod
    def prepare_for_capture(self, batch: Batch) -> None:
        """
        为CUDA Graph capture准备batch
        
        在capture CUDA Graph之前调用，确保metadata使用
        固定的内存地址和数据结构。
        
        参数:
            batch: 要capture的batch
        
        注意事项:
            - Capture时所有内存地址必须固定
            - 不能使用动态大小的张量
            - 需要使用预分配的workspace
        """
        ...

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None:
        """
        为CUDA Graph replay准备batch
        
        在replay CUDA Graph之前调用，更新metadata中
        可变的部分（如序列长度、token数量等）。
        
        参数:
            batch: 要replay的batch
        
        工作内容:
            - 更新序列长度等动态信息
            - 更新block table
            - 保持内存地址不变
        """
        ...


class HybridBackend(BaseAttnBackend):
    """
    混合注意力后端
    
    根据batch的phase（prefill/decode）自动选择合适的后端。
    这是Mini-SGLang默认使用的注意力后端。
    
    设计理念:
        - Prefill阶段：使用FlashAttention
          - 处理长序列，计算密集型
          - FlashAttention在长序列上性能好
        
        - Decode阶段：使用FlashInfer
          - 每次只处理1个token，访存密集型
          - FlashInfer针对decode优化，支持PagedAttention
    
    属性:
        prefill_backend: Prefill阶段使用的后端（通常是FlashAttention）
        decode_backend: Decode阶段使用的后端（通常是FlashInfer）
    
    使用示例：
        prefill = FlashAttnBackend(...)
        decode = FlashInferBackend(...)
        hybrid = HybridBackend(prefill, decode)
        
        # 自动根据batch.phase选择后端
        output = hybrid.forward(q, k, v, layer_id, batch)
    
    性能特点:
        - Prefill: FlashAttention高效处理长序列
        - Decode: FlashInfer的PagedAttention减少内存碎片
        - 最佳组合：充分利用两种后端的优势
    """
    
    def __init__(
        self,
        prefill_backend: BaseAttnBackend,
        decode_backend: BaseAttnBackend,
    ) -> None:
        """
        初始化混合后端
        
        参数:
            prefill_backend: Prefill阶段的注意力后端
            decode_backend: Decode阶段的注意力后端
        """
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        """
        执行注意力计算，自动选择后端
        
        根据batch.is_prefill选择对应的后端执行计算。
        
        参数:
            q: Query张量
            k: Key张量
            v: Value张量
            layer_id: 层ID
            batch: 当前批次
        
        返回:
            注意力输出张量
        """
        # 根据阶段选择后端
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.forward(q, k, v, layer_id, batch)

    def prepare_metadata(self, batch: Batch) -> None:
        """
        准备元数据，自动选择后端
        
        参数:
            batch: 当前批次
        """
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.prepare_metadata(batch)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        """
        初始化CUDA Graph
        
        只为decode后端初始化，因为：
        - Decode阶段batch size相对固定，适合CUDA Graph
        - Prefill阶段输入长度变化大，不适合CUDA Graph
        
        参数:
            max_seq_len: 最大序列长度
            bs_list: batch size列表
        """
        self.decode_backend.init_capture_graph(max_seq_len, bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        """
        为CUDA Graph capture准备（只用于decode）
        
        参数:
            batch: 要capture的batch
        """
        self.decode_backend.prepare_for_capture(batch)

    def prepare_for_replay(self, batch: Batch) -> None:
        """
        为CUDA Graph replay准备（只用于decode）
        
        参数:
            batch: 要replay的batch
        """
        self.decode_backend.prepare_for_replay(batch)
