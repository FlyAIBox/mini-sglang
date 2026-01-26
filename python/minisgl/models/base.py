"""
Mini-SGLang 模型基类模块

定义所有大语言模型的通用接口。所有具体的模型实现（如Llama、Qwen3等）
都必须继承自BaseLLMModel并实现其抽象方法。

设计理念：
- 统一接口：所有模型使用相同的forward签名
- 继承BaseOP：获得state_dict/load_state_dict能力
- 抽象基类：强制子类实现必要的方法
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from minisgl.layers import BaseOP

if TYPE_CHECKING:
    import torch


class BaseLLMModel(ABC, BaseOP):
    """
    大语言模型基类
    
    所有LLM模型的抽象基类，定义模型必须实现的接口。
    
    继承关系：
        BaseLLMModel → ABC（抽象基类）+ BaseOP（支持权重操作）
    
    子类必须实现：
        - forward(): 模型的前向传播逻辑
    
    子类示例：
        - LlamaModel (python/minisgl/models/llama.py)
        - Qwen3Model (python/minisgl/models/qwen3.py)
    
    使用示例：
        # 创建模型实例（实际使用工厂函数）
        model = create_model("Qwen/Qwen3-0.6B", model_config)
        
        # 加载权重
        state_dict = load_hf_weight(...)
        model.load_state_dict(state_dict)
        
        # 执行推理
        output = model.forward()  # 使用当前batch的信息
    
    注意事项：
        - forward()不接受显式参数，而是从全局Context中获取batch信息
        - 这种设计简化了调用接口，避免在每层传递大量参数
        - 模型内部通过get_global_ctx()获取当前batch和输入数据
    """
    
    @abstractmethod
    def forward(self) -> torch.Tensor:
        """
        模型前向传播
        
        执行一次完整的模型推理，从输入token生成logits。
        
        返回:
            torch.Tensor: 输出logits，shape为[batch_size, vocab_size]
                - batch_size: 当前batch中的token数量
                  - Prefill阶段：可能是多个token（所有请求的新增token）
                  - Decode阶段：通常等于请求数量（每个请求一个token）
                - vocab_size: 词表大小
        
        工作流程：
            1. 从Context获取当前batch和输入数据
            2. 通过Embedding层转换token到向量
            3. 经过多层Transformer处理
            4. 通过输出层生成logits
            5. 返回logits供采样器选择下一个token
        
        注意：
            - 不接受显式参数，所有信息从全局Context获取
            - KV Cache的读写由注意力层自动处理
            - 在Tensor Parallelism模式下，各GPU计算部分参数，最后All-Reduce聚合
        """
        ...
