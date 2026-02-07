
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import LlamaConfig


@lru_cache()
def _load_config(model_path: str) -> Any:
    """
    加载 Hugging Face 模型配置 (带缓存)
    
    使用 lru_cache 避免重复从磁盘读取配置文件。
    """
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_path)


def cached_load_hf_config(model_path: str) -> LlamaConfig:
    """
    加载 Hugging Face 模型配置并返回副本
    
    返回配置对象的深拷贝，防止后续修改影响缓存的原始配置。
    
    Args:
        model_path: 模型路径或 Hugging Face Hub ID
        
    Returns:
        LlamaConfig: 模型配置对象
    """
    # deep copy the config to avoid modifying the original config
    config = _load_config(model_path)
    # 通过重新实例化来实现深拷贝
    return type(config)(**config.to_dict())
