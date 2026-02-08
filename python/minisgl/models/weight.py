from __future__ import annotations

import glob
import os
from typing import Dict

import safetensors
import torch
from huggingface_hub import snapshot_download
from minisgl.distributed import get_tp_info
from minisgl.utils import divide_up
from tqdm.asyncio import tqdm


class DisabledTqdm(tqdm):
    """
    禁用进度条的 tqdm 包装类。
    用于在不需要显示进度条的场合（如日志已被重定向）替代标准 tqdm。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)


def _shard_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    根据张量并行 (TP) 策略对权重进行切分。
    
    在多 GPU 环境下，每个 GPU 只需加载部分权重。此函数根据张量名称
    将权重切分并只保留当前 Rank 需要的部分。
    
    切分规则:
    1. 列并行 (Column Parallel): 权重在 dim 0 切分 (如 q_proj, k_proj, v_proj, gate_proj, up_proj)
    2. 行并行 (Row Parallel): 权重在 dim 1 切分 (如 o_proj, down_proj)
    3. 词表并行 (Vocab Parallel): lm_head 和 embed_tokens 在 dim 0 切分
    
    Args:
        state_dict: 完整的权重字典 (加载自文件)
        
    Returns:
        Dict[str, torch.Tensor]: 切分后的权重字典，只包含当前 Rank 的数据。
    """
    shard_state_dict: Dict[str, torch.Tensor] = {}
    tp_info = get_tp_info()
    r = tp_info.rank
    n = tp_info.size
    
    # 需要在 dim 0 切分的层 (通常是 LinearColParallel)
    SPLIT_DIM_0_LIST = [
        ".q_proj",
        ".k_proj",
        ".v_proj",
        ".gate_proj",
        ".up_proj",
    ]
    # 需要在 dim 1 切分的层 (通常是 LinearRowParallel)
    SPLIT_DIM_1_LIST = [
        ".o_proj",
        ".down_proj",
    ]
    
    for key, value in state_dict.items():
        if any(key.count(sub) for sub in SPLIT_DIM_0_LIST):
            # Column Parallel: 切分 dim 0，取第 r 份
            shard_state_dict[key] = value.chunk(n, dim=0)[r]
        elif any(key.count(sub) for sub in SPLIT_DIM_1_LIST):
            # Row Parallel: 切分 dim 1，取第 r 份
            shard_state_dict[key] = value.chunk(n, dim=1)[r]
        elif key.count("lm_head") or key.count("embed_tokens"):
            # Vocab Parallel: 切分词表维度 (dim 0)
            # 注意：词表大小可能不能被 TP size 整除，所以不能直接用 chunk
            num_embeddings = value.shape[0]
            num_embeddings_per_partition = divide_up(num_embeddings, n)
            vocab_start_idx = r * num_embeddings_per_partition
            vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
            shard_state_dict[key] = value[vocab_start_idx:vocab_end_idx, :]
        else:
            # 其他参数 (如 Norm, Bias 等) 不切分，每个 Rank 都持有完整副本
            shard_state_dict[key] = value
            
    return shard_state_dict


def _merge_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    合并权重以符合 Mini-SGLang 的内部格式。
    
    Mine-SGLang 使用合并的 Linear 层来提高效率：
    1. q_proj, k_proj, v_proj -> qkv_proj
    2. gate_proj, up_proj -> gate_up_proj
    
    此函数在加载权重时执行这种合并。
    
    Args:
        state_dict: 原始权重字典
        
    Returns:
        Dict[str, torch.Tensor]: 合并后的权重字典
    """
    filtered_state_dict: Dict[str, torch.Tensor] = {}
    for key in list(state_dict.keys()):
        if key.count(".q_proj"):
            # 合并 QKV
            q_proj = state_dict[key]
            k_proj = state_dict[key.replace(".q_proj", ".k_proj")]
            v_proj = state_dict[key.replace(".q_proj", ".v_proj")]
            new_key = key.replace(".q_proj", ".qkv_proj")
            # 在 dim 0 拼接
            filtered_state_dict[new_key] = torch.cat([q_proj, k_proj, v_proj], dim=0)
            # 删除原始 key
            del state_dict[key]
            del state_dict[key.replace(".q_proj", ".k_proj")]
            del state_dict[key.replace(".q_proj", ".v_proj")]
        elif key.count(".gate_proj"):
            # 合并 MLP Gate/Up
            gate_proj = state_dict[key]
            up_proj = state_dict[key.replace(".gate_proj", ".up_proj")]
            new_key = key.replace(".gate_proj", ".gate_up_proj")
            filtered_state_dict[new_key] = torch.cat([gate_proj, up_proj], dim=0)
            del state_dict[key]
            del state_dict[key.replace(".gate_proj", ".up_proj")]
        elif key.count(".k_proj") or key.count(".v_proj") or key.count("up_proj"):
            # 这些 key 已经被合并处理过了，跳过
            continue
        else:
            # 其他 key 直接保留
            filtered_state_dict[key] = state_dict[key]
    return filtered_state_dict


def load_hf_weight(model_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    加载 HuggingFace 格式的模型权重。
    
    支持从本地目录或 HuggingFace Hub 加载。
    会自动处理 safetensors 格式，根据 TP 设置切分权重，并合并 QKV/GateUp 层。
    
    Args:
        model_path: 本地路径或 HuggingFace Repo ID
        device: 目标设备 (CPU/GPU)
        
    Returns:
        Dict[str, torch.Tensor]: 处理好的权重字典
    """
    # 1. 确定模型文件夹路径
    if os.path.isdir(model_path):
        hf_folder = model_path
    else:
        # 尝试从 HF Hub 下载 (只下载 safetensors)
        try:
            hf_folder = snapshot_download(
                model_path,
                allow_patterns=["*.safetensors"],
                tqdm_class=DisabledTqdm,
            )
        except Exception:
            raise ValueError(
                f"Model path '{model_path}' is neither a local directory nor a valid HuggingFace repository ID"
            )

    # 2. 查找并加载 safetensors 文件
    files = glob.glob(f"{hf_folder}/*.safetensors")
    state_dict: Dict[str, torch.Tensor] = {}
    for file in sorted(files):
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            for name in f.keys():
                state_dict[name] = f.get_tensor(name)

    # 3. 如果开启了 TP，进行权重切分
    if get_tp_info().size > 1:
        state_dict = _shard_state_dict(state_dict)

    # 4. 移动到目标设备
    state_dict = {k: v.to(device) for k, v in state_dict.items()}
    
    # 5. 合并 QKV/MLP 权重
    return _merge_state_dict(state_dict)

