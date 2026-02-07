
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from minisgl.core import Req


@dataclass
class BaseCaptureData:
    """CUDA Graph Capture 数据的基类"""
    input_ids: torch.Tensor
    seq_lens: torch.Tensor
    positions: torch.Tensor
    cu_seqlens_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    page_table: torch.Tensor
    out_loc: torch.Tensor

    @classmethod
    def create(cls, max_bs: int, max_seq_len: int, device: torch.device, **kwargs):
        """
        创建并初始化 Capture Data
        
        分配所有必要的 Tensors，这些 Tensors 的内存地址在 Capture 后是固定的。
        """
        return cls(
            input_ids=torch.zeros((max_bs,), dtype=torch.int32, device=device),
            seq_lens=torch.ones((max_bs,), dtype=torch.int32, device=device),
            positions=torch.zeros((max_bs,), dtype=torch.int32, device=device),
            cu_seqlens_k=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            cu_seqlens_q=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            page_table=torch.zeros((max_bs, max_seq_len), dtype=torch.int32, device=device),
            out_loc=torch.zeros((max_bs,), dtype=torch.int32, device=device),
            **kwargs,
        )


def make_positions(device: torch.device, reqs: List[Req]) -> torch.Tensor:
    """
    生成 Position IDs (用于 RoPE)
    
    为当前 batch 中的每个 token 生成对应的位置索引。
    
    逻辑：
    - 对于 Prefill 请求：生成 [0, 1, ..., extend_len-1] (如果 cached_len=0)
      或者 [cached_len, cached_len+1, ..., device_len-1]
    - 对于 Decode 请求：生成 [device_len-1] (只包含最后一个位置)
    
    此函数在 CPU 上构建索引，然后异步拷贝到 GPU。
    """
    needed_size = sum(req.extend_len for req in reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)
