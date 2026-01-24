"""
分布式通信模块

本模块提供Mini-SGLang中张量并行（Tensor Parallelism）所需的全部功能：

1. 分布式信息管理（info.py）:
   - DistributedInfo: 存储rank和size
   - set_tp_info/get_tp_info: 管理全局TP配置

2. 通信实现（impl.py）:
   - All-Reduce: 张量求和并广播
   - All-Gather: 张量收集并拼接
   - 支持TorchDistributed和PyNCCL两种后端

使用示例:
    # 1. 设置TP信息（每个进程启动时）
    from minisgl.distributed import set_tp_info
    set_tp_info(rank=0, size=4)
    
    # 2. 使用通信原语
    from minisgl.distributed import DistributedCommunicator
    comm = DistributedCommunicator()
    result = comm.all_reduce(tensor)
    
    # 3. 启用PyNCCL获得更好性能（可选）
    from minisgl.distributed import enable_pynccl_distributed
    enable_pynccl_distributed(tp_info, cpu_group, max_bytes)
"""

from .impl import DistributedCommunicator, destroy_distributed, enable_pynccl_distributed
from .info import DistributedInfo, get_tp_info, set_tp_info, try_get_tp_info

__all__ = [
    # 分布式信息
    "DistributedInfo",
    "get_tp_info",
    "set_tp_info",
    "try_get_tp_info",
    # 通信接口
    "DistributedCommunicator",
    "enable_pynccl_distributed",
    "destroy_distributed",
]
