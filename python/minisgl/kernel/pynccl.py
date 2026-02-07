
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal

from minisgl.env import ENV

from .utils import load_aot

if TYPE_CHECKING:
    from abc import abstractmethod

    import torch
    from tvm_ffi import Module

    class PyNCCLCommunicator:
        """
        PyNCCL 通信器接口定义
        
        这是 C++ NCCL Wrapper 的 Python 类型提示接口。
        实际实现是通过 TVM FFI 加载的 C++ 对象。
        """
        @abstractmethod
        def all_reduce(self, input: torch.Tensor, op: Literal["sum"]) -> None: ...
        @abstractmethod
        def all_gather(self, output: torch.Tensor, input: torch.Tensor) -> None: ...
        @abstractmethod
        def get_buffer(self) -> int: ...

else:
    PyNCCLCommunicator = Any


@lru_cache(maxsize=None)
def _load_nccl_module() -> Module:
    """加载 NCCL C++ 模块 (AOT 编译)"""
    return load_aot("pynccl", cuda_files=["pynccl.cu"], extra_ldflags=["-lnccl"])


@lru_cache(maxsize=None)
def _get_pynccl_wrapper_cls():
    """获取 PyNCCLWrapper 类 (通过 TVM FFI 注册)"""
    import tvm_ffi

    @tvm_ffi.register_object("minisgl.NCCLWrapper")
    class PyNCCLImpl(tvm_ffi.Object):
        def __init__(self, *args):
            self.__ffi_init__(*args)

    return PyNCCLImpl


def init_pynccl(
    *,
    tp_rank: int,
    tp_size: int,
    tp_cpu_group: torch.distributed.ProcessGroup,
    max_size_bytes: int = 0,
) -> PyNCCLCommunicator:
    """
    初始化 PyNCCL 通信器
    
    使用 PyTorch 分布式组 (Gloo/NCCL) 交换 NCCL Unique ID，
    然后在每个进程上创建 NCCL Communicator。
    """
    import torch

    max_size_bytes = min(max_size_bytes, ENV.PYNCCL_MAX_BUFFER_SIZE.value)

    module = _load_nccl_module()
    cls = _get_pynccl_wrapper_cls()

    if tp_rank == 0:
        # Rank 0 创建 NCCL ID 并广播给其他 Rank
        id_list = [module.create_nccl_uid()]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )
    else:
        # 其他 Rank 接收 NCCL ID
        id_list = [None]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )

    nccl_id = id_list[0]
    assert not nccl_id is None, f"Failed to get NCCL unique ID on {tp_rank = }"

    # bypass type checking for the FFI object
    return cls(tp_rank, tp_size, max_size_bytes, nccl_id)  # type: ignore
