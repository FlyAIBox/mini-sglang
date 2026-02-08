"""
Kernel 模块 - 自定义 CUDA/C++ 内核的 Python 接口

本模块提供了 Mini-SGLang 中所有自定义 CUDA/C++ 内核的 Python 绑定。
这些内核通过 TVM FFI 进行编译和加载，提供比 PyTorch 原生实现更高的性能。

主要内核类别：
1. **indexing**: Embedding/词表索引操作（gather 指定行）
2. **fast_compare_key**: Radix Tree 前缀匹配（快速比较两个序列）
3. **store_cache**: KV Cache 存储（Paged Attention 分散写入）
4. **test_tensor**: 测试用内核（验证 Tensor 传递）
5. **PyNCCL**: NCCL 通信库的 Python 包装（多 GPU 通信）

编译方式：
- **AOT (Ahead-Of-Time)**: 预先编译，适用于静态代码（如 NCCL wrapper）
- **JIT (Just-In-Time)**: 运行时编译，适用于模板特化（如根据 head_size 特化的 kernel）

依赖：
- TVM FFI: 提供 C++/Python 互操作和 JIT 编译能力
- CUDA Toolkit: 编译 CUDA 内核
- NCCL: 多 GPU 通信库
"""

from .index import indexing
from .pynccl import PyNCCLCommunicator, init_pynccl
from .radix import fast_compare_key
from .store import store_cache
from .tensor import test_tensor

__all__ = [
    "indexing",            # Embedding/词表索引内核
    "fast_compare_key",    # Radix Tree 前缀比较内核
    "store_cache",         # KV Cache 存储内核
    "test_tensor",         # 测试内核
    "init_pynccl",         # 初始化 NCCL 通信器
    "PyNCCLCommunicator",  # NCCL 通信器类型
]
