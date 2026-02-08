"""
测试 Tensor 传递内核（TVM FFI 基础测试）

本测试用于验证 Python 和 C++ 之间的 Tensor 传递机制，包括：
1. 非连续 Tensor 的传递（stride 不是标准的）
2. 跨设备 Tensor 的传递（CPU 和不同 GPU）
3. 不同数据类型的 Tensor（int32, int64）

目的：
- 验证 TVM FFI 绑定的正确性
- 确保 C++ 端能正确解析 PyTorch Tensor 的元数据
- 测试边界情况（非连续内存、跨设备）

这是一个基础测试，确保更复杂的内核（如 indexing、store_cache）
的 Tensor 参数传递是可靠的。
"""

from __future__ import annotations

from minisgl.kernel import test_tensor
from minisgl.utils import call_if_main
import torch


@call_if_main()
def main():
    """
    测试 Tensor 在 Python 和 C++ 之间的传递
    
    测试场景：
    1. 非连续 Tensor（通过切片创建）
    2. 不同设备（CPU vs GPU）
    3. 不同数据类型（int32 vs int64）
    """
    # ========================================
    # 创建测试 Tensor
    # ========================================
    # x: 非连续的 CPU Tensor
    # 先创建 [12, 2048] 的 Tensor，然后切片为 [12, 1024]
    # 切片操作不会复制数据，所以 x 是非连续的（stride 不规则）
    x = torch.empty((12, 2048), dtype=torch.int32, device="cpu")[:, :1024]
    
    # y: GPU Tensor（在 GPU 1 上）
    # 数据类型为 int64（与 x 不同）
    y = torch.empty((12, 1024), dtype=torch.int64, device="cuda:1")
    
    # ========================================
    # 调用 C++ 测试函数
    # ========================================
    # test_tensor 是一个 C++ 函数，通过 TVM FFI 绑定到 Python
    # 它会检查：
    #   1. Tensor 的 shape 是否正确传递
    #   2. Tensor 的 dtype 是否正确识别
    #   3. Tensor 的 device 是否正确识别
    #   4. 非连续 Tensor 的 stride 是否正确传递
    #   5. 数据指针是否有效
    test_tensor(x, y)
    
    # ========================================
    # 验证点（在 C++ 端）
    # ========================================
    # C++ 端的 test_tensor 函数会验证：
    # - x.shape == (12, 1024)
    # - x.dtype == int32
    # - x.device == CPU
    # - x.is_contiguous() == False
    # - y.shape == (12, 1024)
    # - y.dtype == int64
    # - y.device == CUDA:1
    # - y.is_contiguous() == True
    
    # 如果所有检查通过，函数正常返回
    # 如果有任何检查失败，C++ 端会抛出异常
