#!/usr/bin/env python3
"""
Mini-SGLang 依赖检查脚本

用途: 验证所有核心依赖是否正确安装
运行: python scripts/check_dependencies.py
"""

import sys
from importlib import import_module
from typing import Dict, List, Tuple

# 定义依赖检查列表
CORE_DEPENDENCIES = {
    "torch": "PyTorch 深度学习框架",
    "transformers": "HuggingFace Transformers",
    "accelerate": "分布式推理加速",
    "sgl_kernel": "SGLang 自定义 CUDA 内核",
    "flashinfer": "FlashInfer 注意力优化",
    "fastapi": "FastAPI Web 框架",
    "uvicorn": "ASGI 服务器",
    "openai": "OpenAI API 客户端",
    "msgpack": "MessagePack 序列化",
    "zmq": "ZeroMQ Python 绑定",
    "prompt_toolkit": "交互式命令行界面",
}

DEV_DEPENDENCIES = {
    "pytest": "单元测试框架",
    "black": "代码格式化工具",
    "ruff": "Python Linter",
    "mypy": "静态类型检查",
    "matplotlib": "绘图和可视化",
}


def check_package(package_name: str, description: str) -> Tuple[bool, str]:
    """检查单个包是否已安装"""
    try:
        module = import_module(package_name)
        version = getattr(module, "__version__", "未知版本")
        return True, version
    except ImportError as e:
        return False, str(e)


def print_header(text: str, char: str = "=") -> None:
    """打印格式化的标题"""
    print(f"\n{char * 60}")
    print(f"  {text}")
    print(f"{char * 60}\n")


def check_cuda() -> None:
    """检查 CUDA 可用性"""
    try:
        import torch

        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {'✅ 是' if torch.cuda.is_available() else '❌ 否'}")

        if torch.cuda.is_available():
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠️  警告: CUDA 不可用,某些功能将无法使用")
    except Exception as e:
        print(f"❌ 检查 CUDA 时出错: {e}")


def main() -> int:
    """主函数"""
    print_header("Mini-SGLang 依赖检查", "=")

    # 检查核心依赖
    print_header("核心依赖检查", "-")
    core_results: Dict[str, Tuple[bool, str]] = {}
    for package, description in CORE_DEPENDENCIES.items():
        status, info = check_package(package, description)
        core_results[package] = (status, info)

        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {package:20s} - {description}")
        if status:
            print(f"   版本: {info}")
        else:
            print(f"   错误: {info}")

    # 检查 CUDA
    print_header("CUDA 环境检查", "-")
    check_cuda()

    # 检查开发依赖 (可选)
    print_header("开发依赖检查 (可选)", "-")
    dev_results: Dict[str, Tuple[bool, str]] = {}
    for package, description in DEV_DEPENDENCIES.items():
        status, info = check_package(package, description)
        dev_results[package] = (status, info)

        status_icon = "✅" if status else "⚠️ "
        print(f"{status_icon} {package:20s} - {description}")
        if status:
            print(f"   版本: {info}")

    # 统计结果
    print_header("检查结果汇总", "=")
    core_installed = sum(1 for status, _ in core_results.values() if status)
    core_total = len(CORE_DEPENDENCIES)
    dev_installed = sum(1 for status, _ in dev_results.values() if status)
    dev_total = len(DEV_DEPENDENCIES)

    print(f"核心依赖: {core_installed}/{core_total} 已安装")
    print(f"开发依赖: {dev_installed}/{dev_total} 已安装")

    if core_installed == core_total:
        print("\n✅ 所有核心依赖已正确安装!")
        print("   你可以开始使用 Mini-SGLang 了。")
        return 0
    else:
        print("\n❌ 部分核心依赖缺失!")
        print("   请运行以下命令安装:")
        print("   pip install -e .")
        return 1


if __name__ == "__main__":
    sys.exit(main())

