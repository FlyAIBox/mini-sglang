# Mini-SGLang 依赖说明

本文档详细说明 Mini-SGLang 项目的所有依赖包及其版本要求。

---

## 📦 核心依赖

### 深度学习框架

| 包名 | 版本要求 | 说明 |
|------|---------|------|
| **torch** | >=2.0.0, <2.6.0 | PyTorch 深度学习框架,用于模型加载和推理 |
| **transformers** | >=4.56.0, <=4.57.3 | HuggingFace Transformers,支持各种预训练模型 |
| **accelerate** | >=0.34.0 | HuggingFace 加速库,用于分布式训练和推理优化 |

### CUDA 和 GPU 优化

| 包名 | 版本要求 | 说明 |
|------|---------|------|
| **sgl_kernel** | >=0.3.17.post1 | SGLang 自定义 CUDA 内核,提供高性能算子 |
| **flashinfer-python** | >=0.5.3 | FlashInfer 注意力机制优化内核 |
| **nvidia-cutlass-dsl** | ==4.3.1 | NVIDIA CUTLASS 模板库 DSL |
| **apache-tvm-ffi** | >=0.1.4 | Apache TVM 外部函数接口 |

**重要说明**:
- 这些包需要 CUDA 11.8+ 或 12.1+ 环境
- 某些包会在首次运行时进行 JIT 编译
- 需要安装 NVIDIA CUDA Toolkit

### Web 服务框架

| 包名 | 版本要求 | 说明 |
|------|---------|------|
| **fastapi** | >=0.115.0 | 现代高性能 Web 框架,用于构建 API 服务 |
| **uvicorn** | >=0.32.0 | ASGI 服务器,运行 FastAPI 应用 |
| **openai** | >=1.57.4 | OpenAI Python 客户端,兼容 OpenAI API 格式 |

### 数据序列化和通信

| 包名 | 版本要求 | 说明 |
|------|---------|------|
| **msgpack** | >=1.1.0 | MessagePack 序列化库,用于高效数据传输 |
| **pyzmq** | >=26.2.0 | ZeroMQ Python 绑定,用于进程间通信 |

### 命令行界面

| 包名 | 版本要求 | 说明 |
|------|---------|------|
| **prompt_toolkit** | >=3.0.36 | 交互式命令行界面库,用于 Shell 模式 |

---

## 🛠️ 开发依赖

### 测试工具

| 包名 | 版本要求 | 说明 |
|------|---------|------|
| **pytest** | >=8.3.0 | Python 单元测试框架 |
| **pytest-cov** | >=6.0.0 | Pytest 测试覆盖率插件 |
| **pytest-asyncio** | >=0.24.0 | Pytest 异步测试支持 |

### 代码质量和格式化

| 包名 | 版本要求 | 说明 |
|------|---------|------|
| **black** | >=24.8.0 | Python 代码格式化工具 |
| **ruff** | >=0.11.0 | 极快的 Python linter 和格式化工具 |
| **flake8** | >=7.1.0 | Python 代码风格检查工具 |
| **mypy** | >=1.13.0 | Python 静态类型检查工具 |
| **pre-commit** | >=4.0.0 | Git 预提交钩子框架 |

### 性能分析和可视化

| 包名 | 版本要求 | 说明 |
|------|---------|------|
| **matplotlib** | >=3.10.5 | Python 绘图库,用于性能可视化 |
| **numpy** | >=1.26.0 | 数值计算库 |

---

## 📋 安装方式

### 1. 使用 pip 安装 (推荐)

```bash
# 安装核心依赖
pip install -e .

# 安装包含开发依赖
pip install -e ".[dev]"
```

### 2. 使用 uv 安装 (更快)

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装核心依赖
uv pip install -e .

# 安装包含开发依赖
uv pip install -e ".[dev]"
```

### 3. 使用 requirements.txt

```bash
# 仅核心依赖
pip install -r requirements.txt

# 包含开发依赖
pip install -r requirements-dev.txt
```

---

## 🔧 系统要求

### Python 版本
- **支持**: Python 3.10, 3.11, 3.12
- **推荐**: Python 3.12

### CUDA 版本
- **支持**: CUDA 11.8+ 或 CUDA 12.1+
- **推荐**: CUDA 12.1

### 操作系统
- **支持**: Linux (Ubuntu 20.04+, CentOS 7+)
- **注意**: Windows 用户需要使用 WSL2,macOS 不支持

---

## ⚠️ 常见问题

### 1. PyTorch CUDA 版本不匹配

如果遇到 CUDA 版本不匹配的错误:

```bash
# 查看系统 CUDA 版本
nvidia-smi

# 安装匹配的 PyTorch
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 2. flashinfer 或 sgl_kernel 安装失败

确保已安装 CUDA Toolkit:

```bash
# 检查 CUDA Toolkit
nvcc --version

# 设置环境变量
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 3. 内存不足

安装时如果遇到内存不足:

```bash
# 减少并行编译任务
MAX_JOBS=4 pip install -e .

# 或禁用缓存
pip install -e . --no-cache-dir
```

---

## 📊 依赖关系图

```
Mini-SGLang
├── 深度学习框架
│   ├── torch (核心)
│   ├── transformers (模型)
│   └── accelerate (加速)
│
├── CUDA 优化
│   ├── sgl_kernel (自定义内核)
│   ├── flashinfer-python (注意力优化)
│   ├── nvidia-cutlass-dsl (模板库)
│   └── apache-tvm-ffi (编译器)
│
├── Web 服务
│   ├── fastapi (API 框架)
│   ├── uvicorn (服务器)
│   └── openai (客户端)
│
└── 工具库
    ├── msgpack (序列化)
    ├── pyzmq (通信)
    └── prompt_toolkit (CLI)
```

---

## 🔄 更新日志

### 2026-01-26
- ✅ 补充所有依赖的具体版本号
- ✅ 添加详细的依赖说明和分类
- ✅ 更新开发依赖到最新稳定版本
- ✅ 添加安装和故障排除指南

---

## 📝 维护说明

### 版本更新原则

1. **核心依赖**: 谨慎更新,需要充分测试
2. **开发依赖**: 可以更新到最新稳定版
3. **CUDA 相关**: 与 PyTorch 版本保持兼容
4. **API 框架**: 注意破坏性变更

### 依赖检查

```bash
# 检查过期的包
pip list --outdated

# 安全漏洞检查
pip-audit

# 依赖树查看
pipdeptree
```

---

如有问题,请提交 Issue 或查看 [安装指南](./INSTALL_GUIDE_ZH.md)。

