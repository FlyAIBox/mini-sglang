# 📋 Mini-SGLang 依赖检查清单

> 打印此页面,在安装时逐项检查 ✓

---

## 系统要求检查

- [ ] **操作系统**: Linux (Ubuntu 20.04+, CentOS 7+)
- [ ] **Python 版本**: 3.10, 3.11, 或 3.12
- [ ] **CUDA 版本**: 11.8+ 或 12.1+
- [ ] **NVIDIA 驱动**: 525+ (CUDA 12.x) 或 450+ (CUDA 11.x)
- [ ] **GPU**: NVIDIA GPU with Compute Capability 7.0+
- [ ] **显存**: 8GB+ (小模型) 或 24GB+ (中大模型)
- [ ] **内存**: 16GB+ 系统内存
- [ ] **存储**: 20GB+ 可用空间

---

## 核心依赖安装检查

### 深度学习框架
- [ ] **torch** (>=2.0.0, <2.6.0)
- [ ] **transformers** (>=4.56.0, <=4.57.3)
- [ ] **accelerate** (>=0.34.0)

### CUDA 和 GPU 优化
- [ ] **sgl_kernel** (>=0.3.17.post1)
- [ ] **flashinfer-python** (>=0.5.3)
- [ ] **nvidia-cutlass-dsl** (==4.3.1)
- [ ] **apache-tvm-ffi** (>=0.1.4)

### Web 服务
- [ ] **fastapi** (>=0.115.0)
- [ ] **uvicorn** (>=0.32.0)
- [ ] **openai** (>=1.57.4)

### 工具库
- [ ] **msgpack** (>=1.1.0)
- [ ] **pyzmq** (>=26.2.0)
- [ ] **prompt_toolkit** (>=3.0.36)

**核心依赖总数**: 13 个

---

## 开发依赖安装检查 (可选)

### 测试工具
- [ ] **pytest** (>=8.3.0)
- [ ] **pytest-cov** (>=6.0.0)
- [ ] **pytest-asyncio** (>=0.24.0)

### 代码质量
- [ ] **black** (>=24.8.0)
- [ ] **ruff** (>=0.11.0)
- [ ] **flake8** (>=7.1.0)
- [ ] **mypy** (>=1.13.0)
- [ ] **pre-commit** (>=4.0.0)

### 可视化
- [ ] **matplotlib** (>=3.10.5)
- [ ] **numpy** (>=1.26.0)

**开发依赖总数**: 10 个

---

## 安装步骤检查

### 1. 环境准备
- [ ] 创建虚拟环境 (venv/conda/uv)
- [ ] 激活虚拟环境
- [ ] 验证 Python 版本: `python --version`
- [ ] 验证 CUDA 可用: `nvidia-smi`

### 2. 安装 PyTorch
- [ ] 确定 CUDA 版本
- [ ] 安装对应版本的 PyTorch
- [ ] 验证: `python -c "import torch; print(torch.cuda.is_available())"`

### 3. 安装 Mini-SGLang
- [ ] 克隆仓库: `git clone https://github.com/sgl-project/mini-sglang.git`
- [ ] 进入目录: `cd mini-sglang`
- [ ] 安装: `pip install -e .` 或 `uv pip install -e .`

### 4. 验证安装
- [ ] 运行检查脚本: `python scripts/check_dependencies.py`
- [ ] 验证导入: `python -c "import minisgl; print('OK')"`
- [ ] 测试 CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

---

## 功能测试检查

### 基础功能
- [ ] 启动服务器: `python -m minisgl --model "Qwen/Qwen3-0.6B"`
- [ ] 测试 API: `curl http://localhost:8000/v1/chat/completions ...`
- [ ] 交互式 Shell: `python -m minisgl --model "Qwen/Qwen3-0.6B" --shell`

### 高级功能 (可选)
- [ ] 多 GPU 测试: `python -m minisgl --model "..." --tp 2`
- [ ] Radix Cache: `python -m minisgl --model "..." --cache radix`
- [ ] Python API: 测试 `from minisgl.llm import LLM`

---

## 常见问题检查

### CUDA 相关
- [ ] CUDA Toolkit 已安装
- [ ] 环境变量已设置 (CUDA_HOME, PATH, LD_LIBRARY_PATH)
- [ ] PyTorch CUDA 版本与系统 CUDA 版本匹配

### 内存相关
- [ ] GPU 显存足够 (使用 `nvidia-smi` 查看)
- [ ] 系统内存足够
- [ ] 磁盘空间足够 (模型缓存)

### 网络相关
- [ ] 可以访问 HuggingFace (或配置镜像)
- [ ] 可以访问 PyPI (或配置镜像)

---

## 性能优化检查 (可选)

- [ ] 使用最新的 NVIDIA 驱动
- [ ] 启用 Radix Cache
- [ ] 调整 batch size 和 max tokens
- [ ] 使用合适的 attention backend
- [ ] 多 GPU 时使用 Tensor Parallelism

---

## 文档阅读检查

- [ ] 阅读 [README.md](./README.md)
- [ ] 阅读 [DEPENDENCIES_SUMMARY.md](./DEPENDENCIES_SUMMARY.md)
- [ ] 查看 [DOCUMENTATION_INDEX.md](./DOCUMENTATION_INDEX.md)
- [ ] 遇到问题查看 [DEPENDENCIES.md](./DEPENDENCIES.md)

---

## 快速命令参考

```bash
# 检查系统
nvidia-smi                              # 查看 GPU 和 CUDA 版本
python --version                        # 查看 Python 版本
nvcc --version                          # 查看 CUDA Toolkit 版本

# 安装
pip install -e .                        # 安装核心依赖
pip install -e ".[dev]"                 # 安装包含开发依赖
uv pip install -e .                     # 使用 uv 安装 (更快)

# 验证
python scripts/check_dependencies.py   # 检查所有依赖
python -c "import minisgl"             # 验证安装
python -c "import torch; print(torch.cuda.is_available())"  # 验证 CUDA

# 运行
python -m minisgl --model "Qwen/Qwen3-0.6B"              # 启动服务器
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell      # 交互式 Shell
python -m minisgl --model "Qwen/Qwen3-14B" --tp 2        # 多 GPU
```

---

## 完成标记

安装完成后,在此签名并记录:

- **安装日期**: _______________
- **Python 版本**: _______________
- **CUDA 版本**: _______________
- **GPU 型号**: _______________
- **安装者**: _______________

**状态**: ⬜ 安装中 | ⬜ 安装完成 | ⬜ 测试通过

---

**提示**: 
- 打印此清单,在安装过程中逐项勾选
- 遇到问题时参考 [DEPENDENCIES.md](./DEPENDENCIES.md)
- 使用 `python scripts/check_dependencies.py` 自动检查

