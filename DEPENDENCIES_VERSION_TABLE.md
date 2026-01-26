# 依赖版本对照表

> 快速查看所有依赖的版本要求 | 更新时间: 2026-01-26

## 📦 生产环境依赖

| # | 包名 | 版本要求 | PyPI 最新版本 | 安装状态 |
|---|------|---------|--------------|---------|
| 1 | torch | >=2.0.0, <2.6.0 | [查看](https://pypi.org/project/torch/) | ⬜ |
| 2 | transformers | >=4.56.0, <=4.57.3 | [查看](https://pypi.org/project/transformers/) | ⬜ |
| 3 | accelerate | >=0.34.0 | [查看](https://pypi.org/project/accelerate/) | ⬜ |
| 4 | sgl_kernel | >=0.3.17.post1 | [查看](https://pypi.org/project/sgl-kernel/) | ⬜ |
| 5 | flashinfer-python | >=0.5.3 | [查看](https://pypi.org/project/flashinfer/) | ⬜ |
| 6 | nvidia-cutlass-dsl | ==4.3.1 | [查看](https://pypi.org/project/nvidia-cutlass-dsl/) | ⬜ |
| 7 | apache-tvm-ffi | >=0.1.4 | [查看](https://pypi.org/project/apache-tvm-ffi/) | ⬜ |
| 8 | fastapi | >=0.115.0 | [查看](https://pypi.org/project/fastapi/) | ⬜ |
| 9 | uvicorn | >=0.32.0 | [查看](https://pypi.org/project/uvicorn/) | ⬜ |
| 10 | openai | >=1.57.4 | [查看](https://pypi.org/project/openai/) | ⬜ |
| 11 | msgpack | >=1.1.0 | [查看](https://pypi.org/project/msgpack/) | ⬜ |
| 12 | pyzmq | >=26.2.0 | [查看](https://pypi.org/project/pyzmq/) | ⬜ |
| 13 | prompt_toolkit | >=3.0.36 | [查看](https://pypi.org/project/prompt-toolkit/) | ⬜ |

## 🛠️ 开发环境依赖

| # | 包名 | 版本要求 | PyPI 最新版本 | 安装状态 |
|---|------|---------|--------------|---------|
| 1 | pytest | >=8.3.0 | [查看](https://pypi.org/project/pytest/) | ⬜ |
| 2 | pytest-cov | >=6.0.0 | [查看](https://pypi.org/project/pytest-cov/) | ⬜ |
| 3 | pytest-asyncio | >=0.24.0 | [查看](https://pypi.org/project/pytest-asyncio/) | ⬜ |
| 4 | black | >=24.8.0 | [查看](https://pypi.org/project/black/) | ⬜ |
| 5 | ruff | >=0.11.0 | [查看](https://pypi.org/project/ruff/) | ⬜ |
| 6 | flake8 | >=7.1.0 | [查看](https://pypi.org/project/flake8/) | ⬜ |
| 7 | mypy | >=1.13.0 | [查看](https://pypi.org/project/mypy/) | ⬜ |
| 8 | pre-commit | >=4.0.0 | [查看](https://pypi.org/project/pre-commit/) | ⬜ |
| 9 | matplotlib | >=3.10.5 | [查看](https://pypi.org/project/matplotlib/) | ⬜ |
| 10 | numpy | >=1.26.0 | [查看](https://pypi.org/project/numpy/) | ⬜ |

## 📊 版本兼容性矩阵

### Python 版本兼容性

| Python | torch | transformers | fastapi | 状态 |
|--------|-------|--------------|---------|------|
| 3.10 | ✅ | ✅ | ✅ | 支持 |
| 3.11 | ✅ | ✅ | ✅ | 支持 |
| 3.12 | ✅ | ✅ | ✅ | 推荐 ⭐ |

### CUDA 版本兼容性

| CUDA | torch | flashinfer | sgl_kernel | 状态 |
|------|-------|------------|------------|------|
| 11.8 | ✅ | ✅ | ✅ | 支持 |
| 12.1 | ✅ | ✅ | ✅ | 推荐 ⭐ |
| 12.4 | ✅ | ✅ | ✅ | 支持 |

### 操作系统兼容性

| OS | 状态 | 说明 |
|----|------|------|
| Ubuntu 20.04+ | ✅ 支持 | 推荐 |
| Ubuntu 22.04+ | ✅ 支持 | 推荐 ⭐ |
| CentOS 7+ | ✅ 支持 | - |
| Debian 11+ | ✅ 支持 | - |
| RHEL 8+ | ✅ 支持 | - |
| Windows (WSL2) | ⚠️ 有限支持 | 需要 WSL2 + CUDA |
| macOS | ❌ 不支持 | 缺少 CUDA 支持 |

## 🔍 检查安装状态

运行以下命令检查所有依赖的安装状态:

```bash
# 方式 1: 使用我们提供的检查脚本 (推荐)
python scripts/check_dependencies.py

# 方式 2: 手动检查单个包
python -c "import torch; print(f'torch: {torch.__version__}')"
python -c "import transformers; print(f'transformers: {transformers.__version__}')"

# 方式 3: 列出所有已安装的包
pip list | grep -E "(torch|transformers|fastapi|flashinfer|sgl)"
```

## 📈 版本更新历史

### 2026-01-26 - 依赖整理

**核心依赖**:
- ✅ torch: 添加版本范围 `>=2.0.0,<2.6.0`
- ✅ transformers: 保持 `>=4.56.0,<=4.57.3`
- ✅ accelerate: 添加最低版本 `>=0.34.0`
- ✅ fastapi: 添加最低版本 `>=0.115.0`
- ✅ uvicorn: 添加最低版本 `>=0.32.0`
- ✅ openai: 添加最低版本 `>=1.57.4`
- ✅ msgpack: 添加最低版本 `>=1.1.0`
- ✅ pyzmq: 添加最低版本 `>=26.2.0`
- ✅ prompt_toolkit: 添加最低版本 `>=3.0.36`

**开发依赖**:
- ✅ pytest: 更新到 `>=8.3.0` (从 >=6.0)
- ✅ pytest-cov: 更新到 `>=6.0.0` (从 >=2.0)
- ✅ black: 更新到 `>=24.8.0` (从 >=22.0)
- ✅ flake8: 更新到 `>=7.1.0` (从 >=4.0)
- ✅ mypy: 更新到 `>=1.13.0` (从 >=0.950)
- ✅ pre-commit: 更新到 `>=4.0.0` (从 >=3.0.0)
- ✅ pytest-asyncio: 新增 `>=0.24.0`
- ✅ numpy: 新增 `>=1.26.0`

## 🔄 升级建议

### 近期可升级 (低风险)
- uvicorn: 可升级到最新稳定版
- fastapi: 可升级到最新稳定版
- openai: 可升级到最新稳定版
- 所有开发依赖

### 需要测试后升级 (中风险)
- torch: 新版本可能影响性能
- transformers: 需要测试模型兼容性
- accelerate: 需要测试分布式功能

### 暂不建议升级 (高风险)
- sgl_kernel: 核心内核,需要充分测试
- flashinfer-python: 性能关键组件
- nvidia-cutlass-dsl: 固定版本依赖

## 💡 使用提示

### 安装推荐顺序

```bash
# 1. 首先安装 PyTorch (选择合适的 CUDA 版本)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2. 安装其余依赖
pip install -e .

# 3. 验证安装
python scripts/check_dependencies.py
```

### 虚拟环境推荐

```bash
# 使用 uv (推荐,速度快)
uv venv --python=3.12
source .venv/bin/activate
uv pip install -e .

# 使用 conda
conda create -n minisgl python=3.12
conda activate minisgl
pip install -e .

# 使用 venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

**说明**: 
- ✅ = 已测试,稳定可用
- ⚠️ = 有限支持,可能需要额外配置
- ❌ = 不支持
- ⬜ = 待检查 (运行检查脚本后更新)

