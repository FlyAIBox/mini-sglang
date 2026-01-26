# 依赖版本快速参考

> 最后更新: 2026-01-26

## 🚀 核心依赖 (生产环境)

| 分类 | 包名 | 版本 | 用途 |
|------|------|------|------|
| **深度学习** | torch | >=2.0.0, <2.6.0 | PyTorch 核心 |
| | transformers | >=4.56.0, <=4.57.3 | 模型库 |
| | accelerate | >=0.34.0 | 推理加速 |
| **GPU 优化** | sgl_kernel | >=0.3.17.post1 | 自定义内核 |
| | flashinfer-python | >=0.5.3 | 注意力优化 |
| | nvidia-cutlass-dsl | ==4.3.1 | CUTLASS 模板 |
| | apache-tvm-ffi | >=0.1.4 | TVM 编译器 |
| **Web 服务** | fastapi | >=0.115.0 | API 框架 |
| | uvicorn | >=0.32.0 | ASGI 服务器 |
| | openai | >=1.57.4 | API 客户端 |
| **工具库** | msgpack | >=1.1.0 | 序列化 |
| | pyzmq | >=26.2.0 | 进程通信 |
| | prompt_toolkit | >=3.0.36 | 命令行界面 |

## 🛠️ 开发依赖 (仅开发环境)

| 分类 | 包名 | 版本 | 用途 |
|------|------|------|------|
| **测试** | pytest | >=8.3.0 | 单元测试 |
| | pytest-cov | >=6.0.0 | 覆盖率 |
| | pytest-asyncio | >=0.24.0 | 异步测试 |
| **代码质量** | black | >=24.8.0 | 格式化 |
| | ruff | >=0.11.0 | Linter |
| | flake8 | >=7.1.0 | 风格检查 |
| | mypy | >=1.13.0 | 类型检查 |
| | pre-commit | >=4.0.0 | Git 钩子 |
| **可视化** | matplotlib | >=3.10.5 | 绘图 |
| | numpy | >=1.26.0 | 数值计算 |

## 📦 安装命令

```bash
# 方式 1: 使用 pip (标准方式)
pip install -e .                    # 仅核心依赖
pip install -e ".[dev]"             # 包含开发依赖

# 方式 2: 使用 uv (推荐,更快)
uv pip install -e .                 # 仅核心依赖
uv pip install -e ".[dev]"          # 包含开发依赖

# 方式 3: 使用 requirements.txt
pip install -r requirements.txt              # 仅核心依赖
pip install -r requirements.txt -r requirements-dev.txt  # 全部依赖
```

## ⚙️ 系统要求

- **Python**: 3.10, 3.11, 或 3.12 (推荐 3.12)
- **CUDA**: 11.8+ 或 12.1+ (推荐 12.1)
- **操作系统**: Linux (Ubuntu 20.04+, CentOS 7+)
- **GPU**: NVIDIA GPU with Compute Capability 7.0+

## 📊 版本选择说明

### PyTorch 版本
- 使用 `>=2.0.0, <2.6.0` 确保兼容性
- 需要与 CUDA 版本匹配

### Transformers 版本
- 限制在 `4.56.0-4.57.3` 之间
- 这个范围内的版本经过充分测试

### CUDA 相关包
- `nvidia-cutlass-dsl` 固定为 `4.3.1`
- `sgl_kernel` 和 `flashinfer-python` 使用最低版本限制

## 🔍 常见问题

### Q1: 如何选择 PyTorch 的 CUDA 版本?

```bash
# 查看系统 CUDA 版本
nvidia-smi

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Q2: 开发依赖是否必需?

生产环境不需要开发依赖,只用于:
- 运行测试 (pytest)
- 代码格式化 (black, ruff)
- 类型检查 (mypy)
- 性能分析 (matplotlib)

### Q3: 如何验证安装?

```bash
# 验证核心安装
python -c "import minisgl; print('✅ Mini-SGLang installed')"

# 验证 CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 验证 FlashInfer
python -c "import flashinfer; print('✅ FlashInfer installed')"
```

---

详细说明请参考 [DEPENDENCIES.md](./DEPENDENCIES.md)

