<p align="center">
<img width="400" src="/assets/logo.png">
</p>

# Mini-SGLang

一个**轻量级但高性能**的大语言模型推理框架。

---

Mini-SGLang 是 [SGLang](https://github.com/sgl-project/sglang) 的精简实现，旨在揭秘现代大语言模型服务系统的复杂性。它拥有仅 **约5000行Python代码**的紧凑代码库，既是一个高性能的推理引擎，也是研究人员和开发者的透明参考实现。

## ✨ 核心特性

- **高性能**：通过先进的优化技术实现业界领先的吞吐量和延迟。
- **轻量级且可读**：简洁、模块化、完整类型注解的代码库，易于理解和修改。
- **先进优化技术**：
  - **Radix Cache**：跨请求复用共享前缀的KV缓存。
  - **分块预填充（Chunked Prefill）**：降低长上下文服务的峰值内存使用。
  - **重叠调度（Overlap Scheduling）**：用GPU计算隐藏CPU调度开销。
  - **张量并行（Tensor Parallelism）**：跨多GPU扩展推理能力。
  - **优化内核**：集成 **FlashAttention** 和 **FlashInfer** 以获得最高效率。
  - ...

## 🚀 快速开始

> **⚠️ 平台支持**：Mini-SGLang 目前仅支持 **Linux**（x86_64 和 aarch64）。由于依赖于Linux特定的CUDA内核（`sgl-kernel`、`flashinfer`），不支持Windows和macOS。我们建议Windows用户使用 [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)，或使用Docker以实现跨平台兼容性。

### 1. 系统要求

**硬件要求：**
- **GPU**：NVIDIA GPU，CUDA计算能力7.0+（V100、T4、A100、H100、RTX 3090、RTX 4090等）
- **显存**： 
  - 小模型（0.6B-7B）：8GB+ GPU显存
  - 中等模型（7B-30B）：24GB+ GPU显存（或使用张量并行的多GPU）
  - 大模型（30B+）：多个40GB+显存的GPU
- **CPU**：多核处理器（建议4核以上）
- **内存**：16GB+ 系统内存

**软件要求：**
- **操作系统**：Linux（Ubuntu 20.04+、CentOS 7+或其他现代发行版）
- **Python**：3.10、3.11或3.12
- **CUDA**：11.8+或12.1+（必须与驱动版本匹配）
- **NVIDIA驱动**：525+（CUDA 12.x）或450+（CUDA 11.x）
- **Git**：用于克隆仓库

### 2. 检查环境

安装前，验证您的系统是否满足要求：

```bash
# 检查NVIDIA GPU和驱动版本
nvidia-smi

# 检查CUDA版本（应与驱动匹配）
nvcc --version  # 如果已安装CUDA toolkit

# 检查Python版本
python --version  # 应为3.10、3.11或3.12
```

### 3. 环境设置

我们推荐使用 `uv` 进行快速可靠的安装（注意 `uv` 不会与 `conda` 冲突）。

**选项A：使用uv（推荐）**

```bash
# 如果尚未安装，先安装uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境（推荐Python 3.12）
uv venv --python=3.12
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows (PowerShell)
```

**选项B：使用venv**

```bash
# 创建虚拟环境
python3.12 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows (PowerShell)
```

**选项C：使用conda**

```bash
# 创建conda环境
conda create -n minisgl python=3.12
conda activate minisgl
```

### 4. 安装

**前置条件**：Mini-SGLang 依赖于JIT编译的CUDA内核。请确保已安装 **NVIDIA CUDA Toolkit**，且其版本与驱动版本匹配。

**分步安装：**

```bash
# 1. 克隆仓库
git clone https://github.com/FlyAIBox/mini-sglang.git

# 2. 激活虚拟环境
source .venv/bin/activate  # 或使用conda环境

# 3. 安装Mini-SGLang
# 如果使用uv：
uv pip install -e .

# 如果使用pip：
pip install -e .

# 4. 验证安装
python -c "import minisgl; print('Mini-SGLang安装成功！')"
```

**安装时间**：
- 首次安装：5-15分钟（取决于网络速度）
- CUDA内核将在首次使用时进行JIT编译（首次运行时增加1-3分钟）

**安装问题排查：**

<details>
<summary><b>🔧 找不到CUDA Toolkit</b></summary>

如果遇到缺少CUDA toolkit的错误：

1. **安装CUDA Toolkit**：
   ```bash
   # Ubuntu/Debian
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   sudo apt-get install cuda-toolkit-12-1
   
   # 或从这里下载：https://developer.nvidia.com/cuda-downloads
   ```

2. **设置环境变量**：
   ```bash
   export CUDA_HOME=/usr/local/cuda
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   ```

3. **添加到~/.bashrc使其永久生效**：
   ```bash
   echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
   echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

</details>

<details>
<summary><b>🔧 PyTorch/CUDA版本不匹配</b></summary>

如果遇到CUDA版本不匹配问题：

1. **检查CUDA版本**：
   ```bash
   nvidia-smi  # 查看右上角的CUDA Version
   ```

2. **安装匹配的PyTorch**：
   ```bash
   # CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **验证PyTorch CUDA**：
   ```bash
   python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
   ```

</details>

<details>
<summary><b>🔧 安装时内存不足</b></summary>

如果安装因内存问题失败：

```bash
# 减小pip的缓存大小
pip install -e . --no-cache-dir

# 或限制并行度安装
MAX_JOBS=4 pip install -e .
```

</details>

<details>
<summary><b>💡 在Windows上安装（WSL2）</b></summary>

由于Mini-SGLang需要Linux特定依赖，Windows用户应使用WSL2：

1. **安装WSL2**（如果尚未安装）：
   ```powershell
   # 在PowerShell中（以管理员身份）
   wsl --install
   ```

2. **在WSL2上安装CUDA**：
   - 按照 [NVIDIA的WSL2 CUDA指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
   - 确保Windows GPU驱动支持WSL2

3. **在WSL2中安装Mini-SGLang**：
   ```bash
   # 在WSL2终端内
   git clone https://github.com/sgl-project/mini-sglang.git
   cd mini-sglang && uv venv --python=3.12 && source .venv/bin/activate
   uv pip install -e .
   ```

4. **从Windows访问**：服务器可在Windows浏览器和应用程序中通过 `http://localhost:8000` 访问。

</details>

### 5. 首次运行 - 快速测试

从小模型开始验证一切正常：

```bash
# 使用小模型（0.6B参数）启动服务器
python -m minisgl --model "Qwen/Qwen3-0.6B"

# 服务器将：
# 1. 从HuggingFace下载模型（仅首次运行，约1.2GB）
# 2. 将模型加载到GPU显存
# 3. 在 http://localhost:8000 启动API服务器
# 4. 准备就绪时显示"Server started successfully"
```

**预期输出：**
```
[INFO] Loading model: Qwen/Qwen3-0.6B
[INFO] Initializing tokenizer...
[INFO] Initializing scheduler worker (TP Rank 0/1)...
[INFO] Initializing engine...
[INFO] Loading weights...
[INFO] Starting API server on http://0.0.0.0:8000
[INFO] Server started successfully!
```

**测试服务器：**

```bash
# 在新终端中，使用curl测试
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "你好！你好吗？"}
    ]
  }'
```

### 6. 在线服务（生产使用）

使用各种配置启动OpenAI兼容的API服务器。

**基本用法：**

```bash
# 单GPU默认设置
python -m minisgl --model "Qwen/Qwen3-0.6B"

# 指定端口和主机
python -m minisgl --model "Qwen/Qwen3-0.6B" --host 0.0.0.0 --port 8000

# 使用Radix Cache提升性能（默认）
python -m minisgl --model "Qwen/Qwen3-0.6B" --cache radix

# 禁用Radix Cache（用于调试或简单行为）
python -m minisgl --model "Qwen/Qwen3-0.6B" --cache naive
```

**多GPU张量并行：**

```bash
# 在2个GPU上部署
python -m minisgl --model "Qwen/Qwen3-14B" --tp 2

# 在4个GPU上部署，使用自定义端口
python -m minisgl --model "meta-llama/Llama-3.1-70B-Instruct" --tp 4 --port 30000

# 在8个GPU上部署（超大模型）
python -m minisgl --model "meta-llama/Llama-3.1-405B" --tp 8
```

**高级选项：**

```bash
# 调整内存和批处理大小
python -m minisgl \
  --model "Qwen/Qwen3-14B" \
  --tp 2 \
  --max-batch-size 256 \
  --max-total-tokens 8192

# 使用特定注意力后端
python -m minisgl \
  --model "Qwen/Qwen3-0.6B" \
  --attention-backend flashinfer  # 或 flashattention

# 调整KV缓存页大小
python -m minisgl \
  --model "Qwen/Qwen3-0.6B" \
  --page-size 16  # 默认为16
```

**常用命令行参数：**

| 参数 | 默认值 | 说明 |
|----------|---------|-------------|
| `--model` | 必需 | HuggingFace模型ID或本地路径 |
| `--tp` | 1 | 张量并行使用的GPU数量 |
| `--host` | "0.0.0.0" | 服务器主机地址 |
| `--port` | 8000 | 服务器端口号 |
| `--cache` | "radix" | 缓存策略："radix"或"naive" |
| `--attention-backend` | "flashinfer" | "flashinfer"或"flashattention" |
| `--max-batch-size` | 128 | 最大批处理大小 |
| `--max-total-tokens` | 4096 | 批处理中的最大总token数 |
| `--page-size` | 16 | KV缓存页大小 |

查看所有选项请运行 `python -m minisgl --help`。

**使用API：**

服务器运行后，您可以通过以下方式发送请求：

```bash
# 使用curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "解释量子计算"}],
    "max_tokens": 512,
    "temperature": 0.7
  }'

# 使用Python和OpenAI库
pip install openai
python examples/client.py  # 参见examples/目录
```

```python
# Python示例
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Mini-SGLang不需要认证
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[
        {"role": "user", "content": "生命的意义是什么？"}
    ],
    temperature=0.7,
    max_tokens=512
)

print(response.choices[0].message.content)
```

### 7. 交互式Shell

通过添加 `--shell` 标志直接在终端与模型对话。

```bash
# 启动交互式shell
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell

# 使用多GPU
python -m minisgl --model "Qwen/Qwen3-14B" --tp 2 --shell
```

![shell-example](https://lmsys.org/images/blog/minisgl/shell.png)

**Shell命令：**
- 输入消息并按Enter进行对话
- `/reset` - 清除对话历史并重新开始
- `/exit` 或 Ctrl+D - 退出shell
- `/help` - 显示帮助信息

### 8. 将Mini-SGLang用作Python库

您也可以直接在Python中使用Mini-SGLang，无需启动服务器：

```python
from minisgl.llm import LLM
from minisgl.core import SamplingParams

# 初始化LLM
llm = LLM(model="Qwen/Qwen3-0.6B", tp=1)

# 单个prompt
output = llm.generate(
    "解释机器学习中的transformer是如何工作的",
    SamplingParams(temperature=0.8, max_tokens=256)
)
print(output)

# 批量prompts
prompts = [
    "什么是人工智能？",
    "解释量子计算",
    "神经网络如何工作？"
]
outputs = llm.generate_batch(
    prompts,
    SamplingParams(temperature=0.7, max_tokens=128)
)

for prompt, output in zip(prompts, outputs):
    print(f"问：{prompt}")
    print(f"答：{output}\n")
```

### 9. 停止服务器

优雅地停止服务器：

```bash
# 在运行服务器的终端按Ctrl+C
# 服务器将：
# 1. 停止接受新请求
# 2. 等待当前请求完成
# 3. 清理资源并退出
```

**强制停止**（如果优雅关闭挂起）：
```bash
# 查找进程
ps aux | grep minisgl

# 终止它
kill -9 <PID>
```

## 性能基准测试

### 离线推理

详情请参见 [bench.py](./benchmark/offline/bench.py)。设置 `MINISGL_DISABLE_OVERLAP_SCHEDULING=1` 可进行重叠调度的消融研究。

测试配置：

- 硬件：1xH200 GPU
- 模型：Qwen3-0.6B、Qwen3-14B
- 总请求数：256个序列
- 输入长度：100-1024 tokens随机采样
- 输出长度：100-1024 tokens随机采样

![offline](https://lmsys.org/images/blog/minisgl/offline.png)

### 在线推理

详情请参见 [benchmark_qwen.py](./benchmark/online/bench_qwen.py)。

测试配置：

- 硬件：4xH200 GPU，通过NVLink连接
- 模型：Qwen3-32B
- 数据集：[Qwen trace](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon/blob/main/qwen_traceA_blksz_16.jsonl)，重放前1000个请求

启动命令：

```bash
# Mini-SGLang
python -m minisgl --model "Qwen/Qwen3-32B" --tp 4 --cache naive

# SGLang
python3 -m sglang.launch_server --model "Qwen/Qwen3-32B" --tp 4 \
    --disable-radix --port 1919 --decode-attention flashinfer
```

![online](https://lmsys.org/images/blog/minisgl/online.png)

## 📚 了解更多

### 📖 文档索引

- **[📑 完整文档索引](./DOCUMENTATION_INDEX.md)**：按主题和用例组织的所有文档。

### 英文文档

- **[📦 依赖项摘要](./DEPENDENCIES_SUMMARY.md)**：包版本和安装命令的快速参考。
- **[📋 依赖项指南](./DEPENDENCIES.md)**：所有依赖项、版本和安装故障排除的完整指南。
- **[📊 依赖项版本表](./DEPENDENCIES_VERSION_TABLE.md)**：版本兼容性矩阵和升级建议。
- **[✨ 详细功能](./docs/features.md)**：探索所有可用功能和命令行参数。
- **[🏗️ 系统架构](./docs/structures.md)**：深入了解Mini-SGLang的设计和数据流。

### 中文文档

- **[中文安装和使用指南](./docs/zh/README_ZH.md)**：详细的中文安装、配置和使用教程
- **[核心概念详解](./docs/zh/concepts.md)**：深入讲解大模型推理的核心技术和原理
- **[开发者指南](./docs/zh/developer.md)**：面向开发者的代码结构、调试技巧和贡献指南
- **[文档总结](./docs/zh/SUMMARY.md)**：中文文档和代码注释的完整说明

## 🔧 项目配置文件说明

Mini-SGLang 使用现代 Python 项目管理工具和配置文件来确保代码质量和开发效率。以下是主要配置文件及其作用：

### `pyproject.toml` - 项目配置中心

这是项目的**核心配置文件**，采用现代 Python 打包标准（PEP 518/621），统一管理项目的所有配置。

**主要功能：**

1. **项目元数据**：定义项目名称、版本、描述、许可证等基本信息
2. **依赖管理**：声明核心依赖和开发依赖，支持版本约束
   - `dependencies`：运行时必需的核心依赖（PyTorch、FastAPI、FlashInfer 等）
   - `optional-dependencies.dev`：开发工具依赖（pytest、black、ruff 等）
3. **工具配置**：集中配置代码质量工具
   - `[tool.black]`：代码格式化配置（行长100字符，Python 3.10+）
   - `[tool.ruff]`：快速 linter 配置（代码风格检查、导入排序）
   - `[tool.pytest]`：测试框架配置（测试路径、覆盖率报告）
   - `[tool.mypy]`：类型检查配置（严格模式、类型覆盖）

**为什么重要：** 单一配置文件减少了配置碎片化，确保所有开发者使用一致的工具设置。

### `requirements.txt` - 生产环境依赖

列出**运行 Mini-SGLang 所必需的核心依赖**，适用于生产部署。

**依赖分类：**

- **深度学习框架**：PyTorch、Transformers、Accelerate
- **CUDA 优化**：sgl_kernel、flashinfer、TVM FFI
- **Web 服务**：FastAPI、Uvicorn、OpenAI SDK
- **通信**：ZeroMQ、MessagePack
- **CLI**：prompt_toolkit

**使用场景：**
```bash
# 生产环境安装
pip install -r requirements.txt
```

### `requirements-dev.txt` - 开发环境依赖

包含**开发、测试和代码质量工具**的完整依赖列表。

**额外工具：**

- **测试工具**：pytest、pytest-cov、pytest-asyncio
- **代码格式化**：black、ruff、flake8
- **类型检查**：mypy
- **Git 钩子**：pre-commit
- **数据分析**：matplotlib、numpy

**使用场景：**
```bash
# 开发环境安装（包含所有依赖）
pip install -r requirements.txt -r requirements-dev.txt
```

### `.pre-commit-config.yaml` - Git 提交前检查

配置 **Git pre-commit 钩子**，在每次提交代码前自动执行代码质量检查。

**检查项目：**

1. **基础检查**（via pre-commit-hooks）：
   - 删除行尾空格
   - 确保文件以换行符结尾
   - 验证 YAML/TOML 语法
   - 检测大文件、合并冲突、私钥泄漏

2. **Python 代码格式化**（via black）：
   - 自动格式化 Python 代码为统一风格

3. **Python 代码检查**（via ruff）：
   - 快速检查代码风格问题
   - 自动修复简单问题（如导入排序）

4. **C++/CUDA 代码格式化**（via clang-format）：
   - 格式化自定义 CUDA 内核代码

**激活方法：**
```bash
# 安装 pre-commit 钩子
pip install pre-commit
pre-commit install

# 手动运行所有检查
pre-commit run --all-files
```

**为什么重要：** 在提交前自动捕获代码质量问题，确保所有提交的代码符合项目规范，减少 code review 负担。

### `.python-version` - Python 版本管理

指定项目**推荐的 Python 版本**（3.12）。

**作用：**

- 被 `pyenv`、`asdf` 等版本管理工具自动识别
- 确保团队使用一致的 Python 版本
- 避免因 Python 版本差异导致的兼容性问题

**使用示例（with pyenv）：**
```bash
# pyenv 会自动读取 .python-version
cd mini-sglang
pyenv install  # 安装指定版本
python --version  # Python 3.12.x
```

---

### 配置文件之间的关系

```
pyproject.toml (主配置)
    ├── 定义所有依赖 → requirements.txt (生产)
    │                 └── requirements-dev.txt (开发)
    │
    ├── 配置代码工具 → .pre-commit-config.yaml (Git 钩子)
    │                 └── black, ruff, pytest, mypy
    │
    └── 指定 Python → .python-version
```

**最佳实践：**

1. **日常开发**：安装 `requirements-dev.txt` 获得完整开发环境
2. **启用 pre-commit**：确保代码质量从源头把控
3. **生产部署**：仅安装 `requirements.txt` 减小镜像体积
4. **版本管理**：使用 `pyenv` 或 `uv` 自动管理 Python 版本

