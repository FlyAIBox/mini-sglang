# Mini-SGLang 安装和使用完整指南

## 📋 目录

1. [系统要求](#系统要求)
2. [安装前准备](#安装前准备)
3. [安装步骤](#安装步骤)
4. [首次运行](#首次运行)
5. [使用方法](#使用方法)
6. [常见问题](#常见问题)
7. [性能优化](#性能优化)
8. [故障排除](#故障排除)

---

## 系统要求

### 硬件要求

| 组件 | 最低要求 | 推荐配置 | 说明 |
|------|---------|---------|------|
| **GPU** | NVIDIA GPU (Compute Capability 7.0+) | A100/H100/RTX 4090 | V100/T4/RTX 3090也可用 |
| **显存** | 8GB | 24GB+ | 小模型8GB，中大模型需24GB+ |
| **CPU** | 4核心 | 8核心+ | 多核心提升tokenization速度 |
| **内存** | 16GB | 32GB+ | 加载大模型需要更多内存 |
| **存储** | 20GB可用空间 | 100GB+ SSD | 模型缓存和临时文件 |

### 软件要求

| 软件 | 版本要求 | 检查命令 |
|------|---------|----------|
| **操作系统** | Linux (Ubuntu 20.04+, CentOS 7+) | `lsb_release -a` |
| **Python** | 3.10, 3.11, or 3.12 | `python --version` |
| **CUDA** | 11.8+ or 12.1+ | `nvcc --version` |
| **NVIDIA驱动** | 525+ (CUDA 12.x) or 450+ (CUDA 11.x) | `nvidia-smi` |
| **Git** | 任何最新版本 | `git --version` |

### 支持的模型

| 模型系列 | 最小显存 | 推荐TP数 | 示例 |
|---------|---------|---------|------|
| **超小模型** (0.6B-1B) | 4GB | 1 | Qwen3-0.6B, Llama-1B |
| **小模型** (7B-8B) | 16GB | 1 | Qwen3-7B, Llama-3-8B |
| **中等模型** (14B-30B) | 24GB | 2-4 | Qwen3-14B, Qwen3-32B |
| **大模型** (70B+) | 40GB+ | 4-8 | Llama-3.1-70B, Qwen3-72B |
| **超大模型** (405B+) | 80GB+ | 8+ | Llama-3.1-405B |

---

## 安装前准备

### 1. 检查GPU和驱动

```bash
# 检查GPU是否可用
nvidia-smi

# 应该看到类似输出：
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.54.03    Driver Version: 535.54.03    CUDA Version: 12.2     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# ...
```

**重要信息**：
- **Driver Version**: 驱动版本（必须 >= 525 for CUDA 12.x）
- **CUDA Version**: 驱动支持的最高CUDA版本
- **Memory-Usage**: 显存使用情况

### 2. 安装CUDA Toolkit（如果未安装）

```bash
# Ubuntu 22.04 示例
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-1

# 设置环境变量
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 添加到~/.bashrc使其永久生效
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证安装
nvcc --version
```

### 3. 安装Python 3.10+（如果需要）

```bash
# Ubuntu
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev

# 验证
python3.12 --version
```

---

## 安装步骤

### 方法一：使用uv（推荐，最快）

```bash
# 1. 安装uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 克隆仓库
git clone https://github.com/sgl-project/mini-sglang.git
cd mini-sglang

# 3. 创建虚拟环境
uv venv --python=3.12
source .venv/bin/activate  # Linux/Mac

# 4. 安装Mini-SGLang
uv pip install -e .

# 5. 验证安装
python -c "import minisgl; print('Mini-SGLang installed successfully!')"
```

**预计时间**: 5-10分钟（取决于网络速度）

### 方法二：使用pip

```bash
# 1. 克隆仓库
git clone https://github.com/sgl-project/mini-sglang.git
cd mini-sglang

# 2. 创建虚拟环境
python3.12 -m venv .venv
source .venv/bin/activate

# 3. 升级pip
pip install --upgrade pip

# 4. 安装Mini-SGLang
pip install -e .

# 5. 验证安装
python -c "import minisgl; print('Installation successful!')"
```

**预计时间**: 10-15分钟

### 方法三：使用conda

```bash
# 1. 创建conda环境
conda create -n minisgl python=3.12
conda activate minisgl

# 2. 克隆仓库
git clone https://github.com/sgl-project/mini-sglang.git
cd mini-sglang

# 3. 安装
pip install -e .

# 4. 验证
python -c "import minisgl; print('Installation successful!')"
```

---

## 首次运行

### 1. 快速测试（小模型）

```bash
# 启动服务器（会自动下载模型，约1.2GB）
python -m minisgl --model "Qwen/Qwen3-0.6B"

# 预期输出：
# [INFO] Loading model: Qwen/Qwen3-0.6B
# [INFO] Downloading model... (首次运行)
# [INFO] Initializing tokenizer...
# [INFO] Initializing scheduler worker (TP Rank 0/1)...
# [INFO] Free memory before loading model: 23.5 GB
# [INFO] Initializing engine...
# [INFO] Loading weights...
# [INFO] Free memory after initialization: 22.3 GB
# [INFO] Starting API server on http://0.0.0.0:8000
# [INFO] Server started successfully!
```

**启动时间**：
- 首次运行（含下载）：2-5分钟
- 后续运行：30-60秒

### 2. 测试API

在另一个终端：

```bash
# 测试聊天API
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "Hello! How are you?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'

# 预期输出（JSON格式）：
# {
#   "id": "chatcmpl-xxx",
#   "object": "chat.completion",
#   "created": 1234567890,
#   "model": "Qwen/Qwen3-0.6B",
#   "choices": [{
#     "index": 0,
#     "message": {
#       "role": "assistant",
#       "content": "Hello! I'm doing well, thank you for asking..."
#     },
#     "finish_reason": "stop"
#   }],
#   ...
# }
```

### 3. 交互式Shell测试

```bash
# 启动交互式shell
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell

# 在shell中对话
You: Hello!
Assistant: Hello! How can I help you today?

You: What is AI?
Assistant: AI stands for Artificial Intelligence...

# 命令
/reset   - 重置对话历史
/exit    - 退出shell
Ctrl+D   - 退出shell
```

---

## 使用方法

### 1. 基本服务器启动

```bash
# 单GPU，默认配置
python -m minisgl --model "Qwen/Qwen3-0.6B"

# 指定端口和host
python -m minisgl --model "Qwen/Qwen3-0.6B" --host 0.0.0.0 --port 8000

# 使用本地模型路径
python -m minisgl --model /path/to/local/model
```

### 2. 多GPU（Tensor Parallelism）

```bash
# 2个GPU
python -m minisgl --model "Qwen/Qwen3-14B" --tp 2

# 4个GPU
python -m minisgl --model "meta-llama/Llama-3.1-70B-Instruct" --tp 4

# 8个GPU
python -m minisgl --model "meta-llama/Llama-3.1-405B" --tp 8
```

### 3. 性能调优选项

```bash
# 使用Radix Cache（默认，推荐）
python -m minisgl --model "Qwen/Qwen3-0.6B" --cache radix

# 禁用Radix Cache（调试用）
python -m minisgl --model "Qwen/Qwen3-0.6B" --cache naive

# 调整batch size
python -m minisgl --model "Qwen/Qwen3-14B" --max-batch-size 256

# 调整最大token数
python -m minisgl --model "Qwen/Qwen3-14B" --max-total-tokens 8192

# 选择注意力后端
python -m minisgl --model "Qwen/Qwen3-0.6B" --attention-backend flashinfer  # 默认
python -m minisgl --model "Qwen/Qwen3-0.6B" --attention-backend flashattention

# 调整KV Cache页大小
python -m minisgl --model "Qwen/Qwen3-0.6B" --page-size 16  # 默认16
```

### 4. 使用Python API

```python
from openai import OpenAI

# 创建客户端
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Mini-SGLang不需要认证
)

# 聊天补全
response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    max_tokens=512,
    temperature=0.7,
    top_p=0.9
)

print(response.choices[0].message.content)

# 流式输出
stream = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "Tell me a story"}],
    max_tokens=1000,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

### 5. 作为Python库使用

```python
from minisgl.llm import LLM
from minisgl.core import SamplingParams

# 初始化模型
llm = LLM(
    model="Qwen/Qwen3-0.6B",
    tp=1,  # Tensor Parallelism数量
    cache_mode="radix"  # 或 "naive"
)

# 单个prompt
output = llm.generate(
    prompt="What is the capital of France?",
    sampling_params=SamplingParams(
        temperature=0.7,
        max_tokens=100
    )
)
print(output)

# 批量prompts
prompts = [
    "What is AI?",
    "Explain machine learning",
    "What is deep learning?"
]

outputs = llm.generate_batch(
    prompts=prompts,
    sampling_params=SamplingParams(
        temperature=0.8,
        max_tokens=200
    )
)

for prompt, output in zip(prompts, outputs):
    print(f"Q: {prompt}")
    print(f"A: {output}\n")
```

---

## 常见问题

### Q1: 如何选择合适的模型大小？

**答**：根据显存选择：

| 显存 | 推荐模型 | TP配置 |
|------|---------|--------|
| 4-8GB | 0.6B-1B | tp=1 |
| 16GB | 7B-8B | tp=1 |
| 24GB | 14B | tp=1 or tp=2 |
| 2×24GB | 32B-34B | tp=2 |
| 4×24GB | 70B | tp=4 |
| 8×80GB | 405B | tp=8 |

### Q2: 模型下载到哪里？

**答**：HuggingFace模型默认缓存在：
```bash
~/.cache/huggingface/hub/
```

查看已下载的模型：
```bash
ls -lh ~/.cache/huggingface/hub/
```

清理缓存：
```bash
rm -rf ~/.cache/huggingface/hub/*
```

设置自定义缓存路径：
```bash
export HF_HOME=/your/custom/path
```

### Q3: 如何停止服务器？

**答**：
- **正常停止**: 按 `Ctrl+C`（等待当前请求完成）
- **强制停止**: 按 `Ctrl+C` 两次，或 `kill -9 <PID>`

### Q4: 支持哪些模型？

**答**：目前支持：
- **Qwen系列**: Qwen3-0.6B, Qwen3-7B, Qwen3-14B, Qwen3-32B等
- **Llama系列**: Llama-3-8B, Llama-3.1-70B, Llama-3.1-405B等
- 其他基于Llama架构的模型

### Q5: 如何提高推理速度？

**答**：
1. 启用Radix Cache（默认启用）
2. 使用多GPU（`--tp`）
3. 调整batch size（`--max-batch-size`）
4. 使用FlashInfer后端（默认）
5. 启用CUDA Graph（自动）

### Q6: Windows用户如何使用？

**答**：使用WSL2：

```powershell
# 1. 安装WSL2（以管理员身份运行PowerShell）
wsl --install

# 2. 重启电脑

# 3. 在WSL2中安装CUDA（在WSL终端中）
# 按照Linux安装步骤

# 4. 从Windows访问服务器
# http://localhost:8000 在Windows浏览器中可用
```

---

## 性能优化

### 1. 内存优化

```bash
# 减小batch size
python -m minisgl --model "Qwen/Qwen3-14B" --max-batch-size 64

# 减小最大序列长度
python -m minisgl --model "Qwen/Qwen3-14B" --max-seq-len 4096

# 调整page size
python -m minisgl --model "Qwen/Qwen3-14B" --page-size 8
```

### 2. 吞吐量优化

```bash
# 增大batch size
python -m minisgl --model "Qwen/Qwen3-0.6B" --max-batch-size 512

# 增加最大token数
python -m minisgl --model "Qwen/Qwen3-0.6B" --max-total-tokens 16384

# 使用Radix Cache
python -m minisgl --model "Qwen/Qwen3-0.6B" --cache radix
```

### 3. 延迟优化

```bash
# 减小batch size（优先响应速度）
python -m minisgl --model "Qwen/Qwen3-0.6B" --max-batch-size 32

# 使用FlashInfer
python -m minisgl --model "Qwen/Qwen3-0.6B" --attention-backend flashinfer
```

---

## 故障排除

### 错误1: CUDA out of memory

**症状**：
```
RuntimeError: CUDA out of memory. Tried to allocate X GB...
```

**解决方案**：
```bash
# 1. 减小batch size
python -m minisgl --model "Qwen/Qwen3-14B" --max-batch-size 64

# 2. 使用更多GPU
python -m minisgl --model "Qwen/Qwen3-14B" --tp 2

# 3. 使用更小的模型
python -m minisgl --model "Qwen/Qwen3-7B"

# 4. 清理GPU缓存
python -c "import torch; torch.cuda.empty_cache()"
```

### 错误2: CUDA toolkit not found

**症状**：
```
RuntimeError: CUDA toolkit not found...
```

**解决方案**：
```bash
# 安装CUDA toolkit
sudo apt install cuda-toolkit-12-1

# 设置环境变量
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 错误3: ImportError: flashinfer not found

**症状**：
```
ImportError: cannot import name 'flashinfer'...
```

**解决方案**：
```bash
# 重新安装依赖
pip uninstall flashinfer-python
pip install flashinfer-python>=0.5.3
```

### 错误4: 模型下载失败

**症状**：
```
HTTPError: 403 Forbidden...
```

**解决方案**：
```bash
# 1. 设置HuggingFace token（如果模型需要授权）
export HF_TOKEN=your_token_here

# 2. 使用镜像站（中国用户）
export HF_ENDPOINT=https://hf-mirror.com

# 3. 手动下载模型
git clone https://huggingface.co/Qwen/Qwen3-0.6B /path/to/model
python -m minisgl --model /path/to/model
```

### 错误5: 端口被占用

**症状**：
```
OSError: [Errno 98] Address already in use
```

**解决方案**：
```bash
# 1. 使用其他端口
python -m minisgl --model "Qwen/Qwen3-0.6B" --port 8001

# 2. 查找并终止占用端口的进程
lsof -i :8000
kill -9 <PID>

# 3. 或者直接
pkill -f minisgl
```

---

## 附录

### A. 完整命令行参数

```bash
python -m minisgl --help

# 主要参数：
#   --model MODEL              模型名称或路径
#   --tp TP                    Tensor Parallelism GPU数量（默认1）
#   --host HOST                服务器地址（默认0.0.0.0）
#   --port PORT                服务器端口（默认8000）
#   --cache {radix,naive}      缓存策略（默认radix）
#   --attention-backend {flashinfer,flashattention}
#                              注意力后端（默认flashinfer）
#   --max-batch-size SIZE      最大batch size（默认128）
#   --max-total-tokens TOKENS  最大总token数（默认4096）
#   --max-seq-len LEN          最大序列长度（默认8192）
#   --page-size SIZE           KV Cache页大小（默认16）
#   --shell                    启动交互式shell
```

### B. 环境变量

```bash
# HuggingFace配置
export HF_HOME=/path/to/cache          # HF缓存目录
export HF_TOKEN=your_token             # HF访问token
export HF_ENDPOINT=https://hf-mirror.com  # HF镜像（中国用户）

# Mini-SGLang配置
export MINISGL_DISABLE_OVERLAP_SCHEDULING=1  # 禁用重叠调度
export MINISGL_SHELL_MAX_TOKENS=4096         # Shell最大token数
export MINISGL_SHELL_TEMPERATURE=0.7         # Shell温度参数

# CUDA配置
export CUDA_HOME=/usr/local/cuda
export CUDA_VISIBLE_DEVICES=0,1,2,3    # 指定使用的GPU
```

### C. 资源链接

- **项目主页**: https://github.com/sgl-project/mini-sglang
- **文档**: `docs/` 目录
- **问题反馈**: https://github.com/sgl-project/mini-sglang/issues
- **SGLang主项目**: https://github.com/sgl-project/sglang

---

**最后更新**: 2026-01-26  
**文档版本**: 1.0

