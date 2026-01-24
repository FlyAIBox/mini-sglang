<p align="center">
<img width="400" src="/assets/logo.png">
</p>

# Mini-SGLang 中文文档

一个**轻量级且高性能**的大语言模型推理框架。

---

Mini-SGLang 是 [SGLang](https://github.com/sgl-project/sglang) 的精简实现，旨在揭开现代大语言模型服务系统的复杂面纱。仅用**约5000行Python代码**，它既是一个高性能的推理引擎，也是研究人员和开发者的透明参考实现。

## ✨ 核心特性

- **高性能**: 通过先进的优化技术实现业界领先的吞吐量和延迟
- **轻量且可读**: 代码库简洁、模块化、完全类型注解，易于理解和修改
- **高级优化技术**:
  - **Radix Cache（基数缓存）**: 在多个请求间复用共享前缀的KV缓存
  - **Chunked Prefill（分块预填充）**: 降低长上下文服务的峰值内存使用
  - **Overlap Scheduling（重叠调度）**: 将CPU调度开销隐藏在GPU计算中
  - **Tensor Parallelism（张量并行）**: 跨多GPU扩展推理能力
  - **优化内核**: 集成 **FlashAttention** 和 **FlashInfer** 实现最高效率
  - ...

## 📋 目录

- [安装指南](#-安装指南)
  - [系统要求](#系统要求)
  - [环境准备](#环境准备)
  - [安装步骤](#安装步骤)
  - [Windows用户（WSL2）](#windows用户wsl2)
- [快速开始](#-快速开始)
  - [在线推理服务](#在线推理服务)
  - [交互式Shell](#交互式shell)
  - [Python API](#python-api)
- [核心概念](#-核心概念)
- [高级功能](#-高级功能)
- [性能基准测试](#-性能基准测试)
- [故障排除](#-故障排除)

## 🚀 安装指南

### 系统要求

> **⚠️ 平台支持**: Mini-SGLang 目前**仅支持 Linux**（x86_64 和 aarch64）。由于依赖于Linux特定的CUDA内核（`sgl-kernel`、`flashinfer`），Windows和macOS不被支持。Windows用户建议使用 [WSL2](https://learn.microsoft.com/zh-cn/windows/wsl/install)，或使用Docker实现跨平台兼容性。

**硬件要求：**
- NVIDIA GPU（计算能力 >= 8.0，推荐使用 A100/H100/L40s 等）
- 最少 16GB 显存（根据模型大小而定）
- 推荐使用 NVLink 或 PCIe 互连的多GPU配置（用于张量并行）

**软件要求：**
- Linux操作系统（Ubuntu 20.04+ 或其他主流发行版）
- Python 3.10 或更高版本（推荐 3.12）
- NVIDIA CUDA Toolkit（版本需要与驱动程序版本匹配）
- NVIDIA驱动程序（检查方法：运行 `nvidia-smi`）

### 环境准备

#### 1. 检查CUDA环境

首先确认你的CUDA环境是否正确配置：

```bash
# 检查NVIDIA驱动和CUDA版本
nvidia-smi

# 检查CUDA Toolkit是否安装
nvcc --version
```

如果 `nvcc` 命令不可用，你需要安装CUDA Toolkit：

```bash
# Ubuntu/Debian 示例（请根据你的CUDA版本调整）
# 访问 https://developer.nvidia.com/cuda-downloads 获取最新安装指令

# 或者使用 conda 安装
conda install -c nvidia cuda-toolkit
```

#### 2. 创建Python虚拟环境

我们推荐使用 `uv` 进行快速可靠的安装（注意：`uv` 不会与 `conda` 冲突）：

```bash
# 安装 uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境（推荐 Python 3.12）
uv venv --python=3.12

# 激活虚拟环境
source .venv/bin/activate
```

或者使用传统的 `venv`：

```bash
# 创建虚拟环境
python3.12 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate
```

### 安装步骤

#### 从源码安装（推荐）

```bash
# 1. 克隆仓库
git clone https://github.com/sgl-project/mini-sglang.git
cd mini-sglang

# 2. 创建并激活虚拟环境
uv venv --python=3.12
source .venv/bin/activate

# 3. 安装依赖
uv pip install -e .
```

#### 验证安装

```bash
# 检查安装是否成功
python -m minisgl --help

# 你应该看到命令行参数的帮助信息
```

### Windows用户（WSL2）

由于Mini-SGLang依赖Linux特定的库，Windows用户需要使用WSL2：

#### 1. 安装WSL2

```powershell
# 在PowerShell中以管理员身份运行
wsl --install

# 重启计算机后，设置Ubuntu用户名和密码
```

#### 2. 在WSL2中安装CUDA

```bash
# 在WSL2 Ubuntu终端中执行
# 参考：https://docs.nvidia.com/cuda/wsl-user-guide/index.html

# 下载并安装WSL2专用的CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-wsl-ubuntu-12-2-local_12.2.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-2-local_12.2.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

#### 3. 验证WSL2 GPU访问

```bash
# 在WSL2中验证GPU访问
nvidia-smi

# 你应该能看到你的GPU信息
```

#### 4. 在WSL2中安装Mini-SGLang

```bash
# 在WSL2 Ubuntu终端中执行
git clone https://github.com/sgl-project/mini-sglang.git
cd mini-sglang
uv venv --python=3.12
source .venv/bin/activate
uv pip install -e .
```

#### 5. 从Windows访问服务

启动服务后，你可以从Windows浏览器和应用程序通过 `http://localhost:8000` 访问。

## 🎯 快速开始

### 在线推理服务

Mini-SGLang 提供兼容 OpenAI API 的服务端点，可以无缝集成到现有工具和客户端中。

#### 基本用法

```bash
# 在单GPU上部署 Qwen/Qwen3-0.6B 模型
python -m minisgl --model "Qwen/Qwen3-0.6B"

# 使用4个GPU的张量并行部署 Llama-3.1-70B-Instruct，并指定端口为30000
python -m minisgl --model "meta-llama/Llama-3.1-70B-Instruct" --tp 4 --port 30000
```

#### 常用命令行参数

```bash
# 完整参数说明
python -m minisgl --help

# 主要参数：
# --model: HuggingFace模型路径或本地路径
# --tp: 张量并行度（GPU数量）
# --port: 服务端口（默认8000）
# --host: 服务主机地址（默认127.0.0.1）
# --cache: KV缓存策略（radix或naive，默认radix）
# --attn: 注意力后端（fa,fi 表示prefill用FlashAttention，decode用FlashInfer）
# --max-prefill-length: 分块预填充的最大长度
# --cuda-graph-max-bs: CUDA Graph的最大批次大小
```

#### 发送请求

服务启动后，你可以使用 `curl` 或任何兼容OpenAI的客户端发送请求：

```bash
# 使用 curl 发送请求
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "介绍一下大语言模型"}
    ],
    "stream": true
  }'
```

```python
# 使用 OpenAI Python SDK
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"  # Mini-SGLang不需要API密钥
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[
        {"role": "user", "content": "介绍一下大语言模型"}
    ],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### 交互式Shell

为了方便测试和演示，Mini-SGLang 提供了交互式Shell模式：

```bash
# 启动交互式Shell
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell
```

在Shell中：
- 直接输入你的问题，模型会实时生成回答
- 使用 `/reset` 命令清除对话历史并开始新会话
- Shell会自动缓存对话历史以保持上下文

![shell示例](https://lmsys.org/images/blog/minisgl/shell.png)

### Python API

Mini-SGLang 也提供了Python API，方便集成到你的应用中：

```python
from minisgl.llm import LLM

# 初始化LLM（会自动启动后台服务）
llm = LLM(model="Qwen/Qwen3-0.6B")

# 生成单个回复
response = llm.generate("介绍一下大语言模型")
print(response)

# 流式生成
for text in llm.generate_stream("介绍一下大语言模型"):
    print(text, end="", flush=True)

# 批量生成
prompts = ["介绍一下Python", "介绍一下PyTorch"]
responses = llm.generate_batch(prompts)
for resp in responses:
    print(resp)
```

## 💡 核心概念

### 1. 张量并行（Tensor Parallelism）

张量并行将模型的参数分割到多个GPU上，每个GPU只存储部分模型参数，从而支持超大模型的推理。

```bash
# 使用4个GPU进行张量并行
python -m minisgl --model "meta-llama/Llama-3.1-70B-Instruct" --tp 4
```

**工作原理：**
- 模型的每一层被水平切分到多个GPU
- 前向传播时，每个GPU计算部分结果
- 通过All-Reduce操作同步结果
- 适用于单卡无法容纳的大模型

### 2. Radix Cache（基数缓存）

Radix Cache 使用基数树（Radix Tree）数据结构管理KV缓存，自动识别和复用请求间的共享前缀。

**优势：**
- 减少重复计算：共享系统提示词或对话历史的请求可以复用KV缓存
- 节省显存：避免存储重复的KV缓存
- 提高吞吐量：特别适用于多轮对话和相似prompt的场景

```bash
# 使用 Radix Cache（默认）
python -m minisgl --model "Qwen/Qwen3-0.6B" --cache radix

# 使用朴素缓存策略（用于对比）
python -m minisgl --model "Qwen/Qwen3-0.6B" --cache naive
```

### 3. 分块预填充（Chunked Prefill）

将长输入分成多个小块分别处理，避免一次性处理超长输入导致的显存溢出。

```bash
# 设置最大预填充长度为2048 tokens
python -m minisgl --model "Qwen/Qwen3-0.6B" --max-prefill-length 2048
```

**适用场景：**
- 处理长上下文输入（如长文档问答）
- 显存受限的环境
- 需要平衡prefill和decode请求的混合批处理

### 4. CUDA Graph

CUDA Graph 捕获和重放GPU操作序列，减少CPU启动开销。

```bash
# 设置CUDA Graph最大批次大小为32
python -m minisgl --model "Qwen/Qwen3-0.6B" --cuda-graph-max-bs 32

# 禁用CUDA Graph
python -m minisgl --model "Qwen/Qwen3-0.6B" --cuda-graph-max-bs 0
```

### 5. 重叠调度（Overlap Scheduling）

将CPU的调度开销与GPU计算重叠，提高整体吞吐量。

```bash
# 默认启用，可以通过环境变量禁用（用于性能对比）
MINISGL_DISABLE_OVERLAP_SCHEDULING=1 python -m minisgl --model "Qwen/Qwen3-0.6B"
```

## 🎨 高级功能

### 支持的模型

当前支持的模型架构：

- **Llama系列**: 
  - Llama-3, Llama-3.1, Llama-3.2
  - 示例：`meta-llama/Llama-3.1-8B-Instruct`
  
- **Qwen系列**: 
  - Qwen-3
  - 示例：`Qwen/Qwen3-0.6B`, `Qwen/Qwen3-14B`, `Qwen/Qwen3-32B`

### 注意力后端

Mini-SGLang 集成了高性能注意力内核：

```bash
# 使用 FlashAttention 进行 prefill 和 decode
python -m minisgl --model "Qwen/Qwen3-0.6B" --attn fa,fa

# 使用 FlashAttention 进行 prefill，FlashInfer 进行 decode（推荐）
python -m minisgl --model "Qwen/Qwen3-0.6B" --attn fa,fi

# 使用 FlashInfer 进行 prefill 和 decode
python -m minisgl --model "Qwen/Qwen3-0.6B" --attn fi,fi
```

**推荐配置：**
- **Hopper GPU（H100/H200）**: `--attn fa,fi` （prefill用FlashAttention3，decode用FlashInfer）
- **Ampere/Ada GPU（A100/L40s）**: `--attn fa,fa` 或 `--attn fa,fi`

### 自定义采样参数

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "写一首诗"}],
    temperature=0.8,      # 温度参数（0.0-2.0），越高越随机
    top_p=0.95,          # 核采样参数
    top_k=50,            # Top-K采样参数
    max_tokens=512,      # 最大生成token数
    stream=True
)
```

## 📊 性能基准测试

### 离线推理性能

测试配置：
- **硬件**: 1x H200 GPU
- **模型**: Qwen3-0.6B, Qwen3-14B
- **请求数**: 256个序列
- **输入长度**: 100-1024 tokens（随机采样）
- **输出长度**: 100-1024 tokens（随机采样）

详见 [benchmark/offline/bench.py](../../benchmark/offline/bench.py)

![offline性能](https://lmsys.org/images/blog/minisgl/offline.png)

### 在线推理性能

测试配置：
- **硬件**: 4x H200 GPU（NVLink连接）
- **模型**: Qwen3-32B
- **数据集**: [Qwen trace](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon)（前1000个请求）

启动命令：
```bash
# Mini-SGLang
python -m minisgl --model "Qwen/Qwen3-32B" --tp 4 --cache naive

# SGLang（对比）
python3 -m sglang.launch_server --model "Qwen/Qwen3-32B" --tp 4 \
    --disable-radix --port 1919 --decode-attention flashinfer
```

详见 [benchmark/online/bench_qwen.py](../../benchmark/online/bench_qwen.py)

![online性能](https://lmsys.org/images/blog/minisgl/online.png)

### 运行基准测试

```bash
# 离线推理基准测试
cd benchmark/offline
python bench.py --model "Qwen/Qwen3-0.6B"

# 在线推理基准测试
# 1. 启动服务
python -m minisgl --model "Qwen/Qwen3-0.6B"

# 2. 运行基准测试
cd benchmark/online
python bench_simple.py
```

## 🔧 故障排除

### 常见问题

#### 1. CUDA内核编译失败

**症状**: 启动时报错 "JIT compilation failed"

**解决方案**:
```bash
# 检查CUDA Toolkit版本是否与驱动匹配
nvidia-smi  # 查看驱动支持的CUDA版本
nvcc --version  # 查看CUDA Toolkit版本

# 确保两者版本一致，如不一致，重新安装匹配的CUDA Toolkit
```

#### 2. 显存不足（OOM）

**症状**: "CUDA out of memory"

**解决方案**:
```bash
# 1. 减小最大批次大小（通过减小prefill chunk大小）
python -m minisgl --model "Qwen/Qwen3-0.6B" --max-prefill-length 1024

# 2. 使用张量并行分散显存压力
python -m minisgl --model "Qwen/Qwen3-14B" --tp 2

# 3. 减小CUDA Graph批次大小
python -m minisgl --model "Qwen/Qwen3-0.6B" --cuda-graph-max-bs 16
```

#### 3. 模型下载失败

**症状**: "Connection timeout" 或 "403 Forbidden"

**解决方案**:
```bash
# 使用镜像站点（国内用户）
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载模型到本地
git lfs install
git clone https://hf-mirror.com/Qwen/Qwen3-0.6B
python -m minisgl --model ./Qwen3-0.6B
```

#### 4. 多GPU通信失败

**症状**: "NCCL error" 或 "torch.distributed timeout"

**解决方案**:
```bash
# 检查GPU互连状态
nvidia-smi topo -m

# 确保所有GPU可见
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 增加NCCL超时时间
export NCCL_TIMEOUT=1800
```

#### 5. WSL2 GPU不可用

**症状**: WSL2中 `nvidia-smi` 报错

**解决方案**:
```powershell
# 在Windows PowerShell中更新GPU驱动
# 下载最新的NVIDIA GeForce Game Ready或Studio驱动
# 确保驱动版本 >= 510.x（支持WSL2）

# 在WSL2中验证
wsl
nvidia-smi
```

### 性能调优建议

#### 1. 选择合适的注意力后端

```bash
# Hopper架构（H100/H200）
python -m minisgl --model "..." --attn fa,fi

# Ampere架构（A100）
python -m minisgl --model "..." --attn fa,fa
```

#### 2. 调整预填充长度

```bash
# 长上下文场景
python -m minisgl --model "..." --max-prefill-length 4096

# 短上下文高吞吐场景
python -m minisgl --model "..." --max-prefill-length 512
```

#### 3. 优化CUDA Graph设置

```bash
# 高吞吐场景（大批次）
python -m minisgl --model "..." --cuda-graph-max-bs 64

# 低延迟场景（小批次）
python -m minisgl --model "..." --cuda-graph-max-bs 8
```

### 获取帮助

如果遇到其他问题：

1. **查看日志**: 启动服务时会输出详细的日志信息
2. **提交Issue**: [GitHub Issues](https://github.com/sgl-project/mini-sglang/issues)
3. **查看文档**: [详细功能文档](../features.md) | [系统架构文档](../structures.md)

## 📚 更多资源

- **[详细功能说明](../features.md)**: 探索所有可用功能和命令行参数
- **[系统架构设计](../structures.md)**: 深入了解Mini-SGLang的设计和数据流
- **[核心概念解析](./concepts.md)**: 理解关键技术和优化原理
- **[开发者指南](./developer.md)**: 学习如何修改和扩展Mini-SGLang

## 🤝 贡献

欢迎贡献代码、报告问题或提出建议！请查看 [贡献指南](https://github.com/sgl-project/mini-sglang/blob/main/CONTRIBUTING.md)。

## 📄 许可证

本项目采用 Apache 2.0 许可证 - 详见 [LICENSE](https://github.com/sgl-project/mini-sglang/blob/main/LICENSE) 文件。

---

**祝你使用愉快！如有问题欢迎提Issue！** 🚀


