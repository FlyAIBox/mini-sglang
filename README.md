<p align="center">
<img width="400" src="/assets/logo.png">
</p>

# Mini-SGLang

A **lightweight yet high-performance** inference framework for Large Language Models.

---

Mini-SGLang is a compact implementation of [SGLang](https://github.com/sgl-project/sglang), designed to demystify the complexities of modern LLM serving systems. With a compact codebase of **~5,000 lines of Python**, it serves as both a capable inference engine and a transparent reference for researchers and developers.

## ✨ Key Features

- **High Performance**: Achieves state-of-the-art throughput and latency with advanced optimizations.
- **Lightweight & Readable**: A clean, modular, and fully type-annotated codebase that is easy to understand and modify.
- **Advanced Optimizations**:
  - **Radix Cache**: Reuses KV cache for shared prefixes across requests.
  - **Chunked Prefill**: Reduces peak memory usage for long-context serving.
  - **Overlap Scheduling**: Hides CPU scheduling overhead with GPU computation.
  - **Tensor Parallelism**: Scales inference across multiple GPUs.
  - **Optimized Kernels**: Integrates **FlashAttention** and **FlashInfer** for maximum efficiency.
  - ...

## 🚀 Quick Start

> **⚠️ Platform Support**: Mini-SGLang currently supports **Linux only** (x86_64 and aarch64). Windows and macOS are not supported due to dependencies on Linux-specific CUDA kernels (`sgl-kernel`, `flashinfer`). We recommend using [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) on Windows or Docker for cross-platform compatibility.

### 1. System Requirements

**Hardware Requirements:**
- **GPU**: NVIDIA GPU with CUDA Compute Capability 7.0+ (V100, T4, A100, H100, RTX 3090, RTX 4090, etc.)
- **Memory**: 
  - For small models (0.6B-7B): 8GB+ GPU memory
  - For medium models (7B-30B): 24GB+ GPU memory (or multi-GPU with Tensor Parallelism)
  - For large models (30B+): Multiple GPUs with 40GB+ each
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 16GB+ system memory

**Software Requirements:**
- **OS**: Linux (Ubuntu 20.04+, CentOS 7+, or other modern distributions)
- **Python**: 3.10, 3.11, or 3.12
- **CUDA**: 11.8+ or 12.1+ (must match your driver version)
- **NVIDIA Driver**: 525+ (for CUDA 12.x) or 450+ (for CUDA 11.x)
- **Git**: For cloning the repository

### 2. Check Your Environment

Before installation, verify your system meets the requirements:

```bash
# Check NVIDIA GPU and driver version
nvidia-smi

# Check CUDA version (should match driver)
nvcc --version  # If CUDA toolkit is installed

# Check Python version
python --version  # Should be 3.10, 3.11, or 3.12
```

### 3. Environment Setup

We recommend using `uv` for a fast and reliable installation (note that `uv` does not conflict with `conda`).

**Option A: Using uv (Recommended)**

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment (Python 3.12 recommended)
uv venv --python=3.12
source .venv/bin/activate  # On Linux/Mac
# .venv\Scripts\activate   # On Windows (PowerShell)
```

**Option B: Using venv**

```bash
# Create a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# .venv\Scripts\activate   # On Windows (PowerShell)
```

**Option C: Using conda**

```bash
# Create a conda environment
conda create -n minisgl python=3.12
conda activate minisgl
```

### 4. Installation

**Prerequisites**: Mini-SGLang relies on CUDA kernels that are JIT-compiled. Ensure you have the **NVIDIA CUDA Toolkit** installed and that its version matches your driver's version.

**Step-by-step Installation:**

```bash
# 1. Clone the repository
git clone https://github.com/sgl-project/mini-sglang.git
cd mini-sglang

# 2. Activate your virtual environment
source .venv/bin/activate  # or use your conda environment

# 3. Install Mini-SGLang
# If using uv:
uv pip install -e .

# If using pip:
pip install -e .

# 4. Verify installation
python -c "import minisgl; print('Mini-SGLang installed successfully!')"
```

**Installation Time**: 
- First-time installation: 5-15 minutes (depending on network speed)
- CUDA kernels will be JIT-compiled on first use (adds 1-3 minutes on first run)

**Troubleshooting Installation Issues:**

<details>
<summary><b>🔧 CUDA Toolkit not found</b></summary>

If you get errors about missing CUDA toolkit:

1. **Install CUDA Toolkit**:
   ```bash
   # Ubuntu/Debian
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   sudo apt-get install cuda-toolkit-12-1
   
   # Or download from: https://developer.nvidia.com/cuda-downloads
   ```

2. **Set environment variables**:
   ```bash
   export CUDA_HOME=/usr/local/cuda
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   ```

3. **Add to ~/.bashrc** to make permanent:
   ```bash
   echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
   echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

</details>

<details>
<summary><b>🔧 PyTorch/CUDA version mismatch</b></summary>

If you get CUDA version mismatches:

1. **Check your CUDA version**:
   ```bash
   nvidia-smi  # See CUDA Version in top-right
   ```

2. **Install matching PyTorch**:
   ```bash
   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Verify PyTorch CUDA**:
   ```bash
   python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
   ```

</details>

<details>
<summary><b>🔧 Out of memory during installation</b></summary>

If installation fails due to memory issues:

```bash
# Reduce pip's cache size
pip install -e . --no-cache-dir

# Or install with limited parallelism
MAX_JOBS=4 pip install -e .
```

</details>

<details>
<summary><b>💡 Installing on Windows (WSL2)</b></summary>

Since Mini-SGLang requires Linux-specific dependencies, Windows users should use WSL2:

1. **Install WSL2** (if not already installed):
   ```powershell
   # In PowerShell (as Administrator)
   wsl --install
   ```

2. **Install CUDA on WSL2**:
   - Follow [NVIDIA's WSL2 CUDA guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
   - Ensure your Windows GPU drivers support WSL2

3. **Install Mini-SGLang in WSL2**:
   ```bash
   # Inside WSL2 terminal
   git clone https://github.com/sgl-project/mini-sglang.git
   cd mini-sglang && uv venv --python=3.12 && source .venv/bin/activate
   uv pip install -e .
   ```

4. **Access from Windows**: The server will be accessible at `http://localhost:8000` from Windows browsers and applications.

</details>

### 5. First Run - Quick Test

Start with a small model to verify everything works:

```bash
# Launch the server with a small model (0.6B parameters)
python -m minisgl --model "Qwen/Qwen3-0.6B"

# The server will:
# 1. Download the model from HuggingFace (first time only, ~1.2GB)
# 2. Load the model into GPU memory
# 3. Start the API server on http://localhost:8000
# 4. Display "Server started successfully" when ready
```

**Expected output:**
```
[INFO] Loading model: Qwen/Qwen3-0.6B
[INFO] Initializing tokenizer...
[INFO] Initializing scheduler worker (TP Rank 0/1)...
[INFO] Initializing engine...
[INFO] Loading weights...
[INFO] Starting API server on http://0.0.0.0:8000
[INFO] Server started successfully!
```

**Test the server:**

```bash
# In a new terminal, test with curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "Hello! How are you?"}
    ]
  }'
```

### 6. Online Serving (Production Use)

Launch an OpenAI-compatible API server with various configurations.

**Basic Usage:**

```bash
# Single GPU with default settings
python -m minisgl --model "Qwen/Qwen3-0.6B"

# Specify port and host
python -m minisgl --model "Qwen/Qwen3-0.6B" --host 0.0.0.0 --port 8000

# Use Radix Cache for better performance (default)
python -m minisgl --model "Qwen/Qwen3-0.6B" --cache radix

# Disable Radix Cache (for debugging or simpler behavior)
python -m minisgl --model "Qwen/Qwen3-0.6B" --cache naive
```

**Multi-GPU with Tensor Parallelism:**

```bash
# Deploy on 2 GPUs
python -m minisgl --model "Qwen/Qwen3-14B" --tp 2

# Deploy on 4 GPUs with custom port
python -m minisgl --model "meta-llama/Llama-3.1-70B-Instruct" --tp 4 --port 30000

# Deploy on 8 GPUs (for very large models)
python -m minisgl --model "meta-llama/Llama-3.1-405B" --tp 8
```

**Advanced Options:**

```bash
# Adjust memory and batch size
python -m minisgl \
  --model "Qwen/Qwen3-14B" \
  --tp 2 \
  --max-batch-size 256 \
  --max-total-tokens 8192

# Use specific attention backend
python -m minisgl \
  --model "Qwen/Qwen3-0.6B" \
  --attention-backend flashinfer  # or flashattention

# Adjust KV cache page size
python -m minisgl \
  --model "Qwen/Qwen3-0.6B" \
  --page-size 16  # default is 16
```

**Common Command-Line Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | Required | HuggingFace model ID or local path |
| `--tp` | 1 | Number of GPUs for Tensor Parallelism |
| `--host` | "0.0.0.0" | Server host address |
| `--port` | 8000 | Server port number |
| `--cache` | "radix" | Cache strategy: "radix" or "naive" |
| `--attention-backend` | "flashinfer" | "flashinfer" or "flashattention" |
| `--max-batch-size` | 128 | Maximum batch size |
| `--max-total-tokens` | 4096 | Maximum total tokens in a batch |
| `--page-size` | 16 | KV cache page size |

See `python -m minisgl --help` for all options.

**Using the API:**

Once the server is running, you can send requests using:

```bash
# Using curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Explain quantum computing"}],
    "max_tokens": 512,
    "temperature": 0.7
  }'

# Using Python with OpenAI library
pip install openai
python examples/client.py  # See examples/ directory
```

```python
# Python example
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Mini-SGLang doesn't require authentication
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[
        {"role": "user", "content": "What is the meaning of life?"}
    ],
    temperature=0.7,
    max_tokens=512
)

print(response.choices[0].message.content)
```

### 7. Interactive Shell

Chat with your model directly in the terminal by adding the `--shell` flag.

```bash
# Start interactive shell
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell

# With multi-GPU
python -m minisgl --model "Qwen/Qwen3-14B" --tp 2 --shell
```

![shell-example](https://lmsys.org/images/blog/minisgl/shell.png)

**Shell Commands:**
- Type your message and press Enter to chat
- `/reset` - Clear chat history and start over
- `/exit` or Ctrl+D - Exit the shell
- `/help` - Show help message

### 8. Using Mini-SGLang as a Python Library

You can also use Mini-SGLang directly in Python without starting a server:

```python
from minisgl.llm import LLM
from minisgl.core import SamplingParams

# Initialize the LLM
llm = LLM(model="Qwen/Qwen3-0.6B", tp=1)

# Single prompt
output = llm.generate(
    "Explain how transformers work in machine learning",
    SamplingParams(temperature=0.8, max_tokens=256)
)
print(output)

# Batch prompts
prompts = [
    "What is AI?",
    "Explain quantum computing",
    "How do neural networks work?"
]
outputs = llm.generate_batch(
    prompts,
    SamplingParams(temperature=0.7, max_tokens=128)
)

for prompt, output in zip(prompts, outputs):
    print(f"Q: {prompt}")
    print(f"A: {output}\n")
```

### 9. Stopping the Server

To gracefully stop the server:

```bash
# Press Ctrl+C in the terminal running the server
# The server will:
# 1. Stop accepting new requests
# 2. Wait for current requests to complete
# 3. Clean up resources and exit
```

**Force stop** (if graceful shutdown hangs):
```bash
# Find the process
ps aux | grep minisgl

# Kill it
kill -9 <PID>
```

## Benchmark

### Offline inference

See [bench.py](./benchmark/offline/bench.py) for more details. Set `MINISGL_DISABLE_OVERLAP_SCHEDULING=1` for ablation study on overlap scheduling.

Test Configuration:

- Hardware: 1xH200 GPU.
- Model: Qwen3-0.6B, Qwen3-14B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100-1024 tokens
- Output Length: Randomly sampled between 100-1024 tokens

![offline](https://lmsys.org/images/blog/minisgl/offline.png)

### Online inference

See [benchmark_qwen.py](./benchmark/online/bench_qwen.py) for more details.

Test Configuration:

- Hardware: 4xH200 GPU, connected by NVLink.
- Model: Qwen3-32B
- Dataset: [Qwen trace](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon/blob/main/qwen_traceA_blksz_16.jsonl), replaying first 1000 requests.

Launch command:

```bash
# Mini-SGLang
python -m minisgl --model "Qwen/Qwen3-32B" --tp 4 --cache naive

# SGLang
python3 -m sglang.launch_server --model "Qwen/Qwen3-32B" --tp 4 \
    --disable-radix --port 1919 --decode-attention flashinfer
```

![online](https://lmsys.org/images/blog/minisgl/online.png)

## 📚 Learn More

### 📖 Documentation Index

- **[📑 Complete Documentation Index](./DOCUMENTATION_INDEX.md)**: Find all documentation organized by topic and use case.

### English Documentation

- **[📦 Dependencies Summary](./DEPENDENCIES_SUMMARY.md)**: Quick reference for package versions and installation commands.
- **[📋 Dependencies Guide](./DEPENDENCIES.md)**: Complete guide to all dependencies, versions, and installation troubleshooting.
- **[📊 Dependencies Version Table](./DEPENDENCIES_VERSION_TABLE.md)**: Version compatibility matrix and upgrade suggestions.
- **[✨ Detailed Features](./docs/features.md)**: Explore all available features and command-line arguments.
- **[🏗️ System Architecture](./docs/structures.md)**: Dive deep into the design and data flow of Mini-SGLang.

### 中文文档 (Chinese Documentation)

- **[中文安装和使用指南](./docs/zh/README_ZH.md)**: 详细的中文安装、配置和使用教程
- **[核心概念详解](./docs/zh/concepts.md)**: 深入讲解大模型推理的核心技术和原理
- **[开发者指南](./docs/zh/developer.md)**: 面向开发者的代码结构、调试技巧和贡献指南
- **[文档总结](./docs/zh/SUMMARY.md)**: 中文文档和代码注释的完整说明
