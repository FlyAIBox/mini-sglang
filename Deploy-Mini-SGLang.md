### 1. 系统环境

```bash
# cat /etc/lsb-release 
DISTRIB_ID=Ubuntu
DISTRIB_RELEASE=22.04
DISTRIB_CODENAME=jammy
DISTRIB_DESCRIPTION="Ubuntu 22.04.5 LTS"
```

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

操作

```bash
(base) root@user-u7ifnl4w-7afc0e87-b772-49e3-9a42-e246e704736a:~# nvidia-smi
Mon Jan 26 15:23:47 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.133.20             Driver Version: 570.133.20     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H100 80GB HBM3          On  |   00000000:9B:00.0 Off |                    0 |
| N/A   32C    P0             69W /  700W |       0MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
(base) root@user-u7ifnl4w-7afc0e87-b772-49e3-9a42-e246e704736a:~# nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Oct_29_23:50:19_PDT_2024
Cuda compilation tools, release 12.6, V12.6.85
Build cuda_12.6.r12.6/compiler.35059454_0
(base) root@user-u7ifnl4w-7afc0e87-b772-49e3-9a42-e246e704736a:~# python --version
Python 3.12.11
```

### 3. 环境设置

我们推荐使用 `uv` 进行快速可靠的安装（注意 `uv` 不会与 `conda` 冲突）。

**选项A：使用uv（推荐）**

```bash
# 如果尚未安装，先安装uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 创建虚拟环境（推荐Python 3.12）
uv venv --python=3.12
source .venv/bin/activate  # Linux/Mac
```

### 4. 安装

**前置条件**：Mini-SGLang 依赖于JIT编译的CUDA内核。请确保已安装 **NVIDIA CUDA Toolkit**，且其版本与驱动版本匹配。

**分步安装：**

```bash
# 1. 克隆仓库
git clone https://github.com/FlyAIBox/mini-sglang.git
cd mini-sglang

# 2. 安装Mini-SGLang
# 如果使用uv：
uv pip install -e .

# 3. 验证安装
python -c "import minisgl; print('Mini-SGLang安装成功！')"
```

**安装时间**：

- 首次安装：5-15分钟（取决于网络速度）
- CUDA内核将在首次使用时进行JIT编译（首次运行时增加1-3分钟）

### 5. 首次运行 - 快速测试

下载模型

```bash
uv pip install modelscope

mkdir -p /workspace/model/Qwen3-0.6B

modelscope download --model Qwen/Qwen3-0.6B --local_dir /workspace/model/Qwen3-0.6B
```

从小模型开始验证一切正常：

```bash
# python -m minisgl --model /workspace/model/Qwen3-0.6B --host 0.0.0.0 --port 8000 --tp 1
[2026-01-26|17:18:00] INFO     Parsed arguments:
ServerArgs(model_path='/workspace/model/Qwen3-0.6B', tp_info=DistributedInfo(rank=0, size=1), dtype=torch.bfloat16, max_running_req=256, attention_backend='auto', cuda_graph_bs=None, cuda_graph_max_bs=None, page_size=1, memory_ratio=0.9, distributed_timeout=60.0, use_dummy_weight=False, use_pynccl=True, max_seq_len_override=None, num_page_override=None, max_extend_tokens=8192, cache_type='radix', offline_mode=False, _unique_suffix='.pid=21792', server_host='0.0.0.0', server_port=8000, num_tokenizer=0, silent_output=False)
[2026-01-26|17:18:03|initializer] INFO     Tokenize server 0 is ready
[2026-01-26|17:18:03|core|rank=0] INFO     Free memory before loading model: 78.61 GiB
[2026-01-26|17:18:04|core|rank=0] INFO     Allocating 648224 pages for KV cache, K + V = 69.24 GiB
[2026-01-26|17:18:04|core|rank=0] INFO     Auto-selected attention backend: fa,fi
[2026-01-26|17:18:04|core|rank=0] INFO     Using hybrid attention backend: prefill=fa, decode=fi
[2026-01-26|17:18:04|core|rank=0] INFO     Free memory after initialization: 7.69 GiB
[2026-01-26|17:18:04|core|rank=0] INFO     Start capturing CUDA graphs with sizes: [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160]
[2026-01-26|17:18:04|core|rank=0] INFO     Free GPU memory before capturing CUDA graphs: 7.58 GiB
Capturing graphs: bs = 1   | avail_mem = 7.30 GiB: 100%|████████████████████████████| 23/23 [00:02<00:00,  8.12batch/s]
[2026-01-26|17:18:07|core|rank=0] INFO     Free GPU memory after capturing CUDA graphs: 7.29 GiB
[2026-01-26|17:18:08|core|rank=0] INFO     Scheduler is idle, waiting for new reqs...
[2026-01-26|17:18:08|initializer] INFO     Scheduler is ready
[2026-01-26|17:18:08|FrontendAPI] INFO     API server is ready to serve on 0.0.0.0:8000
INFO:     Started server process [21792]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)


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

| 参数                  | 默认值       | 说明                           |
| --------------------- | ------------ | ------------------------------ |
| `--model`             | 必需         | HuggingFace模型ID或本地路径    |
| `--tp`                | 1            | 张量并行使用的GPU数量          |
| `--host`              | "0.0.0.0"    | 服务器主机地址                 |
| `--port`              | 8000         | 服务器端口号                   |
| `--cache`             | "radix"      | 缓存策略："radix"或"naive"     |
| `--attention-backend` | "flashinfer" | "flashinfer"或"flashattention" |
| `--max-batch-size`    | 128          | 最大批处理大小                 |
| `--max-total-tokens`  | 4096         | 批处理中的最大总token数        |
| `--page-size`         | 16           | KV缓存页大小                   |

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