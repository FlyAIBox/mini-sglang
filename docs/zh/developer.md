# Mini-SGLang 开发者指南

本指南面向希望深入理解、修改或扩展Mini-SGLang的开发者。

## 目录

- [快速开始](#快速开始)
- [代码结构](#代码结构)
- [核心模块详解](#核心模块详解)
- [添加新模型](#添加新模型)
- [性能调优](#性能调优)
- [调试技巧](#调试技巧)
- [常见问题](#常见问题)

---

## 快速开始

### 开发环境设置

```bash
# 1. 克隆仓库
git clone https://github.com/sgl-project/mini-sglang.git
cd mini-sglang

# 2. 创建开发环境
uv venv --python=3.12
source .venv/bin/activate

# 3. 安装为可编辑模式（修改代码立即生效）
uv pip install -e ".[dev]"

# 4. 安装开发工具
uv pip install pytest black isort mypy ruff
```

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/core/test_scheduler.py

# 运行单个测试函数
pytest tests/core/test_scheduler.py::test_basic_scheduling
```

### 代码风格

项目使用以下工具保持代码质量：

```bash
# 格式化代码
black python/minisgl/
isort python/minisgl/

# 类型检查
mypy python/minisgl/

# Linting
ruff check python/minisgl/
```

---

## 代码结构

```
mini-sglang/
├── python/minisgl/          # 主要源代码
│   ├── core.py             # 核心数据结构（Req, Batch, Context）
│   ├── distributed/         # 分布式通信（TP）
│   │   ├── info.py         # TP信息管理
│   │   └── impl.py         # All-Reduce/All-Gather实现
│   ├── attention/           # 注意力机制
│   │   ├── base.py         # 注意力后端基类
│   │   ├── fa.py           # FlashAttention后端
│   │   └── fi.py           # FlashInfer后端
│   ├── kvcache/            # KV Cache管理
│   │   ├── base.py         # KV Cache基类
│   │   ├── naive_manager.py # 朴素缓存管理器
│   │   └── radix_manager.py # Radix Tree缓存管理器
│   ├── scheduler/          # 调度器
│   │   ├── scheduler.py    # 主调度逻辑
│   │   ├── prefill.py      # Prefill调度
│   │   └── decode.py       # Decode调度
│   ├── engine/             # 推理引擎
│   │   ├── engine.py       # Engine类（模型执行）
│   │   └── graph.py        # CUDA Graph管理
│   ├── models/             # 模型实现
│   │   ├── llama.py        # Llama模型
│   │   └── qwen3.py        # Qwen3模型
│   ├── layers/             # 模型层实现
│   │   ├── linear.py       # 线性层（支持TP）
│   │   ├── attention.py    # 注意力层
│   │   └── rotary.py       # RoPE位置编码
│   ├── server/             # API服务器
│   │   ├── api_server.py   # FastAPI服务器
│   │   └── launch.py       # 启动逻辑
│   ├── tokenizer/          # Tokenizer worker
│   └── kernel/             # CUDA kernels
└── tests/                  # 测试代码
    ├── core/
    ├── kernel/
    └── misc/
```

---

## 核心模块详解

### 1. 核心数据结构 (`core.py`)

#### Req（请求对象）

表示单个推理请求的完整状态。

**关键属性：**
- `input_ids`: 输入token序列（CPU tensor）
- `cached_len`: 已缓存的token数量
- `device_len`: 当前总token数
- `max_device_len`: 最大允许token数

**状态转换：**

```python
# 初始状态
req = Req(
    input_ids=torch.tensor([1, 2, 3, 4, 5]),
    cached_len=0,
    output_len=10,
    ...
)
# cached_len=0, device_len=5, max_device_len=15

# Prefill后
req.complete_one()
# cached_len=5, device_len=6

# 第1次decode
next_token = sample(logits)
req.append_host(next_token)
req.complete_one()
# cached_len=6, device_len=7, input_ids=[1,2,3,4,5,6]
```

#### Batch（批处理对象）

组合多个请求进行并行处理。

```python
# 创建decode batch
batch = Batch(
    reqs=[req1, req2, req3],
    phase="decode"
)

# 调度器准备输入
scheduler.prepare_inputs(batch)
# batch.input_ids: [batch_size, 1]
# batch.out_loc: [batch_size]

# Engine执行
with ctx.forward_batch(batch):
    logits = engine.forward()
```

#### Context（全局上下文）

存储全局配置和状态。

```python
# 初始化
ctx = Context(
    page_size=16,
    attn_backend=flash_attn_backend
)
set_global_ctx(ctx)

# 在模型中使用
class MyModel:
    def forward(self, x):
        ctx = get_global_ctx()
        batch = ctx.batch
        if batch.is_prefill:
            # prefill逻辑
        else:
            # decode逻辑
```

### 2. 分布式通信 (`distributed/`)

#### 设置TP信息

```python
from minisgl.distributed import set_tp_info, get_tp_info

# 每个进程初始化时
set_tp_info(rank=0, size=4)

# 在代码中使用
tp_info = get_tp_info()
if tp_info.is_primary():
    print("I am rank 0!")
```

#### 使用通信原语

```python
from minisgl.distributed import DistributedCommunicator

comm = DistributedCommunicator()

# All-Reduce: 求和
tensor = torch.randn(10, 20).cuda()
result = comm.all_reduce(tensor)
# result在所有GPU上相同

# All-Gather: 拼接
tensor = torch.randn(2, 10).cuda()
result = comm.all_gather(tensor)
# result.shape = [2*tp_size, 10]
```

### 3. 调度器 (`scheduler/`)

调度器是系统的大脑，负责：
- 接收新请求
- 管理请求状态
- 组batch
- 调用Engine执行
- 处理输出

**主要流程：**

```python
class Scheduler:
    def event_loop(self):
        while True:
            # 1. 接收新请求
            new_reqs = self.recv_requests()
            self.waiting.extend(new_reqs)
            
            # 2. 调度prefill
            prefill_batch = self.schedule_prefill()
            if prefill_batch:
                self.engine.forward(prefill_batch)
            
            # 3. 调度decode
            decode_batch = self.schedule_decode()
            if decode_batch:
                outputs = self.engine.forward(decode_batch)
                self.process_outputs(outputs)
            
            # 4. 发送完成的结果
            self.send_outputs()
```

### 4. Engine (`engine/`)

Engine负责实际的模型执行。

```python
class Engine:
    def __init__(self, config):
        # 加载模型
        self.model = load_model(config.model_path)
        
        # 初始化KV Cache
        self.kv_cache = RadixCacheManager(...)
        
        # 初始化CUDA Graph
        if config.cuda_graph_max_bs > 0:
            self.graph_runner = CUDAGraphRunner(...)
    
    def forward(self, batch: Batch) -> torch.Tensor:
        # 准备输入
        input_ids = batch.input_ids.cuda()
        
        # 准备注意力元数据
        self.attn_backend.prepare_metadata(batch)
        
        # 执行模型
        with ctx.forward_batch(batch):
            if batch.is_decode and self.use_cuda_graph:
                logits = self.graph_runner.run(batch)
            else:
                logits = self.model(input_ids)
        
        return logits
```

---

## 添加新模型

以添加新的LLama变体为例：

### 1. 创建模型配置

```python
# python/minisgl/models/my_model.py

from minisgl.models.base import BaseModelForCausalLM
from minisgl.models.config import ModelConfig

@dataclass
class MyModelConfig(ModelConfig):
    """你的模型配置"""
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    # ... 其他配置
```

### 2. 实现模型类

```python
class MyModel(BaseModelForCausalLM):
    def __init__(self, config: MyModelConfig):
        super().__init__(config)
        
        # 嵌入层
        self.embed = Embedding(
            config.vocab_size,
            config.hidden_size
        )
        
        # Transformer层
        self.layers = nn.ModuleList([
            MyDecoderLayer(config)
            for _ in range(config.num_layers)
        ])
        
        # 输出层
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False
        )
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 获取全局上下文
        ctx = get_global_ctx()
        batch = ctx.batch
        
        # 嵌入
        x = self.embed(input_ids)
        
        # 逐层forward
        for layer in self.layers:
            x = layer(x)
        
        # 输出
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits
```

### 3. 注册模型

```python
# python/minisgl/models/__init__.py

from minisgl.utils.registry import ModelRegistry
from .my_model import MyModel, MyModelConfig

ModelRegistry.register("my-model", MyModel, MyModelConfig)
```

### 4. 使用新模型

```bash
python -m minisgl --model "path/to/my-model"
```

---

## 性能调优

### 1. Profiling

使用PyTorch Profiler分析性能：

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    for _ in range(10):
        engine.forward(batch)

# 输出结果
print(prof.key_averages().table(
    sort_by="cuda_time_total", row_limit=20
))

# 保存trace
prof.export_chrome_trace("trace.json")
# 在 chrome://tracing 中查看
```

### 2. 优化建议

**减少CPU开销：**
```python
# 使用CUDA Graph
engine = Engine(config, cuda_graph_max_bs=32)

# 启用重叠调度
# （默认启用，不需要额外配置）
```

**优化显存使用：**
```python
# 减小page_size降低碎片
config.page_size = 16  # 默认是16

# 使用更小的max_prefill_length
config.max_prefill_length = 1024
```

**提高吞吐量：**
```python
# 使用PyNCCL（多GPU场景）
enable_pynccl_distributed(tp_info, cpu_group, max_bytes)

# 调整batch size
config.max_batch_size = 64  # 根据显存调整
```

### 3. 性能监控

添加自定义metrics：

```python
import time

class Scheduler:
    def __init__(self):
        self.metrics = {
            "prefill_time": [],
            "decode_time": [],
            "batch_size": []
        }
    
    def forward_prefill(self, batch):
        start = time.time()
        result = self.engine.forward(batch)
        elapsed = time.time() - start
        
        self.metrics["prefill_time"].append(elapsed)
        self.metrics["batch_size"].append(batch.size)
        
        return result
    
    def print_stats(self):
        import numpy as np
        print(f"Avg prefill time: {np.mean(self.metrics['prefill_time']):.3f}s")
        print(f"Avg batch size: {np.mean(self.metrics['batch_size']):.1f}")
```

---

## 调试技巧

### 1. 日志配置

```python
from minisgl.utils.logger import get_logger

logger = get_logger(__name__)

# 设置日志级别
logger.setLevel("DEBUG")

# 在关键位置添加日志
logger.debug(f"Batch size: {batch.size}, phase: {batch.phase}")
logger.info(f"Generated token: {token_id}")
```

### 2. 断点调试

```python
# 使用pdb
import pdb; pdb.set_trace()

# 或使用ipdb（更友好）
import ipdb; ipdb.set_trace()

# 条件断点
if batch.size > 10:
    import pdb; pdb.set_trace()
```

### 3. 可视化

可视化Radix Tree：

```python
def visualize_radix_tree(tree):
    """打印Radix Tree结构"""
    def _print_node(node, prefix="", is_last=True):
        print(prefix + ("└── " if is_last else "├── ") + str(node.key))
        children = list(node.children.values())
        for i, child in enumerate(children):
            _print_node(
                child,
                prefix + ("    " if is_last else "│   "),
                i == len(children) - 1
            )
    
    _print_node(tree.root)
```

### 4. 单元测试

编写测试用例：

```python
# tests/test_my_feature.py

import pytest
import torch
from minisgl.core import Req, Batch, SamplingParams

def test_req_state_transition():
    """测试Req状态转换"""
    req = Req(
        input_ids=torch.tensor([1, 2, 3]),
        cached_len=0,
        output_len=5,
        table_idx=0,
        uid=0,
        sampling_params=SamplingParams(),
        cache_handle=None
    )
    
    assert req.device_len == 3
    assert req.max_device_len == 8
    assert req.remain_len == 5
    
    # 完成一次decode
    req.complete_one()
    assert req.cached_len == 3
    assert req.device_len == 4
    assert req.remain_len == 4

@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
def test_batch_processing(batch_size):
    """测试不同batch size"""
    reqs = [create_dummy_req() for _ in range(batch_size)]
    batch = Batch(reqs=reqs, phase="decode")
    # ... 测试逻辑
```

---

## 常见问题

### Q1: 如何添加新的采样方法？

修改 `engine/sample.py`：

```python
def my_sampling(logits, temperature, my_param):
    """自定义采样方法"""
    # 应用temperature
    logits = logits / temperature
    
    # 自定义逻辑
    # ...
    
    # 采样
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token
```

### Q2: 如何支持新的注意力后端？

实现 `BaseAttnBackend` 接口：

```python
from minisgl.attention.base import BaseAttnBackend

class MyAttentionBackend(BaseAttnBackend):
    def prepare_metadata(self, batch: Batch):
        """准备注意力计算所需的元数据"""
        # 实现你的逻辑
        pass
    
    def forward(self, q, k, v, metadata):
        """执行注意力计算"""
        # 实现你的注意力kernel
        pass
```

### Q3: 显存不足怎么办？

**短期解决方案：**
```bash
# 减小batch size
python -m minisgl --model "..." --max-batch-size 8

# 减小prefill chunk size
python -m minisgl --model "..." --max-prefill-length 512

# 禁用CUDA Graph
python -m minisgl --model "..." --cuda-graph-max-bs 0
```

**长期解决方案：**
- 使用更多GPU（张量并行）
- 实现KV Cache量化
- 使用PagedAttention优化显存分配

### Q4: 如何调试多GPU问题？

```bash
# 设置NCCL日志
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 单独运行每个rank方便调试
# Terminal 1
CUDA_VISIBLE_DEVICES=0 python -m minisgl --model "..." --tp 2

# Terminal 2  
CUDA_VISIBLE_DEVICES=1 python -m minisgl --model "..." --tp 2
```

### Q5: 性能不如预期？

**检查清单：**
1. ✅ 是否启用了CUDA Graph？
2. ✅ 是否使用了合适的注意力后端？（Hopper用FA3+FI）
3. ✅ 多GPU是否启用了PyNCCL？
4. ✅ batch size是否足够大？
5. ✅ 是否禁用了重叠调度？（应该启用）

**Benchmark对比：**
```bash
# 运行官方benchmark
cd benchmark/offline
python bench.py --model "Qwen/Qwen3-0.6B"

# 与SGLang对比
# 查看 throughput (tokens/s) 和 latency (ms)
```

---

## 贡献指南

欢迎贡献！提交PR前请确保：

1. **代码质量：**
   ```bash
   black python/minisgl/
   isort python/minisgl/
   mypy python/minisgl/
   ruff check python/minisgl/
   ```

2. **测试通过：**
   ```bash
   pytest tests/
   ```

3. **添加文档：**
   - 在代码中添加详细的中文注释
   - 更新README和相关文档
   - 添加使用示例

4. **性能验证：**
   - 运行benchmark确保没有性能退化
   - 如果是优化PR，提供性能对比数据

---

## 学习资源

### 论文

- [SGLang](https://arxiv.org/abs/2312.07104): Radix Attention原理
- [FlashAttention](https://arxiv.org/abs/2205.14135): 高效注意力计算
- [FlashAttention-2](https://arxiv.org/abs/2307.08691): FlashAttention改进
- [NanoFlow](https://arxiv.org/abs/2408.12757): Overlap Scheduling
- [Sarathi-Serve](https://arxiv.org/abs/2403.02310): Chunked Prefill

### 相关项目

- [SGLang](https://github.com/sgl-project/sglang): 完整版实现
- [vLLM](https://github.com/vllm-project/vllm): PagedAttention
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM): NVIDIA官方推理引擎

### 博客和教程

- [LMSYS Blog](https://lmsys.org/blog/): SGLang团队的技术博客
- [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/): 推理性能分析

---

有问题或建议？欢迎提Issue或加入我们的讨论！🚀

