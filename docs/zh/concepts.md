# Mini-SGLang 核心概念详解

本文档深入讲解 Mini-SGLang 中的核心概念和关键技术，帮助初学者理解大模型推理系统的工作原理。

## 目录

- [基础概念](#基础概念)
- [系统架构](#系统架构)
- [关键技术](#关键技术)
- [性能优化](#性能优化)
- [数据流程](#数据流程)

---

## 基础概念

### 1. 什么是大语言模型推理？

**训练 vs 推理**

- **训练（Training）**: 使用大量数据调整模型参数，让模型学习语言知识。这个过程需要大量计算资源和时间（数周到数月）。

- **推理（Inference）**: 使用训练好的模型生成文本。给定一个输入（prompt），模型逐个生成输出token。这是用户实际使用模型的阶段。

**自回归生成**

大语言模型采用自回归（Auto-regressive）方式生成文本：

```
输入: "今天天气"
步骤1: 模型生成 "很" (输入变为: "今天天气很")
步骤2: 模型生成 "好" (输入变为: "今天天气很好")
步骤3: 模型生成 "，" (输入变为: "今天天气很好，")
...依此类推，直到生成结束符号或达到最大长度
```

每一步都依赖前面所有步骤的输出，这使得生成过程无法并行化。

### 2. Token和Tokenization

**什么是Token？**

Token 是模型处理文本的基本单位。一个token可能是：
- 一个完整的单词（如 "hello"）
- 一个汉字（如 "我"）
- 一个子词（如 "ing"）
- 一个标点符号（如 "，"）

**Tokenization（分词）过程：**

```python
# 原始文本
text = "Hello, world! 你好世界"

# Tokenization后（示例）
tokens = [15496, 11, 1917, 0, 20001, 19082, 13995, 244]
# 每个数字代表词汇表中的一个token

# Detokenization（逆向转换）
decoded_text = "Hello, world! 你好世界"
```

### 3. KV Cache（键值缓存）

**为什么需要KV Cache？**

在Transformer模型的注意力机制中，每次生成新token时都需要计算与之前所有token的注意力。为了避免重复计算，我们可以缓存之前计算的结果。

**工作原理：**

```
不使用KV Cache:
步骤1: 处理 [token1] -> 生成 token2
步骤2: 处理 [token1, token2] -> 生成 token3  (重复计算token1)
步骤3: 处理 [token1, token2, token3] -> 生成 token4  (重复计算token1,2)

使用KV Cache:
步骤1: 处理 [token1] -> 缓存K1,V1 -> 生成 token2
步骤2: 处理 [token2] + 使用缓存的K1,V1 -> 缓存K2,V2 -> 生成 token3
步骤3: 处理 [token3] + 使用缓存的K1,V1,K2,V2 -> 生成 token4
```

**显存占用：**

KV Cache会占用大量显存。对于一个70B参数的模型：
- 每个token的KV Cache约占用 1-2 KB
- 生成2048个token需要 2-4 MB
- 处理100个并发请求可能需要 200-400 MB

因此，高效的KV Cache管理至关重要。

### 4. Prefill 和 Decode

LLM推理包含两个不同的阶段：

**Prefill（预填充）阶段：**
- 处理用户输入的完整prompt
- 一次性计算所有输入token的KV Cache
- 计算密集型：需要处理多个token
- 可以高度并行化

```
输入: "请介绍一下人工智能的发展历史"
Prefill: 一次性处理所有14个token（假设分词后是14个）
输出: 生成第一个回复token
```

**Decode（解码）阶段：**
- 逐个生成输出token
- 每次只处理一个新token
- 访存密集型：需要读取大量KV Cache
- 串行执行，无法并行

```
Decode步骤1: 基于Prefill结果生成第一个token "人工"
Decode步骤2: 基于前面所有token生成第二个token "智能"
Decode步骤3: 生成第三个token "是"
...持续到生成结束
```

**性能特点对比：**

| 特性 | Prefill | Decode |
|------|---------|--------|
| 计算量 | 大 | 小 |
| 显存访问 | 少 | 多 |
| 瓶颈 | 计算带宽 | 内存带宽 |
| 并行度 | 高 | 低 |
| 延迟 | 较高 | 需要低延迟 |

---

## 系统架构

### 整体架构图

```
┌─────────────┐
│   用户请求   │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│   API Server    │  ← FastAPI服务器，提供OpenAI兼容API
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Tokenizer      │  ← 将文本转换为token ID
│  Worker         │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│         Scheduler Worker (Rank 0)           │
│  ┌─────────────────────────────────────┐   │
│  │  Scheduler: 调度和批处理管理         │   │
│  └────────────┬────────────────────────┘   │
│               │                             │
│               ▼                             │
│  ┌─────────────────────────────────────┐   │
│  │  Engine: 模型执行引擎                │   │
│  │  - Model (Llama/Qwen3)              │   │
│  │  - Attention Backend (FA/FI)        │   │
│  │  - KV Cache Manager                 │   │
│  │  - CUDA Graph                       │   │
│  └─────────────────────────────────────┘   │
└────────────┬────────────────────────────────┘
             │
             │ (NCCL/torch.distributed)
             │
    ┌────────┴────────┬─────────────┐
    ▼                 ▼             ▼
┌─────────┐    ┌─────────┐   ┌─────────┐
│Scheduler│    │Scheduler│   │Scheduler│
│ Rank 1  │    │ Rank 2  │   │ Rank 3  │
│ (GPU 1) │    │ (GPU 2) │   │ (GPU 3) │
└─────────┘    └─────────┘   └─────────┘
             │
             ▼
┌─────────────────┐
│  Detokenizer    │  ← 将token ID转换回文本
│  Worker         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   API Server    │  ← 返回结果给用户
│  (返回响应)      │
└─────────────────┘
```

### 进程模型

Mini-SGLang 采用多进程架构，不同组件运行在独立的进程中：

**1. API Server进程**
- 负责：接收HTTP请求，返回HTTP响应
- 通信：通过ZeroMQ与Tokenizer通信

**2. Tokenizer Worker进程**
- 负责：文本↔Token转换
- 通信：通过ZeroMQ与API Server和Scheduler通信

**3. Detokenizer Worker进程**
- 负责：Token↔文本转换（用于输出）
- 通信：通过ZeroMQ与Scheduler和API Server通信

**4. Scheduler Worker进程（每个GPU一个）**
- 负责：请求调度、批处理、模型执行
- 通信：
  - Rank 0通过ZeroMQ与Tokenizer/Detokenizer通信
  - 所有Rank通过NCCL进行GPU间通信

**为什么使用多进程？**
- **隔离性**：不同组件崩溃不会影响其他组件
- **并行性**：Tokenizer和GPU计算可以并行执行
- **灵活性**：可以独立扩展各个组件
- **GPU亲和性**：每个Scheduler进程绑定到特定GPU

### 通信机制

**ZeroMQ (消息队列)**
- 用于：控制消息和小数据传输
- 特点：低延迟、异步、支持多种通信模式
- 使用场景：
  - API Server ↔ Tokenizer
  - Tokenizer ↔ Scheduler (Rank 0)
  - Scheduler (Rank 0) ↔ Detokenizer
  - Detokenizer ↔ API Server

**NCCL (NVIDIA Collective Communications Library)**
- 用于：GPU之间的大规模张量通信
- 特点：高带宽、针对GPU优化
- 使用场景：
  - 张量并行中的All-Reduce操作
  - 多GPU间的模型参数同步
  - KV Cache的gather操作

---

## 关键技术

### 1. 张量并行 (Tensor Parallelism)

**基本原理：**

将模型的每一层按列或行切分到多个GPU上，每个GPU只存储和计算部分参数。

**线性层切分示例：**

```
单GPU模式：
输入X (batch, seq, hidden) @ 权重W (hidden, ffn_dim) = 输出Y (batch, seq, ffn_dim)

张量并行（4个GPU）：
GPU 0: X @ W_0 (hidden, ffn_dim/4) = Y_0 (batch, seq, ffn_dim/4)
GPU 1: X @ W_1 (hidden, ffn_dim/4) = Y_1 (batch, seq, ffn_dim/4)
GPU 2: X @ W_2 (hidden, ffn_dim/4) = Y_2 (batch, seq, ffn_dim/4)
GPU 3: X @ W_3 (hidden, ffn_dim/4) = Y_3 (batch, seq, ffn_dim/4)

最后拼接: Y = [Y_0, Y_1, Y_2, Y_3]
```

**实现细节：**

```python
# 列并行（Column Parallel）：输出维度切分
class ColumnParallelLinear:
    def forward(self, x):
        # 每个GPU计算部分输出
        output_parallel = F.linear(x, self.weight)
        # 不需要通信，每个GPU保留部分结果
        return output_parallel

# 行并行（Row Parallel）：输入维度切分
class RowParallelLinear:
    def forward(self, x):
        # 输入已经是切分的
        output_parallel = F.linear(x, self.weight)
        # 需要All-Reduce聚合结果
        output = all_reduce(output_parallel)
        return output
```

**注意力层的张量并行：**

```
多头注意力 (MHA):
- num_heads = 32
- 使用4个GPU做张量并行

每个GPU负责:
- 8个注意力头 (32 / 4 = 8)
- 每个GPU独立计算自己的注意力头
- 最后通过All-Reduce聚合结果
```

**优缺点：**

✅ 优点：
- 支持超大模型（单GPU放不下）
- 线性扩展显存容量
- 计算并行化

❌ 缺点：
- GPU间通信开销
- 需要高速互连（NVLink/InfiniBand）
- 受限于最慢的GPU

### 2. Radix Cache（基数缓存）

**数据结构：Radix Tree（基数树）**

基数树是一种前缀树，用于高效存储和查找共享前缀。

```
示例对话：
请求1: "你好，请介绍一下北京"
请求2: "你好，请介绍一下上海"
请求3: "你好，请介绍一下广州"

传统缓存：
- 为每个请求独立存储完整的KV Cache
- 共享前缀"你好，请介绍一下"被重复存储3次

Radix Cache：
         root
           |
        "你好，"
           |
      "请介绍一下"
           |
    ┌──────┼──────┐
   "北京"  "上海"  "广州"
   
只存储一次共享前缀，节省显存
```

**工作流程：**

```python
# 1. 新请求到来
tokens = ["你好", "，", "请", "介绍", "一下", "北京"]

# 2. 在Radix Tree中查找最长匹配前缀
matched_prefix = tree.match(tokens)
# 找到: ["你好", "，", "请", "介绍", "一下"]

# 3. 复用匹配部分的KV Cache
cache_handle = tree.get_cache(matched_prefix)

# 4. 只计算新增部分 ["北京"]
compute_new_tokens(["北京"], cache_handle)

# 5. 将新结果插入树中
tree.insert(tokens, cache_handle)
```

**LRU淘汰策略：**

当显存不足时，使用最近最少使用（LRU）策略淘汰旧的缓存：

```python
# 每个节点记录最后访问时间
node.last_access_time = current_time

# 显存不足时，从叶子节点开始淘汰
def evict():
    # 找到最久未访问的叶子节点
    oldest_leaf = find_oldest_leaf()
    # 释放其KV Cache
    release_cache(oldest_leaf)
    # 如果父节点变成叶子，也考虑淘汰
    prune_tree()
```

**性能提升：**

- **多轮对话**：系统提示词只计算一次
- **相似查询**：共享前缀的查询大幅减少计算
- **批处理**：多个请求可以共享prefill阶段

### 3. 分块预填充 (Chunked Prefill)

**问题场景：**

处理长输入时，如果一次性计算所有token的注意力：
- 显存需求 = O(sequence_length²)
- 长文档（如8K tokens）会导致显存溢出
- 阻塞其他请求，降低吞吐量

**解决方案：**

将长输入分成多个chunk，分批处理：

```
完整输入: [4096个tokens]

分块策略 (chunk_size=1024):
Chunk 1: tokens[0:1024]     → 计算KV Cache
Chunk 2: tokens[1024:2048]  → 计算KV Cache（可以与decode交错）
Chunk 3: tokens[2048:3072]  → 计算KV Cache
Chunk 4: tokens[3072:4096]  → 计算KV Cache

每个chunk独立处理，降低峰值显存
```

**与Decode交错执行：**

```
时间线：
T1: [Prefill Chunk 1 for Request A]
T2: [Prefill Chunk 2 for Request A] + [Decode for Request B, C, D]
T3: [Prefill Chunk 3 for Request A] + [Decode for Request B, C, D]
T4: [Prefill Chunk 4 for Request A] + [Decode for Request B, C, D]

优势：
- Prefill不会完全阻塞Decode
- 提高GPU利用率
- 降低平均延迟
```

**实现细节：**

```python
def chunked_prefill(tokens, max_chunk_size):
    num_chunks = (len(tokens) + max_chunk_size - 1) // max_chunk_size
    
    for i in range(num_chunks):
        start = i * max_chunk_size
        end = min((i + 1) * max_chunk_size, len(tokens))
        chunk = tokens[start:end]
        
        # 处理当前chunk
        # 使用之前chunk计算的KV Cache
        output = model.forward(
            chunk,
            use_cache=True,
            past_kv_cache=kv_cache
        )
        
        # 更新KV Cache
        kv_cache.append(output.kv_cache)
```

**参数调优：**

- **chunk_size太小**：GPU利用率低，总体时间增加
- **chunk_size太大**：显存压力大，可能OOM
- **推荐值**：512-2048，根据GPU显存和模型大小调整

### 4. CUDA Graph

**什么是CUDA Graph？**

CUDA Graph是一种优化技术，将一系列CUDA操作录制成图，然后重放（replay），减少CPU启动开销。

**传统方式 vs CUDA Graph：**

```
传统方式（每个iteration）：
1. CPU准备参数
2. CPU发起CUDA kernel
3. CPU等待同步
4. GPU执行kernel
开销：CPU启动kernel需要5-20微秒

CUDA Graph方式：
首次录制：
1. 记录完整的kernel序列
2. 生成可执行图

后续iteration：
1. CPU直接重放图（1-2微秒）
2. GPU执行所有kernel
开销大幅降低！
```

**适用场景：**

✅ 适合CUDA Graph:
- Decode阶段（输入shape固定）
- 批次大小固定的推理
- 重复执行的计算

❌ 不适合CUDA Graph:
- Prefill阶段（输入长度变化）
- 动态batch大小
- 包含CPU-GPU数据传输的操作

**实现示例：**

```python
class CUDAGraphRunner:
    def __init__(self, model, max_batch_size):
        self.graphs = {}
        # 为每个批次大小录制一个graph
        for bs in range(1, max_batch_size + 1):
            self.graphs[bs] = self._capture_graph(model, bs)
    
    def _capture_graph(self, model, batch_size):
        # 预热
        dummy_input = torch.zeros(batch_size, ...)
        for _ in range(3):
            model(dummy_input)
        
        # 开始录制
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = model(dummy_input)
        
        return graph, dummy_input, output
    
    def run(self, batch_size, input_data):
        # 使用对应批次大小的graph
        graph, dummy_input, output = self.graphs[batch_size]
        # 拷贝输入数据
        dummy_input.copy_(input_data)
        # 重放graph
        graph.replay()
        # 返回输出
        return output.clone()
```

**性能提升：**

- **小批次decode**: 2-3x加速
- **大批次decode**: 1.2-1.5x加速（计算本身占比高）

### 5. 重叠调度 (Overlap Scheduling)

**问题分析：**

在传统调度中，CPU调度和GPU计算是串行的：

```
传统调度：
|--CPU调度--|--GPU计算--|--CPU调度--|--GPU计算--|
            ↑                      ↑
         等待GPU               等待GPU
```

CPU调度包括：
- 决定哪些请求进入batch
- 分配KV Cache
- 准备输入tensor
- 调用GPU kernel

这些操作可能耗时5-20毫秒，降低吞吐量。

**解决方案：**

将下一步的CPU调度与当前GPU计算重叠：

```
重叠调度：
Step N:   |--GPU计算N--|
Step N+1:     |--CPU调度N+1--|--GPU计算N+1--|
Step N+2:         |--CPU调度N+2--|--GPU计算N+2--|

GPU利用率提升！
```

**实现机制：**

```python
class OverlapScheduler:
    def __init__(self):
        self.stream_compute = torch.cuda.Stream()  # GPU计算流
        self.stream_schedule = torch.cuda.Stream()  # 调度流
        
    def run(self):
        # 初始化
        batch_n = self.prepare_batch()
        
        while True:
            # 在计算流中执行GPU计算
            with torch.cuda.stream(self.stream_compute):
                output_n = self.engine.forward(batch_n)
            
            # 同时在调度流中准备下一个batch
            with torch.cuda.stream(self.stream_schedule):
                batch_n_plus_1 = self.prepare_next_batch()
            
            # 等待计算完成
            self.stream_compute.synchronize()
            
            # 处理输出
            self.process_output(output_n)
            
            # 下一轮
            batch_n = batch_n_plus_1
```

**关键点：**

1. **双缓冲**：准备两套输入buffer，交替使用
2. **CUDA Stream**：使用不同stream实现并行
3. **内存管理**：避免调度流和计算流的内存冲突

**性能提升：**

- 小batch（1-4）: 30-50%吞吐量提升
- 中等batch（8-16）: 15-25%吞吐量提升  
- 大batch（32+）: 5-10%吞吐量提升

---

## 性能优化

### 注意力计算优化

**Flash Attention原理：**

传统注意力计算：
```python
# 需要存储中间结果，显存占用O(N²)
Q @ K^T → S (N×N矩阵)  # 巨大的中间矩阵！
softmax(S) → P (N×N矩阵)
P @ V → O
```

Flash Attention优化：
```python
# 分块计算，显存占用O(N)
for block_q in split(Q):
    for block_k, block_v in split(K, V):
        # 只计算小块的注意力
        block_o = flash_attention_block(block_q, block_k, block_v)
        # 增量更新输出
        merge(block_o, output)
```

**Flash Infer优化：**

专门针对decode阶段（batch size大，sequence length = 1）：
- 优化GPU内存访问模式
- 减少kernel启动次数
- 针对小batch优化

### 显存优化策略

**1. KV Cache量化**

将FP16的KV Cache量化为INT8：
```python
# 原始: 2 bytes per element
kv_cache_fp16 = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16)

# 量化: 1 byte per element + scale
kv_cache_int8, scale = quantize(kv_cache_fp16)

# 显存节省50%，精度损失<1%
```

**2. PagedAttention**

将KV Cache分成固定大小的pages，按需分配：
```
传统方式：
Request 1: [==========] 预分配max_length的空间，浪费！
Request 2: [====] 实际只用了一半

PagedAttention：
Request 1: [Page1][Page2][Page3]  按需分配
Request 2: [Page1]  只分配需要的

显存利用率提升2-3倍
```

### 批处理优化

**Continuous Batching：**

传统批处理需要等所有请求完成：
```
Batch 1: [Req A]===== (完成)
         [Req B]========== (等待B完成才能开始新batch)
         [Req C]====== (完成)

Batch 2: 等待Batch 1全部完成...
```

Continuous Batching可以动态调整：
```
Step 1: [Req A][Req B][Req C]
Step 2: [Req A][Req B][Req C]
Step 3: [Req A完成，加入Req D][Req B][Req C]
Step 4: [Req D][Req B][Req C]
...

GPU利用率大幅提升！
```

---

## 数据流程

### 完整请求生命周期

让我们跟踪一个实际请求的完整流程：

**输入请求：**
```json
{
  "model": "Qwen/Qwen3-0.6B",
  "messages": [
    {"role": "user", "content": "介绍一下人工智能"}
  ],
  "stream": true,
  "max_tokens": 100
}
```

**流程步骤：**

```
1. API Server接收请求
   ↓
   创建Request对象
   ↓
   通过ZMQ发送到Tokenizer Worker

2. Tokenizer Worker处理
   ↓
   应用chat template:
   "<|im_start|>user\n介绍一下人工智能<|im_end|>\n<|im_start|>assistant\n"
   ↓
   Tokenization:
   [151644, 872, 198, 115124, 105484, 236, 151645, 198, 151644, 78191, 198]
   ↓
   通过ZMQ发送到Scheduler (Rank 0)

3. Scheduler (Rank 0) 调度
   ↓
   在Radix Tree中查找缓存: 找到系统提示词的缓存
   ↓
   创建Req对象:
   - input_ids: [11个tokens]
   - cached_len: 8 (系统提示词已缓存)
   - output_len: 100
   ↓
   广播到所有Scheduler Ranks (通过NCCL)

4. 所有Scheduler Ranks执行
   ↓
   Prefill阶段:
   - 只计算新增的3个tokens
   - 复用cached的8个tokens的KV Cache
   ↓
   Engine.forward():
     ├─ Embedding层
     ├─ Decoder层 0-27:
     │    ├─ Self-Attention (使用FlashAttention)
     │    ├─ MLP
     │    └─ LayerNorm
     └─ LM Head
   ↓
   得到logits: [batch=1, seq=3, vocab=151936]
   ↓
   Sampling: 选择最后一个token的logits进行采样
   ↓
   生成第一个输出token: "人工" (token_id: 77848)

5. Scheduler (Rank 0) 收集结果
   ↓
   通过ZMQ发送token到Detokenizer Worker

6. Detokenizer Worker处理
   ↓
   将token_id转换为文本: "人工"
   ↓
   通过ZMQ发送到API Server

7. API Server返回
   ↓
   SSE格式返回:
   data: {"choices":[{"delta":{"content":"人工"},...}]}
   ↓
   用户收到第一个chunk

8-N. Decode阶段循环
   ↓
   每次生成一个token，重复步骤4-7
   ↓
   直到生成<|im_end|>或达到max_tokens=100

最终输出:
"人工智能（Artificial Intelligence, AI）是计算机科学的一个分支..."
```

### 张量并行的数据流

以4-GPU张量并行为例：

```
输入: X [batch=4, seq=1, hidden=3072]

┌─────────────────────────────────────────────┐
│              ColumnParallelLinear            │
│            (QKV projection)                 │
└─────┬───────┬────────┬────────┬─────────────┘
      │       │        │        │
  GPU 0   GPU 1    GPU 2    GPU 3
  权重:   权重:    权重:    权重:
  W[0:768] W[768:1536] W[1536:2304] W[2304:3072]
      │       │        │        │
      ▼       ▼        ▼        ▼
   Q0,K0,V0  Q1,K1,V1  Q2,K2,V2  Q3,K3,V3
   (8 heads) (8 heads) (8 heads) (8 heads)
      │       │        │        │
      ▼       ▼        ▼        ▼
┌─────────────────────────────────────────────┐
│         Self-Attention (独立计算)            │
│  每个GPU计算自己的8个attention heads         │
└─────┬───────┬────────┬────────┬─────────────┘
      │       │        │        │
      ▼       ▼        ▼        ▼
   Attn_0   Attn_1   Attn_2   Attn_3
      │       │        │        │
      └───────┴────────┴────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│              RowParallelLinear               │
│            (Output projection)               │
└───────┬─────────────────────────────────────┘
        │
        ▼
   All-Reduce (NCCL通信)
        │
        ▼
    最终输出: [batch=4, seq=1, hidden=3072]
```

### 重叠调度的时序图

```
Timeline:
─────────────────────────────────────────────────────►

Iteration 1:
CPU:  [准备Batch 1]────┐
GPU:                    └─[计算Batch 1]────┐
                                            │
Iteration 2:                                │
CPU:              [准备Batch 2]────┐        │
GPU:                                └─[计算Batch 2]────┐
                                                        │
Iteration 3:                                            │
CPU:                      [准备Batch 3]────┐            │
GPU:                                        └─[计算Batch 3]

节省的时间：CPU准备时间与GPU计算重叠！
```

---

## 总结

Mini-SGLang通过以下关键技术实现高性能LLM推理：

1. **张量并行**：支持超大模型，线性扩展GPU资源
2. **Radix Cache**：智能缓存管理，减少重复计算
3. **分块预填充**：降低显存峰值，提高GPU利用率
4. **CUDA Graph**：减少CPU开销，提升decode性能
5. **重叠调度**：隐藏CPU调度延迟，提高吞吐量
6. **优化内核**：FlashAttention/FlashInfer加速注意力计算

这些技术协同工作，使Mini-SGLang在保持代码简洁的同时达到业界领先的性能。

---

## 延伸阅读

- [SGLang论文](https://arxiv.org/abs/2312.07104)
- [FlashAttention论文](https://arxiv.org/abs/2205.14135)
- [NanoFlow论文](https://arxiv.org/abs/2408.12757) (Overlap Scheduling)
- [Sarathi-Serve论文](https://arxiv.org/abs/2403.02310) (Chunked Prefill)

如有问题，欢迎提Issue讨论！🚀

