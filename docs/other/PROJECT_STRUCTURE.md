# Mini-SGLang 项目结构文档

## 📋 目录结构总览

```
mini-sglang/
├── assets/                    # 资源文件（Logo等）
├── benchmark/                 # 性能基准测试
│   ├── offline/              # 离线推理测试
│   └── online/               # 在线推理测试
├── docs/                     # 英文文档
│   ├── features.md           # 功能特性说明
│   ├── structures.md         # 系统架构说明
│   └── zh/                   # 中文文档
│       ├── README_ZH.md      # 中文安装使用指南
│       ├── concepts.md       # 核心概念详解
│       ├── developer.md      # 开发者指南
│       ├── SUMMARY.md        # 文档总结
│       └── WELCOME.md        # 欢迎页面
├── python/minisgl/           # 核心代码（详见下文）
├── tests/                    # 测试代码
│   ├── core/                 # 核心模块测试
│   ├── kernel/               # 内核测试
│   └── misc/                 # 其他测试
├── pyproject.toml            # Python项目配置文件
├── README.md                 # 项目说明文档（英文）
└── PROJECT_STRUCTURE.md      # 本文件：项目结构说明
```

## 🔧 核心代码结构 (python/minisgl/)

### 1. 核心模块（Core Modules）

#### `core.py` - 核心数据结构
- **功能**：定义整个推理系统的基础数据结构
- **关键类**：
  - `SamplingParams`: 采样参数配置（temperature、top_k、top_p等）
  - `Req`: 单个推理请求的状态管理
  - `Batch`: 批处理请求集合
  - `Context`: 全局推理上下文
- **重要性**：★★★★★（必读，理解系统的基础）

#### `env.py` - 环境变量管理
- **功能**：管理系统环境变量和配置
- **作用**：控制特性开关、调试选项等
- **重要性**：★★★☆☆

### 2. 分布式模块（distributed/）

#### `info.py` - 分布式信息
- **功能**：管理张量并行（TP）的配置信息
- **关键类**：
  - `DistributedInfo`: 存储TP的rank、world_size等信息
- **重要性**：★★★★☆

#### `impl.py` - 分布式通信实现
- **功能**：实现All-Reduce、All-Gather等集合通信操作
- **实现**：
  - `TorchDistributed`: 基于PyTorch的分布式后端（默认）
  - `PyNCCL`: 基于NCCL的高性能后端
  - `DistributedCommunicator`: 统一接口
- **重要性**：★★★★☆

### 3. 调度器模块（scheduler/）

#### `scheduler.py` - 主调度器
- **功能**：管理请求的调度和执行
- **职责**：
  - 接收新请求
  - 决定何时执行prefill和decode
  - 管理请求生命周期
  - 与其他TP rank通信
- **重要性**：★★★★★

#### `prefill.py` - Prefill调度策略
- **功能**：处理新请求的首次forward（计算prompt的KV Cache）
- **优化**：支持Chunked Prefill（分块处理长prompt）
- **重要性**：★★★★☆

#### `decode.py` - Decode调度策略
- **功能**：处理自回归生成（每次生成一个token）
- **优化**：批处理多个请求的decode
- **重要性**：★★★★☆

#### `cache.py` - 缓存调度
- **功能**：管理KV Cache的分配和释放
- **重要性**：★★★★☆

#### `io.py` - 输入输出处理
- **功能**：准备batch的输入数据
- **重要性**：★★★☆☆

#### `table.py` - 请求表管理
- **功能**：维护所有活跃请求的索引表
- **重要性**：★★★☆☆

#### `config.py` - 调度器配置
- **功能**：调度器的配置参数
- **重要性**：★★★☆☆

#### `utils.py` - 调度器工具
- **功能**：调度器的辅助函数
- **重要性**：★★☆☆☆

### 4. 引擎模块（engine/）

#### `engine.py` - 推理引擎
- **功能**：管理模型的forward执行
- **职责**：
  - 模型初始化和加载
  - KV Cache管理
  - 注意力后端管理
  - CUDA Graph优化
- **重要性**：★★★★★

#### `graph.py` - CUDA Graph管理
- **功能**：实现CUDA Graph优化（减少kernel启动开销）
- **重要性**：★★★★☆

#### `sample.py` - 采样策略
- **功能**：从logits中采样下一个token
- **实现**：贪心采样、Top-K、Top-P等
- **重要性**：★★★★☆

#### `config.py` - 引擎配置
- **功能**：引擎的配置参数
- **重要性**：★★★☆☆

### 5. 模型模块（models/）

#### `base.py` - 模型基类
- **功能**：定义模型的通用接口
- **重要性**：★★★★☆

#### `llama.py` - Llama模型实现
- **功能**：实现Llama系列模型（包括Llama-2、Llama-3等）
- **重要性**：★★★★☆

#### `qwen3.py` - Qwen3模型实现
- **功能**：实现Qwen3模型
- **重要性**：★★★★☆

#### `config.py` - 模型配置
- **功能**：从HuggingFace配置加载模型参数
- **重要性**：★★★☆☆

#### `weight.py` - 权重加载
- **功能**：加载和分片模型权重（支持TP）
- **重要性**：★★★★☆

#### `utils.py` - 模型工具
- **功能**：模型相关的辅助函数
- **重要性**：★★★☆☆

### 6. 层模块（layers/）

#### `base.py` - 层基类
- **功能**：定义支持TP的层基类
- **重要性**：★★★★☆

#### `linear.py` - 线性层
- **功能**：实现支持TP的线性层
- **类型**：
  - `ColumnParallelLinear`: 列并行
  - `RowParallelLinear`: 行并行
- **重要性**：★★★★☆

#### `attention.py` - 注意力层
- **功能**：实现多头注意力机制
- **集成**：FlashAttention、FlashInfer
- **重要性**：★★★★★

#### `embedding.py` - 嵌入层
- **功能**：Token到向量的嵌入
- **重要性**：★★★☆☆

#### `norm.py` - 归一化层
- **功能**：RMSNorm等归一化层
- **重要性**：★★★☆☆

#### `rotary.py` - 旋转位置编码
- **功能**：实现RoPE（Rotary Position Embedding）
- **重要性**：★★★★☆

#### `activation.py` - 激活函数
- **功能**：SiLU、GELU等激活函数
- **重要性**：★★★☆☆

### 7. 注意力模块（attention/）

#### `base.py` - 注意力后端基类
- **功能**：定义注意力计算的统一接口
- **重要性**：★★★★☆

#### `fa.py` - FlashAttention后端
- **功能**：集成FlashAttention（适合prefill）
- **重要性**：★★★★☆

#### `fi.py` - FlashInfer后端
- **功能**：集成FlashInfer（针对decode优化）
- **重要性**：★★★★☆

#### `utils.py` - 注意力工具
- **功能**：注意力计算的辅助函数
- **重要性**：★★★☆☆

### 8. KV缓存模块（kvcache/）

#### `base.py` - KV缓存基类
- **功能**：定义KV Cache的接口
- **重要性**：★★★★☆

#### `mha_pool.py` - 多头注意力缓存池
- **功能**：管理KV Cache的内存池
- **重要性**：★★★★☆

#### `naive_manager.py` - 简单缓存管理器
- **功能**：基础的KV Cache管理（无共享）
- **重要性**：★★★☆☆

#### `radix_manager.py` - Radix缓存管理器
- **功能**：使用Radix Tree共享KV Cache（核心优化）
- **重要性**：★★★★★

### 9. 服务器模块（server/）

#### `launch.py` - 启动入口
- **功能**：启动所有子进程（API Server、Tokenizer、Scheduler等）
- **重要性**：★★★★☆

#### `api_server.py` - API服务器
- **功能**：提供OpenAI兼容的HTTP API
- **端点**：`/v1/chat/completions`、`/v1/completions`等
- **重要性**：★★★★☆

#### `args.py` - 命令行参数
- **功能**：定义和解析命令行参数
- **重要性**：★★★☆☆

### 10. Tokenizer模块（tokenizer/）

#### `server.py` - Tokenizer服务
- **功能**：运行tokenizer和detokenizer worker
- **重要性**：★★★☆☆

#### `tokenize.py` - Token化
- **功能**：文本转token
- **重要性**：★★★☆☆

#### `detokenize.py` - 去Token化
- **功能**：token转文本（支持流式输出）
- **重要性**：★★★☆☆

### 11. 消息模块（message/）

#### `frontend.py` - 前端消息
- **功能**：定义API Server <-> Tokenizer的消息格式
- **重要性**：★★★☆☆

#### `backend.py` - 后端消息
- **功能**：定义Tokenizer <-> Scheduler的消息格式
- **重要性**：★★★☆☆

#### `tokenizer.py` - Tokenizer消息
- **功能**：Tokenizer相关的消息定义
- **重要性**：★★☆☆☆

#### `utils.py` - 消息工具
- **功能**：消息序列化/反序列化
- **重要性**：★★☆☆☆

### 12. 内核模块（kernel/）

#### `index.py` - 索引内核
- **功能**：高效的索引操作（CUDA JIT）
- **重要性**：★★★☆☆

#### `store.py` - 存储内核
- **功能**：KV Cache的读写操作（CUDA JIT）
- **重要性**：★★★☆☆

#### `radix.py` - Radix Tree内核
- **功能**：Radix Tree的CUDA实现
- **重要性**：★★★★☆

#### `tensor.py` - 张量操作
- **功能**：自定义张量操作
- **重要性**：★★★☆☆

#### `pynccl.py` - NCCL Python绑定
- **功能**：直接调用NCCL库
- **重要性**：★★★☆☆

#### `utils.py` - 内核工具
- **功能**：内核相关的辅助函数
- **重要性**：★★☆☆☆

#### `csrc/` - C/CUDA源代码
- **功能**：底层CUDA内核实现
- **重要性**：★★★☆☆

### 13. 工具模块（utils/）

#### `logger.py` - 日志管理
- **功能**：统一的日志记录
- **重要性**：★★★☆☆

#### `mp.py` - 多进程工具
- **功能**：进程管理和ZMQ通信封装
- **重要性**：★★★☆☆

#### `hf.py` - HuggingFace工具
- **功能**：从HuggingFace加载模型和配置
- **重要性**：★★★☆☆

#### `torch_utils.py` - PyTorch工具
- **功能**：PyTorch相关的辅助函数
- **重要性**：★★★☆☆

#### `arch.py` - 架构检测
- **功能**：检测GPU架构（用于优化）
- **重要性**：★★☆☆☆

#### `misc.py` - 其他工具
- **功能**：通用工具函数
- **重要性**：★★☆☆☆

#### `registry.py` - 注册表
- **功能**：模型和组件的注册机制
- **重要性**：★★★☆☆

### 14. 其他模块

#### `llm/llm.py` - LLM接口
- **功能**：提供简单的Python API接口
- **用途**：不启动server，直接在Python中使用
- **重要性**：★★★☆☆

#### `shell.py` - 交互式Shell
- **功能**：命令行交互界面
- **重要性**：★★☆☆☆

#### `benchmark/` - 基准测试工具
- **功能**：性能测试和分析
- **重要性**：★★★☆☆

#### `__main__.py` - 主入口
- **功能**：`python -m minisgl`的入口点
- **重要性**：★★★☆☆

## 📊 模块依赖关系

```
                    ┌─────────────┐
                    │ API Server  │
                    └──────┬──────┘
                           │ (ZMQ)
                    ┌──────▼──────┐
                    │  Tokenizer  │
                    └──────┬──────┘
                           │ (ZMQ)
            ┌──────────────┼──────────────┐
            │              │              │
       ┌────▼────┐    ┌───▼─────┐   ┌───▼─────┐
       │Scheduler│    │Scheduler│   │Scheduler│
       │ (Rank 0)│    │ (Rank 1)│   │ (Rank N)│
       └────┬────┘    └────┬────┘   └────┬────┘
            │              │              │
            │        (NCCL/All-Reduce)    │
            │              │              │
       ┌────▼────┐    ┌───▼─────┐   ┌───▼─────┐
       │ Engine  │    │ Engine  │   │ Engine  │
       │ (GPU 0) │    │ (GPU 1) │   │ (GPU N) │
       └─────────┘    └─────────┘   └─────────┘
            │
            │ (ZMQ)
       ┌────▼──────┐
       │Detokenizer│
       └───────────┘
```

## 🎯 数据流程

### 1. 请求处理流程

```
用户请求 
  → API Server (接收HTTP请求)
  → Tokenizer (文本→token)
  → Scheduler Rank 0 (接收新请求)
  → All Schedulers (广播请求，TP通信)
  → Engines (并行计算)
  → Scheduler Rank 0 (收集结果)
  → Detokenizer (token→文本)
  → API Server (返回HTTP响应)
  → 用户
```

### 2. Prefill阶段

```
新请求 
  → Scheduler.add_request()
  → Scheduler.schedule_prefill()
  → Engine.forward(batch, phase="prefill")
  → Model.forward()
    ├→ Attention (计算Q·K^T)
    ├→ 存储KV Cache
    └→ 生成首个token
  → Scheduler.complete_prefill()
```

### 3. Decode阶段

```
待生成的请求
  → Scheduler.schedule_decode()
  → Engine.forward(batch, phase="decode")
  → Model.forward()
    ├→ 读取KV Cache
    ├→ Attention (只计算最后一个token)
    └→ 采样下一个token
  → Scheduler.complete_decode()
  → 检查是否完成（EOS或max_tokens）
    ├→ 完成：释放KV Cache，返回结果
    └→ 未完成：继续decode
```

## 🔍 关键技术实现位置

| 技术 | 实现位置 | 说明 |
|------|---------|------|
| **Tensor Parallelism** | `distributed/impl.py`, `layers/linear.py` | 模型并行，分片到多GPU |
| **Radix Cache** | `kvcache/radix_manager.py`, `kernel/radix.py` | 共享KV Cache前缀 |
| **Chunked Prefill** | `scheduler/prefill.py` | 分块处理长prompt |
| **CUDA Graph** | `engine/graph.py` | 减少kernel启动开销 |
| **Overlap Scheduling** | `scheduler/scheduler.py` | CPU/GPU并行执行 |
| **FlashAttention** | `attention/fa.py` | 高效注意力计算 |
| **FlashInfer** | `attention/fi.py` | Decode优化注意力 |
| **Paged KV Cache** | `kvcache/mha_pool.py` | 分页内存管理 |
| **Continuous Batching** | `scheduler/scheduler.py` | 动态批处理 |

## 📚 学习路径推荐

### 初学者（1-3天）
1. 阅读 `README.md` 和 `docs/zh/README_ZH.md`
2. 运行示例：`python -m minisgl --model "Qwen/Qwen3-0.6B"`
3. 阅读 `docs/zh/concepts.md` 前半部分（基础概念）

### 中级（1-2周）
1. 深入阅读 `python/minisgl/core.py`（核心数据结构）
2. 阅读 `python/minisgl/scheduler/scheduler.py`（调度逻辑）
3. 阅读 `python/minisgl/engine/engine.py`（推理引擎）
4. 运行benchmark，理解性能特征

### 高级（2-4周）
1. 阅读 `python/minisgl/kvcache/radix_manager.py`（Radix Cache）
2. 阅读 `python/minisgl/attention/`（注意力后端）
3. 阅读 `python/minisgl/distributed/`（分布式通信）
4. 阅读 `python/minisgl/kernel/`（CUDA内核）
5. 尝试添加新功能或优化

### 专家（持续）
1. 对比其他框架（vLLM、TensorRT-LLM、SGLang）
2. 阅读相关论文
3. 贡献代码和优化
4. 性能调优和profiling

## 🛠️ 开发建议

### 添加新模型
1. 在 `models/` 创建新文件（参考 `llama.py`）
2. 继承 `BaseModel`
3. 实现 `forward()` 方法
4. 在 `models/__init__.py` 注册模型
5. 在 `utils/registry.py` 添加模型映射

### 修改调度策略
1. 修改 `scheduler/prefill.py` 或 `scheduler/decode.py`
2. 调整 batch 大小、chunked prefill 参数等
3. 重新运行 benchmark 评估性能

### 添加新的采样策略
1. 修改 `engine/sample.py`
2. 在 `SamplingParams` 添加新参数
3. 实现采样逻辑

### 优化性能
1. 使用 profiling 工具（`docs/zh/developer.md`）
2. 检查是否启用了所有优化（CUDA Graph、Radix Cache等）
3. 调整超参数（page_size、max_batch_size等）

## 📝 代码规范

1. **类型注解**：所有函数都有完整的类型注解
2. **文档字符串**：关键类和函数都有详细的docstring
3. **代码风格**：遵循Black和Ruff规范
4. **模块化**：功能清晰分离，高内聚低耦合
5. **测试**：关键功能有对应的测试用例

## 🐛 调试技巧

### 启用调试日志
```bash
export MINISGL_LOG_LEVEL=DEBUG
python -m minisgl --model "Qwen/Qwen3-0.6B"
```

### 禁用优化（便于调试）
```bash
# 禁用CUDA Graph
export MINISGL_DISABLE_CUDA_GRAPH=1

# 禁用Radix Cache
python -m minisgl --model "Qwen/Qwen3-0.6B" --cache naive

# 禁用Overlap Scheduling
export MINISGL_DISABLE_OVERLAP_SCHEDULING=1
```

### 使用Python Debugger
```python
# 在代码中添加断点
import pdb; pdb.set_trace()

# 或使用ipdb（需要安装）
import ipdb; ipdb.set_trace()
```

## 🚀 性能优化检查清单

- [ ] 启用Tensor Parallelism（多GPU）
- [ ] 启用Radix Cache（共享前缀）
- [ ] 启用CUDA Graph（减少启动开销）
- [ ] 启用Overlap Scheduling（CPU/GPU并行）
- [ ] 启用FlashInfer（decode优化）
- [ ] 调整page_size（平衡内存和效率）
- [ ] 调整max_batch_size（提高吞吐量）
- [ ] 启用Chunked Prefill（处理长文本）

## 📞 获取帮助

- **GitHub Issues**: [mini-sglang/issues](https://github.com/sgl-project/mini-sglang/issues)
- **中文文档**: `docs/zh/`目录
- **代码注释**: 关键模块有详细的中文注释
- **SGLang主项目**: [sglang](https://github.com/sgl-project/sglang)

---

**最后更新**: 2026-01-26  
**维护者**: Mini-SGLang团队

