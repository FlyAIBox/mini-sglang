# Mini-SGLang 系统架构

## 系统架构

Mini-SGLang 被设计为一个分布式系统，用于高效处理大语言模型（LLM）推理。它由多个独立进程协同工作组成。

### 核心组件

- **API Server（API 服务器）**：用户的入口点。它提供与 OpenAI 兼容的 API（例如 `/v1/chat/completions`）来接收提示词并返回生成的文本。
- **Tokenizer Worker（分词器工作进程）**：将输入文本转换为模型可以理解的数字（token）。
- **Detokenizer Worker（去分词器工作进程）**：将模型生成的数字（token）转换回人类可读的文本。
- **Scheduler Worker（调度器工作进程）**：核心工作进程。在多 GPU 设置中，每个 GPU 对应一个 Scheduler Worker（称为 **TP Rank**）。它管理该特定 GPU 的计算和资源分配。

### 数据流

组件之间使用 **ZeroMQ (ZMQ)** 进行控制消息通信，使用 **NCCL**（通过 `torch.distributed`）进行 GPU 之间的重型张量数据交换。

![进程概览图](https://lmsys.org/images/blog/minisgl/design.drawio.png)

**请求生命周期：**

1. **用户（User）** 向 **API Server** 发送请求。
2. **API Server** 将请求转发至 **Tokenizer**。
3. **Tokenizer** 将文本转换为 token 并发送至 **Scheduler (Rank 0)**。
4. **Scheduler (Rank 0)** 将请求广播至所有其他 Scheduler（如果使用多个 GPU）。
5. **所有 Scheduler** 调度该请求并触发其本地 **Engine** 计算下一个 token。
6. **Scheduler (Rank 0)** 收集输出 token 并发送至 **Detokenizer**。
7. **Detokenizer** 将 token 转换为文本并发送回 **API Server**。
8. **API Server** 将结果流式传输回 **用户**。

## 代码组织（`minisgl` 包）

源代码位于 `python/minisgl`。以下是面向开发者的模块详解：

- `minisgl.core`：提供核心数据类 `Req` 和 `Batch`，表示请求的状态；类 `Context` 保存推理上下文的全局状态；类 `SamplingParams` 保存用户提供的采样参数。
- `minisgl.distributed`：提供张量并行中的 all-reduce 和 all-gather 接口，以及数据类 `DistributedInfo`，保存 TP 工作进程的 TP 信息。
- `minisgl.layers`：实现构建 LLM 的基本构建块，支持 TP，包括 linear、layernorm、embedding、RoPE 等。它们共享 `minisgl.layers.base` 中定义的通用基类。
- `minisgl.models`：实现 LLM 模型，包括 Llama 和 Qwen3。还定义了从 Hugging Face 加载权重和分片权重的实用工具。
- `minisgl.attention`：提供注意力后端的接口，并实现 `flashattention` 和 `flashinfer` 后端。它们由 `AttentionLayer` 调用，并使用存储在 `Context` 中的元数据。
- `minisgl.kvcache`：提供 KVCache 池和 KVCache 管理器的接口，并实现 `MHAKVCache`、`NaiveCacheManager` 和 `RadixCacheManager`。
- `minisgl.utils`：提供实用工具集合，包括日志设置和 zmq 包装器。
- `minisgl.engine`：实现 `Engine` 类，它是单个进程上的 TP 工作进程。它管理模型、上下文、KVCache、注意力后端和 CUDA 图重放。
- `minisgl.message`：定义在 api_server、tokenizer、detokenizer 和 scheduler 之间交换的消息（通过 zmq）。所有消息类型支持自动序列化和反序列化。
- `minisgl.scheduler`：实现 `Scheduler` 类，它在每个 TP 工作进程上运行并管理相应的 `Engine`。rank 0 的 scheduler 从 tokenizer 接收消息，与其他 TP 工作进程上的 scheduler 通信，并向 detokenizer 发送消息。
- `minisgl.server`：定义 CLI 参数和 `launch_server`，后者启动 Mini-SGLang 的所有子进程。还在 `minisgl.server.api_server` 中实现了一个 FastAPI 服务器作为前端，提供诸如 `/v1/chat/completions` 等端点。
- `minisgl.tokenizer`：实现 `tokenize_worker` 函数，处理分词和去分词请求。
- `minisgl.llm`：提供 `LLM` 类作为 Python 接口，方便与 Mini-SGLang 系统交互。
- `minisgl.kernel`：实现自定义 CUDA 内核，由 `tvm-ffi` 支持，提供 Python 绑定和 JIT 接口。
- `minisgl.benchmark`：基准测试实用工具。
