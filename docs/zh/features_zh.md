# Mini-SGLang 功能特性

## 在线服务（Online Serving）

Mini-SGLang 支持在线服务，提供与 OpenAI 兼容的 API 服务器。它提供标准的 `/v1/chat/completions` 端点，允许与现有工具和客户端无缝集成。有关详细的命令行参数和配置选项，请运行 `python -m minisgl --help`。

## 交互式 Shell 模式

为了便于演示和测试，Mini-SGLang 提供了交互式 Shell 模式。在此模式下，用户可以直接输入提示词，LLM 将实时生成响应。Shell 会自动缓存聊天历史以维持上下文。要清除对话历史并开始新会话，请使用 `/reset` 命令。

示例：

```bash
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell
```

## 分布式服务（Distributed Serving）

为了在多个 GPU 上扩展性能，Mini-SGLang 支持张量并行（Tensor Parallelism, TP）。您可以通过 `--tp n` 参数指定 GPU 数量来启用分布式服务，其中 `n` 是并行度。

## 支持的模型

我们的框架目前支持以下稠密模型架构：

- [`Llama-3`](https://huggingface.co/collections/meta-llama/llama-31) 系列
- [`Qwen-3`](https://huggingface.co/collections/Qwen/qwen3) 系列

## Chunked Prefill（分块预填充）

Chunked Prefill 是由 [Sarathi-Serve](https://arxiv.org/abs/2403.02310) 引入的一种技术，默认启用。该功能在预填充阶段将长提示词分割为更小的块，显著降低峰值内存使用，防止长上下文服务中出现内存不足（OOM）错误。可以使用 `--max-prefill-length n` 配置块大小。注意，不建议将 `n` 设置为非常小的值（例如 128），因为这可能会显著降低性能。

## 注意力后端（Attention Backends）

Mini-SGLang 集成了高性能注意力内核，包括 [`FlashAttention`](https://github.com/Dao-AILab/flash-attention) 和 [`FlashInfer`](https://github.com/flashinfer-ai/flashinfer)。它支持在预填充和解码阶段使用不同的后端以最大化效率。例如，在 NVIDIA Hopper GPU 上，默认情况下预填充使用 `FlashAttention3`，解码使用 `FlashInfer`。

您可以使用 `--attn` 参数指定后端。如果提供两个值（例如 `--attn fa,fi`），第一个值指定预填充后端，第二个值指定解码后端。

## CUDA Graph

为了最小化解码期间的 CPU 启动开销，Mini-SGLang 支持捕获和重放 CUDA 图。该功能默认启用。可以使用 `--cuda-graph-max-bs n` 设置 CUDA 图捕获的最大批次大小。将 `n` 设置为 `0` 可禁用此功能。

## Radix Cache（基数缓存）

采用来自 [SGLang](https://github.com/sgl-project/sglang.git) 的原始设计，Mini-SGLang 实现了 Radix Cache 来管理键值（KV）缓存。这允许在请求之间复用共享前缀的 KV 缓存，减少冗余计算。该功能默认启用，但可以使用 `--cache naive` 切换到简单的缓存管理策略。

![radix](https://lmsys.org/images/blog/sglang/radix_attn.jpg)
*Radix Attention 示意图，来自 [LMSYS Blog](https://lmsys.org/blog/2024-01-17-sglang/)。*

## Overlap Scheduling（重叠调度）

为了进一步降低 CPU 开销，Mini-SGLang 采用了重叠调度技术，这是 [NanoFlow](https://arxiv.org/abs/2408.12757) 中提出的一种方法。该方法将 CPU 调度开销与 GPU 计算重叠，提高整体系统吞吐量。

![overlap](https://lmsys.org/images/blog/sglang_v0_4/scheduler.jpg)
*重叠调度示意图，来自 [LMSYS Blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)。*
