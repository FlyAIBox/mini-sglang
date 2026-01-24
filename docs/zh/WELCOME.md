# 欢迎来到 Mini-SGLang！

一个**轻量级且高性能**的大语言模型推理框架，专为学习和研究设计。

---

## 🎯 快速导航

### 🚀 新手入门

如果你是第一次使用Mini-SGLang，请按以下顺序开始：

1. **[安装和快速开始](./docs/zh/README_ZH.md)** ⭐ 从这里开始！
   - 系统要求和环境准备
   - 详细的安装步骤（Linux / WSL2）
   - 第一个推理示例
   - 常见问题解决

2. **[核心概念详解](./docs/zh/concepts.md)** 📚 深入理解
   - 大模型推理基础（Token、KV Cache、Prefill/Decode）
   - 系统架构设计
   - 5大关键技术详解（张量并行、Radix Cache、分块预填充等）
   - 完整的数据流程演示

3. **[开发者指南](./docs/zh/developer.md)** 🛠️ 动手实践
   - 代码结构说明
   - 添加新模型教程
   - 性能调优技巧
   - 调试方法和工具

### 📖 学习路径

```
第一天：安装和基本使用
├── 阅读安装指南
├── 成功运行第一个示例
└── 了解基本命令行参数

第2-3天：理解核心概念
├── 学习Token和KV Cache
├── 理解Prefill和Decode
└── 了解批处理的重要性

第4-7天：深入技术细节
├── 张量并行原理
├── Radix Cache工作机制
├── CUDA Graph优化
└── 重叠调度策略

第2周：代码实践
├── 阅读核心代码注释
├── 运行性能测试
├── 尝试修改参数
└── 添加自定义功能

持续学习：
├── 阅读相关论文
├── 对比其他框架
└── 参与开源贡献
```

## 📂 文档结构

```
docs/zh/
├── README_ZH.md      # 安装和使用指南（推荐首先阅读）
├── concepts.md       # 核心概念和技术原理
├── developer.md      # 开发者指南和调试技巧
└── SUMMARY.md        # 文档总结和学习建议

python/minisgl/       # 核心代码（含详细中文注释）
├── core.py          # ✅ 核心数据结构（Req, Batch, Context）
├── distributed/     # ✅ 分布式通信（张量并行）
├── scheduler/       # ✅ 调度器（已添加关键注释）
├── attention/       # 注意力机制
├── kvcache/         # KV缓存管理
├── engine/          # 推理引擎
├── models/          # 模型实现
└── ...
```

## ✨ 特色功能

- **📖 全中文文档**：详细的安装、使用和开发指南
- **💡 丰富的代码注释**：核心模块配有详尽的中文注释
- **🎓 教学友好**：从基础概念讲起，适合入门者
- **⚡ 高性能**：业界领先的吞吐量和延迟
- **🔧 易于修改**：简洁的代码结构，便于学习和扩展

## 🎮 快速体验

### 1. 安装（5分钟）

```bash
# 克隆仓库
git clone https://github.com/sgl-project/mini-sglang.git
cd mini-sglang

# 创建虚拟环境
uv venv --python=3.12
source .venv/bin/activate

# 安装依赖
uv pip install -e .
```

### 2. 运行第一个示例（1分钟）

```bash
# 启动服务器（使用小模型）
python -m minisgl --model "Qwen/Qwen3-0.6B"
```

### 3. 发送请求

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[
        {"role": "user", "content": "介绍一下人工智能"}
    ],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## 🤔 常见问题

### Q: 我是完全的新手，应该从哪里开始？

**A:** 按顺序阅读以下文档：
1. [安装指南](./docs/zh/README_ZH.md) - 确保能成功运行
2. [核心概念 - 基础部分](./docs/zh/concepts.md#基础概念) - 理解基本原理
3. [核心概念 - 系统架构](./docs/zh/concepts.md#系统架构) - 了解整体设计

### Q: 我想理解代码，应该怎么阅读？

**A:** 推荐顺序：
1. `python/minisgl/core.py` - 理解核心数据结构
2. `python/minisgl/distributed/` - 理解分布式通信
3. [开发者指南](./docs/zh/developer.md) - 查看代码结构说明
4. 根据兴趣阅读其他模块

### Q: 显存不足怎么办？

**A:** 尝试以下方法：
```bash
# 1. 使用更小的模型
python -m minisgl --model "Qwen/Qwen3-0.6B"

# 2. 减小batch size
python -m minisgl --model "..." --max-batch-size 8

# 3. 减小prefill chunk size
python -m minisgl --model "..." --max-prefill-length 512

# 4. 使用多GPU（张量并行）
python -m minisgl --model "..." --tp 2
```

详见：[故障排除](./docs/zh/README_ZH.md#-故障排除)

### Q: 如何调优性能？

**A:** 参考以下文档：
- [性能调优建议](./docs/zh/README_ZH.md#性能调优建议)
- [开发者指南 - 性能调优](./docs/zh/developer.md#性能调优)
- [核心概念 - 性能优化](./docs/zh/concepts.md#性能优化)

## 📊 性能表现

Mini-SGLang在保持代码简洁（~5000行）的同时，达到了业界领先的性能：

| 场景 | 硬件 | 吞吐量 | 对比 |
|------|------|--------|------|
| 离线推理 | 1x H200 | ~45K tokens/s | 接近SGLang |
| 在线推理 | 4x H200 | 高并发低延迟 | 与SGLang相当 |

详见：[性能基准测试](./docs/zh/README_ZH.md#-性能基准测试)

## 🤝 参与贡献

我们欢迎各种形式的贡献：

- 📝 改进文档和注释
- 🐛 报告或修复bug
- ✨ 添加新功能
- 🎓 分享学习心得
- 💬 参与讨论和答疑

请查看：[开发者指南 - 贡献指南](./docs/zh/developer.md#贡献指南)

## 📚 延伸学习

### 核心论文

- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [NanoFlow: Towards Optimal Large Language Model Serving Throughput](https://arxiv.org/abs/2408.12757)

### 相关项目

- [SGLang](https://github.com/sgl-project/sglang) - 完整版实现
- [vLLM](https://github.com/vllm-project/vllm) - PagedAttention
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) - 高性能注意力kernel

### 社区资源

- [LMSYS Blog](https://lmsys.org/blog/) - SGLang团队技术博客
- [GitHub Issues](https://github.com/sgl-project/mini-sglang/issues) - 问题讨论

## 🌟 致谢

感谢以下项目和团队：

- SGLang团队提供的优秀设计和实现
- FlashAttention和FlashInfer提供的高性能kernel
- PyTorch和HuggingFace提供的基础设施
- 所有贡献者和用户的支持

---

**开始你的大模型推理学习之旅吧！** 🚀

有任何问题欢迎提Issue或查看文档！

