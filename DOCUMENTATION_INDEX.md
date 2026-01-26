# 📚 Mini-SGLang 文档索引

> 快速找到你需要的文档 | 更新时间: 2026-01-26

---

## 🚀 快速开始

| 文档 | 描述 | 适合人群 |
|------|------|---------|
| [README.md](./README.md) | 项目主页,快速开始指南 | 所有用户 ⭐ |
| [INSTALL_GUIDE_ZH.md](./INSTALL_GUIDE_ZH.md) | 详细的中文安装指南 | 中文用户 |

---

## 📦 依赖管理文档

| 文档 | 描述 | 使用场景 |
|------|------|---------|
| [DEPENDENCIES_SUMMARY.md](./DEPENDENCIES_SUMMARY.md) | 依赖快速参考 | 快速查看版本和安装命令 ⭐ |
| [DEPENDENCIES.md](./DEPENDENCIES.md) | 完整依赖文档 | 深入了解所有依赖 |
| [DEPENDENCIES_VERSION_TABLE.md](./DEPENDENCIES_VERSION_TABLE.md) | 依赖版本对照表 | 检查版本兼容性 |
| [requirements.txt](./requirements.txt) | 核心依赖列表 | pip 安装 |
| [requirements-dev.txt](./requirements-dev.txt) | 开发依赖列表 | 开发环境安装 |
| [pyproject.toml](./pyproject.toml) | 项目配置文件 | 项目构建和配置 |

---

## 🏗️ 项目结构文档

| 文档 | 描述 | 适合人群 |
|------|------|---------|
| [PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md) | 项目结构说明 | 开发者 |
| [docs/structures.md](./docs/structures.md) | 系统架构设计 | 架构师、高级开发者 |
| [docs/features.md](./docs/features.md) | 功能特性详解 | 所有开发者 |

---

## 🇨🇳 中文文档

| 文档 | 描述 | 内容 |
|------|------|------|
| [docs/zh/README_ZH.md](./docs/zh/README_ZH.md) | 中文主文档 | 安装、配置、使用 |
| [docs/zh/concepts.md](./docs/zh/concepts.md) | 核心概念详解 | 技术原理、优化策略 |
| [docs/zh/developer.md](./docs/zh/developer.md) | 开发者指南 | 代码结构、调试技巧 |
| [docs/zh/SUMMARY.md](./docs/zh/SUMMARY.md) | 中文文档总结 | 文档导航 |
| [INSTALL_GUIDE_ZH.md](./INSTALL_GUIDE_ZH.md) | 中文安装指南 | 详细安装步骤 |

---

## 📝 工作总结文档

| 文档 | 描述 | 内容 |
|------|------|------|
| [依赖整理总结.md](./依赖整理总结.md) | 依赖整理工作总结 | 2026-01-26 完成的依赖整理工作 |
| [工作完成总结.md](./工作完成总结.md) | 项目工作总结 | 项目开发历史记录 |

---

## 🛠️ 工具和脚本

| 文件 | 描述 | 使用方法 |
|------|------|---------|
| [scripts/check_dependencies.py](./scripts/check_dependencies.py) | 依赖检查脚本 | `python scripts/check_dependencies.py` |
| [.python-version](./.python-version) | Python 版本指定 | 配合 pyenv 使用 |

---

## 📖 按使用场景查找文档

### 🆕 我是新用户,想快速开始

1. 阅读 [README.md](./README.md) - 了解项目概况
2. 查看 [DEPENDENCIES_SUMMARY.md](./DEPENDENCIES_SUMMARY.md) - 了解依赖要求
3. 按照 [README.md](./README.md) 的 Quick Start 部分安装
4. 运行 `python scripts/check_dependencies.py` 验证安装

**中文用户**:
1. 阅读 [docs/zh/README_ZH.md](./docs/zh/README_ZH.md)
2. 查看 [INSTALL_GUIDE_ZH.md](./INSTALL_GUIDE_ZH.md)

### 🔧 我遇到了安装问题

1. 查看 [DEPENDENCIES.md](./DEPENDENCIES.md) 的故障排除部分
2. 运行 `python scripts/check_dependencies.py` 检查环境
3. 查看 [DEPENDENCIES_VERSION_TABLE.md](./DEPENDENCIES_VERSION_TABLE.md) 确认版本兼容性
4. 参考 [INSTALL_GUIDE_ZH.md](./INSTALL_GUIDE_ZH.md) 的常见问题部分

### 💻 我是开发者,想了解代码结构

1. 阅读 [PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md) - 代码组织
2. 查看 [docs/structures.md](./docs/structures.md) - 系统架构
3. 阅读 [docs/zh/developer.md](./docs/zh/developer.md) - 开发指南
4. 查看 [docs/features.md](./docs/features.md) - 功能实现

### 📚 我想深入了解技术原理

1. 阅读 [docs/zh/concepts.md](./docs/zh/concepts.md) - 核心概念
2. 查看 [docs/structures.md](./docs/structures.md) - 架构设计
3. 阅读 [docs/features.md](./docs/features.md) - 功能特性

### 🔄 我想升级或维护依赖

1. 查看 [DEPENDENCIES_VERSION_TABLE.md](./DEPENDENCIES_VERSION_TABLE.md) - 版本对照
2. 阅读 [DEPENDENCIES.md](./DEPENDENCIES.md) 的维护说明部分
3. 参考 [依赖整理总结.md](./依赖整理总结.md) - 了解依赖管理策略

### 🚀 我想部署到生产环境

1. 阅读 [README.md](./README.md) 的 Online Serving 部分
2. 查看 [DEPENDENCIES_SUMMARY.md](./DEPENDENCIES_SUMMARY.md) - 确认依赖
3. 运行 `python scripts/check_dependencies.py` 验证环境
4. 参考 [docs/features.md](./docs/features.md) 了解配置选项

---

## 📊 文档统计

### 文档类型分布

- **安装指南**: 2 个 (README.md, INSTALL_GUIDE_ZH.md)
- **依赖文档**: 5 个 (DEPENDENCIES*.md, requirements*.txt)
- **技术文档**: 3 个 (structures.md, features.md, concepts.md)
- **开发文档**: 2 个 (PROJECT_STRUCTURE.md, developer.md)
- **工作总结**: 2 个 (依赖整理总结.md, 工作完成总结.md)
- **配置文件**: 2 个 (pyproject.toml, .python-version)
- **工具脚本**: 1 个 (check_dependencies.py)

### 语言分布

- **英文文档**: 9 个
- **中文文档**: 8 个
- **双语文档**: README.md (主要英文,包含中文链接)

---

## 🔗 外部资源

### 官方资源
- [SGLang 项目](https://github.com/sgl-project/sglang) - 原始项目
- [Mini-SGLang GitHub](https://github.com/sgl-project/mini-sglang) - 本项目仓库

### 依赖项目
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/) - 模型库
- [FastAPI](https://fastapi.tiangolo.com/) - Web 框架
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) - 注意力优化

### 学习资源
- [CUDA 编程指南](https://docs.nvidia.com/cuda/) - NVIDIA 官方文档
- [Tensor Parallelism](https://arxiv.org/abs/1909.08053) - 相关论文

---

## 🆘 获取帮助

### 问题排查顺序

1. **搜索文档**: 使用 Ctrl+F 在相关文档中搜索关键词
2. **运行检查脚本**: `python scripts/check_dependencies.py`
3. **查看 Issues**: 访问 GitHub Issues 查看类似问题
4. **提交 Issue**: 如果问题未解决,提交新的 Issue

### 常见问题文档位置

| 问题类型 | 查看文档 |
|---------|---------|
| 安装失败 | [DEPENDENCIES.md](./DEPENDENCIES.md) → 常见问题 |
| CUDA 错误 | [INSTALL_GUIDE_ZH.md](./INSTALL_GUIDE_ZH.md) → 故障排除 |
| 版本冲突 | [DEPENDENCIES_VERSION_TABLE.md](./DEPENDENCIES_VERSION_TABLE.md) |
| 使用问题 | [README.md](./README.md) → Quick Start |
| 性能问题 | [docs/features.md](./docs/features.md) |

---

## 📅 文档更新日志

### 2026-01-26
- ✅ 创建依赖管理文档系统
- ✅ 添加 DEPENDENCIES.md (完整文档)
- ✅ 添加 DEPENDENCIES_SUMMARY.md (快速参考)
- ✅ 添加 DEPENDENCIES_VERSION_TABLE.md (版本对照)
- ✅ 创建 requirements.txt 和 requirements-dev.txt
- ✅ 添加依赖检查脚本
- ✅ 创建本文档索引

---

## 💡 文档使用建议

### 对于新用户
- 从 README.md 开始
- 逐步深入到具体文档
- 遇到问题查看对应的故障排除部分

### 对于开发者
- 先了解项目结构 (PROJECT_STRUCTURE.md)
- 深入学习架构设计 (docs/structures.md)
- 参考开发指南 (docs/zh/developer.md)

### 对于维护者
- 定期检查依赖更新 (DEPENDENCIES_VERSION_TABLE.md)
- 更新文档时同步更新本索引
- 保持文档的准确性和时效性

---

**提示**: 使用 Ctrl+F 或 Cmd+F 在本页面快速搜索你需要的文档!

**文档完整性**: ✅ 所有文档已创建并链接正确

