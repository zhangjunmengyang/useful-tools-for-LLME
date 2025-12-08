# LLM Tools Workbench

一站式 LLM 可视化工具集，包含 **TokenLab** (分词实验室)、**EmbeddingLab** (向量分析工作台) 和 **ModelLab** (模型工具箱)。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行应用
streamlit run app.py
```

---

## 项目结构

```
├── app.py                     # 应用入口 & 导航控制
├── shared/                    # 全局共享资源
│   └── styles.py              # 全局样式系统
├── token_lab/                 # TokenLab 模块
│   ├── tokenizer_utils.py     # Tokenizer 核心工具
│   ├── playground.py          # 分词编码页面
│   ├── arena.py               # 模型对比页面
│   └── chat_builder.py        # Chat Template 页面
├── embedding_lab/             # EmbeddingLab 模块
│   ├── embedding_utils.py     # Embedding 核心工具
│   ├── vector_arithmetic.py   # 向量运算页面
│   ├── model_comparison.py    # 模型对比页面
│   ├── vector_visualization.py# 向量可视化页面
│   └── semantic_similarity.py # 语义相似度页面
├── model_lab/                 # ModelLab 模块
│   ├── model_utils.py         # 模型工具函数
│   └── memory_estimator.py    # 显存估算页面
├── doc/                       # 项目文档
│   ├── architecture.md        # 架构设计文档
│   └── design.md              # UI 设计规范
└── requirements.txt           # 依赖清单
```

---

## 🔤 TokenLab - 分词实验室

一站式 LLM 分词器可视化、调试与效率分析工作台。

### 功能模块

| 模块 | 文件 | 功能 |
|------|------|------|
| **分词编码** | `playground.py` | 交互式编解码、彩虹分词、压缩率统计、Byte Fallback 分析 |
| **模型对比** | `arena.py` | 多模型分词效果对比、效率指标可视化 |
| **Chat Template** | `chat_builder.py` | 对话模版渲染、特殊 Token 高亮 |

### 支持的模型厂商

- OpenAI (GPT-2, GPT-3.5)
- Meta (Llama-2, Llama-3, Llama-4)
- Alibaba (Qwen2.5, Qwen3)
- DeepSeek (V3, V3.2, R1)
- Google (Gemma)
- MiniMax (M1, M2)
- Moonshot (Kimi)

---

## 🧬 EmbeddingLab - 向量分析工作台

可视化的向量分析工作台，解构大语言模型的"潜空间"（Latent Space）。

**设计理念**：Visible, Interactable, Explainable (可见、可交互、可解释)

### 功能模块

| 模块 | 文件 | 功能 |
|------|------|------|
| **向量运算** | `vector_arithmetic.py` | Word2Vec 类比推理、向量计算器、Bias 分析 |
| **模型对比** | `model_comparison.py` | TF-IDF/BM25 vs Dense Embedding 对比 |
| **向量可视化** | `vector_visualization.py` | 3D 空间漫游、PCA/t-SNE/UMAP 降维 |
| **语义相似度** | `semantic_similarity.py` | Token 相似度热力图、各向异性分析 |

### 支持的 Embedding 模型

- **Dense**: MiniLM (多语言), BGE-Small-ZH (中文)
- **Sparse**: TF-IDF, BM25

---

## 🔧 ModelLab - 模型工具箱

模型相关的实用工具集，帮助开发者更好地了解和使用 LLM。

### 功能模块

| 模块 | 文件 | 功能 |
|------|------|------|
| **显存估算** | `memory_estimator.py` | 估算模型推理/训练所需显存，支持多精度对比 |

### 显存估算功能

- 支持 HuggingFace Hub 上的 `transformers` 和 `timm` 模型
- 计算不同精度 (float32/float16/int8/int4) 的显存需求
- 显示推理最小显存 (最大层大小)
- 显示 Adam 训练峰值显存 (约 4x 模型大小)
- 详细展示训练各阶段的显存分布

---

## 技术栈

### 核心依赖

| 依赖 | 用途 |
|------|------|
| `streamlit` | Web 框架 |
| `transformers` | Tokenizer 加载 |
| `plotly` | 交互式图表 |

### EmbeddingLab 依赖

| 依赖 | 用途 |
|------|------|
| `sentence-transformers` | Dense Embedding |
| `gensim` | Word2Vec/GloVe |
| `scikit-learn` | PCA, t-SNE |
| `umap-learn` | UMAP 降维 |

### ModelLab 依赖

| 依赖 | 用途 |
|------|------|
| `accelerate` | 显存估算计算 |
| `huggingface_hub` | 模型信息获取 |

---

## 文档

- **[架构设计](doc/architecture.md)** - 模块结构、API 设计、开发规范
- **[UI 设计规范](doc/design.md)** - 配色、字号、组件样式