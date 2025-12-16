# LLM Tools Workbench

一站式 LLM 可视化工具集，面向 LLM 工程师的底层机制探索与调试平台。

包含 **TokenLab** (分词实验室)、**EmbeddingLab** (向量分析工作台)、**GenerationLab** (生成机制探索)、**InterpretabilityLab** (可解释性分析)、**DataLab** (数据工程实验室) 和 **ModelLab** (模型工具箱)。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行 Gradio 应用
python app_gradio.py
```

访问 `http://localhost:7860` 开始使用。

---

## 项目结构

```
├── app_gradio.py              # Gradio 应用入口
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
├── generation_lab/            # GenerationLab 模块
│   ├── generation_utils.py    # 生成核心工具
│   ├── logits_inspector.py    # Logits 可视化
│   ├── beam_visualizer.py     # Beam Search 可视化
│   └── kv_cache_sim.py        # KV Cache 模拟器
├── interpretability_lab/      # InterpretabilityLab 模块
│   ├── interpretability_utils.py # 可解释性工具
│   ├── attention_map.py       # 注意力图可视化
│   ├── rope_explorer.py       # RoPE 探索器
│   └── ffn_activation.py      # FFN 激活分析
├── data_lab/                  # DataLab 模块
│   ├── data_utils.py          # 数据处理工具
│   ├── hf_dataset_viewer.py   # HF Dataset 浏览器
│   ├── cleaner_playground.py  # 数据清洗工具
│   └── instruct_formatter.py  # SFT 格式转换器
├── model_lab/                 # ModelLab 模块
│   ├── model_utils.py         # 模型工具函数
│   ├── memory_estimator.py    # 显存估算
│   ├── peft_calculator.py     # PEFT 参数计算器
│   └── config_diff.py         # Config 差异对比
└── requirements.txt           # 依赖清单
```

---

## TokenLab - 分词实验室

一站式 LLM 分词器可视化、调试与效率分析工作台。

| 模块 | 功能 |
|------|------|
| **分词编码** | 交互式编解码、彩虹分词、压缩率统计、Byte Fallback 分析 |
| **模型对比** | 多模型分词效果对比、效率指标可视化 |
| **Chat Template** | 对话模版渲染、特殊 Token 高亮 |

**支持的模型**: OpenAI (GPT-2/3.5)、Meta (Llama-2/3/4)、Alibaba (Qwen)、DeepSeek、Google (Gemma)、MiniMax、Moonshot

---

## EmbeddingLab - 向量分析工作台

可视化的向量分析工作台，解构大语言模型的"潜空间"（Latent Space）。

**设计理念**: Visible, Interactable, Explainable (可见、可交互、可解释)

| 模块 | 功能 |
|------|------|
| **向量运算** | Word2Vec 类比推理、向量计算器、Bias 分析 |
| **模型对比** | TF-IDF/BM25 vs Dense Embedding 对比，揭示稀疏/稠密表示差异 |
| **向量可视化** | 3D 空间漫游、PCA/t-SNE/UMAP 降维 |
| **语义相似度** | Token 相似度热力图、各向异性分析 |

---

## GenerationLab - 生成机制探索

深入理解 LLM 文本生成的底层机制。

| 模块 | 功能 |
|------|------|
| **Logits Inspector** | Next Token 概率分布、Temperature 实验、Top-K/Top-P 截断可视化 |
| **Beam Search** | 搜索树可视化、剪枝过程展示、候选序列追踪 |
| **KV Cache 模拟器** | 显存计算、Prefill/Decode 增长模拟、PagedAttention 分配可视化 |

---

## InterpretabilityLab - 可解释性分析

窥探 Transformer 内部工作机制。

| 模块 | 功能 |
|------|------|
| **Attention 可视化** | 注意力热力图、Token 分析、模式识别（对角线/首Token/局部/全局） |
| **RoPE 探索** | 旋转动画、多频率分解、相对位置衰减特性、不同 Base 对比 |
| **FFN 激活** | 激活函数曲线、SwiGLU 门控可视化、架构对比 |

---

## DataLab - 数据工程实验室

SFT 数据处理与质量分析。

| 模块 | 功能 |
|------|------|
| **Dataset Viewer** | HuggingFace 数据集流式加载、字段分析、长度分布、质量检查 |
| **数据清洗** | 规则清洗（HTML/URL/邮箱）、Unicode 规范化、PPL 过滤 |
| **格式转换** | Alpaca/ShareGPT/ChatML/Llama-2 格式互转 |

---

## ModelLab - 模型工具箱

模型部署与微调的实用工具集。

| 模块 | 功能 |
|------|------|
| **显存估算** | 多精度显存计算、训练各阶段显存分布 |
| **PEFT 计算器** | LoRA 参数量计算、可训练比例、显存估算 |
| **Config 对比** | 双模型配置差异、GQA/RoPE 分析、参数量估算 |

---

## 技术栈

| 分类 | 依赖 | 用途 |
|------|------|------|
| **核心** | `gradio` | Web 框架 |
| | `transformers` | Tokenizer & Model 加载 |
| | `plotly` | 交互式图表 |
| **Embedding** | `sentence-transformers` | Dense Embedding |
| | `gensim` | Word2Vec/GloVe |
| | `scikit-learn` | PCA, t-SNE |
| | `umap-learn` | UMAP 降维 |
| **Model** | `accelerate` | 显存估算 |
| | `huggingface_hub` | 模型配置获取 |

---

## 后续规划

### 短期目标 (v1.1)

1. **TokenLab 增强**
   - 添加 Tiktoken (OpenAI) 对比
   - 支持自定义词表上传分析
   - Token 频率统计与 Zipf 定律可视化

2. **GenerationLab 扩展**
   - Speculative Decoding 模拟器
   - 多种采样策略对比 (Greedy/Beam/Nucleus/Contrastive)
   - 生成质量指标 (Repetition/Diversity)

3. **InterpretabilityLab 深化**
   - Probing 分析（探测特定层对任务的贡献）
   - 激活 Patch 实验（因果干预）
   - Logit Lens 可视化

### 中期目标 (v1.2)

4. **推理优化实验室 (InferenceLab)**
   - 量化效果对比 (GPTQ/AWQ/GGUF)
   - Flash Attention 性能分析
   - Continuous Batching 模拟
   - Speculative Decoding 性能测试

5. **训练分析工作台 (TrainingLab)**
   - Loss 曲线分析与异常检测
   - 梯度分布可视化
   - Learning Rate Schedule 可视化
   - 混合精度训练显存估算

6. **RAG 调试器 (RAGLab)**
   - 检索召回率分析
   - Chunk 策略对比
   - Reranker 效果可视化
   - Embedding 相似度调试

### 长期目标 (v2.0)

7. **多模态扩展**
   - Vision Transformer 注意力可视化
   - 图文相似度分析
   - Tokenizer 对比（文本 vs 视觉）

8. **Agent 调试工具**
   - Tool Calling 流程可视化
   - ReAct 思维链追踪
   - Memory 管理分析

9. **评测中心 (EvalLab)**
   - 常用 Benchmark 一键测评
   - LLM-as-Judge 对比
   - 人类偏好收集界面

10. **部署助手**
    - vLLM/TGI 配置生成器
    - 多 GPU 分布策略推荐
    - 容器化部署模板

---

## 设计理念

1. **底层优先**: 不只是黑盒调用，而是深入机制的可视化
2. **交互探索**: 支持参数调整，观察变化，建立直觉
3. **工程实用**: 解决 LLM 工程师日常遇到的实际问题
4. **美观专业**: 清晰的数据可视化，专业的 UI 设计

---

## 贡献指南

欢迎提交 Issue 和 Pull Request！

- 新功能请先开 Issue 讨论
- 代码风格保持一致，禁止 emoji
- 保持模块独立性，便于维护

---

## License

MIT License
