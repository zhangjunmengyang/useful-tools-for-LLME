# LLM Tools Workbench - 架构设计文档

> 本文档定义了项目的架构设计、模块接口、开发规范，供后续开发参考。

---

## 1. 项目架构

### 1.1 目录结构

```
├── app.py                     # 应用入口 (导航控制、页面路由)
├── shared/                    # 全局共享资源
│   ├── __init__.py
│   └── styles.py              # 全局 CSS 样式、颜色常量
├── token_lab/                 # TokenLab 模块 (分词相关)
│   ├── __init__.py
│   ├── tokenizer_utils.py     # 核心工具函数
│   ├── playground.py          # 页面: 分词编码
│   ├── arena.py               # 页面: 模型对比
│   └── chat_builder.py        # 页面: Chat Template
├── embedding_lab/             # EmbeddingLab 模块 (向量相关)
│   ├── __init__.py
│   ├── embedding_utils.py     # 核心工具函数
│   ├── vector_arithmetic.py   # 页面: 向量运算
│   ├── model_comparison.py    # 页面: 模型对比
│   ├── vector_visualization.py# 页面: 向量可视化
│   └── semantic_similarity.py # 页面: 语义相似度
├── generation_lab/            # GenerationLab 模块 (推理解码)
│   ├── __init__.py
│   ├── generation_utils.py    # 核心工具函数
│   ├── logits_inspector.py    # 页面: Logits 显微镜
│   ├── beam_visualizer.py     # 页面: Beam Search 可视化
│   └── kv_cache_sim.py        # 页面: KV Cache 模拟器
├── interpretability_lab/      # InterpretabilityLab 模块 (可解释性)
│   ├── __init__.py
│   ├── interpretability_utils.py # 核心工具函数
│   ├── attention_map.py       # 页面: Attention 热力图
│   ├── rope_explorer.py       # 页面: RoPE 可视化
│   └── ffn_activation.py      # 页面: FFN 激活探测
├── data_lab/                  # DataLab 模块 (数据工程)
│   ├── __init__.py
│   ├── data_utils.py          # 核心工具函数
│   ├── hf_dataset_viewer.py   # 页面: Dataset 透视镜
│   ├── cleaner_playground.py  # 页面: 数据清洗工坊
│   └── instruct_formatter.py  # 页面: 格式化转换器
├── model_lab/                 # ModelLab 模块 (模型相关)
│   ├── __init__.py
│   ├── model_utils.py         # 核心工具函数
│   ├── memory_estimator.py    # 页面: 显存估算
│   ├── peft_calculator.py     # 页面: PEFT 参数计算器
│   └── config_diff.py         # 页面: Config 差异对比
└── doc/                       # 项目文档
    ├── architecture.md        # 本文档
    └── design.md              # UI 设计规范
```

### 1.2 设计原则

1. **模块自治**: 每个功能模块包含自己的工具函数和页面
2. **共享最小化**: `shared/` 只存放真正全局共享的内容 (如样式)
3. **页面即模块**: 每个页面文件导出 `render()` 函数供 `app.py` 调用

---

## 2. 应用入口 (app.py)

### 2.1 导航结构

```python
NAV_STRUCTURE = {
    "TokenLab": {
        "分词编码": "playground",
        "模型对比": "arena",
        "Chat Template": "chat_builder"
    },
    "EmbeddingLab": {
        "向量运算": "vector_arithmetic",
        "模型对比": "embedding_comparison",
        "向量可视化": "vector_visualization",
        "语义相似度": "semantic_similarity"
    },
    "GenerationLab": {
        "Logits 显微镜": "logits_inspector",
        "Beam Search": "beam_visualizer",
        "KV Cache": "kv_cache_sim"
    },
    "InterpretabilityLab": {
        "Attention 热力图": "attention_map",
        "RoPE 可视化": "rope_explorer",
        "FFN 激活": "ffn_activation"
    },
    "DataLab": {
        "Dataset 透视镜": "hf_dataset_viewer",
        "数据清洗": "cleaner_playground",
        "格式转换": "instruct_formatter"
    },
    "ModelLab": {
        "显存估算": "memory_estimator",
        "PEFT 计算器": "peft_calculator",
        "Config 对比": "config_diff"
    }
}
```

### 2.2 页面加载模式

```python
if current_module == "logits_inspector":
    from generation_lab import logits_inspector
    logits_inspector.render()
```

---

## 3. 功能模块

### GenerationLab 模块

**核心定位**: 解构 `model.generate()` 的过程

#### 核心工具 (generation_utils.py)

| 函数 | 用途 |
|------|------|
| `load_model_and_tokenizer` | 加载模型和 tokenizer (带缓存) |
| `get_next_token_logits` | 获取下一个 token 的 logits 和概率 |
| `apply_temperature` | 应用温度缩放 |
| `apply_top_k` / `apply_top_p` | 采样截断 (Top-K / Nucleus) |
| `get_sampling_distribution` | 获取经过采样策略处理后的概率分布 |
| `beam_search_step` | 执行一步 Beam Search |
| `calculate_kv_cache_size` | 计算 KV Cache 显存占用 |
| `simulate_kv_cache_growth` | 模拟 Prefill/Decode 阶段 KV Cache 增长 |
| `simulate_paged_attention` | 模拟 PagedAttention Block 分配 |
| `get_model_config` | 获取预设模型配置 |
| `format_bytes` | 格式化字节数显示 |

#### 常量

| 常量 | 用途 |
|------|------|
| `DEMO_MODELS` | 轻量级演示模型配置 (GPT-2 系列) |
| `MODEL_CONFIGS` | 常见模型配置 (用于 KV Cache 计算) |

#### 页面

| 页面 | 功能 |
|------|------|
| `logits_inspector.py` | Logits 可视化、Temperature/Top-K/Top-P 演示、熵分析 |
| `beam_visualizer.py` | Beam Search 搜索树可视化 (含 `beam_search_with_history` 函数) |
| `kv_cache_sim.py` | KV Cache 显存计算、Prefill/Decode 模拟、PagedAttention |

---

### InterpretabilityLab 模块

**核心定位**: 打开 Transformer 的黑盒

#### 核心工具 (interpretability_utils.py)

| 函数 | 用途 |
|------|------|
| `load_model_with_attention` | 加载模型并启用 attention 输出 |
| `get_attention_weights` | 获取所有层所有头的注意力权重 |
| `apply_causal_mask` | 应用因果掩码 (下三角) |
| `compute_attention_entropy` | 计算注意力熵 (分布分散程度) |
| `get_attention_patterns` | 分析注意力模式 (对角线/首token/局部/全局) |
| `compute_rope_frequencies` | 计算 RoPE 旋转频率 |
| `apply_rope_rotation` | 对向量应用 RoPE 旋转 |
| `compute_rope_decay` | 计算 RoPE 相对位置衰减特性 |
| `analyze_ffn_activations` | 分析 FFN 层激活情况 |
| `compare_activation_functions` | 对比激活函数 (GELU/ReLU/SiLU/SwiGLU) |

#### 常量

| 常量 | 用途 |
|------|------|
| `INTERPRETABILITY_MODELS` | 轻量级模型配置 (GPT-2/DistilGPT-2) |
| `MODEL_ARCHITECTURES` | 模型架构对比信息 |

#### 页面

| 页面 | 功能 |
|------|------|
| `attention_map.py` | Attention 热力图、Causal Mask 演示 |
| `rope_explorer.py` | RoPE 旋转可视化、衰减特性 |
| `ffn_activation.py` | FFN 激活分析、激活函数对比 |

---

### DataLab 模块

**核心定位**: 数据工程实验室

#### 核心工具 (data_utils.py)

| 函数 | 用途 |
|------|------|
| `apply_cleaning_rule` | 应用单个清洗规则 |
| `clean_text` | 批量应用清洗规则 |
| `normalize_unicode` | Unicode 规范化 (NFC/NFD/NFKC/NFKD) |
| `convert_to_format` | 转换数据到指定格式 |
| `validate_chat_format` | 验证 Chat 格式是否正确 |
| `calculate_perplexity` | 计算文本的 PPL (困惑度) |
| `batch_calculate_ppl` | 批量计算多个文本的 PPL |
| `filter_by_ppl` | 根据 PPL 阈值过滤文本 |
| `get_ppl_quality_label` | 根据 PPL 值返回质量标签 |

#### 常量

```python
CHAT_TEMPLATES = {
    "alpaca": {...},
    "sharegpt": {...},
    "chatml": {...},
    "llama2": {...}
}

CLEANING_RULES = {
    "remove_html": {...},
    "remove_urls": {...},
    "normalize_whitespace": {...},
    ...
}

PPL_MODELS = {
    "GPT-2 (Small)": {...},
    "DistilGPT-2": {...}
}
```

#### 页面

| 页面 | 功能 |
|------|------|
| `hf_dataset_viewer.py` | HuggingFace Dataset 流式预览、分析、质量检查 |
| `cleaner_playground.py` | 清洗规则测试、PPL 过滤 |
| `instruct_formatter.py` | SFT 数据格式转换 (Alpaca/ShareGPT/ChatML/Llama-2) |

---

### ModelLab 模块 (扩展)

#### 新增页面

| 页面 | 功能 |
|------|------|
| `peft_calculator.py` | LoRA/QLoRA 参数量计算 |
| `config_diff.py` | 模型配置对比 |

---

## 4. 全局样式 (shared/styles.py)

### CSS 变量

```css
:root {
    --font-size-base: 14px;
    --color-accent: #2563EB;
    --color-success: #059669;
    --color-warning: #D97706;
    --color-error: #DC2626;
    --spacing-md: 12px;
}
```

### 使用方式

```python
from shared.styles import GLOBAL_CSS
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
```

---

## 5. 开发规范

### 新增页面

1. 在对应模块目录创建 `.py` 文件
2. 实现 `render()` 函数
3. 在 `app.py` 的 `NAV_STRUCTURE` 添加导航项
4. 在 `app.py` 的模块加载逻辑添加 import

### 缓存策略

```python
@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    ...

@st.cache_data
def compute_embeddings(texts: tuple):
    ...
```

---

## 6. 扩展指南

### 新增功能模块

1. 创建目录 `xxx_lab/`
2. 创建 `__init__.py`
3. 创建 `xxx_utils.py` (核心工具)
4. 创建页面文件
5. 在 `app.py` 的 `NAV_STRUCTURE` 添加新分组

---

## 附录: 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 3.1 | 2024-12 | 完善文档、补充缺失函数说明、增强 Dataset 透视镜、添加 PPL 过滤 |
| 3.0 | 2024-12 | 新增 GenerationLab、InterpretabilityLab、DataLab，扩展 ModelLab |
| 2.0 | 2024-12 | 重构项目结构，模块化设计 |
| 1.0 | 2024-12 | 初始版本 |
