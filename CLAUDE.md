# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 项目概述

**LLM Tools Workbench** 是一个面向 LLM 工程师的可视化工具集，用于探索大语言模型的底层机制。项目基于 Gradio 构建，包含六个独立的实验室模块。

核心设计理念：
- **底层优先**：深入可视化 LLM 内部机制，而非黑盒调用
- **交互探索**：支持参数调整与实时反馈
- **模块自治**：每个实验室独立维护工具函数与页面

---

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 启动应用（默认端口 7860）
python app_gradio.py
```

访问 `http://localhost:7860` 使用应用。

---

## 项目架构

### 核心入口

- **app_gradio.py**: Gradio 应用主入口，包含全局 CSS、主题定义、所有模块的 Tab 路由

### 模块结构

所有功能模块遵循统一的结构：

```
{module}_lab/
├── __init__.py
├── {module}_utils.py      # 核心工具函数（缓存、模型加载、计算逻辑）
├── page1.py               # 页面文件，导出 render() 函数
├── page2.py
└── page3.py
```

**六大模块**：

1. **token_lab/** - 分词实验室
   - `tokenizer_utils.py`: 分词器加载、缓存、压缩率计算、Byte Fallback 分析
   - `playground.py`: 交互式分词编码
   - `arena.py`: 多模型分词效果对比
   - `chat_builder.py`: Chat Template 渲染

2. **embedding_lab/** - 向量分析工作台
   - `embedding_utils.py`: Embedding 模型加载、向量运算、降维（PCA/t-SNE/UMAP）
   - `vector_arithmetic.py`: Word2Vec 类比推理
   - `model_comparison.py`: 稀疏/稠密表示对比
   - `vector_visualization.py`: 3D 空间可视化
   - `semantic_similarity.py`: Token 相似度热力图

3. **generation_lab/** - 生成机制探索
   - `generation_utils.py`: 模型加载、Logits 提取、采样策略（Temperature/Top-K/Top-P）、Beam Search、KV Cache 计算
   - `logits_inspector.py`: Next Token 概率分布可视化
   - `beam_visualizer.py`: Beam Search 搜索树
   - `kv_cache_sim.py`: KV Cache 显存模拟

4. **interpretability_lab/** - 可解释性分析
   - `interpretability_utils.py`: Attention 权重提取、RoPE 频率计算、FFN 激活分析
   - `attention_map.py`: Attention 热力图
   - `rope_explorer.py`: RoPE 旋转可视化
   - `ffn_activation.py`: 激活函数对比

5. **data_lab/** - 数据工程实验室
   - `data_utils.py`: 清洗规则、PPL 计算、格式转换
   - `hf_dataset_viewer.py`: HuggingFace 数据集流式加载
   - `cleaner_playground.py`: 数据清洗（HTML/URL/Unicode）
   - `instruct_formatter.py`: SFT 格式转换（Alpaca/ShareGPT/ChatML/Llama-2）

6. **model_lab/** - 模型工具箱
   - `model_utils.py`: 配置获取、参数量计算
   - `memory_estimator.py`: 多精度显存估算
   - `peft_calculator.py`: LoRA 参数量计算
   - `config_diff.py`: 双模型配置对比

### 关键常量定义位置

- **Token 颜色**: `app_gradio.py` 中的 `TOKEN_COLORS` 和 CSS `.token-color-*` 类
- **全局样式**: `app_gradio.py` 中的 `CUSTOM_CSS` 和 `CUSTOM_THEME`
- **模型列表**: 各 `*_utils.py` 文件（如 `token_lab/tokenizer_utils.py` 的 `MODEL_CATEGORIES`）

---

## 开发规范

### 页面文件规范

每个页面文件必须：
1. 导出 `render()` 函数供 `app_gradio.py` 调用
2. 所有 UI 组件创建在 `render()` 内部
3. 如需页面加载时初始化，返回包含 `load_fn` 和 `load_outputs` 的字典：
   ```python
   def render():
       # ... UI 组件定义 ...
       output_component = gr.Textbox()

       def load_data():
           return "初始值"

       return {
           'load_fn': load_data,
           'load_outputs': [output_component]
       }
   ```

### 缓存策略

- **模型/分词器加载**: 使用 `@lru_cache` 或模块级字典缓存（如 `_model_cache`）
- **计算结果**: 不缓存（Gradio 自动管理交互状态）

### 工具函数规范

在 `*_utils.py` 中定义：
- 模型加载函数（带缓存）
- 核心计算逻辑（如分词、向量运算、Logits 提取）
- 常量定义（模型列表、配置）

### 样式规范

- 禁止使用 emoji
- 使用 `CUSTOM_CSS` 中定义的样式类（如 `.module-title`, `.token`, `.stat-card`）
- Token 可视化使用 `TOKEN_COLORS` 数组循环取色
- 字体：界面用 Inter，代码用 JetBrains Mono

---

## 技术栈

| 分类 | 依赖 | 用途 |
|------|------|------|
| **核心** | `gradio` | Web 框架 |
| | `transformers` | Tokenizer & Model 加载 |
| | `plotly` | 交互式图表 |
| **深度学习** | `torch` | 模型推理、Logits 提取 |
| | `accelerate` | 显存估算 |
| **Embedding** | `sentence-transformers` | Dense Embedding |
| | `gensim` | Word2Vec/GloVe |
| **降维** | `scikit-learn` | PCA, t-SNE |
| | `umap-learn` | UMAP 降维 |
| **数据处理** | `datasets` | HuggingFace 数据集 |
| | `pandas`, `numpy` | 数据处理 |

---

## 关键设计模式

### 1. 模块路由机制

`app_gradio.py` 使用嵌套 Tab 结构，主 Tab 对应模块，子 Tab 对应页面。页面导入使用延迟加载：

```python
with gr.Tab("TokenLab"):
    from token_lab import playground, arena
    with gr.Tabs():
        with gr.Tab("分词编码"):
            playground.render()
```

### 2. 页面加载初始化

部分页面需在加载时初始化（如填充模型列表）。这些页面的 `render()` 返回字典，主应用在 `app.load()` 中统一执行：

```python
load_events = []
result = some_page.render()
if result:
    load_events.append(result)

# 在 app.load() 中执行
app.load(fn=combined_load, outputs=all_load_outputs)
```

### 3. 工具函数缓存

避免重复加载模型，使用：
- **轻量级**: `@lru_cache` (如分词器)
- **重量级**: 模块级字典 `_model_cache` (如 GPT-2 模型)

### 4. 演示模型选择

各模块优先使用轻量级模型（GPT-2 系列、DistilGPT-2）用于演示，避免下载大模型。模型配置定义在 `*_utils.py` 的常量中。

---

## 常见任务

### 新增页面

1. 在对应 `*_lab/` 目录创建 `new_page.py`
2. 实现 `render()` 函数
3. 在 `app_gradio.py` 对应模块的 Tab 中添加：
   ```python
   from {module}_lab import new_page
   with gr.Tab("新页面标题"):
       new_page.render()
   ```

### 新增模块

1. 创建目录 `new_lab/`
2. 创建 `__init__.py` 和 `new_utils.py`
3. 创建页面文件
4. 在 `app_gradio.py` 添加新的主 Tab

### 修改全局样式

编辑 `app_gradio.py` 中的 `CUSTOM_CSS` 或 `CUSTOM_THEME`。

### 调试页面加载问题

检查：
1. `render()` 返回值格式是否正确
2. `load_events` 是否正确收集
3. `combined_load()` 中的异常处理

---

## 重要注意事项

1. **依赖环境变量**: `token_lab/tokenizer_utils.py` 设置 `TRANSFORMERS_VERBOSITY=error` 抑制警告
2. **设备管理**: 所有模型加载默认使用 `device_map="cpu"`，避免 GPU 依赖
3. **精度选择**: 使用 `torch.float32` 而非 `float16`，确保 CPU 兼容
4. **HuggingFace 缓存**: 模型自动缓存到 `~/.cache/huggingface/`
5. **错误处理**: 工具函数应捕获异常并返回友好错误信息，避免页面崩溃

---

## 相关文档

- `doc/architecture.md` - 详细架构设计与扩展指南
- `doc/design.md` - UI 设计规范（字号、配色、间距系统）
- `README.md` - 用户使用文档

---

## 代码风格

- Python 遵循 PEP 8
- 类型提示：工具函数必须标注参数和返回值类型
- 文档字符串：核心函数使用 Google 风格 docstring
- **界面文字**：所有用户界面文字（label、placeholder、markdown标题等）使用英文
- **注释和docstring**：保持中文，便于中文开发者理解
- 中英文之间增加空格（文档文案）
- 禁止使用 emoji

## 界面语言规范

- **用户可见文字**：所有 Gradio 组件的 label、placeholder、按钮文字、Markdown 标题等**全部英文化**
- **代码注释**: 保持中文，便于中文开发者维护
- **函数文档字符串 (docstring)**: 保持中文
- **图表说明**：图表标题、轴标签全部英文；复杂说明通过外部 Markdown 提供，不放在图表内部
- **图表优化**：所有 Plotly 图表已添加 `autosize=True` 和合理的 `height` 参数，确保正确显示

示例：
```python
def render_plot():
    """渲染数据可视化图表"""  # docstring: 中文

    # 创建图表 - 注释：中文
    fig = go.Figure()

    fig.update_layout(
        title="Performance Metrics",  # 图表标题：英文
        xaxis_title="Time",  # 轴标签：英文
        yaxis_title="Accuracy"
    )

    return fig

def render():
    """渲染页面"""  # docstring: 中文

    gr.Markdown("## Model Comparison")  # 用户界面：英文

    input_box = gr.Textbox(
        label="Input Text",  # 英文
        placeholder="Enter text here..."  # 英文
    )
```

---

## 用户偏好设定

### 语言策略

| 场景 | 语言 | 说明 |
|------|------|------|
| 产品页面 UI | 英文 | label、placeholder、Markdown 标题、按钮文字 |
| 工程文档 | 中文 | CLAUDE.md、架构文档等 |
| 代码注释 | 中文 | 便于中文开发者维护 |
| 函数 docstring | 中文 | 便于中文开发者理解 |
| 对话交流 | 中文 | 与开发者的沟通 |

### Tab 页面 UX 规范

1. **首次渲染**：当用户切换到某个 Tab 时，确保所有图表正确触发渲染（使用 `tab.select()` 事件）
2. **图表自适应**：所有 Plotly 图表必须包含 `autosize=True` 和合理的 `height` 参数
3. **默认值规则**：
   - 输入框**有**默认值 → 自动计算并显示结果
   - 输入框**无**默认值 → 不显示任何计算结果（等待用户输入）

### 图表最佳实践

```python
fig.update_layout(
    autosize=True,      # 必须
    height=450,         # 合理高度
    plot_bgcolor='#FFFFFF',
    paper_bgcolor='#FFFFFF'
)
```

对于使用 Tab 的页面，需要在 `tab.select()` 中重新渲染图表：

```python
with gr.Tab("My Tab") as my_tab:
    plot = gr.Plot(value=render_chart())

my_tab.select(fn=render_chart, outputs=[plot])
```
