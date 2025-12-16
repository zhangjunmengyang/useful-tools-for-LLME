# Gradio 迁移计划

## 项目背景

将 LLM Tools Workbench 从 Streamlit 重构为 Gradio 实现。保持所有原有功能，但使用 Gradio 的组件和布局系统，必须确保用户友好且美观易用。

## 迁移说明 (更新于 2025-12-16)

- **命名规范**: 不再使用 `*_gradio.py` 后缀，直接使用原文件名
- **代码风格**: 禁止使用 emoji，避免 AI 味的编码风格
- **Streamlit 版本**: 已删除，不再保留

---

## 已完成

### 1. 主应用入口
- [x] `app_gradio.py` - 主入口文件，包含全局 CSS 样式和 Tab 导航结构
  - 已更新导入路径，移除 emoji

### 2. TokenLab 模块 (3/3)
- [x] `token_lab/playground_gradio.py` - 分词编码
- [x] `token_lab/arena_gradio.py` - 模型对比
- [x] `token_lab/chat_builder_gradio.py` - Chat Template 调试器

> 注: TokenLab 模块保留 `*_gradio.py` 命名，待后续统一重命名

### 3. EmbeddingLab 模块 (4/4) - 已完成
- [x] `embedding_lab/vector_arithmetic.py` - 向量运算 (Word2Vec 类比)
- [x] `embedding_lab/model_comparison.py` - 模型对比 (TF-IDF/BM25/Dense)
- [x] `embedding_lab/vector_visualization.py` - 向量可视化 (PCA/t-SNE/UMAP)
- [x] `embedding_lab/semantic_similarity.py` - 语义相似度 (热力图+各向异性)

### 4. GenerationLab 模块 (3/3) - 已完成
- [x] `generation_lab/logits_inspector.py` - Logits 检查器
- [x] `generation_lab/beam_visualizer.py` - Beam Search 可视化
- [x] `generation_lab/kv_cache_sim.py` - KV Cache 模拟器
- [x] `generation_lab/generation_utils.py` - 工具函数 (已移除 Streamlit 依赖)

### 5. InterpretabilityLab 模块 (3/3) - 已完成
- [x] `interpretability_lab/attention_map.py` - 注意力图可视化
- [x] `interpretability_lab/rope_explorer.py` - RoPE 探索器
- [x] `interpretability_lab/ffn_activation.py` - FFN 激活分析
- [x] `interpretability_lab/interpretability_utils.py` - 工具函数 (已移除 Streamlit 依赖)

### 6. DataLab 模块 (3/3) - 已完成
- [x] `data_lab/hf_dataset_viewer.py` - HuggingFace Dataset 流式预览与分析
  - 样本预览: 分页浏览器、原始 JSON
  - 数据统计: 字段结构、统计表格、分布分析图
  - 质量检查: 质量报告、问题样本展示
- [x] `data_lab/cleaner_playground.py` - 数据清洗
  - 规则清洗: 规则选择、Unicode 规范化、自定义正则、实时预览
  - PPL 过滤: 模型选择、阈值设置、单条/批量计算
- [x] `data_lab/instruct_formatter.py` - 格式化转换器
  - JSON 输入、目标格式选择、System Prompt 自定义、格式验证
- [x] `data_lab/data_utils.py` - 工具函数 (无 Streamlit 依赖)

### 7. ModelLab 模块 (3/3) - 已完成
- [x] `model_lab/memory_estimator.py` - 显存估算
  - 模型名称输入、模型库选择、精度类型多选
  - 结果: 主表格、训练阶段详情、使用建议
- [x] `model_lab/peft_calculator.py` - PEFT 参数计算器
  - 模型选择 (预设/自定义)、LoRA 参数配置 (rank, alpha)
  - 目标模块选择、参数量/显存估算、参数分布表
- [x] `model_lab/config_diff.py` - Config 差异对比
  - 双模型选择 (预设/自定义)、配置对比表格
  - 关键差异分析 (GQA、RoPE)、参数量估算
- [x] `model_lab/model_utils.py` - 工具函数 (已移除 Streamlit 依赖)

---

## 待完成

### 8. 最终步骤

#### 8.1 TokenLab 重命名 (可选)
TokenLab 模块仍使用 `*_gradio.py` 命名，可考虑统一为无后缀命名：
- `token_lab/playground_gradio.py` -> `token_lab/playground.py`
- `token_lab/arena_gradio.py` -> `token_lab/arena.py`
- `token_lab/chat_builder_gradio.py` -> `token_lab/chat_builder.py`

#### 8.2 创建启动脚本
```bash
# run_gradio.sh
python app_gradio.py
```

#### 8.3 测试
- 检查样式一致性
- 验证数据流正确性和逻辑正确性

---

## 重要 utils 文件

这些文件已确认无 Streamlit 依赖：
- `token_lab/tokenizer_utils.py`
- `embedding_lab/embedding_utils.py`
- `generation_lab/generation_utils.py`
- `interpretability_lab/interpretability_utils.py`
- `data_lab/data_utils.py`
- `model_lab/model_utils.py`

---

## 迁移进度摘要

| 模块 | 状态 | 文件数 |
|------|------|--------|
| app_gradio.py | 完成 | 1 |
| TokenLab | 完成 | 3 |
| EmbeddingLab | 完成 | 4 + utils |
| GenerationLab | 完成 | 3 + utils |
| InterpretabilityLab | 完成 | 3 + utils |
| DataLab | 完成 | 3 + utils |
| ModelLab | 完成 | 3 + utils |

**总计**: 所有核心模块已完成 Gradio 迁移

---

## 设计原则

1. **功能不变**: 保持与 Streamlit 版本完全一致的功能
2. **用户友好**: 使用 Gradio 组件优化交互体验
3. **工程可维护**: 清晰的代码结构，便于后续扩展
4. **代码风格**: 禁止 emoji，避免 AI 味编码
