# Eval Lab — 自动化评测可视化

## 定位
第11个 Lab 模块，展示 LLM 评测方法论和自动化评测能力。

## 核心功能（3个Tab）

### Tab 1: Benchmark Explorer
- 可视化主流 benchmark 体系（MMLU / HumanEval / GSM8K / MT-Bench / IFEval 等）
- 交互式对比：选择多个模型，雷达图/柱状图展示各维度得分
- 数据来源：Open LLM Leaderboard JSON + 手动补充
- **亮点**: 自动拉取 HuggingFace leaderboard 最新数据

### Tab 2: LLM-as-Judge 演示
- 用户输入 prompt + 两个模型的回答
- 展示 LLM-as-Judge 评分流程（pointwise / pairwise / reference-based）
- 可视化评分 rubric + 评分理由
- 支持自定义评分维度（helpfulness / accuracy / safety / creativity）
- **亮点**: 直观展示 judge 偏差（position bias / verbosity bias）

### Tab 3: 自动化评测 Pipeline
- 上传测试集（JSON/CSV），选择评测方式
- 支持多种评测指标：BLEU / ROUGE / BERTScore / Exact Match / Pass@k
- 批量运行 + 结果可视化（confusion matrix / error analysis）
- 导出评测报告（Markdown）
- **亮点**: 一键对比多个模型在自定义数据集上的表现

## 技术方案
- 无需外部 API（用本地小模型做 judge 演示，或 mock 数据）
- 指标计算用 evaluate / rouge-score / nltk
- 可视化用 Plotly + Gradio 原生组件
- 数据缓存避免重复拉取

## 文件结构
```
eval_lab/
├── __init__.py
├── eval_utils.py          # 评测指标计算
├── benchmark_explorer.py  # Tab 1
├── llm_judge.py           # Tab 2
├── eval_pipeline.py       # Tab 3
└── DESIGN.md
```

## 面试展示价值
- 评测方法论的系统理解（不只是跑分，理解指标含义和局限性）
- LLM-as-Judge 的偏差分析能力
- 工程化评测 pipeline 的设计和实现
