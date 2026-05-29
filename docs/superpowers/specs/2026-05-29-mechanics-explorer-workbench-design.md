# Mechanics Explorer Workbench Design

## Goal

将现有 LLM Tools Workbench 从 Gradio 页面集合重构为一个面向 LLM 工程师的 **Mechanics Explorer**。第一版不追求旧页面一比一迁移，而是围绕“理解大模型机制”重新组织体验，并把底层能力沉淀成普通 HTTP API，供未来 OpenClaw、Hermes、Codex skills 等 agent 直接调用。

## Product Positioning

新产品有两层定位：

- **Human Workbench**：React 前端，面向人类使用者，核心体验是沿着 LLM 机制链路探索、调参、观察结果、复制 API 调用。
- **Mechanics Toolkit**：Python 工具内核，面向 agent 和自动化脚本，提供 stateless HTTP endpoints 和 CLI fallback。

第一版不做 MCP。后续如果 API contract 稳定且确实需要 connector 化，再单独包一层 MCP server。

## Core Decisions

- UI 技术边界：抛弃 Gradio 作为新产品 UI 约束，使用 React 构建主工作台。
- 计算内核：复用现有 `workbench_tools` registry 和各 lab 的 Python utility 能力。
- 服务层：使用 FastAPI 暴露普通 JSON API。
- 持久化：第一版无 run history、无 session、无 project workspace。每次请求即时返回结果。
- Artifact：保留可选 Markdown/JSON 导出能力，但 artifact 是显式输出动作，不代表系统维护历史状态。
- Agent 集成：通过 skills 教 agent 调用 HTTP endpoint、传 JSON 参数、解析返回结果。CLI 作为批处理和复现实验保底。

## Primary User Story

用户不是来“找一个工具按钮”，而是来理解一次 LLM 机制：

1. 用户沿左侧 Pipeline Rail 选择机制阶段。
2. 用户在当前阶段选择一个实验工具或预设。
3. 用户输入文本、logits、dataset sample、trace JSON 或模型配置。
4. 工作台立即运行 stateless API。
5. 中央画布展示机制可视化结果。
6. 右侧面板解释关键发现，并展示 API endpoint、request payload、response schema、copy curl 和可选 export。

## Information Architecture

主导航保留 Pipeline Rail 形态，但一级分类不锁死为 5 个。第一版使用 7 个一级类，覆盖当前内容并保持边界清楚。

### 1. Input & Tokens

文本进入模型前发生什么。

当前能力映射：

- Tokenizer Encode
- Unicode Analysis
- Tokenizer Arena
- Chat Template
- compression stats
- byte fallback analysis

典型问题：

- 同一句话为什么不同模型 token 数差异很大？
- Unicode 规范化会不会改变模型输入？
- chat template 最终喂给模型的字符串长什么样？

### 2. Representation Space

token、word、sentence 如何进入向量空间，语义相似度如何出现。

当前能力映射：

- Vector Arithmetic
- Embedding Model Comparison
- Vector Visualization
- Semantic Similarity

典型问题：

- dense embedding 和 sparse retrieval 的行为差异在哪里？
- 向量空间里的相近是否等于任务上可召回？
- 类比推理和 bias 能不能被可视化解释？

### 3. Probability & Decoding

模型输出 logits 后，如何变成下一个 token。

当前能力映射：

- Logits Inspector
- Sampling Distribution
- Temperature
- Top-K
- Top-P
- Beam Search

典型问题：

- temperature 改变的是概率形状还是排序？
- top-k/top-p 如何截断候选 token？
- beam search 为什么有时更保守？

### 4. Transformer Anatomy

Transformer 内部机制本身。

当前能力映射：

- Attention Map
- RoPE Explorer
- FFN Activation
- KV Cache Growth
- KV Cache Estimate

典型问题：

- attention 权重实际集中在哪里？
- RoPE 距离衰减如何影响长上下文？
- prefill/decode 阶段 KV cache 如何增长？
- FFN 激活函数之间有什么行为差异？

### 5. Data & Context

数据集和上下文进入模型前的工程层。数据集不是 RAG 的附属品；retrieval 是 Data & Context 下面的一个重要场景。

当前能力映射：

- Dataset Viewer
- Dataset Quality Check
- Data Cleaning
- Instruction Format
- Chunking
- Retrieval Simulator
- RAG Lexical Retrieval

边界规则：

- 训练/微调数据集本身属于 Data & Context。
- RAG 文档库、知识库 corpus 属于 Data & Context。
- benchmark/eval 数据集属于 Evaluation & Traces。
- training token count、batch、epoch 等成本变量属于 Adaptation & Cost。

典型问题：

- dataset 字段是否适合当前训练或评测任务？
- 样本是否有重复、空字段、长度异常？
- chunking 策略如何改变上下文边界？
- lexical retrieval 为什么召回了这批文档？

### 6. Adaptation & Cost

模型如何被改造、训练、部署时成本如何变化。

当前能力映射：

- Memory Estimator
- PEFT Calculator
- LoRA Explorer
- Training Cost
- Config Diff

典型问题：

- LoRA rank 和 target modules 如何影响参数量？
- 某个模型配置下显存是否足够？
- 全参微调和 LoRA 的 FLOPs、时间、费用差多少？
- 两个模型 config 的关键差异是什么？

### 7. Evaluation & Traces

如何判断系统行为，以及一次 model/agent run 到底发生了什么。

当前能力映射：

- Eval Metrics
- LLM Judge
- Benchmark Explorer
- Eval Pipeline
- Trace Viewer
- Trace Analyzer

典型问题：

- 当前 predictions/references 的指标如何？
- judge 评分和 pairwise comparison 结果如何解释？
- agent trace 的 critical path 和 bottleneck 在哪里？
- benchmark 结果在哪些任务维度上失衡？

## Layout

桌面端采用三栏工作台：

- 左栏：Pipeline Rail。固定显示 7 个一级类，并展示简短 subtitle。
- 中栏：Mechanics Canvas。显示当前机制的主实验、图表、token/trace/table 可视化。
- 右栏：Inspector。包含 controls、explanation、API drawer、export controls。

移动端采用顺序折叠：

- 顶部选择当前一级类。
- 然后显示工具选择。
- Canvas 在 controls 之前或之后根据页面任务决定，但 API drawer 默认折叠。

## API Design

第一版 API stateless，不提供 run history。

核心 endpoint：

- `GET /api/tools`
- `GET /api/tools/{tool_id}`
- `POST /api/tools/{tool_id}/run`
- `POST /api/tools/{tool_id}/export`

`POST /api/tools/{tool_id}/run` 返回：

- `tool_id`
- `status`
- `inputs`
- `result`
- `duration_ms`
- `error`

`POST /api/tools/{tool_id}/export` 返回：

- `tool_id`
- `status`
- `result`
- `artifact`
- `duration_ms`
- `error`

API 不需要 server-side session。前端如果需要保留最近一次结果，使用 client state。

## Skill Integration

未来 OpenClaw、Hermes、Codex skills 直接描述：

- 何时调用某个 endpoint。
- 请求 payload schema。
- 典型参数示例。
- 返回字段解释。
- 出错时如何调整输入。

因为 API 是普通 HTTP，skills 不依赖 MCP、浏览器自动化或 Gradio 页面结构。

## Visual Direction

视觉语言继续参考 `VoltAgent/awesome-design-md` 中的 Vercel、Linear、Raycast 方向：

- 白底、黑色主操作、低饱和灰阶、单一蓝色链接/信息色。
- 密集但清楚的工程工作台，不做 marketing hero。
- 左侧 rail 紧凑，内容区强调可视化实验。
- Cards 只用于工具项、结果片段、inspector panel，不堆叠大面积装饰卡片。
- 所有用户可见 UI 文案使用英文。

## Non-Goals

- 不做 Gradio 页面 parity。
- 不做 MCP server。
- 不做登录、多用户、云同步。
- 不做 run history、session、project workspace。
- 不在第一版重写所有旧 lab 页面。
- 不把工作台做成纯 API catalog 或纯课程目录。

## Validation Criteria

设计实现完成后至少验证：

- React app desktop 截图中能清楚看到 Pipeline Rail、Canvas、Inspector 三栏。
- mobile 截图中一级导航、工具选择、结果区域不重叠。
- 每个一级类至少有一个当前可运行工具接入真实 API。
- `GET /api/tools` 能返回工具元数据和所属一级类。
- `POST /api/tools/{tool_id}/run` 对无模型下载工具可在本地快速返回。
- 右侧 API drawer 展示 endpoint、payload、response schema、copy curl。
- 无持久化：刷新页面不会从服务端恢复 run history。
