import type { Language } from "@/lib/i18n";
import type { LabPage, MechanicsCategory, ToolSpec } from "@/types";

const TOOL_ZH: Record<string, { label: string; description: string }> = {
  unicode_analyze: {
    label: "Unicode 分析",
    description: "看字符、UTF-8 字节，以及不同规范化结果差在哪。",
  },
  tokenizer_encode: {
    label: "分词编码",
    description: "用 HuggingFace 词表把文本编成 token，并给出长度和压缩信息。",
  },
  vector_similarity: {
    label: "向量相似度",
    description: "对给定向量算余弦相似度矩阵，过程可核对。",
  },
  sampling_distribution: {
    label: "采样分布",
    description: "对一组 logits 做温度、top-k、top-p，看下一个 token 怎么抽。",
  },
  rope_frequencies: {
    label: "RoPE 频率",
    description: "算出 RoPE 频率矩阵，以及相对距离变远时的衰减。",
  },
  ffn_activation_compare: {
    label: "激活函数比较",
    description: "对比 GELU、ReLU、SiLU 和简化 SwiGLU 的曲线。",
  },
  kv_cache_growth: {
    label: "KV Cache 增长",
    description: "模拟预填充和逐 token 解码时，缓存怎么随长度涨。",
  },
  kv_cache_estimate: {
    label: "KV Cache 估算",
    description: "按层数、隐层维和序列长度估解码时的缓存内存。",
  },
  dataset_quality_check: {
    label: "数据集质量检查",
    description: "查重复、空文本和长度分布。",
  },
  data_clean: {
    label: "数据清洗",
    description: "按固定规则清洗文本，方便准备训练数据。",
  },
  instruct_format: {
    label: "指令格式转换",
    description: "把一条指令样本转成 Alpaca、ShareGPT、ChatML 或 Llama-2 格式。",
  },
  rag_chunk: {
    label: "RAG 切块",
    description: "把长文切成块，并给出块数和长度统计。",
  },
  rag_lexical_retrieval: {
    label: "词法检索",
    description: "用词重叠给文档排序，方便对照检索结果。",
  },
  lora_params_estimate: {
    label: "LoRA 参数估算",
    description: "估算可训练的 LoRA 参数量，以及可选的显存开销。",
  },
  training_cost_estimate: {
    label: "训练成本估算",
    description: "估算微调的 FLOPs、墙钟时间和 GPU 费用。",
  },
  eval_metrics: {
    label: "评测指标",
    description: "算常见文本生成评测指标。",
  },
  trace_analyze: {
    label: "轨迹分析",
    description: "看轨迹耗时、关键路径和瓶颈。",
  },
};

const CATEGORY_ZH: Record<string, { label: string; subtitle: string; description: string }> = {
  input_tokens: {
    label: "输入与分词",
    subtitle: "文本变成模型能读的 token。",
    description: "看分词、Unicode 规范化、压缩比和对话模板怎么展开。",
  },
  representation_space: {
    label: "表示空间",
    subtitle: "向量、相似度和潜空间几何。",
    description: "看嵌入空间、向量运算、语义相似，以及稀疏和稠密的差别。",
  },
  probability_decoding: {
    label: "概率与解码",
    subtitle: "从 logits 到下一个 token。",
    description: "看 logits、温度、top-k、top-p 和束搜索怎么改分布。",
  },
  transformer_anatomy: {
    label: "Transformer 内部",
    subtitle: "注意力、RoPE、FFN 和 KV Cache。",
    description: "看模型内部，以及推理时内存怎么涨。",
  },
  data_context: {
    label: "数据与上下文",
    subtitle: "进模型之前的数据和上下文。",
    description: "看数据集、清洗、格式、切块和检索诊断。",
  },
  adaptation_cost: {
    label: "适配与成本",
    subtitle: "微调、显存和预算。",
    description: "估 LoRA 参数、训练费用、模型内存和配置差异。",
  },
  evaluation_traces: {
    label: "评测与轨迹",
    subtitle: "指标、评判器和一次运行的行为。",
    description: "评预测结果，并查看模型或智能体轨迹。",
  },
};

const LAB_GROUP_ZH: Record<string, { group: string; description: string }> = {
  "Research Toolbox": {
    group: "研究工具箱",
    description: "跑可复用工具，并导出结构化结果。",
  },
  "Core Mechanics": {
    group: "核心机制",
    description: "看 token、向量、生成和模型内部。",
  },
  "Knowledge & Data": {
    group: "知识与数据",
    description: "准备数据集，并核对检索行为。",
  },
  "Model Ops": {
    group: "模型运维",
    description: "估显存、微调费用和推理约束。",
  },
  Evaluation: {
    group: "评测",
    description: "看智能体轨迹，并比较评测流水线。",
  },
};

const LAB_PAGE_ZH: Record<string, string> = {
  toolbox_tool_runner: "工具运行器",
  token_playground: "试玩台",
  token_arena: "对台",
  token_chat_template: "对话模板",
  embedding_vector_arithmetic: "向量运算",
  embedding_model_comparison: "模型对比",
  embedding_visualization: "可视化",
  embedding_semantic_similarity: "语义相似",
  generation_logits: "Logits 检查",
  generation_beam: "束搜索",
  generation_kv_cache: "KV Cache",
  interpretability_attention: "注意力",
  interpretability_rope: "RoPE 探索",
  interpretability_ffn: "FFN 激活",
  data_dataset_viewer: "数据集查看",
  data_cleaner: "数据清洗",
  data_formatter: "格式转换",
  rag_chunking: "切块",
  rag_retrieval: "检索",
  model_memory: "显存估算",
  model_peft: "PEFT 计算",
  model_config_diff: "配置对比",
  finetune_lora: "LoRA 探索",
  finetune_training_cost: "训练成本",
  inference_throughput: "吞吐",
  inference_quantization: "量化",
  agent_trace_viewer: "轨迹查看",
  agent_trace_analyzer: "轨迹分析",
  eval_benchmark: "Benchmark",
  eval_llm_judge: "LLM Judge",
  eval_pipeline: "评测流水线",
};

const LAB_NAME_ZH: Record<string, string> = {
  Token: "Token",
  Embedding: "Embedding",
  Generation: "Generation",
  Interpretability: "可解释",
  Data: "数据",
  RAG: "RAG",
  Model: "模型",
  FineTune: "微调",
  Inference: "推理",
  "Agent Trace": "智能体轨迹",
  Eval: "评测",
  Toolbox: "工具箱",
};

const FIELD_ZH: Record<string, string> = {
  text: "文本",
  model_name: "词表 / 模型名",
  vectors: "向量 (JSON)",
  labels: "标签 (JSON)",
  dim: "维数（偶数）",
  max_position: "最大位置",
  max_distance: "最大距离",
  base: "base",
  x_values: "x 取值 (JSON)",
  logits: "logits (JSON)",
  tokens: "token (JSON)",
  temperature: "温度",
  top_k: "top-k",
  top_p: "top-p",
  prompt_length: "提示长度",
  generation_length: "生成长度",
  num_layers: "层数",
  hidden_size: "隐层维",
  seq_length: "序列长度",
  rank: "秩",
  target_modules: "目标模块 (JSON)",
  intermediate_size: "中间层维",
  num_heads: "头数",
  model_params: "参数量",
  gpu_tflops: "GPU TFLOPS",
  cost_per_hour: "每小时费用",
  mfu: "MFU",
  rules: "规则 (JSON)",
  samples: "样本 (JSON)",
  text_fields: "文本字段 (JSON)",
  data: "数据 (JSON)",
  target_format: "目标格式",
  method: "切块方法",
  chunk_size: "块大小",
  overlap: "重叠",
  query: "查询",
  documents: "文档 (JSON)",
  predictions: "预测 (JSON)",
  references: "参考 (JSON)",
  trace_json: "轨迹 JSON",
};

const FIELD_EN: Record<string, string> = {
  text: "Text",
  model_name: "Vocab / model name",
  vectors: "Vectors (JSON)",
  labels: "Labels (JSON)",
  dim: "Dim (even)",
  max_position: "Max position",
  max_distance: "Max distance",
  base: "base",
  x_values: "x values (JSON)",
  logits: "logits (JSON)",
  tokens: "tokens (JSON)",
  temperature: "Temperature",
  top_k: "top-k",
  top_p: "top-p",
  prompt_length: "Prompt length",
  generation_length: "Generation length",
  num_layers: "Layers",
  hidden_size: "Hidden size",
  seq_length: "Sequence length",
  rank: "Rank",
  target_modules: "Target modules (JSON)",
  intermediate_size: "Intermediate size",
  num_heads: "Heads",
  model_params: "Parameters",
  gpu_tflops: "GPU TFLOPS",
  cost_per_hour: "Cost per hour",
  mfu: "MFU",
  rules: "Rules (JSON)",
  samples: "Samples (JSON)",
  text_fields: "Text fields (JSON)",
  data: "Data (JSON)",
  target_format: "Target format",
  method: "Chunk method",
  chunk_size: "Chunk size",
  overlap: "Overlap",
  query: "Query",
  documents: "Documents (JSON)",
  predictions: "Predictions (JSON)",
  references: "References (JSON)",
  trace_json: "Trace JSON",
};

export function toolLabel(language: Language, tool: Pick<ToolSpec, "id" | "label">): string {
  if (language === "zh") return TOOL_ZH[tool.id]?.label ?? tool.label;
  return tool.label;
}

export function toolDescription(language: Language, tool: Pick<ToolSpec, "id" | "description">): string {
  if (language === "zh") return TOOL_ZH[tool.id]?.description ?? tool.description;
  return tool.description;
}

export function categoryCopy(language: Language, category: MechanicsCategory): MechanicsCategory {
  if (language !== "zh") return category;
  const zh = CATEGORY_ZH[category.id];
  if (!zh) return category;
  return { ...category, ...zh };
}

export function labPageLabel(language: Language, page: LabPage): string {
  if (language === "zh") return LAB_PAGE_ZH[page.id] ?? page.label;
  return page.label;
}

export function labName(language: Language, page: LabPage): string {
  if (language === "zh") return LAB_NAME_ZH[page.lab_label] ?? page.lab_label;
  return page.lab_label;
}

export function labGroup(language: Language, group: string): string {
  if (language === "zh") return LAB_GROUP_ZH[group]?.group ?? group;
  return group;
}

export function labGroupDescription(language: Language, page: LabPage): string {
  if (language === "zh") return LAB_GROUP_ZH[page.group]?.description ?? page.group_description;
  return page.group_description;
}

export function fieldLabel(language: Language, key: string, fallback: string): string {
  const table = language === "zh" ? FIELD_ZH : FIELD_EN;
  return table[key] ?? fallback;
}
