import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson51Diagram: LessonDiagram = {
  lessonId: "51",
  title: "分步推理必须引用视觉证据",
  summary:
    "图像先被写成带标签的阶段文本。过程奖励只看推理 span 是否点到真值格子；答案对而引用为空时过程分为 0。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l51-image",
      label: ["图像 + 计数题"],
      meta: "红杯在哪些格",
      kind: "input",
      x: 88,
      y: 180,
      width: 150,
    },
    {
      id: "l51-stages",
      label: ["阶段标签"],
      meta: "SUMMARY / CAPTION",
      kind: "transform",
      x: 268,
      y: 180,
      width: 150,
    },
    {
      id: "l51-reason",
      label: ["推理 token"],
      meta: "T_reason",
      kind: "state",
      x: 458,
      y: 88,
      width: 140,
    },
    {
      id: "l51-answer",
      label: ["答案 token"],
      meta: "T_ans ∩ T_reason=∅",
      kind: "state",
      x: 458,
      y: 272,
      width: 150,
    },
    {
      id: "l51-cite",
      label: ["引用格子？"],
      meta: "C(y) ∩ G",
      kind: "decision",
      x: 648,
      y: 88,
      width: 140,
    },
    {
      id: "l51-rans",
      label: ["答案奖励"],
      meta: "r_ans = 1[â=a*]",
      kind: "decision",
      x: 648,
      y: 272,
      width: 150,
    },
    {
      id: "l51-rproc",
      label: ["过程奖励"],
      meta: "无引用则 0",
      kind: "output",
      x: 838,
      y: 180,
      width: 140,
    },
  ],
  edges: [
    {
      id: "l51-e-image-stages",
      from: "l51-image",
      to: "l51-stages",
      label: "看见图",
      labelAt: { x: 178, y: 228 },
    },
    {
      id: "l51-e-stages-reason",
      from: "l51-stages",
      to: "l51-reason",
      label: "REASONING",
      via: [
        { x: 348, y: 180 },
        { x: 348, y: 88 },
      ],
      labelAt: { x: 304, y: 124 },
    },
    {
      id: "l51-e-stages-answer",
      from: "l51-stages",
      to: "l51-answer",
      label: "CONCLUSION",
      via: [
        { x: 348, y: 180 },
        { x: 348, y: 272 },
      ],
      labelAt: { x: 304, y: 236 },
    },
    {
      id: "l51-e-reason-cite",
      from: "l51-reason",
      to: "l51-cite",
      label: "解析 (i,j)",
      labelAt: { x: 552, y: 64 },
    },
    {
      id: "l51-e-answer-rans",
      from: "l51-answer",
      to: "l51-rans",
      label: "对金标",
      labelAt: { x: 552, y: 296 },
    },
    {
      id: "l51-e-cite-rproc",
      from: "l51-cite",
      to: "l51-rproc",
      label: "有交则 1",
      via: [
        { x: 738, y: 88 },
        { x: 738, y: 180 },
      ],
      labelAt: { x: 758, y: 124 },
    },
    {
      id: "l51-e-rans-rproc",
      from: "l51-rans",
      to: "l51-rproc",
      label: "合取可选",
      via: [
        { x: 738, y: 272 },
        { x: 738, y: 180 },
      ],
      labelAt: { x: 758, y: 236 },
    },
    {
      id: "l51-e-reason-rans",
      from: "l51-reason",
      to: "l51-rans",
      label: "不得混 span",
      via: [
        { x: 508, y: 180 },
      ],
      labelAt: { x: 428, y: 180 },
    },
  ],
  steps: [
    {
      title: "先看见图再写阶段",
      description:
        "直接吐数字会跳过看见了什么。LLaVA-CoT 用 SUMMARY / CAPTION / REASONING / CONCLUSION 把一次生成切成可检索的 span。",
      focus: ["l51-image", "l51-stages", "l51-e-image-stages"],
    },
    {
      title: "两套 token 分账",
      description:
        "推理 token 与答案 token 的位置集合不相交。格子编号只从 REASONING 里解析；写进 CONCLUSION 不算引用。",
      focus: [
        "l51-stages",
        "l51-reason",
        "l51-answer",
        "l51-e-stages-reason",
        "l51-e-stages-answer",
      ],
    },
    {
      title: "引用格对真值",
      description:
        "C(y) 是推理 span 里的 (行,列)。与金标格子 G 有交才给过程分。第 23 课的命中在这里变成可执行奖励。",
      focus: ["l51-reason", "l51-cite", "l51-e-reason-cite"],
    },
    {
      title: "答案对可以单独记账",
      description:
        "r_ans 只看最终数字。关掉必须引用后，答案仍可对，C(y) 为空。这是 Lab 要先预测再揭晓的那一例。",
      focus: ["l51-answer", "l51-rans", "l51-e-answer-rans"],
    },
    {
      title: "无引用则过程奖励为 0",
      description:
        "r_proc = 1[C(y) ∩ G ≠ ∅]。空引用或只点到干扰格都是 0。第 17 课奖答案、第 38 课奖接触，本课奖证据。",
      focus: ["l51-cite", "l51-rans", "l51-rproc", "l51-e-cite-rproc", "l51-e-rans-rproc"],
    },
  ],
  facts: [
    "LLaVA-CoT 在 Llama-3.2-11B-Vision-Instruct 上用约 100k 结构化样本做全参微调；摘要写加上测试时缩放后平均比基座高 9.4 个百分点。",
    "其 Table 2 给出六项基准均分：基座 49.8、直接训原问答 54.3、去掉阶段标签 55.7、完整结构化 57.6。",
    "Vision-R1-7B 在 MathVista 上 73.5%，比公开榜上的 OpenAI o1 73.9% 低 0.4 个百分点；硬格式结果奖励只在格式与最终答案同时对时给 1。",
    "Vision-R1-cold 约 200K；作者统计 Wait 出现 585719 次，同表 LLaVA-CoT-100k 为 2300 次。",
    "本课过程奖励在引用格为空时必须为 0，与论文里的结果奖励不是同一列。",
  ],
};
