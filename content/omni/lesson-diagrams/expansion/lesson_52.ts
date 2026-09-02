import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson52Diagram: LessonDiagram = {
  lessonId: "52",
  title: "看见图之后先校验工具调用，再执行",
  summary:
    "图像先被读成数字、框和物体名。决策头选择直接答或调用目录里的工具。缺必填参数的调用停在 schema，不得进入计算器、裁剪、深度或搜索。合法观察才能写进最终回答。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l52-image",
      label: ["图像"],
      meta: "发票 / 场景",
      kind: "input",
      x: 88,
      y: 180,
    },
    {
      id: "l52-perceive",
      label: ["感知"],
      meta: "OCR / 框 / 点",
      kind: "transform",
      x: 248,
      y: 180,
    },
    {
      id: "l52-decide",
      label: ["调用决策"],
      meta: "直接答或出调用",
      kind: "decision",
      x: 408,
      y: 180,
    },
    {
      id: "l52-schema",
      label: ["Schema 门"],
      meta: "必填参数 / 类型",
      kind: "state",
      x: 568,
      y: 72,
    },
    {
      id: "l52-exec",
      label: ["工具执行"],
      meta: "算 / 裁 / 深 / 搜",
      kind: "transform",
      x: 728,
      y: 72,
    },
    {
      id: "l52-obs",
      label: ["观察"],
      meta: "合法返回值",
      kind: "state",
      x: 728,
      y: 288,
    },
    {
      id: "l52-answer",
      label: ["最终回答"],
      meta: "写入工具结果",
      kind: "output",
      x: 880,
      y: 180,
      width: 130,
    },
  ],
  edges: [
    {
      id: "l52-e-image-perceive",
      from: "l52-image",
      to: "l52-perceive",
      label: "像素",
      labelAt: { x: 168, y: 228 },
    },
    {
      id: "l52-e-perceive-decide",
      from: "l52-perceive",
      to: "l52-decide",
      label: "候选参数",
      labelAt: { x: 328, y: 156 },
    },
    {
      id: "l52-e-decide-answer",
      from: "l52-decide",
      to: "l52-answer",
      label: "直接答可错进位",
      via: [
        { x: 408, y: 248 },
        { x: 880, y: 248 },
      ],
      labelAt: { x: 620, y: 232 },
    },
    {
      id: "l52-e-decide-schema",
      from: "l52-decide",
      to: "l52-schema",
      label: "名字+参数",
      labelAt: { x: 456, y: 112 },
    },
    {
      id: "l52-e-schema-exec",
      from: "l52-schema",
      to: "l52-exec",
      label: "Valid=1",
      labelAt: { x: 648, y: 48 },
    },
    {
      id: "l52-e-schema-obs",
      from: "l52-schema",
      to: "l52-obs",
      label: "缺参则 ⊥",
      via: [
        { x: 568, y: 288 },
      ],
      labelAt: { x: 620, y: 268 },
    },
    {
      id: "l52-e-exec-obs",
      from: "l52-exec",
      to: "l52-obs",
      label: "返回值",
      labelAt: { x: 780, y: 180 },
    },
    {
      id: "l52-e-obs-answer",
      from: "l52-obs",
      to: "l52-answer",
      label: "写入回答",
      labelAt: { x: 840, y: 248 },
    },
  ],
  steps: [
    {
      title: "先看见，再决定要不要调工具",
      description:
        "发票、场景图先被读成数字、框和物体名。这些只是候选参数，不是小计、距离或外部事实。",
      focus: ["l52-image", "l52-perceive", "l52-e-image-perceive"],
    },
    {
      title: "直接答会把进位算进下一句 token",
      description:
        "决策头可以跳过工具，把 18.90+26.50+15.80 心算成 51.20。看见了数字仍可能漏掉十位进位。",
      focus: ["l52-decide", "l52-answer", "l52-e-decide-answer"],
    },
    {
      title: "调用必须先过 schema 门",
      description:
        "工具名必须在目录里，必填键必须齐，类型必须对。缺 expression 的计算器、缺框的裁剪不得进入执行。",
      focus: ["l52-schema", "l52-e-decide-schema", "l52-e-schema-obs"],
    },
    {
      title: "四类通用工具在执行器里分叉",
      description:
        "计算器做精确算术，裁剪放大局部，深度读点上的相对距离，搜索查图外知识。它们都不是 GUI 点击，也不是检索层选择。",
      focus: ["l52-exec", "l52-e-schema-exec"],
    },
    {
      title: "只有合法观察能写进最终回答",
      description:
        "执行返回值进入观察槽，再由模型写成自然语言。被拒绝的调用留下拒绝记录，不能用下一句文本假装已经算过。",
      focus: ["l52-obs", "l52-answer", "l52-e-exec-obs", "l52-e-obs-answer"],
    },
  ],
  facts: [
    "GPT4Tools 把成功调用拆成 Thought / Action / Arguments，Vicuna-13B 微调后 seen-tool SR=94.1，GPT-3.5 提示为 84.8（原文 Table 2）。",
    "LLaVA-Plus 的工具对话比 LLaVA 多一段 skill_use 与 skill_result；All Tools 在 LLaVA-Bench In-the-Wild 从 57.1 升到 69.5（原文 Table 4）。",
    "ViperGPT 把 compute_depth 做成 ImagePatch 方法，GQA test-dev 零样本 48.1，高于 BLIP-2 的 44.7（原文 Table 2）。",
    "V*Bench 上无搜索的 Vicuna-7B 总体 45.02，接入 V* 搜索后 75.39（原文 Table 2）。",
    "ToRA-Code-34B 在 MATH 上 50.8，超过 GPT-4 CoT 的 42.5，接近 GPT-4 PAL 的 51.8（原文 Table 2）。",
  ],
};
