import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson58Diagram: LessonDiagram = {
  lessonId: "58",
  title: "医学图文从自然配方里拆开",
  summary:
    "PMC 图注先做概念对齐，再做领域指令；报告字段分开 mask；阳性发现必须带框，空图关掉门控仍会无框报肺炎。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l58-pmc",
      label: ["PMC 图注对"],
      meta: "600K 对齐 / 60K 指令",
      kind: "input",
      x: 92,
      y: 180,
      width: 148,
    },
    {
      id: "l58-align",
      label: ["概念对齐"],
      meta: "只训投影 W",
      kind: "transform",
      x: 286,
      y: 86,
      width: 148,
    },
    {
      id: "l58-instruct",
      label: ["领域指令"],
      meta: "GPT-4 多轮对话",
      kind: "transform",
      x: 286,
      y: 274,
      width: 148,
    },
    {
      id: "l58-mask",
      label: ["报告字段 mask"],
      meta: "FINDINGS 进损失",
      kind: "state",
      x: 500,
      y: 86,
      width: 160,
    },
    {
      id: "l58-gate",
      label: ["无框断言门控"],
      meta: "阳性必须带框",
      kind: "decision",
      x: 500,
      y: 274,
      width: 160,
    },
    {
      id: "l58-empty",
      label: ["空图探针"],
      meta: "语言先验肺炎",
      kind: "state",
      x: 714,
      y: 86,
      width: 148,
    },
    {
      id: "l58-eval",
      label: ["协议分数"],
      meta: "U / 开放 recall",
      kind: "output",
      x: 714,
      y: 274,
      width: 148,
    },
  ],
  edges: [
    {
      id: "l58-e-pmc-align",
      from: "l58-pmc",
      to: "l58-align",
      label: "图注扩写",
      via: [
        { x: 168, y: 180 },
        { x: 168, y: 86 },
      ],
      labelAt: { x: 118, y: 128 },
    },
    {
      id: "l58-e-pmc-instruct",
      from: "l58-pmc",
      to: "l58-instruct",
      label: "caption+citance",
      via: [
        { x: 168, y: 180 },
        { x: 168, y: 274 },
      ],
      labelAt: { x: 96, y: 232 },
    },
    {
      id: "l58-e-align-mask",
      from: "l58-align",
      to: "l58-mask",
      label: "助手 token",
      labelAt: { x: 392, y: 62 },
    },
    {
      id: "l58-e-instruct-gate",
      from: "l58-instruct",
      to: "l58-gate",
      label: "对话监督",
      labelAt: { x: 392, y: 298 },
    },
    {
      id: "l58-e-mask-empty",
      from: "l58-mask",
      to: "l58-empty",
      label: "INDICATION 不训",
      labelAt: { x: 604, y: 62 },
    },
    {
      id: "l58-e-gate-eval",
      from: "l58-gate",
      to: "l58-eval",
      label: "U 计数",
      labelAt: { x: 604, y: 298 },
    },
    {
      id: "l58-e-empty-eval",
      from: "l58-empty",
      to: "l58-eval",
      label: "空图仍报阳",
      labelAt: { x: 780, y: 180 },
    },
    {
      id: "l58-e-mask-gate",
      from: "l58-mask",
      to: "l58-gate",
      label: "字段 ≠ caption",
      labelAt: { x: 454, y: 180 },
    },
  ],
  steps: [
    {
      title: "用 PMC 图注替换 CC3M caption",
      description:
        "LLaVA-Med 从 PMC-15M 抽 600K 图文对做概念对齐，数据不再是自然图像短句。",
      focus: ["l58-pmc", "l58-align", "l58-e-pmc-align"],
    },
    {
      title: "盲模型写指令，眼睛仍冻着",
      description:
        "GPT-4 只看 caption 和文内引用写成 60K 对话；第二阶段仍冻视觉编码器，更新投影和语言模型。",
      focus: ["l58-instruct", "l58-e-pmc-instruct", "l58-e-instruct-gate"],
    },
    {
      title: "报告字段不能整段当 caption 算损失",
      description:
        "INDICATION 和 COMPARISON 不是当前图像生成的目标；FINDINGS / IMPRESSION 才进 M_med。",
      focus: ["l58-mask", "l58-e-align-mask", "l58-e-mask-gate"],
    },
    {
      title: "阳性发现必须带框",
      description:
        "无框断言门控关掉之后，空图仍会根据语言先验写出肺炎。U 计数必须抓住这一条。",
      focus: ["l58-gate", "l58-empty", "l58-e-mask-empty", "l58-e-empty-eval"],
    },
    {
      title: "开放 recall 不能代替 U",
      description:
        "LLaVA-Med 对开放题用 recall、对封闭题用准确率。recall 为 1 时，无框肯定仍可存在。",
      focus: ["l58-eval", "l58-e-gate-eval", "l58-empty"],
    },
  ],
  facts: [
    "LLaVA-Med 第一阶段从 PMC-15M 采样 600K 图文对，冻视觉编码器和语言模型，只更新投影（论文第 4 节）。",
    "第二阶段用 60K GPT-4 指令数据，视觉编码器仍冻，继续更新投影和语言模型；60K-IM 聊天相对分为 50.2（表 1）。",
    "八张 A100、batch 128：第一阶段 1 个 epoch 6.8 小时，60K 指令 3 个 epoch 8.0 小时（表 5）。",
    "开放集用生成序列对真值 token 的 recall，封闭集用准确率（第 5.2 节与表 4）。",
    "PMC-15M 含 15,282,336 对图注，比 MIMIC-CXR 大约两个数量级（BiomedCLIP / LLaVA-Med）。",
  ],
};
