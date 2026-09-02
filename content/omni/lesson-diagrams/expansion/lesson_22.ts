import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson22Diagram: LessonDiagram = {
  lessonId: "22",
  title: "标准 VLM 的三阶段解冻顺序",
  summary:
    "冻结视觉编码器，先把投影 W 训到词嵌入维，再打开 LLM 或 LoRA；过早解冻 ViT 会抬升图文对齐，同时打穿旧文本能力。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l22-image",
      label: ["图像-文本对"],
      meta: "caption / instruction",
      kind: "input",
      x: 96,
      y: 168,
    },
    {
      id: "l22-vit",
      label: ["视觉编码器"],
      meta: "默认冻结 ViT",
      kind: "state",
      x: 278,
      y: 86,
    },
    {
      id: "l22-proj",
      label: ["投影 W"],
      meta: "H = W Z",
      kind: "transform",
      x: 470,
      y: 86,
    },
    {
      id: "l22-schedule",
      label: ["解冻顺序"],
      meta: "只训 W / +LoRA / +ViT",
      kind: "decision",
      x: 278,
      y: 268,
      width: 168,
    },
    {
      id: "l22-llm",
      label: ["语言模型"],
      meta: "全量或 LoRA",
      kind: "transform",
      x: 662,
      y: 168,
    },
    {
      id: "l22-align",
      label: ["图文对齐"],
      meta: "视觉条件生成",
      kind: "output",
      x: 854,
      y: 78,
    },
    {
      id: "l22-text",
      label: ["旧文本能力"],
      meta: "纯文本探针",
      kind: "output",
      x: 854,
      y: 258,
    },
  ],
  edges: [
    {
      id: "l22-e-image-vit",
      from: "l22-image",
      to: "l22-vit",
      label: "patch 特征",
      via: [{ x: 176, y: 168 }, { x: 176, y: 86 }],
      labelAt: { x: 128, y: 122 },
    },
    {
      id: "l22-e-vit-proj",
      from: "l22-vit",
      to: "l22-proj",
      label: "Z_v",
      labelAt: { x: 374, y: 62 },
    },
    {
      id: "l22-e-proj-llm",
      from: "l22-proj",
      to: "l22-llm",
      label: "视觉 token",
      via: [{ x: 566, y: 86 }, { x: 566, y: 168 }],
      labelAt: { x: 516, y: 132 },
    },
    {
      id: "l22-e-image-llm",
      from: "l22-image",
      to: "l22-llm",
      label: "文本条件",
      labelAt: { x: 430, y: 186 },
    },
    {
      id: "l22-e-sched-vit",
      from: "l22-schedule",
      to: "l22-vit",
      label: "冻 / 解冻",
    },
    {
      id: "l22-e-sched-proj",
      from: "l22-schedule",
      to: "l22-proj",
      label: "先开 W",
      via: [{ x: 400, y: 268 }, { x: 470, y: 200 }],
      labelAt: { x: 430, y: 248 },
    },
    {
      id: "l22-e-llm-align",
      from: "l22-llm",
      to: "l22-align",
      label: "next-token",
      labelAt: { x: 756, y: 96 },
    },
    {
      id: "l22-e-llm-text",
      from: "l22-llm",
      to: "l22-text",
      label: "文本回归",
      labelAt: { x: 756, y: 238 },
    },
  ],
  steps: [
    {
      title: "冻视觉，只训投影",
      description:
        "ViT 与 LLM 保持冻结，只更新 W，把视觉特征映到词嵌入维。LLaVA 第一阶段在过滤后的 595K 图文对上做这件事。",
      focus: [
        "l22-image",
        "l22-vit",
        "l22-proj",
        "l22-schedule",
        "l22-e-image-vit",
        "l22-e-vit-proj",
        "l22-e-sched-proj",
      ],
    },
    {
      title: "视觉 token 进入语言空间",
      description:
        "H_v = W Z_v 之后，视觉位置按普通 token 与文本拼接，损失只算助手回答。",
      focus: [
        "l22-proj",
        "l22-llm",
        "l22-e-proj-llm",
        "l22-e-image-llm",
      ],
    },
    {
      title: "再打开 LLM 或 LoRA",
      description:
        "投影稳定后再训语言模型。LLaVA 第二阶段更新投影与 LLM；课程玩具把第二阶段写成 W 加 LoRA，便于数可训练参数。",
      focus: ["l22-schedule", "l22-llm", "l22-align", "l22-e-llm-align"],
    },
    {
      title: "过早解冻 ViT 的分叉",
      description:
        "若第一步就让 ViT 和 LLM 同时更新，对齐分数可以继续升，旧文本探针会掉下去。",
      focus: [
        "l22-schedule",
        "l22-vit",
        "l22-e-sched-vit",
        "l22-text",
        "l22-e-llm-text",
      ],
    },
    {
      title: "用三项指标验收",
      description:
        "报告图文对齐、指令跟随和旧文本能力。只报平均分会把遗忘藏起来。",
      focus: ["l22-align", "l22-text", "l22-llm"],
    },
  ],
  facts: [
    "LLaVA 第一阶段冻结视觉编码器和 LLM，只训练投影，数据是过滤后的 595K 图文对。",
    "LLaVA 第二阶段仍冻结视觉编码器，继续更新投影和语言模型。",
    "BLIP-2 的 Q-Former 约 188M 可训练参数，两侧塔在预训练中保持冻结。",
    "Flamingo 在冻结语言模型上插入 gated xattn-dense，消融显示解冻预训练 LM 会掉分。",
  ],
};
