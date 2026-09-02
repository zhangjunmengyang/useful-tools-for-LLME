import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson21Diagram: LessonDiagram = {
  lessonId: "21",
  title: "对比学习把图和字放进同一张相似度表",
  summary:
    "双塔各自编码后做余弦相似度，温度缩放 logits，再按 InfoNCE 或 pairwise sigmoid 拉近正对、推开同 batch 负对。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l21-image",
      label: ["图像 batch"],
      meta: "N 张图",
      kind: "input",
      x: 88,
      y: 92,
    },
    {
      id: "l21-text",
      label: ["文字 batch"],
      meta: "N 条 caption",
      kind: "input",
      x: 88,
      y: 268,
    },
    {
      id: "l21-vit",
      label: ["图像编码器"],
      meta: "ViT / ResNet",
      kind: "transform",
      x: 278,
      y: 92,
    },
    {
      id: "l21-txtenc",
      label: ["文字编码器"],
      meta: "Transformer",
      kind: "transform",
      x: 278,
      y: 268,
    },
    {
      id: "l21-sim",
      label: ["相似度矩阵"],
      meta: "N×N 余弦",
      kind: "state",
      x: 490,
      y: 180,
    },
    {
      id: "l21-tau",
      label: ["温度 τ"],
      meta: "logits ÷ τ",
      kind: "decision",
      x: 690,
      y: 92,
    },
    {
      id: "l21-loss",
      label: ["InfoNCE", "或 sigmoid"],
      meta: "正对 / 负对",
      kind: "transform",
      x: 690,
      y: 268,
    },
    {
      id: "l21-space",
      label: ["共享空间"],
      meta: "检索 / 零样本",
      kind: "output",
      x: 868,
      y: 180,
    },
  ],
  edges: [
    {
      id: "l21-e-image-vit",
      from: "l21-image",
      to: "l21-vit",
      label: "像素",
      labelAt: { x: 183, y: 64 },
    },
    {
      id: "l21-e-text-enc",
      from: "l21-text",
      to: "l21-txtenc",
      label: "token",
      labelAt: { x: 183, y: 300 },
    },
    {
      id: "l21-e-vit-sim",
      from: "l21-vit",
      to: "l21-sim",
      label: "L2 向量 x",
      via: [{ x: 390, y: 92 }, { x: 390, y: 180 }],
      labelAt: { x: 348, y: 128 },
    },
    {
      id: "l21-e-txt-sim",
      from: "l21-txtenc",
      to: "l21-sim",
      label: "L2 向量 y",
      via: [{ x: 390, y: 268 }, { x: 390, y: 180 }],
      labelAt: { x: 348, y: 232 },
    },
    {
      id: "l21-e-sim-tau",
      from: "l21-sim",
      to: "l21-tau",
      label: "s_ij",
      labelAt: { x: 590, y: 64 },
    },
    {
      id: "l21-e-tau-loss",
      from: "l21-tau",
      to: "l21-loss",
      label: "缩放后分类",
    },
    {
      id: "l21-e-sim-loss",
      from: "l21-sim",
      to: "l21-loss",
      label: "正对在对角",
      labelAt: { x: 560, y: 300 },
    },
    {
      id: "l21-e-loss-space",
      from: "l21-loss",
      to: "l21-space",
      label: "拉近 / 推开",
      labelAt: { x: 790, y: 300 },
    },
  ],
  steps: [
    {
      title: "双塔分别编码",
      description:
        "图像塔和文字塔不共享中间层。各自把样本映到同一维数，再做 L2 归一化，后续只比较余弦。",
      focus: [
        "l21-image",
        "l21-text",
        "l21-vit",
        "l21-txtenc",
        "l21-e-image-vit",
        "l21-e-text-enc",
      ],
    },
    {
      title: "拼出 N×N 相似度",
      description:
        "对角是本 batch 声称的正对；其余 N²−N 格是同 batch 负对。没有额外负样本队列。",
      focus: [
        "l21-vit",
        "l21-txtenc",
        "l21-sim",
        "l21-e-vit-sim",
        "l21-e-txt-sim",
      ],
    },
    {
      title: "温度缩放 logits",
      description:
        "CLIP 把 τ 初始化为 0.07，并把 logit 乘数上限裁到 100，相当于 τ 不低于 0.01。τ 过小，softmax 接近 one-hot；τ 过大，正负对分不开。",
      focus: ["l21-sim", "l21-tau", "l21-e-sim-tau"],
    },
    {
      title: "选损失：行内 softmax 或逐格 sigmoid",
      description:
        "InfoNCE 对每一行、每一列做 softmax，依赖整段 batch。SigLIP 把每一格当成独立的是否配对二分类，不必看全局归一化。",
      focus: [
        "l21-tau",
        "l21-loss",
        "l21-sim",
        "l21-e-tau-loss",
        "l21-e-sim-loss",
      ],
    },
    {
      title: "读出共享空间",
      description:
        "训练完成后，用文字编码器把类名或检索词变成权重，和图像向量比余弦，即可做零样本分类和跨模态检索。",
      focus: ["l21-loss", "l21-space", "l21-e-loss-space"],
    },
  ],
  facts: [
    "CLIP 在 4 亿图文对上从零训练，最佳模型 ImageNet 零样本 76.2%，batch 为 32768。",
    "ALIGN 用 18 亿噪声 alt-text，ImageNet 零样本 76.4%；温度作为可学习参数，约收敛到 1/64。",
    "SigLIP 的 sigmoid 损失不对 batch 做 softmax；论文观察到 32k 的 batch 已经够用。",
    "LiT 锁住预训练图像塔、训练文字塔，ViT-g/14 上 ImageNet 零样本 85.2%。",
  ],
};
