import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson33Diagram: LessonDiagram = {
  lessonId: "33",
  title: "同一段未来：像素重建对表征预测",
  summary:
    "可见上下文进入编码器；预测器在表征空间对齐 EMA 目标。像素解码把不可预测纹理算进 L2，接触边被糊掉；表征回归只保留位置与重叠。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l33-clip",
      label: ["视频片段"],
      meta: "8 步 / 遮挡未来",
      kind: "input",
      x: 88,
      y: 180,
    },
    {
      id: "l33-mask",
      label: ["上下文 / 目标"],
      meta: "multi-block 遮挡",
      kind: "transform",
      x: 248,
      y: 180,
    },
    {
      id: "l33-ctx",
      label: ["上下文编码器"],
      meta: "E_θ(x)",
      kind: "transform",
      x: 408,
      y: 88,
    },
    {
      id: "l33-ema",
      label: ["EMA 目标"],
      meta: "sg(Ē(y))",
      kind: "state",
      x: 408,
      y: 272,
    },
    {
      id: "l33-pred",
      label: ["预测器"],
      meta: "P_φ(z, Δy)",
      kind: "transform",
      x: 590,
      y: 180,
    },
    {
      id: "l33-pixel",
      label: ["像素 L2"],
      meta: "纹理进入损失",
      kind: "output",
      x: 790,
      y: 88,
      width: 150,
    },
    {
      id: "l33-latent",
      label: ["表征回归"],
      meta: "接触 / 分离",
      kind: "output",
      x: 790,
      y: 272,
      width: 150,
    },
  ],
  edges: [
    {
      id: "l33-e-clip-mask",
      from: "l33-clip",
      to: "l33-mask",
      label: "切 tubelet",
      labelAt: { x: 168, y: 148 },
    },
    {
      id: "l33-e-mask-ctx",
      from: "l33-mask",
      to: "l33-ctx",
      label: "可见块",
      via: [
        { x: 328, y: 180 },
        { x: 328, y: 88 },
      ],
      labelAt: { x: 286, y: 124 },
    },
    {
      id: "l33-e-mask-ema",
      from: "l33-mask",
      to: "l33-ema",
      label: "目标块",
      via: [
        { x: 328, y: 180 },
        { x: 328, y: 272 },
      ],
      labelAt: { x: 286, y: 236 },
    },
    {
      id: "l33-e-ctx-pred",
      from: "l33-ctx",
      to: "l33-pred",
      label: "条件表征",
      labelAt: { x: 500, y: 64 },
    },
    {
      id: "l33-e-ema-pred",
      from: "l33-ema",
      to: "l33-pred",
      label: "L1 / L2",
      labelAt: { x: 500, y: 296 },
    },
    {
      id: "l33-e-pred-pixel",
      from: "l33-pred",
      to: "l33-pixel",
      label: "若解码到像素",
      via: [
        { x: 690, y: 180 },
        { x: 690, y: 88 },
      ],
      labelAt: { x: 718, y: 124 },
    },
    {
      id: "l33-e-pred-latent",
      from: "l33-pred",
      to: "l33-latent",
      label: "若停在表征",
      via: [
        { x: 690, y: 180 },
        { x: 690, y: 272 },
      ],
      labelAt: { x: 718, y: 236 },
    },
  ],
  steps: [
    {
      title: "同一段视频",
      description:
        "片段被切成时空 patch。本课比较的不是“会不会预测未来”，而是损失写在像素还是写在表征。",
      focus: ["l33-clip", "l33-mask", "l33-e-clip-mask"],
    },
    {
      title: "上下文与目标",
      description:
        "I-JEPA / V-JEPA 把可见块送给上下文编码器，目标块经 EMA 编码器得到表征；目标支路 stop-gradient。",
      focus: [
        "l33-mask",
        "l33-ctx",
        "l33-ema",
        "l33-e-mask-ctx",
        "l33-e-mask-ema",
      ],
    },
    {
      title: "预测器对齐表征",
      description:
        "预测器读上下文表征和位置掩码，回归目标表征。V-JEPA 用 L1，I-JEPA 用 L2。常数输出会被 VICReg 方差项或 EMA 结构拦住。",
      focus: ["l33-ctx", "l33-ema", "l33-pred", "l33-e-ctx-pred", "l33-e-ema-pred"],
    },
    {
      title: "像素路糊掉接触",
      description:
        "若把同一预测解码回像素，不可预测纹理进入 L2，接触边被平均成灰带，接触/分离探针失效。",
      focus: ["l33-pred", "l33-pixel", "l33-e-pred-pixel"],
    },
    {
      title: "表征路保留重叠",
      description:
        "表征只保留物体位置和 overlap。遮挡未来之后，接触序列和分离序列的表征差仍大于像素探针。",
      focus: ["l33-pred", "l33-latent", "l33-e-pred-latent"],
    },
    {
      title: "动作后训练不改编码器",
      description:
        "V-JEPA 2-AC 冻结编码器，另训动作条件预测器。62 小时 Droid 声明只在无标签、图像子目标、两台未见 Franka 这些条件下成立。",
      focus: ["l33-ema", "l33-pred", "l33-latent"],
    },
  ],
  facts: [
    "I-JEPA 在表征空间用 L2 回归目标块；把损失改到像素后，1% ImageNet 线性探测从 66.9 掉到 40.7（ViT-L/16）。",
    "V-JEPA 用 L1 + EMA + stop-gradient；VideoMix2M 上 ViT-L/16 冻结 K400：特征目标 73.7，像素目标 68.6。",
    "V-JEPA 2-AC 用不到 62 小时 Droid 无标签视频做动作条件后训练；Table 2 的抓取数字是 N=10、图像目标、两台实验室外 Franka。",
    "VICReg 用方差铰链和协方差去相关防塌缩，不依赖 EMA；I-JEPA / V-JEPA 正文用的是 EMA 目标编码器，不是 VICReg 损失。",
  ],
};
