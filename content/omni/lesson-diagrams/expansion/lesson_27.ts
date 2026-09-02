import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson27Diagram: LessonDiagram = {
  lessonId: "27",
  title: "自回归 VLA 从双视觉走到串行或并行动作",
  summary:
    "图像经 SigLIP 与 DINOv2 拼接后投影进 Llama；动作占用词表尾部 256 个编号。串行 CE 走 7 或 7H 步，并行 L1 一次吐出整段 chunk。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l27-obs",
      label: ["图像+指令"],
      meta: "224px / 语言",
      kind: "input",
      x: 88,
      y: 180,
      width: 152,
    },
    {
      id: "l27-vit",
      label: ["SigLIP+DINOv2"],
      meta: "通道拼接",
      kind: "transform",
      x: 270,
      y: 78,
      width: 160,
    },
    {
      id: "l27-proj",
      label: ["MLP 投影"],
      meta: "patch-as-token",
      kind: "transform",
      x: 270,
      y: 282,
      width: 160,
    },
    {
      id: "l27-llama",
      label: ["Llama 2 解码器"],
      meta: "词表尾 256 格",
      kind: "transform",
      x: 490,
      y: 180,
      width: 156,
    },
    {
      id: "l27-mode",
      label: ["串行或并行"],
      meta: "因果 vs 双向",
      kind: "decision",
      x: 680,
      y: 180,
      width: 150,
    },
    {
      id: "l27-ce",
      label: ["7 步 CE"],
      meta: "延迟 ≈ 7Δt",
      kind: "output",
      x: 870,
      y: 78,
      width: 150,
    },
    {
      id: "l27-l1",
      label: ["并行 L1 头"],
      meta: "1 步吐 H×7",
      kind: "output",
      x: 870,
      y: 282,
      width: 150,
    },
  ],
  edges: [
    {
      id: "l27-e-obs-vit",
      from: "l27-obs",
      to: "l27-vit",
      label: "图像 patch",
      labelAt: { x: 150, y: 92 },
    },
    {
      id: "l27-e-vit-proj",
      from: "l27-vit",
      to: "l27-proj",
      label: "拼接特征",
    },
    {
      id: "l27-e-proj-llama",
      from: "l27-proj",
      to: "l27-llama",
      label: "视觉 token",
      labelAt: { x: 400, y: 268 },
    },
    {
      id: "l27-e-obs-llama",
      from: "l27-obs",
      to: "l27-llama",
      label: "语言 token",
      labelAt: { x: 290, y: 198 },
    },
    {
      id: "l27-e-llama-mode",
      from: "l27-llama",
      to: "l27-mode",
      label: "hidden",
    },
    {
      id: "l27-e-mode-ce",
      from: "l27-mode",
      to: "l27-ce",
      label: "逐维 token",
      labelAt: { x: 790, y: 92 },
    },
    {
      id: "l27-e-mode-l1",
      from: "l27-mode",
      to: "l27-l1",
      label: "连续回归",
      labelAt: { x: 790, y: 268 },
    },
  ],
  steps: [
    {
      title: "双视觉编码",
      description:
        "同一张 224×224 图分别过 SigLIP 与 DINOv2，按通道拼接后再经两层 MLP 投进语言嵌入空间。",
      focus: ["l27-obs", "l27-vit", "l27-proj", "l27-e-obs-vit", "l27-e-vit-proj"],
    },
    {
      title: "覆盖词表尾部",
      description:
        "7 维动作每维 256 个 bin，覆盖 Llama 词表最少用到的末 256 个编号；语言指令走原词表。",
      focus: ["l27-obs", "l27-llama", "l27-e-obs-llama", "l27-e-proj-llama"],
    },
    {
      title: "串行 next-token",
      description:
        "teacher-forced CE 只在动作位置计 loss。推理时一步一维，H=1 要 7 次解码，H>1 要 7H 次。",
      focus: ["l27-mode", "l27-ce", "l27-e-mode-ce"],
    },
    {
      title: "改并行并加 chunk",
      description:
        "空动作槽加双向注意力，一次前向吐出 H×7。LIBERO 上常用 H=8，吞吐大约按 H 放大。",
      focus: ["l27-mode", "l27-l1", "l27-e-mode-l1"],
    },
    {
      title: "CE 与 L1 的边界",
      description:
        "跨 bin 边界时 CE 把邻近连续值当成不同类别；L1 按绝对值线性计罚，不再吃量化台阶。",
      focus: ["l27-ce", "l27-l1", "l27-mode"],
    },
  ],
  facts: [
    "OpenVLA 把每维动作按训练数据 1%–99% 分位均匀切成 256 档，并覆盖 Llama 词表最后 256 个 token。",
    "串行解码步数是 7 或 7H；并行解码一次前向输出整段 H×7。",
    "OpenVLA-OFT 在 LIBERO 四套件（Spatial / Object / Goal / Long）上把微调 OpenVLA 的平均成功率从 76.5% 提到 97.1%，A100 上动作吞吐约 26 倍（论文表 I、表 II）。",
    "斯坦福 SAIL 博文里 MiniVLA 约 1B，Libero-90 上 61.4%，对照同设置 OpenVLA 7B 的 62%。",
  ],
};
