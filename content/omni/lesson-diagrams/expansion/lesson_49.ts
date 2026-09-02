import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson49Diagram: LessonDiagram = {
  lessonId: "49",
  title: "视频理解 CE 与生成帧差分账",
  summary:
    "同一段短视频打成联合序列。理解交叉熵只写在答案 token，生成帧差只写在未来格子。两张 mask 不相交。杯子可以在理解答对的同时从下一帧消失。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l49-clip",
      label: ["短视频"],
      meta: "5 帧可见 + 下一帧",
      kind: "input",
      x: 88,
      y: 180,
    },
    {
      id: "l49-seq",
      label: ["联合序列"],
      meta: "历史 / 问句 / 答案 / 未来",
      kind: "transform",
      x: 268,
      y: 180,
      width: 150,
    },
    {
      id: "l49-ce",
      label: ["理解 CE"],
      meta: "只计答案 token",
      kind: "state",
      x: 468,
      y: 78,
      width: 150,
    },
    {
      id: "l49-l2",
      label: ["生成帧差"],
      meta: "只计未来格子",
      kind: "state",
      x: 468,
      y: 282,
      width: 150,
    },
    {
      id: "l49-mask",
      label: ["有效位置"],
      meta: "M_und ∩ M_gen = ∅",
      kind: "decision",
      x: 668,
      y: 180,
    },
    {
      id: "l49-probe",
      label: ["杯子探针"],
      meta: "占用 1 对 0",
      kind: "output",
      x: 858,
      y: 78,
      width: 140,
    },
    {
      id: "l49-books",
      label: ["两本账"],
      meta: "Video-MME ≠ VBench",
      kind: "output",
      x: 858,
      y: 282,
      width: 140,
    },
  ],
  edges: [
    {
      id: "l49-e-clip-seq",
      from: "l49-clip",
      to: "l49-seq",
      label: "打包",
      labelAt: { x: 178, y: 152 },
    },
    {
      id: "l49-e-seq-ce",
      from: "l49-seq",
      to: "l49-ce",
      label: "问杯子还在",
      via: [{ x: 268, y: 78 }],
      labelAt: { x: 348, y: 52 },
    },
    {
      id: "l49-e-seq-l2",
      from: "l49-seq",
      to: "l49-l2",
      label: "画 t=5",
      via: [{ x: 268, y: 282 }],
      labelAt: { x: 348, y: 312 },
    },
    {
      id: "l49-e-ce-mask",
      from: "l49-ce",
      to: "l49-mask",
      label: "下标 164",
      labelAt: { x: 568, y: 118 },
    },
    {
      id: "l49-e-l2-mask",
      from: "l49-l2",
      to: "l49-mask",
      label: "165–196",
      labelAt: { x: 568, y: 248 },
    },
    {
      id: "l49-e-mask-probe",
      from: "l49-mask",
      to: "l49-probe",
      label: "占用探针",
      via: [{ x: 668, y: 78 }],
      labelAt: { x: 760, y: 52 },
    },
    {
      id: "l49-e-mask-books",
      from: "l49-mask",
      to: "l49-books",
      label: "禁止兑账",
      via: [{ x: 668, y: 282 }],
      labelAt: { x: 760, y: 312 },
    },
    {
      id: "l49-e-ce-probe",
      from: "l49-ce",
      to: "l49-probe",
      label: "答还在",
      labelAt: { x: 668, y: 40 },
    },
  ],
  steps: [
    {
      title: "同一段视频打成联合序列",
      description:
        "5 帧历史、4 个问句 token、1 个答案 token、32 个未来格子。理解与生成读同一段媒体，写入不同切片。",
      focus: ["l49-clip", "l49-seq", "l49-e-clip-seq"],
    },
    {
      title: "理解 CE 只写在答案 token",
      description:
        "问“杯子还在不在”。prompt 与历史像素的 label 为跳过。CPU 夹具里有效位置是下标 164。",
      focus: ["l49-ce", "l49-e-seq-ce"],
    },
    {
      title: "生成帧差只写在未来格子",
      description:
        "v-prediction 或 flow 或教学用 L2，有效位置是未来帧。历史像素与答案 token 不进这笔损失。",
      focus: ["l49-l2", "l49-e-seq-l2"],
    },
    {
      title: "两张 mask 不相交，杯子仍可消失",
      description:
        "交为 0。理解 p(还在)=0.917，生成占用可以是 0。均值填充的 L2 可以低于抄上一帧。",
      focus: ["l49-mask", "l49-probe", "l49-e-ce-mask", "l49-e-l2-mask"],
    },
    {
      title: "Video-MME、相机、物体永久不是同一张表",
      description:
        "Video-MME 进理解 C2。VBench 动态度、相机类型命中、物体消失率进生成账。禁止兑成一个视频分。",
      focus: ["l49-books", "l49-e-mask-books"],
    },
  ],
  facts: [
    "CogVideoX（arXiv:2408.06072）生成最长 10 秒、16 fps、768×1360；3D causal VAE 空间 8×8、时间 4 倍压缩；专家 AdaLN 分模态调制；5B 对人评 Kling 总分 2.74 对 2.17。",
    "HunyuanVideo（arXiv:2412.03603）13B，Causal 3D VAE $c_t=4,c_s=8,C=16$，双流 20 块加单流 40 块。60 名评估员、1533 条 prompt：Overall 41.3%，Motion 66.5%，高于 Gen-3 alpha 的 27.4%。",
    "Video-MME：900 视频、2700 题。Gemini 1.5 Pro 无字幕 75.0%。该数字是第 47 课 C2 理解准确率，不能填进 VBench 或物体永久性。",
    "CPU 夹具：序列长 197，CE 有效位置 1 个（下标 164），帧差有效位置 32 个（165–196），交为 0。历史占用 1.0，遗忘生成占用 0.0，caption CE 0.0868。",
  ],
};
