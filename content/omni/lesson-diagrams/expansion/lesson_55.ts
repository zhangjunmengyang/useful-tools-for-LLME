import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson55Diagram: LessonDiagram = {
  lessonId: "55",
  title: "同一序列上按模态量量化损伤",
  summary:
    "同一隐藏向量进入三个头。8/4 bit 按行量化之后，文本看 top-1，视觉看重建 L2，动作看 bin 是否越界。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l55-seq",
      label: ["同一序列"],
      meta: "h 共享",
      kind: "input",
      x: 88,
      y: 180,
    },
    {
      id: "l55-q",
      label: ["8 / 4 bit"],
      meta: "按行 absmax",
      kind: "transform",
      x: 250,
      y: 180,
    },
    {
      id: "l55-text",
      label: ["文本头"],
      meta: "CE · top-1",
      kind: "state",
      x: 430,
      y: 68,
    },
    {
      id: "l55-vis",
      label: ["视觉头"],
      meta: "重建 L2",
      kind: "state",
      x: 430,
      y: 180,
    },
    {
      id: "l55-act",
      label: ["动作头"],
      meta: "7 维 × 8 箱",
      kind: "state",
      x: 430,
      y: 292,
    },
    {
      id: "l55-gate",
      label: ["跳类判定"],
      meta: "边界先于 CE",
      kind: "decision",
      x: 640,
      y: 180,
    },
    {
      id: "l55-out",
      label: ["分模态损伤"],
      meta: "禁止合成一个分",
      kind: "output",
      x: 830,
      y: 180,
      width: 150,
    },
  ],
  edges: [
    {
      id: "l55-e-seq-q",
      from: "l55-seq",
      to: "l55-q",
      label: "同一 h",
      labelAt: { x: 168, y: 152 },
    },
    {
      id: "l55-e-q-text",
      from: "l55-q",
      to: "l55-text",
      label: "W_text",
      via: [
        { x: 318, y: 180 },
        { x: 318, y: 68 },
      ],
      labelAt: { x: 286, y: 108 },
    },
    {
      id: "l55-e-q-vis",
      from: "l55-q",
      to: "l55-vis",
      label: "W_vis",
      labelAt: { x: 338, y: 156 },
    },
    {
      id: "l55-e-q-act",
      from: "l55-q",
      to: "l55-act",
      label: "W_act",
      via: [
        { x: 318, y: 180 },
        { x: 318, y: 292 },
      ],
      labelAt: { x: 278, y: 252 },
    },
    {
      id: "l55-e-text-gate",
      from: "l55-text",
      to: "l55-gate",
      label: "top-1",
      via: [{ x: 640, y: 68 }],
      labelAt: { x: 548, y: 52 },
    },
    {
      id: "l55-e-vis-gate",
      from: "l55-vis",
      to: "l55-gate",
      label: "L2",
      labelAt: { x: 536, y: 156 },
    },
    {
      id: "l55-e-act-gate",
      from: "l55-act",
      to: "l55-gate",
      label: "bin 是否变",
      via: [{ x: 640, y: 292 }],
      labelAt: { x: 548, y: 308 },
    },
    {
      id: "l55-e-gate-out",
      from: "l55-gate",
      to: "l55-out",
      label: "分列记账",
      labelAt: { x: 736, y: 152 },
    },
  ],
  steps: [
    {
      title: "同一隐藏向量进三个头",
      description:
        "文本、视觉、动作不是三条无关请求。端侧量化改的是同一套权重，损伤必须在同一条序列上比。",
      focus: ["l55-seq", "l55-q", "l55-e-seq-q"],
    },
    {
      title: "按行做 8 bit 或 4 bit absmax",
      description:
        "对称网格、半格朝正无穷舍入。本课不测 CUDA graph，也不切 FSDP。量化发生在推理权重上。",
      focus: ["l55-q", "l55-e-q-text", "l55-e-q-vis", "l55-e-q-act"],
    },
    {
      title: "文本看 top-1，不把 CE 当跳类",
      description:
        "夹具里 4 bit 的 CE 甚至略降，top-1 仍是标签 0。margin 约 2.2，远大于量化噪声。",
      focus: ["l55-text", "l55-gate", "l55-e-text-gate"],
    },
    {
      title: "视觉 L2 上升不等于粗类跳了",
      description:
        "8 bit 重建 L2 约 0.002，4 bit 约 0.049。最大分量仍指向同一视觉槽。",
      focus: ["l55-vis", "l55-gate", "l55-e-vis-gate"],
    },
    {
      title: "动作 bin 在 4 bit 越过边界",
      description:
        "pitch 全精度落在 −0.5，箱宽 0.25，箱号 2。4 bit 后到 −0.514，箱号 1。8 bit 仍在箱 2。",
      focus: ["l55-act", "l55-gate", "l55-e-act-gate"],
    },
    {
      title: "三列损伤禁止合成一个分数",
      description:
        "BitVLA Table II 里 OpenVLA 的 INT4 比 OFT 掉得更明显。离散箱和连续 L1 不是同一把尺子。",
      focus: ["l55-gate", "l55-out", "l55-e-gate-out"],
    },
  ],
  facts: [
    "GPTQ 把 OPT-175B 的 WikiText2 4 bit 困惑度从 RTN 的 10.54 收到 8.37，全精度是 8.34（Table 5）。",
    "LLM.int8() 在约 6.7B 起出现系统性离群通道；99.9% 的乘仍走 INT8，离群维走 FP16。",
    "AWQ 在 OpenFlamingo-9B、INT4-g128、32-shot COCO 上把 CIDEr 降幅从 RTN 的 4.57 收到 1.17（Table 6）。",
    "BitVLA Table II：OpenVLA INT4 平均成功率 76.5% 降到 72.7%；OpenVLA-OFT INT4 从 97.1% 到 96.9%。",
    "QVLA 摘要：OpenVLA-OFT 量化版占原 VRAM 的 29.2%，保持原性能的 98.9%，相对 SmoothQuant 高 22.6%。",
  ],
};
