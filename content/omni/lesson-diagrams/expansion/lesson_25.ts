import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson25Diagram: LessonDiagram = {
  lessonId: "25",
  title: "四种动作表示的重建与串行代价",
  summary:
    "7 维示教先按分位归一化，再分别走均匀分箱、连续 L1 和 DCT；解码时用分块长度决定开环时长与串行深度。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l25-demo",
      label: ["7 维示教"],
      meta: "T=64 · 20 Hz",
      kind: "input",
      x: 88,
      y: 180,
    },
    {
      id: "l25-norm",
      label: ["分位归一化"],
      meta: "1%-99% 映到 [-1,1]",
      kind: "transform",
      x: 258,
      y: 180,
    },
    {
      id: "l25-bin",
      label: ["均匀分箱"],
      meta: "每维 B 个 bin",
      kind: "transform",
      x: 448,
      y: 70,
    },
    {
      id: "l25-l1",
      label: ["连续 L1"],
      meta: "并行回归头",
      kind: "transform",
      x: 448,
      y: 180,
    },
    {
      id: "l25-dct",
      label: ["DCT 保留系数"],
      meta: "量化后非零项",
      kind: "transform",
      x: 448,
      y: 292,
    },
    {
      id: "l25-decode",
      label: ["串行或并行"],
      meta: "深度 7H 或 1",
      kind: "decision",
      x: 662,
      y: 180,
    },
    {
      id: "l25-pareto",
      label: ["重建 / token", "开环 H/f"],
      meta: "Pareto 三点",
      kind: "output",
      x: 852,
      y: 180,
      width: 168,
    },
  ],
  edges: [
    {
      id: "l25-e-demo-norm",
      from: "l25-demo",
      to: "l25-norm",
      label: "夹到分位区间",
      labelAt: { x: 172, y: 214 },
    },
    {
      id: "l25-e-norm-bin",
      from: "l25-norm",
      to: "l25-bin",
      label: "离散 CE",
      via: [{ x: 330, y: 180 }, { x: 330, y: 70 }],
      labelAt: { x: 292, y: 96 },
    },
    {
      id: "l25-e-norm-l1",
      from: "l25-norm",
      to: "l25-l1",
      label: "连续值",
    },
    {
      id: "l25-e-norm-dct",
      from: "l25-norm",
      to: "l25-dct",
      label: "按维 DCT",
      via: [{ x: 330, y: 180 }, { x: 330, y: 292 }],
      labelAt: { x: 286, y: 268 },
    },
    {
      id: "l25-e-bin-decode",
      from: "l25-bin",
      to: "l25-decode",
      via: [{ x: 560, y: 70 }, { x: 560, y: 180 }],
    },
    {
      id: "l25-e-l1-decode",
      from: "l25-l1",
      to: "l25-decode",
      label: "分块长度 H",
      labelAt: { x: 556, y: 154 },
    },
    {
      id: "l25-e-dct-decode",
      from: "l25-dct",
      to: "l25-decode",
      via: [{ x: 560, y: 292 }, { x: 560, y: 180 }],
    },
    {
      id: "l25-e-decode-pareto",
      from: "l25-decode",
      to: "l25-pareto",
      label: "L2 · 词表 · 深度",
      labelAt: { x: 758, y: 214 },
    },
  ],
  steps: [
    {
      title: "按分位把动作装进固定区间",
      description:
        "OpenVLA 用训练集每维 1% 与 99% 分位划分区间，避免离群点把箱子拉宽。本图的示教已落在 [-1, 1]。",
      focus: ["l25-demo", "l25-norm", "l25-e-demo-norm"],
    },
    {
      title: "均匀分箱把连续值变成编号",
      description:
        "每维切成 B 个等宽箱，编号进入 next-token 交叉熵。量化误差上界是箱宽的一半，B=2 时高频来回会掉进同一箱。",
      focus: ["l25-norm", "l25-bin", "l25-e-norm-bin"],
    },
    {
      title: "连续 L1 一次吐出整段动作",
      description:
        "OpenVLA-OFT 用并行解码加 L1 回归，不再逐步吐离散编号。重建可以贴着原轨迹，串行深度降到 1。",
      focus: ["l25-l1", "l25-e-norm-l1", "l25-decode"],
    },
    {
      title: "DCT 先丢掉高频再计 token",
      description:
        "FAST 对每个动作维做 DCT、缩放取整，再把稀疏系数交给 BPE。保留低频会压掉示教里的快速抖动。",
      focus: ["l25-dct", "l25-e-norm-dct", "l25-decode"],
    },
    {
      title: "用 H/f 和串行深度读 Pareto",
      description:
        "分块长度 H 与控制频率 f 决定开环时长 H/f。自回归 7 维一步要 7 个 token，FAST 论文 Table I 里 1 秒 chunk 的 token 数随频率变化。",
      focus: ["l25-decode", "l25-pareto", "l25-e-decode-pareto"],
    },
  ],
  facts: [
    "OpenVLA 把 7 维动作每维均匀分成 256 箱，箱宽取训练集 1% 与 99% 分位之间的区间。",
    "开环时长等于动作分块长度除以控制频率，写作 H/f；H 加倍则开环时长加倍。",
    "FAST 论文 Table I：1 秒 chunk 上 Shirt Fold 由朴素 700 个 token 压到 53 个，压缩比 13.2。",
    "OpenVLA-OFT 在 LIBERO 四套件上把成功率从 76.5% 升到 97.1%，8 步分块吞吐提高 26 倍。",
  ],
};
