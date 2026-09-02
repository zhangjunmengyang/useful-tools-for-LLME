import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson37Diagram: LessonDiagram = {
  lessonId: "37",
  title: "固定预算下维数升高丢掉开环或箱宽",
  summary:
    "示教含低频躯干与高频手指。token 预算 C=Hd 固定时，d 升高则分块变短或每维箱宽变粗，手指高频先坏。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l37-demo",
      label: ["混合频率示教"],
      meta: "T=64 · 20 Hz",
      kind: "input",
      x: 96,
      y: 180,
    },
    {
      id: "l37-budget",
      label: ["固定预算 C"],
      meta: "168 token / 336 bit",
      kind: "input",
      x: 286,
      y: 78,
    },
    {
      id: "l37-dim",
      label: ["动作维数 d"],
      meta: "7 到 24",
      kind: "state",
      x: 286,
      y: 282,
    },
    {
      id: "l37-h",
      label: ["分块 H=C/d"],
      meta: "开环 H/f",
      kind: "transform",
      x: 498,
      y: 78,
    },
    {
      id: "l37-width",
      label: ["每维箱宽"],
      meta: "Δ=2/2^⌊b⌋",
      kind: "transform",
      x: 498,
      y: 282,
    },
    {
      id: "l37-band",
      label: ["频带分账"],
      meta: "手指 vs 躯干",
      kind: "decision",
      x: 698,
      y: 180,
    },
    {
      id: "l37-error",
      label: ["高频重建误差"],
      meta: "d 升则手指先坏",
      kind: "output",
      x: 868,
      y: 180,
      width: 156,
    },
  ],
  edges: [
    {
      id: "l37-e-demo-dim",
      from: "l37-demo",
      to: "l37-dim",
      label: "按维切开",
      via: [{ x: 96, y: 282 }],
      labelAt: { x: 148, y: 254 },
    },
    {
      id: "l37-e-budget-h",
      from: "l37-budget",
      to: "l37-h",
      label: "token 守恒",
      labelAt: { x: 392, y: 52 },
    },
    {
      id: "l37-e-budget-width",
      from: "l37-budget",
      to: "l37-width",
      label: "比特守恒",
      via: [{ x: 286, y: 180 }, { x: 498, y: 180 }],
      labelAt: { x: 360, y: 168 },
    },
    {
      id: "l37-e-dim-h",
      from: "l37-dim",
      to: "l37-h",
      via: [{ x: 392, y: 282 }, { x: 392, y: 78 }],
    },
    {
      id: "l37-e-dim-width",
      from: "l37-dim",
      to: "l37-width",
      label: "d 升高",
      labelAt: { x: 392, y: 310 },
    },
    {
      id: "l37-e-h-band",
      from: "l37-h",
      to: "l37-band",
      label: "时间样本变稀",
      labelAt: { x: 598, y: 96 },
    },
    {
      id: "l37-e-width-band",
      from: "l37-width",
      to: "l37-band",
      label: "箱变粗",
      labelAt: { x: 598, y: 268 },
    },
    {
      id: "l37-e-band-error",
      from: "l37-band",
      to: "l37-error",
      label: "按频带记账",
      labelAt: { x: 786, y: 214 },
    },
  ],
  steps: [
    {
      title: "示教同时含慢躯干和快手指",
      description:
        "64 步、20 Hz 的固定示教里，第 0 维是 2.5 Hz 手指，第 1 维是慢速躯干。平均 L2 会被躯干和填充维稀释。",
      focus: ["l37-demo", "l37-dim", "l37-e-demo-dim"],
    },
    {
      title: "token 预算守恒则分块变短",
      description:
        "C=Hd=168 时，d=7 得到 H=24，d=24 得到 H=7。开环时长从 1.2 s 缩到 0.35 s。",
      focus: ["l37-budget", "l37-h", "l37-e-budget-h", "l37-e-dim-h"],
    },
    {
      title: "比特预算守恒则每维变粗",
      description:
        "C_bit=336、H=8 时，d=7 每维 6 bit、箱宽 0.03125；d=24 每维 1.75 bit、箱宽 1.0。锚点量化误差贴着半箱宽。",
      focus: ["l37-budget", "l37-width", "l37-e-budget-width", "l37-e-dim-width"],
    },
    {
      title: "同一笔账先打在高频手指上",
      description:
        "H 变短时 2.5 Hz 先低于有效奈奎斯特；箱变粗时小幅摆动掉进同一箱。低频躯干仍能用少量样本重建。",
      focus: ["l37-h", "l37-width", "l37-band", "l37-e-h-band", "l37-e-width-band"],
    },
    {
      title: "对照公开系统的维数，不要编未公开的布局",
      description:
        "OpenVLA 写明 7 维。GR-3 的 ByteMini 是 22 自由度、rollout 用 19 维。Helix 官方博文写 35-DoF、200 Hz。Gemini Robotics 2 博文写 22 自由度五指手，未公布全身动作向量布局。",
      focus: ["l37-band", "l37-error", "l37-e-band-error"],
    },
  ],
  facts: [
    "固定 token 预算 C=Hd 时，d 从 7 升到 24，H 从 24 降到 7，开环 H/f 从 1.2 s 降到 0.35 s（f=20 Hz）。",
    "固定比特预算 336、H=8 时，d=7 每维 6 bit、箱宽 0.03125；d=24 每维 1.75 bit、箱宽 1.0。",
    "GR-3 技术报告：ByteMini 为 22-DoF 双臂移动机器人，策略 rollout 控制 19 个自由度。",
    "Figure 官方博文：Helix 以 200 Hz 输出 35-DoF 上身连续控制，System 2 约 7–9 Hz。",
    "Gemini Robotics 2 官方博文给出 SharpaWave 五指手 22 自由度，并写明多指灵巧仍困难；未公布全身动作向量的逐维布局。",
  ],
};
