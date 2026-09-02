import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson36Diagram: LessonDiagram = {
  lessonId: "36",
  title: "厨房指令拆成两套不重叠的动作词表",
  summary:
    "一句「去厨房拿杯子」先按身体分流：手臂走 7 维分箱，底盘走 (v,ω) 或路点索引。路点依赖地图；丢地图后索引非法，速度仍合法但会撞墙或转圈。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l36-instr",
      label: ["语言指令"],
      meta: "去厨房拿杯子",
      kind: "input",
      x: 92,
      y: 180,
      width: 148,
    },
    {
      id: "l36-split",
      label: ["身体分流"],
      meta: "模式 / 子目标",
      kind: "decision",
      x: 278,
      y: 180,
      width: 148,
    },
    {
      id: "l36-arm",
      label: ["手臂词表"],
      meta: "7 维 × B bin",
      kind: "state",
      x: 478,
      y: 72,
      width: 150,
    },
    {
      id: "l36-base",
      label: ["底盘速度"],
      meta: "(v, ω) 另切箱",
      kind: "state",
      x: 478,
      y: 180,
      width: 150,
    },
    {
      id: "l36-wp",
      label: ["路点索引"],
      meta: "i ∈ [0, N)",
      kind: "state",
      x: 478,
      y: 288,
      width: 150,
    },
    {
      id: "l36-map",
      label: ["地图门"],
      meta: "N=0 则非法",
      kind: "decision",
      x: 678,
      y: 288,
      width: 140,
    },
    {
      id: "l36-out",
      label: ["电机命令"],
      meta: "臂 / 底盘 / 停",
      kind: "output",
      x: 848,
      y: 180,
      width: 140,
    },
  ],
  edges: [
    {
      id: "l36-e-instr-split",
      from: "l36-instr",
      to: "l36-split",
      label: "子目标",
      labelAt: { x: 186, y: 158 },
    },
    {
      id: "l36-e-split-arm",
      from: "l36-split",
      to: "l36-arm",
      label: "抓杯子",
      via: [
        { x: 278, y: 72 },
      ],
      labelAt: { x: 338, y: 86 },
    },
    {
      id: "l36-e-split-base",
      from: "l36-split",
      to: "l36-base",
      label: "走过去",
      labelAt: { x: 360, y: 158 },
    },
    {
      id: "l36-e-split-wp",
      from: "l36-split",
      to: "l36-wp",
      label: "拓扑节点",
      via: [
        { x: 278, y: 288 },
      ],
      labelAt: { x: 338, y: 268 },
    },
    {
      id: "l36-e-wp-map",
      from: "l36-wp",
      to: "l36-map",
      label: "查图",
      labelAt: { x: 576, y: 266 },
    },
    {
      id: "l36-e-arm-out",
      from: "l36-arm",
      to: "l36-out",
      label: "7-DoF",
      via: [
        { x: 848, y: 72 },
      ],
      labelAt: { x: 720, y: 86 },
    },
    {
      id: "l36-e-base-out",
      from: "l36-base",
      to: "l36-out",
      label: "合法 (v,ω)",
      labelAt: { x: 680, y: 158 },
    },
    {
      id: "l36-e-map-out",
      from: "l36-map",
      to: "l36-out",
      label: "非法索引 / 停",
      via: [
        { x: 848, y: 288 },
      ],
      labelAt: { x: 790, y: 266 },
    },
  ],
  steps: [
    {
      title: "读入跨身体指令",
      description:
        "「去厨房把杯子拿来」同时需要底盘位移和末端抓取。第 24 课的 7 维分箱只管后半段。",
      focus: ["l36-instr", "l36-e-instr-split"],
    },
    {
      title: "按身体分流",
      description:
        "RT-1 用三模式开关分时控臂、控底盘或结束回合。教学缩小版把两套词表显式切开。",
      focus: ["l36-split", "l36-e-split-arm", "l36-e-split-base", "l36-e-split-wp"],
    },
    {
      title: "手臂与底盘不共享箱边界",
      description:
        "末端位移常用无量纲 [-1,1]，线速度与角速度的物理区间不同。同一套 bin 边会把停车速度编进中位箱子。",
      focus: ["l36-arm", "l36-base", "l36-e-arm-out", "l36-e-base-out"],
    },
    {
      title: "路点索引查拓扑图",
      description:
        "LM-Nav 与 Mobility VLA 的可执行对象是图上的节点。索引合法当且仅当 0 ≤ i < N。",
      focus: ["l36-wp", "l36-map", "l36-e-wp-map"],
    },
    {
      title: "丢掉地图后失败模式分叉",
      description:
        "N=0 时路点策略仍吐出旧节点号，解码为非法索引。速度策略 token 仍合法，电机会撞墙或转圈。",
      focus: ["l36-map", "l36-out", "l36-e-map-out"],
    },
  ],
  facts: [
    "RT-1 把臂的 7 维、底盘 3 维和 terminate 各自均匀切成 256 个 bin，并用三模式开关分时控臂或控底盘，35M 参数模型以 3 Hz 出动作。",
    "底盘 (v,ω) 的物理区间与手臂末端位移不同，拼接词表时偏移必须不重叠；共用手臂 [-1,1] 箱边会把 v=0 编进中位格子。",
    "Mobility VLA 在 836 m² 办公环境里，让长上下文 VLM 直接出路点动作的端到端成功率为 0%；接上拓扑图后仿真端到端成功率为 90%。",
    "NaVid 在连续 VLN 中只输出 FORWARD / TURN-LEFT / TURN-RIGHT / STOP 及距离或角度，不依赖预先建好的节点表；其路点变体被原文消融判定为极难学习。",
  ],
};
