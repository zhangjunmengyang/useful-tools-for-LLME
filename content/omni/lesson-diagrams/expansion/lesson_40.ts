import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson40Diagram: LessonDiagram = {
  lessonId: "40",
  title: "力超限切断动作块",
  summary:
    "执行 chunk 第 i 步后测量接触力。若范数大于 F_max，丢掉 i+1 以后的剩余步，进入 SAFE_HOLD 保持当前姿态。语音 PAUSE 可丢未播 PCM；接触造成的位移不能 undo。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l40-chunk",
      label: ["动作 chunk"],
      meta: "H 步",
      kind: "input",
      x: 88,
      y: 88,
      width: 128,
    },
    {
      id: "l40-exec",
      label: ["执行第 i 步"],
      meta: "末端推进",
      kind: "transform",
      x: 278,
      y: 88,
      width: 132,
    },
    {
      id: "l40-force",
      label: ["接触力 F_i"],
      meta: "||F||",
      kind: "state",
      x: 478,
      y: 88,
      width: 132,
    },
    {
      id: "l40-gate",
      label: ["力门限"],
      meta: "||F|| ? F_max",
      kind: "decision",
      x: 698,
      y: 88,
      width: 148,
    },
    {
      id: "l40-hold",
      label: ["SAFE_HOLD"],
      meta: "剩余步 = 0",
      kind: "state",
      x: 478,
      y: 268,
      width: 148,
    },
    {
      id: "l40-human",
      label: ["人类接管"],
      meta: "姿态冻结",
      kind: "output",
      x: 718,
      y: 268,
      width: 140,
    },
    {
      id: "l40-pause",
      label: ["语音 PAUSE"],
      meta: "可丢 PCM",
      kind: "transform",
      x: 88,
      y: 268,
      width: 140,
    },
  ],
  edges: [
    {
      id: "l40-e-chunk-exec",
      from: "l40-chunk",
      to: "l40-exec",
      label: "按控制周期下发",
      labelAt: { x: 178, y: 54 },
    },
    {
      id: "l40-e-exec-force",
      from: "l40-exec",
      to: "l40-force",
      label: "测接触合力",
      labelAt: { x: 368, y: 54 },
    },
    {
      id: "l40-e-force-gate",
      from: "l40-force",
      to: "l40-gate",
      label: "与 F_max 比较",
      labelAt: { x: 588, y: 54 },
    },
    {
      id: "l40-e-gate-exec",
      from: "l40-gate",
      to: "l40-exec",
      label: "未超限继续",
      via: [
        { x: 698, y: 168 },
        { x: 278, y: 168 },
      ],
      labelAt: { x: 488, y: 148 },
    },
    {
      id: "l40-e-gate-hold",
      from: "l40-gate",
      to: "l40-hold",
      label: "超限则丢掉剩余步",
      labelAt: { x: 640, y: 200 },
    },
    {
      id: "l40-e-hold-human",
      from: "l40-hold",
      to: "l40-human",
      label: "接触未解除不重规划",
      labelAt: { x: 608, y: 318 },
    },
    {
      id: "l40-e-pause-hold",
      from: "l40-pause",
      to: "l40-hold",
      label: "PCM 可丢，力不能 undo",
      labelAt: { x: 278, y: 318 },
    },
  ],
  steps: [
    {
      title: "动作块进入执行",
      description:
        "一次观察预测长度为 H 的动作块。调度器按控制周期依次下发第 i 步，而不是把整块当成不可打断的开环。",
      focus: ["l40-chunk", "l40-exec", "l40-e-chunk-exec"],
    },
    {
      title: "每步测量接触力",
      description:
        "执行第 i 步之后读取力/力矩传感器，得到接触合力 F_i。没有力通道时，不得把视觉失败检测器冒充成力门限。",
      focus: ["l40-exec", "l40-force", "l40-e-exec-force"],
    },
    {
      title: "用范数判定超限",
      description:
        "若 ||F_i|| > F_max，丢掉 i+1 以后的剩余步。等于阈值不切断，严格大于才切断。",
      focus: ["l40-force", "l40-gate", "l40-e-force-gate"],
    },
    {
      title: "SAFE_HOLD 保持姿态",
      description:
        "切断后剩余步计数为 0，末端停在当前姿态，额外滴答不得推进。杯子已经发生的位移不能退回超限前。",
      focus: ["l40-gate", "l40-hold", "l40-e-gate-hold"],
    },
    {
      title: "人类接管，禁止带接触重规划",
      description:
        "SAFE_HOLD 之后把权限交给操作者。接触未解除时不得把旧 chunk 或新 REPLAN 接到仍顶着物体的末端上。",
      focus: ["l40-hold", "l40-human", "l40-e-hold-human"],
    },
    {
      title: "对照语音 PAUSE",
      description:
        "第 07 课 PAUSE 冻结 pending PCM，REPLAN 才丢掉未播音频。音频缓冲可丢；热水、碎杯、已经发生的接触力不能 undo。",
      focus: ["l40-pause", "l40-hold", "l40-e-pause-hold"],
    },
  ],
  facts: [
    "在 chunk 步 i 若 ||F_i|| > F_max，丢弃 i+1 以后的剩余步，执行保持姿态。",
    "SAFE_HOLD 期间剩余步为 0，额外控制滴答不得推进末端。",
    "音频 PAUSE 可丢未播放 PCM；接触造成的物体位移不能回到超限前。",
    "SafeVLA 用 CMDP 约束累计安全成本，相对 FLaRe 降低 83.58%，成功率 +3.85%。",
    "SafeVLA-Bench 将接触力主档实例化为 200 N 的仿真代理；LIBERO 高成功率仍有 13% 到 15% 不安全回合。",
  ],
};
