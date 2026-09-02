import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson54Diagram: LessonDiagram = {
  lessonId: "54",
  title: "合成轨迹补分布，不补条数幻觉",
  summary:
    "人类源演示先切成物体中心片段并按相对位姿变换，拒绝失败后再按目标域写入混合物。重复哈希不增加有效样本量；只灌最大域会让 α=1 更糟。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l54-src",
      label: ["人类源演示"],
      meta: "每任务约 10 条",
      kind: "input",
      x: 88,
      y: 180,
    },
    {
      id: "l54-seg",
      label: ["物体中心切段"],
      meta: "已知子任务序列",
      kind: "transform",
      x: 258,
      y: 180,
    },
    {
      id: "l54-warp",
      label: ["相对位姿变换"],
      meta: "T_C' = T_O' T_O^{-1} T_C",
      kind: "transform",
      x: 430,
      y: 78,
    },
    {
      id: "l54-keep",
      label: ["成功才入库"],
      meta: "生成率 ≠ 政策成功率",
      kind: "decision",
      x: 430,
      y: 282,
    },
    {
      id: "l54-hash",
      label: ["唯一哈希账"],
      meta: "源 ID × 复位箱",
      kind: "state",
      x: 620,
      y: 180,
    },
    {
      id: "l54-mix",
      label: ["目标域混合"],
      meta: "p ∝ n^α",
      kind: "decision",
      x: 800,
      y: 78,
    },
    {
      id: "l54-out",
      label: ["D_eff / n_eff"],
      meta: "分布还是重复",
      kind: "output",
      x: 800,
      y: 282,
    },
  ],
  edges: [
    {
      id: "l54-e-src-seg",
      from: "l54-src",
      to: "l54-seg",
      label: "遥操作轨迹",
      labelAt: { x: 172, y: 214 },
    },
    {
      id: "l54-e-seg-warp",
      from: "l54-seg",
      to: "l54-warp",
      label: "新物体位姿",
      via: [
        { x: 330, y: 180 },
        { x: 330, y: 78 },
      ],
      labelAt: { x: 286, y: 104 },
    },
    {
      id: "l54-e-seg-keep",
      from: "l54-seg",
      to: "l54-keep",
      label: "执行并判定",
      via: [
        { x: 330, y: 180 },
        { x: 330, y: 282 },
      ],
      labelAt: { x: 278, y: 256 },
    },
    {
      id: "l54-e-warp-hash",
      from: "l54-warp",
      to: "l54-hash",
      label: "新复位箱",
      labelAt: { x: 526, y: 104 },
    },
    {
      id: "l54-e-keep-hash",
      from: "l54-keep",
      to: "l54-hash",
      label: "失败丢弃",
      labelAt: { x: 526, y: 256 },
    },
    {
      id: "l54-e-hash-mix",
      from: "l54-hash",
      to: "l54-mix",
      label: "加到哪一域",
      via: [
        { x: 710, y: 180 },
        { x: 710, y: 78 },
      ],
      labelAt: { x: 668, y: 104 },
    },
    {
      id: "l54-e-hash-out",
      from: "l54-hash",
      to: "l54-out",
      label: "重复不算新样本",
      via: [
        { x: 710, y: 180 },
        { x: 710, y: 282 },
      ],
      labelAt: { x: 652, y: 256 },
    },
    {
      id: "l54-e-mix-out",
      from: "l54-mix",
      to: "l54-out",
      label: "α=1 账本",
    },
  ],
  steps: [
    {
      title: "源演示先按物体中心切开",
      description:
        "MimicGen 假设任务是已知的物体中心子任务序列。每条人类演示被切成相对单个物体坐标系的片段，而不是整段世界坐标回放。",
      focus: ["l54-src", "l54-seg", "l54-e-src-seg"],
    },
    {
      title: "用相对位姿接到新复位",
      description:
        "新场景里物体位姿变了，片段按 T_C' = T_O' T_O^{-1} T_C 变换，再从当前末端插值到新片段起点。附录 M 最后一行把两个物体位姿写反了，夹具用几何正确式。",
      focus: ["l54-warp", "l54-e-seg-warp", "l54-hash"],
    },
    {
      title: "成功才入库，生成率不是政策成功率",
      description:
        "执行完全部片段后只保留任务成功的尝试。真实机械臂 Stack 的生成成功率 82.3%，训出政策 36%。两列数字不能横着抄。",
      focus: ["l54-keep", "l54-e-seg-keep", "l54-e-keep-hash"],
    },
    {
      title: "合成必须写进目标域，而不是最大域",
      description:
        "四域真实计数 8000/800/200/50。2000 条只加到厨房时 α=1 的有效域数下降；同样 2000 条补双臂时小域份额从 0.55% 升到 18.6%。",
      focus: ["l54-mix", "l54-e-hash-mix", "l54-out"],
    },
    {
      title: "重复哈希不得增加有效样本量",
      description:
        "有效样本量 n_eff = (Σ n_i)² / Σ n_i²，哈希是源演示 ID 与复位箱。把 2000 条写成同一条厨房轨迹的复制，唯一哈希不变，n_eff 从 9050 掉到约 30。",
      focus: ["l54-hash", "l54-out", "l54-e-hash-out", "l54-e-mix-out"],
    },
  ],
  facts: [
    "MimicGen 摘要：用约 200 条人类演示生成超过 5 万条新演示，覆盖 18 个任务、两个仿真器和一台真实机械臂。",
    "相对位姿保持要求 T^{C'}_W = T^{O'}_W (T^{O}_W)^{-1} T^{C}_W。附录 M 推导前两行正确，最后一行把 T_O 与 T_{O'} 写反。",
    "RoboCasa：25 个原子任务各 50 条人类演示共 1250 条；MimicGen 为 24 个原子任务各生成 3000 条，共 7.2 万，另有 2.8 万条含 AI 物体，合计 10 万+。",
    "本课四域玩具：真实 8000/800/200/50。合成灌最大域时 D_eff 从 1.27 降到 1.21；补最小域时 D_eff 升到 1.77。重复轨迹的 n_eff 下降。",
  ],
};
