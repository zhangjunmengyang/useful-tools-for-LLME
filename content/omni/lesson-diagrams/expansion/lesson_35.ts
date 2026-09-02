import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson35Diagram: LessonDiagram = {
  lessonId: "35",
  title: "同一射线：RGB 命中不等于三维接触",
  summary:
    "像素 (u,v) 只定一条射线。缺深度时用场景均值取点，夹爪与杯子在图像上重合，却停在接触带外；接上 z 之后才落到可闭合的三维点，再进入 Ego3D 或自适应动作网格。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l35-uv",
      label: ["像素 (u,v)"],
      meta: "RGB 接地",
      kind: "input",
      x: 88,
      y: 108,
    },
    {
      id: "l35-z",
      label: ["深度 z"],
      meta: "传感 / 估计 / 均值",
      kind: "input",
      x: 88,
      y: 252,
    },
    {
      id: "l35-back",
      label: ["针孔反投影"],
      meta: "p = z · d(u,v)",
      kind: "transform",
      x: 268,
      y: 180,
    },
    {
      id: "l35-point",
      label: ["相机系点"],
      meta: "(X, Y, Z)",
      kind: "state",
      x: 448,
      y: 180,
    },
    {
      id: "l35-decide",
      label: ["接触带判定"],
      meta: "RGB 命中 vs 三维接触",
      kind: "decision",
      x: 628,
      y: 88,
    },
    {
      id: "l35-grid",
      label: ["动作网格"],
      meta: "体素 / Ego3D / SE(3)",
      kind: "transform",
      x: 628,
      y: 272,
    },
    {
      id: "l35-grasp",
      label: ["闭合夹爪"],
      meta: "接触带内才算抓",
      kind: "output",
      x: 832,
      y: 180,
      width: 150,
    },
  ],
  edges: [
    {
      id: "l35-e-uv-back",
      from: "l35-uv",
      to: "l35-back",
      label: "射线方向",
      via: [
        { x: 178, y: 108 },
        { x: 178, y: 180 },
      ],
      labelAt: { x: 132, y: 132 },
    },
    {
      id: "l35-e-z-back",
      from: "l35-z",
      to: "l35-back",
      label: "尺度",
      via: [
        { x: 178, y: 252 },
        { x: 178, y: 180 },
      ],
      labelAt: { x: 148, y: 228 },
    },
    {
      id: "l35-e-back-point",
      from: "l35-back",
      to: "l35-point",
      label: "三维点",
      labelAt: { x: 358, y: 214 },
    },
    {
      id: "l35-e-point-decide",
      from: "l35-point",
      to: "l35-decide",
      label: "是否进带",
      via: [
        { x: 538, y: 180 },
        { x: 538, y: 88 },
      ],
      labelAt: { x: 498, y: 124 },
    },
    {
      id: "l35-e-point-grid",
      from: "l35-point",
      to: "l35-grid",
      label: "空间分箱",
      via: [
        { x: 538, y: 180 },
        { x: 538, y: 272 },
      ],
      labelAt: { x: 498, y: 236 },
    },
    {
      id: "l35-e-decide-grasp",
      from: "l35-decide",
      to: "l35-grasp",
      label: "接触真值",
      via: [
        { x: 742, y: 88 },
        { x: 742, y: 180 },
      ],
      labelAt: { x: 778, y: 124 },
    },
    {
      id: "l35-e-grid-grasp",
      from: "l35-grid",
      to: "l35-grasp",
      label: "离散动作",
      via: [
        { x: 742, y: 272 },
        { x: 742, y: 180 },
      ],
      labelAt: { x: 778, y: 236 },
    },
  ],
  steps: [
    {
      title: "像素只定射线",
      description:
        "RGB 接地得到 (u,v)。针孔模型里，该像素对应一条从光心出发的射线，射线上每个深度都投到同一格。",
      focus: ["l35-uv", "l35-back", "l35-e-uv-back"],
    },
    {
      title: "深度给出尺度",
      description:
        "反投影 p = z · d(u,v)。z 来自传感深度、ZoeDepth 估计，或无深度时的场景均值。均值把点钉在错误尺度上。",
      focus: ["l35-z", "l35-back", "l35-point", "l35-e-z-back", "l35-e-back-point"],
    },
    {
      title: "两项判定必须拆开",
      description:
        "RGB 成功看图像距离；三维接触看 |Z| 与 XY 是否落入接触带。同一射线、错误 z 时前者为真、后者为假。",
      focus: ["l35-point", "l35-decide", "l35-e-point-decide"],
    },
    {
      title: "第三维进入动作",
      description:
        "PerAct 把点写进 100³ 体素；SpatialVLA 用 Ego3D 给视觉 token 加相机系位置，再用自适应网格把平移写成极坐标格子。",
      focus: ["l35-point", "l35-grid", "l35-e-point-grid"],
    },
    {
      title: "闭合发生在接触带",
      description:
        "夹爪只有进接触带才算抓住。PointVLA 把点云注入冻住的动作专家，救的是高度变化和照片欺骗，不是把 RGB 命中改名为成功率。",
      focus: ["l35-decide", "l35-grid", "l35-grasp", "l35-e-decide-grasp", "l35-e-grid-grasp"],
    },
  ],
  facts: [
    "SpatialVLA 用 ZoeDepth 估计深度，在相机系做 Ego3D 位置编码，不依赖机器人–相机外参；去掉 Ego3D 后 SimplerEnv 变体聚合从 81.6% / 79.2% 降到 68.9% / 66.7%。",
    "Adaptive Action Grids 把平移写成极坐标 (φ,θ,r)，词表 V=8194，每步 3 个空间动作 token，动作块 T=4。",
    "PerAct 默认 100³ 体素对应 1.0 m³，旋转每轴 5°、72 档，三个轴共 216 维 logits。",
    "Act3D 用传感深度把冻结的 CLIP 特征抬到 3D；真机失败的主因之一是深度噪声。",
    "PointVLA 冻住动作专家，只在可跳过的扩散块上加点云；桌面泡沫从训练 3 mm 升到测试 52 mm 时，纯 2D VLA 按训练高度去抓。",
  ],
};
