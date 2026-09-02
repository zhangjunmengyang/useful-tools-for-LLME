import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson50Diagram: LessonDiagram = {
  lessonId: "50",
  title: "同一份 SLAT，三路解码器",
  summary:
    "条件先生成稀疏占用，再生成附着在活跃体素上的局部 latent。这份 Structured LATent 被三个只读解码器分别变成 mesh、3D Gaussian 和辐射场；改高斯半径不得写回 SLAT，也不得改 mesh 拓扑。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l50-cond",
      label: ["文本 / 图像条件"],
      meta: "CLIP / DINOv2",
      kind: "input",
      x: 88,
      y: 180,
    },
    {
      id: "l50-struct",
      label: ["结构生成 G_S"],
      meta: "占用网格 O → {p_i}",
      kind: "transform",
      x: 268,
      y: 88,
    },
    {
      id: "l50-latgen",
      label: ["局部 latent 生成 G_L"],
      meta: "rectified flow",
      kind: "transform",
      x: 268,
      y: 272,
    },
    {
      id: "l50-slat",
      label: ["SLAT"],
      meta: "{(z_i, p_i)} 只读",
      kind: "state",
      x: 468,
      y: 180,
    },
    {
      id: "l50-pick",
      label: ["选择解码器"],
      meta: "输出层不同",
      kind: "decision",
      x: 648,
      y: 180,
    },
    {
      id: "l50-mesh",
      label: ["Mesh"],
      meta: "FlexiCubes / SDF",
      kind: "output",
      x: 848,
      y: 72,
      width: 140,
    },
    {
      id: "l50-gs",
      label: ["3D Gaussian"],
      meta: "K=32, x=p+tanh(o)",
      kind: "output",
      x: 848,
      y: 180,
      width: 140,
    },
    {
      id: "l50-rf",
      label: ["辐射场"],
      meta: "Strivec CP, 8³",
      kind: "output",
      x: 848,
      y: 288,
      width: 140,
    },
  ],
  edges: [
    {
      id: "l50-e-cond-struct",
      from: "l50-cond",
      to: "l50-struct",
      label: "条件",
      via: [
        { x: 178, y: 180 },
        { x: 178, y: 88 },
      ],
      labelAt: { x: 128, y: 124 },
    },
    {
      id: "l50-e-cond-lat",
      from: "l50-cond",
      to: "l50-latgen",
      label: "条件",
      via: [
        { x: 178, y: 180 },
        { x: 178, y: 272 },
      ],
      labelAt: { x: 128, y: 236 },
    },
    {
      id: "l50-e-struct-slat",
      from: "l50-struct",
      to: "l50-slat",
      label: "活跃体素 p",
      via: [
        { x: 368, y: 88 },
        { x: 368, y: 180 },
      ],
      labelAt: { x: 318, y: 124 },
    },
    {
      id: "l50-e-lat-slat",
      from: "l50-latgen",
      to: "l50-slat",
      label: "局部 z",
      via: [
        { x: 368, y: 272 },
        { x: 368, y: 180 },
      ],
      labelAt: { x: 322, y: 236 },
    },
    {
      id: "l50-e-slat-pick",
      from: "l50-slat",
      to: "l50-pick",
      label: "共享、只读",
      labelAt: { x: 548, y: 214 },
    },
    {
      id: "l50-e-pick-mesh",
      from: "l50-pick",
      to: "l50-mesh",
      label: "D_M",
      via: [
        { x: 742, y: 180 },
        { x: 742, y: 72 },
      ],
      labelAt: { x: 698, y: 112 },
    },
    {
      id: "l50-e-pick-gs",
      from: "l50-pick",
      to: "l50-gs",
      label: "D_GS",
      labelAt: { x: 748, y: 214 },
    },
    {
      id: "l50-e-pick-rf",
      from: "l50-pick",
      to: "l50-rf",
      label: "D_RF",
      via: [
        { x: 742, y: 180 },
        { x: 742, y: 288 },
      ],
      labelAt: { x: 698, y: 248 },
    },
  ],
  steps: [
    {
      title: "条件进入两段生成",
      description:
        "文本走 CLIP，图像走 DINOv2。第一段 rectified flow 生成稀疏占用，第二段在给定 {p_i} 上生成局部向量 z_i。",
      focus: ["l50-cond", "l50-struct", "l50-latgen", "l50-e-cond-struct", "l50-e-cond-lat"],
    },
    {
      title: "SLAT 是占用加局部特征",
      description:
        "z={(z_i,p_i)}，默认 64³ 网格上约 2 万个活跃体素。粗形状在 p，细几何和外观在 z。",
      focus: ["l50-struct", "l50-latgen", "l50-slat", "l50-e-struct-slat", "l50-e-lat-slat"],
    },
    {
      title: "解码器只读，输出层不同",
      description:
        "三个解码器骨干可以同构，输出层必须分开：FlexiCubes 权与 SDF、每体素 32 个高斯、Strivec CP 分解。",
      focus: ["l50-slat", "l50-pick", "l50-e-slat-pick"],
    },
    {
      title: "高斯半径停在 D_GS 里",
      description:
        "尺度 s 是高斯解码器的局部属性，位置还被 tanh 钉在体素附近。改 s 不得写回 z_i，也不得改 SDF 符号。",
      focus: ["l50-pick", "l50-gs", "l50-e-pick-gs"],
    },
    {
      title: "Mesh 与辐射场走另外两张表",
      description:
        "D_M 上采样到 256³ 抽 0 等值面；D_RF 组装 8³ 局部体积。写坏共享 latent 时这两路必须一起失败。",
      focus: ["l50-mesh", "l50-rf", "l50-e-pick-mesh", "l50-e-pick-rf"],
    },
  ],
  facts: [
    "TRELLIS 默认 SLAT 分辨率 64³、通道 8，平均活跃体素数约 2 万；表 3 中 32³ 即使把通道加到 64，PSNR 仍停在 31.85，改到 64³/8 才到 32.74。",
    "高斯解码每活跃体素预测 K=32 个高斯，位置 x=p+tanh(o)；辐射场为秩 16 的 CP 分解，局部体积 8³；mesh 经两次稀疏上采样到 256³，每格 FlexiCubes 参数 45 维、8 个 SDF。",
    "编码器与高斯解码器端到端训练后冻结编码器，再从头训练辐射场与 mesh 解码器。",
    "Toys4k 重建：外观 PSNR 32.74 / 辐射场 32.19，LPIPS 0.025 / 0.029，Chamfer 0.0083，F-score 0.9999。",
    "XL 文生 3D 在 Toys4k 上 CLIP 26.70、FD_dinov2 237.48；训练约 50 万资产、64×A100 40G、40 万步、batch 256。",
  ],
};
