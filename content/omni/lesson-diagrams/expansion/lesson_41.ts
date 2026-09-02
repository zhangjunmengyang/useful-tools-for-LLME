import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson41Diagram: LessonDiagram = {
  lessonId: "41",
  title: "离散图像 token 与文本共用 next-token",
  summary:
    "像素经 VQ 变成码本编号，与文本进入统一词表；任务决定理解双向或生成因果 mask，共享 Transformer 再对同一套 softmax 做 next-token。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l41-img",
      label: ["像素网格"],
      meta: "H×W 图像",
      kind: "input",
      x: 100,
      y: 90,
    },
    {
      id: "l41-vq",
      label: ["VQ tokenizer"],
      meta: "下采样 f，码本 K",
      kind: "transform",
      x: 280,
      y: 90,
    },
    {
      id: "l41-vocab",
      label: ["统一词表"],
      meta: "V_text + K",
      kind: "state",
      x: 470,
      y: 90,
    },
    {
      id: "l41-core",
      label: ["共享 Transformer"],
      meta: "同一套参数",
      kind: "transform",
      x: 680,
      y: 90,
    },
    {
      id: "l41-text",
      label: ["文本 token"],
      meta: "BPE / 句子片",
      kind: "input",
      x: 100,
      y: 270,
    },
    {
      id: "l41-mask",
      label: ["注意力 mask"],
      meta: "理解全图 / 生成因果",
      kind: "decision",
      x: 470,
      y: 270,
    },
    {
      id: "l41-head",
      label: ["共享 softmax"],
      meta: "图像 CE = 文本 CE",
      kind: "output",
      x: 858,
      y: 180,
    },
  ],
  edges: [
    {
      id: "l41-e-img-vq",
      from: "l41-img",
      to: "l41-vq",
      label: "切块查表",
      labelAt: { x: 190, y: 54 },
    },
    {
      id: "l41-e-vq-vocab",
      from: "l41-vq",
      to: "l41-vocab",
      label: "离散 id",
      labelAt: { x: 375, y: 54 },
    },
    {
      id: "l41-e-text-vocab",
      from: "l41-text",
      to: "l41-vocab",
      label: "文本 id",
      via: [
        { x: 100, y: 180 },
        { x: 280, y: 180 },
      ],
      labelAt: { x: 168, y: 214 },
    },
    {
      id: "l41-e-vocab-mask",
      from: "l41-vocab",
      to: "l41-mask",
      label: "按任务打包",
      labelAt: { x: 404, y: 180 },
    },
    {
      id: "l41-e-mask-core-u",
      from: "l41-mask",
      to: "l41-core",
      label: "理解：视觉双向",
      via: [
        { x: 560, y: 270 },
        { x: 560, y: 90 },
      ],
      labelAt: { x: 598, y: 214 },
    },
    {
      id: "l41-e-mask-core-g",
      from: "l41-mask",
      to: "l41-core",
      label: "生成：禁止未来格",
      via: [
        { x: 640, y: 270 },
        { x: 680, y: 200 },
      ],
      labelAt: { x: 720, y: 252 },
    },
    {
      id: "l41-e-core-head",
      from: "l41-core",
      to: "l41-head",
      label: "next-token",
      labelAt: { x: 778, y: 64 },
    },
  ],
  steps: [
    {
      title: "图像变成编号",
      description:
        "VQ tokenizer 把 H×W 图压成 (H/f)×(W/f) 个离散 id。每个 id 必须落在码本 [0, K-1]。",
      focus: ["l41-img", "l41-vq", "l41-e-img-vq"],
    },
    {
      title: "写入统一词表",
      description:
        "文本 BPE 与图像码本拼成一张表。图像 id 加上文本词表偏移后，和字用同一套嵌入。",
      focus: ["l41-vq", "l41-text", "l41-vocab", "l41-e-vq-vocab", "l41-e-text-vocab"],
    },
    {
      title: "按任务选 mask",
      description:
        "理解路径让图像 token 互相看见，才能读全图。生成路径按光栅顺序因果，当前格不能看未来像素 token。",
      focus: ["l41-vocab", "l41-mask", "l41-e-vocab-mask"],
    },
    {
      title: "共享骨干",
      description:
        "Chameleon / Emu3 在因果 mask 上做 next-token；Show-o 对图像另用全注意力的 mask token 预测。参数仍是同一个 Transformer。",
      focus: [
        "l41-mask",
        "l41-core",
        "l41-e-mask-core-u",
        "l41-e-mask-core-g",
      ],
    },
    {
      title: "同一套 softmax",
      description:
        "图像 CE 与文本 CE 共享 V_text+K 维 softmax。loss 盖在文本位置还是图像位置，由任务决定，不是两套分类器。",
      focus: ["l41-core", "l41-head", "l41-e-core-head"],
    },
  ],
  facts: [
    "Chameleon 把 512×512 图编成 1024 个离散 token，码本 8192，BPE 词表 65536 含图像编号。",
    "Emu3 把 512×512 图或 4×512×512 视频编成 4096 个离散 token，码本 32768，8B 模型只做 next-token 交叉熵。",
    "Show-o 用 MAGVIT-v2：256×256 到 16×16、码本 8192；文本因果、图像全注意力，损失为 MTP 加 NTP。",
    "Show-o 附录对照：同样 512×512、下采样 16 时自回归要 1024 步，离散扩散约 50 步，大约少 20 倍。",
  ],
};
