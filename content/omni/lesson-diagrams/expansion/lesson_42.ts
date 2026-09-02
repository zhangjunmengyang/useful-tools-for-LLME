import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson42Diagram: LessonDiagram = {
  lessonId: "42",
  title: "字、图、声各走采样器，只把提交结果写入共享 KV",
  summary:
    "日程表决定阶段顺序。文本自回归逐步提交；图像在工作区里走完扩散或流匹配内步，只把干净块写入 KV；声音同样有独立采样器。内步不得占用文本的因果位置。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l42-prompt",
      label: ["用户提示"],
      meta: "要一段字+图+声",
      kind: "input",
      x: 90,
      y: 180,
    },
    {
      id: "l42-schedule",
      label: ["阶段日程"],
      meta: "S = 字 / 图 / 声",
      kind: "decision",
      x: 270,
      y: 180,
    },
    {
      id: "l42-text",
      label: ["文本 AR"],
      meta: "next-token / 逐步提交",
      kind: "transform",
      x: 460,
      y: 72,
    },
    {
      id: "l42-image",
      label: ["图像工作区"],
      meta: "T 步扩散 / 流匹配",
      kind: "state",
      x: 460,
      y: 180,
    },
    {
      id: "l42-audio",
      label: ["声音采样器"],
      meta: "codec 帧 / Talker",
      kind: "transform",
      x: 460,
      y: 288,
    },
    {
      id: "l42-kv",
      label: ["已提交 KV"],
      meta: "只含干净前缀",
      kind: "state",
      x: 650,
      y: 180,
    },
    {
      id: "l42-reply",
      label: ["交错回复"],
      meta: "顺序等于日程",
      kind: "output",
      x: 840,
      y: 180,
      width: 150,
    },
  ],
  edges: [
    {
      id: "l42-e-prompt-schedule",
      from: "l42-prompt",
      to: "l42-schedule",
      label: "排阶段",
      labelAt: { x: 180, y: 228 },
    },
    {
      id: "l42-e-schedule-text",
      from: "l42-schedule",
      to: "l42-text",
      label: "字阶段",
      via: [
        { x: 365, y: 180 },
        { x: 365, y: 72 },
      ],
      labelAt: { x: 318, y: 112 },
    },
    {
      id: "l42-e-schedule-image",
      from: "l42-schedule",
      to: "l42-image",
      label: "图阶段",
      labelAt: { x: 365, y: 152 },
    },
    {
      id: "l42-e-schedule-audio",
      from: "l42-schedule",
      to: "l42-audio",
      label: "声阶段",
      via: [
        { x: 365, y: 180 },
        { x: 365, y: 288 },
      ],
      labelAt: { x: 318, y: 248 },
    },
    {
      id: "l42-e-text-kv",
      from: "l42-text",
      to: "l42-kv",
      label: "提交 token",
      via: [
        { x: 555, y: 72 },
        { x: 555, y: 180 },
      ],
      labelAt: { x: 582, y: 108 },
    },
    {
      id: "l42-e-image-kv",
      from: "l42-image",
      to: "l42-kv",
      label: "只提交干净块",
      labelAt: { x: 555, y: 152 },
    },
    {
      id: "l42-e-audio-kv",
      from: "l42-audio",
      to: "l42-kv",
      label: "提交音频帧",
      via: [
        { x: 555, y: 288 },
        { x: 555, y: 180 },
      ],
      labelAt: { x: 582, y: 252 },
    },
    {
      id: "l42-e-kv-reply",
      from: "l42-kv",
      to: "l42-reply",
      label: "按日程播放",
      labelAt: { x: 745, y: 228 },
    },
  ],
  steps: [
    {
      title: "先排日程，再选采样器",
      description:
        "一段回复被切成阶段。每个阶段指定模态、提交长度、内步数和写入策略。KV 图和采样器图不是同一张。",
      focus: ["l42-prompt", "l42-schedule", "l42-e-prompt-schedule"],
    },
    {
      title: "文本逐步提交",
      description:
        "文本阶段每预测一个 token 就写入共享 KV，后续位置按因果 mask 读取。",
      focus: ["l42-schedule", "l42-text", "l42-kv", "l42-e-schedule-text", "l42-e-text-kv"],
    },
    {
      title: "图像内步停在工作区",
      description:
        "BOI 之后进入扩散或流匹配。Transfusion 在原位覆盖同一组 patch；BAGEL 的噪声 VAE 不作为后续文本的 key。只有干净块进入 KV。",
      focus: ["l42-image", "l42-kv", "l42-e-schedule-image", "l42-e-image-kv"],
    },
    {
      title: "声音也是独立采样器",
      description:
        "codec 解码或 Talker 的内部帧不等于文本 token。内步同样不得占用文本的因果位置。",
      focus: ["l42-audio", "l42-kv", "l42-e-schedule-audio", "l42-e-audio-kv"],
    },
    {
      title: "输出顺序等于日程",
      description:
        "把字-图-字改成字-字-图，提交顺序必须变。共享 KV 不能把三种模态抹成同一次写出。",
      focus: ["l42-schedule", "l42-kv", "l42-reply", "l42-e-kv-reply"],
    },
  ],
  facts: [
    "Transfusion 在采样到 BOI 后把噪声 patch 接到序列上，每步覆盖同一组向量，不能回头看更早的噪声步。",
    "0.76B Transfusion 把图内注意力从因果改成双向后，MS-COCO FID 从 61.3 降到 20.3（线性编码，表 5）。",
    "BAGEL 规定后续图或字只看干净 VAE 和 ViT，不看对应的噪声 VAE；推理时生成完毕即用干净块替换噪声块。",
    "Show-o2 用 omni-attention：序列方向因果，统一视觉表示内部全连接；混合生成时预测到 BOI 再接噪声出图。",
    "Transfusion 训练损失为 L_LM + λ L_DDPM，文中取 λ=5。",
  ],
};
