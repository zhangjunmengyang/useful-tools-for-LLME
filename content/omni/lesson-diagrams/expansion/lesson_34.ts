import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson34Diagram: LessonDiagram = {
  lessonId: "34",
  title: "数据引擎看违规，控制器看滚动执行",
  summary:
    "观察和动作先进入世界模型；生成路用物体 ID 计数器与重力符号探针当数据门禁，控制路用同一套预测选动作。两条出路的数字不能横着比。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l34-obs",
      label: ["观察视频"],
      meta: "落下的方块",
      kind: "input",
      x: 88,
      y: 108,
    },
    {
      id: "l34-act",
      label: ["动作条件"],
      meta: "c_t / 潜伏动作",
      kind: "input",
      x: 88,
      y: 252,
    },
    {
      id: "l34-tok",
      label: ["视频 tokenizer"],
      meta: "连续或离散 token",
      kind: "transform",
      x: 268,
      y: 180,
    },
    {
      id: "l34-wfm",
      label: ["世界模型"],
      meta: "AR / diffusion / MoT",
      kind: "transform",
      x: 448,
      y: 180,
    },
    {
      id: "l34-probe",
      label: ["物理探针"],
      meta: "ID 丢失 / 重力符号",
      kind: "decision",
      x: 628,
      y: 180,
    },
    {
      id: "l34-engine",
      label: ["数据引擎"],
      meta: "保真 + 违规率",
      kind: "output",
      x: 820,
      y: 88,
      width: 150,
    },
    {
      id: "l34-ctrl",
      label: ["控制器"],
      meta: "滚动执行误差",
      kind: "output",
      x: 820,
      y: 272,
      width: 150,
    },
  ],
  edges: [
    {
      id: "l34-e-obs-tok",
      from: "l34-obs",
      to: "l34-tok",
      label: "x_{0:t}",
      via: [
        { x: 168, y: 108 },
        { x: 168, y: 180 },
      ],
      labelAt: { x: 138, y: 132 },
    },
    {
      id: "l34-e-act-wfm",
      from: "l34-act",
      to: "l34-wfm",
      label: "动作条件",
      via: [
        { x: 168, y: 252 },
        { x: 168, y: 300 },
        { x: 448, y: 300 },
      ],
      labelAt: { x: 250, y: 318 },
    },
    {
      id: "l34-e-tok-wfm",
      from: "l34-tok",
      to: "l34-wfm",
      label: "token 序列",
      labelAt: { x: 358, y: 148 },
    },
    {
      id: "l34-e-wfm-probe",
      from: "l34-wfm",
      to: "l34-probe",
      label: "预测帧",
      labelAt: { x: 538, y: 148 },
    },
    {
      id: "l34-e-probe-engine",
      from: "l34-probe",
      to: "l34-engine",
      label: "丢掉 ID 即拒",
      via: [
        { x: 720, y: 180 },
        { x: 720, y: 88 },
      ],
      labelAt: { x: 742, y: 124 },
    },
    {
      id: "l34-e-probe-ctrl",
      from: "l34-probe",
      to: "l34-ctrl",
      label: "符号反了即拒",
      via: [
        { x: 720, y: 180 },
        { x: 720, y: 272 },
      ],
      labelAt: { x: 742, y: 236 },
    },
    {
      id: "l34-e-wfm-ctrl",
      from: "l34-wfm",
      to: "l34-ctrl",
      label: "选动作",
      via: [
        { x: 448, y: 330 },
        { x: 820, y: 330 },
      ],
      labelAt: { x: 620, y: 348 },
    },
  ],
  steps: [
    {
      title: "观察加动作",
      description:
        "世界模型吃过去的视频，以及当前扰动：文本、相机、末端位移，或 Genie 那种从无标签视频学到的潜伏动作。",
      focus: ["l34-obs", "l34-act", "l34-tok", "l34-e-obs-tok"],
    },
    {
      title: "预测下一帧",
      description:
        "Cosmos 1 用扩散或自回归；Cosmos 3 把语言、图像、视频、音频、动作放进同一套 Mixture-of-Transformers。输出仍是未来观察。",
      focus: ["l34-tok", "l34-wfm", "l34-e-tok-wfm", "l34-e-act-wfm"],
    },
    {
      title: "物理探针",
      description:
        "跨帧物体 ID 计数器抓自发消失；自由下落的 Δv_y 符号必须与真重力一致。Cosmos 原文承认这两类失败。",
      focus: ["l34-wfm", "l34-probe", "l34-e-wfm-probe"],
    },
    {
      title: "数据引擎出路",
      description:
        "合成视频给别人训时，门禁是保真和物理违规率。丢掉 ID 或违反重力的片段不得写进训练桶。",
      focus: ["l34-probe", "l34-engine", "l34-e-probe-engine"],
    },
    {
      title: "控制器出路",
      description:
        "用预测选动作时，门禁是真实世界里的滚动执行误差。重力符号反了，捕捉目标会偏到对面。教学数字不能写成真机成功率。",
      focus: ["l34-probe", "l34-ctrl", "l34-e-probe-ctrl", "l34-e-wfm-ctrl"],
    },
  ],
  facts: [
    "Cosmos（arXiv:2501.03575）在 Limitations 中承认模型仍缺物体永久性，生成视频也不总遵守重力、光照和流体。",
    "同一论文 5.3.2 把物体不永久性写成自发出现和消失，并把违反重力列为刚体仿真里的失败模式。",
    "Cosmos-Predict1-7B-Video2World 在 9 帧条件下物理符合度表的 PSNR 为 21.06、平均 IoU 为 0.592，且更大模型并未在该表上更好。",
    "Genie（arXiv:2402.15391）从无标签视频学 |A|=8 的潜伏动作，11B 模型在 16 帧、10 FPS 上逐步可控。",
    "Cosmos 3（arXiv:2606.02800）用 Mixture-of-Transformers 同时处理语言、图像、视频、音频和动作，Physics-IQ 上 Super 的 I2V 直出分为 43.8。",
  ],
};
