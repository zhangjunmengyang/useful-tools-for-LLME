import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson56Diagram: LessonDiagram = {
  lessonId: "56",
  title: "L2 序与偏好序在打分口对打",
  summary:
    "同一组生成图先走像素 L2，再走偏好奖励模型。两把尺子各自给出排序，Kendall τ 为负时不得只用重建误差验收。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l56-cands",
      label: ["候选图"],
      meta: "同一 prompt",
      kind: "input",
      x: 96,
      y: 86,
      width: 132,
    },
    {
      id: "l56-ref",
      label: ["参考或均值"],
      meta: "GT / MMSE",
      kind: "input",
      x: 96,
      y: 254,
      width: 132,
    },
    {
      id: "l56-l2",
      label: ["像素 L2"],
      meta: "重建误差",
      kind: "transform",
      x: 318,
      y: 86,
      width: 132,
    },
    {
      id: "l56-rm",
      label: ["偏好 RM"],
      meta: "BT 标量 r",
      kind: "transform",
      x: 318,
      y: 254,
      width: 132,
    },
    {
      id: "l56-rank-l2",
      label: ["L2 序"],
      meta: "低误差靠前",
      kind: "state",
      x: 540,
      y: 86,
      width: 132,
    },
    {
      id: "l56-rank-rm",
      label: ["偏好序"],
      meta: "高 r 靠前",
      kind: "state",
      x: 540,
      y: 254,
      width: 132,
    },
    {
      id: "l56-tau",
      label: ["Kendall τ"],
      meta: "两序相关",
      kind: "decision",
      x: 742,
      y: 170,
      width: 140,
    },
  ],
  edges: [
    {
      id: "l56-e-cands-l2",
      from: "l56-cands",
      to: "l56-l2",
      label: "逐张算 MSE",
      labelAt: { x: 208, y: 54 },
    },
    {
      id: "l56-e-ref-l2",
      from: "l56-ref",
      to: "l56-l2",
      label: "减参考像素",
      via: [
        { x: 200, y: 170 },
        { x: 318, y: 170 },
      ],
      labelAt: { x: 230, y: 154 },
    },
    {
      id: "l56-e-cands-rm",
      from: "l56-cands",
      to: "l56-rm",
      label: "图+文打分",
      via: [
        { x: 96, y: 170 },
        { x: 260, y: 170 },
      ],
      labelAt: { x: 150, y: 186 },
    },
    {
      id: "l56-e-l2-rank",
      from: "l56-l2",
      to: "l56-rank-l2",
      label: "升序误差",
    },
    {
      id: "l56-e-rm-rank",
      from: "l56-rm",
      to: "l56-rank-rm",
      label: "降序 r",
    },
    {
      id: "l56-e-l2rank-tau",
      from: "l56-rank-l2",
      to: "l56-tau",
      label: "秩向量",
      labelAt: { x: 660, y: 86 },
    },
    {
      id: "l56-e-rmrank-tau",
      from: "l56-rank-rm",
      to: "l56-tau",
      label: "秩向量",
      labelAt: { x: 660, y: 286 },
    },
  ],
  steps: [
    {
      title: "同一组候选进两把尺子",
      description:
        "prompt 固定。候选可以是过平滑重建，也可以是锐利但像素错位的样本。参考图只服务 L2，不进入奖励模型的必要条件。",
      focus: ["l56-cands", "l56-ref", "l56-e-cands-l2", "l56-e-cands-rm"],
    },
    {
      title: "L2 量像素差",
      description:
        "MSE 是像素独立的欧氏距离。1 像素平移会让边对不齐，误差暴涨；盒滤波抹掉高频，误差往往更小。",
      focus: ["l56-l2", "l56-ref", "l56-e-ref-l2", "l56-e-l2-rank"],
    },
    {
      title: "RM 量人更想留下哪张",
      description:
        "Bradley–Terry 把 pairwise 选择写成 σ(r_i-r_j)。ImageReward / PickScore / HPSv2 学的是这个标量，不是像素重建。",
      focus: ["l56-rm", "l56-rank-rm", "l56-e-rm-rank"],
    },
    {
      title: "两序对打",
      description:
        "L2 低的排前面，r 高的排前面。过平滑常赢 L2、输偏好。这不是并列，是方向相反。",
      focus: ["l56-rank-l2", "l56-rank-rm", "l56-tau"],
    },
    {
      title: "Kendall τ 为负则拒用单尺",
      description:
        "τ=(C-D)/C(n,2)。夹具四档完全反序时 τ=-1。负值出现时，不得把更低 L2 写成更好生成。",
      focus: ["l56-tau", "l56-e-l2rank-tau", "l56-e-rmrank-tau"],
    },
  ],
  facts: [
    "ImageReward Table 1：真实用户 prompt 上 FID 与人类排序的 Spearman ρ=0.09，ImageReward 为 1.00。",
    "Pick-a-Pic：MS-COCO 标题上 PickScore 与人类相关 0.917，FID 为 -0.900。",
    "本课 CPU 夹具：过平滑 L2 最低且偏好最低，平移图相反，Kendall τ-a = -1。",
    "奖励损失与 ImageReward 式 (1) 同形：-log σ(r_w-r_l)，不是像素 MSE。",
  ],
};
