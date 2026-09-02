import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson30Diagram: LessonDiagram = {
  lessonId: "30",
  title: "控制时钟上的分块、延迟与重规划",
  summary:
    "观察进入推理后要等延迟 d 才得到长度为 H 的动作块；用 H/f 判定是否过期，再按 CONTINUE / PAUSE / REPLAN 决定执行前 k 步还是丢弃剩余步。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l30-obs",
      label: ["相机观察"],
      meta: "t_obs",
      kind: "input",
      x: 88,
      y: 88,
      width: 128,
    },
    {
      id: "l30-infer",
      label: ["动作推理"],
      meta: "延迟 d",
      kind: "transform",
      x: 278,
      y: 88,
      width: 132,
    },
    {
      id: "l30-chunk",
      label: ["动作 chunk"],
      meta: "H 步",
      kind: "state",
      x: 478,
      y: 88,
      width: 132,
    },
    {
      id: "l30-stale",
      label: ["过期判定"],
      meta: "d ? H/f",
      kind: "decision",
      x: 698,
      y: 88,
      width: 140,
    },
    {
      id: "l30-control",
      label: ["控制头"],
      meta: "CONTINUE/PAUSE/REPLAN",
      kind: "decision",
      x: 278,
      y: 268,
      width: 196,
    },
    {
      id: "l30-exec",
      label: ["执行前 k 步"],
      meta: "提交窗口 k/f",
      kind: "transform",
      x: 548,
      y: 268,
      width: 148,
    },
    {
      id: "l30-grasp",
      label: ["抓住或抓空"],
      meta: "截止前接触",
      kind: "output",
      x: 790,
      y: 268,
      width: 140,
    },
  ],
  edges: [
    {
      id: "l30-e-obs-infer",
      from: "l30-obs",
      to: "l30-infer",
      label: "编码观察",
      labelAt: { x: 178, y: 54 },
    },
    {
      id: "l30-e-infer-chunk",
      from: "l30-infer",
      to: "l30-chunk",
      label: "available_at = t_obs + d",
      labelAt: { x: 368, y: 54 },
    },
    {
      id: "l30-e-chunk-stale",
      from: "l30-chunk",
      to: "l30-stale",
      label: "开环窗口 H/f",
      labelAt: { x: 588, y: 54 },
    },
    {
      id: "l30-e-stale-exec",
      from: "l30-stale",
      to: "l30-exec",
      label: "fresh chunk",
      via: [
        { x: 698, y: 168 },
        { x: 548, y: 168 },
      ],
      labelAt: { x: 640, y: 148 },
    },
    {
      id: "l30-e-stale-control",
      from: "l30-stale",
      to: "l30-control",
      label: "过期则丢弃",
      via: [
        { x: 698, y: 200 },
        { x: 278, y: 200 },
      ],
      labelAt: { x: 470, y: 182 },
    },
    {
      id: "l30-e-control-exec",
      from: "l30-control",
      to: "l30-exec",
      label: "CONTINUE 执行剩余步",
      labelAt: { x: 408, y: 318 },
    },
    {
      id: "l30-e-control-infer",
      from: "l30-control",
      to: "l30-infer",
      label: "REPLAN 新推理",
    },
    {
      id: "l30-e-exec-grasp",
      from: "l30-exec",
      to: "l30-grasp",
      label: "接触判定",
      labelAt: { x: 668, y: 318 },
    },
  ],
  steps: [
    {
      title: "观察进入推理",
      description:
        "控制时钟在 t_obs 锁存图像与本体感觉，启动一次长度为 H 的动作推理。",
      focus: ["l30-obs", "l30-infer", "l30-e-obs-infer"],
    },
    {
      title: "延迟后 chunk 才可用",
      description:
        "chunk 的 available_at 是 t_obs + d。d 包含视觉编码、网络前向和任何积分或去噪步。",
      focus: ["l30-infer", "l30-chunk", "l30-e-infer-chunk"],
    },
    {
      title: "用 H/f 判定过期",
      description:
        "开环窗口是 H/f。若 d 已经大于等于该窗口，到达的块描述的是已经过去的时间，必须丢弃。",
      focus: ["l30-chunk", "l30-stale", "l30-e-chunk-stale", "l30-e-stale-control"],
    },
    {
      title: "三动作状态机",
      description:
        "CONTINUE 继续当前块，PAUSE 冻结剩余步，REPLAN 丢弃旧剩余步并开新推理。与第 07 课同构，时钟换成控制周期。",
      focus: ["l30-control", "l30-infer", "l30-e-control-infer"],
    },
    {
      title: "只提交前 k 步",
      description:
        "后退视野只执行前 k 步，提交窗口为 k/f。要在提交结束前拿到下一块，需要 d < k/f。",
      focus: ["l30-exec", "l30-stale", "l30-e-stale-exec", "l30-e-control-exec"],
    },
    {
      title: "截止前抓住或抓空",
      description:
        "物体中途被挪走后，只有 fresh chunk 能改闭合时刻；过期块被丢弃则夹爪在空处闭合或根本不合。",
      focus: ["l30-exec", "l30-grasp", "l30-e-exec-grasp"],
    },
  ],
  facts: [
    "开环窗口定义为 T_open = H/f，提交窗口定义为 T_commit = k/f。",
    "稳定条件的直觉形式：推理延迟 d 必须小于 H/f，否则执行的是过期 chunk。",
    "REPLAN 将旧 branch 标为 superseded，未执行的剩余步不得再下发。",
    "ACT 的 ALOHA 在 50 Hz 记录与控制；π0 使用 H=50 的动作块，最高控制频率 50 Hz。",
    "OpenVLA 在 A100 上生成单步 7 维动作约 0.33 s；OFT 在 K=8 并行解码后吞吐约 26 倍。",
  ],
};
