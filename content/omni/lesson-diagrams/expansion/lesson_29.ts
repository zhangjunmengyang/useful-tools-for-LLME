import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson29Diagram: LessonDiagram = {
  lessonId: "29",
  title: "快慢双时钟：子目标进入高频动作专家",
  summary:
    "System 2 按规划周期写出子目标或 VLM token；System 1 在每个控制周期消费当前条件，过期则钉在陈旧目标上。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l29-obs",
      label: ["图像与指令"],
      meta: "相机 + 语言",
      kind: "input",
      x: 108,
      y: 88,
    },
    {
      id: "l29-s2",
      label: ["System 2"],
      meta: "Eagle VLM · ΔT2",
      kind: "transform",
      x: 308,
      y: 88,
    },
    {
      id: "l29-latent",
      label: ["当前子目标"],
      meta: "token / 语义",
      kind: "state",
      x: 512,
      y: 88,
    },
    {
      id: "l29-expire",
      label: ["过期判定"],
      meta: "age > T_exp",
      kind: "decision",
      x: 742,
      y: 88,
    },
    {
      id: "l29-state",
      label: ["本体感受"],
      meta: "关节 / 夹爪",
      kind: "state",
      x: 308,
      y: 262,
    },
    {
      id: "l29-s1",
      label: ["System 1"],
      meta: "DiT · ΔT1",
      kind: "transform",
      x: 512,
      y: 262,
    },
    {
      id: "l29-action",
      label: ["动作块"],
      meta: "H 步关节",
      kind: "output",
      x: 742,
      y: 262,
    },
  ],
  edges: [
    {
      id: "l29-e-obs-s2",
      from: "l29-obs",
      to: "l29-s2",
      label: "视觉语言编码",
      labelAt: { x: 208, y: 58 },
    },
    {
      id: "l29-e-s2-latent",
      from: "l29-s2",
      to: "l29-latent",
      label: "规划周期写出",
      labelAt: { x: 410, y: 58 },
    },
    {
      id: "l29-e-latent-expire",
      from: "l29-latent",
      to: "l29-expire",
      label: "计算年龄",
      labelAt: { x: 628, y: 58 },
    },
    {
      id: "l29-e-latent-s1",
      from: "l29-latent",
      to: "l29-s1",
      label: "交叉注意 / 子任务",
    },
    {
      id: "l29-e-expire-s1",
      from: "l29-expire",
      to: "l29-s1",
      label: "新鲜或陈旧",
      via: [{ x: 820, y: 88 }, { x: 820, y: 262 }],
      labelAt: { x: 868, y: 176 },
    },
    {
      id: "l29-e-state-s1",
      from: "l29-state",
      to: "l29-s1",
      label: "每控制步",
      labelAt: { x: 410, y: 292 },
    },
    {
      id: "l29-e-s1-action",
      from: "l29-s1",
      to: "l29-action",
      label: "flow / 去噪",
      labelAt: { x: 628, y: 292 },
    },
    {
      id: "l29-e-action-state",
      from: "l29-action",
      to: "l29-state",
      label: "执行后更新",
      via: [{ x: 742, y: 330 }, { x: 308, y: 330 }],
      labelAt: { x: 525, y: 348 },
    },
  ],
  steps: [
    {
      title: "低频编码观察与指令",
      description:
        "相机帧和语言进入 System 2。GR00T N1 的 Eagle-2 在 L40 上约 10 Hz 产出视觉语言表征，而不是每控制步重跑整颗 VLM。",
      focus: ["l29-obs", "l29-s2", "l29-e-obs-s2"],
    },
    {
      title: "写出当前子目标",
      description:
        "规划周期把条件写成可被专家消费的对象：GR00T 是 VLM token，π0.5 是语义子任务文本。该对象在两次规划之间保持不变。",
      focus: ["l29-s2", "l29-latent", "l29-e-s2-latent"],
    },
    {
      title: "核对年龄与过期",
      description:
        "子目标年龄 age = nΔT1 − t_plan。超过 T_exp 后条件陈旧：专家仍会出动作，但任务时钟应记失败并准备重规划。",
      focus: ["l29-latent", "l29-expire", "l29-e-latent-expire", "l29-e-expire-s1"],
    },
    {
      title: "高频专家消费当前条件",
      description:
        "每个控制周期，System 1 读取本体感受和当前子目标，用 DiT / flow 生成动作块。规划暂停时它不会发明下一阶段，只会钉在最后条件上。",
      focus: [
        "l29-state",
        "l29-s1",
        "l29-latent",
        "l29-e-state-s1",
        "l29-e-latent-s1",
      ],
    },
    {
      title: "执行动作块并回写状态",
      description:
        "动作块交给底层跟踪。GR00T N1 在 L40 上 4 步去噪、块长 16；π0.5 以 50 Hz 发目标位姿。错误子目标会把末端轨迹从货架改到别处。",
      focus: ["l29-s1", "l29-action", "l29-state", "l29-e-s1-action", "l29-e-action-state"],
    },
  ],
  facts: [
    "GR00T N1：System 2 在 L40 上 10 Hz，System 1 120 Hz；H=16，K=4，bf16 采样 16 步 63.9 ms（arXiv:2503.14734）。",
    "规划周期与控制周期满足 ΔT2 = k ΔT1；控制环每步消费当前子目标，规划环按 k 步触发。",
    "π0.5 先采样子任务 ℓ̂ 再出动作块，控制 50 Hz；动作分布条件于 ℓ̂ 而不是原始任务句（arXiv:2504.16054）。",
    "机理研究实测 π0.5、OpenVLA-OFT、X-VLA、SmolVLA、GR00T N1.5、ACT；多通路里专家偏动作、VLM 偏目标语义（arXiv:2603.19233）。",
  ],
};
