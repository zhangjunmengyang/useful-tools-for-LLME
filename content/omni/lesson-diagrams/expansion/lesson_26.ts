import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson26Diagram: LessonDiagram = {
  lessonId: "26",
  title: "异构机体数据的混合与捷径探针",
  summary:
    "OXE 先做末端动作的粗对齐，再按 n^α 与每域上限组成 batch；打乱指令或机体标签，用来区分真语言条件和机体 ID 捷径。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l26-pool",
      label: ["OXE 多机体池"],
      meta: "22 机体 · 1M+",
      kind: "input",
      x: 88,
      y: 180,
    },
    {
      id: "l26-align",
      label: ["7D 末端粗对齐"],
      meta: "不统一坐标轴",
      kind: "transform",
      x: 258,
      y: 180,
    },
    {
      id: "l26-mix",
      label: ["温度混合"],
      meta: "p ∝ n^α",
      kind: "transform",
      x: 430,
      y: 78,
    },
    {
      id: "l26-cap",
      label: ["每域条数上限"],
      meta: "min(n, C)",
      kind: "decision",
      x: 430,
      y: 282,
    },
    {
      id: "l26-batch",
      label: ["训练 batch"],
      meta: "小域频率可见",
      kind: "state",
      x: 620,
      y: 180,
    },
    {
      id: "l26-probe",
      label: ["负对照探针"],
      meta: "打乱指令 / 机体",
      kind: "decision",
      x: 820,
      y: 78,
    },
    {
      id: "l26-policy",
      label: ["政策或捷径"],
      meta: "去掉语言后准确率",
      kind: "output",
      x: 820,
      y: 282,
    },
  ],
  edges: [
    {
      id: "l26-e-pool-align",
      from: "l26-pool",
      to: "l26-align",
      label: "相机与动作异构",
      labelAt: { x: 172, y: 214 },
    },
    {
      id: "l26-e-align-mix",
      from: "l26-align",
      to: "l26-mix",
      label: "按域计数加权",
      via: [
        { x: 330, y: 180 },
        { x: 330, y: 78 },
      ],
      labelAt: { x: 292, y: 104 },
    },
    {
      id: "l26-e-align-cap",
      from: "l26-align",
      to: "l26-cap",
      label: "限制大域",
      via: [
        { x: 330, y: 180 },
        { x: 330, y: 282 },
      ],
      labelAt: { x: 286, y: 256 },
    },
    {
      id: "l26-e-mix-batch",
      from: "l26-mix",
      to: "l26-batch",
      label: "采样概率",
      labelAt: { x: 526, y: 104 },
    },
    {
      id: "l26-e-cap-batch",
      from: "l26-cap",
      to: "l26-batch",
      label: "截断后归一化",
      labelAt: { x: 534, y: 256 },
    },
    {
      id: "l26-e-batch-probe",
      from: "l26-batch",
      to: "l26-probe",
      label: "语言 / 机体 ID",
      via: [
        { x: 720, y: 180 },
        { x: 720, y: 78 },
      ],
      labelAt: { x: 678, y: 104 },
    },
    {
      id: "l26-e-batch-policy",
      from: "l26-batch",
      to: "l26-policy",
      label: "动作监督",
      via: [
        { x: 720, y: 180 },
        { x: 720, y: 282 },
      ],
      labelAt: { x: 668, y: 256 },
    },
    {
      id: "l26-e-probe-policy",
      from: "l26-probe",
      to: "l26-policy",
      label: "准确率是否掉",
    },
  ],
  steps: [
    {
      title: "先承认输入本来对不齐",
      description:
        "Open X-Embodiment 把 22 种机体、60 个数据集合成超过 100 万条轨迹。相机外参、末端坐标系和夹爪约定仍然各写各的。",
      focus: ["l26-pool", "l26-align", "l26-e-pool-align"],
    },
    {
      title: "用温度指数压住按条数采样",
      description:
        "域 d 的采样概率 p_d 正比于 n_d 的 α 次方。α=1 按原始条数，α=0 对各域均匀。RT-X 实际只在 9 种操作臂上训，不是 22 种全上。",
      focus: ["l26-mix", "l26-e-align-mix", "l26-batch"],
    },
    {
      title: "给大域加条数上限",
      description:
        "先把 n_d 截成 min(n_d, C) 再算权重。Octo 在 25 个数据集上按相对规模加权，并对更多样的集合加倍、对重复集合降权。",
      focus: ["l26-cap", "l26-e-align-cap", "l26-e-cap-batch", "l26-batch"],
    },
    {
      title: "打乱指令，看语言是不是真条件",
      description:
        "若去掉或打乱语言后准确率几乎不变，模型多半在用桌布、相机位或机体外观走捷径。Octo 预训练里只有 56% 样本带语言标注。",
      focus: ["l26-probe", "l26-e-batch-probe", "l26-policy"],
    },
    {
      title: "打乱机体标签，抓 ID 泄漏",
      description:
        "显式机体占位符或可反推的机器人 ID 会变成捷径。把机体标签打乱后泄漏政策必须掉点，语言政策在无泄漏时仍应跟着指令走。",
      focus: ["l26-probe", "l26-policy", "l26-e-probe-policy", "l26-e-batch-policy"],
    },
  ],
  facts: [
    "Open X-Embodiment 摘要：22 种机器人、21 家机构、527 种技能、160266 项任务、超过 100 万条真实轨迹、60 个既有数据集。",
    "RT-X 的机器人数据混合物来自 9 种操作臂，少于全集的 22 种机体。",
    "Octo 从约 150 万 OXE episode 中整理出 80 万条，覆盖 25 个数据集；OpenVLA 在 97 万条真实演示上训练。",
    "域采样公式为 p_d ∝ n_d^α；α=1 按条数，α=0 对域均匀。有效域数定义为 1/Σ p_d²。",
  ],
};
