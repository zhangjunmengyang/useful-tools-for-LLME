import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson38Diagram: LessonDiagram = {
  lessonId: "38",
  title: "同一组抓取：稀疏成功与接触 dense",
  summary:
    "示教只提供成功轨迹。同一组 rollout 先被稀疏成功或接触 dense 打分，再减组均值得到优势；全失败时稀疏方差为零，dense 仍能排序并更新。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l38-demo",
      label: ["示教成功"],
      meta: "模仿学不到失败",
      kind: "input",
      x: 88,
      y: 180,
    },
    {
      id: "l38-rollout",
      label: ["同题 G 条轨迹"],
      meta: "抓取 rollout 组",
      kind: "transform",
      x: 268,
      y: 180,
    },
    {
      id: "l38-sparse",
      label: ["稀疏成功"],
      meta: "I_success ∈ {0,1}",
      kind: "decision",
      x: 458,
      y: 88,
    },
    {
      id: "l38-dense",
      label: ["接触 dense"],
      meta: "接近 / 接触 / 力",
      kind: "decision",
      x: 458,
      y: 272,
    },
    {
      id: "l38-adv",
      label: ["组内优势"],
      meta: "Â = r − r̄",
      kind: "state",
      x: 648,
      y: 180,
    },
    {
      id: "l38-zero",
      label: ["零方差门"],
      meta: "全失败则 Â=0",
      kind: "decision",
      x: 788,
      y: 88,
      width: 140,
    },
    {
      id: "l38-update",
      label: ["政策更新"],
      meta: "非零 Â 才有梯度",
      kind: "output",
      x: 788,
      y: 272,
      width: 140,
    },
  ],
  edges: [
    {
      id: "l38-e-demo-rollout",
      from: "l38-demo",
      to: "l38-rollout",
      label: "冷启动",
      labelAt: { x: 176, y: 228 },
    },
    {
      id: "l38-e-rollout-sparse",
      from: "l38-rollout",
      to: "l38-sparse",
      label: "成功谓词",
      via: [
        { x: 348, y: 180 },
        { x: 348, y: 88 },
      ],
      labelAt: { x: 304, y: 124 },
    },
    {
      id: "l38-e-rollout-dense",
      from: "l38-rollout",
      to: "l38-dense",
      label: "接触程序",
      via: [
        { x: 348, y: 180 },
        { x: 348, y: 272 },
      ],
      labelAt: { x: 304, y: 236 },
    },
    {
      id: "l38-e-sparse-adv",
      from: "l38-sparse",
      to: "l38-adv",
      label: "0/1 组",
      labelAt: { x: 552, y: 64 },
    },
    {
      id: "l38-e-dense-adv",
      from: "l38-dense",
      to: "l38-adv",
      label: "可排序分",
      labelAt: { x: 552, y: 296 },
    },
    {
      id: "l38-e-adv-zero",
      from: "l38-adv",
      to: "l38-zero",
      label: "方差=0",
      via: [
        { x: 718, y: 180 },
        { x: 718, y: 88 },
      ],
      labelAt: { x: 738, y: 124 },
    },
    {
      id: "l38-e-adv-update",
      from: "l38-adv",
      to: "l38-update",
      label: "方差>0",
      via: [
        { x: 718, y: 180 },
        { x: 718, y: 272 },
      ],
      labelAt: { x: 738, y: 236 },
    },
  ],
  steps: [
    {
      title: "示教里没有失败",
      description:
        "行为克隆只拟合成功轨迹。杯子打翻、夹空、压溃这些状态不在教材里，模仿无法给它们打相对分。",
      focus: ["l38-demo", "l38-rollout", "l38-e-demo-rollout"],
    },
    {
      title: "同一组两条判分",
      description:
        "稀疏成功只看物体是否离桌并进盒。接触 dense 用接近、夹爪闭合和力超限给失败轨迹打不同分。",
      focus: [
        "l38-rollout",
        "l38-sparse",
        "l38-dense",
        "l38-e-rollout-sparse",
        "l38-e-rollout-dense",
      ],
    },
    {
      title: "组内减均值",
      description:
        "优势写成 Â_i = r_i − r̄。第 17 课的标准差归一化在方差为零时同样得到接近 0 的优势。本课不重推 clip。",
      focus: ["l38-sparse", "l38-dense", "l38-adv", "l38-e-sparse-adv", "l38-e-dense-adv"],
    },
    {
      title: "全失败批次",
      description:
        "四条轨迹都没放入时，稀疏奖励全是 0，方差为零，政策权重不变。dense 仍能排出擦边优于空抓。",
      focus: ["l38-adv", "l38-zero", "l38-update", "l38-e-adv-zero", "l38-e-adv-update"],
    },
    {
      title: "可验证而不是可讨好",
      description:
        "成功谓词、接触带和力阈值都是确定性程序。世界模型路线用生成帧与参考帧的 L1 / LPIPS 当 verified reward，仍要按原文写，不能口头换成任意视频生成。",
      focus: ["l38-sparse", "l38-dense", "l38-update"],
    },
  ],
  facts: [
    "ConRFT 在八项真机任务上，45–90 分钟在线微调后平均成功率 96.3%，相对监督基线提高 144%，回合长度缩短 1.9 倍；离线只用 20–30 条示教。",
    "SimpleVLA-RL 用轨迹级 0/1 结果奖励做 GRPO；每任务 1 条示教时，LIBERO-Long 从 17.3% 升到 91.7%。全失败或全成功组被动态采样丢掉。",
    "VLA-RFT 用世界模型生成未来帧，以 L1 与 LPIPS 相对参考轨迹作为 verified reward；约 400 步把 LIBERO 平均从 86.6% 升到 91.1%。",
    "SafeVLA 把安全谓词写成代价，在 CMDP 里用拉格朗日约束；相对 FLaRe 累计代价下降 83.58%，成功率 +3.85%。",
    "组内优势 Â_i = r_i − r̄。稀疏奖励在未成功组方差为零，不产生更新。",
  ],
};
