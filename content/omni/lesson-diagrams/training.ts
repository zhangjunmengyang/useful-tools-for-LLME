import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const trainingDiagrams: LessonDiagram[] = [
  {
    lessonId: "15",
    title: "Joint SFT 的训练时间线",
    summary:
      "任务桶先按有效 token 计量，再按固定阶段表进入同一组 connector 与 LoRA 参数。",
    nodes: [
      {
        id: "l15-task-buckets",
        label: ["六类任务桶"],
        meta: "text / image / audio / speech",
        kind: "input",
        x: 80,
        y: 180,
      },
      {
        id: "l15-token-sampler",
        label: ["token-balanced", "sampler"],
        meta: "non-padding token",
        kind: "transform",
        x: 245,
        y: 180,
      },
      {
        id: "l15-stage-plan",
        label: ["四阶段课程"],
        meta: "5% / 20% / 60% / 15%",
        kind: "state",
        x: 410,
        y: 180,
      },
      {
        id: "l15-shared-params",
        label: ["connectors +", "共享 LoRA"],
        meta: "三臂参数名相同",
        kind: "transform",
        x: 575,
        y: 95,
      },
      {
        id: "l15-replay",
        label: ["固定 replay", "manifest"],
        meta: "训练前写入 SHA-256",
        kind: "state",
        x: 575,
        y: 275,
      },
      {
        id: "l15-multi-loss",
        label: ["文本 CE +", "8 路 codec CE"],
        meta: "分目标记录 token 与梯度",
        kind: "transform",
        x: 740,
        y: 95,
      },
      {
        id: "l15-evaluation",
        label: ["固定 test +", "配对 bootstrap"],
        meta: "C−A 与 C−B",
        kind: "decision",
        x: 890,
        y: 95,
        width: 126,
      },
      {
        id: "l15-checkpoint",
        label: ["joint-sft-v1"],
        meta: "旧能力均不低于 97%",
        kind: "output",
        x: 890,
        y: 275,
        width: 126,
      },
    ],
    edges: [
      {
        id: "l15-e-buckets-sampler",
        from: "l15-task-buckets",
        to: "l15-token-sampler",
        label: "样本 + token 长度",
      },
      {
        id: "l15-e-sampler-plan",
        from: "l15-token-sampler",
        to: "l15-stage-plan",
        label: "累计任务 token",
      },
      {
        id: "l15-e-plan-params",
        from: "l15-stage-plan",
        to: "l15-shared-params",
        label: "Stage 1–3",
      },
      {
        id: "l15-e-plan-replay",
        from: "l15-stage-plan",
        to: "l15-replay",
        label: "Stage 4",
      },
      {
        id: "l15-e-replay-params",
        from: "l15-replay",
        to: "l15-shared-params",
        label: "只读样本集",
      },
      {
        id: "l15-e-params-loss",
        from: "l15-shared-params",
        to: "l15-multi-loss",
        label: "同一可训练范围",
      },
      {
        id: "l15-e-loss-eval",
        from: "l15-multi-loss",
        to: "l15-evaluation",
        label: "阶段 checkpoint",
      },
      {
        id: "l15-e-eval-output",
        from: "l15-evaluation",
        to: "l15-checkpoint",
        label: "全部门槛通过",
      },
    ],
    steps: [
      {
        title: "按有效 token 计量",
        description:
          "采样器同时记录样本比例和 non-padding token 比例，避免长音频和 codec target 隐式占据更多更新。",
        focus: [
          "l15-task-buckets",
          "l15-token-sampler",
          "l15-e-buckets-sampler",
        ],
      },
      {
        title: "冻结阶段预算",
        description:
          "A、B、C 三臂使用相同累计任务 token，只允许任务进入训练的时间顺序不同。",
        focus: [
          "l15-token-sampler",
          "l15-stage-plan",
          "l15-e-sampler-plan",
        ],
      },
      {
        title: "更新同一组参数",
        description:
          "vision/audio connector 与 backbone、Talker LoRA 在各阶段保持相同参数名和参数量。",
        focus: [
          "l15-shared-params",
          "l15-multi-loss",
          "l15-e-params-loss",
        ],
      },
      {
        title: "执行预注册 replay",
        description:
          "C 臂最后 15% token 只读取训练前冻结并登记 hash 的 replay manifest。",
        focus: [
          "l15-stage-plan",
          "l15-replay",
          "l15-e-plan-replay",
          "l15-e-replay-params",
        ],
      },
      {
        title: "用固定测试判定",
        description:
          "三个 seed 完成后计算 C−A、C−B，并同时检查每项旧能力是否保留。",
        focus: [
          "l15-multi-loss",
          "l15-evaluation",
          "l15-checkpoint",
          "l15-e-loss-eval",
          "l15-e-eval-output",
        ],
      },
    ],
    facts: [
      "sample-balanced 不等于 token-balanced；优化份额按实际进入 loss 的 token 计算。",
      "文本 label 为 [T−1]，音频 label 为 [8,T−1]，每个 codebook 的 stop token 都参与监督。",
      "三臂的 LoRA 配置、optimizer 参数名和可训练参数总数必须完全相同。",
      "replay manifest 在第一个 optimizer step 前冻结，不能根据 Stage 3 结果换样本。",
      "正向结论要求 C−A、C−B 同时超过预注册效应量，且旧能力不低于起点 97%。",
    ],
  },
  {
    lessonId: "16",
    title: "mDPO 的三个训练信号",
    summary:
      "同一 chosen 回答分别比较回答质量、原图与低信息 crop，并由冻结 reference 提供相对 reward。",
    nodes: [
      {
        id: "l16-pair",
        label: ["图像偏好对"],
        meta: "prompt / chosen / rejected",
        kind: "input",
        x: 80,
        y: 180,
      },
      {
        id: "l16-policy",
        label: ["可训练 policy"],
        meta: "三路 sequence log-prob",
        kind: "transform",
        x: 260,
        y: 85,
      },
      {
        id: "l16-crop",
        label: ["确定性 Crop", "0–20%"],
        meta: "box、seed、hash 固定",
        kind: "transform",
        x: 260,
        y: 275,
      },
      {
        id: "l16-reference",
        label: ["冻结 reference"],
        meta: "step-0 policy 副本",
        kind: "state",
        x: 455,
        y: 275,
      },
      {
        id: "l16-rewards",
        label: ["reference-relative", "reward"],
        meta: "β(log πθ − log πref)",
        kind: "transform",
        x: 455,
        y: 85,
      },
      {
        id: "l16-losses",
        label: ["DPO + CoPO", "+ AncPO"],
        meta: "β=0.1，三项等权",
        kind: "transform",
        x: 650,
        y: 85,
      },
      {
        id: "l16-update",
        label: ["C0 policy", "checkpoint"],
        meta: "仅 image-only 主实验",
        kind: "output",
        x: 850,
        y: 85,
      },
    ],
    edges: [
      {
        id: "l16-e-pair-policy",
        from: "l16-pair",
        to: "l16-policy",
        label: "chosen / rejected",
      },
      {
        id: "l16-e-pair-crop",
        from: "l16-pair",
        to: "l16-crop",
        label: "原图",
      },
      {
        id: "l16-e-crop-policy",
        from: "l16-crop",
        to: "l16-policy",
        label: "同一 chosen",
      },
      {
        id: "l16-e-pair-reference",
        from: "l16-pair",
        to: "l16-reference",
        label: "同一 token / mask",
        via: [
          { x: 120, y: 330 },
          { x: 455, y: 330 },
        ],
      },
      {
        id: "l16-e-crop-reference",
        from: "l16-crop",
        to: "l16-reference",
        label: "crop + 同一 chosen",
      },
      {
        id: "l16-e-policy-rewards",
        from: "l16-policy",
        to: "l16-rewards",
        label: "policy log-prob",
      },
      {
        id: "l16-e-reference-rewards",
        from: "l16-reference",
        to: "l16-rewards",
        label: "reference log-prob",
      },
      {
        id: "l16-e-rewards-losses",
        from: "l16-rewards",
        to: "l16-losses",
        label: "三个 margin",
      },
      {
        id: "l16-e-losses-update",
        from: "l16-losses",
        to: "l16-update",
        label: "只更新 policy",
      },
    ],
    steps: [
      {
        title: "固定偏好对与 crop",
        description:
          "C0 保持问题和 chosen 文本不变，只把原图换成由 pair ID 确定的低信息 crop。",
        focus: [
          "l16-pair",
          "l16-crop",
          "l16-e-pair-crop",
        ],
      },
      {
        title: "计算两套 log-prob",
        description:
          "policy 和 step-0 reference 使用相同 tokenizer、图像处理、response mask 与长度约定。",
        focus: [
          "l16-policy",
          "l16-reference",
          "l16-e-pair-reference",
          "l16-e-crop-reference",
          "l16-e-policy-rewards",
          "l16-e-reference-rewards",
        ],
      },
      {
        title: "形成相对 reward",
        description:
          "每一路 reward 都是 policy 相对 reference 的 log-prob 变化，而不是原始概率。",
        focus: [
          "l16-policy",
          "l16-reference",
          "l16-rewards",
          "l16-e-policy-rewards",
          "l16-e-reference-rewards",
        ],
      },
      {
        title: "合并三个目标",
        description:
          "response DPO 比 chosen/rejected，CoPO 比原图/crop，AncPO 约束原图 chosen reward 不低于 0。",
        focus: [
          "l16-rewards",
          "l16-losses",
          "l16-e-rewards-losses",
        ],
      },
      {
        title: "只发布 C0 结论",
        description:
          "错图、移除图像、音频和视频属于 C1 扩展，不能回写 image-only mDPO 主表。",
        focus: [
          "l16-losses",
          "l16-update",
          "l16-e-losses-update",
        ],
      },
    ],
    facts: [
      "response DPO 改变回答，CoPO 保持 chosen 文本不变并只改变图像条件。",
      "C0 固定 β=0.1、anchor=0、sequence-sum log-prob 和三项等权。",
      "crop 必须预先生成并保存 box、面积比例、seed、实现版本及输入输出 hash。",
      "reference 全程无梯度，训练前后文件 hash 必须相同。",
      "pair margin 上升不能替代 held-out fresh generation 的 grounded 正确率。",
    ],
  },
  {
    lessonId: "17",
    title: "一轮 GRPO / RLVR 更新",
    summary:
      "旧 policy 生成同题多回答，确定性 verifier 给分，组内 advantage 再作用到有效回答 token。",
    nodes: [
      {
        id: "l17-prompts",
        label: ["训练 prompts"],
        meta: "题目、媒体、答案规格",
        kind: "input",
        x: 80,
        y: 180,
      },
      {
        id: "l17-old-policy",
        label: ["policy_old"],
        meta: "本轮采样快照",
        kind: "state",
        x: 250,
        y: 80,
      },
      {
        id: "l17-rollouts",
        label: ["每题 G 个", "新回答"],
        meta: "保存 token 与 old_logp",
        kind: "transform",
        x: 250,
        y: 180,
      },
      {
        id: "l17-verifier",
        label: ["确定性 verifier"],
        meta: "outcome / format / evidence",
        kind: "transform",
        x: 430,
        y: 180,
      },
      {
        id: "l17-advantage",
        label: ["组内 reward", "标准化"],
        meta: "shape [B,G]",
        kind: "transform",
        x: 600,
        y: 180,
      },
      {
        id: "l17-update",
        label: ["clipped policy", "update"],
        meta: "只覆盖有效 response token",
        kind: "transform",
        x: 760,
        y: 180,
      },
      {
        id: "l17-reference",
        label: ["冻结 reference"],
        meta: "KL penalty",
        kind: "state",
        x: 760,
        y: 290,
      },
      {
        id: "l17-sync",
        label: ["新 policy 快照"],
        meta: "下一轮 policy_old",
        kind: "output",
        x: 895,
        y: 80,
        width: 118,
      },
    ],
    edges: [
      {
        id: "l17-e-prompts-rollouts",
        from: "l17-prompts",
        to: "l17-rollouts",
        label: "同一 prompt",
      },
      {
        id: "l17-e-old-rollouts",
        from: "l17-old-policy",
        to: "l17-rollouts",
        label: "采样参数固定",
      },
      {
        id: "l17-e-rollouts-verifier",
        from: "l17-rollouts",
        to: "l17-verifier",
        label: "回答 + 证据",
      },
      {
        id: "l17-e-verifier-advantage",
        from: "l17-verifier",
        to: "l17-advantage",
        label: "标量 reward",
      },
      {
        id: "l17-e-advantage-update",
        from: "l17-advantage",
        to: "l17-update",
        label: "广播到 token",
      },
      {
        id: "l17-e-reference-update",
        from: "l17-reference",
        to: "l17-update",
        label: "KL",
      },
      {
        id: "l17-e-update-sync",
        from: "l17-update",
        to: "l17-sync",
        label: "optimizer step",
      },
      {
        id: "l17-e-sync-old",
        from: "l17-sync",
        to: "l17-old-policy",
        label: "完成 minibatch 后同步",
        via: [
          { x: 895, y: 22 },
          { x: 250, y: 22 },
        ],
      },
    ],
    steps: [
      {
        title: "用旧快照生成回答",
        description:
          "同一 prompt 由 policy_old 生成 G 个回答，并保存采样时的 token 级 old_logp。",
        focus: [
          "l17-prompts",
          "l17-old-policy",
          "l17-rollouts",
          "l17-e-prompts-rollouts",
          "l17-e-old-rollouts",
        ],
      },
      {
        title: "先完成 verifier",
        description:
          "verifier 异常或超时重试一次；仍失败时丢弃整个 group，不计算 advantage，也不更新参数。",
        focus: [
          "l17-rollouts",
          "l17-verifier",
          "l17-e-rollouts-verifier",
        ],
      },
      {
        title: "计算组内相对优势",
        description:
          "每题的 reward 独立标准化；组内 reward 全相同时 advantage 接近 0。",
        focus: [
          "l17-verifier",
          "l17-advantage",
          "l17-e-verifier-advantage",
        ],
      },
      {
        title: "执行裁剪更新",
        description:
          "response-level advantage 广播到有效生成 token，prompt 和 padding 的 policy loss 为 0。",
        focus: [
          "l17-advantage",
          "l17-update",
          "l17-reference",
          "l17-e-advantage-update",
          "l17-e-reference-update",
        ],
      },
      {
        title: "同步下一轮权重",
        description:
          "全部 minibatch 更新完成后才刷新 policy_old；reference 始终保持冻结。",
        focus: [
          "l17-update",
          "l17-sync",
          "l17-old-policy",
          "l17-e-update-sync",
          "l17-e-sync-old",
        ],
      },
    ],
    facts: [
      "reward 的 shape 是 [B,G]，每个回答的标量 advantage 只广播到该回答的有效生成 token。",
      "zero-variance group 消耗采样算力，但几乎不提供 policy 更新信号。",
      "verifier exception 与模型答案错误是两种状态；前者不能伪装成零分样本。",
      "B、C 两臂必须使用相同 prompt 顺序、group size、生成 token 和 policy update token。",
      "训练 reward 上升必须由 hidden test、备用 verifier 和逐例审计共同确认。",
    ],
  },
  {
    lessonId: "18",
    title: "Nemotron Omni 的 LoRA 更新边界",
    summary:
      "CORD 样本经 processor 进入冻结的 30B-A3B 基座，只有明确命中的低秩矩阵参与反向。",
    viewBox: "0 0 1040 360",
    nodes: [
      {
        id: "l18-cord",
        label: ["CORD-v2"],
        meta: "image + structured target",
        kind: "input",
        x: 80,
        y: 180,
      },
      {
        id: "l18-processor",
        label: ["processor +", "collator"],
        meta: "image_flags / labels",
        kind: "transform",
        x: 240,
        y: 180,
      },
      {
        id: "l18-base",
        label: ["冻结 BF16 base"],
        meta: "Nemotron 30B-A3B",
        kind: "state",
        x: 420,
        y: 75,
      },
      {
        id: "l18-lora",
        label: ["LM linear", "LoRA r=64"],
        meta: "约 55M 可训练参数",
        kind: "transform",
        x: 420,
        y: 285,
      },
      {
        id: "l18-forward",
        label: ["8-rank forward"],
        meta: "正式 H100 lane：EP=8",
        kind: "transform",
        x: 590,
        y: 180,
      },
      {
        id: "l18-loss",
        label: ["assistant-only", "token loss"],
        meta: "base 无梯度",
        kind: "transform",
        x: 760,
        y: 180,
      },
      {
        id: "l18-adapter",
        label: ["完整训练状态"],
        meta: "adapter / optimizer / scheduler / RNG",
        kind: "output",
        x: 950,
        y: 75,
        width: 118,
      },
      {
        id: "l18-merged",
        label: ["adapter 或", "merged HF"],
        meta: "推理产物 / logits parity",
        kind: "output",
        x: 950,
        y: 285,
        width: 118,
      },
    ],
    edges: [
      {
        id: "l18-e-cord-processor",
        from: "l18-cord",
        to: "l18-processor",
        label: "conversation + image",
      },
      {
        id: "l18-e-processor-forward",
        from: "l18-processor",
        to: "l18-forward",
        label: "input_ids / labels",
      },
      {
        id: "l18-e-base-forward",
        from: "l18-base",
        to: "l18-forward",
        label: "冻结权重",
      },
      {
        id: "l18-e-lora-forward",
        from: "l18-lora",
        to: "l18-forward",
        label: "ΔW=(α/r)BA",
      },
      {
        id: "l18-e-forward-loss",
        from: "l18-forward",
        to: "l18-loss",
        label: "结构化 target",
      },
      {
        id: "l18-e-loss-lora",
        from: "l18-loss",
        to: "l18-lora",
        label: "只更新 A、B",
        via: [
          { x: 760, y: 338 },
          { x: 500, y: 338 },
        ],
      },
      {
        id: "l18-e-loss-adapter",
        from: "l18-loss",
        to: "l18-adapter",
        label: "保存恢复状态",
      },
      {
        id: "l18-e-adapter-merged",
        from: "l18-adapter",
        to: "l18-merged",
        label: "导出 adapter → merge",
      },
      {
        id: "l18-e-base-merged",
        from: "l18-base",
        to: "l18-merged",
        label: "W + ΔW",
        via: [
          { x: 420, y: 14 },
          { x: 952, y: 14 },
          { x: 952, y: 285 },
        ],
        labelAt: { x: 690, y: 12 },
      },
    ],
    steps: [
      {
        title: "先验证数据 contract",
        description:
          "processor 把图像占位符、image_flags 和 assistant-only labels 组成同一批次事实。",
        focus: [
          "l18-cord",
          "l18-processor",
          "l18-e-cord-processor",
        ],
      },
      {
        title: "冻结基座并挂载 LoRA",
        description:
          "vision/audio tower、lm_head 与基础语言权重保持冻结，只有 allowlist 内的 LM linear LoRA 可训练。",
        focus: [
          "l18-base",
          "l18-lora",
          "l18-forward",
          "l18-e-base-forward",
          "l18-e-lora-forward",
        ],
      },
      {
        title: "正式 lane 使用 EP=8",
        description:
          "官方硬件条件是 8×H100 80GB；CPU companion 只验证 LoRA 数学、merge 和 label contract。",
        focus: [
          "l18-forward",
          "l18-loss",
          "l18-e-processor-forward",
          "l18-e-forward-loss",
        ],
      },
      {
        title: "只更新低秩矩阵",
        description:
          "反向后基础矩阵 W 没有梯度，A、B 有梯度；恢复训练还必须保存 optimizer、scheduler、dataloader 与 RNG 状态。",
        focus: [
          "l18-lora",
          "l18-loss",
          "l18-e-loss-lora",
          "l18-e-loss-adapter",
        ],
      },
      {
        title: "验证合并产物",
        description:
          "adapter-only 与 merged HF 是推理产物，不是完整恢复点；二者在固定输入上的 logits 必须落在预注册数值容差内。",
        focus: [
          "l18-base",
          "l18-adapter",
          "l18-merged",
          "l18-e-adapter-merged",
          "l18-e-base-merged",
        ],
      },
    ],
    facts: [
      "本课必做的是官方 CORD-v2 LoRA 示例；full SFT 只是满足硬件门槛后的可选参考。",
      "一个线性层新增的 LoRA 参数量是 r(d_in+d_out)，基础矩阵 W 保持冻结。",
      "官方示例固定 r=64、alpha=128，vision/audio tower 与 lm_head 不进入 LoRA target。",
      "validation 只选 checkpoint；冻结后 base 与 LoRA 各在 test 上运行一次。",
      "模型接受 text、image、video、audio 输入，但输出类型是 text。",
    ],
  },
  {
    lessonId: "19",
    title: "Thinker、因果桥与 Talker 的双工链路",
    summary:
      "Thinker token 与 MiniMind clock 先在 UTF-8 字节轴上对齐，scheduler 再根据真实监听事件控制 Talker。",
    viewBox: "0 0 1040 360",
    nodes: [
      {
        id: "l19-request",
        label: ["多模态请求"],
        meta: "text / image / video / audio",
        kind: "input",
        x: 80,
        y: 180,
      },
      {
        id: "l19-thinker",
        label: ["Nemotron", "Thinker"],
        meta: "text + selected hidden states",
        kind: "transform",
        x: 250,
        y: 80,
      },
      {
        id: "l19-alignment",
        label: ["UTF-8 byte", "alignment"],
        meta: "prefix-monotone",
        kind: "transform",
        x: 425,
        y: 80,
      },
      {
        id: "l19-clock",
        label: ["MiniMind causal", "clock bridge"],
        meta: "K,V ≤ text_token_index[s]",
        kind: "transform",
        x: 600,
        y: 80,
      },
      {
        id: "l19-talker",
        label: ["冻结 Talker", "+ Mimi"],
        meta: "8 路 codec 自回归",
        kind: "transform",
        x: 775,
        y: 80,
      },
      {
        id: "l19-pcm",
        label: ["可播放 PCM"],
        meta: "audio frame + DAC timestamp",
        kind: "output",
        x: 960,
        y: 80,
        width: 126,
      },
      {
        id: "l19-listener",
        label: ["AEC + VAD +", "streaming ASR"],
        meta: "20ms frame / 480ms chunk",
        kind: "transform",
        x: 250,
        y: 280,
      },
      {
        id: "l19-scheduler",
        label: ["session", "scheduler"],
        meta: "continue / pause / replan",
        kind: "decision",
        x: 600,
        y: 280,
      },
    ],
    edges: [
      {
        id: "l19-e-request-thinker",
        from: "l19-request",
        to: "l19-thinker",
        label: "已提交用户语义",
      },
      {
        id: "l19-e-thinker-alignment",
        from: "l19-thinker",
        to: "l19-alignment",
        label: "answer token + hidden",
      },
      {
        id: "l19-e-alignment-clock",
        from: "l19-alignment",
        to: "l19-clock",
        label: "text_token_index",
      },
      {
        id: "l19-e-clock-talker",
        from: "l19-clock",
        to: "l19-talker",
        label: "clock_states [B,S,Hmm]",
      },
      {
        id: "l19-e-talker-pcm",
        from: "l19-talker",
        to: "l19-pcm",
        label: "Mimi codes → PCM",
      },
      {
        id: "l19-e-request-listener",
        from: "l19-request",
        to: "l19-listener",
        label: "持续用户音频",
      },
      {
        id: "l19-e-listener-scheduler",
        from: "l19-listener",
        to: "l19-scheduler",
        label: "candidate / partial / endpoint",
      },
      {
        id: "l19-e-scheduler-talker",
        from: "l19-scheduler",
        to: "l19-talker",
        label: "hold / pause / resume",
        via: [{ x: 775, y: 280 }],
      },
      {
        id: "l19-e-pcm-listener",
        from: "l19-pcm",
        to: "l19-listener",
        label: "AEC playback reference",
        via: [
          { x: 960, y: 345 },
          { x: 250, y: 345 },
        ],
      },
    ],
    steps: [
      {
        title: "Thinker 产生回答条件",
        description:
          "Nemotron 输出回答文本与候选层 hidden state；默认缓存实验不把缓存释放时间当在线 TTFT。",
        focus: [
          "l19-request",
          "l19-thinker",
          "l19-e-request-thinker",
        ],
      },
      {
        title: "在字节轴上对齐",
        description:
          "两套 tokenizer 都映射到原始 UTF-8 byte span；已提交 token 前缀被改写时立即退回 EOS-buffered 19A。",
        focus: [
          "l19-thinker",
          "l19-alignment",
          "l19-e-thinker-alignment",
        ],
      },
      {
        title: "构造因果 clock state",
        description:
          "第 s 个 MiniMind 位置只 cross-attend 到 text_token_index[s] 及以前的 Nemotron state。",
        focus: [
          "l19-alignment",
          "l19-clock",
          "l19-e-alignment-clock",
        ],
      },
      {
        title: "Talker 生成 8 路 codec",
        description:
          "bridge state、前序 Mimi code 和固定 speaker/reference condition 共同进入冻结 Talker。",
        focus: [
          "l19-clock",
          "l19-talker",
          "l19-pcm",
          "l19-e-clock-talker",
          "l19-e-talker-pcm",
        ],
      },
      {
        title: "播放期间持续监听",
        description:
          "AEC 使用真实 playback reference；VAD frame 与 streaming ASR chunk 在 Talker 播放期间持续消费。",
        focus: [
          "l19-request",
          "l19-listener",
          "l19-pcm",
          "l19-e-request-listener",
          "l19-e-pcm-listener",
        ],
      },
      {
        title: "由 scheduler 执行动作",
        description:
          "快速分类器决定是否 FAST_PAUSE，完整 ASR 语义再决定 resume 或 cancel + re-prefill + replan。",
        focus: [
          "l19-listener",
          "l19-scheduler",
          "l19-talker",
          "l19-e-listener-scheduler",
          "l19-e-scheduler-talker",
        ],
      },
    ],
    facts: [
      "19A 使用完整冻结 MiniMind Thinker 把确定文本展开成 Talker 所需的逐位置 clock state。",
      "跨 tokenizer 对齐使用 UTF-8 byte span，不能直接按 token 序号、字符比例或 DTW 对齐。",
      "主矩阵只训练 bridge.*；Nemotron、MiniMind Thinker、Talker、speaker 与 codec projection 全部冻结。",
      "默认课内结果是 duplex replay：真实 listener/Talker 加缓存 Thinker 事件，不报告在线 TTFT/TTFA。",
      "只有在线 Thinker 完成 cancel + re-prefill 并保留完整 trace，才能报告 system_full_duplex=true。",
    ],
  },
  {
    lessonId: "20",
    title: "理解路径与生成路径如何共享 core",
    summary:
      "两条路径复用 MiniMind block 权重，但输入 token、attention mask 和训练目标彼此独立。",
    nodes: [
      {
        id: "l20-image",
        label: ["理解输入图像"],
        meta: "256×256 / frozen split",
        kind: "input",
        x: 80,
        y: 85,
      },
      {
        id: "l20-semantic",
        label: ["冻结 SigLIP2"],
        meta: "[B,64,768]",
        kind: "transform",
        x: 255,
        y: 85,
      },
      {
        id: "l20-adapter",
        label: ["理解 adapter"],
        meta: "3 blocks → 32 prefix tokens",
        kind: "transform",
        x: 430,
        y: 85,
      },
      {
        id: "l20-generation-input",
        label: ["生成输入"],
        meta: "prompt + 256 noisy latents zₜ + t",
        kind: "input",
        x: 80,
        y: 280,
      },
      {
        id: "l20-x0-target",
        label: ["VAE x₀ target"],
        meta: "仅训练可见 / 推理时不可用",
        kind: "state",
        x: 430,
        y: 280,
      },
      {
        id: "l20-core",
        label: ["共享 MiniMind", "block 权重"],
        meta: "route-specific tokens + mask",
        kind: "state",
        x: 625,
        y: 180,
      },
      {
        id: "l20-language-output",
        label: ["language head", "→ 文本答案"],
        meta: "assistant-only CE",
        kind: "output",
        x: 865,
        y: 85,
      },
      {
        id: "l20-flow-output",
        label: ["flow head → VAE", "→ 生成图像"],
        meta: "vθ → z₀ → decode",
        kind: "output",
        x: 865,
        y: 280,
      },
    ],
    edges: [
      {
        id: "l20-e-image-semantic",
        from: "l20-image",
        to: "l20-semantic",
        label: "semantic path",
        labelAt: { x: 168, y: 30 },
      },
      {
        id: "l20-e-semantic-adapter",
        from: "l20-semantic",
        to: "l20-adapter",
        label: "64 semantic tokens",
        labelAt: { x: 342, y: 145 },
      },
      {
        id: "l20-e-adapter-core",
        from: "l20-adapter",
        to: "l20-core",
        label: "32-token U prefix",
      },
      {
        id: "l20-e-generation-core",
        from: "l20-generation-input",
        to: "l20-core",
        label: "G route：不经过理解 adapter",
      },
      {
        id: "l20-e-image-x0",
        from: "l20-image",
        to: "l20-x0-target",
        label: "训练时 VAE encode",
        via: [
          { x: 80, y: 210 },
          { x: 430, y: 210 },
        ],
      },
      {
        id: "l20-e-core-language",
        from: "l20-core",
        to: "l20-language-output",
        label: "UNDERSTAND mask",
      },
      {
        id: "l20-e-core-flow",
        from: "l20-core",
        to: "l20-flow-output",
        label: "GENERATE mask",
      },
      {
        id: "l20-e-x0-flow",
        from: "l20-x0-target",
        to: "l20-flow-output",
        label: "target ε − x₀",
      },
    ],
    steps: [
      {
        title: "构造理解输入",
        description:
          "理解路径只读取冻结 SigLIP2 的 64 个 semantic token；这里没有 VAE latent。",
        focus: [
          "l20-image",
          "l20-semantic",
          "l20-adapter",
          "l20-e-image-semantic",
          "l20-e-semantic-adapter",
        ],
      },
      {
        title: "压缩成理解前缀",
        description:
          "3-block adapter 只属于 UNDERSTAND 路径，把 64 个语义 token 压成 32 个 visual prefix token。",
        focus: [
          "l20-semantic",
          "l20-adapter",
          "l20-core",
          "l20-e-semantic-adapter",
          "l20-e-adapter-core",
        ],
      },
      {
        title: "单独构造生成输入",
        description:
          "GENERATE 路径把 prompt、256 个 noisy-latent token zₜ 和时间 t 直接交给 core，不经过理解 adapter。",
        focus: [
          "l20-generation-input",
          "l20-core",
          "l20-e-generation-core",
        ],
      },
      {
        title: "只在训练时读取 x₀",
        description:
          "训练图像经 VAE 得到 x₀，用它构造 velocity target ε−x₀；推理从噪声出发，不读取目标图像或 x₀。",
        focus: [
          "l20-image",
          "l20-x0-target",
          "l20-flow-output",
          "l20-e-image-x0",
          "l20-e-x0-flow",
        ],
      },
      {
        title: "用各自的 head 输出",
        description:
          "理解用 language head 自回归生成文本；生成用 flow head 预测 velocity，再把反向积分得到的 z₀ 交给 VAE 解码。",
        focus: [
          "l20-core",
          "l20-language-output",
          "l20-flow-output",
          "l20-e-core-language",
          "l20-e-core-flow",
        ],
      },
    ],
    facts: [
      "理解前缀固定为 32 token；生成输入固定为 256 个 noisy-latent token，二者不是同一种 semantic token。",
      "3-block adapter 只处理理解输入，生成 latent 不得先经过这条压缩路径。",
      "VAE x₀ 是 flow matching 的训练 target；推理时不能读取目标图像或 x₀。",
      "UNDERSTAND 中视觉 prefix 双向、文本因果；GENERATE 中 latent 双向且可读取全部 prompt。",
      "flow 目标是 ε−x0，采样从 t=1 噪声反向积分到 t=0 数据。",
      "只有联合训练、两条 route 的 mask 测试和两类 held-out 评测都通过，才能称为共享 checkpoint。",
    ],
  },
];
