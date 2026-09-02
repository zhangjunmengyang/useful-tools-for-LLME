export interface LessonPractice {
  lessonId: string;
  purpose: string;
  proves: readonly string[];
  doesNotProve: string;
  sourceFile: string;
}

export const practiceRepositoryUrl =
  "content/cl/experiments";

const practices = [
  {
    lessonId: "01",
    purpose: "二维线性分类器上顺序学习任务 A 再学任务 B，测量 A 的准确率下降。",
    proves: [
      "任务 A 先能学到 >0.90",
      "接着训 B 后 A 下降超过 0.25 并落到 0.70 以下",
    ],
    doesNotProve: "不是 Split MNIST 或论文表格里的绝对数字。",
    sourceFile: "lesson_01.py",
  },
  {
    lessonId: "02",
    purpose: "冻骨干只训头，对照全网接着训，把两点画在稳定性-可塑性平面上。",
    proves: [
      "冻骨干的旧任务高于 naive 至少 0.05",
      "冻骨干的新任务低于 naive 至少 0.10",
      "冻住的骨干位移为 0",
    ],
    doesNotProve: "不是 EWC 的正式复现，那是第 05 课。",
    sourceFile: "lesson_02.py",
  },
  {
    lessonId: "03",
    purpose: "构造「只会最后任务」的结果矩阵，对比平均准确率和 BWT。",
    proves: [
      "最后任务专家的 ACC 可以 >0.70 同时 BWT < -0.40",
      "两张 ACC 相同的矩阵 BWT 可以差 0.35 以上",
    ],
    doesNotProve: "不是 Avalanche 日志解析器本身正确。",
    sourceFile: "lesson_03.py",
  },
  {
    lessonId: "04",
    purpose: "20 条座位事实，比较有名录、检索、改权重在撤掉名录之后还能不能答。",
    proves: [
      "有上下文时 prompt 与 RAG 都能满分",
      "撤掉名录后 prompt 与 RAG 失败，权重仍 >0.95",
    ],
    doesNotProve: "不是 SmolLM 微调分数；权重列是线性联想记忆。",
    sourceFile: "lesson_04.py",
  },
  {
    lessonId: "05",
    purpose: "标签翻转任务上扫描 EWC 的 λ，验证 λ=0 等于 naive、λ 过大则新任务学不会。",
    proves: [
      "λ=0 与 naive 权重 L2 为 0",
      "大 λ 时旧任务 >0.85 且新任务 <0.30",
    ],
    doesNotProve: "不是 Kirkpatrick 文中的 Atari 分数。",
    sourceFile: "lesson_05.py",
  },
  {
    lessonId: "06",
    purpose: "同一对任务上对比无回放和缓冲回放的遗忘。",
    proves: [
      "无回放的旧任务下降比有回放至少多 0.08",
      "有回放时新旧任务都 >0.80",
    ],
    doesNotProve: "不是 CIFAR-10 上 DER++ 的论文表，那是复现 #1。",
    sourceFile: "lesson_06.py",
  },
  {
    lessonId: "07",
    purpose: "按幅度剪枝并上锁，检查任务 1 的知识是否住在占用掩码里。",
    proves: [
      "占用比例接近一半且占用权重冻结",
      "清占用则任务 1 塌，清空闲则任务 1 还在",
    ],
    doesNotProve: "不是 DualPrompt / L2P 的视觉基准分数。",
    sourceFile: "lesson_07.py",
  },
  {
    lessonId: "08",
    purpose: "把违反旧任务约束的更新投影到半平面，检查投影后点积。",
    proves: [
      "违规更新能被点积检查出来",
      "投影后与法向点积 ≤ 1e-10",
      "A-GEM 对梯度的投影与本课符号约定重合",
    ],
    doesNotProve: "不是 GDumb 对 A-GEM 的分类分数，那是复现 #2。",
    sourceFile: "lesson_08.py",
  },
  {
    lessonId: "09",
    purpose: "预训练后接到窄领域，对比纯领域续训和混 30% 通用数据。",
    proves: [
      "纯领域会让通用指标下降超过 0.15",
      "混入通用数据后下降更少，领域仍能学会",
    ],
    doesNotProve: "不是 135M 语言模型续预训练。",
    sourceFile: "lesson_09.py",
  },
  {
    lessonId: "10",
    purpose: "四个冲突方向顺序训练，填 4×4 准确率矩阵并算 BWT。",
    proves: [
      "对角线全 >0.88",
      "下三角出现遗忘，BWT 为负",
      "最后一行峰值在最后一列",
    ],
    doesNotProve: "不是 TRACE 八任务指令基准。",
    sourceFile: "lesson_10.py",
  },
  {
    lessonId: "11",
    purpose: "两个 rank-1 LoRA，对比 naive 夹角和正交投影后的夹角。",
    proves: [
      "naive 余弦明显不为 0",
      "正交后余弦接近 0，且叠在一起时任务 1 保持更好",
    ],
    doesNotProve: "不是 T5-large 官方 O-LoRA 表，那是复现 #3 的仓库档。",
    sourceFile: "lesson_11.py",
  },
  {
    lessonId: "12",
    purpose: "两个正交任务向量相加，对比单任务模型和负向量。",
    proves: [
      "相加后两任务都 >0.85",
      "好于拿单任务模型去测另一任务",
      "负任务向量会伤害任务 1",
    ],
    doesNotProve: "不是 mergekit 在真实语言模型上的排行榜分数。",
    sourceFile: "lesson_12.py",
  },
  {
    lessonId: "13",
    purpose: "覆盖、追加、图跳转、工作记忆清空，四条写入规则对照。",
    proves: [
      "覆盖读回新座位并丢掉旧值",
      "追加同时保留两个值",
      "知识图能两跳，词袋检索不能",
      "清空工作记忆后语义名录仍在",
    ],
    doesNotProve: "不是 Letta / Mem0 的线上召回率。",
    sourceFile: "lesson_13.py",
  },
  {
    lessonId: "14",
    purpose: "对线性联想记忆做 rank-1 编辑，对照整表重写。",
    proves: [
      "目标键靠近新值",
      "目标位移明显大于邻居",
      "忘掉目标时邻居仍在，naive 重写则会伤邻居",
    ],
    doesNotProve: "不是 ROME 在 GPT-2 XL + CounterFact 上的四指标表。",
    sourceFile: "lesson_14.py",
  },
  {
    lessonId: "15",
    purpose: "长序列随机线性分类，对比 SGD 和按饱和度重初始化。",
    proves: [
      "SGD 后期增益下降、死神经元上升、后期学习速度变慢",
      "重初始化后后期增益更高、死神经元更少",
    ],
    doesNotProve: "不是 Nature 文的 ImageNet 800 任务。",
    sourceFile: "lesson_15.py",
  },
  {
    lessonId: "16",
    purpose: "四类经验 × 四种写入的符号化通过矩阵。",
    proves: [
      "至少一格记忆过、权重不过（事实/文档）",
      "至少一格反过来（流程/规则）",
      "撤掉上下文后 RAG 为 0",
      "编辑只改目标事实",
    ],
    doesNotProve: "不是 40 题真模型实战矩阵。",
    sourceFile: "lesson_16.py",
  },
  {
    lessonId: "17",
    purpose: "TTT-Linear 在短序列上做内环回归，对照 RNN 隐状态。",
    proves: [
      "内环之后 ||ΔW||_F > 0，多步大于一步，lr=0 时为 0",
      "重建损失下降",
      "RNN 状态是向量，TTT 状态是矩阵",
    ],
    doesNotProve: "不是官方 TTT 语言模型在 Pile 上的分数。",
    sourceFile: "lesson_17.py",
  },
  {
    lessonId: "18",
    purpose: "惊讶门控记忆：稀有 token 与常见 token 的写入幅度。",
    proves: [
      "稀有写入幅度大于常见，且大于 1.4 倍",
      "无门控时每次写入都是 1",
    ],
    doesNotProve: "不是 Titans 论文的长上下文基准。",
    sourceFile: "lesson_18.py",
  },
  {
    lessonId: "19",
    purpose: "快权重每 token、慢权重每序列，关掉其中一层看丢什么。",
    proves: [
      "两时间尺度能同时拟合 token 和风格",
      "关掉慢权重则风格误差上升，关掉快权重则 token 误差上升",
      "慢更新次数等于序列数",
    ],
    doesNotProve: "Hope 完整语言模型不能练。",
    sourceFile: "lesson_19.py",
  },
  {
    lessonId: "20",
    purpose: "二维策略上对比离线 SFT 和 on-policy 小步到原点的距离与遗忘。",
    proves: [
      "SFT 的 L2 和 KL 都大于 on-policy",
      "SFT 遗忘更多，且距离与遗忘正相关",
    ],
    doesNotProve: "不是 rl-razor-mnist 的像素实验，那是复现 #5。",
    sourceFile: "lesson_20.py",
  },
  {
    lessonId: "21",
    purpose: "网格世界技能库：成功入库后，后续任务尝试次数是否下降。",
    proves: [
      "有库时后期尝试次数下降",
      "库在成功后增长并被检索",
      "从零对照的库为空",
    ],
    doesNotProve: "不是 Voyager 官方 Minecraft 物品数。",
    sourceFile: "lesson_21.py",
  },
  {
    lessonId: "22",
    purpose: "三条数据河跑多日，关掉环境河后技能数是否停止增长。",
    proves: [
      "打开环境河的最终技能数高于网页+专家包",
      "关掉环境河后第 2 天起不再增长",
      "后期配方只出现在环境日程里",
    ],
    doesNotProve: "不是真实互联网或机器人数据流。",
    sourceFile: "lesson_22.py",
  },
  {
    lessonId: "23",
    purpose: "三轮自生成数据，对比有验证筛选和关掉筛选。",
    proves: [
      "关掉筛选后训练错误率上升",
      "筛选后错误率低于无筛选",
      "生成数据本身含错",
    ],
    doesNotProve: "不是 SEAL 全量 RL，更不是自己训练下一版基础模型。",
    sourceFile: "lesson_23.py",
  },
  {
    lessonId: "24",
    purpose: "14 个工作日对照：会写记忆/技能的 Agent 对冻结 Agent。",
    proves: [
      "第 14 日学习者旧任务保持高于冻结对照",
      "座位冲突被覆盖成新值",
      "新工具入库后尝试次数下降",
    ],
    doesNotProve: "不是真实两个月，也没有开放世界里的未写明惯例。四通道预衡协议见 python3 run.py capstone。",
    sourceFile: "lesson_24.py",
  },
];

export const practiceByLessonId: Record<string, LessonPractice> = Object.fromEntries(
  practices.map((item) => [item.lessonId, item]),
);
