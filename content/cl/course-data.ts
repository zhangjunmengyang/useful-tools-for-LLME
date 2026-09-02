export type CourseDifficulty = "入门" | "进阶" | "高级" | "研究级";

export type CourseUnitId =
  | "forget"
  | "toolkit"
  | "llm"
  | "memory"
  | "nested"
  | "agent";

export interface CourseUnit {
  id: CourseUnitId;
  order: number;
  title: string;
  question: string;
}

export interface CourseHardware {
  minimum: string;
  recommended: string;
  notes: string;
}

export interface CourseMisconception {
  myth: string;
  truth: string;
}

export interface CourseMetadata {
  id: string;
  slug: string;
  shortTitle: string;
  unit: CourseUnit;
  essentialQuestion: string;
  hook: string;
  outcomes: readonly string[];
  misconception: CourseMisconception;
  prerequisites: readonly string[];
  labId: string;
  readingTime: number;
  difficulty: CourseDifficulty;
  hardware: CourseHardware;
  learningMode: readonly string[];
}

export const courseUnits = [
  {
    id: "forget",
    order: 1,
    title: "看见遗忘",
    question: "直接接着训，旧任务为什么会塌？这件事怎么量？RAG 算不算学会了？",
  },
  {
    id: "toolkit",
    order: 2,
    title: "四类补丁",
    question: "正则、回放、扩结构、约束梯度各自把新知识写在哪？GDumb 为什么能打脸？",
  },
  {
    id: "llm",
    order: 3,
    title: "大模型接龙",
    question: "续预训练、顺序指令、LoRA 正交、模型合并，哪一步才是在线学习？",
  },
  {
    id: "memory",
    order: 4,
    title: "记忆、编辑、可塑性",
    question: "外挂记忆能叫来小王，为什么还不够？什么时候必须改权重？",
  },
  {
    id: "nested",
    order: 5,
    title: "学习变成架构",
    question: "测试时更新、惊讶门控、嵌套时间尺度、自己给自己出题，距离学习如何学习还有多远？",
  },
  {
    id: "agent",
    order: 6,
    title: "在岗学习",
    question: "Agent 怎样在连续工作日里变熟，并且周一学的周五还在？",
  },
] as const satisfies readonly CourseUnit[];

const unitById = Object.fromEntries(
  courseUnits.map((unit) => [unit.id, unit]),
) as Record<CourseUnitId, CourseUnit>;

const cpu = {
  minimum: "Mac / CPU",
  recommended: "1×24GB CUDA GPU（本课主线不需要）",
  notes: "浏览器实验和 CPU 机制实验即可完成主线。",
} as const;

const gpuOptional = {
  minimum: "Mac / CPU 完成机制实验",
  recommended: "1×24GB CUDA GPU",
  notes: "小模型 LoRA 或微调标在推荐档；7B 是加分项。",
} as const;

export const courseMetadata: CourseMetadata[] = [
  {
    id: "01",
    slug: "catastrophic-forgetting",
    shortTitle: "把遗忘跑出来",
    unit: unitById.forget,
    essentialQuestion:
      "同一个网络，先学任务 A 再学任务 B，A 的准确率为什么会塌？",
    hook:
      "不讲任何抗遗忘方法之前，先用 Split MNIST 把任务 1 的准确率看着掉下去。",
    outcomes: [
      "画出任务-时间热力图，说清 naive fine-tune 在本课设定下掉到多少。",
      "分清 task / domain / class incremental 三种设定。",
      "留下固定种子、命令和曲线，作为后面 23 课的对照基线。",
    ],
    misconception: {
      myth: "模型容量够大就不会忘。",
      truth: "容量不够会忘，容量够了用梯度接着训照样会把旧决策边界抹掉。本课用很小的 MLP 就能看见这件事。",
    },
    prerequisites: ["会跑 Python 和 PyTorch。", "不用先学任何课。"],
    labId: "lab-01-forgetting-slider",
    readingTime: 50,
    difficulty: "入门",
    hardware: cpu,
    learningMode: ["现场实验", "热力图", "三种增量设定"],
  },
  {
    id: "02",
    slug: "stability-plasticity",
    shortTitle: "既要记得住又要学得进",
    unit: unitById.forget,
    essentialQuestion: "把旧的钉死，新的就学不会。这个矛盾在实验里长什么样？",
    hook: "冻骨干、降学习率、混一点旧数据，四个点会落在稳定性-可塑性平面的不同位置。",
    outcomes: [
      "填一张稳定性-可塑性平面。",
      "能说出海马-新皮层类比在哪里失效。",
      "给第 05 课的 EWC 留下「为什么需要弹簧」的动机。",
    ],
    misconception: {
      myth: "稳定性就是把学习率调小。",
      truth: "调小学习率两头都弱。真正的方法要知道哪些参数能对旧任务负责。",
    },
    prerequisites: ["第 01 课的遗忘曲线。"],
    labId: "lab-02-stability-plane",
    readingTime: 50,
    difficulty: "入门",
    hardware: cpu,
    learningMode: ["对照实验", "平面图", "生物类比失效"],
  },
  {
    id: "03",
    slug: "cl-evaluation",
    shortTitle: "怎么量才算学会了",
    unit: unitById.forget,
    essentialQuestion: "只报最终平均准确率，为什么会把只会最后一件事的方法夸成好方法？",
    hook: "故意构造一份「最后任务满分、前面全忘」的矩阵，看哪些指标会上当。",
    outcomes: [
      "会算 Average Accuracy、Forgetting、BWT、FWT。",
      "一份后面 21 课共用的评测协议模板。",
      "知道 prequential 和「每个任务结束打一次分」的差别。",
    ],
    misconception: {
      myth: "平均准确率高就是持续学习成功。",
      truth: "平均可以被最后几个任务抬起来。必须同时看遗忘和后向迁移。",
    },
    prerequisites: ["第 01 课。"],
    labId: "lab-03-metric-liar",
    readingTime: 55,
    difficulty: "入门",
    hardware: cpu,
    learningMode: ["指标打假", "协议模板"],
  },
  {
    id: "04",
    slug: "not-just-rag",
    shortTitle: "把上下文塞满不等于学会了",
    unit: unitById.forget,
    essentialQuestion: "现在 Agent 靠长上下文和 RAG 活着。两个月上岗差在哪？",
    hook: "同一份员工名录，分别塞进 prompt、拿去检索、写进权重。然后把名录藏起来。",
    outcomes: [
      "上下文 / 检索 / 权重三栏对照表。",
      "说清检索在事实上很强、在技能和偏好上不够。",
      "能复述梁文峰「叫小王」例子，并标明转写未确认。",
    ],
    misconception: {
      myth: "上下文够长就等于持续学习。",
      truth: "上下文是工作记忆。撤掉名录之后，只有改过权重或写进长期记忆的还在。",
    },
    prerequisites: ["第 01-03 课。"],
    labId: "lab-04-call-xiaowang",
    readingTime: 60,
    difficulty: "进阶",
    hardware: gpuOptional,
    learningMode: ["三方法对照", "员工类比"],
  },
  {
    id: "05",
    slug: "ewc-regularization",
    shortTitle: "重要的权重不许动太多",
    unit: unitById.toolkit,
    essentialQuestion: "EWC 怎么知道哪些权重对旧任务重要？",
    hook: "Fisher 对角线是弹簧劲度。λ 从 0 扫到很大，点在稳定性-可塑性平面上滑动。",
    outcomes: [
      "λ 扫描曲线。",
      "Fisher 直方图。",
      "naive vs EWC vs SI vs LwF 的对照数字。",
    ],
    misconception: {
      myth: "EWC 能在完全没有旧数据时解决一切遗忘。",
      truth: "它近似旧任务曲率，λ 太大新任务学不会，而且对 class-incremental 往往不够。",
    },
    prerequisites: ["第 02 课。"],
    labId: "lab-05-fisher-pins",
    readingTime: 65,
    difficulty: "进阶",
    hardware: cpu,
    learningMode: ["正则", "Fisher", "λ 扫描"],
  },
  {
    id: "06",
    slug: "replay-der",
    shortTitle: "把旧样本带在身上",
    unit: unitById.toolkit,
    essentialQuestion: "回放为什么稳？DER 多蒸馏的那一项在防什么？",
    hook: "缓冲只有 N 个格子。N 从 200 加到 2000，遗忘曲线怎么走。",
    outcomes: [
      "缓冲大小-遗忘曲线。",
      "DER 蒸馏项消融。",
      "论文复现 #1：DER++ 方向性优于同等缓冲的 ER。",
    ],
    misconception: {
      myth: "回放就是把旧数据集存下来，等于没做持续学习。",
      truth: "缓冲远小于旧数据。问题是存什么、怎么跟新损失一起训。GDumb 会在第 08 课追问这件事。",
    },
    prerequisites: ["第 03、05 课。"],
    labId: "lab-06-replay-backpack",
    readingTime: 70,
    difficulty: "进阶",
    hardware: { ...cpu, notes: "CIFAR-10 对照建议单卡。" },
    learningMode: ["回放", "蒸馏", "复现 #1"],
  },
  {
    id: "07",
    slug: "architecture-prompts",
    shortTitle: "不改旧权重就再长一块",
    unit: unitById.toolkit,
    essentialQuestion: "扩网络、冻旧柱、加 adapter、加 prompt，新知识写在哪？",
    hook: "PackNet 给权重上锁。L2P 把任务指令放进 prompt 池。",
    outcomes: [
      "PackNet 掩码图。",
      "prompt pool 是否按任务分开的探针。",
      "和「每个任务一个头 + 冻骨干」的对照。",
    ],
    misconception: {
      myth: "加模块就不会忘。",
      truth: "容量用完一样学不动。共享骨干若还在更新，旧柱仍会被带偏。",
    },
    prerequisites: ["第 05 课。"],
    labId: "lab-07-packnet-wall",
    readingTime: 65,
    difficulty: "进阶",
    hardware: gpuOptional,
    learningMode: ["扩结构", "prompt", "掩码"],
  },
  {
    id: "08",
    slug: "gem-gdumb",
    shortTitle: "梯度不许踩旧任务，以及那个尴尬的基线",
    unit: unitById.toolkit,
    essentialQuestion: "若只把样本存下来、每个阶段从头训，分数往往也不差。这说明什么？",
    hook: "同一协议下跑 A-GEM、DER++、GDumb。看谁被打脸。",
    outcomes: [
      "三方法对照表。",
      "「你的设定会不会被 GDumb 打脸」检查清单。",
      "论文复现 #2。",
    ],
    misconception: {
      myth: "GDumb 赢了说明持续学习无用。",
      truth: "它说明很多实验的任务边界太干净，缓冲里的 i.i.d. 重训已经很强。",
    },
    prerequisites: ["第 03、06 课。"],
    labId: "lab-08-gradient-projection",
    readingTime: 65,
    difficulty: "进阶",
    hardware: cpu,
    learningMode: ["梯度投影", "强基线", "复现 #2"],
  },
  {
    id: "09",
    slug: "continual-pretraining",
    shortTitle: "换领域时旧能力怎么掉",
    unit: unitById.llm,
    essentialQuestion: "接到窄领域语料上续训，掉的是知识还是格式？",
    hook: "三条配方：原学习率硬上、学习率降 10 倍、每 batch 混 30% 通用数据。",
    outcomes: [
      "通用/领域双曲线。",
      "说清这和第 06 课回放是同一件事的语言模型版。",
    ],
    misconception: {
      myth: "续预训练只要数据对、loss 降就行。",
      truth: "必须同时报通用能力。学习率回热和通用数据回放是常见补丁，不是万能。",
    },
    prerequisites: ["第 04、06 课。"],
    labId: "lab-09-data-mix",
    readingTime: 70,
    difficulty: "高级",
    hardware: gpuOptional,
    learningMode: ["续预训练", "数据配比"],
  },
  {
    id: "10",
    slug: "sequential-instruction",
    shortTitle: "指令任务一个接一个",
    unit: unitById.llm,
    essentialQuestion: "先教数学再教摘要，模型会不会只会最后那件事？",
    hook: "四个指令任务依次 LoRA，填 4×4 矩阵，用第 03 课协议算遗忘。",
    outcomes: [
      "4×4 准确率矩阵。",
      "和「混训上限」的差距。",
    ],
    misconception: {
      myth: "指令微调过就不会忘。",
      truth: "顺序指令微调照样冲掉前一个任务，只是任务看起来更「像聊天」。",
    },
    prerequisites: ["第 03、09 课。"],
    labId: "lab-10-task-heatmap",
    readingTime: 70,
    difficulty: "高级",
    hardware: gpuOptional,
    learningMode: ["TRACE 风格", "4×4 矩阵"],
  },
  {
    id: "11",
    slug: "olora-treelora",
    shortTitle: "低秩更新为什么要正交",
    unit: unitById.llm,
    essentialQuestion: "每个任务一个 LoRA，互相对齐时会抢同一方向。正交在防什么？",
    hook: "测两个 LoRA 矩阵的夹角。O-LoRA 之后应接近 90°。",
    outcomes: [
      "LoRA 方向夹角。",
      "naive LoRA vs O-LoRA 的 4 任务矩阵。",
      "论文复现 #3。",
    ],
    misconception: {
      myth: "LoRA 参数少所以不会忘。",
      truth: "少只是省显存。方向重叠时，新任务仍会改写旧任务用过的低秩子空间。",
    },
    prerequisites: ["第 10 课。"],
    labId: "lab-11-lora-orthogonal",
    readingTime: 75,
    difficulty: "高级",
    hardware: gpuOptional,
    learningMode: ["PEFT-CL", "正交", "复现 #3"],
  },
  {
    id: "12",
    slug: "model-merging",
    shortTitle: "不接着训，把几个模型加起来",
    unit: unitById.llm,
    essentialQuestion: "任务向量相加为什么有时有效？合并算不算持续学习？",
    hook: "线性相加、TIES、DARE 三份合并，在两个任务上测。",
    outcomes: [
      "三份合并对照。",
      "书面判断：合并是事后缝合，不是在线持续学习。",
    ],
    misconception: {
      myth: "合并等于多任务学习。",
      truth: "合并发生在训练结束之后，没有在线约束，符号冲突要靠 TIES 一类规则修。",
    },
    prerequisites: ["第 10 课。"],
    labId: "lab-12-task-vector",
    readingTime: 60,
    difficulty: "进阶",
    hardware: cpu,
    learningMode: ["任务向量", "mergekit"],
  },
  {
    id: "13",
    slug: "external-memory",
    shortTitle: "把日记写在模型外面",
    unit: unitById.memory,
    essentialQuestion: "分层记忆、抽取-更新、海马索引，各自解决「叫得动小王」的哪一段？",
    hook: "喂 20 条公司事实，隔一轮再问。再写入一条冲突事实。",
    outcomes: [
      "冲突写入案例。",
      "外挂记忆不会的三件事：新技能、新推理模式、新运动策略。",
    ],
    misconception: {
      myth: "记忆系统就是持续学习。",
      truth: "它解决事实的写入和召回。权重没动，技能和推理习惯通常不会变。",
    },
    prerequisites: ["第 04 课。"],
    labId: "lab-13-memory-drawers",
    readingTime: 70,
    difficulty: "进阶",
    hardware: gpuOptional,
    learningMode: ["Letta / Mem0", "冲突写入"],
  },
  {
    id: "14",
    slug: "knowledge-editing",
    shortTitle: "改一条事实，别把邻居改坏",
    unit: unitById.memory,
    essentialQuestion: "ROME 凭什么说事实存在某层 MLP 的某几个关键？",
    hook: "测可靠性、泛化、局部性、流畅性。再做一个忘掉指定事实的对照。",
    outcomes: [
      "四指标表。",
      "编辑 vs 微调 vs RAG 适用表。",
    ],
    misconception: {
      myth: "编辑成功就是那一条问答对了。",
      truth: "邻居事实被带偏、同义问法答不上、语言模型变结巴，都不算成功。",
    },
    prerequisites: ["第 04 课。"],
    labId: "lab-14-locate-edit",
    readingTime: 70,
    difficulty: "高级",
    hardware: gpuOptional,
    learningMode: ["ROME/MEMIT", "四指标", "unlearning"],
  },
  {
    id: "15",
    slug: "loss-of-plasticity",
    shortTitle: "学着学着学不动了",
    unit: unitById.memory,
    essentialQuestion: "没有旧任务考试，网络在一长串新任务之后也会失去学习能力。这和遗忘是两件事吗？",
    hook: "画第 k 个任务的学习速度。打开 continual backprop，看死神经元是否被重置。",
    outcomes: [
      "学习速度曲线。",
      "死神经元比例。",
      "论文复现 #4。",
    ],
    misconception: {
      myth: "持续学习的唯一问题是忘。",
      truth: "还会学不动。Dohare et al. 2024 Nature 把这件事从遗忘里分开。",
    },
    prerequisites: ["第 01、02 课。"],
    labId: "lab-15-dead-neurons",
    readingTime: 65,
    difficulty: "高级",
    hardware: cpu,
    learningMode: ["可塑性", "continual backprop", "复现 #4"],
  },
  {
    id: "16",
    slug: "when-weights-must-move",
    shortTitle: "什么时候必须改权重",
    unit: unitById.memory,
    essentialQuestion: "梁文峰说的瓶颈，能不能靠把日记写好来绕过去？",
    hook: "四类新东西 × 四种写入位置，看谁在哪一类过关。",
    outcomes: [
      "四类经验 × 四种写入的通过矩阵。",
      "第六幕的设计依据。",
    ],
    misconception: {
      myth: "记忆写好了就不需要改权重。",
      truth: "事实可以外挂。技能、偏好、新的计分规则通常要进权重或等价的内环学习。",
    },
    prerequisites: ["第 04、10、13、14 课。"],
    labId: "lab-16-router",
    readingTime: 60,
    difficulty: "高级",
    hardware: gpuOptional,
    learningMode: ["分流", "综合对照"],
  },
  {
    id: "17",
    slug: "test-time-training",
    shortTitle: "读这段话的时候权重正在动",
    unit: unitById.nested,
    essentialQuestion: "TTT 层的隐状态是一套可以用梯度更新的权重。这和普通 RNN 差在哪？",
    hook: "短序列上执行 TTT-Linear 内环，打印每步后 W 的变化范数。",
    outcomes: [
      "W 更新范数曲线。",
      "TTT vs RNN 的状态含义对照表。",
    ],
    misconception: {
      myth: "测试时学习就是多走几步梯度的微调。",
      truth: "TTT 层把内环更新写进架构，对当前序列做，不一定把慢权重写回磁盘。",
    },
    prerequisites: ["第 16 课。"],
    labId: "lab-17-inner-loop",
    readingTime: 75,
    difficulty: "研究级",
    hardware: gpuOptional,
    learningMode: ["TTT-Linear", "内环"],
  },
  {
    id: "18",
    slug: "titans-surprise",
    shortTitle: "惊讶的事情才值得写入长期记忆",
    unit: unitById.nested,
    essentialQuestion: "用惊讶当写入门控，和注意力的「全部看见」差在哪？",
    hook: "合成序列里周期性插入稀有 token，验证写入幅度更大。",
    outcomes: [
      "稀有 vs 常见 token 的写入幅度表。",
      "明确本课是机制复现，不对齐 Titans 语言模型分数。",
    ],
    misconception: {
      myth: "长期记忆就是更长的上下文。",
      truth: "Titans 要决定写什么。惊讶门控是一种选择规则，不是把窗口拉长。",
    },
    prerequisites: ["第 17 课。"],
    labId: "lab-18-surprise-gate",
    readingTime: 70,
    difficulty: "研究级",
    hardware: cpu,
    learningMode: ["惊讶门控", "机制复现"],
  },
  {
    id: "19",
    slug: "nested-learning",
    shortTitle: "优化器也是一层记忆",
    unit: unitById.nested,
    essentialQuestion: "架构和优化器看成不同时间尺度的嵌套学习问题。Hope 比 Titans 多了什么？",
    hook: "两层更新频率不同的线性记忆：内环每 token，外环每序列。",
    outcomes: [
      "两时间尺度 vs 一时间尺度对照。",
      "Hope 完整语言模型训练标「不能练」。",
    ],
    misconception: {
      myth: "深度网络的层数就是学习的层数。",
      truth: "Nested Learning 说的层是更新频率不同的优化问题，Adam 的动量也可以看成一层记忆。",
    },
    prerequisites: ["第 17、18 课。"],
    labId: "lab-19-nested-clocks",
    readingTime: 80,
    difficulty: "研究级",
    hardware: cpu,
    learningMode: ["嵌套学习", "Hope", "只讲档"],
  },
  {
    id: "20",
    slug: "seal-rl-razor",
    shortTitle: "自己出题，以及为什么 RL 比较不易忘",
    unit: unitById.nested,
    essentialQuestion: "on-policy RL 比 SFT 更不易遗忘，是因为分布离原模型更近吗？",
    hook: "同一新任务，SFT vs on-policy RL，测旧任务保持和到原模型的 KL。",
    outcomes: [
      "rl-razor-mnist 复现曲线，论文复现 #5。",
      "SEAL 能跑通的部分写实战记录。",
    ],
    misconception: {
      myth: "RL 不易忘是因为奖励更聪明。",
      truth: "RL's Razor 的主张是分布偏移更小。要用 KL 和遗忘的相关性来检验，不能停在口头。",
    },
    prerequisites: ["第 10、17 课。"],
    labId: "lab-20-rl-razor",
    readingTime: 80,
    difficulty: "研究级",
    hardware: gpuOptional,
    learningMode: ["SEAL", "RL's Razor", "复现 #5"],
  },
  {
    id: "21",
    slug: "voyager-skill-library",
    shortTitle: "技能写成能再调用的代码",
    unit: unitById.agent,
    essentialQuestion: "Voyager 把成功的程序放进技能库。这算持续学习吗？差哪一口？",
    hook: "网格世界里提出任务、写程序、验证、入库、检索。对照每次从零写。",
    outcomes: [
      "有库之后后续任务尝试次数是否下降。",
      "书面判断：这是经验外存，对应第 16 课的「流程技能 / 记忆」格。",
    ],
    misconception: {
      myth: "技能库就是权重里学会了技能。",
      truth: "权重没变。下次检索到旧脚本，模型仍然要靠当前上下文执行。",
    },
    prerequisites: ["第 13、16 课。"],
    labId: "lab-21-skill-drawer",
    readingTime: 70,
    difficulty: "高级",
    hardware: cpu,
    learningMode: ["Voyager 机制档", "技能库"],
  },
  {
    id: "22",
    slug: "era-of-experience",
    shortTitle: "数据从世界来",
    unit: unitById.agent,
    essentialQuestion: "经验时代和把互联网再爬一遍差在哪？",
    hook: "固定专家包、离线随机轨迹、on-policy 与环境交互，三条河流向同一个技能库。",
    outcomes: [
      "三种数据来源的技能增长曲线。",
      "经验时代和普通 RL 的差别：时间跨度、非平稳、没有任务边界。",
    ],
    misconception: {
      myth: "经验就是更多离线轨迹。",
      truth: "Silver 和 Sutton 强调的是与世界持续交互、自己产生数据。离线包没有这一段。",
    },
    prerequisites: ["第 20、21 课。"],
    labId: "lab-22-three-rivers",
    readingTime: 60,
    difficulty: "高级",
    hardware: cpu,
    learningMode: ["经验时代", "非平稳"],
  },
  {
    id: "23",
    slug: "self-iteration",
    shortTitle: "自己做研究、自己出下一版",
    unit: unitById.agent,
    essentialQuestion: "今天公开技术走到哪一步？哪一步还只是故事？",
    hook: "三轮自改进：生成、筛选、训练、评测。关掉筛选，看会不会越训越错。",
    outcomes: [
      "3 轮日志。",
      "已实现 / 实验室 / 猜想三栏表。梁文峰说的奇点放在猜想栏。",
    ],
    misconception: {
      myth: "模型能生成训练数据，就已经会自我迭代。",
      truth: "没有验证筛选，错误会自我强化。本课实验不是 AGI 自迭代。",
    },
    prerequisites: ["第 20 课。"],
    labId: "lab-23-self-improve",
    readingTime: 70,
    difficulty: "研究级",
    hardware: gpuOptional,
    learningMode: ["自编辑", "诚实分档"],
  },
  {
    id: "24",
    slug: "two-month-hire",
    shortTitle: "两个月上岗（缩小版）",
    unit: unitById.agent,
    essentialQuestion: "一个 Agent 在连续多天的工作流里，能不能越用越熟，并且周一学的周五还在？",
    hook: "14 个模拟工作日。对照冻结 / 只记忆 / 记忆+技能 / 再加权重。",
    outcomes: [
      "毕业报告：协议、对照、遗忘数字、失败案例。",
      "写清离梁文峰说的那种持续学习还差什么。",
    ],
    misconception: {
      myth: "做完 14 日模拟就等于解决了持续学习。",
      truth: "这是缩小版。没有真实两个月、没有开放世界、权重更新常常是加分项。",
    },
    prerequisites: ["第 13、16、20、21 课。"],
    labId: "lab-24-fourteen-days",
    readingTime: 90,
    difficulty: "研究级",
    hardware: gpuOptional,
    learningMode: ["毕业设计", "14 日看板"],
  },
];

export const courseById = Object.fromEntries(
  courseMetadata.map((course) => [course.id, course]),
) as Record<string, CourseMetadata>;

export const courseBySlug = Object.fromEntries(
  courseMetadata.map((course) => [course.slug, course]),
) as Record<string, CourseMetadata>;

export function getCourseMetadata(idOrSlug: string): CourseMetadata {
  const course = courseById[idOrSlug] ?? courseBySlug[idOrSlug];
  if (!course) {
    throw new Error(`Unknown course: ${idOrSlug}`);
  }
  return course;
}
