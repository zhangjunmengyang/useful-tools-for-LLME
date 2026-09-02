// Learn WM 的课程元数据。内容来源：/CURRICULUM.md（九幕蓝图）。
// 课程正文放在 web/content/lessons/NN_slug.md；这里只放地图级信息。
// status 标记正文是否已发布：published = 正文已入库，planned = 未开课。
// 课数不设上限；当前主线是 45 课。

export type CourseDifficulty = "入门" | "进阶" | "高级" | "研究级";

export type CourseStatus = "published" | "planned";

export type CourseUnitId =
  | "vmc"
  | "engine"
  | "generative"
  | "jepa"
  | "craft"
  | "spatial"
  | "embodied"
  | "deskpet"
  | "frontier";

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

export interface CourseAnchor {
  name: string;
  url?: string;
}

export interface CourseMetadata {
  id: string;
  slug: string;
  shortTitle: string;
  unit: CourseUnit;
  status: CourseStatus;
  essentialQuestion: string;
  hook: string;
  outcomes: readonly string[];
  anchors: readonly CourseAnchor[];
  readingTime: number;
  difficulty: CourseDifficulty;
  hardware: CourseHardware;
}

export const courseUnits = [
  {
    id: "vmc",
    order: 1,
    title: "第一个世界模型",
    question:
      "在 CarRacing 上复现 2018 年的 World Models，理解 V-M-C 三个组件。",
  },
  {
    id: "engine",
    order: 2,
    title: "潜空间方法",
    question:
      "比较 RSSM、Dreamer、MuZero 和 TD-MPC2 的状态表示、训练与规划方法。",
  },
  {
    id: "generative",
    order: 3,
    title: "生成式世界模型",
    question:
      "比较 token、扩散和 latent action 三种生成未来的方法。",
  },
  {
    id: "jepa",
    order: 4,
    title: "JEPA 路线",
    question:
      "从教学实现、官方代码和公开权重理解表征预测路线。",
  },
  {
    id: "craft",
    order: 5,
    title: "评测与改造",
    question:
      "学习统一评测、缩放实验、消融和结构改造。",
  },
  {
    id: "spatial",
    order: 6,
    title: "空间与物体",
    question:
      "给状态装上持久 3D/4D 和物体槽：离开视野的杯子还在，桌上的东西是分开的。",
  },
  {
    id: "embodied",
    order: 7,
    title: "接到身体上",
    question:
      "把世界模型接到连续动作、接触和安全约束上，分清模仿策略、VLA 和动力学。",
  },
  {
    id: "deskpet",
    order: 8,
    title: "桌宠毕业设计",
    question:
      "在一张桌子上做出会看、会想、会克制的小机器人。没有真机就用摄像头档完成。",
  },
  {
    id: "frontier",
    order: 9,
    title: "刷榜、声音与配方",
    question:
      "读 2025-2026 的榜、全模态、声音、训练配方和架构。毕业标准仍在第 32 课和第 33 课。",
  },
] as const satisfies readonly CourseUnit[];

const unitById = Object.fromEntries(
  courseUnits.map((unit) => [unit.id, unit]),
) as Record<CourseUnitId, CourseUnit>;

const repos = {
  ctallec: {
    name: "ctallec/world-models",
    url: "https://github.com/ctallec/world-models",
  },
  hardmaru: {
    name: "hardmaru/WorldModelsExperiments",
    url: "https://github.com/hardmaru/WorldModelsExperiments",
  },
  dreamerTorch: {
    name: "NM512/dreamerv3-torch",
    url: "https://github.com/NM512/dreamerv3-torch",
  },
  planet: {
    name: "Kaixhin/PlaNet",
    url: "https://github.com/Kaixhin/PlaNet",
  },
  dreamerJax: {
    name: "danijar/dreamerv3",
    url: "https://github.com/danijar/dreamerv3",
  },
  muzero: {
    name: "werner-duvaud/muzero-general",
    url: "https://github.com/werner-duvaud/muzero-general",
  },
  tdmpc2: {
    name: "nicklashansen/tdmpc2",
    url: "https://github.com/nicklashansen/tdmpc2",
  },
  iris: {
    name: "eloialonso/iris",
    url: "https://github.com/eloialonso/iris",
  },
  diamond: {
    name: "eloialonso/diamond",
    url: "https://github.com/eloialonso/diamond",
  },
  tinyworlds: {
    name: "AlmondGod/tinyworlds",
    url: "https://github.com/AlmondGod/tinyworlds",
  },
  onexgpt: {
    name: "1x-technologies/1xgpt",
    url: "https://github.com/1x-technologies/1xgpt",
  },
  openOasis: {
    name: "etched-ai/open-oasis",
    url: "https://github.com/etched-ai/open-oasis",
  },
  matrixGame: {
    name: "SkyworkAI/Matrix-Game",
    url: "https://github.com/SkyworkAI/Matrix-Game",
  },
  keonJepa: {
    name: "keon/jepa",
    url: "https://github.com/keon/jepa",
  },
  ijepa: {
    name: "facebookresearch/ijepa",
    url: "https://github.com/facebookresearch/ijepa",
  },
  ebJepa: {
    name: "facebookresearch/eb_jepa",
    url: "https://github.com/facebookresearch/eb_jepa",
  },
  vjepa2: {
    name: "facebookresearch/vjepa2",
    url: "https://github.com/facebookresearch/vjepa2",
  },
  vggt: {
    name: "facebookresearch/vggt",
    url: "https://github.com/facebookresearch/vggt",
  },
  cut3r: {
    name: "CUT3R/CUT3R",
    url: "https://github.com/CUT3R/CUT3R",
  },
  cswm: {
    name: "tkipf/c-swm",
    url: "https://github.com/tkipf/c-swm",
  },
  dinowm: {
    name: "DINO-WM",
    url: "https://dino-wm.github.io/",
  },
  cosmos: {
    name: "nvidia-cosmos/cosmos-predict2.5",
    url: "https://github.com/nvidia-cosmos/cosmos-predict2.5",
  },
  daydreamer: {
    name: "danijar/daydreamer",
    url: "https://github.com/danijar/daydreamer",
  },
  lerobot: {
    name: "huggingface/lerobot",
    url: "https://github.com/huggingface/lerobot",
  },
  openvla: {
    name: "openvla/openvla",
    url: "https://github.com/openvla/openvla",
  },
  openpi: {
    name: "Physical-Intelligence/openpi",
    url: "https://github.com/Physical-Intelligence/openpi",
  },
  genesis: {
    name: "Genesis-Embodied-AI/genesis-world",
    url: "https://github.com/Genesis-Embodied-AI/genesis-world",
  },
  reachyMini: {
    name: "Reachy Mini",
    url: "https://huggingface.co/docs/reachy_mini/en/index",
  },
  cosmos3: {
    name: "nvidia/cosmos",
    url: "https://github.com/nvidia/cosmos",
  },
  worldscore: {
    name: "haoyi-duan/WorldScore",
    url: "https://github.com/haoyi-duan/WorldScore",
  },
  worldmodelbench: {
    name: "WorldModelBench",
    url: "https://github.com/WorldModelBench-Team/WorldModelBench",
  },
  physicsiq: {
    name: "google-deepmind/physics-iq-benchmark",
    url: "https://github.com/google-deepmind/physics-iq-benchmark",
  },
  selfForcing: {
    name: "Self-Forcing",
    url: "https://github.com/guandeh17/Self-Forcing",
  },
  worldmem: {
    name: "WorldMem",
    url: "https://xizaoqu.github.io/worldmem/",
  },
} as const;

export const courseMetadata: CourseMetadata[] = [
  {
    id: "01",
    slug: "world-models-hands-on",
    shortTitle: "跑通你的第一个世界模型",
    unit: unitById.vmc,
    status: "published",
    essentialQuestion:
      "一个“会做梦”的智能体由哪几个零件组成？观察、状态、动作、预测各自落在代码的哪里？",
    hook:
      "装环境、采随机策略轨迹、把 ctallec 仓库完整跑一遍小规模流程，拆出 V、M、C 三个组件的输入输出。",
    outcomes: [
      "一个能跑的 CarRacing 环境，加上第一份自己录的轨迹数据集。",
      "一份 V/M/C 零件清单笔记：每个零件吃什么、吐什么、坏了会怎样。",
      "复现、测量和留证据所需的基本记录：训练曲线、评估协议和随机种子。",
    ],
    anchors: [repos.ctallec, repos.hardmaru],
    readingTime: 60,
    difficulty: "入门",
    hardware: {
      minimum: "Mac / 纯 CPU（阅读、采数据、小规模训练）",
      recommended: "1×24GB CUDA GPU",
      notes:
        "gymnasium CarRacing-v3 与旧 gym API 有差异，课内给补丁说明。",
    },
  },
  {
    id: "02",
    slug: "vae-visual-compression",
    shortTitle: "用 32 个数表示一帧赛道",
    unit: unitById.vmc,
    status: "published",
    essentialQuestion:
      "压缩一定会丢信息。怎么知道丢掉的是不是要命的那部分？",
    hook:
      "训练 VAE、检查重建，并用插值、遍历和探针分析 latent 保存了哪些信息。",
    outcomes: [
      "一个训练好的 VAE，重建质量有量化记录。",
      "latent 走查报告：哪些信息保住了、哪些丢了、为什么无所谓或有所谓。",
      "不同 β 和 latent 维数的对比记录。",
    ],
    anchors: [repos.ctallec],
    readingTime: 55,
    difficulty: "入门",
    hardware: {
      minimum: "Mac / 纯 CPU（VAE 小训练可完成）",
      recommended: "1×24GB CUDA GPU",
      notes: "数据用第 01 课自己录的轨迹，不需要重新采集。",
    },
  },
  {
    id: "03",
    slug: "mdn-rnn-action-conditioned",
    shortTitle: "让 M 按动作预测下一步",
    unit: unitById.vmc,
    status: "published",
    essentialQuestion:
      "为什么预测未来要输出一团分布，而不是一个点？动作到底有没有被模型用起来？",
    hook:
      "动作对换对照是判断“动作盲”的试金石：同一状态配不同动作，预测必须分岔；分不了岔，模型就没在看动作。",
    outcomes: [
      "一个动作条件的动力学模型（MDN-RNN）。",
      "动作对换对照报告：证明动作真的改变了预测。",
      "一条多步 rollout 的误差漂移曲线，用于观察误差如何累积。",
    ],
    anchors: [repos.ctallec],
    readingTime: 65,
    difficulty: "进阶",
    hardware: {
      minimum: "1×消费级 CUDA GPU",
      recommended: "1×24GB CUDA GPU",
      notes: "MDN-RNN 训练依赖第 02 课的 VAE 产物。",
    },
  },
  {
    id: "04",
    slug: "controller-dream-training",
    shortTitle: "训练控制器，并把训练搬进梦里",
    unit: unitById.vmc,
    status: "published",
    essentialQuestion:
      "控制器只有 867 个参数，凭什么能开车？在自己模型的想象里训练策略，什么时候会被“钻空子”？",
    hook:
      "先用 CMA-ES 在真实环境训练线性 controller，然后写少量胶水代码让 MDN-RNN 生成梦境轨迹、在梦里训练、回真实环境验收，调温度 τ 看策略怎么钻模型学错的地方。",
    outcomes: [
      "World Models 复现报告（论文复现 #1）：真实分数方向性对照、梦境移植实验、失败案例分析。",
      "一组模型内外的分数对照，用来验证策略是否利用了模型误差。",
    ],
    anchors: [repos.ctallec, repos.hardmaru],
    readingTime: 75,
    difficulty: "进阶",
    hardware: {
      minimum: "1×消费级 CUDA GPU",
      recommended: "1×24GB CUDA GPU",
      notes: "CMA-ES 并行评估吃 CPU 核数；无显示器环境用 xvfb-run。",
    },
  },
  {
    id: "05",
    slug: "rssm-planet",
    shortTitle: "RSSM 的确定状态与随机状态",
    unit: unitById.engine,
    status: "published",
    essentialQuestion:
      "纯 RNN 状态为什么记不住又赌不准？确定通道和随机通道各管什么？",
    hook:
      "精读 dreamerv3-torch 的 RSSM 模块，在 DMC 一个任务上跑通重建与多步想象，再用 CEM 在潜空间里第一次做规划。",
    outcomes: [
      "RSSM 结构笔记，对照代码逐行讲清两条通道。",
      "一个任务的想象序列可视化。",
      "KL 平衡、free bits 各自防什么病的消融记录。",
    ],
    anchors: [repos.dreamerTorch, repos.planet],
    readingTime: 70,
    difficulty: "进阶",
    hardware: {
      minimum: "1×消费级 CUDA GPU",
      recommended: "1×24GB CUDA GPU（数小时）",
      notes: "dreamerv3-torch 已归档冻结，教学反而稳定；对照 PlaNet 原始实现。",
    },
  },
  {
    id: "06",
    slug: "dreamerv3-imagination",
    shortTitle: "DreamerV3 的想象训练",
    unit: unitById.engine,
    status: "published",
    essentialQuestion:
      "为什么在想象里训练策略比在真环境里训便宜一个量级？一套超参凭什么通吃 150+ 任务？",
    hook:
      "单任务复现 DreamerV3（DMC vision 或 Crafter），把 symlog、twohot、KL 平衡这些“通吃”的工程细节逐个做有/无对照。",
    outcomes: [
      "单任务复现曲线（论文复现 #3）。",
      "稳定性技巧消融小报告：每个细节各救了什么。",
    ],
    anchors: [repos.dreamerTorch, repos.dreamerJax],
    readingTime: 75,
    difficulty: "高级",
    hardware: {
      minimum: "1×24GB CUDA GPU",
      recommended: "1×24GB CUDA GPU（数小时到一两天）",
      notes: "JAX 官方仓库用作对照精读，主实验在 PyTorch 版上做。",
    },
  },
  {
    id: "07",
    slug: "muzero-value-equivalence",
    shortTitle: "MuZero 的价值等价模型",
    unit: unitById.engine,
    status: "published",
    essentialQuestion:
      "不重建观察、只预测价值/策略/奖励的模型，凭什么还配叫世界模型？",
    hook:
      "在 CartPole 上训练 MuZero，记录 MCTS 的访问次数与 Q 值，再用线性探针检查隐状态保存了哪些物理量。",
    outcomes: [
      "MCTS 搜索过程记录。",
      "价值等价探针报告：模型“对”的标准从像素对搬到了决策对。",
    ],
    anchors: [repos.muzero],
    readingTime: 60,
    difficulty: "进阶",
    hardware: {
      minimum: "笔记本 CPU 即可",
      recommended: "笔记本 CPU；GPU 可加速但不必需",
      notes: "教学向社区实现，MIT 许可，适合整仓精读。",
    },
  },
  {
    id: "08",
    slug: "tdmpc2-decoder-free",
    shortTitle: "TD-MPC2 的连续控制",
    unit: unitById.engine,
    status: "published",
    essentialQuestion:
      "把“重建”彻底扔掉之后，靠什么信号防止潜空间坍缩？",
    hook:
      "单任务在线 RL 复现，用官方 checkpoint 练评测，再和第 06 课的 Dreamer 做同任务同预算的正面对比。",
    outcomes: [
      "Dreamer vs TD-MPC2 对照实验报告：重建 vs 免重建的第一手证据。",
      "这份报告会在第 16 课“路线之争”里再次上场。",
    ],
    anchors: [repos.tdmpc2],
    readingTime: 70,
    difficulty: "高级",
    hardware: {
      minimum: "8GB 显存即可",
      recommended: "1×24GB CUDA GPU",
      notes: "官方仓库活跃维护，附 300+ 官方 checkpoints。",
    },
  },
  {
    id: "09",
    slug: "iris-token-world-model",
    shortTitle: "IRIS 把未来写成 token",
    unit: unitById.generative,
    status: "published",
    essentialQuestion:
      "把图像离散化成 token 会丢什么？为什么语言模型的配方能直接搬来建世界？",
    hook:
      "先用预训练权重进到世界模型里面玩 Breakout（几分钟跑通，全课最好的动机实验），再单游戏训练 Atari 100k，拆 VQ 码本坍缩问题。",
    outcomes: [
      "单游戏训练记录（论文复现 #4）。",
      "“在梦里玩游戏”的体验报告：模型哪里穿帮，为什么。",
    ],
    anchors: [repos.iris],
    readingTime: 70,
    difficulty: "高级",
    hardware: {
      minimum: "1×消费级 CUDA GPU",
      recommended: "1×24GB CUDA GPU",
      notes: "代码量极小，有 HF 预训练权重可直接体验。",
    },
  },
  {
    id: "10",
    slug: "diamond-diffusion-world-model",
    shortTitle: "DIAMOND 预测下一帧",
    unit: unitById.generative,
    status: "published",
    essentialQuestion:
      "连续扩散比离散 token 到底多保住了什么？为什么弹道、准星这类小细节恰恰最重要？",
    hook:
      "玩 Atari 和 CSGO 的预训练世界模型（Mac MPS 可跑 demo），再与第 09 课的 IRIS 同游戏并排对比视觉保真与 agent 分数。",
    outcomes: [
      "IRIS vs DIAMOND 对照报告（论文复现 #5）：token vs 扩散，谁在哪类细节上赢。",
    ],
    anchors: [repos.diamond],
    readingTime: 70,
    difficulty: "高级",
    hardware: {
      minimum: "Mac MPS 可玩 demo",
      recommended: "1×24GB CUDA GPU（单游戏训练）",
      notes: "与 IRIS 同作者，天然构成对比作业。",
    },
  },
  {
    id: "11",
    slug: "genie-latent-actions",
    shortTitle: "Genie 从视频中发现动作",
    unit: unitById.generative,
    status: "published",
    essentialQuestion:
      "只有视频、没有按键记录，模型怎么自己发现“世界上存在 8 种动作”？",
    hook:
      "用 tinyworlds（最小 Genie 教学实现）在小游戏数据集上训练，检查学出的 latent action 是否对应可解释的操作，再用 1xgpt 的真实人形机器人数据跑 baseline。",
    outcomes: [
      "latent action 可解释性分析。",
      "真实机器人数据上的 GENIE-style baseline 训练记录。",
    ],
    anchors: [repos.tinyworlds, repos.onexgpt],
    readingTime: 70,
    difficulty: "高级",
    hardware: {
      minimum: "1×消费级 CUDA GPU",
      recommended: "1×24GB CUDA GPU（1xgpt 有 4090 基准）",
      notes: "1xgpt 提供 100+ 小时预 token 化的真实机器人数据。",
    },
  },
  {
    id: "12",
    slug: "frontier-landscape",
    shortTitle: "哪些系统算世界模型",
    unit: unitById.generative,
    status: "published",
    essentialQuestion:
      "用统一标准比较 GameNGen、Oasis、Genie 3、Cosmos、Sora 类和 Marble 是否能支持决策。",
    hook:
      "跑 open-oasis 体验长时程一致性崩坏，建立“预测/生成/规划”三分评测观，把前沿系统分进开源可跑、仅权重、纯 demo 三档。",
    outcomes: [
      "前沿地图笔记：三档清单，加上各系统在三分评测下的定位。",
      "明确知道哪些系统无权重不能练（Genie 3、World Labs Marble、GAIA-2）。",
    ],
    anchors: [repos.openOasis, repos.matrixGame],
    readingTime: 60,
    difficulty: "进阶",
    hardware: {
      minimum: "1×消费级 CUDA GPU（open-oasis 500M 权重）",
      recommended: "1×24GB CUDA GPU（Matrix-Game 实时交互）",
      notes: "NVIDIA Cosmos 只做一次性体验，不锚定。",
    },
  },
  {
    id: "13",
    slug: "ijepa-from-scratch",
    shortTitle: "I-JEPA 预测表征",
    unit: unitById.jepa,
    status: "published",
    essentialQuestion:
      "不重建图像，模型为什么不会摆烂输出常数（坍缩）？EMA target 和 masking 设计各挡住哪条坍缩路径？",
    hook:
      "在 CIFAR-10 上训练 I-JEPA 教学实现，用线性探针验收，再分别改掉 EMA 和 masking 做坍缩审计。",
    outcomes: [
      "小规模 I-JEPA 复现报告（论文复现 #2）：探针精度加坍缩消融。",
      "官方 ijepa 仓库 masking/EMA 参考实现的精读笔记。",
    ],
    anchors: [repos.keonJepa, repos.ijepa],
    readingTime: 75,
    difficulty: "高级",
    hardware: {
      minimum: "CPU / MPS / CUDA 都能跑",
      recommended: "1×消费级 CUDA GPU",
      notes: "官方原配置是 16×A100，不复现官方训练，只精读实现。",
    },
  },
  {
    id: "14",
    slug: "eb-jepa-action-conditioned",
    shortTitle: "给 JEPA 接上动作",
    unit: unitById.jepa,
    status: "published",
    essentialQuestion:
      "JEPA 学到的表征怎么用于“推演未来 + 规划动作”？能量视角到底在说什么？",
    hook:
      "把 eb_jepa 的三个例子全部跑通，重点在动作条件 Video JEPA：在 Two Rooms 里用 JEPA 世界模型做规划，与第 05 课的 CEM-in-latent 对照。",
    outcomes: [
      "AC-JEPA 规划实验报告：JEPA 潜空间规划 vs RSSM 潜空间规划的并排笔记。",
    ],
    anchors: [repos.ebJepa],
    readingTime: 70,
    difficulty: "高级",
    hardware: {
      minimum: "1×消费级 CUDA GPU",
      recommended: "1×24GB CUDA GPU（每个例子单卡几小时）",
      notes: "FAIR 官方教学库：Image JEPA、Video JEPA、AC Video JEPA 三个例子递进。",
    },
  },
  {
    id: "15",
    slug: "vjepa2-in-practice",
    shortTitle: "用 V-JEPA 2 的公开权重做评测",
    unit: unitById.jepa,
    status: "published",
    essentialQuestion:
      "一个 2B 的视频表征模型，“理解物理”体现在哪些可测量的地方？",
    hook:
      "冻结 backbone 做 attentive probe 评测，小数据微调，再精读 V-JEPA 2-AC 动作条件世界模型的机器人规划设计（不复现训练，讲清接口）。",
    outcomes: [
      "探针评测报告（自选下游任务）。",
      "V-JEPA 2-AC 接口笔记。",
    ],
    anchors: [repos.vjepa2],
    readingTime: 70,
    difficulty: "高级",
    hardware: {
      minimum: "1×消费级 CUDA GPU（探针评测单卡可行）",
      recommended: "1×24GB CUDA GPU",
      notes: "ViT-B 80M 到 ViT-G 2B 权重一行加载；这课属于实战/体验档，不算复现。",
    },
  },
  {
    id: "16",
    slug: "three-roads-debate",
    shortTitle: "三条世界模型路线怎么选",
    unit: unitById.jepa,
    status: "published",
    essentialQuestion:
      "在固定算力下比较重建像素、预测表征和物体中心槽三条路线。",
    hook:
      "用第一幕自己录的 CarRacing 数据，同一批帧分别训 VAE 目标和 JEPA 目标，用同一个下游探针（预测路弯度）比表征质量；物体中心路线的真实验放到第 23 课。",
    outcomes: [
      "一份三路线比较报告，引用第 08、10、14 课的实验结果。",
    ],
    anchors: [{ name: "第一幕自录 CarRacing 数据 + 08/10/14 课对照报告" }],
    readingTime: 65,
    difficulty: "进阶",
    hardware: {
      minimum: "1×消费级 CUDA GPU",
      recommended: "1×24GB CUDA GPU",
      notes: "素材全部来自前面各课的自产实验，不引入新训练任务。",
    },
  },
  {
    id: "17",
    slug: "evaluating-world-models",
    shortTitle: "分别评测预测、生成与规划",
    unit: unitById.craft,
    status: "published",
    essentialQuestion:
      "一个世界模型“好”，到底是三种完全不同的好。你要哪种？",
    hook:
      "为 World Models、Dreamer 和 IRIS 建立统一评测，比较一步误差、多步漂移、视觉保真和控制分数。",
    outcomes: [
      "统一评测报告，至少包含一个四指标互相矛盾的案例。",
      "公开 benchmark 地图：Atari 100k、DMC、1X 挑战赛、WorldScore 各自度量什么、漏掉什么。",
    ],
    anchors: [{ name: "前四幕全部自产模型与数据" }],
    readingTime: 70,
    difficulty: "高级",
    hardware: {
      minimum: "CPU 起步（评测已有产物）",
      recommended: "1×24GB CUDA GPU",
      notes: "不训练新模型，评测对象全部来自前面各课。",
    },
  },
  {
    id: "18",
    slug: "scaling-study",
    shortTitle: "小模型的结论什么时候能外推",
    unit: unitById.craft,
    status: "published",
    essentialQuestion:
      "小数据小模型跑不出的能力，凭什么不能断言大模型也没有？反过来，什么时候能？",
    hook:
      "在 Dreamer 或 IRIS 设置上跑一个真的 mini scaling study：参数量、数据量、可见历史、状态表示四条轴选两条，每点多种子，同时看连续指标和能力门槛。",
    outcomes: [
      "mini scaling 报告，含“这个实验能说什么、不能说什么”一节。",
      "学会写“在这份数据、这个模型、这笔算力下观察到什么”。",
    ],
    anchors: [repos.dreamerTorch, repos.iris],
    readingTime: 75,
    difficulty: "研究级",
    hardware: {
      minimum: "1×24GB CUDA GPU（多种子串行）",
      recommended: "1×24GB；可选短租多卡放大网格",
      notes: "网格规模按预算裁剪，先保证每点多种子。",
    },
  },
  {
    id: "19",
    slug: "surgery-experiments",
    shortTitle: "替换一个模型组件",
    unit: unitById.craft,
    status: "published",
    essentialQuestion:
      "改结构的实验怎么设计才不白跑？预算、对照、预期效应、失败判据，缺一样都算白跑。",
    hook:
      "三选一动真刀：给 dreamerv3-torch 的 RSSM 换 Transformer 骨干；把 JEPA 目标接进 Dreamer 的重建位；给 IRIS 换连续 token。每个手术先写清改哪些文件、预算多少卡时、什么结果算失败。",
    outcomes: [
      "一次完整的结构手术报告；失败但诊断清楚同样合格。",
    ],
    anchors: [repos.dreamerTorch, repos.iris],
    readingTime: 80,
    difficulty: "研究级",
    hardware: {
      minimum: "1×24GB CUDA GPU",
      recommended: "1×24GB CUDA GPU（按手术预算控制卡时）",
      notes: "动刀之前必须先复现被改仓库的基线曲线。",
    },
  },
  {
    id: "20",
    slug: "spatial-3d-state",
    shortTitle: "从多张图得到持久的 3D 状态",
    unit: unitById.spatial,
    status: "published",
    essentialQuestion:
      "一帧画面不是世界。换个视角，房间和物体为什么不该跟着漂？",
    hook:
      "用 VGGT 把一组桌面照片变成相机位姿和点图，量遮挡面还在不在，并分清重建和世界模型。",
    outcomes: [
      "一份自己桌子的点图和误差表。",
      "书面区分：重建不是 $P(s_{t+1}|s_t,a_t)$。",
    ],
    anchors: [repos.vggt],
    readingTime: 70,
    difficulty: "进阶",
    hardware: {
      minimum: "消费级 GPU 或较大内存的 Mac（推理）",
      recommended: "1×24GB CUDA GPU",
      notes: "跑公开权重，不从头训练 VGGT。",
    },
  },
  {
    id: "21",
    slug: "persistent-4d",
    shortTitle: "边走边改的 4D 状态",
    unit: unitById.spatial,
    status: "published",
    essentialQuestion:
      "每帧都像真的，却不是同一个空间随时间变化，算哪门子世界模型？",
    hook:
      "用 CUT3R 在绕杯子走一圈的视频上做物体恒常测试：挡住、移走、转头，看状态还记不记得。",
    outcomes: [
      "一条带时间戳的漂移曲线。",
      "物体恒常测试记录。",
    ],
    anchors: [repos.cut3r],
    readingTime: 70,
    difficulty: "高级",
    hardware: {
      minimum: "消费级 GPU",
      recommended: "1×24GB CUDA GPU",
      notes: "体验档：跑公开权重，不从头训练。",
    },
  },
  {
    id: "22",
    slug: "foundation-video-wm",
    shortTitle: "视频基础模型什么时候才算世界模型",
    unit: unitById.spatial,
    status: "published",
    essentialQuestion:
      "Cosmos、Genie、Sora 类都会往后播。动作起不起作用？能规划吗？",
    hook:
      "用 DINO-WM 或同等特征动力学做动作对换，把 Cosmos 等系统放进开源三档 × 三分评测。",
    outcomes: [
      "动作对换是否分岔的记录。",
      "一张前沿系统评测表。",
    ],
    anchors: [repos.dinowm, repos.cosmos, repos.onexgpt],
    readingTime: 75,
    difficulty: "高级",
    hardware: {
      minimum: "1×24GB CUDA GPU（DINO-WM）；Cosmos 大模型不够就只读",
      recommended: "1×24GB CUDA GPU",
      notes: "Cosmos 训不动就标成只讲，不装复现。",
    },
  },
  {
    id: "23",
    slug: "object-centric-wm",
    shortTitle: "物体中心世界模型",
    unit: unitById.spatial,
    status: "published",
    essentialQuestion:
      "状态是一条向量时，桌上两样东西怎么分开推演？",
    hook:
      "在 C-SWM 官方小环境上复现槽化动力学，可视化槽，并对照向量 RSSM 做不到的事。",
    outcomes: [
      "C-SWM 方向性复现报告（论文复现 #6）。",
      "槽与物体的对应图。",
    ],
    anchors: [repos.cswm],
    readingTime: 75,
    difficulty: "高级",
    hardware: {
      minimum: "CPU / MPS / CUDA 都能跑",
      recommended: "1×消费级 CUDA GPU",
      notes: "官方小环境，这是第 16 课第三条路的动手课。",
    },
  },
  {
    id: "24",
    slug: "visual-foresight",
    shortTitle: "看见未来再动手",
    unit: unitById.embodied,
    status: "published",
    essentialQuestion:
      "会预测下一帧，怎样变成先在脑子里推一把杯子，再决定真不真推？",
    hook:
      "在 PushT 或桌面仿真里用动作条件模型做 MPC，对照动作盲基线；精读 DayDreamer，不假装复现全部真机。",
    outcomes: [
      "规划超过动作盲基线的记录（论文复现 #7）。",
      "DayDreamer 接口精读笔记。",
    ],
    anchors: [repos.daydreamer, repos.dinowm],
    readingTime: 80,
    difficulty: "高级",
    hardware: {
      minimum: "1×消费级 CUDA GPU",
      recommended: "1×24GB CUDA GPU",
      notes: "真机四机器人结果只讲不复现。",
    },
  },
  {
    id: "25",
    slug: "lerobot-imitation",
    shortTitle: "先模仿，再谈世界模型",
    unit: unitById.embodied,
    status: "published",
    essentialQuestion:
      "ACT 和 Diffusion Policy 没有显式动力学，桌宠能走多远？缺了什么？",
    hook:
      "用 LeRobot 回放公开桌面臂轨迹，训练 ACT 或 Diffusion Policy 小配置，对照世界模型规划的失败模式。",
    outcomes: [
      "一条能回放的数据集加训练记录（论文复现 #8）。",
      "模仿 vs 动力学对照笔记。",
    ],
    anchors: [repos.lerobot],
    readingTime: 80,
    difficulty: "高级",
    hardware: {
      minimum: "1×消费级 CUDA GPU；无真机用公开数据集",
      recommended: "1×24GB CUDA GPU；可选 SO-101",
      notes: "不把买手臂写成及格条件。",
    },
  },
  {
    id: "26",
    slug: "vla-vs-world-model",
    shortTitle: "VLA 和世界模型各管一段",
    unit: unitById.embodied,
    status: "published",
    essentialQuestion:
      "OpenVLA / π0 直接吐动作，世界模型吐未来。桌宠什么时候该用哪一个？",
    hook:
      "精读 OpenVLA 和 openpi 的接口与评测，能跑则跑一个小推理，并设计世界模型当安全过滤器。",
    outcomes: [
      "VLA 与世界模型的接口对照表。",
      "至少一处写明 VLA 成功不等于理解物理。",
    ],
    anchors: [repos.openvla, repos.openpi],
    readingTime: 70,
    difficulty: "研究级",
    hardware: {
      minimum: "阅读 + 可选 24GB 推理",
      recommended: "1×24GB CUDA GPU",
      notes: "训不动就标体验/只讲。π0.5 未开源，不能练。",
    },
  },
  {
    id: "27",
    slug: "sim-to-real",
    shortTitle: "仿真里会了，真桌子上为什么还摔",
    unit: unitById.embodied,
    status: "published",
    essentialQuestion:
      "接触、延迟、标定和安全约束，哪一样会把梦里的高分变成真机碰撞？",
    hook:
      "在 Genesis 或 MuJoCo 搭一张桌子，随机化摩擦，给规划器加碰撞安全过滤。",
    outcomes: [
      "一张 sim-to-real 失败清单。",
      "带安全过滤的规划记录。",
    ],
    anchors: [repos.genesis, repos.lerobot],
    readingTime: 75,
    difficulty: "高级",
    hardware: {
      minimum: "CPU / GPU 跑仿真",
      recommended: "1×24GB CUDA GPU",
      notes: "没有安全层不准上真机。",
    },
  },
  {
    id: "28",
    slug: "desk-pet-brief",
    shortTitle: "为什么桌宠是具身智能的入口",
    unit: unitById.deskpet,
    status: "published",
    essentialQuestion:
      "为什么不从仓库机械臂或人形开干，而从一张桌子上的小机器人开干？",
    hook:
      "把你的桌子写成 POMDP，选定摄像头档 / Reachy Mini / SO-101，拍 2 分钟桌面视频。",
    outcomes: [
      "一份填完的桌面 POMDP 表。",
      "2 分钟桌面视频和硬件档选择。",
    ],
    anchors: [repos.reachyMini, repos.lerobot],
    readingTime: 60,
    difficulty: "进阶",
    hardware: {
      minimum: "笔记本摄像头",
      recommended: "摄像头；可选 Reachy Mini 或 SO-101",
      notes: "几乎无训练。人格不是世界模型。",
    },
  },
  {
    id: "29",
    slug: "desk-perception",
    shortTitle: "把桌子看成状态",
    unit: unitById.deskpet,
    status: "published",
    essentialQuestion:
      "桌宠的状态至少要有物体、人的注意、自己的身体。像素和关节角怎样变成这三样？",
    hook:
      "从自己的桌面视频提出杯子位置、人脸朝向和头部姿态，做成时间表，做物体恒常检查。",
    outcomes: [
      "至少 1 分钟的状态日志。",
      "观察与推断的区分说明。",
    ],
    anchors: [repos.vggt, repos.cut3r],
    readingTime: 70,
    difficulty: "高级",
    hardware: {
      minimum: "笔记本摄像头",
      recommended: "摄像头 + 消费级 GPU",
      notes: "禁止用 LLM 看图写「桌上很乱」当作状态。",
    },
  },
  {
    id: "30",
    slug: "desk-world-model",
    shortTitle: "给自己的桌子训一个会听动作的世界模型",
    unit: unitById.deskpet,
    status: "published",
    essentialQuestion:
      "同一张桌子，头左转和手往杯方向伸，未来必须不一样。",
    hook:
      "在自己的桌面轨迹或 Genesis 桌面场景上训小世界模型，动作对换必须分岔。",
    outcomes: [
      "动作对换报告（论文复现 #9）。",
      "多步漂移图和动作盲负对照。",
    ],
    anchors: [repos.genesis, repos.dinowm],
    readingTime: 85,
    difficulty: "研究级",
    hardware: {
      minimum: "1×消费级 CUDA GPU；摄像头档用键盘当假身体",
      recommended: "1×24GB CUDA GPU",
      notes: "架构复用前面课已跑通的一种，不新发明。",
    },
  },
  {
    id: "31",
    slug: "interaction-memory",
    shortTitle: "人会不会看过来",
    unit: unitById.deskpet,
    status: "published",
    essentialQuestion:
      "桌宠的世界里最大的外生过程是人。人不是可控动作，但必须被预测。",
    hook:
      "训一个小头预测 1 秒后「人是否看镜头 / 手是否靠近杯子」，对照惯性基线，并加上短时记忆。",
    outcomes: [
      "人反应预测相对惯性基线的提升或失败分析。",
      "一段说明：人格不是动力学。",
    ],
    anchors: [{ name: "第 29-30 课的桌面日志" }],
    readingTime: 70,
    difficulty: "高级",
    hardware: {
      minimum: "笔记本摄像头",
      recommended: "摄像头 + 消费级 GPU",
      notes: "禁止用聊天模型扮演用户来生成假互动数据。",
    },
  },
  {
    id: "32",
    slug: "ship-desk-pet",
    shortTitle: "装上身体，做出五件会先想再做的行为",
    unit: unitById.deskpet,
    status: "published",
    essentialQuestion:
      "怎样证明桌宠用了世界模型，而不只是套了个表情包？",
    hook:
      "接上感知、世界模型、安全过滤和执行，完成物体恒常、对视、克制、动作对换展示、失败承认五件行为。",
    outcomes: [
      "五件行为的录像或日志，每件都查询了世界模型。",
      "一页后续研究提案，只写一个真瓶颈。",
    ],
    anchors: [repos.reachyMini, repos.lerobot],
    readingTime: 90,
    difficulty: "研究级",
    hardware: {
      minimum: "摄像头档：屏幕当脸 + 声音",
      recommended: "Reachy Mini 或 SO-101",
      notes: "没有查询世界模型的行为不算数。不把买机器写成及格条件。",
    },
  },
  {
    id: "33",
    slug: "embodiment-degrees",
    shortTitle: "给系统打具身程度",
    unit: unitById.embodied,
    status: "published",
    essentialQuestion:
      "具身智能有几种意思？怎样用已经会做的实验，给系统打 E0 到 E5，而不是贴标签？",
    hook:
      "拆开 Wilson 和 Ziemke 的分类，对齐 Brooks 与前向模型，再用七条是否题给已学系统和自己的桌宠打分。",
    outcomes: [
      "一份 E0 到 E5 打分表，至少四个系统，每档指向是否题。",
      "桌宠还差哪一档、升一档的最小实验（只选一个方向）。",
    ],
    anchors: [
      { name: "Wilson 2002；Ziemke 2003；Brooks 1991；Wolpert et al. 1995" },
    ],
    readingTime: 75,
    difficulty: "研究级",
    hardware: {
      minimum: "纸笔或网页实验",
      recommended: "不需要 GPU",
      notes: "第 24 课前先读第 1 到 5 节；第 32 课后再打自己的档。",
    },
  },
  {
    id: "34",
    slug: "leaderboard-map",
    shortTitle: "刷榜地图：三份榜各测什么",
    unit: unitById.frontier,
    status: "published",
    essentialQuestion:
      "2025-2026 年世界模型榜上的第一名，测的是生成、续写、指令跟随，还是规划？",
    hook:
      "把 WorldScore、WorldModelBench、Physics-IQ 还原成输入、输出和评分器，再用第 12 课三问给十个系统填表。",
    outcomes: [
      "一张开源三档 × 三分评测的地图，没测过的格子写未测。",
      "能说明 WorldScore 第一名不必能给桌宠做 MPC。",
    ],
    anchors: [repos.worldscore, repos.worldmodelbench, repos.physicsiq],
    readingTime: 70,
    difficulty: "研究级",
    hardware: {
      minimum: "笔记本即可读协议、抽样本",
      recommended: "1×24GB 只用于 clone 评测仓，不跑完整榜",
      notes: "闭源模型的分数只抄官方表，标成宣称。",
    },
  },
  {
    id: "35",
    slug: "cosmos3-omnimodal",
    shortTitle: "Cosmos 3 的全模态 MoT",
    unit: unitById.frontier,
    status: "published",
    essentialQuestion:
      "语言、图像、视频、声音、动作进同一套权重，哪些配置是 VLM，哪些才是世界模拟器？",
    hook:
      "拆 Cosmos 3 的 Reasoner 通路和 Generator 通路，对照第 22 课的 Predict2.5，按显存决定体验还是只讲。",
    outcomes: [
      "一张输入输出配置分类表。",
      "一段 Predict2.5 与 Cosmos 3 的对照笔记。",
    ],
    anchors: [repos.cosmos3],
    readingTime: 80,
    difficulty: "研究级",
    hardware: {
      minimum: "读报告和仓库文档",
      recommended: "1×24GB 仅当 Nano 权重官方显存够用",
      notes: "16B/64B 和完整后训练标成只讲。官方排行榜不是你复现的结果。",
    },
  },
  {
    id: "36",
    slug: "audio-world-model",
    shortTitle: "声音是观察还是配乐",
    unit: unitById.frontier,
    status: "published",
    essentialQuestion:
      "模型会出声，就等于它听得懂动作造成的物理声吗？",
    hook:
      "用 AVWM 的视听 POMDP 对照 Veo/Sora 2 一类音画生成器，在自己桌子上做一次声音对换。",
    outcomes: [
      "一份声音作为观测通道的对照笔记。",
      "能指出音画生成器过不了动作对换的原因。",
    ],
    anchors: [repos.cosmos3],
    readingTime: 70,
    difficulty: "高级",
    hardware: {
      minimum: "笔记本麦克风 + 摄像头",
      recommended: "1×24GB 仅当 Cosmos 音频推理官方显存够用",
      notes: "闭源音画模型只讲。麦克风是观察，喇叭可以是动作。",
    },
  },
  {
    id: "37",
    slug: "training-recipes",
    shortTitle: "flow matching 与 self-forcing",
    unit: unitById.frontier,
    status: "published",
    essentialQuestion:
      "同样的 DiT，换训练协议为什么能从整段往后编变成边看边播？",
    hook:
      "用第 03 课 MDN-RNN 量曝光偏差，再对照 CausVid / Self-Forcing / flow matching 的公开配方。",
    outcomes: [
      "一张训练配方对照表。",
      "一条 teacher forcing 对自由 rollout 的漂移曲线。",
    ],
    anchors: [repos.selfForcing, repos.ctallec],
    readingTime: 80,
    difficulty: "研究级",
    hardware: {
      minimum: "CPU 可做第 03 课对照",
      recommended: "1×24GB 尝试 Wan 1.3B 或 Self-Forcing 推理",
      notes: "训不动大视频模型就停，写清失败原因。",
    },
  },
  {
    id: "38",
    slug: "architecture-zoo",
    shortTitle: "架构动物园",
    unit: unitById.frontier,
    status: "published",
    essentialQuestion:
      "AR、DiT、MoT、JEPA、latent action 各自把状态放在哪，桌宠该借哪一块？",
    hook:
      "给桌宠写一份零件选型：编码器、状态、动力学、动作编码、规划器，用第 33 课是否题打一档。",
    outcomes: [
      "一份选型说明书，每个选择写清赌什么和 24GB 能不能跑。",
    ],
    anchors: [repos.iris, repos.diamond, repos.tinyworlds, repos.cosmos3],
    readingTime: 70,
    difficulty: "研究级",
    hardware: {
      minimum: "读已有仓库模块图",
      recommended: "不新训大模型",
      notes: "不要重复第 16 课的三条路线之争。",
    },
  },
  {
    id: "39",
    slug: "long-horizon-memory",
    shortTitle: "滑窗、状态、记忆库",
    unit: unitById.frontier,
    status: "published",
    essentialQuestion:
      "走开再回头，杯子还在不在？这是窗口长度问题还是状态问题？",
    hook:
      "对照滑窗崩坏、RSSM 漂移和 WorldMem 的记忆取回，Genie 3 的数分钟一致性标成宣称。",
    outcomes: [
      "三种记忆的对照表。",
      "能区分第 21 课的 4D 状态更新和生成式记忆库。",
    ],
    anchors: [repos.worldmem, repos.openOasis, repos.matrixGame],
    readingTime: 70,
    difficulty: "高级",
    hardware: {
      minimum: "引用第 12 课已有崩坏记录",
      recommended: "1×24GB 仅当 WorldMem 或 Matrix-Game 官方显存够用",
      notes: "Genie 3 无权重，不能练。",
    },
  },
  {
    id: "40",
    slug: "action-conditioned-industrial",
    shortTitle: "工业级动作条件",
    unit: unitById.frontier,
    status: "published",
    essentialQuestion:
      "把视频基础模型后训练成会听动作的模拟器，缺哪几步？",
    hook:
      "拆开 Cosmos Policy 的策略后训练和 DINO-WM 的动力学，画一条必须经过动作对换的流水线。",
    outcomes: [
      "后训练流水线检查表。",
      "能说出谁在学 P(s'|s,a)，谁在学 π(a|s)。",
    ],
    anchors: [repos.cosmos3, repos.dinowm, repos.onexgpt],
    readingTime: 70,
    difficulty: "高级",
    hardware: {
      minimum: "读文档 + 第 22 课已有 DINO-WM 记录",
      recommended: "1×24GB 复查动作编码落点",
      notes: "Cosmos 后训练多半只讲。不要重复 PushT 全流程。",
    },
  },
  {
    id: "41",
    slug: "driving-world-models",
    shortTitle: "驾驶世界模型",
    unit: unitById.frontier,
    status: "published",
    essentialQuestion:
      "多相机和结构化条件的驾驶世界模型，验证者和桌面世界模型差在哪？",
    hook:
      "读 GAIA-2 的条件列表：自车是动作，他车是外生，天气是条件。开源对照能跑再体验。",
    outcomes: [
      "条件分类表。",
      "给开环合成数据生成器打一档具身程度。",
    ],
    anchors: [repos.cosmos],
    readingTime: 65,
    difficulty: "研究级",
    hardware: {
      minimum: "读论文和官方博客",
      recommended: "开源驾驶模型仅当 24GB 官方够用",
      notes: "GAIA-2/3 无公开权重，只讲。",
    },
  },
  {
    id: "42",
    slug: "robot-foundation-wm",
    shortTitle: "机器人基础世界模型 vs VLA",
    unit: unitById.frontier,
    status: "published",
    essentialQuestion:
      "Reasoner、Generator、Policy 三套头，哪一个才是 P(s'|s,a)？",
    hook:
      "把 Cosmos 3、π0、OpenVLA、DINO-WM 和第 30 课桌面模型放进同一张三列表。",
    outcomes: [
      "三列表：理解头、生成头、动作头。",
      "桌宠最小接入：策略提议，世界模型过滤扫杯。",
    ],
    anchors: [repos.cosmos3, repos.openvla, repos.openpi, repos.dinowm],
    readingTime: 70,
    difficulty: "研究级",
    hardware: {
      minimum: "读文档，不新训",
      recommended: "不需要额外 GPU",
      notes: "VLA 默认 E3。查询了前向模型才进 E4。",
    },
  },
  {
    id: "43",
    slug: "interactive-game-wm",
    shortTitle: "可玩的世界：Genie 3 与 Matrix-Game",
    unit: unitById.frontier,
    status: "published",
    essentialQuestion:
      "实时交互加上数分钟一致性，开源系统现在能复现到哪一步？",
    hook:
      "Genie 3 只讲。Matrix-Game 和 open-oasis 按当前 README 决定体验还是精读。",
    outcomes: [
      "开源可玩系统的实测或诚实失败记录。",
      "Genie 3 宣称与可验证事实分开的表。",
    ],
    anchors: [repos.matrixGame, repos.openOasis],
    readingTime: 70,
    difficulty: "高级",
    hardware: {
      minimum: "读博客和论文",
      recommended: "1×24GB 仅当 Matrix-Game README 声明可交互",
      notes: "不把网页演示的流畅写成你测过的 24 fps。",
    },
  },
  {
    id: "44",
    slug: "physics-understanding",
    shortTitle: "懂物理怎么操作化",
    unit: unitById.frontier,
    status: "published",
    essentialQuestion:
      "画面真和物理对是同一根尺子吗？Physics-IQ 测的是续写还是规划？",
    hook:
      "用真实物理实验续写当答案，对照视觉质量和 Physics-IQ 分数是否同序。",
    outcomes: [
      "一份画面真、物理对、规划好的三列笔记。",
      "两条公开样本的对错预期。",
    ],
    anchors: [repos.physicsiq, repos.worldmodelbench],
    readingTime: 70,
    difficulty: "研究级",
    hardware: {
      minimum: "读论文和公开分数表",
      recommended: "CPU 跑评测仓入口，不跑闭源生成",
      notes: "分数只抄论文表或仓库 README。",
    },
  },
  {
    id: "45",
    slug: "objectives-and-takeaway",
    shortTitle: "目标函数动物园，带回桌子",
    unit: unitById.frontier,
    status: "published",
    essentialQuestion:
      "重建、JEPA、价值等价、flow、蒸馏各自在优化什么？24GB 该改第 30 课的哪一块？",
    hook:
      "从第九幕只选一个搬得动的改造带回桌面世界模型。做不成就写失败报告。毕业标准仍在 32/33。",
    outcomes: [
      "第九幕带回桌子的改造记录或失败记录。",
      "重申毕业在第 32 课，档在第 33 课。",
    ],
    anchors: [{ name: "第 30 课自制桌面世界模型" }],
    readingTime: 75,
    difficulty: "研究级",
    hardware: {
      minimum: "第 30 课已有产物",
      recommended: "1×24GB 训一小截对照",
      notes: "不要后训练 Cosmos 3。只选一个改造。",
    },
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

// 学习路线：只影响推荐顺序，不锁课。core 是全课主线，
// 其余按兴趣切入，第 01 课永远是共同起点。
export type LearningRouteId =
  | "core"
  | "latent"
  | "generative"
  | "jepa"
  | "embodied"
  | "frontier";

export interface LearningRoute {
  label: string;
  summary: string;
  lessonIds: readonly string[];
}

export const learningRoutes: Record<LearningRouteId, LearningRoute> = {
  core: {
    label: "主线全修",
    summary:
      "按课序学习：从 2018 年的 World Models，到桌宠毕业，再读 2025-2026 刷榜系统。",
    lessonIds: courseMetadata.map((course) => course.id),
  },
  latent: {
    label: "潜空间引擎",
    summary:
      "先跑通第一课，再集中学习 RSSM、Dreamer、MuZero、TD-MPC2，最后完成评测与改造三课。",
    lessonIds: ["01", "05", "06", "07", "08", "17", "18", "19"],
  },
  generative: {
    label: "生成式世界模型",
    summary:
      "比较 token、扩散、latent action，再读 2025-2026 的榜、Cosmos 3 和可玩世界。",
    lessonIds: ["01", "09", "10", "11", "12", "34", "35", "37", "43"],
  },
  jepa: {
    label: "JEPA 路线",
    summary:
      "实现 I-JEPA，接上动作与规划，评测 V-JEPA 2 公开权重，最后比较三条路线。",
    lessonIds: ["01", "13", "14", "15", "16"],
  },
  embodied: {
    label: "具身与桌宠",
    summary:
      "跑通第一课后进入空间、机器人学习和桌宠毕业设计。没有真机用摄像头档完成。",
    lessonIds: [
      "01",
      "20",
      "21",
      "22",
      "23",
      "24",
      "25",
      "26",
      "27",
      "28",
      "29",
      "30",
      "31",
      "32",
      "33",
    ],
  },
  frontier: {
    label: "刷榜与配方",
    summary:
      "跑通第一课后，直接读 2025-2026 的榜、全模态、声音、训练配方和架构。毕业仍在桌宠课。",
    lessonIds: [
      "01",
      "12",
      "17",
      "22",
      "33",
      "34",
      "35",
      "36",
      "37",
      "38",
      "39",
      "40",
      "41",
      "42",
      "43",
      "44",
      "45",
    ],
  },
};
