export type CourseDifficulty = "入门" | "进阶" | "高级" | "研究级";

export type CourseUnitId =
  | "mechanism"
  | "realtime"
  | "vision"
  | "backbone"
  | "alignment"
  | "frontier"
  | "vl-foundation"
  | "vla"
  | "embodied-agent"
  | "world-model"
  | "spatial-body"
  | "vla-post"
  | "native-unified"
  | "omni-ops"
  | "embodied-omni"
  | "gen-native"
  | "reason-agent"
  | "data-deploy"
  | "domain-research";

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
    id: "mechanism",
    order: 1,
    title: "基础结构与可信复现",
    question: "小型 Omni 如何连接文本、语音和离散声码，并验证每一步的输入与输出？",
  },
  {
    id: "realtime",
    order: 2,
    title: "实时语音与全双工",
    question: "系统如何在用户仍在说话时持续监听、判断话轮并处理打断？",
  },
  {
    id: "vision",
    order: 3,
    title: "图像、视频与视觉预算",
    question: "小模型如何处理高分辨率图像和视频时序，同时控制视觉 token 数量？",
  },
  {
    id: "backbone",
    order: 4,
    title: "现代骨干与八卡系统",
    question: "如何公平测量 MoE、Mamba、长上下文和分布式并行带来的质量与效率变化？",
  },
  {
    id: "alignment",
    order: 5,
    title: "联合训练与后训练",
    question: "如何安排多模态数据和训练目标，使各项能力同时提高并控制遗忘？",
  },
  {
    id: "frontier",
    order: 6,
    title: "系统集成与前沿统一",
    question: "如何组合开放组件，并验证多模态理解、语音生成和图像生成的实际能力边界？",
  },
  {
    id: "vl-foundation",
    order: 7,
    title: "图文对齐与视觉地基",
    question: "冻结视觉编码器从哪来，标准 VLM 怎么训，以及答对了是否等于看见了位置？",
  },
  {
    id: "vla",
    order: 8,
    title: "视觉–语言–动作",
    question: "动作如何成为下一种模态，并在数据、表示、闭环和评测上被拆开验证？",
  },
  {
    id: "embodied-agent",
    order: 9,
    title: "屏幕与跨身体智能",
    question: "屏幕点击和机械臂动作能否共用一套空间接地？",
  },
  {
    id: "world-model",
    order: 10,
    title: "世界模型",
    question: "预测像素还是预测表征，世界模型当数据引擎还是当控制器？",
  },
  {
    id: "spatial-body",
    order: 11,
    title: "深度、导航与全身",
    question: "缺深度、换底盘、升高动作维数时，同等预算丢掉什么？",
  },
  {
    id: "vla-post",
    order: 12,
    title: "试错、记忆与切断",
    question: "可验证奖励、子目标栈和力超限切断如何接到已有的双工状态机？",
  },
  {
    id: "native-unified",
    order: 13,
    title: "离散统一与输出日程",
    question: "图像离散 token 如何与文本共用 next-token，一段回复里字、图、声如何排期？",
  },
  {
    id: "omni-ops",
    order: 14,
    title: "听、读、检索、上线与评测",
    question: "音频理解、文档版面、长视频检索、推理调度和六类评测如何分开记账？",
  },
  {
    id: "embodied-omni",
    order: 15,
    title: "听说动手运行时",
    question: "语音打断和手臂重规划能否共用一张状态表，又不把已发生的接触当成可回放缓冲？",
  },
  {
    id: "gen-native",
    order: 16,
    title: "视频与三维生成",
    question: "生成下一帧和看懂这一段、生成网格和高斯，各自的损失和评测能否分开？",
  },
  {
    id: "reason-agent",
    order: 17,
    title: "分步推理、工具与跨会话记忆",
    question: "看见图以后何时该写推理、调用工具，跨会话该存像素还是存摘要？",
  },
  {
    id: "data-deploy",
    order: 18,
    title: "合成数据、端侧与出处",
    question: "合成补的是分布还是条数，量化先伤哪类 token，生成能否只用 L2，图进训练集要留下什么？",
  },
  {
    id: "domain-research",
    order: 19,
    title: "领域约束与课的活法",
    question: "医学图文和非语音音频为何不能套自然图像或语音 codec，新论文如何接到验收口径？",
  },
] as const satisfies readonly CourseUnit[];

const unitById = Object.fromEntries(
  courseUnits.map((unit) => [unit.id, unit]),
) as Record<CourseUnitId, CourseUnit>;

export const courseMetadata: CourseMetadata[] = [
  {
    id: "01",
    slug: "baseline-reproduction",
    shortTitle: "冻结可信基线",
    unit: unitById.mechanism,
    essentialQuestion:
      "把代码、数据、种子和推理参数全固定之后，重跑能得到同一条 Thinker–Talker（想的脑、说的嘴）链路吗？每个 token 和每个延迟数字，你都说得清来历吗？",
    hook:
      "同样的配置跑两次，结果却不一样？那就得挨个排查：样本、mask、随机种子、checkpoint 和时间统计，总有一处没固定。",
    outcomes: [
      "亲手画出 1 路文本和 8 路 Mimi code（声音压成的整数编号）的训练、delay 和还原时间线。",
      "拿到一条生成 case，能顺藤摸瓜查回它的数据行、配置、checkpoint、随机种子和有效 loss 区间。",
      "分清 TTFT、TTFA、RTF 和端点检测这几种延迟各测什么，还能证明记运行日志不会改变 logits。",
      "做出后面十九课都要反复用的 baseline-v1、golden cases 和回归检查。",
    ],
    misconception: {
      myth: "loss 能降、WebUI 能出声，复现就算完成了。",
      truth:
        "那只说明程序跑起来了。要让人信这份复现，还得锁死输入和版本、讲清训练目标怎么摆、试过断点续训，并且每个指标、每条 case 都能查到出处。",
    },
    prerequisites: [
      "会跑基本的 Python 和 PyTorch 命令。",
      "知道 causal language model（只看前文、预测下一个词的模型）的输入和 next-token target 长什么样。",
      "不用先学任何课；这就是整门课的起点。",
    ],
    labId: "lab-01-baseline-forensics",
    readingTime: 55,
    difficulty: "入门",
    hardware: {
      minimum: "1×24GB CUDA GPU",
      recommended: "1×24GB；8 卡仅用于并行 seed 与评测",
      notes: "mini 训练方案只是教学用的缩小版，代表不了完整 minimind-3o 的能力。",
    },
    learningMode: ["代码考古", "128 样本过拟合", "确定性评测", "逐 case 取证"],
  },
  {
    id: "02",
    slug: "multimodal-connector",
    shortTitle: "跨模态连接器",
    unit: unitById.mechanism,
    essentialQuestion:
      "把 encoder、LLM、参数量和训练 token 都大致固定，learned-query connector（用一小组可学的查询向量去挑重点的连接器）真比逐 token MLP 换来更好的信息—token—延迟平衡吗？",
    hook:
      "连接器是眼睛耳朵和大脑之间的转接线：既要把 encoder 的输出尺寸对上 LLM，还得决定给 LLM 塞多少图像或音频信息。",
    outcomes: [
      "分清连接器要干的三件事：维度投影、分布对齐、信息瓶颈，别混成一锅。",
      "用同一套接口写出 MLP、Perceiver Resampler 和轻量 Q-Former 三种连接器。",
      "用打乱或干脆抽掉图片音频的负对照实验，验证模型是不是真在看、真在听。",
      "在参数量和 token 预算都可比时，画出 accuracy–tokens–latency 的 Pareto 图（哪种方案在哪项上占优，一图看清）。",
    ],
    misconception: {
      myth: "Q-Former 比两层 MLP 复杂，所以肯定更强。",
      truth:
        "复杂连接器可能过拟合、拖慢速度，或者弄丢长音频里的信息。要下结论，得在参数量和 token 数相近时，用没参与训练的测试集加模态依赖实验说话。",
    },
    prerequisites: [
      "理解 attention 里的 Q/K/V 和 softmax。",
      "能看懂 frozen encoder feature 和 LLM embedding 的 shape（张量形状）。",
      "推荐用实验 01 的 baseline-v1；直接从官方 checkpoint 开始也行。",
    ],
    labId: "lab-02-connector-bottleneck",
    readingTime: 55,
    difficulty: "进阶",
    hardware: {
      minimum: "1×24GB CUDA GPU",
      recommended: "1–2×24GB；8 卡用于三臂多 seed",
      notes: "固定 encoder、LLM 和模态 token 上限，别把多花的预算错当成结构本身的收益。",
    },
    learningMode: ["结构拆解", "三臂消融", "模态负对照", "Pareto 决策"],
  },
  {
    id: "03",
    slug: "audio-codec",
    shortTitle: "可插拔音频 Codec",
    unit: unitById.mechanism,
    essentialQuestion:
      "目标波形相同、有效码率相近、Talker 预算一样时，换一个 codec（把声音压成整数编号、也能解回声音的压缩器），重建音质、token 好不好预测、流式延迟这三样会怎么一起变？",
    hook:
      "一个 codec 重建出来的声音更好听，产出的 token 却可能更密、更难预测——最后 Omni 反而说得更慢、内容更差。",
    outcomes: [
      "用帧率、码本数和词表大小算出 nominal bitrate（名义码率），并明白它和有效熵是两回事。",
      "写出一个 AudioCodec adapter，把流式状态、采样率和 special token 的约定都管起来。",
      "先单独比 codec 本身，再用相同预算重训 Talker，两步分开，别让变量搅在一起。",
      "把重建、WER、说话人保持、token PPL、TTFA 和边界伪影放在同一张报告里看。",
    ],
    misconception: {
      myth: "codec 重建指标最高，就最适合 Omni。",
      truth:
        "Omni 要的是 rate–distortion–predictability–latency（码率、失真、可预测性、延迟）四样端到端的平衡；后层 code 好不好预测、流式边界处理得怎么样，同样决定成败。",
    },
    prerequisites: [
      "理解向量量化和 residual vector quantization（多层残差量化，逐层补细节）。",
      "能摆弄 waveform、采样率、frame/hop 和流式 buffer。",
      "手上得有带原始 assistant waveform 的数据，只存旧 codec 编号是不够的。",
    ],
    labId: "lab-03-codec-pareto",
    readingTime: 65,
    difficulty: "高级",
    hardware: {
      minimum: "1–2 张 CUDA GPU 做 codec 初筛",
      recommended: "4–8 卡重训同规模 Talker",
      notes: "先比重建质量做初筛；只有过了初筛的 codec 才值得花卡时进 Talker 训练。",
    },
    learningMode: ["公式推导", "接口一致性测试", "听觉 A/B", "端到端公平比较"],
  },
  {
    id: "04",
    slug: "multicodebook-talker",
    shortTitle: "多码本 Talker",
    unit: unitById.mechanism,
    essentialQuestion:
      "让模型开口说话，难点到底在哪：是跨时间把一帧帧接顺，还是同一帧里 q0→q7 这 8 个编号之间的先后依赖？",
    hook:
      "一帧声音里装着 8 个 RVQ code（8 层码本各出一个编号）。8 个一起生成，还是按 q0→q7 一个个生成？两种走法条件关系不同，延迟也不同。",
    outcomes: [
      "把 8×T 的 delay schedule（错位排布）和它的逆变换画出来、单测过。",
      "动手实现两种嘴：同帧 8 个头各说各的（independent heads），和一个轻量 depth decoder 逐层接力。",
      "分清两种进步：teacher-forced CE（喂标准答案时的分数）变好，和 free-running（自己接自己）生成的音频变好。",
      "拿质量、串行深度、RTF 和 TTFA 四个数，选定 Talker 的拓扑。",
    ],
    misconception: {
      myth: "后层码本的 PPL 降了，声音就一定更好。",
      truth:
        "喂标准答案算出的 token 指标未必换来波形上的收益，还可能搭进去帧内串行延迟；必须做 free-running 生成和人耳感知评测。",
    },
    prerequisites: [
      "掌握 RVQ 的层级依赖、teacher forcing 和 exposure bias（训练时喂答案、推理时只能靠自己的落差）。",
      "能看懂 tensor 里的 codebook、frame 和 sequence 三个维度各是什么。",
      "codec 沿用 Mimi 就行，不要求先做完实验 03。",
    ],
    labId: "lab-04-talker-topology",
    readingTime: 65,
    difficulty: "高级",
    hardware: {
      minimum: "1×24GB CUDA GPU 做小规模预实验",
      recommended: "4–8 卡完成三臂多 seed",
      notes: "三臂对照的参数、active FLOPs、训练步数和解码设置必须一一对齐，否则比了也白比。",
    },
    learningMode: ["时序可视化", "拓扑实现", "exposure stress test", "音质—延迟 Pareto"],
  },
  {
    id: "05",
    slug: "streaming-listener",
    shortTitle: "因果流式 Listener",
    unit: unitById.realtime,
    essentialQuestion:
      "用户话还没说完，系统能一边听一边持续更新语音和 Thinker 的状态吗？而且你能证明它任何时刻都没偷看还没到的 waveform 吗？",
    hook:
      "输出的音频能边生成边播，不代表输入也是流式的。Listener（负责听的那半边）必须按声音到达的顺序，一点一点往前推状态。",
    outcomes: [
      "分清三种听法：整段离线、切块但每次重算、真正带状态的因果 encoder。",
      "实现 init_state、push_chunk、finalize 三件套，外加一条查得到账的 available_at 时间戳。",
      "把双向的 SenseVoice 当老师，蒸馏出一个只看过去的 causal student，再验证相同前缀不受后面内容影响。",
      "让新进来的 audio token 持续更新 Thinker 的 KV，而不是等人说完再把整段音频重新编一遍。",
    ],
    misconception: {
      myth: "把完整音频切成 320ms 的小块再喂给模型，这就是 streaming 了。",
      truth:
        "只要块内或特征预计算偷看了未来、每块都重算完整前缀、或者最终回答还是重新编码全句，就没有真正可复用的因果状态。",
    },
    prerequisites: [
      "理解 attention cache、卷积状态和 receptive field（感受野：一个输出能看到多远的输入）。",
      "知道 lookahead（提前看一小段）也得算进算法延迟里。",
      "需要原始的用户 waveform；实验 02 的 connector 不是前置条件。",
    ],
    labId: "lab-05-causality-proof",
    readingTime: 70,
    difficulty: "高级",
    hardware: {
      minimum: "1×24GB CUDA GPU",
      recommended: "1–4 卡训练 student；8 卡并行 chunk/lookahead/seed",
      notes: "实时回放必须按声音本来到达的时间一步步推进，不能一口气把整个文件读进来。",
    },
    learningMode: ["因果性证明", "teacher–student 蒸馏", "状态机实现", "真实时间 replay"],
  },
  {
    id: "06",
    slug: "turn-policy",
    shortTitle: "学习式话轮策略",
    unit: unitById.realtime,
    essentialQuestion:
      "学出来的话轮策略，能同时少抢话、少干等、别把“嗯嗯”当打断，还得在用户真要插话时立刻反应过来吗？",
    hook:
      "固定的静音阈值只知道“声音停了”，分不清用户是停下来想词，还是真的说完了。",
    outcomes: [
      "把五件事掰开：检测有没有人声、判断一句话说完没、该不该接话、“嗯嗯”式附和、用户真打断。",
      "做出双声道事件标注，并实现 HOLD、TAKE_TURN、BACKCHANNEL、BARGE_IN 四个动作（憋住、接话、附和、被插话让位）。",
      "用事件窗口、概率校准和 hysteresis（迟滞：加缓冲防抖动）来评估，别用被 HOLD 占大头带偏的 frame accuracy。",
      "用合成反事实和 blind replay（盲评回放）检验它在真实会话里的表现。",
    ],
    misconception: {
      myth: "VAD 够准、静音阈值调好，话轮问题就解决了。",
      truth:
        "声学上的“有没有声”不包含话说完没有、谁在扮演什么角色、话里的意图；话轮策略需要时序、语义，外加校准过的动作约定。",
    },
    prerequisites: [
      "理解二分类、多分类、概率校准和类别不平衡。",
      "能处理双声道音频和事件时间戳。",
      "特征提取器用因果的或只读前缀的都行，不要求先做完实验 05。",
    ],
    labId: "lab-06-turn-events",
    readingTime: 70,
    difficulty: "高级",
    hardware: {
      minimum: "1 张 CUDA GPU 可训练策略头",
      recommended: "1–4 卡训练；8 卡并行数据、seed 与回放",
      notes: "这课最贵的通常是靠谱的双声道标注，不是模型参数。",
    },
    learningMode: ["事件标注", "策略头训练", "校准实验", "反事实会话回放"],
  },
  {
    id: "07",
    slug: "full-duplex-routing",
    shortTitle: "真双工 Routing",
    unit: unitById.realtime,
    essentialQuestion:
      "助手正说着话，用户新说的话能持续进入语义状态，让系统在 Continue（接着说）、Pause（先停停）、Replan（换个说法）里做选择吗？还是只会一刀切地取消？",
    hook:
      "一听到人声就停止生成，那只是做了个“取消”按钮。全双工要求嘴上在输出时，耳朵进来的新内容还在持续更新状态、改变后面的决定。",
    outcomes: [
      "搭一条双流 wall-clock（真实挂钟时间线），把 captured、available、consumed、emitted、played 五个时刻都记上账。",
      "实现两种路由：通道直接融合，和带门控的交叉注意力记忆。",
      "设计可抢占的 microstep scheduler、带版本号的状态、能撤回的播放 buffer，以及过期输出的处理。",
      "在“嗯嗯”附和、旁人说话、明确打断、不该打断这四类场景上逐 case 验收。",
    ],
    misconception: {
      myth: "助手说话时能被 VAD 打断，就是 full duplex。",
      truth:
        "full duplex 要求输出期间输入照样往前走、还能影响语义决策；说自己会 PAUSE/RESUME 和 REPLAN，得拿出状态变化和新内容的证据。",
    },
    prerequisites: [
      "理解增量生成的 prefill/decode、KV cache 和异步队列。",
      "能看懂 causal listener 和双声道时间轴。",
      "用本课自带的最小 listener/control head 就能独立开始，不强制先做实验 05/06。",
    ],
    labId: "lab-07-duplex-replay",
    readingTime: 90,
    difficulty: "研究级",
    hardware: {
      minimum: "4 张 CUDA GPU 完成缩小训练",
      recommended: "4–8 卡训练；低延迟单会话推理优先单卡细粒度调度",
      notes: "AEC（回声消除）、jitter、播放 buffer、wall-clock deadline 这些工程细节，都算模型结论的一部分，躲不开。",
    },
    learningMode: ["双流时间轴", "异步系统实现", "routing 对照", "live soak test"],
  },
  {
    id: "08",
    slug: "dynamic-vision",
    shortTitle: "动态分辨率与 M-RoPE",
    unit: unitById.vision,
    essentialQuestion:
      "保住宽高比的动态切片，配上二维位置编码，在同样的 token 预算下，真能把 OCR、小目标和多图理解做得更好吗？",
    hook:
      "把一张细长的收据硬缩到 256×256，上面的小字直接糊掉。这课就比一比：动态切片能不能在同样的 token 预算里把这些细节留住。",
    outcomes: [
      "实现按像素预算规划的动态 tile planner（切块规划器）加一张全局缩略图。",
      "给整图、切块和二维坐标定好明确的位置字段，接进 M-RoPE（二维旋转位置编码）。",
      "在动态 batch 里把变长 mask 和 position ids 拼对。",
      "用坐标打乱实验和同 token 预算对照，分清收益来自分辨率还是来自位置编码。",
    ],
    misconception: {
      myth: "分辨率越高，模型一定越强。",
      truth:
        "tile 切得多，token、prefill 和显存也跟着涨；只有预算对齐、坐标负对照、分层任务结果三样都在手，才能说机制真有效。",
    },
    prerequisites: [
      "理解 patch embedding、RoPE 和 attention position ids。",
      "能处理各种宽高比的图像和动态 batch。",
      "视觉端用冻结的 SigLIP2 和固定 connector；不要求先做完实验 02–07。",
    ],
    labId: "lab-08-vision-tiling",
    readingTime: 65,
    difficulty: "进阶",
    hardware: {
      minimum: "1×24GB CUDA GPU 做功能验证",
      recommended: "2–4×24GB 做标准实验",
      notes: "connector 和 backbone 都固定，别让连接器的变化混进动态视觉的结论里。",
    },
    learningMode: ["几何可视化", "position 单测", "同预算消融", "OCR 逐 case"],
  },
  {
    id: "09",
    slug: "native-video",
    shortTitle: "原生视频与 AV 对齐",
    unit: unitById.vision,
    essentialQuestion:
      "帧数和 LLM token 预算都一样时，模型是真用上了时间先后和声画对应，还是只在挨个认单帧画面？",
    hook:
      "把帧的顺序打乱、音轨换错，答案居然没变？那所谓的“视频理解”，多半只是在认几张静态图。",
    outcomes: [
      "分清三种视频前端：直接拼帧、后接时序 adapter、一开始就用 tubelet/Conv3D 看时空块。",
      "给每一帧和每段音频 chunk 建一条共享的毫秒时间轴。",
      "构造乱序、倒放、错配音轨这类不泄露答案的负对照。",
      "画出 fps、token、质量、短暂事件召回和延迟之间的 Pareto 取舍图。",
    ],
    misconception: {
      myth: "均匀抽几帧、把 token 拼起来，就已经是原生视频模型了。",
      truth:
        "拼帧只是必要的基线；结构和评测都得对顺序、状态变化、声画绑定敏感，时序建模才算立住了。",
    },
    prerequisites: [
      "理解图像 encoder 的 token 和基本的时序建模。",
      "数据里必须存好 frame_time_ms 和 audio_span_ms 两个时间字段。",
      "没做实验 08 就用固定的 256×256 帧，别临时把动态视觉这个变量掺进来。",
    ],
    labId: "lab-09-temporal-counterfactuals",
    readingTime: 65,
    difficulty: "高级",
    hardware: {
      minimum: "1×24GB 做 4–8 帧小规模预实验",
      recommended: "4–8 卡做标准时序实验",
      notes: "三臂对照必须共用同一套抽帧、帧数、token 上限和冻结的视觉/音频前端，否则比不出真差异。",
    },
    learningMode: ["时间轴建模", "三类前端对照", "负例构造", "AV 依赖审计"],
  },
  {
    id: "10",
    slug: "video-token-reduction",
    shortTitle: "视觉 Token Reduction",
    unit: unitById.vision,
    essentialQuestion:
      "保留率相同的前提下，按画面变化或相似度来压缩视觉 token，能比随机丢和均匀丢更好地保住 OCR、小目标和短暂事件，还真把端到端延迟压下来吗？",
    hook:
      "把视觉 token 砍掉一半，系统未必就快了：reducer 自己也要算力，prefill、显存、各类任务的退化还得分开查。",
    outcomes: [
      "搞清 pooling、merging、pruning、sampling 四种压法各把成本花在哪个环节。",
      "实现 EVS 和 similarity merge（按相似度合并），并保住每个 token 原来的时空 position id。",
      "按 100/75/50/25% 四档保留率做扫描，配上随机、均匀和内容感知三组对照。",
      "把 decode、vision、reducer、prefill、generation 的延迟一段段拆开计。",
    ],
    misconception: {
      myth: "理论 FLOPs 或输入 token 降了，系统就更快了。",
      truth:
        "在 encoder 之后剪枝，省不掉视频解码和 vision tower 的开销，reducer 本身也要花时间；必须实测端到端和分阶段的 wall-clock。",
    },
    prerequisites: [
      "理解视觉 token 的 frame、x、y 和原始位置含义。",
      "手上已有确定性的图像/视频前端和一个不压缩的 baseline。",
      "用逐帧 SigLIP2 就能独立做，不强制沿用实验 09 的赢家。",
    ],
    labId: "lab-10-token-pareto",
    readingTime: 50,
    difficulty: "高级",
    hardware: {
      minimum: "1×24GB 做 inference-only reduction",
      recommended: "2–8 卡做 uptraining 与多 budget sweep",
      notes: "单图 64 token 太少，压了也没多大意思；真正值钱的场景是高分辨率多 tile 或多帧视频。",
    },
    learningMode: ["成本明细", "reducer 实现", "同预算负对照", "分层退化分析"],
  },
  {
    id: "11",
    slug: "tiny-moe",
    shortTitle: "现代 Tiny MoE",
    unit: unitById.backbone,
    essentialQuestion:
      "每个 token 实际动用的 FFN 计算量差不多时，routed 加 shared experts（按需派活的专家＋人人都过的公共专家）带来的提升，是 dense 对照解释不了的真容量收益吗？",
    hook:
      "MoE 的好处是总参数变多、每个 token 的计算量却不用同步涨；但路由偏科、专家躺平、token 被丢弃，任何一样都能让结论作废。",
    outcomes: [
      "分清三笔账：total parameters、active parameters、真实 active FLOPs。",
      "搭起三组公平对照：dense-iso-active、dense-iso-total、routed+shared。",
      "盯住 expert load、entropy、overflow、dead expert 和模态×专家的分布。",
      "单卡 reference 验证通过后，再搬到 8 卡 Expert Parallel 上。",
    ],
    misconception: {
      myth: "MoE 分数比原来的小 dense 高，就证明稀疏路由有效。",
      truth:
        "它还得跟 iso-active 和 iso-total 两组对照掰手腕，并证明收益不是多出来的参数、token drop 或牺牲少数模态换的。",
    },
    prerequisites: [
      "理解 SwiGLU/FFN、softmax top-k 和梯度训练。",
      "能记下每个 token 的 modality id 和 loss mask。",
      "不依赖视觉课程；本课把 attention、connector、codec 和数据 mixture 全部固定。",
    ],
    labId: "lab-11-router-observatory",
    readingTime: 55,
    difficulty: "高级",
    hardware: {
      minimum: "1×24GB 完成功能与 iso 对照",
      recommended: "8 卡完成 Expert Parallel",
      notes: "先在单卡上把专家路由的数值算对，再去测 all-to-all 通信的性能，顺序别反。",
    },
    learningMode: ["参数量明细", "路由可视化", "因果负对照", "EP 迁移"],
  },
  {
    id: "12",
    slug: "mamba-attention-hybrid",
    shortTitle: "Mamba-2 × Attention",
    unit: unitById.backbone,
    essentialQuestion:
      "参数和训练 token 都配平之后，hybrid（大部分层换 Mamba、留几层 Attention）真能在长序列上省 cache/HBM 或提吞吐，还不丢 Attention 那种精确翻旧账的本事吗？",
    hook:
      "O(n) 只是复杂度，不是延迟的保票；序列太短、kernel 不给力，或者任务需要精确检索时，纯 Attention 反而可能更合算。",
    outcomes: [
      "能讲明白 selective SSM、scan/recurrent 两种等价算法和 Mamba-2 的 SSD。",
      "实现可配置的 attention/Mamba 层排布，以及两种层各自不同的 inference state。",
      "把非 embedding 参数、训练 token 和长度分布配平到可比。",
      "真实 QA、needle/copy 检索、prefill、decode、KV cache、SSM state 一起测，一个不落。",
    ],
    misconception: {
      myth: "Mamba 理论上是线性的，所以长上下文一定更快、更好。",
      truth:
        "硬件 kernel、序列长度、要不要精确翻旧账，三者一起决定结果；hybrid 值不值，得靠真实的质量—吞吐—cache 曲线说话。",
    },
    prerequisites: [
      "掌握 self-attention、KV cache 和自回归生成。",
      "理解基本的状态空间递推（上一步状态加当前输入，滚出下一步）。",
      "CUDA 环境能跑官方 Mamba-2 kernel；本课把 MoE 关掉。",
    ],
    labId: "lab-12-hybrid-mixer",
    readingTime: 60,
    difficulty: "研究级",
    hardware: {
      minimum: "1×24GB 做 8 层小规模预实验",
      recommended: "4–8 卡做标准长序列实验",
      notes: "短、中、长序列都得测，别只挑理论上占便宜的区间报数。",
    },
    learningMode: ["公式到 kernel", "结构配平", "异构 cache 实现", "长序列压力测试"],
  },
  {
    id: "13",
    slug: "distributed-8gpu",
    shortTitle: "八卡训练系统",
    unit: unitById.backbone,
    essentialQuestion:
      "FSDP2、EP、CP 这几种并行切法，能在不动 global batch、不改优化语义的前提下，把单卡显存压下来、把 useful-token 吞吐提上去吗？",
    hook:
      "八个训练进程能起来，只说明通信握上手了。梯度对不对、loss 怎么归一、padding 怎么算、checkpoint 能不能恢复，都还得一项项验。",
    outcomes: [
      "分清 DP、FSDP、TP、PP、EP、CP 六种并行各切什么东西、走哪种 collective 通信。",
      "用 global useful tokens 固定每次更新的语义，别拿 micro batch 凑合近似。",
      "先用 FP32 微型模型把计算算对，再检查 BF16 正式模型的数值是否一致。",
      "测 strong scaling、MFU、通信占比、p95 step time，还有换并行度后能否照常恢复训练。",
    ],
    misconception: {
      myth: "吞吐上去了、最终 loss 也差不多，并行实现就是对的。",
      truth:
        "样本顺序、global batch、loss 归一方式、token drop，随便哪一个变了都在悄悄改训练本身；必须一步步比对 forward、gradient 和 update。",
    },
    prerequisites: [
      "必须有同一台机器上的 8 张 CUDA GPU 可用。",
      "理解 optimizer state、collective 通信和 NCCL process group。",
      "官方 MoE checkpoint 的单卡 reference 必须先跑通；实验 11 不是运行依赖。",
    ],
    labId: "lab-13-parity-before-speed",
    readingTime: 65,
    difficulty: "研究级",
    hardware: {
      minimum: "同节点 8 张 CUDA GPU",
      recommended: "8×80GB NVSwitch/H100；24GB/48GB 也可做分层协议",
      notes: "GPU 型号、互联、driver、CUDA、NCCL、存储吞吐，这些环境信息一样都不能少记。",
    },
    learningMode: ["并行拓扑推演", "数值 parity", "profiler 分析", "checkpoint reshard"],
  },
  {
    id: "14",
    slug: "long-context-curriculum",
    shortTitle: "长上下文课程",
    unit: unitById.backbone,
    essentialQuestion:
      "按 8K→32K→128K 一步步把窗口拉长，能不能比一上来就 128K 更稳地学会“用上远处的跨模态证据”，还不把短上下文的本事忘掉？",
    hook:
      "把 max_position_embeddings 改成 131072，只说明张量可能塞得下；模型能不能找到 100K token 之外的证据，是另一回事。",
    outcomes: [
      "把三件事分开验证：窗口能跑、位置能外推、证据能被用上。",
      "弄清 RoPE、YaRN 各管到哪，以及训练长度 provenance（这模型到底见过多长的序列）的边界。",
      "构造防抄近路的长距离检索、跨段推理和模态 ablation 任务。",
      "用相同训练 token 比 direct 和 progressive 两种扩窗课程，并检查短任务有没有退步。",
    ],
    misconception: {
      myth: "配置写着支持 128K，needle 测试也过了，就说明模型有长上下文能力。",
      truth:
        "配置容量不等于训练历史，单点捞针也算不上稳健的证据使用；得有 provenance、防捷径任务、跨模态依赖和短上下文回归四样齐全。",
    },
    prerequisites: [
      "理解 RoPE、attention mask、KV cache 和变长 packing。",
      "能导出 token_type、time_index 和 loss_mask。",
      "不要求先做实验 11–13；默认训练方案用 DDP，且 CP=1。",
    ],
    labId: "lab-14-evidence-at-distance",
    readingTime: 85,
    difficulty: "研究级",
    hardware: {
      minimum: "8 卡用于 32K/128K 训练；显存不足时，只报告实际验证通过的长度",
      recommended: "8×80GB 执行完整 128K sweep",
      notes: "连续 50 步的显存验收没过，就老实标成未完成；不许拿配置里写的长度顶替实测结果。",
    },
    learningMode: ["provenance 审计", "位置编码单测", "反捷径数据构造", "三 seed Pareto"],
  },
  {
    id: "15",
    slug: "joint-multimodal-sft",
    shortTitle: "现代 Joint SFT",
    unit: unitById.alignment,
    essentialQuestion:
      "训练 token 一样多时，“先预热连接器、再混着练、最后回放旧任务”这套三段式，真比按顺序一个个练或一股脑混着练拿到更好的综合能力、更少的遗忘吗？",
    hook:
      "把几个 JSONL 文件直接倒在一起训练，各模态练多少就没人管了：长音频样本贡献的 token 和梯度多得多，实际比例早就跑偏了配置。",
    outcomes: [
      "把文本、图像、音频理解和语音输出的样本字段、loss 算法统一成一套约定。",
      "分清三种配比方式：sample-balanced（按样本数配平）、token-balanced（按 token 数配平）和 temperature sampling（温度采样调冷热门比例）。",
      "测出模态之间的梯度冲突、实际 token 占比，以及 capability replay（旧任务回放补课）修复了多少。",
      "产出能接着做偏好优化或 RL 的 joint-sft-v1，附一张分模态能力对照表。",
    ],
    misconception: {
      myth: "每种模态抽一样多的样本，就是公平的 joint SFT。",
      truth:
        "不同样本产出的条件 token、目标 token 和 loss 密度差别很大；token 量、loss 权重、更新频率三样得一起控住。",
    },
    prerequisites: [
      "理解 instruction tuning、assistant-only mask 和多目标 loss。",
      "起点模型至少得会文本、单图、短音频推理。",
      "可以从官方 MiniMind-O 直接开始；视频要等视觉前端固定后才能作为扩展加入。",
    ],
    labId: "lab-15-capability-ledger",
    readingTime: 75,
    difficulty: "高级",
    hardware: {
      minimum: "1–2 卡检查数据字段并完成 1k 步预实验",
      recommended: "8 卡完成三臂、梯度分析与多 seed",
      notes: "默认主矩阵只含 text/image/audio；视频前端还没固定时，video 权重必须写 0。",
    },
    learningMode: ["数据约定", "采样模拟器", "梯度诊断", "能力回放"],
  },
  {
    id: "16",
    slug: "multimodal-preference-optimization",
    shortTitle: "多模态 DPO / mDPO",
    unit: unitById.alignment,
    essentialQuestion:
      "普通 DPO 是不是只学了个说话腔调？加上媒体条件偏好和 reward anchor 之后，模型挑答案时会更认真去看真实的图像、音频或视频证据吗？",
    hook:
      "chosen 答案更长、更客气的话，把图片抽掉模型照样猜得对偏好——win rate 涨了，可能跟“有没有看图”半点关系没有。",
    outcomes: [
      "从 Bradley–Terry 模型一步步推出 DPO，讲清 reference、β 和藏在里面的 KL 约束各干什么。",
      "构造换错图、抽掉图、打乱图这三种媒体反事实，加上“说得流利但事实错”的 hard negative。",
      "实现 CoPO 和 AncPO，跟普通 DPO 做等 pair 数、等预算的对照。",
      "审计三件事：模型是否真依赖媒体条件、chosen 概率有没有掉、基础能力忘了多少。",
    ],
    misconception: {
      myth: "偏好准确率或 judge win rate 涨了，多模态对齐就变好了。",
      truth:
        "模型可能只是学会了蹭长度、蹭格式、模仿 teacher 的腔调；只有“配对的媒体”和“错误/抽掉的媒体”之间 margin 拉开了，才说明它真在看证据。",
    },
    prerequisites: [
      "理解 sequence log-prob、二选一偏好数据和冻结的 reference policy。",
      "手上有个能算 chosen/rejected 条件概率的 SFT checkpoint。",
      "不要求先做实验 15；但三臂必须从完全相同的起点复制出来。",
    ],
    labId: "lab-16-grounded-preference",
    readingTime: 70,
    difficulty: "高级",
    hardware: {
      minimum: "1×24GB 可做小规模 DPO/LoRA 预实验",
      recommended: "4–8 卡完成三臂与生成式 blind evaluation",
      notes: "这课只做离线的 preference optimization，别把在线的 GRPO 掺进来。",
    },
    learningMode: ["目标推导", "pair 审计", "媒体反事实", "blind preference evaluation"],
  },
  {
    id: "17",
    slug: "grpo-rlvr",
    shortTitle: "多模态 GRPO / RLVR",
    unit: unitById.alignment,
    essentialQuestion:
      "让程序当判分员来训练 GRPO，模型在没练过的任务上正确率真会涨吗？你怎么证明 reward 上涨不是靠凑格式、背模板或者钻验证器的空子？",
    hook:
      "reward 涨了不等于本事涨了。模型可能只是学会了讨好解析格式、背下答题模板，或者摸到了 verifier（判分程序）的漏洞。",
    outcomes: [
      "亲手推导 group-relative advantage（组内相对打分）、clipped objective 和 KL 约束。",
      "给 OCR、表格、grounding、ASR 和时序任务各写一个带版本号的 verifier。",
      "盯住 zero-variance group、pass 分布、entropy、答案长度漂移和 rollout 成本。",
      "用 public/hidden 双套 verifier、没见过的题目模板和人工逐 case 审计，揪出 reward hacking。",
    ],
    misconception: {
      myth: "RLVR 的 reward 是程序算出来的，所以天然客观、攻不破。",
      truth:
        "解析规则、数值容差、数据泄漏、环境状态，模型全都可能钻空子；验证器得写清输入输出约定，还要过单元测试、隐藏测试集和攻击测试三道关。",
    },
    prerequisites: [
      "理解 policy gradient、importance ratio 和 on-policy sampling（用当前模型自己生成的答案来训练）。",
      "起始模型在预实验上的 pass@k 不能全是 0 或全是 1，否则没梯度可学。",
      "先在 text/math 上把生成、验证、更新整条流程跑通，再搬到多模态环境。",
    ],
    labId: "lab-17-break-the-verifier",
    readingTime: 75,
    difficulty: "研究级",
    hardware: {
      minimum: "1–2 卡运行小模型、单环境教学实验",
      recommended: "8 卡分离 rollout、verifier 与更新 worker",
      notes: "算力大头花在在线生成上；有效 policy token 和 wall-clock 都得如实报出来。",
    },
    learningMode: ["目标推导", "verifier 红队", "在线 rollout", "reward hacking 审计"],
  },
  {
    id: "18",
    slug: "nemotron-finetuning",
    shortTitle: "Nemotron 官方微调",
    unit: unitById.alignment,
    essentialQuestion:
      "把权重、AutoModel、CORD-v2 数据和硬件版本全部锁死之后，官方的 LoRA recipe 能一步不差地复现出来吗？之后只换 dataset/collator，能把它迁到你自己的任务上吗？",
    hook:
      "Nemotron 放出了权重和部分训练示例，但完整的预训练和后训练流程没公开。这课只复现公开的那部分，并把缺了什么讲明白。",
    outcomes: [
      "讲明白 30B total / 约 3B active 是怎么回事，还有 hybrid backbone 和 EP=8。",
      "把 base→LoRA→save→新进程 load→resume→inference 这条链完整走一遍。",
      "读懂 processor、collator、image flags 和 label mask，然后迁移到自定义监督任务。",
      "分清权重、代码、数据、recipe、license 各开放到哪一步，别混为一谈。",
    ],
    misconception: {
      myth: "Nemotron 有开放权重和官方教程，所以它的 Omni 训练可以完整复现。",
      truth:
        "公开的 recipe 只盖住了特定的 CORD-v2 LoRA/full-SFT 示例；完整预训练数据、全部 adapter 训练和 SFT→MPO→GRPO 那条链都不在里面。",
    },
    prerequisites: [
      "理解 LoRA、Sparse MoE、Mamba/Attention hybrid 和数据 collator。",
      "会核验模型 revision、逐文件 hash 和许可证。",
      "本课可独立执行；但只有 8×H100 80GB 才够得上官方的完整训练配置。",
    ],
    labId: "lab-18-official-recipe-audit",
    readingTime: 90,
    difficulty: "研究级",
    hardware: {
      minimum: "8×H100 80GB（复现官方 LoRA 训练配置）",
      recommended: "8×H100 80GB、NVLink/NVSwitch、充足本地高速存储",
      notes: "官方实测约 LoRA 30GiB/GPU、full SFT 49GiB/GPU；别的八卡组合不能按“总显存一样”来等价替换。",
    },
    learningMode: ["开放边界审计", "环境冻结", "官方 recipe 复现", "collator 迁移"],
  },
  {
    id: "19",
    slug: "capstone-thinker-talker",
    shortTitle: "Thinker × Talker 毕业系统",
    unit: unitById.frontier,
    essentialQuestion:
      "让现代感知 Thinker 用 hidden-state bridge（跳过文字、直接递内部状态的桥）接上 Talker，能比文字桥留住更多跨模态和语气信息吗？这套桥还进得了能暂停、重规划、恢复的双工系统吗？",
    hook:
      "MiniMind 的 Talker 不是随便接段文字就能念的 TTS：它同时吃 MiniMind Thinker 的逐位置 bridge state、历史 8 路 codec 编号和说话人条件，缺一样都不行。",
    outcomes: [
      "保留三个逐级可诊断的版本：文字桥、hidden bridge、双工会话层，出问题知道该查哪层。",
      "抓取并比较 Nemotron 不同层的 hidden state，只训一个小 adapter，30B 的 Thinker 一个参数不动。",
      "把故障拆到五个环节定位：perception、reasoning、bridge、speech rendering、turn policy。",
      "交出带完整时间戳、逐 case replay、TTFA/RTF/stop/replan 指标的毕业原型。",
    ],
    misconception: {
      myth: "把 Nemotron 输出的文字交给 MiniMind Talker，就是一个现代开源 GPT-4o 复刻。",
      truth:
        "这是个有价值但边界清楚的模块化原型；文字桥会丢掉连续模态信息，公开数据、模型规模、RL 环境和端到端训练也都跟闭源系统不是一回事。",
    },
    prerequisites: [
      "能跑完整的 MiniMind-O checkpoint 和 Nemotron Omni 推理。",
      "理解 teacher forcing、hidden-state capture、音频 codec 和异步 session。",
      "用指定的 streaming ASR 加规则策略就能独立开始，不强制做完实验 1–18。",
    ],
    labId: "lab-19-open-omni-capstone",
    readingTime: 120,
    difficulty: "研究级",
    hardware: {
      minimum: "8×48GB 仅做已缓存 hidden-state bridge 训练",
      recommended: "8×80GB 运行默认 BF16 缓存方案和端到端实验",
      notes: "先把 Thinker 的 hidden states 离线缓存好，免得每训一步 adapter 都要重跑一遍 30B 模型。",
    },
    learningMode: ["接口设计", "表征层 sweep", "异步系统集成", "端到端验收"],
  },
  {
    id: "20",
    slug: "unified-understanding-generation",
    shortTitle: "统一理解与图像生成",
    unit: unitById.frontier,
    essentialQuestion:
      "骨干、数据、更新预算全都固定后，“语义视觉路径＋低层 VAE 路径”双管齐下，真能比单独一条路更好地兼顾看懂图和画出图这两件事吗？",
    hook:
      "看懂图靠的是语义；画出图还得管颜色、纹理这些低层细节。两类目标共用一套参数，梯度照样可能互相拆台。",
    outcomes: [
      "分清三种视觉表示装了多少信息：VQ token、VAE latent、语义视觉 feature。",
      "亲手实现一个最小的 flow matching（学“噪声到图像”的变形路径）、采样和 classifier-free guidance。",
      "公平比较三臂：只用 VAE、双路径融合、Janus 式理解生成分家。",
      "用 specialist gap 和 interference delta 两个量，测共享训练到底是两头都帮，还是互相拖后腿。",
    ],
    misconception: {
      myth: "给 VLM 接个 Stable Diffusion API，或者同时挂俩 head，就算统一理解与生成了。",
      truth:
        "共享的核心必须真参与两类训练目标，再用参数量、计算量、消融和 specialist 对照把共享训练的效果量出来。",
    },
    prerequisites: [
      "理解 VQ/VAE、diffusion 或 flow matching、DiT/adaLN。",
      "能跑起冻结的 MiniMind-O、SigLIP2 和 SD VAE。",
      "这是高阶选修，不依赖语音双工主线；建议先修过至少一门讲结构公平对照的课。",
    ],
    labId: "lab-20-unification-stress-test",
    readingTime: 100,
    difficulty: "研究级",
    hardware: {
      minimum: "8×80GB 做缩小版机制实验",
      recommended: "8×80GB；论文规模通常需要更多资源",
      notes: "最低交付是图像理解＋图像生成两样都行；视频生成只算加分扩展。",
    },
    learningMode: ["生成机制推导", "toy flow 实作", "三臂联合训练", "干扰审计"],
  },
  {
    id: "21",
    slug: "clip-siglip-contrastive",
    shortTitle: "图文对比空间",
    unit: unitById["vl-foundation"],
    essentialQuestion:
      "冻结的视觉编码器为什么能给语言模型当眼睛？没有分类标签时，对比损失具体拉近什么、推开什么？",
    hook:
      "MiniMind-O 里的 SigLIP2 是冻住的。冻之前，它把图和字推进同一张余弦表。本课把那张表拆开手算。",
    outcomes: [
      "写出对称 InfoNCE，并说明温度过小会把 softmax 逼近 one-hot、过大则分不开正负对",
      "写出 SigLIP 的 pairwise sigmoid 损失，说明它不需要对 batch 做 softmax",
      "解释同 batch 负样本、双塔编码，以及 LiT 锁图像塔、训文字塔的做法",
      "在 4×4 矩阵上验证：打乱配对后损失不下降，温度从 0.01 调到 0.2 时正对概率峰值下降",
    ],
    misconception: {
      myth: "CLIP 分数高就等于下游看图问答强；batch 小一点只是训得慢。",
      truth:
        "对比分数测的是共享空间里的检索，不是 VQA。InfoNCE 的负样本来自同 batch，batch 太小会改变任务本身。",
    },
    prerequisites: [
      "会写 Python，知道 softmax 和交叉熵",
      "不必先学 MiniMind-O 第 01–20 课",
    ],
    labId: "lab-21-contrastive",
    readingTime: 55,
    difficulty: "入门",
    hardware: {
      minimum: "CPU，完成本课手算实验与浏览器互动实验",
      recommended: "可选 1 张消费级 GPU，加载公开 CLIP 小模型做检索演示，不训练",
      notes: "复现 CLIP 论文规模需要约 256 块 TPUv3 训练约 10 天；ALIGN 用 1024 块 TPUv3。本课主路径不训练大模型。",
    },
    learningMode: [
      "阅读",
      "交互实验",
      "CPU 机制实验",
    ],
  },
  {
    id: "22",
    slug: "standard-vlm-recipe",
    shortTitle: "标准 VLM 配方",
    unit: unitById["vl-foundation"],
    essentialQuestion:
      "冻视觉、只训投影、再解冻 LLM，这个顺序少一步会怎样？",
    hook:
      "一套开关决定模型是学会看图，还是把已经会的文字忘掉。",
    outcomes: [
      "能按模块写出只训投影、投影加 LoRA、再解冻 ViT 的可训练参数量",
      "能说明 LLaVA 两阶段、BLIP-2 双冻塔和 Flamingo 门控插入各自冻什么",
      "能用图文对齐、指令跟随和旧文本探针判断解冻顺序是否合格",
      "能把第 02 课的 connector 消融和本课的业界配方分开记账",
    ],
    misconception: {
      myth: "视觉编码器越早一起训，图文对齐越好，也不会伤到原来的语言模型。",
      truth:
        "从已经对比对齐过的 ViT 出发时，过早解冻视觉塔会改写文本方向，旧文本能力先掉。",
    },
    prerequisites: [
      "会写 PyTorch 的 requires_grad 与参数组",
      "知道因果语言模型的 next-token 损失",
      "建议读过第 21 课的对比学习共享空间；可独立读",
    ],
    labId: "lab-22-unfreeze-schedule",
    readingTime: 65,
    difficulty: "进阶",
    hardware: {
      minimum: "无 GPU：读完全文，跑 CPU 机制实验和浏览器开关实验",
      recommended: "复现 LLaVA-1.5 公开配方需要约 8×A100 一天；教学缩小版 1×24GB 可做冻结消融",
      notes: "CPU 实验证明梯度 mask 和参数计数，不报告真实 VQA 成功率",
    },
    learningMode: [
      "阅读",
      "浏览器实验",
      "CPU 机制实验",
    ],
  },
  {
    id: "23",
    slug: "grounding-ocr-spatial",
    shortTitle: "指代 OCR 空间探针",
    unit: unitById["vl-foundation"],
    essentialQuestion:
      "选择题答对，是否等于模型用对了像素？幻觉、错数、读错字、左右颠倒分别该用什么探针。",
    hook:
      "同色两只杯子，模型说「红色」完全正确，注意力却落在另一只杯子上。答对了不等于看见了位置。",
    outcomes: [
      "用 IoU 和注意力质量分数定义 grounding 命中，并证明它可以和 VQA 准确率脱钩",
      "按随机、热门、对抗三种负采样搭 POPE 式是否存在探针，只在负例上算幻觉率",
      "把 OCR 当成字形区域上的定位，而不是生成一段碰巧含字的配文",
      "用左右、计数和 Set-of-Mark 编号把空间关系从语言先验里拆出来",
    ],
    misconception: {
      myth: "VQA 或多项选择准确率高，说明模型已经看对了物体位置。",
      truth:
        "文字答案可以来自语言共现或同色干扰物。没有框、点或注意力命中，不能声称模型看见了位置。",
    },
    prerequisites: [
      "第 21 课图文对比学习的共享空间",
      "第 22 课视觉语言模型的标准解冻顺序",
      "第 08 课动态分辨率与二维位置编码的评测口径",
    ],
    labId: "lab-23-grounding-ocr-spatial",
    readingTime: 60,
    difficulty: "进阶",
    hardware: {
      minimum: "无 GPU：读完全文并跑 CPU 机制实验",
      recommended: "1×24GB 可对开源 2B 级 VLM 做指代与 POPE 探针；多卡仅在复现 Kosmos-2 / Grounding DINO 原训练时需要",
      notes: "CPU 实验证明 IoU、注意力命中和幻觉率可手算复核，不代表真实模型成功率。Kosmos-2 原文训练使用 256 张 V100、约 60k step。",
    },
    learningMode: [
      "机制拆解",
      "探针设计",
      "CPU 复核",
    ],
  },
  {
    id: "24",
    slug: "vlm-to-vla",
    shortTitle: "动作接入 VLM",
    unit: unitById["vla"],
    essentialQuestion:
      "VLM 写出“抓住杯子”时机械臂为何不动，动作怎样进入同一套 next-token？",
    hook:
      "同一张桌面图可以只生成文字、吐出离散动作 token，或把文字映射成技能。切断视觉或改错指令，三条通路的动作会不会一起变，能把“会说”和“会控臂”拆开。",
    outcomes: [
      "写出 7 维末端动作的均匀分箱、词表偏移和与文本共享的 next-token 交叉熵",
      "分清只出文字、离散动作 token、文字映射技能三条通路，并能指出各自失败时手臂停在哪一层",
      "用视觉捷径探针解释：场景唯一决定任务时错指令可以不改动作；多目标同场景时必须改",
      "证明只训练独立语言头时动作 token 行梯度为 0，而联合 softmax 会经配分函数漏梯度",
    ],
    misconception: {
      myth: "只要视觉语言模型会写 pick up the cup，再在外面挂一个强化学习控制器，就算做成了 VLA。",
      truth:
        "会写那句话只证明文字通路通了。VLA 要把动作放进同一套条件生成；外挂控制器既不共享词表，也不接受与文本相同的 next-token 损失。",
    },
    prerequisites: [
      "建议先修第 22 课。",
      "建议先修第 23 课。",
    ],
    labId: "lab-24-action-token",
    readingTime: 70,
    difficulty: "进阶",
    hardware: {
      minimum: "无 GPU：读完全文并跑 CPU 分箱与梯度实验",
      recommended: "1×24GB 仅用于打开公开 VLM 权重做定性对照，不是本课验收条件",
      notes: "复现 RT-2-PaLI-X-55B 需要论文中的多 TPU 云端服务；原文写 55B 控制频率为 1–3 Hz。CPU 实验不证明真机成功率。",
    },
    learningMode: [
      "机制推导",
      "浏览器教具",
      "CPU 数值核对",
    ],
  },
  {
    id: "25",
    slug: "action-tokenization",
    shortTitle: "动作表示对照",
    unit: unitById["vla"],
    essentialQuestion:
      "均匀分箱、连续回归、动作分块和 DCT 压缩各自保住什么、丢掉什么？",
    hook:
      "机械臂的 7 维连续动作要进语言模型，先得变成能逐步预测的编号，或者改成一次吐出的连续块。选错表示，高频接触会消失，开环窗口会拖过环境变化。",
    outcomes: [
      "写出均匀分箱的量化误差上界，并说明 B=2 时高频来回为何会塌进同一箱",
      "用 H/f 计算动作分块的开环时长，并对照自回归 7 步与并行 1 步的串行深度",
      "解释 FAST 的 DCT 加量化如何抑制高频抖动，且只引用论文 Table I 的压缩比",
      "对照 OpenVLA 的 256-bin 交叉熵与 OpenVLA-OFT 的并行 L1，列出重建、词表和吞吐的取舍",
    ],
    misconception: {
      myth: "动作 token 分得越细、词表越大，控臂一定越准，也一定越适合接进 VLM。",
      truth:
        "箱数只限制量化误差上界；高频来回在粗箱里会消失，而自回归逐步吐编号会把控制频率卡在数 Hz。连续 L1 和 DCT 压缩改的是可预测性与串行深度，不是把箱子加厚。",
    },
    prerequisites: [
      "建议先修第 22 课。",
      "建议先修第 24 课。",
    ],
    labId: "lab-25-action-repr",
    readingTime: 70,
    difficulty: "高级",
    hardware: {
      minimum: "无 GPU：读完全文，跑浏览器互动实验和 CPU 机制实验",
      recommended: "1×24GB 用于阅读公开权重的推理形状；不要求复现预训练",
      notes: "OpenVLA 预训练约为 64 张 A100、14 天、21500 A100-hours。本课主路径不训练 7B VLA，CPU 数字不能写成真机成功率。",
    },
    learningMode: [
      "阅读",
      "互动实验",
      "CPU 机制实验",
    ],
  },
  {
    id: "26",
    slug: "robot-data-mixture",
    shortTitle: "异构数据混合",
    unit: unitById["vla"],
    essentialQuestion:
      "几十种机体、相机和动作空间拼在一起时，怎样混合才不会让大域淹没小域、让模型靠机体 ID 走捷径？",
    hook:
      "按原始条数采样，厨房里那台数据最多的手臂会占满 batch。把语言遮掉，准确率若几乎不动，模型多半在认桌布颜色和机体外形，而不是听指令。",
    outcomes: [
      "写出域采样 p_d ∝ n_d^α，并计算 α=1、α=0 和每域上限下的 batch 组成与有效域数",
      "说明末端位姿与关节、绝对/相对/速度、相机外参为何不能直接叠进同一张量",
      "用打乱指令和打乱机体标签两套负对照，区分真语言条件与机体 ID 泄漏",
      "对照 OXE / Octo / OpenVLA 公开的数据规模、混合物和过滤规则，列出可执行的缩小版配比实验",
    ],
    misconception: {
      myth: "把 Open X-Embodiment 的 tfrecord 拼起来随机抽 batch，模型就会自动变成跨机体通才。",
      truth:
        "按条数采样时最大域会占满梯度。动作坐标系、夹爪开合约定和相机外参仍是各写各的；机体 ID 或桌布颜色一旦能预测任务，语言监督就变成摆设。",
    },
    prerequisites: [
      "建议先修第 24 课。",
      "建议先修第 25 课。",
    ],
    labId: "lab-26-mixture",
    readingTime: 65,
    difficulty: "高级",
    hardware: {
      minimum: "无 GPU：读完全文，跑浏览器互动实验和 CPU 机制实验",
      recommended: "1×24GB 仅用于打开公开数据加载器看字段，不要求复现预训练",
      notes: "OpenVLA 预训练为 64 张 A100、14 天、21500 A100-hours。Octo-Base 为 TPU v4-128、batch 2048、30 万步、约 14 小时。本课主路径不训练这些模型，CPU 数字不能写成真机成功率。",
    },
    learningMode: [
      "阅读",
      "互动实验",
      "CPU 机制实验",
    ],
  },
  {
    id: "27",
    slug: "autoregressive-vla",
    shortTitle: "诊断自回归 VLA",
    unit: unitById["vla"],
    essentialQuestion:
      "把 7 维动作塞进 Llama 词表尾部逐步吐出之后，延迟从哪来，OFT 的并行 L1 和 chunk 各改了吞吐和哪一套 LIBERO 成功率？",
    hook:
      "LIBERO 平均成功率看起来很高，控制频率却只有数 Hz。先把串行 7 步和并行 1 步的延迟条拉开，再拆开四套件，不要只报一个平均数。",
    outcomes: [
      "写出动作 token 如何覆盖 Llama 词表尾部，以及 teacher-forced CE 的 loss mask 盖在哪些位置。",
      "手算串行步数 7 或 7H，并对照并行一次吐出 H×7 时延迟和吞吐怎么变。",
      "解释 CE 在 bin 边界上的类别跳变，以及连续 L1 为什么对同一误差更平滑。",
      "读 LIBERO 数字时强制拆 Spatial / Object / Goal / Long，拒绝把 97% 平均成功率写成通用操作已解决。",
    ],
    misconception: {
      myth: "OpenVLA-OFT 在 LIBERO 上平均 97%，通用操作已经被解决了。",
      truth:
        "97.1% 来自论文表 I：四个套件各自微调、过滤失败示教、仿真 Franka、每套件 500 次试验。LIBERO-Long 仍低于另外三套，真机和其他机体不在这个平均数里。",
    },
    prerequisites: [
      "第 25 课：离散 bin、连续回归和动作 chunk 各自丢掉什么。",
      "第 26 课：异构机器人数据不能按原始条数直接混。",
      "会写因果语言模型的 next-token loss 和 assistant-only mask（第 01 课）。",
    ],
    labId: "lab-27-openvla-serial-parallel",
    readingTime: 75,
    difficulty: "高级",
    hardware: {
      minimum: "无 GPU：读完全文并跑 CPU 机制实验",
      recommended: "复现论文规模：OpenVLA 预训练约 64×A100、14 天；LoRA 微调约 1×A100、10–15 小时；OFT 微调约 8×A100/H100",
      notes: "CPU 实验和浏览器 Lab 都是教学夹具，不得写成 LIBERO 或真机成功率。",
    },
    learningMode: [
      "词表与延迟对照",
      "CE / L1 mask 单测",
      "套件拆桶",
      "遮挡负对照",
    ],
  },
  {
    id: "28",
    slug: "flow-matching-vla",
    shortTitle: "流匹配生成动作块",
    unit: unitById["vla"],
    essentialQuestion:
      "同一套 flow matching 速度场，从画图像 latent 改成吐一段连续动作块，训练目标、积分方向和开环窗口分别改了什么？",
    hook:
      "自回归 VLA 一步吐一个动作编号，50 Hz 的账单对不上。π0 让动作专家从噪声积出整段连续轨迹，再只执行前几步就重规划。",
    outcomes: [
      "能把第 20 课的直线路径速度场改写到动作块 A ∈ R^{H×d}，并写出 v_θ(a_t, t, x_vlm)。",
      "能对照 π0 与第 20 课的两种时间箭头，手算一个时间步的目标速度并核对 Euler 积分方向。",
      "能说明动作专家、动作分块和执行前 k 步后重规划各自挡住哪类失败。",
      "能按原文条件和数据转述 π0.5 的开放世界声明，不把它写成任意房间任意任务都成功。",
    ],
    misconception: {
      myth: "flow matching 一次生成整段动作，就等于已经在看最新画面做闭合控制。",
      truth:
        "生成的是开环动作块。块还没执行完时环境可以变；必须执行前 k 步再规划，否则目标被挪走仍沿旧轨迹。",
    },
    prerequisites: [
      "第 25 课：知道离散 bin、连续回归和动作分块各自丢掉什么。",
      "建议第 20 课：读过图像版 flow matching 的直线路径和 Euler 采样。",
      "不要求 GPU；CPU 实验只核对动作向量上的公式。",
    ],
    labId: "lab-28-flow-action",
    readingTime: 80,
    difficulty: "研究级",
    hardware: {
      minimum: "无 GPU 可读完全文并跑 CPU 实验",
      recommended: "复现 π0 论文规模需要多机多卡与真机，本课不提供",
      notes: "交互实验是二维抓取的教学模拟；CPU 实验证明速度场和积分方向可复核，不证明真机成功率。",
    },
    learningMode: [
      "速度场推导",
      "动作块积分",
      "分块再规划",
      "文献对照",
    ],
  },
  {
    id: "29",
    slug: "dual-system-vla",
    shortTitle: "快慢双系统 VLA",
    unit: unitById["vla"],
    essentialQuestion:
      "慢速 VLM 规划与高频动作专家之间到底传什么，两个时钟怎样同时跑？",
    hook:
      "7B 自回归模型既要读懂“把杯子放到架子第二层”，又要 100 Hz 出关节，算力账单对不上。拆开 System 2 / System 1 之后，暂停规划看手臂会不会还在执行最后一条子目标。",
    outcomes: [
      "写出规划周期与控制周期的关系 ΔT2 = k ΔT1，并定义子目标年龄与过期。",
      "对照 GR00T N1 的 Eagle VLM + DiT 与 π0.5 的先子任务后动作块，说明中间条件分别是 token 还是语义文本。",
      "根据机理研究实际测过的六套模型，区分专家通路的动作程序与 VLM 通路的目标语义。",
      "用双时钟实验证明：规划暂停过久任务失败，错误子目标改变末端轨迹。",
    ],
    misconception: {
      myth: "快慢双系统等于两个独立模型，中间只传一句自然语言，System 1 暂停规划后会自己发明下一阶段。",
      truth:
        "GR00T N1 把 VLM token 交叉注意进 DiT 并联合训练；π0.5 用同一套权重先出子任务再出动作。System 1 每步只消费当前子目标，不会在 System 2 暂停时改写任务阶段。",
    },
    prerequisites: [
      "建议先修第 27 课。",
      "建议先修第 28 课。",
    ],
    labId: "lab-29-dual-clock",
    readingTime: 75,
    difficulty: "研究级",
    hardware: {
      minimum: "无 GPU：阅读全文并运行 CPU 双时钟实验",
      recommended: "论文规模后训练：NVIDIA 博文给出的 1×RTX A6000 或 1×GeForce RTX 4090；N1 预训练约 50,000 H100 GPU hours；N1.5 预训练 1,000×H100、250k steps、global batch 16384",
      notes: "CPU 实验与浏览器 Lab 是教学模拟，不能写成真机或 LIBERO 成功率。",
    },
    learningMode: [
      "阅读机制",
      "浏览器双时钟实验",
      "CPU 离散时间循环",
    ],
  },
  {
    id: "30",
    slug: "closed-loop-control",
    shortTitle: "频率分块与延迟",
    unit: unitById["vla"],
    essentialQuestion:
      "一次吐出 H 步动作、环境已经变了时，怎样用控制频率、执行前缀和推理延迟决定哪些步还能做、哪些必须丢掉？",
    hook:
      "开环放出几十步动作，传送带上的杯子已经被挪走。分块能减少推理次数，却加长了必须盲走的窗口；提高频率又要求模型更快。被打断时，没执行完的 chunk 不能当没发生。",
    outcomes: [
      "写出开环窗口 H/f 与提交窗口 k/f，并能用 d 与这两个量判断 chunk 是否过期。",
      "把第 07 课的 CONTINUE / PAUSE / REPLAN 接到控制时钟上：剩余动作步对应未播放 PCM，REPLAN 后不得执行旧剩余步。",
      "对照 ACT 的动作分块、Diffusion Policy 的后退视野、OpenVLA-OFT 的并行块、π0 的 H=50 / 50 Hz，说明各自把延迟藏进了哪一段窗口。",
      "在传送带教具和 CPU 事件回放里找到能抓住的一组参数，以及延迟大于开环窗口而抓空的一组参数。",
    ],
    misconception: {
      myth: "动作块越长越稳，提高控制频率总能补上推理延迟。",
      truth:
        "H 变长会线性加长开环窗口；f 变高若 H 不变，窗口反而缩短，更容易被延迟打穿。过期 chunk 必须丢弃，不能靠更高频率把过期计划执行得更细。",
    },
    prerequisites: [
      "读过动作分块与 H/f 开环时长的定义，或能接受本课重写这两条公式。",
      "建议先看第 07 课的 CONTINUE / PAUSE / REPLAN 与 available_at，本课只借用状态机，不实现音频路由。",
      "会写 Python，能读离散事件时间线；不要求有机械臂或 GPU。",
    ],
    labId: "lab-30-horizon",
    readingTime: 65,
    difficulty: "高级",
    hardware: {
      minimum: "无 GPU：读完全文、跑浏览器教具与 CPU 事件回放。",
      recommended: "复现 OpenVLA-OFT / π0 规模推理需要 NVIDIA A100 或同级；ACT 原文在单张 11GB RTX 2080 Ti 上约 0.01 s 推理。",
      notes: "教具和 CPU 实验证明调度协议可复核，不证明真机抓取成功率。",
    },
    learningMode: [
      "控制时钟调度",
      "过期 chunk 回放",
      "传送带教具",
      "文献对照",
    ],
  },
  {
    id: "31",
    slug: "vla-evaluation",
    shortTitle: "可拆层 VLA 评测",
    unit: unitById["vla"],
    essentialQuestion:
      "LIBERO 四套件平均、SIMPLER 视觉 gap、CALVIN 长程、真机 N=25，这四类数字为什么不能横着比？",
    hook:
      "同一政策在固定初始态上很高，换随机摆放和 distractor 就掉下去。掉的是捷径，不是“模型突然变笨”。N=25、成功率 0.8 的区间有三十个百分点宽。",
    outcomes: [
      "把 LIBERO 四套件、SIMPLER 视觉域、CALVIN 长程链和真机小样本拆成不可横比的四类数字，并写明是否 fine-tune。",
      "为一次评测写出协议卡：套件桶、初始态种子、相机/纹理、指令改写、成功谓词。",
      "亲手计算二项成功率的 Wilson 与正态近似区间；N=25、成功率 0.8 时 Wilson 约为 [0.609, 0.911]。",
      "在三种协议上先预测排序再揭晓：固定初始态最高、加 distractor 最低，并拒绝把单一数字标成 SOTA。",
    ],
    misconception: {
      myth: "仿真平均成功率高，就等于真机能力强；一个百分数可以当 SOTA。",
      truth:
        "成功率绑定协议。套件、初始态、视觉域、语言、成功定义和样本量任一改变，数字就不是同一个量。SIMPLER 连相对排序相关都要单独测，更不能把 LIBERO 平均写成真机能力。",
    },
    prerequisites: [
      "建议先修第 27 课。",
      "建议先修第 28 课。",
    ],
    labId: "lab-31-eval-protocol",
    readingTime: 70,
    difficulty: "高级",
    hardware: {
      minimum: "无 GPU：阅读全文、跑浏览器三种协议实验和 CPU 区间实验",
      recommended: "若要复现论文规模仿真评测：单卡即可跑 LIBERO / SIMPLER 推理；真机评测需要对应机械臂与固定相机外参",
      notes: "CPU 实验与浏览器 Lab 是教学模拟，不能写成 LIBERO、SIMPLER 或真机成功率。",
    },
    learningMode: [
      "协议拆桶",
      "置信区间手算",
      "三种协议教学模拟",
      "sim-to-real 声明",
    ],
  },
  {
    id: "32",
    slug: "gui-agent-grounding",
    shortTitle: "统一GUI接地",
    unit: unitById["embodied-agent"],
    essentialQuestion:
      "点按钮和抓杯子是否都是在视觉观察里接地一个空间动作，Set-of-Mark 与连续坐标能否共用一套训练信号？",
    hook:
      "屏幕上的点击和桌面上的末端位移，差在动作空间，不差在“先看见位置再出手”这一步。",
    outcomes: [
      "把 GUI 点击和机械臂末端 xy 写成同一套 [0,1]^2 归一化坐标，并核对分箱与编号范围。",
      "比较 SoM 分类损失和连续坐标回归损失对视觉分辨率的敏感度。",
      "说清 click / type / scroll 与 6D 末端加夹爪在何处可以共享、何处必须分叉。",
      "按 Magma 原文区分 SoM（动作接地）和 ToM（轨迹规划），并对照 CogAgent、SeeClick、OS-Atlas、UI-TARS。",
    ],
    misconception: {
      myth: "会点屏幕就等于会控臂，或者给 VLM 外挂一个鼠标 API 就算统一了身体。",
      truth:
        "统一的是视觉观察上的空间接地；动作语义、接触力和不可逆后果仍要单独建模，低分辨率下连续坐标比编号分类更脆。",
    },
    prerequisites: [
      "第 23 课：能区分答对和看对位置。",
      "第 24 课：知道动作如何进入 token 或连续头。",
    ],
    labId: "lab-32-som-grounding",
    readingTime: 75,
    difficulty: "研究级",
    hardware: {
      minimum: "无 GPU 可读完全文并跑 CPU 机制实验",
      recommended: "1×24GB 试小规模接地微调；复现论文规模需要多卡与大规模截图/机器人轨迹",
      notes: "浏览器实验是教学模拟，不是模型输出。CPU 实验只核对归一化、编号范围和 toy loss。",
    },
    learningMode: [
      "接地损失对照",
      "分辨率夹具",
      "跨身体共享头",
    ],
  },
  {
    id: "33",
    slug: "world-model-latent",
    shortTitle: "像素还是表征",
    unit: unitById["world-model"],
    essentialQuestion:
      "生成未来帧和在表征空间预测，训练信号差在哪？V-JEPA 2 用不到 62 小时机器人视频做动作条件后训练、零样本抓取，这个声明的条件和限制是什么？",
    hook:
      "像素 L2 把叶子和接触边算进同一笔损失。JEPA 把损失写在表征上，接触还可以被分开；把同一预测解码回像素，接触就会糊掉。",
    outcomes: [
      "能对照写出像素 L2 与表征 L1/L2 回归，并指出不可预测纹理只进入前者。",
      "能说明 JEPA 用 EMA + stop-gradient 防塌缩，VICReg 用方差铰链和协方差去相关，两条路不能抄成一句。",
      "能在遮挡未来 patch 后用接触/分离探针比较两条路，而不是只看重建好看不好看。",
      "能按 V-JEPA 2 原文条件转述“不到 62 小时 Droid 视频、零样本抓取”，并列出图像子目标、相机位、N=10 这些限制。",
    ],
    misconception: {
      myth: "世界模型就是把下一帧像素画出来；画得越清楚，就越会控臂。",
      truth:
        "像素重建把不可预测纹理算进损失，接触边会被平均掉。表征预测丢掉纹理、保留位置与重叠；V-JEPA 2-AC 的抓取声明还附带数据、目标和评测条件。",
    },
    prerequisites: [
      "建议第 09 课：视频有时间轴，不是一叠无序图片。",
      "建议第 28 课：连续动作可以在另一空间生成；本课的样本是视频表征，不是动作块。",
      "不要重做第 20 课的图像 flow matching。",
    ],
    labId: "lab-33-world-model-latent",
    readingTime: 75,
    difficulty: "研究级",
    hardware: {
      minimum: "无 GPU 可读完全文并跑 CPU 机制实验",
      recommended: "复现 I-JEPA / V-JEPA / V-JEPA 2 论文规模需要多卡与大规模视频，本课不提供",
      notes: "浏览器实验是教学模拟，不是模型输出。CPU 实验只核对 8 步序列上的像素 L2、表征回归和接触探针。",
    },
    learningMode: [
      "损失对照",
      "遮挡探针",
      "防塌缩夹具",
      "文献条件",
    ],
  },
  {
    id: "34",
    slug: "world-model-platform",
    shortTitle: "数据引擎或控制器",
    unit: unitById["world-model"],
    essentialQuestion:
      "Cosmos 生成物理视频给别人训，和 V-JEPA 2 或 Cosmos 3 自己拿预测去控，评价标准为什么不是同一张表？",
    hook:
      "落下的方块一加动作，生成路会丢物体 ID，控制路会把重力符号弄反。两件事都发生在看起来很像物理的视频里。",
    outcomes: [
      "把世界模型的两种用法拆开：数据引擎看保真和物理违规率，控制器看滚动执行误差。",
      "写出跨帧物体 ID 计数器和自由下落 Δv_y 符号探针，并能指出 Cosmos 原文承认的物体永久性与重力失败。",
      "对照 Cosmos 1 的 AR / diffusion、Cosmos 3 的 Mixture-of-Transformers，以及 Genie 从无标签视频学潜伏动作。",
      "在教学模拟里先预测再揭晓，触发物体消失或重力方向错误，并拒绝把该数字写成真机成功率。",
    ],
    misconception: {
      myth: "视频看起来像物理，就可以当仿真器去控臂，滚动成功率也能写成真机能力。",
      truth:
        "Cosmos 自己写明当前模型还不能当可靠物理仿真器。数据引擎和控制器要用不同门禁：丢掉物体 ID 或重力符号反了，生成路应拒收，控制路应报失败，两者都不能冒充真机成功率。",
    },
    prerequisites: [
      "第 33 课：能区分像素重建和表征预测。",
      "第 09 课：知道视频是有时间轴的观察，不是一叠独立图片。",
    ],
    labId: "lab-34-world-model-platform",
    readingTime: 75,
    difficulty: "研究级",
    hardware: {
      minimum: "无 GPU 可读完全文并跑 CPU 机制实验",
      recommended: "复现 Cosmos / Genie 论文规模需要多机多卡与大规模视频；本课不提供该算力",
      notes: "浏览器实验是教学模拟，不是 Cosmos 或 Genie 的前向输出。CPU 实验只核对 ID 计数器和重力符号，不证明真机能力。",
    },
    learningMode: [
      "数据引擎对照",
      "控制器探针",
      "物体 ID 计数器",
      "重力符号报警",
    ],
  },
  {
    id: "35",
    slug: "spatial-depth-vla",
    shortTitle: "深度接入抓取",
    unit: unitById["spatial-body"],
    essentialQuestion:
      "二维图像缺深度时，抓取会在哪一步失败？如何把 (u,v,z) 写成相机系点和可分箱的空间动作？",
    hook:
      "夹爪和杯子在图像上重合，只说明它们共线。缺深度时闭合发生在射线上错误的尺度，RGB 判定为成功，三维接触为假。",
    outcomes: [
      "写出针孔反投影，说明同一 (u,v) 对应整条射线，z 只决定尺度。",
      "构造无深度取均值时 RGB 命中为真、接触带判定为假的夹具，并与真实深度对照。",
      "对照 PerAct 体素、Act3D 特征场、SpatialVLA 的 Ego3D 与自适应动作网格、PointVLA 的点云旁路，说清深度从哪进来、改哪一段。",
      "把平移动作写成极坐标格子，并指出 3 个空间 token 与 7 维独立分箱的差别。",
    ],
    misconception: {
      myth: "图像上夹爪对准了杯子，就等于抓住了；VLA 看 RGB 就够。",
      truth:
        "对准只证明共线。接触发生在三维点上。无深度时用均值或训练桌面高度取 z，会在错误深度闭合。",
    },
    prerequisites: [
      "第 24 课：动作如何进入 token。",
      "第 25 课：分箱、连续表示和动作块。",
    ],
    labId: "lab-35-depth-grasp",
    readingTime: 70,
    difficulty: "高级",
    hardware: {
      minimum: "无 GPU 可读完全文并跑 CPU 反投影实验",
      recommended: "1×24GB 仅用于打开 SpatialVLA / PointVLA 公开权重做定性对照，不是本课验收条件",
      notes: "浏览器实验是教学模拟，不是模型输出。CPU 实验只核对 (u,v,z) 反投影和接触带，不证明真机抓取成功率。",
    },
    learningMode: [
      "反投影夹具",
      "接触带对照",
      "空间动作网格",
    ],
  },
  {
    id: "36",
    slug: "navigation-action-space",
    shortTitle: "导航动作词表",
    unit: unitById["spatial-body"],
    essentialQuestion:
      "「去厨房把杯子拿来」跨手臂 7 维和底盘速度或路点两套动作空间，词表偏移和失败模式如何分开？",
    hook:
      "同一句厨房指令里，走到台面和抓住杯子不是同一张词表。关掉地图后，路点政策吐出非法索引，速度政策仍输出合法 (v,ω) 却会撞墙或转圈。",
    outcomes: [
      "写出底盘 (v,ω) 与手臂 7 维各自的 bin 边界，并证明两套词表偏移不重叠。",
      "把厨房任务切成走到台面与抓杯子，指出 RT-1 三模式开关如何分时使用两套动作。",
      "对照路点索引与连续速度：丢掉地图后前者输出非法节点，后者仍合法但会撞墙或转圈。",
      "按原文区分 RT-1 导航子集、LM-Nav / Mobility VLA 的拓扑图、NaVid / NaVILA 的中层语言动作。",
    ],
    misconception: {
      myth: "底盘动作只是手臂动作多两维，共用同一套 256 bin 和同一张词表即可。",
      truth:
        "线速度、角速度和末端位移的物理区间不同，箱边界不能共用；路点索引还依赖地图节点数，丢图后会变成非法 id，这和速度撞墙不是同一种失败。",
    },
    prerequisites: [
      "建议先修第 24 课。",
      "建议先修第 35 课。",
    ],
    labId: "lab-36-nav-action",
    readingTime: 65,
    difficulty: "高级",
    hardware: {
      minimum: "无 GPU：读完全文，跑浏览器互动实验和 CPU 机制实验",
      recommended: "1×24GB 仅用于打开公开导航 VLA 权重做定性对照，不是本课验收条件",
      notes: "浏览器实验是教学模拟，不是模型输出。CPU 实验只核对词表偏移和非法索引，不证明真机导航成功率。",
    },
    learningMode: [
      "阅读",
      "互动实验",
      "CPU 机制实验",
    ],
  },
  {
    id: "37",
    slug: "dexterous-humanoid",
    shortTitle: "高维动作预算",
    unit: unitById["spatial-body"],
    essentialQuestion:
      "平行夹爪约 7 维、灵巧手 20+ 维、人形更高，同等 token 预算下丢掉的是开环窗口还是每维量化宽度？",
    hook:
      "维数加到 24 并不会白送精度。预算写死之后，不是分块变短，就是手指那一维的箱子变粗。",
    outcomes: [
      "写出固定预算 C=Hd 与 C_bit=H d b，说明 d 升高时 H 下降或每维箱宽变粗。",
      "在 d=7 与 d=24 上对照每维量化宽度，并核对半箱宽上界只对量化锚点成立。",
      "把高频手指和低频躯干分开记账，说明平均重建 L2 会掩盖手指失败。",
      "按原文引用 GR-3 的 19/22 自由度，按官方博文引用 Helix 的 35-DoF、200 Hz，不编 Helix / Gemini Robotics 未公开的架构。",
    ],
    misconception: {
      myth: "动作维数越高越灵巧，把 7 维配方直接扩到 24 维或全身即可。",
      truth:
        "预算 C=Hd 固定时，维数升高会缩短开环窗口或加粗每维量化；高频手指先坏，平均 L2 看不出来。",
    },
    prerequisites: [
      "建议先修第 25 课。",
      "建议先修第 29 课。",
    ],
    labId: "lab-37-action-dim",
    readingTime: 70,
    difficulty: "研究级",
    hardware: {
      minimum: "无 GPU 可读完全文并跑 CPU 机制实验",
      recommended: "1×24GB 用于阅读公开权重的动作头形状；不要求复现 GR-3 或人形真机",
      notes: "浏览器实验是教学模拟，不是模型输出。CPU 实验只核对预算守恒、箱宽和高频误差，不能写成真机成功率。",
    },
    learningMode: [
      "阅读",
      "互动实验",
      "CPU 机制实验",
    ],
  },
  {
    id: "38",
    slug: "vla-rlvr",
    shortTitle: "VLA可验证强化学习",
    unit: unitById["vla-post"],
    essentialQuestion:
      "示教里没有的失败，怎样写成第 17 课那种可验证奖励，并让组内相对优势在全失败批次上仍然有意义？",
    hook:
      "四条都没抓住时，稀疏成功的组内优势全是零。接触距离还能排出谁更接近，政策才知道往哪边改。",
    outcomes: [
      "把抓取成功写成确定性谓词，并把接触、接近、力超限写成可复核的 dense 奖励，不重推 GRPO。",
      "手算 Â_i = r_i − r̄；指出全失败或全成功组方差为零，标准化形式同样得不到更新。",
      "对照 ConRFT 的离线 BC+Q 与在线强化、SimpleVLA-RL 的 0/1 GRPO、VLA-RFT 的世界模型 verified reward，以及 SafeVLA 的代价约束。",
      "在失败批次上先预测再揭晓：稀疏优势全零，dense 仍有非零更新。",
    ],
    misconception: {
      myth: "接上 GRPO 就等于会从失败里学习；成功率奖励在还抓不住时也有梯度。",
      truth:
        "组内相对优势只比较同组分数。全是 0 或全是 1 时方差为零，更新为零。失败轨迹要有可验证的接触或过程分，才能排序。",
    },
    prerequisites: [
      "第 17 课：会写组内相对优势，知道零方差组没有更新。",
      "第 27 课或第 28 课：知道动作如何成为 token 或连续块，本课只改奖励。",
    ],
    labId: "lab-38-vla-rlvr",
    readingTime: 75,
    difficulty: "研究级",
    hardware: {
      minimum: "无 GPU 可读完全文并跑 CPU 机制实验",
      recommended: "1×24GB 试小规模仿真 rollout；复现 ConRFT 真机或 SimpleVLA-RL 论文规模需要机械臂或并行仿真",
      notes: "浏览器实验是教学模拟，不是模型输出。CPU 实验只核对奖励、优势和零方差更新。",
    },
    learningMode: [
      "稀疏对 dense",
      "零方差组",
      "先预测再揭晓",
    ],
  },
  {
    id: "39",
    slug: "long-horizon-memory",
    shortTitle: "子目标栈记忆",
    unit: unitById["vla-post"],
    essentialQuestion:
      "长程第二步失败后，该把历史塞进第 14 课那种 token 窗口，还是 pop 子目标栈只重试失败步？",
    hook:
      "CALVIN 上同一套 MCIL，短程 53.9%，五步 0.08%。掉点在交接处。把历史拼进 128K 窗口，会把已经打开的抽屉再拉一次。",
    outcomes: [
      "写出栈深 k 与窗口长度 T 的单位差异，拒绝用加长上下文代替已提交表。",
      "对照 CALVIN LH-MTLC：当前句成功才切下一句；状态差谓词下重做已提交步会得到假失败。",
      "在四步任务第二步失败时，先预测再揭晓：窗口回放重做第一步，pop 只重试失败步。",
      "把 SayCan / Inner Monologue / RT-H 写成对栈顶的改写，而不是对 KV cache 的续写。",
    ],
    misconception: {
      myth: "长程失败是因为上下文不够长，把历史和五句指令塞进 128K 窗口就能恢复。",
      truth:
        "机器人记忆是世界状态加可 pop 的子目标栈。窗口回放会重做已提交步；栈协议只重试失败步。k 与 T 不是同一个量。",
    },
    prerequisites: [
      "建议先修第 14 课。",
      "建议先修第 29 课。",
      "建议先修第 31 课。",
    ],
    labId: "lab-39-subgoal-memory",
    readingTime: 70,
    difficulty: "高级",
    hardware: {
      minimum: "无 GPU：阅读全文、跑浏览器四步失败实验和 CPU 栈回放",
      recommended: "若要复现 CALVIN / HULC 论文规模：单卡加载对应政策并跑官方仿真评测",
      notes: "浏览器实验是教学模拟，不是模型输出。CPU 实验只核对栈操作、提交不变式和窗口重放，不能写成 CALVIN 成功率。",
    },
    learningMode: [
      "栈与窗口对照",
      "先预测再揭晓",
      "失败恢复卡",
    ],
  },
  {
    id: "40",
    slug: "force-safety-cutoff",
    shortTitle: "力超限切断动作块",
    unit: unitById["vla-post"],
    essentialQuestion:
      "动作块执行中接触力超限时，为什么不能像语音 PAUSE 那样丢掉缓冲再重说，而必须停在当前姿态并丢掉剩余步？",
    hook:
      "说错一句话可以停嘴重说。倒热水时力已经超限，杯子不会退回超限前的位置。未执行的 chunk 可以丢，已经发生的接触不能 undo。",
    outcomes: [
      "写出切断规则：chunk 步 i 若 ||F_i|| > F_max，丢掉 i+1 以后的剩余步，进入 SAFE_HOLD。",
      "对照第 07 课 PAUSE：未播放 PCM 可丢可重说；接触力、洒出的水和移位的杯子不能回放。",
      "在倒水教具里调力阈值，验证超限后末端停住、剩余步为 0、物体位置不能回到超限前。",
      "区分运行时力门限、危险示范过滤、SafeVLA 的训练期约束，以及 SAFE 失败检测器：四者解决不同层的问题。",
    ],
    misconception: {
      myth: "力超限时发一个 PAUSE 或 REPLAN，世界就会像音频缓冲一样回到超限前。",
      truth:
        "PAUSE 只能停住还没发生的输出。已经测到的接触改变了物体位姿；SAFE_HOLD 必须保持当前姿态，剩余步清零，物体不得回绕。",
    },
    prerequisites: [
      "第 07 课：CONTINUE / PAUSE / REPLAN 与 pending PCM。本课只借用状态机，不实现音频路由。",
      "第 30 课：动作分块、剩余步和过期丢弃。本课在剩余步上加力门限。",
    ],
    labId: "lab-40-force-cutoff",
    readingTime: 65,
    difficulty: "高级",
    hardware: {
      minimum: "无 GPU：读完全文、跑浏览器倒水教具与 CPU 切断回放。",
      recommended: "真机核对需要带力/力矩传感的手臂；复现 SafeVLA 规模对齐需要多卡与 AI2-THOR 仿真。",
      notes: "教具和 CPU 实验证明切断协议可复核，不证明 ISO/TS 15066 合规，也不证明真机倒水成功率。",
    },
    learningMode: [
      "力门限切断",
      "SAFE_HOLD 姿态保持",
      "语音 PAUSE 对照",
      "文献对照",
    ],
  },
  {
    id: "41",
    slug: "discrete-any-to-any",
    shortTitle: "离散统一图文",
    unit: unitById["native-unified"],
    essentialQuestion:
      "把图像也变成离散 token 之后，怎样和文本共用一套 next-token，同时让理解看见全图、生成看不见未来像素？",
    hook:
      "第 20 课用连续 latent 做 flow。Chameleon、Emu3、Show-o 把图查成码本编号，理解与生成差在 mask，不差在另做一套分类器。",
    outcomes: [
      "能写出图像 token 与文本 token 共享的 softmax 交叉熵，并指出码本大小与空间格子数的乘积。",
      "能画理解 mask 与生成 mask：理解路径看全图，生成路径不能看未来像素 token。",
      "能对照 Chameleon 的全因果 early-fusion、Emu3 的纯 next-token、Show-o 的 NTP+MTP 与 omni-attention。",
      "能把 tokenizer 重建上限、码本越界、未来泄漏三件失败分开定位，不把它们写成 flow matching 问题。",
    ],
    misconception: {
      myth: "只要把图像离散化塞进 LLM 词表，理解和生成就自动是同一件事。",
      truth:
        "共享的是词表和 softmax；理解必须看见整张图，生成必须挡住未来像素。mask 写错，CE 再降也在泄漏或瞎猜。",
    },
    prerequisites: [
      "第 20 课：知道统一理解与生成有离散、离散扩散、连续 flow 三条路，本课只走离散。",
      "会写因果语言模型的 next-token 损失；不要求 GPU。",
    ],
    labId: "lab-41-discrete-unify",
    readingTime: 75,
    difficulty: "研究级",
    hardware: {
      minimum: "无 GPU 可读完全文并跑 CPU 机制实验",
      recommended: "1×24GB 可试 Show-o 1.3B 或 Emu3 tokenizer 的 encode-decode；复现论文规模需要多机多卡",
      notes: "浏览器实验是教学模拟，不是模型输出。CPU 实验只核对两种 mask 与码本索引范围。",
    },
    learningMode: [
      "VQ 词表",
      "理解/生成 mask",
      "共享 softmax",
      "文献对照",
    ],
  },
  {
    id: "42",
    slug: "interleaved-schedule",
    shortTitle: "交错生成日程",
    unit: unitById["native-unified"],
    essentialQuestion:
      "一段回复里先出字、再出图、再出声音时，KV 缓存和采样器为什么不是同一张图？图像内步怎样才不会污染文本的因果位置？",
    hook:
      "同一句回答可以是字-图-字，也可以是字-字-图。调换的是阶段日程，不是把三种模态倒进同一条 next-token。",
    outcomes: [
      "画出阶段日程：每个阶段的模态、提交长度、内步数和写入策略。",
      "写出按模态分段的注意力可见性，并指出图像内步不得出现在文本 KV 的因果位置。",
      "对照 Transfusion、Show-o2、BAGEL、Janus / Janus-Pro 的分流：AR+扩散、omni-attention+流匹配、噪声 VAE 隔离、理解/生成分编码。",
      "在字-图-字教具上验证：工作区策略泄漏为零；调换日程后输出顺序改变。",
    ],
    misconception: {
      myth: "统一模型只有一条 KV，图像去噪的每一步都该写成下一个文本 token。",
      truth:
        "共享的是已提交前缀。图像采样器在工作区里迭代，只有干净块或离散 id 进入文本可读的因果位置；调换日程会改变写出顺序。",
    },
    prerequisites: [
      "第 20 课：知道文本 CE 与图像 flow / 扩散可以接在同一骨干上，且 Janus 理解/生成编码可以分家。",
      "第 41 课规格：离散图像 token 与文本共用 next-token；本课不重做那条 CE。",
    ],
    labId: "lab-42-interleaved-schedule",
    readingTime: 75,
    difficulty: "研究级",
    hardware: {
      minimum: "无 GPU 可读完全文并跑 CPU 机制实验",
      recommended: "1×24GB 试很小的字-图交错前向；复现论文规模需要多节点 A100 / H100",
      notes: "浏览器实验是教学模拟，不是模型输出。CPU 实验只核对日程表、mask 和泄漏计数。",
    },
    learningMode: [
      "日程编排",
      "mask 泄漏对照",
      "先预测再揭晓",
    ],
  },
  {
    id: "43",
    slug: "audio-language-understand",
    shortTitle: "转写与指令跟随",
    unit: unitById["omni-ops"],
    essentialQuestion:
      "同一句语音上，听写交叉熵和指令跟随交叉熵的有效 token 集合差在哪一段？转写对了为什么仍可能指令错？",
    hook:
      "Librispeech 可以到 1.6%，口述 MMLU 可以只有 33.2。差不在 codec，差在 mask 盖住了听写还是盖住了执行。",
    outcomes: [
      "在同一条序列上画出 ASR mask 与指令 mask，并证明两个有效集合不相等。",
      "用手算说明复读模型的 ASR 损失可以很低、指令损失仍然很高。",
      "用 SALMONN 的任务过拟合解释：转写对了，讲故事跟随率仍可以是 0。",
      "对照 Qwen2.5-Omni 表 4：口述指令成绩不能用 WER 代替。",
    ],
    misconception: {
      myth: "WER 低就等于会听指令；助手角色里出现的字都可以算进同一张 loss。",
      truth:
        "WER 测的是听写参考。指令跟随的有效 token 是执行回复。两张集合不相等；转写对、指令错是合法失败态。",
    },
    prerequisites: [
      "第 01 课：知道 loss mask 只对助手目标算账。",
      "不必先重做第 03–04 课的 RVQ 与 delay schedule。",
    ],
    labId: "lab-43-audio-understand",
    readingTime: 65,
    difficulty: "进阶",
    hardware: {
      minimum: "无 GPU 可读完全文并跑 CPU 机制实验",
      recommended: "1×24GB 试公开 Instruct 权重的三条双协议探针；复现论文规模需要多卡与大规模语音指令数据",
      notes: "浏览器实验是教学模拟，不是模型输出。CPU 实验只核对两类 mask 的有效集合与玩具交叉熵。",
    },
    learningMode: [
      "阅读",
      "交互实验",
      "CPU 机制实验",
    ],
  },
  {
    id: "44",
    slug: "document-layout",
    shortTitle: "文档版面拆分",
    unit: unitById["omni-ops"],
    essentialQuestion:
      "第 23 课探针单图文字。文档还要版面、表格单元格、跨页引用。读对数字但框在表头，算不算看懂了发票？",
    hook:
      "合计是 32.00，模型也输出 32.00，框却盖在表头「金额」上。内容命中和框命中可以脱钩，只有两列都过才记版面命中。",
    outcomes: [
      "写出单元格命中公式：内容匹配且框 IoU 过阈值，并构造内容对、框在表头的反例",
      "用阅读顺序把双栏栅格扫描和版面顺序拆开，说明栅格会把右栏金额插入左栏条款",
      "把跨页 key-value 写成「键所在页 ∪ 值所在页」的可见性约束，缺页则字段失败",
      "对照 Donut 的 OCR-free JSON 生成和 Pix2Struct 的截图解析预训练，并与字段 F1 / TED 分列",
    ],
    misconception: {
      myth: "发票问答字符串对了，就等于模型读懂了表格和版面。",
      truth:
        "字符串可以从表头、邻行、页眉发票号或语言先验来。没有单元格框和阅读顺序，不能声称版面对。",
    },
    prerequisites: [
      "第 23 课：能把 OCR 拆成区域列和字符串列",
      "第 08 课：知道分辨率不够时小字会先糊掉，不要和版面失败混报",
    ],
    labId: "lab-44-document-layout",
    readingTime: 65,
    difficulty: "进阶",
    hardware: {
      minimum: "无 GPU：读完全文并跑 CPU 机制实验",
      recommended: "1×24GB 可对开源文档 VLM 做发票字段探针；复现 Donut 预训练为 64×A100、200k step，Pix2Struct-Base 为 64 TPU、270k step",
      notes: "CPU 实验证明内容命中与框命中可脱钩，不代表 CORD / DocVQA 真实成功率。浏览器实验是教学模拟。",
    },
    learningMode: [
      "版面命中公式",
      "发票单元格探针",
      "CPU 复核",
    ],
  },
  {
    id: "45",
    slug: "multimodal-retrieve",
    shortTitle: "长视频检索层",
    unit: unitById["omni-ops"],
    essentialQuestion:
      "长视频和长操作记录不能全塞进上下文时，该检索字幕、中间特征还是像素？错误层召回为何让精读永远看不到目标段？",
    hook:
      "一小时带子，目标在第 47 分钟。只检索字幕会错过无对白动作；只检索像素会把阅读预算烧光。",
    outcomes: [
      "把长视频检索拆成字幕、中间特征、像素三层索引，并写出每层能召回和不能召回的证据。",
      "用手算 Recall@k：目标不在该层索引时，任意 k 的召回为 0，阅读器看不见该段。",
      "比较三层的阅读预算，说明像素层 Top-k 或整段扫描如何超过上下文配额。",
      "对照第 14 课的窗口扩展和第 39 课的子目标栈，说明本课解决的是选层，不是把历史塞长或改栈操作。",
    ],
    misconception: {
      myth: "窗口加到 128K，或把整段历史、整段操作记录塞进上下文，长视频问题就解决了。",
      truth:
        "进不了当前阅读器的片段等于不存在。选错检索层时，目标段根本不会进入 Top-k，加长窗口和回放子目标都救不回来。",
    },
    prerequisites: [
      "第 14 课：知道窗口可运行、位置可外推、证据可使用是三件独立的事。",
      "第 39 课：长程失败时机器人记忆更多是状态和子目标栈，不是无限拼接历史。",
    ],
    labId: "lab-45-mm-retrieve",
    readingTime: 70,
    difficulty: "高级",
    hardware: {
      minimum: "无 GPU 可读完全文并跑 CPU 三层召回实验",
      recommended: "1×24GB 试开源 LVLM 加检索流水线；复现论文规模需要多卡、长视频语料和 ASR / OCR 工具链",
      notes: "浏览器实验是教学模拟，数字由夹具公式算出，不是 Video-RAG 或 Goldfish 的前向输出。CPU 实验只核对召回公式和预算。",
    },
    learningMode: [
      "分层索引",
      "Recall@k 夹具",
      "先预测再揭晓",
      "检索后再精读",
    ],
  },
  {
    id: "46",
    slug: "omni-serving",
    shortTitle: "多模态推理调度",
    unit: unitById["omni-ops"],
    essentialQuestion:
      "理解、语言生成、flow 采样和动作专家为什么不能捕获进同一条 CUDA graph，stage graph 怎样按阶段组批并隔离 KV？",
    hook:
      "三条请求（纯文本、带图、带动作专家）若强行合成一条静态 GPU 图，padding 会吞掉有效 token，或者动作专家读到别人的 KV 页。",
    outcomes: [
      "能画出理解编码、语言 decode、flow / 动作专家三条不同的 batch 维，并说明 CUDA graph 锁住了哪一类形状。",
      "能对两条变长视觉请求写出 padding mask，使无效 patch 不进入有效 token 计数和注意力分母。",
      "能区分合法的跨模态条件传递和错误的 KV 页别名。",
      "能按 vLLM-Omni 的 stage graph 把 Qwen-Omni 式 Thinker–Talker–Vocoder 拆开调度，并对照第 13 课的训练并行。",
    ],
    misconception: {
      myth: "把整个 Omni 模型捕获成一条 CUDA graph 就能同时加速理解和生成。",
      truth:
        "CUDA graph 要求静态形状与静态控制流。理解、AR decode 和 flow 的循环次数、张量秩都不同；合图只能靠 padding 或共用 KV，两者都会算错账单。",
    },
    prerequisites: [
      "第 13 课：知道 FSDP / EP / CP 是训练并行，本课不重做数值对账。",
      "建议第 20 课：见过理解路径和 flow 生成路径为什么目标不同。",
      "建议第 08 课：见过变长视觉 token；建议第 28 课：见过动作专家的时间步循环。",
    ],
    labId: "lab-46-serve-graph",
    readingTime: 70,
    difficulty: "高级",
    hardware: {
      minimum: "无 GPU 可读完全文并跑 CPU 机制实验",
      recommended: "复现 vLLM-Omni 论文数字需要双卡 80GB 与 Qwen-Omni 权重；本课不提供",
      notes: "浏览器实验是教学模拟，不是模型输出。CPU 实验只核对 padding mask 与有效 token 计数，不报告真实 JCT。",
    },
    learningMode: [
      "CUDA graph 合图",
      "padding mask",
      "KV 隔离",
      "stage 组批",
    ],
  },
  {
    id: "47",
    slug: "eval-taxonomy",
    shortTitle: "六类评测分桶",
    unit: unitById["omni-ops"],
    essentialQuestion:
      "MMMU、Video-MME、OmniBench、OSWorld、LIBERO、SIMPLER 各测什么、不测什么，为什么六类数字不能横着比？",
    hook:
      "同一张幻灯片上并排 76.5%、75%、56%、12%、r=0.924。它们单位不同、协议不同、成功定义不同。LIBERO 平均进不了真机能力这一格。",
    outcomes: [
      "给每条评测数字打上六类互斥标签，并拒绝把一类数字写成另一类能力",
      "为一条数字写出协议卡：基准、划分、模态、成功定义、样本量、是否 fine-tune、单位",
      "指出 MMMU 准确率测不了接地，Video-MME 测不了三模态硬约束，SIMPLER 的 r 不是成功率",
      "在分桶实验里把 LIBERO 平均拖进真机能力时看见标红，并沿用第 31 课 Wilson 区间只做同类比较",
    ],
    misconception: {
      myth: "评测分数都可以换成百分数再平均；LIBERO 高就等于真机强，Omni 高就等于六类都强。",
      truth:
        "六类是互斥账本。准确率、执行成功率、相关系数不是同一个量。LIBERO 是仿真套件，SIMPLER 测排序相关，都不能填进真机能力。",
    },
    prerequisites: [
      "第 01 课：golden case 分桶、禁止把官方评测写成自训复现",
      "第 23 课：VQA 答对不等于看对位置，幻觉率只在负例上算",
      "第 31 课：LIBERO / SIMPLER / CALVIN / 真机小样本不能横着比，并会算 Wilson 区间",
    ],
    labId: "lab-47-eval-taxonomy",
    readingTime: 70,
    difficulty: "高级",
    hardware: {
      minimum: "无 GPU：读完全文、跑浏览器分桶实验和 CPU 互斥标签夹具",
      recommended: "若要复现论文规模评测：单卡可跑开源 VLM 的 MMMU val / Video-MME 抽帧；OSWorld 需要虚拟机；LIBERO / SIMPLER 推理需要能加载 7B 级 VLA 的单卡；真机评测另需机械臂",
      notes: "本课不训新模型。CPU 实验与浏览器 Lab 是教学夹具，不能写成 MMMU、OSWorld 或真机成功率。",
    },
    learningMode: [
      "协议分类",
      "分桶记账",
      "非法映射标红",
      "CPU 互斥夹具",
    ],
  },
  {
    id: "48",
    slug: "speech-action-runtime",
    shortTitle: "双时钟状态表",
    unit: unitById["embodied-omni"],
    essentialQuestion:
      "第 19 课 Thinker–Talker 加上第 29 课快慢 VLA 之后，CONTINUE / PAUSE / REPLAN 在音频时钟和控制时钟上能否共用一行状态，却必须分列记录两个 available_at？",
    hook:
      "助手还在说“我去拿桌上的杯子”，有人把杯子挪走。停嘴是音频 PAUSE，改抓取是手臂 REPLAN。一次点击若同时撤回已播声音和已经发生的接触，历史就被伪造了。",
    outcomes: [
      "写出一行状态、两列时间戳：audio_available_at 与 action_available_at，并分别用音频帧和 H/f 定义过期。",
      "把第 07 课的 CONTINUE / PAUSE / REPLAN 接到控制时钟上，同时说明音频 PAUSE 不等于第 40 课的力切断。",
      "规定 REPLAN 同时取消未播 PCM 和未执行 chunk，且不得 undo 已播放音频或已发生接触。",
      "在双时钟教具里用两次独立事件完成停嘴和重规划，并在 CPU 回放里证明旧 PCM 与旧剩余步不再执行。",
    ],
    misconception: {
      myth: "语音打断和手臂重规划是同一按钮：停嘴就会自动撤回已播声音，也会把已经碰到的杯子退回原处。",
      truth:
        "它们可以共用一张状态表和同一组动词，但必须分列时钟。PAUSE 只冻结未播出队；REPLAN 只取消未来队列。已播 PCM 和已发生接触是历史，不能 undo。",
    },
    prerequisites: [
      "第 07 课：CONTINUE / PAUSE / REPLAN 与 available_at。",
      "第 19 课：Thinker–Talker 与双工回放边界。",
      "第 29 课：快慢双时钟与子目标过期。",
      "第 30 课：开环窗口 H/f 与过期 chunk。",
      "第 40 课规格：力超限后 SAFE_HOLD，接触不可回放。",
    ],
    labId: "lab-48-duplex-body",
    readingTime: 80,
    difficulty: "研究级",
    hardware: {
      minimum: "无 GPU：读完全文、跑浏览器双时钟教具与 CPU 状态表回放。",
      recommended: "对照 Qwen2.5-Omni / GR00T / π0 的公开权重需要对应论文里的推理卡；本课不要求加载这些权重。",
      notes: "产物是接口说明书加教学状态机，不是发布级双工机器人。教具和 CPU 实验证明协议可复核，不证明真机成功率，也不声称复现 GPT-4o 或 Helix。",
    },
    learningMode: [
      "双时钟状态表",
      "先预测再揭晓",
      "CPU 事件回放",
      "文献对照",
    ],
  },

  {
    id: "49",
    slug: "video-generation",
    shortTitle: "视频生成拆账",
    unit: unitById["gen-native"],
    essentialQuestion: "生成未来帧的训练目标和理解 caption 的交叉熵能否共用有效位置？物体永久性、相机轨迹与 Video-MME 为什么不是同一张表？",
    hook: "同一段桌上有杯子的短视频。理解 CE 答“还在”，把下一帧涂成均值以后杯子占用掉到 0，帧差还可能比抄上一帧更低。",
    outcomes: [
      "写出理解 CE 与生成帧差（或 v-prediction / flow）的有效位置，并证明两张 mask 不相交。",
      "造出理解答对、生成帧物体消失的夹具，且不把 Video-MME 准确率写成生成质量。",
      "对照 CogVideoX 的 3D VAE 与专家 Transformer、HunyuanVideo 的双流到单流与 14 类相机标注，不编造 Sora 层数或参数量。",
      "把物体永久性、相机可控、VBench 动态度与 Video-MME 分列记账。",
    ],
    misconception: {
      myth: "会看视频就会生成视频；Video-MME 高就等于下一帧杯子还在，帧差低就等于物体永久。",
      truth: "理解 CE 写在答案 token，生成损失写在未来格子。位置不相交。均值填充可以让 L2 下降同时把杯子抹掉。",
    },
    prerequisites: [
      "第 09 课：视频有时间轴，本课不重做 TMRoPE 或音视频交错。",
      "第 10 课：理解侧 token 压缩，本课不重做 Pareto。",
      "第 20 课：图像理解与生成分路径；本课把分账接到视频。",
      "第 33 课：像素 L2 与表征预测的差别；本课不重做 JEPA。",
      "第 47 课：Video-MME 属于 C2 理解账，不能横着写成生成能力。",
    ],
    labId: "lab-49-video-generation",
    readingTime: 80,
    difficulty: "研究级",
    hardware: {
      minimum: "无 GPU：读完全文、跑浏览器分账教具和 CPU 不相交 mask 夹具",
      recommended:
        "若要跑公开权重：CogVideoX-2B 约 18GB、5B-480p 约 26GB（论文 H800、50 步）；HunyuanVideo 1.5 在卸载后 720p 121 帧峰值约 13.6GB。本课不要求加载这些权重",
      notes:
        "浏览器实验是教学模拟，不是模型输出。CPU 实验只核对 5 帧历史加 1 帧生成上的 CE mask、帧差位置和杯子占用，不能写成 VBench 或真机能力。",
    },
    learningMode: [
      "损失分账",
      "先预测再揭晓",
      "CPU 不相交 mask",
      "文献对照",
    ],
  },
  {
    id: "50",
    slug: "3d-generation",
    shortTitle: "三维资产分解码",
    unit: unitById["gen-native"],
    essentialQuestion: "网格、高斯和辐射场为何不能共用一个解码器？同一份结构化 latent 怎样分出三种资产，且改高斯半径不得改 mesh 拓扑？",
    hook:
      "第 35 课的三维点是抓取闭合点。本课同一份 SLAT 要导出 mesh、3D Gaussian 和辐射场。三种输出层的 shape 不同；把高斯半径写回共享 latent，另两路必须失败。",
    outcomes: [
      "写出 SLAT 为 {(z_i, p_i)}，并说明稀疏占用与局部向量各保住什么。",
      "分别写出 mesh / 3D Gaussian / 辐射场解码器的输出 shape 契约，指出它们不能共用最后一层。",
      "构造“加大高斯半径、mesh 拓扑不变”的夹具，以及“解码器写回 latent、另两路失败”的对照。",
      "对照 TRELLIS 的两段 rectified flow、冻结编码器再训另两路解码器，以及和第 35 课闭合点的边界。",
    ],
    misconception: {
      myth: "三维统一 latent 意味着一个解码器能同时吐出网格、高斯和辐射场；改高斯球大小只是外观，网格会自动跟着对。",
      truth: "统一的是 SLAT，不是输出层。高斯半径是 D_GS 的局部属性。写回 z_i 才会改 SDF 符号，那时辐射场因子也会坏。",
    },
    prerequisites: [
      "第 20 课：连续 latent 与 flow matching 的速度场，本课把样本从图像 latent 换成结构化 3D latent。",
      "第 35 课：深度服务抓取闭合点。本课产出可导出资产，不报接触带。",
    ],
    labId: "lab-50-trellis-slat",
    readingTime: 75,
    difficulty: "研究级",
    hardware: {
      minimum: "无 GPU 可读完全文并跑 CPU 机制实验",
      recommended: "官方仓库声明推理至少 16GB NVIDIA GPU；复现 XL 为 64×A100 40G、40 万步",
      notes: "浏览器实验是教学模拟，不是 TRELLIS 前向。CPU 实验只核对三种解码 shape 与写坏共享 latent 的失败传播。",
    },
    learningMode: [
      "SLAT 只读",
      "三路解码器",
      "先预测再揭晓",
      "shape 契约",
    ],
  },
  {
    id: "51",
    slug: "multimodal-cot",
    shortTitle: "视觉分步推理",
    unit: unitById["reason-agent"],
    essentialQuestion: "直接出答案会跳过看见了什么。分步推理的奖励对象，为什么必须是过程有没有引用视觉证据，而不能只看最终数字对不对？",
    hook: "红色杯子有两只，模型写「2」完全正确，推理栏却一个格子都没点。关掉必须引用之后，答案对、引用格为空。",
    outcomes: [
      "把一次生成拆成推理 token 集合与答案 token 集合，并证明两个位置集合不相交",
      "写出过程奖励：推理 span 未引用真值格子则为 0，即使最终数字正确",
      "对照第 17 课的答案奖励与第 38 课的接触奖励，说明本课只改奖励对象",
      "在计数教具里先预测再揭晓：关掉必须引用后答案对、引用格为空",
    ],
    misconception: {
      myth: "只要最终数字对，分步推理就算看见图了；强化学习奖对答案就等于奖了视觉证据。",
      truth: "答案可以从语言共现来。过程奖励只认推理 span 里的格子引用。引用为空时过程分为 0。",
    },
    prerequisites: [
      "第 17 课：会写可验证奖励和组内相对优势",
      "第 23 课：能把文字对和格子命中拆开",
      "第 38 课：知道同一套组内结构可以只改 r 的定义",
    ],
    labId: "lab-51-multimodal-cot",
    readingTime: 70,
    difficulty: "研究级",
    hardware: {
      minimum: "无 GPU：读完全文、跑浏览器计数教具与 CPU 机制实验",
      recommended:
        "复现 LLaVA-CoT 为单节点 8×H100 全参微调；Vision-R1 冷启动用 Llama-Factory 做 2 个 epoch，强化学习在 Verl 上跑两阶段 PTST",
      notes:
        "浏览器实验是教学模拟，不是模型输出。CPU 实验只核对 token 分账与过程奖励，不代表 MathVista 或 MMStar 成功率。Vision-R1 强化学习的精确卡时其附录未写成可引用的 GPU·小时。",
    },
    learningMode: [
      "阶段标签",
      "过程奖励",
      "先预测再揭晓",
      "CPU 复核",
    ],
  },
  {
    id: "52",
    slug: "multimodal-tools",
    shortTitle: "看图后调工具",
    unit: unitById["reason-agent"],
    essentialQuestion: "看见图上的数字、框和物体之后，何时必须调用计算器、裁剪、深度或搜索，而不能把下一句生成当成已经算完、看清或查过？",
    hook:
      "发票三行 18.90、26.50、15.80，模型把数字读对了，心算漏掉十位进位写成 51.20。同一组数字交给计算器才是 61.20。看见了不等于该直接答。",
    outcomes: [
      "写出工具调用的合法性：名字在目录里、必填参数齐、类型过关，缺一项不得进入执行。",
      "把发票小计拆成三列：OCR 是否命中印刷数字、心算是否漏进位、计算器是否等于真值。",
      "区分第 32 课的屏幕点击、第 44 课的版面单元格、第 45 课的检索层，本课只管通用工具。",
      "对照 LLaVA-Plus、GPT4Tools、ViperGPT、V* 与 ToRA，说明训练型调用和提示链调用差在规划器是不是看见了图。",
    ],
    misconception: {
      myth: "视觉语言模型看见图就会算术、会看清小字、会查外部事实；再加一个会点屏幕的头，工具问题就结束了。",
      truth: "看见数字、框对单元格、检索到片段，都只提供参数。计算器、裁剪、深度、搜索是另一条执行通道。缺参数的调用必须拒绝，不能用下一句文本顶上。",
    },
    prerequisites: [
      "第 23 课：能把 OCR 字符串和区域命中拆开。",
      "第 32 课：知道 GUI 点击是空间接地，不是通用工具目录。",
      "第 44 课：知道字段字符串对不等于版面对。",
      "第 45 课：知道长视频先选检索层，工具参数只能指向已召回片段。",
    ],
    labId: "lab-52-multimodal-tools",
    readingTime: 70,
    difficulty: "高级",
    hardware: {
      minimum: "无 GPU 可读完全文并跑 CPU 工具 schema 校验",
      recommended: "1×24GB 可试开源视觉语言模型加计算器或裁剪脚本；复现 LLaVA-Plus / GPT4Tools 规模需要多卡与视觉工具链",
      notes:
        "浏览器实验是教学模拟，数字由夹具公式算出，不是 LLaVA-Plus 或 GPT-4V 的前向输出。CPU 实验只核对 schema 与进位，不证明真实发票准确率。",
    },
    learningMode: [
      "工具 schema 校验",
      "发票进位夹具",
      "先预测再揭晓",
      "看图后调用",
    ],
  },
  {
    id: "53",
    slug: "cross-session-memory",
    shortTitle: "跨会话记忆",
    unit: unitById["reason-agent"],
    essentialQuestion: "隔天再问桌上杯子的颜色时，该把昨日原图、空间框还是一句话摘要写进外部记忆？",
    hook: "昨天是红杯子，今天换成蓝的。只存摘要会答错颜色；只存原图像素会打穿字节上限。",
    outcomes: [
      "写出一条跨会话记录的 schema：摘要、框、像素分列，并按 UTF-8 与 H×W×3 计算字节。",
      "用三条记录核对字节上限：全留像素超限，过期删像素留摘要后回到上限内。",
      "说明不可改写的首条摘要会把昨日红色带进今日答案，最新观察必须改写同一实体。",
      "对照第 14 课窗口、第 39 课子目标栈和第 45 课检索层，说明本课对象是会话关闭后仍在的 payload。",
    ],
    misconception: {
      myth: "把昨天的图和对话继续塞进 128K 窗口，或者像第 39 课那样 pop 子目标，隔天就能答对杯子颜色。",
      truth: "关会话之后窗口和栈都清空。隔天能用的只有外部记录。只留摘要会冻结昨日颜色；只留原图会超过字节上限。",
    },
    prerequisites: [
      "第 14 课：窗口可运行、位置可外推、证据可使用是三件独立的事。",
      "第 39 课：一轮任务内的记忆是世界状态加可 pop 的子目标栈。",
      "第 45 课：长视频先选检索层，未入 Top-k 的片段阅读器看不见。",
    ],
    labId: "lab-53-session-memory",
    readingTime: 60,
    difficulty: "高级",
    hardware: {
      minimum: "无 GPU：读完全文、跑浏览器红蓝杯子教具与 CPU 三条记录实验。",
      recommended: "对照 LoCoMo / MIRIX / M3-Agent 的公开权重或截屏流水线需要各论文自己的推理卡；本课不要求加载这些权重。",
      notes:
        "浏览器实验是教学模拟，数字由夹具公式算出，不是 MemoryBank 或 M3-Agent 的前向输出。CPU 实验只核对字节上限和过期协议。",
    },
    learningMode: [
      "跨会话 payload",
      "先预测再揭晓",
      "字节上限夹具",
      "过期删像素",
    ],
  },
  {
    id: "54",
    slug: "synthetic-data",
    shortTitle: "合成补分布",
    unit: unitById["data-deploy"],
    essentialQuestion: "MimicGen / RoboCasa 一类合成操作数据补的是长尾分布，还是把已经最多的域再复制一遍？",
    hook:
      "再生成两千条厨房抓放，α=1 的有效域数会掉。同样两千条若去补双臂摆盘，小域才重新出现在 batch 里。复制同一条轨迹，有效样本量不升反降。",
    outcomes: [
      "写出合成前后 p_d ∝ n_d^α 与有效域数 D_eff，并指出只灌最大域时 α=1 更糟、补最小域时小域频率回升。",
      "把 MimicGen 的物体中心切段、相对位姿变换和成功才入库写成可复核步骤，并区分生成成功率与政策成功率。",
      "用源 ID × 复位箱定义唯一哈希，证明重复轨迹不增加有效样本量 n_eff。",
      "对照 RoboCasa 的原子任务规模化与复合任务仍然接近零成功，列出可执行的缩小版配比实验。",
    ],
    misconception: {
      myth: "合成轨迹条数越多越好。把 MimicGen 生成的演示全部倒进已经最大的域，模型就会自动覆盖长尾。",
      truth: "α=1 时条数就是采样权重。合成若只复制最大域，有效域数下降。重复同一条源轨迹时唯一哈希不变，n_eff 还会被赫芬达尔公式压低。",
    },
    prerequisites: [
      "26",
      "建议 34",
    ],
    labId: "lab-54-synth-data",
    readingTime: 70,
    difficulty: "高级",
    hardware: {
      minimum: "无 GPU：读完全文，跑浏览器互动实验和 CPU 机制实验",
      recommended: "1×24GB 仅用于打开 robosuite / RoboCasa 公开加载器看字段，不要求复现 10 万条生成或真机采集",
      notes:
        "MimicGen 论文规模是约 200 条人类演示生成 5 万+ 条；RoboCasa 原子任务生成 10 万条。本课主路径不跑这些生成器。CPU 数字不能写成真机成功率。",
    },
    learningMode: [
      "阅读",
      "互动实验",
      "CPU 机制实验",
    ],
  },
  {
    id: "55",
    slug: "ondevice-quant",
    shortTitle: "端侧量化损伤",
    unit: unitById["data-deploy"],
    essentialQuestion: "INT8 / INT4 之后，同一条多模态序列上文本、视觉和动作哪一类 token 先跳类？",
    hook: "把三个头量化到 4 bit 时，动作 pitch 已经换箱，文本 top-1 还在原词上。平均 CE 看不出这件事。",
    outcomes: [
      "写出对称 absmax 量化与动作均匀分箱，并说明箱边界上的值为什么比大间隔的文本 logit 先换类。",
      "在同一隐藏向量上分列文本 top-1、视觉 L2 和动作 bin，拒绝合成一个“量化分数”。",
      "对照第 13 课的训练并行和第 46 课的 stage graph，说明本课量的是推理权重损伤。",
      "按原文引用 GPTQ / AWQ / SmoothQuant / LLM.int8 / QVLA / BitVLA 的表内数字，不编端侧延迟。",
    ],
    misconception: {
      myth: "困惑度或文本 CE 没崩，就说明 4 bit 对视觉和动作也安全。",
      truth: "文本 top-1 有间隔保护；动作 bin 是硬边界。同一套 4 bit 可以让 CE 略降、pitch 已经换箱。",
    },
    prerequisites: [
      "第 24 / 25 课：均匀分箱与箱宽。",
      "对照第 13 课：训练并行不是本课对象。",
      "对照第 46 课：推理 stage graph 不是本课对象。",
    ],
    labId: "lab-55-ondevice-quant",
    readingTime: 70,
    difficulty: "高级",
    hardware: {
      minimum: "无 GPU 可读完全文并跑 CPU 机制实验",
      recommended: "复现 GPTQ / AWQ / QVLA 论文表需要对应论文里的 GPU；本课不提供",
      notes: "浏览器实验是教学模拟，不是模型输出。CPU 实验只核对分模态误差与 bin 跳转，不报告真机延迟。",
    },
    learningMode: [
      "8/4 bit 对照",
      "先预测再揭晓",
      "分模态误差",
      "文献对照",
    ],
  },
  {
    id: "56",
    slug: "generation-reward",
    shortTitle: "生成偏好打分",
    unit: unitById["data-deploy"],
    essentialQuestion: "图像或视频生成的偏好奖励与像素 L2 何时会给出相反的排序，为什么不能只用重建误差验收生成质量？",
    hook: "一张过平滑的图 L2 更低，一张锐利但像素错位的图人更喜欢。FID 还能和人类排序负相关。生成账本必须单列偏好序。",
    outcomes: [
      "写出像素 L2 / FID 与 Bradley–Terry 偏好奖励的差别，并说明 MMSE 均值为什么发糊",
      "用 ImageReward、PickScore、HPSv2 的公开表说明自动指标何时跟上人类、何时和 FID 反向",
      "计算两组排序的 Kendall τ，并在 L2 序与偏好序相反时拒绝“L2 更低就是更好”",
      "在浏览器里先预测再揭晓，造出过平滑 L2 更低、偏好分也更低的一对",
    ],
    misconception: {
      myth: "生成质量就是重建误差：L2 更低、FID 更低、PSNR 更高，图就更好，人也一定更喜欢。",
      truth: "MSE 最优是后验均值，会把多种锐利可能平均成糊图。人偏好硬边、对齐和少畸变。L2 序与偏好序可以完全相反。",
    },
    prerequisites: [
      "第 16 课：偏好对、Bradley–Terry 与 DPO 隐式奖励，针对的是回答文本",
      "第 20 课：flow matching 在 VAE latent 上回归速度，重建目标是连续 L2",
      "第 21 课：CLIP 余弦可当图文相似度，但不能直接当人类偏好",
      "第 49 课规格：生成帧差和理解 CE 必须分账，本课接管生成这一侧用什么尺子",
    ],
    labId: "lab-56-generation-reward",
    readingTime: 80,
    difficulty: "研究级",
    hardware: {
      minimum: "无 GPU：读完全文、跑浏览器两图对打教具与 CPU Kendall 夹具。",
      recommended:
        "若要复现 ImageReward / PickScore / HPSv2 推理：单卡可加载对应公开权重；ReFL 或 Diffusion-DPO 微调需要能加载 SD / SDXL 的训练卡。本课不要求加载这些权重。",
      notes: "教具和 CPU 实验证明 L2 与偏好可以反向，不证明真实 ImageReward 分数，也不能写成 FID 或人类 MOS。",
    },
    learningMode: [
      "先预测再揭晓",
      "两序对打",
      "CPU Kendall 夹具",
      "文献对照",
    ],
  },
  {
    id: "57",
    slug: "data-provenance",
    shortTitle: "训练图像出处",
    unit: unitById["data-deploy"],
    essentialQuestion: "一张图进训练集前要留下哪些可核查证据：许可、来源 URL、内容哈希、是否合成、是否可撤回，缺哪一项就必须拒收？",
    hook: "路径写进了文件列表，不等于这张图能进训练集。许可空着、哈希空着，或哈希对不上字节，这一行必须拒收。水印检测率不能代替这两项。",
    outcomes: [
      "写出六项必填 sidecar 字段，并把缺许可或缺哈希判为非法",
      "用 SHA-256 硬绑定核对图像字节，区分 C2PA 硬绑定与软绑定（水印或指纹）",
      "把合成标记、来源 URL 和可撤回标记写成可过滤字段，而不把它们当成质量分数",
      "对照 Data Provenance Initiative 的许可审计和 C2PA 公开规格，列出可执行的缩小版准入实验",
    ],
    misconception: {
      myth: "图下载下来、水印检测器亮了，或 C2PA 小徽章在，就可以进训练集。",
      truth:
        "训练集准入是字段谓词。缺许可或缺哈希为非法。C2PA 验证的是签名与绑定，不代替许可合同；软绑定检测是概率事件，本课不把检测率写进 Gate。",
    },
    prerequisites: [
      "01",
      "建议 26",
      "对照 54 规格",
    ],
    labId: "lab-57-provenance",
    readingTime: 60,
    difficulty: "进阶",
    hardware: {
      minimum: "无 GPU：读完全文，跑浏览器三条样本教具和 CPU 机制实验",
      recommended: "1×24GB 仅用于打开公开图像索引或 C2PA 校验工具看字段，不要求复现任何生成模型",
      notes: "C2PA / 水印只按公开规格写，不编检测率。CPU 数字不能写成真机或真实爬虫合规率。",
    },
    learningMode: [
      "阅读",
      "互动实验",
      "CPU 机制实验",
    ],
  },
  {
    id: "58",
    slug: "medical-vlm",
    shortTitle: "医学图文拆配方",
    unit: unitById["domain-research"],
    essentialQuestion: "第 22 课的自然图像 VLM 配方，为什么不能原样搬到医学图文上？报告字段、多图检查和无框病灶断言分别要改哪一条协议？",
    hook: "关掉“禁止无框断言”之后，一张什么都没有的示意胸片仍会写出肺炎。那不是临床判断，是语言先验穿过了错误的损失掩码。",
    outcomes: [
      "对照第 22 课写出 LLaVA-Med 的两阶段：600K 生物医学概念对齐与 60K 指令微调各自冻什么、训什么",
      "把医学报告的 INDICATION / COMPARISON / FINDINGS / IMPRESSION 拆成与自然 caption 不同的 loss mask",
      "定义无框肯定计数 U，并在空图上证明关掉门控后 U≥1",
      "把封闭集准确率和开放集 recall 分列，拒绝用 caption 指标代替病灶协议",
    ],
    misconception: {
      myth: "医学图文只是换一批图和更长的 caption，LLaVA 的冻视觉、训投影、再训 LLM 原样复用即可。",
      truth:
        "顺序可以借用，数据和验收不能借用。报告字段不能整段当 caption 算损失；空图上的无框阳性必须单独计数。本课只讲训练与评测协议，不构成临床建议。",
    },
    prerequisites: [
      "第 22 课：冻视觉、只训投影、再动 LLM 的自然图像配方",
      "第 23 课：答对不等于看对位置；本课把该纪律改写成病灶框",
      "第 01 课：assistant-only loss mask 与 golden case",
    ],
    labId: "lab-58-medical-unboxed",
    readingTime: 80,
    difficulty: "研究级",
    hardware: {
      minimum: "无 GPU：读完全文，跑浏览器胸片门控实验和 CPU 无框断言 / mask 夹具",
      recommended:
        "复现 LLaVA-Med 公开配方需要 8×A100；论文表 5 记录 batch 128 时第一阶段 1 epoch 约 6.8 小时、60K 指令 3 epoch 约 8.0 小时",
      notes: "本课不提供临床建议。CPU 实验和浏览器 Lab 是教学夹具，不能写成影像诊断准确率，也不能标成复现了 LLaVA-Med 的 VQA 分数",
    },
    learningMode: [
      "阅读",
      "浏览器实验",
      "CPU 机制实验",
    ],
  },
  {
    id: "59",
    slug: "music-sound",
    shortTitle: "音乐和环境声",
    unit: unitById["domain-research"],
    essentialQuestion: "同一套 8 路语音码本上，为什么语音句可以仍可懂，鼓点网格却塌掉？事件边界 F1 和语音 WER 为什么不能共用一个分数？",
    hook:
      "80 ms 一帧的语音 RVQ 听写可以 WER 为 0，20 ms 间距的鼓点 flam 会被并进同一格。可懂不是音高，也不是 onset。",
    outcomes: [
      "说明音乐要保住的是音高与节拍网格，环境声要保住的是事件边界，二者都不是语音可懂度。",
      "用手算证明 12.5 Hz 帧把 20 ms flam 并进同一格：Recall = 1/2，F1 = 2/3，而语音 WER 仍为 0。",
      "写出 WER 与事件 F1 的标签契约：词序列对毫秒时间戳，不得合成 (1-WER+F1)/2。",
      "对照 MusicGen 的 32 kHz / 50 Hz tokenizer 和 AudioLDM 的连续 latent，指出它们不是「再接一路 Mimi」。",
    ],
    misconception: {
      myth: "8 路 RVQ 已经能重建语音，接上音乐和环境声只是另一路同样的 codec；用一个音频分数就能验收。",
      truth:
        "语音 codec 的帧率和残差是按音节与共振峰训的。同一网格会让鼓点并帧、让谐波糊掉。WER 与事件 F1 使用不同标签，共用分会把网格塌缩藏过去。",
    },
    prerequisites: [
      "第 03 课：知道 RVQ、码率和重建不等于可预测。",
      "第 04 课：知道 delay schedule，本课不重做。",
      "第 43 课：知道转写层和指令层要分账，本课把分账从文本 mask 转到音频域标签。",
    ],
    labId: "lab-59-music-sound",
    readingTime: 70,
    difficulty: "高级",
    hardware: {
      minimum: "无 GPU 可读完全文并跑 CPU 机制实验",
      recommended:
        "1×24GB 试公开 EnCodec / MusicGen / AudioLDM 权重的三条探针；复现论文规模需要多卡与许可音乐或 AudioCaps",
      notes: "浏览器实验是教学模拟，不是模型输出。CPU 实验只核对 WER 与事件 F1 的标签家族、80 ms 并帧和非法共用分。",
    },
    learningMode: [
      "阅读",
      "交互实验",
      "CPU 机制实验",
    ],
  },
  {
    id: "60",
    slug: "living-protocol",
    shortTitle: "新论文收编卡",
    unit: unitById["domain-research"],
    essentialQuestion: "课会过时之后，一篇新论文怎样填收编卡、接到已有课桶，并且在缺 N、缺套件、把 LIBERO 写成真机时被拒收？",
    hook: "每月都会多一张 VLA 总表。本课不增加新模型，只增加一张卡：进哪一课、规模还是机制、缩小版能不能复现方向、缺哪条协议字段。",
    outcomes: [
      "给一篇新论文的每一行数字填收编卡：课桶、规模或机制、第 47 课类标签、第 31 课套件、N、单位、是否 fine-tune、缩小版能否复现方向",
      "拒绝缺 N、缺套件的 LIBERO 行，并拒绝把 LIBERO 宏平均写成真机能力",
      "把规模声明接到已有课当对照，不新开模型课；只对机制声明判断缩小版能否复现同方向趋势",
      "在虚构新 VLA 表上先预测哪几张卡会标红，再拖桶，并让 CPU 夹具与第 31、47 课标签保持兼容",
    ],
    misconception: {
      myth: "新论文分数更高，就要新开一课、换一个骨干；LIBERO 平均可以先写进真机节撑场面，N 和套件以后再补。",
      truth: "收编只引入规则。缺字段不得入账。LIBERO 是 C5 仿真操作。参数变大是规模，接到第 27 课当对照，不是新模型课。",
    },
    prerequisites: [
      "第 01 课：golden case 分桶、禁止把官方评测写成自训复现",
      "第 31 课：LIBERO 四套件、SIMPLER 的 r、Wilson 区间、真机小样本不能横着比",
      "第 47 课：六类互斥标签，LIBERO 平均进不了真机能力",
    ],
    labId: "lab-60-living-protocol",
    readingTime: 70,
    difficulty: "进阶",
    hardware: {
      minimum: "无 GPU：读完全文、跑浏览器收编实验和 CPU 必填字段夹具",
      recommended: "若要对照论文规模评测：单卡可加载开源 VLA 跑 LIBERO 推理；真机评测另需机械臂。本课不要求这些。",
      notes:
        "本课不训新模型，不引入新骨干。CPU 实验与浏览器 Lab 是教学夹具，不能写成 LIBERO、SIMPLER 或真机成功率，也不能把虚构 NovaVLA 表当文献。",
    },
    learningMode: [
      "收编卡",
      "分桶标红",
      "先预测再揭晓",
      "CPU 必填字段",
    ],
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
