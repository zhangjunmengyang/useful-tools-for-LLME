// Learn WM 概念词典。定义从 CURRICULUM.md 的讲法出发：先一句人话，
// 再给"检查时看什么"。词条会同时用于 /glossary 页、⌘K 搜索和正文划词弹层。

export type GlossaryTerm = {
  term: string;
  alias?: string;
  definition: string;
  diagnostic: string;
  lessons: string[];
};

export function glossaryAnchor(term: string) {
  return `term-${term
    .toLocaleLowerCase("en-US")
    .replace(/[^a-z0-9㐀-鿿]+/g, "-")
    .replace(/^-+|-+$/g, "")}`;
}

export const glossary: GlossaryTerm[] = [
  {
    term: "世界模型",
    alias: "World Model",
    definition:
      "看着现在的状态、掂量要做的动作，推演接下来会发生什么的模型。它的产出是可能的未来，用来支撑决策，或者回答“接下来会怎样”。",
    diagnostic:
      "看它接不接收动作输入、输出的预测能不能用于决策。只会无条件生成好看视频的，先别叫世界模型。",
    lessons: ["01", "12", "16"],
  },
  {
    term: "观察 vs 状态",
    definition:
      "观察是传感器直接给的原始数据（像素、视频帧、关节角、人脸框）；状态是模型内部对“现在什么情况”的压缩描述。观察可以冗余、缺失、带噪声，状态必须够用来做预测。",
    diagnostic:
      "检查模型预测下一步时吃的是原始观察还是压缩后的状态，再检查压缩丢掉的信息是不是任务需要的。",
    lessons: ["01", "02", "29"],
  },
  {
    term: "latent",
    alias: "潜变量 / 潜空间",
    definition:
      "把高维观察压成的低维向量。世界模型大多在 latent 空间里做预测，因为逐像素预测太贵，而且大部分像素与决策无关。",
    diagnostic:
      "在 latent 上做插值和逐维遍历，看变化是否对应可解释的语义（路弯了、车偏了）。全是噪声说明编码没学好。",
    lessons: ["02", "05", "08"],
  },
  {
    term: "动作条件",
    alias: "action-conditioned",
    definition:
      "预测下一状态时把动作当作输入。同一状态配不同动作，预测必须不一样，模型才算真把动作用起来了。",
    diagnostic:
      "做动作对换对照：同一状态换动作，看预测分不分岔。不分岔就是“动作盲”，这样的模型没法用于规划。",
    lessons: ["03", "11", "14"],
  },
  {
    term: "rollout",
    definition:
      "让模型从当前状态出发，把自己的预测再喂给自己，一步接一步推演出的一段未来轨迹。",
    diagnostic:
      "看误差怎么随步数累积。一步预测准不代表十步准，多步漂移曲线才是 rollout 质量的真话。",
    lessons: ["03", "04", "17"],
  },
  {
    term: "teacher forcing",
    definition:
      "训练自回归模型时，每一步都喂真实历史，不喂模型自己的上一步输出。训练省事，但模型没练过“自己接自己”。",
    diagnostic:
      "对比 teacher-forced 误差和自由 rollout 误差。差距大，说明训练状态和使用状态脱节。",
    lessons: ["03", "09"],
  },
  {
    term: "exposure bias",
    definition:
      "teacher forcing 的后遗症：推理时模型只能吃自己生成的上一步，小误差会被反复放大，越走越偏。",
    diagnostic:
      "画多步漂移曲线，看误差是否随步数超线性增长；再看误差大的位置是不是训练分布之外的状态。",
    lessons: ["03", "04"],
  },
  {
    term: "RSSM",
    alias: "循环状态空间模型",
    definition:
      "世界模型的记忆器官，把状态拆成两条通道：确定通道像行车记录仪，忠实累积看过的历史；随机通道像押注，对“现在到底什么情况”保留几种可能。",
    diagnostic:
      "分别砍掉一条通道做消融：长历史记不住，是确定通道的问题；噪声世界里赌不准，是随机通道的问题。",
    lessons: ["05", "06"],
  },
  {
    term: "MPC",
    alias: "模型预测控制",
    definition:
      "每一步都用模型往前推演若干条候选动作序列，挑得分最高那条，只执行它的第一步，下一步再重新规划。",
    diagnostic:
      "调规划视野和候选数量，看控制分数和耗时怎么变；视野加长反而变差，通常是模型多步误差在拖后腿。",
    lessons: ["05", "08", "14"],
  },
  {
    term: "CEM",
    alias: "交叉熵方法",
    definition:
      "一种挑动作序列的办法：随机撒一把候选，留下得分最高的一小撮，用它们的均值方差再撒下一把，几轮后收敛到好动作附近。",
    diagnostic:
      "看精英比例和迭代轮数变化时规划得分是否稳定；得分抖得厉害，多半是评分模型本身不可靠。",
    lessons: ["05", "14"],
  },
  {
    term: "actor-critic",
    definition:
      "策略网络（actor）管选动作，价值网络（critic）管估计“这样走下去值多少分”，两个一起训。Dreamer 把这对搭档整个放进想象里训练。",
    diagnostic:
      "检查 critic 的价值估计和真实回报的偏差；critic 估歪了，actor 一定跟着学歪。",
    lessons: ["06", "08"],
  },
  {
    term: "MDN",
    alias: "混合密度网络",
    definition:
      "输出一组高斯分布（权重、均值、方差），而不是单个点，用来表达“下一步有好几种可能”。World Models 的 M 就是 MDN-RNN。",
    diagnostic:
      "检查各混合分量有没有分工：不同分量对应不同的未来才有意义，全挤在一起等于退化成单点预测。",
    lessons: ["03"],
  },
  {
    term: "VQ / token 化",
    definition:
      "把连续的图像块查码本换成离散编号（token）。世界一旦变成 token 序列，语言模型“预测下一个 token”的配方就能直接搬来建世界。",
    diagnostic:
      "盯码本使用率：大部分编号从来不被用到，就是码本坍缩，生成质量的上限直接被砍。",
    lessons: ["09", "11"],
  },
  {
    term: "扩散模型",
    alias: "diffusion",
    definition:
      "从噪声出发一步步去噪生成图像的模型。当世界引擎用时，按“当前帧 + 动作”为条件去噪出下一帧，保细节能力比离散 token 强。",
    diagnostic:
      "对比它和 token 路线在小细节上的保真度：弹道、准星这类几个像素的东西，恰恰决定模型能不能支撑决策。",
    lessons: ["10", "12"],
  },
  {
    term: "latent action",
    definition:
      "没有按键记录时，模型从相邻帧的变化里自己归纳出的“动作词表”。Genie 路线靠它把纯视频变成可交互的世界。",
    diagnostic:
      "固定其他输入、逐个切换 latent action，看画面变化是否对应稳定、可解释的操作（左移、跳）。",
    lessons: ["11"],
  },
  {
    term: "JEPA",
    alias: "联合嵌入预测架构",
    definition:
      "不重建像素，只在表征空间里预测被遮住部分的表征。省掉逐像素画图的开销，赌的是“预测对表征就够用了”。",
    diagnostic:
      "同时监控表征方差和探针精度：方差塌掉是坍缩，方差正常但探针上不去是白学。",
    lessons: ["13", "14", "15", "16"],
  },
  {
    term: "线性探针",
    alias: "linear probe",
    definition:
      "冻结表征，在上面只训一个线性分类或回归头，用它的精度衡量表征里到底存了多少有用信息。",
    diagnostic:
      "和随机初始化表征上的探针对比；比不过随机基线，说明表征没学到内容。",
    lessons: ["13", "15", "16"],
  },
  {
    term: "物体中心 / 槽",
    alias: "object-centric / slot",
    definition:
      "状态不是一条向量，是一组槽，每个槽绑定一个物体或身体部位。转移可以在槽内走，关系用图或注意力。",
    diagnostic:
      "可视化槽和物体的对应：挡住杯子，手机那个槽不该跟着崩。对不上就还是一条糊向量。",
    lessons: ["16", "23", "30"],
  },
  {
    term: "动作对换",
    alias: "action-swap",
    definition:
      "固定同一段历史，只换一个合法动作，比较模型给出的未来。预测必须分岔，模型才算把动作用起来了。",
    diagnostic:
      "分岔失败就是动作盲。桌宠上：看左和伸手如果给出同一条未来，就不能拿去规划。",
    lessons: ["03", "24", "30"],
  },
  {
    term: "物体恒常",
    definition:
      "离开视野或被挡住的物体，状态里还应该在。灯该不该灭，要看杯子是被挡住了还是真的被端走了。",
    diagnostic:
      "挡住 2 秒灯仍绿，端走灯红，转头再转回还是原来的杯子。滑窗模型常在挡住时就把杯子忘掉。",
    lessons: ["12", "21", "29"],
  },
  {
    term: "点图",
    alias: "pointmap",
    definition:
      "每个像素对应的三维点。VGGT / DUSt3R / CUT3R 用它当 3D 状态，而不是先做传统 SfM。",
    diagnostic:
      "换视角后同一物体的点是否还叠在一起。漂了就还不是持久空间。",
    lessons: ["20", "21"],
  },
  {
    term: "VLA",
    alias: "Vision-Language-Action",
    definition:
      "观察和语言指令进，动作出。OpenVLA 吐离散动作，π0 用 flow matching 吐连续动作块。它是策略，不是 P(s'|s,a)。",
    diagnostic:
      "把世界模型当安全过滤器接在 VLA 后面：VLA 要伸手，世界模型说会碰杯，就截断。",
    lessons: ["25", "26", "32"],
  },
  {
    term: "外生事件",
    definition:
      "规划器选不了、只能当条件吃进去的变化。桌宠里，人伸手、把杯子端走，都是外生的。",
    diagnostic:
      "不要把人的行为写成桌宠的动作空间。预测人，但不要假装能控制人。",
    lessons: ["28", "31"],
  },
  {
    term: "安全过滤",
    definition:
      "动作出口上的硬限制：世界模型若预测会碰杯、过桌沿或熵太高，就截断或停下。",
    diagnostic:
      "关掉过滤后再看同一条动作会不会扫落杯子。没有这层，不准上真机。",
    lessons: ["27", "32"],
  },
  {
    term: "前向模型",
    alias: "forward model",
    definition:
      "根据当前身体或世界状态、以及即将发出的动作，预测下一刻的感觉或状态。运动控制里的说法，和本课的动作条件世界模型是同一张式子。",
    diagnostic:
      "同一状态换动作，预测必须分岔。多步会漂，要把漂移曲线报出来。",
    lessons: ["03", "24", "33"],
  },
  {
    term: "具身程度",
    alias: "E0 到 E5",
    definition:
      "本课作业用的六档：无动作、离线动作条件、可重置回路、真机开环策略、模型在动作回路、不可暂停的世界。不是 Wilson 或 Ziemke 的编号。",
    diagnostic:
      "升级只看是否题。真机成功只说明可能到了 E3；到 E4 必须有世界模型查询日志。聊天不升档。",
    lessons: ["28", "32", "33"],
  },
  {
    term: "赋权",
    alias: "empowerment",
    definition:
      "从当前状态出发，动作序列还能让未来感觉走多少条不同的路。杯子落地后，可选项通常变少。",
    diagnostic:
      "安全层改写的应是会砍掉未来选项的动作，例如扫杯，而不是无害的转头。算出赋权不自动升到 E5。",
    lessons: ["32", "33"],
  },
  {
    term: "逆向模型",
    alias: "inverse model",
    definition:
      "给定想要的下一状态，反推该发什么动作。ACT、VLA 更接近这条，不是前向世界模型。",
    diagnostic:
      "抓放成功率高，只能说明逆向或直接策略可用。Q1 动作对换仍可能从未做过。",
    lessons: ["25", "26", "33"],
  },
  {
    term: "开源三档",
    definition:
      "可跑（代码和权重齐）、仅权重（有推理脚本无训练代码）、纯 demo（连权重都没有）。",
    diagnostic:
      "Genie 3、GAIA-2、Marble 属于纯 demo。open-oasis 属于仅权重。不要把官网演示写成你练过。",
    lessons: ["12", "22", "34"],
  },
  {
    term: "WorldScore",
    definition:
      "把世界生成拆成下一场景任务的统一评测，覆盖 3D、4D、文生视频和图生视频。",
    diagnostic:
      "第一名说明下一场景生成好看、可控。它不测动作对换，也不测桌宠规划。",
    lessons: ["34", "44", "45"],
  },
  {
    term: "Physics-IQ",
    definition:
      "用真实物理实验录像当前后文，让生成模型续写，再和真实续写比重合。分数测的是续写，不是规划。",
    diagnostic:
      "画面真和 Physics-IQ 可以不同序。低分只说明这把尺子上的物理续写差。",
    lessons: ["34", "44"],
  },
  {
    term: "Mixture-of-Transformers",
    alias: "MoT",
    definition:
      "给不同模态分专家、共享注意力的骨架。Cosmos 3 用它同时跑理解通路和生成通路。",
    diagnostic:
      "先看输入输出配置：同一套 MoT 可以是 VLM，也可以是世界模拟器。",
    lessons: ["35", "38", "42"],
  },
  {
    term: "self-forcing",
    definition:
      "训练自回归视频模型时，让它吃自己刚生成的历史（常用 KV cache），对准推理时的分布。",
    diagnostic:
      "对照 teacher forcing 的自由 rollout 误差。训练曲线变差、多步漂移变慢，才说明补丁吃上了。",
    lessons: ["03", "37", "45"],
  },
  {
    term: "flow matching",
    definition:
      "学一条从噪声到数据的连续路径，用来生成图像、视频、声音或动作块。π0 和 Cosmos 3 Generator 都用这类目标。",
    diagnostic:
      "它是生成目标，不是动力学保证。没有动作条件，flow 再稳也还是画师。",
    lessons: ["26", "35", "37"],
  },
  {
    term: "AVWM",
    alias: "视听世界模型",
    definition:
      "把同步的画面和声音当成部分可观察世界的观测，按动作预测下一刻的视听。",
    diagnostic:
      "同一画面换声轨，预测分不分岔。只给视频配环境声，不算这一类。",
    lessons: ["36"],
  },
];
