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
    .replace(/[^a-z0-9\u3400-\u9fff]+/g, "-")
    .replace(/^-+|-+$/g, "")}`;
}

export const glossary: GlossaryTerm[] = [
  {
    term: "Token",
    definition:
      "模型一次读取或预测的离散编号。文本、图像块和音频码本都可以被表示成 token，但它们的词表、时间含义和损失位置并不相同。",
    diagnostic:
      "先确认 token 属于哪种模态、编号范围是什么、一个 token 对应多少文字或多少毫秒。",
    lessons: ["01", "03", "10"],
  },
  {
    term: "Logit",
    definition:
      "模型在 softmax 之前为每个候选 token 给出的未归一化分数。logit 的大小关系决定概率排序。",
    diagnostic:
      "比较两条实现是否等价时，优先比较同一输入下的 logits，而不是只比较最终采样文本。",
    lessons: ["01", "02"],
  },
  {
    term: "Causal mask",
    definition:
      "限制当前位置只能读取过去和当前允许范围的注意力掩码。流式模型还要把明确配置的 lookahead 算进可见范围。",
    diagnostic:
      "改变尚未到达的未来输入；如果当前输出跟着变化，mask 或 cache 存在未来泄漏。",
    lessons: ["01", "05", "07"],
  },
  {
    term: "Loss mask",
    definition:
      "指定哪些序列位置参与训练损失的掩码。条件、用户输入、padding 和不需要预测的模态位置通常应被排除。",
    diagnostic:
      "逐位置导出 labels；被排除的位置应是 -100，目标位置应与 next-token shift 对齐。",
    lessons: ["01", "04", "15"],
  },
  {
    term: "Teacher forcing",
    definition:
      "训练自回归模型时，把真实历史 token 作为下一步输入。推理时模型读取的是自己刚生成的 token，因此两种运行状态并不完全相同。",
    diagnostic:
      "短序列正常、长序列重复时，同时检查 teacher-forced loss 和自由生成轨迹。",
    lessons: ["01", "04", "19"],
  },
  {
    term: "Checkpoint",
    definition:
      "某个训练时刻保存的模型权重和恢复状态。可恢复训练还需要 optimizer、学习率调度器、随机数状态和数据游标。",
    diagnostic:
      "从 checkpoint 接着运行固定步数，检查数据顺序、学习率和 loss 是否与不中断运行一致。",
    lessons: ["01", "13", "18"],
  },
  {
    term: "Seed",
    alias: "Random seed",
    definition:
      "随机数生成器的初始值。固定 seed 能减少数据打乱、初始化和采样带来的变化，但不能自动消除非确定性 kernel。",
    diagnostic:
      "记录 Python、NumPy、PyTorch 和采样器使用的 seed，并用重复运行测量仍然存在的漂移。",
    lessons: ["01", "13"],
  },
  {
    term: "Manifest",
    definition:
      "逐条记录数据来源、文件哈希、切分、许可和处理版本的清单。它用于证明训练和评测实际读取了哪些样本。",
    diagnostic:
      "随机抽一个输出，从 case 反查到 manifest 行、原始文件和预处理版本。",
    lessons: ["01", "15", "18"],
  },
  {
    term: "Golden case",
    definition:
      "固定保存、人工检查过的回归样例。它包含输入、允许结果、禁止行为和运行条件，不等同于训练样本。",
    diagnostic:
      "版本变更后逐条比较，不要只看平均分；golden case 也不能进入训练集。",
    lessons: ["01", "15", "19"],
  },
  {
    term: "Trace",
    definition:
      "一次运行中按时间保存的中间张量摘要、事件和指标。trace 用来定位错误发生在数据、模型、解码还是播放阶段。",
    diagnostic:
      "先比较打开和关闭 trace 时的 logits，确认记录行为没有改变随机数消耗或计算结果。",
    lessons: ["01", "05", "19"],
  },
  {
    term: "Codec",
    definition:
      "把波形编码为连续或离散表示，并把表示解码回波形的模块。Omni Talker 通常预测 codec token，而不是直接预测 PCM。",
    diagnostic:
      "分别测 codec 重建和 Talker 预测；重建好不代表生成时这些 token 容易预测。",
    lessons: ["03", "04"],
  },
  {
    term: "Codebook",
    definition:
      "向量量化中可选择的离散向量集合。一个 codebook index 只说明选中了哪个向量，必须结合帧率和码本层级解释时间和码率。",
    diagnostic:
      "检查每路 index 范围、特殊 token、帧数以及多路码本的对齐方式。",
    lessons: ["03", "04"],
  },
  {
    term: "Connector",
    definition:
      "把图像或音频 encoder 的输出变换为语言模型可接收表示的模块。它可能同时改变 hidden size 和 token 数。",
    diagnostic:
      "固定 encoder 与语言模型，分别报告输入/输出 shape、参数量和模态打乱后的性能差。",
    lessons: ["02", "08"],
  },
  {
    term: "VAD",
    alias: "Voice Activity Detection",
    definition:
      "判断一段音频中是否存在语音的模块。它能提供说话开始和结束的候选时间，不能判断用户是在打断、附和还是对旁人说话。",
    diagnostic:
      "把 VAD 事件与轮次策略事件分开记录，并单独统计误触发和漏检。",
    lessons: ["05", "06", "07"],
  },
  {
    term: "Endpoint",
    definition:
      "系统判定当前用户话语已经结束的时间点。它通常依赖静音长度、声学状态和语义完整性。",
    diagnostic:
      "延迟从真实语义结束时间开始算，不能只从 VAD 报出静音的时刻开始算。",
    lessons: ["05", "06"],
  },
  {
    term: "Barge-in",
    definition:
      "助手正在播放语音时，用户插话并使系统暂停、停止或重新规划回答的交互。",
    diagnostic:
      "分别记录用户开口、检测、停止生成和 DAC 停播时间；只有一个取消按钮不算模型具备 barge-in。",
    lessons: ["06", "07", "19"],
  },
  {
    term: "TTFT / TTFA / RTF",
    definition:
      "TTFT 是请求到首个文本 token 的时间；TTFA 是请求到首段可播放 PCM 的时间；RTF 是处理时长除以生成音频时长。",
    diagnostic:
      "报告每个指标的起止事件。首个 codec logit、首个完整 codec frame 和首段可播放 PCM 不是同一时刻。",
    lessons: ["01", "04", "19"],
  },
  {
    term: "KV cache",
    definition:
      "自回归 attention 保存的历史 key 和 value，后续 token 可直接复用。cache 的长度、位置编号和 session 归属必须同步。",
    diagnostic:
      "比较一次性 forward 与逐 token 生成的 logits，并在两个交错 session 中检查状态不会串线。",
    lessons: ["05", "07", "12"],
  },
  {
    term: "MoE",
    alias: "Mixture of Experts",
    definition:
      "每个 token 只激活部分专家子网络的稀疏结构。总参数量可以很大，但单个 token 的实际计算由激活专家数决定。",
    diagnostic:
      "同时报告 total parameters、active parameters、专家负载和跨设备通信。",
    lessons: ["11", "13", "18"],
  },
  {
    term: "Load-balancing loss",
    definition:
      "鼓励 MoE 路由器把 token 较均匀地分给专家的辅助损失。权重过大时可能牺牲任务目标。",
    diagnostic:
      "按模态和专家统计 token 数；平均负载正常时仍要检查少数模态是否集中到单个专家。",
    lessons: ["11", "13"],
  },
  {
    term: "SSM",
    alias: "State Space Model",
    definition:
      "用递归状态更新处理序列的模型。推理时状态大小可以不随历史长度线性增长，但有限状态不能无损保存所有历史细节。",
    diagnostic:
      "同时测试长序列吞吐和需要精确回忆远处内容的样例，不能只根据显存判断长程能力。",
    lessons: ["12", "18"],
  },
  {
    term: "SFT",
    alias: "Supervised Fine-Tuning",
    definition:
      "使用给定输入和目标输出做监督微调。多模态 SFT 还必须明确每种模态的 token、mask 和数据比例。",
    diagnostic:
      "按任务桶报告有效训练 token 和 held-out 指标，避免长音频或长视频在无意中主导更新。",
    lessons: ["15", "18"],
  },
  {
    term: "Reference model",
    definition:
      "偏好优化中保持冻结的参考策略，用于衡量当前策略相对原模型改变了多少。它通常与训练起点权重相同。",
    diagnostic:
      "确认 reference 没有梯度、没有被 LoRA 合并覆盖，并记录它与 policy 的确切 revision。",
    lessons: ["16", "17"],
  },
  {
    term: "Reward hacking",
    definition:
      "模型找到奖励规则的漏洞而得到高分，但没有完成真实任务。例如输出固定格式骗过不完整的解析器。",
    diagnostic:
      "优先检查高奖励错答，并用隐藏测试、规则扰动和人工复核验证奖励是否代表目标能力。",
    lessons: ["17"],
  },
  {
    term: "Bootstrap confidence interval",
    definition:
      "从已有评测单位中反复有放回抽样，估计指标差值的不确定范围。配对实验应让同一个样例在各组中一起被抽中。",
    diagnostic:
      "先写清抽样单位是样例、说话人还是来源组；不能把同源切片当成完全独立样本。",
    lessons: ["14", "16", "20"],
  },
  {
    term: "Non-inferiority margin",
    definition:
      "在实验前规定的最大可接受退化量。新方法只有在置信区间表明退化不超过该值时，才能称为非劣。",
    diagnostic:
      "margin 必须在看 test 结果前确定，并用任务单位表示，例如准确率百分点或 WER 绝对差。",
    lessons: ["14", "18", "20"],
  },
  {
    term: "VAE",
    alias: "Variational Autoencoder",
    definition:
      "把图像或视频压缩到连续 latent，并从 latent 重建像素的生成模型组件。latent 中能保留多少细节限制了后续生成上限。",
    diagnostic:
      "先单独测 VAE reconstruction，再把重建损失与生成模型错误分开。",
    lessons: ["20"],
  },
  {
    term: "VQ tokenizer",
    definition:
      "把图像、视频或音频映射为离散 code 的 tokenizer。生成模型预测 code，解码器再把 code 还原为信号。",
    diagnostic:
      "检查词表大小、下采样率、码率和重建质量，不要把 tokenizer 误差算成 Transformer 误差。",
    lessons: ["03", "20"],
  },
  {
    term: "DiT",
    alias: "Diffusion Transformer",
    definition:
      "在扩散或流模型中处理带噪 latent 的 Transformer。它读取 timestep 条件，预测噪声、速度或其他训练目标。",
    diagnostic:
      "确认 timestep 注入位置、预测目标和采样器一致；混用不同参数化会让训练和推理不匹配。",
    lessons: ["20"],
  },
  {
    term: "Classifier-free guidance",
    alias: "CFG",
    definition:
      "生成时组合有条件和无条件预测，增强输出对条件的响应。它要求训练阶段以固定概率丢弃条件。",
    diagnostic:
      "同时记录 condition dropout 和推理 guidance scale；scale 变大可能提高遵循度，也可能降低多样性或产生伪影。",
    lessons: ["20"],
  },
  {
    term: "Thinker",
    definition:
      "接收文本、图像、视频或音频表示并产生语义推理状态的主模型。它是否直接生成语音，取决于系统架构。",
    diagnostic: "先问它输出的是文字 token、hidden state，还是音频 token 的条件。",
    lessons: ["01", "19"],
  },
  {
    term: "Talker",
    definition:
      "把文本或 Thinker 状态转换成离散音频码本序列的生成模块，不等同于声码器。",
    diagnostic: "区分 Talker 的 token 预测延迟和 codec 的波形解码延迟。",
    lessons: ["01", "04", "19"],
  },
  {
    term: "RVQ",
    alias: "Residual Vector Quantization",
    definition:
      "逐层量化残差的离散表示。多个码本共同描述一个音频帧，后层码本通常补充更细节的信息。",
    diagnostic: "码本数增加会提升 bitrate；不自动保证感知质量或生成更容易。",
    lessons: ["03", "04"],
  },
  {
    term: "Diagonal delay",
    definition:
      "把同一音频帧的不同码本沿时间错开，使序列模型能使用低层码本预测高层码本。",
    diagnostic: "计算首个完整帧时要算串行深度，不能只看帧率。",
    lessons: ["04"],
  },
  {
    term: "Streaming cache",
    definition:
      "流式编码器保留的历史状态。正确 cache 只包含过去和允许的 lookahead，不能偷偷看到未来。",
    diagnostic: "用 future-leakage probe 检查输出是否随不可见未来变化。",
    lessons: ["05", "07"],
  },
  {
    term: "Lookahead",
    definition:
      "为了提升当前输出质量而额外等待的未来输入窗口。它直接增加算法延迟。",
    diagnostic: "报告 chunk 大小时必须同时报告 lookahead，二者不能混为一谈。",
    lessons: ["05"],
  },
  {
    term: "Turn policy",
    definition:
      "决定 HOLD、TAKE_TURN、BACKCHANNEL、BARGE_IN 等交互动作的策略层。",
    diagnostic: "VAD 只说明有没有声音，不能替代语义轮次决策。",
    lessons: ["06", "07"],
  },
  {
    term: "Full duplex",
    definition:
      "系统在生成语音期间仍持续接收并理解用户流，并能依据新信息继续、暂停或重规划。",
    diagnostic: "能被声音打断不等于真双工；检查生成期间 listener 是否真的更新。",
    lessons: ["07", "19"],
  },
  {
    term: "M-RoPE",
    alias: "Multimodal Rotary Position Embedding",
    definition:
      "把时间、行、列等多轴坐标注入旋转位置编码，让视觉 token 保留空间或时空结构。",
    diagnostic: "多图场景还需要 image id 或边界语义，二维坐标本身不够。",
    lessons: ["08", "09"],
  },
  {
    term: "Token reduction",
    definition:
      "通过池化、合并、剪枝或选择减少进入大模型的视觉 token。",
    diagnostic: "同时看准确率、prefill、峰值显存和真实 encoder 成本，不能只报保留率。",
    lessons: ["10"],
  },
  {
    term: "Top-k routing",
    definition:
      "MoE 路由器为每个 token 选择得分最高的 k 个专家。",
    diagnostic: "top-k 提高会增加 active parameters 和通信，不必然解决专家塌缩。",
    lessons: ["11"],
  },
  {
    term: "Capacity factor",
    definition:
      "专家可接收 token 上限相对于理想均匀负载的倍数。",
    diagnostic: "过小会产生 overflow，过大则浪费 padding 与显存。",
    lessons: ["11", "13"],
  },
  {
    term: "Mamba-2 state",
    definition:
      "选择性状态空间层对历史序列的压缩状态，大小通常不随上下文长度线性增长。",
    diagnostic: "常数状态不等于任意长距离事实都能无损保留。",
    lessons: ["12"],
  },
  {
    term: "FSDP",
    definition:
      "在数据并行 rank 间切分参数、梯度与优化器状态的训练方式。",
    diagnostic: "峰值显存还取决于 all-gather 窗口、activation 和未切分模块。",
    lessons: ["13", "18"],
  },
  {
    term: "Expert Parallel",
    alias: "EP",
    definition:
      "把不同 MoE 专家放在不同设备上，通过 all-to-all 路由 token。",
    diagnostic: "EP 规模必须与专家数、拓扑和每 rank batch 一起设计。",
    lessons: ["11", "13", "18"],
  },
  {
    term: "Context Parallel",
    alias: "CP",
    definition:
      "沿序列维度切分超长上下文的计算与激活。",
    diagnostic: "CP 解决系统承载，不会自动让模型学会利用远距离信息。",
    lessons: ["13", "14"],
  },
  {
    term: "RoPE scaling",
    definition:
      "调整旋转位置编码频率以外推到更长上下文的一类方法。",
    diagnostic: "配置能接收 128K 与模型真的会用 128K 是两件事。",
    lessons: ["14"],
  },
  {
    term: "Token-balanced mixture",
    definition:
      "按实际训练 token 而非样本条数控制多模态数据占比。",
    diagnostic: "一条长视频样本可能抵得上数百条短文本，按行采样会严重失真。",
    lessons: ["15"],
  },
  {
    term: "DPO margin",
    definition:
      "偏好样本中 policy 相对 reference 对 chosen 与 rejected 的对数概率差。",
    diagnostic: "多模态 DPO 必须检查模型是否忽略图像或音频条件。",
    lessons: ["16"],
  },
  {
    term: "RLVR",
    alias: "Reinforcement Learning with Verifiable Rewards",
    definition:
      "用可以程序化核验的答案、结构或定位结果作为强化学习奖励。",
    diagnostic: "verifier 可执行不等于不可被钻空子，要做 reward hacking 审计。",
    lessons: ["17"],
  },
  {
    term: "GRPO",
    definition:
      "在同一问题的一组采样之间计算相对优势，避免单独训练 value model 的策略优化方法。",
    diagnostic: "组内奖励零方差时没有可用学习信号。",
    lessons: ["17"],
  },
  {
    term: "LoRA rank",
    definition:
      "低秩适配器的内维度，影响可训练参数、容量与优化器显存。",
    diagnostic: "先确认 target modules 覆盖了需要更新的路径，再比较不同 rank 的效果与显存。",
    lessons: ["18"],
  },
  {
    term: "Stable prefix",
    definition:
      "增量识别或生成中，已经不再被后续输入改写的前缀。",
    diagnostic: "跨 tokenizer bridge 应按 UTF-8 byte/span 对齐，而不是假设 token 一一对应。",
    lessons: ["19"],
  },
  {
    term: "Flow matching",
    definition:
      "学习把简单噪声分布沿连续向量场运输到数据分布的生成目标。",
    diagnostic: "视觉 flow head 与文本自回归 head 的时间变量和 mask 不能混用。",
    lessons: ["20", "28"],
  },
  {
    term: "InfoNCE",
    definition:
      "对比学习损失：把配对样本的相似度从同 batch 其他样本里用 softmax 挑出来。温度越小，分布越尖。",
    diagnostic: "打乱配对后损失应上升；温度从很低调高时，正对概率峰值应下降。",
    lessons: ["21"],
  },
  {
    term: "VLA",
    alias: "Vision-Language-Action",
    definition:
      "同时吃视觉和语言、直接吐出机器人动作的模型。动作可以是离散 token、连续回归或流匹配轨迹。",
    diagnostic: "会写“抓住杯子”只证明文字通路；要看动作是否进入同一套条件生成，以及评测协议绑的是哪套套件。",
    lessons: ["24", "27", "28", "31"],
  },
  {
    term: "Action chunk",
    definition:
      "一次预测未来 H 步动作。开环窗口是 H/f；推理延迟必须短于这个窗口，否则执行的是过期计划。",
    diagnostic: "目标被挪走后若仍沿旧轨迹，检查是否执行了整段 chunk 而没有在前 k 步重规划。",
    lessons: ["25", "30"],
  },
  {
    term: "Set-of-Mark",
    alias: "SoM",
    definition:
      "给图上可操作物体编号，让模型输出编号而不是像素坐标。Magma 用它做动作接地。",
    diagnostic: "低分辨率下连续坐标误差通常大于编号分类；答案对仍要核对编号是否指到被问物体。",
    lessons: ["23", "32"],
  },
];
