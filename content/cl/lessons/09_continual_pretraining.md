---
id: 09_continual_pretraining
title: "换领域时旧能力怎么掉"
summary: "接到窄领域语料上续训，掉的是知识还是格式？"
unit: llm
play_tools: []
checkpoints:
  - "通用/领域双曲线。"
  - "说清这和第 06 课回放是同一件事的语言模型版。"
---

# 第 09 课：换领域续训时，通用能力怎么掉

> 类型：实战（短续预训练三条配方对照；CPU 机制实验必做，GPU 短续训按档）<br>
> 建议周期：2-4 天<br>
> 硬件：Mac / CPU 完成本课浏览器实验和 `python3 run.py run 09`；单张 8-24GB 卡跑 SmolLM2-135M 短续训（主线）；TinyLlama-1.1B 标加分；7B 不在本课范围内<br>
> 锚定仓库与权重：[HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)（基座，不要用 Instruct）、[bigcode/the-stack-smol](https://huggingface.co/datasets/bigcode/the-stack-smol) 的 Python 子集；配方对照 [Ibrahim et al., 2024](https://arxiv.org/abs/2403.08763)（学习率回热、再退火、回放）与 [Wu et al., 2024](https://arxiv.org/abs/2402.01364) 的 continual pretraining 一节<br>
> 产物：三条配方的通用 / 领域双曲线、一张「掉的是知识还是格式」探针表、以及「混 30% 通用数据」对应 [第 06 课](06_replay_der.md) 哪件事的书面说明

## 1. 这一课做什么

前两幕把病看清楚了，也把四类补丁在图像和 MLP 上做过一遍。[第 08 课](08_gem_gdumb.md) 停在 GEM / A-GEM 的梯度投影，以及那个让人难堪的基线 GDumb：很多「平均准确率第一」其实只是缓冲里的 i.i.d. 重训已经很强。从这一课起进入第三幕，对象换成语言模型。主循环还是同一套：

```text
新经验进来
  先决定写到哪里（上下文 / 外挂记忆 / 快速权重 / 慢速权重）
  再决定怎么写（覆盖、追加、压缩、约束、正交、合并）
  写完立刻测两件事：新任务会了没、旧任务还在不在
```

本课写的位置是慢速权重（基座模型的全部或绝大部分参数），写法是接着做因果语言模型的下一词预测。Wu et al. 2024 把大模型持续学习按训练阶段拆成三截：续预训练（continual pretraining, CPT）、持续指令微调（continual instruction tuning, CIT）、持续对齐（continual alignment, CA）。本课只做第一截。下一课才把指令任务一个接一个接上。

具体场景：手里有一个已经在通用网页、书籍、代码上预训练过的小模型，你想让它更会写 Python。最省事的做法是把 Python 语料灌进去，学习率沿用预训练峰值，loss 看着降。问题是通用英文、常识填空、普通句子的流畅度往往会一起掉。课内把这件事拆成三条可对照的配方：

1. 只用领域数据，学习率用预训练峰值（硬上）。
2. 只用领域数据，学习率降 10 倍。
3. 每个 batch 混 30% 通用数据，学习率与配方 1 相同，只改配比。

每条配方都画两条曲线：领域验证损失（或领域填空）和通用验证损失（或 LAMBADA / WikiText 一类通用指标）。交点在哪、哪条先塌，是本课要你自己跑出来的，不是从内部报告抄的。

这一课在主循环里加的零件是「领域续训时的学习率与回放」。没有它，你只会在通用模型上做一次 SFT 或一次领域微调，却说不清掉的是事实、格式，还是两者都掉。做完你能验证三件事：硬上会把通用曲线抬高（更差）；降学习率两边都变慢；混通用数据相当于 [第 06 课](06_replay_der.md) 的经验回放，只是旧样本换成了通用语料。

第三幕四课的分工也先钉死，避免后面把方法叠在同一张表上却说不清测的是哪一截。第 09 课：基座、下一词、领域对通用。第 10 课：Instruct 或带指令模板的 LoRA，任务对任务，矩阵是 4×4。第 11 课：每个任务一块低秩更新，要它们近似正交。第 12 课：训练已经结束，把几份权重加起来，那是事后缝合。梁文峰转写里把持续学习放在 Agent 之后的下一级台阶（转写未获 DeepSeek 官方确认）。本课这种「换语料接着预训练」只是台阶上的螺丝钉：它证明慢速权重能写进新领域，也证明旧分布会被覆盖。它还不是「两个月上岗」那种在岗学习。上岗要等到第六幕。把螺丝钉拧紧仍然必要，否则后面的指令接龙会把领域遗忘和任务遗忘混成一笔糊涂账。

本课分档必须写清：GPU 短续训是实战，不列入课程正式复现清单（CURRICULUM.md §4 的五项复现不含第 09 课）。CPU 机制实验钉死的是「混入通用数据后通用指标掉得更少」，不下载权重。浏览器实验叫「数据配比」，全程在页面里算完。

术语速查：

| 术语 | 一句解释 |
|---|---|
| 续预训练 CPT | 在已经预训练好的模型上，继续用下一词预测吃新语料，通常为了换领域、补事实或扩语言 |
| 领域偏移 | 新语料的 token 分布和原预训练差一截，比如从网页混料换成几乎全是 Python |
| 学习率回热 | 预训练结束时学习率已经很小；接着训时先把学习率抬回去再衰减，否则新数据学不动 |
| 硬上 | 课内说法：直接用预训练峰值学习率、不加回放地灌领域数据 |
| 通用回放 | 每个新 batch 里留固定比例的通用文本，对应第 06 课缓冲里的旧样本 |
| 双曲线 | 同一训练过程上同时画通用指标和领域指标，禁止只报领域 loss |
| LAMBADA | 根据长上下文猜最后一个词的阅读基准，课内当通用能力的一种读数 |
| 基座 / Instruct | 基座只做过预训练；Instruct 还经过指令微调。本课续训必须用基座 |
| bits-per-byte | 按字节计的压缩损失，跨词表比较时比 raw perplexity 更公平 |

## 2. 问题

把一个通用小模型接到窄领域语料上，最常见的失败不是「领域没学会」，而是「领域学会了，人话不会说了」。loss 曲线只看领域验证集会骗你：领域损失下降，只说明模型在拟合当前 token 分布。通用网页、对话、填空用的是另一套分布。两者共用同一份权重，梯度每一步都在改这份权重。

本课要回答四个具体问题：

1. 原学习率硬上时，通用曲线掉多少、掉得多快？Ibrahim et al. 2024 在 405M 上把 Pile 接到德语 Common Crawl（更强偏移）时，不用回放的 Pile 验证损失会从约 2.4 抬到约 3.56；接到同为英文的 SlimPajama（弱偏移）时，同样不回放也会从约 2.4 抬到约 2.44 量级，领域侧则明显下降。课内 135M、百万 token 级预算不会复现他们的绝对数字，但方向应当同类：硬上对通用不友好。
2. 学习率降 10 倍能不能换稳定性？调小学习率是 [第 02 课](02_stability_plasticity.md) 已经试过的笨办法：两边都变慢。续预训练里它仍然有用，因为它限制每步改动；它解决不了「领域数据里根本没有通用句子」这件事。
3. 每 batch 混 30% 通用数据，等价于第 06 课的哪一步？缓冲里不再是 MNIST 数字，而是 WikiText 或原预训练同类文本。Ibrahim 在弱偏移用 5% 回放、强偏移用 25% 回放，就能把平均损失拉到接近「把新旧数据并在一起从头训」的上限。课内 30% 是为了在很小的 token 预算上把效应看清楚，不是他们的生产配比。
4. 掉的是知识还是格式？领域代码续训之后，模型可能仍会补全 `def` 和缩进（格式还在），却把「The capital of France is」补成乱码或代码片段（通用格式被带偏）；也可能英语还通顺，却把原来会的事实填空答错（知识被冲）。课内用两套探针分开记，禁止把「生成看起来像代码」当成通用能力还在。

一个必须划清的界限：本课短续训的 token 数是百万级，SmolLM2-135M 原预训练是 2 万亿 token，学习率日程是 Warmup-Stable-Decay，峰值 $3.0\times 10^{-3}$。你在 24GB 卡上跑几十分钟，不能声称复现了 Ibrahim 的 300B token 实验，也不能拿课内 WikiText 困惑度去对比模型卡上的 HellaSwag 42.1。模型卡数字只用来确认你加载的是对的权重；配方之间的相对方向，才是本课验收。

## 3. 准备

- 会用 Python 3.10+、虚拟环境、Hugging Face `transformers` / `datasets`。不需要先会 DeepSpeed。用到的下一词损失和 LoRA 公式本课当场写。
- 档 A（Mac / CPU）：浏览器实验 + `experiments/` 下的机制实验即可过关。不要在 CPU 上硬训 135M 全参数，那会空转一晚上。
- 档 C（单张 24GB，主线假设）：能把 SmolLM2-135M 以 `bfloat16` 或 `float16` 加载，序列长度 512 或 1024，微批次 4 到 8。磁盘预留 10GB：权重约 0.3GB，the-stack-smol 的 Python 子集下载后约几百 MB，WikiText-2 很小。
- 档加分：同一套脚本把模型换成 `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`（1.1B，预训练约 3T token，峰值学习率论文写明为 $4\times 10^{-4}$，上下文 2048）。显存不够就开梯度检查点，仍保持三条配方对照。
- 账号与许可：Hugging Face 可匿名下载 SmolLM2-135M（Apache-2.0）和 WikiText。the-stack-smol 来自 BigCode，使用前阅读数据集卡上的许可与 opt-out 说明；课内只用 `data/python` 这一个语言目录，不要一次拉全部 2.6GB 的 30 种语言。
- 不要用 Instruct 权重做续预训练。Instruct 已经改过指令分布，通用曲线的起点和「预训练刚结束」不是一回事。指令顺序是第 10 课的事。
- 建议固定种子 42，并把每条配方的命令、学习率、混合比、步数写进同一份笔记。后面三条曲线要能对上这三条命令。

## 4. 学习目标

1. 用 Wu et al. 2024 的三阶段划分，指出本课落在 CPT，并说出 CPT 和一次领域 SFT 在损失、数据格式上的差别。
2. 写出因果语言模型续训的目标函数，解释为什么领域数据上的梯度会改写通用数据上的预测。
3. 对照 Ibrahim et al. 2024：能口述学习率回热、再退火、回放各自防的是适应不足还是遗忘，以及课内「硬上 / 降 10 倍 / 混 30%」和论文配方差在哪。
4. 独立跑完三条配方（GPU）或 CPU 机制实验，交出通用、领域两条曲线，并标出交叉点对应的领域数据比例。
5. 用至少一种知识探针和一种格式探针，判断掉的主要是事实还是表面格式。
6. 写清「混 30% 通用数据」对应第 06 课回放的哪一项（缓冲内容、混合比、compute-equivalent），以及 GDumb 式「存下来从头训」在续预训练里为什么通常不可行。

## 5. 原理

六个机制按同一节奏：为什么需要、怎么运转、定义、代码落点、怎么验证。

### 5.1 大模型持续学习的三截，本课只做第一截

一次训完就冻住的基座，知识停在预训练语料的截止日期。换领域、补新事实、加一门语言，都要改权重。Wu et al. 把改法按原训练流水线对齐成三截：CPT 继续做自监督下一词预测，扩大语言和领域理解；CIT 在指令-输出对上做监督，教模型听人话、迁到新任务；CA 用人类反馈或偏好数据对齐价值和口味。图 2 还标了跨阶段遗忘：一个已经指令微调过的模型如果再去做 CPT，指令跟随会掉。

课内含义很具体。本课加载的是 SmolLM2-135M 基座，损失是下一词交叉熵，数据没有「User / Assistant」模板。你若误用 Instruct 权重，观测到的「通用掉了」里会混进指令格式被冲掉，和 CPT 要测的不是一件事。CIT 留给第 10 课；CA 更后面。

数学上 CPT 仍是标准语言建模。记当前领域语料为 $\mathcal{D}_{\text{dom}}$，模型参数为 $\theta$，序列 $x_{1:T}$：

$$
\mathcal{L}_{\text{CPT}}(\theta)=-\mathbb{E}_{x\sim\mathcal{D}_{\text{dom}}}\frac{1}{T}\sum_{t=1}^{T}\log p_{\theta}(x_t\mid x_{<t})
$$

验证时把同一公式分别打在 $\mathcal{D}_{\text{dom}}^{\text{val}}$ 和 $\mathcal{D}_{\text{gen}}^{\text{val}}$ 上，得到领域损失和通用损失。只优化第一项、却用第二项当验收，就是本课的整张图。

落点：Hugging Face 的 `AutoModelForCausalLM` 前向默认返回这个交叉熵（`labels` 与 `input_ids` 对齐、padding 位置为 `-100`）。Ibrahim 的大规模实现挂在 EleutherAI `gpt-neox` 的 PR 1194 / 1200，课内不跑那个仓库，只借它的三条策略。

验证：加载基座后，在 WikiText-2 和 Python 验证集上各算一次损失，应看到 Python 损失未必更低（基座见过 The Stack，但分布仍偏网页）。记下这两个数当 $t=0$ 的双曲线起点。

### 5.2 领域偏移：同一套权重，两套下一词

直觉：模型像一个只带一支笔的书记员。你逼他连续抄一万行 Python，他会把笔迹练成缩进和括号，再让他写普通英文段落，句子会被括号和 `self.` 污染。类比失效处：人可以换本子，模型没有第二套权重（扩结构是第 07 课，本课不用）。

机制：预训练数据 $\mathcal{D}_0$ 是 FineWeb-Edu、DCLM、The Stack 等的混合物（SmolLM2-135M 模型卡写明 2T token）。领域数据 $\mathcal{D}_1$ 课内取 the-stack-smol 的 Python，10,000 个文件。Python 的 token 高频项是缩进、`def`、`self`、点号；WikiText 的高频项是英文函数词。SGD / AdamW 的每一步

$$
\theta\leftarrow\theta-\eta\nabla_{\theta}\mathcal{L}(\theta;\mathcal{B})
$$

里，batch $\mathcal{B}$ 若全来自 $\mathcal{D}_1$，梯度就几乎只降低 $p_{\theta}(\text{Python token}\mid\text{Python 上文})$。对通用上文，同一批参数被推离原来的条件分布，表现为通用损失上升。这就是第 01 课灾难性遗忘在语言模型上的样子，任务边界不再是「数字 0-1 vs 2-3」，而是「网页英语 vs Python」。

Gururangan et al. 2020（Don't Stop Pretraining）已经说明：在领域语料上继续预训练能提高下游领域任务。他们的设定更接近「再适应一次」，没有强制你同时保住通用基准。Ibrahim 的设定是 token 数达到 100B 以上的真续预训练，必须和「新旧并在一起从头训」比。课内预算远小于 100B，所以只借机制，不借他们的「匹配从头训」结论。

验证：同一 checkpoint，对两条验证集做 teacher-forcing 损失。若领域损失降、通用损失升，就是偏移在发生。若两者一起降，要么通用验证集其实含代码，要么步数太少还在噪声里。

### 5.3 学习率：硬上、降 10 倍、以及论文里的回热

开放权重模型在预训练结束时，学习率通常已经被余弦或 WSD 日程降到峰值的十分之一甚至接近 0。Ibrahim 的核心观察：如果接着用这个已经很小的 $\eta_{\min}$ 去吃几百 B 新 token，适应会不够；必须把学习率先抬回去（回热），再按新的 token 预算衰减（再退火）。他们还发现，回热初期通用损失会先跳一下，拉得太高则遗忘加剧。对 405M、Pile 接到 SlimPajama 或德语时，回热到原峰值附近再衰减，比钉死在 $\eta_{\min}$ 更能把新数据学会。

课内三条学习率处理，和论文不是一一对应，对应关系如下。

| 课内配方 | 学习率 | 和 Ibrahim 的关系 |
|---|---|---|
| 1 硬上 | 常数 $3.0\times 10^{-3}$（SmolLM2-135M 论文写明的峰值） | 相当于回热到峰值后不再衰减。短预算上这是最猛、也最伤通用的 |
| 2 降 10 倍 | 常数 $3.0\times 10^{-4}$ | 更接近「舍不得回热」。适应慢，遗忘也慢 |
| 3 混 30% | 与配方 1 同 LR，只改数据 | 用来单独看回放，不把 LR 和配比搅在一起 |

SmolLM2-135M 的 $3.0\times 10^{-3}$ 对已经训完的模型是很大的一步。短续训里用常数 LR 而不是完整余弦，是因为总步数只有几百，余弦还没弯就结束了。你若把预算加到数万步，应当改成：从当前 $\eta_{\min}$ 线性回热到选定峰值，再余弦到 $0.1\eta_{\max}$，这才是 Ibrahim 的日程。课内主线不强制实现完整日程，但验收时要能说出差别。

TinyLlama-1.1B 加分项的预训练峰值是 $4\times 10^{-4}$（仓库 README 写明，余弦、2000 warmup）。配方 1 用 $4\times 10^{-4}$，配方 2 用 $4\times 10^{-5}$，不要误把 135M 的 $3\times 10^{-3}$ 套过去。

为什么硬上特别伤通用，可以用一步更新的尺度来看。AdamW 在预训练末期，二阶矩估计已经适应了「小梯度、小学习率」的状态；你突然把 $\eta$ 拉回 $3\times 10^{-3}$，并且把数据换成几乎全是 Python，有效步长会比预训练稳定段大一截。Ibrahim 报告过回热初期上游和下游损失都会先抬一下，然后新数据侧才降下去。课内 400 step 很可能停在「先抬起来」的那一段，所以配方 1 的通用曲线难看，并不等于「再训 300B token 之后仍然最差」。短训读的是初期动力学，长训读的是日程走完之后的终点。两者都要测，本课预算只覆盖前者。

WSD（Warmup-Stable-Decay）和余弦的差别也写在这里。SmolLM2-135M 用 WSD：warmup 之后有一段恒定峰值，最后 20% 步数衰减到 0。余弦则从峰值光滑降到 $\eta_{\min}$，总长必须事先知道。续训时若你不知道还要吃多少新 token，余弦会逼你猜一个假预算。这就是 Ibrahim 第 7 节讨论「无限学习率日程」的动机。课内常数 LR 是 WSD 稳定段的粗糙模仿：配方 1 模仿「又回到稳定段峰值」，配方 2 模仿「停留在已经衰减过的小 LR」。加分改造才把衰减接回去。

验证：配方 1 的领域损失应下降最快；配方 2 的通用损失升幅应小于配方 1。若配方 2 领域损失几乎不动，说明 10 倍可能降过头，或步数不够，记下来，不要改成「方法无效」。

### 5.4 混 30% 通用数据：第 06 课回放的语言模型版

第 06 课的经验回放：新任务的 batch 里掺进缓冲里的旧样本，DER++ 还对齐旧 logits。语言模型续预训练的旧任务是「通用文本上的下一词」，旧样本就是通用语料。没有人能把 2T 预训练数据放进环形缓冲，做法是从原分布的一个可再分发代理里抽样：课内用 WikiText-2 raw；Ibrahim 用的是真正的 Pile 回放，弱偏移 5%、德语强偏移 25%，并且是 compute-equivalent（新数据 token 减少，总计算量不变）。

课内混合损失：

$$
\mathcal{L}_{\text{mix}}=(1-\rho)\mathcal{L}_{\text{dom}}+\rho\mathcal{L}_{\text{gen}}
$$

$\rho=0.30$ 时，每个 batch 约 30% token 来自 WikiText-2，70% 来自 Python。实现上按 token 或按样本抽都可以，但必须固定一种并写进笔记。按样本抽时，Python 文件往往比 Wiki 段落长，真实 token 比例会偏领域，曲线会看起来「回放不够」。

和第 06 课的对应与失效：

- 对应：都是在新梯度里保留一条指向旧分布的分量，降低 $\theta$ 逃离旧损失谷底的速度。
- 失效 1：第 06 课缓冲远小于旧数据集；本课 WikiText-2 也远小于 FineWeb。代理分布不等于原预训练，回放只能缓解，不能还原。
- 失效 2：GDumb 把缓冲攒满再从头训，在 2T token 的基座上不可行。续预训练没有「用 500 条样本重训 135M」这种便宜上限。第 08 课的打脸在这里打不响，这正是要换对象的原因。
- 失效 3：Ibrahim 的 5% 在 300B token 上够用，课内只有百万 token，5% 可能看不出差别，所以把 $\rho$ 提到 0.30。不要把 0.30 写进论文复述里当成他们的超参。

验证：配方 1 和配方 3 学习率相同。若配方 3 的通用损失升幅明显小于配方 1，而领域损失仍下降，回放就成立。若领域损失也不降，说明 30% 把领域信号稀释过头，或通用代理太容易、模型在「偷懒」拟合 WikiText。

### 5.5 双曲线：禁止只报领域 loss

只报领域困惑度，会把配方 1 评成赢家。持续学习的验收从 [第 03 课](03_cl_evaluation.md) 起就必须同时看旧任务。续预训练没有任务编号，旧任务用通用验证集代替。

横轴用训练 token 数（不要用 epoch，Python 子集很小，重复一遍不算「更多新数据」）。纵轴两条：通用验证损失 $L_{\text{gen}}(t)$、领域验证损失 $L_{\text{dom}}(t)$。浏览器实验把横轴改成领域数据比例 $\alpha$，用一个固定步数的代理动力学看交叉点。GPU 实验则是时间轴上的真实损失。

交叉点的读法：若某配方在 $t^*$ 之后 $L_{\text{gen}}$ 已经高于起点很多，而 $L_{\text{dom}}$ 还在降，你得到的是一个领域专家、一个残废的通用模型。产品上这不一定错（代码补全插件可以牺牲写诗），但必须是你选的，不能是没测到。

SmolLM2-135M 模型卡上的通用读数（加载后可对照，作 $t=0$ 的 sanity check，不是本课训练目标）：HellaSwag 42.1、ARC 平均 43.9、PIQA 68.4、MMLU cloze 31.5、Winogrande 51.3、GSM8K 5-shot 1.4。短续训不要每 50 step 跑一遍这些，太贵；用 WikiText-2 损失当高频曲线，训练结束再抽一小份 LAMBADA 或课内 20 题通用填空。

### 5.6 知识还是格式：两套探针

「通用掉了」需要拆开。格式指表面续写习惯：英文大小写、标点、会不会突然输出 `import os`；代码格式指缩进、冒号、括号配对。知识指条件事实：LAMBADA 那种长上下文最后一个词，或「Paris is the capital of」。

课内最小探针（固定 20 条，写在 `probes.json`，三条配方共用）：

- 格式-通用：10 个普通英文前缀，看续写是否仍是英文词，而不是代码。
- 知识-通用：10 个单句填空，答案是一个专有名词或数字。
- 格式-领域：10 个 Python 前缀（`def add(a, b):`），看能否补出缩进和 `return`。
- 知识-领域：10 个 API / 关键字填空（`__init__`、`self`、`None`）。

评分用精确匹配或人工三档（通顺 / 跑题 / 串域）。串域指用 Python 续写英文探针，或用自然语言续写代码探针，这是格式被领域污染的直接证据。

探针和损失不要互相替代。WikiText 损失升，只说明平均下一词更差，可能是标点、大小写、罕见专名一起变差。LAMBADA 更接近「长上下文里的最后一个词还能不能猜中」，对 135M 基座本来就很难，短续训后再测，绝对分数会很低，有用的是三条配方的相对差。课内 20 题的优点是你能逐题看串域；缺点是方差大，20 题不够当论文表。正确用法：损失画曲线，探针讲故事。两者矛盾时（损失升但探针仍通顺，或反过来），把矛盾写进笔记，这往往说明掉的是尾部 token 分布而不是开头几个常见词。

和 [第 04 课](04_not_just_rag.md) 的关系：那一课比较的是「每次把名录塞进上下文 / 检索 / 改权重」。本课的 CPT 是改权重的一种，而且改的是下一词分布，不是一条条事实的定位编辑（第 14 课）。你在 Python 上续训，不会自动记住「小王坐哪」；你在 WikiText 上回放，也不会自动保住代码缩进。写入位置对了，写入内容仍然要和你想保住的能力同类。这是主循环里「写到哪里」之后必须立刻问的「写的是什么」。

Lin et al. 2023（Speciality vs Generality, arXiv:2309.06256）讨论过基础模型微调时的特殊性与一般性权衡；TRACE 后来把通用能力变化写成 General Ability Delta。本课 135M 短训不必算 GAD，但探针表要留着，第 10 课的指令跟随掉点会再遇到同类问题。

## 6. 源码导读

本课没有「官方 CPT 仓库必须 clone 才能训 135M」这一说。锚定的是公开权重、公开领域数据和 Ibrahim 写明的三条策略。胶水训练脚本用 Hugging Face `Trainer`，你要能指出下列对象在自己脚本里的位置，而不是背一个不存在的官方路径。

| 对象 | 你要确认的事实 |
|---|---|
| `HuggingFaceTB/SmolLM2-135M` | 基座；架构为 Transformer decoder，GQA，hidden 576，约 30 层，上下文 8192，词表 49152；预训练 2T token，精度 bfloat16 |
| `AutoModelForCausalLM.from_pretrained` | 加载后 `model.config.max_position_embeddings` 应为 8192；短训可以把序列截到 512 以省显存 |
| `bigcode/the-stack-smol` 的 `data/python` | 10,000 行，字段含 `content`、`lang`、`path`、`licenses` |
| `wikitext` 的 `wikitext-2-raw-v1` | 通用代理。字段 `text`；空行要滤掉 |
| `DataCollatorForLanguageModeling(mlm=False)` | 因果 LM：`labels` 等于 `input_ids`，pad 位 `-100` |
| `Trainer` 的 `learning_rate` / `lr_scheduler_type` | 配方 1、3 用 `constant` 或 `constant_with_warmup`；加分档才上 cosine |
| 自定义采样器或 `interleave_datasets` | $\rho=0.30$ 必须在这里落地，而不是「先训 Python 再训 Wiki」 |
| Ibrahim 的 gpt-neox PR | 只读：回热、再退火、compute-equivalent replay 的大规模实现。课内不跑 |

SmolLM2-135M 的官方训练栈是 Hugging Face nanotron，那是从随机初始化训 2T token 的，和续训不是同一条命令。对齐手册里的 `recipes/smollm2` 是 Instruct 的 SFT / DPO，本课不要跑。你若打开模型卡，应当看到基座指标表（HellaSwag 42.1 那张），以及「Instruct 版本用 smol-smoltalk 做 SFT」的链接：那是后训练，会污染 CPT 实验。

the-stack-smol 的加载方式以数据集卡为准：

```python
from datasets import load_dataset
py = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train")
```

不要写 `load_dataset("bigcode/the-stack-smol", "python")`，当前卡上的语言切分是目录 `data_dir`，不是官方 config 名。

Ibrahim et al. 2024 的实验设定（供对照，不是课内超参）：405M 与约 10B 的 decoder-only；数据 Pile 约 300B 再接到 SlimPajama 约 300B（弱偏移）或德语 Common Crawl 约 200B（强偏移）；AdamW；序列 2048；回放按 batch 比例掺入 $\mathcal{D}_0$，同时减少 $\mathcal{D}_1$ 的独特 token 以保持总计算量不变。论文图 1 的结论是：回热 + 再退火 + 适量回放，平均验证损失和一组 LM 基准可以接近「并集从头训」，计算量少一截。课内没有并集从头训这条上限（2T+领域从零训 135M 不在预算内），所以上限改成「基座在通用验证集上的 $t=0$ 损失」。

Ke et al. 2023（ICLR，arXiv:2302.03241）研究的是无旧领域数据时的领域自适应持续预训练，用软掩码保护通用知识。本课配方 3 有通用代理数据，比他们的无旧数据设定更接近第 06 课。读他们是为了知道：没有 WikiText 可混时，人们会去正则化权重而不是回放。

## 7. 实验

三层都做。浏览器和 CPU 在任何机器上必须完成；GPU 短续训是档 C 主线，缺卡则把双曲线改用 CPU 实验输出，并在笔记里标明「未跑 135M」。

### Step 0: 浏览器实验「数据配比」

打开本课页面，找到交互实验（页面里的 lab，id 为 `lab-09-data-mix`）。横轴是领域数据比例 $\alpha$（0 表示纯通用，1 表示纯领域），纵轴是两条能力曲线的代理分数。系统用一个固定步数的缩小动力学更新：每一步的梯度是 $\alpha$ 份领域方向加 $1-\alpha$ 份通用方向，学习率档位可切换「硬上 / 降 10 倍」。

先预测再运行。预测题：在硬上档，$\alpha$ 至少到多少时，通用曲线会掉到与领域曲线交叉？选一个区间（例如 0.4-0.6 / 0.7-0.9 / 几乎不交叉）。运行后看交叉点。把 $\alpha$ 改到 0.70（对应课内 30% 通用回放）再跑一次，看通用曲线是否明显高于纯领域。改滑块会作废上次运行，必须重新预测。

这一步对应 5.4 和 5.5：配比是回放比 $\rho=1-\alpha$，交叉点是双曲线的几何版本。它不证明 135M 的真实 PPL，只强迫你在烧 GPU 之前先有一个假说。

### Step 1: CPU 机制实验

在仓库的 `experiments/` 目录：

```bash
python3 run.py run 09
```

现在应当全绿：命令结束打印 `[PASS]`，写出 `artifacts/lesson09/result.json`，五条 `checks` 全为真。

| check | 本机为真时在说什么 |
|---|---|
| `pretrained_general_above_0_90` | 预训练后通用准确率高于 0.90 |
| `domain_only_general_drops` | 纯领域续训后，通用指标下降超过 0.15 |
| `mix_drops_less_than_domain_only` | 混 30% 通用数据时，通用掉幅比纯领域再少 0.08 以上 |
| `mix_still_learns_domain` | 混数据后领域准确率仍高于 0.85 |
| `domain_only_learns_domain` | 纯领域续训确实学会了领域 |

本机一次运行（Python 3.13.13，seed 9）：纯领域通用从 0.989 掉 0.286；混 30% 只掉 0.054，领域仍 0.954。换机器数字会变，方向不应变。读 `summary` 里写明的阈值。

这一层是两套冲突方向上的线性分类器，不下载 Hugging Face 权重，不证明 135M 续预训练，也不是 Ibrahim 的论文分数。后面 GPU 曲线的方向应与这些 check 同类，数量级不必相同。不要手写一份假 JSON。

### Step 2: 安装与加载基座（档 C）

```bash
pip3 install "transformers>=4.45" "datasets>=2.20" accelerate evaluate
```

用下面这段确认权重是基座而不是 Instruct（Python 脚本，不要把它拆成 shell）：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
name = "HuggingFaceTB/SmolLM2-135M"
tok = AutoTokenizer.from_pretrained(name)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(name)
print(model.num_parameters(), model.config.max_position_embeddings)
```

预期：参数量在 1.35e8 量级，上下文 8192。若路径被你改成 `SmolLM2-135M-Instruct`，停下来换基座。

### Step 3: 准备领域与通用数据

```python
from datasets import load_dataset, Dataset
py = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train")
wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
```

从 Python 集划 512 个文件做训练、64 个做领域验证（按文件 hash 切，不要随机打乱后切同一文件）。WikiText-2 的 `train` 做混合回放池，`validation` 做通用验证。把每条 `content` / `text` 按 tokenizer 截到 512 token，丢掉空文本。统计两边的 token 总数，写进笔记：领域训练集应当在 0.5M-3M token 量级，足够看到曲线分叉，又不必过夜。

LAMBADA 可选。若已安装 `lm-eval`，训练结束用 `lambada_openai` 的一小份；没装就用课内 20 题探针，不要为了 LAMBADA 卡住主线。

### Step 4: 三条配方共用的训练入口

把训练写成一份 `train_cpt.py`（胶水，允许）。命令行至少暴露 `--lr`、`--mix_gen`、`--steps`、`--seed`。`mix_gen=0.3` 时用 `datasets.interleave_datasets([py_train, wiki_train], probabilities=[0.7, 0.3], seed=42)`。注意 `interleave_datasets` 的概率是按样本不是按 token；打印每个 epoch 实际吃到的 Python token 占比，若低于 0.60，改成按 token 拼 batch。

优化器 AdamW，weight decay 0.01，warmup 为总步数的 1% 或 0（短训差别小）。`fp16` 或 `bf16` 按卡决定。梯度裁剪 1.0。每 50 step 在两个验证集上算一次平均 loss，写入 `artifacts/lesson09/gpu/{recipe}.jsonl`。

配方 1：

```bash
python3 train_cpt.py --recipe naive --lr 3e-3 --mix_gen 0.0 --steps 400 --seed 42
```

配方 2：

```bash
python3 train_cpt.py --recipe lowlr --lr 3e-4 --mix_gen 0.0 --steps 400 --seed 42
```

配方 3：

```bash
python3 train_cpt.py --recipe replay --lr 3e-3 --mix_gen 0.3 --steps 400 --seed 42
```

三条必须同一 `--steps`、同一截断长度、同一验证集。不要给配方 3 加更多步数来「补偿」通用数据，那会破坏对照。Ibrahim 的 compute-equivalent 是减少领域独特 token；课内用固定步数近似：配方 3 的领域 token 会少约 30%，这是回放的真实代价，写进曲线标题。

单张 24GB、序列 512、微批次 8，400 step 通常在一到两小时内。若 8GB 卡 OOM，把微批次降到 2，开 `gradient_accumulation_steps=4`，保持有效 batch 粗略不变。

### Step 5: 画双曲线并填探针表

用 jsonl 画图：横轴 step 或 token，纵轴验证 loss，六条线（三配方 × 两验证集）可以分两个图，每个图三条配方。读图时只允许用你自己跑出来的数。预期方向（不是验收数字）：

- 配方 1：领域损失下降最快，通用损失上升最多。
- 配方 2：两条都平一些；领域改善弱于配方 1。
- 配方 3：通用损失升幅小于配方 1；领域损失仍应低于 $t=0$，但终点通常差于配方 1。

然后跑 5.6 的探针。把「串域」次数单独记一列。若配方 1 在英文探针上开始吐 `def` / `import`，你就得到了格式被污染的直接证据，这比损失数字更好讲。

### Step 6: 加分项 TinyLlama-1.1B

把 `--model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` 换上，学习率改成 `4e-4` 与 `4e-5`，序列可 1024。仍是三条配方、同一验证协议。1.1B 全参数续训在 24GB 上会紧，需要梯度检查点和小微批次。若只训得动 LoRA，本课加分项视为未完成：CPT 要动的是慢速基座，LoRA 续预训练是另一套问题，会和第 10、11 课搅在一起。

7B 不在本课。不要为了「看起来更像生产」去灌 7B。

### Step 7: 写对应第 06 课的那一段

交付物里必须有一段不超过 200 字的说明，回答：配方 3 的 30% 通用数据，对应第 06 课缓冲的哪一个角色？compute-equivalent 有没有做到？若把 WikiText 换成 Python 验证集里抽回放（同分布），曲线会怎样，你有没有测？没测就写没测。

## 8. 配置与预算

| 项 | 主线（135M） | 加分（1.1B） | 档 A |
|---|---|---|---|
| 模型 | SmolLM2-135M 基座 | TinyLlama-1.1B 中间 checkpoint | 不加载 |
| 原论文峰值 LR | $3.0\times 10^{-3}$（WSD，20% decay） | $4\times 10^{-4}$（余弦，2000 warmup） | 机制实验自定 |
| 配方 1 / 2 LR | $3\times 10^{-3}$ / $3\times 10^{-4}$ | $4\times 10^{-4}$ / $4\times 10^{-5}$ | 相对比例 10 倍即可 |
| 领域数据 | the-stack-smol `data/python`，512+64 文件 | 同左，文件数可加倍 | 合成分布 |
| 通用代理 | WikiText-2 raw | 同左 | 合成分布 |
| 序列长度 | 512 | 1024 | 短 |
| 步数 | 400（约 0.5M-2M token，视 batch 而定） | 400-800 | 秒级 |
| 混合比 $\rho$ | 0.30 | 0.30 | 与 GPU 同方向 |
| 单卡时间 | 24GB 约 1-3 小时三条 | 24GB 数小时到过夜 | 小于 30 秒 |
| 显存 | 135M bf16 全参数通常小于 8GB 激活峰值，24GB 宽裕 | 需检查点 | 0 |

缩小配置：步数 100、文件数 128，只用来确认脚本能出 jsonl。缩小配置的曲线不得当作交付双曲线。拉长配置：把 Python 用满 10,000 文件、步数 2000，仍是短续训，远远不到 Ibrahim 的 100B+。

随机性：种子 42。单 seed 即可交作业。若两条配方终点差在验证噪声里，加 seed 43、44，报均值，不要挑最好的那次。

## 9. 验收

- 浏览器实验：先预测交叉区间，再运行；改 $\alpha$ 后重新预测。预测对过关条件以页面反馈为准。
- CPU：`python3 run.py run 09` 现在应当全绿，`checks` 全真：`pretrained_general_above_0_90`、`domain_only_general_drops`、`mix_drops_less_than_domain_only`、`mix_still_learns_domain`、`domain_only_learns_domain`。
- GPU 主线（有卡）：三份 jsonl + 一张双曲线图 + 探针表。图上必须同时出现通用和领域。
- 书面：配方 3 对应第 06 课回放的说明；以及「本课数字不是 Ibrahim 论文数字」的一句话。
- 禁止项：用 Instruct 权重；只报领域 loss；把模型卡 HellaSwag 42.1 写成你训出来的；把 30% 写成 Ibrahim 的超参；声称复现了 300B token 实验。

量化线（有 GPU 时，方向性，不是绝对阈值）：配方 1 的通用验证损失升幅应大于配方 3；配方 1 的领域验证损失终点应低于或接近配方 2。若方向反了，先查 `interleave` 是否按 token、学习率是否写反、验证集是否切错，再怀疑「135M 太小学不会」。CPU 机制实验的阈值以 `summary` 为准。

## 10. 排错

| 症状 | 原因 | 验证 | 修法 |
|---|---|---|---|
| 加载后指标和模型卡差很远 | 下到 Instruct 或损坏缓存 | 打印 `name_or_path` 和 `num_parameters` | 换 `SmolLM2-135M`，清 HF 缓存后重下 |
| OOM | 序列 8192 或微批次过大 | `nvidia-smi` 峰值 | 截断 512，微批次 2，梯度累积 |
| 配方 3 和配方 1 曲线重合 | 混合没生效 | 打印每个 step 的数据源计数 | 检查 `interleave` 概率、是否误把两个 dataset 先 concatenate 再 shuffle |
| 通用损失也降 | 验证集含代码，或步数极少 | 抽 20 条 Wiki 验证文本人工看 | 换 `wikitext-2-raw-v1` 的 validation；滤掉空行和代码围栏 |
| 领域损失不降 | LR 过小、数据太少、pad 全算进 loss | 看 `labels` 里 `-100` 比例 | 配方 1 必须用 $3\times 10^{-3}$；确认 collator `mlm=False` |
| Python 下载极慢 | 拉了全部语言 | 数据集卡 `num_rows` | 只设 `data_dir="data/python"` |
| LAMBADA 装不上 | 依赖冲突 | 不必卡主线 | 改用课内 20 题探针 |
| TinyLlama 乱码 | 用了 Chat 权重或错 tokenizer | 检查模型 id 是否含 `Chat` | 用 `...-intermediate-step-1431k-3T` |
| `domain_only_general_drops` 为假 | 纯领域没冲掉通用方向 | `metrics.drop_domain_only` 应大于 0.15 | 看 5.2：两套下一词方向是否真冲突；学习率或步数是否过小 |
| `mix_drops_less_than_domain_only` 为假 | 混 30% 没起到回放作用 | 对比 `drop_mix` 与 `drop_domain_only` | 看 5.4 的 $\rho$；确认 batch 里写进了 `replay_frac=0.3` |
| `mix_still_learns_domain` 为假 | 混数据后领域也没学上 | `mix_domain` 应高于 0.85 | 先确认 `domain_only_learns_domain` 为真；领域方向不应被通用方向淹没 |

## 11. 前沿与改造

前沿怎么做：Ibrahim et al. 2024 之后，回热 + 再退火 + 回放已经成为开源续预训练的默认口头禅。Atreya et al. 2026（[arXiv:2608.17530](https://arxiv.org/abs/2608.17530)）把「混多少旧数据」再拆成「这一步该复习哪几条」，按样本困惑度排 SuperMemo-2 日程，不改损失和优化器。DeepSeek-V2 公开技术报告写过用未衰减完的中间 checkpoint 接续训、并混约 30% 预训练回放去加代码能力（这是他们报告里的设定，不是你能在 135M 上复现的）。Glorioso et al. 2024 在高质量衰减阶段用过更高回放比。另一条线是「无限学习率日程」（inverse-sqrt 或 WSD 的稳定段不绑定总 token），用来避免每次回热把通用知识掀一遍。Ke et al. 2023 则走无旧数据的软掩码。LLaMA Pro（Wu et al. 2024, arXiv:2401.02415）用扩块来写新知识、冻旧块，那是第 07 课扩结构在 LLM 上的亲戚，本课配方不扩参数。

我们差在哪：token 少六个数量级；没有并集从头训上限；通用代理是 WikiText 不是 FineWeb；学习率是常数不是完整回热余弦；模型 135M 不是 10B。差这些并不是偷工，是为了把「双曲线」和「回放=第 06 课」跑完。

动手改造（2-4 个，01-12 课可以精简，这里仍给可执行的）：

1. 把配方 3 的 $\rho$ 扫 0.05、0.10、0.30、0.50，画通用升幅对 $\rho$ 的曲线。预算：135M 再跑三条短程。预期：弱偏移下 0.05 可能几乎看不见，0.30 才分开。失败标准：所有 $\rho$ 的通用曲线在噪声带内重合，且数据源计数证明混合有效，这时才有资格说「本预算下回放无用」。
2. 给配方 1 加上余弦再退火（峰值 $3\times 10^{-3}$，终点 $3\times 10^{-4}$）。预期：通用损失的末期比常数硬上更回得来一点。失败标准：400 step 内余弦与常数无差别（很可能，因为步数太短），那就加到 2000 step 再判。
3. 把领域从 Python 换成 the-stack-smol 的 `data/javascript`，或课内可再分发的小型法律/医学样本。预期：偏移形态类似，绝对损失不同。失败标准：用了不可再分发的内部语料。
4. 知识/格式探针改成「把名录撤掉」式的 20 条公司事实（接 [第 04 课](04_not_just_rag.md)）。预期：CPT 不保证记住事实条款，下一词损失降也不等于事实写入。失败标准：用生成流畅度代替事实准确率。

顺手复现映射：本课不是正式复现。读懂 Ibrahim 的图 1 和表 2 即可。第 11 课才对 O-LoRA 做方向性复现。

## 12. 论文与延伸

1. Ibrahim et al., 2024, *Simple and Scalable Strategies to Continually Pre-train Large Language Models*, [arXiv:2403.08763](https://arxiv.org/abs/2403.08763)。
贡献：回热、再衰减、回放合在一起，就能在 405M 和 10B 上把续预训练拉到接近「新旧并集从头训」。
机制：预训练末期学习率已经很小，他们先把学习率抬回去再按余弦衰减，让新分布吃得动。回放按 batch 比例掺旧语料，并按 compute-equivalent 少吃同等数量的新 token，总计算量对齐。HTML 图 1：弱偏移（Pile 接到 SlimPajama）用 5% 回放，强偏移（接到德语）用 25%。405M 表：纯续训 Pile 验证损失 2.44，5% 回放到 2.23，并集从头训 2.17；德语纯续训旧损失 3.56，25% 回放到 2.33，并集 2.26。课内没有并集上限。
和本课：CPU 的 `mix_drops_less_than_domain_only` 看见「混旧数据后通用掉得更少」，`mix_still_learns_domain` 看见领域仍能学。回热/再衰减、405M/10B、并集从头训本课答不了。
阅读问题：课内 $\rho=0.30$ 且固定步数，不是论文的 5%/25% compute-equivalent。你的 `drop_mix` 是否仍比 `drop_domain_only` 小至少 0.08？这只支持「混有用」，不支持「匹配从头训」。

2. Li and Lee, 2024, *Examining Forgetting in Continual Pre-training of Aligned Large Language Models*, [arXiv:2401.03129](https://arxiv.org/abs/2401.03129)。
贡献：在已经对齐的 Llama-2-7b-chat 上用约 10 亿繁中 token 做 CPT，分输出格式、知识、可靠性三路看遗忘。
机制：全参续训，对照冻前/后 10 层、冻注意力或 MLP、LoRA、(Ia)$^3$。格式侧测语言识别和 BPE 级 rep-n；知识侧测 ARC / HellaSwag / MMLU / C-eval-tw；可靠性测 TruthfulQA / ToxiGen / BOLD。HTML 结论：知识基准几乎不动，可靠性下降；中文提示下 chat-cp 的 rep-4 从约 0.103 升到约 0.552。冻层和适配器解不掉这件事。
和本课：主线加载的是基座不是 Instruct，CPU check 测的是分类准确率不是重复率和毒性。探针表若看见英文填空吐 `def`，对应他们的输出格式被冲。可靠性三项本课实验答不了。
阅读问题：若你误用 Instruct 权重做配方 1，通用曲线掉的更像对齐格式被冲，还是下一词被 Python 带偏？用加载的模型名和探针表回答；若你没用 Instruct，写「本课没跑这一对照」。

3. Wu, Luo, Li, Pan, Vu, Haffari, 2024, *Continual Learning for Large Language Models: A Survey*, [arXiv:2402.01364](https://arxiv.org/abs/2402.01364)。
贡献：按续预训练、指令微调、对齐给 LLM 持续学习分阶段，并对照 RAG 和模型编辑。
机制：这篇改的是评测地图，不改损失。CPT 继续做自监督下一词；CIT 在指令-输出上做监督；CA 用偏好数据对齐。摘要还写了跨阶段风险：已经指令微调过的模型再去做 CPT，指令跟随会掉。
和本课：三条配方落在 CPT。CPU 双曲线只能看见领域对通用，看不见「Instruct 再 CPT」那一截；那截要对照第 2 篇，并且本课禁止用 Instruct 做主线。
阅读问题：用本课双曲线说明，为什么「领域验证损失下降」不能单独证明 CPT 成功？指出必须同时报哪一条 check。

4. Shi et al., 2024, *Continual Learning of Large Language Models: A Comprehensive Survey*, [arXiv:2404.16789](https://arxiv.org/abs/2404.16789)。
贡献：把 LLM 持续学习拆成纵向（通用到专用）和横向（跨时间、跨领域），训练阶段写成 CPT、领域自适应预训练 DAP、持续微调 CFT。
机制：同样是评测与分类，不改你的优化器。CPT 是接着做大规模自监督；DAP 是接到某个领域语料上适应；CFT 才是指令或任务监督。本课 Python 续训更接近「横向换领域的 CPT」，不是 CFT。
和本课：浏览器「数据配比」和 CPU 混 30% 对应他们说的水平连续。纵向「通用基座接到专用能力」本课只做了下一词，没有下游任务头。
阅读问题：若有人把本课三条配方写成 DAP 而不是 CPT，你用损失形式反驳：本课有没有任务标签？

5. Roth et al., 2024, *A Practitioner's Guide to Continual Multimodal Pretraining*, [arXiv:2408.14471](https://arxiv.org/abs/2408.14471)。
贡献：给介于「几年灌一次大更新」和「每条样本都更新」之间的实际部署，做了持续多模态预训练的测试床和操作指南。
机制：基准叫 FoMo-in-Flux，63 个视觉-语义数据集，算力和部署约束写进协议。他们扫数据配比与流顺序、从微调/传统 CL 到 PEFT 和模型合并、学习率日程，以及模型和计算规模。代码在 ExplainableML/fomo_in_flux。
和本课：配比和日程这两条，浏览器 $\alpha$ 滑块和配方 1/2/3 能对上方向。63 数据集、视觉编码器、合并，本课实验答不了。
阅读问题：把 FoMo-in-Flux 的「子域更新」映射到本课，Python 子集相当于他们的哪一类流？你的 `mix_general_frac=0.3` 对应他们的数据配比，还是对应模型合并？

6. Atreya, Batra, Mantri, Bantug, Cowan, Khraishi, 2026, *When to Review: Spaced Repetition for Continual Pre-Training of Language Models*, [arXiv:2608.17530](https://arxiv.org/abs/2608.17530)。
贡献：续预训练的回放除了选一个全局新旧比，还要按样本决定这一时刻复习哪几条。
机制：SRT 给每条样本存 SuperMemo-2 状态（易度、间隔、到期步）。前向算出的困惑度映射成回忆质量，难样本早回来，稳样本拉长间隔。模型、下一词损失、优化器都不改。摘要写：在按时间切开的维基和代码上，相对 naive CPT 能找回 5 到 37 个百分点的旧知识准确率，新知识不掉或更好。
和本课：配方 3 的 30% 是全局均匀混。CPU 能证明「混比不混少忘」，不能证明「按困惑度挑哪几条更好」。SRT 的百分点本课答不了。
阅读问题：若把 CPU 实验的 `replay_frac` 从均匀 0.3 改成「通用样本里损失最高的先回放」，你预期 `drop_mix` 会怎样变？本课没实现调度器，写「答不了，差在没有 per-example 状态」即可。

7. Allal et al., 2025, *SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model*, [arXiv:2502.02737](https://arxiv.org/abs/2502.02737)。
贡献：用数据配比和分阶段混料，把 1.7B 小模型在约 11 万亿 token 上训到能打 Qwen2.5-1.5B / Llama3.2-1B 这一档，并公开 FineMath、Stack-Edu、SmolTalk。
机制：多阶段训练，网页文本混进数学、代码、指令数据；现有集合太小或太脏的阶段就换新集合。小规模消融加上按上一阶段指标手调混合比。这是从随机初始化的预训练，不是 Ibrahim 那种接着训。
和本课：主线加载的是同系列 135M 基座，模型卡和数据故事来自这里。11T token、1.7B、分阶段手调配比，本课 400 step 短续训答不了。
阅读问题：本课为什么必须用基座而不是 Instruct？结合这篇的「指令数据出现在后阶段」和第 10 课 CIT，写清你加载错权重时双曲线在测什么。

8. Tu, Fang, Wang, Xie, Yan, 2026, *Chain-of-Experience for Continual LLM Improvement*, [arXiv:2608.18027](https://arxiv.org/abs/2608.18027)。
贡献：把「持续变好」写成推理时的经验链：模型靠自反馈或环境信号迭代攒轨迹，不在这一设定里更新 $\theta$。
机制：反馈可以是模型自评，也可以是对错或公开代码测试通过率。评测覆盖数学、代码、知识，用了包括 GPT-5、Gemini-2.5 Pro、Claude-4.5 Sonnet 在内的 8 个模型。摘要写相对无反馈基线总体约 5.6% 提升、API 成本约低 19%；互补反馈通道还能再涨，多数收益出现在早期迭代。
和本课：CPU 混 30% 改的是慢速权重。这篇改的是推理轨迹，权重不动。本课实验答不了 5.6% 和成本，因为没有环境反馈。
阅读问题：若产品经理把 CoE 的「持续改进」和本课配方 3 写成同一件事，你用「写到哪里」反驳：一条改 $\theta$，一条只改上下文里的经验。本课哪条 check 能证明你改了 $\theta$？

下一课要把对象从「下一词混料」换成「一条条指令任务」。CPT 掉的是通用分布；CIT 掉的是前一个任务的听指令方式。Wu 的第二截从那里开始。[第 11 课](11_olora_treelora.md) 再问：每个任务一个 LoRA，为什么还要两两正交。本课的双曲线请留着：后面有人把「指令遗忘」和「通用语言遗忘」说成一件事时，你可以用第 09 课的 WikiText 曲线和第 10 课的 4×4 把它们拆开。

