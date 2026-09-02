---
id: 09_iris_token_world_model
title: "IRIS 把未来写成 token"
summary: "把图像离散化成 token 会丢什么？为什么语言模型的配方能直接搬来建世界？"
unit: generative
play_tools: []
checkpoints:
  - "单游戏训练记录（论文复现 #4）。"
  - "“在梦里玩游戏”的体验报告：模型哪里穿帮，为什么。"
---

# 第 09 课：IRIS 把未来写成 token

> 类型：体验（进预训练世界模型里玩 Breakout）+ 复现（论文复现 #4：IRIS 单游戏方向性复现）<br>
> 建议周期：4-7 天（体验与码本审计一天内搞定，100k 训练的大头是机器在跑）<br>
> 硬件：单张 24GB 卡训练绰绰有余；玩预训练权重只要几分钟，普通 N 卡即可；码本审计 CPU 能跑<br>
> 锚定仓库：[eloialonso/iris](https://github.com/eloialonso/iris)（ICLR 2023 notable top-5% 论文官方实现，src 下不足 3000 行 Python，HuggingFace 提供 26 个游戏的预训练权重）<br>
> 产物："在梦里玩游戏"体验报告（穿帮镜头集 + 逐条归因）、码本使用率审计报告、单游戏 Atari 100k 训练记录

## 1. 这一课做什么

前面的模型主要服务于控制：只要奖励、价值或动作选择可靠，生成画面是否清晰并不重要。
从这一课开始，模型还要把预测结果直接展示给人。球消失、砖块复活或分数乱码都会立即
破坏交互，因此生成保真不再是可选项。

IRIS（ICLR 2023）提供了一条紧凑的实现路径。它先用离散自编码器把每帧画面切成 16 个
"词"（token，从一本 512
词的词表里选），再让一个 GPT 式的 transformer 学着续写这门语言，预测未来从此变成
"预测下一个 token"，从而直接复用自回归语言模型的训练方式。它的官方实现较小，src
目录不足 3000 行 Python，transformer 基于 minGPT，tokenizer 参考
taming-transformers，适合完整读一遍。

作者还发布了 26 个 Atari 游戏的预训练权重。训练自己的模型前，可以先运行 Breakout：
你按键盘，transformer 续写下一帧，画面是从 token 解码出来的、从未真实发生过的未来。
运行时要记录可复现的失败片段，并区分问题来自 token 化的信息损失，还是上下文长度
不足。

本课的产物包括一段带归因的失败视频、一份 512 个码字的使用率审计，以及一条 Atari
100k 训练曲线。下一课的 DIAMOND 会在相同作者、基准和训练预算下换掉预测引擎，因此
这三项记录将构成 token 与扩散对照的基线。

术语速查：

| 术语 | 一句人话 |
|---|---|
| token | 画面被切成的离散"词"：一帧 64×64 的画在 IRIS 里是 16 个词，每个词是 0 到 511 的一个整数 |
| 码本（codebook） | 词表本体：512 个可学习的向量，编码时按"谁最近选谁"查表，词的编号就是向量在表里的行号 |
| VQ / 向量量化 | 把连续向量换成码本里最近的那个码字，第 02 课"压成向量"从此变成"切成词" |
| straight-through | 绕过"查表"这步不可导操作的梯度小手术：前向用量化结果，反向假装量化没发生过 |
| 码本坍缩 | 码本的慢性病：大多数码字从没被选中过，512 词的词表实际只剩几十个活词 |
| LPIPS | 感知损失：用预训练视觉网络的特征差距离量两张图"看起来"差多远，补逐像素损失的盲区 |
| 自回归 | 一个词一个词往外蹦，每个词以前面全部词为条件，GPT 的出词方式，这里用来出"未来" |
| KV cache | transformer 推理提速件：已算过的键值存下来，新 token 只算增量；你能实时玩梦全靠它 |
| Atari 100k | 样本效率标尺：只许与真游戏交互 10 万步（40 万帧，约两小时人类游戏时间），看能学多好 |
| 人类归一化分数（HNS） | 把各游戏分数换算到"随机策略 0 分、人类玩家 1 分"的统一刻度再平均 |
| burn-in | 做梦前先用几帧真实历史把记忆热起来，别让模型从失忆状态开始想象 |
| 穿帮 | 本课的工作词：模型播放的未来里违背游戏规则的瞬间，球消失、砖复活、比分乱跳 |

## 2. 问题

需要回答三个问题：

1. 图像离散化会损失哪些信息。一帧 64×64×3 的画面有 12288 个连续数，
   token 化之后只剩 16 个整数，每个整数至多 9 个比特，信息上限被词表大小和词数
   焊死了。丢东西是必然的，问题是丢的是什么：你会在梦里看到答案（几个像素大
   的球、快速闪烁的细节最先遭殃），再用码本审计把这个答案变成数字。
2. 语言模型配方为何适用于世界模型。第 03 课的 MDN-RNN 为了表达"未来有
   好几种可能"，专门造了混合高斯这门手艺；而世界一旦变成词序列，softmax 天生就是
   512 路的多峰分布，多样性免费附送。transformer、交叉熵、温度采样、KV cache，
   语言模型十年攒下的全套基础设施，一个都不用改。这里把这次"搬家"的每个对应关系
   讲清楚，你会发现 IRIS 的训练循环和第 04 课"梦里训 C"、第 06 课"想象里训
   actor-critic"是同一张图纸。
3. tokenizer 的验收方法。前两幕的验收工具（分数、损失、探针）都量不出
   "生成保真"。本课建立第一件专用工具：码本使用率审计，把 512 个码字的使用
   直方图跑出来，数清多少个词从没被用过。这是第三幕后面几课反复要用的手艺。

先划界限：这次复现是**单游戏、单种子的方向性复现**。论文的招牌数字（26 个游戏
平均 HNS 1.046、10 个游戏超过人类）需要 26 个游戏各 5 个种子，那是几十倍的预算；
你复现的是其中一个切片，验收标准是落在官方该游戏的种子区间附近。穿帮报告同理，
它是定性证词，配合归因才有价值，单独一句"画面会糊"什么也说明不了。

## 3. 准备

- 场地：本课的体验环节是一个 pygame 实时窗口加键盘操作，**必须在有屏幕的机器上
  跑**，远程服务器套 xvfb 只能骗过渲染，骗不出一块你能按键的屏幕。推荐分工：训练
  挂在远程 GPU 上，体验环节在本地机器做（预训练权重才 127MB，本地有一张普通 N 卡
  就够；没有 N 卡的话把 `config/trainer.yaml` 里 `common.device` 改成 `cpu` 也能试，
  帧率以实测为准，卡就用 `-f` 把每秒帧数调低）。
- 软件：这是 2022 年的仓库，README 注明代码用 torch 1.11 时代开发，依赖清单里
  钉死了 `gym==0.21.0`、`ale-py==0.7.4`、`hydra-core==1.1.1`。给它单开一个
  Python 3.9 或 3.10 的虚拟环境，一个 pin 都别动，老 gym 在新工具链下的安装坑，
  第 10 节症状表第一行伺候。
- 硬件：训练单卡即可，24GB 很宽裕（显存大头是 tokenizer 的 batch 256）；内存
  留 32GB 以上最稳，训练数据全部住在内存里，默认上限 30G，小内存机器有一行配置
  可调（第 8 节讲）。磁盘几个 GB 足够。
- 上一课的产物：无文件级依赖，但有三个概念要热在手边：第 03 课的多步漂移曲线
  和"动作盲"审讯、第 04 课的温度 τ 与"策略钻模型空子"、第 06 课的想象训练与
  λ-return。忘了的话各自回去翻五分钟，本课处处要用。
- 心理预期：体验环节几分钟就有回报；100k 训练则是 600 个 epoch 的马拉松，
  官方没给参考耗时，按"单卡数天"的量级安排，挂后台跑，中途每天回来看一眼
  `media/` 目录里的重建图和想象片段就好。

## 4. 学习目标

1. 白纸上画出 IRIS 的数据流：一帧画面如何变成 16 个 token、一步交互如何变成 17 个
   token 的块、transformer 如何在 340 个 token 的窗口里续写未来、未来又如何被解码
   回像素，每个箭头标注张量形状和词表大小；
2. 手推 VQ 的三件套：最近邻查表怎么算、straight-through 让梯度怎么走、commitment
   损失的两项各拽住谁；
3. 解释码本坍缩的成因（赢者通吃、输者永远没梯度），说出两味常见药（EMA 码字更新、
   重启死码），并指出 IRIS 的 tokenizer 用没用它们（答案：都没用，正因如此才有
   你的审计作业）；
4. 把 IRIS 和第 03 课的 MDN-RNN 摆成一张对照表：都是自回归预测下一状态，谁在管
   多峰、谁在管记忆、温度旋钮各拧在哪；
5. 独立完成一次码本使用率审计：使用直方图、死码计数、困惑度三个数各是什么、怎么算；
6. 看到梦里的一个穿帮镜头，能把它归因到三类病灶之一，token 语言词汇不够
   （tokenizer 的锅）、上下文遗忘（transformer 窗口的锅）、逐 token 采样抖动
   （解码方式的锅），并说出用什么实验区分。

## 5. 原理

五个机制，照老规矩走：直觉、机制、数学、代码落点、验证。

### 5.1 VQ：从"压成向量"到"切成词"

第 02 课的 VAE 把画面压成 32 个连续数，第 05 到 08 课的潜状态也全是
连续向量。连续表示有个天生的别扭：想在它上面做"预测未来"，就得建模连续分布，
第 03 课你为此学了混合高斯，调过高斯个数，见识过它的娇气。而语言模型那边风景
完全不同：词表是有限的，预测下一个词就是一次 softmax 分类，训练是交叉熵，采样是
掷骰子，十年工程把这条路铺得又宽又稳。VQ（向量量化）就是过路费：把连续的图像
表示硬掰成离散的词，掰完之后，整个语言模型工具链对你敞开。

IRIS 的 tokenizer 是一个卷积自编码器加一本码本。编码器把 64×64×3 的
画面（Atari 原始画面 210×160，先双线性缩放到 64×64）逐级下采样四次，得到一张
4×4 的特征图，每个格子是一个 512 维向量，注意此时还是连续的。量化这一步：拿
每个格子的向量去码本（512 行、每行 512 维的一张表）里找**欧氏距离最近**的那一行，
把向量替换成那一行，同时记下行号。16 个格子就得到 16 个行号，这是这一帧的
16 个 token。解码器只看被替换后的向量，把它们放大回 64×64 的画面。词表大小 512、
每帧 16 个词，都写在 `config/tokenizer/default.yaml` 里（`vocab_size: 512`，
分辨率 64、四次下采样到 4×4）。

记编码器在格子 $i$ 的输出为 $z_i$，码本为 $e_1, \dots, e_{512}$。量化：

$$
k_i = \arg\min_j \, \lVert z_i - e_j \rVert_2, \qquad \hat z_i = e_{k_i}
$$

argmin 不可导，梯度到这里就断了。straight-through 的做法：解码器的输入写成
$z_i + \operatorname{sg}(\hat z_i - z_i)$（$\operatorname{sg}$ 是停止梯度），前向时
这个式子恰好等于 $\hat z_i$，反向时 $\operatorname{sg}$ 项贡献为零，重建损失的梯度
原封不动流回 $z_i$，等于假装量化没发生。码本自己则靠 VQ 损失学：

$$
\mathcal{L}_{\mathrm{VQ}} = \lVert \operatorname{sg}(z_i) - \hat z_i \rVert_2^2 + \beta \, \lVert z_i - \operatorname{sg}(\hat z_i) \rVert_2^2
$$

第一项把被选中的码字往编码器输出上拉（训码本），第二项把编码器输出往码字上拉
（训编码器，术语叫 commitment：你选了这个词就别老想跑）。IRIS 取 $\beta = 1$，
源码注释专门写了一句：β 的位置与原版 VQ-VAE 论文一致、和 taming-transformers 不同，
原版默认用 0.25。重建那头，IRIS 用 L1 逐像素损失加 LPIPS 感知损失，不用 VAE 的
KL 项，tokenizer 不需要潜空间规整，只需要词表好用。

`src/models/tokenizer/tokenizer.py::Tokenizer`。最近邻查表在
`encode`：先算 `dist_to_embeddings`（展开的欧氏距离矩阵），一行
`tokens = dist_to_embeddings.argmin(dim=-1)` 完成切词。straight-through 在
`forward`：`decoder_input = outputs.z + (outputs.z_quantized - outputs.z).detach()`。
三项损失全在 `compute_loss`：`commitment_loss`（上面那两项，β=1）、
`reconstruction_loss`（L1）、`perceptual_loss`（LPIPS）。码本就是一个
`nn.Embedding(vocab_size, embed_dim)`，初始化为 ±1/512 的均匀分布。编码器和解码器
在同目录 `nets.py`，师承 taming-transformers。

拿任何一帧过一遍 `tokenizer.encode(frame, should_preprocess=True)`，
打印 `.tokens.shape`，应该是 `(1, 16)`：一帧画面，16 个 0 到 511 的整数。第 7 节
Step 6 的审计脚本第一步就是它。

### 5.2 码本坍缩：赢者通吃的慢性病，和 IRIS 没吃的两味药

最近邻查表是一场赢者通吃的选举：每个格子只把票投给离自己最近的那个
码字，**被选中的码字才有梯度**，VQ 损失的第一项只作用在 $\hat z_i$ 上，落选的
510 个码字这一步纹丝不动。麻烦在于初始化是随机的：有些码字生下来就恰好站在编码器
输出的聚集区附近，开局连赢；有些生在荒郊野外，一票没有。没票就不更新，不更新就
永远追不上编码器输出的分布，于是强者恒强、弱者饿死。训练结束一盘点，512 个词的
词表可能只有几十个词在干活，这是码本坍缩（codebook collapse）。词表名义上 512，
实际容量缩水到几十，画面里的细节自然没词可说。

机制与常见药。社区攒了几味标准药方。第一味：**EMA 码字更新**，不靠梯度，
直接拿"最近被分到这个码字的所有编码器输出的滑动平均"去更新码字，原版 VQ-VAE
论文附录就给了这个变体，好处是码字追得快、不受损失权重摆布。第二味：**重启死码**
，训练中定期盘点，把长期没被选中的码字直接重置到某个当前批次的编码器输出上，
等于把饿死的词空投到人口稠密区。此外还有降低码字维度、对码本做归一化等做法，
思路都一样：别让距离比较在高维空间里僵住。**IRIS 的 tokenizer 一味药都没吃**：
没有 EMA，没有重启，就是最朴素的 VQ 损失。这不是疏忽就是取舍，Atari 画面简单，
可能根本用不满 512 个词。是哪种？别猜，第 7 节 Step 6 你审计。

审计用三个数说话。把一批真实画面切词，统计每个码字被用的次数，归一化
成使用分布 $p_1, \dots, p_{512}$：死码数是 $p_j = 0$ 的个数；头部集中度是使用最多
的 10 个码字的概率和；困惑度是

$$
\mathrm{PPL} = \exp\Big(-\sum_j p_j \log p_j\Big)
$$

它的读法：512 个词被完全均匀地使用时 PPL 等于 512；PPL 等于 40 就意味着这本词表
的有效词汇量只相当于 40 个词。

这一节的代码落点是个**负空间**：打开 `tokenizer.py` 从头读到尾
（一共 105 行），确认里面没有任何 EMA、重启或使用率统计的代码，`compute_loss`
就三项损失，`encode` 就一次 argmin。确认"没有"和确认"有"一样是读代码的正经
产出，第 11 节的改造清单会让你把重启死码补进去。

第 7 节 Step 6 的审计脚本，以及一个对照：同一脚本跑预训练权重和你自己
100k 训出的权重，比较两者的死码数和困惑度。

### 5.3 世界模型 = 序列建模：语言模型配方的搬家清单

画面变成词之后，一局游戏就变成了一篇文章：第 1 帧的 16 个词、你按的
第 1 个动作（动作本来就是离散的，天生是词）、第 2 帧的 16 个词、第 2 个动作……
"预测按下这个键之后世界会怎样"，从此和"续写这篇文章"是同一道题。GPT 怎么写文章，
IRIS 就怎么造未来。

每步交互打包成一个 17 个 token 的块（`tokens_per_block: 17`）：16 个
画面词加 1 个动作词，动作词和画面词各用各的嵌入表。transformer 是 minGPT 血统的
标准因果结构：10 层、4 头、嵌入 256 维，窗口 `max_blocks: 20` 个块，也就是
20 帧交互、340 个 token（`config/world_model/default.yaml` 全写着）。它头上顶三个
输出头：**观察头**在每个位置预测下一个画面词（512 路 softmax）；**奖励头**只在
动作词的位置发言，预测这一步奖励的符号（三分类：负、零、正）；**终止头**同样在
动作词位置，预测"这局是不是到头了"（二分类）。奖励头和终止头你在第 04 课的梦境
环境里见过同款，梦要自己发工资、自己喊停，否则策略在里面没法训。生成未来时
一帧的成本是 17 次前向：喂进动作词，依次采样出 16 个画面词，KV cache 让每次前向
只算增量。

三个头全是交叉熵，没有一个回归项。对照第 03 课，这是"搬家"的核心
交易：MDN-RNN 预测下一个连续 $z$，被迫用混合高斯去表达"未来有几种可能"，高斯
个数是超参、NLL 训练容易翻车；IRIS 预测下一个离散词，softmax 天生就是 512 路
多峰分布，多样性不要钱。两者又惊人地同构，都是自回归模型按动作条件预测下一
状态，都从预测分布里**采样**（这是梦有随机性的来源），都有温度旋钮可拧（第 04
课拧 τ 防梦被钻空子，IRIS 的采样代码里同样能插 τ，第 11 节改造清单第 4 条）。一张
对照表钉住：

| | 第 03 课 MDN-RNN | 本课 IRIS |
|---|---|---|
| 状态表示 | 32 维连续向量 $z$ | 16 个离散词（512 词表） |
| 预测器 | LSTM，256 隐单元 | 因果 transformer，10 层 256 维 |
| 多峰未来怎么表达 | 混合高斯（高斯个数是超参） | softmax 天生多峰 |
| 训练目标 | 混合高斯 NLL | 交叉熵 |
| 记忆 | 递归隐状态，理论无限长、实际会淡忘 | 显式窗口 20 帧，窗口内无损、窗口外全忘 |
| 奖励与终止 | 附加的标量头 | 三分类与二分类头 |

记忆那一行是本课穿帮实验的伏笔：LSTM 的遗忘是渐变的，transformer 的遗忘是断崖
，第 20 帧还记得清清楚楚，第 21 帧整个世界从缓存里被清掉。

`src/models/world_model.py::WorldModel`。三个头是三个 `Head`，各配
一个 0/1 模式串决定在哪些位置发言（`src/models/slicer.py` 负责按模式切片）；
其中观察头的模式 `all_but_last_obs_tokens_pattern` 在构造函数里有一处值得盯住的
细节：模式串的倒数第二位被置零，因为在最后一个画面词的位置，"下一个 token"
是动作词，而动作是玩家的自由意志，世界模型不该去预测它（对照第 07 课：MuZero
恰恰要预测策略）。损失在 `compute_loss`：观察头的标签就是错开一位的 token 序列，
奖励标签是 `rewards.sign() + 1`。KV cache 在 `src/models/kv_caching.py`。

体验环节按同一套历史重玩几次同一个动作，画面走向应当不同，那是
softmax 采样在掷骰子。如果每次完全一样，说明你看的是重放不是想象。

### 5.4 在 token 的想象里训练 actor-critic：Dreamer 图纸，token 引擎

到这里 IRIS 只是个"能续写的世界"，还没有会玩的手。训练策略的图纸
直接抄第 06 课：把 actor-critic 整个泡在想象里，一步真实交互都不花。差别只在
引擎，Dreamer 的想象在 RSSM 潜空间里滚，IRIS 的想象是 transformer 逐词续写、
再**解码回像素**给策略看。

每轮想象训练：从回放数据里抽一段真实片段，除最后一帧外的历史帧先过
一遍 tokenizer 的编码再解码（重建帧），喂给策略网络做 burn-in 热记忆；然后从片段
的最后一帧出发，接下来的 20 步全在梦里，策略（一个 CNN 加 LSTM 的小网络）看着
解码出的画面选动作，
transformer 吃动作吐出下一帧的 16 个词、奖励符号和终止判断，词再解码成画面还给
策略，循环 20 步（`imagine_horizon: 20`）。训练目标和第 06 课同款：λ-return 做
critic 的靶子（$\gamma = 0.995$、$\lambda = 0.95$），actor 吃策略梯度加熵正则
（权重 0.001）。整个训练循环是三段轮转：每个 epoch 先在真环境里采 200 步，然后
依次训 tokenizer、世界模型、actor-critic 各 200 个梯度步，三个零件一直在同步
长大，不是第 04 课那种"训完 V/M 再训 C"的两段式。

一个容易漏看的设计。策略在真环境里行动时，看到的也**不是原始画面**：
`agent.py::Agent.act` 里，真实观察先过 `tokenizer.encode_decode` 变成重建帧，再进
策略网络。梦里看重建、真环境里也看重建，两边画质一致，策略就不会因为"训练时看
糊图、上场时看高清"而水土不服。这是对第 04 课"梦和现实有落差、策略钻空子"问题
的一记直接补丁，消除落差的办法是把现实降级到梦的画质。代价当然也有：策略永远
隔着 tokenizer 看世界，tokenizer 没词表达的东西，策略就永远看不见。

λ-return 的定义第 06 课推过，此处不再展开；唯一要记的结构差异：IRIS 的
critic 直接回归标量价值（MSE），没有 twohot 和 symlog，Atari 单游戏奖励已被
符号化到 $\{-1, 0, 1\}$，用不上那套跨量级稳定件。这反过来印证第 08 课读 DreamerV3
时的判断：那些稳定件是为"一套超参通吃百种任务"准备的工程通用件，单游戏场景
可以省。

`src/models/actor_critic.py::ActorCritic`：`imagine` 方法就是上面
的想象循环（里面原地造了一个 `WorldModelEnv`），`compute_loss` 是 λ-return 加
策略梯度；burn-in 在 `reset` 方法。想象引擎本体在
`src/envs/world_model_env.py::WorldModelEnv`：`step` 里那个循环 17 次的 for 就是
"一个动作词进、16 个画面词出"，奖励和终止直接从 Categorical 分布里
采样。三段轮转的调度在 `src/trainer.py::Trainer.train_agent`，采集在
`src/collector.py`（训练采集用 $\epsilon = 0.01$ 的采样策略，测试采集温度 0.5）。

训练目录 `media/episodes/imagination/` 里存着想象轨迹的视频，模型
自己做的梦定期落盘。拿它和 `media/episodes/train/` 的真实轨迹对照看，是比损失
曲线直观得多的健康检查。

### 5.5 离散化丢什么：小而快的东西最先死

现在正面回答第 2 节的第一个问题：离散化到底丢了什么。一帧 64×64 的
画面切成 4×4 的格子，
每个词管一块 16×16 像素的领地。Breakout 的球直径只有两三个像素，在词的
领地里，球只是 256 个像素中的几个点。词表只有 512 个词，同一块领地里"有球在
左上角"、"有球在右下角"、"没有球"可能压根分不到三个不同的词；就算分到了，
重建损失也未必在乎，L1 按像素平均计账，球占的像素连百分之一都不到，牺牲它换
背景和砖墙的精确，账面上稳赚。LPIPS 补救了一部分"看起来像不像"，但感知网络对
两三个像素的小目标同样不敏感。

于是丢失有了清晰的谱系，玩梦时对号入座：

1. 词汇不够（tokenizer 的锅）：球时隐时现、球的位置跳格、砖块边缘对不齐，
   领地内的小细节没有专属的词，量化时被四舍五入到最近的"近义词"。
2. 上下文断崖（transformer 的锅）：窗口只有 20 帧，更早的历史被 KV cache 清掉。
   玩的时候画面每隔一阵轻微跳变一次，那是缓存清空、世界只凭当前帧重新开局，
   README 自己写明"为了交互流畅，transformer 的记忆每 20 帧清空一次"。清空前后，
   已消掉的砖可能复活，比分可能重置。
3. 采样抖动（自回归生成的锅）：16 个词逐个采样，每个词都可能掷出小概率结果，
   而且后采的词以先采的为条件，一个词跑偏，剩下的词跟着圆谎。多步下去就是第 03
   课漂移曲线的 token 版：误差不是渐渐模糊（连续模型的死法），而是突然跳到另一个
   "语法通顺但事实错误"的画面。

信息账很好算：16 个词、每个 $\log_2 512 = 9$ 比特，一帧的表示上限是
144 比特；原始帧是 $64 \times 64 \times 3 \times 8 = 98304$ 比特。压缩率近 700 倍，
必有牺牲，问题从"丢不丢"变成"丢谁"。VQ 的回答由损失函数决定：L1 加 LPIPS
投票，票权正比于像素面积和感知显著度，小而快的物体两头都输。

没有单独的代码，这是 5.1 到 5.3 三个机制的合谋。要说有，就是
`config/tokenizer/default.yaml` 里的 `vocab_size: 512` 和
`config/world_model/default.yaml` 里的 `max_blocks: 20`，两个决定"丢什么"的
总旋钮，第 11 节改造清单第 1、2 条分别拧它们。

本课交付物"穿帮镜头集"就是这一节的验证：每个镜头标注三类病灶之一，
并给出你的证据（比如：跳变恰好每 20 帧一次是上下文断崖的指纹；同一局面反复重置
看球是否随机出现，是采样抖动的指纹）。这份清单同时是第 10 课的入场券，DIAMOND
的全部动机，就是用连续扩散救回这里死掉的小物体。

## 6. 源码导读

整个 src 目录不足 3000 行 Python，是全课程最适合通读的仓库。建议按下表顺序，每个
文件带着问题进去：

| 文件 | 是什么 | 带着什么问题读 |
|---|---|---|
| `src/main.py` | 训练入口 | 一共几行？hydra 装饰器指向哪个配置？|
| `config/trainer.yaml` | 总配置 | 找出这几个数各在哪：600 个 epoch、采集停在第 500 个 epoch、每 epoch 采 200 步、三个零件各自的开训 epoch（5/25/50） |
| `src/models/tokenizer/tokenizer.py` | 切词器 | argmin 在哪一行？straight-through 在哪一行？找不找得到 EMA 或死码重启？（答案：找不到） |
| `src/models/world_model.py` | token 世界模型 | 三个头各自的模式串长什么样？为什么观察头的模式串倒数第二位是 0？奖励标签为什么取 sign？|
| `src/models/slicer.py` | 模式切片工具 | `Embedder` 怎么让动作词和画面词各用各的嵌入表？|
| `src/models/transformer.py` | minGPT 式骨干 | `max_tokens` 怎么由 17 乘 20 算出来？因果掩码在哪？|
| `src/models/kv_caching.py` | 推理缓存 | 缓存按什么尺寸预分配？和 `max_tokens` 什么关系？|
| `src/envs/world_model_env.py` | 梦境引擎 | `step` 里为什么循环 17 次？奖励和终止是算出来的还是采样出来的？缓存满了那个 if 分支干了什么？|
| `src/models/actor_critic.py` | 策略与价值 | `imagine` 的 burn-in 喂的是原始帧还是重建帧？λ-return 在哪算？|
| `src/agent.py` | 三件套的壳 | `act` 方法里那次 `encode_decode` 是干什么的？（5.4 节"容易漏看的设计"） |
| `src/collector.py` | 数据采集 | 训练采集的 epsilon 是多少？测试采集的温度是多少？|
| `src/trainer.py` | 训练调度 | 每个 epoch 的顺序是什么？开局打印的三行参数量在哪里生成？|
| `src/play.py` 与 `src/game/keymap.py` | 体验入口 | 四种模式各对应 play.sh 的哪个旗标？键盘映射怎么按游戏的动作表过滤？|
| `results/data/` | 官方原始分数 | IRIS.json 里你的游戏有几个种子、各多少分？HUMAN.json 和 RANDOM.json 的基线呢？|

两处最容易读岔的地方提前点破。其一，`world_model_env.py` 的 `step` 里有个分支：
KV cache 攒满 340 个 token 时，用**当前帧的 16 个词**重开缓存，这是 README
那句"每 20 帧清空记忆"的代码本体，清空后世界只剩一帧的记忆，5.5 节第 2 类穿帮
的案发现场就在这四行。其二，`config/env/default.yaml` 里训练环境
`done_on_life_loss: True`（丢一条命就算一局，帮世界模型多见"终止"）而测试环境
是 False、上限 108000 帧，两套口径是 Atari 100k 的标准做法，你对比分数时别拿
训练日志里的"局"当测试的"局"。

## 7. 实验

前五个 Step 是体验与审计（一天内），后面是训练与复现（数天挂机）。照旧：先写预期，
再跑，再对照。

### Step 1: 克隆与环境

```bash
git clone https://github.com/eloialonso/iris.git
```

建一个 Python 3.9 或 3.10 的虚拟环境（老依赖在 3.11+ 上没验证过），先把两个打包
工具钉回旧版，这是安装 `gym==0.21.0` 的著名前置咒语，不念必炸：

```bash
pip install "setuptools==65.5.0" "wheel==0.38.4"
```

然后按 PyTorch 官网装好与你 CUDA 匹配的 torch 和 torchvision（README 注明代码
开发时用的是 torch 1.11，新一点的版本一般也能跑，出问题优先回退），最后：

```bash
pip install -r requirements.txt
```

依赖里的 `gym[accept-rom-license]` 会自动下载 Atari ROM，README 提醒过：装了
就等于你声明自己有使用这些 ROM 的许可。

### Step 2: 请一个预训练世界模型回家

预训练权重在 HuggingFace 的 `eloialonso/iris` 仓库 `pretrained_models/` 目录下，
26 个游戏各一个 `.pt`，每个约 127MB。下载 Breakout：

```bash
wget https://huggingface.co/eloialonso/iris/resolve/main/pretrained_models/Breakout.pt
```

按 README 的指引摆位：在仓库根目录建一个 `checkpoints` 目录，把权重放进去改名
`last.pt`（play 脚本按这个固定路径找模型）：

```bash
mkdir checkpoints
```

```bash
cp Breakout.pt checkpoints/last.pt
```

再告诉配置这是哪个游戏：打开 `config/env/default.yaml`，把 `train.id` 从 `null`
改成 `BreakoutNoFrameskip-v4`（测试环境和键盘映射都会跟着这个值走）。没有 N 卡的
机器顺手把 `config/trainer.yaml` 的 `common.device` 改成 `cpu`。

### Step 3: 先看 agent 打真环境（顺便看清"它眼中的世界"）

```bash
./scripts/play.sh -r
```

默认模式是 agent 在**真实环境**里打，`-r` 开三联屏：左边原始画面，中间缩放到
64×64 的输入，右边 tokenizer 的重建，README 原话，右边就是"agent 实际看到的
东西"（5.4 节讲过，策略吃的是重建帧）。预期：agent 打得有模有样（官方权重在
Breakout 上的水平是超过人类基线的）；盯着右边看几分钟，先在**真实**环境里认识
tokenizer 的画质，球在重建里是不是就已经时隐时现了？这个观察直接决定你后面把
穿帮归因给谁。

### Step 4: 进世界模型里面玩（本课的动机实验）

```bash
./scripts/play.sh -w
```

这是那个几分钟的魔法时刻：窗口里的每一帧都是 transformer 续写、解码器画出来
的，**没有游戏引擎在跑**。第一帧取自真实环境的重置画面，此后你按一个键，模型
就把"这个键的后果"编出来给你看。Breakout 的有效按键由动作表过滤而来：空格是
fire（发球），d 右移，a 左移，其余键无效。fps 默认 15，嫌快嫌慢用 `-f` 调。

带着任务玩至少 15 分钟：

1. 正常玩：发球、接球、消砖。体会一下"物理规律是学出来的"，球的反弹大体
   像回事，就是训练数据里 10 万步的统计规律在说话。
2. 找穿帮：按 5.5 节的三类病灶找证据。重点观察：球飞快时会不会突然消失或
   瞬移（词汇不够）；画面是否每隔约 20 帧轻微跳变一次，跳变后已消的砖有没有
   复活、比分有没有乱（上下文断崖，README 明说交互模式每 20 帧清一次记忆）；
   同一局面多玩几次，走向是否每次不同（采样，这是特性；但离谱的小概率画面是
   抖动，这是病）。
3. 录证据：按 `,` 键开始录制，再按一次停止，片段以 mp4 和 numpy 两种格式存进
   `media/recordings`（README 与 `src/game/game.py` 都写着）。每类穿帮至少录一段。

预期之外的一个提醒：奖励和终止也是模型采样出来的，你可能会在梦里"无缘无故
死掉"或者"没接住球却没死"，这不是 bug，是终止头的概率判断，同样值得录进报告。

### Step 5: 换 agent 进梦里玩给你看

```bash
./scripts/play.sh -a
```

`-a` 模式是 agent 在世界模型里自己玩（第 06 课想象训练的可视化版本：策略与世界
模型互相喂，你旁观）。预期：它玩得比你稳，毕竟它就是在这个梦里训出来的，梦的
统计规律它门儿清。对照你自己玩的体感想一个问题：有没有可能 agent 的高分部分来自
"钻梦的空子"？第 04 课的老病，在这里没有真环境立刻拆穿它，答案要等你训练自己
的模型后拿测试分数说话。

### Step 6: 码本使用率审计

给 tokenizer 做体检。在仓库根目录存一个胶水脚本 `audit_codebook.py`（本课唯一的
自写代码，其余全是官方脚本）：

```python
import sys
sys.path.append('src')

import torch
from hydra import initialize, compose
from hydra.utils import instantiate

from envs import make_atari
from utils import extract_state_dict

DEVICE = 'cuda:0'          # 没有 N 卡改成 'cpu'
GAME = 'BreakoutNoFrameskip-v4'
N_STEPS = 2000             # 采样帧数，CPU 跑就减到 500

with initialize(config_path='config'):
    cfg = compose(config_name='trainer')

# with_lpips=False：审计不算感知损失，跳过 LPIPS 的 VGG 权重；
# 因此 load 用 strict=False，忽略 checkpoint 里的 lpips.* 键
tokenizer = instantiate(cfg.tokenizer, with_lpips=False).to(DEVICE).eval()
state_dict = torch.load('checkpoints/last.pt', map_location=DEVICE)
tokenizer.load_state_dict(extract_state_dict(state_dict, 'tokenizer'), strict=False)

env = make_atari(GAME, size=64, max_episode_steps=None, noop_max=30,
                 frame_skip=4, done_on_life_loss=False, clip_reward=False)

counts = torch.zeros(tokenizer.vocab_size, dtype=torch.long)
obs = env.reset()
with torch.no_grad():
    for t in range(N_STEPS):
        frame = torch.tensor(obs, dtype=torch.float32).div(255)
        frame = frame.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        tokens = tokenizer.encode(frame, should_preprocess=True).tokens
        counts += torch.bincount(tokens.flatten().cpu(),
                                 minlength=tokenizer.vocab_size)
        obs, _, done, _ = env.step(env.action_space.sample())
        if done:
            obs = env.reset()

probs = counts.float() / counts.sum()
used = int((counts > 0).sum())
top10 = probs.sort(descending=True).values[:10].sum().item()
nonzero = probs[probs > 0]
ppl = torch.exp(-(nonzero * nonzero.log()).sum()).item()

print(f'码本大小           : {tokenizer.vocab_size}')
print(f'被用过的码字       : {used}（死码 {tokenizer.vocab_size - used} 个）')
print(f'前 10 个码字的占比 : {top10:.1%}')
print(f'使用分布困惑度     : {ppl:.1f}（完全均匀时应为 {tokenizer.vocab_size}）')
```

```bash
python audit_codebook.py
```

脚本做的事：随机策略在真实 Breakout 里跑 2000 步，每帧切成 16 个词，统计 512 个
码字的使用直方图，输出 5.2 节的三个体检数。预期：Breakout 画面元素极少（黑底、
几排砖、一块板、一颗球、一行比分），合理预期是使用高度集中、相当一部分码字长期
闲置；到底闲置多少，以你跑出的数字为准，死码过半就是教科书级的坍缩现场，意外
地均匀也如实记录，两种结果都是合格的审计。随机策略只逛得到游戏的开局区域，顺手
把 `env.action_space.sample()` 换成读取 agent 动作（或干脆把 N_STEPS 内的动作偏向
fire 和左右移）再跑一遍，看看"见识广一点"的帧会不会唤醒更多码字，这一步是
第 01 课"数据覆盖决定模型见识"的 token 版重演。

### Step 7: 启动你自己的 Atari 100k 训练

README 的原始命令只需要挑一个游戏名：

```bash
python src/main.py env.train.id=BreakoutNoFrameskip-v4 common.device=cuda:0 wandb.mode=online
```

不想用 wandb 就把最后一项换成 `wandb.mode=disabled`（日志默认同步到 wandb，README
写明可以这样关）。训练会在 `outputs/日期/时间/` 下新建自己的运行目录，和 Step 2
摆在仓库根的预训练权重互不干扰。开跑后按顺序核对四个预期：

1. 开局打印三行参数量（tokenizer、world_model、actor_critic 各多少参数），这是
   `trainer.py` 里写的冒烟自检，记进笔记；
2. 前 5 个 epoch 只采数据不训练，第 6 个 epoch 起 tokenizer 开训，第 26 个 epoch
   起世界模型加入，第 51 个 epoch 起 actor-critic 加入（配置里三个
   `start_after_epochs`：5、25、50），损失面板前期缺项是设计，不是坏了；
3. 每 5 个 epoch 跑一次测试采集（8 个并行环境、16 局、温度 0.5）；
4. `media/reconstructions/` 里定期落盘"原帧对重建帧"的拼图，
   `media/episodes/imagination/` 里落盘想象片段，训练期间每天看一眼这两个目录，
   比盯损失曲线有用。

账目再核一遍：采集在第 500 个 epoch 停止，每个 epoch 采 200 步，合计 10 万步真实
交互；`frame_skip: 4`，等于 40 万帧原始画面，这是 Atari 100k 协议的口径（10 万
个 agent 决策步，不是 10 万帧）。之后的第 501 到 600 个 epoch 只训练不采集。

### Step 8: 正式评测与官方对照

训练结束后进运行目录（`outputs/日期/时间/`），用官方评测脚本：

```bash
python ./scripts/eval.py
```

它会加载 `checkpoints/last.pt` 的 tokenizer 和 actor_critic，起 25 个并行环境收
100 局测试分（参数 `-n` 和 `-p` 可调）。一个小补丁：这个脚本把 `wandb.mode=online`
写死在命令串里，不用 wandb 的话打开 `scripts/eval.py` 把那一行的 `online` 改成
`disabled` 再跑。

对照官方：仓库 `results/data/IRIS.json` 存着论文的原始分数，Breakout 一栏是五个
种子的最终分 104.6、85.6、69.7、70.3、88.1（均值 83.7）；`HUMAN.json` 给人类基线
30.5，`RANDOM.json` 给随机基线 1.7。你的验收线：100 局均分显著高于随机、与官方
五种子区间（70 到 105 一带）同一量级；单种子落在区间外一截不算失败，系统性差一个
量级才需要去第 10 节排查。想复刻论文图表，`results/results_iris.ipynb` 里是官方
的作图代码（用的 rliable 统计法，第 17 课评测学会再见到它）。

### Step 9: 用自己的模型重走一遍体验与审计，写两份报告

把训练运行目录里的 `checkpoints/last.pt` 当作 Step 2 的权重（运行目录里就有
`scripts/play.sh` 的副本，直接在运行目录里执行同样的命令），重做 Step 4 的梦里
游玩和 Step 6 的码本审计。预期：你的 100k 模型比官方权重更容易穿帮（官方权重也是
100k 训练，但人家的种子和你不同，更重要的是你现在有了"找茬的眼睛"）；两份审计
数字放同一张表里比。

然后交两份作业：

1. 体验报告：至少 3 段录像，每段一行归因（三类病灶之一）加一行证据；外加一段
   总结，如果让你只修一个零件来减少穿帮，你修 tokenizer 还是 transformer？为什么？
2. 训练记录：老规矩 `NOTES.md` 五行起步，本课的样子：

```text
日期与机器、仓库 commit
命令：train 与 eval 的完整命令行（含游戏名与 device）
口径：100k 决策步 = 400k 帧；测试协议 25 环境 100 局
分数：我的 100 局均分 对 官方五种子 69.7-104.6 对 人类 30.5 对 随机 1.7
码本审计：死码数 / 前10占比 / 困惑度（预训练权重与我的权重各一行）
```

## 8. 配置与预算

| 档位 | 内容 | 预算（参考量级） | 用途 |
|---|---|---|---|
| 体验档 | Step 2-5：下载 127MB 权重，三种 play 模式 | 下载几分钟 + 游玩半小时起 | 本课动机实验，何时结束取决于你玩上瘾的程度 |
| 审计档 | Step 6：码本体检 | 分钟级，CPU 可跑 | 交付物之一，预训练与自训各跑一次 |
| 复现档 | Step 7-8：单游戏 100k 全程 600 epoch | 单卡数天（官方未给参考耗时），挂后台 | 论文复现 #4 |
| 加餐档 | 第 11 节改造清单 | 每条从半天到两天不等 | 选做 |

预算细账，都以配置文件为出处：训练三个零件的 batch 分别是 tokenizer 256、世界
模型 64、actor-critic 64（各自 `batch_num_samples`），24GB 卡放得下默认配置；
显存吃紧先调 tokenizer 的 batch 或加 `grad_acc_steps`。内存方面，回放数据整个住在
内存里，`config/datasets/default.yaml` 的 `max_ram_usage: 30G` 是上限，进程内存
超过 30G 后按先进先出丢最老的 episode；机器内存小就把这个值改小，代价是模型能
回看的历史变短。学习率三个零件统一 1e-4，`sequence_length` 等于世界模型窗口 20，
全部默认值不建议动，第一次复现的黄金法则从第 01 课起没变过：除了游戏名和
device，一个默认值都别碰。

换游戏的成本是零：26 个预训练权重覆盖的游戏名就是 HuggingFace 目录里的文件名，
训练时换 `env.train.id` 即可。想给第 10 课的对照实验省事，就选 DIAMOND 也有预训练
权重的游戏（第 10 课开场会给清单），同一游戏两课各训一份，并排对比才成立。

## 9. 验收

验收清单：

- [ ] 白纸画出全链数据流：64×64 帧切成 16 个词、加动作词拼成 17 词块、20 块进
      340 词窗口、三个头各自输出什么、16 个词怎么解码回帧，每个箭头标形状，
      词表大小 512 标在查表那一步；
- [ ] Step 3 的三联屏观察有记录：真实环境下 tokenizer 的重建里，球和砖的表现
      如何（这是穿帮归因的基线证据）；
- [ ] 至少 3 段穿帮录像躺在 `media/recordings`，每段配一行归因（词汇不够 / 上下文
      断崖 / 采样抖动）和一行证据；
- [ ] 码本审计跑过两份权重（官方预训练与你自己的），死码数、前 10 占比、困惑度
      六个数进了同一张表；
- [ ] 100k 训练跑完 600 个 epoch，eval 100 局均分显著高于随机基线 1.7、与官方五
      种子区间（69.7 到 104.6）同量级，同方向即验收，逐点对齐不要求；
- [ ] `NOTES.md` 五行齐全，能只看它复述实验；
- [ ] 能不看笔记回答三连问：观察头的模式串为什么倒数第二位是 0？梦里的奖励是算
      出来的还是采样出来的？MDN-RNN 和 IRIS 各用什么表达"未来有几种可能"？

## 10. 排错

| 症状 | 最可能的原因 | 怎么确认 | 怎么修 |
|---|---|---|---|
| 装 gym==0.21.0 时报错 | 新版打包工具不认老包的元数据 | 报错信息含 setup command 或 wheel 构建失败 | 先 `pip install "setuptools==65.5.0" "wheel==0.38.4"`，再装 requirements |
| 报 ROM 缺失或 ALE 相关 ImportError | ale-py 与 gym 版本不配套 | pip list 对照 requirements（ale-py 应为 0.7.4） | 按 requirements 的精确版本重装，别自行升级任何一个 |
| play.sh 报 pygame 显示相关错误 | 无屏幕环境 | 是否在 SSH 会话里 | 体验环节必须本地有屏机器；训练与审计不受影响 |
| play.sh 报找不到 checkpoint | 权重摆位不对 | 当前目录下有没有 `checkpoints/last.pt` | 预训练权重放仓库根的 checkpoints 下；自训模型在各自运行目录里跑 play |
| play 窗口开了但画面不动 | Breakout 还没发球 | 按空格后是否恢复 | 空格是 fire；另外训练环境丢一条命即一局，梦里"猝死"重开也正常 |
| 训练开局停在 wandb 提示 | 没登录却开着在线日志 | 终端卡在登录询问 | 命令行加 `wandb.mode=disabled`；运行目录 eval.py 里写死的 online 手改成 disabled |
| 前几十个 epoch 损失面板缺项 | 三个零件的开训时间不同 | 对照 trainer.yaml 的 5/25/50 | 设计如此，等 |
| 训练中途显存 OOM | tokenizer 的 batch 256 是显存大头 | 看崩溃时在训哪个零件 | 调低 `training.tokenizer.batch_num_samples` 或加 `grad_acc_steps`，改动记进 NOTES |
| 内存占用一路涨到 30G | 回放数据整个住内存，30G 是配置上限 | 对照 `max_ram_usage: 30G` | 正常；小内存机器调低该值，代价是可回放历史变短 |
| 梦里每隔固定间隔跳变、砖复活 | 20 帧记忆清空 | 数一数间隔是否约 20 帧 | 这是特性兼本课教材（5.5 节第 2 类穿帮），不用修；想验证机制就做第 11 节改造 2 |
| 审计脚本打印缺键警告 | strict=False 忽略了 lpips 的权重 | 警告键名是否都带 lpips 前缀 | 无害，审计用不到感知损失；其他键名出现在警告里才要警惕 |
| eval 均分远低于官方区间 | 动过默认值、训练没跑满、或游戏名不一致 | diff 运行目录里的 config 和仓库默认；确认跑满 600 epoch | 回到全默认只改游戏名和 device 重跑；单种子波动大，差一截先别慌，差量级才排查 |

## 11. 前沿与改造

"世界变成 token、未来变成下一个 token"在 2023 年就不止 IRIS
一家：同届 ICLR 的 TWM（arXiv:2303.07109）用 Transformer-XL 当骨干，token 不走
VQ 图像词，改用压缩隐状态、动作、奖励三种模态混排，专门缓解本课"20 帧断崖"这类
上下文问题。你在第 06 课拆过的 DreamerV3 其实也用离散潜变量（32 组分类分布），
只是没把它们当语言交给 transformer 续写，离散化和序列化是两个独立的决定，IRIS
把两个都选了。往 2024 年之后看，这条路线的放大版就是第 11 课的 Genie 路线：
时空 token、更大的词表、从无动作标签的海量视频里连"动作"都自己挖；1x 公司的
人形机器人数据集干脆直接发预先 token 化的版本。token 路线最大的红利始终是那条：
语言模型攒了十年的基础设施与缩放经验直接继承。

规模一半：512 词的码本、16 词一帧、20 帧窗口，每个数字放大十倍
就是前沿系统的量级（钱能解决）。机制一半：离散化对小物体的系统性歧视、逐词采样
的误差累积，放大词表只能缓解、不能根治，第 10 课的扩散路线就是冲这两条来的
（本课教的审计和归因手艺，到那边直接复用）。

动手改造清单（选做，各自写清预算与失败判据）：

1. 码本减到 64：把 `config/tokenizer/default.yaml` 的 `vocab_size` 从 512 改成
   64，训练前 100 个 epoch（三个零件都已开训），对比 `media/reconstructions/` 的
   重建质量并重跑审计。预算：半天到一天。预期：重建变糊、球更常丢；死码比例反而
   下降（词表小了用得满）。失败判据：重建肉眼无差别，说明 Breakout 根本用不满
   64 个词，把结论如实记下，换画面更复杂的游戏再试。
2. 窗口减到 10：把 `config/world_model/default.yaml` 的 `max_blocks` 从 20 改
   成 10（`sequence_length` 和想象长度会通过配置插值自动跟随），短跑训练后进 `-w`
   模式玩。预算：半天到一天。预期：记忆清空的跳变周期从约 20 帧变约 10 帧，长时
   一致性更差。失败判据：体感无差别，Breakout 的状态几乎当前帧可见（球速要两帧），
   对窗口不敏感本身就是一个值得记录的结论。
3. 给 tokenizer 补一味药：在 `tokenizer.py` 里加死码重启，训练中维护每个
   码字的使用计数，每隔固定步数把长期零使用的码字重置为当前批次里随机一个编码器
   输出。预算：改码半天，短跑一天，前后各审计一次。预期：死码数下降、困惑度上升；
   重建和分数的变化如实记录。失败判据：困惑度不动（重启完又饿死，检查重启频率）。
4. 给梦装温度旋钮：`world_model_env.py` 的 `step` 里，三处 `Categorical(logits=...)`
   改成除以温度 τ，`-w` 模式下 τ 取 0.5 和 1.5 各玩十分钟。预算：几行代码加游玩
   时间。预期与第 04 课同方向：低温的梦更死板保守，高温的梦更疯、穿帮更密。这是
   把第 04 课在 MDN 上做过的温度实验原样搬到 softmax 上，搬完你对"τ 是采样分布
   的通用旋钮、与模型家族无关"就有了两份证据。

论文招牌论断"transformer 世界模型是样本高效的"落到 26 个游戏是
HNS 1.046、10 个游戏超人类；你的单游戏切片是：Breakout 上 100k 步训出的分数超过
人类基线 30.5。达到了，你就用二十六分之一的预算摸到了论文结论的方向；没达到但
显著超过随机基线，也算方向性支持，把差距写进报告。

## 12. 论文与延伸

1. Transformers are Sample-Efficient World Models（Micheli, Alonso & Fleuret,
   ICLR 2023 notable top-5%，[arXiv:2209.00588](https://arxiv.org/abs/2209.00588)）
   ，本课主论文。带着三个问题读：actor-critic 为什么吃重建像素而不是 token 或
   隐向量，论文给了什么理由（或者压根没解释）？结果表里 IRIS 分数最难看的那几个
   游戏，画面上有什么共同点，和你的穿帮清单对得上吗？它与同表的 SimPLe、
   EfficientZero 各自"模型"的用法有何不同（回忆第 07、08 课的坐标系）？
2. Neural Discrete Representation Learning（van den Oord, Vinyals &
   Kavukcuoglu, 2017，[arXiv:1711.00937](https://arxiv.org/abs/1711.00937)），
   VQ-VAE 原典，5.1 节全部数学的出处。带着两个问题读：commitment 项的 β 两个
   方向各防什么病，IRIS 取 β=1 而论文默认 0.25，这个自由度说明什么？论文声称
   离散潜变量天然避开后验塌缩（第 02 课那味病），论证在哪一段？顺手找到 EMA
   码字更新写在哪里，那就是 IRIS 没吃的那味药的原始药方。
3. Transformer-based World Models Are Happy With 100k Interactions（Robine
   et al., ICLR 2023，[arXiv:2303.07109](https://arxiv.org/abs/2303.07109)），
   同届同题的另一条路。带着两个问题读：它的 token 由哪三种模态构成、和 IRIS 的
   纯图像词各有什么代价？Transformer-XL 的段级递归怎么缓解"窗口断崖"，对照
   你在梦里数出来的 20 帧跳变。
4. **Estimating or Propagating Gradients Through Stochastic Neurons for
   Conditional Computation**（Bengio, Léonard & Courville, 2013，
   [arXiv:1308.3432](https://arxiv.org/abs/1308.3432)），选读，straight-through
   估计器的出处。带着一个问题读：这个估计器明明是有偏的，为什么十年来从 VQ-VAE
   到各家 tokenizer 都在用（论文自己给的辩护是什么）？
5. 仓库配套资产，README（play 各模式与权重摆位的第一手说明）、
   `results/results_iris.ipynb` 加 `results/data/`（论文全部原始分数，你的验收
   标尺）、credits 一栏（transformer 师承 karpathy/minGPT，tokenizer 师承
   CompVis/taming-transformers）。带着问题翻：对照 minGPT 源码，`transformer.py`
   为交互式推理加了什么（提示：KV cache 不是 minGPT 自带的）。

第 10 课请出与 IRIS 同作者的 DIAMOND：把"查词表"换成连续扩散，画面
不再经过 512 个词的窄门。你在本课录下的那颗消失的球，就是下一课开庭时的第一件
证物，同游戏、同预算、同一双找茬的眼睛，token 对扩散，正面对决。Mac 用户有
彩蛋：DIAMOND 的预训练世界模型在 MPS 上就能玩。
