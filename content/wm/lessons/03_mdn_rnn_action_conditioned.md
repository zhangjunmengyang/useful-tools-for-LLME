---
id: 03_mdn_rnn_action_conditioned
title: "让 M 按动作预测下一步"
summary: "为什么预测未来要输出一团分布，而不是一个点？动作到底有没有被模型用起来？"
unit: vmc
play_tools: []
checkpoints:
  - "一个动作条件的动力学模型（MDN-RNN）。"
  - "动作对换对照报告：证明动作真的改变了预测。"
  - "一条多步 rollout 的误差漂移曲线，用于观察误差如何累积。"
---

# 第 03 课：让 M 按动作预测下一步

> 类型：复现（World Models 复现的第三棒，01-04 课合力完成）<br>
> 建议周期：2-3 天<br>
> 硬件：单张 24GB 显卡数小时；Mac/纯 CPU 也能完成训练与全部探针实验，只是更慢<br>
> 锚定仓库：[ctallec/world-models](https://github.com/ctallec/world-models)（PyTorch 复现），对照精读官方 [hardmaru/WorldModelsExperiments](https://github.com/hardmaru/WorldModelsExperiments)<br>
> 产物：训练好的动作条件动力学模型（MDN-RNN）、动作对换对照报告、多步漂移曲线

## 1. 这一课做什么

VAE 已经把一帧赛道压成 32 个数，但系统还不会预测。现在加入 M：一个 LSTM 依次读取
$(z_t, a_t)$，每一步给出 $z_{t+1}$ 的分布。它预测的不是笼统的下一帧，而是“执行这个
动作之后”的下一帧；同一状态下，方向盘左打和右打应当产生不同结果。

这里有两个需要单独验证的问题。第一个是：**为什么预测未来要输出分布，而不是一个点？**
出了这条直道，赛道
可能左拐也可能右拐（CarRacing 的赛道是随机生成的）。如果强迫模型只报一个点，它会
报出"左拐和右拐的平均"，一条笔直插进草地、现实中永远不会出现的鬼影路。处理这种
"未来有好几种可能"的场面，需要让模型一次报出好几个带权重的候选，这是 MDN
（混合密度网络）。

第二个是：**动作到底有没有被模型用起来？** CarRacing 里车
有惯性，下一帧和这一帧长得很像，一个完全无视动作的模型光靠"惯性外推"也能把平均
误差做得很好。因此损失下降并不能证明模型使用了动作。接下来会做动作对换实验：固定
状态和历史，只替换输入动作，观察预测是否分岔。不分岔的模型就是“动作盲”。

还要比较两种 rollout：每一步喂真实历史，与持续把自身预测接回输入。前者衡量单步拟合，
后者暴露误差如何累积。第 04 课会直接在这个模型生成的轨迹里训练 controller，所以
`action_swap.png` 和 `drift_curve.png` 不是附加可视化，而是进入下一阶段前的验收项。

术语速查：

| 术语 | 一句人话 |
|---|---|
| M / MDN-RNN | 本课主角：LSTM 负责记忆，MDN 负责把预测说成一团分布，合起来按动作预测下一个 $z$ |
| MDN（混合密度网络） | 网络不直接输出预测值，输出的是一个混合高斯分布的全部参数：几个候选中心、各自的把握范围、各自的权重 |
| 混合高斯 | 好几个高斯分布加权叠起来的分布，能表达"未来有几种可能"这种多峰的形状 |
| 负对数似然（NLL） | 给"猜分布"打分的标准办法：你给真实答案分配的概率越高，罚分越低 |
| 隐状态 $h_t$ | LSTM 肚子里那 256 个数，浓缩了到目前为止的历史；下一课 controller 要拿它当一半的输入 |
| teacher forcing | 训练时每一步都喂真实历史，模型只需负责"下一步"；像做题永远对着标准答案的前几步 |
| exposure bias | teacher forcing 的后遗症：上场后没人喂标准答案了，模型吃着自己的错误输出越走越偏 |
| 自由 rollout | 让模型拿自己的预测当输入接着往下预测，滚出一整段想象的未来；第 04 课的"梦"就是它 |
| 动作盲 | 模型表面上接收动作输入，实际上预测根本不随动作变化；平均误差看不出来，动作对换一测一个准 |
| log-sum-exp | 算混合分布对数概率时防数值下溢的标准手法，`gmm_loss` 里就有教科书式的一份 |
| reward/done 头 | ctallec 版 M 顺手多预测的两样：这一步的奖励、这一局结不结束；做全套梦境时缺不了 |

## 2. 问题

1. 把"预测下一个 $z$"从单点升级成分布。上一课 VAE 的重建损失是像素级 MSE，
   那是因为"这一帧长什么样"没有歧义；"下一帧长什么样"有歧义，同样的现在可以
   通向不同的未来。需要搞清楚：为什么 MSE 在这里在数学上就等于一个注定失败的
   假设，混合高斯怎么修，负对数似然怎么给分布打分。
2. 验证动作真的进了模型。搭一个可复用的审讯工具：加载训好的 M，同一状态配
   不同动作做前向对比，把"动作有没有被用起来"从感觉变成一个数。这个工具是你的
   胶水代码，接下来会完整写出来。
3. 量出"一步准"和"多步准"之间的鸿沟。训练损失度量的是 teacher forcing 下的
   一步预测；第 04 课用的是几十步的自由 rollout。两者之间隔着 exposure bias。本课
   把两条误差曲线画在同一张图上，让这道鸿沟有具体的宽度。

一个要先划清的界限：M 只在 latent 空间干活，输入输出都是 32 维的 $z$，全程不碰
像素，想看图，就把 $z$ 交给上一课 VAE 的解码器。所以本课所有评价都在 $z$ 空间
里量距离，解码出的图像只当肉眼佐证。这个设计是论文的原意：预测压缩后的世界，
比预测每个像素便宜得多。

## 3. 准备

- 第 01 课的环境和仓库（还是那个独立虚拟环境），以及 `exp_dir/vae/best.tar`，
  上一课训好并验收过的 VAE。VAE 的质量是 M 的天花板：$z$ 本身糊，预测 $z$ 再准
  也没用。上一课的 latent 走查报告这时候就是体检证明。
- 数据要补足到 1000 条轨迹。这里埋着本仓库最阴的一个坑：`data/loaders.py`
  固定把文件列表的**最后 600 个**划为测试集（源码就是 `self._files[:-600]` 和
  `self._files[-600:]`，写死的数字）。如果你第 01 课只采了 100 条冒烟数据，训练集
  会是空的，训练脚本要么除零要么一轮都没真正训。第 7 节 Step 1 教你体检和补采。
- 磁盘再留出几个 GB（1000 条轨迹的量级），显卡随意，M 本体只有约 38 万参数，
  显存压力可以忽略，时间主要花在 VAE 在线编码和数据读取上。
- 胶水代码要画图，环境里装好 matplotlib（第 7 节 Step 0 处理）。
- 如果你跳过了第 02 课直接来的：回去至少把 VAE 训出来并看一眼重建质量，M 这一课
  没有 VAE 寸步难行。

## 4. 学习目标

1. 白纸上写出 M 的输入输出：吃 $(a_t, z_t)$ 和隐状态，吐一个 5 分量混合高斯的全部
   参数外加 reward、done 两个标量头，并说出输出层为什么恰好是 327 维；
2. 用"路口的均值鬼影"讲清楚单高斯回归在多峰问题上怎么死的，MDN 怎么救；
3. 看懂 `gmm_loss` 里的每一行，包括那个 log-sum-exp 数值技巧，并解释为什么这个
   损失出现负值不用慌；
4. 说出 teacher forcing 在训练代码的哪一行发生、exposure bias 会在第 04 课以什么
   形式讨债；
5. 独立设计并跑通动作对换对照实验，给任何一个号称"动作条件"的动力学模型出体检
   报告；
6. 解释为什么平均预测误差查不出动作盲，以及什么样的实验才查得出。

## 5. 原理

五个机制，每个按同一节奏走：为什么需要（直觉）、怎么运转（机制）、精确定义
（数学）、在源码哪里（代码）、怎么证明做对了（验证）。

### 5.1 为什么输出一团分布：路口的均值鬼影

想象你在直道尽头，前方赛道被弯道挡住视线。下一秒的画面有两种可能：
路向左弯，或者路向右弯，五五开。现在强迫你只报一个预测，而且按"预测和真相的
平方距离"罚你，最优策略是什么？报两种可能的平均：一条不左不右、笔直插进草地
中央的路。这个预测把罚分最小化了，但它描述的画面**永远不会发生**。这是均值
鬼影：对多峰的未来做单点回归，最优解落在几个峰之间的无人区。世界模型要是用鬼影
当梦境，车手会学着在一条不存在的路上开车。

出路是让模型别报一个点，报一整个分布，而且是能长出多个峰的分布。MDN
的做法：网络最后一层不输出预测值本身，输出一个混合高斯分布的全部参数，$K$ 个
分量，每个分量有一个中心 $\mu_k$（一个候选未来）、一组标准差 $\sigma_k$（这个候选
的模糊程度）、一个权重 $\pi_k$（押这个候选的概率）。路口场景下，理想的 MDN 输出
两个高权重分量：一个中心在"左弯画面"的 $z$，一个中心在"右弯画面"的 $z$。谁也
不用替谁平均。

M 对下一个 latent 的预测是条件混合高斯：

$$
p(z_{t+1} \mid a_t, z_t, h_t) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}\!\left(z_{t+1};\, \mu_k,\, \mathrm{diag}(\sigma_k^2)\right)
$$

其中 $\pi_k, \mu_k, \sigma_k$ 都是 LSTM 隐状态的函数（所以随历史和动作变化），
协方差取对角，每个 latent 维度独立。本仓库 $K=5$，latent 32 维，于是输出头要吐
$5 \times 32$ 个均值、$5 \times 32$ 个标准差、$5$ 个权重，再加 reward 和 done 两个
标量（5.5 节讲），合计 $(2 \times 32 + 1) \times 5 + 2 = 327$ 维。两个约束用老办法
保证：$\sigma$ 要为正，网络输出先过 $\exp$；$\pi$ 要归一化成概率，输出过 softmax
（代码里存的是 log softmax，配合后面的对数似然）。

`models/mdrnn.py` 里的 `_MDRNNBase`：一行
`nn.Linear(hiddens, (2 * latents + 1) * gaussians + 2)` 就是上面那笔账；
`MDRNN.forward` 里把这 327 维切成 `mus`、`sigmas`（过 `torch.exp`）、`pi`（过
`f.log_softmax`）和最后两维 `rs`、`ds`。训练脚本 `trainmdrnn.py` 里
`MDRNN(LSIZE, ASIZE, RSIZE, 5)` 那一句敲定 $K=5$。

两个办法。粗的：第 11 节的改造实验把 $K$ 砍成 1 重训，看损失和解码画面
怎么退化。细的：训好后挑一个出弯瞬间，分别解码"混合分布的总均值"和"权重最大
的那个分量的均值"，前者会比后者糊，因为前者把几个候选未来平均了，你等于
看一次鬼影是怎么调出来的。

### 5.2 负对数似然：给"猜分布"打分

模型现在交上来的作业是一团分布，怎么判分？原则很朴素：真相揭晓后，看
你事先给真相分配了多大概率。给得高，说明你的分布押对了地方；给得低，说明你的
概率质量堆错了位置。把"分配给真相的概率"取对数再取负，就是负对数似然：越小
越好，而且它同时惩罚两种失职，押错中心（$\mu$ 偏了）和瞎报把握（$\sigma$ 太大
太小都吃亏）。

训练时每个时间步都这么判：拿真实的下一个 latent $z_{t+1}$，代入模型
输出的混合高斯密度函数，得到一个密度值，取负对数，对时间和 batch 求平均。这里有
个数值上的坑：32 维高斯的密度是 32 个一维密度连乘，动辄小到浮点数直接归零，取
对数变成负无穷。标准解法是全程在对数域干活，求和时用 log-sum-exp 技巧：先把每个
分量的对数概率算出来，减掉当中的最大值再取指数求和，最后把最大值加回去。平移
不改变结果，却保证了指数运算不下溢。

单步损失：

$$
\mathcal{L}_{\mathrm{gmm}} = -\log \sum_{k=1}^{K} \exp\!\left( \log \pi_k + \sum_{d=1}^{32} \log \mathcal{N}(z_{t+1,d};\, \mu_{k,d},\, \sigma_{k,d}) \right)
$$

顺手看清两件事。第一，如果 $K=1$ 且 $\sigma$ 固定为常数，这个式子退化成 MSE 加
常数，所以"用 MSE 训预测"等价于"假设未来是单峰等宽高斯"，5.1 节的鬼影正是
这个假设的报应。第二，这是**连续分布的密度**，密度可以大于 1，所以损失完全可以
是负数；训练日志里看到负的 loss 别慌，接着降就是好事。

`models/mdrnn.py::gmm_loss`：`Normal(mus, sigmas).log_prob(batch)`
算逐维对数密度，`logpi + torch.sum(..., dim=-1)` 合成每个分量的对数概率，
`max_log_probs` 那三行就是 log-sum-exp。注意函数注释里专门说了：损失没有沿特征维
求平均，所以它的量级和 latent 维数大致成正比。这解释了 `trainmdrnn.py::get_loss`
里那个奇怪的除法：总损失是 gmm 损失加 done 头的 BCE（可选再加 reward 头的 MSE），
除以 `LSIZE + 1`（带 reward 时除以 `LSIZE + 2`），把"32 维的 gmm 损失"和
"1 维的 BCE"放回同一个量级，不然后者的梯度会被前者淹没。

看 `trainmdrnn.py` 跑出来的 train/test loss 同步下降即可入门；更硬的
验证在 Step 5：一步预测的 $z$ 空间误差要明显小于"直接拿 $z_t$ 冒充 $z_{t+1}$"
的偷懒基线，否则模型只学会了惯性。

### 5.3 动作条件与"动作盲"陷阱：平均误差是钝器，动作对换是利器

领航员的价值在于"你左打会怎样、右打会怎样"是两份不同的预报。但请
注意一个不舒服的事实：CarRacing 这种世界惯性很大，单帧之内车动不了多少，画面
连续性极强。一个彻底无视动作输入的模型，靠"下一帧约等于这一帧再顺着惯性挪一点"
就能把平均预测误差刷得相当体面。动作对下一帧的影响，在总误差里只占一小块；而
损失函数算的恰恰是平均。所以**平均误差是钝器**：它敲不出"模型究竟听没听方向盘"
这个决定生死的区别。要用利器：把其他所有条件钉死，只对换动作，看输出变不变。
变，说明动作在模型的计算里有实权；不变，动作输入就是个摆设。

动作进入模型的路径很短：每一步把动作向量和 latent 拼接成一个 35 维
输入（动作 3 维在前，$z$ 32 维在后），一起喂给 LSTM。就这一处。没有别的旁路。
所以动作盲不盲，全看训练有没有迫使 LSTM 的输入权重给那 3 维分配实权。对换实验的
设计因此很纯粹：同一个 $z_t$、同一份隐状态（同一段历史），只换动作向量，前向一
次，比较输出。因为整个前向是确定性函数，输出有差异只可能来自动作。

还有个实验设计细节：一步之内动作的影响本来就小（惯性物理决定的），所以除了比
一步预测，还要让每个动作**持续按住往前滚**若干步，左打满 15 步和右打满 15 步，
两个想象里的世界应该越离越远。一步分岔看灵敏度，多步分岔看动作是否真的在改写
未来。

给分岔一个量纲无关的读数。取探针动作集合 $\{a^{(i)}\}$（左打满、右打
满、全油门、急刹），从同一状态出发各自滚 $n$ 步，记 $\hat{z}^{(i)}_n$ 为动作
$a^{(i)}$ 分支第 $n$ 步预测的混合分布均值。分岔指数定义为两两距离对"真实世界
一步位移"的比值：

$$
D_n = \frac{\mathrm{mean}_{i \neq j} \, \lVert \hat{z}^{(i)}_n - \hat{z}^{(j)}_n \rVert_2}{\mathrm{mean}_t \, \lVert z_{t+1} - z_t \rVert_2}
$$

分母是这条轨迹上真实相邻帧的平均 $z$ 位移，充当比例尺。动作盲模型的 $D_n$ 贴着
0；健康模型的 $D_n$ 随 $n$ 增长，多步之后应与 1 同量级或更大（意思是：换个动作
造成的未来差异，赶上或超过世界自己一步的变化量）。

拼接发生在 `models/mdrnn.py::MDRNN.forward` 的
`torch.cat([actions, latents], dim=-1)`（单步版 `MDRNNCell.forward` 里是
`torch.cat([action, latent], dim=1)`）。另一个决定成败的位置在数据侧：
`data/loaders.py::RolloutSequenceDataset` 把观察切成 `[i, i+seq_len)`、动作却切成
`[i+1, i+seq_len+1)`，因为采数据时先取动作再记录执行后的画面，存盘数组里
`actions[t+1]` 才是作用在 `observations[t]` 上的那个动作。自己写数据管道时在这里
错一位，模型看到的就是"上一步的动作"，动作条件直接废掉一半。本课胶水代码会
沿用同样的对齐方式。

就是重点 Step 4。附带一个内建的对照：同一个动作喂两遍，距离必须
恰好为 0（前向是确定性的），这一格不为 0 说明你的探针代码本身有随机性没锁住。

### 5.4 teacher forcing 与 exposure bias：一步错，步步错

训练时模型过的是好日子：预测第 $t+1$ 步时，输入的 $z_t$ 永远来自真实
录像，像做数学题时每一步都先看一眼标准答案的前几行再写下一行。这叫 teacher
forcing。它训练快、稳定，是序列模型的标准做法。但上了考场（第 04 课的梦境、或者
任何多步预测），没有标准答案可抄了：模型第 1 步的输出带一点误差，第 2 步只能吃
着这个带误差的输入继续预测，误差再叠一层……输入离训练时见过的分布越来越远，
而模型从没学过怎么在自己制造的偏差里自救。这是 exposure bias：训练时永远暴露
在真实数据下，使用时却暴露在自己的错误下。

误差滚雪球有两个通道。一是显式输入：预测的 $\hat{z}$ 代替真实 $z$ 喂回
模型；二是隐状态：LSTM 的 $h$ 是沿着被污染的输入序列累积的，记忆本身也在变质。
两个通道互相喂养，所以自由 rollout 的误差增长常常快于"每步误差简单相加"的
顺带说清一件容易混淆的事：本仓库训练时每个 batch 还会给 $z$ 加上 VAE 后验
的采样噪声（5.5 节讲），那是输入端的独立噪声，和 exposure bias 这种"自己的系统性
误差被喂回来"的复利效应两码事，后者才是多步预测的主要杀手。

不需要新公式，需要一个实验定义。teacher forcing 误差曲线：对每个起点
$t$，用真实 $(z_t, h_t)$ 预测一步，记 $\lVert \hat{z}_{t+1} - z_{t+1} \rVert$，它
衡量"每步都对答案"时的水平，随预测时刻基本走平。自由 rollout 误差曲线：从 $t_0$
出发，此后每步吃自己的预测（动作仍用真实录像里的动作，保证两条曲线只差在状态
来源），记第 $k$ 步预测与真实 $z_{t_0+k}$ 的距离，看它随 $k$ 怎么长。两条曲线在
$k=1$ 处必须重合（输入完全相同），之后的裂口就是 exposure bias 的定量画像。

teacher forcing 藏在 `trainmdrnn.py::get_loss` 的结构里：`latent_obs`
整段来自真实观察过 VAE，模型一次前向吃满整个真实序列，损失逐步对着真实的
`latent_next_obs` 算。整个仓库的训练流程里没有任何自由 rollout，这正是第 04 课
翻车的伏笔。训练序列长度 `SEQ_LEN = 32` 也值得记住：Step 5 我们故意把 rollout 滚
到 40 步，越过训练视野看看会怎样。

Step 5 的两条曲线。另一个便宜的心理准备：这个病没有本课能吃的特效药，
第 05 课的 RSSM 和第 06 课"整个 actor-critic 都在想象里训"是两种缓解思路，
第 11 节先给你指个方向。

### 5.5 M 的完整输出：reward/done 头，以及 z 从哪来

到目前为止我们只说 M 预测下一个 $z$。但要在梦里训练策略，梦境还得会
发工资、会喊停：每一步的 reward、这一局是否结束（done）。不然 controller 在梦里
永远拿不到分数、也永远开不完一局。原论文里 CarRacing 的 M 只预测 $z$（分数用真实
环境算），到 VizDoom 梦境实验才加了 done 预测；ctallec 的实现干脆统一：M 的输出
头常备 reward 和 done 两个标量位。

327 维输出的最后两维：倒数第二维是 reward 的直接预测（训练用 MSE，
默认**不参与训练**，要开 `--include_reward` 开关才计入损失），最后一维是 done 的
logit（训练用带 logits 的 BCE，始终计入）。另一件必须如实交代的事是 $z$ 的来源：
`trainmdrnn.py` **不预计算** latent 序列：每个 batch 现场把图像过一遍冻结的
VAE（`to_latent` 函数），而且喂给 M 的 $z$ 还带着后验采样噪声，按
$z = \mu + \sigma \epsilon$ 现场抽一次，抽到什么用什么。这和官方 hardmaru 实现"先跑 series.py 把全部 z 存盘"的做法相反。在线
编码多花计算，换来两个好处：每个 epoch 看到的 $z$ 都带一点新鲜的后验噪声，等于
免费的数据增强、也逼着 M 对 $z$ 的小抖动稳健；此外换一个 VAE 不用重新生成数据集。

总损失（`get_loss`）：

$$
\mathcal{L} = \frac{\mathcal{L}_{\mathrm{gmm}} + \mathrm{BCE}(d_{\mathrm{logit}}, \mathrm{terminal}) + \mathbb{1}[\texttt{include\_reward}] \cdot \mathrm{MSE}(r, \mathrm{reward})}{\mathrm{LSIZE} + 1 + \mathbb{1}[\texttt{include\_reward}]}
$$

`models/mdrnn.py` 里 `rs = gmm_outs[:, :, -2]`、
`ds = gmm_outs[:, :, -1]`；`trainmdrnn.py::to_latent`（在线编码与采样）和
`get_loss`（三项损失与除法）。

CarRacing 的 done 几乎总在最后一步才是 1，正样本极稀，BCE 很快降到
很小的数，别把它当成"模型学会了预测终局"，那只是学会了报 0。这个头真正的
用武之地在第 04 课的 Doom 式梦境；本课知道它在哪、别被它的损失数字迷惑即可。

## 6. 源码导读

还是那个仓库，这次只有四个文件是主战场，每个带着问题进去：

| 文件 | 管什么 | 带着什么问题读 |
|---|---|---|
| `models/mdrnn.py` | M 的本体 | `_MDRNNBase`、`MDRNN`、`MDRNNCell` 三个类各自存在的理由是什么？327 维输出怎么切成五份？`gmm_loss` 的 log-sum-exp 是哪三行？|
| `trainmdrnn.py` | M 的训练 | `to_latent` 里 $z$ 是均值还是采样？`get_loss` 为什么除以 `LSIZE + 1`？`include_reward` 开关改变了什么？|
| `data/loaders.py` | 序列数据 | 动作为什么从 `seq_index + 1` 开始切？最后 600 个文件去哪了？|
| `utils/misc.py` | 常量与示范 | `LSIZE, ASIZE, RSIZE` 是多少？`RolloutGenerator` 加载 `MDRNNCell` 时那句 `strip('_l0')` 在干什么？|

`MDRNN` 和 `MDRNNCell` 的分工要专门说透，因为胶水代码全靠它：两个类共享同一个
输出头（都继承 `_MDRNNBase`），但 `MDRNN` 内置 `nn.LSTM`，吃整段序列
（`forward(actions, latents)`，序列维在前），训练用它，一次前向算完 32 步，快；
`MDRNNCell` 内置 `nn.LSTMCell`，一次走一步（`forward(action, latent, hidden)`，
显式传入传出隐状态），交互式 rollout 用它，第 04 课在梦里开车、本课的探针，
都得一步一步喂。麻烦在于 PyTorch 给两者的参数起名不同：`nn.LSTM` 的权重叫
`weight_ih_l0`，`nn.LSTMCell` 的叫 `weight_ih`。训练存下的是 `MDRNN` 的
checkpoint，加载进 `MDRNNCell` 前得把键名里的 `_l0` 后缀去掉。仓库
`utils/misc.py::RolloutGenerator` 的写法是
`{k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()}`，能用，但注意
`str.strip` 去的是"字符集合"而非子串，恰好这里所有键名都安全；换了自己的模型
别盲抄这句。

## 7. 实验

先训练，后审讯。每一步先写预期，再跑，再对照。

### Step 0: 补装画图依赖

```bash
pip install matplotlib
```

装在第 01 课那个虚拟环境里。仓库本身不依赖它，我们的探针要画图。

### Step 1: 数据体检与补采

先数一数手里有多少条轨迹（按 loaders.py 同样的"一层子目录"方式数）：

```bash
python -c "import glob; print(len(glob.glob('datasets/carracing/*/*.npz')))"
```

预期：至少 1000。少于 1000（尤其第 01 课只采了 100 条冒烟数据的），补采：

```bash
python data/generation_script.py --rollouts 1000 --threads 8 --rootdir datasets/carracing
```

这是 README 的正式配置，多核机器一两个小时的量级；服务器没有显示环境的话，套上
第 01 课 Step 5 用过的 `xvfb-run` 前缀。为什么必须补：第 3 节说过，loaders.py 把
文件列表的最后 600 个划为测试集，1000 条数据实际上是 400 训 / 600 测，比例怪，
但这是仓库现状，如实接受。然后确认 VAE 在位：

```bash
ls exp_dir/vae/best.tar
```

数据补充过之后，建议回头用新数据把 VAE 重训一遍再进下一步（第 02 课的流程原样
再走一次），让 V 和 M 吃的是同一锅饭。

### Step 2: 训练 MDN-RNN

```bash
python trainmdrnn.py --logdir exp_dir
```

它会从 `exp_dir/vae/best.tar` 加载冻结的 VAE（没有就直接 assert 失败），然后训
30 个 epoch：每个 epoch 载入 30 条轨迹的缓冲，切成 32 步的序列，batch 16，RMSprop
学习率 1e-3，外加"测试损失 5 个 epoch 不降就减半学习率"的调度。预期：train 和
test 的 loss 稳定下降，出现负值属正常（5.2 节）；`exp_dir/mdrnn/` 下长出
`best.tar`（测试损失最优）和 `checkpoint.tar`（最新）。单卡量级是数小时，时间
大头不在这 38 万参数的模型上，而在每个 batch 现场过 VAE 编码和轨迹文件 IO 上；
Mac/CPU 也能跑，预期慢数倍。

一个提前存钱的选项：第 04 课的梦境训练需要 M 会发工资（预测 reward）。现在就用
`python trainmdrnn.py --logdir exp_dir --include_reward` 训练，能省下届时的一次
重训。本课后续步骤对两种训法都适用。

### Step 3: 写胶水代码：把审讯工具装订成册

在仓库根目录新建 `probe_mdrnn.py`，全文如下，可直接运行。它做两件事：动作对换
（`--probe swap`）和漂移曲线（`--probe drift`）。模型加载方式、常量、动作对齐
全部沿用仓库自己的写法，出处在注释里。

```python
"""probe_mdrnn.py ， 第 03 课胶水代码：动作对换 + 多步漂移探针。

放在 ctallec/world-models 仓库根目录运行，
依赖已训好的 logdir/vae/best.tar 与 logdir/mdrnn/best.tar。
"""
import argparse
import glob
from os.path import join, exists

import numpy as np
import torch
import torch.nn.functional as f
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.mdrnn import MDRNNCell
from models.vae import VAE
from utils.misc import LSIZE, ASIZE, RSIZE, RED_SIZE

parser = argparse.ArgumentParser("MDRNN probes")
parser.add_argument('--logdir', type=str, required=True,
                    help='含 vae/best.tar 与 mdrnn/best.tar 的实验目录')
parser.add_argument('--datadir', type=str, default='datasets/carracing')
parser.add_argument('--probe', type=str, default='both',
                    choices=['swap', 'drift', 'both'])
parser.add_argument('--warmup', type=int, default=30,
                    help='先用真实历史把 LSTM 隐状态预热多少步')
parser.add_argument('--branch', type=int, default=15,
                    help='动作对换后每个分支往前滚多少步')
parser.add_argument('--horizon', type=int, default=40,
                    help='漂移曲线的自由 rollout 步数')
parser.add_argument('--windows', type=int, default=8,
                    help='漂移曲线对多少个起点求平均')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- 1. 加载 VAE 与 MDRNNCell（照抄 utils/misc.py::RolloutGenerator 的方式） ----
vae_file = join(args.logdir, 'vae', 'best.tar')
rnn_file = join(args.logdir, 'mdrnn', 'best.tar')
assert exists(vae_file), "缺 " + vae_file + "，先跑 trainvae.py"
assert exists(rnn_file), "缺 " + rnn_file + "，先跑 trainmdrnn.py"

vae = VAE(3, LSIZE).to(device)
vae.load_state_dict(torch.load(vae_file, map_location=device)['state_dict'])
vae.eval()

mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(device)
rnn_state = torch.load(rnn_file, map_location=device)['state_dict']
# 训练存的是 MDRNN（nn.LSTM，键名带 _l0），加载进 MDRNNCell（nn.LSTMCell）要去后缀
mdrnn.load_state_dict({k.strip('_l0'): v for k, v in rnn_state.items()})
mdrnn.eval()

# ---- 2. 取一条轨迹，编码成 z 序列，动作按 loaders.py 的方式对齐 ----
files = sorted(glob.glob(join(args.datadir, '*', '*.npz')))
assert files, "在 " + args.datadir + " 下没找到 npz 轨迹"
data = np.load(files[-1])
print("使用轨迹:", files[-1])

T = min(len(data['observations']), 300)
with torch.no_grad():
    obs = torch.as_tensor(data['observations'][:T],
                          dtype=torch.float32).permute(0, 3, 1, 2) / 255
    obs = f.interpolate(obs, size=RED_SIZE, mode='bilinear',
                        align_corners=True)
    zs, _ = vae.encoder(obs.to(device))   # 探针用后验均值当 z，锁死随机性
# actions[t+1] 才是作用在 observations[t] 上的动作（同 loaders.py 的切法）
acts = torch.as_tensor(data['actions'][1:T], dtype=torch.float32).to(device)

# 真实相邻帧的平均 z 位移：所有距离的比例尺，也是"惯性外推"的偷懒基线
scale = torch.norm(zs[1:] - zs[:-1], dim=1).mean().item()


def step(action, z, hidden):
    """单步前向，返回混合分布的总均值 E[z'] 和新隐状态（忽略 r、d 两个头）。"""
    mus, sigmas, logpi, r, d, next_hidden = mdrnn(action, z, hidden)
    pi = torch.exp(logpi).unsqueeze(-1)          # (1, 5, 1)
    z_mean = torch.sum(pi * mus, dim=1)          # (1, LSIZE)
    return z_mean, next_hidden


def warm_hidden(upto):
    """用真实 (动作, z) 序列把 LSTM 隐状态推进到时刻 upto。"""
    hidden = [torch.zeros(1, RSIZE).to(device) for _ in range(2)]
    with torch.no_grad():
        for t in range(upto):
            _, hidden = step(acts[t:t + 1], zs[t:t + 1], hidden)
    return hidden


PROBE_ACTIONS = [
    ('left',  [-1.0, 0.0, 0.0]),
    ('right', [1.0, 0.0, 0.0]),
    ('gas',   [0.0, 1.0, 0.0]),
    ('brake', [0.0, 0.0, 0.8]),
]


def swap_probe():
    """同一状态、同一段记忆，只换动作，各滚 branch 步，看分岔。"""
    t0 = args.warmup
    hidden0 = warm_hidden(t0)
    names, trajs = [], []
    with torch.no_grad():
        for name, a in PROBE_ACTIONS:
            action = torch.tensor([a], dtype=torch.float32).to(device)
            z = zs[t0:t0 + 1]
            hidden = tuple(h.clone() for h in hidden0)
            branch = []
            for _ in range(args.branch):
                z, hidden = step(action, z, hidden)
                branch.append(z)
            names.append(name)
            trajs.append(torch.cat(branch))       # (branch, LSIZE)

    print("\n比例尺（真实一步 z 位移均值）: %.3f" % scale)
    for k, label in [(0, "第 1 步"), (args.branch - 1, "第 %d 步" % args.branch)]:
        print("\n[%s] 动作两两距离 / 比例尺（对角线应为 0）:" % label)
        print("%8s" % "" + "".join("%8s" % n for n in names))
        for i, ni in enumerate(names):
            row = "%8s" % ni
            for j in range(len(names)):
                dist = torch.norm(trajs[i][k] - trajs[j][k]).item() / scale
                row += "%8.2f" % dist
            print(row)

    with torch.no_grad():
        imgs = vae.decoder(torch.stack([tr[-1] for tr in trajs])).cpu()
    fig, axes = plt.subplots(1, len(names) + 1,
                             figsize=(3 * (len(names) + 1), 3))
    axes[0].imshow(obs[t0].permute(1, 2, 0).numpy())
    axes[0].set_title('t0 (real)')
    for ax, name, img in zip(axes[1:], names, imgs):
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title('%s +%d' % (name, args.branch))
    for ax in axes:
        ax.axis('off')
    out = join(args.logdir, 'mdrnn', 'action_swap.png')
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print("\n分岔解码图已存到", out)


def drift_probe():
    """teacher forcing 一步误差 vs 自由 rollout 累积误差，多窗口平均。"""
    assert T - args.horizon - 2 > args.warmup, "轨迹太短，调小 --horizon"
    t0s = np.linspace(args.warmup, T - args.horizon - 2,
                      args.windows).astype(int)
    tf_err = np.zeros(args.horizon)
    fr_err = np.zeros(args.horizon)
    with torch.no_grad():
        for t0 in t0s:
            hidden = warm_hidden(t0)
            # teacher forcing：每步都喂真实 z
            h_tf = tuple(h.clone() for h in hidden)
            for k in range(args.horizon):
                zhat, h_tf = step(acts[t0 + k:t0 + k + 1],
                                  zs[t0 + k:t0 + k + 1], h_tf)
                tf_err[k] += torch.norm(zhat - zs[t0 + k + 1]).item()
            # 自由 rollout：喂自己的预测（动作仍用真实动作，只换状态来源）
            h_fr = tuple(h.clone() for h in hidden)
            z = zs[t0:t0 + 1]
            for k in range(args.horizon):
                z, h_fr = step(acts[t0 + k:t0 + k + 1], z, h_fr)
                fr_err[k] += torch.norm(z - zs[t0 + k + 1]).item()
    tf_err /= len(t0s)
    fr_err /= len(t0s)

    print("\n偷懒基线（每步都猜 z 不变）: %.3f" % scale)
    print("第 1 步误差   teacher=%.3f  free=%.3f（两者必须相等）"
          % (tf_err[0], fr_err[0]))
    print("第 %d 步误差  teacher=%.3f  free=%.3f"
          % (args.horizon, tf_err[-1], fr_err[-1]))

    plt.figure(figsize=(7, 4))
    ks = np.arange(1, args.horizon + 1)
    plt.plot(ks, tf_err, label='teacher forcing (one-step)')
    plt.plot(ks, fr_err, label='free rollout')
    plt.axhline(scale, linestyle='--', linewidth=1,
                label='lazy baseline (copy z)')
    plt.axvline(32, linestyle=':', linewidth=1, label='train SEQ_LEN')
    plt.xlabel('steps ahead')
    plt.ylabel('L2 error in z space')
    plt.legend()
    out = join(args.logdir, 'mdrnn', 'drift_curve.png')
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print("漂移曲线已存到", out)


if args.probe in ('swap', 'both'):
    swap_probe()
if args.probe in ('drift', 'both'):
    drift_probe()
```

代码里三个容易看漏的忠实细节：加载 `MDRNNCell` 用的是仓库自己的 `strip('_l0')`
方案；动作用 `actions[1:T]` 对齐，和 `loaders.py` 的 `seq_index + 1` 切法同源；
探针全程用后验均值当 $z$（训练时是采样的），因为审讯要求确定性，分岔必须
100% 归因于动作。

### Step 4: 动作对换对照实验（重点）

```bash
python probe_mdrnn.py --logdir exp_dir --probe swap
```

预期读数，按顺序检查四件事：

1. 对角线全为 0.00。同一动作喂两遍输出必须一字不差，这验证探针本身没有漏掉
   的随机性。不为 0，先修探针再谈别的。
2. 第 1 步的分岔小但非零。惯性世界里一步之内动作影响本来就小，`left` 对
   `right` 的距离除以比例尺通常只有零点几；这一格等于 0.00 才是坏消息。
3. 第 15 步的分岔显著大于第 1 步，其中 `left` 对 `right` 应是全表最大，两个
   反向打满的方向盘，15 步后的世界理应差得最远。健康模型这一格能到 1 的量级或
   以上（动作改写未来的幅度，赶上了世界自己一步的变化量）；如果全表仍贴着 0，
   宣布动作盲，去第 10 节。
4. 打开 `exp_dir/mdrnn/action_swap.png` 肉眼验收：五张图，第一张是出发时刻的
   真实画面，后四张是四个动作各自持续 15 步后的解码预测。left 和 right 两张的
   赛道走向、车身姿态应当肉眼可辨地不同。图会糊（混合均值本来就是几个候选的加权
   平均，再加 VAE 解码的糊），糊不要紧，方向要分得开。

把数字表和图存进证据目录，这是"动作对换对照报告"的原始材料。

### Step 5: 多步自由 rollout 的误差滚雪球

```bash
python probe_mdrnn.py --logdir exp_dir --probe drift
```

预期读数，按顺序检查四件事：

1. 两条曲线在第 1 步严格重合。第一步两种喂法的输入完全相同，数值必须相等，
   这是又一个内建的代码正确性检查。
2. teacher forcing 曲线全程基本走平，且明显低于图中虚线的偷懒基线（"每步都猜
   $z$ 不变"的误差，正好等于真实一步位移）。低于基线，说明 M 学到的东西超过了
   惯性外推；高于基线，这个 M 白训了。
3. 自由 rollout 曲线一路上扬，与 teacher forcing 的裂口越来越宽。这是
   exposure bias 的定量画像：模型每一步都在吃自己上一步的误差。
4. 注意 32 步那条竖线（训练序列长度 SEQ_LEN）。越过它之后模型进入训练时从未
   见过的时程；曲线在此处不必突变，但你应该记住：第 04 课的梦要做几百步，全部
   在这条线右边。

自由 rollout 误差涨到偷懒基线以上并不奇怪，滚了几十步后，预测和真相已经是两条
不同的赛道了，这时逐点 L2 距离本身也开始失去意义（两个都合理的未来，逐点比较
必然距离很大）。这个"多步之后逐点误差不再公道"的观察先记下，第 17 课评测学
会正面处理它。

### Step 6: 写对照报告，归档证据

在 `exp_dir` 的 `NOTES.md` 里追加本课一节，模板：

```text
数据规模与来源（多少条轨迹、是否补采、探针用了哪条轨迹文件）
训练命令与配置（是否开 include_reward、最终 train/test loss）
动作对换：比例尺数值、第 1 步与第 15 步的 left-right 距离、结论（动作盲/不盲）
漂移曲线：第 1 步误差、第 40 步 teacher 与 free 的误差、裂口出现的大致步数
两张图的文件路径与生成命令
本实验只说明：M 在此数据分布上的一步与多步行为；不说明梦境训练一定可行（那是第 04 课的事）
```

## 8. 配置与预算

| 项 | 本课配置 | 说明 |
|---|---|---|
| 数据 | 1000 条轨迹（README 正式配置） | 受 loaders.py 的固定切分影响，实际 400 训 / 600 测 |
| 模型 | LSTM 隐层 256，混合高斯 5 分量 | 约 38 万参数（LSTM 约 30 万 + 输出头 8.4 万），显存忽略不计 |
| 训练 | 30 epoch，batch 16，序列长 32 | 每个 epoch 只轮换 30 条轨迹的缓冲，30 轮大致把训练文件过一遍 |
| 优化器 | RMSprop，lr 1e-3，alpha 0.9 | 仓库写死；测试损失 5 轮不降则学习率减半 |
| 早停 | patience 30 | 默认只训 30 epoch，这个早停实际上不会触发，纯当保险 |
| 训练耗时 | 单卡数小时 | 瓶颈是 VAE 在线编码与轨迹文件 IO；Mac/CPU 可跑，慢数倍 |
| 探针耗时 | 数分钟 | 纯前向，CPU 足够 |
| 检查点 | `exp_dir/mdrnn/best.tar`、`checkpoint.tar` | best 按测试损失挑选；探针只认 best.tar |

预算提示：本课新增的计算大头其实在 Step 1 的数据补采和 VAE 重训上，MDN-RNN 训练
本身很便宜。这个"预测器比想象中便宜"的观感记住它，到第 09、10 课换成
Transformer 和扩散引擎时，同一个位置的零件会贵三个数量级。

## 9. 验收

验收清单：

- [ ] train/test 损失曲线双降并存档；能指出损失里 gmm、BCE 两项（开了
      `--include_reward` 则三项）各自是什么；
- [ ] 漂移图上 teacher forcing 曲线全程低于偷懒基线虚线，一步预测确实胜过
      惯性外推；
- [ ] 动作对换数字表：对角线全 0；`left` 对 `right` 的距离在第 15 步显著大于
      第 1 步；
- [ ] `action_swap.png` 里 left 与 right 两个分支的解码画面肉眼可辨地不同；
- [ ] 漂移图两条曲线第 1 步重合、自由 rollout 一路上扬，裂口讲得出成因；
- [ ] `NOTES.md` 按 Step 6 模板更新，报告里"这个实验只说明什么"一句写清；
- [ ] 能口头回答四连问：为什么输出分布而非点？NLL 为什么可能是负的？为什么平均
      误差查不出动作盲？为什么一步准不代表多步准？

## 10. 排错

| 症状 | 最可能的原因 | 怎么确认 | 怎么修 |
|---|---|---|---|
| `trainmdrnn.py` 一启动就 assert 失败 | logdir 里没有训好的 VAE | 看 `exp_dir/vae/best.tar` 在不在 | 先完成第 02 课流程，或检查 `--logdir` 拼写 |
| 训练除零报错，或每个 epoch 一瞬间结束 | 轨迹少于 600 条，训练集被切成空 | 跑 Step 1 的计数命令 | 补采到 1000 条 |
| loss 是负数 | 不是病：连续分布密度可大于 1 | 看趋势是否仍在下降 | 不用修 |
| loss 不降或 NaN | VAE 质量差、z 尺度异常或学习率过高 | 先看第 02 课的重建验收有没有过 | 重训/换 VAE；仍不行再考虑调低 lr |
| 探针加载报 missing/unexpected keys | 键名没转换成功，或加载了别的文件 | 打印 `rnn_state.keys()` 看有没有 `_l0` 后缀 | 确认加载 `best.tar` 且保留 `strip('_l0')` 那行 |
| `torch.load` 报 weights_only 或 Unpickling 错误 | 新版 PyTorch 默认拒绝反序列化自定义对象（checkpoint 里存了早停对象） | 查 torch 版本 | 用第 01 课的老虚拟环境；新环境则给 `torch.load` 加 `weights_only=False` |
| 对换表对角线不为 0 | 探针里混入了随机性 | 检查是否改动了 Step 3 代码（均值编码、`eval()`） | 恢复原版胶水代码 |
| 分岔全贴 0：动作盲实锤 | 动作对齐错位、训练不足或数据里动作太温和 | 先用第 11 节的"致盲对照"确认探针本身工作正常；再检查自己改过的数据管道有没有差一位 | 用仓库原版 loaders 重训；加数据加 epoch；确认采数据用的是布朗策略 |
| 数字表分岔正常但四张解码图几乎一样 | z 的差异没大到解码可见 | 数字表第 15 步的读数是否仍偏小 | 把 `--branch` 加大到 30；或改成解码权重最大的单个分量 |
| 自由 rollout 曲线不上扬，和 teacher forcing 重合 | 改代码时把真实 z 误喂进了自由分支 | 两条曲线处处相等就是铁证 | 对照 Step 3 原版恢复状态来源那两行 |

## 11. 前沿与改造

本课的三个难题，后面每一幕都有人换零件重解。多峰未来：IRIS
（第 09 课）把 $z$ 换成离散 token，"下一个 token 的 softmax"天然就是多峰分布，
混合高斯的活被词表接管；DIAMOND（第 10 课）用扩散模型直接建模下一帧的完整分布，
连"几个峰"都不用事先决定；RSSM（第 05 课）则把随机性从输出端挪进状态本身。
误差滚雪球：PlaNet（第 05 课的论文）训练时就做多步预测，正面消解"只练一步"；
DreamerV3（第 06 课）干脆认下模型会错这个命，让策略全程活在模型的想象里，错误
成了训练环境的一部分；更早的 Scheduled Sampling（Bengio et al., 2015,
arXiv:1506.03099）思路是训练时按概率把真实输入换成模型自己的输出，让模型提前
见识自己的错误。动作条件：现代系统照样是把动作拼进或注意进预测器，真正的新花样
在第 11 课，Genie 一族连动作标签都不要，从纯视频里自己挖出动作变量。

机制差距占大头：单步 teacher forcing 训练、输出端才引入随机性、
一条 LSTM 撑全部记忆，这三条分别被上面三组工作换掉，而且换法在单卡尺度就能
体验（第 05、06、09 课都是单卡实验）。规模差距反而其次：这个 38 万参数的 M 在
CarRacing 上够用，第 18 课会正经讨论它什么时候不够用。

动手改造清单（第 1 个强烈建议做，是本课论点的收尾；后两个选做）：

1. 致盲对照训练：故意训一个动作盲模型，证明平均误差真的查不出它。做法：在
   `trainmdrnn.py::get_loss` 开头加一行 `action = action * 0`，logdir 换成
   `exp_dir_blind`（先把 `exp_dir/vae` 整目录复制过去，M 要用同一个 VAE），重训。
   预算：与主训练相同，数小时。预期：致盲版的 test loss 只比正常版差一点（惯性
   世界给它兜底），但用探针一测，分岔指数塌到 0 附近，同一份平均误差下藏着
   两个天壤之别的模型，钝器与利器的全部论证到此闭合成你自己的实验数据。失败
   判据：如果致盲版 loss 大幅变差，说明 CarRacing 的动作信息比预想的重，这同样
   是有价值的记录（顺带证明你的探针没白造）。
2. 高斯个数扫描：把 `trainmdrnn.py` 里 `MDRNN(LSIZE, ASIZE, RSIZE, 5)` 的 5
   改成 1 和 10 各训一版（探针里 `MDRNNCell` 的对应参数同改）。预算：每版数小时。
   预期：$K=1$ 的 test NLL 更差，且在弯道样本上被迫用更大的 $\sigma$ 掩盖多峰
   （鬼影的另一种表现形式：中心不敢选边站，只好把伞撑大）；$K=10$ 与 $K=5$
   接近。失败判据：$K=1$ 与 $K=5$ 打平，说明这份数据里真正多峰的场面比预想少，
   把它写进报告，这是关于环境的知识。
3. 均值 z 对照采样 z：把 `to_latent` 里的采样行为改成直接用 $\mu$，重训一版。
   预算：数小时。预期：两版的一步损失接近，但漂移曲线分开，采样噪声练出来的
   模型见惯了输入抖动，自由 rollout 时对自己的预测误差更皮实。失败判据：两版
   曲线不可区分，说明这份免费数据增强在当前规模下没兑现，也如实记录。

两个经典结论能在本课设置里看到方向。其一，Bishop 1994 年报告的
核心论断"多峰条件分布的条件均值是个糟糕的预测器"：你的缩小版对应物就是 5.1 节
验证里那张"混合总均值解码图糊过单分量解码图"，加上改造实验 2 里 $K=1$ 的退化。
其二，Graves 手写生成论文的观察"采样温度控制生成序列的保守与狂野"：把探针的
自由 rollout 从"喂回均值"改成"从混合分布里采样喂回"，你已经站在第 04 课门口
，论文的温度 τ 公式下一课给出，配着梦境可视化一起玩。

## 12. 论文与延伸

1. World Models（Ha & Schmidhuber, 2018，[arXiv:1803.10122](https://arxiv.org/abs/1803.10122)）
   ，这次只重读 M 相关的小节。带着三个问题：CarRacing 版的 controller 并不直接
   读 M 输出的预测分布，只读隐状态 $h$，那 M 的预测训练到底把什么东西灌进了
   $h$？VizDoom 梦境版的 M 为什么必须多预测一个 done？温度 τ 被加在 M 的哪个
   输出上、防的是什么病？（最后一个问题的答案是第 04 课的主线剧情。）
2. Mixture Density Networks（Bishop, 1994，Aston University 技术报告
   NCRG/94/004），MDN 的出生证明，比 World Models 早 24 年。带着两个问题读：
   他用哪个玩具问题证明"条件均值在一对多映射上必然失败"？他对 $\pi, \mu, \sigma$
   的参数化（softmax 管权重、指数管尺度）和 `mdrnn.py` 里那几行是否一字不差？
   读完你会发现 2018 年的 M 在这份 1994 年的报告面前几乎没有新数学。
3. Generating Sequences With Recurrent Neural Networks（Graves, 2013,
   [arXiv:1308.0850](https://arxiv.org/abs/1308.0850)），MDN 接上 RNN 的经典
   之作，用来生成手写笔迹。带着三个问题读：下一个笔尖位置为什么天然多峰？他的
   输出头除了混合高斯还有一个"抬笔"伯努利分量，和我们的 done 头是什么关系？
   他生成整页手写时靠什么撑住长序列不散架，对照你的漂移曲线想。
4. 选读：[worldmodels.github.io](https://worldmodels.github.io) 的 MDN-RNN 交互
   演示，第 01 课玩过的页面，这次专门玩预测分布那部分，拖动作滑块看预测怎么变，
   相当于官方版的动作对换实验。

收工前看一眼全局：三件套已经齐了两件，V 把画面压成 32 个数，M 握着这串数和
你的动作往前推演，而且你验过它确实在听方向盘、也量出了它做长梦会飘。下一课
（第 04 课）装上最后一件：867 个参数的 controller 用 CMA-ES 在真实赛道上练出来
之后，我们把整个训练搬进 M 生成的梦里，到时候你会看到，策略这个学生专挑模型
学错的地方钻，而你手里的漂移曲线和温度旋钮，就是抓它作弊的证据和工具。动作对换这一条，在第 33 课是具身程度的 Q1：分岔失败，后面的档全部免谈。
