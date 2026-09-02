---
id: 04_controller_dream_training
title: "训练控制器，并把训练搬进梦里"
summary: "控制器只有 867 个参数，凭什么能开车？在自己模型的想象里训练策略，什么时候会被“钻空子”？"
unit: vmc
play_tools: []
checkpoints:
  - "World Models 复现报告（论文复现 #1）：真实分数方向性对照、梦境移植实验、失败案例分析。"
  - "一组模型内外的分数对照，用来验证策略是否利用了模型误差。"
---

# 第 04 课：训练控制器，并把训练搬进梦里

> 类型：复现 + 移植实验（World Models 复现 #1 在本课收官）<br>
> 建议周期：4-7 天（大头是机器在跑，人的操作集中在头尾各半天）<br>
> 硬件：controller 进化吃 CPU 多核；单张 24GB 卡 + 多核 CPU，复现档数天；梦境侧实验 Mac/纯 CPU 可完成<br>
> 锚定仓库：[ctallec/world-models](https://github.com/ctallec/world-models)，对照精读 [hardmaru/WorldModelsExperiments](https://github.com/hardmaru/WorldModelsExperiments) 的 doomrnn 目录<br>
> 产物：World Models 复现报告 + 梦境移植实验报告

## 1. 这一课做什么

V 和 M 已经通过各自的验收，最后缺的是根据潜状态选择动作的 controller。本课先复现
论文在 CarRacing 上的训练方式，再做一个论文之外的移植实验：把 controller 的训练场
从真实环境搬进 MDN-RNN 生成的轨迹。

论文中的 CarRacing controller 是一个
只有 867 个参数的线性层，用 CMA-ES（一种不算梯度的进化算法）在真实赛道上一代代筛出来。
我们用 ctallec 仓库的 `traincontroller.py`，按复现档预算（1000 条轨迹数据、加大的种群）
把这条路线完整走一遍，跑出 100 局均值±方差，和论文的 906±21 做方向性对照，写出这门课
的第一份论文复现报告。

需要明确的是：论文的梦境训练是在 VizDoom 的 Take Cover 任务上做的，并未在
CarRacing 上报告这一结果。我们的移植实验会写少量胶水代码，把第 03 课训好的 MDN-RNN
包装成一个"假环境"：不渲染任何像素，rollout 全程在 $z$ 空间里滚动，done 和 reward
由 M 自己的输出头给出。controller 在这个梦里进化，练完拉回真实赛道验收。然后调采样
温度 τ，观察论文 Doom 实验中的核心现象，梦太"确定"，策略可能会利用模型误差：
在梦里分数上天，回真实环境立刻撞车。这个实验允许失败（毕竟换了任务、换了 reward 的
来源），但失败判据和诊断路径本身就是需要教的内容。

最终产物有两份：一份是 100 局真实赛道的均值与方差，并列论文的 906±21；另一份按
温度记录同一策略在模型内外的分数。两者的差距会把“模型内训练是否转移到真实环境”
从一句印象变成可检查的数字，也为第 06 课的 Dreamer 留下直接参照。

术语速查：

| 术语 | 一句人话 |
|---|---|
| C / controller | 系统的手：读 $z_t$（V 的压缩）和 $h_t$（M 的记忆），线性输出转向、油门、刹车，全部家当 867 个参数 |
| CMA-ES | 不算梯度的优化器：撒一群候选参数、按得分排座次、把搜索分布往高分方向挪并拉伸，循环往复 |
| 适应度（fitness） | 进化算法给每个候选打的分，这里就是一局（或几局平均）的赛车总分 |
| 种群（population） | 每一代同时评估的候选参数个数，`--pop-size` 控制 |
| 搜索椭球 | CMA-ES 撒点用的多维高斯分布，均值是当前最优猜测，协方差矩阵决定往哪些方向撒得开 |
| 梦境环境（dream environment） | 把 MDN-RNN 包装成的假环境：状态是 $z$，下一步靠采样 M 的预测分布，全程不碰真实模拟器 |
| 温度 τ | 从 MDN 采样时的"加噪旋钮"：混合权重的 logits 除以 τ，高斯方差乘以 τ；τ 越大梦越随机 |
| reward 头 / done 头 | MDN-RNN 除了预测下一个 $z$，还各留了一个标量输出预测本步奖励和"这局结束没"；梦境环境全靠它们发工资、喊停 |
| 钻空子（exploitation） | 策略在模型学错的角落里刷分：梦里合法，真实世界不认账 |
| 方向性复现 | 跑出与论文同方向、同量级的结论（比如"远超基线、接近解决线"），不要求逐点对齐原文数字 |

## 2. 问题

三个具体问题，外加一条要先划清的界限。

1. 用无梯度方法把 867 个参数训到能开车。这里的拦路虎不是模型大小，而是信号形态：
   controller 的好坏只体现在"一整局跑下来多少分"，中间隔着上千步环境交互和随机生成
   的赛道，没有可反传的梯度。CMA-ES 靠种群加排序绕开这个问题，代价是每一代要烧掉
   几百局真实 rollout，所以这一步吃的是 CPU 核数，不是显卡。
2. 把 M 从预测器改造成模拟器。第 03 课的 M 只被要求"预测得准"；当模拟器用，
   它还得会发奖励、会喊停。这课你会撞上一个仓库的真实细节：`models/mdrnn.py` 的
   网络里 reward 头和 done 头一直都在，但 `trainmdrnn.py` 的 reward 损失默认是关的
   （`--include_reward` 不加就不训），不知道这一点，你的梦境环境会拿一个随机初始化
   的 reward 头给 controller 发工资，梦里的一切繁荣都是假账。
3. 给"在想象里训练"建立信任边界。什么时候能信梦里的分数？本课的答案是一套
   可操作的流程：先校准（拿真实轨迹对拍 M 的 reward 预测），再训练，再回真实环境
   验收，最后用温度 τ 做压力测试。论文 Doom 实验给出了方向（τ 太低，虚拟分 2086
   而真实分只有 193；τ 合适，虚拟分反而低但真实分 1092），我们检验自己的移植版
   是否出现同方向的裂缝。

界限：上半场是**复现**（论文的 CarRacing 路线就是真实环境训练，我们按同样路线缩小
预算跑）；下半场是**移植实验**（论文没在 CarRacing 上做过梦境训练，reward 来源也
和 Doom 的"活着就算分"本质不同）。两份报告必须分开写，结论不能互相串门，把移植
实验的结果说成"复现了论文的梦境训练"，是这课最不能犯的错。

## 3. 准备

- 第 01 课的环境：ctallec 仓库能跑、xvfb 可用（远程机必备）。CMA-ES 用的 `cma`
  库随 `requirements.txt` 安装；如果 `import cma` 报缺，单独 `pip install cma` 补上。
- 复现档数据：1000 条轨迹（README 的正式配置）。第 03 课如果已经采过就直接用；
  只有第 01 课那 100 条的话，Step 1 先补齐。按第 01 课的磁盘比例估算，1000 条
  大约十几 GB，磁盘留出 20GB。
- 上两课的产物：`exp_dir/vae/best.tar`（第 02 课）和 `exp_dir/mdrnn/best.tar`
  （第 03 课）。跳过了那两课的话，最短补课路径是 `trainvae.py` 与 `trainmdrnn.py`
  各跑一遍（命令见第 01 课 Step 3/4），但 latent 走查和动作对换实验强烈建议补上，
  本课诊断梦境问题时会反复用到那两套手艺。
- 一台多核机器：16 核以上舒服，8 核也能跑（时间大致翻倍）。`traincontroller.py`
  的并行是多进程，每个 worker 独立跑完整局真实 rollout，核越多每代越快。
- 心理预算：真实环境进化那一步是"开着机器过周末"级别的任务。先读完第 8 节把
  账算清楚再按回车。

## 4. 学习目标

1. 讲清 CMA-ES 的工作循环（撒点、排序、挪均值、变形协方差），并说出它为什么恰好
   配得上 867 个参数、为什么配不上 V 和 M 的百万级参数；
2. 解释"每个候选评估多局取平均"在对抗什么，并能用自己的 100 局评估数据说明单局
   分数为什么没资格参与比较；
3. 交出一份合格的复现报告：设置、与论文的已知差异、均值±方差与局数、方向性结论
   和它的边界；
4. 把一个训好的 MDN-RNN 包装成 gym 风格的假环境，说清 done、reward、初始状态三件事
   各从哪来、各有什么坑；
5. 写出温度 τ 的两处数学作用位置（混合权重 logits 除以 τ、高斯方差乘以 τ），并解释
   它为什么能抑制钻空子；
6. 说出我们的移植实验与论文 Doom 实验的三处本质差异（任务、reward 来源、模型规模），
   并据此判断哪些结论可以指望方向一致、哪些不能。

## 5. 原理

五个机制，每个还是那套节奏：直觉、机制、数学、代码落点、验证。

### 5.1 为什么 867 个参数的手够用：难的部分早就被 V 和 M 干完了

教练带新手开卡丁车，从来不教视网膜怎么处理光信号，那部分人脑已经建好了。
教的只是一句话级别的映射："弯道内侧贴近点、出弯再给油"。World Models 的分工同理：
V 把一帧画面浓缩成 32 个数（路的弯度、车的位置），M 的隐状态 $h_t$ 里攒着时序信息
（速度、正在转弯还是直行），这两样合起来已经是一份"驾驶简报"。C 要做的只剩下把
简报线性组合成三个操作量。表征学习的重活由不需要奖励信号的 V 和 M 承包，需要奖励的
部分被压到最小，这是 2018 年能在一台机器上跑通这套系统的核心算计。

C 读两个向量：$z_t$（32 维）和 $h_t$（256 维，LSTM 的隐状态），拼接成
288 维，过一个线性层直接输出 3 维动作。没有隐藏层，没有非线性堆叠。

全部参数就是一个权重矩阵加偏置：

$$
a_t = W\,[z_t; h_t] + b
$$

$W$ 是 $3 \times 288 = 864$ 个数，$b$ 是 3 个，合计 867。参数少到什么程度？写进一条
一维向量里，CMA-ES 直接把它当搜索空间里的一个点。

`models/controller.py::Controller`，全文有效代码不到十行：
`nn.Linear(latents + recurrents, actions)` 加一个把输入拼接起来的 `forward`。它在
真实环境里怎么被调用，看 `utils/misc.py::RolloutGenerator.get_action_and_transition`：
先 `self.vae(obs)` 取出 latent 均值，然后 `self.controller(latent_mu, hidden[0])` 出
动作，最后 `self.mdrnn(action, latent_mu, hidden)` 把记忆滚一步。注意 C 用的是
`hidden[0]`，LSTM 隐状态那一半，不是 cell state。

两个层面。数参数：`sum(p.numel() for p in controller.parameters())` 必须
等于 867。看证据：论文的对照实验里，只给 $z$ 不给 $h$ 的版本 100 局均分 632±251，
加上 $h$ 的完整版 906±21，时序信息（$h$ 里的速度感）值 270 多分。顺带一个第 12 节
还会回来的悬念：ctallec 的项目页报告，把 M 换成随机初始化、完全不训练，真实环境分数
几乎不掉。读出头够用的另一面是：它对上游的要求可能比你以为的低，但这条捷径到了
梦境训练立刻死路，因为没训过的 M 当不了模拟器。

### 5.2 CMA-ES：往打分高的方向拉伸搜索椭球

蒙眼在丘陵地上找最高点，你带着一支 32 人的小队。每一轮你让队员在你周围
散开站定，逐个报海拔；你走向报数高的那几个人的重心，下一轮撒人时还刻意往"刚才高个
们连成的方向"撒得更开、往没人报高分的方向收紧。几轮之后，这支队伍撒开的形状（一个
椭球）会自己顺着山脊的走向拉长，不用任何坡度信息，光靠"谁比谁高"的排序就把地形
摸出来了。CMA-ES 的全名是协方差矩阵自适应进化策略，"协方差矩阵自适应"就是那个会
自己变形的撒人形状。

每一代四步：从当前高斯分布采样 $\lambda$ 个候选参数向量；每个候选放进
环境跑几局，取平均分当适应度；按适应度排序，取前面一部分的加权平均作为新均值；
用"这一步成功的移动方向"更新协方差矩阵（拉伸成功方向）和整体步长 $\sigma$。只用
排序、不用分数的具体数值，这让它对适应度的噪声和量纲天然不敏感。

采样写出来是：

$$
x_i = m + \sigma\, y_i,\qquad y_i \sim \mathcal{N}(0, C),\qquad i = 1,\dots,\lambda
$$

$m$ 是均值（当前最优猜测），$\sigma$ 是全局步长，$C$ 是协方差矩阵（椭球的形状）。
均值更新是排序加权：$m \leftarrow \sum_{i=1}^{\mu} w_i\, x_{i:\lambda}$，其中
$x_{i:\lambda}$ 表示按分数排第 $i$ 名的候选，权重 $w_i$ 递减。$C$ 的更新用成功步的
外积把椭球往那个方向拉长（精确公式见第 12 节 Hansen 教程，工程上记住"拉伸成功方向"
这个作用就够）。为什么这套东西配 867 维恰好？三笔账：其一，$C$ 是 $d \times d$ 矩阵，
867 维时约 75 万个数，更新和采样都轻松，但到 V/M 的百万参数级别就是天文数字，
维度的平方是 CMA-ES 的硬天花板；其二，适应度要跑完整局才有一个数，环境和渲染器
不可微，梯度方法根本进不了场；其三，种群天生并行，正好把多核 CPU 喂满。

`traincontroller.py` 里就一行核心：
`cma.CMAEvolutionStrategy(flatten_parameters(parameters), 0.1, {'popsize': pop_size})`
，第三方 `cma` 库，初始均值是一个新建 Controller 展平后的参数向量（
`utils/misc.py::flatten_parameters`，一行 `torch.cat` 拼成 numpy 数组），初始步长
0.1，种群大小从命令行 `--pop-size` 进来。反方向的转换是
`utils/misc.py::load_parameters`：把 CMA-ES 给的一维向量切片、变形、拷回网络。还有
一个符号约定要盯住：`RolloutGenerator.rollout` 返回的是**负的**累计奖励，因为 `cma`
库做最小化；日志里看到分数带负号别慌，取反就是赛车分。

训练日志里最好个体的（取反后）分数应该呈"锯齿上行"：单代之间抖，隔几代
看趋势涨。`traincontroller.py` 每 3 代会把当前最优候选拿去跑 100 局完整评估（源码里
`evaluate` 函数，默认 `rollouts=100`），这个数比种群内的适应度可信得多，盯它。

### 5.3 每个候选评几局：和噪声掰手腕

CarRacing 每局的赛道是随机生成的。同一个 controller，抽到缓弯多的赛道
能拿 800，抽到发卡弯连环的赛道可能只有 600。如果每个候选只跑一局就定生死，CMA-ES
排序排的一半是实力、一半是签运，进化会朝着"手气好"的方向走。解法朴素：每个候选
多跑几局取平均，把签运稀释掉。

`traincontroller.py` 里，每个候选被塞进任务队列 `--n-samples` 次，结果
按候选累加平均（源码 `r_list[r_s_id] += r / n_samples`）。于是每一代的真实环境
rollout 总数是种群大小乘以评估局数，这是预算公式，也是这一步吃 CPU 的原因：
论文原始配置种群 64、每候选 16 局，一代就是 1024 局。

均值的标准误随局数按 $1/\sqrt{n}$ 缩小：4 局平均把单局的波动砍半，16 局
砍到四分之一。但预算按 $n$ 线性涨，所以 $n$ 是花在"排序别排错"上的保险费，买多少
取决于候选之间的真实差距有多大：进化前期候选间差几百分，$n$ 小点无妨；后期都挤在
850 附近，排序需要的分辨率变高，$n$ 太小就会在原地随机游走。

`traincontroller.py` 的 `--n-samples` 与 `--pop-size` 参数；并行侧是
`torch.multiprocessing` 起 worker 进程（上限 `--max-workers`，默认 32），每个 worker
里建一个 `RolloutGenerator(logdir, device, time_limit=1000)` 反复干活。

第 7 节 Step 5 你会给最终 controller 跑 100 局：把 100 个单局分数的直方图
画出来，看看标准差有多大，再回头算算"如果当初每候选只评 1 局，排序会错得多离谱"。
这一眼会让你永远记住为什么报分数必须带局数。

### 5.4 把 M 扭成模拟器：梦境环境的搭法

第 03 课的 M 是个预报员：你报当前路况和操作，它预报下一秒。现在我们要
提拔它当整个世界：预报完下一秒，就把预报当成真的，接着在预报之上再预报。像飞行员
上模拟器训练，省钱、安全、可以无限重开。但类比在关键处失效：真模拟器是工程师照
物理规律造的，误差有界；M 这台"模拟器"是从 1000 条随机驾驶轨迹里学出来的，哪里
没见过数据哪里就是幻觉，而下半场的主角（被进化压力驱动的策略）恰恰会把搜索火力
集中到幻觉最离谱的角落。

一个环境要能当训练场，得答上三问：状态怎么滚、什么时候停、奖励谁来发。
梦境环境的答案全在 M 身上。状态滚动：从 M 输出的混合高斯里**采样**一个 $z_{t+1}$
（注意是采样，取均值会把多峰分布捏成单点，第 03 课讲过多峰的意义）。什么时候停：
M 的 done 头输出一个 logit（还没过 sigmoid 的原始打分），过 sigmoid 超过阈值就算这局结束，另设步数上限兜底。
奖励：M 的 reward 头输出一个标量当本步奖励。全程没有渲染器、没有物理引擎，一步
就是一次 LSTMCell 前向加一次采样，比真实环境快两个数量级以上。

论文 Doom 实验里这套东西的完整形态是：M 建模
$P(z_{t+1} \mid z_t, a_t, h_t)$ 加终止概率；reward 不用学，Take Cover 任务里
"每活一步得一分"，所以梦境里的累计奖励就是存活步数，done 头一个零件同时兼任
裁判和出纳。**这正是我们的移植和论文的第一个本质差异**：CarRacing 的分数是"压过
新的赛道格子"，是状态的函数，不能白嫖 done 头，必须让 reward 头真的学会预测它。

三处。`models/mdrnn.py::MDRNNCell`：单步版 M，`forward(action,
latent, hidden)` 返回六元组 `(mus, sigmas, logpi, r, d, next_hidden)`，五个高斯的
均值方差、混合权重的 log、reward 预测、done 的 logit、滚动后的隐状态。输出层
`gmm_linear` 的输出维度是 `(2 * latents + 1) * gaussians + 2`，那个 `+ 2` 就是
reward 头和 done 头的座位。`trainmdrnn.py`：损失里 done 的 BCE 一直在训，但 reward
的 MSE 挂在 `--include_reward` 开关上，**默认不加就不训**（不含 reward 时损失是
`(gmm + bce) / (LSIZE + 1)`，含则是 `(gmm + bce + mse) / (LSIZE + 2)`）。第 03 课
按 README 默认命令训的 M，reward 头就是一堆随机初始化的权重，Step 3 要带
开关重训一份。第三处是官方对照：hardmaru 仓库 `doomrnn/doomrnn.py` 里的
`DoomCoverRNNEnv` 类，论文的梦境环境本尊，就是一个 gym 环境类，`_step` 方法内联了
MDN 采样，我们的胶水代码抄的就是这个结构。

拿一条真实轨迹的动作序列在梦里重放，用 VAE 的 `decoder` 把梦出来的 $z$
序列还原成图像条：路还像路、弯还会弯，梦才算及格。再对拍 reward：真实轨迹的
$(z_t, a_t)$ 喂给 M，把 reward 头的逐步预测和数据里存的真实 reward 画散点。这两步
是 Step 8 的正式内容，先校准模拟器，再进模拟器训练，顺序不可换。

### 5.5 温度 τ：往梦里掺噪声，让空子不好钻

梦境训练的死穴：策略的进化压力会自动搜索"模型哪里学错了"。Doom 梦里，
论文观察到 agent 找到一种诡异走位，让梦里的怪物干脆不发火球，真实游戏里怪物可
不惯着它。为什么会这样？M 学到的分布在数据稀疏处过于自信（某些状态下它笃定"不会
有火球"），策略就把家安在这些笃定错了的角落。解药反直觉地简单：把梦调得**更混乱
一点**。采样时加大随机性，模型笃定的地方也强制保留意外，策略就没法指望任何一条
侥幸路径稳定复现，空子还在，但踩不实了。

τ 拧的是 MDN 采样的两个旋钮：挑哪个高斯（混合权重），以及挑中之后噪声
放多大（方差）。τ 大于 1，权重分布被抹平（冷门峰也常被抽中）、每个峰的噪声也变大；
τ 趋近 0，权重坍缩到最大峰、噪声消失，M 退化成一台确定性机器，论文里的说法是
接近普通 LSTM，最容易被钻。

设混合权重的原始 logits 为 $\alpha_k$，温度采样是：

$$
k \sim \mathrm{Cat}\!\left(\mathrm{softmax}(\alpha / \tau)\right),\qquad
z_{t+1} = \mu_k + \sigma_k \sqrt{\tau}\,\epsilon,\qquad \epsilon \sim \mathcal{N}(0, I)
$$

两处动作：logits 除以 τ 再 softmax；标准差乘 $\sqrt{\tau}$（等价于方差乘 τ）。这与
官方 doomrnn 代码逐行对得上：`logmix2 = np.copy(logmix)/temperature` 后接 softmax
归一化，采样噪声是 `randn(OUTWIDTH)*np.sqrt(temperature)`。一个实现细节：ctallec
的 `MDRNNCell` 返回的 `logpi` 已经过了 log_softmax，直接拿它除以 τ 再 softmax 也
严格等价，log_softmax 只比原始 logits 差一个常数，除以 τ 后还是常数偏移，softmax
一步吃掉。

ctallec 仓库没有现成的温度采样（它的 `RolloutGenerator` 只在真实
环境里跑，M 只用来滚隐状态），所以这是本课胶水代码的核心函数，第 7 节 Step 7 给出
完整实现。官方参照物在 `doomrnn/doomrnn.py::DoomCoverRNNEnv._step`。

论文 Doom 实验的温度表就是判决书（虚拟分和真实分均为 100 局均值±标准差）：

| τ | 梦里的分 | 真实环境的分 |
|---|---|---|
| 0.10 | 2086 ± 140 | 193 ± 58 |
| 0.50 | 2060 ± 277 | 196 ± 50 |
| 1.00 | 1145 ± 690 | 868 ± 511 |
| 1.15 | 918 ± 546 | 1092 ± 556 |
| 1.30 | 732 ± 269 | 753 ± 139 |

对照基线：随机策略真实分 210±108，任务的解决线是 100 局平均存活 750 步。读这张表
的三个要点：τ=0.10 时梦里 2086、真实 193，比随机策略还差，策略学的全是钻空子的
本事；τ=1.15 拿到真实最高分 1092，此时梦里的分反而不高，**梦越难，学出来的越真**；
τ=1.30 真实分回落到 753 但方差最小，噪声太大信号也被淹了。我们的移植实验（Step 10）
就是在 CarRacing 梦里复刻这张表的三个点，看裂缝方向是否一致。

## 6. 源码导读

上半场全部在 ctallec 仓库，下半场加一个官方对照文件。建议按表内顺序读：

| 文件与位置 | 是什么 | 带着什么问题读 |
|---|---|---|
| `models/controller.py::Controller` | C 本体 | 确认 867 个参数的构成；`forward` 为什么用 `torch.cat` 收多个输入 |
| `utils/misc.py`（常量区） | 全局尺寸 | `ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE = 3, 32, 256, 64, 64`，控制器输入 288 维从哪来，一目了然 |
| `utils/misc.py::flatten_parameters` 与 `load_parameters` | 网络参数与一维向量互转 | CMA-ES 只认平坦向量；切片顺序靠什么保证两边一致 |
| `utils/misc.py::RolloutGenerator` | 真实环境评估器 | `__init__` 从 `mdir` 下的 `vae/`、`mdrnn/`、`ctrl/` 各找 `best.tar`；`rollout` 为什么返回负分；`get_action_and_transition` 里 V、C、M 的调用顺序 |
| `traincontroller.py` | CMA-ES 主循环 | 初始步长 0.1 写死在哪；`n_samples` 平均怎么实现；每 3 代的 100 局评估（`evaluate` 函数）；`best.tar` 什么条件下被覆盖 |
| `trainmdrnn.py::get_loss` | M 的损失装配 | `--include_reward` 开关拨动了哪两行；损失除以 `LSIZE + 2` 是在抵消什么 |
| `models/mdrnn.py::MDRNNCell` | 梦境引擎 | 六元组返回值的顺序；`gmm_linear` 输出维度公式里 `+ 2` 是谁 |
| `test_controller.py` | 单局试跑 | 数一数它到底跑几局、打印什么，答案是 1 局、什么都不打印，所以第 9 节的验收它干不了 |
| hardmaru 仓库 `doomrnn/doomrnn.py::DoomCoverRNNEnv` | 论文梦境环境本尊（只读对照） | `_step` 里温度作用的两个位置；梦境被包装成 gym 环境后，训练代码为什么可以完全不知道自己在做梦 |

## 7. 实验

上半场 Step 1-6（复现），下半场 Step 7-10（移植）。Step 3 吃 GPU、Step 4 吃 CPU，
两者没有依赖关系，建议同时开跑。

### Step 1: 把数据补到复现档

第 03 课如果已经采过 1000 条就跳过。只有第 01 课那 100 条的话：

```bash
python data/generation_script.py --rollouts 1000 --rootdir datasets/carracing --threads 8
```

预期：数小时（纯 CPU），磁盘涨十几 GB。抽查两条轨迹动画确认车在正常跑（第 01 课
的手艺）。V 和 M 用 100 条数据训过的话，最好也用新数据重训一轮再进本课，上游
质量决定这课的天花板。

### Step 2: 检查产物，分出梦境专用目录

确认 `exp_dir/vae/best.tar` 和 `exp_dir/mdrnn/best.tar` 都在，然后复制一份：

```bash
cp -r exp_dir exp_dream
```

再把复制过来的旧 controller 清掉（第 01 课冒烟档留下的，会污染梦境实验的起点）：

```bash
rm -rf exp_dream/ctrl
```

之后两条线互不打扰：`exp_dir` 走真实环境复现，`exp_dream` 走梦境移植。

### Step 3: 带 reward 头重训梦境版 M

```bash
python trainmdrnn.py --logdir exp_dream --include_reward --noreload
```

预期：GPU 数小时；日志里除了 gmm 和 bce，多出一项非零且随 epoch 下降的 mse，
那就是 reward 头在学发工资。`--noreload` 保证从头训而不是接着旧 checkpoint。这一步
只动 `exp_dream`，真实环境那条线（`exp_dir`）继续用第 03 课的 M，保持复现路线与
仓库默认配置一致。

### Step 4: 真实环境 CMA-ES（复现档）

```bash
xvfb-run -s "-screen 0 1400x900x24" python traincontroller.py --logdir exp_dir --n-samples 8 --pop-size 32 --target-return 900 --max-workers 16 --display
```

有显示器的机器可去掉 `xvfb-run` 前缀；`--max-workers` 按核数设。README 的示例是
种群 4、每候选 4 局的冒烟值，我们抬到 32×8：每代 256 局真实 rollout。`--target-return
900` 是论文的解决线，复现档大概率到不了，这个参数在这里的实际作用是"永不自动停，
跑满预算手动停"。放心停：`exp_dir/ctrl/best.tar` 只在 100 局评估创出新高时才被覆盖，
Ctrl-C 不丢东西。但注意重跑同一条命令时只会读回历史最好成绩当门槛，CMA-ES 的搜索
分布是从头再来的，长跑尽量一次挂完。主要仪表是每 3 代打印的
100 局评估分（记得取反）；预期它从负数爬到几百，数天内在某个位置进入平台期。

### Step 5: 100 局正式评估加随机基线

仓库自带的 `test_controller.py` 只跑 1 局且不打印分数，当不了验收工具。在仓库根目录
写一个 20 行的评估脚本 `eval_controller.py`：

```python
""" 100 局评估：报均值与标准差（放在仓库根目录跑） """
import argparse
import numpy as np
import torch
from utils.misc import RolloutGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str)
parser.add_argument('--rollouts', type=int, default=100)
args = parser.parse_args()

gen = RolloutGenerator(args.logdir, torch.device('cpu'), 1000)
scores = []
with torch.no_grad():
    for _ in range(args.rollouts):
        scores.append(-gen.rollout(None))   # rollout 返回负分，取反
print('{:.1f} +/- {:.1f} ({} rollouts)'.format(
    np.mean(scores), np.std(scores), len(scores)))
```

```bash
xvfb-run -s "-screen 0 1400x900x24" python eval_controller.py --logdir exp_dir --rollouts 100
```

随机基线：把脚本里的 `gen.rollout(None)` 临时换成
`gen.rollout(np.random.randn(867) * 0.1)`（每局一组新的随机参数，正好是 CMA-ES
第 0 代的水平），同样跑 100 局。预期：训好的 controller 显著高于随机基线；顺手把
100 个单局分画个直方图，感受一下 5.3 节说的方差。

### Step 6: 写复现报告

一页即可，五个部分：

```text
设置：数据 1000 rollouts（布朗噪声）；V/M 沿用 02/03 课配置；CMA-ES 种群 32、每候选 8 局、初始步长 0.1
差异声明：与论文的已知差异一览（数据 1000 对 10000；种群 32 对 64；评估局数 8 对 16；输入图缩到 64x64；论文每 25 代做 1024 局评估，仓库每 3 代做 100 局）
结果表：随机基线 / 我的 controller / ctallec 项目页 860±120 / 论文 906±21，每行带局数
方向性结论：一句话判断，例如"远超随机基线并进入论文分数区间的同一量级，方向一致"
边界：这份结果能说明什么、不能说明什么；与 906 的差距最可能来自哪笔被砍掉的预算
```

方向性复现和逐点对齐的区别在这里落地：我们的验收问的是"同样的方法在十分之一预算下，
是否重现了'867 参数远超基线、逼近解决线'这个结论"，而不是"906 这个数字有没有再现"。
ctallec 自己用完整配置也只到 860±120，并把差距归因于被调低的 CMA-ES 预算，这条
参照线帮你把预期钉在现实上。

### Step 7: 写梦境环境胶水

在仓库根目录建 `dream_env.py`。结构抄 `utils/misc.py::RolloutGenerator`，接口保持
一模一样（`rollout(params)` 返回负累计奖励），区别是整个世界换成了 M：

```python
""" 梦境环境：把训好的 MDN-RNN 包装成 rollout 生成器（放在仓库根目录） """
from os.path import join, exists
import torch
from torch.distributions import Categorical
from models import VAE, MDRNNCell, Controller
from utils.misc import LSIZE, ASIZE, RSIZE, load_parameters

class DreamRolloutGenerator(object):
    def __init__(self, mdir, device, time_limit, tau=1.0):
        vae_file, rnn_file, ctrl_file = \
            [join(mdir, m, 'best.tar') for m in ['vae', 'mdrnn', 'ctrl']]
        vae_state, rnn_state = [
            torch.load(fname, map_location={'cuda:0': str(device)})
            for fname in (vae_file, rnn_file)]
        self.vae = VAE(3, LSIZE).to(device)
        self.vae.load_state_dict(vae_state['state_dict'])
        self.mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(device)
        self.mdrnn.load_state_dict(
            {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})
        self.controller = Controller(LSIZE, RSIZE, ASIZE).to(device)
        if exists(ctrl_file):
            ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(device)})
            self.controller.load_state_dict(ctrl_state['state_dict'])
        self.device = device
        self.time_limit = time_limit
        self.tau = tau
        self.init_z = torch.load('dream_init_z.pt')  # Step 8 造的真实起点库

    def sample_next_z(self, mus, sigmas, logpi):
        pi = torch.softmax(logpi / self.tau, dim=-1)   # 温度作用点一：logits 除 τ
        k = Categorical(pi).sample().item()
        eps = torch.randn_like(mus[:, k])
        return mus[:, k] + sigmas[:, k] * (self.tau ** 0.5) * eps  # 作用点二：方差乘 τ

    def rollout(self, params):
        if params is not None:
            load_parameters(params, self.controller)
        i0 = torch.randint(len(self.init_z), (1,)).item()
        z = self.init_z[i0:i0 + 1].to(self.device)
        hidden = [torch.zeros(1, RSIZE).to(self.device) for _ in range(2)]
        cumulative, i = 0, 0
        while True:
            action = self.controller(z, hidden[0])
            mus, sigmas, logpi, r, d, hidden = self.mdrnn(action, z, hidden)
            z = self.sample_next_z(mus, sigmas, logpi)
            cumulative += r.item()                     # reward 头发工资
            if torch.sigmoid(d).item() > 0.5 or i > self.time_limit:
                return - cumulative                    # 与真实版同号约定
            i += 1
```

三个设计决定值得停一秒。起点从真实帧的编码里抽（而不是从先验瞎采），因为 M 只在
真实数据分布附近靠谱，起点就离谱等于开局即幻觉。done 用 sigmoid 过 0.5 判，但
CarRacing 的数据里绝大多数局是超时结束的，done 头学到的信号很弱，所以 `time_limit`
才是真正的兜底，这和 Doom 里 done 头当主角的情形正相反。加载 MDRNN 权重那行照抄了
`utils/misc.py` 的键名处理（训练用 `MDRNN`、推理用 `MDRNNCell`，两者参数名差一个
LSTM 后缀）。

### Step 8: 先校准，再训练

进梦训练之前，先用三个检查确认这个梦值得进。

第一，造起点库。用 VAE 把一批真实轨迹的首帧编码成 $z$，存成 `dream_init_z.pt`：

```python
""" 造梦境起点库：真实轨迹首帧经 VAE 编码取均值 """
from glob import glob
from os.path import join
import numpy as np
import torch
from torchvision import transforms
from models import VAE
from utils.misc import LSIZE, RED_SIZE

transform = transforms.Compose(
    [transforms.ToPILImage(), transforms.Resize((RED_SIZE, RED_SIZE)),
     transforms.ToTensor()])
state = torch.load(join('exp_dream', 'vae', 'best.tar'), map_location='cpu')
vae = VAE(3, LSIZE)
vae.load_state_dict(state['state_dict'])
files = sorted(glob('datasets/carracing/**/rollout_*.npz', recursive=True))[:200]
zs = []
with torch.no_grad():
    for f in files:
        obs = transform(np.load(f)['observations'][0])
        _, mu, _ = vae(obs.unsqueeze(0))
        zs.append(mu)
torch.save(torch.cat(zs), 'dream_init_z.pt')
```

第二，对拍 reward 头。取 20 条真实轨迹，把每步的真实帧编码成 $z_t$，沿真实动作序列
逐步调用 `mdrnn(action, z, hidden)`（喂真实 $z$，不喂采样值），收集 reward 头的逐步
预测，和 npz 里存的 `rewards` 列画散点、算相关系数。本课设定的经验线：相关系数高于
0.5 放行；0.2 到 0.5 之间可以继续但预警"梦里的分数只能看趋势不能看数值"；低于 0.2
停下，回 Step 3 检查 mse 是否真的降了、数据是否够。

第三，动作重放看图。从同一个起点出发，把一条真实轨迹的动作序列在梦里重放（Step 7
的类，把 controller 输出换成录好的动作），用 `vae.decoder` 把梦出的 $z$ 序列逐帧
解码，横排拼成图像条。及格标准与第 03 课多步 rollout 的观感一致：前几十步路还是路、
弯还会拐，几百步后糊掉、漂移都正常。第一步就不成画面的，别进 Step 9。

### Step 9: 在梦里进化，回真实环境验收

复用 `traincontroller.py` 的整个 CMA-ES 框架，只换掉"世界"：

```bash
cp traincontroller.py traincontroller_dream.py
```

对 `traincontroller_dream.py` 动三处：把 `from utils.misc import RolloutGenerator`
换成 `from dream_env import DreamRolloutGenerator`；argparse 加一个 `--tau`
（float，默认 1.0）；`slave_routine` 里建 `RolloutGenerator` 的那行换成
`DreamRolloutGenerator(logdir, device, time_limit, tau=args.tau)`。其余的队列、
种群循环、checkpoint 逻辑一个字不用动，这是把梦境包装成同接口环境的回报。

```bash
python traincontroller_dream.py --logdir exp_dream --n-samples 4 --pop-size 32 --target-return 900 --tau 1.0
```

不需要 xvfb（没有任何渲染），Mac 也能跑。预期：每代从真实环境的几分钟缩到秒级，
几小时内梦里评估分明显上行，感受一下"在想象里训练便宜一个量级"是什么体感。
练到平台期后，回真实环境验收：

```bash
xvfb-run -s "-screen 0 1400x900x24" python eval_controller.py --logdir exp_dream --rollouts 100
```

`exp_dream` 里现在恰好凑齐了三件：VAE、带 reward 头的 M、梦里练出的 controller，
`RolloutGenerator` 直接加载。记下两个数：梦里 100 局均值（训练日志的 evaluate 输出）
和真实 100 局均值。

### Step 10: 温度扫描，写移植实验报告

给 τ=0.1 和 τ=1.3 各开一个独立目录，避免 controller 互相覆盖：

```bash
cp -r exp_dream exp_dream_t01
```

```bash
rm -rf exp_dream_t01/ctrl
```

τ=1.3 同理建 `exp_dream_t13`。然后各自跑 Step 9 的训练命令（换 `--logdir` 和
`--tau`）加真实环境验收，把六个数填进一张三行表：每个 τ 一行，"梦里 100 局均值
±方差"和"真实 100 局均值±方差"各一列。

预先写下失败判据，跑完对号入座：

- 判据 A（钻空子实锤，这是预期中的"成功的失败"）：τ=0.1 组梦里均值比真实均值
  高 300 分以上（本课经验线），且真实分不高于随机基线。诊断动作：解码几条低 τ 梦境
  rollout 看图，找"梦里永远不出弯、压不到草"之类的幻觉福利。
- 判据 B（梦不可训）：某个 τ 下梦里评估分 30 代不上行。诊断路径：先查 Step 8
  的 reward 相关系数，再查是不是忘了 `--include_reward`（训练日志里 mse 恒为零就是
  铁证），最后查起点库是否正常。
- 判据 C（方向不符）：三个 τ 的"梦真裂缝"没有随 τ 降低而变大，甚至反向。这
  说明在我们的设置里温度不是主导变量，移植实验失败，但报告照写：把 M 的容量、
  数据覆盖、reward 头噪声三个嫌疑逐个排查的记录写进去，这份诊断和成功的表格同样
  是合格交付。

移植报告的骨架：

```text
梦境设置：M 带 reward 头重训；起点库 200 条；时间上限 1000 步；done 阈值 0.5 兜底靠超时
校准记录：reward 相关系数数值、动作重放图像条的观感结论
温度表：tau 0.1 / 1.0 / 1.3 三行，梦里与真实各 100 局均值±方差
方向性判定：梦真裂缝是否随 tau 降低而放大，与论文 Doom 表（0.1 时 2086 对 193，1.15 时 918 对 1092）方向是否一致
失败判据核对：A/B/C 各自是否触发，触发后的诊断记录
差异提醒：任务不同（赛车对躲火球）、reward 来源不同（学出来的头对免费的存活计数）、规模不同；一切结论只谈方向，不谈数值
```

## 8. 配置与预算

| 阶段 | 配置 | 主要吃什么 | 参考耗时 |
|---|---|---|---|
| 数据补齐（Step 1） | 1000 rollouts、8 线程 | CPU 多核 + 磁盘 | 数小时 |
| M 带 reward 重训（Step 3） | 仓库默认超参加 `--include_reward` | 单卡 GPU | 与第 03 课相当，数小时 |
| 真实环境 CMA-ES（Step 4） | 种群 32 × 每候选 8 局 = 256 局/代 | CPU 多核 | 数天（挂机为主） |
| 论文原始配置（不要求跑） | 种群 64 × 16 局 = 1024 局/代，每 25 代评 1024 局 | 当年的多机 CPU | 复现档的四倍以上，只算账 |
| 梦里进化（Step 9） | 种群 32 × 每候选 4 局，全程无渲染 | CPU 即可 | 每个 τ 数小时 |
| 温度扫描（Step 10） | 三个 τ 各一轮梦训 + 100 局真实验收 | CPU | 一天上下 |

三条预算心得。第一，真实环境 CMA-ES 的时间几乎全花在环境模拟和渲染上，GPU 全程
接近围观，这是"controller 阶段吃 CPU 多核"的原因，也解释了为什么梦境训练快
两个数量级：一步 LSTMCell 前向对一步物理模拟加渲染。第二，Step 3（GPU）和 Step 4
（CPU）并行开跑，等待期正好用来写 Step 7 的胶水。第三，评估局数的账要提前算：
Step 10 的真实验收是 3×100 局，每局最长 1000 步，纯 CPU 单进程要跑几个小时，
嫌慢可以把 `eval_controller.py` 里的 device 换成 GPU 或仿照 `traincontroller.py`
开多进程，但报告里的局数不许砍。

## 9. 验收

第一幕验收清单：

- [ ] 复现报告完整：设置、差异声明、四行结果表（随机基线、你的 controller、ctallec
      860±120、论文 906±21，每行带局数）、方向性结论、边界各就各位；
- [ ] 你的 controller 100 局均值显著高于随机基线，且达到本课设定的方向线 700 分
      （这是按 ctallec 完整配置 860 打了预算折扣后的课程验收线，够不到先走第 10 节）；
- [ ] 能口头说清"方向性复现"和"逐点对齐"的区别，并用自己的差异声明举例；
- [ ] 梦境校准三件套齐全：起点库、reward 相关系数、动作重放图像条，且相关系数过了
      放行线才开始的梦训；
- [ ] 温度表三行六格填满，每格都是 100 局统计；
- [ ] 方向性判定有明确结论：梦真裂缝随 τ 降低而放大（与论文 Doom 表同方向），或者
      判据 C 触发并附诊断记录，两者都算合格，含糊其辞不算；
- [ ] 眼见为实：留一张 τ=0.1 的梦境解码图像条，指得出策略在梦里占了什么幻觉便宜
      （或说明没找到，及原因猜想）；
- [ ] 能口头回答三个问题：论文为什么把梦境训练放在 Doom 而不是 CarRacing（提示：
      reward 从哪来）；`test_controller.py` 为什么当不了验收工具；`--include_reward`
      这个坑是怎么回事。

## 10. 排错

| 症状 | 最可能的原因 | 怎么确认 | 怎么修 |
|---|---|---|---|
| Step 4 跑了一天，最优分还在负数徘徊 | V 或 M 质量差，controller 读到的简报是乱码 | 回第 02/03 课的验收：V 重建认不认得出路、M 损失降没降 | 先修上游再回来；上游没问题就查 n-samples 是否太小（噪声淹没排序） |
| 最优分爬到三四百进入死平台 | 预算不够或上游天花板 | 看每 3 代的 100 局评估是否还有缓慢爬升；对照直方图看方差 | 有缓升就继续挂机；彻底走平可加大 pop-size 重跑，或接受并在报告里写明 |
| 训练进程无输出且不报错 | 无显示环境下没用 xvfb，worker 静默失败 | 看 `exp_dir/tmp` 里的 worker 日志（README 明说日志在这里） | 所有真实环境命令套上 `xvfb-run` |
| worker 报 GPU 显存不够 | 每个 worker 都往 GPU 装了一套 V+M | 报错栈里有 cuda malloc | 调低 `--max-workers`（README 给的正是这个用途），或让 eval 用 CPU |
| 梦里分数 30 代不上行 | reward 头没训（忘加开关）或信号太弱 | 训练日志里 mse 恒为零就是没加 `--include_reward`；再看 Step 8 相关系数 | 回 Step 3 重训 M；相关系数低就加数据或接受判据 B |
| 梦里一局只走几步就结束 | done 头输出漂，sigmoid 常越阈值 | 打印每步 `torch.sigmoid(d)` 的值看分布 | 阈值提到 0.9，或干脆忽略 done 只用 time_limit，CarRacing 的 done 本来就近乎全靠超时，报告里注明即可 |
| 梦里分数正常上行，真实验收却和随机基线持平 | 钻空子（判据 A），或起点分布与真实开局不符 | 解码梦境 rollout 看图找幻觉福利；检查起点库是不是真用首帧造的 | 提高 τ 重训；这本身是需要的现象，截图存报告 |
| τ=1.3 梦里完全学不动 | 噪声太大，信号被淹 | 对比 τ=1.0 的上行曲线 | 正常现象，对应论文表里 1.30 那行的回落；写进方向性判定 |
| `strip('_l0')` 那行加载 MDRNN 报键名错 | 你的 checkpoint 是 MDRNN 训的，键名带 LSTM 后缀不规则 | 打印 `rnn_state['state_dict'].keys()` 对照 | 照 `utils/misc.py` 的原样写法改键名映射，两边保持一致 |

## 11. 前沿与改造

你这课手搓的"在学出来的模型里训练策略"，正是后面半门课的主旋律，
但 2018 年这套裸奔版的三个软肋都被现代系统换过零件。其一，整局做梦：我们从起点一路
梦到超时，误差滚一千步；DreamerV3（第 06 课）只从真实数据的状态出发做 15 步左右的
短梦，用价值函数接住剩下的未来，误差还没滚起来梦就醒了。其二，事后加噪：τ 是训完
才拧的旋钮；RSSM（第 05 课）把随机性做进状态本身，KL 项在训练时就逼着模型对没把握
的地方保持分布。其三，全信模型：我们把梦当真实环境用；MuZero 和 TD-MPC2（第 07、08
课）只在短规划窗口里咨询模型，从不让策略在里面长住。还有一支干脆绕开梯度问题的路线
被我们用了，CMA-ES 这类进化方法今天仍是小参数量策略头的实用选择，但 Dreamer
证明了想象里可以直接反传梯度训大得多的 actor，第 06 课见。

规模一半（1000 条对 10000 条数据、32 对 64 种群，钱和时间能解决），
机制一半（长梦、事后加噪、全信模型，分别是 05、06 课的正题）。分清这两半，是第
17-19 课研究手艺的起点。

动手改造清单（选做）：

1. τ 细扫：在 0.1 和 1.3 之间补 0.5 和 1.15 两个点（各一轮梦训加验收，预算约
   一天 CPU）。预期：真实分对 τ 呈先升后降的拱形，最高点未必在 1.15，那是 Doom
   的数字，不是定律。失败判据：五个点单调或杂乱无章，回判据 C 的诊断路径。
2. 梦训热启动真实训练：改 `traincontroller.py` 约一行，初始化时用
   `load_parameters` 把 `exp_dream/ctrl/best.tar` 的参数装进那个 dummy controller
   再展平，CMA-ES 的初始均值就成了梦里练出的手。预算：改码半小时加一轮 Step 4。
   预期：到达相同分数所需的真实 rollout 数明显少于冷启动。失败判据：不省预算，
   说明梦里学的东西迁移不过去，把这个结果和 τ 的选择联系起来分析。
3. 随机 M 的照妖镜：把 `exp_dream` 的 mdrnn 换成随机初始化的 checkpoint，重跑
   Step 8 校准（预算一小时）。预期：reward 相关系数塌到零附近，动作重放图像条不成
   画面，ctallec 项目页"随机 M 真实环境照样 870"的反直觉结论，在模拟器用途上
   当场破产。这一对照把"h 当特征够用"和"M 当世界不够用"钉在一起。
4. Doom 式 reward 移植：把 `dream_env.py` 的 `cumulative += r.item()` 换成
   每步加 1（活着就算分），梦训一轮（预算数小时）。预期：学出"苟活流"，真实
   验收分数崩，因为 CarRacing 的分来自压格子而不是活着。用最便宜的方式证明：
   梦境训练的成败，一半押在 reward 信号的来源上。

论文 Doom 温度表的核心结论"低 τ 的梦里高分是假账、适度高 τ 用梦里
低分换真实高分"，映射到本课就是 Step 10 的三行表。方向能对上，你就用一个换了任务、
换了 reward 来源的设置旁证了它的稳健性；对不上，你收获的是一份关于"这个结论依赖
什么前提"的一手证据，两头都是赚。

## 12. 论文与延伸

1. World Models（Ha & Schmidhuber, 2018，[arXiv:1803.10122](https://arxiv.org/abs/1803.10122)）
   ，这次精读 Doom 那一章和温度实验。带着三个问题：Take Cover 的 reward 在梦里
   为什么不需要 reward 头？τ 从 1.15 再往上抬，换来了什么、亏掉了什么？文中 agent
   骗过怪物不发火球那段，用本课的话说是钻了哪类空子？
2. 交互版论文 [worldmodels.github.io](https://worldmodels.github.io)，重点玩
   调 τ 看梦境变疯的演示；第 01 课玩过一遍，这次你能看懂它每个滑块背后的公式了。
3. The CMA Evolution Strategy: A Tutorial（Hansen，
   [arXiv:1604.00772](https://arxiv.org/abs/1604.00772)），不用通读，带着三个问题
   翻：步长 σ 和协方差 C 为什么分开自适应？为什么只用排序不用分数值，这对噪声适应度
   意味着什么？维度升到多少时该换别的方法？
4. hardmaru/WorldModelsExperiments 的 `doomrnn/doomrnn.py`（只读对照），
   把 `DoomCoverRNNEnv._step` 和你的 `dream_env.py` 并排放：温度的两个作用点、done
   的处理、reward 的来源，逐项打钩或标不同。这是检验你真读懂了梦境环境的最快方式。
5. ctallec 的项目页 [ctallec.github.io/world-models](https://ctallec.github.io/world-models/)
   ，带着问题读：860±120 的差距被归因于什么？"随机初始化的 M 得分 870"这个实验
   对 5.1 节的读出头哲学、对你的梦境实验，分别意味着什么？
6. 选读：**Learning to Drive in a Day**（Kendall et al., 2018，
   [arXiv:1807.00412](https://arxiv.org/abs/1807.00412)），Wayve 把"VAE 压缩加小
   策略头"开上了真实公路。带着问题读：真车上最贵的资源是什么？如果给他们一个
   本课这样的梦境环境，最想省掉的是哪一步？

第一幕到此收官。盘点一下手里的东西：一台完整复现的 2018 年世界模型，V 压缩、M
预测、C 决策，真实环境的分数带着局数和方差写进了报告；外加一次的梦境移植，你
见过策略钻模型空子的现场，也知道了往梦里掺噪声这味解药的剂量讲究。但这套三件套有
一条与生俱来的裂缝：V 压缩时根本不知道 M 要预测什么，M 预测时也改不动 V 的压缩，
两个器官各训各的，中间只靠 32 个数传话。第 05 课的 RSSM 把它们焊成一台
机器：状态一半确定、一半随机，压缩和预测在同一个损失里互相校准，"梦"从此成为
训练的一等公民。第一幕用的零件是 2018 年的，从下一课起，我们开始换现代引擎。
