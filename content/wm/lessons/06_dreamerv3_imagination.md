---
id: 06_dreamerv3_imagination
title: "DreamerV3 的想象训练"
summary: "为什么在想象里训练策略比在真环境里训便宜一个量级？一套超参凭什么通吃 150+ 任务？"
unit: engine
play_tools: []
checkpoints:
  - "单任务复现曲线（论文复现 #3）。"
  - "稳定性技巧消融小报告：每个细节各救了什么。"
---

# 第 06 课：DreamerV3 在想象中训练 actor-critic

> 类型：复现（论文复现 #3：DreamerV3 单任务方向性复现）<br>
> 建议周期：4-7 天（大头是机器在跑，人的操作集中在开跑前半天和收尾半天）<br>
> 硬件：单张 24GB 卡；基线一条 1M 步跑数小时到一两天，每条消融再短一半以上；Mac/纯 CPU 只能跑 debug 冒烟档和读代码<br>
> 锚定仓库：[NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch)（PyTorch，2026-07-05 归档只读，教学锚定版），对照 [danijar/dreamerv3](https://github.com/danijar/dreamerv3)（JAX 官方，Nature 2025 论文配套）<br>
> 产物：dmc_walker_walk 复现曲线（与仓库 README 公布曲线方向性对照）+ 五个稳定性技巧的有/无消融小报告

## 1. 这一课做什么

前几课留下了两种限制。World Models 先采一批数据、训练并冻结模型，再在固定模型里
优化 controller；一旦策略走到旧数据没有覆盖的状态，模型就容易失真。PlaNet 的 CEM
则每一步都重新搜索大量动作序列，推理成本高，搜索结果也不会沉淀成一个可直接执行的
策略。

DreamerV3 用一个持续更新的循环处理这两个问题：**世界模型、actor、critic 一起训练**。
actor（策略网络：看着状态直接吐动作，不用现场搜）在真环境里边跑边
把轨迹存进 replay buffer；世界模型不断从 buffer 里抽数据更新自己；actor 和 critic
的训练则**完全在世界模型的想象里进行**，从 buffer 里的真实状态出发，让模型往前滚
15 步虚拟轨迹，策略梯度全部吃这些虚拟转移，一步真实交互都不用。策略变好，采回的
数据就把 buffer 带到新地界，模型跟着刷新见识，想象变得更可信，策略再变好。第 04 课
"训完 M 再训 C"的两段式因此变成在线循环，策略变化带来的新数据也会继续更新模型。

本课除了复现 walker_walk，还会分别移除 symlog、twohot、KL balancing、free bits 和
return normalization，记录训练稳定性与回报曲线的变化。这些处理不是 Dreamer 独有的
装饰项，而是现代世界模型和强化学习代码中常见的数值设计。结果会作为第 08 课对比
TD-MPC2、以及第 18 课缩放实验的基线。

术语速查：

| 术语 | 一句人话 |
|---|---|
| actor / critic | 手和账房：actor 看着状态直接输出动作分布；critic 估"从这个状态往后还能拿多少分"，给 actor 的每一步定功过 |
| replay buffer | 数据仓库：真环境跑出的轨迹进仓，训练时随机抽片段；新旧数据混着住 |
| 想象训练（latent imagination） | 从 buffer 里的真实状态出发，世界模型往前滚 15 步虚拟轨迹，actor-critic 只吃这些虚拟转移 |
| train_ratio | 重放节奏旋钮：平均每个环境步要重放多少帧数据用于训练；它决定"真实交互换想象训练"的汇率 |
| λ-return | critic 的训练目标：把想象里的多步奖励和 critic 自己的尾巴按 λ 加权混出来，忘了的话回第 04 课看 CMA-ES 时代我们连它都没有 |
| symlog | 对称 log 压缩：$\mathrm{sign}(x)\ln(1+\lvert x\rvert)$，大数压扁、小数几乎不动、负数对称处理 |
| twohot | 软分类回归：连续目标劈给相邻两个桶，按距离分权重；回归变成 255 类交叉熵 |
| KL balancing | 同一个 KL 写两遍、两头不同力度：先验使劲追后验，后验只轻微向先验靠 |
| free bits | KL 的保底条款：低于 1 nat 就不再产生梯度，防表征被压塌 |
| return normalization | 把优势除以回报的 5%-95% 分位距（且除数不小于 1），让熵正则那 3e-4 在任何奖励量级下都说得上话 |
| unimix | 类别分布里掺 1% 均匀噪声，防某个类的概率钉死在 0 上 |
| 开环预测（open-loop） | 只给模型开头几帧真图，之后全靠它自己往下播，检验想象质量的照妖镜 |

## 2. 问题

三个问题，外加一条界限。

1. 为什么在想象里训练策略比在真环境里训便宜一个量级？这不是一句口号，是一笔
   能精确算出来的账：配置下，每一次梯度更新，actor-critic 消费 15360 个想象
   转移，而同期真实环境只走了 2 步。第 5.2 节把这笔账连同它的隐藏成本（模型误差、
   墙钟时间）一起算清，并用 `train_ratio` 消融让你拨这个汇率旋钮。
2. 一套超参凭什么通吃 150+ 任务？不同任务的奖励量级差好几个数量级、回报分布
   有的平滑有的暴烈，按老规矩每换个任务就得重调学习率和熵系数。DreamerV3 的答案
   是五个稳定性零件，各堵一种翻车方式。重点就是把它们逐个拆下来看车怎么翻：
   symlog（5.3）、twohot（5.4）、KL balancing 加 free bits（5.5）、return
   normalization（5.6），每个都设计了改真实配置键的有/无对照。顺带点破一个容易被
   宣传掩盖的细节：打开 `configs.yaml` 你会看到，所谓"一套超参"固定的是学习率、
   KL 系数、熵系数这些**学习超参**；模型尺寸、train_ratio、动作重复这些**预算超参**
   在不同任务组之间仍然不同（Crafter 配置把 `dyn_deter` 开到 4096，DMC 用 512）。
   这不是造假，是论文本来的主张，但你得看到边界在哪。
3. 在线互喂的循环怎么治分布偏移？治到什么程度？第 01 课埋的病根在这课得到
   系统性缓解：数据不再是离线采一次，策略把数据分布带去哪，模型就跟到哪。但 buffer
   里旧数据还在、模型永远慢策略半拍，"策略专挑模型学错处钻"只是被压制，没有根除
   ，第 5.1 节讲清这半味药的药理和残余病灶。

界限先划好：本课是**单任务方向性复现**（walker_walk 一条 1M 步曲线，与仓库 README
公布曲线对照走向和量级），消融用的是缩短预算加单随机种子。消融结论只够说"拆掉这个
零件后在这个任务上观察到什么方向的变化"，不够说"这个零件在所有任务上值多少分"，
后者是论文用 150+ 任务和多种子买来的话语权，我们第 18 课再学怎么花小钱说大话不翻车。

## 3. 准备

- 上一幕的债：第 04 课的两段式梦境训练和"钻空子"现场必须做过，否则这课
  的在线循环你只能看到"它好"，看不到"它治了什么"。第 05 课的 RSSM 双通道拆解
  最好也完成了，这里不再重讲 $h_t$ 与 $z_t$ 的分工，参见第 05 课。
- 硬件：单张 24GB 卡够全程；显存实际占用远小于 24GB，但 1M 步的基线要在卡上
  连续跑数小时到一两天，确认机器能挂机。Mac/纯 CPU 可以完成克隆、读码、debug
  冒烟档（几分钟出日志），跑不动正式训练。
- 软件：仓库 README 声明依赖在 python 3.11 下测试，给它单开一个 3.11 的虚拟
  环境，别在老环境上硬装。DMC（dm_control 物理仿真套件）走 `requirements.txt`
  就装上了；Atari/Minecraft 才需要 `envs/` 目录里的安装脚本，这里用不到。
- 无头服务器：dm_control 渲染需要 EGL 或类似后端，远程机器上先确认渲染可用
  （报 `glGetError` 相关错就是它，修法见第 10 节；仓库根目录留了 `xvfb_run.sh`）。
- 磁盘：replay buffer 会以 npz 片段落盘，`dataset_size: 1000000` 帧的 64×64
  图像加上多条消融 run 的日志，预留 30GB 不算奢侈。
- 心理预期：这课人手操作不多，机器时间很多。正确姿势是把基线挂上就去精读源码
  （第 6 节的导览就是给挂机时间准备的），别守着刷屏。

## 4. 学习目标

1. 白纸画出 DreamerV3 的在线循环数据流：真环境、replay buffer、世界模型、actor、
   critic 五个节点之间各流动什么张量、各自的更新节奏由哪个配置键控制；
2. 现场算出"一步真实交互换多少步想象训练"这笔账，并说出账面便宜背后的两项隐藏
   成本；
3. 默写 symlog 和 symexp 的公式，解释为什么 critic 回归的是回报而不是单步奖励这一
   事实让 symlog 在 DMC 上也不可省；
4. 解释 twohot 离散回归比 MSE 稳在哪，以及 255 个桶配上 symlog 变换能覆盖多大的
   数值范围；
5. 说出 `kl_free`、`dyn_scale`、`rep_scale` 三个键各自动谁的梯度、防哪种病，并从
   TensorBoard 的 `kl` 曲线上认出 free bits 生效的形状；
6. 解释 return normalization 为什么用分位距而不是标准差、除数为什么有个"不小于 1"
   的底，各防什么翻车；
7. 拿着自己的消融表，对每一行说一句"拆掉它，症状是什么、先出现在哪个指标上"。

## 5. 原理

六个机制。前两个讲循环本身（结构与账），后四个讲让循环在任何任务上都不散架的
稳定性零件。每个机制都按老规矩五步走：直觉、机制、数学、代码落点、验证，验证
一律绑定第 7 节的具体实验编号。

### 5.1 从"训完再用"到"在线互喂"：分布偏移的半味药

第 04 课的两段式像函授教育：教材（数据）印好之后就再也不更新，学生（策略）
水平涨上去之后，教材覆盖不到的地方只能靠脑补，还专挑教材印错的地方钻。在线互喂
改成跟班师傅：徒弟今天开到哪条新路，师傅明天就把那条路的经验补进教案。数据分布
跟着策略走，模型的见识永远吊在策略身后不远处。

`dreamer.py` 的主循环里，三件事交替发生。第一，**采集**：actor 在真环境
里执行（连续控制任务的动作从一个学出来的高斯分布采样，天然带探索噪声），轨迹一段段
写进磁盘上的 replay buffer；开局前先用随机策略灌 `prefill: 2500` 步垫底，不然第一批
训练没米下锅。第二，**训世界模型**：从 buffer 均匀抽 `batch_size: 16` 条、每条
`batch_length: 64` 步的片段，更新 RSSM 和三个预测头（重建、奖励、继续标志）。第三，
训 actor-critic：拿刚才那批片段过完 RSSM 得到的每一个后验状态当起点，在想象里
滚 15 步，只用这些虚拟转移更新 actor 和 critic。三件事的节奏由一个数字锁定：
`tools.Every(batch_steps / train_ratio)`，其中 `batch_steps = batch_size × batch_length
= 1024`，dmc_vision 的 `train_ratio: 512` 代入，就是**每 2 个环境步做一轮完整的
模型加策略更新**。

两段式的病可以写成一行：模型在分布 $d_{\pi_0}$（初始采数据策略的状态
分布）上训练，却要在 $d_{\pi^*}$（训练后策略的状态分布）上被使用，两个分布的错位
随策略改进单调扩大。在线循环把训练分布换成 buffer 混合分布
$d_{\text{buffer}} = \frac{1}{K}\sum_k d_{\pi_k}$，历代策略状态分布的平均。错位仍然存在（最新策略只占
buffer 的一小份），但不再随时间扩大：策略每往新地界走一步，新数据就把 buffer 往那边
拽一点。这是为什么说它是"半味药"：治好了**错位发散**，治不掉**滞后**，模型
永远比策略慢半拍，第 04 课那种钻空子行为仍会小规模发生，只是每次都会被下一批真实
数据纠正，形成"钻空子、被打脸、改邪归正"的小周期。

`dreamer.py`：`Dreamer.__init__` 里的 `self._should_train =
tools.Every(batch_steps / config.train_ratio)` 是节奏器；主函数里先用随机 agent 走
`tools.simulate(...)` 完成 prefill，然后 `while agent._step < config.steps +
config.eval_every` 的大循环里交替评估和"边采边训"，采集时每次调用 agent 本身就
会按节奏触发 `self._train(next(self._dataset))`。

绑定第 7 节 Step 3 和 Step 4：基线跑起来后，在 TensorBoard 同屏摆
`reward_loss` 与评估回报。离线训练的损失是单调下降的；在线互喂的模型损失会周期性
抬头，每次策略闯进新地界，模型就"重新变笨"一小段，然后压回去。看到这个锯齿，
你就看到了循环在转。反面对照是 Step 5 的 E1：把 `train_ratio` 砍到 128，模型和策略
的互喂节奏放缓四倍，同样环境步数下曲线应该明显爬得更慢。

### 5.2 想象训练的账：一步真实交互换 7680 步想象

策略梯度是出了名的吃样本：信号是"这一整串动作最后拿了多少分"，噪声
巨大，得靠海量转移平均掉。真环境一步的成本是物理仿真加渲染，想象一步的成本只是
一次 GRU 前向，既然 RSSM 已经把世界的动力学学在手里，凭什么还让真环境陪策略
一遍遍试错？DreamerV3 的分工是：真实交互只负责**喂世界模型**（覆盖"世界怎么动"），
策略试错的天文数字全部转嫁给想象。

每轮更新，从 replay 抽出的 16×64 = 1024 个真实后验状态全部当作想象的
起跑线。`ImagBehavior._imagine` 从每个起点出发：actor 看着当前想象状态出动作（特征
做了 detach，想象里策略的梯度不回流进世界模型），RSSM 的 `img_step` 用**先验**推进
一步（没有真实观察可看，这正是第 05 课先验通道存在的意义），如此滚
`imag_horizon: 15` 步。critic 在这些虚拟轨迹上算 λ-return 当训练目标，actor 朝
高 λ-return 方向更新，连续控制任务的梯度直接顺着可微的想象动力学反传回动作
（`imag_gradient: 'dynamics'`），离散任务（Crafter/Atari 配置）则换成 REINFORCE
估计（`imag_gradient: 'reinforce'`）。

账本如下。每轮更新的想象转移数：$1024 \times 15 = 15360$。更新频率：
每 $1024 / 512 = 2$ 个环境步一轮。所以**每个环境步摊到 $15360 / 2 = 7680$ 个想象
转移**，actor-critic 的训练流量是真实交互流量的近四个数量级。反过来读同一笔账：
如果这些梯度全要真环境买单，同样的策略训练量需要多走 7680 倍的环境步。这是
"便宜一个量级"的保守说法的来历，实际汇率取决于 train_ratio 怎么拨。两项隐藏
成本也要入账：其一，想象转移不是真转移，模型误差会给策略喂偏见，所以 horizon 只敢
开 15 步（第 03 课你量过误差滚雪球的速度，这里是同一条曲线在管账）；其二，账省的是
环境交互（样本效率），不是**墙钟时间**，每 2 个环境步塞一轮 1024 帧的训练，
GPU 是一刻不闲的，train_ratio 越高墙钟越慢。

`models.py::ImagBehavior._imagine`：`tools.static_scan(step,
[torch.arange(horizon)], (start, None, None))` 就是那 15 步循环，`horizon` 由
`_train` 传入的 `self._config.imag_horizon` 决定；λ-return 在 `tools.py::lambda_return`
（注释原话：λ=1 是折扣蒙特卡洛回报，λ=0 是一步回报，配置用 `discount_lambda: 0.95`、
`discount: 0.997`）。

绑定 Step 5 的 E1：`--train_ratio 128` 与基线 512 对照，同样跑到 3e5
环境步。预期方向：128 的曲线在同一环境步数下落后（每步真实交互兑换的训练量少了
四分之三），但墙钟更快。顺手把两条 run 的"环境步数-墙钟时间"记下来，你就有了一张
自己的汇率表：样本效率和算力时间在 train_ratio 这个旋钮上互相买卖。

### 5.3 symlog：把所有量级的数换算成同一种货币

网络回归一个目标，梯度大小跟着目标大小走：目标是 3，误差量级是个位数，
学习率 1e-4 正好；目标是 30000，同一个学习率直接把权重拽飞。跨 150+ 任务用一套
超参，第一件事就是保证**进网络的数和网络要预测的数，量级都被摁在同一个区间里**。
symlog 就是那台货币兑换机：像 log 一样把大数压扁，但对小数几乎不动手（零点附近
斜率是 1，不像 log 在零点爆炸），负数按对称规则同样处理。

所有回归型的头都在 symlog 空间工作：网络预测 symlog 后的目标，读出时用
反函数 symexp 换回原币种。向量观察进 encoder 前也先过 symlog（`symlog_inputs:
True`），向量重建头用 `symlog_mse`。图像不用，像素本来就在固定区间里。



$$
\mathrm{symlog}(x) = \mathrm{sign}(x)\,\ln\!\big(1 + |x|\big)
$$

$$
\mathrm{symexp}(x) = \mathrm{sign}(x)\,\big(e^{|x|} - 1\big)
$$

两个函数互为反函数，都过原点、都单调、零点附近近似恒等。这里有个最容易想岔的点：
"DMC 每步奖励本来就在 0 到 1 之间，symlog 是不是白装了？"错在忘了 critic 回归的
不是单步奖励，是**回报**。折扣 $\gamma = 0.997$ 下，walker_walk 这种每步奖励接近 1
的任务，价值量级大约是 $1/(1-\gamma) \approx 333$；symlog 把它压到
$\ln(334) \approx 5.8$，稳稳落在网络舒服的区间。symlog 吞下的不是奖励的量级，是
回报的量级，这也是为什么它在"看起来奖励很温和"的任务上照样不能省。

`tools.py` 顶部两行：`symlog(x) = torch.sign(x) * torch.log(torch.abs(x)
+ 1.0)` 与对应的 `symexp`；`DiscDist` 构造函数默认 `transfwd=symlog, transbwd=symexp`
（往下 5.4 就讲它）；`networks.py` 的 MLP 里 `if self._symlog_inputs: x =
tools.symlog(x)`。

绑定 Step 5 的 E2：这个消融配置文件里没有现成开关（`networks.py` 的
MLP 头不提供"无 symlog 的普通 MSE"选项），所以动一次两行胶水补丁，把 `tools.py`
里 `DiscDist` 的 `transfwd`、`transbwd` 默认值换成恒等函数。后果立竿见影：桶的覆盖
范围从"symlog 空间的 $[-20, 20]$，等效原币 $\pm(e^{20}-1) \approx \pm 4.8$ 亿"
缩水成"原币 $[-20, 20]$"，而 walker 的真实回报量级是 333，远在最右桶之外。预期：
critic 的 `value` 统计量顶死在 20 附近、`target` 却在它头上，评估回报明显低于基线
，你会看到"回归目标超出表示范围"这种翻车长什么样。

### 5.4 twohot：把"猜一个数"变成"投 255 个桶"

就算量级摁住了，MSE 回归还有两个老毛病。一，它的梯度和误差成正比：
遇到重尾的回报分布，一个离群大目标就是一记重锤，把网络砸得晃三晃。二，它只能输出
单峰的"平均数"：想象轨迹里"要么拿分要么归零"的双峰局面，MSE 会预测一个哪边都
不是的中间值。分类没有这两个毛病，交叉熵的梯度天然有界（概率顶多从 0 走到 1），
softmax 天然能表达多峰。twohot 就是把回归伪装成分类：值域切成 255 个桶，连续目标
劈给最近的两个桶，权重按远近分，所以叫"两热"，是 one-hot 的连续版。

critic 和奖励头不再输出一个标量，而是输出 255 维 logits。训练时把目标值
（先过 symlog）编码成 twohot 向量当交叉熵的软标签；推断时对桶中心求概率加权平均，
再 symexp 回原币。这个编码是**保均值**的：twohot 向量对桶中心的期望恰好还原原目标，
所以离散化不引入系统性偏差，只引入桶宽以内的量化误差。

桶中心 $b_1 < \cdots < b_{255}$ 等距铺在 symlog 空间的 $[-20, 20]$ 上。
目标 $y$（已 symlog）落在 $[b_k, b_{k+1}]$ 之间时，twohot 标签为：

$$
t_k = \frac{b_{k+1} - y}{b_{k+1} - b_k}, \qquad t_{k+1} = 1 - t_k
$$

其余分量为零。损失是交叉熵 $-\sum_i t_i \log p_i$，对 logits 的梯度是 $p - t$，
每个分量绝对值不超过 1，这是"离群目标砸不飞网络"的代数原因：目标再大，也
只是换了个桶投票，锤子的分量不变。

`tools.py::DiscDist`：`self.buckets = torch.linspace(low, high,
steps=255, device=device)`，默认 `low=-20.0, high=20.0`；`log_prob` 里
`F.one_hot(below, ...) * weight_below[..., None]` 加上 `above` 的对称项就是上面那
两行公式。挂载点在 `networks.py::MLP.dist` 的 `"symlog_disc"` 分支；配置里
`reward_head` 和 `critic` 的 `dist: 'symlog_disc'` 就是它。

绑定 Step 5 的 E3：把 `configs.yaml` 里 `reward_head` 和 `critic` 的
`dist` 从 `'symlog_disc'` 改成 `'symlog_mse'`（`MLP.dist` 支持的合法取值，改法见
Step 5），保留 symlog、只摘掉离散回归，干净地隔离 twohot 的贡献。预期要诚实：
walker_walk 奖励稠密平滑，正是 MSE 最舒服的地形，两条曲线可能接近；论文的消融
（读第 12 节第 1 篇时对照）显示这类零件的价值集中在奖励稀疏、回报暴烈的任务上。
单任务看不出差距本身就是个合格结论，所谓"一套超参通吃"，买的是跨任务的保险，
不是单任务的加速。手头宽裕的话，把同一对照搬到 Crafter 配置上再跑一遍（成就奖励
离散且稀疏），差距应当更可见。

### 5.5 KL balancing + free bits：两根缰绳拴一头双头兽

第 05 课讲过 RSSM 的 KL 项拴着两个分布：后验（看着当前观察猜状态）和
先验（不看观察硬猜，即世界模型本体）。这一个 KL 有两种死法。死法一：**先验学不动**。
KL 的梯度同时拽两头，而"后验装傻凑合先验"比"先验真学会预测"容易得多，表征
迁就一个烂先验，重建和奖励头跟着烂；更糟的是想象全靠先验推进（5.2 讲过），先验烂
等于梦全是错的。死法二：**KL 被压到零**。后验彻底塌成先验，观察信息进不了状态，
模型闭眼开车。两根缰绳各治一种死法：KL balancing 决定**两头各用多大力气互相追**，
free bits 决定**追到多近就收手**。

balancing 把同一个 KL 抄成两份：dyn 份对后验做停梯度（stop-gradient），
只有先验挨训，负责"先验追后验"；rep 份对先验做停梯度，只有后验挨训，负责"后验
向先验适度靠拢"（表征太任性，先验也确实追不上）。两份的权重是 0.5 比 0.1，先验
追得紧，后验只被轻轻带。这是 DreamerV2 引入的机制（那版用一个插值系数 α=0.8），
V3 把它升级成两个独立系数。free bits 则给两份 KL 各设 1 nat 的地板：低于地板就
clip 掉，不再产生梯度，模型没必要把 KL 从 0.8 压到 0.3，那点力气不如花在重建和
奖励上；同时后验永远保有"离先验至少值 1 nat 信息"的豁免权，堵死死法二。

记后验 $q$、先验 $p$、停梯度 $\mathrm{sg}$：

$$
\mathcal{L}_{\mathrm{dyn}} = \max\!\big(1,\ \mathrm{KL}[\,\mathrm{sg}(q)\ \|\ p\,]\big),
\qquad
\mathcal{L}_{\mathrm{rep}} = \max\!\big(1,\ \mathrm{KL}[\,q\ \|\ \mathrm{sg}(p)\,]\big)
$$

$$
\mathcal{L}_{\mathrm{KL}} = 0.5\,\mathcal{L}_{\mathrm{dyn}} + 0.1\,\mathcal{L}_{\mathrm{rep}}
$$

`networks.py::RSSM.kl_loss(post, prior, free, dyn_scale, rep_scale)`：
`sg = lambda x: {k: v.detach() ...}` 是停梯度；`dyn_loss` 与 `rep_loss` 各自
`torch.clip(..., min=free)` 后按 `dyn_scale * dyn_loss + rep_scale * rep_loss` 相加。
三个数来自 `configs.yaml` 的 `kl_free: 1.0`、`dyn_scale: 0.5`、`rep_scale: 0.1`，
由 `models.py::WorldModel._train` 传入。TensorBoard 侧的探针也是现成的：`kl`（未
clip 的原始值）、`dyn_loss`、`rep_loss`、`prior_ent`、`post_ent` 全在日志里。

绑定 Step 5 的两条 run。E4a `--kl_free 0.0`：拆掉地板，预期 `kl` 曲线
被压到明显低于 1 的位置、`post_ent` 跟着掉，开环视频（Step 4）想象段变糊或漂移
加快，观察信息被挤出状态的模样。E4b `--dyn_scale 0.1 --rep_scale 0.5`：力度对调，
表征反过来迁就先验，预期重建损失上升、评估回报爬得更慢。walker_walk 任务温和，
两条 run 的症状可能都偏轻，指标级的变化（`kl`、`post_ent`、`image_loss`）比最终
分数更早也更可靠地暴露病灶，这正是你要练的读图本事。

### 5.6 return normalization：分位距归一，防两头翻车

actor 的损失里有一项固定系数 3e-4 的熵正则，管着"别过早收敛、留点
探索"。这个系数要跨任务通吃，前提是它对面的那项，优势（λ-return 减去 critic
基线），量级稳定。可优势的量级完全由任务定：回报动辄上千的任务里，3e-4 的熵是
尘埃，策略会光速收敛到第一个凑合的解；回报只有零点几的任务里，熵项反客为主，策略
永远在掷骰子。老办法是除以回报的标准差，但它在另一头翻车：奖励稀疏的任务前期回报
几乎全零，标准差趋近于零，一除等于把噪声放大成信号。DreamerV3 的方案是"**大的
压下来，小的不放大**"：用回报的 5% 到 95% 分位距当除数，且除数封底为 1。

每轮更新，取这批想象轨迹的 λ-return，算 5% 和 95% 两个分位数，用衰减
系数 0.01 的指数滑动平均（EMA）平滑到跨批次稳定；优势除以两者之差，但差值先过一道
`max(1, S)`，回报散布本来就小于 1 时，不除，保持原样。分位距而不用极差
（max 减 min），是为了不让单条离群轨迹绑架整个缩放；EMA 是为了防缩放系数本身抖动
（它一抖，等效学习率跟着抖）。

记这批 λ-return 的分位数 $\mathrm{Per}(R, q)$，EMA 后的分位距为 $S$：

$$
S \leftarrow (1-\alpha)\,S + \alpha\,\big[\mathrm{Per}(R,95) - \mathrm{Per}(R,5)\big],
\qquad \alpha = 0.01
$$

$$
\mathrm{adv} = \frac{(R - \mathrm{offset}) - (v - \mathrm{offset})}{\max(1, S)}
$$

其中 offset 取 5% 分位的 EMA。分子里 offset 对目标和基线同加同减，本会抵消，
代码保留它是为了让归一后的量落在可读区间，日志好看懂。

`models.py::RewardEMA`：`self.range = torch.tensor([0.05, 0.95])`、
`alpha=1e-2`、`scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)`。调用处在
`ImagBehavior._compute_actor_loss`：`if self._config.reward_EMA:` 分支里算
`normed_target`、`normed_base` 再相减；else 分支就是裸的 `adv = target - base`，
配置键 `reward_EMA: True` 一关，整套机制原地退役，天然的消融开关。两个分位数的
EMA 会记进日志：`EMA_005` 和 `EMA_095`。

绑定 Step 5 的 E5：`--reward_EMA False`。walker_walk 的回报量级是几百，
优势不归一后熵项等效缩水成原来的几百分之一，预期 `actor_entropy` 曲线比基线掉得
更快更深，策略过早定型，评估曲线更抖、平台更低或更晚到。同样要诚实：单任务上症状
可能只是"更抖"而不是"崩盘"，这个零件真正的用武之地是它让**同一个** 3e-4 在
回报量级差一万倍的任务之间都成立，你在单任务里看到的只是它的侧影。

## 6. 源码导读

仓库已归档只读，代码不会再漂移，精读正当时。根目录一共就七个 Python 文件加一份
配置，按这个顺序读，每个文件带着问题进去：

| 文件 | 管什么 | 带着什么问题读 |
|---|---|---|
| `configs.yaml` | 全部配置 | defaults 段和 dmc_vision、crafter 段逐键对比：哪些是学习超参（跨任务不变），哪些是预算超参（随任务组变）？5.1 到 5.6 的每个键都找到 |
| `dreamer.py` | 在线循环主干 | `Dreamer.__call__` 里训练是怎么被"顺路"触发的？`batch_steps / train_ratio` 的节奏账和 5.2 对得上吗？两段式 argparse 怎么把 `--configs dmc_vision debug` 堆叠成最终配置？|
| `models.py` | WorldModel 与 ImagBehavior | `WorldModel._train` 的 `model_loss` 由哪几项加成？`ImagBehavior._imagine` 里哪一行 detach 挡住了策略梯度回流世界模型？`RewardEMA` 的 `max(1, S)` 在哪 |
| `networks.py` | RSSM 与各种头 | `RSSM.kl_loss` 的两次 `torch.clip(min=free)` 与 5.5 的公式逐行对照；`MLP.dist` 一共支持哪几种分布？（数一数，E3 消融的合法选项就在里面） |
| `tools.py` | 工具箱 | `symlog`/`symexp` 两行函数；`DiscDist.log_prob` 的 below/above 两项怎么拼出 twohot；`lambda_return` 的递归和第 04 课的蒙特卡洛回报差在哪 |
| `envs/` | 环境包装 | dmc 包装层做了什么预处理（图像尺寸、动作重复）？`--task dmc_walker_walk` 的字符串在哪被 `split('_', 1)` 拆开分发 |
| `exploration.py` | 探索策略插件 | defaults 里 `expl_behavior: 'greedy'` 意味着本课根本没用它，什么时候才需要？|
| `parallel.py` | 多环境并行 | dmc_vision 配置的 `envs: 4` 靠它撑起来；只需知道它存在 |

读码时留个心眼：`RSSM.kl_loss` 里有条注释说原版实现用的是 `maximum` 而不是 `clip`
（越界时梯度处理有细微差别），归档仓库的好处是这类注释成了化石证据，你能看到
复现者在哪些地方拿不准、和官方版比对过什么。JAX 官方仓库 danijar/dreamerv3 的
`dreamerv3/configs.yaml` 是同一套思想的另一份实现，两边配置键名不完全相同（比如
官方的 `run.train_ratio` 与本仓库的 `train_ratio` 定义和默认值都不同），做完本课
实验后翻一翻官方版的 Tips 一节，能校准哪些细节是算法本体、哪些是实现自由度。

## 7. 实验

主线是一条 1M 步基线加六条 3e5 步消融。基线先挂上，挂机时间读第 6 节；消融等基线
过了 3e5 步再开也不迟，它们要拿基线在 3e5 处的截面当对照。

### Step 1: 克隆与环境

```bash
git clone https://github.com/NM512/dreamerv3-torch.git
```

```bash
pip install -r requirements.txt
```

先建一个 python 3.11 的虚拟环境再装（README 声明依赖在 3.11 下测试）。仓库已归档，
克隆下来的就是最终版，不存在"过两天上游又改了"的问题，这门课选它当教学锚定版，
图的就是这份稳定。装完在仓库目录建个本地分支，后面 E2、E3 要动两处代码，改动全部
留在分支里，证据好收。

### Step 2: 冒烟测试（Mac/CPU 也能跑这步）

```bash
python3 dreamer.py --configs dmc_vision debug --task dmc_walker_walk --logdir ./logdir/smoke
```

`--configs` 接受多个名字按序堆叠（`dreamer.py` 把它们拼成 `['defaults', 'dmc_vision',
'debug']` 逐层覆盖），`debug` 段把 batch 缩小、prefill 砍到 1，几分钟内就该看到
日志目录出现、终端开始打印训练信息。这一步只验证"管线通"，debug 配置学不出东西。
预期问题一次排掉：渲染报错见第 10 节第一行。

### Step 3: 挂基线（重点，README 同款命令）

```bash
python3 dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/walker_base
```

这条命令与仓库 README 的示例逐字一致：dmc_vision 配置，1M 环境步（`steps: 1e6`），
每 1e4 步做一次 10 局评估（`eval_every: 1e4`、`eval_episode_num: 10`），checkpoint
落在 `logdir/walker_base/latest.pt`。另开一个终端起看板：

```bash
tensorboard --logdir ./logdir
```

预期：前 2500 步是随机策略灌 prefill，之后训练接管；评估回报在几万步内离开随机
水位，之后持续爬升。验收线到第 9 节对，现在只确认两件事：曲线在动、`kl` 稳定在
1 附近或以上（低于 1 说明 free bits 在托底，正常；钉死在 1.0 一动不动也正常，
那是 clip 生效的形状，不是 bug）。

### Step 4: 眼见为实：看模型的梦

dmc_vision 配置里 `video_pred_log: True`，TensorBoard 的 IMAGES/视频栏里会出现
开环预测拼图：6 个样本，前 5 帧给真实观察，之后全靠模型想象往下播；上中下三条带
分别是真值、模型预测、两者的误差图。训练早期，想象段几帧后就糊成一团；训练后期，
walker 的姿态能在想象里连贯地走出十几帧。这是 actor-critic 每天生活的世界，
它的训练数据全部长这样。对照第 04 课：当年你要写胶水代码才能把 MDN-RNN 的梦
可视化，这里是系统自带的体检项目。给报告存两张截图：早期一张、后期一张。

### Step 5: 稳定性零件消融（六条 run，全部 3e5 步）

每条 run 改且只改一件事，其余配置与基线逐字相同（含默认 `seed: 0`）；对照组统一用
基线 run 在 3e5 步处的截面。`kl_free`、`dyn_scale` 这些顶层配置键都会被 `dreamer.py`
自动注册成同名命令行旗标，直接命令行覆盖；嵌套在字典里的键（E3 的 `dist`）命令行
不可靠，直接改 `configs.yaml` 并把 `git diff` 存进证据目录。

E1 想象的汇率（绑定 5.1/5.2）：

```bash
python3 dreamer.py --configs dmc_vision --task dmc_walker_walk --steps 3e5 --train_ratio 128 --logdir ./logdir/abl_train_ratio_128
```

预期：同环境步数下曲线落后基线，墙钟时间明显更短。把两条 run 的"步数-墙钟"都记
下来，算出你机器上的实际汇率。

E2 拆 symlog（绑定 5.3）：先在分支上改 `tools.py`，`DiscDist.__init__` 的
默认参数原样是 `transfwd=symlog, transbwd=symexp`，改成恒等：

```python
transfwd=lambda x: x,
transbwd=lambda x: x,
```

然后：

```bash
python3 dreamer.py --configs dmc_vision --task dmc_walker_walk --steps 3e5 --logdir ./logdir/abl_no_symlog
```

跑完立刻还原 `tools.py`（这是要开分支的原因）。预期：`value` 的统计量顶死在 20
附近、`target` 在其上方，评估回报显著低于基线，5.3 算过，walker 的真实回报量级
约 333，恒等变换下的桶只够到 20。

E3 拆 twohot（绑定 5.4）：改 `configs.yaml` defaults 段两处，`reward_head`
的 `dist: 'symlog_disc'` 与 `critic` 的 `dist: 'symlog_disc'` 都改为
`'symlog_mse'`，然后：

```bash
python3 dreamer.py --configs dmc_vision --task dmc_walker_walk --steps 3e5 --logdir ./logdir/abl_symlog_mse
```

跑完还原。预期：可能与基线接近（5.4 讲过为什么），差距小也是结论，写进报告时
注明"稠密平滑奖励下 twohot 的保险没被触发"。

E4a 拆 free bits（绑定 5.5）：

```bash
python3 dreamer.py --configs dmc_vision --task dmc_walker_walk --steps 3e5 --kl_free 0.0 --logdir ./logdir/abl_kl_free0
```

预期：`kl` 被压到明显低于 1，`post_ent` 下滑，开环视频想象段质量下降。

E4b 对调 KL balancing（绑定 5.5）：

```bash
python3 dreamer.py --configs dmc_vision --task dmc_walker_walk --steps 3e5 --dyn_scale 0.1 --rep_scale 0.5 --logdir ./logdir/abl_kl_swap
```

预期：`image_loss` 偏高（表征迁就先验），评估曲线爬得更慢。

E5 拆 return normalization（绑定 5.6）：

```bash
python3 dreamer.py --configs dmc_vision --task dmc_walker_walk --steps 3e5 --reward_EMA False --logdir ./logdir/abl_no_return_norm
```

布尔键在命令行上写 `True`/`False` 字符串（按默认值类型解析）；不放心就改
`configs.yaml`。预期：`actor_entropy` 比基线掉得更快更深，评估曲线更抖或平台更低。

### Step 6: 汇总消融小报告

一张表收口，每行一条 run：

```text
run 名 | 改了哪个键（从什么到什么） | 3e5 步评估回报（均值±方差） | 最先异常的指标 | 一句话结论
```

评估回报从 TensorBoard 的评估曲线上读 3e5 步附近的值（每次评估本身就是 10 局的
平均，`eval_episode_num: 10`）。"最先异常的指标"一栏最值钱：E2 该填 `value` 饱和、
E4a 该填 `kl` 塌陷、E5 该填 `actor_entropy` 早衰，症状先于分数出现，这是你以后
调不认识的系统时的救命本事。最后照第 01 课的规矩把 `NOTES.md` 补齐：命令、分支
diff、种子、曲线截图，一样不缺。

## 8. 配置与预算

| run | 步数 | 单卡时间（量级参考） | 产出 |
|---|---|---|---|
| smoke（debug 档） | 几百步即可停 | 分钟级，Mac/CPU 可跑 | 管线通了 |
| walker_base | 1M | 数小时到一两天（这套 PyTorch 实现偏慢，作者的继任仓库自称快约 5 倍，见第 11 节） | 复现曲线 + 消融对照组 |
| 消融 ×6（E1-E5） | 各 3e5 | 各约为基线的三分之一 | 消融表六行 |

几条实操账：

- 卡时紧张的最小套餐：E2、E4a、E5 三条预期症状最显眼的先跑，E1、E3、E4b 列为
  加餐。消融全跑约等于再跑两条基线的时间。
- 种子：全部 run 用默认 `seed: 0`，改配置不改种子，保证单一变量。单种子消融
  只能给方向不能给显著性，报告里写明白，别让表格显得比实验更硬气。
- 并发：24GB 卡的显存装得下两条 dmc_vision run 同跑，但会互抢算力，墙钟账
  自己权衡；`device: 'cuda:0'` 是配置键，多卡机器可以把消融摊开。
- 磁盘：每条 run 的 replay 片段独立落在各自 logdir 下，六条消融加基线预留 30GB。
- 断点：checkpoint 每轮存到 `logdir 下的 latest.pt`；中断后能否原地续跑取决于
  `dreamer.py` 对已有 checkpoint 的加载分支，长挂机前自己翻一眼那几行代码再决定
  敢不敢中断。

## 9. 验收

验收清单：

- [ ] **基线方向性达标**：walker_walk 的评估回报曲线与仓库 README 的 DMC Vision
      曲线图（`imgs/dmcvision.png`）同方向、同量级，DMC 任务满分 1000，
      walker_walk 属于其中较容易的任务，1M 步的评估回报应显著高于随机水位并进入
      高位平台；若不足 600，先按第 10 节排查再谈结论；
- [ ] **看到过循环的锯齿**：能在自己的 `reward_loss` 曲线上指出至少一处"策略闯进
      新地界、模型损失抬头再压回"的痕迹（5.1 的验证）；
- [ ] **开环视频两张截图**：训练早期与后期各一张，能口头指出想象段从第几帧开始
      穿帮、后期比早期晚了多少帧；
- [ ] **消融表成形**：至少三行（最小套餐 E2/E4a/E5），每行填了"最先异常的指标"，
      且每个异常都能对回第 5 节的机制解释；
- [ ] **账算得出来**：不看课文，从 `batch_size`、`batch_length`、`train_ratio`、
      `imag_horizon` 四个键现场推出"每环境步 7680 个想象转移"；
- [ ] **口试过关**：五个稳定性零件，每个一句话说清防哪种翻车（说不出的回第 5 节
      对应小节）；
- [ ] `NOTES.md` 四件套齐全：命令、分支 diff、种子、曲线，第 01 课立的规矩，
      这课六条 run 全部适用。

## 10. 排错

| 症状 | 最可能的原因 | 怎么确认 | 怎么修 |
|---|---|---|---|
| 一启动就报 `glGetError` 或 NoneType 渲染错 | 无头机器上 dm_control 没有可用渲染后端 | 报错栈里有 GL/render 字样（README 的故障排查也点了名） | 给 dm_control 配 EGL 后端（设 `MUJOCO_GL=egl` 环境变量），或用仓库自带的 `xvfb_run.sh` 包一层 |
| 依赖装不上或 import 报错 | Python 版本不是 3.11 | `python3 --version` | 重建 3.11 虚拟环境，从 Step 1 重来 |
| 开跑后第一批训练特别久像卡死 | `compile: True` 的首次图编译 | 等几分钟看是否恢复打印 | 是正常现象；确要排除可用 `--compile False` 对照一次 |
| 显存溢出 | 同卡并发了多条 run 或别的进程占卡 | `nvidia-smi` | 串行跑；别靠砍 `batch_size` 救急，batch 一变，5.2 的账和消融可比性全变 |
| 评估曲线长期趴在随机水位 | prefill 没走完，或训练没被触发 | 日志步数是否过了 2500；logdir 的数据目录里有没有持续新增的 npz | 等过 prefill；检查是否误改了 `train_ratio` 或 `batch_size` 导致节奏异常 |
| `kl` 钉在 1.0 一动不动 | free bits 的 clip 生效 | 对照 `dyn_loss`、`rep_loss` 也贴着 1.0 | 不是病，是 5.5 的机制在工作；写进笔记当读图案例 |
| `value` 顶死在 20 附近 | E2 的恒等补丁没还原就跑了别的 run | `git status` 看 `tools.py` | 还原补丁重跑；这是 Step 1 让你开分支的原因 |
| 评估回报单点跳动大 | 每次评估只有 10 局，方差本来就大 | 看相邻多个评估点的趋势 | 报告里用趋势和多点平均说话，别引用单点 |
| 消融曲线与预期方向相反 | 单种子噪声，或改动没生效 | 训练日志开头会打印完整配置，逐键核对 | 先确认配置真的变了；确认后如实记录"未复现预期方向"，单种子实验本就允许这个结局 |

## 11. 前沿与改造

本课复现的算法就是当前在线模型基 RL 的参考答案：JAX 官方仓库
danijar/dreamerv3 是 Nature 2025 论文《Mastering diverse control tasks through
world models》的配套代码，一套学习超参配上 size 家族（README 里 `--configs crafter
size50m` 这种堆叠写法）覆盖从爬虫到 Minecraft 的全部任务组。锚定仓库的作者本人
也已经把火力转向继任项目：NM512/r2dreamer（ICLR 2026，主打免解码器加表征去冗余
正则）内置了一份自称比 dreamerv3-torch 快约 5 倍的 DreamerV3 复现。本课仍锚定
归档版，理由在第 6 节说过，代码不再漂移，课文里的每个键名十年后还对得上；追
速度的时候你自然知道去哪。

规模侧：单任务对 150+ 任务、单种子对多种子、512 单元对官方的
size 家族，全是钱和卡时能解决的。机制侧：Dreamer 仍然靠重建像素训表征，第 02
课"草地纹理该不该扔"的老问题它没回答，只是用大 decoder 硬扛；第 08 课 TD-MPC2
把重建整个拿掉，两边对打的证据你自己来做。

动手改造清单。

1. 想象长度扫描：`--imag_horizon` 取 5 和 30 各跑一条 3e5（基线 15 已有）。
   预算：两条消融的卡时。预期：5 的信用分配太短、学得慢；30 把更多模型误差喂给
   critic，收益不增反可能更抖，第 03 课的误差滚雪球曲线在这里定价。失败判据：
   三条曲线无序且找不到指标层面的解释。
2. JAX 官方版对照跑：装 danijar/dreamerv3，按其 README 跑
   `--configs crafter --run.train_ratio 32`（注意两仓库的 train_ratio 定义与默认值
   不同，不能拿数字互套）。预算：单卡一天量级。预期：方向一致、墙钟明显更快。
   顺带体会同一算法两种实现的工程差距。
3. 换任务组压力测试：用锚定仓库跑 `--configs crafter`（模型自动放大到
   `dyn_deter: 4096`、`units: 1024`，策略梯度切换成 REINFORCE），与 README 的
   Crafter 曲线对照。预算：1M 步，一两天。预期：能看到"一套超参"的真实边界，
   学习超参没动，预算超参和梯度估计器换了档。
4. 顺手复现：论文的稳定性消融结论（去掉这些零件后部分任务显著变差）在本课
   缩小版设置里对应 Step 5 全套，你的 E2、E4a、E5 若出现同方向症状，就是用单
   任务预算复现了论文消融的方向；E3 打平则复现了"这类零件的价值在任务分布的
   尾部"这半句。

## 12. 论文与延伸

1. DreamerV3（Hafner et al., [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)；
   期刊版即 Nature 2025 的《Mastering diverse control tasks through world models》）
   ，带着三个问题读：正文的每个稳定性零件各自对应你消融表的哪一行、症状描述
   一致吗？附录超参表里哪些数字跨任务组固定、哪些随预算档位变？"不调参玩出
   Minecraft 钻石"为什么被选作招牌结果，它考验的是五个零件里的哪几个？
2. DreamerV2（Hafner et al., [arXiv:2010.02193](https://arxiv.org/abs/2010.02193)）
   ，KL balancing 的出生地（当年是单个插值系数 α=0.8）。带着问题读：离散
   latent（32 组 32 类）凭什么赢了高斯 latent？V3 在它基础上改了哪三件事、
   各是为了通吃哪类任务？
3. DreamerV1（Hafner et al., [arXiv:1912.01603](https://arxiv.org/abs/1912.01603)）
   ，"在想象里训 actor-critic"的原点，想象 horizon 15 从这里沿用至今。带着
   问题读：它靠什么把策略梯度顺着想象动力学直通反传？这条路在离散动作上为什么
   走不通（对照 configs 里 Crafter 的 `imag_gradient: 'reinforce'`）？
4. Crafter（Hafner, [arXiv:2109.06780](https://arxiv.org/abs/2109.06780)），
   走第 11 节改造 3 或 E3 加餐路线的必读。带着问题读：22 项成就的分数为什么用
   几何平均而不是算术平均？这个设计想逼出智能体的哪种能力谱？
5. 两份 README 当文献读：锚定仓库 README 顶部的"Outdated Implementation"
   告示，开源复现的生命周期本身就是一课；danijar/dreamerv3 README 的 Tips 一节
   ，官方作者自己踩过的坑清单，和你第 10 节的症状表对照着看。

收个尾。到这一课为止，你的系统第一次"活"了：不再有采数据、训模型、训策略的
先后之分，三件事咬合成一个不停转的循环，第 01 课诊断的分布偏移在循环里得到了
系统性的压制。但有一根大动脉我们从第 02 课起就没质疑过：这一路全靠**重建像素**
给表征供血，Dreamer 的 decoder 还在一帧帧画出草地纹理，哪怕开车根本用不上它。
第 07 课MuZero 把这根动脉直接结扎：不重建任何观察，模型只预测价值、
策略和奖励三样对决策有用的东西。凭什么这也配叫世界模型？CPU 笔记本就能跑的
实验会给你答案。
