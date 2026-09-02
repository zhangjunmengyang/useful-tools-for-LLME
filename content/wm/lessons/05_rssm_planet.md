---
id: 05_rssm_planet
title: "RSSM 的确定状态与随机状态"
summary: "纯 RNN 状态为什么记不住又赌不准？确定通道和随机通道各管什么？"
unit: engine
play_tools: []
checkpoints:
  - "RSSM 结构笔记，对照代码逐行讲清两条通道。"
  - "一个任务的想象序列可视化。"
  - "KL 平衡、free bits 各自防什么病的消融记录。"
---

# 第 05 课：RSSM 的确定状态与随机状态

> 类型：实战/体验（PlaNet 缩小档训练 + 规划器拆解）+ 精读（dreamerv3-torch 的 RSSM 模块，只读不训）<br>
> 建议周期：2-3 天<br>
> 硬件：单张 24GB 卡数小时（显存要求不高，8GB 也够）；Mac/纯 CPU 可完成全部精读与探针实验，训练慢数倍<br>
> 锚定仓库：[Kaixhin/PlaNet](https://github.com/Kaixhin/PlaNet)（跑），[NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch)（RSSM 精读）；论文 PlaNet（arXiv:1811.04551）<br>
> 产物：RSSM 结构笔记（两份实现逐行对照）、walker-walk 想象序列可视化、CEM 规划体验记录

## 1. 这一课做什么

2018 年的 World Models 有一个结构性问题：V 和 M 分开训练。V 在压缩画面时不知道后续
要预测什么，M 也无法反过来调整 V 的表示。PlaNet 在 2019 年提出的 RSSM（循环状态
空间模型）把编码器、动力学和解码器放进同一个训练目标，使表示开始为预测服务。

RSSM 还把状态拆成两条通道。一条**确定通道** $h_t$，一个 GRU
一步步滚下去，累积已经看过的历史；一条**随机通道** $z_t$，每步采样一组变量，对
"现在到底是哪种情况"保留几种可能。纯确定的通道会把多种可能的未来平均成一张鬼影
（第 03 课 MDN 处理过的那个问题），纯随机的
通道又记不牢长历史，信息每一步都要穿过采样噪声，走几十步就被冲刷干净了。两条各管
不同的问题；第 11 节会通过移除随机通道来测量差异。

控制方法也随之改变。第 04 课训练了一个固定的 controller；这里不训练策略参数，而是
每走一步都在潜空间采样 1000 条、每条 12 步的动作序列，让模型推演并打分，再保留高分
样本、收缩搜索范围，最后只执行最好序列的第一步。
这套"每步现场规划"叫 MPC，搜索引擎叫 CEM。它和第 04 课的 CMA-ES 原理相近，但用在
完全不同的地方，一个离线进化策略参数，一个在线搜索动作序列，第 5.5 节把账算清。

接下来会留下两类结果：一条开环预测图带，用来观察模型从真实帧过渡到纯预测后何时开始
漂移；一份 CEM 日志，用来检查候选分数是否上升、搜索分布是否逐轮收缩。RSSM 的先验、
后验和 KL 处理随后会直接出现在 DreamerV3 中，也是第 08、14 课的重要对照。

术语速查：

| 术语 | 一句人话 |
|---|---|
| RSSM | 循环状态空间模型：状态拆成确定、随机两条通道，压缩和预测在同一个损失里一起训 |
| 确定通道 / deter / $h_t$ | GRU 一步步滚出来的历史浓缩，负责"记牢"；PlaNet 代码里叫 belief |
| 随机通道 / stoch / $z_t$ | 每步现抽的随机变量，负责"押注"，对当下情况保留几种可能 |
| 先验（prior） | 不看当前观察、只凭历史和动作猜 $z_t$ 的分布；闭眼推演（想象、规划）全靠它 |
| 后验（posterior） | 看了当前观察再定 $z_t$ 的分布；训练和真实环境里的状态跟踪用它 |
| free nats | KL 的免罚额度：低于额度不罚，防止随机通道被 KL 压得什么信息都不装 |
| KL balancing | 把 KL 对先验、后验两侧的梯度分开缩放：先验使劲追后验，后验少迁就先验 |
| CEM（交叉熵方法） | 撒一把候选动作序列、模型里推演打分、留下尖子重估分布，循环几轮 |
| MPC（模型预测控制） | "每步现场规划一小段、只执行第一步、下一步重来"的用法，CEM 是它的搜索引擎 |
| DMC | DeepMind Control Suite，连续控制标准任务集；这里用 walker-walk 教平面双足小人走路 |
| 32×32 categorical | DreamerV3 的随机通道形态：32 个各有 32 格的骰子，取代 PlaNet 的 30 维高斯 |

## 2. 问题

1. 给状态动手术。第一幕的状态是两截拼的：VAE 的 $z$ 加 LSTM 的 $h$，中间没有
   共同的训练信号。需要搞清楚 RSSM 怎么把"压缩、记忆、预测"装进一个损失，以及
   两条通道各治什么病：确定通道治"记不住"，随机通道治"赌不准"。这不只是结构
   审美，你会在 walker-walk 上跑出证据。
2. 管好一个模型的两副面孔。训练时观察在手，$z_t$ 由后验说了算；上场推演时
   没有观察，$z_t$ 只能听先验的。两副面孔必须长得像（否则训练和使用脱节），又不能
   一模一样（否则观察白看了）。拉扯它们的是 KL 项，而 KL 项本身会生两种病，压死
   信息、两侧梯度打架，对应两味药：free nats（PlaNet/DreamerV1 就有）和
   KL balancing（DreamerV2 才引入）。年代归属需要讲对，因为它们防的病不同，混为
   一谈的人调不出正确的超参。
3. 不训练策略，直接规划。用 CEM 在潜空间里每步在线搜动作，体验世界模型的
   另一种用法：不养一只"手"，每次现场想。同时把它和第 04 课的 CMA-ES 摆在一起，
   分清"进化参数"和"搜索动作序列"这两件常被混淆的事。

界限先划清：本课对 PlaNet 是**实战/体验**档，缩小预算跑通训练循环、看重建与想象、
拆规划器，不承诺复现论文分数（课程复现清单里没有 PlaNet；第二幕的复现任务是下一课
的 DreamerV3）。dreamerv3-torch 在本课**只读不训**，它的训练同样是下一课的事。

## 3. 准备

- 手艺：第 03 课的漂移曲线画法（本课想象可视化直接复用这个思路）、第 04 课
  "模型里的分数不可全信"的警觉。概念上要能随口说出 KL 散度是什么（参见第 02 课）、
  多峰分布为什么不能用单点回归（参见第 03 课）。上一幕的模型产物这里用不上，
  任务域从 CarRacing 换到了 DMC，这是一次干净的重新开始。
- 环境：Python 3 加 PyTorch。DMC 的安装在 2026 年只剩一条命令：`pip install
  dm_control`，物理引擎 MuJoCo 会作为依赖自动装上，它 2021 年被 DeepMind 收购后
  免费开放、2022 年完全开源，当年"先去申请 license 文件"的仪式已成历史。PlaNet
  仓库的 `environment.yml` 是 2019 年的老清单，不要照抄，第 7 节给现代装法。
- 不需要 gym：PlaNet 代码里 gym 是懒加载（只在跑 Gym 任务时才 import），DMC
  任务的代码路径完全不碰它，省掉一整类版本地狱。
- 内存：经验回放池默认预留一百万帧 64×64 图像，约 12GB 内存。机器内存小于
  16GB 的，第 7 节的命令里记得把 `--experience-size` 调小。
- 无显示器的服务器：设环境变量 `MUJOCO_GL=egl`（仓库 README 的原话建议），
  渲染走 GPU 不走屏幕。
- 心理预算：PlaNet 的每一"集"= 100 次梯度更新 + 完整采集一局（每步都要跑一遍
  CEM，1000 条候选 × 12 步推演）。进度条会打印单集耗时，先跑 3 集估好总账再决定
  挂多久。

## 4. 学习目标

1. 白纸画出 RSSM 单步的数据流：$h$、$z$、动作、观察嵌入各从哪进、往哪去，先验头
   和后验头各接在哪里、各在什么时候被调用；
2. 用第 03 课的鬼影问题解释随机通道为什么存在，用"信息穿过采样噪声会被逐步冲刷"
   解释确定通道为什么存在；
3. 说出 free nats 和 KL balancing 各防什么病、各出自哪一代系统，并能在两个锚定仓库
   的代码里指出对应的那几行；
4. 讲清 PlaNet 的 30 维对角高斯和 DreamerV3 的 32×32 categorical（加 1% unimix）
   差在哪、换掉的动机是什么；
5. 默写 CEM 的采样-评分-收缩循环和它的四个超参（视野 12、迭代 10、候选 1000、
   尖子 100），并与 CMA-ES 说出至少三点本质区别；
6. 拿到任何一个训好的 PlaNet checkpoint，独立画出想象序列图像条和先验漂移曲线。

## 5. 原理

五个机制，还是那套节奏：直觉、机制、数学、代码落点、验证。

### 5.1 双通道：确定的负责记，随机的负责赌

回想第 03 课的路口：出了直道，赛道可能左拐也可能右拐。一个纯确定的 RNN
被迫报一个点，最优解是"左右拐的平均"，笔直插进草地的鬼影。MDN 的药方是在**输出头**
上装一团混合分布；RSSM 换了个位置下药：在**状态本身**里留一条随机通道，每步从分布里
抽一个 $z_t$，左拐世界抽到左拐的 $z$，右拐世界抽到右拐的 $z$，从源头上就不需要谁去
平均谁。同一个病，两种药，药效位置不同，这是理解 RSSM 的第一把钥匙。那为什么不干脆
全随机？想象一个只有随机通道的模型：每一步的状态都要从头采样，上一步的信息想传到
下一步，必须挤过采样这道噪声闸门。走一步丢一点，走五十步，开局看到的那点关键信息
（比如 walker 摔倒前的姿态趋势）早被冲没了。于是分工：**确定通道负责把历史记牢
（无损地滚），随机通道负责把不确定押准（每步现抽）**。

每一步三个动作：先用 GRU 把上一步的 $(h_{t-1}, z_{t-1}, a_{t-1})$ 滚成
新的 $h_t$（确定，无采样）；再从以 $h_t$ 为条件的分布里抽出 $z_t$（随机）；下游
所有零件，解码器、奖励头、规划器，吃的都是 $[h_t; z_t]$ 拼起来的完整状态。
PlaNet 里 $h$ 是 200 维、$z$ 是 30 维；dreamerv3-torch 的 DMC 配置里 $h$ 是 512 维、
$z$ 是 32×32 的离散变量（5.4 节细说）。

单步转移写出来是：

$$
h_t = f_{\mathrm{GRU}}\big(h_{t-1},\ [z_{t-1};\ a_{t-1}]\big), \qquad z_t \sim p(z_t \mid h_t)
$$

状态是二元组 $(h_t, z_t)$：$h_t$ 是历史的确定函数，$z_t$ 是给定历史后仍存的不确定性。
对比第一幕：World Models 的状态 $[z_t; h_t]$ 也是两截，但那两截来自**两个分开训练的
模型**；RSSM 的两条通道长在同一个模型里、同一个损失下，$h$ 滚动时吃的正是上一步抽出
的 $z$，两条通道每一步都在互相喂。

PlaNet 侧：`models.py::TransitionModel.forward`，一个 for 循环滚时间，
`beliefs[t + 1] = self.rnn(hidden, beliefs[t])` 是确定通道（`nn.GRUCell`），紧接着
`prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(...)`
是随机通道的采样。dreamerv3-torch 侧：`networks.py::RSSM.img_step`，同样的三拍，
`self._img_in_layers`（拼 $z$ 和动作）、`self._cell`（GRU 滚 `deter`）、
`self._suff_stats_layer("ims", x)` 加采样（出 `stoch`）。两份代码里状态都是一个字典/
一组张量，键名就叫 `deter` 和 `stoch`（PlaNet 叫 `belief` 和 `state`）。

双通道各自的存在价值，PlaNet 论文的消融给了方向：把模型换成纯确定
（GRU only）或纯随机（无 $h$ 的状态空间模型），多数任务分数都掉。第 11 节的
改造实验 2 是它的缩小版：两行代码把采样换成均值，看想象序列变成"平均姿态"。

### 5.2 先验与后验：闭眼用先验想，睁眼用后验校

随机通道的分布 $p(z_t \mid \cdot)$ 其实有两个版本，像天气预报的两个时刻：
昨晚的预报（只凭历史推："明天多半下雨"）和今早拉开窗帘后的判断（观察到位："确实
在下"）。前者是**先验**，不看今天的观察，只凭 $h_t$ 猜；后者是**后验**，把当前
观察也吃进去再定。用哪个取决于你有没有窗帘可拉：训练时和真实环境跟踪时，观察在手，
用后验；想象未来、规划动作时，未来的观察根本不存在，只能用先验。世界模型的"预测"
本质上就是：**用后验把状态校准到现在，再用先验一步步推向未来。**

两个分布头长在同一台 GRU 上。先验头只看 $h_t$；后验头看 $h_t$ 拼上当前
观察的编码 embed$(o_t)$。训练时三股损失同时拉：解码器从后验状态重建观察（保证 $z$
里装了真东西）、奖励头从后验状态预测奖励（规划时要用）、KL 项把先验往后验拉近
（保证闭眼猜的和睁眼看的别差太远）。KL 这一项是全套系统的枢纽：它训练先验去预测
"后验将会看到什么"，这正是"预测下一步"在潜空间里的形态。多说一句它的第二重
身份：KL 同时也在约束后验别把太多只有观察才知道的细节塞进 $z$，逼着信息往可预测的
方向组织。

每个时间步的损失是三项之和（对时间求和、对 batch 求平均）：

$$
\mathcal{L}_t = -\ln p(o_t \mid h_t, z_t) \; - \ln p(r_t \mid h_t, z_t) \; + \; \mathrm{KL}\big[\, q(z_t \mid h_t, o_t) \,\|\, p(z_t \mid h_t) \,\big]
$$

前两项用后验的 $z_t$ 算（睁眼训练），第三项就是先验和后验的拉扯。这是变分推断的
标准形态，和第 02 课 VAE 的"重建 + KL"同宗，只是先验从固定的标准正态换成了
学出来的、随历史变化的 $p(z_t \mid h_t)$。这个换法就是"世界模型"三个字的数学
落点：先验不再是死的，它就是动力学。

PlaNet 侧全在两处：`models.py::TransitionModel.forward` 里，
`fc_state_prior` 输出先验的均值方差，`fc_embed_belief_posterior` 把 `beliefs[t + 1]`
和观察嵌入拼起来、`fc_state_posterior` 输出后验的均值方差；关键开关是 forward 的
`observations` 参数，传了就双轨并行（训练模式，第 47 行 `_state = prior_states[t]
if observations is None else posterior_states[t]` 决定链条用谁），传 `None` 就只剩
先验单轨（想象模式，CEM 规划器就是这么调它的）。损失装配在 `main.py` 训练循环里，
`observation_loss`、`reward_loss`、`kl_loss` 三行一目了然。dreamerv3-torch 侧：
`networks.py::RSSM.obs_step` 是后验（先内部调一次 `img_step` 拿先验和新 `deter`，
再拼 embed 出后验），`img_step` 是先验，`observe` 把 `obs_step` 沿时间扫出
`(post, prior)` 两串，`imagine_with_action` 只扫 `img_step`，四个函数名把"睁眼/
闭眼"分得清清楚楚。

第 7 节 Step 4 的图像条就是这个机制的可视化判决：同一条轨迹，前 5 步喂
观察（后验），之后只喂动作（先验）。后验段的重建应该步步贴着真实；先验段前十几步
靠谱、越往后越漂，漂移速度就是你这个先验的成色。如果后验段就糊，问题在重建；
如果先验第一步就崩，问题在 KL（先验根本没学会追后验）。这一张图把两个分布的病
分开诊断。

### 5.3 free nats 与 KL balancing：KL 项的两种病，两个年代的两味药

KL 项是枢纽，也是全系统最容易生病的地方。病有两种，别混。**病一：压死
信息。** KL 罚的是后验偏离先验的程度，优化器发现了一条邪路，让后验干脆完全等于
先验，KL 直接归零。代价是 $z$ 里不再装任何来自观察的信息，随机通道形同虚设，重建
全靠 $h$ 硬扛。这病在 VAE 文献里叫 posterior collapse，第 02 课的 β 旋钮你已经见过
它的亲戚。**病二：两侧梯度打架。** KL 的梯度同时推两边：把先验往后验拉（好事，
这是先验学预测的唯一途径），也把后验往先验拽（早期先验很烂，等于让好学生迁就差
学生）。火力均分时，训练前期后验会被烂先验拖着走，表征学得又慢又糊。

两味药，出生年代不同，治的病不同，归属讲错了就会在错误的位置调参。**free nats
治病一**，PlaNet（2019）就有，DreamerV1 沿用：给 KL 设一个免罚额度（PlaNet 是
3 nats），低于额度的部分不计损失。后验因此可以"免费"从观察里带走 3 nats 的信息量，
邪路（KL 归零）无利可图，因为归零和 3 都不罚。**KL balancing 治病二**，DreamerV2
（2021，arXiv:2010.02193）引入、V3 沿用：把 KL 拆成两份拷贝，一份冻结后验只训先验，
一份冻结先验只训后验，两份权重不同，V2 用 0.8 比 0.2，先验追后验的火力是后验迁就
先验的四倍；V3 改成 0.5 比 0.1，还把名字换成了 free bits、额度降到 1 nat（名字变了，
和 free nats 是同一味药）。

free nats 是一个 `max` 操作：KL 低于额度时损失恒等于额度，梯度为零，
高于额度才恢复正常罚。KL balancing 是两次带停止梯度的 KL 计算：`KL(sg(后验) ||
先验)` 这份的梯度只会流进先验（dreamerv3-torch 里叫 `dyn_loss`，动力学损失，先验
就是动力学），`KL(后验 || sg(先验))` 这份只流进后验（叫 `rep_loss`，表征损失），
然后各乘各的系数相加。

两味药合在一起，DreamerV3 的 KL 项完整形态是：

$$
\mathcal{L}_{\mathrm{KL}} = \beta_{\mathrm{dyn}} \max\big(1, \mathrm{KL}[\mathrm{sg}(q) \,\|\, p]\big) + \beta_{\mathrm{rep}} \max\big(1, \mathrm{KL}[q \,\|\, \mathrm{sg}(p)]\big)
$$

其中 $\mathrm{sg}$ 是停止梯度，$\beta_{\mathrm{dyn}} = 0.5$、$\beta_{\mathrm{rep}} = 0.1$、
额度 1 nat（三个数都在 configs.yaml 里：`dyn_scale`、`rep_scale`、`kl_free`）。PlaNet
的版本则只有额度没有平衡：$\max(3, \mathrm{KL}[q \,\|\, p])$，两侧梯度同流。

PlaNet 侧一行：`main.py` 里 `kl_loss = torch.max(kl_divergence(...)
.sum(dim=2), free_nats).mean(...)`，`--free-nats` 默认 3。dreamerv3-torch 侧一个
函数：`networks.py::RSSM.kl_loss(post, prior, free, dyn_scale, rep_scale)`，
`rep_loss` 那行对 prior 做 `sg`（代码里是字典逐键 `detach`），`dyn_loss` 那行对 post
做 `sg`，然后各自 `torch.clip(min=free)` 再加权。两个仓库并排看，一目了然哪味药
是后来加的。

训练日志里盯 `kl_loss`：PlaNet 前期它会贴着 3.0 纹丝不动（额度罩着，
正常），随后浮上去。第 11 节改造实验 1 是这味药的剂量实验：`--free-nats 0` 一组、
9 一组，看额度太小压死信息、太大先验放羊。

### 5.4 从 30 维高斯到 32×32 骰子：随机通道的现代化

PlaNet 的 $z$ 是 30 维对角高斯，30 个连续旋钮，每个带均值和方差。
DreamerV2 换成了一副怪牌：32 个 categorical 变量，每个从 32 个格子里抽一格（可以
想成掷 32 颗各有 32 面的骰子），抽完拼成一个 32×32 的 one-hot 矩阵当 $z$。V3 沿用。
为什么？V2 论文自己说这是经验发现、给的解释是候选性的：离散分布天然多峰（高斯被迫
单峰对称，表达"左拐或右拐"还是费劲）；one-hot 的取值范围有界，配上 KL 时数值更
稳定，不会出现高斯方差项那类病态。你不必把解释当定论，但要记住现象：换成离散后
Atari 上分数明显涨，此后主流重建系的世界模型基本都用离散 latent（第 09 课 IRIS 的
token 是另一条更激进的离散化路线）。

先验头和后验头的输出从"均值 + 方差"换成 32×32 个 logit；采样用
straight-through 直通梯度，前向真抽样（不可导），反向把梯度当作直接流过概率
（有偏但好用的近似）。V3 还加了一个小保险：**unimix**，每个骰子的概率先和均匀分布
按 99:1 混合，保证任何格子概率不低于约 0.03%，防止先验对某格过度自信后，KL 在
它赌错时爆炸。

离散 latent 的分布是 32 个独立 categorical 的乘积；unimix 写出来是
$p' = 0.99\,p + 0.01/32$。KL 在两个 categorical 之间有闭式解，数值行为比高斯之间的
KL 平顺，这是"更稳定"的具体含义。

全在 dreamerv3-torch。`configs.yaml`：`dyn_stoch: 32`、
`dyn_discrete: 32`（32 个变量 × 每个 32 类）、`unimix_ratio: 0.01`。
`networks.py::RSSM._suff_stats_layer` 里 `if self._discrete:` 分支输出 logit 并
reshape 成 `[stoch, discrete]`；`get_dist` 用 `tools.OneHotDist(logit, unimix_ratio=...)`；
`tools.py::OneHotDist.sample` 的最后一行 `sample += probs - probs.detach()` 就是
直通梯度本尊，前向值不变，反向梯度借道 `probs`。下游取用时 `get_feat` 把 32×32
摊平成 1024 维再拼上 `deter`。对照 PlaNet：`models.py` 里 `fc_state_prior` 输出
`2 * state_size`，一半均值一半方差，`F.softplus(std) + min_std_dev` 保方差为正，
高斯时代的全部家当。

精读时数维度：DMC 配置下 RSSM 的完整状态是 `deter` 512 维加 `stoch`
32×32 = 1024 维，`get_feat` 输出 1536 维，第 6 节的对照表里你要能把每个数字的
来路说清。这里不训练它，行为层面的验证留给下一课。

### 5.5 CEM：在潜空间里每步现搜一段未来

第 04 课的用法是"养一只手"：花几天把 867 个参数进化好，上场后闭着眼
执行。MPC 反过来："每步现想"：走一步之前，在脑子里把接下来 12 步的几百种走法都
过一遍，挑最好的那种，只迈出它的第一步，落地后重新想。好处是不用训练任何策略参数、
模型一好规划立刻变好；代价是每走一步都要烧一轮搜索的算力。CEM 就是那轮搜索：撒点、
打分、把撒点的分布往高分区收缩，和 CMA-ES 一个家族的套路，但简化得多。

一轮规划四拍，重复 10 次：（一）从当前的动作序列分布（每步一个独立
高斯，初始均值 0 方差 1）抽 1000 条 12 步动作序列；（二）把当前状态复制 1000 份，
用**先验**把每条序列在潜空间里推演 12 步，注意全程没有一帧像素，只有 GRU 前向；
（三）奖励头给每条推演打分（12 步奖励求和），排序；（四）取前 100 名（尖子），用
它们的均值方差重新估计动作分布。10 轮后分布已收缩到高分区，执行第 0 步的均值。

优化目标是 $J(a_{t:t+H}) = \sum_{\tau=t}^{t+H} \hat r_\tau$，$H = 12$；
CEM 的更新就是"截断分布拟合"：

$$
\mu \leftarrow \mathrm{mean}(\text{前 } K \text{ 名}), \qquad \sigma \leftarrow \mathrm{std}(\text{前 } K \text{ 名}), \qquad K = 100
$$

和 CMA-ES 对比着记（这两个最容易混）：

| 维度 | CMA-ES（第 04 课） | CEM-MPC（本课） |
|---|---|---|
| 优化对象 | 策略参数（867 维，一次性） | 动作序列（12 步 × 动作维数，每个时间步重来） |
| 发生时机 | 训练期，离线，跑几天 | 执行期，在线，每步几百毫秒到几秒 |
| 分布形态 | 全协方差椭球，逐代自适应变形 | 各步独立的对角高斯，每步重置重搜 |
| 评分来源 | 真实环境完整一局 | 世界模型想象 12 步 |
| 产物 | 一只训好的"手" | 什么都不留，下一步从头再想 |

`planner.py::MPCPlanner`，全文 39 行，四拍逐行可辨：`forward` 里
`action_mean + action_std_dev * torch.randn(...)` 是撒点，
`self.transition_model(state, actions, belief)` 是潜空间推演（传的正是三个参数，
没有 observations，纯先验），`self.reward_model(...).sum(dim=0)` 是打分，
`returns.topk(self.top_candidates)` 加 `best_actions.mean/std` 是收缩。四个超参从
`main.py` 进：`--planning-horizon 12 --optimisation-iters 10 --candidates 1000
--top-candidates 100`，与论文设置一致。真实环境的每一步怎么用它，看
`main.py::update_belief_and_act`：先用后验把状态校到当下，再把 `belief` 和
`posterior_state` 交给 planner，5.2 节"后验校准现在、先验推演未来"的分工，在
这个函数里只有五行。

第 7 节 Step 5 的探针脚本把这个循环拆开逐轮打印：搜索分布的平均标准差
应该从 1.0 一路收缩，前 100 名的平均评分应该逐轮上涨。两条曲线一起动，才说明
"采样-评分-收缩"真的在工作；只有分数涨、宽度不缩，是收缩逻辑没生效；只缩不涨，
是奖励头在瞎打分。

## 6. 源码导读

两个仓库，先跑的后读的都在这张表里。仓库状态先交底：Kaixhin/PlaNet 自 2021 年
之后没有新提交，未归档但实质停更，好消息是它极小（六个 Python 文件、主模型不到
180 行），README 里致谢了 PlaNet 原作者 Hafner（@danijar）亲自帮忙对齐复现结果，
教学锚点的可信度反而高；NM512/dreamerv3-torch 已于 2026 年年中在 GitHub 上正式
归档，代码不再变动，README 顶部自己挂了"实现已过时"的告示并指路续作 r2dreamer，
对精读来说，归档等于教材定稿，我们只管读它的 RSSM，训练的事（和告示的影响）下一课
再谈。

| 文件与位置 | 是什么 | 带着什么问题读 |
|---|---|---|
| PlaNet `models.py::TransitionModel` | RSSM 前身（论文正是在这套结构上定名 RSSM） | forward 的 `observations is None` 开关切换了什么？`posterior_states[t]` 和 `prior_states[t]` 谁在喂下一步？|
| PlaNet `models.py` 其余三件 | 编码器、解码器、奖励头 | 四个模块共用一个优化器（`main.py` 的 `param_list`），"压缩为预测服务"落实在哪一行？|
| PlaNet `planner.py::MPCPlanner` | CEM 规划器 | 39 行里找齐四拍：撒点、推演、打分、收缩各在哪行？为什么 `transition_model` 只传三个参数？|
| PlaNet `main.py`（损失区） | 三股损失装配 | `free_nats` 出现在哪一行、以什么算子生效？KL 是对哪两个 `Normal` 算的？|
| PlaNet `main.py::update_belief_and_act` | 每步交互的骨架 | 后验和 planner 的接力棒在哪交接？`explore=True` 时加的噪声是干什么的（提示：想想第 01 课的数据覆盖） |
| PlaNet `env.py` | DMC 包装层 | `CONTROL_SUITE_ENVS` 有哪七个任务？各 domain 的推荐 action repeat 是多少？观察是怎么压到 64×64、5 bit 的？|
| dv3 `networks.py::RSSM` | 本课精读主角 | `img_step`/`obs_step`/`observe`/`imagine_with_action` 四个方法各对应 5.2 节哪个动作？`kl_loss` 里两次 `detach` 各冻结了谁？|
| dv3 `networks.py::RSSM._suff_stats_layer` | 分布头 | `self._discrete` 两个分支的输出各是什么形状？高斯分支的 `std_act` 和 PlaNet 的 softplus 是不是一回事？|
| dv3 `tools.py::OneHotDist` | 离散采样 | `sample` 最后一行的 `+ probs - probs.detach()` 数值上等于什么、梯度上等于什么？unimix 混在哪一步？|
| dv3 `models.py::WorldModel` | RSSM 的雇主 | `_train` 里 `observe` 和 `kl_loss` 怎么衔接？`video_pred` 前 5 步和之后各用了 RSSM 的哪个方法（这是官方版的 Step 4 实验）？|
| dv3 `configs.yaml` | 超参总表 | `dyn_stoch`、`dyn_discrete`、`dyn_deter`、`kl_free`、`dyn_scale`、`rep_scale` 各是多少？对着 5.3/5.4 节逐个对号 |

读 PlaNet 的顺序建议：`env.py` 到 `models.py` 到 `planner.py` 到 `main.py`，从世界
读到脑子再读到手。读 dv3 只读表里五处，别陷进 actor-critic 的部分，那是下一课的
正餐。

## 7. 实验

Step 1-2 装环境跑训练，Step 3-5 三个探针（重建、想象、规划），Step 6 精读对照，
Step 7 留证据。训练（Step 2）挂机时间正好用来做 Step 6。

### Step 1: 装环境

```bash
git clone https://github.com/Kaixhin/PlaNet.git
```

```bash
pip install torch torchvision opencv-python plotly tqdm
```

```bash
pip install dm_control
```

老规矩开独立虚拟环境。不要跑 `conda env create -f environment.yml`，那是 2019 年
的清单，还会给你装一个这里用不着的老 gym。装完自检渲染（无显示器机器把
`MUJOCO_GL=egl` 一并带上）：

```bash
MUJOCO_GL=egl python -c "from dm_control import suite; env = suite.load('walker', 'walk'); print(env.physics.render(camera_id=0).shape)"
```

预期打印 `(240, 320, 3)` 一类的三元组。报 GL/EGL 错先查第 10 节。顺带一提 README
里写的启动命令是 `python.main.py`，2019 年留到今天的笔误，实际是 `python main.py`。

### Step 2: 冒烟训练 walker-walk

```bash
MUJOCO_GL=egl python main.py --env walker-walk --id walker-smoke --episodes 200 --experience-size 200000
```

参数解释：`--env walker-walk` 必须显式给（默认值是 Pendulum-v0，会走 gym 路径直接
报错）；`--episodes 200` 是冒烟档（默认 1000 是论文档，这里不要求）；
`--experience-size 200000` 把回放池从一百万帧压到二十万帧，内存从约 12GB 降到
2.5GB，200 集（最多约 10 万步）根本填不满，无损。walker 的推荐 action repeat 是 2，
恰好是默认值，不用动；以后换 cheetah-run 玩记得 `--action-repeat 4`（`env.py` 里有
整张推荐表，repeat 不对会打印警告）。

预期：先用随机动作采 5 集种子数据，然后进入"训练 100 步、采集 1 集"的循环。
每 25 集在 10 个测试环境上评估一次并写视频，每 50 集存一个 checkpoint。所有产出都
在 `results/walker-smoke/` 下：`train_rewards.html`（plotly 曲线，浏览器打开）、
`test_rewards.html`、三条损失曲线、`test_episode_*.mp4` 和同名 png、
`models_50.pth` 等。盯两个趋势：`observation_loss` 稳定下降；`train_rewards` 前
几十集在低位徘徊（模型还没学出个世界，CEM 在噪声里搜索），之后开始爬。前 3 集
记下单集耗时，乘以 200 估算总时长，超出你的耐心就把 `--episodes` 砍半，本课所有
探针在 100 集的模型上照样能做，只是想象漂得快些。

赶时间的旁路：README 说官方 releases 页挂了预训练模型和结果，下载后用
`--models 路径 --test` 可以直接试跑。但本课建议至少自己训到 100 集，三条损失
曲线怎么动起来的，看别人的 checkpoint 看不到。

### Step 3: 验收重建：看 test_episode 视频

不用写代码，仓库送的。打开 `results/walker-smoke/` 里最新的 `test_episode_*.mp4`：
每帧左半是真实观察、右半是模型从**后验状态**解码的重建（`main.py` 测试段
`torch.cat([observation, observation_model(belief, posterior_state)...], dim=3)`，
沿宽度拼接）。验收标准继承第 02 课的手艺：50 集时右半糊但能认出 walker 的躯干和
双腿姿态，就算过；200 集时四肢相位应该基本贴住左半。整幅右半是一团均匀的糊、
认不出姿态，回第 10 节。

这里停一秒想清楚：这个视频验证的只是**后验加解码器**（睁眼重建），还没碰先验。
第一幕的教训别忘，重建好只是入场券，闭眼推演的成色要用下一步的想象条来验。

### Step 4: 多步想象可视化：本课的招牌产物

写第一段胶水 `imagine_strip.py`，放在 PlaNet 仓库根目录。思路照抄 dreamerv3-torch
的 `video_pred`（第 6 节表里那格）：前 $L$ 步喂观察走后验，把状态校准；之后只喂
动作走先验，解码成画面和真实帧并排：

```python
""" 想象条：前 L 步后验跟踪，之后纯先验想象，解码对比真实帧（放 PlaNet 仓库根目录） """
import argparse
import torch
from torchvision.utils import save_image
from env import Env
from models import bottle, Encoder, ObservationModel, TransitionModel

parser = argparse.ArgumentParser()
parser.add_argument('--models', type=str, required=True)   # 如 results/walker-smoke/models_200.pth
parser.add_argument('--env', type=str, default='walker-walk')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--context', type=int, default=5)      # 睁眼步数 L
parser.add_argument('--horizon', type=int, default=45)     # 闭眼步数
args = parser.parse_args()

BELIEF, STATE, HIDDEN, EMBED = 200, 30, 200, 1024          # main.py 的默认尺寸
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env = Env(args.env, False, args.seed, 1000, 2, 5)

transition_model = TransitionModel(BELIEF, STATE, env.action_size, HIDDEN, EMBED).to(device)
observation_model = ObservationModel(False, env.observation_size, BELIEF, STATE, EMBED).to(device)
encoder = Encoder(False, env.observation_size, EMBED).to(device)
ckpt = torch.load(args.models, map_location=device)
transition_model.load_state_dict(ckpt['transition_model'])
observation_model.load_state_dict(ckpt['observation_model'])
encoder.load_state_dict(ckpt['encoder'])
for m in (transition_model, observation_model, encoder):
    m.eval()

# 采一条真实轨迹（随机动作），录下帧和动作
T = args.context + args.horizon
obs, observations, actions = env.reset(), [], []
observations.append(obs)
for _ in range(T):
    action = env.sample_random_action()
    obs, _, done = env.step(action)
    observations.append(obs)
    actions.append(action.float())
    if done:
        break
T = len(actions)
L = min(args.context, T - 1)
observations = torch.cat(observations).unsqueeze(1).to(device)   # (T+1, 1, 3, 64, 64)
actions = torch.stack(actions).unsqueeze(1).to(device)           # (T, 1, 动作维)

with torch.no_grad():
    init_belief = torch.zeros(1, BELIEF, device=device)
    init_state = torch.zeros(1, STATE, device=device)
    # 睁眼段：时间对齐照抄 main.py，actions[:L] 配 observations[1:L+1]
    embed = bottle(encoder, (observations[1:L + 1],))
    beliefs, _, _, _, post_states, _, _ = transition_model(
        init_state, actions[:L], init_belief, embed)
    # 闭眼段：从第 L 步的后验出发，observations=None 就是 5.2 节那个开关
    imag_beliefs, imag_states, _, _ = transition_model(
        post_states[-1], actions[L:], beliefs[-1], None)
    recon = bottle(observation_model, (beliefs, post_states))
    imag = bottle(observation_model, (imag_beliefs, imag_states))

model_frames = torch.cat([recon, imag], dim=0).squeeze(1).cpu() + 0.5
truth_frames = observations[1:].squeeze(1).cpu() + 0.5
save_image(torch.cat([truth_frames, model_frames.clamp(0, 1)], dim=0),
           'imagine_strip.png', nrow=T)
drift = ((model_frames - truth_frames) ** 2).mean(dim=(1, 2, 3))
for t, d in enumerate(drift.tolist(), start=1):
    tag = '后验' if t <= L else '先验'
    print('step {:3d} [{}] 像素MSE {:.5f}'.format(t, tag, d))
```

```bash
MUJOCO_GL=egl python imagine_strip.py --models results/walker-smoke/models_200.pth
```

产出 `imagine_strip.png`：上排 50 帧真实，下排 50 帧模型，下排前 5 帧是后验重建，
之后 45 帧是纯想象。预期读法：后验段贴住上排；先验段前 10-20 步 walker 的姿态还
连贯合理（腿在按物理规律摆），之后逐渐糊化、和上排分道扬镳。终端打印的逐帧 MSE
就是第 03 课漂移曲线的 RSSM 版：后验段一条低平线，切到先验段后爬升。把这串数字
复制进任何画图工具存成 `drift_curve.png`。两点提醒：动作是随机的，上排的"真实
未来"只是众多可能之一，先验段和上排不同不等于错（多峰！），要看的是**姿态合理性**
和**漂移趋势**；其次 `--context` 改成 1 再跑一次，感受"校准不足，想象全歪"，
这是第 04 课梦境起点库那个坑的 RSSM 版。

### Step 5: CEM 收缩探针：看规划器"想清楚"的过程

第二段胶水 `cem_probe.py`，同样放仓库根目录。它把 `planner.py` 的循环复刻出来，
逐轮打印搜索分布的宽度和评分榜：

```python
""" CEM 探针：复刻 MPCPlanner 的循环，逐轮打印椭球宽度与评分（放 PlaNet 仓库根目录） """
import argparse
import torch
from env import Env
from models import Encoder, RewardModel, TransitionModel

parser = argparse.ArgumentParser()
parser.add_argument('--models', type=str, required=True)
parser.add_argument('--env', type=str, default='walker-walk')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--warmup', type=int, default=30)      # 先随机走几步，让状态有内容
args = parser.parse_args()

BELIEF, STATE, HIDDEN, EMBED = 200, 30, 200, 1024
H_PLAN, ITERS, J, K = 12, 10, 1000, 100                    # 论文与 main.py 的默认规划超参
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env = Env(args.env, False, args.seed, 1000, 2, 5)
A = env.action_size
lo, hi = env.action_range

transition_model = TransitionModel(BELIEF, STATE, A, HIDDEN, EMBED).to(device)
reward_model = RewardModel(BELIEF, STATE, HIDDEN).to(device)
encoder = Encoder(False, env.observation_size, EMBED).to(device)
ckpt = torch.load(args.models, map_location=device)
transition_model.load_state_dict(ckpt['transition_model'])
reward_model.load_state_dict(ckpt['reward_model'])
encoder.load_state_dict(ckpt['encoder'])
for m in (transition_model, reward_model, encoder):
    m.eval()

# 热身：随机走 warmup 步，沿途用后验跟踪状态（update_belief_and_act 的骨架）
obs = env.reset()
belief = torch.zeros(1, BELIEF, device=device)
post = torch.zeros(1, STATE, device=device)
act = torch.zeros(1, A, device=device)
with torch.no_grad():
    for _ in range(args.warmup):
        b, _, _, _, s, _, _ = transition_model(
            post, act.unsqueeze(0), belief, encoder(obs.to(device)).unsqueeze(0))
        belief, post = b.squeeze(0), s.squeeze(0)
        raw = env.sample_random_action()
        act = raw.float().unsqueeze(0).to(device)
        obs, _, _ = env.step(raw)

    # CEM 主循环：撒点、推演、打分、收缩
    belief_rep, state_rep = belief.expand(J, -1), post.expand(J, -1)
    mean = torch.zeros(H_PLAN, 1, A, device=device)
    std = torch.ones(H_PLAN, 1, A, device=device)
    for it in range(ITERS):
        acts = (mean + std * torch.randn(H_PLAN, J, A, device=device)).clamp_(lo, hi)
        bs, ss, _, _ = transition_model(state_rep, acts, belief_rep)
        rets = reward_model(bs.view(-1, BELIEF), ss.view(-1, STATE)).view(H_PLAN, J).sum(0)
        top, idx = rets.topk(K)
        best = acts[:, idx]
        mean = best.mean(dim=1, keepdim=True)
        std = best.std(dim=1, unbiased=False, keepdim=True)
        print('轮 {:2d}  分布宽度 {:.3f}  前{}名均分 {:.3f}  最高分 {:.3f}'.format(
            it + 1, std.mean().item(), K, top.mean().item(), rets.max().item()))
```

```bash
MUJOCO_GL=egl python cem_probe.py --models results/walker-smoke/models_200.pth
```

预期：十行输出，"分布宽度"从 1.0 附近逐轮收缩（尾轮通常掉到零点二三以下），
"前 100 名均分"逐轮上涨后趋平。评分的绝对数值不用纠结，那是模型自己的奖励头
打的、以模型步为单位的 12 步总和，训练集数不同数值差很远；要看的是**两条曲线的
形状**（5.5 节"验证"的三种病对号入座）。加餐一问：把 `ITERS` 改成 1 再跑，输出
的首步动作和 10 轮版差多远？这个差距就是"多想九轮"买到的东西。

### Step 6: 精读 dreamerv3-torch 的 RSSM，填对照表

```bash
git clone https://github.com/NM512/dreamerv3-torch.git
```

只读不跑（归档仓库照样能 clone）。读第 6 节表里列的五处，边读边填下面这张对照表
，填满它就是本课交付的"RSSM 结构笔记"的骨架，每格写"文件::类或函数::关键行"
加一句人话：

| 对照条目 | PlaNet（Kaixhin） | DreamerV3（NM512） |
|---|---|---|
| 确定通道及其尺寸 | `TransitionModel` 的 `self.rnn`（GRUCell，belief 200 维） | 待你填（提示：`_cell` 与 `dyn_deter`） |
| 随机通道形态 | 30 维对角高斯 | 待你填（提示：5.4 节） |
| 先验头 | `fc_state_prior` | 待你填 |
| 后验头 | `fc_embed_belief_posterior` 加 `fc_state_posterior` | 待你填 |
| 闭眼/睁眼的切换方式 | forward 的 `observations` 参数传不传 | 待你填（提示：两个方法名） |
| KL 免罚额度 | `torch.max(kl, free_nats)`，3 nats | 待你填（数值不同） |
| KL 两侧梯度 | 同流（无平衡） | 待你填（两次 detach 各冻谁、系数各多少） |
| 防过度自信 | 无 | 待你填（提示：unimix） |

填的时候顺手回答三道理解题，写进笔记：（一）`obs_step` 为什么要先调 `img_step`？
（答案在 5.2 节，但要用代码行号说话）；（二）`get_feat` 拼出来的向量在 DMC 配置下
是多少维、怎么算的？（三）PlaNet 的 `min_std_dev=0.1` 和 dv3 的 `kl_free=1.0`，
哪个也在暗中防"压死信息"？两者防的机制一样吗？

### Step 7: 留证据

实验目录在 `results/walker-smoke/`，照第 01 课的规矩补一个 `NOTES.md`：

```text
日期、机器、单集耗时（前 3 集实测）
命令与改动：训练命令全文、experience-size 改小的理由、两个胶水脚本的版本
训练曲线读数：observation_loss 首末值、kl_loss 贴 3.0 持续到第几集、train_rewards 首末值
想象条结论：先验段大约撑到第几步开始崩（附 imagine_strip.png 和 drift_curve.png）
CEM 探针结论：分布宽度首末值、榜单均分首末值（附十行原始输出）
对照表：Step 6 填完的 RSSM 结构笔记
```

## 8. 配置与预算

| 阶段 | 配置 | 主要吃什么 | 参考耗时 |
|---|---|---|---|
| 冒烟训练（Step 2） | walker-walk，200 集，回放池 20 万帧 | GPU（训练与推演）+ 少量 CPU（物理模拟） | 单卡数小时量级；先跑 3 集实测单集耗时再乘 |
| 快验档 | 同上砍到 100 集 | 同上 | 冒烟档一半；全部探针照做，想象漂得更快 |
| 论文档（不要求跑） | 1000 集（`--episodes` 默认值） | 同上 | 冒烟档五倍以上，只算账 |
| 想象条 + CEM 探针（Step 4/5） | 各一次前向为主 | 单卡秒级到分钟级；CPU 也就几分钟 | 忽略不计 |
| dv3 精读（Step 6） | 无训练 | 人的时间 | 半天 |

三条预算心得。第一，PlaNet 的时间大头不在梯度而在**采集**：每个环境步都要跑
10 轮 × 1000 候选的 CEM，这是"每步现想"的账单，也是下一课 Dreamer 用一个
前向的 actor 取代在线搜索的动机，先在这里亲身痛一次。第二，显存压力很小
（模型总共几百万参数、batch 50×50 的 64×64 图），瓶颈更多在内存（回放池）和
单卡吞吐，8GB 卡完全能跑。第三，Mac/纯 CPU 用户：探针实验毫无压力，训练建议
砍到 50-100 集或直接用官方 releases 的预训练 checkpoint 做 Step 3-5。

## 9. 验收

验收清单：

- [ ] RSSM 结构笔记完成：Step 6 对照表八行填满，每格落到文件与函数名，三道理解题
      有答案；
- [ ] 能不看笔记画出 RSSM 单步数据流，先验头、后验头、GRU、解码器的位置和调用时机
      全对；
- [ ] `test_episode_*.mp4` 里右半认得出 walker 姿态（后验重建过关）；
- [ ] `imagine_strip.png` 上下两排、`drift_curve.png` 一条曲线齐活，能指着图说出
      "后验段"和"先验段"的分界在哪、先验大约撑了多少步；
- [ ] CEM 探针十行输出：分布宽度单调收缩、榜单均分上行后趋平，能解释若两条曲线
      只动一条各是什么病；
- [ ] `train_rewards.html` 整体上行且明显高于前 5 集种子（随机动作）水平，只要求
      趋势，不设分数线（冒烟档的分数不构成对论文的任何判断）；
- [ ] 口头四连问过关：先验和后验各在什么时候被用？free nats 防什么病、出自哪代？
      KL balancing 防什么病、出自哪代？CEM 和 CMA-ES 至少三点不同？
- [ ] `NOTES.md` 六项齐全。

## 10. 排错

| 症状 | 最可能的原因 | 怎么确认 | 怎么修 |
|---|---|---|---|
| dm_control 报 GL/EGL/OpenGL 相关错误 | 无显示环境下渲染后端没选对 | 报错栈里有 `glfw`、`egl` 或 `OpenGL` 字样 | 命令前加 `MUJOCO_GL=egl`（README 同款建议）；还不行换 `MUJOCO_GL=osmesa`（慢但稳） |
| 一启动就报 gym 相关错误 | 没传 `--env`，吃了默认值 Pendulum-v0 走进 Gym 路径 | 看打印的 Options 里 env 是不是 Pendulum-v0 | 显式传 `--env walker-walk`；DMC 路径根本不 import gym |
| 启动时刷 `SyntaxWarning: "is not" with a literal` | `main.py` 里 2019 年的 `is not ''` 写法 | 警告而非报错，训练照常 | 无害，忽略；洁癖可改成 `!=` |
| 新版 PyTorch 上 `torch.jit` 编译报错 | 三个模型类都继承 `jit.ScriptModule`，老写法偶尔和新版本打架 | 报错栈指向 `jit.script_method` 或 ScriptModule | 把 `jit.ScriptModule` 改成 `nn.Module`、删掉 `@jit.script_method` 装饰器（`models.py`/`planner.py` 通用，教学运行不靠 JIT 提速） |
| 进程吃内存十几 GB 或直接被杀 | 回放池默认预留一百万帧 | 启动即涨（numpy 预分配），与训练进度无关 | `--experience-size 200000`（Step 2 已带） |
| `kl_loss` 曲线长期钉死在 3.0 | free nats 额度罩着，KL 还没顶到额度 | 前几十集属正常；同时看 observation_loss 是否在降 | 重建在降就不用管；上百集后仍钉死且重建不降，模型没在学，查数据采集是否正常（train_rewards 有没有非零值） |
| 重建（Step 3）清楚，想象（Step 4）第一步就崩 | 先验没学会追后验 | 想象条先验段首帧即糊；对拍：`kl_loss` 是否一直巨大不降 | 多训一段；仍崩则检查是否动过 `--free-nats`（额度设太大，先验彻底放羊） |
| 想象条整条都糊，包括后验段 | 解码器/训练不足，问题在重建不在先验 | 和 test_episode 视频交叉验证 | 回 Step 2 加集数；50 集内本来就该糊 |
| `train_rewards` 一百多集还纹丝不动 | 奖励头没学好，CEM 在瞎打分 | `reward_loss` 曲线降没降；CEM 探针榜单均分是否只缩不涨 | 继续训练（walker 前期爬得慢属正常）；reward_loss 不降查数据里 reward 是否非零 |
| 换 cheetah-run 后效果奇差 | action repeat 没按 domain 推荐值 | 启动日志会打印推荐值警告 | `--action-repeat 4`（`env.py::CONTROL_SUITE_ACTION_REPEATS` 有全表） |
| 跑探针脚本报尺寸不匹配 | checkpoint 是非默认尺寸训的 | 对照训练命令里有没有改过 `--belief-size` 等 | 把脚本顶部四个常量改成当时的训练值 |

## 11. 前沿与改造

你这课拆的两样东西，RSSM 状态和 CEM 规划，2019 到 2026 年的
演化路线截然不同。RSSM 是长寿零件：DreamerV2 把随机通道换成离散骰子、给 KL 装上
balancing，DreamerV3 调整额度与配比、加上 unimix 后原样服役至今（下一课你会在同一个
dreamerv3-torch 里训练它），改动的只是随机通道的形态，双通道结构八年没动。CEM
在线搜索则被逐步替换：Dreamer 系用想象里训练的 actor 一个前向出动作（每步几毫秒
对本课的几百毫秒，这是你在 Step 2 亲身痛过的账单）；TD-MPC2（第 08 课）保留了
"每步现想"的 MPC 骨架但换了搜索引擎，MPPI 变体，用软加权取代硬排序的收缩，
且在 12 步视野之外用价值函数接住，算是 CEM 路线的现代化正统。至于状态零件的更远处：IRIS（第 09 课）
用 Transformer 取代 GRU 当记忆，JEPA 一系（第四幕）干脆不要解码器，那时你会
回头发现，本课"重建 + KL"里的重建项才是各路人马分歧的火药桶。

规模一半：deter 200 对 DreamerV3 DMC 配置的 512（Minecraft 配置
4096）、集数 200 对论文档 1000，这些钱和时间能解决。机制一半：高斯对离散、无平衡
对 balancing、在线搜索对想象训练的 actor，前两样你已在 Step 6 逐行对照过，第三样
是下一课的正题。

动手改造清单（选做）：

1. free nats 剂量实验：`--free-nats 0` 与 `--free-nats 9` 各训 100 集（预算
   各约冒烟档一半），和默认 3 的想象条并排。预期：0 那组 KL 被压得很低、想象条
   先验段崩得更早（信息被压死，先验没东西可学）；9 那组 KL 长期低于额度等于没有
   拉扯，先验同样追不上后验。失败判据：三组想象条肉眼无差，说明 100 集还没到
   KL 起作用的阶段，把结论如实写成"本预算下不敏感"。
2. 砍掉随机通道：`models.py` 里把
   `prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(...)`
   改成 `prior_states[t + 1] = prior_means[t + 1]`（posterior 同一行同理），重训
   100 集。预期：重建大体还行，想象条出现"平均姿态"的糊帧（5.1 节的鬼影回魂），
   train_rewards 低于对照。这是 PlaNet 论文"纯确定不行"消融的缩小版。失败判据：
   与对照无差，walker-walk 的随机性可能不足以暴露此病，报告里写明并留给更随机的
   任务。
3. 规划视野扫描：`--planning-horizon` 取 6/12/24 各训（或各测）50 集。预期
   拱形：6 太短视（walker 迈腿的收益要多步后才兑现），24 想象漂移吃掉打分精度。
   失败判据：单调，把它和你的漂移曲线放在一起解释为什么。
4. 规划预算减半：`--candidates 500 --top-candidates 50`，只影响采集速度与
   分数，不用重训模型（用 `--models` 载入已训 checkpoint 加 `--test` 对比测试分）。
   预期：单集耗时近半，分数小幅回落。这笔"算力换分数"的账，第 08 课评测 TD-MPC2
   时还要算一遍。

PlaNet 论文的核心论断之一"确定与随机两条通道缺一不可"，映射到
本课就是改造 2：砍掉随机通道后想象质量与分数同向下滑，你就在 walker-walk 上旁证了
它的方向；砍不出差别，你收获的是"这个结论在低随机性任务上需要多大预算才能显形"
的一手记录，两头都是赚。

## 12. 论文与延伸

1. PlaNet: Learning Latent Dynamics for Planning from Pixels（Hafner et al., 2019，
   [arXiv:1811.04551](https://arxiv.org/abs/1811.04551)），RSSM 的出生证。带着四个
   问题读：消融实验里纯确定和纯随机各败在哪类任务？free nats 在文中哪里出现、给了
   多少额度？规划超参（12/10/1000/100）和你 Step 5 探针里的四个常量对上了吗？
   latent overshooting（多步 KL 正则）是什么？Kaixhin 仓库把它的三个开关默认设为
   0，对照论文的消融结果和仓库的 issue 区，找找这个默认值的依据。
2. World Models（Ha & Schmidhuber, 2018，[arXiv:1803.10122](https://arxiv.org/abs/1803.10122)）
   ，这次只重读 M 那一章，做一次对照阅读：MDN-RNN 把随机性装在输出头（预测的
   分布），RSSM 把随机性装进状态本身（$z_t$ 参与下一步滚动）；前者的多峰押注用完
   即弃，后者的押注会被 GRU 记住并影响后续所有推演。带着一个问题：第 03 课的动作
   对换实验，在 RSSM 上要怎么改写才能做（提示：对 `img_step` 喂不同动作，比较先验
   分布的距离）？
3. Dream to Control（DreamerV1）（Hafner et al., 2019，
   [arXiv:1912.01603](https://arxiv.org/abs/1912.01603)），只当下一课的预告片翻
   一遍：同一个 RSSM，扔掉 CEM，在想象里用反传梯度训练 actor-critic。带着本课的
   体感读一个问题：CEM 每个环境步要烧 12 万次模型前向（10 轮 × 1000 候选 × 12 步），
   Dreamer 凭什么敢用一个前向替掉整场搜索、又用什么补上"每步重想"丢掉的即时纠错？
   答案下一课验。
4. Mastering Atari with Discrete World Models（DreamerV2）（Hafner et al., 2021，
   [arXiv:2010.02193](https://arxiv.org/abs/2010.02193)），KL balancing 和离散
   latent 的出生证，本课 5.3/5.4 两节的原始出处。带着两个问题读：0.8 比 0.2 的
   火力偏向谁、论文怎么论证"先验该多学一点"？对"离散为什么比高斯好"，作者给了
   哪几条候选解释、哪条你觉得最站得住？
5. 选读：**Google AI 博客 Introducing PlaNet**（PlaNet 仓库 README 里有链接），
   官方通俗版，配动图。适合读完论文后快速过一遍，检查自己能不能挑出博客里被
   简化过头的说法。

盘点一下换零件的进度：状态这一格已经从"VAE 的 32 个数加 LSTM 隐状态"换成了
RSSM 的双通道，压缩、记忆、预测焊在同一个损失里，先验负责闭眼推演，后验负责
睁眼校正，free nats 和 KL balancing 两味药各守一关。但"手"还是笨办法：CEM 每走
一步都要在潜空间里烧掉十几万次前向去现想，而且 12 步视野之外的未来它一概不管，
迈出这条腿十几步后才摔的跤，它今天看不见。第 06 课DreamerV3 一次解决
这两笔账：在想象里训练一对 actor-critic，actor 把"现想"压缩成一个前向，critic 把
12 步之外的未来折成一个数接在视野尽头，整个强化学习循环从此搬进梦里，你在第 04
课手搓的那个梦境训练，将以工业强度的形态重新登场。
