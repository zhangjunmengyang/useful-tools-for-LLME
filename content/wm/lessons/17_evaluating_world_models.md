---
id: 17_evaluating_world_models
title: "分别评测预测、生成与规划"
summary: "一个世界模型“好”，到底是三种完全不同的好。你要哪种？"
unit: craft
play_tools: []
checkpoints:
  - "统一评测报告，至少包含一个四指标互相矛盾的案例。"
  - "公开 benchmark 地图：Atari 100k、DMC、1X 挑战赛、WorldScore 各自度量什么、漏掉什么。"
---

# 第 17 课：把预测、生成与规划分开评测

> 类型：实战（评测工程：不训练新模型，给前四幕训出的三个模型建同一套体检协议）<br>
> 建议周期：2-3 天<br>
> 硬件：无需训练；三个取证脚本各需一次推理（有卡几分钟，纯 CPU 也能忍）；汇总与统计 Mac/CPU 全程可做<br>
> 锚定仓库：前四幕自己的三份资产，[ctallec/world-models](https://github.com/ctallec/world-models)、[NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch)、[eloialonso/iris](https://github.com/eloialonso/iris)；感知指标用 LPIPS 官方 pip 包（装不上退 MSE，如实标注）<br>
> 产物：统一评测协议 `wm_eval/`（三个取证脚本 + 一个报告脚本）、一张 4 指标 × 3 模型的排名表、一份含矛盾案例的统一评测报告

## 1. 这一课做什么

这一课不再引入新的模型家族，而是给 World Models、DreamerV3 和 IRIS 建立一套共同的
评测协议。首先要把“模型好”拆开，因为它至少包含三类相互独立的能力：

- 预测准：给它现在的状态和动作，它报的下一个状态离真实答案近不近；多走几步，
  误差滚雪球滚得快不快。这是第 03 课漂移曲线量的东西。
- 生成真：它播出来的未来画面，人眼看着像不像真的。弹道清晰、砖块棱角分明，
  是第 09、10 课并排对比时你肉眼打分的东西。
- 规划好：拿它当模拟器去训练或搜索策略，最后真实环境里的分数高不高。这是
  第 04 课梦境训练、第 06 课想象训练最终交卷的东西。

三者常常相关，但不保证同步变化。MuZero 已经从结构上说明，不会重建画面的隐状态仍可
支持决策；需要在已有模型上测出这种差异。统一报告会包含一步误差、多步漂移、视觉
保真和下游控制四个指标，并分别给出排名，而不是合并成一个总分。

这套协议将直接成为第 18 课缩放曲线的纵轴和第 19 课结构修改的术前、术后量表。最终的
`eval_report.md` 应让读者准确回答“这个模型在哪项指标上更好”，而不是笼统地宣布某个
模型最好。

术语速查：

| 术语 | 一句人话 |
|---|---|
| 评测协议（protocol） | 把"怎么量"写死的全部约定：数据哪来、上下文几帧、往前滚几步、跑几个窗口、怎么归一化、怎么汇总 |
| 一步预测误差 | 每步都喂真实历史（teacher forcing，参见第 03 课），只考"下一步"报得准不准 |
| 多步漂移 | 让模型吃自己的预测往下滚，误差随步数滚雪球的曲线；第 03 课的 `drift_curve.png` 就是它 |
| 偷懒基线（lazy baseline） | "预测下一步等于这一步"这个零成本策略的误差；本课所有误差都拿它当分母 |
| 有效想象长度 | 自由 rollout 的误差撑到第几步才劣化到偷懒基线的水平；模型的"梦能做几步"的一个量法 |
| rollout 视觉保真 | 模型播出的未来帧与真实未来帧的感知差异；这里用 LPIPS，装不上退 MSE |
| LPIPS | 学出来的感知距离：两张图各过一遍深度网络，比特征而非比像素；出处 arXiv:1801.03924 |
| FVD | 视频生成界的分布级指标：真视频集合与生成视频集合在 I3D 特征空间的 Frechet 距离；出处 arXiv:1812.01717 |
| 下游控制分数 | 拿模型支撑决策后，策略在真实环境里的分数；三个模型各有各的原生量法，要归一化才能同桌 |
| 人类归一化分数 | Atari 界的换算：(分数 - 随机) / (人类 - 随机)，让打砖块和开赛车勉强坐上同一张桌 |
| 取证脚本（trace） | 本课的架构件：在每个模型自己的环境和依赖里跑推理，把中间产物落成统一格式的 npz |

## 2. 问题

三个具体问题，一个比一个硬。

第一，三种"好"在文献里普遍混着说。论文标题写"更好的世界模型"，正文可能量的
是 Atari 分数（规划好），可能量的是 FVD（生成真），也可能量的是预测损失（预测准）。
三种好互相不背书：第 06 课的 Dreamer 重建糊得认不出腿，walker 照样跑到高分；一个
视频模型能生成以假乱真的街景，拿去规划可能一步就撞墙。这里把三种好拆成四个可执行
的指标，逼你以后说"好"必带定语。

第二，你的三个模型活在三个不同的世界。World Models 在 CarRacing（连续动作、
赛道随机），Dreamer 在 DMC walker（连续控制、物理仿真），IRIS 在 Atari Breakout
（离散动作、确定性极强）。潜空间也是三套：32 维连续向量、RSSM 的确定加随机双通道、
每帧一串离散 token。**没有任何一个原始数字能横着比**：0.3 的潜空间 MSE 在 CarRacing
是好是坏，和 Breakout 的 token 错误率 5% 之间没有汇率。先说清楚：这不是你的实验
设计失误，这是世界模型评测在 2026 年的真实处境，Atari 100k 上 IRIS 和 DIAMOND
打一架，DMC 上 Dreamer 和 TD-MPC2 打一架，真实机器人数据上 1X 挑战赛自成一桌，
视频世界模型另有 WorldScore 这样的榜，四桌人各说各话，因为没有统一环境。本课的
解法是社区目前最诚实的一种：环境统一不了，就统一**量法**，每个指标都除以该环境
自己的"偷懒基线"，把数字变成无量纲的比值，然后只比排名结构，不比绝对值，并把
剩下比不了的部分明说。

第三，评测代码比训练代码更容易写出谎话。上下文给几帧、rollout 滚几步、取均值
还是取中位数、IRIS 的 token 用采样还是 argmax，每个看似无害的选择都能挪动排名。
第 15 课你已经见过一次：同一个冻结 backbone，线性探针和 attentive 探针给出的
"谁的表征好"可以不一致。这里把这个教训推广成一句话，**量法即立场**，并让你
把自己协议里的每个立场都写在明面上。

一条要先划的界限：本课比较的是"同一批模型在不同指标下的排名结构"，不回答"三个
模型谁最强"，它们环境不同，后一个问题本身不成立。这句话在你的评测报告里要
原样出现。

## 3. 准备

- 第 04 课的资产：`exp_dir/`（含 `vae/best.tar`、`mdrnn/best.tar`、
  `ctrl/best.tar`）、`datasets/carracing/` 的轨迹数据、以及那份 100 局评估脚本
  `eval_controller.py`。缺哪个回第 01-04 课补哪个。
- 第 06 课的资产：`logdir/walker_base/`（含 `latest.pt` 和 `eval_eps/` 下的
  npz 轨迹）。
- 第 09 课的资产：IRIS 的训练输出目录 `outputs/日期/时刻/`（含
  `checkpoints/last.pt`）。第 09 课没训完的话，可以按其 README 从
  HuggingFace（`eloialonso/iris` 的 `pretrained_models/`）取预训练权重充当，但
  报告里必须写明这一档是"实战"权重，不是你的复现产物。
- 三个独立的虚拟环境，即第 01、06、09 课各自的环境。别试图合并：老 gym、dmc、
  atari 的依赖互相打架，这个工程现实正是本课评测架构（取证脚本分居、报告脚本统一）
  的由来。
- 汇总环境里装感知指标：`pip install lpips`（LPIPS 官方 pip 包，会自动下载
  AlexNet 权重，需要 torch）。装不上就退化用 MSE，第 7 节的报告脚本两条路都写好了，
  用哪条要在报告里标注。
- 磁盘几个 GB：取证 npz 里存了帧序列。

## 4. 学习目标

1. 说出"预测准""生成真""规划好"各自的精确定义、对应指标和典型的脱钩案例；
2. 给任何一个新世界模型写出取证适配器：只要它能编码、能单步预测、能自由 rollout、
   能解码，就能进你的协议；
3. 解释偷懒基线归一化治什么病、治不了什么病（环境可预测性的混淆因素它就治不了）；
4. 拿到任何一个 benchmark，先答两问：它量什么，它漏什么；
5. 看到四个指标排名矛盾不慌，能讲出每对矛盾背后的机制原因；
6. 给一门课程、一个团队设计"数据、指标、协议"三件套齐全的一页小基准规格。

## 5. 原理

四个机制。前两个讲清"为什么三种好会脱钩"和"不同环境怎么同桌"，后两个把四个
指标钉成精确定义、把"量法即立场"落到可操作的清单上。

### 5.1 三种"好"为什么会脱钩

三个职业对"好地图"的标准不一样。测绘员要**准**：每个路口的坐标误差
多少米。画师要**真**：印出来的地图看着像不像那座城。出租车司机要**有用**：照着它
开能不能最快到机场。一张把全城小巷画得惟妙惟肖但主干道标错单行方向的地图，画师
满分、司机骂街；一张只画主干道和红绿灯的示意图，测绘员摇头、司机爱不释手。世界
模型是"未来的地图"，同一套三岔口原样成立。

脱钩不是巧合，是三个训练目标各自的结构性偏科：

- 为"准"优化的模型会糊。未来有多种可能时（出弯是左拐还是右拐），最小化
  均方误差的最优解是所有可能的加权平均，一条笔直插进草地的鬼影路。第 03 课
  MDN 的存在理由就是它。所以纯拿一步 MSE 排名，会系统性偏爱输出"安全平均值"
  的模型，而平均值恰恰是画师眼里最假的画面。
- 为"真"优化的模型会飘。想让画面锐利，就得从分布里采样而非取均值（IRIS 的
  token 是采出来的，DIAMOND 的帧是去噪采样出来的）。采样意味着每步注入随机性：
  单看每帧都真，串起来看轨迹早就离开了真实未来，逐帧对齐的误差反而比糊模型高。
  锐利和逐点准，在多峰分布下是一对天生的冤家。
- 为"有用"优化的模型可以两头都不要。第 07 课 MuZero 的隐状态解码不出棋盘，
  第 08 课 TD-MPC2 干脆没有解码器；它们只保"价值、奖励、策略算得对"这一条轴上
  的准。价值等价的世界模型可以在所有决策无关的维度上错得离谱，而控制分数毫发无损。

把三种好写成三个不同的泛函就看清了。记世界模型为 $m$，真实动力学为
$P^\*$。预测准量的是逐点条件误差，形如
$\mathbb{E}\,[\,d(m(s_t,a_t),\ s_{t+1})\,]$；生成真量的是**分布之间**的距离，形如
$D(\,p_m(\tau),\ p^\*(\tau)\,)$（FVD 的 Frechet 距离就是一种 $D$），它不要求逐条
轨迹对上，只要求轨迹的集合像；规划好量的是复合量
$J(\pi_m)$，先用 $m$ 造出策略 $\pi_m$，再回真环境测回报，模型误差要经过"策略
优化"这个非线性放大器才作用到分数上，放大器只放大决策相关的误差。三个泛函没有
互相控制的不等式，排名当然可以各排各的。

三种好对应第 7 节四个指标的归属：预测准是指标一（一步误差）和
指标二（多步漂移），生成真是指标三（视觉保真），规划好是指标四（控制分数）。

1X 挑战赛（第 11 课的真实机器人数据）把这套三分法直接做进了赛制：
压缩赛量 teacher-forced 交叉熵（预测准），采样赛量生成的未来帧（生成真），评估赛
干脆给你 N 个策略、问世界模型能不能排出它们的真实优劣（规划好，模型当裁判）。
一个工业界比赛拆成三个赛段发三份奖金，就是"三种好互不背书"的行业级承认。

### 5.2 三个模型不同环境，怎么坐上同一张桌

温度计量出 38 度，体重秤量出 70 公斤，谁"更大"？问题不成立，直到
你把每个数除以它自己领域的参照物（38 度比正常体温高 1.4%，70 公斤比标准体重高
10%），才勉强能说后者偏离更多。跨环境比世界模型是同一个困境：CarRacing 的潜空间
L2 和 Breakout 的 token 错误率之间没有汇率，硬比就是拿温度比体重。

本课的解法分三层，一层比一层诚实：

1. 每个模型在自己的环境里、用自己的潜空间量。不做任何跨环境迁移，IRIS 没在
   CarRacing 上训过，硬喂只会量出"分布外有多惨"，那是另一个实验。
2. 每个指标都除以该环境自己的偷懒基线。偷懒基线是"预测下一步等于这一步"的
   误差：CarRacing 里是相邻两帧潜向量的平均位移（第 03 课的比例尺、那条虚线），
   Breakout 里是相邻两帧 token 的天然翻动率。除完之后所有数字无量纲：0.4 的意思
   统一成"误差是躺平不动的 40%"。
3. 只比排名结构，不比绝对值，并把剩下的混淆因素写在报告里。归一化治不了
   "环境本身可预测性不同"这个混淆：Breakout 一帧到下一帧几乎不动、偷懒基线本身
   就强得变态，除出来的比值天然难看；CarRacing 弯道说来就来、偷懒基线弱，比值
   天然好看。所以"IRIS 的一步误差比是 1.1、World Models 是 0.5"**不能**读成
   后者的模型更强，只能读成各自相对自己环境的躺平线的位置。能安全跨模型比较的，
   是"同一个模型在四个指标下的相对强弱侧写"，以及"这些侧写的排名是否一致"。

记模型一步误差为 $e_1$、偷懒一步误差为 $\ell_1$，一步误差比：

$$
R_1 = \frac{\mathbb{E}[e_1]}{\mathbb{E}[\ell_1]}
$$

$R_1 < 1$ 说明模型至少赢过躺平，$R_1 \ge 1$ 说明这个模型的一步预测还不如复读机
，你会在自己的表里见到这种行。多步漂移曲线同理归一化：自由 rollout 第 $k$ 步的
潜空间误差 $e(k)$ 除以偷懒渐近线 $\ell_\infty$（真实相邻步位移均值，误差涨到这个
量级就等于在瞎猜）。由此定义**有效想象长度**：

$$
H_{\mathrm{eff}} = \max\{\,k :\ e(k) < \ell_\infty\,\}
$$

大白话：这个模型的梦做到第几步就烂成随机噪声。它是"多步漂移"压缩成的单个
可比数字。

归一化全部集中在第 7 节 Step 7 的 `report.py`；三个取证脚本只负责
如实记录各自空间里的原始距离和偷懒距离，不做任何换算，原始数与换算逻辑分开落盘，
将来协议改了不用重跑推理。

每份取证文件的自检：$R_1$ 算出来先看量级，教师强制下模型连躺平都赢不了
的话（$R_1 \ge 1$），先怀疑取证脚本的时间对齐错了一格（这是这类代码的头号 bug，
症状表里有），排除工程错误后才允许下"模型差"的结论。

### 5.3 四个指标的精确定义

上一节定了归一化的总原则，这一节把四个指标逐个钉死。评测代码的美德和
法律条文一样：无聊、精确、没有发挥空间。每个指标写四样东西，量什么、在哪个空间
量、分母是什么、聚合方式是什么。



指标一：一步预测误差比 $R_1$（预测准，短程）。教师强制模式：每步喂真实历史，
只考下一步。三个模型各用原生量法，

- World Models：MDN 混合分布的总均值 $\hat z_{t+1} = \sum_k \pi_k \mu_k$ 与真实
  $z_{t+1}$ 的 L2 距离（另记 NLL 作参考，它是训练损失的原样）；偷懒分母
  $\lVert z_{t+1} - z_t \rVert$。
- Dreamer：RSSM 先验特征与后验特征的 L2 距离。第 05 课讲过，`observe` 沿真实序列
  走时每步同时产出先验（没看观察的预测）和后验（看了观察的修正），两者的差距
  正是"一步预测错多少"；偷懒分母是相邻两步后验特征的位移。
- IRIS：从真实帧出发单步生成下一帧的 token，与真实下一帧 token 的错配率（16 个
  token 里错几个）；偷懒分母是相邻两帧 token 的天然翻动率。

指标二：有效想象长度 $H_{\mathrm{eff}}$（预测准，长程）。自由 rollout：给
$C$ 帧真实上下文暖机，之后喂真实动作序列但吃自己的状态预测，滚 $H=20$ 步，每步记
潜空间误差，取 $W=8$ 个起点窗口平均，按 5.2 的定义读出 $H_{\mathrm{eff}}$。这是
第 03 课漂移曲线的标准化版：当年只画给 MDN-RNN 一个模型看，今天三个模型各画一条，
分母统一成各自的偷懒渐近线。1X 挑战赛把这两档叫 temporally teacher-forced 和
fully autoregressive，行业已经在用同一对概念，名字不同而已。

指标三：视觉保真比 $V$（生成真）。把指标二那批自由 rollout 解码回像素，与
真实帧逐帧比感知距离。LPIPS 的定义：两张图各过一遍预训练卷积网（默认 AlexNet），
在多层特征上做通道加权的 L2，权重是在人类"两张图哪张更像原图"的判断数据上学出来
的（arXiv:1801.03924 的贡献就是发现这样量出的距离远比像素 MSE 贴近人眼）。分母
照旧是偷懒参照：真实相邻两帧之间的 LPIPS 均值。汇总取前 10 步的均值，超过
$H_{\mathrm{eff}}$ 之后轨迹内容已经跑偏，逐帧对齐的保真没有意义，这个截断本身
就是协议的一个立场（写进报告）。两句诚实标注：LPIPS 的骨干网是在 ImageNet 自然
图像上预训练的，量 64×64 的游戏画面属于超范围使用，行业惯例如此、但要写明；分布级
的 FVD 更对"生成真"的题意，可它需要成百上千段视频喂 I3D 网络才稳定，我们每模型
只有 8 个窗口，硬算等于掷骰子，所以这里用逐帧 LPIPS 当代理并如实标注（第 11 节
有把 FVD 装上的改造实验）。

指标四：归一化控制分数 $C$（规划好）。三个模型用各自的原生评测，再各自归一：

- World Models：第 04 课的 100 局评估均分，换算
  $(\bar s - s_{\mathrm{rand}}) / (906 - s_{\mathrm{rand}})$，
  906 是论文报告的均分（第 04 课引过），
  $s_{\mathrm{rand}}$ 用你第 01 课自己量的随机基线；
- Dreamer：walker_walk 评估回报除以 1000（DMC 任务单步奖励在 0 到 1 之间、每局
  1000 步，满分 1000）；
- IRIS：Breakout 分数按 Atari 100k 文献的人类归一化换算
  $(\bar s - s_{\mathrm{rand}})/(s_{\mathrm{human}} - s_{\mathrm{rand}})$，随机分和
  人类分抄 IRIS 论文（arXiv:2209.00588，第 09 课引过）附表的数值，别凭记忆填。

指标一、二的原始量在三个取证脚本里（Step 3、4、5），指标三、四的
换算在 `report.py`（Step 7）。协议参数集中一处：$C$ 按各模型原生习惯（World
Models 30 帧、Dreamer 5 帧、IRIS 1 帧，差异的原因和后果见 5.4），$H = 20$、
$W = 8$，随机种子固定并写进 npz。

协议定稿的标准：换一个人、只看你的第 7 节文字和代码，能跑出同一张表。
达不到就是协议还有暗参数。

### 5.4 量法即立场

第 15 课的探针实验给过你一次警告：同一个冻结 backbone，线性探针读出
"表征不行"，attentive 探针读出"信息都在，就是排列得深"。两个协议都没作弊，
但结论相反，因为"用多强的读出头去挖"本身是个立场：线性探针量的是"信息是否
浅摆着"，attentive 探针量的是"信息是否存在"。本课的四个指标同样每个都藏着一摞
这样的立场，评测者的全部职业道德，就是把它们挖出来写在明面上。

过一遍本课协议里已经做出的立场选择，每个都能挪动排名：

- IRIS 的 token 用采样还是 argmax？这里用采样（IRIS 的想象本来就是采样式的，
  用它的原生姿势），代价是一步错配率里混入采样噪声，$R_1$ 变差；改成 argmax，
  $R_1$ 立刻好看，但 rollout 会变"保守"，视觉保真的风格也跟着变。一个开关，
  两个指标反向移动，第 11 节让你扳这个开关。
- 上下文给几帧？Dreamer 原生 5 帧（`video_pred` 的写法），World Models 的
  LSTM 惯例暖机 30 步，IRIS 的 `WorldModelEnv` 从单帧重置。上下文越长对模型越
  友好；三个模型拿到的"起跑线"事实上不同。统一成 1 帧对 Dreamer 和 WM 不公平，
  统一成 30 帧 IRIS 的接口做不到，本课选"各用原生姿势并记录在案"，这是立场，
  不是真理。
- 聚合用均值还是分位数？8 个窗口的漂移曲线取均值，一个災难窗口（比如恰好
  压线出弯）就能拖垮整条曲线；取中位数则把"偶发崩坏"藏起来。本课取均值并同时
  保存逐窗口原始值，让报告读者可以自己换聚合。
- 控制分数的参照系。CarRacing 除以论文分 906，walker 除以理论满分 1000，
  Breakout 除以人类分，三种参照系哲学各不同（前人最好成绩、物理上限、人类水平），
  换任何一种，三个模型的 $C$ 值都会平移。

没有新公式，只有一句元定理式的提醒：任何指标都是从"模型的全部行为"
到一个实数的投影，投影必然丢维度；两个指标排名矛盾，等价于两个投影方向不同。
矛盾不是评测失败，是信息，它告诉你被比较的对象在被丢掉的维度上有实质差异。

本课交付的评测报告必须有一节"协议立场清单"：把上面四条（以及你自己
新引入的任何选择）逐条列出，每条一句"换成另一个选择，预计哪个指标向哪边动"。
写不出预计方向，说明你还没理解自己的协议。

## 6. 源码导读

评测要挂进三个仓库的推理接口，动手前先把钩子认全。都是前面课程摸过的文件，这次
带着"取证"的问题重读：

| 仓库 | 文件 | 带着什么问题读 |
|---|---|---|
| ctallec/world-models | `models/mdrnn.py`、`models/vae.py` | `MDRNNCell` 单步前向的输入输出各是什么？第 03 课 `probe_mdrnn.py` 的加载三细节（`strip('_l0')`、动作对齐、后验均值当 $z$）还记得吗 |
| ctallec/world-models | `utils/misc.py` | `RolloutGenerator` 怎么加载三件套？取证脚本照抄它的方式 |
| NM512/dreamerv3-torch | `models.py` | `WorldModel.video_pred` 就是官方写好的开环探针：5 帧 `observe` 暖机、`imagine_with_action` 滚剩下的、`heads["decoder"]` 解码，本课 Dreamer 取证脚本就是它的改写，先把它逐行读懂 |
| NM512/dreamerv3-torch | `networks.py` | `RSSM.observe` 返回的 post 和 prior 各是什么？`get_feat` 拼的是哪两条通道？|
| NM512/dreamerv3-torch | `dreamer.py` 文件末尾 | `__main__` 段怎么把 configs.yaml 和命令行拼成 config？取证脚本要照抄这段装配 |
| NM512/dreamerv3-torch | `tools.py` | `load_episodes` 怎么读 `eval_eps/` 的 npz？episode 字典里有哪些键？|
| eloialonso/iris | `src/envs/world_model_env.py` | `WorldModelEnv.step` 一步要过几次 Transformer 前向？token 是怎么自回归采出来的？`decode_obs_tokens` 怎么变回图像？|
| eloialonso/iris | `src/play.py` | tokenizer、world model、actor-critic 三件是怎么用 Hydra 拼成 `Agent` 再 `load('checkpoints/last.pt')` 的？取证脚本照抄这个组装 |
| eloialonso/iris | `src/models/tokenizer/lpips.py` | 惊喜：IRIS 自带一份 LPIPS 实现（tokenizer 训练拿它当感知损失）。指标三用的正是同一族距离，生成路线的模型连训练目标都在向"生成真"看齐，这是理解 5.1 脱钩的一条线索 |
| eloialonso/iris | `scripts/eval.py` | 官方评估入口：默认 100 局、25 个并行环境，只加载 tokenizer 和 actor-critic。控制分数从这来 |

## 7. 实验

架构一句话：**三个取证脚本分居三个仓库三个虚拟环境，各自把原始量落成同一格式的
npz；一个报告脚本在任意环境里统一换算出四个指标。** 模型和协议解耦，将来第 19
课做完手术的模型，写一个新取证脚本就能进同一张表。

### Step 1: 建评测工作区，定死取证格式

在你的课程项目目录下建 `wm_eval/`，四个脚本和所有 npz、报告都收在这里（脚本会
被分别拷到三个仓库里跑，产物拷回来）。先把取证 npz 的格式写成规格，后面三个脚本
共同遵守：

```text
取证文件 wm_trace_<模型名>.npz 的键：
  model         字符串，模型名
  env           字符串，环境名
  space         字符串，潜空间描述（如 "vae32"、"rssm-feat"、"tokens16x512"）
  context       整数，暖机上下文步数（各模型原生值，如实记录）
  horizon       整数，自由 rollout 步数，统一 20
  seed          整数，本次取证的随机种子
  onestep_model 形状 (N,) 教师强制一步误差，各自空间的原生单位
  onestep_lazy  形状 (N,) 同一批时刻的偷懒一步误差
  drift_model   形状 (W, H) 每个窗口每步的自由 rollout 潜空间误差
  drift_scale   标量，偷懒渐近线（真实相邻步位移均值，原生单位）
  frames_real   形状 (W, H, 高, 宽, 3) uint8，真实未来帧
  frames_model  形状 (W, H, 高, 宽, 3) uint8，模型 rollout 解码帧
```

约定两条：取证脚本只记原生单位的原始量，一切归一化归 `report.py`；帧一律存
uint8、通道在最后，分辨率不强求一致（LPIPS 对输入尺寸不挑，报告里如实记录各自
分辨率）。

### Step 2: 先校准尺子：LPIPS 在你自己的数据上说人话吗

指标拿来量模型之前，先量它自己。用第 01 课的 CarRacing 真实帧做三组配对：相邻帧
（应该很近）、隔 10 帧（应该中等）、不同轨迹随机配对（应该很远）。一个合格的
感知距离必须把这三组排对顺序。在 `wm_eval/` 下建 `calibrate_lpips.py`：

```python
"""calibrate_lpips.py ， 第 17 课 Step 2：用 CarRacing 真实帧校准感知距离。
用法示例：python calibrate_lpips.py --datadir <world-models 仓库>/datasets/carracing
"""
import argparse
import glob
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', required=True, help='第 01 课的轨迹目录')
parser.add_argument('--pairs', type=int, default=64)
args = parser.parse_args()

files = sorted(glob.glob(args.datadir + '/*/*.npz'))
assert len(files) >= 2, '至少需要两条轨迹'
ep_a = np.load(files[-1])['observations']   # (T, 96, 96, 3) uint8
ep_b = np.load(files[-2])['observations']

def to_t(x):
    """uint8 HWC 帧批变成 LPIPS 要的 [-1, 1] NCHW 张量。"""
    t = torch.from_numpy(x.astype(np.float32) / 255.0)
    return t.permute(0, 3, 1, 2) * 2 - 1

try:
    import lpips
    net = lpips.LPIPS(net='alex')
    def dist(x, y):
        with torch.no_grad():
            return net(to_t(x), to_t(y)).flatten().numpy()
    name = 'LPIPS(alex)'
except ImportError:
    def dist(x, y):
        d = (x.astype(np.float32) - y.astype(np.float32)) / 255.0
        return (d ** 2).mean(axis=(1, 2, 3))
    name = 'MSE（lpips 未安装，退化档）'

rng = np.random.default_rng(0)
T = min(len(ep_a) - 11, len(ep_b) - 1)
idx = rng.integers(0, T, size=args.pairs)
groups = [
    ('相邻帧      ', dist(ep_a[idx], ep_a[idx + 1])),
    ('隔 10 帧    ', dist(ep_a[idx], ep_a[idx + 10])),
    ('跨轨迹随机  ', dist(ep_a[idx], ep_b[rng.integers(0, T, size=args.pairs)])),
]
print('指标:', name)
for label, d in groups:
    print('%s 均值 %.4f  标准差 %.4f' % (label, d.mean(), d.std()))
```

在装了 lpips 的汇总环境里跑：

```bash
python calibrate_lpips.py --datadir ../world-models/datasets/carracing
```

预期：三组均值严格递增。顺手做一个对照，把 `try` 块临时改成强制走 MSE 分支再跑
一遍，比较两个指标给"隔 10 帧"和"跨轨迹"的区分度。CarRacing 的画面大半是草地
和路面纹理，像素 MSE 常常觉得"隔 10 帧"和"跨轨迹"差不多远，而 LPIPS 拉得开
，这正是 LPIPS 论文的核心结论（深度特征比像素差更贴人眼）在你自己数据上的缩小版
复现。三组排不出顺序的话别往下走：尺子是弯的，量什么都白量。记下相邻帧的 LPIPS
均值，它就是指标三的偷懒参照的量级。

### Step 3: 取证 World Models

把下面的 `trace_worldmodels.py` 拷到 ctallec/world-models 仓库根目录，在第 01 课
的虚拟环境里跑。加载方式与第 03 课 `probe_mdrnn.py` 完全同源（照抄
`utils/misc.py::RolloutGenerator`），漂移部分就是当年那条曲线的取证版，差别只有
一个：这次把每步的解码帧也存下来，给指标三用。

```python
"""trace_worldmodels.py ， 第 17 课取证脚本 A：World Models（第 04 课资产）。
拷到 ctallec/world-models 仓库根目录，用第 01 课的虚拟环境跑。
"""
import argparse
import glob
from os.path import join, exists

import numpy as np
import torch
import torch.nn.functional as f

from models.mdrnn import MDRNNCell
from models.vae import VAE
from utils.misc import LSIZE, ASIZE, RSIZE, RED_SIZE

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', required=True)
parser.add_argument('--datadir', default='datasets/carracing')
parser.add_argument('--out', default='wm_trace_worldmodels.npz')
parser.add_argument('--context', type=int, default=30)
parser.add_argument('--horizon', type=int, default=20)
parser.add_argument('--windows', type=int, default=8)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vae_file = join(args.logdir, 'vae', 'best.tar')
rnn_file = join(args.logdir, 'mdrnn', 'best.tar')
assert exists(vae_file) and exists(rnn_file), '先完成第 02、03 课的训练'
vae = VAE(3, LSIZE).to(device)
vae.load_state_dict(torch.load(vae_file, map_location=device)['state_dict'])
vae.eval()
mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(device)
st = torch.load(rnn_file, map_location=device)['state_dict']
mdrnn.load_state_dict({k.strip('_l0'): v for k, v in st.items()})
mdrnn.eval()


def step(action, z, hidden):
    """单步前向：返回 MDN 混合分布的总均值 E[z'] 和新隐状态。"""
    mus, sigmas, logpi, r, d, next_hidden = mdrnn(action, z, hidden)
    pi = torch.exp(logpi).unsqueeze(-1)
    return torch.sum(pi * mus, dim=1), next_hidden


def warm_hidden(upto):
    hidden = [torch.zeros(1, RSIZE).to(device) for _ in range(2)]
    with torch.no_grad():
        for t in range(upto):
            _, hidden = step(acts[t:t + 1], zs[t:t + 1], hidden)
    return hidden


# 用测试段最后一条轨迹；z 取后验均值锁死随机性，动作对齐同 loaders.py
files = sorted(glob.glob(join(args.datadir, '*', '*.npz')))
data = np.load(files[-1])
T = min(len(data['observations']), 300)
assert T - args.horizon - 2 > args.context, '轨迹太短，调小 --horizon'
with torch.no_grad():
    obs = torch.as_tensor(data['observations'][:T],
                          dtype=torch.float32).permute(0, 3, 1, 2) / 255
    obs = f.interpolate(obs, size=RED_SIZE, mode='bilinear',
                        align_corners=True)
    zs, _ = vae.encoder(obs.to(device))
acts = torch.as_tensor(data['actions'][1:T], dtype=torch.float32).to(device)

# 指标一原始量：教师强制一步误差 与 偷懒一步误差
one_model, one_lazy = [], []
hidden = [torch.zeros(1, RSIZE).to(device) for _ in range(2)]
with torch.no_grad():
    for t in range(T - 2):
        zhat, hidden = step(acts[t:t + 1], zs[t:t + 1], hidden)
        if t >= args.context:
            one_model.append(torch.norm(zhat - zs[t + 1:t + 2]).item())
            one_lazy.append(torch.norm(zs[t + 1:t + 2] - zs[t:t + 1]).item())
scale = float(np.mean(one_lazy))

# 指标二、三原始量：自由 rollout 的潜空间误差 + 双方帧
t0s = np.linspace(args.context, T - args.horizon - 2,
                  args.windows).astype(int)
drift = np.zeros((args.windows, args.horizon), dtype=np.float32)
fr_real = np.zeros((args.windows, args.horizon, RED_SIZE, RED_SIZE, 3),
                   dtype=np.uint8)
fr_model = np.zeros_like(fr_real)
with torch.no_grad():
    for w, t0 in enumerate(t0s):
        hidden = warm_hidden(t0)
        z = zs[t0:t0 + 1]
        for k in range(args.horizon):
            z, hidden = step(acts[t0 + k:t0 + k + 1], z, hidden)
            drift[w, k] = torch.norm(z - zs[t0 + k + 1:t0 + k + 2]).item()
            dec = vae.decoder(z)[0].permute(1, 2, 0).cpu().numpy()
            fr_model[w, k] = np.clip(dec * 255, 0, 255).astype(np.uint8)
            real = obs[t0 + k + 1].permute(1, 2, 0).numpy()
            fr_real[w, k] = np.clip(real * 255, 0, 255).astype(np.uint8)

np.savez_compressed(
    args.out,
    model='worldmodels', env='CarRacing', space='vae32',
    context=args.context, horizon=args.horizon, seed=args.seed,
    onestep_model=np.array(one_model, dtype=np.float32),
    onestep_lazy=np.array(one_lazy, dtype=np.float32),
    drift_model=drift, drift_scale=np.float32(scale),
    frames_real=fr_real, frames_model=fr_model)
print('已写出', args.out)
print('一步误差比预览: %.3f（正式换算在 report.py）'
      % (np.mean(one_model) / np.mean(one_lazy)))
```

运行：

```bash
python trace_worldmodels.py --logdir exp_dir
```

预期：几十秒跑完（推理很轻，CPU 也行），预览的一步误差比明显小于 1（第 03 课
你验证过 teacher forcing 曲线压在偷懒基线之下，这里是同一个事实的比值形式）。
把 `wm_trace_worldmodels.npz` 拷回 `wm_eval/`。一个如实记录的局限：这份取证只用了
一条测试轨迹的 8 个窗口，样本少、方差不小，协议在 npz 里存了逐窗口原始值，
报告环节会带着离散度说话；想加厚就多循环几条轨迹，格式不用动。

### Step 4: 取证 DreamerV3

把 `trace_dreamer.py` 拷到 dreamerv3-torch 仓库根目录，在第 06 课的虚拟环境里跑。
它是 `models.py::WorldModel.video_pred` 的取证版改写：同样 5 帧 `observe` 暖机、
同样 `imagine_with_action` 开环滚动、同样 `heads["decoder"]` 解码；多做的事是把
先验与后验特征的距离逐步记下来。配置装配那一段逐行照抄 `dreamer.py` 文件末尾的
`__main__`（仓库若更新，以它为准）；checkpoint 键名前缀 `_wm.` 来自
`dreamer.py::Dreamer.__init__` 里的 `self._wm`。

```python
"""trace_dreamer.py ， 第 17 课取证脚本 B：DreamerV3（第 06 课资产）。
拷到 NM512/dreamerv3-torch 仓库根目录，用第 06 课的虚拟环境跑。
"""
import argparse
import pathlib
import sys

import numpy as np
import ruamel.yaml as yaml
import torch

import models
import tools
from dreamer import make_env

probe = argparse.ArgumentParser()
probe.add_argument('--configs', nargs='+')
probe.add_argument('--out', default='wm_trace_dreamer.npz')
probe.add_argument('--context', type=int, default=5)   # 与 video_pred 一致
probe.add_argument('--horizon', type=int, default=20)
probe.add_argument('--windows', type=int, default=8)
probe.add_argument('--seed', type=int, default=0)
pargs, remaining = probe.parse_known_args(sys.argv[1:])

# ---- 配置装配：照抄 dreamer.py 末尾 __main__ ----
configs = yaml.safe_load(
    (pathlib.Path(__file__).parent / 'configs.yaml').read_text())

def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value

defaults = {}
for name in ['defaults', *(pargs.configs or [])]:
    recursive_update(defaults, configs[name])
parser = argparse.ArgumentParser()
for key, value in sorted(defaults.items(), key=lambda x: x[0]):
    arg_type = tools.args_type(value)
    parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
config = parser.parse_args(remaining)

torch.manual_seed(pargs.seed)
logdir = pathlib.Path(config.logdir).expanduser()

# ---- 建一个环境只为拿动作/观察空间，然后组装 WorldModel 并载权重 ----
env = make_env(config, 'eval', 0)
acts = env.action_space
config.num_actions = acts.n if hasattr(acts, 'n') else acts.shape[0]
wm = models.WorldModel(env.observation_space, acts, 0, config).to(config.device)
ckpt = torch.load(logdir / 'latest.pt', map_location=config.device)
wm_state = {k[len('_wm.'):]: v
            for k, v in ckpt['agent_state_dict'].items()
            if k.startswith('_wm.')}
wm.load_state_dict(wm_state)
wm.eval()
env.close()

# ---- 读一条评估轨迹（第 06 课跑评估时落在 eval_eps/ 的 npz） ----
eps = tools.load_episodes(logdir / 'eval_eps', limit=None)
assert eps, '先跑过第 06 课的评估，eval_eps/ 里要有 npz'
ep = max(eps.values(), key=lambda e: len(e['action']))
T = len(ep['action'])
C, H, W = pargs.context, pargs.horizon, pargs.windows
assert T > C + H + 2, '轨迹太短'

data = wm.preprocess({k: v[None, :T] for k, v in ep.items()})
with torch.no_grad():
    embed = wm.encoder(data)
    post, prior = wm.dynamics.observe(
        embed, data['action'], data['is_first'])
    feat_post = wm.dynamics.get_feat(post)[0]    # (T, F)
    feat_prior = wm.dynamics.get_feat(prior)[0]

    # 指标一原始量（先验 vs 后验；跳过 t=0，先验尚无历史可依）
    one_model = (feat_prior[1:] - feat_post[1:]).norm(dim=-1).cpu().numpy()
    one_lazy = (feat_post[1:] - feat_post[:-1]).norm(dim=-1).cpu().numpy()
    scale = float(one_lazy.mean())

    # 指标二、三原始量：从 C-1 处的后验出发，开环想象 H 步
    t0s = np.linspace(C, T - H - 2, W).astype(int)
    drift = np.zeros((W, H), dtype=np.float32)
    fr_real, fr_model = None, None
    for w, t0 in enumerate(t0s):
        init = {k: v[:, t0 - 1] for k, v in post.items()}
        roll = wm.dynamics.imagine_with_action(
            data['action'][:, t0:t0 + H], init)
        feat_roll = wm.dynamics.get_feat(roll)[0]
        drift[w] = (feat_roll - feat_post[t0:t0 + H]).norm(
            dim=-1).cpu().numpy()
        dec = wm.heads['decoder'](feat_roll[None])['image'].mode()[0]
        dec = ((dec + 0.5).clamp(0, 1) * 255).byte().cpu().numpy()
        real = ((data['image'][0, t0:t0 + H] + 0.5).clamp(0, 1) * 255
                ).byte().cpu().numpy()
        if fr_real is None:
            fr_real = np.zeros((W,) + real.shape, dtype=np.uint8)
            fr_model = np.zeros_like(fr_real)
        fr_real[w], fr_model[w] = real, dec

np.savez_compressed(
    pargs.out,
    model='dreamerv3', env='dmc_walker_walk', space='rssm-feat',
    context=C, horizon=H, seed=pargs.seed,
    onestep_model=one_model[C:], onestep_lazy=one_lazy[C:],
    drift_model=drift, drift_scale=np.float32(scale),
    frames_real=fr_real, frames_model=fr_model)
print('已写出', pargs.out)
print('一步误差比预览: %.3f' % (one_model[C:].mean() / one_lazy[C:].mean()))
```

运行（configs 与 task 必须和第 06 课训练时一致，config 装配才能对上）：

```bash
python trace_dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/walker_base
```

预期：一分钟内跑完。两个位置容易硌脚，都属于"读仓库代码能自己解决"的级别：
其一，`preprocess` 期望 episode 字典里的键齐全（`image`、`action`、`is_first`
等），我们把 npz 的所有键原样传入就是为此；其二，若 `load_state_dict` 报键名
不匹配，打印 `ckpt['agent_state_dict']` 的前几个键，对照实际前缀改一行，以
`dreamer.py` 当前版本为准。解码帧的 `+ 0.5` 与 `video_pred` 里
`truth = data["image"] + 0.5` 的写法同源：预处理时像素被平移到了以 0 为中心。
跑完把 npz 拷回 `wm_eval/`。

### Step 5: 取证 IRIS

IRIS 的取证走它自己的原生通道：`src/envs/world_model_env.py::WorldModelEnv`，
第 09 课你"进到世界模型里面玩"用的就是它。组装方式照抄 `src/play.py`（Hydra
instantiate 三件套、`Agent.load('checkpoints/last.pt')`），配置直接读训练输出目录
里的 `.hydra/config.yaml` 快照，省掉 Hydra 的目录魔法。把 `trace_iris.py` 放在
IRIS 仓库根目录，**cd 进训练输出目录**（`outputs/日期/时刻/`，就是 play.sh 要求的
那个位置）再跑。

先说清这份取证的两个原生特性，报告里要写进协议立场清单：其一，`WorldModelEnv`
从单帧重置，所以 IRIS 的上下文是 1 帧（Transformer 的记忆在 rollout 过程中自己
攒）；其二，token 是**采样**出来的（它的原生姿势），错配率里含采样噪声，而且
token 空间的错配对采样式模型天生苛刻，两串不同的 token 可能解码出肉眼几乎相同
的图。指标三在感知空间量，恰好补这个盲区；两个指标对照着读，谁也别单飞。

```python
"""trace_iris.py ， 第 17 课取证脚本 C：IRIS（第 09 课资产）。
放在 eloialonso/iris 仓库根目录；cd 进训练输出目录（含 checkpoints/last.pt
与 .hydra/config.yaml）后运行。组装方式照抄 src/play.py。
"""
import argparse
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch

root = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'src' / 'agent.py').exists())
sys.path.append(str(root / 'src'))

from hydra.utils import instantiate
from omegaconf import OmegaConf

from agent import Agent
from envs.single_process_env import SingleProcessEnv
from envs.world_model_env import WorldModelEnv
from models.actor_critic import ActorCritic
from models.world_model import WorldModel

parser = argparse.ArgumentParser()
parser.add_argument('--out', default='wm_trace_iris.npz')
parser.add_argument('--steps', type=int, default=140)
parser.add_argument('--horizon', type=int, default=20)
parser.add_argument('--windows', type=int, default=8)
parser.add_argument('--onestep', type=int, default=60)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg = OmegaConf.load('.hydra/config.yaml')

env = SingleProcessEnv(partial(instantiate, config=cfg.env.test))
tokenizer = instantiate(cfg.tokenizer)
world_model = WorldModel(obs_vocab_size=tokenizer.vocab_size,
                         act_vocab_size=env.num_actions,
                         config=instantiate(cfg.world_model))
actor_critic = ActorCritic(**cfg.actor_critic,
                           act_vocab_size=env.num_actions)
agent = Agent(tokenizer, world_model, actor_critic).to(device)
agent.load(Path('checkpoints/last.pt'), device)
agent.eval()
wm_env = WorldModelEnv(agent.tokenizer, agent.world_model, device)

# ---- 用随机动作在真实环境采一条轨迹（观察 64x64x3 uint8，以 wrappers 为准） ----
rng = np.random.default_rng(args.seed)
obs = env.reset()
frames, actions = [obs[0]], []
for _ in range(args.steps):
    a = np.array([rng.integers(0, env.num_actions)])
    obs, reward, done, _ = env.step(a)
    actions.append(int(a[0]))
    frames.append(obs[0])
    if done[0]:
        break
T = len(actions)
H, W = args.horizon, args.windows
assert T > H + 4, '这局结束太早，加大 --steps 或换个种子重采'


def to_t(frame):
    return torch.from_numpy(frame).permute(2, 0, 1)[None].float().div(255).to(device)


with torch.no_grad():
    tokens = [agent.tokenizer.encode(to_t(fr),
                                     should_preprocess=True).tokens[0]
              for fr in frames]                     # 每帧 (K,) 离散 token


def act_t(a):
    return torch.LongTensor([a]).to(device)


def frame_of(o):
    x = o if isinstance(o, np.ndarray) else o.detach().cpu().numpy()
    x = x[0]
    if x.shape[0] == 3:
        x = x.transpose(1, 2, 0)
    return np.clip(x * 255, 0, 255).astype(np.uint8)


# 指标一原始量：单步生成的 token 错配率 vs 相邻帧天然翻动率
one_model, one_lazy = [], []
N1 = min(T - 1, args.onestep)
for t in range(N1):
    wm_env.reset_from_initial_observations(to_t(frames[t]))
    wm_env.step(act_t(actions[t]), should_predict_next_obs=True)
    one_model.append((wm_env.obs_tokens[0] != tokens[t + 1]).float()
                     .mean().item())
    one_lazy.append((tokens[t] != tokens[t + 1]).float().mean().item())
scale = float(np.mean(one_lazy))

# 指标二、三原始量：自由 rollout 的 token 错配 + 双方帧
t0s = np.linspace(0, T - H - 1, W).astype(int)
drift = np.zeros((W, H), dtype=np.float32)
hw = frames[0].shape[:2]
fr_real = np.zeros((W, H, hw[0], hw[1], 3), dtype=np.uint8)
fr_model = np.zeros_like(fr_real)
for w, t0 in enumerate(t0s):
    wm_env.reset_from_initial_observations(to_t(frames[t0]))
    for k in range(H):
        o, r, d, _ = wm_env.step(act_t(actions[t0 + k]),
                                 should_predict_next_obs=True)
        drift[w, k] = (wm_env.obs_tokens[0] != tokens[t0 + k + 1]).float() \
            .mean().item()
        fr_model[w, k] = frame_of(o)
        fr_real[w, k] = frames[t0 + k + 1]

np.savez_compressed(
    args.out,
    model='iris', env=str(cfg.env.train.id), space='tokens',
    context=1, horizon=H, seed=args.seed,
    onestep_model=np.array(one_model, dtype=np.float32),
    onestep_lazy=np.array(one_lazy, dtype=np.float32),
    drift_model=drift, drift_scale=np.float32(scale),
    frames_real=fr_real, frames_model=fr_model)
print('已写出', args.out)
print('一步误差比预览: %.3f' % (np.mean(one_model) / scale))
```

进入训练输出目录后运行（脚本在仓库根目录，相对路径三层）：

```bash
python ../../../trace_iris.py
```

预期：有卡几分钟（一步取证要过 60 次"1 个动作 token + 每帧全部观察 token"的
自回归前向，rollout 又是 8 × 20 步）。这里的一步误差比预览**完全可能大于 1**
，别急着骂模型，回读上面第二个原生特性，然后带着这个数字去 Step 7 看它和视觉
保真怎么互相打架。跑完把 npz 拷回 `wm_eval/`。

### Step 6: 收三个控制分数

指标四不用新代码，用各模型的原生评测，然后手工换算成归一化分：

World Models：第 04 课的 100 局评估原样再跑一遍（或直接抄当时报告里的数）：

```bash
xvfb-run -s "-screen 0 1400x900x24" python eval_controller.py --logdir exp_dir --rollouts 100
```

换算：$(\bar s - s_{\mathrm{rand}}) / (906 - s_{\mathrm{rand}})$，随机基线用你第 01
课 Step 6 的一行统计量出的那份。

DreamerV3：打开第 06 课的 TensorBoard，读 walker_base 最后一次评估的回报均值
（每次评估 10 局，第 06 课配置写死的），除以 1000。

IRIS：两条路。快路：抄第 09 课训练日志里最后的评估均分。稳路：用仓库自带的
`scripts/eval.py` 重测（默认 100 局、25 个并行环境；它开头有一行
`assert`，要求当前目录下存在 `checkpoints/last.pt`，目录约定和 play.sh 相同，
细节以脚本本身为准）。换算成人类归一化分：
$(\bar s - s_{\mathrm{rand}}) / (s_{\mathrm{human}} - s_{\mathrm{rand}})$，
Breakout 的随机分和人类分**抄 IRIS 论文
附表**（第 09 课引过的 arXiv:2209.00588），别凭记忆填数。

三个分数连同各自的局数、离散度记进笔记，下一步当命令行参数喂给报告脚本。

### Step 7: 汇总出表，抓排名矛盾

在 `wm_eval/` 下建 `report.py`。它做的事：读三份取证 npz，算 5.3 定义的四个指标，
每个指标排一次名，再两两检查排名是否一致，最后连同警告一起写成 `eval_report.md`。

```python
"""report.py ， 第 17 课汇总：三份取证 npz + 三个控制分数，产出四指标排名表。
在 wm_eval/ 下运行；lpips 装不上自动退化为 MSE 并在报告里标注。
"""
import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--traces', nargs='+', required=True)
parser.add_argument('--control', nargs='+', required=True,
                    help='模型名=归一化控制分，如 worldmodels=0.62')
parser.add_argument('--vis-steps', type=int, default=10)
parser.add_argument('--out', default='eval_report.md')
args = parser.parse_args()
control = dict(kv.split('=') for kv in args.control)

try:
    import torch
    import lpips
    _net = lpips.LPIPS(net='alex')

    def pdist(a, b):
        ta = torch.from_numpy(a.astype(np.float32) / 255
                              ).permute(0, 3, 1, 2) * 2 - 1
        tb = torch.from_numpy(b.astype(np.float32) / 255
                              ).permute(0, 3, 1, 2) * 2 - 1
        with torch.no_grad():
            return _net(ta, tb).flatten().numpy()
    VIS_NAME = 'LPIPS(alex)'
except ImportError:
    def pdist(a, b):
        d = (a.astype(np.float32) - b.astype(np.float32)) / 255
        return (d ** 2).mean(axis=(1, 2, 3))
    VIS_NAME = 'MSE(像素，lpips 未安装)'

rows, curves = [], {}
for path in args.traces:
    z = np.load(path)
    name = str(z['model'])
    r1 = float(z['onestep_model'].mean() / z['onestep_lazy'].mean())
    drift = z['drift_model'].mean(axis=0) / float(z['drift_scale'])
    below = drift < 1.0
    heff = int(np.argmax(~below)) if (~below).any() else len(drift)
    n_win, hor = z['frames_real'].shape[:2]
    k = min(args.vis_steps, hor)
    dm, dr = [], []
    for w in range(n_win):
        dm.append(pdist(z['frames_model'][w, :k], z['frames_real'][w, :k]))
        dr.append(pdist(z['frames_real'][w, :k - 1],
                        z['frames_real'][w, 1:k]))
    vis = float(np.concatenate(dm).mean() / np.concatenate(dr).mean())
    rows.append((name, str(z['env']), r1, float(heff), vis,
                 float(control[name])))
    curves[name] = drift


def ranks(vals, higher_better):
    order = np.argsort([-v if higher_better else v for v in vals])
    rk = np.empty(len(vals), dtype=int)
    rk[order] = np.arange(1, len(vals) + 1)
    return rk


names = [r[0] for r in rows]
metrics = [
    ('一步误差比 R1，低者好', [r[2] for r in rows], False),
    ('有效想象长度 Heff，高者好', [r[3] for r in rows], True),
    ('视觉保真比 V，低者好，' + VIS_NAME, [r[4] for r in rows], False),
    ('归一化控制分 C，高者好', [r[5] for r in rows], True),
]
lines = ['# 统一评测报告（第 17 课）', '',
         '环境：' + '、'.join('%s 在 %s' % (r[0], r[1]) for r in rows),
         '协议：horizon 20、窗口 8、上下文取各模型原生值（npz 内有记录）。',
         '警告：三个模型活在不同环境，绝对值不可横比，只读排名结构。', '',
         '| 指标 | ' + ' | '.join(names) + ' |',
         '|---|' + '---|' * len(names)]
rank_rows = []
for label, vals, hb in metrics:
    rk = ranks(vals, hb)
    rank_rows.append(rk)
    lines.append('| %s | ' % label + ' | '.join(
        '%.3f（第 %d 名）' % (v, r) for v, r in zip(vals, rk)) + ' |')

lines += ['', '## 排名一致性', '']
found = False
for i in range(len(metrics)):
    for j in range(i + 1, len(metrics)):
        if not np.array_equal(rank_rows[i], rank_rows[j]):
            lines.append('- 矛盾：按「%s」与按「%s」的排名不一致。'
                         % (metrics[i][0], metrics[j][0]))
            found = True
if not found:
    lines.append('- 四个指标排名完全一致（少见，按第 10 节排查后如实报告）。')

with open(args.out, 'w') as fh:
    fh.write('\n'.join(lines) + '\n')
print('\n'.join(lines))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name, d in curves.items():
        plt.plot(np.arange(1, len(d) + 1), d, label=name)
    plt.axhline(1.0, linestyle='--', linewidth=1, label='lazy asymptote')
    plt.xlabel('steps ahead')
    plt.ylabel('latent error / lazy scale')
    plt.legend()
    plt.tight_layout()
    plt.savefig('drift_normalized.png')
    print('漂移对比图已存到 drift_normalized.png')
except ImportError:
    pass
```

运行（控制分换成你 Step 6 算出的三个数）：

```bash
python report.py --traces wm_trace_worldmodels.npz wm_trace_dreamer.npz wm_trace_iris.npz --control worldmodels=0.62 dreamerv3=0.88 iris=0.31
```

预期输出一张 4 行 3 列的指标表加"排名一致性"清单，外加一张三条归一化漂移曲线
同框的图，第 03 课那条孤零零的曲线，今天有了两个邻居。

怎么读表。常见的方向（你的具体数字会不同）：Dreamer 在 Heff 和 C 上占优
（RSSM 加想象训练本来就是为"稳定滚动、支撑决策"生的）；IRIS 的 V 排名靠前
（token 解码的帧锐利，Breakout 画面又简单），但 R1 很可能垫底甚至大于 1
（采样噪声叠加偷懒基线过强，Breakout 相邻帧几乎不动）；World Models 的 R1
常常意外地体面（CarRacing 帧间变化大，偷懒基线弱），但 Heff 短，第 03 课你
看过它的误差滚雪球。只要出现一对矛盾（比如 V 第一名的模型 R1 垫底），
本课的核心论点就在你自己的数据上成立了。

### Step 8: 写统一评测报告

`eval_report.md` 是脚本吐的原料，交付物是你自己写的报告，骨架五节，一页到两页：

```text
协议摘要：数据来源、四指标定义一句话版、协议参数（horizon、窗口、上下文、种子）
指标表：report.py 的表原样贴入，每个数字带样本量
矛盾案例：挑至少一对排名矛盾，各配一段机制解释（用 5.1 的三岔口语言）
协议立场清单：5.4 的四条选择逐条列出，每条写"换成另一选择，预计哪个指标怎么动"
边界声明：本报告只比较排名结构；环境可预测性差异未被归一化消除
```

如果实测四个指标排名完全一致：先过第 10 节排查（八成是某个取证的时间对齐或归一化
写错了）；排查后还一致，就如实写"在本协议、本批模型上未观察到排名矛盾"，并分析
原因，最常见的合法原因是三个模型的质量差距太悬殊，大差距会淹没指标间的结构差异
（好比三个学生分差 40 分时，换哪种阅卷标准都是同一个排名；分差 3 分时才见立场）。
这个分析本身就是合格的交付。

## 8. 配置与预算

本课零训练，预算全在推理和你自己的脑子上：

| 环节 | 硬件 | 耗时（参考） | 备注 |
|---|---|---|---|
| Step 2 校准 | CPU/Mac | 几分钟 | 首次跑 lpips 会下载 AlexNet 权重（几十 MB） |
| Step 3 WM 取证 | CPU 即可 | 一两分钟 | 就是第 03 课探针的推理量 |
| Step 4 Dreamer 取证 | 有卡更快，CPU 可忍 | 几分钟 | `make_env` 需要 dmc 依赖齐全（第 06 课环境自带） |
| Step 5 IRIS 取证 | 建议有卡 | 卡上几分钟，CPU 半小时级 | 自回归逐 token 采样是大头 |
| Step 6 控制分数 | WM 那份吃 CPU 多核 | 100 局数小时 | 第 04 课跑过的话直接抄旧数，别重烧 |
| Step 7 汇总 | CPU/Mac | 一分钟 | LPIPS 过 8×10 帧×3 模型，很轻 |

协议参数的推荐值就是正文写死的那组：horizon 20、窗口 8、上下文各模型原生值、
视觉保真截前 10 步。想改可以，四个字的纪律：改完全改，三个模型必须吃同一组
协议参数，且报告里记录改动。

## 9. 验收

验收清单：

- [ ] `calibrate_lpips.py` 的三组距离严格递增，且能说出 LPIPS 和 MSE 在哪组上
      区分度差别最大；
- [ ] 三份取证 npz 齐全，键名与 Step 1 的格式规格逐字一致（拿 `np.load` 打印
      `files` 自查）；
- [ ] 每份取证的一步误差比预览值能解释：为什么小于 1 或为什么不小于 1；
- [ ] `eval_report.md` 的指标表 4 行 3 列填满，每个控制分带局数与离散度；
- [ ] 归一化漂移图三条曲线同框，能指着图说出每条的 $H_{\mathrm{eff}}$ 在哪里；
- [ ] 至少一对排名矛盾被抓出并配了机制解释；实测无矛盾的话，排查记录加原因分析
      顶替；
- [ ] 协议立场清单至少四条，每条有"换选择后的预计方向"；
- [ ] 能口头回答：为什么这里不把三个模型放进同一个环境重训再比？（提示：答案有
      工程半句和原理半句，重训预算和仓库支持是工程半句；就算重训了，"哪个环境"
      本身又是一个立场，是原理半句。）

眼见为实检查：从 `wm_trace_iris.npz` 里随手解一个窗口的 `frames_model` 拼图看看
，token 采样的帧应该锐利但偶尔"跳变"（砖块突然消失、球瞬移）；再看
`wm_trace_dreamer.npz` 的帧，应该连贯但糊。两种病长得完全不同，这是 5.1
的脱钩用肉眼看的样子。

## 10. 排错

| 症状 | 最可能的原因 | 怎么确认 | 怎么修 |
|---|---|---|---|
| 某模型一步误差比大于等于 1，且你不服 | 取证脚本时间对齐错一格（预测第 t+1 帧却拿第 t 帧对答案） | 手工检查一个样本：把"预测目标"换成真实 t 帧，误差比应骤降到接近 0 | 对照第 03 课"动作用 `actions[1:T]` 对齐"的写法逐行核 |
| lpips 安装失败或下载权重超时 | 网络或 torch 版本 | pip 报错信息 | 退 MSE 分支照常走完，报告里如实标注指标名 |
| Dreamer 取证报 state_dict 键不匹配 | checkpoint 前缀与 `_wm.` 假设不符（仓库版本差异） | 打印 `ckpt['agent_state_dict']` 前几个键 | 按实际前缀改一行过滤；以 `dreamer.py::Dreamer.__init__` 为准 |
| Dreamer 取证报 preprocess 缺键 | eval_eps 的 npz 键与 preprocess 期望不符 | 打印 `ep.keys()` | 对照 `models.py::WorldModel.preprocess` 的用键补齐或裁剪 |
| IRIS 取证在 instantiate 处崩 | 没在训练输出目录里跑，`.hydra/config.yaml` 或 `checkpoints/last.pt` 不在当前目录 | `ls .hydra checkpoints` | cd 进 `outputs/日期/时刻/` 再跑；用 HF 预训练权重的按 README 把权重拷成 `checkpoints/last.pt` |
| IRIS 随机动作那局几步就 done | Breakout 丢球太快或种子不巧 | 打印 T | 换 `--seed` 或加 `--steps`；脚本的 assert 会拦住太短的局 |
| 三个取证的漂移曲线有一条从第 1 步就在 1.0 以上 | 对该空间来说偷懒基线太强（Breakout 常见），或该模型确实弱 | 对照该模型的一步误差比：一步也大于 1 则是空间特性叠加模型噪声 | 不是 bug 就不修，把它写进报告当发现，这正是矛盾案例的原料 |
| report.py 报 control 里找不到模型名 | `--control` 的名字和 npz 里 `model` 字段不一致 | 打印两边字符串 | 统一用 worldmodels、dreamerv3、iris 三个名 |
| 三个虚拟环境来回切换搞混 | 谁都会犯 | `pip list` 看关键包 | 每个终端窗口固定一个环境并改窗口标题，土办法最有效 |

## 11. 前沿与改造

你今天手搓的这套协议，就是公开 benchmark 们各自固化的那个东西。
把四个常见的摆上桌，每个都用"它量什么、它漏什么"过一遍：

| benchmark | 它量什么 | 它漏什么 |
|---|---|---|
| Atari 100k（出自 SimPLe 论文，arXiv:1903.00374；第 09、10 课用过） | 10 万次交互（约两小时游戏时间）预算下的 agent 分数，**样本效率**，规划好的一种苛刻形态 | 模型质量本身：分数混合了模型、策略学习、探索三者的功劳；一步误差再烂，只要策略学得动就能得分。样本效率不等于模型质量，是它最容易被误读的地方 |
| DMC（DeepMind Control Suite；第 06、08 课用过） | 连续控制回报（0 到 1000 的统一量纲是它的好设计），规划好 | 预测准和生成真完全不管；本体感知版任务连视觉都不考；任务之间难度差异大，平均分掩盖单任务崩坏 |
| 1X 世界模型挑战赛（1xgpt 仓库配套；第 11 课用过） | 真实机器人视频上的三段：压缩赛量 teacher-forced 交叉熵（预测准），采样赛量未来帧生成（生成真），评估赛量"能否用世界模型排出 N 个策略的优劣"（规划好，模型当裁判） | 单一形态（EVE 机器人第一人称）、2Hz 低帧率；赛段虽全但相互独立，没回答三个赛段排名矛盾时听谁的 |
| WorldScore（arXiv:2504.00983，ICCV 2025；视频世界模型榜） | 把"世界生成"拆成一串下一场景生成任务，按相机轨迹指定布局，量可控性、质量、动态三轴，3000 个测试样例，统一评了 3D/4D/视频生成共 20 个模型 | 没有 agent、没有奖励：动作被简化成相机轨迹，"规划好"整轴缺席；它证明的是统一协议可行，代价是把"世界模型"收窄成"可控视频生成器"，第 12 课三分观的旧相识 |

对照完你会发现：每个 benchmark 都在四个指标里选了一两个当立场，没有谁全量，
所以"XX 在 YY 榜登顶"这句话，信息量取决于你知不知道 YY 漏了什么。

规模差距（数据量、模型数、每格多种子）是钱能解决的；机制差距有
两条：其一，我们的"生成真"用逐帧 LPIPS 代理，前沿标准是分布级的 FVD（在动作
识别预训练的 I3D 特征上算 Frechet 距离，arXiv:1812.01717），需要的样本量我们
出不起；其二，我们的四指标是并列展示，1X 评估赛那种"模型当策略裁判"的量法
（直接考模型的决策支撑力，不经过策略训练这个放大器）我们没做，它是第 19 课
手术验收时值得补的一刀。

动手改造清单（选做，各自独立）：

1. 课程内统一小基准提案（推荐，纸面作业）：把本课协议升级成一页正式规格，
   三件套写全，数据（第 01 课 CarRacing 数据集的固定切分：哪些轨迹做上下文、
   哪些做考题，种子写死）、指标（本课四个，每个给精确公式和聚合方式）、协议
   （上下文、horizon、窗口数、报告格式、必须附带的立场清单）。目标读者是第 19 课
   做完手术的你自己：任何新模型只要提交一份取证 npz 就能上榜。预算半天。失败
   判据：规格给另一个人看，对方提出两个以上"这里没定义清楚"的问题。
2. 把 FVD 装上：找一个开源 FVD 实现（选型自查，注意 I3D 权重来源），对三份
   取证的 `frames_model` 与 `frames_real` 各算一次分布距离。预算半天。预期：
   窗口太少（8 个）时 FVD 数值抖动大，把同一模型跑三个种子的 FVD 波动量出来，
   你就有了"小样本 FVD 不可信"的第一手证据；如果它和 LPIPS 的排名还不一致，
   恭喜，第五个指标又带来一个新立场。失败判据：装不上 I3D 或算出 NaN，如实
   记录，这也是"FVD 工程成本高"这个行业抱怨的亲测版。
3. IRIS 采样协议消融：把 `world_model_env.py` 里 token 的 `Categorical(...)
   .sample()` 临时改成 argmax（就一行），重跑 Step 5 与 Step 7。预算一小时。
   预期：R1 改善、V 变差或 rollout 变"死板"（多样性塌掉），一个开关同时挪动
   两个指标的实证。失败判据：两个指标都不动，说明 Breakout 的下一帧分布太尖，
   采样和 argmax 本来就几乎重合，这个结论同样值得记录。
4. 探针协议翻转（兑现第 15 课的钩子）：回到第 16 课那批"同一批帧、VAE 目标
   对 JEPA 目标"的表征，把当时的下游探针从线性换成两层带注意力的读出头，其余
   全不动，看"谁的表征好"的结论翻不翻。预算两小时。预期：不保证翻，但探针容量
   加大后两者差距缩小是常见方向；无论翻不翻，你都完成了一次"量法即立场"的对照
   实验。失败判据：无，两种结果都是合格数据点。

两条论文级结论在你的小设置里可以看到方向：LPIPS 论文的核心论断
（深度特征比像素差贴近人眼）在 Step 2 的三组配对上已经复现过缩小版；SimPLe 立起
Atari 100k 时的隐含前提"模型帮策略省样本"与本课的发现"agent 分数量不出模型
本身的病"是一体两面，你表里 IRIS 的控制分和它的一步误差比排名对不上，就是这句
话的实证。

## 12. 论文与延伸

1. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric
   （Zhang et al., CVPR 2018，[arXiv:1801.03924](https://arxiv.org/abs/1801.03924)）
   ，LPIPS 的出生证。带着两个问题读：人类"两选一"数据是怎么采的，为什么这种
   监督恰好校准了"感知"；论文说感知相似性是深度表征的涌现属性、连随机初始化的
   网络都有一点，那 LPIPS 的"立场"藏在哪一层选择里？
2. Towards Accurate Generative Models of Video: A New Metric & Challenges
   （Unterthiner et al., 2018，[arXiv:1812.01717](https://arxiv.org/abs/1812.01717)）
   ，FVD 的出处。带着问题读：Frechet 距离比逐帧距离多捕捉了什么（时间一致性、
   分布覆盖）？用动作识别网络 I3D 的特征当"眼睛"，会天然偏心哪类视频、亏待
   哪类？对照你 Step 2 校准 LPIPS 的思路，想想怎么"校准"FVD。
3. Model-Based Reinforcement Learning for Atari（Kaiser et al., 2019，
   [arXiv:1903.00374](https://arxiv.org/abs/1903.00374)），SimPLe，Atari 100k
   的立标之作。带着问题读：它为什么选 10 万次交互这个数（摘要里那句"两小时
   实时游戏"是人类对照的锚）？它量的是世界模型的哪种好？后来 IRIS、DIAMOND 在
   这个榜上的进步，有多少能归因给"模型更准"，论文自己给了什么证据？
4. WorldScore: A Unified Evaluation Benchmark for World Generation
   （Duan et al., ICCV 2025，[arXiv:2504.00983](https://arxiv.org/abs/2504.00983)）
   ，视频世界模型统一评测榜的代表。带着问题读：它怎么把 3D 场景生成和视频生成
   拉到同一套考题上（相机轨迹当"动作"是关键一招）？可控性、质量、动态三轴与
   本课"预测准、生成真、规划好"哪两轴能对上、哪轴对不上？榜单上 3D 路线在静态
   可控性上压过视频路线，这个结论换一套指标还成立吗？
5. IRIS 论文附表（Micheli et al., [arXiv:2209.00588](https://arxiv.org/abs/2209.00588)，
   第 09 课精读过），这次只读附表：逐游戏的随机分、人类分、各方法分。带着问题
   读：人类归一化分数在 Breakout 和在 Freeway 上"1.0"的含义一样吗？同一个方法
   在 26 个游戏上的排名波动有多大，单游戏分数还能信到什么程度？
6. 选读：**DreamerV3 论文**（[arXiv:2301.04104](https://arxiv.org/abs/2301.04104)，
   第 06 课引过）的评测章节，带着问题读：它为什么强调"一套超参跨 150+ 任务"
   而几乎不报预测误差？按本课的语言，它把哪种"好"立成了默认立场？

第 18 课用今天的协议当纵轴，把参数量、数据量当横轴：小模型上量出的
scaling 趋势，什么时候有资格外推、什么时候只是这份预算下的幻觉，评测学之后，
是缩放实验的手艺。三种「好」和第 33 课的 E 档不是同一条轴：生成真可以把 E0 测得很漂亮，规划好在可重置环境里可以把 E2 测得很漂亮，都不能代替「模型是否在真身体的动作回路里」。
