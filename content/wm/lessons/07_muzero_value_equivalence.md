---
id: 07_muzero_value_equivalence
title: "MuZero 的价值等价模型"
summary: "不重建观察、只预测价值/策略/奖励的模型，凭什么还配叫世界模型？"
unit: engine
play_tools: []
checkpoints:
  - "MCTS 搜索过程记录。"
  - "价值等价探针报告：模型“对”的标准从像素对搬到了决策对。"
---

# 第 07 课：MuZero 只预测决策所需的信息

> 类型：复现（CartPole 方向性复现）+ 体验（Connect4 训练与对弈）<br>
> 建议周期：2-4 天（大头是 Connect4 在后台自己下棋，人的操作集中在头尾）<br>
> 硬件：笔记本 CPU 即可，这是全课程硬件门槛最低的一课，Mac 用户的主场<br>
> 锚定仓库：[werner-duvaud/muzero-general](https://github.com/werner-duvaud/muzero-general)（MIT，教学向社区实现）；论文 MuZero（arXiv:1911.08265）<br>
> 产物：CartPole 训练记录、一份 MCTS 搜索过程记录、一份价值等价探针报告、一盘你和自己训的 MuZero 下的 Connect4

## 1. 这一课做什么

RSSM 和 Dreamer 都要求从隐状态重建观察，因此模型学得怎样还能通过图像直接检查。MuZero
第一次在这门课里去掉这条要求：隐状态不需要还原画面，只需要支持决策。

MuZero（DeepMind，2019，后发表于 Nature）由三个网络组成：representation 把观察压成
隐状态；dynamics 根据“隐状态 + 动作”预测下一个隐状态和即时奖励；prediction 从任意
隐状态给出策略和价值。整个结构没有解码器，也没有重建损失。它只对三个问题负责：
马上会得多少奖励、当前局面值多少、下一步该怎么走。这个立场通常称为**价值等价**
（value equivalence）：模型不必忠于世界的外观，只需让价值与决策结果保持正确。

用法也跟着换了。Dreamer 拿模型当梦境跑步机：展开成千上万条想象轨迹，当训练数据喂给策略。MuZero 拿模型当沙盘：每一步真正落子之前，在模型里做一次 MCTS（蒙特卡洛树搜索），把几十上百条候选未来推演一遍、比完分数再出手。模型从"训练数据的产地"变成了"决策时刻被反复查询的推演引擎"。凭这一套，同一个算法在不知道规则的前提下达到了 AlphaZero 在围棋、国际象棋、将棋上的水平，同时在 Atari 上拿下了当时的最好成绩，AlphaZero 是拿着规则书搜索的，MuZero 连规则书都是自己学的。

Dreamer 把模型展开成虚拟轨迹来训练策略；MuZero 则在每次真实落子前用 MCTS 查询模型，
比较候选分支后再行动。它在不知道环境规则的前提下，把同一套算法用于棋类和 Atari。

接下来会保留一份搜索记录，显示网络先验如何经过多次模拟变成根节点的访问次数分布；还会
用线性探针从 CartPole 的 8 维隐状态读取 4 个物理量，与第 02 课的 VAE 探针直接比较。
第 08 课的 TD-MPC2、以及第 13 课的 JEPA 都会沿着“免重建”继续走，但用不同目标约束
隐状态。本课也会正面讨论：只为价值和策略服务的模型，是否仍应称为世界模型。

术语速查：

| 术语 | 一句人话 |
|---|---|
| MCTS（蒙特卡洛树搜索） | 落子前在脑子里试走几十上百步：优先深挖看起来有戏的分支，越试越有谱 |
| representation / $h$ | 眼睛：把观察压成一条隐状态向量，每步决策只做一次 |
| dynamics / $g$ | 沙盘引擎：吃"隐状态 + 动作"，吐"下一个隐状态 + 这一步的奖励"，搜索深处全靠它 |
| prediction / $f$ | 参谋：看着任一隐状态报两件事，先验策略（该怎么走）和价值（这局面值多少分） |
| 隐状态（hidden state） | $h$ 和 $g$ 产出的那条向量，CartPole 配置里是 8 个数；没有任何约束要求它能还原观察 |
| 价值等价 | 模型合格的标准从"预测得像"换成"算出来的价值/奖励/策略对"：忠于决策，不忠于外观 |
| pUCT | 搜索选分支的打分公式：Q 值项吃老本，先验乘新鲜度项开新路，配比随访问数自动调 |
| 访问次数分布 | 搜索结束后根节点各动作被试次数的占比；既是选动作的依据，也是训练策略头的老师 |
| n-step return | 价值头的学习目标：先收 n 步真实奖励，再用当时搜索算出的根价值兜底 |
| support 表示 | 把标量价值摊成一排"桶"上的概率分布来学，回归问题变分类问题，训练更稳 |
| 自博弈（self-play） | 自己跟自己下棋攒训练数据；单人游戏里就是自己玩自己录 |
| Dirichlet 根噪声 | 训练时往根节点先验里掺一点随机，逼搜索偶尔看一眼冷门招，防止自我固化 |

## 2. 问题

需要判断的是：一个不重建观察、只预测价值、策略和奖励的模型，是否仍然算世界模型。
这个判断可以拆成四个问题：

1. 一个"不重建"的模型在代码里到底长什么样？三个网络的输入输出接口、隐状态在它们之间怎么流动、reward 为什么必须由 dynamics 顺手预测，这些在 muzero-general 的 `models.py` 里总共几百行，本课逐段读完。
2. 搜索怎么从一个随机初始化的网络里逼出棋力？训练开始时三个网络都是垃圾，但"先验 + 搜索"的组合有个奇妙性质：搜索出来的动作分布几乎总比先验强一点。拿搜索结果当老师训练先验，先验变强又让下一轮搜索更强，这个自我提升的循环不需要环境规则参与。Connect4 上你会看着它从乱下长到会堵你的三连。
3. "价值等价"能不能从口号变成实验？论文说隐状态只需保留对决策有用的信息。有用没用，探针可测：从隐状态线性回归原始观察的 4 个物理量，哪个读得出、哪个读不出、和随机初始化的网络差在哪，数字说话。这是本课的交付物之一，也是给第 16 课攒的证据。
4. 两边的论据到底是什么？支持方说"模型的忠实度本来就该按用途度量"，反对方说"回答不了'接下来会看到什么'的东西顶多算个策略评估器"。这里不裁决，但要求你能把双方论据完整摆出来，第 5.2 节交货。

顺带划清本课的诚实边界：CartPole 属于方向性复现（从零训练、结果与"任务被解决"同方向）；Connect4 属于体验档，它的完整配置是 10 万训练步的残差网络，笔记本 CPU 跑不满也不必跑满，我们训到"能下出人样的棋"就收，棋力验收看趋势不看绝对水平。仓库 README 自己也把丑话说在了前面："我们并不总能系统性地达到人类水平"、"某些环境训练一段时间后会观察到性能回退"、"提供的配置肯定不是最优的，我们目前不专注于超参数优化"。这是一个 MIT 协议的教学向社区实现（README 自述 educational purpose），作者的测试机是 16GB 内存加 GTX 1050Ti Max-Q 的笔记本，这恰好是它当本课锚定的理由：它证明这套东西不需要 DeepMind 的机房。

## 3. 准备

- 手艺依赖，不是产物依赖。这里不用前六课训出的任何模型，单独可跑。但三样手艺会直接用上：第 01 课的留证据习惯、第 02 课的线性探针方法（本课探针实验与它正面对照）、第 03 课"动作条件"的判断力（dynamics 也可能动作盲，检验思路相通）。
- 一台笔记本就够。CartPole 用全连接网络，作者本人就是在 GTX 1050Ti 级别的笔记本上测的，纯 CPU 完全可行；Connect4 的残差网络在 CPU 上慢但能跑（我们本来也不跑满）。全课程硬件最轻松的一课，之前因为没有 NVIDIA 卡跳过 05/06 课实验的 Mac 用户，这课一步不缺地跟。
- 依赖是 2021 年的技术栈，给它开独立虚拟环境。仓库的 `requirements.lock` 钉死了 torch 1.10.0、gym 0.21.0、ray 1.5.2、tensorboard 2.7.0，这套组合在 Python 3.8/3.9 上最顺（路线 A，x86 Linux 或 Intel Mac 首选）。Apple Silicon 上 torch 1.10 没有官方安装包，走路线 B：新版 torch/ray/tensorboard/seaborn 加钉在 0.21 的 gym（本仓库只用这些库的基础接口，但这属于偏离官方锁定版本的组合，出了怪问题先回路线 A 或上 Colab，仓库自带 `notebook.ipynb`，README 给 Windows 用户的官方建议就是它）。gym 0.21 在新版 pip 下装不上是社区里人尽皆知的老坑，修法见第 10 节。
- 可选：graphviz。仓库的诊断工具能把整棵 MCTS 搜索树画成 PDF，但依赖 graphviz，注意 `requirements.lock` 里没有它，要额外装（Python 包加系统二进制，Mac 用 brew）。不装也不报错，只是少一张很好看的图，代码会打印一句提示然后跳过。
- 磁盘几百 MB 足够，这里没有大数据集，观察是 4 个数或一个 6×7 的棋盘，不是像素。

## 4. 学习目标

1. 白纸画出 $h$、$g$、$f$ 三个网络的输入输出，以及 MCTS 一次模拟从根到叶再回传的完整路径，标出模型在哪几处被调用；
2. 写出 pUCT 公式，指着 `self_play.py` 说出公式每一项对应哪几行代码，并解释访问数不大时 $\log$ 那一项为什么约等于常数；
3. 说清训练的三个目标各自从哪来：策略目标来自搜索的访问次数分布、价值目标来自 n-step return、奖励目标来自环境真值，以及局终之后的"吸收态"怎么填；
4. 解释隐状态在没有重建约束时为什么（在损失还能下降的前提下）不会坍缩成常数，并对比 VAE、RSSM、MuZero 三家各靠什么撑住 latent；
5. 用自己的探针实验给"价值等价"提供一份证据，并做到无论结果偏向哪个方向都能如实解读；
6. 对"MuZero 算不算世界模型"完整复述两边论据，并给出自己的立场和依据。

## 5. 原理

五个机制，仍按老节奏走：为什么需要（直觉）、怎么运转（机制）、精确定义（数学）、在源码哪里（代码）、怎么证明做对了（验证）。

### 5.1 三网络分工：眼睛、沙盘、参谋

下盲棋的棋手不摆棋盘。他脑子里维护的东西不必是 64 个格子的照片，只要够他推演："马走到这里，对方象必须回防，三步后我多一个兵。"支撑推演的三种能力，正好对应 MuZero 的三个网络：把眼前局面收进脑子（representation，下面记 $h$）；在脑子里落一步子、局面跟着变（dynamics，记 $g$）；看着脑中任一局面判断"谁优、下哪"（prediction，记 $f$）。三样都不需要"把脑中局面画回棋盘"的能力，盲棋手可能真的画不准棋盘照片，但棋照样下赢你。

每步决策开始时，$h$ 把当前观察压成隐状态 $s^0$，这是全程唯一一次接触观察。之后搜索
每往深走一步，就调用一次 $g$：给它当前隐状态和一个候选动作，它吐出下一个隐状态和
即时奖励。reward 必须由 $g$ 预测，因为搜索深处没有真实环境，树里第三层"如果我先左
再右再左"的奖励只能由模型给出。每个新展开的节点再由 $f$ 给出先验策略和价值估计。

记观察为 $o_t$，候选动作序列为 $a^1, a^2, \dots$，MuZero 的模型是三段函数的组合：

$$
s^0 = h(o_t), \qquad (s^k, r^k) = g(s^{k-1}, a^k), \qquad (\mathbf{p}^k, v^k) = f(s^k)
$$

上标 $k$ 是"想象中往前走了几步"。对照第 01 课的定义 $P(s_{t+1} \mid s_t, a_t)$：$g$ 的位置和它一模一样，区别只在 $s$ 的身份，第 01 课的 $s$ 由 VAE 重建损失锚在观察上，这里的 $s$ 什么锚都没有，是纯粹的内部记号。另有两个工程细节：其一，value 和 reward 都不用标量回归，而是摊在 `2 * support_size + 1` 个"桶"上学分类分布（CartPole 配置 support_size = 10，即 21 个桶），数值先经一个压缩变换（代码注释引用 arXiv:1805.11593）再入桶；其二，隐状态每次产出后都被 min-max 归一化到 $[0,1]$ 区间（论文附录的做法），给这个没有锚的空间套一层数值笼头。

全部在 `models.py`：工厂类 `MuZeroNetwork` 按 `config.network` 分发到 `MuZeroFullyConnectedNetwork`（CartPole 用）或 `MuZeroResidualNetwork`（Connect4 用）。三个方法名与论文一字不差：`representation(observation)` 返回归一化隐状态；`dynamics(encoded_state, action)` 返回 `(next_encoded_state_normalized, reward)`，动作以 one-hot 拼进输入；`prediction(encoded_state)` 返回 `(policy_logits, value)`。对外再包两层便捷接口：`initial_inference(observation)` 一次跑完 $h + f$（返回值里的 reward 恒为零分布，代码注释写明只为接口一致），`recurrent_inference(encoded_state, action)` 一次跑完 $g + f$，搜索代码只认识这两个入口。桶和标量的换算是模块级函数 `support_to_scalar` / `scalar_to_support`。

CartPole 配置下把 shape 数一遍：观察 $(1,1,4)$ 展平成 4 维进 $h$，隐状态 `encoding_size = 8`，$g$ 的输入是 $8 + 2$（隐状态拼 2 维 one-hot 动作），policy logits 是 2 维，value/reward 各 21 维 logits。第 7 节 Step 5 的胶水脚本会真的调一遍这些接口，shape 对不上当场就崩，崩了说明你改错了配置。

### 5.2 价值等价：模型不必忠于世界，只需忠于价值计算

你公司楼下有个停车场收费员，他对世界的模型粗糙得离谱：他不知道你的车是什么牌子、什么颜色、后备箱里有什么，他的"世界状态"只有一张时间戳。但对"该收你多少钱"这个决策，他的模型和一台全知摄像机算出的结果分毫不差。**对于收费这个用途，他的破模型和上帝视角是等价的。**MuZero 把这个立场做成了训练目标：隐状态爱长什么样长什么样，只要沿着任意动作序列展开后，模型报出的价值、奖励、策略和"真实世界里会算出的数"一致，它就是合格的模型。Grimm 等人 2020 年把这个直觉形式化成了"价值等价原则"：两个模型等价，当且仅当对给定的一类策略和价值函数，它们诱导的贝尔曼更新结果相同，模型的忠实度应当以用途度量，而重建观察这个用途，决策者其实从来没点过单。

落到训练上就一句话：从回放的轨迹里取一个时刻 $t$，把模型沿着**当时真实执行过的动作**展开 $K$ 步（仓库 `num_unroll_steps`，CartPole 配 10，论文用 5），每一步的三个输出各自对答案，policy 对搜索留下的访问分布，value 对 n-step return，reward 对环境真值。损失里没有第四项。隐状态是三条损失曲线中间的自由变量，梯度爱把它捏成什么样就捏成什么样。

单个时刻的损失（略去正则项与 PER 权重）：

$$
l_t(\theta) = \sum_{k=0}^{K} \Big[ \, l^p\big(\pi_{t+k},\, \mathbf{p}_t^k\big) + l^v\big(z_{t+k},\, v_t^k\big) + l^r\big(u_{t+k},\, r_t^k\big) \Big]
$$

$\pi$ 是搜索访问分布，$z$ 是 n-step return，$u$ 是真实奖励；三项在仓库里全部用交叉熵实现（value/reward 因为有桶表示，天然可以这么做，代码注释说比 MSE 收敛更好）。两个防爆细节值得记：展开链条上每一步的隐状态梯度减半（`hidden_state.register_hook(lambda grad: grad * 0.5)`，照论文附录），每步损失再除以展开步数拉平量纲；$k=0$ 那步不算 reward 损失，因为 `initial_inference` 的 reward 本来就是摆设。现在回头正面回答本课的核心问题，**这还算世界模型吗？**两边论据都摆全：

支持方，三条。第一，第 01 课那条主干循环，压成状态、按动作预测下一状态、展开多条未来、打分、选动作，MuZero 一个环节不少，只是"预测下一状态"的验收方式从"重建出观察"换成了"报对价值统计量"；接口齐全的东西你很难说它不是世界模型。第二，"世界里与决策无关的细节本来就不属于需要建模的世界"，CarRacing 的草地纹理你在第 02 课已经确认过是废话，价值等价只是把"扔废话"从副作用升格成设计原则。第三，实证：Connect4 没人告诉它规则，它学会了堵三连，环境的因果结构里对赢棋有用的那部分，确实被它抓在手里了。

反对方，也是三条。第一，它回答不了"接下来会看到什么"：不能生成观察，不能给人看未来，你没法像第 05/06 课那样把想象序列解码出来用眼睛验货，模型的一切只能靠最终分数间接检验，放到第 17 课"预测/生成/规划"的三分评测里，它只在"规划"一列有成绩。第二，学到的模型和奖励函数焊死了：换个任务（同一个棋盘，改成"比谁先连成三个"），价值、策略、奖励三个头全部作废，隐状态里剩下什么能迁移没有保证；而一个忠于观察的模型至少环境动力学部分照用。第三，哲学上它更像"可展开的价值函数"或者说策略评估器，它对世界的全部知识都以"对我得分有何影响"的形式存在，你可以质疑这到底是建模世界，还是只是把 Q 函数做深了。

这里不裁决。但给一个判断工具：争论双方其实共享一个前提，模型的好坏由用途定义。分歧只在"你打算要多少种用途"。只干一个任务，价值等价是把刀，专切有用的肉；想要一个能换任务、能被人检查、能支撑多种决策的世界引擎，重建或表征预测阵营的锚就有它的道理。把这段争论放进你的笔记，第 16 课对决时它是第一件呈堂证物。

`trainer.py` 的 `update_weights` 与 `loss_function`：三项交叉熵、`value_loss_weight`（注释里写论文建议 0.25，CartPole 配置实际给 1）、两处梯度缩放 hook 都在这一个文件里。你可以用第 01 课的老办法验证"没有重建项"：全文搜 `decoder`、搜 `reconstruction`，零命中。

价值等价是可检验的预言：隐状态应当保留决策相关的信息、丢弃无关的。第 7 节 Step 6 用线性探针把这句话变成四个 $R^2$ 数字，无论结果偏哪边，解读规则提前写好（见那一步的"如实解读"清单）。

### 5.3 pUCT：搜索怎么在"吃老本"和"开新路"之间分配模拟次数

你有 50 次模拟的预算，根节点下有若干个候选动作。全花在目前 Q 值最高的动作上？万一它只是先验一开始瞎捧的，你就被自己的偏见锁死了。均匀撒？绝大多数预算浪费在明显的臭棋上。pUCT 是这场预算分配的裁判，它给每个候选打一个分，分数由两股力量拉扯：这个分支实测的平均回报（利用），和"先验看好它但它还没被试够"的程度（探索）。关键性质是探索项随访问次数递减，一个分支被试得越多，它就越需要靠实打实的 Q 值说话，先验的推荐信会慢慢失效。

一次模拟四拍：从根出发，每层用 pUCT 选分数最高的孩子往下走，直到碰到没展开过的叶子（选择）；对叶子调一次 `recurrent_inference`，拿到新隐状态、奖励、先验和价值（扩展与评估）；把这个价值沿来路一路回加，路径上每个节点的访问数加一（回传）；回到根，开始下一次。全部预算用完后，根节点各动作的访问次数就是搜索的结论。训练时选动作按访问次数的温度分布采样（CartPole 配置：训练进度前 50% 温度 1.0，到 75% 降到 0.5，之后 0.25；评估时温度 0，直接取访问最多的动作）。另有两个防自闭装置：训练时根节点先验会掺 Dirichlet 噪声（`root_dirichlet_alpha = 0.25`，掺入比例 `root_exploration_fraction = 0.25`），保证冷门招偶尔也被看一眼；树里的 Q 值用一个 `MinMaxStats` 对象按树内见过的最大最小值归一化后再参与打分，免得不同环境的奖励量纲把探索项淹掉。

仓库 `ucb_score` 的实现（与论文伪代码一致），对父节点 $s$（访问数 $N$）的某个孩子（访问数 $n_a$、先验 $P(a)$、边上奖励 $r_a$、平均价值 $Q(a)$）：

$$
\mathrm{score}(a) = \bar{Q}(a) + P(a)\cdot\frac{\sqrt{N}}{1+n_a}\cdot\Big(c_{\mathrm{init}} + \log\frac{N + c_{\mathrm{base}} + 1}{c_{\mathrm{base}}}\Big)
$$

其中 $\bar{Q}(a)$ 是 $r_a + \gamma\, Q(a)$ 经 MinMax 归一化后的值（没访问过的孩子记 0；双人游戏取 $-Q(a)$，因为孩子的局面轮到对手），$c_{\mathrm{init}} = 1.25$、$c_{\mathrm{base}} = 19652$（`pb_c_init`、`pb_c_base`，沿用 AlphaZero 的取值）。看清 $\log$ 项的脾气：$N$ 只有几十几百时，$N / 19652$ 约等于零，整个括号就是常数 1.25，它是为 $N$ 上十万的场合预留的缓慢加压阀，我们这种 50 次模拟的小场面里探索强度基本就由 $\frac{\sqrt N}{1+n_a}$ 决定。

全在 `self_play.py`：`MCTS.run` 是主循环（签名里有 `add_exploration_noise` 开关，返回根节点和一个含 `max_tree_depth`、`root_predicted_value` 的信息字典）；`Node` 存着 `visit_count`、`prior`、`value_sum`、`reward`、`children`、`hidden_state`，`value()` 就是 `value_sum / visit_count`；`ucb_score` 实现上面的公式；`select_action` 是静态方法，实现温度采样（温度 0 取 argmax，无穷大等价均匀随机，其余按 `visit_counts ** (1/temperature)` 归一化采样）。

Step 5 的搜索记录会把先验和访问分布并排打印：如果两者几乎一样，说明搜索没干活（模拟数太少或 Q 值全被归一化抹平）；健康的搜索里，访问分布应当比先验更尖，而且模拟数从 10 加到 200 时越来越尖、树也越钻越深。

### 5.4 训练目标从哪来：搜索给自己出考题

三个头要学，老师是谁？MuZero 的答案自成一体：**搜索本身就是策略改进算子**。给定同一个网络，"先验 + 50 次搜索"的动作分布几乎总比裸先验强，搜索用算力买来了一次提纯。那就把提纯后的结果（访问分布、搜索根价值）存进回放缓冲，回头当标签训练网络；网络变强，下一轮搜索的起点更高，提纯效果更好。整个上升螺旋不需要环境规则，只需要环境肯报 reward。

三个目标三条来路。策略目标 $\pi_t$：自博弈每走一步，把根节点各孩子的访问数归一化存下（`GameHistory.store_search_statistics`，存进 `child_visits`）。价值目标 $z_t$：n-step return，真实奖励打底、搜索价值兜底：

$$
z_t = \sum_{i=1}^{n} \gamma^{\,i-1}\, u_{t+i} + \gamma^{\,n}\, \nu_{t+n}
$$

$\nu_{t+n}$ 是当时存下的搜索根价值 `root_values[t+n]`（开了 Reanalyse 的话会用新网络重算一遍，CartPole 配置 `use_last_model_value = True`），CartPole 配 `td_steps = 50`、`discount = 0.997`；bootstrap 越过局终就只剩奖励和，价值记 0。奖励目标最省事，环境真值直接抄。还有个边角要交代：展开 $K$ 步可能越过局终，越界的步按"吸收态"填，价值 0、奖励 0、策略给均匀分布。回放侧默认开了优先级采样（`PER = True`，优先级是根价值与 n-step 目标的偏差，思路来自 arXiv:1803.00933），预测得越离谱的时刻越常被抽出来补课。

`replay_buffer.py` 的 `compute_target_value` 与 `make_target` 逐行对应上面每句话；`Reanalyse` 类在同一个文件里。自博弈侧的存目标动作在 `self_play.py::GameHistory`。

TensorBoard 的 `3.Loss` 组把三条损失分开画。健康形态是三条都往下走但不归零（自博弈在不断产出更强的对局，考题水涨船高）；哪一条独自发疯，第 10 节症状表有它的行。

### 5.5 没有重建约束，隐状态凭什么不坍缩

第 02 课的 VAE latent 靠重建损失撑着：敢丢信息，解码那头立刻疼。现在重建没了，隐状态凭什么不摆烂成常数向量？答案：三个头就是三根承重柱。假设 $h$ 真把所有观察都映射到同一个点，那 $f$ 在所有局面上只能报同一个价值，可回放缓冲里明摆着有的局面 n-step return 高、有的低，价值损失立刻压不下去；策略头同理，杆子左倒和右倒的访问分布截然相反，常数隐状态一个都拟合不了；reward 头沿着展开链条还会追究 dynamics：动作序列不同、奖励不同，$g$ 的输出必须跟着分岔。所以只要三条损失还在下降，隐状态就被迫携带区分这些答案所需的全部信息。这里的信息筛选是隐式的：**梯度只保护"改变了价值/策略/奖励预测"的信息，其余信息没有靠山，被挤掉时无人喊疼。**这正是价值等价的机械实现。

机制对照。把三家的承重结构摆一排：VAE 靠重建项（第 02 课），信息保留标准是"像素还原得像"；RSSM 靠重建加 KL（第 05 课），标准是"还原得像且编码规整"；MuZero 靠 value/policy/reward 三头，标准是"决策统计量报得对"。第 08 课的 TD-MPC2 会给出第四种承重（latent 一致性），第 13 课 JEPA 给第五种（EMA target 加表征预测），防坍缩手段的谱系，是贯穿这门课下半场的一条暗线。另外两句实话。其一，min-max 归一化把每个隐状态压进 $[0,1]$，这不防坍缩（常数向量也能归一化），防的是这个没有外部锚的空间数值漂移，给展开链条上的梯度一个稳定的落脚面。其二，不坍缩不等于不退化：README 承认部分环境训练一段时间后性能回退，仓库没有给出定论；三个头的目标全是自举出来的（搜索教网络、网络又当搜索的地基），这种结构对超参本来就敏感，README 也明说超参没有系统调优。你在 TensorBoard 上看到回退曲线时，那是已知现象，按第 10 节处理，不必怀疑自己装错了什么。

用法之别，顺手说透。同样是"在模型里跑未来"，Dreamer 和 MuZero 的姿势差在时机上。Dreamer 在**训练时**大量想象：模型展开成千上万条轨迹喂 actor-critic，决策时策略网络一次前向就出手，模型不在场。MuZero 在**决策时**搜索：每步落子前现场推演几十上百条未来，模型被反复查询；训练时模型只展开 $K$ 步对答案，从不长程做梦。代价结构因此相反，Dreamer 出手快但想象质量全靠模型长程不跑偏（第 03 课的误差滚雪球你见过），MuZero 每步出手都贵（50 次模拟就是 50 次网络前向）但只需模型在浅层展开里靠谱。这也解释了为什么 MuZero 家族称霸棋类（每步值得多花算力深算）而 Dreamer 家族长于连续控制和长时程任务。两种姿势第 08 课会在 TD-MPC2 身上合流：训练像 MuZero 一样免重建，决策像 Dreamer 一样在潜空间滚动规划。

坍缩与否可以直接量：Step 6 的探针脚本顺手输出隐状态各维的方差，若某些维度方差接近零，说明 8 维里有闲置维（第 02 课你见过同样的现象，当时叫"拧了没反应的旋钮"）；四个 $R^2$ 若全部趴在零附近而训练分数又正常，那才是怪事，先怀疑脚本加载错了权重。

## 6. 源码导读

仓库总共十来个 Python 文件，一天能读完，建议顺序与问题如下：

| 文件 | 是什么 | 带着什么问题读 |
|---|---|---|
| `muzero.py` | 入口与交互菜单 | 无参数运行时列出 games 目录让你选，随后的菜单选项逐字是：Train、Load pretrained model、Diagnose model、Render some self play games、Play against MuZero、Test the game manually、Hyperparameter search、Exit，每一项各调用 MuZero 类的哪个方法？|
| `games/cartpole.py` | 本课主角的配置与环境包装 | `MuZeroConfig` 里 `encoding_size = 8`、`num_simulations = 50`、`td_steps = 50` 各出现在第 5 节哪个公式里？`Game.step` 为什么把观察包成 $(1,1,4)$？|
| `games/connect4.py` | 双人游戏配置 | `network = "resnet"`、`num_simulations = 200`、`max_moves = 42`；评估对手 `opponent = "expert"` 的启发式代码在哪（提示：扫描所有 4×4 子棋盘找三连）？|
| `models.py` | 三网络本体 | `representation`/`dynamics`/`prediction` 的返回值分别是什么？min-max 归一化出现在哪两个方法里？`support_to_scalar` 的加权求和在哪行？|
| `self_play.py` | 自博弈 + MCTS + 数据记录 | `ucb_score` 与 5.3 的公式逐项对上；`GameHistory` 存了哪九样东西？`store_search_statistics` 怎么把访问数变成策略目标？|
| `replay_buffer.py` | 目标生成与采样 | `compute_target_value` 的 bootstrap 索引怎么算？越过局终的三种情况 `make_target` 各怎么填？|
| `trainer.py` | 损失与更新 | 三项损失都是交叉熵？两处 `register_hook` 各在缩放什么梯度？`checkpoint_interval = 10` 时权重多久写回共享存储一次？|
| `shared_storage.py` | 训练状态的公告板 | `save_checkpoint` 默认写到哪个路径？（这决定了你 Ctrl+C 之后还剩什么。） |
| `diagnose_model.py` | 诊断工具 | `compare_virtual_with_real_trajectories` 对比的是哪两条轨迹？`plot_mcts` 用什么库画树、输出什么文件、缺依赖时怎么表现？|

`requirements.lock` 也值得看一眼：钉死的版本就是第 3 节那些坑的来源；`notebook.ipynb` 是 Colab 逃生通道。

## 7. 实验

以下所有命令都在仓库根目录、激活了本课专用虚拟环境的终端里执行。

### Step 1: 克隆与环境

```bash
git clone https://github.com/werner-duvaud/muzero-general.git
```

进入目录后建虚拟环境（路线 A 用 Python 3.8/3.9）：

```bash
python3.9 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.lock
```

装不动的按第 3 节选路线：Apple Silicon 换新版 torch/ray/tensorboard/seaborn 加 gym 0.21（gym 装不上先看第 10 节第一行）；都不想折腾就开 Colab 跑仓库自带的 `notebook.ipynb`。想要搜索树 PDF 的顺手装可选件：

```bash
pip install graphviz
```

系统层的 graphviz 二进制也要有（Mac 用 `brew install graphviz`，Debian 系用包管理器装 `graphviz`）。

### Step 2: 从零训练 CartPole

先用交互入口感受一下这个仓库的性格：

```bash
python muzero.py
```

它会列出 `games/` 目录下的全部游戏让你选编号（列表按文件名排序，找到 cartpole），然后给出第 6 节表格里那八个菜单项。选 `Train` 即开训。以后重跑可以走直通车，带参数启动会跳过菜单直接训练：

```bash
python muzero.py cartpole
```

训练过程会持续把权重落盘到 `results/cartpole/<时间戳>/`（`checkpoint_interval = 10`，即每 10 个训练步就把权重写回共享存储并保存 `model.checkpoint`，Ctrl+C 停掉不心疼；`replay_buffer.pkl` 则在你退出训练时写盘，Step 6 的探针要用它，所以请让训练自然结束或用 Ctrl+C 从日志循环里退出，别直接杀终端）。完整跑完是 `training_steps = 10000` 步，全连接小网络加 `num_workers = 1` 个自博弈进程，笔记本 CPU 的耗时在小时级，具体看机器，判断进度别掐表，看下一步的曲线。

### Step 3: 用 TensorBoard 读训练诊断

另开一个终端（同一虚拟环境）：

```bash
tensorboard --logdir ./results
```

浏览器打开它给的地址。这个仓库的日志分三组，每组都值得会读：

- `1.Total_reward` 组：`1.Total_reward` 是评估局总分，CartPole-v1 每撑一步得 1 分、500 步封顶，所以这条曲线爬向 500 并贴住，就是"解决"的形态；`2.Mean_value` 是评估局的平均搜索根价值；`3.Episode_length` 在 CartPole 里和总分应当重合（一步一分）。
- `2.Workers` 组：`1.Self_played_games` 与 `2.Training_steps` 要一起涨；`5.Training_steps_per_self_played_step_ratio` 是训练与自博弈的配速比（配置里 `ratio = 1.5`），它失控说明某一侧卡死了。
- `3.Loss` 组：总损失加三个分项。对照 5.4 的"验证"读。

顺带做一件诚实的事：README 说部分环境训练一段时间后会性能回退。盯着 `1.Total_reward`，如果它冲上 500 之后又滑下来，把那一段截图存进证据目录，这是你见到的"自举训练不稳定"标本，不是你的错。分数曲线到过 500 附近、且大部分评估局能贴顶，本步即算过关。

### Step 4: 仓库自带的模型诊断

回到交互菜单（`python muzero.py`，选 cartpole），先用 `Load pretrained model` 载入你刚训的 run（菜单会列出 `results/cartpole/` 下的历史目录供选），再选 `Diagnose model`。它调用 `diagnose_model.py`，做一件很聪明的事：让模型完全脱离真实环境、只靠 $g$ 和 $f$ 往前"盲走"30 步（每步照常做 MCTS），再把同一串动作放回真实环境重放，对比两条轨迹、标出开始分岔的步数（`trajectory_divergence_index`）。弹出的 seaborn 热图里重点看两张：`Prior policies`（裸先验）对 `Policies after planning`（搜索后的访问分布），两张图的差距就是"搜索买到的提纯"的可视化；以及 `MCTS depth` 随时间的变化。装了 graphviz 的话，当前目录还会多一个 `mcts.pdf`：整棵搜索树，每个节点标着 Action、Value、Visit count、Prior、Reward，访问最多的主干染成橙色。没装则终端打印一句提示后照常出热图。

### Step 5: 把一次 MCTS 搜索过程记录下来（交付物一）

诊断工具给的是图，我们再写一小段胶水直接从 `MCTS` 类里取数，顺便验证你真的能徒手驱动这套接口。在仓库根目录建 `probe_mcts.py`：

```python
import torch

import models
from games.cartpole import MuZeroConfig, Game
from self_play import MCTS

CHECKPOINT = "results/cartpole/model.checkpoint"  # 仓库自带权重；看自己的模型就换成你的 run 目录下的 model.checkpoint

config = MuZeroConfig()
model = models.MuZeroNetwork(config)
model.set_weights(torch.load(CHECKPOINT)["weights"])
model.eval()

game = Game(seed=0)
observation = game.reset()

for sims in [10, 50, 200]:
    config.num_simulations = sims
    with torch.no_grad():
        root, info = MCTS(config).run(
            model,
            observation,
            game.legal_actions(),
            to_play=0,
            add_exploration_noise=False,
        )
    print(f"=== num_simulations = {sims} ===")
    print(
        f"先验根价值 {info['root_predicted_value']:+.3f}  "
        f"搜索后根价值 {root.value():+.3f}  "
        f"最大树深 {info['max_tree_depth']}"
    )
    for action, child in sorted(root.children.items()):
        print(
            f"  {game.action_to_string(action):32s}"
            f"visits={child.visit_count:4d}  prior={child.prior:.3f}  "
            f"Q={child.value():+.3f}  r={child.reward:+.3f}"
        )
```

```bash
python probe_mcts.py
```

每个动作一行：访问次数、先验、平均价值 Q、dynamics 预测的即时奖励。读法对照 5.3 的"验证"：先验来自一次 `prediction` 前向，访问分布是搜索的结论；模拟数从 10 到 200，看三件事，访问分布是否越来越向一个动作集中、最大树深是否增长、搜索后根价值相对先验根价值动了多少。把三段输出连同你的三句解读存成 `mcts_log.md`，这是交付物一。想看双人版的戏码，把 import 换成 `games.connect4`、`CHECKPOINT` 换成 Step 7 训出的权重、`to_play` 按局面填，再跑一遍，200 次模拟的树和堵三连的 Q 值分布比 CartPole 精彩得多。

### Step 6: 价值等价探针（交付物二）

第 02 课你从 VAE 的 32 维 latent 里线性读出了"路的弯度"；现在对 MuZero 的 8 维隐状态做同一套手术，问四个问题：小车位置 $x$、小车速度 $v$、杆角 $\theta$、杆角速度 $\omega$，各还能读出多少？对照组是同结构、随机初始化、没训过一步的网络。在仓库根目录建 `probe_value_equiv.py`：

```python
import pickle

import numpy
import torch

import models
import self_play  # noqa: F401  反序列化 replay_buffer 里的 GameHistory 需要它
from games.cartpole import MuZeroConfig

RUN_DIR = "results/cartpole/换成你的时间戳目录"

torch.manual_seed(0)
config = MuZeroConfig()

with open(f"{RUN_DIR}/replay_buffer.pkl", "rb") as f:
    buffer = pickle.load(f)["buffer"]
obs = numpy.concatenate(
    [numpy.array(g.observation_history, dtype=numpy.float32) for g in buffer.values()]
)  # 形状 (N, 1, 1, 4)
targets = obs.reshape(len(obs), 4)
names = ["小车位置 x", "小车速度 v", "杆角 theta", "杆角速度 omega"]


def probe(weights):
    model = models.MuZeroNetwork(config)
    if weights is not None:
        model.set_weights(weights)
    model.eval()
    with torch.no_grad():
        H = model.representation(torch.from_numpy(obs)).numpy()  # (N, 8)
    print("  隐状态各维方差：", numpy.round(H.var(axis=0), 4))
    n = len(H)
    idx = numpy.random.RandomState(0).permutation(n)
    tr, te = idx[: int(n * 0.8)], idx[int(n * 0.8):]
    A = numpy.hstack([H, numpy.ones((n, 1))])
    scores = []
    for d in range(4):
        w, *_ = numpy.linalg.lstsq(A[tr], targets[tr, d], rcond=None)
        pred = A[te] @ w
        ss_res = ((targets[te, d] - pred) ** 2).sum()
        ss_tot = ((targets[te, d] - targets[te, d].mean()) ** 2).sum()
        scores.append(1 - ss_res / ss_tot)
    return scores


print("训练过的网络：")
trained = probe(torch.load(f"{RUN_DIR}/model.checkpoint")["weights"])
print("随机初始化对照：")
random_init = probe(None)
print(f"\n{'物理量':14s}{'训好 R2':>10s}{'随机 R2':>10s}")
for name, a, b in zip(names, trained, random_init):
    print(f"{name:14s}{a:10.3f}{b:10.3f}")
```

```bash
python probe_value_equiv.py
```

如实解读，提前把规则写死。先记下一个和第 02 课的关键不同：那里是 12288 个像素挤进 32 个数，扔信息天经地义；这里是 4 个数住进 8 维空间，容量根本不缺，"扔"没有算术上的必然性。所以看结果时按下面的分支走，别硬把数字掰成自己期待的故事：

- 四个 $R^2$ 全高，训好与随机差不多。结论：CartPole 太小，价值等价在这里表现为"没必要扔"，没有重建约束不等于信息被扔掉，只是没人强制保留。这个结果不推翻价值等价，只说明它的"扔"要在观察维度远大于决策所需时才会显形（这正是 Connect4 用 $3\times6\times7$ 观察配残差网络的场合，可惜棋盘的"原始观察"没有干净的低维物理量可回归，线性探针在那边不好使）。
- $x$（或 $v$）明显低于 $\theta$、$\omega$。方向与价值等价的预言一致：CartPole 的奖励在 $\vert x\vert$ 撞到 2.4 的边界之前几乎不关心小车在哪，而杆角和角速度直接决定生死。位置信息没有靠山，被挤掉不冤。
- 训好的反而比随机初始化的低。别慌，这同样是价值等价的证据，而且更硬：训练主动重组了空间，把与决策无关的方向压平了；随机网络只是一个恰好可逆的非线性投影，什么都没舍弃。
- 数字很怪（负的 $R^2$、方差为零的维度扎堆）。先查工程：`RUN_DIR` 是否指对、buffer 是否太小（训练太短就退出会只有几局数据）。此外记住一个统计陷阱：buffer 里的数据来自越训越好的策略，杆子基本立着、$x$ 变化范围窄，方差小的维度上 $R^2$ 本身就抖。

把两行四列的 $R^2$ 表、隐状态方差、你命中的分支和一段对照第 02 课探针的讨论写成 `value_equiv_report.md`，交付物二完成。无论落在哪个分支，这份报告在第 16 课三路对决时都是你自己的第一手证据。

### Step 7: 训练 Connect4 并与它对弈

仓库没有自带 Connect4 预训练权重（`results/` 里只有 cartpole 和 lunarlander），想下棋只能自己训：

```bash
python muzero.py connect4
```

完整配置是 10 万步、3 个残差块、每步 200 次模拟，笔记本 CPU 别指望跑满，也不需要：训练每 10 步都在落盘 checkpoint，挂后台跑半天或过夜，随时 Ctrl+C 收工。之后回交互菜单（选 connect4），`Load pretrained model` 选你的 run，再选 `Play against MuZero`，这条菜单项以 MuZero 执先手（`muzero_player=0`）开局，轮到你时终端提示输入落子列号（0 到 6），棋盘直接打印在终端里。下几盘，重点观察一件事：**你摆出三连时它堵不堵。**堵，说明"三连会输"这条从没写进任何代码的规则，已经长在它的价值头里了。TensorBoard 上还有个客观版棋力计：Connect4 配置的评估对手是 `opponent = "expert"`（内置的扫描 4×4 子棋盘找三连的启发式棋手），`1.Total_reward/4.MuZero_reward` 对 `5.Opponent_reward` 的差距拉开的过程，就是棋力的成长曲线。

### Step 8: 留证据

老规矩，`results/` 旁建 `NOTES.md`：两次训练的完整命令与训练步数、你的依赖路线（A 还是 B、Python 与关键包版本）、TensorBoard 关键曲线截图（含性能回退段，如果撞见了）、`mcts_log.md` 与 `value_equiv_report.md` 的路径、Connect4 对弈战绩和"它堵没堵三连"的观察。第 16 课回来取证时，这页纸就是索引。

## 8. 配置与预算

| 实验 | 配置要点 | 硬件与耗时 | 验收口径 |
|---|---|---|---|
| CartPole 训练（必做） | 仓库默认：全连接网络、encoding 8、50 次模拟、10000 步 | 笔记本 CPU，小时级（看机器） | `1.Total_reward` 到过 500 附近且多数评估局贴顶 |
| MCTS 搜索记录（必做） | 模拟数扫 10/50/200，关探索噪声 | CPU，分钟级 | 三段输出加三句解读 |
| 价值等价探针（必做） | 自己 run 的 checkpoint 加 replay buffer，随机初始化对照 | CPU，分钟级 | 两行四列 $R^2$ 表加如实解读 |
| Connect4 训练与对弈（必做，体验档） | 仓库默认：残差网络 3 块、200 次模拟；不跑满 10 万步 | CPU 后台半天到过夜；有 NVIDIA 卡会自动用上（`train_on_gpu` 按 `torch.cuda.is_available()` 自动开） | 能下出人样、开始堵三连即可收 |
| 消融加餐（选做，见第 11 节） | 用 JSON 覆盖机制改单个超参重训 CartPole | 每档一次 CartPole 预算 | 与默认档同图对比 |

两句预算实话：其一，本课所有必做实验加起来的算力开销低于第 05/06 课任何一次训练，时间大头是 Connect4 后台挂机，人不用守。其二，别在 CartPole 上追求"每次都稳定 500"，README 明说超参未调优、回退现象存在，方向对了就往前走，把完美复现的执念留给第 04 课那种有论文数字可对的场合。

## 9. 验收

验收清单：

- [ ] TensorBoard 里 CartPole 的 `1.Total_reward` 曲线到过 500 附近，截图在证据目录；如出现回退段，同样截图并标注；
- [ ] `mcts_log.md`：三档模拟数的根节点记录齐全，能指着数字说出"先验与访问分布差在哪、树深怎么变、搜索把根价值修正了多少"；
- [ ] `value_equiv_report.md`：训好与随机两组各四个 $R^2$、隐状态方差、命中的解读分支、与第 02 课探针的对照讨论，四样齐全；
- [ ] Connect4 与你对弈至少三盘，记录它是否堵三连；TensorBoard 上 MuZero 对 expert 的分差曲线截图入档；
- [ ] 白纸默画 $h$/$g$/$f$ 加一次模拟的四拍流程，并标出 `initial_inference` 和 `recurrent_inference` 各覆盖哪两段；
- [ ] 口头过一遍 5.2 的辩论：正反各三条论据不看笔记说全，并说出自己的立场；
- [ ] 能回答：为什么 dynamics 必须预测 reward？为什么隐状态没有重建约束也不坍缩？（答案分别在 5.1 和 5.5。）

## 10. 排错

| 症状 | 最可能的原因 | 怎么确认 | 怎么修 |
|---|---|---|---|
| `pip install gym==0.21.0` 报 metadata 相关错误 | 新版 pip 拒绝 gym 0.21 的老包元数据 | 报错信息含 metadata 字样 | 虚拟环境里先降 pip（`pip install "pip<24.1"`）再装；或整体走 Colab |
| 装 `requirements.lock` 时 torch/ray 找不到匹配版本 | Python 太新或 Apple Silicon 无老版安装包 | `python --version` 超过 3.9 | 路线 A 换 Python 3.8/3.9；Apple Silicon 走第 3 节路线 B |
| `python muzero.py` 启动时 ray 报错或卡住 | ray 与 Python 版本不配，或旧 ray 进程残留 | 报错栈里带 ray 字样 | 版本按上一行修；残留进程杀掉重来 |
| Windows 上各种诡异崩溃 | README 明说 Windows 支持是实验性的 | 换 WSL 或 Colab 立好 | 用仓库自带 `notebook.ipynb` 上 Colab |
| 训练跑着但 TensorBoard 一片空白 | logdir 指错或训练还没写出事件文件 | 看 `results/` 下有无本次时间戳目录 | `--logdir ./results` 从仓库根目录起跑；稍等再刷新 |
| `Total_reward` 长期趴在 10 以内不动 | 自博弈或训练一侧卡死 | 看 `2.Workers` 组：两条计数曲线是否都在涨 | 哪条不涨查哪侧终端输出；重启训练 |
| 分数冲上 500 后回落 | README 已知现象：部分环境训练一段时间后性能回退 | 对照 README 的 Known issues | 用回落前的 checkpoint（每 10 步都存了）；接受现状并记录，属正常 |
| `Diagnose model` 没出搜索树 PDF | graphviz 未装（lock 里本来就没有） | 终端有一句安装提示 | Python 包和系统二进制都装上，重跑 |
| 探针脚本 unpickle 报找不到模块 | 没在仓库根目录跑，`self_play` 导入失败 | 报错含模块名 | 回仓库根目录执行，脚本别挪走 |
| 探针 $R^2$ 出现大负数 | RUN_DIR 指错、buffer 数据太少或方差陷阱 | 打印 buffer 局数和各维方差 | 换正确 run；训练久一点再退出；对照 Step 6 第四分支 |
| Connect4 对弈时它下得像随机 | 训练步数太少，或加载了空白模型 | TensorBoard 看它对 expert 的分差；确认走了 Load pretrained model | 再挂几小时；重新加载正确 run |
| `Render some self play games` 在 CartPole 上渲染报错 | gym 0.21 的经典控制渲染依赖不在 lock 里 | 报错提示缺渲染后端 | 本课流程不依赖弹窗渲染，跳过；Connect4 是终端文本渲染，不受影响 |

## 11. 前沿与改造

价值等价这条线 2019 年之后往两头长。一头是样本效率：原版 MuZero 是大规模分布式自博弈喂出来的，数据胃口惊人，EfficientZero（arXiv:2111.00210，NeurIPS 2021）对症下了三味药，给隐状态加自监督一致性损失（让 $g$ 推出来的隐状态去贴 $h$ 编码真实下一帧的结果）、把逐步 reward 预测改成端到端的 value prefix、用模型对陈旧价值目标做离策略修正，在只有约两小时游戏经验的 Atari 100k 基准上第一次把均值分拉过人类水平。注意第一味药的意味深长：一致性损失等于把一只脚伸回了"隐状态要贴住真实观察的表征"的阵营，纯血价值等价在小数据下不够吃，这个张力你读论文时会反复看到。另一头是随机环境：MuZero 的 $g$ 是确定性函数，掷骰子类环境会露馅，Stochastic MuZero（Antonoglou 等，ICLR 2022）给转移加上离散机会变量建模随机分支，在 2048 和西洋双陆棋上补齐了这块。至于本课反方论据里那句"换个奖励函数模型就作废"，前沿的回应思路是把模型学得更任务无关，那正是第 13 课 JEPA 阵营的开场白，这里按下不表。

规模差距（钱能解决的）：DeepMind 版棋类每步 800 次模拟、网络是深残差塔加分布式自博弈，我们 50 到 200 次模拟加笔记本网络。机制差距（本课内容能解决的）：muzero-general 没有一致性损失、没有随机转移建模，超参没调过，这三样每一样都是明白写在纸上的改造方向，不神秘。

动手改造清单（选做，都用仓库自带的 JSON 覆盖机制，不用改文件就能做超参消融）：

1. 模拟数消融。`python muzero.py cartpole '{"num_simulations": 5}'` 与 25、100 各训一次（这个覆盖机制是 `muzero.py` 明文支持的第三个启动方式）。预算：每档一次 CartPole 训练，CPU 小时级。预期：模拟数太少时策略目标退化（访问分布几乎等于先验），学习变慢或不稳。失败判据：三档曲线无可辨差异，那说明 CartPole 太简单撑不起这个消融，换 Connect4 短训再试。
2. n-step 消融。覆盖 `{"td_steps": 5}` 对比默认 50。预期：短 bootstrap 方差小偏差大，冷启动阶段（搜索根价值还是垃圾时）学得更稳还是更慢，你的曲线说了算。预算同上。
3. 挤压隐状态。覆盖 `{"encoding_size": 2}`。8 维装 4 维的观察绰绰有余，2 维就真的要做取舍了，训完重跑 Step 6 探针，看哪个物理量先被扔下车。预期：分数仍能学起来（CartPole 的决策核心信息 2 维装得下），探针里 $x$ 或 $v$ 的 $R^2$ 先塌。这是把价值等价"只留有用的"逼到墙角的实验，预算一次训练加十分钟探针。
4. 把 MuZero 拉回重建阵营（小手术）。给 `models.py` 的全连接网络加一个从隐状态回观察的两层解码头，`trainer.py` 损失里加重建项。预算：改两处各十几行，重训 CartPole 一次。预期：分数不明显变化，探针四个 $R^2$ 整体抬高。失败判据：加了重建后分数明显掉，那本身就是个值得写进报告的发现（重建目标与价值目标打架了）。

MuZero 论文的核心论断"不给规则也能规划"在缩小版上方向可见：Connect4 学会堵三连，就是"规则从奖励信号里自己长出来"的最小证据。EfficientZero 的方向也可以蹭一口：改造 4 反着做（不加重建、只加一致性损失）就是它第一味药的缩小版，动手能力强的可以把两个版本的样本效率画在同一张图上。

## 12. 论文与延伸

1. MuZero: Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model（Schrittwieser 等，[arXiv:1911.08265](https://arxiv.org/abs/1911.08265)，后发表于 Nature），核心参考。带着三个问题读：训练目标的三项里，哪一项都没约束隐状态像真实状态，那附录里为什么还要给隐状态做尺度归一化？搜索伪代码与本仓库 `self_play.py` 有哪些一字不差、哪些被简化了？Reanalyse 在论文里为什么对样本效率那么重要（对照仓库 `use_last_model_value` 那个寒酸的简化版）？
2. AlphaZero: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm（Silver 等，[arXiv:1712.01815](https://arxiv.org/abs/1712.01815)），"有规则模拟器"的前身对照。带着一个问题读：它的 MCTS 每往深走一步靠真棋盘规则推演，把这一步换成学出来的 $g$ 之后，MuZero 的伪代码里还多出了哪些 AlphaZero 不需要的东西？（提示：reward 预测、价值的 MinMax 归一化，棋类只有终局胜负，Atari 步步有分。）
3. EfficientZero: Mastering Atari Games with Limited Data（Ye 等，[arXiv:2111.00210](https://arxiv.org/abs/2111.00210)），样本效率方向的代表作。带着两个问题读：三个改动各治 MuZero 的什么病？自监督一致性损失算不算对价值等价立场的部分撤退？
4. The Value Equivalence Principle for Model-Based Reinforcement Learning（Grimm 等，NeurIPS 2020），把本课口号变成定理的那篇。选读，带着一个问题：论文里"对哪类策略和价值函数等价"是有前提的，MuZero 满足的是哪个弱化版本？
5. Planning in Stochastic Environments with a Learned Model（Antonoglou 等，ICLR 2022，即 Stochastic MuZero），选读。带着一个问题：确定性 $g$ 在随机环境里具体怎么露馅（提示：想想 2048 里随机弹出的方块），机会变量插在 $g$ 的哪个位置？
6. [werner-duvaud/muzero-general](https://github.com/werner-duvaud/muzero-general) 的 README 与 wiki，带着问题读 Known issues：作者列的三条局限（不稳定达到人类水平、训练后期回退、超参未调优），你在本课各见到了哪条？

第 08 课把价值等价的思路搬进连续控制：TD-MPC2 同样不重建观察，但它不做树搜索（连续动作空间里树没法穷举分支），改在潜空间里滚动规划，防坍缩的柱子也从三个头换成 latent 一致性，正好是本课改造清单 4 号实验反过来做的那味药。到时候记得带上今天的探针报告，第 08 课结束你手里会有"重建对免重建"的第一份正面对照证据。
