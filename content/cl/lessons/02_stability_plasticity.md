---
id: 02_stability_plasticity
title: "既要记得住又要学得进"
summary: "把旧的钉死，新的就学不会。这个矛盾在实验里长什么样？"
unit: forget
play_tools: []
checkpoints:
  - "填一张稳定性-可塑性平面。"
  - "能说出海马-新皮层类比在哪里失效。"
  - "给第 05 课的 EWC 留下「为什么需要弹簧」的动机。"
---

# 第 02 课：既要记得住，又要学得进

> 类型：实战（机制实验；EWC 只跑现象，公式与 Fisher 在第 05 课展开）<br>
> 建议周期：2-3 天<br>
> 硬件：CPU / Mac 即可；不需要 GPU<br>
> 锚定仓库：[ContinualAI/avalanche](https://github.com/ContinualAI/avalanche) 的 `EWC` / `EWCPlugin`（文档：[Training 教程](https://avalanche.continualai.org/from-zero-to-hero-tutorial/04_training)）+ 课内四种写法对照<br>
> 产物：稳定性-可塑性平面图、海马-新皮层类比的失效说明、四个方法在同一 Split MNIST 协议上的旧保持 / 新准确率

## 1. 这一课做什么

[第 01 课](01_catastrophic_forgetting.md) 留下了病：naive fine-tune 在 Class-IL 的 Split MNIST 上会把任务 1 的准确率冲下去，热力图下三角变暗。你已经有种子、命令和 $R_{1,T}$。本课不换数据集，换的是「怎么写新经验」。贯穿主干里，这一课动的是第二句：

```text
新经验进来
  先决定写到哪里（本课仍写进同一组慢速权重）
  再决定怎么写（钉死、少更新、混旧数据，或预告里的弹簧）
  写完立刻测两件事：新任务会了没、旧任务还在不在
```

你现在仍在第一幕。第 01 课解决「会不会忘」，本课解决「记住和学会是一对矛盾」。把旧的钉死，新的往往学不会；放开学新的，旧的又没了。文献把这个矛盾叫做稳定性-可塑性困境（stability-plasticity dilemma）：稳定是旧知识还在，可塑是新知识进得去。本课把四个具体写法画在同一张平面上，横轴是旧任务保持，纵轴是新任务准确率。四个点的位置比任何一句口号都有用。

生物系统看起来同时做到了两头。McClelland、McNaughton 和 O'Reilly 1995 年的互补学习系统（Complementary Learning Systems, CLS）给出一个双系统故事：海马快速记下情节，新皮层慢慢抽取结构，睡眠里的再激活把新记忆交错写进新皮层。这是本课唯一展开生物类比的地方。类比必须立刻写出失效处：你的 MLP 没有睡眠、没有分离的编码器、也没有两条时间尺度。混一点旧数据只是廉价的交错，不是海马。冻骨干只是把可塑性关到接近零，不是新皮层的慢学习。EWC 的弹簧更接近「保护重要突触」，正式推导在第 05 课；本课只跑现象，看它落在平面的哪一侧。

四种写法（同一份 Split MNIST、同一种子）：

1. naive：全网继续训，学习率与第 01 课相同。
2. 冻骨干：前几层 `requires_grad=False`，只训分类头。
3. 降学习率：全网可训，学习率缩小 10 倍。
4. 混旧数据：每个 batch 里留一小比例任务 1 的样本。

Avalanche 上再加第五个点作预告：`EWC` 策略，默认或文档示例里的 $\lambda$，不扫参。课程正式讲 Fisher、扫 $\lambda$、和 SI / LwF 对照，全部留给第 05 课。本课若把 EWC 写成「已经学会」，后面那课就没有动机。

本课三档划分：

1. **实战**：四种写法的对照表 + Avalanche `Naive` vs `EWC` 各跑一条 Split MNIST。
2. **机制实验**：课内 CPU 实验断言「冻骨干比 naive 更稳、更塑不了」；浏览器里先猜四个点再揭晓。
3. **只讲**：完整的 CLS 神经解剖、睡眠纺锤波、小鼠树突棘实验。课文引用结论，不要求你做生物实验。

术语速查（第 01 课已出现的不再重复，见 [第 01 课](01_catastrophic_forgetting.md) 术语表）：

| 术语 | 一句人话 |
|---|---|
| 稳定性（stability） | 学新的时候，旧任务准确率还能留住多少 |
| 可塑性（plasticity） | 对新任务还学不学得进；本课用新任务准确率当操作定义 |
| 稳定性-可塑性平面 | 横轴旧保持、纵轴新准确率；好方法往右上走，naive 常在左上 |
| 冻骨干 | 锁住提取特征的层，只让最后的分类头动 |
| 互补学习系统（CLS） | 海马快记情节、新皮层慢抽结构，再激活把新记忆交错写入 |
| 交错 / interleaving | 新旧样本在时间上插着出现，避免同一方向的梯度连打许多步 |
| EWC（弹性权重巩固） | 给对旧任务重要的权重加弹簧，更新时少动它们；本课只看现象 |
| $\lambda$（EWC） | 弹簧总劲度：0 接近 naive，过大则新任务学不动 |
| 保持率 | 旧任务在新任务训完后的准确率，常和刚学完时的 $R_{i,i}$ 一起报 |

## 2. 问题

第 01 课的 naive 说明：共享权重上的顺序 SGD 会覆盖旧边界。一个本能反应是「那就少动权重」。少动有三种廉价实现，本课全做：把学习率乘 0.1；把骨干冻住；在新 batch 里混回几张旧图。它们都会改变平面上的位置，但方式不同。核心问题不是「哪一个永远最好」，而是：

1. 钉死、减速、混旧数据，分别把点往平面的哪一角推？有没有一个开关能同时到达右上角？
2. 生物双系统故事在什么意义上解释了这件事，又在哪几处根本套不到你的 MLP 上？
3. EWC 声称「重要的权重要慢一点」，本课用 Avalanche 最小例子看它是更像冻骨干，还是更像混旧数据。不在本课证明 Fisher 为什么能当重要性。

一个要先划清的界限：冻骨干在多任务、类增量里经常让新头没有可用特征，新任务准确率上不去，这会被误读成「稳定成功」。平面强制你同时看两轴。只报旧任务保持的人，会把一个学不会新类的死模型当成抗遗忘冠军。第 03 课的平均准确率会把另一种死模型（只会最后一件事）抬起来。两课打的是对称的假。

另一个界限：混旧数据已经踏进第 06 课回放的门口。本课只混「任务 1 的一小撮、且任务数是 2」，用来证明交错有效。不要把 5% 回放写成 DER++，也不要在这里扫缓冲大小。

## 3. 准备

- 第 01 课的笔记：种子、Split MNIST 切分、naive 的 $R_{1,1}$ 与 $R_{1,T}$。本课四种写法必须落在同一协议上，否则四个点不共面。
- 同一套 CPU 环境、同一个 `avalanche-lib` 版本。若你第 01 课钉的是 `0.6`，本课不要悄悄升级。
- 仍不需要 GPU。四种写法各训两个短任务，时间按小时的小数计。
- 先做网页上的稳定性-可塑性平面（先猜点的位置，再揭晓），再写 Python。猜错是预期的一部分，把初始猜测抄下来。
- 读 McClelland 1995 只需抓住双系统分工和「快速写入分布式网络会灾难性干扰」这一句。不要在本课尝试复述海马分区解剖。

## 4. 学习目标

1. 在空白纸上画出稳定性-可塑性平面，标出 naive、冻骨干、小学习率、混旧数据四个点的**预期象限**，再用实验数字修正。
2. 写出稳定性、可塑性在本课的操作定义（旧任务保持、新任务准确率），并说明为什么不能只报其中一个。
3. 用 CLS 的两句话讲清海马 / 新皮层分工，紧接着列出至少三条类比失效（无睡眠、无分离编码器、无两条时间尺度）。
4. 独立完成三层实验：浏览器先猜后揭晓、`python3 run.py run 02`、四种写法加 Avalanche EWC 预告。
5. 解释 EWC 的损失里那根弹簧在干什么，并明确：Fisher 怎么算、$\lambda$ 怎么扫，不属于本课验收。
6. 判断「把学习率调小」为什么通常两头都弱，不能当成免费的持续学习算法。

## 5. 原理

五个机制，每个仍走直觉、机制、数学、代码、验证。生物类比只在 5.3 和 5.4 展开。

### 5.1 一张平面，两个必须同时成立的数字

直觉。只盯旧任务，最稳的办法是把学习率调到 0：模型冻成照片，任务 1 百分之百还在，任务 2 零分。只盯新任务，最塑的办法是 naive：任务 2 很高，任务 1 被擦掉。持续学习的最低合格线是两个数字都过门槛。把它们看成平面上的一个点，争论从「谁的平均分高」变成「这个点在哪、往哪移」。

机制。记 $S$ 为旧任务在新任务训完后的准确率（稳定性的操作定义），$P$ 为新任务准确率（可塑性的操作定义）。本课任务数为 2 时：

$$
S = R_{1,2},\qquad P = R_{2,2}
$$

也可以报相对保持 $S_{\mathrm{rel}}=R_{1,2}/R_{1,1}$，避免「任务 1 本来就没学会」被记成稳定。本课主图用绝对 $S$，因为 0 对 1 对两层 MLP 几乎总能学会，$R_{1,1}$ 接近 1，两种定义差不多。若你的 $R_{1,1}$ 低于 0.8，先回去把任务 1 训够，再谈平面。

任务更多时，$S$ 可改成旧任务的平均保持，$P$ 改成当前新任务准确率；第 03 课再换成 Average Accuracy 和 Forgetting。本课故意用两个标量，让平面可画。

四个方法是四个不同的更新规则，点的移动方向可以事先猜：

| 方法 | 对 $S$ 的直觉 | 对 $P$ 的直觉 | 预期落点 |
|---|---|---|---|
| naive | 低 | 高 | 左上 |
| 冻骨干 | 高 | 低或中 | 右下或右中 |
| 小学习率 | 中低 | 中低 | 靠近中间偏下 |
| 混旧数据 | 中高 | 中高 | 比 naive 更靠右，比冻骨干更靠上 |

这些是预测，不是定理。特征若碰巧对 2、3 也好用，冻骨干的 $P$ 会高于你的猜测；旧数据太少，混数据的点会靠近 naive。实验的价值是修正这张表。

数学。理想点是 $(S,P)=(1,1)$。任何只优化 $P$ 的规则都可以把点送到左上；任何只约束 $\|\theta-\theta_1^\star\|$ 的规则都可以把点送到右下。持续学习方法的技术含量，体现在约束的**选择性**：哪些方向能走，哪些方向被刹住。小学习率没有选择性，每个坐标一视同仁地慢。冻骨干按层选择：浅层全刹、头全放。混数据通过梯度平均来选择：$\nabla \mathcal{L}_2 + \beta \nabla \mathcal{L}_1$。EWC 按参数重要性选择，第 05 课展开。

代码。四个方法必须共用同一套 `R` 计算函数。差别只能出现在「任务 2 怎么 `optimizer.step()`」。若冻骨干时你连评测口径都改成了 Task-IL，平面上的位移就不是方法造成的。

验证。naive 的点应明显在冻骨干左侧（$S$ 更小），并在冻骨干上方或至少不更低（$P$ 更大）。若四个点重叠，多半是任务 2 太像任务 1，或训练步数为 0。

一组只用来校准读图的虚构数字（不要抄进你的笔记当实验结果）：

| 方法 | $S$ | $P$ | 读法 |
|---|---|---|---|
| naive | 0.18 | 0.97 | 新的会了，旧的没了 |
| 冻骨干 | 0.91 | 0.42 | 旧的还在，新的半会不会 |
| 小学习率 | 0.45 | 0.58 | 两头都没打满 |
| 混旧数据 | 0.72 | 0.90 | 交错把点往右上推了一截 |
| 联合训练上界 | 0.96 | 0.97 | 两个任务的数据从头混在一起，不是持续学习 |

联合训练（joint / offline）把 $\mathcal{D}_1$ 和 $\mathcal{D}_2$ 同时给模型，Avalanche 的 `examples/joint_training.py` 就是这条上界。它不进入四点对照，因为它违反「训练任务 2 时不能把任务 1 整库拿回来」这条持续学习约束。混数据用 $\beta=0.1$ 只是上界的一个极瘦影子。若你的混数据点已经贴上联合训练，说明 $\beta$ 太大或任务太容易，把 $\beta$ 降下来再画。

读图时禁止把距离原点的远近当成总分。左上和右下离原点都可以很远，一个是会学不会记，一个是会记不会学。本课合格的讨论必须同时点名 $S$ 和 $P$。

### 5.2 四种写法各自动了哪一个旋钮

直觉。把网络看成「眼睛 + 笔」。眼睛是骨干，笔是头。naive 眼睛和笔一起改；冻骨干只改笔；小学习率两者都改但每步更短；混旧数据是改的时候还拿旧作业对照。旋钮不同，副作用不同。

机制。

**冻骨干。** 任务 1 训完后，对除最后一层以外的参数设 `requires_grad=False`，优化器只包含头。任务 2 的梯度到不了特征。旧决策边界在特征空间里几乎不动，所以 $S$ 高。新类若需要新的特征方向（2 的环、3 的开口），头在冻结特征上可能线性不可分，$P$ 就上不去。这是稳定性换可塑性。Class-IL 里头还要为新类长出 logit；若你冻住的是整网含头，那就变成学习率 0，点会掉到右下角的底。

**小学习率。** $\eta'=\eta/10$。每一步对旧边界的破坏变小，对新技术的吸收也变小。两头都弱是默认结果，不是调参失败。只有新任务极容易、旧任务极稳时，它才会看起来「也行」。本课的 0/1 接到 2/3 足够用来展示两头都弱。

**混旧数据。** 任务 2 的每个 batch 抽 $(1-\beta)$ 的新样本和 $\beta$ 的任务 1 样本，$\beta$ 取 0.1 或 0.2。这是最浅的 experience replay。梯度变成新旧损失的凸组合，旧边界每步都被轻轻拉回去。$S$ 通常高于 naive；$P$ 是否掉取决于 $\beta$ 和任务难度。它已经在用旧数据，严格说越出了「无旧数据持续学习」。本课允许，是为了让平面上出现一个靠右上的点，证明交错在数学上对症。第 06 课才讨论缓冲满了怎么办。

**naive。** 对照原点。没有它，你不知道其他三个点移动了多少。

数学。冻骨干是硬约束：$\theta_{\text{backbone}}=\theta_{\text{backbone}}^{(1)}$。小学习率是把 SGD 的步长改成 $\eta/10$，可行域仍是全空间。混数据的一步期望更新正比于

$$
(1-\beta)\nabla \mathcal{L}_2(\theta)+\beta\nabla \mathcal{L}_1(\theta)
$$

若 $\nabla \mathcal{L}_1^\top \nabla \mathcal{L}_2<0$，混数据会缩短沿冲突方向的步长。这和 EWC 的二次惩罚不是一回事：一个改数据项，一个改参数先验。

代码。冻骨干时必须重建 optimizer，否则 Adam / SGD 的状态里还留着已冻参数的动量。混数据时两个 DataLoader 要独立打乱，禁止把任务 1 的测试集拿去混。Avalanche 里，混数据对应 `ReplayPlugin(mem_size=...)`，冻骨干没有官方一键策略，课内自己写 `p.requires_grad = False`。

验证。打印 `sum(p.numel() for p in model.parameters() if p.requires_grad)`：冻骨干后应变小。混数据时打印每个 batch 里任务 1 标签的比例，应接近 $\beta$。小学习率组的 loss 下降应明显慢于 naive。

### 5.3 互补学习系统：为什么生物不像 naive

直觉。人学新同事的名字，不会把乘法表忘成空白。McCloskey 1989 已经展示机器会。CLS 的回答是：人并不把新情节立刻以大步长写入那个负责语义和技能的分布式网络。新情节先进入一个专门快速编码、模式分离的系统（海马），再在休息和睡眠中以小步、交错的方式重放到新皮层。新皮层每次只改一点点，许多记忆插着重放，灾难性覆盖就不会发生。

机制。McClelland 等 1995 年的论文标题把问题说完整：为什么海马和新皮层要有互补的学习系统，洞察来自联结主义模型的成功和失败。失败正是第 01 课那种：在重叠表征的网络里快速嵌入新任意联想，会灾难性干扰。成功是：同一类网络若把新项目和旧项目交错训练，可以慢慢抽出共享结构。于是系统被劈成两个时间尺度：

- 海马：稀疏、模式分离，允许快速记下新情节，彼此少重叠，干扰小。
- 新皮层：分布式、重叠，适合抽取统计结构，但必须慢，而且需要交错。

海马损伤的经典模式（新近记忆没了、远期记忆还在）在这个故事里对应：远期记忆已经写进新皮层，新近的还在海马、还没巩固。Kirkpatrick 2017 引用的小鼠树突棘实验走的是另一条神经故事：新皮层突触本身也可以通过降低可塑性来保护旧技能。那是 EWC 的生物动机，和 CLS 的「两个系统 + 重放」不是同一条机制。本课必须把两条故事分开，否则你会以为 EWC 就是海马。

数学。CLS 没有给你一个可直接贴进 PyTorch 的公式。它给出的是训练协议约束：若表征重叠且学习率大且无交错，则 $S$ 崩。交错版本近似于对联合分布做 SGD：

$$
\theta \leftarrow \theta - \eta \nabla_\theta \big( \mathbb{E}_{\mathcal{D}_1}\ell + \mathbb{E}_{\mathcal{D}_2}\ell \big)
$$

这就是离线多任务上界，也是第 01 课 Avalanche 文档里 `joint training` 例子在做的事。海马重放是在不能把 $\mathcal{D}_1$ 整库留下时，用一个生成/检索系统近似 $\mathbb{E}_{\mathcal{D}_1}$。你的 $\beta=0.1$ 混数据是这个期望的极小蒙特卡洛。

代码。本课没有海马模块。混数据函数就是 CLS 在玩具上的可运行投影。不要把 `EWCPlugin` 注释成海马：它既不存储情节，也不在离线相位重放。

验证。把 $\beta$ 从 0 调到 0.5，点应向右上或至少向右移动。这是「交错有效」的最小证据。它**不能**证明你实现了 CLS。笔记里要写这一句。

### 5.4 类比在哪里失效

风格规定：类比必须服务于机制，用完紧跟精确定义，并指出失效处。下面五条写进交付物，缺一条验收不算过。

1. **没有分离的编码器。** CLS 的海马用稀疏、模式分离的编码降低重叠。你的 MLP 从像素到隐藏层是稠密 ReLU，任务 1 和任务 2 共用几乎全部单元。冻骨干没有创造模式分离，它只是停止更新。
2. **没有睡眠和离线巩固相位。** 生物重放发生在与环境解耦的一段时间。你的训练循环是「下一个 batch 立刻来」。混数据发生在在线步里，和睡眠巩固的时间结构不同。
3. **没有两条时间尺度。** CLS 要求海马快、新皮层慢。你要么所有参数一个学习率，要么人工把一层的学习率设成 0。EWC 给不同权重不同有效学习率，这是「参数级时间尺度」，仍然是一个网络，不是两个系统。
4. **没有情节存储器。** 海马能按事件取回。$\beta$ 混数据是从任务 1 的张量子集均匀抽样，没有事件边界，也没有「今天过的哪几件事更该重放」。
5. **巩固方向被写反的风险。** EWC 保护的是已经学过的权重，对应突触巩固，更像「让新皮层里已有的技能变稳」。海马的工作是先把新事件快速存下来。把 EWC 说成「给网络加了一个海马」，会在第 05 课和第 13 课（外挂记忆）同时误导你。

精确定义（类比之后必须留下的那句）：本课的对象是单网络、单优化器、可选一小撮旧样本的顺序监督训练。稳定性是 $R_{1,2}$，可塑性是 $R_{2,2}$。CLS 是关于这个问题的一个神经理论，不是本课实现的架构。

验证。交付物里的失效说明若只写「生物更复杂」，不合格。必须点名睡眠、分离编码器、时间尺度三条中的至少三条，并用你的代码指认「哪一行对应的是缺失」。

### 5.5 EWC 预告：按重要性给权重加弹簧

直觉。冻骨干是对一层一刀切。小学习率是对所有权重一刀切。EWC 想做的是：对旧任务敏感的权重少动，不敏感的放开学。Kirkpatrick 等人把这根「按重要性变化的拉力」写成二次惩罚，名字叫弹性权重巩固：参数被弹簧拉在旧解附近，弹簧劲度不是常数。

机制。任务 A 学完得到 $\theta_A^\star$。学任务 B 时，除了 $\mathcal{L}_B$，再加一项把 $\theta$ 拉向 $\theta_A^\star$。拉得有多紧，由每个参数的 Fisher 信息对角线 $F_i$ 决定：$F_i$ 大表示旧任务对这个参数敏感。总损失（论文公式 (3)）是

$$
\mathcal{L}(\theta)=\mathcal{L}_B(\theta)+\sum_i \frac{\lambda}{2} F_i (\theta_i-\theta_{A,i}^\star)^2
$$

$\lambda=0$ 退回 naive。$\lambda$ 极大且 $F_i$ 全为正，行为接近把所有敏感参数钉死，平面上会像冻骨干。选择性来自 $F_i$ 的差异：不重要的参数弹簧很软，还能给新任务用。Fisher 如何从梯度算出来、对角线近似坑在哪，第 05 课用直方图和 $\lambda$ 扫描讲。本课只需要：这是第三种「少动」的办法，选择性介于冻骨干和小学习率之间。

Avalanche 把这件事做成策略和插件两套入口。训练教程的导入列表包含 `EWC`；同一页用 `EWCPlugin(ewc_lambda=0.001)` 和 `ReplayPlugin` 一起挂到 `SupervisedTemplate` 上。本课跑独立的 `EWC` 策略对比 `Naive`，不要一上来做混合体，否则平面上的点说不清是弹簧还是回放。

数学。二次项的梯度是 $\lambda F_i (\theta_i-\theta_{A,i}^\star)$，加到 $\nabla \mathcal{L}_B$ 上。它不插入旧样本，所以 EWC 被归为正则方法，可以在「不能存旧数据」的设定里用。van de Ven 2019 的警告仍然有效：Class-IL 上正则方法常常不够。本课若看见 EWC 的 $S$ 高于 naive、但 $P$ 和 $S$ 都远谈不上解决 Class-IL，这是符合文献方向的现象，不是你实现错了。

代码。定位安装包中的类：

```bash
python3 -c "from avalanche.training.supervised import EWC; import inspect; print(inspect.getfile(EWC))"
```

```bash
python3 -c "from avalanche.training.plugins import EWCPlugin; import inspect; print(inspect.getfile(EWCPlugin))"
```

读构造函数签名，记下 `ewc_lambda` 参数名。不要抄网上过时的 `lambda` 作为关键字，以你安装版本的签名为准。

验证。同一协议下，EWC 的 $S$ 应不低于 naive（允许误差），$P$ 允许略低。若 $S$ 反而更差且 $P$ 也更差，先查 $\lambda$ 是否大到数值不稳，或 Fisher 在过短的任务 1 上接近 0（弹簧没挂上，等于 naive 加噪声）。第 05 课会专门做 $\lambda=0$ 应接近 naive 的断言。

## 6. 源码导读

本课读四块：第 01 课已经定位过的 `Naive` / `SplitMNIST` / `SimpleMLP`；冻骨干与混数据的课内写法；Avalanche 的 `EWC` 与 `EWCPlugin`；CLS 论文本身（无官方代码）。

| 对象 | 从哪导入或打开 | 带着什么问题读 |
|---|---|---|
| `Naive` | `avalanche.training.supervised` | 和第 01 课是否同一骨架？本课四个点必须以它为原点 |
| `EWC` | `avalanche.training.supervised` | `ewc_lambda` 的默认值？`mode` 若存在，`separate` 和 `online` 差在能否随任务数线性涨内存？ |
| `EWCPlugin` | `avalanche.training.plugins` | 它挂在循环的哪一个 `before/after` 上？和 `ReplayPlugin` 能否同时挂（教程说可以，本课不要同时挂）？ |
| `ReplayPlugin` | `avalanche.training.plugins` | `mem_size` 是样本数还是每类样本数？本课手写混数据要和它区分：手写是固定 $\beta$，插件是缓冲 |
| `SimpleMLP` | `avalanche.models` | 哪一段算骨干、哪一段算头？冻骨干要切在哪一层？ |
| `SupervisedTemplate` | `avalanche.training.templates` | 自定义冻骨干能不能当插件写，还是任务 2 开始前改 `requires_grad` 更直接？ |

训练教程里这段混合例子只读不跑：

```python
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.plugins import ReplayPlugin, EWCPlugin

replay = ReplayPlugin(mem_size=100)
ewc = EWCPlugin(ewc_lambda=0.001)
strategy = SupervisedTemplate(
    model, optimizer, criterion,
    plugins=[replay, ewc])
```

它证明框架把「少动」和「混旧数据」当成可组合插件。本课若跑这一段，平面上多出来的点无法归因。把它留给第 05、06 课做消融。

课内机制实验：`experiments/src/learn_cl_experiments/lessons/lesson_02.py`。`python3 run.py run 02` 钉死「冻骨干比 naive 更稳、更塑不了」，`checks` 见第 7 节。

CLS 没有参考实现。你能对照的是自己的混数据函数：它近似的是论文所说的 interleaving，不是海马编码。读论文时把「sparse / pattern-separated」标出来，回到你的 `nn.Linear` 看有没有对应物。没有，就是 5.4 的第 1 条。

## 7. 实验

同一协议贯穿全节：任务 1 = MNIST 的 0 和 1，任务 2 = 2 和 3，Class-IL，`seed=42`，两层 MLP 或 `SimpleMLP`。四个课内写法用同一模型定义；Avalanche EWC 用 `SplitMNIST` 五段也可以，但和四点平面分开报，不要把五段平均硬塞进两任务平面。

### Step 0: 浏览器里先猜四个点

本课 lab id 是 `lab-02-stability-plane`，页面锚点仍是 `#interactive-lab`。平面横轴为旧任务保持，纵轴为新任务准确率。四个方法是四个点。运行前必须提交预测：naive、冻骨干、小学习率、混旧数据各在哪一象限（左上 / 右上 / 左下 / 右下，或「靠近中心」）。

先猜再揭晓。揭晓用的是浏览器里的小仿真（二维高斯或线性可分玩具），不是你的 MNIST 数字。两边方向应一致：naive 偏左上，冻骨干偏右下，混数据相对 naive 右移。若玩具和 MNIST 方向相反，笔记里写「设定差异」，不要改 MNIST 数字去迁就玩具。

过关：预测至少对 naive 和冻骨干的相对位置（谁更稳、谁更塑）。小学习率和混数据允许猜错，但必须在揭晓后用 5.2 的机制解释误差。

### Step 1: 课内 CPU 机制实验

```bash
cd experiments
```

```bash
python3 run.py run 02
```

预期：命令打印 `[PASS]`，写出 `artifacts/lesson02/result.json`，五个 `checks` 全真。这层钉的是机制，不是论文分数：不是 EWC 正式复现，第 05 课才是。本机一次运行（Python 3.13.13，seed=2），冻骨干 A 0.996 / B 0.546，naive A 0.718 / B 1.000，冻骨干位移 0。换机器会变，方向不应变。

真实 `checks` 键名：

- `task1_learned_above_0_90`：任务 A 先学到 >0.90（本机 `acc_task1_after_task1`=0.996429）；
- `freeze_more_stable_than_naive`：冻骨干旧任务比 naive 高 0.05 以上（本机 0.996429 对 0.717857）；
- `freeze_less_plastic_than_naive`：冻骨干新任务比 naive 低 0.10 以上（本机 0.546429 对 1.0）；
- `frozen_backbone_does_not_move`：冻骨干的骨干位移为 0（本机 `backbone_l2_freeze`=0.0）；
- `naive_backbone_moves`：naive 的骨干确实动了（本机 `backbone_l2_naive`=3.276874）。

### Step 2: 四种写法对照

沿用第 01 课 Step 2 的数据切分和 `acc()`。任务 1 都用相同的 2 个 epoch 训完，存 `state_dict`，再分四路做任务 2。任务 2 也固定 2 个 epoch，只有更新规则不同。

```python
import copy
import torch

def clone_from(src):
    m = copy.deepcopy(src)
    return m

def train_task2(model, loader, lr, freeze_backbone=False, replay_loader=None, beta=0.0):
    if freeze_backbone:
        for name, p in model.named_parameters():
            if "0" in name or "1" in name:
                p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr)
    replay_iter = iter(replay_loader) if replay_loader is not None else None
    for x, y in loader:
        if replay_iter is not None:
            try:
                xr, yr = next(replay_iter)
            except StopIteration:
                replay_iter = iter(replay_loader)
                xr, yr = next(replay_iter)
            x = torch.cat([x, xr], dim=0)
            y = torch.cat([y, yr], dim=0)
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    return model
```

上面的混数据写法是「每个新 batch 再拼一个旧 batch」，$\beta$ 随两个 batch 大小变化。更干净的做法是按 $\beta$ 从两个 loader 按比例切片。验收看的是旧标签确实出现在任务 2 的训练里，不是 $\beta$ 精确到小数点后三位。

四路：

1. naive：`lr=0.01`，不冻，无回放。
2. 冻骨干：`lr=0.01`，冻 `Sequential` 里第 0 层 `Linear` 和第 1 层 `ReLU` 对应的参数（即第一层权重），只训最后 `Linear`。
3. 小学习率：`lr=0.001`，不冻，无回放。
4. 混旧数据：`lr=0.01`，`replay_loader=train_t1`。

每路报告 $(S,P)=(R_{1,2}, R_{2,2})$。把四个点画在同一坐标轴，刻度 0 到 1。原点附近的点意味着两头都没学会；右上是本课没有免费到达的区域。

预期方向（允许幅度因初始化而变，不允许方向整体反转）：

- naive：$P$ 最高档之一，$S$ 最低档之一；
- 冻骨干：$S$ 高于 naive，$P$ 低于 naive；
- 小学习率：$S$ 和 $P$ 都不会是四个里最好，常见是双中等或双偏低；
- 混旧数据：$S$ 高于 naive， $P$ 接近 naive。

若冻骨干的 $P$ 反而最高，检查是不是冻失败（`requires_grad` 全还是 True）或任务 2 其实只用了线性可分的残差。若混数据的 $S$ 没有高于 naive，打印 batch 内标签直方图。

### Step 3: Avalanche Naive vs EWC（预告，不扫 lambda）

继续用第 01 课的 `SplitMNIST` 脚本骨架。复制一份，把策略换成 `EWC`。`ewc_lambda` 先用构造函数默认值；若默认值让训练数值爆炸，再改成 `0.4`（许多 Avalanche 版本的常见默认）或 `1.0`，并在笔记写明你用的数。不要扫 8 个 $\lambda$，那是第 05 课。

```python
from avalanche.training.supervised import EWC

cl_strategy = EWC(
    model,
    SGD(model.parameters(), lr=0.001, momentum=0.9),
    CrossEntropyLoss(),
    ewc_lambda=0.4,
    train_mb_size=500,
    train_epochs=1,
    eval_mb_size=100,
    evaluator=eval_plugin,
)
```

```bash
python3 naive_split_mnist.py
```

EWC 脚本用另一文件名，例如 `ewc_split_mnist.py`，同样 `seed=42`。

```bash
python3 ewc_split_mnist.py
```

预期：EWC 在早期经验上的保持高于或至少不低于 Naive，最后经验的准确率可能略低。把两个策略的「任务 1 经验准确率」抄下来，作为平面上的第五个点，标注「预告，未扫 $\lambda$」。若 EWC 与 Naive 完全重合，读日志确认任务切换时是否计算了重要性；过短的 1 epoch 可能导致 Fisher 很小，弹簧等于没挂。此时把 `train_epochs` 加到 2 再比一次，仍重合就记下，留给第 05 课。

### Step 4: 把点填进平面并写失效说明

纸或 `NOTES.md` 里画：

```text
P (new task acc)
1 |
  |     mix?
  | naive
  |           EWC?
  |    small lr
  |                    frozen
0 +--------------------------- S
  0                           1
```

把实测数字写在点旁，不要只画示意图。然后用五句话写 CLS 失效，对应 5.4 的五条，每条点一个代码事实。例如：「冻骨干那路 `requires_grad=False` 作用在同一套 `Linear` 上，没有第二套稀疏编码器。」

失效说明的最低完整度如下，可以改写，不能删条件：

```text
1 编码器：只有一份 Linear-ReLU，没有稀疏码
2 睡眠：没有离线相位，batch 之间不休息
3 时间尺度：除冻结构外，全体参数共用一个 lr
4 情节存储：混数据是均匀抽旧张量，不是按事件取回
5 EWC 不是海马：它不写情节，只拉参数
```

若你愿意多做一条对照：把任务 1 和任务 2 的训练集按 1:1 拼起来从头训一个模型，得到联合训练点。它应接近右上。把这个点画成空心圆，注明「上界，不是本课方法」。空心圆的作用是防止你把混数据的小幅右移夸成「已经解决持续学习」。


### Step 5: 对照第 01 课的 naive 数字

第 01 课的 $R_{1,2}$ 必须能在误差内复现为本课 naive 点的 $S$。若差很多，先查协议有没有改（输出维、epoch、是否 Task-IL）。平面上的位移只在对照成立时有意义。

## 8. 配置与预算

| 档位 | 内容 | 时间（CPU，参考） | 用途 |
|---|---|---|---|
| 浏览器 | 四方法点预测 | 10 分钟 | 先猜相对位置 |
| 课内机制 | `python3 run.py run 02` | 通常 < 30 秒 | 断言冻骨干更稳、更塑不了 |
| 四点对照 | 同一 MLP，四路任务 2 | 十几分钟 | 交付平面的四个实测点 |
| Avalanche 预告 | Naive 与 EWC 各一条 Split MNIST | 各数分钟到十几分钟 | 看弹簧是否把点右移 |
| 加分 | `train_epochs=2` 重跑 EWC | 再加一倍时间 | 排除 Fisher 没挂上 |

$\beta$ 建议 0.1 或 0.2，学习率组用 `0.01` vs `0.001`，不要额外网格搜索。EWC 的 $\lambda$ 只许改一次（默认不行时改成 `0.4`），改了必须记。磁盘和内存与第 01 课相同。

## 9. 验收

- [ ] 浏览器实验：运行前提交过四个点的象限预测；能解释揭晓后和预测的差。
- [ ] `python3 run.py run 02`：`checks` 全真（`task1_learned_above_0_90`、`freeze_more_stable_than_naive`、`freeze_less_plastic_than_naive`、`frozen_backbone_does_not_move`、`naive_backbone_moves`）。
- [ ] 四点表：naive、冻骨干、小学习率、混旧数据都有 $(S,P)$，种子 42，Class-IL。
- [ ] 平面图：两轴有刻度，点有标签，naive 的 $S$ 能对上第 01 课。
- [ ] 方向：冻骨干 $S$ 高于 naive，且 $P$ 低于 naive（与 brief 一致）。
- [ ] 失效说明五条中至少三条，每条指向你代码里缺的东西。
- [ ] EWC 预告：抄了 `ewc_lambda`、任务 1 保持、最后任务准确率；明确写「未扫 $\lambda$，正式分析见第 05 课」。
- [ ] 口头：为什么「学习率乘 0.1」通常不是解决方案。

## 10. 排错

| 症状 | 最可能的原因 | 怎么确认 | 怎么修 |
|---|---|---|---|
| `freeze_more_stable_than_naive` 或 `freeze_less_plastic_than_naive` 为假 | 冻失败，或两路用了不同评测 | 看 `freeze_acc_task1` / `naive_acc_task1` 与对应的任务 2 | 对照 `summary`：旧任务应高 0.05 以上、新任务应低 0.10 以上 |
| `frozen_backbone_does_not_move` 为假 | 冻骨干仍更新了骨干 | 看 `backbone_l2_freeze` | 该位移必须是 0 |
| 冻骨干和 naive 数字几乎一样 | 没有重建 optimizer，或冻错了层 | 打印 `requires_grad` | 冻完再 `SGD(filter(...))`；确认第一层 `weight.requires_grad is False` |
| 冻骨干 $P$ 接近随机，$S$ 也掉 | 把头也冻了，或 Class-IL 头维度不够 | 看最后一层是否可训、输出是否 4 维 | 只冻特征层；输出含任务 2 的类 |
| 混数据不提升 $S$ | 旧 loader 空，或拼错标签 | `y.unique()` | 确认 `replay_loader` 来自任务 1 训练集 |
| 小学习率 $P$ 反而最高 | 任务 2 过短，naive 已经过冲 | 看 naive 的训练 loss 是否先降后升 | 两边用同样 epoch；必要时 naive 降到 1 epoch 再比，但要记协议变更 |
| EWC 与 Naive 重合 | $\lambda$ 过小或 Fisher 近 0 | 读 `EWC` 源码里重要性是否在 `after_training_exp` 更新 | 加 epoch 或略增 $\lambda$ 一次；仍重合则记下留给第 05 课 |
| EWC 两个任务都崩 | $\lambda$ 过大，数值不稳 | loss 是否出现极大值 | 把 $\lambda$ 降回 `0.4` 或默认 |
| 四点不共面 | 某路改了评测口径 | 是否对冻骨干用了 Task-IL 切片 | 四路共用同一个 `acc()` |
| 与第 01 课 naive 对不上 | 种子、切分、epoch 变了 | 对照两份 `NOTES.md` | 先复现第 01 课单点，再分叉四路 |
| `EWC` 导入失败 | 版本过旧或装错包名 | `pip show avalanche-lib` | 与第 01 课同一环境；导入路径以 Training 教程为准 |
| 浏览器点和 MNIST 点左右相反 | 玩具任务冲突方式不同 | 看二维高斯是否重叠 | 以 MNIST 为交付数字；玩具只负责相对概念 |

## 11. 前沿与改造

稳定性-可塑性是整门课反复出现的坐标系。第 05 课扫 $\lambda$，你会看到 EWC 在这张平面上走出一条从左上到右下的轨迹。第 06 课扫缓冲大小，混数据的点会继续向右上走，直到缓冲变成「几乎多任务」。第 07 课扩结构是另一条路：不在旧权重上做选择，直接给新任务新格子。第 15 课会告诉你还有第三轴：学得动学不动（可塑性丢失），那时 $P$ 会随任务序号自己往下掉，即使没有旧任务考试。第 20 课的 RL's Razor 把「离原模型远近」和遗忘联系起来，可以看成这张平面在策略空间里的版本。§12 的主阅读从 Dohare 等 2024 的 ImageNet 长序列开始：可塑性丢失是学着学着 $P$ 自己往下掉。本课冻骨干是你把更新关掉，两件事分开记。

CLS 在 2016 年有一篇更新（Kumaran, Hassabis, McClelland, *Trends in Cognitive Sciences*）：智能体需要什么样的学习系统。大模型时代的外挂记忆（第 13 课）更像把情节写到网络外面，测试时学习（第 17 课）则把快权重塞进架构。它们都在回答本课这张平面，只是「写到哪里」换了。本课的失效清单仍然适用：没有睡眠的单循环训练，补丁都是在近似交错或近似巩固。

动手改造（01-12 课可精简，下列三条够用）：

1. **$\beta$ 三段。** 混数据取 $\beta\in\{0, 0.1, 0.5\}$。预算：三倍任务 2 训练。预期：$\beta=0$ 即 naive；$0.5$ 的 $S$ 最高，$P$ 可能略降。失败判据：$0.5$ 的 $S$ 仍等于 0，说明旧样本没进计算图。
2. **只冻第一层 vs 冻到倒数第二层。** 预算：两路重训任务 2。预期：冻得越深，$S$ 越高、$P$ 越低，点沿右下移动。失败判据：两路重合，仍是冻失败。
3. **EWC $\lambda=0$ 冒烟。** 若构造函数允许 `ewc_lambda=0`，它应接近 Naive。预算：一条短 Split MNIST。预期：点靠近 naive。失败判据：$\lambda=0$ 仍和 `0.4` 一样稳，说明实现把重要性用到了别处，第 05 课优先读源码。不要把这一条写成论文复现。

## 12. 论文与延伸

每篇对应一个能用本课实验回答或明确答不了的问题。读完把答案写进 `NOTES.md`。主阅读是 2024–2026 的可塑性丢失；谱系只留本课实验真用到的两篇。

1. McClelland, J. L., McNaughton, B. L. & O'Reilly, R. C., 1995, *Why there are complementary learning systems in the hippocampus and neocortex: insights from the successes and failures of connectionist models of learning and memory*, [DOI 10.1037/0033-295X.102.3.419](https://doi.org/10.1037/0033-295X.102.3.419)。
贡献：用联结主义模型的灾难性干扰，论证需要海马（快、分离）和新皮层（慢、交错）两套系统。机制发明处，不是本课主阅读。
机制：改的是训练协议：重叠表征上快速写入新任意联想会干扰；同一网络若新旧交错、小步更新，可以慢慢抽结构。没有可直接贴进 PyTorch 的损失项。
和本课：混旧数据是交错的最小投影。冻骨干对应不到海马：`frozen_backbone_does_not_move` 只说明骨干位移为 0。睡眠巩固对应不到你的任何一行代码。
阅读问题：论文认为「快速把新任意联想写入重叠网络」为何必然干扰？用 naive 点的 $S$ 回答。睡眠巩固对应你代码的哪一行？若对应不上，写进失效说明。

2. Kirkpatrick, J. et al., 2017, *Overcoming catastrophic forgetting in neural networks*, [arXiv:1612.00796](https://arxiv.org/abs/1612.00796)，期刊 [DOI 10.1073/pnas.1611835114](https://doi.org/10.1073/pnas.1611835114)。
贡献：把突触巩固写成 EWC，用 Fisher 对角线当弹簧劲度。机制发明处，不是本课主阅读。
机制：改损失：$\mathcal{L}_B+\sum_i (\lambda/2) F_i(\theta_i-\theta_{A,i}^\star)^2$。不存旧样本。摘要在 MNIST 分类和顺序 Atari 上展示可以记住旧任务仍学新任务。
和本课：Step 3 的 Avalanche `EWC` 是现象预告。`freeze_more_stable_than_naive` 是硬约束版「少动」，不是 Fisher。Fisher 怎么算本课答不了。
阅读问题：均匀把学习率乘 0.1，和按 $F_i$ 给不同弹簧，平面上应差在哪？用小学习率点与 EWC 预告点回答。对角线 Fisher 本课实验答不了，留给第 05 课。

3. Dohare, S. et al., 2024, *Loss of plasticity in deep continual learning*, *Nature* 632, 768-774, [DOI 10.1038/s41586-024-07711-7](https://doi.org/10.1038/s41586-024-07711-7)；预印本 [arXiv:2306.13812](https://arxiv.org/abs/2306.13812)（预印本标题 *Maintaining Plasticity in Deep Continual Learning*）。
贡献：系统证明标准深度学习在持续设定里会丢掉可塑性，直到学得不比浅层网络好。
机制：改评测：把 ImageNet 收成约 2000 个二分类任务。摘要：早期任务约 89%，第 2000 个任务掉到约 77%，接近线性网络。补丁是持续反向传播：每步按效用重初始化一小部分少用单元。L2 加权重扰动也能减轻。
和本课：本课 $P=R_{2,2}$ 是两个任务上的可塑性。论文的「第 2000 个任务还学不学得进」本课答不了。冻骨干让 $P$ 低，来自你把 `requires_grad` 关掉。论文里是单元逐渐饱和、有效秩下降。
阅读问题：`freeze_less_plastic_than_naive` 为真，能不能当成「可塑性丢失」？用骨干位移是 0 还是单元饱和来区分。论文的 2000 任务曲线本课实验答不了。

4. Elsayed, M. & Mahmood, A. R., 2024, *Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning*, [arXiv:2404.00781](https://arxiv.org/abs/2404.00781)。
贡献：用同一套效用门控，同时保护有用单元、扰动无用单元，对流式、未知任务边界的设定同时打遗忘和可塑性丢失。
机制：改更新：效用高的权重几乎不动，效用低的权重加大梯度加噪声。不存回放、不需要任务边界。PPO 上 Adam 后期掉分，UPGD 能避开。
和本课：冻骨干是效用门控的极端：骨干效用被你设成无穷。本课没有流式几百次非平稳，也没有效用估计。
阅读问题：若只冻骨干、不扰动无用单元，平面上 $P$ 会落在哪一侧？用 `freeze_acc_task2` 对 `naive_acc_task2` 回答。论文的流式几百次任务本课答不了。

5. Wang, J., Chandra, R. & Zhang, S., 2025, *Experience Replay Addresses Loss of Plasticity in Continual Learning*, [arXiv:2503.20018](https://arxiv.org/abs/2503.20018)。
贡献：提出假说：经验回放加上 Transformer 处理缓冲，可塑性丢失会消失。
机制：改存储和架构：标准反传、标准激活、不加正则，只加回放缓冲，并用 Transformer 读缓冲。猜想靠上下文学习。回归、分类、策略评估上都做了。
和本课：混旧数据是回放的极瘦影子，没有 Transformer。`mix` 点的 $P$ 若接近 naive，只能说明两个任务上交错有效，答不了「可塑性丢失消失」。
阅读问题：把 $\beta$ 从 0 调到 0.1，$P$ 有没有掉、掉多少？这验证的是交错对当前新任务的代价。论文的「加 Transformer 后可塑性丢失消失」本课实验答不了。

6. Prakash, A. et al., 2025, *Spectral Collapse Drives Loss of Plasticity in Deep Continual Learning*, [arXiv:2509.22335](https://arxiv.org/abs/2509.22335)。
贡献：论证新任务初始化处 Hessian 谱坍缩，有意义的曲率方向消失，梯度下降随后失效。
机制：线性化 ReLU 网上给出成功训练的 $\epsilon$-rank 条件，并证明损失加权 Gram 与广义 Gauss-Newton 谱等价。补丁方向：保持特征有效秩，加 L2。
和本课：本课不算 Hessian。冻骨干后特征不动，谱被你钉死，和训练过程中的谱坍缩要分开。
阅读问题：冻骨干之后 $P$ 低，是因为曲率方向没了，还是因为特征对任务 2 不够用？本课没有谱，只能用「骨干位移为 0、头还在训」排除前一种说法；论文的 $\epsilon$-rank 本课答不了。

7. Lillo, L. & Cheney, N., 2025, *Activation Function Design Sustains Plasticity in Continual Learning*, [arXiv:2509.22562](https://arxiv.org/abs/2509.22562)。
贡献：激活函数形状是减轻可塑性丢失的主杠杆，不换宽度、不换任务超参。
机制：改非线性：根据负支形状和饱和行为提出 Smooth-Leaky 与 Randomized Smooth-Leaky。在类增量监督和非平稳 MuJoCo 上评。
和本课：CPU 实验是线性骨干加头，没有隐藏激活可换。手写 MLP 的 ReLU 可以换，但本课四点对照没要求你换激活。
阅读问题：若你在手写 MLP 里把 ReLU 换成 Leaky ReLU，冻骨干的 $P$ 会不会明显上升？本课标准实验答不了，除非你做了这条改造。

8. Hernandez-Garcia, J. F., Dohare, S., Luo, J. & Sutton, R. S., 2025, *Reinitializing weights vs units for maintaining plasticity in neural networks*, [arXiv:2508.00212](https://arxiv.org/abs/2508.00212)。
贡献：比较重初始化单元与重初始化权重，并提出按效用重初始化最无用权重。
机制：改的是哪些参数被重置。对照持续反传和 ReDo（重初始化单元）。摘要：网络很窄，或带 LayerNorm 时，重初始化权重更有效；够宽且无 LayerNorm 时两者差不多。
和本课：冻骨干是零更新，不是重初始化。本课没有 LayerNorm，也没有按效用重置。
阅读问题：冻骨干等于把骨干效用设成无穷且永不重置。这会把平面点推向哪一角？用 `freeze_acc_task1` 和 `freeze_acc_task2` 回答。

9. Joudaki, A. et al., 2025, *Barriers for Learning in an Evolving World: Mathematical Understanding of Loss of Plasticity*, [arXiv:2510.00304](https://arxiv.org/abs/2510.00304)。
贡献：用动力系统给可塑性丢失下定义：参数空间里的稳定流形困住梯度轨迹。
机制：指出两类陷阱：激活饱和造成的冻单元，表征冗余造成的克隆单元流形。静态泛化喜欢的低秩、简单性偏差，在持续设定里会喂给这些陷阱。缓解：结构选择或定向扰动。
和本课：线性骨干几乎没有饱和。冻骨干是你外加的硬约束，不是优化自己走进稳定流形。
阅读问题：`frozen_backbone_does_not_move` 为真，对应论文的冻单元还是外加约束？用代码里 `freeze_backbone=True` 时 `continue` 跳过骨干梯度来答。

10. Hernandez-Garcia, J. F., Figliolia, T. & Millidge, B., 2026, *Can Scale Save Us From Plasticity Loss in Large Language Models?*, [arXiv:2606.24752](https://arxiv.org/abs/2606.24752)。
贡献：在 5M 到 314M 非嵌入参数的 GPT 式模型上问：放大能不能单独救可塑性丢失。
机制：改评测：多语言持续学习，用留出的越南语探针。摘要：可塑性丢失的发生按可预测的缩放律走，随模型变大亚线性推迟；稳态多语言训练里也会丢可塑性。结论：只加参数数量不够彻底防止。
和本课：本课小 MLP 已经否定「容量单独消灭遗忘」。这篇否定的是「容量单独消灭可塑性丢失」，设定是语言模型，本课答不了越南语探针。
阅读问题：冻骨干的 $P$ 低于 naive，加宽隐藏层会不会把这个缺口补上？本课标准四点对照没扫宽度，答不了；论文的 314M 缩放律本课实验也答不了。

现在你有一张会说话的平面：四个点证明「少动」和「交错」不是同一旋钮，CLS 故事在 MLP 上缺睡眠、缺编码器、缺双时间尺度。系统仍然不会在真实任务上持续学习，但它终于知道稳定和可塑会打架。下一课把同一张 $R$ 矩阵变成可复用的评测协议，并演示平均准确率怎样把「只会最后一件事」夸成好方法。去 [第 03 课](03_cl_evaluation.md)。



