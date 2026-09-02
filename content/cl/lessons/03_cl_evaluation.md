---
id: 03_cl_evaluation
title: "怎么量才算学会了"
summary: "只报最终平均准确率，为什么会把只会最后一件事的方法夸成好方法？"
unit: forget
play_tools: []
checkpoints:
  - "会算 Average Accuracy、Forgetting、BWT、FWT。"
  - "一份后面 21 课共用的评测协议模板。"
  - "知道 prequential 和「每个任务结束打一次分」的差别。"
---

# 第 03 课：怎么量才算学会了

> 类型：实战（按 GEM 论文公式手算并写指标；Avalanche 官方 metric 与 Mammoth 日志对照）<br>
> 建议周期：2-3 天<br>
> 硬件：CPU / Mac 即可，本课不训大模型<br>
> 锚定仓库：[ContinualAI/avalanche](https://github.com/ContinualAI/avalanche) 的 evaluation 模块；对照 [aimagelab/mammoth](https://github.com/aimagelab/mammoth) 的运行日志<br>
> 产物：一张被拆穿的「最后任务满分」矩阵、后面 21 课共用的评测协议模板、对 GDumb 的预习判断

## 1. 这一课做什么

整门课始终在造同一个循环里的零件：

```text
新经验进来
  先决定写到哪里（上下文 / 外挂记忆 / 快速权重 / 慢速权重）
  再决定怎么写（覆盖、追加、压缩、约束、正交、合并）
  写完立刻测：新任务会了没、旧任务还在不在
  长期还要测：还能不能继续学
```

第一幕要先把病看清楚，再谈补丁。[第 01 课](01_catastrophic_forgetting.md) 用 naive 接着训，把任务 1 的准确率看着塌下去。[第 02 课](02_stability_plasticity.md) 把四种写法落到稳定性-可塑性平面上：冻骨干更稳、更学不进；放开学新的，旧的又掉。本课换零件：不改训练方法，改怎么打分。

没有这一课，后面所有「我的方法更好」都可能只是换了一种会说谎的汇总方式。最常见的谎是：只报最终平均准确率。一个网络可以在最后一个任务上接近满分，把前面的任务全部忘光，平均分仍然好看，尤其当你把「刚学完时的分数」和「学完全程后的分数」混在一起报。

做完你会拿到三样能一直用到第 24 课的东西：

1. 一张任务-时间准确率矩阵 $R$，以及从它算出的 Average Accuracy、Average Forgetting、后向迁移（BWT，backward transfer：学新任务对旧任务的影响）、前向迁移（FWT，forward transfer：学过的东西对还没训的任务有没有帮助）、学习准确率。
2. 同一串预测上，任务结束打分和按时间顺序「先预测再训练」（prequential / walk-forward）的差别。
3. 一份书面协议：以后每课报告数字时，必须写清设定、矩阵、分母、种子和对照。

本课是实战，不是论文数字复现。课内按 Lopez-Paz 与 Ranzato 2017 的公式（[arXiv:1706.08840](https://arxiv.org/abs/1706.08840)）自己算；Avalanche 用来核对官方 metric 的符号和触发时机；Mammoth 用来看研究仓库实际把什么写进日志。你自己算出来的分母和符号，比仓库打印的某一个叫 accuracy 的标量更值得信。GDumb（把缓冲里的样本存下来、每个阶段从头训）为什么能打赢很多方法，正式打脸放在 [第 08 课](08_gem_gdumb.md)，本课只预习：如果你的协议允许「到点用背包重训」，平均分高并不能证明你在持续学习。先把题出对，再谈谁赢。

术语速查：

| 术语 | 一句解释 |
|---|---|
| 准确率矩阵 $R$ | 第 $i$ 行第 $j$ 列：$R_{i,j}$ 是刚学完任务 $i$ 之后，在任务 $j$ 测试集上的准确率 |
| Average Accuracy（ACC） | GEM 定义：学完全程后，对所有任务测试准确率的平均，也就是 $R$ 最后一行的均值 |
| Learning Accuracy（LA） | 每个任务刚学完、还没被后面任务冲过时的准确率平均，也就是 $R$ 对角线的均值 |
| Forgetting（遗忘） | Avalanche 默认：某任务「第一次记下的准确率」减「最后一次记下的准确率」；越大忘得越多 |
| BWT（后向迁移） | GEM / Avalanche：最后一次减第一次；负值就是遗忘，正值表示后来的学习帮了旧任务 |
| FWT（前向迁移） | 还没轮到任务 $j$ 时，模型在 $j$ 上的准确率相对随机初始化的增益 |
| prequential | 流式设定：每条样本先预测再训练，用在线预测对错的平均当成绩，不依赖任务边界 |
| EvaluationPlugin | Avalanche 里挂指标和日志的插件：策略每走到约定时刻，它负责算 metric、发给 logger |
| 经验 / experience | Avalanche 对「一段连续数据」的称呼，本课和「任务」在 SplitMNIST 上一一对应 |

## 2. 问题

第 01 课的热力图已经说明：naive fine-tune 会把旧决策边界抹掉。第 02 课的平面图说明：稳和塑经常对着干。现在的麻烦更隐蔽：同一张热力图，换一种汇总，结论可以反过来。

设五个任务顺序到来。有人交出这样的成绩单：「平均准确率 98%。」他的算法是：每个任务开始时把网络当成新的来训，只在当前任务的测试集上打分，再把五个「刚学完」的分数取平均。对角线全是满分，旧任务从来没被重测。这不是持续学习，这是五次独立的短训。

另有人交出：「最终平均准确率 60%。」他的算法只会最后一件事，旧任务掉到随机。在二分类任务增量里，随机大约 50%，60% 看起来「比乱猜好」。如果你不看 BWT，你会以为它学到了东西。

本课要回答四个具体问题：

1. 只报最终平均准确率，会把哪种「根本没在持续学」的方法夸成好方法？
2. Forgetting 和 BWT 差一个符号，Avalanche 源码里各自定义成什么？和 GEM 原文是否一致？
3. 每个任务结束打一次分，和按时间顺序逐条预测再训练，量到的分别是什么？后者会不会也漏掉遗忘？
4. 后面 21 课共用的协议最少要写哪几项，才不会在 [第 08 课](08_gem_gdumb.md) 被 GDumb 打脸时说不清「我到底在比什么」？

本课不讲 EWC、回放、LoRA。那些是第二幕以后的写法。这里只把打分规则写死。分母错了，[第 05 课](05_ewc_regularization.md) 把 $\lambda$ 扫一遍也没有意义。

## 3. 准备

- 第 01、02 课的结论要能口述：naive 会忘；冻骨干更稳更塑不了。本课实验不依赖那两课的 checkpoint。
- Python 3.10+、能装 PyTorch。机制实验在 `learn-cl/experiments/` 下跑，不下载模型、不需要 GPU。
- 锚定仓库实验需要一个独立虚拟环境：Avalanche 用 `pip install avalanche-lib`（官方安装页当前写法）；Mammoth 按仓库文档 `pip install -r requirements.txt`。不要和日常环境混装。
- 纸和计算器，或者任意能跑几行 numpy 的解释器。第 7 节有一张故意造假的 $5\times 5$ 矩阵，你要先手算再让程序核对。
- 浏览器能打开本课页面上的「指标打假器」。网页实验全部在本地算完，不请求模型 API。

## 4. 学习目标

1. 给定任意 $T\times T$ 的 $R$，能手算 ACC、LA、逐任务 Forgetting、BWT、FWT，并指出分母里有没有包含任务 1 的 BWT、任务 $T$ 的 FWT（这两项按定义不存在）。
2. 构造并解释一个「最后任务接近满分、前面掉到随机」的矩阵：指出哪个汇总会给它高分，哪个会揭穿它。
3. 说明 Avalanche 的 `Forgetting` 是「第一次减去最后一次」，`BWT` 是「最后一次减去第一次」，二者在只比较 $R_{i,i}$ 与 $R_{T,i}$ 时互为相反数。
4. 说出 prequential 成绩高，为什么仍可能把旧分布忘光；任务结束重测为什么仍可能漏掉「学得慢」。
5. 独立写下一份评测协议（第 9 节清单），并说明 GDumb 会在哪些协议条款下合法地拿到高 ACC。
6. 能把 Avalanche `EvaluationPlugin` 的一次 `eval(test_stream)` 对上 $R$ 的一行，把 Mammoth 日志里的逐任务准确率填进同一张表。

## 5. 原理

五个机制，每个按同一节奏：为什么需要、怎么运转、精确定义、代码落点、怎么证明做对了。

### 5.1 准确率矩阵：先把时间轴摊开，再谈平均

持续学习的原始记录不是一个数，是一张表。任务按 $1,2,\ldots,T$ 到来。每学完一个任务 $i$，立刻用**同一个**测试协议，把全部 $T$ 个任务（含还没训过的）测一遍。记

$$
R_{i,j}=\text{学完任务 }i\text{ 之后，任务 }j\text{ 测试集上的准确率。}
$$

行是时间，列是考哪门课。对角线 $R_{i,i}$ 是「刚学会时有多好」。最后一行 $R_{T,\cdot}$ 是「全部学完后还剩多少」。上三角 $R_{i,j}$（$j>i$）是「还没教 $j$，先考 $j$」。下三角 $R_{i,j}$（$j<i$）是「教了后面的，前面的还在不在」。

类比：学期成绩单。只看毕业考试总分，会把「大四一门满分、大一大二全忘」的人排到前面。类比失效处：学校至少还有学分和必修，你的训练脚本没有，除非你自己规定必须重测旧任务。

GEM 原文把这张表写成 $R\in\mathbb{R}^{T\times T}$，并规定 $R_{i,j}$ 是「观察到任务 $i$ 的最后一个样本之后」在任务 $j$ 上的测试分类准确率。更细的版本可以每看到一个 mini-batch 就加一行，那是学习曲线，不是本课默认协议。本课默认：每个任务结束写一行。

验证：任意实验，先问「$R$ 在哪」。答不出 $R$，后面的平均分作废。

### 5.2 Average Accuracy 会说谎，Learning Accuracy 更会说谎

直觉上「平均准确率」应该能代表整体水平。分母一换，代表的东西就换了。

GEM 的 Average Accuracy 只用最后一行：

$$
\mathrm{ACC}=\frac{1}{T}\sum_{i=1}^{T}R_{T,i}.
$$

它问的是：全部学完之后，旧的和新的平均还剩多少。这已经比「只报最后一个任务」诚实。它仍然会被两种写法抬高。

第一种：最后任务特别容易，或者最后任务的测试集特别大，你却用微平均（按样本数加权）而不是按任务宏平均。最后一门占分多，前面塌了看不出来。

第二种，也是本课要故意构造的：模型是「最后任务专家」。每个任务刚学完都很好，一换任务就把旧的冲到随机。最后一行接近 $(\varepsilon,\varepsilon,\ldots,\varepsilon,1)$，其中 $\varepsilon$ 是随机水平。ACC 等于 $((T-1)\varepsilon+1)/T$。$T=5$、任务增量二分类、$\varepsilon=0.5$ 时，ACC $=0.6$。0.6 看起来像「有用的方法」。

Learning Accuracy 只看对角线：

$$
\mathrm{LA}=\frac{1}{T}\sum_{i=1}^{T}R_{i,i}.
$$

最后任务专家的 LA 可以接近 1。有人把 LA 叫做 average accuracy，论文摘要里就出现「我们达到 98%」。那 98% 从未询问旧任务还在不在。

Díaz-Rodríguez 等人 2018（[arXiv:1810.13166](https://arxiv.org/abs/1810.13166)）把 Accuracy 改成对下三角（含对角线）全部 $R_{i,j}$（$i\ge j$）再平均。分母是 $T(T+1)/2$。它承认「过程中的表现」，因此更会把「当时会、后来忘」算进总分。本课协议里允许把它当作**过程准确率**另报，禁止用它替换 GEM 的 ACC。

数学上把「最后任务专家」写出来。任务增量、每任务两类、随机 $\varepsilon=0.5$：

$$
R^{\text{last}}=\begin{pmatrix}
0.98 & 0.50 & 0.50 & 0.50 & 0.50 \\
0.51 & 0.97 & 0.50 & 0.50 & 0.50 \\
0.50 & 0.52 & 0.96 & 0.50 & 0.50 \\
0.49 & 0.50 & 0.51 & 0.98 & 0.50 \\
0.50 & 0.50 & 0.50 & 0.50 & 0.99
\end{pmatrix}
$$

则 $\mathrm{ACC}=(0.50+0.50+0.50+0.50+0.99)/5=0.598$，$\mathrm{LA}=(0.98+0.97+0.96+0.98+0.99)/5=0.976$。只报 ACC 像及格，只报 LA 像满分。两者都没描述遗忘。

再看两张不会出现在「最后任务专家」里、但报告里很常见的矩阵。

**只测当前任务。** 评测脚本每次只调用 `eval(test_stream[i])`，从不回头。你拿到的不是 $R$，是对角线五个数。对外报「平均 97.6%」完全合法地等于 LA。读者以为那是 ACC。拆穿方法只有一个：要最后一行。

**GDumb 风格的最终重训。** 五个任务期间模型可以乱忘。学完任务 5，用缓冲里的平衡样本从头训一个分类器，再测全部任务，得到最后一行 $(0.88, 0.90, 0.87, 0.91, 0.89)$。ACC $\approx 0.89$，BWT 若仍用「当时的 $R_{i,i}$ 对现在」来算，可能是小负数（从头训不一定达到当时峰值），看起来像「记住了」。它记住的是背包，不是权重里的持续更新。协议必须加一行：`retrained_from_buffer_at_eval: yes/no`。第 08 课才会跑真的 GDumb；本课只要在表上能认出这种最后一行。

验证：第 7 节 CPU 实验会断言「同一矩阵上 LA 高、BWT 差」。你手算必须和断言一致。对角线上报和最后一行上报必须分成两个字段。

### 5.3 遗忘和后向迁移：同一个差，符号相反

旧任务掉了多少，最直的量是「当时会多少，现在会多少」。

GEM 的 Backward Transfer：

$$
\mathrm{BWT}=\frac{1}{T-1}\sum_{i=1}^{T-1}\bigl(R_{T,i}-R_{i,i}\bigr).
$$

任务 1 没有「更早的任务」可迁移，任务 $T$ 没有「更晚的任务」来伤害它，所以平均只对 $i=1,\ldots,T-1$。BWT 为负就是遗忘；为正表示后来的学习帮了旧任务（正后向迁移）。GEM 的卖点之一正是允许正 BWT：约束写成不等式，旧损失可以下降，不许上升。

Avalanche 把同一件事拆成两个 standalone metric，源码在 `avalanche/evaluation/metrics/forgetting_bwt.py`（API 文档把 `Forgetting` 和 `BWT` 都标到这个文件）：

- `Forgetting`：某个 key 的**第一次**记录减去**最后一次**记录。
- `BWT`：最后一次减去第一次。

当第一次就是 $R_{i,i}$、最后一次就是 $R_{T,i}$ 时，

$$
\mathrm{Forgetting}_i=R_{i,i}-R_{T,i}=-\mathrm{BWT}_i.
$$

Stream 级再对已经见过的经验做平均，得到 `StreamForgetting`、`StreamBWT`。对上面的 $R^{\text{last}}$：

$$
\mathrm{BWT}=\frac{(0.50-0.98)+(0.50-0.97)+(0.50-0.96)+(0.50-0.98)}{4}=-0.4725,
$$

$$
\mathrm{Forgetting}=0.4725.
$$

有的论文用 $\max_{t\in\{i,\ldots,T-1\}}R_{t,i}-R_{T,i}$，防止「刚学完不是峰值、中间某次评测更高」。本课默认 GEM / Avalanche 的「第一次对最后一次」。你改用 max 版必须在协议里写明。

Díaz-Rodríguez 把 BWT 扩成对所有 $i>j$ 的 $(R_{i,j}-R_{j,j})$ 再平均，并拆成 Remembering 与正 BWT 两项，映射到 $[0,1]$ 方便加权打总分。本课不采用他们的 $CL_{score}$ 作为主指标：权重是主观的，21 课无法共用一套权重。他们提出的**内存占用、样本缓冲占用、计算量**三项，本课协议要求记录，不折进一个分数里。

验证：同一矩阵，Forgetting 与 BWT 之和必须在数值误差内为 0。CPU 实验会钉这条。

### 5.4 前向迁移：还没教的课，现在能考几分

FWT 问的是：学了前面的任务，对还没见过的任务有没有帮助。GEM 定义：

$$
\mathrm{FWT}=\frac{1}{T-1}\sum_{i=2}^{T}\bigl(R_{i-1,i}-\bar{b}_i\bigr),
$$

其中 $\bar{b}_i$ 是随机初始化（尚未看到任何任务）时任务 $i$ 的测试准确率。对任务 $i$，用的是「刚学完任务 $i-1$、还没学任务 $i$」那一行。整数任务编号几乎给不出正 FWT；正 FWT 更多出现在任务描述有结构、或者表示在任务间可复用时。

$R^{\text{last}}$ 的上三角几乎全是 $0.50$，若 $\bar{b}_i=0.50$，则 $\mathrm{FWT}=0$。它没在未来任务上变差，也没变好。一个把未来任务也破坏掉的方法（共享头被最后一类占满）会得到负 FWT。

Avalanche 的 `ForwardTransfer`（`avalanche/evaluation/metrics/forward_transfer.py`）按文档：某 key 在「前一个经验训完之后」的值，减去「训练开始前随机初始化」的值。这和 GEM 一致。插件版本是 `forward_transfer_metrics(experience=True, stream=True)`。

Wu 等人 2024 的大模型综述（[arXiv:2402.01364](https://arxiv.org/abs/2402.01364)）§7.1 把 ACC / BWT / FWT 列为 LLM 持续学习的三项典型指标，并引用 GEM。BWT、ACC 的写法与 GEM 一致。该节 FWT 的下标写成了 $A_{T,i}$，和 GEM 的 $R_{i-1,i}$ 不是同一格；本课以 GEM 原文和 Avalanche 实现为准。综述真正要提前记住的是另一件事：大模型还有跨阶段遗忘（续预训练、指令微调、对齐互相冲），那是第三幕的量法，本课的 $R$ 仍然适用，只是「任务」换成阶段或指令集。

验证：算 FWT 之前必须有一行「训任务 1 之前」或明确的随机基线 $\bar{b}$。没有 $\bar{b}$ 就不要报 FWT，改报上三角原始值。

### 5.5 任务结束打分，对不上逐条预测；协议必须两者都能写

任务边界清楚时，$R$ 矩阵是标准量法。真实部署常常没有边界：样本一条条来，你不能等「这个任务结束」再考试。

prequential（预训练式 / 逐条先考后学）做法：对时刻 $t$ 的样本 $x_t$，先用当前模型预测，记下对错或损失，再把 $x_t$ 用于更新。全程平均就是 prequential 准确率或 prequential 损失。Walk-forward 是它的窗口版：用过去一段时间训练，预测下一段时间，窗口向前滚。

它量到的是「在到来的分布上，当时猜得准不准」。它**量不到**「三个月前的分布现在还会不会」，除非旧分布会再次出现，或者你另外保留旧任务测试集。最后任务专家在 prequential 上可以很好看：每个阶段的新样本它都能很快学会，旧样本不再出现，遗忘零成本。

反过来，只在任务结束打分，会漏掉「学得很慢、但考前刷到了」。一个方法可以在任务内部过拟合到测试分布，任务结束分数高，线上逐条预测一塌糊涂。

所以协议里要写清评测时刻：

| 时刻 | 量到什么 | 漏掉什么 |
|---|---|---|
| 每个任务结束，重测全部任务 | $R$ 的一行；遗忘、BWT | 任务内部的在线表现 |
| 每个 mini-batch / 每条样本先预测再训练 | 在线适应速度 | 不再出现的旧分布 |
| 学完全程后只测一次 | 最终 ACC | 过程、正负迁移、学得慢还是学得快 |

用数字把 prequential 的盲区写死。假设五个阶段各 100 条样本，最后任务专家在每个阶段的前 20 条乱猜（0.50），后 80 条学会当前任务（0.95）。prequential 平均是

$$
\frac{5\times(20\times 0.50+80\times 0.95)}{500}=0.86.
$$

0.86 看起来强。与此同时 ACC $=0.598$，BWT $=-0.4725$。在线分数和持续学习分数讲的不是同一件事。若旧样本以 10% 的概率重现，prequential 才会开始吃到遗忘；那已经是带回放的世界，对应第 06 课，不是本课的默认流。

第 24 课的 14 日上岗会同时用「当天任务成功率」（接近 prequential）和「周一的题周五再考」（接近 $R$ 的旧列）。现在就把两种时刻写进习惯。

后面 21 课共用的最小协议如下，第 9 节验收按它勾：

1. 设定：task / domain / class incremental（第 01 课三种），以及有没有任务标识。
2. 序列：任务个数、每任务类别或指令集、顺序是否打乱、种子。
3. 原始记录：完整 $R$，或流式设定下的逐段日志；禁止只交一个平均数。
4. 汇总：ACC（GEM）、LA、Forgetting 或 BWT（写明用哪一个）、FWT（有 $\bar{b}$ 才报）。
5. 对照：naive；可能的话 joint / iid 混训上限。缓冲类方法必须报缓冲大小。从第 08 课起加上 GDumb。
6. 预算：epoch、学习率、是否多遍扫描每任务。GEM 强调「每条样本只见一次」时，多 epoch 会夸大遗忘，必须写清。
7. 可塑性：长序列还要看后期任务的学习速度，那是 [第 15 课](15_loss_of_plasticity.md) 的主指标，本课先占一个空位。

GDumb 预习：它把样本放进固定大小的平衡缓冲，评测时用缓冲从头训练一个模型。任务边界干净、缓冲里近似 i.i.d. 时，ACC 可以很高，BWT 看起来也不差，因为它根本不在旧权重上接着训。协议如果只比最终 ACC，GDumb 会赢。这不证明持续学习无用，只证明你的题太干净。第 08 课用同一协议跑 A-GEM、DER++、GDumb；本课先把「同一协议」四个字钉死。

同一套公式对着四种假想方法，数字全部由矩阵算出来，方便你以后读论文摘要时对照。随机基线 $\bar{b}_i=0.50$。

| 方法 | 最后一行（示意） | ACC | LA | BWT | FWT | 只看 ACC 会怎样 |
|---|---|---|---|---|---|---|
| 最后任务专家 | 0.50, 0.50, 0.50, 0.50, 0.99 | 0.598 | 0.976 | -0.47 | 0.00 | 以为及格 |
| 独立头、冻骨干 | 0.90, 0.88, 0.87, 0.89, 0.86 | 0.880 | 0.890 | -0.01 | 0.00 | 看起来很好，可塑性另测 |
| 正后向迁移 | 0.99, 0.98, 0.97, 0.96, 0.95 | 0.970 | 0.930 | +0.04 | 0.02 | 和上限难分，要看有没有偷用旧数据 |
| 缓冲重训 | 0.88, 0.90, 0.87, 0.91, 0.89 | 0.890 | （训练中对角线可很低） | 含义含糊 | 0.00 | 会赢很多「在线方法」 |

独立头那一行对应第 02 课冻骨干：旧任务几乎不掉，新任务若共享表示不够会变弱，LA 和 ACC 接近。正后向迁移那一行才是 GEM 想展示的现象：后来的任务把前面也带高一点。缓冲重训那一行警告你：ACC 高、BWT 字段含糊时，先问评测前有没有从头训。四行都合法地「平均准确率很高」，只有把 $R$ 摊开以后才分得开。

## 6. 源码导读

读代码带着问题进去。路径以你克隆时的仓库为准；import 以官方文档为准。Avalanche 安装页当前命令是 `pip install avalanche-lib`。From Zero to Hero 的 Evaluation 教程示例里写过 `avalanche-lib==0.6`，指标名字和 `EvaluationPlugin` 用法在 0.6 与当前 latest API 文档中一致。不确定版本时，先 `python -c "import avalanche; print(avalanche.__version__)"` 记进实验笔记。

| 位置 | 带着什么问题读 |
|---|---|
| `avalanche.evaluation.metrics.accuracy_metrics` | `experience=True` 和 `stream=True` 各对应 $R$ 的一格还是一行的平均？ |
| `avalanche.evaluation.metrics.forgetting_metrics` / `bwt_metrics` | Forgetting 与 BWT 是否只是符号相反？第一次记录发生在什么回调？ |
| `avalanche.evaluation.metrics.forward_transfer_metrics` | $\bar{b}$ 在哪一次 `eval` 写入？若你从未在训练前 eval，FWT 会不会是空的？ |
| `avalanche.training.plugins.EvaluationPlugin` | `strict_checks` 在干什么？为什么教程示例把它设成 `False`？ |
| `avalanche.training.Naive` + `SplitMNIST` | 每个 `experience` 训完后 `eval(benchmark.test_stream)` 是否正是在写 $R$ 的新一行？ |
| Mammoth `main.py` | `--model` 对应 `models/<name>.py`；`--dataset seq-mnist` 是 class-IL，同时会报 task-IL（推理时掩码到当前任务类别）。 |
| Mammoth `utils/loggers.py` 的 `log_accs` | 原始逐任务、逐类准确率如何落盘？文档说默认写在 `data/results/<setting>/<dataset>/<model>/logs.pyd`，每行一个字典。 |
| Mammoth 文档「First steps」 | 不配 `--wandb_project` 就不走 WandB。本课不强制 WandB。 |

Avalanche 官方教程把指标挂进策略的写法如下，这是本课锚定实验要对照的最小集合（完整可跑脚本在第 7 节）：

```python
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    forgetting_metrics,
    bwt_metrics,
    loss_metrics,
)
from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger

eval_plugin = EvaluationPlugin(
    accuracy_metrics(epoch=True, experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    bwt_metrics(experience=True, stream=True),
    loss_metrics(experience=True, stream=True),
    loggers=[InteractiveLogger()],
    strict_checks=False,
)
```

`strategy.train(experience)` 和 `strategy.eval(test_stream)` 的返回值是字典，键是带相位路径的 metric 名，例如教程里的 `Top1_Acc_Epoch/train_phase/train_stream/Task000`。你要自己把 `Top1_Acc_Exp/.../Exp00k` 抽出来填进 $R$ 的一行。不要直接相信字典里某一个叫 accuracy 的标量等于 GEM 的 ACC：它可能是 stream 微平均，也可能只含当前 experience。

把一次 `eval(benchmark.test_stream)` 展开。SplitMNIST 五个 experience，测试集按类别切开，每个 experience 大约 2000 张测试图（两类 × 约 1000）。插件按顺序走完五个 experience，每个结束发一个 `ExperienceAccuracy`，全部走完发一个 `StreamAccuracy`。`StreamAccuracy` 是这 10000 张图上的微平均。若五个任务测试集一样大，它碰巧等于 GEM 的 ACC；一旦最后两个 experience 的测试集更大，它就偏向新任务。`StreamForgetting` 只对「训练阶段已经见过」的经验平均，最后一个刚学完的任务通常不进遗忘分母，这和 GEM 的 $T-1$ 一致。读日志时先数分母，再看数字。

`EvaluationPlugin(..., benchmark=benchmark)` 会记住这条 test stream 的身份。中途改成 `eval(test_stream[:2])` 再改回去，`strict_checks=True` 会叫停。教程示例关掉它，是为了演示不被检查绊住。本课要完整 $R$，没有理由切片。

Mammoth 的入口按官方文档，在仓库根目录执行 `python main.py --model <model-name> --dataset <dataset-name>`。

可复现配置常用 `--model_config best`（`models/config/` 下的 yaml）。本课 naive 对照用 `--model sgd --dataset seq-mnist`。文档写明：class-IL 设定下会同时计算 task-IL 指标，task-IL 在推理时掩掉非当前任务类别。两套数字不要混着写进同一张 $R$。

## 7. 实验

三层都做。浏览器先预测；CPU 实验钉公式；锚定仓库核对官方实现和日志格式。工作目录：机制实验在 `learn-cl/experiments/`，Avalanche / Mammoth 在各自克隆目录。

### Step 1: 浏览器实验（指标打假器）

打开本课页面的交互实验（`lab-03-metric-liar`）。它给你一张 $5\times 5$ 的准确率矩阵，你先判断「这个方法算不算持续学习还在干活」，再运行。系统用遗忘和 BWT 揭晓。

先预测再点运行。建议按这三条规则猜：

- 最后一行接近随机、只有最后一列或对角线好看：判「不会持续学习」，即使某个平均分很高。
- 最后一行都高、对角线也高、BWT 接近 0：可能是真记住了，也可能是 GDumb 那种「到点重训」。网页若提供「是否从头重训」开关，把它打开再看一次。
- 上三角明显高于随机：才谈得上正 FWT。

改矩阵或选项会作废上次运行，需要重新预测。预测与 BWT / Forgetting 的判定一致才算过关。

### Step 2: CPU 机制实验

在 `learn-cl/experiments/` 下：

```bash
python3 run.py run 03
```

实验不训网络。它在固定种子下构造三张准确率矩阵：2×2 最后任务专家、同 ACC 但几乎不遗忘的对照、以及 5×5 专家用来钉 FWT。对矩阵按 GEM 公式算 ACC、LA、Forgetting、BWT、FWT。

预期写入 `artifacts/lesson03/result.json`，六个 `checks` 全真。这层钉的是公式和符号，不是论文分数，也不是 Avalanche 日志格式本身。本机一次运行（Python 3.13.13，seed=3），最后任务专家 ACC=0.74、BWT=−0.48、LA=0.98；对照同样 ACC=0.74、BWT=−0.03。换机器会变，方向不应变。构造矩阵本身按种子写死。

真实 `checks` 键名：

- `last_task_specialist_final_row`：专家矩阵最后一行旧任务掉到随机附近、新任务仍高（本机 `specialist2_matrix` 为 `[[0.98, 0.5], [0.5, 0.98]]`）；
- `average_accuracy_looks_high`：专家 ACC>0.70（本机 `specialist2_average_accuracy`=0.74）；
- `bwt_strongly_negative`：专家 BWT<−0.40（本机 `specialist2_bwt`=−0.48）；
- `learning_accuracy_hides_forgetting`：LA>0.95 且遗忘 >0.40（本机 LA=0.98，`specialist2_average_forgetting`=0.48）；
- `same_acc_different_bwt`：对照与专家 ACC 差 <0.02，但 BWT 差 >0.35（本机对照 ACC=0.74、BWT=−0.03）；
- `fwt_stays_at_chance_before_learning`：学之前的前向迁移停在随机（本机 `specialist5_fwt`=0.5）。

### Step 3: 手算打假矩阵

把 5.2 的 $R^{\text{last}}$ 抄下来，禁止用计算器的「矩阵平均」一键功能，按定义逐项加。你应该得到：

| 指标 | 值（保留四位小数） | 它在夸什么 |
|---|---|---|
| ACC | 0.5980 | 学完后平均，被最后一门 0.99 和随机 0.50 抬到「及格」 |
| LA | 0.9760 | 刚学完都很好，完全不看遗忘 |
| BWT | -0.4725 | 旧任务从约 0.97 掉到 0.50 |
| Forgetting | 0.4725 | 与 BWT 反号 |
| FWT（$\bar{b}_i=0.50$） | 0.0000 | 没帮到未来任务，也没提前破坏 |

再造一张「只报最后一列」的谎：有人把 $R_{5,5}=0.99$ 写成 average accuracy。协议上这是非法字段。CPU 实验如果提供 `last_task_only_score`，它必须不能通过「可作为 ACC 上报」的检查。

第三张对照矩阵：joint 上限，最后一行全是 $0.95$ 左右，对角线也高，BWT 接近 0。ACC 高且 BWT 不差，这才是「记得住」的样子。它用了全部旧数据，不是持续学习方法，是上限。

### Step 4: Avalanche 官方 evaluation 最小循环

独立虚拟环境，先装核心包（官方 How to Install）：

```bash
pip install avalanche-lib
```

若你想对齐教程笔记本的钉版本：

```bash
pip install avalanche-lib==0.6
```

下面脚本与 From Zero to Hero「Evaluation」教程同一结构：`SplitMNIST(n_experiences=5)`、`Naive`、`EvaluationPlugin` 挂上 accuracy / forgetting，每个 experience 训完评估整个 `test_stream`。缩小 `train_epochs` 只为在 CPU 上跑完。

```python
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    forgetting_metrics,
    bwt_metrics,
    loss_metrics,
)
from avalanche.models import SimpleMLP
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training import Naive

benchmark = SplitMNIST(n_experiences=5, seed=42)
model = SimpleMLP(num_classes=benchmark.n_classes)
eval_plugin = EvaluationPlugin(
    accuracy_metrics(
        minibatch=False, epoch=True, experience=True, stream=True
    ),
    loss_metrics(experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    bwt_metrics(experience=True, stream=True),
    loggers=[InteractiveLogger()],
    benchmark=benchmark,
    strict_checks=False,
)
cl_strategy = Naive(
    model,
    SGD(model.parameters(), lr=0.001, momentum=0.9),
    CrossEntropyLoss(),
    train_mb_size=128,
    train_epochs=1,
    eval_mb_size=128,
    evaluator=eval_plugin,
    eval_every=-1,
)
results = []
for experience in benchmark.train_stream:
    cl_strategy.train(experience)
    results.append(cl_strategy.eval(benchmark.test_stream))
```

预期：五个任务各一行日志。naive 在 SplitMNIST 的 class-incremental 设定下，早先 experience 的 `ExperienceAccuracy` 会掉，`StreamForgetting` 为正。把五个 `eval` 字典里各 experience 的 Top1 准确率填成 $5\times 5$ 的 $R$，用 Step 3 同一套公式手算 ACC 与 BWT，对照插件给出的 stream 值。允许数值因实现细节（微平均 / 宏平均、是否含未来 experience）有差，不允许符号反了：naive 的 BWT 必须为负。

`strict_checks=True` 时，插件会检查你每次 `eval` 是否对着同一条 stream。教程把它关掉是为了演示省事。本课协议要求：评测 stream 固定，中途不许换成另一半测试集。

可选：加上 `forward_transfer_metrics(experience=True, stream=True)`。若 FWT 全空，先在训练循环开始前对 `test_stream` 做一次 `eval`，给 `ForwardTransfer` 写入初始 $\bar{b}$。

### Step 5: Mammoth 日志对照

```bash
git clone https://github.com/aimagelab/mammoth.git
```

按仓库 README / 文档安装依赖（Pytorch 文档要求 $\ge 2.1.0$ 才能用部分 ViT 实现；本课 MNIST 用默认 backbone，CPU 可跑）：

```bash
pip install -r requirements.txt
```

在 Mammoth 仓库根目录跑 naive 短实验。`--debug_mode 1` 会把每个任务的 iteration 数减到很少（模型基类默认约 5 次），只为看到日志格式：

```bash
python main.py --model sgd --dataset seq-mnist --debug_mode 1
```

官方可复现口径是 `python main.py --model <name> --dataset <name> --model_config best`。本课不是复现 DER，不要把 debug 短跑的准确率写成论文对照。

预期：训练结束在 `data/results/` 下出现按 setting / dataset / model 组织的 `logs.pyd`。文档写明每行是一个含参数和结果的字典；class-IL 会同时给出 task-IL 数字。打开日志，找到各任务准确率，填 $R$。问自己三个问题：

1. 他们的「average accuracy」分母是 $T$ 还是见过的任务数？
2. class-IL 与 task-IL 两行能不能对上 Avalanche 同设定？
3. 有没有单独的 forgetting 字段？没有就自己用对角线减最后一行。

这一步的交付是「日志字段说明书」半页，不是分数。后面 DER、GDumb 都在这个仓库跑，字段对不上会从本课开始错。

### Step 6: 把协议写成可粘贴模板

在本次实验目录放 `PROTOCOL.md`，直接复制下面这块，填空，以后每课改任务名和种子。

```text
lesson:
setting: task-IL / domain-IL / class-IL
task_ids:
n_tasks: T
seed:
samples_seen_once: yes/no
epochs_per_task:
buffer_size: 0
metrics_from_R:
  ACC: GEM last-row mean
  LA: diagonal mean
  BWT: GEM (R_T,i - R_i,i) mean over i < T
  Forgetting: Avalanche first-minus-last
  FWT: GEM (R_{i-1,i} - b_i), b_i recorded: yes/no
prequential: on/off
baselines: naive, joint (if affordable)
not_reported_as_success: last-task accuracy, LA alone
notes:
```

一份 naive SplitMNIST 模板（把 `to-fill` 换成你 Step 4 跑出来的）：

```text
lesson: 03
setting: class-IL
task_ids: [0-1, 2-3, 4-5, 6-7, 8-9]
n_tasks: 5
seed: 42
samples_seen_once: no
epochs_per_task: 1
buffer_size: 0
R_last_row: [to-fill, to-fill, to-fill, to-fill, to-fill]
ACC: to-fill
LA: to-fill
BWT: to-fill
Forgetting: to-fill
FWT: not reported (b_i not recorded)
prequential: off
baselines: naive
retrained_from_buffer_at_eval: no
not_reported_as_success: last-task accuracy, LA alone
notes: SplitMNIST class-IL random floor is 0.10 if 10-way; do not copy the 0.50 toy matrix
```

`to-fill` 必须换成真实数之后，这份文件才能进第 05 课的对照目录。数字留空等于没写协议。

## 8. 配置与预算

| 档 | 做什么 | 耗时（参考） | 用途 |
|---|---|---|---|
| 浏览器 | 指标打假器，先预测再运行 | 15-30 分钟 | 建立「平均分会说谎」的手感 |
| CPU 机制 | `python3 run.py run 03` 加手算 $R^{\text{last}}$ | 30 分钟内 | 钉公式、钉符号 |
| Avalanche 冒烟 | SplitMNIST 5 任务、Naive、1 epoch | CPU 上通常数分钟到十几分钟 | 对照官方 metric 名称与 $R$ 的一行 |
| Mammoth 冒烟 | `sgd` + `seq-mnist` + `--debug_mode 1` | 数分钟（含第一次下 MNIST） | 对照日志字段 |
| 加分 | 去掉 debug，把 Mammoth naive 跑满默认 epoch；同一 $R$ 上再算 Díaz-Rodríguez 的下三角平均 | 数十分钟 | 看过程准确率怎样进一步抬分 |

主线在 CPU 完成。不要为了本课去跑 CIFAR 或 70 个方法；那是第 06、08 课的预算。磁盘：MNIST 很小；Mammoth 默认下到 `data/`，可用 `--base_path` 改位置。

随机性：手算矩阵是构造出来的，无种子。Avalanche 示例写 `seed=42`。Mammoth 把 seed 写进日志字典，抄到 `PROTOCOL.md`。比较两次运行时，设定、种子、epoch、是否 task-IL 掩码必须相同。

## 9. 验收

- [ ] 能在白纸上画出 $R$ 的行（时间）和列（任务），并标出 ACC、LA、BWT、FWT 各用哪些格子。
- [ ] 对 $R^{\text{last}}$ 手算 ACC、LA、BWT、Forgetting、FWT，与 5.2 / Step 3 表一致（四位小数）。
- [ ] 口头回答：为什么 LA $=0.976$ 不能写成「平均准确率 97.6%」。
- [ ] 浏览器实验已先预测再运行，判定与遗忘 / BWT 一致。
- [ ] `python3 run.py run 03` 的 `checks` 全真（`last_task_specialist_final_row`、`average_accuracy_looks_high`、`bwt_strongly_negative`、`learning_accuracy_hides_forgetting`、`same_acc_different_bwt`、`fwt_stays_at_chance_before_learning`）。
- [ ] Avalanche 循环跑通，naive 的 BWT 为负或 StreamForgetting 为正；你从返回字典抽出的 $R$ 至少有最后一行。
- [ ] Mammoth 冒烟跑通，能指出日志里 class-IL 与 task-IL 两套准确率；没有把 debug 短跑当成论文分数。
- [ ] `PROTOCOL.md` 按 Step 6 填完，明确「最后任务准确率」和「LA 单独」不得作为成功标准。
- [ ] 能用协议里的句子解释：GDumb 若只比最终 ACC，为什么可能赢；这件事留到第 08 课验证，本课只要求说得清。

## 10. 排错

| 症状 | 最可能的原因 | 怎么确认 | 怎么修 |
|---|---|---|---|
| 手算 BWT 和 Avalanche 差一个符号 | 你用了 Forgetting 的公式却标成 BWT | 看 $R_{T,i}-R_{i,i}$ 的正负 | GEM / Avalanche BWT 是最后减第一次；忘得多应为负 |
| StreamAccuracy 很高，经验级准确率已塌 | stream 是按样本微平均，最后经验样本多 | 打印每个 Exp 的 ExperienceAccuracy | 协议用宏平均 ACC，也就是对任务平均，不按样本加权 |
| FWT 字典为空 | 训练前没 eval，初始 $\bar{b}$ 没写入 | 查 `ForwardTransfer.initial` | 循环开始前 `eval(test_stream)` 一次 |
| `strict_checks` 报 stream 不一致 | 有的任务 eval 了 `test_stream`，有的只 eval 当前 experience | 对照每次 `eval` 的参数 | 固定评测 stream；本课要完整 $R$ 就必须 eval 全部 |
| 装 `avalanche-lib[all]` 把 PyTorch 锁到 2.0 以下 | extra 依赖限制 | pip 提示版本冲突 | 本课只装核心 `avalanche-lib`，不要 `[all]` |
| zsh 里 `pip install avalanche-lib[extra]` 失败 | 方括号被 shell 吃掉 | 报错像找不到包 | 本课不需要 extra；若你坚持，给包名加引号 |
| Mammoth `logs.pyd` 找不到 | 结果目录不在当前工作目录，或 `--base_path` 改过 | 搜 `data/results` | 看文档：默认 `data/results/<setting>/<dataset>/<model>/` |
| Mammoth 一启动要 WandB | 环境变量或旧脚本强制了 entity | 参数里是否出现 `--wandb_project` | 按 First steps：不传 project/entity 即可 |
| debug 模式分数接近随机 | iteration 被减到几次，没真正学完 | `--debug_mode 1` 仍在 | 这是格式冒烟，不是成绩；要看遗忘曲线就关掉 debug |
| `same_acc_different_bwt` 为假 | 两张矩阵 ACC 没对齐，或对照也忘光了 | 对照 `specialist2_average_accuracy` 与 `honest2_average_accuracy`，再看两个 BWT | 专家应 ACC=0.74、BWT=−0.48，对照同 ACC、BWT=−0.03；公式用最后一行减对角线 |
| 把 task-IL 的 90% 和 class-IL 的 30% 写进同一张表 | Mammoth 默认两套一起打 | 日志字段名含 class-il / task-il | 一张 $R$ 只对应一种设定 |

## 11. 前沿与改造

经典 CL 论文多数停在 GEM 这三项再加遗忘。2018 年 Díaz-Rodríguez 等人补上内存、缓冲占用、计算量，并提醒「平均准确率不是唯一指标」。2024 年 Wu 等人把同一套 ACC / BWT / FWT 搬到大模型。TRACE 另外加上跨阶段的 General Ability / Instruction Following / Safety 变化。你现在的协议覆盖经典三项和过程 / 在线两种时刻；还没覆盖：缓冲字节数、每任务额外参数、对齐阶段的安全回退。那些在第 06、10、11 课各加一列，不要提前假装已经量过。

和前沿的差距，一半是规模（TRACE 那种异构指令序列，本课只有 $5\times 5$ 玩具矩阵），一半是诚实（很多 LLM 持续学习报告仍只给最终平均）。本课把诚实先做掉。Zheng 等人把「成绩掉了」拆成任务对齐丢失与知识丢失；ForgetBench 把顺序编辑下的保持画成时间矩阵。那些列写进协议空位即可，本课实验不测。

动手改造（01-12 课精简版，选做）：

1. **max-forgetting 对照。** 在 CPU 实验的矩阵计算里增加 $\max_{t\ge i}R_{t,i}-R_{T,i}$。改动位置：`experiments/src/learn_cl_experiments/lessons/lesson_03.py`。预算：纯 CPU，分钟级。预期：对单调下降的 $R^{\text{last}}$，max 版与 Avalanche 版相同；若你插入一行「中间任务把旧任务暂时抬高再砸下去」，max 版更大。失败：两种遗忘在非单调矩阵上仍完全相等，说明 max 没在 $t\in[i,T]$ 上取。
2. **微平均 vs 宏平均。** 给五个任务不同测试集大小（例如 100、100、100、100、2000），用最后一行算按样本加权 ACC 和按任务平均 ACC。预算：纸面或十几行 numpy。预期：最后任务专家的加权 ACC 会被 2000 条样本拉到接近 0.99。失败：两种平均在不平衡测试集上仍相同，说明加权没接上样本数。
3. **Avalanche 自定义 PluginMetric。** 按 Evaluation 教程的 `PluginMetric` 骨架，做一个只在 `after_eval_exp` 把当前 $R$ 行打印成 CSV 的插件。预算：半天，CPU。预期：五个任务结束时得到合法 $5\times 5$ CSV，用本课公式复算与 `StreamBWT` 同号。失败：CSV 只有对角线，说明 eval 时没扫完整 test stream。

顺手复现映射：本课不列入课程五项正式复现。公式级对照 GEM §2 的 ACC / BWT / FWT；实现级对照 Avalanche evaluation 教程。第 08 课才会在同一协议下比较 A-GEM 与 GDumb。

## 12. 论文与延伸

每篇对应一个能用本课实验回答或明确答不了的问题。读完把答案写进 `NOTES.md`。

1. Lopez-Paz & Ranzato, 2017, *Gradient Episodic Memory for Continual Learning*, [arXiv:1706.08840](https://arxiv.org/abs/1706.08840)。
贡献：把持续学习写成连续数据上的训练协议，并定义 $R$、ACC、BWT、FWT。机制发明处，不是本课主阅读。
机制：每个任务结束后测全部 $T$ 个任务，填矩阵 $R$。ACC 是最后一行均值（式 (2)）。BWT 是最后一次减对角线、分母 $T-1$（式 (3)）。FWT 是上三角减随机基线 $\bar{b}$（式 (4)）。负 BWT 就是遗忘；任务 1 没有 BWT，任务 $T$ 没有 FWT。
和本课：CPU 实验按这些式子算 `specialist2` / `honest2` / `specialist5`。checks `average_accuracy_looks_high`、`bwt_strongly_negative`、`same_acc_different_bwt` 钉的就是 ACC 会说谎、同 ACC 不同 BWT。GEM 的梯度投影算法本课不跑。
阅读问题：对 `specialist2` 用手算式 (2)(3)，ACC 是否 $>0.70$、BWT 是否 $<-0.40$？能答。文中 MNIST / CIFAR 表本课答不了，因为实验不训网络。

2. Wu, Luo, Li, Pan, Vu & Haffari, 2024, *Continual Learning for Large Language Models: A Survey*, [arXiv:2402.01364](https://arxiv.org/abs/2402.01364)。
贡献：把 LLM 持续学习分成续预训练、指令微调、对齐三阶段，并用 Table 1 对照 RAG、模型编辑与持续学习各自覆盖哪些信息。
机制：§7.1 仍报平均准确率、FWT、BWT。BWT 写成 $A_{T,i}-A_{i,i}$，与 GEM 一致。该节 FWT 的下标写成 $A_{T,i}$，和 GEM 的 $R_{i-1,i}$ 不是同一格。Table 1 里 RAG 覆盖 Fact，Skills (Tool use)、Values、Preference 三格是叉。
和本课：5.4 节已写明 FWT 以 GEM 和 Avalanche 为准。CPU 的 `fwt_stays_at_chance_before_learning` 看的是上三角，不是最后一行。Table 1 的技能行本课矩阵没有技能任务，答不了。
阅读问题：把 `specialist5` 的最后一行误当成 FWT，和本课 check 用的上三角，两个数会不会一样？能答：最后一行均值是 ACC $=0.60$，上三角是 $0.50$。Wu 把 FWT 写到 $A_{T,i}$ 时，你会把遗忘矩阵的最后一行错当成前向迁移。

3. Shi et al., 2024, *Continual Learning of Large Language Models: A Comprehensive Survey*, [arXiv:2404.16789](https://arxiv.org/abs/2404.16789)。
贡献：把 LLM 持续学习拆成垂直连续（通用到专用）和水平连续（跨时间与领域），并单列评测协议与数据源。
机制：§2.2.3 给出四项：Overall Performance（到当前阶段为止的平均）、Forgetting（各任务最大掉幅再平均）、BWT（把遗忘取负）、FWT（未来任务相对随机初始化的增益）。§5 收集 CPT / DAP / CFT 能公开拿到的协议。OP 会把过程高分算进去，和只报最后一行的 GEM ACC 分母不同。
和本课：`specialist2` 与 `honest2` 同 ACC、BWT 差很大，正好说明只报 OP / ACC 不够。Shi 的「最大掉幅」遗忘本课 CPU 没算；协议空位可以记下，实验矩阵是单调下降，max 版与「第一次对最后一次」相同。
阅读问题：`same_acc_different_bwt` 为真时，若你只向别人报 Shi 的 OP（或 GEM 的 ACC），两张矩阵能被拆开吗？不能。必须把 BWT 或遗忘一起报。本课实验能答这一句；文中 DAP 表里谁做了后向迁移，答不了。

4. Zheng, Cai, Qiu & Ma, 2025, *Spurious Forgetting in Continual Learning of Language Models*, [arXiv:2501.13453](https://arxiv.org/abs/2501.13453)。
贡献：提出虚假遗忘：成绩掉了常常是任务对齐丢了，底层知识还在。
机制：合成 Biography 上，新任务前约 150 步会把旧任务成绩从接近满分打到约 10%，但用旧任务一半数据做恢复，知识几乎还在。他们把成绩写成「任务对齐 + 底层知识」，并把早期权重更新联系到近正交方向。不存旧数据时，冻住底层（含词嵌入）把顺序微调准确率从 11% 拉到 44%，其它对照最高约 22%。
和本课：`learning_accuracy_hides_forgetting` 造的就是「对角线接近满分、遗忘很大」的形状，和他们看到的成绩单同类。本课不训网络，看不见「再训十条就能救回来」，也测不到冻底层。
阅读问题：`specialist2` 的 LA 高、Forgetting 高，按 GEM 该判遗忘。若 Zheng 的恢复实验成立，这份成绩单还够不够单独给「知识没了」定罪？本课只能答：矩阵形状支持「成绩掉了」；「知识还在」本课实验答不了，因为没有恢复微调。

5. Harrington et al., 2026, *When Does Continual Learning Require Learning*, [arXiv:2607.07847](https://arxiv.org/abs/2607.07847)。
贡献：把持续学习定义成世界在变时提高能力，并用同一套与机制无关的顺序协议比较 prompt、SFT、RL 和上下文压缩。
机制：变化拆成空间（新领域）和时间（同一任务下事实漂移）。更新算子 $\mathcal{U}_k$ 可以改权重、改提示或写外存；每阶段后测全部阶段得到 $R_{i,j}$。HTML Table 1：GEPA / ACE 带走提示，SFT / GRPO 带走参数。摘要：prompt 新阶段拟合快、后面掉；蒸馏更稳但改旧事实慢；在线 RL 更会改知识，但对噪声奖励敏感。
和本课：`last_task_specialist_final_row` 就是「当前阶段满分、旧列掉到随机」，和「只拟合当前阶段」同一类谎。本课不跑 GEPA / GRPO，他们的分家族数字答不了。
阅读问题：若某方法只改 prompt、权重不动，本课这张 $R$ 还要不要写？要。$R$ 不问你改的是提示还是权重，只问每个时刻各任务多少分。本课实验能答形状；Qwen3-8B 上八种方法的表答不了。

6. Gu, Zhang & Wang, 2026, *ForgetBench: Benchmarking Forgetting Dynamics of Long-Term Parametric Memory in Language Models*, [arXiv:2607.26455](https://arxiv.org/abs/2607.26455)。
贡献：在连续知识编辑下量参数记忆的长期遗忘，而不是单次编辑成不成。
机制：concept-based QA 管孤立事实，scenario-based QA 管关系图。知识按时间流顺序写入，编辑年龄用「之后又改了多少次」定义。矩阵 $\mathcal{Z}[t,i]$ 记录第 $t$ 轮评测时第 $i$ 次编辑还对不对；对角平均是即时成功率，对过去条目平均是保持。现有编辑方法在长期保持和泛化之间做不到两边都好。
和本课：$R$ 的行是任务结束时刻，$\mathcal{Z}$ 的行是编辑轮次。`specialist5` 的最后一行能看见旧列掉到 $0.50$，相当于终局保持很差。时间衰减曲线本课没有，因为只有终局矩阵，没有按「年龄」切片。
阅读问题：把 `specialist5` 最后一行当成 ForgetBench 的 Retention，你会报什么？$0.50,0.50,0.50,0.50,1.00$ 的平均 $0.60$。这只是终局；他们要的衰减曲线本课实验答不了。

7. Chen, Zhu, Luo, Shen, Gao & Song, 2024, *CoIN: A Benchmark of Continual Instruction tuNing for Multimodel Large Language Model*, [arXiv:2403.08350](https://arxiv.org/abs/2403.08350)。
贡献：给多模态顺序指令微调做基准：10 个数据集、8 类任务，并把成绩拆成指令格式对不对、推理知识还在不在。
机制：LoRA 顺序微调，骨干冻住。Truth Alignment 按标准答案打分；Reasoning Capability 另用 LLM 只看关键信息。BWT 分母是 $T$（含最后一列差分为 0），不是 GEM 的 $T-1$；另报 Mean Average Accuracy，过程高分会抬总分。HTML：LLaVA 的 Truth Alignment BWT 为 $-32.62$；grounding 上推理分从 58% 到 52%，Truth Alignment 从 31.27% 到 0.00%。
和本课：CPU 矩阵能演示「MAA 比 ACC 好看」。本课没有多模态，也没有第二套 LLM 评分器，CoIN 那两列数字答不了。
阅读问题：对 `specialist5`，GEM ACC 是最后一行均值 $0.60$。CoIN 的 MAA 要不要把前几行对角线（全是 $1.00$）算进去？要。算进去会比 $0.60$ 高。本课能手算这一句；LLaVA 的 $-32.62$ 答不了。

8. Wang et al., 2023, *TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models*, [arXiv:2310.06762](https://arxiv.org/abs/2310.06762)。
贡献：给已对齐 LLM 做顺序学习基准，除目标任务外还量通用能力、指令遵循和安全的变化。机制发明处，不是本课主阅读。
机制：8 个数据集（领域、多语言、代码、数学），每任务抽样 5000 条训练、2000 条测试。目标序列仍报 OP 与 BWT，但 BWT 分母写成当前 $t$ 而不是 $t-1$。另外三列是 General Ability Delta、Instruction Following Delta、Safety Delta，相对训练前基线作差。摘要例子：llama2-chat 13B 在 gsm8k 上从 28.8% 掉到 2%。
和本课：GEM 的 ACC 只平均任务序列里的列，不含 gsm8k 这种「序列外能力」。`PROTOCOL.md` 若只写任务 ACC，就会漏掉 TRACE 那三列。本课玩具矩阵没有通用能力探针。
阅读问题：`average_accuracy_looks_high` 为真时，你能不能据此说「通用能力也还在」？不能。ACC 只看任务序列最后一行。gsm8k 那 28.8% 到 2% 本课实验答不了。

9. Qiao, Zhang, Tan, Qu, Ding & Xie, 2024, *Large Continual Instruction Assistant*, [arXiv:2410.10868](https://arxiv.org/abs/2410.10868)。
贡献：在顺序指令微调里用可调系数的 EMA 平衡新旧参数，并用指令语义相似度决定扩不扩训练参数。
机制：普通梯度更新会把参数推向新数据集。EMA 把新旧参数做加权和，固定权重又跟不上任务更换。他们对损失做泰勒展开，用梯度与已学参数自动算平衡系数 $\beta_t$。指令按语义分组，相似指令共用一套可训参数，组数大约是任务数的一半。代码仓库与 CoIN 基准相连，但论文本身改的是更新规则，不是再出一套 $R$ 公式。
和本课：本课不跑 EMA。`same_acc_different_bwt` 说明：只报最终 ACC，EMA 稳健更新和「最后任务专家」可能看起来一样好。要拆开必须看 BWT。
阅读问题：若某人用动态 EMA，最后一行都是 $0.90$，对角线也是 $0.90$，BWT 接近 0，这和 `honest2` 更像还是和 `specialist2` 更像？和 `honest2` 更像。本课能答形状；他们的 $\beta_t$ 公式本课实验答不了。

公式与日志核对仍用 Avalanche [Evaluation 教程](https://avalanche.continualai.org/from-zero-to-hero-tutorial/05_evaluation) 和 Mammoth [First steps](https://aimagelab.github.io/mammoth/getting_started/index.html)。它们不是论文清单。

现在尺子校准了。循环里「写完立刻测两件事、长期测第三件」终于有了共同的写法。下一课把同一套打分对准梁文峰举过的例子（转写未获 DeepSeek 确认）：每次把员工名录塞进上下文，算学会「小王是谁」了吗？打开 [第 04 课](04_not_just_rag.md)。



