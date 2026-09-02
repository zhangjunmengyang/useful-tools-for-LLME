---
id: 01_catastrophic_forgetting
title: "把遗忘跑出来"
summary: "同一个网络，先学任务 A 再学任务 B，A 的准确率为什么会塌？"
unit: forget
play_tools: []
checkpoints:
  - "画出任务-时间热力图，说清 naive fine-tune 在本课设定下掉到多少。"
  - "分清 task / domain / class incremental 三种设定。"
  - "留下固定种子、命令和曲线，作为后面 23 课的对照基线。"
---

# 第 01 课：把灾难性遗忘跑出来

> 类型：实战（机制实验；不列入课程正式复现表）<br>
> 建议周期：2-3 天<br>
> 硬件：CPU / Mac 即可；不需要 GPU<br>
> 锚定仓库：[ContinualAI/avalanche](https://github.com/ContinualAI/avalanche)（PyPI 包名 `avalanche-lib`，文档站 [avalanche.continualai.org](https://avalanche.continualai.org)）+ 课内手写两层 MLP<br>
> 产物：一张任务-时间热力图、三种增量设定对照笔记、naive fine-tune 在 Split MNIST 上任务 1 掉到的数字

## 1. 这一课做什么

一次训完就冻住的模型，只会干你写进这次训练数据里的事。持续学习要解决的是：模型在部署之后还能从新经验里学，并且旧本事不完全丢。这门 24 课是一个连续的大项目，对准公开讨论里那一级台阶：语言模型、思维链、Agent、持续学习。2026 年 5 月梁文峰闭门交流的转写把持续学习放在 Agent 之后的瓶颈上；转写于 2026-07 公开，DeepSeek 没有正式确认，课内只当路线判断，不当官方规格。

整门课始终在造同一个循环的某个零件：

```text
新经验进来
  先决定写到哪里（上下文 / 外挂记忆 / 快速权重 / 慢速权重）
  再决定怎么写（覆盖、追加、压缩、约束、正交、合并）
  写完立刻测两件事：新任务会了没、旧任务还在不在
  长期还要测第三件：还能不能继续学
```

六幕结构：

| 幕 | 课 | 一句话 |
|---|---|---|
| 第一幕：看见遗忘 | 01-04 | 把遗忘跑出来，学会评测，分清 RAG 和持续学习 |
| 第二幕：四类补丁 | 05-08 | 正则、回放、扩结构、约束梯度，加上 GDumb |
| 第三幕：大模型接龙 | 09-12 | 续预训练、顺序指令、O-LoRA、模型合并 |
| 第四幕：记忆、编辑、可塑性 | 13-16 | 外挂记忆、知识编辑、学不动了、何时必须改权重 |
| 第五幕：学习变成架构 | 17-20 | TTT、Titans、嵌套学习、SEAL 与 RL's Razor |
| 第六幕：在岗学习 | 21-24 | 技能库、经验时代、自迭代猜想、14 日上岗 |

你现在在第一幕第 1 课。上一课不存在：还没有基线，也没有抗遗忘方法。这一课加的零件是「病本身」：同一个网络，先学任务 A 再学任务 B，A 的准确率为什么会塌。方法是 naive fine-tune（朴素微调：新任务来了就接着用随机梯度下降往下训，不加回放、不加正则、不扩结构）。完成后你必须能拿出三样东西：一张任务-时间热力图、三种增量设定的对照笔记、以及你自己机器上任务 1 掉到的那个数字。没有这三样，后面 23 课所有「我的方法更稳」都没有对照。

梁文峰说的持续学习，不是 CIFAR 上把 10 类拆成 5 个任务那种作业。那种作业是前两幕的训练场，用来把遗忘、稳定性、可塑性、评测量清楚。真正要对准的目标是：一个已经会推理、会用工具的 Agent，在几个月的真实工作里积累技能、事实、偏好和流程。本课故意停在最小设定上。数字只对你写下的协议负责，不对 2017 年以后任何一篇论文的表格负责。课程正式复现五项从第 06 课才开始；本课标题里不许出现「复现某某论文」。

本课三档划分：

1. **实战**：用 Avalanche 的 `Naive` 在 `SplitMNIST` / `PermutedMNIST` 上跑通顺序训练，记录每个经验结束后的逐任务准确率。
2. **机制实验**：课内两层 MLP，固定种子，把「任务 1 随任务 2 训练步数下降」钉死；浏览器里用两个二维高斯分类任务拖动训练步数。
3. **只讲**：无。本课没有不能练的部分。

术语速查：

| 术语 | 一句人话 |
|---|---|
| 持续学习（continual learning） | 任务一个接一个来，模型既要学会新的，又不能把旧的整段抹掉 |
| 灾难性遗忘（catastrophic forgetting） | 先学 A 再学 B 之后，A 的表现突然垮掉；通常是新梯度把旧决策边界推走 |
| naive fine-tune | 新任务来了就接着 SGD，不加任何抗遗忘零件；本课的对照基线 |
| 经验 / experience | Avalanche 里一次「当前可训练的数据块」，对应文献里的一个任务或一个阶段 |
| Split MNIST | 把 10 个手写数字按类切开，例如 5 段、每段 2 类，按顺序学 |
| Permuted MNIST | 每段仍是 10 类，但像素被一套固定置换打乱；标签空间不变，输入分布变 |
| Task-IL / Domain-IL / Class-IL | 三种增量设定：测试时给不给任务编号、要不要自己推断现在是哪一类 |
| 任务-时间热力图 | 矩阵 $R_{i,j}$：学完第 $j$ 个任务之后，第 $i$ 个任务的准确率；行随时间变暗就是遗忘 |
| 决策边界 | 模型把输入空间切开的那条线或那张面；被新梯度推走之后，旧类就判错 |
| 随机种子（seed） | 控制所有随机性的起点数字；同种子同代码理应出同方向的结果 |

## 2. 问题

同一个多层感知机，先在任务 A 的数据上把损失压下去，再拿到任务 B 的数据上继续算梯度。A 的测试准确率往往不是慢慢掉，而是几个 epoch 之内塌到接近乱猜。1989 年 McCloskey 和 Cohen 把这件事叫做灾难性干扰（catastrophic interference），后来文献更常说灾难性遗忘。本课要回答的不是「有没有方法能缓解」，那是第 05 课以后的事。本课只回答三件更靠前的事：

1. 把这件事在你自己的机器上跑出来。浏览器里先看二维高斯的决策边界被抹掉；CPU 上用缩小版数字钉断言；再在 Avalanche 的 Split MNIST 上留下一张热力图。三层实验说的是同一句话：共享权重、顺序更新、没有旧数据，旧任务就会被新梯度覆盖。
2. 分清三种增量设定。同一串 Split MNIST 任务，测试时给不给任务编号、要不要从 10 类里做选择，难度完全不同。后面读论文时，有人报 99%、有人报 20%，经常是设定不同，不是方法差了一个数量级。
3. 建立这门课的对照习惯。从本课起，每个数字必须带着：命令、配置、种子、量法。只报一个最终平均准确率，会把「只会最后一件事」的方法夸成好方法，第 03 课会专门打假。本课至少要留下热力图，不要只留一个平均数。

一个要先划清的界限：本课的数字不构成对 McCloskey 1989、French 1999 或 van de Ven 2019/2022 的论文复现判断。那些文章用的网络、加法事实、迭代次数和你今天的两层 MLP 不是同一套协议。你要复现的是现象的方向：顺序学习之后，旧任务准确率显著下降。把冒烟档的 0.17 写成「论文结果」，是后面所有课都禁止的自欺。

French 1999 的综述还写过一句常被忽略的话：生物认知系统一般不会以这种灾难性的方式遗忘。这不是说人不会忘，是说人的忘是逐渐的、有结构的，不是学完 twos 加法表之后 ones 表整页空白。机器为什么会、人为什么相对不会，第 02 课才会把互补学习系统拿出来。本课先把机器这边的现场留下。

## 3. 准备

- 会用命令行和 Python 3.10+，装过 PyTorch（CPU 版即可）。不需要事先读过持续学习论文，用到的术语本课当场给定义。
- 一台 Mac 或 Linux，8GB 内存够用。MNIST 第一次下载大约几十 MB；Avalanche 会把数据放到它的默认缓存目录。
- 给 Avalanche 单独建一个虚拟环境。它依赖 PyTorch 和 torchvision，和你日常训练环境混装，版本冲突时排错成本很高。
- 先完成本课网页上的交互实验（遗忘滑块），再写 Python。二维高斯看懂了，Split MNIST 的热力图只是同一件事换成像素。
- 准备一个纯文本笔记，五行起步：日期与机器、命令、种子、设定（Task / Domain / Class-IL）、任务 1 在学完任务 2 之后的准确率。第 7 节 Step 6 给模板。

## 4. 学习目标

1. 用自己的话讲清灾难性遗忘：共享参数、顺序更新、没有旧数据时，新任务的梯度如何把旧决策边界推走。
2. 在白纸上画出 $2 \times 2$ 或 $5 \times 5$ 的任务-时间矩阵，指出哪一格是「刚学完」、哪一格是「后来忘了」，并写出 $R_{i,j}$ 的定义。
3. 对同一串 Split MNIST 任务，分别按 Task-IL、Domain-IL、Class-IL 写出测试时模型面对的选择题；能指出 Class-IL 为什么通常最难。
4. 独立跑通三层实验：浏览器遗忘滑块（先预测再运行）、`python3 run.py run 01`、Avalanche `Naive` + 课内两层 MLP。
5. 留下本课基线数字：naive 在你选定的 Split MNIST 协议上，任务 1 在学完后续任务之后掉到多少，并注明种子和设定。
6. 口头回答一个界限问题：容量加大能不能单独消灭遗忘？本课的小 MLP 已经够用来否定「只有容量不够才会忘」。

## 5. 原理

五个机制，每个按同一节奏：为什么需要（直觉）、怎么运转（机制）、精确定义（数学）、在源码哪里（代码）、怎么证明做对了（验证）。

### 5.1 灾难性遗忘：旧边界被新梯度整段推走

直觉。你刚学会区分 0 和 1。参数 $\theta$ 把像素空间划出一块「像 0」的区域。接着只给你 2 和 3，损失变成「把 2、3 分对」。梯度不管 0 和 1 还在不在，它只看当前 batch。几个 epoch 之后，原来那块「像 0」的区域被改去伺候 2 和 3。测试时再拿出 0，模型输出的最大类变成了 2 或 3。这就是灾难性遗忘：不是记忆慢慢淡，是决策边界被新目标函数改写。

McCloskey 和 Cohen 1989 年的现场更干净。他们先让反向传播网络学 17 道 ones 加法（$1+1$ 到 $9+1$，以及 $1+2$ 到 $1+9$），误差稳步下降。再学 17 道 twos 加法。每学一轮 twos，就回头测 ones。结果：往往只需一轮 twos，ones 的输出已经更像一个错误数字，而不像正确答案。两组都包含的 $1+2$ 和 $2+1$ 也在第一轮被冲乱。他们把这种现象叫做顺序学习问题（the sequential learning problem）。Ratcliff 1990 在识别记忆的联结主义模型里观察到同一类崩塌。French 1999 综述把它收成一个领域事实：分布式、重叠的表征加上梯度覆盖，就会灾难性遗忘。

机制。多层网络把知识写在共享权重里。任务 A 的输入 $x_A$ 和任务 B 的输入 $x_B$ 会激活重叠的隐藏单元。更新

$$
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}_B(\theta)
$$

只保证 $\mathcal{L}_B$ 下降，不保证 $\mathcal{L}_A$ 不升。若 $\nabla \mathcal{L}_B$ 在 $\theta$ 处与 $\nabla \mathcal{L}_A$ 方向冲突，走一步 B 就伤害 A。这和人的遗忘不同：人忘一张旧电话号码，通常不会把乘法表一起变成空白。

数学。记任务 $k$ 的经验风险为 $\mathcal{L}_k(\theta)=\mathbb{E}_{(x,y)\sim \mathcal{D}_k}[\ell(f_\theta(x),y)]$。顺序学习在只访问 $\mathcal{D}_t$ 时求 $\theta_t$，使得 $\mathcal{L}_t(\theta_t)$ 小。灾难性遗忘的操作定义：存在 $i<t$，使得 $\mathcal{L}_i(\theta_t)$ 显著大于 $\mathcal{L}_i(\theta_i)$。准确率版本更常用。令 $R_{i,j}$ 为学完第 $j$ 个任务之后、在第 $i$ 个任务测试集上的准确率，则任务 $i$ 在时刻 $T$ 的遗忘是

$$
f_i = R_{i,i} - R_{i,T}
$$

$f_i>0$ 表示退步。本课先把 $R$ 画成热力图；Average Accuracy、BWT、FWT 的完整协议在第 03 课。

代码。课内实验把这条更新写成普通的 `loss.backward(); optimizer.step()`，没有 replay buffer，没有 EWC 项。Avalanche 的 `Naive` 策略就是这个循环的框架版：每个 experience 上做若干 epoch 的监督训练，然后 `eval` 整条 test stream。对应导入在官方五分钟教程里写明：

```python
from avalanche.training.supervised import Naive
```

包装类还出现在 `from avalanche.training import Naive`。两套导入在当前文档里都能用；本课统一用 `avalanche.training.supervised`，和训练教程一致。

验证。任务 1 刚学完时准确率应明显高于随机（两分类约 0.5，十分类约 0.1）。任务 2 训完后，在 Class-IL 下任务 1 应显著下降。若任务 1 几乎不掉，先查是不是不小心把旧数据混进了 DataLoader，或测试时用了任务编号把输出限制在旧类上（那就已经是 Task-IL）。

### 5.2 共享权重为什么会互相覆盖

直觉。一张白板可以先写课文 A 再写课文 B。如果规定只能留一块板，写 B 就得擦 A。神经网络的权重就是这块板。容量更大时，板子更宽，理论上能并排写下 A 和 B；但优化过程没有「并排写」的指令，它只接到「把当前损失减小」。于是即使用 100 万参数去学 MNIST 这种小任务，顺序 SGD 仍然可能把旧方向覆盖掉。本课用很小的 MLP 就能看见这件事，用来否定「容量够大就不会忘」。容量不够会忘，容量够了用梯度接着训，照样会把旧边界抹掉。

机制。French 1999 把根源指向表征重叠（representational overlap）：不同模式共用同一组单元和权重。重叠是泛化的来源，也是干扰的来源。减少重叠（正交化、稀疏化、任务专用子网络）可以减轻遗忘，代价是迁移变差、容量被切开。本课不实现这些补丁，只确认默认的稠密 MLP 重叠足够大，干扰足够猛。

一个常见误会是把遗忘理解成「过拟合新任务所以旧任务差」。过拟合是在同一分布的训练/测试裂隙上谈的。这里旧任务的测试分布在学 B 的过程中从未作为训练目标出现，失败模式是分布外目标被覆盖，不是普通的 train/test gap。

数学。两任务在 $\theta$ 处的一阶干扰可用梯度内积估计：

$$
\nabla \mathcal{L}_A(\theta)^\top \nabla \mathcal{L}_B(\theta)
$$

内积为负时，沿 B 的下降方向会抬高 A。EWC 后面用 Fisher 对角线给每个坐标加弹簧，本质是不想让 $\theta$ 在对 A 敏感的方向上走远。本课不计算 Fisher，但你要能指着这个内积说：冲突发生在参数空间，不只发生在「模型太小」。

代码。手写 MLP 时把 backbone 和分类头放在同一个 `nn.Sequential` 里，不要给每个任务复制一份网络。Avalanche 的 `SimpleMLP` 默认就是一份共享权重加一个分类层。若改用 `MultiHeadClassifier` 并为每个任务接一个头，你已经在做 Task-IL 的架构补丁，数字会好看很多，那不再是 naive 基线。

验证。在任务 2 的某一步，用任务 1 的一小批数据算 $\nabla \mathcal{L}_A$，再和当前 $\nabla \mathcal{L}_B$ 做余弦。Class-IL 下余弦经常为负或接近零。若你几乎总得到强正相关，设定可能已经变成了「两个任务其实是同一件事」。

### 5.3 三种增量设定：测试时你到底在答哪道题

直觉。同样是「先学 0/1，再学 2/3」，考试可以完全不同。老师告诉你「这张卷子只考 0 和 1」，你只要在两个类里选。老师不告诉你现在考哪一单元，但每张卷子仍然是二选一（只是输入风格变了）。老师把十个数字混在一起，让你从 0 到 9 里选。三种考法对应 van de Ven 与 Tolias 2019 年提出、2022 年与 Tuytelaars 写成 Nature Machine Intelligence 标准表述的三种增量学习：Task-IL、Domain-IL、Class-IL。

机制。2019 年那篇 [arXiv:1904.07734](https://arxiv.org/abs/1904.07734) 的分类标准是：测试时是否提供任务身份（task identity）；若不提供，是否还必须推断任务身份。

| 设定 | 测试时给任务编号 | 模型必须做的事 | Split MNIST 上的考题 |
|---|---|---|---|
| Task-IL | 给 | 在已知任务的类里做选择 | 已知这是第 1 个任务，这张图是 0 还是 1 |
| Domain-IL | 不给 | 做和训练时结构相同的题，不必说出这是哪一任务 | 不知道现在是哪一组置换，但这张图是哪个数字 |
| Class-IL | 不给 | 在所有见过的类里做选择，等于要推断任务 | 从 0 到 9 里选出这是哪个数字 |

任何「分得清的任务序列」都可以按三种设定来考。Split MNIST 最常被拿来做 Task-IL（有时叫 multi-headed split MNIST）和 Class-IL（single-headed）。它也能做成 Domain-IL：每个任务仍是两类，但测试时不告诉你是哪两类、你必须在「当前这组的第一类还是第二类」之间选。Permuted MNIST 最自然的是 Domain-IL（标签空间始终是 10 类，变的是像素置换）；硬做成 Class-IL 则变成「既要认数字，又要认出是哪一套置换」。

论文里一个反复出现的结论必须记住：正则类方法（例如 EWC）在 Task-IL 上可以很好看，在 Class-IL 上常常接近崩溃；回放类方法在三种设定里都更有机会活下来。本课只跑 naive，用来体会设定本身的难度差。第 05 课会在同一协议上把 EWC 放进来，你就能看见「方法 × 设定」的交互，而不是把两种设定的数字横着比。

数学。Task-IL 的预测是 $p(y \mid x, t)$，其中 $t$ 已知，归一化只在任务 $t$ 的输出头上做。Class-IL 的预测是 $p(y \mid x)$，归一化在迄今所有类上做。这不是同一损失的两个超参数，是两个不同的条件分布。把 Task-IL 的 99% 和 Class-IL 的 20% 写进同一张「方法排名」，属于量法错误。

代码。Avalanche 的 `SplitMNIST(n_experiences=5, seed=1)` 默认是类增量切分。`return_task_id=True` 会给样本附上任务标签，这时才有条件做 Task-IL。`class_ids_from_zero_in_each_exp=True` 会把每个经验的标签从 0 重新编号，这是 Domain-IL 常用的标签布局（每个任务都是「类 0 vs 类 1」），千万不要和 Class-IL 的全局 0 到 9 标签混用。文档示例见 [Benchmarks 教程](https://avalanche.continualai.org/from-zero-to-hero-tutorial/03_benchmarks)。

验证。同一份权重，分别用「只评当前头」和「十类全开」打分，两个数字必须一起写。若你只报了 Task-IL，笔记里要标明，禁止在讨论 Class-IL 方法时引用它。

### 5.4 任务-时间热力图：遗忘长什么样

直觉。折线图可以画「平均准确率随任务序号变化」，但它把「刚学会的新任务很高、旧任务已经为零」平均成一个还过得去的数。热力图把时间放在横轴、任务放在纵轴，颜色表示准确率。理想的持续学习是下三角和对角线都亮：旧的还在，新的也会。naive 的典型样子是对角线亮、对角线以下迅速变暗。那一竖条变暗的过程，就是遗忘的时间展开。

机制。训练循环每结束一个 experience，就在**所有已经见过的**测试集上评一遍，不要只评当前任务。得到矩阵 $R\in\mathbb{R}^{T\times T}$，$R_{i,j}$ 仅在 $j\ge i$ 时有定义（还没学到的任务可以留空，或填学习前的零样本准确率，后者以后给 FWT 用）。把 $R$ 用同一色标画出来，禁止每个任务单独归一化，否则「从 0.95 掉到 0.40」和「从 0.50 掉到 0.45」会看起来差不多。

数学。对角线 $R_{i,i}$ 是学习准确率（learning accuracy）：刚学完时会不会。第 $j$ 列的平均 $\frac{1}{j}\sum_{i=1}^{j} R_{i,j}$ 是学到任务 $j$ 时的平均准确率。本课主看两件事：

- $R_{1,1}$ 是否明显高于随机，证明任务 1 曾经学会；
- $R_{1,T}$ 相对 $R_{1,1}$ 掉了多少，证明后来忘了。

第 03 课会把同一张矩阵变成 Forgetting、BWT、FWT。本课若过早引入一堆指标，注意力会从「看见遗忘」滑到「填表」。

代码。手写循环里，每训完一个任务就把 `eval_acc(model, loader_i)` 填进 `R[i, j]`。Avalanche 把这件事交给 `EvaluationPlugin` 和 `accuracy_metrics(experience=True, stream=True)`：每个 experience 评完会打出逐经验准确率，`forgetting_metrics(experience=True)` 会按插件自己的定义算遗忘。本课要求你**另外**把逐格数字抄进笔记或 `result.json`，不要只依赖终端滚动日志。日志默认不一定按 $R_{i,j}$ 的矩阵形状打印。

验证。合格热力图满足：对角线明显高于随机；Class-IL 的 naive 下三角明显暗于对角线。若整个矩阵都暗，是没学会，不是遗忘。若整个矩阵都亮，先检查测试集是不是只用了当前任务、或者标签被重映射成了 Domain-IL。

### 5.5 naive fine-tune 为什么必须先跑

直觉。后面每一类补丁都在 naive 上面加零件：EWC 加弹簧，回放加旧样本，扩结构加新格子，GEM 加梯度投影。不先测量 naive，你无法知道零件做了多少功，也无法知道自己的实现是不是其实还在跑 naive。GDumb（第 08 课）还会告诉你：有些「很强」的分数其实是缓冲里的 i.i.d. 重训。对照链的第一环必须是「什么都不加」。

机制。naive 的训练目标就是当前经验的监督损失。它合法、常用、也是 Avalanche 文档里的默认基线。官方五分钟教程把整个实验收成：建 `SplitMNIST` 或 `PermutedMNIST`，建 `SimpleMLP`，建 `Naive`，对 `train_stream` 逐经验 `train`，再 `eval` 整条 `test_stream`。你今天要做的事和这篇教程同一骨架，差别是本课强制导出热力图，并且要按 5.3 声明设定。

数学。naive 没有额外项：

$$
\mathcal{L}_{\text{naive}}(\theta)=\mathcal{L}_t(\theta)
$$

任何声称「抗遗忘」的方法，都必须在同一 $\mathcal{L}_t$ 之外写出多出来的项或数据。写不出那一项，就还是 naive。

代码。Avalanche 源码里，`Naive` 建立在 `SupervisedTemplate` 的训练/评测循环上，插件列表为空（或只有评测插件）。训练教程把这句话说得很干脆：多数持续学习策略，大致就是 naive（也叫 finetuning）加上一段对抗遗忘的行为。EWC、Replay 在框架里经常是插件，挂在同一条循环上。所以读 `Naive` 不是读一个玩具，是读后面所有策略共用的骨架。

验证。你的 naive 必须同时满足：新任务准确率上升；旧任务准确率下降。只满足前者，可能是数据泄漏或任务太像。只满足后者、新任务也不会，那是优化坏了，先修学习率，再谈遗忘。

## 6. 源码导读

先读 Avalanche 文档，再打开对应模块文件。官方当前安装命令和五分钟示例以 [How to Install](https://avalanche.continualai.org/getting-started/how-to-install) 与 [Learn Avalanche in 5 Minutes](https://avalanche.continualai.org/getting-started/learn-avalanche-in-5-minutes) 为准；训练循环与插件点以 [Training 教程](https://avalanche.continualai.org/from-zero-to-hero-tutorial/04_training) 为准。仓库根目录是 [ContinualAI/avalanche](https://github.com/ContinualAI/avalanche)。文档把库组织成五个模块：Benchmarks、Training、Evaluation、Models、Logging。

克隆不是本课必须，但读源码时建议对着已安装包的真实路径。用下面三条命令定位本课最关键的三个对象（每条命令单独跑）：

```bash
python3 -c "from avalanche.benchmarks.classic import SplitMNIST; import inspect; print(inspect.getfile(SplitMNIST))"
```

```bash
python3 -c "from avalanche.training.supervised import Naive; import inspect; print(inspect.getfile(Naive))"
```

```bash
python3 -c "from avalanche.models import SimpleMLP; import inspect; print(inspect.getfile(SimpleMLP))"
```

读文件时带着问题，不要按字母序扫目录：

| 对象 | 从哪导入 | 带着什么问题读 |
|---|---|---|
| `SplitMNIST` | `avalanche.benchmarks.classic` | `n_experiences=5` 时每个 experience 几类？`return_task_id` 和 `class_ids_from_zero_in_each_exp` 分别改变测试题的哪一部分？ |
| `PermutedMNIST` | `avalanche.benchmarks.classic` | 置换作用在像素上还是标签上？默认 `n_experiences` 是多少？ |
| `SimpleMLP` | `avalanche.models` | 输入是不是把 28×28 拉成 784？默认隐藏层多大、输出几个类？ |
| `MultiHeadClassifier` / `as_multitask` | `avalanche.models` | 多头是怎样按 task id 切 batch 的？这和 naive Class-IL 基线有什么关系？ |
| `Naive` | `avalanche.training.supervised` | `train` 和 `eval` 接受单个 experience 还是整条 stream？内部是否改 loss？ |
| `EvaluationPlugin` | `avalanche.training.plugins` | 指标在哪些回调点更新？`experience=True` 和 `stream=True` 各对应热力图的一格还是一列平均？ |
| `accuracy_metrics` / `forgetting_metrics` | `avalanche.evaluation.metrics` | forgetting 的定义是否等于本课的 $R_{i,i}-R_{i,T}$？不一致就要自己算 $R$ |
| `InteractiveLogger` | `avalanche.logging` | 终端打印的键名是什么？你要从日志里抄哪些键才能填矩阵 |

训练教程还给出了循环骨架，建议对照着读，直到能用自己的话复述下面这段（文档原结构，不是本课发明）：

```text
train
    before_training
    before_training_exp     # 每个 experience
        before_training_epoch
            before_training_iteration
                before_forward / after_forward
                before_backward / after_backward
            after_training_iteration
        after_training_epoch
    after_training_exp
    after_training
```

`Naive` 在这条循环上几乎什么都不改。EWC、Replay 以后作为插件挂上 `before_backward` 或换 DataLoader。你今天读懂这条循环，第 05、06 课就不用重新学框架。

课内机制实验的落点是 `experiments/src/learn_cl_experiments/lessons/lesson_01.py`。它必须只返回 `summary`、`metrics`、`checks` 三个字段；CLI 会补上 schema、runtime 和源码摘要。读它的时候问：种子写死了没有？任务 1 下降的阈值是多少？有没有在网上下载额外模型（不允许）？

## 7. 实验

三层都做。浏览器先建立手感，CPU 机制实验钉断言，Avalanche 对照确认同一现象在公认框架里也在。每一步先写预期，再跑，再对照。

### Step 0: 打开网页，先预测再拖滑块

本课浏览器实验是**遗忘滑块**：两个二维高斯分类任务共享一个线性或浅层分类器。任务 1 先学到分得开，然后拖动任务 2 的训练步数，看任务 1 的决策边界被抹掉。实验嵌在本课页面的交互实验区（页面锚点 `#interactive-lab`，课程元数据里的 lab id 是 `lab-01-forgetting-slider`）。本地预览：

```bash
cd web
```

```bash
npm run dev
```

浏览器打开本课，不要一上来就按运行。先在预测控件里回答：

> 任务 2 大概训多少步之后，任务 1 会掉到接近随机（两类约 50%）？

可选档通常是「几乎立刻」「几十步」「要到任务 2 完全收敛」。选完再运行。改滑块会作废上次运行，必须重新预测。过关条件：预测与仿真落在同一量级，并且你能指着图说清「哪一条边界属于任务 1、它往哪边倒」。

这个二维问题故意做得极小，所以塌得很快。它要证明的是机制，不是 MNIST 上的步数。你的预测若是「要等任务 2 收敛才会忘」，运行后多半会错：冲突梯度在早期就已经抬走旧边界。把预测对错写进笔记，这是本课真正的第一笔数据。

### Step 1: 跑课内 CPU 机制实验

```bash
cd experiments
```

```bash
python3 run.py run 01
```

预期：命令打印 `[PASS]`，写出 `artifacts/lesson01/result.json`，四个 `checks` 全真。这层钉的是机制，不是论文分数：二维线性分类器，对不上 Split MNIST 论文表。本机一次运行（Python 3.13.13，seed=1），任务 A 从 0.938 降到 0.608（掉 0.329），任务 B 仍 0.979。换机器会变，方向不应变。

真实 `checks` 键名：

- `task1_learned_above_0_90`：任务 A 先学到 >0.90（本机 `acc_task1_after_task1`=0.9375）；
- `task2_learned_above_0_90`：任务 B 也能 >0.90（本机 `acc_task2_after_task2`=0.979167）；
- `task1_drop_exceeds_0_25`：接着训 B 后 A 下降超过 0.25（本机 `task1_drop`=0.329167）；
- `task1_after_task2_below_0_70`：训完 B 后 A 落到 0.70 以下（本机 `acc_task1_after_task2`=0.608333）。

打开结果文件时核对：`lesson_id` 为 `"01"`，上面四个键都是 `true`，`metrics.seed` 为 1。不要手改 JSON 去凑绿。

### Step 2: 手写两层 MLP，在 Split MNIST 上看任务 1 往下掉

这一步不经过 Avalanche，对照公式。固定 `seed=42`。协议用**两个经验的 Class-IL**：任务 1 为数字 0 和 1，任务 2 为数字 2 和 3，输出层 4 维（或 10 维，多余类不用），测试时在任务各自的真实标签上算准确率，**不要**在测试时把输出限制到当前任务的两类，那会偷偷变成 Task-IL。

网络：`Linear(784, 256)`、ReLU、`Linear(256, 4)`。优化器 SGD，学习率 `0.01`，batch `128`。任务 1 训 2 个 epoch，存下 $\theta$ 在任务 1 测试集上的准确率 $R_{1,1}$。任务 2 开始后，每隔固定步数（例如每 50 step）测一次任务 1 和任务 2。把任务 1 的曲线画出来：横轴是任务 2 的训练步，纵轴是准确率。

最小可跑骨架（自行补上 MNIST 下载与 DataLoader；保持种子和 Class-IL 评测口径）：

```python
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1)),
])
mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

def by_labels(ds, labels):
    idx = [i for i, t in enumerate(ds.targets) if int(t) in labels]
    return Subset(ds, idx)

train_t1 = DataLoader(by_labels(mnist, {0, 1}), batch_size=128, shuffle=True)
train_t2 = DataLoader(by_labels(mnist, {2, 3}), batch_size=128, shuffle=True)
test_t1 = DataLoader(by_labels(mnist_test, {0, 1}), batch_size=256)
test_t2 = DataLoader(by_labels(mnist_test, {2, 3}), batch_size=256)

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 4),
)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

def acc(loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).argmax(dim=1)
            correct += int((pred == y).sum())
            total += int(y.numel())
    model.train()
    return correct / total

def run_epoch(loader):
    for x, y in loader:
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()

for _ in range(2):
    run_epoch(train_t1)
r11 = acc(test_t1)
curve = []
step = 0
for epoch in range(2):
    for x, y in train_t2:
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
        step += 1
        if step % 50 == 0:
            curve.append((step, acc(test_t1), acc(test_t2)))
print("R11", r11, "curve", curve[-1] if curve else None)
```

预期：$R_{1,1}$ 应在 0.9 以上（0 vs 1 对两层 MLP 很简单）。任务 2 训完后，任务 1 的 Class-IL 准确率应明显掉到 0.5 以下，常见情况是掉到接近把 0/1 全部判成 2 或 3。把最后一行 `(step, acc_t1, acc_t2)` 抄进笔记。若任务 1 不掉，检查是不是评测时做了 `logits[:, :2]` 这种按任务切片。

把 `curve` 画成折线，再把两个任务、两个时刻收成 $2\times 2$ 热力图：

```text
          after T1    after T2
task 1      R11          R12
task 2      (empty)       R22
```

`text` 围栏里不要写箭头。`R12` 相对 `R11` 的落差就是本课要的遗忘幅度。

### Step 3: Avalanche naive 对照（Split MNIST）

独立虚拟环境里安装与官方文档一致的包名。先装与你环境匹配的 PyTorch 和 torchvision，再：

```bash
pip install avalanche-lib==0.6
```

`0.6` 是 From Zero to Hero 教程当前页写明的版本。若该版本在你的 Python 上装不上，退回文档首页的未钉版本，并在笔记里写明实际版本：

```bash
pip install avalanche-lib
```

```bash
pip show avalanche-lib
```

下面脚本对齐五分钟教程：`SplitMNIST(n_experiences=5)`、`SimpleMLP`、`Naive`、逐经验训练、每次训完评整条 test stream。本课把 epoch 缩到 1、batch 放到 500，CPU 上几分钟内应能跑完。把脚本存成 `naive_split_mnist.py` 再运行。

```python
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger
from avalanche.models import SimpleMLP
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive

seed = 42
torch.manual_seed(seed)
benchmark = SplitMNIST(n_experiences=5, seed=seed)
model = SimpleMLP(num_classes=benchmark.n_classes)
eval_plugin = EvaluationPlugin(
    accuracy_metrics(epoch=True, experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    loggers=[InteractiveLogger()],
)
cl_strategy = Naive(
    model,
    SGD(model.parameters(), lr=0.001, momentum=0.9),
    CrossEntropyLoss(),
    train_mb_size=500,
    train_epochs=1,
    eval_mb_size=100,
    evaluator=eval_plugin,
)
results = []
for experience in benchmark.train_stream:
    print("experience", experience.current_experience,
          "classes", experience.classes_in_this_experience)
    cl_strategy.train(experience)
    results.append(cl_strategy.eval(benchmark.test_stream))
```

```bash
python3 naive_split_mnist.py
```

预期：五个 experience 各含两个类（具体配对由 `seed=42` 决定，以打印的 `classes` 为准）。这是 Class-IL 口径：`SimpleMLP` 单头、10 类输出，评测时不提供任务编号。学完第 5 个经验后，早期经验的 experience accuracy 应远低于刚学完时。把每个 `eval` 返回字典里与 `Top1_Acc_Exp` / forgetting 相关的键抄下来，手工排成 $5\times 5$ 能填的部分。第一行（任务 1 在五个时刻的准确率）就是本课交付的遗忘曲线。

van de Ven 2019 在 split MNIST 上把 None（标准顺序训练）当作下界，并报告正则方法在 Class-IL 上接近失败。你不需要对齐他们的绝对数字（网络宽度、迭代次数不同），只需要对齐方向：naive 在 Class-IL 的旧任务上会垮。

### Step 4: Permuted MNIST 对照，体会设定差

同一套 `Naive` + `SimpleMLP`，把 benchmark 换成：

```python
from avalanche.benchmarks.classic import PermutedMNIST
benchmark = PermutedMNIST(n_experiences=3, seed=42)
```

五分钟教程用的就是 `PermutedMNIST(n_experiences=3)`。每个经验仍是 10 类，变的是像素置换。这更接近 Domain-IL：标签空间不变，输入分布变。预期：遗忘仍然发生，但往往没有 Split MNIST 的 Class-IL 那么惨，因为输出头的十个类一直都在，模型不必「把旧类的 logit 关掉」。把两种 benchmark 的旧任务保持写在同一张对照笔记里，这就是 5.3 节的实验版。

若时间只够跑一个 Avalanche 实验，保留 Split MNIST Class-IL。Permuted MNIST 是加深理解用的第二根柱子，不是验收的唯一数字。

### Step 5: 三种设定对照笔记（同一串任务，三道考题）

不必把 Task-IL 的多头模型完整训到收敛。用 Step 2 训完的 Class-IL 权重，**只改评测**：

1. Class-IL：`pred = logits.argmax(dim=1)`，标签是全局 0 到 3。这是你已经跑过的。
2. Task-IL：评任务 1 时只在 `logits[:, 0:2]` 里取 argmax，评任务 2 时只在 `logits[:, 2:4]` 里取。等于考试时告诉你现在是哪一单元。
3. Domain-IL 近似：把任务 2 的标签映射成 0/1（2 变成 0、3 变成 1），输出也只用前两个 logit。这是「结构相同、输入分布不同」的最小演示，不是 van de Ven 论文的完整 Domain-IL 协议。

预期：同一份已经被任务 2 改过的权重，Task-IL 口径下任务 1 通常明显高于 Class-IL。笔记里写三行数字，并写一句：差距来自测试题，不是来自又训了一次。这能解释你以后在论文里看到的 99% 对 20%。

### Step 6: 留下基线

在实验目录放 `NOTES.md`：

```text
日期与机器
avalanche-lib 版本（pip show）与 PyTorch 版本
种子 42
Split MNIST：n_experiences、Class-IL
命令全文
R11、R12（任务 1 在学完任务 2 或学完全序列之后）
热力图文件名
浏览器滑块的预测与对错
run.py 01 的 PASS/FAIL 与 checks 摘要
```

三个月后只看这份笔记，应能复述：你在哪种设定下、用哪条命令、得到任务 1 掉到了多少。能，本课的对照就算立住了。

## 8. 配置与预算

| 档位 | 数据与模型 | 时间（CPU，参考） | 用途 |
|---|---|---|---|
| 浏览器 | 两个二维高斯，线性或浅层分类器 | 5 分钟 | 先预测，看见边界被推走 |
| 课内机制 | `python3 run.py run 01`，禁止下载大模型 | 通常 < 30 秒 | 断言：任务 1 显著下降 |
| 手写 MLP | MNIST 子集，0/1 然后 2/3，两层 256 | 数分钟 | 对照公式，画步级曲线 |
| Avalanche 冒烟 | `SplitMNIST(5)`，`SimpleMLP`，1 epoch，batch 500 | 数分钟到十几分钟 | 公认框架里的 naive 基线 |
| Avalanche 对照 | 再跑 `PermutedMNIST(3)`，同样 Naive | 再加数分钟 | 体会 Domain-IL 与 Class-IL 的差 |

硬件：全程 CPU。有 GPU 也不会改变本课结论，不必为 MNIST 抢卡。磁盘：MNIST 加 Avalanche 缓存约数百 MB。内存 8GB 够用。

超参数不要在本课里扫。学习率、epoch、宽度只为「任务 1 先学会、再被冲掉」。若 $R_{1,1}$ 已经接近随机，先加 epoch 或学习率，不要立刻宣布「遗忘不明显」。若 $R_{1,1}$ 很高而 $R_{1,T}$ 几乎不掉，先查评测口径，再查是否把全部数字一次性放进了训练。

## 9. 验收

- [ ] 浏览器遗忘滑块：运行前写过预测；能指着图说出任务 1 的边界往哪边倒。
- [ ] `python3 run.py run 01`：`checks` 全真（`task1_learned_above_0_90`、`task2_learned_above_0_90`、`task1_drop_exceeds_0_25`、`task1_after_task2_below_0_70`）。
- [ ] 手写 MLP：固定 `seed=42`；给出 $R_{1,1}$ 和任务 2 之后的任务 1 准确率；Class-IL 下后者明显更低。
- [ ] Avalanche `Naive` + `SplitMNIST`：五段经验的逐经验准确率抄成热力图或至少抄出第一行（任务 1 随时间）。
- [ ] 三种增量设定对照笔记：同一串任务、三道考题、三个数字，并写明差距来自测试协议。
- [ ] `NOTES.md` 含命令、版本、种子、设定、任务 1 掉到的数字。
- [ ] 能口头回答：加大 MLP 宽度是不是本课消灭遗忘的正路？（不是。本课要你看见覆盖，不要求你把容量加到遗忘消失。）
- [ ] 能指出 naive 在代码里没有额外损失项；任何多出来的项都属于后面的课。

本课基线数字以你笔记里的 $R_{1,T}$ 为准。不同机器、不同 Avalanche 小版本会有抖动，方向必须一致：Class-IL naive 下，任务 1 在后续任务之后显著下降。

## 10. 排错

| 症状 | 最可能的原因 | 怎么确认 | 怎么修 |
|---|---|---|---|
| `pip install avalanche-lib` 失败 | PyTorch / torchvision 未装或版本过新过旧 | 看报错里的依赖名 | 先按 PyTorch 官网装 CPU 轮子，再装 `avalanche-lib`；必要时钉教程写的 `==0.6` |
| MNIST 下载超时 | 网络或数据源 | 数据目录是否空 | 换镜像、手动放到 torchvision 默认路径，或设 `download=True` 后重试 |
| 任务 1 从不下降 | 评测时按任务切了 logit，或训练时混入了旧类 | 打印 `pred.unique()` 和 `y.unique()` | 改回 Class-IL 全类 argmax；检查 DataLoader 的标签过滤 |
| 任务 1、任务 2 都接近随机 | 学习率过大过小，或输入没拉平 | 看训练 loss 是否下降 | 先单任务过拟合 0/1；确认 `view(-1)` 或 `SimpleMLP` 吃到 784 维 |
| Avalanche 评测数字对不上手写 | 设定不同：多头、标签重映射、test stream 含未见类 | 对比 `return_task_id`、`n_classes` | 两边用同一设定再比；不要把 Task-IL 日志拿来当 Class-IL |
| `forgetting_metrics` 和自己算的 $f_i$ 不同 | 插件定义与 $R_{i,i}-R_{i,T}$ 不完全相同 | 读指标文档并打印 $R$ | 本课以自己算的 $R$ 为准，第 03 课再对齐官方指标 |
| `task1_drop_exceeds_0_25` 或 `task1_after_task2_below_0_70` 为假 | 任务 B 没覆盖旧边界，或评测口径错了 | 看 `task1_drop` 与 `acc_task1_after_task2` | 对照 `summary`：下降应超过 0.25 并落到 0.70 以下；不要手改 `result.json` |
| Mac 上 DataLoader 卡死 | `num_workers>0` 的 spawn 问题 | 卡住发生在第一个 batch | 设 `num_workers=0`（五分钟教程里的 4 在笔记本上可改 0） |
| 热力图全亮 | 每次 eval 只测了当前 experience | 日志里只有一个 experience 的 acc | `eval(benchmark.test_stream)` 传整条流，不要只传 `test_stream[j]` |
| 热力图全暗 | 从未学会 | $R_{i,i}$ 是否高于随机 | 加 epoch；先验证单任务 MNIST 能到 90%+ |

## 11. 前沿与改造

你今天跑的 naive，2017 年以后每一类方法都在它上面加零件。EWC / SI / MAS 给重要权重加弹簧（第 05 课）；Experience Replay、iCaRL、DER++ 把旧样本或旧 logits 带回 batch（第 06 课）；Progressive Nets、PackNet、L2P 把新知识写到新格子或 prompt 里（第 07 课）；GEM / A-GEM 把更新投影到不增加旧损失的半平面（第 08 课）。大模型阶段会看到同一现象换皮：顺序指令微调把前一个技能冲掉（第 10 课），O-LoRA 要低秩方向近似正交（第 11 课）。测试时学习则把「写到哪里」从训练阶段挪到推理内环（第 17 到 20 课）。本课的热力图不会作废：那些系统改的是怎么写，不是「写完必须测旧任务」这条纪律。

近两年把同一现场换到语言模型，读法见 §12：顺序指令会冲通用知识（Luo 等），成绩掉了也可能只是任务对齐断了（Zheng 等），连续往权重里写事实则可能只改了问法（O'Neill 等）。本课热力图仍是最小对照。

缩小版和前沿的差距，一半是规模（类数、模型、任务边界是否干净），一半是设定（有没有 task id、能不能存旧数据）。van de Ven 2019 已经用 MNIST 级别的实验说明：Class-IL 上正则方法可以全面失败，回放却仍可能超过 90%。你还没有跑 EWC，所以本课不要把这句话当成你验证过的结论；把它当成第 05、06 课的预测。

动手改造清单（选做，每个都写清预算和失败标准）：

1. **加大宽度对照。** 把隐藏层从 256 改到 1024，其余不动。预算：CPU 十几分钟。预期：任务 1 仍会掉，幅度未必按宽度等比例变小。失败判据：宽度加大后任务 1 几乎不掉，同时你发现 DataLoader 混入了旧类。那就不是容量救了遗忘，是实验坏了。
2. **Task-IL 多头。** 用 `return_task_id=True` 的 `SplitMNIST`，模型换成文档里的 `MultiHeadClassifier` 或 `as_multitask`。预算：一个晚上。预期：旧任务保持明显高于单头 Class-IL。失败判据：多头仍然崩到随机，说明评测时没有把 task label 喂给模型。这是架构补丁，不是 naive 基线，笔记里必须分栏。
3. **步级热力图。** 在任务 2 的训练中每 N 步评一次任务 1，而不是每个 epoch 评一次。预算：评测变密，时间大约翻倍。预期：下降发生在任务 2 的早期步，而不是等收敛之后。失败判据：曲线先稳住很久再突然掉，先查学习率是否小到几乎没更新。
4. **顺手复现映射。** 本课不进入正式复现表。若你想提前对照文献，只允许做方向核对：Goodfellow 等 2013 的 Permuted MNIST（[arXiv:1312.6211](https://arxiv.org/abs/1312.6211)）报告顺序学习会忘；Kirkpatrick 等把 EWC 画在同一类曲线上。你的冒烟数字无权写入「复现成功」。

## 12. 论文与延伸

每篇对应一个能用本课实验回答或明确答不了的问题。读完把答案写进 `NOTES.md`。主阅读是 2024–2026；谱系只留本课实验真用到的两篇。

1. McCloskey, M. & Cohen, N. J., 1989, *Catastrophic interference in connectionist networks: The sequential learning problem*, [DOI 10.1016/S0079-7421(08)60536-8](https://doi.org/10.1016/S0079-7421%2808%2960536-8)。
贡献：用 ones 加法表接 twos 加法表，展示一轮新学习就能把旧输出冲到更像错误答案。机制发明处，不是本课主阅读。
机制：共享权重、顺序更新、不回放旧题。损失只看当前任务，旧决策边界被新梯度整段改写。评测是每学一轮 twos 就回头测 ones。
和本课：`python3 run.py run 01` 的 `task1_drop_exceeds_0_25` 与 `task1_after_task2_below_0_70` 就是同一现象换成二维线性分类器。论文里的加法事实表本课答不了。
阅读问题：你的任务 A 先到 0.90 以上、再被任务 B 拉到 0.70 以下，对应论文里 ones 表被冲乱的哪一步？若 `task1_drop` 不到 0.25，是评测口径错了还是两个任务太像？

2. van de Ven, G. M. & Tolias, A. S., 2019, *Three scenarios for continual learning*, [arXiv:1904.07734](https://arxiv.org/abs/1904.07734)。
贡献：按测试时是否给任务身份、是否必须推断任务身份，分成 Task-IL / Domain-IL / Class-IL。机制发明处，不是本课主阅读。
机制：同一串任务可以按三种设定来考。摘要写明：必须推断任务身份时（Class-IL），正则方法如 EWC 失败，回放更像刚需。评测用 split 与 permuted MNIST。
和本课：Step 5 同一份权重三道考题就是这篇的表。CPU 实验是两类输出、没有任务编号，更接近 Domain-IL 玩具；论文里「EWC 在 Class-IL 失败」本课答不了，差在还没跑 EWC。
阅读问题：Step 5 里 Task-IL 切片后任务 1 是否明显高于 Class-IL 全类 argmax？若几乎一样，先查你有没有真的按任务切了 logit。

3. Wu, T. et al., 2024, *Continual Learning for Large Language Models: A Survey*, [arXiv:2402.01364](https://arxiv.org/abs/2402.01364)。
贡献：按续预训练、指令微调、对齐给 LLM 持续学习分阶段，并对照检索增强与模型编辑。
机制：改的是地图和评测口径，不改某条损失。摘要强调 LLM 不能频繁重训，更新必须和检索增强、编辑分开谈。
和本课：本课 naive 对应他们说的「接着微调、旧域掉下去」。三阶段划分、RAG 对照本课答不了，第 04、09、10 课才碰到。
阅读问题：本课热力图量的是哪一阶段的遗忘？若你只跑了 Split MNIST naive，它对应综述的哪一块，哪一块你必须写「本课实验答不了」？

4. Shi, H. et al., 2024, *Continual Learning of Large Language Models: A Comprehensive Survey*, [arXiv:2404.16789](https://arxiv.org/abs/2404.16789)。
贡献：把 LLM 持续学习拆成纵向（从通用到专用）和横向（跨时间、跨域），再按 CPT / DAP / CFT 三段写。
机制：改的是分类框架和评测协议清单，不是一条新损失。水平连续对应「世界在变还接着训」，垂直连续对应「通用模型往下专」。
和本课：你的任务 1 接任务 2 是横向、同一模型上的 CFT 玩具。CPT 与 DAP 本课答不了。
阅读问题：把本课 Class-IL naive 放进他们的水平连续，还缺哪一项评测（旧任务、新任务、还是还能不能继续学）？用你的 $R_{1,1}$ 和 $R_{1,2}$ 指出缺的那一项。

5. Luo, Y. et al., 2025, *An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning*, [arXiv:2308.08747](https://arxiv.org/abs/2308.08747)（v5，2025-01-05；2023 投稿）。
贡献：在 1b 到 7b 的生成式模型上，用领域知识、推理、阅读理解量顺序指令微调时的通用知识遗忘。
机制：五个指令任务按固定顺序全参微调，旧任务数据不可见。遗忘指标是相对初始模型的相对下降。摘要写：该尺度内模型越大遗忘越重；decoder-only 的 BLOOMZ 比 encoder-decoder 的 mT0 忘得少；通用指令微调能减轻后续遗忘。
和本课：`task1_drop_exceeds_0_25` 对应「接着训新指令、旧通用能力掉」。尺度效应、架构差、混通用指令本课答不了。
阅读问题：若你做了改造清单第 1 条（隐藏层 256 改 1024），任务 1 还会不会掉过 0.25？若没做，标准 CPU 实验已经用很小的分类器看见覆盖，够用来否定「只有容量不够才会忘」。论文里 1b 到 7b 的尺度结论本课实验答不了。

6. Zheng, J., Cai, X., Qiu, S. & Ma, Q., 2025, *Spurious Forgetting in Continual Learning of Language Models*, [arXiv:2501.13453](https://arxiv.org/abs/2501.13453)。
贡献：提出虚假遗忘：成绩掉了常常是任务对齐断了，底层知识还在。
机制：合成传记数据上，新任务前约 150 步会撤掉旧对齐。理论连到权重的正交更新。补丁是冻底层；文中写顺序微调从约 11% 提到约 44%，其他技术最高约 22%。
和本课：冻底层对应第 02 课冻骨干，本课 naive 只看见成绩掉。本课没有「用少量旧格式例子把旧分捞回来」的恢复协议，答不了「知识还在」。
阅读问题：若你只看 $R_{1,2}$ 低于 0.70，能不能区分「边界被擦掉」和「对齐断了、知识还在」？用本课现有评测回答；若分不开，写「本课实验答不了，差在没有恢复训练」。

7. Yang, S. et al., 2024, *Is Parameter Collision Hindering Continual Learning in LLMs?*, [arXiv:2410.10179](https://arxiv.org/abs/2410.10179)。
贡献：论证防碰撞比强制正交更关键，并提出对 LoRA 增量加 $\ell_1$ 的 N-LoRA。
机制：改损失：任务损失加上 $\lambda\|\Delta W_i\|_1$，旧 LoRA 冻住。摘要写相对当时 SOTA：成绩 +2.9，任务正交约 4.1 倍，参数碰撞约 58.1 分之一。
和本课：本课全参稠密更新，碰撞是默认状态。N-LoRA、O-LoRA 本课答不了，第 11 课才做正交 LoRA。
阅读问题：本课两层 MLP 的共享权重，属于「正交但仍碰撞」还是「根本没做子空间隔离」？用 Step 2 是否切分参数来答。

8. Harrington, A. et al., 2026, *When Does Continual Learning Require Learning*, [arXiv:2607.07847](https://arxiv.org/abs/2607.07847)。
贡献：把环境变化拆成空间（新域）和时间（同一任务下事实漂移），并问何时必须改权重。
机制：改评测：把现成 LLM 基准改成序列，同一协议比较提示法（GEPA、ACE）、监督（SFT、SDFT）、强化学习（GRPO、SDPO）和上下文压缩（Cartridges、原地 TTT）。摘要：提示法适应当前阶段快、以后掉；蒸馏稳但改旧事实难；压缩提效率、不太提高学新任务；在线 RL 更新知识最有效，但怕噪声奖励。
和本课：本课没有提示外挂，只有改权重的 naive。空间/时间两轴、提示对权重本课答不了。
阅读问题：本课任务 2 是新域还是同一任务下的漂移？用标签空间有没有变来答。论文里「何时必须改权重」本课实验答不了。

9. O'Neill, C., 2026, *Can a Language Model Learn Facts Continually in Its Weights?*, [arXiv:2607.11020](https://arxiv.org/abs/2607.11020)。
贡献：在 Qwen3 上连续写入虚构事实，问权重通道能不能累积、后来的写入会不会让旧事实问不到。
机制：每条事实写进权重，再用五种留出问题测。摘要：干陈述把「会背到会用」的缺口从 27.4 收到 5.4 需要多样复述；20 次顺序写入后，干陈述事实准确率 1%，宽数据事实 46%；干陈述设定下 70% 的错答含最新写入的事实；把已忘的学习型事实塞回提示可回到 77-80%。
和本课：顺序写入对应任务 1 接任务 2。事实还在、只是问法被改道，本课 argmax 准确率答不了。
阅读问题：本课任务 1 掉到 0.70 以下之后，你能量的是准确率塌了。`result.json` 不给权重方向，也没有把旧事实塞回提示的恢复臂。论文的 log-prob 保留和 77-80% 提示恢复本课实验答不了。

10. Chen, H., Sun, Z., Ye, H., Li, K. & Lin, X., 2026, *Beyond Static Models: An Evolving Framework for Continual Learning in Large Language Models across Training Stages*, [arXiv:2603.12658](https://arxiv.org/abs/2603.12658)。
贡献：按续预训练、持续微调、持续对齐写 LLM 持续学习，并在回放 / 正则 / 扩结构下面再按遗忘机制细分。
机制：改的是框架和评测清单（遗忘率、知识迁移、新兴基准），不是一条新更新式。
和本课：naive 是他们说的静态预训练之后、不做任何抗遗忘项的下界。三阶段对照本课答不了。
阅读问题：本课 $R_{i,j}$ 热力图对应他们哪一项指标？Average Accuracy 要到第 03 课才正式算，本课只要求你能指出 $R_{1,T}$ 相对 $R_{1,1}$ 掉了多少。

现在系统还不会抗遗忘。它只会在顺序更新下把旧任务忘掉，并且你能量这件事。下一课要把同一份 Split MNIST 放进稳定性-可塑性平面：冻骨干、降学习率、混一点旧数据，四个点各在哪；以及海马-新皮层类比在你的 MLP 上何处失效。去 [第 02 课](02_stability_plasticity.md)。



