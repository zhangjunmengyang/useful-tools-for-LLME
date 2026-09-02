---
id: 07_architecture_prompts
title: "不改旧权重就再长一块"
summary: "扩网络、冻旧柱、加 adapter、加 prompt，新知识写在哪？"
unit: toolkit
play_tools: []
checkpoints:
  - "PackNet 掩码图。"
  - "prompt pool 是否按任务分开的探针。"
  - "和「每个任务一个头 + 冻骨干」的对照。"
---

# 第 07 课：不改旧权重，就再长一块

> 类型：实战（Mammoth DualPrompt / L2P）+ 机制实验（PackNet 掩码）<br>
> 建议周期：3-4 天<br>
> 硬件：PackNet 与冻骨干对照用 CPU / Mac 即可；prompt 方法需要单卡，建议 8-24GB，下载 ViT-B/16 权重约 330MB<br>
> 锚定仓库：[aimagelab/mammoth](https://github.com/aimagelab/mammoth) 的 `dualprompt` / `l2p`（必须 `timm==0.9.8`）；[ContinualAI/avalanche](https://github.com/ContinualAI/avalanche) 的 `PackNet`；课内 CPU 实验对照论文公式做掩码占用<br>
> 产物：PackNet 掩码图 + prompt pool 是否按任务分开的探针报告 + 「冻骨干只训头」对照数字

## 1. 这一课做什么

[第 06 课](06_replay_der.md) 把新经验写进一条固定容量的回放缓冲：样本进背包，旧图和当前 batch 一起训。缓冲够用时，DER++ 这类方法很稳；缓冲变小，旧任务就开始掉。背包有两个硬限制：一是容量，二是隐私，有的场景根本不许把旧样本带在身上。

这一课换零件。主干循环里，「写到哪里」不再是「同一组权重再覆盖一遍」，也不再是「把旧图再喂一遍」，而是：**给网络再长一块地方，把旧的锁上，新的写进新块**。

四块地方，本课都要摸到：

1. **扩网络**：每个任务一根新柱子，旧柱冻住，新柱用侧向连接读旧特征。代表是 Progressive Neural Networks（渐进网络：任务来了就加一列，旧列不许改）。
2. **冻旧柱、只训头**：骨干完全不动，每个任务一个线性分类头。这是扩结构里最省的写法，也是后面 prompt 方法的对照基线。
3. **在固定宽度里砌墙**：不增加神经元，只给权重上锁。PackNet（打包网络：剪掉不重要的权重，空位留给下一任务）把任务 1 占用的权重量成占用，任务 2 只能用剩下的。
4. **加很薄的指令**：骨干冻住，学习一小段 prompt（提示向量：插在输入或注意力里的可训练短序列）。L2P 用一个共享的 prompt 池按样本检索；DualPrompt 把共享指令和任务专属指令拆开。

没有这块零件会缺什么：你会默认「抗遗忘 = 动旧权重时加弹簧或回放」。2022 年以后，很多视觉持续学习论文的主线已经变成「冻住预训练 ViT，只训 prompt」。读不懂这块，第三幕的 LoRA、第 11 课的正交低秩更新会对不上。

做完你能验证三件事：PackNet 的掩码里，任务 1 占用的位置在任务 2 训练时确实没被动；L2P / DualPrompt 选中的 prompt 索引是不是按任务分开；冻骨干只训头在同一 ViT 上能到什么分数，prompt 方法比它多出的是检索还是容量。

本课在第二幕「四类补丁」的第三类：扩结构。[第 05 课](05_ewc_regularization.md) 正则是给旧权重量弹簧，第 06 课回放是把旧样本带在身上，[第 08 课](08_gem_gdumb.md) 会把更新方向投影到「不增加旧损失」的半平面。四类补丁学完，2024 年以前的持续学习论文分类表你就能自己填。

术语速查：

| 术语 | 一句解释 |
|---|---|
| 扩结构 | 新任务来了，给网络加新参数或新通路，而不是覆盖旧权重 |
| 柱 / 列（column） | 渐进网络里每个任务对应的那一套层；旧柱冻住，新柱可以读旧柱 |
| 侧向连接（lateral connection） | 新柱某一层读取旧柱上一层特征的那组权重 |
| 参数隔离 | 用掩码或冻结，保证任务 A 的权重在学任务 B 时不变 |
| PackNet 掩码 | 一张和权重点对点对齐的 0/1 表：1 表示这个权重属于已经上锁的旧任务 |
| Adapter | 插在层与层之间的小瓶颈模块，只训它，骨干冻住 |
| Prompt / 提示向量 | 一小段可训练向量，接到输入或注意力的键值上，用来「指挥」冻住的骨干 |
| Prompt 池（prompt pool） | L2P 维护的一组 prompt；每张图用查询向量选出最相近的几条 |
| G-Prompt / E-Prompt | DualPrompt 里共享的任务不变指令，以及按任务检索的专家指令 |
| 无回放（rehearsal-free） | 训练时不存旧图；本课的 prompt 方法走这条路 |

## 2. 问题

第 05 课的弹簧和第 06 课的背包，都还在改同一组慢速权重。弹簧的问题是：你永远在「少动」和「学不会」之间拧 λ。背包的问题是：缓冲一缩小，分数就塌；有的部署场景旧数据依法不能留。

扩结构这条路问的是另一句话：**如果旧权重根本不许改，新知识还能写在哪？**

这个问题会立刻裂成四个更具体的坑：

1. **写在新柱里**。旧柱冻住，遗忘按构造为零。代价是参数量随任务数涨，测试时还要知道用哪一根柱。Rusu 等人 2016 年的渐进网络就是这个答案。它免疫遗忘，但柱子会越来越多，论文自己把可扩展性写成未解决的限制。
2. **写在同一张网上的空位里**。不增加宽度，只把「不重要」的权重剪掉、上锁，空位留给下一任务。这是 PackNet。它几乎不增加推理算力，但容量用完时新任务准确率上不去，而且测试时必须知道任务编号才能戴对掩码。
3. **写在冻住骨干上的薄插件里**。Adapter、LoRA、prompt 都属这一家。插件很小，旧骨干不动。问题变成：插件之间会不会抢同一条通路？测试时没有任务编号，怎么选出该用哪条插件？
4. **写在输入端的指令里**。L2P 和 DualPrompt 不改 ViT 的块权重，只学 prompt，再用查询机制在测试时选出指令。听起来已经接近「不用任务编号、不用回放」。你必须用实验打假：prompt 池是真按任务分开了，还是所有任务其实在抢同一组向量？

本课不声称扩结构已经解决持续学习。它解决的是「旧权重不许动时，新知识的落点」。容量用完、任务编号未知、预训练骨干本身带偏见，这三件事都会在实验里露出来。第 08 课的 GDumb 还会追问：如果任务边界很干净，把缓冲里的图拿来从头训一个模型，分数往往也不差。扩结构赢了，不自动等于你的设定接近真实部署。

类比先说清楚，再说它在哪失效。扩结构像给公司加工位：旧员工的桌子上锁，新活在新桌子上干。类比失效处有三：神经网络的「桌子」是权重，不是物理空间，冻住不等于理解了旧任务，只是不再改那些数；新柱通过侧向连接读旧特征，不是新员工路过旧工位看一眼那么简单，侧向连接本身是要学的；工位加完还要有人告诉你今天该坐哪张桌子，任务编号就是这张工牌，真实推理常常没有工牌。

## 3. 准备

- [第 01 课](01_catastrophic_forgetting.md) 的三种增量设定（任务增量 / 域增量 / 类增量）和 [第 03 课](03_cl_evaluation.md) 的平均准确率、遗忘、BWT。本课 PackNet 默认是任务增量：测试时给任务编号。L2P / DualPrompt 的主场是类增量：测试时不给任务编号。
- 第 05 课：EWC 把「重要权重」钉住，仍允许微动。本课的冻结是硬钉，掩码是硬隔离。
- 第 06 课：回放缓冲的容量曲线。本课多数方法故意不回放，对照时不要把缓冲大小偷偷加大。
- Python 3.10+、PyTorch、命令行。CPU 机制实验不下载权重。
- 跑 Mammoth 的 prompt 方法：独立虚拟环境，**先固定** `timm==0.9.8`。仓库 `requirements.txt` 写死了这个版本；README 的模型列表对 DualPrompt、L2P、CODA-Prompt 都标注了同一条依赖。版本不对，ViT 的 prompt 注入形状会对不上。
- 磁盘：ViT-B/16 的 ImageNet-21k 预训练权重大约几百 MB，第一次跑 DualPrompt 会从 Google Storage 拉 `ViT-B_16.npz`。公司网如果拦了这个地址，提前把文件放到缓存目录。
- 不需要先会 Transformer 的全部公式。用到自注意力的 Q/K/V 时，第 5 节当场定义。

建议目录：

```text
workdir/
  mammoth/
  avalanche/
  notes/lesson07.md
```

实验记录至少留下：代码提交 SHA、`timm` 精确版本、随机种子、命令、平均准确率和遗忘。后面第 08 课三方法对照时，协议必须能对上。

## 4. 学习目标

1. 在白纸上画出四条扩结构路线：新柱、冻骨干加新头、PackNet 掩码、prompt / adapter，标出「新知识写在哪、旧知识靠什么保住、测试时要不要任务编号」。
2. 写出渐进网络一层的前向公式，并指出侧向连接为什么让「冻住」仍然能迁移。
3. 用 PackNet 的掩码图回答：任务 1 占用了多少比例的权重，任务 2 训练时这些位置的梯度是不是零。
4. 解释 L2P 的 key-value 查询：查询向量从哪来、选几条 prompt、分类损失之外那一项拉近损失在干什么。
5. 说出 DualPrompt 的 G-Prompt 和 E-Prompt 分别贴在 ViT 的哪些层、默认为什么用 prefix-tuning 而不是前置拼接。
6. 独立跑通 Mammoth 的 `dualprompt` 或 `l2p`（缩小配置也算），并做一次 prompt 索引直方图探针。
7. 判断一条失败：容量用完、任务编号未知、prompt 池塌成共用，分别会在数字上长什么样。

## 5. 原理

六个机制，按同一节奏：为什么需要、怎么运转、数学定义、代码落点、怎么验证。读的时候带着一张表：每一行的「写入位置」必须能指到一个张量。

### 5.1 渐进网络：旧柱冻住，新柱侧向读取

直接微调会把旧任务的函数冲掉。最硬的保护是：旧参数变成常数。每个新任务实例化一根新柱，随机初始化，只训新柱；旧柱的激活通过侧向连接送进新柱，供它选择复用、改造或忽略。

记第 $k$ 根柱第 $i$ 层激活为 $h_i^{(k)}$，权重为 $W_i^{(k)}$，从旧柱 $j<k$ 连过来的侧向权重为 $U_i^{(k:j)}$。Rusu 等人给出：

$$
h_i^{(k)}=f\left(W_i^{(k)}h_{i-1}^{(k)}+\sum_{j<k}U_i^{(k:j)}h_{i-1}^{(j)}\right)
$$

$f$ 是逐元非线性，原文中间层用 ReLU。因为 $\{W^{(j)},U^{(j:\cdot)}\}$ 在训第 $k$ 柱时全部冻结，旧任务的前向计算不变，遗忘按构造为零。侧向连接只从旧到新，新特征不会回流污染旧柱。

实际实现里，侧向通路会加一层 adapter：先用可学习标量对齐尺度，再投影到较低维，避免第 $k$ 柱的侧向参数随 $k$ 平方膨胀。卷积层用 $1\times 1$ 卷积做同样的降维。

代码落在 Mammoth 的 `models/pnn.py`。`COMPATIBILITY = ['task-il']` 写得很干脆：测试必须给任务编号，才能选中对应的 `self.nets[task_label]`。新任务开始时 `get_pnn_backbone(..., old_cols=...)` 把旧柱列表传进新骨干；`backbone/MNISTMLP_PNN.py` 和 `backbone/ResNet18_PNN.py` 负责具体的侧向连接。Avalanche 的文档把 Progressive Neural Networks 列在 Architectural 策略下，和 multi-head、incremental classifier 并列。

验证有两条。第一，学完任务 2 之后，把任务 1 的测试集只送到第 1 柱，准确率应与刚学完任务 1 时相同，误差只来自数值实现。第二，把侧向连接输出换成零，新任务学习应变慢或变差：这才能说明新柱真的在用旧特征，而不是自己从头学。论文用 Average Fisher Sensitivity 看各柱各层被依赖的程度；课内不做 Fisher，用「侧向置零」就够。

这篇论文的主实验在强化学习（Atari、3D 迷宫），不是 CIFAR 分类。本课把它当机制来源，不宣称复现它的迁移分数。限制也写在原文里：参数量随任务增长；推理要任务标签。后文的 PackNet 和 prompt，就是在回答这两条限制。

### 5.2 冻骨干只训头：最省的扩结构

如果已经有一个在 ImageNet 上预训练好的 ViT，最省的持续学习是：骨干冻结，每个任务（或每个新类别集合）只训一个线性头。新知识写在分类头的几万个参数里，旧知识写在冻住的表征里。

记冻住的编码器为 $f_{\theta^*}$，第 $t$ 个任务的头为 $g_{\phi_t}$。训练目标只对 $\phi_t$ 求导：

$$
\min_{\phi_t}\ \frac{1}{n_t}\sum_{(x,y)\in\mathcal{D}_t}\ell\big(g_{\phi_t}(f_{\theta^*}(x)), y\big)
$$

类增量设定下通常不用多个头，而是一个会变宽的共享头：新类对应的列随机初始化，旧类的列也可以选择冻结。L2P / DualPrompt 在 Mammoth 里用的技巧更粗：当前任务训练时，把尚未出现的类以及「非当前任务的旧类」的 logits 置成 $-\infty$，只在当前任务的类别上算交叉熵。见 `models/l2p.py` 的 `logits[:, :self.n_past_classes] = -float('inf')`。这是训练时的类别掩码，不是 PackNet 那种权重掩码。

冻骨干只训头是本课必须跑的对照，不是「弱基线」可以省略。如果 prompt 方法只比它高一点点，你学到的可能只是「预训练 ViT 本身就很强」，不是 prompt 池的功劳。DualPrompt 论文第 5.3 节专门问过：更强的骨干会不会让持续学习分数无条件变好？结论是：同样的 ViT，顺序微调照样大遗忘；方法必须会用这块骨干。

验证：同一份 `seq-cifar100-224`，冻全部 `blocks` 只训 `head`，记下平均准确率和遗忘。后面 DualPrompt 的数字必须和这份对照放在同一张表。

### 5.3 PackNet：在固定宽度里砌墙

渐进网络加柱，推理变贵。PackNet 的想法是：深度网里有冗余，剪掉对当前任务不重要的权重，空出来的位置留给下一任务；留下来的位置上锁，永远不再改。

流程按论文图 1：

1. 在任务 1 上把网络训密。
2. 按绝对值排序，剪掉比例为 $\rho_1$ 的权重（常见 50% 或 75%），再短训若干 epoch 把精度捞回来。存活的权重打上任务 1 的锁。
3. 任务 2 只准改仍然为 0 的那些位置。训完后再从「任务 2 自己的权重」里剪一刀，上锁。
4. 重复，直到没有空位。

设第 $t$ 个任务的二进制掩码为 $m^{(t)}\in\{0,1\}^p$，与参数 $\theta\in\mathbb{R}^p$ 点对点对齐。任务 $t$ 训练时，梯度被掩码挡住：

$$
\theta \leftarrow \theta - \alpha\ \big(m^{\text{free}}\odot\nabla_\theta\ell_t\big),\qquad m^{\text{free}}=1-\bigvee_{k<t}m^{(k)}
$$

上锁之后，$m^{(t)}$ 记录「本任务新占用的位置」。推理任务 $k$ 时，使用

$$
\theta^{(k)}=\theta\odot\Big(\bigvee_{j\le k}m^{(j)}\Big)
$$

也就是任务 $k$ 能看见自己的权重，以及它复用的更早任务的权重。论文强调：任务 2 的滤波器是灰（任务 1）加橙（任务 2）的叠加；任务 1 推理时把橙色关掉。

剪枝本身是按层、按绝对值的一次性剪枝，不是迭代式 Lottery Ticket。偏置和 BatchNorm 的仿射参数，原文在第一轮剪完之后就冻住，不再为每个任务单独学一套，以降低每任务额外存储。额外开销主要是每任务一张掩码。参数若被多个任务共用，编码每个位置的任务编号最多需要 $\log_2 N$ bit；工程上 Avalanche 用整型 buffer 存状态。

类比：一堵墙，每块砖只能刷一次漆然后封上。新任务只能刷还没上色的砖。类比失效处：砖的「重要性」用绝对值当代理，这不是 Fisher，也不是对旧任务损失的敏感度；绝对值小的权重对旧任务仍可能关键。另外，推理必须知道今天刷的是哪一层漆，否则会把后续任务的砖也算进去。

代码落在 Avalanche：`avalanche/models/packnet.py` 定义 `PackNetModule` 的状态机，顺序为 TRAINING、POST_PRUNE、EVAL；`prune(prune_proportion)` 只允许在 TRAINING 调用，`freeze_pruned()` 把本任务占用提交为不可变。策略封装在 `avalanche/training/supervised/strategy_wrappers.py` 的 `PackNet`，插件是 `PackNetPlugin(post_prune_epochs, prune_proportion)`。Mammoth 没有名为 PackNet 的模型；本课 PackNet 以 Avalanche 或课内最小实现为准，不要到 Mammoth 里找同名文件。

验证就是掩码图。把每一层的掩码拉平，按任务上色：任务 1 一种颜色，任务 2 一种，空位一种。检查三件事：任务 1 的颜色在任务 2 的 backward 之后数值不变；空位比例随任务下降；空位耗尽时，新任务的训练损失降不下去。浏览器实验「PackNet 砌墙」把同一件事缩到一排格子上。

PackNet 原论文的主结果是 ImageNet 预训练的 VGG-16 上依次加 CUBS、Stanford Cars、Oxford Flowers，以及再加 Places365。课内用 MLP+MNIST 只验证掩码机制，分数不对论文表格。

### 5.4 Adapter：在层间塞一个小瓶颈

冻整根骨干只训头，表达能力往往不够：新领域的低层纹理和高层语义都还在用 ImageNet 的滤波器。Adapter 的折中是：在 Transformer 或卷积块里插入很小的下投影-非线性-上投影，只训这些瓶颈和层归一化，骨干权重仍冻结。

一个典型的瓶颈 adapter 作用在隐状态 $h\in\mathbb{R}^{d}$ 上：

$$
h' = h + W_{\text{up}}\,\sigma(W_{\text{down}} h)
$$

其中 $W_{\text{down}}\in\mathbb{R}^{r\times d}$，$r\ll d$。残差保证初始时 adapter 接近恒等。LoRA 把「低秩更新」直接加在某张权重上：$\Delta W=BA$，本质同样是「新知识写在新的小参数里」。本课不训 LoRA，第 11 课才把两个 LoRA 的方向正交当成主题。

Adapter 和 PackNet 的差别：Adapter **增加** 少量参数；PackNet **不增加** 宽度，只重新分配已有参数。和 prompt 的差别：Adapter 改的是层内部的函数，prompt 改的是送进层的条件。持续学习里，Adapter 仍然要回答「测试时用哪一套 adapter」。若每任务一套、测试给编号，它退回任务增量；若想做类增量，就要像 L2P 那样学检索。低秩更新之间为什么要正交，放到 [第 11 课](11_olora_treelora.md)。

本课 Adapter 档是「只讲机制，不作为仓库主实验」。原因：Mammoth 主锚定是 DualPrompt / L2P；Avalanche 本课主锚定是 PackNet。不要并列三套完整训练。

### 5.5 L2P：用 prompt 池当短记忆

预训练模型已经会很多视觉特征。持续学习若再去改这些特征，旧任务的线性可分性会被后任务的梯度拧歪。L2P（Learning to Prompt，CVPR 2022）把问题改写成：骨干冻住，学习一组短指令，按输入选出若干条接到 patch embedding 前面，让同一套 ViT 条件地做当前分类。

Prompt 池为 $\mathbf{P}=\{P_1,\ldots,P_M\}$，每条 $P_j\in\mathbb{R}^{L_p\times D}$。输入 $x$ 经冻住的 embedding 层得到 $x_e\in\mathbb{R}^{L\times D}$。选出下标 $\{s_i\}_{i=1}^{N}$ 后：

$$
x_p=[P_{s_1};\cdots;P_{s_N};x_e]
$$

再送进冻住的自注意力堆栈。查询不使用任务编号。每个 prompt 配一把可学习钥匙 $k_j\in\mathbb{R}^{D}$。查询函数 $q(x)$ 取冻住 ViT 的 `[class]` 特征（无梯度）。用余弦距离 $\gamma$ 选出 top-$N$：

$$
\mathbf{K}_x=\arg\min_{\{s_i\}\subseteq[1,M]}\sum_{i=1}^{N}\gamma\big(q(x),k_{s_i}\big)
$$

训练同时更新池、钥匙和分类头：

$$
\min_{\mathbf{P},\mathbf{K},\phi}\ \ell\big(g_\phi(f_r^{\text{avg}}(x_p)),y\big)+\lambda\sum_{k\in\mathbf{K}_x}\gamma\big(q(x),k\big)
$$

第二项把选中的钥匙拉向当前查询，让「以后长得像这张图的样本」更容易拿到同一组 prompt。论文把分类头的输入取成 prompt 位置输出的平均；Mammoth 实现里 `head_type` 默认为 `'prompt'`，与论文一致。

超参数在 L2P 论文第 5.2 节写死过一组：CIFAR-100 与 CORe50 用 $M=10,N=5,L_p=5$；5-datasets 用 $M=20,N=4,L_p=5$。$\lambda=0.5$。这些数会让 prompt 参数量大约是 4.6 万到 9.2 万，相对 ViT-B/16 的 86M 可以忽略。Mammoth `models/l2p.py` 的默认是 `pool_size_l2p=10`、`length=5`、`top_k=5`、`pull_constraint_coeff=0.1`，和论文不完全同一套，跑仓库时以命令行和 `models/config/l2p.yaml` 为准，不要混用论文表格里的 $\lambda$。

代码路径：

- `models/l2p.py`：策略。`get_parameters()` 只返回名字里带 `prompt` 或 `head` 的参数，骨干确实没在更新。
- `models/l2p_utils/prompt.py`：`Prompt.forward` 做 L2 归一化、余弦相似度、`topk`、可选的 batchwise majority vote，然后 `torch.cat([batched_prompt, x_embed], dim=1)`。
- `models/l2p_utils/l2p_model.py`：把 prompt 模块嵌进 ViT。
- 自定义骨干警告写在 `l2p.py` 文件头：`vit_base_patch16_224`，ImageNet-21k 预训练再在 ImageNet-1k 上微调，权重来自 `https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz`。

有一项实现细节必须写进实验记录：`--batchwise_prompt`。打开之后，测试时一个 batch 内先各自检索，再按多数票强制整个 batch 用同一组 prompt。Mammoth 在 `l2p.py` 里警告这会导致和不做多数票的方法不公平。`models/config/l2p.yaml` 对 `seq-cifar100-224` 写了 `batchwise_prompt: 1`。本课探针实验把这项关掉，否则直方图会被多数票抹平，你看不出实例级检索。

验证：对每个任务的测试集，记录 `prompt_idx` 的直方图。任务相似（Split CIFAR-100 的相邻超类）时，直方图可以大量重叠；任务差得远（5-datasets）时，应更分家。若所有任务的直方图几乎一样，查询机制没工作，池退化成「一条共享 prompt」。L2P 论文表 5 的消融就是这件事：去掉池、只用一条 prompt，5-datasets 平均准确率从 81.14 掉到 51.96。

### 5.6 DualPrompt：共享指令和专家指令拆开贴

L2P 的池不区分「所有任务都该遵守的指令」和「只有这个任务需要的指令」。DualPrompt（ECCV 2022，同一作者组）按互补学习系统的直觉，把 prompt 空间拆成两块：

- **G-Prompt** $g$：所有任务共享，学任务不变的指令。
- **E-Prompt** $\{e_t\}$：每个任务一套，测试时用任务钥匙检索。

它们贴在 ViT 的不同层。论文在 Split ImageNet-R 的验证集上搜过位置：G-Prompt 偏浅（约第 1-2 个 MSA），E-Prompt 偏深（约第 3-5 个 MSA），两段不重叠。直觉是浅层更偏通用视觉，深层更偏任务语义。Mammoth 默认与此对齐：`g_prompt_layer_idx=[0,1]`，`e_prompt_layer_idx=[2,3,4]`（0 起始）。

「怎么贴」比「贴哪」同样关键。Prompt-Tuning 把同一段向量拼到 Q/K/V 前面，输出序列变长；Prefix-Tuning 把 prompt 切成 $p_K,p_V$，只拼到注意力的 K 和 V 上，Q 保持原序列，输出长度不变。DualPrompt 实验里 Prefix-Tuning 更好，也更适合多层粘贴。Mammoth 默认 `use_prefix_tune_for_g_prompt=1`、`use_prefix_tune_for_e_prompt=1`。

训练任务 $t$ 时，选中 $e_t$ 与共享的 $g$，和分类头一起更新，外加钥匙匹配损失：

$$
\min_{g,e_t,k_t,\phi}\ \ell\big(f_\phi(f_{g,e_t}(x)), y\big)+\lambda\,\gamma\big(q(x),k_t\big)
$$

测试时 $t$ 未知，用 $q(x)$ 对所有 $k_t$ 取最近的一把，取出对应 E-Prompt。G-Prompt 始终在。

Mammoth `models/dualprompt.py` 里还有两处和论文叙述不完全相同、但跑仓库必须知道的工程选择：

1. 文件头写明自定义骨干，和 L2P 同一份 `ViT-B_16.npz`。学习率按 `lr * batch_size / 256` 缩放。
2. `begin_task` 会把上一任务的 E-Prompt 切片复制到当前任务的切片上，当作初始化。池的 `size` 默认 10，`top_k` 默认 1，任务数超过可分配切片时复制被跳过。
3. 训练时同样把非当前任务类别的 logits 置成 $-\infty$。这是类增量训练的常规技巧，评估时则在所有已见类上比。
4. `observe` 里 `loss = loss_clf - pull_constraint_coeff * reduce_sim`。符号是减：`reduce_sim` 越大（查询和选中钥匙越同向），损失越低。

`models/dualprompt_utils/prompt.py` 的 `EPrompt` 在 prefix 模式下把 prompt 张量排成 `(num_layers, 2, pool_size, length, num_heads, head_dim)`，那一个 2 就是 K 和 V。`batchwise_prompt` 默认在 argparse 里是 1，配置文件 `models/config/dualprompt.yaml` 对 `seq-cifar100-224` 只写了 `dataset_config: l2p`、`batch_size: 128`、`lr: 0.03`。探针时同样建议关掉多数票。

验证：G-Prompt 在任务之间应几乎不动（或只轻微漂）；E-Prompt 做 t-SNE 应按任务分开。论文图 4 就是这两张图。课内最小探针是 E-Prompt 的检索索引混淆矩阵：任务 $t$ 的测试图有多少比例拿到了 $e_t$。索引准确率和最终准确率相关，但不是一回事：索引错了，模型仍可能靠 G-Prompt 和部分共享特征蒙对。

到这里，写入位置可以收成一张表：

| 方法 | 新知识写在哪 | 旧知识怎么锁 | 测试要不要任务编号 | 参数是否随任务增长 |
|---|---|---|---|---|
| 渐进网络 | 新柱 + 侧向连接 | 旧柱冻结 | 要 | 是，近似平方 |
| 冻骨干加头 | 分类头 | 编码器冻结 | 类增量下不要 | 头会变宽 |
| PackNet | 未上锁的权重 | 掩码把已占用位置梯度打零 | 要 | 否，容量用完即停 |
| Adapter / LoRA | 小模块或低秩增量 | 骨干冻结 | 取决于检索 | 每任务一小块 |
| L2P | prompt 池 + 头 | 骨干冻结 | 不要（靠查询） | 池大小固定 |
| DualPrompt | G-Prompt + E-Prompt + 头 | 骨干冻结 | 不要（靠钥匙） | E-Prompt 按任务增 |

下一节按执行路径把这些张量在仓库里的名字对上。

## 6. 源码导读

不要按目录字母序读。按一条样本的实际路径走：数据、骨干、prompt 注入、损失、参数过滤、任务切换时的复制。

### 6.1 Mammoth：DualPrompt / L2P

入口仍是仓库根目录的 `main.py`。模型名写在 `models/<name>.py` 的 `NAME` 上，配置在 `models/config/<name>.yaml`。

| 文件 | 带着什么问题读 |
|---|---|
| `requirements.txt` | 是否写死 `timm==0.9.8` |
| `models/l2p.py` | 哪些参数 `requires_grad`？当前任务类别怎么掩？ |
| `models/l2p_utils/prompt.py` | top-$N$ 怎么选？`reduce_sim` 怎么算？ |
| `models/l2p_utils/l2p_model.py` | prompt 拼到 embedding 的哪一维？ |
| `models/dualprompt.py` | G/E 的层下标默认值；`begin_task` 复制切片 |
| `models/dualprompt_utils/prompt.py` | prefix 张量为什么有一个大小为 2 的维 |
| `models/dualprompt_utils/model.py` | 冻住的 `original_model` 只用于查询吗 |
| `models/config/l2p.yaml`、`dualprompt.yaml` | `--model_config best` 会加载什么 |
| `models/pnn.py` | 任务增量如何选柱；和 prompt 方法的 `COMPATIBILITY` 差在哪 |

读 `Prompt.forward` 时在纸上跟一次形状。假设 `pool_size=10`、`length=5`、`top_k=5`、`embed_dim=768`、batch 为 $B$：

- `prompt`：`(10, 5, 768)`
- `prompt_key`：`(10, 768)`
- `cls_features`：`(B, 768)`
- `similarity`：`(B, 10)`
- `idx`：`(B, 5)`
- `batched_prompt`：`(B, 25, 768)`，因为 $5\times 5=25$
- 拼上 patch embedding 之后，序列长度增加 25

`reduce_sim` 的实现是选中钥匙与查询的逐元乘积再对 batch 求和平均，不是论文里严格的余弦和。它和 `l2_normalize` 一起用时，方向对，尺度是「归一化之后的点积和」。对照公式时写明这一点，不要把仓库输出的 `reduce_sim` 直接当成论文 $\gamma$ 的数值。

DualPrompt 的 E-Prompt 在 prefix 模式下形状是 `(num_layers, 2, pool_size, length, num_heads, head_dim)`。ViT-B/16 有 12 个头，`768 / 12 = 64`，所以最后一维是 64。那个 2 对应 K 和 V。`begin_task` 里的切片在 prefix 模式下是 `(slice(None), slice(None), slice(cur_start, cur_end))`，第三维才是池下标。

官方命令形态（与当前 README「Run a model」一节一致）：

```bash
python main.py --model dualprompt --dataset seq-cifar100-224 --model_config best
```

缩小冒烟时不要删 `timm==0.9.8`，只减 epoch 和任务数。Mammoth 用 `--n_epochs` 和数据集自己的任务切分；若你改数据集切分，报告里必须写明，不能再引用论文的 Split CIFAR-100 十任务设定。

### 6.2 Avalanche：PackNet 与可选的 L2P

PackNet 不走 Mammoth。路径：

| 文件 | 带着什么问题读 |
|---|---|
| `avalanche/models/packnet.py` | 状态机三个状态？`prune` 和 `freeze_pruned` 各允许在哪一态调用 |
| `avalanche/training/supervised/strategy_wrappers.py` 中 `class PackNet` | `post_prune_epochs` 为什么必须小于 `train_epochs` |
| `tests/models/test_packnet.py` | 官方用什么 MLP、怎么断言旧任务不退化 |

`PackNetModule` 的注释写得很清楚：每个任务占用一个参数子集，后任务建立在前任务子集之上并共享已冻结部分，但只有未共享部分可变。带动量的优化器可能在梯度为零时仍然改参数，这是已知陷阱。课内 PackNet 演示用 SGD、动量 0，避开这个问题。

Avalanche 也实现了 L2P：`avalanche/training/supervised/l2p.py` 的 `LearningToPrompt`，骨干来自 `avalanche.models.vit.create_model`，同样依赖 timm。本课主实验用 Mammoth，是因为课程从第 06 课起把 DER 官方库当视觉方法的主仓库，配置文件和日志格式与你已经熟悉的 `derpp` 一致。 Avalanche 的 L2P 只作对照阅读，避免两套超参数搅在一起。

安装 Avalanche 的当前 README 写法：

```bash
pip install avalanche-lib
```

若你需要改 PackNet 源码做掩码导出，改为从 Git 克隆后 `pip install -e .`。

### 6.3 课内 CPU 实验在钉什么

`experiments/src/learn_cl_experiments/lessons/lesson_07.py` 不下载 ViT，不访问网络。它应当在固定种子下构造一个小 MLP 和两段合成任务，走完「训密、按幅值剪枝、上锁、训任务 2」四步，然后断言：

- 任务 1 掩码的占用比例落在剪枝率决定的区间；
- 掩码为 1 的位置在任务 2 的一次反向之后数值不变；
- 任务 2 只改了空位。

这是机制断言，不是论文分数。`python3 run.py run 07` 把结果写到 `artifacts/lesson07/result.json`。

## 7. 实验

三层都要做。浏览器只建立 PackNet 的容量直觉；CPU 实验把掩码占用钉死；GPU 上才碰 Mammoth 的 ViT prompt。每一层都先写预测，再跑，再对照。

### Step 0: 浏览器实验，PackNet 砌墙

打开本课网页的「PackNet 砌墙」。界面是一排格子，表示一层权重。你先拖「任务 1 占用比例」和「剩余容量」，再预测两件事：

1. 任务 2 训练结束后，任务 1 已经上锁的格子会不会变色。
2. 当剩余空位低于某一阈值，任务 2 的准确率会停在随机附近还是仍能缓慢上升。

预测提交之前，运行按钮应不可用。改滑块必须作废上次运行。过关条件：你预测「上锁格子不变」，并且「空位耗尽时新任务学不动」；系统用掩码规则揭晓。格子总数很小，计算在浏览器里完成，不请求网络。

把你的预测和揭晓结果抄进 `notes/lesson07.md`。后面 CPU 实验的掩码图，应和这里的着色规则是同一套：先到的任务先占用，后到的只能用白色空位。

### Step 1: CPU 机制实验

在课程仓库里：

```bash
cd experiments
```

```bash
python3 run.py run 07
```

`python3 run.py run 07` 现在应当全绿，终端打印 `[PASS]`，`artifacts/lesson07/result.json` 的 `checks` 全真。键名是 `occupied_fraction_near_half`、`occupied_are_larger_than_free`、`occupied_weights_frozen`、`free_weights_moved`、`task1_lives_in_occupied_mask`、`task1_kept_after_task2`。

MLP 上按幅度剪掉 50% 最小权重，剩下的标成任务 1 占用并上锁。本机一次运行（seed 7）：占用比例 0.5，任务 2 时占用权重位移 0.0、空闲权重位移 0.818；清掉占用后任务 1 准确率 0.529，清掉空闲仍有 1.0。换机器会变，方向不应变。不要改断言阈值去凑过；阈值是对掩码规则的承诺。

这是掩码占用的机制断言，不是 DualPrompt 的 ImageNet-R 数字。

可选：自己把掩码导出成一张图，横轴是参数下标，颜色是任务编号。这张图就是本课交付物里的 PackNet 掩码图。MLP+MNIST 或合成高斯都可，只要和 `result.json` 用同一份掩码。

### Step 2: 冻骨干只训头，对照基线

这一步仍建议用 Mammoth，骨干和 DualPrompt 相同，避免「ResNet vs ViT」混在一张表。先装依赖，**版本以仓库为准**：

```bash
git clone https://github.com/aimagelab/mammoth.git
```

```bash
pip install -r requirements.txt
```

确认：

```bash
python -c "import timm,sys; print(timm.__version__)"
```

输出必须是 `0.9.8`。不是的话，不要继续训 prompt。

冻骨干只训头：Mammoth 没有单独名叫 `linear-probe` 的模型时，用 L2P 的参数冻结列表做对照阅读，并在记录里写「对照 = 冻 `blocks/patch_embed/cls_token/norm/pos_embed`，只训 head」。若你改仓库加一个最小线性探针脚本，标注为课内胶水，不算论文方法。

缩小配置（单卡数小时内应能跑完一轮冒烟）：`seq-cifar10-224` 比 `seq-cifar100-224` 便宜，任务更少。正式对照用 `seq-cifar100-224` 才和 DualPrompt 配置文件一致。冒烟通过后再上正式集。

### Step 3: Mammoth DualPrompt 主实验

```bash
python main.py --model dualprompt --dataset seq-cifar100-224 --model_config best
```

`--model_config best` 会读 `models/config/dualprompt.yaml`。当前文件对 `seq-cifar100-224` 给出 `batch_size: 128`、`lr: 0.03`、`dataset_config: l2p`。自定义骨干会在日志里打印那一行 `ViT-B_16.npz` 警告，这是正常的。

缩小配置（档 A 机器不要硬跑全量）：把 epoch 减到 1、用 `seq-cifar10-224`，只为验证能出平均准确率和遗忘。缩小配置的分数**不构成**对 DualPrompt 论文的复现判断。

跑完记录：Average Accuracy、Forgetting、每任务结束的准确率矩阵。矩阵用第 03 课的协议算 BWT。本课不是五项正式复现之一，不要在标题写「复现 DualPrompt」。你要的是方向：无回放的 DualPrompt 应明显高于顺序微调，并接近或超过「冻骨干只训头」。

### Step 4: L2P 与 prompt 索引探针

```bash
python main.py --model l2p --dataset seq-cifar100-224 --model_config best --batchwise_prompt 0
```

这里显式关掉多数票。`models/config/l2p.yaml` 默认 `batchwise_prompt: 1`，不覆盖的话探针会被抹平。

探针做法（课内胶水，标注非官方）：在 `Prompt.forward` 已经写出的 `out['prompt_idx']` 上挂钩子，对每个任务的测试集累计索引直方图。交付物是一张「任务 × prompt 编号」的热力图。

阅读问题用实验回答：

- Split CIFAR-100 上直方图是否大量共享？L2P 论文图 3 左侧说「是」，因为类之间视觉相近。
- 若你加跑 `seq-imagenet-r`（DualPrompt 提出的 Split ImageNet-R，Mammoth 数据集名 `seq-imagenet-r`），风格变化更大，E-Prompt / 池检索应更分家。全量 ImageNet-R 标成加分项。

若直方图几乎均匀或几乎全部塌到同一条 prompt，先查查询向量有没有梯度泄漏到冻住的 `original_model`（不应有），再查 `embedding_key` 是否为 `cls`。

### Step 5: PackNet 在 Avalanche 里砌一次墙

CPU 即可。用 Avalanche 自带的 `PackNetModel` 包一个 `SimpleMLP`，基准用 SplitMNIST 或 PermutedMNIST，任务数 3，剪枝比例 0.5，`train_epochs` 大于 `post_prune_epochs`。策略类签名见 `strategy_wrappers.py`：`PackNet(model=..., optimizer=..., post_prune_epochs=..., prune_proportion=...)`。

导出每层掩码，画成本课的第二张掩码图。和 Step 1 的课内实验对照：占用比例应同方向，不必同数字（网络宽度和剪枝实现都不同）。

若 Avalanche 安装失败，Step 1 的课内掩码图仍可作为本课 PackNet 交付；在报告里写明「未跑 Avalanche，仅课内实现」。不要用自制剪枝冒充 Mallya & Lazebnik 的 VGG 数字。

### Step 6: 对照表

把同一协议下能跑的方法填进表。协议三要素必须写在表注：数据集、是否给测试任务编号、缓冲大小（prompt 方法为 0）。

| 方法 | 档 | 平均准确率 | 遗忘 | 测试要任务编号 | 缓冲 |
|---|---|---|---|---|---|
| 顺序微调（naive） | 实战对照 | 你的数 | 你的数 | 否 | 0 |
| 冻骨干只训头 | 实战对照 |  |  | 否 | 0 |
| PackNet（MLP） | 机制 |  |  | 是 | 0 |
| L2P | 实战 |  |  | 否 | 0 |
| DualPrompt | 实战 |  |  | 否 | 0 |

空格由你填。方向性预期：naive 遗忘最大；冻骨干只训头遗忘小、新类可塑性受表征上限约束；prompt 方法应高于冻骨干只训头，或在遗忘上明显更低。若 DualPrompt 低于冻骨干只训头，先查 `timm` 版本、是否误把骨干解冻、是否打开了不公平的 `batchwise_prompt` 却在对照里关掉。

## 8. 配置与预算

| 项目 | 缩小 / 冒烟 | 本课主线 | 加分 |
|---|---|---|---|
| 数据 | CPU 合成 + Split MNIST | `seq-cifar100-224` | `seq-imagenet-r` |
| 骨干 | 课内 MLP；ViT 只加载不训满 | ViT-B/16，仓库指定的 21k 预训练再 1k 微调权重 | 更大 ViT，不建议 |
| 任务数 | 2-5 | CIFAR-100 的官方切分（10 任务） | ImageNet-R 10 任务 |
| epoch | PackNet 数个 epoch；prompt 1 epoch 冒烟 | DualPrompt yaml 的默认 | 论文 5 epoch / 任务 |
| 缓冲 | 0 | 0 | L2P-R（论文带缓冲的变体，Mammoth 主实现默认无缓冲） |
| 硬件 | Mac CPU 完成 Step 0-1、PackNet MLP | 单卡 8-24GB 完成 DualPrompt / L2P | 24GB 更从容 |
| 墙钟 | CPU 实验秒级；PackNet MNIST 数分钟到一小时 | DualPrompt 全量数小时级，视 epoch 和 I/O | ImageNet-R 更长 |
| 磁盘 | 可忽略 | ViT 权重 + CIFAR 224 缓存，预留数 GB | ImageNet-R 更大 |

学习率：DualPrompt / L2P 在代码里按 `lr * batch_size / 256` 缩放。yaml 写 `lr: 0.03`、`batch_size: 128` 时，实际学习率是 $0.03\times 128/256=0.015$。记录里要写实际值，否则别人无法复跑。

随机种子：Mammoth 用 `--seed`。对照实验只改方法名，不改种子、不改数据增强、不改类别顺序。类别顺序一旦打乱，类增量的难度会变，平均准确率不能横向比。

`timm==0.9.8` 不是建议，是仓库契约。新版本 timm 的 ViT 模块名、注意力实现、`scaled_dot_product_attention` 路径都可能变。Mammoth README 还要求 PyTorch ≥ 2.1.0 以使用 `scaled_dot_product_attention`；若不能升级，按 README 去 `backbone/vit.py` 注释掉那几行，改用慢实现，并在报告里注明。

## 9. 验收

全部勾上才算本课完成。

- 浏览器「PackNet 砌墙」先预测再运行，预测命中「锁住的格子不变」和「空位耗尽则新任务学不动」。
- `python3 run.py run 07` 的 `checks` 全真（`occupied_fraction_near_half`、`occupied_are_larger_than_free`、`occupied_weights_frozen`、`free_weights_moved`、`task1_lives_in_occupied_mask`、`task1_kept_after_task2`）；`artifacts/lesson07/result.json` 存在。
- 有一张 PackNet 掩码图，能指出任务 1 占用的位置。
- 有一张 prompt 索引热力图或直方图，并写明 `batchwise_prompt` 的取值。
- 对照表至少包含「冻骨干只训头」和 DualPrompt 或 L2P 之一，协议相同。
- 能口头回答：渐进网络、PackNet、L2P 三者各自把新知识写在哪个张量上，测试时要不要任务编号。
- 独立环境里 `timm.__version__ == '0.9.8'`。
- 没有把缩小配置的分数写成论文复现。

本课不在五项正式复现清单里。DualPrompt 论文表 1 在 Split CIFAR-100 上报告平均准确率 86.51、遗忘 5.16（缓冲 0）。你的 Mammoth 运行不必对齐到小数点，但若远低于冻骨干只训头，优先查实现，不要先调学习率。

## 10. 排错

| 症状 | 原因 | 验证 | 修法 |
|---|---|---|---|
| `import timm` 失败或版本不是 0.9.8 | 装到了环境外的 timm，或被其它包升级 | `python -c "import timm; print(timm.__file__, timm.__version__)"` | 独立 venv，`pip install timm==0.9.8` |
| DualPrompt 一启动就形状报错 | 骨干不是仓库那份 ViT-B/16，或 prefix 的 head 维对不上 | 日志是否打印 `ViT-B_16.npz` 警告 | 不要自己换 `timm.create_model` 的别名；用仓库 `Model` |
| 下载 `ViT-B_16.npz` 超时 | Google Storage 不可达 | 浏览器或 `curl` 测该 URL | 手动下载后放到缓存路径，保持文件名 |
| GPU 显存溢出 | 224 分辨率、batch 128、ViT-B | `nvidia-smi` | 降 `batch_size`，同时按 256 规则重算学习率 |
| prompt 直方图所有任务同一条 | `batchwise_prompt=1`，或查询没用 cls | 打印 `prompt_idx` 在 `train=False` 时是否随样本变 | 评估探针设 `--batchwise_prompt 0` |
| 旧任务准确率在 PackNet 任务 2 之后下降 | 带动量的优化器改了「梯度为零」的权重；或推理没用对任务掩码 | 检查优化器动量；推理是否 `activate_task(k)` | SGD 动量 0；按任务切换掩码 |
| Mammoth `pnn` 在类增量设定报不兼容 | `COMPATIBILITY = ['task-il']` | 看报错信息 | 只在任务增量协议下跑 PNN |
| GEM / quadprog 相关错误 | 你进错了第 08 课的模型 | 命令里的 `--model` | 本课不要跑 `gem` |
| `occupied_weights_frozen` 为假 | 任务 2 的梯度没乘空闲掩码，占用权重仍在动 | 看 `occupied_max_update` | 对照 `lesson_07.py` 的 `mask=free`；占用位移应为 0 |
| `task1_lives_in_occupied_mask` 为假 | 幅值剪枝没把任务 1 知识留在大权重上 | 看 `task1_if_zero_occupied` 与 `task1_if_zero_free` | 清占用后任务 1 应掉（本机一次运行到 0.529），清空闲仍应高（本机 1.0） |
| L2P 损失变成 NaN | 类别掩码把整行 logits 置成 `-inf` 后仍在那些位置算 CE | 检查 `n_past_classes` 与当前 label 范围 | 确认 label 落在 `[offset_1, offset_2)` |
| 对照表 DualPrompt 远低于论文，但高于 naive | 正常的实现与增强差距 | 数据增强、epoch、224 预处理是否与 yaml 一致 | 先对齐配置，再谈分数；本课通过线是方向不是小数 |

## 11. 前沿与改造

扩结构这条线 2022 年以后几乎全部接到了预训练 Transformer 上。CODA-Prompt（Smith 等人，arXiv:2211.13218，CVPR 2023）把固定 prompt 池换成一组可加权混合的 prompt 分量，用输入条件的注意力系数拼出 prompt，目标是端到端优化、减少 L2P 那种硬检索。Mammoth 有 `coda-prompt`，同样要求 `timm==0.9.8`。DAP、STAR-Prompt、SLCA、RanPAC 都在同一仓库的模型列表里，它们的共同点是：骨干尽量少动，把持续学习的可塑性放到很小的附加参数或分类头对齐上。2024 年以后主阅读换成 LoRA 和专家：InfLoRA 把无干扰写进注入子空间，TreeLoRA 按梯度相似把层内适配器挂成树，JumpLoRA 用稀疏门控隔离参数。第 12 节列这些论文。

我们差在哪：课内没有把 prompt 检索的错误率当成一等公民指标；也没有在「任务边界模糊」的流上测 PackNet，因为 PackNet 靠任务编号切掩码。真实部署里任务编号常常不存在，这时 PackNet 和渐进网络都会退回「先猜任务再选子网」，猜错就戴错面具。

动手改造清单（01-12 课精简版，仍要可执行）：

1. **关掉 G-Prompt**。在 `models/dualprompt.py` 把 `g_prompt_layer_idx` 设成空列表（若 argparse 拒绝空列表，则在 `Model` 里跳过 G 的粘贴）。预算：一次 `seq-cifar10-224` 冒烟。预期：平均准确率下降或遗忘上升。失败标准：指标几乎不变，说明你的 G-Prompt 根本没贴上，先打日志确认层下标。
2. **把 L2P 的池缩小到 $M=1$**。`--pool_size_l2p 1 --top_k 1`。预期：接近「单条共享 prompt」，遗忘明显变大，对应 L2P 论文表 5 第一行。失败标准：分数不变，说明池大小没传到 `Prompt` 构造函数。
3. **PackNet 剪枝率扫描**。$\rho\in\{0.3,0.5,0.7\}$，同一 MLP、同一 Split MNIST。预期：$\rho$ 太大，任务 1 精度在短训后仍回不来；$\rho$ 太小，任务 3 没有空位。失败标准：三条曲线重合，掩码可能没生效。
4. **侧向置零**。若你跑了 Mammoth `pnn`，在前向里把旧柱输入乘 0。预期：新任务学习曲线变差。失败标准：完全不变，说明当前骨干实现没有把 `old_cols` 接到那一层。

顺手复现映射：本课改造 2 对应 L2P 论文的 prompt pool 消融；改造 1 对应 DualPrompt 论文表 4 的 G-Prompt / E-Prompt 消融。完整论文分数不是本课验收。第 11 课的 O-LoRA 会把「附加参数之间抢方向」再做一次，和这里的 E-Prompt 隔离是同一类问题，只是写到了低秩矩阵上。

## 12. 论文与延伸

每篇对应一个能用本课实验回答或明确答不了的问题。读完把答案写进 `notes/lesson07.md`。谱系只留 PackNet，因为 CPU 实验真的在砌掩码。2024 年以后主阅读是 LoRA 和专家；ViT prompt 池退到谱系，本课不再单列。

1. Mallya and Lazebnik, 2018, *PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning*, [arXiv:1711.05769](https://arxiv.org/abs/1711.05769)。
贡献：在固定宽度里按幅值剪枝腾空位，推理只多一张掩码。机制发明处，不是本课主阅读。
机制：任务训密后按绝对值剪掉一部分权重，存活位置上锁；下一任务只准改仍为 0 的位置。梯度被空闲掩码挡住，推理按任务编号戴对应掩码。摘要写明始终优化当前任务，不用代理损失去保旧任务。
和本课：浏览器砌墙、`python3 run.py run 07` 的 `occupied_are_larger_than_free`、`occupied_weights_frozen`、`task1_lives_in_occupied_mask`、`task1_kept_after_task2` 对应「大权重占用、上锁后不动、知识在占用集合里」。答不了 ImageNet 预训练 VGG-16 接细粒度分类的分数。
阅读问题：任务 2 之后 `occupied_max_update` 是否小于 $10^{-12}$？若 `occupied_weights_frozen` 为真，你看见了「占用位置梯度打零」。本课实验答不了「没有任务编号时怎么推理」，因为 CPU 实验始终知道任务切片。

2. Liang and Li, 2024, *InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning*, [arXiv:2404.00228](https://arxiv.org/abs/2404.00228)。
贡献：把无干扰写进注入子空间：微调注入参数等价于在预设计子空间里动预训练权重。
机制：每个新任务扩一条 LoRA 枝，先用旧任务梯度空间的正交补去设计 $B_t$，再冻住 $B_t$ 和旧枝，只训 $A_t$。前向把各枝加进冻住的 $W$。无回放、推理不用任务编号。损失用当前任务类别上的局部交叉熵。
和本课：`mask=free` 时占用权重位移为 0，和「新更新不许踩旧占用」同类。答不了 DualGPM 怎么近似旧梯度、也答不了 ViT 上把枝融回 $W_t$ 之后参数量是否恒等于一条枝。
阅读问题：PackNet 锁的是坐标（掩码），InfLoRA 锁的是 $B_t$ 张成的子空间。你的 `occupied_weights_frozen` 能类比哪一句？本课没有 LoRA 枝，子空间设计本身答不了。

3. Qian, Xu, Zhang, Zhao, Zhou, 2025, *TreeLoRA: Efficient Continual Learning via Layer-Wise LoRAs Guided by a Hierarchical Gradient-Similarity Tree*, [arXiv:2506.10355](https://arxiv.org/abs/2506.10355)。
贡献：按层把 LoRA 挂到梯度相似树上，用 bandit 的下置信界搜枝，再做稀疏更新。
机制：浅层节点共享一组任务，深层节点拆成更专的枝。新任务到来时不扫全部旧任务梯度，只拉一条最有希望的枝算残差梯度，再用 L1 正则把更新限制在相关低秩适配器上。任务结束把该任务适配器插入最近叶子。论文报告相对先前方法，ViT 训练最多约 3.2 倍加速、LLM 约 2.4 倍。
和本课：PackNet 是按幅值切坐标，TreeLoRA 是按梯度相似切任务组。浏览器空位耗尽时新任务学不动，对应树上容量用完。答不了 LCB 搜枝和层深超参。
阅读问题：若把本课 `mask=free` 改成「按层不同的空闲比例」，你预期哪一层应留更多空位给共享知识？本课实验答不了，因为所有张量共用同一 50% 分位数。

4. He, Duan, Zhu, 2025, *CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning*, [arXiv:2505.24816](https://arxiv.org/abs/2505.24816)。
贡献：共享适配器加任务专属适配器，无 exemplar 做类增量。
机制：共享枝用随机正交下投影，只持续更新上投影；早退点做蒸馏，并用上一任务共享上投影的 L2 范数重分配蒸馏梯度。专属枝加可学习的逐块缩放，块权重之间加正交项，减少任务互踩。推理用原型分类器：先走共享块，再对每个已见任务跑自己的专属块，按余弦相似度取类。
和本课：DualPrompt 的 G/E 分工是同一直觉，写在 prompt 上；CL-LoRA 写在 LoRA 上。本课索引直方图能看见 E-Prompt 是否按任务分开，看不见共享上投影的梯度重分配。
阅读问题：关掉 G-Prompt 之后平均准确率若下降，你更支持「共享指令有用」还是「只是少了容量」？本课改造 1 能答方向；CL-LoRA 的蒸馏加梯度重分配，本课实验答不了。

5. Chen, Li, Zhuang, Chen, Lyu, 2024, *Replay-Free Continual Low-Rank Adaptation with Dynamic Memory*, [arXiv:2411.00623](https://arxiv.org/abs/2411.00623)。
贡献：正交适配器加残差适配器并行挂在预训练权重旁，推理用动态记忆按输入调节残差。
机制：方法名 DualLoRA。正交枝的更新投影到与旧特征子空间正交的方向，保稳定性；残差枝在最近任务多出来的基上更新，保可塑性。推理时用残差基与当前注意力的相似度缩放残差输出，压掉与测试样本无关的分量。另用这些基估任务身份，按置信度校准分类头 logits。
和本课：PackNet 的掩码是静态 0/1，推理必须给任务编号；DualLoRA 用动态记忆在无编号时调残差。`task1_kept_after_task2` 能看见硬隔离保住旧任务，看不见推理期缩放。
阅读问题：本课推理若不用 `activate_task`、把后续任务的砖也算进去，旧任务准确率会怎样？这能类比「关掉动态记忆、残差全开」的风险。论文里的任务身份预测，本课实验答不了。

6. Zhang, Bai, Yang, Liang, 2025, *C-LoRA: Continual Low-Rank Adaptation for Pre-trained Models*, [arXiv:2502.17920](https://arxiv.org/abs/2502.17920)。
贡献：用一个可学习路由矩阵管所有任务的低秩更新，不再每任务一套 LoRA。
机制：共享 $A,B$，中间乘路由 $\mathcal{R}$。$\mathcal{R}$ 拆成冻结的 $\mathcal{R}_{\mathrm{old}}$（旧任务重要性）和可训的 $\mathcal{R}_{\delta}$（本任务增量），并对 $\mathcal{R}_{\delta}$ 加正交约束，减少新任务改写旧子空间。损失是当前任务分类加正交正则。问的是「一条 LoRA 能不能替多条」。
和本课：渐进网络和 DualPrompt 的参数随任务涨；C-LoRA 把增长收到 $r\times r$ 的路由里。本课对照表能看见「冻骨干只训头」参数几乎不涨，看不见路由矩阵如何拆 $\mathcal{R}_{\mathrm{old}}$。
阅读问题：若你只跑一个共享 prompt（L2P 池 $M=1$），遗忘是否明显变大？改造 2 能答「共享一条指令不够」。C-LoRA 的路由是否真能代替多条 LoRA，本课实验答不了。

7. Zhang, Ren, Li, Yu, Dong, Li, Ji, Bai, 2025, *Enhancing Multimodal Continual Instruction Tuning with BranchLoRA*, [arXiv:2506.02041](https://arxiv.org/abs/2506.02041)。
贡献：针对多模态持续指令微调，把 MoE-LoRA 改成不对称的树干加树枝。
机制：共享 $A$ 当树干，多条 $B$ 当枝。新任务用 top-$k$ 稀疏选枝，训完把最常激活的枝冻住；再给每个任务单独路由器，避免共享路由被最近任务带偏。推理用图文钥匙自动选路由器，不要求任务编号。论文在 CoIN 上报告相对 MoE-LoRA 遗忘更小。
和本课：PackNet 测试必须给编号；BranchLoRA 用任务选择器补编号。本课 `batchwise_prompt` 探针能看见「检索塌成一条」的失败模式，类比选择器选错路由器。答不了 LLaVA 上的 CoIN 分数。
阅读问题：任务编号未知时，PackNet 戴错掩码和 BranchLoRA 选错路由器，旧任务会怎样掉？本课只能演示前者（不用对掩码）。选择器准确率本课实验答不了。

8. Kang, Huang, Hou, Zhao, Yan, Bai, 2025, *Self-Evolving LLMs via Continual Instruction Tuning*, [arXiv:2509.18133](https://arxiv.org/abs/2509.18133)。
贡献：工业规模持续指令微调的 MoE-CL：每任务一条专属 LoRA 专家，外加一条共享专家。
机制：专属专家参数独立，用来挡住遗忘；共享专家做跨任务迁移。共享通路上加任务感知判别器（GAN），只让任务对齐的信息过去，减少噪声迁移。公开基准 MTL5 和工业基准 Tencent3。摘要写明腾讯视频内容审核 A/B 测试里人工审核成本降了 15.3%。
和本课：专属专家像新柱或 PackNet 占用块，共享专家像 DualPrompt 的 G-Prompt。本课没有对抗训练，也没有工业 A/B。
阅读问题：若把侧向连接（共享通路）置零，新任务应变慢还是不变？改造 4 能答 Mammoth `pnn` 这一句。判别器如何滤噪声，本课实验答不了。

9. Mohta, Ak, Lee, Dimitriadis, Xu, Shen, 2025, *Routing-Based Continual Learning for Multimodal Large Language Models*, [arXiv:2511.01831](https://arxiv.org/abs/2511.01831)。
贡献：用路由把新能力接进多模态大模型，数据量和算力不随任务序列变长而线性涨。
机制：token 级路由在专家池里选通路，训练效率接近顺序微调，效果接近多任务学习上界。摘要写明在 2B 到 8B 上路由与 MTL 相当；消融显示专家池变大仍然稳，且能利用任务相关性做跨模态迁移。
和本课：L2P 的 `prompt_idx` 是样本级检索，这篇是 token 级路由。本课直方图能看见检索有没有按任务分开，看不见 token 级专家分配，也看不见 2B 到 8B 的模型。
阅读问题：把 `batchwise_prompt` 打开后，索引直方图会不会被多数票抹平？这还算实例级路由吗？本课探针能答。token 级跨模态迁移本课实验答不了。

10. Dragomir et al., 2026, *JumpLoRA: Sparse Adapters for Continual Learning in Large Language Models*, [arXiv:2604.16171](https://arxiv.org/abs/2604.16171)。
贡献：用 JumpReLU 门控让 LoRA 块自适应变稀疏，做动态参数隔离。
机制：先前方法约束新适配器相对旧适配器的子空间或坐标冲突；JumpLoRA 在 LoRA 块里加可学习门，让不同任务激活不同坐标。模块可插到已有 LoRA 持续学习上。摘要写明显著抬升 IncLoRA，并超过当时的 ELLA。
和本课：PackNet 用幅值一次性剪枝再上锁，门是固定 0/1；JumpLoRA 的门随训练变。`occupied_are_larger_than_free` 检验的是剪枝阈值，不是 JumpReLU。
阅读问题：本课按 50% 分位数锁门之后，空闲权重是否真的在动（`free_weights_moved`）？这能类比「稀疏隔离让新任务仍有可写坐标」。JumpReLU 阈值怎么学，本课实验答不了。

现在整根系统多了第四种写入位置：新柱、空位、薄插件、prompt 指令。2024 年以后同一位置多半写成 LoRA 枝、路由或专家。旧权重可以真正不动。下一课要把「动权重，但更新方向不许增加旧损失」写成二次规划，再请出 GDumb：把缓冲里的图拿来从头训一个模型。若 GDumb 在你的协议上接近或超过 A-GEM，先检查任务边界是不是太干净，不要据此宣布持续学习无用。语言模型上的同一问题，从 [第 09 课](09_continual_pretraining.md) 开始。



