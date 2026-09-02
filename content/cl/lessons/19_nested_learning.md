---
id: 19_nested_learning
title: "优化器也是一层记忆"
summary: "架构和优化器看成不同时间尺度的嵌套学习问题。Hope 比 Titans 多了什么？"
unit: nested
play_tools: []
checkpoints:
  - "两时间尺度 vs 一时间尺度对照。"
  - "Hope 完整语言模型训练标「不能练」。"
---

# 第 19 课：优化器也是一层记忆

> 类型：机制实验（两时间尺度线性记忆）+ 只讲（Hope 完整语言模型，无官方训练配方）<br>
> 建议周期：2-4 天<br>
> 硬件：CPU / Mac 完成本课机制与浏览器实验；Hope 语言模型标「不能练」<br>
> 锚定：Behrouz et al. *Nested Learning: The Illusion of Deep Learning Architectures*，NeurIPS 2025，[arXiv:2512.24695](https://arxiv.org/abs/2512.24695)；Google Research 博客 2025-11-07；Schmidhuber 1992 自指学习。社区 HOPE 实现只对照结构，公式以论文为准<br>
> 产物：两时间尺度 vs 单时间尺度的合成任务对照；关掉某一层后丢哪类信息的记录；Hope 只讲笔记（不能练）

## 1. 这一课做什么

第五幕的标题是「学习变成架构」。[第 17 课](17_test_time_training.md)把隐状态从一条向量换成一套能在测试时用梯度更新的权重（TTT）。[第 18 课](18_titans_surprise.md)给写入加了一道门：惊讶的事情才进长期记忆。两课都在改「推理时权重能不能动」。本课把镜头再往后拉一级：优化器本身也是记忆。Adam 的动量、线性层的权重、注意力里的键值表，在 Nested Learning 的说法里，都是在压缩各自那条「上下文流」，只不过更新频率差了几个数量级。

上一课留下的零件是一块 surprise-gated 的长期黑板：token 流过，损失大的才写进去。那块黑板仍然只有两档转速。Google Research 2025-11-07 的博客把 Titans 说成「两级参数更新，一阶的上下文学习」。Hope 多出来的，是把更新规则也做成可在上下文里改的东西，再叠上一串转速不同的 MLP，论文叫连续记忆系统（Continuum Memory System，CMS）。完整语言模型实验没有官方可复现配方，本课标「不能练」。能练的是 CMS 的最小内核：两层更新频率不同的线性记忆，内环每 token 更新，外环每序列更新，任务要求既记住本句里的键值，又记住本篇的风格。

没有这一课，第五幕会停在「测试时有一块快权重」。你会以为优化器是训练阶段的附属工具，推理时关掉就行；也会把 Titans 的两档转速当成记忆系统的上限。有了这一课，主线里「写到哪里」那一栏多出明确的频率轴：上下文、外挂记忆、快速权重、慢速权重，对应的是不同的更新周期，不是四种互不相干的产品。

做完你要能验证三件事。浏览器「嵌套钟表」里，停掉快针、慢针或更慢针，你能事先说对丢的是哪类信息。CPU 实验里，关掉慢权重之后，跨序列的风格准确率必须掉下去；两时间尺度必须同时保住本句键值和本篇风格。Hope 的语言模型分数只允许作为论文阅读，不许写成你跑出来的数。

本课的合成任务刻意做成「两种信息、两种寿命」。本句键值每一步都换，像对话里刚出现的指代；本篇风格整段不变，像一个人说话时一直用的隐喻。单时间尺度只能保住其中一类。这和 Titans 的惊讶门不同：惊讶门决定「这一拍写不写」，频率决定「这一层多久才允许写」。两件事可以叠，本课先把频率单独测干净。

整门课的循环还是这一句：

```text
新经验进来
  先决定写到哪里（上下文 / 外挂记忆 / 快速权重 / 慢速权重）
  再决定怎么写（覆盖、追加、压缩、约束、正交、合并）
  写完立刻测：新任务会了没、旧任务还在不在
  长期还要测：还能不能继续学
```

本课换的零件是「写到哪里」里的频率，以及「怎么写」里把优化步骤本身当成一次压缩。

术语速查：

| 术语 | 一句解释 |
|---|---|
| 嵌套学习（Nested Learning） | 把模型和训练过程看成一套嵌套或并行的优化问题，每一层有自己的上下文流和更新频率 |
| 上下文流（context flow） | 某一层优化真正看见的数据：可能是 token，也可能是梯度，不一定是文本 |
| 更新频率 | 单位时间内某块参数被改几次。注意力接近每 token 一次，预训练 MLP 接近零 |
| 联想记忆（associative memory） | 学一个从键到值的映射，并把映射压进参数里 |
| 快权重 / 慢权重 | 同一套联想记忆，快的每 token 改，慢的每序列或每任务才改 |
| 连续记忆系统（CMS） | 一串按不同频率更新的 MLP，用来替代「短期注意力 + 长期 FFN」这种两档切分 |
| Hope | Nested Learning 的概念验证架构：自修改 Titans 加上 CMS。本课不能练完整语言模型 |
| 自指学习 | 网络输出里包含对自己权重的修改指令。Schmidhuber 1992 技术报告；Hope 引用它 |
| 快权重编程器（FWP） | 慢网络用外积指令改另一套快权重。Schlag et al. 2021 证明线性注意力就是这件事 |
| 只讲 / 不能练 | 无开源训练配方。社区实现只对照结构，分数以论文为准，课内不宣称对齐 |

## 2. 问题

第 17、18 课已经证明：推理时权重可以动，而且可以按惊讶程度决定写不写。把这两件事拼起来，仍然缺一块。Transformer 看起来有很多异构零件（注意力、MLP、Adam 的一阶和二阶动量），训练时却几乎用同一个学习率扫过所有层。Nested Learning 的诊断是：这种「架构归架构、优化器归优化器」的切法，会把本来按频率分层的系统拍扁。拍扁之后，你只能看见深度（叠多少层），看不见计算在时间轴上有几档转速。

具体会卡在三个问题上。

第一，优化器到底在记什么。SGD 的一步可以写成「沿着当前梯度走一小步，并且别离开现在的权重太远」。带动量的 SGD 多了一块 $m_t$，它累积的是过去的梯度，不是过去的 token。如果你承认「把映射压进参数」就是记忆，那么 $m_t$ 就是一层只看见梯度的记忆。这不是修辞。论文把 Adam、带动量的 SGD、AdaGrad 都拆成两层嵌套优化：内层压缩梯度，外层用压缩结果去改慢权重。

第二，两档转速不够。Titans 的长期记忆按惊讶写入，投影层在预训练里更新、推理时冻结，这已经是两级。Google 博客原话是：Titans 尽管记忆管理强，却只有两级参数更新，得到的是一阶上下文学习。Hope 要的是无界层级：更新规则本身也可以在上下文里被改，CMS 再提供一串中间频率。中间频率干什么？记本篇风格、本会话的惯例、本任务的计分规则。这些东西比一个 token 慢，比预训练快，正好是 [第 16 课](16_when_weights_must_move.md) 里「外挂记忆不够、必须改权重」的那一格。

第三，官方配方不存在。论文报告了把 Llama-3 的 MLP 改成可按频率更新的 CMS、再做约 15B token 的续预训练，以及 Hope 在 CLINC / Banking / DBpedia 的 class-incremental、RULER 的 needle、BABILong 长上下文上的数字。这些实验没有开源训练脚本、没有数据配方、没有官方权重。社区仓库（如 `Sk16er/hope_nano`）实现的是 Titans 风格的 delta 记忆加多周期 MLP，公式和论文第 8.3 节并不逐项相同。本课允许你打开社区代码看模块怎么摆，不允许把社区分数写成 Nested Learning 的复现。

所以本课的问题可以收成一句：如果优化器和架构是同一类联想记忆、只是转速不同，那么最少需要几档转速，才能同时记住本句和本篇？停掉其中一档，丢的是哪类信息？Hope 多出来的自修改和 CMS，哪些能在 CPU 上用线性记忆看见，哪些只能读论文？

## 3. 准备

- 第 17 课的 TTT 直觉：测试时可以更新一块矩阵 $W$，隐状态是权重不是向量。第 18 课的惊讶门：写入幅度跟当前损失或梯度有关。两课的公式不必默写，但要知道「快权重」指的是推理时还在改的那块。
- [第 05 课](05_ewc_regularization.md) 的 Fisher 弹簧和第 15 课的可塑性丢失：本课不重跑它们，但结尾会对照。EWC 是在慢权重上加弹簧；Nested Learning 是给不同块分配不同更新周期。可塑性丢失是学不动；本课的失败模式是「该慢的层被更新得太勤，把旧压缩冲掉」。
- Python 3.10+，NumPy。机制实验和浏览器实验都不需要 GPU，不下载模型。
- 读论文前先打开 Google Research 博客（2025-11-07，*Introducing Nested Learning*）。博客把 Hope 定义成 Titans 的变体：自修改循环架构 + CMS，并链到 Schmidhuber 1992 的 PDF。论文 HTML 很长，本课精读第 3 节（嵌套定义）、第 4 节（优化器是记忆）、第 7 节（CMS 公式）、第 8.3 节（Hope 前向）、第 9 节（实验设定，只讲）。
- 若要对照社区结构，克隆 `Sk16er/hope_nano` 只读 `model.py` 和 `config.py`。不要用它的训练曲线代替论文表。公式冲突时以 arXiv:2512.24695 为准。
- 梁文峰转写里「还差一步：学习如何学习」对应第五幕。这段话出自 2026-07 公开录音整理，DeepSeek 没有正式确认。本课把它当作路线判断，不当官方技术规格，也不引用转写里的卡数或内部方法。

## 4. 学习目标

1. 用 Nested Learning 的定义，把「一层 MLP + SGD」和「一层线性注意力 + SGD」都画成按更新频率排序的嵌套系统，并指出每一层的上下文流是 token 还是梯度。
2. 写出两时间尺度线性记忆的更新：快矩阵每 token 一次，慢矩阵每序列一次；说明读取时两者如何相加。
3. 在合成任务上预测并验证：关掉慢矩阵后，本句键值还在，跨序列风格掉下去。
4. 对照论文公式，说出 Hope 比 Titans 多的两件东西（自修改更新规则、CMS），以及为什么完整语言模型实验标「不能练」。
5. 打开一篇社区 HOPE 实现时，能指出至少一处与论文第 8.3 节不一致的简化，并拒绝把它的分数写成论文复现。

## 5. 原理

五个机制按同一节奏：为什么需要、怎么运转、数学定义、代码落点、怎么验证。Hope 的完整前向只在 5.5 讲档，课内代码停在两时间尺度线性记忆。

### 5.1 一层网络的训练已经是在压缩上下文

直觉。你拿一批样本训一个线性层，权重 $W$ 最后变成「看见过的输入到该有的输出」的压缩版。训完把数据丢掉，$W$ 还在，这就是记忆。类比：把一学期的讲义缩成一页公式表。失效处：公式表不会在考试当场改写自己；普通 MLP 在推理时也不改 $W$。Nested Learning 后半段要处理的，正是「压缩完之后还要不要继续压缩」。

机制。论文定义联想记忆：给定键集合和值集合，找一个算子 $\mathcal{M}$，让 $\mathcal{M}(\text{键})$ 尽量接近值。训练一个线性层 $y = Wx$ 去拟合任务损失 $\mathcal{L}$，等价于让 $W$ 记住「输入 $\mapsto$ 局部惊讶」。局部惊讶是输出空间里的梯度 $\nabla_y \mathcal{L}$：当前预测和目标差多少。梯度为零时，这一条样本不再让人惊讶，也就不必再写。

数学。随机梯度下降一步：

$$
W_{t+1} = W_t - \eta_t \nabla_{W}\mathcal{L}(W_t; x_t)
$$

它等价于在 $W_t$ 附近做一次带二次近端项的线性化最小化（论文公式 (1)(2)）：走得太远会罚。对线性层，$W$ 的梯度是输出梯度与输入的外积，所以写入本身就是一次键值关联：键是 $x_t$，值是这条样本的惊讶。

代码。课内实现没有官方仓库可指。对照落在论文第 3.1 节的 MLP 例子，以及本课 7 节脚本里 `assoc_update()`：外积写入、标量步长、可选衰减。社区 `hope_nano` 的 `model.py` 里也能看到外积写入，但那是推理时记忆，不是预训练 SGD。

验证。把 $\eta$ 设成 0，$W$ 必须不变。把同一条 $(x, u)$ 写很多次，检索 $Wx$ 必须靠近 $u$。这两条在 CPU 实验里用断言钉死。

### 5.2 动量是第二层记忆，上下文流换成了梯度

直觉。带动量的 SGD 多了一个速度向量 $m$。它不是网络的一层，却在记「最近梯度往哪走」。Nested Learning 的判断：这是另一层联想记忆，键仍然可以看成数据，值是梯度。类比：第一层记课文，第二层记你改错题时手往哪边偏。失效处：真实 Adam 还有逐坐标的二阶统计和偏置修正，课内两层拆解覆盖的是带动量的 SGD 和「Adam 可视为两层嵌套」这个视角，不实现完整 AdamW。

机制。外层用 $m$ 去改 $W$（慢）。内层把新梯度写进 $m$（快）。两层的梯度流是断开的：更新 $m$ 时不把图反传到 $W$ 的计算图里再走一遍，因为当前梯度在 $W_t$ 给定后可以预先算出来。论文把这叫两级嵌套优化，并强调它和经典「多层规划」不是同一个数学对象。

数学。论文 (10)(11) 的形式是：

$$
W_{t+1} = W_t - m_{t+1}
$$

$$
m_{t+1} = m_t + \eta_{t+1} \nabla_{W}\mathcal{L}(W_t; x_{t+1})
$$

内层可以再写成对 $m$ 的近端问题：在靠近 $m_t$ 的前提下，让 $m$ 对齐当前梯度。于是「带动量的 SGD」=「一层压缩梯度的记忆 + 一层被这份压缩推动的慢权重」。

代码。本课脚本不实现 Adam。你若打开社区 HOPE 训练器，看到的通常是普通 AdamW 在训模型参数；那是外层优化器，不是 CMS 内部的动量记忆。论文第 4 节还讨论了更表达的动量（深层记忆、更强学习规则）和 Delta Gradient Descent（更新依赖当前权重状态，从而放松 i.i.d. 假设）。课内不实现 DGD，只在 5.5 读 Hope 更新时会再见到它。

验证。构造两段相反的梯度。无动量时，$W$ 立刻跟当前梯度走；有动量时，$W$ 会在几步内仍带着前一段的方向。CPU 实验不测这条，避免和主断言抢篇幅。读论文时用它检查你是否真把 $m$ 当成状态，而不是当成超参数。

### 5.3 线性注意力是另一条嵌套：内层压缩 token，外层训投影

直觉。第 17 课的 TTT 和第 18 课的 Titans，内环都在改一块矩阵。线性注意力的标准写法 $\mathcal{M}_t = \mathcal{M}_{t-1} + v_t k_t^{\top}$ 是同一件事的 Hebb 版：每个 token 一次秩一写入。投影 $W_k, W_v, W_q$ 在预训练里学，推理时冻住。Nested Learning 把这读成两级：内级在序列上压缩键值，外级在语料上压缩「该怎么把 token 变成键值」。Schlag, Irie 与 Schmidhuber 2021 的结论是：线性化注意力在形式上就是 1990 年代的快权重编程器（FWP，arXiv:2102.11174）。慢网通过自造的键值外积，给快权重下编程指令。

机制。内层目标取点积匹配 $\tilde{\mathcal{L}} = -\langle \mathcal{M} k, v \rangle$，用步长 1 的梯度下降，正好得到线性注意力的写入。外层对投影做普通 SGD 或 Adam。两层的反向传播互不穿过对方的状态：更新 $\mathcal{M}$ 时投影当作常数，更新投影时 $\mathcal{M}$ 的逐步写入当作前向过程。论文图 3 把「带可学初始记忆的线性注意力」画成和 MLP 同构、只是多了上下文适应。

数学。

$$
k_t = W_k x_t,\quad v_t = W_v x_t,\quad q_t = W_q x_t
$$

$$
\mathcal{M}_t = \mathcal{M}_{t-1} + v_t k_t^{\top},\qquad y_t = \mathcal{M}_t q_t
$$

若再给外层配上动量，整个系统变成三级：token 记忆、梯度记忆、投影。频率排序按论文定义 2：单位时间内更新几次；次数相同则看谁的状态依赖谁。注意力块的频率是序列长度量级，MLP 在标准 Transformer 里频率是 0（推理时不更新）。

代码。课内快矩阵用这条外积规则。Titans 的惊讶门是在写入前乘一个与损失有关的系数；本课为了把「频率」从「门控」里拆出来，快矩阵每个 token 都写，慢矩阵等到序列结束才写。要看门控，回 [第 18 课](18_titans_surprise.md)。

验证。只开快矩阵：同一序列内，用最后若干个键去查，应对上刚写过的值；换下一序列并把快矩阵清零后，上一句的键值必须消失。这是 7 节 Step 3 的一半。

### 5.4 连续记忆系统：记忆是频谱，不是短/长两档

直觉。标准叙事是：注意力是工作记忆，FFN 是预训练长期记忆。人脑的振荡却有一串频率（论文引言用 gamma / beta / theta 作类比）。类比失效处必须写明：你的网络没有睡眠、没有海马回放、没有真正的神经振荡；CMS 只借用「多时间尺度」这件事。CMS 的工程含义是：把原来那一块冻结 FFN，换成 $k$ 块 MLP，第 $\ell$ 块每隔 $C^{(\ell)}$ 步才更新一次。快块记近处，慢块记稳的东西。更新一块时，别的块还留着旧压缩，知识可以从慢块再流回来。论文把这叫时间维上的环，用来缓解灾难性遗忘。

机制。论文给了三种接线。嵌套 CMS：高层 MLP 的初值由低层元学习，上下文结束就复位，得到更高阶的上下文学习。顺序 CMS：块串起来，所有块的初值在最低频层一起学。独立（按头）CMS：各块看同一输入，再用聚合（例如可学加权和）合成输出。Hope 实验用的是自修改 Titans 后面接一串顺序 MLP（论文公式 (97)）。教学实验用独立相加：快矩阵 + 慢矩阵，避免还没搞懂频率就先被残差和 LayerNorm 淹没。

数学。论文 (70)(71)。令第 $\ell$ 块频率为 $f_\ell$，块大小 $C^{(\ell)} = \max_i C^{(i)} / f_\ell$。前向是函数复合：

$$
y_t = \mathrm{MLP}^{(f_k)}\big(\mathrm{MLP}^{(f_{k-1)}}(\cdots \mathrm{MLP}^{(f_1)}(x_t))\big)
$$

参数只在步数能被 $C^{(\ell)}$ 整除时更新，否则增量为 0：

$$
\theta^{(f_\ell)}_{i+1} = \theta^{(f_\ell)}_{i} - \begin{cases} \sum_{t=i-C^{(\ell)}}^{i} \eta^{(\ell)}_t f(\theta^{(f_\ell)}_t; x_t) & \text{若 } i \equiv 0 \pmod{C^{(\ell)}} \\ 0 & \text{否则} \end{cases}
$$

$f(\cdot)$ 是所用优化器的误差项，SGD 时就是梯度。标准 Transformer 的 FFN 是 $k=1$ 且更新频率为 0 的特例。

课内两时间尺度是这条公式的线性、两块、独立相加版。快块 $C^{f}=1$（每 token），慢块 $C^{s}=T$（每序列）：

$$
M^{f}_{t} = \gamma_f M^{f}_{t-1} + \eta_f v_t k_t^{\top}
$$

$$
M^{s} \leftarrow \gamma_s M^{s} + \eta_s \sum_{t=1}^{T} v_t^{\mathrm{style}} (k^{\mathrm{style}})^{\top}
$$

$$
\hat{y}_t = \big(M^{f}_{t} + M^{s}\big) q_t
$$

新序列开始时 $M^{f}$ 清零，$M^{s}$ 留下。这是教学定义，不是 Hope 官方层。$\gamma$ 是为了在有限容量下衰减旧写入，论文 CMS 的 MLP 用优化器误差项，不一定带这种标量衰减。

代码。`experiments/src/learn_cl_experiments/lessons/lesson_19.py` 按上面三式实现。社区 `hope_nano` 的 README 把 CMS 写成 period 为 1/4/16 的 MLP 堆，并且写明早期版本 `cms_update_periods` 曾定义了却没接上。这恰好说明：社区代码可以帮你找「多周期」这个结构，不能当论文实现。

验证。主断言：`use_slow=False` 时，风格探针的余弦相似度必须明显低于两时间尺度；`use_fast=False` 时，本句键值检索必须掉。浏览器实验用同一逻辑：停慢针丢风格，停快针丢本句绑定。

### 5.5 Hope 比 Titans 多什么，以及为什么不能练

直觉。Titans 已经有惊讶门控的长期记忆，但投影和更新规则是固定的。Hope 让键、值、查询、学习率、遗忘门各自变成一块可在上下文里更新的记忆，而且每块用自己当前的记忆去生成「自己该写的值」。这就是论文说的自修改，引用的是 Schmidhuber 1992 *Steps Towards 'Self-referential' Neural Learning*（科罗拉多大学技术报告 CU-CS-627-92）：网络的输出通道里包含对自身权重的修改。CMS 再提供比「每 token / 永不」更密的频谱。两者互补：自修改 Titans 容量小、规则表达力强；CMS 容量大、规则简单、转速分档。拼起来才叫 Hope。

机制。论文 8.1 先把普通联想记忆写成：用 $W_k,W_v,W_q,W_\eta,W_\alpha$ 从 $x_t$ 得到 $k,v,q,\eta,\alpha$，再优化 $\mathcal{M}$。然后把这些投影也换成记忆 $\mathcal{M}_k,\ldots$。自修改的关键一步是 $\hat{v}_{\square,t} = \mathcal{M}_{\square,t-1}(v_t)$：每一块用自己当前状态生成自己的目标值。目标用 $L_2$，$\mathcal{L}(\mathcal{M};k,v)=\|\mathcal{M}(k)-v\|_2^2$。优化器用带衰减的 Delta Gradient Descent，因为 token 高度相关，普通点积 Hebb 不看当前权重状态。所有记忆的初态在更高层、跨序列元学习。8.3 的 Hope 模块：先跑自修改 Titans 得到 $o_t$，再把 $o_t$ 送进 CMS 的 MLP 链。另有 Hope-Attention 变体：用 softmax 全局注意力替换自修改 Titans，只保留 CMS。实验里还对 $q,k$ 做 $L_2$ 归一化，并用窗口 4 的局部卷积。

数学。论文 (94)–(97)，去掉归一化和卷积后：

$$
o_t = \mathcal{M}_{\mathrm{memory},t-1}(q_t),\quad
k_t = \mathcal{M}_{k,t-1}(x_t),\quad
v_t = \mathcal{M}_{v,t-1}(x_t)
$$

$$
\eta_t = \mathcal{M}_{\eta,t-1}(x_t),\quad
\alpha_t = \mathcal{M}_{\alpha,t-1}(x_t),\quad
\hat{v}_{\square,t} = \mathcal{M}_{\square,t-1}(v_t)
$$

$$
\mathcal{M}_{\square,t} = \mathcal{M}_{\square,t-1}\big(\alpha_t I - \eta_t k_t k_t^{\top}\big) - \eta_t \nabla\mathcal{L}_{\mathcal{M}_{\square,t-1}}(\mathcal{M}_{\square,t-1}; k_t, \hat{v}_{\square,t})
$$

其中 $\square \in \{k,v,q,\eta,\alpha,\mathrm{memory}\}$，然后

$$
y_t = \mathrm{MLP}^{(f_k)}(\mathrm{MLP}^{(f_{k-1})}(\cdots\mathrm{MLP}^{(f_1)}(o_t)))
$$

社区 `hope_nano` 常用的是

$$
M_t = M_{t-1}(I - \alpha_t k_t k_t^{\top}) + \beta_t v_t k_t^{\top}
$$

这是 Titans / DeltaNet 风格的门控 delta 规则，没有「每块生成自己的 $\hat{v}$」，也没有把 $\eta,\alpha$ 做成独立记忆。结构能对照，公式以论文为准。

代码。无官方路径。`Sk16er/hope_nano` 的 `model.py` 提供 TitansL2、多周期 CMS、`HOPEBlock`；`config.py` 里有 `cms_update_periods`。`kmccleary3301/nested_learning` 体量更大，含 `train_fsdp.py`，仍然不是 Google 官方配方。本课禁止用这些仓库的语言模型分数填验收表。

验证。Hope 语言模型：不能练。论文第 9 节可核对的设定包括：class-incremental 用 CLINC（150 个 in-scope 意图）、Banking（77 意图）、DBpedia（70 类，论文写明采样约 1 万训练 / 1 千测试）；骨干是 Llama-3 8B 与 3B，把 MLP 改成可适应后做约 15B token 续预训练；对照 ICL、EWC、InCA。长上下文用 RULER 的 needle、LongHealth、QASPER、BABILong。博客概括为：Hope 在语言建模上更强，长期记忆管理优于当时对比的模型。这些句子停留在「论文声称」。你能做的验证只有：两时间尺度合成任务的方向，是否和「多频率优于单频率」一致。方向一致不构成 Hope 复现。

### 5.6 和前四幕补丁放在同一张频率表上

直觉。前四幕的方法并没有消失，它们占用的是最低频那一档：任务边界到来才改慢权重。Nested Learning 要你看见：同一套外积写入，只改「多久写一次」，就会从 EWC 滑到 TTT。类比：同一本日记，每天写一行是情节记忆，每学期写一段是风格，毕业后再也不改是预训练。失效处：EWC 的 Fisher 弹簧约束的是参数偏离，CMS 约束的是更新日程；两者都想保护旧压缩，手段不同。

机制。把本课主线的四个写入位置标上频率：

| 写入位置 | 大约多久改一次 | 本课对应 | 前课对应 |
|---|---|---|---|
| 上下文 | 当前前向，下一步就没了 | 不改权重的 ICL | [第 04 课](04_not_just_rag.md) 把名录塞进 prompt |
| 外挂记忆 | 每次会话或每次抽取 | 本课不实现 | [第 13 课](13_external_memory.md) |
| 快速权重 | 每 token 或每段测试序列 | $M^{f}$、TTT、Titans 内环 | 第 17、18 课 |
| 慢速权重 | 每序列、每任务、或预训练结束 | $M^{s}$、CMS 低频块、普通 SGD | 第 05–12 课的训练期更新 |

EWC（第 05 课）给慢权重加弹簧：重要坐标少动。CMS 给不同块排班：有的坐标根本不在这一步的更新集合里。回放（第 06 课）把旧样本再送进同一频率的优化器。CMS 的主张是：旧知识还可以住在一块这会儿不更新的参数里，不一定非要再见到旧样本。这是假说，课内线性记忆只能看见「慢块没被本句写入冲掉」，看不见 15B token 续训后的真实保持率。

可塑性（[第 15 课](15_loss_of_plasticity.md)）是另一件事。频率排班解决的是「改太勤会冲掉旧压缩」；学不动是「改不动了」。Hope 若真能在 CMS 里循环回写，理论上既要防遗忘也要留一点高频块继续学。课内脚本的慢矩阵带衰减 $\gamma_s=0.92$，衰减本身会忘，这是容量有限下的选择，不是巩固。

数学。把 EWC 的二次惩罚和 CMS 的门控放在一起看，不要混成一个公式。EWC 是

$$
\mathcal{L}_{\mathrm{EWC}} = \mathcal{L}_{\mathrm{new}} + \frac{\lambda}{2}\sum_i F_i (\theta_i - \theta_i^{\star})^2
$$

CMS 是「这一步 $\Delta\theta^{(f_\ell)}=0$」。一个是软约束，一个是日程表。课内慢矩阵两样都没有，它只是更新得少。

验证。若你把课文脚本的慢更新放进 token 循环（$C^{s}=1$），风格和本句会抢同一块矩阵，两条分数一起坏。这和「EWC 的 $\lambda$ 极大时新任务学不会」同方向：保护过度或更新过勤，都会让两个时间尺度塌成一个。


## 6. 源码导读

本课没有官方实现。读代码的顺序是：先把论文公式当成规范，再看社区仓库怎么摆模块，最后把课内两时间尺度脚本和公式逐行对上。结论以 arXiv:2512.24695 v1 为准。

| 位置 | 你要确认的事实 | 和课内实验的关系 |
|---|---|---|
| 论文定义 2、定义 3 | 频率 = 单位时间更新次数；层级按频率排序 | 快矩阵频率 $T$，慢矩阵频率 1（每个序列一次） |
| 论文 (70)(71) | CMS 是按 $C^{(\ell)}$ 门控更新的 MLP 链 | 课内改成两块线性记忆相加，不是 MLP 链 |
| 论文 (94)–(97) | Hope = 自修改 Titans + CMS | 课内不实现自修改，也不实现 DGD |
| Google 博客 2025-11-07 | Titans 两级更新；Hope 无界层级 + CMS；链到 Schmidhuber 1992 | 用来划「只讲」边界 |
| `Sk16er/hope_nano/model.py` | `TitansL2`、CMS 周期、`HOPEBlock` | 对照结构。更新式是 $M(I-\alpha kk^{\top})+\beta vk^{\top}$，与论文 (96) 不同 |
| `Sk16er/hope_nano/config.py` | `cms_update_periods` | 看它默认周期是不是 1/4/16。那是社区选择，不是论文强制 |
| `Sk16er/hope_nano/README.md` | 作者承认早期 CMS 没接上、门控曾是静态标量 | 社区实现会静默简化，不能当复现 |
| `kmccleary3301/nested_learning/` | 更大的非官方训练入口（`train.py`、`train_fsdp.py`） | 可浏览，不可填本课验收数字 |
| `experiments/src/learn_cl_experiments/lessons/lesson_19.py` | 两时间尺度线性记忆 + 风格/键值任务 | 本课唯一带断言的实现 |

带着问题读社区 `model.py`：

1. 记忆矩阵是每 token 更新，还是按 `t mod period` 更新？
2. $\alpha,\eta$ 是从输入预测的，还是全局参数？
3. 有没有 $\hat{v}_{\square}=\mathcal{M}_{\square}(v)$ 这一步？没有就不是论文的自修改 Titans。
4. CMS 输出是串行复合还是求和？论文 Hope 用串行复合 (97)；`hope_nano` README 画的是并行求和。

Schmidhuber 1992 没有代码。报告的思想实验是：网络除了对外输出，还输出对自身权重的修改，从而能在任务进行中改自己的学习规则。Hope 的「用当前记忆生成自己的写入目标」是这件事在联想记忆上的具体化，不是 1992 报告的逐权重实现。

## 7. 实验

三层都做。浏览器和 CPU 用同一套「停一层丢一类信息」逻辑。锚定仓库实验在本课退化为：精读论文设定 + 有选择地打开社区结构，不训 Hope。

### Step 0: 浏览器实验「嵌套钟表」（先预测）

打开本课网页实验。三根针同时转：快针每 token 一格，慢针每序列一格，更慢针每任务一格。文本流里混着两类信息：本句里的键值对（谁刚说过哪句话），以及本篇风格（全文用一种隐喻、一种计分、一种称呼）。先选一项预测，再运行：

- 停快针：本句键值丢，风格还在。
- 停慢针：风格丢，本句键值还在。
- 停更慢针：跨任务的先验丢（相当于预训练 MLP 被冻死之后还强行改它），本句和本篇暂时还在。

预测选错会提示对照 5.4 的频率表，不算过关。改滑块必须作废上次运行，重新预测。过关条件：三次停针的预测都与运行结果一致。

### Step 1: CPU 机制实验

在课程仓库的 `experiments/` 目录执行：

```bash
python3 run.py run 19
```

预期写入 `artifacts/lesson19/result.json`。`python3 run.py run 19` 现在应当全绿。`checks` 六条（阈值写在 `summary` 里）：

| 检查名 | 含义 |
|---|---|
| `two_timescales_fit_token_and_style` | 快+慢同时开时，token MAE 与 style MAE 都低于阈值 |
| `no_slow_loses_style` | 关掉慢权重后，风格误差明显高于完整模型 |
| `no_fast_loses_token` | 关掉快权重后，token 误差明显高于完整模型 |
| `slow_updates_once_per_sequence` | 慢更新次数等于序列数 |
| `fast_updates_once_per_token` | 快更新次数等于 token 数 |
| `slow_is_rarer_than_fast` | 慢更新次数少于快更新 |

本机一次运行：完整模型 style MAE=0.469、token MAE=0.938；关掉慢权重后 style MAE 升到 2.000；快更新 144 次、慢更新 24 次。换机器会变，方向不应变。Hope 语言模型不能练。Step 2 的独立脚本只作对照阅读，不再替代 `run 19`。

Agent 日记也可以按两档转速写：白天进快权重，夜里写入慢权重，再清掉快的。

```bash
python3 run.py extra run sleep
```

不做夜间巩固、只清快权重，人就叫不回来。这不是 Hope 的语言模型。

### Step 2: 两时间尺度线性记忆（课内最小实现，非官方）

合成任务故意做成两件互不替代的事。本句键值：每个 token 一个随机单位向量当键、一个当值，查最后一个键应取回最后一个值。这只能靠快矩阵，因为下一序列会把 $M^{f}$ 清零。本篇风格：全序列共享一个风格向量，序列结束才写入 $M^{s}$，下一序列用固定 `style_key` 取回。风格在 token 之间不变，所以慢更新足够；快矩阵若不清零，风格会伪装成「快路径也会」，那是 bug。论文 CMS 用 MLP 链压缩一块上下文；课内用一张线性表存一个风格向量，容量极小，只为看见频率，不为看见 Hope 的语言建模。

下面的脚本对照论文 (70)(71) 的频率门控，把 MLP 换成线性联想记忆，把「每 $C^{(\ell)}$ 步更新」实例化成每 token / 每序列。它不是 Hope。把内容存成 `two_timescale_memory.py` 后运行：

```bash
python3 two_timescale_memory.py
```

```python
import numpy as np

rng = np.random.default_rng(19)
D = 24
TOKENS = 20
N_SEQ = 48
N_STYLE = 4
EVAL_FROM = N_SEQ // 2


def assoc_update(M, k, v, eta, gamma):
    return gamma * M + eta * np.outer(v, k)


def retrieve(M, q):
    return M @ q


def cosine(a, b):
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (na * nb))


def make_unit(n):
    x = rng.normal(size=(n, D))
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x


styles = make_unit(N_STYLE)
style_key = np.ones(D) / np.sqrt(D)


def run(use_fast: bool, use_slow: bool):
    Mf = np.zeros((D, D))
    Ms = np.zeros((D, D))
    local_scores = []
    style_scores = []
    for n in range(N_SEQ):
        s = n % N_STYLE
        if use_fast:
            Mf = np.zeros((D, D))
        keys = make_unit(TOKENS)
        vals = make_unit(TOKENS)
        for t in range(TOKENS):
            if use_fast:
                Mf = assoc_update(Mf, keys[t], vals[t], eta=0.85, gamma=0.97)
        M_read = np.zeros((D, D))
        if use_fast:
            M_read = M_read + Mf
        if use_slow:
            M_read = M_read + Ms
        local_scores.append(cosine(retrieve(M_read, keys[-1]), vals[-1]))
        if use_slow:
            Ms = assoc_update(Ms, style_key, styles[s], eta=0.35, gamma=0.92)
        style_q = Ms if use_slow else Mf
        style_scores.append(cosine(retrieve(style_q, style_key), styles[s]))
    return (
        float(np.mean(local_scores[EVAL_FROM:])),
        float(np.mean(style_scores[EVAL_FROM:])),
    )


both_local, both_style = run(True, True)
fast_local, fast_style = run(True, False)
slow_local, slow_style = run(False, True)
print("both", both_local, both_style)
print("fast_only", fast_local, fast_style)
print("slow_only", slow_local, slow_style)
assert both_local > 0.55
assert both_style > 0.55
assert both_style - fast_style > 0.25
assert both_local - slow_local > 0.25
```

预期方向（具体数字随 NumPy 版本会抖，断言阈值已经留了余量）：

- `both`：本句和风格都明显高于 0.55。
- `fast_only`：本句仍高，风格接近 0 附近的噪声。
- `slow_only`：风格仍高，本句掉到接近随机。

先改一处超参数再跑：把慢矩阵的 `eta` 改成 0。这等价于停慢针，`both_style` 应变差。不要同时改快、慢两个学习率，否则你分不清是频率的问题还是步长的问题。

把一条序列在纸上走一遍，确认你没有把频率写错。第 $n$ 个序列开始：$M^{f}$ 清成全零，$M^{s}$ 仍是上一篇写进去的风格。然后 20 个 token 依次外积写入 $M^{f}$。序列结束时，用最后一个键去查 $M^{f}+M^{s}$，这是本句分数；再用固定的 `style_key` 去查 $M^{s}$，这是风格分数；最后才把当前风格向量写入 $M^{s}$。顺序不能反：若先写风格再查本句，慢矩阵会泄漏到本句分数里，`slow_only` 的本句会假高。余弦相似度不是准确率。随机单位向量在 24 维的期望内积接近 0，所以 0.55 已经远离噪声；若你把 $D$ 改成 4，阈值必须重测，不要沿用 0.55。

### Step 3: 锚定对照（只读，不训 Hope）

没有官方仓库命令。按这个清单做书面对照，作为本课「锚定」部分的交付：

1. 打开 [arXiv:2512.24695](https://arxiv.org/abs/2512.24695) HTML 第 8.3 节，把 (94)–(97) 抄进笔记，标出课内脚本实现了哪几项（外积写入、两档频率），明确没实现哪几项（自修改 $\hat{v}$、DGD、CMS MLP 链、局部卷积）。
2. 打开 Google 博客，核对三句话是否属实：Nested Learning 把模型看成嵌套优化；CMS 把记忆看成频谱；Hope 是 Titans 变体且引用 Schmidhuber 1992。博客地址：
   `https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/`
3. 可选。克隆社区结构对照：

```bash
git clone https://github.com/Sk16er/hope_nano.git
```

读 `model.py` 时填一张三列纸：论文符号、社区符号、是否一致。发现不一致就停，不要「修到能训语言模型」。本课验收不包含社区训练 loss。

### Step 4: 把结果写进对照表

用 Step 1 的 `result.json` 填（Step 2 脚本只作对照，不能代替 `checks`）：

| 配置 | token MAE | style MAE | 对应停哪根针 |
|---|---|---|---|
| 快+慢 |  |  | 都不停 |
| 仅快（关掉慢权重） |  |  | 停慢针 |
| 仅慢（关掉快权重） |  |  | 停快针 |

本机一次运行：快+慢 style MAE=0.469；关掉慢权重后升到 2.000。「仅快丢风格」必须成立，否则 5.4 的机制在课包实验里没落地。若仅快也保住了风格，先查慢权重是否根本没进预测。

## 8. 配置与预算

| 项目 | 主线 | 缩小 | 不要做 |
|---|---|---|---|
| 硬件 | CPU / Mac，8GB 内存够 | 把 `N_SEQ` 改成 24 仍应看见方向 | 不要为了对齐论文去租多卡训 Llama-3 |
| 课包实验 | `python3 run.py run 19`，秒级 | 同左 | 不要改种子去刷断言 |
| 课文脚本 | `D=24, TOKENS=20, N_SEQ=48` | `N_SEQ=24` | 不要加非线性 MLP 冒充 Hope |
| Hope 语言模型 | 只讲 | 读论文图 6、图 7、表 2 的设定 | 不能练。无官方配方 |
| 社区 HOPE | 读 `model.py` 半小时 | 不克隆，只读 README | 不报社区 perplexity |
| 论文续预训练 | Llama-3 + 约 15B token，论文设定 | 无缩小版 | 不是本课作业 |

磁盘：社区仓库加上论文 PDF，几百 MB 量级。机制实验不下载数据。

## 9. 验收

- 浏览器「嵌套钟表」三次停针预测与运行一致。
- `python3 run.py run 19` 的 `checks` 全为真。本机一次运行完整 style MAE=0.469，关掉慢权重后升到 2.000。换机器会变，方向不应变。Hope 语言模型不能练。
- 对照表用 `result.json` 的 MAE 填写；课文脚本若另跑，只作对照，不能代替 checks。
- 书面：Hope 比 Titans 多自修改更新规则和 CMS；完整语言模型标「不能练」；至少指出社区实现与论文 (96) 的一处公式差异。
- 禁止项：用社区 HOPE 的生成样例或 loss 曲线代替论文第 9 节；把两时间尺度合成任务的分数写成 Nested Learning 复现。

## 10. 排错

| 症状 | 原因 | 验证 | 修法 |
|---|---|---|---|
| `both_style` 也接近 0 | 慢矩阵写入的键和查询不是同一个 `style_key` | 打印 `style_key` 是否每次重新采样 | 风格键必须跨序列固定 |
| `fast_only` 风格仍然很高 | 快矩阵没有在序列开始清零，风格残差留在 $M^{f}$ | 看 `if use_fast: Mf = 0` 是否执行 | 新序列必须复位快权重 |
| `slow_only` 本句仍然很高 | 慢矩阵每个 token 都写了键值，频率退化成 1 | 慢更新是否在 `t` 循环内 | 慢更新只能在序列结束处 |
| 课文脚本断言偶发失败 | `EVAL_FROM` 太早，风格还没写稳 | 看前几步 `style_scores` | 保持 `EVAL_FROM = N_SEQ // 2`，不要从 0 平均 |
| 打开 `hope_nano` 和论文对不上 | 社区用门控 delta，论文用 DGD + 自生成值 | 搜 `hat{v}` 或 `self-modif` | 以论文为准，把差异记进对照纸 |
| 想复现 CLINC 数字 | 无官方数据流程和 15B 续训配方 | 论文 9.1 没有开源清单 | 停。标不能练 |
| `no_slow_loses_style` 为假 | 慢权重没在序列级更新，或风格没进预测 | 看 `no_slow_style_mae` 是否明显高于 `both_style_mae` | 对照 `lesson_19.py`：慢更新只在每段序列结束处 |
| 浏览器改了针却仍显示旧结果 | 未作废上次运行 | 看是否要求重新预测 | 改控件后必须重新选预测再运行 |

## 11. 前沿与改造

前沿怎么做。论文把「叠层」之外的新维度说成「叠频率」。后续公开讨论集中在三处：把预训练好的 Transformer 的 MLP 原地改成 CMS（论文 7.3 的 ad-hoc level stacking）；给优化器也做多动量时间尺度（论文的 M3，Multi-scale Momentum Muon，本课不实现）；把自修改从 token 记忆推广到「模型改自己的更新算法」。SEAL（[第 20 课](20_seal_rl_razor.md)）走的是另一条路：不改架构频率，而让模型生成自己的微调数据和指令。两条路都对准「学习如何学习」，一个写进模块转速，一个写进外环策略。同家族后续文章里，Jafari et al. 让嵌套层数和更新频率在训练或推理中可调，MIRAS 换内目标与保留门，ATLAS 把记忆优化从最后一个输入扩到当前加过去 token；本课 $C^{f}$、$C^{s}$ 写死，这三句读 §12。

我们差在哪。课内只有两档线性记忆，没有自生成值、没有 DGD、没有在真实语言上的 CMS。Hope 的 class-incremental 数字依赖 Llama-3 和 15B token，课内看不见遗忘曲线的真实形状。社区实现已经暴露一个风险：配置项写了多周期，代码路径却可能没接上。任何「我复现了 Nested Learning」的声明，缺官方配方就不能过。

动手改造清单：

1. 第三档频率。位置：课文脚本里再加 $M^{g}$，每 `N_STYLE` 个序列更新一次，键用任务级探针。预算：CPU 分钟级。预期：同一风格族跨更长间隔仍能取回；关掉 $M^{g}$ 后，间隔超过慢矩阵衰减长度的风格掉下去。失败标准：三档分数与两档无法区分，说明新层没在独立的时间尺度上工作。
2. 给快矩阵加惊讶门。位置：`assoc_update` 前用 $\|v_t - M^{f} k_t\|$ 当写入系数，对照第 18 课。预算：CPU。预期：周期性插入的稀有键，写入幅度大于常见键。失败标准：有无门控的风格分数差小于噪声。不要宣称这是 Titans。
3. 把慢更新从「序列结束一次」改成「每 $C$ 个 token 一次」，扫 $C \in \{1,4,16,T\}$。位置：同一脚本。预算：CPU。预期：$C=1$ 时快慢退化成一块，本句和风格互相冲；$C=T$ 时回到本课主结果。失败标准：所有 $C$ 的两条曲线重叠。
4. 只读改造：在 `hope_nano/model.py` 标注论文 (96) 对应的函数，列出缺失的 $\mathcal{M}_\eta,\mathcal{M}_\alpha$ 自修改。预算：一小时阅读。预期：一张符号对照表。失败标准：对照表把社区 $\beta_t v k^{\top}$ 直接写成论文的 $-\eta \nabla\mathcal{L}$。

顺手复现映射。本课机制实验对应课程复现承诺之外的「方向性机制」，不占 1–5 号。论文复现 #5 在第 20 课（RL's Razor MNIST）。若你要在 2026 年跟一篇「多时间尺度记忆」新文章，先问：有没有官方训练配方？有，再谈复现；没有，就按本课的两时间尺度协议做机制对照。

## 12. 论文与延伸

1. Behrouz, Razaviyayn, Zhong, Mirrokni, 2025, *Nested Learning: The Illusion of Deep Learning Architectures*，NeurIPS 2025，[arXiv:2512.24695](https://arxiv.org/abs/2512.24695)。
贡献：把一个模型写成一套嵌套或多级优化，每级有自己的上下文流，并给出更表达的优化器、自修改序列模块、连续记忆系统（CMS）和概念验证架构 Hope。
机制：摘要把已知的梯度优化器（Adam、带动量的 SGD）读成压缩梯度的联想记忆，再在这个视角上换更深的记忆或更强的学习规则。自修改模块学的是自己的更新算法。CMS 把短/长两档记忆扩成一串按不同频率更新的块。Hope 把自修改序列模型和 CMS 拼在一起。课内不实现自修改，也不实现 CMS 的 MLP 链。
和本课：`lesson_19.py` 的两时间尺度线性记忆是 CMS 的最小内核。`no_slow_loses_style` 看见「低频块没被这一拍更新冲掉」；`two_timescales_fit_token_and_style` 看见两档一起才能同时保住本句和本篇。Hope 语言模型、class-incremental 数字、自生成值，本课实验答不了。
阅读问题：关掉慢矩阵之后风格 MAE 升高，你能否把它对应到 CMS「更新一块时旧压缩仍留在更低频块」？若你把慢更新放进 token 循环（§11 改造 3），`no_slow_loses_style` 还会不会成立？

2. Jafari, Ozcinar, Anbarjafari, 2025, *Dynamic Nested Hierarchies: Pioneering Self-Evolution in Machine Learning Architectures for Lifelong Intelligence*，[arXiv:2511.14823](https://arxiv.org/abs/2511.14823)。
贡献：在 Nested Learning 把更新频率写成常数之后，让优化层数、嵌套结构和各层频率在训练或推理中自己改。
机制：Nested Learning 的诊断是固定转速。这篇文章加的是调度器：层数可增可减，嵌套可改，更新周期可改。摘要把它对应到「顺行性遗忘」：固定架构只能用当前窗口或预训练里的静态知识。本课没有这套自适应调度器，也没有他们的收敛证明实验。
和本课：`fast_updates_once_per_token` 和 `slow_updates_once_per_sequence` 把次数钉死为每 token 一次、每序列一次。`slow_is_rarer_than_fast` 只验证「慢比快少」，不验证「模型自己决定少多少」。层数在推理中生长，本课实验答不了。
阅读问题：若推理时允许再长出第三档频率，你要新增哪一条 check 才能证明新档真的在独立时间尺度上工作？用 §11 改造 1 的失败标准回答；本课默认实验答不了，因为 $C^{f}$、$C^{s}$ 写死了。

3. Behrouz, Zhong, Mirrokni, 2025, *Titans: Learning to Memorize at Test Time*，[arXiv:2501.00663](https://arxiv.org/abs/2501.00663)。
贡献：注意力当短期记忆，一块神经长期记忆在测试时写入历史，并给出三种把长期记忆接进注意力的结构。
机制：长期记忆压缩过去上下文，注意力仍看当前窗口。摘要写三种变体，解决的是「记忆怎么接进主干」，不是「记忆有几档转速」。写入默认跟惊讶有关：损失大的才进长期黑板。本课为了单独测频率，快矩阵每个 token 都写，没有这扇门。
和本课：Hope 的前身，细节在 [第 18 课](18_titans_surprise.md)。`fast_updates_once_per_token` 为真，说明本课测的是频率不是门控。Titans 的 needle / 2M 上下文，本课实验答不了。
阅读问题：本课快矩阵每个 token 都写。若你先按第 18 课给写入乘上惊讶系数，`no_fast_loses_token` 还会不会成立？本课默认实验没有惊讶门，要答这题必须做 §11 改造 2，否则写「本课实验答不了，因为没有门控」。

4. Behrouz, Razaviyayn, Zhong, Mirrokni, 2025, *It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization*，[arXiv:2504.13173](https://arxiv.org/abs/2504.13173)。
贡献：把序列模型收成四个选择（联想记忆结构、注意力偏置目标、保留门、记忆学习算法），并给出 Moneta、Yaad、Memora。
机制：摘要观察到现有序列模型的内目标几乎只有点积或 L2 回归。他们换这项目标，并把现代结构里的遗忘门重新解释成保留正则。四个旋钮可以独立拧。本课脚本只拧了「谁多久写一次」，内目标仍是外积加法。
和本课：`assoc_update` 对应他们说的点积/外积这一档，没有换 L2 回归以外的偏置，也没有新的 forget gate。语言建模、常识推理、召回型任务上的表，本课实验答不了。
阅读问题：你若只扫慢矩阵的更新间隔 $C$，没有换键值匹配的损失，这算不算做过 MIRAS 的四个选择之一？用 `slow_updates_once_per_sequence` 回答「改了哪一档、没改哪一档」。

5. Behrouz, Li, Kacham, Daliri, Deng, Zhong, Razaviyayn, Mirrokni, 2025, *ATLAS: Learning to Optimally Memorize the Context at Test Time*，[arXiv:2505.23735](https://arxiv.org/abs/2505.23735)。
贡献：长期记忆不再只对最后一个输入做在线一步，而对当前和过去 token 一起优化；并给出 DeepTransformers 这一族。
机制：摘要把线性循环记忆的短板收成三件：容量受结构和特征映射限制、在线只看最后一个输入、固定容量管理弱。ATLAS 改的是第二条：记忆的目标函数看见一段过去，而不只看见 $x_t$。摘要写在 BABILong 10M 上下文上，ATLAS 相对 Titans 再提高，准确率为 +80%。
和本课：慢矩阵在序列结束时用整段残差更新，已经不是「只看最后一个 token」，和这篇同方向。容量仍是一块线性矩阵，没有他们的高容量记忆，也没有 10M 上下文。`no_slow_loses_style` 能看见「跨序列的东西要低频块」，看不见 BABILong。
阅读问题：本课慢更新发生在长度为 6 的序列结束处。这和 ATLAS「对当前加过去一起优化」是不是同一件事？若你认为差在容量和窗口，写明本课哪条 check 仍然答不了 10M。

6. Zweiger, Pari, Guo, Akyürek, Kim, Agrawal, 2025, *Self-Adapting Language Models*，[arXiv:2506.10943](https://arxiv.org/abs/2506.10943)。
贡献：模型生成自编辑（改写数据或更新指令），内环 SFT 把编辑写进权重，外环用更新后的下游表现当奖励。
机制：自编辑是自然语言，不是外挂超网络吐出来的 LoRA。内环仍是监督微调，知识线常用 LoRA。奖励依赖当前参数，所以他们坚持 on-policy。对照 Hope：SEAL 改的是「写什么文本进慢权重」，Hope 改的是更新规则和转速。两条都对准「学习如何学习」，写入位置不同。
和本课：本课 CPU 没有生成文本、没有 LoRA、没有外环奖励。频率实验只能帮你分清「写到哪一档」，分不清「谁来出题」。连续自编辑的遗忘曲线在 [第 20 课](20_seal_rl_razor.md)，本课实验答不了。
阅读问题：Hope 的自修改和 SEAL 的自编辑，课内两时间尺度脚本实现了哪一个？若两个都没有，用 `fast_updates` / `slow_updates` 说明你实际改的是日程表，不是出题策略。

7. Behrouz, Mirrokni / Google Research，2025-11-07，*Introducing Nested Learning: A new ML paradigm for continual learning*，[实验室博客](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/)。这是博客，不是 arXiv。
贡献：把 Nested Learning 说成「架构和优化器是同一类、只是层级不同」，并把 Hope 定义成 Titans 变体：自修改循环加 CMS。
机制：公开叙述把 Titans 写成只有两级参数更新，得到一阶上下文学习；Hope 的层数不受这两级限制，并能用自指过程优化自己的记忆。博客链到 Schmidhuber 1992 的 PDF，不当成论文编号。实验数字以 [arXiv:2512.24695](https://arxiv.org/abs/2512.24695) 为准，博客只提供叙事。
和本课：浏览器「嵌套钟表」的三档转速来自这篇叙事。`slow_is_rarer_than_fast` 看见两档转速，看不见「更新规则也在上下文里学」。$\eta_f$、$\eta_s$ 在脚本里是常数。
阅读问题：博客说 Titans 是一阶上下文学习。结合第 18 课，一阶指的是「记忆在更新、更新规则本身不在上下文里学」吗？本课实验有没有碰到更新规则的学习？没有的话写「本课实验答不了，因为步长是手写常数」。

8. Schlag, Irie, Schmidhuber, 2021, *Linear Transformers Are Secretly Fast Weight Programmers*，[arXiv:2102.11174](https://arxiv.org/abs/2102.11174)。
贡献：线性化自注意力的外积更新，与 1990 年代快权重编程器形式等价；并把纯加法改成接近 delta 的规则，让步长可学。机制发明处，不是本课主阅读。
机制：慢网用自造的键值外积给快网下指令。快权重是有限容量的记忆，慢网学的是如何改这份记忆。delta 规则让同一键上的旧值可以被改写，而不只是累加。本课快矩阵用的是加法外积，没有 delta，也没有可学步长。
和本课：5.3 节把课内 $M^{f}$ 写成一条 FWP 指令流。`fast_updates_once_per_token` 对应「每条指令写一次」；慢矩阵跨序列留下，对应慢状态。把慢矩阵也改成每 token 更新之后，`slow_is_rarer_than_fast` 会变成假。
阅读问题：FWP 的慢网在本课对应哪一块参数？若你把慢矩阵改成每 token 更新，还算不算「慢网编程快网」？用 `slow_is_rarer_than_fast` 回答。

现在系统里多了一根频率轴：快权重记本句，慢权重记本篇，更慢的仍冻结在预训练里。下一课不再加转速，而问另一件事：模型能不能给自己出微调题，以及为什么 on-policy 的强化学习往往比离线 SFT 更不容易把旧能力拉走。Hope 不能练的那一档，也不许用下一课的 SEAL 分数来填。那是 [第 20 课](20_seal_rl_razor.md)。



