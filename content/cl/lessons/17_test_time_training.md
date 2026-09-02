---
id: 17_test_time_training
title: "读这段话的时候权重正在动"
summary: "TTT 层的隐状态是一套可以用梯度更新的权重。这和普通 RNN 差在哪？"
unit: nested
play_tools: []
checkpoints:
  - "W 更新范数曲线。"
  - "TTT vs RNN 的状态含义对照表。"
---

# 第 17 课：读这段话的时候，权重正在动

> 类型：机制实验（短序列内环；语言模型分数只引用论文，本课不训）<br>
> 建议周期：2-4 天<br>
> 硬件：CPU / Mac 完成本课机制实验和浏览器实验；单张 24GB 卡可跑官方 PyTorch 模型定义和短序列前向。大规模训练按官方说明走 JAX 仓库，本课不要求<br>
> 锚定仓库：[test-time-training/ttt-lm-pytorch](https://github.com/test-time-training/ttt-lm-pytorch)（官方 PyTorch 模型定义）。对照阅读 [ttt-lm-jax](https://github.com/test-time-training/ttt-lm-jax)。仓库 README 写明：不建议用这份纯 PyTorch 代码做大规模训练<br>
> 产物：短序列上 TTT-Linear 的 $W$ 更新范数曲线、TTT 与线性 RNN 的状态含义对照表、内环一步的公式对照笔记

## 1. 这一课做什么

前四幕把「学习」当成训练阶段的事：任务来了，改慢权重，或者把日记写在模型外面。
[第 16 课](16_when_weights_must_move.md) 留下一张分流表：事实可以外挂，技能和
新的计分规则通常要进权重。那张表有一个缺口。推理正在进行时，模型能不能在
**当前这段输入上**改一小块权重，读完这段再把这块丢掉或留给下一层？如果可以，
「必须改权重」就不只发生在离线微调里，也可以发生在一次前向过程里。

这是第五幕（第 17-20 课）的第一块零件。整门课的六幕里，这一幕的任务是把学习
写进架构和测试时计算。公开转写里梁文峰把「学习如何学习」说成 Agent 之后还差
的一步（2026-05 交流，2026-07 公开整理；转写未获 DeepSeek 确认）。课内只把这
句话当路线判断：测试时更新、惊讶门控、嵌套时间尺度、自编辑，是目前能对着论
文和代码摸到的公开技术。本课不讨论内部方法、卡数或定价。

主干循环在本课换成这个零件：

```text
新经验进来（当前这段 token）
  写到哪里：快速权重 W（内环），慢权重 θ（外环，本课只读）
  怎么写：对当前 token 做一步自监督梯度
  立刻测：W 动了没有；同规模线性 RNN 的隐状态是不是仍只是一条向量
```

2024 年 Sun 等人的论文 *Learning to (Learn at Test Time): RNNs with Expressive
Hidden States*（[arXiv:2407.04620](https://arxiv.org/abs/2407.04620)，当前
v4，2025-08-31；实验在第一版完成）把这件事做成一层序列模型，叫做测试时训练
层（TTT layer：读当前序列时，隐状态按自监督损失做梯度更新）。它的隐状态不是
RNN 里那条固定长度的向量，也不是注意力里随时间变长的 KV 列表。隐状态是一个
小模型的权重 $W$。每来一个 token，就在 $W$ 上走一步梯度。读完这段话，$W$ 已
经被这段话训过一遍。

本课分档必须先说清：

| 档 | 本课做什么 | 本课不做什么 |
|---|---|---|
| 机制实验（主线） | 按论文公式在短序列上跑 TTT-Linear 内环，打印每步 W 的更新范数 | 不把短序列范数当成语言模型分数 |
| 锚定仓库（实战/体验） | 克隆官方 PyTorch 仓库，读 `ttt.py` 的模型定义，用随机短序列走一遍前向 | 不按这份 PyTorch 代码做大规模训练。README 写了训练会很慢 |
| 只讲 | 论文在 The Pile / Books3 上 125M 到 1.3B 的困惑度曲线 | 不复现 Kaplan / Chinchilla 配方，不报自己的 Pile 分数 |

没有 TTT 层会缺什么：你仍然可以靠长上下文、检索或离线 LoRA 应付「这段里刚出现
的事实」。你没有一个**固定大小、却能对当前序列做多步学习**的压缩器。线性 RNN
把历史压进一条向量，压完就不能再对这段历史做梯度。注意力把历史全留下，代价随
长度二次增长。TTT 要的是第三条路：把历史压进 $W$，压缩启发式本身是一次学习。

做完你能验证三件事：内环之后 $W$ 的更新范数大于 0；同规模线性 RNN 的隐状态是
向量，不能对当前序列做多步梯度；官方 PyTorch 定义里，内环更新发生在前向过程，
不发生在你另外写的 `optimizer.step()` 里。

术语速查：

| 术语 | 一句解释 |
|---|---|
| 测试时训练层（TTT layer） | 一层序列模型，读当前序列时用自监督梯度更新自己的隐状态 |
| 隐状态 $W$ | TTT 层里被内环更新的那套权重，形状随你选的小模型而定 |
| 内环 | 对当前序列、对 $W$ 做的梯度步。测试时也走 |
| 外环 | 训练更大网络时，对 $\theta_K,\theta_Q,\theta_V$ 等慢权重做的更新 |
| TTT-Linear | 内环小模型是线性映射 $f(x)=Wx$，再加 LayerNorm 和残差 |
| TTT-MLP | 内环小模型是两层 MLP（隐层宽度 $4\times$，GELU） |
| 线性 RNN | 隐状态是固定长度向量，更新规则是关于 $h_{t-1}$ 和 $x_t$ 的固定映射 |
| mini-batch TTT | 内环按 $b$ 个 token 一组算梯度。论文主实验 $b=16$ |
| 对偶形式（dual form） | 不显式物化每步的 $G_t$、$W_t$，用矩阵乘法一次算出 $W_b$ 和 $z_{1:b}$ |

## 2. 问题

核心问题：TTT 层的隐状态是一套可以用梯度更新的权重。这和普通 RNN 差在哪？

普通 RNN（含 LSTM、以及论文拿来对照的 Mamba）必须把已经看过的 token 压进一块
固定大小的状态。状态是向量时，更新规则再精巧，能表达的关系也受这块容量限制。
论文 Figure 2 右图给出一个具体观察：在匹配训练 FLOPs、上下文收到 32k 的设定
下，Transformer 的按位置平均困惑度随 token 下标继续下降；Mamba 在约 16k 之后
走平。数字来自论文，本课不复现这条曲线。它说明的机制是：线性复杂度的好处要
在长上下文才显现，而现有 RNN 恰恰在这段长度上用不好多出来的条件信息。

自注意力走另一条极端。它的隐状态是 KV 缓存，一份随 $t$ 变长的列表。更新规则
是把当前 $(k_t,v_t)$ 追加进去，输出规则是扫过截至 $t$ 的全部元组。表达力够，
每个新 token 的代价也随 $t$ 线性增长，整段是二次。

TTT 的主张写在论文第 2.1 节：把历史上下文 $x_1,\ldots,x_t$ 当成一份无标注
数据集，把隐状态当成一个模型 $f$ 的权重 $W$。压缩启发式就是自监督学习。大
语言模型用下一 token 预测把互联网压进慢权重；TTT 层用重建任务把**当前这段**
压进 $W$。$W$ 会记住产生大梯度的输入，也就是让小模型「多学到一点」的那些
token。

本课要在短序列上钉死三件事：

1. 内环一步之后，$W$ 必须动。更新范数为 0，说明你把 $W$ 当成了外环参数，或
   者梯度没传进内环。
2. 线性 RNN 的 $h_t$ 是向量。你可以对 $h_t$ 做线性变换，不能在层内对当前序列
   做多步 $\nabla_W \ell$。
3. 官方 PyTorch 仓库提供的是模型定义，不是大规模训练配方。把仓库跑出的短序列
   范数写成 Pile 分数，是把档次写错了。

谱系上要和 2020 年那篇 TTT 分开。Sun 等人 ICML 2020 的 *Test-Time Training with
Self-Supervision for Generalization under Distribution Shifts*
（[arXiv:1909.13231](https://arxiv.org/abs/1909.13231)）解决的是测试分布偏移：
对**一张**测试图做自监督微调，提高分类准确率。2024 这篇把同名想法做成序列
层：对**一段** token 逐步更新 $W$，目标是长上下文语言建模。本课只把 2020 当
谱系，不跑 CIFAR。

和前四幕的补丁也要划开。EWC 给慢权重加弹簧，回放把旧样本再喂一遍，LoRA 在
慢权重旁边加低秩方向。它们都发生在「任务边界」上：任务 2 的 dataloader 开始
之后，才改参数。TTT 层没有任务边界。$x_t$ 一进层，内环就走一步。序列结束，
$W$ 可以丢掉。这更接近「读这段话时临时学」，不是「这个季度微调一次」。第
19 课会把不同更新频率嵌套起来；本课只要求你看见最快的那一层。

## 3. 准备

- 会用 Python 和 NumPy 或 PyTorch 写一层线性映射、一次 MSE、一次 `autograd`。
  不需要先会 Mamba 或 JAX。
- 读过 [第 16 课](16_when_weights_must_move.md) 的分流结论：外挂记忆覆盖事实
  查询，覆盖不了必须改权重的那一类经验。本课不依赖第 16 课的代码产物。
- 机器：机制实验和浏览器实验用 CPU 即可。克隆官方仓库、实例化 `TTTForCausalLM`
  并在长度 16 到 64 的随机序列上前向，Mac CPU 也跑得动，只是第一次装
  `transformers[torch]` 会花时间。
- 磁盘：克隆 PyTorch 仓库很小。不要在这一课下载 The Pile 或 Books3。
- Hugging Face 上的 Llama-2 分词器（README 示例用 `meta-llama/Llama-2-7b-hf`）
  需要账号和许可。本课主线**不用**分词器：随机张量足够验证内环。需要对照官方
  `generate` 示例时，再处理许可；没有许可不挡验收。
- 建议先打开论文 HTML：[arXiv:2407.04620v4](https://arxiv.org/html/2407.04620v4)
  的 Figure 1、Figure 3、Figure 5 和公式 (1)–(5)。本课公式全部从这里抄，不从
  博客抄。

## 4. 学习目标

1. 画出序列层的三件套：初始状态、更新规则、输出规则，并填进 RNN、自注意力、
   TTT 三种实例。
2. 默写 TTT-Linear 的内环更新和输出（论文公式 (2)、(4)、(5)），指出哪些符号
   是内环变量、哪些是外环参数。
3. 解释为什么 mini-batch 大小 $b=1$ 是 online GD、$b=T$ 是 batch GD、论文选
   $b=16$，以及 dual form 改的是墙钟时间而不是数学输出。
4. 陈述 Theorem 1 的条件：线性 $f$、$W_0=0$、batch GD、$\eta=1/2$ 时，TTT 层
   与线性注意力输出相同。
5. 在短序列上跑内环，得到 $W$ 更新范数曲线，并对照同规模线性 RNN。
6. 打开官方 `ttt.py`，指出 `TTTLinear` / `TTTCache` 对应论文的哪一段；口头说
   清为什么本课不拿这份代码训 1.3B。

## 5. 原理

七个机制，每个按同一节奏：为什么需要、怎么运转、精确定义、代码落点、怎么验
证。公式编号跟论文走。

### 5.1 序列层的三件套：状态、更新、输出

任何自回归序列层都可以写成：读入 $x_t$，改一块隐状态，再吐出 $z_t$。论文
Figure 3 把三件套列成表。线性 RNN 的状态是向量，更新是 $s_t=\sigma(\theta_{ss}s_{t-1}+\theta_{sx}x_t)$，
每步 $O(1)$。自注意力的状态是列表，更新是 `append(k_t, v_t)`，输出扫过截至 $t$
的全部键值，每步 $O(t)$。TTT 的状态是 $W_t=f.\mathrm{params}()$，更新是对自监督
损失走一步梯度，输出是 $f$ 在更新后的权重上做预测，每步仍是 $O(1)$（相对序列
长度；对隐状态维度 $d$ 则是 $O(d^2)$ 量级）。

直觉：RNN 像一张写满就擦的便签，注意力像把每句话都复印进档案柜，TTT 像带一
个小练习本，每读一句就在练习本上改一笔。类比失效处：练习本上的损失是重建
MSE，不是「理解这句话」；小模型可以是线性层，容量有限。

验证：第 7 节对照实验里，线性 RNN 打印的是向量 $h_t$ 的欧氏范数，TTT 打印的
是矩阵 $W_t$ 的 Frobenius 范数。两者量纲不同，不能比大小，只能比「状态是什么
形状、会不会对当前 token 做 $\nabla_W$」。

### 5.2 内环：把当前 token 写成一次自监督步进

论文第 2.1 节给出最简 TTT。输出规则：

$$
z_t = f(x_t; W_t)
$$

更新规则：

$$
W_t = W_{t-1} - \eta \nabla \ell(W_{t-1}; x_t)
$$

最简重建损失先把 $x_t$ 损坏成 $\tilde{x}_t$，再要求 $f$ 从损坏版还原 $x_t$：

$$
\ell(W; x_t) = \|f(\tilde{x}_t; W) - x_t\|^2
$$

$\eta$ 是内环学习率。$W_0$ 在这一小节可先当成 0，第 5.6 节改成可学习的
$\theta_{\mathrm{init}}$。

为什么是重建而不是下一 token 预测：层内还没有词表上的分类头。重建逼 $f$ 发现
$x_t$ 各维之间的相关，才能从部分信息还原。论文也试过在 $f$ 后再加一个解码器
$g$，重建略好，但整体训练更不稳、算量明显增加，所以正文用 encoder-only。

梯度大的输入会被 $W$ 记住。这是压缩启发式，不是注意力分数。验证方法：构造一
个「第 5 个 token 的某一维特别大」的短序列，看 $\|W_5-W_4\|_F$ 是否大于邻
近步。浏览器实验「内环台阶」把这件事画成格子。

代码落点：论文 Figure 5 的 `Learner.train` / `Learner.predict`。官方仓库
[`ttt.py`](https://github.com/test-time-training/ttt-lm-pytorch/blob/main/ttt.py)
里，对应的是 `TTTLinear`（继承 `TTTBase`）在前向中维护的快速权重，以及
`TTTCache` 保存的最近隐状态和梯度。本课短序列实验按公式手写一版 primal form，
再去仓库核对类名，不把社区博客里的伪代码当公式来源。

### 5.3 外环学会「内环在学什么」：三视图

最简重建把损坏和标签都写死。论文第 2.3 节改成可学习的多视图重建。训练视图、
标签视图、测试视图分别是三个外环矩阵：

$$
\ell(W; x_t) = \bigl\| f(\theta_K x_t; W) - \theta_V x_t \bigr\|^2
$$

$$
z_t = f(\theta_Q x_t; W_t)
$$

$\theta_K x_t$ 是训练视图：$W$ 实际吃进去、用来算内环梯度的那一面。
$\theta_V x_t$ 是标签视图：值得写进 $W$ 的那一部分，不必是 $x_t$ 的全部坐标。
$\theta_Q x_t$ 是测试视图：用来读 $W$、生成当前层输出 $z_t$ 的那一面。三套
矩阵和自注意力的 $W_K,W_V,W_Q$ 位置类似，性质不同：它们是外环参数，内环里当
成损失的超参，不算进 $W$。

内环只优化 $W$。外环优化 $\theta_K,\theta_Q,\theta_V$ 以及网络其余部分
$\theta_{\mathrm{rest}}$。对 $\nabla\ell$ 再求导就是对梯度求梯度，论文把
这称为元学习里的标准技术。Figure 5 用代码把这件事写死：`Task` 是 `nn.Module`，
里面的 $\theta$ 会进外环；`Learner` 不是 `nn.Module`，`state.model` 只在
`state.train` 里手动更新。

验证：在短序列实现里，把 $\theta_K,\theta_V,\theta_Q$ 设成 `requires_grad=False`
，只让 $W$ 变。内环范数仍应大于 0。若你发现 $\theta$ 也在变，说明误把外环参数
放进了内环 optimizer。

### 5.4 mini-batch 内环，以及为什么要 dual form

online GD 的 $W_t$ 既出现在减号前，又出现在 $\nabla\ell(W_{t-1};x_t)$ 里，
梯度部分无法并行。论文第 2.4 节把一般 GD 写成

$$
W_t = W_{t-1} - \eta G_t = W_0 - \eta \sum_{s=1}^{t} G_s
$$

$G_t$ 的取法决定并行度。全部对 $W_0$ 求导是 batch GD，有效搜索空间小，语言
建模更差。全部对 $W_{t-1}$ 求导是 online GD，无法并行。折中是 mini-batch：令
$t'=t-\mathrm{mod}(t,b)$（上一块的结尾，第一块为 0），

$$
G_t = \nabla\ell(W_{t'}; x_t)
$$

一块里 $b$ 个 $G$ 互不依赖，可以一起算。论文全部主实验取 $b=16$。Figure 7
左图：更小的 $b$ 困惑度更好（步数更多）；$b=16$ 时 125M TTT-Linear 的困惑度
是 11.09，对应 Figure 10 的最终点。右图：dual form 下前向时间随 $b$ 先降后升。
这些是论文数字，本课短序列不重跑 Figure 7。

即使 $G_t$ 可并行，primal form 仍要为每个 token 做外积，得到 $d\times d$ 的
$G_t$，内存带宽打满，Tensor Core 吃不饱。第 2.5 节的 dual form 观察：只要最
终能得到这块结束时的 $W_b$ 和 $z_1,\ldots,z_b$，中间的 $G$、$W$ 不必物化。
以 $\theta_K=\theta_V=\theta_Q=I$、$f(x)=Wx$、第一块为例：

$$
G_t = 2(W_0 x_t - x_t) x_t^{\top}
$$

$$
W_b = W_0 - 2\eta (W_0 X - X) X^{\top}
$$

其中 $X=[x_1,\ldots,x_b]$。输出侧令 $\delta_t=\sum_{s=1}^{t}(W_0 x_s-x_s)x_s^{\top}x_t$，
则

$$
\Delta = (W_0 X - X)\,\mathrm{mask}(X^{\top}X)
$$

$\mathrm{mask}$ 是上三角、零而不是 $-\infty$ 的那种。于是
$Z = W_0 X - 2\eta\Delta$。primal 与 dual 输出等价。论文写 JAX 实现里 dual
训练比 primal 快 5 倍以上。本课短序列用 primal，便于逐步打印 $\|G_t\|_F$。
官方大规模训练在 JAX 仓库里走 fused kernel / dual，不要用 PyTorch 仓库去追这
个 5 倍。

### 5.5 线性模型加 batch GD，就是线性注意力

论文 Theorem 1：内环模型 $f(x)=Wx$，更新规则是 batch GD 且 $\eta=1/2$，
$W_0=0$。则对同一输入序列，公式 (5) 的输出与线性注意力相同。

证明只依赖第 5.3 节的损失。$W_0=0$ 时
$\nabla\ell(W_0;x_t)=-2(\theta_V x_t)(\theta_K x_t)^{\top}$。batch GD 累加后

$$
W_t = \sum_{s=1}^{t} (\theta_V x_s)(\theta_K x_s)^{\top}
$$

代入输出规则：

$$
z_t = \sum_{s=1}^{t} (\theta_V x_s)(\theta_K x_s)^{\top}(\theta_Q x_t)
$$

这就是去掉 softmax 的注意力。线性注意力因此是 TTT 家族里最简的一个点：线性
小模型 × batch GD。论文 Table 1 从这一点往上加零件（125M，The Pile 配方）：
原始线性注意力约 15.91；去掉归一化和特征展开后约 15.23；换 mini-batch
（$b=16$ 而不是 $b=T=2048$）是最大的一跳；再加上 LayerNorm 与残差、可学习
$W_0$、可学习 $\eta$，最终 TTT-Linear 为 11.09。本课不重跑这张表，只用它理
解「TTT-Linear 从线性注意力沿内环优化器走出来」。

Theorem 2 把非参数学习者（Nadaraya–Watson 核估计）对应回 softmax 注意力。隐
状态变回数据列表，更新是追加，输出是核扫描。课内不实现 Theorem 2，知道 TTT
的定义可以宽到包含注意力即可。Figure 9：参数学习者的 TTT 也是 RNN（状态大小
固定）；非参数 TTT 可以表示自注意力。

### 5.6 TTT-Linear、TTT-MLP，以及三个稳定化零件

第 2.7 节给出两个实例。TTT-Linear：$f_{\mathrm{lin}}(x)=Wx$，$W$ 是方阵。
TTT-MLP：两层，隐层宽度为输入的 4 倍，GELU，结构和 Transformer 的 FFN 同类。
为了内环稳定，真正拿去调用的是

$$
f(x) = x + \mathrm{LN}(f_{\mathrm{res}}(x))
$$

$f_{\mathrm{res}}$ 是上面的线性或 MLP。残差让 $W=0$ 时 $f$ 仍是恒等；LN 挡住
内环尺度漂移。

$W_0$ 在所有序列之间共享，后续 $W_1,\ldots,W_T$ 每条序列一份。论文不把 $W_0$
钉在 0，而把它当成外环参数 $\theta_{\mathrm{init}}$。单独学 $W_0$ 略伤
Table 1 的一行，但没有它后面几行训不稳。内环学习率也外环化：

$$
\eta(x) = \eta_{\mathrm{base}}\,\sigma(\theta_{\mathrm{lr}} \cdot x)
$$

$\eta_{\mathrm{base}}$ 对 TTT-Linear 取 1，对 TTT-MLP 取 0.1。$\eta(x)$ 也可
以读成 $\nabla\ell$ 的门。

骨干：作者试过把 TTT 层直接换进 Transformer 的注意力位置，也试过 Mamba 那种
带时间卷积的骨干。主实验默认 Mamba 骨干，图中标 `(M)`；消融里的 `(T)` 是
Transformer 骨干。时间卷积对表达力较弱的线性隐状态帮助更大，所以 TTT-Linear
从卷积里获益比 TTT-MLP 明显。本课短序列没有骨干，只跑一层 TTT-Linear。

TTT-MLP 在长上下文上论文认为潜力更大，但内存 I/O 仍是限制。不要在 CPU 机制
实验里改成 MLP 还指望看到论文 Figure 2 那种 32k 曲线。

### 5.7 和线性 RNN 对照：状态含义，不是速度

写一个最小线性 RNN，隐状态是向量 $h_t\in\mathbb{R}^{d}$：

$$
h_t = \sigma(W_{ss} h_{t-1} + W_{sx} x_t), \qquad
z_t = W_{zs} h_t + W_{zx} x_t
$$

$W_{ss},W_{sx},W_{zs},W_{zx}$ 是外环参数。前向时 $h_t$ 在变，$W_{\ast}$ 不变。
你不能对当前序列做 $\nabla_{W_{ss}}\|W_{ss}h-x\|^2$ 这种内环。若你强行对
$W_{ss}$ 做梯度，那已经不是线性 RNN 层，而是你手写的另一个 TTT 层。

对照表（本课交付物，范数用你的短序列填写）：

| 项目 | 线性 RNN | TTT-Linear |
|---|---|---|
| 隐状态 | 向量 $h_t\in\mathbb{R}^{d}$ | 矩阵 $W_t\in\mathbb{R}^{d\times d}$（再加 LN 统计量） |
| 更新规则 | 固定映射 $h_{t-1},x_t\mapsto h_t$ | $W_t=W_{t-1}-\eta\nabla\ell(W_{t-1};x_t)$ |
| 对当前序列做多步梯度 | 否 | 是（online 时每 token 一步；mini-batch 时每块一步） |
| 测试时慢权重 $\theta$ | 冻结 | 冻结 |
| 测试时快权重 | 无（$h$ 不是权重） | $W$ 每条序列一份 |
| 本课验证量 | $h_t$ 欧氏范数有限、形状是 `(d,)` | 各步 W 更新的 Frobenius 范数之和大于 0 |

类比到第 16 课的分流：线性 RNN 的 $h_t$ 像工作记忆，容量固定、读完即转写。
TTT 的 $W$ 像一块只对当前文档有效的快速权重，文档结束可以丢掉（推理缓存清
空），也可以留给下一层。它仍然不是把「公司里小王是谁」写进慢权重的离线微调。

### 5.8 论文规模上他们测了什么（本课只引用，不跑）

第 3 节实验全部在 JAX 代码库完成，协议尽量对齐 Mamba 论文：模型四档 125M、
350M、760M、1.3B（Mamba 对应 130M、370M、790M、1.4B）；The Pile 上 2k 与 8k
上下文；Books3 上 1k 到 32k 按 2 倍递增。训练按 Mamba 文中的 Chinchilla 配方，
每 batch 0.5M token。分词器他们统一用 Llama，不跟 Mamba 原文在 GPT-2 与
GPT-NeoX 之间切换。基线 Transformer 基于 Llama 结构。作者明确不做混合架构
（一层注意力加一层 TTT），以免评测变含糊。

短上下文（Pile，Figure 10）：2k 时 TTT-Linear（Mamba 骨干）、Mamba、
Transformer 三条线大致重叠；TTT-MLP 在大 FLOP 预算下略差，因为它每步更贵。
8k 时两条 TTT 线明显好过 Mamba，Transformer 单点困惑度仍强，但按 FLOPs 连线
不占优。长上下文（Books3，Figure 11）：32k 上 TTT-Linear 与 TTT-MLP 继续好过
Mamba；TTT-MLP 配 Transformer 骨干在 1.3B 附近追得很紧。作者的稳健观察是：
上下文越长，TTT 相对 Mamba 的优势越宽。

这些数字不能从本课 16 个 token 的范数推出来。短序列只能证明「$W$ 作为隐状态
会在内环里动」，证明不了「动了之后 32k 困惑度会下降」。把 Figure 10 抄进你
的 `result.json` 是档次错误。

论文自己列的限制也要读：框架能实例化任意网络当隐状态，但墙钟时间仍可能很高；
TTT-MLP 的内存 I/O 没有解决。2025 年的 LaCT 正是冲着利用率去的，第 11 节再
讲。

## 6. 源码导读

克隆后先读 README，再读一个文件。PyTorch 仓库几乎把层定义收在
[`ttt.py`](https://github.com/test-time-training/ttt-lm-pytorch/blob/main/ttt.py)。
当前目录还能看到 `TTTForCausalLM`、`TTTConfig`、`TTT_STANDARD_CONFIGS` 这些
Hugging Face 风格入口。JAX 仓库才是论文实验的训练入口。

读之前记住 README 原句：这份代码是纯 PyTorch、没有系统优化，不建议用来训练，
设备 batch 小时尤其慢。训练或复现论文数字去
[ttt-lm-jax](https://github.com/test-time-training/ttt-lm-jax)；推理 kernel 或
速度对照去 [ttt-lm-kernels](https://github.com/test-time-training/ttt-lm-kernels)。

| 位置 | 带着什么问题读 |
|---|---|
| `ttt-lm-pytorch/README.md` | 安装依赖是哪一条？Quick Start 是否从随机初始化的 `TTTConfig()` 建模型？有没有提供预训练权重下载？ |
| `ttt.py` 中 `TTTBase` / `TTTLinear` / `TTTMLP` | 内环小模型如何切换？LN 和残差加在哪？ |
| `ttt.py` 中 `TTTCache` | 推理时缓存的是 $W$ 还是梯度和最后一块隐状态？和 KV cache 差在哪？ |
| `ttt.py` 中 mini-batch / gradient checkpoint 注释 | 是否把 4 个 mini-batch 打成一组 checkpoint？这对应论文哪一节的并行？ |
| `TTTConfig` / `TTT_STANDARD_CONFIGS` | `'1b'` 这种键改的是层数和宽度，还是内环 $b$？ |
| `ttt-lm-jax/train.py` | `dataset_path`、`mesh_dim` 在哪；本课是否需要碰它们（答案：不需要） |
| `ttt-lm-jax/ttt/README.md` | JAX 侧如何描述 TTT 层配置，和 PyTorch `TTTConfig` 能否对上 |

README 的 Quick Start（写课前已打开仓库页面核对）结构如下。先装依赖：

```bash
pip install "transformers[torch]"
```

仓库根目录下，模型定义按 Hugging Face 接口露出。下面这段是 README 的用法，
不是本课要跑的训练脚本：

```python
from transformers import AutoTokenizer
from ttt import TTTForCausalLM, TTTConfig, TTT_STANDARD_CONFIGS

configuration = TTTConfig()
model = TTTForCausalLM(configuration)
model.eval()
configuration = model.config
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
```

README 注释写明：`TTTConfig(**TTT_STANDARD_CONFIGS['1b'])` 与默认 `TTTConfig()`
等价。本课短序列不需要 1B 配置。把宽度降到 32 或 64，随机 `input_ids` 走
`model(...)`，检查输出 logits 形状，即可确认前向能跑。Llama-2 分词器拉不下
时，跳过 `generate` 示例，不影响第 9 节验收。

在 `ttt.py` 里按符号搜，不要按文件名幻想第二个 `modeling_ttt.py`。写课前打开
的 GitHub 目录里，PyTorch 仓库的层实现就收在这一个 `ttt.py`。你应能找到：

- `class TTTLinear(TTTBase)`：线性内环。注释里写过把 4 个 mini-batch 打成一组
  gradient checkpoint 以省显存，这是工程折中，不是论文公式的一部分。
- `class TTTMLP(TTTBase)`：两层 MLP 内环，对应第 5.6 节。
- `TTTCache`：保存最近的隐状态和梯度，给自回归解码用。它扮演的角色类似
  KV cache，内容却是 $W$ 而不是键值列表。
- `TTTForCausalLM` / `TTTConfig`：把 TTT 层嵌进因果语言模型外壳，接口模仿
  Hugging Face Transformers。

核对清单，结论以你 clone 下来的 commit 为准：

- `TTTLinear` 的内环权重在前向里更新，不在外层 `Adam.step()` 里更新。
- `TTTCache` 跨解码步保存快速权重，序列边界应重置。
- 仓库没有把 The Pile 训练脚本放进这个 PyTorch 目录。看到有人用它报 1.3B
  分数，那是另一份 JAX 运行，不能写进本课实验记录。

## 7. 实验

三层都做。浏览器先预测再运行；CPU 实验用断言钉死「$W$ 动了」；锚定仓库只跑
模型定义和短序列内环，不训语言模型。

### Step 0: 浏览器实验「内环台阶」

打开本课页面的交互实验（课程里登记为 `lab-17-inner-loop`）。一段 16 个
token 的短序列，矩阵 $W$ 画成格子。每来一个 token，系统按论文公式 (2) 做一
次梯度更新，点亮被当前 token 碰到的那一条（外积 $(Wx-x)x^{\top}$：若 $x$
接近 one-hot，主要改对应的那一列；界面把它画成一行台阶，避免和「注意力行」
搞混）。

先预测再运行。预测题：第 5 个 token 更新之后，被点亮的台阶是否就是第 5 条，
而不是「整张 $W$ 均匀变淡」或「只有对角线变」。改序列或学习率会作废上次运
行，必须重新预测。预测正确且跑过一遍，这一层才算过关。

这一层证明的是形状和因果：哪一个 token 写入哪一条。它不证明困惑度，也不证
明 mini-batch dual form 的速度。

建议在运行前把预测写在纸上，三选一：

1. 第 5 步只点亮第 5 条台阶（当前 token 对应的槽）。
2. 前 5 条台阶一起均匀变亮（像注意力把前缀全加上）。
3. 整张 $W$ 同时变，看不出和 token 下标的对应。

合格答案是 1，带一个限定：若 $x_t$ 不是严格 one-hot，邻槽会有漏光，但最亮
的仍应是当前 token 那一条。选 2 的人把 TTT 当成了注意力；选 3 的人把学习率
加在了整张 $W$ 的权重衰减上，而不是外积更新。

### Step 1: 进入课内实验目录

```bash
cd experiments
```

### Step 2: CPU 机制实验

```bash
python3 run.py run 17
```

入口是 `experiments/src/learn_cl_experiments/lessons/lesson_17.py`，结果写到
`artifacts/lesson17/result.json`。固定种子。`python3 run.py run 17` 现在应当
全绿。`checks` 六条：`inner_loop_moves_W`、`more_steps_larger_delta`、
`zero_lr_freezes_W`、`reconstruction_loss_drops`、`rnn_state_is_vector`、
`ttt_state_is_matrix`。

本机一次运行：$\|\Delta W\|_F=1.003$（只走 1 步是 0.117，学习率为 0 时是 0）；
重建损失从 0.621 降到 0.130。换机器会变，方向不应变。对照臂是同规模线性
RNN：隐状态是长度为 4 的向量，前向路径里没有对当前序列的多步 $\nabla_W\ell$。
TTT 的 $W$ 是 $4\times 4$ 矩阵。

建议同时看这些字段（名称以 `result.json` 为准）：

| 字段意图 | 本机一次运行 | 合格方向 |
|---|---|---|
| `delta_w_full`（全程 Frobenius） | 1.003 | 大于 0，且大于只走 1 步的 0.117 |
| 线性 RNN 的 $h_t$ | 长度 4 的向量 | 不拿来和矩阵范数比大小 |
| 隐状态形状 | TTT 为 $4\times 4$ | RNN 最后一维等于 $d$、不是 $d\times d$ |

这一层证明机制，不是 Pile 上的 TTT-Linear 语言模型。

### Step 3: 手写 primal 内环（可选，用来看懂 Step 2）

下面是论文公式 (2)+(3) 在 $\tilde{x}_t=x_t$、无三视图时的最小实现，方便你
对照 `result.json`。它不是官方仓库代码，也不是 TTT-Linear 的完整版（缺 LN、
残差、可学习 $\eta$）。

```python
import torch

torch.manual_seed(0)
T, d, eta = 16, 8, 0.1
x = torch.randn(T, d)
W = torch.zeros(d, d)
norms = []
for t in range(T):
    xt = x[t]
    W = W.detach().requires_grad_(True)
    loss = (W @ xt - xt).pow(2).sum()
    (grad,) = torch.autograd.grad(loss, W)
    W = (W - eta * grad).detach()
    norms.append(float(grad.norm()))
print(sum(norms) > 0, [round(v, 4) for v in norms])
```

预期：打印的第一个值是 `True`，后面 16 个范数里多数大于 0。把 `eta` 改成 0，
范数仍可能大于 0（那是梯度范数），但 $W$ 不再变；此时应改打印
`\|(W_{\mathrm{new}}-W_{\mathrm{old}})\|_F`，它必须变成 0。这就是验收时
「更新范数」和「梯度范数」不能混用的原因。

### Step 4: 线性 RNN 对照

同一批 $x_t$，跑向量隐状态。不要对 $W_{ss}$ 做内环。

```python
import torch

torch.manual_seed(0)
T, d = 16, 8
x = torch.randn(T, d)
W_ss = torch.randn(d, d) / d**0.5
W_sx = torch.randn(d, d) / d**0.5
h = torch.zeros(d)
hs = []
for t in range(T):
    h = torch.tanh(W_ss @ h + W_sx @ x[t])
    hs.append(h.norm().item())
print(h.shape, hs[-1])
```

预期：`h.shape` 是 `torch.Size([8])`，不是 `[8, 8]`。循环体内没有
`autograd.grad`。若你在这里对 `W_ss` 求梯度，对照就失效了。

### Step 5: 克隆官方 PyTorch 定义

换一个干净目录，不要和课内 `experiments/` 混装依赖。

```bash
git clone https://github.com/test-time-training/ttt-lm-pytorch.git
```

```bash
pip install "transformers[torch]"
```

用 `TTTConfig` 把宽度和层数改小（读 `TTTConfig` 字段，以仓库为准；常见是
`hidden_size` 与 `num_hidden_layers`）。随机整数 token、长度 16，
`model.eval()` 后做一次前向。预期：logits 的时间维等于 16，没有异常 NaN。

官方 README 未把小权重当作这个 PyTorch 仓库的默认交付。本课按课程蓝图标
「只跑机制」：不做 needle、不做长拷贝评测。若你后来在 Hugging Face 上找到
作者组织下的权重，那是加分项，写进笔记时标明权重名、commit 和许可，不要写
成「第 17 课复现了 Pile 125M」。

### Step 6: 对照阅读 JAX 仓库（只读）

```bash
git clone https://github.com/test-time-training/ttt-lm-jax.git
```

打开 `train.py` 和 `ttt/README.md`。记下三件事：训练入口在 JAX 不在
PyTorch；`dataset_path` 指向预处理后的 Pile/Books 目录；`mesh_dim` 控制切分。
本课到此停止，不下载数据、不提交训练作业。

### Step 7: 留下交付物

在实验笔记里保存四行以上：

```text
日期与机器
ttt-lm-pytorch 的 commit
python3 run.py run 17 的 checks
逐步 W 更新范数（可从 result.json 抄）以及 RNN 隐状态形状
```

把对照表（第 5.7 节那张）用你的数字填完。这是本课验收要看的产物。

## 8. 配置与预算

| 档 | 序列 | 模型 | 时间 | 用途 |
|---|---|---|---|---|
| 浏览器 | 16 token | 二维示意 $W$ | 10 分钟 | 预测写入位置 |
| CPU 机制 | 16 到 64、宽 8 到 32 | 一层 TTT-Linear primal + 线性 RNN | 数秒 | 断言 $W$ 范数 |
| 仓库冒烟 | 长度 16，缩小 `TTTConfig` | 官方 `TTTForCausalLM` 随机初始化 | 装依赖为主，前向秒级 | 确认定义能 import |
| 论文配方（本课不跑） | The Pile 2k/8k，Books3 到 32k | 125M 到 1.3B，Chinchilla 风格，JAX | 多卡、按 JAX README | 只讲 |

主线按档 C 的精神写命令，但第 17 课的机制实验必须能在档 A（Mac / CPU）完成。
不要把整课锁在 1.3B 上。内环学习率扫一遍（$0, 10^{-2}, 10^{-1}, 1$）只增加
分钟级 CPU，可以看到 $\eta=0$ 时更新范数为 0，这是最便宜的负对照。

## 9. 验收

- [ ] 白纸画出 Figure 3 那张三件套，RNN / 注意力 / TTT 各填一行。
- [ ] 默写公式 (2)、(4)、(5)，并标出 $W$ 是内环、$\theta_K,\theta_Q,\theta_V$
      是外环。
- [ ] 浏览器「内环台阶」先预测再运行，预测正确。
- [ ] `python3 run.py run 17` 的 `checks` 全为真。本机一次运行
      $\|\Delta W\|_F=1.003$（1 步 0.117，学习率为 0 时为 0），损失从 0.621
      降到 0.130。换机器会变，方向不应变。不是 Pile 上的 TTT-Linear 语言模型。
- [ ] 对照表填完：RNN 隐状态形状是向量，TTT 的 $W$ 是矩阵。
- [ ] 打开过 `ttt.py`，能指出 `TTTLinear` 与 `TTTCache` 对应论文哪一节。
- [ ] 口头回答：为什么本课不宣称复现了 TTT-Linear 在 The Pile 上的困惑度。
- [ ] 笔记里有仓库 commit 和命令，三个月后能复原这次短序列运行。

## 10. 排错

| 症状 | 最可能的原因 | 怎么确认 | 怎么修 |
|---|---|---|---|
| 更新范数恒为 0 | $\eta=0$，或 $W$ 被当成外环冻住，或 loss 没连到 $W$ | 打印 `W.requires_grad`、`loss.grad_fn`、$\eta$ | 按 Step 3 最小实现重跑；不要在 `torch.no_grad()` 里做内环 |
| 更新范数爆炸 | $\eta$ 过大，或没 LN/残差 | 看逐步范数是否单调放大 | 短序列先把 $\eta$ 降到 $10^{-2}$；完整 TTT-Linear 必须加 LN 与残差 |
| RNN 对照也出现矩阵状态 | 你把 $h_t$ 写成了 $d\times d$，或对 $W_{ss}$ 做了内环 | 打印 `h.shape` | 回到第 5.7 节公式，隐状态保持 `(d,)` |
| `import ttt` 失败 | 没在仓库根目录，或没装 `transformers[torch]` | `python -c "import ttt"` | 在 clone 目录下装 README 那一条依赖 |
| 一 clone 就想跑训练 | 看错仓库 | README 是否出现 do not recommend training | 训练去 JAX；本课停在模型定义 |
| Llama 分词器 401/许可 | README 示例依赖 Llama-2 | 报错含 gated repo | 跳过 tokenizer，用随机 `input_ids` |
| 短序列范数写成 Pile 分数 | 档次写错 | 笔记里是否出现 11.09 却没有 JAX 训练日志 | 删掉，11.09 只能当论文 Table 1 的引用 |
| dual form 和 primal 对不上 | mask 方向或系数 2 丢了 | 同一输入比较 $z_t$ | 教学用 primal；要对齐再逐项对照论文 (7)(8) |
| `inner_loop_moves_W` 或 `zero_lr_freezes_W` 为假 | 内环没连到 $W$，或把梯度范数当成更新范数 | 看 `delta_w_full`、`delta_w_zero_lr` | 学习率非 0 时更新范数应大于 0；学习率为 0 时必须是 0 |

## 11. 前沿与改造

2024 这篇 TTT 层之后，测试时训练沿两条公开线往前走。一条改**层内**怎么更新
$W$：更大的块、更非线性的快权重、更强的测试时优化器。Zhang 等人 2025 的
*Test-Time Training Done Right*（[arXiv:2505.23884](https://arxiv.org/abs/2505.23884)）
把 mini-batch 从论文的 16 / 64 拉到 2K 到 1M，称为 LaCT（Large Chunk TTT）。
他们给出的原因是：小块导致 TTT 层的 FLOPs 利用率常常低于 5%，非线性大状态
难以写进自定义 kernel；大块把利用率抬到可在纯 PyTorch 里到约 70%（A100，论
文 Figure 1），并把非线性状态扩到约占模型参数 40%。语言建模里他们用 2K 或
4K 的块加滑动窗口注意力；新视角合成里块可以是整段图像 token。本课的 16
token 内环证明 $W$ 会动，证明不了大块利用率。

另一条改**损失写在哪**。Tandon 等人 2025 的 *End-to-End Test-Time Training for
Long Context*（[arXiv:2512.23675](https://arxiv.org/abs/2512.23675)）不再用层
内 KV 绑定 MSE，而在测试时对整网做下一 token 预测，训练时用元学习优化「TTT
之后」的损失。架构是带滑动窗口的 Transformer，TTT 只更新最后 1/4 块里的
MLP。论文 Figure 1：3B、164B token 时，TTT-E2E 的损失随上下文长度的走势与全
注意力同方向，128K 上解码延迟仍接近常数，约为全注意力的 $1/2.7$。代码在
[test-time-training/e2e](https://github.com/test-time-training/e2e)。本课不训
3B。2026 年还改快权重放哪、内环监督写什么：把现成 MLP 下投影当场更新、把值
目标改成下一位置隐状态或长短窗口蒸馏、先圈证据再训，以及把内环拆成可组合
模块。条目和阅读问题在第 12 节。

和本课的差距具体有三处：我们用 primal、一块 16 个 token、一层线性 $W$；前沿
用大块或整网 CE、Muon 或元学习、窗口注意力补块内因果。慢权重仍然冻结，快速
权重仍然按序列重置。这还不是梁文峰说的那种跨月上岗（转写未获确认），只是把
「读这段时学习」做成了层。

动手改造（选做，失败标准写死）：

1. 给 Step 3 加上论文 2.7 节的 LN 与残差，再扫 $\eta\in\{0.01,0.1,1\}$。
   预算：CPU 一小时。预期：同样 $\eta$ 下带 LN 的逐步范数更稳。失败：出现
   NaN，或 $\eta=0$ 时范数仍大于 0（更新规则写错）。
2. 把 $b$ 从 1 改到 4、16，比较 $\|W_T-W_0\|_F$ 和逐步范数之和。预算：一
   小时。预期：$b$ 变大，步数变少，总位移通常变小。失败：三种 $b$ 的 $W_T$
   完全相同（你其实一直在用 batch GD 对 $W_0$ 求导）。
3. 在官方 `TTTLinear` 上对长度 16 的随机输入，取出内环 $W$ 在逐步前后的差。
   预算：装好仓库之后一小时。预期：差的 Frobenius 范数 $>0$。失败：差为 0，
   说明取到了外环参数或 `eval` 路径跳过了内环，对照 `ttt.py` 的 cache 逻辑。
4. 只读 LaCT 论文第 3.1 节，把「先 apply 再 update」和「先 update 再 apply」
   画成两种块状因果遮罩，标出哪一种会在块内看到未来。预算：阅读两小时，不
   训练。预期：你能指出语言模型必须用移位后的块状遮罩。失败：两种遮罩画成
   一样。

顺手复现映射：本课没有列入课程 §4 的五项正式复现。不要在标题或笔记里写
「复现 Sun et al. 2024」。方向性的机制断言只有一条：$W$ 在内环后必须动。

下一课 [第 18 课](18_titans_surprise.md) 在同一条测试时更新的路上加门控。
TTT 层默认每步都更新 $W$；Titans 用惊讶（当前损失或梯度范数）决定写入幅度，
稀有 token 多写，常见 token 少写。本课先保证门还没装上时，$W$ 确实会动。把范数曲线和对照表收进笔记，第 18
课会在同一条合成序列上比较「每步都写」和「按惊讶写」的写入幅度。

## 12. 论文与延伸

1. Sun, Li, Dalal 等, 2024, *Learning to (Learn at Test Time): RNNs with
   Expressive Hidden States*,
   [arXiv:2407.04620](https://arxiv.org/abs/2407.04620)（v4，2025-08-31）。
   贡献：把隐状态做成可训练的小模型，读测试序列时也走自监督梯度。
   机制：内环对当前 token 做多视图重建 $\|f(\theta_K x;W)-\theta_V x\|^2$，
   再 $W_t=W_{t-1}-\eta\nabla\ell$。TTT-Linear 的 $W$ 是方阵，TTT-MLP 是两
   层、隐层 $4\times$、GELU，都加 LN 和残差。主实验 $b=16$ 的 mini-batch，
   对偶形式避免物化每步 $G_t$。Theorem 1：线性模型加 batch GD、$W_0=0$ 时
   等于线性注意力。125M 到 1.3B 上，Mamba 约 16k 之后按位置困惑度走平，
   TTT 仍随更多 token 下降。官方 PyTorch 仓库 README 写明不建议用这份代码
   报大规模训练分数。
   和本课：`inner_loop_moves_W`、`more_steps_larger_delta`、
   `zero_lr_freezes_W`、`reconstruction_loss_drops` 对上逐步 MSE 更新；
   `ttt_state_is_matrix` 对上 $W\in\mathbb{R}^{d\times d}$，
   `rnn_state_is_vector` 对上同规模 RNN。Figure 2 的 32k 曲线和 Table 1
   的 11.09 本课答不了。
   阅读问题：本课 16 步的 $\|\Delta W\|_F$ 是否大于只走 1 步？学习率改成 0
   时 `zero_lr_freezes_W` 该是真还是假？

2. Zhang, Bi, Hong 等, 2025, *Test-Time Training Done Right*,
   [arXiv:2505.23884](https://arxiv.org/abs/2505.23884)。
   贡献：把 TTT 的更新块拉到 2K 至 1M token，非线性快权重可到模型参数约
   40%。
   机制：小块（每 16 或 64 个 token 更新）让 TTT 层 FLOPs 利用率常低于
   5%。大块加窗口注意力，纯 PyTorch 就能跑，并可接 Muon。语言建模必须用
   移位后的块状因果，否则块内会看到未来。14B 自回归视频扩到 56K token；
   新视角合成上下文到 1M。
   和本课：本课 16 个 token、逐步更新，正是他们批评的小块。
   `inner_loop_moves_W` 只证明 $W$ 会动，证明不了利用率。
   阅读问题：若把本课 16 个 token 当成一整块、先 apply 再 update，块内还
   保留因果吗？本课实验是逐步更新，答不了块状遮罩，只能答「本课每步都
   先算损失再写 $W$」。

3. Tandon, Dalal, Li 等, 2025, *End-to-End Test-Time Training for Long
   Context*, [arXiv:2512.23675](https://arxiv.org/abs/2512.23675)。
   贡献：测试时用下一 token 交叉熵，训练时用元学习优化「TTT 之后」的损失。
   机制：骨架是滑动窗口 Transformer，只更新最后 1/4 块里的 MLP。3B、164B
   token 时，损失随上下文长度的走势与全注意力同方向；128K 解码延迟仍接
   近常数，约为全注意力的 $1/2.7$。代码在
   [test-time-training/e2e](https://github.com/test-time-training/e2e)。
   和本课：本课损失是层内 MSE，E2E 损失在网络末端。
   `reconstruction_loss_drops` 推不出他们 Figure 1。
   阅读问题：本课 $W$ 更新范数大于 0，能不能推出「上下文变长、测试损失
   仍下降」？答不了，因为本课没有外环元学习和下一 token 损失。

4. Feng, Luo, Hua 等, 2026, *In-Place Test-Time Training*,
   [arXiv:2604.06169](https://arxiv.org/abs/2604.06169)（ICLR 2026 Oral）。
   贡献：把现成 LLM 的 MLP 最后一层投影当场当快权重，不必从零改骨架。
   机制：通用重建目标换成对准下一 token 预测的目标，按块更新，可与上下
   文并行兼容。摘要写 4B 模型作为 in-place 增强，在最长 128k 的任务上有
   效；从头预训练时也优于对照的 TTT 写法。代码在
   [ByteDance-Seed/In-Place-TTT](https://github.com/ByteDance-Seed/In-Place-TTT)。
   和本课：本课的 $W$ 是单独的 $d\times d$ 矩阵，不是 MLP 下投影。
   `ttt_state_is_matrix` 能看见「有一块在动的权重」，看不见「动的是哪一
   层投影」。
   阅读问题：若把本课的 $W$ 理解成「整网最后一层」，`zero_lr_freezes_W`
   还能保证什么？本课实验答不了 in-place 插入，只能答学习率为 0 时这块
   矩阵不动。

5. Ouyang, Cai & Hu, 2026, *Test-Time Training with Next-Token
   Prediction*, [arXiv:2606.21803](https://arxiv.org/abs/2606.21803)。
   贡献：快权重的写入目标改成同一层下一位置的上下文隐状态，对准下一
   token 预测。
   机制：键是当前 gated MLP 激活，值是下一位置隐状态再经一层线性。训练
   时按块做因果前缀和；推理时对 prompt 做一次 ridge 闭式写入。RULER
   Full-13（4k/8k/16k/32k 平均）上 Llama-3.1-8B +3.9、Mistral-7B-v0.3
   +3.0、Qwen3-4B +4.1、Qwen3-0.6B +2.9；LongBench-v2 上 Llama +5.6、
   Mistral +3.7。
   和本课：本课目标是重建当前向量，不是下一位置隐状态。
   `reconstruction_loss_drops` 测的是当前 MSE。
   阅读问题：本课若把 target 改成「下一个 token 的向量」，
   `reconstruction_loss_drops` 还保证什么？本课没有下一位置，答不了
   NTP 目标，只能答当前重建是否下降。

6. Tang, Qin, Pan 等, 2026, *Modular TTT: Rethinking Test-Time Training
   as Composable Modules*,
   [arXiv:2608.07110](https://arxiv.org/abs/2608.07110)。
   贡献：把内环表示成有向无环图，快权重网络、损失、学习率、权重衰减、
   归一化都是可替换模块。
   机制：自动把 train-view 前向、train-view 反向、因果 query-view 拼成
   整图。消融：小学习率初始化、权重衰减、单层非线性有帮助；MSE 和内积
   损失相近；更深快权重和归一化容易激活过大；残差和门控收益不明显。
   410M 与 1.45B、100B token 上训练损失和基准与 Gated DeltaNet 相当。
   代码在
   [ByteDance-Seed/Modular-TTT](https://github.com/ByteDance-Seed/Modular-TTT)。
   和本课：本课把损失、学习率、线性 $W$ 写死。你可以改 `lr` 看
   `zero_lr_freezes_W`，改不了他们的模块图。
   阅读问题：把本课学习率从 0.08 改到 0，哪条 check 必须翻成真？这对应
   他们把学习率当成可替换模块的哪一句？

7. Wang, Dang, Zhu 等, 2026, *Learning What to Remember: Test-Time
   Training via Context Distillation*,
   [arXiv:2608.01672](https://arxiv.org/abs/2608.01672)。
   贡献：用长窗口教师监督短窗口学生的快权重，按「对未来预测有没有用」
   分配有限记忆。
   机制：教师与学生的隐状态差当稠密自监督信号。落地版 IP-TTCD 仍改
   MLP 下投影。从头预训练优于 DeltaNet、Gated DeltaNet、滑动窗口注意
   力和 TTT；也可给已预训练 Transformer 做轻量增强。760M 上 RULER
   NIAH 32K 从 IP-TTT 的 9.29 升到 21.96。
   和本课：本课没有双窗口，也没有蒸馏残差。`inner_loop_moves_W` 只说明
   有梯度。
   阅读问题：本课重建损失下降，能不能说明 $W$ 记住的是对未来预测有用的
   信息？答不了，因为没有教师窗口。

8. Zhu, Xu, Wei 等, 2026, *Self-Guided Test-Time Training for
   Long-Context LLMs*,
   [arXiv:2607.09415](https://arxiv.org/abs/2607.09415)。
   贡献：先让模型自己圈出与问题有关的证据段，再用标准语言模型损失只在
   这些段上做 TTT。
   机制：全上下文或随机段噪声大。LongBench-v2 上，Qwen3-4B-Thinking 的
   随机段 TTT 从 40.4 降到 38.9，oracle 段升到 45.9。S-TTT 在
   LongBench-v2 和 LongBench-Pro 上对 Qwen3-4B-Thinking 与
   Llama-3.1-8B-Instruct 相对提升最高约 15%。用 LoRA，每例 16 步；最
   后仍用全文解码。
   和本课：本课 16 个 token 全部写入，没有选段。
   `more_steps_larger_delta` 与选段无关。
   阅读问题：本课每步都写，对应他们说的「随机段会掺噪声」吗？本课没有
   问题、也没有段，答不了选段质量，只能答全部写入时 $W$ 会动。

9. Behrouz, Zhong & Mirrokni, 2025, *Titans: Learning to Memorize at
   Test Time*, [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)。
   贡献：用惊讶梯度写长期记忆，并给出 MAC / MAG / MAL。下一课主文，本
   课只预告。
   机制：联想 MSE 的梯度当瞬时惊讶，动量当过去惊讶，再加遗忘门。注意
   力当短期，神经记忆当长期。摘要写可扩到大于 2M 上下文。
   和本课：本课每步都写，没有门。`inner_loop_moves_W` 推不出「稀有
   token 写得更多」。
   阅读问题：本课 $W$ 更新范数大于 0，能不能推出稀有 token 写入幅度更
   大？答不了。第 18 课的 `rare_write_exceeds_common` 才测这件事。

10. Sun, Wang, Liu 等, 2020, *Test-Time Training with Self-Supervision
    for Generalization under Distribution Shifts*, ICML,
    [arXiv:1909.13231](https://arxiv.org/abs/1909.13231)。
    贡献：测试样本上的自监督微调，针对分布偏移。机制发明处，不是本课
    主阅读。
    机制：一张未标注测试图变成旋转预测等问题，先更新参数再分类，也可
    以接到在线数据流。对象是图像分布偏移，不是序列里的 token。
    和本课：同名不同对象。`ttt_state_is_matrix` 在 2020 设定里没有意
    义，那时隐状态不是矩阵 $W$。
    阅读问题：本课哪条 check 在「一张图、一个内环」的 2020 设定里没有
    对应物？用 `ttt_state_is_matrix` 或 `rnn_state_is_vector` 回答。

第 18 课把「每步都写」改成「惊讶才写」。你会见到动量形式的惊讶 $S_t$、遗忘
门 $\alpha_t$，以及 MAC / MAG / MAL 三种把长期记忆接进注意力的方法。本课的
$W$ 还在，只是更新规则要乘上一扇门。
