---
id: 08_gem_gdumb
title: "梯度不许踩旧任务，以及那个尴尬的基线"
summary: "若只把样本存下来、每个阶段从头训，分数往往也不差。这说明什么？"
unit: toolkit
play_tools: []
checkpoints:
  - "三方法对照表。"
  - "「你的设定会不会被 GDumb 打脸」检查清单。"
  - "论文复现 #2。"
---

# 第 08 课：梯度不许踩旧任务，以及那个尴尬的基线

> 类型：复现 #2（方向性）+ 机制实验（梯度投影）<br>
> 建议周期：3-4 天<br>
> 硬件：CPU 可完成投影机制、Split MNIST 小缓冲对照；单卡能把同一协议扩到 CIFAR-10<br>
> 锚定仓库：[ContinualAI/avalanche](https://github.com/ContinualAI/avalanche) 的 `GEM` / `AGEM`；[aimagelab/mammoth](https://github.com/aimagelab/mammoth) 的 `gdumb`、`agem`、`derpp`<br>
> 产物：A-GEM、DER++、GDumb 三方法对照表 + 「你的设定会不会被 GDumb 打脸」检查清单 + 复现 #2 记录

## 1. 这一课做什么

第二幕要收官。[第 05 课](05_ewc_regularization.md) 给重要权重量弹簧，[第 06 课](06_replay_der.md) 把旧样本放进背包一起训，[第 07 课](07_architecture_prompts.md) 把新知识写到新柱、空位或 prompt 上，旧权重可以硬冻结。三条路都还没回答一个更窄的问题：**如果必须改同一组权重，有没有办法保证这一步更新不把旧任务的损失推高？**

GEM（梯度情景记忆：用旧样本的梯度当护栏，把当前更新投影到「旧损失不升高」的一侧）把这句话写成带不等式约束的二次规划。A-GEM 把「每个旧任务一道护栏」收成「所有旧样本平均出一道护栏」，投影变成一次向量减法。它们仍然要用缓冲，但缓冲的用法和第 06 课不同：DER++ 把旧图混进损失里再训；GEM 主要拿旧图算约束，当前损失才是目标。

同一节课必须请出 GDumb（贪心采样加笨学习器：缓冲里尽量把各类存均衡，每个阶段用缓冲从头训一个模型）。Prabhu、Torr、Dokania 在 ECCV 2020 的实验里，这个「不算持续学习」的基线，在多种公开协议上接近或超过当时的 SOTA，包括 A-GEM。课程把它列为正式复现 #2：在小缓冲、类增量 MNIST 上，GDumb 不低于 A-GEM，或你写明本设定下的反例和缓冲大小。

没有这课会缺什么：你会把「平均准确率第一」读成方法赢了。[第 03 课](03_cl_evaluation.md) 已经预告过，只报最终平均准确率会把「根本没在线学、只会拿背包重训」的方法夸成好方法。本课把那句预告跑成数字。

做完你手里有三样东西：一张同一协议的三方法表；一份检查清单，用来判断自己以后的设定会不会被 GDumb 打脸；一段复现记录，方向与 GDumb 论文一致，或诚实写下不一致时的缓冲和任务切分。

本课仍在主干循环里换零件：新经验进来之后，「怎么写」变成「沿约束投影」或「先存后重训」。写完立刻测新任务会了没、旧任务还在不在。GDumb 提醒你第三件事：有些协议里，这两项用离线重训就能同时很高，因为它的任务边界太干净。

术语速查：

| 术语 | 一句解释 |
|---|---|
| 情景记忆（episodic memory） | GEM / A-GEM 里按任务或按容量存下的一小撮旧样本，用来算旧梯度，不是拿来当主训练集 |
| 约束梯度 | 当前更新必须落在「不增加旧损失」的半平面里 |
| 二次规划（QP） | GEM 每一步要解的小优化：找一个离当前梯度最近、又满足所有旧任务不等式的向量 |
| 投影 | A-GEM 在违反约束时，把当前梯度减掉它在旧梯度方向上的分量 |
| 参考梯度 $g_{\mathrm{ref}}$ | A-GEM 从记忆里抽一批旧样本算出的平均梯度 |
| 向后迁移（BWT） | 第 03 课：学新任务之后，旧任务变好还是变差；GEM 论文允许正向 BWT |
| GDumb | 贪心均衡采样 + 用缓冲从头训练；几乎不编码任务边界 |
| 贪心均衡采样器 | 新类来了就给它腾格子，从目前样本最多的类里随机丢掉一个 |
| 复现 #2 | 本课正式复现：小缓冲类增量 MNIST 上 GDumb 不低于 A-GEM，或记录反例 |
| 任务边界 | 数据流在何时从一类集合切到另一类集合；干净的硬切分让 i.i.d. 重训特别强 |

## 2. 问题

GEM 要同时做两件事：新任务的损失往下降，旧任务的损失不许往上升。只降新损失，就是 naive 微调，第 01 课已经看过遗忘。把旧损失加进目标当正则，接近第 05 课，但那样会禁止旧损失下降，也就禁止正向的向后迁移。Lopez-Paz 和 Ranzato 的选择是不等式：旧损失可以降，不能升。

把「不能升」写成对每个旧任务一条约束，每一步都要解一个二次规划。任务一多，每条约束还要算一遍全参数梯度，墙钟时间会炸。A-GEM 问：能不能只留一条平均约束，投影用闭式公式算完？可以。代价是失去「最坏那个旧任务」的逐任务保证，换来接近 EWC 的计算和内存。

然后 GDumb 进场。它不投影，不蒸馏，不估 Fisher。流过的样本按类均衡塞进容量为 $k$ 的缓冲；需要预测时，用缓冲里的数据从头训练一个网络。若测试协议其实是「任务边界清晰、类数固定、缓冲里的样本几乎是该类的 i.i.d. 子集」，这就是在缓冲上做普通监督学习。GDumb 论文表 4 的 A2 设定里，MNIST、$k=500$：A-GEM 29.0 ± 5.3，GDumb 91.9 ± 0.5。CIFAR-10、$k=500$：A-GEM 18.5 ± 0.6，GDumb 45.8 ± 0.9。这些数字不是课内必复现的绝对值，但方向必须面对。

核心问题因此有两层。第一层是机制：投影公式对不对，投影之后更新还指不指向新任务。第二层是评测：若 GDumb 赢了，你的方法到底在持续学习，还是在一个「存得下、切得干净」的作业上绕远路。

GDumb 赢了，并不等于持续学习无用。它说明很多实验把问题简化到了「任务硬切分 + 固定类数 + 缓冲可重训」，i.i.d. 重训已经很强。真实部署里任务边界会糊、类会增长、旧数据不能整包重训、技能和偏好也不是一张图一个标签。第 22 课会把非平稳经验再摊开。本课的任务是：先在经典协议上被打脸，再学会检查自己的设定。

类比：GEM 像在斜坡上走路，脚下画了一条线，「不许往旧房子的方向滑」；A-GEM 把好多条线平均成一条。类比失效处：线是用缓冲里那几张图估的，不是真正的旧任务损失；缓冲不代表旧分布时，你以为没踩线，旧考试照样崩。GDumb 像每次考试前把书包倒出来重新上学。类比失效处：书包容量固定，类一多每类只剩几张图；它也没有在线学习速度，第 08 课协议若禁止「任务结束才重训」，GDumb 的用法要改写。

## 3. 准备

- 第 03 课的准确率矩阵、$R_{i,j}$、ACC、BWT、FWT。GEM 论文就是这套符号的来源之一。本课对照表至少报平均准确率和遗忘（或 BWT）。
- 第 06 课的 DER++：同一缓冲大小下，DER++ 把旧 logits 当蒸馏目标。本课它作为「回放类强方法」进同一张表，不是为了再消融 $\alpha,\beta$。
- 第 07 课：扩结构可以不改旧权重。本课改权重，但改之前先看方向。PackNet 要任务编号；GEM / A-GEM / GDumb 的主场可以是类增量。
- 线性代数：向量内积、把 $g$ 投影到与 $g_{\mathrm{ref}}$ 垂直的超平面。浏览器实验会画二维版。
- Python 3.10+、PyTorch。解 GEM 的二次规划：Avalanche 用 `qpsolvers`；Mammoth 的 `gem` 优先 `quadprog`（README 写明 Windows 上不可用），否则尝试 `qpsolvers` 并可能直接报「尚未接好」。本课主线用 A-GEM，避开这块平台坑。
- Mammoth 的 `gdumb` 需要能把 CIFAR/MNIST 跑完一次「最后任务才 fit buffer」。`fitting_epochs` 默认 256，缩小配置必须改这个数，否则 CPU 上一晚上都在重训。

建议目录：

```text
workdir/
  avalanche/
  mammoth/
  notes/lesson08.md
```

协议在动手前写进笔记，三方法共用，不许中途给 GDumb 加大缓冲或给 A-GEM 加任务编号。

## 4. 学习目标

1. 从「旧损失不许升高」写出 GEM 的不等式，并说明它如何变成梯度内积 $\langle \tilde g, g_k\rangle \ge 0$。
2. 在二维纸上画出当前梯度 $g$、旧梯度 $g_{\mathrm{ref}}$、投影后的 $\tilde g$，标出半平面。
3. 写出 A-GEM 的闭式投影，并指出它何时不投影（内积已经非负）。
4. 说出 GEM 与 A-GEM 在约束条数、每步反向次数、内存上的差别，并能在 Avalanche 插件代码里指到对应行。
5. 用伪代码走完 GDumb 的贪心均衡采样，解释为什么「最后才从头训」能在干净切分上得到高平均准确率。
6. 在同一协议下跑 A-GEM、DER++、GDumb，填对照表；完成复现 #2 的方向判断。
7. 用检查清单判断：自己以后读到的一篇 CL 论文，会不会被 GDumb 打脸。

## 5. 原理

六个机制。前三个把投影写清楚，后三个处理 GDumb 和评测诚实。

### 5.1 GEM：旧损失当不等式，更新做一次二次规划

持续学习的数据是一条流 $(x_i,t_i,y_i)$。任务局部 i.i.d.，全局非平稳。GEM 给每个已见任务 $k$ 留一块情景记忆 $\mathcal{M}_k$，容量在总预算 $M$ 已知时按任务均分。记忆上的经验损失为

$$
\ell(f_\theta,\mathcal{M}_k)=\frac{1}{|\mathcal{M}_k|}\sum_{(x,y)\in\mathcal{M}_k}\ell\big(f_\theta(x,k),y\big)
$$

观察到当前样本 $(x,t,y)$ 时，GEM 解：

$$
\begin{aligned}
\min_\theta\quad &\ell\big(f_\theta(x,t),y\big)\\
\text{s.t.}\quad &\ell(f_\theta,\mathcal{M}_k)\le \ell(f_{\theta}^{t-1},\mathcal{M}_k)\quad \forall k<t
\end{aligned}
$$

$f_{\theta}^{t-1}$ 是上一任务结束时的参数。不等式允许旧损失下降（正向 BWT），禁止上升（灾难性遗忘）。

每一步若真去比「更新后的旧损失」，还得把旧模型存下来。GEM 做了局部线性近似：参数走一小步 $g$（当前损失的梯度）时，旧损失升高当且仅当 $g$ 与旧损失梯度 $g_k$ 的夹角大于 90 度。于是约束写成

$$
\langle g, g_k\rangle \ge 0 \qquad \forall k<t
$$

若有一条不满足，就把 $g$ 投影到最近的可行点 $\tilde g$：

$$
\begin{aligned}
\min_{\tilde g}\quad &\frac12\|\tilde g-g\|_2^2\\
\text{s.t.}\quad &\langle \tilde g, g_k\rangle \ge 0 \quad \forall k<t
\end{aligned}
$$

这是参数空间里的 QP，变量个数等于网络参数量 $p$，可能上百万。对偶问题的变量个数等于已见任务数 $t-1$。解出对偶变量 $v^\star$ 后，

$$
\tilde g = g + G^\top v^\star
$$

其中 $G$ 的行是各旧任务梯度。原文还往 $v^\star$ 上加一个小的 $\gamma\ge 0$，把投影偏向更有利于正向 BWT 的一侧。Avalanche 把这个 $\gamma$ 叫 `memory_strength`，默认 0.5。

更新规则是 $\theta \leftarrow \theta - \alpha\tilde g$。因此 $\langle \tilde g, g_k\rangle \ge 0$ 意味着实际参数步 $-\alpha\tilde g$ 与旧损失上升方向 $g_k$ 的内积非正：旧损失在一阶近似下不升高。浏览器实验里的「半平面」就是 $\{\,u:\langle u,g_k\rangle\le 0\,\}$，其中 $u$ 是实际迈出的那一步。

验证：构造一个二维二次损失，旧任务等值线是椭圆，约束是半平面。投影前若 $g$ 指向旧损失升高的一侧，投影后点积符号必须翻转。CPU 实验用随机高维向量模拟同一件事。

GEM 原文实验：$T=20$ 的 Permuted MNIST、Rotated MNIST、Incremental CIFAR-100，每个样本只见一次。课内不做 20 任务单遍的完整复现；机制对上即可。正向 BWT 在 CIFAR-100 上被论文作为卖点，MNIST 上更常见的是「遗忘接近零」。

### 5.2 A-GEM：一条平均约束，一次向量减法

GEM 每步要对每个旧任务做一次完整反向，再解 QP。Chaudhry 等人把约束收成：记忆里所有旧任务合在一起的平均损失不许升。

$$
\min_\theta\ \ell(f_\theta,\mathcal{D}_t)\quad\text{s.t.}\quad \ell(f_\theta,\mathcal{M})\le \ell(f_\theta^{t-1},\mathcal{M}),\quad \mathcal{M}=\bigcup_{k<t}\mathcal{M}_k
$$

对应的可行条件只剩一条。$g_{\mathrm{ref}}$ 用从 $\mathcal{M}$ 抽出的一小批样本计算。若 $\langle g, g_{\mathrm{ref}}\rangle\ge 0$，不投影；否则

$$
\tilde g = g - \frac{g^\top g_{\mathrm{ref}}}{g_{\mathrm{ref}}^\top g_{\mathrm{ref}}}\,g_{\mathrm{ref}}
$$

这是把 $g$ 投影到 $\{z:\langle z,g_{\mathrm{ref}}\rangle=0\}$ 上，也就是减掉 $g$ 在 $g_{\mathrm{ref}}$ 方向的分量。几何上：当前梯度指向「旧损失也升高」时，把那一部分削掉，剩下的部分与 $g_{\mathrm{ref}}$ 正交。新任务的下降方向会被打折，但不会完全翻转到只保护旧任务。

Mammoth `models/agem.py` 把同一公式写成：

```python
def project(gxy, ger):
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger
```

`observe` 里先对当前 batch 反向得到 `grad_xy`，再对缓冲样本反向得到 `grad_er`，`dot_prod < 0` 才调用 `project`。Avalanche `avalanche/training/plugins/agem.py` 的 `after_backward` 是同一条：`alpha2 = dotg / (g_ref·g_ref)`，`grad_proj = current - g_ref * alpha2`。

验证有三条，CPU 实验应对上后两条：

1. $\langle g, g_{\mathrm{ref}}\rangle\ge 0$ 时，$\tilde g=g$。
2. $\langle g, g_{\mathrm{ref}}\rangle<0$ 时，投影后 $\langle \tilde g, g_{\mathrm{ref}}\rangle$ 在数值误差内为 0（正交投影到边界）。
3. $\|\tilde g-g\|_2$ 应小于把 $g$ 直接翻转到 $-g_{\mathrm{ref}}$ 的距离：投影是「最近可行点」，不是「沿旧梯度反走」。

二维数值例子。取 $g=(2,-1)$，$g_{\mathrm{ref}}=(1,0)$。内积 $\langle g,g_{\mathrm{ref}}\rangle=2>0$，不投影，新任务的下降方向保持原样。再取 $g=(-2,-1)$，内积 $-2<0$，于是

$$
\tilde g = (-2,-1)-\frac{-2}{1}(1,0)=(0,-1)
$$

投影后与 $g_{\mathrm{ref}}$ 正交；实际参数步沿 $(0,1)$ 走，旧损失在一阶意义下不变，新任务若在第二维上还有下降空间，这一维仍然能学。浏览器实验就是把这两个向量拖来拖去。

A-GEM 论文还引入了 Learning Curve Area（LCA，学习曲线面积：前几个 mini-batch 的平均准确率围成的面积），用来量「新技能学得多快」。本课主指标仍是平均准确率和遗忘；LCA 作为阅读问题，不进验收。

### 5.3 两条约束的代价差在哪

GEM 的保证更强：记忆上每个旧任务都不升。A-GEM 的保证更弱：平均值不升，某个旧任务仍可能升。换来的是：

- 不必把形状为 $(\text{任务数}\times p)$ 的矩阵 $G$ 常驻内存；
- 不必每步解 QP；
- 违反次数随任务增多涨得更慢，因为约束更少。

A-GEM 论文报告：相对原版 GEM，大约快两个数量级、内存低一个数量级，准确率同向或更好。这些倍数来自他们的实现和 $T=20$ 设定，课内不要把「100 倍」写进自己的日志当测量值。你要测的是：同一 MLP、同一 Split MNIST，GEM 每步是否明显更慢。若 Mammoth 的 `gem` 在你的机器上因 `quadprog` 装不上而直接失败，主线只跑 A-GEM，把 GEM 留在 Avalanche 插件阅读。

Avalanche 的 GEM 插件在 `before_training_iteration` 里对每个已见经验各做一次反向，堆成 `self.G`；`after_backward` 里若 `(G @ g < 0).any()` 就调 `solve_quadprog`。注释写明对偶求解来自官方 [facebookresearch/GradientEpisodicMemory](https://github.com/facebookresearch/GradientEpisodicMemory)。A-GEM 插件用 `GroupBalancedInfiniteDataLoader` 从各经验缓冲里抽 `sample_size` 条算 `reference_gradients`。

Mammoth 的 GEM 还依赖任务标签把缓冲里的梯度按任务分开（`buf_task_labels.unique()`），和 Lopez-Paz 原文的「整数任务描述符」一致。A-GEM 的 Mammoth 实现 `end_task` 只从当前 `train_loader` 取一个 batch 写入缓冲，**不是**把本任务样本均匀填满预算。对照实验若用 Mammoth `agem`，要在笔记里写清这条采样偏差；更干净的对照是 Avalanche `AGEM(patterns_per_exp=..., sample_size=...)`。

### 5.4 GDumb：贪心采样，笨学习器

GDumb 把持续学习的分类问题尽量少假设：流上每个时刻来一个 $(x_t,y_t)$，已见类集合 $Y_t$ 只增不减，测试样本应落在已见类上（论文把开放集留给未来）。它不需要任务边界、不需要在线只许看一次就不能重训、不需要测试时的任务编号。

两个零件：

**贪心均衡采样器。** 容量 $k$。新样本若其类在缓冲里的计数低于 $k/|Y|$，并且缓冲未满，则直接放入；若已满，则从当前样本数最多的类里随机抽一张丢掉，再放入新样本。新类出现时，总是从最大类那里抢格子。算法见论文 Algorithm 1。这样缓冲里各类计数趋向均匀，不依赖「每个任务两类」这种作业设定。

**笨学习器。** 在需要给出模型时（论文的推理阶段；Mammoth 实现选择在最后一个任务结束时），用缓冲里全部样本从头训练一个网络。优化器是固定的 SGD + 余弦重启，CIFAR 上加 CutMix，不按方法调参。这是故意的：GDumb 若还要为每个数据集搜超参，它就不再是「笨」基线。

预测时可以对 softmax 乘一张类掩码 $m$，从而在类增量和任务增量之间切换。$m$ 全 1 就是类增量；$m$ 只在当前任务的类上为 1 就是任务增量。GDumb 论文强调：学习器始终按类增量训练，掩码只在推理改。

Mammoth `models/gdumb.py` 的对应关系：

- `observe` 只 `buffer.add_data`，返回损失 0，真正的学习不在在线循环里。
- `n_epochs` 被强制为 1，因为那 1 个 epoch 只是为了把数据流过采样器。
- `end_task` 里，**不是最后一个任务就直接 return**；到了最后一个任务，`self.net = get_backbone(...)` 重新实例化骨干，再 `fit_buffer` 默认 256 个 epoch。
- `fit_buffer` 用 `maxlr=5e-2`、`minlr=5e-4`、CutMix $\alpha=1.0$。

这和论文「测试时用记忆从头训」一致，也解释了为什么 GDumb 的墙钟时间集中在最后一轮，而 A-GEM 的时间摊在每一步投影上。

验证：缓冲满了之后，各类计数的极差应远小于「先进先出不加均衡」的极差。若你注入一个新类，被丢掉的样本应来自当前最大类。CPU 实验若只钉投影，GDumb 的采样器可以在仓库实验里用断言或手工计数检查。

走一遍容量 $k=4$ 的小例子。缓冲空。来了三张类 A、一张类 B：缓冲变成 AAA B。再来一张类 C 时，C 的计数 0 低于 $4/3$，缓冲已满，从最大类 A 里随机丢掉一张，放入 C，变成 AA B C。再来一张类 C：C 的计数仍低于均衡值，再从 A 丢掉一张，变成 A B C C。贪心规则保证新类立刻占到格子，旧类不会被某一类独占到 $k$ 张。这和 DER++ 的 reservoir（按时间均匀，不按类）不同：GDumb 故意牺牲时间均匀，换类均匀，因为最后要在缓冲上做普通多类训练。

### 5.5 为什么它常赢，赢了说明什么

GDumb 在「类增量、在线、任务硬切分」的 A 类协议上对 A-GEM 的优势很大，因为这类协议有三层简化：

1. **任务硬切分。** 一段时间内只来固定两类或固定 5 类，缓冲里的旧类不会以奇怪的长尾继续涌入。均衡采样几乎就是每类存 $k/C$ 张 i.i.d. 图。
2. **离线重训被允许。** 笨学习器对缓冲做上百 epoch，相当于在一个小而均衡的数据集上做普通监督学习。A-GEM 每步只看当前 mini-batch，还可能被投影削掉有效学习方向。
3. **评价指标是最终平均准确率。** 第 03 课构造过「最后任务满分、前面全忘」也能把平均准确率抬上去。GDumb 反过来：它几乎没有「在线学会当前任务」的过程，但最终模型在缓冲覆盖到的类上可以很高。若你不报学习曲线、不报 LCA、不报缓冲外分布，它就会显得全能。

GDumb 并非在所有协议上都第一。论文表 5 的 B2（CIFAR-100，20 任务，缓冲 2000）里，GDumb 的平均准确率低于 BiC 和 iCaRL 约 10 到 20 个点；表 6 的 D 设定（任务增量在线、小记忆）里 A-GEM 63.1 ± 1.24，GDumb 60.3 ± 0.85。所以复现 #2 的通过标准写成「不低于，或写明反例及缓冲大小」，而不是「必须高出 60 个点」。

说明什么：很多 2017-2020 年的方法，在自己挑选的简化协议上调到了最优，却没拿「存样本再重训」当对照。GDumb 把这件对照强制公开。它没有否定投影、回放、正则在更脏的流上的价值：任务边界模糊、禁止离线重训、类数未知增长、技能不是分类标签时，书包重训会先失效。

### 5.6 同一协议：三方法到底在比什么

A-GEM、DER++、GDumb 必须共享：

- 同一数据集与同一类增量切分（本课主线：Split MNIST，10 类 / 5 任务，每任务两类）；
- 同一缓冲容量 $k$（主线建议 200 和 500 各跑一次）；
- 同一骨干宽度（MNIST 用两层 MLP，隐层 100 或 Mammoth 默认）；
- 同一测试协议（类增量：测试时不给任务编号）；
- 同一随机种子记录。

不能共享、必须写进表注的差异：

- A-GEM 每步更新，GDumb 最后才训练；
- DER++ 把缓冲样本和当前样本混在同一个交叉熵加蒸馏损失里，A-GEM 的缓冲主要用于约束；
- Mammoth `agem` 的缓冲填充与 `derpp` 的 reservoir 不同，见 5.3。

若你用 Avalanche 跑 A-GEM、用 Mammoth 跑 GDumb 和 DER++，骨干和预处理可能不一致。能同一仓库就同一仓库。Mammoth 三者都有：`agem`、`derpp`、`gdumb`。Avalanche 的 GEM/AGEM 更接近论文插件结构，适合做投影代码导读。推荐：机制读 Avalanche，对照数字跑 Mammoth。

## 6. 源码导读

按「当前梯度怎么被改掉」这条路径读，不要从 `benchmarks` 目录逛起。

### 6.1 Avalanche：GEMPlugin 与 AGEMPlugin

安装仍以当前 README 为准：

```bash
pip install avalanche-lib
```

需要读源码时克隆仓库。关键文件：

| 文件 | 带着什么问题读 |
|---|---|
| `avalanche/training/plugins/gem.py` | `G` 何时堆好？`to_project` 的判定是不是「任一条内积为负」 |
| `avalanche/training/plugins/gem.py` 的 `solve_quadprog` | `memory_strength` 加在对偶不等式的哪一侧 |
| `avalanche/training/plugins/agem.py` | `sample_size` 怎么从多个经验的 buffer 里抽 |
| `avalanche/training/supervised/strategy_wrappers.py` 的 `GEM`、`AGEM` | 构造函数默认 `train_epochs=1`，对应单遍设定 |
| `avalanche/training/plugins/agem.py` 的 `update_memory` | 每个经验只留 `patterns_per_experience` 条 |

`GEMPlugin.after_backward` 把所有参数的梯度拉平拼成 $g$，用 `torch.mv(self.G, g) < 0` 判断是否有违反。注意 `self.G` 的行是旧任务梯度，所以 `G g` 就是各 $\langle g_k, g\rangle$。投影解出的 `v_star` 被 reshape 回每张参数矩阵。`patterns_per_experience` 控制记忆大小，不是 A-GEM 的 `sample_size`。

`AGEMPlugin` 把记忆做成 `List[AvalancheDataset]`，每个经验一份。`before_training_iteration` 先用记忆样本算出 `reference_gradients` 并把优化器梯度清零，避免污染当前 batch。`num_workers > 0` 会触发警告：已知会严重拖慢 A-GEM。课内设 0。

策略封装的最小调用形态（与 README 的 Naive 例子同一风格，把 `Naive` 换成 `AGEM`）：

```python
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.models import SimpleMLP
from avalanche.training import AGEM
```

具体超参在 Step 3 给出。不要把这段当成可复制的完整脚本；完整训练循环仍是 `for train_exp in train_stream: strategy.train(train_exp)`。

### 6.2 Mammoth：agem、gem、gdumb、derpp

| 文件 | 带着什么问题读 |
|---|---|
| `models/agem.py` | `project` 与论文式 (11) 是否逐项相同 |
| `models/gem.py` 的 `project2cone2` | `margin` 默认 0.5，对应原文 $\gamma$ |
| `models/gdumb.py` | 为什么 `end_task` 在非最后任务直接返回 |
| `models/derpp.py` | 第 06 课已读：$\alpha$ 蒸馏、$\beta$ 回放 CE |
| `models/config/gdumb.yaml` | `seq-cifar10` 下 `lr: 0`，学习率走 `maxlr` |
| `README.md` 模型列表 | `gem` 标注 Unavailable on windows；`agem_r` 是带 reservoir 的 A-GEM |

Mammoth GEM 在构造时 `import quadprog`，失败则尝试 `qpsolvers` 并 `raise Exception('QPSolvers is just a suggestion but does not work at the moment...')`。不要把时间花在接 QP 上，除非你已经有 Linux 且 Python ≤ 3.10 的 `quadprog`。课程主线的投影机制以 A-GEM 和课内 CPU 实验为准。

GDumb 的 `fit_buffer` 把整个缓冲 `get_data(len(self.buffer.examples))` 拉到内存再按 `batch_size` 切片。缓冲 500、MNIST，CPU 可承受；缓冲 2000、CIFAR，要估显存和 `fitting_epochs`。缩小配置把 `--fitting_epochs` 降到 5 或 10 做冒烟，正式复现再加回去。默认 256 是为了靠近论文的充分重训，不是为了你的笔记本。

### 6.3 课内 CPU 实验在钉什么

`experiments/src/learn_cl_experiments/lessons/lesson_08.py` 不训练网络。它应构造一对向量 $g$ 与 $g_{\mathrm{ref}}$，使得 $\langle g, g_{\mathrm{ref}}\rangle<0$，应用 5.2 的公式，然后断言投影后内积不再为负（在浮点容差内接近 0）。再构造一对已经满足约束的向量，断言投影是恒等。`python3 run.py run 08` 写入 `artifacts/lesson08/result.json`。

这对应 EXPERIMENT_AGENT_BRIEF 里「违反旧任务约束的梯度投影后与约束法向点积 ≤ 0」：若把「约束法向」取成旧损失上升方向、把「更新」取成 $-\tilde g$，内积 ≤ 0；若按论文把 $\tilde g$ 本身与 $g_{\mathrm{ref}}$ 做内积，则 ≥ 0。两种写法差一个符号，报告里写你用的是哪一个，断言与公式必须一致。

## 7. 实验

浏览器先建立半平面直觉；CPU 把公式钉死；仓库上才比三个方法。复现 #2 只看方向和记录，不要求对齐 GDumb 论文表 4 的 91.9。

### Step 0: 浏览器实验，梯度投影

打开本课网页的「梯度投影」。左侧是二维向量：当前梯度 $g$（或你将要迈出的步 $u$）、旧任务约束的法向 $g_{\mathrm{ref}}$。半平面一侧标成「旧损失升高，禁止」。你先预测：投影之后，箭头还指不指向新任务的下降方向。旁边是 GDumb 面板：不投影，把背包里的点重新拟合一条线性边界。

预测必须在运行前完成。改向量或改夹角应作废上次运行。过关条件：当 $g$ 落入禁止半平面时，你预测「需要投影」，且投影后与法向的内积不再落在禁止侧；当夹角已经锐角，你预测「不投影」。GDumb 侧不要求预测分数，只要看见它忽略投影、只用背包。

把夹角从钝角拖到锐角，抄三次结果：需要投影、刚好正交、不需要投影。后面 CPU 实验的随机向量，应落在同一套符号约定里。

### Step 1: CPU 机制实验

```bash
cd experiments
```

```bash
python3 run.py run 08
```

`python3 run.py run 08` 现在应当全绿，终端打印 `[PASS]`，`checks` 全真。键名是 `violating_update_detected`、`projected_dot_non_positive`、`non_violating_update_unchanged`、`random_projected_dot_non_positive`、`agem_matches_update_projection`。

这是几何课，不训练网络。更新 $u$ 若 $u\cdot n>0$ 就会抬高旧损失，投影到半平面 $u\cdot n\le 0$。本机一次运行（seed 8）：违规点积 2.0，投影后 0.0；未违规点积 -2.0，投影保持不变；16 维随机例子投影后点积同样为 0.0。A-GEM 对梯度的投影与 $-u$ 投影重合（`update_proj_neg_agem_l2`=0.0）。换机器会变，方向不应变：投影后点积 $\le 0$。

不要把这组点积写成复现 #2。GDumb 与 A-GEM 的分数对照走 Mammoth（Step 3）。

### Step 2: 同一协议的三方法对照（缩小）

克隆 Mammoth，依赖按它的 `requirements.txt`。主线用 `seq-mnist`，类增量，缓冲 200。缩小 epoch，确认三条命令都能出平均准确率再上正式配置。

```bash
python main.py --model agem --dataset seq-mnist --buffer_size 200 --n_epochs 1 --lr 0.03
```

```bash
python main.py --model derpp --dataset seq-mnist --buffer_size 200 --n_epochs 1 --alpha 0.5 --beta 0.5 --lr 0.03
```

```bash
python main.py --model gdumb --dataset seq-mnist --buffer_size 200 --fitting_epochs 10
```

GDumb 的 `lr` 在 yaml 里常为 0，真正起作用的是 `maxlr`。`fitting_epochs 10` 只是冒烟；正式复现把该值提高到 50 或保持默认 256，按你的机器写进记录。

三方法的输出日志格式应能读出每个任务结束时的准确率。填 5×5 矩阵，用第 03 课公式算平均准确率和 BWT。冒烟分数不作复现结论。

### Step 3: 复现 #2（方向性）

同一协议，缓冲改到 500（与 GDumb 论文 A2 的 $k=500$ 同量级），MNIST 类增量。A-GEM 与 GDumb 必须都跑完。DER++ 作为第 06 课方法进同一张表，帮助你判断「赢 GDumb 的是投影还是回放蒸馏」。

正式 GDumb：

```bash
python main.py --model gdumb --dataset seq-mnist --buffer_size 500 --fitting_epochs 50
```

正式 A-GEM：

```bash
python main.py --model agem --dataset seq-mnist --buffer_size 500 --n_epochs 1
```

若你更信任 Avalanche 的记忆填充，用 `SplitMNIST(n_experiences=5, return_task_id=False)` 配 `AGEM(patterns_per_exp=100, sample_size=64)`（5 任务 × 100 = 总记忆 500）。两种实现不要混在同一行数字里比较，表里加「仓库」列。

通过标准（课程蓝图 §4）：

- GDumb 的平均准确率不低于 A-GEM；或
- GDumb 更低，但你写明缓冲大小、任务切分、epoch、以及你认为反例成立的原因（例如 `fitting_epochs` 太小、Mammoth `agem` 的缓冲采样与论文不同）。

GDumb 论文 A2 表在 $k=500$ 的 MNIST 上 A-GEM 约 29、GDumb 约 92。你的实现如果两边都在 80 以上，仍然可以「GDumb 不低于 A-GEM」算通过；不要为了靠近 29 去把 A-GEM 训崩。方向优先。

DER++ 在同一缓冲下通常应高于 A-GEM（第 06 课的经验：蒸馏项稳）。若 DER++ 也低于 GDumb，把它写进检查清单的「本设定会被打脸」一栏，这是有信息量的结果，不是实验失败。

### Step 4: 可选 GEM 对照

仅在 `quadprog` 可装时：

```bash
python main.py --model gem --dataset seq-mnist --buffer_size 500 --gamma 0.5
```

记录墙钟时间和平均准确率。预期：不比 A-GEM 差很多，但更慢。装不上就跳过，阅读 `gem.py` 即可。

### Step 5: 检查清单

对照表填完之后，用下面这张清单审查你的设定。每条答「是」都增加「GDumb 会接近或超过花哨方法」的概率。

| 检查项 | 是 / 否 | 若「是」意味着什么 |
|---|---|---|
| 任务硬切分，每段只出现固定几类 |  | i.i.d. 子样本已经能代表该类 |
| 允许在任务结束或测试前用缓冲充分重训 |  | 笨学习器的主场 |
| 只报最终平均准确率，不报 LCA / 在线曲线 |  | 不惩罚「当时没学会」 |
| 缓冲按类均衡，且 $k/C$ 仍有数十张图 |  | 重训不会严重欠拟合 |
| 测试分布与缓冲同类、同类增强 |  | 没有开放集、没有风格漂移 |
| 方法的超参为该数据集单独搜过，GDumb 没有 |  | 不公平，GDumb 论文故意不搜 |
| 类增量却在测试时用了任务掩码 |  | 把问题改成了任务增量 |

把答「是」的条数写进复现记录。条数越多，GDumb 打脸越不令人意外。第 22 课会把其中若干条去掉，那时再回来看投影和回放还剩多少价值。

## 8. 配置与预算

| 项目 | 缩小 / 冒烟 | 复现 #2 主线 | 加分 |
|---|---|---|---|
| 数据 | seq-mnist，5 任务 | seq-mnist，$k=200$ 与 $500$ | seq-cifar10，单卡 |
| 骨干 | MLP | Mammoth 默认 MNIST MLP | 缩小 ResNet18 |
| 缓冲 | 200 | 200 与 500 | 1000 |
| A-GEM epoch | 1 | 1（贴近单遍） | 多 epoch 扫描，需在记录声明 |
| GDumb fit | 10 epoch | 50 或 256 | 论文式充分重训 |
| DER++ | 与第 06 课相同 $\alpha,\beta$ | 同缓冲对照 | 关蒸馏项再比一次 |
| 硬件 | CPU | CPU 足够 MNIST | CIFAR 用单卡数小时 |
| 墙钟 | 三方法冒烟各数分钟到数十分钟 | GDumb 256 epoch 可能数小时 | CIFAR 更长 |

学习率：A-GEM / DER++ 用命令行 `--lr`；GDumb 用 `--maxlr` / `--minlr`。不要给 GDumb 设一个和 A-GEM 相同的 `--lr` 然后疑惑为什么日志里优化器用的是另一套。

种子：`--seed` 固定后，三方法各自跑，不要为了让 GDumb 赢或输去改种子。方向判断允许误差，不允许事后选种子。

Mammoth 的 `agem_r` 用 reservoir 缓冲，更接近「流上均匀记忆」。若主线 `agem` 的 `end_task` 单 batch 填充让你不安，加一列 `agem_r` 作为附录，不要替换已经写进复现 #2 的那一行，除非你从头重跑三方法。

## 9. 验收

- 浏览器「梯度投影」先预测再运行：钝角时投影，锐角时不投影。
- `python3 run.py run 08` 的 `checks` 全真（`violating_update_detected`、`projected_dot_non_positive`、`non_violating_update_unchanged`、`random_projected_dot_non_positive`、`agem_matches_update_projection`）。
- 能在纸上写出 A-GEM 的闭式投影，并能指到 `models/agem.py` 的 `project` 或 `plugins/agem.py` 的 `after_backward`。
- 三方法对照表：同一数据集、同一 $k$、同一增量设定；列至少包括平均准确率、遗忘或 BWT、墙钟、仓库名。
- 复现 #2：GDumb ≥ A-GEM，或书面反例含缓冲大小与切分。
- 「会不会被 GDumb 打脸」清单填完，至少六条有是/否。
- 没有把 GDumb 的胜利写成「持续学习无用」；报告里有一小节解释任务边界和 i.i.d. 重训。
- 没有把冒烟的 `fitting_epochs=10` 当成论文对照。

正式复现五项里本课是第 2 项。通过标准见课程蓝图 §4，是方向不是小数点。

## 10. 排错

| 症状 | 原因 | 验证 | 修法 |
|---|---|---|---|
| `quadprog` 导入失败 | Mammoth GEM 的硬依赖，Windows 或新 Python | 看 `models/gem.py` 构造函数异常 | 主线改 A-GEM；或用 Avalanche + `qpsolvers` |
| A-GEM 与 naive 分数几乎一样 | 缓冲是空的：`end_task` 没执行或 `buffer_size` 过小 | 打印 `buffer.is_empty()` | 确认任务循环调用了 `end_task`；$k$ 至少覆盖若干 batch |
| 投影后 NaN | $g_{\mathrm{ref}}$ 为零向量（记忆样本全对或全坏） | 检查 `g_ref.norm()` | 跳过参考梯度范数为零的步；加 $\varepsilon$ |
| GDumb 准确率接近随机 | `fitting_epochs` 太小，或 `cutmix` 在 MNIST 上过猛 | 看 `fit_buffer` 的 loss 是否下降 | MNIST 可把 `cutmix_alpha` 设为 0；加 epoch |
| GDumb 显存爆 | `get_data(len(buffer))` 一次拉满 | 缓冲尺寸 × 图像尺寸 | 降 `buffer_size` 或改循环，不要一次把 CIFAR 全放 GPU |
| 三方法分数不可比 | 一个用了任务编号，一个没有 | 测试日志是否含 task-il 指标 | 全部 class-il；关掉测试掩码 |
| DER++ 远低于第 06 课 | epoch、增强、缓冲采样不一致 | 对照第 06 课命令 | 复用当时的 `derpp` 命令，只统一 `buffer_size` |
| Avalanche A-GEM 极慢 | `num_workers>0` | 警告日志 | worker 设 0 |
| CPU 实验内积符号与浏览器相反 | 一个用 $g$，一个用步长 $u=-g$ | 两边的点积定义 | 统一约定并写进 `notes/lesson08.md` |
| `projected_dot_non_positive` 为假 | 投影公式减错了法向分量，或把 $u$ 和 $g$ 的符号搞反 | 看 `projected_dot`、`violating_raw_dot` | 对照 `_project_update`：违规点积 2.0 投影后应 $\le 0$ |
| `agem_matches_update_projection` 为假 | A-GEM 投影的是 $g$，比较时没落到 $-u$ | 看 `update_proj_neg_agem_l2` | 半平面 $g\cdot g_{\mathrm{ref}}\ge 0$ 与 $u\cdot n\le 0$ 差一个符号；本课几何实验不是复现 #2 |
| GDumb 低于 A-GEM 很多 | fit 不足，或缓冲没均衡 | 各类计数；loss 曲线 | 加 `fitting_epochs`；检查 `Buffer.add_data` 是否按类丢弃。仍低则写反例 |

## 11. 前沿与改造

约束梯度这条线后来分叉。一边是把 GEM 的约束做得更便宜、更随机（A-GEM、S-GEM）；一边是承认「记忆里的梯度」只是旧分布的残缺估计，转而把记忆直接拿去回放，DER++、ER-ACE 在类增量上通常更稳。GDumb 之后，不少论文被迫把「从缓冲重训」写进基线栏。2022 年起预训练模型上的 L2P / DualPrompt 又把「不改骨干」推回舞台，第 07 课已经做过。2024 年起同一条投影接到 PEFT 和全量微调：PEGP 给 Adapter / LoRA / Prompt 统一正交梯度，Sculpting Subspaces 用自适应 SVD 限制全量更新方向，DOC 跟踪功能方向漂移后再切梯度。记忆够用时，Forget Forgetting 把主问题从遗忘挪到可塑性。第 12 节列这些论文。

我们差在哪：课内协议仍然是干净的 Split MNIST。GDumb 打脸最狠的地方，往往就是这种协议。若你只在这里证明「我的投影比 GDumb 强」，说服力有限。真正的压力测试是模糊边界、单遍、禁止重训。

动手改造清单：

1. **关掉投影。** 在 `models/agem.py` 的 `if dot_prod.item() < 0` 里强制走 `else`，始终用当前梯度。预算：一次 seq-mnist，$k=500$。预期：旧任务遗忘上升，接近 naive。失败标准：分数不变，说明缓冲本来就是空的，投影从未触发。
2. **GDumb 去掉均衡。** 把采样改成纯 reservoir 或先进先出，再 `fit_buffer`。预期：类不均衡时平均准确率下降。失败标准：各类本来就均衡（Split MNIST 每类样本数接近），差异被淹没。可改用不均衡的子采样数据流。
3. **同一 $k$ 扫 DER++ 的蒸馏项。** $\alpha=0$ 对比默认 $\alpha$。预期：关蒸馏后更接近普通 ER，对 GDumb 的优势缩小。失败标准：$\alpha$ 没传到损失，日志里 DER 项恒为 0。
4. **把 GDumb 的重训提前到每个任务结束。** 改 `end_task` 去掉「只有最后任务才 fit」的判断。预期：中间任务的考试曲线变好，总墙钟变长。失败标准：实现时忘了重新初始化骨干，变成在旧权重上继续训，那就不再是 GDumb。

顺手复现映射：改造 1 对应「A-GEM 相对 naive 的增益来自投影」；改造 4 对应论文「笨学习器何时被调用」。复现 #2 的主数字仍应来自未改仓库的命令。

## 12. 论文与延伸

每篇对应一个能用本课实验回答或明确答不了的问题。读完把答案写进 `notes/lesson08.md`。谱系只留 GEM 和 GDumb，因为 CPU 实验在做半平面投影，复现 #2 在对照重训基线。A-GEM 是 GEM 的闭式便宜版，写进 GEM 条，不再单列。2024 年以后主阅读是正交子空间和「记忆够用」这两条。

1. Lopez-Paz and Ranzato, 2017, *Gradient Episodic Memory for Continual Learning*, [arXiv:1706.08840](https://arxiv.org/abs/1706.08840)。
贡献：定义 ACC / BWT / FWT，并用情景记忆把「旧损失不升高」写成可投影的二次规划。机制发明处，不是本课主阅读。
机制：每个旧任务一块记忆。当前损失当目标，旧记忆上的损失当不等式。一阶近似后约束变成 $\langle \tilde g, g_k\rangle\ge 0$。违反时把当前梯度投影到最近可行点。不等式允许旧损失下降（正向 BWT），禁止上升。A-GEM 后来把多条约束收成一条平均约束，投影变成一次向量减法；本课 CPU 实验实现的是这一便宜版。
和本课：浏览器半平面、`violating_update_detected`、`projected_dot_non_positive`、`non_violating_update_unchanged`、`random_projected_dot_non_positive`、`agem_matches_update_projection`。答不了原文 $T=20$ 单遍 MNIST / CIFAR-100 的表。
阅读问题：违规更新投影之后与法向的点积是否 $\le 0$？`projected_dot_non_positive` 必须为真。用同一张二维图说明：不等式为何仍允许旧损失下降（正向 BWT）。

2. Prabhu, Torr, Dokania, 2020, *GDumb: A Simple Approach that Questions Our Progress in Continual Learning*，ECCV 2020，PDF：[robots.ox.ac.uk 版本](https://www.robots.ox.ac.uk/~tvg/publications/2020/gdumb.pdf)。
贡献：贪心均衡采样加缓冲重训，在多种简化协议上接近或超过当时 SOTA。机制发明处，不是本课主阅读。
机制：容量 $k$ 的缓冲按类尽量均匀；新类来了就从当前最多的类里丢掉一个。需要模型时（Mammoth 选在最后任务结束）用缓冲从头训一个网络。不编码任务边界，也不在流上做约束投影。摘要写明它并非为持续学习特制，却在几乎所有对照实验里拿到很高的准确率。
和本课：复现 #2 与「会不会被 GDumb 打脸」清单。答不了论文全部协议变体；你只在同一 $k$ 的 Split MNIST 上比方向。
阅读问题：同一协议下 GDumb 是否不低于 A-GEM？若否，书面反例必须含缓冲大小与切分。`fitting_epochs=10` 的冒烟不能当论文对照。

3. Qiao, Zhang, Tan, Qu, Zhang, Han, Xie, 2024, *Gradient Projection For Continual Parameter-Efficient Tuning*, [arXiv:2405.13383](https://arxiv.org/abs/2405.13383)。
贡献：把 Adapter、LoRA、Prefix、Prompt 收成同一套参数高效梯度投影（PEGP）。
机制：从「旧输入在更新后输出不变」推出 $x_t\Delta E=0$，即新梯度落在旧特征张成子空间的正交方向。用旧特征的 SVD 取近零奇异向量当投影矩阵，额外内存和墙钟都很小。在 ViT 和 CLIP 上测类增量、在线类增量、域增量、任务增量和跨模态。
和本课：CPU 的 `_project_update` 是同一几何，作用对象是整网更新 $u$；PEGP 作用对象是 PET 模块。`agem_matches_update_projection` 能看见闭式投影，看不见对 Prompt / LoRA 分别做 SVD。
阅读问题：锐角（不违规）时投影是否保持原向量？`non_violating_update_unchanged` 必须为真。PEGP 在 CLIP 上是否减轻零样本塌缩，本课实验答不了。

4. Nayak et al., 2025, *Sculpting Subspaces: Constrained Full Fine-Tuning in LLMs for Continual Learning*, [arXiv:2504.07097](https://arxiv.org/abs/2504.07097)。
贡献：全量微调加自适应 SVD，把更新限制在与旧任务关键方向正交的低秩子空间，不另增每任务参数、不存旧梯度。
机制：每层对权重做 SVD，大切值方向视为旧知识，小切值方向拿来学新任务。按层输入输出相似度分配保留比例，新任务梯度投影到低秩子空间。摘要写明在 T5-Large 和 LLaMA-2 7B 上平均准确率可比 O-LoRA 高到 7%，并把遗忘压到接近可忽略。
和本课：GEM 用记忆梯度当护栏；这篇用权重谱当护栏，不要回放。本课投影公式能类比「切掉危险方向」，答不了自适应秩和 LLM 安全 / 指令跟随。
阅读问题：把 GEM 的 $g_{\mathrm{ref}}$ 换成「权重大奇异方向」，投影后点积还应不应 $\le 0$？几何上应该。本课没有 SVD，自适应秩本身答不了。

5. Cao and Wu, 2025, *Orthogonal Low-rank Adaptation in Lie Groups for Continual Learning of Large Language Models*, [arXiv:2509.06100](https://arxiv.org/abs/2509.06100)。
贡献：OLieRA。在任务子空间正交之外，用李群乘法更新保住参数几何，推理仍无回放、无任务编号。
机制：O-LoRA 的加法 $W\leftarrow W+\Delta W$ 会拧歪预训练几何。OLieRA 把 $\Delta W=BA$ 放进指数映射，做 $W\odot\exp(\Delta W)$，再用泰勒展开成可算的 Hadamard 形式。正交损失加在整个 $\exp(\Delta W)$ 上，不只加在 $B$ 上。论文在 Standard CL 的 T5-large 上报告平均准确率 79.6，接近多任务上界 80.0。
和本课：A-GEM 的减法投影是欧氏空间里削法向分量。OLieRA 先乘再正交。`update_proj_neg_agem_l2` 能核对欧氏投影，核对不了指数映射。
阅读问题：本课投影是 $u - \frac{u\cdot n}{n\cdot n}n$。若改成「沿参数逐元缩放后再正交」，二维点积符号还会不会翻？本课实验答不了，因为 `_project_update` 没有 Hadamard 乘法。

6. Zhang, Wei, Sun, 2025, *Dynamic Orthogonal Continual Fine-tuning for Mitigating Catastrophic Forgettings*, [arXiv:2509.23893](https://arxiv.org/abs/2509.23893)。
贡献：DOC。指出正则方法在长序列上失败的一个原因是功能方向会漂，于是在线跟踪这些方向再切梯度。
机制：用 LoRA 增量当功能方向，Online PCA 从当前任务数据里抽出并更新主分量（无旧数据）。新任务的 $B$ 梯度减去它在这些主分量上的分量，使新增量与已跟踪的历史方向正交；$A$ 的梯度不动，当动量。功能方向不再是任务结束时冻住的那一组。
和本课：A-GEM 的 $g_{\mathrm{ref}}$ 来自记忆、一步一算但方向不跟踪漂移。`violating_update_detected` 能看见「方向冲突才投影」。答不了 Online PCA，也答不了长链 LLM 指令任务。
阅读问题：若把本课的法向 $n$ 在投影前随机转一个小角度（模拟漂移），投影后点积还保证 $\le 0$ 吗？对错误的 $n$ 仍会垂直，但对真正的旧损失不再保证。DOC 的跟踪本身本课实验答不了。

7. Yang, Ning, Liu, Yao, Tian, Song, Yuan, 2024, *Is Parameter Collision Hindering Continual Learning in LLMs?*, [arXiv:2410.10179](https://arxiv.org/abs/2410.10179)。
贡献：正交还不够，参数碰撞才是更紧的因素；提出 N-LoRA，用低碰撞率做持续学习。
机制：两个更新非碰撞（同一位置至少一方为 0）能推出正交，反过来不行。O-LoRA 的子空间正交仍可能在坐标上互踩。N-LoRA 给当前任务的 $\Delta W$ 加 $\ell_1$，把更新挤得很稀，旧 LoRA 冻住可并回骨干。摘要写明相对 SOTA 平均准确率 $+2.9$，任务正交约 $4.1$ 倍，参数碰撞约低 $58.1$ 倍。
和本课：GEM 约束的是梯度夹角，N-LoRA 约束的是坐标重叠。本课没有稀疏 LoRA，测不了碰撞率。
阅读问题：投影后 $\tilde g$ 与 $g_{\mathrm{ref}}$ 正交，是否意味着两个任务的权重更新在同一坐标上也不重叠？不一定。本课实验只能看见角度，看不见碰撞。碰撞率数字本课实验答不了。

8. Cho, Moon, Chunara, Cho, Cha, 2025, *Forget Forgetting: Continual Learning in a World of Abundant Memory*, [arXiv:2502.07274](https://arxiv.org/abs/2502.07274)。
贡献：存储不再贵、GPU 才贵时，核心从遗忘换成可塑性；提出权重空间巩固（重置休眠参数加权重平均）。
机制：传统 CL 把 exemplar 压得很小。记忆够用时，简单回放就能在低 GPU 成本下超过许多 SOTA；模型却偏向旧任务、学新任务变慢。方法用梯度一、二阶矩找出休眠参数，软重置回上一任务权重，再在训练轨迹上做随机权重平均。在类增量和 LLM 持续指令微调上验证。
和本课：GDumb 是「记忆够就重训」的极端；这篇停在中间地带：记忆够用但不做从头全量重训。复现 #2 能看见小 $k$ 上重训很强。答不了「记忆放到数据集百分之几十」时可塑性怎么掉。
阅读问题：同一 $k$ 下 GDumb 是否不低于 A-GEM？复现 #2 能答。论文说记忆放到够用之后主问题变成可塑性；本课没有扫大记忆，这一句答不了。

9. Liu, Wan, Xu, Zhang, Xie, Xiong, 2026, *Attribution-Guided Continual Learning for Large Language Models*, [arXiv:2605.05285](https://arxiv.org/abs/2605.05285)。
贡献：用层间相关性传播（LRP）按模型内部计算估参数重要性，再按重要性约束更新。
机制：回放、冻结、正则都不区分「哪一颗参数在存旧知识」。持续微调时，对旧任务重要的参数少更新，不相关的参数留给新任务。摘要写明相对基线减少遗忘并保住新任务适应力，强调机制归因对 LLM 持续微调有用。
和本课：GEM 用旧样本梯度当重要性；这篇用 LRP 归因。`projected_dot_non_positive` 检验的是梯度半平面，不是归因图。本课没有 LRP。
阅读问题：EWC 的 Fisher 和第 05 课弹簧，跟 LRP 归因是不是同一张重要性图？本课实验答不了，因为既没有 Fisher 也没有 LRP。你能答的是：只用记忆梯度当护栏时，投影后点积必须非正。

现在四类补丁齐了：弹簧、背包、新块、投影，外加一个让平均准确率丢脸的重训基线。近两年把投影写进 PEFT 和全量微调，也开始问记忆够用时还要不要投影。[第 09 课](09_continual_pretraining.md) 离开 CIFAR 和 MNIST，把同一套「接着写权重」的问题搬到语言模型：换领域续预训练时，旧的通用能力掉的是知识还是格式。缓冲在那里会变成「混一点通用语料」，对应本课 DER++ 做过的事，只是样本从图片换成了 token。



