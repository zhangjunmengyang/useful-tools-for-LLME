---
id: 05_ewc_regularization
title: "重要的权重不许动太多"
summary: "EWC 怎么知道哪些权重对旧任务重要？"
unit: toolkit
play_tools: []
checkpoints:
  - "λ 扫描曲线。"
  - "Fisher 直方图。"
  - "naive vs EWC vs SI vs LwF 的对照数字。"
---

# 第 05 课：重要的权重不许动太多

> 类型：实战（对照官方策略跑通；缩小配置的数字不算论文复现）<br>
> 建议周期：2-4 天<br>
> 硬件：Split MNIST 用 CPU / Mac；Split CIFAR-10 建议单卡<br>
> 锚定仓库：[ContinualAI/avalanche](https://github.com/ContinualAI/avalanche) 的 `avalanche.training.EWC` 与 `avalanche.training.plugins.EWCPlugin`；[aimagelab/mammoth](https://github.com/aimagelab/mammoth) 的 `ewc_on`、`si`、`lwf`<br>
> 产物：λ 扫描曲线、Fisher 对角线直方图、「没有旧数据时 EWC 还能不能用」的书面回答

## 1. 这一课做什么

第一幕已经把病看清楚了。[第 01 课](01_catastrophic_forgetting.md) 让同一个网络先学任务 A 再学任务 B，A 的准确率会塌；[第 02 课](02_stability_plasticity.md) 把这件事画成稳定性-可塑性平面；[第 03 课](03_cl_evaluation.md) 规定了以后怎么量，不许只报最终平均准确率；[第 04 课](04_not_just_rag.md) 说明把名录塞进上下文或检索出来，和改权重不是同一件事。

从这一课起进入第二幕：四类补丁。整门课的主干没变，变的是「写到哪里」和「怎么写」：

```text
新经验进来
  先决定写到哪里（上下文 / 外挂记忆 / 快速权重 / 慢速权重）
  再决定怎么写（覆盖、追加、压缩、约束、正交、合并）
  写完立刻测：新任务会了没、旧任务还在不在
  长期还要测：还能不能继续学
```

本课换的零件是：**写进慢速权重，写法是约束**。新任务的梯度照样回传到同一套参数，但每个参数身上多了一根弹簧。弹簧的劲度由「这个参数对旧任务有多重要」决定。重要的少动，不重要的放开学。这就是 EWC（弹性权重巩固，Elastic Weight Consolidation）：给重要的权重加弹簧，不让它跑远。

没有这个零件会怎样？你只能在三种坏里挑：直接接着训（naive fine-tune），旧任务被冲掉；把学习率拧到接近零，新任务学不进；把整个骨干冻住，可塑性当场死掉。第 02 课已经在平面上见过这几个点。EWC 想走第三条路：同一套权重继续动，但动的方向被旧任务的曲率卡住。

本课不做三件事。不存旧样本（那是 [第 06 课](06_replay_der.md) 的背包）；不给网络再长一块（[第 07 课](07_architecture_prompts.md)）；不把「不增加旧损失」写成二次规划（[第 08 课](08_gem_gdumb.md)）。本课也不把缩小配置跑出来的准确率写成「复现了 Kirkpatrick 等人 2017 年的 Nature 数字」。课程蓝图把第 05 课标成实战：你要能解释 Fisher 对角线、能扫 λ、能对照 naive / EWC / SI / LwF，并书面回答「没有旧数据时 EWC 还能不能用」。

做完你手里会有四样东西：一张 Fisher 对角线的直方图（大部分权重几乎不被钉，少数被钉死）；一条 λ 从 0 扫到很大时，稳定性-可塑性平面上的轨迹；同一协议下 naive、EWC、SI、LwF 的对照表；一段不超过一页的判断，写清 EWC 不需要回放缓冲，但在 class-incremental（类增量：每个任务带来新类别，测试时也不告诉你现在是哪个任务）上常常撑不住。

| 术语 | 一句解释 |
|---|---|
| 正则化持续学习 | 不存旧样本，只在损失里加一项，惩罚「对旧任务重要的权重」偏离旧解 |
| EWC | 弹性权重巩固：用 Fisher 对角线当每根弹簧的劲度 |
| Fisher 信息 | 参数对预测分布有多敏感；对角元大，说明这个权重稍动，旧任务的对数似然就掉 |
| 经验 Fisher | 用训练损失对参数的梯度平方，在旧数据上取平均，近似真正的 Fisher 对角 |
| λ（ewc_lambda / e_lambda） | 弹簧总开关：越大越不许动旧的重要权重 |
| Online EWC | 不给每个旧任务各存一份 Fisher，而是按衰减系数叠成一张重要性图 |
| SI | 突触智能：用训练轨迹上「梯度 × 位移」的路径积分估计重要性，不必再扫一遍旧数据 |
| MAS | 记忆感知突触：看网络输出函数对参数的敏感度，可以不需要标签 |
| LwF | 无遗忘学习：不钉权重，而用旧模型在新数据上的输出当软标签做蒸馏 |
| Laplace 近似 | 把旧任务的参数后验近似成高斯，均值是旧最优解，精度矩阵用 Fisher |

## 2. 问题

核心问题只有两个，后面所有公式和命令都围着它们转。

第一个：EWC 怎么知道哪些权重对旧任务重要？网络里几万到几百万个参数，你不可能人工点名。EWC 的答案是 Fisher 信息矩阵的对角线。对角元大的位置，旧任务的对数似然对这个权重敏感，更新时就少动；对角元接近 0 的位置，旧任务几乎不在乎，留给新任务去改。本课要你自己画出这张对角线的直方图，用图确认它是稀疏的：大部分钉子很软，少数钉子很硬。

第二个：没有旧数据时，这根弹簧还能不能用？EWC、SI、MAS、LwF 被归到同一类，正因为它们在学新任务时可以不回放旧图像。这和「它在 class-incremental 上一定赢」是两件事。Kirkpatrick 等人的主实验是 Permuted MNIST（像素被固定打乱，类别集合不变，更接近 domain-incremental）。本课主线却是 Split MNIST / Split CIFAR-10（每段只给两个新类，测的时候十个类一起考）。你要在这两种设定的缝里看清楚：算法能跑，和旧任务还能考过，不是同一个结论。

连带要拆开的还有三件容易混的事。

均匀 L2 和 EWC 不是一种弹簧。给所有权重加同样的 $(\theta - \theta^*)^2$，等于假定每个参数一样重要。EWC 原文图 1 把这条路画成绿箭头：旧任务是保住了，新任务的可行域被勒死。EWC 的红箭头能拐进两块低损失区的交集，全靠 $F_i$ 不是常数。

「重要」有至少四种量法。EWC 用 Fisher（预测分布对参数的敏感度，需要标签或至少需要旧任务的似然）；SI 用你已经走过的优化路径；MAS 用输出函数的敏感度，可以在无标签数据上累加；LwF 干脆不估计权重重要性，改钉输出。本课把四种都跑一遍，是为了让你看见「正则」这个词下面至少有两条完全不同的钉子：钉参数，或钉输出。

λ 不是「调大就更不忘」。它是稳定性-可塑性平面上的滑块。λ = 0 时惩罚消失，EWC 退回 naive；λ 大到一定程度，新任务的梯度推不动被钉住的方向，可塑性先死。本课的交付之一就是把这条轨迹画出来，而不是报告某一个「最佳 λ」。

本课要回答、并且必须用实验回答的具体问题：

1. Fisher 对角线是不是真的长尾分布？
2. λ 从 0 扫到很大，旧任务保持和新任务准确率怎样对掉？
3. 同一份 Split MNIST，naive、EWC、SI、LwF 谁在稳定性-可塑性平面上站得住？
4. 不给旧图像，EWC 在 class-incremental 上还能不能单独当补丁用？

第 4 题的合格答案不是口号。合格答案长这样：EWC 的计算图不需要旧样本，只需要旧参数和一张重要性图；因此「能用」。同一套实现放到 Split CIFAR-10 的 class-incremental 协议上，旧类的 logit 在新任务交叉熵里根本不被看见，弹簧往往钉不住分类头，于是「不够用」。下一课的回放就是冲着这道裂缝来的。

## 3. 准备

- 第 01 到 03 课的概念要在：三种增量设定、稳定性-可塑性、Average Accuracy / Forgetting / BWT。第 04 课读过即可，本课不改语言模型。
- 会用 Python、虚拟环境和 PyTorch。本课命令在 CPU 上就能把 Split MNIST 跑完；Split CIFAR-10 建议一块 8 GB 以上的 GPU。
- 磁盘留 2 GB 给 MNIST / CIFAR-10 和两个仓库。Avalanche 在首次构造 `SplitMNIST` 时会按 torchvision 的默认位置下载数据。
- 两个锚定仓库各开一个干净虚拟环境，不要和日常训练环境混装。Avalanche 当前文档的安装入口是 `avalanche-lib`；Mammoth 按官方文档 `pip install -r requirements.txt`，并用仓库根目录的 `python main.py`。旧博客里的 `python utils/main.py` 以你克隆到的 README 为准，不要混用。
- 浏览器实验不装任何库。先做网页上的「Fisher 钉钉子」，再碰仓库。
- 准备一个笔记目录，至少记下：仓库 commit、随机种子、λ 或 c 的取值、每个任务结束时的逐任务准确率。本课不要求你对上 Nature 论文里的 Atari 分数。

## 4. 学习目标

1. 在白纸上写出 EWC 损失，并指出 Avalanche 实现里少了论文公式中的 $1/2$、λ 因此不能直接抄论文数值。
2. 用自己的话解释 Fisher 对角线为什么能当弹簧劲度，以及经验 Fisher 在代码里对应哪一行平方梯度。
3. 说出 Online EWC 的一张 Fisher 是怎么用 γ 叠出来的，以及它和「每个旧任务一份惩罚」在内存上差什么。
4. 对照 SI 的路径积分和 MAS 的输出敏感度，说明它们各自还要不要在任务结束时再扫一遍旧数据。
5. 写出 LwF 的蒸馏项，并说明它钉的是输出而不是权重。
6. 在稳定性-可塑性平面上标出 λ = 0、中等 λ、过大 λ 三个点，并解释 class-incremental 上 EWC 单独使用时常见的失败模式。
7. 独立跑通 Avalanche 的 `EWC` 与 Mammoth 的 `ewc_on` / `si` / `lwf`，留下命令和逐任务准确率。

## 5. 原理

五个机制按同一节奏走：为什么需要、怎么运转、精确定义、代码落在哪、怎么证明做对了。对照仓库时以你克隆到的文件为准；路径在写课当天已用 Avalanche 最新 API 文档和 Mammoth 文档站的源码页核对。

### 5.1 为什么均匀弹簧不够，必须按重要性加权

网络过度参数化。学完任务 A 之后，参数空间里通常有一整块区域，任务 A 的损失都低。任务 B 的低损失区往往和这块区域有交集。直接按任务 B 的梯度走，会冲出任务 A 的安全区；给所有参数加同一根弹簧，又会把可行域收成一个太小的球，B 学不进去。EWC 的想法是：安全区其实是个椭圆，椭圆的扁长方向对应「旧任务不敏感」的参数，正好留给 B。

类比：你在一面钉满照片的墙上再钉一张新的。每根钉子的松紧不同。旧照片最靠它定位的那几根，你几乎不许拔；旁边那些可有可无的，可以拔下来给新照片用。类比失效处：墙上的钉子是离散的、互不相关的；网络参数之间有相关，完整的 Fisher 是矩阵不是对角。EWC 为了算得动，把矩阵丢掉，只留对角线。后文 5.2 会写清这个近似付出了什么。

贝叶斯写法能把这根弹簧说成「旧任务的后验当新任务的先验」。数据分成 $\mathcal{D}_A$ 和 $\mathcal{D}_B$ 之后：

$$
\log p(\theta\mid\mathcal{D}) = \log p(\mathcal{D}_B\mid\theta) + \log p(\theta\mid\mathcal{D}_A) - \log p(\mathcal{D}_B)
$$

左边仍是全部数据下的参数后验。右边第一项只含任务 B 的似然，任务 A 的全部信息被吸进 $p(\theta\mid\mathcal{D}_A)$。真正的后验算不出，EWC 用 Laplace 近似：把它看成高斯，均值取 $\theta_A^*$，精度取 Fisher 的对角。于是「靠近旧解、并且按重要性加权」从一句直觉变成一项二次惩罚。

验证这件事的最小实验是论文图 1 的三条箭头：无约束、均匀 L2、EWC。本课浏览器实验就是把这张图做成可拖的 λ。你要先预测「λ 变大时，新任务最优点会沿着哪根轴被卡住」，再运行。

### 5.2 Fisher 对角线：重要性从哪来

Fisher 信息量的是：参数动一点点，模型的预测分布会变多少。对单个参数 $\theta_i$，在旧任务最优解处：

$$
F_i = \mathbb{E}_{x\sim\mathcal{D}_A}\left[\left(\frac{\partial\log p_{\theta}(y\mid x)}{\partial\theta_i}\right)^2\right]_{\theta=\theta_A^*}
$$

对角元大，说明这个权重稍改，旧任务的对数似然就掉，它就是该被钉住的钉子。对角元接近 0，旧任务几乎感觉不到它，可以交给新任务。

三个必须写进笔记的性质（Kirkpatrick 等人引用 Pascanu & Bengio 2013）：在极小值附近，Fisher 等价于损失的二阶导；它可以只靠一阶梯度算出来，大模型也负担得起；它半正定，拿来当弹簧劲度不会把惩罚做成鞍。完整 Fisher 是 $|\theta|\times|\theta|$ 的矩阵，存不下也乘不起，所以大家用对角线。这就是「对角高斯」近似：假装参数互相独立。

代码里你几乎看不到「先算 Hessian 再取对角」。Avalanche 的 `EWCPlugin.compute_importances` 在旧经验的训练集上再扫一遍：前向、用当前 `criterion` 算损失、反向，然后把每个参数的梯度平方累加，最后除以 dataloader 的长度。对应源码是 `avalanche/training/plugins/ewc.py` 里这句：`imp.data += p.grad.data.clone().pow(2)`。这是经验 Fisher：用已观测标签的损失梯度平方，而不是对模型自己的预测分布再采样。分类问题上两者常常接近，写报告时仍要写明你用的是哪一种。

Mammoth 的 `ewc_on` 更靠近「对单样本对数似然的梯度平方」。`models/ewc_on.py` 的 `end_task` 对每张训练图单独前向，用 `nll_loss` 的相反数构造项，再累加 `get_grads() ** 2`，最后除以 `len(train_loader) * batch_size`。它还乘了一个 `exp_cond_prob`。读这一段时不要跳过：这是在线 EWC 的实现细节，和 Avalanche 的「按 minibatch 平均平方梯度」不是同一条公式，λ 更不能跨仓库比较。

直方图该长什么样？把所有 $F_i$ 画出来，你会看到长尾：绝大多数靠近 0，少数大几个数量级。这正是弹簧能工作的原因。若直方图是一块砖，EWC 就退化成均匀 L2。本课 CPU 实验会把「对角线稀疏」钉成断言；仓库实验里你自己 `hist` 一次 `importances` 里的张量。

类比失效再补一句。Fisher 量的是「局部曲率」，不是「这个权重在因果上属于任务 A」。两个任务若共用输出层附近的特征，Fisher 会在深层重叠，EWC 原文图 2C 用 Fisher 重叠度说明了这一点：输入差得远时浅层分开，输出域共享时深层仍共用。重叠不是 bug，它表示表示在共享。共享过度时，弹簧会把对新任务有用的方向也钉死，这就是 λ 过大时可塑性死亡的几何来源。

### 5.3 EWC 损失：弹簧怎样进总损失

论文公式（Kirkpatrick et al., arXiv:1612.00796，式 3）：

$$
\mathcal{L}(\theta)=\mathcal{L}_B(\theta)+\sum_i\frac{\lambda}{2}F_i(\theta_i-\theta_{A,i}^*)^2
$$

$\mathcal{L}_B$ 只在当前任务的数据上算。$\theta_A^*$ 是任务 A 结束时的参数快照。$F_i$ 是上一节的对角元。λ 是你要扫的那个滑块。到第三个任务时，可以给 A、B 各留一项二次惩罚，也可以合成一项：两个二次型的和还是二次型。

Avalanche 的 `before_backward` 写的是：

$$
\texttt{strategy.loss}\mathrel{+}=\texttt{ewc\_lambda}\times\sum_i F_i(\theta_i-\theta_i^*)^2
$$

没有论文里的 $1/2$。于是 Avalanche 的 `ewc_lambda=40` 并不等于论文的 $\lambda=40$。官方教程里还有一处把 `EWCPlugin(ewc_lambda=0.001)` 和回放插件拼在一起的例子，那是在演示插件组合，不是给你的扫描起点。扫描时以你自己仓库里的实现为准，从 0、1、10、100、1000 这种对数网格起步，看平面上的轨迹，不要抄任何一篇博客里的「最佳 λ」。

`mode="separate"` 时，每个已结束的 experience 各留一份 `saved_params` 和 `importances`，惩罚对历史经验求和。这是论文说的「两项二次惩罚」。`mode="online"` 时只保留上一份，并用 `decay_factor` 把旧重要性折进新图。插件源码要求：用 `online` 就必须给 `decay_factor`，反之亦然。策略包装器 `avalanche.training.EWC` 的文档还写过 `onlinesum` / `onlineweightedsum` 这种更细的名字；你装好之后先 `help(EWC)` 或读当前 `strategy_wrappers.py`，不要只凭一篇过期 API 页传参。

第一段经验（`train_exp_counter == 0`）没有旧任务，`before_backward` 直接返回，损失就是普通交叉熵。弹簧从第二段才挂上。任务结束时 `after_training_exp` 才算重要性、才拷贝参数。顺序反了就会用错 $\theta^*$。

验证三件事：λ = 0 时总损失与 naive 相同；λ 很大时参数更新范数掉下来，尤其是 $F_i$ 最大的那些坐标；把某一层的重要性整层置零，相当于宣布这层对旧任务不重要，旧任务遗忘应上升。第三件适合放在第 11 节的改造清单里做。

### 5.4 Online EWC：一张 Fisher 记所有旧任务

每个任务存一份完整对角 Fisher 和一份参数快照，任务一多内存就线性涨。Schwarz 等人 2018 年的 Progress & Compress（arXiv:1805.06370）给出在线写法：新任务结束时算出当前 Fisher $F^{(t)}$，再和历史合并：

$$
\tilde{F}^{(t)}=\gamma\tilde{F}^{(t-1)}+F^{(t)}
$$

γ 接近 1 时旧任务的钉子衰减慢；γ 小则更偏向最近的任务。锚点通常改成「到目前为止的参数」，不再为每个旧任务各留一个 $\theta^*$。

Mammoth 的 `ewc_on` 就是这条路。`models/ewc_on.py` 里 `NAME = 'ewc_on'`，兼容 `class-il` / `domain-il` / `task-il`，命令行要 `--e_lambda` 和 `--gamma`。`end_task` 先在当前训练集上累加平方梯度得到 `fish`，若已有历史则 `self.fish *= gamma; self.fish += fish`，再 `self.checkpoint = get_params().clone()`。`observe` 里一旦存在 checkpoint，先用 `get_penalty_grads()` 把 $2\,\lambda\,F\odot(\theta-\theta^*)$ 写进梯度，再反传当前交叉熵。注意：`penalty()` 函数按 $\lambda\sum F(\theta-\theta^*)^2$ 定义，解析梯度却带了因子 2，和把 $1/2$ 写进公式、λ 定义不同的写法要分清。

Avalanche 的 `mode="online"` 用的是 `decay_factor * old_imp + curr_imp`，和 Mammoth 的 γ 是同一类衰减，符号相反的习惯没有：两边都是「旧的乘一个小于 1 的系数再加新的」。

Online 的代价是历史被压扁。很早以前、和当前任务 Fisher 重叠很小的钉子，会被 γ 一次次缩小。若你的序列里任务 1 和任务 5 差得很远，separate 模式通常更稳，也更占内存。本课 Split MNIST 只有 5 段，两种模式都能跑；写笔记时标明你用的是哪一种。

### 5.5 SI：重要性沿训练轨迹积分，不必再扫旧集

EWC 在任务结束时要再遍历旧训练集才能估计 $F$。有时旧集当场就丢掉，有时你不想为估计重要性再付一次完整 epoch。Zenke、Poole 和 Ganguli 的 Synaptic Intelligence（ICML 2017，arXiv:1703.04200）改用量：这个参数刚刚在优化轨迹上为降低损失出过多少力。

记 $g_i=\partial\mathcal{L}/\partial\theta_i$。参数沿连续时间 $\theta_i(t)$ 移动时，它对损失下降的贡献是路径积分：

$$
\omega_i=\int g_i(\theta(t))\,\dot{\theta}_i(t)\,dt
$$

离散实现就是每一步累加「更新前梯度 × 实际位移」。任务结束时，把这段 $\omega_i$ 按位移平方归一化，避免「走得很远但其实不重要」的参数被高估：

$$
\Omega_i\leftarrow\Omega_i+\frac{\omega_i}{(\theta_i-\theta_i^{\mathrm{prev}})^2+\xi}
$$

ξ（论文和 Mammoth 里的 `xi`，Avalanche 里的 `eps`）防止分母为零。学下一个任务时，损失变成：

$$
\tilde{\mathcal{L}}=\mathcal{L}_{\mathrm{new}}+c\sum_i\Omega_i(\theta_i-\theta_i^*)^2
$$

c 的地位等于 EWC 的 λ。Mammoth `models/si.py` 里 `small_omega` 在 `observe` 中按 `grad * (pre_params - post_params)` 累加，`end_task` 把它折进 `big_omega`，`get_penalty_grads` 返回 `c * 2 * big_omega * (θ - checkpoint)`。Avalanche 的包装器是 `avalanche.training.SynapticIntelligence`，参数名是 `si_lambda` 和 `eps`；文档同时引用了原文 arXiv:1703.04200 和后来讨论单增量任务场景的 arXiv:1806.08568。

SI 的类比：EWC 在终点拍一张曲率快照；SI 在整段下山路上记下谁出过力。失效处：路径积分依赖优化器实际走的路，学习率、batch、梯度裁剪都会改 $\omega$。Mammoth 的 SI 在 `observe` 里把梯度值裁到 1，这会改变 $\omega$ 的尺度，c 同样不能跨实现抄。

验证：同一随机种子、同一模型，关掉 SI（c = 0）应回到 naive；打开 SI 后，任务结束时 `big_omega` 应为非负稀疏向量。若 `big_omega` 几乎是均匀的，ξ 或训练步数很可能不对。

### 5.6 MAS 与 LwF：钉输出函数，或干脆钉输出

MAS（Memory Aware Synapses，Aljundi et al.，ECCV 2018，arXiv:1711.09601）问的是另一个问题：参数动一点点，网络的输出函数变多少？它不需要旧标签。对输出 $F(x)$（论文用输出向量的平方 L2 作为标量函数），重要性是：

$$
\Omega_i=\frac{1}{N}\sum_{k=1}^{N}\left\|\frac{\partial F(x_k)}{\partial\theta_i}\right\|
$$

新任务同样加 $\lambda\sum_i\Omega_i(\theta_i-\theta_i^*)^2$。MAS 可以在无标签数据上更新重要性，论文把它连到「按测试条件决定该忘什么」。本课不把 MAS 当成必须跑通的锚定命令：Avalanche / Mammoth 是否带同名实现，以你克隆到的模型列表为准，不要编一个 `--model mas`。机制上你要记住：EWC 钉的是带标签的似然曲率，MAS 钉的是输出敏感度，后者在「旧数据只剩未标注图像」时仍然可算。

LwF（Learning without Forgetting，Li & Hoiem，ECCV 2016 / 期刊版 arXiv:1606.09282）走另一条路。它不估计权重重要性。学新任务之前，先用旧模型把当前能拿到的输入跑一遍，记下旧输出；训练时既拟合新标签，又用蒸馏把旧输出的软分布留住：

$$
\mathcal{L}=\mathcal{L}_{\mathrm{new}}(\theta;x,y)+\alpha\,\tau^{2}\,\mathrm{KL}\big(\sigma(z^{\mathrm{old}}/\tau)\,\|\,\sigma(z^{\mathrm{new}}/\tau)\big)
$$

$\tau$ 是温度。Avalanche 的 `avalanche.training.LwF` 把这两个旋钮叫 `alpha` 和 `temperature`。Mammoth 的 `models/lwf.py` 里是 `--alpha`（默认 0.5）和 `--softmax_temp`（默认 2），蒸馏写成 `modified_kl_div(smooth(softmax(old), T), smooth(softmax(new), T))`。这里有一处实现细节必须写进笔记：Mammoth 先做 softmax，再对概率做 $p^{1/T}$ 后重新归一；常见教科书是先把 logit 除以 T 再 softmax。两者都在做「软化旧分布」，数值上不是同一个温度。对照论文公式时不要混。

LwF 的 `begin_task` 还会在旧骨干上只热身新分类头，再缓存当前训练集的旧 logit。它兼容 `class-il` 和 `task-il`，不宣称支持 general-continual。蒸馏用的是新任务的输入，不是旧任务的图像：这就是「没有旧数据」的含义。失效处也在这里。新任务图像若和旧任务差得很远，旧模型在这些图像上的输出并不能代表旧任务自己的决策面，蒸馏会钉错地方。class-incremental 里旧类 logit 在新数据上往往被压扁，LwF 单独使用时和新任务抢分类头，遗忘仍然可以很重。本课要你把它和 EWC 放在同一张表里，不是为了宣布谁永远第一，而是为了分清「钉权重」和「钉输出」。

### 5.7 没有旧数据：能跑，和够用，是两件事

EWC / SI / MAS / LwF 的共同卖点是：学任务 B 时可以不把任务 A 的图像留在磁盘上。EWC 和 SI 留下的是参数加一张和参数同形状的重要性图；LwF 留下的是一份旧模型（或旧模型在新数据上的 logit）。按「写到哪里」来归类，它们都写在慢速权重里，外加很小的辅助状态。

够不够用，取决于设定。

domain-incremental（域增量：类别集合不变，输入分布变，例如 Permuted MNIST）上，旧任务的分类头仍被新任务的交叉熵看见，弹簧钉的是「同一组类别在新像素排列下别把旧排列忘光」。EWC 原文主要在这个设定上证明它比均匀 L2 和 dropout 更能续接多个排列。

class-incremental 上，新任务 batch 的标签只含新类。交叉熵不会因为你把旧类 logit 弄崩而受罚。Fisher 若主要来自旧任务训练结束时的那一遍，对「新任务阶段分类头被改坏」的约束常常不够。结果是：算法能跑完，旧类测试准确率仍可能掉到接近乱猜。这不是你没调好 λ 的偶然现象，是这一类方法在「无任务标识、无旧样本、共享分类头」下的结构弱点。

所以本课交付里那句「没有旧数据时 EWC 还能不能用」，标准答案分两层：计算上能用；在本课的 Split CIFAR-10 class-incremental 协议上，它通常不能单靠自己把旧任务保持在可接受的水平。下一课把旧样本带在身上，就是承认这根弹簧需要另一类零件配合。第 08 课的 GDumb 还会进一步打脸：有时「背包里的样本重新训一个模型」比精巧的正则更强。现在先把弹簧本身做会。

## 6. 源码导读

读代码按一条样本的真实路径走，不要按目录字母序。两个仓库的职责不同：Avalanche 把 EWC 做成可插进任意监督策略的 plugin；Mammoth 把每种方法做成一个 `ContinualModel`，用 `observe` 写训练步。对照时以你检出的 commit 为准。

| 路径 | 零件 | 带着什么问题读 |
|---|---|---|
| `avalanche/training/plugins/ewc.py` 中的 `EWCPlugin` | 弹簧本体 | `before_backward` 如何把惩罚加进 `strategy.loss`？`compute_importances` 平方的是谁的梯度？ |
| `avalanche/training/supervised/strategy_wrappers.py` 中的 `EWC` | 策略包装 | 它是不是只是 `SupervisedTemplate` 加上一个 `EWCPlugin`？`mode` 当前接受哪些字符串？ |
| `avalanche/training/plugins/synaptic_intelligence.py` | SI | `si_lambda` / `eps` 在哪一步改 `strategy.loss`？和 EWC 是否抢同一份 `saved_params`？ |
| `avalanche/training/supervised/strategy_wrappers.py` 中的 `LwF`、`Naive` | 蒸馏与下限 | LwF 的 `alpha`、`temperature` 默认要你自己传；Naive 是否真的零正则？ |
| `avalanche/benchmarks/classic/cmnist.py` 中的 `SplitMNIST` | 数据流 | `n_experiences=5` 时每段几个类？`return_task_id=False` 时任务标签是不是全 0？ |
| `mammoth/models/ewc_on.py` | 在线 EWC | `end_task` 里 Fisher 如何按样本累加？`gamma` 乘在旧图还是新图上？ |
| `mammoth/models/si.py` | SI | `small_omega` 的符号是 `grad * (pre - post)` 还是反过来？`xi` 加在分母哪里？ |
| `mammoth/models/lwf.py` | LwF | `smooth` 是对 logit 除温还是对概率做幂？`begin_task` 热身的是哪一层？ |
| `mammoth/models/utils/continual_model.py` | 公共底座 | `observe` / `begin_task` / `end_task` 谁必须实现？`COMPATIBILITY` 挡住哪些设定？ |
| `mammoth/main.py` | 入口 | `--model ewc_on` 如何映射到 `models/ewc_on.py`？ |

Avalanche 侧先看惩罚何时注入。`EWCPlugin.before_backward` 在第一次经验时直接 return；之后对 `named_parameters` 逐个取当前值、历史值和重要性，做 `imp * (cur - saved).pow(2)` 再求和。新长出来的参数（键不在 `saved_params` 里）不参与惩罚，这是给动态扩头留的口子，本课用固定 `SimpleMLP` 时通常碰不到。`after_training_exp` 调用 `compute_importances` 时会把模型切到 `eval()`，但若设备是 CUDA 且网络里有 `RNNBase`，它会被迫切回 `train()` 以避免 CUDA 在 eval 模式下反传失败。你的 MLP 没有这个问题，读到这段注释即可。

`compute_importances` 用的是 `strategy._criterion` 和 `strategy.experience.dataset`，也就是刚训完的那一段经验的训练集，不是测试集，也不是过去所有经验的并集。separate 模式下，任务 3 开始时惩罚里同时有任务 1 和任务 2 的两份对角 Fisher，各自锚在各自结束时的参数。不要以为最新一份 Fisher 已经包含全部历史。

策略包装器 `avalanche.training.EWC` 的构造函数是关键字参数（`*, model, optimizer, ...`）。官方 From Zero to Hero 教程里还有一种更老的位置参数写法：`Naive(model, optimizer, criterion, train_mb_size=100, ...)`。两种在你安装的版本里哪一种能跑，以 `help(EWC)` 为准。教程里 `from avalanche.training.supervised import EWC` 和 API 页的 `from avalanche.training import EWC` 指向同一包装；不要同时混用两个名字空间里各抄一半参数。

Mammoth 侧先看 `EwcOn.end_task`。它先把 `fish` 初始化成和 `get_params()` 同形状的零向量，再对 `dataset.train_loader` 做双重循环：外层是 minibatch，内层是单张图。单张图上用 `LogSoftmax` 加 `nll_loss(..., reduction='none')` 的相反数，平均之后反传，再 `fish += exp_cond_prob * grads ** 2`。扫完除以 `len(loader) * batch_size`。历史合并是先 `self.fish *= gamma` 再加。`observe` 在存在 checkpoint 时先 `set_grads(get_penalty_grads())`，然后 `loss.backward()` 把交叉熵梯度累加上去。这意味着惩罚梯度不走计算图，是解析写进去的。调试时若只在 `loss` 上设 breakpoint，会看不见弹簧。

`SI.observe` 的顺序值得画时间线：先克隆 `pre_params`，再反传当前损失得到 `cur_small_omega = grads`，若已有 `big_omega` 就把惩罚梯度加进去，裁剪，`step`，再用 `cur_small_omega *= (pre_params - post_params)` 累进 `small_omega`。也就是说 ω 用的是「未加弹簧之前的损失梯度」乘「实际发生的位移」。实际位移已经被弹簧改过，路径积分和纯 SGD 轨迹不同。这是实现选择，写进笔记。

`Lwf.observe` 只在传入 `logits` 时加蒸馏。这些 logits 来自 `begin_task` 里用当前网络（热身新头之后）在训练集上缓存的输出，并挂到 dataset 的 `extra_return_fields`。蒸馏范围是 `[:, :n_past_classes]`，当前交叉熵用 `[:, :n_seen_classes]`。旧类靠软标签活着，新类靠硬标签学习。若你发现旧类从任务 2 起准确率立刻崩，先打印 `n_past_classes` 和 `logits` 的第二维，确认蒸馏没有接到空切片上。

公共约定。Mammoth 用文件名当 `--model` 的值：`models/ewc_on.py` 对应 `--model ewc_on`。每个文件只允许一个模型类。数据集同样：`datasets/seq_mnist.py` 的 `NAME = 'seq-mnist'`，5 个任务、每任务 2 类、设定是 `class-il`；`seq-cifar10` 同结构，图像 32×32。结果默认写到 `data/results/<setting>/<dataset>/<model>/logs.pyd`。不传 `--wandb_project` 和 `--wandb_entity` 就不会连 WandB。

读完应能回答下面四个问题，答不出就回到对应文件：

1. Avalanche 的经验 Fisher 除以的是 dataloader 长度，Mammoth `ewc_on` 除以的是样本数近似值。两者差一个 batch 因子吗？
2. 为什么 `ewc_on` 的惩罚梯度要在 `backward` 之前 `set_grads`？
3. LwF 在 Mammoth 里到底有没有一份被冻结的 `old_net`？还是只缓存了 logit？
4. `SplitMNIST(..., return_task_id=False)` 时，EWC 文档说「不使用任务标识」到底意味着评估时会不会按任务掩码分类头？

第 4 题连回第 03 课：class-incremental 的主指标是不给任务标识的准确率。Avalanche 默认评估是否掩码，要看你挂的 metric，不要假设框架替你做了 class-IL。Mammoth 在 `class-il` 数据集上会同时报 class-IL 和 task-IL，后者推理时掩掉非本任务类别。写表时两列都留，讨论时以 class-IL 列为准。

## 7. 实验

三层都要做。浏览器先建立椭圆和 λ 的手感；CPU 实验把「λ = 0 等于 naive、λ 过大新任务学不会」钉成断言；锚定仓库才碰真实 Split MNIST / Split CIFAR-10。每一层先写预测，再跑，再对照。

### Step 0: 浏览器实验「Fisher 钉钉子」

打开本课页面上的交互实验。画面是二维权重空间：灰线是旧任务损失的等高线，椭圆是 Fisher 给出的局部二次近似，奶油色区域是新任务的低损失区。你可以拖动 λ。

先预测再运行，预测题是实验过关条件的一部分：

1. λ = 0 时，新任务最优点会落在新任务低损失区的哪里？旧任务损失会不会离开灰色安全区？
2. λ 调到中等，最优点是沿椭圆长轴滑动，还是被吸到椭圆中心？
3. λ 极大时，新任务准确率对应的那个点还会不会离开 $\theta_A^*$？
4. 把椭圆压扁（模拟只有一个方向 Fisher 很大）时，新任务还会不会沿着扁的那根轴学到东西？

合格预测：λ = 0 冲出旧任务安全区；中等 λ 停在两块区域的交集附近，并且主要沿 Fisher 小的轴移动；λ 极大则钉在 $\theta_A^*$，新任务学不会。压扁的椭圆允许沿长轴（不重要方向）移动。运行后系统用同一套二维二次型核对你的选择。改滑块会作废上次运行，必须重新预测。

这个实验验证的是 5.1 和 5.3，不验证 CIFAR 上的绝对准确率。二维高斯加对角 Fisher 已经足够让你看懂弹簧；真正的网络还有非对角耦合，那是对角近似的失效处，写在预测题旁边的说明里。

### Step 1: CPU 机制实验

在课程仓库的 `experiments/` 目录：

```bash
python3 run.py run 05
```

`python3 run.py run 05` 现在应当全绿，结果写入 `artifacts/lesson05/result.json`。打开文件后 `checks` 应全部为真，键名是 `lambda0_matches_naive`、`lambda0_forgets_task1`、`lambda0_learns_task2`、`large_lambda_keeps_task1`、`large_lambda_blocks_task2`。

任务 B 是任务 A 的标签翻转，重要权重必须动才能学会 B。本机一次运行（seed 5）：λ=0 与 naive 权重 L2=0.0，任务 A 掉到 0.025、任务 B 到 0.975；λ=2e5 时 A 保持 0.975，B 只有 0.025。换机器会变，方向不应变。`summary` 里的阈值：λ=0 与 naive 权重 L2<1e-9；大 λ 时 A>0.85 且 B<0.30。

这些数字钉的是弹簧两端的机制，不是 PNAS 的 Atari 数字，也不能当成 Split MNIST 成绩。λ 扫描和 Fisher 直方图在后面的 Avalanche 步骤里做。

### Step 2: 安装 Avalanche 并构造 Split MNIST

独立虚拟环境中安装（官方 PyPI 包名，教程页曾固定演示 `avalanche-lib==0.6`；你可先装当前稳定版，装完打印版本写入笔记）：

```bash
pip install avalanche-lib
```

需要对照源码时再克隆：

```bash
git clone https://github.com/ContinualAI/avalanche.git
```

下面这段是 Avalanche 文档里标准循环的缩小版：5 段 Split MNIST，固定种子，先跑 Naive 当下限。保存成你自己的 `lesson05_naive.py`。

```python
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.models import SimpleMLP
from avalanche.training import Naive
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics, loss_metrics
from avalanche.logging import InteractiveLogger

device = torch.device("cpu")
benchmark = SplitMNIST(n_experiences=5, seed=1, return_task_id=False)
model = SimpleMLP(num_classes=10)
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = CrossEntropyLoss()
eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    loss_metrics(epoch=True, experience=True, stream=True),
    loggers=[InteractiveLogger()],
)
strategy = Naive(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_mb_size=128,
    train_epochs=2,
    eval_mb_size=128,
    device=device,
    evaluator=eval_plugin,
)
results = []
for experience in benchmark.train_stream:
    strategy.train(experience)
    results.append(strategy.eval(benchmark.test_stream))
```

若你安装的版本要求位置参数而不是关键字，按 `help(Naive)` 改。预期：每一段训完再测全部 5 个测试经验，任务 1 的准确率随后续训练下降。把最后一次 `eval` 里每个 experience 的准确率抄进表，这是后面所有方法的下限。`train_epochs=2` 是为了在 CPU 上当天跑完；它不是论文配置。

### Step 3: 同一协议换 EWC，并导出 Fisher 直方图

把 `Naive` 换成 `EWC`，其余数据、模型、种子、epoch 不动：

```python
from avalanche.training import EWC

strategy = EWC(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    ewc_lambda=10.0,
    mode="separate",
    train_mb_size=128,
    train_epochs=2,
    eval_mb_size=128,
    device=device,
    evaluator=eval_plugin,
)
```

`ewc_lambda=10.0` 只是扫描网格上的一个点。跑完从 `strategy.plugins` 里找到 `EWCPlugin` 实例，读取 `importances` 里最后一份经验的张量，把所有对角元拼成一维数组画直方图（对数横轴更清楚）。预期：大部分质量堆在靠近 0 的一侧，右尾很长。若直方图接近均匀，先检查你是不是把参数值当成了重要性，或者第一段经验尚未调用 `after_training_exp`。

对照 Step 2 的表：任务 1 在全部任务结束后的准确率，EWC 应当不低于 Naive（同种子、同 epoch）。新任务（最后一段）的准确率允许略低。若两者几乎一样，λ 可能太小；若最后一段接近随机，λ 可能太大。不要在这一步就开始改模型结构。

### Step 4: λ 扫描，画稳定性-可塑性平面

只改 `ewc_lambda`，建议网格：

```text
lambda 网格: 0, 1, 10, 100, 1000
```

每个点记录两个数：旧任务保持（任务 1 在全部任务结束后的准确率，或前 4 个经验的平均），新任务准确率（最后一段经验刚训完时自己的测试准确率）。画成平面上的五个点，横轴旧、纵轴新，和 [第 02 课](02_stability_plasticity.md) 同一张坐标系。

预期轨迹：λ = 0 贴近 Naive（旧低、新高）；λ 增大时点向右再向下走；极大 λ 停在「旧任务还行、新任务学不会」的底部。若五个点挤成一团，说明你的 epoch 太少或模型太弱，两个方法都没真正拟合，先把 `train_epochs` 加到 4 再扫，仍要在笔记里标明。

这一步的产物就是本课交付的 λ 扫描曲线。曲线来自你的机器、你的种子，允许和别人差几个百分点；不允许的是只有一个 λ、却写出「EWC 优于 Naive」这种全称判断。

### Step 5: Avalanche 上对照 SI 与 LwF

同一 `SplitMNIST(seed=1)`、同一 `SimpleMLP`、同一优化器超参：

```python
from avalanche.training import SynapticIntelligence, LwF

si = SynapticIntelligence(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    si_lambda=1.0,
    eps=1e-7,
    train_mb_size=128,
    train_epochs=2,
    eval_mb_size=128,
    device=device,
    evaluator=eval_plugin,
)
lwf = LwF(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    alpha=0.5,
    temperature=2.0,
    train_mb_size=128,
    train_epochs=2,
    eval_mb_size=128,
    device=device,
    evaluator=eval_plugin,
)
```

每个方法重新初始化模型和优化器，不要在已经训过 EWC 的权重上接着训 SI。`si_lambda=1.0`、`alpha=0.5`、`temperature=2.0` 是为了让命令能跑的起点：SI 的 1.0 对应论文里的 c 量级需要你自己扫；LwF 的 0.5 和 2.0 与 Mammoth 默认一致，便于两边对照。把四个方法（Naive、EWC、SI、LwF）的最终平均准确率和任务 1 保持填进同一张表。

预期（方向，不是数字）：在 2 个 epoch 的 Split MNIST 上，正则方法相对 Naive 应能抬高任务 1 的保持，或至少在遗忘指标上同方向改善。若 SI 或 LwF 全面弱于 Naive，先检查是否复用了旧模型、是否 `eval` 时用了不同的 stream。写进笔记的句子必须带设定：种子、epoch、λ / c / α，禁止只写方法名和「更好」。

### Step 6: Mammoth 的 ewc_on / si / lwf

另开环境，克隆并按官方文档安装：

```bash
git clone https://github.com/aimagelab/mammoth.git
```

```bash
pip install -r requirements.txt
```

在仓库根目录先看入口：

```bash
python main.py --help
```

再跑在线 EWC。`--e_lambda` 与 `--gamma` 在当前文档里是必填；下面的数值是扫描起点，不是论文最优：

```bash
python main.py --model ewc_on --dataset seq-mnist --e_lambda 1000 --gamma 1.0 --lr 0.03
```

SI：

```bash
python main.py --model si --dataset seq-mnist --c 0.5 --xi 0.1 --lr 0.03
```

LwF（α 和温度有默认值，仍建议显式写出）：

```bash
python main.py --model lwf --dataset seq-mnist --alpha 0.5 --softmax_temp 2 --lr 0.03
```

日志在 `data/results/` 下按设定 / 数据集 / 模型分目录，每行一个字典。同时看终端打印的 class-IL 准确率。需要缩短冒烟时间时，先 `python main.py --help` 确认 `--n_epochs`、`--debug_mode` 是否存在；模型文档写过 `--debug_mode 1` 时默认只跑很少迭代，只用来确认命令能走通，数字作废。

`seq-mnist` 在文档里是 5 任务、每任务 2 类、`SETTING = 'class-il'`。CPU 上应当能跑完。把 Mammoth 的 class-IL 列和 Avalanche Step 5 的表放在一起时，只比较方向（谁更能保住任务 1），不比较绝对数字：骨干、epoch、数据增强、评估掩码都不保证相同。

### Step 7: Split CIFAR-10（单卡，可选但建议做）

有 GPU 时把数据集换成 `seq-cifar10`（Mammoth）或 Avalanche 的 `SplitCIFAR10`。Mammoth 文档给出的通用命令形态与 README 示例一致，只是把模型换成本课的正则方法。CIFAR 上正则方法相对 Naive 的优势通常比 MNIST 更弱，有时弱到看不出来。这正是本课要你写进「没有旧数据」那一页里的观察，不是实验失败。

缩小配置：先 `--debug_mode 1` 确认数据能下载、命令能结束；再开极少 epoch 的正式扫描。完整 50 epoch 级配置按你机器留到加分项，不要为了凑数字通宵跑一次就停。笔记里分开写「冒烟已通」和「可引用的扫描」。

### Step 8: 书面回答「没有旧数据时 EWC 还能不能用」

用本课跑出来的表写一段，结构固定：

```text
计算：EWC / SI / LwF 在学新任务时有没有读旧图像（有 / 无）
它们各自留下了什么辅助状态（Fisher + 旧参数 / Omega + 旧参数 / 旧输出或旧模型）
Split MNIST 上任务 1 保持相对 Naive 的方向
Split CIFAR-10 上若你跑了，方向是否仍然成立
结论：能用（计算约束）/ 不够用（本课 class-IL 协议下的观察）
下一课准备换哪个零件
```

这段话是验收的一部分。只写「EWC 不需要旧数据所以能用」而不提 class-IL 的失败模式，不算过。

## 8. 配置与预算

| 档 | 数据与模型 | 方法 | 耗时量级 | 用途 |
|---|---|---|---|---|
| 浏览器 | 二维二次型 | 拖 λ | 几分钟 | 建立椭圆直觉 |
| CPU 机制 | 课内合成 / 缩小网络 | `run.py run 05` | 秒到一分钟 | 断言钉死 λ 的两端 |
| 冒烟 | Split MNIST，`SimpleMLP`，2 epoch | Naive 与一个 λ 的 EWC | CPU 上数十分钟内 | 确认命令、导出 Fisher 直方图 |
| 主线 | Split MNIST，同一模型，5 个 λ + SI + LwF | Avalanche 全表 | CPU 上数小时 | λ 扫描与四方法对照 |
| Mammoth 对照 | `seq-mnist` | `ewc_on` / `si` / `lwf` | CPU 上数小时 | 核对另一套官方实现 |
| 加分 | `seq-cifar10` 或 `SplitCIFAR10` | 同一组方法，少 epoch | 单卡数小时 | 观察 class-IL 上正则变弱 |

主线按档「主线」写命令。Mac / 纯 CPU 必须能完成浏览器、CPU 机制、Split MNIST 冒烟和至少一组 λ 扫描。CIFAR 标成加分，不锁死整课。

超参纪律：一次只改一个旋钮。λ 扫描时冻结学习率、epoch、batch、种子。换 SI 的 c 时不要顺手改学习率。Avalanche 的 `ewc_lambda` 和 Mammoth 的 `e_lambda` 量纲不同，禁止把一边的「好数字」填进另一边。

数据下载。MNIST 和 CIFAR-10 都是公开集，Avalanche 走 torchvision 默认根目录，Mammoth 默认 `data/`，可用 `--base_path` 改。公司机器若禁网，事先把数据拷到对应目录。

## 9. 验收

- [ ] 白纸写出论文中的 EWC 损失，并标出 Avalanche 实现缺少 $1/2$。
- [ ] Fisher 直方图已保存，能指出长尾而不是均匀砖块。
- [ ] λ 扫描至少 4 个非零点和 λ = 0，稳定性-可塑性平面上的轨迹从「旧低新高」走向「旧高新低」。
- [ ] Naive、EWC、SI、LwF 四行表填完，每行带种子、epoch 和关键超参。
- [ ] Mammoth 的 `ewc_on`、`si`、`lwf` 至少各成功跑完 `seq-mnist` 一次（debug 冒烟可另计，正式数字要用非 debug）。
- [ ] 浏览器实验预测通过，且能口述「椭圆长轴是不重要方向」。
- [ ] `python3 run.py run 05` 的 `checks` 全真（`lambda0_matches_naive`、`lambda0_forgets_task1`、`lambda0_learns_task2`、`large_lambda_keeps_task1`、`large_lambda_blocks_task2`）。
- [ ] 书面回答「没有旧数据时 EWC 还能不能用」，包含「能跑」和「在本课 class-IL 上够不够」两层。
- [ ] 能指出 SI 不必在任务结束时再扫旧图像，EWC 需要；LwF 钉的是输出。

口头抽查（给自己或给同伴）：把 $F_i$ 全部换成常数，EWC 变成什么？答案是均匀 L2，论文绿箭头。再问：class-incremental 的交叉熵为什么看不见旧类？答案是当前 batch 没有旧标签，梯度不经过旧类 logit 的正确类项。答得出，这一课的机制就算进脑子了。

## 10. 排错

| 症状 | 原因 | 验证 | 修法 |
|---|---|---|---|
| `EWC()` 报 unexpected keyword / missing positional | 你装的版本构造函数是位置参数或关键字-only，和抄来的片段不一致 | `help(EWC)` 看签名 | 按当前签名改；不要混用 `avalanche.training` 与 `avalanche.training.supervised` 两套抄参 |
| `mode='online'` 立刻 AssertionError | 插件要求 online 必须带 `decay_factor` | 读 `EWCPlugin.__init__` 里两条 `assert` | 补 `decay_factor`（例如 0.9），或改回 `separate` |
| λ = 0 仍比 Naive 忘得少 | 模型和优化器从上一轮接着用，或评估 stream 不同 | 检查是否重新 `SimpleMLP()`、是否每次 `eval(benchmark.test_stream)` | 每个方法、每个 λ 都重新建模型与优化器 |
| Fisher 直方图是一块砖 | 取出的是参数值不是 `importances`；或只看了第一层 bias | 打印 `importances[t][name].data` 的均值与最大值 | 拼接所有参数的重要性；用对数横轴 |
| 极大 λ 时新任务仍很高 | epoch 太少，新任务本来也没学到头；或 λ 还不够大 | 看 Naive 最后一段准确率是否已经很高 | 先加 epoch 让 Naive 真正拟合，再把 λ 往上一个数量级 |
| Mammoth `ewc_on` 要求 `--e_lambda` | 当前文档把 `e_lambda`、`gamma` 标成必填 | `--help` 里是否 `required` | 必须显式传，不要指望默认 |
| `python utils/main.py` 找不到文件 | 你看的是旧博客；当前文档入口是根目录 `main.py` | 仓库根是否有 `main.py` | 用 `python main.py`；以 README 为准 |
| LwF 从任务 2 起损失变 nan | 温度平滑后的概率有 0，再取 log | 打印 `smooth(...)` 的最小值 | 先确认 Mammoth 版本；必要时在笔记记录后换 Avalanche LwF 对照 |
| class-IL 准确率接近随机，task-IL 还行 | 评估时未掩码 vs 掩码的差别，不是 EWC 没干活 | Mammoth 两列一起看 | 主结论用 class-IL 列；task-IL 只能说明「若告诉任务编号还能分」 |
| CUDA 内存不够 | 误把 CIFAR 和过大 batch 放到本课冒烟 | `nvidia-smi` | 退回 Split MNIST CPU；CIFAR 减 batch、减 epoch |
| `pip install avalanche-lib` 与本地克隆源码不是同一版本 | 跑的是 pip 包，读的是 git 里另一份 `ewc.py` | 打印 `avalanche.__version__`，和 git log 对比 | 要么都用 pip，要么 `pip install -e .` 装你正在读的那份 |
| `EvaluationPlugin` 或 `accuracy_metrics` 导入失败 | 你装的版本把评估类换了名字空间 | `help(avalanche.training)` 与官方 From Zero to Hero 第 4 章 | 改用该版本的 `default_evaluator`，或按当前教程改 import；不要为了对齐本课片段降级到随意旧版 |
| `lambda0_matches_naive` 为假 | λ=0 仍走了 Fisher 近端步，或 naive 与 λ=0 不是同一快照 | 看 `lambda0_weight_l2_vs_naive` | 对照 `lesson_05.py`：λ=0 必须与 naive 同一套 SGD |
| `large_lambda_keeps_task1` 或 `large_lambda_blocks_task2` 为假 | 大 λ 没钉住重要方向，或任务 B 不是标签翻转 | 看 `lambda_2e5_acc_task1` / `lambda_2e5_acc_task2` | 本实验 B 是 A 的标签翻转，重要权重必须动才能学 B；确认 `lambda_large` 为 200000 |

Fisher 算得很慢时，先确认你不是在每一步训练里重算。正确位置是任务结束的 `after_training_exp` / `end_task`。若自己改代码把 `compute_importances` 放进 minibatch 循环，那是另一算法，不要再叫 EWC。

## 11. 前沿与改造

前沿怎么做。2017 年之后，正则这条线并没有消失，但很少再单独扛 class-incremental 图像分类的榜。常见接法有三种：和回放拼在一起（Avalanche 教程里 `ReplayPlugin` + `EWCPlugin` 就是最小拼法）；把重要性估计从对角 Fisher 换成更便宜或更在线的量（SI、MAS、后来的各种梯度投影，见 [第 08 课](08_gem_gdumb.md)）；在大模型上把「弹簧」打到 LoRA 或其他低秩更新上，而不是全量权重（[第 11 课](11_olora_treelora.md)）。2024-2026 年讨论 LLM 持续学习时，EWC 更多作为「改权重时记得加锚」的原型，而不是生产配方。近两年主阅读见第 12 节：MoFO 只更新动量最大的坐标，EAFT 用 token 熵压冲突梯度，SAE 正则把惩罚写到特征激活上，AWARe 按激活冻通道。这些做法都不再算全网 Fisher。

我们差在哪。本课用的是对角经验 Fisher、固定 MLP、任务边界清晰的 Split MNIST。真实部署常常没有干净的任务切分，Fisher 的任务结束扫描无处可做，分类头的旧类 logit 照样不被新损失看见。大模型上存一份与参数同形状的 Fisher 本身就贵。不要把本课扫描出来的 λ 写进任何线上配置。

动手改造（01-12 课精简版，四个里选做，预算按 Split MNIST CPU）：

1. 均匀 L2 对照。在 Avalanche 里把 `importances` 整份换成常数 1，保留同一 λ 网格。预期：轨迹更接近论文绿箭头，新任务先死。失败标准：常数重要性反而全面超过对角 Fisher，这时先查你是否把 Fisher 取反了。
2. 关掉任务结束时的 Fisher 扫描，改用训练过程中的梯度平方做滑动平均。模块位置：复制 `compute_importances` 的平方梯度行到 `after_training_iteration`，去掉 `after_training_exp` 的第二遍数据。预期：省一次 epoch，旧任务保持略降。失败标准：训练变慢而不是变快（说明你每步都扫了全量数据）。
3. EWC + 极小回放。按官方教程把 `ReplayPlugin(mem_size=200)` 和 `EWCPlugin` 一起挂到 `SupervisedTemplate`。预期：任务 1 保持高于纯 EWC。失败标准：比纯回放还差很多，优先查两个插件是否抢 dataloader。完整回放对比留给第 06 课。
4. 把 EWC 的锚点从「任务结束的参数」改成「指数滑动平均的参数」。预期：锚更平滑，对提前结束的任务段更稳。失败标准：遗忘上升且新任务也更差，回到原锚点。

顺手复现映射。本课不是课程承诺的五项复现之一。你若把 Permuted MNIST 上的 EWC 对 Naive 做成方向性对照，只能写进个人笔记，不要标成「复现 Kirkpatrick 2017」。正式复现从第 06 课 DER++ 对 ER 开始。

## 12. 论文与延伸

谱系只留本课 CPU 实验真正实现的 EWC。主阅读是 2024-2026：大模型微调很少再算全网 Fisher，弹簧改成动量掩码、token 熵门、SAE 特征约束或激活冻结。

1. Kirkpatrick, Pascanu, Rabinowitz, Veness, Desjardins, Rusu, Milan, Quan, Ramalho, Grabska-Barwinska, Hassabis, Clopath, Kumaran, Hadsell, 2017, *Overcoming catastrophic forgetting in neural networks*, [arXiv:1612.00796](https://arxiv.org/abs/1612.00796)。
贡献：把旧任务后验的对角 Laplace 近似写成弹性弹簧，按重要性减慢学习。机制发明处，不是本课主阅读。
机制：任务结束时用对角 Fisher 当每根弹簧的劲度，新损失里加 $\lambda\sum_i F_i(\theta_i-\theta_i^*)^2$。CPU 实验的近端步就是这项惩罚。摘要写在 MNIST 分类和 Atari 顺序游戏上验证。
和本课：Step 1 的 `lambda0_matches_naive` 对应 λ=0 时惩罚消失；`large_lambda_keeps_task1` 与 `large_lambda_blocks_task2` 对应 λ 极大时钉死旧解。Avalanche 的 λ 扫描对应原文均匀 L2 对照。本课实验答不了 Atari 分数，也答不了非对角 Fisher。
阅读问题：任务 B 是任务 A 的标签翻转，重要权重必须动才能学会 B。你的 `large_lambda_blocks_task2` 若为真，说明弹簧把该动的方向也钉死了。这和「重要的少动、不重要的放开」差在哪一处设定？用本课标签翻转回答。

2. Chen, Wang, Zhang, Lin, Zhang, Sun, Ding, Sun, 2024, *MoFO: Momentum-Filtered Optimizer for Mitigating Forgetting in LLM Fine-Tuning*, [arXiv:2407.20999](https://arxiv.org/abs/2407.20999)。
贡献：微调时每步只更新动量绝对值最大的那一截参数，不需要预训练数据。
机制：在 Adam 每个参数块里，按动量模长取 top-α 再更新，其余坐标本步不动。它改的是优化器掩码，不是 Fisher 二次项，也不存旧样本。摘要写：在拿不到预训练语料的开源权重微调里，微调成绩接近默认算法，通用能力掉得更少。
和本课：CPU 实验的大 λ 是「重要坐标几乎不许动」；MoFO 是「每步只放行动量最大的一小撮」。`large_lambda_blocks_task2` 看见的是全钉死。本课没有动量滤波器，答不了 α 取多少才既学会新任务又少忘。
阅读问题：本课大 λ 几乎钉死全部重要坐标，MoFO 每步只动动量最大的一小撮。用 `large_lambda_blocks_task2` 说明「少更新」已经能保住任务 A；再用 `lambda0_learns_task2` 说明完全放开才能学会任务 B。MoFO 的 α 落在两端之间，本课实验给不出它的取值，只能标出两端。

3. Diao, Yang, Gong, Zhang, Yan, Han, Liang, Xu, Ma, 2026, *Entropy-Adaptive Fine-Tuning: Resolving Confident Conflicts to Mitigate Forgetting*, [arXiv:2601.02151](https://arxiv.org/abs/2601.02151)。
贡献：把 SFT 遗忘归因到低熵、低概率的「自信冲突」token，用熵给交叉熵加门。
机制：标准 CE 对每个 token 一视同仁。EAFT 用 top-K 词汇熵归一化后乘到该 token 的 CE 上：低熵冲突的梯度被压下去，高熵位置仍按普通 SFT 学。改的是损失加权，不算 Fisher，不存旧数据。摘要写在 Qwen 与 GLM、4B 到 32B、数学 / 医学 / 工具调用上，下游接近 SFT，通用能力掉得更少。
和本课：EWC 按参数曲率钉权重；EAFT 按 token 熵压梯度。本课 CPU 实验没有 token 熵，也没有 LLM。`large_lambda_keeps_task1` 只能类比「把破坏性更新压住」；「低熵低概率才是冲突」本课实验答不了。
阅读问题：本课任务 B 把标签整盘翻过来，每个样本都在强迫模型改口。这更像 EAFT 说的自信冲突，还是高熵不确定？用标签翻转设定回答。EAFT 的门控数值本课实验答不了，因为没有 token 分布。

4. Ning, Xue, Lou, Guo, 2026, *From Weights to Features: SAE-Guided Activation Regularization for LLM Continual Learning*, [arXiv:2606.26629](https://arxiv.org/abs/2606.26629)。
贡献：指出 LLM 上按权重算重要性太粗，改在 SAE 特征激活上做稳定 / 可塑双侧约束，并把 EWC 写成单侧权重惩罚的特例。
机制：预训练 SAE 当特征字典。当前任务数据算出特征掩码，之后只留这份掩码。训练时对低掩码特征加保护损失（漂移超出预算才罚），对高掩码特征加引导损失（动得不够才罚）。不回放旧样本。摘要写在 TRACE 与 MedCL 上超过 EWC。HTML 里 TRACE 上该方法 OP 为 0.545，EWC 为 0.447；EWC 的可塑性只有 0.453，无保护微调是 0.569。
和本课：本课对角 Fisher 就是他们说的权重坐标。CPU 实验能看见「单侧惩罚过大则新任务学不会」（`large_lambda_blocks_task2`）。「任务在 SAE 特征里可分、在权重里不可分」那一句，本课 MLP 没有 SAE，答不了。
阅读问题：作者把 EWC 写成只有稳定、没有可塑下限的特例。你的 `large_lambda_keeps_task1` 为真且 `large_lambda_blocks_task2` 为真，是否支持「只罚偏离、不强制该动的方向动起来」？用这两条 check 回答。

5. Ling, Zhang, Zhao, Pan, Li, 2025, *LoRA-Based Continual Learning with Constraints on Critical Parameter Changes*, [arXiv:2504.13407](https://arxiv.org/abs/2504.13407)。
贡献：正交 LoRA 之后，对旧任务重要的 ViT 参数矩阵仍会明显改动；于是直接冻住最关键的那些矩阵，再用 QR 把新 LoRA 正交拼进去。
机制：每学完一个任务，按损失敏感度给 ViT 参数矩阵打分，后续任务冻 top 比例的重要矩阵。新任务只训新的正交 LoRA，旧 LoRA 冻住，用可学习权重拼起来。改的是掩码加低秩更新，不是对角 Fisher。摘要写 Split CIFAR-100 上准确率提高 6.35%，遗忘降低 3.24%。
和本课：冻关键矩阵相当于把对应 $F_i$ 设成无穷大。本课大 λ 是软钉；这篇是硬冻矩阵。本课没有 LoRA，也没有 ViT 矩阵级重要性，答不了「正交 LoRA 为何仍改到关键参数」。
阅读问题：本课把 $F_i$ 换成常数就变成均匀 L2。若你冻住 Fisher 最大的那几根坐标、其余照常 SGD，预期更接近这篇的硬约束还是本课的软弹簧？用 λ 扫描的两端回答。本课实验没有逐矩阵冻结，具体百分比答不了。

6. Lewandowski, Bortkiewicz, Kumar, György, Schuurmans, Ostaszewski, Machado, 2024, *Learning Continually by Spectral Regularization*, [arXiv:2406.06811](https://arxiv.org/abs/2406.06811)。
贡献：把每层最大奇异值钉在 1 附近，用来维持可训练性。
机制：损失里加谱正则，权重矩阵的最大奇异值用幂迭代估，罚它偏离 1，偏置的谱罚向 0。改的是层谱范数，不存旧数据，也不算 Fisher。目标是保持梯度多样性，让新任务还能学。摘要写在持续监督和强化学习里，对超参更不敏感。
和本课：本课弹簧保护旧任务；谱正则保护「还能继续训」。CPU 实验的大 λ 会让任务 B 学不会，这是稳定性过头。谱正则担心的是反过来：训久了最大奇异值涨起来，新任务学不动。本课两任务、固定 MLP，答不了谱范数随任务增长。
阅读问题：本课 `large_lambda_blocks_task2` 为真，说明弹簧可以把可塑性打死。谱正则想保住的是可塑性。这两项能同时开吗？本课实验没有奇异值日志，答不了「开了谱正则之后 B 会不会重新学会」，只能指出两项目标相反。

7. Liao, Lv, Wang, Zheng, Xiao, Tang, 2026, *AWARe: Mitigating Catastrophic Forgetting via Activation-Weighted Adaptive REtention*, [arXiv:2608.11758](https://arxiv.org/abs/2608.11758)。
贡献：用校准集上的激活幅度给参数打分，冻高激活通道，其余继续微调，给多模态大模型用。
机制：前向一次校准样本，按输出通道的序列 L2 再做样本内归一，得到 saliency。全局取 top-ρ 通道，对应权重行的梯度掩成 0。改的是训练时掩码，不改结构，不回放。摘要写可与现有推理引擎兼容。HTML 写常冻自注意力里约 top 30% 的参数。
和本课：激活 saliency 接近「输出对参数有多敏感」，和 Fisher 不是同一张图。本课 Fisher 直方图能看见「少数钉子很硬」；AWARe 的通道级冻结本课没有校准前向，答不了。
阅读问题：本课书面题问「没有旧数据时弹簧还能不能用」。AWARe 用校准集代替旧任务数据来打分。若你只有新任务图像、没有任何校准前向，AWARe 的冻结掩码还做得出吗？对照本课 Step 8 的两层答案：能跑，和 class-IL 上够不够。

8. Elsayed, Mahmood, 2024, *Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning*, [arXiv:2404.00781](https://arxiv.org/abs/2404.00781)。
贡献：用权重效用同时管遗忘和学不动，流式设定、任务边界未知。
机制：效用近似成「把该权重置零损失变多少」，用一阶加对角二阶泰勒。更新时梯度加扰动，再乘 $(1-\bar U)$：有用的少动，没用的多扰动。改的是逐权重更新规则，不需要任务结束扫描。摘要写在数百次非平稳的流式问题上，许多基线准确率随任务下降，UPGD 还能往上走。
和本课：效用门控相当于每步都在算一种重要性，不必等 `after_training_exp`。本课 Fisher 是任务结束扫一遍。CPU 实验有清晰任务边界和一次 Fisher，答不了流式、未知边界。`lambda0_matches_naive` 仍能说明：门控若恒为 0，就退回普通 SGD。
阅读问题：本课改造清单第 2 条把平方梯度挪到每步滑动平均。UPGD 的效用也是在线的。两者差在「只缩小更新」还是「对无用权重加扰动」？本课 CPU 实验没有扰动项，答不了 UPGD 的 σ；你可以只答本课 Fisher 是离线还是在线。

现在整个系统长这样：经验仍然直接写进同一套慢速权重，但写的时候多了一张重要性图，重要坐标被二次惩罚拉住。测的时候你已经会看逐任务准确率和遗忘，也会看 λ 把点拖到平面的哪一侧。缺的零件是旧样本本身。没有它们，class-incremental 的分类头照样可以在新交叉熵里把旧类推下最大 logit。下一课把一个容量有限的背包挂上，看回放为什么是最稳的一类方法，以及 Dark Experience Replay 多蒸馏的那一项到底在防什么。
