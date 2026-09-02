---
id: 15_loss_of_plasticity
title: "学着学着学不动了"
summary: "没有旧任务考试，网络在一长串新任务之后也会失去学习能力。这和遗忘是两件事吗？"
unit: memory
play_tools: []
checkpoints:
  - "学习速度曲线。"
  - "死神经元比例。"
  - "论文复现 #4。"
---

# 第 15 课：学着学着学不动了

> 类型：复现（论文复现 #4，方向性）<br>
> 建议周期：2-4 天<br>
> 硬件：CPU / Mac 即可完成浏览器实验与课内机制实验；官方仓库的缩小配置也在 CPU 上跑<br>
> 锚定仓库：[shibhansh/loss-of-plasticity](https://github.com/shibhansh/loss-of-plasticity)（Dohare et al. 2024 Nature 官方代码；预印本 arXiv:2306.13812）<br>
> 产物：第 $k$ 个任务的学习速度曲线、死神经元比例、SGD / L2 / shrink-and-perturb / continual backprop 对照、复现报告

## 1. 这一课做什么

你现在站在第四幕的第三课。上一课（[第 14 课](14_knowledge_editing.md)）把一条事实定位进某层 MLP 的几个关键，改完再测邻居有没有被带偏。那一课默认网络还学得动：给它新标签、新样本，梯度还能把损失压下去。这一课把这个默认拆掉。

整门课的主干一直是同一圈：

```text
新经验进来
  先决定写到哪里（上下文 / 外挂记忆 / 快速权重 / 慢速权重）
  再决定怎么写（覆盖、追加、压缩、约束、正交、合并）
  写完立刻测：新任务会了没、旧任务还在不在
  长期还要测：还能不能继续学
```

前十四课几乎都在盯第二、第三条：忘没忘、新任务会不会。第四条常常被省略。省略的代价是：你可能造出一个「旧任务还在、新任务刚开始也还行、但第 50 个任务之后怎么训都学不动」的系统。Dohare、Hernandez-Garcia、Lan、Rahman、Mahmood、Sutton 在 2024 年 *Nature* 上把这件事从遗忘里拆出来，叫可塑性丢失（loss of plasticity：学着学着学不动了）。期刊论文 doi 是 [10.1038/s41586-024-07711-7](https://doi.org/10.1038/s41586-024-07711-7)，预印本是 [arXiv:2306.13812](https://arxiv.org/abs/2306.13812)，标题在预印本上写成 *Maintaining Plasticity in Deep Continual Learning*。两份文本讲的是同一组实验，Nature 版把 ImageNet 和强化学习写进摘要，预印本把 Permuted MNIST 的长序列写得更细。本课两边都读，数字以你打开的页面为准。

上一课留下的零件是「局部改一条事实」。没有这一课，你会把「后期任务分数掉了」一律记成遗忘，然后去加大正则、加大回放。那两味药治的是旧任务被冲掉。可塑性丢失的现场是：就算你不考旧任务，新任务自己也学不快了。网络还在做梯度下降，损失曲线却越来越平。

做完这一课，你手里有四样东西。第一，一张「任务序号–学习速度」曲线：横轴是第 $k$ 个任务，纵轴是这个任务在固定预算里能学到多好。第二，各层死神经元（dead unit：对当前任务几乎永远不激活、梯度接近零的隐藏单元）比例随 $k$ 怎么涨。第三，同一设定下 SGD、L2、shrink-and-perturb、continual backprop 四条线。第四，一份方向性复现报告：长序列上标准反向传播的后期任务变慢，continual backprop 或 shrink-and-perturb 能缓解。这是课程复现承诺的第 4 项，通过标准写在第 9 节，不要求对齐 Nature 原文的绝对准确率。

下一课（[第 16 课](16_when_weights_must_move.md)）把外挂记忆、编辑、改权重放进同一张通过矩阵。没有「还能不能继续学」这一条，那张矩阵会把「写进去当时对」误当成「以后还写得进」。

术语速查：

| 术语 | 一句解释 |
|---|---|
| 可塑性 | 网络接到新数据之后，还能用梯度把预测改对的能力。第 02 课讲过它和稳定性对着干 |
| 可塑性丢失 | 训练时间变长之后，同样的新任务变难学，哪怕你不考旧任务 |
| 死神经元 | ReLU 一类激活对当前样本几乎全是零，对应梯度也接近零，这个单元暂时退出学习 |
| 有效秩 | 一层表示里真正有贡献的方向有多少；低了说明很多单元在说同一句话 |
| shrink-and-perturb | 先把权重往小缩，再加一点随机噪声。Ash 和 Adams 2020 用来修热启动 |
| continual backprop | 普通反向传播之外，每步按效用把一小部分低使用率单元重新随机初始化 |
| 替换率 $\rho$ | 每步每层准备替换的单元比例，官方实现里常用 $10^{-4}$ 到 $10^{-6}$ |
| 成熟阈值 $m$ | 新初始化的单元在前 $m$ 步不许再被替换，避免刚出生就被判死刑 |
| 学习速度 | 本课操作定义：任务 $k$ 在固定样本预算下达到的在线准确率，或达到阈值所用的步数 |
| 复现 #4 | 长序列上标准 BP 后期变慢，CBP 或 shrink-and-perturb 缓解；对齐方向，不抄绝对数字 |

## 2. 问题

[第 01 课](01_catastrophic_forgetting.md) 的现场是：先学任务 A 再学任务 B，A 的准确率塌掉。评测协议（[第 03 课](03_cl_evaluation.md)）用遗忘、BWT 把这件事量出来。那一套默认网络还愿意学 B，只是学 B 的时候把 A 踩坏了。

可塑性丢失换了一个问题：B、C、D 一直在来，你甚至不回头考 A，网络对后面这些新任务的学习速度自己往下掉。预印本把 ImageNet 改成一串二分类：1000 类、每类 700 张、两两配对，最长做到 2000 个任务。他们报告的代表结果是：早期任务测试准确率约 89%，第 2000 个任务掉到约 77%，接近一个线性网络。掉的是「再给一对新类别，同样训 250 个 epoch，学出来的分类器变差」，旧类别当时根本没在考。

这个问题有三个容易混的地方，本课必须拆开。

第一，遗忘和学不动可以同时发生，也可以分开发生。强化学习里的 PPO 在平稳环境里分数掉下来，Nature 文明确写：曾经学会的策略忘了，是遗忘；再学一遍学不回去，是可塑性丢失。监督设定里，Online Permuted MNIST 几乎不要求你记住上一轮置换，因为它每轮像素排列都换掉，上一轮的决策边界本来就用不上。这个设定专门放大「还能不能学新的」，把「旧的还在不在」放到次要位置。你不能拿 Permuted MNIST 的曲线去宣称自己解决了 class-incremental CIFAR 的遗忘。

第二，五六个任务看不出这件事。经典持续学习论文常用 Split MNIST 的 5 对、Split CIFAR-10 的 5 个任务。短序列上，学习率、宽度、运气都还能撑。Dohare 等人把任务数拉到几百、两千，曲线才从「先升后平」变成「先升后掉到线性基线附近」。本课的动手因此必须远多于 5 个任务。课内机制实验用缩小网络和缩短的任务流，官方仓库用 Slow-Changing Regression 或缩短的 Permuted MNIST。800 个任务、每任务 6 万张图的原设定，不是这一课下午能跑完的东西。

第三，一次训完的常用零件，放到长序列上不一定帮忙。预印本在 Permuted MNIST 上试了 Adam、dropout、在线归一化、L2、shrink-and-perturb。Adam 和较大概率的 dropout 让后期掉得更狠；L2 能压住权重幅度，但死神经元和有效秩仍在恶化；shrink-and-perturb 在他们报告的超参下几乎把在线准确率的下滑压住。L2 有用，但只钉权重大小不够，还得持续往网络里注入多样性。continual backprop 把这件事做成「每步替换一小撮低效用单元」。

本课要回答的核心问题因此很具体：在一条很长的新任务流上，标准反向传播的后期学习速度会不会下降？死神经元是不是同步变多？L2、shrink-and-perturb、continual backprop 谁能把速度掉下来这件事缓解掉？

界限先写清。复现 #4 是方向性复现：你的缩小实验必须看到「后期变慢」和「CBP 或 shrink-and-perturb 更慢得少」。它不构成对 Nature 图 1、图 4 绝对数字的复现。官方仓库的 ImageNet 二分类长序列和 PPO 实验标成加分项。Lyle 等人关于损失曲率和休眠单元的工作、Zyphra 2026 年在 GPT 式 Transformer 上的观察，只在第 11 节按他们已经写明的设定引用，不把那些数字写进本课验收。

## 3. 准备

- 概念：[第 01 课](01_catastrophic_forgetting.md) 的 Permuted MNIST / Split MNIST 现场，[第 02 课](02_stability_plasticity.md) 的稳定性–可塑性平面。本课不需要 EWC 的 Fisher 公式，不需要 LoRA。激活函数会用到 ReLU：输入为负时输出零，对应位置的局部梯度也是零。
- 语言和环境：课内机制实验在 `learn-cl/experiments/` 下跑，Python 3.10+，只依赖课程实验包，不下载模型。官方仓库声明在 Ubuntu 20.04、Python 3.8 上测过，安装脚本按 3.8 写。你用 3.10 或 3.11 多半也能装，但出了依赖冲突先回到 3.8 的虚拟环境。
- 磁盘与网络：官方 Permuted MNIST 需要先下载 MNIST。Slow-Changing Regression 自己生成数据，不依赖图像数据集，单次运行官方 README 写的是约 15 个 CPU 分钟。课内机制实验应在几十秒内结束。
- 硬件：档 A（Mac / CPU，8GB 内存）能做完浏览器、课内实验、Slow-Changing Regression 的单次运行。不要在本课主线上开 ImageNet 或 CIFAR 残差网。
- 代码习惯：官方仓库用 JSON 配置生成一批 `temp_cfg/`，再对其中一份跑单次实验。读配置里的 `replacement_rate`、`maturity_threshold`、学习率，再决定改哪一个。
- 上一课产物：第 14 课的编辑实验这里用不上。本课是一条新的机制线，从随机初始化的小网络开始。

## 4. 学习目标

1. 用自己的话区分灾难性遗忘和可塑性丢失，并指出哪一种评测能把它们拆开。
2. 写出本课的学习速度操作定义，说明为什么「最终平均准确率」会把学不动藏起来。
3. 解释死神经元、权重幅度、有效秩三项和可塑性丢失的相关；同时写明 Lyle 等人为什么认为「只数死神经元」不够。
4. 默写 shrink-and-perturb 的两步，以及 continual backprop 的替换率、成熟阈值、效用三项。
5. 在官方仓库里指出 `cbp.py`、`cbp_linear.py`、Permuted MNIST 与 Slow-Changing Regression 的入口，并说清 README 和目录不一致时以目录为准。
6. 跑出复现 #4 的方向：后期任务学习速度下降，continual backprop 或 shrink-and-perturb 缓解。

## 5. 原理

五个机制，按同一节奏：为什么需要、怎么运转、数学定义、代码落点、怎么验收。

### 5.1 遗忘是旧的没了，可塑性丢失是新的学不进

把网络想成一块黑板。遗忘是：你写第 2 题的时候把第 1 题擦掉了。可塑性丢失是：黑板表面越写越滑，粉笔几乎留不下新痕迹。两件事都能让「后面的考试」难看，原因完全不同。类比失效处：真黑板的「滑」来自物理磨损，网络的「滑」来自参数分布离开了刚初始化时那个容易被梯度推动的区域。

形式化一点。记任务序列为 $\mathcal{T}_1,\mathcal{T}_2,\ldots,\mathcal{T}_K$。灾难性遗忘看的是：在 $\mathcal{T}_k$ 上训完之后，$\mathcal{T}_j$（$j<k$）的风险上升了多少。可塑性看的是：从当前参数 $\theta^{(k-1)}$ 出发，在 $\mathcal{T}_k$ 上用固定优化器和固定预算，能把 $\mathcal{T}_k$ 的损失压到多低。Lyle、Zheng、Nikishin 等人在 ICML 2023 的 *Understanding Plasticity in Neural Networks*（arXiv:2303.01486）里把可塑性写成：从 $\theta_t$ 出发，对一批新的探测损失做固定步数优化，看还能不能把损失压下去。他们的探测任务是随机目标回归，本课的探测任务是「下一个置换后的 MNIST」。量的都是「现在这个参数还肯不肯学」。

设任务 $k$ 的数据流是 $\{(x_{k,t}, y_{k,t})\}_{t=1}^{N_k}$，优化器为 $\mathcal{O}$，预算为 $B$ 步。学习速度用两种等价写法之一即可，报告里写明你用的是哪一种：

$$
S_k \;=\; \frac{1}{N_k}\sum_{t=1}^{N_k}\mathbf{1}\!\left[\arg\max_c f_{\theta_{k,t}}(x_{k,t})=y_{k,t}\right]
$$

这是 Dohare 预印本在 Online Permuted MNIST 上的在线准确率：每个样本先预测再更新，整轮 6 万张的命中率就是这个任务的成绩。缩小实验也可以用「达到准确率 $\tau$ 所需的步数」$T_k(\tau)$，学习速度取 $1/T_k(\tau)$。两种写法都对「最终只报平均准确率」构成否决：一个方法可以在前 3 个任务上极高、后 20 个任务上接近随机，平均分仍然好看。

第 01 课的 naive fine-tune 曲线是遗忘；本课的 $S_k$ 对 $k$ 的下滑是可塑性。同一张图上可以两条都画，但验收时分开判。

### 5.2 长序列 Permuted MNIST：专门放大「学不动」

Permuted MNIST 在第 01 课出现过：同一组手写数字，像素按一个固定置换重排，标签不变。一张「7」看起来像雪花，但仍叫 7。本课用它的在线长序列版，设定按预印本第 3 节：每次随机抽一个置换，把 6 万张训练图全部按这个置换重排，逐张送给网络，单遍、无小批量；置换切换时不给网络任何边界信号。他们最长画到 800 个任务。网络是三层全连接，每层 2000 个 ReLU，输出 10 类，交叉熵，Kaiming 初始化。

为什么这个设定适合量可塑性？因为相邻两个任务的输入分布几乎不共享空间结构，上一任务学到的像素特征帮不上什么忙。网络几乎每次都要从头找新的边缘。若后期任务的在线准确率掉下来，很难用「负迁移」或「旧头干扰」解释干净。预印本在 Continual ImageNet 里还重置了输出头的入边，专门排除 Chaudhry 等人指出的「旧头干扰」伪影。Permuted MNIST 连任务边界都不告诉网络，更没有「每来一个任务加一个头」。

学习速度曲线的预期形状，预印本图 2 写得很清楚：前几个任务 $S_k$ 往往先升（宽度还没用完、特征还能共用一点笔划统计），随后持续下降。他们扫了步长、宽度（每层 100 / 1000 / 2000 / 10000）、置换切换频率（每 1 万 / 6 万 / 10 万 / 100 万样本换一次）。掉的幅度随宽度变大而变缓，但最大的网也没有完全免掉。激活函数换成 sigmoid、tanh、ELU、leaky-ReLU、Swish，他们在更便宜的 Slow-Changing Regression 上同样看到后期变差。

课内机制实验不会重跑 800×60000 次前向。它保留三件事：任务数远大于 5、每个任务固定预算、记录 $S_k$ 或等价的早期损失下降。官方仓库的完整 Permuted MNIST 是对照，不是当天下午的必做。

### 5.3 三件伴随物：死单元、权重大了、秩低了

可塑性丢失发生时，网络里同时发生三件可以数的事。预印本第 4 节把它们当作相关物，不是充分因果。Lyle 等人后来证明：换环境和架构之后，这三件里任何一件都可以跟可塑性正相关或负相关。本课先把预印本的三项测出来，再在第 11 节加上 Lyle 的限制。

死单元。对 ReLU，一个隐藏单元在当前任务的抽样输入上输出全是 0，就叫死。它对下游的贡献是 0，反传回来的局部梯度也是 0，这个单元暂时退出学习。预印本的数法：每个新任务开始时，抽 2000 张图，数输出恒为 0 的单元比例。步长 0.01、800 个任务之后，他们看到死单元可以到约 25%。sigmoid / tanh 对应的是饱和：输出贴在 0 或 1（或 $\pm 1$）附近，局部导数接近 0。

权重幅度。把全部权重的绝对值加起来再除以参数个数。预印本图 3 中间一幅：标准 BP 的平均幅度随任务单调涨。大幅度让有效步长变野，也让 ReLU 更容易整段掉到负半轴。L2 正则直接压这件事。

有效秩。一层在一批输入上的激活矩阵 $\Phi\in\mathbb{R}^{n\times m}$，奇异值为 $\sigma_1,\ldots,\sigma_q$，令 $p_k=\sigma_k/\|\boldsymbol{\sigma}\|_1$，则

$$
\mathrm{erank}(\Phi)=\exp\!\left(-\sum_{k=1}^{q}p_k\log p_k\right)
$$

它介于 1 和矩阵秩之间。有效秩低，说明少数方向撑起了这一层，多数单元互相重复。预印本观察到：BP 的有效秩随任务下降。低秩解可能对当前任务够用，却是下一任务的糟糕初始化。

这三项在 `lop/permuted_mnist/plots/` 里可以用不同 `--metric` 画出来。课内实验至少要钉死「后期死单元比例高于前期」这一条；幅度和有效秩能算则算，算不出就在报告里写「本实现未统计」。

Lyle、Zheng、Nikishin、Pires、Pascanu、Dabney（arXiv:2303.01486）给了一句必须写进报告的限制：可塑性丢失可以在几乎没有饱和单元时发生，真正麻烦的往往是新任务损失曲面变尖、小批量之间互相干扰。所以本课把死单元当可见症状，不当唯一病因。你重置死单元有效，不证明病因就是死单元。

### 5.4 L2 只按住幅度，shrink-and-perturb 还往里加噪声

标准训练损失加上权重惩罚：

$$
\mathcal{L}_{\mathrm{L2}}(\theta)=\mathcal{L}_{\mathrm{task}}(\theta)+\frac{\lambda}{2}\|\theta\|_2^2
$$

对应的梯度多一项 $\lambda\theta$，每步都把参数往原点拉一点。预印本确认：L2 能阻止平均幅度继续涨，后期在线准确率高于裸 BP。它不能阻止死单元变多，也不能阻止有效秩下降。只按住长度，方向照样塌缩。

shrink-and-perturb 来自 Ash 和 Adams，*On Warm-Starting Neural Network Training*（NeurIPS 2020，arXiv:1910.08475）。原问题是：数据一块块来，用上一块训完的权重接着训下一块，训练损失能降，泛化往往不如从随机初始化重训。他们的修法是：先把当前权重大约按比例缩小，再加高斯噪声。预印本用的在线版和 L2 共用一项收缩，并在每次更新时加噪声：

$$
\theta \leftarrow \theta - \alpha\nabla\mathcal{L}_{\mathrm{task}}(\theta) - \alpha\lambda\theta + \varepsilon,\qquad \varepsilon\sim\mathcal{N}(0,\sigma^2 I)
$$

收缩压幅度，扰动把一部分死掉或共线的方向重新激活。预印本图 4：在他们选的超参下，shrink-and-perturb 在 800 个任务上几乎没有在线准确率下滑，死单元增长也慢于 BP。它对 $\sigma$ 很敏：噪声太大，可塑性丢得比 BP 还快；太小，等于没加。L2 单独用也只在很窄的 $\lambda$ 区间里帮忙。

官方对照配置在 `lop/permuted_mnist/cfg/l2.json` 和 `cfg/snp.json`。课内实验用缩小版的同一对操作，不复制他们的 $\lambda$ 和 $\sigma$。你要报告的是四条线的相对形状，不是某一组超参的绝对准确率。

Adam 和 dropout 在同一张图上是反面教材。Adam 用梯度二阶矩做自适应步长，任务一换，一阶矩和二阶矩短暂错位，更新会爆；预印本里它的有效秩掉得最戏剧。dropout 概率加大，后期掉得更快。一次训完的默认工具，搬到长序列上要重新量。

### 5.5 Continual backprop：每步把低效用单元送回初始化

普通反向传播做两件事：开始时用小随机数初始化，之后每步只做梯度下降。continual backprop 把初始化延长到每一个时间步：每步在每一层按替换率 $\rho$ 找出效用最低、且已经过了成熟期的单元，把它们的入边重新随机初始化，出边置零。出边置零是为了不立刻改掉已经学会的函数；成熟阈值 $m$ 是为了避免「出边刚被置零、效用变成 0、下一步又被替换」的自杀循环。

效用分两块。贡献效用看这个单元对下游有没有用：激活的绝对值乘出边绝对值之和。因为下游可以慢慢把平均值吸进偏置，他们用去均值后的激活：

$$
f_{l,i,t}=\eta f_{l,i,t-1}+(1-\eta)h_{l,i,t}
$$

$$
\hat f_{l,i,t}=\frac{f_{l,i,t-1}}{1-\eta^{a_{l,i,t}}}
$$

$$
z_{l,i,t}=\eta z_{l,i,t-1}+(1-\eta)\,|h_{l,i,t}-\hat f_{l,i,t}|\,\sum_{k=1}^{n_{l+1}}|w_{l,i,k,t}|
$$

其中 $h_{l,i,t}$ 是第 $l$ 层第 $i$ 个单元在时刻 $t$ 的输出，$a_{l,i,t}$ 是这个单元的年龄，$\eta$ 是滑动平均衰减，官方默认 0.99。适应效用是入边幅度之和的倒数：权小，相对变化大，更容易改自己代表的函数。总效用是两者之商的滑动平均（预印本公式 (6)–(8)）：

$$
y_{l,i,t}=\frac{|h_{l,i,t}-\hat f_{l,i,t}|\,\sum_{k}|w_{l,i,k,t}|}{\sum_{j}|w_{l-1,j,i,t}|}
$$

$$
u_{l,i,t}=\eta u_{l,i,t-1}+(1-\eta)y_{l,i,t},\qquad
\hat u_{l,i,t}=\frac{u_{l,i,t-1}}{1-\eta^{a_{l,i,t}}}
$$

替换时，他们还把该单元的平均贡献 $\hat f\cdot w$ 加到下游偏置上，减轻突然拆掉一个单元对已有函数的冲击。

官方实现有两代。论文图表用的是 `lop/algos/cbp.py`（全连接）、`convCBP.py`（卷积）、`res_gnt.py`（残差）。后来加了层式接口 `cbp_linear.py` 和 `cbp_conv.py`，用法接近 Dropout：在网络里插入一层，让激活从它经过。`lop/algos/README.md` 写明：层式接口方便，但「没有像论文用的那份测得那么细，可能有小 bug」。本课对照论文公式时以 `cbp.py` 为准；自己改网络时可以用 `CBPLinear`，并在报告里标明用的是哪一份。

`CBPLinear` 的关键参数，按仓库 README 原文：

| 参数 | 含义 | 仓库给出的常用范围 |
|---|---|---|
| `in_layer` / `out_layer` | 这个隐藏层的入边层、出边层 | 必填 |
| `replacement_rate` | 每步替换的单元数比例 $\rho$ | $10^{-4}$ 到 $10^{-6}$ |
| `maturity_threshold` | 保护步数 $m$ | 100 到 10000 |
| `decay_rate` | 效用滑动平均 $\eta$ | 默认 0.99；0 会更快但估计更糙 |
| `util_type` | 效用种类 | 默认 `contribution`，对 ReLU 类激活可用 |
| `init` / `act_type` | 重初始化分布和激活名 | 默认 kaiming / relu |

Nature 摘要把结论说到了算法原则：只靠梯度下降不够，持续的深度学习需要一个随机的、非梯度的成分来维持多样性。continual backprop 就是把这个随机成分做成「按效用重置」。它不保存旧样本，也不钉旧权重，所以它不是 EWC，也不是回放。它保的是学习能力，不是旧任务成绩。若你同时要旧任务，还得另加第 05–08 课的零件。

## 6. 源码导读

锚定仓库是 [shibhansh/loss-of-plasticity](https://github.com/shibhansh/loss-of-plasticity)。写课前打开的是该仓库 `main` 的 README、`lop/` 目录和 `lop/algos/README.md`、`lop/permuted_mnist/README.md`、`lop/slowly_changing_regression/README.md`。根 README 写明代码对应 Nature 论文 *Loss of Plasticity in Deep Continual Learning*。安装段落要求 Python 3.8 虚拟环境。

读代码按一条样本的路径走，不要按字母表。

| 路径 | 带着什么问题读 |
|---|---|
| `lop/algos/bp.py` | 普通反向传播在这个仓库里长什么样，作为 CBP 的对照 |
| `lop/algos/cbp.py` | 论文全连接结果用的实现：效用、替换、成熟保护在哪几段 |
| `lop/algos/cbp_linear.py` | 层式接口：`replacement_rate`、`maturity_threshold` 的默认值和调用顺序 |
| `lop/algos/README.md` | 哪份文件对应论文哪类网络；层式接口的免责声明 |
| `lop/nets/conv_net2.py` | README 点名的 `CBPLinear` 使用例子 |
| `lop/permuted_mnist/load_mnist.py` | MNIST 怎么落到 `data/` |
| `lop/permuted_mnist/multi_param_expr.py` | 如何从一份 JSON 展开成若干 `temp_cfg/` |
| `lop/permuted_mnist/online_expr.py` | 单次在线实验入口（见下方 README 不一致说明） |
| `lop/permuted_mnist/cfg/bp/`、`cfg/l2.json`、`cfg/snp.json`、`cfg/cbp.json`、`cfg/adam.json` | 四条对照线加 Adam 反例的配置 |
| `lop/permuted_mnist/plots/` | `bp_metrics.py` 的 `--metric` 能画准确率还是死单元 |
| `lop/slowly_changing_regression/slowly_changing_regression.py` | 数据怎么生成 |
| `lop/slowly_changing_regression/expr.py` | 单次 15 分钟 CPU 实验的真实入口 |
| `lop/imagenet/`、`lop/incremental_cifar/`、`lop/rl/` | 加分项：ImageNet 二分类、class-incremental CIFAR、PPO |

有一处必须写进实验记录：`lop/permuted_mnist/README.md` 写的单次命令是 `python3.8 expr.py -c temp_cfg/0.json`，但写课前看到的该目录文件列表是 `load_mnist.py`、`multi_param_expr.py`、`online_expr.py`，没有 `expr.py`。同仓库的 `lop/slowly_changing_regression/` 里确实有 `expr.py`。Permuted MNIST 的命令以目录里真实存在的脚本为准，优先试 `online_expr.py`；若作者之后补回 `expr.py`，再改回 README。不要假装 README 和目录已经对齐。

配置里和论文公式对应的名字：

- 替换率：`replacement_rate`
- 成熟阈值：`maturity_threshold`
- 效用衰减：`decay_rate`
- L2 系数和噪声方差：分别在 `l2.json`、`snp.json`

`CBPLinear` 要求激活经过这一层。漏掉这件事，替换逻辑不会跑，曲线会退化成普通 BP。验证方法：在 forward 里对 CBP 层做一次计数，确认每步调用次数等于层数。

课内机制实验的落点是 `experiments/src/learn_cl_experiments/lessons/lesson_15.py`，由 `python3 run.py run 15` 调用。它不 import 官方仓库，用缩小的全连接网和缩短的任务流把第 5 节的断言钉死。官方仓库是对照，不是课内实验的运行时依赖。

## 7. 实验

三层都做。浏览器先建立「死单元随任务涨、打开 CBP 会重置」的直觉；课内 CPU 实验给出可断言的方向；官方仓库证明你读的是同一算法，而不是课内简化版的自说自话。

### Step 0: 预测死神经元

打开本课网页实验「死神经元」（`lab-15-dead-neurons`）。画布是各层的柱状图：横轴任务序号，柱高是该层饱和或恒零单元的比例。先不要按运行。写下三个预测：

1. 只用 SGD、任务数拉到远大于 5 之后，后期柱子会不会明显高于前期。
2. 打开 L2，柱子会不会明显回落，还是只变一点。
3. 打开 continual backprop 之后，哪些柱子会被打矮，是均匀打矮还是只打最死的那一截。

预测提交之前，运行按钮应无效。改滑块会作废上一次运行。过关条件由页面判定，课文侧的合格预测是：SGD 后期死单元上升；L2 不一定救死单元；CBP 会重置一部分低使用率单元。和预印本图 3、图 4 同方向即可。

### Step 1: 跑课内机制实验

```bash
cd experiments
```

```bash
python3 run.py run 15
```

预期：标准输出打印若干 `checks`，全部为真时状态是 `PASS`，结果写到 `artifacts/lesson15/result.json`。`python3 run.py run 15` 现在应当全绿。`checks` 五条：`sgd_late_gain_drops`、`sgd_dead_ratio_rises`、`sgd_late_speed_slower`、`cbp_late_gain_beats_sgd`、`cbp_late_dead_below_sgd`。

本机一次运行：标准 SGD 后期 tanh 增益从 0.196 掉到 0.065，死神经元比例从 0 升到 0.20；按饱和度重初始化之后，后期死神经元 0.033，后期增益高于同期 SGD。换机器会变，方向不应变。失败阈值写在 `summary`：后期增益不低于前期的 75%，死神经元上升不足 0.15，或后期准确率提升不低于前期 0.04。

对照线是裸 SGD 对上 continual backprop（课内按饱和度重初始化 3 个隐单元）。`checks` 必须能失败：改坏替换逻辑时，后期速度差应消失，对应布尔应变假。不要改断言阈值去凑过。

这一层是复现 #4 的课内证据。它用缩小网络，不下载 MNIST 官方完整流程，不能拿来填 Nature 表格。不是 Nature 文的 ImageNet 800 任务。

Agent 连续多天往同一张 $W$ 里写，也会把列占满，后面的批次比空白矩阵难学：

```bash
python3 run.py extra run plastic
```

GPU 对照是 `python3 run.py gpu print plasticity`。

### Step 2: 装官方仓库（独立环境）

```bash
git clone https://github.com/shibhansh/loss-of-plasticity.git
```

根 README 的安装顺序如下，每条单独跑。他们写的是 Python 3.8；若系统没有 3.8，用你已经验证能装上 `requirements.txt` 的版本，并在报告里记下。

```bash
mkdir ~/envs
```

```bash
virtualenv --python=/usr/bin/python3.8 ~/envs/lop
```

```bash
source ~/envs/lop/bin/activate
```

```bash
cd loss-of-plasticity
```

```bash
pip3 install -r requirements.txt
```

```bash
pip3 install -e .
```

装完在报告里记：Python 版本、`git rev-parse HEAD`、是否改过依赖。不要把 `main` 当版本号。

### Step 3: Slow-Changing Regression（官方、CPU、约 15 分钟）

这是官方 README 标明「单次约 15 个 CPU 分钟」的入口，适合档 A。目录里的 `expr.py` 和 README 一致。

```bash
cd lop/slowly_changing_regression
```

```bash
mkdir env_temp_cfg temp_cfg
```

```bash
python3.8 multi_param_expr.py -c cfg/prob.json
```

上面会按 `cfg/prob.json` 展开很多份环境配置。先只生成第 0 号数据：

```bash
python3.8 slowly_changing_regression.py -c env_temp_cfg/0.json
```

再展开一次学习配置。裸 BP：

```bash
python3.8 multi_param_expr.py -c cfg/sgd/bp/relu.json
```

continual backprop：

```bash
python3.8 multi_param_expr.py -c cfg/sgd/cbp/relu.json
```

单次学习（README 原文，`expr.py` 在这个目录里存在）：

```bash
python3.8 expr.py -c temp_cfg/0.json
```

官方建议至少 30 个 seed 才画他们那种平均图。本课验收只要求：同一份数据、BP 与 CBP 各至少 1 个 seed，后期误差或后期滑动平均误差 CBP 不差于 BP。完整 100 seed 标加分。画图：

```bash
cd plots
```

```bash
python3.8 online_performance.py -c ../cfg/sgd/bp/relu.json
```

把 BP 和 CBP 的图并排贴进复现报告，写明 bin 大小（README 说误差按 20000 步分箱）。

### Step 4: Permuted MNIST 缩短版（复现 #4 的主对照）

完整 800 任务 × 6 万样本超出本课预算。缩短版仍然必须「远多于 5 个任务」，建议至少 20 个任务、每任务样本数按配置能在数小时 CPU 内结束。先下载数据：

```bash
cd lop/permuted_mnist
```

```bash
mkdir data
```

```bash
python3.8 load_mnist.py
```

展开 BP 配置：

```bash
python3.8 multi_param_expr.py -c cfg/bp/std_net.json
```

单次运行以目录里的脚本为准。写课前列表中是 `online_expr.py`，先试：

```bash
python3.8 online_expr.py -c temp_cfg/0.json
```

若报找不到模块或参数，打开 `online_expr.py` 和 `multi_param_expr.py` 的参数解析，按文件头注释改。不要盲目抄 README 里的 `expr.py`。

L2、shrink-and-perturb、CBP 各跑一份缩短配置，配置模板分别是 `cfg/l2.json`、`cfg/snp.json`、`cfg/cbp.json`。能改任务数的字段在 JSON 里搜 `num_tasks`、`num_examples` 或同类名字；没有就在报告里写「本版配置写死了任务数，缩短版改为修改某某行」。

画图目录的 README 示例：

```bash
cd plots
```

```bash
python3.8 bp_metrics.py --cfg_file ../cfg/bp/std_net.json --metric accuracy
```

把 `--metric` 换成仓库支持的内部量（README 写了平均权重幅度一类）。能画死单元就画；画不出就用课内实验的死单元图顶上，并注明官方脚本未出该指标。

复现 #4 在这一层的通过标准：缩短长序列上，BP 的后期任务学习速度（或在线准确率）低于前期；CBP 或 shrink-and-perturb 的后期速度高于同期 BP。方向对即可。

### Step 5: 写复现报告

报告至少五段：设定（任务数、网络、种子、提交哈希）、四条曲线、死单元、和论文同方向或不同向的判断、失败时改过什么。不同向时写反例，不要改数字去贴论文。ImageNet 与 PPO 目录只要求你读一遍 README，不跑。

## 8. 配置与预算

| 档 | 做什么 | 时间 | 内存 |
|---|---|---|---|
| 浏览器 | 死神经元预测与运行 | 15 分钟 | 浏览器 |
| 课内 CPU | `python3 run.py run 15` | 应少于 30 秒 | 小于 1GB |
| 官方 Slow-Changing Regression | BP 与 CBP 各 1 seed | 各约 15 分钟 CPU | 8GB 足够 |
| 官方 Permuted MNIST 缩短 | 四方法各 1 seed，任务数 $\ge 20$ | 数小时 CPU | 8GB 足够，需下载 MNIST |
| 加分：官方 800 任务 | 单方法多 seed | 按仓库硬件自行估 | 不纳入本课必做 |
| 加分：`lop/imagenet`、`lop/rl` | 读 README 或抽 1 个短配置 | 不估 | 需要 GPU 时标加分 |

学习率、$\lambda$、$\sigma$、$\rho$、$m$ 全部写入报告。官方 README 给的 CBP 常用区间是 $\rho\in[10^{-6},10^{-4}]$，$m\in[100,10000]$，$\eta=0.99$。课内实验会另选一套能在几十秒内分出方向的值，两套数字不要混着比。

数据量：课内实验用合成置换或缩小版数字图，不下载外部大包。官方 Permuted MNIST 用 `load_mnist.py` 拉 MNIST。Slow-Changing Regression 在本地生成。

随机种子：课内实验固定。官方实验至少记 1 个 seed；声称「和论文一样稳」时至少 5 个。预印本 Permuted MNIST 主图是 30 个 run 的均值和标准误，你没有 30 个 run 就不要画误差带冒充。

## 9. 验收

复现 #4 通过，当且仅当下面全部成立。

- 浏览器实验：先预测再运行，页面判定预测合格。
- 课内实验：`python3 run.py run 15` 的 `checks` 全为真。本机一次运行 SGD 后期增益 0.196→0.065、死神经元 0→0.20，CBP 后期死神经元 0.033。换机器会变，方向不应变。不是 Nature 文的 ImageNet 800 任务。
- 方向性对照：同一设定下，SGD 的 $S_k$（或等价量）在后期低于前期；CBP 或 shrink-and-perturb 的后期值高于同期 SGD。
- 死神经元：SGD 的后期比例高于前期。CBP 打开后，该比例低于同期 SGD，或报告里写明「本实现只重置不统计」。
- 书面区分：报告里有一段明确写「本实验不测量遗忘」或「本实验同时记录了旧任务，但复现 #4 只看新任务速度」。
- 仓库对照：至少跑通 Step 3 或 Step 4 之一，命令与当前仓库文件一致，提交哈希写在报告里。

未通过的典型写法：只交一张最终平均准确率；任务数只有 5；把 Adam 掉下去当成「方法有效」；用课内缩小数字填写 Nature 表格。

## 10. 排错

| 症状 | 原因 | 验证 | 修法 |
|---|---|---|---|
| `sgd_late_gain_drops` 或 `sgd_dead_ratio_rises` 为假 | 任务太少、网太宽，或死单元统计时机不对 | 看 `sgd_early_gain` / `sgd_late_gain`、`sgd_dead_curve` | 后期增益应低于前期的 75%，死神经元应上升超过 0.15 |
| 后期速度不掉 | 任务太少、网太宽、预算太大 | 打印每个 $k$ 的 $S_k$ | 加任务、减宽度、把每任务步数固定住 |
| 四条线完全重合 | CBP 层没接到 forward，或 $\rho=0$ | 在替换函数里设计数器 | 对照 `conv_net2.py`，确认激活经过 CBP 层 |
| CBP 比 SGD 更差 | $\rho$ 太大或 $m$ 太小，刚学会的单元被拆掉 | 扫 $\rho$ 三个数量级 | 先回到 README 的 $10^{-5}$ 附近 |
| shrink-and-perturb 一塌糊涂 | $\sigma$ 过大 | 看权重幅度是否爆炸 | 先关掉噪声只留收缩，再把 $\sigma$ 往小调 |
| L2 后期更差 | $\lambda$ 过大，新任务学不动 | 新任务损失降不下去 | 减小 $\lambda$，L2 本来就不能单独当完整解 |
| `expr.py` 找不到 | Permuted MNIST 的 README 过时 | `ls lop/permuted_mnist` | 改用 `online_expr.py` 或 Slow-Changing Regression 的 `expr.py` |
| `load_mnist.py` 失败 | 无网络或镜像变了 | 看报错是超时还是 404 | 手动把 MNIST 放到 README 指定的 `data/` |
| Python 3.11 装不上旧依赖 | 仓库按 3.8 写 | `pip` 报编译错误 | 新建 3.8 环境 |
| 死单元全是 0 | 统计写在训练前，或用了 leaky-ReLU | 打印一层激活的负值比例 | 在任务切换处、对当前任务样本统计 |
| 有效秩算不动 | 激活矩阵太大 | 形状是否是「样本 × 单元」 | 随机抽 512 个样本再算 SVD |
| ImageNet 脚本要 GPU | 那是加分项 | README 的网络表 | 本课主线不要跑 |

## 11. 前沿与改造

公开方案。Dohare 等人 2024 的原则是：梯度下降之外要持续注入随机多样性。Lyle 等人 2023（arXiv:2303.01486）在深度强化学习里把可塑性定义成「对随机新目标还能不能快速拟合」，并给出一条和本课不完全相同的药方：层归一化、更平滑的损失曲面，往往比单纯重置最后一层更稳。他们还用一套可证伪的相关分析表明：权重范数、特征秩、死单元在换环境和奖励结构之后，与可塑性的相关符号会翻转。Lyle 等人 2024 的后续 *Disentangling the Causes of Plasticity Loss in Neural Networks*（arXiv:2402.18762）进一步说：单一机制的干预不够，层归一化加权重衰减在一批非平稳任务上更稳。这些论文不替代 Dohare 的长序列监督实验，它们限制你怎么解释死单元。第 12 节补了 2024–2026 的干预：谱正则、按权重重置、改激活形状、注入线性、剪幅度，以及用回放加 Transformer 顶住可塑性丢失。

Zyphra 2026 年 6 月 24 日的公开说明 *Plasticity Loss in Continual Learning*，以及对应预印本 Hernandez-Garcia、Figliolia、Millidge *Can Scale Save Us From Plasticity Loss in Large Language Models?*（arXiv:2606.24752），只引用他们已经写明的设定。设定是：GPT 式仅解码 Transformer，非嵌入参数从 5M 到 314M；多语言持续学习问题按英、中文书面语、法、日、西、德、葡、俄循环，每种语言 50 亿 token；每个循环结束用从未参与训练的越南语 50 亿 token 做探测，探测在检查点副本上进行，更新丢弃。他们报告探测任务验证损失曲线下面积相对第一轮的变化，后期 AUC 上升视为学得更慢。他们拟合的发病时间（以任务实例数计）为 $T=1.3\times 10^{-5}\,P^{0.8269}$，$P$ 是非嵌入参数量，并写明这是次线性。他们还把八种语言混成一份平稳语料，在 5M / 12M / 27M 上看到类似的后期 AUC 上升。本课不重跑这些实验，不把 $T$ 的系数写进验收，不外推到未写明的模型规模。

差距。课内网络是几百到几千个隐藏单元的 MLP，官方主线也是 MNIST 和 Slow-Changing Regression。Transformer、十亿参数、万亿 token 上的可塑性，本课只能读不能练。机制上能带走的是：学习速度必须单独画；重置低效用单元和 shrink-and-perturb 是两类可实现的多样性注入；死单元是症状不是完整病因。

动手改造（2–4 个，均可在课内实验上做）：

1. 关掉成熟保护。位置：CBP 替换逻辑里的年龄判断，对应官方 `maturity_threshold`。预算：课内实验再跑 1 次。预期：新单元刚重置就被再重置，后期速度回不到 CBP 原曲线。失败标准：关掉 $m$ 之后曲线几乎不变。
2. 随机效用对照。位置：把效用换成 $U[0,1]$ 随机数。预印本附录 D 在 Slow-Changing Regression 上做过。预算：1 次课内实验或 1 次官方 15 分钟。预期：随机替换弱于贡献效用。失败标准：随机替换全面超过贡献效用且死单元更少，此时要检查原效用是不是算反了。
3. 只收缩、不扰动。位置：shrink-and-perturb 的噪声项。预算：1 次。预期：接近 L2，死单元仍涨。失败标准：去掉噪声之后仍与完整 S&P 重合。
4. 加一层 LayerNorm 再跑裸 SGD。位置：每个隐藏层激活之后。预算：1 次课内实验。预期：按 Lyle 的方向，后期速度掉得更少，但不必降到 CBP 的程度。失败标准：把 LayerNorm 的收益写成「复现了 Dohare」。Dohare 的主结论是随机再初始化，不是归一化。

顺手复现映射：

| 论文结论 | 缩小版对应 | 预期 |
|---|---|---|
| 长序列上 BP 后期变慢 | 课内 $S_k$–$k$ 曲线 | 能看到同方向 |
| CBP 缓解可塑性丢失 | 课内 CBP 对照 | 能看到同方向 |
| S&P 几乎维持在线准确率 | 缩短 Permuted MNIST 的 `snp.json` | 方向可复现，幅度不必对齐 |
| Adam 掉得更狠 | 若实现 Adam 对照 | 可复现方向；不是本课必做 |
| 死单元升到约 25% | 课内死单元比例 | 只比升降，不对齐 25% |
| Zyphra 的次线性发病律 | 无 | 本课不能复现 |

## 12. 论文与延伸

1. Dohare, Hernandez-Garcia, Lan, Rahman, Mahmood, Sutton, 2024, *Loss of plasticity in deep continual learning*, *Nature* 632:768–774, doi:[10.1038/s41586-024-07711-7](https://doi.org/10.1038/s41586-024-07711-7)；预印本 *Maintaining Plasticity in Deep Continual Learning*, [arXiv:2306.13812](https://arxiv.org/abs/2306.13812)。
贡献：标准反向传播在持续设定里会把可塑性丢掉，直到学得不比浅网好；持续把低使用率单元随机再初始化，才能无限期维持。
机制：评测改成每个新任务固定预算下的新任务成绩，旧任务可以不考。ImageNet 被拆成一对对二分类。算法改的是反向传播本身：每步按效用替换一小撮单元，出边置零，成熟期内不许再被替换。Nature 摘要把原则写成：只靠梯度下降不够，还要有随机的非梯度成分。预印本摘要写 Continual ImageNet 早期约 89%、第 2000 个任务约 77%。
和本课：`sgd_late_gain_drops`、`sgd_late_speed_slower`、`sgd_dead_ratio_rises` 看见后期变慢和死单元上涨；`cbp_late_gain_beats_sgd`、`cbp_late_dead_below_sgd` 看见按饱和度重置 3 个隐单元的方向。官方 `cbp.py` 的效用公式本课 CPU 没有实现。掉到线性基线这一句，课内没有线性对照，答不了。
阅读问题：你的缩短序列上，SGD 后期增益掉了多少？有没有资格写成「掉到线性基线附近」？

2. Elsayed and Mahmood, 2024, *Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning*, [arXiv:2404.00781](https://arxiv.org/abs/2404.00781)（ICLR 2024）。
贡献：用 UPGD 同时处理可塑性丢失和遗忘：有用的单元少动，没用的单元多扰动。
机制：改的是更新规则，不是损失里再加一项正则。效用高的单元步长更小，用来护旧知识；效用低的单元步长更大并加扰动，用来恢复可塑性。设定是流式学习，有几百次非平稳、任务边界未知。摘要写许多现有方法至少踩中一端，准确率随任务往下掉；UPGD 在他们的问题上继续涨，并在 PPO 上避免 Adam 那种学会之后再掉。
和本课：课内 CBP 只重置最饱和的 3 个单元，对应「救没用的」这一半。UPGD 的「有用单元少动」本课没有实现，也没有旧任务考试，看不见遗忘那一端。PPO 曲线本课实验答不了。
阅读问题：若你把重置改成「所有单元都加同一强度噪声」，`cbp_late_gain_beats_sgd` 还会不会为真？用一次课内实验回答；答不了就写你没改替换逻辑。

3. Wang, Chandra and Zhang, 2025, *Experience Replay Addresses Loss of Plasticity in Continual Learning*, [arXiv:2503.20018](https://arxiv.org/abs/2503.20018)。
贡献：假设回放本身就能消除可塑性丢失；在回归、分类和策略评估上，给回放加 Transformer 处理后，可塑性丢失消失。
机制：改的是存储和读法：把经验放进回放，再用 Transformer 处理回放里的数据。摘要写他们不改反向传播、不改激活、不加正则。猜想是上下文学习在起作用。
和本课：课内实验没有回放缓冲，也没有 Transformer。你看见的是裸 SGD 后期增益掉、死单元涨。这篇主张「不必改激活、不必重置」，本课实验答不了，因为没有回放臂。
阅读问题：若只在课内 SGD 流上加一个容量为 16 的旧样本袋、仍用同一套 MLP，后期增益还会不会掉？本课默认实验没有这条臂，必须另写对照才能答。

4. Prakash, He, Guo, Tiwari, Tao, Serapio, Greenwald and Konidaris, 2025, *Spectral Collapse Drives Loss of Plasticity in Deep Continual Learning*, [arXiv:2509.22335](https://arxiv.org/abs/2509.22335)。
贡献：新任务开始时 Hessian 谱塌缩，有曲率的方向没了，梯度下降跟着失效。
机制：在线性化 ReLU 网上给出可训练的 $\varepsilon$-秩条件，并证明损失加权 Gram 矩阵与广义 Gauss-Newton 谱等价。药方对准谱塌缩：保持特征有效秩，再加 L2。监督和强化学习的持续任务上，两味药合用能保住可塑性。
和本课：课内 `sgd_late_gain_drops` 看见 tanh 增益塌，方向接近「有效学习方向变少」，但没有算 Hessian，也没有 $\varepsilon$-秩。L2 在官方仓库有 `l2.json`，课内 CPU 没跑。谱条件本课实验答不了。
阅读问题：你的后期死单元上涨和增益下降，能不能代替「Hessian 谱塌缩」这句话？若不能，差在哪一条测量？

5. Lillo and Cheney, 2025, *Activation Function Design Sustains Plasticity in Continual Learning*, [arXiv:2509.22562](https://arxiv.org/abs/2509.22562)。
贡献：激活函数的负半轴形状和饱和行为，是跨架构缓解可塑性丢失的杠杆。
机制：改的是非线性本身，不扩容量、不按任务调超参。他们给出 Smooth-Leaky 和 Randomized Smooth-Leaky 两个可替换激活，在类增量监督和非平稳 MuJoCo 上测。诊断把激活形状和「分布一变还能不能适应」连起来。
和本课：课内网用 tanh，死单元按 $| \tanh | > 0.97$ 计。这篇要你换负半轴，不重置单元。默认实验没有换激活，`sgd_dead_ratio_rises` 只对 tanh 成立。
阅读问题：若把 tanh 换成 leaky 一类（负半轴不掐死），`sgd_dead_ratio_rises` 还会不会为真？本课默认实验答不了，除非你改激活再跑一次。

6. Hernandez-Garcia, Figliolia and Millidge, 2026, *Can Scale Save Us From Plasticity Loss in Large Language Models?*，[arXiv:2606.24752](https://arxiv.org/abs/2606.24752)；说明页 [Zyphra, 2026-06-24](https://www.zyphra.com/our-work/plasticity-loss-in-continual-learning)。
贡献：5M–314M 的 GPT 式模型在他们写明的多语言循环和越南语探测上出现可塑性丢失，发病时间对参数量次线性，平稳混合数据里也出现。
机制：改的是评测，不是优化器。每个语言循环结束，在检查点副本上用从未训练过的越南语做探测，更新丢弃。后期探测验证损失 AUC 相对第一轮上升，视为学得更慢。他们还把八种语言混成一份平稳语料，在 5M / 12M / 27M 上看到类似的后期 AUC 上升。
和本课：第 11 节只读。本课 $S_k$ 是同一条权重点上继续训，他们的探测更新是丢弃的。发病律 $T=1.3\times 10^{-5}\,P^{0.8269}$ 本课实验答不了。
阅读问题：他们的探测更新是丢弃的，本课的 $S_k$ 是接着训，这两件事能直接比吗？

7. Hernandez-Garcia, Dohare, Luo and Sutton, 2025, *Reinitializing weights vs units for maintaining plasticity in neural networks*, [arXiv:2508.00212](https://arxiv.org/abs/2508.00212)。
贡献：提出按效用重置权重（selective weight reinitialization），并指出小网和带 LayerNorm 时，重置权重比重置单元更稳。
机制：每隔若干步，按 $|w\cdot g_w|$ 给权重打效用，用阈值或比例剪掉最低的一截，再从初始化分布重采样。对照是 CBP 和 ReDo（二者重置单元）。Permuted MNIST 上，每层 10 个单元或加了 LayerNorm 时，单元重置掉得更明显；权重重置四条设定都能维持。
和本课：课内网是 10 个 tanh 隐单元，重置的是最饱和的 3 个单元，正好落在他们说的「小网、重置单元」设定。`cbp_late_gain_beats_sgd` 为真，只说明单元重置在这个玩具网上够用。换权重级重置、加 LayerNorm，本课默认实验都没做。
阅读问题：按这篇，10 个隐单元该不该改成重置权重？你若只跑了默认 CBP，必须写「本课实验答不了权重级重置」。

8. Shin, Oh, Cho and Yun, 2024, *DASH: Warm-Starting Neural Network Training in Stationary Settings without Loss of Plasticity*, [arXiv:2410.23495](https://arxiv.org/abs/2410.23495)（NeurIPS 2024）。
贡献：平稳数据一块块到来时，热启动仍会丢可塑性；他们把主因写成记住了噪声，并用按方向收缩（DASH）忘掉噪声、留下特征。
机制：改的是热启动之后的权重处理，不是任务边界上的正则。设定是平稳分布下数据集逐渐变大。对照是从头训和 shrink-and-perturb。摘要写 DASH 在视觉任务上同时改善测试准确率和训练步数。
和本课：课内是非平稳任务流，每个任务换一个线性教师，不是同一分布加数据。官方 `snp.json` 是均匀收缩加噪声，不是按方向收缩。噪声记忆这一句本课实验答不了。
阅读问题：本课每个任务换教师，还算不算他们说的「平稳热启动」？用任务构造回答，不必编测试准确率。

9. Rohani, Khajavi, Chung, Chen and Vaswani, 2025, *Preserving Plasticity in Continual Learning with Adaptive Linearity Injection*, [arXiv:2505.09486](https://arxiv.org/abs/2505.09486)（CoLLAs 2025）。
贡献：每个神经元学一个门，按梯度流往激活里注入线性，叫 AdaLin。
机制：改激活，不加正则、不定时重置、不需要任务边界。深线性网不易丢可塑性，所以他们把线性成分按需注回非线性单元。在 Random Label / Permuted MNIST、Shuffled CIFAR-10、Class-Split CIFAR-100，以及带 ResNet-18 的类增量和离轨 RL 上，接在 ReLU、Tanh、GeLU 上都能抬表现。消融写明必须做到神经元级，不能整层共用一个门。
和本课：课内 tanh 饱和被当成死单元。AdaLin 要的是「别让梯度流断」，不是重置。默认实验没有 per-neuron 门。`sgd_late_gain_drops` 看见增益掉了，看不见门有没有把线性注回去。
阅读问题：若只把 tanh 改成 $\tanh(\cdot)+\alpha\cdot(\cdot)$ 且 $\alpha$ 全网共用，这还算不算他们的神经元级 AdaLin？用消融那句回答；本课没有实现门，不能报分数。

10. Tang, Obando-Ceron, Castro, Courville and Berseth, 2025, *Mitigating Plasticity Loss in Continual Reinforcement Learning by Reducing Churn*, [arXiv:2506.00592](https://arxiv.org/abs/2506.00592)（ICML 2025）。
贡献：从 churn（小批量更新导致批外输出乱跳）看持续 RL 的可塑性丢失，并给出 C-CHAIN 去压 churn。
机制：他们把可塑性丢失和 NTK 矩阵秩逐渐变低连在一起；压 churn 能挡住秩塌缩，并自适应地改普通 RL 梯度的步长。C-CHAIN 在 Gym Control、ProcGen、DeepMind Control、MinAtar 的持续环境上超过他们列出的基线。
和本课：课内是监督线性分类，没有 RL 批外输出，也没有 NTK。死单元和 tanh 增益是另一套相关物。C-CHAIN 和 NTK 秩本课实验答不了。
阅读问题：本课有没有测量「更新之后、未进本批的样本，输出跳了多少」？没有的话，不能把 `sgd_late_speed_slower` 说成已经验证了 churn。

11. Elsayed, Lan, Lyle and Mahmood, 2024, *Weight Clipping for Deep Continual and Reinforcement Learning*, [arXiv:2407.01704](https://arxiv.org/abs/2407.01704)（RLC 2024）。
贡献：许多持续学习和 RL 失败伴随权重大了动不了；把权重剪到固定区间，可以叠在现有优化器上。
机制：改参数本身的可行域，不换优化器、不换结构。摘要写它有助于泛化、缓解可塑性丢失和策略崩，并方便大 replay ratio。动机是权重大了有效步长变野，也更容易过拟合。
和本课：预印本第 4 节把权重幅度当作相关物；课内 CPU 没有把幅度写入 `metrics`。官方仓库的 L2 / S&P 是按住或扰动幅度，不是硬剪。剪区间本课实验答不了。
阅读问题：你若打印一层权重绝对值的均值，后期相对前期涨了多少？没打印就写「本课实验答不了，因为结果 JSON 没有权重范数」。

12. Joudaki, Lanzillotta, Razlighi, Mirzadeh, Alizadeh, Hofmann, Farajtabar and Faghri, 2025, *Barriers for Learning in an Evolving World: Mathematical Understanding of Loss of Plasticity*, [arXiv:2510.00304](https://arxiv.org/abs/2510.00304)。
贡献：用动力系统把可塑性丢失写成参数空间里困住梯度轨迹的稳定流形，并指出两条造阱机制：激活饱和冻住单元，以及表征重复造成的克隆单元流形。
机制：这是分析，不是新损失。静态设定里利于泛化的低秩和简单性偏好，在持续设定里会变成陷阱。他们用数值实验验证，并讨论结构选择和针对性扰动作为缓解。
和本课：`sgd_dead_ratio_rises` 直接对应「饱和冻住单元」这一条。克隆单元流形和稳定流形的证明，课内 10 单元 tanh 网验不了。低秩有利于静态泛化这一句，本课实验答不了。
阅读问题：你的死单元比例从前期到后期涨过 0.15 了吗？若涨了，它支持他们的哪一条阱？若没涨但 `sgd_late_speed_slower` 仍为真，你该写哪一句答不了？

做完这一课，主干循环的第四条「还能不能继续学」终于有了量法。下一课不再加新的优化器零件，而是回到梁文峰说的那个员工类比（转写未获 DeepSeek 确认）：日记写得再好，哪些经验仍然必须改权重。
