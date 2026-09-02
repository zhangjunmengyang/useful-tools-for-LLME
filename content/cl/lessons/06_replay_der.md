---
id: 06_replay_der
title: "把旧样本带在身上"
summary: "回放为什么稳？DER 多蒸馏的那一项在防什么？"
unit: toolkit
play_tools: []
checkpoints:
  - "缓冲大小-遗忘曲线。"
  - "DER 蒸馏项消融。"
  - "论文复现 #1：DER++ 方向性优于同等缓冲的 ER。"
---

# 第 06 课：把旧样本带在身上

> 类型：复现（方向性）+ 实战。正式复现 #1：class-incremental CIFAR-10 上，同等缓冲时 DER++ 优于 ER<br>
> 建议周期：2-4 天<br>
> 硬件：`seq-mnist` 用 CPU / Mac；`seq-cifar10` 用单卡，数小时量级<br>
> 锚定仓库：[aimagelab/mammoth](https://github.com/aimagelab/mammoth) 官方 `er` / `icarl` / `der` / `derpp`<br>
> 产物：缓冲大小-遗忘曲线、DER 蒸馏项消融、复现报告（只要求方向，不要求对齐论文表格里的绝对数字）

## 1. 这一课做什么

[第 05 课](05_ewc_regularization.md) 给慢速权重加了弹簧。弹簧的好处是学新任务时可以不把旧图像留在磁盘上；坏处也立刻暴露了：class-incremental 里新 batch 没有旧标签，交叉熵看不见旧类 logit，Fisher 钉不住分类头。本课换零件：**写到哪里**仍然包括权重，但**怎么写**改成「把旧样本带在身上，训练时混进去」。

主干没变：

```text
新经验进来
  先决定写到哪里（上下文 / 外挂记忆 / 快速权重 / 慢速权重）
  再决定怎么写（覆盖、追加、压缩、约束、正交、合并）
  写完立刻测：新任务会了没、旧任务还在不在
  长期还要测：还能不能继续学
```

回放把旧经验追加进一个容量有限的背包（replay buffer，回放缓冲：只保留一小撮过去的样本，训练新任务时把它们和当前 batch 拼在一起）。背包里装的若是带标签的图像，就是普通 Experience Replay（经验回放，课内简称 ER）。若再把当时网络吐出的 logits（未过 softmax 的原始得分）一并存下，训练时用均方误差把当前输出拉回这些旧得分，就是 Dark Experience Replay（DER，Buzzega et al. NeurIPS 2020）。DER++ 在 DER 之上再加一项：对缓冲里的真标签做交叉熵。

没有这个零件会怎样？正则方法在 Split CIFAR-10 的 class-incremental 协议上经常只比 naive 好一截，有时几乎不好。你会误以为「持续学习就是调 λ」。加上背包之后，同一份骨干、同一套评测，旧任务保持通常会拉开一截。这也是为什么 2020 年前后的图像持续学习榜单里，回放类方法长期压过纯正则。

本课不做三件事。不训 GAN、VAE 去「生成」旧样本（生成式回放只讲机制，见 5.6）；不把「不增加旧损失」写成二次规划（[第 08 课](08_gem_gdumb.md) 的 GEM / A-GEM）；不把缓冲撑到能装下整个训练集（那叫联合训练上限，用来当参照，不是本课方法）。本课的正式复现承诺是方向性的：同等缓冲、class-incremental 小图像上，DER++ 的平均准确率更高或遗忘更低，方向与论文一致。对不上论文表格里某一格的绝对数字，不算失败；同等缓冲下 DER++ 反而稳定弱于 ER，才需要你在报告里写清设定差异。

做完你手里会有：200 / 500 / 2000 三档缓冲的遗忘或平均准确率曲线；ER、iCaRL、DER++ 的对照表；把 DER 的 logits 蒸馏项关掉（α = 0，保留 β）之后遗忘增加多少的消融；一段缓冲内容可视化（每个类存了哪些图，有没有塌成类原型）。

| 术语 | 一句解释 |
|---|---|
| 回放 / rehearsal | 训练新任务时把一小撮旧样本混进 batch，让梯度同时看见旧类 |
| 缓冲 / memory buffer | 容量固定为 N 的背包，装不下就挤掉旧的 |
| 水库采样 | 流式数据上维护固定容量均匀子集的算法，每个新样本以递减概率替换背包里的一位 |
| ER | 普通经验回放：背包只装图像和标签，损失是当前加旧样本的交叉熵 |
| iCaRL | 增量分类与表示学习：按类选原型样本（herding），蒸馏旧输出，测试用类均值近邻 |
| herding | 挑最能代表类均值的样本进背包，而不是随机抽 |
| NME | 近邻类均值分类：用特征空间里每类 exemplar 的均值当原型，测距离不测线性头 |
| logits | softmax 之前的向量；DER 存它，是为了把当时网络的整张决策面留下来 |
| DER | 暗经验回放：背包额外存 logits，用 MSE 强迫当前网络复述自己的过去 |
| DER++ | DER 再加一项缓冲标签交叉熵，α 管蒸馏，β 管旧标签 |
| 生成式回放 | 不存真图，另训一个生成器在需要时造旧样本；本课只讲，不训 |

## 2. 问题

回放为什么是最稳的一类方法？因为 class-incremental 的交叉熵有一个结构漏洞：当前 batch 若不含旧类，旧类 logit 再怎么崩，这一项损失也可以是 0。正则用弹簧间接护着权重；回放把旧类的图像和标签重新放进损失，梯度再次经过那些旧类坐标。漏洞被直接补上。稳，指的是在「共享分类头、测试时不给任务编号、缓冲远小于全数据」这条最常用的赛道上，回放类方法更不容易掉到接近乱猜。不表示它在所有约束下都该上（有的场景不许存用户图像），也不表示缓冲可以无限小。

Dark Experience Replay 多蒸馏的那一项到底在防什么？只回放硬标签，网络只需在缓冲点上输出正确类。决策面在这些点附近可以很尖、很抖，换一张同属旧类、但没被装进背包的图，就可能翻车。DER 把样本写入背包的那一瞬间，把当时整段 logits 存下来。之后网络继续走，DER 用均方误差要求：你在这张旧图上的输出，不要离开当时的那条向量。Hinton 把 softmax 里藏着的类间关系叫做 dark knowledge（暗知识：除了谁是对的，还记下「当时觉得猫有点像狗」这种软关系）。DER 的名字由此而来。它防的是「只记得背包里那几张图的硬标签，却把当时的整张分类几何忘掉」。

由此引出本课必须用实验回答的四件事。

1. 背包容量 N 怎样换旧任务保持。N = 200、500、2000 是 Mammoth 和 DER 论文里常用的三档，本课沿用。N 增大时遗忘应下降，但不是线性的；你要画出曲线，而不是只报最大 N 的一个点。
2. 同等 N 下，ER、iCaRL、DER++ 谁更稳。iCaRL 的背包不是随机的，它按类选原型，测试也不走线性头。DER++ 的背包是水库采样，但多存了 logits。三者不是「同一个回放加三个装饰」，评测协议必须相同才谈优劣。
3. 关掉 DER 的蒸馏项（α = 0，β 保持）遗忘增加多少。这是消融，用来证明多出来的那一项不是装饰。若关掉之后几乎不变，说明你的 α 本来就没起作用，或缓冲已经大到硬标签足够。
4. 缓冲里每个类存的图，看上去像随机相册还是像类原型。iCaRL 应当更像原型；ER / DER 的水库采样更像随手抓的旧照片。可视化是为了看见「压缩」这件事发生在样本层还是特征层。

一个界限要先划清。本课的复现是方向性的：DER++ 相对同等缓冲的 ER，平均准确率更高或遗忘更低。论文表格里的绝对数字依赖骨干、epoch、增强、种子和当年的代码版本，禁止把某一格抄进你的报告当「我复现成功了」。课程蓝图把这一条列为正式复现 #1，通过标准写在第 9 节。

另一个界限：生成式回放（用生成器代替真背包）能解决「不能存用户图像」的合规问题，但本课不训 GAN。5.6 只把它放到地图上，避免你以为回放只有「真存图」一种。真要做生成式，那是另一笔训练预算，不要塞进本课的 2-4 天。

## 3. 准备

- [第 05 课](05_ewc_regularization.md) 的 EWC / LwF 对照表要在，尤其是「class-incremental 上弹簧不够」那一页。[第 03 课](03_cl_evaluation.md) 的 Average Accuracy、Average Forgetting、BWT 是本课报表的列。
- 独立虚拟环境安装 Mammoth。官方文档：仓库根目录 `pip install -r requirements.txt`，入口 `python main.py`。PyTorch 按你的 CUDA 或 CPU 先装好。需要 2.1 及以上才会用到部分 ViT 注意力实现；本课主线是 MNIST / CIFAR 卷积或 MLP 骨干，以 `--help` 和 `requirements.txt` 为准。
- MNIST 在 CPU 上跑三档缓冲足够画曲线。CIFAR-10 的复现对照建议单卡；没有 GPU 时把 CIFAR 标成加分项，主线用 `seq-mnist` 把 ER / DER++ / 蒸馏消融跑完，并在报告里写明「方向性复现改在 MNIST 上做，CIFAR 未跑」。
- 磁盘：CIFAR-10 加上运行日志，留 3 GB 以上。Mammoth 默认把数据放在 `data/`，可用 `--base_path` 改。
- 浏览器实验不装库。先做「背包容量」，再开训练。
- 笔记至少记下：commit、`--buffer_size`、`--alpha`、`--beta`、`--lr`、种子（若你传了）、每个任务结束的 class-IL 准确率。不传 `--wandb_project` 就不会连 WandB。

## 4. 学习目标

1. 画出回放如何补上 class-incremental 交叉熵看不见旧类的漏洞，并指出背包容量是在压缩什么。
2. 写出 ER、DER、DER++ 三个损失，标出 α 蒸馏项和 β 旧标签项各自作用在哪些张量上。
3. 说明水库采样在流式数据上维护的是什么子集，以及 iCaRL 的 herding 和它差在哪。
4. 解释 iCaRL 测试时为什么走 NME 而不是线性分类头。
5. 独立跑通 Mammoth 的 `er`、`icarl`、`der`、`derpp`，完成 200 / 500 / 2000 三档缓冲和 α = 0 消融。
6. 写出复现 #1 的报告：同等缓冲下 DER++ 对 ER 的方向，带设定、不抄论文绝对数字。
7. 能口头说明生成式回放解决的是存图约束，以及本课为什么不训 GAN。

## 5. 原理

六个机制。每个仍走直觉、运转、公式、代码、验证。公式以 Buzzega et al. 2020 与 Rebuffi et al. 2017 为准；代码以 Mammoth 文档站当前源码页为准。

### 5.1 回放在补交叉熵的盲区

第 05 课末尾的裂缝可以写成一句话：新任务 minibatch 的标签集合若与旧类无交，交叉熵对旧类 logit 没有「你必须把正确旧类排第一」的项。弹簧约束的是参数离旧解的距离，是间接的。回放把若干旧样本拼进当前 batch，损失里重新出现旧类标签，梯度重新穿过旧类坐标。这是直接补洞。

类比：你复习考试，弹簧相当于「别把笔记本上圈出来的公式改掉」；回放相当于「书包里永远带着几道旧题，每天抽几道再做一遍」。类比失效处：书包容量有限，旧题会被新题挤出去；而且你带的若只是答案字母，不含当时的完整思路，换一道没带的旧题仍可能不会。后一句就是 DER 要补的。

最小形式（ER）在 Mammoth `models/er.py` 的 `observe` 里几乎是字面翻译：缓冲非空时 `get_data` 取出 `buf_inputs, buf_labels`，与当前 `inputs, labels` 在 batch 维拼接，一次交叉熵，一次 `backward`。写入缓冲用的是未做数据增强的 `not_aug_inputs` 和当前真实标签，避免把增强后的像素当成「那张旧图」。验证：缓冲大小设为 0 或跳过拼接，应退回 naive；缓冲里若从不出现任务 1 的标签，任务 1 的保持应接近 naive。

回放稳，还有一个评测上的原因。第 08 课的 GDumb 会把背包里的样本拿出来从头训练一个模型，常常已经很强。这说明许多 class-incremental 实验的任务边界干净、类内足够 i.i.d.，光是「背包里有一份近似 i.i.d. 的旧数据」就解决了大半遗忘。本课先把背包本身做会，第 08 课再用 GDumb 打脸「你的方法是真在持续学，还是缓冲重训」。

### 5.2 背包容量与水库采样

缓冲是固定整数 N。CIFAR-10 训练集 5 万张，N = 500 只占 1%。装不下时必须丢。怎么丢，决定了背包是「过去的均匀切片」还是「被新任务写满的相册」。

水库采样（reservoir sampling）维护的是：到目前为止见过 t 个样本时，缓冲是这 t 个的均匀随机子集。算法本身没有任务的概念，只有流。前 N 个样本直接装满背包。之后第 t 个样本（$t>N$）以概率 $N/t$ 进入，进入则均匀随机替换背包里的一位；以 $1-N/t$ 丢掉。于是任意一个已经见过的样本，当前仍留在背包里的概率都是 $N/t$（当 $t\ge N$）。这保证「均匀」，不保证「按类均匀」：某个类如果很早就结束，它的样本会在后续很长的 t 里被慢慢稀释，每类张数变成随机波动而不是定额。t 越大，单个新样本挤走旧样本的概率越小，但旧任务的绝对张数期望仍是 $N\times(\text{该类样本占已见样本的比例})$。这就是浏览器实验要你预测的那件事：N 太小，任务 1 的照片会在任务 2、3 里被挤光，任务 1 的准确率掉到随机附近。

Mammoth 的 `utils/buffer.py`（`from utils.buffer import Buffer`）被 ER / DER / DER++ 共用。`Buffer.add_data` 在 ER 里只收 `examples` 和 `labels`；在 DER 里收 `examples` 和 `logits`；在 DER++ 里三者都收。读 `add_data` 时盯两个问题：替换是不是水库；`get_data` 取出时会不会再做一份 `self.transform`（会，DER 的 MSE 因此是在增强后的图上对齐「写入时那份 logit」，这是一种故意的噪声，论文把它当作数据增强下的一致性）。

验证：在任务 1 结束后打印缓冲内标签直方图，应几乎全是任务 1 的两类；全部 5 个任务结束后，直方图应接近 5 个任务各占一份（有随机波动）。若任务 5 独占 80% 以上，水库没有在工作，或你把缓冲在每个任务开始时清空了。

N = 200 / 500 / 2000 不是魔法。DER 论文和后续 Mammoth 实验常用这三档，便于和文献曲线放在同一横轴上。本课沿用，是为了让你的复现报告和论文横轴对齐，不是因为 500 在生产上有任何特权。N 相对「已见类数」才有意义：10 类、N = 200 时每类大约 20 张；100 类时每类大约 2 张，故事完全不同。写曲线时把「每类样本数」标在 N 旁边。

### 5.3 iCaRL：按类选原型，测试不走线性头

Rebuffi、Kolesnikov、Sperl、Lampert 的 iCaRL（CVPR 2017，arXiv:1611.07725）把回放做成一套完整的增量分类器，而不只是「把旧图拼进 batch」。三件套：

1. 表示学习。当前任务的图像和缓冲里的旧 exemplar 一起训练。旧类的目标不是独热，而是旧模型在这些图上的输出（蒸馏），新类才是硬标签。Mammoth `models/icarl.py` 的 `get_loss` 在任务 0 用 `binary_cross_entropy_with_logits` 对独热目标；之后把 `logits[:, :n_past_classes]`（旧网的 sigmoid 目标）和当前新类独热拼成 `comb_targets`，再对全部输出做 BCE。这是论文里「分类 + 蒸馏」的一种实现，注意它用的是逐类 BCE 而不是 softmax 交叉熵。
2. Exemplar 选择。任务结束时 `fill_buffer(..., use_herding=True, normalize_features=True)`。herding 按特征空间里到类均值的距离挑样本，让选中集合的均值逼近该类全部训练样本的均值。背包按类均分：N 固定，类数增加，每类能留的张数下降，旧类 exemplar 会被削减。这是显式的「压缩成原型」，和水库的「随机相册」不同。
3. 分类。`forward` 不算线性头的 argmax。它取当前网络的特征，L2 归一化，和每个已见类的均值比欧氏距离，返回负距离当「分数」。类均值来自缓冲里该类 exemplar 的特征（含水平翻转平均）。这就是 NME（nearest-mean-of-exemplars）。线性头在训练时仍在（BCE 打在 logits 上），测试时被弃用，因为增量过程里头的尺度会漂。

`COMPATIBILITY = ['class-il', 'task-il']`，没有 `general-continual`：iCaRL 依赖任务结束这个时刻来选 exemplar、拷贝 `old_net`。流式、无边界的设定上它不自称能用。ER / DER / DER++ 的兼容列表含 `general-continual`，因为它们每步 `add_data`，不需要切任务。

验证三件事。缓冲可视化：iCaRL 每个类的图应更「像那一类的平均脸」，极端样本更少。关掉 NME、改回线性头 argmax，class-IL 准确率通常掉一截（方向依实现而定，你要自己测，不要默认论文数字）。任务数增加后打印每类 exemplar 数量，应随类数下降，总和仍约等于 N。

iCaRL 的代价是：任务边界必须知道；特征均值对骨干变化敏感；herding 要扫当前类的训练集。它不是 DER 的替代品，是另一条「压缩成原型」的路。本课把三者放在同一 N 下对照，就是为了看见随机回放、原型回放、logits 回放不是一回事。

### 5.4 DER：把当时的 logits 当作暗经验

Buzzega、Boschini、Porrello、Abati、Calderara，*Dark Experience for General Continual Learning: a Strong, Simple Baseline*，NeurIPS 2020，arXiv:2004.07211。设定目标是 General Continual Learning：任务边界可以模糊，不能假设离线多 epoch、不能假设测试时有任务编号。方法却简单：回放加一项 logits 蒸馏。

记当前流上的有标签 batch 为 $\mathcal{D}$，缓冲为 $\mathcal{M}$，网络在 softmax 前的输出为 $h_\theta(x)$，分类损失为 $\ell$（实践里就是交叉熵）。DER 的目标是：

$$
\mathcal{L}_{\mathrm{DER}}=\mathbb{E}_{(x,y)\sim\mathcal{D}}\big[\ell(\sigma(h_\theta(x)),y)\big]+\alpha\,\mathbb{E}_{(x,z)\sim\mathcal{M}}\big[\|h_\theta(x)-z\|_2^2\big]
$$

$z$ 是该样本写入缓冲时的 $h_{\theta_{\mathrm{old}}}(x)$，之后不再更新。第一项学新的；第二项要求网络在旧图上的整段 logits 靠近「当时的自己」。它和 LwF 的差别：LwF 用一份冻结旧模型，在新数据上蒸馏；DER 没有旧模型，蒸馏的对象是缓冲里存下的历史输出，输入是旧图。它和第 05 课 EWC 的差别：EWC 钉参数，DER 钉输出轨迹。

Mammoth `models/der.py` 几乎是公式的直译：

```text
outputs = net(inputs)
loss = CE(outputs, labels)
若缓冲非空:
    buf_inputs, buf_logits = buffer.get_data(...)
    loss += alpha * MSE(net(buf_inputs), buf_logits)
backward, step
buffer.add_data(examples=not_aug_inputs, logits=outputs.data)
```

`--alpha` 必填。写入用 `outputs.data`，切断梯度，避免把「存进缓冲」变成可微的第二张计算图。`get_data` 会按 `self.transform` 再增强，于是 MSE 对齐的是「增强后的当前输出」和「写入时未增强图上的旧 logits」，这是实现选择。读论文公式时把它理解成带增强的一致性约束，不要以为代码在逐像素复现当时那一张。

为什么叫 Dark？因为 logits 里有类间相对关系：当时网络觉得这张 6 有点像 8，这种软结构被 MSE 一起保住。只回放硬标签时，网络可以在缓冲点上打出很尖的独热，类间几何被冲掉。α 过大，网络会拒绝改变任何旧输出，新类学不进；α = 0，DER 退化成「缓冲里只有图、损失却没使用标签」的奇怪 ER，实际上比 ER 还弱，因为 ER 至少对旧标签做了交叉熵。所以单独的 DER（没有 β）和 DER++ 必须分开报。

验证：同一 N，把 α 从 0 扫到比较大，旧任务保持应先升后可能再降。α = 0 的 DER 不应被写成「消融成功的 DER++」，那是另一种方法。

### 5.5 DER++：蒸馏之外再加旧标签交叉熵

DER++ 把硬标签回放加回来：

$$
\mathcal{L}_{\mathrm{DER{+}{+}}}=\mathcal{L}_{\mathrm{DER}}+\beta\,\mathbb{E}_{(x,y)\sim\mathcal{M}}\big[\ell(\sigma(h_\theta(x)),y)\big]
$$

β 管「背包里的题还要做对」。α 管「当时的完整答案卷要像」。两者打在可能不同的缓冲子样本上：Mammoth `models/derpp.py` 连续两次 `get_data`，一次取 `buf_inputs, _, buf_logits` 做 MSE，一次取 `buf_inputs, buf_labels, _` 做交叉熵。两次独立抽样，不是同一 mini-batch 的两项。`add_data` 同时写入 `examples`、`labels`、`logits`。

`--alpha` 与 `--beta` 都是必填。官方 README 的示例命令（写课当天文档站与仓库 README 一致）如下。这是「随机超参」示例，不是最优。

```bash
python main.py --model derpp --dataset seq-cifar100 --alpha 0.5 --beta 0.5 --lr 0.001 --buffer_size 500
```

同页还写：有配置的模型可用 `--model_config best` 加载 `models/config/` 里针对数据集（以及缓冲大小，若适用）的更好参数。本课复现用同等缓冲对比 ER 与 DER++，超参必须写进报告；若你用 `best`，把配置文件名和加载结果一并记下。没有 `best` 时，用 README 示例量级当扫描起点，不要假装那是论文原表。

消融本课规定为：关掉蒸馏项，即 α = 0，β 保持。预期：遗忘上升或平均准确率下降，方向与「蒸馏项在干活」一致。若几乎不变，检查 α 是否本来就没进损失（缓冲为空、`get_data` 失败、α 其实没传到 `args`）。另一侧消融 β = 0 会回到 DER，可作为加分，不占用复现 #1 的通过标准。

验证时看三列：ER（只有硬标签）、DER（只有 MSE）、DER++（两项都有）。复现 #1 比的是 DER++ 对 ER。DER 这一列用来解释「多出来的蒸馏到底有没有独立贡献」。

### 5.6 生成式回放：只讲机制，本课不训

真背包有两道墙。合规：旧图可能是用户数据，不能存。容量：N 再大也是线性的，类数涨到上千时每类只剩一两张。生成式回放的想法是另训一个生成器 $p_\psi(x\mid y)$ 或无条件生成器，学新任务时用它采样「看起来像旧任务」的图，再当回放。Deep Generative Replay（Shin et al.）以及后续把生成器换成 VAE / 扩散的工作，都走这条路。

机制上它仍是回放：损失里要出现旧类。差别在旧样本从磁盘来还是从生成器来。失效处很硬：生成器自己会忘、会 mode collapse，造出来的旧类一旦塌成几张平均脸，分类器就在假数据上过拟合。训生成器的预算通常高于训分类器。本课明确标成「只讲」：不克隆 GAN 仓库、不报生成式方法的准确率、不把「没存图」写成你已经做了生成式回放。

地图上记住位置即可：正则写权重约束，真回放写样本，生成式回放写生成器权重。[第 13 课](13_external_memory.md) 的外挂记忆是另一处「写在模型外面」；和生成式回放不要混。

## 6. 源码导读

本课主仓库是 Mammoth。Avalanche 也有 `Replay` 策略和 `ReplayPlugin`，第 05 课教程里见过它和 EWC 拼接；本课对照以 Mammoth 官方 DER 实现为准，因为那是论文作者实验室维护的代码。读的时候以你检出的 commit 为准。

| 路径 | 零件 | 带着什么问题读 |
|---|---|---|
| `models/er.py` 中的 `Er.observe` | 普通回放 | 当前 batch 和缓冲是拼接后一次 CE，还是两个损失相加？写入用的是哪份图像？ |
| `models/der.py` 中的 `Der.observe` | DER | MSE 打在 `buf_outputs` 和 `buf_logits` 的哪一侧？`outputs.data` 为什么要 `.data`？ |
| `models/derpp.py` 中的 `Derpp.observe` | DER++ | 两次 `get_data` 是不是独立抽样？α、β 乘在哪一项上？ |
| `models/icarl.py` 中的 `get_loss` / `forward` / `end_task` | iCaRL | 蒸馏目标怎样拼？测试是否走 NME？herding 在哪调用？ |
| `utils/buffer.py` 中的 `Buffer` | 背包 | `add_data` 的替换策略是什么？`get_data` 是否再做 `transform`？ |
| `utils/buffer.py` 中与 iCaRL 相关的 `fill_buffer`、`icarl_replay` | 原型填充 | 每类配额如何随任务数下降？ |
| `datasets/seq_cifar10.py`、`datasets/seq_mnist.py` | 基准 | `N_TASKS`、`N_CLASSES_PER_TASK`、`SETTING` 分别是什么？ |
| `models/utils/continual_model.py` | 底座 | `COMPATIBILITY` 如何挡住 iCaRL 跑 general-continual？ |
| `main.py` | 入口 | `--model derpp` 如何映射到文件名？`--model_config best` 读哪个目录？ |

ER 的 `observe` 先记下 `real_batch_size = inputs.shape[0]`，拼接后再用 `labels[:real_batch_size]` 写入缓冲。若忘了这刀，缓冲会把「当前图 + 刚刚抽出来的旧图」当作新样本再存一遍，标签也会错位。这是回放代码里最常见的 off-by-one 类错误。读的时候把这一行标出来。

DER 只存 `examples` 和 `logits`，`get_data` 因此返回两个张量。MSE 是 `F.mse_loss(buf_outputs, buf_logits)`，当前输出在前、旧 logits 在后，默认 `reduction='mean'`，再乘 `alpha`。没有温度、没有 KL：和 LwF 的软交叉熵不是同一项。有人会把 logits 先过 softmax 再 MSE，那是另一种方法，不要改完还叫 DER。

DER++ 的缓冲三项都存。第一次 `get_data` 解包 `buf_inputs, _, buf_logits`，第二次 `buf_inputs, buf_labels, _`。下划线丢掉的那一维每次不同。两次前向是两次独立的 `self.net(...)`，显存和时间大约是 DER 的一点五到两倍，换 N 时要把这个算进预算。

iCaRL 最长，建议按时间线读：`begin_task` 调用 `icarl_replay` 把 exemplar 混进当前训练集；`observe` 在 `current_task > 0` 时用 `old_net` 算 sigmoid 目标；`end_task` `deepcopy` 网络，再 `fill_buffer(..., use_herding=True)`；之后下一次评估走 `forward` 的 NME。`class_means` 在 `observe` 里被置 `None`，因为骨干一更新均值就过期，真正计算推迟到下一次 `forward`。若你在训练中间打印 NME 准确率，会触发一次均值重算，速度会掉，不是 bug。

`seq-cifar10`：`NAME = 'seq-cifar10'`，10 类、5 任务、每任务 2 类、`SETTING = 'class-il'`，图像 32×32。训练增强是 RandomCrop(padding=4) + RandomHorizontalFlip + Normalize，均值 `(0.4914, 0.4822, 0.4465)`，标准差 `(0.247, 0.2435, 0.2615)`。`seq-mnist` 同为 5×2 的 class-IL，28×28。Mammoth 在 class-IL 数据集上会同时报 class-IL 和 task-IL，主结论用 class-IL。

官方 README（写课当天文档站与搜索摘要一致）给出的运行形态：

```text
在仓库根目录调用 main.py
模型名等于 models/ 下的文件名
--buffer_size 是回放方法的背包容量
--model_config best 加载 models/config/ 中的更好超参（若该模型有）
```

旧 issue 里的 `python utils/main.py --model=icarl` 是早期布局。以你克隆到的 `README` 和 `python main.py --help` 为准。

读完应能回答：

1. ER 有没有用到 logits？DER 有没有用到旧标签？
2. DER++ 的 α = 0 之后，还剩哪一项回放？它和 ER 是否同一实现（抽样次数、增强、是否拼接）？
3. iCaRL 的缓冲替换是 herding 削减还是水库随机替换？
4. 为什么 iCaRL 不能声称支持 `general-continual`，而 DER 可以？

第 2 题直接连到消融：α = 0 的 DER++ 仍有 β 交叉熵，但两次独立 `get_data`、不拼接当前 batch，和 ER 的「拼成一个大 batch 一次 CE」不是同一条计算图。报告里不要写「α = 0 等于 ER」。

## 7. 实验

三层都做。浏览器先估容量；CPU 实验钉死「无回放遗忘更大」；Mammoth 上完成三档缓冲、三方法和蒸馏消融。CIFAR-10 的 DER++ 对 ER 是正式复现 #1。

### Step 0: 浏览器实验「背包容量」

打开本课页面上的交互实验。画面是一只格子数为 N 的背包，任务按顺序往里塞样本；满了就按水库规则挤出旧的。下方有任务 1 的保持曲线。

先预测再运行：

1. N 很小（个位数）时，任务 2 结束之后，背包里还能剩几张任务 1？任务 1 准确率大概掉到哪一档（接近随机 / 还能用 / 几乎不动）？
2. 要使任务 1 在全部任务结束后仍保住约 80%，N 至少要到哪一档？先在滑块上指出你的估计，再运行。
3. 同样的 N，把替换规则从水库改成「先进先出」或「只留最新任务」，任务 1 会不会更惨？

合格预测：N 太小则任务 1 被挤光、保持掉到接近随机；存在某个最小 N 让保持回到 80% 附近；只留最新任务时，旧任务崩溃应比水库更狠。系统用缩小的二维分类或小标签流模拟挤出过程，对了才算过关。改 N 会作废上次运行。

这个实验验证 5.2，不替代 CIFAR 上的绝对准确率。80% 是浏览器里的过关阈值，不是论文指标，不要写进复现报告冒充 DER 数字。

### Step 1: CPU 机制实验

在课程仓库的 `experiments/` 目录：

```bash
python3 run.py run 06
```

`python3 run.py run 06` 现在应当全绿，结果写入 `artifacts/lesson06/result.json`。`checks` 应全部为真，键名是 `no_replay_forgets_more`、`no_replay_task1_below_0_75`、`replay_keeps_task1_above_0_80`、`replay_learns_task2_above_0_80`。

同一对二维任务，缓冲 80 条旧样本、每步混 50% 回放。本机一次运行（seed 6）：无回放任务 A 从 0.95 掉到 0.504（下降 0.446），有回放掉到 0.883（下降 0.067），任务 B 仍有 0.879。换机器会变，方向不应变。`summary` 阈值：无回放下降比有回放至少多 0.08；有回放 A、B 都 >0.80；无回放 A<0.75。

这只证明回放在机制上改变了遗忘方向，不是 CIFAR-10 上的 DER++ 分数。复现 #1（同等缓冲下 DER++ vs ER）走 Mammoth，见 Step 6。

### Step 2: 安装 Mammoth 并确认入口

```bash
git clone https://github.com/aimagelab/mammoth.git
```

```bash
pip install -r requirements.txt
```

```bash
python main.py --help
```

确认你能看到 `--model`、`--dataset`、`--buffer_size`。DER++ 还应出现 `--alpha`、`--beta`。把 `git rev-parse HEAD` 记进笔记。不要用旧教程里的 `utils/main.py`，除非你检出的那份 README 仍然这么写。

### Step 3: 同一缓冲下跑 ER 与 DER++（先 MNIST，CPU）

在仓库根目录。`--buffer_size` 对回放方法必填。学习率若你不传，先看 `--help` 和数据集默认；下面显式写出，便于笔记对齐。数值是可跑的起点，不是最优。

```bash
python main.py --model er --dataset seq-mnist --buffer_size 500 --lr 0.03
```

```bash
python main.py --model derpp --dataset seq-mnist --buffer_size 500 --alpha 0.5 --beta 0.5 --lr 0.03
```

官方 README 在 `seq-cifar100` 上演示的是 `--lr 0.001 --alpha 0.5 --beta 0.5 --buffer_size 500`。MNIST 上沿用 0.5 / 0.5，学习率用数据集更常见的 0.03 量级；若 `--help` 或 `dataset_config` 给出默认，以默认加你改过的项为准，并写进笔记。

预期：两条命令都能跑完 5 个任务；DER++ 的 class-IL 平均准确率不低于 ER，或平均遗忘更低。若相反，先不要改方法，检查两者是否真的都是 500、学习率是否被你改成了两个值。CPU 上这一步是机制对照，正式复现 #1 在 Step 6 的 CIFAR-10。

日志在 `data/results/` 下。用 class-IL 列填表。

### Step 4: 缓冲 200 / 500 / 2000，画容量-遗忘曲线

只改 `--buffer_size`，ER 与 DER++ 各跑三档（预算不够时，ER 三档 + DER++ 的 500 一档也行，报告里写清缺了哪格）：

```bash
python main.py --model er --dataset seq-mnist --buffer_size 200 --lr 0.03
```

```bash
python main.py --model er --dataset seq-mnist --buffer_size 2000 --lr 0.03
```

DER++ 把 `--buffer_size` 换成 200 和 2000，α、β、lr 保持与 Step 3 相同。横轴 N，纵轴最终平均遗忘或最终 class-IL 平均准确率（第 03 课怎么定义就怎么报，两种都报更好）。在 N 旁边标注大约每类张数：10 类时 N=200 约 20 张，N=500 约 50 张，N=2000 约 200 张。

预期：N 增大，遗忘下降或平均准确率上升，曲线凹，200 到 500 的间隔通常大于 500 到 2000。若 2000 反而更差，先查是否 epoch / batch 被你改过，或某次跑失败却抄了旧日志。

### Step 5: 加入 iCaRL，并做缓冲可视化

```bash
python main.py --model icarl --dataset seq-mnist --buffer_size 500 --lr 0.03
```

iCaRL 还有 `--opt_wd`（默认 `1e-5`）、`--use_original_icarl_transform`（默认关，且原论文增强只对 `seq-cifar100` 开放）。MNIST 上保持默认。跑完把 ER、iCaRL、DER++ 三行写入对照表，N 都是 500。

可视化：从 `Buffer` 或 Mammoth 保存的 checkpoint / 日志里取出缓冲样本（若当前版本没有现成导出，写一段不超过 30 行的脚本：实例化同样的 `Buffer` 逻辑，或在 `end_task` 后把 `examples` 存成图片网格）。每个类一排。预期：iCaRL 更整齐、更「像均值」；ER / DER++ 更杂、含笔迹很怪的数字。若三类看起来完全一样，你导出的可能是训练集而不是缓冲。

可视化是交付的一部分。没有图，只写「iCaRL 用了 herding」，验收不算过。

### Step 6: 论文复现 #1（CIFAR-10，方向性）

有 GPU 时把数据集换成 `seq-cifar10`，N 至少跑 500 这一档的 ER 与 DER++。命令形态与官方 README 一致，只改数据集和模型：

```bash
python main.py --model er --dataset seq-cifar10 --buffer_size 500 --lr 0.03
```

```bash
python main.py --model derpp --dataset seq-cifar10 --buffer_size 500 --alpha 0.5 --beta 0.5 --lr 0.03
```

若该模型在 `models/config/` 下有 `seq-cifar10` 的 best 配置，可用下面这条加载（以 `--help` 和仓库 `REPRODUCIBILITY.md` 是否列出为准）：

```bash
python main.py --model derpp --dataset seq-cifar10 --buffer_size 500 --model_config best
```

用了 `best` 就必须给 ER 也提供可比较的超参：要么 ER 也有 `best`，要么在报告里声明「DER++ 用了 best，ER 用了手写 lr，因此只解释方向、降低因果力度」。更干净的做法是两边都用手写的同一 lr、同一 epoch。

通过标准（课程蓝图 §4）：

```text
同缓冲、class-incremental 小图像
DER++ 的平均准确率更高，或平均遗忘更低
方向与 Buzzega et al. 2020 一致
不对齐论文表格中的绝对数字
```

预算：Mammoth 的 `seq-cifar10` 默认 epoch 以数据集代码为准，单卡数小时是量级不是承诺。先 `--debug_mode 1`（若存在）确认数据下载和命令能结束，数字作废；再跑完整。没有 GPU 时，把本步改在 `seq-mnist` 上做，报告标题写成「方向性复现（MNIST 代替 CIFAR-10）」，不要假装跑过 CIFAR。

### Step 7: 关掉蒸馏项（α = 0）

同一数据集、同一 N、同一 β 与 lr，只把 α 置 0：

```bash
python main.py --model derpp --dataset seq-cifar10 --buffer_size 500 --alpha 0.0 --beta 0.5 --lr 0.03
```

CPU 路线把 `seq-cifar10` 换成 `seq-mnist`。把「DER++（α=0.5）」「DER++（α=0）」「ER」三列放在一起。预期：α = 0 相对完整 DER++ 遗忘增加或平均准确率下降。增加的幅度写入交付。若三列几乎重合，先打印训练日志里是否出现 `loss_mse`（Mammoth 的 WandB 自动记录以 `loss` 开头的变量；不开 WandB 就在 `observe` 里临时 print `loss_mse.item()`）。α = 0 时这项应为 0。

加分：再跑 `--model der`（只有 α，没有 β），放进第四列。命令：

```bash
python main.py --model der --dataset seq-mnist --buffer_size 500 --alpha 0.5 --lr 0.03
```

DER 没有旧标签项，通常弱于 DER++。它用来解释 β 的贡献，不占用复现 #1。

### Step 8: 写复现报告

结构固定，禁止抄论文里未在你机器上出现的数字：

```text
设定：数据集、任务数、class-IL、缓冲 N、骨干（仓库默认）、epoch、种子
命令：完整一行，含 commit
表：ER / iCaRL / DER++ / DER++(α=0)，列用第 03 课的平均准确率与遗忘
曲线：N = 200, 500, 2000
消融：关掉 α 之后遗忘或准确率的方向
可视化：缓冲网格图的路径
判断：复现 #1 通过或未通过；未通过时写设定差异，不写「论文有误」
```

「通过」只意味着方向一致。写「我复现了 DER++ 在 CIFAR-10 上 xx.x%」却对不上论文表格，是本课明确禁止的句子。

## 8. 配置与预算

| 档 | 数据 | 方法与 N | 硬件 | 用途 |
|---|---|---|---|---|
| 浏览器 | 小标签流 | 拖 N | 任意 | 预测 80% 所需容量 |
| CPU 机制 | 课内缩小数据 | `run.py run 06` | CPU，秒级 | 断言：无回放遗忘更大 |
| 主线 CPU | `seq-mnist` | ER / DER++ 的 200、500、2000；iCaRL 500；α=0 | Mac / CPU，数小时 | 曲线、消融、可视化 |
| 复现 | `seq-cifar10` | ER 与 DER++ 至少 N=500 | 单卡，数小时 | 正式复现 #1 |
| 加分 | `seq-cifar10` | 三档 N + iCaRL + `der` + `model_config best` | 单卡更久 | 更完整的表 |
| 只讲 | 无 | 生成式回放 | 不跑 | 地图上的位置 |

主线按「主线 CPU」写命令。没有 GPU 可以完成除复现 #1 原设定以外的所有验收；复现 #1 改到 MNIST 时必须在报告标题里写明替代。不要为了凑绝对数字把 epoch 加到和论文一样却不记命令。

超参纪律：比 ER 和 DER++ 时，N、数据集、骨干、epoch、学习率一次只允许「方法」这一个差别。消融时只允许 α 变。换 N 时不要顺手改 α。官方 README 的 `--lr 0.001` 出现在 `seq-cifar100` 示例里，搬到 MNIST / CIFAR-10 之前先看数据集默认。

`--minibatch_size` 是缓冲抽样大小，默认可以等于当前 batch。若你改了它，等于改了「每步看见多少旧样本」，必须在表里单列。不要把它和 `--buffer_size` 当成同一个数。

## 9. 验收

- [ ] 白纸写出 DER 与 DER++ 的损失，能指出 α 打在 logits 的 MSE 上、β 打在缓冲标签的交叉熵上。
- [ ] 能口述水库采样的进入概率 $N/t$，以及 iCaRL herding 与它的差别。
- [ ] 浏览器实验预测通过：给出一个使任务 1 保住约 80% 的 N，并说明只留最新任务会更差。
- [ ] `python3 run.py run 06` 的 `checks` 全真（`no_replay_forgets_more`、`no_replay_task1_below_0_75`、`replay_keeps_task1_above_0_80`、`replay_learns_task2_above_0_80`）。
- [ ] 缓冲 200 / 500 / 2000 的曲线已画，横轴旁标注每类大约张数。
- [ ] ER、iCaRL、DER++ 在同一 N 下的对照表已填，指标用第 03 课定义。
- [ ] 缓冲可视化已保存：能指出哪一排更像原型、哪一排更像随机相册。
- [ ] α = 0 消融已跑，遗忘或平均准确率的方向写入报告。
- [ ] 复现 #1：CIFAR-10（或声明替代的 MNIST）上，同等缓冲 DER++ 对 ER 的方向已判定通过或未通过。
- [ ] 报告里没有出现「论文表格里的绝对数字被我复现了」这类句子。
- [ ] 能说明生成式回放本课为什么标成只讲。

口头抽查：α = 0 的 DER++ 等于 ER 吗？答案是否定的，抽样和是否拼接都不同。再问：iCaRL 测试时为什么不用线性头？答案是增量过程里头会漂，NME 用 exemplar 均值更稳。答得出，回放三件套就算分开了。

## 10. 排错

| 症状 | 原因 | 验证 | 修法 |
|---|---|---|---|
| `--buffer_size` 报 required | 回放方法没有默认 N | `--help` 里该参数是否必填 | 必须显式传 200 / 500 / 2000 |
| `--alpha` / `--beta` 报 required | DER / DER++ 当前实现把它们标成必填 | 模型文档页 | 按 README 示例先填 0.5；消融再改 |
| `python utils/main.py` 找不到 | 早期布局，当前入口是根目录 `main.py` | 根目录是否有 `main.py` | 改用 `python main.py` |
| ER 比 DER++ 好很多，且你想写「论文反了」 | 两边 lr、epoch、N 不一致；或 DER++ 的 α、β 没传上 | 对比两条完整命令 | 先对齐设定；仍反转则在报告写「本设定未通过」，列差异 |
| α = 0 与完整 DER++ 曲线重合 | 蒸馏项没进图；缓冲一直为空；α 没进 `args` | print `loss_mse`；查 `observe` 是否走进 `if not buffer.is_empty()` | 确认第一步之后缓冲已有样本；确认命令行真的解析了 α |
| iCaRL 一启动就断言数据集 | `--use_original_icarl_transform 1` 只用在 `seq-cifar100` | 读 `ICarl.__init__` | MNIST / CIFAR-10 保持默认 0 |
| iCaRL 警告 optimizer weight decay 被忽略 | 它用自己的 `--opt_wd` | 日志里的 warning | 不要同时设通用 `optim_wd` 还指望它生效 |
| 缓冲可视化全是噪声或全黑 | 存的是 Normalize 之后的张量，直接当图片显示 | 看取值范围是否在约 [-2, 2] | 用数据集的 denormalization，或导出 `not_aug` 像素 |
| 缓冲几乎全是最后一个任务 | 每任务重建了 Buffer；或没用水库 | 任务 1 结束与任务 5 结束各打一次标签直方图 | 确认 `Buffer` 跨任务存活；不要在 `begin_task` 里重新 `__init__` |
| CUDA OOM | CIFAR + 过大 batch + DER++ 两次额外前向 | `nvidia-smi` | 减 `--batch_size` / `--minibatch_size`；先 MNIST |
| `--model_config best` 找不到配置 | 该模型尚未提供该数据集的 best | `models/config/` 与 `REPRODUCIBILITY.md` | 退回手写 α、β、lr，不要空造一份 yaml |
| 两个方法的 class-IL / task-IL 抄反 | Mammoth 两列都打 | 日志字段名 | 主表只用 class-IL；task-IL 可附录 |
| 复现报告写了论文里的 72.x% 之类 | 你没有在自己机器上得到这个数 | 报告中每个数字能否追溯到一条命令 | 删掉无法追溯的数字，只保留你跑出来的 |
| `no_replay_forgets_more` 为假 | 回放没混进 batch，或两条链不是同一任务对 | 看 `drop_no_replay` 与 `drop_replay` | 对照 `lesson_06.py`：无回放下降应比有回放至少多 0.08 |
| `replay_keeps_task1_above_0_80` 为假 | 缓冲为空，或 `replay_frac` 没进 SGD | 看 `replay_acc_task1`、`buffer_size` | 确认缓冲 80、每步混 50%；不要把这组二维数字写成复现 #1 |

训练到一半缓冲仍为空，几乎一定是 `add_data` 没被调用或 `buffer_size` 被解析成 0。DER 的 `observe` 在 `step` 之后才 `add_data`，第一个 minibatch 没有蒸馏是正常的，整段任务都没有就不正常。

## 11. 前沿与改造

前沿怎么做。2020 年之后，回放仍然是 class-incremental 图像分类最稳的底座。后续工作多在三处加料：缓冲里存什么（logits、特征、注意力图）；怎么抽样（平衡类、难例、梯度匹配）；和正则、投影、提示学习怎么拼。大模型阶段，回放变成「混一点旧指令或旧通用语料」，[第 09 课](09_continual_pretraining.md) 会把它认成「每 batch 混 30% 通用数据」。DER 的「存当时输出」在 LLM 里对应存旧 logits 或旧分布，算力更贵，思想相同。近两年主阅读见第 12 节：InsCL 按指令相似度调回放配额，Ibrahim 把学习率回热和旧 token 混料拼在一起，SRT 按每条样本的遗忘速度排复习。回放对象换成 token 与指令，CIFAR 图像那一套不能直接搬。

我们差在哪。本课缓冲最多 2000，任务边界干净，类数 10。真实数据流没有 `end_task`，类长尾，还可能不许存原图。iCaRL 的 herding 依赖任务结束扫全类；DER 的水库对长尾会进一步饿死稀有类。不要把 N = 500 的 CIFAR-10 曲线写成产品容量规划。

动手改造（精简四个，预算按 `seq-mnist` CPU）：

1. 类平衡抽样。改 `get_data`，强制每次缓冲 batch 各类张数尽量均匀。模块：`utils/buffer.py`。预期：小 N 时尾类保持上升。失败标准：总平均准确率明显下降且尾类也没好，说明实现把多数类抽没了。
2. 存 softmax 而不是 logits。在 `der.py` 的 `add_data` 前对 `outputs` 做 softmax，MSE 改打在概率上。预期：对温度更敏感，旧任务保持通常不如原版 logits。失败标准：若反而全面超过原版，先查你是否同时改了 α。
3. 把 DER++ 的两次 `get_data` 合成一次，同一批样本上既做 MSE 又做 CE。预期：更快，方差更小；平均准确率与原版同方向。失败标准：速度没变（你其实还是两次前向）。
4. 极小 N 的 GDumb 预习。N = 200，每个任务结束只用缓冲从头训一个新模型（可手写，不必等第 08 课仓库命令）。预期：这个傻瓜基线可能接近甚至超过你没调好的 ER。失败标准：把它的分数写成本课 DER++ 的分数。正式对比在第 08 课。

顺手复现映射。本课就是课程五项复现的第 1 项。第 08 课的复现 #2 会在小缓冲 MNIST 上拿 GDumb 打 A-GEM，用的还是这只背包。你现在画的 N 曲线会直接被那一课引用。

## 12. 论文与延伸

谱系只留本课机制实验真正用到的 DER。主阅读是 2024-2026：大模型回放的是 token 和指令，抽样从均匀混料变成按相似度、按间隔、按能不能帮下一个任务。

1. Buzzega, Boschini, Porrello, Abati, Calderara, 2020, *Dark Experience for General Continual Learning: a Strong, Simple Baseline*, [arXiv:2004.07211](https://arxiv.org/abs/2004.07211)。
贡献：把回放和「对齐历史 logits」合成极简基线 DER / DER++，面向任务边界可以模糊的 GCL。机制发明处，不是本课主阅读。
机制：缓冲存图和当时的 logits，新损失里加 MSE 对齐旧输出；DER++ 再加旧标签交叉熵。改的是存储内容和损失，不扩网络。摘要写在标准基准和 MNIST-360 上，有限资源下超过当时一批方法。
和本课：Mammoth `der.py` / `derpp.py` 是官方实现；复现 #1 比同等缓冲下 DER++ 对 ER。CPU 实验只有标签回放，没有 logits。`no_replay_forgets_more` 只能支持「有缓冲优于无缓冲」；蒸馏项要看 Step 7 关 α。
阅读问题：你关掉 α 之后遗忘是否上升？若否，你的实验能支持「蒸馏项在防什么」，还是只能支持「有缓冲优于无缓冲」？

2. Wang, Liu, Shi, Li, Chen, Lu, Yang, 2024, *InsCL: A Data-efficient Continual Learning Paradigm for Fine-tuning Large Language Models with Instructions*, [arXiv:2403.11435](https://arxiv.org/abs/2403.11435)。
贡献：按指令相似度动态决定回放多少，再按指令信息量偏向高质量样本。
机制：用指令嵌入的 Wasserstein 距离估任务相似度，差得远的旧任务多回放。InsInfo 用指令标签的数量和稀有度打分，高分指令多抽。改的是回放抽样，不改模型结构。摘要写 16 个任务、多种顺序，全部训完后相对随机回放 Relative Gain +3.0，相对无回放 +27.96。
和本课：CPU 实验是均匀抽 80 条旧样本。浏览器实验能看见 N 太小旧任务被挤光。InsCL 的「按任务相似度调配额」本课没有指令嵌入，答不了 Wasserstein。
阅读问题：本课水库是均匀替换。若任务 1 和任务 2 很不像，InsCL 会给任务 1 更大回放份额。你的 CPU 实验两个任务方向正交，属于「差得远」。`replay_keeps_task1_above_0_80` 只验证了有回放能保住，答不了「份额该不该按相似度调」。

3. Hickok, 2025, *Scalable Strategies for Continual Learning with Replay*, [arXiv:2505.12512](https://arxiv.org/abs/2505.12512)。
贡献：把 LoRA、任务结束后的巩固回放、顺序权重合并接到同一套回放工具里，降低朴素回放的样本成本。
机制：学新任务时把回放比例压低，省下的步数拿到任务结束后专训旧样本。再把训前和训后权重做顺序合并。摘要写 consolidation 最多能把达到同一成绩所需的回放样本减少 55%。改的是回放日程和权重合，不改 DER 那种 logits。
和本课：本课每步固定混 50% 回放，没有「任务后再巩固」阶段。`replay_frac=0.5` 对应他们 RR 较高那一档的朴素用法。巩固阶段本课实验答不了。
阅读问题：本课 CPU 实验从头到尾回放比例固定。若把回放从训练过程抽走、只在任务 B 结束后用缓冲再训几步，`replay_keeps_task1_above_0_80` 还会不会过？本课没有这个变体，应写「本课实验答不了，因为没有任务后巩固阶段」。

4. Wang, Chandra, Zhang, 2025, *Experience Replay Addresses Loss of Plasticity in Continual Learning*, [arXiv:2503.20018](https://arxiv.org/abs/2503.20018)。
贡献：提出假说：经验回放可以消除持续学习里的可塑性丢失；证据是回放加 Transformer 处理缓冲后，可塑性丢失消失。
机制：缓冲存最近 n 条，把记忆和当前样本拼成序列，用 Transformer 读。不改反传、不改激活、不加正则。摘要写在回归、分类、策略评估上都看到可塑性丢失消失；MLP 和 RNN 即使加回放仍学不动。作者猜测是上下文学习。
和本课：CPU 实验的线性分类器加回放，保住的是旧任务准确率，测的不是「对新任务是否越来越学不会」。本课只有两个任务，答不了可塑性丢失。`replay_learns_task2_above_0_80` 只说明这一次还能学会 B。
阅读问题：本课无回放时任务 A 掉到检查阈值以下，有回放 A、B 都在 0.80 以上。这支持的是抗遗忘，还是抗可塑性丢失？用任务数量回答：两个任务看不出「越往后越学不会」。可塑性那一句本课实验答不了。

5. Cho, Moon, Chunara, Cho, Cha, 2025, *Forget Forgetting: Continual Learning in a World of Abundant Memory*, [arXiv:2502.07274](https://arxiv.org/abs/2502.07274)。
贡献：当缓冲已经大到能压住遗忘、但还付不起从头重训时，主矛盾从稳定变成可塑；并提出权重空间巩固。
机制：大缓冲让旧分布近似得好，模型偏向旧任务，新任务梯度被冲淡。方法是按梯度矩给参数打分，把休眠参数往预训练值软重置，再做训练中的权值滑动平均。改的是权重操作，回放仍然在。摘要写在 class-IL 和 LLM 指令持续微调上，成绩超过一批对照，算力接近朴素回放。
和本课：CPU 缓冲 80/240，不算「充足记忆」。本课主矛盾仍是遗忘。`no_replay_forgets_more` 在小缓冲下成立；「缓冲再加大之后新任务反而学不动」本课没有扫 N 到接近全数据，答不了。
阅读问题：本课 Step 4 把 N 从 200 扫到 2000。曲线若在大 N 上旧任务更好、新任务开始掉，才接近这篇的可塑性问题。你的容量曲线新任务一列有没有掉？没有的话，写「本课 N 还没到充足记忆，答不了可塑性那一句」。

6. Ibrahim, Thérien, Gupta, Richter, Anthony, Lesort, Belilovsky, Rish, 2024, *Simple and Scalable Strategies to Continually Pre-train Large Language Models*, [arXiv:2403.08763](https://arxiv.org/abs/2403.08763)。
贡献：学习率回热、再退火、加旧数据回放，这三件简单的事拼起来，就能在损失和平均基准上追上用全部数据从头训。
机制：从已经余弦衰减到很小的检查点接着训，必须把学习率重新拉高再降下来，否则新语料学不进去。回放按计算量等价替换新 token。HTML 图注写弱偏移（Pile 到 SlimPajama）用 5% 回放，强偏移（到德语）用 25%。改的是优化日程和混料，不是图像缓冲。
和本课：本课回放的是带标签的旧图。Ibrahim 回放的是旧 token。CPU 实验没有学习率日程。第 09 课才会把约 30% 通用语料混进去。`replay_keeps_task1_above_0_80` 只能类比「混一点旧数据能保住旧分布」。
阅读问题：本课每步 50% 回放，远高于他们英文到英文的 5%。为什么图像 class-IL 要这么多？用「当前 batch 没有旧类标签，分类头看不见旧类」回答。5% 在本课 CPU 设定够不够，本课没扫回放比例，应写答不了具体百分比。

7. Atreya, Batra, Mantri, Bantug, Cowan, Khraishi, 2026, *When to Review: Spaced Repetition for Continual Pre-Training of Language Models*, [arXiv:2608.17530](https://arxiv.org/abs/2608.17530)。
贡献：续预训练的回放不该全局均匀混，应按每条样本忘得多快排复习时间。
机制：每条样本存 ease、复习次数、间隔、到期步。用这条的困惑度映射成 1 到 5 的回忆质量，再按 SuperMemo-2 更新下次间隔。到期的旧样本和新样本一起组 batch。模型、损失、优化器都不改。摘要写在时间切开的维基和代码上，能挽回朴素续预训练丢掉的 5 到 37 个百分点旧知识准确率。
和本课：本课水库均匀抽。SRT 按困惑度决定谁该回来。CPU 实验每条样本没有复习状态。`no_replay_forgets_more` 只对比有无回放，不对比均匀和间隔复习。
阅读问题：浏览器实验把替换改成「只留最新任务」会更惨。SRT 做的是反方向：忘得快的旧样本更早回来。本课实验能支持「均匀挤出有害」，支持不了「按间隔排期更好」，因为没有按样本的日程。

8. Meng, Liu, Zhao, Chen, 2026, *Rethinking Transfer in Continual Learning: A Replay-Based Realisation*, [arXiv:2607.15587](https://arxiv.org/abs/2607.15587)。
贡献：先问前向迁移何时存在，再用按任务签名挑选的回放去实现它。
机制：三个条件：目标任务自己的监督还留有提升空间；载体必须在优化过程中一直在（数据回放比一次性初始化参数更持久）；来源任务要相容。TSR 用共享初始化上的梯度签名给旧任务打分，按 softmax 路由回放，并用该任务结束时的快照做蒸馏保稳定。摘要写在低预算设定下，前向迁移上升，稳定也还在。
和本课：本课回放一律来自任务 A，只有一个旧任务，谈不上选源。`replay_learns_task2_above_0_80` 看见的是「旧样本没挡住学 B」，看不见「哪一类旧样本在帮 B」。前向迁移矩阵本课答不了。
阅读问题：本课两个任务方向正交。TSR 会认为它们签名不相容，少回放 A。你的 `replay_keeps_task1_above_0_80` 依赖的正是回放 A。这说明本课实验优化的是旧任务保持，不是新任务迁移。本课答不了「该选哪些旧任务来帮 B」。

现在整个系统长这样：新经验仍写入慢速权重，但训练 batch 里混进一只容量为 N 的背包。背包可以装随机旧图（ER）、类原型（iCaRL）、旧图加当时的 logits（DER / DER++）。测的时候你已经会看容量曲线、蒸馏消融，以及 class-IL 列。缺的零件是：有人选择不改旧权重、只再长一块网络或加一组 prompt；还有人把「旧损失不得升」写成约束，而不是写进损失。[第 07 课](07_architecture_prompts.md) 先走扩结构这条路，看 PackNet 怎样给权重砌墙、提示学习把新知识写在哪。[第 08 课](08_gem_gdumb.md) 里，同一只背包会变成梯度投影的约束集，并拿出 GDumb 来打脸。

这只背包对 Agent 记忆来说仍是外挂。卸掉再学新的，旧事实会没。不存旧键、让旧 $W$ 自己出题，是另一条路：

```bash
python3 run.py extra run buffer
```

```bash
python3 run.py extra run gendream
```

GPU 上对照 `gpu print vandeven-er`（要留样本）和 `gpu print vandeven-dgr`（生成回放）。Mammoth：`gpu print mammoth-icarl`、`gpu print mammoth-lwf`。
