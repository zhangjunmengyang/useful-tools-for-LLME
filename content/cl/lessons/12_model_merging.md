---
id: 12_model_merging
title: "不接着训，把几个模型加起来"
summary: "任务向量相加为什么有时有效？合并算不算持续学习？"
unit: llm
play_tools: []
checkpoints:
  - "三份合并对照。"
  - "书面判断：合并是事后缝合，不是在线持续学习。"
---

# 第 12 课：不接着训，把几个模型加起来

> 类型：实战（mergekit 合并与评测）+ 机制复现（任务向量加减、TIES、DARE）<br>
> 建议周期：2-3 天<br>
> 硬件：合并本身 CPU 即可；评测需要能加载小模型。档 A 完成浏览器、CPU 机制和 `mergekit-pytorch` 小张量合并<br>
> 锚定仓库：[arcee-ai/mergekit](https://github.com/arcee-ai/mergekit)<br>
> 产物：线性相加 / TIES / DARE 三份合并对照、一次负任务向量粗遗忘、一段书面判断（合并是事后缝合，不是在线持续学习）

## 1. 这一课做什么

第三幕最后一课。[第 09 课](09_continual_pretraining.md) 续预训练，[第 10 课](10_sequential_instruction.md) 顺序指令，[第 11 课](11_olora_treelora.md) 用 O-LoRA 把新任务的低秩方向尽量正交到旧任务之外。那三条路都还在「接着训」：数据流过 GPU，优化器改权重，旧能力靠约束或回放保。

本课把优化器关掉。手里有同一预训练出发、各自微调好的几个模型（或几份 LoRA），在权重空间里做加法和减法，得到一个新的权重文件。不再看旧数据，也不再反向传播。这条路叫模型合并（把多个已有权重合成一份，推理仍是一张网）。工业界用 [arcee-ai/mergekit](https://github.com/arcee-ai/mergekit) 做这件事；论文源头是任务向量（微调权重减预训练权重，指向「学会这件事」的方向）。

没有这一课，你会把 Hugging Face 上成千上万的微调权重当成互不相干的文件，也会把「把两个 LoRA 加起来」误认成持续学习。做完之后你能验证：二维上加、减、TIES 修剪分别把合成向量扔到哪一象限；小模型上线性相加、TIES、DARE 三份结果谁在两个任务上都还能用；负任务向量能把目标任务成绩打掉多少、无关任务掉多少。书面上你要写清：合并是事后缝合，能在不碰旧数据时回收多任务能力，但它不是在线持续学习。

主干里本课换的是「怎么写」：写的动作变成合并；写的时机是几个专家都已经训完之后。写到哪里仍是慢速权重（整网或 LoRA 增量），不是第 13 课的外挂日记。

梁文峰转写里有一句：全世界还没找到真正管用的持续学习方法，还在摸索（转写未获 DeepSeek 确认）。合并在开源社区很管用，那是因为它解决的是「已经有几个专家，想合成一个」；它不解决「模型在岗的几个月里自己接着变」。课内把这条边界写死，免得把 mergekit 排行榜当成第 24 课的毕业标准。

术语速查：

| 术语 | 一句解释 |
|---|---|
| 任务向量 | $\tau=\theta_{\mathrm{ft}}-\theta_{\mathrm{pre}}$，微调相对预训练挪了哪一截 |
| 任务算术 | 对任务向量做加、减、类比，再加回预训练权重 |
| 线性合并 / model soup | 几个权重按系数求平均；soup 多指同一任务不同超参的汤 |
| 符号冲突 | 同一参数在两个任务向量里一正一负，直接平均会互相抵消 |
| TIES | 先剪掉小变化，再投票选出符号，只平均符号一致的值 |
| DARE | 随机丢掉一部分增量并按 $1/(1-p)$ 放大剩下的，再拿去合并 |
| 负任务向量 | $-\tau$，从预训练往「不会这件事」的方向走，粗遗忘 |
| mergekit | 按 YAML 在 CPU 或少量显存上执行上述算法的工具 |
| 事后缝合 | 专家已经各自训完，才在权重空间里拼；中间没有在线学习循环 |

## 2. 问题

第 11 课的正交 LoRA 要求你在训练时就知道任务边界，并且按顺序更新。现实里更常见的库存是：社区已经放出「数学 LoRA」「代码 LoRA」「医疗全量微调」，你没有他们的训练数据，也不想再花一遍算力。问题变成：这些成品能不能合成一个还能同时做几件事的模型？

任务向量给出一个意外地简单的答案。微调相对预训练的差 $\tau$ 往往近正交（Ilharco et al. 在 CLIP 上画过余弦，不同任务的 $\tau$ 大多接近 90°）。近正交时，把几个 $\tau$ 加起来，彼此干扰小，加回预训练就能同时变强。一旦不正交，或者同一坐标上一正一负，加法就会把有用的增量削掉。TIES 修的是后一种：符号冲突。DARE 修的是另一种冗余：SFT 的增量里绝大部分坐标可以随机丢掉，只要把留下的放大，期望嵌入近似不变。

第三条问题是遗忘。若加法能「学会」，减法能不能「忘掉」？Ilharco 把 $\tau$ 取负，毒性生成或某个分类任务的准确率下降，对照任务（WikiText 困惑度、ImageNet）变化不大。这和 [第 14 课](14_knowledge_editing.md) 的定位编辑、机器遗忘不是同一分辨率：负任务向量是粗粒度的「往反方向走一整截」，ROME/MEMIT 是「改某层 MLP 里某几个关键」。本课只做粗的，细的留给第 14 课。

第四条必须书面回答：这算不算持续学习？持续学习的循环是：新经验进来，决定怎么写，写完立刻测新旧，长期还要测还能不能继续学。合并发生在所有经验都已经变成成品权重之后，没有在线写入，也没有对第三条「还能不能继续学」负责。它能做的是：不回放、不训练，回收多任务能力。课内把这件事叫事后缝合。

## 3. 准备

- 第 10 课若留下了两份指令 LoRA，本课直接拿来合并。没有的话，用两个公开的、从同一基座微调的小模型或 adapter，或用 CPU 实验里合成的张量。
- Python 3.10+。mergekit 需要较新的 pip（官方 README：若 editable 安装报找不到 `setup.py`，先把 pip 升到 21.3 以上）。
- 独立虚拟环境。不要和 O-LoRA 那套 `transformers==4.28.0` 混装。
- 磁盘：两个小模型加一份合并输出，GPT-2 级几百 MB；1B 级按实际权重算。
- 本课网页自带「任务向量加法」实验。先预测再运行。
- 不必 7B。合并算法对张量形状敏感，对参数量不敏感；评测用你能加载的最小同构模型。

Ilharco 的官方代码在 [mlfoundations/task_vectors](https://github.com/mlfoundations/task_vectors)，TIES 在 [prateeky2806/ties-merging](https://github.com/prateeky2806/ties-merging)，DARE 在 [yule-BUAA/MergeLM](https://github.com/yule-BUAA/MergeLM)。课内锚定只认 mergekit，那三份是读论文时对照公式用的，不要求安装。mergekit 把它们收成统一 YAML，避免三套依赖。

这和档 A 的分工一致：几何在浏览器和 CPU 上钉死，工具链用小张量或小模型跑通，大模型只是同一份 YAML 换路径。不要为了「看起来更像论文」去下载 13B 示例里的四个仓库。磁盘、显存和评测时间会把本课从两天拖成一周，而公式一步都不会因此变更清楚。论文数字只用来对照方向，例如 Ilharco 两两相加保住 98.9% 归一化准确率：你的两个小任务只要合成后双任务都高于基座，方向就对了。达不到 98.9% 很正常，任务更难、模型更小、$\lambda$ 没扫过，都会把这个比例压下来。把「高于基座」当通过线，「接近专家平均」当加分线，不要把论文的 98.9 当成自己的验收阈值。CLIP 八任务和两个指令 LoRA 也不是同一难度，只借几何，不借分数。几何就是：近正交则相加干扰小，符号冲突则要修剪或投票。这两句够你预测浏览器实验的象限，也够你解释 YAML 里为什么必须写上基座。

## 4. 学习目标

1. 写出任务向量的定义，以及加、减之后如何加回预训练权重，包括缩放 $\lambda$。
2. 在二维上指出：相加落在哪一象限、取负落在哪一象限、TIES 在符号冲突时丢哪一支。
3. 说出 TIES 三步（修剪、选符号、只合并同号）各自在防哪一种干扰。
4. 写出 DARE 的丢弃与 $1/(1-p)$ 缩放，说明为什么丢掉的必须是增量而不是微调后的全量权重。
5. 用 mergekit 的 YAML 跑通 `task_arithmetic`、`ties`、`dare_ties` 三种 `merge_method`，命令与当前 README 一致。
6. 做一次负任务向量，记录目标任务和对照任务的变化，并写清它和第 14 课 unlearning 的差别。
7. 交一段书面判断：合并是事后缝合，不是在线持续学习；它能在不碰旧数据时回收多任务能力。

## 5. 原理

五个机制，节奏仍是直觉、运转、定义、代码、验证。

### 5.1 任务向量：微调相对预训练挪了哪一截

同一个架构、同一次预训练 $\theta_{\mathrm{pre}}$，在任务 $t$ 上微调得到 $\theta_{\mathrm{ft}}^t$。差

$$
\tau_t=\theta_{\mathrm{ft}}^t-\theta_{\mathrm{pre}}
$$

就是任务向量。它是权重空间里的一根箭头：沿这个方向走，这个任务变好。$\lambda=1$ 时把 $\tau_t$ 加回预训练，得到的就是那份微调模型。$\lambda$ 取在 0 和 1 之间，相当于在预训练和微调之间插值；$\lambda>1$ 是沿同一方向多走一步。

类比：GPS 里「家」是预训练，「公司」是微调，任务向量是那条通勤路。把「公司减家」和「超市减家」加起来，有时能得到一个既像去公司又像去超市的点。类比失效处：通勤路近正交时加法才干净；两条路严重重叠时，合成点可能卡在河中间。Ilharco et al. 的 Figure 5 显示 CLIP 上多数任务向量两两余弦接近 0，相似任务（MNIST / SVHN / GTSRB 都要认数字）余弦更高。近正交是加法有效的经验前提，不是定理。

编辑后的权重一律写成

$$
\theta_{\mathrm{new}}=\theta+\lambda\,\tau_{\mathrm{new}}
$$

$\theta$ 通常是 $\theta_{\mathrm{pre}}$。$\lambda$ 用一块留出的验证集挑，没有验证集时 TIES 论文给过固定配方：留最大的 20%、$\lambda=1$；Task Arithmetic 原文常用 $\lambda=0.4$ 量级。课内 CPU 实验把 $\lambda$ 扫几个值，看二维点的落点，不把某一个 $\lambda$ 神化。

限制：只对同一架构、同一预训练出发的权重做逐元素运算。换骨架或换初始化，不能直接减。Ilharco 也只在「微调不引入新参数」（开放词表分类、text-to-text）上做主实验。

从 LoRA 到任务向量。第 11 课存的是 $A,B$，不是整网。折回 $\Delta W=AB$（O-LoRA 论文字母）再当作 $\tau$ 即可，因为冻住的 $W$ 就是 $\theta_{\mathrm{pre}}$，LoRA 增量就是 $\tau$。两份 LoRA 相加对应 $\theta_{\mathrm{pre}}+\lambda(\Delta W_A+\Delta W_B)$。不要把 $A_A$ 和 $A_B$ 直接平均：秩和尺度都对不上，正交约束也不是为平均设计的。若 mergekit 吃完整模型更省事，先把 adapter `merge_and_unload` 成两个目录再写 YAML。

Ilharco 还有一类比：任务满足「A 之于 B 如同 C 之于 D」时，用 $\tau_C+(\tau_B-\tau_A)$ 去提高 D，即使 D 没有标注。课内主线不做类比，知道有这第三种算术就行，避免把「任务向量」理解成只会加和减。
### 5.2 加法：多任务；减法：粗遗忘

加法。$n$ 个任务向量求和

$$
\tau_{\mathrm{new}}=\sum_i \tau_i,\qquad \theta_{\mathrm{new}}=\theta_{\mathrm{pre}}+\lambda\sum_i \tau_i
$$

得到一个理论上同时会几件事的模型。Ilharco 在 CLIP 八个分类任务上：两两相加平均保住微调模型 98.9% 的归一化准确率；八个全加，最好的合成模型平均到 91.2%。T5 上他们从 Hub 搜检查点往已微调模型上加，四个 GLUE 任务各有零点几个点的增益。课内你只有两个任务时，对照三个数：只加 $\tau_A$、只加 $\tau_B$、加 $\tau_A+\tau_B$。合成模型在 A、B 上都应明显高于「只用另一个任务的投影」。

减法 / 取负。$\tau_{\mathrm{new}}=-\tau$ 时，模型沿微调的反方向走。Ilharco 用 CLIP ViT-L/14 忘掉八个目标任务，平均目标准确率掉 45.8 个百分点，ImageNet 对照几乎不动；GPT-2 Large 上，用 Civil Comments 高毒性子集训出的 $\tau$ 取负，毒性生成从 4.8% 降到 0.8%，WikiText-103 困惑度仍在预训练附近。随机向量做不到；沿着损失上升方向微调（gradient ascent）会把对照任务一并毁掉。所以「取负」不是乱走，是沿那条已经学过的任务轴退回去。

和第 14 课的差别写在这里，实验里还要再测一遍。负任务向量动的是整网、整条任务轴，粒度是「这个分类头 / 这种毒性风格」。知识编辑（ROME 等）动的是某层 MLP 里少数关键，粒度是「X 的首都是 Y」。机器遗忘基准（MUSE）还要看相邻知识在不在、能否从残留里反推被忘的内容。负 $\tau$ 过不了这些细指标，它只是粗旋钮。本课验收不要求局部性，只要求：目标掉、对照掉得少。

类比失效处：退回预训练方向，不等于「没学过」。残留的相关能力、提示词下的迂回问法，仍可能漏出来。这正是粗遗忘的「粗」。

Ilharco 在图像侧还试过 OCR、人脸识别这类「想让模型不会」的能力，附录里负向量同样能把目标准确率打下去，对照 ImageNet 保住。课内没有 OCR 数据，不要重做那张表；你的两个指令任务里，把其中一个当「想忘的」、另一个当「想留的」即可。若两个任务语义太近（都是情感分类），对照也会跟着掉，这不能怪负向量失败，要先看 $\cos(\tau_A,\tau_B)$。
### 5.3 TIES：修剪、选符号、只合并同号

直接平均有两种干扰。一是冗余：微调时很多坐标动了一点点，对损失几乎没贡献，平均时却会把别的任务上真正重要的坐标往 0 拉。Yadav et al. 在 IA3 这种参数高效微调上看到，只保留幅度最大的 20%，任务准确率几乎不掉。二是符号冲突：同一坐标上任务 A 要正、任务 B 要负，平均接近 0，两个任务都受伤。冲突比例随合并个数上升，两个模型就会出现，十一个模型时更严重。

TIES（TrIm, Elect Sign & Merge）三步：

1. 修剪。对每个 $\tau_t$ 只留幅度最大的 $k\%$（论文无验证集配方用 20%），其余置 0，等价于把那些坐标退回预训练。
2. 选符号。对每个坐标 $p$，把各任务修剪后的值相加，符号取这个和的符号：$\gamma_m^p=\mathrm{sgn}(\sum_t \hat\tau_t^p)$。质量大的一侧赢。
3. 只合并同号。坐标 $p$ 上只平均那些符号已经等于 $\gamma_m^p$ 的任务，零不进分母。最后 $\theta_m=\theta_{\mathrm{init}}+\lambda\tau_m$。

浏览器实验把第 2、3 步缩到二维：两个箭头一、三象限对打，TIES 选出质量大的那一侧，另一侧丢掉，合成点不会落在原点附近。普通平均会。

二维算例，和浏览器对答案。预训练在原点。$\tau_A=(2,1)$，$\tau_B=(-1,2)$。逐坐标看：

| 坐标 | $\tau_A$ | $\tau_B$ | 平均 | TIES（不修剪） |
|---|---|---|---|---|
| x | $2$ | $-1$ | $0.5$ | $2$（正侧质量和 $2>1$） |
| y | $1$ | $2$ | $1.5$ | $1.5$（同号，两边都留） |

加法落在第一象限偏上；TIES 的 x 不会被负向的 $B$ 拉回 0.5，而是留在 2。若再把 $\tau_B$ 改成 $(-3,2)$，x 的负侧质量和变成 3，TIES 会改选负号，合成 x 为 $-3$。预测时先比两侧质量和，再决定象限。
Ablation（论文 Table 12）：去掉缩放、或把置零的坐标重新填进平均（不再 disjoint），分数掉得最明显。课内不必重做整张表，CPU 实验会断言：符号冲突的坐标上，TIES 的合成幅度大于简单平均。

mergekit 里这个方法叫 `ties`，必须提供 `base_model`。每模型可设 `weight` 和 `density`（保留比例，对应 $k\%$）。官方 `examples/ties.yml` 还演示了按层给不同 `density` / `weight` 梯度。

### 5.4 DARE：丢掉增量，留下的要放大

Yu et al.（arXiv:2311.03099）观察：SFT 相对预训练的增量绝对值常常小于 0.002，极度冗余。DARE（Drop And REscale）对增量做伯努利丢弃，留下的除以 $1-p$：

$$
m\sim\mathrm{Bernoulli}(p),\quad \tilde\delta=(1-m)\odot\delta,\quad \hat\delta=\tilde\delta/(1-p)
$$

再 $\theta_{\mathrm{DARE}}=\theta_{\mathrm{PRE}}+\hat\delta$。对线性层，这个缩放让嵌入的期望等于没丢之前，所以单模型上可以丢掉 90% 甚至 99% 增量而成绩几乎不动。模型越大，能承受的 $p$ 越高。丢掉微调后的全量权重（预训练加增量）则会崩：7B/13B 上丢 10% 微调权重，GSM8K / HumanEval 可以掉到接近 0。结论：SFT 主要是在解锁预训练里已有的能力，增量本身很稀。

合并时先对每个专家做 DARE，再交给 Task Arithmetic 或 TIES，减少专家之间的坐标打架。mergekit 提供 `dare_linear`（只做随机丢弃加缩放）和 `dare_ties`（丢弃之后再走 TIES 的符号投票）。`density` 是保留比例，约等于 $1-p$。`dare_linear` 默认 `rescale: true`，关掉缩放就会变成论文消融里的 DropOnly，嵌入余弦掉、成绩掉。

DARE 的适用边界：增量必须小。WizardCoder-Python-13B 若错误地相对 Llama-2-13b 而不是 CodeLlama 取差，增量会到 0.01 以上，DARE 失效。课内只合并「同一基座、只做了 SFT」的模型；中间若有继续预训练，先别上 DARE。

### 5.5 合并不是在线持续学习

持续学习要的循环是：新经验进来，决定写到哪里、怎么写，写完立刻测新旧，长期还要测还能不能学。合并的输入是已经冻住的几份 $\theta_{\mathrm{ft}}$，输出是另一份冻住的 $\theta_m$。没有新经验的在线通道，也没有可塑性监测。它解决的是部署侧的库存问题：专家已经存在，推理只想跑一张网，数据还拿不到。

和前面课的对照：

| 做法 | 何时写 | 要不要旧数据 | 推理成本 | 还能否继续学 |
|---|---|---|---|---|
| 顺序 LoRA / O-LoRA | 每个新任务到来时 | O-LoRA 不要 | 一份权重 | 可以再挂新 $A$，受正交或容量限制 |
| 混训上限 | 所有任务同时 | 要全部数据 | 一份权重 | 任务集变了要重训 |
| 外挂记忆（第 13 课） | 写入日记时 | 日记本身 | 检索延迟 | 能写新事实，写不进新技能 |
| 模型合并 | 所有专家训完之后 | 不要 | 一份权重 | 默认不再训；当新初始化再微调是另一件事 |

TIES 论文做过「先合并再当初始化去微调新任务」，那是把缝合结果送回训练循环，已经不是纯合并。课内主线停在纯合并。书面判断就写这一段的结论：合并是事后缝合，不是在线持续学习，但它能在不碰旧数据时回收多任务能力。

Wortsman et al. 的 model soup（arXiv:2203.05482）是同一任务、不同超参的权重平均，目标是提高单任务准确率、不增加推理。mergekit 的 `linear` 就是这类平均，不需要 `base_model`。它和跨任务的任务算术不要混：soup 是同一道菜多煮几次取汤，任务算术是几道菜倒进同一口锅。同一任务的 soup 几乎总是安全的，因为轨迹共享、符号冲突少；跨任务算术才会把 TIES 和 DARE 请出来。你若只有两个不同随机种子的同一任务检查点，先 `linear`，不要一上来就 `dare_ties`。
## 6. 源码导读

mergekit 的入口是 YAML 加 `mergekit-yaml`。算法细节在 `docs/merge_methods.md`，示例在 `examples/`。读代码前先把这张表和 README 的 Method Overview 对上。

| 路径或命令 | 带着什么问题读 |
|---|---|
| README「Usage」 | 主命令是 `mergekit-yaml config.yml ./output-model-directory`，可选 `--cuda`、`--lazy-unpickle` |
| README「Installation」 | `pip install -e .`；pip 过旧会报没有 `setup.py` |
| `docs/merge_methods.md` 的 Task Arithmetic / TIES / DARE 三节 | `base_model` 是否必须、`weight` / `density` / `lambda` / `rescale` 含义 |
| `examples/linear.yml` | `merge_method: linear`，无 `base_model`，只有 `weight` |
| `examples/ties.yml` | `merge_method: ties`，有 `base_model`，`density` 可以是标量、列表（层梯度）或按 `filter` 分支 |
| `mergekit-pytorch` | 对 `.pt` / `.safetensors` 做同一套算法，没有层切片和 tokenizer，适合 CPU 小张量 |
| `mergekit-extract-lora` | 从全量微调里抽出 PEFT 兼容的低秩近似，本课不加分可不做 |

`ties.yml` 里出现的模型名（Orca Mini、Platypus2、WizardMath、Llama-2-13B）是官方演示，13B 级。课内不要照抄这些 Hugging Face ID 当作业模型，只抄 YAML 结构：`models` 列表、`parameters.density`、`merge_method`、`base_model`、`dtype`。

三种方法在 YAML 里最小可运行结构如下。路径换成你的基座和两个微调目录。`linear` 没有任务向量，不写 `base_model`；后两种必须写。

线性平均（model soup 风格，官方 `examples/linear.yml` 同结构）：

```yaml
models:
  - model: ./models/task_a
    parameters:
      weight: 1.0
  - model: ./models/task_b
    parameters:
      weight: 1.0
merge_method: linear
dtype: float16
```

任务算术：

```yaml
models:
  - model: ./models/task_a
    parameters:
      weight: 1.0
  - model: ./models/task_b
    parameters:
      weight: 1.0
merge_method: task_arithmetic
base_model: ./models/base
parameters:
  lambda: 1.0
dtype: float16
```

TIES 与 DARE-TIES 把 `merge_method` 换成 `ties` 或 `dare_ties`，并给每个模型加 `density`（保留比例，0.5 表示丢掉约一半增量）。`dare_linear` 另有全局 `rescale`，默认 true。

负任务向量没有单独的 `merge_method`。做法：任务算术里只放一个专家，把该模型的 `weight` 写成负数，或把 `lambda` 写成负的。CPU 实验里直接算 $\theta_{\mathrm{pre}}-\lambda\tau$，不必绕 YAML。

tokenizer：现代字段是 `tokenizer.source: union` 或 `base`。两个指令模型若特殊 token 不同，合并后对话模板会乱。课内小模型若共用一个 tokenizer，写 `source: base` 最省事。`chat_template: auto` 会选输入模型里最常见的模板；两个模板打平则行为依赖实现，课内写死成基座模板更可复现。

dtype 主线用 `float16` 或 `bfloat16`，与基座一致。CPU 上若没有 bf16，用 float16 或 float32。官方示例是 float16。不要在 YAML 里混用不同 dtype 的源模型而不写 `dtype`，mergekit 会按配置转换，转换误差可能被你当成「TIES 无效」。
## 7. 实验

浏览器证象限，CPU 证公式，mergekit 证工具链。书面判断写在 Step 5，和数字一起交。

### Step 0: 浏览器「任务向量加法」

打开本课页面交互实验。平面上有预训练原点、任务 A 的向量、任务 B 的向量。先预测再运行：

- A、B 都在第一象限且夹角很小：相加之后更该靠近哪一根轴，还是两者之间？
- A 在第一象限、B 在第三象限（符号冲突）：普通加法会靠近原点吗？TIES 修剪后还会吗？
- 对 A 取负：点会落到第几象限？对照任务（正交的第三根轴）该不该大幅移动？

改向量或方法会作废上次运行，需要重新预测。过关：符号冲突时能指出「平均≈抵消、TIES 保留质量大的一侧」；取负时对照轴几乎不动。

### Step 1: CPU 机制实验

在 `experiments/` 目录：

```bash
python3 run.py run 12
```

现在应当全绿：写出 `artifacts/lesson12/result.json`，六条 `checks` 全为真。

| check | 本机为真时在说什么 |
|---|---|
| `add_both_tasks_above_0_85` | 相加后两任务准确率都高于 0.85 |
| `add_beats_other_single_on_task1` | 相加在任务 1 上高于「只训任务 2」至少 0.15 |
| `add_beats_other_single_on_task2` | 相加在任务 2 上高于「只训任务 1」至少 0.15 |
| `add_beats_projection_on_task1` | 相加在任务 1 上高于把和投影到 $\tau_2$ |
| `add_beats_projection_on_task2` | 相加在任务 2 上高于把和投影到 $\tau_1$ |
| `negative_task_vector_hurts_task1` | $-\tau_1$ 让任务 1 明显下降 |

本机一次运行（Python 3.13.13，seed 12）：相加 0.988 / 0.992，单任务在另一任务上 0.433 / 0.442。换机器数字会变，方向不应变。

这一层是两个近正交的二维任务从同一原点微调，不证明 7B mergekit 排行榜。TIES / DARE 的符号冲突在浏览器和下面的 mergekit YAML 里看；课内 CPU 钉的是加法与单任务投影。不要手写一份假 JSON。

### Step 2: 安装 mergekit 并核对 README

```bash
git clone https://github.com/arcee-ai/mergekit.git
```

```bash
git -C mergekit rev-parse HEAD
```

进入仓库后按 README 安装（pip 过旧先升级 pip，那是另一次命令）：

```bash
pip install -e .
```

```bash
mergekit-yaml --help
```

确认帮助里仍是「配置文件 + 输出目录」。把 SHA 和你看到的 `merge_method` 列表抄进报告。若 README 改了命令名，以你打开的那一版为准。

mergekit 自称 out-of-core：权重可以懒加载，不必把几个 7B 同时装进内存。课内小模型感觉不到这一点；若你加分去碰 7B，先读 README 的 Features 里 lazy loading 那几条，再决定要不要 `--cuda`。`--allow-crimes` 是仓库给危险组合留的开关，主线不要开。
### Step 3: 三份合并对照

准备同一基座上的两个微调：优先第 10 课的两份指令 LoRA（先折回基座再合并，或对 adapter 做任务算术）；否则两个公开小模型，必须同架构、同 tokenizer 家族。不要拿 T5 和 GPT-2 对打。

写三份 YAML，分别 `merge_method: task_arithmetic`、`ties`、`dare_ties`，`base_model` 指向基座，`density` 主线用 0.5 或 0.7。线性平均可加第四份 `linear` 作对照，但验收只要求前三种。

```bash
mergekit-yaml configs/task_arithmetic.yml ./merged/task_arithmetic
```

TIES、DARE 各跑一次，输出目录分开。CPU 足够；有 GPU 可加 `--cuda`，那是第三条命令，不要和 YAML 路径写在同一围栏。

评测：两个任务的验证集或课内小探针，加上基座零样本。表里至少六个数：两个单任务专家、三份合并、基座。同一套解码参数，不要一份用贪婪、一份用采样。若任务是分类，报准确率；若是生成，报课内固定的小题集对错，不要换一套开放生成再和专家比「手感」。

预期（方向，不是小数）：任务算术在两个任务上同时高于基座；TIES 或 DARE 在符号可能冲突时不差于任务算术。两个专家在对方任务上通常接近基座，合成模型应把这两条短板同时抬起来，这才叫回收多任务能力。若合成模型只在较强的那个任务上高、另一任务回到基座，加法没有成功，优先查 $\lambda$ 是否只偏向一侧，或两个 $\tau$ 夹角是否已经很小（几乎在抢同一方向，回到第 11 课的几何）。若 DARE 明显崩，先查增量是否其实很大（5.4 的适用边界）。
档 A 加载不了 Transformers 模型时，改用：

```bash
mergekit-pytorch configs/raw_two_tasks.yml ./merged/pytorch_out
```

`raw_two_tasks.yml` 结构与上面相同，输入是两份 `.pt`。这仍算锚定仓库实验，评测改成张量上的投影或合成分类准确率，并在报告标明「pytorch 后端，无 tokenizer」。

### Step 4: 负任务向量粗遗忘

用任务 A 的 $\tau_A$，做 $\theta_{\mathrm{neg}}=\theta_{\mathrm{pre}}-\lambda\tau_A$。$\lambda$ 从 0.5、1.0 扫。测任务 A 和任务 B。预期：A 明显下降，B 下降少。对照臂：同一幅度的随机向量；随机不该定向打掉 A。

把「目标掉多少、对照掉多少」记进表。后面写三句边界：这是整条任务轴上的粗旋钮；它不保证同义问法一起忘、也不保证无关事实不动；第 14 课的编辑和 unlearning 才量可靠性、泛化、局部性、流畅性。本课数字不得写成「已完成机器遗忘」。

### Step 5: 书面判断

单独一小节，不要混在表格脚注里。结论用这一句：合并不是在线持续学习，而是事后缝合，但它能在不碰旧数据时回收多任务能力。补两句限定：专家必须同基座；任务向量近正交时加法才稳；还能否继续学，本课没测。

建议按三段写，每段三到五句。第一段：你做了哪三份合并、负向量看到了什么。第二段：对照第 11 课的顺序训练，合并发生在什么时刻、缺了循环里的哪一步。第三段：它仍然值得留在工具箱里的理由（无旧数据、推理仍是一张网）以及不能代替第 24 课在岗学习的理由。不要把 mergekit 社区模型的宣传语写进来。
## 8. 配置与预算

| 档 | 做什么 | 时间与资源 | 算哪一档 |
|---|---|---|---|
| A | Step 0–1，可选 `mergekit-pytorch` | 分钟到一小时，CPU | 机制 + 工具链冒烟 |
| C | 1B 或 135M 两个 LoRA，三份 YAML + 评测 + 负向量 | 合并分钟级，评测看生成长度 | 实战 |
| D | 7B 公开模型按 `examples/ties.yml` 结构合并 | 磁盘和内存是瓶颈，mergekit 可 out-of-core | 加分，不对齐任何排行榜 |

官方 README 写：合并可全 CPU，8GB 显存也能加速。这与课程档 A 一致。不要为了合并去申请多卡。评测才是显存瓶颈：生成式探针要把基座和 tokenizer 加载进去。135M 级在 CPU 上也能评一小集；1B 级建议单卡。合并三份 YAML 的墙钟通常远小于评测六次前向。

超参主线：`lambda: 1.0`；TIES/DARE 的 `density: 0.5`；需要更接近专家时把 density 提到 0.7。没有验证集不要网格搜索二十组再挑最好的一组当「方法赢了」。固定配方，三份合并用同一套 $\lambda$ 和评测脚本。TIES 论文无验证集时用保留 20%（density 0.2）和 $\lambda=1$；Task Arithmetic 原文常用更小的 $\lambda$。课内为了三份 YAML 可对照，统一 $\lambda=1.0$、density 0.5。若你的合成模型两任务都弱于专家平均，先把 density 提到 0.7 再下结论，不要第一步就换算法。

第 10 课的 LoRA 若还是 adapter 文件，先确认 mergekit 吃的是完整模型目录（含 `config.json` 与 tokenizer）。只含 `adapter_model.safetensors` 时，先把 LoRA 折回基座再写 YAML，或使用仓库当前文档里针对 PEFT 的路径。文档若未提供 PEFT 直接合并，不要臆造开关。

## 9. 验收

- 浏览器：符号冲突与取负两次预测正确；能讲清 TIES 为什么不落在原点。
- CPU：`python3 run.py run 12` 现在应当全绿，`checks` 全真：`add_beats_other_single_on_task1`、`add_beats_other_single_on_task2`、`add_beats_projection_on_task1`、`add_beats_projection_on_task2`、`add_both_tasks_above_0_85`、`negative_task_vector_hurts_task1`。课内数字来自二维任务向量，不是 7B mergekit 排行榜。
- 三份合并：`task_arithmetic`、`ties`、`dare_ties` 的输出目录存在，评测表含两个任务。任务算术在两任务上均高于基座（方向）。
- 负任务向量：目标任务下降，对照任务下降幅度更小；写明这是粗遗忘。
- 书面判断：含「事后缝合」「不是在线持续学习」「不碰旧数据时回收多任务能力」三点。
- 命令核实：报告里的 `mergekit-yaml` 行与你克隆的 README Usage 一致，并写 SHA。
- 诚实：13B 官方示例分数、Open LLM Leaderboard、Yu et al. 的 supermario 排名，一律不许抄进「我的结果」。

报告目录建议：`artifacts/lesson12/` 下放 YAML 副本、三个输出目录的 `config.json` 路径、评测表、负向量表、书面判断。配置抄命令和 SHA，不抄论文表格。第三幕结束时回头看第 09 到第 12 课：你会配比续训、填 4×4、量 $A$ 的夹角、把专家缝回去。缺的零件交给第四幕：事实写在外面、改一条别伤邻居、以及学不动了怎么办。
## 10. 排错

| 症状 | 原因 | 验证 | 修法 |
|---|---|---|---|
| `add_both_tasks_above_0_85` 为假 | 相加后至少一任务没过 0.85 | `acc_add_task1` / `acc_add_task2` | 看 5.2：$\theta_{\mathrm{pre}}+\tau_1+\tau_2$；两任务是否近正交（看 `task_vector_cosine`） |
| `add_beats_projection_on_task1` 为假 | 加法不优于单任务投影 | 对比 `acc_add_task1` 与 `acc_proj_tau2_on_task1` | 投影到 $\tau_2$ 会丢掉任务 1 分量；加法应同时保留两轴 |
| `negative_task_vector_hurts_task1` 为假 | $-\tau_1$ 没打掉任务 1 | `acc_neg_tau1_on_task1` | 看 5.2 的减法：$\theta_{\mathrm{pre}}-\tau_1$ 应沿任务 1 往回走 |
| `File "setup.py" or "setup.cfg" not found` | pip 过旧 | `pip --version` | `python3 -m pip install --upgrade pip` 后再 `pip install -e .` |
| YAML 报需要 `base_model` | `ties` / `dare_*` / `task_arithmetic` 没写基座 | 对照 `docs/merge_methods.md` 的 Base Model 列 | 补上 `base_model`；`linear` 才可以不写 |
| 合并后乱码或拒答 | tokenizer 特殊 token 被平均坏了 | 看输出目录的 `tokenizer.json` 与 `chat_template` | `tokenizer.source: base`，不要默认 `union` |
| DARE 成绩崩掉 | 增量太大，或对全量权重做了丢弃 | 抽查增量绝对值的分位数 | 确认两模型相对同一预训练；`rescale` 保持 true |
| TIES 和平均几乎一样 | 没有符号冲突，或 `density=1` 没修剪 | 统计 $\tau_A$ 与 $\tau_B$ 同号比例 | 换更不相似的两个任务，或把 density 降到 0.5 |
| 负向量把对照任务一并打掉 | $\lambda$ 过大，或两个任务向量不正交 | 看 $\cos(\tau_A,\tau_B)$ | 减小 $\lambda$；对照应选更不相关的探针 |
| 13B 示例 OOM | 照抄 `examples/ties.yml` 的模型 ID | 目录体积 | 换课内小模型；mergekit 再小也要能把单个张量装进内存 |
| LoRA 目录不被识别 | 只有 adapter 没有完整 `config.json` | `ls` 模型目录 | 先折回基座，或查当前 mergekit 文档是否支持 PEFT 输入 |

## 11. 前沿与改造

合并从 2022 年的 soup 与任务算术，走到 2023 年的 TIES / DARE，再走到 mergekit 里一长串变体：DELLA（按行幅度自适应修剪）、breadcrumbs（去掉最大和最小的差）、SCE、Model Stock、以及面向 RL 智能体的 RAM。社区把 YAML 当乐高，排行榜上的「新模型」有相当一部分是缝出来的。持续学习这边，MagMax（Marczak et al., [arXiv:2407.06322](https://arxiv.org/abs/2407.06322)）先顺序微调再按幅度最大选坐标；Merge to Learn（Morrison et al., [arXiv:2410.12937](https://arxiv.org/abs/2410.12937)）把新技能单独训完再并回通用模型；K-Merge（Shenaj et al., [arXiv:2510.13537](https://arxiv.org/abs/2510.13537)）在设备存储预算内在线并 adapter。前沿差在两头：一头是理论（何时权重可线性连、Ortiz-Jimenez 的切空间任务算术）；一头是持续学习（缝完之后怎么接着在线写，而不把缝合结果重新冻住）。

我们差在：主线只有两个专家、没有 Fisher / RegMean、没有验证集上的 $\lambda$ 扫描曲线，也没有把第 11 课正交后的 LoRA 再送进 mergekit 看符号冲突是否变少。

动手改造：

1. 同一对专家，扫 `density ∈ {0.2,0.5,0.8}` × `lambda ∈ {0.4,1.0}`，画两任务平均分。预算：每次合并 CPU 分钟级，评测看模型。预期：过稀（0.2）两任务都弱，过密接近普通加法。失败：只报最好一组当 TIES「赢了」。
2. 把第 11 课的两份 $A$ 折成 $\Delta W=AB$（注意字母约定），当任务向量做加法，对照未经正交的 IncLoRA 两份。预期：正交后的余弦更低，加法更稳。失败：没折 $\Delta W$ 就把 $A$ 和 $B$ 分别平均。
3. 负向量之后，用 [第 03 课](03_cl_evaluation.md) 的 BWT 语言写一句：这是人为制造的反向迁移，不是学习过程中的遗忘。不要把它填进第 11 课复现 #3 的表格。
4. 读 mergekit 的 DELLA 或 breadcrumbs 一节，用 TIES 三步的词汇解释它多做了哪一次修剪。不要求跑。

顺手复现映射：本课不是 CURRICULUM §4 的五项正式复现。数字只用于机制和实战对照。

## 12. 论文与延伸

1. Yadav et al., 2023, *TIES-Merging: Resolving Interference When Merging Models*, [arXiv:2306.01708](https://arxiv.org/abs/2306.01708)。
贡献：修剪小变化、投票选符号、只平均同号坐标。机制发明处，不是本课主阅读。
机制：先把每个任务向量里幅度小的坐标置零，再按各任务修剪后的和选符号，最后只合并与该符号一致的值。改的是合并规则，不回放、不再训练。无验证集时论文用留最大的 20%、$\lambda=1$。
和本课：浏览器符号冲突、`ties` YAML。CPU 的 `add_both_tasks_above_0_85` 是直接相加，两个合成任务近正交（`task_vector_cosine` 约 $-0.05$），符号冲突弱，答不了「冲突坐标上 TIES 幅度大于简单平均」。
阅读问题：主线 `density: 0.5` 比论文无验证集的 20% 更狠还是更松？用 YAML 回答修剪比例。冲突坐标上的幅度差要看浏览器或自己数符号。CPU 六条 check 答不了这一句，因为 `task_vector_cosine` 已接近 0、符号冲突弱。

2. Yu et al., 2023/2024, *Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch*, [arXiv:2311.03099](https://arxiv.org/abs/2311.03099)。
贡献：随机丢掉增量再按 $1/(1-p)$ 放大，然后插进已有合并。机制发明处，不是本课主阅读。
机制：DARE 只丢 $\delta=\theta_{\mathrm{ft}}-\theta_{\mathrm{pre}}$，留下的除以 $1-p$。摘要写 SFT 增量绝对值常在 0.002 以内，可丢掉 90% 甚至 99% 仍近似原嵌入。丢微调后的全量权重会崩。改的是合并前的稀疏化。
和本课：`dare_ties` YAML。CPU 没有伯努利丢弃，`add_beats_projection_on_task1` 只比较加法与单任务投影。丢全量会崩、丢增量还在，本课实验答不了，因为没有对全量权重做 dropout 的臂。
阅读问题：你的 `dare_ties` 有没有把 `rescale` 关掉？关掉就变成论文消融里的 DropOnly。CPU 答不了 $p=0.9$ 时嵌入是否还在，标「要看 mergekit 评测表」。

3. Marczak et al., 2024, *MagMax: Leveraging Model Merging for Seamless Continual Learning*, [arXiv:2407.06322](https://arxiv.org/abs/2407.06322)。
贡献：顺序微调得到任务向量，再按幅度最大选坐标并回预训练，把合并当成任务训完之后的整合。
机制：每个任务结束后 $\tau_i=\theta_i-\theta_0$，合并时逐坐标保留幅度最大的那个。顺序微调用来减少符号冲突；合并发生在训练循环外面。HTML 写 running statistics 只需存两套权重。改的是「何时合并、按什么选坐标」，训练损失本身不变。
和本课：CPU 是两个任务从同一原点独立微调再相加，接近独立微调后的任务算术，不是顺序微调。`task_vector_cosine` 接近 0 来自输入轴正交，答不了「顺序微调是否减少符号冲突」。最大幅度选择本课实验答不了，因为没有按幅度掩码的臂。
阅读问题：若把本课两个专家改成「先训任务 1，再从 $\theta_1$ 训任务 2」，`add_beats_other_single_on_task2` 还该不该成立？本课实验答不了，因为当前脚本两臂都从 `base_w` 出发。

4. Alexandrov et al., 2024, *Mitigating Catastrophic Forgetting in Language Transfer via Model Merging*, [arXiv:2407.08699](https://arxiv.org/abs/2407.08699)。
贡献：Branch-and-Merge 把数据切成子集分别微调再迭代合并，用更小幅度、更高质量的权重变化做语言迁移。
机制：每次只在一部分目标语言数据上微调，得到的更新并回当前模型，再开下一枝。摘要写在保加利亚语和德语上，相对继续预训练和指令微调，源域忘得少、目标域不差。改的是「一小枝数据 + 合并」，不是一次看完全部新语料。
和本课：书面判断里的事后缝合。CPU 两个任务没有「源语言 vs 目标语言」，`negative_task_vector_hurts_task1` 是整条轴取负，答不了「源域能力还在多少」。
阅读问题：BaM 的「更小幅度更新」和本课负向量是同一根旋钮吗？用 5.2 的 $\lambda$ 回答幅度；语言迁移分数本课实验答不了，因为没有保加利亚语或德语探针。

5. Morrison et al., 2024, *Merge to Learn: Efficiently Adding Skills to Language Models with Model Merging*, [arXiv:2410.12937](https://arxiv.org/abs/2410.12937)。
贡献：新技能单独训完，再用任务向量并回通用模型，避免在更新后的混料上重训。
机制：并行训隔离技能，合并时用任务向量加回通用权重。摘要写科学文献、安全、代码三类；并行训再合并在安全上，比继续微调更能既听话又拒危险提示。改的是数据配方和合并时机，推理仍是一张网。
和本课：CPU 正是「两任务从同一原点各训一份，再相加」。`add_beats_other_single_on_task1` / `on_task2` 对应「合成模型同时抬起两个短板」。论文的安全拒答本课实验答不了，因为没有安全提示集。
阅读问题：你的相加在两个任务上都高于「只用另一个专家」至少 0.15 了吗？看 `add_beats_other_single_*`。若其中一个没过，是 $\tau$ 不正交还是 $\lambda=1$ 过猛？`task_vector_cosine` 能答前半句。

6. Yang et al., 2024, *Model Merging in LLMs, MLLMs, and Beyond: Methods, Theories, Applications and Opportunities*, [arXiv:2408.07666](https://arxiv.org/abs/2408.07666)。
贡献：给合并方法做分类地图，并把持续学习单列为应用场景。
机制：不改你的损失或 YAML，改的是归类：权重插值、稀疏修剪、任务向量、顺序合并各算哪一族。摘要强调不收集原始训练数据、不再付一次训练算力。
和本课：三份 `merge_method` 可对上地图里的任务算术 / TIES / DARE。综述里的理论可合并条件、多模态表，本课实验答不了，因为 CPU 只有二维合成任务。
阅读问题：把本课 `task_arithmetic`、`ties`、`dare_ties` 三份 YAML 各归进综述的哪一类？书面判断里「事后缝合」该放到持续学习那一节还是多任务那一节？用 5.5 的表回答后一句。

7. Tang et al., 2025, *Merging Models on the Fly Without Retraining: A Sequential Approach to Scalable Continual Model Merging*, [arXiv:2501.09522](https://arxiv.org/abs/2501.09522)。
贡献：模型一个接一个到，用正交投影加自适应缩放做顺序合并，内存不随模型数涨。
机制：把新任务向量投到已合并更新的正交补，再用缩放稳住参数距离。无训练、常数内存。摘要写 CLIP-ViT 上平均准确率高 5 到 8 个点，并对任务顺序更稳。
和本课：CPU 一次加上 $\tau_1+\tau_2$，没有「先并一个再投第二个」。`add_beats_projection_on_task1` 是把和投影到单一 $\tau$，方向相反：论文要的是新 $\tau$ 对已并空间的正交补。顺序投影本课实验答不了，因为没有这条臂。
阅读问题：若先得到 $\theta_{\mathrm{pre}}+\tau_1$，再把 $\tau_2$ 投到 $\tau_1$ 的正交补后加上去，任务 2 的准确率该接近 `acc_add_task2` 还是接近 `acc_proj_tau1_on_task2`？本课实验答不了，除非你改 `lesson_12.py`。

8. Qiu, Zhang, Qiao & Nie, 2025, *Train with Perturbation, Infer after Merging: A Two-Stage Framework for Continual Learning*, [arXiv:2505.22389](https://arxiv.org/abs/2505.22389)。
贡献：每任务训完后把新模型与旧模型做凸组合；训练时沿任务向量加扰动，逼近合并带来的二阶损失。
机制：合并系数在总损失增量最小的目标下有闭式解。扰动用任务向量方向上的二阶对称差分近似 Hessian 项，不额外做一次完整前后向。再和 LoRA 结合以降内存。改的是训练目标（加扰动）和推理权重（用合并结果）。
和本课：CPU 固定 $\lambda=1$ 相加，没有扰动、没有闭式系数。`add_both_tasks_above_0_85` 只说明这一次加法够用。论文的 Hessian 正则本课实验答不了，因为没有沿 $\tau$ 的有限差分。
阅读问题：本课 $\lambda=1$ 的加法是不是他们闭式系数的一个特例？用 5.1 的 $\theta_{\mathrm{pre}}+\lambda(\tau_1+\tau_2)$ 对照凸组合。系数是否最优，本课实验答不了，因为没有扫验证集损失。

9. Sokar et al., 2025, *Continual Learning in Vision-Language Models via Aligned Model Merging*, [arXiv:2506.03189](https://arxiv.org/abs/2506.03189)。
贡献：新任务训练时就把权重往旧任务对齐，合并时少打架，顺序微调不再一边倒向最近任务。
机制：在微调阶段加对齐，使新任务参数与已学参数更容易线性合；合并仍发生在任务结束后。评测在大型视觉语言模型上，看遗忘、任务顺序和任务相似度。改的是训练约束，让事后加法更干净。
和本课：CPU 两任务独立 SGD，没有对齐项。`task_vector_cosine` 已经接近 0，加法好做；若两个 $\tau$ 很对齐，论文要的训练期对齐还没有发生。对齐损失本课实验答不了，因为训练脚本没有该项。
阅读问题：第 11 课把两个 $A$ 拉正交，本课把两个 $\tau$ 相加。若正交已经让 $\cos(\tau_1,\tau_2)$ 接近 0，还需要这篇的训练期对齐吗？用本课 `task_vector_cosine` 回答几何前提；VLM 顺序表本课实验答不了。

10. Qiu, Xu, He et al., 2025, *MINGLE: Mixture of Null-Space Gated Low-Rank Experts for Test-Time Continual Model Merging*, [arXiv:2505.11883](https://arxiv.org/abs/2505.11883)。
贡献：测试时用少量无标签样本做持续合并；低秩专家加零空间门，限制门控别激活旧任务。
机制：定义测试时持续模型合并。专家是 LoRA；门控更新被限制在与旧任务表示正交的子空间。自适应放松按测试时看到的干扰调约束强度。摘要写平均高 7 到 9 个点。改的是推理期门控和合并，训练数据仍看不到。
和本课：mergekit 三份 YAML 都在部署前一次性缝完，没有测试时无标签适应。CPU 更没有门控。本课实验答不了零空间门，因为没有旧任务表示基，也没有测试时更新。
阅读问题：本课书面判断写「合并发生在所有专家训完之后」。MINGLE 把合并挪到测试时，还算 5.5 表里的「事后缝合」吗？用表里「何时写」那一列回答；门控分数本课实验答不了，因为没有无标签测试批。

11. Phan et al., 2025, *Toward a Holistic Approach to Continual Model Merging*, [arXiv:2509.23592](https://arxiv.org/abs/2509.23592)。
贡献：在合并前、中、后三处动手：切空间微调、用优化器状态、合并后校正表示差。
机制：合并前在切空间微调，让各任务权重更好拆开。合并中用优化器状态里的功能信息，不回访旧数据。合并后校正合并前后的表示差。常数内存。改的是微调几何和合并后的一步校正，不只改 YAML 的 `merge_method`。
和本课：CPU 与 mergekit 都没有切空间线性化，也没有优化器状态。`add_beats_projection_*` 只说明二维加法优于单轴投影。三阶段校正本课实验答不了，因为没有保存 Adam 状态。
阅读问题：本课改造清单第 4 项读 DELLA / breadcrumbs，对应这篇的「合并中」还是「合并后」？用 TIES 三步的词汇回答修剪发生在哪一步。切空间微调本课实验答不了，因为没有线性化臂。

12. Shenaj et al., 2025, *K-Merge: Online Continual Merging of Adapters for On-device Large Language Models*, [arXiv:2510.13537](https://arxiv.org/abs/2510.13537)。
贡献：设备上 LoRA 一个接一个到达，在存储预算内无数据地选哪些留下、怎么并。
机制：假设设备只能存有限个 adapter。新 LoRA 到来时，按计算预算选择并合并，保住已支持任务。无训练数据、面向端侧。改的是在线库存：哪一份 adapter 被留下、哪一份被并掉。
和本课：主线只有两个专家、没有 $K$ 槽淘汰。`add_both_tasks_above_0_85` 说明两份可以同时留下。第三份到来时谁被并掉，本课实验答不了，因为没有存储预算参数。
阅读问题：若 $K=1$，你只能留一份合并结果，这还是 5.5 表里「推理仍是一张网」吗？是。旧任务会不会在第三次合并后掉到基座以下，本课实验答不了，因为只有两个 $\tau$。

第三幕到此结束。系统现在会：续预训练时用数据配比保护通用能力，顺序指令时用 4×4 量遗忘，用正交 LoRA 减少方向互抢，用合并在专家已经存在时回收多任务能力。下一幕换地方写：日记放到模型外面、改一条事实、以及学着学着学不动。带着本课那句书面判断去 [第 13 课](13_external_memory.md)：外挂记忆也是「不改慢速权重」的写法，和合并一样不是梁文峰说的在岗学习本身，但会成为第六幕 Agent 的零件。第四幕开始后，你会反复用到本课的分档：能缝就缝、该写日记就写日记、必须改权重时再动 LoRA 或编辑。现在先把三份合并和书面判断交出去。
