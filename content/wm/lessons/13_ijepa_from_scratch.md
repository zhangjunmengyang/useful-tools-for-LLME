---
id: 13_ijepa_from_scratch
title: "I-JEPA 预测表征"
summary: "不重建图像，模型为什么不会摆烂输出常数（坍缩）？EMA target 和 masking 设计各挡住哪条坍缩路径？"
unit: jepa
play_tools: []
checkpoints:
  - "小规模 I-JEPA 复现报告（论文复现 #2）：探针精度加坍缩消融。"
  - "官方 ijepa 仓库 masking/EMA 参考实现的精读笔记。"
---

# 第 13 课：I-JEPA 预测表征，不预测像素

> 类型：复现（论文复现 #2：I-JEPA 小规模复现 + 官方代码精读）<br>
> 建议周期：3-5 天<br>
> 硬件：CPU/MPS 全程可跑（100 epoch 在 M 系 Mac 约 1 小时），CUDA 单卡更快，全课程对 Mac 最友好的复现课<br>
> 锚定仓库：[keon/jepa](https://github.com/keon/jepa)（单文件教学复现，训练）+ [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa)（官方实现，只读精读，不复现训练）<br>
> 产物：CIFAR-10 上从零训练的 I-JEPA、线性探针成绩、坍缩审计报告（EMA 消融 + masking 消融）

## 1. 这一课做什么

VAE、RSSM、Dreamer、IRIS 和 DIAMOND 都要求模型重建或生成观察。第 02 课已经出现一个
重要反例：重建损失主要受草地纹理影响，而弯道探针的排名并不一致。生成得更像，不保证
表示中保留了下游真正需要的信息；同时，像素质量和生成时长都会增加计算成本。

第 07、08 课你已经见过两次"不重建"：MuZero 的隐状态只对价值、策略、奖励负责，TD-MPC2 靠一致性损失加奖励头和 Q 头把潜空间撑住。但它们有共同的靠山，任务信号。奖励和价值是环境发的外部工资，隐状态想摆烂，工资单第一个不答应。现在把靠山抽掉：没有奖励，没有标签，没有重建，只有一堆静态图片，模型还能学到东西吗？

I-JEPA（Image-based Joint-Embedding Predictive Architecture）提供了另一种目标：把
图像切成小块，遮住其中几块，让模型根据可见部分预测被遮区域的**表征**，而不是像素。
它学习的是不同区域在抽象空间里的关系，不需要画出缺失内容。

问题在于预测目标也由模型生成。若编码器把所有图都映射成同一个常数，预测器可以得到
很低的损失，却没有学到任何信息，这是坍缩。重建目标不能用一个常数还原所有图片，
而 JEPA 必须额外约束这种平凡解。接下来会分别移除 EMA target encoder 和 multi-block
masking，通过表征方差和线性探针检查是否发生坍缩。

LeCun 在 2022 年的《A Path Towards Autonomous Machine Intelligence》中给出的论点是：
世界里有许多难以预测、又与任务无关的细节，模型不必为下一帧每片树叶的位置分配容量；
预测应在更抽象的表征空间中进行。I-JEPA 是这一思路在图像上的实现之一。本课先检验
机制，不预先接受它对生成路线的价值判断。

复现报告会同时给出健康配置和消融配置的表征方差曲线，以及 CIFAR-10 线性探针结果。
目标是判断结果是否与锚定仓库报告的 52.7% 同方向，并如实保留与监督基线的差距。第
14 课将在此基础上加入时间和动作，第 15 课再使用 V-JEPA 2 的公开权重。

术语速查：

| 术语 | 一句人话 |
|---|---|
| JEPA / 联合嵌入预测架构 | 两个编码器各看一部分输入，一个预测器在表征空间里把两边接上；从头到尾不生成像素 |
| 坍缩（collapse） | 模型发现的作弊通道：把一切输入映成同一个向量，预测稳赢，信息全无 |
| 平凡解 | 坍缩的数学名字：损失为零但什么都没学到的解 |
| context encoder | 干活的编码器 $f_\theta$：看露出来的图块，吃梯度更新 |
| target encoder | 出题的编码器 $\bar f$：编码被遮的图块当预测目标，不吃梯度 |
| EMA / 动量更新 | target encoder 的更新方式：不训练，只做 context encoder 权重的指数滑动平均，是个慢半拍的影子 |
| stop-gradient | 掐断梯度回流：目标那一侧只提供数字，不参与反向传播 |
| multi-block masking | I-JEPA 的出题方式：一个大块当上下文，四个中等块当预测目标，块要大到有语义 |
| predictor | 预测器 $g_\phi$：拿着上下文表征和"要预测哪个位置"的查询，输出对目标表征的猜测 |
| 线性探针 | 第 02、07 课用过的老工具：冻住编码器，只训一个线性层读标签，量"信息在不在"（参见第 02 课） |
| 表征方差 | 本课的坍缩温度计：一批图的特征逐维算标准差，趋零即坍缩，从 SimSiam 论文抄来的仪表 |

## 2. 问题

需要解释的是：模型不重建图像时，怎样避免所有输入都退化成同一个常数表征。这个问题
可以拆成四部分：

1. 坍缩到底是什么、为什么免重建目标天生带着它？这不是工程 bug，是目标函数里真实存在的全局最优解。第 5.2 节把它算给你看：常数解让损失严格为零。重建、价值等价、表征预测三条路线对这个平凡解的免疫力完全不同，这里把三者摆在一张表里对账。
2. EMA target 挡的是哪条路？target encoder 慢半拍之后，"追自己"变成"追一个滞后的影子"，坍缩不再是优化的不动点。但要诚实：常数解在损失景观里并没有消失，EMA 改变的是梯度下降滑不滑得进去，这是实验事实加直觉论证，不是定理，学界至今没有公认的完整解释（BYOL 靠它防住了，SimSiam 又证明不用它也行）。你的消融实验 (a) 会亲自下场检验。
3. masking 挡的是哪条路？出题方式决定学什么。目标块够大，答题就得靠语义（"那个位置应该是狗头"）；目标换成随机撒的小碎块，答题退化成局部纹理插值。消融实验 (b) 把 multi-block 换成随机散点，看探针掉多少。
4. "没坍缩、学到了"怎么量化？两块仪表配合：表征方差（塌没塌）+ 线性探针（有没有货）。缺一不可，方差健康但探针拉胯、探针没崩但方差趋零，是两种不同的病。

诚实分档先说清：**keon/jepa 上的 CIFAR-10 训练属于复现档**，从零训练、结果与"表征可用"同方向，但规模是 ViT-tiny 加 3.2 万像素小图，与论文的 ViT-H/14 加 ImageNet 差着几个数量级，我们对照的是机制和方向，不是榜单数字。**官方 facebookresearch/ijepa 属于只读精读**：仓库 2024 年 8 月 1 日已归档，README 写明 ViT-H/14 配置要在 16 张 A100 80G 上跑出 2048 的有效 batch size，论文摘要的原话是"16 块 A100 上 72 小时以内"训完 ViT-Huge/14。这个训练我们不复现，但它的 masking 和 EMA 参考实现值得逐行读，第 6 节给对照表。

## 3. 准备

- 手艺依赖，不是产物依赖。这里不用前面训出的任何模型，单独可跑。但三样手艺直接上场：第 02 课的线性探针（本课的验收主尺）、第 07/08 课建立的"免重建靠什么撑住潜空间"参照系（这里把第三种答案补齐）、第 01 课的留证据习惯（三组消融对比，没有 NOTES.md 会乱成一锅粥）。
- 硬件门槛全课程最低之一。锚定仓库 README 原话："Runs on CUDA, MPS, or CPU."。教程作者的 100 epoch 结果就是在一台 M 系 Mac 上约 1 小时跑出来的。第 05、06、08 课因为没有 NVIDIA 卡只能围观训练的 Mac 用户，这课和第 07 课一样，一步不缺地跟。
- 环境要求很新但很少。Python 3.10 以上（仓库在 3.13.5 上测试），依赖只有 torch、torchvision、matplotlib、scikit-learn、numpy、pillow 六件，`requirements.txt` 钉死了版本（torch 2.11.0）。照例开独立虚拟环境。
- 磁盘几百 MB。CIFAR-10 约 170MB 自动下载，模型是 ViT-tiny 量级，没有大 checkpoint。
- 先花 20 分钟通读锚定仓库的 `FAITHFULNESS.md`，这个文件本身就是教材：作者逐条列出每个实现保住了论文的哪些关键机制、又在哪里做了教学简化。带着"如果是我来砍，我敢砍哪刀"的问题读。

## 4. 学习目标

1. 白纸画出 I-JEPA 三件套（context encoder、target encoder、predictor）的数据流，标出梯度在哪里流、在哪里被掐断、EMA 往哪个方向拷贝权重；
2. 用一行数学写出坍缩的平凡解，并解释为什么重建目标天然免疫它、纯联合嵌入目标躲不开它；
3. 说清 EMA 防坍缩的直觉论证到哪一步为止是可靠的、从哪一步开始是未解之谜，并能引用 BYOL 和 SimSiam 各自的证据；
4. 解释 multi-block masking 的三个设计量（目标块尺度 0.15-0.2、上下文块尺度 0.85-1.0、块数 4）各自防什么退化；
5. 独立设计并跑完一次坍缩审计：方差仪表 + 探针仪表 + 消融组，并对三种可能结局（全塌、半塌、没塌）各给一句正确判读；
6. 说出 keon 复现与官方实现、官方实现与论文公式之间的已知差异各至少两处（FAITHFULNESS.md 和教程都如实列了）。

## 5. 原理

五个机制，老节奏：为什么需要（直觉）、怎么运转（机制）、精确定义（数学）、在源码哪里（代码）、怎么证明做对了（验证）。

### 5.1 把 decoder 从系统里拿掉：联合嵌入预测架构

考你"这张被遮住一角的照片，缺的那块是什么"，有两种答法。生成式答法：把缺的那块画出来，逐像素打分，MAE、以及我们第一幕的 VAE 都是这一路，代价是每根草的位置都算进考分，可每根草的位置本来就是猜不中也不必猜中的。JEPA 的答法：用一个向量概括缺的那块（"一段向右弯的路缘"），只在向量层面对答案。概括对了就得分，草的方向随便。预测的战场从像素空间挪进表征空间，不可预测的细节在编码那一步就被抛掉了。

I-JEPA 用三个零件实现这个答法。context encoder $f_\theta$（一个 ViT）只看露出来的图块，产出上下文表征；target encoder $\bar f$ 看被遮的图块，产出目标表征，它是 $f_\theta$ 的 EMA 影子，不吃梯度；predictor $g_\phi$（一个更窄的 ViT）拿着上下文表征加"要预测的位置"的查询（一个可学习的 mask token 加上该位置的位置编码），输出对目标表征的猜测。整个系统没有 decoder，任何一步都不产出像素。

一张图切成 $N$ 个 patch，采样出上下文块 $x$ 和 $M$ 个目标块 $\{y_i\}$，损失是表征空间的回归：

$$
\mathcal{L} = \frac{1}{M}\sum_{i=1}^{M} d\Big(g_\phi\big(f_\theta(x),\, q_i\big),\ \bar f(y_i)\Big)
$$

其中 $q_i$ 是第 $i$ 个目标块的位置查询，$d$ 论文里是 L2 距离，官方代码和 keon 复现都换成了 smooth L1 并对目标做了 LayerNorm（教程原话承认这是对论文公式的两处偏离，跟的是官方训练脚本，图的是稳定）。

keon/jepa 的 `ijepa.py`，165 行装下全部三件：`Encoder` 类（注释标着 `# f_theta (context encoder)`，target encoder 是它的深拷贝）、`Predictor` 类（带 `mask_token` 和自己的位置编码）、`train()` 函数里那行 `full = F.layer_norm(tgt_enc(imgs), (D,))` 就是出题过程。官方仓库里同样的分工在 `src/models/` 与 `src/train.py`。

数参数。keon 配置里 encoder 是 dim 128、depth 6，predictor 是 dim 64、depth 4，predictor 明显更窄。官方 ViT-H 配置同款不对称：encoder 1280 维，predictor 钉在 384 维、12 层。教程一句话点破：predictor 只负责预测 patch 表征，不负责提取，小一号就够。这个不对称你读代码时应该能确认。

### 5.2 坍缩：免重建目标的原罪

上面那个损失有个致命的美中不足：等号两边都是模型自己算出来的。学生答题，判卷的是另一个自己。只要两个自己串通，所有输入都编码成同一个向量 $c$，预测器永远输出 $c$，每道题都判满分。模型不是"学不会"才坍缩，是目标函数明明白白告诉它：坍缩是满分答案里最省力的那个。

把三条路线对这个作弊通道的免疫力摆在一起看。重建路线（VAE、MAE、Dreamer）：损失要求从表征解回像素，$f(x)=c$ 意味着 decoder 只能输出一张固定的图，去匹配一万张不同的图，损失巨大，常数解自动出局，防坍缩是白送的，代价是为纹理付费。价值等价路线（MuZero、TD-MPC2）：预测目标是环境发的奖励、回报、价值，不同状态的目标不同，常数表征喂不饱它们，外部信号撑住了潜空间，代价是没有任务就没有信号。表征预测路线（JEPA 系）：目标是模型自产的，两头都是软的，上面两根拐杖全没有，防坍缩必须靠架构设计硬扛。这是为什么第 07、08 课的"免重建"不需要这一课的机制，而 I-JEPA 需要。

设 $f_\theta$ 与 $\bar f$ 权重相同且同步更新（先不设防），取 $f_\theta(\cdot) \equiv c$，predictor 学恒等输出 $c$，则

$$
\mathcal{L} = \frac{1}{M}\sum_{i=1}^{M} d(c,\ c) = 0
$$

严格全局最优。更隐蔽的亲戚是**维度坍缩**：不塌成一个点，但特征挤进一个低维子空间，大部分维度形同虚设，损失照样低，信息照样丢。所以量坍缩不能只看损失，要看分布的形状：把一批图的特征做 L2 归一化后逐维算标准差，特征在球面上铺开时它约为 $1/\sqrt{D}$（keon 配置 $D=128$，参考刻度约 0.088），塌了就趋零。这块仪表直接抄自 SimSiam 论文的诊断图。

这一节的"代码"是你自己写的：第 7 节 Step 5 的 `collapse_audit.py`，核心就是上面这个逐维标准差，挂在 `train()` 的 `on_epoch_end` 回调上逐 epoch 记录。

跑完健康基线后故意作一次死：把审计脚本的 EMA 参数归零重训（Step 6），看方差曲线是不是应声往下走。能让病发作，才算真懂了病理。

### 5.3 EMA target：让"追自己"变成"追一个滞后的影子"

串通作弊需要两个自己步调一致。EMA 干的事是给判卷的那个自己套上惯性：target encoder 不做梯度更新，只按指数滑动平均缓慢跟随 context encoder，是个慢半拍的影子。现在 context encoder 想往常数解挪一步，发现判卷标准还停在过去，影子手里的题目答案仍然因图而异，摆烂拿不到满分。想串通，得两边一起挪；可影子从不主动挪，只以千分之四的速度被动跟随。坍缩这个不动点还在损失景观里，但通往它的那条"两边同时滑过去"的捷径被惯性堵住了。

每步梯度更新完 context encoder 后，执行一次无梯度的权重混合。动量 $m$ 不是常数：从 0.996 出发，随训练步数线性升到 1.0，早期影子跟得紧（目标快速变好），后期影子几乎冻结（目标稳定，收尾不抖）。同时目标侧全程 stop-gradient：梯度只流经 context encoder 和 predictor。

记 context encoder 参数为 $\theta$、target encoder 参数为 $\bar\theta$，第 $t$ 步：

$$
\bar\theta_{t+1} = m_t\,\bar\theta_t + (1 - m_t)\,\theta_t,\qquad m_t = 0.996 + (1.0 - 0.996)\cdot \frac{t}{T-1}
$$

展开看，$\bar\theta$ 是 $\theta$ 全部历史的指数加权平均，预测目标由"现在的我"变成"过去一段时间的我的平均"。

诚实边界，一句话说清再往下走。上面的论证解释了 EMA 为什么**堵住那条捷径**，没有证明梯度下降**一定不会**从别的路滑进坍缩。BYOL（EMA 防坍缩的前身，本课 EMA 机制的直接出处）在论文里也只给了实验证据加假说；一年后 SimSiam 用实验反将一军：动量编码器根本不必要，stop-gradient 加一个预测头就能防住（摘要原话说 stop-gradient "plays an essential role in preventing collapsing"）。为什么这些不对称就够、什么条件下会失效，2026 年的今天仍是有争论的研究问题。这门课的立场：直觉论证要会讲，边界要会标，然后用自己的实验说话，你的消融 (a) 恰好同时构成 BYOL 式检验（去掉动量，目标每步即时同步）和 SimSiam 式检验（剩下的正是 stop-grad + predictor），两派谁赢，你的方差曲线裁决。

keon `ijepa.py` 的 `ema_update()` 函数（三行：`pt.mul_(m).add_(po.detach(), alpha=1-m)`）和 `train()` 里的线性动量插值；stop-gradient 是双保险，target encoder 初始化时就 `requires_grad_(False)`，前向再套 `torch.no_grad()`。官方实现的对应逻辑在 `src/train.py` 的训练循环里，动量区间同样是 [0.996, 1.0]（写在 configs 的 yaml 里）。

训练日志每 50 步打印一次 `ema=` 的当前值，你应该看到它从 0.9960 单调爬向 1.0000。这个数字若一直不动，说明你改配置时把 schedule 改坏了。

### 5.4 masking 即任务设计：题目出得好，模型才学得深

同一套架构，题目不同，学出来的东西天差地别。让模型预测"随机撒的几个 4×4 小碎块"，最优策略是看邻近像素做纹理延拓，草地旁边是草地，这题用局部平滑就能答，语义一分不学。I-JEPA 的出题原则：**目标块要大到必须用语义才答得动**。遮掉四分之一张图上一整块，光靠边缘延拓补不出来，只能动用"这个位置按整图布局应该是什么"的理解。同时上下文要足够大且信息充分，否则题目无解，模型学到的只是瞎猜的方差。论文摘要把这条明说成方法的核心设计。

每张图采样 4 个目标块：面积占全图 0.15 到 0.2 倍，长宽比在 0.75 到 1.5 之间随机，中等大小、近似方形，一块就是一个语义单元。上下文块采一个大的：面积占 0.85 到 1.0 倍、长宽比 1.0，然后**把与目标块重叠的 patch 从上下文里挖掉**（官方配置 `allow_overlap: false`），不挖的话答案就印在题目里。四个目标共享同一个上下文，一次前向答四道题。

这套设计在信息上的账：目标块面积比 $s_t \in [0.15, 0.2]$ 保证单块含足够语义；上下文名义面积比 $s_c \in [0.85, 1.0]$，但挖掉四个目标块的并集后实际剩余大约只有三到五成，且形状破碎，预测既不能靠"看见全图"，也不能靠"块内插值"。keon 在 8×8 的 patch 网格上按同比例缩小：目标块约 $64 \times 0.175 \approx 11$ 个 patch。

keon `ijepa.py` 的 `sample_ijepa_masks()`（配 `_bsize` 算块尺寸、`_block` 生成矩形索引），context 减目标并集就是一行集合运算 `c = _block(...) - set().union(*ts)`；块尺寸整个 batch 共享、位置逐样本独立，是官方 mask collator 行为的简化保留。官方实现在 `src/masks/` 目录，四个数字（块数 4、目标尺度 [0.15, 0.2]、上下文尺度 [0.85, 1.0]、长宽比 [0.75, 1.5]）写在 `configs/in1k_vith14_ep300.yaml` 里，两边逐项能对上，这个对账本身就是第 6 节的精读作业。

跑 `ijepa_extras.py` 会生成 `samples/ijepa_masks.png`：原图、上下文（带洞的大块）、四个目标块并排画出来。看一眼就知道采样代码有没有被你改坏，消融 (b) 动完刀之后务必回来再画一次。

### 5.5 predictor 与线性探针：怎么证明表征里真的有货

还剩两个容易被当成配角的零件。predictor 为什么必须存在？如果让 context encoder 直接输出对目标位置的预测，那"编码"和"预测"两个职能就挤在同一个网络里，表征会被迫长成"预测起来省事"的样子。把预测职能剥给一个专职的窄网络，encoder 才能专心把内容编好，SimSiam 的消融甚至表明这个不对称的预测头是防坍缩拼图的一块。而验收端的线性探针，回答的是"学到的东西够不够格"：冻住 encoder，只训一个线性层去读类别标签。容量卡死在线性，探针读得出来，说明信息以线性可及的形式躺在表征里；读不出来，要么没学到，要么埋得太深。它不像微调，微调的大容量可以把烂表征修好，探针作不了这个弊，所以是自监督社区量表征质量的标准尺（第 02 课你用它读过路的弯度，第 07 课读过 CartPole 物理量）。

predictor 的查询构造：每个待预测位置一个 token，内容是共享的可学习 `mask_token` 加该位置的 sincos 位置编码，和上下文表征拼在一起过若干层 attention，取查询位置的输出作为预测。探针协议（keon `ijepa_extras.py` 的实现）：对 encoder 输出的 patch token 做平均池化得到图级特征，`nn.Linear(128, 10)` 读十类，AdamW、lr 1e-3、3 个 epoch、batch 512，报测试集 top-1。

`ijepa.py::Predictor`（mask_token、位置编码、窄 ViT）；`ijepa_extras.py::linear_probe`（冻结特征在 `@torch.no_grad()` 里算）。注意教程的探针对象是 **EMA target encoder**，影子比本体平均意义上更平滑，这个选择本身可以当一个对照实验（第 11 节改造清单第 4 条）。

三个刻度记住：随机猜 10%；keon 教程报告 8 epoch 约 35%、100 epoch 52.7%；同架构同预算的监督基线 70% 以上（教程原话如实报了这条差距）。你的复现落在哪个刻度之间、消融之后掉到哪个刻度，就是本课全部实验的坐标系。

## 6. 源码导读

两个仓库，两种读法。keon/jepa 是你要动刀的主战场，逐函数读；官方 ijepa 是参考答案，带着对照问题读。

主战场：keon/jepa（`ijepa.py` 165 行 + `ijepa_extras.py`）。算法文件零共享依赖，只 import torch 和 torchvision，一个下午能读完：

| 位置 | 是哪个机制 | 带着什么问题读 |
|---|---|---|
| `ijepa.py::Encoder` | 5.1 的 $f_\theta$ | patch 化在哪一步？`forward(imgs, idx)` 的 `idx` 参数怎么实现"只编码部分 patch"？|
| `ijepa.py::Predictor` | 5.5 的 $g_\phi$ | mask_token 是几个？（一个，共享）位置信息从哪进来？为什么 dim 是 64 而 encoder 是 128？|
| `ijepa.py::sample_ijepa_masks` | 5.4 的出题器 | 目标块和上下文块的尺寸为什么整个 batch 共享、位置却逐样本采？重叠是在哪一行被挖掉的？|
| `ijepa.py::_bsize` / `_block` | 5.4 的几何 | 面积比和长宽比怎么换算成 (h, w)？越界怎么夹住？|
| `ijepa.py::ema_update` | 5.3 的影子 | 为什么整个函数套 `@torch.no_grad()`？`po.detach()` 是防什么？|
| `ijepa.py::train` | 全部合体 | 找到 stop-gradient 的两道保险；找到 `F.layer_norm(tgt_enc(imgs), (D,))` 那行，对照 5.1 的"官方代码偏离论文公式"之说 |
| `ijepa.py::lr_warmup_cosine` / `param_groups` | 训练工程 | warmup 占比多少（5%）？哪些参数不做 weight decay？|
| `ijepa_extras.py::linear_probe` | 5.5 的验收尺 | 探针喂的是哪个 encoder？特征是 CLS 还是平均池化？|
| `ijepa_extras.py::save_pca` 系列 | 坍缩的目视仪表 | PCA 散点如果坍缩会长什么样？为什么 extras 还留了 `ep=-1`（随机权重）的快照？|
| `FAITHFULNESS.md` | 诚实清单 | I-JEPA 条目下"保留"和"简化"各几条？课文 5.1-5.4 的说法能不能逐条对上？|

参考答案：facebookresearch/ijepa（只读，已归档）。入口 `main.py` 读 yaml 进 `src/train.py`；`src/masks/` 是 mask collator（multi-block 的正式版）；`src/models/` 是 ViT 与 predictor；`src/helper.py` 管初始化与 checkpoint。精读方法是拿 keon 版当地图按图索骥，重点做一件事，**参数对账**：

| 设计量 | 官方 `in1k_vith14_ep300.yaml` | keon `ijepa.py` | 一致？|
|---|---|---|---|
| 目标块数（num_pred_masks） | 4 | `n_targets=4` | 一致 |
| 目标块尺度（pred_mask_scale） | [0.15, 0.2] | `uniform(0.15, 0.20)` | 一致 |
| 上下文尺度（enc_mask_scale） | [0.85, 1.0] | `uniform(0.85, 1.0)` | 一致 |
| 长宽比（aspect_ratio） | [0.75, 1.5] | `uniform(0.75, 1.5)`，上下文固定 1.0 | 一致 |
| 上下文与目标重叠（allow_overlap） | false | 集合减法挖掉 | 一致 |
| EMA 动量 | [0.996, 1.0] | `ema_start=0.996, ema_end=1.0` 线性 | 一致 |
| encoder | ViT-H/14，crop 224 | dim 128 / depth 6，图 32 | 规模简化 |
| predictor | 384 维 / 12 层 | 64 维 / 4 层 | 规模简化，窄于 encoder 的关系保留 |
| 训练 | 300 epoch，有效 batch 2048，16×A100 | 100 epoch，batch 256，一台 Mac | 规模简化 |
| 数据增强 | 仅 crop_scale [0.3, 1.0]，无色彩抖动/模糊/翻转 | `RandomResizedCrop(32, scale=(0.3, 1.0))` | 一致（这条容易被忽略：I-JEPA 不靠增强堆料） |

对完这张表你会发现：**防坍缩的机制参数一个没动，动的全是规模**。这正是"小规模复现看方向"能成立的前提。

## 7. 实验

八步。前四步把复现档跑出来，后四步是坍缩审计，重点。每步先写预期再跑。

### Step 1: 克隆与环境

```bash
git clone https://github.com/keon/jepa.git
```

进入仓库目录后建虚拟环境（要求 Python 3.10 以上）：

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

依赖钉死了版本（torch 2.11.0、torchvision 0.26.0 等），六个包装完即可。CUDA 用户若默认 wheel 不带你要的 CUDA 版本，按 PyTorch 官网装对应 build 再装其余依赖。

### Step 2: 冒烟：默认 8 个 epoch

```bash
python ijepa.py
```

首次运行自动下载 CIFAR-10 到 `./data/`（约 170MB）。预期输出：先打印 `device: mps`（或 cuda/cpu），然后每 50 步一行 `ep=.. step=.. loss=.. lr=.. ema=..`。三件事逐项核对：loss 从 0.5 上下往下走；lr 先升后降（5% warmup 加 cosine）；`ema=` 从 0.9960 缓慢爬升。MPS/单卡几分钟到十几分钟跑完。这一步只验流程，不下任何质量结论。

### Step 3: 带仪表盘重跑：可视化 + 探针

```bash
python ijepa_extras.py
```

同样是 8 个 epoch，但训练挂上了快照回调，跑完在 `samples/` 下生成：`ijepa_masks.png`（原图 / 挖了洞的上下文 / 四个目标块，5.4 节的机制画成图）、`ijepa_loss.png`、PCA/LDA/t-SNE 随 epoch 的演化网格（含 `ep=-1` 随机权重基线），最后打印 `linear probe test accuracy: ...`。预期：探针在 35% 上下（教程口径），显著高于随机 10%；PCA 散点从随机权重时的一团开始出现结构。t-SNE 每个快照要跑十几二十秒，属正常。

### Step 4: 复现档：100 epoch

`ijepa_extras.py` 的入口是 `main(epochs=8)` 函数默认参数，没有命令行开关，直接用一行 Python 调用：

```bash
python -c "import ijepa_extras; ijepa_extras.main(epochs=100)"
```

预期：M 系 Mac 约 1 小时（教程作者的原始口径就是这么跑的），CUDA 单卡更快；loss 教程口径从约 0.55 降到 0.08 附近走平；探针 50% 上下，教程报告值 52.7%，你的数字因种子和设备略有出入，落在 50±3 的带子里就算同方向。这是论文复现 #2 的主验收点：**没有像素重建、没有标签、没有奖励，表征里读得出五成的类别信息**。同时把丑话对齐：同架构监督基线 70%+，自监督小模型小数据就是有这条沟，报告里如实写。

### Step 5: 装上坍缩温度计：审计脚本 + 健康基线

从这步起进入重点。`ijepa.py` 的 `train()` 留了一个 `on_epoch_end` 回调，每个 epoch 结束把模型对象递给你（外加训练前 `epoch=-1` 的随机权重快照），这是挂仪表的正式接口。在仓库根目录新建 `collapse_audit.py`，训练主体全部来自 `ijepa.py`，这个文件只有仪表和两个消融开关，属于课程允许的胶水代码：

```python
"""collapse_audit.py -- 坍缩审计：方差仪表 + 探针仪表 + 两个消融开关。
放在 keon/jepa 仓库根目录运行。"""
import argparse, json, random
import torch
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as T
import ijepa
from ijepa import train, MEAN, STD, _bsize, _block

def loader(train_split, batch_size=512, shuffle=False):
    tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    ds = tv.datasets.CIFAR10("./data", train=train_split, download=True, transform=tf)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

@torch.no_grad()
def feature_std(enc, device, max_batches=10):
    """SimSiam 式温度计：L2 归一化特征的逐维标准差，健康参考刻度约 1/sqrt(D)。"""
    feats = []
    for i, (x, _) in enumerate(loader(False)):
        if i >= max_batches:
            break
        z = enc(x.to(device)).mean(dim=1)          # patch token 平均池化 -> (B, D)
        feats.append(F.normalize(z, dim=-1).cpu())
    return torch.cat(feats).std(dim=0).mean().item()

def probe(enc, device, epochs=3):
    """冻结 encoder 的线性探针，协议对齐 ijepa_extras.py。"""
    head = torch.nn.Linear(enc.dim, 10).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    for _ in range(epochs):
        for x, y in loader(True, shuffle=True):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                z = enc(x).mean(dim=1)
            loss = F.cross_entropy(head(z), y)
            opt.zero_grad(); loss.backward(); opt.step()
    hit = tot = 0
    with torch.no_grad():
        for x, y in loader(False):
            z = enc(x.to(device)).mean(dim=1)
            hit += (head(z).argmax(-1).cpu() == y).sum().item()
            tot += len(y)
    return hit / tot

def sample_scatter_masks(B, grid, n_targets=4, min_ctx=4, rng=None):
    """消融 (b)：目标从 4 个矩形块换成 4 组随机散点 patch，每组 patch 数
    与原版目标块面积同量；上下文采样流程保持原版不变，变量隔离在目标形状上。"""
    rng = rng or random.Random()
    N = grid * grid
    th, tw = _bsize(grid, rng.uniform(0.15, 0.20), rng.uniform(0.75, 1.5))
    ch, cw = _bsize(grid, rng.uniform(0.85, 1.0), 1.0)
    k = th * tw
    ctx_list, tgt_lists = [], [[] for _ in range(n_targets)]
    for _ in range(B):
        ts = []
        for t in range(n_targets):
            idx = set(rng.sample(range(N), k))     # 唯一的改动：散点代替矩形
            tgt_lists[t].append(sorted(idx))
            ts.append(idx)
        c = set()
        for _try in range(10):
            ct, cl = rng.randrange(grid - ch + 1), rng.randrange(grid - cw + 1)
            c = _block(grid, ct, cl, ch, cw) - set().union(*ts)
            if len(c) >= min_ctx:
                break
        ctx_list.append(sorted(c) if c else [0])
    L = min(len(c) for c in ctx_list)
    return [sorted(rng.sample(c, L)) for c in ctx_list], tgt_lists

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--ema-start", type=float, default=0.996)
    ap.add_argument("--ema-end", type=float, default=1.0)
    ap.add_argument("--masking", choices=["block", "scatter"], default="block")
    ap.add_argument("--tag", default="healthy")
    args = ap.parse_args()
    if args.masking == "scatter":
        # 打的是模块属性补丁：train() 运行时按名字查 ijepa.sample_ijepa_masks
        ijepa.sample_ijepa_masks = sample_scatter_masks
    curve = []
    def on_epoch_end(state):
        enc = state["tgt_enc"]
        s = feature_std(enc, next(enc.parameters()).device)
        curve.append({"epoch": state["epoch"], "std": s})
        print(f"[audit] epoch={state['epoch']:3d} feature_std={s:.4f}")
    out = train(epochs=args.epochs, ema_start=args.ema_start,
                ema_end=args.ema_end, on_epoch_end=on_epoch_end)
    acc = probe(out["tgt_enc"], out["device"])
    print(f"[audit] tag={args.tag} probe_acc={acc:.4f}")
    with open(f"audit_{args.tag}.json", "w") as f:
        json.dump({"tag": args.tag, "curve": curve, "probe_acc": acc}, f, indent=2)
```

先跑健康基线：

```bash
python collapse_audit.py --epochs 100 --tag healthy
```

预期：每个 epoch 一行 `[audit] epoch=.. feature_std=..`，健康组的 std 全程稳定在明显非零的水平（$1/\sqrt{128} \approx 0.088$ 是"特征在球面上完全铺开"的参考刻度，实际值在它同一数量级即算健康），收尾打印的 `probe_acc` 与 Step 4 的探针一致（都在 50% 上下）。耗时与 Step 4 相当（每 epoch 多算一次 std，秒级开销；收尾探针几分钟）。赶时间的话可以先用 `--epochs 30` 把三组曲线的趋势都看一遍（约 20 分钟一组），报告用 100 epoch 的正式版。

### Step 6: 消融 (a)：去掉 EMA，看方差塌不塌

把动量起止都归零：

```bash
python collapse_audit.py --epochs 100 --ema-start 0.0 --ema-end 0.0 --tag no_ema
```

机制上发生了什么：$m=0$ 时 `ema_update` 退化成整份拷贝，target encoder 每步与 context encoder 完全同步，"影子"消失，两个编码器实质共享权重、直接更新；但 stop-gradient 仍在（目标侧照旧不回传梯度）。这一刀同时构成两个经典检验：BYOL 视角，这是"即时目标"设置，预期训练崩坏；SimSiam 视角，剩下的恰好是"stop-grad + 预测头"组合，预期还能撑住一部分。谁对，看曲线。三种结局的判读表：

| 结局 | 现象 | 判读 |
|---|---|---|
| 全塌 | std 曲线趋零，探针跌向 10-20% | 教科书式坍缩：EMA 在此设置里是必要防线（BYOL 方向） |
| 半塌 | std 掉一截后稳住，探针明显低于健康组但远高于随机 | 维度坍缩或部分坍缩：stop-grad + predictor 兜住了底，但表征质量确实付了代价 |
| 没塌 | std 与探针都和健康组打平（探针差距 2 个点以内） | SimSiam 现象在此小规模设置成立：EMA 不必要，这同样是合格结论，如实记录 |

预算与失败判据：预算与健康组相同（MPS 约 1 小时）。**实验本身的失败**（区别于"结论与预期不同"）只有两种：loss 出现 NaN 或爆炸导致训练跑不完，把发散的 step 记下来，"无 EMA 时训练发散"本身就是有效结论；或者忘了同时改 `--ema-start` 和 `--ema-end`，动量没有真正归零（看训练日志里 `ema=` 是否恒为 0.0000）。

### Step 7: 消融 (b)：multi-block 换随机散点，看探针掉多少

```bash
python collapse_audit.py --epochs 100 --masking scatter --tag scatter
```

机制上发生了什么：目标不再是 4 个矩形语义块，而是 4 组随机撒开的 patch（每组数量与原目标块面积同量，上下文采样流程原封不动）。5.4 节的论断在此受审：如果"块大到必须用语义才答得动"成立，散点版的任务会退化成局部纹理插值，探针应该掉。预期与判读：

- 主预期：scatter 组探针低于健康组几个百分点，std 未必塌，这恰好演示两块仪表各管一摊：masking 消融打击的是语义质量（探针），不是分布形状（方差）。
- 一个反直觉现象要盯住：scatter 组的训练 loss 很可能比健康组**更低**。题目变简单，考分当然高，但探针更差。loss 降不等于表征好，第 02 课"重建好不等于压得对"的教训在免重建世界里原样重演。
- 诚实预期管理：论文的 masking 消融是在 ImageNet 的 16×16 patch 网格上做的，multi-block 显著优于随机与传统方案；我们的网格只有 8×8，"块"与"散点"的几何差异本来就小，探针差距缩水是正常的。失败判据：差距小于 1 个百分点或反超，如实记为"本设置分辨率过低，未复现出方向性差异"，并在报告里写明怀疑对象（网格太粗）。这不丢人，丢人的是把 0.5 个点的差距吹成结论。

改完刀先别急着跑 100 epoch：把 `ijepa_extras.py` 的 mask 可视化借来对散点版画一张图（或直接目视打印几个 `tgt_lists`），确认目标真的散开了、上下文真的还是带洞的大块，采样器改坏了，后面 1 小时白跑。

### Step 8: 汇总复现报告

把三组 json 叠成一张图，新建 `plot_audit.py`：

```python
"""plot_audit.py -- 三组审计曲线叠画。"""
import json, glob
import matplotlib.pyplot as plt
for fn in sorted(glob.glob("audit_*.json")):
    d = json.load(open(fn))
    xs = [p["epoch"] for p in d["curve"]]
    ys = [p["std"] for p in d["curve"]]
    plt.plot(xs, ys, label=f'{d["tag"]} (probe {d["probe_acc"]:.1%})')
plt.axhline(1 / 128 ** 0.5, ls="--", c="gray", label="uniform ref")
plt.xlabel("epoch")
plt.ylabel("per-dim std of normalized features")
plt.legend()
plt.tight_layout()
plt.savefig("audit_curves.png", dpi=150)
print("saved audit_curves.png")
```

```bash
python plot_audit.py
```

复现报告四段式：复现表（keon 报告 52.7% / 你的数字 / 监督基线 70%+ 三个刻度）；一张 `audit_curves.png` 加三行判读（每组落在 Step 6/7 判读表的哪一格）；已知差异清单（照 FAITHFULNESS.md 加第 6 节对账表如实转述）；NOTES.md 证据四件套（命令、代码改动、种子说明，注意 `ijepa.py` 未固定全局种子，跑分请注明"单种子"，有闲预算再补第二个种子验证方向稳定）。

## 8. 配置与预算

| 档位 | 内容 | 设备与耗时（参考） | 用途 |
|---|---|---|---|
| 冒烟档 | Step 2/3，默认 8 epoch | MPS/CPU 几分钟到半小时 | 验流程，探针 35% 量级 |
| 复现档 | Step 4，100 epoch | M 系 Mac 约 1 小时，CUDA 单卡更快 | 对照 52.7%，论文复现 #2 主验收 |
| 审计粗跑 | Step 5-7 三组各 30 epoch | MPS 合计约 1 小时 | 先看 std 趋势，确认消融生效 |
| 审计正式 | Step 5-7 三组各 100 epoch | MPS 合计约 3 小时，CUDA 单卡约摸减半 | 报告用数据 |

训练超参照抄 `ijepa.py::train` 默认值即可，一个不用调：batch 256、AdamW lr 3e-4（5% warmup + cosine）、weight decay 0.05、EMA 0.996 线性升 1.0。对照读官方 yaml 时注意规模差异带来的账目差异：官方 ViT-H/14 是 lr 峰值 0.001、warmup 40 个 epoch、weight decay 从 0.04 退火到 0.4、有效 batch 2048、300 epoch、16×A100 80G，这份账单就是"原配置不复现训练"的全部理由。磁盘：数据加全部产出不到 1GB。

## 9. 验收

验收清单：

- [ ] 复现档探针不低于 45%，与 keon 报告的 52.7% 同方向；冒烟档在 35% 量级，两个数字都远高于随机 10%，且都如实标注了与监督基线 70%+ 的差距；
- [ ] loss 曲线整体下降且无发散，三组消融的训练日志都留档；
- [ ] `audit_curves.png` 上三条 std 曲线叠画完成，健康组稳定在 $1/\sqrt{128}$ 同数量级；
- [ ] 消融 (a) 与 (b) 各自落进 Step 6/7 判读表的哪一格，报告里各有一句判读加一句证据；
- [ ] 指着 `samples/ijepa_masks.png` 能说清：上下文那些洞是哪行代码挖的、为什么必须挖；
- [ ] 能口头回答：为什么 scatter 组 loss 可能更低但探针更差（任务变容易，学到的东西变浅）；
- [ ] 第 6 节的参数对账表核过一遍官方 yaml，能说出"机制参数全保留、规模参数全缩小"这个结论是怎么核出来的；
- [ ] NOTES.md 四件套齐全，注明单种子还是多种子。

眼见为实的两个附加检查：翻 `ijepa_pca_evolution.png`，`ep=-1`（随机权重）到最后一个快照之间应该看得到聚类结构逐渐浮现；把消融 (a) 若塌掉的那组也画一次 PCA，散点挤成一团的样子看过一次，以后在任何自监督训练里认得出它。

## 10. 排错

| 症状 | 最可能的原因 | 怎么确认 | 怎么修 |
|---|---|---|---|
| pip 装不上钉死版本 | Python 低于 3.10 | `python --version` | 用 3.10+ 重建虚拟环境（仓库在 3.13.5 上测试） |
| 打印 `device: cpu` 但机器有 GPU | torch 装的是纯 CPU 版，或 Mac 上用了 x86 Python | `python -c "import torch; print(torch.backends.mps.is_available())"`（Mac）或查 CUDA 版本 | 按官网重装对应 build；Apple Silicon 确认用 arm64 Python |
| loss 卡在 0.5 附近不降 | lr/wd 被改动过，或 mask 采样被改坏（上下文退化成 `[0]`） | 用 extras 的 mask 可视化画一张，看上下文是不是几乎空了 | 恢复默认超参；检查自己的采样改动里 `min_ctx` 逻辑 |
| 探针恰好 10% 上下 | 特征全同（真坍缩）或探针数据忘了 Normalize | 先看 `feature_std`：趋零是坍缩；正常则查探针的 transform 是否用了 `MEAN, STD` | 坍缩组这是预期结果照实记录；探针 bug 就修 transform |
| no_ema 组 loss 掉到接近 0 且快得反常 | 坍缩红旗：题目和答案在互相靠拢 | 对照该组 std 曲线，大概率同步趋零 | 这是消融要的现象，截图存报告，别当成"训得好" |
| scatter 组与健康组结果完全一致 | 补丁没生效：`from ijepa import sample_ijepa_masks` 拿到的是旧引用 | 在 `sample_scatter_masks` 里加一行 print 看是否被调用 | 必须打模块属性 `ijepa.sample_ijepa_masks = ...`（脚本里已是这个写法，自己改写时别换成 from-import） |
| 100 epoch 探针远低于 50%（如 40% 出头） | epochs 没传进去（还是默认 8），或训练中途换了设备 | 数训练日志里 `ep=` 走到多少 | 确认 `main(epochs=100)` 或 `--epochs 100` 的写法；重跑 |
| extras 可视化阶段久久不动 | t-SNE 每快照要跑十几秒，100 epoch 有十来个快照 | 看进程 CPU 占用 | 属正常；只要曲线和探针可用 `collapse_audit.py`（无 t-SNE）代替 |
| 两次运行探针差 2-3 个点 | `ijepa.py` 未固定全局种子 | 翻源码确认没有 `manual_seed` | 属正常抖动；报告写单种子口径，比较消融时看差距是否显著大于这个抖动 |

## 11. 前沿与改造

I-JEPA（2023）证明了表征预测在图像上成立之后，这条路线的重心立刻转向视频：官方 ijepa 仓库 2024 年 8 月归档，Meta 的火力换到了 V-JEPA 和 V-JEPA 2，把"遮空间块"扩展成"遮时空管"，2B 参数的 ViT-G 权重公开可下载，还接上了动作条件版本用于机器人规划（第 15 课拿真权重干活）。LeCun 的立场文里，JEPA 只是六模块蓝图中"世界模型"那一格的架构候选，且他主张层级化 JEPA 做多时间尺度预测，这部分 2026 年的今天仍在施工，眼下能跑的是第 14 课的官方教学库 eb_jepa（含动作条件 JEPA 和 Two Rooms 规划）。防坍缩这个议题在前沿也没结案：EMA、stop-grad、方差正则（keon 仓库里 `leworldmodel.py` 的 SIGReg 是最新一路：不要 EMA 不要 stop-grad，用正则项直接顶住方差）三家仍在并行演化。

机制上不差：第 6 节的对账表核过，防坍缩的全部机制参数（块数、尺度、动量区间、stop-grad、predictor 不对称）与官方一字未改。差的全是规模，ViT-tiny 对 ViT-H/14 是三个数量级的参数差距，CIFAR-10 对 ImageNet 是两个数量级的数据差距，一台 Mac 对 16×A100 是纯粹的钱。规模买得到 52.7% 到论文水准的距离，买不到你这三条 std 曲线给出的理解。

动手改造清单（选做，各含预算与失败判据）：

1. 动量扫描：把线性 schedule 换成固定动量 $m \in \{0.9, 0.99, 0.999\}$ 各训一组（改 `--ema-start/--ema-end` 为同一个值即可）。预算：三组各 1 小时。预期：动量太低接近消融 (a) 的病态方向，太高则目标更新过慢、同 epoch 数下探针略低，0.99 附近有个甜区。失败判据：三组全打平，记为"CIFAR 规模对动量不敏感"，这本身回答了"schedule 是否关键"。
2. target LayerNorm 消融：把 `train()` 里 `F.layer_norm(tgt_enc(imgs), (D,))` 的 LN 去掉直接回归原始目标。预算：1 小时。预期：这是官方代码偏离论文公式的稳定性补丁，去掉后训练更抖或探针略降。失败判据：无差异，记为"小规模下 LN 不关键"，并想想为什么大模型才需要它。
3. predictor 砍深：`Predictor` 的 depth 从 4 砍到 1。预算：1 小时。预期方向开放：predictor 太浅时"预测"的负担被迫内化进 encoder 表征，探针可能掉，但这在小规模上未经证实，跑之前先写下你的赌注。
4. 探针对象对照：`collapse_audit.py` 收尾同时探 `out["ctx_enc"]` 和 `out["tgt_enc"]`（改两行）。预算：几分钟。预期：教程选择探 EMA 影子；影子是历史权重的平均，普遍略稳。两边差距大于单种子抖动才算有结论。

三条论文结论到缩小版的映射：I-JEPA 论文"multi-block 优于随机 masking"对应消融 (b)，预期方向能复现但差距缩水（8×8 网格几何差异小）；BYOL 的"动量目标网络挡住坍缩"对应消融 (a) 的 BYOL 判读；SimSiam 的"stop-grad + 预测头足以防塌"对应消融 (a) 的第三种结局，三种结局无论出哪个，都能挂到已有文献的坐标上，这正是审计设计成三格判读表的原因。

## 12. 论文与延伸

1. I-JEPA（Assran et al., 2023, [arXiv:2301.08243](https://arxiv.org/abs/2301.08243)），核心参考。带着三个问题读：3.1 masking 策略一节的四个数字与你核过的 yaml 是否一致？predictor 为什么设计成"窄而深"（384 维对 encoder 的 1280 维）？摘要说不依赖手工数据增强，可 crop 还在，这个"不依赖"的边界画在哪里？
2. BYOL（Grill et al., 2020, [arXiv:2006.07733](https://arxiv.org/abs/2006.07733)），EMA 防坍缩的前身，I-JEPA 目标编码器机制的直接出处。带着问题读：它的两个分支看的是同一图的两个增强视图，I-JEPA 换成了同一图的不同空间块，这个替换动了问题的哪个部分？论文对"为什么不坍缩"给到了哪一步（提示：是假说加消融，找到它承认没有理论保证的段落）？
3. SimSiam（Chen & He, 2020, [arXiv:2011.10566](https://arxiv.org/abs/2011.10566)），"EMA 不必要"的反例，摘要直言 stop-gradient 对防坍缩起关键作用。带着问题读：它诊断坍缩用的输出标准差图，和你 `collapse_audit.py` 里的 `feature_std` 是同一块仪表，对照它的图 2，你的消融 (a) 曲线像哪一条？它的结论能否直接外推到 I-JEPA 的"不同空间块"设置？（你的实验就是答案。）
4. A Path Towards Autonomous Machine Intelligence（LeCun, 2022, [OpenReview](https://openreview.net/pdf?id=BZ5a1r-kVsf)），立场文，故意发在 OpenReview 供公开批评。带着问题读：JEPA 在他的六模块架构（configurator、perception、world model、cost、actor、短期记忆）里占哪格？他反对生成式预测的核心论据（为不可预测细节付费）与你第 02 课的草地账单、本课的 scatter 消融各对上哪一段？读时保持清醒：这是提案不是实验论文，随手标注哪些主张有实证、哪些是赌注，第 16 课对决时这份标注直接派上用场。
5. keon/jepa 的 `ijepa_tutorial.md` 与 `FAITHFULNESS.md`，写得最像本课风格的第三方材料。带着问题读：教程里"共享权重则损失坍缩"的论证与本课 5.2/5.3 的版本有无出入？FAITHFULNESS 的 I-JEPA 条目所列简化，你的复现报告是否一条不落转述了？
6. 官方 [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa) README 与 configs（已归档，只读），工程账本。带着问题读：从 `in1k_vith14_ep300.yaml` 里再找一个本课没讲的参数（比如 `min_keep` 或 bfloat16），猜它防的是什么坑，再去 `src/` 里验证你的猜测。

---

到这课为止，系统的版图是：重建派（01-06）、价值等价派（07-08）、生成派（09-12）之外，第四条路正式开张，不画像素、不领工资，靠 EMA 和出题设计在纯自监督里撑住一个能读出语义的表征空间，而且你拆过它的两道防线，知道哪道是承重墙。但眼下这个 I-JEPA 只会看静态图片，说不出"接下来会发生什么"，更接不了动作，一个不会推演未来的表征器官还算不上世界模型。下一课把缺的两样补上：Meta 官方教学库 eb_jepa，从 Image JEPA 到 Video JEPA，再到动作条件的 Video JEPA 在 Two Rooms 里做规划，JEPA 路线从"学表征"走到"当世界模型使"，与第 05 课 RSSM 潜空间规划正面对表。
