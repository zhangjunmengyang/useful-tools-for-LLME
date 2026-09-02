---
id: 04_multicodebook_talker
title: "多码本 Talker"
summary: "让模型开口说话，难点到底在哪：是跨时间把一帧帧接顺，还是同一帧里 q0→q7 这 8 个编号之间的先后依赖？"
unit: mechanism
play_tools: []
checkpoints:
  - "把 8×T 的 delay schedule（错位排布）和它的逆变换画出来、单测过。"
  - "动手实现两种嘴：同帧 8 个头各说各的（independent heads），和一个轻量 depth decoder 逐层接力。"
  - "分清两种进步：teacher-forced CE（喂标准答案时的分数）变好，和 free-running（自己接自己）生成的音频变好。"
  - "拿质量、串行深度、RTF 和 TTFA 四个数，选定 Talker 的拓扑。"
---

# 第 04 课：比较 Talker 的多码本生成拓扑

Talker（模型的嘴：读 Thinker 递来的语义状态和已经生成的 codec code，继续预测下一步
语音 code 的模块；忘了这套分工的话回[第 01 课](01_baseline_reproduction.md)的速查表）
每走一步都面对同一个选择题：Mimi 每个音频帧包含 8 个 RVQ code，Talker 需要决定这些
code 以什么顺序产生。这个顺序，就是本课说的"生成拓扑"。

先做一个不花训练预算的实验。固定同一个输入、voice condition（音色条件：指定生成
音色的参考音频与说话人向量）和采样参数，将同一句长回答连续生成三遍。除保存 loss 外，
还要逐段检查以下位置：

- 辅音刚出现的瞬间，例如 "t""k""s"；
- 数字、英文缩写和中英文切换；
- 逗号后的重新起声；
- 句尾音量变小、模型准备停下来的两秒。

这些位置可能出现轻微金属感、擦音发散、重复音节或突然截断。错误可能来自 Thinker、
Mimi 或 Talker，单听波形无法确定。本课专门检验其中 Talker 的嫌疑：它是否充分建模了
同一帧内八个 RVQ code 之间的条件依赖。

实验先用三个码本、两帧语音手算 delay schedule（延迟调度表：规定每个 code 在第几个
生成步出现），再到 MiniMind-O 源码中逐项核对索引和张量。随后只改变 Talker 的生成
拓扑，比较三种模型：

1. 官方 diagonal delay；
2. 同帧八路并行；
3. 两组四路的 grouped depth decoder。

最终交付一张质量、串行深度和 TTFA 的 Pareto 图。质量越高越好，串行深度和 TTFA
越低越好。若方案 A 在三项指标上都不差于方案 B，且至少一项更好，则 A 支配 B；
未被其他方案支配的方案集合就是 Pareto frontier（帕累托前沿：所有"没被人全面压制"
的候选）。每个点都要能追溯到配置、seed、逐码本指标和可试听样例；不能根据结构名称
预先指定胜者。

## Talker 为什么有八路输出

 第一幕四课到这里收官，回头数数手里的零件：第 01 课没改模型，
造的是尺子和证据链；`baseline-v1`、100 个 golden case、不改变计算的 trace；
[第 02 课](02_multimodal_connector.md)拆的是"耳朵眼睛到脑子的接线"，把 connector
换成 MLP、Resampler、Q-Former 做过受控比较；[第 03 课](03_audio_codec.md)拆的是
"声音和编号的换算器"，搞清了 codec 重建分数高不代表接回系统后 Talker 也好训。
这一课拆最后一段：嘴。Talker 每帧要吐 8 个编号，谁先谁后；这个看起来无关紧要的
次序问题，就是全课主角。

三种拓扑用一个排队报数的类比就能讲明白。8 路码本像 8 个报数的人。**对角延迟**让
他们错开入场：第 1 路慢一拍，第 7 路慢七拍，后入场的人能听到前面的人已经报过的数，
代价是凑齐完整的第一帧要多等好几步。**同帧并行**让 8 个人同一瞬间开口：首帧最快，
但谁也听不见谁，只能各自看同一份提词稿（temporal hidden state）。**分组深度**折中：
前 4 人先报，后 4 人听完前 4 人的结果再报，一帧之内串行两步。类比在两处失效，
先记下防止带歪：第一，真实模型里"听到"靠的是把已采样 code 的 embedding 送进因果
历史或 depth block 的条件输入，没有谁抢话筒这回事；第二，并行的 8 个 head 也不是
两眼一抹黑；它们共享的 temporal state 可能已经编码了大部分可预测信息。所以三种
拓扑孰优孰劣是实验问题，拍脑袋答不了，本课要用三个只差一处的实验臂把它固定。

生成拓扑一手托两件事：帧内条件依赖（影响音质）和首帧等待
（直接吃进 TTFA）。第二幕（05-07 课）要把系统从回合制改成能实时插话，延迟预算按
几十毫秒抠；要是说不清自己的 Talker 产一帧要几次串行计算、首帧要等几步，流式改造
就是盲改。

**顺便小结第一幕拼出的完整系统。** 你的声音进冻结的 SenseVoice 变成特征，connector
把特征接进 8 层 Thinker；Thinker 想好要说的话，中间层 hidden state 过 bridge 递给
4 层 Talker；Talker 按本课研究的调度每帧吐 8 个编号；冻结的 Mimi 把编号解回 24 kHz
波形。四课下来，这条链路每一段你都亲手拆开、测量、留过证据。它现在的硬伤只剩一个：
回合制。用户语音要整段录完才送进 SenseVoice，模型只能"你说完我再想，想完我再说"。
第二幕开刀的就是这里：[第 05 课](05_streaming_listener.md)把"听"改成随到随编码的
流水线，[第 06 课](06_turn_policy.md)教它判断该不该开口，[第 07 课](07_full_duplex_routing.md)
做边说边听的全双工调度。那三课的每笔延迟账，都要用到你本课测出的 warmup 步数和
串行深度。

做完这课，你可以在本页的拓扑调度器里单步执行三种调度，预先算出
任意一格 code 的生成位置并让界面判你对错；可以拿同一句话试听 A/B/C 三臂的生成差异，
指着某个辅音说"这是拓扑的锅"；可以交出一张每个点都能回放到配置、seed 和音频的
Pareto 图。

本课术语（RVQ、TTFA、WER、teacher forcing 这些第 01 课的词，忘了就回去翻）：

| 术语 | 简要解释 |
|---|---|
| 生成拓扑 | 8 路码本谁先谁后的调度规则；本课比较对角、并行、分组深度三种 |
| diagonal delay | 第 $q$ 路整体推迟 $q$ 步：浅层先出、深层后出，深层能"看见"同帧浅层 |
| 同帧并行 | 一步吐齐一帧 8 个编号，首帧最快，但 8 个 head 互相看不见 |
| depth decoder | 大模型每帧算一次，小模块在帧内展开码本；时间、深度两个时钟分开 |
| canonical codes | 无 delay、无 shift 的原始 `[Q, F]` 码表，数据管线唯一长期保存的版本 |
| teacher-forced | 训练式前向：模型读真实历史；只说明"喂对历史时它行不行" |
| free-running | 推理式前向：模型读自己刚采样的输出，错误会一路累积 |
| NLL / PPL | 正确 token 的平均负对数似然；PPL 是它的指数，可读作等效候选数 |
| warmup | 从开始生成到第一帧凑齐的额外步数，一步不少地计入 TTFA |
| 串行深度 | 产出一帧需要几次没法并行的前向，墙钟延迟的下界来源 |
| exposure cascade | 前一组的采样错误顺着条件链传给后一组；teacher-forced 时看不见 |
| Pareto frontier | 没被任何方案全面压制的候选集合；本课的交付形式 |

## 如何区分源码事实和实验假设

写这类报告最常见的事故，是把"我以为代码是这样"写成"代码就是这样"，审计时查无对证。
所以全文把陈述分为三类，报告时也要沿用相同标记：

- **官方代码事实**：可以在 MiniMind-O 固定提交中逐行找到；
- **本课设计**：为了隔离变量而提出的实现或评测方法；
- **待验证假设**：只有跑完三臂实验后才能接受或拒绝。

源码讲解固定在 MiniMind-O commit
[`a10fa6c`](https://github.com/jingyaogong/minimind-o/tree/a10fa6c148ed274d66f96dc119689e93e01be823)。
使用其他提交时，先重新运行本课的 shape assertion 和 legacy parity test。代码行号、
special token 和生成索引都要以该提交的实际实现为准。

**官方代码事实**：该提交使用 8 个音频码本、2112 个音频词表项，默认
Talker hidden size 为 768、层数为 4。`2049/2050/2051` 分别是
audio pad、stop 和 speaker token。可在
[`OmniConfig`](https://github.com/jingyaogong/minimind-o/blob/a10fa6c148ed274d66f96dc119689e93e01be823/model/model_omni.py#L10-L29)
中核对。

**待验证假设**：显式恢复一部分帧内条件依赖，可能降低后层码本的 free-running
错误并改善长句边界，同时增加串行计算。若改善只出现在 teacher-forced CE，波形和
WER 不变，或延迟代价超过质量收益，则实验不支持该假设。

## 用 3×2 样例推导 Diagonal Delay

先用 8 路真数据。8 路乘几百帧的张量出了错，你看到的只是一片异常的 loss；
3 码本乘 2 帧的 toy 例子出了错，你能指着那一格说"就是这里"。索引类 bug 的调试成本
和张量大小成正比，所以先手算最小例子，再对源码。

### 1. 一个三码本、两帧的语音

设 codec 每帧只产生 3 个 codebook index，一共 2 帧。为了看清位置，不使用真实
token id，给它们起名字：

| codebook | frame 0 | frame 1 |
|---|---:|---:|
| $q_0$ | 10 | 11 |
| $q_1$ | 20 | 21 |
| $q_2$ | 30 | 31 |

这张表是 **canonical codes**：没有 delay，没有 BOS，也没有为了语言模型做
next-token shift。它应当是数据管线中唯一长期保存的版本——所有 schedule 都从它
现场推导，谁也别把加工过的中间产物存下来当真值。

现在给第 $q$ 路整体增加 $q$ 个 temporal step 的延迟。某个 code 出现的调度位置是

$$
s(f,q)=f+q,
$$

其中 $f$ 是原始 frame index，$q$ 是从 0 开始的 codebook index。

逐个代入：

| code | 手算 | schedule step |
|---|---|---:|
| $q_{0,0}=10$ | $0+0$ | 0 |
| $q_{0,1}=11$ | $1+0$ | 1 |
| $q_{1,0}=20$ | $0+1$ | 1 |
| $q_{1,1}=21$ | $1+1$ | 2 |
| $q_{2,0}=30$ | $0+2$ | 2 |
| $q_{2,1}=31$ | $1+2$ | 3 |

于是对角调度表是：

| codebook / schedule step | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| $q_0$ | 10 | 11 | PAD | PAD |
| $q_1$ | BOS | 20 | 21 | PAD |
| $q_2$ | BOS | BOS | 30 | 31 |

按列读取该表，可得到每个 step 的预测目标：

1. step 0 预测 $q_0$/frame 0；
2. step 1 预测 $q_0$/frame 1 和 $q_1$/frame 0；
3. step 2 预测 $q_1$/frame 1 和 $q_2$/frame 0；
4. step 3 预测 $q_2$/frame 1。

同一个 temporal step 的多个 head 并没有预测同一帧。它们预测的是一条对角线。
完整的 frame 0 要等到 step 2 才凑齐：依次取 step 0 的 $q_0$、step 1 的
$q_1$ 和 step 2 的 $q_2$，得到 `[10, 20, 30]`。frame 1 沿下一条对角线
重组，依次取 step 1、2、3 的对应 code，得到 `[11, 21, 31]`。这就是"错开入场"的
代价和收益同时成立的地方：首帧要等对角线填满，但 $q_2$ 开口时，同帧的
$q_0$、$q_1$ 已经躺在历史里。

### 2. Schedule shift 与 next-token shift

常见错误是两种位移长得像、来源不同。上表描述"目标应该放在哪个
schedule step"，是拓扑自己的 $q$ 位延迟；因果语言模型还要把输入和 label 错开
一位——当前位置输入历史，预测下一位置的目标。前者是"这一路本来就晚出发"，
后者是"所有路都永远预测下一步"，两者叠加才是训练用的张量。

MiniMind-O 先构造长度为 $T$ 的 `Y_audio_layers`，然后执行：

```python
X_audio     = Y_audio_layers[:, :-1]
audio_label = target_layers[:, 1:]
```

因此模型序列位置 $s$ 的 logit 对齐原始 target 数组的 $s+1$。调试时必须同时打印：

| 对象 | shape |
|---|---|
| canonical codes | `[Q, F]` |
| delayed `Y` targets | `[Q, T]` |
| `X_audio` | `[Q, T-1]` |
| audio labels | `[Q, T-1]` |

调试时必须同时保存三份对象。只保存最终 label 时，无法区分 schedule 的 $q$ 位延迟
和 causal LM 的一位右移——两种 bug 在最终张量里长得一模一样。

### 3. 用单测验证索引

先用上面的 3×2 例子验证索引，再运行八路真实样本：

```python
canonical = torch.tensor([
    [10, 11],
    [20, 21],
    [30, 31],
])

scheduled = build_diagonal_targets(canonical, delay=1)

assert scheduled.tolist() == [
    [10, 11, PAD, PAD],
    [BOS, 20, 21, PAD],
    [BOS, BOS, 30, 31],
]

assert reconstruct_frames(scheduled).tolist() == canonical.T.tolist()
```

这里的 `BOS`、`PAD` 只是 toy schedule 的记号，不要直接映射成 MiniMind-O 的
special token。真实代码还包含 assistant 起点、speaker condition、reference codes、
stop token 和文本序列，它们应该在 toy case 通过后逐项加入。

完成手算后，逐题写出 schedule step 和条件信息：

1. 为什么 3 个码本的 frame 0 要到 step 2 才完整？
2. step 1 的两个真实 code 属于同一帧吗？
3. 如果 delay 改为 0，frame 0 什么时候完整？模型失去了哪种条件信息？
4. 如果 delay 改为 2，frame 0 什么时候完整？增加的是吞吐成本还是 warmup 成本？

## RVQ 码本之间的条件依赖

拓扑之争的根子在这：8 路码本到底是 8 个不相干的序列，还是一条链上的蚂蚱？
答案由 RVQ 的构造方式决定。

### 4. RVQ 的残差递推

Residual Vector Quantization（残差向量量化，RVQ；第 01 课用"找零钱"打过比方：
第 0 层先给粗略近似，后面各层专记没找清的差额）并没有把同一个向量交给八个互不
相关的分类器。第 0 层先量化主要结构，后续层继续量化前面没解释完的 residual。

设 codec encoder 在 frame $t$ 输出连续向量 $z_t$，第 $k$ 层码本为
$E_k=\{e_{k,j}\}$。RVQ 的递推可以写成：

$$
r_{0,t}=z_t,
$$

$$
q_{k,t}=\arg\min_j\left\|r_{k,t}-e_{k,j}\right\|_2^2,
$$

$$
r_{k+1,t}=r_{k,t}-e_{k,q_{k,t}}.
$$

最终的量化向量是

$$
\hat z_t=\sum_{k=0}^{K-1}e_{k,q_{k,t}}.
$$

因此 $q_7$ 量化的是前七层处理后剩余的残差，它与 $q_0$ 不满足统计独立。
后层可能更多影响细节，但具体分工取决于 codec 的训练目标和码本使用情况。找零类比
到此为止：零钱面额是固定的，码本的"面额"是训练学出来的，"后层等于音色细节"这类
感知排序必须通过逐层替换或消融验证，不能当成固定事实。

### 5. 用链式法则写出完整目标

给定 Talker 对 frame $t$ 的条件状态 $h_t$，同一帧的联合分布总可以按链式法则分解：

$$
p(q_{0:K-1,t}\mid h_t)
=\prod_{k=0}^{K-1}
p(q_{k,t}\mid h_t,q_{<k,t}).
$$

这里的 $q_{<k,t}$ 表示同一帧里编号更小的 code。这是"完整版"目标：每一路都看得到
同帧比它浅的所有决定。

如果八个 head 只共享 $h_t$，彼此看不到同帧已生成的 code，模型实际采用的是条件独立
近似：

$$
p_{\text{parallel}}(q_{0:K-1,t}\mid h_t)
=\prod_{k=0}^{K-1}p(q_{k,t}\mid h_t).
$$

该近似是否足够，需要实验确定。temporal backbone 可能已将大部分可预测信息编码进
$h_t$，后层 code 对波形的边际贡献也可能较小。公式只说明两个模型采用了不同的条件
分解，不能据此判断质量高低——这正是三臂实验存在的理由。

### 6. Diagonal Delay 将码本深度映射到时间步

Diagonal delay 没有在单个 step 内依次生成 $q_0,q_1,\ldots$。它把 codebook depth
搬到外部 temporal axis 上：深度上的"先后"被翻译成时间上的"早晚"。

令 $Y_{k,s}=q_{k,s-k}$ 表示 delay 后第 $k$ 路在 step $s$ 的有效目标。当前实现更接近

$$
p_{\text{diag}}(Y)
=\prod_s\prod_{k\in\mathcal V_s}
p(Y_{k,s}\mid C_{\le s},Y_{<s}),
$$

其中 $\mathcal V_s$ 是 step $s$ 上有效的 codebook 集合，$C$ 表示 Thinker bridge
提供的语义条件。给定同一个 temporal hidden state，当前 step 的八个 head 仍然并行；
但是 $q_{1,t}$ 被推迟一拍，$q_{2,t}$ 被推迟两拍，所以较浅的同帧 code 已经出现在
causal history 中。

它不属于纯粹的 frame-local factorization。预测 $q_{2,t}$ 时，历史里既有
$q_{0,t},q_{1,t}$，也混有相邻帧的浅层 code。Temporal Transformer 要同时学习
时间演化和帧内 depth 两种关系——一个模型打两份工，能不能打好，也是实验问题。

### 7. Depth decoder 把两种时钟拆开

depth decoder 的思路是让两份工各归各：大模型对每个音频帧运行一次（时间时钟），
再由较小的 depth module 在帧内展开（深度时钟）：

$$
h_t=\operatorname{Temporal}(C_{\le t},q_{:,<t}),
$$

$$
p(q_{0:K-1,t}\mid h_t)
=p(q_{0,t}\mid h_t)
\prod_{k=1}^{K-1}
p(q_{k,t}\mid h_t,q_{<k,t}).
$$

Full depth autoregression 要串行运行 $K$ 次小模块。本课先做更实用的两组方案：

$$
p(q_{0:7,t}\mid h_t)
=p(q_{0:3,t}\mid h_t)\,
p(q_{4:7,t}\mid h_t,\phi(q_{0:3,t})),
$$

其中每组内部四个 head 并行，因此实现中还会作近似：

$$
p(q_{0:3,t}\mid h_t)
\approx\prod_{k=0}^{3}p(q_{k,t}\mid h_t),
$$

$$
p(q_{4:7,t}\mid h_t,\phi)
\approx\prod_{k=4}^{7}p(q_{k,t}\mid h_t,\phi).
$$

$\phi$ 汇聚第一组 code embedding，作为第二组的条件。外部 temporal step 仍按一帧一次
运行，内部串行深度为 2。该设计建模组间依赖，但组内四个 code 仍采用条件独立近似。
它只在"完整链式"和"全并行"之间买了最便宜的一档：一次组间依赖。这一档
值不值票价，就是臂 C 要回答的问题。

### 8. Teacher-forced 与 free-running 的输入差异

训练 depth decoder 时，第二组通常读取真实的 $q_{0:3,t}$；推理时则读取第一组的采样
结果。两种输入分布不同，因此第一组错误可能继续影响第二组——这就是速查表里的
exposure cascade，第 01 课讲过的 exposure bias 在组间条件上的翻版。

因此下面两个数字必须分开：

$$
\Delta_{\text{TF}}
=\operatorname{NLL}_{\text{depth, TF}}
-\operatorname{NLL}_{\text{diag, TF}},
$$

$$
\Delta_{\text{free}}
=\operatorname{WER}_{\text{depth, free}}
-\operatorname{WER}_{\text{diag, free}}.
$$

如果 $\Delta_{\text{TF}}<0$ 而 $\Delta_{\text{free}}\approx 0$，只能说明 depth
模型在给定正确历史时具有更低 NLL，不能得出生成质量改善的结论。考卷上教练扶着
方向盘考出的分，说明不了独自上路的水平。

## MiniMind-O 中的九路张量与生成索引

toy 例子过关后，回到真实代码。这一节全部是可以逐行指认的**官方代码事实**，
你的任务是拿着链接亲眼确认每一条。

### 9. 数据集将一维 codes 转为九路序列

**官方代码事实**：`answer_audios` 按 frame 交错保存。数据集每次取 8 个整数，
分别放进八个列表，再给每一路追加 stop token。见
[`OmniDataset.__getitem__`](https://github.com/jingyaogong/minimind-o/blob/a10fa6c148ed274d66f96dc119689e93e01be823/dataset/omni_dataset.py#L256-L266)。

若一条样本有 $F$ 个 Mimi frame，解交错后的 `last_audio_codes` 是
`list[8][F + 1]`；额外的一列来自每一路末尾的 `audio_stop`。

数据集先找到最后一个 assistant 区间；若前 50 个 token 内扫描到 `think_end_ids`
（思考结束标记），又会把锚点移动到 think-end 之后。令这个**完成扫描后的
audio target anchor** 为 $a$，第 $q$ 路从 $a+q+1$ 开始写 target：

$$
\operatorname{targetPos}(f,q)=a+q+1+f.
$$

拆开看：$a$ 是"从哪开始说话"，$q$ 是对角延迟，$1$ 是那次 next-token shift——
第 2 小节的两种位移在这一个公式里会合。这段代码在
[`omni_dataset.py` L311–L318](https://github.com/jingyaogong/minimind-o/blob/a10fa6c148ed274d66f96dc119689e93e01be823/dataset/omni_dataset.py#L311-L318)。
随后它构造 9 路输入并做 next-token shift：

| 对象 | 单样本 shape | 含义 |
|---|---|---|
| `Y_audio_layers` | `[8, T]` | delay 后、尚未做 causal shift |
| `X_audio` | `[8, T-1]` | 八路 Talker 输入 |
| `X_text` | `[T-1]` | Thinker 输入 |
| `input_ids` | `[9, T-1]` | 前八路 audio + 最后一路 text |
| `audio_labels` | `[8, T-1]` | 每路下一个 audio target |
| `text_labels` | `[T-1]` | 下一个 assistant text target |

对应源码是
[`omni_dataset.py` L320–L329](https://github.com/jingyaogong/minimind-o/blob/a10fa6c148ed274d66f96dc119689e93e01be823/dataset/omni_dataset.py#L320-L329)。

在你改 schedule 之前，加入下面的断言：

```python
assert input_ids.ndim == 2 and input_ids.shape[0] == 9
assert audio_labels.shape == input_ids[:8].shape
assert text_labels.shape == input_ids[8].shape
assert (audio_labels != -100).sum(dim=1).min() > 0
```

最后一条不适用于无 assistant audio 的混合样本，真实实现应先按样本类型判断；它适合
本课纯 Text-to-Audio（T2A，文字进语音出）toy batch。

### 10. Forward 中八路信息的合并位置

一个 batch 进入 `MiniMindOmni.forward` 后，shape 依次是：

| 代码对象 | shape |
|---|---|
| `input_ids` | `[B, 9, S]` |
| `text_ids` | `[B, S]` |
| `audio_ids` | `[B, 8, S]` |
| `bridge_states` | `[B, S, H]` |
| `talker_emb` | `[B, S, H_t]` |
| Talker fused hidden | `[B, S, H_t]` |
| `h_talker` | `[B, S, H_t]` |
| `audio_logits[q]` | `[B, S, 2112]`，共 8 个 |

默认 $H=H_t=768$。拆分发生在
[`model_omni.py` L245–L252](https://github.com/jingyaogong/minimind-o/blob/a10fa6c148ed274d66f96dc119689e93e01be823/model/model_omni.py#L245-L252)。

`TalkerEmbedding` 没有把八路 token 拼成 $8H_t$ 的长向量。它对每一路使用 shared
embedding 加该路 adapter，然后把八路结果求平均：

$$
e_s=\frac{1}{8}
\sum_{q=0}^{7}
\left(E_{\text{base}}(x_{q,s})+A_q(x_{q,s})\right).
$$

源码见
[`TalkerEmbedding`](https://github.com/jingyaogong/minimind-o/blob/a10fa6c148ed274d66f96dc119689e93e01be823/model/model_omni.py#L68-L76)。
这意味着八路在进入 4 层 temporal Talker 之前已经压成一个 `[B,S,H_t]` 表示。

Talker 的输入是 Thinker 中层 bridge 与 codec history 的加权和：

$$
u_s
=\alpha\,P_{\text{text}}(b_s)
+\beta\,P_{\text{codec}}(e_s),
$$

其中 $\alpha$、$\beta$ 是可学习标量，初始化分别为 3 和 1。融合与四层 Talker
forward 可在
[`model_omni.py` L295–L306](https://github.com/jingyaogong/minimind-o/blob/a10fa6c148ed274d66f96dc119689e93e01be823/model/model_omni.py#L295-L306)
核对。

最后，`TalkerHead` 先算一个 shared linear，再加 8 个独立 adapter，返回 8 个
`[B,S,2112]` logits。这里的 head 是"把 hidden state 映射到某一路词表"的输出层，
logit 是 softmax 之前的未归一化分类分数：

$$
\ell_{q,s}=W_{\text{base}}h_s+A_q(h_s).
$$

见
[`TalkerHead`](https://github.com/jingyaogong/minimind-o/blob/a10fa6c148ed274d66f96dc119689e93e01be823/model/model_omni.py#L57-L65)。
八个 head 在同一 forward 中并行；帧内依赖来自 delay 后的历史，不来自 head 之间的
当前步通信。把这句话核实到源码，第 5 小节的"条件独立近似"就从公式落到了实现。

### 11. 第一帧的完整生成时刻

真实 `stream_generate` 还在 schedule 外加了一拍 text-to-audio delay：

```python
step = input_ids.shape[1] - start_pos
audio_step = step - 1
```

第 $i$ 路只有在 `audio_step >= i` 时才开始采样。见
[`model_omni.py` L363–L378](https://github.com/jingyaogong/minimind-o/blob/a10fa6c148ed274d66f96dc119689e93e01be823/model/model_omni.py#L363-L378)。

所以需要区分两个数字：

- 相对 $q_0$，凑齐 $q_7$ 需要 7 个额外 outer steps；
- 按源码的 `step` 计数，还存在最初的 `audio_step = step - 1`，第一帧在
  `step == 8` 时才可重组。

重组公式是：

```python
frame = [audio_codes[i][step - 7 + i] for i in range(8)]
```

见
[`model_omni.py` L386–L390](https://github.com/jingyaogong/minimind-o/blob/a10fa6c148ed274d66f96dc119689e93e01be823/model/model_omni.py#L386-L390)。

使用日志验证 `global_step`、`audio_step`、`head_q`、sampled list index、
original frame index 和 reconstructed frame。六个字段必须出现在同一条逐步 trace 中，
否则无法检查 code 从采样位置到物理帧的映射。

先在 greedy 模式（每步取概率最高的 token，不掷骰子）打印前 12 步。若日志无法按
源码公式还原第一帧，应先修正索引实现，不开始架构改造。

### 12. Stop Token 权重对 Loss 的影响

**官方代码事实**：训练脚本对八路分别计算 masked cross entropy，再取平均；
target 为 `2050` 的 stop token 权重是普通 token 的 10 倍。见
[`train_sft_omni.py` L82–L96](https://github.com/jingyaogong/minimind-o/blob/a10fa6c148ed274d66f96dc119689e93e01be823/trainer/train_sft_omni.py#L82-L96)。

三臂必须使用相同 stop 权重，并单独报告 stop loss。短样本中少量高权重 stop token
会显著影响平均 audio loss；若不拆分，无法判断差异来自普通 codec token 还是 stop
token——你以为在比拓扑，实际在比谁的样本更短。

推理路径的停止判定比训练 target 更宽：当前代码把任意 `code >= 2048` 都记作该路
停止，而不只检查 `2050`。这可以在
[`model_omni.py` L373–L380](https://github.com/jingyaogong/minimind-o/blob/a10fa6c148ed274d66f96dc119689e93e01be823/model/model_omni.py#L373-L380)
核对。三臂比较时先保持这一规则一致，同时记录实际采样到的 special id；否则
"更早停"可能只是某个 head 更容易采到未解码的 special token。

## 用页面调度器核对三种拓扑

页面实验只展示三种确定性的生成调度，不模拟神经网络输出：

- **对角延迟**：$s(f,q)=f+q\Delta$；
- **同帧并行**：$s(f,q)=f$；
- **两组深度**：outer temporal step 仍是 $f$，同一帧内部再按 group 运行两个
  inner step。

控件支持 3/4/5/6/8 个 codebook 和 2–7 帧。先复现开头的 3×2 手算，再检查真实
8 路布局。网格只验证每个 code 的生成位置，不预测音质，也不用于选择实验臂。
它的价值是把第 1-3 小节的手算变成可以点错、被纠正、再点对的肌肉记忆。

### 13. 第一轮：在网格里复现 3×2 对角线

把生成拓扑设为“对角延迟”，`Codebooks` 设为 3，`Frames` 设为 2，
`Δ / layer` 设为 1。

页面要求填写 `Q3` 的 `F1` 所在 step。界面标签从 `Q1` 开始，公式中的 codebook index
从 $q=0$ 开始，因此 `Q3` 对应 $q=2$：

$$
s(1,2)=1+2\times1=3.
$$

输入 `3`，锁定预测，然后单步执行。step 0 应出现 `Q1:F0`；step 1
出现 `Q1:F1` 和 `Q2:F0`；step 2 出现 `Q2:F1` 和 `Q3:F0`；
step 3 最后出现 `Q3:F1`。

frame 0 在 step 2 才完整，目标 `Q3:F1` 在 step 3 出现。运行后对照表格写出
frame 0 的三个 code 分别来自哪个 step，以验证重组规则。

### 14. 第二轮：切成同帧并行

保持 3×2 不变，把生成拓扑切到“同帧并行”。现在目标位置是：

$$
s(1,2)=1.
$$

step 0 会同时出现 `Q1:F0`、`Q2:F0`、`Q3:F0`，step 1 同时出现三路 `F1`。
首帧不再等待对角线填满，但同一步的三个 head 也看不到彼此刚采样的 code。它们只共享
同一个 temporal state $h_t$。因此该对照减少了首帧等待，同时省略同帧 code 之间的
采样条件。两种拓扑的取舍在网格上一目了然：省下的等待和丢掉的条件，是同一枚硬币
的两面。

### 15. 第三轮：把 temporal step 和 inner depth 分开

再切到“两组深度”。3 个 codebook 按 `ceil(3/2)` 分组：inner step 0
处理 `Q1, Q2`，inner step 1 处理 `Q3`。

线性网格供逐格播放使用，坐标写成

$$
\operatorname{slot}(f,q)=2f+g(q).
$$

因此 `Q3:F1` 位于 slot 3，但它表示 outer temporal frame `F1`
中的 inner depth step 1。

slot 3 不是第三个 temporal step。它表示同一帧中的第二次 inner 计算。性能测试应记录
每个 inner step 给单帧增加的墙钟时间，不能用线性网格的 slot 数代替 temporal step。

再把 `Codebooks` 改成 8、`Frames` 改成 3。确认 `Q1–Q4` 在 inner 0，
`Q5–Q8` 在 inner 1；`Q8:F2` 位于 slot 5，对应 temporal `F2`、inner 1。
用该结果核对 MiniMind-O 的八路布局。

### 16. 从教学网格回到真实系统

完成三轮后，把观察映射回 MiniMind-O：

| 网格概念 | 真实系统中的对象 |
|---|---|
| 对角线或并行的一列 step | 一次 Talker outer temporal forward |
| grouped 的一列 slot | 某个 frame 内的一次 inner depth 计算 |
| 一格 `Q#:F#` | 某个 head 当前要预测的 codec code |
| BOS 区 | 该 codebook 尚未开始产生有效 frame |
| PAD 区 | 该 codebook 已结束或当前无有效 target |
| 对角线凑齐 | `stream_generate` 的 frame reconstruction |

页面实验没有覆盖 assistant 起点、text-to-audio 一拍延迟、reference codes、speaker
token、stop 和 KV cache（推理时缓存的注意力键值，续写时不必重算前缀）。真实代码的
单测必须覆盖这些边界；网格上点得再熟，也代替不了对真实张量的断言。

## 将生成拓扑封装为可替换接口

### 17. 先封装 legacy，不改变任何行为

改造的第一步偏偏是"什么都不改"。先将数据构造、生成状态和 frame reconstruction
统一到一个接口，再增加新 schedule；不要直接在 `forward` 中分散添加条件分支——
分支散进模型代码后，"只改拓扑、其他全同"就再也无法保证，三臂比较的地基就塌了：

```python
class CodeSchedule(Protocol):
    def build_train(
        self,
        canonical_codes: Tensor,  # [B, Q, F]
        assistant_start: Tensor,
    ) -> TrainSchedule:
        ...

    def begin(self, batch_size: int, device: torch.device) -> DecodeState:
        ...

    def append(
        self,
        state: DecodeState,
        logits: list[Tensor],
    ) -> DecodeState:
        ...

    def pop_complete_frames(
        self,
        state: DecodeState,
    ) -> Tensor | None:           # [B, n_ready, Q]
        ...
```

第一份实现只搬运官方 diagonal 逻辑。对固定 batch 和固定 greedy decode，它必须与
旧路径逐项比较 `X_audio`、`audio_labels`、valid mask、每步 8 路 logits、
sampled code、重组后的 frame 和 stop position。

索引张量和 greedy code 应完全一致。若重构改变 kernel 路径，先重复运行原路径以测量
浮点自然漂移，再按 dtype 和设备声明 `atol/rtol`。公差要写入测试配置，不能在失败后
临时放宽。这套流程第 01 课的 trace parity 做过一遍，这里原样再用。

### 18. Same-frame parallel 对照

对照臂 B 保持 outer temporal clock 不变，但同一 frame 的八路 target 放在同一个
step：

$$
Y_{q,t}=q_{q,t}.
$$

最直观的版本让八个 head 都只条件于同一个 $h_t$：

```python
audio_logits = [head_q(h_t) for head_q in heads]
```

臂 B 用于测量移除 delay 后的后层 PPL、波形和延迟变化，不预设为最终候选。
帧内条件本身的因果效应由后面的 B/C 容量匹配对照检验。

正式比较 B/C 时要控制模型容量。C 增加一个 depth block 后，如果 B 没有同形模块，
结果会同时包含参数量和条件方式两项差异——到时候你说不清赢在"多了条件"还是
"多了参数"。可让 B/C 共享同样的 group-1 block：

```python
# 两臂都有同形状的 H -> D -> H 路径
base_g1 = h_t + out_proj(depth_block(h_proj(h_t)))

# B: group 1 只看 base_g1，可与 group 0 并行
# C: group 1 在 D 维空间额外读取 sampled_g0
```

B/C 使用相同 block、hidden size 和 head 数；差别只在 group-1 是否读取已采样的
$q_{0:3,t}$。这样 B 仍是帧内条件独立对照，只是后四个 head 多了一次不依赖 code
的非线性变换。

实现时不要改变：

- Thinker checkpoint 与 bridge layer；
- Mimi target codes 与 decoder；
- speaker/reference condition；
- outer temporal hidden size 和层数；
- sampling policy、stop 规则和训练暴露的有效 frame 数。

### 19. Grouped depth 对照

臂 C 每个 outer step 先用 temporal state 预测 $q_0$ 到 $q_3$，再把第一组的 code
embedding 汇聚后送进与 B 同形状的 block，预测 $q_4$ 到 $q_7$：

```python
# h_t: [B, 1, H]
logits_g0 = [heads[q](h_t) for q in range(4)]

# train: teacher codes; inference: sampled codes
codes_g0 = choose_group0(logits_g0, targets, teacher_forcing)
cond_g0 = group_embed(codes_g0)              # [B, 1, D]

depth_in = h_proj(h_t) + cond_proj(cond_g0)   # [B, 1, D]
depth_h = h_t + out_proj(depth_block(depth_in))  # [B, 1, H]
logits_g1 = [heads[q](depth_h) for q in range(4, 8)]
```

其中 `h_proj: H -> D`、`cond_proj: D -> D`、`out_proj: D -> H`；因此示例配置里
$H=768,D=384$ 时张量可以相加。B 使用同一套投影和 block，只把 code condition
去掉。第一版只用一个小 depth block。先让串行关系可测、可 profile（逐段计时），
再讨论 full 8-depth Transformer。组内仍是条件独立近似。本课只检验增加一层组间
条件后，收益是否超过额外串行开销。

训练时同时记录两种前向：teacher-forced group 1 读取 ground-truth
`q0:q3`，free-running group 1 读取实际 sampled `q0:q3`。

若只测第一种，你看不到 exposure cascade。

## 只改变生成拓扑的三组实验

### 20. 三个实验臂分别隔离的变量

| 臂 | outer schedule | 帧内条件 | inner serial depth | 它回答的问题 |
|---|---|---|---:|---|
| A | diagonal | 通过 temporal history 间接获得 | 1 | 官方设计的质量与 warmup 基线 |
| B | same-frame | group-1 block 只看 $h_t$，不看 group 0 code | 1 | 没有帧内 code 条件时能做到什么 |
| C | same-frame | 同一个 group-1 block 额外读取 group 0 code | 2 | 一次采样依赖是否换来可见收益 |

Full 8-depth AR 可作为研究附录，不临时加入主表。主表配置在运行 test 前冻结，避免
根据 test 结果增删实验臂。

A 是"现有系统参考"，B/C 才是严格的帧内条件消融对。A 与 B/C 同时改变 schedule，
因此 A/C 的差异适合报告系统 Pareto，不足以单独证明某一种条件分解更优。若要把
diagonal 与 same-frame 的纯拓扑效应也做成因果结论，需要再加容量匹配对照，或把
三臂全部重写成参数相同的 scaffold。报告中必须明确 A/C 同时改变了哪些变量。

### 21. 机制比较与 warm-start 升级分开报告

如果从已训练的 diagonal Talker 初始化，A 能沿用原分布，B/C 需要适应新 schedule。
该设置适合测量升级成本，不适合单独判断拓扑效果——起跑线不同的比赛只能比"谁转型
快"，比不了"谁天赋好"。

因此建议保留两个 track：

**机制 track（主结论）**

- 冻结同一个 Thinker、bridge 与 Mimi；
- B/C 的 Talker scaffold、group-1 block 与 head 从同一随机种子初始化；
- A 也使用同一份 neutral Talker 初始化，不继承已经训好的 diagonal 优势；
- shape 相同的 temporal blocks 使用相同初值；A 仍只作为系统参考，因为它与 B/C
  的 scaffold 不完全同形；
- 三臂看到相同 canonical codes、相同顺序、相同有效 audio frames 和 optimizer
  updates。

**warm-start track（工程结论）**

- 三臂都从 `baseline-v1` 开始；
- A 直接加载；B/C 明确记录哪些权重可复用、哪些重置；
- 比较在固定追加预算内，哪一臂最快恢复并超过 baseline。

两个 track 使用独立表格和结论，不合并排名。

### 22. 128 样本、10k Pilot 和 Standard 的用途

**128 样本**

从 T2A mini 固定 128 条，关闭 augmentation 和随机采样。该阶段只验证三臂的 mask、
head、stop 与 reconstruction 能否学习，不用于比较拓扑质量。128 条都背不下来的臂，
先修实现再谈比较——第 01 课的过拟合纪律原封不动搬过来。

**10k pilot**

从官方 T2A mini 取英文短句、长句、数字、缩写、标点边界和不同 speaker condition。
三臂至少运行两个 seed，用于估计方差、测量延迟并锁定正式配置。该集合属于 dev/pilot，
不能在调参后作为 test 报告。

**standard**

使用同一版 `sft_t2a_mini` canonical Mimi codes。Audio-to-Audio（A2A，语音进语音出）
作为单独验证桶，避免输入语音编码器变化掩盖 Talker 结论。数据 manifest 至少记录：

```yaml
sample_id: stable-hash
source_revision: string
text: string
target_codes_path: string
target_codes_shape: [8, frames]
reference_codes_path: string | null
speaker_id_hash: string
language: en | zh | mixed
frame_count: int
split: train | dev | test
license_or_terms: string
```

代码仓库的 Apache-2.0 许可证不自动覆盖每份训练语料。若 codes 的来源或再分发权限
不清楚，只发布 hash、脚本和统计，不发布原始数据。

官方 mini 数据是英文且无视觉。中文和中英混说只能放在独立扩展桶：使用官方 full
数据或来源、许可均可审计的自建数据，另建 manifest，不得把它描述成 mini standard
覆盖。

### 23. 三阶段训练

#### 阶段一：索引与过拟合

1. 跑 3×2 schedule 单测；
2. 跑 official 8×F layout trace；
3. 验证 legacy wrapper parity；
4. 三臂各过拟合 128 条；
5. 用 frozen Mimi 解码固定 10 条，人工听重组是否错位。

出现周期性爆音时，先查 schedule，不要立即调学习率。周期性是索引错位的签名：
学习率的问题不会每隔固定帧数发作一次。

#### 阶段二：teacher-forced pilot

固定同一 batch order 和有效 audio frame 数，训练到相同 optimizer update。每路记录：

$$
\operatorname{NLL}_q
=-\frac{1}{|M_q|}
\sum_{(b,s)\in M_q}
\log p(q_{b,q,s}\mid \text{condition}),
$$

$$
\operatorname{PPL}_q=\exp(\operatorname{NLL}_q).
$$

NLL 是正确 token 的平均负对数似然，数值越低越好。PPL 是 NLL 的指数变换，可解释为
模型在每一步面对的等效候选数量；两者对同一组结果给出相同排序。

平均 audio loss 只能作为训练曲线，不能替代 per-codebook 诊断。拓扑改动的效果
往往集中在后几路码本，平均数会把它抹平。

#### 阶段三：free-running 与压力测试

在 dev 上使用同一个 sampler 跑：

1. greedy；
2. 锁定 temperature、top-k/top-p 的随机采样；
3. 第一组 code 以固定概率替换为模型采样；
4. 第一组 code 做 5% 和 10% 的随机 corruption（人为注错：故意污染部分 code，
   看错误在条件链上传多远）；
5. 长句连续生成，不在中途重新 teacher force。

corruption 比例是实验自变量，不属于最终 recipe。若 free-running 出现错误累积，
可在下一轮实验中为所有臂统一加入 scheduled sampling（按概率把真实历史换成模型
自己的预测，第 01 课 5.2 节讲过）。不能只为 C 增加正则化后直接比较拓扑。

### 24. 八卡任务分配

Talker 规模较小，八张卡优先运行独立实验：

| GPU | 任务 |
|---:|---|
| 0 | A / seed 42 |
| 1 | A / seed 43 |
| 2 | B / seed 42 |
| 3 | B / seed 43 |
| 4 | C / seed 42 |
| 5 | C / seed 43 |
| 6 | frozen eval：ASR、speaker、waveform metrics |
| 7 | latency profiler、第三个 seed 或失败 case 重放 |

pilot 稳定后补第三个 seed。若单个任务需要 DDP（PyTorch 多卡数据并行，第 01 课
用过），按每次 update 的有效 audio frame 配平 global batch。不同 schedule 的
tensor 长度不同，单独对齐 `batch_size` 不等价。

## 同时评测 Token、波形与系统延迟

三种拓扑改变的东西分布在三个层面：token 的可预测性、波形的听感、系统的墙钟延迟。
只测其中一层，另外两层的回归就会漏网，所以三套量尺一起上。

### 25. 内容正确性

固定同一个 ASR（自动语音识别，把波形转回文字）、同一版 text normalization 和
同一语言分桶。英文报告 WER：

$$
\operatorname{WER}=\frac{S+D+I}{N}.
$$

中文同时报告 CER（字错误率，定义回第 01 课）。保存 ASR transcript，人工复核极端值；
ASR 的识别偏差不能直接归因于 Talker。

### 26. 声学质量与说话人

至少包含：

- frozen 无参考质量 proxy，例如 UTMOS（一个预测人耳打分的模型，不需要参考音频）；
  它不是人类偏好的真值；
- ViSQOL 只在存在时间对齐的参考 waveform 时使用，因为它是 full-reference 指标
  （必须拿生成波形和真值波形逐段对比）；如果只有 Mimi codes、没有原始 assistant
  waveform，就不报告 ViSQOL；
- CAM++ speaker cosine，固定裁剪和采样率；
- click、dropout 与重复片段 detector；
- 盲听 AB（两版音频不标来源，人只判哪个好），重点标注辅音、跨语言、停顿恢复和句尾；
- 长句头、中、尾三段 speaker similarity。

重点保留两类样例：WER 相同但盲听可稳定区分，以及 token PPL 明显改善但盲听无法区分。
两类结果分别说明自动内容指标和 token 指标的覆盖范围，报告时不得只展示支持假设的音频。

### 27. 系统时间线

为每个请求依次记录 `request_received`、`thinker_prefill_end`、
`first_text_token`、`first_q0_code`、`first_complete_codec_frame`、
`first_mimi_pcm_chunk`、`playback_enqueued` 和 `generation_end`。

TTFA 必须取首个可播放 PCM；首个 q0 logit 不算数，那时用户耳朵里还什么都没有
（秒表纪律回第 01 课 5.5 节）：

$$
\operatorname{TTFA}
=t_{\text{first PCM}}-t_{\text{request}}.
$$

实时因子为

$$
\operatorname{RTF}
=\frac{\text{generation wall time}}
{\text{generated audio duration}}.
$$

`RTF < 1` 表示平均生成吞吐快于播放速度，是持续播放的必要条件，但不能保证没有卡顿：
短时 jitter（抖动）仍可能让某些 frame 或 PCM chunk 错过 deadline。可接受 TTFA 则
取决于产品。实验前应写明目标阈值；没有产品要求时，只报告相对 baseline 变化。
流式稳定性还要结合下面的 MissRate、frame interval p95 和播放 buffer underflow
（播放器缓冲吃空）次数判断。

若 Mimi 配置的 frame rate 为 $f_c=12.5$ Hz，单帧周期是

$$
D_{\text{frame}}=\frac{1}{f_c}=80\ \text{ms}.
$$

记录 frame-ready interval 超过该周期的比例：

$$
\operatorname{MissRate}
=\frac{
\#\{\Delta t_{\text{frame}}>D_{\text{frame}}\}
}{
\#\{\text{generated frames}\}
}.
$$

这仍不是完整 TTFA；codec chunking、PCM queue 和播放缓冲会继续增加等待。

### 28. 用 Pareto 条件选择配置

不要在看到结果后临时设定 WER、UTMOS 和 TTFA 的加权总分——权重是看完结果凑的，
结论就成了循环论证。先分别绘制 WER–TTFA、human CMOS–RTF（CMOS：盲听对比打分，
两版同句对比给相对分），以及 `q4:q7` PPL–inner serial depth 三组关系。

按开头定义的支配关系筛掉全面落后的配置。互不支配的配置应同时保留，再根据预先声明的
延迟或质量预算选择。

选择规则在 test 前写好：

1. 如果目标是实时对话，先淘汰无法持续实时生成的配置；
2. 在产品预先声明的延迟预算内，比较内容和盲听结果；
3. 若两个点互不支配，保留 Pareto，不强行宣布唯一 winner；
4. teacher-forced CE 只用于解释，不覆盖 free-running 与人评结论。

## 根据症状定位失败环节

### 29. 失败诊断表

调试顺序和第 01 课一样：从最便宜、最可观测的证据查起，先索引后参数，先 schedule
后学习率。

| 你看到的现象 | 第一怀疑 | 先看什么证据 | 下一步 |
|---|---|---|---|
| $q_0$ loss 降，$q_7$ 完全不降 | label shift 或 head/mask 错 | per-q valid count、3×2 trace | 修索引，不调参 |
| 每隔固定帧出现 click | reconstruction 对角线错位 | generated code heatmap、frame index log | 对照 toy schedule |
| teacher-forced 很好，free-running 迅速坏 | exposure cascade | TF/sample/corruption 三条曲线 | group dropout 或统一 scheduled sampling |
| C 的后层 PPL 降，波形无差别 | 后层改进感知不敏感 | q-wise replace、盲听 | 接受负结论或换感知目标 |
| WER 下降但声音更金属 | coarse 内容改善，细节或 sampler 退化 | 分层 ablation、频谱、ASR transcript | 查 group 1 与采样 |
| 音质改善但 TTFA 激增 | depth 串行或 kernel launch 过多 | temporal/depth/Mimi profiler | batch heads、grouped depth |
| B 异常优于 A/C | 训练预算或 schedule 暴露不公平 | frames/update、params、FLOPs ledger | 重做配平 |
| 三臂结论随 temperature 反转 | sampler 主导 | greedy 与固定 sweep | 预注册共同 sampler |
| stop 八路不同步 | 独立 stop head 或权重效应 | stop position by q | 单独研究 frame-level stop |
| speaker prompt 在新 schedule 失效 | condition slot 被移动 | speaker marker position trace | 将 condition 与 schedule 解耦 |
| C 一开始训练震荡 | 新 depth block 尺度不匹配 | activation RMS、grad norm | 调初始化或 residual scale |
| 只有超长句崩溃 | temporal memory 而非帧内 depth | 按长度分桶、KV trace | 不要把问题归给 RVQ |

每次诊断都要记录现象、可观察证据和修改项。仅记录笼统原因无法用于复查。

### 30. 每个失败 case 要留下什么

为每个样例保存：

```text
cases/<case_id>/
  input.json
  target_text.txt
  target_codes.npy
  arm_a.wav
  arm_b.wav
  arm_c.wav
  asr.json
  codes_generated_{a,b,c}.npy
  per_q_metrics.json
  stop_positions.json
  frame_timeline.json
  speaker_metrics.json
  human_notes.md
```

`human_notes.md` 要记录差异出现的时间范围、对应词、错误类型和是否能盲辨，不能只写
总体偏好。优先检查：

- 三臂 WER 相同但听感差异最大的样例；
- 后层 PPL 改善最大但听感不变的样例；
- TTFA 最差的样例；
- 长句后半段崩溃的样例；
- stop 分歧最大的样例。

## 可直接运行的配置与产物目录

### 31. 配置骨架

下面是**本课设计**，不是 MiniMind-O 官方训练配置。字段名要按你的实现调整：

```yaml
experiment: lesson04_talker_topology_mechanism
upstream_sha: a10fa6c148ed274d66f96dc119689e93e01be823
base_checkpoint: baseline-v1
talker_init: neutral_shared_seed

frozen:
  thinker: true
  bridge: true
  mimi: true

codec:
  name: mimi
  codebooks: 8
  vocab_size: 2112
  canonical_codes_only: true

talker:
  temporal_layers: 4
  hidden_size: 768
  topology: diagonal  # diagonal | parallel | grouped_depth
  grouped_depth:
    groups: [[0, 1, 2, 3], [4, 5, 6, 7]]
    layers: 1
    hidden_size: 384

training:
  compare_by: valid_audio_frames
  updates: 30000
  seeds: [42, 43, 44]
  scheduled_sampling: 0.0
  stop_weight: 10.0

eval:
  greedy: true
  sampled:
    temperature: fixed-before-test
    top_k: fixed-before-test
    seeds: [1001, 1002, 1003]
```

`updates: 30000` 和 depth hidden 384 是 pilot 初始值，不代表最优配置。根据 10k
pilot 的收敛和显存结果调整后，冻结 standard 配置；test 期间不再调参。

### 32. 参数与计算账本

三臂至少记录：

- trainable parameters；
- total loaded parameters；
- 每个完整 audio frame 的 active FLOPs（浮点运算次数）；
- 每帧 outer temporal forward 次数；
- 每帧 inner serial depth；
- 每个 optimizer update 的有效 audio frame 数；
- frames/s；
- GPU-hours；
- peak allocated HBM（显存占用峰值）。

参数量相近不代表 active compute 相近，FLOPs 相近也不代表墙钟延迟相近。Depth
decoder 的小矩阵和 kernel launch（GPU 每启动一个小算子都有固定开销）可能在理论
FLOPs 很低时仍显著拉高 TTFA，所以参数、计算和墙钟时间三份账都要保留。

### 33. 完成标准

本课不设统一的 WER 改善幅度。满足以下条件时，实验结果才足以支持结论：

- 3×2 toy schedule 与 frame reconstruction 单测通过；
- legacy wrapper 在固定输入上与官方路径等价；
- 三臂都能过拟合同一 128 样本，或失败被定位到具体实现证据；
- 主比较没有混用数据、sampler、stop 权重或有效 frame 预算；
- per-q teacher-forced 与 free-running 指标齐全；
- TTFA 测到第一段 PCM，RTF 和分段 profiler 可复查；
- test 在配置冻结后只运行一次；
- 每个聚合结论都能落到可听的 case；
- 若没有 arm 支配 baseline，报告负结论，不得改写成功条件。

建议交付目录：

```text
artifacts/lesson04/
  source.lock
  data_manifest.jsonl
  configs/{arm_a,arm_b,arm_c}.yaml
  schedule_tests/
  checkpoints/
  metrics/per_q.jsonl
  metrics/system.jsonl
  profiler/
  plots/pareto/
  cases/
  listening_test/
  report.md
```

## 前沿对照与改造方向

本课的三种拓扑在公开系统里都有真身，可以按串行深度从高到低排一列。
[VALL-E](https://arxiv.org/abs/2301.02111) 用两段式：第一个码本整句自回归，其余
码本由非自回归模型并行补全——要等粗层整句生成完才补细层，天生离线。
[MusicGen](https://arxiv.org/abs/2306.05284) 系统比较过 delay、parallel、flattening
几种 codebook pattern：flattening 把各路码本全部展平串行，是精确的链式分解但步数
按码本数成倍增长；论文结论是 delay pattern 用单阶段模型就能接近它的质量——
MiniMind-O 官方的 diagonal 正是这条路线。[Moshi](https://arxiv.org/abs/2410.00037)
走的是本课 C 臂的完整版：大 Temporal Transformer 每帧只算一次，小 Depth
Transformer 在帧内逐码本自回归展开，两个时钟彻底分开，还保持流式——每个 temporal
step 结束就能凑出完整一帧；它用的 codec 正是 Mimi 本尊。
[SoundStorm](https://arxiv.org/abs/2305.09636) 换了思路，用 mask-and-refine 的并行
迭代精化替代自回归，离线批量生成很合适，面对流式 frame deadline 时迭代轮数本身
就是延迟。第 01 课引用过的 [Qwen2.5-Omni 技术报告](https://arxiv.org/abs/2503.20215)
同样采用 Thinker-Talker 分工，Talker 自回归产语音 token，再交给滑动窗口受限的
流式解码器还原波形，用窗口截断未来依赖来压首包延迟；其 Talker 帧内码本调度的
实现细节，技术报告公开有限。

规模差距明显：前沿系统的 temporal backbone 是数十亿参数，我们的
Talker 只有 4 层 768 维——这部分砸钱能缩小。但码本结构和帧率是同级的（Moshi 同样
是 Mimi 的 8 路 12.5 Hz），而"条件分解、warmup、串行深度怎么换"这道题在 26M 和
几十亿参数上是同一道数学题，本课的三臂设计足以在缩小版上把它做成有对照的结论。
真正缺的机制是前沿系统把文本流和多路音频流对齐在同一时间轴上联合建模（Moshi 的
Inner Monologue），那要等第二幕把流式输入打通之后才有条件谈。



1. **full 8-depth decoder（研究附录 B.1 的正式版）。** 把 C 臂的两组 depth 改成
   帧内逐码本 8 步自回归：在 `CodeSchedule` 的 grouped_depth 实现里把 `groups`
   换成 8 个单元素组，depth block 循环 8 次，每次读取本帧已采样 code 的 embedding。
   预算：与 C 臂相同（10k pilot、30000 updates、seed 42/43，单卡可跑，profiler
   仍占 GPU 7）。预期：`q4:q7` 的 teacher-forced NLL 相对 C 再降，每帧墙钟时间与
   TTFA 明显上升，kernel launch 计数成倍增加。失败判定：free-running WER 和盲听
   相对 C 无差异——把负结论写进 Pareto 报告，说明两组近似在本规模已经够用。
2. **Δ=2 的对角延迟。** 只改 schedule 常数：$s(f,q)=f+2q$。先在页面调度器里把
   `Δ / layer` 设为 2 单步验证，再照第 17 节接口新增一个 diagonal 变体训练 A' 臂，
   同步修改 `stream_generate` 的重组索引。预算：与 A 臂相同的 pilot 预算，一张卡。
   预期：按第 11 节的 `step` 计数，首帧完整时刻从 `step == 8` 推迟到 `step == 15`，
   warmup 接近翻倍；深层码本能看到更多浅层历史，后层 NLL 可能小幅下降。失败判定：
   质量指标与 A 无差异，则多付的 warmup 白花，保持 Δ=1 并记录该负结论。
3. **VALL-E 式离线参考臂。** 新增一个非流式 `CodeSchedule` 实现：$q_0$ 整句自回归，
   $q_{1:7}$ 条件于完整 $q_0$ 序列一次并行输出，复用臂 B 的 head 和 block。预算：
   与 B 臂相同。预期：该臂没有可用的 TTFA（必须等整句），但给出本规模下"看全粗层
   再补细层"的质量参考线，用来标定 A/B/C 与它的差距有多少来自条件分解。失败判定：
   该臂质量不高于 C，说明瓶颈在 temporal 容量而非帧内条件——这本身就是值得写进
   报告的结论。

MusicGen 论文"delay 接近 flattening、parallel 明显更差"的排序，
对应本课 A 臂对 B 臂：预期同方向——B 首帧等待最短，但后层码本 NLL 和盲听落后于
A；若 B 全面不输 A，先回第 29 节诊断表查训练预算配平（frames/update、参数量），
再怀疑结论本身。Moshi"帧内深度展开换质量、每帧一次 temporal 前向保流式"的方向，
对应 C 臂对 B 臂：预期 C 的 `q4:q7` teacher-forced NLL 更低；free-running 是否
兑现，正是本课主实验要回答的问题，复现不出同方向趋势时优先检查第 19 节的两种
前向是否都记录了。

## 必读论文与研究附录

### A. 论文与对应的源码检查

每篇材料带着能在自己产物里指认答案的问题去读；答不上来就回对应章节补查。

#### MusicGen：Codebook Delay Pattern

- [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284)
- [AudioCraft 官方实现](https://github.com/facebookresearch/audiocraft)

重点读 codebook pattern modeling。读完后回答：

1. delay、parallel、flattening 各自改变了条件依赖、序列长度和首帧等待中的哪一项？
   对照你在第 13-15 节网格里单步跑出的三种布局，逐项填成一张三行三列的表；
2. 为什么 delay steady state 可以每 step 产出完整 frame？用第 11 节的重组公式
   `audio_codes[i][step - 7 + i]` 在自己的 trace 里验证一次；
3. pattern provider 应该属于数据集、模型，还是一个独立 schedule object？对照
   第 17 节的 `CodeSchedule` 接口写出你的划分理由。

#### Moshi：Temporal Transformer 与 Depth Transformer

- [Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/abs/2410.00037)
- [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi)

重点读 Temporal Transformer、Depth Transformer 和 streaming generation。读完后回答：

1. depth model 为什么没有把大 temporal sequence 扩大八倍？先算一笔账：若按
   flattening 展平，12.5 Hz、8 码本的序列每秒要多少个位置？
2. 训练时 depth input 与推理时 sampled input 有何不同？对照第 8 节的
   $\Delta_{\text{TF}}/\Delta_{\text{free}}$ 和第 19 节记录的两种前向；
3. MiniMind-O 的 bridge 时钟与 Moshi 的多流时钟并不完全相同，哪些设计不能直接
   照搬？把答案写进第 21 节机制 track 的报告。

#### SoundStorm：Confidence-based Parallel Decoding

- [SoundStorm](https://arxiv.org/abs/2305.09636)

重点读 confidence-based parallel decoding，并回答以下实现问题：

1. mask-and-refine 需要几轮才能稳定？每一轮是一次全序列前向，轮数就是延迟；
2. 它适合离线生成，还是能满足严格的流式 frame deadline？用第 27 节的 MissRate
   语言回答；
3. 若只 refinement 后层 code，内容与音质的收益如何拆开测？对照第 26 节要求保留的
   两类样例设计实验。

#### VALL-E：粗层 AR 与细层 NAR

- [Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers](https://arxiv.org/abs/2301.02111)

思考 coarse/semantic 与 fine/acoustic token 的职责划分。带着第 4 节的提醒去读：
RVQ 各层分工是训出来的，没有天生的语义/声学分界。不要把 VALL-E 的 codec
层级直接等同于 Mimi 的每个 codebook；先用 q-wise ablation（逐路替换消融）验证。

#### Multi-token Prediction：与多码本预测的差异

- [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)

这篇第 01 课出现过，当时的问题是"MiniMind-O 的 8 个 head 预测什么"。这次把答案
补全：通用 MTP 预测同一词表、同一序列上的未来 token；本课预测同一音频帧的不同 RVQ
层。两者都使用多个 head，但条件结构、词表语义和解码重组不同。读完后用两列表格逐项
比较这三点。

### B. 后续研究问题

主三臂完成后只选择一个扩展，并作为独立实验记录：

1. full 8-depth Transformer 与 2-group depth 的质量、kernel launch 和 TTFA 差异；
2. $q_0$ speculative generation（先便宜地猜、再统一验证的推测式生成），$q_{1:7}$
   并行 verifier；
3. frame-level unified stop head，替代八路独立 stop；
4. 根据 q-wise waveform ablation 学习感知 loss weight；
5. 只对后四层做 SoundStorm 式 refinement；
6. depth 条件使用 sampled embedding、soft distribution 或 straight-through
   estimator（前向照常离散采样、反向近似当作恒等映射的求梯度技巧）；
7. 不同 Thinker bridge layer 是否改变 topology 的相对收益；
8. 将[第 03 课](03_audio_codec.md)的 challenger codec 接入同一 `CodeSchedule`，
   检验结论能否跨 codec。

### C. 完成检查：六个问题

1. RVQ 的后层为什么在统计上依赖前层？
2. diagonal delay 如何把 frame-local depth 搬到 temporal history？
3. 当前 `TalkerHead` 为什么仍然是“同一步八路并行”？
4. 3×2 toy case 中，frame 0 如何从三条对角位置重组？
5. 为什么 teacher-forced PPL 改善不能证明 free-running 音频改善？
6. 为什么 TTFA 必须量到 PCM，而不能停在第一个 codec logit？

答案必须同时引用源码位置、公式中的变量和一条真实 case 的 trace。三类证据缺少任一项，
都需要返回对应章节补查。

六个问题都能带齐三类证据回答，第一幕就算真正收官：一台每个零件都被你拆开、测量、
留过证据的回合制 omni 系统。下一课动第二幕第一刀：[第 05 课](05_streaming_listener.md)
把"整段录完再送"的听，改成随到随编码的流水线——你在本课测出的 warmup 步数和串行
深度，会直接进入那边的延迟预算表。
