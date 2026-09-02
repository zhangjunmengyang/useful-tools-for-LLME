---
id: 46_omni_serving
title: "多模态推理调度"
summary: "理解、语言生成、flow 采样和动作专家为什么不能捕获进同一条 CUDA graph，stage graph 怎样按阶段组批并隔离 KV？"
unit: omni-ops
play_tools: []
checkpoints:
  - "能画出理解编码、语言 decode、flow / 动作专家三条不同的 batch 维，并说明 CUDA graph 锁住了哪一类形状。"
  - "能对两条变长视觉请求写出 padding mask，使无效 patch 不进入有效 token 计数和注意力分母。"
  - "能区分合法的跨模态条件传递和错误的 KV 页别名。"
  - "能按 vLLM-Omni 的 stage graph 把 Qwen-Omni 式 Thinker–Talker–Vocoder 拆开调度，并对照第 13 课的训练并行。"
---

# 第 46 课：用 stage graph 调度多模态推理

> 类型：Omni 服务调度，推理图与变长视觉批处理，不改训练并行<br>
> 建议周期：阅读约 70 分钟；浏览器三条请求实验约 10 分钟；CPU padding 实验数分钟<br>
> 硬件：无 GPU 可完成本课阅读、教学模拟与 CPU 机制实验。复现 vLLM-Omni 论文数字需要双卡 80GB 与对应 Qwen-Omni 权重<br>
> 产物：一张 stage graph、padding mask 与有效 token 比、三条请求的 fused CUDA graph 对照记录<br>
> 独立性：对照 [第 13 课](13_distributed_8gpu.md) 的训练并行，本课只处理推理调度；统一理解与生成对照 [第 20 课](20_unified_understanding_generation.md)。本课不改 MiniMind-O 训练脚本

## 1. 理解和生成为什么不能挤进同一条 CUDA graph

前两波课把模型本身越装越多。Thinker 会看图、会听、会写字；Talker 会吐 codec；[第 20 课](20_unified_understanding_generation.md) 又给生成侧接上 flow matching（流匹配：沿一条从噪声到数据的路径回归速度，再按时间步积分出图）；[第 28 课](28_flow_matching_vla.md) 把同一套速度场接到动作专家，一次积出一段连续动作块。训练侧，[第 13 课](13_distributed_8gpu.md) 已经能把一份优化语义摊到八张卡上，对账 loss 和梯度。

推理侧还停在“调用一次 `generate()`”。三条同时进来的请求可以长这样：

1. 纯文本：16 个字，0 个视觉 patch（图像小方块，视觉编码器的最小输入单位），不走动作专家。
2. 带图理解：十几张到几十张 patch，再接一段问句，只要文本答案。
3. 带动作专家：既有图，又要跑若干积分步，吐出形状为 $H\times d$ 的动作块。

若把这三条请求连同理解编码器、语言 decode、flow / 动作专家一起捕获成一条 CUDA graph（把一段 GPU 核函数启动记录下来、下次按同一形状重放的机制），图会要求：视觉长度取三者最大值，文本长度取三者最大值，动作积分步也取三者最大值。纯文本请求于是带着一长串空 patch 和空动作步上场。更糟的情况是，捕获时只准备了一张 KV 页表（key-value cache 的物理页：注意力已经算过的键和值，解码时反复读），动作专家读到语言 decode 的页，或两条请求共用同一段物理 KV。浏览器 Lab 里把这件事做成可点的夹具：先预测，再捕获，必须看见 padding 浪费或错误共享 KV。

类比到此结束。类比失效处：厨房里“把三道菜放进同一个蒸箱”至少还共享温度这一个控制量；这里连循环次数都不共享。理解编码往往一遍过完整视觉序列；语言 decode 一步一个 token，KV 随步增长；flow 采样沿时间 $t$ 积分，步数在捕获前就必须定死。CUDA graph 能省的是同一形状上的 CPU launch 开销，省不掉“三种控制流”。vLLM-Omni（[arXiv:2602.02204](https://arxiv.org/abs/2602.02204)）的回答是 stage graph：节点是可独立服务的阶段，边是用户写的数据变换，每个阶段自己组 batch、自己管 KV、自己分加速器。

本课改的是推理调度协议，不是再训一个 Omni 模型，也不重做第 13 课的 FSDP / EP / CP。要验证的结果很具体：两条变长视觉请求组 batch 时，有效 token 数必须等于真实视觉长度加文本长度，padding 位置不得进入该计数；错误 mask 会把无效 key 的注意力质量变成正数；把纯文本和动作专家硬塞进同一静态形状会继续拉低有效比，并让动作页与 AR 页重叠。CPU 实验和浏览器 Lab 都是教学夹具。通过只说明你读懂了调度切分，不说明 Qwen3-Omni 的 91.4% JCT 降幅被你复现了。没有 80GB 双卡的读者，主路径是读完公式、跑通 CPU、在浏览器里合一次图；论文数字只许对照，不许当作自己的测量。

完成接口可以用三句话验收自己有没有读懂第 1 节。第一句：CUDA graph 锁的是核函数序列的形状与控制流，所以异构循环不能合录。第二句：stage graph 把理解、语言 decode、flow / 动作专家写成不同节点，每节点自己组批、自己的 KV。第三句：变长视觉的有效 token 只统计 mask 为 1 的位置，padding 进分母就算协议失败。三句都能用 CPU 数字或 Lab 屏幕指认，再往下读公式。指认不了，先回头把默认三请求表抄到纸上，把 116 和 216 除一遍。

本课术语：

| 术语 | 简要解释 |
|---|---|
| CUDA graph | 把一段 GPU 核函数启动记录成图，重放时跳过逐个 launch；形状和控制流在捕获时锁死 |
| stage graph | 把异构模型拆成节点（阶段）和边（数据变换），每阶段独立调度 |
| KV cache | 注意力里已经算过的 key / value，解码后续 token 时反复读取 |
| padding | 把短序列补到和长序列一样长，便于写成规则张量 |
| padding mask | 0/1 表，标出哪些位置是补上去的、不算数 |
| 有效 token 比 $\rho$ | 真正有内容的位置占 padded 张量的比例 |
| JCT | job completion time，从提交请求到整单完成的时间 |
| RTF | real-time factor，端到端处理时间除以生成音频时长；小于 1 才能实时出声 |
| TPS | tokens per second，每秒生成的 token 数；Thinker 和 Talker 要分开报 |
| connector | 阶段之间搬 hidden state、embedding、音频或图像张量的通道 |
| continuous batching | 迭代级组批：每步结束可以有请求进出，不必等整条序列结束 |
| chunked prefill | 把长 prompt 切成块，和 decode 混在同一次 iteration 里 |
| PagedAttention | 把 KV 按固定页存放，逻辑连续、物理不必连续 |
| 动作专家 | 专门从噪声积出连续动作块的小网络，循环在时间步上，不在文本 token 上 |
| 捕获范围 | CUDA graph 实际录下来的核函数集合；范围一旦跨过两个异构循环，形状就必须取全局 max |
| 页别名 | 两个阶段或两条请求把同一物理 KV 页当成自己的历史来读 |

把默认的三条请求写成一张表，后面 Lab 和手算都认这组数。带图请求的 patch 数、动作步可以在浏览器里拉动；表里写的是默认值。

| 请求 | 视觉 $v$ | 文本 $t$ | 动作步 $N$ | 真正需要的阶段 |
|---|---|---|---|---|
| 纯文本 | 0 | 16 | 0 | 语言 decode |
| 带图理解 | 48 | 12 | 0 | 视觉编码，语言 decode |
| 带动作专家 | 24 | 8 | 8 | 视觉编码，语言 decode，动作专家 |

合图时形状锁成 $v=48$、$t=16$、$N=8$，每条请求占用 $48+16+8=72$ 个槽，三条共 216 个槽。真正有内容的位置是 $16+60+40=116$。有效比 $\rho=116/216\approx 0.537$。纯文本一行为别人付了 48 个空 patch 加 8 个空动作步。stage graph 则让纯文本根本不进编码器，动作专家只跑第三条，有效比回到按阶段计算的值。CPU 夹具把同一逻辑缩成更小的数字：两条视觉请求 $(3,6)$ 与 $(9,2)$，便于手算；第三条纯文本再缩成 $(0,4)$，用来证明“再塞一个短请求，$\rho$ 继续掉”。两套数字单位相同，绝对值不要兑。

CUDA graph 在单阶段、定长 decode 上仍然有用。vLLM-Omni 给 MiMo-Audio 打开 execution-graph compilation 之后，RTF 从 0.60 降到 0.12。那张图录的是音频生成阶段内部的核函数序列，不是理解加生成的联图。本课反复强调的边界是捕获范围，不是“永远不要用 CUDA graph”。

## 2. 本课解决的问题

当前课程能画出 Thinker–Talker、能写出 flow 速度场，也能在八卡上把训练语义对上账。缺的是推理时怎么把这些异构前向变成可服务的系统。Hugging Face Transformers 式的实现把编码器、Thinker、Talker、Vocoder 写进同一次 `generate()`，vLLM-Omni 论文 §2.2 指出：这种写法用不上 continuous batching 和 chunked prefill，也不能按阶段分配加速器。本课把该诊断收成四个可证伪命题。

1. 存在一组三条请求（纯文本、带图、带动作专家），使得捕获成一条静态 CUDA graph 后，有效 token 比 $\rho$ 严格低于按 stage 分别组批；下降来自 padding，不来自模型权重。
2. 同一组请求上，若动作专家与语言 decode 共用从 0 号开始的物理 KV 页，则存在被双方同时占用的页；按 stage 隔离后该交集为空。
3. 两条变长视觉请求按 $\max v$、$\max t$ 填充后，有效 token 计数必须等于各请求真实长度之和；任一 padding 下标进入该计数，协议失败。
4. 错误地把 padding 位置的 mask 置 1，均匀 logits 下无效 key 的注意力质量等于无效位数除以 padded 长度，严格大于 0。

四条都通过，只能说明调度协议可审计。它们不能说明 vLLM-Omni 在 Qwen3-Omni 上相对 Transformers 基线的 91.4% JCT 降幅可以搬到你的笔记本，也不能说明一条 CUDA graph 在纯文本、定长 decode 场景里没有价值。定长、单阶段、反复重放的 decode 核，正是 graph compilation 该上场的地方；本课反对的是跨阶段合图。

把四个命题和夹具的对应写死，避免验收时互相顶替。命题 1 主要靠 Lab：fused 的 $\rho$ 必须低于 stage，并且纯文本空槽随 $\max v$ 上升。命题 2 两边都测：Lab 的 KV 别名开关，CPU 的页号交集。命题 3 只靠 CPU：有效计数 20，padding 下标 3–8 不在 A 的有效集合。命题 4 只靠 CPU：错误质量 0.4，正确质量 0。有人用 Lab 的百分比代替 0.4，或者用 91.4% 代替 $\rho$ 下降，属于用错尺子。四个命题全部是调度语义，没有一个需要 30B 权重。权重只出现在论文数字列，那一列永远单独放。

本课明确不解决的问题：不训练 MiniMind-O；不实现 vLLM 的 C++/CUDA 内核；不比较 FSDP2 和 ZeRO-3 的显存切法（那是第 13 课）；不比较 Janus / Show-o / BAGEL 的训练目标（那是第 20 课）；不把屏幕点击和机械臂写成同一种身体（那是第 32 课）。本课只引入推理图。

和相邻课的边界再钉死一次。[第 13 课](13_distributed_8gpu.md) 的单位是训练 step：参数怎么分片、专家怎么 all-to-all、长序列怎么切 context，验收是 loss / 梯度 / 一步更新与单进程参考一致。本课的单位是请求和阶段 iteration：哪条请求进哪个引擎、KV 页归谁、padding 算不算有效 token。第 13 课的 CUDA 设备是训练网格里的 rank；本课的 CUDA graph 是推理捕获。两者都碰 GPU，切的东西不同。 [第 20 课](20_unified_understanding_generation.md) 关心理解 CE 和 flow 速度 MSE 会不会在同一组参数里互相干扰；本课假定那些头已经训好，问的是它们能不能共享一次 launch。 [第 08 课](08_dynamic_vision.md) 已经要求按视觉 token 数分桶、大图小图不要混 batch；本课把同一纪律从训练搬到服务，并加上“无效 patch 不得进入注意力分母”。 [第 07 课](07_full_duplex_routing.md) 的 CONTINUE / PAUSE / REPLAN 是会话层状态机；本课的流式 stage 输出是引擎之间交部分结果，两套时钟以后在第 48 课才会并到一张表上，本课不抢那道题。

[第 19 课](19_capstone_thinker_talker.md) 把 Thinker 与 Talker 接成可演示的毕业系统，hidden-state bridge 已经存在；本课问的是这座桥在服务进程里是队列加两份 KV，还是仍缩在一次 `generate()` 里。第 19 课可以单进程通过 golden case；本课在同样的 golden case 上要求日志出现两份页表。 [第 28 课](28_flow_matching_vla.md) 给出动作专家的时间步循环；本课禁止把 $N_a$ 编进 token 轴。阅读顺序建议：先有 13 的“切开不能改语义”、20 或 28 的“生成循环不同于 AR”，再读本课。缺 28 也可以：把动作列想成 DiT Vocoder 的去噪步，公式同构，只是张量从波形换成 $H\times d$。

## 3. 开始前需要准备什么

本课没有 MiniMind-O 训练步骤。开始前把上游事实和本课约定分开写进实验记录。事实来自已打开的 arXiv 页面；约定是本课夹具的长度和门禁。两者混写会导致有人把 20/30 当成 Qwen3-Omni 的实测有效比。

CPU 夹具的长度一旦写下就不要改：A 视觉 3 文本 6，B 视觉 9 文本 2，第三人视觉 0 文本 4，动作步 8。改长度可以当作自己的练习，但本课标准答案和 `lesson_46.py` 按这组数锁定。浏览器 Lab 的 48/16/8 是另一组，用来让条纹看得见。手算前先声明用的是哪一组。

**上游事实（打开过的页面，不是口口相传）：**

- vLLM-Omni：[arXiv:2602.02204](https://arxiv.org/abs/2602.02204)，HTML：[arXiv HTML v1](https://arxiv.org/html/2602.02204)。any-to-any 模型的全拆分服务系统；stage graph；Qwen3-Omni 上 JCT 最多降 91.4%。代码仓库在论文摘要给出：[github.com/vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni)。
- vLLM / PagedAttention：[arXiv:2309.06180](https://arxiv.org/abs/2309.06180)。KV 按页管理；相对 FasterTransformer / Orca 吞吐 2–4×；现有系统实测只有 20.4%–38.2% 的 KV 内存真正存了 token 状态。
- SARATHI chunked prefill：[arXiv:2308.16369](https://arxiv.org/abs/2308.16369)。把 prefill 切块，和 decode 拼成 decode-maximal batch。vLLM-Omni §3.3 把该技术继承到每个 AR stage。
- DistServe：[arXiv:2401.09670](https://arxiv.org/abs/2401.09670)。把 prefill 和 decode 拆到不同 GPU。vLLM-Omni 的统一连接器与 EPD（encode–prefill–decode）拆分兼容，但 EPD 仍是单模型阶段内部的拆法，不是理解与 flow 的合图。
- Qwen2.5-Omni / Qwen3-Omni 的 Thinker–Talker–Vocoder 结构以 vLLM-Omni 论文 §2.1、§3.2 为准。Thinker 规模：Qwen2.5-Omni 7B，Qwen3-Omni 30B（论文 §4.2）。Vocoder：Qwen2.5-Omni 用 DiT，Qwen3-Omni 用轻量 CNN（论文脚注）。其技术报告未在本课展开的训练细节，不编造。

**本课约定：**

- CPU 实验文件：`experiments/src/learn_omni_experiments/lessons/lesson_46.py`。编排者登记进 `registry.py` 之前，可以直接导入该模块跑 `run()`；登记之后用仓库脚本跑第 46 课。
- 浏览器 Lab：`Lesson46ServeGraphLab`。标有“教学模拟”，三条请求和 KV 页由夹具生成。
- 不把 Lab 的有效 token 比、CPU 的 20/30、论文的 91.4% 写进同一列当“本课复现结果”。
- 引用 vLLM-Omni 数字时必须出现模型名、基线（Transformers 或 Diffusers 或原实现）和指标名（JCT / RTF / TPS）。缺一项就删掉该数字。

需要会的前置技能：矩阵形状；softmax 分母；Python 列表求和；第 13 课“并行之后语义不能变”的直觉；第 08 课“视觉 token 数随图变化”。不要求会写 CUDA 内核，不要求有 80GB 加速器。没有八卡也可以读本课；第 13 课提供对照语言，不是运行依赖。缺第 20 课时，把 flow 理解为“按时间步重复的前向”即可，四个命题不受影响。

读论文时准备一张空白的四列表：阶段名、动态尺寸、KV 是否增长、捕获是否允许。填完 Qwen-Omni 的 Thinker / Talker / Vocoder 三行，再填本课 Lab 的编码 / decode / 动作三行。两张表结构相同，数字不同。Thinker 的动态尺寸是输入 token（含视觉）加输出文本；Talker 是音频 token；Vocoder 是去噪步或 CNN 前向次数。Lab 的动作列对应 Vocoder / 动作专家这一类“时间步循环”。填表时若某一行的“捕获是否允许”写成是，必须同时写清该行的形状如何被静态化：定长 decode、分页 KV、固定积分步。三行都写“是”且形状各不相同，就是本课要拆的合图。

Qwen2.5-Omni 与 Qwen3-Omni 不要在笔记里混成一个“Qwen-Omni 数字”。Thinker 7B 对 30B，Vocoder DiT 对 CNN，相对加速 61% 量级对 91% 量级。本课引用时永远带代次。Ming-Omni、LongCat-Flash-Omni、Step-Audio、BAGEL 在论文 §2.1 作为结构例子出现，本课不引用它们未在 vLLM-Omni 实验节给出的吞吐数字。BAGEL 的 9.64 s / 11.12 s 可以引用，因为那是 §4.2 实测。LongCat 的 560B、Step-Audio 的 130B 只说明“AR 后面还会接一个专门解码器”，不说明你的调度器已经能服务它们。

注意力直觉只需到这一层：query $i$ 对 key $j$ 的权重是

$$
\alpha_{ij}=\frac{m_j\exp(s_{ij})}{\sum_k m_k\exp(s_{ik})}
$$

$m_j=0$ 的位置必须从分母消失。$s_{ij}=0$、$m$ 全 1 时，长度为 $L$ 的序列每个位置得 $1/L$；若其中 $P$ 个位置本该是 padding，错误 mask 会把 $P/L$ 的质量分给无效 key。不必先学 FlashAttention 也能完成本课。

硬件记录也要分开写。读课文和跑 CPU：笔记本即可。跑 Lab：现代浏览器。复现论文 §4.2：两张 80GB 加速器、24 CPU 核、192 GB 内存、vLLM 0.12.0，输入取 librispeech_asr / food101 / ucf101-subset 各前 100 条。那是论文测试机，不是本课验收机。没有那两张卡就不要填 JCT 格子。

## 4. 完成后应具备的能力

读完第 4 节，应能在不看屏幕的情况下写出两条视觉请求的 $\rho$ 和错误质量。写不出就先做第 8 节 Step 1–2，再回来勾能力清单。GPU 不是这一节的门槛。纸和计算器足够：20 除以 30 得三分之二，6 除以 15 得 0.4。这两个数会在 CPU `metrics` 里原样出现，对不上先查 mask 行，再查自己有没有把 padded 形状当成有效。

完成后，拿到任意 Omni / 统一模型的服务脚本，应能做以下检查：

1. 指出该脚本把哪些模块当成一个 stage：视觉编码、AR Thinker、AR Talker、DiT / flow、动作专家、Vocoder；拒绝“整个 `forward` 就是一个 stage”。
2. 对给定请求集合写出每阶段的 batch 维：视觉是 patch 数，语言 decode 是 token，flow 是积分步，动作块是 $H\times d$。
3. 计算 padded 形状下的 $\rho$，并指出哪条请求在为别人的 max 付费。
4. 画出 padding mask，核对无效下标与有效下标不相交、并集覆盖 $[0,L)$。
5. 用均匀 logits 手算错误 mask 的无效注意力质量，确认它等于无效位数除以 $L$。
6. 区分三种 KV 关系：同请求前缀共享（PagedAttention 合法）、跨阶段条件传递（Talker 读 Thinker hidden state）、物理页别名（本课判失败）。
7. 对照 vLLM-Omni Table 1：Qwen2.5-Omni 上 Thinker2Talker 共享内存 5.49 ms、Mooncake 8.28 ms，Talker2Vocoder 分别为 0.53 ms 和 3.34 ms；相对数十秒推理，连接器不是主因。
8. 对照 §4.2：Qwen3-Omni 视频输入平均 841.6 token、文本输出 150.9、音频输出 545.4；解释为什么 Talker 占掉大部分 JCT。
9. 在 Lab 里先提交预测，再捕获一条 CUDA graph；把 padding 浪费或 KV 别名写成合图效应，而不是“GPU 比较慢”。
10. 看到“我们把 Omni 编译成一条 CUDA graph，吞吐提升”时，能在 30 秒内问：形状有没有锁死、纯文本有没有垫视觉、动作专家有没有自己的页表。问不出就从讲稿里删掉该句。

额外准备两个口头计算，面试或报告被追问时用。第一问：A 视觉 3、B 视觉 9，文本分别 6 和 2，$\rho$ 是多少？答 $20/30$。第二问：错误 mask、均匀 logits，A 的无效注意力质量？答 $6/15=0.4$。两问都过，才说明 mask 公式进了手，而不是只记住“padding 不好”。再追问第三问：动作专家从页 0 写 8 页，AR 也从页 0 写，交集长度？答 8。这三问覆盖命题 3、4、2。命题 1 用 Lab 的 $\rho$ 下降覆盖。

## 5. 原理：边造边讲

八个机制，每个按同一节奏：为什么需要、怎么运转、精确定义、在公开实现里落在哪、怎么证明做对了。本课没有 MiniMind-O 服务运行时可改，代码落点改到 vLLM-Omni 论文给出的阶段函数名，以及 Hugging Face 式 `generate()` 这条反例。

### 5.1 CUDA graph 实际锁住了什么

为什么需要。GPU 上每一个核函数都要 CPU 参与 launch。短 decode 步、小 batch 时，launch 开销可以压过有效计算。CUDA graph 把一段已经跑通过的 launch 序列录下来，之后按同一套参数重放。vLLM-Omni 把这件事叫 runtime execution-graph compilation，并且只在阶段内部做：MiMo-Audio 在 SeedTTS 上，不开编译 RTF 0.60，打开后 0.12，相对原实现 1.39 达到 11.58×（§4.2）。数字支持“按阶段编译图”，不支持“把理解和生成录成一张图”。

怎么运转。捕获发生时，驱动记录的是核函数身份、grid / block、当时的张量地址和形状、以及这段记录里实际走了哪条控制流。重放时这些都得对得上。AR decode 每步输入一个新 token、KV 变长，要靠分页或预分配把形状维持成捕获时的样子。flow 采样每步改 $t$，循环次数 $N$ 必须在捕获前选好。视觉编码的 patch 数随图变。动作专家的张量秩是 $H\times d$，和 token 序列不是同一个轴。把这些录进同一张图，只剩两条路：全部 pad 到全局 max，或让不该出场的模块读别人的缓冲。

数学。记阶段 $s$ 在请求 $b$ 上的动态尺寸为 $n_{s,b}$。单阶段捕获要求

$$
\hat n_s=\max_b n_{s,b}
$$

跨阶段合图要求一个全局形状

$$
\hat n=\big(\max_{s,b}n_{s,b}^{(v)},\ \max_{s,b}n_{s,b}^{(t)},\ \max_{s,b}N_{s,b}^{\text{flow}}\big)
$$

对纯文本请求，$n^{(v)}=0$、$N^{\text{flow}}=0$，却仍按 $\hat n$ 执行。这就是 Lab 默认配置里纯文本空槽等于 $\max v+\max N$ 的来源。

代码落点。vLLM-Omni §3.3：每个引擎可以打开 chunked prefill 和 execution-graph compilation，继承 LLM 服务系统的收益。论文没有提供跨 Thinker 与 DiT 合图的 API，也没有报告这种合图的加速比。其技术报告未公开该细节，本课不补一个假接口。MiniMind-O 的 `MiniMindOmni.stream_generate` 把 Thinker 与 Talker 串在同一次生成里，属于论文 §2.2 描述的手工编排：先 `generate()` 出文本和 hidden state，再拼进 Talker，再交给 Vocoder。它能出声，但不能按阶段组批。

验证。Lab 把模式打到“一条 CUDA graph”时，三条请求的槽数必须相等，且等于三个 max 之和。若纯文本请求的槽数仍是 16，捕获没有锁形状，夹具失败。CPU 夹具把动作步当成单独的 padded 维：两条没有动作的请求各垫 8 个空步，`dummy_action_steps == 16`。

捕获和重放还可以再拆一层。捕获时设备必须按真实控制流跑一遍，驱动才能记住核函数顺序；重放时不再走 Python 里的 if / for。因此 Python 里写着“如果没有图就跳过 encoder”这句话，在图已经录好之后不会再被执行。跳过变成了“跑一个输入全是 padding 的 encoder”。服务日志如果只看 Python 分支覆盖率，会误以为纯文本没进视觉核；Nsight 时间线上那个核仍然在。本课把时间线当作最终证据：fused 模式下 encoder 核的 batch 维必须是 3，stage 模式下必须是 2。没有 profiler 时，用 Lab 的编码列 batch 数字代替，并写明教学模拟。

定长 AR decode 为什么可以录图，这里给一个够用的充分条件：本步 query 长度固定（通常每请求 1 个新 token），KV 用页表间接访问，页大小 $B_{\text{page}}$ 固定，参与本步的物理页数有上界。PagedAttention 满足后两项。变长视觉 prefill 不满足“query 长度固定”，除非 pad 到 $\max v$ 或切成定长 chunk。动作专家不满足“每步一个 token”：它每步更新整块 $H\times d$。充分条件缺任何一条，捕获范围就应停在该阶段的边界上。

### 5.2 Stage graph：异构前向变成可调度节点

为什么需要。any-to-any 模型不是“一个 Transformer 加几个头”那么整齐。vLLM-Omni §2.1 举了三类：Qwen-Omni 的双 AR（Thinker 出文本，Talker 出音频 token）再加 Vocoder；GLM-Image 的 9B AR 接 7B 单流 DiT；BAGEL 的 Mixture-of-Transformers 把理解和生成专家看成两个阶段。现有框架要么只会 AR decode（vLLM、SGLang），要么只会 DiT 去噪。开发者只好在框架外自己搬张量，于是 continuous batching 用不上，卡也没法按阶段分。

怎么运转。用户把模型写成有向图。节点实现两类函数：`forward`（这一阶段的 batched 计算）和 `preprocess`（每步开始时，用上游产物改输入）。边实现 transfer（只在阶段切换时调用一次，或按流式协议多次）。Qwen2.5-Omni 在论文 Figure 4 里的落地是三个节点：Thinker、Talker、Vocoder。论文点名的函数包括 `thinker_forward`、`talker_forward`、`dit_decode`、`mm_encode`、`process_input`、`Thinker2Talker`、`Talker2Vocoder`。多模态编码器可以单独成 stage，也可以并进 Thinker；该例并进 Thinker，以跟随 vLLM 对视觉输入的处理。Talker 的 `process_input` 在每一步 decode 都要把 Thinker hidden state 拼进当前输入，所以 preprocess 每 iteration 都跑；transfer 函数只在阶段边界调用。

数学。图 $G=(S,E)$。请求 $r$ 的活跃节点集合 $S(r)\subseteq S$。调度器维护每节点的队列 $Q_s$，只把 $s\in S(r)$ 的请求送进引擎 $s$。引擎 $s$ 的一次 iteration 计算

$$
Y_s=\mathrm{Forward}_s\big(\mathrm{Preprocess}_s(X_s,H_{<s})\big)
$$

其中 $H_{<s}$ 是上游经边 $e\in E$ 送来的中间量。请求级完成时间

$$
\mathrm{JCT}(r)=\max_{s\in S(r)}T_s^{\text{end}}(r)-T^{\text{submit}}(r)
$$

若下游允许流式输入，$\max$ 可以变成重叠：Vocoder 的 $T^{\text{start}}$ 可以早于 Talker 的 $T^{\text{end}}$。论文 §3.3 Streaming stage output 写的就是这件事。

代码落点。论文 §3.1：后端有 orchestrator 管请求和阶段；每阶段独立引擎；model runner 取 batched 请求，跑 preprocess，再跑一次 batched forward；统一 connector 搬中间数据。开发者视角（Figure 3b）是 preprocess 加 batched forward；用户视角（Figure 3c）是给不同阶段配并行策略和内存预算，不必改模型代码。本课 Lab 的“stage graph”模式把三条请求拆进编码 / decode / 动作三列，只是把这张图收成玩具尺寸。

验证。能画出 Qwen-Omni 三个节点、两条边，并且指出 Talker 的 preprocess 每步都跑、Thinker2Talker 只跑一次，这一项才算读懂。若有人把 Talker2Vocoder 画成每 token 都调用的边，和论文“they are only called once”冲突，除非启用了流式输出协议并明确改写边语义。流式是增量交数据，仍不是把 Vocoder 编进 Talker 的 CUDA graph。

把一条带语音输出的请求走一遍时间线，数字用论文视频任务的平均数。输入侧 841.6 个 token（含视频）进入 Thinker 的 prefill；Thinker 再 decode 约 150.9 个文本 token；Talker 为音频 decode 约 545.4 个 codec token；Vocoder 把 codec 变成波形。stage graph 上这是四段（若 encoder 并进 Thinker 则为三段）可以重叠的工作。合图若坚持“整单结束才返回”，Vocoder 的起始时刻不得早于 Talker 的结束时刻，$\Delta_{\text{overlap}}=0$。论文 §3.3 的 streaming stage output 就是把 Vocoder 的起始时刻前移到 Talker 产出首批 codec 之后。MiniMind-O 若要接到这张图上，对应关系是：Thinker 8 层对应 Thinker 节点，Talker 4 层对应 Talker 节点，Mimi 解码对应 Vocoder 节点；当前 `stream_generate` 把前两个节点写在同一个 Python 循环里，本课要做的第一刀是循环拆开，而不是先上 30B。

### 5.3 四类 stage 的 batch 维不同

为什么需要。组 batch 的资格是：同一段核函数、同一套形状语义。视觉编码吃的是 patch 序列；AR decode 吃的是“本步新 token + 历史 KV 页”；flow 吃的是整块 latent 加时间 $t$；动作专家吃的是 $H\times d$ 的动作块加 $t$。把它们在第 0 维拼成一个 batch，第 1 维用 padding 对齐，只是把不同单位的计数器接到同一根轴上。

怎么运转。四类尺寸：

| 阶段 | 动态尺寸 | 组批对象 | 捕获时必须静态的量 |
|---|---|---|---|
| 理解 / 视觉编码 | patch 数 $v_b$ | 带图请求 | $\max v$，或分页后的页数上界 |
| 语言 decode | 本步 token 数，通常每请求 1 | 所有要出字的请求 | KV 页大小、本步 query 长度 |
| flow / DiT | 积分步 $N$，空间 $H_g\times W_g$ | 要出图的请求 | $N$、分辨率、CFG 是否成对前向 |
| 动作专家 | 积分步 $N_a$，动作块 $H\times d$ | 要出动作的请求 | $N_a$、$H$、$d$ |

vLLM-Omni 给 AR 阶段用 vLLM 引擎，给 DiT 阶段用独立 diffusion 引擎（§3.3）。BAGEL 在 1024×1024、80GB 单卡上，T2I 的 JCT 从 23.12 s 降到 9.64 s（2.40×），I2I 从 41.39 s 降到 11.12 s（3.72×）。DiT 相对 Diffusers 总体 1.26×。这些加速发生在“图像生成阶段内部”，没有要求把 T2I 和纯文本 QA 写成同一条图。

数学。令请求 $b$ 的类型指示 $\tau_b\in\{\text{text},\text{vision},\text{action}\}$。阶段 $s$ 的合法 batch 是 $\{b:\tau_b\in A_s\}$，其中 $A_{\text{enc}}=\{\text{vision},\text{action}\}$，$A_{\text{ar}}$ 可以是全体，$A_{\text{flow}}=\{\text{action}\}$（本课把图像生成和动作专家都归到时间步循环一类，Lab 里动作用于第三人）。合图把 $A_s$ 强行改成全体，于是 $\text{text}\in A_{\text{enc}}$ 且 $\text{text}\in A_{\text{flow}}$。CPU 夹具里第三条请求视觉长度为 0，仍按 $\max v=9$ 占位，就是这条规则的数字版。

代码落点。论文 §3.3 对 Qwen3-Omni 的资源分配：Thinker 最大（30B），多分加速器内存；Talker 更小但更吃算力，可以少分内存、提高并行、多给加速器。这是“按阶段的计算特征分配”，前提是阶段已经拆开。若仍是一条 `generate()`，这两句话没有可执行的落点。

验证。Lab 在 stage 模式下，编码列只统计带图的 2 条，动作列 batch=1；fused 模式下三列都写 batch=3。切换模式而不改请求，batch 维必须跟着变。CPU 夹具检查 `fused_action_steps_are_a_separate_padded_dim`：动作 max 等于第三条的 8，前两条共垫 16 个空步。

同一 AR 阶段内部，prefill 和 decode 也已经是两种 batch 维。SARATHI 的办法是把长 prefill 切成定长 chunk，再和若干 decode 步拼成一次 iteration，让 decode 搭上 prefill 的计算饱和。LLaMA-13B 在 A6000 上 decode 吞吐最多 10×，端到端最多 1.33×。这仍然是同一个语言模型、同一套 KV 语义里的拼盘。多模态服务容易犯的错，是看见“prefill 可以和 decode 拼”，就推论“视觉 prefill 可以和动作积分拼”。拼得合法的前提是核函数序列相同。视觉 encoder 的卷积或 ViT 块、DiT 的 adaLN、AR 的 next-token 头，三套核不是同一序列。chunked prefill 可以出现在 Thinker 节点内部；它不能充当跨节点的胶水。

动作块的形状再写一次，避免和视觉 patch 混淆。[第 28 课](28_flow_matching_vla.md) 的动作专家输入是 $A\in\mathbb{R}^{H\times d}$，速度场 $v_\theta(a_t,t,x_{\text{vlm}})$，循环下标是积分时间，不是 token 下标。把 $H$ 个动作步 pad 进视觉序列，等于声称“第 $h$ 个动作维和第 $h$ 个 patch 参加同一层自注意力”。公式一经写出就知道这是错的：位置编码、causal mask、RoPE 轴全对不上。Lab 用第三种颜色标动作步，就是防止它和视觉条纹混成一种 padding。

### 5.4 KV：合法复用、条件传递、错误别名

为什么需要。KV 占的是加速器内存的大头。vLLM 论文 Figure 1：13B 模型在 40GB A100 上，权重约 65%，KV 约 30%，激活值很小；现有连续分配方案里真正存了 token 状态的 KV 只有 20.4%–38.2%。Omni 服务在此之上又多了两件事：视觉 token 的 KV 往往比文本长；Talker 每步还要读 Thinker 的 hidden state。不先分清“谁可以读谁的页”，PagedAttention 的前缀共享会被误用成跨模型共享。

怎么运转。三种关系必须分开记账。

合法的同模型共享。PagedAttention 把一条序列的 KV 切成块，逻辑块映射到物理块。平行采样时，prompt 的物理块引用计数大于 1，copy-on-write 只发生在开始分叉的那一页（vLLM §4.4）。这是同一套权重、同一套注意力公式、同一套位置编号。

合法的跨阶段条件。Qwen-Omni 的 Talker 并不去读 Thinker 的 KV 页。它读的是 Thinker 的 hidden state 和多模态 embedding，经 `process_input` 拼到自己的输入 embedding 上，并且每步都拼（vLLM-Omni §3.2）。Thinker 的 KV 仍留在 Thinker 引擎的 KV manager 里；Talker 引擎有自己的 scheduler、KV manager、model runner（§3.3）。连接器搬的是 embedding / hidden state / 音频张量，不是把两本页表合成一本。

错误别名。合图时只有一个物理池、一张从 0 编号的块表。语言 decode 把文本请求写进页 $[0,L_{\text{ar}})$，动作专家也从 0 开始写 $N_a$ 步的中间状态，交集 $[0,\min(L_{\text{ar}},N_a))$ 被双方占用。读出来的 key 不再对应“该请求、该阶段、该位置”的三元组。Lab 的 12 页表把前若干页标成“AR+动作”，就是这个交集。CPU 夹具里 fused 碰撞页等于 `range(action_max)`，stage 模式下动作页从 `padded_three` 之后开始，交集为空。

数学。记物理页集合 $\mathcal{P}$，阶段 $s$、请求 $b$ 的占用 $\mathcal{P}_{s,b}\subseteq\mathcal{P}$。隔离条件：

$$
\mathcal{P}_{s,b}\cap\mathcal{P}_{s',b'}=\emptyset
\quad\text{若 }(s,b)\neq(s',b')
$$

前缀共享是例外，必须满足 $s=s'$、权重相同、并且引用计数协议完整。条件传递不进入该公式：它搬运的是值 $H$，不是页号。合图失败判定写成交集非空：

$$
\big|\mathcal{P}_{\text{ar}}\cap\mathcal{P}_{\text{action}}\big|>0
$$

代码落点。vLLM-Omni §3.4：统一连接器从 vLLM 的 prefill–decode KV 传输推广而来，能搬 embedding、hidden state、音频或图像张量。单机用控制队列加共享内存；跨机用 Ray 编排，Mooncake 走 TCP 或 RDMA，控制面只传轻量元数据。连接器也处理阶段内部的 KV（prefill 到 decode）和 MM cache（encoder 到 prefill），并与 EPD 拆分兼容。EPD 仍要求 encode、prefill、decode 是同一语言模型流水线里的三段；它不能证明 Thinker 的 KV 页可以借给动作专家。

验证。CPU 检查 `fused_kv_aliases_action_and_ar_pages`。Lab 在 fused 捕获后 KV 别名必须为“是”，stage 模式下为“否”。若有人把 Talker 读 Thinker hidden state 写成“共享 KV”，口头纠正：共享的是条件向量，页表仍是两份。

条件向量有自己的字节账。设 Thinker hidden 为每层 $d_{\text{model}}$，Talker 每步要拼接的是一层（或一层投影后）长度为 $T_{\text{thinker}}$ 的向量。Qwen-Omni 的 Talker 在每一步 decode 都拼接，所以传输量随 Talker 步数线性增长，但连接器搬的是这份激活，不是 Thinker 全部层的 KV。OPT-13B 单个 token 的 KV 约 800 KB（vLLM 论文：2 × 5120 × 40 层 × 2 字节）。Omni 里视频 prefill 平均 841.6 个输入 token，若按同量级粗算，Thinker KV 本身就是数百 MB 量级；把它整本借给 Talker 或动作专家，既贵又语义错误。本课夹具不模拟 800 KB 这个数字，只模拟页号交集。引用 800 KB 时必须写“OPT-13B、vLLM 论文 §3，不是 Qwen3-Omni 实测”。Qwen3-Omni 单 token KV 字节数其技术报告未公开该细节，不要用 30B 去除硬套。

### 5.5 变长视觉批处理与 padding mask

为什么需要。[第 08 课](08_dynamic_vision.md) 已经说明：固定 256×256、固定 64 个视觉 token 会把发票小字糊掉。服务侧面对的是同一事实的批量版。请求 A 可能 3 个 patch，请求 B 可能 9 个。写成规则张量就要 pad 到 9。若把 pad 位置当成真实 patch 送进注意力，softmax 分母会被空位稀释，上下文向量被垃圾或零向量拉偏。训练里用 loss mask 跳过 padding；推理里没有 loss，只剩 attention mask 和有效 token 计数两道闸。

怎么运转。布局约定本课固定为：先视觉槽 $[0,V)$，再文本槽 $[V,V+T)$。请求 $b$ 的真实长度 $(v_b,t_b)$，batch 最大值 $(V,T)=( \max v_b,\max t_b)$。mask

$$
m_b[i]=
\begin{cases}
1 & 0\le i<v_b \text{ 或 } V\le i<V+t_b\\
0 & \text{其他}
\end{cases}
$$

CPU 夹具的两条请求：A 为 $(3,6)$，B 为 $(9,2)$，故 $V=9$、$T=6$、$L=15$。A 的 mask 是三节 1、六节 0（视觉 pad）、六节 1。B 的 mask 是九节 1、两节 1、四节 0（文本 pad）。A 的无效下标 $\{3,4,5,6,7,8\}$，B 的无效下标 $\{11,12,13,14\}$。有效下标与无效下标不相交，并集是 $\{0,\ldots,14\}$。

数学。有效计数只许走 mask：

$$
n_{\text{valid}}=\sum_b\sum_{i=0}^{L-1}m_b[i]=\sum_b(v_b+t_b)
$$

本课数字：$n_{\text{valid}}=(3+6)+(9+2)=20$。Padded 形状给出 $B\cdot L=2\times 15=30$。任何把 $n_{\text{valid}}$ 写成 30 的实现，等于宣布 padding 是内容。注意力侧，正确 mask 在均匀 logits 下给 A 的每个有效 key 质量 $1/9$，无效质量 0；错误 mask 给 15 个位置各 $1/15$，无效质量 $6/15=0.4$。若 pad 位置的 value 为 0，输出幅度被乘上 $9/15$；若 pad 位置残留了上一条请求的向量，输出直接串味。第二种比第一种更危险，因为它不报错、只改答案。

代码落点。公开 VLM 服务通常在 encoder 或 LLM 预处理里构造 `attention_mask` / `image_pad_mask`。vLLM 对多模态输入把 encoder 当作 LLM 阶段的一部分（论文脚注 4）。本课不绑定某一个文件名，因为仓库布局会变；验收只问：打印出来的有效下标集合是否与 $(v_b,t_b)$ 一致。MiniMind-O 训练侧的 loss mask 在 `dataset/omni_dataset.py`，那是训练账本，不能拿来当推理 mask 的替代物。推理 mask 盖的是“哪些 key 可以参加注意力”；loss mask 盖的是“哪些位置算 CE”。两者都是 0/1 表，区间不同。

验证。CPU 检查 `invalid_positions_excluded_from_valid_count` 与 `wrong_padding_mask_puts_mass_on_invalid_keys`。把 A 的下标 3 放进有效集合，第一项失败。把 mask 改成全 1，第二项的无效质量不再是 0。Lab 用条纹表示 padding，纯文本那一行在 fused 模式下必须出现长条纹。

把 A 的 15 个位置写成一行，便于对照 CPU `metrics["valid_indices_a"]`：

| 下标 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 种类 | 视 | 视 | 视 | pad | pad | pad | pad | pad | pad | 文 | 文 | 文 | 文 | 文 | 文 |
| $m_A$ | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |

B 的行：

| 下标 | 0–8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|
| 种类 | 九个视觉 | 文 | 文 | pad | pad | pad | pad |
| $m_B$ | 全 1 | 1 | 1 | 0 | 0 | 0 | 0 |

有效集合 $A=\{0,1,2,9,10,11,12,13,14\}$，$B=\{0,1,2,3,4,5,6,7,8,9,10\}$。下标 3 对 A 是 pad，对 B 是真实视觉。batch 维上同一列的 mask 可以不同，这正是“按请求写 mask”而不是“按列写死”的原因。若实现成“第 3 列对所有请求都有效，因为 B 在用”，A 的无效视觉就会进注意力。CPU 夹具按行存 mask，禁止按列或。

均匀 logits 只是探针。真实模型里 pad 位置若没被清成 0 向量，残留的是这块显存上一次请求留下的 embedding。服务进程长时间运行后，串味会从“偶发胡话”变成“稳定复读上一张图的物体名”。第 01 课要求 golden case 可追溯；本课对应的运行时纪律是：padding 位置的 value 要么乘 $m_j=0$ 从分母分子同时消失，要么在写入时清零。只清零不清 mask，softmax 仍会给零向量分配质量，输出被稀释。只写 mask 不清零，内核若忽略 mask 就会读垃圾。两道闸都要。

### 5.6 有效 token 比

为什么需要。吞吐数字喜欢报 tokens / s。分母如果是 padded 形状，padding 越多看起来越忙，实际有效工作越少。第 13 课已经要求 global useful tokens 可精确计数，禁止吞吐来自少算 token 或乱 pad。本课把同一口径用在推理：$\rho$ 是调度质量，不是模型质量。

怎么运转。定义

$$
\rho=\frac{\sum_b(v_b+t_b)}{B\cdot(\max_b v_b+\max_b t_b)}
$$

两条视觉请求：$\rho=20/30=2/3$。把视觉长度为 0、文本长度为 4 的纯文本请求垫进同一 batch，$V$ 仍为 9，$T$ 仍为 6，$B=3$，有效 $20+4=24$，padded $45$，$\rho=24/45=0.533\ldots$，严格下降。这就是命题 1 在 CPU 上的第三人称。若再把动作步 $N_{\max}=8$ 编进同一静态形状，计算槽变成 $45+3\times 8=69$，有效槽 $24+8=32$，$\rho$ 继续下降。Packed 布局（只拼接真实 token、靠页表找回位置）的分母等于分子，$\rho=1$。PagedAttention 的目标之一，正是逼近这种 packed 记账，而不是在显存里为 max length 预留整块连续 KV（vLLM §3.1）。

数学。$\rho$ 有下界。最坏情况是 $B-1$ 条请求长度为 0，一条请求取到两个 max，则

$$
\rho\ge\frac{\max v+\max t}{B(\max v+\max t)}=\frac{1}{B}
$$

三条模态齐全时，$B=3$ 给出 $1/3$。Lab 默认配置里 fused 的 $\rho$ 会落在 $1/3$ 和 1 之间，并且对 patch 滑条单调：带图请求的 patch 越多，$\max v$ 越大，纯文本那一行浪费越多。这是可在浏览器里拉动的单调性，不是论文表。

代码落点。服务日志若只报 `batch_tokens = B * L`，本课视为未完成测量。至少要并列 `valid_tokens` 和 `padded_tokens`。vLLM-Omni 论文报告的是 JCT、RTF、TPS，没有直接报告 $\rho$。其技术报告未公开该细节。本课把 $\rho$ 当作缩小版必测项，不把论文没写的数字填进论文列。

验证。CPU：`valid_token_ratio_uses_mask_not_padded_shape` 要求 $\rho=20/30$，naive 比率为 1，且 mask 版严格更小；`text_only_request_in_fused_batch_drops_ratio` 要求第三人加入后变为 $24/45$。Lab 揭晓后的 $\rho$ 必须小于 1，且纯文本空槽等于当前 $\max v+\max N$。

把 Lab 默认值也算一遍，避免只记住 CPU 的 $2/3$。fused：$V=48$，$T=16$，$N=8$，$B=3$，分母 $3\times 72=216$，分子 $0+16+0+48+12+0+24+8+8=116$，$\rho=116/216\approx 0.537$。纯文本空槽 $48+8=56$。stage 模式下编码只批带图的两条，视觉分母 $2\times 48=96$，视觉分子 $48+24=72$；decode 按真实文本 $16+12+8=36$ 计；动作只跑 8 步。把三阶段的有效比分开写，比合成一个全局 $\rho$ 更有用：编码 $\rho_{\text{enc}}=72/96=0.75$，动作 $\rho_{\text{act}}=1$，decode 若走分页可以逼近 1。全局合成 $\rho$ 会把已经做好的阶段拉低，掩盖“哪一段还在 pad”。服务日志按阶段报 $\rho$，不要只报一行。

$\rho$ 与吞吐的关系可以写成下限。设有效核函数时间正比于有效 token，padding 额外时间正比于无效 token，则同一张图上

$$
T_{\text{fused}}\ge T_{\text{packed}}\cdot\frac{1}{\rho}
$$

$\rho=0.537$ 意味着合图至少比 packed 多花约 86% 的槽位时间，这还没算错误 KV 造成的重算或错答。这个不等式是教学下界，真实内核有固定开销，差距可能更大或更小。不要把它写成论文结论。vLLM-Omni 没有报告 $\rho$，所以本课不把 86% 和 91.4% 放在同一句里。

### 5.7 连接器、流式交出、按阶段分资源

为什么需要。拆开之后，阶段之间要搬家。搬得太慢，拆分得不偿失；搬得太死，Talker 必须等 Thinker 把整段文本吐完才能开口，TTFA（到首个可播放音频的时间）被整段文本长度绑住。[第 04 课](04_multicodebook_talker.md) 已经说明 codec 按帧出声；服务层若不能流式交 codec，帧结构的优势用不上。

怎么运转。vLLM-Omni 的 output processor 在 AR 阶段把 transfer 的结果先放到 CPU 内存，再送到下游所在设备（§3.3）。流式模式下，部分 token 或 embedding 异步送给下一阶段，上游继续跑。Qwen-Omni 的 Vocoder 可以在 Talker 刚吐出前若干 codec 时就开始波形合成。资源分配与搬家正交：Thinker 30B 多分内存，Talker 多分加速器和并行（§3.3）；vLLM-Omni 的评测配置把 Thinker 做跨两张卡的张量并行，Talker 放 device-1，Vocoder 放 device-0（§4.2）。基线 Transformers 使用默认张量并行，三个模块仍绑在同一次生成里。

数学。忽略重叠时，流水线 JCT 是各阶段耗时之和。允许 Vocoder 在 Talker 产出 $k$ 个 codec 后启动，则

$$
\mathrm{JCT}\approx T_{\text{thinker}}+T_{\text{talker}}+T_{\text{vocoder}}-\Delta_{\text{overlap}}
$$

$\Delta_{\text{overlap}}$ 的上界是 $T_{\text{vocoder}}$ 与 Talker 尾部时间的较小值。连接器延迟 $d$ 必须满足 $d\ll T_{\text{stage}}$，否则 overlap 被搬家吃掉。Table 1 给出 Qwen2.5-Omni 上的实测：$d$ 为数毫秒，论文称端到端是数十秒量级，因此连接器不是主因。主因在 Talker 的迭代次数：视频任务上音频 token 平均 545.4，文本 150.9（§4.2）。

代码落点。边 `Thinker2Talker`、`Talker2Vocoder` 在非流式语义下各调用一次；流式是 output processor 的增量发送，不改变节点上的 `forward` 属于哪个引擎。单机共享内存 5.49 ms / 0.53 ms，Mooncake 8.28 ms / 3.34 ms。跨机数字不能直接抄到单机日志里。

验证。本课 CPU 不模拟毫秒级连接器。能复述 Table 1 四格、能指出 Talker 迭代更多，这一项口头验收即过。若实验报告把 91.4% 的 JCT 降幅归因于“连接器更快”，判失败：论文明确说连接器开销可忽略，大头是阶段内引擎和 graph compilation。

流式交出还改变 TTFA 的定义。整段结束后才解码波形，TTFA 至少包含 Thinker 全文加 Talker 全文加 Vocoder 启动。Talker 每产出一帧 codec 就交给 Vocoder，TTFA 可以缩短到 Thinker 首段文本加 Talker 若干帧加 Vocoder 一块波形。第 01 课已经把 TTFT / TTFA / RTF 分列；本课要求服务日志在打开流式边之后，这三列都在，且明确写 Vocoder 是否等完整 codec。RTF 用生成音频时长做分母，流式与否都会改变分子里的等待，但不改变“音频 token 有 545 个就得算 545 步”这件事。优化 TTFA 靠 overlap；优化 RTF 靠 Talker 引擎本身。两件事不要写成同一个开关。

单机共享内存和 Mooncake 的选用按拓扑，不按“更先进”。同机、小 payload 走控制队列，大 payload 走共享内存；跨机才上 TCP / RDMA。Table 1 里 Mooncake 的 Thinker2Talker 8.28 ms 仍远小于推理的数十秒，所以跨机拆阶段在延迟上可行。可行不等于必须：MiniMind-O 量级两进程同卡已经够练隔离，不必为了抄 Table 1 去搭 RDMA。报告里写实际拓扑。没有跨机就不要填 Mooncake 那一行。

### 5.8 和第 13 课对照：训练并行不是推理调度

为什么需要。两课都出现 GPU、batch、padding、通信。混在一起会把 FSDP 的 all-gather 写成 stage connector，或把 CUDA graph 当成八卡 mesh 的第 4 维。

怎么运转。对照表只用于划清单位，不把第 13 课的公式搬过来重算。

| 项目 | 第 13 课 | 本课 |
|---|---|---|
| 目标 | 切开后 loss / 梯度 / 更新与单进程一致 | 切开后 mask / KV 归属 / 有效 token 计数正确 |
| 单位 | 训练 step、rank、global batch | 请求、stage iteration、JCT |
| 切分对象 | 参数、专家、上下文 | 理解编码、AR decode、flow / 动作 |
| 通信 | NCCL collective | 阶段连接器（共享内存 / Mooncake） |
| padding 错误的后果 | 吞吐虚高或 loss 归一化错 | 无效 patch 进入注意力，或空动作步空转 |
| CUDA 角色 | 设备是 mesh 里的 rank | graph 是阶段内可重放的核函数序列 |
| 验收数字 | 与单卡参考的绝对误差 | $\rho$、无效注意力质量、KV 交集 |

第 13 课允许、甚至要求八张卡合作算同一次训练步。本课禁止三个异构阶段合作算同一次 launch。看似相反，约束其实相同：语义由谁定义，谁就不能被图或 mesh 偷偷改掉。训练语义是优化目标；推理语义是“这条请求的这条 key 来自它自己的历史”。

合图失败有两条可独立触发的路径，Lab 的四个预测选项对应它们的组合。路径一是 padding 浪费：形状取全局 max，有效比下降，答案仍可能碰巧对，因为 mask 若写对，空槽不进分母。路径二是错误共享 KV：页表从 0 开始共用，答案会串。验收“必须看到 padding 浪费或错误共享 KV”允许只命中一条；两条同时出现时选“both”。教学上更有用的是把它们拆开：先假设 mask 正确、页表错误，得到串味；再假设页表正确、mask 全 1，得到稀释。CPU 夹具把稀释做成 0.4 的闭式解，把别名做成页号交集，就是拆开的两个探针。

视觉 padding 比文本 padding 更伤，原因在注意力方向。文本 decode 通常是因果的：新 query 看历史，pad 若接在序列尾、mask 又写对，伤害有限。视觉 encoder 常用双向注意力：每个 patch 看所有 patch。把 6 个 pad patch 和 3 个真 patch 放在同一层双向注意力里，即使 mask 写对，实现稍一疏漏（例如只 mask 了 logits 没 mask softmax 的数值稳定项，或 fused kernel 忽略 mask），三个真 patch 的表示就会被空位拉偏。因此本课把视觉 mask 写成硬验收，而不是“和文本 pad 一样对付一下”。[第 08 课](08_dynamic_vision.md) 训练时按视觉 token 数分桶，减少跨分辨率混 batch；服务时若无法分桶，至少要按请求写 mask，并在日志里把 $\rho_{\text{enc}}$ 单独列出来。

显存账单也可以不靠 GPU 估一个下界。fused 需要同时常驻：encoder 权重、AR 权重、动作专家权重，加上一份按全局 max 预留的 KV。stage graph 允许 encoder 在纯文本高峰时少占 batch，动作专家在没有动作请求时整段不 launch，KV 按阶段分页增长。vLLM 对 OPT-13B、最大 2048 的预分配方案里，KV 真正有效的只有 20.4%–38.2%。Omni 合图等于在这份浪费上再乘视觉 max 与动作步 max。本课不要求你在笔记本上量出 20.4% 这个数；要求你承认：合图的显存下界高于分阶段的显存下界，差来自空槽和空模块，不来自“多模态本身更贵”。多模态可以很贵（841.6 个输入 token），但那是内容长度，不是 padding。

数学。第 13 课的全局有效 token loss 是 $\sum \ell / \sum n_{\text{valid}}$，不能改成各 rank 先平均再平均。本课的 $\rho$ 是 $\sum n_{\text{valid}} / \sum n_{\text{padded}}$，不能改成 padded 形状当有效。两个公式的 $n_{\text{valid}}$ 定义相同：mask 为 1 的位置。本课 CPU 实验甚至可以看作第 13 课那条“useful tokens 可精确计数”在推理侧的单进程版。

通信对象也不同。第 13 课 all-reduce 的是梯度或 router 统计，语义是“每个 rank 都要拿到同一份和”。本课连接器搬的是某一条请求的 hidden state，语义是“只有下游阶段、只有这条请求需要这份激活”。把 all-reduce 用在 Talker 输入上，会把不同用户的 hidden state 加在一起，比 KV 别名更糟。本课实验不实现 NCCL，但验收口头题要能拒绝这种搬法。张量并行出现在 Thinker 内部（论文把 Thinker 铺到两张卡）是合法的，因为它仍在同一阶段、同一套权重上切矩阵，和 13 课的 TP 同构。跨阶段的“并行”只许是流水重叠，不许是把三个 `forward` 写成一次 collective。

代码落点。第 13 课的实验文件数的是 mesh 坐标和 all-reduce 后的 router 统计。本课实验文件数的是 mask 下标和注意力质量。不要把 `lesson_13.py` 的 rank 循环抄进 `lesson_46.py`。

把一条 Qwen3-Omni 视频请求的阶段内工作量再拆一次，用论文平均数当量纲。Thinker prefill 处理 841.6 个输入 token，这是一次（或按 SARATHI 切成几块的）compute-bound 前向；随后 150.9 步 decode，每步 1 个新 token 加整段 KV。Talker 545.4 步 decode，每步还要跑 `process_input` 去拼接 Thinker hidden state。Vocoder 若是 Qwen2.5-Omni 的 DiT，循环次数由去噪步决定；Qwen3-Omni 换成 CNN，循环次数变成帧数或卷积块次数。四段的“一步”不是同一个物理时间。合图若用一个 for 循环驱动所有模块，循环上界只能取四段里最长的那个，其余模块在多数 iteration 里空转。空转在 CUDA graph 里仍然是录进去的核函数，只是输入是 padding。这就是 $\rho$ 会掉、显存会涨、profiler 上 encoder 核却还在跳的共同原因。

非均匀 logits 下的错误 mask 可以再算一例，避免只会 0.4。令 A 的 9 个有效位置分数为 1，6 个 pad 分数为 0。正确 mask：pad 被置 $-\infty$，有效位置 softmax 后各 $e/(9e)=1/9$，无效质量 0。错误 mask：有效位置权重正比于 $e^1$，pad 正比于 $e^0=1$，分母 $9e+6$，无效质量 $6/(9e+6)\approx 0.197$。比均匀时的 0.4 小，但不是 0。真实模型里 pad 位置若残留上一请求的高分 embedding，无效质量可以超过 0.4。CPU 夹具用均匀探针，是因为它有闭式解、不依赖随机数；线上要用真实分数再打一遍日志，本课不要求。手算这一例的目的只有一个：证明“pad 分数低就能省 mask”不成立。

验证。口头题：FSDP2 能不能加速 Qwen-Omni 的 Talker decode？可能，那是阶段内部的张量并行或分片，仍然要先有 Talker 这个 stage。它不能替代 stage graph。Lab 通过条件里没有“八卡”，CPU 通过条件里没有 NCCL。出现这两项，说明题目抄错课。

## 6. 在公开实现中定位这些机制

本课不改 MiniMind-O 训练脚本。能打开的公开落点如下。路径以论文接口名为准；上游仓库若重组文件，以函数职责核对，不以过期路径核对。

**反例：单次 `generate()`。** vLLM-Omni §2.2 把 Qwen-Omni 的手工流程写成：多模态输入进 Thinker 的端到端 `generate()`，跑完编码和 AR 循环；取出 hidden state，变换成 Talker 输入；Talker 再跑自己的 `generate()`；最后 Vocoder 出波形。MiniMind-O 的 `MiniMindOmni.stream_generate` 属于同一家族：一次调用内部串起 Thinker 与 Talker。它在课程前半段是正确的产品形态（先能出声），在本课是要拆开的服务形态。

**正例：stage 节点。** 论文要求每个 AR 阶段提供 `forward` 与 `preprocess`，每条边提供 transfer。Qwen2.5-Omni 对照表：

| 符号 | 职责 | 调用频率 |
|---|---|---|
| `mm_encode` | 音频 / 图像 / 视频变 embedding，拼进 Thinker 输入 | Thinker 开始时 |
| `thinker_forward` | 文本 AR | Thinker 每 iteration |
| `Thinker2Talker` | 把 Thinker 产物交给 Talker 引擎 | 阶段边界（非流式：一次） |
| `process_input` | 把 Thinker hidden state 拼进 Talker 当前步输入 | Talker 每 iteration |
| `talker_forward` | 音频 token AR | Talker 每 iteration |
| `Talker2Vocoder` | codec 序列交给 Vocoder | 阶段边界或流式增量 |
| `dit_decode` | Qwen2.5-Omni 的 DiT Vocoder | Vocoder 的去噪循环 |

Qwen3-Omni 的 Thinker / Talker 切法相同，Vocoder 改为轻量 CNN，因此 Vocoder 节点的 `forward` 不再是 DiT 步。这是“图结构稳定、节点实现可换”的例子：stage graph 该稳定的是边的数据契约，不是某一个核函数。

**引擎内部。** 每个 AR 引擎自带 scheduler、KV manager、model runner（§3.3）。preprocess 读一张 per-request 字典，transfer 和 preprocess 都能改它。这张字典是跨阶段的请求级便笺，不是共享 KV。DiT 引擎内部另有一套注意力与缓存（论文提到 flash attention、SAGE、TurboAttention、TeaCache、cache-dit、RingAttention、Ulysses），全部停在扩散阶段里。不要把 TeaCache 的时间步缓存写成 AR 的 KV 页。

**连接器。** 单机共享内存与跨机 Mooncake 是同一接口的两种运输实现（Table 1）。EPD 拆分走同一连接器的“阶段内”模式。DistServe 把 prefill 和 decode 拆到不同 GPU，优化的是 TTFT 与 TPOT 解耦；它仍然假设两端是同一个 LLM。把它的 7.4× 请求率数字写进 Omni 多阶段表，属于跨课挪用。

**核对清单。** 打开任意 Omni 服务脚本时按行问：哪一段 `forward` 对应哪个节点？哪一次张量拷贝对应哪条边？KV manager 有几份？视觉 `attention_mask` 在哪生成？CUDA graph 的捕获范围包不包括超过一个节点？五问里有一问答成“整个模型”，本课视为未定位。

MiniMind-O 对照可以写得更死，方便回头改推理入口而不碰训练。Thinker 8 层、Talker 4 层、SenseVoice、SigLIP2、Mimi 五个模块里，SenseVoice 与 SigLIP2 属于理解编码（可并进 Thinker 节点），Thinker 是 AR 文本节点，Talker 是 AR 音频节点，Mimi 解码是 Vocoder 节点。训练脚本 `trainer/train_sft_omni.py` 和数据集 `dataset/omni_dataset.py` 管的是 loss mask，不是服务 mask。推理 `stream_generate` 若仍按回合吐完整 8 路 code 再解码，相当于 Talker2Vocoder 一次性 transfer、无流式。第 07 课的全双工 routing 发生在会话层，可以在 stage graph 之外再包一层；本课不把 CONTINUE / PAUSE 画进 CUDA graph。改 MiniMind-O 服务时，允许的最小 diff 是：Thinker 循环结束把 hidden state 放入队列，Talker 循环从队列读，两份 KV 列表地址不同。diff 里如果出现“复用 `past_key_values` 给 Talker”，按 5.4 判失败。

公开仓库 [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni) 的目录会随版本变。读代码时按论文符号搜 `Thinker2Talker`、`process_input`、`dit_decode`，不要赌某个 `omni_model.py` 永远存在。搜不到符号，先核对该版本是否仍实现论文 §3.2 的抽象；抽象变了，以代码契约为准，并在实验记录写版本哈希。没有哈希的“官方 main”不能当复现基线，这条纪律与第 01 课相同。

## 7. 数据与服务 recipe

本课没有新的训练数据。Recipe 指的是请求混合物、阶段切分、资源预算和日志字段。缺任何一项，JCT 数字不可复现。

**请求混合物。** vLLM-Omni §4.2 用 librispeech_asr、food101、ucf101-subset 的前 100 条分别当音频、图像、视频输入。这是评测负荷，不是训练语料。本课 Lab 的混合物更硬：三条请求模态集合不同，专门逼出合图失败。真实服务的混合物会随产品变化；报告必须写清“本次负荷里纯文本占比、带图占比、带动作占比”。只报平均 JCT、不报混合物，数字停用。

**阶段切分 recipe。** 最小可运行图按模型家族选：

| 模型家族 | 节点 | 边 |
|---|---|---|
| Qwen2.5-Omni | Thinker（含 encoder）、Talker、DiT Vocoder | Thinker2Talker，Talker2Vocoder |
| Qwen3-Omni | 同上结构，Vocoder 换 CNN | 同上 |
| GLM-Image 类 | AR LLM，DiT 解码器 | AR2DiT |
| 统一理解 + 动作 | 视觉编码，VLM decode，动作专家 | 编码到 VLM，VLM 条件到专家 |

视觉编码器单独成节点，只在带图流量足够大、需要和 prefill 解耦时才值得。论文允许两种放法，本课默认：流量未知时先并进 Thinker，少一条边；$\rho$ 被视觉 max 严重拉低时再拆。

**资源预算。** 论文测试机两张 80GB。Thinker 30B 用张量并行铺满两张卡，Talker 与 Vocoder 各钉在一张卡上与 Thinker 重叠。缩小版（MiniMind-O 量级）没有 30B，预算改成：Thinker 与 Talker 分两个进程，哪怕共占一张消费级卡，也要两份 KV manager。单进程里用 if 把模块串起来，不算完成拆分。

**编码器放哪。** 论文允许视觉 / 音频 encoder 单独成 stage，或并进 LLM stage。并进的好处是少一条边、少一次连接器往返，vLLM 现有的多模态输入路径可以直接用。拆出的好处是：纯文本高峰时 encoder 卡可以去服务带图队列，或干脆缩容。决策规则用 $\rho_{\text{enc}}$ 和流量：若纯文本占比高、$\rho_{\text{enc}}$ 长期低于 0.5，拆；若几乎每条请求都带图，并进。Lab 的纯文本请求专门制造低 $\rho_{\text{enc}}$，用来练习这条规则。不要把“encoder 并进 Thinker”写成“encoder 和 Thinker 捕获成一张图还要再并进动作专家”。并进只发生在理解路径内部。

**日志字段。** 每次评测至少写：

| 字段 | 含义 | 失败时的典型脏值 |
|---|---|---|
| `valid_tokens` | mask 求和 | 等于 `B * L` |
| `padded_tokens` | 张量元素数 | 缺省 |
| `rho` | 二者之比 | 恒为 1 |
| `stage_batch[s]` | 阶段 $s$ 的请求数 | 所有 $s$ 都等于总请求数 |
| `kv_pages[s]` | 阶段页数 | 各 $s$ 起始页都是 0 且有交集 |
| `jct_ms` / `rtf` / `tps_thinker` / `tps_talker` | 论文同名指标 | 只有一个 TPS |

**推荐对照。** 先跑 Transformers 式单进程基线，再跑按阶段拆开、仍无 CUDA graph 的版本，再给每个 AR 阶段打开 graph compilation。三行数字才能解释 MiMo-Audio 那组 1.39 / 0.60 / 0.12：原实现、阶段引擎、阶段内编译。缺中间行，不要把 0.12 归功于“Omni 合图”。

**混合物的最小回归集。** 除论文用的 100 条音频 / 图像 / 视频查询外，本课要求自己的服务至少常驻三类合成请求：纯文本短问、单图短问、带动作的短观察。三类各至少一条进入每次发布前的回归。纯文本用来揭 encoder 是否仍被 launch；单图用来揭视觉 mask；带动作用来揭页表是否从 0 重叠。三类都通过，才能说 stage graph 在功能上站住。论文的 100 条查询测的是负荷下的 JCT，测不了这三种失败。两套回归不要互相替代。

缩小版没有 Qwen 权重时，三类回归可以全部落在夹具上：Lab 的三条请求加上 CPU 的 A/B mask。报告写“功能回归通过、负荷回归未跑”。有单卡之后，用 MiniMind-O 的 `stream_generate` 拆成两进程，三类请求各跑 golden case。文本侧对 token id，音频侧对 8 路 code 的哈希。动作侧若还没有真专家，用固定 $H=2$ 的零速度场积分，只检查页号不重叠。这样仍然测不到 91.4%，但能测到本课四个命题里除命题 1 的 Lab 版之外的全部 CPU 版。

## 8. 按依赖顺序执行实验

实验分两层。浏览器 Lab 负责“先预测再看见合图失败”。CPU 负责“无效位置不得进入有效计数”。都标教学夹具。不要根据它们报告真实 JCT。

### Step 0：固定两条视觉请求的数字

纸上写下 A：视觉 3、文本 6；B：视觉 9、文本 2。不要改。后面所有手算用这一对。若改长度，CPU 夹具的绝对值会变，检查项按公式仍应成立，但本课标准答案按这对长度写。

### Step 1：手算 mask 与 $\rho$

$V=9$，$T=6$，$L=15$。写出长度 15 的两行 0/1。确认 A 有效 9、B 有效 11、合计 20，padded 30，$\rho=2/3$。无效下标不得出现在有效列表里。这一步不过，不要打开 Lab 的揭晓。

### Step 2：手算错误 mask 的注意力质量

A 的 15 个 logits 全是 0。正确 mask：无效质量 0。错误 mask：无效质量 $6/15=0.4$。把 0.4 写进实验记录。CPU 用同一组分数，浮点误差应在 $10^{-12}$ 内。

Lab 默认值建议先在纸上算一遍再拖滑条，避免揭晓时对数字没感觉。

| 请求 | $v$ | $t$ | $N$ | 有效槽 |
|---|---|---|---|---|
| 纯文本 | 0 | 16 | 0 | 16 |
| 带图 | 48 | 12 | 0 | 60 |
| 动作 | 24 | 8 | 8 | 40 |
| 合计 | max 48 | max 16 | max 8 | 116 |

fused 分母 $3\times(48+16+8)=216$，$\rho=116/216$。纯文本空槽 $48+8=56$。把带图 patch 从 48 拖到 80，分母变成 $3\times(80+16+8)=312$，分子 $16+(80+12)+40=148$，$\rho=148/312\approx 0.474$，浪费上升。这个单调性必须在 Lab 里能摸到：加长别人的图，纯文本更亏。stage 模式不该出现同样幅度的下跌，因为纯文本根本不进 encoder。

### Step 3：浏览器 Lab，先预测后捕获

打开 `Lesson46ServeGraphLab`。不要先点“捕获并揭晓”。先在四个选项里选合图之后会发生什么。再确认模式是“一条 CUDA graph”。拉动 patch 数和动作步，观察条纹（padding）是否变长，但数字栏在捕获前应是破折号。点捕获。验收：必须看到 padding 浪费或 KV 别名；预测若选“只是慢一点，数值仍然正确”，门禁失败。

### Step 4：对照 stage graph

不要把 Step 3 的 fused 结果清掉心里账。切到 stage graph 再捕获一次，看编码列是否只收带图请求、动作列 batch 是否变为 1、KV 页是否分成 AR 与动作两色。对照完必须切回“一条 CUDA graph”再捕获，门禁才认。这是故意的：本课要你强行合图，而不是只看正确调度的漂亮图。

### Step 5：跑 CPU 夹具

在 `experiments/` 下，编排者登记之前可用：

```bash
python3 -c "from learn_omni_experiments.lessons.lesson_46 import run; print(run()['checks'])"
```

登记之后：

```bash
python3 run.py run 46
```

`checks` 必须全为 True。重点看 `invalid_positions_excluded_from_valid_count` 与 `wrong_padding_mask_puts_mass_on_invalid_keys`。前者对应命题 3，后者对应命题 4。`text_only_request_in_fused_batch_drops_ratio` 把 Lab 的第三条请求缩成视觉 0、文本 4，确认 $\rho$ 从 $20/30$ 降到 $24/45$。

CPU `metrics` 的标准答案（公式固定，浮点按夹具打印）：

| 字段 | 值 |
|---|---|
| `valid_a` / `valid_b` | 9 / 11 |
| `valid_total` / `padded_total` | 20 / 30 |
| `valid_token_ratio` | $2/3$ |
| `invalid_attention_mass_correct_mask` | 0 |
| `invalid_attention_mass_wrong_mask` | 0.4 |
| `valid_indices_a` | 0,1,2,9,10,11,12,13,14 |
| `invalid_indices_a` | 3,4,5,6,7,8 |
| `fused_three_valid` / `fused_three_padded` | 24 / 45 |
| `dummy_action_steps` | 16 |
| `fused_kv_collision_pages` | 0 到 7 |
| `stage_kv_collision_pages` | 空 |

手算与夹具不一致时，先查自己有没有把 A 的视觉 pad 算进有效。不要改夹具去迁就手算。`naive_valid_token_ratio` 必须是 1：它假装 padded 形状全有效，专门当反例。若有人把 naive 比率写进报告当成绩，按第 10 节第 8 条失败处理。

### Step 6：把论文数字抄进另一列

另开一列写 vLLM-Omni §4.2 / Table 1，禁止与 Step 5 的 20/30 对齐相减。至少抄：Qwen2.5-Omni RTF -61.4%、JCT -61.6%、Thinker TPS 1.29×、Talker TPS 1.97×；Qwen3-Omni RTF -90.7%、JCT -91.4%、Thinker TPS 12.97×、Talker TPS 7.98×；视频 token 841.6 / 150.9 / 545.4；Table 1 四格。抄漏模型名或基线名，视为未完成。

## 9. 评测与测量

服务评测容易把“更快”写成“对了”。本课把指标分成三桶，桶之间禁止横着减。

**正确性桶。** $\rho$、无效注意力质量、KV 交集、阶段 batch 维。CPU 和 Lab 只填这一桶。$\rho$ 没有公开论文对照值。无效质量在均匀 logits 下有闭式解，不依赖 GPU。

**时延桶。** JCT、TTFT、TTFA、RTF、TPS。必须写模型、基线、输入模态。Qwen3-Omni 的 91.4% 是相对 Transformers 默认实现，测试机两张 80GB，输入各数据集前 100 条。换成本课 Lab 的三条玩具请求，JCT 没有定义。Talker TPS 和 Thinker TPS 必须分列：论文里 Qwen3-Omni 的 Thinker 加速 12.97×、Talker 7.98×，作者把 Thinker 的巨大相对收益部分归因于基线没有用好 execution graph compilation，以及 30B 比 7B 更能摊掉优化管线（§4.2）。把两个 TPS 平均成一个“Omni TPS”，本课视为漏报。

**资源桶。** 每阶段显存、张量并行度、连接器延迟。Table 1 属于这个桶。连接器数毫秒不能解释数十秒级 JCT 差异。BAGEL 与 DiT 的加速属于扩散引擎桶，不要和 Qwen-Omni 的 RTF 排在同一张“系统加速”总表里冒充同构。

测量协议：

1. 先冻结请求混合物和阶段图，再测 JCT。
2. 同一混合物上必须同时记 `valid_tokens` 与 `padded_tokens`。
3. 打开 graph compilation 前后各测一次，编译范围写到阶段名。
4. 流式交出打开前后，TTFA 与 JCT 分列；只降 JCT 不降 TTFA，说明 overlap 没发生。
5. 基线必须是论文用的那一类：Qwen-Omni 用 Transformers，BAGEL / MiMo-Audio 用原实现，DiT 用 Diffusers。用 vLLM 只跑 Thinker 再声称打败 vLLM-Omni，属于换基线。

不确定性。论文没有给 JCT 的误差条或 seed 数。其技术报告未公开该细节。缩小版若只跑一次 Lab，不要写置信区间。需要区间时，用第 17 课已经登记的算法，并且 $N$ 是请求数，不是 patch 数。

Qwen3-Omni 相对 Qwen2.5-Omni 的加速差异也要单独记账，避免写成“新模型一定更快”。Qwen2.5-Omni：RTF -61.4%，JCT -61.6%，Thinker TPS 1.29×，Talker TPS 1.97×。Qwen3-Omni：RTF -90.7%，JCT -91.4%，Thinker TPS 12.97×，Talker TPS 7.98×。作者给出的机制解释是：基线没有充分用上 execution graph compilation，且 Qwen3-Omni 的 Thinker 30B 比 7B 更能摊掉优化管线。相对收益大，不自动等于绝对 JCT 小。没有两张 80GB 卡，禁止把 12.97× 抄进自己的系统表。缩小版若用 MiniMind-O 量级对比“拆阶段 vs 单次 generate”，只许报告 golden case 是否一致和是否出现两份 KV，不许报告“接近 12.97×”。

DiT 引擎那 1.26×（相对 Diffusers，VBench，Qwen-Image / Wan2.2，1024×1024 或 480×640、80 帧）属于视觉生成阶段内部优化：flash-attention 后端、去噪缓存、序列并行。把它加到 Omni 文本 TPS 上没有代数意义。BAGEL 的 T2I 9.64 s、I2I 11.12 s 同样只属于 BAGEL 那一节。本课允许三张分表，不允许一张“本系统加速倍数”总表把 1.26、2.40、12.97、91.4% 四格平均。

## 10. 验收条件

同时满足以下条款，本课才算通过。任何一条用论文 JCT 代替夹具，或用夹具代替论文 JCT，整课作废。

1. 课文与实验记录里出现第 13 课对照表，或等价的四行以上文字，明确训练并行与推理调度的单位不同。
2. 两条视觉请求的有效 token 数为 20，padded 为 30，$\rho=2/3$。无效下标不在有效列表中。
3. 错误 mask 的无效注意力质量为 0.4，正确 mask 为 0。
4. Lab 在“一条 CUDA graph”模式下先提交预测再揭晓；预测不是“数值仍然正确”；屏幕上出现 padding 浪费或 KV 别名。
5. CPU `checks` 全为 True。
6. 论文数字单独成列，至少包含 Qwen3-Omni JCT -91.4% 与视频 841.6 / 150.9 / 545.4，并写明 Transformers 基线。
7. 交付物里的 stage graph 至少三个节点、两条边，Talker preprocess 的每步调用被标明。
8. 没有把 DistServe 的 7.4× 或 vLLM 的 2–4× 写成 vLLM-Omni 的多阶段加速。
9. 实验记录第一页写明教学模拟、CPU 不报 JCT、两套长度不兑、91.4% 带模型与基线。
10. 机制图或手绘 stage graph 至少覆盖理解编码、语言 decode、flow / 动作三类节点，并标明 KV 与连接器。

失败的快捷方式（出现即未通过）：把 MiniMind-O 的一次 `stream_generate` 称为已经完成 stage graph；把 TeaCache 当作 AR KV；把 $\rho=1$ 写进 fused 合图日志；在讲稿里写“合图之后理解和生成互相加速”；把 Lab 的 $\rho$ 与论文 JCT 降幅写进同一格子。

门禁顺序建议：先 CPU（命题 3、4），再 Lab 合图（命题 1、2），最后抄论文列。倒过来先抄 91.4%，很容易用它覆盖夹具失败。CPU 全 True 而 Lab 预测仍选“数值仍然正确”，本课仍不算过，因为验收明文要求先预测再看见浪费或别名。

## 11. 根据症状定位失败环节

按观测到的现象找阶段，不要先改学习率。本课没有训练。

| 症状 | 先查 | 不是 |
|---|---|---|
| 纯文本延迟随最大图片分辨率上涨 | fused 形状把 $\max v$ 锁进图；纯文本仍进 encoder | 语言模型变慢 |
| 有效 token 数等于 $B\times L$ | mask 全 1 或根本没建 mask | tokenizer 多切了词 |
| 答案串进另一张图的物体 | KV 页别名，或 pad value 未清零 | 视觉编码器“理解能力不够” |
| 动作轨迹在对话轮次之间漂移 | 动作专家读了 AR 页，或积分步被 pad 进 token 轴 | 动作块 $H$ 选小了 |
| Talker 要等整段文本才出声 | 边是一次性 transfer，流式 output processor 未开 | Vocoder 太慢（先看是否已重叠） |
| JCT 降了但 RTF 仍大于 1 | 音频 token 数仍按 545 量级在走 Talker；阶段内 decode 没加速 | 连接器 Table 1 |
| 打开 CUDA graph 直接报形状错误 | 捕获范围跨了阶段，或 $v_b$ 变化未 pad | NCCL 超时（那是第 13 课） |
| stage 模式下 $\rho$ 仍接近 fused | 所谓 stage 只是 UI 分列，实际仍按全局 max pad | 连接器太慢 |
| Thinker TPS 大涨、Talker TPS 不动 | graph compilation 只打在 Thinker；Talker 仍是逐步 Python 循环 | 数据混合物变了 |
| 与论文 91.4% 差一个数量级 | 基线不是 Transformers 整图 `generate()`，或模型不是 Qwen3-Omni 30B Thinker | 夹具 $\rho$ 算错 |
| 纯文本输出里出现图中物体名 | 视觉 pad 未 mask，或 KV 串到带图请求 | 幻觉探针（那是第 23 课） |
| 动作块长度变成文本 token 数 | 动作维被 pad 进序列轴 | 动作分块 $H$ 的超参 |
| graph compilation 后结果漂移 | 捕获时的 batch 组成与重放时不同 | 学习率 |
| Vocoder 声音断续 | 流式边一次交的 codec 帧数小于解码窗 | Talker 采样温度 |
| 显存随纯文本 QPS 涨 | encoder 权重或视觉 KV 仍为文本请求常驻 | 只有 AR 权重在涨 |

定位时打印三张表：每请求 $(v,t,N)$、每阶段 batch 集合、每阶段页号区间。三张表能对上，再谈内核。对不上，换内核是浪费。第三张表的起始页如果都是 0，先假设别名，不要先假设模型坏了。第一张表如果 $v$ 全相等且等于全局 max，先假设 fused 形状，不要先假设数据集分辨率真的一样。

线上事故的时间顺序也值得记一笔。合图往往先表现为“吞吐还行、偶发串图”，因为 QPS 低时 batch 里碰巧都是带图请求，$\rho$ 虚高。等到纯文本流量进来，空槽和别名同时爆发。所以回归负荷必须包含本课这三类请求，不能只用带图基准。第 01 课的 golden case 若全是 A2A，加三条：纯文本、单图问答、带动作（可用玩具 $H=2$）。服务回归不过这三条，不允许宣称 stage graph 完成。

## 12. 交付物

交一份目录，不要交一个“跑通了”的截图。目录可以很短，但七项都要能点开或翻到页码。缺 mask 手算、缺 Lab 预测记录、缺“未复现 91.4%”声明，这三项任何一项都会让别人把夹具读成论文成绩。

1. `stage-graph.json` 或等价图：节点名、`forward` / `preprocess` / transfer、调用频率。
2. `mask-workbook`：A/B 两行 mask、有效下标、无效下标、$\rho=20/30$、错误质量 0.4。
3. Lab 记录：预测选项、fused 的 $\rho$ 与 KV 别名、stage 对照、确认门禁通过。
4. CPU `run()` 的 `checks` 与 `metrics` 原文，不要手改 20 和 30。
5. 论文数字列：JCT / RTF / TPS / Table 1 / 视频 token 三元组，附 arXiv:2602.02204 与节号。
6. 和第 13 课的边界声明：一份不超过一页的对照表。
7. 未做项清单：没有 80GB 双卡就明确写“未复现 91.4%”。

缺第 7 项、却在摘要里出现 91.4%，交付失败。

实验记录的第一页固定四句话，防止以后回看时把夹具当成绩：

1. 浏览器实验是教学模拟，不是模型输出。
2. CPU 实验证明 mask 与有效计数可复核，不证明真机 JCT。
3. 20/30 与 116/216 是两套长度，只共享公式。
4. 91.4% 来自 vLLM-Omni §4.2、Qwen3-Omni、Transformers 基线、双卡 80GB、各数据集前 100 条。

四句缺一句，归档时补上。Lab 截图可以附，但截图不能替代 `checks` 原文。stage graph 图纸手画即可，节点用 Thinker / Talker / Vocoder 或编码 / decode / 动作，边标函数名。不要用 ASCII 箭头在 `text` 围栏里画框图；网站机制图组件会渲染 `lesson46Diagram`。图上七个节点从请求到分阶段输出，编排器是决策点：拒绝单条 CUDA graph，把视觉、AR、动作送到不同变换。事实框里的 91.4%、841.6 / 150.9 / 545.4、Table 1 毫秒数都指向论文节号，抄进交付物时连节号一起抄。

## 13. 前沿对照与改造方向

**公开方案。** 2024–2026 年的服务系统沿着两条线长，本课只引用已打开的页面。单模型 AR 线：vLLM 用 PagedAttention 把 KV 浪费从预分配整块压到接近一页内部碎片，吞吐相对 FasterTransformer / Orca 提升 2–4×（[arXiv:2309.06180](https://arxiv.org/abs/2309.06180)）；SARATHI 用 chunked prefill 让 decode 搭上 prefill 的计算饱和，LLaMA-13B 在 A6000 上 decode 吞吐最多 10×、端到端最多 1.33×（[arXiv:2308.16369](https://arxiv.org/abs/2308.16369)）；DistServe 把 prefill 与 decode 拆到不同 GPU，在时延约束内可服务 7.4× 请求或 12.6× 更紧的 SLO（[arXiv:2401.09670](https://arxiv.org/abs/2401.09670)）。多阶段 any-to-any 线目前公开的系统论文就是 vLLM-Omni：stage graph、阶段引擎、统一连接器。Qwen3-Omni 上相对 Transformers 基线 JCT -91.4%；BAGEL T2I 2.40×；MiMo-Audio 含编译 11.58×。 [第 20 课](20_unified_understanding_generation.md) 已经把 vLLM-Omni 点名为理解 / 语言生成 / flow sampling 的工程切法，本课把它从一句点名展开成可执行的调度协议。GPT-4o 级产品的服务编排没有公开技术报告可查其 CUDA graph 范围，其技术报告未公开该细节。

**差距。** 规模差：你没有 30B Thinker，也没有论文那两张 80GB 卡，也没有 librispeech / food101 / ucf101 的服务负荷。机制差：mask、$\rho$、KV 交集、阶段 batch 维不依赖模型大小。钱能缩小的是 JCT 绝对值；钱缩不小的是“合图是否改变语义”。vLLM-Omni 相对 Transformers 的数量级加速，来自阶段内引擎（continuous batching、paged KV、graph compilation）加上按阶段放卡。缩小版用两条长度为 3 和 9 的视觉请求就能复现“pad 会进注意力”和“别名页非空”。不能复现的是 12.97× Thinker TPS，因为没有 30B 和未优化的 Transformers 基线。

机制上可以追上的部分写清楚。padding mask、有效计数、合图失败判定、Thinker / Talker / Vocoder 三节点图，这些在 CPU 和浏览器里就能验收。规模上追不上的部分也写清楚：双卡 80GB、Qwen3-Omni 权重、100 条真实音频 / 图像 / 视频查询、Mooncake RDMA。报告里把能验收的机制和不能验收的规模分成两段。

**动手改造清单。**

1. **把 MiniMind-O 的一次生成拆成两个进程。** 改动位置：推理入口里 Thinker 与 Talker 的串行调用，改成两个引擎加一块共享内存或队列，传递 hidden state 而不是 KV 页。预算：0 训练，CPU 或单卡；用第 01 课的 golden case 比文本和对齐后的 codec。预期：同一条 golden 的文本 token 与 8 路 code 不变，日志里出现两份 KV 统计。失败判定：Talker 进程读到 Thinker 的物理页号，或文本发生变化。没有 GPU 时，用本课 CPU 夹具的 `stage_kv_collision_pages == []` 代替真进程，并在报告标明夹具。有单卡时优先改推理入口，不要为了这条改造去重训 MiniMind-O。拆进程前后的 golden case 文本必须逐 token 相同，8 路 code 的哈希必须相同；只变日志里的页表份数，不变模型输出。若输出变了，先查 hidden state 队列是否丢帧，再查 Talker 是否误读了 Thinker 的页。页表份数从一份变成两份、输出哈希不变，这一项才算改造 1 在缩小版里通过。
2. **变长视觉 mask 门禁。** 改动位置：任何把图像 batch 写成 `(B, max_v, d)` 的预处理，补上 mask，断言 `valid == sum(v_b)`。预算：CPU，用 A=3、B=9 这对长度。预期：错误地把下标 3 算进 A 的有效集合会触发断言。失败判定：断言在 mask 全 1 时仍通过。
3. **禁止跨阶段 CUDA graph。** 改动位置：捕获 API 的范围参数，写成按节点捕获。预算：若无 CUDA，用 Lab 的模式开关代替，标明教学模拟。预期：捕获范围若包含编码与动作两列，fused 的 $\rho$ 低于 stage。失败判定：两模式 $\rho$ 相同且 KV 别名均为否，说明合图没有真正锁全局 max。
4. **阶段资源表。** 改动位置：启动配置，给 Thinker / Talker / 动作专家三行内存与并行度，禁止只写一个 `tensor_parallel=2`。预算：无训练。预期：日志按阶段打印 batch 与页数。失败判定：三行配置在运行时被折叠成一个进程，`stage_batch` 各阶段相等且等于总请求数。

**顺手复现。** 论文结论“手工 `generate()` 用不上阶段内优化”（§2.2）在缩小版对应改造 1 的两份 KV 统计，预期同方向。论文结论“Talker 迭代更多所以占 JCT”（§4.2，545.4 vs 150.9）在缩小版不能复现绝对值，只能复现记账：音频 token 与文本 token 分列。论文结论“连接器可忽略”（Table 1）在缩小版不要去测毫秒；对应纪律是禁止把 JCT 差异写成连接器。论文结论“graph compilation 有收益”（MiMo-Audio 0.60 到 0.12）在缩小版对应改造 3：收益只许记在单节点捕获上。若有人在 Lab 里看到 fused $\rho\approx 0.5$ 就写“接近论文 91.4% 的另一半”，判失败。两个数字的单位分别是有效比和 JCT 降幅，不能兑。

改造实验的失败判定要能被本课夹具触发。实验 2 用下标 3 就能失败。实验 3 用 Lab 两种模式就能失败。实验 4 用 `stage_batch` 三列相等就能失败。实验 1 在无 GPU 时退回 KV 交集检查。四条都不必等 80GB。

若以后真的拿到双卡 80GB，把改造 1 升级成论文配置：Thinker 张量并行铺两张卡，Talker 钉 device-1，Vocoder 钉 device-0，负荷用各数据集前 100 条。失败判定仍不改成“必须达到 91.4%”。改成：相对本机 Transformers 基线，JCT 下降；Talker TPS 与 Thinker TPS 分列；连接器延迟低于 20 ms；`valid_tokens` 与 `padded_tokens` 同时出现。91.4% 是作者机器、作者基线、作者 100 条查询上的点估计，不是本课门禁。门禁用方向和记账，不用点估计。

统一模型（Show-o / Janus / BAGEL）的服务切法与 Qwen-Omni 不同，但 stage 原则可迁移。BAGEL 的理解和生成专家在 vLLM-Omni 里被看成两个阶段（论文 Figure 2c）。训练时它们可以共享一部分主干；服务时仍然按请求类型决定进哪个专家引擎。第 20 课的 Arm B 在缩小版里是理解 adapter 加 flow head；服务时对应两个节点，而不是把 Euler 积分步编进 Thinker 的 CUDA graph。迁移时只搬“节点 + 边 + 每节点自己的图编译”，不搬 Qwen 的 Thinker2Talker 函数名。

## 14. 论文与必读材料

按“多阶段服务、单模型 KV、拆分与预填充”顺序读。每篇材料对应一个能在 mask 表或 Lab 里验证的问题。

### 14.1 多阶段 any-to-any 服务

- [vLLM-Omni](https://arxiv.org/abs/2602.02204)：读 §2 动机、§3.2 阶段抽象、§3.3 执行与流式、§3.4 连接器、§4.2 端到端、Table 1。带着问题：Qwen-Omni 为什么不能表达在 step-centric 的 vLLM 接口里？Talker 的 preprocess 为什么每步都跑，而 Thinker2Talker 默认只跑一次？91.4% 的分母是谁？读完写出三个节点的 batch 维，以及视频 841.6 / 150.9 / 545.4 分别进哪一维。HTML：[v1](https://arxiv.org/html/2602.02204)。仓库：[vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni)。
- 读 Figure 7 的时间分解时再带一个计算题：若文本输出 150.9、音频 545.4，且每步耗时相近，Talker 迭代大约是 Thinker 的 $545.4/150.9\approx 3.6$ 倍。把 JCT 优化全部砸在 Thinker 上，上限清晰。论文还报告 Thinker TPS 相对收益可以高于 Talker（Qwen3-Omni 12.97× vs 7.98×），那是因为基线 Thinker 更大、更吃未优化的 Python / 未编译图。两句话同时成立：迭代次数在 Talker，相对加速可以在 Thinker。读的时候不要只抄其中一个。

### 14.2 单模型 KV 与组批

- [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)：读 §2 的 prefill / decode 分解、§3 的碎片数字 20.4%–38.2%、§4.1–4.4 的页表与 copy-on-write。带着问题：平行采样共享的是哪一段 KV？为什么那不能推广成 Talker 共享 Thinker 的物理页？把答案写进本课 5.4 的三种关系表。
- [SARATHI](https://arxiv.org/abs/2308.16369)：读 chunked prefill 与 decode-maximal batching。带着问题：prefill 切块解决的是同一 AR 阶段里 compute 与 memory 两种步的拼盘，还是理解与 DiT 的拼盘？答案必须是前者。vLLM-Omni 把 SARATHI 继承在阶段内（§3.3），本课禁止把它写成跨阶段合图的理由。

### 14.3 拆分，但仍然是同一个 LLM

- [DistServe](https://arxiv.org/abs/2401.09670)：读 prefill / decode 干扰、TTFT 与 TPOT 解耦、7.4× / 12.6× 的前提（约束内、$>90\%$ 请求满足时延）。带着问题：拆开的两端是否共享同一套权重和同一套 KV 语义？是。那它是 EPD / PD 家族，不是 stage graph 家族。数字可以进“单模型服务”笔记，不能进“Omni 多阶段”总表。

读 Table 1 时做一个数量级核对：Thinker2Talker 共享内存 5.49 ms，Mooncake 8.28 ms。视频任务 Talker 要走约 545 步，若每步 10 ms 量级，仅 Talker 就数秒。连接器即使来回两次，仍远小于阶段内计算。因此优化顺序是：先拆阶段并让每阶段用上 paged KV 与 graph compilation，再考虑跨机 RDMA。反过来先上 Mooncake、仍用手工 `generate()`，论文没有提供这种配置的收益数字，本课也不发明。

读 §3.5 硬件插件时带着一个边界问题：跨平台是否改变 stage 抽象？论文的回答是插件注册硬件实现，编程模型不变。本课因此不把“换成某品牌加速器”列为改造实验。换硬件可以改连接器后端和核函数，不该改节点与边的契约。契约一改，Qwen-Omni 那张三节点图就要重写，收益无法和论文 §4.2 对照。

读 DistServe 摘要里的 7.4× 与 12.6× 时，把约束抄下来：在 TTFT 与 TPOT 同时满足、且超过 90% 请求落在时延约束内的前提下，系统能承接的最大速率。缺约束的倍数本课不用。把它和 vLLM-Omni 的 91.4% JCT 降幅比较时，问三件事：基线是谁、模型是不是 any-to-any、指标是速率还是单请求完成时间。三件事有一件不同，就分开放。SARATHI 的 10× 是 decode 吞吐、1.33× 是端到端，同样要带模型名 LLaMA-13B 和设备 A6000。本课论文节把这些数字留下，是为了防止“服务系统加速”变成一个无单位的形容词。

读完材料回头看：权重一行都不必改，服务却已经从一次 `generate()` 变成一张可审计的 stage graph。同一条请求在 fused CUDA graph 和 stage graph 上的 $\rho$ 差异，不是模型变聪明了，而是调度把无效位置从分母里拿掉了。你现在应该能拿着任意一份 Omni 推理脚本，在五分钟内指出它的节点、边、mask 和页表，并决定能不能捕获成一条图。下一课把评测数字拆桶：JCT 再低，也不能和 MMMU、LIBERO、OSWorld 横着写成同一种能力。
