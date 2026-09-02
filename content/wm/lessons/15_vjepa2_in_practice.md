---
id: 15_vjepa2_in_practice
title: "用 V-JEPA 2 的公开权重做评测"
summary: "一个 2B 的视频表征模型，“理解物理”体现在哪些可测量的地方？"
unit: jepa
play_tools: []
checkpoints:
  - "探针评测报告（自选下游任务）。"
  - "V-JEPA 2-AC 接口笔记。"
---

# 第 15 课：用 V-JEPA 2 的公开权重做评测

> 类型：实战/体验（跑公开权重做探针与微调，**不从头训练**）+ 精读（V-JEPA 2-AC 只读接口，不复现训练）<br>
> 建议周期：2-3 天<br>
> 硬件：单张 24GB 卡可完成全部必做实验；Mac 可完成权重加载、特征提取和 AC 笔记（视频解码用 torchcodec 或 eva-decord 绕开 decord）<br>
> 锚定仓库：[facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)（4.4k★，MIT，活跃维护，2026-03 发布 V-JEPA 2.1）<br>
> 产物：探针评测报告（Diving48，冻结 backbone + attentive probe + 小数据微调对照）+ V-JEPA 2-AC 接口笔记（与第 05、14 课规划器的同构对照表）

## 1. 这一课做什么

前两课用小模型确认了 JEPA 的训练目标、防坍缩机制和动作条件规划。本课转向 V-JEPA 2
的公开权重。Meta 使用超过 100 万小时互联网视频训练这一系列模型，规模从约 8000 万
参数的 ViT-B 到 20 亿参数的 ViT-G；其中仍能看到 EMA teacher、多块 masking 和表征
空间损失等设计。

实验分三部分。第一，加载权重并冻结 backbone，在 Diving48（48 类跳水动作
识别）上训一个 attentive probe，拿到你自己的探针评测数字；第二，在小数据子集上
比较冻结探针与端到端微调，判断小数据场景下是否值得更新 backbone；第三，分析
V-JEPA 2-AC，Meta 把这个 action-free 的视频表征模型接上 62 小时机器人数据、
变成能在真实 Franka 机械臂上零样本规划的动作条件世界模型。AC 的训练超出课程算力
（也超出大多数实验室的算力），我们不复现，但把接口彻底讲透：动作怎么进预测器、
规划时能量怎么算、和你第 05 课的 CEM、第 14 课的 Two Rooms 规划是同一张图纸。

“理解物理”需要拆成不同测量：动作分类探针、违反预期视频的表征变化，以及机器人任务
的实际成功率不是同一种能力，不能合并成一句结论。本课报告会并列随机猜测基线（2.1%）、
冻结特征上的探针精度和官方分类权重在同一测试集上的结果，并说明协议差异。这个拆法
会直接用于第 16 课的路线比较和第 17 课的评测设计。

术语速查：

| 术语 | 一句人话 |
|---|---|
| backbone | 模型的主干编码器，本课全程冻结它，只在上面加小零件 |
| attentive probe | 注意力探针：一个带可学习查询向量的小注意力模块加线性层，从几千个 token 里"点名"任务相关的信息 |
| 线性探针 | 最朴素的探针：冻结特征上直接放一层线性分类器，第 07 课用它读过 MuZero 的隐状态 |
| tubelet | 视频版的 patch：2 帧 × 16 × 16 像素的小时空块，视频 token 化的最小单位 |
| fpc | frames per clip，一段输入视频取多少帧；HF 模型名里的 fpc64 就是 64 帧 |
| SSv2 | Something-Something v2 数据集：类别是"把东西从左往右推"这种，不看时序做不对 |
| Diving48 | 48 类竞技跳水数据集：背景全是跳台和水花，单帧几乎给不出类别信息，专测动作理解 |
| EK100 | Epic-Kitchens-100 动作预判任务：看第一视角厨房视频，预判接下来会发生的操作 |
| 违反预期（VoE） | 发展心理学借来的测法：给模型看物理上不可能的视频，看它"惊讶"不惊讶 |
| V-JEPA 2-AC | 在冻结的 V-JEPA 2 上后训练出的动作条件预测器，机器人规划用 |
| 能量 | 第 14 课的老朋友：一个"当前状态与目标有多不匹配"的标量，规划就是把它压低 |
| receding horizon | 规划一段、只执行第一步、下一步重来，第 05 课 MPC 的标准用法 |

## 2. 问题

1. 把"理解物理"变成三个可测量的口径。探针分类精度、违反预期的惊讶度、机器人
   规划成功率，这里把 V-JEPA 2 系列论文里支撑这句话的每一类证据都过一遍：数字
   是多少、量法是什么、量法的边界在哪。你自己动手测的是第一种，另外两种精读论文
   原文，只转述核实过的数字。
2. 学会拿别人发布的大模型权重干活的完整工作流。加载、看形状、冻结、加探针、
   小数据微调、和官方参考结果对表。这套流程和第 08 课用 TD-MPC2 官方 checkpoint、
   第 09 课用 IRIS 预训练权重是同一门手艺，但这次模型大了两个量级，工程细节
   （视频解码、显存、评测配置）都更接近你以后工作里的真实情况。
3. 读懂一个工业级动作条件世界模型的接口。V-JEPA 2-AC 是这门课出现的第一个
   部署到真实机器人上的世界模型。它的训练你复现不了，但它的接口，冻结表征、
   动作条件预测器、能量最小化选动作，每一块你都在前面的课里造过缩小版。
   读懂它，第 16 课的路线对决你才有资格下判断。

先划清本课的档位：**实战/体验档**。按课程复现清单（CURRICULUM §4），跑公开权重
不算复现；本课所有训练都发生在探针和小规模微调层面，backbone 从头到尾是 Meta 训
好的。课文里凡是引用论文数字的地方，都注明出处；你自己跑出的数字和论文数字之间
的差距，第 9 节会教你逐项归因。

## 3. 准备

- 完成第 13、14 课。这里默认你已经清楚 JEPA 的三件套（context 编码器、EMA target
  编码器、预测器）和"能量"这个词的用法，相关机制见第 13、14 课。
- 一张 24GB 显卡（探针训练和微调用）；Mac 用户可以完成权重加载、特征提取、AC
  notebook 和所有精读环节，探针训练会很慢但不是不可能。
- Python 3.12 环境（仓库 README 的推荐配置），PyTorch 按官网装。
- 磁盘：权重按 fp32 估算，ViT-L 约 1.2GB，AC 的 ViT-g 编码器加 300M 预测器合计
  5GB 上下；Diving48 视频数据预留几十 GB。
- macOS 的 decord 坑，动手前先看：仓库依赖 decord 做视频解码，README 明说
  decord 不支持 macOS 且已停止维护，社区验证过的替代品是 eva-decord（仓库 PR #1）
  或 decord2（PR #31），任选。更省事的路线：Mac 上只走 HuggingFace 这条线，官方
  模型卡的示例代码用 torchcodec 解码视频，完全不碰 decord。
- HuggingFace 权重无需申请，MIT 许可（仓库里有三个数据工具文件是 Apache 2.0，
  用权重不受影响）。

## 4. 学习目标

1. 一行代码加载 V-JEPA 2 任一档权重（torch.hub 和 HuggingFace 两条路都会），并
   能推算给定帧数和分辨率下的 token 数与特征形状；
2. 说清 attentive probe 和线性探针量的东西差在哪，以及为什么两个探针协议下的
   数字不能直接比；
3. 用仓库的 `evals/` 在一个下游视频任务上完成冻结探针评测，拿到能和官方口径
   对表的数字；
4. 用实验回答：小数据场景下，冻结探针和端到端微调谁更稳，为什么；
5. 白纸画出 V-JEPA 2-AC 的推理数据流：目标图像和当前观察从哪进、动作序列从哪进、
   能量在哪算、CEM 在哪循环、机械臂执行哪一步；
6. 填完一张三列同构表：第 05 课 PlaNet 的 CEM 规划、第 14 课 Two Rooms 规划、
   V-JEPA 2-AC 规划，逐行对出状态、动力学、打分、搜索、执行五个槽位各放了什么。

## 5. 原理

五个机制，每个按老节奏走：直觉、机制、数学、代码落点、验证。

### 5.1 从图像 JEPA 到视频 JEPA：mask 从平面变成时空块

第 13 课的 I-JEPA 遮住图像的一块区域，让模型从周围猜被遮区域的表征。
把同一招搬到视频，有个偷懒解法会立刻废掉训练目标：如果只在单帧上挖洞，模型可以
从相邻帧把答案抄过来，第 t 帧被遮的那块，第 t+1 帧多半原样摆在那。抄袭比理解
便宜，模型一定选抄袭。所以视频 JEPA 的 mask 必须是**时空块**：同一个空间位置在
连续多帧上一起遮掉，逼模型去推"这个东西接下来会怎么动"，而不是"隔壁帧长什么样"。

V-JEPA 2 把视频切成 2 帧 × 16 × 16 像素的 tubelet（视频版 patch），沿用
V-JEPA 的 multiblock masking：采样几个大的连续时空块作为遮挡区域，短边贯穿整个
时间轴。训练目标和第 13 课完全同构：context 编码器看露出来的 tubelet，预测器在
表征空间猜被遮 tubelet 的表征，target 由 EMA teacher（学生权重的滑动平均）提供，
L1 损失只算被遮的位置，target 侧停梯度。规模是新东西：预训练数据 VideoMix22M
共 2200 万个样本、超过 100 万小时视频（论文口径，混合了 SSv2、Kinetics、
HowTo100M、YT-Temporal-1B 的筛选子集和 ImageNet 图像）。两个为了"大"而生的
工程件值得记：位置编码用 3D-RoPE（把特征维度切成时间、高、宽三段分别做旋转
编码，论文报告它让最大档模型的训练比绝对位置编码稳定）；训练日程用渐进分辨率
（主段 16 帧 256 分辨率，只在最后的退火段升到 64 帧 384 分辨率，论文报告这比
全程高分辨率省 8.4 倍 GPU 时间）。

与第 13 课同一条损失，换了记号也认得出：

$$
\mathcal{L} = \frac{1}{|M|} \sum_{i \in M} \bigl\| P_\phi\bigl(E_\theta(x_{\setminus M})\bigr)_i - \bar{E}_{\bar\theta}(x)_i \bigr\|_1
$$

$M$ 是被遮 tubelet 的下标集合，$E_\theta$ 是 context 编码器，$P_\phi$ 是预测器，
$\bar{E}_{\bar\theta}$ 是 EMA target 编码器。和你手写版的差别只有三处：patch 换成
tubelet、L2 换成 L1、模型放大一千倍。

编码器在 `src/models/vision_transformer.py`，预训练用的预测器在
`src/models/predictor.py`（带 mask token 的那种，和第 13 课你写的角色相同）。
HuggingFace 版的配置字段直接印证上面的机制：`tubelet_size: 2`、`patch_size: 16`、
`frames_per_clip: 64`、`crop_size: 256`。

加载权重后算一笔 token 账：64 帧除以 tubelet 的 2 得 32 个时间片，
256 除以 16 得每片 16 × 16 = 256 个空间位置，合计 32 × 256 = 8192 个 token。
第 7 节 Step 2 你会看到 ViT-L 输出形状是 (1, 8192, 1024)，形状对得上，
说明你真的理解了 token 化怎么发生的。

### 5.2 attentive probe：探针变强了，量的东西也变了

探针你用过两次：第 02 课在 VAE latent 上、第 07 课在 MuZero 隐状态上，
都是线性探针，32 维或几百维向量上放一层逻辑回归。现在特征是 8192 个 token，
每个 1024 维。线性探针的标准做法是先平均池化再分类，但"手正在把杯子从左推到右"
这个信号只活在少数 token 里，平均池化等于让几千个背景 token 投票，把信号淹掉。
attentive probe 的解法：给探针配一个可学习的查询向量，用 cross-attention 从 8192
个 token 里按需检索，再分类。探针自己会学"该看哪"。

仓库的探针（`src/models/attentive_pooler.py`）是一个小注意力模块：
可学习 query 对全部 token 做 cross-attention 池化，Diving48 评测配置给它配了
4 个注意力块、16 个头（`configs/eval/vitl/diving48.yaml` 里的
`num_probe_blocks: 4`、`num_heads: 16`）。HuggingFace 版的
`VJEPA2ForVideoClassification` 结构相同：官方文档写明它是"attentive pooler 之上
一层线性分类器"。训练时 backbone 全程冻结，梯度只流过探针，探针相对 backbone
是个小零件，但它自己是带学习能力的，这一点马上要较真。

核心就一行 cross-attention：

$$
\mathrm{probe}(H) = \mathrm{softmax}\!\left(\frac{q\,K(H)^\top}{\sqrt{d}}\right) V(H)
$$

$H$ 是冻结特征，$q$ 是可学习查询向量，$K$、$V$ 是探针自己的投影。分类头接在
池化结果上。

这算不算作弊？这是本课最值得较真的问题。探针从线性换成注意力，量的东西就
变了：线性探针测"信息是否线性可读"，attentive probe 测"信息是否以可被注意力
检索的形式存在"，后者是个宽松得多的标准，探针本身就带几百万参数的学习能力。
所以规矩是：**比较两个 backbone，探针协议必须一致；比较两篇论文的数字，先查
探针协议是否一致，不一致就不许放进同一张表**。V-JEPA 系论文自己是守这条规矩的：
README 表格里 InternVideo2-1B 的 69.7%（SSv2）和 V-JEPA 2 的 77.3% 是同一探针
口径下的数字。这不是 V-JEPA 独有的问题，第 17 课评测学你会看到，"量法是结论
的一部分"在世界模型评测里处处适用；本课先在探针这个最小案例上把它焊进习惯。

`src/models/attentive_pooler.py`（探针本体）、
`evals/video_classification_frozen/`（冻结分类评测的完整流程）。

第 11 节改造实验 1：同一份冻结特征上，平均池化加线性层对比 attentive
probe，两个数字的差就是"注意力检索榨出来的那部分信息"，你会量出量法
本身值多少个百分点。

### 5.3 "理解物理"的三个可测量口径

"模型理解物理"是一句检验不了的话，除非你说清量法。V-JEPA 2 系列
论文实际用了三种，能力各不相同，别混。

口径一：探针分类，时序信息在不在特征里。SSv2 的类别是"把东西从左往右推"
"假装把东西放进盒子"，Diving48 的类别是 48 种翻腾转体组合，两者的共同点是单帧
不够、必须看运动。V-JEPA 2 的冻结探针成绩（仓库 README 口径，其中 SSv2 与
EK100 两项亦见论文摘要）：SSv2 77.3%
（ViT-g 384 档；ViT-L 档 73.7%），Diving48 90.2%（ViT-L 档 89.0%），EK100 动作
预判 recall-at-5 39.7（此前最好公开结果 27.6）。这个口径量的是**运动与交互模式
被编码了**，还谈不上物理定律。

口径二：违反预期，给模型看不可能的世界。这套测法在单独一篇论文里
（arXiv:2502.11831，本课必读第 4 篇）：给模型看成对视频，一条物理正常，一条有
鬼（物体被挡住后凭空消失之类），量模型的"惊讶度"。惊讶度不用另训，就是
V-JEPA 的本职工作：拿前文预测后文的表征，预测误差越大越惊讶。模型若在不可能
视频上系统性更惊讶，就说明对应的物理属性（物体恒存、形状恒常等）已经在表征里。
结果：V-JEPA 类模型在 IntPhys 基准上 98%、GRASP 66%、InfLevel-lab 62%（论文
均值口径），而像素空间预测的 VideoMAEv2 和多模态大模型（论文测了 Qwen2-VL-7B、
Gemini 1.5 Pro）接近随机水平。论文同样如实报告了测不出来的：颜色恒常、固体性、
碰撞这几类属性上，V-JEPA 不显著优于未训练的基线。

口径三：规划成功率，表征能不能撑起动作。V-JEPA 2-AC 在真实 Franka 机械臂
上零样本执行（机制下一节讲），论文报告的成功率：伸够 100%，抓杯子 60%，抓盒子
20%，杯子搬运 80%，盒子搬运 50%。注意这些数字的诚实之处：抓盒子只有 20%，论文
没有藏。此外论文还有第四类证据，把 V-JEPA 2 接上语言模型做视频问答（8B 规模下
PerceptionTest 84.0、TempCompass 76.9），这里不展开。

规模和"物理直觉"是什么关系？如实转述，这里没有干净的涌现故事。三个方向的
证据：其一，VoE 论文发现物理直觉出现得**早**，1.15 亿参数的模型在 IntPhys 上
就超过 85%，把训练数据砍到 128 小时仍在各属性上保持 70% 以上的成对精度；其二，
探针成绩随规模涨得**慢而稳**，V-JEPA 2 论文的消融里，模型从 ViT-L 放大到
ViT-g（3 亿到 10 亿参数）在六个分类任务上平均只涨 1.5 个点，EK100 预判从 32.7
到 38.0 再到 39.7（加分辨率）近似线性；其三，复杂场景**大家都不行**，Meta
自己随 V-JEPA 2 发布的 IntPhys 2 基准（arXiv:2506.09849）把同类测试搬进复杂
合成场景，大多数模型掉回随机水平，人类接近满分。一句话总结这三条：基础物性
直觉小模型小数据就有，规模带来的是平滑的增益而非台阶，而"理解物理"离解决
还远。谁把 2B 说成物理直觉的门槛，你现在有三条证据反驳。

口径一就是你要跑的 `evals/`（`video_classification_frozen`、
`action_anticipation_frozen`、`image_classification_frozen` 三个子目录）；口径二、
三在论文里，仓库只发了 AC 权重和能量景观 notebook。

你自己的 Diving48 探针数字（第 7 节）加上对论文表格的复述，就是你
报告里"理解物理"一节的全部素材，每个数字都要能说出口径。

### 5.4 V-JEPA 2-AC：给冻结的眼睛配一个听动作的预测器

V-JEPA 2 本体是 action-free 的：100 万小时网络视频里没有一条动作
标签，它学到的是"世界自己会怎么动"。机器人需要的是"我这么动，世界会怎么变"
，第 03 课就立过的规矩：预测必须随动作分岔，否则只是惯性外推器。Meta 的解法
不碰那 100 万小时的成果：编码器整个冻结，另训一个会听动作的预测器接在上面。
便宜得惊人，后训练数据只有约 62 小时的 Droid 机械臂视频（约 2.3 万条轨迹，
连失败的都要），不到预训练数据的万分之一。

推理时的数据流：每帧图像过冻结的 ViT-g 编码器，得到该帧的表征
（论文口径：每帧 16 × 16 个 patch、1408 维）。动作条件预测器是一个约 3 亿参数的
transformer（24 层、16 头、1024 维），输入按时间排成序列：每个时间步放三样
东西，该帧的 patch 表征、7 维动作（末端执行器的 3 维位移、3 维欧拉角转动、
1 维夹爪开合）、末端执行器状态。注意力是 **block-causal**：同一时间步内随便看，
跨时间步只准看过去，这让它可以自回归往前滚：predict 出 t+1 的表征后，把它当
输入接着 predict t+2。训练目标两项：teacher forcing 的 L1（每步都喂真表征、猜
下一步）加一个两步 rollout 损失（把自己的预测喂回去再猜一步），后者专治你
第 03 课看过的误差滚雪球。

规划就是能量最小化。给一张目标图像 $g$（想让世界变成的样子），
候选动作序列 $a_{1:T}$ 的能量定义为：

$$
E(a_{1:T}) = \bigl\| \hat{s}_{T}(a_{1:T}) - E_\theta(g) \bigr\|_1
$$

$\hat{s}_T$ 是从当前观察出发、按候选动作自回归滚 $T$ 步后的预测表征，
$E_\theta(g)$ 是目标图像的编码。搜索器是你第 05 课手写过探针的 CEM：高斯分布
初始化为零均值单位方差，每轮采 800 条动作序列、按能量排序、取尖子重估均值方差，
迭代 10 轮（论文配置）。执行是 receding horizon：只执行第一个动作，机械臂动完、
拿到新观察、整个流程重来。两个落地细节见论文：每步动作被限制在 L1 半径 0.075 的
球内（约 13 厘米的末端位移上限，防 CEM 采出暴走动作）；长任务拆成子目标图像序列
（抓住、移到目标上方、放下），按固定步数切换。论文报告的推理速度：每个动作约
16 秒，对照组 Cosmos 用推理式规划要约 4 分钟。

类比与失效处。你可以把 AC 理解成"给冻结的行车记录仪配了个副驾"：记录仪
只管把画面变成表征，副驾听着你打算怎么打方向盘，在表征空间里预判下一秒。类比
失效的地方：副驾没见过方向盘之外的世界变化，62 小时数据全是单臂桌面操作，
预测器学到的动作效应仅覆盖这个分布，换个双臂或移动底盘场景，冻结的眼睛还能用，
副驾得重训。

预测器在 `src/models/ac_predictor.py`（对照 `predictor.py` 读，
一个吃 mask token，一个吃动作 token）；一行加载在 `hubconf.py` 注册的
`vjepa2_ac_vit_giant`，返回（编码器，AC 预测器）二元组；训练配置
`configs/train/vitg16/droid-256px-8f.yaml`；能量景观演示
`notebooks/energy_landscape_example.ipynb`，仓库自带一条真实 Franka 轨迹
`notebooks/franka_example_traj.npz`，没有机器人也能跑。

跑通能量景观 notebook 后做一个最小实验：把轨迹里某一步的真实动作
替换成反方向，能量应该升高。动作对换的老试金石（第 03 课），用在 20 亿参数的
模型上，问题一个字没变。

### 5.5 三代规划器同构：你已经第三次写同一张图纸

这门课到现在出现过三个"在潜空间里选动作"的系统。把它们并排放好，
你会发现换的从来是零件，不是图纸。

逐槽位对照（第 14 课那列以你跑过的 eb_jepa 实验为准）：

| 槽位 | 第 05 课 PlaNet | 第 14 课 eb_jepa Two Rooms | 本课 V-JEPA 2-AC |
|---|---|---|---|
| 状态 | RSSM 双通道 $[h; z]$ | 小 JEPA 编码器表征 | 冻结 ViT-g 表征（10 亿参数） |
| 动力学 | RSSM 先验（与编码器联合训练） | 动作条件预测器（与编码器同库训练） | 3 亿参数 block-causal 预测器（编码器冻结，后训练） |
| 打分 | 学出来的奖励头，累加多步 | 能量：预测表征与目标的距离 | 能量：滚 $T$ 步后与目标图表征的 L1 |
| 搜索 | CEM，1000 候选 × 10 轮 | 以第 14 课实验配置为准 | CEM，800 候选 × 10 轮 |
| 执行 | MPC：只执行第一步 | 同 | receding horizon：只执行第一步 |

真正的分水岭在打分那一行。PlaNet 的奖励头 $r_\psi(s)$ 必须用环境奖励
标签训练，没有奖励函数就没法规划；能量式打分只要一张目标图像，标签成本为零，
自监督一路到底。代价是表达力：能量只会说"到达长得像 $g$ 的状态"，说不了"越快
越好""累计得分最大"这类回报型目标。奖励头和目标图，各是一种任务语言。

三份代码你都摸过或即将摸到：第 05 课 `planner.py::MPCPlanner`
39 行、第 14 课 eb_jepa 的规划例子、本课 `notebooks/energy_landscape_example.ipynb`。

不看课文，把五行表默写出来，再答一个问题：为什么三家都选"只执行
第一步"？（答案在第 05 课 5.5 节：模型误差随推演步数滚雪球，第一步是可信度
最高的一步，执行完立刻用真实观察纠偏。）

## 6. 源码导读

克隆仓库后按这个顺序读，每个文件带着问题进去：

| 文件 | 是什么 | 带着什么问题读 |
|---|---|---|
| `hubconf.py` | 一行加载的真身 | `vjepa2_1_vit_base_384` 和 `vjepa2_ac_vit_giant` 各返回什么？权重从哪个 URL 拉？|
| `src/models/vision_transformer.py` | 编码器 | tubelet 化在哪发生？3D-RoPE 把维度切成几段？|
| `src/models/predictor.py` | 预训练预测器 | mask token 怎么占位？对照第 13 课你手写的预测器找同名角色 |
| `src/models/ac_predictor.py` | AC 预测器 | 7 维动作在哪一层进入序列？block-causal 的 mask 怎么构造？|
| `src/models/attentive_pooler.py` | 探针本体 | 可学习 query 有几个？池化输出接的是什么头？|
| `evals/main.py` | 探针评测入口 | `--fname` 和 `--devices` 之外还接什么？怎么把一个 yaml 变成一次训练？|
| `configs/eval/vitl/diving48.yaml` | 你要改的评测配置 | 找到 `checkpoint_key: target_encoder`：为什么读的是 EMA teacher 那份权重？`out_layers: [17, 19, 21, 23]`：探针居然吃四层特征？|
| `notebooks/energy_landscape_example.ipynb` | AC 能量景观 | 能量对哪个动作维度最敏感？|
| `app/` | 预训练主循环 | 只读：VideoMix22M 的采样和 masking 在训练循环的哪个位置？|

两个当前状态提示（写课时以 GitHub 仓库 main 分支核实）：README 本地评测示例里
写的 `configs/eval/vitl16/...` 路径已经过时，实际目录只有 `configs/eval/vitl/` 和
`configs/eval/vitg-384/` 两个，跑的时候用后面这两个；`configs/eval/vitl/` 下现有
七个任务配置：`coin.yaml`、`diving48.yaml`、`ek100.yaml`、`in1k.yaml`、
`jester.yaml`、`k400.yaml`、`ssv2.yaml`，本课主线用 Diving48，其余你手头有哪个
数据集就能换哪个。

## 7. 实验

Step 1-2 是全平台热身，Step 3-6 是探针主线（需要 GPU），Step 7 是 AC 精读。
每一步先写预期，再跑，再对照。

### Step 1: 装环境

```bash
conda create -n vjepa2-312 python=3.12
```

```bash
conda activate vjepa2-312
```

```bash
git clone https://github.com/facebookresearch/vjepa2.git
```

```bash
pip install -e .
```

以上是 README 的推荐流程（PyTorch 先按官网装好）。macOS 用户在 `pip install -e .`
卡在 decord 时按第 3 节的说明处理：装 eva-decord 替代，或干脆只走 HuggingFace 加
torchcodec 路线完成本课（Step 2 的第二段代码就是这条路线）。

### Step 2: 一行加载，先对形状账

先用 torch.hub 加载课程指定的最小档，V-JEPA 2.1 的 ViT-B（8000 万参数，
384 分辨率），确认权重真的能到你手上：

```python
import torch

encoder = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_1_vit_base_384')
n_params = sum(p.numel() for p in encoder.parameters())
print(f"params: {n_params/1e6:.0f}M")   # 预期：80M 上下
```

再走 HuggingFace 这条线加载 ViT-L 并跑一段真视频，验证 5.1 节的 token 账
（模型卡的官方示例，用 torchcodec 解码，Mac 可跑）：

```python
import numpy as np
from torchcodec.decoders import VideoDecoder
from transformers import AutoVideoProcessor, AutoModel

hf_repo = "facebook/vjepa2-vitl-fpc64-256"
model = AutoModel.from_pretrained(hf_repo)
processor = AutoVideoProcessor.from_pretrained(hf_repo)

vr = VideoDecoder("你的任意一段mp4")
video = vr.get_frames_at(indices=np.arange(0, 64)).data   # T x C x H x W
inputs = processor(video, return_tensors="pt")
feats = model.get_vision_features(**inputs)
print(feats.shape)   # 预期：(1, 8192, 1024)
```

8192 = 64 帧 ÷ 2（tubelet）× 16 × 16（256 ÷ 16 的平方）。形状对不上就停下，
先回 5.1 节把 token 化想清楚，这一步没过，后面全是玄学。

### Step 3: 准备 Diving48

从 Diving48 官方项目页（UCSD 的 RESOUND 项目，搜 "Diving48 dataset" 即到；页面
偶尔抽风，换个时间再试）下载视频包和官方 train/test 标注。然后自己写十几行胶水
生成两份 csv：`configs/eval/vitl/diving48.yaml` 里引用的
`Diving48_train_paths.csv` 和 `Diving48_test_paths.csv`，内容是视频路径与类别标签
（具体分隔格式以仓库 `src/datasets/` 的加载代码为准，别猜，打开看）。

抽查环节照旧：随机播放五段视频，确认画面是跳水、标签在 0 到 47 之间、路径全部
可读。数据错一位，探针白训一夜。

### Step 4: 冻结探针评测（本课主实验）

打开 `configs/eval/vitl/diving48.yaml`，改四处：`folder`（你的实验目录）、两个
csv 路径、`checkpoint`（指向下载好的 `vitl.pt`，README 的 V-JEPA 2 checkpoint 表
里有直链）。注意这份 yaml 的默认口径是官方集群配置，8 节点 × 8 卡、100 个
epoch、每卡 batch 2、三个学习率的探针并行扫描。单卡照跑不误（评测入口就是给
单机用的），但要把 `num_epochs` 先降到 20 并有心理预期：解码 32 帧 × 4 段 × 3
视图是 CPU 密集活，多给 `--help` 里的数据加载参数一点耐心。

```bash
python -m evals.main --fname configs/eval/vitl/diving48.yaml --devices cuda:0
```

预期：训练日志里探针精度从 2% 附近（48 类随机线）起步、随 epoch 稳步爬升，
20 epoch 的缩水档最终落在论文口径（89.0%）以下若干个点，差距来自 epoch 数、
多视图评测和学习率扫描的缺席，第 9 节逐项归因。三个学习率的探针哪个最好，
日志里直接可读。

### Step 5: 小数据微调对照

问题：手里只有小数据时，动 backbone 是好主意吗？做个干净的对照。从 Diving48
训练集每类抽 25 条（共 1200 条）做小数据集，同一子集跑两个设置：

- A 冻结探针：Step 4 的配置，把训练 csv 换成子集版；
- B 端到端微调：HuggingFace 的 `VJEPA2ForVideoClassification`（attentive
  pooler 加分类头，backbone 一起放开），骨架如下（胶水代码，训练循环自己补）：

```python
import torch
from transformers import AutoVideoProcessor, VJEPA2ForVideoClassification

model = VJEPA2ForVideoClassification.from_pretrained(
    "facebook/vjepa2-vitl-fpc64-256", num_labels=48,
).to("cuda")   # 分类头随机初始化，加载时的相应提示属预期
processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")

loss = model(**inputs, labels=labels).loss   # 训练循环里的核心一行
```

两边同样的 epoch 预算，B 用 bf16 加梯度累积伺候显存（3 亿参数全放开，24GB
够用但不宽裕）。在**完整测试集**上对比。预期方向：1200 条数据喂 3 亿参数，
B 的训练精度会很好看、测试精度大概率不如 A 稳，学习率稍大还会把预训练特征
洗坏，但这是预期不是结论，两条曲线放进报告，写你实际看到的。

### Step 6: 拿官方分类权重对表

HuggingFace 上有官方发布的 Diving48 分类权重
`facebook/vjepa2-vitl-fpc32-256-diving48`（模型卡未报精度数字，也未写明训练时
backbone 是否冻结，只当外部参考点用）。用它在你的测试集上推理一遍，得到第三个
数字。现在你手里有：随机线 2.1%、你的探针（Step 4）、你的微调（Step 5）、官方
分类权重（本步）。四个数字一张表，每个都注明口径，这是探针评测报告的骨架。

### Step 7: V-JEPA 2-AC 能量景观与接口笔记

下载 AC 权重（1B 编码器加 300M 预测器，fp32 合计 5GB 上下，24GB 单卡推理
无压力；Mac 走 CPU 很慢但能跑）：

```python
import torch

encoder, ac_predictor = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_ac_vit_giant')
```

然后跑仓库自带的能量景观 notebook：

```bash
jupyter lab notebooks/energy_landscape_example.ipynb
```

它用仓库自带的真实 Franka 轨迹 `notebooks/franka_example_traj.npz` 演示：给定
轨迹的观察和动作，算不同候选动作下的能量面。做两个动手改动：其一，把某一步
真实动作换成反方向，确认能量升高（5.4 节的验证）；其二，扫描单个动作维度
（比如末端 x 位移）画能量曲线，看最低点是否落在真实动作附近。

跑完写接口笔记，至少回答五问：动作在预测器的哪一层进入序列？注意力为什么是
block-causal 而不是全因果？能量为什么定义在表征空间而不是像素空间？规划为什么
只执行第一步？目标图像这种任务语言表达不了哪类任务？最后把 5.5 节的同构表
填一遍，放进笔记。

### Step 8: 留证据

老规矩，实验目录里放 `NOTES.md`：

```text
日期与机器、仓库 commit
四个数字及各自口径（随机线 / 探针 / 微调 / 官方权重，各自的帧数、视图数、epoch）
Step 4 的完整命令与 yaml 改动清单
Step 5 子集的抽样种子与两条训练曲线
AC 接口笔记与同构表的存放路径
```

## 8. 配置与预算

| 步骤 | 硬件 | 耗时（参考） | 说明 |
|---|---|---|---|
| Step 1-2 加载与形状账 | Mac/CPU 即可 | 半小时 | 权重下载走网络，ViT-B 数百 MB |
| Step 3 数据准备 | CPU | 半天（下载为主） | Diving48 几十 GB 量级 |
| Step 4 冻结探针 20 epoch | 单张 24GB 卡 | 数小时到一夜 | 瓶颈常在视频解码（CPU），不在 GPU |
| Step 5 微调对照 | 单张 24GB 卡 | 数小时 | bf16 加梯度累积，1200 条小数据 |
| Step 6 官方权重推理 | 单卡或 Mac | 一小时内 | 纯推理 |
| Step 7 AC notebook | 单卡最好，Mac 可忍 | 一两小时 | 纯推理，权重 5GB 上下 |

官方满配（yaml 默认的 64 卡 100 epoch、三学习率并行）不要求也不建议单卡硬扛；
要冲论文口径的数字，加钱不加课。更大 backbone（ViT-g 及以上）的探针与微调属于
课程蓝图里的"8 卡短租加餐"，非必做。

## 9. 验收

验收清单：

- [ ] Step 2 的两个形状账全对，并能口算任意 fpc 与分辨率组合下的 token 数；
- [ ] 探针评测报告四数字齐全（随机线 / 冻结探针 / 小数据微调 / 官方分类权重），
      每个数字旁边写清口径：帧数、时间段数、空间视图数、epoch 数、探针结构；
- [ ] 你的探针精度显著高于随机线，且能逐项解释与论文 89.0% 的差距来源（epoch
      少、无多视图评测、无学习率扫描、单卡 batch，一项项列，不许笼统写"配置
      小"）；
- [ ] 加做一条对照：同样的探针训练跑在**随机初始化、未预训练**的 ViT-L 上（用
      仓库 `src/models/vision_transformer.py` 初始化一个 vit_large、把 state dict
      存成 checkpoint、yaml 指过去），确认你的数字是预训练特征的功劳而不是探针
      自己学出来的，这条基线塌了，整份报告作废；
- [ ] Step 5 两条曲线（训练与测试）在报告里，结论如实写观察，不写"应该"；
- [ ] AC 接口笔记五问全答，能量对动作反向的敏感性实验有截图或数字；
- [ ] 5.5 节同构表默写版和课文核对无误；
- [ ] 口头关：向没上过这门课的人解释"V-JEPA 2 理解物理吗"，要求说满三个口径、
      每个带一个核实过的数字、外加一个它做不到的事（IntPhys 2 或抓盒子 20%）。

## 10. 排错

| 症状 | 最可能的原因 | 怎么确认 | 怎么修 |
|---|---|---|---|
| macOS 上装依赖卡在 decord | decord 不支持 macOS 且已停更 | pip 报错信息里有 decord | 装 eva-decord 或 decord2 替代（README 推荐，见仓库 PR #1、#31）；或全程走 HF 加 torchcodec |
| 找不到 `configs/eval/vitl16/...` | README 示例路径过时 | 去仓库看 `configs/eval/` 目录 | 用 `configs/eval/vitl/` 或 `configs/eval/vitg-384/` |
| 探针精度死在 2% 附近 | csv 标签错位、路径不可读，或 checkpoint 没加载上 | 单独跑一批数据打印标签分布；看日志里权重加载是否报 missing keys | 修 csv；确认 yaml 的 `checkpoint_key: target_encoder` 没改动 |
| 探针精度爬到十几就平了 | 视频解码大量失败、静默给黑帧 | 抽样解码几十条数据看是否报错 | 换解码后端或重下损坏视频 |
| Step 4 GPU 利用率极低 | 解码是 CPU 瓶颈 | 看 CPU 占用 | 加数据加载进程数；降低视图数先跑通 |
| Step 5 微调 OOM | 3 亿参数全放开加 64 帧输入 | 显存曲线 | bf16、梯度累积、减 batch，仍不行就减输入帧数 |
| 微调训练精度高测试塌 | 1200 条喂 3 亿参数，过拟合 | 两条曲线的剪刀差 | 这是预期现象，写进报告；想救就冻结大部分层或加增广 |
| 权重下载中断 | 网络 | 文件大小对不上 | HF 权重用 huggingface-cli 断点续传；fbaipublicfiles 直链支持重试 |
| AC notebook 能量面平坦无起伏 | 没用仓库自带轨迹，或动作扰动幅度太小 | 先原样跑 `franka_example_traj.npz` 复现默认输出 | 恢复默认输入，扰动幅度参考 5.4 节的动作上限 0.075 |

## 11. 前沿与改造

给机器人配世界模型，2025-2026 年公开工作里有两条路正面交锋。
一条是显式生成：把未来逐帧生成出来再挑动作，V-JEPA 2 论文里的对照组 Cosmos 走
这条路，代价在论文的速度对比里，规划一个动作约 4 分钟，V-JEPA 2-AC 约 16 秒，
因为后者只在表征空间滚未来，一个像素都不生成。另一条就是本课的配方，而且它正在
变成通用套路：拿海量无标签视频把眼睛训大训好，再用少量交互数据后训练一个动作
条件预测器，62 小时 Droid 数据换来两个实验室的 Franka 零样本执行，这个"底座
不动、小件后训"的杠杆比是全篇论文最值得记住的工程数字。仓库本身也在往前走：
2026-03 发布的 V-JEPA 2.1 给出时序一致稠密特征的配方，2B 的 ViT-G 加入 checkpoint
表，8000 万参数的 ViT-B 从命名看是 2B 模型的蒸馏版（checkpoint 文件名含
`dist_vitG`，README 未展开训练细节，以仓库后续文档为准）。

规模差距：100 万小时预训练、64 卡的探针满配、8B 档的视频问答
，这些是钱的问题，课程不碰。机制差距：零。本课你用的探针协议、能量规划、
微调判断，与论文同一套，单卡全部摸得到。这是"实战课"的全部意义：前沿系统的
机制层对你已经没有黑箱。

动手改造清单（选做，每个写清预算与失败判据）：

1. 量法本身值几个点：Step 4 的探针换成平均池化加线性层（改探针结构即可，
   特征不重算），同数据同 epoch。预算：数小时。预期：线性版明显低于 attentive
   版，差值就是 5.2 节"量法变了"的实测大小。失败判据：两者持平，那说明
   Diving48 的信号已经均匀摊在 token 上，把这个反直觉结果写进报告，第 17 课
   有用。
2. 规模曲线自己画：V-JEPA 2.1 家族内（同配方蒸馏系）ViT-B 对 ViT-L 跑同一
   探针协议。预算：Step 4 再来一遍，一夜。预期：L 高于 B，方向与论文的规模消融
   一致（论文 L 到 g 平均涨 1.5 个点）；失败判据：B 反超 L 且复查协议无误，
   如实上报，这会是个有意思的发现。
3. 帧数消融：`frames_per_clip` 从 32 降到 16 再跑探针。预算：半天。预期：
   Diving48 是时序任务，帧数砍半精度应明显下降；对照组 in1k（图像任务）应几乎
   不动。这是"这个任务真的在测运动理解"的直接证据。
4. 能量景观扩展：Step 7 的单维扫描扩成二维热力图（末端 x、y 位移联合扫描），
   看能量谷的形状与真实动作的偏差。预算：纯推理，一两小时。预期：谷底邻域平滑、
   离谱动作能量单调上升；失败判据：能量面出现大片平坦，记录下来，这正是
   目标图规划在该状态下退化的样子。

论文的规模消融结论（模型放大、探针分数稳步上涨）对应改造实验 2，
单卡可验方向；VoE 论文的"小模型也有物理直觉"不在本课复现范围（需要 IntPhys
评测管线），但它的开源结果与你在改造实验 2 里看到的"规模收益平缓"互为印证。

## 12. 论文与延伸

1. V-JEPA 2（Assran, Bardes, Fan, Garrido 等，
   [arXiv:2506.09985](https://arxiv.org/abs/2506.09985)），本课的主论文。带着
   三个问题读：标题里 understanding、prediction、planning 三个词各由哪个实验
   板块支撑，量法分别是什么？AC 后训练为什么冻结编码器，省算力之外，还防了
   什么（提示：62 小时数据要是放开 10 亿参数会发生什么）？找到能量最小化和
   CEM 的段落，核对 5.4 节的每个数字。
2. V-JEPA（Bardes, Garrido 等，[arXiv:2404.08471](https://arxiv.org/abs/2404.08471)）
   ，前作，读方法与评测协议。带着问题：只用特征预测一个目标（无像素重建、无
   对比损失）为什么够？它怎么给所有对照模型统一探针协议，这正是 5.2 节那条
   规矩的出处。
3. I-JEPA（Assran 等，[arXiv:2301.08243](https://arxiv.org/abs/2301.08243)，
   第 13 课已精读，本课回读），带着新问题扫一遍：你手写版里的 EMA、multiblock
   masking、预测器深度，哪些设计到了 V-JEPA 2 的尺度上被原样保留，哪些被换掉了
   （提示：位置编码和 mask 的时间维）？
4. 直觉物理从自监督预训练中涌现（Garrido 等，
   [arXiv:2502.11831](https://arxiv.org/abs/2502.11831)），口径二的原始出处。
   带着问题：惊讶度怎么从预测误差构造成对比指标？哪些物理属性测出来了、哪些
   没有？"128 小时数据就够"这个结论的实验设置是什么？
5. IntPhys 2（Bordes, Garrido 等，[arXiv:2506.09849](https://arxiv.org/abs/2506.09849)，
   选读），带着问题：从 IntPhys 到 IntPhys 2，场景复杂度加了什么，为什么就把
   大多数模型打回随机水平？人与模型的差距具体差在哪类判断上？

到这里，第四幕三段路走完了：第 13 课你手写了 JEPA 的骨架，第 14 课给它接上动作
学会了规划，本课确认了同一套机制放大到 20 亿参数、100 万小时数据后真的能在
机械臂上干活，并且学会了用三种口径去量"它到底懂了多少"。第 16 课
是路线对决：重建像素、预测表征、结构化装配，三条路各自的证据都已经在你自己的
报告里了，第 08 课的重建对免重建、第 10 课的 token 对扩散、第 14 课的 JEPA 对
RSSM，加上本课的探针报告。同一笔算力押哪条路？下一课用你自己的数据说话。
