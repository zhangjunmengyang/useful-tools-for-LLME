---
id: 31_interaction_memory
title: "人会不会看过来"
summary: "桌宠的世界里最大的外生过程是人。人不是可控动作，但必须被预测。"
unit: deskpet
play_tools: []
checkpoints:
  - "人反应预测相对惯性基线的提升或失败分析。"
  - "一段说明：人格不是动力学。"
---

# 第 31 课：预测对面那个人下一秒会不会看过来

> 类型：实战（自己桌上采真实互动、训小预测头）+ 研究（短时记忆消融；风格先验与世界模型分界）<br>
> 建议周期：2-3 天<br>
> 硬件：S 档笔记本或手机摄像头 + CPU/Mac 可完成全部必做实验；单张 24GB 卡只在选做 Gaze-LLE 权重体验时用得上<br>
> 锚定：官方 [MediaPipe Face Landmarker Python 指南](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python) 与任务模型；对照精读 [ejcgt/attention-target-detection](https://github.com/ejcgt/attention-target-detection)（Chong 等，CVPR 2020）。桌面日志沿用第 29、30 课，没有就本课按协议现采。<br>
> 产物：1 秒后「人是否看镜头 / 手是否靠近杯子」的预测头、惯性基线对照、环形短时记忆消融记录、摄像头直播概率条

## 1. 这一课做什么

整门课的循环还是那句：

```text
观察 先压成状态 再按动作预测下一状态 然后展开多条未来 给未来打分 最后选动作
```

你现在在第八幕。第 28 课把桌子收成有界问题，第 29 课把像素收成状态，第 30 课给这张桌子训了一个会听动作的世界模型：$P(s_{t+1} \mid s_t, a_t)$。头左转和手往杯方向伸，未来必须分岔，那是你自己身体的动力学。

这一课要加的零件是对面那个人。人是这张桌子上最大的外生过程：他会看过来、低头看手机、伸手碰杯、挥手、出声。这些量进状态、进预测，但不进你的动作空间。你不能把「请看镜头」当成 $a_t$ 喂进去还指望干预成立。桌宠若不会预测人，第 32 课的「对视」只能写成「画面里有脸就转头」，那是反应式摆件。

具体要造三样东西。第一，一个小头，吃最近的状态，吐 1 秒后两个二元事件的概率：人是否看镜头、手是否靠近杯子。第二，一条必须先过的基线：永远猜当前值，学术上叫持续预测或惯性基线。人经常连续盯着同一处好几秒，这条基线会很强；你的模型若只在整体准确率上略胜，多半是在背「他现在看着，下一秒大概率还看着」，不是真的在预报转折。第三，一个环形短时记忆：最近 $N$ 步状态，外加几个未完成意图位（手还举着、头还低着、正在靠近杯子）。记忆的唯一用途是条件化预测，不是给桌宠写人物小传。

规划器可以有风格先验：更爱对视、更少伸手。那是选动作时的偏好，不是世界怎么动。把「我是一个粘人的桌宠」写进动力学，等于让模型在人已经低头看手机的时候，仍想象对方会看过来，因为人设如此。下一课装身体时，这种幻觉会变成不停打扰。

做完你能验证的是：在自己的桌子上，1 秒后人反应的预测相对惯性基线有没有增益；环形记忆有没有增益；没有增益就按第 9 节的口径写失败分析，不许用聊天模型扮演用户来把数字做漂亮。

术语速查：

| 术语 | 一句人话 |
|---|---|
| 外生过程 | 会改变世界、但你下不了命令的那部分，这里主要是对面的人；天气、另一只宠物也属此类 |
| 人状态 $s_t^{\mathrm{human}}$ | 从画面推断的注意、朝向、手的位置，是状态的一部分，不是动作 |
| 注视估计 | 从当前帧读出人现在看哪；MediaPipe 给的是朝向和眼部 blendshape，不是眼动仪 |
| 注视预测 / 预判 | 从过去几帧预报未来看哪；本课的目标是 1 秒后的两个二元事件 |
| 惯性基线 | 永远用当前标签当未来标签；人盯着不动时它几乎满分 |
| 转折帧 | 当前标签和 1 秒后标签不同的那些时刻，惯性基线在这里全错 |
| 环形短时记忆 | 固定长度的最近 $N$ 步缓冲，新状态进来、最老的挤出去 |
| 未完成意图 | 已经开始、尚未结束的短过程：手举着、头低着、正在靠近杯子，用滞回开关表示 |
| 风格先验 | 规划器更偏好某些动作（对视、少伸手），属于选动作，不属于 $P(s_{t+1}\mid s_t,a_t)$ |
| 校准 | 概率条上写 0.7，大约十次里有七次发生；本课用 Brier 分数量 |

## 2. 问题

第 30 课的世界模型听的是你的动作。人不是你的动作。把人硬塞进 $a_t$，实验上会立刻穿帮：动作对换要求你能真的执行被换进去的那个动作，你换不了别人的眼球。本课要分清三件常被混在一起的事。

1. 估计现在，和预测下一秒，不是同一个任务。Gaze360、ETH-XGaze、Chong 的注视目标检测，做的是当前帧或当前片段里人在看哪。桌宠要的是：他现在低头，下一秒会不会抬起来。后者才是外生过程的动力学。V-JEPA 2 在 Epic-Kitchens-100 上做的动作预判（论文报告 ViT-g 档 recall-at-5 为 39.7）是同一类测量：看过去，报未来，不是给当前帧贴标签。
2. 短时记忆不是人格。RSSM 的隐状态 $h_t$（第 05 课）已经是短时记忆：它条件化下一步预测。环形缓冲是同一职责的最小实现。Generative Agents（Park 等，arXiv:2304.03442）把经历写成自然语言记忆流，再检索、反思、计划，评的是「像不像人」。像人说话不等于下一秒注视预报得准。本课把那篇论文当反例读，不把它的小镇模拟当成实验。
3. 整体准确率会被惯性骗。若测试集里 85% 的帧「1 秒后标签等于现在」，惯性基线就是 85 分。模型拿到 87 分可以只是稍微平滑了一下。必须单列转折帧上的准确率和概率校准。直播概率条是给这两项看的：模型先报概率，1 秒后揭晓，你自己数它慌不慌。

数据纪律写在问题里，免得后面想抄近路。标签必须来自真实摄像头或第 29、30 课已经采过的桌面视频，用 MediaPipe 这类公开感知库离线抽出。禁止用聊天模型扮演用户生成假互动再证明自己准。禁止把社会模拟、角色对话当主实验。

## 3. 准备

- 第 01 到 05 课的循环和第 15 课 V-JEPA 2 的探针测法要熟：冻结大模型、只训小头、和随机或惯性基线对表。第 15 课的 Diving48 探针你不必重跑，记住「先定基线再谈增益」即可。
- 第 29 课若已有至少 1 分钟状态日志（杯子、人脸、是否看镜头），第 30 课若已有带动作的桌面轨迹，本课直接复用，再按第 7 节补采挥手、不理人、出声各若干段。两课都还没做也没关系：本课的采集脚本从摄像头从头写日志，不依赖尚未落盘的课文件。
- 硬件：笔记本或手机摄像头对准你坐的位置，桌上放一只杯子，光线稳定。S 档足够。Reachy Mini 或云台若已装好，用真机头部朝向当「自己的身体」状态即可，预测头的输入输出不变。
- Python 3.10 或更新，独立虚拟环境。必装：`mediapipe`、`opencv-python`、`numpy`、`torch`（CPU 版即可）。选做 Gaze-LLE 再加 GPU 版 PyTorch。
- 磁盘：官方 Face Landmarker 任务包约数 MB，Hand Landmarker、EfficientDet-Lite0 各数 MB；日志按 10 Hz 写 jsonl，10 分钟大约几十 MB。
- 摄像头权限：macOS 第一次打开摄像头会弹窗，终端里跑 Python 也要授权给那个终端。
- 不要准备聊天模型来「模拟用户」。本课没有这条路。

## 4. 学习目标

1. 在纸上把桌面状态拆成三块：物体、自己的身体、人；标出哪一块你能下命令，哪一块只能预报。
2. 说清注视估计和注视预测的差别，并指出 Gaze360 / ETH-XGaze 和本课 1 秒二元头各属于哪一类。
3. 用自己的日志算出惯性基线的整体准确率、转折帧准确率和 Brier 分数；没有这三项，禁止宣布「模型赢了」。
4. 实现环形短时记忆（最近 $N$ 步加未完成意图），训一个小头，与「只用当前帧」对照；增益或无增益都按同一口径写。
5. 把风格先验从世界模型里拿出来：能举一个「先验会改规划、不该改 $P(s_{t+1}\mid s_t,a_t)$」的例子。
6. 对着摄像头跑通概率条：模型先报「你下一秒会不会看我」，1 秒后揭晓；能口头解释 Generative Agents 的记忆流为什么不能替代这一步。

## 5. 原理

### 5.1 人是外生过程，不是可控动作

第 01 课给世界模型的最小定义是 $P(s_{t+1} \mid s_t, a_t)$。$a_t$ 必须是你真能执行的干预：转向、油门，后来是转头、伸手。对面的人会动，但他的动不是这条通道里的动作。把人的状态单列出来写，更清楚：

$$
s_t = \bigl(s_t^{\mathrm{obj}},\, s_t^{\mathrm{self}},\, s_t^{\mathrm{human}}\bigr)
$$

物体怎么响应你的手，走第 30 课的动作条件预测。人怎么走，走条件分布 $P(s_{t+1}^{\mathrm{human}} \mid s_t, a_t)$。$a_t$ 仍然可以影响人：你突然转头、出声，人更可能看过来。那是人对外界的反应，不是你在对人做动作对换。动作对换实验（第 03 课）的合法对象是你控制得了的通道。对人，合法的检验是预报：同一段真实历史，模型给出 1 秒后的概率，真实的人给出答案。

类比：气象是外生的。飞行计划会避开风暴，但飞行员不能把「明天转晴」写成油门。类比失效处：天气几乎不听飞机的；坐在对面的人会听你。所以 $a_t$ 作为人预测的条件是有用的，只是它不是人的动作本身。本课的小头先不把桌宠动作当输入（摄像头档常常没有稳定的 $a_t$ 日志），只用人自己的短历史来预报人。有第 30 课动作日志的人，第 11 节改造 2 再把 $a_t$ 拼进去。

### 5.2 先估计现在，再预测下一秒

摄像头给的是像素。要得到 $s_t^{\mathrm{human}}$，需要一个当前帧读出器。本课用 MediaPipe Face Landmarker：478 个 3D 面部关键点、52 个 blendshape（含 `eyeLookDownLeft` 这类眼部动作系数）、以及把标准脸变换到检测脸上的 4×4 矩阵。矩阵的旋转部分给出头部朝向；眼部 blendshape 给出眼球相对眼眶的偏移。两者合起来，才能近似「是不是在看镜头」：头正对相机但眼睛大幅下看，多半在看手机；头转开但眼睛还瞄着镜头，仍可能在看你。

这是估计，不是预测。估计的标准工作可以对照三篇已经核对过的论文。Gaze360（Kellnhofer 等，ICCV 2019，arXiv:1910.10088）在 238 人、室内外、接近 360° 朝向上做 3D 注视估计，模型吃多帧并输出不确定度（pinball 分位数损失）。ETH-XGaze（Zhang 等，ECCV 2020，arXiv:2007.15837）用 18 台单反、110 名被试，超过一百万张高分辨率图，专攻极端头姿。Chong 等（CVPR 2020，arXiv:2003.02501）输出场景里的注视目标热图，并单独判断目标是否在画面内；仓库 `model.py` 里 `ModelSpatial` 是单帧版，`ModelSpatioTemporal` 用 ConvLSTM 吃时间。

桌宠真正要的是下一秒。注视预判的公开工作里，Lai 等（ECCV 2024，arXiv:2305.03907）在 Ego4D 和 Aria 上从过去的第一人称视频和音频预报尚未出现的注视落点，音频分别带来 +2.5% 和 +2.4% 的 F1。那是第一人称、连续热图、带声音。本课缩成第三人称桌面、1 秒、两个二元事件，为的是单卡甚至 CPU 当天能训完，并且指标能和惯性基线直接减。

标签函数必须固定下来，训练和直播用同一套，否则你在拟合自己的阈值漂移。本课采用可复现的规则，不假装它等于眼动仪：

- 看镜头：面部 4×4 矩阵解出的偏航、俯仰绝对值都小于 20°，且左右 `eyeLookDown` 平均小于 0.35，且当前帧检测到脸。
- 手靠近杯子：Hand Landmarker 给出手关键点，Object Detector（COCO 的 `cup` 或 `bottle`）给出杯框；手部任意点到杯框中心的归一化距离小于 0.18。没检测到杯子则该标签记为空，训练时不算损失。

规则里的角度和距离是起点，第 7 节会让你抽 20 帧肉眼核对。阈值可以改，但必须改完重算基线，不许只重训模型。

### 5.3 惯性基线为什么必须先过

记 $y_t \in \{0,1\}$ 为某一二元标签，$\Delta$ 为预报地平线（本课 $\Delta = 1$ 秒，10 Hz 日志就是 10 步）。惯性基线是：

$$
\hat y_{t+\Delta}^{\mathrm{persist}} = y_t
$$

它没有参数。整体准确率等于测试集里「标签在 $\Delta$ 内不变」的比例。人看手机、看屏幕、发呆，这个比例通常很高。所以整体准确率不是本课的主指标。主指标有三项：

$$
\mathrm{Acc}_{\mathrm{all}},\quad
\mathrm{Acc}_{\mathrm{flip}}
=\mathrm{Acc}\bigl(\{t: y_{t+\Delta}\neq y_t\}\bigr),\quad
\mathrm{Brier}=\frac{1}{n}\sum_i (\hat p_i - y_i)^2
$$

惯性基线的 $\mathrm{Acc}_{\mathrm{flip}}$ 恒为 0，Brier 等于转折帧比例（它在不变帧上贡献 0，在转折帧上贡献 1）。小头输出 $\hat p = \sigma(z)$，用二元交叉熵训练：

$$
\mathcal{L} = -\sum_{k\in\mathcal{K}_t}
\Bigl[y^{(k)}\log \sigma(z^{(k)}) + \bigl(1-y^{(k)}\bigr)\log\bigl(1-\sigma(z^{(k)})\bigr)\Bigr]
$$

$\mathcal{K}_t$ 是该帧有标签的任务集合（杯子缺失时不含第二项）。赢的定义写死：测试集上 $\mathrm{Acc}_{\mathrm{flip}}$ 高于惯性（惯性为 0，所以任何能抓到转折的模型都在这项上加分），并且 Brier 低于惯性。只有 $\mathrm{Acc}_{\mathrm{all}}$ 升高、Brier 不降，算没赢。

Brier 管的是概率条上的数字可不可信。一个永远输出 0.85 的头，在「看镜头占 85%」的测试集上整体准确率可以很高，直播时你低头的那一秒它仍画着长条，桌宠就会按「他还会看我」去打扰。第 7 节的概率条先报后揭晓，就是把 Brier 变成你能数的次数：报 0.7 的那些回合里，大约十次应有七次揭晓为看。校准坏掉时，优先查标签泄漏（用了 $t$ 当 $t+\Delta$）和转折样本是不是太少。

### 5.4 环形短时记忆只条件化预测

RSSM 用 GRU 把历史压进 $h_t$，再预测 $z_{t+1}$。本课没有新发明架构，把同一职责收成环形缓冲：长度为 $N$ 的数组，下标 $t \bmod N$ 写入最新状态，读出时按时间排成 $s_{t-N+1},\ldots,s_t$。每个 $s$ 是一个短向量：看镜头、手头距离、偏航、俯仰、眼部下看系数、脸是否在画面、手是否在画面、杯是否在画面，以及你自己的动作（有就写，没有就填 0）。

未完成意图 $i_t$ 是几个滞回开关，不是愿望清单。手举过肩线持续 3 帧则 `hand_up` 置 1，连续 5 帧低于肩线则清 0；俯仰向下超过 20° 持续 3 帧则 `head_down` 置 1，抬回再清；手头距离连续下降 5 帧则 `approaching_cup` 置 1，距离转升或杯子消失则清。滞回是为了不在阈值附近抖动。这些位拼进 $m_t$：

$$
m_t = \bigl(s_{t-N+1},\ldots,s_t,\, i_t\bigr)
$$

预测头学 $P(y_{t+\Delta} \mid m_t)$。对照实验只有一处不同：当前帧头只吃 $s_t$（可加 $i_t$），环形头吃整段 $m_t$。网络都是小 MLP 或一层 GRU 加线性层，参数量控制在 $10^4$ 量级，CPU 上几分钟能训完。

环形记忆可能没有增益。若 1 秒后的标签几乎由当前朝向决定，多看 0.8 秒历史只是重复。无增益时写清楚：在你的桌子、你设定的 $\Delta$ 和 $N$ 上，多步历史没有降低 Brier。这是合法结论。禁止靠加人格词向量、加聊天摘要把数字做上去。

### 5.5 风格先验属于规划器

规划器打分时可以加一项「更想对视」：

$$
\mathrm{score}(a) = r_{\mathrm{task}}(\hat s_{t+1:t+H}) + \lambda_{\mathrm{gaze}} \cdot \mathbf{1}[a\text{ 是看人}]
$$

$\lambda_{\mathrm{gaze}}$ 是风格。它改的是在几条想象轨迹里选哪一条，不改想象轨迹本身。世界模型如果被灌进「我是粘人桌宠」，会在人低头的轨迹上仍给出高看镜头概率，规划器再根据这个假概率决定去打扰。第 32 课「人低头则降低打扰」依赖的恰恰是世界模型肯报「他下一秒还在看手机」。

人格词（粘人、高冷、内向）可以出现在规划器的代价或约束里，甚至出现在对用户说的话里。它们不出现在 $P(s_{t+1}\mid s_t,a_t)$ 的参数里。这一段就是验收里要求的那句：人格不是动力学。

### 5.6 反例：像人说话不等于预报对

Generative Agents 的架构是记忆流、检索、反思、计划。记忆对象是自然语言句子，带创建时间和最近访问时间；检索按近因、重要度、相关性打分；重要度累积过阈值就写一条更高层的反思。论文的实验是 25 个智能体在沙盒小镇里过几天，用人来评「像不像」。这套东西解决的是可信的社会表演，不是 $P(y_{t+\Delta}\mid m_t)$ 的校准。

本课只要求你读懂它的记忆流和本课环形缓冲的差别：前者为了生成接下来要说的话，后者为了压低 1 秒后两个事件的 Brier。把记忆流接到桌宠上当世界模型，会得到会聊天、不会躲杯子的装置。第 12 节给了阅读问题，仓库可以克隆来翻目录，不作为本课训练数据来源。

### 5.7 和 V-JEPA 2 动作预判同一把尺

第 15 课用 Diving48 探针测「当前片段是哪类跳水」，那是现在时分类。V-JEPA 2 论文另外报了 Epic-Kitchens-100 动作预判：看过去的厨房视频，预报接下来的动词和名词，ViT-g 档 recall-at-5 为 39.7，相对此前任务专用模型约 44% 的相对提升（论文口径）。测的是未来。本课的小头是同一把尺的桌面缩微版：冻结或根本不用大视频骨干，只在自己抽出的低维状态上预报 1 秒。差别是规模和输出形状，不是问题类型。不要把「模型能描述桌上很乱」当成预判成功。

## 6. 源码导读

本课有两处真实代码要读：官方感知库怎么把像素变成朝向，以及注视目标文献里时间是怎么进网络的。桌宠小头是课内胶水，第 7 节现场写。不要去克隆 MediaPipe 那个巨型 C++ 单仓，Python 任务 API 随 `pip install mediapipe` 进来。

| 位置 | 零件 | 带着什么问题读 |
|---|---|---|
| MediaPipe Python 指南中的 `FaceLandmarkerOptions` | 当前帧读出 | `output_face_blendshapes` 和 `output_facial_transformation_matrixes` 默认都是关的，不打开你拿不到朝向和眼部下看 |
| 同上，`detect_for_video(image, timestamp_ms)` | 视频模式 | 为什么必须用 VIDEO 而不是逐帧 IMAGE？时间戳不单调会怎样？ |
| `ejcgt/attention-target-detection/model.py` 的 `ModelSpatial` | 单帧注视目标 | 场景通路输入为什么是 4 通道？第四通道的 `head` 是什么？ |
| 同一文件的 `ModelSpatioTemporal` | 多帧注视目标 | ConvLSTM 接在哪一层之后？它预报的是未来，还是把当前估计做稳？ |
| 该仓库 `demo.py`、`eval_on_videoatttarget.py` | 数据流 | demo 吃的是仓库自带的 `data/demo`，评测吃的是 VideoAttentionTarget |
| 选做：`fkryan/gazelle` 的 `gazelle/model.py` 与 `hubconf.py` | 2025 年的注视目标 | 骨干冻结、只训解码器，和本课「小头」是同一分工，只是它报的是当前热图不是 1 秒后二元事件 |

Face Landmarker 的构造按官方 Python 指南，VIDEO 模式如下（`model_path` 指向你下载的 `face_landmarker.task`）：

```python
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1,
)
```

指南写明：IMAGE 管单张图，VIDEO 管已解码视频帧，LIVE_STREAM 管摄像头异步回调。VIDEO 和 LIVE_STREAM 都要传该帧的时间戳。官方结果示例里，`face_landmarks` 每个脸 478 点，`face_blendshapes` 52 项（示例打印了 `browDownLeft` 等名字），`facial_transformation_matrixes` 是 4×4。Hand Landmarker、Object Detector 的任务对象在同一个 `mp.tasks.vision` 下面，模式同样分 IMAGE / VIDEO / LIVE_STREAM。完整可跑笔记本在 [mediapipe-samples 的 Face Landmarker Python 示例](https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/face_landmarker/python)。

Chong 仓库的 `ModelSpatial.forward(self, images, head, face)` 把三件事拼起来：人脸裁剪走 `conv1_face` 那条 ResNet 式通路；场景 RGB 和头部掩膜在通道维拼成 4 通道，走 `conv1_scene`；头部掩膜和人脸特征再算出一组 7×7 注意力，乘到场景特征上。出口有两路：反卷积得到注视热图，`fc_inout` 得到「目标是否在画面内」。这是当前帧（加头部框）的估计器。`ModelSpatioTemporal` 把若干层换成 `BottleneckConvLSTM`，用 `lib/pytorch_convolutional_rnn` 里的卷积 LSTM 在时间上累积。读的时候盯住一件事：ConvLSTM 的隐状态是为了让当前注视估计更稳、更能处理目标出画，并没有一个「预报 1 秒后热图」的损失。本课的 $\Delta=1$ 秒头，是在他们已经估计好的当前量上面再叠一层预测。仓库 README 写明代码按 Python 3.5、PyTorch 0.4 验证，`environment.yml` 也停在那一代。本课以读结构为主；`python demo.py` 能跑是惊喜，跑不起按第 10 节处理，不要为了 demo 降级你的全局 PyTorch。

Gaze-LLE（Ryan 等，CVPR 2025 Highlight，arXiv:2412.09586）把这件事做成「冻结 DINOv2，只训注视解码器」，PyTorch Hub 入口在 `fkryan/gazelle`。它输出的是当前注视热图和 in/out 分数，仍然是估计。选做体验可以加载 `gazelle_dinov2_vitb14`，在你的一张桌面照片上画热图：热图落在镜头方向，和本课「看镜头」标签应当同向。它替代不了 1 秒预报头。

## 7. 实验

工作目录建议单独建。下面默认你在 `~/learn-wm-l31`。每一步先写预期，再跑，再对照。数据必须是你或同伴在镜头前的真实运动。

### Step 1: 环境与模型文件

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install mediapipe opencv-python numpy torch
```

```bash
mkdir -p models data ckpt
```

Face Landmarker 官方 float16 任务包（指南「Model」一节指向 Google 托管的任务文件，直链如下）：

```bash
curl -L -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

```bash
curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

```bash
curl -L -o models/efficientdet_lite0.tflite https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite
```

预期：三个文件都非空，`face_landmarker.task` 大约 3 到 4 MB 量级。若 curl 被环境拦住，改用指南页面上的模型下载按钮，手动放到 `models/`。

对照阅读用的注视目标仓库（只读结构，不作为训练数据）：

```bash
git clone https://github.com/ejcgt/attention-target-detection.git
```

打开 `attention-target-detection/model.py`，找到 `class ModelSpatial` 和 `class ModelSpatioTemporal`，对照第 6 节的问题把答案写进 `NOTES.md`。`sh download_models.sh` 和 `python demo.py` 是选做，依赖老 PyTorch，失败不算本课失败。

### Step 2: 把像素变成状态日志

把下面脚本存成 `collect_desk.py`。它按 10 Hz 写 jsonl：当前是否看镜头、手头距离、朝向、blendshape、三个意图位。`--clip` 用来标记采集条件，后面按 clip 切训练测试，禁止按帧随机切。

```python
import argparse, json, math, time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions


def yaw_pitch(mat4):
    r = np.array(mat4).reshape(4, 4)[:3, :3]
    pitch = math.atan2(-r[2, 1], math.sqrt(r[2, 0] ** 2 + r[2, 2] ** 2))
    yaw = math.atan2(r[2, 0], r[2, 2])
    return math.degrees(yaw), math.degrees(pitch)


def blend_map(face_blendshapes):
    if not face_blendshapes:
        return {}
    return {c.category_name: float(c.score) for c in face_blendshapes[0]}


def box_center(det):
    box = det.bounding_box
    return box.origin_x + box.width / 2.0, box.origin_y + box.height / 2.0


class Hysteresis:
    def __init__(self, on_n, off_n):
        self.on_n, self.off_n = on_n, off_n
        self.state = 0
        self.cnt = 0

    def update(self, cond):
        if self.state == 0:
            self.cnt = self.cnt + 1 if cond else 0
            if self.cnt >= self.on_n:
                self.state, self.cnt = 1, 0
        else:
            self.cnt = self.cnt + 1 if (not cond) else 0
            if self.cnt >= self.off_n:
                self.state, self.cnt = 0, 0
        return self.state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip", required=True)
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--out", default="data")
    args = p.parse_args()

    face_opt = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/face_landmarker.task"),
        running_mode=VisionRunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    hand_opt = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
    )
    det_opt = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path="models/efficientdet_lite0.tflite"),
        running_mode=VisionRunningMode.VIDEO,
        score_threshold=0.3,
        max_results=5,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("cannot open camera")

    out_path = Path(args.out) / f"{args.clip}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hz = 10
    interval = 1.0 / hz
    hand_up = Hysteresis(3, 5)
    head_down = Hysteresis(3, 5)
    approaching = Hysteresis(5, 5)
    dist_hist = deque(maxlen=6)

    with FaceLandmarker.create_from_options(face_opt) as face, \
            HandLandmarker.create_from_options(hand_opt) as hand, \
            ObjectDetector.create_from_options(det_opt) as det, \
            out_path.open("w") as f:
        t0 = time.time()
        frame_i = 0
        last = t0
        while time.time() - t0 < args.seconds:
            ok, bgr = cap.read()
            if not ok:
                break
            now = time.time()
            if now - last < interval:
                continue
            last = now
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            ts = int((now - t0) * 1000)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            h, w, _ = bgr.shape

            fr = face.detect_for_video(mp_img, ts)
            hr = hand.detect_for_video(mp_img, ts)
            od = det.detect_for_video(mp_img, ts)

            yaw = pitch = down = 0.0
            face_on = 1 if fr.face_landmarks else 0
            look = 0
            if fr.facial_transformation_matrixes:
                yaw, pitch = yaw_pitch(fr.facial_transformation_matrixes[0])
                bm = blend_map(fr.face_blendshapes)
                down = 0.5 * (
                    bm.get("eyeLookDownLeft", 0.0) + bm.get("eyeLookDownRight", 0.0)
                )
                look = int(
                    face_on
                    and abs(yaw) < 20
                    and abs(pitch) < 20
                    and down < 0.35
                )

            hands = []
            for lmset in hr.hand_landmarks:
                for lm in lmset:
                    hands.append((lm.x, lm.y))
            cup = None
            for d in od.detections:
                name = d.categories[0].category_name
                if name in ("cup", "bottle"):
                    cx, cy = box_center(d)
                    cup = (cx / w, cy / h)
                    break
            dist = None
            if cup and hands:
                dist = min(
                    math.hypot(hx - cup[0], hy - cup[1]) for hx, hy in hands
                )
            hand_near = int(dist is not None and dist < 0.18)

            dist_hist.append(dist if dist is not None else 9.0)
            shrinking = (
                len(dist_hist) == dist_hist.maxlen
                and dist is not None
                and dist_hist[0] - dist_hist[-1] > 0.02
            )
            rec = {
                "t": round(now - t0, 3),
                "clip": args.clip,
                "face_on": face_on,
                "look": look,
                "yaw": round(yaw, 2),
                "pitch": round(pitch, 2),
                "eye_down": round(down, 3),
                "hand_on": int(bool(hands)),
                "cup_on": int(cup is not None),
                "hand_cup_dist": None if dist is None else round(dist, 3),
                "hand_near": None if cup is None else hand_near,
                "intent_hand_up": hand_up.update(
                    bool(hands) and min(hy for _, hy in hands) < 0.45
                ),
                "intent_head_down": head_down.update(pitch > 20),
                "intent_approach": approaching.update(shrinking),
            }
            f.write(json.dumps(rec) + "\n")
            frame_i += 1
            tag = "LOOK" if look else "away"
            cv2.putText(
                bgr,
                f"{args.clip} {tag} yaw={yaw:.0f} dist={dist}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("collect_desk", bgr)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"wrote {frame_i} frames to {out_path}")


if __name__ == "__main__":
    main()
```

先冒烟 10 秒，确认窗口里标签会随你转头翻转：

```bash
python collect_desk.py --clip smoke --seconds 10
```

预期：`data/smoke.jsonl` 大约 80 到 110 行（摄像头启动会丢几帧），`face_on` 在你入画后为 1。打开前 5 行，确认字段齐全。`look` 若完全不翻转，先自己转头、低头看手机再看镜头，不要先调模型。

### Step 3: 按协议补采三段互动

第 29、30 课若已有 jsonl 且含 `look` 一类字段，可跳过重复部分，仍要补下面三类。每类按口令做，口令写进 `--clip` 名，方便按 clip 切分。坐在镜头正前方，杯子放在画面下半、你右手够得到的地方。

对视与不理人（交替，逼出转折）：看镜头约 4 秒，低头看手机约 4 秒，重复到满 90 秒。

```bash
python collect_desk.py --clip look_phone --seconds 90
```

挥手：举手挥 4 秒，手放回桌面 4 秒，重复到满 60 秒。

```bash
python collect_desk.py --clip wave --seconds 60
```

靠近杯子：手伸向杯 3 秒，缩回 3 秒，重复到满 60 秒。杯子要用检测器认得出的那只，纸杯、马克杯、水瓶都可以，开采前在窗口里确认 `dist=` 不是 `None`。

```bash
python collect_desk.py --clip reach_cup --seconds 60
```

出声（可选但建议有）：对着镜头说话或拍手 30 秒。当前小头不吃音频，这段用来检查「出声时人是否更常看镜头」的描述统计，给第 11 节改造 3 留种子。

```bash
python collect_desk.py --clip voice --seconds 30
```

采完立刻做两件事。第一，抽每个 clip 各 10 帧，对照窗口截图或当时记忆，看 `look` 和「我当时在不在看镜头」是否同向；错一半以上就改 5.2 的角度或 `eye_down` 阈值，删掉旧 jsonl 重采。第二，统计每个 clip 的 `look` 均值，若某个 clip 全是 0 或全是 1，这段对转折毫无贡献，重采。

禁止的做法：用生成模型写一段「用户正在挥手」的伪 jsonl；用一段电影里的对视镜头冒充你的桌子（域差会让阈值崩溃，报告也无法接到第 32 课）。

### Step 4: 惯性基线先报出来

把下面存成 `train_head.py`。它同时算惯性、当前帧 MLP、环形 GRU，三个头共用同一套按 clip 切分的样本。`--horizon` 默认 10，对应 10 Hz 下的 1 秒。

```python
import argparse, json, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


FEATS = [
    "look", "yaw", "pitch", "eye_down", "face_on", "hand_on", "cup_on",
    "hand_cup_dist", "hand_near", "intent_hand_up", "intent_head_down",
    "intent_approach",
]


def read_jsonl(path):
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def vec(row):
    x = []
    for k in FEATS:
        v = row.get(k)
        if v is None:
            v = 0.0
        if k in ("yaw", "pitch"):
            v = float(v) / 90.0
        x.append(float(v))
    return np.array(x, dtype=np.float32)


def make_samples(clips, horizon, n_hist):
    samples = []
    for rows in clips:
        for i in range(n_hist, len(rows) - horizon):
            y_look = rows[i + horizon]["look"]
            y_hand = rows[i + horizon]["hand_near"]
            hist = np.stack([vec(rows[j]) for j in range(i - n_hist + 1, i + 1)])
            persist_look = rows[i]["look"]
            persist_hand = rows[i]["hand_near"]
            samples.append(
                {
                    "hist": hist,
                    "y_look": float(y_look),
                    "y_hand": None if y_hand is None else float(y_hand),
                    "p_look": float(persist_look),
                    "p_hand": None if persist_hand is None else float(persist_hand),
                    "clip": rows[i]["clip"],
                }
            )
    return samples


def split_by_clip(all_rows, seed=0):
    names = sorted({r["clip"] for rows in all_rows for r in rows})
    rng = random.Random(seed)
    rng.shuffle(names)
    n_test = max(1, int(round(0.25 * len(names))))
    test_names = set(names[:n_test])
    train, test = [], []
    for rows in all_rows:
        (test if rows[0]["clip"] in test_names else train).append(rows)
    return train, test, sorted(test_names)


class CurrentMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 2)
        )

    def forward(self, hist):
        return self.net(hist[:, -1, :])


class RingGRU(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gru = nn.GRU(d, 32, batch_first=True)
        self.fc = nn.Linear(32, 2)

    def forward(self, hist):
        _, h = self.gru(hist)
        return self.fc(h[-1])


def batches(samples, bs, rng):
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    for s in range(0, len(idx), bs):
        chunk = [samples[i] for i in idx[s : s + bs]]
        hist = torch.tensor(np.stack([c["hist"] for c in chunk]))
        y = torch.tensor([[c["y_look"], c["y_hand"] if c["y_hand"] is not None else 0.0] for c in chunk])
        mask = torch.tensor([[1.0, 0.0 if c["y_hand"] is None else 1.0] for c in chunk])
        yield hist, y, mask, chunk


def bce_masked(logit, y, mask):
    loss = nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


@torch.no_grad()
def eval_pack(model, samples, persist=False):
    stats = {}
    for key, yk, pk in (
        ("look", "y_look", "p_look"),
        ("hand", "y_hand", "p_hand"),
    ):
        ys, ps, flips = [], [], []
        for s in samples:
            y = s[yk]
            if y is None:
                continue
            if persist:
                p = s[pk]
                if p is None:
                    continue
            else:
                hist = torch.tensor(s["hist"][None])
                logit = model(hist)[0, 0 if key == "look" else 1]
                p = float(torch.sigmoid(logit))
            ys.append(y)
            ps.append(p)
            flips.append(abs(y - (s[pk] if s[pk] is not None else y)) > 0.5)
        ys, ps = np.array(ys), np.array(ps)
        pred = (ps >= 0.5).astype(np.float32)
        acc = float((pred == ys).mean()) if len(ys) else float("nan")
        flip_idx = np.array(flips, dtype=bool)
        acc_flip = (
            float((pred[flip_idx] == ys[flip_idx]).mean())
            if flip_idx.any()
            else float("nan")
        )
        brier = float(((ps - ys) ** 2).mean()) if len(ys) else float("nan")
        stats[key] = {
            "n": int(len(ys)),
            "n_flip": int(flip_idx.sum()) if len(ys) else 0,
            "acc": acc,
            "acc_flip": acc_flip,
            "brier": brier,
        }
    return stats


def train_one(model, samples, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(0)
    model.train()
    for _ in range(epochs):
        for hist, y, mask, _ in batches(samples, 64, rng):
            opt.zero_grad()
            loss = bce_masked(model(hist), y, mask)
            loss.backward()
            opt.step()
    model.eval()
    return model


def fmt(stats):
    lines = []
    for k, v in stats.items():
        lines.append(
            f"{k}: n={v['n']} flip={v['n_flip']} "
            f"acc={v['acc']:.3f} acc_flip={v['acc_flip']:.3f} brier={v['brier']:.3f}"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--n_hist", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out", default="ckpt/head.pt")
    args = ap.parse_args()

    files = sorted(Path(args.data).glob("*.jsonl"))
    files = [p for p in files if p.stem != "smoke"]
    if len(files) < 2:
        raise SystemExit("need at least 2 clips in data/*.jsonl")
    all_rows = [read_jsonl(p) for p in files]
    train_clips, test_clips, test_names = split_by_clip(all_rows)
    print("test clips:", test_names)
    train = make_samples(train_clips, args.horizon, args.n_hist)
    test = make_samples(test_clips, args.horizon, args.n_hist)
    print(f"train {len(train)} test {len(test)}")

    persist = eval_pack(None, test, persist=True)
    print("PERSIST")
    print(fmt(persist))

    d = train[0]["hist"].shape[-1]
    mlp = train_one(CurrentMLP(d), train, args.epochs, 1e-3)
    gru = train_one(RingGRU(d), train, args.epochs, 1e-3)
    print("CURRENT_MLP")
    print(fmt(eval_pack(mlp, test)))
    print("RING_GRU")
    print(fmt(eval_pack(gru, test)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "gru": gru.state_dict(),
            "mlp": mlp.state_dict(),
            "d": d,
            "n_hist": args.n_hist,
            "horizon": args.horizon,
        },
        args.out,
    )
    print("saved", args.out)


if __name__ == "__main__":
    main()
```

```bash
python train_head.py
```

预期输出里有三块：`PERSIST`、`CURRENT_MLP`、`RING_GRU`，每块都有 `look` 和 `hand` 的 `n`、`flip`、`acc`、`acc_flip`、`brier`。先看 `n_flip`：看镜头若低于 20，说明测试 clip 里几乎没有抬头低头，结论不可信，回去补采。若 `test clips` 打印出来只有 `voice`，把 `look_phone` 再采一段换名字（例如 `look_phone_b`），让切分至少分到一段有交替的对视。惯性的 `acc_flip` 应是 0.000。MLP 和 GRU 的 `acc_flip` 只要稳定高于 0，就说明它们在转折上不是瞎猜；再比 Brier，谁低谁校准得好。

环形记忆的增益定义：GRU 的 look-Brier 比 MLP 低，且不是随机抖动（种子 0 再跑一次应同向）。若 Brier 持平或更差，写「在 N=8、Δ=1s、我的桌子上，环形历史无增益」。这是合格答案。常见无增益原因：你的运动在 1 秒内近似一阶，当前 yaw/pitch 已经够用；或者意图位和当前特征高度共线。

### Step 5: 直播概率条，先报后揭晓

把下面存成 `live_probe.py`。它加载 GRU 头，画面上方画一条「下一秒看我」的概率条；1 秒后在旁边写出当时揭晓的真标签。挥手、低头看手机，看概率条是抢先动还是事后才动。

```python
import math
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import torch

from collect_desk import (
    BaseOptions,
    FaceLandmarker,
    FaceLandmarkerOptions,
    HandLandmarker,
    HandLandmarkerOptions,
    Hysteresis,
    ObjectDetector,
    ObjectDetectorOptions,
    VisionRunningMode,
    blend_map,
    box_center,
    yaw_pitch,
)
from train_head import FEATS, RingGRU


def feat_row(look, yaw, pitch, down, face_on, hand_on, cup_on, dist, hand_near, iu, idn, ia):
    vals = {
        "look": look,
        "yaw": yaw / 90.0,
        "pitch": pitch / 90.0,
        "eye_down": down,
        "face_on": face_on,
        "hand_on": hand_on,
        "cup_on": cup_on,
        "hand_cup_dist": 0.0 if dist is None else dist,
        "hand_near": 0.0 if hand_near is None else hand_near,
        "intent_hand_up": iu,
        "intent_head_down": idn,
        "intent_approach": ia,
    }
    return np.array([vals[k] for k in FEATS], dtype=np.float32)


def main():
    ckpt = torch.load("ckpt/head.pt", map_location="cpu", weights_only=True)
    model = RingGRU(ckpt["d"])
    model.load_state_dict(ckpt["gru"])
    model.eval()
    n_hist = ckpt["n_hist"]
    horizon = ckpt["horizon"]

    face_opt = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/face_landmarker.task"),
        running_mode=VisionRunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    hand_opt = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
    )
    det_opt = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path="models/efficientdet_lite0.tflite"),
        running_mode=VisionRunningMode.VIDEO,
        score_threshold=0.3,
        max_results=5,
    )

    cap = cv2.VideoCapture(0)
    hist = deque(maxlen=n_hist)
    pending = deque()
    hand_up = Hysteresis(3, 5)
    head_down = Hysteresis(3, 5)
    approaching = Hysteresis(5, 5)
    dist_hist = deque(maxlen=6)
    t0 = time.time()
    last = t0
    hits = []

    with FaceLandmarker.create_from_options(face_opt) as face, \
            HandLandmarker.create_from_options(hand_opt) as hand, \
            ObjectDetector.create_from_options(det_opt) as det:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            now = time.time()
            if now - last < 0.1:
                continue
            last = now
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            ts = int((now - t0) * 1000)
            h, w, _ = bgr.shape
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            fr = face.detect_for_video(mp_img, ts)
            hr = hand.detect_for_video(mp_img, ts)
            od = det.detect_for_video(mp_img, ts)

            yaw = pitch = down = 0.0
            face_on = 1 if fr.face_landmarks else 0
            look = 0
            if fr.facial_transformation_matrixes:
                yaw, pitch = yaw_pitch(fr.facial_transformation_matrixes[0])
                bm = blend_map(fr.face_blendshapes)
                down = 0.5 * (
                    bm.get("eyeLookDownLeft", 0.0) + bm.get("eyeLookDownRight", 0.0)
                )
                look = int(face_on and abs(yaw) < 20 and abs(pitch) < 20 and down < 0.35)

            hands = [(lm.x, lm.y) for lmset in hr.hand_landmarks for lm in lmset]
            cup = None
            for d in od.detections:
                if d.categories[0].category_name in ("cup", "bottle"):
                    cx, cy = box_center(d)
                    cup = (cx / w, cy / h)
                    break
            dist = None
            if cup and hands:
                dist = min(math.hypot(hx - cup[0], hy - cup[1]) for hx, hy in hands)
            hand_near = None if cup is None else int(dist is not None and dist < 0.18)
            dist_hist.append(dist if dist is not None else 9.0)
            shrinking = (
                len(dist_hist) == dist_hist.maxlen
                and dist is not None
                and dist_hist[0] - dist_hist[-1] > 0.02
            )
            iu = hand_up.update(bool(hands) and min(hy for _, hy in hands) < 0.45)
            idn = head_down.update(pitch > 20)
            ia = approaching.update(shrinking)
            row = feat_row(
                look, yaw, pitch, down, face_on, int(bool(hands)),
                int(cup is not None), dist, hand_near, iu, idn, ia,
            )
            hist.append(row)
            p = None
            if len(hist) == n_hist:
                with torch.no_grad():
                    logit = model(torch.tensor(np.stack(hist)[None]))[0, 0]
                    p = float(torch.sigmoid(logit))
                pending.append((now + horizon * 0.1, p, look))

            revealed = None
            while pending and pending[0][0] <= now:
                _, p_old, _ = pending.popleft()
                revealed = (p_old, look)
                hits.append(abs((p_old >= 0.5) - look))

            bar_w = int((p if p is not None else 0.0) * 400)
            cv2.rectangle(bgr, (20, 20), (420, 60), (40, 40, 40), -1)
            cv2.rectangle(bgr, (20, 20), (20 + bar_w, 60), (0, 180, 0), -1)
            msg = f"P(look in 1s)={0 if p is None else p:.2f}"
            cv2.putText(bgr, msg, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if revealed is not None:
                tag = "YES" if revealed[1] else "NO"
                cv2.putText(
                    bgr,
                    f"reveal: {tag} (p was {revealed[0]:.2f})",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
            if hits:
                acc = 1.0 - float(np.mean(hits[-50:]))
                cv2.putText(
                    bgr,
                    f"last50 acc={acc:.2f}",
                    (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
            cv2.imshow("live_probe", bgr)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

两个脚本要放在同一目录，直播才会找到 `collect_desk.py` 和 `train_head.py`。`torch.load` 若报不认识 `weights_only`，删掉该参数（旧版 PyTorch 没有它）。

```bash
python live_probe.py
```

协议：先不要动，看概率条停在高位还是低位；再突然低头看手机，观察条是在你低头之前就开始掉，还是等你已经低头才掉。后者说明模型只是在抄当前值，和惯性基线一个货色。挥手同理：条应该在手举起来的过程中升，不要等你已经看回镜头才动。做满大约 20 次「看我 / 看手机」交替，把 `last50 acc` 和你主观感觉写进 `NOTES.md`。ESC 退出。

### Step 6: 留证据

在 `~/learn-wm-l31/NOTES.md` 写这些，短句即可：

```text
日期、机器、摄像头、是否 Mac
mediapipe 与 torch 版本
各 clip 秒数与 look 均值、hand_near 非空比例
测试 clip 名单（按 clip 切分，不是按帧）
PERSIST / CURRENT_MLP / RING_GRU 的 acc、acc_flip、brier（look 与 hand 分列）
环形记忆：有增益 / 无增益，一句话原因
直播 20 次交替：概率条是否先于动作
人格不是动力学：我的规划器若更爱对视，改的是哪一项，没改哪一项
```

## 8. 配置与预算

| 档位 | 数据 | 训练 | 耗时 | 用途 |
|---|---|---|---|---|
| 必做（本课） | 对视/不理 90s，挥手 60s，伸杯 60s，可选出声 30s，10 Hz | MLP + GRU 各 15 epoch，CPU | 采集半小时（含核对标签），训练两三分钟，直播 15 分钟 | 惯性对照与记忆消融 |
| 加量 | 每类再采 3 分钟，换一次光线或衣服 | 同上，epoch 30 | 多一小时采集 | 看 Brier 是否随数据降 |
| 选做体验 | 一张自己的桌面照片 | 不训练，加载 Gaze-LLE Hub 权重 | 有网、有 8GB 以上内存即可，GPU 更快 | 看当前注视热图，不替代 1 秒头 |

超参默认：$\Delta=1$ 秒（`horizon=10`），$N=8$ 步历史，学习率 $10^{-3}$，batch 64，种子 0，按 clip 切 25% 测试。不要在测试 clip 上扫阈值。若要改「看镜头」的 20° 或 `eye_down<0.35`，改完必须重跑 Step 4 的三块数字。

Mac / CPU：全部必做可跑。MediaPipe 任务在 CPU 上 10 Hz 足够。Gaze-LLE 的 ViT-L 档会慢，选做改 ViT-B。

Reachy Mini 或云台档：把头部关节角写进 `s_t^{\mathrm{self}}` 的空位（现在填 0 的动作通道），第 11 节改造 2 再训。本课验收不要求真机。

## 9. 验收

- [ ] 能在纸上指出：人的朝向是状态，桌宠转头是动作，风格先验是规划器打分项。
- [ ] 至少三段真实 jsonl（对视/不理、挥手、伸杯），每段 `look` 均值不在 0 或 1 的死角；抽检过标签与肉眼同向。
- [ ] 测试按 clip 切分；报告里有惯性、当前帧、环形三套 `acc` / `acc_flip` / `brier`，look 与 hand 分列。
- [ ] `acc_flip`：惯性为 0；至少一个学习头明显高于 0。若两个学习头的 `acc_flip` 都接近 0.5 且 Brier 不优于惯性，写失败分析，不算没做完。
- [ ] 环形记忆：有增益或无增益，原因写到特征或数据，不写到「可能还要调超参」。
- [ ] 直播概率条先报后揭晓；20 次看我/看手机交替有记录。
- [ ] `NOTES.md` 里有一段「人格不是动力学」：举你自己的 $\lambda_{\mathrm{gaze}}$ 例子，说明它不进预测头的损失。
- [ ] 没有使用聊天模型生成互动数据。

## 10. 排错

| 症状 | 最可能的原因 | 怎么确认 | 怎么修 |
|---|---|---|---|
| `cannot open camera` | 权限或编号不对 | 换 `--camera 1`；系统设置里看终端是否有摄像头权限 | macOS 把权限给到你启动 Python 的那个应用 |
| `look` 永远为 0 | 矩阵符号和阈值不适合你的座位 | 打印 yaw/pitch/eye_down，转头时看哪个量在动 | 先放大角度阈值到 30°，确认会翻转，再收紧；座位正对镜头 |
| `look` 永远为 1 | 你几乎没低头，或 `eye_down` 从不升高 | jsonl 里 `pitch` 的最大最小值 | 按 Step 3 的 4 秒交替重采，真的看手机 |
| `hand_cup_dist` 总是空 | 检测器没认出杯子 | 窗口里 `dist=None`；换马克杯、水瓶 | 类别只认 `cup`/`bottle`；背景杂物太多就换纯色桌 |
| 训练 `n_flip` 接近 0 | 测试 clip 刚好是「一直盯着」的那段 | 打印 test clip 名和该 clip 的 look 均值 | 每个条件至少两段，切分才切不空 |
| MLP 整体 acc 高于惯性、Brier 更差 | 模型在转折上乱报高置信 | 看 `acc_flip` 和概率直方图 | 降学习率或减 epoch；直播里过尖的条也是这个病 |
| GRU 不优于 MLP | 1 秒过程近似一阶，或 N 太大过拟合 | 把 `n_hist` 改成 4 和 12 各跑一次 | 三次同向无增益就据实记录 |
| `live_probe` 导入失败 | 没和两个 `.py` 放在一起，或先 import 了会执行的脚本 | 报错栈 | 三个脚本同目录，先 `python train_head.py` 再直播 |
| 概率条永远跟着当前 `LOOK` 走 | 头在背惯性 | 低头瞬间条是否滞后约 1 秒 | 检查训练标签是 $t+\Delta$ 而误用了 $t$ |
| Chong 的 `demo.py` 因 PyTorch 报错 | 仓库按 0.4 验证 | README 的环境声明 | 放弃运行，只读 `model.py`；本课不依赖它出数 |
| EfficientDet 把脸认成 person、把杯认成 bowl | COCO 名称不匹配 | 临时打印 `category_name` | 把 `bowl` 加进杯子名单，或换一只更像 cup 的杯子 |

## 11. 前沿与改造

当前帧注视目标，2025 年的公开做法是 Gaze-LLE：冻结 DINOv2，只训轻量解码器，在 GazeFollow 上训练，再到 VideoAttentionTarget 上微调 in/out 头。官方仓库 `fkryan/gazelle` 提供 Hub 加载，`gazelle_dinov2_vitb14_inout` 同时出热图和「目标是否在画面内」。这和 Chong 2020 的双通路 ResNet 加 ConvLSTM 是同一任务的新零件：骨干换成视觉基础模型，时间建模变弱，单帧更强。Gaze360 走的是另一条：3D 方向加不确定度，时间用来消单帧歧义，不是用来预报 1 秒后。Lai 等的 CSTS 才是预报：第一人称、视听、未来热图。桌宠既需要第三人称的「他看不看我」（更接近 LAEO / in-out / 看镜头分类），也需要 1 秒预判（更接近 CSTS 的时间方向，只是输出被收成二元）。

我们差在两头。规模上，没有 GazeFollow 那种跨场景标注，也没有 Ego4D 的小时级第一人称；这是钱和标注，本课不装成已经跨过。机制上，有三件本课教的东西前沿也还分开做：外生过程与可控动作分通道、预报对惯性基线的增益、记忆只条件化预测而不写人格。Gaze-LLE 再准，也只告诉你现在看哪；不接上 $\Delta$ 步预测，桌宠仍是反应式的。Generative Agents 的记忆流再像人，也不给出 Brier。

动手改造（选做，每个都写预算和失败判据）：

1. 换读出器，不换预报头。用 Gaze-LLE 的 in/out 或热图在镜头附近的积分，替代 5.2 的角度规则，当新的 `look` 标签，重跑 Step 4。预算：有网下载 ViT-B 权重，CPU 可推一张图，直播 10 Hz 在 CPU 上可能掉到几 Hz，可离线打标签。预期：转折帧更干净，Brier 下降。失败：标签和肉眼对视不一致更严重（热图在键盘上，你其实在看镜头），则退回角度规则。
2. 把桌宠动作拼进 $m_t$。有第 30 课动作日志或键盘假头部时，把「看左 / 看右 / 伸手 / 不动」做成 one-hot，拼到 `vec` 末尾，重训 GRU。预算：改 `FEATS` 和采集脚本各十几行，重采 10 分钟带动作的对视。预期：你突然转头或出声后，人看过来的概率升得比无动作条件更快。失败：动作通道被模型学成常数（你几乎总是不动），消融时去掉 $a_t$ 数字不变。
3. 音频作外生线索。Lai 等报音频在 Ego4D / Aria 上各 +2 点 F1。你在 Step 3 的 `voice` clip 上加一维：短时能量是否超过阈值。预算：`sounddevice` 或 OpenCV 不管声，用系统麦克风写 RMS 进 jsonl，一小时。预期：出声后 1 秒 `look` 概率升。失败：宿舍噪声让该维恒为 1，Brier 不变。失败就停，不要上语音识别。
4. $N$ 和 $\Delta$ 各扫一列。$N\in\{1,4,8,16\}$，$\Delta\in\{5,10,20\}$ 步（0.5/1/2 秒）。预算：CPU 上一小时。预期：$\Delta$ 越长，惯性越弱、学习头相对增益越大，直到 2 秒时人人都无法预报、大家一起崩。失败：所有格子都无增益，检查标签是否泄漏了未来（`make_samples` 的索引）。

顺手能看到的论文方向：Chong 说时间有助于视频里的注视目标；你的 GRU 对 MLP 的 Brier 差，是这个方向在二元、1 秒、单人桌面上的缩微版。同向即 GRU 更好，反向则写清设定差异（他们估现在，你报未来；他们热图，你二元）。Gaze360 说多帧加不确定度有帮助；你可以把 GRU 的 $|\sigma(z)-0.5|$ 当自信度，看低自信帧的 Brier 是否更差，这是不确定度是否有序，不是论文数字复现。

## 12. 论文与延伸

1. Gaze360（Kellnhofer, Recasens, Stent, Matusik, Torralba，ICCV 2019，[arXiv:1910.10088](https://arxiv.org/abs/1910.10088)）。238 名被试、室内外、近 360° 朝向的 3D 注视估计；模型吃多帧，并用 pinball 损失报不确定度。阅读问题：它解决的是估计还是预报？多帧输入消的是哪类歧义？超市顾客注意力那个应用，和桌宠「他看不看我」差在时间地平线上还是差在输出空间上？
2. ETH-XGaze（Zhang, Park, Beeler, Bradley, Tang, Hilliges，ECCV 2020 Spotlight，[arXiv:2007.15837](https://arxiv.org/abs/2007.15837)）。110 人、18 台单反、超过一百万张图，极端头姿下的注视估计，并给出统一协议。阅读问题：为什么极端头姿对桌宠是刚需（人低头看手机、侧过身拿杯子）？论文强调头姿变化大时，只看眼睛或只看头都会坏，这对你 5.2 节同时用矩阵和 `eyeLookDown` 意味着什么？
3. Detecting Attended Visual Targets in Video（Chong, Wang, Ruiz, Rehg，CVPR 2020，[arXiv:2003.02501](https://arxiv.org/abs/2003.02501)），仓库 [ejcgt/attention-target-detection](https://github.com/ejcgt/attention-target-detection)。输出注视热图加 in/out；`ModelSpatial` 的场景 4 通道和 `ModelSpatioTemporal` 的 ConvLSTM 是第 6 节的精读对象。阅读问题：in/out 头和本课「看镜头」是不是同一件事（不是，看镜头是目标等于相机，in/out 是目标在不在画面里）？ConvLSTM 的损失是当前帧热图还是未来帧热图？`demo.py` 跑的是当前估计，怎样改才变成 $\Delta=1$ 秒预报（不要在老 PyTorch 上硬改，只写设计）？
4. Listen to Look into the Future（Lai, Ryan, Jia, Liu, Rehg，ECCV 2024，[arXiv:2305.03907](https://arxiv.org/abs/2305.03907)）。第一篇同时用视频和音频做第一人称注视预判的工作，Ego4D 与 Aria 上音频各 +2.5%、+2.4% F1。阅读问题：$\tau_o$ 观察窗和 $\tau_a$ 预判窗在问题定义里如何分开？为什么作者认为视听必须空间、时间分开融合（第一人称头部一转，场景和声音源的同步关系会断）？桌宠是第三人称摄像头，音频改造（第 11 节第 3 条）还能搬哪一部分、搬不动哪一部分？
5. LAEO-Net（Marín-Jiménez, Kalogeiton, Medina-Suárez, Zisserman，CVPR 2019，[arXiv:1906.05261](https://arxiv.org/abs/1906.05261)）。检测视频里两个人是否在对视（Looking At Each Other）。阅读问题：桌宠与人的对视，是不是 LAEO 的二人特例（一方是相机）？为什么互看需要一段时间窗口，单帧朝向点积不够？这和环形记忆的动机如何对应？
6. Gaze-LLE（Ryan, Bati, Lee, Bolya, Hoffman, Rehg，CVPR 2025 Highlight，[arXiv:2412.09586](https://arxiv.org/abs/2412.09586)），代码 [fkryan/gazelle](https://github.com/fkryan/gazelle)。冻结 DINOv2，只训注视解码器。阅读问题：它和本课小头的分工有何同构（冻结感知、学习一个小出口）？Hub 权重报的是现在还是 1 秒后？若你用它的热图当标签再训预报头，泄漏未来的错误会出在哪一步？
7. V-JEPA 2（Assran 等，[arXiv:2506.09985](https://arxiv.org/abs/2506.09985)）。动作预判口径：Epic-Kitchens-100 上 ViT-g 的 recall-at-5 为 39.7。阅读问题：厨房动作预判和桌面「会不会看我」共享哪一条评测纪律（先定地平线，再和强基线比）？为什么不能把视频问答分数拿来代替这一课的 Brier？
8. Generative Agents（Park, O'Brien, Cai, Morris, Liang, Bernstein，[arXiv:2304.03442](https://arxiv.org/abs/2304.03442)）。记忆流、反思、计划；评的是像不像人。阅读问题：记忆流里一条「他喜欢和我说话」若被写进桌宠世界模型，1 秒注视的校准会往哪边坏？论文的检索三项（近因、重要度、相关性）哪一项最接近本课环形缓冲，哪两项必须留在规划器？

选读：Recasens, Khosla, Vondrick, Torralba 的 GazeFollow（NIPS 2015，项目页 [gazefollow.csail.mit.edu](http://gazefollow.csail.mit.edu/)）是注视目标检测的起点数据集，Chong 与 Gaze-LLE 都还在用。读它是为了分清「图里的人看图里的哪」和「对面的人看不看相机」。

第 32 课要把这一课的人预测接到身体上：预测人会看过来时主动看人，人低头则降低打扰。那一步会查询你今天这个头。头若只是惯性的复读机，对视行为会变成「你看我我就看你」的镜子，装上安全层也救不回先想再做。




