---
id: 16_multimodal_preference_optimization
title: "多模态 DPO / mDPO"
summary: "普通 DPO 是不是只学了个说话腔调？加上媒体条件偏好和 reward anchor 之后，模型挑答案时会更认真去看真实的图像、音频或视频证据吗？"
unit: alignment
play_tools: []
checkpoints:
  - "从 Bradley–Terry 模型一步步推出 DPO，讲清 reference、β 和藏在里面的 KL 约束各干什么。"
  - "构造换错图、抽掉图、打乱图这三种媒体反事实，加上“说得流利但事实错”的 hard negative。"
  - "实现 CoPO 和 AncPO，跟普通 DPO 做等 pair 数、等预算的对照。"
  - "审计三件事：模型是否真依赖媒体条件、chosen 概率有没有掉、基础能力忘了多少。"
---

# 第 16 课：多模态 DPO 与 mDPO

> 本课产出：完成一个边界清楚的 mDPO 复现实验。主实验只做论文中的图像方法；错图、其他图像退化以及音频、视频都放进独立扩展，不与论文结果混在一起。

## 1. 偏好训练必须真的看媒体

 到[第 15 课](15_joint_multimodal_sft.md)为止，我们手里有一份联合 SFT 出来的 checkpoint：读文字、看图、听音频、开口说话，各项能力都在线。但从第 1 课到第 15 课，训练方式只有一种：SFT；把标准答案摆在模型面前让它模仿。模仿有个天生盲区：模型从没见过"错误示范"。数据里全是对的，它不知道什么叫错，更分不清"两个都通顺的回答里，哪个是看图说的、哪个是编的"。多模态模型最常见的病；看着收据把 37.50 读成 57.50，语气还特别笃定；SFT 治不了，因为病例从来没进过训练集。

一条偏好优化训练管线。原料换了：不再是单份标准答案，而是偏好对（preference pair）：同一个问题、同一张图，配一份较好回答（chosen）和一份较差回答（rejected），整条数据叫一个 pair。核心算法是 DPO（direct preference optimization，直接偏好优化），先说直觉：把好答案的生成概率抬高、坏答案的概率压低，同时拿一份冻结的出发点模型当锚，别让模型为了拉开差距跑得离出发点太远。多模态还有个特有的坑：模型可以完全不看图，仅凭"哪边措辞更像好答案"就把这道二选一做对；mDPO 论文管这叫无条件偏好（unconditional preference）。它的解法是再加一场"图像对照赛"：问题和好答案都不动，只把原图换成挖掉大部分信息的裁剪图，要求模型在原图条件下更愿意说出这个答案。本课主线就是把 mDPO 的图像方法边界清楚地复现出来（代号 C0）；音频、视频扩展另开赛道（代号 C1），不和论文结果混桌吃饭。

三个理由。一，幻觉问题在 SFT 框架里无解：负例必须显式进入训练目标，这课是它第一次进来。二，[第 17 课](17_grpo_rlvr.md)的 GRPO/RLVR 要用 reference、KL、reward、偏好投机这一整套概念和基础设施，本课是这套东西最便宜的训练场；离线数据、不用在线采样，翻车成本低。三，这条路线是工业流水线的正式一站：Nemotron 的官方后训练顺序就是 SFT、MPO、GRPO 一路排下去。跳过这课直接上 RL，等于没学走先学跑。

拿一张收据照片问总金额：B 臂（普通 DPO）可能照样答错，C0 臂答对；把原图换成预注册的裁剪图，C0 的 chosen 隐式 reward 从正值掉到零下（第 19 节的样例里是 0.31 掉到 -0.02）。这两个数字不在论文里，在你自己的逐 case 审计文件里，每一个都能按 hash 重放。

本课术语：

| 术语 | 简要解释 |
|---|---|
| 偏好对（pair） | 同一问题配一好一坏两份回答的训练数据，好的叫 chosen，差的叫 rejected |
| 奖励模型（reward model） | 专门学"给回答打分"的另一个模型；经典 RLHF 先训它再用 RL 刷分，DPO 把这一步省掉了 |
| 参考模型（reference） | 训练起点 policy 的冻结副本，当锚用：衡量新模型偏离出发点多少 |
| KL 约束 | KL 散度是量两个概率分布差多远的尺子；训练时拿它拴住模型，别离参考模型太远 |
| 隐式 reward | DPO 不训奖励模型，直接拿 β 乘上"policy 与 reference 的 log 概率差"当分数 |
| 无条件偏好 | 不看图也能把偏好二选一做对；mDPO 要治的病 |
| CoPO / AncPO | mDPO 加的两项损失：原图对裁剪图的图像对照赛；chosen 分数不许跌破 0 的保底锚 |
| likelihood displacement | chosen 和 rejected 概率一起往下沉：损失只看差值，差值拉开了、两边都在跌 |
| fresh generation | 别只在旧 pair 上算概率，让训好的模型现场生成新回答再评 |
| 离线 / 在线偏好 | 离线：训练前固定全部 pair（本课）；在线：边训边让当前模型生成新回答（第 17 课） |

## 2. 本课要解决的问题

偏好优化使用成对回答训练模型。每条数据包含同一个问题和媒体、一个较好回答（`chosen`）以及一个较差回答（`rejected`），也就是上面说的偏好对。本课只研究离线设置：训练前固定全部偏好对，训练过程中不让当前模型重新生成回答。

先把实验名称固定，后文不再改变含义：

- `A`：chosen-only SFT，课程增加的训练量对照；
- `B`：标准 DPO；
- `C`：主表里的第三臂，**始终等于 `C0`**；
- `C0`：论文复现。只接收图像；CoPO 比较"原图"和论文规定的低信息裁剪，问题与
  `chosen` 回答完全相同；同时加入 AncPO；
- `C1`：课程扩展。错图、移除图像、模糊/遮挡/压缩等其他变换，以及音频、视频实验
  全部放在这里。C1 使用独立的数据、配置、实验 ID 和报告，不能写成论文原 recipe；
- `exp16x_mpo`：MPO 扩展，仍是另一项实验，不占用 C 或 C1 的名称。

这一区分很重要，也是本课最容易翻车的地方。mDPO 论文研究的是图像，默认 CoPO 负图像是从原图随机裁出的 `Crop 0–20%` 低信息区域。随机错图只是论文中的消融之一，音频和视频没有出现在原论文 recipe 中。把自己的扩展写成论文复现，整份报告直接作废。

三个主实验组必须从同一个经过基础指令微调的 checkpoint 起步——第 15 课交付的 `joint-sft-v1` 是自然起点。若从 MiniMind-O checkpoint 开始，本课复现的是 mDPO 的方法和变量控制，不是论文中 Bunny 或 LLaVA 数值的逐点复现。报告中要写"mDPO 方法复现（MiniMind-O 移植）"，不能写"复现论文表 1 数值"。

本课不做在线采样，也不把 GRPO 放进同一张对照表。DPO 与 C0 使用训练前固定的偏好对；[第 17 课](17_grpo_rlvr.md)才研究由当前模型持续生成回答的 RLVR。

## 3. 要验证的结论与失败条件

**C0 的研究问题：** 普通 DPO 是否可能只学会 chosen 与 rejected 的语言差异，而没有充分使用图像？在 response DPO 之外加入图像 CoPO 和 chosen-reward anchor 后，模型基于图像作答的正确率能否提高，同时不损害原有能力？

**预期结果：**

- 普通 DPO 会放大较好回答与较差回答之间的语言差异；
- C0 的 CoPO 让同一个 chosen 回答在原图上的隐式 reward 高于低信息裁剪图；
- AncPO 把 chosen 回答的隐式 reward 约束在预注册 anchor 之上，避免 chosen
  likelihood 一直下降；
- C0 相对 B 的 held-out 图像正确率和原图/裁剪图区分能力提高。

**拒绝标准：** 在偏好对和训练预算相同的条件下，如果普通 DPO 与 mDPO 在证据正确率、图像条件依赖和能力回归上没有显著差异，就不能声称 C0 更好。C1 即使成功，也不能替 C0 补结果；两者回答的不是同一个问题。

## 4. 固定训练起点

```text
checkpoints/exp16_start/
├── policy/
│   ├── model.safetensors
│   └── config.json
├── tokenizer/
├── modality_connectors/
├── talker/                  # 可冻结；本课主要优化文本响应
└── sft_metrics.json
```

必须保存一份不可训练的参考模型（reference policy）。它是 step 0 policy 的逐参数精确副本，即 `reference = exact_copy(policy_at_step_0)`；复制完成后立即冻结，并在第一批数据上核对两者 logits。别小看这一步：reference 一旦悄悄变化，后面所有隐式 reward 的定义都失效。

起点要求：

- 至少能对较好回答和较差回答计算条件 log-prob；
- image token 在训练模型和参考模型中完全一致；
- assistant-only loss mask 正确（忘了 mask 的定义回[第 01 课](01_baseline_reproduction.md)第 5 节）；
- 对同一偏好对可导出 `logp_chosen` 和 `logp_rejected`；
- 起点在固定 SFT 回归集上有基线；
- 图像预处理可以显式接收预生成的原图和裁剪图，训练时不会再次随机裁剪。

## 5. 学完后应能完成

完成后你应能：

- 从 Bradley–Terry 模型（把"谁赢谁"的概率写成两个实力分之差过 sigmoid 的经典配对比较模型）推导 DPO 的二分类目标；
- 解释参考模型、β 和隐式 KL 的作用；
- 解释 CoPO 为什么保持问题和 chosen 回答不变，只改变图像；
- 从原图生成可重放的 `Crop 0–20%` 低信息裁剪，并验证其 hash；
- 说明 AncPO 解决什么问题，以及它为什么不能替代 response DPO；
- 识别 position bias、length bias、style bias 和 teacher bias（位置、长度、文风、出题老师留下的指纹——四种让模型不看内容也能猜对的偏差）；
- 用 fresh generation 和图像任务 verifier（可执行的规则判分程序）判断模型是否真的变好；
- 明确区分 C0 论文复现与 C1 Omni 扩展；
- 区分偏好胜率、任务正确率、校准和能力遗忘；
- 检查较好回答的生成概率是否下降，以及模型是否过度适应偏好数据。

## 6. 原理:边造边讲

三个机制，每个都按同一节奏走：为什么需要（直觉）、怎么运转（机制）、精确定义（数学）、在哪里实现（代码落点）、怎么证明做对了（验证）。原文公式一个不少。

### 6.1 DPO 目标：抬高 chosen、压低 rejected、绳子拴在 reference 上

经典 RLHF 分两步：先用人类标注训一个奖励模型（reward model，专门学"给回答打分"的模型），再用强化学习让 policy 把分数刷高，同时加一条 KL 约束（KL 散度：衡量两个概率分布差多远的尺子）把 policy 拴在出发点附近。为什么要拴？因为分数是学出来的，有漏洞；不拴，模型会跑到奖励模型没见过的区域刷高分，输出却越来越怪。DPO 的贡献是一步数学化简：在"KL 约束下最大化 reward"这个问题里，最优 policy 与 reward 之间存在解析对应，反过来就能用 policy 自己的概率表示 reward——奖励模型被消掉了，两步流水线塌缩成一个监督式损失。翻译成人话：好答案概率抬高、坏答案概率压低，同时别跑离参考模型太远；β 就是那根绳子的松紧旋钮。类比失效处：真绳子只拉不推，而 β 同时缩放奖励差——β 越大，越小的偏离就足够判出胜负，模型反而更不需要跑远。

每个 pair 四次 forward：policy 和 reference 各算一遍 chosen 和 rejected 的 sequence log-prob，共四个数。policy 相对 reference 的 log 概率差乘上 β，就是那份回答的隐式 reward；chosen 与 rejected 的隐式 reward 之差过 sigmoid 做二分类，要求 chosen 赢。"胜率等于分差过 sigmoid"正是 Bradley–Terry 模型的形式，DPO 就是从它出发推的。

DPO 把 pair 偏好写成二分类目标。设 $x$ 为共享 prompt，$y_w$ 为 chosen，$y_l$ 为 rejected，$\pi_{\mathrm{ref}}$ 为冻结 reference policy。定义 reference-relative 隐式 reward：

$$
r_\theta(x,y)
=
\beta\left[
\log\pi_\theta(y\mid x)
-\log\pi_{\mathrm{ref}}(y\mid x)
\right]
$$

DPO 最小化：

$$
\mathcal L_{\mathrm{DPO}}
=
-\log\sigma\!\left(
r_\theta(x,y_w)-r_\theta(x,y_l)
\right)
$$

注意损失里只有差值。这埋着一个后面反复出现的坑：只要 rejected 的概率掉得比 chosen 快，两边可以一起沉，损失照样下降——这就是 likelihood displacement，第 18 节诊断表和 6.3 的 AncPO 都冲着它来。

第 14 节伪代码里的 `sequence_logp` 与 `response_logit`；真实现参考第 13.1 节锁定的公开 `mdpo_trainer.py`。prompt 和 padding 不计入 sequence log-prob，靠的还是 assistant-only mask。

先给定一个 toy sequence 及其 policy/reference token 概率，手算四个 sequence log-prob，再代入两式。这里核对的是给定数值下的公式与 mask，不是声称已经得到真实模型的 log-prob。随后解释：

- β 变大/变小时，policy 偏离 reference 的代价怎样变化；
- chosen/rejected 共享 prompt 为什么重要（做减法时 prompt 部分相消，比较才落在回答本身）；
- sequence log-prob 是否按长度归一化；
- 为什么 chosen 与 rejected 都可能同时降概率。

用单样本脚本核对手算值与 trainer 输出，并分别测试 sequence-sum 和 length-normalized 两种约定。公式、代码与配置中的归一化选择一致后，本节才通过。

### 6.2 什么叫“忽略图像”

一个 pair 中，chosen 往往比 rejected 写得更完整、更有条理。模型可能只根据措辞判断哪边更好，即使图像被换掉，判断也不变。这就是 mDPO 论文讨论的 unconditional preference（无条件偏好）：偏好确实学会了，条件里的图像却形同虚设。这个病肉眼看不出来——margin 在涨、pair accuracy 在涨，所有常规指标都好看。

C0 用一个很直接的控制变量实验查它：问题不动、chosen 回答不动，只换图像，看隐式 reward 有没有反应。

记文本问题为 $t$，原图为 $m^+$，从原图生成的低信息裁剪为 $m^-_{\text{crop}}$。问题不变，chosen 回答也不变：

$$
x^+=(t,m^+), \qquad
x^-=(t,m^-_{\text{crop}}), \qquad
y=y_w
$$

定义 chosen 回答的图像条件差：

$$
g_\theta
=
r_\theta(x^+,y_w)
-
r_\theta(x^-,y_w)
$$

裁剪图在数据编译阶段生成并冻结（第 12 节 Step 3），训练时走第 14 节伪代码的 `prompt_crop` 路径；两路的文本 token、labels 与 response mask 必须逐位相同，差异只允许出现在图像张量上。

$g_\theta>0$ 表示 policy 相对 reference 更愿意在原图条件下生成该 chosen 回答。它只说明训练目标区分了两种图像，不能单独证明生成答案更正确；因此第 16 节还会检查 held-out 图像任务的 fresh generation。

这里不放错图、图像移除、模糊或音视频。只要实验引入这些变量，无论用于训练还是评测，都登记为 C1。

### 6.3 C0 的 CoPO 与 AncPO：一场对照赛，一条保底线

CoPO（conditional preference optimization，条件偏好优化）是把 6.2 的诊断直接变成训练目标：既然模型可能不看图，那就明着奖励"原图下愿意说 chosen"、惩罚"裁剪图下也照说不误"。AncPO（anchored preference optimization，锚定偏好优化）则是给 likelihood displacement 上保险：DPO 一家只管差值，chosen 可以边赢边沉；AncPO 给 chosen 的隐式 reward 立一条 0 分保底线，不许沉到水面以下。

C0 在标准 response DPO 之外增加两项损失。

第一项是 CoPO。它使用相同的 $t$ 和 $y_w$，只比较原图与论文的低信息裁剪：

$$
\mathcal L_{\mathrm{CoPO}}
=
-\log\sigma\!\left(
r_\theta(x^+,y_w)-r_\theta(x^-,y_w)
\right)
$$

第二项是 AncPO。它要求原图条件下 chosen 回答的隐式 reward 高于 anchor $a$：

$$
\mathcal L_{\mathrm{AncPO}}
=
-\log\sigma\!\left(
r_\theta(x^+,y_w)-a
\right)
$$

论文默认只给 chosen reward 加正向 anchor。C0 固定 $a=0$，不再给 rejected 回答或裁剪图像增加负向 anchor。最终目标是：

$$
\mathcal L_{\mathrm{C0}}
=
\mathcal L_{\mathrm{response-DPO}}
+
\mathcal L_{\mathrm{CoPO}}
+
\mathcal L_{\mathrm{AncPO}}
$$

三项默认等权相加，$\beta=0.1$，sequence log-prob 使用求和而非长度归一化。若要改权重、anchor、裁剪范围或归一化，必须另起 ablation ID，不能把结果并入 C0 主表。

三项损失各自控制什么变量，一张表看清：

| 损失 | 保持不变 | 改变的变量 | 训练信号 |
|---|---|---|---|
| response DPO | 原图与问题 | chosen / rejected 回答 | 哪个回答更好 |
| CoPO | 问题与 chosen 回答 | 原图 / 低信息 crop | chosen 回答是否得到图像支持 |
| AncPO | 原图、问题与 chosen 回答 | 不构造成对样本 | chosen reward 不低于 0 |

若 CoPO batch 中 chosen 文本发生变化，它就同时混入了语言偏好，无法再解释为图像条件实验。

三项 loss 在第 14 节伪代码里各占一行；配置对应第 13 节 `objective` 块的 `copo` 与 `ancpo` 字段。

先在 32 个 pair 上分别关闭 CoPO 和 AncPO，确认 loss 与梯度只改变对应项。MPO 的 quality/generation 目标不属于 C0，也不属于 C1——它有自己的实验名 `exp16x_mpo`。

## 7. 偏好数据字段

### 7.1 C0：论文裁剪数据

C0 的一条记录同时提供 response DPO 所需的 chosen/rejected，以及 CoPO 所需的原图/裁剪关系。CoPO 不复制回答文本，只用 `response_ref: chosen` 指回同一个 chosen，避免编译时悄悄产生两份不同回答。

```json
{
  "schema_version": "exp16.c0.v1",
  "track": "C0",
  "pair_id": "pref_img_000103",
  "source_item_id": "receipt_000103",
  "prompt": {
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image", "asset_id": "img0"},
          {"type": "text", "text": "收据上的总金额是多少？"}
        ]
      }
    ],
    "assets": [
      {
        "asset_id": "img0",
        "uri": "sha256://...",
        "sha256": "ORIGINAL_IMAGE_SHA256",
        "type": "image"
      }
    ]
  },
  "chosen": {
    "text": "总金额是 37.50 美元。",
    "evidence": [{"asset_id": "img0", "region": [0.61, 0.78, 0.84, 0.86]}]
  },
  "rejected": {
    "text": "总金额是 57.50 美元。",
    "error_type": "visual_digit_error"
  },
  "copo": {
    "response_ref": "chosen",
    "positive_image_sha256": "ORIGINAL_IMAGE_SHA256",
    "negative_image": {
      "derived_from_sha256": "ORIGINAL_IMAGE_SHA256",
      "uri": "sha256://...",
      "sha256": "CROPPED_IMAGE_SHA256",
      "transform": {
        "name": "paper_crop_0_20",
        "implementation": "exp16_crop_v1",
        "seed": 17,
        "crop_box_xywh_px": [811, 1042, 302, 190],
        "retained_area_ratio": 0.143,
        "resize_to_px": [336, 336],
        "interpolation": "bicubic"
      }
    }
  },
  "preference": {
    "annotators": 3,
    "agreement": 1.0,
    "rubric_version": "grounded_v2"
  },
  "source": "public-or-self-collected",
  "license": "explicit",
  "split": "train",
  "generator": {"chosen": "human_or_model_id", "rejected": "mutation_v3"},
  "difficulty": {"modality_required": true, "hard_negative": true}
}
```

`hard_negative`（难负例）指表面看着很像对的错误答案——最能逼模型真去看图的那种。

论文把默认负图像写成从原图随机裁出的 `Crop 0–20%`。本课把它落实为：裁剪框面积占原图面积的比例满足 $0<\rho\le0.20$，随后按冻结的插值方式缩放到模型输入尺寸。论文没有公开完整的图像预处理流水线，因此 `exp16_crop_v1` 是本课明确记录的实现选择；报告必须给出 crop box、面积比例、resize、插值、seed 和输入/输出 hash。

裁剪在数据编译阶段完成，训练时只按 hash 读取结果。schema 校验器逐条检查：

1. `track` 必须为 `C0`，输入模态必须为 image；
2. `copo.response_ref` 必须为 `chosen`，CoPO 不允许另一份回答文本；
3. `positive_image_sha256` 和 `derived_from_sha256` 都等于原图 hash；
4. 裁剪面积比例在 $(0,0.20]$，记录值与 crop box 实算值一致；
5. 用记录的实现版本、crop box、resize 和插值重放后，输出 hash 完全相同；
6. 原图和裁剪图 hash 不同；
7. C0 记录中不得出现 `wrong_media`、其他 `degradation`、audio 或 video 字段。

prompt hash、原图 hash 和回答 hash 都要进入 pair ID。按 `source_item_id` 与原图 hash group 切分，原图及其裁剪不能跨 train/dev/test。

### 7.2 C1：课程扩展数据

错图、移除媒体、除论文 crop 以外的模糊/遮挡/压缩/打乱，以及所有音频、视频反事实，使用另一套 schema。下面只展示扩展字段；它不能出现在 C0 JSONL 中。

```json
{
  "schema_version": "exp16.c1.v1",
  "track": "C1",
  "extension_id": "c1_audio_wrong_segment_v1",
  "base_pair_id": "pref_audio_000271",
  "modality": "audio",
  "counterfactual": {
    "kind": "wrong_asset",
    "operator": "registry_lookup",
    "implementation": "c1_counterfactual_v1",
    "seed": 17,
    "parameters": {"asset_id": "audio_wrong_019"},
    "input_sha256": "ORIGINAL_AUDIO_SHA256",
    "output_sha256": "WRONG_AUDIO_SHA256"
  },
  "claim_scope": "course_extension_not_original_mdpo_recipe"
}
```

C1 中每种模态和反事实操作都使用独立 `extension_id`。错资产必须记录 registry ID 与 hash；确定性变换必须记录实现版本、全部参数、seed 和输入/输出 hash；媒体移除也要显式写成 `kind: removed`，不能用缺失字段表示。C1 的 checkpoint、指标与报告单独落盘。

### 7.3 response rejected 的类型

下面列的是 chosen/rejected 回答之间的错误类型，不是 CoPO 的负图像类型。C0 的 CoPO 负图像始终只有论文 crop。

1. **事实错误**：语言自然，但媒体事实错；
2. **unsupported reasoning**：结论可能对，推理引用不存在证据；
3. **visual relation error**：对象、数量、属性或空间关系错误；
4. **OCR error**：字符或数字读错；
5. **style-only**：事实相同，只有风格差异，只进入诊断集；
6. **refusal error**：有充分证据却拒答；
7. **hallucinated detail**：加入图像中不存在的细节。

编译后按错误类型统计 pair 数与 response token 数，并抽查每类 20 对。若标注者不看图像也能稳定选择 chosen，该 pair 只能进入 style 诊断集，不能进入 C0 主训练集——它训练的只会是文风偏好。

## 8. 数据来源与开放边界

先核验论文仓库指向的
[`fwnlp/mDPO-preference-data`](https://huggingface.co/datasets/fwnlp/mDPO-preference-data)：记录实际可下载的 pair 字段、图像来源、许可、split 和样本数。若它不能直接提供第 7 节所需的裁剪 manifest，就从可追溯原图离线生成 C0 crop，并把这一段写成"课程补全的数据编译"，不能说是仓库原样提供。

也可以从以下来源建立 C0 图像 pair：

- 自有 SFT held-out prompt，由人工或可追踪生成器产生多个回答；
- 规则可验证任务，如 OCR、计数和坐标；
- 明确许可的公开图像数据。

ASR、音频和视频数据只用于 C1，不得进入 C0 manifest。MMPR 数据只用于 `exp16x_mpo`，也不得并入 C0。

生成器使用 closed model 不等于数据不可用，但必须记录：

- 模型/版本/日期；
- prompt 和 sampling；
- 是否人工复核；
- 过滤、去重；
- 是否允许再分发。

完成 registry 后，每个来源随机抽 20 对，核对资产可访问性、license 和生成记录。论文公布 aggregate 结果，不代表完整偏好数据、原图、裁判 prompt 或训练 mixture 已公开。报告分别填写 `pairs open?`、`assets open?`、`judge open?`、`recipe open?`；没有证据的字段填 `unknown`。

禁止将评测集标注转成训练 pairs；禁止用最终 test judge 同时生成 rejected。

## 9. C0 数据规模与边界

论文实验使用约 10K 图像偏好数据。课程先复现这个量级，不要一开始扩成全模态百万 pair。

| 档位 | pair 数 | 输入 | 目的 |
|---|---:|---|---|
| unit | 32 | image | 手算三项 loss，验证 crop 与 hash |
| smoke | 1k | image | 验证数值、mask、checkpoint 恢复 |
| C0 main | 10k | image | A/B/C（C 即 C0）三 seed |

C0 main 中所有 pair 都必须标记 `modality_required=true`。style-only pair 只留在诊断集。如果要做 50K/100K 的 scale-up，命名为 `c0_scale_ablation_*`，不能替换 10K 主表。

数据编译后重算 pair 比例和 response token 比例，并按 source、asset hash 分组检查 train/dev/test。任何跨 split 资产，或比例超出预注册误差的版本，都要重新生成 manifest。

C1 没有"默认规模"。每个 extension 先用 1K smoke，再单独写数据规模、模态、操作和停止条件。C1 的 pair 数不能加到 C0 数据量里。

## 10. 三个对照实验组

主表只比较以下三臂。表中的 C 就是 C0，没有另一个名为 C 的扩展版本。

| 臂 | 目标 | 作用 |
|---|---|---|
| A：chosen-only SFT | $-\log \pi_\theta(y_w\mid x)$ | 排除额外 chosen 训练的影响 |
| B：DPO | 标准 DPO | 离线偏好基线 |
| C（即 C0）：image-only mDPO | response DPO + 原图/`Crop 0–20%` CoPO + AncPO | 复现论文方法 |

为什么要 A 臂？C0 的 AncPO 相当于给 chosen 加训练量；没有 A，你分不清收益来自偏好信号还是来自"chosen 又多学了几遍"。主表只保留这三臂。C1 不增加第四行，也不把指标追加到 C 的单元格。训练前 diff 三份 resolved config；除 objective 与 C0 crop forward 所需字段外，差异必须逐项解释。

公平控制：

- 相同起点 policy；
- B/C0 使用同一冻结 reference；
- 相同 pair 和顺序；
- 相同累计 prompt/response token；
- 相同 trainable modules；
- 相同 LR、global batch、seed；
- A 的 token 预算与 B/C0 的 policy forward token 对齐；
- 同一 generation/eval 配置。

若做论文数值复现，应另外使用论文的 Bunny/LLaVA 起点、数据与评测；MiniMind-O 主表只支持"方法移植"的结论。两类结果不能放在同一列。

## 11. 目标 recipe

先完成 C0，再决定是否做 C1：

| 阶段 | 工作与进入下一阶段的条件 |
|---|---|
| Stage 0 | 审计 C0 pair、原图和 crop manifest |
| Stage 1 | 通过 32 pair 公式单测与 1k pair smoke |
| Stage 2 | 冻结 C0 配置与统计计划 |
| Stage 3 | 运行 A/B/C 主实验，其中 C 固定为 C0 |
| Stage 4 | 对固定 checkpoint 做离线评测与 fresh-generation 评测 |
| Stage 5 | 完成 C0 逐 case 审计后再写结论 |
| Stage 6 | 可选地另起 C1 extension，结果不回写 C0 主表 |

Talker 与 audio connector 在 C0 中冻结。若要偏好语音输出，把内容偏好与声学偏好放入独立 C1 实验，避免把两种结论混在一起。每个 Stage 的进入条件写入状态文件；上一阶段未通过时不启动下一阶段。

## 12. 实验步骤

### Step 1：冻结 SFT 起点和 reference

reference policy 是 step 0 policy 的冻结副本，用于衡量 policy 的相对变化。对 policy/reference 同一 batch 验证 logits 完全一致或只在允许精度误差内不同。

记录 trainable parameter names，并在 optimizer step 前后比较 reference hash 与梯度。hash 不变、所有 reference gradient 为空且 logits 测试通过，才算冻结成功。

### Step 2：审计图像 pair 可判定性

pair 若两边都对、都错或只差风格，会把噪声直接写入目标。随机抽 300 对，由至少两名标注者回答：

- chosen 是否确实更好；
- 差异是否必须看图；
- rejected 错误类型；
- 是否存在两者都错/都对；
- 是否有长度/格式泄漏。

按来源和错误类型计算 agreement。低于 0.8 的来源不得进入 standard；同时保存分歧原因，不能只删除低一致性样本而不修正生成流程——删样本治标，修流程才治本。

### Step 3：编译 C0 crop manifest

先按 `source_item_id` 与原图 hash 切分，再为 train/dev/test 分别生成 crop。每个 pair 只生成一张冻结裁剪图：

1. 用 pair ID 派生 seed；
2. 在原图上采样面积比例 $0<\rho\le0.20$ 的 crop box；
3. 使用预注册插值缩放到模型输入尺寸；
4. 写入 crop box、$\rho$、seed、实现版本和输入/输出 hash；
5. 立即重放一次并核对输出 hash。

不要在 `Dataset.__getitem__` 或 collator 中随机裁剪。否则同一 pair 每个 epoch 看到的图像不同，无法逐 case 复核，也不再符合本课的可重放协议。

### Step 4：做 32-pair 单测和 1k-pair smoke

32-pair 单测先验证公式方向：

- 交换 chosen/rejected 后，response DPO logit 符号翻转；
- 交换原图/crop 后，CoPO logit 符号翻转；
- $a=0$ 时，AncPO 只读取原图 chosen reward；
- CoPO 的原图与 crop forward 使用完全相同的 token IDs 和 response mask；
- reference 的三路 forward 都在 `no_grad` 中。

随后用 1k pair 做小规模过拟合。它用于检查实现，不用于报告泛化能力：

- chosen-rejected margin 上升；
- 原图/crop chosen-reward gap 上升；
- reference 无梯度；
- β 和 loss 数值正常；
- padding 不进入 log-prob；
- checkpoint 恢复后 margin 连续；
- chosen/rejected 没有被交换。

从 100 个 pair 中人工选 5 个手算 margin，和训练日志逐项比对。reference 出现梯度、mask 包含 padding 或恢复后 margin 跳变时，停止后续训练。

### Step 5：冻结 C0 配置

C0 主实验固定论文默认设置：$\beta=0.1$、三项 loss 等权、$a=0$、sequence-sum log-prob、`Crop 0–20%`。训练前保存：

- `resolved_config.yaml`；
- `crop_manifest.jsonl` 与 schema；
- policy/reference SHA；
- tokenizer、image processor 与代码 SHA；
- `stats_plan.yaml`。

若 MiniMind-O 移植因数值问题必须改设置，先保留 C0 原配置和失败日志，再另起 `c0_port_ablation_*`。不要直接改写 C0 的含义。

### Step 6：执行 A/B/C 三臂

以累计 non-padding prompt/response token 作为横轴，每到固定位置保存 checkpoint 并评测：

- pair accuracy；
- chosen/rejected log-prob；
- reference-relative KL proxy；
- response length；
- SFT regression。

三臂最终 token、训练样本顺序和可训练参数必须落在预注册误差内。日志和表头都写 `C/C0`，避免后续把 C1 结果误填进 C。任何一项不一致时，不计算 arm 间胜负。

### Step 7：生成式 blind evaluation

训练式 pair margin 只说明模型更偏向给定 chosen；部署时模型会生成新答案。因此要在 held-out prompt 上重新生成，并冻结解码参数。

对 held-out 图像 prompt 生成新回答，再用：

- 规则 verifier；
- 人工双盲；
- 不接收图像的 style judge（judge：拿另一个模型当裁判打分）；
- 可访问原图的 grounded judge。

两类 judge 分别评分：style judge 不接收图像，grounded judge 必须访问原图——前者衡量文风，后者衡量有据。再用规则 verifier 和人工双盲复核分歧 case；单一 LLM judge 的分数不能作为主要结论。

### Step 8：做 condition dependency test

保持问题、chosen 回答和解码配置不变，先比较 C0 预注册的原图与裁剪图：

- answer accuracy；
- chosen implicit reward；
- 原图/crop reward gap；
- refusal；
- confidence；
- 生成答案变化率。

主表只报告原图/crop 差值和置信区间。答案变化但原图正确率不升时，只能说明模型对图像扰动更敏感，不能说明 grounding 改善。

错图、图像移除和其他变换不在本步骤运行。若要评估它们，先登记 C1，使用独立配置、case 清单、指标文件和报告；即使只做诊断，也不能写进 C0 主报告的结果表。

### Step 9：三 seed 汇总和失败审计

按模态、错误类型、回答长度、来源和难度分层汇总三个 seed，并显示单 seed 点与置信区间。

至少人工看：

- 最高 margin 的 20 个；
- 最低 margin 的 20 个；
- reward/pref 提升但任务变差的 20 个；
- 原图与 crop 输出完全相同的 20 个。

为每个样本记录原图、裁剪图、三臂答案、margin 和人工判断。若总体提升只来自一个 teacher 来源、一个长度区间或一个 seed，报告该限制，不标记为稳定收益。

### Step 10：可选 C1 扩展

只有 C0 报告冻结后才能启动 C1。每个 C1 实验先写一页预注册，至少包含：

- 唯一 `extension_id`；
- 输入模态；
- 反事实操作与它为何有明确的"更差"关系；
- 独立数据 manifest、配置和验收指标；
- 与哪个冻结基线比较；
- `claim_scope: course_extension_not_original_mdpo_recipe`。

例如 `c1_image_wrong_asset_v1`、`c1_audio_mask_span_v1` 和 `c1_video_frame_order_v1` 是三个实验，不是一个"全模态 C1"结果。它们不能回写 A/B/C 主表，也不能写成 mDPO 论文复现。

## 13. 配置示例

下面是主表 C 臂，也就是 C0 的配置。A/B 复用同一模型、数据顺序和训练参数，只更改 `experiment.arm` 与 `objective`。三臂都用 LoRA（低秩适配：冻结原权重，只训练插在旁边的小矩阵，省显存且好回滚）训练同一组模块。

```yaml
experiment:
  id: exp16_c0_mdpo_image_crop
  track: C0
  arm: C
  claim_scope: paper_method_reproduction_image_only
  seeds: [17, 23, 41]
model:
  policy_init: checkpoints/exp16_start/policy
  reference_init: checkpoints/exp16_start/policy
  freeze_reference: true
  trainable: [backbone_lora, vision_connector_lora]
  frozen: [audio_connector, video_connector, talker]
adapter:
  type: lora
  rank: 64
  alpha: 128
data:
  schema: data/exp16/c0/schema.json
  train: data/exp16/c0/train.jsonl
  dev: data/exp16/c0/dev.jsonl
  crop_manifest: data/exp16/c0/crop_manifest.jsonl
  required_track: C0
  required_modality: image
  group_split_by: [original_image_sha256, source_item_id]
  max_prompt_tokens: 16384
  max_response_tokens: 1024
preprocessing:
  crop_implementation: exp16_crop_v1
  runtime_random_image_augmentation: false
  image_processor_sha256: IMAGE_PROCESSOR_SHA256
objective:
  type: mdpo_c0
  beta: 0.10
  sequence_logp: sum
  response_dpo_weight: 1.0
  copo:
    weight: 1.0
    positive_image: original
    negative_image: paper_crop_0_20
    response_ref: chosen
    require_same_prompt_tokens: true
    require_precomputed_crop: true
    retained_area_ratio: {gt: 0.0, lte: 0.20}
  ancpo:
    weight: 1.0
    target: chosen_reward
    anchor: 0.0
train:
  epochs: 3
  lr: 1.0e-5
  scheduler: cosine
  warmup_ratio: 0.10
  global_batch_pairs: 32
  bf16: true
distributed:
  world_size: 8
  fsdp: full_shard
eval:
  suites: [pair, fresh_generation, grounded_image, original_vs_crop, sft_regression]
```

`fsdp: full_shard` 即全分片数据并行：参数、梯度和优化器状态切片到 8 张卡上。这些训练超参数对应论文公开设置。若模型移植需要改动，把改动放进 `c0_port_ablation_*`，并保持 A/B/C 公平；不要覆盖这份 C0 resolved config。

### 13.1 公开代码能提供什么

本课锁定
[`luka-group/mDPO@2221bb3`](https://github.com/luka-group/mDPO/tree/2221bb317a29dfe0f8d2c710cd8572e9d7ec101a)
（核验于 2026-07-23）。这个版本公开了：

- [`mdpo_trainer.py`](https://github.com/luka-group/mDPO/blob/2221bb317a29dfe0f8d2c710cd8572e9d7ec101a/mdpo_trainer.py)：
  三项 loss 的 trainer 入口；
- [`bunny/run_mdpo_bunny.py`](https://github.com/luka-group/mDPO/blob/2221bb317a29dfe0f8d2c710cd8572e9d7ec101a/bunny/run_mdpo_bunny.py)：
  Bunny 启动入口。

它不是完整的端到端 recipe。该提交的
[`README`](https://github.com/luka-group/mDPO/blob/2221bb317a29dfe0f8d2c710cd8572e9d7ec101a/README.md)
中 `Installation` 和 `Evaluation` 仍为 `TBD`，并明确说完整训练/评测代码仍在整理。更具体地说，公开 trainer 的第三路 forward 使用 `mask_visual_tokens=True`，变量名是 `imageless`；论文主实验写的是 `Crop 0–20%`。因此本课只能把公开代码当作 loss 和启动方式的参考，裁剪编译、环境安装、数据适配与评测由课程补全。

报告要准确写：

> 基于 mDPO 论文复现 C0 目标；参考公开 trainer 与 Bunny 启动入口；数据裁剪、环境和
> 评测流水线为本课程实现。

不能写"直接运行官方完整 recipe"或"官方代码已经提供端到端复现"。

### 13.2 C0 启动断言

启动前把 YAML 解析为 `resolved_config.yaml`，并检查：

- policy/reference 指向同一 step-0 SHA，reference 冻结；
- 每条记录都是 `track=C0` 且只有 image 输入；
- CoPO 的 `response_ref=chosen`，原图与 crop 的 token IDs、labels 和 response mask
  完全相同；
- crop 的输入 hash 等于原图 hash，面积比例不超过 0.20；
- 重放 crop 后输出 hash 与 manifest 一致；
- 数据中不存在 C1 counterfactual 字段；
- 所有配置键都被消费，未知键直接报错。

断言不能只写不测。故意修改一个 crop box、一个输出 hash，并给 C0 样本加入 `wrong_media` 字段，确认 launcher 都能打印 pair ID 后退出。

### 13.3 C1 配置必须另起文件

```yaml
experiment:
  id: exp16_c1_audio_wrong_segment_v1
  track: C1
  claim_scope: course_extension_not_original_mdpo_recipe
  frozen_c0_report: artifacts/exp16/c0_main/report.md
data:
  schema: data/exp16/c1/audio_wrong_segment/schema.json
  train: data/exp16/c1/audio_wrong_segment/train.jsonl
counterfactual:
  modality: audio
  kind: wrong_asset
  implementation: c1_counterfactual_v1
output:
  root: artifacts/exp16/c1_extensions/audio_wrong_segment_v1
```

C1 不得读取 `data/exp16/c0/train.jsonl` 后在运行时偷偷替换输入；它要有独立、可审计的 manifest。图像错图、图像模糊、音频 mask、视频帧打乱也分别使用不同配置。

## 14. 伪代码

下面只实现 C0。`prompt_original` 与 `prompt_crop` 的文本 token 完全相同，差别只在图像张量。crop 已在数据编译阶段生成，训练代码不得重新采样。

```python
original_image = load_and_verify(
    pair.prompt.assets[0],
    expected_sha256=pair.copo.positive_image_sha256,
)
crop_image = load_and_verify(
    pair.copo.negative_image,
    expected_sha256=pair.copo.negative_image.sha256,
)
assert pair.track == "C0"
assert pair.copo.response_ref == "chosen"
assert 0.0 < pair.copo.negative_image.transform.retained_area_ratio <= 0.20
assert pair.copo.negative_image.derived_from_sha256 == sha256(original_image)

prompt_original = encode_prompt(pair.prompt, image=original_image)
prompt_crop = encode_prompt(pair.prompt, image=crop_image)
assert_equal_text_tokens_labels_and_response_mask(
    prompt_original,
    prompt_crop,
    response=pair.chosen.text,
)

with no_grad():
    ref_chosen_original = sequence_logp(
        reference, prompt_original, pair.chosen.text, reduction="sum"
    )
    ref_rejected_original = sequence_logp(
        reference, prompt_original, pair.rejected.text, reduction="sum"
    )
    ref_chosen_crop = sequence_logp(
        reference, prompt_crop, pair.chosen.text, reduction="sum"
    )

pol_chosen_original = sequence_logp(
    policy, prompt_original, pair.chosen.text, reduction="sum"
)
pol_rejected_original = sequence_logp(
    policy, prompt_original, pair.rejected.text, reduction="sum"
)
pol_chosen_crop = sequence_logp(
    policy, prompt_crop, pair.chosen.text, reduction="sum"
)

response_logit = beta * (
    (pol_chosen_original - ref_chosen_original)
    - (pol_rejected_original - ref_rejected_original)
)
copo_logit = beta * (
    (pol_chosen_original - ref_chosen_original)
    - (pol_chosen_crop - ref_chosen_crop)
)
chosen_reward = beta * (pol_chosen_original - ref_chosen_original)

loss_response = -logsigmoid(response_logit).mean()
loss_copo = -logsigmoid(copo_logit).mean()
loss_ancpo = -logsigmoid(chosen_reward - 0.0).mean()

loss = loss_response + loss_copo + loss_ancpo
loss.backward()
```

batch size 为 2 时，三个 logit 都应是 `[2]`，三个 mean loss 都应是标量。逐项关闭 `loss_response`、`loss_copo` 和 `loss_ancpo`，确认梯度变化符合预期，并检查 reference 参数始终无梯度。

锁定的公开 trainer 用 `imageless` forward 计算第三路 log-prob；上面的 C0 伪代码把这一路显式改成论文的预生成 crop。最终实现必须在 `patches/c0_crop_forward.diff` 中说明这项差异。C1 的 `resolve_counterfactual()` 不得被 C0 trainer import 或调用。

## 15. 训练预算与 8 卡运行方案

先用 unit/smoke 验证目标，再运行 10K 图像 pair 的三臂三 seed。预算表给出规划区间，正式报告以作业日志累计 GPUh 为准。

| lane | 硬件 | 规模 | 预算 |
|---|---|---|---|
| unit | 1×24GB | 32 pairs | <1 GPUh |
| smoke | 1×24GB | 1k pairs、LoRA | 2–10 GPUh |
| C0 main | 8×24–80GB | 10k image pairs、三臂三 seed | 以实测吞吐估算 |
| C1 | 单独登记 | 每个 extension 独立 | 不计入 C0 |

reference forward 会显著增加算力和显存——C0 每个 pair 要跑六路 forward（policy 三路带梯度、reference 三路不带）。可采用：

- reference CPU/offload：省 HBM、慢；
- 预计算 reference log-prob：快，但 tokenizer、truncation、mask 必须冻结；
- policy/reference 共享 base 权重加 adapter：需确认实现不泄漏梯度。

无论采用哪种 reference 方案，都要冻结 tokenizer、image processor、truncation 和 mask，并在 100 个 pair 上复算原图与 crop 的 log-prob。缓存值与在线值在容差内一致，且 reference hash 不变，方案才通过。三臂公平性报告实际 GPU-hours，不能只报 steps。

## 16. 指标

### 偏好与 grounded 能力

这一组指标回答模型偏好是否与图像事实一致。grounded（有据）指回答里的事实能在原图中指认出证据。按错误类型和来源分别记录：

- pair accuracy；
- chosen-rejected margin；
- human grounded win rate；
- OCR、计数、属性与关系任务 exact/F1；
- hallucination/error-type rate；
- original-vs-crop chosen-reward gap。

每个汇总值都要保留 pair/case ID。pair accuracy 上升但规则任务正确率不升时，不计为 grounded 改善。

错图、模糊、音频和视频指标只进入对应 C1 报告。它们不能出现在 C0 主指标的分子或分母。

### 训练动力学

这一组指标用于发现过度优化、长度偏差和 likelihood displacement：

- chosen log-prob；
- rejected log-prob；
- KL proxy；
- loss 与 margin 分布；
- 输出长度、entropy；
- reference drift check。

画出同一累计 token 位置的三臂曲线，并同时显示 chosen 与 rejected。只展示 margin 会隐藏两者同时下降——第 6.1 节说过，损失只看差值，两条曲线一起沉时 margin 照样好看。

### 系统与回归

这一组指标记录成本，以及偏好训练对 SFT 能力和 Talker 的影响：

- pairs/s、tokens/s、GPU-hours；
- peak HBM；
- policy/reference forward 占比；
- 原 SFT text/image/audio/video；
- JSON/格式有效率；
- Talker 内容/音质回归，如 Talker 未冻结。

使用起点相同的冻结回归集与解码参数。系统吞吐以 non-padding token 计算，能力变化报告绝对差和置信区间。

### 统计口径

主指标是 held-out `modality_required=true` 图像 case 上的 grounded 正确率。规则可判定任务直接使用预注册 verifier；开放问答先由两名不知道模型身份的标注者判定，分歧再交给第三人。style judge 的分数不进入主指标。

主效应定义为：

$$
\Delta_{\mathrm{grounded}}
=
\operatorname{Accuracy}_{C0}
-
\operatorname{Accuracy}_{B}
$$

图像条件的次要指标先按 pair 判断原图是否优于 crop：

$$
\operatorname{ConditionAcc}_{k}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf 1\!\left[
r_k(x_i^+,y_{w,i})
>
r_k(x_i^-,y_{w,i})
\right],
\qquad k\in\{B,C0\}
$$

再定义 C0 相对 B 的提升：

$$
\Delta_{\mathrm{crop}}
=
\operatorname{ConditionAcc}_{C0}
-
\operatorname{ConditionAcc}_{B}
$$

这里的 `crop` 只指 C0 manifest 中预注册的 `Crop 0–20%`。统计单位是唯一的 `source_item_id`/原图 hash group。同一原图、裁剪图和多个问题先在组内求平均，不能当成独立样本。A/B/C0 对同一 case 和 seed 使用相同解码 seed，因此臂间差值按 case 配对。

使用 3 个训练 seed，并做 10,000 次分层配对 bootstrap（bootstrap：反复有放回抽样、重算指标，用抽样结果的波动估计置信区间的统计方法）：先对训练 seed 有放回抽样，再在错误类型和来源层内对唯一 source group 有放回抽样。主表报告绝对百分点差、95% percentile CI、每个 seed 的差值和样本数。`stats_plan.yaml` 必须在查看 test 前固定主指标、最小效应量、分层字段和脚本 SHA。

支持 C0 优于 B 的最小效应量预注册为：

- $\Delta_{\mathrm{grounded}}\ge 5$ 个百分点，且其 95% CI 下界高于 0；
- $\Delta_{\mathrm{crop}}\ge 10$ 个百分点，且其 95% CI 下界高于 0。

C1 必须重新定义自己的主指标和最小效应量。例如音频扩展不能沿用 `original-vs-crop`，也不能引用 C0 的显著性结论。

## 17. 验收条件

先判断 C0 运行是否有效。以下条件只检查协议和数据，不要求 C0 必须获胜：

- reference hash 在训练前后相同；
- test 原图及其 crop 与训练集无重叠；
- 3 个 seed 的配置、数据顺序和评测参数可追溯；
- 每条 CoPO pair 都通过原图/crop hash、面积比例和重放断言；
- CoPO 的问题、chosen token IDs、labels 与 response mask 在两路完全相同；
- C0 数据和 trainer 没有读取 C1 counterfactual；
- A/B/C0 的累计 token、可训练参数和 checkpoint 位置符合公平控制；
- grounded 主指标保留唯一 source group 级别的配对记录；
- `stats_plan.yaml` 在 test 前固定且实际脚本 SHA 一致；
- 80 个重点 case 已逐个检查。

只要这些协议条件全部满足，C0 就完成了。C0 与 B 没有显著差异，或 C0 更差，都是有效负结果；必须原样报告。

在运行有效的前提下，只有同时满足以下条件，才能写"在本课图像设置中，C0 优于普通 DPO"：

- C0 相对 B 的 grounded 正确率提高至少 5 个百分点，且配对 95% CI 下界高于 0；
- C0 相对 B 的 original-vs-crop condition accuracy 增加至少 10 个百分点，且配对
  95% CI 下界高于 0；
- 3 个 seed 的效应方向一致。

最后单独判断 C0 checkpoint 能否发布。除上面的正向结论外，还必须满足：

- chosen log-prob 不持续下降；
- response length 变化解释的 win 不超过一半；
- 任一 SFT 主能力回退不超过 3%；
- held-out grounded 正确率与 preference win 同向。

若偏好胜率上升而 grounded 正确率下降，结论写为"出现偏好投机"；这仍是有效的研究结果，但 C0 checkpoint 不得标记为改进版或发布为 `mdpo-c0-image-v1`。若协议条件失败，才把该 run 记为无效并重跑。

C1 的完成条件另写。C1 成功不改变 C0 的通过状态，也不能把发布名升级成笼统的"omni mDPO"。

## 18. 失败诊断表

调试顺序照旧：先查数据和量尺，再怀疑目标函数，最后才动模型。

| 症状 | 可能原因 | 证据 | 修复 |
|---|---|---|---|
| win 高但准确率不升 | judge/style hacking | 规则 verifier | 增 hard negative |
| 回答越来越长 | length bias | win-vs-length | 长度匹配 pairs |
| chosen/rejected 都降 | likelihood displacement | 两条 logp 曲线 | quality/NLL anchor |
| crop 图上的 chosen reward 也很高 | 没学会图像条件 | original-vs-crop gap | 检查 CoPO 数据与梯度 |
| 总是拒答 | 保守捷径 | answerable refusal | 加可回答 chosen |
| 格式好、事实错 | 格式成为 proxy | 去格式 judge | 事实 verifier |
| 训练 margin 爆增、test 不涨 | pair 记忆 | source split | 加强媒体级去重 |
| reference 变化 | 冻结失败 | hash/grad | 独立加载 reference |
| C0 不如 DPO | crop/anchor 或实现错误 | 三项 loss 与 reward | 32-pair 单测与 1k smoke |
| crop 每个 epoch 都不同 | collator 仍在随机裁剪 | 同 pair 的 image hash | 只读冻结 crop manifest |
| C0 日志出现 audio/video | 训练边界被破坏 | resolved config 与 batch schema | 停止运行，拆到 C1 |
| C1 错图实验很好 | 只说明扩展有效 | C1 独立报告 | 不回写 C0 结论 |
| 某 teacher 来源全胜 | teacher fingerprint | 按来源分层 | 混合/人工复核 |

## 19. 逐个样例检查

聚合指标不能说明模型读取了图像中的什么。每个 C0 case 同时保存原图、裁剪图、证据区域、三臂生成答案和三路 log-prob：

```yaml
pair_id: pref_img_000103
track: C0
original_image_sha256: sha256:...
crop:
  sha256: sha256:...
  crop_box_xywh_px: [811, 1042, 302, 190]
  retained_area_ratio: 0.143
  implementation: exp16_crop_v1
required_evidence: "region[0.61,0.78,0.84,0.86] reads 37.50"
chosen_supported: true
rejected_error: visual_digit_error
generated:
  A_original: "57.50"
  B_original: "57.50"
  C0_original: "37.50"
  C0_crop: "无法从裁剪区域确认"
logp:
  C0_chosen_original: -4.9
  C0_rejected_original: -7.3
  C0_chosen_crop: -8.1
implicit_reward:
  C0_chosen_original: 0.31
  C0_chosen_crop: -0.02
human_grounded_vote: [chosen, chosen, chosen]
failure_layer: null
```

按顺序审：

1. pair 本身是否有效；
2. 必要图像证据是否存在；
3. chosen 是否只是更长/格式更好；
4. crop 是否符合面积比例并能按 hash 重放；
5. A/B/C0 在原图上分别使用了哪些证据；
6. 原图换成预注册 crop 后，chosen reward 与生成答案如何变化；
7. 提升来自 response DPO、CoPO 还是 AncPO；
8. 原图正确率是否真的提高，而不只是 crop 上的概率下降；
9. 是否出现新幻觉或遗忘。

由第二名审阅者随机复核 20 个 case，并运行脚本检查原图/crop hash、答案文件和 checkpoint 是否存在。证据无法在原图定位的 case 不进入 grounded 统计。

## 20. 交付物

```text
artifacts/exp16/
├── c0_main/
│   ├── configs/{a_chosen_sft,b_dpo,c0_mdpo_image_crop}.yaml
│   ├── data/schema.json
│   ├── data/{train,dev,test,crop_manifest}.jsonl
│   ├── provenance/{assets.csv,annotators.md,generators.md}
│   ├── patches/c0_crop_forward.diff
│   ├── checkpoints/index.json
│   ├── metrics/{pair,generation,grounded,original_vs_crop,regression}.jsonl
│   ├── cases/audit.md
│   ├── judges/{rubrics,prompts,agreement}.json
│   ├── stats_plan.yaml
│   └── report.md
└── c1_extensions/
    └── <extension_id>/
        ├── config.yaml
        ├── data_manifest.jsonl
        ├── metrics.jsonl
        └── report.md
```

C0 报告必须分别给出：

- offline pair margin；
- fresh generation win；
- 可验证任务正确率；
- 人工 grounded win；
- original-vs-crop gap；
- 系统成本。

报告首页明确写 `A/B/C（C = C0）`，并单列"相对论文公开 recipe 的本地补全部分"。C1 报告不得复制 C0 标题后只换数据；它要写自己的 extension ID、模态、操作与结论边界。

## 21. 复现清单

- [ ] policy/reference 起点完全相同；
- [ ] reference 全程冻结且 hash 不变；
- [ ] 主表明确写 C = C0，未混入任何 C1 结果；
- [ ] A/B/C0 的 pair、token、参数与 seed 对齐；
- [ ] padding/prompt 不计 response log-prob；
- [ ] 图像 pair 经过人工可判定性审计；
- [ ] 每条 C0 CoPO pair 都是同一问题、同一 chosen、原图对论文 crop；
- [ ] crop 面积比例、box、seed、实现版本和输入/输出 hash 完整；
- [ ] crop 可重放，训练时没有再次随机采样；
- [ ] C0 schema 拒绝 wrong-media、其他 degradation、audio 与 video 字段；
- [ ] 原图/source group split 完成；
- [ ] test 未用于生成 rejected；
- [ ] $\beta=0.1$、三项等权、$a=0$ 与 sequence-sum 在 test 前冻结；
- [ ] chosen/rejected/KL 曲线都保存；
- [ ] fresh generation 而非只算 pair；
- [ ] original-vs-crop 已评测；
- [ ] C0−B grounded 正确率与 crop gap 已按预注册配对 bootstrap 报告；
- [ ] 输出长度、拒答、格式偏差已控制；
- [ ] SFT 回归已完成；
- [ ] case 与 license/provenance 可追；
- [ ] README 的 Installation/Evaluation `TBD` 与公开代码边界已在报告说明；
- [ ] 若运行 C1，使用独立 extension ID、schema、配置、目录和报告。

## 22. 前沿对照与改造方向

四个参照点，都来自本课或前面课程已引用的材料。其一，偏好优化在前沿流水线里的位置：[Nemotron Omni 官方 recipe](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/omni3/README.md) 的后训练顺序是 SFT、MPO、Text GRPO、Vision GRPO——偏好优化排在 SFT 之后、强化学习之前，当的是"便宜的中间站"：离线数据、无在线采样，先把明显的坏习惯修掉，再上昂贵的 RL。本课在第五幕里的位置（第 15 课 SFT、本课偏好、第 17 课 GRPO）就是这条工业顺序的缩小版。其二，偏好数据怎么来：本课 10K pair 靠人工与可追踪生成器；[MPO / MMPR](https://arxiv.org/abs/2411.10442) 走自动化流水线——有标准答案的任务让模型自己采样多份回答，按对错切成 chosen/rejected；没有标准答案的任务把一份好回答截断，让模型续写补完，续写段容易出幻觉，拼回去当 rejected（论文称 Dropout NTP）。数据规模因此远超人工路线，确切数字以论文为准。其三，损失怎么改：MPO 在 DPO 的相对差之外混入 quality 目标（单独要求 chosen 的分数为正、rejected 为负）和 generation 目标（chosen 上的普通下一 token 预测损失），后者直接托住 chosen 的绝对概率——和本课 AncPO 治的是同一个病（likelihood displacement），药方不同。其四，偏好优化不只用于文本回答：[Qwen2.5-Omni 技术报告](https://arxiv.org/abs/2503.20215)（第 01 课引过）在语音生成侧同样用了 DPO，对 Talker 做偏好训练来提升长句语音的稳定性、抑制吐词错误；其偏好数据构造细节报告未完全公开。这说明"好坏成对、reference 拴绳"这套机制对 codec token 序列同样成立，正是本课 C1 语音扩展的方向。

规模问题（钱能解决的）：policy 只有 26M，偏好数据只有 10K 图像 pair，前沿系统两者都大若干个量级；基础模型能力弱，推理类任务上偏好训练的收益可能直接被地板效应吃掉。机制问题（本课教的东西能解决的）：无条件偏好与 likelihood displacement 都是结构性病，与参数量无关——CoPO 的条件对照、AncPO 与 quality/generation 的托底，在 26M 上和在几十亿上是同一套动作；偏好数据构造策略（rejected 从哪来、噪声怎么审）是方法问题，本课的可判定性审计与 hash 重放协议在任何规模都成立。真正的机制缺口有两个：C0 只优化文本回答，语音输出侧的偏好完全没动；数据是纯离线的，rejected 不来自模型自己的错误分布。下面的改造清单就冲这些缺口去。



1. **损失变体：实现 `exp16x_mpo` 混合目标。** 把第 14 节伪代码的三行 loss 换成 MPO 组合：保留 response DPO，加 quality 目标（chosen 隐式 reward 过 sigmoid 后要求为正、rejected 为负）与 generation 目标（chosen 上的 NTP loss，复用第 15 课的 SFT loss 路径）；配置在第 13 节 `objective` 块新增 quality/generation 权重字段。数据、起点、LoRA 与 B 臂完全对齐，只换目标。预算：先按 smoke 档 2–10 GPUh 在 1k pair 上走通，再跑 10k 单 seed，量级与 C0 单臂相当。预期：chosen log-prob 曲线不再单调下降（generation 项托底），margin 仍上升，SFT 回归优于 B。失败判定：chosen log-prob 仍持续下降，或任一 SFT 主能力回退超过 3%——先查 generation 项的权重与 mask 是否真的生效。结果只与 B 比，按第 2 节的命名纪律，不回写 C0 主表。
2. **偏好数据构造策略：自采样对错切分（MMPR correctness 流水线缩小版）。** 只用第 8 节"规则可验证任务"（OCR、计数、坐标）的 prompt，从 `exp16_start` policy 带温度采样 8 份回答，规则 verifier 判对错：对的当 chosen、错的当 rejected，凑一份 10k 的 selfgen pair 集；schema 沿用 `exp16.c0.v1`，`generator` 字段写明自采样实现与 seed。训练一个 B'（DPO on selfgen），与 B 等 token、等超参对比。预算：26M 模型采样便宜，8 万份短回答的生成加判分在单卡数 GPUh 内，训练预算同 B。预期：等 pair 数下 B' 的 fresh-generation 正确率增益不低于 B——rejected 来自模型自己的真实错误分布，修正信号更贴身。失败判定：B' 不优于 B 且 Step 2 式可判定性审计 agreement 低于 0.8，说明 verifier 切分噪声太大，先修 verifier 再谈结论。
3. **损失变体：给裁剪图加负向 anchor。** C0 按论文默认只给 chosen 加正向 anchor（$a=0$）。另起 `c0_ablation_neg_anchor`：在第 14 节伪代码 `loss_ancpo` 旁加一项 $-\log\sigma(-r_\theta(x^-,y_w))$，要求裁剪图条件下的 chosen reward 为负——除了"原图下更愿意说"，还要"裁剪图下明确不愿意说"。预算：1k smoke 加 10k 单 seed，LoRA，8×24GB。预期：original-vs-crop reward gap 与 ConditionAcc 进一步拉大，grounded 正确率不降。失败判定：原图侧 chosen reward 被连带压低（displacement 顺着共享参数传染），或拒答率明显上升——负向 anchor 过强，降权重或放弃。
4. **把偏好搬到语音输出：`c1_audio_talker_v1`（Qwen2.5-Omni 方向）。** 解冻 Talker（改第 13 节 `model.frozen` 列表），对同一段文本回答让 Talker 采样两份语音，用第 01 课的固定 ASR 协议算 WER：低 WER 为 chosen、高 WER 为 rejected；DPO 的 sequence log-prob 换成 8 路 codebook 的逐路求和，reference 是冻结的 step-0 Talker。按第 2 节纪律登记为 C1，内容偏好与声学偏好分开，不与 C0 混表。预算：1k 语音 pair，2–10 GPUh。预期：held-out 长句 WER 下降，重复与漏词 case 减少，文本能力零变化（Thinker 冻结）。失败判定：WER 不降，或 CAM++ speaker similarity 跌破第 01 课回归线——偏好信号压到音色头上去了，先查 pair 里两份语音的音色是否可比。

三条论文结论到缩小版的映射：

- DPO 论文"margin 上升不保证 chosen 概率上升"：对应 Stage 1 的 1k smoke，同时画 chosen/rejected log-prob 曲线（第 16 节训练动力学组）。预期方向可复现——这是损失函数的结构性质，与规模无关；若两条曲线都在涨，多半是 pair 太容易，rejected 单边下坠要在更难的数据上才出现。
- mDPO 的诊断"普通 DPO 学到的偏好可以不依赖图像"：对应 B 与 C0 的 ConditionAcc 差（$\Delta_{\mathrm{crop}}$，预注册 ≥10 个百分点）。预期方向可复现——无条件偏好是 DPO 目标自身的盲区，26M 上同样存在；若 B 的 ConditionAcc 本来就高，先怀疑 pair 里语言差异太小、模型被迫看图，回 Step 2 查审计记录。
- mDPO 主结论"CoPO+AncPO 提升 grounded 表现"：对应 $\Delta_{\mathrm{grounded}}$（预注册 ≥5 个百分点）。预期方向可能复现但不保证——26M 的基础视觉能力弱，两臂可能都贴着地板，出现"reward gap 拉开了、生成正确率不动"。这正是第 17 节说的偏好投机形态之一，是有效负结果，按验收条件原样报告。

## 23. 原始论文与官方 recipe 精读

### [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)

精读：RLHF 到 DPO 的推导、理论部分、实验中的 β 与 reference 设置。

阅读任务：亲手推一遍 DPO 隐含的 reward parameterization——从"KL 约束下最大化 reward"的最优解反解出 reward，看奖励模型是怎么被消掉的；解释 chosen likelihood 下降的条件，对照你自己 1k smoke 里的两条 log-prob 曲线；列出离线 pair 分布偏差，想清楚它与第 17 课在线采样的关系。把论文符号逐项映射到第 14 节伪代码，一个变量都不能落空。

### [mDPO](https://arxiv.org/abs/2406.11839)

精读：第 2.2 节 unconditional preference、第 3.1 节 CoPO、第 3.2 节 AncPO，以及表 2–4 的消融。

阅读后必须能回答，且每个答案能在本课产物里指认：

1. response DPO 与 CoPO 分别改变哪个变量（对照 6.3 节的三行对照表）；
2. 为什么 CoPO 必须使用同一个 chosen 回答（对照 13.2 节的 token 一致断言）；
3. 为什么论文默认选择 `Crop 0–20%`，而随机图像只是消融；
4. AncPO 的 anchor 加在哪个 reward 上；
5. 表 3 的结果为什么不能证明错图、音频或视频也有效（这正是 C0/C1 边界的依据）。

随后对照锁定仓库：列出论文 crop 与公开 trainer `imageless` forward 的差异，以及 README 中尚未公开的安装和评测步骤。完成这张差异表后才开始写 C0 代码。

### [MPO / MMPR](https://arxiv.org/abs/2411.10442)

精读：MMPR 数据构造、Mixed Preference Optimization 公式、quality/preference/generation 目标、消融。

阅读任务：说明三个目标分别处理的失败类型，核对 MMPR 实际公开的数据字段，并判断能否在相同 pair 上与 DPO 公平比较。不可获得的字段标记为 `unknown`。MPO 只能进入 `exp16x_mpo`，不能改写 C0 公式。

### [Nemotron Omni 官方 recipe](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/omni3/README.md)

精读：SFT→MPO→Text GRPO→Vision GRPO 的顺序、公开配置和数据说明。

阅读任务：标出 MPO 在整条 post-training 中的位置，并列出未公开的内部数据和过滤字段。复现声明只覆盖有论文或配置证据的部分。

### 起点实现

- [MiniMind-O 官方仓库](https://github.com/jingyaogong/minimind-o)：确认 Thinker/Talker loss 与可训练模块；
- [OpenOmni 官方仓库](https://github.com/RainBowLuoCS/OpenOmni)：查看公开 DPO 路径和数据接口。

代码任务：分别导出两套实现的 response mask、image condition 和 reference 构造，用一条相同图像 pair 比较 log-prob。差异必须在迁移说明中逐项记录。C0 只接 vision connector；不要为了"以后会做 Omni"提前把 audio/video 分支接进 trainer。

## 24. 扩展题

下面每一项都另起实验，不改 C0 主结果：

1. `c0_ablation_length_norm`：比较 sequence-sum 与 length-normalized log-prob；
2. `c0_ablation_crop_range`：复现论文不同 crop 范围和随机图像消融；
3. `c1_image_wrong_asset_v1`：把错图作为 CoPO negative；
4. `c1_image_blur_v1`：使用确定性模糊；单独证明它比原图信息更少；
5. `c1_audio_mask_span_v1`：音频内容 preference；
6. `c1_audio_talker_v1`：内容与声学 preference 分两阶段训练（第 22 节改造清单第 4 项给了具体做法）；
7. `c1_video_frame_order_v1`：视频时间顺序反事实；
8. 用 active sampling 选择 policy 最不确定的 pairs；
9. 分离 outcome preference 与 rationale preference；
10. 训练可审计的小型 grounded reward model，与 DPO 比较；
11. 把最严重的 preference hacking case 迁移成[第 17 课](17_grpo_rlvr.md)的 verifier；
12. 另起 `exp16x_mpo`：实现 MMPR 的 preference/quality/generation 目标，结果单独与 B
    比较，不回写 C0（第 22 节改造清单第 1 项给了具体做法）。

## 25. `mdpo-c0-image-v1` 的发布内容

只有同时满足两项 C0−B 正向结论和 checkpoint 发布条件，才能标记为 `mdpo-c0-image-v1`。

必须附带：

- reference checkpoint hash；
- pair 数据版本；
- crop manifest 与 `exp16_crop_v1` 实现 SHA；
- C0 三项目标公式和固定权重；
- grounded/original-vs-crop 指标；
- SFT 回归；
- 明确失败类型；
- 论文、公开 trainer 与本地补全的差异表。

只有在 fresh generation 和 grounded verifier 同时提升时，才把它作为第 17 课的 RL 起点。

C1 checkpoint 使用 `mdpo-c1-<modality>-<counterfactual>-v1`，例如 `mdpo-c1-audio-mask-span-v1`。这个名字只表示课程扩展，不能发布成 `mdpo-c0-image-v1`，也不能标成论文官方 checkpoint。

收个尾。到这一课为止，系统第一次见过"错误示范"：它知道两份回答里该挑哪份，也被逼着在挑之前看一眼图；你手里多了 reference 管理、隐式 reward 观测、偏好投机审计这三样工具。但离线偏好有天花板——好答案坏答案都得预先造好，模型自己犯的新错误进不了训练信号。下一课[第 17 课](17_grpo_rlvr.md)把这层天花板拆掉：让当前模型对同一道题现场生成一组回答，用可执行的验证程序打分，边生成、边评分、边更新。本课这套 reference、KL 观测和投机审计的手艺，原样带过去。
