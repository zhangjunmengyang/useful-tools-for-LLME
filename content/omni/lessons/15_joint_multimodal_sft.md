---
id: 15_joint_multimodal_sft
title: "现代 Joint SFT"
summary: "训练 token 一样多时，“先预热连接器、再混着练、最后回放旧任务”这套三段式，真比按顺序一个个练或一股脑混着练拿到更好的综合能力、更少的遗忘吗？"
unit: alignment
play_tools: []
checkpoints:
  - "把文本、图像、音频理解和语音输出的样本字段、loss 算法统一成一套约定。"
  - "分清三种配比方式：sample-balanced（按样本数配平）、token-balanced（按 token 数配平）和 temperature sampling（温度采样调冷热门比例）。"
  - "测出模态之间的梯度冲突、实际 token 占比，以及 capability replay（旧任务回放补课）修复了多少。"
  - "产出能接着做偏好优化或 RL 的 joint-sft-v1，附一张分模态能力对照表。"
---

# 第 15 课：多模态联合 SFT

> 本课产出：训练一份覆盖文本、图像、音频理解与语音生成的联合 SFT checkpoint，并用实际 token 比例、逐任务回归和反事实媒体测试判断它是否合格。视频只在锁定 frontend 后作为扩展。

## 1. 把多种模态放进一次 SFT

第四幕（11-14 课）刚落幕：MoE 稀疏专家装了、Mamba 混合骨干比过了、8 卡分布式训练跑通了、长上下文课程也做完了。硬件和结构都换好了；现在这台机器的"身体"不再是瓶颈。瓶颈换成了教法：到目前为止，我们的训练基本还是官方 mini recipe 那套按任务分阶段的顺序训练，先教一样、再教下一样。第五幕（15-17 课）整幕只干一件事：升级训练方法。本课联合 SFT，下一课偏好优化，再下一课 GRPO 可验证奖励强化学习。

一个 omni 模型要同时会四样活：读文字、看图、听音频、开口说话。分开教很容易，麻烦在"同时"：把四种任务的数据倒进同一个训练阶段、交错采样、更新同一组共享参数，这就是 joint SFT（联合监督微调）。而只把几个 JSONL 文件合并起来喂进去，得到的往往是一台什么都会一点、什么都不精的机器；甚至更糟：学会了说话，忘了怎么读图。

**为什么会互相拆台。** 这是本课最重要的直觉：**多任务共享参数就像几个人坐一块跷跷板**。音频任务坐上去，文本那头就翘起来。原因有三层。第一层是数据量：一条 20 秒音频展开成的 codec token 可能是一条短问答文本 token 的几十倍，"每类任务抽同样多样本"实际等于让音频霸占绝大多数训练 token。第二层是梯度：不同任务对同一个共享参数想要的更新方向可能相反，你往东我往西，合力谁也不满意。第三层是分布：新任务改变了输出格式和风格分布，旧能力的输出习惯被冲掉，这叫 catastrophic forgetting（灾难性遗忘）。类比失效处要说清：跷跷板暗示总量守恒、一头涨一头必跌，但多任务学习不是零和；共享语言空间里图文和音频任务可以互相帮忙（正迁移），前提是配比、损失和课程设计得对。本课要造的就是这套设计，并用实验证明它比"随便混"强在哪。

第 16 课的偏好优化、第 17 课的 GRPO 都要从一个各模态能力齐全且可复现的 SFT checkpoint 出发。没有本课的 `joint-sft-v1`，post-training 实验的起点五花八门，结果彼此没法比。

做完这课，你手里有三臂对照实验（顺序训练、纯联合、warmup-联合-回放）的完整结果，一份能继续做 DPO/GRPO 的 `joint-sft-v1`，和一张分模态能力对照表；每项能力的起点分、训练后分、回退阈值和 GPU 成本都在上面，平均分再好看也藏不住某一项的崩塌。

本课术语：

| 术语 | 简要解释 |
|---|---|
| joint SFT | 多种任务在同一训练阶段交错采样、更新同一组共享参数的监督微调 |
| 任务桶（bucket） | 按输入输出模态给样本分的类，是采样和回归报告的统计单位 |
| token-balanced | 按"实际消耗的训练 token"配平各任务份额，而非按样本条数 |
| catastrophic forgetting | 学了新任务，旧能力显著下降 |
| negative transfer | 一起训反而比单独训差：共享参数被别的任务带偏 |
| capability replay | 训练末段回放预留的旧任务数据，把被冲掉的能力补回来 |
| 梯度余弦 | 两个任务在共享参数上梯度方向的夹角余弦，负值提示可能在冲突 |
| 反事实媒体测试 | 删掉/换错图或音频再问同一问题，检查模型是真看了媒体还是靠语言先验蒙 |
| LoRA | 低秩适配：冻住原权重，只训小的低秩增量矩阵，省显存且好控制变量 |
| anchor | 训练中一直保留的旧任务数据配额，防止模型只顾新任务 |

## 2. 本课要解决的问题

Joint SFT 指在同一训练阶段交错采样多种任务，并更新一组共享参数。仅合并 JSONL 文件不会自动形成有效课程；采样单位、损失范围、序列 contract（序列布局约定：哪个位置放什么 token）和遗忘控制都会改变优化结果。

本课可以独立从 MiniMind-O 基线开始；完成[第 14 课](14_long_context_curriculum.md)不是前置条件。默认主矩阵覆盖文本、图像、音频理解与语音输出，不暗中依赖[第 09 课](09_native_video.md)的视频 frontend（前端编码器：把视频帧变成 token 的模块）。若使用长上下文 checkpoint，必须把它当成固定起点，三臂都用同一个 SHA。

本课结束时要交付可继续用于 DPO/MPO 或 GRPO 的 `joint-sft-v1`，以及一张分模态能力对照表。后者逐项记录起点、训练后分数、回退阈值与 GPU 成本，用于阻止平均分掩盖单项退化。

## 3. 要验证的结论与失败条件

**研究问题：** 在相同训练 token 下，`输入对齐 warmup → joint interleaved SFT → capability replay` 是否比纯 sequential 或纯 joint 训练获得更高的跨模态平均能力，并把任一既有能力回退控制在 3% 内？

**预期结果：**

- warmup 先增加图像和音频理解数据的密度，用于稳定模态到语言空间的映射；
- token-balanced joint sampling 防止音频 token 与语音 codec token 主导默认优化；启用 video extension 时，同样约束视频 token；
- 末段 capability replay 能修复文本与基础单模态遗忘；
- 默认联合课程中的 image-text、audio-text 与 speech 任务能通过共享语言空间产生单任务训练没有的迁移；video/AV 组合泛化仅在锁定扩展中检验。

**拒绝标准：** sequential-only 在等 token、等起点、等可训练参数下获得同等或更高的综合分数，并且回归更低，则 joint 课程没有获得支持。如果老老实实按顺序教反而更好，就承认这个结果，别硬给联合课程找补。

## 4. 固定训练起点

```text
checkpoints/exp15_start/
├── model.safetensors
├── config.json
├── tokenizer/
├── vision_connector/
├── audio_connector/
├── talker/                 # 默认主实验包含语音输出，必须提供
└── baseline_metrics.json
```

起点要求：

- 文本、单图和短音频至少能推理；
- modality sentinel（模态哨兵 token：在文本序列里占位、标记"这里插入媒体特征"的特殊 token）、token type、loss mask 定义明确；
- 默认训练能分别记录 text/image/audio/audio-code token 数；启用 video extension 后再增加 video token 统计；
- 能对 Thinker 文本 loss 和 Talker codec loss 分开反向；
- 起点 checkpoint 与数据 manifest 都有 hash。

若已有第 01 课产物可以使用，但不是前置条件。视频只能通过三臂共用且冻结的 frame/tubelet frontend 扩展接入；不要在本课同时比较视频架构——那是两个变量，一课只动一个。

独立执行时固定：

| 锁定项 | 固定值 |
|---|---|
| 模型 | `jingyaogong/minimind-3o` |
| 模型 revision | `ee3febbd08cc5b2bd41c039c825a8934232fee33` |
| 代码版本 | `jingyaogong/minimind-o@a10fa6c148ed274d66f96dc119689e93e01be823` |
| 主实验输入模态 | `text`、`image`、`audio` |
| 主实验输出 | `text`、`mimi_audio` |

上面的官方 revision 就是本课自己的 `baseline-v1`；下载后把所有文件 SHA256 写入 `checkpoints/exp15_start/provenance.json`。若要加入视频，必须提供一份三臂共用的 `video_frontend.lock.json`，内含 encoder code commit、checkpoint SHA、frame/tubelet 配置、输出 tensor shape 和冻结状态；否则视频与 AV 桶权重必须为 0。

## 5. 学完后应能完成

完成后你应能：

- 定义覆盖文本、图像、音频理解和语音输出的统一 sample schema，并能在显式 lock 下扩展到视频；
- 解释 sample-balanced、token-balanced 和 temperature sampling；
- 正确构造 assistant-only、modality reconstruction 和 codec 多码本 loss mask；
- 测量模态梯度冲突而不把它误当成因果证明；
- 设计 anchor/replay 来诊断 catastrophic forgetting；
- 区分单模态覆盖增加、表征改善和跨模态推理增强；
- 给每一项能力提升标注数据、训练 token 和 GPU 成本。

## 6. 原理:边造边讲

四个机制，每个按同一节奏走：为什么需要（直觉）、怎么运转（机制）、精确定义（数学）、怎么验证。它们合起来回答一个问题：跷跷板为什么会翘，怎么把它压平。

### 6.1 Instruction tuning：只对答案算账

一条图文对话序列里混着用户问题、图像占位符、助手回答和 padding。要学的只有"给定前文接出助手回答"这一件事。第 01 课在纯语音场景讲过 loss mask（忘了回[第 01 课](01_baseline_reproduction.md)）；多模态下它更容易出错，因为媒体占位符也混在序列里，一不小心就把不该算的位置算了进去。

SFT（supervised fine-tuning，监督微调）使用 teacher forcing：计算第 $t$ 个目标 token 时，条件中包含真实的先前目标 $y_{<t}$。

文本目标为：

$$
\mathcal L_{\text{text}}
=
-\frac{1}{\sum_t m_t}
\sum_t m_t
\log p_\theta\!\left(y_t\mid x,y_{<t}\right)
$$

其中 $m_t\in\{0,1\}$ 是 loss mask。

请拿一条图文对话逐 token 标出 `input_ids`、`labels` 和 $m_t$；用户 prompt、媒体占位符和 padding 默认只提供条件，不计 assistant text loss。验收时，所有 $m_t=0$ 的位置都应对应 `labels=-100`，且手算 loss 与 trainer 输出一致。手算这一步别省——mask 错了，loss 照样下降，模型只是在优化错误的位置。

### 6.2 多目标损失：三笔账合成一笔，谁在真正拉动更新

同一 batch 可能同时产生 Thinker 文本 loss、Talker codec loss 和 connector 辅助 loss。三笔账加权求和成一笔总账，但权重写多少和实际影响多大是两回事——就像三个人拔河，绳子往哪边走不只看谁名义上力气大，还看谁真的在场、出手多频繁。

若保留 Talker，总目标为：

$$
\mathcal L
=
\lambda_{\text{text}}\mathcal L_{\text{text}}
+\lambda_{\text{codec}}\sum_q w_q\mathcal L_{\text{codec},q}
+\lambda_{\text{aux}}\mathcal L_{\text{connector}}
$$

权重 $\lambda$ 的数值不能直接表示一个目标对更新的影响；还要看梯度范数、该目标出现的频率和参与 loss 的 token 数。$\lambda_{\text{codec}}$ 名义上不大，但 8 路码本的有效 token 数是文本的许多倍时，codec 项照样主导共享层的更新——这就是跷跷板的损失侧成因。

请在固定 probe batch（探针批次：固定不变、专门用来体检的一小批数据）上分别反向三项 loss，记录共享层梯度范数与夹角。若某一项贡献持续高出一个数量级，先检查 mask 和 token 计数，再调整权重。顺序别反：权重是最后一招，多数"失衡"其实是 mask 或计数 bug。

### 6.3 采样单位：抽样本，还是配 token

采样器抽取的是样本，但优化器消耗的是 token。一条 20 秒音频产生的 token 可能远多于一个短文本问答，因此相同样本数不代表相同训练份额。这是跷跷板的数据侧成因，也是最常被忽略的一个：很多"音频吃掉了文本能力"的事故，根源只是按条数抽样。

三种配平方式：

- sample-balanced：每种任务抽相同样本数；
- token-balanced：每种任务消耗相近训练 token；
- temperature sampling：平滑数据集规模差异（温度越高越接近均匀分布，越低越接近按数据量比例）。

先用 10k 条模拟抽样手算并实测三种采样器，再选择主实验方案。报告同时展示样本抽中比例和实际进入模型的 non-padding token 比例；两者之和都应为 100%，且 token 比例与 recipe 的绝对误差不超过预注册阈值。

### 6.4 遗忘与负迁移：跌下去的能力，先问清是谁推的

catastrophic forgetting（灾难性遗忘）指学习新任务后已有能力显著下降；negative transfer（负迁移）指共享训练使某项任务低于独立训练结果。两者症状相似、成因不同：遗忘是时间维度的（后学的冲掉先学的），负迁移是共享维度的（一起学互相拖累）。诊断时如果混为一谈，开出的药方多半是错的。

增加新模态时，下降可能来自：

- 数据比例改变；
- 参数共享导致梯度冲突；
- 输出格式分布改变；
- tokenizer/position contract 改变；
- 学习率过高；
- 合成数据风格污染。

六个嫌疑人里只有第二个是"任务本质冲突"，其余五个都是工程问题。这也是为什么本课坚持先跑反事实和逐桶回归、再谈干预：跷跷板翘了，先查是不是有人把秤砣放错了地方。

请在训练前冻结逐能力回归集，并在每个 checkpoint 原样回放。任一主能力低于起点 97% 时，即使 macro average 上升，也要标记为回归失败并定位首个退化阶段。

## 7. 统一数据字段

统一 schema 的作用是让采样、collator（整理器：把一批样本拼成训练 tensor 的代码）、loss mask 和 provenance（来源记录）使用同一份事实来源。每条记录为 JSONL。默认 loader 只接受 `text`、`image`、`audio` 输入以及 text/audio-code 目标；下面这条 image+text 记录是默认主课程的完整可运行样例，属于官方 loader 已支持的 `image_text` 桶：

```json
{
  "id": "sft_image_000042",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "asset_id": "i0"},
        {"type": "text", "text": "图片中的主要乐器是什么？给出可见依据。"}
      ]
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "主要乐器是原声吉他；可见木质共鸣箱、圆形音孔和六根琴弦。"}]
    }
  ],
  "assets": [
    {"asset_id": "i0", "uri": "sha256://...", "type": "image", "width": 1024, "height": 768}
  ],
  "targets": {
    "text": true,
    "audio_codes": false,
    "evidence": {
      "image_regions": [{"asset_id": "i0", "bbox_xyxy": [118, 164, 902, 741]}]
    }
  },
  "bucket": "image_text",
  "task": "image_instrument_identification",
  "modalities": ["image", "text"],
  "quality_score": 0.94,
  "source": "dataset-or-generator-id",
  "license": "explicit-license-id",
  "generation": {"teacher": null, "prompt_hash": null},
  "split": "train",
  "loss_policy": "assistant_text_only"
}
```

先把该样例保存为单行 JSONL，再运行 schema adapter。写入训练 Parquet 前，adapter 必须执行以下确定性映射：

| 原始字段 | adapter 写入的训练字段 |
|---|---|
| `messages` | `conversations`；图像内容写成 `<image>`，音频 token 由 loader 注入 |
| `assets[type=image]` | `image_bytes[0]` |
| `assets[type=audio]` | 最后一个 user 轮次的 `question_audios` |
| assistant 的 `audio_codes` | 对应 assistant 轮次的 `answer_audios` |
| asset registry 中的 SHA/URI | 先校验 hash，再解析为 bytes |

锁定 commit 的 `OmniDataset` 会读取首个 `image_bytes`，所以该样例不需要视频 frontend 或额外 loader patch。默认 pilot 限制为每条最多一张图，audio 桶限制为每个 user 轮次一段音频。若要支持多图、同轮多音频或单条 image+audio 强依赖任务，必须先修正 sentinel 注入策略，固定 code SHA 并新增对齐测试，不能只改 schema。

默认运行前至少为 `text_general`、`image_text`、`audio_text`、`text_audio` 和负控制各做一条同 schema 的 loader/collator 测试。测试要导出 token 序列、资产对应范围和 loss mask，并与人工标注逐项一致。若某条不包含某种资产，则相应 `content`、`assets` 和 `modalities` 项直接省略，不得填虚假空媒体。

### 7.1 三条可训练语音 fixture

语音输出样本不能只写一句 assistant 文本。MiniMind-O 训练还需要目标 Mimi codes、speaker embedding 和可选的 reference codes。下面三条 fixture（固定测试样例）专门用于 loader、collator 和 128-sample overfit 测试；正式指标仍使用 held-out 数据。

sidecar（伴随文件：和 JSONL 记录配套存放的二进制数据）使用固定二进制格式：

| 文件 | 内容与用途 |
|---|---|
| `answer.wav` | PCM waveform；只用于听检和重新编码校验 |
| `answer_codes.i32le` | shape 为 `[F,8]`，按 frame-major 保存；每帧依次为 `q0...q7` |
| `spk_emb.f32le` | shape 为 `[192]` |
| `ref_codes.i32le` | shape 为 `[R,8]`，按 frame-major 保存；每帧依次为 `q0...q7` |
| `input.wav` | 仅 `audio_audio` 和 `negative_control` 使用 |

fixture builder 先用锁定的 Mimi encoder 把 `answer.wav` 编成 codes，再计算每个文件的 SHA-256，并拒绝未替换的 hash 占位符。训练直接读取 `answer_codes.i32le`；`answer.wav` 用于听检、重新编码校验和内容对齐，不是第二份 loss target。`spk_emb` 与 `ref_codes` 必须来自该 answer speaker；speaker ID 或任一 value hash 不一致时拒绝样本，不能从另一条语音随意补齐。

#### `text_audio`

```json
{
  "id": "fixture_t2a_0001",
  "messages": [
    {
      "role": "user",
      "content": [{"type": "text", "text": "请朗读：系统检查完成。"}]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "系统检查完成。"},
        {"type": "audio_codes", "sidecar": "fixture_t2a_0001/answer_codes.i32le"}
      ]
    }
  ],
  "assets": [],
  "speech_target": {
    "answer_audio": {
      "path": "fixture_t2a_0001/answer.wav",
      "sha256": "REQUIRED_64_HEX",
      "sample_rate": 24000
    },
    "answer_codes": {
      "path": "fixture_t2a_0001/answer_codes.i32le",
      "sha256": "REQUIRED_64_HEX",
      "dtype": "int32_le",
      "shape": ["F", 8],
      "layout": "frame_major_q0_to_q7"
    }
  },
  "voice_condition": {
    "spk_emb": {
      "path": "voice/v0.spk.f32le",
      "sha256": "REQUIRED_64_HEX",
      "dtype": "float32_le",
      "shape": [192]
    },
    "ref_codes": {
      "path": "voice/v0.ref.i32le",
      "sha256": "REQUIRED_64_HEX",
      "dtype": "int32_le",
      "shape": ["R", 8],
      "layout": "frame_major_q0_to_q7"
    }
  },
  "targets": {"text": true, "audio_codes": true},
  "bucket": "text_audio",
  "task": "read_text",
  "modalities": ["text"],
  "source": "exp15_debug_fixture_v1",
  "source_group": "fixture_t2a_0001",
  "license": "fixture-only",
  "split": "debug",
  "loss_policy": "assistant_text_and_audio"
}
```

#### `audio_audio`

```json
{
  "id": "fixture_a2a_0001",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "audio", "asset_id": "q0"},
        {"type": "text", "text": "请简短回答录音中的问题。"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "今天是星期二。"},
        {"type": "audio_codes", "sidecar": "fixture_a2a_0001/answer_codes.i32le"}
      ]
    }
  ],
  "assets": [
    {
      "asset_id": "q0",
      "uri": "sha256://REQUIRED_64_HEX",
      "path": "fixture_a2a_0001/input.wav",
      "type": "audio",
      "sample_rate": 16000,
      "transcript": "今天星期几？"
    }
  ],
  "speech_target": {
    "answer_audio": {
      "path": "fixture_a2a_0001/answer.wav",
      "sha256": "REQUIRED_64_HEX",
      "sample_rate": 24000
    },
    "answer_codes": {
      "path": "fixture_a2a_0001/answer_codes.i32le",
      "sha256": "REQUIRED_64_HEX",
      "dtype": "int32_le",
      "shape": ["F", 8],
      "layout": "frame_major_q0_to_q7"
    }
  },
  "voice_condition": {
    "spk_emb": {
      "path": "voice/v0.spk.f32le",
      "sha256": "REQUIRED_64_HEX",
      "dtype": "float32_le",
      "shape": [192]
    },
    "ref_codes": {
      "path": "voice/v0.ref.i32le",
      "sha256": "REQUIRED_64_HEX",
      "dtype": "int32_le",
      "shape": ["R", 8],
      "layout": "frame_major_q0_to_q7"
    }
  },
  "targets": {"text": true, "audio_codes": true},
  "bucket": "audio_audio",
  "task": "spoken_question_spoken_answer",
  "modalities": ["audio", "text"],
  "source": "exp15_debug_fixture_v1",
  "source_group": "fixture_a2a_0001",
  "license": "fixture-only",
  "split": "debug",
  "loss_policy": "assistant_text_and_audio"
}
```

#### `negative_control`

这条记录由一条正例替换输入音频得到。正例的答案是“三点”，替换后的音频不包含会议改期信息。因此 target 必须说明“当前媒体不支持这个答案”，不能继续复述正例答案。

```json
{
  "id": "fixture_neg_0001",
  "counterfactual_of": "fixture_positive_meeting_0001",
  "negative_kind": "wrong_media",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "audio", "asset_id": "wrong0"},
        {"type": "text", "text": "录音中会议改到几点？"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "当前录音没有提供会议改期信息，无法确定。"},
        {"type": "audio_codes", "sidecar": "fixture_neg_0001/answer_codes.i32le"}
      ]
    }
  ],
  "assets": [
    {
      "asset_id": "wrong0",
      "uri": "sha256://REQUIRED_64_HEX",
      "path": "fixture_neg_0001/input.wav",
      "type": "audio",
      "sample_rate": 16000,
      "content_note": "不含会议改期信息"
    }
  ],
  "speech_target": {
    "answer_audio": {
      "path": "fixture_neg_0001/answer.wav",
      "sha256": "REQUIRED_64_HEX",
      "sample_rate": 24000
    },
    "answer_codes": {
      "path": "fixture_neg_0001/answer_codes.i32le",
      "sha256": "REQUIRED_64_HEX",
      "dtype": "int32_le",
      "shape": ["F", 8],
      "layout": "frame_major_q0_to_q7"
    }
  },
  "voice_condition": {
    "spk_emb": {
      "path": "voice/v0.spk.f32le",
      "sha256": "REQUIRED_64_HEX",
      "dtype": "float32_le",
      "shape": [192]
    },
    "ref_codes": {
      "path": "voice/v0.ref.i32le",
      "sha256": "REQUIRED_64_HEX",
      "dtype": "int32_le",
      "shape": ["R", 8],
      "layout": "frame_major_q0_to_q7"
    }
  },
  "target_semantics": {
    "support_status": "unsupported_by_current_media",
    "must_not_assert": ["会议改到三点"],
    "required_action": "state_insufficient_evidence"
  },
  "targets": {"text": true, "audio_codes": true},
  "bucket": "negative_control",
  "task": "wrong_audio_counterfactual",
  "modalities": ["audio", "text"],
  "source": "exp15_debug_fixture_v1",
  "source_group": "fixture_positive_meeting_0001",
  "license": "fixture-only",
  "split": "debug",
  "loss_policy": "assistant_text_and_audio"
}
```

负控制不是“把正例答案设成较低权重”。它仍使用普通 assistant text CE 和 8 路 codec CE，只是监督答案变了：

- `wrong_media` 和 `empty_media`：明确说当前媒体不足以支持问题中的具体答案；
- `conflicting_media`：指出冲突，并分别陈述各媒体实际支持的内容；不能擅自选一边；
- assistant audio 必须朗读同一条负控制文本，不能沿用正例的 `answer_audios`。

### 7.2 从 sidecar 到九路 tensor

adapter 对三条 fixture 分别生成以下 Parquet 字段：

| fixture | `question_audios` | `answer_audios` | `spk_emb` | `ref_audios` |
|---|---|---|---|---|
| `text_audio` | `[]` | `[flatten(answer_codes)]` | 192 个 `float32` | `flatten(ref_codes)` |
| `audio_audio` | `[input.wav bytes]` | `[flatten(answer_codes)]` | 192 个 `float32` | `flatten(ref_codes)` |
| `negative_control` | `[input.wav bytes]` | `[flatten(answer_codes)]` | 192 个 `float32` | `flatten(ref_codes)` |

`flatten` 固定按 `frame0(q0...q7), frame1(q0...q7), ...` 展开。三条 debug trace 都关闭 scheduled sampling、音频 augmentation 和随机 reference dropout；否则同一 fixture 不会得到固定 tensor。

设 tokenizer 和 chat template 处理后的长度为 `T`，目标与 reference 的 frame 数分别为 `F`、`R`，最后一个 assistant range 的起点为 `a`。前 50 个 assistant token 内存在 `"</think>\n\n"` 时，`a` 移到该字面量之后。构造过程固定如下：

先创建 shape 为 `[8,T]` 的 `Y_audio`，并把全部位置填成 audio pad `2049`。令 `ref_start=max(1,a-R)`，把 reference codes 右对齐写入 `Y_audio[:, ref_start:a]`，再把八路 speaker token `2051` 写在 `ref_start-1`。构造前必须检查 `T > a+F+8`，否则第 7 路的 stop token 会被截断。

对每个码本 `q=0...7`，先取该路的 `F` 个 answer code，再在末尾追加 stop token `2050`。这段长度为 `F+1` 的目标从 `a+q+1` 开始，同时写入 `Y_audio[q]` 和 `audio_target[q]`。八路错开一个位置，是 Talker 的 delay pattern（第 01 课讲过的对角错位；忘了回[第 01 课](01_baseline_reproduction.md)），不是 padding。

完成写入后，按下表做一次 next-token shift：

| tensor | 构造方式 | shape |
|---|---|---|
| `X_audio` | `Y_audio[:, :-1]` | `[8,T-1]` |
| `X_text` | `text_ids[:-1]` | `[T-1]` |
| `input_ids` | 在第 0 维拼接 `X_audio` 与 `X_text[None, :]` | `[9,T-1]` |
| `audio_labels` | `audio_target[:, 1:]` | `[8,T-1]` |
| `text_labels` | `assistant_text_target[1:]` | `[T-1]` |
| `audio_loss_mask` | `audio_labels != -100` | `[8,T-1]` |
| `text_loss_mask` | `text_labels != -100` | `[T-1]` |

`spk_emb[192]` 作为独立 tensor 传给模型，并只在八路都为 `2051` 的位置经过 speaker projection 注入。reference 与 speaker prefix 只提供条件，不能进入 codec loss。每路末尾的 `2050` 属于有效 label，loss mask 为 1；三臂沿用同一 10× stop-token 权重。`audio_audio` 和 `negative_control` 还分别保存 `audio_inputs`、`audio_len` 以及 text lane 中 audio sentinel 的位置；输入音频特征不属于九路中的第十路。

三条 fixture 都把 sidecar 中的真实 code 值逐项写进 trace。令 `c^T_{f,q}`、`c^A_{f,q}`、`c^N_{f,q}` 分别表示三个 answer sidecar 的第 `f` 帧、第 `q` 路 code：

| fixture | `target_q`（已含 stop） | `Y_audio` 写入位置 | 返回后的 `audio_labels` / `audio_loss_mask=1` 位置 |
|---|---|---|---|
| `text_audio` | `[c^T_{0,q}, ..., c^T_{F_T-1,q}, 2050]` | `a+q+1 ... a+q+F_T+1` | `a+q ... a+q+F_T` |
| `audio_audio` | `[c^A_{0,q}, ..., c^A_{F_A-1,q}, 2050]` | `a+q+1 ... a+q+F_A+1` | `a+q ... a+q+F_A` |
| `negative_control` | `[c^N_{0,q}, ..., c^N_{F_N-1,q}, 2050]` | `a+q+1 ... a+q+F_N+1` | `a+q ... a+q+F_N` |

每条 fixture 必须导出：

| trace 文件 | shape 或内容 |
|---|---|
| `trace/<id>/input_ids.i64le` | `[9,T-1]` |
| `trace/<id>/text_labels.i64le` | `[T-1]` |
| `trace/<id>/audio_labels.i64le` | `[8,T-1]` |
| `trace/<id>/text_loss_mask.u8` | `[T-1]` |
| `trace/<id>/audio_loss_mask.u8` | `[8,T-1]` |
| `trace/<id>/spk_emb.f32le` | `[192]` |
| `trace/<id>/trace.json` | `a`、`ref_start`、audio sentinel span 与各文件 SHA-256 |

自动测试逐元素重建上表，并断言 prompt、input audio sentinel、speaker/reference prefix 和 padding 的 loss mask 都为 0。对 `negative_control` 再检查正例答案“三点”没有出现在 `text_labels`，其 `answer_audios` hash 也与正例不同；三条 `answer.wav` 的转写都要与各自 assistant 文本一致。

### 7.3 视频与音视频扩展字段

视频与音视频任务会引入额外 frontend、时间位置和 token 数，不能悄悄进入主实验。下面的 AV 记录只属于 video extension。它必须存入独立的扩展 manifest，不能出现在默认 `data/exp15/train.jsonl`；当 `video_extension.enabled=false` 时，默认 loader 读到它必须立即报错。

```json
{
  "id": "sft_av_000042",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "video", "asset_id": "v0"},
        {"type": "audio", "asset_id": "a0"},
        {"type": "text", "text": "声音与画面是否同步？给出时间依据。"}
      ]
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "不同步，声音约晚 0.8 秒。"}]
    }
  ],
  "assets": [
    {"asset_id": "v0", "uri": "sha256://...", "type": "video", "duration_s": 12.0},
    {"asset_id": "a0", "uri": "sha256://...", "type": "audio", "duration_s": 12.0}
  ],
  "targets": {
    "text": true,
    "audio_codes": false,
    "evidence": [{"start_s": 4.1, "end_s": 6.3}]
  },
  "bucket": "audio_video_text",
  "task": "av_sync_qa",
  "modalities": ["video", "audio", "text"],
  "quality_score": 0.91,
  "source": "dataset-or-generator-id",
  "license": "explicit-license-id",
  "generation": {"teacher": null, "prompt_hash": null},
  "split": "train",
  "loss_policy": "assistant_text_only"
}
```

必须另建 asset registry，不能只留本地绝对路径：

registry 是一张 CSV，必须包含 `asset_id`、`sha256`、`canonical_uri`、`license`、`source_url`、`duration` 和 `split_group` 七列。

请先用一条记录做 fail-closed 测试（默认拒绝测试：功能关着的时候，相关数据进来必须报错而非默默通过）：关闭 extension 时应报出样本 ID；启用后还要验证 lock 文件、输出 tensor shape 和时间索引。两种行为都符合预期后，才能批量编译扩展数据。

### 7.4 任务桶

任务桶决定 sampler 的统计单位，也决定回归报告的粒度。每个样本只能有一个主桶，并保留更细的 `task` 字段。至少包含：

- `text_general`：对话、指令遵循、结构化输出；
- `image_text`：描述、VQA、OCR、grounding；
- `audio_text`：ASR、音频问答、非语音声音；
- `video_text`：事件、时序、计数；仅 video extension；
- `audio_video_text`：同步、因果、错配；仅 video extension；
- `text_audio`：文本到语音；
- `audio_audio`：语音到语音；默认主实验占 5% token；
- `negative_control`：错媒体、空媒体、冲突媒体。

对 manifest 运行桶计数、token 计数和必需字段检查。每个启用桶至少抽取 20 条人工复核；无法从 `bucket + task + modalities` 唯一确定 collator 路径的记录不得进入训练。

## 8. 数据来源与开放边界

本节用于区分方法参考与可训练资产。MiniMind-O、LLaVA、BLIP-2、OpenOmni、SLAM-Omni 与 Qwen3-Omni 可作为方法参考；论文或代码公开不代表训练集、过滤流水线和内部合成器全部公开。

数据进入训练前逐源记录：

- 官方下载页和版本；
- license 与允许的用途；
- 是否允许再分发媒体；
- 是否为人工、规则或模型合成；
- teacher 模型和生成 prompt；
- 去重键；
- 隐私与 PII 处理；
- benchmark contamination 检查（污染检查：确认评测题没混进训练集）。

如果只能公开 manifest，不能公开原媒体，就将 `redistributable=false` 写入 registry。

完成 registry 后，随机抽取每个来源 20 条，核对媒体可访问性、license 和生成记录。禁止把 benchmark test 反向生成成训练问答；没有 provenance 的混合数据只能标记为来源未知，不能写入开放 recipe。

## 9. 数据三档

数据档位对应三种不同问题：pilot 检查实现，standard 比较训练策略，full 验证规模趋势。不要用 pilot 的波动结果宣称能力提升。

| 档位 | 训练 token | 媒体量 | 用途 |
|---|---:|---:|---|
| pilot | 20M–50M | 每个已启用桶 2k–10k | 检查 schema、mask、采样 |
| standard | 0.5B–2B | 每个默认主桶 50k+ | 三臂与三 seed |
| full | 5B–30B | 百万级 | 仅在 standard 通过后 |

默认档位不统计 `video_text` 或 `audio_video_text`；video extension 的媒体量、token 预算和三臂结果在独立报告中重新核算。

独立默认 recipe 的有效 token 比例：

| 任务桶 | 有效 token 比例 |
|---|---:|
| `text_general` | 25% |
| `image_text` | 25% |
| `audio_text` | 20% |
| `text_audio` | 15% |
| `audio_audio` | 5% |
| `negative_control` | 10% |

该比例是预注册起点。训练中记录实际 token 比例，并依据分模态能力对照表决定下一次实验是否调整；当前 run 的比例不能在看到 test 结果后更改——看完结果再改配比，等于把 test 集当成了调参集。

启用已锁定 video frontend 后，扩展 recipe 才改为：

| 任务桶 | 有效 token 比例 |
|---|---:|
| `text_general` | 20% |
| `image_text` | 20% |
| `audio_text` | 15% |
| `video_text` | 15% |
| `audio_video_text` | 10% |
| `text_audio` | 10% |
| `audio_audio` | 5% |
| `negative_control` | 5% |

两套 recipe 分别生成独立报告。编译 manifest 后检查默认臂的 video/AV 记录数为 0；否则立即停止。默认臂与 video 臂不能混入同一张三臂比较表。

## 10. 三个对照实验组

三臂只比较任务进入训练的时间顺序。总 token、每任务累计 token、可训练参数和优化器保持一致——变量只有一个：什么时候教什么。

| 臂 | recipe | 采样 |
|---|---|---|
| A：sequential | image→audio understanding→speech→text | 每个阶段只采预注册任务 |
| B：joint-only | 从第一步按固定比例混合全部任务 | 每个阶段使用相同全局比例 |
| C：warmup-joint-replay | 输入对齐→单模态稳定→联合训练→固定回放 | 每阶段使用预注册比例 |

公平控制：

- 同一起点 checkpoint；
- 相同原始样本池和总训练 token；
- 每个任务在整个 run 中消耗相同的累计 token；
- 每个阶段和每个臂使用完全相同的可训练参数；
- 同一 optimizer、LR 面积、precision；
- 同样的三 seed；
- 同样的评测 prompt 和解码参数。

三臂每个任务的累计 token 都固定为：`text_general 25%`、`image_text 25%`、`audio_text 20%`、`text_audio 15%`、`audio_audio 5%`、`negative_control 10%`。每项均按全 run 的 non-padding token 计算，臂间绝对误差不得超过总预算的 0.5 个百分点。

运行前 diff 三份 resolved config，并保存差异白名单。白名单只允许 stage sampler 和数据顺序不同；模型、LoRA、冻结范围、总 token、各任务累计 token、优化器和评测都不能不同。GradNorm 不进入主比较，只在扩展题中评估；若差异白名单之外还有字段变化，本次比较无效。

## 11. 目标 recipe

C 臂把输入对齐、单模态稳定、联合训练和回放分成四个时间段。A/B 使用相同的四段 token 边界，以保证学习率和 checkpoint 位置可以直接比较：

| 阶段 | 在全程 token 进度中的区间 |
|---|---|
| Stage 0 | 训练前：冻结基线并完成数据审计 |
| Stage 1 | 0%–5% |
| Stage 2 | 5%–25% |
| Stage 3 | 25%–85% |
| Stage 4 | 85%–100% |

下面各数字都是“占整个 run 的 token 百分比”，不是阶段内部百分比。每行之和等于该阶段预算，每列在四个阶段之和等于第 9 节的全局任务比例。

| 臂与阶段 | text | image→text | audio→text | text→audio | audio→audio | 负控制 |
|---|---:|---:|---:|---:|---:|---:|
| A / Stage 1 | 0 | 5 | 0 | 0 | 0 | 0 |
| A / Stage 2 | 0 | 20 | 0 | 0 | 0 | 0 |
| A / Stage 3 | 10 | 0 | 20 | 15 | 5 | 10 |
| A / Stage 4 | 15 | 0 | 0 | 0 | 0 | 0 |
| B / Stage 1 | 1.25 | 1.25 | 1.00 | 0.75 | 0.25 | 0.50 |
| B / Stage 2 | 5.00 | 5.00 | 4.00 | 3.00 | 1.00 | 2.00 |
| B / Stage 3 | 15.00 | 15.00 | 12.00 | 9.00 | 3.00 | 6.00 |
| B / Stage 4 | 3.75 | 3.75 | 3.00 | 2.25 | 0.75 | 1.50 |
| C / Stage 1 | 0 | 2.50 | 2.50 | 0 | 0 | 0 |
| C / Stage 2 | 0 | 10.00 | 10.00 | 0 | 0 | 0 |
| C / Stage 3 | 15.00 | 10.50 | 5.50 | 15.00 | 5.00 | 9.00 |
| C / Stage 4 | 10.00 | 2.00 | 2.00 | 0 | 0 | 1.00 |

A 的 Stage 3 再按固定次序执行三个连续子段：`audio_text 20%`；`text_audio 15% + audio_audio 5%`；`text_general 10% + negative_control 10%`。B 在四个阶段都使用同一全局比例。C 的 Stage 1/2 只用图像和音频理解数据，Stage 3 联合训练，Stage 4 使用训练前固定的回放数据。

主实验不允许把 LoRA 当成可选项。A/B/C 从 Stage 1 到 Stage 4 都使用同一参数范围：

- 全量训练 `vision_connector` 与 `audio_connector`；
- 训练 backbone attention/MLP LoRA 和 Talker attention/MLP LoRA；
- LoRA 的 target modules、rank、alpha 和 dropout 在三臂、四阶段完全相同；
- 冻结 vision encoder、audio encoder、backbone 原始权重、Talker 原始权重和 codec；
- optimizer 参数名清单和 `trainable_parameter_count` 必须逐臂相同。

默认 LoRA 使用 `rank=16`、`alpha=32`、`dropout=0.05`。若锁定源码中的模块名不匹配，先修正 target modules 并为三臂生成新的共同配置，不能让某一臂跳过 LoRA。只训练 connector 的方案属于另一个消融实验，不能替换本课主表。

每次阶段切换前后保存 100 条固定 probe 的输出、逐任务 loss 和梯度范数；切换记录缺失时不能归因阶段效果。

## 12. 实验步骤

### Step 1：建立分模态基线对照表

先定义训练后不能丢失的能力。默认对每个非视频任务桶固定 100–500 个 held-out cases。

记录基线、随机水平、指标方向和不可退化阈值。

默认只为非视频主桶建表；只有 `video_frontend.lock.json` 校验通过且三臂共同启用 extension 时，才追加 `video_text` 与 `audio_video_text` 的独立扩展基线。用同一 seed 重跑 10% case；指标不能复现时先冻结评测环境。

### Step 2：验证数据字段、序列布局和 loss mask

错误的 sentinel 或 loss mask 会让模型优化错误位置。默认对每种非视频任务各取 10 条；若另启用 video extension，再对扩展任务各取 10 条并单独报告。导出：

- 序列 token 和 token type；
- asset 到 token 的对齐；
- labels 与 `-100` 区域；
- codec codebook 的 delay/mask；
- position/time id；
- packing 后 segment mask。

人工逐 token 审一次，确认资产顺序、目标范围和 padding；随后把同一断言写成自动测试。人工标注与 collator 输出任一位置不一致时，不启动 pilot。

### Step 3：清洗、去重和 contamination 审计

同一媒体的裁剪或改写跨 split 会虚增评测分数。至少按以下层级去重：

- exact ID/hash；
- 文本近重复；
- 图像 perceptual hash；
- 音频 fingerprint；
- video extension 启用时：视频关键帧 + 音轨组合；
- 同源媒体的不同裁剪按 group 切分。

输出每层去重前后数量、阈值和被删除样例。随机检查 50 组近重复；发现 test 源条目进入训练时，重建 split 并生成新数据版本。

### Step 4：跑 1k-step pilot

pilot 用于验证 sampler、loss 和系统负载。三臂各跑一个 seed：

- 观察各桶实际 token 比例；
- 默认查 loss 是否被长音频或 codec 主导；启用 video extension 时再检查长视频 token；
- 查梯度范数；
- 查吞吐、padding 和 OOM；
- 每 250 steps 跑小回归。

任何桶实际 token 比例偏离目标超过 10% 相对值，先修 sampler。还要从 step 500 checkpoint 恢复并接续 50 steps；token 计数、学习率或数据游标不连续时，pilot 不通过。

### Step 5：执行三臂 standard

每消耗固定 non-padding token 做 checkpoint 和全桶 eval。三臂在同一累计 token 位置比较，不等待 epoch 结束。

不同臂的 packing 与样本长度不同，因此 epoch 不能作为公平的横轴。验收时，三个最终 checkpoint 的总 token 和各任务 token 都要落在预注册误差范围内。

### Step 6：测梯度相互作用

共享参数可能收到方向相反的任务梯度——这就是给跷跷板装上传感器。每 500–1000 steps，在固定 probe batch 上分别反向各任务 loss 并记录：

```python
for task in task_buckets:
    g[task] = grad_vector(loss(task))
cosine[i, j] = dot(g[i], g[j]) / (norm(g[i]) * norm(g[j]))
```

cosine 只描述该 probe 上的梯度方向，不证明任务之间存在因果冲突。把 cosine 与相邻 checkpoint 的逐任务变化并列；只有负 cosine 反复出现且对应能力下降，才进入下一轮干预假设。单次负余弦就喊"任务冲突"，和听见一声咳嗽就诊断肺炎一样草率。

### Step 7：执行模态条件反事实

高分回答可能来自语言先验，未必使用了媒体。对所有多模态 case 生成：

- remove modality；
- shuffle modality；
- wrong modality；
- degraded modality；
- text-only paraphrase。

保持问题与解码配置不变，只改媒体条件。模型在错媒体上仍给出同一具体答案时，该 case 记为未验证 grounding；报告原媒体与各反事实条件的准确率差。

### Step 8：做 capability replay

capability replay 指在训练末段重放预留旧任务数据。只对 C 臂执行预注册的 replay：

- Stage 4 在总预算中的比例固定为 15%；
- 其中 `text_general 10%`、`image_text 2%`、`audio_text 2%`、`negative_control 1%`；
- 数据只能来自预留训练池，不能包含 dev/test；
- 在第一个 optimizer step 前生成 `replay_manifest.jsonl`，记录每条样本与 token 数，并把 SHA256 写入三臂共同的实验登记文件；
- Stage 3 结束后不得根据回归结果更换桶、样本或比例。

保存 Stage 3 与 Stage 4 的逐桶差值。被回放桶改善而未回放桶明显退化时，报告为能力转移，不能只展示改善项。若想根据 Stage 3 结果自适应选择回放桶，必须新建一个不进入 A/B/C 主比较的实验。

### Step 9：三 seed 汇总并确定赢家

训练前冻结 `eval/primary_cases.jsonl`。每行写明 `case_id`、`source_group`、`bucket`、`score_fn` 和 evaluator hash。主分数只使用五个默认新增/组合桶：`image_text`、`audio_text`、`text_audio`、`audio_audio`、`negative_control`。每个 case 先由固定 evaluator 产生 `[0,1]` 分数；语音内容使用 `1-min(ContentWER,1)`，负控制只有同时满足 `support_status` 和 `must_not_assert` 才记 1。UTMOS、speaker similarity 和旧能力回归保留为单独门槛，不塞进主分数。

先在同一 `source_group` 内平均 case，再在每个 bucket 内平均 source group，最后对五个 bucket 等权平均，得到每个 arm、每个 seed 的 $S_{a,s}$。这一步防止一个源媒体生成许多改写后获得更高权重。

两个预注册主对比是：

$$
D_{C-A}=\frac{1}{3}\sum_s(S_{C,s}-S_{A,s}),\qquad
D_{C-B}=\frac{1}{3}\sum_s(S_{C,s}-S_{B,s}).
$$

最小实际效应固定为 `0.02`，即主分数 2 个百分点。统计脚本执行 10,000 次分层配对 bootstrap：

1. 三个 seed 成对有放回抽样；
2. 在每个抽到的 seed 和 bucket 内，对 `source_group` 有放回抽样；
3. 同一组 seed、source group 和 case 同时用于 C 与对照臂；
4. 生成失败按 0 分保留，不能从某一臂单独删除。

对两个主对比分别取单侧 97.5% lower bound；这是对两次比较做 Bonferroni 后的 family-wise 5% 标准。汇总表必须同时给每个 seed 的差值、三 seed mean/SD、bootstrap lower bound、完成率和 GPU-hours。

先判断**实验是否有效**：三臂使用同一 test manifest/evaluator，三个 seed 全部完成，累计 token、可训练参数和解码配置通过公平性检查，且没有 test 泄漏。有效实验可以得到负结果，不能因为 C 没赢就改成“实验失败”。

只有同时满足以下条件，才写“C 相比 sequential 和 joint-only 有正向优势”：

- `C−A` 与 `C−B` 的 97.5% lower bound 都大于 `0.02`；
- 两个对比在三个 seed 上都分别为正；
- 所有既有能力不低于起点 97%，其他硬性验收项也通过。

只通过一个对比时，只能写“C 高于该对照”；lower bound 大于 0 但不超过 0.02 时写“方向为正，实际效应不足”。逐 bucket 结果用于解释来源，不设显著数量门槛，也不能替代两个主对比。

## 13. 配置示例

```yaml
experiment:
  name: exp15_warmup_joint_replay
  seeds: [17, 23, 41]
model:
  init_checkpoint: jingyaogong/minimind-3o
  revision: ee3febbd08cc5b2bd41c039c825a8934232fee33
  trainable_all_stages:
    [vision_connector, audio_connector, backbone_lora, talker_lora]
  frozen:
    [vision_encoder, audio_encoder, backbone_base, talker_base, codec]
  lora:
    required: true
    rank: 16
    alpha: 32
    dropout: 0.05
    target_groups: [backbone_attention_mlp, talker_attention_mlp]
data:
  manifest: data/exp15/train.jsonl
  asset_registry: data/exp15/assets.csv
  arm_stage_mixture: data/exp15/arm_stage_mixture.yaml
  replay_manifest: data/exp15/replay_manifest.jsonl
  replay_manifest_sha256: REQUIRED_BEFORE_TRAINING
  speech:
    audio_codebooks: 8
    audio_pad_token: 2049
    audio_stop_token: 2050
    audio_spk_token: 2051
    require_answer_audio_codes: true
    require_spk_emb_192: true
    require_ref_codes: true
    scheduled_sampling_probability: 0.0
    reference_dropout_probability: 0.0
    trace_fixtures: [fixture_t2a_0001, fixture_a2a_0001, fixture_neg_0001]
  video_extension:
    enabled: false
    frontend_lock: null
  sampler: token_balanced_temperature
  temperature: 0.7
  pack_by: [modality_signature, length_bucket]
loss:
  assistant_text: 1.0
  codec_q0: 1.0
  codec_q1_q7: 0.5
  log_per_task: true
train:
  total_tokens: 1000000000
  stage_token_fraction: [0.05, 0.20, 0.60, 0.15]
  lr: 2.0e-5
  warmup_ratio: 0.03
  bf16: true
  gradient_checkpointing: true
distributed:
  world_size: 8
  fsdp: full_shard
eval:
  suites: [text, image, audio, speech, counterfactual]
  primary_manifest: eval/primary_cases.jsonl
  primary_buckets: [image_text, audio_text, text_audio, audio_audio, negative_control]
  primary_contrasts: [C_minus_A, C_minus_B]
  min_effect_absolute: 0.02
  missing_generation_score: 0.0
  bootstrap:
    unit: source_group
    paired: true
    hierarchical_seed_resampling: true
    resamples: 10000
    rng_seed: 150041
    one_sided_lower_bound: 0.975
```

YAML 中 `frontend_lock: null` 表示默认禁用视频，是机器可检验的运行状态：loader 遇到 `video` 或 `audio_video` 样本必须立即报错。把 `enabled` 改为 `true` 且 lock 文件校验通过后，才能加入 `video`、`av` suites。

launcher 必须在启动时断言 `lora.required=true`，四个阶段的 optimizer 参数名清单相同，三臂 `trainable_parameter_count` 相同，并验证 `arm_stage_mixture.yaml` 的每行、每列各组的任务比例合计必须与上方阶段表一致。`replay_manifest_sha256` 必须在训练前替换为真实 hash；占位值、文件变化或 C 的 Stage 4 比例不匹配时立即退出。

先以默认配置运行一条 video 样本，确认 loader 以样本 ID 报错；再检查训练日志中的 video/AV token 计数为 0。配置、错误行为和计数三者一致，才算默认边界生效。

## 14. 训练预算与 8 卡运行方案

先用最低档验证数据与 mask，再扩大到三臂比较。预算表中的 GPUh 是规划区间，正式报告以作业日志累计值为准。

| lane | 硬件 | 规模 | 目的 |
|---|---|---|---|
| debug | 1×24GB | 500 samples | schema/mask/sampler |
| pilot | 2–4×24–48GB | 20M–50M token | 三臂一 seed |
| standard | 8×48–80GB | 0.5B–2B token | 三臂三 seed |
| full | 8×80GB 或多节点 | 5B+ token | 扩规模验证 |

8 卡先做 200-step profile：

- global batch 以有效 token 固定；
- gradient accumulation 调到每卡负载接近；
- 默认对长音频采用 length bucket；启用 video extension 后，再把长视频纳入独立的 length bucket；
- 保存 data wait、host decode、H2D、forward、backward、collective 时间；
- 若 data wait 超过 step time 10%，先修数据流水线。

profile 重复两次并保存 trace。第二次运行没有 OOM/NaN，八张卡负载差异在预注册范围内，且 data wait 低于阈值后，才进入 standard。多卡训练的手艺在[第 13 课](13_distributed_8gpu.md)已经练过；本课直接沿用那套 profile 纪律。

## 15. 指标

### 能力

能力指标按任务桶和语言分别报告，用于回答各项任务是否改善：

- text instruction-following、exact/format pass；
- image VQA/OCR/grounding；
- audio ASR WER、audio QA；
- video extension 启用时：video temporal/causal QA；
- video extension 启用时：audio-video sync/mismatch；
- TTS 内容 WER、speaker similarity、UTMOS；
- 同时依赖两种模态的组合任务 macro score。

每个汇总分都要能下钻到 case ID；缺少 per-case 输出的指标不进入主要结论。

### 系统

系统指标用来解释训练成本与数据瓶颈。统一以 non-padding token 计吞吐：

- 每任务有效 tokens/s；
- peak HBM；
- padding 比例；
- data wait；
- 每十亿有效 token 的 GPU-hours；
- checkpoint 保存/恢复；
- 训练中 NaN、OOM、skipped batch。

三臂使用同一 profiler 配置。若吞吐差来自某一臂 padding 更高，要同时报告有效 token 与 raw token，不能直接比较 step time。

### 回归与依赖性

回归指标检查旧能力，依赖性指标检查模型是否使用输入媒体：

- 每个起点能力的绝对变化；
- remove/shuffle/wrong-modality gap；
- 输出长度和格式有效率；
- text perplexity；
- 每桶梯度范数与 cosine。

在起点和每个阶段末运行同一套 dev 与 regression-dev。变化超过验收阈值时，标记首次发生的阶段和相关任务桶。固定 test manifest 在所有阶段、checkpoint、阈值和统计脚本冻结后只运行一次，不能用来选择阶段或修改训练配比。

## 16. 验收条件

- 所有主任务桶 held-out 均不低于起点 97%；
- `C−A` 与 `C−B` 的 source-group 配对 bootstrap 单侧 97.5% lower bound 均大于 `0.02`；
- `C−A` 与 `C−B` 在三个 seed 上分别都为正；
- wrong-modality 与原媒体正确率差至少 20 个百分点；
- 任一数据桶实际 token 比例偏差小于目标的 5% 绝对值；
- 无 benchmark test 泄漏；
- 每条训练样本 provenance/license 可追；
- 训练恢复后 100 steps 动力学与未中断 run 相符；
- 三臂每个任务的累计 token 误差都在总预算的 0.5 个百分点内；
- 三臂 LoRA 配置、optimizer 参数名和可训练参数总数完全相同；
- C 的 replay manifest 在训练前固定，Stage 4 使用的文件 hash 与登记值一致；
- 逐 case 审计能区分 coverage、representation 和 reasoning。

前三臂公平性、三个 seed、固定 evaluator 和完整 paired case 通过时，实验本身有效；上面两个主对比未达阈值时记录为有效的零结果或负结果。只有全部条目通过，才接受“C 是本课赢家”的正向结论。

如新能力上升但文本或已有模态下降超过 3%，结论写为：联合训练产生负迁移，不满足发布条件。

## 17. 失败诊断表

| 症状 | 可能原因 | 检查 | 最小修复 |
|---|---|---|---|
| audio loss 支配总 loss | codec token 多 | 分目标 token/loss | token balance、调 λ |
| joint 比 sequential 差 | warmup 不足或冲突 | stage probe | 增加 connector warmup |
| 文本能力下降 | anchor 太少 | text replay curve | 固定 10–20% anchor |
| 图像答对、错图也答对 | 语言先验 | wrong-image | 加 hard negative |
| video extension 中视频只看首帧 | 数据模板捷径 | shuffle/drop frames | 时序反例 |
| TTS 音质升、内容错 | codec 目标压过语义 | 输出 ASR | 提高 semantic/text loss |
| 训练吞吐骤降 | 解码与 padding | profiler | 离线特征、bucket packing |
| 高质量桶过拟合 | 重复/teacher style | n-gram 与来源统计 | 去重、增人工样本 |
| replay 修复一项又伤另一项 | 追涨式采样 | 预注册比例 | 固定 replay budget |
| average 升但小语种崩 | macro 掩盖尾部 | 分语言报告 | 分层 sampler |

## 18. 逐个样例检查

至少审计 100 个不同来源的 case。这里的“不同”要求 `source_item_id` 和资产 hash group 都不重复；同一媒体的原始、删除、打乱和错媒体版本只算一个 case：

- 六个默认主任务桶各 10 个，共 60 个；
- 另外选择 10 个 image-text 和 10 个 audio-text grounding case，共 20 个；
- 再选择 20 个失败或反事实 case。

三组来源不得重叠，因此总数至少为 100。每个 case 可以带多个反事实输入，但 `cases/audit.md` 必须用同一个 `case_id` 把这些输入归在一起。默认课程用下面的完整非视频模板。

### 18.1 默认非视频审计模板

该模板把一项分数还原为输入证据、encoder/connector 输出和最终回答。请先完整填写一个图像 case，再用脚本校验 hash、必填字段与答案文件是否存在。

```yaml
case_id: sft_image_000042
bucket: image_text
task: image_instrument_identification
modalities: [image, text]
input_assets:
  - {asset_id: i0, type: image, sha256: "sha256-of-image"}
required_facts:
  - producer: vision_encoder
    evidence: {asset_id: i0, bbox_xyxy: [118, 164, 902, 741]}
    fact: "木质共鸣箱、圆形音孔和六根琴弦"
gold: "主要乐器是原声吉他。"
gold_supported_by_input: true
encoder_connector_observation:
  image_fact_present: true
baseline_answer: "这是一个乐器。"
joint_answer: "主要乐器是原声吉他；可见木质共鸣箱、圆形音孔和六根琴弦。"
remove_image_answer: "仅凭问题无法确定图片中的乐器。"
wrong_image_answer: "替换图片中的主要乐器是钢琴，不是吉他。"
near_duplicate_in_train: false
first_changed_stage: stage3
failure_layer: null
data_source: source_x_v2
license: explicit-license-id
```

随机抽取 10 个 case 由第二人复核。`required_facts` 无法在原媒体中定位，或 `first_changed_stage` 无对应 checkpoint 时，该 case 不得用于归因。

### 18.2 Video/AV extension-only 审计模板

只有 `video_frontend.lock.json` 校验通过并实际运行独立 video extension 报告时，才增加下面的 AV case；它不计入默认课程的 100-case 必做集合。

```yaml
case_id: sft_av_000042
bucket: audio_video_text
task: av_sync_qa
required_facts:
  - {producer: audio_encoder, span: 4.9-5.4, fact: "撞击声"}
  - {producer: video_encoder, span: 4.1-4.6, fact: "物体接触"}
baseline_answer: "同步"
joint_answer: "不同步，声音晚约 0.8 秒"
gold: "不同步，声音晚 0.7-0.9 秒"
wrong_audio_answer: "无法确定"
failure_layer: null
data_source: source_x_v2
```

默认模板和启用后的扩展模板都逐项回答：

1. gold 是否可由输入支持；
2. 必要事实是否被 encoder/connector 产生；
3. 模型是否同时使用两种模态；
4. 输出错是 coverage、representation、reasoning 还是 Talker；
5. 该 case 是否与训练近重复；
6. 哪个训练阶段首次改变了答案。

回放脚本应同时加载默认与扩展 case，并确认默认报告的 AV case 数为 0。该断言用于防止扩展结果混入主结论。

## 19. 交付物

```text
artifacts/exp15/
├── configs/{sequential,joint,warmup_joint_replay}.yaml
├── data/{schema.json,assets.csv,mixture.yaml}
├── eval/{primary_cases.jsonl,evaluator.lock.json}
├── traces/{fixture_t2a_0001,fixture_a2a_0001,fixture_neg_0001}/
├── provenance/{licenses.csv,generators.md,dedup.json}
├── checkpoints/index.json
├── metrics/{per_task,regression,counterfactual,primary_bootstrap,system}.jsonl
├── gradients/cosine_snapshots.npz
├── cases/audit.md
├── capability_ledger.csv
└── report.md
```

默认交付物不得包含未锁定的视频资产或 AV 指标。若另做 video extension，把其 manifest、frontend lock、指标与 cases 放在独立的 `artifacts/exp15_video_extension/`，不得覆盖默认报告。

`capability_ledger.csv` 至少包含：

表中至少要有 `capability`、`baseline`、`arm_a`、`arm_b`、`arm_c`、`delta`、`cost_gpu_h` 和 `acceptance` 八列。

## 20. 复现清单

- [ ] 起点 checkpoint/hash 固定；
- [ ] 三臂总训练 token、逐任务累计 token 与可训练参数一致；
- [ ] LoRA 在三臂四阶段均启用，配置与参数名清单一致；
- [ ] schema 和 loss mask 自动测试通过；
- [ ] T2A、A2A、negative-control 三条 fixture 的 9 路 tensor、labels 和 loss mask trace 逐元素通过；
- [ ] 三类语音样本的 answer codes、`spk_emb`、reference sidecar 与 SHA-256 完整；
- [ ] 默认 manifest 仅含 text/image/audio，且禁用 extension 时 video/AV loader 拒绝测试通过；
- [ ] asset registry 无悬空项；
- [ ] sample 比例与 token 比例同时记录；
- [ ] 原始媒体级去重已执行；
- [ ] benchmark contamination 已审计；
- [ ] teacher、prompt、过滤规则已记录；
- [ ] 3 seed 与固定 eval prompt 已完成；
- [ ] `primary_cases.jsonl`、evaluator hash、两项主对比和 `min_effect=0.02` 在训练前固定；
- [ ] source-group paired bootstrap 保留失败 case，并输出每 seed 差值和 97.5% lower bound；
- [ ] remove/shuffle/wrong-media 已完成；
- [ ] 所有旧能力回归已报告；
- [ ] 训练可断点恢复；
- [ ] 至少 100 个不同 `source_item_id`/资产组的 cases 可回放；
- [ ] replay manifest 在训练前固定并保存 hash；
- [ ] license/不可再分发边界已标记。

video extension 不属于默认结课条件；若启用，另检查 frontend lock、独立 manifest、扩展三臂公平性与 AV case 回放。

## 21. 前沿对照与改造方向

本课的三臂之争（先分后合、一锅端、warmup-联合-回放），前沿系统早就用真金白银投过票，投给的基本都是 C 臂的放大版。[LLaVA](https://arxiv.org/abs/2304.08485) 是最早把这条路走通的公开工作之一：第一阶段冻住语言模型和视觉编码器，只训 projector 做特征对齐；第二阶段才开放语言模型做视觉指令微调——对应本课 C 臂 Stage 1/2 的 connector warmup。[BLIP-2](https://arxiv.org/abs/2301.12597) 更极端：两侧 tower 全程冻结，先做表征学习阶段再做生成学习阶段，全部压力交给中间的 Q-Former。[Qwen3-Omni](https://arxiv.org/abs/2509.17765) 把这套课程扩到全模态：报告描述了分阶段的多模态训练课程，先对齐各模态编码器，再做大规模联合训练，并强调通过在训练早期混合单模态与跨模态数据来控制模态间折损；其报告声称在同规模单模态模型对照下语言和视觉能力没有明显牺牲，但训练数据的具体构成与过滤流水线未完全公开。[Moshi](https://arxiv.org/abs/2410.00037)（第 01 课引用过）解决跷跷板的思路是给语音流配一条文本流做"内心独白"，让文本监督一直在场、不给遗忘留窗口。共同点很清楚：没有一家是把所有数据倒进一个锅里从第一步就联合训练的——B 臂在前沿是缺席的。

三类差距要分开记账。规模差距：前沿联合 SFT 的数据量和参数量都在千倍以上，全参数训练，本课 standard 档 0.5B–2B token、LoRA 加 connector；这类差距花钱能缩小。机制差距：本课教的 token-balanced 采样、逐桶回归、预注册 replay、梯度余弦监控，就是前沿在用的模态平衡工具箱的缩小版——这部分做完本课你并不落后。真正的断档在数据工程：前沿的配比不是拍脑袋定的 25/25/20/15/5/10，而是靠大规模消融和数据质量流水线迭代出来的，且这一部分几乎从不公开。本课的预注册配比表是诚实的替代：承认配比是假设，用三臂实验检验它。



1. **数据配比敏感性扫描（配比策略级）。** 把默认 recipe 中 `audio_audio` 与 `text_audio` 的合计 20% 分别改成 10% 和 30%（从 `text_general` 与 `image_text` 等量借还），共两个新配比，其余全部沿用 C 臂配置。改动位置：`data/exp15/arm_stage_mixture.yaml`，每列总和仍须过 launcher 校验。预算：pilot 档 20M–50M token、单 seed，2–4×24GB，一天内完成两个配比。预期：语音桶份额升则 TTS 内容 WER 降、text held-out 跌，反向亦然，画出配比对能力的单调曲线。失败判定：两个端点的逐桶指标与默认配比无稳定差异，说明 pilot 规模太小或评测噪声过大，结论不可用，需升到 standard 档再扫。
2. **codec 损失权重重设计（损失设计级）。** 默认 `codec_q0: 1.0`、`codec_q1_q7: 0.5` 是平的；改成按码本层级指数衰减 $w_q = 0.5^q$，再加一个只训 q0–q3、推理时 q4–q7 由 greedy 补全的激进版。直觉：RVQ 深层码本记的是听感细节，对"说对内容"贡献小，却占走一半 codec 梯度。改动位置：配置 `loss` 段与 `train_sft_omni.py` 中逐码本 loss 加权处。预算：pilot 档单 seed，每个方案约 0.5–1 天。预期：指数衰减版 TTS 内容 WER 持平或改善、UTMOS 略降；text 桶回退变小（codec 对共享层的拉扯减弱）。失败判定：内容 WER 明显上升，说明深层码本并非纯细节，恢复默认权重并记录。
3. **梯度手术式模态平衡（模态平衡策略级）。** Step 6 只测梯度余弦不干预；本改造把它变成干预：在共享 LoRA 参数上，当两任务梯度余弦为负时，把其中一方在冲突方向上的分量投影掉再更新（gradient surgery 思路）。改动位置：optimizer step 前加一个梯度后处理 hook，只作用于 `backbone_lora` 参数组；三臂主表不动，另开实验 ID。预算：standard 档单 seed 一次，8×48GB，约为 C 臂单 seed 的 1.1–1.3 倍时长（多一次逐任务反向）。预期：负余弦高发的桶对（如 `text_audio` 对 `text_general`）回退幅度收窄，主分数持平或小升。失败判定：吞吐掉一半以上或主分数下降，说明该规模下冲突本来就轻、手术只剩开销。
4. **replay 预算扫描（课程设计级）。** C 臂 Stage 4 的 15% replay 是预注册值，不是真理。另开三个 run 把 Stage 4 比例改为 5%、15%、25%（Stage 3 相应伸缩，全局任务比例不变，需重算配比表并过校验）。预算：pilot 档三个 run 单 seed，共约 2–3 天。预期：replay 越多旧能力回得越满、新能力（语音桶）终点越低，找出 3% 回退红线内的最小 replay。失败判定：三点连不成单调趋势，说明 pilot 噪声淹没了效应，只能在 standard 档下结论。

三条论文结论可以在本课设置里验方向：

- LLaVA "先对齐 projector 再指令微调更稳"：对应 C 臂对 B 臂。预期能复现同方向趋势——C 臂 Stage 1/2 后 image_text 桶的早期 loss 曲线比 B 臂同 token 位置更平滑、pilot 阶段梯度异常更少；若 26M 规模下两臂无差异，记录为"该规模下 warmup 效应弱于噪声"，也是有效结果。
- BLIP-2 "冻结双塔、只训中间件也能对齐"：对应本课"只训 connector"消融（第 11 节明确它不进主表）。预期部分复现：connector-only 在理解桶能追近 LoRA 版，但语音生成桶明显落后，因为 Talker 侧没有可训练容量。
- Qwen3-Omni "混合课程可以控制模态折损"：对应 C 臂对 A 臂的旧能力回归对比。预期方向一致——A 臂 Stage 4 前 text 桶回退更深；但"无折损"这个强结论在 26M 参数、2B token 内大概率复现不出来，只能看到折损变小。这正是区分规模问题和机制问题的活教材。

## 22. 原始论文与官方 recipe 精读

### [LLaVA](https://arxiv.org/abs/2304.08485)

精读：两阶段训练、visual instruction tuning data、ScienceQA 实验与消融。

阅读任务：带着两个问题读。一，一个简单 projector 什么时候就够用了？把论文的条件（数据量、任务类型、backbone 强度）列出来，对照本课 connector warmup 判断哪些条件在 26M 规模下不成立。二，论文结论哪些只对图像-文本成立？逐条标出，用表格对应本课 connector warmup 的相同点与差异——音频和语音生成桶不能直接套用它的结论。

### [BLIP-2](https://arxiv.org/abs/2301.12597)

精读：Q-Former、representation learning stage、generative learning stage。

阅读任务：比较"先对齐再生成"与本课 connector warmup 的异同：BLIP-2 冻的是两侧 tower，本课 LoRA 给 backbone 留了一条缝。读完回答：冻结 tower 的能力上限卡在哪里？把论文中的可训练模块填入本课 Stage 1/2 表格，检查每个模块在本课里对应"全量训、LoRA 训、冻结"三档中的哪一档。

### [MiniMind-O 论文](https://arxiv.org/abs/2605.03937) 与 [官方仓库](https://github.com/jingyaogong/minimind-o)

精读：训练阶段、数据组织、Thinker–Talker loss；代码重点读 dataset、collator、`train_sft_omni.py`。

阅读任务：从代码导出 T2A、A2A、I2T 的 loss mask，检查它们是否一致——不一致的地方就是官方 sequential recipe 里埋着的桶间差异，本课 joint 训练会把它放大。再按数据 provenance 判断能否支持公平的 codec 或 joint curriculum 比较；答案写进第 8 节的来源登记。

### [OpenOmni 官方仓库](https://github.com/RainBowLuoCS/OpenOmni)

精读：README 中的阶段、训练脚本、数据准备和 DPO 路径。

阅读任务：列出可以直接复现的训练阶段，以及缺少完整 provenance 的数据或 checkpoint 依赖。每项判断附代码路径或发布页。读的时候问：它的阶段划分对应本课 A、B、C 哪一臂？缺 provenance 的环节如果照抄，会违反本课第 8 节哪几条登记要求？

### [SLAM-Omni 官方实现](https://github.com/X-LANCE/SLAM-LLM/tree/main/examples/s2s)

精读：speech-to-speech example 的数据格式、encoder/projector、训练配置。

阅读任务：计算一个 speech-to-speech 样本的输入/目标 token 比例，并从配置中列出冻结模块。把结果与本课 token-balanced sampler 对照：如果它的一条样本 token 数是短文本问答的 30 倍，按样本均衡采样时语音实际吃掉多少训练份额？算出这个数字，你就彻底理解了 6.3 节。

### [Qwen3-Omni](https://arxiv.org/abs/2509.17765)

精读：Thinker–Talker、multi-modal training、数据课程与评测章节。

阅读任务：把规模化 recipe 拆成两列：可在小模型上验证的机制（阶段顺序、模态混合、replay 思路），和无法复现的数据依赖（内部数据源、过滤器、teacher 模型）。没有公开证据的字段明确标记为 `unknown`——这一列的长度，就是第 21 节说的"数据工程断档"的直观度量。

## 23. 扩展题

1. 将固定 loss 权重替换为 GradNorm，只对 C 臂做二级消融；
2. 比较 sample-balanced 与 token-balanced，默认观察音频能力和吞吐；启用 video extension 后再增加视频指标；
3. 加入按质量分数 anneal 的 sampler；
4. 研究多语言 anchor 是否比纯英文 anchor 更能抑制遗忘；
5. 把 Talker 暂时冻结，分离 understanding joint SFT 与 speech generation 冲突；
6. 引入 1% 无答案/冲突样本，测模型拒答与幻觉；
7. 将本课 capability ledger 作为[第 16 课](16_multimodal_preference_optimization.md)的偏好数据选题依据。

## 24. `joint-sft-v1` 的发布内容

通过验收的输出命名为 `joint-sft-v1`。

同时发布：

- 精确数据 mixture；
- trainable module 清单；
- 每项能力和回归；
- counterfactual dependency score；
- 默认支持的 text/image/audio 输入与 text/speech 输出；
- 若存在，单独列出通过 frontend lock 验证的 video extension，不得把它写成默认能力；
- 已知失败边界。

只有这个版本可以作为第 16/17 课的公共起点，避免 post-training 实验因 SFT 起点不同而失去可比性。

到这里，系统的状态是：第四幕换好的现代结构，加上一份各模态能力齐全、退化有账可查的联合 SFT 权重，跷跷板被课程设计和 replay 压到了验收线以内。但 SFT 只教模型"模仿好答案"，没教它"分辨好坏"——模型可能看都没看图，靠语言先验蒙出高分答案；反事实测试抓得住这种案例，SFT 本身治不了。下一课[第 16 课](16_multimodal_preference_optimization.md)引入偏好优化：用成对的好坏回答，先复现 image-only mDPO，专治"不看媒体就作答"。
