---
id: 02_multimodal_connector
title: "跨模态连接器"
summary: "把 encoder、LLM、参数量和训练 token 都大致固定，learned-query connector（用一小组可学的查询向量去挑重点的连接器）真比逐 token MLP 换来更好的信息—token—延迟平衡吗？"
unit: mechanism
play_tools: []
checkpoints:
  - "分清连接器要干的三件事：维度投影、分布对齐、信息瓶颈，别混成一锅。"
  - "用同一套接口写出 MLP、Perceiver Resampler 和轻量 Q-Former 三种连接器。"
  - "用打乱或干脆抽掉图片音频的负对照实验，验证模型是不是真在看、真在听。"
  - "在参数量和 token 预算都可比时，画出 accuracy–tokens–latency 的 Pareto 图（哪种方案在哪项上占优，一图看清）。"
---

# 第 02 课：比较三种多模态 Connector

> 内容：跨模态表示与受控消融<br>
> 建议周期：4-7 天<br>
> 硬件：1-2×24GB；8 卡用于并行运行 3 个实验臂和多个 seed<br>
> 产物：图像/音频 connector 权衡图、可插拔接口、winner checkpoint

## 1. 把图像和声音接进 Thinker

上一课完成了 MiniMind-O 基线复现，保存了 `baseline-v1`、100 个 golden case 和零扰动 trace。本课开始修改模型结构，目标是 connector。

SigLIP2 将 256×256 图像划分为 64 个 patch，并为每个 patch 输出特征向量；SenseVoice 输出逐帧语音特征。这些向量与 Thinker 的文字 embedding 维度不同，需要 connector 做映射。MiniMind-O 默认使用两层 MLP projector 逐个转换特征，不压缩序列，因此 64 个 patch 会占用 64 个 token。

本课为 connector 定义统一接口，并比较默认 MLP、Perceiver Resampler 和 Q-Former。后两种结构使用固定数量的可训练 query 汇聚输入，因此可以控制输出 token 数。三组实验保持 encoder、Thinker、训练 token 和参数量可比，分别测量准确率、token 数和延迟。

这项压缩在当前 64 个视觉 token 上影响有限，但任意分辨率图像和视频会显著增加 patch 数。第 8 至 10 课会直接使用本课得到的 connector 接口和测量结果；第 3 课替换 codec、第 11 课替换 MoE 时也会沿用同样的受控比较方法。

验收时，对同一问题分别输入正确图片、全黑图片和错配图片，确认模型输出确实依赖视觉信息。最终报告还要给出 accuracy-tokens-latency Pareto 图，并能追溯每个点对应的 connector、query 数量和 checkpoint。

本课术语（第 1 课讲过的不再重复）：

| 术语 | 简要解释 |
|---|---|
| connector（连接器） | encoder 和语言模型之间的翻译模块，把视觉/音频特征变成 Thinker 能读的 embedding |
| encoder feature / patch 特征 | 眼睛耳朵的原始输出：图像每块、语音每帧各一个向量 |
| learnable query | 可训练的"提问向量"，主动去特征堆里打捞相关信息 |
| cross-attention | query 与每个特征算相似度、按相似度加权取内容的读取机制 |
| Perceiver Resampler | 用固定数量 query 做 cross-attention 的压缩器，输出长度只由 query 数决定 |
| Q-Former | query 先互相商量（self-attention）再读特征（cross-attention），交替进行的连接器 |
| marker（占位 token） | prompt 里给模态特征预留的占位符，数量必须等于 connector 实际输出数 |
| `Tin` / `Tout` | connector 的输入/输出 token 数；两者之比就是压缩率 |
| Pareto frontier | 多个指标一起比时，没有被任何对手全面压住的方案集合；赢家只能从这里挑 |
| 反事实评测（zero/shuffled） | 把模态输入换成全零或别人的，分数不掉说明模型没在用这个模态 |

## 2. 本课解决的问题

先做个小实验：准备正确图片、全黑图片和随机错配图片三个输入，问同一个问题。如果模型给出相同答案，就无法证明模型使用了图像信息——可能是语言先验（不看图，光凭文字统计规律也能蒙对的捷径）在答题。后续实验会把这种检查做成固定的反事实评测。

MiniMind-O 当前将冻结 encoder 输出的每个时间特征或 patch 特征逐点输入两层 MLP。这个 MLP 就是现任 connector：它完成维度映射，但不压缩 token，也不显式选择更重要的视觉区域或语音片段。本课在相同训练和计算约束下比较 MLP、Resampler 和 Q-Former。结构名称和发布时间不参与选择，最终结论只依据受控实验。

实验固定 encoder、LLM、训练 token 和近似的 connector 参数量，再比较 learned-query connector 与逐 token MLP 的任务分数、输入 token 数和延迟。输出不是单一总分，而是三项指标构成的 Pareto frontier。任务分数越高越好，输入 token 数和延迟越低越好。若方案 A 在三项指标上都不差于方案 B，且至少一项更好，则称 A 支配 B；未被任何方案支配的集合就是 Pareto frontier。

出现以下任一结果时，不保留复杂 connector：

- Resampler/Q-Former 只在训练集变好，held-out（留出集：训练时藏起来、专门用于检验的数据）不变或变差；
- 打乱图像/语音后性能不降，说明模型没使用模态；
- 收益完全来自更多参数或更多视觉 token；
- 连接器变强但文本能力明显遗忘；
- 音频用固定 query 压缩后，短语音好、长语音灾难性丢失。

如果 MLP 的结果最好，就保留 MLP，并在报告中记录复杂 connector 没有带来可测收益。"简单结构赢了"同样是有效结论，这门课要的是证据。

报告要区分三类内容：

- **官方实现事实：** 当前 projector 保持 encoder token 数，不做 learned-query 压缩。
- **课程设计：** 三臂共享 encoder、backbone、训练 token 和评测集。
- **待验证假设：** learned query 能在更少输入 token 下保留任务所需信息。

## 3. 开始前需要准备什么

可从两个独立起点任选一个：

- 推荐：第 01 课的 `baseline-v1`；
- 独立模式：官方 `minimind-3o` checkpoint + 官方 eval set。

独立模式没有现成的 golden case，训练前先冻结这一课自己的回归集 `connector-regression-v1`：

- 至少 20 个 text-only、20 个 audio、20 个 image case；
- 优先使用仓库 `dataset/eval_omni/`，不足部分从 held-out 数据补齐；
- 保存输入 hash、期望语义、base 输出、推理参数和 checkpoint revision；
- 该集合不得进入 connector 训练或 query 数选择。

后文说"golden 回归"时，指：已有第 01 课产物时复用其集合，否则用上述本课集合。

要让结论只归因于 connector，以下条件在三臂（臂：同一实验里并排比较的方案分支）间保持一致：

- SigLIP2 与 SenseVoice revision；
- MiniMind Thinker、Talker；
- tokenizer 与 special token IDs；
- 图像 256×256，视觉 encoder 输出 64 token；
- 音频 16 kHz 与 SenseVoice feature；
- 训练/验证切分和 global optimizer step。

本课最终 checkpoint 命名为：

```text
connector-v1-{vision|audio}-{mlp|resampler|qformer}-q{N}
```

## 4. 完成后应具备的能力

完成后，应能从 Pareto 图上任一点定位到对应样例和配置，并完成以下工作：

1. 区分 dimension projection（维度投影）与 information bottleneck（信息瓶颈）；
2. 实现统一 `ModalityConnector` 契约；
3. 理解 Perceiver 的 latent cross-attention；
4. 理解 Q-Former 的 learnable query 与两阶段对齐；
5. 设计参数量、token 预算和计算量可比的实验；
6. 用 paired/unpaired probe（探针测试：配对/错配输入的对照检查）确认模态是否真的被使用；
7. 画 accuracy-tokens-latency 的 Pareto 图；
8. 根据输入长度和任务差异判断 audio 与 vision connector 哪些部分可以共享。

## 5. 原理:边造边讲

四个机制，沿用第 1 课的节奏：直觉、机制、数学、代码落点、验证。

### 5.1 Modality gap：两个空间说两种方言

冻结的视觉或语音 encoder 是在自己的任务上训出来的，feature 空间不天然等于 LLM 的 token embedding 空间：维度、数值尺度、语义组织方式都不同。这个落差叫 modality gap。两层 MLP 可以解决维度和尺度问题，却不自动解决信息取舍。

connector 可能承担四件事，当前 MLP 只做前两件：

- 维度变换；
- 尺度与分布校准；
- 时空压缩；
- 对语言生成有用的信息选择。

当前 projector 是 LN-Linear-GELU-Linear 结构：对每个特征 $f_i\in\mathbb{R}^{D_{in}}$ 独立计算 $e_i=W_2\,\mathrm{GELU}(W_1\,\mathrm{LN}(f_i))$，把 $D_{in}$ 维映射到 $D_{llm}$ 维。逐点计算意味着 $T_{out}=T_{in}$，序列长度原样透传。

`model/model_omni.py::MMVisionProjector` / `MMAudioProjector`（第 1 课第 6 节的表里确认过）；audio hidden 512、image hidden 768 定义在 `OmniConfig`。

逐点结构压不了序列长度，不用训练就能验证：打印任意样本的 connector 输入输出 shape，`Tout` 恒等于有效 `Tin`。想压缩就得换结构，这就是后面两个机制。

### 5.2 Cross-attention：派 query 去特征堆里打捞

与其把 64 个 patch 全部抬进 Thinker，可以派几个"专职记者"进场，各自带着关心的问题采访全场，把现场浓缩成固定几句话。learnable query 就是这些记者：一组数量固定的可训练向量，通过 cross-attention 读取全部特征。类比失效处：记者的问题是临场想的，query 的"提问方向"是训练时学死在参数里的，推理时对每个样例都用同一套。

在实现中，encoder feature 作为 `K,V`，可训练 query 参数产生 `Q`，attention mask 排除 padding（凑长度的空位）。

令 learnable latent 为 $Q$，encoder feature 为 $K,V$：

$$
\operatorname{Attn}(Q,K,V)=
\operatorname{softmax}(QK^\top/\sqrt d)V
$$

输出长度由 query 数决定，不随输入 patch 或 frame 数直接变化。压缩能力正来自这一条：输入 64 个还是 6400 个特征，输出都是 Q 行。

本课新增的 Resampler/Q-Former backend（接口见第 7 节）；padding 信息从 `feature_mask` 传入。

最容易翻车的是 mask 方向：写反后训练仍可能继续、loss 照样降，但 query 读的是 padding——这类 bug 不报错。Step 3 因此必须包含 padding invariance 单测：只改变 padding 长度时，有效输出应保持不变。

### 5.3 信息瓶颈：压缩省下的钱，谁在付账

query 数就是瓶颈宽度。挤掉的信息不挑软柿子：OCR 字符、时间顺序、小物体这类细节最先丢，而它们常常就是答案所在。

query 数减少会降低 `Tout` 和 prefill 成本（prefill 忘了回第 1 课），也可能丢失 OCR 字符、时间顺序和小物体信息。

比较 $Q=16$ 与 $Q=64$ 时，必须同时报告任务分数、`Tout` 和延迟，并按音频长度与 OCR 难度分桶。只报告平均分无法判断压缩损失发生在哪类样例——平均分纹丝不动，OCR 桶可能已经塌了。

query 数是 connector 的构造参数（`num_queries`，见第 11 节配置）；分桶统计在评测脚本里实现。

Step 8 的 query 扫描加长度/难度分桶就是本机制的验证实验；第 14 节的验收条件要求逐桶报告。

### 5.4 两阶段对齐：先让翻译上岗，再让脑子配合

connector 和 Thinker 一起训，结果变好了功劳算谁的？算不清。所以先冻住脑子只训翻译，翻译合格后再小范围放开脑子。


- 阶段 A：冻结 encoder/LLM，只让 connector 学会把模态映射到语言空间；
- 阶段 B：connector + LLM LoRA（低秩适配：不动原权重，只训一对小矩阵做微调）或小范围解冻，学习 instruction following；
- 直接全量联合训练可能把 connector 与 backbone 的变化混在一起。

上述两阶段流程是**本课的因果隔离设计**，不属于上游仓库既定 recipe。阶段 A 是判断 connector 本身是否有效的主实验：三臂都冻结 Thinker，只比较 connector。阶段 B 是阶段 A 完成并封存结果后的联合升级——Thinker LoRA 也会改变模型行为，其结果只能写成"connector + Thinker LoRA"的联合效果，不能回写成 connector-only 结论。

用可训练参数集说清归因边界：阶段 A 只更新 $\theta_{conn}$，阶段 B 更新 $\theta_{conn}\cup\theta_{lora}$；两个阶段的增量各自报告。

`trainer/train_sft_omni.py --mode` 当前只有 audio_proj / vision_proj 两种参数组，本课扩展到 connector 参数组（见第 6 节的表）；两阶段配置见第 11 节的 `stage_a` / `stage_b`。

阶段 A 失败时，先检查 connector、mask 和 marker。此时直接开启阶段 B 会同时改变 Thinker，再也说不清 connector 本身是否有效——这一步偷懒，主结论直接作废。

## 6. MiniMind-O 当前如何注入多模态特征

读代码时先记录 connector 输入和输出长度，再核对 prompt 中的 marker 数量。prompt 会在 connector 运行前预留位置（marker 先占坑，算好后逐个替换），任何长度不一致都会造成截断或 shape 错误：

| 文件/符号 | 当前行为 | 本课改造 |
|---|---|---|
| `OmniConfig` | audio hidden 512、image hidden 768 | 增加 connector type/query 数 |
| `MMAudioProjector` | LN-Linear-GELU-Linear，保持帧数 | 实现 audio connector registry |
| `MMVisionProjector` | 同上，参数虽有 token 参数但不压缩 | 实现视觉 Resampler/Q-Former |
| `encode_audio_inputs` | SenseVoice 输出逐帧投影 | 支持 mask 与固定/分段 latent |
| `encode_image_inputs` | 64 patch 逐 token 投影 | 输出 Q 个 latent |
| `inject_audio_features` | 替换连续 audio marker | marker 数应等于实际输出 token |
| `count_vision_proj` | 替换 image marker | 动态读取 connector 输出长度 |
| `OmniDataset.create_chat_prompt` | 按 feature 长度插 audio marker | 改为 connector length |
| `train_sft_omni.py --mode` | audio_proj / vision_proj | 扩展到 connector 参数组 |

当前 `audio_features_length` 来自 SenseVoice valid length。加入压缩后，必须在构造 prompt 前确定 connector 的输出 token 数，否则 marker 与 feature 数量对不上。对不上有两种死法：shape mismatch 直接报错，算你走运；更麻烦的是截断不触发异常，模型靠语言先验给出表面正常的答案。

## 7. 目标架构与统一接口

先实现统一接口，并让 `type=mlp` 也通过该接口运行。旧 MLP 和新 MLP 路径通过数值等价测试之后，再比较其他 connector。这样接口迁移错误和架构差异被分开，之后的数值异常能说清是搬家搬坏的还是新结构的问题：

```python
class ModalityConnector(nn.Module):
    def forward(
        self,
        features,          # [B, Tin, Din]
        feature_mask=None, # [B, Tin], True=valid
        timestamps=None,   # optional [B, Tin, 2]
    ):
        return {
            "embeddings": ...,  # [B, Tout, Dllm]
            "mask": ...,        # [B, Tout]
            "aux": {...},       # attention map / entropy
        }
```

实现三种 backend：

| Connector | 如何读取 encoder features | 输出长度 |
|---|---|---|
| MLP | 每个 `feature_i` 独立经过同一个投影网络 | 与有效输入 token 数相同 |
| Resampler | learned queries 通过 cross-attention 读取全部有效 features | 固定为 query 数 Q |
| Q-Former | learned queries 交替执行 self-attention 和 cross-attention | 固定为 query 数 Q |

对于长音频，本课设计一条分段支线：每 2 秒 SenseVoice feature 生成 `q_chunk` 个局部 latent，并为整段音频增加 1 个 global latent。

为什么分段？固定 32 个 query 处理 2 秒和 2 分钟的音频，压缩率差几十倍，长音频会被挤得面目全非，分段用于控制这个变量。它既非官方实现，也不预设为最终方案；留不留，由长度分桶结果决定。

## 8. 数据 recipe

语言先验和数据重复可能让 connector 在没使用模态时仍拿到高分。因此数据集除正样本外，还必须能构造 zero 和 shuffled 反事实。同图近重复或同一录音的切片不得跨 split。

### 8.1 来源

视觉：

- Pilot：官方 Image-to-Text（I2T）中 5k-10k 行；
- Standard：官方 `sft_i2t` + 冻结的 VQAv2/TextVQA 小验证集（两个公开视觉问答基准，后者偏重读出图中文字）；
- 强负例：颜色、计数、OCR、左右关系的人工最小集。

音频：

- Pilot：官方 Audio-to-Audio（A2A）中 5k-10k 行；
- Standard：官方 `sft_a2a_mini`；
- 诊断集：LibriSpeech dev-clean/test-clean（公开英文有声书语料，语音识别领域的标准考题）用于内容 probe；
- 语义 QA：从官方 A2A held-out 构造 exact/F1。

使用外部数据前逐项确认原始 license；课程不重新分发受限内容。

### 8.2 Schema

```yaml
sample_id: stable-id
modality: image | audio
instruction: string
target_text: string
image_bytes: bytes | null
audio_bytes: bytes | null
duration_ms: int | null
source: string
license: string
split: train | dev | test
paired_id: string
probe_tags: [ocr, count, color, speech_content, speaker, noise]
```

训练仍转换回 MiniMind-O parquet；此 sidecar（伴随主数据的补充记录文件）用于审计和评测。

### 8.3 预处理

- 图像保持官方 256×256，避免把分辨率变化混入本课；
- 音频统一 mono 16 kHz；
- encoder feature 可离线缓存，cache key 包含 encoder revision；
- 缓存 FP16 feature 时先验证与在线 FP32/BF16 的差异；
- 统计每个输入的 `Tin` 与 connector `Tout`；
- paired negative 通过同 batch 随机错配模态与文本生成。

### 8.4 切分

- 同图近重复不能跨 split；
- 同一原始录音的切片不能跨 split；
- speaker-disjoint（同一说话人不跨训练与评测，第 1 课的约定）音频 probe；
- OCR 文本模板去重；
- dev 用于选 query 数，test 只开一次。

### 8.5 三档规模

| 档位 | 图像/音频样本 | 训练目的 |
|---|---:|---|
| pilot | 各 2k-10k | 接口、过拟合、query sweep |
| standard | 官方单任务全量 | 三臂正式比较 |
| full | joint I2T+A2A | 验证 connector 能否共存 |

### 8.6 缺失数据边界

- 官方 mini 是英文、无视觉，不能拿它回答视觉 connector 的问题；
- A2A target 主要是 Mimi codes，但 connector 评测应先看 Thinker 文本；
- 没有 frame-level 标签时，attention map 只能做解释线索，不能当真值；
- 不把 ASR transcript 泄漏进 audio-only prompt。

## 9. 按依赖顺序执行实验

按以下顺序实验：先验证旧路径迁移等价、新 connector 能过拟合小数据，随后检查模型是否真的依赖输入模态，最后才比较任务收益与额外 token、延迟。前一项不通过时，不运行三臂全量。

### Step 1：建立 projector-only 基准

冻结 encoder、Thinker、Talker，只训练原始 MLP。对 128 对样本过拟合，记录能达到的最低 loss。这和第 1 课 Step 3 是同一招。

该结果是后续 connector 的最小功能基准。如果新结构无法过拟合同一批 128 对样本，先检查 mask、marker、shape 和优化器，不增加数据规模；128 对都背不下来，加数据只会把错误藏得更深。

### Step 2：实现统一 connector 与 mask

先让 `type=mlp` 走新接口。同 checkpoint、同 batch 对比旧/新路径的 embeddings 和 logits。这是迁移等价性检查点。它不通过时，先修接口，不进入架构比较。

### Step 3：实现 Perceiver Resampler

- 2-4 层 cross/self attention；
- query 数 16/32/64；
- attention mask 必须屏蔽 padding；
- 输出接 LayerNorm 和投影；
- 保存 query attention entropy（注意力分布的集中程度：熵高是平均扫全场，熵低是只盯少数几帧），不把它加入主指标。

entropy 可用于定位 query collapse（所有 query 学成一个样）或 padding 泄漏。没有 frame-level 标注时，attention map 只能说明权重分布，不能证明某个 query 已识别特定物体。

### Step 4：实现轻量 Q-Former

- learnable query；
- alternating self-attn/cross-attn；
- 不引入额外文本 encoder；
- 参数量通过 hidden/层数控制在 MLP/Resampler 的约束范围；
- 本课不照搬 BLIP-2 的全部预训练目标。

### Step 5：阶段 A 对齐

冻结 encoder 与 LLM：

- vision connector 只训 I2T；
- audio connector 只训 A2A 的 Thinker 文本目标；
- 固定 optimizer updates（epoch 长短随数据量变，updates 才可比）；
- 每 500 step 跑 paired/unpaired probe。

### Step 6：阶段 B 指令微调

各臂都开启相同 rank 的 Thinker LoRA。Talker 默认冻结，防止语音生成噪声掩盖 connector 结论。

开始阶段 B 前，先冻结阶段 A 的 checkpoint、dev/test 结果和统计脚本 hash。阶段 B 必须使用新的 experiment ID，并分别报告相对各自阶段 A checkpoint 的增量。任何阶段 B 收益都不能替代阶段 A 的三臂比较。

### Step 7：做"移除/打乱模态"负对照

先只在 dev 上运行三种条件：

1. 正确模态；
2. 全零模态；
3. batch 内错配模态。

定义模态依赖差：

$$
\Delta_{mod}=Score(x,m)-Score(x,\operatorname{shuffle}(m))
$$

若 $\Delta_{mod}\approx0$，高分可能来自语言先验。

逐 case 检查三种输入的输出。正确图与错图都答对时，样例可能可由语言先验解决；两者都答错时，应检查 connector 是否完成对齐。这两类样例要分别统计。

### Step 8：query 数和长度桶分析

仅在 dev 上对主臂分别扫描 Q=16/32/64。每个 Q 都对应一份独立训练配置和 checkpoint，不能在同一个已训练 checkpoint 上临时改 query 数——query 是训出来的参数。

音频按 `<3s`、`3-10s`、`>10s` 分桶；视觉按 OCR 字符数、小物体/计数分桶。

图像和音频各自选择 query 数，不用一个 Q 同时代表两种模态。选择完成后，把 connector 类型、Q、checkpoint、dev 指标和选择脚本的 SHA-256 写入 frozen config。

### Step 9：先回归，再冻结配置并一次性运行 test

跑冻结的 text/audio/image golden set：

- 有第 01 课产物时，复用其 golden set；
- 独立模式运行本课 Step 2 前冻结的 `connector-regression-v1`；
- 两种路径都必须包含同样的 zero/shuffled modality 反事实。

回归通过后，按 dev Pareto 选择图像 winner 和音频 winner，不以单一 benchmark 最高分决定。随后写出 resolved config、checkpoint hash 和唯一的 test run ID，再一次性运行 test 的 correct/zero/shuffled 三种条件。test 结果只用于最终报告；无论结果好坏，都不能再修改 connector、query 数、阈值或 winner。若要继续调参，必须建立新的 dev 实验版本，并保留当前 test 已经打开过的记录。

## 10. 三个对照实验组

三臂只改变 connector，用于比较信息选择机制。其余训练条件保持一致：

| 臂 | 连接器 | 输出 token | 参数约束 | 训练 |
|---|---|---:|---|---|
| A | 原始 2-layer MLP | 输入等长/64 | reference | Stage A 主归因；Stage B 联合升级 |
| B | Perceiver Resampler | 32 | A 的 ±5% 或另报 | Stage A 主归因；Stage B 联合升级 |
| C | 轻量 Q-Former | 32 | A 的 ±5% 或另报 | Stage A 主归因；Stage B 联合升级 |

若无法在 ±5% 内匹配参数，至少同时报告：

- connector 参数；
- trainable 总参数；
- FLOPs（浮点运算次数，衡量计算量）；
- `Tout`；
- global train token。

不要同时改变 encoder、图像分辨率或 backbone。

## 11. 配置示例

```yaml
experiment: lesson02_connector
base_checkpoint: hf://jingyaogong/minimind-3o
base_revision_or_sha256: ee3febbd08cc5b2bd41c039c825a8934232fee33
seed: 42
freeze:
  vision_encoder: true
  audio_encoder: true
  talker: true
connector:
  modality: vision
  type: perceiver_resampler
  input_dim: 768
  llm_dim: 768
  num_queries: 32
  layers: 2
  heads: 8
  dropout: 0.0
stage_a:
  train: connector
  updates: 10000
  lr: 5.0e-4
stage_b:
  train: connector+thinker_lora
  lora_rank: 16
  updates: 10000
  lr: 2.0e-5
```

音频分段配置：

```yaml
audio_compression:
  chunk_ms: 2000
  queries_per_chunk: 8
  global_queries: 1
  max_chunks: 16
```

## 12. 训练预算与 8 卡分配

| 模式 | 预算 | 分配 |
|---|---|---|
| pilot | 每臂 2k updates | GPU0-2 各一臂，GPU3 eval |
| standard | 每臂 10k-30k updates×3 seed | 8 卡跑 8 个独立 job，完成后补第 9 个 |
| full | winner 做 joint | 4-8 卡 DDP，保持 global batch |

113M 主体较小，建议每张卡运行一个独立的实验臂或 seed。只有单个任务无法放入一张卡或单卡时间不可接受时再使用 DDP（多卡数据并行，第 1 课提过会放大 global batch），并单独记录通信开销。

若使用 DDP：

$$
B_{global}=B_{device}\times N_{gpu}\times accumulation
$$

所有臂保持相同 $B_{global}$ 与 optimizer updates。

## 13. 评测数据与指标

### 13.1 视觉

- VQAv2/TextVQA 固定小切片；
- POPE（检查模型会不会"看见"图里其实没有的物体的公开基准）或人工 object-presence 子集；
- OCR exact/CER；
- count accuracy；
- paired image-text retrieval Recall@1 作为 representation probe。

### 13.2 音频

- LibriSpeech frozen probe WER（词错误率，量尺定义回第 1 课）；
- A2A held-out QA exact/F1；
- 噪声/速度增强鲁棒性；
- speaker-disjoint 分桶；
- 长度桶性能。

### 13.3 系统

- `Tout/Tin` compression ratio；
- connector wall time；
- Thinker prefill p50/p95；
- peak memory；
- tokens/s；
- accuracy per 100 input tokens。

### 13.4 表征依赖

$$
R_{use}=
\frac{Score_{correct}-Score_{shuffled}}
{\max(|Score_{correct}|,\epsilon)}
$$

同时报告 zero 与 shuffled；二者区分"无模态"和"错误模态"。

## 14. 保留复杂 Connector 的验收条件

运行 test 前先冻结以下判据，运行后不得根据个别高分样例修改选择规则：

- [ ] 新 MLP 接口与旧路径数值等价；
- [ ] 阶段 A 三臂固定 encoder、Thinker、Talker、数据和 updates；
- [ ] 阶段 A 的 connector-only 结果已在阶段 B 前封存；
- [ ] 阶段 B 使用独立 experiment ID，并明确标为 connector + Thinker LoRA 联合升级；
- [ ] 每臂至少 3 seed，或清楚标记 pilot；
- [ ] 正确模态显著优于 shuffled；
- [ ] 报告参数、FLOPs、Tout、prefill 与显存；
- [ ] 按音频长度/OCR 难度逐桶；
- [ ] test 未用于选择 query 数；
- [ ] 文本回归下降不超过预注册阈值；
- [ ] winner 位于 Pareto frontier；
- [ ] 即使 MLP 胜出也形成有效结论。

下面是课程建议的预注册起点，不是跨数据集通用定律。正式实验应根据 baseline 波动、样本量和业务容忍度写下自己的阈值：

- 主任务相对提升 ≥3% 或相同性能下 prefill 降低 ≥20%；
- text-only golden 回退 ≤1 个百分点；
- shuffled drop 不小于 baseline 的 90%。

## 15. 失败诊断表

使用下表时先根据症状选诊断项。例如 TextVQA 下降时，先检查 OCR patch 是否在压缩后丢失，再决定是否增加 query 数；不能靠加 Q-Former 层数把症状盖过去。

| 症状 | 原因 | 诊断 | 修复 |
|---|---|---|---|
| Resampler loss 不降 | query/feature 尺度不匹配 | 看 norm、attention entropy | pre/post LN，降低 LR |
| 所有 query 相同 | query collapse | pairwise cosine | query dropout/初始化调整 |
| padding 获得高注意力 | mask 方向错 | 可视化 masked logits | 单测 padding invariance |
| 音频短句好长句差 | 固定 Q 压缩过强 | 长度桶曲线 | chunked latent |
| TextVQA 大跌 | OCR patch 被压掉 | OCR 字符召回 | 增 Q 或保留 local token |
| shuffled 后仍高分 | 数据语言捷径 | counterfactual prompt | 重做 hard negative |
| Q-Former 胜出但很慢 | 层数/参数不公平 | profiler | 参数/FLOPs 对齐 |
| joint 后语音退化 | Talker 被无意更新 | 参数 diff | 冻结 Talker |
| marker 数错位 | prompt 仍用 encoder 长度 | 打印 marker/latent count | connector 先声明 output length |

## 16. 逐 case 分析

每个 case 至少保存：

- `case_id`；
- arm、seed 和 checkpoint；
- 输入模态 hash；
- `Tin` 与 `Tout`；
- question、target 和 prediction；
- correct、zero、shuffled 三种条件的预测；
- connector latency；
- attention summary；
- error taxonomy。

视觉错误标签：

- `ocr_loss`；
- `small_object`；
- `count`；
- `spatial_relation`；
- `language_prior`；
- `hallucination`。

音频错误标签：

- `phonetic_content`；
- `long_context_compression`；
- `noise`；
- `speaker_bias`；
- `endpoint_truncation`；
- `language_prior`。

报告里主动挑选：

- 10 个复杂连接器逆转 MLP 的 case；
- 10 个 MLP 反而更好的 case；
- 10 个三臂全错的 case；
- 10 个 shuffled 仍答对的泄漏 case。

## 17. 交付物

1. `ModalityConnector` 接口与三种实现；
2. 旧 MLP 数值等价测试；
3. 三臂配置与 checkpoint；
4. 数据/feature cache manifest；
5. Pareto 图；
6. paired/zero/shuffled 结果；
7. 逐 case 浏览树；
8. winner 选择报告；
9. text/audio/image 回归结果；
10. `connector-regression-v1` manifest 与 base 输出；
11. 可供后续课加载的 `connector-v1`。

## 18. 复现清单

- [ ] base checkpoint hash；
- [ ] 独立模式的 `connector-regression-v1` 已在训练前冻结；
- [ ] encoder revision/hash；
- [ ] feature cache key；
- [ ] query 初始化 seed；
- [ ] trainable 参数清单；
- [ ] 参数/FLOPs/Tout；
- [ ] optimizer updates 与 global batch；
- [ ] train/dev/test 去重报告；
- [ ] negative pairing seed；
- [ ] 所有 arm 推理参数；
- [ ] test 只打开一次；
- [ ] license sidecar 完整。

## 19. 前沿对照与改造方向

connector 这个问题，前沿走过一个来回。[BLIP-2](https://arxiv.org/abs/2301.12597) 用 32 个 learnable query 的 Q-Former 加两阶段训练（表示学习 + 生成学习）做对齐，是重连接器路线的代表；[Flamingo](https://arxiv.org/abs/2204.14198) 用 Perceiver Resampler 把变长视觉特征压成固定数量 token，以此支撑交错多图输入。随后 [LLaVA](https://arxiv.org/abs/2304.08485) 证明简单投影加指令微调就能对齐得不错，不少后继开源系统转向"轻连接器 + 强 backbone"。音频侧路线更散：[Qwen2.5-Omni](https://arxiv.org/abs/2503.20215)（第 1 课引过）给音频、视觉各配独立 encoder 再进 Thinker，与本课分工相同，其技术报告未公开 connector 层面的消融细节；[Moshi](https://arxiv.org/abs/2410.00037)（第 1 课引过）干脆绕开连续特征这条路，语音直接以 Mimi 离散 token 进主干。前沿没有统一答案——这正是本课用受控实验选型的理由。

encoder、backbone 的参数量和训练数据量差几个数量级，这是规模问题，钱能解决。机制问题才是本课教的：输出 token 数如何随输入伸缩、marker 与 mask 纪律、query 数怎么选、怎么证明模型真的用了模态——在 26M 上练会，搬到大模型照样用。还要承认：本课图像固定 256×256、只有 64 patch，可压缩的分子小；压缩收益要等第 8-10 课把输入放大后才兑现。



1. **local tokens + global queries 混合输出。** 纯 query 压缩最容易丢 OCR 字符，改为 Q 个全局 latent 拼上按规则保留的 K 个局部 token（均匀下采样或高注意力区域）。改动位置：第 7 节 Resampler backend 的 `forward`，`embeddings` 与 `mask` 同步拼接，marker 逻辑跟着 `Tout` 走不用另改。预算：pilot 档视觉臂 2k updates，单卡数个 GPU-hour。预期：OCR 难度桶分数向 MLP 靠拢，`Tout` 仍明显低于 64。失败判定：OCR 桶不回升，或 K 加到接近 64——那等于绕回 MLP，方案作废。
2. **给 Resampler 加时间戳/2D 位置编码。** 接口的 `timestamps` 字段本就预留。音频把每帧起止时间编码进 K/V，视觉给 patch 加 2D 位置编码。改动位置：Resampler backend 的 K/V 构造处，几十行。预算：pilot 档音频、视觉臂各 2k updates。预期：音频 `>10s` 桶与左右关系强负例改善，attention entropy 分布改变（query 各管一段）。失败判定：两处均无差异，说明当前任务规模对时序/空间不敏感，记录后撤销改动。

**更多顺手扩展**（课后自选）：

- 分别为图像和音频搜索不同 query 数（Step 8 的延伸，扫更细的网格）；
- 做 encoder 最后 1-2 层解冻，但作为新实验单独立项，不混入三臂主结论；
- 用 CKA/linear probe（两种衡量表征相似度与可读性的探针方法）比较三种 connector 学出的表征；
- 把 winner 接入[第 08 课](08_dynamic_vision.md)的动态分辨率输入，重新检验 token budget；
- 研究图像与音频共用一个 shared connector 是否造成跨模态负迁移。

其一，[Perceiver](https://arxiv.org/abs/2103.03206) 的核心卖点"计算量与输入长度解耦"可在测量层面复现：对不同时长音频记录 connector wall time 与 Thinker prefill，MLP 臂随时长线性上涨，Resampler/Q-Former 臂因 `Tout` 固定基本持平。这是结构性质，预期必能复现；复现不出先查计时脚本。其二，BLIP-2/Flamingo 的"固定少量 query 保住大部分任务分数"预期只能部分复现：通用问答桶方向一致（Q=32 接近 MLP 等长成绩），OCR/小物体桶大概率复现不出——26M backbone、64 patch，可压缩冗余本来就少。若 Q=16 与 Q=64 全桶无差异，先怀疑分桶没拉开难度，再谈"压缩免费"的结论。

## 20. 必读论文与阅读问题

按 LLaVA、Perceiver/Flamingo、BLIP-2 的顺序读，连接器由轻到重。每读完一个结构，都在本课第 7 节的接口上标出它改变的是 `Tin`、`Tout`、参数量还是训练目标，并写出对应的实现位置——写不出就是还没读懂。

### 20.1 简单 projector

- [LLaVA](https://arxiv.org/abs/2304.08485)：
  读 §3 Approach，尤其 feature alignment 与 visual instruction tuning 两阶段。带着问题：它的两阶段与本课阶段 A/B 如何对应？阅读后写出简单线性 projection 能完成哪些对齐、不能完成哪些压缩，对照 5.1 节的四件事标出 LLaVA 做了哪几件。
- [MiniMind-O](https://arxiv.org/abs/2605.03937)：
  读 multimodal input 与 projector 部分。带着问题：当前 projector 优化的是维度对齐还是 token 压缩？答案要能在第 6 节的表里指认——`MMVisionProjector` 带着 token 参数却不压缩，这个设计给本课的改造留了什么口子？

### 20.2 Perceiver / Flamingo

- [Perceiver](https://arxiv.org/abs/2103.03206)：
  读 latent bottleneck 与 asymmetric attention。带着问题：cross-attention 的计算量里，输入长度与 latent 数各出现在哪一项？阅读后写出计算复杂度随两者变化的公式，并拿第 19 节"顺手复现"的 wall time 实测曲线去对。
- [Flamingo](https://arxiv.org/abs/2204.14198)：
  读 §2.1 Perceiver Resampler 和 gated cross-attention。带着问题：固定视觉 token 数在交错多图场景换来了什么、丢掉了什么？阅读后写出这笔账，并预测它对应第 15 节失败诊断表里的哪几行（提示：OCR 与小物体）。

### 20.3 Q-Former

- [BLIP-2](https://arxiv.org/abs/2301.12597)：
  读 §3，重点 Q-Former、representation learning 与 generative learning。带着问题：本课只复刻结构、不复刻完整预训练目标，隔离掉的变量是什么？阅读后写出：若三臂里 Q-Former 输了，哪些原因归"缺预训练目标"，哪些归结构本身。
- [BLIP-2 官方仓库/LAVIS](https://github.com/salesforce/LAVIS)：
  查看 Q-Former 配置和 mask 写法，对照 Step 3 的 padding invariance 单测；不直接复制没理解的训练代码。

读完材料回头看：系统的眼睛和耳朵还是原装的 SigLIP2 和 SenseVoice，但它们通向脑子的那根线已经换成带契约的可插拔接口——赢家 connector 凭 Pareto 证据入选，输家的完整数据也留档在报告里。下一课把刀移到声音的另一端：[第 03 课](03_audio_codec.md)拆的是 Mimi，同样先立统一的 `AudioCodec` 契约，先测 codec 自己的重建质量，再同预算重训 Talker 做端到端公平比较。你会再次撞见本课的教训：单项指标最好的部件，接进系统后不一定赢。
