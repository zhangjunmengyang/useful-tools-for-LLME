---
id: 01_baseline_reproduction
title: "冻结可信基线"
summary: "把代码、数据、种子和推理参数全固定之后，重跑能得到同一条 Thinker–Talker（想的脑、说的嘴）链路吗？每个 token 和每个延迟数字，你都说得清来历吗？"
unit: mechanism
play_tools: []
checkpoints:
  - "亲手画出 1 路文本和 8 路 Mimi code（声音压成的整数编号）的训练、delay 和还原时间线。"
  - "拿到一条生成 case，能顺藤摸瓜查回它的数据行、配置、checkpoint、随机种子和有效 loss 区间。"
  - "分清 TTFT、TTFA、RTF 和端点检测这几种延迟各测什么，还能证明记运行日志不会改变 logits。"
  - "做出后面十九课都要反复用的 baseline-v1、golden cases 和回归检查。"
---

# 第 01 课：建立可追溯的 MiniMind-O 基线

> 内容：复现、测量与实验基础设施<br>
> 建议周期：2-4 天<br>
> 硬件：1×24GB 可完成 mini；8 卡用于并行运行 seed，不改变算法<br>
> 产物：`baseline-v1` checkpoint、golden cases、结构化 trace、复现报告

## 1. 从可复现开始

普通语言模型接收并生成文本。Omni 模型还要处理图像和声音，并直接生成语音，而不是把文本回答交给独立的朗读软件。

系统先把不同模态转成模型可以处理的 token 或向量。文字经过 tokenizer，图片经过视觉编码器，声音经过 codec（把声音压成整数编号，也能把编号解回声音）。模型在混合序列上预测后续 token；生成的声音编号再由 codec 解码成波形。

MiniMind-O 是一个教学规模的实现：可训练主体约 26M 参数，一张消费级显卡约两小时可以跑完官方 mini 训练。模型包含 8 层 Thinker、4 层 Talker，以及三个冻结模块：SenseVoice 处理音频输入，SigLIP2 处理图像，Mimi 编解码语音。当前推理按完整回合运行，不能在生成过程中接收新的用户输入。

这门 20 课是一个连续的大项目：把这个回合制玩具改造成能实时插话、能看视频、用上现代架构、经过强化学习、最后接上大模型脑子的现代 Omni 系统。六幕剧本：

| 幕 | 课 | 一句话 |
|---|---|---|
| 第一幕：拆开看懂 | 01-04 | 把一个能听会说的最小模型跑起来，拆开，搞懂声音怎么变 token、token 怎么变声音 |
| 第二幕：学会插话 | 05-07 | 从"你说完我再说"改造成"边听边想边说"：流式听、判断该不该开口、全双工调度 |
| 第三幕：长出眼睛 | 08-10 | 从固定尺寸看图，到任意分辨率、原生视频，再把视频 token 压到算力扛得住 |
| 第四幕：换强心脏 | 11-14 | 给模型换现代内核：MoE 稀疏专家、Mamba 混合架构、8 卡并行、长上下文 |
| 第五幕：教它变聪明 | 15-17 | 训练方法升级：多模态联合 SFT、偏好优化、GRPO 可验证奖励强化学习 |
| 第六幕：毕业设计 | 18-20 | 把现代大脑（Nemotron）接到自己的嘴（Talker）上，做出真双工系统；选修：理解与生成统一 |

第一课不改模型结构，先建立复现、测量和取证流程。后续课程会更换 connector、流式机制和骨干架构；每次比较都需要同一个基线 checkpoint、固定的评测协议，以及可以回查的 trace 和 golden case。缺少其中任何一项，loss 或听感的变化都可能来自 voice prompt、采样温度或 ASR 版本，而不是模型改动本身。

做完这课你手里会有：从头训出、每个文件都有指纹（hash）的 `baseline-v1`；100 个冻结的 golden case，一条命令重跑；一套不改变计算结果的 trace。任何可疑输出都能查到来自哪行数据、mask 盖了哪里、吐了哪 8 路编号、用了什么解码参数。

本课术语：

| 术语 | 简要解释 |
|---|---|
| Thinker | 模型的脑子：8 层小语言模型，理解各模态输入，想好要说的话 |
| Talker | 模型的嘴：4 层小模型，把 Thinker 的想法翻译成声音编号 |
| codec / Mimi | 音频压缩器：声音压成编号、编号解回声音；Mimi 是本课用的，始终冻结 |
| 码本（codebook） | codec 的编号字典：每个编号对应一小段声音特征；每帧查 8 本得 8 个编号 |
| RVQ | 残差向量量化：8 本码本逐层"找零钱"式逼近原声，先记大概再补差额 |
| loss mask | 标出哪些位置参与算 loss 的 0/1 表：答案算数，提示词和 padding 不算 |
| teacher forcing | 训练时永远拿正确答案前缀当历史，不管模型自己预测了什么 |
| TTFT / TTFA | 从发请求到看见第一个字（TTFT）、听见第一段声音（TTFA）的时间 |
| trace | 训练和推理时记下的结构化日志（loss 分项、梯度、mask 统计），出问题时回放 |
| golden case | 固定的测试样例和判定标准，每次改完拿它回归，防止改好 A 坏了 B |

## 2. 本课解决的问题

训练脚本成功退出、loss 在降、WebUI 能出声——这只证明程序跑通了，不证明两次运行用的是同一个系统。换一台机器后，生成内容或 TTFA（请求到首个可播放 PCM 的时间；PCM 是可直接播放的原始音频）变了，原因可能在模型，也可能在 voice prompt、随机采样、ASR（自动语音识别）版本或计时终点。

本课为这些变量建立可复查记录。一次生成依次保存：数据行、9 路训练序列（8 路音频编号加 1 路文字）、Thinker hidden state（中间层的向量表示）、Talker 生成的 8 路 code、Mimi 解码的 waveform，以及 ASR、听感和时间轴结果。任何结果都应能沿这条处理顺序回查。

复现成功要同时满足两条：重新运行时使用相同的代码提交、数据、配置、seed（随机种子）和推理参数；每个文本 token、8 路 Mimi code 和延迟数字都能追溯到对应的处理步骤。

先划清能力边界。standard lane 复现的是官方 `_mini` 快速训练链路，不是发布版 `minimind-3o` 的完整能力。官方 README 说明 mini 仅含英文、无视觉；发布权重则来自包含中英文 Text-to-Audio（T2A，文字进语音出）、Audio-to-Audio（A2A，语音进语音出）与 Image-to-Text（I2T，图片进文字出）的 full 数据。两者 recipe 不同，不能把性能差值直接记为本课的复现误差。

出现以下任一情况时，不得把结果标记为可信基线：

- loss 能下降，但恢复训练后轨迹明显改变；
- 生成音频听起来正常，却无法说明 loss mask 覆盖了哪里；
- WebUI 能"打断"，但没有区分 VAD 事件（VAD：检测有没有人说话的小模型）、生成取消和模型行为；
- 两次评测使用了不同 voice prompt、temperature 或 ASR；
- 只记录总 loss，没有按模态、码本、长度桶拆开。

执行顺序固定：先让 128 个样本过拟合，再运行完整 mini；先证明 trace 不改变 logits（softmax 之前的原始得分），再用 trace 定位失败。

## 3. 开始前需要准备什么

本课可从一台空白机器开始，不依赖后续课程。开始前在实验记录里分开写明上游事实和本课约定，报告不得混用——事实是查出来的，约定是你定的，混着写没法审计。

**上游事实：**

- 代码：[jingyaogong/minimind-o](https://github.com/jingyaogong/minimind-o)；
- 初始权重：官方 `llm_768.pth`；
- 数据：`sft_t2a_mini.parquet`、`sft_a2a_mini.parquet`；
- 外部冻结模块：SenseVoice-Small、SigLIP2、Mimi、CAM++（说话人声纹模型）；

**课程约定：**

- 记录实际 commit SHA（代码仓库精确版本的指纹），不用会移动的 `master` 充当版本号；
- 输出 checkpoint：`baseline-v1`，其完整名称是
  `minimind-o-mini-mechanism-baseline-v1`。

只做本课评测不训练时，可从官方 `minimind-3o` checkpoint 开始；报告中标记为 `official-eval-only`。这条支线测的是官方权重，不代表你复现了训练。

## 4. 完成后应具备的能力

完成后，拿到任意异常样例，你应能完成以下检查：

1. 画出 9 路训练序列：8 路 audio code + 1 路 text；
2. 解释 Thinker、middle-layer bridge、Talker 的信息流；
3. 精确指出文本和音频 loss 的有效区间；
4. 解释 Mimi delay schedule 如何恢复同一音频帧；
5. 区分 TTFT、TTFA、RTF、endpoint latency；
6. 从任一生成样例追到数据行、配置、checkpoint 和随机种子；
7. 证明 trace 插桩没有改变 logits；
8. 形成后续课程统一的回归门禁。

## 5. 原理:边造边讲

五个机制，每个按同一节奏：为什么需要（直觉）、怎么运转（机制）、精确定义（数学）、在源码哪里（代码）、怎么证明做对了（验证）。

### 5.1 Causal language modeling：只对该学的位置算账

一条训练序列里混着 system prompt、用户问题、助手回答、凑长度的 padding。要学的只有一件事：给定前文，接出助手部分。把用户问题也算进 loss，模型会分心学模仿提问，最坏情况学会复读。所以需要一张表标出哪些位置的预测错误算数——这就是 loss mask。

每个位置的 label 要么是真实 token id（参与 loss），要么是 `-100`（PyTorch CrossEntropyLoss 约定跳过的值）；label 不为 `-100` 的位置组成集合 $M_t$。

对序列 $x_{1:T}$，训练目标是：

$$
\mathcal{L}_{text}=-\frac{1}{|M_t|}
\sum_{t\in M_t}\log p(x_t\mid x_{<t})
$$

集合 $M_t$ 只能包含 assistant target；system、user、padding 与条件区域的 label 应为 `-100`。

mask 在 `dataset/omni_dataset.py::OmniDataset` 里构造，loss 在 `trainer/train_sft_omni.py::train_epoch` 里计算。实现时先分别统计每类 token 的有效数量，再检查总 loss。若 mask 错误，即使 loss 正常下降，模型也可能只是在复制用户输入。

更换 user prompt 后，若输出几乎不变，或 audio loss 始终为 0，先打印 mask 和各类有效 token 数。确认 mask 正确后再调整学习率。

### 5.2 Teacher forcing 与 exposure bias：教练什么时候松手

teacher forcing 好比学车时教练全程扶着方向盘：你永远沿正确路线走，代价是没练过"开偏了怎么救"。推理时没有教练，模型读的是自己上一步生成的 token，一步错可能步步错。类比失效处：教练松手是突然的，scheduled sampling 可以按概率逐步松手。

训练时模型读真实历史 token；推理时读自己已采样的 token。两种输入分布的差异称为 exposure bias。MiniMind-O 的 scheduled sampling 会按设定概率把少量目标历史替换成模型自己的预测。本课固定并记录该概率，不在基线复现中调整——调参是后面课的事。

训练最大化 $\log p(x_t\mid x_{<t})$，历史 $x_{<t}$ 来自数据集；推理时模型实际面对的是 $p(x_t\mid \hat{x}_{<t})$，其中 $\hat{x}_{<t}$ 是它自己逐步采样出来的。训练中从未出现带错误的历史，没有机制约束错误累积，序列越长偏得越远。

scheduled sampling 的概率是 `OmniDataset` 的构造参数，上游默认值 `0.05`，原训练入口没有命令行开关；推理侧读自身输出的位置在 `MiniMindOmni.stream_generate`。

若短句正常而长句出现重复或提前停止，同时检查逐码本 loss、stop rate 和生成长度。前者可提示自回归误差累积，后两项可提示 stop target 错位；仅凭听感无法区分。

### 5.3 RVQ 与多码本：8 本码本轮流找零

一本 2048 个编号的字典描述一帧声音太粗，细节全丢。RVQ（residual vector quantization，残差向量量化）像找零钱：第 0 本码本先给粗略近似，第 1 本专门量化残差（没记准的差额），如此叠 8 层，越深的码本记的越是细节。类比失效处：找零的面额是固定的，而每层码本的"面额"是训练学出来的。

Mimi 每个时间帧给出 8 个 codebook index。数据文件里这 8 路编号被拍平成一维数组，用之前先还原成 `[8, frames]`：8 行，每行是一路码本的时间序列。构造生成目标时还要加 diagonal delay（对角错位）：粗码本先出、细码本后出，预测细节时能参考同一帧已定的粗轮廓。

diagonal delay 让第 $q$ 路相对第 0 路右移 $q$ 个时间位置；构造因果语言模型 target 时，所有码本再共同增加一位 next-token shift，因此第 $q$ 路的 target 起点偏移是 $q+1$。

delay target 的构造在 `dataset/omni_dataset.py::OmniDataset`；推理侧的 delay 与 audio frame 重排在 `MiniMindOmni.stream_generate`。

Step 2 要把 target 导出为 $8\times T$ 表格，逐行核对第 $q$ 行偏移是否为 $q+1$。若某一行整体偏移一个位置，对应码本的 loss 通常会异常升高。此时即使个别波形仍可辨识，也不表示调度正确。

### 5.4 Thinker-Talker bridge：脑子想的怎么递给嘴

为什么不让 Thinker 直接输出声音编号？分工：语义和发音是节奏、难度都不同的两种活，Thinker 想内容，Talker 念出来，中间靠 bridge 传话，传的是中间层 hidden state。

Thinker 的中间层 hidden state 经 `talker.embed_proj` 变换后，作为 Talker 的语义条件。Talker 还读历史 codec embedding，经 4 层因果 block 预测 8 路 code。为什么取中间层？带着这个问题去第 20 节读论文。

设 bridge 层输出为 $h^{(l)}$（本课配置 $l=3$），条件为 $c=\mathrm{proj}(h^{(l)})$。每个生成位置由 8 个 head 各自给出一路编号的分布 $p(a_t^{(q)}\mid a_{<t},c)$，$q=0,\dots,7$；由于 5.3 的 delay，同一位置 8 个 head 预测的编号分属不同时间帧。

`model/model_omni.py::OmniConfig` 里的 bridge layer 参数；`TalkerEmbedding` / `TalkerHead`（shared base + per-codebook adapter）；注入路径在 `MiniMindOmni.forward`。

检查 grad norm 时，如果 Talker 参数在更新而 bridge 梯度长期接近零，应先检查 bridge 的注入位置和 loss 路径，不能直接把发音错误归因于 codec。

### 5.5 延迟定义：秒表按在哪里

"响应快不快"在说清秒表起止点之前没有意义。最容易翻车的是终点：模型算出第一个 codec logit 时，用户耳朵里还什么都没有——logit 要变成完整 code frame、解码成 PCM、送进播放器，声音才存在。

统一使用以下定义：

- `TTFT`：请求到首个可显示文本 token；
- `TTFA_server`：请求到服务端产出首个可播放 PCM chunk；
- `TTFA_e2e`：请求到该 PCM chunk 已进入播放器队列；
- `RTF = 生成墙钟时间 / 生成音频时长`；
- `xRT = 生成音频时长 / 生成墙钟时间 = 1/RTF`；
- 峰值显存使用 `torch.cuda.max_memory_allocated()`，每个 case 前 reset。

RTF 小于 1 表示生成比播放快，流式跟得上；大于 1 则越播越欠。

计时打点在 Step 7 的 timeline 里实现；播放链路的 SSE/WS（服务端推流的两种方式）与 PCM streaming 在 `webui/web_demo.py`。

不能用 `first_codec_frame` 代替任何一种 TTFA。codec logit 和完整 codec frame 都不是可播放音频。本课把 `TTFA_e2e` 作为主指标，同时报告 `TTFA_server`，两者之差就是 PCM 从产出到入队的传输排队时间。无播放器的离线评测只能报告 `TTFA_server`，不能把它写成端到端 TTFA。

## 6. 在 MiniMind-O 源码中定位这些机制

读代码别按文件顺序，按一条样例的实际执行路径，依次定位下表中的符号。结论以实验记录中的 upstream SHA 为准，不用可能已变化的默认分支：

| 文件/符号 | 需要确认的事实 |
|---|---|
| `model/model_omni.py::OmniConfig` | 8 码本、2112 vocab、bridge layer、64 image token |
| `MMAudioProjector` / `MMVisionProjector` | 都是 LN-Linear-GELU-Linear |
| `TalkerEmbedding` / `TalkerHead` | shared base + per-codebook adapter |
| `MiniMindOmni.forward` | 9 路输入拆分、特征注入、Thinker 与 Talker |
| `MiniMindOmni.stream_generate` | text sampling、delay、stop、audio frame 重排 |
| `dataset/omni_dataset.py::OmniDataset` | parquet schema、mask、delay target |
| `trainer/train_sft_omni.py::train_epoch` | text/audio/aux loss 与 stop-token 10×权重 |
| `eval_omni.py` | 离线生成与 Mimi decode |
| `webui/web_demo.py` | SSE/WS、PCM streaming、VAD 近双工 |

这些是**官方实现事实**，不是本课准备证明的假设：

- SenseVoice、SigLIP2、Mimi 默认冻结；
- 训练主体约 113M，不等于运行时总加载参数；
- 用户语音当前整段进入 SenseVoice，不是流式 listener；
- `RealtimeSession` 是外部 Silero VAD 状态机；
- WebUI 的 "interrupt" 是停止当前生成，不等于模型边说边理解新语音。

## 7. 基线目标架构

信息流按行读，每行是一条处理路径：

| 输入 | 处理路径 | 输出或去向 |
|---|---|---|
| text ids | 直接进入 8 层 Thinker | text head |
| 16 kHz speech | SenseVoice → audio MLP → Thinker | Thinker hidden state |
| image | SigLIP2 → vision MLP → Thinker | Thinker hidden state |
| Thinker 中间层 hidden | bridge projection | 4 层 Talker 的语义条件 |
| Mimi history、speaker/reference | Talker embedding | 8 个 codebook head |
| 每帧 8 个 code | 冻结 Mimi decoder | 24 kHz PCM |

本课不改动该架构，只在各模块边界增加可关闭的观测点。观测点关闭时不应产生额外状态；开启时不应改变随机数消耗、logits 或最终输出。

## 8. 数据 recipe

### 8.1 来源选择

- Pilot：官方 `_mini` 两个 parquet（列式数据表文件）；
- Standard：官方 mini 全量，严格复用官方三阶段顺序；
- Full scale track：官方 `sft_t2a`、`sft_a2a`、`sft_i2t`，
  只在资源与数据均具备时另开实验；
- Golden eval：仓库 `dataset/eval_omni/` + 自建冻结 case。

`standard` 与 `full scale track` 使用不同 experiment ID、manifest 和报告。除非 full 数据、训练顺序、初始化和评测协议均与发布 recipe 对齐，不得把 standard mini checkpoint 与官方 full checkpoint 的差异解释成"复现误差"。

官方 README 报告的 mini 规模约为：

- T2A assistant audio：470.14 小时；
- A2A question audio：74.64 小时；
- A2A answer audio：56.60 小时。

这些是语音时长，不是独立样本数；manifest 还要另行统计行数。

### 8.2 训练数据字段

```yaml
sample_id: stable-string          # 新增，hash(source,row)
conversations:                    # parquet 中为 JSON string
  - {role: user, content: "..."}
  - {role: assistant, content: "..."}
question_audios: [bytes]           # 原始用户波形；可为空
answer_audios: [int, int, ...]     # 8 路 Mimi code 交错展开
image_bytes: [bytes]               # 可为空
ref_audios: [int, int, ...]        # voice reference Mimi codes
spk_emb: [float x 192]             # CAM++ 条件
source: string                     # 审计 sidecar 中补充
license: string                    # 审计 sidecar 中补充
```

### 8.3 预处理与切分

1. 在训练前生成只读 `manifest.jsonl`（每行一条 JSON）；
2. 对每个 binary 字段存 SHA-256，不复制内容；
3. 对文本、question audio、answer codes 分别去重；
4. speaker 相关评测按 speaker-disjoint 切分（同一说话人不同时出现在训练和评测）；
5. train/dev/test 建议 98/1/1，golden set 永不进入训练；
6. 统计长度、语言、speaker、模态组合和缺失字段；
7. 拒绝 code 数不能被 8 整除的样本；
8. 检查每路 code 是否处于 `[0, 2047]`，特殊 token 另记。

### 8.4 三档规模

| 档位 | 数据 | 用途 |
|---|---|---|
| pilot | 各任务 2k 行或约 2 GPU-hour | schema、过拟合 128 样本、trace |
| standard | 两个 mini parquet 全量 | 冻结 `baseline-v1` |
| full | 官方 full 三任务 | 只用于验证趋势是否扩展 |

### 8.5 License 与缺失边界

- 代码 Apache-2.0 不自动覆盖所有训练语料；
- 数据 manifest 逐来源记录 license/terms；
- 缺少原始 assistant waveform 时，不能补做 codec 公平比较；
- 合成 TTS 数据要记录合成模型及其使用条款；
- 无法追溯来源的行标成 `unknown`，不得发布衍生数据。

## 9. 按依赖顺序执行实验

以下步骤按顺序执行。环境没有固定时，不开始数据实验；128 个样本无法过拟合时，不运行完整 mini；trace 会改变 logits 时，不使用 trace 分析模型。

### Step 1：冻结环境

先记录本次运行的唯一标识和完整环境信息：

```bash
git rev-parse HEAD
```

```bash
python -V
```

```bash
pip freeze
```

```bash
nvidia-smi -q
```

记录 CUDA、PyTorch、Transformers、FunASR、GPU 型号、驱动和 hostname。

### Step 2：先加入结构化 trace，并证明它不改变计算

每 N step 输出 JSONL：

- `text_valid_tokens`；
- `audio_valid_tokens_by_q[8]`；
- 固定 parity batch 的 `target_ids_by_q[8,T]` 与 `loss_mask_by_q[8,T]`；
- `loss_text`、`loss_audio_by_q[8]`、`aux_loss`；
- `grad_norm/{audio_proj,vision_proj,thinker,talker,head_q}`；
- `audio_stop_rate_by_q`；
- `tokens_per_second`、`allocated_gib`、`reserved_gib`。

trace 默认关闭。开启后不得改变 forward 的张量和随机数消耗。对同一 batch 保存 trace off/on 的 logits，并用明确的数值公差比较，不能只比较最终文本或音频。

先在一个固定 batch 上完成下列检查，再开始任何训练：

1. trace off 连续运行两次，得到当前硬件和精度下的自然数值漂移；
2. 恢复同一 RNG state（随机数生成器的内部状态），分别运行 trace off/on；
3. 比较 text logits、8 路 audio logits、loss 和采样前 RNG state；
4. bitwise 不可用时，按第 8 步的方法预注册 `atol/rtol`；
5. parity 失败时停止，不能用该 trace 记录后续训练。

三阶段训练从第一个 step 起就有可用 trace，而不是训练结束后才补插桩。

### Step 3：做 128 样本过拟合测试

全量训练前。分别选 T2A、A2A，各固定 128 行。关闭 shuffle、augmentation 和 scheduled sampling，训练到 loss 明显接近下限。上游 scheduled sampling 默认 `0.05` 且无命令行开关（见 5.2）；本课的 overfit 配置必须显式把构造参数设为 `0.0`，不能只在报告里写"已关闭"。如果不能过拟合，先查 mask、code delay、权重加载，不进入 full mini。

该测试只检查数据、mask、forward、loss 和优化器能否共同工作，不评估泛化能力。128 个样本都背不下来，说明实现有错；扩大数据只会把错误藏得更深。

### Step 4：复现 mini 三阶段训练

按官方顺序：

1. T2A，从 `llm_768.pth` 初始化；
2. A2A，只训练 `audio_proj`；
3. A2A，全量训练。

每个阶段保存 optimizer、scaler、epoch、step 和 RNG state。standard 复现使用的 scheduled-sampling 概率也必须写入 resolved config；它与第 3 步 overfit 的 `0.0` 是两个不同配置。

### Step 4B：可选 full scale track

这不是 standard lane 的第四个主实验臂。资源允许时另开 `lesson01_full_scale_track`，按官方数据流执行：

1. `sft_t2a`：训练文本到语音链路；
2. `sft_a2a`：先以 `audio_proj` 模式对齐语音输入；
3. `sft_a2a`：再按官方 full 配置训练完整可训练主体；
4. `sft_i2t`：最后以 `vision_proj` 模式对齐视觉路径。

full track 必须记录实际官方配置、数据 hash 和每阶段起点。只有该 track 才可与发布权重做同 recipe 的能力比较；数据或配置不完整时，结论只能写"基于公开 recipe 的部分复现"。

### Step 5：实现 deterministic eval

固定：

- seed=42；
- `temperature=0` 或 greedy；
- `top_p=1`；
- voice/ref condition；
- max tokens；
- 同一个 ASR 模型和 normalization；
- 同一设备精度。

保留采样评测，但另设 `sampling-seed`，不与确定性回归混合。

### Step 6：建立 100 个 golden cases

standard mini 建议：

- 25 英文 T2A；
- 25 英文 A2A，覆盖口音、速度、噪声；
- 10 中文 T2A/A2A smoke；
- 10 I2T（图像到文本）视觉理解 smoke；
- 15 voice prompt；
- 15 长回答、异常输入和 stop 边界。

I2T 只表示图像到文本；若继续经过 Talker 输出语音，应另记为 `image→text→audio` 端到端 smoke。每个 case 保存输入、期望语义、允许变体、禁止行为和人工备注。其中中文与视觉 case 对 standard mini 只用于记录能力边界和接口回归，不进入"mini recipe 是否复现成功"的质量判定。full scale track 另建中英文与视觉正式 golden 分桶。

### Step 7：测量推理时间轴

使用 `time.perf_counter_ns()` 依次记录 `request_received`、`encoder_start`、`encoder_end`、`prefill_end`（prefill：整段输入一次算完的预填充阶段）、`first_text_token`、`first_codec_frame`、`first_pcm_chunk`、`playback_enqueued` 和 `generation_end`。这些字段必须来自同一个请求的单调时钟，不能把服务端和浏览器的墙钟直接相减。

分别计算：

$$
TTFA_{server}
=t_{\text{first\_pcm\_chunk}}-t_{\text{request\_received}},
$$

$$
TTFA_{e2e}
=t_{\text{playback\_enqueued}}-t_{\text{request\_received}}.
$$

主表使用 `TTFA_e2e`。若还测到客户端真正开始播放的时间，可另报 `playback_started - request_received`，但不要覆盖上述两个字段。

### Step 8：恢复训练与插桩等价性

- 在 step K 保存，恢复后跑到 K+100；
- 同硬件同精度比较 loss 曲线；
- 用相同 batch 比较 trace off/on logits；
- 确定性同 kernel 路径优先要求 bitwise equality；
- 若底层 kernel 非确定，先用 trace-off 重复运行估计自然漂移，
  再预注册 `atol` 与 `rtol`，用 `torch.testing.assert_close` 检查；
- 公差必须同时包含绝对与相对项，并随 dtype、logit 尺度和设备记录，
  不使用固定的 BF16 `1e-5` 绝对阈值。

### Step 9：冻结基线

生成：

```text
artifacts/baseline-v1/
  checkpoint/
  config.yaml
  data_manifest.jsonl
  environment.lock
  train_metrics.jsonl
  eval_metrics.json
  cases/
  report.md
```

只读保存并记录 SHA-256。它就是后面 19 课所有比较的参照物。

## 10. 受控实验矩阵

最多三臂，本课不比较新架构：

| 臂 | 权重 | trace | 目的 |
|---|---|---|---|
| A | 官方 full checkpoint | off | 外部能力参考，不参与 mini 质量复现 |
| B | 自复现 mini `baseline-v1` | off | mini 机制基线 |
| C | 同 B | on | 验证插桩零扰动 |

本课主工程结论只比较 B/C。A 用于展示发布模型行为和能力上界，不得与 B 做"复现质量"或数据效率结论。full scale track 若执行，在独立报告中与 A 比较，不并入本三臂矩阵。

## 11. 推荐配置

```yaml
experiment: lesson01_baseline
upstream_sha: a10fa6c148ed274d66f96dc119689e93e01be823
seed: 42
precision: bf16
model:
  hidden_size: 768
  thinker_layers: 8
  talker_layers: 4
  bridge_layer: 3
  audio_codebooks: 8
  audio_vocab_size: 2112
data:
  stage1: sft_t2a_mini.parquet
  stage2: sft_a2a_mini.parquet
  stage3: sft_a2a_mini.parquet
  source_revision: d6588e12ac2ac8ced65eb58a7d7b3eef4aa220de
  manifest_sha256_file: artifacts/input_manifest.sha256
eval:
  greedy: true
  golden_cases: 100
logging:
  trace_every_steps: 20
  per_codebook_loss: true
```

## 12. 训练预算与 8 卡分配

官方说明 mini 单张 RTX 3090 约 2 小时跑通，实际受软件版本影响。

| 资源 | 分配方式 | 说明 |
|---|---|---|
| 1×24GB | 官方单卡 recipe | 最接近教学复现 |
| 4 卡 | 3 个 seed + 1 个 eval worker | 不改变 global batch |
| 8 卡 | B/C parity、多 seed 与独立 eval/ASR | 首选并行实验 |
| 8 卡 DDP | 仅做扩展实验 | global batch、LR 必须等价 |

DDP（PyTorch 的多卡数据并行）会成倍放大 global batch。不要为了用满八卡把 batch 扩大八倍——那就不再是官方 recipe。

## 13. 评测与测量

### 13.1 文本

- token-level NLL（负对数似然，越低越好）；
- QA exact match / token F1；
- image QA 人工 correctness；
- 文本回归：greedy token 序列是否一致。

### 13.2 语音内容

WER（word error rate，词错误率）用 ASR 转写结果对照参考答案计算：

$$
WER=\frac{S+D+I}{N}
$$

其中 $S$ 是替换、$D$ 是删除、$I$ 是插入的词数，$N$ 是参考文本词数。ASR、文本 normalization、大小写与标点规则必须固定——量尺一换，数字变化说不清来源。中文同时报告 CER（字错误率）。

### 13.3 语音条件

- CAM++ cosine speaker similarity（生成音色与参考音色的余弦相似度）；
- 参考音频与输出都使用同一裁剪和采样率；
- 至少按 speaker 分桶，不能只报平均数。

### 13.4 系统

- TTFT/TTFA p50、p95；
- encoder、prefill、decode 分段耗时；
- RTF、tokens/s、frames/s；
- 峰值 allocated/reserved 显存；
- 5 分钟连续会话内存是否增长。

## 14. `baseline-v1` 的验收条件

只有下列项目全部满足，才能把 checkpoint 标记为 `baseline-v1`。这些证据用于在后续改动出现回归时确定具体失败环节：

- [ ] 128 样本可过拟合；
- [ ] 三阶段训练从干净环境成功完成；
- [ ] resume 后 100 step 无明显轨迹跳变；
- [ ] trace on/off 的 logits 在声明公差内；
- [ ] 100 个 golden cases 可一条命令复跑；
- [ ] 每个指标可追溯到 case 和生成参数；
- [ ] 音频 loss 有 8 路分项；
- [ ] 同时报 `TTFA_server` 与 `TTFA_e2e`，主指标终点是播放器入队而非 codec logit；
- [ ] 明确写出 near-duplex 不是真双工；
- [ ] 明确写出 `baseline-v1` 是 mini 机制基线，不是发布权重质量复现；
- [ ] 没有把官方 full checkpoint 与自训 mini 的分数差写成复现误差；
- [ ] `baseline-v1` 所有核心文件有 hash。

## 15. 根据症状定位失败环节

调试时从最便宜、最可观测的一层开始查。比如音频全是噪声，先查 code range，别急着重训模型；TTFA 异常好看，先核对时间戳语义。

| 症状 | 可能原因 | 诊断 | 修复 |
|---|---|---|---|
| text loss 正常、audio loss 为 0 | answer codes/mask 为空 | 打印每路 valid count | 检查 parquet schema 与 assistant range |
| 第 q 路 loss 异常高 | delay 对齐错 | 可视化 8×T target | 对照 `start_pos=q+1` |
| 音频全噪声 | special token 被送入 Mimi | 统计 code range | decode 前处理 `>=2048` |
| 训练首阶段立即发散 | 错权重或 LR | 检查 key diff、首 20 step | 重新加载 `llm_768`，恢复官方 LR |
| projector 无梯度 | dummy/mode/注入失败 | grad norm + marker count | 验证占位 token 连续区 |
| resume 后 loss 跳跃 | RNG/optimizer/scaler 未恢复 | 比较 state dict | 保存所有状态与 sampler step |
| WER 波动巨大 | 采样或 ASR 不固定 | 保存 waveform/hash | greedy + 固定 ASR |
| TTFA 看似极低 | 测的是首 code | 查时间戳语义 | 改到首个 PCM chunk |
| WebUI 打断后显存涨 | generator/queue 未释放 | 连续 50 次中断 | finally 清理引用与 CUDA event |

## 16. 逐 case 分析要求

聚合分数之外，再生成可浏览 case tree：

```text
cases/<case_id>/
  input.json
  input_audio.wav
  input_image.png
  expected.md
  generated_text.txt
  generated_audio.wav
  codes.npy
  timeline.json
  trace.json
  judgment.json
```

每个失败 case 标一个主因：

- `thinker_semantics`；
- `audio_input_alignment`；
- `talker_pronunciation`；
- `speaker_condition`；
- `codec_decode`；
- `stop_timing`；
- `system_latency`；
- `evaluation_error`。

人工试听至少 30 个语音 case，并复核所有指标极端值。自动指标用于筛选需要检查的样例，不能代替听觉判断。

## 17. 交付物

1. `baseline-v1` checkpoint；
2. 可运行的三阶段配置；
3. 完整 data manifest；
4. 100 个 golden cases；
5. `metrics.json` 与逐 case JSONL；
6. 训练/推理 timeline；
7. trace 插桩与等价性测试；
8. `report.md`，明确可复现与不可复现的部分；
9. 5-10 个代表性音频样例；
10. 后续课程统一 regression 命令。

## 18. 复现清单

- [ ] upstream SHA 已固定；
- [ ] 外部模型 revision/hash 已固定；
- [ ] parquet SHA-256 已固定；
- [ ] 环境与 GPU 已记录；
- [ ] 所有随机种子已记录；
- [ ] 训练顺序和 optimizer state 已保存；
- [ ] sampling 参数已保存；
- [ ] ASR/normalizer 版本已保存；
- [ ] case 输入未进入训练；
- [ ] 结果包含均值、分布和失败样例；
- [ ] 没有把官方 checkpoint 评测写成自己训练复现。

## 19. 前沿对照与改造方向

这套"基线 + 测量 + 证据"的纪律，前沿系统同样在用，只是规模更大。[Moshi](https://arxiv.org/abs/2410.00037) 把评测拆层：codec 质量、文本侧语言能力、语音问答与对话质量各自单独评；延迟当核心指标正式报告——摘要给出理论延迟 160 ms、实测约 200 ms，理论值直接来自帧结构：Mimi 一帧 80 ms（12.5 Hz），加上流内错位帧数就有下界——本课 Step 7 的放大版。[Qwen2.5-Omni 技术报告](https://arxiv.org/abs/2503.20215) 把 benchmark 按输入输出模态拆桶：图像理解对照同规模纯视觉模型，音频对照纯音频模型，混合模态用 OmniBench；再用同一套推理任务分别以文字和语音提问，量化语音输入折损；语音生成单独评鲁棒性和自然度。本课 golden case 按模态和任务分桶、中文与视觉 case 不进 mini 质量判定，是同一思路。

参数量（26M 对几十亿）和数据时长是规模问题，砸资源就能缩小。剩下的是机制问题：输入是请求式整段送入，Moshi 是双流实时；"打断"靠外部 VAD 状态机，前沿在模型层面处理；视觉只有静态图——分别在第 5-7 课和第 8-10 课动刀。而测量纪律，本课做完你就和前沿在同一水平线，这是全课程最便宜的一次追平。



1. **golden case 扩容分桶。** 把 100 个 case 扩到 500 个，按长度、语言、speaker 分桶。改动位置：评测脚本与 `cases/` 目录，不动模型与训练代码。预算：无训练，单卡推理加 ASR 约 2-4 GPU-hour。预期：长度桶间 WER 与 TTFA 分位数出现稳定梯度，speaker 桶间 similarity 有可见差异，p95 更稳。失败判定：各桶指标与全局平均无差异，说明 case 来源不够多样，重新设计分桶采样。
2. **数值漂移矩阵。** 给 Step 2 的 parity 脚本加精度开关，比较 BF16、FP16、FP32（三种浮点精度，位数越少越快、抖动越大）小批次的自然漂移和 trace on/off 差异。改动位置：parity 脚本加 dtype 参数，训练代码不动。预算：pilot 数据一个固定 batch 重复运行，2 GPU-hour 以内。预期：FP32 同 kernel 路径接近 bitwise 重复，BF16 漂移最大，每种精度要各自的预注册公差。失败判定：任一精度下 trace on/off 差异超出自然漂移区间，说明插桩有扰动，回 Step 2 修。

Moshi "理论延迟可从帧结构算出"的结论在本课能复现方向：Mimi 一帧 80 ms，按 diagonal delay，第 0 帧的 8 路 code 到第 8 个生成位置才凑齐，由此算出首帧理论下界，与实测 `TTFA_server` 对比。预期同方向：实测远高于下界，差值主要落在整段 ASR 编码与 prefill。若实测低于下界，是时间戳语义错了，回 Step 7 查打点。

**更多顺手扩展：**

- 给 WebUI 加 timeline overlay，但不得改变生成路径；
- 加 30 分钟 soak test（长时间连续运行测试），测内存、队列和状态泄漏；
- 为每个新课程分支自动运行本课 regression；
- 复现官方 MoE checkpoint 的纯评测（MoE：稀疏专家模型，第 11 课的主角），暂不研究路由机制；
- 写一页当前能力边界，明确列出尚未实现的流式输入、语义 turn policy 和持续双工。

## 20. 论文与必读材料

按"当前系统、对照系统、基础原理"顺序读。每篇材料对应一个能在代码或实验里验证的问题，读完把答案写进报告；答不上来就回去重读。

### 20.1 MiniMind-O

- [MiniMind-O Technical Report](https://arxiv.org/abs/2605.03937)：
  读架构、input sequence、training pipeline、evaluation。带三个问题进去，答案要能在自己的产物里指认：bridge 为什么取中层（对照 Step 2 的 grad norm，看信息从哪层流向 Talker）；8 路 code 如何错位（对照 5.3 节的 $8\times T$ 表，第 $q$ 行起点是不是 $q+1$）；"near-duplex"由模型还是 VAD 实现（对照 `RealtimeSession` 源码）。
- [MiniMind-O 官方仓库](https://github.com/jingyaogong/minimind-o)：
  逐行读 `model_omni.py`、`omni_dataset.py`、`train_sft_omni.py`，拿第 6 节的表逐格打勾，每个符号都亲眼看到定义处。

### 20.2 对照系统

- [Moshi](https://arxiv.org/abs/2410.00037)：
  读 §2 Architecture、§3 Mimi、§4 data/training 与 latency。带着问题：Moshi 把"自己说的"和"听到的"两条音频流放在同一时间轴并行建模，MiniMind-O 是收完整段再回答。阅读后写出这个差别如何决定两者 TTFA 的量级差异。
- [Moshi 官方仓库](https://github.com/kyutai-labs/moshi)：
  只看 streaming state、codec model 和 inference entry，先不复现训练。带着问题：streaming 状态里缓存了什么？`stream_generate` 里对应物是什么、缺什么？这份笔记第 5 课改流式 listener 时直接用。

### 20.3 基础原理

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：
  重读 masked self-attention 与 teacher forcing。带着问题：对 8 路错位的 audio code 来说，"未来"具体指哪些位置？阅读后写出训练目标不能读取未来 audio code 的原因，和 5.3 节的 $q+1$ 偏移接上。
- [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)：
  读 multi-token heads 与 loss。带着问题：通用 MTP 的多个 head 预测同一路序列往后的多个 token，MiniMind-O 的 8 个 head 预测的是什么？阅读后写出两者在条件和输出上的差异。

读完材料回头看：系统一行没改，但它现在跑在可复现的环境里，每个输出有证据链，每个指标有定义，每个文件有 hash。下一课动第一刀：[第 02 课](02_multimodal_connector.md)把图像和音频进 Thinker 的 connector 拆下来，换几种结构做受控比较——哪种更好，判定标准就是本课冻结的 golden cases 和 regression 命令。
