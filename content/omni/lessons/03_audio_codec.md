---
id: 03_audio_codec
title: "可插拔音频 Codec"
summary: "目标波形相同、有效码率相近、Talker 预算一样时，换一个 codec（把声音压成整数编号、也能解回声音的压缩器），重建音质、token 好不好预测、流式延迟这三样会怎么一起变？"
unit: mechanism
play_tools: []
checkpoints:
  - "用帧率、码本数和词表大小算出 nominal bitrate（名义码率），并明白它和有效熵是两回事。"
  - "写出一个 AudioCodec adapter，把流式状态、采样率和 special token 的约定都管起来。"
  - "先单独比 codec 本身，再用相同预算重训 Talker，两步分开，别让变量搅在一起。"
  - "把重建、WER、说话人保持、token PPL、TTFA 和边界伪影放在同一张报告里看。"
---

# 第 03 课：理解并比较音频 Codec

> 内容：语音离散表示、重建评测与系统接口<br>
> 建议周期：7-12 天<br>
> 硬件：1-2 卡运行 codec screen；4-8 卡重训同规模 Talker<br>
> 产物：统一 `AudioCodec` 契约、codec-only leaderboard、端到端公平比较

## 1. 先把声音的表示搞清楚

第 01 课把 MiniMind-O 跑通、冻成了有指纹的 `baseline-v1`；第 02 课拆的是输入侧：encoder 和 Thinker 之间的 connector 换了几种结构做受控比较。这一课转到输出侧最底层的部件；codec，模型的"声带"。第 01 课黑话表里只给过它一句话：声音压成编号、编号解回声音。这句话够跑通训练，不够动刀；本课把它展开成完整机制。

三样东西。第一，把"声音变成整数编号、编号变回声音"这条链拆透：一段 24 kHz 波形怎么被压成每秒 12.5 帧、每帧 8 个整数；这 8 个整数为什么要 8 本码本轮流出，而不是一本大字典一次查完；解码时又怎么还原回波形。第二，把散落在 MiniMind-O 十处源码里的 Mimi 写死常量（8 路、2112 词表、320 ms 缓冲）抠出来，收进一个统一的 `AudioCodec` 接口；从此 codec 是可插拔零件。第三，用这个接口做一场公平比赛：Mimi 对上 EnCodec 和 DAC/SNAC，先比"自己压自己解"的重建质量，入选者再拿同样的预算重训 Talker，比端到端的内容正确性、音质和延迟。

原因有两个。其一，codec 的选择是一笔"码率-音质-延迟"的账：帧率定了 Talker 每秒要预测多少步，码本数定了每步要出几个编号，帧长定了模型开口的最小单位。这笔账不算清，第 04 课改 Talker 生成拓扑、第 05-07 课做流式和全双工时，你分不清瓶颈在模型还是在 codec。其二，重建好的 codec 不一定好用：它可能把声音压得更保真，代价是编号序列更长、更难预测，Talker 学不动，端到端反而更差。这个"部件指标和系统指标脱节"的坑，只有把两类评测分开做才能看见。

做完这课，你可以打开三臂对照页，同一句话听六段音频：三种 codec 的重建、三种 Talker 的生成，逐段对着 spectrogram 找差别；还能在 rate-distortion-predictability 图上指出哪个 codec 在哪个码率点被谁支配。给你一个失败样例，你能说出错在波形、编号、Talker 还是流式解码那一层。

本课术语（Mimi、RVQ、码本第 01 课各给过一句话，忘了先回去看那张表）：

| 术语 | 简要解释 |
|---|---|
| VQ（向量量化） | 查字典式压缩：连续向量不存原值，只存字典里最像它的那一条的编号 |
| 码本（codebook） | 那本字典本身：一组可学习向量，编号就是行号；Mimi 每本 2048 行 |
| RVQ | 8 本码本接力：第 0 本记大概，后面每本专门记上一本没记准的差额 |
| dead code | 字典里长期没人查的词条：某个编号几乎从不出现 |
| codebook collapse | 大量输入挤到少数编号上，字典大部分白印了 |
| nominal bitrate | 名义码率：帧率 × 码本数 × 每个编号的比特数，纸面上的每秒开销 |
| effective bitrate | 熵编码后实际要存的比特数，通常低于名义值 |
| frame rate | 每秒多少个 codec 帧：Mimi 是 12.5 Hz，一帧管 80 ms 声音 |
| semantic / acoustic code | 分层码本里偏"说了什么"的层和偏"听起来怎样"的层 |
| algorithmic latency | 算法天生的等待：不看算力，光看结构就必须攒够多少毫秒才能出声 |
| conformance test | 给每个 codec adapter 过的统一体检：shape、取值、时长、流式等价 |

## 2. 本课解决的问题

audio codec 把 waveform 编码为离散 code，又从这些 code 重建 waveform。麻烦在于：codec 的重建指标提高，不代表接进 Omni 系统后端到端也会提高。一个 codec 可能在 encode-decode 测试里声音更清楚，但在相同预算下重训 Talker 后，WER（ASR 词错误率）更高、TTFA（请求到首个可播放 PCM 的时间）更慢、长句错误更严重。两组结果衡量的是流水线上不同的环节，不能互相代言。

MiniMind-O 的音频输出空间固定为 Mimi：

- 24 kHz waveform；
- 12.5 Hz codec frame；
- 8 个 codebook；
- 每个 codebook 2048 个普通 code；
- 另有 pad/stop/speaker special token。

评估时要把三个问题分开问：

1. codec 自己能不能重建语音；
2. Talker 能不能预测这些 codec token；
3. 流式解码能不能在低延迟下不出边界伪影。

本课的顺序也照这三问排：先测 codec 重建，再把通过筛选的 codec 接回 Talker。在相同原始目标波形、近似有效码率和相同 Talker 训练预算下，分别测量帧率、码本拓扑和语义/声学分层对内容正确性、音质、说话人保持和实时生成成本的影响。

判定规则先立好：如果 challenger 只改善 codec reconstruction，却让 Talker token perplexity、WER 或 TTFA 变差，就不能凭单项重建分数替换现有 codec。

报告要区分三类内容：

- **官方实现事实：** 当前 MiniMind-O 围绕 Mimi 的 8 个 codebook 和固定 special ID 编写。
- **课程设计：** 先做 codec-only screen，再给入选 codec 同预算训练 Talker。
- **待验证假设：** 码率接近时，token 统计和时间拓扑会改变 Talker 的可预测性。

## 3. 开始前需要准备什么

两种起点：

- `baseline-v1`：用于验证旧 Mimi adapter 的数值等价；
- 官方 `minimind-3o`：用于不依赖第 01 课的独立执行。

你还需要另备**带原始 assistant waveform**的语料。常见错误是：MiniMind-O parquet 的 `answer_audios` 存的主要是 Mimi codes，而量化是有损的，不能从这些 codes 无损恢复原始真值。

本课独立输出：

```text
codec-v1-<name>-reconstruction
talker-v1-<name>-iso-budget
```

不要覆盖 `baseline-v1`。

## 4. 完成后应具备的能力

完成后，拿到一个失败样例，你应能把它定位到 waveform、code、Talker 或 streaming 中的具体一层：

1. 解释 VQ-VAE、RVQ、codebook collapse；
2. 从 frame rate、codebook 数、vocab 算 nominal bitrate；
3. 区分 nominal bitrate 与熵编码后的 effective bitrate；
4. 解释 semantic code 与 acoustic refinement code；
5. 为任意 causal codec 实现相同 encode/decode/stream 接口；
6. 先做 codec-only screen，再做 Talker 训练；
7. 识别"用旧 codec decode 作为新 codec 真值"的循环污染；
8. 同时报质量、可预测性、延迟和显存。

## 5. 原理:边造边讲

五个机制，沿"声音变编号、编号变回声音"这条链从里到外讲。每个按同一节奏：为什么需要（直觉）、怎么运转（机制）、精确定义（数学）、在哪测（代码与验证）。

### 5.1 向量量化:查字典,只存行号

声音是连续信号，24 kHz 采样、16 bit 精度的一秒钟波形就是 38.4 万比特。想让语言模型处理它，得先把连续值换成有限集合里的整数——好比不存整幅画，只在一本印好的图册里找最像的那页，记下页码。页码就是 code，图册就是码本。类比失效处：图册的页是人挑的、固定的，码本的每个向量是训练学出来的，而且查的是"最近的"那条，几乎永远查不到"完全一样的"——差出来的那截误差不会消失，它是 5.2 节 RVQ 存在的全部理由。

encoder 先把波形压成低帧率的连续向量 $z$，量化器从码本 $E=\{e_k\}$ 中取最近向量：

$$
q(z)=e_{k^*},\quad k^*=\arg\min_k\|z-e_k\|_2
$$

训练通常包含 reconstruction、adversarial/perceptual、commitment 与 codebook update 几项损失。实现中，`argmin` 产生离散 ID，decoder 根据 ID 读回 embedding 再还原波形——"编号变回声音"这半条链，就是查表取向量、卷积网络上采样回波形。

本课不训 codec，量化器都冻在各家权重里，你的落点是 `AudioCodec.encode` 出来的整数序列本身。长期不出现的 ID 叫 dead code；大量输入挤到少数 ID 上，就是 codebook collapse——字典再厚，常用的只有几页，压缩效率和表达力同时报废。Step 4 因此对每个 codec 同时统计 occupancy、entropy 和波形指标：编号用得匀不匀，和解出来像不像，是两件事，要分开量。

### 5.2 Residual Vector Quantization:8 本码本轮流补差额

一本 2048 行的字典描述一帧声音太粗。直接把字典加厚到 $2048^8$ 行？存不下也训不动。RVQ 的办法是接力：第 0 本码本先给个大概，量出误差；第 1 本专门量化这个误差，再量出更小的误差；如此叠 8 层。像画素描：先起轮廓，再一层层加阴影和细节，每一层只画上一层还没画到的部分。第 01 课那句"找零钱"说的也是这件事。类比失效处有两个：素描画家知道自己在画耳朵还是鼻子，而每层码本量化的只是数值残差，后层具体承载什么信息取决于 codec 的训练目标，不能预设它们一定对应某类感知细节；另外找零面额是固定的，每层码本的"面额"是训练学出来的。

第 0 层先量化输入，后续层依次量化前一层留下的残差：

$$
r_0=z,\quad q_i=Q_i(r_i),\quad r_{i+1}=r_i-q_i
$$

重建为 $\hat z=\sum_i q_i$。解码时 8 层各自查表，向量相加，一次性还原这一帧——这就是"每帧 8 个编号"的来历，也是 Talker 每个时间步要出 8 路预测的来历。

8 路编号怎么摆进训练序列、diagonal delay 为什么让第 $q$ 路错位 $q+1$，第 01 课 5.3 节推过，本课不重复；对应实现在 `dataset/omni_dataset.py::OmniDataset` 和 `MiniMindOmni.stream_generate`。本课的验证要点是：Talker 的后层码本 loss 高于第 0 层，可能只是正常统计差异（残差本来就更接近噪声、更难预测）；但如果某一路 loss 完全不下降，别急着怪码本，先检查 layout、mask 和 delay 对齐。

### 5.3 码率:每秒花多少比特描述声音,这笔账怎么算

码率就是描述声音的"每秒预算"。原始 24 kHz、16 bit 的 PCM 每秒要 384000 bit；一个 codec 的全部本事，就是用远小于这个数的预算，让人耳听不出太大差别。预算怎么花由三个旋钮决定：每秒几帧（帧率）、每帧几个编号（码本数）、每个编号几比特（词表大小取对数）。

等 vocab 时 nominal bitrate：

$$
R=f_{frame}\times N_q\times \log_2 V
$$

Mimi 的普通 code 若 $f=12.5$、$N_q=8$、$V=2048$，则 $12.5\times 8\times 11$，nominal rate 约 1100 bit/s——把 384000 压到 1100，三百多倍。这是"名义"账：如果再做熵编码（常用编号给短码、罕见编号给长码），实际存储的 effective bitrate 还会更低；本课比较统一用 nominal 值对齐。

实验配置要用该公式核对各臂码率。这是公平比较的第一道闸：如果 challenger 的码率高出数倍，它音质好可能只是因为预算多，质量提升不能全部归因于结构。special token 不计入 codec 码率——它们是系统约定，不是声音信息。

### 5.4 Codec 可预测性:压得好不等于学得动

codec 的下游不是人耳，是一个要逐步预测这些编号的自回归 Talker。对 Talker 来说，理想的编号序列是"有规律、好接龙"的；而重建更好的 codec，往往靠把信息塞得更满来实现——信息越满，序列越接近随机，越难预测。压缩率和可预测性天生互相拉扯。

重建更好的 codec 可能：

- frame 更密；
- codebook 更多；
- token entropy 更高；
- 跨层依赖更强。

每一条都在给 Talker 加活：frame rate 翻倍后，同一秒语音要处理两倍时间步（序列长一倍，推理每秒要跑的步数也翻倍）；增加 codebook 就是增加每帧的分类目标。这是本课坚持"重建质量、Talker 可预测性、系统延迟三样同时比"的原因——它们是三个会互相冲突的指标，不是一个指标的三种写法。

Step 4 的 token statistics 与 14.2 节的逐码本 PPL 就是这条机制的量具；判定用 14.4 节的 rate-distortion-predictability 图，不用单点分数。

### 5.5 Causal 与 algorithmic latency:结构决定的等待,算力救不了

延迟账里有一部分和 GPU 快慢无关：结构上必须攒够多少输入才能算第一次、必须凑齐多少 code 才能解第一块。这部分叫 algorithmic latency——机器无限快它也在。

algorithmic latency 至少包含：

- encoder lookahead（编码器要偷看未来多少毫秒才出第一帧）；
- frame/hop 等待（一帧本身覆盖的时长，Mimi 是 80 ms）；
- decoder receptive field（解码器看多宽的 code 窗口才能出稳定波形）；
- Omni 侧积累多少 code 才调用一次 decode；
- overlap-add 播放缓冲。

论文报告的 codec latency 不能直接当系统 TTFA 用。即使 waveform 的理论输出周期是 80 ms，如果 decoder 要累计 320 ms 的 code 才能首次运行，首段 PCM 至少要等这段累计时间——第 01 课"秒表按在哪里"的教训在这里重演：TTFA 的终点必须是可播放 PCM。Step 10 会扫描 decode chunk size，把这笔账逐项量出来。

## 6. MiniMind-O 中写死的 Mimi 假设

先在仓库中搜索硬编码的 `8`、`2112` 和 `320ms`，再把每处用法归类为 codec spec、special token policy 或 WebUI 缓冲策略。这张表就是"Mimi 长在系统里的十个位置"：

| 文件/符号 | Mimi 假设 | 应改为 |
|---|---|---|
| `OmniConfig.audio_vocab_size=2112` | 单层 2048 + 64 special | `vocab_sizes[]` + special policy |
| `audio_pad/stop/spk=2049/2050/2051` | 固定 ID | codec adapter 分配 |
| `TalkerHead(... num_layers=8)` | 8 路等 vocab | `num_codebooks` 与 per-q vocab |
| `TalkerEmbedding(... num_layers=8)` | 同上 | 读 codec spec |
| `MiniMindOmni.forward` | audio tensor `[B,8,T]` | `[B,Q,T]` 或 ragged group |
| `stream_generate` | `range(8)` 与 delay=7 | 由 schedule 生成 |
| `OmniDataset` | 每 8 个交错 code 解包 | 由 manifest layout 解包 |
| `train_sft_omni.py` | audio loss 除以 8 | 按 Q/有效 token 加权 |
| `eval_omni.py` | 直接 `MimiModel.decode` | codec registry |
| `webui/web_demo.py` | 4 Mimi frames≈320ms | codec frame_hz 推导 |

这些表项是**当前实现事实**；"应改为"一列是**课程改造设计**，并不意味着所有 codec 最终都能被规整成等长张量。还要做一个 backward compatibility test：`codec.type=mimi_legacy` 时，旧 checkpoint 的 logits 和 waveform 一致——重构接口不许顺手改行为。

## 7. 目标接口

`CodecSpec` 显式记录不同 codec 的采样率、帧率、码本数和 lookahead。业务代码必须从 spec 读取这些字段，不能继续使用 Mimi 的默认常量：

```python
@dataclass
class CodecSpec:
    name: str
    sample_rate: int
    frame_hz: float
    num_codebooks: int
    vocab_sizes: list[int]
    causal: bool
    encoder_lookahead_ms: float
    code_layout: str       # [B,Q,T]
    license: str
    revision: str

class AudioCodec:
    spec: CodecSpec
    def encode(self, wav, wav_lens=None, state=None):
        """return codes[B,Q,T], code_lens, new_state"""
    def decode(self, codes, code_lens=None, state=None):
        """return waveform[B,1,S], wav_lens, new_state"""
    def flush(self, state):
        """flush causal decoder tail"""
```

special token 不属于 codec 原始词表，单独放在 `CodecTokenSpace`：

```python
class CodecTokenSpace:
    codec_vocab_sizes: list[int]
    pad_id: list[int]
    stop_id: list[int]
    speaker_id: list[int]
```

如果各 codebook 的 vocab 大小不同，各 head 不能共用一个固定 `out_features`。为每一路添加 code range 断言，并在训练前运行；否则错误可能直到某一路出现越界 code 或异常 loss 才暴露。

## 8. 候选 codec

主实验最多选择 3 臂：

1. Mimi：基线；
2. EnCodec 24 kHz：成熟 RVQ 对照；
3. DAC 或 SNAC：二选一 challenger。

接口可以支持更多 codec，但正式 Talker 训练限制为三臂，以保证每臂都有多 seed 结果和逐 case 分析。

选择条件：

- 官方公开权重与代码可得；
- license 允许当前研究用途；
- 能离线批量 encode/decode；
- 最好提供 causal/streaming 路径；
- 采样率可统一或明确重采样；
- 有稳定 revision。

## 9. 数据 recipe

跨 codec 比较必须从同一份原始 waveform 开始。如果先将 Mimi codes 解码，再交给其他 codec 编码，其他 codec 的输入已经包含 Mimi 的重建误差，不能用于公平比较——这就是第 4 节能力清单里那条"循环污染"。

### 9.1 数据来源选择

Codec-only 推荐使用带原始 waveform 的开源语音：

- [LibriTTS](https://arxiv.org/abs/1904.02882) / OpenSLR 60；
- 可加公开噪声、混响和多说话人子集；
- 自有数据只有在许可和隐私审计后使用。

Talker 公平训练：

- 从 LibriTTS 建 text→speech pairs；
- voice condition 来自同 speaker 的**不同** reference clip；
- 不能把 target clip 本身作为 reference；
- 若做中文，另选许可清晰的中文语料，并独立报告。

### 9.2 统一数据字段

```yaml
utterance_id: string
speaker_id: string
text: string
language: en
wave_path: relative/path.wav
wave_sha256: hex
sample_rate: int
duration_ms: int
split: train | dev | test
source: librtts
license: CC-BY-4.0
reference_utterance_id: string | null
codec_name: mimi | encodec | dac
codec_revision: string
codes_path: relative/path.npy
codes_sha256: hex
num_frames: int
```

### 9.3 预处理

1. 解码为 float32 mono；
2. 去除 NaN、clipping 严重和空音频；
3. 保留原始 waveform hash；
4. 每个 codec 使用其原生采样率 encode；
5. 评测时统一重采样到指标要求的采样率；
6. code 保存为整数，不用有损压缩；
7. 验证 encode→decode 长度和时间戳；
8. 对每个 codec 统计 code occupancy、entropy、dead codes。

### 9.4 切分

- speaker-disjoint test（同一说话人不同时出现在训练和测试）；
- reference 与 target 不同 utterance；
- 同书/章节尽量不跨 split；
- test 波形永不用于 codec/Talker 选择；
- 噪声/RIR 资源也按 source 分开。

### 9.5 三档规模

| 档位 | 原始波形 | 用途 |
|---|---:|---|
| pilot | 2-10 小时 | 接口、重建、128 样本过拟合 |
| standard | 100 小时 | 三臂 Talker 公平比较 |
| full | LibriTTS 全量约 585 小时 | 验证可扩展性 |

### 9.6 License 与缺失边界

- LibriTTS 论文和下载页均需记录版本与 license；
- codec 代码 license、权重 license 可能不同，分别记录；
- MiniMind-O answer Mimi codes 不是原始 waveform；
- 禁止先 Mimi decode 再让其他 codec encode，并称为"同真值"；
- 若只有 codes，只能做 Talker topology，不做跨 codec 音质结论；
- MOS 评测使用真人时需同意书与匿名化。

## 10. 按依赖顺序执行实验

顺序即闸门：先验证 legacy adapter，再做 codec-only 测试，然后接入 Talker，最后测试 streaming。legacy 适配不等价时，不加入 challenger；codec-only 不通过时，不训练 Talker；128 样本无法过拟合时，不开始 100 小时训练。

### Step 1：抽出 legacy Mimi adapter

把当前所有 Mimi 常量迁入 `CodecSpec`。用官方 checkpoint 对 20 个 case 比较：

- codes；
- audio logits；
- decode waveform；
- streaming chunk 数；
- TTFA。

允许的 waveform 误差在运行前写入报告；相同 kernel 下应尽量 bitwise 一致。该测试用于确认接口重构没有改变基线。不同设备或 kernel 路径不要求逐 bit 相同，但公差必须在运行 test 前按 dtype 和设备登记——先跑再定公差，等于没有公差。

### Step 2：实现 codec conformance tests

每个 adapter 必须通过以下检查：

- shape 和每个码本的取值范围；
- 编码、解码后的时长是否一致；
- 空 waveform 和短 waveform；
- batch padding 是否影响有效区域；
- offline 与 streaming 是否在声明的公差内等价；
- reset 后不同 session 的 state 是否隔离；
- encode/decode 是否确定；
- 普通 code 区域是否拒绝 special token。

### Step 3：做 codec-only reconstruction

冻结全部 Omni 模型。对同一批 waveform 运行每个 codec。先不训练 Talker，避免把生成误差混进 codec 质量。

该步骤播放的是 codec 重建音频，不是 Talker 生成音频。报告页面必须将两类音频分栏，并标明来源，避免把 Talker 的预测错误计入 codec 重建评价。

### Step 4：分析 token statistics

每层统计：

- unigram entropy；
- code usage；
- dead-code ratio；
- adjacent-frame mutual information proxy；
- cross-codebook conditional entropy；
- 平均 run length。

这些统计用于解释 Talker CE，不能代替端到端评测。更高的 code entropy 表示 token 分布更分散；它是否带来更丰富的重建细节，以及是否增加 Talker 预测难度，要分别通过 codec-only 和 Talker 实验验证——5.4 节说的拉扯关系，在这一步第一次变成数字。

### Step 5：按码率配平

若候选支持可变 codebook 数，选使 nominal bitrate 落在基线 ±15% 的设置。无法配平则画多点 rate-distortion curve，不能只挑一个高码率设置——那是给 challenger 偷偷加预算。

### Step 6：重编码同一训练集

以原始 waveform 为唯一源，为每个 codec 产生独立 code cache。manifest 保证每个 `utterance_id` 三臂一一对应。

### Step 7：适配 Talker

只做必要结构变化：

- `Q` 路 embedding/head；
- vocab sizes；
- delay schedule；
- stop/pad；
- frame rate 对时间轴的影响。

Thinker、文本 target、speaker reference、训练 updates 保持不变。改得越少，结论越干净。

### Step 8：128 样本过拟合

每个 codec 都要在相同 128 utterance 上明显过拟合。如果 challenger 不能过拟合，先查 layout/head，不进入 standard——128 个样本都背不下来，多半是接线错了。

### Step 9：同预算训练

保持：

- trainable 参数尽量 ±5%；
- global updates；
- 每步目标音频秒数；
- optimizer；
- text condition；
- speaker split；
- sampling strategy。

同时报告按 token 与按音频秒的 compute。帧率不同的 codec，同样的音频秒数对应不同的 token 数，所以两种口径都要给。

### Step 10：流式 decode 与边界测试

扫描 80/160/320/640ms decode chunk。对每个 chunk size 测：

- boundary click rate；
- TTFA；
- RTF；
- waveform discontinuity；
- quality drop。

### Step 11：端到端评测

对同一文本、voice reference、seed 生成。用统一 ASR、speaker encoder、MOS/UTMOS 和延迟脚本。

### Step 12：冻结 winner

选择 winner 时同时比较内容正确性、音质、speaker 保持、TTFA 和 RTF，不把这些维度先压成一个临时加权总分。

内容正确性、音质和 speaker 相似度越高越好，TTFA 和 RTF 越低越好。若方案 A 在所有指标上都不差于方案 B，且至少一项更好，则 A 支配 B；未被其他方案支配的方案集合就是 Pareto frontier。winner 必须位于该集合中，而非单看 UTMOS。

## 11. 三个对照实验组

三臂允许保留各 codec 的实际结构差异，但必须明确控制码率和 Talker 训练预算：

| 臂 | codec | 码率策略 | Talker |
|---|---|---|---|
| A | Mimi legacy | 原始 8Q/12.5Hz | 原始拓扑 |
| B | EnCodec 24k | 选最接近基线 bitrate | 同参数预算 |
| C | DAC 或 SNAC | 选最接近基线 bitrate | 同参数预算 |

如果 C 是分层 frame rate codec，同时报告其真实串行/对齐代价，不能强行伪装成等长 `[Q,T]` 后忽略空位。

## 12. 配置示例

```yaml
experiment: lesson03_codec
base_checkpoint: hf://jingyaogong/minimind-3o
base_revision_or_sha256: ee3febbd08cc5b2bd41c039c825a8934232fee33
source_wave_manifest: manifests/librtts_100h.jsonl
codec:
  name: facebook/encodec_24khz
  revision: c1dbe2ae3f1de713481a3b3e7c47f357092ee040
  sample_rate: 24000
  causal: true
  target_nominal_bps: 1500
  active_codebooks: 2
token_space:
  special_tokens_per_codebook: 3
training:
  audio_seconds_per_update: 320
  updates: 30000
  seed: 42
  precision: bf16
evaluation:
  decode_chunk_ms: [80, 160, 320, 640]
  speaker_disjoint: true
```

该示例选择 EnCodec 的 2 个 75Hz、10-bit codebook，nominal bitrate 约为 1.5 kb/s——套 5.3 节的公式：$75\times 2\times 10=1500$。它是可选离散配置中较接近 Mimi 约 1.1 kb/s 的设置，但两者码率并不相同，报告里不能写成"等码率"。正式结果还要报告 1-codebook、0.75 kb/s 的 rate sweep 点；主实验臂固定为上述 2-codebook 配置。

接口拒绝硬编码：

```python
assert codes.shape[1] == codec.spec.num_codebooks
for q, vocab in enumerate(codec.spec.vocab_sizes):
    assert codes[:, q].max() < vocab
```

## 13. 训练预算与 8 卡分配

### Pilot

- GPU0-2：三种 codec 的 Talker 128 样本过拟合；
- GPU3-5：codec-only reconstruction batch；
- GPU6：统一 ASR/speaker/quality eval；
- GPU7：streaming latency runner。

### Standard

- 三臂 × 2 seed 占 6 卡；
- 1 卡 eval，1 卡补第 3 seed 或 profiling；
- 小模型优先独立单卡 job，不强制 DDP。

### Full

- 每臂 2-4 卡 DDP；
- 逐臂串行以保持机器负载相近；
- 如果显存允许，提高每卡 batch，但保持音频秒/global update 一致。

报告：

- GPU-hours；
- encoded audio hours/s；
- train audio seconds/s；
- peak GiB；
- storage footprint。

## 14. 评测数据与指标

### 14.1 Codec-only

- SI-SDR；
- STOI；
- PESQ（遵守实现许可与适用采样率）；
- ViSQOL；
- UTMOS 作为自动 proxy；
- CAM++ speaker cosine；
- nominal/effective bitrate；
- encode/decode RTF；
- algorithmic latency。

### 14.2 Talker token

$$
PPL_q=\exp\left(
-\frac{1}{N_q}\sum\log p(c_{q,t}\mid context)
\right)
$$

逐 codebook 报 CE/PPL、stop F1、长度误差和 exposure failure。PPL（困惑度）可以读成"模型每步平均在几个候选里犹豫"。

### 14.3 端到端

- ASR WER/CER；
- target text 与 ASR transcript 的语义分；
- UTMOS/人工 CMOS；
- speaker similarity；
- 长句 drift；
- TTFA p50/p95；
- RTF 与 frames/s；
- 5 分钟连续生成稳定性。

### 14.4 Rate-distortion-predictability

至少分别绘制 bitrate 与 reconstruction quality、Talker PPL、WER 的关系，以及 frame rate 与 TTFA、RTF 的关系。每张图保留实际 codec 配置，不能只显示臂名称。

## 15. Challenger Codec 的验收条件

满足以下条件后，challenger 才能进入后续 Omni 课程。结论只适用于本课测试的语音数据和系统配置，不外推到音乐、环境声或其他码率：

- [ ] legacy Mimi adapter 与旧路径等价；
- [ ] 所有 codec 通过 conformance tests；
- [ ] 所有臂使用同一原始 waveform；
- [ ] 码率匹配或完整报告 rate-distortion；
- [ ] 128 样本均可过拟合；
- [ ] 报告逐码本 token statistics 与 PPL；
- [ ] codec-only 和 end-to-end 结果分开；
- [ ] streaming decode 使用跨 chunk state，或明确标记为 overlap workaround；
- [ ] TTFA 测到可播放 PCM；
- [ ] license/weight revision 完整；
- [ ] 未将 Mimi 重建音频标记为原始真值；
- [ ] winner 是 Pareto 选择。

## 16. 失败诊断表

先根据症状判断错误发生在 reconstruction、token prediction 还是 streaming，从最便宜的一层查起。例如每个 chunk 边界都出现 click 时，先检查 decoder state 和 overlap，再决定是否修改 Talker。下表给出每种症状的第一项验证：

| 症状 | 原因 | 诊断 | 修复 |
|---|---|---|---|
| challenger 重建很好但 WER 差 | token 难预测 | 看 entropy/PPL | 降码本/帧率或改 Talker |
| 音频速度不对 | sample/frame rate 配错 | 时长 round-trip | 由 `CodecSpec` 推导 |
| 每块有 click | decoder 无状态/重叠不足 | 边界差分 | causal state 或 overlap-add |
| 短音频全空 | receptive field/flush | 50-500ms sweep | 正确 pad 与 flush |
| code 超 vocab | adapter layout 错 | per-q range | 不共享错误 vocab |
| stop 太早 | special token 与普通 code 冲突 | stop confusion matrix | 独立 token space |
| speaker sim 高、内容错 | 指标单一 | 联看 WER | 不以 speaker sim 选 winner |
| PPL 不可比 | vocab/熵不同 | bits/token 与 bits/s | 同时报 normalized NLL |
| 训练磁盘爆炸 | 多 codec cache | 统计 bytes/hour | chunked shard、hash 去重 |
| streaming 比 offline 差很多 | 非 causal checkpoint | 查模型声明 | 标记 offline-only，不伪称流式 |

## 17. 逐 case 分析

每个 utterance 的三臂对照页包含：

- 原始 waveform；
- 三种 codec reconstruction；
- 三种 Talker generation；
- spectrogram 和 waveform boundary；
- 原始 transcript 与 ASR transcript；
- per-codebook entropy/PPL；
- speaker similarity；
- quality metrics；
- TTFA 与 RTF；
- 人工偏好及错误标签。

错误 taxonomy：

- `codec_reconstruction`；
- `semantic_code_loss`；
- `talker_prediction`；
- `speaker_drift`；
- `stop_length`；
- `stream_boundary`；
- `resampling`；
- `metric_disagreement`。

报告中安排以下人工复核：

- 自动质量最好与最差各 10 例；
- 自动指标互相冲突 10 例；
- 长句 10 例；
- streaming click 10 例；
- speaker similarity 异常 10 例。

## 18. 交付物

1. `AudioCodec` / `CodecSpec` / `CodecTokenSpace`；
2. Mimi legacy adapter；
3. 至少两个 challenger adapter；
4. conformance test suite；
5. 原始 waveform 与 code manifest；
6. codec-only leaderboard；
7. 三臂 Talker checkpoint；
8. rate-distortion-predictability 图；
9. streaming chunk sweep；
10. 逐 case 可试听报告；
11. winner 与"不采用"的理由。

## 19. 复现清单

- [ ] 原始 waveform SHA；
- [ ] codec 代码/权重 revision；
- [ ] sample rate/frame rate/Q/vocab；
- [ ] code cache SHA；
- [ ] reference 与 target 隔离；
- [ ] train/dev/test speaker-disjoint；
- [ ] 码率计算脚本；
- [ ] Talker 参数与 update 对齐；
- [ ] ASR/quality/speaker 模型 revision；
- [ ] decode chunk 与 buffer；
- [ ] 人评协议与随机顺序；
- [ ] 所有 license 已登记。

## 20. 前沿对照与改造方向

本课那笔"码率-音质-延迟-可预测性"的账，前沿系统各有各的算法。[Moshi](https://arxiv.org/abs/2410.00037) 的答案是 Mimi 本身：把帧率压到 12.5 Hz，让语言模型每秒只需处理少量时间步，同时全链路 causal 以支撑流式；并用 semantic distillation 把语义信息蒸进第一个 codebook，让"说了什么"集中在最先生成的那一路。[SpeechTokenizer](https://arxiv.org/abs/2308.16692) 走同一方向：用 HuBERT 表示蒸馏第一层 RVQ，显式做语义/声学分层。[SoundStream](https://arxiv.org/abs/2107.03312) 和 [EnCodec](https://arxiv.org/abs/2210.13438) 的答案是弹性：训练时随机丢弃部分量化层（quantizer dropout），一套权重覆盖多个码率点，部署时按带宽选层数。[Qwen2.5-Omni 技术报告](https://arxiv.org/abs/2503.20215)（第 01 课已引）则把功夫下在解码侧：Talker 出离散语音 code 后，用滑动窗口限制感受野的流式解码器把 code 变波形，控制首包延迟——正是 5.5 节"decoder 攒多少 code 才能出声"那一项。

规模问题：前沿 codec 用大规模语音数据加对抗训练炼出来，我们只做冻结权重的评测和选型，不重训 codec，这部分砸钱能追。机制问题（本课工具就能量的）：帧率和分层怎么选、semantic 层要不要单独对待、流式解码状态怎么管——本课的三臂加 conformance test 就能给出自己的答案，不必照抄某家配置。



1. **EnCodec rate sweep 三线图。** 把第 12 节的 rate sweep 从两点扩成 1/2/4 codebook 三点（0.75/1.5/3 kb/s），每点各做 codec-only 重建和一次同预算 Talker 训练，在同一张图上画 bitrate 对 reconstruction quality、Talker PPL、WER 三条线。改动位置：配置里的 `active_codebooks` 与 Step 5/Step 9，接口代码不动。预算：codec-only 部分单卡数 GPU-hour；Talker 部分每点等于一个 standard 臂（100 小时数据、30000 updates），比主实验多一臂。预期：重建质量随码率单调升，PPL 同步升，WER 出现"中间某点最好"或持平的非单调形态。失败判定：三条线完全同向且比例不变，说明该码率区间内不存在取舍，结论照实写。
2. **砍层消融：quantizer dropout 稳健性。** encode 后只保留前 $q$ 层 code 再 decode（`codes[:, :q]`），对 Mimi 和 EnCodec 各画质量-层数曲线。改动位置：评测脚本在 `AudioCodec.decode` 前截断，模型零改动。预算：无训练，单卡 2-4 GPU-hour。预期：EnCodec 训练时用过 quantizer dropout，砍层后应平缓退化；未按此方式训练的 codec 可能断崖式崩坏——曲线形状本身就是"该 codec 能否当弹性码率用"的答案。失败判定：不存在,这是观测型实验,任何曲线形状都是有效结论；但若同一 codec 两次运行曲线不一致,先回 Step 2 查 decode 确定性。
3. **流式解码状态对照。** 给 Mimi adapter 各实现一版真 causal state decode（跨 chunk 传 `state`，尾部 `flush`）和一版无状态逐块 decode 加 overlap-add 补丁，在 Step 10 的四档 chunk 上对比。改动位置：`AudioCodec.decode/flush` 的 state 路径与 streaming runner。预算：无训练，单卡 2 GPU-hour 内。预期：causal state 版 boundary click rate 明显更低，且 chunk 越小差距越大。失败判定：两版无差异，多半是 chunk 太长掩盖了边界问题，缩小 chunk 重测；仍无差异再检查 state 是否真的被传递。

Moshi 论文"低帧率让语言模型序列变短、利于实时"的结论,本课三臂直接能验方向：Mimi 12.5 Hz 对 EnCodec 75 Hz,同一秒语音的 Talker 时间步差 6 倍,预期 RTF 和 TTFA 同方向拉开,且 EnCodec 臂的每秒生成步数按比例上升。若实测两臂延迟接近,先查 5.5 节清单里是不是别的项（decode 积累、播放缓冲）在主导,而不是急着推翻论文结论。

**更多顺手扩展：**

- 做 semantic-first + acoustic-refinement 两阶段 Talker；
- 学习 entropy coding，比较 nominal 与真实存储 bitrate；
- 让 codec adapter 输出 timestamp span；
- 测环境音、音乐和非语音事件，检验 speech codec 的 Omni 边界；
- 训练小型 codec 作为机制课，但不要把它混入本课预训练权重比较；
- 研究 packet loss concealment；
- 为最终真双工系统测 echo 场景下 codec robustness。

## 21. 必读论文与阅读问题

按 SoundStream/EnCodec、SpeechTokenizer/Moshi、SNAC 的顺序阅读。每篇读完做同一个动作：在 `CodecSpec` 里标出它改变的字段——frame rate、$Q$、vocab、因果性或解码状态——并完成对应的码率计算。答不上就回去重读。

### 21.1 RVQ 与低延迟 codec

- [SoundStream](https://arxiv.org/abs/2107.03312)：
  读 §2 Architecture、§3 quantizer dropout、loss。带着问题：改造清单第 2 条的砍层消融赌的就是这篇的机制。阅读后写出：一套权重实现可变码率的方法，以及 causal 约束所在模块。
- [High Fidelity Neural Audio Compression / EnCodec](https://arxiv.org/abs/2210.13438)：
  读 model、loss balancer、streaming/real-time 评测。带着问题：多项损失互相冲突时怎么配平？阅读后写出：loss balancer 如何同时影响感知质量和训练稳定性。
- [Descript Audio Codec](https://arxiv.org/abs/2306.06546)：
  读 improved RVQ、loss 和 evaluation。带着问题：这篇的重建指标很强，那 5.4 节的拉扯对它成立吗？阅读后写出：DAC 的重建指标不能直接预测 codec LM 难度的原因。

### 21.2 语义/声学分层

- [SpeechTokenizer](https://arxiv.org/abs/2308.16692)：
  读 semantic distillation 和 hierarchical tokens。带着问题：蒸馏进第一层的"语义"从哪来？阅读后写出：semantic distillation 如何改变第一个 codebook 的内容信息。
- [Moshi](https://arxiv.org/abs/2410.00037)：
  重点读 Mimi codec 的 frame rate、causal/streaming 设计。带着问题：对照"顺手复现"那条 12.5 Hz 对 75 Hz 的账。阅读后写出：低帧率对 Talker 序列长度、TTFA 和重建质量的影响。
- [SNAC](https://arxiv.org/abs/2410.14411)：
  读 multi-scale residual quantization。带着问题：各层帧率不同，第 11 节说的"不能伪装成等长 `[Q,T]`"具体卡在哪？阅读后写出：不同时间尺度的 code 如何映射到 Talker 序列。

### 21.3 数据

- [LibriTTS](https://arxiv.org/abs/1904.02882)：
  读 corpus construction、speaker/text alignment 和 split。带着问题：9.4 节的切分规则哪几条是这篇直接给的？阅读后写出：speaker-disjoint 切分和 reference-target 隔离规则。

读完材料回头看：系统现在多了一层"可换喉咙"的自由——codec 成了带 spec、带体检、带 leaderboard 的可插拔零件，你也算清了换一个喉咙在码率、音质、延迟上各要付什么价。但不管选了哪个 codec，每帧 8 路编号始终要有人按正确的顺序生成出来。[第 04 课](04_multicodebook_talker.md)就去动这个生成者：拆开 Talker 的 diagonal delay，比较三种多码本生成拓扑，看同一帧内 8 路编号之间的条件依赖到底该怎么建。
