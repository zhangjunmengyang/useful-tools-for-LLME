---
id: 05_streaming_listener
title: "因果流式 Listener"
summary: "用户话还没说完，系统能一边听一边持续更新语音和 Thinker 的状态吗？而且你能证明它任何时刻都没偷看还没到的 waveform 吗？"
unit: realtime
play_tools: []
checkpoints:
  - "分清三种听法：整段离线、切块但每次重算、真正带状态的因果 encoder。"
  - "实现 init_state、push_chunk、finalize 三件套，外加一条查得到账的 available_at 时间戳。"
  - "把双向的 SenseVoice 当老师，蒸馏出一个只看过去的 causal student，再验证相同前缀不受后面内容影响。"
  - "让新进来的 audio token 持续更新 Thinker 的 KV，而不是等人说完再把整段音频重新编一遍。"
---

# 第 05 课：实现因果流式 Listener

> 内容:因果语音编码、增量状态与真实时间评测<br>
> 建议周期:10–16 天<br>
> 硬件:1–4 卡训练 student;8 卡可并行测试 chunk、lookahead 和 seed<br>
> 产物:`StreamingAudioEncoder`、因果性测试、半双工增量 Thinker

## 1. 让模型边听边处理

[第 01 课](01_baseline_reproduction.md)建立了可复现基线，[第 02 课](02_multimodal_connector.md)比较了多模态 connector，[第 03 课](03_audio_codec.md)分析了 Mimi codec，[第 04 课](04_multicodebook_talker.md)比较了 Talker 的多码本生成顺序。此时输出侧已经能够逐帧生成和播放，输入侧仍要等 VAD 检测到静音后，才把整段录音送入 SenseVoice。用户说话期间，模型不会处理内容。

第 05 至 07 课依次实现流式输入、话轮判断和全双工调度。本课先把音频按真实到达时间切成 320ms 的 chunk，每个 chunk 到达后立即编码并更新 Thinker 的 KV cache。endpoint 到达时，系统直接在已有前缀上追加控制 token，不再重新编码整段音频。

实现难点有两个。第一，每个 chunk 的输出依赖前面的注意力 K/V、卷积尾部和降采样余数；任何状态遗漏都会造成边界处的特征偏差。第二，模型必须满足因果约束：某个时刻的输出不能使用尚未到达的音频。逐帧 mask 和前缀不变性测试共同验证这一点。

第 06 课的 turn policy 每 80ms 做一次话轮决策，第 07 课需要在系统输出期间继续接收用户输入，两者都依赖实时更新的输入特征。

验收包括真实时间 replay、deadline 记录和前缀不变性检查。替换 480ms 之后的波形后，共享前缀的输出必须保持不变；endpoint 后的首响应延迟也不应随整段语音长度线性增长。

本课术语:

| 术语 | 简要解释 |
|---|---|
| listener | 系统的输入侧:接收用户语音,编码成 Thinker 能吃的特征 |
| 因果(causal) | 时刻 t 的输出只依赖 t 之前(加声明过的偷看量)已到达的声音 |
| chunk | 一次送进编码器的一小段新音频(如 320ms);是投料批量,不是可见范围 |
| lookahead | 显式允许"偷看"的未来时长;偷看多少,延迟至少多多少 |
| left context | 每帧输出允许回看的历史长度,决定状态多大 |
| KV cache / prefix | Thinker 已读内容的键值缓存;增量追加,免得每次重读前缀 |
| 蒸馏(distillation) | 让只能看过去的 student 模仿能看全文的 teacher(SenseVoice) |
| endpoint | 声学上"这句话说完了"的判定,只管声音停没停,不管对话意图 |
| available_at | 一帧特征最早合法可用的时刻:源音频末尾加 lookahead(真实回放再加计算) |
| deadline miss | 真实时间回放里,特征算完的时刻晚于它本该可用的时刻 |

## 2. 本课解决的问题

MiniMind-O 可以逐步生成并播放 codec 音频,但用户语音会先完整录制,再一次性送入 SenseVoice。因此现有系统只有流式输出,没有流式输入。

listener 指系统中接收并编码用户语音的输入侧。本课只改造 listener。验收时要求三件事同时成立:用户尚未说完时,系统按音频的真实到达时间增量编码;每个 chunk 更新可复用的 Thinker prefix state;任意时刻的状态都不能依赖尚未到达的 waveform。

本课仍是半双工:

- 用户说话时系统持续听;
- 识别到声学 endpoint 后才开始回答;
- 不训练 backchannel(听人说话时的"嗯""对"这类短促回应)、语义打断或边说边 replanning。

backchannel、语义打断和生成期间的 replanning 留到[第 06 课](06_turn_policy.md)和[第 07 课](07_full_duplex_routing.md)。

常见错误是造出假流水线。出现以下任一情况时,不得把实现标记为流式 listener:

- 把音频切块,但每块都重新编码整个前缀;
- 先读取完整文件再"模拟 chunk";
- student 的当前输出偷看未来帧;
- endpoint 后又重新 full encode,先前 cache 没被使用;
- chunk 变小只是延迟下降,任务质量不可接受;
- 报告模型 compute latency,却没有真实时间 replay。

## 3. 开始前需要准备什么

需要准备:

- 第 01 课的 `baseline-v1`;
- 或官方 `minimind-3o`;
- 冻结 SenseVoice 作为 offline teacher(能看全文的老师;本课新训的因果 student 当学生);
- 冻结 Thinker/Talker,先训练 streaming student——一次只动一个部件;
- 数据必须有原始 question waveform,不能只有 codec codes。

本课独立 checkpoint,名字直接带上关键配置:

```text
listener-v1-causal-{chunk_ms}ms-la{lookahead_ms}ms
```

没有第 02 课的 winner connector 时,继续使用原始 `MMAudioProjector`,不影响本课实验。

## 4. 完成后应具备的能力

1. 区分 offline、chunked offline 和有状态因果处理;
2. 理解 receptive field(感受野:一帧输出实际"看得见"的输入范围)、left context、lookahead;
3. 实现 `init_state/push_chunk/finalize` 三段式接口;
4. 从 offline SenseVoice 蒸馏 causal student;
5. 让 audio token 按 chunk 增量进入 Thinker KV;
6. 设计 shared-prefix future-leakage test(共享前缀的未来泄漏测试);
7. 测 incremental prefix stability(增量前缀的稳定性);
8. 在真实音频到达时间轴上测 deadline miss rate 和 chunk latency。

## 5. 原理:边造边讲

五个机制,沿用第 01 课的节奏:直觉、机制、数学、代码落点、验证。

### 5.1 因果性与 lookahead:时刻 t 允许读到哪里

offline encoder 像剪辑室里的后期,拿到完整素材前后随便翻;流式 listener 像直播,只能基于已发生的声音干活。lookahead 相当于直播故意加的"延迟播出":导播比观众早看一小段,代价是观众听到的一切都晚这么多。类比失效处:直播延时只是整体平移,而 lookahead 改变模型能看到什么——不同的 $L$ 训出的是不同的模型,必须写进配置和日志。

偷看量 $L$ 必须显式配置;每帧输出要等它需要的最后一点未来到达后才能发布,$L$ 直接进延迟账。

在允许 lookahead $L$ 时,时刻 $t$ 的输出只能依赖:

$$
h_t=f(x_{\le t+L})
$$

其中 $L$ 是显式配置的 lookahead。若 $L=160ms$,输出可读取到 $t+160ms$,同时算法延迟至少增加 160ms。日志必须记录 $L$,不能把它算作零延迟。

配置里的 `student.lookahead_ms`;Step 1 的时间戳契约把它折进每帧的 `available_at_ms`。

Step 11 的真实时间 replay 逐帧断言消费不早于 `available_at`;13.1 节的因果性测试给出数值上限。

### 5.2 Chunk attention 的可见范围:投料批量不等于可见许可

chunk 好比食堂窗口一次放 320ms 的一批人进门:一次放多少人是物流安排,不代表队伍最前面的人能回头看见整批人。类比失效处:排队的人互相看不看得见无所谓,张量里的帧只要 mask 不拦就全可见——"谁能看谁"必须由逐帧 mask 单独规定。不能因为 320ms 音频已经放进同一个 tensor,就让 chunk 开头的输出读取整个 320ms。

设 frontend 输入为 `X: [B, Tin, Din]`,encoder 输出为 `H: [B, Tout, D]`。`input_time: [B, Tin]` 和 `output_time: [B, Tout]` 记录每一帧的名义时间,`input_valid: [B, Tin]` 标记 padding。可见与否逐帧判定,只看时间和有效性,不看 chunk 归属。

允许 lookahead $L$、left context $C$ 时,逐帧可见 mask 为:

$$
M_{b,i,j}=
\mathbf 1[
u_{b,j}\le t_{b,i}+L
\land
u_{b,j}\ge t_{b,i}-C
\land
\operatorname{valid}_{b,j}
].
$$

其中 $u_{b,j}$ 是第 $j$ 个输入帧的时间,$t_{b,i}$ 是第 $i$ 个输出帧的时间。`M: [B, Tout, Tin]` 中 `True` 表示允许读取;传给使用 `True=masked` 的 PyTorch 接口前必须取反。若 left context 不截断,就去掉第二个时间条件。

mask 就几行,方向和边界最容易写错:

```python
# input_time: [B, Tin], output_time: [B, Tout]
visible = (
    input_time[:, None, :]
    <= output_time[:, :, None] + lookahead_ms
)
if left_context_ms is not None:
    visible &= (
        input_time[:, None, :]
        >= output_time[:, :, None] - left_context_ms
    )
visible &= input_valid[:, None, :]
assert visible.shape == (B, Tout, Tin)
```

full-tensor 对照和 stateful 实现必须使用同一张逐帧 mask。full-tensor 一次收到完整 waveform,也不能越过该 mask;stateful 实现只有在 $t_i+L$ 对应的输入已经到达后,才能发布输出 $h_i$。cache 可以包含 K/V、卷积状态和 subsampling tail(降采样攒了一半、还没凑够一帧的余数)。每种状态都要写入显式数据结构,并在 session reset 后清空。

先做一个确定性单测:输入 960ms,chunk 为 320ms,lookahead 为 160ms。复制输入后,只修改 480ms 之后的 waveform。所有满足 `output_time + 160ms <= 480ms` 的输出 feature、timestamp 和 endpoint logit 必须在预注册公差内一致。再直接断言这些输出对应的 mask 在 480ms 之后全部为 `False`。该测试同时阻止"完整当前 chunk 可见"和 padding mask 方向写反。

### 5.3 Offline teacher 蒸馏:让同声传译对着编辑成稿练习

SenseVoice 是"看完全文再下笔"的编辑:每帧 hidden 都用了双向上下文。因果 student 是同声传译,听到哪译到哪。蒸馏就是让同传拿编辑的成稿当参考答案练习。类比失效处:同传的目标从来不是逐字复刻编辑成稿——teacher 用了未来信息,student 原理上追不平,蒸馏目标是可用近似。

冻结 teacher,student 输出经投影 $P$ 对到 teacher 的表示空间,按时间对齐后算距离;梯度只流向 student($\operatorname{sg}$ 是 stop-gradient:teacher 侧数值当常数用,不回传梯度)。

SenseVoice teacher 的 hidden $H^T$ 使用双向上下文,student $H^S$ 只能近似:

$$
\mathcal L_{distill}=
\lambda_1\|P(H^S)-\operatorname{sg}(H^T)\|_2^2+
\lambda_2(1-\cos(P(H^S),H^T))
$$

teacher 和 student 的输出必须按时间对齐。teacher 使用未来上下文,student 不使用,因此蒸馏目标是可用近似,不要求 student 逐点复制 teacher。

时间对齐在 Step 4,蒸馏训练在 Step 5;三个损失权重在配置的 `distillation` 段。

按长度桶评估 hidden cosine 和 frozen ASR probe(冻结的识别探针:检查表示里还留着多少内容信息),见 13.2 节。

### 5.4 增量 LLM prefix:收一箱记一箱,不重盘整个仓库

Thinker 读过的每个 token 都在注意力里留下键值对,存进 KV cache。增量消费像仓库收货:每来一箱登记入库;full prefill 则是每来一箱重盘整个仓库,音频越长盘点越贵,而且全堆在 endpoint 之后付账。类比失效处:仓库里的箱子互不影响,KV 是因果的——往后追加便宜,想改前面已登记的内容就得整体重算。流式前缀"不能撤回",这是 13.3 节度量 prefix stability 的原因。

流式音频 prefix 从 `<user><audio_start>` 开始。每个音频 chunk 的 projected tokens 在完成编码后依次追加;话轮结束时再追加 `<audio_end><assistant>`。每个 chunk 到达并完成编码后立即推进 Thinker KV。endpoint 触发后只追加结束 token 和 assistant header,不重新 prefill 已处理的音频。

Step 7 的 `consume_*` API;现状 `MiniMindOmni.forward` 只在 `start_pos==0` 时注入音频特征(见第 6 节表),必须拆开。

第 13.5 节"是否发生 endpoint 后 full re-encode"指标,加上诊断表"endpoint 后仍慢"一行的 profiler 检查。

### 5.5 Endpoint 与 turn-taking:传感器只管声音停没停

本课的 endpoint 是声学传感器,只回答一个窄问题:这段话语在声音层面结束了没有。离"该我说话了"的判断还差一整课。

本课 endpoint 只判断当前声学 utterance 是否结束,不判断以下对话行为:

- 用户只是停顿还是让出话轮;
- "嗯"是 backchannel 还是新问题;
- 用户是否要求打断助手。

Step 9 的 endpoint head,每 40 或 80 ms 出一次判定。

验收条件要求写清 endpoint 与语义 turn policy 的边界;对话层面的接话决策是[第 06 课](06_turn_policy.md)的主角。

## 6. MiniMind-O 当前的离线输入路径

先看清离线路径把"整段"假设写死在哪些符号里,右列是本课的改造方向:

| 文件/符号 | 当前离线行为 | 本课改造 |
|---|---|---|
| `SenseVoiceAudioProcessor` | 整段 waveform→fbank | 增量 fbank state |
| `load_sensevoice` | 冻结 offline encoder | 保留 teacher,新增 student |
| `encode_audio_inputs` | 一次 encoder forward | streaming encoder interface |
| `inject_audio_features` | 仅 `start_pos==0` 注入 | 允许 chunk append |
| `MiniMindOmni.forward` | KV 主要用于生成 | 暴露 prefix consume API |
| `OmniDataset.load_audio_inputs` | 整段增强/feature | 输出 waveform + chunk timeline |
| `RealtimeSession` | VAD 聚合到 `speech_end` 后取整段 | chunk 到 listener,同时保留 VAD |
| `webui/web_demo.py::realtime` | endpoint 后 `prepare_turn` | 收到时即 `push_chunk` |

fbank(波形切小窗算出的滤波器组频谱特征,语音识别的标准输入)也有状态:窗间有重叠,切 chunk 时尾巴要留着。

当前 `forward` 只在 `start_pos==0` 时编码并注入 audio feature。改造时要把两个操作拆成独立 API:encoder 增量产生 feature,Thinker 增量消费 projected audio token。若仍由一次 `forward` 同时完成,endpoint 后通常还会重复编码全部前缀——即第 2 节禁止的第四条。

## 7. 目标接口与架构

### 7.1 Encoder

三段式契约:开局建状态、来一块推一块、收尾清空。

```python
class StreamingAudioEncoder(nn.Module):
    def init_state(self, batch_size, device):
        return EncoderState(...)

    def push_chunk(self, pcm16k, state, is_final=False):
        """
        pcm16k: [B, samples_new]
        return:
          features [B,Tnew,D]
          feature_timestamps [B,Tnew,2]
          endpoint_logit [B,Tnew]
          new_state
        """

    def finalize(self, state):
        """flush subsampler/conv tail; no hidden future"""
```

推荐 student 先用 streaming fbank/subsampler 处理新 waveform,再经过 4 层 causal Conformer 或 Emformer 产生 512 维 feature,最后接入现有 audio projector。Conformer(卷积加注意力混着堆的语音编码器)与 Emformer(带记忆库的流式 transformer)细节看第 20 节论文,选型在 Step 3。

### 7.2 Session

```python
class ListenerSession:
    encoder_state
    thinker_prefix_kv
    audio_clock_ms
    def push_pcm(chunk, arrival_ts):
        feats = encoder.push_chunk(...)
        projected = audio_proj(feats)
        thinker_prefix_kv = thinker.consume_audio(projected, timestamps)
    def close_user_turn():
        thinker_prefix_kv = thinker.consume_control(AUDIO_END, ASSISTANT)
```

### 7.3 训练输入与在线输入的区别

训练时可以将完整 waveform 切成 chunks,并在同一 graph 中 unroll(按 chunk 依次展开前向,方便一次反传)。在线运行时,每次调用只能接收新到达的 chunk,state 跨调用保存,输入对象中不得包含未来 waveform。

## 8. 数据 recipe

### 8.1 来源

主数据:

- 官方 Audio-to-Audio(A2A)的 `question_audios`,用于保持与 MiniMind 任务一致;
- [LibriSpeech](https://www.openslr.org/12) train-clean-100,用于 ASR/因果表示 probe;
- 可选 [FLEURS](https://arxiv.org/abs/2205.12446) 做多语言泛化。

Endpoint 数据:

- utterance 尾部保留真实静音——尾巴剪光,endpoint 就没得学;
- 合成不同 pause 长度仅作为 augmentation;
- 真实会话 turn endpoint 留给第 06 课。

### 8.2 Schema

```yaml
sample_id: string
audio_path: relative.wav
audio_sha256: hex
sample_rate: 16000
duration_ms: int
transcript: string | null
instruction_target: string | null
speaker_id: string | null
language: string
source: string
license: string
split: train | dev | test
chunks:
  - {start_ms: 0, end_ms: 320, is_final: false}
  - {start_ms: 320, end_ms: 640, is_final: false}
endpoint_ms: int
teacher_feature_path: relative.npy
teacher_revision: string
```

### 8.3 预处理

1. waveform 转 mono 16 kHz float32;
2. 不预先裁掉所有尾部静音;
3. 记录原始 duration 和 speech-active span;
4. offline SenseVoice 只生成 teacher cache(teacher hidden 提前算好存盘);
5. teacher cache key 含模型 revision、frontend 参数、wave hash;
6. chunk 在训练时动态切,保存 schedule seed;
7. fbank/subsample state 必须跨 chunk;
8. augmentation 在 chunk 前作用于完整 waveform,避免边界不连续;
9. 不让 feature cache 泄漏 test label。

### 8.4 切分

- speaker-disjoint test(同一说话人不同时出现在训练和评测,第 01 课的老规矩);
- 同录音切片不跨 split;
- 相同 transcript 尽量不跨;
- 噪声/RIR source 隔离(RIR:房间冲激响应,模拟混响的滤波器);
- shared-prefix causality pairs 单独留出;
- test 只用于最终 chunk/lookahead 配置。

### 8.5 三档规模

| 档位 | 音频 | 用途 |
|---|---:|---|
| pilot | 5–10 小时 | state、对齐、future leakage |
| standard | 100 小时 | student 蒸馏 + MiniMind LoRA |
| full | 1k 小时级许可清晰语音 | 多语言/噪声扩展 |

### 8.6 License 与缺失边界

- OpenSLR 下载条目与 corpus license 均写入 manifest;
- FLEURS 按 dataset card/license 审计;
- question audio 有 waveform 才能流式化;
- 只有 fbank/codes 的样本不能测 frontend state;
- 合成 pause 不可代替真实会话 endpoint;
- 录音可能含个人信息,发布前做隐私审查。

## 9. 按依赖顺序执行实验

### Step 1:定义时间戳契约

对每个输出 feature 记录 `source_start_ms`、`source_end_ms` 和 `available_at_ms`。若只考虑算法 lookahead,则 `available_at_ms = source_end_ms + lookahead_ms`;真实回放还要计入编码耗时。

后续任何消费不得早于 `available_at_ms`。

### Step 2:实现 streaming frontend

先只做 waveform→fbank/subsample,对比 offline/full waveform 与 chunked concatenation:

- feature shape;
- 时间轴;
- chunk 边界数值;
- final flush。

如果 offline 与 chunked frontend 的时间戳或有效输出不一致,先修复 frontend,不开始 encoder 训练。

### Step 3:实现 causal student

从以下两种结构中选择一种,并固定后再比较 chunk/lookahead:

- 4 层 causal Conformer;
- 4–8 层 Emformer。

用显式 dataclass 保存 K/V、conv 和 memory state,不使用全局变量。单测要同时运行两个 session 并交错推送 chunk,验证状态不会串线。

### Step 4:teacher–student 时间对齐

若 teacher/student frame rate 不同:

- 使用固定 pooling/interpolation;
- 保存 alignment map;
- 忽略 teacher 的 padding;
- 不通过可训练强大 decoder 掩盖错位。

先用正弦信号、脉冲和短语音分别验证 timestamps。脉冲位置已知,可直接检查边界输出被分配到哪个时间区间。

student 的输出 rate 必须从固定 teacher revision 的实际 `valid_len` 与时间戳测得:

$$
f_{teacher}\approx
\frac{\text{valid output frames}}
{\text{有效 waveform 秒数}}
$$

SenseVoice 默认 frontend 的 10ms frame shift 与 LFR stride 6(LFR:把相邻若干帧拼成一帧来降帧率)通常对应约 60ms 一帧、约 16.67Hz,但边界处理以实测 timestamp contract 为准。Mimi 的 12.5Hz 是**输出 codec**帧率,与 listener 输入特征率无关,不得拿来硬编码 student rate。

### Step 5:表示蒸馏

阶段 A:

- 冻结 teacher;
- 训练 student + projection;
- MSE/cosine;
- 可加 CTC/ASR auxiliary(CTC:不需要逐帧对齐标注的语音识别损失,给表示加一点"内容必须可辨认"的压力);
- 不接 Talker。

每个长度桶评估 hidden cosine 与 ASR probe。

### Step 6:三臂表示隔离

建立:

1. offline SenseVoice;
2. student 接收完整波形,但严格使用 5.2 的逐帧 block mask;
3. student 按真实 chunks stateful 运行。

臂 2 与 3 使用相同 `input_time/output_time/input_valid`、chunk 边界和 lookahead。两者逐帧输出 feature、timestamp、endpoint logit 应在预注册公差内一致;若只比较整段平均 cosine,局部未来泄漏可能被掩盖。该检查不通过时,先修 mask 或 state,不进入 Thinker 增量消费实验。

### Step 7:改造 Thinker prefix 消费

新增 API:

```python
consume_text_tokens(...)
consume_audio_embeddings(...)
finalize_user_turn(...)
```

不要预先构造包含未来空位的大 tensor。每个 chunk 的 projected states 只有在 `available_at` 到达后才能推进 KV。

### Step 8:训练 MiniMind 对齐

阶段 B:

- student 固定;
- train audio projector;
- Thinker 可用相同 rank LoRA(低秩适配器:只训一对小矩阵的增量,便宜地微调大部件);
- Talker 冻结;
- 目标先用 Thinker text,隔离输入理解。

### Step 9:声学 endpoint head

Endpoint head 每 40 或 80 ms 输出 `CONTINUE` 与 `UTTERANCE_END` 两类 logits。

该 head 只使用声学标签。运行时可用 hysteresis(迟滞:连续几帧过线才改判,防抖动)与最短 speech/silence 约束控制 session,同时保存 raw probability,供阈值校准和失败分析。

### Step 10:未来泄漏测试

构造两段具有相同前缀、不同后缀的音频:A 使用 `shared_prefix + suffix_A`,B 使用 `shared_prefix + suffix_B`。

在 shared prefix 的 `available_at` 之前:

- feature 应一致;
- endpoint logit 应一致;
- Thinker prefix state 应一致;
- 允许的误差写入测试。

再运行 property test(性质测试:随机生成输入变体,验证不变量恒成立):只修改未来 waveform,重新执行后比较共享前缀上的 feature、endpoint logit 和 Thinker state。

### Step 11:真实时间 replay

按 20ms packet 和固定 seed 的网络 jitter(包到达间隔的抖动)回放音频。listener 只能在 packet arrival 后开始对应计算。逐 feature 记录完成时间,并与其 deadline 比较。这一步考察逐帧准时,不看平均吞吐。

### Step 12:半双工端到端

endpoint 后:

- 直接在已存在 Thinker KV 上追加 assistant header;
- 生成 text/audio;
- 不重新 encode 用户语音;
- 测 user-end→first-audio。

### Step 13:chunk/lookahead sweep

在 winner 架构上扫描 160、320、640 ms 三种 chunk,以及 0、80、160 ms 三种 lookahead。

主三臂固定 320ms/160ms;sweep 只做 Pareto 比较。任务质量越高越好,ASR WER、延迟和计算开销越低越好。若方案 A 在这些指标上都不差于方案 B,且至少一项更好,则 A 支配 B;未被其他方案支配的方案集合就是 Pareto frontier(第 04 课用过,规则一样)。

## 10. 三个对照实验组

| 臂 | Encoder | 输入方式 | Thinker |
|---|---|---|---|
| A | offline SenseVoice | 整段 | endpoint 后 full prefill |
| B | causal student | 整段 tensor、causal mask | endpoint 后 full prefill |
| C | 同 B | 320ms stateful +160ms lookahead | 每 chunk 增量 KV |

A/B 差异是 teacher 到 student 的表征损失,是"放弃看未来"付的学费;B/C 用同一个 student,差异用于检查 stateful chunk 实现和增量 Thinker——理论上应落在公差内,超了是 bug,不是学费。

## 11. 配置示例

```yaml
experiment: lesson05_streaming_listener
base_checkpoint: hf://jingyaogong/minimind-3o
base_revision_or_sha256: ee3febbd08cc5b2bd41c039c825a8934232fee33
teacher:
  type: sensevoice_small
  checkpoint: FunAudioLLM/SenseVoiceSmall
  revision: 3847d57b6bdf2dd8875cb1508d2af43d80a16bf7
student:
  type: causal_conformer
  layers: 4
  hidden_size: 512
  heads: 8
  conv_kernel: 15
  chunk_ms: 320
  lookahead_ms: 160
  left_context_ms: 8000
  output_hz: DERIVE_FROM_TEACHER_TIMESTAMPS
  output_frame_shift_ms: MEASURE_AND_RECORD
distillation:
  mse_weight: 1.0
  cosine_weight: 1.0
  ctc_weight: 0.2
alignment:
  train_audio_projector: true
  thinker_lora_rank: 16
  freeze_talker: true
endpoint:
  frame_ms: 80
  threshold: 0.7
  hysteresis_frames: 3
```

两个大写占位值是故意的:必须按 Step 4 实测填写,给默认值就会有人跳过实测。

运行伪代码:

```python
state = session.init()
for packet, arrival_ts in network_stream:
    assert clock.now() >= arrival_ts
    new_feats, feature_timestamps, endpoint_logits, state = (
        listener.push_chunk(packet, state)
    )
    for feat, ts in zip(new_feats, feature_timestamps):
        assert clock.now_ms() >= ts.available_at_ms
    session.consume_features(
        new_feats,
        feature_timestamps=feature_timestamps,
        endpoint_logits=endpoint_logits,
    )
session.finalize_user_turn()
session.generate_response_from_cached_prefix()
```

两个 `assert` 分别保证不在包到达前计算、不在 `available_at` 前消费,它们就是契约本身,别删掉。

## 12. 训练预算与 8 卡分配

### Pilot

- GPU0:frontend/state 单测;
- GPU1–3:A/B/C 小数据;
- GPU4–6:chunk/lookahead 组合;
- GPU7:实时 replay 与 eval。

### Standard

- 3 臂×2 seed = 6 卡;
- GPU6/7 运行 ASR、QA、latency;
- 补第三 seed 时串行占卡。

### Full

- student 蒸馏 4–8 卡 DDP(多卡数据并行,第 01 课用过);
- MiniMind alignment 2–4 卡;
- 不把 teacher 同时复制到每卡时可离线 cache hidden;
- cache 必须记录 revision/hash。

报告:

- GPU-hours;
- audio hours/s;
- state bytes/session;
- listener RTF(定义回第 01 课:生成墙钟时间除以音频时长,小于 1 才跟得上);
- deadline miss;
- peak memory。

## 13. 评测与指标

### 13.1 因果性

$$
Leak(t)=\max|H^A_{\le t}-H^B_{\le t}|
$$

其中 A/B 在 $t+lookahead$ 之前的 waveform 完全相同。任一前缀位置超过预注册公差即判定因果性测试失败,不能用全序列平均误差掩盖局部泄漏。

### 13.2 表征

- teacher/student cosine;
- offline vs chunked student max error;
- frozen ASR probe WER(WER 定义见第 01 课);
- MiniMind QA exact/F1;
- noise/speed robustness;
- 长度桶。

### 13.3 Prefix stability

若每 chunk 产生增量 transcript $y^{(k)}$:

$$
RevisionRate=
\frac{\sum_k EditDistance(y^{(k)}, prefix(y^{(k+1)}))}
{\sum_k |y^{(k)}|}
$$

改口率:后一步推翻前一步多少。没有 transcript decoder 时,比较相邻 chunk 共有时间范围内的 hidden drift,并明确有效位置 mask。

### 13.4 Endpoint

- endpoint latency:

$$
L_{end}=t_{decision}-t_{last\_speech}
$$

- false early cut(话没说完就判结束);
- false late(说完很久才判);
- miss(整段漏判);
- threshold calibration/ECE(校准误差:模型报的把握和实际正确率差多少)。对 $n$ 个被打分的决策时刻,listener 在时刻 $i$ 输出 endpoint 概率 $p_i$;在 dev 集锁定阈值 $\tau$ 后,$\hat y_i=\mathbf{1}[p_i\ge\tau]$。$y_i=1$ 表示人工标注认为此刻已经到达合法 endpoint,尚未到达则为 0。分类置信度与是否判断正确分别是

  $$
  c_i=
  \begin{cases}
  p_i,&\hat y_i=1\\
  1-p_i,&\hat y_i=0
  \end{cases},
  \qquad
  a_i=\mathbf{1}[\hat y_i=y_i]
  $$

  将置信度按 $B_m=\{i\mid (m-1)/M<c_i\le m/M\}$ 放入预先固定的 $M$ 个等宽区间;$c_i=0$ 时归入第一个区间。于是

  $$
  ECE=\sum_{m:\,|B_m|>0}\frac{|B_m|}{n}
  \left|
  \frac{1}{|B_m|}\sum_{i\in B_m}a_i
  -
  \frac{1}{|B_m|}\sum_{i\in B_m}c_i
  \right|
  $$

  本课固定 $M=15$,并报告 $\tau$ 与正负标签数量。这里的 $a_i$ 是当前决策是否正确,不是整段音频最终有没有成功结束;
- pause-length bucket。

### 13.5 系统

- listener RTF;
- per-chunk p50/p95;
- deadline miss rate;
- state memory;
- endpoint→TTFT/TTFA(秒表起止定义在第 01 课);
- 是否发生 endpoint 后 full re-encode;
- 30 分钟 session state 泄漏。

## 14. 验收条件

- [ ] encoder 有显式 init/push/finalize;
- [ ] frontend offline/chunk timestamps 已验证;
- [ ] student output rate 来自 teacher timestamps/valid_len 实测,未复用 Mimi 12.5Hz;
- [ ] shared-prefix future leakage test 通过;
- [ ] B/C 表征差在预注册公差内;
- [ ] 每个 feature 有 `available_at`;
- [ ] Thinker 每 chunk 更新 KV;
- [ ] endpoint 后没有 full re-encode;
- [ ] 320ms/160ms 主配置 listener RTF <1;
- [ ] p95 chunk compute 小于可用 wall-clock budget;
- [ ] QA/ASR 回退不超过预注册阈值;
- [ ] endpoint 与语义 turn policy 边界写清;
- [ ] 真实时间 replay 结果完整。

以下数值可作为 pilot 起点,正式阈值要根据 baseline 波动预注册(先测自然抖动再定线,规矩同第 01 课):

- shared-prefix leak ≤数值公差;
- stateful vs offline-causal hidden cosine ≥0.999;
- ASR WER 相对退化 ≤10%;
- deadline miss rate <1%;
- p95 chunk latency 小于该配置的可用 wall-clock budget;
- endpoint p95 ≤500ms(不是第 06 课的最终 turn latency)。

## 15. 失败诊断表

| 症状 | 原因 | 诊断 | 修复 |
|---|---|---|---|
| chunk 边界 hidden 跳变 | frontend/subsampler state 丢失 | impulse test | 缓存 tail |
| B 好 C 差 | cache/mask 错 | offline causal vs stateful | 层级 state 单测 |
| future leakage | chunk tensor 含未来/卷积非因果 | shared-prefix pair | causal padding/裁剪 |
| 输出 feature 重复 | overlap 被消费两次 | timestamp monotonicity | 明确 emitted range |
| 输出 feature 缺失 | flush 不完整 | duration round-trip | finalize tail |
| 长音频显存增长 | cache 未裁 left context | state bytes vs time | 固定 memory/window |
| endpoint 过早 | 合成 pause 偏差 | pause bucket | 校准/真实数据 |
| endpoint 很晚 | 依赖固定静音 | raw logit timeline | endpoint auxiliary |
| endpoint 后仍慢 | 又 full prefill | profiler | 复用 prefix KV |
| QA 差但 ASR 好 | projector/LLM 对齐不足 | frozen ASR vs QA | stage B LoRA |
| 蒸馏长期错一帧 | 把 codec rate 当 input rate | valid_len/duration、timestamp map | 从 teacher 实测 output rate |

## 16. 逐 case 分析

每个 case 保存:

- waveform 和 speech activity;
- packet arrival timeline;
- chunk boundaries 与 lookahead;
- feature availability timeline;
- 每帧 teacher/student similarity;
- endpoint probability;
- incremental prediction;
- Thinker prefix token count;
- QA output;
- deadline misses;
- error label。

错误 taxonomy:

- `frontend_boundary`;
- `encoder_representation`;
- `future_leak`;
- `cache_state`;
- `endpoint_early`;
- `endpoint_late`;
- `thinker_alignment`;
- `runtime_deadline`。

强制审查:

- 所有 leakage 非零 case;
- deadline miss 最严重 20 例;
- early/late endpoint 各 20 例;
- offline 对、streaming 错 20 例;
- >30s 长音频 10 例;
- 噪声和中途长停顿各 10 例。

## 17. 交付物

1. `StreamingAudioEncoder` 契约;
2. causal student checkpoint;
3. teacher feature manifest;
4. frontend/encoder state 单测;
5. future leakage property tests;
6. Thinker incremental consume API;
7. 三臂配置与结果;
8. chunk/lookahead Pareto;
9. 实时 replay timeline;
10. `listener-v1` checkpoint;
11. 逐 case 浏览报告。

## 18. 复现清单

- [ ] teacher/student revision;
- [ ] waveform/hash/license;
- [ ] frontend 参数;
- [ ] teacher/student output rate 与 alignment map;
- [ ] chunk/lookahead/left context;
- [ ] timestamp convention;
- [ ] state schema/version;
- [ ] distillation alignment map;
- [ ] augmentation seed;
- [ ] real-time replay packet/jitter seed;
- [ ] endpoint threshold/hysteresis;
- [ ] Thinker prefix cache hash test;
- [ ] test 未用于选超参。

## 19. 前沿对照与改造方向

[Moshi](https://arxiv.org/abs/2410.00037)(第 01 课引过)把这个问题整个绕开:用户音频由流式 Mimi 连续变 token,和模型自己说的音频流在同一时间轴上并行进模型,听是常态,没有"收完整段再交接"的时刻,也就没有单独的 listener 阶段。[Qwen2.5-Omni](https://arxiv.org/abs/2503.20215)(也在第 01 课引过)的技术报告描述了音频编码器分块处理以支持流式输入,配合 Thinker-Talker 的流式生成;块长等细节以报告原文为准。本课论文节里的 [Seamless](https://arxiv.org/abs/2312.05187) 更进一步:SeamlessStreaming 用可学习的 simultaneous policy 决定"再多听一点还是现在就输出"——本课的声学 endpoint 只回答"说完没有",是它的退化特例。而 [LLaMA-Omni](https://arxiv.org/abs/2409.06666) 和 [Mini-Omni](https://arxiv.org/abs/2408.16725) 宣传的低延迟主要在输出侧,输入侧是否持续监听、encoder 是否有状态因果,正是第 20 节让你核对的问题:"流式"在不同系统里覆盖的范围差得很远。

规模问题:student 只有 4 层 512 维,standard 档蒸馏数据 100 小时,前沿系统的语音编码器和数据量大若干个量级,这部分砸资源能缩。机制问题(本课的内容能解决):有状态因果、逐帧可见 mask、增量 KV、真实时间评测,这套骨架与前沿同构。真正还缺的机制是听和说没在同一时间轴上并行——本课仍是半双工,这一刀在第 06、07 课。



1. **Conformer 换 Emformer 对照。** 补上 Step 3 没选的那一臂:用 4–8 层 Emformer 重训 student,复用同一 teacher cache 与 chunk/lookahead 配置。改动位置:`StreamingAudioEncoder` 主干替换,state dataclass 增加 memory bank 字段。预算:standard 档 100 小时音频,单臂 2 seed,4 卡约 1–2 天。预期:>30s 长音频桶上 per-chunk p95 更平稳,hidden cosine 与 Conformer 臂相当。失败判定:因果性测试或 B/C parity 不过,说明 memory bank 泄漏未来,回 Step 3 修 state。
2. **动态 chunk。** 语音活跃时用 160ms 小 chunk 压延迟,长静音期合并到 640ms 省算力。改动位置:replay 驱动器与 `push_chunk` 调用方;encoder 本体不动——5.2 的 mask 按 `input_time` 计算,天然支持变长投料。预算:无训练,推理 sweep 约 2–4 GPU-hour。预期:平均 chunk latency 下降,WER 与固定 320ms 相当。失败判定:deadline miss rate 上升或 endpoint latency 变差。
3. **joint CTC/semantic endpoint。** 把 Step 9 的纯声学判定升级:CTC 辅助头的 blank 概率持续走高是"内容说完了"的语义线索,把它和声学特征拼起来喂 endpoint head,标签不变。改动位置:endpoint head 的输入特征;CTC 头在 Step 5 已顺手训过。预算:只训 head,1 卡数小时。预期:长停顿桶上 false early cut 下降,endpoint p95 不升。失败判定:ECE 变差——概率不可信时先重新校准再比较。
4. **INT8 listener。** 把 student 量化到 INT8,重跑全部因果性测试与真实时间 replay。改动位置:仅推理侧量化,训练不动。预算:无训练,数 GPU-hour。预期:listener RTF 与 state memory 明显下降,WER 退化在预注册阈值内。失败判定:shared-prefix leak 被量化噪声顶破公差——量化版要重新预注册公差并说明理由,不得悄悄放宽标准。



- Emformer 论文"缓存历史避免重复计算"的结论可直接验证:造一个反例臂,每 chunk 重新编码全部历史 waveform(第 2 节禁止的第一条),与 stateful 版比 per-chunk 计算时间随音频时长的曲线。预期同方向:重编码版随时长线性上涨,stateful 版近乎常数;若 stateful 版也上涨,先查 left context 有没有截断(诊断表"长音频显存增长"行)。
- "输出流式、输入整段"设计(LLaMA-Omni/Mini-Omni 一类)的代价可用三臂 A 和 C 复现:画 user-end→first-audio 对用户音频时长的曲线。预期 A 随时长上升(整段编码加 full prefill 全压在 endpoint 后),C 近乎平坦;方向反了,先查 C 是否偷偷 full re-encode。

**更多顺手扩展**:

- 限定延迟下的 lookahead predictor;
- multilingual student;
- 把[第 02 课](02_multimodal_connector.md) winner connector 接入;
- packet loss/jitter concealment;
- 多 session continuous batching;
- 为[第 07 课](07_full_duplex_routing.md)暴露可并发读取的 `user_memory`,但本课不启动生成时监听。

## 20. 必读论文与阅读问题

每篇带一个能在自己代码或实验里核对答案的问题。

### 20.1 Streaming encoder

- [Emformer: Efficient Memory Transformer Based Acoustic Model](https://arxiv.org/abs/2010.10759):
  读 architecture、memory bank、left/right context 与 latency 实验。带着两个问题:memory bank 存的东西对应你 Step 3 state dataclass 的哪个字段;right context 相当于本课哪个参数、有没有算进延迟账(对照 5.1 的 $L$)。阅读后写出:memory 避免重复计算的方式,以及 right context 对延迟的贡献。
- [Conformer](https://arxiv.org/abs/2005.08100):
  读 macaron FFN、MHSA、convolution module。带着问题:原版卷积是对称 padding、两边都看的,改成 causal 后 padding 怎么改、attention mask 怎么改。阅读后写出这两处变化,对照诊断表 "future leakage" 一行。

### 20.2 Teacher

- [SenseVoice 官方仓库](https://github.com/FunAudioLLM/SenseVoice):
  看 frontend、encoder 输出 rate 与模型许可。阅读后核对:当前 MiniMind 注入的 `valid_len` 是否等于实际 encoder 输出长度——Step 4 实测帧率的分子分母都取决于它。
- [MiniMind-O](https://arxiv.org/abs/2605.03937):
  重读 audio input pipeline 与实时交互部分。阅读后标注:论文中每处 "streaming" 分别覆盖输入、输出还是 UI 行为,并与第 01 课"near-duplex 不是真双工"的结论对上。

### 20.3 流式 S2S 参照

- [Seamless: Multilingual Expressive and Streaming Speech Translation](https://arxiv.org/abs/2312.05187):
  读 SeamlessStreaming 与 latency/evaluation。带着问题:simultaneous policy 的输入含语义状态,输出是"读还是写"的动作;本课声学 endpoint 的输入输出分别是什么。阅读后写成两行对照,第 06 课设计 turn policy 时直接用。
- [LLaMA-Omni](https://arxiv.org/abs/2409.06666):
  读 speech encoder/adaptor 与 streaming speech decoder。阅读后核对:低输出延迟是否同时包含持续监听,答案与第 19 节顺手复现第二条的曲线连起来。
- [Mini-Omni](https://arxiv.org/abs/2408.16725):
  读 batch-parallel generation 与 streaming 输出。阅读后核对:输入 encoder 是否为有状态因果实现;不是的话,属于第 2 节禁止清单的哪一条。

读完材料回头看系统:耳朵已经流水线化——声音按真实到达时间进来,逐 chunk 变特征、推进 Thinker 的记忆,endpoint 一到就在已有前缀上直接开口,还有一套证明"没偷看未来"的测试守着。但开不开口仍由死板的声学规则说了算:`RealtimeSession` 靠约 800ms 静音判定你说完,"嗯——"会被当成让位,一声咳嗽能打断它的长篇回答。下一课([第 06 课](06_turn_policy.md))把这条规则换成学习式 turn policy:判断该接话、该继续等、该短促回应一声,还是该停下自己的话。
