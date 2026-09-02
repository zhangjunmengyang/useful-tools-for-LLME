---
id: 19_capstone_thinker_talker
title: "Thinker × Talker 毕业系统"
summary: "让现代感知 Thinker 用 hidden-state bridge（跳过文字、直接递内部状态的桥）接上 Talker，能比文字桥留住更多跨模态和语气信息吗？这套桥还进得了能暂停、重规划、恢复的双工系统吗？"
unit: frontier
play_tools: []
checkpoints:
  - "保留三个逐级可诊断的版本：文字桥、hidden bridge、双工会话层，出问题知道该查哪层。"
  - "抓取并比较 Nemotron 不同层的 hidden state，只训一个小 adapter，30B 的 Thinker 一个参数不动。"
  - "把故障拆到五个环节定位：perception、reasoning、bridge、speech rendering、turn policy。"
  - "交出带完整时间戳、逐 case replay、TTFA/RTF/stop/replan 指标的毕业原型。"
---

# 第 19 课：集成 Nemotron Thinker 与 MiniMind Talker

> 类型：核心毕业实验<br>
> 建议周期：2–3 周<br>
> 推荐硬件：默认可复现实验使用 8×80GB；8×48GB 只运行已缓存 hidden-state 的 bridge 训练，或另开低精度 Thinker 扩展<br>
> 独立性：可以只使用**完整的** MiniMind-O checkpoint、Nemotron checkpoint 与本课指定的 streaming ASR 开始，不要求先完成第 1–18 课

## 1. 把 Thinker 和 Talker 接起来

前 17 课在 26M MiniMind-O 上完成了流式输入、视频、MoE、混合架构和后训练；[第 18 课](18_nemotron_finetuning.md)验证了 30B-A3B Nemotron 3 Nano Omni 的多模态理解与微调。本课把两条工作线合并：Nemotron 负责理解与文本推理，MiniMind-O Talker 负责语音生成，实时会话层负责监听、中断与恢复。

系统包含以下模块：

- **Thinker**：冻结的 Nemotron 30B，接收 text/image/video/audio 并生成回答条件。
- **Talker**：冻结的 MiniMind-O 4 层 Talker，每个位置接收语义条件向量、8 路 Mimi code 历史和说话人条件。
- **clock**：冻结的 MiniMind Thinker。它对定稿文本执行 teacher forcing，产生 Talker 所需的逐位置 `bridge_states`。
- **bridge**：唯一训练的模块，是一个 10M–100M 参数的 adapter，用来对齐 tokenizer、hidden size 和时间轴。
- **Mimi codec**：将每帧 8 路 code 解码为波形。
- **listener**：FunASR 流式 Paraformer，在用户说话期间持续转写。
- **realtime session**：由 VAD、120ms 快速分类器、事件总线和调度器组成，处理打断、继续和重新规划。

公开的原生全模态系统通常联合训练理解与语音生成模块，本课采用已有 checkpoint 的组合方案。19A 使用文字接口，稳定但会丢失未显式写入文本的信息；19B 使用 hidden state 接口，需要解决跨 tokenizer 对齐和时间轴转换；19C 加入实时会话层，验证打断与恢复。

默认交付是双工回放：listener 与 Talker 真实运行，Thinker 事件按预先记录的时间戳回放。它可以验证 bridge 和调度器，但不能证明在线 Thinker 的端到端延迟。只有额外完成在线实验，结果才能称为在线真双工。

本课术语：

| 术语 | 简要解释 |
|---|---|
| 19A / 19B / 19C | 三个逐级版本：文本桥、隐状态桥、双工会话层 |
| bridge_states / clock | MiniMind Thinker 逐位置产出的条件向量序列，Talker 每步吃一个，像节拍器 |
| adapter | 外挂的小可训练模块；本课唯一动梯度的零件，10M–100M 参数 |
| T-Embed / H-Mid / H-Last | 主矩阵三臂：桥的 K/V 分别取自 token embedding、中间层、最后一层 |
| stable frontier | 已确定不会被后续 token 改写、允许发给 Talker 的字节前缀 |
| guard_tokens | 对齐时故意扣住不提交的末尾 2 个 token，防 tokenizer 事后反悔 |
| duplex replay | 真 listener 真 Talker，但 Thinker 事件按原时间戳回放；默认验收等级 |
| AEC | 回声消除：从麦克风信号里减掉自家扬声器的声音，不然模型会听见自己 |
| KV cache | 注意力的键值缓存，存着已读过的上下文，免得每步从头重算 |
| rank-local replica | 8 个进程各自加载一份完整模型，参数不切分，各算各的数据 |

## 2. 本课要完成的系统

本课把已经独立验证的感知、推理、语音生成和实时调度模块组成一个系统。分工如下：Thinker 负责理解 text/image/video/audio 输入并生成回答条件；Talker 根据这些条件和历史 Mimi code 生成语音；bridge 把两者的时间轴和 hidden size 对齐；realtime session 负责监听、播放、中断和恢复。

先划清默认能验收什么。默认实验先验证可持续监听、流式发声、打断和恢复的**双工回放系统**：Listener 与 Talker 真实运行，Thinker 事件按原时间戳回放。只有额外完成在线端到端实验后，才能把结果称为在线真双工。验收依据是逐阶段运行记录、离线与在线一致性、语音内容指标和中断时间线；演示视频顺不顺畅不算数，测量才算。

本课有三个逐级版本：

1. **19A 文本桥**：Nemotron 输出文本；冻结的 MiniMind Thinker 将该文本展开成 Talker 所需的逐位置 `bridge_states`，再由 Talker 生成语音；
2. **19B 隐状态桥**：在同一 MiniMind 因果时间轴上，用 adapter 注入 Nemotron 中间表征；
3. **19C 双工会话层**：因果 listener、Thinker、Talker 并行运行，支持 `continue/pause/replan`。

三种版本都要保留，造完新的也不许扔旧的。19A 提供对连续表征依赖最少的文本基线；当 19B 或 19C 出错时，先用同一输入重放 19A，一步区分故障位于 bridge 还是实时调度。

本课使用三个延迟缩写（第 01 课的老朋友）。首个文本 token 延迟（time to first token，TTFT）是请求就绪到 Thinker 产生第一个 token 的时间；首个可播放音频延迟（time to first audio，TTFA）是请求就绪到第一段 PCM（可直接播放的原始音频）已完成解码并可以播放的时间；实时率（real-time factor，RTF）是处理时长除以音频时长，数值越低表示生成越快。TTFT 和 TTFA 只用于在线 Thinker 扩展；默认的缓存回放不报告这两个指标——回放里的"请求时刻"是假的，报出来就是自欺。

再强调一遍 Talker 的挑食属性：MiniMind-O 的 Talker 不能作为接受任意文本的独立 TTS 使用。它在每个序列位置同时消费 MiniMind Thinker 的 bridge-layer state、此前已经生成的 8 路 Mimi code，以及 speaker/reference-code condition。缺少其中任何一项，都不是原模型定义的 Talker 输入。

因此实验加载并冻结完整的 `jingyaogong/minimind-3o`。公开资产中没有可替代该依赖的官方 Talker-only checkpoint。加载后应检查 Thinker、Talker、speaker projection 和 codec projection 的参数均来自同一 revision。

## 3. 研究范围与能力边界

本项目只验证 GPT-4o 类系统中的部分公开机制，不声称复刻其完整能力：

- Nemotron 3 Nano Omni 原生接收文本、图像、视频和音频，但输出文本；
- MiniMind-O 能输出流式语音，但规模、数据和交互能力有限；
- 初始实现是模块化系统，没有经过大规模 interleaved data（多模态交错数据）的端到端统一预训练；
- 公开数据、训练预算、RL 环境、端到端安全训练都与闭源前沿系统不同。

本课交付一个开放、可训练且组件边界明确的 Omni 研究原型。报告分别量化 19A 原生冻结文本桥、T-Embed/H-Mid/H-Last 三个可训练 adapter 和 duplex routing 对内容、语音质量、延迟、显存和故障恢复的影响。后文只用 **19A** 指原生冻结文本桥；可训练的离散文本基线始终写作 **T-Embed**，两者不能混称为 “text bridge”——名字混了，后面统计比较的角色就混了。

## 4. 要验证的结论与失败条件

本课把 Pilot 能回答的问题和 Standard 能回答的问题分开，先说清楚每个规模的数据到底能支撑什么结论：

- **Pilot 工程假设**：在 `answer_text`、Talker、speaker/reference condition、8 路
  Mimi target 和解码设置全部相同时，H-Mid 相比 T-Embed 能降低 held-out
  codec-token CE（交叉熵，越低表示 codec 预测越准），且 `ContentWER` 不退化。它只检验**同一段文字的 codec 预测和朗读
  内容**，不能据此声称保留了韵律、非语言声学条件或跨模态信息。
- **Standard 研究假设**：在严格配对的
  `(multimodal_request, answer_text, performed_answer_audio, speaker_condition,
  prosody_label)` 上，H-Mid 相比 T-Embed 能改善预注册的韵律或模态条件指标。每个
  tuple 必须原样提供给所有 arm，不能从另一条记录借用目标音频或说话人条件。

19A 原生冻结 forced-text 文本桥只作外部回归基线，不进入统计比较中的 T-Embed arm 角色。
任一结论都只在对应质量指标改善且 bridge 计算耗时、显存和故障率同时报告时成立。
只有在线扩展通过后，才另外报告 TTFA。

实验还要比较以下设计变量：

1. Nemotron 哪一层最适合作为 Talker 条件？
2. Standard 中 hidden bridge 的收益来自输入模态，还是只来自更多连续特征？
3. 生成时继续监听会不会污染 Thinker context？
4. 打断后应恢复旧回答、重规划，还是开始新 turn？

H-Mid 和 H-Last 未超过 T-Embed 时，报告该负结果及对应的层和训练信号——负结果照样毕业，改结论才不行。
不能在 test 后更换 Talker、增加数据或改变采样参数来修改结论。

## 5. 进入条件

### 5.1 最低软件资产

先完成软件 preflight（起跑前检查），再生成训练 cache。每个资产都要有可执行 smoke test：

- [MiniMind-O](https://github.com/jingyaogong/minimind-o) 可推理 checkpoint；
- canonical lane 固定使用 [Nemotron 3 Nano Omni BF16](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16)；低精度 checkpoint 只属于 14.3 的独立扩展；
- 能用 teacher forcing 将一段既定回复送入完整 MiniMind-O，并抓取逐位置 `bridge_states`；
- FFmpeg/torchaudio 音频处理；
- 一套可保存每阶段时间戳的 tracing。

下表记录本课写作时核验过的入口。运行实验时仍需计算本地文件 SHA256，并写入
`artifacts.lock.json`；revision 字符串不能替代文件校验。

| 资产 | 固定 revision（核验于 2026-07-23） | 用途 |
|---|---|---|
| `jingyaogong/minimind-o` | `a10fa6c148ed274d66f96dc119689e93e01be823` | 代码 |
| `jingyaogong/minimind-3o` | `ee3febbd08cc5b2bd41c039c825a8934232fee33` | 完整 MiniMind Thinker–Talker |
| `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` | `24e67ea000b7c2837fc8f9488aa2008524fac8ba` | 默认 Thinker |
| `jingyaogong/mimi` | `b4e362bbfbba9444b162de486da40af6639e0b98` | 8 路 codec |
| `modelscope/FunASR` | `c3e147cc2bf9c007a1cc57d85e45200b1517b5ac` | `paraformer-zh-streaming` 参考 listener |

### 5.2 最低数据资产

这些数据分别用于 bridge 监督、voice regression、listener 和 duplex replay，不能相互
拼接来补缺失字段：

- `jingyaogong/minimind-o_dataset` 中 2,000 条固定 T2A 样本；
- `mythicinfinity/libritts_r` 中按 speaker hash 固定的 3–10 小时原始目标语音；
- Mozilla Data Collective 的 Common Voice Scripted Speech 25.0 `zh-CN` 中按 client/speaker 切分的 listener/duplex 语音；
- 100 条固定 text/image/audio/video 输入；
- 按本课 `synthetic_duplex_v1` recipe 生成的 120 条双声道会话，其中 20 条只用于最终回放。

### 5.3 可以完全不依赖前课的替代

没做过前面的课也能进场，但要用下表的固定替代，并在报告中同时记录替代项——别把规则基线或缓存
replay 写成 learned/online 能力。

| 前课能力 | 没完成时的替代 |
|---|---|
| 可插拔 codec | 使用 MiniMind-O 原生 Mimi |
| 流式 listener | 使用 FunASR `paraformer-zh-streaming` 的真实 cache/chunk 接口；不得把每 320ms 重跑离线 SenseVoice 称为因果 listener |
| learned TurnHead | 使用固定规则，但必须标记为规则基线 |
| MiniMind MoE/Mamba | 使用官方 dense 完整 MiniMind-O |
| 自建 GRPO checkpoint | 使用官方 Nemotron reasoning checkpoint |

## 6. 学习目标

完成本课后，应能：

1. 定义 Thinker、bridge、Talker、realtime session 的稳定接口；
2. 将 19A 原生冻结路径与 T-Embed/H-Mid/H-Last 放在同一评测框架中，同时保持各自
   的统计角色不混淆；
3. 抓取并解释不同 Nemotron 层的 hidden state；
4. 训练小型 adapter，而不误更新 30B Thinker；
5. 拆分 perception、reasoning、bridge、speech rendering 和 turn policy 故障；
6. 在 8 卡上制定可落地的 GPU placement（模型和任务在卡间的摆放方案）；
7. 严格评测在线扩展的 TTFT/TTFA、语音生成 RTF、stop latency 和 replan success；
8. 提交可逐 case 回放的毕业系统。

## 7. 原理:边造边讲

四个机制，每个按第 01 课立下的节奏走：直觉、机制、数学或精确定义、代码落点、验证。它们分别回答本课的四个基本问题：从大脑哪一层拔线、文字接口丢了什么、异步系统的时间怎么记、打断之后哪些状态必须活着。

### 7.1 表征层级:从哪一层拔线

桥的输入端插在 Nemotron 身上，插哪一层？这件事没有想当然的答案：同一个 token 位置，每一层"知道"的东西不一样。拔太浅，拿到的几乎是词表信息；拔太深，信息已经被压成"下一个词是什么"的判决，别的都扔了。

`hidden state` 是模型在某一层、某一 token 位置输出的连续向量。不同层接收的历史相同，
但训练目标使它们承担不同功能：

- embedding 层直接编码 token identity 和位置信息；
- 中间层可能包含已经融合的语义与模态信息；
- 输出前一层更接近 next-token classification boundary（下一个 token 的分类边界）。

对第 $l$ 层、第 $t$ 个回答 token，捕获对象是向量 $h^{(l)}_t$，整批张量 shape 为 `[batch, answer_token, hidden]`；`answer_token` 维只覆盖已接受的回答 token。“中间层更适合作为语音条件”只是假设，还没资格当结论。要在相同 adapter、数据和训练步数下捕获
早期、中间、后期和输出前 hidden state，再用内容、韵律和延迟指标验证。

`capstone/thinker/hidden_capture.py` 注册只读 hook；候选层固定为 NemotronH block 13、26、39、51（约 1/4、1/2、3/4 和最后一层，见第 13 节配置的 `capture_layers`）。

hook 开关前后 logits checksum 必须相同（Step 3），否则"观测"这个动作本身就改变了模型；层间对照只能在 Step 6 的同预算三臂里做，单独看某一层的 loss 曲线不构成证据。

### 7.2 文本瓶颈:文字接口是一次有损压缩

19A 把 Thinker 的全部思考压成一串文字再交给 Talker，好比让作曲家用短信告诉歌手怎么唱：内容能传到，唱法传不到。采样出一个 token，等于把整个概率分布坍缩成一个点，分布里"还想过什么"的信息全部丢弃。

19A 原生冻结文本桥只把 Thinker 已经离散化为文字的内容交给 Talker，因此可能丢失：

- 情绪和韵律；
- 非语言声学事件；
- 说话人细节；
- 图像/视频中没有被写入最终答案的条件；
- Thinker 对多个候选答案的不确定性。

连续信息减少也带来工程优势，这正是三个版本都要保留 19A 的原因：

- 接口稳定；
- 可审计；
- 可缓存；
- 易于替换 Thinker/Talker；
- 故障定位简单。

描述性回归用同一回答文本比较 19A 与 19B；主矩阵则比较
T-Embed/H-Mid/H-Last。若 hidden bridge 改善只来自不同文本或不同采样，则不能归因于
连续表征——这就是 10.5 强制"同一 `answer_text` 贯穿所有 arm"的原因。

`capstone/bridge/text_bridge.py`（19A 路径）与 `capstone/bridge/hidden_bridge.py`（19B 路径）；两条路径共享 9.3 的 Talker 接口。

主矩阵开跑前，逐 case 校验三臂的 9 路输入、labels、mask 与 voice-condition hash 全等（Step 6）；hash 不等时先修数据，不看指标。

### 7.3 异步实时系统:四个时钟对不上,才需要 trace

回合制系统只有一个时间轴，实时系统里"step"没有单一含义：耳朵、脑、嘴、扬声器各走各的钟。说"延迟 500ms"而不说从哪个钟的哪个刻度量到哪个刻度，等于没说。

本课分别记录四个时钟：

1. 麦克风输入时间；
2. listener 完成时间；
3. Thinker token 生成时间；
4. Talker audio frame 播放时间。

所有时间戳使用同一个 monotonic clock（单调时钟：只会向前走、不受系统对表影响的计时器）。

端到端延迟可以按四段拆开：

$$
t_{\text{play}}-t_{\text{mic}}
=(t_{\text{listen}}-t_{\text{mic}})
+(t_{\text{token}}-t_{\text{listen}})
+(t_{\text{frame}}-t_{\text{token}})
+(t_{\text{play}}-t_{\text{frame}})
$$

报告既给端到端延迟，也给四段时间差；这样可判断等待发生在 listener、Thinker、bridge、Talker 还是 playback queue。

`capstone/realtime/events.py`：所有事件携带 monotonic timestamp、session id 和递增 event id（见 Step 8）。

回放同一 session，按 event id 检查丢失、重复和乱序；任何"延迟异常好看"的数字，先查两个端点是否来自同一个 monotonic clock，再谈优化。

### 7.4 双工状态:打断的另一半是记住停在哪

用户插话时，系统不能失忆式重启：已经说到哪、听到哪、想到哪，都得留底，否则恢复时要么复读要么跳段。所谓双工，一半是并行，另一半是状态管理。

`full duplex` 表示系统在播放助手语音时仍持续接收和处理用户音频。一个 session 至少
维护以下状态：

| 状态 | 保存的事实 |
|---|---|
| `user_audio_state` | 已接收音频、listener cache 与输入时间 |
| `user_semantic_memory` | 已提交的用户语义 |
| `thinker_context` | 当前 prefill 使用的上下文 |
| `thinker_generation_state` | 当前解码位置与可取消生成状态 |
| `talker_codec_state` | 八路历史 code 与解码缓存 |
| `playback_cursor` | 已播放到的音频位置 |
| `turn_policy_state` | 当前轮次动作和 pending event |

执行 pause、resume 和 replan 时分别记录哪些状态被保留、复制或重建。两条不变量随时可查：`playback_cursor` 只能单调前进或显式重置并写明原因；已 commit 的用户语义在 replan 前不得被覆盖。

`capstone/realtime/session.py` 持有上表状态；`capstone/realtime/scheduler.py` 只通过事件总线改它。

重放同一 interrupt case，检查 playback cursor、codec state 和 committed user text
没有无理由归零。归零又说不出原因的，按状态管理 bug 处理，不算"模型行为"。

## 8. 推荐代码边界

在自己的实验仓库中新增：

```text
capstone/
  contracts.py
  thinker/
    nemotron_adapter.py
    hidden_capture.py
    cache_hidden.py
  bridge/
    text_bridge.py
    hidden_bridge.py
    resampler.py
    byte_alignment.py
  talker/
    minimind_talker_adapter.py
    streaming_audio.py
  realtime/
    events.py
    session.py
    scheduler.py
    echo_control.py
    fast_vad.py
  eval/
    offline_eval.py
    duplex_replay.py
    trace_report.py
```

这些目录按稳定接口分离模型适配、实时状态和评测。WebUI 只调用接口，不持有模型状态
或调度规则。单元测试应能在没有 WebUI 的情况下分别重放 Thinker、bridge 和 Talker——出问题时你要能单独摇某一个零件，不必每次把整台机器开起来。

## 9. 稳定接口

### 9.1 Thinker 接口

`ThinkerOutput` 同时返回文本、候选层 hidden state、模态 span 和 token 时间戳。
`answer_token` 维只覆盖已接受的回答 token；prompt 和 padding 由 mask 单独记录。

```python
class ThinkerOutput:
    text_tokens: list[int]
    text: str
    hidden_states: dict[int, Tensor]  # [batch, answer_token, hidden]
    modality_spans: list[dict]
    timestamps: list[float]

class Thinker:
    def prefill(self, multimodal_request) -> ThinkerState: ...
    def decode_step(self, state) -> ThinkerOutput: ...
    def cancel_and_reprefill(self, state, committed_user_text) -> ThinkerState: ...
```

默认 Nemotron 路径不依赖“生成中原地追加 self-KV”能力，因为公开接口没有提供该
保证。新用户语义 commit 后，参考实现先暂停 Talker，再取消当前 Nemotron decode。
随后把已播放回答摘要和新用户文本写入 session log，用更新后的上下文重新 prefill，
最后启动新的回答规划。trace 要分别记录这五步，不能只留下最终 `replan` 事件。

只有实现独立 cross-memory，并通过新旧 KV 在固定输入上的 parity 测试（同输入下两条路径输出的一致性比对）后，才能另开
`append_user_memory` 路径。

### 9.2 Bridge 接口

```python
class BridgeOutput:
    clock_states: Tensor       # [batch, sequence_step, minimind_hidden]
    attention_mask: Tensor
    text_token_index: Tensor   # [sequence_step]，-1 只表示 prompt/control；EOS 后 tail 为 L-1
    nemotron_end_byte: Tensor  # [answer_token]，每个 Nemotron token 后的 UTF-8 累计字节
    minimind_span_byte: Tensor # [sequence_step, 2]，MiniMind token 的 [start, end)
    stable_byte_end: int       # 当前已经允许发给 Talker 的字节前缀
    audio_step_index: Tensor   # [sequence_step]，-1 表示尚未开始 audio
    text_fallback: str
    source_layer: int | None

class Bridge:
    def convert(self, thinker_output: ThinkerOutput) -> BridgeOutput: ...
```

#### 9.2.1 跨 tokenizer 的唯一因果对齐协议

Nemotron 与 MiniMind 使用不同 tokenizer，同一段文字通常会得到不同 token 数和边界。
因此两个 token 下标不能直接比较——"第 5 个 token"在两边根本不指同一段字。本课把两边都映射到**原始 UTF-8 字节轴**，且不做
Unicode normalization（字符规范化转换，转换本身会改变字节序列）：

1. 对已经接受的 Nemotron answer ids，始终使用
   `decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)`；
   chat/control token 由显式 token-id 表排除，不进入回答字节串。
2. 令 `D_i` 为解码到第 `i` 个普通 answer token 后的 UTF-8 bytes，
   `n_end[i] = len(D_i)`。必须逐步断言 `D_i.startswith(D_{i-1})`；
   不满足时该 tokenizer/模板不能走增量 hidden bridge，只能退回等待 EOS 的 19A。
3. 每收到一个新 Nemotron token，就对当前完整文本前缀重新运行 MiniMind fast tokenizer，
   开启 `return_offsets_mapping=True`。tokenizer 返回的字符 offset 必须通过
   `len(text[:offset].encode("utf-8"))` 转成 byte offset，禁止把 Unicode code point 下标当字节下标。
4. 在线只提交“与此前已提交 ids/spans 完全同前缀、位于 `D_{i-1}` 内、并保留末尾
   `guard_tokens=2` 个 MiniMind token”的最长前缀。EOS 时 `guard_tokens=0` 并提交全部。
   任一后续重分词改写已经提交的 id 或 span，立即触发 `ALIGNMENT_PREFIX_REWRITE`，
   停止 hidden bridge 并使用 EOS-buffered 19A；绝不回滚已经送入 Talker 的 clock。
5. 对 byte span 为 `[a_s, b_s)` 的 MiniMind answer token，定义
   `text_token_index[s] = min{i | b_s <= n_end[i]}`。prompt/control step 取 `-1`；
   audio tail step 在 Nemotron EOS 后固定为 `L-1`。因此第 `s` 步最多看见构成该
   MiniMind token 所需的最后一个 Nemotron token，不会看见未来 token。

离线 cache 也按 Nemotron token 逐个 replay 同一算法，不能一次性 tokenize 全文。
`alignment.json` 保存两套 token ids、全部 byte spans、stable frontier 和 fallback 原因。
golden set 用四项断言验证离线与在线路径一致：

| 断言 | 通过条件 |
|---|---|
| clock ids | `offline_replay.clock_ids == online.clock_ids` |
| token 对齐 | `offline_replay.text_token_index == online.text_token_index` |
| 已提交前缀 | `prefix_rewrite_count == 0` |
| bridge 数值 | `max_abs_diff(offline_bridge, incremental_bridge) < 1e-4` |

### 9.3 Talker 接口

```python
class Talker:
    def init(self, voice_prompt) -> TalkerState: ...
    def push_clock_state(self, state, bridge_output, step: int) -> None: ...
    def decode_audio_step(self, state) -> AudioFrame: ...
    def pause(self, state) -> Snapshot: ...
    def resume(self, snapshot) -> TalkerState: ...
```

先用 fake Thinker state 和两帧 codec state 完成接口测试，再优化 kernel。测试至少覆盖
初始化、连续两次 `push_clock_state`、pause/resume 和 EOS 后的 tail step。

## 10. 数据 recipe

### 10.1 样本字段

每条记录把输入、目标输出、交互事件和数据来源分开。`redistributable=false` 表示实验
可以保存派生指标，但交付物不能包含原始资产：

```json
{
  "id": "capstone-000001",
  "input": {
    "text": "可选文本",
    "image": "可选路径",
    "video": "可选路径",
    "audio": "可选路径"
  },
  "assistant": {
    "answer_text": "目标回答",
    "answer_audio": "原始目标波形路径",
    "speaker_id": "speaker-03",
    "emotion": "neutral"
  },
  "voice_condition": {
    "mode": "source_locked",
    "spk_emb_present": true,
    "spk_emb_file": "voice/capstone-000001.spk.f32le",
    "spk_emb_sha256": "sha256",
    "ref_audios_present": true,
    "ref_audios_file": "voice/capstone-000001.ref.i32le",
    "ref_audios_sha256": "sha256"
  },
  "timing": {
    "user_events": [],
    "assistant_events": []
  },
  "source": "dataset-name",
  "source_revision": "immutable-revision",
  "license": "SPDX-or-note",
  "redistributable": false,
  "split": "train"
}
```

`spk_emb_present` 和 `ref_audios_present` 是独立字段，不能用空 hash 代替。speaker
embedding 固定为 192 个 little-endian `float32`；`ref_audios` 固定为按 frame 交错的
little-endian `int32`，长度必须能被 8 整除。两个 SHA-256 都对这段规范化字节本身
计算，不对 JSON 文本或文件名计算。sidecar（主文件旁的附属记录文件）保存实际值，manifest 保存相对路径、
presence 和 hash；这样既能复现输入，也能区分“字段缺失”和“值恰好为空”。

### 10.2 三类数据

#### A. Bridge alignment

这类记录监督 Nemotron 表征到 MiniMind clock 的转换。四个对象必须来自同一条
assistant 回答：

- `conversations` 中最后一个 assistant 之前的对话前缀；
- 对最后一个 assistant 可朗读文本做 teacher forcing 得到的 Thinker hidden state；
- 同一可朗读文本经 MiniMind forced-text clock 得到的 query；
- 同一行 `answer_audios` 中对应的 8 路 Mimi codes。
- 同一行的 192 维 `spk_emb` 和 frame-interleaved `ref_audios`，包括它们的 presence、
  原始值和规范化字节 hash。

Pilot 主矩阵只使用逐行严格配对的记录训练 adapter，回答文本、目标 codes 和 voice
condition 在 T-Embed/H-Mid/H-Last 间逐项相同。100 条多模态输入仅作 held-out probe，
不与其他样本的音频拼接，也不作为多模态韵律监督。因此 Pilot 只能比较同文本 codec
预测和朗读内容，不能形成跨模态韵律结论。

Standard 若要检验跨模态韵律，必须另建 `pair_id`。每个 pair 至少包含两条输入模态不同、
但 `answer_text` 和 speaker/reference condition 完全相同的记录；每条记录的
`performed_answer_audio` 必须由该输入对应的预注册 `prosody_label` 真实演绎得到。训练、
dev、test 按 `pair_id` 整组切分，所有 arm 读取同一 tuple，禁止把另一条样本的音频
当作当前样本 target。

#### B. Voice rendering

这类记录只检查音色与声学渲染是否退化：

- 固定文本；
- 多说话人/多情绪目标音频；
- voice prompt。

同一文本和 voice prompt 要同时通过 19A/19B，比较 speaker similarity、音量和人工听感。

#### C. Duplex events

这类记录验证实时状态转移，不参与 bridge loss：

- 用户/助手独立声道；
- backchannel（"嗯嗯""对"这类附和声）、interrupt、side speech（用户跟旁人说话）、ambient noise；
- 每个事件的期望动作。

用户和助手声道保持独立，事件标签给出预期的 `pause/resume/replan`。混成单声道会
丢失事件来源，因此不能用于 19C 验收。

### 10.3 数据规模

规模表给出实验用途，不代表公开数据已经全部具备。每档开始前先生成 manifest，统计
有效样本、时长、speaker 和事件类型：

| 档位 | Bridge 样本 | 目标语音 | 双声道会话 | 用途 |
|---|---:|---:|---:|---|
| Pilot | 2k | 3–10h | 2h 合成 + 20 条真人 | 验证接口、过拟合和同文本 codec/content 指标 |
| Standard | 50k | 100–300h | 20–50h | 用严格 paired tuple 检验跨模态韵律 |
| Full research | 500k+ | 1k–5k h | 200h+ | 扩展语言/场景 |

### 10.4 可直接执行的 Pilot 来源

| Lane | 确切入口 | 固定 revision | 选择规则与许可 |
|---|---|---|---|
| bridge/Talker code target | [`jingyaogong/minimind-o_dataset`](https://huggingface.co/datasets/jingyaogong/minimind-o_dataset) 的 `sft_t2a_mini.parquet` | `d6588e12ac2ac8ced65eb58a7d7b3eef4aa220de` | 读取 `conversations`、`answer_audios`、`spk_emb`、`ref_audios`；按下述 `row_key` 排序后取前 2,000 个有效行；四个字段必须保持同一行，不能另配 voice condition；仓库卡同时列 Apache-2.0/GPL-3.0，必须逐来源保留原许可；这里拥有的是 Mimi codes，不推断拥有原波形 |
| raw target speech | [`mythicinfinity/libritts_r`](https://huggingface.co/datasets/mythicinfinity/libritts_r) `train.clean.100` | `0d1718db351207d5a493135a62d5ed8cc4733830` | speaker hash 划分，累计 10 小时停止；只用于冻结 Talker 的 voice-rendering regression，不进入 bridge loss，也不与 T2A 行拼接；CC-BY-4.0，保存 attribution |
| listener/duplex speech | [Common Voice Scripted Speech 25.0 `zh-CN`](https://commonvoice.mozilla.org/en/datasets) | dataset release `25.0` + 下载归档 SHA256 | 与中文 streaming listener 对齐；CC0-1.0，但遵守 MDC 不重托管要求，只发布 clip id 与派生事件 manifest |
| listener | [`modelscope/FunASR`](https://github.com/modelscope/FunASR) 的 `paraformer-zh-streaming` alias | code commit 见 5.1；首次下载后冻结 model snapshot SHA | `[0, 8, 4]` chunk 配置起步；模型卡许可单独归档 |
| duplex replay | 本课 `synthetic_duplex_v1` | recipe version `1.0.0` | 从 Common Voice held-out client 取两路音频；事件标签与混音参数可再分发，原音频不随课程重托管 |

### 10.5 `bridge_pilot_v1` 的 teacher-forced 配对构造

该 recipe 解决“文本、连续表征和 Mimi code 是否属于同一回答”的配对问题——配对一旦错行，三臂比较就全部作废，所以这一节的每条规则都值得较真。固定配置：

```yaml
recipe: bridge_pilot_v1
version: 1.1.0
source_file: sft_t2a_mini.parquet
source_revision: d6588e12ac2ac8ced65eb58a7d7b3eef4aa220de
required_columns: [conversations, answer_audios, spk_emb, ref_audios]
assistant_turn: last
spoken_text_rule: suffix_after_last_think_end_else_full
think_end_literal: "</think>\n\n"
nemotron_answer_mode: teacher_forced
minimind_clock_mode: teacher_forced
valid_rows: 2000
split:
  unit: duplicate_group
  assignment: sha256_bucket
  salt: bridge_pilot_v1
  train_buckets: [0, 79]
  dev_buckets: [80, 89]
  test_buckets: [90, 99]
audio_codebooks: 8
audio_pad_token: 2049
audio_stop_token: 2050
audio_spk_token: 2051
voice_condition:
  mode: source_locked
  missing_action: reject
  spk_emb_dtype: float32_le
  spk_emb_length: 192
  ref_audios_dtype: int32_le
  ref_dropout_probability: 0.0
attention_mask_mode: upstream_default_causal
```

按以下顺序逐行处理。相同输入 revision 和 recipe 应得到相同 manifest：

1. 将 `conversations` 解析为 JSON，取最后一个 `assistant` turn；此前所有 turn 是
   `request_messages`。`answer_text` 取 assistant content 最后一个
   `"</think>\n\n"` 后的 suffix；没有该字面量时取全文。禁止随机删除 system prompt、
   随机截断 turn 或使用训练集里的 augmentation。
2. 取与最后一个 assistant turn 同下标的 `answer_audios` flat list。要求非空且长度能被
   8 整除；按源码约定用 `flat[0::8] ... flat[7::8]` 解交织，并在每路末尾追加
   `audio_stop_token=2050`。不满足条件的行记录 reject reason 后跳过。
3. 默认 `source_locked` 要求同一行的 `spk_emb` 恰有 192 个有限数值，`ref_audios`
   非空、长度能被 8 整除且 code 位于允许范围。分别转成连续 `float32_le` 和
   `int32_le` 后计算 hash；不允许沿用上游 `__getitem__` 中随机丢弃 reference 的
   50% 分支。任一字段缺失、为空或非法时，记录 `missing_voice_condition` 后拒绝该行。
4. `row_key` 固定为
   `sha256(canonical_json(request_messages) || 0x00 || utf8(answer_text) || 0x00 || int32le(flat_codes) || 0x00 || float32le(spk_emb) || 0x00 || int32le(ref_audios))`。
   对全部候选行按 `row_key` 字典序排序；因此不依赖 parquet 行号或不存在的 `id` 列。
5. 用固定 Nemotron processor/chat template 编码 `request_messages` 和 assistant generation
   prefix；随后对**同一个 `answer_text`** 的 Nemotron ids 做 teacher-forced forward，
   `output_hidden_states=True`，不调用 `generate()`。保存 answer-token hidden states、
   logits checksum 和 9.2.1 的 Nemotron byte spans。
6. 用固定 MiniMind tokenizer 对同一个 `answer_text` replay 9.2.1，并以完整冻结
   MiniMind Thinker teacher forcing 产生 native clock query。随后按下面的九路协议
   构造 Talker 输入和 loss mask。对应 Mimi codes 是唯一 `L_audio_code` target；
   Nemotron 自由生成的不同答案绝不能与该行 codes 配对。
7. 超过任一模型上下文、回答为空、codebook 长度不等或发生
   `ALIGNMENT_PREFIX_REWRITE` 的样本拒绝；按 `row_key` 顺序继续扫描，直到最终 manifest
   恰有 2,000 行。
8. 对最终 2,000 行做精确重复和文本近重复聚类。共享同一 source asset hash、
   `source_locked` speaker hash 或归一化 `request_messages + answer_text` 模板的记录必须
   落入同一个 connected component（重复关系连出来的同一坨记录）。令
   `bucket = uint32(sha256("bridge_pilot_v1" || component_id)[:4]) % 100`：
   `0..79` 为 train、`80..89` 为 dev、`90..99` 为 test。输出
   `split_manifest.jsonl`，并断言 component 不跨 split；若任一 split 少于 100 行，
   recipe 直接失败，不能把 test 行移入 train。`fixed_voice` 模式不把全局固定 voice
   hash 当作 speaker identity，但仍执行 source/template 去重。

bridge 只读取 train。早停、层选择和超参数选择只读取 dev。三臂配置、checkpoint 和
解码参数冻结后，test 每个 seed 只运行一次；Pilot 中的 held-out codec CE 指这里的
test CE，不能用下文 100 条多模态 probe 替代。

`source_locked` 不是唯一合法选择，但一次实验只能选择一种 voice policy。若源数据缺少
任一条件而又必须保留这些行，先在 resolved config 中改为 `mode=fixed_voice`，并锁定
一组有授权的 192 维 speaker embedding 与 8 路 reference codes。该组条件必须替换
**所有**行、用于**所有** arm，并在 config 中保存两个 value hash；不能只给缺失行补值，
也不能混用 source voice 和 fixed voice。

MiniMind 的 speaker/reference prefix 位于 8 路 audio lane 中，`spk_emb` 则在
`audio_spk_token` 位置经冻结的 speaker projection 注入。令 `a` 为上游扫描完成后的
assistant audio anchor：前 50 个 assistant token 内找到 `"</think>\n\n"` 时移到该
字面量之后，否则保持 assistant start。令 `T` 为 padding 后长度，`F` 为目标音频 frame
数。九路张量必须按上游顺序构造：

1. 建立 `Y_audio[8,T]`，初值全为 `audio_pad_token=2049`；将 `ref_audios` 解交织成
   `R[8,R_len]`，令 `ref_start=max(1, a-R_len)`，把每路 reference 的右侧有效 suffix
   写入 `Y_audio[:, ref_start:a]`。
2. 在八路的 `ref_start-1` 位置都写入 `audio_spk_token=2051`，并把同一个
   `spk_emb[192]` 传给模型。speaker/reference prefix 只提供条件，不参与 audio loss。
3. 将每路目标 codes 追加 `2050` 后，令第 `q` 路第 `f` 个 target 的 pre-shift 位置为
   `a+q+1+f`。在这些位置同时写入 `Y_audio[q,pos]` 和
   `audio_target[q,pos]`；`audio_target` 其余位置为 `-100`。这样训练时
   `Y_audio[:, :-1]` 含有前序真值 code，推理时同一位置改用模型已经采样出的 code。
4. 做一次 next-token shift。各 tensor 的来源和 shape 如下：

| tensor | 构造方式 | shape |
|---|---|---|
| `X_audio` | `Y_audio[:, :-1]` | `[8,T-1]` |
| `X_text` | `minimind_text_ids[:-1]` | `[T-1]` |
| `input_ids_9lane` | 在第 0 维拼接 `X_audio` 与 `X_text[None, :]` | `[9,T-1]` |
| `audio_labels` | `audio_target[:, 1:]` | `[8,T-1]` |
| `text_labels` | `minimind_text_target[1:]` | `[T-1]` |
| `audio_loss_mask` | `audio_labels != -100` | `[8,T-1]` |
| `text_loss_mask` | `text_labels != -100` | `[T-1]` |

主矩阵沿用上游 forward 的 causal attention 行为，并把
`attention_mask_mode=upstream_default_causal` 写入 trace；不要把 loss mask 误当成
attention mask——第 01 课就分清过这两张表，一张管"看什么"，一张管"算谁的账"。T-Embed/H-Mid/H-Last 的 `input_ids_9lane`、两类 labels、两类 loss
mask、`spk_emb` 和 reference prefix hash 必须逐样本全等，只有送入 bridge 的 K/V
表征来源可以不同。

每个 cache record 至少保存：

```json
{
  "row_key": "sha256",
  "duplicate_group_id": "sha256",
  "split": "train|dev|test",
  "voice_condition_mode": "source_locked",
  "spk_emb_present": true,
  "spk_emb_sha256": "sha256",
  "ref_audios_present": true,
  "ref_audios_sha256": "sha256",
  "answer_text_sha256": "sha256",
  "nemotron_answer_ids_sha256": "sha256",
  "minimind_clock_ids_sha256": "sha256",
  "mimi_codes_sha256": "sha256",
  "input_ids_9lane_sha256": "sha256",
  "audio_labels_sha256": "sha256",
  "text_labels_sha256": "sha256",
  "audio_loss_mask_sha256": "sha256",
  "text_loss_mask_sha256": "sha256",
  "alignment_sha256": "sha256",
  "source_revision": "d6588e12ac2ac8ced65eb58a7d7b3eef4aa220de"
}
```

100 条 text/image/audio/video 输入只用于 held-out inference probe。先由冻结 Nemotron
确定性生成回答，再运行 19A/19B；这些记录不进入 adapter 训练。Pilot 的
modality-shuffle（把输入模态故意换错行，看输出会不会跟着变）只能检测推理是否使用多模态 hidden information，不能证明模型学到
有监督的多模态韵律映射。验证后一个结论需要另建 10.2 定义的严格同源 Standard tuple，
并固定 `pair_id`、`prosody_label`、speaker/reference condition 和 performed audio。

`synthetic_duplex_v1` 固定生成规则：

```yaml
seed: 20260723
sample_rate: 16000
events:
  backchannel:   {count: 30, onset_ms: [600, 2400], duration_ms: [120, 450]}
  interrupt:     {count: 30, onset_ms: [800, 4200], duration_ms: [700, 3000]}
  side_speech:   {count: 30, snr_db: [-5, 15], channel: environment}
  self_repair:   {count: 30, gap_ms: [80, 500]}
mix:
  echo_path: [clean, rir_small_room]
  packet_loss_pct: [0, 1, 3]
split_key: speaker_id
```

### 10.6 切分原则

切分目标是阻止同一说话人、源媒体或答案模板同时出现在训练和测试中。生成 split 后，
用 source hash、speaker hash 和文本近重复检测做自动审计：

- speaker 不得跨 train/test；
- 同一视频或音频源不得切片后跨 split；
- 同一答案模板的近重复要去重；
- 合成双工数据和真人双工数据分开报告；
- test 中至少一半事件时序不来自训练模板。

### 10.7 开放边界

以下字段决定数据能否训练、展示或再分发。缺失许可记录时，默认只保留 id、hash 和
派生指标：

- 记录每个语音数据集的说话人同意范围；
- 商业 TTS 生成数据要保留服务条款和可再分发性；
- 仅有 Mimi codes 时，不得记录或发布不存在的原始波形；
- 不要发布没有授权的真人双声道录音。

## 11. 实验实施步骤

### Step 0：环境与资源 preflight

先确认模型能在计划的 GPU placement 上完成单样本 forward。记录 GPU 型号、单卡 HBM（显存）
和 compute capability，NVLink topology（卡间高速互联的连接关系），CUDA/PyTorch/Transformers 版本，
`mamba_ssm` 与 `causal_conv1d` 版本，以及 MiniMind、Nemotron 和 codec 的精确 revision。

保存 profiler 中的 tensor device 和 host-to-device copy。出现隐式 CPU offload（框架偷偷把放不下的参数挪到内存）时停止，
不要用该运行的延迟作为基线。

### Step 1：建立 19A 文本桥

19A 解决“任意 Thinker 文本如何进入 MiniMind 原生 Talker 条件时钟”的问题。Nemotron
先根据 multimodal input 生成回答文本；MiniMind tokenizer 把这段文本编码后，冻结的
MiniMind Thinker 通过 teacher forcing 或 incremental forcing 产生逐位置
`bridge_states`。Talker 再把这些 state 与此前的八路 Mimi code 一起使用，最后由 Mimi
streaming decoder 输出波形。每个阶段都保存输入 hash、输出 shape 和耗时。

这里的 `voice_prompt` 指的是 10.5 中锁定的
`(spk_emb, ref_audios, presence, value hashes)`，一句自由描述文本不算数。19A 与三个可训练 arm 必须读取同一份
condition record；若 hash 不同，该 case 不进入对照表。

保存每阶段时间戳、文本 token、clock state checksum、audio code 和波形。

这里的 `teacher-forcing` 只把 Nemotron 已确定的朗读文本转换为 MiniMind Talker 所需的
因果时钟，不会把目标音频或未来文本提供给 Nemotron。在线版本只能提交 9.2.1 定义的
stable frontier。收到一个 Nemotron token 后，MiniMind tokenizer 仍可能改写末尾边界；
因此 prefix-monotone 断言失败时等待 EOS，并退回 EOS-buffered 19A。

### Step 2：冻结 19A golden set

这一步建立后续 bridge 和 duplex 修改的共同回归集。100 条输入按模态等量覆盖：

至少覆盖：

- 20 个纯文本；
- 20 个图像/OCR；
- 20 个短视频时序；
- 20 个语音问题；
- 20 个音视频绑定问题。

每条 golden case 保存：

| 产物 | 用途 |
|---|---|
| `thinker_text` | 锁定 Thinker 的回答文本 |
| `talker_input_tokens` | 检查跨 tokenizer 后的输入 |
| `voice_condition.json` | 保存 speaker/reference condition 与 hash |
| `input_ids_9lane.safetensors` | 保存 Talker 九路输入 |
| `loss_masks.safetensors` | 保存 text/audio loss mask |
| `audio_codes` | 比较 codec 生成结果 |
| `decoded_waveform` | 听检与 ASR 评测 |
| `ASR transcript` | 计算内容一致性 |
| `trace.json` | 保存时间、shape、版本和 fallback |

### Step 3：捕获 Nemotron hidden states

注册只读 hook，抓取候选层，同时验证 hook 开关前后的 logits checksum 相同：

- 早期 1/4 深度；
- 中间 1/2 深度；
- 后期 3/4 深度；
- 最后一层。

保存原始 dtype/shape/mask，再用只读低维投影检查：

- norm 分布；
- padding/模态 span；
- token 时间对齐；
- 同一答案的跨模态差异。

### Step 4：实现 hidden bridge

Talker 每个自回归位置都需要一个条件向量，固定 `K` 个 latents 不能直接替代该时钟。
本课以 MiniMind clock state 为 query，以 Nemotron answer state 为 key/value：

| 阶段 | 输入 | 输出 |
|---|---|---|
| query | MiniMind forced-text clock | `Q: [B,S,H_mm]` |
| key/value | Nemotron answer states | `K,V: [B,L,H_n]` |
| 维度投影 | `K,V` | 投影到 `H_mm` |
| 因果 cross-attention | `Q` 只读取 `K,V <= text_token_index(S)` | 逐时钟位置的融合 state |
| 输出整形 | gated residual 与 RMSNorm | `Z: [B,S,H_mm]` |

其中：

- `S` 包括 prompt、回答 token 和文本结束后的 audio tail step；
- `L` 是 Nemotron 已生成的回答 token 数；
- `text_token_index[s]` 严格按 9.2.1 的 UTF-8 byte span 公式计算，禁止用 token 序号比例、字符数比例或 DTW 猜测；
- tail step 只能看见已完成的全部文本，不能看未来 audio code；
- 8 路 `codebook lane` 不由 bridge 展开，而仍由原 Talker 的 `TalkerEmbedding`、MTP head 和 diagonal delay 处理；
- speaker/reference prefix、`input_ids_9lane` 和 audio/text loss mask 完全沿用 10.5；
  bridge 不得重新生成、移动或覆盖这些位置；
- attention mask 必须是 prefix-causal，离线/逐 token 输出做 `max_abs_diff < 1e-4` parity。

训练前打印每个张量的 `[B,S,H_mm]` 或 `[B,L,H_n]` shape，并用离线/增量 parity
验证 causal mask。参数量控制在 10M–100M；Nemotron、完整 MiniMind Thinker、Talker layers、
`TalkerEmbedding`、`TalkerHead.base`、8 个 `TalkerHead.adapters`、speaker projection
和 codec projection 全部冻结且处于 eval mode。主矩阵唯一可训练参数是 `bridge.*`。

### Step 5：训练 adapter

主矩阵固定使用下面的训练目标，避免不同 arm 获得额外监督：

$$
\mathcal L_{\text{bridge}}
=
\mathcal L_{\text{audio-code}}
+0.10\,\mathcal L_{\text{clock-cosine}}
$$

其中：

- `L_audio_code`：冻结 Talker 输出的 8 路 codec-token CE；先对每一路所有有效
  target position 取 mean，再对 8 路等权 mean。padding、prompt、speaker/reference
  prefix 不计 loss，`audio_stop_token=2050` 计入。有效位置只能来自
  `audio_loss_mask=(audio_labels != -100)`，不得用连续区间重新推断。
- `L_clock_cosine`：只在 answer-text clock position 上计算
  `mean(1 - cosine(Z_s, stopgrad(Q_native_s)))`；`Z_s` 是 bridge 输出，
  `Q_native_s` 是同一 `answer_text` 经冻结 MiniMind 原生 bridge layer 得到的 clock state。
  prompt 与 audio tail 不计该项。这一项是给 bridge 一根拐杖：让它的输出别离
  Talker 熟悉的原生时钟太远。
- 独立 ASR 的 WER/CER 只作评测，不进入梯度；主矩阵没有辅助 ASR、辅助文本头或可选 loss。

optimizer 创建前执行：

```python
for p in nemotron.parameters():
    p.requires_grad_(False)
for p in minimind.parameters():
    p.requires_grad_(False)
for p in bridge.parameters():
    p.requires_grad_(True)

trainable = {name for name, p in full_system.named_parameters() if p.requires_grad}
assert trainable and all(name.startswith("bridge.") for name in trainable)
```

训练日志保存排序后的 `trainable` 名称、总参数量和每模块 grad norm。任一 Talker 参数
出现梯度即中止；仅把 optimizer 学习率设为零仍会保留无意义的梯度和状态，不能作为冻结。

### Step 6：比较输入表征

主矩阵只比较三个 arm：

1. **T-Embed**：token-embedding adapter；
2. **H-Mid**：middle-layer hidden-state adapter；
3. **H-Last**：last-layer hidden-state adapter。

三个 arm 使用相同的 causal cross-attention adapter shape、参数量、batch order 和训练
步数。T-Embed 的 K/V 来自冻结 token embedding；H-Mid/H-Last 分别来自中间层和
最后层 hidden state。19A 始终指不训练 adapter 的原生冻结 forced-text 文本桥，只作
外部 reference，不进入三个可训练 arm 的显著性比较。

每个 batch 在 forward 前比较
`input_ids_9lane/audio_labels/text_labels/audio_loss_mask/text_loss_mask/spk_emb/ref_audios`
的 hash。任一 hash 在三臂间不一致时立即停止；否则观察到的差异可能来自 voice
condition 或 delay/mask，而不是 K/V 表征。

### Step 7：加入增量条件

离线路径通过后，再把完整 hidden sequence 改为增量 push：

- 每生成若干 text token 更新一次 bridge；
- bridge 使用 cache；
- Talker 不等待完整 Thinker 回答；
- 保持一个 text fallback 缓冲，避免 hidden adapter 崩溃时无声。

验证时逐 token replay 同一回答，比较增量与离线 clock state、audio code 和 EOS
位置。fallback 触发时 trace 要记录原因和已经提交的 stable byte。

### Step 8：实现 session scheduler

调度器负责 `listener_worker`、`thinker_worker` 和 `talker_worker` 三个并行 worker。
模型代码不直接操作播放队列；所有跨 worker 的状态变化都必须通过事件总线。

事件总线按职责分为四组：

| 事件组 | 事件 |
|---|---|
| 用户音频 | `USER_CHUNK`、`VAD_SPEECH_START`、`VAD_SPEECH_END`、`USER_ENDPOINT` |
| 停播判定 | `INTERRUPT_CANDIDATE`、`FAST_PAUSE`、`KEEP_PLAYING`、`BACKCHANNEL` |
| 生成 | `THINKER_TOKEN`、`BRIDGE_UPDATE`、`AUDIO_FRAME` |
| 会话控制 | `PAUSE`、`REPLAN`、`STOP` |

所有事件携带 monotonic timestamp、session id 和递增 event id；回放时据此检查丢失、
重复和乱序。

低延迟停播和语义识别共用经过 AEC 的 20ms PCM 与 WebRTC VAD。VAD 产生
`INTERRUPT_CANDIDATE` 后，一条路径累计 120ms 音频交给 fast classifier，尽快决定
`FAST_PAUSE` 或 `KEEP_PLAYING`；另一条路径继续累计到 480ms，更新 Paraformer cache，
再根据完整语义决定 resume 或 replan。两条路径读取同一批按时间排序的 frame，但维护
各自的累计状态。这是人耳反射和大脑理解的分工：先凭反射闭嘴，再靠理解决定后续。

`FAST_PAUSE` 不等待 480ms ASR chunk，但也不直接由 VAD 触发。固定实现使用
`webrtcvad==2.0.10`、16kHz mono PCM16、20ms frame、aggressiveness mode 2；最近 5 帧
中至少 4 帧为 speech 时发出 `INTERRUPT_CANDIDATE`。onset timestamp 取命中窗口中
第一帧的采集时间。候选事件到来后，scheduler 暂停向播放队列加入新帧，但保留
`playout_guard_ms=160` 的已排队音频；一个独立的因果快速分类器读取候选起点后的前
120ms 干净语音，其中已用于 VAD 的帧直接从环形缓冲读取。分类器只输出
`interrupt_like`、`backchannel_like` 或 `uncertain`。
`interrupt_like` 和 `uncertain` 触发 `FAST_PAUSE`，`backchannel_like` 在 guard 音频
耗尽前恢复入队。`FAST_PAUSE` 发出后 80ms 内必须停止实际扬声器输出。

快速分类器在训练集上拟合，在独立 dev 集上固定阈值，test 期间不得调整。它只负责区分
是否需要立即停播；完整语义仍由 streaming ASR 和 turn policy 决定。正常路径的结构时间
约为 `100ms VAD window + 至多 40ms 补足分类音频 + 80ms pause deadline`，给 p95
400ms 留出特征计算、AEC、线程调度和声卡缓冲余量。

`fast_interrupt_v1` 使用 `synthetic_duplex_v1` 中按 speaker 划分的 train/dev
事件训练：输入为候选起点后 120ms 的 80 维 log-mel（对数梅尔声谱特征），模型为两层单向 GRU
（一种轻量循环网络，hidden size 128）加三分类线性层。训练只使用 train split，分类阈值只在 dev split
固定；test split 和 20 条真人回放只用于最终评测。若数据量不足以使 backchannel
false-stop 达标，应报告未通过，不能用 test 调阈值。

参考事件循环：

```python
from collections import deque
import webrtcvad

vad = webrtcvad.Vad(2)
vad_window = deque(maxlen=5)
pcm_window = deque(maxlen=5)
vad_active = False
unvoiced_run = 0
cache = {}
asr_accumulator = bytearray()
candidate_audio = bytearray()
candidate_onset = None

for frame in microphone_stream(frame_ms=20, sample_rate=16000, pcm="s16le"):
    candidate_started_this_frame = False
    clean = aec.process(frame.pcm, playback_reference.current())
    voiced = vad.is_speech(clean, sample_rate=16000)
    vad_window.append((frame.monotonic_start, voiced))
    pcm_window.append((frame.monotonic_start, clean))
    unvoiced_run = 0 if voiced else unvoiced_run + 1
    asr_accumulator.extend(clean)

    if talker.is_playing and not vad_active and sum(v for _, v in vad_window) >= 4:
        onset = next(ts for ts, v in vad_window if v)
        event_bus.emit("VAD_SPEECH_START", onset=onset)
        event_bus.emit("INTERRUPT_CANDIDATE", onset=onset)
        playback_queue.hold_new_frames(guard_ms=160)
        candidate_onset = onset
        candidate_audio.clear()
        candidate_audio.extend(
            b"".join(pcm for ts, pcm in pcm_window if ts >= candidate_onset)
        )
        candidate_started_this_frame = True
        vad_active = True

    if candidate_onset is not None:
        if not candidate_started_this_frame:
            candidate_audio.extend(clean)
        if len(candidate_audio) >= 6 * 640:  # 120ms
            decision = fast_interrupt_classifier(bytes(candidate_audio))
            if decision in {"interrupt_like", "uncertain"}:
                event_bus.emit("FAST_PAUSE", onset=candidate_onset, decision=decision)
            else:
                event_bus.emit("KEEP_PLAYING", onset=candidate_onset)
                playback_queue.release_hold()
            candidate_onset = None
            candidate_audio.clear()

    if vad_active and unvoiced_run >= 10:  # 200ms silence，允许下一次独立打断
        event_bus.emit("VAD_SPEECH_END", at=frame.monotonic_end)
        vad_active = False
        vad_window.clear()
        pcm_window.clear()

    if len(asr_accumulator) == 24 * 640:  # 24 × 20ms；每帧 320 个 int16 sample
        partial = listener.generate(
            input=bytes(asr_accumulator),
            cache=cache,
            is_final=False,
            chunk_size=[0, 8, 4],
            encoder_chunk_look_back=4,
            decoder_chunk_look_back=1,
        )
        asr_accumulator.clear()
        event_bus.emit("USER_PARTIAL", partial)
```

`640` 字节对应 20ms × 16kHz × 2 bytes。AEC 必须使用实际 playback reference；没有 AEC、
缺少原始麦克风帧记录或逐帧 VAD 判断的运行不能通过验收。每个 20 ms 帧只能进入
VAD 和 accumulator 一次，ASR cache 单调推进，并用“未来尾部替换”测试证明早期输出不变
（这套因果监听纪律在[第 05 课](05_streaming_listener.md)立过规矩）。
若只用离线 SenseVoice 滑窗重算，必须标为 `windowed_concurrent_baseline`，不能通过本课
因果监听器验收。

任何 `FAST_PAUSE` 都立即暂停实际扬声器输出并保存 Talker snapshot；后续 FunASR partial
与 turn policy 再决定 `resume` 或 `replan`。`KEEP_PLAYING` 表示 guard 期间没有发生
实际停播。不能把 `FAST_PAUSE` 只写日志却继续播放，也不能把
`INTERRUPT_CANDIDATE` 记成已经停播。VAD 只负责发现候选语音，不能把 speech detection
直接当成语义 interrupt。

### Step 9：实现 19C

Turn policy 对每次用户事件只输出四个离散动作之一：`continue`、`pause`、`replan` 或
`ignore_backchannel`。动作值必须和输入 event id 一起写入 trace。

打断流程：

1. fast VAD 发出 `INTERRUPT_CANDIDATE`，播放队列进入 160ms guard；
2. 120ms 快速分类结果为 `interrupt_like` 或 `uncertain` 时发出 `FAST_PAUSE`，并在
   80ms deadline 内停止扬声器；结果为 `backchannel_like` 时继续播放；
3. 真实停播时保存 codec/Talker snapshot 与 speaker-stop timestamp；
4. 同一批 20ms frame 继续累计进入 streaming ASR，不能因分类或 pause 丢弃；
5. ASR semantic commit 后进入 Thinker；
6. policy 判为 backchannel 时保持或恢复播放，判为 interrupt 时 replan；
7. replan 时保留已播放内容的文本摘要，避免重复。

默认缓存回放实现标记为 **duplex replay**：助手播放期间 listener 始终工作，缓存的
Thinker event 按原 monotonic timestamp 回放。它验证 listener、fast VAD、Talker 与
scheduler，不验证在线 Nemotron。通过 14.2 的在线扩展后，Nemotron 必须在语义 commit
后执行 cancel + re-prefill，才能报告 `system_full_duplex=true`。再接入
[第 07 课](07_full_duplex_routing.md)的 gated cross-memory 并通过缓存一致性测试后，
才能报告 `model_full_duplex=true`。

### Step 10：正式回放评测

用预录制双声道事件按原 monotonic timestamp 自动 replay。运行期间不允许人工点击；
结束后检查预期事件数、实际动作、speaker-stop timestamp 和 replan 输出。

## 12. 受控实验矩阵

### 12.1 Bridge 主矩阵

主矩阵只改变 Nemotron K/V 的表征来源。adapter、MiniMind clock、Talker 和训练预算保持
一致——一次只动一个变量，是第 01 课以来的老规矩：

| Arm | 输入条件 | Trainable | 主要目的 |
|---|---|---|---|
| T-Embed | token embedding + MiniMind clock | 仅同形同参 `bridge.*` | 可训练的离散文本条件 |
| H-Mid | 中间层 hidden + MiniMind clock | 仅同形同参 `bridge.*` | 测中间表征 |
| H-Last | 最后一层 hidden + MiniMind clock | 仅同形同参 `bridge.*` | 测输出层表征 |

19A 外部 reference 不训练 adapter，直接用 Nemotron 文本驱动完整 MiniMind-O 的
原生冻结 forced-text realization。它检查新 bridge 是否使原生朗读退化，但不与
T-Embed/H-Mid/H-Last 作为同预算 arm 比较。

三个可训练 arm 开始前必须通过 `trainable_names.json` 全等检查；任何一臂解冻
Talker codebook adapter、speaker projection 或 MiniMind Thinker 都使主矩阵无效。
同时逐 case 检查 9 路 input/label/mask 与 voice-condition hash 全等；不能让某一臂随机
丢弃 `ref_audios`，也不能只给缺失条件的样本补固定 voice。

### 12.2 Duplex 子矩阵

先在 dev 上按预注册规则固定 bridge，再比较三种状态处理：

| Arm | 输入到达时行为 | 状态处理 |
|---|---|---|
| Hard cancel | 停止并清空当前输出 | 重建 context |
| Pause/resume | 暂停 Talker | 保存/恢复 state |
| Pause/replan | 暂停后更新 Thinker | 新计划 + 防重复 |

bridge 表征与 duplex policy 分开实验；同一张主表中只能改变其中一个变量。

## 13. 参考配置

下列 YAML 是 19B middle-layer arm 的 resolved config。运行前将所有 revision、维度、
loss reduction、alignment fallback 和 realtime deadline 写入最终配置：

```yaml
experiment: capstone_19b_mid_bridge

thinker:
  model: nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
  revision: 24e67ea000b7c2837fc8f9488aa2008524fac8ba
  precision: bf16
  frozen: true
  capture_layers: [13, 26, 39, 51]
  selected_layer: 26

bridge:
  type: causal_monotonic_cross_attention
  query_source: minimind_forced_text_clock
  kv_source: nemotron_middle_hidden
  hidden_dim: 768
  num_layers: 2
  text_fallback: true

alignment:
  axis: utf8_bytes
  unicode_normalization: "none"
  cleanup_tokenization_spaces: false
  minimind_guard_tokens: 2
  on_prefix_rewrite: fallback_to_eos_buffered_19a

talker:
  checkpoint: jingyaogong/minimind-3o
  revision: ee3febbd08cc5b2bd41c039c825a8934232fee33
  codec: mimi
  frozen: true
  eval_mode: true
  train_codebook_adapters: false

data:
  recipe: bridge_pilot_v1
  recipe_version: 1.1.0
  source_revision: d6588e12ac2ac8ced65eb58a7d7b3eef4aa220de
  valid_rows: 2000
  split_manifest: manifests/bridge_pilot_v1.split.jsonl
  split_assignment: sha256_duplicate_group_bucket
  train_buckets: [0, 79]
  dev_buckets: [80, 89]
  test_buckets: [90, 99]
  reject_component_cross_split: true
  answer_mode: teacher_forced
  voice_condition_mode: source_locked
  reject_missing_spk_emb: true
  reject_missing_ref_audios: true
  ref_dropout_probability: 0.0
  require_equal_9lane_hash_across_arms: true

loss:
  audio_code_ce_weight: 1.0
  clock_cosine_weight: 0.10
  text_consistency_weight: 0.0
  asr_training_weight: 0.0
  audio_reduction: mean_valid_positions_then_equal_codebooks

training:
  max_steps: 10000
  global_batch_size: 64
  lr: 2.0e-4
  bf16: true
  seed: 42
  trainable_modules: [bridge]
  fail_if_trainable_outside_bridge: true
  train_split: train
  checkpoint_selection_split: dev
  final_evaluation_split: test
  final_test_runs_after_freeze: 1

cache_backend:
  type: hf_rank_local_replicas
  world_size: 8
  transformers_version: 4.55.4
  torch_version: 2.7.1
  mamba_ssm_version: 2.2.5
  causal_conv1d_version: 1.5.2
  per_device_batch_size: 1
  max_sequence_length: 2048

realtime:
  listener: paraformer-zh-streaming
  chunk_size: [0, 8, 4]
  user_chunk_ms: 480
  max_lookahead_ms: 240
  fast_vad:
    package: webrtcvad
    version: 2.0.10
    frame_ms: 20
    mode: 2
    start_window_frames: 5
    start_voiced_frames: 4
    end_silence_frames: 10
    pause_deadline_ms: 80
  fast_interrupt_classifier:
    recipe: fast_interrupt_v1
    input_ms: 120
    features: log_mel_80
    architecture: causal_gru_2x128
    classes: [interrupt_like, backchannel_like, uncertain]
    threshold_source: dev_only
    uncertain_action: fast_pause
    playout_guard_ms: 160
  audio_frame_ms: 80
  queue_deadline_ms: 80
```

## 14. 从 CPU 机制练习到 8 卡正式实验

### 14.0 先完成页面顶部的 CPU 实验

页面顶部“跑通”部分使用小型确定性数组检查字节前缀提交、因果可见范围和
pause/replan 状态保留。它不生成真实语音，也不代表 30B 模型质量。CPU 检查全部通过
以后再阅读下面的八卡设计；没有满足正式门槛时，不要自行拼接或启动训练命令。

该产物只能标记为 `toy_fixture_passed`。它没有真实 Listener、Talker、PCM 播放设备或
monotonic runtime trace，因此不能据此声称“真实停播成功”“四个时钟已对齐”
“达到 80ms deadline”或“系统已经双工”。这些结论只能由 14.2 的在线运行和验收 C
中的设备时间戳支持。

### 14.1 Canonical：8×80GB rank-local BF16 cache

`rank-local replica` 表示每个进程在自己的 GPU 上加载一份完整模型，各 rank 之间不切分
参数。默认 lane 不使用 TP、EP、FSDP（三种把模型或优化器状态切到多卡的并行方式，回[第 13 课](13_distributed_8gpu.md)）或 `device_map="auto"`。8 个进程各自在一张
80GB GPU 上加载冻结的 BF16 Nemotron，并按 `row_key % 8` 独立处理数据。该配置无需
expert mesh，可以直接读取 `output_hidden_states`。省下的不只是显存工程量：不切参数，
就没有并行方式引入的数值差异要排查。

环境唯一版本：

| 项目 | 固定值 |
|---|---|
| Python | 3.11 |
| torch | 2.7.1 |
| transformers | 4.55.4 |
| accelerate | 1.10.1 |
| mamba-ssm | 2.2.5 |
| causal-conv1d | 1.5.2 |
| Nemotron revision | `24e67ea000b7c2837fc8f9488aa2008524fac8ba` |
| dtype | `bfloat16` |
| microbatch | 1 |
| max sequence | 2048 |

首次成功环境导出带 wheel SHA256 的 `requirements.lock`。版本或 wheel hash 改变时，
创建新环境 id，不覆盖原结果。`capstone.thinker.cache_hidden` 的加载路径固定为：

```python
import os
import torch
from transformers import AutoModel

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
model = AutoModel.from_pretrained(
    "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
    revision="24e67ea000b7c2837fc8f9488aa2008524fac8ba",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map={"": f"cuda:{local_rank}"},
).eval()
for parameter in model.parameters():
    parameter.requires_grad_(False)
```

这是正式实验，不是可以盲跑的示例。负责人只有在下列门槛全部通过后才能生成 launcher：

| 门槛 | 必须提供的证据 |
|---|---|
| 硬件 | 同一节点 8 张 80GB GPU；每卡单独完成 BF16 单样本 forward；`cpu_offload_count=0`；拓扑和 NCCL 小测已保存 |
| 模型 | 模型、processor、tokenizer 与 remote code 已完整下载；revision 和逐文件 hash 写入 `artifacts.lock.json`；离线加载通过 |
| 数据 | `bridge_pilot_v1.jsonl` 恰好 2000 行；schema、媒体 hash、`row_key` 唯一性和 split 审计全部通过 |
| 磁盘 | 用 golden sample 实测每行四层 hidden cache 的字节数；可用空间至少为 `bytes_per_row × 2000 × 1.2` |
| 代码 | `capstone.thinker.cache_hidden` 已安装；单进程 smoke、rank 0/7 parity 和中断恢复测试通过 |
| 环境 | 上表版本与 wheel hash 一致；8 个 rank 都在 Hub offline 模式下解析同一份本地 snapshot |

锁定的正式 runbook 必须把以下参数写入 resolved config，而不是依赖 shell 默认值：

| 参数 | 固定值 |
|---|---|
| 进程数 | 8，单节点，每 rank 一张 GPU |
| manifest | `data/bridge_pilot_v1.jsonl` |
| model/revision | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` / `24e67...8ba` |
| dtype / batch / length | BF16 / 每卡 1 / 2048 |
| capture layers | block 13、26、39、51 |
| 分片规则 | `row-key-mod-world-size` |
| 输出 | `cache/nemotron_bf16_24e67e` |

`--capture-layers` 使用从 0 开始的 NemotronH block index。Transformers 返回值中读取
`outputs.hidden_states[block_index + 1]`，因为 index 0 是 embedding output。模型文件必须
在启动前完整下载并写入 `artifacts.lock.json`，8 个 rank 在
`HF_HUB_OFFLINE=1/TRANSFORMERS_OFFLINE=1` 下读取同一 snapshot，禁止边跑边更新 remote code。

每个 rank 只处理 `int(row_key, 16) % WORLD_SIZE == RANK` 的记录，写
`rank-{RANK:02d}.safetensors` 与对应 JSONL index；进程之间不做模型通信。输出 hidden
立即转 CPU BF16，文件内保存 sample、模型、processor、chat-template、tokenizer 和
alignment hash。结束后必须恰好覆盖 manifest 一次，无重复、无缺行。

cache preflight 用同一条 golden sample 分别在 rank 0 和 rank 7 前向，并验证：

| 检查 | 通过条件 |
|---|---|
| 输出 token | `argmax_logits_equal == true` |
| selected hidden | `max_abs_diff < 1e-4` |
| 设备放置 | `cpu_offload_count == 0` |
| 行数 | `cached_rows == manifest_rows == 2000` |

Stage 2 读取固定 cache，再运行 8-way DDP（PyTorch 多卡数据并行，回第 13 课）。每卡放置完整冻结 MiniMind-O 和一份
可训练 bridge。正式 runbook 调用 `capstone.training.train_bridge`，使用
`configs/capstone_19b_mid_bridge.yaml` 和已经验收的
`cache/nemotron_bf16_24e67e`，并固定单节点 8 个进程。DDP 启动后再次断言全局
trainable names 只有 `bridge.*`。两个正式 entrypoint、resolved config 和
`requirements.lock` 共同构成 8 卡 recipe；它们不属于初学者默认执行路径。

### 14.2 在线端到端运行：未验证扩展

上述 canonical backend 是 hidden-state **缓存**后端，不验证 Nemotron、hidden
bridge 与 Talker 在同一在线进程中满足实时 deadline。普通 text-serving API 也不能被
假定会返回第 13/26/39/51 层 hidden states。

因此默认 19C 使用真实 20ms 用户音频、真实 listener、真实 Talker 和按原 monotonic
timestamp 回放的缓存 Thinker token/hidden events，最多报告
`duplex_replay_gate=true`。它只报告
`CachedEventToPlayableAudio = t_first_playable_audio - t_cached_thinker_event_release`，
不能报告在线 TTFT/TTFA，也不能写 `system_full_duplex=true`。

在线 hidden-bridge 必须另开实验 id，例如 `capstone_19c_online_unverified`。只有在提交
固定 runtime/container digest、完整 launch command、`device_mesh.json`、hidden/logits
parity、无 CPU offload 证据及端到端 trace 后，才能去掉 `unverified` 并报告实时指标。
课件不再给出未经实测的 `tp=8 或 ep=8` 选项。

### 14.3 8×48GB

48GB lane 默认不运行 BF16 Nemotron cache，只消费 14.1 的产物：

- 默认只消费 14.1 已生成的 BF16 hidden cache，执行 Stage 2 bridge DDP；
- 不在 48GB 卡上假装运行 BF16 cache backend；
- 官方
  [`Reasoning-FP8`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8)
  `@6647b845a4b786c6e2c7adb1b6a909e1aa71fac2`
  与
  [`Reasoning-NVFP4`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4)
  `@dc5f0b0bfddf8b6e0f5891475be9af05b80126fe`
  必须另开低精度实验和 cache；不得与 BF16 三臂混在同一因果矩阵；
- 不再使用“6–7 卡放 Thinker”这类没有 backend contract 的 placement。

### 14.4 8×24GB

24GB lane 的职责限定为 bridge 训练和较小 Thinker 扩展：

- 只训练使用既有 cache 的 bridge；
- 若换 3B–9B Omni Thinker，必须新建实验 id、重新生成三臂 cache，并明确不能称为
  Nemotron BF16 实验；
- CPU offload 或自制量化结果不进入默认实时指标。

### 14.5 缓存 hidden state 的实验作用

主实验比较 bridge 表征，因此先固定 teacher-forced Thinker hidden state，再让三臂读取
相同 cache。这样 Thinker sampling 和在线 scheduler 抖动不会改变训练输入——把 30B 大脑
"冷冻切片"，是让三臂比较只剩一个变量的最便宜手段。cache arm
不能提供 Thinker 在线延迟；端到端延迟只来自通过 14.2 条件的在线扩展。默认 replay
只报告 scheduler、VAD 和 Talker 延迟。

## 15. 指标与测量

量尺先定义再使用；每个数字都要能说清测的是什么、在哪个等级（online 还是 cached replay）测的。

### 15.1 理解能力

这些指标验证 Thinker 回答是否仍使用输入模态。每项按模态独立报告，并用 modality
shuffle 作为依赖性对照：

- text/image/video/audio 各自 held-out；
- OCR exact/ANLS；
- temporal order accuracy；
- audio-video mismatch accuracy；
- modality shuffle 后的性能下降。

### 15.2 语音内容

设 Thinker 文本为 `T`，输出语音经独立 ASR 得到 `A`。`ContentWER` 只测 Talker 是否
朗读了 Thinker 的内容：

$$
\operatorname{ContentWER}
=
\operatorname{WER}(T,A)
$$

同时对 ground-truth answer 测 WER/CER（词/字错误率，定义回第 01 课），避免 Thinker 本身错误被 Talker 指标隐藏。
Pilot 的 confirmatory 指标只包括 held-out codec-token CE 与这里定义的
`ContentWER/CER`；它们回答的是 codec 预测和朗读内容是否改善。

### 15.3 语音质量

内容正确不代表声音自然。本节固定 voice prompt 和响度处理，再测：

- UTMOS（自动预测听感自然度的模型评分）；
- speaker cosine similarity；
- 音量/静音比例；
- click/dropout rate；
- 20–50 条人工 MOS（人工听感打分）。

Pilot 中这些质量指标只作 voice-rendering 回归，所有 arm 固定同一 voice-condition
hash，不能解释为“hidden state 保留了跨模态韵律”。只有 Standard paired tuple 才按
`prosody_label` 报告组内差异，并将 H-Mid 对 T-Embed 作为预注册比较。

### 15.4 实时性

所有端点来自 trace 中的 monotonic timestamp。`first playable audio` 指第一段已完成
codec decode、可以提交播放的 PCM。TTFT 和 TTFA 只用于 14.2 已验证的在线扩展：

$$
\begin{aligned}
\operatorname{TTFT}
&= t_{\text{first thinker token}}-t_{\text{request ready}}, \\
\operatorname{TTFA}
&= t_{\text{first playable audio}}-t_{\text{request ready}}, \\
\operatorname{EndpointToAudio}
&= t_{\text{first playable audio}}-t_{\text{user end}}, \\
\operatorname{RTF}
&= \frac{T_{\text{processing}}}{T_{\text{audio}}}.
\end{aligned}
$$

默认缓存回放不能把缓存释放时刻写成请求就绪时刻，只报告：

$$
\operatorname{CachedEventToPlayableAudio}
=t_{\text{first playable audio}}-t_{\text{cached thinker event release}}.
$$

各项同时报 p50、p95、p99，并在表头标明 `online` 或 `cached replay`。

### 15.5 双工

双工指标按事件类型统计，不把 backchannel、side speech 和 interrupt 合并——三类事件的正确动作不同，合并平均只会互相掩盖：

- interruption stop latency；
- VAD candidate latency、快速分类延迟与 `FAST_PAUSE`→speaker-stop latency；
- backchannel false stop；
- side speech false trigger；
- replan success；
- resume coherence；
- overlap task completion；
- 80ms deadline miss ratio。

### 15.6 系统

系统指标用于解释延迟和崩溃来源。每个数值同时记录时间区间和 worker/GPU id：

- 每 GPU HBM；
- Thinker/Talker utilization；
- queue depth；
- worker idle time；
- host-to-device traffic；
- session crash/recovery。

## 16. 验收条件

验收的作用是把"能不能叫毕业"翻译成可检查的证据，四份验收各管一段。

### 验收 A：19A 原生冻结文本桥可用

验收 A 检查最简单的 19A 路径能否作为后续回归基线：

- 100 条 golden set 全部能完成；
- 输出音频可播放，无死锁；
- Thinker text 与输出 ASR 的内容一致性达到 MiniMind Talker 自身基线范围；
- 每阶段 trace 完整。

### 验收 B：hidden bridge 有可解释结论

验收 B 不要求隐藏状态桥接一定胜出；它要求比较条件公平，并能用分层指标解释结果：

- H-Mid、H-Last 与 T-Embed 使用同一数据与 Talker，逐 case 的 9 路
  input/label/mask、speaker/reference presence、value hash 全等；
- 至少 3 seed；
- 若质量提升，bridge 计算耗时和显存成本必须同时报告；在线扩展另报 TTFA；
- Pilot 只用 codec-token CE 与 `ContentWER/CER` 判断同文本 codec 预测和朗读内容，
  不写跨模态韵律结论；
- modality shuffle 只检测 hidden state 对输入模态是否敏感，不能证明正确使用；
- 只有 10.2 的 Standard paired tuple 才能报告跨模态韵律结论，并以 T-Embed 为主对照；
- 若无提升，能从层或训练信号解释。

### 验收 C：低延迟双工回放成立

下列阈值是本课 replay 配置的预注册验收条件，不代表所有产品的通用标准。必须同时满足：

1. Talker 生成时 20ms VAD frame 与 480ms ASR chunk 都持续消费，
   `consumed_vad_frames / emitted_vad_frames = 100%` 且
   `consumed_asr_chunks / emitted_asr_chunks = 100%`；
2. 输入/输出队列互不阻塞，80ms scheduler deadline miss `< 1%`；
3. 命中窗口第一 voiced frame 到 `INTERRUPT_CANDIDATE` 的 p95 `≤ 140ms`，
   `INTERRUPT_CANDIDATE` 到快速分类完成的 p95 `≤ 140ms`，
   `FAST_PAUSE` 到最后一个实际 DAC frame（数模转换器，声卡真正出声的最后一站）的 p95 `≤ 160ms`；
4. 标注为 interrupt 的事件，从用户 speech onset 到扬声器停止的 p95 `≤ 400ms`；
5. backchannel case 中，只有 `INTERRUPT_CANDIDATE` 不算停播；一旦 trace 记录
   `FAST_PAUSE` 且 DAC 时间线出现实际 audible pause，就计 false stop，false-stop
   rate `≤ 5%`；
6. side-speech false-trigger rate `≤ 5%`；
7. labeled interrupt 的 replan success `≥ 90%`；
8. `overlap_task_completion >= hard_cancel_baseline - 3 percentage points`；
9. pause/replan 不做全 session reset；每次重建的对象与耗时都写入 trace。

所有 latency 均用原始采集 monotonic timestamp 与实际 audio-device completion
timestamp 计算，不能用 ASR transcript 出现时间反推 speech onset。没有逐帧 VAD trace、
播放参考信号、AEC 记录或真实 DAC 停止时间戳不完整的运行不能通过验收 C。

同时单列三个等级，禁止越级命名：

- `duplex_replay_gate=true`：默认课内结果；真实 listener/Talker + 缓存 Thinker event replay；
- `system_full_duplex=true`：必须额外通过 14.2 的在线端到端 runtime 验证，因果 listener
  持续工作，实时 Thinker 可 cancel + re-prefill；
- `model_full_duplex=true`：还必须有可追加的 gated cross-memory、KV parity 与生成中语义更新证据。

### 验收 D：毕业交付完整

验收 D 检查配置、结果、样例和资源记录能否在另一台机器上重放：

- 19A/19B/19C 三套配置；
- 同一测试集结果；
- 20 个可回放 case；
- 故障分类报告；
- GPU-hours 与硬件账；
- 一键复现实验脚本。

## 17. 故障诊断表

调试老规矩：从最便宜、最可观测的一层开始查，先看 trace 再动模型。

| 症状 | 可能原因 | 先查什么 | 修复方向 |
|---|---|---|---|
| hidden bridge 语音乱码 | hidden span/mask 错位 | token 与模态 span trace | 先只喂 answer span |
| 文本正确、语音漏词 | Talker 条件更新太慢 | Thinker token→audio code 对齐 | 增大缓冲或训练增量条件 |
| 在线 TTFA 或缓存事件到可播放音频的延迟过高 | 等待完整 Thinker 输出或 bridge 未增量更新 | timeline | token/chunk 增量 push |
| 中间层不如最后层 | adapter 容量不足 | layer norm/线性 probe | 匹配参数并增加 resampler |
| 说话人漂移 | voice prompt 被 bridge 覆盖 | speaker embedding norm | 分离 semantic/voice condition |
| backchannel 触发 audible pause | 快速分类器误判或 guard 太短 | candidate/decision/FAST_PAUSE/DAC trace | 只在 dev 重现并调阈值或 guard；若问题来自已查看的 test，保留本次结果，另建实验并换新的未见 test |
| 打断后重复开头 | 未记录已播放内容 | playback cursor | 把已播放 transcript 写入 replan |
| 模型听见自己 | 无 AEC/声道混合 | mic waveform 与 speaker output | AEC 或干净输入声道 |
| 8 卡利用率低 | Thinker/Talker pipeline 气泡 | per-worker timeline | microbatch/异步预取 |
| 只在合成双工有效 | 训练模板捷径 | 真人 case 分组 | 增加真实时序与噪声 |
| hidden bridge 看似更好 | 使用了更多训练数据/参数 | manifest 与参数表 | 重新做公平对照 |
| 高 reward 但对话差 | 评测只看内容正确 | duplex/自然性分项 | 加独立交互指标 |

## 18. 逐个样例检查

每个失败 case 建立目录：

```text
cases/case_001/
  request.json
  input_audio_user.wav
  input_video.mp4
  thinker_text.txt
  thinker_spans.json
  bridge_stats.json
  output_audio.wav
  output_asr.txt
  events.jsonl
  timeline.json
  diagnosis.md
```

`diagnosis.md` 必须把故障归到一个首要层：

1. perception；
2. representation/bridge；
3. reasoning；
4. Talker/content rendering；
5. turn policy；
6. realtime/system。

`diagnosis.md` 必须选择一个首要层，并引用对应 trace 或 artifact；“模型整体还不够好”
不提供可验证原因。

## 19. 交付物

- `capstone_19a_text_bridge.yaml`；
- `capstone_t_embed_adapter.yaml`；
- `capstone_h_mid_adapter.yaml`；
- `capstone_h_last_adapter.yaml`；
- `capstone_19b_hidden_bridge.yaml`，记录最终选择的 hidden arm；
- `capstone_19c_duplex_replay.yaml`；
- 可选且只有实测通过后提交 `capstone_19c_online_verified.yaml`；
- bridge adapter checkpoint；
- 完整 MiniMind-O checkpoint lock，不得只写“Talker checkpoint”；
- `artifacts.lock.json` 与 `device_mesh.json`；
- `voice_condition.lock.json`，保存 policy、presence 与 value hash；
- `synthetic_duplex_v1` 生成器与 event manifest；
- 统一离线/在线评测脚本；
- 20+ case tree；
- latency trace viewer；
- `report.md`；
- 3–5 分钟未经剪辑、按 wall-clock 运行的 duplex replay 演示；
- 只有 14.2 验证通过时，才另交未经剪辑的在线端到端演示。

## 20. 复现清单

- [ ] 记录所有 commit/revision；
- [ ] 保存 tokenizer 与 chat template；
- [ ] 保存 UTF-8 byte spans，prefix rewrite 为 0；
- [ ] 固定 `spk_emb/ref_audios` 的 presence、规范化值与 SHA-256；
- [ ] 缺失 voice condition 时拒绝该行，或在所有行、所有 arm 统一使用同一 fixed voice；
- [ ] reference dropout 为 0，三臂不得随机改变 presence；
- [ ] 固定 sampling temperature/top-p；
- [ ] 固定数据 manifest 和 split hash；
- [ ] 记录量化格式；
- [ ] 保存 GPU topology；
- [ ] 离线与在线指标分开；
- [ ] 合成与真人双工分开；
- [ ] 19A/19B 使用同一 Talker；
- [ ] T-Embed/H-Mid/H-Last 的 `trainable_names.json` 全等且只含 `bridge.*`；
- [ ] T-Embed/H-Mid/H-Last 使用同形同参 bridge adapter；
- [ ] `bridge_pilot_v1` 恰有 2,000 条 teacher-forced 严格配对记录；
- [ ] 三臂逐 case 的 9 路 input、labels、loss masks 与 voice-condition hash 全等；
- [ ] 主矩阵 loss 固定为 `1.0 L_audio_code + 0.10 L_clock_cosine`；
- [ ] 记录 MiniMind forced-text clock 与 8 路 codebook lane 对齐；
- [ ] listener 通过未来泄漏与 chunk-consumption 测试；
- [ ] 保存每帧 VAD decision、快速分类结果、FAST_PAUSE、AEC reference 与 DAC stop timestamp；
- [ ] `duplex_replay_gate`、`system_full_duplex` 与 `model_full_duplex` 分开报告；
- [ ] 默认 cache launcher、版本锁和 8 个 shard 通过覆盖/parity 检查；
- [ ] 19C 不靠人工点击触发；
- [ ] 所有事件使用 monotonic clock；
- [ ] 失败 case 可独立回放。

## 21. 前沿对照与改造方向

本课的核心问题——脑子想的怎么变成嘴里的话——前沿系统给出了四种答案，按脑和嘴的耦合程度从松到紧排：

1. **不接嘴。** [Nemotron 3 Nano Omni](https://arxiv.org/abs/2604.24954)（必读 2）干脆不做语音输出：多模态进、文本出。这是本课的起点，也是很多"想给现成大模型加语音"的团队的起点。
2. **后接式桥，本课做法。** 两个分别训好的模型，中间训一个小 adapter。前沿系统里没有谁公开以此为主路线，原因在预算——大厂付得起下面第 3 条路；预算有限时，桥是唯一现实选项，本课教的就是把它做严谨。
3. **原生联合训练的 Thinker-Talker。** [Qwen2.5-Omni 技术报告](https://arxiv.org/abs/2503.20215)（第 01 课引过）的 Talker 同时接收 Thinker 的高层隐表征和采样出的离散文本 token——相当于把本课的 H-Mid 和 T-Embed 两路条件一起喂，而且两者在同一棵模型树里端到端联合训练，梯度能穿过"桥"传回 Thinker。[Qwen3-Omni](https://arxiv.org/abs/2509.17765)（必读 4）把 Thinker、Talker 都换成 MoE 并做流式多码本 codec 生成，技术报告给出理论端到端首包延迟 234 ms。[Qwen3.5-Omni / ARIA](https://arxiv.org/abs/2604.15804)（必读 5）再往前一步：text-speech 时钟的推进比例不再固定，随生成状态动态对齐——正对着本课"9.2.1 静态字节对齐 + 固定 diagonal delay"这套方案的软肋。
4. **脑嘴从未分开。** [Moshi](https://arxiv.org/abs/2410.00037)（必读 3）单模型建模双音频流，文本作为 inner monologue 和音频锁在同一个 12.5 Hz 时钟上，摘要给出理论延迟 160 ms、实测约 200 ms。桥接问题在它那里不存在——没有两个模型，就没有桥。

GPT-4o 放在坐标外单说：OpenAI 的公开介绍写明它是跨文本、视觉、音频端到端训练的单一模型，音频输入响应最快 232 ms、平均 320 ms，接近人类对话反应速度；至于内部怎么把"想"和"说"接在一起，其公开资料未给出该细节。所以它给本课的参照是延迟量级，而非可对照的机制。

先把差距分账。规模问题（钱能解决）：我们的 Talker 只有 4 层、hidden 768，adapter 只有 10M–100M 参数，Pilot 只有 2,000 行配对数据；这决定音质和韵律的上限，但不改变结论的方向。机制问题（本课教的东西能解决，或者说差距本身就是本课的教学内容）有三条。其一，梯度不过桥：Qwen 系的 Talker 与 Thinker 共享数据、时钟和梯度路径，我们两端全冻结、唯一自由度是 `bridge.*`，所以"hidden 里有信息"不等于"Talker 用得上"——这正是主矩阵要测的东西。其二，时钟是静态的：MiniMind clock 由 forced-text 驱动、EOS 后进 audio tail，ARIA 式动态对齐我们没有。其三，等级差一档：Moshi 和 GPT-4o 的延迟是在线数字，我们默认只有 `duplex_replay_gate`，在线 cancel + re-prefill 属于 14.2 的未验证扩展——拿 `CachedEventToPlayableAudio` 去和 232 ms 比较是端点定义错误，14.2 不通过就没有可比的数字。

三个实验都对准桥接方式本身，从便宜到贵排列。

1. **双条件桥：把 H-Mid 和 T-Embed 的 K/V 拼在一起（Qwen2.5-Omni 式）。** 改 `capstone/bridge/hidden_bridge.py` 的 K/V 构造：沿序列维拼接中间层 hidden 与冻结 token embedding，加一个二值 source embedding 区分来源；两路的因果可见范围各自沿用 9.2.1 的 `text_token_index` 规则。为公平，压缩 adapter 宽度或层数使总参数与单臂持平，新增 `capstone_h_mid_plus_t_embed.yaml`。预算：完全复用 14.1 的 BF16 hidden cache（embedding 路 K/V 来自冻结 embedding，无需新 cache）；训练与主矩阵单臂完全同预算——10000 step、global batch 64、3 seed，即在主矩阵之外新增约三分之一卡时。预期：held-out codec-token CE 不高于 H-Mid 与 T-Embed 中较好的一臂，`ContentWER` 持平；若成立，说明离散文本与连续 hidden 是部分互补的条件，这正是 Qwen2.5-Omni 同时喂两路的动机。失败判据：3 个 seed 的 test CE 全部高于 H-Mid 单臂（说明拼接稀释了注意力，检查 source embedding 与 mask）；任一 batch 触发 9 路 hash 不一致则直接作废——那说明拼接实现动了不该动的输入。
2. **流式提交扫描：文本桥换流式 hidden 桥，延迟到底卡在对齐还是调度。** 无需训练，复用已训好的 H-Mid adapter。把 `capstone/bridge/byte_alignment.py` 的 `guard_tokens` 从固定 2 改成可配置，在 {0, 2, 8, 等待 EOS} 四档上对 100 条 golden set 做逐 token replay；同时在 `capstone/realtime/scheduler.py` 记录每次 `BRIDGE_UPDATE` 到首个新 `AUDIO_FRAME` 的间隔。预算：无训练，单卡推理 replay，数小时量级。预期：guard 越小，`CachedEventToPlayableAudio` 越低，但 `ALIGNMENT_PREFIX_REWRITE` 与 19A fallback 次数上升；"等待 EOS"档等价于 EOS-buffered 19A，延迟最高、rewrite 恒为 0。交付一张"提交激进度、延迟分位数、fallback 率"对照表。失败判据：guard=8 仍出现 prefix rewrite，说明 tokenizer 改写超出末尾局部范围，回 9.2.1 查断言 2；延迟不随 guard 单调变化，说明瓶颈在调度或 Talker 解码，不在对齐——别再折腾桥。
3. **恒定时钟桥：把 forced-text 时钟换成 Moshi 式固定速率时钟。** 这是"重写桥接结构"级别的改造。现状：Talker 的条件时钟由 MiniMind forced-text 驱动，文本 token 一个位置一拍，EOS 后进 audio tail。改造：query 时钟从回答一开始就按音频帧速率恒定推进，文本侧 K/V 仍按 9.2.1 的 stable frontier 因果可见——把"文本时钟 + audio tail"换成 Moshi 式"从头到尾一个音频时钟"。改 `capstone/bridge/resampler.py`（生成恒定速率 query 序列）与 `capstone/talker/minimind_talker_adapter.py` 的 step 推进逻辑；Talker 权重仍冻结。要重训 adapter：与主矩阵单臂同预算（10000 step、batch 64），建议先跑 1 seed 探索，方向对了再补满 3 seed。预期：增量模式下首个 audio step 不再等文本时钟铺满，音频可以更早启动；代价是 adapter 要独自学会"语速"。失败判据：audio stop 位置系统性漂移——生成长度分布相对 H-Mid 明显偏移（提前哑火或说个不停）；或 test CE 与 `ContentWER` 同时显著劣于 H-Mid。这个实验一半的信息量本来就在"哪里会坏"，负结果照第 4 节的规矩报告层面原因即可。

三条论文结论能在本课设置里验方向。

- Moshi"理论延迟可以从帧结构直接算出"：第 01 课复现过一次，这里在 19C replay 上再来一次。停播路径的结构时间下界是 Step 8 给出的 `100ms VAD window + 至多 40ms 补足分类音频 + 80ms pause deadline`，对照验收 C 实测的 speech onset 到扬声器停止 p95 ≤ 400ms。预期同方向：实测显著高于结构时间，差值应能在 trace 里逐段指认（特征计算、AEC、线程调度、声卡缓冲）；指认不出来，先回 15.5 查分段计时是否齐全，再谈调参数。
- Qwen2.5-Omni"Talker 同时消费隐表征与文本 token 更好"：改造实验 1 就是它的缩小版。预期方向一致——双条件不劣于单条件。复现不出来时，优先怀疑参数量没对齐或拼接实现有误，规模差三个数量级，可比的只有方向，没有幅度。
- Moshi"inner monologue：文本流当脚手架能稳住语音流"：本课的 text fallback 和 T-Embed 强基线是它的两个远亲。若 Pilot 假设"H-Mid 相对 T-Embed 降 CE"不成立，这个负结果本身就与"文本条件已经足够强"的方向一致——照第 4 节的规矩记录层和训练信号，不必救。

**更多顺手扩展。**

基础扩展：

- 用 Qwen3-Omni 或 MiniCPM-o 4.5 替换 Thinker；
- 用[第 03 课](03_audio_codec.md)的另一个 codec 替换 Mimi；
- 加入可切换说话人和情绪控制。

研究扩展：

- 对不同 Thinker 层做 causal intervention（干预某层激活，看输出的因果变化）；
- adapter 中增加 uncertainty-aware gating；
- 用 distillation 把 30B Thinker 压到 3B 实时模型；
- 对 `pause/resume/replan` 做 offline RL；
- 研究 hidden bridge 是否能传递环境音信息；
- 构造中文 full-duplex benchmark。

系统扩展：

- 推理服务拆成独立 GPU workers；
- session KV/state 的迁移和恢复；
- vLLM-Omni/TensorRT-LLM 服务化；
- 多 session 动态 batching；
- WebRTC jitter/AEC/packet loss 压测。

## 22. 论文精读

### 必读 1：[MiniMind-O](https://arxiv.org/abs/2605.03937)

重点：

- Thinker–Talker bridge；
- loss mask；
- Mimi 多码本；
- streaming 与 barge-in 实现。

带三个问题进去，答案都要能在自己的产物里指认：论文的 bridge 取哪一层、依据是什么——对照你在 Step 6 测出的层间差异，看 26M 模型的结论能否外推到 30B Thinker，缺哪项对照实验；"近双工"里哪些由模型承担、哪些由外部 VAD 规则承担——对照第 01 课 `RealtimeSession` 的结论和本课 Step 8 的事件总线；它的 loss mask 和多码本 delay 与 10.5 的九路协议逐项对得上吗——发现对不上的位置，先怀疑自己的复述，再查上游源码。

### 必读 2：[Nemotron 3 Nano Omni](https://arxiv.org/abs/2604.24954)

重点：

- modality encoders；
- hybrid Mamba/Attention MoE；
- video token reduction；
- SFT/MPO/RL 课程。

带着问题读：多模态融合发生在哪个层级——这决定第 3 节"hidden 里可能有图像信息"的假设是否站得住；模态 tower 的冻结范围和模型输出接口是什么——据此说明它能为 bridge 提供什么信息、不能提供什么，以及为什么仍需独立 Talker。答案要能对上第 3 节的能力边界清单。

### 必读 3：[Moshi](https://arxiv.org/abs/2410.00037)

重点：

- 双音频 stream；
- temporal/depth hierarchy；
- inner monologue；
- theoretical/practical latency。

带着问题读：inner monologue 与本课 text fallback 的输入、监督和推理用途各差在哪——一个是共时钟的建模组件，一个是故障兜底，读完要能写出这句话的证据；用户/助手独立音频流如何支持并发状态——对照 7.4 的状态表，标出 Moshi 用建模消掉了哪几行状态机；它怎么处理 turn segmentation——和本课 turn policy 的四个离散动作比，哪边的失败模式更可观测。

### 必读 4：[Qwen3-Omni](https://arxiv.org/abs/2509.17765)

重点：

- Thinker–Talker MoE；
- streaming codec generation；
- first-packet latency；
- thinking/non-thinking。

带着问题读：原生联合训练的 Talker 相比后接 Talker，到底共享了哪些数据、时钟和梯度路径——逐条列出，再把每条归到"hidden bridge 可近似"或"必须依赖联合训练"两栏；这张两栏表直接决定第 21 节差距分析里的答案，也决定改造实验 1 的预期上限。

### 必读 5：[Qwen3.5-Omni / ARIA](https://arxiv.org/abs/2604.15804)

这里只精读 ARIA 处理的一个问题：文本 tokenizer 和语音 tokenizer 速率不同，固定比例
推进容易漏读、重复或让韵律不稳定，因此 text–speech clock 需要根据当前生成状态动态
对齐。阅读后画出 `text token index → speech/codec step` 的动态映射，并说明它与本课
固定 Mimi diagonal delay、9.2.1 字节对齐分别解决什么问题——一个管码本间错位，一个管跨 tokenizer 边界，ARIA 管的是第三件事。Qwen3.5-Omni 的模型规模、
数据量和 paper-scale training recipe 不进入本课小卡主实验；本课只借用时钟问题和
trace 设计，不把阅读论文写成已经复现。

### 必读 6：Omni serving

- [vLLM-Omni](https://arxiv.org/abs/2602.02204)：把异构模型拆成可独立调度的 stage graph；
- [LiveServe](https://arxiv.org/abs/2606.22983)：让 scheduler 读取 playback progress、
  speech activity 和 barge-in，并为下一轮 KV 的复用安排 eviction、reload 与 prefetch。

阅读后提交一张到 19C trace 的字段映射表：stage graph 对应
`listener/thinker/bridge/talker/playback` 的 `stage_id` 与入出队时间；playback frontier
对应已生成 audio frame、已入队 frame 和最后实际 DAC frame；barge-in 对应
`INTERRUPT_CANDIDATE → FAST_PAUSE → DAC stop → resume/replan`；next-turn KV 对应
`cache_id`、保留/驱逐/预取/重建动作及其时间戳。缺少这四组字段时，只能说明服务跑通，
不能解释 19C 的停播浪费、queue underrun 或下一轮恢复延迟。

### 必读 7：双工评测

- [Full-Duplex-Bench v1.5](https://arxiv.org/abs/2507.23159)；
- [HumDial-FDBench](https://arxiv.org/abs/2604.21406)；
- [How Should LLMs Listen While Speaking?](https://arxiv.org/abs/2605.10199)。

精读时整理四类事件的构造规则：pause、backchannel、side speech 和 interruption——对照 `synthetic_duplex_v1` 的四类事件参数，标出你的合成规则缺了哪些真实分布。
同时记录 channel fusion 导致 context corruption 的条件（对照 10.2 对单声道混音的禁令），以及自动指标与真人体验不一致
的案例——这也解释了验收 C 为什么坚持用 DAC 时间戳而不用 transcript 时间。

## 23. 最终报告必须回答的问题

最终报告应逐项给出以下证据：

- 输入事实对应的感知层、token span 和 probe 结果；
- 19A 原生冻结文本桥丢失的信息，以及 T-Embed/H-Mid/H-Last 主矩阵；
- hidden bridge 使用多模态条件的 modality-shuffle 结果；
- Thinker 错误与 Talker 内容渲染错误的独立标注；
- interrupt 后保留、恢复和重建的状态清单；
- 8 张卡的进程、模型、cache 与显存分配；
- 每项质量变化对应的延迟、显存和 GPU-hours。

报告交齐，毕业设计就算完成：你手里有一个 30B 大脑驱动、26M 嘴发声、能被打断、每个数字都有等级标注和 trace 支撑的双工系统，还有一套分得清"桥坏了还是调度坏了"的三级回归。语音双工主线到此收束。下一课是选修加餐：[第 20 课](20_unified_understanding_generation.md)离开语音，研究另一个统一问题——让同一组核心 Transformer 参数既做视觉理解又做图像生成。造完这课再去看它，你会发现问题的形状似曾相识：又是两种表征、又是一个骨干怎么伺候两种目标。桥接的手艺，换个模态还能再用一次。
