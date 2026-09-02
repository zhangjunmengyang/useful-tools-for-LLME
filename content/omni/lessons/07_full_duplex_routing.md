---
id: 07_full_duplex_routing
title: "真双工 Routing"
summary: "助手正说着话，用户新说的话能持续进入语义状态，让系统在 Continue（接着说）、Pause（先停停）、Replan（换个说法）里做选择吗？还是只会一刀切地取消？"
unit: realtime
play_tools: []
checkpoints:
  - "搭一条双流 wall-clock（真实挂钟时间线），把 captured、available、consumed、emitted、played 五个时刻都记上账。"
  - "实现两种路由：通道直接融合，和带门控的交叉注意力记忆。"
  - "设计可抢占的 microstep scheduler、带版本号的状态、能撤回的播放 buffer，以及过期输出的处理。"
  - "在“嗯嗯”附和、旁人说话、明确打断、不该打断这四类场景上逐 case 验收。"
---

# 第 07 课：实现全双工路由与调度

> 内容：双流建模、异步运行时与语义打断
> 建议周期：3–5 周  
> 推荐硬件：4–8 卡训练；单会话低延迟推理优先单卡或细粒度调度  
> 最终产物：两种 user-stream routing、双工运行时、Full-Duplex-Bench 报告

## 1. 说话时也要继续听

第 05 课实现了 320ms 分块的流式输入，第 06 课加入了 turn policy。当前运行时仍在助手生成期间停止语义输入；检测到用户声音后，只能取消生成、丢弃未播放音频，并把后续内容当作新请求处理。它不能在原计划上应用用户的修正。

本课比较两种语义接入方式：channel fusion 将用户向量追加到主因果序列，cross-attention memory 将用户信息保存在旁路内存中。三分类控制头输出 `CONTINUE`、`PAUSE` 或 `REPLAN`。运行时改为 microstep 调度，每个解码步都处理新输入和控制事件。

第 18 至 19 课会复用这里定义的运行时和路由边界。缺少 microstep 调度时，流式 listener 和 turn policy 最终仍只能触发 hard cancel。

验收场景要求系统在长回答中接收"不是上海，改成杭州"：停止延迟低于预注册阈值，重新生成的内容使用杭州，并且不残留旧计划；附和输入不能触发停止。每个 case 保存毫秒级事件时间轴和决策来源。

本课术语：

| 术语 | 简要解释 |
|---|---|
| 全双工（full duplex） | 打电话模式：两边同时说、同时听；"你说完我再说"的对讲机是半双工 |
| hard cancel | 检测到人声就掐掉生成、丢掉没播的音频、整段重录；当前系统唯一会的"打断" |
| backchannel | 听话人的"嗯、对、哈哈"：表示在听，没想抢话 |
| channel fusion | 用户流向量直接追加进助手正在用的同一条因果序列 |
| cross-attention memory | 用户流存进只追加的外部内存，主模型用交叉注意力按需读取 |
| CONTINUE / PAUSE / REPLAN | 控制头的三个动作：继续说 / 停嘴但保留状态 / 废旧计划重新想 |
| context corruption | 用户已纠正，回答却新旧混说（"……的飞机火车票"） |
| available_at | 一个事件处理完毕、模型最早能用上它的墙钟时刻 |
| branch | 一次回答尝试的完整状态存档（注意力缓存、播放边界等），可暂停、恢复、作废 |
| AEC | 回声消除：把扬声器里自己的声音从麦克风信号里减掉，防止模型自己打断自己 |

## 2. 当前限制与改造范围

本课解决一个具体限制：MiniMind-O 在助手生成和播放期间，不能继续把用户语音送入模型的语义状态。当前流程是先等用户整段说完，再开始生成和播放；播放期间一旦 VAD 检测到人声，代码就中止当前 generator（Python 生成器：当前实现里逐 token 产出回答的那段惰性循环），系统随后重新收集一整段语音，发起下一次请求。

这套流程只会"听到人声就停"。它判断不了用户是在附和、纠正问题，还是与旁人交谈；更不能利用新语义修改正在生成的回答。第 06 课的 turn policy 解决了"分清这是什么事件"；事件分清之后，新内容怎么进入模型、旧回答的状态怎么处理，才是本课的问题。

实验围绕以下问题组织：

> assistant text/audio 生成期间，新到达的 user speech 必须持续被编码并影响
> 模型决策。实验比较 channel fusion 与 cross-attention memory 在语义
> grounding、context corruption 和延迟上的差异。

grounding 指回答真正建立在对方刚说内容上的程度——嘴上说"好的改杭州"、答案里还在推荐上海，就是 grounding 失败。

本课把同时收音和播放称为"双工"。只有同时满足以下条件，实验才算实现了语义层面的全双工：

1. 输入流在输出生成期间持续推进；
2. 输入和输出不被一个长生命周期全局锁串行化；
3. 模型区分 backchannel、旁人语音和需要响应的 interruption；
4. `REPLAN` 使用新语义，而非单纯 cancel；
5. `PAUSE/RESUME` 保留可复用状态，而非清空重来；
6. 所有行为绑定到真实 wall-clock 和 `available_at`。

这六条每一条都在堵一种"假双工"：只收包不消费是假的，能停不能懂是假的，replan 后从零重来也是假的。

## 3. 独立进入条件

本课不强制依赖第 01、05 或 06 课。开始前从以下两个 checkpoint 中选择一个，并在 manifest 中固定选择结果：

- [第 01 课](01_baseline_reproduction.md)的 mini `baseline-v1`；
- 官方 `minimind-3o`，但必须锁定 exact weight revision、
  upstream commit、tokenizer/encoder revision 和文件 SHA-256。

若选择官方 checkpoint，必须在训练前冻结本课自己的
`duplex-baseline-v1`：固定 hard-cancel runtime、至少 60 个
non-overlap/interrupt/backchannel/side-speech cases、base 输出与完整
时间轴。后续三臂都从这个完全相同的起点分叉——起点不同，后面所有比较都白做。

最小独立组件：

- baseline Thinker/Talker/Mimi；
- 一个 frozen 320ms chunk listener：
  可用简单 causal audio adapter、prefix-only 重算，或严格 closed-block
  SenseVoice（这两种因果约定第 05、06 课讲过）；不得读取尚未到达的 waveform；
- 一个本课训练的三分类控制头：
  `CONTINUE | PAUSE | REPLAN`；
- 真实或合成双声道时间轴；
- 干净用户声道，或带 far-end reference（扬声器实际播放信号的参考副本）的 AEC。

若已有：

- `listener-v1`（第 05 课产物）：替换最小 listener；
- `turn-policy-v1`（第 06 课产物）：作为额外 feature；

其中 `listener-v1` 和 `turn-policy-v1` 只替换 B、C 的对应组件。B、C 必须使用
同一 listener 和同一控制头。A 保留当前 VAD 命中后 hard cancel 的路径，只作
工程参考，不参加 B、C 的 routing 机制因果比较。

最小 listener 必须满足：

- `prefix-only`：时刻 $t$ 的输入严格为 `waveform[:t]`；或
- `closed-block`：只在完整 320ms block 到达后编码该 block，
  `available_at = block_end + encode_latency`；
- centered/sliding window 不得包含未来 samples；
- 每个输出 feature 保存 source span 与 `available_at`；
- 改写未来 suffix 不得改变 shared prefix 已发布 feature；
- offline full-utterance SenseVoice hidden 只能做 teacher/upper-bound，
  不得用于 B/C 主结果。

输出：

```text
duplex-v1-hard-cancel
duplex-v1-channel-fusion
duplex-v1-cross-memory
```

## 4. 完成本课需要掌握的操作

1. 严格区分 hard cancel、semantic interruption、full duplex；
2. 给离散 LM 加 wall-clock/channel 表示；
3. 实现 channel fusion；
4. 实现 gated cross-attention user memory；
5. 训练 `CONTINUE/PAUSE/REPLAN`；
6. 建立 state branch、resume 与 stale-output 处理；
7. 把长生命周期 `MODEL_LOCK` 改成可抢占 microstep scheduler；
8. 处理 AEC、jitter（网络抖动：包到达时刻不稳）、播放 buffer 与 stop latency；
9. 使用 FDB 场景和逐 case 时间轴验收。

## 5. 原理:边造边讲

双工的难点不在多线程，在语义和状态：新话怎么进脑子、旧话的状态怎么处置、时间怎么对齐。六个机制，每个按同一节奏走：为什么需要、怎么运转、在哪实现、怎么验证。

### 5.1 双流时间轴

文本 LM 里"位置"是 token index，谁先谁后看下标就行。真实对话里位置是墙钟时间：用户那句"改成杭州"出现在助手说到第 1.8 秒的时刻，系统晚 300ms 才拿到，它的每个下游动作就慢 300ms。想让模型的行为和评测的数字都对得上真实世界，每个事件必须带时间戳，而且要分清"发生""可用""被用"是三个不同时刻。

对每个输入/输出事件至少记录：

- `captured_at`：音频或事件进入系统的时间；
- `available_at`：经过必要处理后，模型最早可以使用它的时间；
- `consumed_at`：模型实际读取它的时间；
- `emitted_at`：模型产生输出事件的时间；
- `played_at`：音频真正开始播放的时间；
- `channel`：事件来自用户还是助手。

`available_at` 表示事件最早能被模型使用的时刻——320ms 的块没收完、编码没算完，内容再重要模型也用不上。

时间戳统一挂在第 7 节的 `StreamEvent` 上，由 Step 1 的确定性 replay 生成。评测时检查
`consumed_at >= available_at`，并分别统计采集、编码、排队、生成和播放延迟。
token index 只表示序列顺序，不能替代这些时间戳；用 index 冒充时间，是本课最常见的自欺方式。

### 5.2 Channel fusion

最直接的接法：用户的话当成新 token，追加到模型正在读写的那条序列里。好比开会时有人递纸条，直接摊在桌面上，后面每个发言的人都看得见。

Channel fusion 把用户流的 token 或 embedding 追加到助手正在使用的同一条因果序列中。例如，同一序列可以依次包含 `assistant_out@t0`、`user_in@t1` 和
`assistant_out@t1`。这种方法让后续 self-attention 直接读取用户语义，通常有利于 grounding。

**风险。** 需要同时检查两个风险：

- 若系统未及时停止，user 与 assistant 语义混入同一上下文；
- 容易 context corruption；
- 插入事件不能修改已缓存的过去位置。

类比失效处：纸条一旦摊上桌就收不回。因果序列只能追加，过时的用户内容和被废弃的旧回答会永远留在上下文里，这正是 5.5 节 context corruption 的温床。

打包规则在 8.1 节写死，实现在 Step 5。

### 5.3 Cross-attention memory

另一种接法：用户的话不上桌，记在旁边的记事本上，主模型每隔几层抬头翻一眼，翻多少由一个可学习的 gate（0 到 1 之间的标量闸门）决定。主序列保持干净，代价是"翻记事本"可能翻得不够勤。

用户流保存在只追加的外部 memory 中，主序列不动：

$$
h'_l=h_l+
g_l\cdot\operatorname{CrossAttn}
(h_l,M_{user},M_{user})
$$

主 self-attention 的 KV cache（注意力历史缓存：已算过的 key/value 存下来，新 token 不用重算整个历史）不直接写入用户 token，因此更容易实现暂停、恢复和按时间开放 memory。

adapter 结构在 8.2/8.3 节写死，实现在 Step 6。

实验需要检查 memory 长度、读取延迟和 gate 值。若 gate 长期接近零，模型实际上没有使用用户输入——记事本记了但从来没翻过。语义融合仍可能弱于 channel fusion；这项差异必须通过置零 user memory 的消融和逐 case 结果确认，不能靠印象。

### 5.4 控制与内容

把话听进去只解决了"输入"，还得决定"听见了怎么办"。全停是过敏（用户一声"嗯"就闭嘴），全不停是装聋，正确行为是三选一。

三分类控制：

- `CONTINUE`：新输入是 backchannel/环境音/无关内容；
- `PAUSE`：立即停止可听输出，等待输入完整或用户明确停止；
- `REPLAN`：新输入改变目标，建立新的 response branch。

执行 `PAUSE` 时保留当前 branch 的 KV、user memory cursor、播放位置和尚未送出的
pending PCM（PCM：可直接播放的原始音频，第 01 课讲过）。播放队列停止出队，但不删除这些 frame；恢复时从同一播放边界继续。
只有执行 `REPLAN` 时才丢弃 pending PCM，从安全 checkpoint 建立新 branch，并把旧
branch 标记为 `superseded`。不能直接在旧答案末尾续写——旧答案的后半截已经作废，续写等于把新语义嫁接在死树上。

控制头训练在 Step 7，状态机在 Step 9 的表格里逐格验收。

### 5.5 Context corruption

最阴险的失败不是不停嘴，是停了嘴、换了话题，答案里却新旧混杂。旧计划已被用户纠正，模型继续混合旧/新答案：

| 说话方 | 内容 |
|---|---|
| 助手 | "火车是下午三点……" |
| 用户 | "不是火车，改成飞机" |
| 助手 | "……的飞机火车票建议……" |

该输出同时包含已经失效的"火车"和新的"飞机"约束，应计为
context corruption。即使停止延迟达标，也必须单独统计这类语义错误——停得快和说得对是两个指标。

量化定义（CCR）在 14.4 节，逐 case 复核在第 17 节强制执行。

### 5.6 回声

助手扬声器输出会进入麦克风。无 AEC 时，listener 可能把自己的回答判为用户输入——模型一直被自己打断，双工变成自言自语的死循环。

评测优先级：

1. 干净独立 user channel；
2. 耳机/直连；
3. 有 far-end reference 的 AEC；
4. 禁止用"完全静音 assistant playback"掩盖问题。

第 4 条单独强调：把自己的声音静音，回声问题看起来消失了，但你测的已经不是双工系统——它一半的输出不存在了。

AEC 条件对比在 Step 11，相关失败在第 16 节诊断表有独立条目。

## 6. MiniMind-O 当前代码落点

按"一次用户插话"的事件路径读这张表：包进来（`recv_loop`）、被察觉（`poll_interrupt`）、影响生成（`stream_generate`）、影响播放（`stream_pcm`），逐格确认当前瓶颈和本课改法：

| 文件/符号 | 当前瓶颈 | 本课改造 |
|---|---|---|
| `MiniMindOmni.forward` | 仅 prefill 初始 audio | user memory/stream event 输入 |
| `stream_generate` | 同步 Python generator | 单步 `decode_step(state)` |
| `past_key_values` | 一条 Thinker+Talker state | 版本化 branch/state snapshot |
| `RealtimeSession` | `generating` + bool interrupt | duplex control state machine |
| `webui/web_demo.py::run_generate` | 整个 generator 被 `MODEL_LOCK` 包住 | microstep 调度/细粒度互斥 |
| `recv_loop` + queue | 能收包，但 poll 时才处理 | ingress 独立持续消费 |
| `poll_interrupt` | VAD 一命中就 break | listener→router→control |
| `stream_pcm` | 已编码 PCM queue | 可撤销 buffer 与 stop ack |
| `state['history']` | 只存完成的整轮文本 | 当前 plan、branch、user memory |

常见错误是并发：不能直接删除 `MODEL_LOCK` 后让多个 Python 线程并发写同一模型状态——那是把确定性 bug 换成随机 bug。
本课使用单 GPU microstep scheduler：每次只执行一个 decode step，随后处理已经
到达的输入和控制事件。这样既允许输入持续推进，也能保证状态更新顺序确定。

## 7. 双工运行时与状态结构

运行时拆成五个阶段，每个阶段只对自己的输入负责：

| 阶段 | 输入 | 输出与责任 |
|---|---|---|
| `AudioIngress` | 按 wall-clock 到达的麦克风 packet | 按顺序交给 listener，不等待当前回答结束 |
| stateful listener | 已完整到达的 320ms 音频块 | 增量 user feature，同时追加到 `UserMemory` |
| `Turn/ControlHead` | listener feature、双方 activity 和当前生成状态 | `CONTINUE`、`PAUSE` 或 `REPLAN` |
| `DuplexRouter` | 控制动作、`UserMemory` 和当前 branch | 保留、暂停或替换 Thinker/Talker decode state |
| 输出路径 | 当前有效 branch 的 text token 和 codec frame | 文本事件；或经 Mimi 解码后写入可撤销 PCM buffer |

`DuplexRouter` 是模型状态的唯一写入者。每个 decode microstep 结束后，它先处理已经
到达的输入和控制事件，再决定是否继续同一 branch。"唯一写入者"是刻意设计：状态只有一个主人，才谈得上确定性回放和逐 case 审计。

### 7.1 统一事件

所有跨阶段通信走同一种事件结构，时间戳字段对应 5.1 节：

```python
@dataclass
class StreamEvent:
    session_id: str
    channel: Literal["user", "assistant"]
    kind: Literal["audio_feat", "text", "codec", "control"]
    payload: Tensor | dict
    captured_at_ms: int
    available_at_ms: int
    sequence_no: int
```

### 7.2 版本化状态

一次回答尝试的全部状态打包成一个 branch：

```python
@dataclass
class ResponseBranch:
    branch_id: int
    parent_id: int | None
    thinker_kv: object
    talker_kv: object
    user_memory_cursor: int
    emitted_text: list[int]
    emitted_codec: list[list[int]]
    played_until_ms: int
    status: Literal["active", "paused", "superseded", "done"]
```

`REPLAN` 将旧 branch 标为 superseded，
从安全 checkpoint + 新 user memory 建新 branch。branch 带 `parent_id`，血缘可查——出了 stale plan 的错，能追到它从哪个 checkpoint 分叉。

## 8. 两种 routing

### 8.1 Channel fusion

每 320ms 产生若干 user embeddings，
加：

- `channel=user` embedding；
- wall-clock bucket/RoPE（RoPE：旋转位置编码，用旋转角度表示位置，第 08 课会把它扩展到二维）；
- `available_at` mask。

把它们作为后续 self-attention event 消费。
不能把 token 插入已经 cache 的历史中间——因果序列只许追加。

这里有个 MiniMind-O 特有的坑。它的输入并非单独一条 text 序列，而是
`input_ids: [B, 9, S]`：前 8 路是 Talker audio history，最后 1 路是
Thinker text/event history。因此一个 user event 也必须写成完整的九路记录，不能只向
Thinker KV 塞一个 embedding，然后让 Talker cache 和位置计数留在旧长度——九路一旦长短不齐，后面每一步都在错位。

本课固定下面的记录规则。`global_pos` 是九路共同的序列位置；
`audio_step` 只统计真正执行 assistant codec 采样的次数：

| 字段 | assistant decode 位置 | user audio event 位置 |
|---|---|---|
| text lane | 当前 assistant history token | `USER_AUDIO_EVENT` marker，其 embedding 替换为 `stream_projection(user_feature)` |
| audio lanes 0–7 | 当前 schedule 所需的 8 路 audio history | 8 路全部写 `audio_pad_token` |
| text label | 正常 next-token target，按原 mask | `-100` |
| audio labels | schedule 中有效位置参与 loss | 8 路全部 `-100` |
| Thinker KV | 追加 1 个位置 | 追加 1 个位置 |
| Talker KV | 追加 1 个位置 | 追加 1 个位置；读取 pad codec embedding 和该位置的 bridge state |
| `global_pos` | 加 1 | 加 1 |
| `audio_step` | 完成一次 codec 采样后加 1 | 不变 |
| Mimi / playback | 完整 frame ready 后可 decode | 不接收该位置的任何 audio logit |

这里让 Talker KV 也追加 user-event 位置，是为了保持九路 cache 长度一致，并让后续
Talker hidden 可以读取已经融合新用户信息的 bridge state。该位置虽然会计算 audio
logits，但运行时必须丢弃，不能让 special ID 触发 stop，也不能送进 Mimi。

audio schedule 必须按 `audio_step` 索引，不能按 `global_pos` 索引。否则每插入一个
user event，diagonal delay（第 01 课讲过的码本对角错位）就会跳过一个位置，后续 frame 会错位。一个 320ms chunk
产生 $K$ 个 user embedding 时，连续追加 $K$ 个九路 event 位置；`global_pos` 增加
$K$，`audio_step` 仍不变。

先用下面的三步 trace 验证计数。为便于阅读，假设进入 trace 前
`global_pos=100`、`audio_step=24`：

| 执行次序 | 事件 | 写入的九路记录 | 采样动作 | 执行后 `(global_pos, audio_step)` |
|---:|---|---|---|---|
| 1 | assistant decode | 8 路 schedule history + 1 路 assistant token | 采样 codec slot 24 | `(101, 25)` |
| 2 | user audio event | 8 路 audio pad + 1 路 user marker/embedding | 不采样；丢弃 audio logits | `(102, 25)` |
| 3 | assistant decode | 8 路 schedule history + 1 路 assistant token | 采样 codec slot 25 | `(103, 26)` |

这个 trace 需要配套下面这组单测：

```python
assert packed.shape == (1, 9, 3)
assert (packed[:, :8, 1] == audio_pad_token).all()
assert (audio_labels[:, :, 1] == -100).all()
assert text_labels[:, 1].item() == -100
assert sampled_audio_steps == [24, 25]  # user event 没有占用 slot
```

最后把第 2 步删除，再检查两条路径产生的 schedule slot 都是 `[24, 25]`。模型采样值可以
因为 user semantic condition 而改变，但 frame index、stop 位置和重组公式不能因为插入
事件而错一位。

### 8.2 Cross-attention memory

`UserMemory` append-only：

```python
memory.append(
    embeddings,
    timestamps,
    channel_mask,
    semantic_boundary
)
```

在 Thinker 选定层加入 gated cross-attention adapter。
Talker 默认只通过更新后的 Thinker/bridge 获得用户影响，
避免直接让 acoustic user feature 污染声学生成——用户的语义应该改变"说什么"，用户嗓音的声学特征不应该渗进"怎么发音"。

### 8.3 公平约束

B、C 的 listener 输出先经过同一个 `stream_projection`，得到相同数量、相同维度
的 user embeddings。两臂只在这些 embeddings 如何进入 Thinker 上不同——控制变量控制到只剩这一个差异，结论才归得了因。

为避免"C 多了一组 cross-attention 参数"这种混杂，本课把两种 adapter 写死：

- B 在 Thinker 第 2、4、6 层各放一个无 bias 的 bottleneck adapter：
  `D → 128 → D`，使用 SiLU 和一个标量残差 gate；
- C 在同三层各放一个 rank-64 的低秩 cross-attention：
  query、key、value 都做 `D → 64`，输出做 `64 → D`，同样使用一个标量
  残差 gate；
- 两者每层的矩阵参数都为 `4 × D × 64`，gate 数也相同。共享的
  `stream_projection`、listener 和控制头不计入 routing 差异。

preflight 必须实例化 B、C 后按 `requires_grad` 逐项导出参数表。两臂总可训练
参数差超过 5% 就停止实验，不能训练后再补参数。其余约束为：

- B、C 的 user token 数相同；
- 主 backbone 都冻结，或使用完全相同的 LoRA（低秩适配：冻住原权重，只训练一对小的低秩增量矩阵）层与 rank；
- B、C 使用同一控制头、数据、时间轴、optimizer updates 和 playback runtime；
- A 只共享 base checkpoint、输入事件 replay、输出 sampler、playback 与 AEC
  条件。A 不使用语义 listener/control 的训练预算，参数表中明确记为
  `routing_trainable_params=0`。

## 9. 数据 recipe

双工数据和普通对话数据差一个维度：时间。光有"说了什么"不够，必须有"什么时刻说的、和对方的哪句话重叠"。

### 9.1 场景构成

至少五类：

1. `NO_OVERLAP`：正常轮次；
2. `LISTENER_BACKCHANNEL`：用户"嗯/对"但不要求停止；
3. `TRUE_INTERRUPTION`：用户提出新问题；
4. `CORRECTION`：用户修改当前请求；
5. `SIDE_OR_AMBIENT`：旁人/背景语音，不应 replan。

另加：

- explicit stop；
- user self-repair；
- long pause；
- simultaneous start；
- packet jitter/AEC residual。

### 9.2 来源

Pilot：

- MiniMind-O T2A/A2A 文本/语音；
- 合成 assistant response timeline；
- 在已知时间插入 scripted user speech；
- 每个 base case 生成反事实五元组（同一段基础对话分别配上 9.1 五类场景的用户事件，其余全同，方便把行为差异归因到场景本身）。

Standard：

- 许可清晰/取得同意的真实双声道助手对话；
- AMI 等多方对话只作 overlap/side-speech 补充；
- 至少人工标 20–100 小时。

Scale track：

- [Moshi-finetune](https://github.com/kyutai-labs/moshi-finetune)
  要求的 stereo dialogue schema；
- 训练一次 Moshi LoRA，作为完整双流架构参考；
- 不把 Moshi 数据许可默认等同于代码许可。

Test：

- Full-Duplex-Bench v1/v1.5（FDB：专测双工行为——打断、附和、抢话——的公开基准）；
- 自建 bilingual corrections/side-speech；
- 真实 live echo 条件。

### 9.3 Schema

```yaml
session_id: string
scenario: CORRECTION
user_channel_path: user.wav
assistant_channel_path: assistant.wav
far_end_reference_path: assistant_played.wav
post_aec_user_path: user_aec.wav
sample_rate: 16000
source: string
license: string
consent_id: string | null
timeline:
  - {t_ms: 0, channel: assistant, event: speech_start}
  - {t_ms: 1800, channel: user, event: speech_start}
  - {t_ms: 1960, label: PAUSE}
  - {t_ms: 2600, label: REPLAN}
user_utterance:
  transcript: "不是上海，改成杭州"
  semantic_update:
    replace: {destination: 杭州}
assistant_plan_before: {destination: 上海}
assistant_plan_after: {destination: 杭州}
acceptable_behavior:
  stop_by_ms: 2200
  answer_contains: [杭州]
  answer_excludes: [继续推荐上海]
```

训练缓存附加：

```yaml
user_features:
  path: ...
  available_at_ms: [...]
assistant_text_tokens:
  ids: [...]
  emitted_at_ms: [...]
assistant_codec_codes:
  path: ...
  emitted_at_ms: [...]
```

### 9.4 预处理

- 保持 user/assistant/far-end 三条逻辑声道；
- 对齐时钟，记录 sync error；
- AEC 前后都保留，评测分别报告；
- 音频以真实 packet arrival replay；
- user feature 不能早于 `available_at`；
- 旧 assistant plan/new plan 显式结构化；
- 同 base dialogue 的反事实必须同 split（否则模型在训练集见过"同一段话的另一个结局"，评测就漏了）；
- 生成的 assistant audio 固定，避免三臂输入不同。

### 9.5 三档规模

| 档位 | 规模 | 目标 |
|---|---:|---|
| pilot | 10k 合成场景 + 2h 真实 | routing/运行时 |
| standard | 100k–500k 场景 + 20–100h 真实 | 主结果 |
| full | 1k h+ stereo/synthetic mixture | Moshi/规模扩展 |

数据平衡按完整控制事件计算，不按音频 frame 计算。报告中必须分别列出每类
train/dev/test 的事件数，避免长语音因为 frame 多而被重复加权。

### 9.6 License 与缺失边界

- 合成数据记录源文本、TTS、speaker 条款；
- 真人双声道需 consent/PII 处理；
- AMI/Fisher 等各自按实际许可；
- FDB 用于评测，不默认加入训练；
- 只有 mixed mono 时无法可靠区分双方；
- 无 far-end reference 时不能把 live echo failure 全怪模型；
- 未开放的商业模型结果只能作外部参考，不能声称 recipe 复现。

## 10. 逐步实验

顺序执行，每一步的产物是下一步的前提：replay 不确定，后面所有延迟数字都是噪声；`decode_step` 没拆出来，调度器无处安身；hard-cancel 基线没量过，"改进"无从谈起。

### Step 1：建立 deterministic event replay

本步骤先排除麦克风、网络和操作时机带来的随机性。读取固定时间轴文件并发送
packet；相同 seed 下必须复现：

- packet arrival；
- listener feature；
- control event；
- output token；
- playback buffer。

在进入 routing 训练前先执行 listener conformance：

样例 A 和样例 B 使用相同的输入前缀，之后分别接上 `suffix_A` 和 `suffix_B`。

在 suffix 的 packet 尚未到达前，A/B 已发布的 listener feature、
control logits 与 user-memory prefix 必须在预注册公差内一致。
所有 feature 还必须满足
`consumed_at >= available_at >= source_end`。

### Step 2：把生成拆成 `prefill` 与 `decode_step`

替换长生命周期：

```python
yield from model.generate(...)
```

为：

```python
state = model.prefill(...)
while state.active:
    state, text_token, codec_frame = model.decode_step(state)
```

每个 `decode_step` 结束后，调度器都能处理 ingress 和 control。验证时在生成期间
注入固定 packet，并确认 listener 的 `consumed_at` 继续增加。

### Step 3：实现 ingress 与 deadline queue

AudioIngress 持续接收，不等 generation poll。
队列按 `(available_at, sequence_no)` 排序，
检测：

- drop；
- reorder；
- queue backlog（积压：进队速度长期高于出队速度）；
- deadline miss。

### Step 4：实现 hard-cancel baseline

严格复刻当前系统的行为：

- VAD/策略命中；
- 停 generator；
- 丢弃未播放 PCM；
- 整段新输入；
- 新请求重启。

必须量出它的真实 stop/restart latency——这组数字是全课的对照底线。

### Step 5：实现 channel fusion

- 定义 channel/time embedding；
- 只 append 新 user token；
- attention mask 保证 available-at；
- 按 8.1 将每个 user embedding 打包为完整 `[8 audio + 1 text]` 记录；
- `global_pos` 随所有事件推进，`audio_step` 只随 assistant codec 采样推进；
- user event 的 text/audio label 全为 `-100`，其 audio logits 不进 stop 与 Mimi；
- 先通过三步 trace 和九路 shape/mask/schedule 单测；
- 在 Thinker 第 2、4、6 层加入 `D → 128 → D` 的 gated bottleneck adapter；
- 训练该 adapter 和共享 control head；
- 记录主序列 user/assistant token 来源。

### Step 6：实现 cross-attention memory

- 在 Thinker 第 2、4、6 层加入 rank-64 的低秩 cross-attention adapter；
- gate 零初始化，保证初始等价（gate 为零时模型行为与 base 完全相同，训练从"没接记事本"平滑起步）；
- memory append-only；
- memory mask 按 time；
- 控制 memory window/summary；
- 记录 gate 与 attention，不把它们当 ground truth。

### Step 7：训练 control head

控制标签阶段：

1. backchannel/side/noise→`CONTINUE`；
2. stop/信息未完整→`PAUSE`；
3. correction/new question 完整→`REPLAN`。

使用 event-balanced sampling；
控制 loss 与内容 loss 分项记录。

### Step 8：训练 semantic replan

冻结主 backbone 或同 rank LoRA。
对 correction/new question：

- 旧 plan；
- 新 user memory；
- 新 plan/answer target；
- 明确 stale facts forbidden。

对 backchannel：

- 输出应与无 backchannel counterfactual 语义一致（用户一声"嗯"不应改变答案内容）。

### Step 9：实现 branch 与 resume

状态机：

| 当前状态 | 动作 | 下一状态与状态处理 |
|---|---|---|
| `GENERATING` | `PAUSE` | 进入 `PAUSED`，保存当前 branch，并冻结 pending PCM 队列 |
| `PAUSED` | `CONTINUE` | 恢复同一 branch 和 pending PCM，进入 `RESUMED` |
| `PAUSED` | `REPLAN` | 取消 pending PCM，将旧 branch 标记为 `superseded`，创建新 branch |

保存：

- 安全 KV checkpoint；
- user memory cursor；
- 已播放/未播放音频边界；
- branch lineage。

### Step 10：实现可撤销播放 buffer

区分四种状态，它们在时间上依次发生、在故障时各自成为证据：

- codec 已生成；
- PCM 已 decode；
- PCM 已发送；
- PCM 已实际播放。

stop latency 从 user speech/控制 anchor 测到扬声器静音，
网络/客户端 buffer 也计入——用户听到的安静才是安静。

### Step 11：AEC 条件

至少比较：

1. clean user channel；
2. mic+far-end without AEC；
3. post-AEC。

记录 echo return loss enhancement（回声抑制量：AEC 把回声压低了多少；若实现支持）、
false control event 和任务性能。

### Step 12：冻结 A 参考，再公平比较 B 与 C

三臂共同固定：

- base checkpoint；
- 同一份输入 packet 与事件时间轴；
- output sampler；
- playback chunk；
- AEC condition。

A 直接运行 Step 4 冻结的 VAD hard-cancel，不训练语义 listener、控制头或
routing adapter。B、C 另外共同固定：

- listener 与 `stream_projection`；
- control head capacity；
- train events；
- routing 参数预算；
- optimizer updates。

主结论只比较 B 与 C。A 用于回答"比现有 hard cancel 好多少"，不能用于声称
某一种 routing 机制在等参数条件下优于另一种。

### Step 13：Full-Duplex-Bench

运行 pause、backchannel、turn-taking、interruption，
以及 v1.5 的：

- user interruption；
- listener backchannel；
- side conversation；
- ambient speech。

保存每个原始 case 的事件日志和音频。

### Step 14：live soak test

soak test 是长时间连续运行测试（第 01 课做过 5 分钟版，这次加码）。至少：

- 30 分钟连续会话；
- 100 次 pause/resume/replan；
- 网络 jitter；
- 多 session 并发（若声称支持）；
- state/memory/queue 无泄漏。

## 11. 三个对照实验组

| 臂 | User stream | 状态处理 | 预期 |
|---|---|---|---|
| A | VAD 后整段重启 | hard cancel/清空 | 冻结的工程参考，语义延迟高 |
| B | channel fusion | user event 进入 self-attn | grounding 强、易污染 |
| C | gated cross-attn memory | 主 KV + 外部 append memory | context 稳、融合可能弱 |

A 是当前工程基线，用于量化只做 hard cancel 的效果。它没有等参数训练，不能进机制比较。
B 与 C 使用相同 listener、控制头、数据和训练预算；两者的差异才用于比较
routing 机制。

## 12. 配置与伪代码

```yaml
experiment: lesson07_full_duplex
base_checkpoint: hf://jingyaogong/minimind-3o
base_revision_or_sha256: ee3febbd08cc5b2bd41c039c825a8934232fee33
clock:
  audio_packet_ms: 20
  listener_chunk_ms: 320
  listener_lookahead_ms: 0
listener:
  applies_to: [B, C]
  mode: causal_or_closed_block
  available_at: source_end_plus_lookahead_plus_compute
  future_suffix_invariance_test: true
comparison:
  adapter_layers: [2, 4, 6]
  gate_init: 0.0
  parameter_tolerance: 0.05
  write_parameter_ledger: true
arms:
  A:
    routing: vad_hard_cancel
    routing_trainable_params: 0
  B:
    routing: channel_fusion
    adapter:
      type: bottleneck_mlp
      hidden_size: 128
      bias: false
  C:
    routing: cross_attention_memory
    adapter:
      type: low_rank_cross_attention
      rank: 64
      bias: false
    user_memory_max_ms: 30000
control:
  applies_to: [B, C]
  actions: [CONTINUE, PAUSE, REPLAN]
  frame_ms: 80
runtime:
  decode_microsteps_before_poll: 1
  pcm_buffer_ms: 160
  freeze_pending_pcm_on_pause: true
  cancel_pending_pcm_on_replan: true
  branch_checkpoints_every_tokens: 8
aec:
  mode: clean_or_far_end_reference
```

Scheduler 的骨架就是"解码一步、抬头一次"：

```python
while session.alive:
    for event in ingress.pop_ready(clock.now_ms()):
        user_memory.append(listener.consume(event))

    action = controller(user_memory, response_state)
    if action is CONTINUE and response_state.is_paused():
        playback.resume_pending()
        response_state.resume_same_branch()
    elif action is PAUSE and not response_state.is_paused():
        playback.pause_dequeue(preserve_pending=True)
        response_state.pause()
    elif action is REPLAN:
        playback.cancel_pending()
        response_state = router.replan(
            safe_checkpoint=response_state.last_safe_checkpoint,
            user_memory=user_memory,
        )

    if response_state.can_decode():
        response_state, out = model.decode_step(response_state)
        playback.enqueue(out.codec_frame)
```

`resume_same_branch()` 只能恢复暂停时保存的 KV checkpoint、user-memory cursor、
播放边界和 pending PCM，不能新建 branch。单测要逐 frame 比较暂停前队列、恢复后队列
和连续播放 reference，确认没有跳帧或重复帧。`REPLAN` 才会取消 pending PCM、废弃旧
branch，并从安全 checkpoint 建立新 branch。

## 13. 训练预算与 8 卡分配

### MiniMind mechanism track

- GPU0–2：A/B/C seed 42；
- GPU3–5：A/B/C seed 43；
- GPU6：FDB + semantic eval；
- GPU7：实时/AEC/soak；
- 第三 seed 在首批完成后补跑。

小模型单会话 live 推理优先放一张 GPU，
以 microstep 调度避免跨 GPU 通信抖动——延迟评测最怕引入额外的通信噪声。

### 1B–3B 扩展

- 4–8 卡 FSDP/DDP（DDP 第 01 课讲过；FSDP 是把参数、梯度、优化器状态切片分摊到多卡的并行方式）；
- listener/router/control 参数很小；
- 报 global event batch 与 active audio seconds；
- 分布式训练不代表分布式单会话推理。

### Moshi scale track

- 先按 `moshi-finetune` 当前官方 README 运行 LoRA；
- 记录实际 GPU 型号、峰值显存、batch 与版本；
- 不从"有 8 卡"推断任意型号都能跑；
- 用同 FDB 场景比较行为定义，不要求 MiniMind 分数击败 Moshi。

## 14. 评测与指标

六组指标各回答一个问题：真在听吗、判断对吗、停得快吗、说得对吗、老本行退步没有、系统扛得住吗。

### 14.1 持续监听证明

- 生成期间被 listener 消费的 chunk 数；
- `consumed_at - available_at`；
- queue backlog；
- 生成期间 user feature 对 control/content 的可测影响；
- 把 user chunks 置零后的性能差（置零后性能不变，说明"在听"是装的）。

### 14.2 Control

- CONTINUE/PAUSE/REPLAN event F1；
- false stop on backchannel；
- false replan on side/ambient；
- true interruption recall；
- control detection latency。

### 14.3 Stop 与响应

$$
L_{stop}=t_{speaker\_silent}-t_{user\_interrupt\_anchor}
$$
$$
L_{replan}=t_{new\_answer\_first\_audio}-t_{new\_user\_end}
$$
报 p50/p95（中位数与 95 分位），并拆：

- listener；
- control；
- scheduler；
- PCM buffer/client。

### 14.4 语义

- correction QA exact/F1；
- new-question answer correctness；
- stale fact rate；
- forbidden old-plan mention；
- resume success；
- task completion。

定义 context corruption：

$$
CCR=
\frac{\#\text{post-overlap outputs containing incompatible old/new plan}}
{\#\text{semantic overlap cases}}
$$
人工规则与判定器版本必须固定，并抽样复核。

### 14.5 Non-interruption regression

学会插话不能忘了说话。检查：

- 无 overlap QA；
- Thinker text；
- Talker WER/UTMOS（无参考的自动语音自然度评分，越高越自然）/speaker；
- TTFA/RTF；
- backchannel counterfactual output semantic consistency。

### 14.6 系统

- 80ms frame deadline miss；
- listener/decoder duty cycle；
- state bytes/session；
- user memory tokens；
- reused KV tokens / re-prefill tokens；
- queue high-water mark；
- 30 分钟内存增长；
- AEC 条件 false events。

## 15. 验收条件

必须全部满足：

- [ ] assistant 生成期间 listener 持续消费真实到达 chunk；
- [ ] 独立模式已冻结 `duplex-baseline-v1` 与 exact checkpoint revision；
- [ ] listener 通过 prefix/suffix 因果性测试，offline full-utterance hidden 未进入 B/C；
- [ ] 不存在包住整个 generator 的全局长锁；
- [ ] 所有 user feature 遵守 `available_at`；
- [ ] B/C 参数、listener、数据、updates 可比；
- [ ] backchannel false stop 达预注册上限；
- [ ] side/ambient false replan 达上限；
- [ ] correction 后回答使用新信息；
- [ ] CCR 被逐 case 复核；
- [ ] PAUSE/RESUME 复用同 branch state；
- [ ] REPLAN 有 branch lineage，不在旧答案尾部盲续；
- [ ] stop latency 测到实际 playback silence；
- [ ] clean/no-AEC/post-AEC 分开报告；
- [ ] FDB 每 case artifact 完整；
- [ ] 30 分钟 soak 无 state/queue 泄漏；
- [ ] 明确 MiniMind 仍是小规模机制原型。

推荐目标：

- true interruption recall ≥0.85；
- backchannel false stop ≤5%；
- side/ambient false replan ≤2%；
- p95 stop latency ≤500ms；
- correction task success ≥70% pilot / ≥80% standard；
- CCR 比 channel-fusion 基线或 hard baseline 明显降低；
- 80ms deadline miss <1%；
- non-overlap QA 回退 ≤2 个百分点。

### B/C 主比较的预注册判据

B/C 使用相同 case、播放 trace 和 seed，主统计按 case 配对。对三个 seed 做两层
bootstrap（有放回重采样估计不确定度的统计方法）：先重采样 seed，再重采样 case，共 10,000 次，报告绝对差和 95% CI。
只有同时满足下面四项，才能写"C 的 cross-attention memory 优于 B 的 channel
fusion"：

1. C 的 grounded correction success 比 B 至少高 3 个百分点，且配对 95% CI 下界
   高于 0；
2. C 的 CCR 比 B 至少低 2 个百分点，且以 `B-C` 表示的改善量其配对 95% CI 下界
   高于 0；
3. C 的 p95 stop latency 相对 B 非劣，预注册界限为增加不超过 100ms；对逐 case
   stop latency 差值，95% CI 上界不超过 100ms；
4. C 的 non-overlap QA 相对 B 非劣，95% CI 下界不低于 -2 个百分点。

若 grounding 或 CCR 达标但 latency 非劣未通过，只能写成质量—延迟权衡，不能宣布
C 整体胜出。上述阈值、case 集和 bootstrap 脚本 hash 必须在查看 test 前写入
`stats_plan.yaml`。先看结果再定判据，统计上等于作弊。

## 16. 失败诊断表

从最便宜、最可观测的一层开始查：先看队列和时间戳，再看 gate 和 branch，最后才怀疑训练本身。

| 症状 | 原因 | 诊断 | 修复 |
|---|---|---|---|
| 声称双工但输入积压 | generator 长锁 | queue backlog/lock trace | microstep scheduler |
| user feature 有未来 | offline batch 泄漏 | available-at assertion | 因果 mask/replay |
| B 语义强但回答混乱 | context corruption | branch/token source trace | 更快 PAUSE 或 C |
| C 完全忽略用户 | gate collapse | gate/zero-user ablation | auxiliary grounding/lr |
| PAUSE 后无法继续 | KV 被清空 | branch diff | snapshot/cursor |
| REPLAN 仍说旧答案 | 从错误 state 续写 | lineage/stale fact | safe checkpoint 新 branch |
| stop command 快、仍听到 1s | PCM/client buffer | buffer timeline | 缩短/可撤销 buffer |
| 模型一直被自己打断 | echo | far-end correlation | AEC/clean channel |
| backchannel 被当新问题 | 数据/控制头弱 | FDB bucket | event-balanced hard negative |
| memory 越聊越慢 | user memory 无界 | tokens/latency vs time | window+summary |
| 多线程运行时进程随机退出 | 同 state 并发写 | race test | 单 owner scheduler |
| 合成 test 好真人差 | prosody/timing 域差 | real-only subset | 真实 stereo 数据 |
| shared prefix 因未来 suffix 改变 | listener 使用 centered/offline future | suffix invariance test | prefix-only 或 closed-block |

## 17. 逐 case 分析

每个 case 必须能播放并查看：

- 用户、助手、far-end 和 post-AEC waveform；
- packet arrival 与 `available_at`；
- listener feature 的可用时间；
- control probability 和 action；
- channel-fusion token source 或 memory gate；
- branch lineage；
- generated、sent 和 played codec boundary；
- old plan 与 new plan；
- final transcript 和 semantic judgment；
- queue/deadline timeline；
- error taxonomy。

错误 taxonomy：

- `not_actually_listening`；
- `control_misroute`；
- `context_corruption`；
- `weak_grounding`；
- `state_reset`；
- `stale_plan`；
- `playback_buffer`；
- `echo_aec`；
- `deadline_runtime`；
- `evaluation_error`。

强制逐个审：

- 所有 context corruption；
- 所有 false replan；
- 所有 backchannel false stop；
- B 对 C 错、C 对 B 错各 20 例；
- p95 stop/replan latency 各 20 例；
- no-AEC 失败/post-AEC 恢复 20 例；
- 30 分钟 soak 中全部异常事件。

自动指标负责筛出要看的样例；结论必须建立在逐 case 的时间轴和音频上。

## 18. 交付物

1. `StreamEvent` 与统一 wall-clock 契约；
2. `decode_step`/microstep scheduler；
3. hard-cancel baseline；
4. channel-fusion adapter；
5. gated cross-attention user memory；
6. control head；
7. versioned branch/state；
8. cancelable PCM buffer；
9. stereo/AEC 数据 manifest；
10. 三臂 checkpoint 与报告；
11. FDB/FDB1.5 逐 case artifact；
12. 30 分钟 soak 报告；
13. Moshi LoRA scale-track 报告（可选但推荐）。

## 19. 复现清单

- [ ] base/listener/checkpoint hash；
- [ ] 独立模式 `duplex-baseline-v1` manifest/base outputs；
- [ ] listener source span/available-at 与 suffix invariance test；
- [ ] stereo waveform/far-end/AEC hash；
- [ ] clock/packet/jitter seed；
- [ ] available-at convention；
- [ ] routing adapter params/FLOPs；
- [ ] control class/event counts；
- [ ] branch/state schema version；
- [ ] sampler/voice/playback buffer；
- [ ] AEC implementation/config；
- [ ] FDB version/commit；
- [ ] evaluator/CCR judge version；
- [ ] runtime trace；
- [ ] license/consent；
- [ ] test 未进训练。

## 20. 前沿对照与改造方向

[Moshi](https://arxiv.org/abs/2410.00037)（本课必读）把"听"和"说"直接建成两条并行音频流：用户流和助手流放在同一时间轴上，外加一条 inner monologue 文本流；同一个模型每一帧都在读对方那条流、写自己这条。它没有"打断检测"这个模块——用户随时开口，下一帧的输入里天然带着用户流的最新内容，要不要停嘴是从数据里学出的行为。延迟由帧结构决定：Mimi 一帧 80ms，其论文摘要报告理论延迟 160ms、实测约 200ms（第 01 课引过这组数字）。[Synchronous LLMs](https://arxiv.org/abs/2409.15594)（本课必读）在 token 序列里显式编码墙钟时间，用合成加真实数据训练，还能在训练中模拟网络延迟。[LSLM](https://arxiv.org/abs/2408.02622) 与 [How Should LLMs Listen While Speaking?](https://arxiv.org/abs/2605.10199)（都在本课必读）和我们同一条路线：给单流模型外接用户通道——前者比较 early/middle/late fusion 的层位，后者正是本课 B/C 受控比较的直接蓝本。评测侧，[Full-Duplex-Bench](https://arxiv.org/abs/2503.04721) 及其 [v1.5](https://arxiv.org/abs/2507.23159) 已是双工行为的公共量尺，本课 Step 13 直接采用。

参数量和数据时长是规模问题：我们 26M 主体对前沿系统十亿量级的主干，砸钱能缩小。机制差距有两条，都是本课正面处理的：其一，双流对单流外挂——Moshi 的用户流占自己的通道、不占主序列位置，天然不存在 8.1 节 `audio_step`/`global_pos` 双计数、插入事件搞乱 diagonal delay 的问题；我们的 B/C 是把第二条流嫁接到单流模型上，嫁接点选主序列还是旁路内存，正是本课的比较对象。其二，打断决策——前沿倾向让停嘴成为模型自身的生成行为，我们用显式三分类控制头外挂：可解释、可单独评测，代价是多一个要标数据的模块。



1. **hybrid 路由：平时走记事本，确认改需求才上桌。** user 事件平时只进 cross-attention memory（C 路）；控制头给出 `REPLAN` 后，才把触发该动作的 user 事件按 8.1 的九路规则补写进新 branch 的主序列（B 路），给重规划最强的 grounding。改动位置：`DuplexRouter` 的 REPLAN 分支加一个事件回灌调用，复用 8.1 的打包函数和单测。预算：复用 B/C 的 listener、数据与控制头，只训 routing adapter，pilot 档 10k 合成场景，4 卡 1–2 天。预期：CCR 接近 C、grounded correction success 接近 B。失败判定：两头都不占——CCR 不低于 B，或 correction success 不高于 C。
2. **把第 06 课的 four-action policy 接成前置层。** 把[第 06 课](06_turn_policy.md) `turn-policy-v1` 的四动作 logits 拼进本课 `Turn/ControlHead` 的输入特征，作为显式可切换的前置层。改动位置：控制头输入拼接，listener 与 routing 不动。预算：控制头参数很小，重训控制头即可，1 GPU-day 以内。预期：backchannel false stop 与 side/ambient false replan 下降，三分类 F1 上升。失败判定：F1 不动，说明第 06 课的声学时序特征与本课 listener 语义特征互相冗余。
3. **双流版玩具（重写输入结构级）。** 把 user 流从"事件"升级成固定节拍的第十条 lane：九路输入再加一路 user-audio lane，每 80ms 一格，无人说话时填 pad——向 Moshi 的并行流靠近一步。改动位置：`OmniDataset` 的打包逻辑与 `MiniMindOmni.forward` 的输入拆分；8.1 的双计数规则整体作废，因为 user 流不再占主序列位置。预算：结构级改动，pilot 合成场景从头训 mini 规模，8 卡 2–4 天。预期：插入错位类 bug 消失，CCR 与 B 持平或更低，grounding 不弱于 C。失败判定：non-overlap QA 回退超过 2 个百分点（第 15 节阈值），说明固定节拍 lane 在这个参数量下挤占了内容容量。

How Should LLMs Listen While Speaking? 的核心对照——channel fusion 的 grounding 更强但更易 context corruption，cross-attention 路线更稳——本课 B/C 两臂就是它的缩小版复现。预期 pilot 档即可看到同方向趋势：B 的 correction success 更高、CCR 也更高。若 C 的 CCR 反而更高，先查 gate 是否塌缩到零（第 16 节诊断表），确认记事本真的被翻过，再怀疑结论不迁移。

**更多顺手扩展：**

- learned memory write/read gate：让模型自己决定哪些 user 内容值得写进 memory、何时读；
- speculative assistant plan + interruption verifier：预判用户会不会打断，提前备好两套计划；
- multilingual overlap：中英混说场景下的打断与纠正；
- 多人/目标说话人 routing：结合声纹，只响应目标说话人的打断；
- 端到端 AEC-aware training：把 AEC 残差当训练条件而不只是评测条件；
- 多 session continuous batching：多路会话共享一张卡的 microstep 调度；
- 把[第 04 课](04_multicodebook_talker.md)的 grouped-depth Talker 接入，并重测 80ms deadline；
- 最终与现代 Thinker（Nemotron 等）对接，但保留本课的 runtime 与路由边界——这正是[第 18 课](18_nemotron_finetuning.md)与[第 19 课](19_capstone_thinker_talker.md)要做的事。

## 21. 必读论文与阅读问题

每篇材料带一个能在本课产物里验证的问题进去，读完把答案写进报告。

### 21.1 完整双流模型

- [Moshi](https://arxiv.org/abs/2410.00037)：
  精读 parallel user/assistant audio streams、inner monologue、
  temporal/depth Transformer、delay 与 streaming evaluation。
  带着问题：并行双流让"完整 user turn 边界"这个概念彻底消失，而本课的 B/C 在哪一步仍然依赖边界信息（提示：控制头的 REPLAN 时机）？
  阅读检查：说明 parallel streams 如何避免依赖完整 user turn 边界。
- [Moshi 官方仓库](https://github.com/kyutai-labs/moshi)：
  看 streaming state、server/inference loop 与双流 tensor layout。
  带着问题：它的推理循环每帧做什么？对照第 12 节的 scheduler 伪代码逐行找对应物，把找不到对应物的行标出来——那就是双流原生设计省掉的复杂度。
- [Moshi-finetune](https://github.com/kyutai-labs/moshi-finetune)：
  看 stereo dataset schema、LoRA 配置和官方硬件说明。
  带着问题：它的双声道 schema 与本课 9.3 的 schema 差哪几个字段？scale track 开跑前先列差异表，缺的字段决定你的数据能不能直接喂进去。

### 21.2 Wall-clock 同步

- [Beyond Turn-Based Interfaces: Synchronous LLMs](https://arxiv.org/abs/2409.15594)：
  读 synchronous modeling、time information、synthetic/real data recipe、
  latency simulation。
  带着问题：它把时间显式编码进序列的方式，对应本课 5.1 的六个时间戳和 8.1 的 wall-clock bucket 中的哪几个？哪些信息我们记录了却没喂给模型？
  阅读检查：说明 token LM 如何表示并模拟 240ms 网络延迟。

### 21.3 Listen while speaking

- [Language Model Can Listen While Speaking](https://arxiv.org/abs/2408.02622)：
  读 listening/speaking channels、streaming SSL encoder、
  early/middle/late fusion 和 interruption experiment。
  带着问题：它的 fusion 层位结论能否指导本课 `adapter_layers: [2, 4, 6]` 的选层？如果要重选三层，你选哪三层、用第 15 节哪条判据验证？
  阅读检查：比较 fusion 层位对已有生成能力的影响。
- [How Should LLMs Listen While Speaking?](https://arxiv.org/abs/2605.10199)：
  精读 channel fusion 与 cross-attention routing 的受控设置、
  QA grounding 和 context corruption 分析。
  带着问题：对照本课 8.3 的公平约束，列出它控制了哪些变量、我们控制了哪些——参数配平到 5% 公差这条，两边谁更严格？
  阅读检查：说明 channel fusion 与 cross-attention 在 grounding 和
  robustness 上的差异。

### 21.4 行为评测

- [Full-Duplex-Bench](https://arxiv.org/abs/2503.04721)：
  读四类行为任务、自动测量和 latency 定义。
  带着问题：它的 latency 秒表起止点与本课 14.3 的 $L_{stop}$、$L_{replan}$ 是否一致？不一致时，报告里怎么同时给出两套可互相换算的数字？
- [Full-Duplex-Bench v1.5](https://arxiv.org/abs/2507.23159)：
  重点读 overlap 四场景与 repair-first/continuity-first 行为差异。
  带着问题：把这份行为清单翻译成 Step 7 的控制标签规则，哪些场景下"停"和"不停"都算对？这些模糊场景在你的标注指南里怎么写？
  阅读检查：按场景列出应立即停止和可以继续输出的条件。

读完材料回头看，第二幕到这里收官。三课连起来：第 05 课把耳朵改成流式，系统能一边收音一边编码；第 06 课教会它分辨"这声音是接话、附和还是打断"；本课把判断接上了执行——现在系统能边听边说，边决定要不要插话：你附和它不停，你纠正它半秒内闭嘴、用新条件重新规划，每个决定都有毫秒级时间轴可查。"回合制玩具"这个标签，从这课起可以撕掉了。下一幕长出眼睛：目前系统看图只会把图片缩放成固定尺寸、切成固定数量的 patch，文档小字和高分辨率细节全被压没。[第 08 课](08_dynamic_vision.md)先解决"任意分辨率怎么看"：动态切片、多图身份与二维位置编码 M-RoPE。
