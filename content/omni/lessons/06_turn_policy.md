---
id: 06_turn_policy
title: "学习式话轮策略"
summary: "学出来的话轮策略，能同时少抢话、少干等、别把“嗯嗯”当打断，还得在用户真要插话时立刻反应过来吗？"
unit: realtime
play_tools: []
checkpoints:
  - "把五件事掰开：检测有没有人声、判断一句话说完没、该不该接话、“嗯嗯”式附和、用户真打断。"
  - "做出双声道事件标注，并实现 HOLD、TAKE_TURN、BACKCHANNEL、BARGE_IN 四个动作（憋住、接话、附和、被插话让位）。"
  - "用事件窗口、概率校准和 hysteresis（迟滞：加缓冲防抖动）来评估，别用被 HOLD 占大头带偏的 frame accuracy。"
  - "用合成反事实和 blind replay（盲评回放）检验它在真实会话里的表现。"
---

# 第 06 课：用四动作策略判断话轮

> 内容：对话时序、事件检测与策略校准<br>
> 建议周期：10–18 天，数据标注是主要成本<br>
> 硬件：1–4 卡训练；8 卡可并行处理数据、seed 和实时回放<br>
> 产物：四动作 `TurnHead`、双声道事件集、可校准的运行时策略

## 1. 模型什么时候该开口

第 05 课已经让音频随到随编码，但系统仍以约 800ms 静音作为开口条件；生成期间一旦检测到人声，它就立即停止。流式编码解决了输入延迟，没有解决话轮判断。

停顿可能表示话轮结束，也可能只是说话人在思考；"嗯"既可能是附和，也可能是插话。VAD（voice activity detection）只能判断当前是否有人声，不能区分这些会话状态。只用 VAD 会产生两类错误：句中停顿时过早回答，以及把附和误判为打断。

本课加入 `TurnHead`，每 80ms 在 `HOLD`、`TAKE_TURN`、`BACKCHANNEL` 和 `BARGE_IN` 四个动作中选择一个。输入同时包含声学和语言特征，让模型利用语法与语义完整性区分停顿和话轮结束。

[第 07 课](07_full_duplex_routing.md)要做真双工：系统边说边听，还能根据你的插话改答案。如果连"该不该开口"都判断不了，双工只会退化成互相抢话；每次检测到人声就重新规划，等于把 800ms 规则的毛病放大成灾难。第 07 课的 `CONTINUE/PAUSE/REPLAN` 需要本课这种事件判断做触发依据。

验收时，同一段双声道对话会交给三套策略并排回放。报告分别统计句中停顿、附和和旁路人声场景，并保存每个决策的时间戳。

本课术语：

| 术语 | 简要解释 |
|---|---|
| turn policy | 根据会话状态决定下一步动作（听、接话、附和、停下）的策略 |
| VAD | 只判断"现在有没有人声"的小模型，不懂语义（第 01 课见过） |
| endpointing（端点检测） | 判断"这句话到这里说完了"的那个时刻 |
| turn-taking | 话轮交接：对话双方轮流拿"发言权"的过程 |
| TRP | transition relevance place，一句话里适合换人说话的时间点 |
| backchannel | "嗯嗯""对"这类表示在听、但不抢话的短反馈 |
| barge-in | 系统说话期间，用户插进来说有实质内容的新话 |
| VAP | 不用人工标签，直接预测未来几秒双方谁会出声的自监督目标 |
| hysteresis | 进门槛高、出门槛低的防抖动开关，像空调温控，防止动作反复横跳 |
| 校准（calibration） | 让模型报的概率和真实命中率对得上：说 0.8 就该有八成对 |

## 2. 本课解决的问题

MiniMind-O 当前 `RealtimeSession` 做两件事：

- 连续约 800ms 静音后判定用户说完；
- 助手生成期间检测到用户语音就 hard interrupt（硬打断：直接掐掉当前生成）。

Silero VAD 估计当前是否有人声，但不能区分以下事件：

- 用户只是在句中停顿，还是已经让出话轮；
- "嗯""对"是 listener backchannel，还是新请求；
- 助手应该接话、继续等、短促回应，还是停止当前回答；
- 背景旁人说话是否应打断。

turn policy 是根据当前会话状态选择下一步对话动作的策略。本课在相同双声道会话、真实到达时序和内容生成器下比较固定规则与学习式策略，分别测量抢话、空等、backchannel 误中断和 barge-in 检测延迟。

本课只训练**动作策略**，不改变 Thinker 回答内容，也不把 overlap（重叠期，即双方同时出声的时段）里的用户内容注入正在生成的主模型——那是第 07 课的活。

## 3. 开始前需要准备什么

本课不依赖第 05 课的 streaming student，需要准备：

- 官方 `minimind-3o`/`baseline-v1`；
- 一个只读取当前时刻及过去 waveform 的 causal/prefix-only feature backend（特征后端：把原始音频加工成决策头输入的模块；causal 指任何时刻只用已到达的音频）；
- 按 80ms 滑窗的双声道音频 replay（回放：按真实时间轴把录好的会话重新喂给系统）；
- 当前 `RealtimeSession` 作为规则基线；
- 一份许可清晰、保留双方独立声道与时间戳的对话集。

如果已有 `listener-v1`，可作为另一 feature backend，但三臂必须共用同一个 backend——否则比出来的差异说不清是策略的还是特征的。

没有第 05 课产物时，可以选择以下一种 feature backend：

1. `prefix-recompute`：时刻 $t$ 只把 `waveform[:t]` 送入冻结 SenseVoice，取稳定前缀摘要；它计算昂贵，但不读取未来；
2. `closed-block`：只处理已经完整到达的 320ms block，block 内可用 offline encoder，但输出的 `available_at`（这份特征在真实运行时最早能拿到的时刻）至少为 block end + compute。

常见错误是偷看未来:不得从完整 utterance 一次性提取双向 SenseVoice hidden，再按 80ms 切片冒充实时 feature。双向编码器的每一帧输出都"看过"整句话，包括还没发生的部分。此类 offline hidden 的 `available_at` 必须是 `utterance_end + encode_latency`，只能做 teacher/upper-bound（教师信号或性能上界参照），不得进入 B/C 实时主结论。

输出 checkpoint：

```text
turn-policy-v1-80ms
```

## 4. 完成后应具备的能力

1. 区分 VAD、endpointing、turn-taking、backchannel、barge-in；
2. 理解 transition relevance place（TRP）；
3. 构造双声道、事件级标注数据；
4. 实现 80ms 的 `HOLD/TAKE_TURN/BACKCHANNEL/BARGE_IN`；
5. 处理严重类别不均衡；
6. 做概率校准与 hysteresis；
7. 使用事件级而非单纯 frame accuracy；
8. 在真实会话和合成反事实上分别验证。

## 5. 原理:边造边讲

五个机制，每个都回答同一串问题：为什么需要、怎么工作、精确定义在哪、落在哪份代码或数据里、怎么验证。

### 5.1 四种动作分别表示什么

人在对话里的每个时刻其实只做四类事：维持现状、接过话头、给个不抢话的回应、或者在对方插话时停下。把"系统下一步应做什么"收敛成这四个动作，策略学习才有明确的标签空间。

以"系统下一步应做什么"为标签：

- `HOLD`：继续听或继续说，不改变 floor（发言权归属）；
- `TAKE_TURN`：用户已让出话轮，系统应开始主回答；
- `BACKCHANNEL`：允许系统发极短反馈，不取得完整话轮；
- `BARGE_IN`：用户在系统输出期间提出有语义的新内容，系统应停止/暂停并交给后续 replanning（根据新内容重新规划回答，第 07 课实现）。

**标注落点。** 标注时按以下规则区分容易混淆的情况：

- 用户说"嗯"时，若是对助手的 listener response，通常不是 barge-in；
- 助手说"嗯哼"时，标签是系统 backchannel action；
- "旁边的人说话"不是目标用户 barge-in；
- 同一个声学形态可能因上下文有不同标签。

最后一条是本课反复回来敲打的钉子：同一声"嗯"，语境不同标签就不同。测试集必须包含同一关键词在不同上下文对应不同动作的样例（见第 6 节），任何靠关键词表实现的策略都过不了这组测试。

### 5.2 VAD 和 VAP 的输入输出

VAD 回答"现在有没有人声"，是关于现在和过去的。而接话决策本质上是预测未来："接下来一小段时间，对方还会不会继续说？"这正是 VAP 的目标。

VAD 估计当前和过去的 speech activity。Voice Activity Projection（VAP）预测未来一段时间内双方的 activity pattern（谁在什么时候出声的模式），可从没有人工语义标签的 stereo audio 学到与 turn-shift 和 backchannel 相关的声学时序信号。

VAP 不直接提供四动作标签。它知道"接下来大概率安静"，但不知道这份安静是"说完了"还是"在想词"——语义完整性它看不见。所以本课把 VAP 当辅助目标（第 7 节的 `vap_head`），不当最终答案。

### 5.3 语言上下文用于判断话轮是否完整

你听到"我想订一张从上海到……"后面跟着 1 秒安静，不会以为对方说完了——句子在语法上悬着。语法、语义和语用完整性可以帮助判断用户是否说完，这是纯声学规则永远补不上的信号。

相同长度的静音在以下两句中应产生不同决策：

| 用户刚说的话 | 静音后的合理决策 |
|---|---|
| "我想订一张从上海到……" | 句子尚未完成，继续 `HOLD` |
| "明天下午三点，可以吗？" | 问题已经完整，可以准备 `TAKE_TURN` |

语言证据来自冻结 Thinker state 或增量文本状态（见第 6 节的 feature 表），不引入新的语言模型。

Step 10 的反事实测试专门构造这类配对：静音长度相同、语义完成度不同，看策略是否给出不同动作。

### 5.4 事件指标与 frame accuracy

连续 80ms frame 中，`HOLD` 数量远多于其他类。一个始终输出 `HOLD` 的模型也能得到很高的 frame accuracy（逐帧分类准确率），却永远不会接话——像一个从不开口的客服，"没说错话率"是 100%。

因此主指标使用事件窗口、延迟和会话行为，frame accuracy 只作辅助。`HOLD` 是"没有触发动作"的状态，不是一条可与 reference 匹配的事件。事件匹配器只接收 `TAKE_TURN`、`BACKCHANNEL` 和 `BARGE_IN`。`HOLD` 单独在无事件区间评测：统计区间内误触发次数、误触发时长，以及应该继续保持时被错误切换到其他动作的 frame 比例。

Step 2 会在跑任何模型之前，先用手工构造的会话把匹配器本身测一遍——量尺不准，后面全白干。

### 5.5 校准与 hysteresis

模型每 80ms 吐一个概率，抖动是常态：0.68、0.72、0.69……如果 0.7 是阈值，动作就会开了关、关了开。人不会这样——决定插话前会攒一点把握，开了口也不会因为半秒的犹豫马上缩回去。

模型输出的 raw probability 先经过以下运行规则，再转换为动作：

- 进入一个动作所需的概率阈值高于保持该动作的退出阈值；
- 概率需要连续满足 `K` 帧，不能由单帧尖峰触发；
- 两次同类动作之间保留 cooldown（冷却期：触发过一次后强制等待的最短间隔）；
- 只有达到最低说话或静音时长后，才允许触发对应动作。

这套"进出双阈值"就是 hysteresis。设进入阈值 $\theta_{enter}$、退出阈值 $\theta_{exit}$，约束 $\theta_{enter} > \theta_{exit}$：概率爬过高门槛才进入动作，跌破更低的门槛才退出，中间的抖动区间被吸收掉。阈值有意义的前提是概率本身可信——这由校准保证，校准质量用第 13 节的 ECE 公式度量。

hysteresis 的进入阈值、退出阈值、连续帧数和 cooldown 必须放入配置，并在报告中记录（Step 8 实现，第 11 节给配置样例）。凡是没写进配置的运行时魔法数字，都算未申报的实验变量。

## 6. MiniMind-O 当前的规则策略

| 文件/符号 | 当前行为 | 本课改造 |
|---|---|---|
| `SileroVAD` | 1024 sample 窗口，输出 speech prob | 保留为 acoustic feature/baseline |
| `RealtimeSession.__init__` | `min_silence_ms=800` | 外置 PolicyConfig |
| `push_chunk` | speech/silence 计数 | 每 80ms 调 TurnHead |
| `self.generating and self.speaking` | 立即 `interrupt` | 由 action 决定 |
| `webui/web_demo.py::poll_interrupt` | VAD interrupt→break generator | 记录/执行 policy event |
| `state['history']` | 仅完整轮次 | 增加当前 dialogue/response context |
| Thinker hidden | 未暴露 turn feature | 冻结 readout 或 prefix summary |

四种动作不能通过固定关键词表实现。语言证据必须来自冻结 Thinker state 或 tokenizer context；测试中要包含同一关键词在不同上下文中对应不同动作的样例。

任何语音/语言 feature 都必须携带四个字段——它们是防"偷看未来"的审计凭证：

- `source_start_ms`：这份 feature 使用的源片段起点；
- `source_end_ms`：这份 feature 使用的源片段终点；
- `available_at_ms`：运行时最早能拿到它的时间；
- `backend_mode`：取值为 `causal`、`prefix_recompute`、`closed_block` 或 `offline_teacher`。

运行时断言 `decision_ms >= available_at_ms`：决策时刻不能早于它所用特征的可用时刻，违反即是泄漏。

## 7. 目标架构

`TurnHead` 每 80ms 接收一次四类输入：

| 输入 | 提供的信息 |
|---|---|
| 用户声道经过 acoustic encoder 的状态 | 用户当前说了什么，以及声音是否还在继续 |
| 助手声道的 activity | 助手是否正在生成或播放语音 |
| 增量对话文本或 Thinker state | 当前话语在语言上是否完整 |
| 当前回复的计划和播放状态 | 系统是否已有回复，以及它进行到了哪里 |

`TurnHead` 输出 `HOLD`、`TAKE_TURN`、`BACKCHANNEL` 和 `BARGE_IN` 四个 logit。校准后的 hysteresis runtime 再把这些 logit 转成实际动作。

推荐 `TurnHead`：

```python
class TurnHead(nn.Module):
    def forward(
        self,
        user_audio_state,
        assistant_activity,
        dialogue_state,
        response_state,
    ):
        fused = self.fuse(
            user_audio_state,
            assistant_activity,
            dialogue_state,
            response_state,
        )
        action_logits = self.action_head(fused)        # [B, T, 4]
        future_activity_logits = self.vap_head(fused)  # [B, T, H, 2]
        assert action_logits.shape[-1] == 4
        return action_logits, future_activity_logits
```

可加 VAP auxiliary（辅助训练目标：和主任务共享骨干、只在训练时提供额外梯度），预测未来 2 秒双方 activity bins。

## 8. 数据 recipe

### 8.1 数据来源和用途

1. **Pilot 合成双声道**：
   用 MiniMind Text-to-Audio（T2A）、Audio-to-Audio（A2A）和 TTS 构造 pause/backchannel/interruption；仅验证接口，不能作为最终有效性证明。
2. **Standard 真实双声道**：
   获得明确同意的双人录音，建议 20–100 小时；或使用许可允许的 [AMI Meeting Corpus](https://www.amiproject.org/ami-scientific-portal/documentation/) 独立近讲声道，并进行目标说话人映射。
3. **Full 授权语料**：
   Fisher/Switchboard 等需相应 LDC 授权；不能因论文使用过就当作开放数据。
4. **评测**：
   [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench) 场景只用于 test/系统比较，不泄漏到训练。

AMI 是会议语料，不等同于助手对话。使用时要单独报告 AMI 和助手对话 test，不能合并平均分后外推。

### 8.2 会话数据字段

```yaml
session_id: string
sample_rate: 16000
user_channel: user.wav
assistant_channel: assistant.wav
channel_sync_error_ms: float
language: string
source: string
license: string
consent_id: string | null
segments:
  - speaker: user | assistant | other
    start_ms: 1200
    end_ms: 2500
    transcript: "..."
    dialog_act: question | statement | acknowledgement | correction
events:
  - type: TAKE_TURN
    anchor_ms: 2550
    tolerance_before_ms: 100
    tolerance_after_ms: 500
    annotators: [a1, a2]
    confidence: 0.9
frames:
  hop_ms: 80
  user_active: [0, 1, ...]
  assistant_active: [1, 1, ...]
```

### 8.3 标注指南

每个 non-HOLD 事件至少由两位标注者独立标注。先记录可观察事实，再选择系统动作：

1. 谁在何时说话；
2. 是否重叠；
3. 当前 utterance 语义是否完整；
4. 短响应是否仅 acknowledge；
5. 用户是否引入新内容/纠正；
6. 理想系统动作；
7. 可接受动作集合和容忍窗口。

允许多标签不确定：

```yaml
acceptable_events: [BACKCHANNEL]
no_event_acceptable: true
preferred: BACKCHANNEL
```

存在合理分歧时保留 `acceptable_events` 和 `no_event_acceptable`，不强制改成唯一事件标签。人在这些场景下本来就会有分歧，硬拗成单一"正确答案"只会把噪声写进标签。`no_event_acceptable: true` 表示该窗口内不触发事件也合理，不是要求预测一条 `HOLD` 事件。

### 8.4 预处理

- 保留 stereo，不先混成 mono——混了就分不清谁在说话、谁在重叠；
- 校正时钟漂移与声道 delay；
- 分离 `other speaker`；
- transcript 对齐到词级或短语级；
- feature 只能使用当前时刻前可用信息；
- full-utterance offline hidden 的 `available_at` 设为 utterance end 加实际 encode latency，并从实时 B/C 数据中排除；
- 系统 response state 来自实际播放，而非完整未来 response；
- 合成数据随机化 pause、prosody（韵律：语调、重音、节奏这些"怎么说"的信息）、噪声、AEC residual（回声消除后的残留：系统自己的播放声漏进麦克风、没消干净的部分）；
- 保存每个变换 seed。

### 8.5 切分

- speaker/session disjoint（同一说话人、同一会话不得跨切分出现）；
- 对话主题尽量分开；
- 合成模板不跨 split；
- 真实 test 不含训练说话人；
- FDB 完全隔离；
- 按语言、噪声和 overlap rate 分层。

### 8.6 三档规模

| 档位 | 规模 | 作用 |
|---|---:|---|
| pilot | 5–10h 合成 + 2h 真实 | 接口/标签/过拟合 |
| standard | 20–100h 真实双声道 | 主结论 |
| full | 500h+ 授权多域对话 | 泛化与稀有 barge-in |

### 8.7 License/隐私边界

- 论文开放不等于语料开放；
- Fisher/Switchboard 必须持有 LDC license；
- 录制真人需同意用途、保留期限、撤回机制；
- transcript 需要 PII redaction（隐私信息脱敏：把姓名、电话这类可识别个人的内容抹掉）；
- 不发布不可再分发的 waveform；
- 合成数据要记录 TTS license；
- 标注者不能看到不必要的身份信息。

## 9. 按依赖顺序执行实验

### Step 1：把当前规则写成可复现配置

把现有逻辑写成配置，不改行为：

```yaml
vad_threshold: 0.8
min_speech_ms: 128
min_silence_ms: 800
interrupt_on_any_speech: true
```

用固定 replay 验证重构前后事件序列完全一致。规则基线要先被固定，后面所有"比规则好"的结论才有对照物。

### Step 2：先验证事件匹配器

训练前——量尺本身要先测。事件匹配必须在同一个 session 内一对一完成。一个预测不能命中两个 reference，两个预测也不能同时把同一个 reference 记成 true positive。按以下顺序实现：

1. 为每个 prediction/reference 组合检查动作是否兼容。prediction 的动作必须等于 reference 的 `preferred`，或位于其 `acceptable_events` 集合；
2. 只保留落在 `[anchor_ms-tolerance_before_ms, anchor_ms+tolerance_after_ms]` 内的候选边；
3. 候选边的 cost 为 `abs(pred_ms-anchor_ms)`；
4. 用带 dummy 节点的 Hungarian matching（匈牙利算法：求二分图最优一对一配对的经典算法；dummy 节点用来容纳配不上对的那些）做最大数量、最小时间差的一对一匹配；
5. 未匹配 prediction 记为 `spurious`，未匹配 reference 记为 `missed`；
6. 已匹配事件按 `pred_ms-anchor_ms` 另记 `early/on_time/late`。early/late 是命中事件的时间属性，不再额外计算一次 true positive。

匹配器输出：

- matched；
- early；
- late；
- missed；
- spurious；
- behavior cost。

下面的最小单测必须通过：

| 类型 | 内容 |
|---|---|
| reference | `TAKE_TURN`，锚点为 1000ms，可接受窗口为 `[900, 1300]` |
| prediction | `TAKE_TURN`，分别发生在 980ms 和 1040ms |

一对一匹配后只能有一个 prediction 被记为 `matched`，另一个必须记为 `spurious`。

再覆盖两个相邻 reference 窗口重叠、动作不兼容、空 prediction、空 reference，以及 `acceptable_events=[BACKCHANNEL], no_event_acceptable=true`。最后单独构造一段没有 reference 事件的 10 秒区间，确认 matcher 不生成 `HOLD` 事件，误触发只进入 no-event 指标。先用手工 20 段会话单测 evaluator，再计算模型指标。

### Step 3：标注 pilot

至少覆盖：

- 句中 200–1200ms pause；
- 用户结束当前话轮；
- 用户 listener backchannel；
- 助手 backchannel 时机；
- 用户纠正/新问题；
- 旁人语音；
- 笑声、咳嗽、填充词。

计算 inter-annotator agreement（标注者间一致性：两个人独立标，结果有多大比例吻合），并在 manifest 中保留争议样例的 `acceptable_events` 与 `no_event_acceptable`。

### Step 4：抽取 frozen feature

每 80ms 组合：

- 当前的 Silero speech probability；
- 用户和助手最近一段时间的 activity history；
- 冻结的 projected audio state 或 Thinker state；
- 当前可用的增量文本状态；
- 当前生成和播放状态。

feature cache key 必须包含 checkpoint 与可用时间。

主实验缓存只能由 `causal`、`prefix_recompute` 或 `closed_block` backend 产生。`offline_teacher` 单独落盘，只用于蒸馏/upper-bound。

构造 shared-prefix property test（共享前缀测试：第 05 课用来抓未来泄漏的同一招）：

样例 A 和样例 B 使用相同的前缀，之后分别接上 `suffix_A` 和 `suffix_B`。

在两个 suffix 均未到达前，A/B 的主实验 feature、TurnHead logits 和 action 必须在预注册公差内一致；否则判定未来泄漏。道理很直白：未来还没发生，两个"未来不同"的样例在此刻就不该有任何可观测差异。

### Step 5：训练二分类 learned endpoint

先只做：

`HOLD` 与 `TAKE_TURN` 的二分类。

该模型构成主矩阵的 B 臂，用于单独比较 learned endpoint 与 800ms 静音规则。先证明"学出来的端点检测"能赢过计时器，再谈四动作——一步一个对照。

### Step 6：训练四动作 TurnHead

使用：

- class-balanced focal loss（对难样本和稀有类加权的分类损失）或 weighted CE；
- VAP future activity auxiliary；
- label tolerance；
- hard-negative mining（专挑模型最容易答错的负样本反复训练）；
- 不用关键词规则补标签。

### Step 7：校准

在 dev 上做 temperature scaling（温度缩放：给 logit 除一个标量温度再 softmax，最简单的事后校准法）。每类输出 reliability diagram（可靠性图：横轴模型报的置信度、纵轴实际正确率，理想情况是对角线）、ECE、Brier。阈值按预注册 cost matrix（各类错误的代价表：比如"误打断用户"比"晚接话 200ms"贵多少）选择，不按 test。

### Step 8：实现 hysteresis runtime

运行时按优先级检查动作。示例规则是：`BARGE_IN` 概率连续两帧高于 0.8 时触发 `BARGE_IN`；否则，`TAKE_TURN` 概率连续三帧高于 0.7 时触发 `TAKE_TURN`；否则，`BACKCHANNEL` 概率高于 0.65 且 cooldown 已结束时触发 `BACKCHANNEL`。动作进入后，只有对应概率连续满足更低的退出阈值才退出；其余情况保持当前动作或回到 `HOLD`。因此每个动作都必须同时配置 `enter` 与 `exit`，并满足 `enter > exit`。

示例中的数值不能直接用于正式实验。实际阈值由 dev 校准后写入配置，并保存对应 reliability diagram 和 cost matrix。

### Step 9：固定内容生成器回放

三臂共享同一 MiniMind response audio。replay 双声道时：

- 不重新采样回复内容；
- 只比较什么时候开始/停止/短回应；
- 记录 policy event 与 playback event。

这一步是本课实验设计的关键控制：回答内容完全一样，三臂唯一的自由度是"什么时候做什么动作"，任何指标差异都只能来自策略。

### Step 10：反事实测试

为同一 utterance 构造配对样例，每次只改变一个变量：

- pause 长度；
- "嗯"位置；
- assistant 是否正在说；
- 用户后续是否继续句子；
- 旁人声道；
- 语义完成度。

比较配对样例的概率和最终动作，确认策略对该变量产生预期响应。每次只改一个变量，否则无法解释动作变化。

### Step 11：真实会话 blind test

使用至少 100 段未见过的 speaker/session。标注者不知道样例来自哪个系统臂。同时统计任务完成率、主观自然度和事件指标；主观分单独报告。

### Step 12：回归

确认本课没有改变：

- Thinker text；
- Talker waveform；
- voice；
- sampler；
- ASR；
- 仅改变播放/接话控制。

拿第 01 课的 golden cases 和 regression 命令跑一遍：本课动的是"什么时候说"，"说什么、什么音色"必须一个字节都没变。

## 10. 三个对照实验组

| 臂 | 决策 | 动作 |
|---|---|---|
| A | Silero + 800ms + any-speech interrupt | HOLD/TAKE/hard stop |
| B | learned acoustic+linguistic endpoint | HOLD/TAKE |
| C | calibrated four-action TurnHead | HOLD/TAKE/BACKCHANNEL/BARGE_IN |

B/C 使用同一份 frozen features。两臂之间的差异限定为动作头、训练标签和对应的运行策略。

## 11. 配置与伪代码

```yaml
experiment: lesson06_turn_policy
base_checkpoint: hf://jingyaogong/minimind-3o
base_revision_or_sha256: ee3febbd08cc5b2bd41c039c825a8934232fee33
frame_ms: 80
feature_backend:
  audio: closed_block_sensevoice
  block_ms: 320
  lookahead_ms: 0
  available_at: block_end_plus_compute
  dialogue: frozen_thinker_state
head:
  hidden_size: 256
  layers: 2
  actions: [HOLD, TAKE_TURN, BACKCHANNEL, BARGE_IN]
loss:
  type: focal
  gamma: 2.0
  class_weights: manifest-derived
  vap_aux_weight: 0.2
calibration:
  temperature_scaling: true
runtime:
  thresholds:
    take_turn: {enter: 0.70, exit: 0.45}
    backchannel: {enter: 0.65, exit: 0.40}
    barge_in: {enter: 0.80, exit: 0.50}
  consecutive_frames: {take_turn: 3, backchannel: 1, barge_in: 2}
  exit_consecutive_frames: {take_turn: 2, backchannel: 1, barge_in: 2}
  cooldown_ms: {backchannel: 1200, barge_in: 500}
```

事件 evaluator 的核心不是逐 pair 判断，而是 session 内的一对一分配。另外两条实现约定写在代码外面：`preds` 和 `refs` 在进入该函数前都必须过滤掉 `HOLD`；若 reference 允许不触发事件，由 `no_event_acceptable` 在窗口级 evaluator 中处理，不向 Hungarian 矩阵添加一条虚构的 `HOLD` reference。

```python
def match_session(preds, refs):
    # 行是 prediction，列是 reference；INF 表示动作不兼容或超出窗口。
    cost = full((len(preds), len(refs)), INF)
    for p_idx, pred in enumerate(preds):
        for r_idx, ref in enumerate(refs):
            action_ok = (
                pred.action == ref.preferred
                or pred.action in ref.acceptable_events
            )
            in_window = (
                ref.anchor_ms - ref.tolerance_before_ms
                <= pred.at_ms
                <= ref.anchor_ms + ref.tolerance_after_ms
            )
            if action_ok and in_window:
                cost[p_idx, r_idx] = abs(pred.at_ms - ref.anchor_ms)

    # 实现时添加 dummy 行列，使目标先最大化有限匹配数，再最小化时间差。
    pairs = hungarian_with_dummies(cost, invalid_cost=INF)
    pairs = [(p, r) for p, r in pairs if cost[p, r] < INF]

    used_pred = {p for p, _ in pairs}
    used_ref = {r for _, r in pairs}
    spurious = [p for p in range(len(preds)) if p not in used_pred]
    missed = [r for r in range(len(refs)) if r not in used_ref]
    return pairs, spurious, missed
```

## 12. 训练预算与 8 卡分配

| 工作 | 卡分配 |
|---|---|
| feature cache | 2–4 卡离线并行，保留 session 顺序 |
| B/C 两臂×3 seed | 6 卡 |
| 实时 replay | 1 卡 |
| FDB/人工试听产物 | 1 卡 |

模型头较小，默认每张卡运行独立的 feature、seed 或评测任务。只有单任务吞吐成为瓶颈时再使用 DDP，并记录通信开销。

预算重点——这课烧的主要不是 GPU，是人：

- 标注人时；
- 真实双声道小时；
- event 数，尤其 backchannel/barge-in；
- feature cache storage；
- 实时 replay 小时。

报告每类事件数量、会话数量和总时长，不能只报告音频小时数。稀有事件的数量决定结论的统计强度：几百小时音频里若只有几十次 barge-in，recall 数字撑不起任何结论。

## 13. 评测指标

### 13.1 事件检测

- event precision/recall/F1；
- early/late/miss/spurious；
- latency p50/p95；
- PR curve；
- 每会话 false event。

Frame accuracy 仅作辅助（原因见 5.4）。

### 13.2 轮次行为

- false cut（抢话）：用户尚未完成就 TAKE；
- missed take-turn（空等）：用户已让出话轮，系统迟迟不接；
- gap：

$$
Gap=t_{assistant\_start}-t_{user\_end}
$$
- overlap duration；
- gap/overlap 分布与真人 reference 的 Wasserstein distance（两个分布之间的"搬土距离"：把一个分布搬成另一个形状的最小代价，比只比均值更能反映整体差异）；
- task completion。

### 13.3 Backchannel

- opportunity-level precision/recall；
- false full-interrupt rate；
- response onset；
- cooldown violation；
- 人评 appropriateness。

### 13.4 Barge-in

- barge-in event F1；
- detection latency；
- stop command latency；
- false stop on user backchannel；
- false stop on side speech；
- 本课不评 replan 内容正确性。

### 13.5 校准

$$
ECE=\sum_b\frac{|B_b|}{N}
|\operatorname{acc}(B_b)-\operatorname{conf}(B_b)|
$$

ECE（expected calibration error，期望校准误差）把预测按置信度分桶，逐桶比较实际正确率和平均置信度的差再加权平均：$B_b$ 是第 $b$ 个置信度桶的样本集合，$N$ 是总样本数。另报 Brier score（预测概率与 0/1 真值的均方差）和 per-class reliability。

## 14. 验收条件

本课的 false-cut/latency Pareto 比较以两项数值越低越好。若方案 A 的 false-cut rate 和 latency 都不高于方案 B，且至少一项更低，则 A 支配 B；未被其他方案支配的方案集合就是 Pareto frontier（帕累托前沿：无法在不牺牲一项的前提下改进另一项的那批方案）。用 Pareto 而不用单一总分，是因为"宁可抢话还是宁可迟钝"取决于产品场景，评测阶段不替业务拍板。

- [ ] 双声道未先混音；
- [ ] 事件标签有操作定义与容忍窗口；
- [ ] 至少两位标注者，报告一致性；
- [ ] 合成数据仅作 pilot，最终含真实 test；
- [ ] 三臂内容生成完全固定；
- [ ] B/C 仅使用 causal/prefix-only feature，shared-prefix leakage test 通过；
- [ ] offline teacher feature 未进入实时主结论；
- [ ] B 相对 800ms 基线在 false-cut/latency 上不被支配，且至少一项更低；
- [ ] C 的 backchannel false-interrupt 显著低于 A；
- [ ] C 的 barge-in recall 达预注册阈值；
- [ ] side speech 不被大量误判；
- [ ] threshold 只用 dev 校准；
- [ ] 报事件 F1，不以 frame accuracy 取代；
- [ ] 明确本课还不是真双工内容融合。

推荐阈值：

- TAKE event F1 ≥0.85；
- false cut 相对 A 降低 ≥25%；
- median gap 不增加超过 150ms；
- barge-in recall ≥0.85；
- backchannel→false stop ≤5%；
- side-speech false stop ≤2%。

以上阈值只作为起始参考。正式实验应根据目标域和 baseline 波动在 test 前预注册。

## 15. 根据症状定位失败环节

| 症状 | 原因 | 诊断 | 修复 |
|---|---|---|---|
| frame accuracy 99%、从不接话 | 类别不均衡 | event recall | focal/balanced sampling |
| pause 一长就抢话 | 只学声学静音 | incomplete-utterance cases | 加语言状态/hard negative |
| 所有"嗯"都打断 | 无语义/双方状态 | backchannel confusion | stereo + context |
| 旁人说话触发 | 目标说话人未知 | side-speech bucket | speaker/channel feature |
| B 好 C 差 | 四类标签噪声 | confusion/IAA | 重写标注指南 |
| 训练好真实差 | 合成 timing 太规整 | real-only report | 增真实数据/domain aug |
| 概率极端不可靠 | 未校准 | reliability plot | temperature scaling |
| 频繁动作抖动 | 无 hysteresis | event burst count | enter/exit threshold |
| backchannel 太多 | 无 cooldown/cost | per-minute rate | cooldown + cost |
| stop 快但误停多 | 阈值偏激进 | PR/cost curve | 按业务 cost 选点 |
| dev latency 异常好 | full-utterance hidden 泄漏未来 | shared-prefix suffix swap | 改 causal/prefix-only backend |

最后一行是本课最隐蔽的坑：泄漏未来的模型在离线评测里像个先知，latency 好得反常。凡是好得反常的数字，先查 `available_at`，再庆祝。

## 16. 逐 case 分析

每个会话提供可播放时间轴：

- 用户 waveform、activity 和 transcript；
- 助手 waveform、activity 和当前回复；
- reference TRP 与事件窗口；
- A/B/C 三个实验臂的 probability 和 action；
- 实际 playback start/stop；
- gap 和 overlap；
- annotator rationale；
- error taxonomy。

错误标签：

- `acoustic_pause`；
- `semantic_incomplete`；
- `backchannel_confusion`；
- `barge_in_miss`；
- `side_speech`；
- `calibration`；
- `hysteresis`；
- `annotation_ambiguity`。

强制逐个看：

- A 错 C 对 20 例；
- C 错 A 对 20 例；
- annotator 分歧全部；
- false cut 全部；
- backchannel false stop 全部；
- side-speech false stop 全部；
- p95 latency 最差 20 例。

自动指标筛出可疑样例，判断还得靠人把时间轴放出来听。对话时序的错误往往只有几百毫秒，光看数字表格感受不到"它抢我话了"有多突兀。

## 17. 交付物

1. 双声道 manifest 与标注指南；
2. event evaluator；
3. 规则 baseline 配置；
4. learned endpoint checkpoint；
5. four-action TurnHead；
6. calibration/reliability 报告；
7. runtime hysteresis state machine；
8. 三臂固定回放结果；
9. 逐 case 时间轴；
10. privacy/license 审计；
11. `turn-policy-v1`。

## 18. 复现清单

- [ ] session/speaker split；
- [ ] channel sync error；
- [ ] source/license/consent；
- [ ] annotation version/annotators；
- [ ] feature checkpoint/hash；
- [ ] feature backend mode 与 shared-prefix leakage test；
- [ ] frame hop/available_at；
- [ ] class counts；
- [ ] seed；
- [ ] threshold/calibration set；
- [ ] action cost matrix；
- [ ] response audio/hash；
- [ ] replay packet/jitter；
- [ ] FDB version；
- [ ] 不可分发音频未进入产物。

## 19. 前沿对照与改造方向

本课的思路——外挂一个策略头判断该不该开口——是一条务实路线，前沿还有一条更激进的路线：让"何时开口"从建模方式里自然长出来。[Moshi](https://arxiv.org/abs/2410.00037)（第 01 课已读）把用户和助手两条音频流放在同一时间轴并行建模，模型自己的流里"沉默"也是正常生成内容，于是不存在一个显式的接话开关——turn-taking 行为直接由双流预测产生。[dGSLM](https://arxiv.org/abs/2203.16502) 更早展示了这条路：在 Fisher 双声道上做 two-tower 生成式建模，不用文本监督也能产生自然的轮换与重叠。而 [VAP](https://arxiv.org/abs/2205.09812) 代表标签效率的另一极：完全不用人工事件标注，只靠预测未来双方 activity，就能 zero-shot 判断 turn-shift 和 backchannel 倾向；[后续工作](https://arxiv.org/abs/2401.04868)证明它能在 CPU 上实时连续运行。评测侧，[Full-Duplex-Bench](https://arxiv.org/abs/2503.04721) 及其 [v1.5](https://arxiv.org/abs/2507.23159) 已把 pause、backchannel、turn-taking、interruption 变成对已发布系统的标准化行为测试。

数据规模是一方面：standard 档 20–100 小时对话，前沿系统的对话数据多几个量级，稀有事件覆盖完全不同——这是钱能解决的。机制差距有两条：其一，我们的 TurnHead 是冻结特征上的外挂头，决策和生成割裂，Moshi 式双流建模里两者是一体的——第 07 课往这个方向走一步；其二，我们的语言证据来自冻结 Thinker state 的 readout，底座没有被训练过"为了接话去理解"，语义完整性判断的上限受制于底座。但外挂路线也有真优势：策略可单独校准、单独回滚、单独审计，出了抢话事故能定位到具体阈值和概率，这在端到端双流系统里反而难做到。



1. **给 TurnHead 加韵律特征。** Step 4 的 80ms 特征组合里加入 f0（基频，即音高轨迹）和 energy：句尾常有音高下降或上扬，是 TRP 的经典声学线索。改动位置：Step 4 的特征抽取脚本加两维时序特征，`TurnHead.fuse` 输入维度相应加宽；模型和数据其余部分不动。预算：重抽 feature cache 约 2–4 GPU-hour，B/C 各 3 seed 重训（头很小）约 6 GPU-hour。预期：incomplete-utterance 桶的 false cut 下降，Step 10 中"pause 长度"反事实的动作翻转点后移。失败判定：各桶指标与不加韵律时无差异，说明 Silero 概率与冻结声学状态已隐含这些信息，记录后撤销改动。
2. **VAP-only 零标签臂。** 只用 `vap_head` 的未来 activity 预测加一套固定映射规则（如"预测未来 1 秒对方持续安静则 TAKE"）构成第四臂 D，不用任何四动作人工标签。改动位置：训练脚本关掉 action loss，运行时加一个规则映射层；评测复用 Step 2 的 matcher。预算：复用 frozen features，训练约 2 GPU-hour。预期：D 的 TAKE_TURN 接近 B，但 backchannel 与 side-speech 区分明显差于 C——VAP 看不见语义，这正好实证 20.2 阅读问题的答案。失败判定：若 D 全面追平 C，说明本课的标注投入没换来信息增量，回头审查标注质量和特征选择。
3. **response-conditioned TurnHead。** 按 [Response-conditioned Turn-taking Prediction](https://arxiv.org/abs/2305.02036) 的思路，把"系统准备说什么"（response candidate 的文本嵌入或长度估计）作为 TurnHead 的第五路输入：准备说短确认和准备说长解释，最佳接话时机不同。改动位置：第 7 节 `TurnHead.forward` 增加一路输入，Step 4 缓存中加入 response plan state。预算：重训 C 臂 3 seed 约 6 GPU-hour。预期：gap 分布与真人 reference 的 Wasserstein distance 缩小。失败判定：事件 F1 下降或 gap 分布不变，说明缩小版的 response 表示太弱，记录为负结果。

TurnGPT 的核心结论——语言完整性信号显著优于纯静音规则——在本课设置里直接对应 B 臂对 A 臂：取 5.3 节那类"静音等长、语义完成度不同"的配对 case，预期 B 在 incomplete-utterance 桶的 false cut 明显低于 A，方向与论文一致；若 B 不赢，先查语言特征是否真的进了模型（打印 dialogue_state 的方差）。VAP 论文"无标签也能学到 turn 时序"的结论对应上面的 D 臂实验，预期能复现同方向趋势，但绝对数字不必对齐——底座和数据域都不同。

**更多顺手扩展：**

- 把 response candidate 作为 TurnHead 条件（改造清单第 3 项的完整版）；
- multilingual turn policy——不同语言的停顿习惯和填充词不同；
- 预测连续未来 activity 而非四类硬标签；
- 成本敏感 policy learning，直接按 cost matrix 优化而非事后选阈值；
- 在线用户个性化 gap/backchannel 频率——有人喜欢抢拍，有人喜欢留白；
- prosody/f0/energy feature（改造清单第 1 项）；
- speaker verification 过滤旁人；
- 将 `turn-policy-v1` 接入[第 07 课](07_full_duplex_routing.md)，但保持策略和内容融合模块可单独关闭。

## 20. 论文与必读材料

### 20.1 语言驱动 turn-taking

- [TurnGPT](https://arxiv.org/abs/2010.10874)：
  读 task formulation、turn-shift token、context ablation。带着问题进去：5.3 节那两句话，静音一样长，决策为什么应该不同？TurnGPT 用什么证据证明语言模型能提供这个区分信号，它的 context ablation 里上下文砍到什么程度这个信号会消失？阅读后写出：语法和语用完整性为纯静音规则补充的信号，并对照你 Step 10 的"语义完成度"反事实结果。
- [Response-conditioned Turn-taking Prediction](https://arxiv.org/abs/2305.02036)：
  读 response-conditioned formulation 与 ambiguous scenarios。带着问题：同一个用户停顿，系统准备说"好的"和准备说一段长解释，接话时机应该一样吗？阅读后写出：response condition 改变接话时机的机制，以及它对应本课 TurnHead 四类输入中的哪一路。

### 20.2 Voice Activity Projection

- [Voice Activity Projection](https://arxiv.org/abs/2205.09812)：
  读 self-supervised objective、future activity bins、四个 zero-shot task。带着问题：完全不给人工标签，VAP 能从 stereo audio 里免费学到什么、永远学不到什么？阅读后写出：没有人工 event 标签时 VAP 能学习和不能学习的目标——这个答案应该能预言第 19 节 D 臂在哪些指标上输给 C。
- [Real-time and Continuous Turn-taking Prediction using VAP](https://arxiv.org/abs/2401.04868)：
  读实时部署、context length 与 CPU 性能。带着问题：一个每 80ms 决策一次的头，连续跑一小时需要维护什么状态、丢弃什么历史？阅读后写出：持续运行所需的 state、context length 和计算预算，对照你 Step 8 runtime 的实现。
- [VAP 官方仓库](https://github.com/ErikEkstedt/VoiceActivityProjection)：
  看 stereo input、event extraction 和 evaluation。对照你 Step 2 的 matcher：它的 event 定义和容忍窗口跟你的差在哪，哪些约定值得直接抄。

### 20.3 双流对话与评测

- [Generative Spoken Dialogue Language Modeling / dGSLM](https://arxiv.org/abs/2203.16502)：
  读 two-tower/cross-attention、Fisher 双声道和 turn-taking evaluation。带着问题：8.4 节坚持"保留 stereo，不先混成 mono"，到底护住了什么？阅读后写出：混为 mono 后丢失的说话人和重叠信息，并解释为什么 backchannel 检测在 mono 数据上几乎不可能做对。
- [Full-Duplex-Bench](https://arxiv.org/abs/2503.04721)：
  读 pause、backchannel、turn-taking、interruption 四类任务与自动指标。带着问题：它评的是"什么时候说"还是"说得对不对"？阅读后写出：行为指标和回答内容指标需要分开报告的原因——这正是本课 Step 9 固定内容生成器的理由。
- [Full-Duplex-Bench v1.5](https://arxiv.org/abs/2507.23159)：
  重点看 user interruption、listener backchannel、side conversation、ambient speech 的区分。对照你 8.3 节的标注指南和第 16 节的错误标签：它的场景分类里，哪几类是你的 pilot 数据还没覆盖的。

读完材料回头看：系统现在每 80ms 做一次有依据的决策，知道你是在停顿还是说完了，知道"嗯"不是打断。但它做完决策只有两招——开始说，或者停下；听到有价值的插话，也只能先闭嘴再从头想。[第 07 课](07_full_duplex_routing.md)解决这最后一块：生成期间继续把用户语音送进模型的语义状态，让系统能 Continue、Pause、Replan——真双工调度。本课的 `turn-policy-v1` 会作为那一课的事件触发器接入，而且要保持可单独关闭，方便归因。
