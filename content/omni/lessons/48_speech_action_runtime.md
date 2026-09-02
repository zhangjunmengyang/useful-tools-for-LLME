---
id: 48_speech_action_runtime
title: "双时钟状态表"
summary: "第 19 课 Thinker–Talker 加上第 29 课快慢 VLA 之后，CONTINUE / PAUSE / REPLAN 在音频时钟和控制时钟上能否共用一行状态，却必须分列记录两个 available_at？"
unit: embodied-omni
play_tools: []
checkpoints:
  - "写出一行状态、两列时间戳：audio_available_at 与 action_available_at，并分别用音频帧和 H/f 定义过期。"
  - "把第 07 课的 CONTINUE / PAUSE / REPLAN 接到控制时钟上，同时说明音频 PAUSE 不等于第 40 课的力切断。"
  - "规定 REPLAN 同时取消未播 PCM 和未执行 chunk，且不得 undo 已播放音频或已发生接触。"
  - "在双时钟教具里用两次独立事件完成停嘴和重规划，并在 CPU 回放里证明旧 PCM 与旧剩余步不再执行。"
---

# 第 48 课：把语音打断和手臂重规划接到一张状态表

> 类型：embodied-omni 研究级，接口说明书与教学状态机<br>
> 建议周期：阅读约 80 分钟；CPU 实验数分钟；浏览器 Lab 一次验收约 12 分钟<br>
> 硬件：无 GPU 可完成本课阅读、教学模拟与 CPU 机制实验。对照 Qwen2.5-Omni、GR00T N1 或 π0 的公开权重需要各论文自己的推理卡，本课不要求加载这些权重<br>
> 产物：一行状态、两列时间戳的接口说明书，加上可回放的教学状态机。不是发布级双工机器人，不声称复现 GPT-4o 或 Helix<br>
> 独立性：需要[第 07 课](07_full_duplex_routing.md)的 `CONTINUE` / `PAUSE` / `REPLAN`，[第 19 课](19_capstone_thinker_talker.md)的 Thinker–Talker 边界，[第 29 课](29_dual_system_vla.md)的快慢时钟，[第 30 课](30_closed_loop_control.md)的 $H/f$。第 40 课文件若尚未写入仓库，按本课给出的力切断规格对照即可。无 GPU 时不要根据 CPU 数字报告真机成功率

## 1. 语音打断和手臂重规划能否共用一张状态表

助手还在说“我去拿桌上的杯子”，手指已经按着动作块往前走。有人把杯子挪到桌子另一头。嘴要不要停，手臂要不要改计划，看起来像同一件事：世界变了，旧句子和旧关节都不该继续。若你只做一颗“全部停下”的按钮，系统会把两件物理上不可回放的历史一起抹掉：已经从扬声器出去的 PCM（可直接播放的原始音频，[第 01 课](01_baseline_reproduction.md)讲过），以及夹爪已经碰上杯壁的那一次接触。听众听见了半句，桌面上的杯子已经歪了，日志里却写成什么都没发生。

[第 07 课](07_full_duplex_routing.md)在音频时钟上把控制头收成三个动词：`CONTINUE`（继续说）、`PAUSE`（停嘴但保留状态）、`REPLAN`（废旧计划另开 branch）。[第 30 课](30_closed_loop_control.md)把同一组动词接到控制时钟：未执行的 chunk 步对应未播放的 PCM。两边的状态机同构，时钟不同。[第 19 课](19_capstone_thinker_talker.md)的 Thinker–Talker 把慢理解和快发声拆开；[第 29 课](29_dual_system_vla.md)的 System 2 / System 1 把慢规划和快控臂拆开。本课问的是下一层协议：这两条慢快链条，能否写进同一张状态表。

答案分两句。第一句：能共用一行。一次会话只有一个 `branch_id`、一份血缘、一套 `CONTINUE` / `PAUSE` / `REPLAN` 动词，否则“我说去拿杯、手却去拿盘”会对不上账。第二句：不能共用一列时间戳。音频事件的 `available_at` 卡的是一块语音加编码延迟；动作事件的 `available_at` 卡的是观察锁存加推理延迟 $d$。过期定义也必须分开：音频用帧或块的时长，手臂用开环窗口 $H/f$。把 200 ms 的动作延迟拿去和 320 ms 的音频块比大小，过期判定会指错模块。

本课改的是运行时接口，不是再训一个会说话的人形。要验证的结果很具体：一张表上同时出现 `audio_available_at_ms` 和 `action_available_at_ms`；语音 `PAUSE` 与手臂 `REPLAN` 是两次事件；`REPLAN` 之后旧 PCM 与旧剩余步都不执行；一次点击不得同时撤回已播放音频和已发生的接触。浏览器 Lab 和 CPU 实验都是教学夹具。它们证明协议可复核，不能写成“我们复现了 GPT-4o 的双工”或“我们复现了 Figure Helix”。GPT-4o 的内部调度其技术报告未公开到可引用的状态表。Helix 的官方博文给出了 System 2 为 7–9 Hz、System 1 为 200 Hz，没有给出本课这种两列 `available_at` 的事件日志。

本课术语：

| 术语 | 简要解释 |
|---|---|
| 音频时钟 | 语音块或 codec 帧的滴答。MiniMind listener 按 320 ms 一块；Moshi 的 Mimi 一帧 80 ms |
| 控制时钟 | 向低层跟踪器下发关节目标的滴答，周期 $1/f$ |
| `audio_available_at` | 音频事件最早能被模型使用的墙钟，通常是 `block_end + encode_latency` |
| `action_available_at` | 动作块最早可执行的墙钟，通常是 $t_{\mathrm{obs}}+d$ |
| 未播 PCM | 已解码、尚未送给扬声器的音频帧；`PAUSE` 冻结出队，`REPLAN` 取消 |
| 剩余步 | 当前 chunk 里尚未 `executed_at` 的关节目标；对应未播 PCM |
| `SAFE_HOLD` | 力超限后的保持姿态。丢弃剩余步，不撤回已经发生的接触 |
| branch | 一次计划尝试的存档。`REPLAN` 把旧 branch 标 `superseded` 并开新行 |
| 接触 | 夹爪或指尖已经对物体或桌面施加过力。物理上不可回放 |

## 2. 本课解决的问题

到第 30 课为止，课程里有两张几乎同构、却从未强制并表的状态机。音频那边，[第 07 课](07_full_duplex_routing.md)要求输入在输出期间继续推进，`PAUSE` 保留 pending PCM，`REPLAN` 取消未播帧。[第 19 课](19_capstone_thinker_talker.md)把这套动词接到 Thinker–Talker：Thinker 可以还在写，Talker 和播放队列必须能被单独停。手臂那边，[第 29 课](29_dual_system_vla.md)要求控制环每步消费当前子目标，规划环不是每步都跑；[第 30 课](30_closed_loop_control.md)要求 $d<H/f$，否则过期 chunk 整块丢掉。两边各自能跑。接到同一个会说话的身体上，就会出现三种假合并。

第一种假合并：共用一个 `available_at`。调度器把用户那句“别拿那个”的音频时间和新图像锁存时间写进同一格。320 ms 的语音块去跟 $H/f=0.40$ s 的开环窗口比，看起来都是“大约三分之一秒”，过期边界会在整除误差里跳来跳去。

第二种假合并：共用一个按钮。产品经理把停嘴、停臂、撤回已说的话、撤回已碰上的杯子画成同一个“打断”。已播出的 PCM 在空气里，已发生的接触在物体上。调度器没有权限假装它们没发生。第 40 课的规格把这件事写死：在 chunk 步 $i$ 若 $\|F_i\|>F_{\max}$，只丢弃 $i+1$ 以后的剩余步，进入 `SAFE_HOLD`。力切断不是把杯子倒回壶里。

第三种假合并：`REPLAN` 只取消一边。嘴已经改口“杯子在右边”，手臂仍走左边那块的剩余步；或者手臂已经改去右边，扬声器还在播“我拿左边这个”。[第 07 课](07_full_duplex_routing.md)把嘴上的这种错误叫 context corruption。[第 30 课](30_closed_loop_control.md)把手上的这种错误叫 stale-plan execution。本课把它们收成同一条断言：`REPLAN` 时刻之后，旧 `branch_id` 的未播 PCM 和未执行剩余步都必须为零。

本课因此固定五个可证伪命题。它们是协议命题，不是模型命题。协议命题可以用 CPU 回放证伪；模型命题需要权重、数据和真机，本课不做。

1. 每一行状态同时拥有 `audio_available_at_ms` 和 `action_available_at_ms`，评测时各自满足 `consumed_at >= available_at`。
2. 同一延迟 $d$ 可以只让一边过期：音频过期用块长 $T_{\mathrm{frame}}$，动作过期用 $H/f$。
3. 通道为 `audio` 的 `PAUSE` 不丢弃手臂剩余步，也不进入 `SAFE_HOLD`。
4. `REPLAN` 将旧 branch 标为 `superseded`，旧 PCM 与旧剩余步都不得再执行。
5. 任何单次点击若把已播 PCM 计数减回去，或把已经为真的接触标志改回假，该路径记为非法 undo，验收失败。

五条都通过，只能说明接口被写成了可回放的状态机。它们不能说明 Qwen2.5-Omni 的流式 Talker 在你的麦克风上延迟多少，也不能说明 Helix 在厨房里能否把没见过的杯子放进抽屉。把五条抄进实验记录时，每条后面跟一个本课夹具数字：两列字段名、320 对 400、180 ms 的音频 `PAUSE` 后手臂仍执行、360 ms 后旧未来为零、非法 undo 才会把已播清零。数字和命题绑在一起，别人才能复核。只交五句口号，等于没做第 48 课。

出现以下任一情况时，不得把结果标成可用的双工身体运行时：

- 只有一张“当前动作”枚举，没有两列时间戳；
- 用 token 下标或控制步下标冒充墙钟；
- `PAUSE` 之后手臂剩余步被一并清空，却把原因写成“用户打断了语音”；
- `REPLAN` 之后执行表或播放表里仍出现旧 `branch_id` 的未来项；
- 日志里已播 PCM 或接触标志在 undo 之后变小或变假；
- 把 GPT-4o 演示视频或 Helix 博文视频的流畅程度，换算成本课 freshness 成立。

把这张清单和前几课的失败清单对齐，避免“本课新发明一套红线”。[第 07 课](07_full_duplex_routing.md)禁止假双工：只收包不消费、能停不能懂、replan 后从零重来。[第 19 课](19_capstone_thinker_talker.md)禁止把回放 TTFA 写成在线 TTFA。[第 29 课](29_dual_system_vla.md)禁止把 System 1 暂停后自己发明下一阶段写成成功。[第 30 课](30_closed_loop_control.md)禁止把过期块抓空写成视觉失败。本课往这四条后面加第五条：禁止把两列时钟焊成一颗撤销按钮。五条同时成立，嘴和手才算接到了同一张可审计的表上。缺第五条时，前四条可以各自通过，联合演示仍会在“说话时挪杯”这一下翻掉。

联合失败还有一种更隐蔽的写法：日志里两列都在，按钮也是两个，内部却在同一条 `if interrupted` 里先清 PCM 再清剩余步再把接触改假。表面满足“两次点击”，实际一次点击的处理函数做了三件非法事。CPU 实验因此要求事件带 `channel`，并且 `PAUSE` / audio 的处理函数不得调用 `drop_remaining`。Lab 则要求两次点击的时间戳或序号不同。只有点击次数、没有通道字段，仍算焊死。

## 3. 开始前需要准备什么

本课没有 MiniMind-O 训练步骤，也不接真机。开始前把上游事实和本课约定分开写进实验记录。

**上游事实（打开过的页面，不是口口相传）：**

- [第 07 课](07_full_duplex_routing.md)：`available_at = block_end + encode_latency`；`PAUSE` 保留 pending PCM；`REPLAN` 取消未播 PCM 并把旧 branch 标 `superseded`。Moshi 摘要报告理论延迟 160 ms、实测约 200 ms，Mimi 一帧 80 ms。
- [第 19 课](19_capstone_thinker_talker.md)：默认验收是双工回放（真 listener、真 Talker，Thinker 事件按时间戳回放），不自动等于在线真双工。课文写明只验证 GPT-4o 类系统中的部分公开机制，不声称复刻完整能力。
- [第 29 课](29_dual_system_vla.md)：GR00T N1 在 L40 上 System 2 约 10 Hz、System 1 约 120 Hz，块长 $H=16$，4 步去噪 63.9 ms。$\pi_{0.5}$ 先采样子任务再出动作块，控制 50 Hz。
- [第 30 课](30_closed_loop_control.md)：开环窗口 $T_{\mathrm{open}}=H/f$，过期判定 `delay_ms >= (H * 1000) // f`。ACT 的 ALOHA 为 50 Hz；π0 取 $H=50$，声明最高 50 Hz。
- Qwen2.5-Omni 技术报告 [arXiv:2503.20215](https://arxiv.org/abs/2503.20215)：Thinker 出文本，Talker 直接消费 Thinker 隐表征和离散文本 token；音频 16 kHz，128 维 mel，窗 25 ms、跳 10 ms，编码器一帧大约对应 40 ms 原声；TMRoPE 把时间、高、宽拆开，音视频按 2 s 一块交错；音频编码器改为 2 s 块注意力以支持预填充；波形侧用流匹配 DiT，感受野限制为 4 块（回看 2、前瞻 1），再经修改过的 BigVGAN 还原。报告给出 seed-tts-eval 的 WER 为 1.42% / 2.33% / 6.54%（test-zh / test-en / test-hard）。该报告未给出名为 `CONTINUE` / `PAUSE` / `REPLAN` 的控制头状态表，也未给出手臂列。WER 不得填进本课 Gate。
- Figure 官方博文 [Helix](https://www.figure.ai/news/helix)（2025-02-20）：System 2 为机载、互联网预训练的 7B VLM，7–9 Hz；System 1 为 80M 的视觉运动 Transformer，200 Hz；约 500 小时遥操作；35 自由度上身；训练时在 S1 与 S2 输入之间加入与部署延迟相匹配的时间偏移。博文未公布两列 `available_at`，也未公布力超限后的 `SAFE_HOLD` 表。
- SafeVLA [arXiv:2503.03480](https://arxiv.org/abs/2503.03480)：用约束马尔可夫决策过程（CMDP：带安全代价约束的强化学习设定）做训练期对齐，相对对照方法把安全违规累积代价降 83.58%，任务成功率 +3.85%。这是训练目标，不是本课的运行时切断表。

**本课约定：**

- 符号：音频块 $T_{\mathrm{frame}}$（CPU 默认 320 ms），音频帧 80 ms，控制频率 $f=20$ Hz，块长 $H=8$，于是 $H/f=400$ ms。接触发生在旧计划的 `step_index=3`。
- CPU 实验文件：`experiments/src/learn_omni_experiments/lessons/lesson_48.py`。编排者登记进 `registry.py` 之前，用模块路径直接调用 `run()`；登记之后用仓库脚本跑 48。
- 浏览器 Lab：`Lesson48DuplexBodyLab`。标有“教学模拟”，杯子位置由你拖动，不是视觉模型输出。
- 第 40 课若尚未成文，力切断仍按规格执行：$\|F_i\|>F_{\max}$ 时丢弃 $i+1$ 以后的剩余步，进入 `SAFE_HOLD`，接触标志保持为真。
- 不把 N1 的仿真表、Helix 博文的“拿起任何小物体”、Qwen2.5-Omni 的 seed-tts-eval WER，横着写成“本课复现结果”。

需要会的前置技能：Python 字典与整数毫秒；第 07 课的 branch 血缘；第 30 课的开环窗口。不必重做 channel fusion、AEC 或 DiT。若还没读第 19 课，记住一句即可：Talker 在 Thinker 暂停时仍可能出声，出声条件是冻结的。若还没读第 29 课，记住一句即可：System 1 在 System 2 暂停时仍会出动作，动作条件是最后一条子目标。

建议的阅读环境：第 07、19、29、30 课各一个标签页，外加 Qwen2.5-Omni HTML、Helix 博文、SafeVLA 摘要。抄数字时写“页或节”。CPU 实验不访问网络。Lab 允许放慢。无 GPU 的读者到 Step 5 为止即完成本课；Step 6 是可选对照，缺卡不算失败。

硬件数字再抄一次，以免和“最低能读完”混在一起。读完和跑 CPU：笔记本电脑即可。跑 Lab：现代浏览器，能拖动杯子。对照 Qwen2.5-Omni 推理：按其技术报告的部署说明，本课不指定卡型，因为我们不跑它。对照 GR00T N1 后训练：NVIDIA 博文写 1×RTX A6000 或 1×GeForce RTX 4090，预训练约 50,000 H100 GPU hours，那是第 29 课的规模标定，本课不布置。对照 Helix：官方写双嵌入式低功耗 GPU 机载运行，没有给型号与功耗数字到可复现的表格，其技术报告未公开该细节。有人把 Helix 视频里手臂的平滑程度换算成 200 Hz 在你机器上可达到，属于把演示帧率当成控制频率。笔记里要拆开：视频帧率、控制频率、生成器内部迭代频率。本课只使用第二项来算 $H/f$。

## 4. 完成后应具备的能力

一个可用的自检办法：把别人的“会说话的机器人”架构图遮住产品名，只留箭头和频率。你应能判断哪条箭头是音频时钟，哪条是控制时钟，中间那一行状态有没有两列时间戳，以及停嘴、停臂、废计划、力切断是不是四个不同的格子。缺两列时间戳时，演示可以很流畅，审计过不了。

完成后，拿到任意“语音加手臂”的实现或论文图，应能做以下检查：

1. 标出音频时钟和控制时钟，写出两个 `available_at` 的定义，拒绝把它们存进同一个字段。
2. 写出两套过期公式，并能举一个“同一毫秒延迟只让一边过期”的数字例子。CPU 夹具里 320 ms 让音频过期、400 ms 的 $H/f$ 仍让动作 fresh。
3. 把 `CONTINUE` / `PAUSE` / `REPLAN` 填进 2（通道）乘 3（动词）的表，指出哪一格会改 PCM 队列，哪一格会改剩余步。
4. 说明音频 `PAUSE` 为什么不等于 `SAFE_HOLD`：前者不停臂，后者不自动停嘴，两者都不撤回接触。
5. 写出 `REPLAN` 的未来集合：未播 PCM 并未执行剩余步；并写出它的禁止集合：已播 PCM、已执行步、接触标志。
6. 对照第 19 课：Thinker 隐表征到 Talker 的桥，相当于 System 2 条件到 System 1 的桥。对照时只借用“快模块每步读冻结条件”，不重做 adapter。
7. 对照 Helix 博文：7–9 Hz 与 200 Hz 是两个时钟，不是一张已经写好的状态表。其技术报告未公开该表。
8. 用本课 CPU 实验证明：两列时间戳存在；`PAUSE` 与 `REPLAN` 分通道；旧 PCM 与旧剩余步在 `REPLAN` 后为零；非法 undo 才会撤回历史。
9. 向同事口述 Lab 的验收：先预测，再说话，拖走杯子，点两次按钮。一次撤销全部必须失败。说不清“两次事件、不撤回历史”，等于没学过并表。
10. 写交付物时能把“接口说明书 + 教学状态机”和“发布级双工机器人”分成两句。前一句是本课通过条件，后一句是本课明确不做的事。

把第 10 条再拆成可对外说的三句限制，避免验收会上被追问时改口。第一句：本课证明协议，不证明识别。杯子是拖动的，插话是按钮，没有 ASR 也没有物体检测。第二句：本课证明回放，不证明在线。Thinker 和 System 2 都可以按时间戳注入，不必在你的笔记本上跑 7B。第三句：本课证明历史不可撤回，不证明下一次抓取会成功。`REPLAN` 之后新块可能仍然抓空，那是 $d$、$H$、$f$ 的事，交给第 30 课的不等式，不要在本课用抓住当 Gate。Lab 的 Gate 亮着只表示两次事件、零泄漏、历史单调。夹爪有没有套住新杯位置，只是画面上的旁证。

若你只能记住一张表，记住下面这张最小自检。遮住实现，只问四格：

| 问 | 应能立刻回答 |
|---|---|
| 现在是哪一列的事件 | `audio` 或 `action`，禁止“通用打断” |
| 这块还 fresh 吗 | 音频比 $T_{\mathrm{frame}}$，动作比 $H/f$ |
| 未来队列清了吗 | 未播 PCM 与剩余步在 `REPLAN` 后为空 |
| 历史被改了吗 | 已播计数和接触不得变小 |

四格都答得出，第 4 节才算过。答得出论文频率、答不出这四格，仍算没过。

## 5. 原理：边造边讲

难点不在多线程，在承诺的范围：新语音何时可用，新观察何时可用，旧声音还剩多少没播，旧动作还剩多少没走，哪些已经进入空气或进入接触、调度器无权改写。下面按同一节奏展开：为什么需要、怎么运转、精确定义、在哪核对、怎么证明做对了。

### 5.1 一张表，两列时钟

语言模型里位置是下标。扬声器和电机上位置是墙钟。50 Hz 的手臂意味着每 20 ms 必须给出一个关节目标；12.5 Hz 的 Moshi 意味着每 80 ms 必须给出一帧 codec。这两个数不能约成一个“实时”。约成一个数以后，你就不知道该用哪一个去判断过期。

把一次会话写成一行，是为了让嘴和手承认彼此在执行同一份计划。`branch_id=1` 的句子是“我去拿左边的杯子”，同一行的剩余步也必须是朝左边走。`REPLAN` 之后 `branch_id=2` 的句子和剩余步一起换成右边。血缘仍用 `parent_id` 串起来，和[第 07 课](07_full_duplex_routing.md)的 `ResponseBranch` 一样，只是字段变多了。

写成两列，是为了让滴答互不冒充。音频列推进的是 listener 块、Talker 帧、播放光标。动作列推进的是 $t_{\mathrm{obs}}$、chunk、`step_index`。墙钟可以相同，字段必须不同。评测脚本如果只看见一个 `available_at_ms`，直接失败。

类比：一张双人餐桌上放两只钟，一只对准说话的换气，一只对准伸手的节拍。类比失效处：两只钟可以对到同一秒，但“这一秒该不该出声”和“这一秒该不该合爪”仍是两个谓词。你不能因为两只钟指向 12:00:00.200，就把语音块判过期或把动作块判 fresh。

CPU 实验把这一行写成字典。每次控制事件追加一张快照，键里必须同时有 `audio_available_at_ms` 和 `action_available_at_ms`。缺一列，`state_table_has_two_timestamp_columns` 为假。

把第 07、19、29、30 课已经出现过的对象，按列登记一次，避免后面各小节再发明一套名字。

| 本课字段 | 第 07 / 19 课对应物 | 第 29 / 30 课对应物 | 允许为空吗 |
|---|---|---|---|
| `branch_id` | `ResponseBranch.branch_id` | 动作计划编号 | 否 |
| `parent_id` | `parent_id` | 上一次 `REPLAN` 的计划 | 根节点可以 |
| `audio_available_at_ms` | 语音块 `available_at` | 无 | 否，音频事件必填 |
| `action_available_at_ms` | 无 | chunk 的 `available_at` | 否，动作事件必填 |
| `audio_mode` | 播放是否出队 | 无 | 否 |
| `action_mode` | 无 | `GENERATING` / `PAUSED` | 否 |
| `pending_pcm` | 可撤销 PCM 缓冲 | 无 | 可以暂时为空 |
| `remaining_steps` | 无 | 未 `executed_at` 的 `step_index` | 可以暂时为空 |
| `contact_occurred` | 无 | 第 40 课规格的接触位 | 否，默认假 |
| `force_n` | 无 | 当前步力或力矩标量 | 无传感器时写 -1 |

“允许为空”指队列当前没有元素，不指字段从 schema 里消失。评测脚本对缺失键直接失败，对空列表放过。`force_n=-1` 表示本回合没有力通道，此时不得发出 `FORCE_CUTOFF`；发出了就属于用空传感器伪造切断。

[第 19 课](19_capstone_thinker_talker.md)的四条时钟（listener 帧、Thinker 出字、Talker 出 codec、播放器出 PCM）全部属于本课的音频列。它们内部仍要排序，本课不重排第 19 课的四段延迟，只要求这四段的结果写进 `audio_available_at_ms`，不要写进动作列。[第 29 课](29_dual_system_vla.md)的 $\Delta T_2$ 若你暂时不实现第三列，就把它折叠进动作列的 $d$：子目标晚到的时间加进 `action_available_at_ms`。折叠必须在笔记里声明。不声明就把规划延迟藏进了“模型变慢”，后面所有 $H/f$ 对比都会被污染。

### 5.2 两个 `available_at`

[第 07 课](07_full_duplex_routing.md)已经强调：发生、可用、被用是三个时刻。本课只是强迫你把“可用”写两遍。

音频列：

$$
t^{\mathrm{audio}}_{\mathrm{avail}}=t_{\mathrm{block\_end}}+d_{\mathrm{enc}}
$$

$t_{\mathrm{block\_end}}$ 是当前语音块收完的墙钟，$d_{\mathrm{enc}}$ 是编码延迟。MiniMind 的 listener 按 320 ms 一块时，$t_{\mathrm{block\_end}}$ 只能落在 320 的倍数附近。Moshi 按 80 ms 一帧时，数字换成帧尾。Qwen2.5-Omni 把音频时间 ID 对齐到 40 ms，那是位置编码的粒度，不是本课的 `available_at`；不要把 40 ms 抄进过期公式冒充块长。

动作列：

$$
t^{\mathrm{action}}_{\mathrm{avail}}=t_{\mathrm{obs}}+d
$$

$t_{\mathrm{obs}}$ 是锁存图像与本体感觉的时刻，$d$ 含视觉编码、规划或动作专家前向、去噪或积分。GR00T N1 在 L40、bf16、$K=4$ 时测到 16 步 63.9 ms，那是他们设定下的 $d$ 的一部分，不是你的 $d$。π0 用观察前缀的 KV 缓存做 10 步欧拉积分，是在把 $d$ 往窗口里压。Helix 博文写训练时给 S1 / S2 加时间偏移，使部署延迟出现在训练分布里。这三句话都在说 $d$ 必须被测量，没有说可以和音频列共用。

消费规则两边相同，比较对象不同：

$$
t^{\mathrm{audio}}_{\mathrm{cons}}\ge t^{\mathrm{audio}}_{\mathrm{avail}},\qquad
t^{\mathrm{action}}_{\mathrm{cons}}\ge t^{\mathrm{action}}_{\mathrm{avail}}
$$

CPU 实验对每条控制事件检查 `consumed_at_ms >= available_at_ms`。事件自己带着通道：`audio` 事件的 `available_at` 只和音频列比，`action` 事件只和动作列比。实现里不要写 `if event.time < min(audio_avail, action_avail)` 这种横着取最小的捷径。最小的那个时钟会把另一个时钟的未就绪事件提前放行。

整数毫秒仍然必要。音频块 320、编码 40、动作延迟 100、控制周期 50，全部能被 10 ms 滴答整除。教具若用浮点秒再显示两位小数，应先做整除再除以 1000，避免 0.32 与 0.320 在边界上吵。

混用两列时间戳的错账至少有三种，写进实验记录当反例。

第一种：用音频块去判动作。$d=350$ ms，$H/f=400$ ms，动作块 fresh；若误用 320 ms 当窗口，350 被判 stale，手臂在杯子仍可追及时停掉。失败看起来像模型不会抓，根因是窗口抄错列。

第二种：用 $H/f$ 去判音频。用户 200 ms 的纠正语已经编码完，窗口若被写成 400 ms，控制头会继续说完旧句后半段。第 07 课的停止延迟超标，CCR 上升，日志却显示“动作列 freshness 良好”。

第三种：取两列最小值当统一门槛。$\min(320,400)=320$。动作在 350 ms 时被错杀，音频在 200 ms 时被放过。表面上“保守”，实际两边都错。CPU 禁止写 `min` 或 `max` 去合并窗口。评测见到单一 `stale` 布尔量且没有通道前缀，直接失败。

Helix 博文的训练时间偏移可以当成“承认两列延迟不同”的旁证：他们在 S1 与 S2 输入之间加偏移，使部署时 S2 更慢这件事出现在训练分布里。旁证不是实现。本课仍然要求推理日志里两列分开写，而不是把偏移折进某一个隐向量就宣布对齐完成。

### 5.3 过期：音频帧与 $H/f$

过期不是“等太久了”的印象，是“这块描述的时间已经结束”。音频和动作结束的方式不同。

音频。一块 320 ms 的用户语音，若编码加排队已经用掉 320 ms，块到达时它声称覆盖的区间结束了。本课把这条写成

$$
\mathrm{stale}_{audio}\iff d_{\mathrm{audio}}\ge T_{\mathrm{frame}}
$$

$T_{\mathrm{frame}}$ 在 CPU 里取 listener 块长 320 ms。若你改用 Moshi 的 80 ms 帧，公式不变，数字换成 80。恰好相等也过期，与[第 30 课](30_closed_loop_control.md)的 `>=` 同向：到达的那一拍，覆盖的最后一拍也结束了，没有可播放的未来。

动作。长度为 $H$ 的块按频率 $f$ 排开，覆盖

$$
T_{\mathrm{open}}=\frac{H}{f}
$$

过期写成

$$
\mathrm{stale}_{action}\iff d\ge\frac{H}{f}
$$

CPU 默认 $H=8$、$f=20$，窗口 400 ms。于是出现本课最有用的那组对照：

| 延迟 | 音频（320 ms） | 动作（400 ms） |
|---|---|---|
| 200 ms | fresh | fresh |
| 320 ms | stale | fresh |
| 400 ms | stale | stale |

同一毫秒数，一边该丢、一边不该丢。若你的调度器共用一个 `stale` 布尔量，320 ms 时要么错杀动作块，要么放过过期语音。CPU 检查 `audio_and_action_expiry_use_different_windows` 锁的就是这张表的中间行。

[第 29 课](29_dual_system_vla.md)还有第三种过期：子目标年龄超过 $T_{\exp}$。那是慢规划列的事。本课状态表可以再加一列 `plan_available_at_ms`，但主验收只锁音频列和动作列。三列都写的人必须继续分字段，禁止回流成一个时间戳。

把公开系统的时钟填进同一张对照表，缺的格子保持空白。空白表示“该文未给出”，不要用另一篇的数补上。

| 系统 | 音频时钟公开值 | 动作时钟公开值 | 过期怎么写 | 本课能借用什么 |
|---|---|---|---|---|
| MiniMind-O listener | 320 ms 块 | 无 | 块尾加编码延迟 | 音频列公式 |
| Moshi / Mimi | 80 ms 帧，理论 160 ms、实测约 200 ms | 无 | 帧结构，无命名 `stale` | 帧长可替换 $T_{\mathrm{frame}}$ |
| Qwen2.5-Omni | 时间 ID 40 ms，流式 DiT | 无 | 报告未给 stale 谓词 | 隐表征桥，不给并表 |
| 第 19 课 19C | Talker + 播放队列 | 无 | 回放时间戳 | 双工回放边界 |
| ACT / ALOHA | 无 | 50 Hz | 分块，无语音列 | $f$ 的例子 |
| π0 | 无 | $H=50$，最高 50 Hz | $H/f=1$ s | 动作列窗口 |
| OpenVLA-OFT ALOHA | 无 | 25 Hz，$K=25$ | 整块 1 s | 整块等于 $k=H$ |
| GR00T N1 | 无 | 10 Hz / 120 Hz，$H=16$，63.9 ms | 子目标年龄在第 29 课 | 快慢条件，不给 PCM |
| π0.5 | 无 | 50 Hz，先子任务后动作块 | 子任务过期应触发 `REPLAN` | 文本 $g$ |
| Helix 博文 | 无公开 codec 帧 | S2 7–9 Hz，S1 200 Hz | 博文未给 stale 公式 | 两个频率，不给状态表 |
| SafeVLA | 无 | 无运行时 $f$ | 训练期代价，非 $H/f$ | 不能代替切断表 |

表中“无”是公开文本里没有，不是你的实现里可以继续没有。你的教学状态机必须把空着的那一列补上，并用夹具数字填，不能把 Helix 的 200 Hz 写进这一列冒充测过。

再手算两笔本课默认账，避免后面只看表格不看毫秒。第一笔：音频帧 80 ms，编码 40 ms，播放滞后 30 ms。0 ms 发出的帧，`audio_available_at=40`，`play_at=70`。180 ms `PAUSE` 时，80 ms 发出的帧已经在 150 ms 播出，160 ms 发出的帧 `play_at=230`，此时属于 pending。`PAUSE` 之后 230 ms 到期，因为 `audio_mode=PAUSED`，它不得进入播放表，必须留在 pending，等 `CONTINUE` 或被 `REPLAN` 取消。第二笔：动作 $f=20$ Hz，周期 50 ms，$d=100$ ms，$H=8$，窗口 400 ms。0 ms 锁存，100 ms 块就绪，100、150、200、250 ms 依次执行 step 0 到 3，250 ms 接触变真。360 ms `REPLAN` 时若还剩 step 4 到 7，这四步进 `discarded`，`reason=replan`。250 ms 的接触行留在执行表。两笔账对不上，先查滴答是不是 10 ms，再查 `PAUSE` 有没有误清 `remaining`。

### 5.4 `CONTINUE` / `PAUSE` / `REPLAN` 的同构与失效

同构写在这张表里。它是本课唯一允许你“抄第 07 课”的部分。

| 动词 | 音频列 | 动作列 |
|---|---|---|
| `CONTINUE` | 附和或环境音，继续出队 PCM | 扰动可忽略，继续走剩余步 |
| `PAUSE` | 停止出队，保留 pending 与 KV | 冻结剩余步，保持最后姿态 |
| `REPLAN` | 取消未播 PCM，旧 branch 作废 | 丢弃剩余步，开新推理 |

通道必须进事件。`PAUSE` 加 `channel=audio` 不得改 `action_mode`。`REPLAN` 在本课的联合计划里会同时取消两种未来，但 Lab 仍要求你先发一次音频 `PAUSE`、再发一次动作 `REPLAN`。原因不是动词不够用，是教学上必须看见两行日志。产品若把联合 `REPLAN` 收成一个 API，内部仍要拆成两列副作用，并且禁止第三种副作用：撤回历史。

状态转移沿用第 07 课 Step 9 的骨架，只把队列对象写成一对：

| 当前 | 事件 | 下一状态 |
|---|---|---|
| 音频 `SPEAKING` | `PAUSE` / audio | 音频 `PAUSED`，pending 冻结，动作列不变 |
| 音频 `PAUSED` | `CONTINUE` / audio | 同一 branch 恢复出队 |
| 动作 `GENERATING` | `PAUSE` / action | 动作 `PAUSED`，剩余步冻结，音频列不变 |
| 任一列 active | `REPLAN` | 旧 branch `superseded`，两种未来队列清空，新 branch `active` |
| 动作 `GENERATING` | `FORCE_CUTOFF` | 动作 `SAFE_HOLD`，剩余步清空，音频列不变，接触不变 |

类比失效处有三条，必须写进实验记录。第一，PCM 没播出还可以丢；已经送给功放的采样点无法从空气里收回。第二，剩余步没下发还可以丢；已经送给电机并造成接触的力无法从物体上收回。第三，第 07 课的 channel fusion 把用户向量追加进同一条因果序列，GR00T 的交叉注意把 VLM token 放在旁路，更像 cross-attention memory。不要把 Eagle 的 chat 格式理解成又在做全双工对话。

CPU 夹具把失效处变成断言。180 ms 发音频 `PAUSE` 之后、360 ms 的 `REPLAN` 之前，手臂仍在执行，接触在 250 ms 发生；这段时间旧 branch 不得再播出 PCM。`PAUSE` 若被你写成联合停止，`audio_pause_does_not_drop_remaining_action_steps` 会失败。

六个格子的合法例子和非法例子各写一条，方便对照 Lab 按钮。

| 格子 | 合法例子 | 非法例子 |
|---|---|---|
| `CONTINUE` / audio | 用户说“嗯”，助手继续原句 | 把“嗯”当成硬取消，清空 pending |
| `PAUSE` / audio | 用户开始纠正，停嘴，手臂仍伸 | 停嘴时把接触标志改假 |
| `REPLAN` / audio | 纠正已完整，取消未播旧半句 | 把已播的“我去拿左边”从播放表删掉 |
| `CONTINUE` / action | 杯子微晃，仍走当前块 | 微晃却整块丢弃，造成空档 |
| `PAUSE` / action | 人伸手进工作区，手臂保持 | 保持时顺带停嘴，调试时以为 Talker 死了 |
| `REPLAN` / action | 杯子被挪走，丢剩余步并新推理 | 新推理还在消费旧 `t_obs` 的闭合时刻 |

Lab 的“语音 PAUSE”对应第二行合法例子，“手臂 REPLAN”对应第六行合法例子，“一次撤销全部”同时触发第二行和第三行的非法例子。Gate 只在合法例子的组合上亮。预测题若选“`PAUSE` 等于力切断”，你会把第二行合法例子做成第五行非法例子：停嘴变成停臂，接触被抹掉。

与第 06 课话轮策略的边界也写清。第 06 课判断“这是不是该开口的事件”；[第 07 课](07_full_duplex_routing.md)判断“开口之后新输入怎么进状态”。本课判断“进状态之后，嘴和手各自的队列怎么处置”。VAD 命中不是 `REPLAN`。旁人说话不是力切断。把第 06 课的四动作再映射一遍：`WAIT` 对应两列都 `CONTINUE` 且不出新计划；`SPEAK` 对应音频列继续出队；`BARGE-IN` 至少触发音频 `PAUSE`，是否触发动作 `REPLAN` 要看观察有没有变。杯子没动、人只是插话问“这是什么杯子”，手臂可以 `CONTINUE`。杯子动了、人没说话，音频可以 `CONTINUE`，动作必须 `REPLAN`。两列独立，才能表达这两种世界。

### 5.5 音频 `PAUSE` 不是力切断

第 40 课问：力超限为什么不能像语音那样重说一遍。本课不抢那一课的安全训练，只借用它的运行时格子。

力切断的教学定义：

$$
\text{若 }\|F_i\|>F_{\max},\quad
\text{丢弃 }i+1,\ldots,H,\quad
\text{action\_mode}=\mathrm{SAFE\_HOLD}
$$

执行保持姿态。接触标志若已为真，保持为真。音频列不改。扬声器可以继续把已经生成的半句话说完，也可以另发一次音频 `PAUSE`；那是第二次事件。

SafeVLA 不代替这个格子。它的 ISA 在 CMDP 里用安全代价约束做训练期对齐，长程移动操作上相对对照方法降低 83.58% 的违规累积代价，任务成功率 +3.85%。读这篇是为了知道“安全可以进损失”。本课要进的是调度器：超限发生在第 $i$ 步时，第 $i+1$ 步不得再下发。训练期对齐没有公开到“第 $i$ 步的事件表”，其技术报告未公开该细节。不要把 83.58% 抄进本课 Lab 的 Gate。

CPU 在 220 ms 注入 `FORCE_CUTOFF` / `action`。此后 `action_mode` 为 `SAFE_HOLD`，`remaining` 为空，`audio_mode` 仍为 `SPEAKING`，220 ms 之后仍有 PCM 被播放。`force_cutoff_is_not_audio_pause` 锁这四件事。若你的实现一切断就停嘴，说明你把第 40 课的格子和第 07 课的格子焊死了。焊死之后，机器人在夹到桌沿时会突然沉默，调试的人会以为 Talker 崩了。

把三种“停”并排写，避免口语里都叫打断。

| 停的种类 | 触发 | 音频列 | 动作列 | 历史 |
|---|---|---|---|---|
| 语音 `PAUSE` | 用户插话、需要听完 | 出队冻结 | 不变 | 已播保留，pending 保留 |
| 动作 `PAUSE` | 人进入工作区、等待确认 | 不变 | 剩余步冻结 | 接触保留 |
| 力切断 | 单步力超过 $F_{\max}$ | 不变 | `SAFE_HOLD`，丢剩余步 | 接触保留且通常已为真 |
| 联合 `REPLAN` | 意图作废 | 取消未播 | 丢剩余步，新推理 | 已播与接触保留 |

四行的触发条件互不蕴含。插话不必超限，超限不必插话，杯子被挪走不必超限。Lab 只演示第一行加第四行里的动作半边：先 `PAUSE` / audio，再 `REPLAN` / action。完整四行要在 CPU 里分别跑，不能靠一次教具点击覆盖。

力阈值本身是第 40 课的教学约定，不是 SafeVLA 原文公式。SafeVLA 把安全写成 CMDP 里的代价约束，优化的是期望累积代价不超过限额。限额 $b_i$ 和本课的 $F_{\max}$ 量纲不同：前者是轨迹上的积分，后者是单步力。你可以同时使用两者：训练时用代价约束减少进入超限的频率，运行时仍要用 $F_{\max}$ 做硬切断。只用前者，一次尚未被训练覆盖的长尾碰撞没有刹车。只用后者，模型会频繁顶到阈值，任务成功率被切断本身打掉。本课夹具只实现后者，因为前者需要 GPU 和 Safety-CHORES。报告里写“运行时硬切断已测，训练期约束未做”。

### 5.6 `REPLAN` 取消两种未来

`REPLAN` 的合法对象是未来集合。写成两个差集：

$$
Q_{\mathrm{pcm}}^{+}=\{q\in Q_{\mathrm{pcm}}:q.\mathrm{played\_at}\ \text{未定义}\}
$$

$$
Q_{\mathrm{step}}^{+}=\{s\in Q_{\mathrm{step}}:s.\mathrm{executed\_at}\ \text{未定义}\}
$$

`REPLAN` 把 $Q_{\mathrm{pcm}}^{+}$ 和 $Q_{\mathrm{step}}^{+}$ 全部标 `superseded` 或直接从队列删除，然后

$$
\mathrm{branch}_{n+1}.\mathrm{parent}=\mathrm{branch}_n,\quad
\mathrm{branch}_n.\mathrm{status}=\mathrm{superseded}
$$

禁止对象是历史集合：

$$
Q_{\mathrm{pcm}}^{-}=\{q:q.\mathrm{played\_at}\le t_{\mathrm{replan}}\},\qquad
Q_{\mathrm{step}}^{-}=\{s:s.\mathrm{executed\_at}\le t_{\mathrm{replan}}\}
$$

$$
C^{-}=\{\mathrm{contact}=1\}
$$

对 $Q_{\mathrm{pcm}}^{-}$、$Q_{\mathrm{step}}^{-}$、$C^{-}$ 做删除或取反，就是非法 undo。一次点击若同时把 $Q_{\mathrm{pcm}}^{-}$ 清空并把 $C^{-}$ 改假，Lab 的 Gate 必须灭。一次点击撤销全部，不是把 `PAUSE` 和 `REPLAN` 合成了更省事的接口，而是把两种不可回放的历史写进了同一条 undo。

[第 30 课](30_closed_loop_control.md)已经断言：`REPLAN` 之后执行表不得再出现旧 `branch_id`。本课加一条对称断言：播放表也不得再出现旧 `branch_id` 在 $t_{\mathrm{replan}}$ 之后的 `played_at`。CPU 里 `old_pcm_played_after_replan` 与 `old_steps_after_replan` 都必须为 0，同时 `new_steps_after_replan` 必须大于 0。只丢不建，机器人会在重规划后冻住，看起来像 `SAFE_HOLD`，原因却是你忘了 `start_inference`。

联合计划里，嘴上的旧半句和手上的旧剩余步属于同一个被作废的意图。所以机制节写“`REPLAN` 同时取消未播 PCM 和未执行 chunk”。Lab 仍把取消拆成两次点击，是为了让日志出现两行：`audio PAUSE` 与 `action REPLAN`。若你在生产 API 里提供 `replan_joint()`，内部顺序应固定为：先停音频出队，再丢剩余步，再开新 branch，最后在 trace 里写两条 `kind=control`。一条 API 可以，一条日志不行。

### 5.7 已播出和已接触不能 undo

“撤回”在软件里太便宜。播放光标减一，布尔量改假，演示就可以重来。身体上这两件事的代价不对称。

已播 PCM。采样点一旦送给扬声器，房间里的人已经听见“我去拿左边的——”。你可以停掉后半句，不能让前半句从记忆里消失。评测若统计 context corruption，前半句仍在上下文里，必须带着它重新规划，而不是从日志里删掉。第 07 课的 CCR 统计的是新旧混说；本课额外统计“历史被抹掉”。抹掉历史会让 CCR 假性变好：旧词不在生成里了，因为你把旧词从记录里删了。

已发生接触。夹爪合上的那一拍，杯壁已经受力。第 40 课规格说超限后停在当前姿态。杯子可能已经滑动。调度器若把 `contact_occurred` 改回假，安全审计会认为没有碰撞，下一次又用同样的力去合。SafeVLA 降低的是训练期累积代价，不给你运行时“撤销碰撞”的权限。

CPU 把合法路径和非法路径并排跑。合法路径：`played_at_replan > 0`，结束时 `played_pcm` 不少掉，`contact_occurred` 仍为真，`contact_undone` 为假。非法路径：同一份事件加上 `illegal_undo=True`，旧 branch 的已播 PCM 被清空，接触被改假。`illegal_undo_rewinds_history_and_must_be_rejected` 为真，表示反例夹具确实伪造了历史；主验收不得走这条函数参数。Lab 把非法路径做成按钮“一次撤销全部”。点它，Gate 灭。

非法路径在实现里往往不叫 `undo()`。它通常是三个看似无害的赋值：`played_pcm.clear()`、`contact_occurred=False`、`played_until_ms=0`。代码审阅时搜这三处。若出现在 `REPLAN` 或 `PAUSE` 分支，就是本课要挡的伪造。合法清理只针对 `pending_pcm` 与 `remaining`。播放列表和执行列表可以追加“已取消”记录，不得删除已经成功的行。数据库里用软删除标志也可以，硬删除历史行不行。夹具用列表长度单调来代替软删除：合法路径上 `len(played_pcm)` 与 `int(contact_occurred)` 对时间不降。

### 5.8 Thinker–Talker 与 System 2 / System 1 接到同一行

[第 19 课](19_capstone_thinker_talker.md)的慢快是理解和发声。Thinker 出文本或隐表征，Talker 按位置消费 `bridge_states` 再出 codec。Thinker 暂停时 Talker 仍可能把队列里的帧说完，条件是冻结的。[第 29 课](29_dual_system_vla.md)的慢快是规划和控臂。System 2 出子目标 $g$，System 1 每步消费 $g$。System 2 暂停时 System 1 仍会出动作，条件也是冻结的。

并表之后，一行状态上最多同时存在两种冻结条件：给 Talker 的桥向量，给动作专家的 $g$。它们可以来自同一个 Thinker / System 2 前向，也可以来自两次前向。本课不规定必须共享一次前向。本课规定：两次前向的 `available_at` 仍要分列。Qwen2.5-Omni 让 Talker 直接读 Thinker 隐表征，训练和推理端到端；那是把桥焊死在模型里，不是把音频时钟焊死在控制时钟上。GR00T N1 让 DiT 交叉注意 Eagle 中层 token，同样焊的是条件，不是墙钟。Helix 博文让 S2 异步写一块共享内存里的隐向量，S1 以 200 Hz 读最新值。共享内存是条件寄存器，仍需要两列时间：隐向量何时写完，动作何时锁存。博文没有把这两列写进可引用的事件表，本课不得补造。

伪代码与 CPU 实验的骨架一致：

```python
def joint_step(now, row, audio_events, action_events):
    for event in due(audio_events, now):
        apply_audio(row, event)
    for event in due(action_events, now):
        apply_action(row, event)
    if row.audio_mode == "SPEAKING":
        emit_or_play_pcm(row, now)
    if row.action_mode == "GENERATING":
        execute_or_infer(row, now)
    return row
```

`apply_audio` 不得调用 `drop_remaining`。`apply_action` 的 `FORCE_CUTOFF` 不得调用 `cancel_pending_pcm`。`REPLAN` 可以调用两者，但必须另写两条 trace。把四个 if 焊成一个 `if interrupted: reset_everything()`，本课全部断言作废。

把第 19 课的四段音频内部时钟嵌进本课音频列，避免有人以为本课否定了四段划分。listener 帧决定 `captured_at`。Thinker 出字决定文本条件何时变。Talker 出 codec 决定未解码帧何时存在。播放器出 PCM 决定 `played_at`。四段都完成后，才更新 `played_until_ms`。本课的 `audio_available_at` 卡在“模型最早能用”这一层，通常是 listener 编码结束或 Talker 条件就绪，不是播放结束。播放结束是历史，属于单调光标。谁把播放结束写进 `available_at`，控制头会看到已经过时的用户话，`PAUSE` 会晚一整句。第 19 课 Lab 用级联取消来清 Thinker、Talker 和队列；本课要求级联取消仍按列记账：取消 Talker 只影响音频未来，取消动作专家只影响动作未来。级联可以同时发生，日志必须仍是两条。

第 29 课的 $g$ 嵌进本课动作列时，注意 $g$ 过期和 chunk 过期是两层。$g$ 过期：规划太久没更新，快环还在追旧路点。chunk 过期：这一块动作自己的 $d\ge H/f$。可以 $g$ fresh 而 chunk stale（规划刚写完，动作专家太慢）；也可以 $g$ stale 而 chunk fresh（规划停了，手里还有一块没过期的旧动作）。第二种必须强制动作 `REPLAN`，否则新鲜块执行的是陈旧意图。本课默认夹具不注入 $g$，等于假设 $g$ 始终 fresh。加第三列之前，不要把 chunk 过期解释成规划过期。

### 5.9 接口说明书

产物的第一件是接口，不是权重。最小可实现的行类型如下。字段名可以改，语义不能改。

```python
@dataclass
class DualClockRow:
    session_id: str
    branch_id: int
    parent_id: int | None
    status: str  # active | superseded | done
    audio_available_at_ms: int
    action_available_at_ms: int
    audio_mode: str  # SPEAKING | PAUSED
    action_mode: str  # GENERATING | PAUSED | SAFE_HOLD
    pending_pcm: list
    played_until_ms: int
    remaining_steps: list
    executed_steps: list
    contact_occurred: bool
    force_n: float
```

```python
@dataclass
class ControlEvent:
    sequence_no: int
    channel: str  # audio | action
    action: str   # CONTINUE | PAUSE | REPLAN | FORCE_CUTOFF
    available_at_ms: int
    captured_at_ms: int
```

禁止事项写成契约，评测按行扫描：

1. 不得出现无名通道的控制事件。
2. 不得用 `action_available_at_ms` 去判断音频过期。
3. 不得在 `audio_mode=PAUSED` 时把 `played_until_ms` 减小。
4. 不得在 `contact_occurred=True` 之后把它写回 `False`。
5. `REPLAN` 必须同时产生 `canceled_pcm` 与 `discarded_steps` 两类记录，或显式写明某一侧当时为空。
6. `FORCE_CUTOFF` 的 `reason` 只能是 `force_cutoff`，不得复用 `replan`。

这份说明书够写一个教学状态机，不够写发布级机器人。缺的东西包括：AEC、抖动缓冲、多机协作、真机力传感器标定、在线 Thinker 的 TTFT。第 19 课已经把双工回放和在线真双工分开。本课停在回放级协议。谁把这份 dataclass 写成“开源 Helix”，属于把接口说明书升级成了产品声明。

字段级读写规则再列一遍，给实现者当代码审阅清单。

`session_id` 在一次 Lab 运行或一次 CPU `replay` 内不变。换杯子位置不换会话。换会话是用户离开又回来，本课不覆盖。

`branch_id` 只在 `REPLAN` 时增加。`PAUSE`、`CONTINUE`、`FORCE_CUTOFF` 都不得新开 branch。谁在 `PAUSE` 时把 `branch_id` 加一，恢复就会变成续写在新树上，第 07 课的 resume 语义被破坏。

`audio_available_at_ms` 只由音频编码路径写入。动作循环即使跑得更勤，也不得每 50 ms 刷新这一列，否则音频过期会被动作滴答带着走。

`action_available_at_ms` 只在推理开始时写成 $t_{\mathrm{obs}}+d$，块就绪时读，不得在执行每一步时改成 `now_ms`。改成 `now_ms` 等于把延迟藏起来，freshness 永远成立。

`pending_pcm` 与 `remaining_steps` 是未来集合，允许被 `REPLAN` 和 stale 路径删除。删除时必须留下 `discard_reason`。静默删除会让评测以为从未生成过。

`played_until_ms`、`executed_steps`、`contact_occurred` 是历史集合。只允许追加。测试里对这三项做“赋值变小”的代码审查，比听音频更有效。

`force_n` 若为 -1，`FORCE_CUTOFF` 事件非法。若为真实力，比较用同一单位：牛顿或牛顿米，不要一列为原始 ADC。单位写进 session 元数据，不写进每一步以免不一致。

对外暴露的函数签名建议保持纯函数：输入 `row, event, now_ms`，输出新的 `row`。随机数、当前系统时间、全局单例播放器都不进函数。需要播放时返回一个 `PlayCommand`，由确定性循环在 `now_ms` 执行。这和[第 07 课](07_full_duplex_routing.md)的 microstep 调度同一纪律：每次只处理已经到达的事件，顺序由 `(available_at, sequence_no)` 决定。

### 5.10 教学状态机

第二件产物是可回放的状态机。它只认整数毫秒、确定事件和两列队列。默认夹具如下。

| 时刻 (ms) | 事件 | 音频列 | 动作列 |
|---|---|---|---|
| 0 | 开机，branch 1 推理 | 开始按 80 ms 产帧 | `t_obs=0`，`available_at=100` |
| 70–150 | 帧到期 | 已播 PCM 增加 | 100、150 执行 step 0、1 |
| 180 | `PAUSE` / audio | 出队冻结，pending 保留 | 继续走剩余步 |
| 200–250 | 控制滴答 | 无新播放 | step 2、3；step 3 接触为真 |
| 360 | `REPLAN` / action | 取消未播旧 PCM | 丢弃旧剩余步，branch 2 推理 |
| 360 之后 | 新 branch | 不得再播 branch 1 | 不得再执行 branch 1 |

180 与 360 必须是两条 `consumed` 记录，通道分别为 `audio` 和 `action`。把它们合成一次 180 ms 的联合事件，Lab 验收失败。把 360 写成“撤销全部”并清空已播计数，CPU 的非法夹具会亮，主夹具会灭。

确定性：同一份事件跑两次，trace 的 JSON 字节级相等。本课无随机前向，这条代替[第 01 课](01_baseline_reproduction.md)的“插桩不改 logits”。你若在播放函数里读系统时间，trace 会漂，验收作废。

默认夹具的逐步计数可以手算核对。控制周期 50 ms，音频帧 80 ms，编码 40 ms，播放滞后 30 ms，动作延迟 100 ms，接触步 3。

| 时刻 (ms) | 音频列计数 | 动作列计数 | 接触 |
|---|---|---|---|
| 0 | 产帧 0，pending+1 | 开始推理 | 假 |
| 70 | 帧 0 播出，played=1 | 推理中 | 假 |
| 100 |  | chunk 就绪，执行 step 0 | 假 |
| 150 | 帧 1（80 ms 产）播出，played=2 | 执行 step 1 | 假 |
| 160 | 产帧 2，`play_at=230` |  | 假 |
| 180 | `PAUSE`，出队冻结 | 剩余仍含 step 2 以后 | 假 |
| 200 | 无新播放 | 执行 step 2 | 假 |
| 230 | 帧 2 到期但 PAUSED，仍 pending |  | 假 |
| 250 |  | 执行 step 3，接触变真 | 真 |
| 360 | `REPLAN` 取消 pending 帧 2 | 丢弃 step 4–7，branch 2 推理 | 真，保持 |

手算应得：`PAUSE` 前已播 2 帧；`PAUSE` 到 `REPLAN` 之间播放增量为 0、执行增量至少 2 步；`REPLAN` 后旧播放增量 0、旧执行增量 0；接触在 250 ms 变真后不再变假。CPU `metrics` 里 `played_pcm` 会大于 2，因为新 branch 在 360 ms 之后继续产帧。比较旧未来时必须按 `branch_id` 过滤，不能拿总播放数当旧播放数。Lab 揭晓的“旧未来泄漏”两格应对 0、0。若你看见 PCM 泄漏 1，多半是 `REPLAN` 时把 pending 留在了已 superseded 的 branch 上，到期后又走进了播放分支。正确的到期逻辑是：`status=superseded` 的到期帧记 `canceled`，不记 `played`。

教具里杯子的位置是观察，不是计划。旧计划的抓取点仍是桌面左侧的闭合带。杯子被拖到右侧之后，旧剩余步若还在跑，夹爪会在空处合上，接触仍可记真（碰到了桌子或空气中的力阈值），抓住为假。新计划必须重新锁存右侧位置。Lab 不训练视觉，拖动直接改 `cupX`。这和[第 30 课](30_closed_loop_control.md)传送带夹具一样：感知误差被设成零，只暴露调度。不要把“拖对了杯子”写成“模型看见了杯子”。

## 6. 在公开实现中定位这些机制

本课没有 MiniMind-O 补丁可合。定位以已经存在的课文、官方仓库和打开过的技术报告为界，不编造未打开的源码行号。

音频列落在[第 07 课](07_full_duplex_routing.md)已经点名的符号上：`RealtimeSession`、`stream_pcm`、`poll_interrupt`、`ResponseBranch.played_until_ms`。当前 MiniMind-O 在 VAD 命中后 hard cancel，相当于只有一种破坏性 `REPLAN`，而且会把不该删的播放历史一起切断。本课要的是：hard cancel 升级成带通道的控制事件，播放光标只前进不回退。

Thinker–Talker 落在[第 19 课](19_capstone_thinker_talker.md)的 19C：listener、Thinker、Talker 并行，支持 continue / pause / replan。默认交付仍是双工回放。本课不把回放升级成在线真双工，只要求回放日志出现两列时间戳。Qwen2.5-Omni 的 Talker 读 Thinker 隐表征，滑动窗口 DiT 限制感受野以降低首包延迟。报告把首包延迟拆成四段：多模态输入处理、第一个文本到达后到第一个语音 token、第一段语音转成可播波形、以及模型规模本身的计算。本课音频列的 $d_{\mathrm{enc}}$ 对应第一段加必要的预填充；`play_at` 相对 `available_at` 的滞后对应第三段。第二段属于 Thinker 与 Talker 的桥，第 19 课已经单独测。第四段不要写进过期公式。你能在报告第 2.4 节定位流式波形和 4 块滑窗；你不能在同一份报告里定位本课的 `SAFE_HOLD` 或手臂 `REPLAN`。没有的格子写“其技术报告未公开该细节”。

Qwen2.5-Omni 还把视频和音频按 2 秒一块交错。2 秒是输入组织粒度，不是本课 $T_{\mathrm{frame}}$，更不是 $H/f$。若有人把 2 秒写进动作开环窗口，50 Hz 上相当于 $H=100$ 的整块盲走。那是把理解侧的打包策略抄进了控制侧。本课对照表里这一格应保持空白或明确写“输入交错，非动作窗口”。

动作列落在[第 30 课](30_closed_loop_control.md)的 chunk 队列和[第 29 课](29_dual_system_vla.md)的双循环。ACT 的 Algorithm 1 / 2 给出分块与时间集成；Diffusion Policy 给出 $T_p$、$T_a$；OpenVLA-OFT 在 ALOHA 上 $K=25$ 整块执行；π0 用 $H=50$。这些系统证明动作侧需要 $H/f$，没有证明它们内部已经和语音路由并表。GR00T N1 的公开入口是 `NVIDIA/Isaac-GR00T` 与 `nvidia/GR00T-N1-2B`。推理时要实测 VLM 调用频率是不是每控制步都跑，不要把论文里的 10 Hz 抄到你的板子上。

Helix 只能定位到官方博文。S2 7–9 Hz、S1 200 Hz、共享隐向量、双嵌入式 GPU、训练时间偏移，这些句子可以引用。博文还写明：S2 是 7B 开源权重 VLM，处理单目图像、腕部位姿与手指位置，以及自然语言命令，把语义压成一个连续隐向量交给 S1；S1 是 80M 的交叉注意编解码 Transformer，视觉骨干在仿真里预训，输出上身 35 自由度目标，并附加一个“任务完成百分比”用于衔接多段行为；约 500 小时多机多操作员遥操作，用自动标注 VLM 生成事后指令；训练端到端回归，梯度经隐向量传回 S2；部署时 S2 与 S1 分卡异步，S1 读共享内存里最新隐向量。CONTINUE 表、力阈值、$H$ 与开环窗口的具体整数，博文没给。本课连复现都不做，只借用“两个时钟、一块条件寄存器、训练时把部署延迟写进输入偏移”这三句。第三句对应本课的 $d$ 必须被测量，不能对应本课已经写好的两列字段名。

把 Helix 的共享内存隐向量硬翻译成本课字段时，最多写成：`plan_vector` 带 `plan_available_at_ms`，S1 每步读当前值。不要翻译成 `audio_available_at_ms`。Helix 博文演示里机器人几乎不说话，没有 PCM 队列可对。谁在对照表里给 Helix 填 80 ms 音频帧，属于把 Moshi 的数借给了 Figure。

SafeVLA 定位到训练损失和 Safety-CHORES 环境。运行时若要接本课的 `FORCE_CUTOFF`，你得自己在控制循环里读力，按第 40 课规格切剩余步。不要在 SafeVLA 仓库里找 `audio_available_at_ms`。

推荐代码边界是四个函数，而不是一颗 7B 加一个 80M：

- `apply_audio(row, event)`
- `apply_action(row, event)`
- `expire_audio(delay_ms, frame_ms)`
- `expire_action(delay_ms, horizon, freq_hz)`

把 Qwen 的 Talker 和 GR00T 的 DiT 填进后两个函数之前的发射器，是有卡之后的事。先把事件通道和过期公式写对。

按“一次说话时挪杯”的事件路径读公开系统，看它们停在哪一格：

1. 麦克风 packet 到达。MiniMind-O 的 `recv_loop` 能收；没有通道字段。Moshi 把用户流写进同一时间轴。Helix 博文不讨论语音输入列。
2. 编码完成，写出 `audio_available_at`。第 07 课有；Qwen2.5-Omni 有块级编码器，没有用这个字段名。
3. 控制头输出带通道的动词。第 07 课有三分类、无通道。第 06 课有话轮、无手臂。其余公开 Omni 报告大多未给可引用的控制头表。
4. 播放队列按 branch 出队。第 07 课要求可撤销缓冲。第 19 课要求打断后旧 PCM 不泄漏。
5. 相机锁存，$t_{\mathrm{obs}}$。ACT / π0 / GR00T 都有观察，频率各写各的。
6. 动作块到达，写出 `action_available_at`。第 30 课有公式。Helix 博文有 200 Hz，没有 $H$。
7. 剩余步执行或丢弃。第 30 课有 `REPLAN` 丢弃。第 40 课规格有力切断丢弃。
8. 接触位置位。SafeVLA 在训练里惩罚不安全行为，不保证这一位单调。

八步里，公开系统往往只实现 1–4 或只实现 5–7。本课夹具把 1–8 串成一次回放，每一步只检查协议，不检查识别率。谁在第 3 步用 VAD 布尔量代替带通道动词，后面的 4、7、8 都会被同一颗按钮带走。

## 7. 数据与训练 recipe

本课不训新模型。所谓 recipe，是事件数据和标签契约。缺契约时，控制头会在“嗯”和“杯子被拿走”上输出同一个 logit。

事件记录建议按[第 01 课](01_baseline_reproduction.md)的结构化 trace 追加下列键：

```text
session_id
branch_id
parent_id
channel
control_action
captured_at_ms
audio_available_at_ms
action_available_at_ms
consumed_at_ms
played_at_ms
executed_at_ms
step_index
frame_id
pending_pcm
remaining_steps
contact_occurred
force_n
discard_reason
```

`discard_reason` 只允许 `stale`、`replan`、`force_cutoff`、`superseded`、`receding_uncommitted`、空。空表示未丢弃。未知字符串让评测失败。`channel` 只允许 `audio`、`action`。`control_action` 只允许 `CONTINUE`、`PAUSE`、`REPLAN`、`FORCE_CUTOFF`。

若你要为控制头收集监督，标签必须带通道。同一段波形可以同时是音频 `PAUSE`（用户开始说话）和动作 `CONTINUE`（杯子没动）。同一段图像可以同时是动作 `REPLAN`（杯子跳变）和音频 `CONTINUE`（没人插话）。把它们压成一个三分类，会回到第 07 课只处理嘴、第 30 课只处理手的旧世界。最小监督是六分类：3 个动词乘 2 个通道。联合 `REPLAN` 若出现，应展开成两条标签，时间戳可以相同，`sequence_no` 必须不同。

数据来源分层写，避免把语音语料的打断标注直接当成手臂标签：

| 层 | 来源 | 能监督什么 | 不能监督什么 |
|---|---|---|---|
| 语音双工 | Full-Duplex-Bench 及 v1.5、自建打断脚本 | 音频列动词 | 接触、力、chunk 过期 |
| 动作分块 | ACT / Diffusion Policy / π0 轨迹 | $H$、$k$、扰动后是否该 `REPLAN` | 未播 PCM |
| 力与接触 | 带力的遥操作或第 40 课夹具 | `FORCE_CUTOFF`、接触不可 undo | 语义打断 |
| 联合教学 | 本课 Lab 与 CPU 回放 | 两列时间戳是否分家 | 真机成功率 |

预算：标 200 条联合时间线，大约 1–2 人日，只够验证协议，不够训 7B。若微调公开 VLA 或 Omni，本课只要求在日志里加两列时间戳，不要求改损失。改损失属于改造清单，见第 13 节。

不要把 Helix 的 500 小时遥操作、GR00T 的 OXE 混合物、Qwen2.5-Omni 的语音指令数据倒进同一个 batch 还声称这是本课 recipe。那些数字属于各自论文。本课 recipe 的验收是：字段齐全、通道互斥、历史字段单调。

单调性写死：

$$
\mathrm{played\_until}(t+\Delta t)\ge\mathrm{played\_until}(t)
$$

$$
\mathrm{contact}(t+\Delta t)\ge\mathrm{contact}(t)
$$

接触用 0/1，只允许 0 变 1。播放光标只允许前进。违反任一式，trace 作废。

## 8. 按依赖顺序执行实验

实验分两层。CPU 层证明状态表。教具层证明两次事件和不可撤回的历史。没有 GPU 的读者把 Step 0 到 Step 5 做完即完成本课主路径。Step 6 标 `skipped-no-gpu`，允许引用第 29、30 课已经核对过的 $d$，不要留空让人以为你测过联合系统。

CPU 与教具共用过期方向 `>=`，参数不同。CPU 用 $f=20$、$H=8$、音频块 320 ms、180 ms `PAUSE`、360 ms `REPLAN`，好让接触落在两次事件中间。教具允许你拖杯子、改帧长和 $f$。不要因为滑条数字不同就认为协议不一致。

CPU 的 `checks` 是验收清单的机器版：

- `state_table_has_two_timestamp_columns` 锁字段；
- `audio_and_action_expiry_use_different_windows` 锁 320 对 400；
- `events_never_consumed_before_own_available_at` 锁偏序；
- `pause_and_replan_are_two_channel_events` 锁两次事件；
- `audio_pause_does_not_drop_remaining_action_steps` 锁 `PAUSE` 不等于停臂；
- `replan_cancels_unplayed_old_pcm` 与 `replan_does_not_execute_old_remaining_steps` 锁未来集合；
- `replan_does_not_undo_played_pcm_or_contact` 锁历史集合；
- `force_cutoff_is_not_audio_pause` 锁第 40 课格子；
- `illegal_undo_rewinds_history_and_must_be_rejected` 锁反例夹具；
- `event_replay_is_deterministic` 锁可复核。

网页练习只展示 `metrics` 里的有限数字：窗口、已播、取消、旧未来泄漏、接触。不要把完整 trace 当成功率。`old_pcm_played_after_replan` 和 `old_steps_after_replan` 必须为 0。

### Step 0：冻结两列符号

在记录里写死 $T_{\mathrm{frame}}$、$f$、$H$、$d_{\mathrm{enc}}$、$d$。断言 $(H,f)=(8,20)$ 的窗口为 400 ms，320 ms 只让音频过期。

```bash
PYTHONPATH=experiments/src python3 -c "from learn_omni_experiments.lessons.lesson_48 import audio_is_stale, action_is_stale; assert audio_is_stale(320) and not action_is_stale(320, 8, 20)"
```

### Step 1：回放默认时间线

实现 `replay`。事件为 `(time_ms, channel, action)`。滴答 10 ms。断言消费不早于各自 `available_at`，快照含两列时间戳。隔离实现位于 `experiments/src/learn_omni_experiments/lessons/lesson_48.py`。登记前从仓库根目录运行 Step 0 同一条导入；登记后在 `experiments` 目录运行课程脚本。未登记时不要改 `lesson_id`。

### Step 2：`PAUSE` 夹具

180 ms 音频 `PAUSE`。此后旧 branch 不得新播 PCM，手臂在 360 ms 之前仍执行，250 ms 接触为真。若 `PAUSE` 清空了 `remaining`，说明焊成了力切断。

### Step 3：`REPLAN` 夹具

360 ms 动作 `REPLAN`。旧 `branch_id=1` 此后不得出现在播放表和执行表。新 branch 必须开始执行。已播计数不得下降。接触保持为真。同一份事件跑两次，trace 相等。

默认参数下，把 360 ms 附近的队列状态抄进记录，防止只看布尔量。`PAUSE` 之后 pending 里至少有一帧（160 ms 产、230 ms 到期但被冻结的那一帧）。`REPLAN` 必须把它写进 `canceled_pcm`，`reason=replan`。动作侧在 250 ms 执行完 step 3 后，若尚未走完 $H=8$，剩余里应有 step 4 到 7 中尚未执行的下标。`REPLAN` 把它们写进 `discarded_steps`。新 branch 的 `parent` 为 1，`audio_mode` 回到 `SPEAKING`，`action_mode` 回到 `GENERATING`，并立刻 `start_inference`。若你希望重规划后暂时保持沉默，应在 `REPLAN` 之后再发一条音频 `PAUSE`，不要把沉默写进 `REPLAN` 的默认副作用。默认副作用只有：取消两种未来、开新 branch、启动新推理。

360 ms 这个数不能随便改成 400 再声称同一份夹具。400 ms 刚好等于 $H/f$，旧块若此时才到达会走 stale 分支，不再走 replan 丢弃，断言名字会对不上。180 ms 也不能改成 100 ms：100 ms 是第一块动作刚就绪的时刻，提前 `PAUSE` 仍合法，但接触可能还没发生，`contact_at_replan` 会变成假，历史单调的检查就少了一半。改参数必须同时改记录里的接触时刻。

### Step 4：力切断对照

220 ms `FORCE_CUTOFF`。`SAFE_HOLD`，剩余步为空，音频继续播放。不要在这一步发音频事件。

### Step 5：双时钟教具

打开本课 Lab。先选预测。开始说话并伸手。把杯子拖走，或点“挪走杯子”。分别点“语音 PAUSE”和“手臂 REPLAN”。不要点“一次撤销全部”。关键数字在运行后揭晓：两列 `available_at`、已播、剩余步、接触、旧未来泄漏。教具标明教学模拟。

推荐操作顺序：预测选两次事件；开始；约 200 ms 后拖杯子；再 `PAUSE`；再 `REPLAN`。若先点撤销全部，必须重置。调滑条会清掉事件；重置会清空预测。页面在未运行时用“揭晓后”占位，防止先看数字再选预测。

教具左侧是 PCM 条：已播、未播、取消用不同填充。右侧是桌面：夹爪随旧计划走向左侧闭合带，杯子可被拖到右侧。接触时夹爪合拢。底部四格是状态表的对外投影，对应 CPU 的两列时间戳、剩余步、接触和旧未来泄漏。事件日志必须出现两行，通道分别为 audio 与 action。同一毫秒点两次在实现里会用 `sequence` 分开；不要依赖“时间不同”这一个条件。若浏览器动画跳到末帧，数字仍应揭晓，Gate 仍只看事件与计数，不看你有没有看完动画。

把教具滑条和 CPU 默认值对一下。CPU：帧 80 ms，$f=20$，$H=8$，$d=100$ ms。教具默认相同。把 $f$ 拖到 50 会让 $H/f$ 变成 160 ms，若 $d=100$ ms 仍 fresh；把 $d$ 拖到 400 且 $f=20$、$H=8$，动作列过期，手臂不再执行，接触可能一直为假。过期抓空是第 30 课的验收，本课 Gate 不要求你找到抓空组。本课 Gate 要求：杯子动过、两次事件、没有非法 undo、旧未来为零。找到抓空组可以作为额外观察，写进笔记，不要当成第 48 课通过条件。

### Step 6：可选，把公开 $d$ 填进两列

若有卡，分别测一次 Talker 出首帧的延迟和一次动作块延迟，填进 $d_{\mathrm{enc}}$ 与 $d$。协议对齐各自论文的输入规格。测不到就引用第 19 课和第 30 课已经抄过的数字，并写“未在本课复测”。不要用 Helix 视频估 200 Hz 当作你的 $d=5$ ms。

## 9. 评测与测量

主指标不是成功率一个数，是时间线是否自洽。

| 指标 | 定义 | 通过条件 |
|---|---|---|
| 双列齐全 | 每一行快照含两个 `available_at` | 缺一列即失败 |
| 偏序 | `consumed_at >= available_at`（按通道） | 全部事件成立 |
| 分窗过期 | 320 ms 只 stale 音频，400 ms 才 stale 动作 | 与手算一致 |
| 分事件 | `PAUSE`/audio 与 `REPLAN`/action 两条 | 通道不同、时刻不同 |
| 未来取消 | `REPLAN` 后旧 PCM 与旧步为零 | 两个计数都是 0 |
| 历史单调 | 已播计数与接触标志不下降 | 合法路径上成立 |
| 力切断正交 | `SAFE_HOLD` 时仍可说话 | 切断后仍有 PCM |
| 确定性 | 两次 replay 的 trace 相等 | 字节级 |

辅助观察：`pause_steps_during_audio_pause` 应大于 0，用来证明停嘴时手臂还在动。它不是成功率。Helix 博文的“拿起未见过的小物体”、GR00T 的仿真表、π0 的 50 Hz 声明，全部不得填进上表的通过条件。

测量时分别统计四段延迟：音频编码、音频排队、动作前向、动作排队。把四段加在一起写成一个“端到端延迟”可以，但过期判定仍用各自的段。第 19 课的 TTFA 是请求到首个可播放 PCM；本课默认回放不报告 TTFA，因为请求时刻是夹具给的。

把本课数字和前几课数字并排写时，单位必须带着对象。320 ms 是 listener 块，不是 Helix 的 S1 周期。400 ms 是 $H/f$，不是 Moshi 的实测 200 ms。63.9 ms 是 GR00T N1 在 L40 上 16 步 4 次去噪，不是本课默认 $d=100$ ms。7–9 Hz 是 Helix S2，不是 GR00T 的 10 Hz。抄进同一单元格而不写对象，等于没抄。评测附录建议三列：数值、对象、出处页。出处写“本课夹具”的行，不得混进“论文表”。

## 10. 验收条件

主路径同时满足下列条目才算通过：

1. 课文第一个 H2 与规格一致，产物被写成接口说明书加教学状态机。
2. CPU `run()["checks"]` 全部为 True，至少覆盖两列时间戳、分窗过期、两次事件、`REPLAN` 后旧未来为零、历史不被撤回。
3. Lab 在先预测的前提下，用两次独立点击完成 `PAUSE` 与 `REPLAN`，Gate 亮；“一次撤销全部”使 Gate 灭。
4. 实验记录里出现 5.10 节那张默认时间线，数字与本地 replay 一致或注明改过的整数参数。
5. 报告明确写：未复现 GPT-4o，未复现 Helix，未在真机上测力。
6. 相对链接只指向仓库里已有的课文文件。

任一条失败，停在本课，不要带着焊死的按钮去写“会说话的 VLA”。

## 11. 根据症状定位失败环节

| 症状 | 先查 | 常见原因 | 处理 |
|---|---|---|---|
| 320 ms 时动作块也被丢 | `expire_action` | 共用音频块长 | 分开 $T_{\mathrm{frame}}$ 与 $H/f$ |
| `PAUSE` 后手臂停死 | `apply_audio` | `PAUSE` 调用了 `drop_remaining` | 音频函数禁止改动作列 |
| 切断后突然沉默 | `FORCE_CUTOFF` | 顺手 `cancel_pending_pcm` | 力切断只改动作列 |
| `REPLAN` 后仍抓住旧杯位置 | 执行表 | 旧 `remaining` 没清，或新推理没启动 | 对旧 branch 断言零步，并检查 `infer_start` |
| `REPLAN` 后仍播出旧半句 | 播放表 | pending 没按 `branch_id` 过滤 | 取消条件写成旧 branch，不是“全部 PCM” |
| 已播计数变小 | 合法路径误走 undo | 播放光标被重置 | 单调性断言 |
| 接触先真后假 | 日志写回 | 把重规划当成物理回放 | 接触只允许 0 变 1 |
| 两次 replay 不一致 | 读了墙钟或字典序 | 集合遍历无序 | 事件按 `(time, sequence)` 排 |
| Lab Gate 不亮 | 预测或点击 | 选了一次撤销，或两次点在同一毫秒且被合并 | 重置后按 Step 5 |
| 报告写成复现 Helix | 措辞 | 把 7–9 Hz / 200 Hz 当成状态表 | 改回“博文给出频率，未给出本课表” |

不要把“听起来停得及时”当成定位工具。及时可能来自 hard cancel，hard cancel 会破坏历史单调性。先看计数，再听声音。

再补四条联合症状，都是嘴和手同时在场才出现的。

| 症状 | 先查 | 常见原因 | 处理 |
|---|---|---|---|
| 说话停了，夹爪却把空杯子位合上 | `PAUSE` 后的执行表 | 只发音频事件 | 杯子位移必须再发动作 `REPLAN` |
| 手臂改去新杯，扬声器仍说“左边” | 播放表旧 branch | 联合 `REPLAN` 只丢了剩余步 | 取消 pending 时按 `branch_id` 过滤 |
| Lab 点了两次，日志只有一行 | 事件合并 | 同一毫秒且无 `sequence` | 用序号打破并列 |
| CPU 合法路径接触为假 | 接触步时刻 | `PAUSE` 太早或 $d$ 太大 | 对默认夹具核对 250 ms |

四条里前两条是产品会当成“模型胡言”的现象。拆开日志后，常常是事件少发了一次，或过滤条件写成了“删掉全部 PCM”而不是“删掉旧 branch 的未播 PCM”。后两条是夹具自己的坑，改代码前先重跑默认时间线。

## 12. 交付物

交卷时手里应有：

1. 一份 `DualClockRow` / `ControlEvent` 说明书，含禁止事项六条。
2. `lesson_48.py` 的 `run()` 输出，`checks` 全 True，`metrics` 里两个旧未来计数为 0。
3. Lab 截图或笔记：预测选项、两次事件日志、未点撤销全部。
4. 5.10 节时间线的本地核对，或一份改参数后的对照表。
5. 文献表：第 07 / 19 / 29 / 30 课，Qwen2.5-Omni，Helix 博文，SafeVLA。每条旁写“能支持什么 / 不能支持什么”。
6. 一段限制声明：教学状态机，非发布系统；不声称复现 GPT-4o 或 Helix。

不要交：7B 权重、真机成功率、把 LIBERO 平均写进本课、把 Helix 视频当 freshness 证据。

交付物里的限制声明建议直接抄这段，再按你的实验改数字：

```text
本课产物是 DualClockRow 接口与确定性教学状态机。
CPU 默认 f=20 Hz, H=8, 音频块 320 ms。
旧 PCM 与旧剩余步在 REPLAN 后执行数为 0。
未加载 Qwen2.5-Omni / GR00T / Helix 权重。
未在真机测力，未复现 GPT-4o 或 Helix。
```

无语言围栏里不要写可执行命令。上面六行是声明，不是脚本。谁把 `python3 run.py` 写进这段声明，构建检查会判失败。真正的命令只出现在第 8 节那个 bash 围栏里，而且只有一条。

## 13. 前沿对照与改造方向

**公开方案。** 2024–2025 年把“慢理解、快输出”写进主路径的系统，正好覆盖本课两列时钟，却几乎都不公开两列 `available_at`。把它们按“嘴的列 / 手的列 / 有没有并表契约”分成三组，读的时候不要混组引用。第一组只管嘴：Moshi、Qwen2.5-Omni、第 19 课 19C。第二组只管手：ACT、Diffusion Policy、OpenVLA-OFT、π0、GR00T N1、$\pi_{0.5}$、Helix 博文。第三组管训练期安全：SafeVLA。本课夹具是把第一组的动词和第二组的窗口写进同一张表，并明确拒绝第三组代替运行时切断。任何一篇单独的论文都不够交卷：缺嘴则没有 PCM 队列，缺手则没有 $H/f$，缺安全规格则会在接触发生后仍想重说一遍。Moshi 用双音频流省掉显式打断模块，帧结构决定 160 / 200 ms 量级的延迟；它没有手臂列。Qwen2.5-Omni 用 Thinker–Talker 和滑动窗口 DiT 做流式嘴；控制头状态表未公开。第 19 课用冻结 Nemotron 加 MiniMind Talker 做可训练桥，默认只验收回放。GR00T N1 与 $\pi_{0.5}$ 把慢规划和快控臂拆开，中间分别是 token 与子任务文本。π0 / ACT / OFT 把 $H/f$ 写成动作侧 freshness。Figure Helix 把 S2 7–9 Hz 与 S1 200 Hz 写进博文，并用训练时间偏移去对部署延迟。SafeVLA 把安全推进损失。这些方案各自解一列，或解训练，不解本课的并表契约。

GPT-4o 的公开材料停留在产品能力和部分延迟印象，没有给出可引用的 `PAUSE` / `REPLAN` 事件表。本课因此禁止把任何演示当作已经实现第 2 节的五条命题。Helix 同理：频率公开，状态表不公开。写“Helix 也是双时钟”可以，写“Helix 复现了本课接口”不可以。

**差距。** 缩小版没有麦克风、没有力传感器、没有 7B。规模差距：500 小时遥操作、L40 上的 63.9 ms、双嵌入式 GPU，钱和卡可以补。机制差距：两列时间戳、分通道事件、历史单调、`PAUSE` 与 `SAFE_HOLD` 分家，不买卡也必须做。缺这四条，把 Qwen 的嘴接到 π0 的手上，仍会出现一次撤销全部：停嘴时把已经合上的爪在日志里张开。另一处机制差距是三时钟：System 2 的规划周期 $\Delta T_2$、Talker 帧、动作 $1/f$。本课主验收只锁两列，第三列留给改造。

缩小版也缺回声消除和多机。Helix 博文演示双机递零食，靠语言指令分工。本课状态表按 `session_id` 可以复制两行，但没有写跨会话的接触权。不要在夹具里加第二只手臂还声称覆盖了那条演示。

**动手改造清单。**

1. **第三列规划时钟。** 在 `lesson_48.py` 增加 `plan_available_at_ms`，令 $\Delta T_2$ 为 $k/f$ 的整数倍，子目标过期时强制动作 `REPLAN`。预算：CPU，小于 1 人日。预期：System 2 暂停超过一个 $\Delta T_2$ 后，快列不得无限 `CONTINUE` 最后一块。失败判定：规划已过期，执行表仍在消耗旧 chunk。
2. **重叠推理加重叠合成。** 当前前缀执行到一半时启动下一次动作推理，同时 Talker 预合成下一句的前两帧。预算：CPU 或浏览器，不训网络。预期：$d>k/f$ 但 $d<H/f$ 时动作空档降到 0；`REPLAN` 后预合成帧全部取消。失败判定：预合成帧在 `REPLAN` 之后仍播放，或 in-flight 旧块被当成新块。
3. **力阈值扫一遍。** 给教具加标量 $F_i$，超限走 `SAFE_HOLD`。预算：浏览器，0.5 人日。预期：超限后接触保持、剩余步为零、PCM 仍按音频列走。失败判定：超限后已播清零，或杯子位置被重置。
4. **六分类控制头探针。** 若已有第 07 课控制头，把它扩成通道乘动词。预算：若有 1×24GB，用合成时间线过拟合 200 条，数小时。预期：杯子跳变时 `action=REPLAN` 的概率高于 `audio=PAUSE`；用户说“嗯”时相反。失败判定：两类扰动输出同一个 argmax，却把准确率写成 100%。

**顺手复现。** 第 07 课“`PAUSE` 保留 pending、`REPLAN` 取消未播”应能在本课音频列再现。第 30 课“`REPLAN` 后旧剩余步为零”应能在动作列再现。第 29 课“规划暂停后快环仍消费旧 $g$”应能在你加第三列之后再现。SafeVLA 的 83.58% 不能在 CPU 夹具上再现，方向也不该被改写成“切断后成功率上升”。Helix 的 200 Hz 不能在教具滑条上再现；滑条最高 50 Hz，而且没有 35 个关节。能复现的是协议方向，不是产品频率。

对照时还容易把第 20 课的图像 flow 和第 28 课的动作 flow 拉进来。可以拉，但只拉“积分步属于 $d$、不属于 $H$”这一句。[第 20 课](20_unified_understanding_generation.md)的流匹配出的是图像，不是关节，更不是 PCM。不要把图像生成的步数写成控制频率。π0 的 10 步欧拉积分、GR00T 的 $K=4$、Qwen2.5-Omni 的 DiT 滑窗，三者都是生成器内部迭代，一律记进延迟列。谁把 $K$ 写进 $H$，开环窗口会假性变长，过期判定会变松。本课改造清单第 2 条的重叠合成，也必须把预合成帧的墙钟写进 `play_at`，用旧 `branch_id` 过滤，而不是因为“已经算出来了”就准播。

若要把本课接口接到真实 Talker，最小增量是：现有 PCM 队列元素加上 `branch_id` 与 `play_at_ms`，播放循环先看 branch 状态再出队。最小增量不是：再训一个联合控制头。控制头可以继续用第 07 课的三分类，外面套一层通道路由器。路由器按事件来源打标签，比重新标注六分类更便宜。六分类是改造清单第 4 条，有卡再做。没卡时路由器加规则即可：VAD 且物体未动则走音频列；物体位移超过阈值且无新语音则走动作列；两者同时发生则写两条事件，音频在前。Lab 的两次点击就是这条规则的人肉版。

## 14. 论文与必读材料

1. [第 07 课](07_full_duplex_routing.md)与 [Moshi](https://arxiv.org/abs/2410.00037)、[Full-Duplex-Bench](https://arxiv.org/abs/2503.04721)。带着问题读：`available_at` 为什么不能用 token 下标代替？`PAUSE` 保留的到底是 KV 还是 pending PCM？Moshi 用双流省掉显式控制头之后，本课的通道字段还有没有存在理由？Full-Duplex-Bench 测的是嘴，怎样避免把它的数字填进手臂列？

2. [第 19 课](19_capstone_thinker_talker.md)与 [Qwen2.5-Omni](https://arxiv.org/abs/2503.20215)。带着问题读：19C 的 continue / pause / replan 作用在哪一段链路？双工回放为什么不能叫在线真双工？Talker 读隐表征时，Thinker 暂停会冻结什么？报告里的 40 ms 时间 ID、滑动窗口 DiT，分别属于 $d_{\mathrm{enc}}$ 的哪一段？哪一张表是本课需要、报告却没给的？

3. [第 29 课](29_dual_system_vla.md)与 [GR00T N1](https://arxiv.org/abs/2503.14734)、[π0.5](https://arxiv.org/abs/2504.16054)。带着问题读：10 Hz / 120 Hz 与 7–9 Hz / 200 Hz 差在哪一层？中间对象是 token、文本还是隐向量？暂停 System 2 之后，System 1 会不会自己发明下一阶段？把 $g$ 过期写成动作 `REPLAN`，要不要同时发音频 `PAUSE`？

4. [第 30 课](30_closed_loop_control.md)与 [ACT](https://arxiv.org/abs/2304.13705)、[Diffusion Policy](https://arxiv.org/abs/2303.04137)、[OpenVLA-OFT](https://arxiv.org/abs/2502.19645)、[π0](https://arxiv.org/abs/2410.24164)。带着问题读：$H/f$ 与音频块长何时数值接近、语义仍不同？`REPLAN` 后旧剩余步为零的断言，怎样原样搬到旧 PCM？整块执行的 OFT 若中途被语音打断，未执行的 $K-k$ 步走哪一个 `discard_reason`？

5. Figure 官方博文 [Helix](https://www.figure.ai/news/helix)。带着问题读：哪些数字可以引用（7–9 Hz、200 Hz、7B、80M、约 500 小时、35 DoF、训练时间偏移、双机同一套权重）？哪些格子必须写“博文未公开”（两列 `available_at`、力阈值、`PAUSE` 表、$H$、$d$ 的毫秒数）？共享内存隐向量对应本课的哪一个寄存器，不对应哪一个时间戳？“任务完成百分比”这个附加动作维，若接到本课状态表，应记进动作列还是另开一列终止条件？读完后用一句话说明：为什么本课不声称复现 Helix。若你同时打开了讨论 OpenHelix 的二次文献，只允许引用它承认尚未开源复现这一句，不要把二次文献里的架构猜测写进本课机制节。

6. [SafeVLA](https://arxiv.org/abs/2503.03480)。带着问题读：83.58% 与 +3.85% 测的是训练期约束，还是运行时切断？Safety-CHORES 能不能代替第 40 课的 $\|F_i\|>F_{\max}$？CMDP 的代价函数若不含“接触不可 undo”，接上本课状态表时还要补哪一条单调性？

读这六组材料时带一张空白的双列表。左栏只准填音频，右栏只准填动作。读完仍空着的格子写“未给出”，不要用另一篇的数字填上。Qwen 的 40 ms 不要填进 ACT 的 50 Hz 那一行。Helix 的 200 Hz 不要填进 GR00T 的 120 Hz 那一行。本课 CPU 的 20 Hz 和 320 ms 只属于教学夹具。

读完本课，手里应有一张能执行的状态表：一行计划，两列时间，三次合法动词，一次单独的力切断，两次禁止的历史撤回。第 47 课处理评测数字不能横着比；本课是第二波规格里的最后一课，停在协议本身。不要为了给课程一个更响亮的结尾，把这张表升级成“开源的具身 GPT-4o”。结尾只值一句事实：嘴和手承认同一份计划，却不再假装共用同一只钟。

阅读时建议按四天拆开，避免一天读完只记住“双时钟很重要”。第一天只读第 07 课的 `available_at` 和第 30 课的 $H/f$，手算 320 对 400 那一行。第二天读第 19 课 19C 和第 29 课 5.7 节的同构与失效，列出三处失效。第三天打开 Qwen2.5-Omni 第 2 节和 Helix 博文，把能引用的频率抄进对照表，空格子画叉。第四天跑 CPU 与 Lab，把 5.10 节时间线对到 `metrics`。任何一天都不允许把 Helix 的 200 Hz 填进本课 `f=20` 的格子。四天结束时，若还无法在白板上面出一行两列，就从第 07 课的 `StreamEvent` 再抄一遍字段，不要从产品演示倒推字段。

若只剩半小时，不要从 Helix 视频倒着读。打开 CPU 文件，从 `replay` 的 180 ms 与 360 ms 两条事件往回追：哪一列被改了，哪一个队列被清了，哪一个计数被禁止变小。能把这三问答给另一个人听，本课的接口说明书就算落地。视频留给有余力的晚上，而且只能用来核对博文里已经写明的频率，不能用来发明本课没写的字段。白板上一行两列画不出来时，先把 `audio_mode` 和 `action_mode` 写成两个方框，中间只连 `branch_id`，不要先画模型。模型可以后填，方框不能省。方框画好后再标两个时间戳的名字，名字必须带 `audio_` 或 `action_` 前缀。前缀丢了，第二天你自己也会把两列加回去。这半小时练习的产出是一张纸，不是一段观后感。纸上有字段、有 180 和 360 两个时刻、有“历史不降”四个字，就算交了短作业。若连纸也懒得画，至少把 CPU 的 `metrics` 抄下来：`old_pcm_played_after_replan` 与 `old_steps_after_replan` 必须是 0，`contact_occurred` 必须是 1，`event_count` 必须是 2。三个数对不上，先不要读第 13 节的改造清单。改造清单是给协议已经站住的人用的；协议没站住时，加第三列只会让错误长出一只新脚。三个数里最容易被忽略的是 `event_count=2`：有人把 `PAUSE` 和 `REPLAN` 写成同一次函数调用里的两个副作用，计数仍可能显示两个队列被清空，事件却仍是一条。本课验的是事件条数和通道，不是副作用清单的长度。副作用可以有两条，事件必须有两条。Lab 的事件日志就是给人眼数这两条用的：少一行就回去再点一次，多一行“撤销全部”就重置。数的时候看通道名字，不要只看按钮文案。文案写成“停一下”却打进动作列，日志仍然算错列。通道是字段，不是修辞。把这一句贴在 Lab 旁边，点按钮之前先念一遍，比把 Helix 频率背下来更接近本课要教的东西。念完再看四格揭晓：两列时间戳、剩余步、接触、旧未来泄漏。四格齐了，预测也选对了，Gate 才会有机会亮。预测选错时四格再漂亮也过不了，因为你还没承认两次事件是协议而不是界面习惯。先改预测，再点开始，不要指望改按钮顺序去迁就错误选项。选项是命题，按钮是操作。
