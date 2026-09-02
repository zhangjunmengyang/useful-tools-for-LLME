---
id: 36_audio_world_model
title: "声音是观察还是配乐"
summary: "模型会出声，就等于它听得懂动作造成的物理声吗？"
unit: frontier
play_tools: []
checkpoints:
  - "一份声音作为观测通道的对照笔记。"
  - "能指出音画生成器过不了动作对换的原因。"
---

# 第 36 课：声音是观察通道，配乐不是动力学

> 类型：机制实战（自己桌子上的视听对换）+ 只讲（AVWM 无公开训练代码；Veo / Sora 2 / Kling 无公开动作端口；Cosmos 3 音频推理 24GB 不够）<br>
> 建议周期：1-2 天<br>
> 硬件：笔记本或手机的摄像头、麦克风、喇叭即可；Mac / 纯 CPU 完成全部必做。单张 24GB 卡也不够装 Cosmos 3 Nano（16B），本课禁止为跑音频去装 16B / 64B。<br>
> 锚定：自己桌上的短视频与声轨。对照精读 AVWM（Wang、Zheng、Wu、Mao、Cheng，[arXiv:2512.00883](https://arxiv.org/abs/2512.00883) v4，标题以写课时最新版为准）与 Cosmos 3 报告音频节（NVIDIA，[arXiv:2606.02800](https://arxiv.org/abs/2606.02800)）。Reachy Mini 的麦克风与喇叭以官方媒体栈为准。第 30 课 DeskWM 当「完全不吃声音」的负对照。<br>
> 产物：两段画面几乎一样、声音不同的桌面视频；频谱图与能量检测记录；视听对换笔记；一份「声音作为观察通道」对照表

## 1. 这一课做什么

整门课的循环没有变：

```text
观察 先压成状态 再按动作预测下一状态 然后展开多条未来 给未来打分 最后选动作
```

你现在在第九幕。第 32 课已经毕业总装，第 33 课已经用 E0 到 E5 打过档。第九幕不改那把尺子，只补 2025-2026 年新公开的模态和配方。第 35 课把 Cosmos 3 的 MoT 拆开：语言、图像、视频、声音、动作可以进同一套权重，输入输出一换，它就从 VLM 变成视频生成器、世界模拟器或世界-动作模型。那一课留下一个没拆开的零件：模型会出声。

会出声很容易被写成「已经听懂世界」。本课要把这句话拆开。音画一起出，只说明解码器多了一条波形通道。声音作为部分可观察世界的观测，要求另一件事：同一段画面，换一段物理上说得通的声，下一秒的预测必须分岔。配乐、旁白、后期 BGM 可以让视频更好看，它们不是 $P(s_{t+1}\mid s_t,a_t)$ 里的动力学。

桌宠把这件事先钉死在硬件上。Reachy Mini 有四只 PDM MEMS 麦克风和一只 5 W 喇叭。麦克风是观察 $o_t$：杯子碰桌沿、键盘敲击、人出声，都从这里进状态。喇叭可以是动作 $a_t$：第 32 课的提示音、失败承认，走 `mini.media.push_audio_sample`。人在对面说话，仍是第 31 课的外生过程，你下不了命令。背景里循环的轻松音乐，什么通道都不是。

必做实验不装大模型。写课时检索过 AVWM 的公开训练代码，没有官方仓库可 clone，本课不许假装有。Cosmos 3 Nano 是 16B，官方推荐工作站级卡（如 RTX PRO 6000）或数据中心卡；主线 24GB 连权重都塞不满，禁止为「听一下环境声」去装。你要做的是：录或选用两段桌面视频，画面几乎一样，声音分别是杯碰桌沿和键盘敲击；用规则能量检测、频谱图、以及（若有）第 30 课的视觉世界模型，问它们会不会对换出不同的下一秒。大模型跑不动就据实写失败。失败本身就是本课的验收材料：视觉世界模型听不见，音画生成器往往没有「把声轨当观察再往前滚」的端口。

做完你能验证的是：能指出一段带环境声的视频为什么仍可能是 E0；能把麦克风、喇叭、配乐三笔记在桌宠的 POMDP 表上；能用自己的两段声轨完成一次声音版动作对换，哪怕对照系统全部失败。

术语速查：

| 术语 | 一句人话 |
|---|---|
| 视听观察 | 同一步既有画面 $o_t^{v}$ 又有一段声 $o_t^{a}$，二者时间对齐，都是对隐藏状态的残缺观测 |
| 配乐 / BGM | 后期加上、或与可见事件无因果关系的声音；听着像「这个世界」，并不进入动力学 |
| 声音版对换 | 钉死画面（和动作，如果有），只换声轨，看下一秒预测分不分岔；第 03 课动作对换的同构实验 |
| AVWM | 视听世界模型：把同步视听观测写成 POMDP，动作用于导航，预测下一刻的画面和声音 |
| AV-CDiT | AVWM 论文里的条件扩散 Transformer，视觉和听觉在共享注意力之后走各自的模态专家 |
| 双耳 / binaural | 左右声道分开的声；头一转，两耳响度差会变，这是空间里的观察，不是立体声混响特效 |
| 能量检测 | 用短时均方根判断「这一帧响不响」；能报事件，分不清杯子和键盘 |
| 喇叭动作 | 桌宠主动出声，是 $a_t$ 的一维，用来提示或招呼，不是给视频配 BGM |
| E0 | 第 33 课最低档：没有动作端口，或对换不分岔；带环境声的视频生成默认仍停在这里 |
| 只讲 | 无公开可跑权重、或 24GB 跑不动；读官方博客和报告，不装、不练、不把演示当复现 |

## 2. 问题

2025 到 2026 年，文生视频开始「原生带声」。Google 在 I/O 2025 的官方博文里把 Veo 3 写成第一次能生成带音频的视频：街景交通声、公园鸟叫、角色对白。OpenAI 在 2025 年 9 月 30 日的官方发布稿里把 Sora 2 写成旗舰视频与音频生成模型，带同步对白和音效。快手 Kling 的官方版本说明把 VIDEO 2.6 写成原生音频，可同时出人声、音效和环境声。这些系统把音画当成同一条视频。观感上，杯子落地会「叮」一声，键盘会「嗒嗒」。本课要处理的不是它们好不好看，是它们有没有把声音当成部分可观察世界的通道。

具体有四个经常被揉在一起的问题。

1. 生成声音，和观测声音，接口不同。生成器通常吃文本、首帧或一段无声画面，吐出视频加声轨。观测通道要求模型吃已经发生的声，用来更新对隐藏状态的信念，再预测下一秒。你把「杯碰桌沿」换成「键盘」，生成器若根本没有「输入声轨」这个端口，对换实验无法进行。无法进行不等于通过。
2. 同步不等于因果。Cosmos 3 自己的数据节写得很清楚：网络视频里的旁白常常在描述画面而不是由画面产生，后期 BGM 会盖住可见事件的物理声。他们在中期训练里用源分离、唇动打分、音乐比例，把对不上号的声扔掉。产品演示里「画面一动就有声」，仍可能只是统计上的同时出现。桌宠要的是：杯子碰到桌沿的那一帧，声的瞬态和接触事件是同一件事。
3. 第 03 课的动作对换仍然适用，只是被换的那一口可以是动作，也可以是作为观察的声。CarRacing 里惯性很大，无视方向盘也能把平均误差做得很体面。桌子更懒：连续几十帧杯子几乎不动，真正改写未来的是半秒接触。平均一步视频误差查不出「模型有没有用声音」。必须钉死画面，只换声，看预测分不分岔。
4. 第 33 课的 E0 不因为多了一条波形就升级。无动作端口的音画生成器，验证者仍是「像不像」。把「视频带环境声」写成已经通过物理理解评测，是把第 17 课的尺子用错了对象。Physics-IQ 一类基准测的是可见物理是否成立；Cosmos-SoundBench 测的是提示里的声事件在不在、是否对得上画面。两张表不能互相替。

界限先划清。本课是机制实战加闭源只讲。实战对象是你自己桌子上的两段视听和三条对照（能量、频谱、第 30 课视觉模型）。AVWM 的公式和导航设定要读懂，但不能复现论文表 4 的 SPL。Cosmos 3 的音频 VAE、SoundBench 数字只允许当报告摘录。Veo、Sora 2、Kling 只引用官方博客或技术报告，禁止编造它们的逐步动作端口。没有可跑的大模型时，对照失败必须写进笔记，不许改用聊天模型「描述下一秒会不会倒」来充数。

## 3. 准备

- [第 03 课](03_mdn_rnn_action_conditioned.md) 的动作对换要能口头复述：固定状态只换动作，预测必须分岔；平均误差查不出动作盲。本课把同一把审讯工具搬到声轨上。
- [第 12 课](12_frontier_landscape.md) 三问和 [第 33 课](33_embodiment_degrees.md) 的 E0 要熟。第一问答不上来，带多少环境声都还是画师。
- [第 30 课](30_desk_world_model.md) 若已经留下 `desk_wm.pt` 和一段桌面轨迹，本课拿它当负对照：它的 `fuse` 只拼视觉特征和动作嵌入，换声不该改变预测。没做过第 30 课也可以完成本课，负对照改成「只用当前画面猜杯子会不会倒」的肉眼基线。
- [第 31 课](31_interaction_memory.md) 的外生过程：人出声、人伸手，不是你的 $a_t$。本课的杯碰桌沿，常常也是人造成的外生接触；它进观察，不进你的动作空间。
- [第 32 课](32_ship_desk_pet.md) 的身体接口：M 档用 `mini.media.get_frame()` 拿画面，用 `mini.media.push_audio_sample` 出声。S 档用系统麦克风和喇叭即可。
- Python 3.10 或更新，独立虚拟环境。必装：`numpy`、`matplotlib`、`opencv-python`。选装 `sounddevice` 只在你要用本机麦克风现录时。不装 PyTorch 也能做完 Step 0 到 Step 4；Step 5 的视觉模型负对照才需要第 30 课那个环境。
- 系统里有 `ffmpeg` 最省事，用来抽声轨、混流、截取下一秒。没有也可以用手机分别导出视频和音频，再按第 7 节的纯 Python 路径读 wav。
- 硬件：摄像头对准桌子，一只杯子放在离桌沿几厘米处，一副键盘或笔记本按键在画面边缘。光线稳定。S 档足够。有 Reachy Mini 的人用它的麦克风阵列录音，用喇叭播提示音；没有真机不要买。
- 磁盘：两段各 3 到 5 秒的 720p 视频加 wav，总共几十 MB。
- 不要准备 Cosmos 3、Veo、Sora 2 的本地权重。24GB 不够就不要装，这一条写在准备里，免得做到一半去下 16B。
- 不要准备聊天模型来「听」这两段声再描写物理。那是在考语言模型的常识，不是在测世界模型的观察通道。

## 4. 学习目标

1. 在纸上把桌宠的声音拆成三笔账：麦克风是 $o_t$，喇叭可以是 $a_t$，配乐不是动力学；能指出人出声属于外生过程。
2. 写出视听 POMDP 的观察定义 $o_t=\{o_t^{v},o_t^{a}\}$，并说明为什么「模型会出声」不等于这条定义已经成立。
3. 把第 03 课动作对换改写成声音版：钉死画面，换杯碰桌沿和键盘两段声，说明什么叫分岔、什么叫听不见。
4. 读懂 AVWM 的设定：同步视听、低层导航动作、双耳声、用 SoundSpaces 2.0 做受控基准；能指出它的声源在数据里基本是锚在场景里的，和桌上会倒的杯子不是同一件事。
5. 读懂 Cosmos 3 报告里音频作为生成通道的做法（可见事件、声源移动、场景上下文），并把它和「声音作为输入观察」分开；24GB 不够时能拒绝安装。
6. 用自己的两段桌面视听跑完能量检测、频谱图、（若有）第 30 课视觉模型三条对照，填一张对照表；过不了对换的系统标成画师或听不见，不把「视频带环境声」写成物理理解已经过关。

## 5. 原理

五个机制。每个仍按这门课的节奏走：为什么需要、怎么运转、精确定义、代码或报告落在哪、怎么证明做对了。

### 5.1 麦克风是观察，喇叭可以是动作，配乐什么都不是

倒车时你会听引擎和轮胎，倒不是为了给画面配 BGM，是因为有些接触在画面里还没看清，耳朵先告诉你蹭到了。桌宠同一件事。杯子慢慢滑向桌沿，RGB 在 5 帧/秒的日志里可能还显示「杯子在桌上」；陶瓷碰到木沿的瞬态，麦克风一个窗就够。人在画面外喊你的名字，第 31 课的注视头若只看脸，会漏掉这一秒。

第 28 课把桌子写成 POMDP 时，观察 $z_t$ 已经列过 RGB、麦克风、关节角。当时麦克风多半是空列。本课把它写进公式。隐藏状态仍是物体、自己的身体、人：

$$
s_t=\bigl(s_t^{\mathrm{obj}},\,s_t^{\mathrm{self}},\,s_t^{\mathrm{human}}\bigr)
$$

同一步的观察是画面、本体感觉和一段短时声音：

$$
o_t=\bigl(I_t,\,q_t,\,m_t\bigr),\qquad m_t\in\mathbb{R}^{L\times C}
$$

$I_t$ 是摄像头帧，$q_t$ 是关节或假头部按键，$m_t$ 是长度为 $L$、通道数为 $C$ 的波形。笔记本麦克风通常 $C=1$；Reachy Mini 的固件把四只 PDM MEMS 混成立体声，官方媒体栈写明虽然有四麦，输出是 stereo。$m_t$ 是 $o_t$ 的一部分，用来更新对 $s_t$ 的信念，尤其是接触、材料、画面外声源、人是否出声。

喇叭是另一笔账。第 32 课把「提示」做成合法动作：预测伸手会碰杯，安全层改写为提示，S 档播警告音，M 档走 `mini.media.push_audio_sample`。这时波形是 $a_t$ 的执行结果，对方可能因此抬头。第 31 课已经强调：人会因为你出声而改变下一步，所以对人的预测可以是动作条件的；那是「你的喇叭动作进入了人的动力学」，不是「配乐进入了杯子的动力学」。

配乐是第三笔。桌上循环播放的列表、视频网站自动配的轻音乐、生成器按「氛围」编出来的垫乐，不由 $s_t$ 和 $a_t$ 产生，也不被桌宠的身体执行。把它喂进 $P(s_{t+1}\mid s_t,a_t)$，模型会学到「悲伤的弦乐之后杯子更可能倒」，那是剪辑习惯，不是牛顿。Cosmos 3 报告 3.2.2 节把这件事写成数据问题：旁白常常在描述视频而不是由视频引起，BGM 会盖住可见事件的物理声；他们在中期训练里把对不上号的声滤掉。产品宣传可以继续说「原生音频」，桌宠的账不能混。

类比失效处。倒车雷达的滴滴是工程师做的提示器，频率由距离规则决定，不是从数据里学来的世界模型。桌宠的喇叭若只按「人脸出现就叫」，那是第 32 课禁止的规则摆件。麦克风也不是免费的全知：宿舍空调、键盘自己的噪声、Reachy Mini 喇叭回流进麦克风，都会把 $m_t$ 变成几乎恒为 1 的能量维。第 31 课改造 3 已经警告过：能量维恒为 1 时 Brier 不变，失败就停，不要上语音识别来掩盖。

验证。立项时在 POMDP 表上加三行：麦克风 / 喇叭 / 配乐，分别填观察、动作、不入账。哪一行填错，后面所有「听懂了」都作废。

### 5.2 视听观测写成 POMDP，并不自动得到世界模型

AVWM 的贡献首先是把接口写清楚。论文 v4 标题是 *Audio-Visual World Models: Learning Physically Grounded Multisensory Dynamics*（v1 到 v3 改过标题，引用以你打开的版本为准）。环境被写成 POMDP 四元组 $(\mathcal{S},\mathcal{O},\mathcal{A},p)$。每一步隐藏状态是 $s_t$，智能体拿到的是残缺观察

$$
o_t=\phi(s_t)=\{o_t^{v},\,o_t^{a}\}
$$

其中 $o_t^{v}\in\mathbb{R}^{H\times W\times 3}$ 是一帧 RGB，$o_t^{a}\in\mathbb{R}^{L\times 2}$ 是一段双耳声。动作 $a_t=(u_t,\omega_t)$ 是低层自运动：平移和朝向变化。执行 $a_t$ 之后环境按 $p(s_{t+1}\mid s_t,a_t)$ 转移，再吐出新的画面和声音。

世界模型要学的是在动作条件下，对未来视听观察的预测。论文写成可跳步的形式：

$$
\hat{o}_{t+\Delta t}\sim p_{\theta}\bigl(o_{t+\Delta t}\mid o_{t-m+1:t},\,a_{t\rightarrow t+\Delta t},\,\Delta t\bigr)
$$

$o_{t-m+1:t}$ 是 $m$ 帧视听上下文。$a_{t\rightarrow t+\Delta t}$ 是从当前到目标时刻的相对运动（位移加净转角），$\Delta t$ 是帧偏移。这样模型不只做下一帧，还能在不同地平线上问「若我这样走，声音和画面会一起怎么变」。

这和「一段带声的视频」差在三处。第一，观察是同步的一对，缺声或声与画面错位，POMDP 的 $\phi$ 就没定义完。第二，动作是低层、可执行、持续时间相同的离散控制，不是文本提示。第三，预测的是执行动作之后的下一刻视听，不是给已有画面贴音效。

AVW-4k 把这三处做成受控基准。数据采集在 Matterport3D 场景上，用 SoundSpaces 2.0（Chen 等，[arXiv:2206.08312](https://arxiv.org/abs/2206.08312)）做几何声渲染：反射、吸收、混响随房间和听者位置变。每个环境里放一个位置已知的声源，循环播放铃声。智能体边走边录：RGB $128\times 128$，每帧对齐 0.15 秒、16 kHz 双耳声。动作空间四格：前进 0.15 m、左转 $10^{\circ}$、右转 $10^{\circ}$、停下。轨迹从三种运动模式里采样：接近声源、远离声源、与声源无关。全数据集约 30 小时、76 个室内场景、4500 条轨迹，训练/验证/测试按 6:1:2 划且场景不重叠。

受控是刻意的。把声源钉死、把源信号固定成铃声，变化就主要来自智能体运动、视角、几何和声传播。论文自己写：这样是为了把预测误差归因到世界动力学，而不是声源自己在乱响。桌宠的杯子不是铃声。杯子会倒、会空、会被手拿开，声源会动，事件不是循环的。AVWM 的公式能搬，AVW-4k 的统计不能直接当桌面验收。

补充材料里还有一张数据集对照表，维度包括：是否同步视听、是否物理一致、真机还是仿真、动作类型、适不适合 AVWM。AudioSet 被标成不适合，因为常有配音、BGM、与画面无因果的成分，也没有低层动作。PLAICraft 有视听和低层动作，但麦上的玩家语音混不出去。Landscape 物理声干净，却没有动作。这张表就是本课第 1 节那句话的文献版：带声不等于观测通道。

验证。能在纸上画出 $o_t^{v}$、$o_t^{a}$、$a_t$ 三个箭头进预测器，缺任何一口都要说出来缺的是观察、动作还是解码。能指出 AVW-4k 的铃声源和桌上的杯子事件差在哪。数字以 v4 PDF 为准，课文不把表 4 的 SPL 抄成你复现的结果。

### 5.3 声音版对换：同一画面，两段声，下一秒必须分岔

第 03 课在 CarRacing 上证明过：赛道惯性大，一个完全无视方向盘的模型靠「下一帧约等于这一帧」也能把平均误差刷体面。桌子比赛道更懒。本课的接触窗口往往只有一两帧。若你拿整段视频的像素误差或 FVD 来评「有没有用声音」，那一两帧会被稀释掉。

审讯工具原样搬过来，只改被换的那一口。记一段长度为 $k$ 的历史，画面固定为 $\bar{I}_{t-k:t}$。准备两段等长的声 $m^{(1)}$、$m^{(2)}$，物理含义不同：陶瓷碰桌沿，和键盘敲击。若模型声称吃视听观察，它必须给出两份下一刻预测 $\hat{o}_{t+1}^{(1)}$、$\hat{o}_{t+1}^{(2)}$。分岔指数可以很粗：

$$
\Delta_{\mathrm{audio}}=\bigl\|\,\hat{s}_{t+1}(\bar{I},m^{(1)},a)-\hat{s}_{t+1}(\bar{I},m^{(2)},a)\,\bigr\|
$$

状态 $\hat{s}$ 在桌宠上至少包括杯子是否还在桌内、是否越过安全边界。没有状态头时，退化为「下一秒画面里杯子中心的位移」或「你自己用笔打的会不会倒」。$\Delta_{\mathrm{audio}}$ 相对「真实一步里杯子位移」要明显大于噪声。两份预测几乎重叠，就叫听不见。

这和经典动作对换是同构的，被换的通道不同：

| 实验 | 钉死什么 | 换什么 | 问什么 |
|---|---|---|---|
| 第 03 课方向盘 | $z_t$ 与 LSTM 隐状态 | 左转 / 右转 | 下一 $z$ 分不分岔 |
| 第 30 课四键 | 桌面视觉特征 | 看左 / 伸手 | 杯子和视角分不分岔 |
| 本课声音观察 | 画面（和动作，如果有） | 杯沿声 / 键盘声 | 下一秒杯子分不分岔 |
| AVWM 导航动作 | 视听历史 | 左转 / 前进 | 下一帧画面和双耳声分不分岔 |
| 喇叭动作 | 当前桌面状态 | 出声 / 静音 | 1 秒后人是否看过来 |

前三行是「观察或动作进了预测没有」。第四行才是 AVWM 论文主实验那种低层动作条件。第五行是第 31、32 课已经有的外生过程：喇叭是你的动作，人是外生。不要把五行收成一句「多模态」。

为什么生成器常常过不了这一关。文生视频加原生音频的典型端口是：文本、可选首帧、可选参考图，输出一整段音画。没有逐步 $a_t$，第 12 课第一问出局。也没有「把一段已发生的声当作条件，再预测下一秒画面」的接口，声音版对换无从下手。你若只是把模型吐出的声轨事后换成键盘，画面已经生成完了，对换发生在播放器里，不发生在模型里。过不了对换，就还是画师。第 33 课写过：参数量、分辨率、像不像，升不了 E0。只有声音的视频生成仍可能是 E0。本课把那句话落到音画产品上。

负对照必须有。只听能量的规则器会对两段瞬态都报警，分岔在「响了」，不分岔在「杯子会不会倒」。第 30 课的 DeskWM 根本不读 $m_t$，换声之后 $\Delta_{\mathrm{audio}}=0$，这是听不见的金标准，不是 bug。频谱图若能分开陶瓷和塑料按键，只说明信号里有材料信息，不说明任何网络用了它。三条对照都过不了「下一秒杯子」这一问，笔记就写：本课没有可跑的视听世界模型，信号层有差异，预测层没有。

### 5.4 AV-CDiT：共享注意力，分模态专家，三阶段才把声加进去

AVWM 的实现叫 AV-CDiT，从视觉导航世界模型 NWM 的 CDiT（Bar、Zhou、Tran、Darrell、LeCun，[arXiv:2412.03572](https://arxiv.org/abs/2412.03572)）扩到双模态。视觉帧和听觉段先分别过冻结编码器，得到 $z^{v}$、$z^{a}$（视觉用 Stable Diffusion 的 VAE，听觉用在 AVW-4k 上训的 SoundStream 类 tokenizer，并去掉一块以适配更短的 0.15 秒片段）。两路经适配层投到共享空间，拼成目标序列 $X_{t+\Delta t}=[h^{v}_{t+\Delta t}:h^{a}_{t+\Delta t}]$，再按扩散加噪。

条件向量把相对运动 $a_{t\rightarrow t+\Delta t}$、时间偏移 $\Delta t$、扩散步 $k$ 加成一个 $c_t$，用 AdaLN 调制。每个块里，视觉 token 和听觉 token 共享自注意力和交叉注意力，用来对齐时间和语义；前馈层则拆成模态专家：共享注意力之后，两路走各自的非线性，再拼回去。论文的动机写得很硬：从视觉预训练模型扩过来时，视觉会压住听觉，专家加分阶段训练是为了让耳朵有自己的容量。

训练目标是标准的噪声预测，两路 L2 相加：

$$
\mathcal{L}_{\mathrm{simple}}=\mathbb{E}_{k,\epsilon_{v},\epsilon_{a}}\bigl[\|\hat{\epsilon}_{v}-\epsilon_{v}\|_{2}^{2}+\|\hat{\epsilon}_{a}-\epsilon_{a}\|_{2}^{2}\bigr]
$$

外加协方差的变分项。前向加噪对两路独立，反向由同一个网络一起解。相关关系留在干净样本和共享注意力里，不靠把两路噪声绑死。

三阶段。第一阶段只训视觉：自注意力、交叉注意力、视觉专家、视觉适配和视觉头。第二阶段冻结共享注意力和视觉部分，只训听觉专家、听觉适配和听觉头。第三阶段视听拼接，整网微调。论文补充实验写：若从第二阶段起解冻全部层，即使调参也会发散；第二阶段把注意力冻住，是为了保住已经学到的空间知识。

评测分两轴。保真：视觉用 LPIPS、DreamSim、PSNR；音频用对数谱距离 LSD 和频谱 SSIM。物理一致性用三条代理：ILD Error（左右耳响度差），AV-Lag Error 和 AV-Corr Error（视觉运动能量序列和音频能量序列的互相关峰的时延与幅度）。后两条用真实画面当参照去评生成的声，避免「生成的画面和生成的声一起偏了，看起来还很同步」。这和本课第 7 节用真实下一秒杯子位置当参照，是同一纪律。

规划实验把 AV-CDiT 接到连续视听导航（SoundSpaces 2.0 上的 AV-Nav）：智能体用第一人称 RGB 和双耳声找声源，在距目标 1 m 内发 stop 算成功。规划器从预训练策略里采样候选动作，用世界模型做 $k$ 步 lookahead，用预测的进度奖励加策略价值打分，束搜索后执行第一条动作。训练场景和导航测试场景不重叠。论文报告在合适的束宽和地平线下，步数下降、效率指标上升；oracle 世界模型（用真环境反馈替换预测）还有余量。具体 SPL 以 v4 表 4 为准，本课不抄成你的结果。

代码落点。写课时用网页检索 AV-CDiT / AVW-4k / 作者名加 github，没有官方训练仓库可指向。NWM 的官方实现在 [facebookresearch/nwm](https://github.com/facebookresearch/nwm)，那是视觉 CDiT，不是本课的视听模型，24GB 训 1B 也不是本课作业。本课可执行的落点是第 7 节你自己的脚本，和第 30 课 `DeskWM.fuse`：那里只有 `visual_dim + act_emb`，没有听觉维。缺的那一口就是本课要你看见的。

### 5.5 Cosmos 3 的声音：生成通道、过滤配乐、仍不是桌宠的麦克风

第 35 课已经把 Cosmos 3 的 MoT 说成：AR 子序列负责语言和理解，扩散子序列负责图像、视频、声音、动作的去噪。本课只补音频这一口。报告 2.1.2 节：生成用的音频 VAE 来自其参考文献中的音频 VAE 架构，立体声 48 kHz，hop 1920 采样点，约每秒 25 个 token，训练时冻结，线性层投进 Transformer 隐空间。位置编码上，音频只走时间轴，$h=w=0$，用绝对时间调制把 25 Hz 的 hop 和视频 VAE 的时间压缩对齐到同一物理秒。

支持的生成模式里，Text-to-Video 可以在噪声视频 token 后面接噪声音频 token，写成 $[\mathbf{S}_{\mathrm{AR}},\,\tilde{v}_{1:N},\,\tilde{s}]$。Image-to-Video / Video-to-Video 同样可选地出声。官方项目页把这件事写成：Cosmos 3 能从文本、图像或片段生成物理上说得通的视频，并让声音跟着可见事件、声源移动和场景上下文。这是生成通道：输出多了一条波形。

数据节比产品页诚实。预训练从视频池里抽出 1.389 亿条带可用声轨的片段，什么都有：剧情声、非剧情对白、旁白、BGM、环境、音乐、物理事件。中期训练滤到 1880 万条：1280 万非语音（环境和物理声），600 万唇动同步的语音。过滤原则三句：语音只在和可见人脸同步时保留；非语音例子里去掉画外语音；非乐器的 BGM 会压过目标声时去掉。工具链包括 SAM-Audio 分轨、SyncNet 唇动、语音/音乐比例、视觉模型判断画面里有没有乐器。他们明确知道：配乐不是物理。

评测用 Cosmos-SoundBench：从 FoleyBench 抽 144 条非语音提示，覆盖环境、撞击、物体交互、工具、车辆、水等。指标 AVQ 把语义视听正确（提示里的声在不在、是否对得上可见源、是否随动作变）和制作质量各一半。报告表 15 里，Cosmos3-Nano 的 AVQ 为 7.34，SAV（语义视听）为 8.35；闭源对照里 Seedance-1.5-Pro 的 AVQ 最高（7.64），主要高在制作质量；Veo-3.1 的 AVQ 为 7.45；Sora 2 的 AVQ 为 6.90。这些数字是 NVIDIA 用自己的法官协议打的生成真，不是 Physics-IQ，不是动作对换，不是你复现的。图 20 用锤子打击的帧和频谱瞬态对齐做定性说明：接触帧有尖峰，非接触帧没有。定性演示可以支持「他们在优化同步」，不能支持「已经通过物理理解评测」。

硬件分档必须写死。官方 GitHub 模型族表把 Sound 标在 Super（64B）和 Nano（16B）的输出栏；Edge（4B）的输出是 Text / Image / Video / Action，没有 Sound。Nano 的推荐硬件是 RTX PRO 6000 / H100 / B200 这一档，不是主线 24GB。16B 的 BF16 权重本身大约 32GB，24GB 卡装不下。NVIDIA 开发者博客把 Nano 写成工作站级推理。本课的规则：24GB 不够就不要装。Diffusers 文档里对带 `sound_tokenizer` 的检查点可以设 `enable_sound=True`，那是给够显存的人看的命令，不是本课 Step。

和 AVWM 的差别。Cosmos 3 可以把声音当输出，也可以在扩散子序列里看到音频 token；报告同时支持动作条件下的未来视频（前向动力学）。这不等于桌宠已经有一条「麦克风波形 $\to$ 信念 $\to$ 下一秒杯子」的通道。把 Nano 当桌宠世界模型，仍要过第 12 课三问和第 30 课那种对换。本课不在 24GB 上假装问过。

### 5.6 闭源音画产品只讲端口，不讲你没见过的动作

Veo。Google 官方 I/O 2025 博文（2025-05-20）把 Veo 3 写成首次能生成带音频的视频，举例是街景交通声、公园鸟叫、角色对白。DeepMind 产品页把 Veo 3 / 3.1 写成原生生成音效、环境声和对白，条件是文本、图像、视频，能力列表包括镜头控制、首尾帧、物体增删。公开材料里的控制是提示词、参考图、相机运动这类创作控件。没有逐步关节或「执行前进 0.15 m」这种动作端口写在官方页上。本课不编。

Sora 2。OpenAI 官方发布稿（2025-09-30）把 Sora 2 写成旗舰视频与音频生成模型，带同步对白和音效，并回指 2024 年 2 月那篇 *Video generation models as world simulators*。第 12 课已经用三问把无逐步动作的 Sora 类系统从决策意义的世界模型里划出去。Sora 2 多了原生声，第一问的端口没有因此出现。OpenAI 帮助中心后来写明 Sora 产品和 App 于 2026 年 4 月 26 日停用，API 另有截止日期；引用以你打开的帮助页为准。本课把它当只讲的产品史，不当前可练权重。

Kling。官方站点把 VIDEO 2.6 的发布说明写成原生音频，可端到端出人声、音效和环境声；VIDEO 3.0 产品页继续把 native audio 写成卖点。同样没有公开的逐步动作端口可引用。禁止把「提示词里写向前走」当成 $a_t$。

三条产品共用一句验收：官方若只展示文本或图像到音画，就按 E0 或「未知」记，不按世界模型记。Cosmos 3 报告表 15 把它们拉进 SoundBench 对照，评的是生成真。你没有这些模型的逐步动作接口文档，就不要在笔记里写「它们其实听动作」。

机制对桌宠的直接推论。Reachy Mini 的四麦是真传感器，喇叭是真执行器。听杯子碰桌，决定出不出声，是观察和动作。给一段桌面录像配 BGM，是后期。两者可以在同一只喇叭里出现，账必须分开。第 32 课的提示音若改成循环歌单，安全层就从「查询后的动作」退化成氛围灯。

## 6. 源码导读

本课没有可 clone 的 AVWM 训练仓库。写课时检索 arXiv:2512.00883 摘要页、论文 HTML，以及 AV-CDiT / AVW-4k / 论文标题加 github，没有官方训练代码链接可跟随。因此本节不发明路径。可执行的代码三块：你第 7 节要跑的桌面脚本；第 30 课已经存在的 `DeskWM`；Reachy Mini 官方 SDK 里已经用过的媒体接口。Cosmos 3 只读报告和官方模型卡，不读一份本课编出来的 `audio.py`。

### 6.1 第 30 课 `DeskWM.fuse`：缺的就是听觉维

[第 30 课](30_desk_world_model.md) 的胶水脚本把冻结 DINOv2 的 patch 和四键动作拼在一起：

视觉特征 $z_t\in\mathbb{R}^{N\times D}$，动作嵌入 $e(a_t)\in\mathbb{R}^{d_a}$，在 patch 维上广播后拼接。预测器是因果 ViT，损失在特征空间。输入列表里没有波形，没有频谱，没有 RMS。同一段历史换声轨，只要画面和动作标签不变，`forward` 的张量就不变，$\Delta_{\mathrm{audio}}$ 精确为 0。这不是实现 bug，是接口定义：它是视觉动作条件模型，听不见。

把这一段当负对照，比空谈「应该加音频 token」有用。若你要在同一骨架上为声音留口，最小改法是：对 $m_t$ 算对数梅尔频谱或直接用短时 RMS 加 16 个频带能量，经一层线性得到 $e_{\mathrm{aud}}\in\mathbb{R}^{d_m}$，与 $e(a_t)$ 一起广播到 patch 上再拼接。那是第 11 节的改造，不是本课必做训练。必做只要求你证明：当前这只模型换声不分岔。

对照官方 DINO-WM 仓库 `gaoyuezhou/dino_wm` 的 `models/visual_world_model.py`：它同样在视觉特征上做动作条件预测，没有听觉分支。不要把官方 DINO-WM 改造成 AVWM，那不是他们的论文。

### 6.2 Reachy Mini 的媒体栈：四麦进观察，喇叭出动作

M 档的接口在 [第 32 课](32_ship_desk_pet.md) 已经核对过，本课只标声音这一侧。Pollen 官方媒体栈博文（Hugging Face，2026-06-10，*Eyes, ears, and a voice: building Reachy Mini's media stack*）写明：麦克风是定制版 Seeed reSpeaker XVF3800，四只 PDM MEMS，硅胶垫隔离，XMOS 语音处理芯片；固件把四麦混成立体声再给你。喇叭 5 W / 4 Ω。Seeed 的 Reachy Mini 文档同样列 4 麦阵列和 5 W 喇叭。SDK 侧：

- `mini.media.get_frame()` 返回 `(H, W, 3)` 的 `uint8`，这是 $I_t$。
- 录音走官方音频设备（USB 音频，XVF3800）。本课不指定一个未经核对本机才存在的方法名去「拿四路原始 PDM」；你拿到的是固件混好的立体声，把它当 $m_t\in\mathbb{R}^{L\times 2}$。
- `mini.media.push_audio_sample(...)` 非阻塞播放 `float32` 波形。这是 $a_t$ 的执行。第 32 课排错过：立刻返回会把提示音切掉，必须按采样率和样本长度自己 `sleep`。

四麦阵列在硬件上能做波束和方向。世界模型若只用混好的立体声能量，等于把空间观察压成「响不响」。ILD（左右耳或左右混音通道的响度差）是 AVWM 用的空间代理，桌宠上可以当探针，不是已经实现的声源定位。不要把 Seeed 宣传里的「声源定位」写成你的 $s_t$ 里已经有方位角。

S 档没有这套芯片。系统默认麦克风就是 $m_t$，系统喇叭就是 $a_t$。公式不改。

### 6.3 AV-CDiT 对着论文读，不对着虚构仓库读

把第 5.4 节的数据流在纸上画成五段，作为「源码导读」的替代物。编码器冻结；适配层可训；共享注意力；模态专家前馈；两路噪声头。三阶段对应三组 `requires_grad`。你若将来见到官方代码，验收它是否真的在第二阶段冻住了共享注意力：论文补充材料说解冻会发散。

基线实现也写在论文里，仍然没有本课可跑的权重。视觉支路用 DIAMOND 或 NWM，音频支路用微调过动作条件的 AudioLDM，再拼起来；联合生成基线是给 AVDiT 注入动作全局条件。论文的结论是：模态拼接和「naive」联合生成都不如 AV-CDiT 的 ILD / AV-Lag / AV-Corr。本课把这句话当阅读结论，不当你复现的表。

NWM 官方仓库 [facebookresearch/nwm](https://github.com/facebookresearch/nwm) 可以当视觉祖先打开看 CDiT 的条件注入（相对运动、$\Delta t$、扩散步）。本课不要求按其 README 训练，也不把 NWM 的下一帧 RGB 当成视听世界模型。

### 6.4 Cosmos 3 音频只读报告口径

官方代码在 [github.com/nvidia/cosmos](https://github.com/nvidia/cosmos)，检查点在 Hugging Face 的 `nvidia/cosmos3` collection。本课 24GB 不装，因此不给一条会在你机器上 OOM 的推理命令当必做。读报告时对准四处：

1. 2.1.2 节：48 kHz、hop 1920、约 25 token/s、VAE 冻结。
2. 2.2.2 节：T2V+Audio 的 token 布局 $[\mathbf{S}_{\mathrm{AR}}, \tilde{v}_{1:N}, \tilde{s}]$。音频在扩散子序列里，排在视觉之后、动作之前。
3. 3.2.2 节：预训练 1.389 亿带声片段，中期 1880 万，BGM 和画外语音被当成污染。
4. 6.2.3 节与表 15：SoundBench / AVQ 是生成真，法官协议用了多家闭源 MLLM。那是他们的评测，不是第 17 课的 Physics-IQ。

NVIDIA 开发者博客的模态表把「Action | Video | Text $\to$ Video」标成动作条件世界模型，把「Text | Image $\to$ Video」标成预测。音频出现在生成能力里。Edge 档官方表不含 Sound。读到这里就够拒绝「桌宠默认换 Cosmos 出声」。

Hugging Face Diffusers 的 Cosmos 3 文档提到：检查点带 `sound_tokenizer` 时，视频调用可设 `enable_sound=True`，再用 `encode_video(..., audio=result.sound, ...)` 混进 MP4。这是给显存够的人的 API 形状，证明官方确实把声当可选输出。本课不把它写进 Step 命令。

## 7. 实验

必做全部在 CPU / Mac 上完成。目标不是训出 AV-CDiT，是做一次声音版对换，并让三条对照系统露出它们各自听不见的方式。目录建议：

```text
learn-wm-audio36/
  clips/raw_cup.mp4
  clips/raw_key.mp4
  clips/silent.mp4
  clips/cup.wav
  clips/key.wav
  out/spec_cup.png
  out/spec_key.png
  out/swap_cup.mp4
  out/swap_key.mp4
  NOTES.md
  audio_swap.py
```

`text` 围栏只放目录，不放命令。

### Step 0: 体检环境

确认 Python 库和（可选）ffmpeg。两条命令分开跑。

```bash
python -c "import numpy, matplotlib, cv2; print('numpy', numpy.__version__, 'cv2', cv2.__version__)"
```

预期：打印版本号，不报 `ModuleNotFoundError`。缺什么就在当前虚拟环境里装什么，不要为这一步去装 PyTorch。

```bash
ffmpeg -version
```

预期：第一行带 `ffmpeg version`。没有 ffmpeg 的人跳过所有 `ffmpeg` 围栏，改用系统录音 App 分别导出 wav，脚本仍能读。

### Step 1: 录两段画面几乎一样、声音不同的视频

剧本固定，3 到 5 秒，摄像头不要动。

1. 杯子放在离你这边桌沿约 5 到 8 厘米处，键盘或笔记本按键出现在画面一侧，手从画面外伸到杯附近，停住。
2. 第一段 `raw_cup.mp4`：用另一只手或一根笔轻碰杯沿，发出清脆的陶瓷/玻璃声。杯子可以微动，但这一段里不要让它掉下去。碰完静止 1 秒。
3. 第二段 `raw_key.mp4`：手走到几乎同一位置，改为敲键盘两三次。杯子尽量别动。
4. 若你还能再录 1 秒「真的下一秒」：杯沿那段之后轻轻把杯推过桌沿（下面垫书，不要砸显示器），键盘那段之后杯子留在原处。这两秒分别存成 `future_cup.mp4`、`future_key.mp4`，后面当参照。没有也不废实验，用笔在 `NOTES.md` 写你认为的下一秒。

手机录的人，把两段导入电脑。Reachy Mini 档用头部相机录画面，用板载麦收音，注意喇叭不要同时放歌。

抽声轨（有 ffmpeg 时）。每条围栏一条命令。

```bash
ffmpeg -y -i clips/raw_cup.mp4 -vn -ac 1 -ar 16000 -acodec pcm_s16le clips/cup.wav
```

```bash
ffmpeg -y -i clips/raw_key.mp4 -vn -ac 1 -ar 16000 -acodec pcm_s16le clips/key.wav
```

做一条无声画面，后面互动要用。从杯沿那段去声：

```bash
ffmpeg -y -i clips/raw_cup.mp4 -an -c:v copy clips/silent.mp4
```

若两段画面时间轴差得明显（手的位置差一截），不要硬混。本实验的诚实前提是「视觉前缀几乎一样」。差太多就重录，不要靠裁剪自欺。

在 `NOTES.md` 写四行：杯子到桌沿的大致厘米数、两段是否同一机位、碰杯时杯子有没有已经明显倾斜、键盘声是否盖过了环境。

### Step 2: 落盘对照脚本

把下面保存为 `audio_swap.py`。它做四件事：读 wav 画对数频谱；算短时 RMS 和 90 分位阈值上的「击发」；把两段声的频谱质心和击发时刻写成 json；可选地读第 30 课检查点，对两段视频的同一视觉前缀做一步预测并比较。没有检查点就跳过最后一块。

```python
"""audio_swap.py  第 36 课：频谱 / 能量 / 可选视觉 WM 负对照。"""
from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_wav(path: Path):
    with wave.open(str(path), "rb") as w:
        nch = w.getnchannels()
        sr = w.getframerate()
        n = w.getnframes()
        sw = w.getsampwidth()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        raise SystemExit(f"只支持 16-bit pcm，收到 sampwidth={sw} @ {path}")
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
    return sr, x


def rms_series(x, sr, win_ms=20.0, hop_ms=10.0):
    win = max(1, int(sr * win_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    if len(x) < win:
        return np.array([float(np.sqrt(np.mean(x * x) + 1e-12))])
    frames = 1 + (len(x) - win) // hop
    out = np.empty(frames, dtype=np.float64)
    for i in range(frames):
        sl = x[i * hop : i * hop + win]
        out[i] = np.sqrt(np.mean(sl * sl) + 1e-12)
    return out


def spectral_centroid(x, sr, nfft=1024):
    n = min(len(x), nfft * 8)
    spec = np.abs(np.fft.rfft(x[:n] * np.hanning(n), n=n))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    den = spec.sum() + 1e-12
    return float((freqs * spec).sum() / den)


def save_spec(path, sr, x, title, out_png):
    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=False)
    t = np.arange(len(x)) / float(sr)
    ax[0].plot(t, x, color="0.2", lw=0.6)
    ax[0].set_title(title)
    ax[0].set_ylabel("amp")
    nfft = 512
    ax[1].specgram(x, NFFT=nfft, Fs=sr, noverlap=nfft // 2, cmap="magma")
    ax[1].set_xlabel("s")
    ax[1].set_ylabel("Hz")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def onsets(rms, frac=0.9):
    thr = float(np.quantile(rms, frac))
    hits = np.where(rms >= thr)[0]
    return thr, [int(i) for i in hits[:12]]


def visual_prefix_hash(video: Path, n=8):
    cap = cv2.VideoCapture(str(video))
    frames = []
    ok = True
    while ok and len(frames) < n:
        ok, im = cap.read()
        if ok:
            small = cv2.resize(im, (64, 64))
            frames.append(small.astype(np.float32))
    cap.release()
    if not frames:
        raise SystemExit(f"读不到帧: {video}")
    arr = np.stack(frames, 0)
    return arr, float(arr.mean()), float(arr.std())


def maybe_desk_wm(args, report):
    ckpt = Path(args.desk_ckpt)
    if not ckpt.is_file():
        report["desk_wm"] = "skip: 没有第 30 课检查点，视觉模型负对照未跑"
        return
    import torch

    blob = torch.load(ckpt, map_location="cpu")
    report["desk_wm"] = {
        "ckpt": str(ckpt),
        "note": "检查点在。本课默认不改 DeskWM 的 forward；换声不改变其输入，预测差应为 0。",
        "keys": list(blob.keys())[:8] if isinstance(blob, dict) else type(blob).__name__,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cup", default="clips/cup.wav")
    p.add_argument("--key", default="clips/key.wav")
    p.add_argument("--cup_mp4", default="clips/raw_cup.mp4")
    p.add_argument("--key_mp4", default="clips/raw_key.mp4")
    p.add_argument("--out", default="out")
    p.add_argument("--desk_ckpt", default="")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sr_c, cup = read_wav(Path(args.cup))
    sr_k, key = read_wav(Path(args.key))
    if sr_c != sr_k:
        raise SystemExit(f"采样率不一致: cup {sr_c} key {sr_k}，请用同一条 ffmpeg 命令抽")

    save_spec(Path(args.cup), sr_c, cup, "cup rim", out / "spec_cup.png")
    save_spec(Path(args.key), sr_k, key, "keyboard", out / "spec_key.png")

    rms_c = rms_series(cup, sr_c)
    rms_k = rms_series(key, sr_k)
    thr_c, on_c = onsets(rms_c)
    thr_k, on_k = onsets(rms_k)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(rms_c, label="cup")
    ax.plot(rms_k, label="key")
    ax.axhline(thr_c, ls=":", color="C0", label="cup 90q")
    ax.axhline(thr_k, ls=":", color="C1", label="key 90q")
    ax.legend()
    ax.set_ylabel("RMS")
    fig.tight_layout()
    fig.savefig(out / "rms.png", dpi=140)
    plt.close(fig)

    arr_c, m_c, s_c = visual_prefix_hash(Path(args.cup_mp4))
    arr_k, m_k, s_k = visual_prefix_hash(Path(args.key_mp4))
    n = min(len(arr_c), len(arr_k))
    vis_l2 = float(np.linalg.norm((arr_c[:n] - arr_k[:n]).reshape(-1)) / n)

    report = {
        "sr": sr_c,
        "cup_centroid_hz": spectral_centroid(cup, sr_c),
        "key_centroid_hz": spectral_centroid(key, sr_k),
        "cup_rms_mean": float(rms_c.mean()),
        "key_rms_mean": float(rms_k.mean()),
        "cup_onset_frames": on_c,
        "key_onset_frames": on_k,
        "visual_prefix_l2_per_frame": vis_l2,
        "visual_mean_cup": m_c,
        "visual_mean_key": m_k,
        "energy_detector": "两段都有高能量击发" if (on_c and on_k) else "至少一段没有击发，重录或降阈值",
        "spectrogram_split": (
            "质心相差超过 400 Hz，频谱层能分开"
            if abs(spectral_centroid(cup, sr_c) - spectral_centroid(key, sr_k)) > 400
            else "质心很近，材料差可能不够，看 spec_*.png 是否仍有条纹差异"
        ),
    }
    maybe_desk_wm(args, report)
    (out / "swap_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

### Step 3: 跑频谱和能量，先看信号层

```bash
python audio_swap.py --cup clips/cup.wav --key clips/key.wav --cup_mp4 clips/raw_cup.mp4 --key_mp4 clips/raw_key.mp4 --out out
```

预期：`out/spec_cup.png`、`out/spec_key.png`、`out/rms.png`、`out/swap_report.json` 生成；终端打印质心、RMS、击发下标、视觉前缀 L2。陶瓷碰沿通常有更宽的高频，键盘更靠近中高频的窄脉冲，但不保证。质心差很小就看图，不要为了「分开」去给其中一段加后期均衡。

`visual_prefix_l2_per_frame` 用来体检「画面几乎一样」。没有绝对阈值；若肉眼已经看出手的位置差了半个杯子，重录。本课不靠这个 L2 宣称两段视觉状态相同，它只是防你拿完全不同的两段视频来玩对换。

把 `energy_detector` 那一行抄进 `NOTES.md`。规则能量检测的合法结论只有两种：两段都响，或一段没录上。它没有「杯子会倒」这一维。

### Step 4: 无声画面切换两条声轨，先猜再揭晓

这是课纲里的互动，用播放器完成，不需要网站组件。

```bash
ffmpeg -y -i clips/silent.mp4 -i clips/cup.wav -c:v copy -c:a aac -shortest out/swap_cup.mp4
```

```bash
ffmpeg -y -i clips/silent.mp4 -i clips/key.wav -c:v copy -c:a aac -shortest out/swap_key.mp4
```

操作顺序必须写进 `NOTES.md`，否则不算做完：

1. 先静音播放 `clips/silent.mp4`，写下一句：下一秒杯子会不会倒。只许写会、不会、不确定。
2. 播放 `out/swap_cup.mp4`，再写一句预测。允许改，必须注明改了没有。
3. 播放 `out/swap_key.mp4`，再写一句。
4. 若有 `future_*.mp4`，现在才打开对照。没有就保持「未揭晓」，不要事后补一个你希望的答案。

人自己就是一台视听世界模型。本步要你体验的是：声轨一旦被当成观察，信念会不会更新。更新了，说明通道在你脑子里存在；模型若更新不了，它就还没把 $m_t$ 接进 $s_t$。

### Step 5: 三条系统对照，诚实写失败

在 `NOTES.md` 填这张表。预测对象统一成「下一秒杯子是否越过桌沿」。答案只许：分岔 / 不分岔 / 无此端口 / 未跑。

| 系统 | 输入里有声吗 | 换声后下一秒杯子 | 你实际看到的 |
|---|---|---|---|
| 规则能量检测 | 有，RMS | 通常不分岔（两段都击发） | 抄 `energy_detector` |
| 频谱图（人眼读） | 有 | 人可以分材料，不自动给出杯子轨迹 | 抄质心和 `spec_*.png` 观察 |
| 第 30 课 DeskWM | 无 | 无此端口，预测不变 | 有 ckpt 则跑下面命令，没有则写 skip |
| 音画生成器（Veo / Sora 2 / Kling / Cosmos 出声） | 官方端口是文本或画面到音画 | 无「把声当观察再滚一步」的公开端口 | 只讲，填无此端口 |
| AV-CDiT | 论文有 | 本课无权重 | 未跑 |

有第 30 课检查点时，把路径传进去，确认脚本只记录「未改 forward」：

```bash
python audio_swap.py --cup clips/cup.wav --key clips/key.wav --cup_mp4 clips/raw_cup.mp4 --key_mp4 clips/raw_key.mp4 --out out --desk_ckpt /path/to/desk_wm.pt
```

把 `/path/to/desk_wm.pt` 换成你第 30 课真实的检查点路径。脚本不会对 DeskWM 做前向对换，避免在不明状态字典上瞎调用。打开第 30 课的 `desk_wm.py`，确认 `fuse` 没有音频维，把这一行抄进笔记：`DeskWM.fuse` 的拼接是视觉加动作，换 `cup.wav` / `key.wav` 不能改变张量。这就是负对照。不要为了让表格好看去改 `fuse` 再训一轮冒充「已经听见」。

若你有闭源产品的网页试用，只允许做一件事：上传同一段无声画面，看它生成的声像杯子还是像键盘。那是「画面到配乐」，不是本课的对换。把结果标成生成真，标成只讲，不许写进「下一秒杯子分岔」。

### Step 6: 选做，24GB 停

显存不够的人到此结束。不要去 `pip install` Cosmos 3、不要去拉 16B / 64B。Edge 档官方输出栏没有 Sound，装 4B 也听不到本课要的那条通道。

显存到工作站级、并且你自愿读官方文档的人，只允许按 [nvidia/cosmos](https://github.com/nvidia/cosmos) 当前 README 和 Diffusers Cosmos 3 页的命令做一次「出不出声」的冒烟，记下分辨率、步数、是否 `enable_sound`。禁止把这次生成写成动作对换，禁止把 SoundBench 表 15 的数字抄成你的分数。冒烟失败（OOM、没有 `sound_tokenizer`、许可证墙）写进 `NOTES.md` 即停。

## 8. 配置与预算

必做实验的预算按「一下午」估，不按卡时估。

| 项目 | 建议 | 不够时 |
|---|---|---|
| 数据 | 两段 3 到 5 秒 720p，单声道 16 kHz wav | 手机 1080p 也行，脚本会把画面缩到 64×64 只为体检前缀 |
| 采集时间 | 含重录 30 到 60 分钟 | 第一遍两段画面差太多就重拍，不要靠裁 |
| 计算 | CPU / Mac 数秒出频谱 | 无 |
| 第 30 课负对照 | 读 `fuse` 五行，0 GPU | 没有 ckpt 就 skip，不要现训 |
| 磁盘 | 小于 200 MB | 不要把 Cosmos 权重算进来 |
| 真机 | 可选 Reachy Mini 麦和喇叭 | S 档系统麦即可 |

超参只有能量窗：默认 20 ms 窗、10 ms hop、90 分位当击发。宿舍很吵时 90 分位会把空调算进去，改看 `rms.png` 再动手调分位，不要一上来做自适应阈值网络。

大模型分档，写进笔记以免临时改主意：

| 系统 | 本课档位 | 24GB | 主线 |
|---|---|---|---|
| 自己的两段视听 + `audio_swap.py` | 实战 | 够 | 必做 |
| 第 30 课 DeskWM 当听不见对照 | 实战（复用旧产物） | CPU 可 | 有就做 |
| AVWM / AV-CDiT | 只讲 | 论文在 8×A100 上训 | 不装 |
| Cosmos 3 Nano 16B 出声 | 只讲 | 不够 | 不装 |
| Cosmos 3 Super 64B | 只讲 | 不够 | 不装 |
| Cosmos 3 Edge 4B | 只讲 | 官方输出无 Sound | 不装来听 |
| Veo 3 / 3.1、Sora 2、Kling 2.6 / 3.0 | 只讲 | 无公开逐步动作端口 | 不把网页试用当对换 |

检查点：本课不产生神经网络权重。要留的是 `out/swap_report.json`、三张 png、两条 `swap_*.mp4`、`NOTES.md`。随机种子对规则脚本无意义；若你日后给 DeskWM 加听觉维再训，种子和配置跟第 30 课走。

## 9. 验收

量化线只有三条，都落在你自己的文件上。

1. 存在两段 wav 和对应 mp4，`visual_prefix_l2_per_frame` 已记录，`NOTES.md` 写了机位是否相同。没有「画面几乎一样」这句话的主观确认，对换无效。
2. `out/swap_report.json` 里能量检测和频谱质心都有数；`NOTES.md` 的对照表五（或六）行填完，未跑的格子写未跑，不许空着让读者猜你是不是跑过 Cosmos。
3. Step 4 的三句预测（无声 / 杯沿声 / 键盘声）写在揭晓之前。时间倒错的笔记按未完成计。

可视检查：`spec_cup.png` 和 `spec_key.png` 能让另一个人指出「这两段不是同一段录音」。若两张图像复制粘贴，重录。`swap_cup.mp4` 和 `swap_key.mp4` 画面同一条，声不同；用播放器切轨时口型或手的动作不得跳帧，否则混流裁短了。

概念检查，口头能答：

- 麦克风、喇叭、配乐三笔账。
- 为什么 DeskWM 换声 $\Delta_{\mathrm{audio}}=0$。
- 为什么 Veo 原生带声仍可能是 E0。
- AVW-4k 的铃声源和桌上会倒的杯子差在哪。
- Cosmos-SoundBench 的 AVQ 评的是哪一种「好」。

失败也验收。三条对照都不给出「下一秒杯子分岔」，只要表填了、原因写了（无端口 / 只报响度 / 无权重），本课通过。用聊天模型补一段「陶瓷声之后杯子会倒」的文字，验收失败。

## 10. 排错

| 症状 | 原因 | 验证 | 修法 |
|---|---|---|---|
| `wave.Error` 或 `sampwidth` 退出 | 抽出来的不是 16-bit PCM | `ffprobe clips/cup.wav` 看 codec | 重跑 Step 1 的 `ffmpeg`，加 `-acodec pcm_s16le` |
| 两段质心几乎相等、频谱也像 | 其实都是同一只手拍桌子，或自动增益把瞬态压平 | 听原始 wav，看时域是否削顶 | 离麦克风近一点，关掉「增强录音」类音效，重录 |
| 能量检测两段都密密击发 | 空调、风扇、宿舍人声抬高了分位 | `rms.png` 基线是否离 0 很远 | 换安静时段；或只比较击发之后 200 ms 的频谱，不调网络 |
| 能量检测一段全 0 | 视频没音轨，或 ffmpeg 抽成空 | 播放器里听 `raw_*.mp4` | 用系统麦单独录 wav，再混流 |
| `visual_prefix_l2` 很大 | 机位动了，或自动曝光把「举手」学成「画面变亮」 | 并排看前 8 帧 | 锁曝光、固定脚架、重录 |
| 混流后画面和声对不上 | `-shortest` 切到了较短那条 | 看两段时长 | 先把 wav 和视频都裁到同一秒数再混 |
| Reachy Mini 录音全 0 | macOS 上 XVF3800 未唤醒，官方 issue 记录过静音 | 对麦说话看电平 | 按 Pollen 媒体栈博文和当前 GitHub issue 处理，本课可退回系统麦 |
| `push_audio_sample` 听不见 | 非阻塞调用被下一条指令切掉 | 日志里播放时长 | 第 32 课的 `sleep`，本课不要把喇叭回流进对换录音 |
| 想装 Cosmos 3 立刻 OOM | 16B BF16 权重大约 32GB | `nvidia-smi` 看 24GB 已满 | 卸载，写 skip。禁止改用 CPU 硬推 16B 当必做 |
| 网页生成器「听起来很像杯子」 | 生成真，不是观察通道 | 你有没有把 `cup.wav` 当输入 | 记入只讲行，不要填分岔 |
| 用 Whisper 转写两段声再比文本 | 任务被换成 ASR | 转写结果不包含「会不会倒」 | 删掉。本课禁止语音识别充数 |
| 两段画面差太多仍继续 | 对换条件不成立 | 前缀 L2 和肉眼 | 停，重录。错误数据上的「不分岔」没有信息量 |

## 11. 前沿与改造

前沿怎么做。同一问题在 2025-2026 年公开系统里至少三条路。第一条，AVWM 把视听写成动作条件 POMDP，用 SoundSpaces 2.0 造受控双耳数据，用模态专家加三阶段训练对抗视觉压制听觉，再用 lookahead 去改善连续视听导航。它的诚实边界写在结论里：合成室内、声源基本锚住、真实世界带精确动作标签的双耳数据稀缺。第二条，Cosmos 3 把音频放进 MoT 的扩散子序列，作为和视频对齐的生成通道，中期训练专门过滤 BGM 和画外语音，用 SoundBench 测提示遵循和同步。它有动作生成模式，但音频在官方叙事里首先是「跟着可见事件出声」，不是桌宠麦克风。第三条，Veo / Sora 2 / Kling 把音画做成一条内容产品，官方评测是偏好、对齐、视听同步，没有逐步动作对换协议公开。

我们差在哪。规模上，没有 30 小时双耳轨迹，没有 1880 万过滤后的音视频，24GB 也装不下 Nano。这是钱和数据，本课不装成已经跨过。机制上，有三件本课教的东西前沿也还经常分开卖：观察通道和生成通道、配乐和物理声、对换和「像不像」。AVWM 把前两件在仿真里接起来了，第三件做在导航动作上，没做在「杯子会不会倒」上。Cosmos 3 把生成通道和配乐过滤做认真了，桌宠麦克风仍要你自己接。音画产品可以把第三件一直不做，仍然刷生成真。

动手改造清单。每条都可单独做，失败就停。

1. 给第 30 课 DeskWM 加 16 维频带能量。改 `fuse`：对与当前帧对齐的 200 ms 窗做 `rfft`，按对数间隔收成 16 个数，线性映到 `act_emb` 同宽，拼进 patch。数据：本课两段不够，按第 30 课协议再采 20 分钟，一半片段在伸手时碰杯发声，一半只敲键盘。预算：CPU 数小时或 24GB 上不到一小时，冻结 DINO。预期：声音版对换的特征 L2 从 0 变成与「真实一步杯子位移」同量级。失败：模型用频带去认「哪一段是训练里的杯沿 clip」（时钟泄漏），把动作打乱后分岔仍在。出现这种结果就加第 30 课那种动作置零负对照，并在测试时把声轨时间乱切。
2. ILD 探针，不做生成。有 Reachy Mini 或任意立体声麦时，在 `audio_swap.py` 里对左右通道分别 RMS，记 $\mathrm{ILD}=20\log_{10}(\mathrm{RMS}_L/\mathrm{RMS}_R)$。人站在左边喊、右边喊各 10 次。预算：一小时。预期：符号随方位变。失败：固件已经把四麦混成几乎单声道，ILD 恒近 0。失败则停，不要开始写 MUSIC 算法。
3. 喇叭动作接第 31 课的 1 秒注视头。动作空间加「短促提示音 / 静音」，标签仍是 1 秒后是否看镜头。预算：重采 10 分钟，CPU 重训小头。预期：出声后看镜头概率升，静音对照不升。失败：喇叭回流进麦，能量维把「自己在叫」当成「人在叫」。加一条：播放时不把麦写进特征，或减掉已知的播放波形。
4. 把 Cosmos 3 报告的中期过滤规则，手工打在你自己的 20 分钟桌面数据上。四类标签：接触物理声、人声对口型、画外语音、BGM。预算：两小时标注。预期：BGM 段上任何「会倒」预测都不可信，训练时丢掉。失败：宿舍里四类分不清（歌单加键盘）。分不清就只保留你自己敲出来的接触声。

顺手复现的方向映射。

- 论文：视觉会压住听觉，所以要模态专家和阶段 2 冻结注意力。缩小版：改造 1 里把音频维尺度乘 0.01，对换分岔应塌掉；乘太大则视觉动作条件变差（第 30 课四键分岔下降）。同向即「容量被视觉占满」在桌面特征上成立；反向则写明 16 维能量太弱，谈不上压制。
- 论文：用真实视频作参照去评生成音频的时延，避免「错得一致」。缩小版：Step 4 必须先猜后揭晓；禁止用模型自己生成的下一秒画面去证明自己的声同步。
- 论文：BGM 污染物理声。缩小版：改造 4 的四类标签。你若在 BGM 开着时训改造 1，对换应变差。

不要做的改造：把两段 wav 喂给任意聊天模型让它输出「杯子会倒」；把 Veo 网页试用的同步音效写成 E4；为 Edge 4B 没有 Sound 去改官方权重。

## 12. 论文与延伸

1. AVWM（Wang, Zheng, Wu, Mao, Cheng，[arXiv:2512.00883](https://arxiv.org/abs/2512.00883) v4，2026-08-04）。*Audio-Visual World Models: Learning Physically Grounded Multisensory Dynamics*。POMDP 视听观察、AVW-4k、AV-CDiT、连续视听导航 lookahead。阅读问题：公式 (1) 里的 $a_{t\rightarrow t+\Delta t}$ 是逐步动作还是一段相对运动？为什么要把声源钉成循环铃声才能谈「物理一致性」？补充材料把 AudioSet、PLAICraft 划掉，依据是配乐、画外语音还是缺低层动作？写课时若标题又改，以你打开的 abs 页为准。
2. Cosmos 3（NVIDIA，[arXiv:2606.02800](https://arxiv.org/abs/2606.02800)）。报告 2.1.2、2.2.2、3.2.2、6.2.3 节与表 15、图 20。项目页把音频写成跟随可见事件、声源移动、场景上下文。阅读问题：音频 token 在序列里是观察还是要被去噪的目标？中期过滤删 BGM，说明他们把哪一种声当成监督、哪一种当成污染？AVQ 里 SAV 和 PQ 各惩罚什么，为什么 Seedance 可以 AVQ 更高但 SAV 不必最高？Edge 档没有 Sound，对桌宠选零件意味着什么？
3. SoundSpaces 2.0（Chen, Schissler 等，[arXiv:2206.08312](https://arxiv.org/abs/2206.08312)）。按网格几何做可在线的声渲染，AVWM 的数据引擎。阅读问题：仿真器给的是「特权几何上的物理声」，学到的世界模型还要不要在未见过的房间里泛化？双耳 ILD 在仿真里几乎无噪声，真机 Reachy Mini 的混音立体声还剩多少空间信息？
4. Navigation World Models（Bar, Zhou, Tran, Darrell, LeCun，[arXiv:2412.03572](https://arxiv.org/abs/2412.03572)），代码 [facebookresearch/nwm](https://github.com/facebookresearch/nwm)。CDiT、相对运动条件、跳步预测，AV-CDiT 的视觉祖先。阅读问题：NWM 的观察里没有声，第一问仍可能通过；加上双耳之后，哪些误差会从「视觉惯性」变成「声源方位」？不要在本课按他们的 README 开训。
5. SoundSpaces: Audio-Visual Navigation in 3D Environments（Chen 等，[arXiv:1912.11474](https://arxiv.org/abs/1912.11474)）。连续视听导航任务的前身，AVWM 规划实验的下游。阅读问题：成功条件是走到声源附近并 stop，这和「预测杯子会不会倒」共享哪一条（动作条件的未来），不共享哪一条（目标是声源位置，不是接触动力学）？
6. Listen to Look into the Future（Lai, Ryan, Jia, Liu, Rehg，[arXiv:2305.03907](https://arxiv.org/abs/2305.03907)）。第 31 课已引用：第一人称视听注视预判，音频在 Ego4D / Aria 上各约 +2 点 F1。阅读问题：那篇把声当观察去预报人的注意，本课把声当观察去预报物体接触，状态里多了哪一块、少了哪一块？为什么第三人称桌面麦不能直接搬他们的空间融合？
7. OpenAI，*Sora 2 is here*（[openai.com/index/sora-2](https://openai.com/index/sora-2/)，2025-09-30）与 *Video generation models as world simulators*（2024-02）。只讲。阅读问题：2024 年那篇用「世界模拟器」称呼视频生成器，2025 年的 Sora 2 加了同步对白和音效，第 12 课第一问的端口有没有因此出现？官方帮助中心后来的停用日期如何影响「可复现」？
8. Google，*Fuel your creativity with new generative media models and tools*（[blog.google I/O 2025](https://blog.google/technology/ai/generative-media-models-io-2025/)，2025-05-20）与 DeepMind Veo 产品页。只讲。阅读问题：官方能力列表里的相机控制和物体增删，是 $a_t$ 还是创作控件？把「真实世界物理」写在宣传句里时，本课要求你追问哪一个实验（对换，而不是样片）？
9. Kling 官方 VIDEO 2.6 原生音频发布说明（[kling.ai release notes](https://kling.ai/release-note/release-notes/c605hp1tzd)）与产品页 native audio 表述。只讲。阅读问题：端到端出人声、音效、环境声，缺了 POMDP 里的哪一口？提示词里的「向前走」为什么不能填进 $a_t$？

选读：Ha 与 Schmidhuber 的 World Models（2018）把 V 的输入写成像素；本课等于问 V 要不要再读一段 $m_t$。第 17 课的三分评测把生成真和预测准分开；SoundBench 属于前者。第 33 课 E0 的是否题 1：对换是否分岔。带环境声的视频若答否，档位不升。

下一课第 37 课进入训练配方：teacher forcing、扩散或 flow matching 的双向去噪、把双向教师蒸馏成因果生成器的 Self-Forcing 一族。Cosmos 3 的 Generator 为什么还要后训练才能当交互世界模型，会在那里拆。本课只要求你记住：配方再巧，若声音只出现在输出端，桌宠的麦克风仍然是空的。毕业标准仍在第 32 课的五件行为和第 33 课的档；第九幕不加一条「会出声就算具身」。

