---
id: 34_leaderboard_map
title: "刷榜地图：三份榜各测什么"
summary: "2025-2026 年世界模型榜上的第一名，测的是生成、续写、指令跟随，还是规划？"
unit: frontier
play_tools: []
checkpoints:
  - "一张开源三档 × 三分评测的地图，没测过的格子写未测。"
  - "能说明 WorldScore 第一名不必能给桌宠做 MPC。"
---

# 第 34 课：榜上第一名测的通常不是规划好

> 类型：研究 + 体验（clone 三份评测仓、按 README 列出入口命令、抽公开样本走评分维度；不跑完整榜）<br>
> 建议周期：2 天<br>
> 硬件：Mac / 纯 CPU 可完成 clone、读样本、人工打分和填表；单张 24GB 不够跑闭源视频模型和完整 Cosmos 评测<br>
> 锚定仓库：[haoyi-duan/WorldScore](https://github.com/haoyi-duan/WorldScore)、[WorldModelBench-Team/WorldModelBench](https://github.com/WorldModelBench-Team/WorldModelBench)、[google-deepmind/physics-iq-benchmark](https://github.com/google-deepmind/physics-iq-benchmark)<br>
> 产物：一张系统地图（开源三档 × 三分评测 × 三份榜 × 24GB）、三份榜的入口命令笔记、人工走过的样本打分表

## 1. 这一课做什么

第八幕把桌宠装上了身体，[第 32 课](32_ship_desk_pet.md) 用五件行为证明循环在转，[第 33 课](33_embodiment_degrees.md) 用 E0 到 E5 给系统打了具身程度。毕业标准停在那里：第 32 课仍是总装，第 33 课仍是档。第九幕不改这两条线。

第九幕要加的零件是读榜。2025 到 2026 年，WorldScore、WorldModelBench、Physics-IQ 把「世界模型」三个字写进了论文标题和排行榜。产业新闻会跟着写 SOTA。桌宠这边真正要的，仍是主干循环里那一句：

```text
观察 先压成状态 再按动作预测下一状态 然后展开多条未来 给未来打分 最后选动作
```

三份榜测的通常是这句话的前半截，而且经常把动作换成文本、首帧或相机轨迹。榜上的第一名可以在生成、续写、指令跟随上很高，[第 03 课](03_mdn_rnn_action_conditioned.md) 的动作对换和第 32 课的克制仍然可以失败。本课把每条榜还原成四件事：输入是什么、输出是什么、评分器看什么、样本怎么来的。做完你要能口头说明：WorldScore 第一名不必能给桌宠做 MPC。

上一课留下的是档，不是新模型。这一课不加新骨架，只加三把公开尺子，并强制和 [第 12 课](12_frontier_landscape.md) 的三问、[第 17 课](17_evaluating_world_models.md) 的三分评测、第 33 课的 E0 到 E5 对表。下一课是 Cosmos 3：同一套权重怎么同时吃语言、图像、视频、声音和动作。本课先把「官方表上的分」和「桌宠用得上的分」分开，否则 Cosmos 3 的宣传页会把你重新卷回名字战争。

工作分两块。第一块是体验：clone 三个评测仓库，按当前 README 抄入口命令，每份榜抽 2 到 3 条公开样本，用纸和笔走一遍评分维度。不跑完整榜。样本量和闭源模型都不够，24GB 也喂不动那些要评的大视频模型。第二块是研究：给至少十个你已经见过或马上会见到的系统填一张地图。没上过榜的格子写「未测」，只在官方仓库或论文表里出现过的数字写「官方表」或「宣称」，不要写成你测过。

术语速查：

| 术语 | 一句人话 |
|---|---|
| 世界生成 | 按指令连续做出下一场景；WorldScore 把它当成考题，不等于 $P(s_{t+1}\mid s_t,a_t)$ |
| 下一场景任务 | 给定当前场景、下一场景说明和相机布局，模型吐下一段视频 |
| 三分评测 | 第 12、17 课的三根尺子：预测准、生成真、规划好 |
| 三问筛选器 | 第 12 课的资格审查：动作起作用吗、离开视野的东西还在吗、有没有人用它选过动作 |
| 开源三档 | 可跑（代码权重齐）/ 仅权重（有推理脚本、无训练代码）/ 纯 demo（连权重都没有） |
| 指令跟随 | WorldModelBench 的一维：生成视频有没有把文本（和首帧）里的动作做完 |
| 物理续写 | Physics-IQ 的任务：看真实实验的开头，往后编 5 秒，再和真实后半段比 |
| 物理方差 | 同一实验拍两遍之间的差别；Physics-IQ 把它归一成满分 100 |
| 只讲 | 本课不跑权重、不训练；读论文和仓库文档，把宣称和可验证事实分开写 |
| 官方表 | 论文或仓库 README 里公布的数字；抄的时候必须标明出处，不能写成你复现的结果 |

## 2. 问题

「SOTA 世界模型」四个字现在同时指三件不同的事。WorldScore 问的是：按相机轨迹走出新场景，画面稳不稳、听不听布局。WorldModelBench 问的是：把视频生成器当作世界模型用时，常识、指令、物理哪条先坏。Physics-IQ 问的是：真实物理实验的后 5 秒，模型和摄像机录到的是不是同一件事。三件事都可以很高，桌宠要的「伸手会不会把杯子扫下桌」仍然可以没有定义。

本课要同时堵住四条交白卷的路。

第一，把官方宣传的数分钟一致性写成你测过。Genie 3 的博客可以宣称，第 12 课已经规定：无权重就标宣称。本课沿用，并且推广到三份榜的第一名：你没有按协议生成那三千条或一百九十八段视频，就没有资格改写官方表上的数字。

第二，把世界生成和动作条件动力学写成同一件事。WorldScore 的「动作」经常是相机矩阵加一句「向左推」，不是桌宠的转头或伸手。相机听话，只说明布局可控，不说明 $a_t$ 换了之后未来会分岔。

第三，把物理续写写成规划好。Physics-IQ 的答案是真实录像，评分器看运动发生在哪里、何时、多少、像素差多少。没有 agent，没有奖励，没有「选一个动作再回真环境验收」。续写对，只说明预测器在这把尺子上接近摄像机，不说明策略能用它选动作。

第四，把尺子本身当成不会漂的地面真值。Physics-IQ Verified（Rädsch 等，arXiv:2606.18943）审计了原榜的提示、伪激活和汇总方式，六成样本被改过，排名会动。本课把这件事当成评测课的复习：量法即立场，第 17 课已经写过。

一条界限先划清：本课比较的是「这三份榜各自在量哪一种好」，不回答「2026 年哪个世界模型最强」。后一个问题缺统一环境、统一动作端口和统一验证者，问出来就不成立。这句话要写进你的地图笔记。

## 3. 准备

- 会口头复述第 12 课三问和第 17 课三分评测。翻得出第 03 课动作对换和第 32 课五件行为的笔记更好；翻不出来也能做本课，但填表时「规划好」那一列只能写未测。
- 读过第 33 课第 1 到 5 节。生成真可以把 E0 测得很漂亮，规划好在可重置环境里可以把 E2 测得很漂亮，都不能代替「模型是否在真身体的动作回路里」。
- git、Python 3.10 左右、能打开网页。完整评测环境（WorldScore 的 DROID-SLAM、Grounding-SAM、SAM2，WorldModelBench 的 VILA 评判器，Physics-IQ 的 ffmpeg）本课不要求装齐。
- 磁盘：三个仓库的代码本身不大。WorldScore 数据集、Physics-IQ 的 396 段 4K 录像、闭源模型的生成视频都很大，本课默认不下完整数据。要抽样本，用项目页、仓库里的 `worldmodelbench.json`、Physics-IQ 的 `descriptions/descriptions_original.csv` 即可。
- 一张 24GB 卡不是本课的及格条件。有卡也不要用它去刷完整榜：协议要求的生成量、闭源 API 和显存都不在主线预算里。
- 选读打底：第 17 课第 11 节已经把 WorldScore 写成「没有 agent、没有奖励」的那一行。本课把它展开，并补上另外两份 2025 年的榜。

## 4. 学习目标

1. 把任意一份「世界模型榜」还原成输入、输出、评分器、样本来源四列，并指出它对应第 17 课的哪一根尺子。
2. 用第 12 课三问给 WorldScore、WorldModelBench、Physics-IQ 各答一遍：哪一问根本没出题。
3. 能说明 WorldScore 的相机轨迹、WorldModelBench 的文本指令、Physics-IQ 的续写提示，为什么都不能代替第 03 课的动作对换。
4. 能说明 Physics-IQ Verified 改了提示、伪激活和汇总之后，排名为什么会动，以及这件事对「抄官方表」意味着什么。
5. 按当前 README 列出三份仓的评测入口命令，并知道哪一条在仓库里和 README 文字不一致。
6. 填完十个系统的地图：未测就写未测，宣称就写宣称，24GB 跑不了就写跑不了。
7. 能口头回答：桌宠以后看到 SOTA，先问测的是哪根尺子；WorldScore 第一名不必能给第 32 课的克制当模拟器。

## 5. 原理

五个机制。前两个是旧尺子回访，后三个把三份榜拆开。每个走完：为什么需要、怎么运转、精确定义、代码落点、怎么验证。

### 5.1 先把第 12 课三问和第 17 课三分评测接到榜上

一段以假乱真的卧室平移视频，和一个能回答「如果我现在伸手，杯子会不会下桌」的系统，观感可以一样，性质是两种东西。第 12 课用三问把名字还原成性质：

1. 动作起作用吗？同一个当前状态，喂两个不同动作，预测必须分岔。
2. 离开视野的东西还在吗？转身回来，杯子还得是那只杯子。
3. 有没有人用它选过动作，并且回到真环境验收过？

第 17 课把「好」拆成三根可以互相矛盾的尺子。预测准量模型和真实动力学的距离，一步误差和多步漂移。生成真量画面像不像，人眼、LPIPS、FVD。规划好量拿模型支撑决策之后，真环境里的分数。MuZero 生成真为零、规划好极强；糊的 Dreamer 重建照样能跑 walker。所以三根尺子必须分开打，不能折成一个总分再去认第一名。

三份 2025 年的榜各自只覆盖这张表的一角。WorldScore 覆盖生成真，外加「听相机和文本」的可控性，规划好整轴缺席。WorldModelBench 覆盖生成真里的常识，外加指令跟随和一批物理违规检查，规划好仍缺席。Physics-IQ 覆盖预测准里很苛刻的一种：真实实验的后续必须对上摄像机，仍然没有 agent。把任何一份榜的第一名读成「最好的世界模型」，就是在用局部尺子给全集打分。

桌宠上的对应很硬。第 32 课的克制要查询 $P(s_{t+1}\mid s_t,a_t)$，再决定伸不伸手。三问的第一问是动作对换，第三问是查询进了动作出口。E4 的证据是日志里看得到这次查询。榜上的生成真可以把 E0 抬得很漂亮，第 33 课已经禁止这种升级。

### 5.2 把一条榜还原成输入、输出、评分器、样本

评测协议是一份把「怎么量」写死的约定。第 17 课对你自己的三个模型写过：数据哪来、上下文几帧、往前滚几步、怎么归一化、怎么汇总。公开榜只是把同一套手续印成论文。读榜时先填四格：

- 输入：模型推理时真正吃进去的东西。文本、首帧、多帧、相机矩阵、低层动作，五者不是同一种条件。
- 输出：评分器看到的东西。几乎总是一段视频。特征空间的下一状态、动作分布、安全层的截断记录，这三份榜都不收。
- 评分器：人、微调过的视觉语言模型、几何一致性、光流 IoU、像素 MSE。裁判换了，排名就可以换。
- 样本怎么来的：谁写的提示、谁拍的录像、有没有第二遍对照、难例是不是事后挑的。

这四格填完，三问和三分评测才有落点。输入里没有逐步动作，第一问直接出局。输出只有视频、评分器只看画面，规划好没有定义。样本若是「看起来像物理的互联网视频」，Physics-IQ 那种受控实验就测不到。

形式化一下。设协议为四元组 $(\mathcal{X},\mathcal{Y},d,\mathcal{D})$。$\mathcal{X}$ 是条件，$\mathcal{Y}$ 是模型输出，$d$ 是评分器，$\mathcal{D}$ 是测试集。榜上的分是

$$
S(g)=\frac{1}{|\mathcal{D}|}\sum_{(x,y^\star)\in\mathcal{D}} d\bigl(g(x),y^\star\bigr)
$$

WorldScore 的 $x$ 是当前场景加布局，没有 $a_t$。WorldModelBench 的 $x$ 是文本加可选首帧，$y^\star$ 其实是人的细粒度标签，不是未来真帧。Physics-IQ 的 $y^\star$ 是摄像机后 5 秒。桌宠要的协议是 $x=(s_t,a_t)$，$y^\star=s_{t+1}$，再加一条「换 $a_t$ 必须换 $y$」。$S(g)$ 很高，只说明 $g$ 在这个 $(\mathcal{X},\mathcal{Y},d,\mathcal{D})$ 上高，不能迁移到另一套四元组。

验证很简单：拿一张官方表，遮住模型名，只看协议四格，说出它能量第 17 课的哪一根。说不出，就还没读懂这份榜。

### 5.3 WorldScore：下一场景，不是下一步动作

WorldScore（Duan、Yu、Chen、Fei-Fei、Wu，ICCV 2025，arXiv:2504.00983）把「世界生成」拆成一串下一场景任务。当前场景 $\mathcal{C}$ 给一张图加一句描述，下一场景 $\mathcal{N}$ 给文本，布局 $\mathcal{L}$ 给相机轨迹 $\mathcal{T}$ 和一句镜头说明。模型被指令去生成视频

$$
\mathbf{V}=g_{\mathrm{world}}\bigl(w_{\mathrm{proc}}(\mathcal{C},\mathcal{N},\mathcal{L})\bigr)
$$

$w_{\mathrm{proc}}$ 按模型类型把同一份世界说明翻译成 T2V、I2V、3D 或 4D 各自吃得下的条件。所有方法最后都交视频，评分器才能坐上同一张桌。数据集约 3000 条：2000 条静态世界（室内外各五类，含风格化对照），1000 条动态世界（五类运动）。静态任务考大范围相机运动和「走到新场景」；动态任务把相机钉死，只考场景内谁该动。

评分聚成三组，一共十个指标。可控性：相机误差

$$
e_{\mathrm{camera}}=\sqrt{e_{\theta}\cdot e_{t}}
$$

再加开放词汇检测的物体可控性、CLIPScore 的内容对齐。质量：DROID-SLAM 重投影误差量 3D 一致性，光流端点误差量光度一致性，Gram 矩阵量风格，CLIP-IQA+ 和美学头量主观质量。动态：运动是否落在指定区域、幅度、平滑。WorldScore-Static 平均可控性和质量；WorldScore-Dynamic 再并入动态三项。3D 模型不会做动态，动态三项记 0。

论文表 2 的关键观察可以当机制，不要当本课测到的排名。3D 路线在 Static 上压过视频路线，因为相机矩阵是它们的一等公民，几何一致性来自表示而不是记忆。视频路线的相机可控性普遍低。项目页后来补了 Voyager、Wan2.1 等行，最新数字以 [WorldScore 项目页](https://haoyi-duan.github.io/WorldScore/) 和 [Hugging Face 榜](https://huggingface.co/spaces/Howieeeee/WorldScore_Leaderboard) 为准。你没有跑 `worldscore/run_evaluate.py`，就不要改写这些格子。

类比：WorldScore 像在考摄影师听不听导播。导播说「向左推再拉出」，摄影师要走出新房间，墙不能闪，风格不能跳。类比失效处：导播给的是镜头，不是手。桌宠的 $a_t$ 是转头、伸手、出声。相机听话，只说明布局可控。[第 20 课](20_spatial_3d_state.md) 的 VGGT 已经说明，持久 3D 状态本身还不是世界模型，缺动作条件和时间。WorldScore 的 3D 高分经常来自同一笔：把世界存成可以渲染的几何，一致性不再经过预测器。Marble 那条「取消预测」的路，第 12 课写过。

三问速答。第一问：条件里通常没有逐步低层动作，只有相机和文本，按第 12 课口径第一问过不了。第二问：3D 路线可以靠表示拿满分，视频滑窗仍然会在长序列上漂，论文自己也写了视频模型怕长序列和室外。第三问：没有 agent，没有奖励，没有回真环境的策略分数。规划好整轴缺席。这正是第 17 课那一行：「把世界模型收窄成可控视频生成器」。

桌宠用途：以后有人把 WorldScore 第一名推荐给你当桌面模拟器，你问三句：有没有低层动作端口、动作对换分不分岔、有没有人用它在真桌子上选过动作。三句都空，它仍是画师。

### 5.4 WorldModelBench：视频生成器当世界模型用时，哪条先坏

WorldModelBench（Li、Fang、Chen 等，NeurIPS 2025，arXiv:2502.20694）专门审视频生成器自称世界模型这件事。作者的出发点很具体：已有视频榜（VBench 一类）主要量画质和文本一致性，机械臂悬在空中这种违反重力的片子仍然可以得高分。他们换了一套应用向题库：350 对「首帧 + 文本」，覆盖自动驾驶、机器人、人体活动、工业、自然、游戏、动画 7 个域、56 个子域，T2V 和 I2V 都能考。

每段生成视频按三维打分，总分 10。指令跟随 0 到 3：主体不在或不动是 0，动了但方向反了是 1，只做了一半是 2，做完是 3。物理遵守拆成五条 0/1：牛顿第一定律（无外力不该自己动）、质量守恒与固体本构（不该乱变形）、流体本构、不可穿透、重力。常识 0 到 2：单帧观感和时间一致性，这一维他们承认只是入场券。论文强调的卖点是前两维能抓住「物体自己变大变小」这种质量守恒违规。

样本怎么来的，决定这把尺子能量什么。参考视频来自 nuScenes、Open X-Embodiment、ActivityNet 等，首帧当 I2V 条件，GPT-4o 把首帧到后续的差别写成「动作」文本，再经人工核对。注意这里的「动作」是自然语言里的动词短语，例如「机械臂把胡萝卜放进金属碗」，不是关节增量。67K 条人类细标用来给 14 个前沿模型打分，再微调一个约 2B 的 VILA 评判器做自动打分。论文写：这个评判器预测世界建模违规的误差低于 GPT-4o。他们还用评判器当奖励去推 Open-Sora，展示指令和闪烁有改善。那是生成器对齐，不是 MPC。

和 VBench 的相关，论文自己算过：逐帧质量的模型对战胜率相关系数约 0.69，物理遵守掉到约 0.28。这句话是本课最该抄进笔记的结论之一：画质榜和物理违规榜可以给出两套排名。它仍然不是规划好。没有人把生成视频接进 CEM，也没有人在真臂上验收「碗还在桌上」。

三问速答。第一问：条件是文本和可选首帧，没有逐步动作端口，出局。第二问：单段短视频，根本不问转身回来。第三问：没有。E 档：默认 E0。指令跟随再高，也只是 E0 上的生成真加语义服从。

桌宠用途：WorldModelBench 的机器人子域看起来最像桌子，题面却是「看一段像真的操作视频」。[第 25 课](25_lerobot_imitation.md) 的 ACT 和 [第 30 课](30_desk_world_model.md) 的桌面世界模型都不该拿这 350 条当验收。你要的是同一历史换动作后未来分岔，以及伸手被安全层改写。那两件事这张榜没有出题。

### 5.5 Physics-IQ：真实实验的后 5 秒

Physics-IQ（Motamed、Culp、Swersky、Jaini、Geirhos，arXiv:2501.09038；项目页引用 WACV 2026）把问题收成一句可操作的话：生成模型懂不懂物理，就看它能不能续写一段真实实验录像。66 个场景，覆盖固体、流体、光学、热力学、磁学。每个场景三台固定机位、拍两遍，共 396 段 8 秒、3840×2160、30 fps 的真录像，不是仿真。前 3 秒当条件（I2V 只用切换帧，V2V 最多用满 3 秒），后 5 秒当答案。切换帧是人工选的：信息够起头，但后 5 秒仍要靠物理，例如多米诺已经推倒第一块、还没碰到第二块。

提示文本描述条件部分，不把答案说破。论文评的模型包括 Sora、Runway Gen 3、Pika 1.0、Lumiere、Stable Video Diffusion、VideoPoet，分 I2V 和多帧两种接口。Luma 的使用条款当时禁止刷榜，Veo 2 当时未普遍开放，论文写明没评。

四项指标对应四个问题。空间 IoU：运动发生在哪。时空 IoU：发生在哪以及何时。加权空间 IoU：发生在哪以及多少。MSE：看起来像不像那次实验。运动掩膜来自固定机位下的像素变化，所以相机一晃，三项 IoU 都会罚。综合分把四项加总（MSE 取负），再按「两遍真录像之间的物理方差」归一，满分 100。论文表 1：物理方差基线 100，当时最高的 VideoPoet 多帧是 29.5，Sora I2V 是 10.0。Sora 在另一项 2AFC 视觉真实感上最难被 Gemini 1.5 Pro 拆穿（55.6%，机会水平 50%），和 Physics-IQ 分没有显著相关。论文结论可以记成一句话：画面真和物理续写对不是同一根尺子。

这些数字是论文表，不是你测的。仓库 README 后来的 Physics-IQ 原榜、Verified 榜已经换了一批 2025 到 2026 年的模型，包括 Cosmos 3 和 Wan 2.2。抄的时候写「官方表，日期见 README」，不要把 29.5 和 39.5 混在同一列而不标版本。

三问速答。第一问：条件是帧或短视频加文本，没有逐步动作。你不能换「推左边那块多米诺」再看分岔。第二问：8 秒固定机位，不问离开视野。第三问：没有。它量的是预测准里最接近「和真世界后 5 秒比」的一种，仍然不是规划好。第 44 课会把「懂物理」继续操作化；本课只把尺子立住。

桌宠用途：杯子会不会下桌，不能用生成视频好不好看来代替。Physics-IQ 证明了这句话在受控实验上成立。它还不能替你做第 03 课的对换：对换要的是同一历史、两个动作、两条未来。

### 5.6 Physics-IQ Verified：尺子本身会漂

Physics-IQ Verified（Rädsch、Asano、Kuehne、Bauer、Jaini、Geirhos、Lüth，arXiv:2606.18943）把原榜当成一份会被审计的仪器。作者做了三件事。

第一，改提示。原提示有的事实错误、有的分不清切换帧前后已经发生了什么、有的漏掉关键约束、有的太含糊。他们按「好的考题」重写：人看到提示加切换帧应当能预见到实验结局，提示又不能把结局写破。再按模型厂商的习惯拆成场景、动作、相机、风格、范围等字段，用模板生成 bpp（best-practice prompt），和原提示 op 对照。

第二，改汇总。原 Physics-IQ 分在整个数据集上对 IoU 做「分子总和除以分母总和」，再减一套 MSE。样本对总分的权重不相等，低物理方差的实验永远到不了 1，失败也难追溯到某一条。Verified 在每条样本上对四项做裁剪后的比值再算术平均，最后对样本再平均。每条样本、每项指标权重相同。

第三，清伪激活。三项 IoU 都跑在相邻帧差分的激活图上。转台、支架、实验结束后的无关晃动，都会在真值掩膜里点亮不该属于物理现象的区域。他们标了现象结束帧和需要冻结的区域。论文写：57.6% 的样本被细化，34.8% 的提示被改过。六款 I2V 上原榜和 Verified 的排名 Kendall $\tau=0.46$。仓库 README 把 Verified 设成 `physiq/run_physics_iq.py` 的默认路径，原榜要显式加 `--original_physics_iq`。

这件事的课内含义比新数字更大。第 17 课写过：上下文几帧、取均值还是中位数、IRIS 的 token 用采样还是 argmax，每个选择都能挪动排名。Verified 是同一句话的公开规模版。所以本课禁止的不只是「编造分数」，还包括「把某一天的 README 第一名写成物理定律」。尺子一改，第一名可以换人。

对桌宠：你自己的第 30 课验收也是一把会漂的尺子。动作对换的阈值、多步漂移的步数、安全层的桌沿厘米数，写进报告时必须和模型版本绑在一起。换了提示或换了汇总，就要当新协议，不能和旧数字横着比。

### 5.7 三份榜都可以很高，动作对换仍然可以失败

把三份榜和课内协议并排。WorldScore 的 $x$ 是场景加相机。WorldModelBench 的 $x$ 是文本加首帧。Physics-IQ 的 $x$ 是切换帧或前 3 秒。第 03 课和第 30 课的 $x$ 是 $(s_t,a_t)$。只有最后一种协议里，「换动作」是合法操作。

因此下面四件事可以同时成立，并不矛盾：

1. 某模型在 WorldScore-Static 上超过一众视频模型，因为它把世界存成 3D，相机误差接近零。
2. 同一模型在 WorldModelBench 的指令维很低，因为它根本不吃那类文本，或会把胡萝卜放进碗写成飘进碗。
3. 另一模型在 Physics-IQ 上续写流体还可以，固体碰撞穿模，视觉真实感仍然很高。
4. 以上任一模型拿去第 30 课的桌子，左转头和伸手给出同一条未来。

第 4 条失败时，它不是世界模型，是画师。第 12 课已经用这句话筛过 Sora。本课只是把筛子印到三份榜的封面背后。

E 档怎么接。三份榜默认服务 E0：无逐步动作，或对换未做。把 Cosmos 3 的官方 Physics-IQ 表抄进笔记，档仍是「未知」或 E0，除非你自己做了对换并且把查询接进动作出口。V-JEPA 2 的公开主线是视频表征，规划接口在 AC 里，[第 15 课](15_vjepa2_in_practice.md) 标过精读不复现；它没上这三份榜，格子写未测，不要用 Diving48 探针冒充 Physics-IQ。DINO-WM 在 PushT 上能对换、能缩小规划，那是 [第 22 课](22_foundation_video_wm.md) 的实战，仍然不是这三份榜的分。Genie 3、GAIA-2 无权重，只讲，不能练。

验证：第 7 节的系统卡片会逼你先猜「它会在哪份榜好看」，再揭晓该榜实际在测什么。猜错是合格数据。猜对却说不出评分器，不算过。

## 6. 源码导读

三份仓都不是训练代码。要看的是：入口命令落在哪个文件、评分器读什么、样本清单在哪。路径以 2026-08 核对的当前仓库为准。

### 6.1 WorldScore：先登记模型，再生成，再评

仓库根目录的 `download.py` 从 Hugging Face 的 `Howieeeee/WorldScore` 拉 `WorldScore-Dataset.zip`，解压目标是环境变量 `DATA_PATH`。没写 `DATA_PATH`，下载脚本会把路径打出来然后在解压时报错。README 要求先在仓库根建 `.env`，再在每个新终端导出：

仓库把评测拆成两条线。生成线是 `world_generators/generate_videos.py`。它用 `fire.Fire`，参数名是 `model_name`。README 写的入口是 `--model-name`。`main` 先 `check_model`，再按 `worldscore/benchmark/utils/modeltype.py` 的 `type2model` 判断类型：`threedgen` 只跑静态，`videogen` 和 `fourdgen` 跑静态加动态。`wan2.1_i2v` 已经登记在 `videogen` 里。Cosmos 3、Genie 3、GAIA-2、DINO-WM、open-oasis、DIAMOND、V-JEPA 2 都不在这张表里。没登记就断言失败，这是「未测」的代码含义。

评测线是 `worldscore/run_evaluate.py`。`main` 同样走 `fire.Fire`，对每个 `visual_movement` 调 `run_evaluation`，最后 `calculate_worldscore`。Static 是可控性加质量的算术平均，Dynamic 再并入动态三项，结果写到该模型 `runs_root` 下的 `worldscore.json`。真正打分在 `worldscore/benchmark/helpers/evaluator.py` 的 `Evaluator`：按场景读已生成视频，用 `get_aspect_evaluator` 挂上具体指标。指标类从 `worldscore/benchmark/metrics/__init__.py` 进来，名字和论文一一对应：`CameraErrorMetric`、`ObjectDetectionMetric`、`CLIPScoreMetric`、`ReprojectionErrorMetric`、`OpticalFlowAverageEndPointErrorMetric`、`GramMatrixMetric`、`MotionAccuracyMetric`、`MotionSmoothnessMetric`。主观质量走 CLIP-IQA+ 和美学头。

`setup.py` 还注册了 `worldscore`、`worldscore-eval`、`worldscore-analysis`、`worldscore-gen`。README 的完整性检查是 `worldscore-analysis -cd --model_name <model_name>`，交榜前检查是 `-cs`。完整评测还要编 DROID-SLAM、Grounding-SAM、SAM2、VFIMamba，官方按 CUDA 12.1 写。本课不装这条链。你要核实的是：入口文件在、模型登记表在、十个指标类名对得上论文。

### 6.2 WorldModelBench：样本清单和评判器是两样东西

仓库现在把数据放在 GitHub 里。README 写过数据曾在 Hugging Face 的 `Efficient-Large-Model/worldmodelbench`，2025-07-28 挪进本仓方便文档。当前清单是根目录 `worldmodelbench.json`，350 条，字段固定：`domain`、`subdomain`、`text_first_frame`、`text_instruction`、`first_frame`。`first_frame` 指向 `images/` 下的 jpg。T2V 的提示是首帧描述和指令用空格拼起来；I2V 用首帧图加 `text_instruction`。生成视频要和首帧同名，只把后缀换成 `.mp4`。

评分脚本的文件名是 `evaluation.py`。README 的命令写成了 `python evaluate.py`，仓库里没有这个文件。以文件为准：

`evaluation.py` 的 `argparse` 要 `--judge`、`--video_dir`、`--model_name`，可选 `--save_name`、`--cot`、`--no-save`。`EvaluationType` 三个枚举：`instruction`、`physical_laws`、`common_sense`。物理五问和常识两问写在 `get_default_question_pool` 里，和论文 3.1 节一致。评判器按提示生成「Score: n」或 Yes/No，`process_results` 再汇总。评判器权重在 Hugging Face 的 `Efficient-Large-Model/vila-ewm-qwen2-1.5b`，还要按 README 去装 VILA。24GB 上装评判器本身可能能转，但你没有 350 段待评视频，本课只读脚本和 json，不跑自动打分。

读 json 时注意域的英文名：自动驾驶写成 `autonomous vehicle`，机器人是 `robotics`，游戏是 `video games`。抽样本用这些键，不要凭印象编题面。

### 6.3 Physics-IQ：默认已经是 Verified

仓库 `google-deepmind/physics-iq-benchmark` 同时放原榜和 Verified。`pyproject.toml` 的包名是 `physics-iq-benchmark`，代码在 `physiq/`。入口 `physiq/run_physics_iq.py` 的必选参数是 `--input_folders`、`--output_folder`、`--descriptions_file`；`--benchmark_base_folder` 默认当前目录；`--original_physics_iq` 是开关，加上才走原真值和原汇总。默认路径会去找 `physics-IQ-benchmark-verified/`。

评分流水线从文件名就能读出来。`physiq/binary_mask_generator.py` 从视频做运动掩膜。`physiq/calculate_and_write_metrics_to_csv.py` 按场景写四项指标。`physiq/calculate_iq_score.py` 是原汇总，`physiq/calculate_iq_score_stable.py` 的 `IQTable` 是 Verified 的逐条分。README 写明：一次运行会同时报原分和 Verified 分，交 Verified 榜用后者。

样本清单在 `descriptions/descriptions_original.csv`。列是 `scenario`、`description`、`category`、`generated_video_name`。第一条中心机位是 `0002_perspective-center_take-1_trimmed-ball-and-block-fall.mp4`，类别 `Solid Mechanics`，提示写两个抓手松开网球和方块、落到枕头上，相机静止。流体第一条中心机位是 `0038_..._trimmed-blow-balloon.mp4`。磁学有 `0095_..._trimmed-magnet-domino.mp4`。Verified 数据在 Hugging Face `Anates-Labs-Research/Physics-IQ-Verified`，目录里能看到 `switch-frames/`、`split-videos/`、`full-videos/`、`video-masks/`。文件名带 `0001_` 这种四位前缀，生成视频必须保留这个前缀。评测前要切到正好 5 秒，V2V 还不能把 3 秒条件段算进这 5 秒。

原数据下载脚本是 `physiq/download_physics_iq_data.py`，要 `--fps 30 --original_physics_iq`。完整 4K 集很大，本课不下载。你要核实的是：csv 里的场景名、入口参数、默认走 Verified 这三件事和 README 一致。

## 7. 实验

实验是读榜，不是刷榜。顺序：先 clone，再按 README 把入口命令抄进笔记，然后每份榜抽公开样本用纸走评分维度，最后填十行地图、做系统卡片。任何一步若开始下载完整 4K 集或调用闭源视频 API，就停，那已经超出本课。

建议在 `~/learn-wm/lesson34/` 下建三个目录。下面的命令都在各自仓库根目录理解。

### Step 1: clone 三个评测仓

```bash
git clone https://github.com/haoyi-duan/WorldScore.git
```

```bash
git clone https://github.com/WorldModelBench-Team/WorldModelBench.git
```

```bash
git clone https://github.com/google-deepmind/physics-iq-benchmark.git
```

三条都成功即可。预期：各有 `README.md`；WorldScore 能看到 `download.py` 和 `worldscore/run_evaluate.py`；WorldModelBench 能看到 `evaluation.py` 和 `worldmodelbench.json`；Physics-IQ 能看到 `physiq/run_physics_iq.py` 和 `descriptions/descriptions_original.csv`。失败：公司代理拦住 github.com，换镜像或同学拷一份代码树，不要假装 clone 过。

### Step 2: 按 README 抄 WorldScore 入口

进入 WorldScore 根目录。README 的 Setup 要求先写 `.env`（`WORLDSCORE_PATH`、`MODEL_PATH`、`DATA_PATH`），每个新终端导出一次。本课如果只读代码，可以不写。若你打算下数据集，入口是：

```bash
python download.py
```

生成线 README 原文：

```bash
python world_generators/generate_videos.py --model-name wan2.1_i2v
```

评测线 README 原文：

```bash
python worldscore/run_evaluate.py --model_name wan2.1_i2v
```

这里用 `wan2.1_i2v`，是因为它已经写在 `modeltype.py` 的 `videogen` 里，不是因为本课要你真的生成。24GB 不够按 WorldScore 协议跑完 Wan2.1 的三千条。完整评测环境还要 CUDA 12.1 和一串第三方库，主线不装。你的交付是：把上面三条命令、`modeltype.py` 里有谁、没有谁，抄进 `leaderboard_map.md`。

### Step 3: 走 3 条 WorldScore 公开样本的评分维度

打开 [WorldScore 项目页](https://haoyi-duan.github.io/WorldScore/) 的 Controllability / Quality / Dynamics 展示，不要下完整 zip。用纸建三列：样本、你认为该看哪几项、看对了和看错了会怎样。

样本 A，静态、卧室。项目页的例子：初始卧室，镜头指令依次是向左摇、向左移、拉出。可控性看相机是否真的走出新空间；质量看墙角会不会闪、床单纹理会不会在帧间爬。看对了：后段应该是另一面墙或门外，几何连续。看错了：相机几乎不动，或房间被重新发明。这一条专门打 VBench 一类单场景画质榜打不到的分。

样本 B，物体可控。项目页用开放词汇检测：下一场景文本点名的物体有没有出现。看对了：检测器能在后段帧上框到文本里的物体。看错了：镜头走了，物体没来，或来了但不是那个。注意它测的是「文本里的名词」，不是「你伸手之后杯子的落点」。

样本 C，动态、运动落点。项目页的章鱼与水母：文本说章鱼在沙上爬，水母不该跟着重排。运动精度比较指定区域内和外的光流。看对了：动的是章鱼。看错了：整段海洋一起漂，或水母在动、章鱼在装死。相机若偷偷跟着动，三项动态都会脏。

写完三行，加一句：这三条里没有任何一条要求你输入关节动作，也没有任何一条拿去搜策略。WorldScore-Static 第一名可以是不会动的 3D 场景生成器。

### Step 4: 按 README 抄 WorldModelBench 入口，并走 3 条 json 样本

进入 WorldModelBench 根目录。先确认文件名：

```bash
ls evaluation.py worldmodelbench.json images
```

README 把脚本写成了 `evaluate.py`。以仓库里的 `evaluation.py` 为准。评测入口（先不要跑，评判器和 350 段视频都没有）：

```bash
python evaluation.py --model_name TESTED_MODEL --video_dir GENERATED_VIDEOS --judge PATH_TO_JUDGE
```

打开 `worldmodelbench.json`，抄下面三条。字段以文件为准。

样本 1，自动驾驶，`subdomain` 为 Stopping。首帧：车在桥上接近红绿灯，两侧有施工围挡，灯是黄的。指令：`The autonomous vehicle stops at the traffic light on the bridge.` 指令维：车继续冲过桥是 0 到 1，明显刹住是 3。物理维：车自己飘起来或穿进围挡，对应重力和不可穿透。常识维：灯闪成别的物体，时间一致性扣分。看对了：车减速停在灯前。看错了：画面很电影，车从桥上飞走。

样本 2，机器人，`subdomain` 为 place。首帧：臂悬在木桌上，桌上有胡萝卜和金属碗。指令：`The robotic arm places the carrot into the metal bowl.` 这是最像桌子的一条。指令维：臂动了但没碰胡萝卜是 1，夹起来没放进碗是 2，放进碗是 3。物理维：胡萝卜穿过碗壁、碗自己放大、臂在空中漂，各扣一条。看对了：接触发生在夹爪和胡萝卜、胡萝卜和碗之间。看错了：胡萝卜溶解进碗，或臂穿过桌面。即便看对了，也只说明这段视频服从文本和五条外观物理，不说明换一个关节增量会得到另一条未来。

样本 3，自然，`subdomain` 为 Landscape and Scenery。指令：浪轻拍卵石岸。多数模型在这一域分会更高，论文热图也写自然域容易、驾驶和机器人难。用它当对照：常识维很容易满分，指令维几乎是「继续像浪」，物理维很难被激发。一条容易题拉高总分，不等于机械臂那条过了。

三行打完，标明：评分器是人或 2B 评判器，不是真实后续帧，更不是桌面回报。

### Step 5: 抄 Physics-IQ 入口，并走 3 条 csv 样本

进入 `physics-iq-benchmark`。Verified 是默认。README 的数据入口：

```bash
hf download Anates-Labs-Research/Physics-IQ-Verified --repo-type dataset --local-dir physics-IQ-benchmark-verified
```

本课可以不下。环境入口若要用官方依赖：

```bash
uv sync
```

评测入口（把占位符留在笔记里，不要伪造生成目录）：

```bash
uv run physiq/run_physics_iq.py --input_folders generated_videos_5s/MODEL-bpp-run_01 --output_folder OUT --descriptions_file descriptions/best_practice/descriptions_base.csv --benchmark_base_folder PARENT
```

原榜要另加开关，并换描述文件：

```bash
uv run physiq/download_physics_iq_data.py --fps 30 --original_physics_iq --benchmark_base_folder PARENT
```

```bash
uv run physiq/run_physics_iq.py --input_folders generated_videos_5s/MODEL --output_folder OUT --descriptions_file descriptions/descriptions_original.csv --benchmark_base_folder PARENT --original_physics_iq
```

打开 `descriptions/descriptions_original.csv`，走三条。提示文本是原榜用的，Verified 改过其中一部分；走维度时用原描述就够理解「考什么」。

样本 1，固体，`trimmed-ball-and-block-fall`。两个抓手松开网球和橙色方块，落到两个枕头上。看对了：两个物体下落，枕头局部凹陷，桌面其余不动。空间 IoU 只问「动的地方对不对」；时空 IoU 还问「是先落下再凹，而不是枕头先跳」；加权空间 IoU 问枕头被压的那一块是不是反复动；MSE 问颜色和形状还是不是那个球。看错了：球漂走、枕头变成床、相机开始推近。

样本 2，流体，`trimmed-blow-balloon`。固定气泵往黑气球里打气。看对了：气球体积变大，形状大致保持，桌面其余静止。看错了：气球瞬移、自己打结、变成另一个物体。流体本构在 WorldModelBench 里是一条 0/1，这里变成和真录像比掩膜。

样本 3，磁学，`trimmed-magnet-domino`。强磁对着旋转台上的多米诺。看对了：多米诺在特定角度被吸倒或吸走，转台按提示转动。看错了：多米诺自己跳，或磁铁飞出画面。这类题最能说明「看过很多多米诺视频」不够：中间有磁，捷径会失效。

写三行时强制加一列「我有没有换动作」。答案都是没有。所以 Physics-IQ 再高，第 03 课的对换仍未开考。仓库 README 的 Verified 榜上有 Cosmos3-Super I2V、Cosmos3-Nano、Wan 2.2 等行，那是投稿方或论文报告的官方表。抄的时候写日期和「官方表」，不要写进「我的实测」。

### Step 6: 填十行地图

把下面这张表抄进 `leaderboard_map.md`，空格按本课口径填。WorldScore / WorldModelBench / Physics-IQ 三列只允许：未测、宣称、官方表（并注明出处）。预测准 / 生成真 / 规划好 引用第 12、17 课的证据，没有证据就写未测。24GB 列写本课能不能在主线卡上做该系统的完整榜协议。

| 系统 | 开源三档 | 本课档位 | WorldScore | WorldModelBench | Physics-IQ | 预测准 | 生成真 | 规划好 | 24GB 完整榜 |
|---|---|---|---|---|---|---|---|---|---|
| open-oasis | 仅权重 | 第 12 课体验，本课不重测 | 未测 | 未测 | 未测 | 第 12 课崩坏记录 | 中（Minecraft 风格） | 无公开决策验收 | 不跑完整榜；500M 推理第 12 课已做过 |
| DIAMOND | 可跑 | 第 10 课复现/体验 | 未测 | 未测 | 未测 | 课内 Atari 漂移 | 课内画质对照 | 课内 agent 分 | 不在这三份榜的协议里 |
| DINO-WM | 可跑 | 第 22 课实战 | 未测 | 未测 | 未测 | 课内对换 | 解码器不是卖点 | 论文表有规划；你的 5 局只看方向 | 不在这三份榜的协议里 |
| Cosmos-Predict2.5 | 可跑但 2B Video2World 官方约 32.54GB | 第 22 课只讲 | 未测 | 未测 | 未测 | 未在 24GB 上测 | 官方 demo / 论文 | 基础模型不直接规划 | 不够 |
| Cosmos 3 | 文档公开；大权重只讲 | 只讲（下一课展开） | 未测 | 未测 | 官方表（Verified README：Super I2V、Nano I2V） | 未做对换 | 宣称全模态生成 | 宣称可接策略头，本课未验收 | 16B/64B 不够；完整榜不够 |
| Genie 3 | 纯 demo，无权重 | 只讲，不能练 | 未测 | 未测 | 未测 | 宣称数分钟一致 | 官方博客 720p 24fps | 定位训练场，无公开验收 | 不能练 |
| Matrix-Game | 可跑（官方显存要求以当天 README 为准） | 本课只填表，第 43 课再体验 | 未测 | 未测 | 未测 | 未测 | 宣称实时交互 | 未展示回真环境规划 | 完整榜不适用；交互 demo 可能卡在 24GB 边界 |
| GAIA-2 | 纯技术报告 | 只讲，不能练 | 未测 | 未测 | 未测 | 多视角短时程（宣称） | 服务驾驶数据合成 | 报告未把它当在线规划器 | 不能练 |
| V-JEPA 2 | 可跑权重 | 第 15 课实战/精读 | 未测 | 未测 | 未测 | 表征探针，不是这三榜 | 不生成像素 | AC 规划只读接口 | 不在这三份榜的协议里 |
| Wan2.1 | 可跑（大视频模型） | 体验失败/只讲：24GB 不够按协议评完 | 官方表（项目页有 Static / Dynamic 两列） | 未测 | 原榜 README 有 Wan2.1 14B 加后处理的行，标官方表 | 未测 | 开源视频生成器 | 无 | 不够按 3000 条或 198 条协议评 |

WorldScore 项目页给 Wan2.1 的 Static 57.56、Dynamic 52.85，是官方表，写进笔记时必须带「项目页，非本课复现」。Cosmos 3 的 Verified 数字同样只抄 README，并写「下一课再拆 MoT，本课不验收」。

### Step 7: 系统卡片，先猜再揭晓

本课互动是六张卡片。每张只给四格：输入、输出、有没有逐步动作端口、有没有可下载权重。先猜它会在哪份榜好看，再看揭晓。网页实验若已挂上就点；没有就用纸。不要先翻第 5 节。

卡片 1。输入：文本或首帧。输出：数秒视频。动作端口：无。权重：有（开源视频模型）。多数人会猜 WorldScore 或 WorldModelBench。揭晓：接口对上了这两份榜。WorldScore 实际测下一场景加相机，不是桌宠动作；WorldModelBench 测指令、常识、五条物理违规。两份都可以高，第 03 课对换仍未开考。

卡片 2。输入：观察加低层动作。输出：下一时刻的 patch 特征。动作端口：有。权重：有。这张很像 DINO-WM。多数人会猜 Physics-IQ，因为「更像世界模型」。揭晓：三份榜都不收特征空间输出，也不吃 PushT 那种动作。它该好看的地方是课内对换和缩小规划，格子在三榜上是未测。

卡片 3。输入：实时键鼠。输出：可玩视频流。动作端口：有（交互）。权重：无。这张像 Genie 3。多数人会猜 WorldScore 的长序列一致性。揭晓：Genie 3 不在 `modeltype.py` 里，三份榜都是未测。官方博客的数分钟一致是宣称。WorldScore 的长序列是「下一场景接下一场景」，不是数分钟可玩。

卡片 4。输入：语言、图像、视频，宣传里还有声音和动作。输出：多种，取决于配置。动作端口：宣称有。权重：有，但 24GB 不够跑大卡。这张像 Cosmos 3。多数人会猜三份榜通吃。揭晓：Physics-IQ Verified 的 README 有官方表，WorldScore / WorldModelBench 本课记未测。官方表不等于你做了对换，也不等于 E4。

卡片 5。输入：自车动力学、他车、天气、道路语义、多相机。输出：多相机驾驶视频。动作端口：自车动态可视为一种条件。权重：无。这张是 GAIA-2。多数人会猜 WorldModelBench 的驾驶域。揭晓：GAIA-2 不能练，没上这三份公开协议。它的验证者是路和数据合成，第 41 课再拆。规划好若存在，也不在这三份榜上。

卡片 6。输入：Minecraft 观察加动作。输出：下一帧。动作端口：有。权重：500M 开源缩小版有。这张是 open-oasis。多数人会猜 WorldScore 动态项。揭晓：未测。你有的预测准证据是第 12 课崩坏帧数。WorldScore 不吃那套动作张量。

六张都猜完，写两行收尾：好看的榜测的是哪一格协议；桌宠要补的是哪一格协议。两行对不上，就不要把第一名写进第 32 课的零件表。

## 8. 配置与预算

无训练。时间：读三篇主论文加 Verified 一天；clone、抄命令、走样本、填表、做卡片半天到一天。GPU：0。完整榜协议不在预算里。

数据。默认不下 WorldScore 的 zip、不下 Physics-IQ 的 4K 全集、不调闭源视频 API。样本来自项目页、`worldmodelbench.json`、`descriptions/descriptions_original.csv`。若你执意下载 Verified 数据，按 Hugging Face 卡片看体积再决定，失败就停，写「磁盘或审核拦住了」。

命令预算。WorldScore 的 `python download.py` 依赖 `.env` 里的 `DATA_PATH`。WorldModelBench 的自动打分依赖 VILA 和 `vila-ewm-qwen2-1.5b`，本课不装。Physics-IQ 评测还要 `ffprobe`（Linux 上通常随 ffmpeg）。Mac 用户把「命令能在 README 里对上、文件在仓库里」当成验收，不把 `uv run physiq/run_physics_iq.py` 跑通当成及格线。

超参。这三份榜没有你要调的训练超参。唯一要钉死的是协议版本：WorldScore 以 ICCV 2025 论文加项目页为准；WorldModelBench 以仓库里的 350 条 json 为准；Physics-IQ 默认 Verified，原榜必须写 `--original_physics_iq`。把不同日期的 README 第一名写在同一格，算协议错误。

检查点。本课没有模型检查点。你的检查点是那份 `leaderboard_map.md`：十行表、三份入口命令、九条人工样本、六张卡片的猜测和揭晓。

## 9. 验收

- [ ] 能不看笔记说出三份榜各自的输入、输出、评分器、样本来源。
- [ ] 能用第 12 课三问给三份榜各答一遍，并指出规划好在哪一份上整轴缺席。
- [ ] 能说明 WorldScore 的相机轨迹为什么不是第 03 课的 $a_t$。
- [ ] 能说明 WorldModelBench 的「动作」是文本动词，指令分高不等于对换分岔。
- [ ] 能说明 Physics-IQ 的 100 分是物理方差，不是规划成功率；Verified 改了什么、为什么排名会动。
- [ ] 三个仓库已 clone；WorldScore、WorldModelBench、Physics-IQ 的入口命令已按当前 README（及第 6 节指出的文件名差异）抄进笔记。
- [ ] 每份榜至少 2 条公开样本有「看对了 / 看错了」记录。
- [ ] 十行地图里，未测的格子没有被填成数字；官方表带了出处；Genie 3 和 GAIA-2 标了只讲、不能练。
- [ ] 能口头回答：WorldScore 第一名不必能给桌宠做 MPC。
- [ ] 没有把官方宣传的数分钟一致性写成自己测过。

## 10. 排错

| 症状 | 可能原因 | 怎么验证 | 怎么修 |
|---|---|---|---|
| 把 WorldScore 第一名写成桌宠模拟器 | 把世界生成和 $P(s_{t+1}\mid s_t,a_t)$ 当成一件事 | 看输入列有没有低层动作 | 改写成生成真加相机可控，规划好写未测 |
| 把 Physics-IQ 高分写成 E4 | 把续写对当成模型在回路 | 查是否题 2：有没有查询后改动作 | 停在 E0 或未知 |
| 抄了 29.5 和 39.5 却不标版本 | 把原榜和 Verified、论文表和 README 表混了 | 对照 arXiv:2501.09038 表 1 和仓库 README 日期 | 拆成两行，各写出处 |
| WorldModelBench 命令 FileNotFound | README 写 `evaluate.py` | `ls` 根目录 | 改用 `evaluation.py` |
| `python download.py` 解压失败 | 没设 `DATA_PATH` | 看脚本打印的 Dataset Path | 按 README 写 `.env` 并导出；或不下数据 |
| 想在 24GB 上评 Wan2.1 全集 | 把体验和完整协议混了 | 数测试条数：3000 或 198 | 停。本课不跑完整榜 |
| 把 V-JEPA 2 探针写成 Physics-IQ | 两把尺子都带「物理」二字 | 看输出是类别还是 5 秒视频 | 探针列保持第 15 课，三榜写未测 |
| 六张卡片先看了第 5 节再猜 | 把揭晓当阅读理解 | 猜测列和时间戳 | 重猜一版，或标明「已读原理后的第二次」 |
| clone 被拒 | 网络或仓库更名 | 用浏览器打开三个 GitHub 地址 | 换网络或记录「以当天地址为准」 |
| 把项目页 Voyager 的 Static 第一写成论文表 2 的结论 | 项目页在论文之后补行 | 对照 arXiv:2504.00983 表 2 和项目页表格 | 论文结论写 3D 路线当时领先，当前第一名指向项目页并标日期 |

## 11. 前沿与改造

2025 到 2026 年，视频生成器、3D 场景生成器和交互游戏引擎抢同一句广告词。公开榜给出的是局部协议。WorldScore 证明统一协议可行，代价是动作退化成相机。WorldModelBench 证明画质榜看不出物理违规，代价是裁判变成人和小 VLM。Physics-IQ 证明画面真推不出续写对，Verified 又证明续写这把尺子自己会漂。三件事都没有把规划好补上。第 44 课会继续操作化「懂物理」；第 37 课会问同一副骨架换训练协议为什么能从整段往后编变成边看边播。本课只要求你读榜时先问协议四格。

缩小版和前沿的差距，钱能解决的是：3000 条生成、198 段 4K 续写、闭源 API、多卡评判器。钱解决不了的是机制：输入里没有 $a_t$，评分器就看不见对换；输出里没有「查询后改写的动作」，E4 就没有证据。桌宠主线仍然用第 30 课的小模型和第 03 课的对换，不把完整榜搬回家。

动手改造清单（都不训大模型，也不跑完整榜）：

1. 给第 30 课的桌面模型做一张「伪 WorldScore」对照页。选你桌子上的一个静态场景，写三句镜头说明（向左转、拉近杯子、拉出看到桌沿），用第 20 或 [第 21 课](21_persistent_4d.md) 已有的点图或自己拍的三张图当「下一场景」，不要调用大视频模型。预算：2 小时纸面加拍照。预期：你能指出相机布局可控和「伸手」可控不是同一输入。失败：写着写着又把镜头说明当成了 $a_t$，重写输入列。
2. 把 WorldModelBench 机器人那条胡萝卜题，改写成第 03 课对换题。固定首帧，写两个低层动作（夹爪左移 3 cm、右移 3 cm），画出你期望的两条未来，再对照 json 里的那句文本指令。预算：1 小时。预期：文本指令只有一条未来，对换题有两条。失败：两份题被你合成一句「让臂听话」，说明还没分开协议。
3. 用 Physics-IQ 的四项名字，给第 32 课「克制」打一份假想分。伸手扫杯的想象视频如果画得很真、落点却在桌沿外，空间 IoU 会怎样、规划好会怎样、E4 是否题 2 会怎样。预算：1 小时。预期：三列可以互相矛盾。失败：三列被你写成同一个数。
4. 审计自己第 17 课的协议，对照 Verified 的三刀：提示清不清、样本权重均不均、有没有伪激活（例如你把暂停菜单算进了运动掩膜）。预算：2 小时纸面。预期：至少改一处汇总或排除规则。失败：审计完一条都不改，又给不出「已经足够干净」的理由。

顺手复现方向。Physics-IQ 论文的核心不是 29.5 这个数，是「视觉真实感和物理续写可以脱钩」。你没有闭源模型，不能复现表 1。能复现的方向：第 10 课 DIAMOND 对 IRIS 的画质对照，若和 Breakout 分数排名不一致，就是同一句话的课内版。WorldScore 论文的核心是「3D 路线在静态可控性上压过视频路线」。你不能复现表 2，能复现的方向：第 20 课点图换视角不漂、第 12 课滑窗会漂，表示一致性对预测一致性。Verified 的核心是「改协议会改排名」。课内对应是第 17 课 IRIS 采样改 argmax 会同时挪动两个指标。数字对不上论文，趋势对上就算这个改造合格。

## 12. 论文与延伸

1. Duan, H., Yu, H.-X., Chen, S., Fei-Fei, L., & Wu, J. (2025). WorldScore: A Unified Evaluation Benchmark for World Generation. ICCV 2025. [arXiv:2504.00983](https://arxiv.org/abs/2504.00983)。项目页 [haoyi-duan.github.io/WorldScore](https://haoyi-duan.github.io/WorldScore/)，代码 [haoyi-duan/WorldScore](https://github.com/haoyi-duan/WorldScore)。带着问题读：下一场景任务怎样把 3D / 4D / T2V / I2V 收成同一输出？十个指标里哪几个在量生成真、哪几个在量听指令、哪一个都不是规划好？论文表 2 里 3D 路线领先 Static，这个结论在项目页补了 Voyager、Wan2.1 之后还该怎么引用？
2. Li, D., Fang, Y., Chen, Y., et al. (2025). WorldModelBench: Judging Video Generation Models As World Models. NeurIPS 2025. [arXiv:2502.20694](https://arxiv.org/abs/2502.20694)。数据与代码 [WorldModelBench-Team/WorldModelBench](https://github.com/WorldModelBench-Team/WorldModelBench)，评判器 `Efficient-Large-Model/vila-ewm-qwen2-1.5b`。带着问题读：指令 0 到 3 和物理五条 0/1 各自抓住什么违规？为什么和 VBench 的逐帧质量相关高、和物理遵守相关低？用评判器推 Open-Sora，改善的是生成真还是规划好？
3. Motamed, S., Culp, L., Swersky, K., Jaini, P., & Geirhos, R. Do generative video models understand physical principles? [arXiv:2501.09038](https://arxiv.org/abs/2501.09038)。代码 [google-deepmind/physics-iq-benchmark](https://github.com/google-deepmind/physics-iq-benchmark)，项目页 [physics-iq.github.io](https://physics-iq.github.io/)。带着问题读：切换帧为什么必须「信息够、答案还没发生」？空间 IoU 和时空 IoU 拆开是为了抓住哪类错？Sora 在 2AFC 真实感上最好、在 Physics-IQ 上当时很低，作者怎样避免把这读成「视频模型毫无价值」？
4. Rädsch, T., Asano, Y. M., Kuehne, H., Bauer, S., Jaini, P., Geirhos, R., & Lüth, C. T. (2026). Physics-IQ Verified. [arXiv:2606.18943](https://arxiv.org/abs/2606.18943)。同一仓库，默认评测路径。带着问题读：四种不清提示分别让模型或人无法作答的方式是什么？逐条平均和全数据集比值，谁在悄悄给高方差样本加权？$\tau=0.46$ 对「引用上周的第一名」意味着什么？
5. 回访 [第 12 课](12_frontier_landscape.md) 的三问和 [第 17 课](17_evaluating_world_models.md) 的四分指标。带着问题读：本课三份榜各自能填进哪一根尺子？第 17 课已经把 WorldScore 写成没有 agent 的那一行，本课补上的两份榜有没有把那一行推翻？
6. 回访 [第 22 课](22_foundation_video_wm.md) 的资格审查和 [第 33 课](33_embodiment_degrees.md) 的 E0 到 E5。带着问题读：一份 Physics-IQ 官方表最多能支持哪一档？要把 Cosmos 3 从未知升到 E1 或 E4，还缺哪条是否题？
7. 选读：Wan 等，Wan: Open and Advanced Large-Scale Video Generative Models，[arXiv:2503.20314](https://arxiv.org/abs/2503.20314)。带着问题读：它在 WorldScore 项目页上的两列分测的是哪一种好？14B 级权重和本课 24GB 主线的关系是什么？不要把它的生成分抄进第 30 课的对换报告。

下一课是 Cosmos 3。同一套 Mixture-of-Transformers 会同时吃语言、图像、视频、声音和动作。本课先把刷榜分数按协议拆开：生成真、续写对、指令跟随，默认不是桌宠要的规划好。Cosmos 3 的配置表只有在这个拆法下才读得动。毕业标准仍在第 32 课和第 33 课。
