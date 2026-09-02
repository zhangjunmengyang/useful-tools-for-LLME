---
id: 44_physics_understanding
title: "懂物理怎么操作化"
summary: "画面真和物理对是同一根尺子吗？Physics-IQ 测的是续写还是规划？"
unit: frontier
play_tools: []
checkpoints:
  - "一份画面真、物理对、规划好的三列笔记。"
  - "两条公开样本的对错预期。"
---

# 第 44 课：把「懂物理」拆成三根能分开打分的尺子

> 类型：研究 + 评测代码体验（不从头训练；完整跑闭源视频模型刷榜做不到）<br>
> 建议周期：1-2 天<br>
> 硬件：Mac / 纯 CPU 可完成克隆、装环境、读公开描述、跑 `--help` 和评分单元测试；完整 198 条 5 秒生成需要闭源 API 或大显存视频模型，本课不要求<br>
> 锚定仓库：[google-deepmind/physics-iq-benchmark](https://github.com/google-deepmind/physics-iq-benchmark)（软件 Apache 2.0，数据 CC-BY；当前默认走 Physics-IQ Verified）<br>
> 产物：一份「画面真 ≠ 物理对 ≠ 规划好」三列笔记、两个公开实验的看对/看错清单、官方分数三列表、评测入口与评分导入记录

## 1. 这一课做什么

整门课仍是同一条循环：观察先压成状态，再按动作预测下一状态，然后展开多条未来，给未来打分，最后选动作。这条线从 [第 01 课](01_world_models_hands_on.md) 接上来。第九幕不改毕业标准。第 32 课仍是桌宠总装，第 33 课仍是 E0 到 E5。本幕只读 2025-2026 的榜、声音、配方和架构，把「SOTA 世界模型」四个字拆回尺子。

第 43 课把可玩性拆开：帧率、动作延迟、回头一致性、动作对换。缺任何一项，网页演示再流畅也不能叫实时世界模型。那一课量的是交互接口。这一课换零件：给「懂物理」做操作化。宣传稿里，一段 4K 水流、一次漂亮的碰撞，都会被写成物理理解。桌宠要的不是海报，是杯子会不会下桌。两件事可以同向，也可以完全反着走。

第 34 课的刷榜地图会把 Physics-IQ 写成单独一行：输入是真实实验录像的前文，输出是 5 秒续写，评分器是运动掩膜上的 IoU 和像素 MSE，样本来自 66 个实拍实验。那一行在地图上只占一格。本课把这一格拆开：协议怎么跑、四项指标各惩罚什么、视觉真实感分数和 Physics-IQ 是否同序、审计之后排名为什么会动、这根尺子仍然量不到规划。WorldModelBench 的物理维和 V-JEPA 2 的世界模型基准是另外两根，放在对照栏，不和 Physics-IQ 合成一个总分。

做完你要能口头完成三句话。第一句：Physics-IQ 考的是「真实实验接下来 5 秒运动发生在哪、何时、多少、像素像不像」，不是「这模型能不能给桌宠做 MPC」。第二句：Sora 在 Motamed 等原文表里视觉真实感最好、Physics-IQ 最低，两根尺子不同序，数字来自论文，不是观感。第三句：物理续写高分仍可以是第 33 课的 E0，因为这套协议没有动作端口。

下一课是第 45 课，把重建、token、扩散、JEPA、价值等价这些目标函数摊开，问 24GB 主线该改第 30 课的哪一块。本课给它准备的输入是：你已经能指出，优化「看起来真」的目标，不必提高 Physics-IQ；提高 Physics-IQ，也不必提高动作对换。

术语速查：

| 术语 | 一句人话 |
|---|---|
| 画面真 / 生成真 | 播出来的视频人眼或感知网络觉得像真的；第 17 课的视觉保真，本课用 MLLM 二选一识别率当对照 |
| 物理对 / 物理续写 | 给定前文，模型续写的运动位置、时刻、强度和像素是否贴近同一次真实实验 |
| 规划好 | 拿模型当模拟器去选动作，回真实环境或真桌子上的后果对不对；第 17 课的控制分数、第 03 课的动作对换 |
| Physics-IQ | Motamed 等用 396 段实拍实验录像做的续写基准，合成分数把物理方差标成 100 |
| switch frame | 8 秒录像里人工选定的切换帧：信息够起头，续写仍必须懂物理才答得上 |
| 物理方差 | 同一实验拍两 take，两段真录像之间的差异；模型的上限参照，不是 0 误差 |
| Spatial IoU | 把运动在时间上压成一张地图，比「动作发生在画面哪一块」 |
| Spatiotemporal IoU | 不压时间，逐帧比运动掩膜，位置对、时刻错也会被扣 |
| Physics-IQ Verified | 对原榜做提示、伪激活、样本级聚合三处审计后的版本，排名会动 |
| WorldModelBench 物理维 | 另一把尺子：指令有没有做完，再加五条物理违规的人工/裁判模型打分 |

## 2. 问题

「视频模型懂物理了」至少被用来指三件不同的事。一件是画面真：分辨率高、光影对、人眼骗得过。一件是续写对：球该弹到哪、水流该进哪只杯子、多米诺在鸭子处该停下。一件是规划好：给它「伸手」和「看左」两个动作，它能说出哪条会把杯子扫下桌，并且你真的按这条选动作。三件可以同时高，也可以一件很高、另外两件接近随机。混成一句话，桌宠会用错过滤器。

本课要处理四个具体问题。

第一，生成真和物理续写会脱钩。Motamed、Culp、Swersky、Jaini、Geirhos 的 Physics-IQ（arXiv:2501.09038，WACV 2026）把这件事写成可复现的表。他们测了 Sora、Runway Gen 3、Pika 1.0、Lumiere、Stable Video Diffusion、VideoPoet。协议是：给真实实验的前文（图生视频模型给 switch frame，多帧模型最多给 3 秒），生成接下来 5 秒，再和真实后文比运动掩膜与 MSE。合成分数把「两段真录像只差物理随机性」标成 100。最好的模型 VideoPoet（multiframe）只有 29.5。Sora 的生成视频最难被 Gemini 1.5 Pro 从真录像里挑出来（识别率 55.6%，接近 50% 乱猜），Physics-IQ 却是表里最低的 10.0。论文报告两者 Pearson $r = -0.46$，$p = 0.249$，相关不显著。观感排序和物理续写排序不是同一张榜。

第二，续写对仍不是规划好。Physics-IQ 的输入没有关节命令，没有「如果你推杯子」这种动作条件。它问的是 $P(\text{后 5 秒} \mid \text{前 3 秒, 文本})$，不是本课主干里的 $P(s_{t+1} \mid s_t, a_t)$。WorldModelBench（Li 等，arXiv:2502.20694）加了指令跟随和物理违规，仍然是「按文本/首帧生成一段视频，人来打违规」，不是在环境里选动作。V-JEPA 2 的世界模型基准走表征和问答，规划数字在 V-JEPA 2-AC 的机械臂成功率上，三套数不能加总。

第三，尺子自己会漂。Physics-IQ Verified（Rädsch 等，arXiv:2606.18943）审计原榜：57.6% 的样本被改过，超过 34.8% 的提示被改写，并改成样本级等权计分。六款图生视频模型上，原榜和 Verified 的排名 Kendall $\tau = 0.46$，中等但足够换第一名。Wan 2.2 从第一落到第三，P-Video 从第四落到最后，Grok Imagine Video 和 Hunyuan Video 1.5 超过 Wan 2.2。报分数必须写清用的是 Original 还是 Verified、提示是 `op` 还是 `bpp`。

第四，桌宠不能用生成视频好不好看来代替「杯子会不会下桌」。第 32 课的克制是：把「伸手」送进第 30 课的世界模型，预测轨迹上杯子会不会过安全边界，会则改动作。过滤器要的是动作条件下的落点，不是 5 秒广告片的观感。Physics-IQ 低分只说明这把续写尺子上的物理差，不说明视频模型没有别的用处；高分也不说明它已经能进动作回路。第 33 课写得很硬：无动作视频生成器是 E0，分辨率不升档。

## 3. 准备

- 会口头复述 [第 12 课](12_frontier_landscape.md) 的三问（动作起作用吗、离开视野的东西还在吗、有没有人真用它选过动作）和三分评测（预测准 / 生成真 / 规划好）。本课把「生成真」和「物理续写」再拆开，规划好仍回 [第 17 课](17_evaluating_world_models.md)。
- 读过 [第 15 课](15_vjepa2_in_practice.md) 开头把「理解物理」拆成探针、违反预期、规划成功率三种口径。本课的 V-JEPA 2 对照沿用那里已经核实过的数字，不重测。
- 知道 [第 33 课](33_embodiment_degrees.md) 的 E0：系统不接收动作，或动作对换完全不分岔。本课会把 Physics-IQ 高分重新放回这一档看一眼。
- 桌宠上下文：[第 30 课](30_desk_world_model.md) 的动作条件模型和 [第 32 课](32_ship_desk_pet.md) 的克制。没有训完也可以读，实验不依赖你的桌面权重。
- 软件：git、Python 3.10+、[uv](https://docs.astral.sh/uv/)（或 pip）、`ffprobe`（ffmpeg 自带）。评分脚本用 OpenCV 和 pandas，不需要 GPU。
- 磁盘：只跑 `--help` 和单元测试，仓库本身很小。若下载 Verified 全集（Hugging Face `Anates-Labs-Research/Physics-IQ-Verified`），那是 4K 实拍，体积以页面为准，本课不强制。
- 第 34 课若你还没写那张地图，先在本课笔记里给 Physics-IQ 留一行：输入、输出、评分器、样本从哪来。地图课以后对得上即可。

## 4. 学习目标

1. 能用一句话说清 Physics-IQ 的输入、输出、四项指标和 100 分的含义，并指出它没有动作端口。
2. 能从官方表抄出「模型、视觉质量、Physics-IQ」三列，判断排序是否一致；Sora 的两个数字必须来自 Motamed 等原文，不许用观感填。
3. 能解释 Physics-IQ Verified 改了提示、伪激活和聚合之后，为什么排名会动，报分时要写哪几个标签。
4. 能把 WorldModelBench 的物理维、V-JEPA 2 的 IntPhys 2 / MVP / 规划成功率，和 Physics-IQ 分成三列，不合成总分。
5. 能对公开集里两个实验写出「看对了会怎样、看错了会怎样」，并按 Spatial IoU 与 Spatiotemporal IoU 的规则给两段假想续写打分。
6. 能说明：物理续写高分仍可能是 E0；杯子会不会下桌，必须回到动作条件预测或真机规划，不能用生成视频好不好看来代替。

## 5. 原理

五个机制。前两个把三根尺子钉死、把 Physics-IQ 的协议写成可照做的输入输出。中间两个讲四项指标的数学、以及原文表里视觉质量和 Physics-IQ 为什么不同序。最后把 Verified、WorldModelBench、V-JEPA 2 和 E0 放回同一张桌。

### 5.1 三根尺子：画面真、物理对、规划好

第 17 课已经把「模型好」拆成预测准、生成真、规划好。本课在生成真里面再撕一刀。一张把水流画得晶莹的视频，可以同时满足「看起来像水」和「水从杯子里消失、从桌腿里长出来」。前者是画面真，后者是物理错。规划好还要再加一个动作端口：同一现在，推杯子和收手必须给出不同的未来，并且你按这个未来选动作。

三根尺子对应三个不同的泛函。记生成模型为 $g$，真实续写为 $x_{t:t+H}^\star$，感知距离为 $d_{\mathrm{vis}}$，物理代理距离为 $d_{\mathrm{phy}}$（本课就是运动掩膜 IoU 和 MSE），策略为 $\pi_g$。

画面真量的是分布或感知：

$$
V = D\bigl(p_g(\tau),\, p^\star(\tau)\bigr)
\quad\text{或}\quad
\mathbb{E}\,[d_{\mathrm{vis}}(g(c), x^\star)]
$$

Physics-IQ 原文用 Gemini 1.5 Pro 做二选一：一对视频里哪个是生成的，识别率越接近 50% 越真。这是 $V$ 的一种操作化，考官是多模态模型，不是物理老师。

物理续写量的是和这一次真实实验对齐：

$$
P_{\mathrm{IQ}} = f\bigl(d_{\mathrm{phy}}(g(c), x^\star),\, d_{\mathrm{phy}}(x^{\star(1)}, x^{\star(2)})\bigr)
$$

分母里的 $x^{\star(1)}, x^{\star(2)}$ 是同一实验的两个 take。真实世界每次扔球落点都会抖，100 分不是像素级复读，是「和另一次真实验一样对」。

规划好量的是复合量 $J(\pi_g)$：先用模型造出或过滤动作，再回真环境测后果。[第 03 课](03_mdn_rnn_action_conditioned.md)动作对换、[第 32 课](32_ship_desk_pet.md)「伸手会碰杯则截断」、V-JEPA 2-AC 的抓放成功率，都落在这一根。Physics-IQ 的 $c$ 里没有 $a$，所以 $P_{\mathrm{IQ}}$ 再高也推不出 $J$。

类比：拍一部撞球广告。画面真问的是台球桌绿不绿、高光漂不漂亮。物理对问的是母球撞击后，目标球是否进了真实录像里的那个袋。规划好问的是：你真的要按模型说的角度出杆，球会不会按你的意图走。广告片可以在第一问满分、第二问交白卷。类比失效处：真实撞球有连续多解，稍微偏一点仍合法；Physics-IQ 用单次实拍当答案，会把「物理上说得通、但和这一 take 不同」判成错。Verified 论文把这点写成已知局限，本课验收不要求你修它，但笔记里必须写上。

### 5.2 Physics-IQ 在考什么：前文、续写、真实答案

数据集是 66 个物理场景，每个场景三个机位（左、中、右），每个机位拍两 take，合计 $66 \times 3 \times 2 = 396$ 段。每段 8 秒、30 FPS、3840×2160，相机固定、没有运镜。五类：固体力学、流体、光学、热力学、磁学。仓库里的 `descriptions/descriptions_original.csv` 给出每条的文本描述和类别；当前副本里固体 228 行、流体 90 行、光学 48 行、热 18 行、磁 12 行（行数含三机位和 take，不是 66 个场景均分）。

协议把 8 秒切成 3 秒前文加 5 秒后文。前文给模型当条件，后文当标准答案。图生视频模型只拿到 switch frame，也就是 3 秒末那一帧；多帧模型最多吃满 3 秒。switch frame 是人工选的：信息够让你看清装置，但正确答案还没发生。多米诺例子里，切换点是第一块被拨倒、尚未碰到第二块。文本描述只讲到条件，不剧透后文。Stable Video Diffusion 是原文里唯一不吃文本的模型。

为什么必须实拍，而不是 Physion、IntPhys 那种仿真？论文的理由是分布偏移。互联网视频模型在自然影像上训练，拿游戏引擎里的方块去考，测到的可能是「仿真长得不像」，不是「物理不懂」。Physics-IQ 把考场做成和训练域同类的相机影像，代价是没有解析的质量、摩擦、弹性系数，只能拿第二次实拍当上限。

评测时必须按模型自己的分辨率和帧率做预处理，再把生成结果裁成刚好 5 秒。仓库的 `validate_generations` 写死了：每个输入目录要有 198 个 mp4（只评 take-1 的三机位，$66 \times 3 = 198$），每个长度 5 秒，文件名以 `0001_` 到 `0198_` 开头。这就是「完整跑闭源模型做不到」的工程含义：你得先让 Sora 或 Runway 吐出 198 段合规视频。本课体验停在入口、描述表、评分函数和官方已发表的数字。

第 34 课地图上的 Physics-IQ 行，可以按下面这张槽位填。输入：switch frame 或最多 3 秒实拍，外加不剧透的文本。输出：5 秒视频。评分器：Spatial IoU、Spatiotemporal IoU、Weighted spatial IoU、MSE，再除以物理方差合成。样本：66 场景 × 3 机位 × 2 take 的实拍，不是网上搜来的库存视频。它量物理续写。它不量动作对换，不量规划回报，不量离开视野后的物体恒常（相机是固定的）。

### 5.3 四项指标：在哪、何时、多少、长得怎样

常用视频指标 PSNR、SSIM、FVD、LPIPS 量的是像不像。论文明确说它们对「运动对不对」不敏感。Physics-IQ 改用量运动的掩膜。相机固定，相邻帧像素差超过阈值（仓库 `binary_mask_generator.py` 默认阈值 10）就标成运动。得到一段 $H \times W \times T$ 的二值掩膜视频。

Spatial IoU 把掩膜在时间上做 max，压成一张 $H \times W$ 的运动地图：这个像素上，整段 5 秒里有没有动过。然后和真录像的运动地图做交并比：

$$
\mathrm{SpatialIoU} = \frac{|M^\star_{\mathrm{sp}} \cap M^g_{\mathrm{sp}}|}{|M^\star_{\mathrm{sp}} \cup M^g_{\mathrm{sp}}|}
$$

它只问「动作发生在画面哪一块」。多米诺被鸭子截断时，鸭子左侧不该倒。模型如果把整列都推倒，并集变大，这根尺子会降。模型如果倒对了半列、但倒得太晚，这根尺子仍可能给高分，因为 max 把时刻吞掉了。

Spatiotemporal IoU 不压时间，直接在 $H \times W \times T$ 上做交并比。位置对、时刻错，这根会单独掉。原文图 6 的一贯现象：所有模型在 Spatial IoU 上相对好看，在 Spatiotemporal IoU 上难看。也就是「大概知道哪会动，说不准哪一帧动」。

Weighted spatial IoU 仍压到 $H \times W$，但按时间平均而不是 max。同一位置反复扫过（摆锤）和只经过一次（滚球）会被分开。公式是逐像素 $\min$ 之和除以逐像素 $\max$ 之和，形状像加权的交并比。

MSE 是逐像素均方误差，管「东西长得怎样、颜色有没有被偷偷换掉」。它最严，也最容易被运镜、切镜、物体幻觉打穿。论文讨论里写：Sora 常出现转场切镜，尽管提示要求固定机位，这会被 MSE 和 IoU 一起惩罚。这不是 Sora「完全不懂物理」的充分统计，是这套代理指标对切镜过敏。报 10.0 的时候必须把这条局限写在旁边。

四项合成分数。Original 把三项 IoU（越高越好）和一项 MSE（越低越好，取负号）加起来，再除以物理方差，使两段真录像得 100。仓库旧实现见 `physiq/calculate_iq_score.py::calculate_iq_score`，返回的是 0 到 100 的百分数。Verified 改成样本级：每个场景先各自除以自己的物理方差，三项 IoU 和一项「MSE 的方差/取值」各占 1/4，再对 198 条取平均，见 `physiq/calculate_iq_score_stable.py::IQTable`。`run_physics_iq.py` 一次跑会同时写出 Original（`final_score_orig`）和 Verified（`final_score_view`）。榜上投稿时，Verified 榜用后者。

没有任何一项直接检查能量守恒或动量守恒。交并比高，只说明运动的时空支撑和真实验重叠。球穿过鸭子却在同一块屏幕区域留下光斑，Spatial IoU 可能仍过得去。这是代理，不是牛顿定律检验。WorldModelBench 用人工标「是否穿透」，补的是另一侧。

### 5.4 原文分数表：视觉质量和 Physics-IQ 不同序

下面三列只抄 Motamed 等 arXiv:2501.09038v3 的表 1 和正文对图 5 的数字。视觉质量用 MLLM 二选一识别率，越低越真，随机是 50%。论文没有给每个模型一张 Excel，只在正文点名了四个识别率；没点名的格子写「正文未给」，不要补。

| 模型 | 视觉质量（MLLM 识别率，越低越真） | Physics-IQ（越高越好，上限 100） |
|---|---|---|
| Physical Variance（两段真录像） | 不适用 | 100.0 |
| VideoPoet (multiframe) | 77.3% | 29.5 |
| Lumiere (multiframe) | 最高提到 86.9% | 23.0 |
| Runway Gen 3 (i2v) | 74.8% | 22.8 |
| VideoPoet (i2v) | 正文未给 | 20.3 |
| Lumiere (i2v) | 正文未给 | 19.0 |
| Stable Video Diffusion (i2v) | 正文未给 | 14.8 |
| Pika 1.0 (i2v) | 正文未给 | 13.0 |
| Sora (i2v) | 55.6% | 10.0 |

表 1 的分项也抄在这里，避免只看合成数。Physical Variance：Spatial IoU 0.678，Spatiotemporal IoU 0.535，Weighted spatial IoU 0.577，MSE 0.002。VideoPoet (multiframe)：0.204 / 0.164 / 0.137 / 0.010。Sora (i2v)：0.138 / 0.047 / 0.063 / 0.030。Sora 的 MSE 是表里最差的 0.030，Spatiotemporal IoU 只有 0.047，和 Pika 的 0.041 同一档。

有 MLLM 数字的四行，视觉质量从真到假大致是 Sora、Runway、VideoPoet (multiframe)、Lumiere (multiframe)。Physics-IQ 从高到低是 VideoPoet (multiframe)、Lumiere (multiframe)、Runway、Sora。第一名和最后一名对调。论文说相关不显著，本课的观察就写到这一句：两根尺子在这张 2025 年初的表上不同序。不要外推成「所有更真的模型物理都更差」，样本只有八个系统，p 值也没有过 0.05。

仓库 README 的 Original 排行榜把这八行原样收进去，日期标 2025-02-19，Sora 仍是 10.0。后面 2025-2026 年 Magi-1、Cosmos 3、Sora 2 等另有投稿分，那是后训练和更强生成器的新数字，和 Motamed 表不是同一批模型。抄三列对照时，用原文表，不要把 Sora 2 的 Verified 26.5 填进 Sora 那一行。

低分的读法要克制。29.5 离 100 很远，论文用了 severely limited。它量的是「这些生成器在这 66 个实拍续写上，运动掩膜和像素对得有多齐」。VideoPoet 在 `paint-on-glass`（旋转刷子刮过玻璃上的红颜料）上可以看起来合理，在 `ball-in-basket`（篮球落入塑料筐）和 `cut-orange`（刀切橘子）上可以完全失败。同一模型，场景级方差很大。把 29.5 写成「视频模型毫无价值」，既超出指标范围，也被论文自己的成功案例打脸。流体在原文里往往好于固体，观察学习某些宏观流动，比学接触碰撞更像那么回事。这只是该数据集上的趋势，不是流体已解决。

### 5.5 尺子会漂：Physics-IQ Verified

Rädsch、Asano、Kuehne、Bauer、Jaini、Geirhos、Lüth 的审计针对三处测量误差。

提示。原描述偏场景陈列，有的事实错误、有的分不清切换帧前后已经发生了什么、有的漏掉决定物理结果的关键物。Verified 把每条拆成 SETUP、SCENE、ACTION、CAM、STYLE、SCOPE 六个字段，再用模型特定的 templater 拼成 `bpp`（best-practice prompt）。CAM 写成固定机位、实时、单镜头；STYLE 写成科学演示；SCOPE 写成不要长出新角色。仓库里 `physiq/templater/physiq_verified.py` 和 `uv run physiq/generate_descriptions.py sora2` 就是这条生产线。`op` 仍指向 `descriptions/descriptions_original.csv`。报分必须标明 `bpp` 还是 `op`。Sora 2 在 Verified 实验里从 `op` 换到 `bpp` 受益最大，因为更好的机位约束压掉了乱切镜头；Wan 2.2 是唯一换 `bpp` 之后变差的。

伪激活。IoU 吃的是像素差掩膜。转台在转、拍完之后有人进画、灯光闪一下，都会在「物理现象」之外点亮掩膜。审计把伪激活分成可从装置预料的、和录制时碰巧出现的。用 `end_effect_frames` 切掉现象结束后的运动，用 `freeze_areas` 在现象进行中挖掉无关区域。清洗之后，IoU 分数普遍下降，Wan 2.2 降得最多：它原来相对靠前，有一部分来自和装置运动重叠，不是现象本身。

聚合。Original 在全数据集上对 IoU 做「分子总和 / 分母总和」，低方差场景永远到不了 1，高方差场景可以超过 1，样本权重不相等。Verified 改成每个样本先算四项归一化分数再平均。这项改动在他们的六模型实验里几乎不改排名，但让你能把低分追到某一条 `0007_..._ball-hits-duck`。

结果：六款 I2V（Wan 2.2、HunyuanV-1.5、Cosmos3-N、Sora 2、P-Video、Grok Imagine Video）在 Original 与 Verified 之间 Kendall $\tau = 0.46$。仓库 Verified 榜（抄 README，日期以仓库为准）I2V 前几名包括 Cosmos3-Super-Image2Video $39.5 \pm 0.8$、Grok Imagine Video $34.8 \pm 0.6$、Hunyuan Video 1.5 $33.4 \pm 0.8$、Wan 2.2 $32.2 \pm 0.6$、Sora 2 $26.5 \pm 0.8$（`bpp`）。多帧档 Magi-1 24B 为 $48.4 \pm 1.1$，再加 GeoPhys 最优-of-N 到 $58.2 \pm 1.8$。58 仍然不是 100。Sora 2 在附录里被单独提醒：2025 年 10 月靠近发布时的一次 run 明显高于 2026 年 4 月；闭源 API 会漂，单次分数要写日期。

本课的纪律：讨论「画面真和物理对无关」时，锚在 Motamed 原文的八模型表；讨论「2026 年生成器进度」时，锚在 README 榜并写明 Verified / `bpp` / 日期。两张表不要横着比绝对值。

### 5.6 另外两根：WorldModelBench 与 V-JEPA 2

WorldModelBench 不拿实拍后文当像素答案。它给 350 条图文条件，覆盖驾驶、机器人、工业、人体、游戏、动画、自然七个域，让模型生成视频，再沿三个方向打分，总分上限 10。指令跟随 0 到 3（主体没动 / 动了但做错 / 做了一半 / 做完）。常识 0 到 2（帧质量、时间质量，来自 VBench 那一系）。物理 0 到 5，五条各 0/1：牛顿第一定律（无外力自己动）、质量守恒与固体（乱变形）、流体是否乱流、不可穿透、重力（飘起来）。67K 条人工标签，14 个模型。2B 的 VILA 裁判在他们的划分上，预测违规的误差低于 GPT-4o。

和 Physics-IQ 的差别要写进笔记。WorldModelBench 的物理维是「有没有犯五类显眼错误」，裁判是人或微调 VLM；Physics-IQ 是「和这一 take 的运动掩膜重叠多少」，裁判是像素差。前者能抓住「机械臂悬在空中反重力」，即使画面很美；后者能抓住「倒了对半边多米诺但时刻错了」，即使人眼觉得还行。前者的文本条件常常是任务指令（车左转、手臂开门），更靠近「听指令的视频世界模型」；仍然没有关节动作序列，也没有环境回报。论文正文写：当时最高的 Kling，只有 61% 的视频把指定任务做完，12% 违反质量守恒，11% 出现穿透。常识分高推不出世界模型分高：Luma 的帧质量和时间质量高于开源的 Mochi，指令跟随更差（做完任务 44% 对 53%），物理接近（4.13 对 4.14）。他们把 WorldModelBench 的物理维和 VBench 总分的模型对战胜率相关做成 0.28，帧质量维则是 0.69。这是另一份「画面真 ≠ 物理对」的独立证据，协议完全不同，不要把 4.13/5 和 Physics-IQ 的 29.5/100 换算。

V-JEPA 2（Assran 等，arXiv:2506.09985）走表征路线，不把续写画成像素再和 Physics-IQ 的掩膜比。第 15 课已经拆过三个口径，本课只把世界模型基准这一列接进来。Meta 同时发了 IntPhys 2、MVPBench、CausalVQA。IntPhys 2 用游戏引擎做成对视频，一条在某时刻打破物理，模型要指出哪一条不可能；人类接近满分，当时的视频模型（含 V-JEPA 2）接近随机。MVPBench 用几乎相同的视频对挡住捷径，V-JEPA 2 的成对准确率 44.5，高于 InternVL-2.5 的 39.9，仍远低于人类（博客口径人类在三份基准上 85% 到 95%）。规划是第四根：V-JEPA 2-AC 在真实 Franka 上零样本抓放，第 15 课引过的数字是伸够 100%、抓杯子 60%、抓盒子 20%、杯子搬运 80%、盒子搬运 50%。表征能在 IntPhys 上「惊讶」、能在机械臂上规划，都不是 Physics-IQ 的 5 秒像素续写。把 V-JEPA 2 的 77.3% SSv2 探针和 VideoPoet 的 29.5 Physics-IQ 排进同一张「谁更懂物理」的表，是在比较温度和公斤。

### 5.7 高分续写仍可能是 E0，杯子下桌不能看海报

第 33 课 E0：系统不接收动作，或动作对换完全不分岔。Physics-IQ、WorldModelBench 的主协议都是文本加首帧（或短视频）进、视频出。没有 $a_t$，动作对换没有定义，档位停在 E0。Magi-1 多帧 48.4、再加推理时搜索到 58.2，改变的是物理续写尺子，不改变「有没有动作端口」。把 Verified 第一名写成桌宠已经可迁移，违反第 33 课的是否题。

规划好在本课的落点是三件你已经做过的事。[第 03 课](03_mdn_rnn_action_conditioned.md)：同一潜状态换动作，预测必须分岔。[第 24 课](24_visual_foresight.md)：Visual Foresight 用模型展开、CEM 选动作。[第 32 课](32_ship_desk_pet.md)：伸手之前查询「杯子会不会过安全边界」，查询失败则 halt。这三件都不看 FVD，也不看 Physics-IQ。

桌宠上的反例要具体。假设你用某个 2026 年视频生成器，提示「一只手把杯子推向桌沿」，生成的 5 秒里杯子稳稳停在桌上，光影极好。画面真很高。若真实桌子上同样推力会把杯子送下去，物理续写和规划都失败。反过来，第 30 课那个糊成色块的小模型，只要在「伸手」条件下预测的杯子轨迹越过 5 cm 边界、你截断了，克制就成立。糊，但是动作条件下的落点对。Physics-IQ 会讨厌糊（MSE 差），桌宠安全层会喜欢分岔。两根尺子服务两个买家。

类比失效再写一次：Physics-IQ 的固定相机、5 秒、无动作，和桌上那只会转头的摄像头在时间尺度和干预方式上都不一样。你不能用「模型在多米诺上 Spatial IoU 高」推出「它知道我伸手之后杯沿会过线」。要推出后一句，必须在你自己的桌子上做动作对换，或至少在有动作标签的桌面轨迹上测分岔。

## 6. 源码导读

克隆后按这个顺序读。仓库根目录的评测入口是 `physiq/run_physics_iq.py`，默认 Verified；加 `--original_physics_iq` 才走原榜数据和 Original 分。

| 文件 | 是哪个零件 | 带着什么问题读 |
|---|---|---|
| `README.md` | 协议说明书 | Verified 与 Original 各用哪条下载命令？`bpp`/`op` 是什么？198 段 5 秒的文件名规则是什么 |
| `descriptions/descriptions_original.csv` | 原提示 | 选 `balls-collide` 和 `juice-in-water` 时，文本有没有剧透后文 |
| `descriptions/best_practice/descriptions_base.csv` | Verified 基础提示 | 和 original 比，CAM / STYLE / SCOPE 多了什么 |
| `physiq/templater/physiq_verified.py` | 提示模板 | `@register("sora2")` 怎么把六个字段拼成一家模型的习惯用语 |
| `physiq/generate_descriptions.py` | 提示生成入口 | `uv run physiq/generate_descriptions.py sora2` 写出哪份 csv |
| `physiq/binary_mask_generator.py::generate_mask` | 运动掩膜 | 阈值默认多少？真录像和生成录像的输出文件名差在哪 |
| `physiq/calculate_and_write_metrics_to_csv.py` | 逐视频指标 | `ViewPaths` 六个路径各指向谁：take-1、take-2、生成、三份掩膜 |
| `physiq/calculate_iq_score.py::calculate_iq_score` | Original 合成分 | 返回值是 0-100 还是 0-1？物理方差从哪些 `variance_*` 列来 |
| `physiq/calculate_iq_score_stable.py::IQTable` | Verified 合成分 | `final_score_orig` 和 `final_score_view` 哪个上 Verified 榜 |
| `physiq/run_physics_iq.py` | 评测入口 | `--input_folders` 为何是复数？四次 run 怎么进 `aggregate_runs_from_csvs.py` |
| `physiq/download_physics_iq_data.py` | 原榜下载 | Verified 分支为什么 `raise NotImplementedError`，数据该去哪下 |
| `physiq/tests/test_scoring_equivalence.py` | 评分回归 | 旧 `calculate_iq_score` 与 `IQTable.compute_final_score_orig` 在什么容差内相等 |
| `generated_videos_5s/README.md` | 生成物约定 | 目录名 `<model>-bpp-run_01` 少写哪一段会让聚合脚本对不上 |

读掩膜时对着 5.3 节。`generate_mask` 打开视频、按相邻帧差做阈值、写出二值 mp4。真录像文件名要嵌 `video-masks` 和 take 号；生成录像没有 take-2，函数会补 `_take-1_`。IoU 全部发生在这些掩膜上，不经过分割网络。所以转台、灯光、镜头脏点都会进分数。Verified 清洗的就是这个通道。

读 `IQTable` 时对着 5.5 节。构造函数把三机位的 list 列收成均值，再用 `joblib.hash` 锁住 DataFrame，后面算分如果有人改了表就抛 `RuntimeError`。`--lazy_integrity` 是给旧 pandas 的逃生口，README 建议升级 pandas 而不是长期开这个开关。`ORIG_SCORE_KEY = "final_score_orig"`，`VERIFIED_SCORE_KEY = "final_score_view"`，打印和 JSON 都用这两个键。投稿 Verified 榜时不要把 Original 的百分数交上去。

`download_physics_iq_data.py` 对 Verified 直接 `NotImplementedError`，注释写去 README 的 Hugging Face 链接。原榜走 `gs://physics-iq-benchmark`，需要本机 `gcloud storage rsync`。24GB 主线的人如果只想看样本，优先用项目页 https://physics-iq.github.io/ 上的 GIF，或只下 `switch-frames/` 加 `descriptions/`，不要默认把 4K 全集灌进笔记本。

## 7. 实验

目标不是刷出 Sora 的分数。目标是：仓库能在你机器上说话，两份公开实验你能用 Physics-IQ 的语言描述对错，官方三列表你抄得对，两段假想视频你能先按观感再按规则打分。完整 198 段闭源生成标成只讲。

### Step 0: 克隆仓库

```bash
git clone https://github.com/google-deepmind/physics-iq-benchmark.git
```

进入该目录之后的命令都在仓库根运行。软件许可是 Apache 2.0，数据是 CC-BY。这不是 Google 官方产品，README 末尾有免责声明。

### Step 1: 安装并跑通评测入口

优先 uv：

```bash
uv sync
```

没有 uv 时用 `pip install .`，并把下面所有 `uv run` 换成 `python`。系统依赖是 `ffprobe`。macOS 用 Homebrew 装 ffmpeg 即可。

跑入口帮助，确认参数和 README 一致：

```bash
uv run physiq/run_physics_iq.py --help
```

预期：必填 `--input_folders`、`--output_folder`、`--descriptions_file`；可选 `--original_physics_iq`、`--benchmark_base_folder`（默认 `.`）、`--n_process`（注释写单进程大约 9GB 内存，CPU 评掩膜时才有意义）、`--lazy_integrity`。缺 198 段视频时不要硬跑评测主流程，`validate_generations` 会断言失败。入口能打印帮助，本课的「评测入口」就算通。

### Step 2: 从公开描述里选出两个实验

不下载 4K 也能做这一步。仓库自带 `descriptions/descriptions_original.csv`。用下面脚本抽出固体碰撞和流体各一条，另加论文讨论过的对照行，存成 `inspect_two.py`：

```python
"""inspect_two.py ， 第 44 课 Step 2：读官方描述，不下载视频。"""
import pandas as pd

df = pd.read_csv("descriptions/descriptions_original.csv")
slugs = [
    "balls-collide",
    "juice-in-water",
    "duck-and-dominos",
    "liquid-on-duck",
    "paint-on-glass",
    "ball-in-basket",
    "cut-orange",
    "match",
]
df["slug"] = df["generated_video_name"].str.extract(r"trimmed-(.+)\.mp4")
out = (
    df[df["slug"].isin(slugs)]
    .drop_duplicates("slug")
    [["slug", "category", "description"]]
)
print(out.to_string(index=False))
print("rows in csv:", len(df), "unique slugs:", df["slug"].nunique())
```

```bash
uv run python inspect_two.py
```

预期：66 个唯一场景；`balls-collide` 类别 Solid Mechanics，描述是木桌两端管子里滚出蓝黄网球相向而行，固定机位；`juice-in-water` 类别 Fluid Dynamics，描述是饮料机把西柚汁倒进已经有一些水的杯子。文本停在「倒」和「滚」，不说碰撞后怎么弹、颜色怎么混。这就是 5.2 节「不剧透后文」在文件里的样子。

项目页 https://physics-iq.github.io/ 有若干场景的真录像 GIF 和各模型续写 GIF。能打开就对着看；打不开就只用 CSV 做纸面实验，笔记里写「未下载视频，按描述推断」。

### Step 3: 写「看对了会怎样、看错了会怎样」

本课指定两条主实验，再各用论文里的成败当注释。答案写进你的笔记，不提交给仓库。

实验 A，固体碰撞，`balls-collide`。看对了：两球从管口滚出，在桌面相遇，接触后沿大致相反的方向离开，桌面其它物体不出现、不消失，相机不动。看错了：互相穿过（不可穿透失败）、融成一个球、在接触前凭空拐弯、撞完变成别的物体、切到另一个镜头。Physics-IQ 会怎么罚：穿过但光斑仍在两球轨迹的并集上，Spatial IoU 可能虚高，Spatiotemporal IoU 和 MSE 会差；切镜会把整张运动地图点亮，四项一起掉。桌宠翻译：这对应「手碰到杯沿之后杯子往哪走」。生成器把碰撞画得很炫，不等于它预测对了动量交换。

实验 B，流体，`juice-in-water`。看对了：红或粉色液体进入已有水的杯中，液面上升，颜色在交界处混合，液体留在容器里，不从杯壁渗出。看错了：液体在接触水面之前消失、从桌面长出来、杯子满了还无限倒、颜色阶跃成另一种饮料品牌的广告色。原文观察过流体往往好于固体，Runway 在 `liquid-on-duck`（红液体浇到橡胶鸭上）上可以看起来合理，同一模型在固体切割 `cut-orange` 上可以失败。你的笔记写成：流体宏观外观好，不保证固体接触对，更不保证守恒。

对照行，论文图 7 和讨论里的例子，用来校准「成功不是全能」。VideoPoet (multiframe) 在 `paint-on-glass` 上刮颜料可以像那么回事；在篮球落入筐（`ball-in-basket`）和切橘子上可以错。Runway Gen 3 把燃着的火柴伸进水杯（`match`）时，论文记录过一种幻觉：火焰碰到水，画面里长出一根蜡烛并被点燃。每一帧都可以很干净，时间上的物体生成是物理上不可能的。MSE 会罚颜色和新生物体，Spatial IoU 不一定罚得动，因为新蜡烛的运动区域可以和真火焰的熄灭区域重叠得很离谱。四项要一起看。

`duck-and-dominos` 是原文用来说明 OOD 的场景：鸭子插在多米诺中间，只有鸭子一侧该倒。训练语料里「完整推倒多米诺」远比「中间卡一只鸭」常见。模型如果倒完整列，就是在复读模式，不是在读这个装置。Spatial IoU 对这个失败相对敏感，因为不该动的半边被点亮，并集变大。

### Step 4: 导入评分，不跑 198 段生成

仓库测试用合成 CSV 检查旧分和新分是否同方向。不需要视频：

```bash
uv run pytest physiq/tests/test_scoring_equivalence.py physiq/tests/test_descriptions.py -q
```

预期：`test_descriptions.py` 确认 `descriptions/data.yaml` 能再生出和 `descriptions_original.csv` 相同的表；`test_scoring_equivalence.py` 确认 `calculate_iq_score` 的 0-100 分和 `IQTable.compute_final_score_orig()` 的 0-1 分相差不超过 0.005（百分刻度）。测试文件开头写明：`compute_final_score_stable` / `final_score_view` 故意与旧公式不同，不在等价范围内。看到这一点，你就理解了 5.5 节「聚合改了、榜要写标签」。

再手动导入一次，把公式从纸上接到对象：

```python
"""probe_score.py ， 第 44 课 Step 4：导入 IQTable，用合成 CSV 算一次 Original 分。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path("physiq").resolve()))
from tests.conftest import make_csv
from calculate_iq_score import calculate_iq_score
from calculate_iq_score_stable import IQTable

path = make_csv(Path("/tmp"))
old, var = calculate_iq_score(str(path))
table = IQTable.from_csv(str(path))
orig = table.compute_final_score_orig()
print("original_0_100", old)
print("iqtable_orig_0_1", orig)
print("physical_variance_component", var)
```

```bash
uv run python probe_score.py
```

预期：两个 Original 数字对得上（0-100 与 0-1 差一百倍）。合成 CSV 默认 mse=0.1、st_iou=0.4、sp_iou=0.5、wsp_iou=0.6，测试文件里有手算。你算出来对不上，先查 `sys.path`，不要先怀疑论文。

### Step 5: 抄官方三列表，看是否同序

把 5.4 节那张表手抄进 `notes/physics_iq_three_rulers.md`（目录自定）。只填论文给过的格子。然后写三行观察：

1. 有 MLLM 数字的模型里，视觉最真的是 Sora（55.6%），Physics-IQ 最低（10.0）；Physics-IQ 最高的是 VideoPoet (multiframe)（29.5），MLLM 是 77.3%，比 Sora 更容易被认出是假的。
2. Runway 视觉第二（74.8%），Physics-IQ 第三（22.8），两榜都不在同一名次。
3. 相关在原文里不显著。本课结论停在「这张表上不同序」，不到「越真越不懂物理」。

禁止把 README 2026 年 Verified 榜上的 Sora 2 $26.5 \pm 0.8$ 填进「Sora 10.0」那一行。那是另一个系统、另一套提示、另一套聚合。若你想看进度，另画第二张表，表头写成「Physics-IQ Verified，bpp，仓库 README」，不要和 MLLM 识别率强行三列对齐（Verified 论文没给那八个 2025 年初模型的新 MLLM 分）。

### Step 6: 同一实验两段视频，先观感后规则

没有生成大模型也能做。用纸面两段，或项目页上真假 GIF。选定实验 A `balls-collide`。

视频 1，好看但违守恒。两球表面的绒毛和桌面木纹极清，接触瞬间有漂亮的运动模糊；接触后两球互相穿过，继续沿原方向滚出画面。视频 2，糙但落点对。压缩伪影明显、色彩发灰；两球在相遇处弹开，离开方向和真 take 大致一致，没有新生物体，相机仍锁定。

先按观感各打 0 到 10，只准看「像不像真摄像机」。再按 Physics-IQ 四项各打高/中/低，理由必须引用 5.3 节：

| 视频 | 观感 0-10 | Spatial IoU | Spatiotemporal IoU | Weighted spatial IoU | MSE | 你会让它给桌宠过滤扫杯吗 |
|---|---|---|---|---|---|---|
| 1 好看穿过 | 先写你的数 | 可能中（轨迹并集仍在通道上） | 低（接触后掩膜和真弹开分叉） | 低（穿过没有碰撞那一帧的往复） | 低或中（物体还在，几何错） | 否 |
| 2 糙但弹开 | 先写你的数 | 高 | 高或中 | 中 | 低（糊，像素差大） | 仍否：没有动作端口 |

最后一列两行都是否，是故意的。即使视频 2 物理续写更好，它仍是无动作生成，进不了第 32 课的查询槽。若你把视频 2 改成「同一历史，动作=伸手 vs 动作=收手，两段分岔且伸手把杯沿送过线」，最后一列才能改成「可以当过滤器的候选，仍要在自己桌子上测对换」。

流体版本用 `juice-in-water`：视频 1 把液面画成广告级焦散，但果汁倒进杯后杯子体积不变、液体从桌腿流出；视频 2 焦散很差，液面上升和颜色混合方向对。规则打分同样先写四项，再写最后一列。

### Step 7: 可选下载与完整评测命令

Verified 数据走 Hugging Face，README 写访问申请会自动过。先装客户端：

```bash
pip install -U huggingface_hub
```

```bash
hf download Anates-Labs-Research/Physics-IQ-Verified --repo-type dataset --local-dir physics-IQ-benchmark-verified
```

目录需含 `full-videos/`、`split-videos/`、`switch-frames/`、`video-masks/real/`。本课不强制这一步。磁盘或网络失败，记「未下载全集」，不影响验收。

原榜数据走 GCS。Verified 下载函数在代码里是 `NotImplementedError`，不要对 Verified 跑下面这条：

```bash
uv run physiq/download_physics_iq_data.py --fps 30 --original_physics_iq --benchmark_base_folder .
```

若你真的有 198 段已裁成 5 秒的生成视频，评测入口按 README 是：

```bash
uv run physiq/run_physics_iq.py --input_folders generated_videos_5s/my-model-bpp-run_01 --output_folder outputs --descriptions_file descriptions/best_practice/descriptions_base.csv --benchmark_base_folder .
```

四次独立 run 再聚合：

```bash
uv run physiq/aggregate_runs_from_csvs.py outputs/physics-IQ-benchmark-verified/results/my-model-bpp-run_01.csv outputs/physics-IQ-benchmark-verified/results/my-model-bpp-run_02.csv outputs/physics-IQ-benchmark-verified/results/my-model-bpp-run_03.csv outputs/physics-IQ-benchmark-verified/results/my-model-bpp-run_04.csv --score-type verified
```

原榜加 `--original_physics_iq`，描述改用 `descriptions/descriptions_original.csv`。单段 trim 的 ffmpeg 例子（多文件循环自行在 shell 里写，一个围栏只放一条命令）：

```bash
ffmpeg -y -i generated.mp4 -t 5 -r 24 generated_5s.mp4
```

没有 198 段视频却跑 `run_physics_iq.py` 主流程，会在 `validate_generations` 上失败。这是预期，不是安装错误。

## 8. 配置与预算

本课没有训练超参。预算按「你实际会碰到的磁盘和墙」写。

| 步骤 | 算力 | 时间 | 磁盘 | 档位 |
|---|---|---|---|---|
| Step 0-1 克隆、`uv sync`、`--help` | CPU | 10-20 分钟（视网络） | 仓库本身很小 | 必做，体验 |
| Step 2-3 描述表 + 看对/看错 | 无 | 1-2 小时 | 可忽略 | 必做，研究 |
| Step 4 pytest + `IQTable` 导入 | CPU | 几分钟 | 可忽略 | 必做，体验 |
| Step 5-6 三列表和两段打分 | 纸或编辑器 | 1 小时 | 可忽略 | 必做，研究 |
| 项目页 GIF | 浏览器 | 30 分钟 | 不落地 | 推荐 |
| Hugging Face Verified 全集 | 网络 | 以页面为准 | 4K 实拍，按页面 | 可选 |
| GCS 原榜 `gcloud rsync` | 需 Google 云权限 | 以桶为准 | 同上 | 可选 |
| 198 段闭源生成 + 四次 run | 闭源 API 账单或 24GB+ 开源视频模型 | 论文级 | 每 run 198 个 5 秒 mp4 | 只讲 |

`--n_process` 大于 1 时，README 警告单进程大约 9GB 内存。笔记本上保持默认 0。完整评测是掩膜和 IoU，不吃 GPU，吃的是解码和磁盘。

提示标签：`op` 用 `descriptions/descriptions_original.csv`；`bpp` 用 `descriptions/best_practice/descriptions_base.csv` 或 `uv run physiq/generate_descriptions.py sora2` 生成的模型专用表。混用标签再比分数，等于换了考题。

检查点：本课没有权重可存。要留的是笔记、pytest 输出、`--help` 文本、以及你抄表的出处（arXiv:2501.09038v3 表 1，或仓库 README 的哪一行、哪一天）。

## 9. 验收

量化线：

1. `uv run physiq/run_physics_iq.py --help` 打印出 `--input_folders` 和 `--original_physics_iq`。
2. `uv run pytest physiq/tests/test_scoring_equivalence.py physiq/tests/test_descriptions.py -q` 通过。
3. 笔记里有 `balls-collide` 和 `juice-in-water` 的「看对了 / 看错了」，每条至少指向 Spatial IoU 或 Spatiotemporal IoU 会怎么变。
4. 三列表的 Physics-IQ 列与 Motamed 表 1 一致：VideoPoet (multiframe) 29.5，Sora (i2v) 10.0，中间六行数字不错位。Sora 的视觉质量若填写，必须是 55.6%，并注明越低越真。
5. 三列笔记「画面真 / 物理对 / 规划好」每列指向具体尺子：画面真指向 MLLM 或第 17 课 LPIPS/FVD；物理对指向 Physics-IQ 或 WorldModelBench 物理维或 IntPhys 2；规划好指向动作对换、控制分数或第 32 课查询日志。
6. 有一句写明：Physics-IQ 高分仍可以是 E0。有一句写明：杯子会不会下桌，不能用生成视频好不好看来代替。

可视检查：Step 6 的表里，观感分和四项 IoU/MSE 允许打架。打架本身就是验收内容。两行的「桌宠过滤」若都填了是，回 5.7 节重写。

没有下载 4K、没有调用任何视频生成 API，只要 1-6 齐，本课算完成。

## 10. 排错

症状：`uv sync` 或 `pip install .` 之后 `import cv2` 失败。
原因：OpenCV 轮子和当前 Python 不一致，或环境没激活。
验证：`uv run python -c "import cv2, pandas; print(cv2.__version__)"`。
修法：确认 `uv sync` 在仓库根执行；不要把系统 Python 和 uv 环境混着用。

症状：`run_physics_iq.py` 主流程断言 `found N videos but expected 198`。
原因：`validate_generations` 要求恰好 198 个 5 秒 mp4，前缀 `0001_` 到 `0198_`。
验证：`ls generated_videos_5s/<run> | wc -l`，再用 `ffprobe` 看时长。
修法：本课到 `--help` 为止。真要评，按 README 把生成物裁到 5 秒并改名。

症状：视频时长是 8 秒或包含了 3 秒条件段。
原因：V2V 模型把前文也写进了输出。
验证：`ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 file.mp4`。
修法：只保留生成的 5 秒，README 写明不要包含 conditioning 段。

症状：pytest 里 Original 等价测试过了，但你以为那就是 Verified 分。
原因：`final_score_view` 故意不同。
验证：读 `test_scoring_equivalence.py` 文件头注释。
修法：Verified 榜只交 `final_score_view`，并标明 `bpp` 或 `op`。

症状：Hugging Face 下载要登录，或 `download_physics_iq_data.py --` 不带 `original` 就 NotImplementedError。
原因：Verified 的 GCS 路径还没接进这个函数，README 指定 HF 数据集。
验证：打开函数里 `if not verified` 分支，Verified 侧第一句就是 raise。
修法：Verified 用 `hf download`；原榜才用 `gcloud` 那条。

症状：自己生成的漂亮视频 Physics-IQ 很低，或糊视频某项 IoU 不低。
原因：切镜、幻觉物体、固定机位被打破会打穿 MSE；糊但运动支撑对，Spatial IoU 可以中等。
验证：把生成视频和真 take 并排，先看相机有没有动，再看运动区域。
修法：先修提示里的 CAM 字段，再谈模型。这正是 Verified 要 `bpp` 的原因。

症状：想把 Physics-IQ 第一名接到桌宠安全层。
原因：把续写尺子当成了规划尺子。
验证：问模型要不要 $a_t$。不要，就是 E0。
修法：回第 30 课做动作对换；过滤器只接受分岔成立的模型。

## 11. 前沿与改造

前沿怎么做。2025 年初 Motamed 表里最好 29.5。2026 年仓库 Original 榜上 Magi-1 多帧 56.0，Cosmos3-Super 多帧 59.7，再加 WMReward 或 GeoPhys 的最优-of-N 可以到 62-64。Verified 榜上 Magi-1 24B 多帧 48.4，加 GeoPhys 到 58.2。数字在涨。涨的方式值得拆：有的是更强的视频生成器（Magi、Cosmos 3、Wan 2.2），有的是推理时用奖励模型或几何物理采样多次取最优（WMReward、GeoPhys BoN），有的是换提示（`bpp`）。Yuan 等 arXiv:2510.21840 用 V-JEPA 2 的奖励信号去拧视频生成的物理，被 Verified 论文列为「这把尺子已经反过来驱动模型训练」的例子。尺子一进入训练循环，分数上涨有多少是真物理、有多少是过拟合 66 个场景，目前公开报告给不出干净的消融。本课不把 58 分写成物理已解决。

我们差在哪。规模差：198 段闭源生成和四次 run，是钱和 API 的问题。机制差有三处，钱买不来。其一，Physics-IQ 没有动作端口，桌宠要的 $P(s_{t+1} \mid s_t, a_t)$ 必须在自己的桌子上测。其二，掩膜 IoU 不是守恒量，穿过、切镜、装置伪激活都会污染；Verified 修了提示和伪激活，没有把动量做成指标。其三，单 take 当唯一答案，会把合法的多解判错。WorldModelBench 用违规类型补了「显眼物理错误」，V-JEPA 2 用 VoE 和真机规划补了表征与控制，三套仍不能加总。

动手改造清单（选做，各自独立）：

1. 样本级失败档案（推荐，纸面加仓库）。在 `descriptions_original.csv` 里再选一个固体和一个流体，按 Step 3 的格式写看对/看错，并标明四项指标各自最敏感的失败模式。预算两小时。预期：你能指出至少一种「Spatial 高、Spatiotemporal 低」的时间错误。失败判据：四项都写「低」且给不出机制。
2. 提示消融（需要任意一个你能调用的图生视频模型，哪怕很小）。同一 switch frame，分别用 `op` 和 `bpp` 生成 1 到 3 个场景，不做完整 198。预算一晚加 API 额度。预期：`bpp` 更少乱切镜头（Sora 2 在论文里的方向）；Wan 类模型可能不涨。失败判据：两份提示看不出差别，如实记录，样本太小本身就是结论。
3. 把掩膜阈值当立场。改 `binary_mask_generator.py` 的 `threshold_value`（默认 10），用两段项目页 GIF 或自己拍的固定相机短视频，看运动地图怎么变。预算一小时。预期：阈值太低，阴影和压缩噪声被算成运动，IoU 分母膨胀；太高，细小液体运动消失。失败判据：改阈值地图不变，先查是不是视频分辨率被缩放过小。
4. 桌面版 Physics-IQ 草稿（对准第 30、32 课）。用自己桌子上的固定摄像头拍「推杯」两次 take，切 3 秒加 5 秒，手算或用仓库掩膜函数比 Spatial IoU。预算半天。预期：两次真推杯的 IoU 远高于「一次真推、一次你用手把杯子拿起来假装」。失败判据：两 take 差异已经大到合成分无意义，说明桌面实验的物理方差高于实验室装置，分数不能直接和 29.5 比。

顺手复现的映射：Motamed 的核心结论「视觉真实感与物理续写无显著相关」在本课 Step 5 的四行表上方向可见，样本量为原文的有数字子集，不能重算 p 值。Verified 的「审计改变排名」在本课只能读表，六模型生成你做不起；缩小版对应实验是 Step 4 看到 `final_score_view` 与 Original 公式不等价。第 15 课 VoE「复杂场景掉到随机」对应 IntPhys 2，和 Physics-IQ 的 29.5 是同一句「离解决还远」的不同尺子。

## 12. 论文与延伸

1. Do generative video models understand physical principles?
   （Motamed, Culp, Swersky, Jaini, Geirhos，[arXiv:2501.09038](https://arxiv.org/abs/2501.09038)，WACV 2026）
   Physics-IQ 原文。带着问题读：表 1 的 29.5 和 10.0 各来自哪四项；图 5 的 MLLM 识别率为什么越低越真；讨论里 Sora 的切镜是指标过敏还是物理失败，作者自己怎么写。
2. Physics-IQ Verified
   （Rädsch 等，[arXiv:2606.18943](https://arxiv.org/abs/2606.18943)）
   尺子审计。带着问题读：三类提示错误哪一类会让正确生成在原则上不可能；伪激活清洗为什么让 Wan 2.2 掉得最多；Kendall $\tau = 0.46$ 时，你还愿不愿意只报 Original 第一名。
3. WorldModelBench: Judging Video Generation Models As World Models
   （Li 等，[arXiv:2502.20694](https://arxiv.org/abs/2502.20694)，NeurIPS 2025）
   指令加物理违规。带着问题读：五条物理 0/1 和 Physics-IQ 的掩膜 IoU 各漏什么；正文写 Kling 61% 做完任务、12% 质量守恒违规，这和「常识分高」为什么可以同时成立；2B 裁判比 GPT-4o 低误差，过拟合 350 条提示的风险在哪。
4. V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning
   （Assran 等，[arXiv:2506.09985](https://arxiv.org/abs/2506.09985)）
   表征路线对照，第 15 课精读过。这次只带一个问题：IntPhys 2 接近随机、MVP 44.5、Franka 抓盒子 20%，三份数字分别落在本课三根尺子的哪一根，为什么不能和 29.5 加总。
5. IntPhys 2
   （Bordes, Garrido 等，[arXiv:2506.09849](https://arxiv.org/abs/2506.09849)，第 15 课选读）
   违反预期搬进复杂合成场景。带着问题读：人类仍接近满分、模型掉到随机，这说明「观察学习物理」的瓶颈在数据还是在任务形式。
6. 回访 [第 17 课](17_evaluating_world_models.md) 的三分评测和 [第 33 课](33_embodiment_degrees.md) 的 E0。带着问题读：生成真可以把 E0 测得很漂亮，物理续写可以把无动作生成器测到 50 分以上，哪一句升不了档。

第 45 课把目标函数摊开：像素重建和扩散赌画面真，JEPA 赌表征对，价值等价赌决策对。本课的三根尺子会变成那一课选择器的纵轴。Physics-IQ 上变好，可能只是生成头更会续写实验录像；桌宠要带走的，仍是动作条件下会不会把杯子扫下桌。毕业标准还在第 32 课和第 33 课。



