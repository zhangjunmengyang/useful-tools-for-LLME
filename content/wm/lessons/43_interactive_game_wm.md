---
id: 43_interactive_game_wm
title: "可玩的世界：Genie 3 与 Matrix-Game"
summary: "实时交互加上数分钟一致性，开源系统现在能复现到哪一步？"
unit: frontier
play_tools: []
checkpoints:
  - "开源可玩系统的实测或诚实失败记录。"
  - "Genie 3 宣称与可验证事实分开的表。"
---

# 第 43 课：可玩的世界，开源现在能玩到哪一步

> 类型：体验（Matrix-Game 2.0，README 写明 NVIDIA 卡至少 24GB）+ 研究（Matrix-Game 3.0 记忆模块与推理时序）+ 只讲（Genie 3、产品 Oasis、HY-World 2.0 对照）<br>
> 建议周期：2-3 天<br>
> 硬件：单张 24GB 卡可尝试 Matrix-Game 2.0 推理；Matrix-Game 3.0 论文里的 720p、40 FPS 是 8 张卡跑 DiT 加 1 张卡跑 VAE 的部署数字，当前 README 没有写 24GB 可实时。无 CUDA 的机器完成本课源码精读、推理时序和检查表。Mac / 纯 CPU 不跑权重。<br>
> 锚定仓库：[SkyworkAI/Matrix-Game](https://github.com/SkyworkAI/Matrix-Game) 当前主分支（目录里同时放 1.0 / 2.0 / 3.0），对照 [etched-ai/open-oasis](https://github.com/etched-ai/open-oasis)<br>
> 产物：一份对着源码写下来的推理时序、一张可玩性检查表、一张 Genie 3 宣称与可验证事实对照表；若 2.0 在你的 24GB 卡上跑通，再附「转身再回头」视频和记录

## 1. 这一课做什么

第九幕读的是 2025-2026 年刷榜系统和配方。第 42 课把机器人基础模型拆成理解头、生成头、动作头：世界模型要的是 $P(s_{t+1}\mid s_t,a_t)$，VLA 要的是 $\pi(a\mid o,\text{instruction})$。本课换到游戏和可探索场景，看那颗生成头被做成「能玩的世界」时，实时交互和长程一致性各自卡在哪。毕业标准仍在 [第 32 课](32_ship_desk_pet.md)、[第 33 课](33_embodiment_degrees.md)，这里不加新的桌宠及格线。

主干循环没变：

```text
观察 先压成状态 再按动作预测下一状态 然后展开多条未来 给未来打分 最后选动作
```

可玩世界模型把「预测下一状态」做成了边看边播的像素流。玩家或智能体每给一个动作，画面必须在几十毫秒到一两帧的时延里更新；走开再回头，刚才那栋房子还得在。两件事叠在一起才叫可玩。少了实时，它是离线视频生成器；少了回头一致性，它是会听按键的幻灯片。

[第 11 课](11_genie_latent_actions.md) 讲过 Genie 怎么从无标签视频里挖 latent action。[第 12 课](12_frontier_landscape.md) 用 open-oasis 量过滑窗自回归的记忆半径：默认 32 帧、20 fps，大约 1.6 秒，转身一圈结构保不住。本课不重测那组帧数，数字直接引用你当时的记录。本课要补的零件是 2025-2026 年开源系统为了「边播边听动作」和「分钟级回头」实际加进去的东西：少步蒸馏、因果 KV 缓存、按相机位姿取回的记忆帧。

动手分两档，按硬件诚实走。Matrix-Game 2.0 的 README 写明 NVIDIA 卡至少 24GB（测过 A100 和 H100）、Linux、64GB 内存，有卡就按官方 `inference.py` / `inference_streaming.py` 冒烟，再用固定动作剧本做转身再回头，对照第 12 课 oasis 的崩坏量级。Matrix-Game 3.0 是当前仓库首页指向的最新版，宣称 720p 实时、长程记忆；它的 README 只写测过 A/H 系列、Linux、64GB 内存，没有写 24GB 可交互。写这课的机器上没有 CUDA，3.0 权重也没有在本课环境里跑起来。跑不了就精读 3.0 的记忆模块，把推理时序写成可以对照源码复核的文字。Genie 3 无权重，只讲。HY-World 2.0 生成的是可探索的 3D 资产，接口不是视频交互世界模型，当对照行。

做完你要能判断：一段网页演示很流畅，是不是实时世界模型；开源仓库写了 Interactive，动作延迟实际落在一帧还是一整段 clip。桌宠这边，延迟、KV、chunk 长度会回到 [第 32 课](32_ship_desk_pet.md) 的控制回路：世界模型若要先想再做，查询必须赶得上身体的节拍。

术语速查：

| 术语 | 一句人话 |
|---|---|
| 实时交互 | 动作进来之后，新画面要在人眼或控制回路能用的时延里出来，不是先把整段视频编完再播 |
| 流式生成 | 按时间一块一块往后编，已生成的块不再重算；通常靠因果注意力和 KV 缓存 |
| 动作延迟 | 从按下键到第一帧受这个键影响的画面出现，中间过了多少毫秒、多少帧 |
| 回头一致性 | 景物离开视野再回来，还是原来那份，不是按「这种地方一般长什么样」现编 |
| 滑窗自回归 | 生成第 $t$ 帧只看最近 $K$ 帧；第 12 课 oasis 的 $K=32$ |
| 少步蒸馏 | 把几十步去噪压成 1 到 4 步，换吞吐；Matrix-Game 3.0 用 DMD 一类分布匹配 |
| 相机感知记忆 | 按当前相机位姿和视场重叠，从历史里取几帧当条件，而不是把全部历史塞进窗口 |
| Plücker 编码 | 把相机射线写成六维几何特征，告诉模型「记忆帧」和「当前帧」差了怎样的视角 |
| research preview | 只有网页或订阅原型，无权重无公开 API；Genie 3 是这一档 |
| 可玩性检查表 | 帧率、动作延迟、回头一致性、动作对换，四项里缺一项就不能叫实时世界模型 |

## 2. 问题

「能玩」是一句广告词，落到工程上是四个可以分别测的量。本课要处理三件具体的事。

第一件是把实时说清楚。生成一段 10 秒、24 fps 的视频，和以 24 fps 的节拍吃动作、吐下一帧，不是同一件事。前者可以离线算一小时；后者要求每帧的计算预算钉死。开源仓库里的 `--interactive` 也经常不是游戏引擎那种逐帧轮询：Matrix-Game 3.0 的交互入口是每个 clip 用标准输入读一次键鼠，再把同一个动作重复几十帧。你要能指出延迟落在哪一层。

第二件是把长程一致性从观感里拆出来。[第 12 课](12_frontier_landscape.md) 已经证明：滑窗模型在窗口内可以很好看，窗口外结构必崩。Genie 3 的官方博客宣称数分钟一致、对交互改动大约记得一分钟。Matrix-Game 3.0 论文宣称分钟级序列上记忆稳定。这些是宣称。开源侧你能核的是：记忆单元存什么、查询键是什么、取回发生在去噪之前还是之后。

第三件是把开源复现到哪一步写成三档，不许把网页演示的流畅写成你测过的 24 fps。Genie 3 只讲。产品 Oasis 只讲。open-oasis 的崩坏帧数引用第 12 课。Matrix-Game 2.0 若在 24GB 上跑通，算体验。Matrix-Game 3.0 以当前 README 为准：能跑再测转身；跑不了，交推理时序。

一条界限先划清。HY-World 2.0、Marble 一类系统把世界存成 3DGS 或网格，转身一致性由资产保证，动作常常退化成相机移动。它们可以很好玩，但和「每一步 $P(o_{t+1}\mid o_t,a_t)$ 的视频世界模型」不是同一接口。本课可以拿来对照，不允许写成同一类系统。

## 3. 准备

- 前课：第 11 课的 latent action 和 $\Delta_t$ PSNR；第 12 课的三问、三分评测、oasis500m 崩坏记录（本课直接抄帧号，不重跑 `generate.py`）；[第 10 课](10_diamond_diffusion_world_model.md) 的扩散世界模型和 GameNGen 对照；[第 17 课](17_evaluating_world_models.md) 的预测准 / 生成真 / 规划好；[第 21 课](21_persistent_4d.md) 的「离开再回来」；第 32 课的控制回路延迟。
- Git 和能读 Python。本课主线不是训练。
- 磁盘：只读源码几乎不占空间。若下载 Matrix-Game 2.0 或 3.0 权重，预留数十 GB，以 Hugging Face 页面为准。
- 硬件三条路，写进笔记里你走的是哪一条：
  1. 无 CUDA：读仓库、写时序、填两张表。这是写这课的环境实际走的路。
  2. 单张 24GB NVIDIA 卡、Linux、内存尽量靠近 64GB：走 Matrix-Game 2.0 的 README。
  3. 多张 A/H 系列卡：才有资格尝试 Matrix-Game 3.0 的 `torchrun` 命令。不要在 24GB 单卡上把论文的 40 FPS 当成目标。
- 账号：Hugging Face token，下载权重时需要。
- 不要为了本课再克隆一份 open-oasis 去测崩坏。第 12 课的记录在，抄过来即可。

## 4. 学习目标

1. 用可玩性检查表的四项（帧率、动作延迟、回头一致性、动作对换）给一个新系统打分，缺哪项就明确说它还不能叫实时世界模型；
2. 说出 Genie 1 的三件套（视频 tokenizer、latent action、动力学）和 Genie 3 官方博客之间的档位差：机制能读的是论文，720p / 24 fps / 数分钟一致是宣称，无权重不能练；
3. 用第 12 课的 32 帧窗口解释 oasis500m 为什么转身必崩，并指出 Matrix-Game 3.0 把记忆挪到了窗口之外的哪一个查询；
4. 对着 `Matrix-Game-3/pipeline/inference_pipeline.py` 写出一段推理时序：clip 怎么切、动作何时进、记忆何时取、去噪几步、VAE 何时解；
5. 分清 Matrix-Game 2.0 的因果 KV 流式和 3.0 的双向窗口加记忆取回，以及开源 `--interactive` 是逐帧键鼠还是逐 clip 标准输入；
6. 把 HY-World 2.0 放在对照行：输出是 3D 资产，一致性由几何保证，不把它写成视频交互世界模型。

## 5. 原理

六个机制。每个仍走直觉、机制、数学、代码落点、验证。

### 5.1 可玩性检查表：四项里缺一项就停用这个名字

一段 24 fps 的演示视频，可以是剪辑师的成品。一个实时世界模型必须在四个互相独立的量上同时成立。

1. 帧率。显示节拍能不能跟上「像游戏」的预期。Genie 3 博客写 24 fps，模型页写 20-24 fps，都是官方宣称。Matrix-Game 2.0 README 写键鼠条件下 25 fps。Matrix-Game 3.0 README 和论文写 5B 模型 720p 最高 40 FPS。这些数字都要标来源，不能抄进「我测到了」。
2. 动作延迟。按键到第一帧受影响画面的时间。帧率高但动作按 clip 提交，延迟仍可能是一整段 40 帧。对桌宠，这一项比帧率更致命：安全层若要在伸手之前改写动作，查询必须落在控制周期里。
3. 回头一致性。景物离窗再回来，还是不是原来那份。第 12 课量的是这一项。滑窗模型在数学上做不到窗口外的同一性，除非记忆放在别处。
4. 动作对换。同一段历史只换动作，预测必须分岔。这是 [第 03 课](03_mdn_rnn_action_conditioned.md) 沿用到第 45 课的试金石。画面再真，动作盲就只是外推器。

四项可以拆开失败。GameNGen 在短时程的帧率和观感上很强，论文展示的是人在模型里玩 DOOM，没有把模型当规划器再回真游戏验收。open-oasis 有动作条件，但开源脚本是预录动作的离线生成，产品网页才是键盘实时。HY-World 2.0 回头一致性可以极强，因为它根本不靠网络记性。检查表的用法是逐项打，不许折成一个「可玩分」。

狭义世界模型仍是：

$$
P(o_{t+1} \mid s_t, a_t)
$$

实时只是对这个条件分布的推理预算加了墙：每一步的计算必须在显示周期内结束，并且 $a_t$ 必须是这一步的输入，而不是整段视频开始前写好的剧本。流式系统常常改成按块预测：

$$
P\bigl(x^{(k)} \mid x^{(k-1)}_{\mathrm{tail}},\, m_k,\, a^{(k)},\, p\bigr)
$$

$x^{(k)}$ 是第 $k$ 个 clip 的潜帧，$x^{(k-1)}_{\mathrm{tail}}$ 是上一段留下的短历史，$m_k$ 是取回的记忆，$a^{(k)}$ 是这个 clip 的键鼠，$p$ 是可选文本。块一长，动作延迟就被块长度绑死。

验证：拿任意系统填四列表，每一格写「测过 / 仓库能看出来 / 官方宣称 / 不适用」。Genie 3 的帧率和回头只能填宣称。open-oasis 的回头填第 12 课帧号。Matrix-Game 3.0 开源交互填「每 clip 读一次标准输入」，不要填「逐帧键盘」。

### 5.2 从 Genie 的潜动作到 Genie 3 的宣称

互联网录像通常没有手柄日志。Genie（Bruce 等，arXiv:2402.15391）的办法是：视频 tokenizer 把画面变成离散 token；latent action model 在相邻帧之间只留几个比特的瓶颈，逼编码器把「历史预测不了的变化」写成动作码；动力学模型根据历史 token 和动作码生成下一帧。第 11 课用 tinyworlds 练过缩小版。Genie 本尊约 11B，无公开权重，只讲。

Genie 2、Genie 3 没有论文级的架构说明书。2025 年 8 月 5 日 DeepMind 博客《Genie 3: A new frontier for world models》写的是：文本提示生成可导航世界，720p，24 fps，数分钟量级保持一致；自回归生成每一帧时要回看不断变长的轨迹，用户一分钟后回到某地，模型必须去翻一分钟前的信息；对交互造成的改动，视觉记忆大约延伸到一分钟前。能力列表还包括可提示的世界事件（改天气、加物体），以及把 SIMA 一类智能体接到生成世界里做目标。限制写得很干脆：智能体可直接执行的动作空间有限；多独立智能体互动仍难；真实地点做不到精确地理还原；文字常要写进提示才能清晰；连续交互是数分钟，不是数小时。发布方式是 limited research preview。模型页 [deepmind.google/models/genie](https://deepmind.google/models/genie/) 把实时写成 20-24 fps，并指向 Project Genie 原型。无权重、无公开训练配方。本课对它的全部数字都标成宣称。

这一档和开源的差别，主要不是「它画得更好看」。差别是外界无法复核四项检查表里的任何一项，也无法把同一套转身剧本打到同一套权重上。第 12 课已经把 research preview 写成不能练。一年过去，能练的距离没有缩短。

代码落点不在 Genie 3，在你第 11 课读过的 tinyworlds：`models/latent_actions.py` 的窄码本，`keep_rate = 0.0` 强迫动作通道干活。Genie 3 若真的从视频里学交互，第一问仍然要用对换来验；没有权重，这条验不了。

### 5.3 滑窗必崩，记忆必须挪走

oasis500m 生成第 $t$ 帧时实际学的是

$$
P(x_t \mid x_{t-K:t-1}, a_{t-K:t}), \qquad K = 32
$$

条件里没有 $x_{<t-K}$。第 12 课把 $K=32$、默认 20 fps 折成约 1.6 秒，并预言：细节漂移可以在窗口内开始，结构替换集中出现在景物离窗超过约 32 帧再回头的时刻。你当时的崩坏记录就是这条预言的数据。本课直接引用，不重测。

要在窗口外还认得那栋房子，只有三条机制路。把窗口做长、做便宜，钱和注意力工程能推进一截，但计算随长度涨。外挂显式记忆，按键查询历史，Matrix-Game 3.0 和 WorldMem 走这条。把世界写成 3D 资产，一致性交给渲染，Marble 和 HY-World 2.0 走这条，代价是动力学往往变弱。第 39 课会把滑窗、隐状态、记忆库三条忘法对在一起。本课只把第三条里「按相机取回视频潜帧」的那一种写到源码级。

open-oasis 的滑窗落在 `dit.py` 的 `max_frames=32` 和 `generate.py` 里动作切片的起点。本课不打开那个仓库做实验。

### 5.4 少步流式：GameNGen 与 Matrix-Game 2.0

扩散世界模型默认要走很多去噪步。交互要把步数压下来，并且改成因果：已经显示过的帧不再改。

GameNGen（Valevski 等，arXiv:2408.14837）把扩散模型当成 DOOM 引擎。训练分两段：先用 RL 智能体打游戏并录轨迹，再训一个以过去帧和动作为条件的下一帧扩散器。论文数字：单 TPU 上 20 fps；下一帧 PSNR 29.4；人在短片段上几乎分不清真游戏和模拟，即使自回归到约 5 分钟。无官方权重。它过第一问：动作是真手柄。第三问要看清方向：人在模型里玩，模型当引擎，没有谁用它选动作再回真 DOOM 报分。第 10 课已经把它写成对照，本课保持这一档。

Matrix-Game 1.0（arXiv:2506.18701）是 Minecraft 上的图像到世界模型，约 17B，两阶段（无标签环境理解再加动作标签），评测用他们提出的 GameWorld Score。它证明开源可以在可控性上超过 oasis 一类基线，但不是本课的实时主线。

Matrix-Game 2.0（arXiv:2508.13009，权重页写 1.8B）把问题收成：双向、多步的视频扩散太慢，交互世界必须少步、自回归、键鼠进每一帧。论文写约 1200 小时带交互标注的数据（虚幻引擎和 GTA5），动作注入模块，基于因果架构的少步蒸馏，宣称 25 fps 生成分钟级视频。Hugging Face 上的 `Skywork/Matrix-Game-2.0` 提供通用场景、GTA 驾驶、TempleRun 三份蒸馏权重。仓库 README 的硬件句是：NVIDIA 卡至少 24GB 显存（测过 A100 和 H100）、Linux、64GB 内存。这是本课唯一写明 24GB 可跑的交互权重。

2.0 的流式推理在 `Matrix-Game-2/pipeline/causal_inference.py` 的 `CausalInferenceStreamingPipeline`：按块推进（配置里 `num_frame_per_block: 3`），维护视觉 KV 以及键鼠各一份 KV，局部注意力半径 `local_attn_size` 不是 $-1$。这是把记忆放在 KV 窗口里，不是 3.0 那种外挂取回。`inference_streaming.py` 的默认动作来自 `utils/conditions.py` 的 `Bench_actions_universal`，先问你一张图的路径，再按预置动作表生成；分辨率在脚本里被 resize 到 $352\times 640$。所以 2.0 开源「流式」首先保证的是边生成边解码，不自动等于「你按一下 W，下一帧立刻前进」。

验证：24GB 卡上跑通 `inference.py` 冒烟，只说明权重能出视频。要过检查表的第一问，必须自己构造两条只差动作的剧本，看画面分不分岔。要过回头，必须让相机转出当前 KV 半径再转回来。这些若没跑，就写「未测」，不要用项目主页的 GIF 代替。

### 5.5 相机感知记忆：Matrix-Game 3.0 把历史变成可查询的帧

3.0 论文（Wang 等，arXiv:2604.08995）的判断是：2.0 一类因果少步模型能实时，但缺少稳定的分钟级记忆；把上下文直接拉长又难和实时部署共存。它的选择是：骨干仍用双向 DiT（从 Wan2.2-TI2V-5B 改来），当前 clip 内部可以互相看；窗口外的世界靠取回。取回不按像素相似度（高噪声阶段不可靠），按相机位姿和视场重叠。

训练时把序列切成「过去潜帧 / 当前要预测的潜帧」。键盘动作走交叉注意力，鼠标走自注意力，沿用 2.0 和 GameFactory 的分工。为了对齐后面的少步蒸馏，基座也要见过不完美的历史：维护误差缓冲 $\mathcal{E}$，把 $x_0$ 预测和真值的残差

$$
\delta = \hat{z}^{i} - z^{i}
$$

写进去，再按

$$
\tilde{x}^{1:k} = x^{1:k} + \gamma_{h}\delta, \qquad \tilde{m}^{1:r} = m^{1:r} + \gamma_{m}\delta
$$

污染短历史和记忆帧。推理时条件来自自己刚生成的潜帧，训练时若只喂干净真值，对不上。论文把这叫 error-aware；思想来自 SVI 一类「把预测误差再喂回去」。

记忆设计有两步。第一步，取回的记忆潜帧、最近的过去潜帧、当前带噪潜帧放进同一套自注意力，不再给记忆单独开一条交叉注意力支路。第二步，用相对位姿的 Plücker 特征标明「记忆帧相对当前目标差了怎样的相机」。推理时还可以把序列前部的一个潜帧当 sink，给整段一个粗的外观锚。位置编码上，记忆、历史、当前各自带它们在整段里的真实帧号；训练时还对每个注意力头扰动 RoPE 基数，减轻周期对齐造成的「远帧被抄近帧」。

少步部署走 DMD：双向学生按多段 rollout 模仿真正推理，上一段的尾当这一段的过去，记忆从在线更新的池子按当前相机取。第一段没有记忆，退化成图生视频。再加 DiT 注意力投影的 INT8、MG-LightVAE 剪枝、GPU 上的视锥采样近似。论文写 5B、720p、最高 40 FPS 时，用的是 8 张卡做 DiT、1 张卡做 VAE。28B MoE（两个 14B）论文有定性结果；3.0 README 写明虚幻加真实的混合数据和 28B「即将发布」，当前 Hugging Face 的 `Skywork/Matrix-Game-3.0` 只有 5B 的 `base_model` 和 `base_distilled_model`，面向虚幻场景第一人称。

开源推理里，记忆取回的实现就是 `utils/cam_utils.py` 的 `select_memory_idx_fov`，以及 `inference_pipeline.py` 在 `clip_idx > 0` 时构造 `x_memory` 的那一段。第 7 节会按执行顺序把它写成时序。这里先记一句验证标准：若取回的索引永远是最近几帧，记忆模块就退化成加长的滑窗；转身再回头时，你应看到取回索引跳回出发附近的旧帧。

### 5.6 对照行：3D 世界生成不是同一接口

HY-World 2.0（仓库 [Tencent-Hunyuan/HY-World-2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0)）把自己和 Genie 3、Cosmos、HY-World 1.5 划开：后者输出像素视频，播完即逝；它输出网格或 3DGS，可进 Blender / Unity / Unreal / Isaac。一致性由三维结构保证，实时来自普通光栅化，不靠每帧扩散。交互是在已经生成好的资产里导航，物理碰撞由引擎做。这解决了检查表的回头一致性，通常也解决了帧率；它没有回答 $P(o_{t+1}\mid o_t,a_{\text{伸手}})$。把杯子推下桌，要的是动力学，不是换一个相机轨迹。

第 12 课把 Marble 写成「用取消预测来买一致性」。HY-World 2.0 是同一家族的开源对照。本课检查表对它的填法：帧率适用（引擎渲染），动作延迟适用（相机或角色控制），回头一致性由资产保证，动作对换若动作只是移动相机则不适用。不要把它写进「视频交互世界模型已经开源到 Genie 3」这种句子里。

## 6. 源码导读

主仓库 `SkyworkAI/Matrix-Game` 根 README 只有版本入口。真正能跑的代码在子目录。按这个顺序读，每个文件带着问题进去。

| 路径 | 管什么 | 带着什么问题读 |
|---|---|---|
| `README.md` | 1.0 / 2.0 / 3.0 发布说明 | 当前首页指向哪一版？有没有 24GB 字样？ |
| `Matrix-Game-2/README.md` | 2.0 安装与推理 | 「至少 24GB」写在哪一句？`inference.py` 和 `inference_streaming.py` 差什么？ |
| `Matrix-Game-2/inference_streaming.py` | 流式入口 | 动作从哪来？是实时键盘还是 `Bench_actions_*`？图像被 resize 成多少？ |
| `Matrix-Game-2/pipeline/causal_inference.py` | 因果 KV 生成 | `num_frame_per_block`、`local_attn_size`、三份 KV（视觉 / 键盘 / 鼠标）各自何时写入？ |
| `Matrix-Game-2/utils/conditions.py` | 预置动作表 | `CAMERA_VALUE_MAP` 的 `0.1` 和键盘 4 维 one-hot 怎么对应「左转」？ |
| `Matrix-Game-3/README.md` | 3.0 安装与推理 | 硬件有没有写 24GB？clone URL 是不是 `Matrix-Game-3.0.git`？`huggingface-cli download` 的模型 ID 完整吗？ |
| `Matrix-Game-3/generate.py` | 3.0 命令行入口 | `--interactive`、`--use_base_model`、`--num_iterations`、`--num_inference_steps` 各自改哪条路径？ |
| `Matrix-Game-3/pipeline/inference_pipeline.py` | 非交互生成 | clip 如何切分？`x_memory` 何时从 `all_latents_list` 切片？ |
| `Matrix-Game-3/pipeline/inference_interactive_pipeline.py` | 交互生成 | `get_current_action()` 每个 clip 问几次？同一动作重复多少帧？ |
| `Matrix-Game-3/utils/cam_utils.py` | 记忆取回 | `select_memory_idx_fov` 的分数是视锥交还是采样点？`use_gpu=True` 走哪条？ |
| `Matrix-Game-3/test.sh` | 官方多卡脚本 | `SYNC_GPU_NUM=8`、`ASYNC_GPU_NUM=7` 说明 40 FPS 不是单卡数字 |
| `Matrix-Game-1/` | Minecraft 17B 基线 | 本课不跑，知道它是可控性而不是实时即可 |

三处必须用铅笔圈出来。

第一处，帧数公式写在 3.0 README 和 `generate.py` 两头。`clip_frame = 56`，`first_clip_frame = 57`，`past_frame = 16`。总帧数

$$
N = 57 + (N_{\mathrm{iter}} - 1)\times 40
$$

默认 `--num_iterations 12` 得到 $57+440=497$ 帧。潜空间按 Wan 的时间步长 4：$(57-1)/4+1=15$ 个潜帧；后续每个 clip 新出 10 个潜帧（40 像素帧），短历史 4 个潜帧对应 16 像素帧。`img_cond = latents[:, :, -4:]` 就是把上一段最后 4 个潜帧钉进下一段。

第二处，记忆取回只在 `first_clip` 为假时发生。`select_memory_idx_fov` 对每个当前视点候选，在过去帧里找视场重叠最大的索引；源码随后执行 `selected_index[-1] = 4`，把其中一个记忆槽钉在序列前部。论文文字里的 sink 是「可选保留第一帧潜变量」。读的时候把「论文的首帧」和「源码钉死的第 4 帧」写成两行，不要自动画等号。取回之后，记忆块的相对位姿被编成 Plücker，和当前窗口的 Plücker 在时间维拼接；`x_memory = src[:, :, latent_idx]` 从已经生成的潜帧库切片。无记忆的对照条件 `conditions_null` 把 `x_memory` 置 `None`，只在 `--use_base_model` 时做 CFG。

第三处，开源「交互」的粒度。`inference_interactive_pipeline.py` 的 `get_current_action()` 在终端里分别要一次鼠标（I/K/J/L/U）和一次键盘（W/S/A/D/Q），然后

```python
keyboard_condition_curr = actions['keyboard'].repeat(action_frames, 1)
mouse_condition_curr = actions['mouse'].repeat(action_frames, 1)
```

`action_frames` 第一段 57、之后 40。也就是：你每回答一次，模型承诺几十帧都执行同一个动作，再用 `compute_all_poses_from_actions` 把键鼠积分成相机外参，供记忆取回使用。这不是游戏循环。延迟的下界是一个 clip 的生成时间，不是一帧的生成时间。2.0 的 `inference_streaming.py` 更彻底：默认根本不问键，动作来自 `Bench_actions_universal` 拼出来的测试表。

`generate.py` 里单卡会自动关掉 FSDP：`ulysses_size <= 1` 时把 `t5_fsdp`、`dit_fsdp` 置假。官方示例命令带着 `--dit_fsdp --t5_fsdp`，那是给 `torchrun --nproc_per_node=$NUM_GPUS` 准备的。单卡照抄会在校验函数里被改掉，多卡才按 `test.sh` 的 7 或 8 进程走。

一个诚实标注：3.0 子目录 README 写 `git clone https://github.com/SkyworkAI/Matrix-Game-3.0.git`，独立仓库当前不存在；`huggingface-cli download Matrix-Game-3.0` 也缺组织名，权重卡是 `Skywork/Matrix-Game-3.0`。命令以 README 原文为准写进第 7 节，失败时按第 10 节的修正跑。2.0 的 clone 命令在 Hugging Face 模型卡里是完整的 `SkyworkAI/Matrix-Game`。

## 7. 实验

主线不是刷出 24 fps。主线是：把仓库当前状态读准，把 3.0 推理时序写成可复核的文字，用固定动作剧本定义「转身再回头」，再填可玩性检查表和 Genie 3 对照表。有 24GB 卡再加 2.0 冒烟。写这课的环境没有 CUDA，3.0 权重未在本课机器上跑起来；你的笔记要写明自己走了哪条硬件路，没有测到的格子填「未测」。

### Step 1: 克隆主仓库

```bash
git clone https://github.com/SkyworkAI/Matrix-Game.git
```

```bash
cd Matrix-Game
```

根目录应能看到 `Matrix-Game-1`、`Matrix-Game-2`、`Matrix-Game-3` 和根 `README.md`。3.0 子目录 README 另写了一条 `git clone https://github.com/SkyworkAI/Matrix-Game-3.0.git`，当前会 404，不要用那条当主路径。

### Step 2: 对照 README 的硬件声明

打开两个文件，把原句抄进笔记，不要改写：

- `Matrix-Game-2/README.md` 的 Requirements：NVIDIA GPU at least 24 GB memory（A100 和 H100 测过），Linux，64 GB RAM。
- `Matrix-Game-3/README.md` 的 Requirements：It supports one gpu or multi-gpu inference. We tested this repo on the following setup: A/H series GPUs are tested. Linux. 64 GB RAM。

3.0 没有「24GB 可交互」这句话。论文 40 FPS 的部署是 8+1 GPU。本课因此把 3.0 实时数字标成论文宣称，把 2.0 标成「README 声明 24GB 可推理」。你的卡若不到 24GB，2.0 也不要硬跑。

### Step 3: 读 3.0 入口参数

```bash
python Matrix-Game-3/generate.py -h
```

无 GPU 时这条可能在 import `torch` 处失败，改用直接读文件。确认这些开关存在：`--ckpt_dir`、`--image`、`--prompt`、`--num_iterations`、`--num_inference_steps`、`--interactive`、`--use_base_model`、`--use_int8`、`--vae_type`、`--size`。默认 `--num_inference_steps` 在 argparse 里是 50（基座），README 蒸馏示例写成 3。`--size` 默认 `1280*704`，README 示例写成 `704*1280`。笔记里写你打算用的那一组，不要混。

3.0 README 的蒸馏推理示例（多卡，随机动作）原文如下，复制时保持单条命令：

```bash
torchrun --nproc_per_node=$NUM_GPUS generate.py --size 704*1280 --dit_fsdp --t5_fsdp --ckpt_dir Matrix-Game-3.0 --fa_version 3 --use_int8 --num_iterations 12 --num_inference_steps 3 --image demo_images/001/image.png --prompt "A colorful, animated cityscape with a gas station and various buildings." --save_name test --seed 42 --compile_vae --lightvae_pruning_rate 0.5 --vae_type mg_lightvae --output_dir ./output
```

权重下载 README 原文是：

```bash
huggingface-cli download Matrix-Game-3.0 --local-dir Matrix-Game-3.0
```

若报找不到模型，改用第 10 节的 `Skywork/Matrix-Game-3.0`。本课不把下载成功当作验收必选项。

### Step 4: 把推理时序写成文字

这是跑不了 3.0 时的主交付。对着 `inference_pipeline.py` 的 `generate` 和 `inference_interactive_pipeline.py` 的同名方法，按时间顺序列出发生了什么。下面是一份必须能和源码对上的草稿，你要自己打开文件把行号补上。

1. 读入首帧图像，文本用 T5 编成 `cond`。VAE 把首帧编成 `img_cond`（时间维长度为 1 的潜帧）。蒸馏权重目录是 `base_distilled_model`，基座是 `base_model`。
2. 预先按 $N=57+(N_{\mathrm{iter}}-1)\times 40$ 准备键鼠序列。非交互路径走 `get_data(...)` 的随机动作；交互路径不预生成全长，进入循环后再问。
3. 对 `clip_idx = 0 \ldots N_{\mathrm{iter}}-1` 循环。`first_clip` 为真时窗口覆盖第 0 到第 56 帧（57 帧）；之后每次窗口长度 56 帧，其中 16 帧与上一段重叠，新生成 40 帧。
4. 交互路径在本段开始时调用一次 `get_current_action()`，把该动作 repeat 成 57 或 40 行，积分得到本段外参，拼进 `extrinsics_all`。
5. 用本段外参做当前窗口的 Plücker。第一段：`x_memory = None`。之后：在 rank 0 上调用 `select_memory_idx_fov(extrinsics_all, current_start_frame_idx, selected_index_base, use_gpu=True)`，再执行 `selected_index[-1] = 4`，广播索引；从 `all_latents_list` 取出对应潜帧作为 `x_memory`；记忆块相对当前参考位姿编成 Plücker，拼在当前窗口 Plücker 前面。
6. 按本段潜帧长度采样噪声，时间维前端用 `img_cond` 替换。时间步前 `img_cond.shape[2]` 个位置置 0，表示这些潜帧不当作噪声预测目标。
7. 蒸馏模型只跑条件分支 `self.model(**model_kwargs)`，步数按 `--num_inference_steps`（README 示例 3）。基座跑有条件和无条件两次，用 `guide_scale` 做 CFG，步数示例 50。调度器是 `FlowUniPCMultistepScheduler`。
8. 去噪结束后，`img_cond` 更新为最后 4 个潜帧。第一段把整段潜帧留下；之后只把最后 10 个潜帧当作本段新内容 `denoised_pred`。
9. VAE：默认同步 `stream_decode`；`--use_async_vae` 时把潜帧丢进另一张卡上的 worker。`test.sh` 在异步时用 7 个进程做 DiT。
10. 把 `denoised_pred` append 进 `all_latents_list`，供以后取记忆。拼视频。

把这十步改写成你自己的话，每一步后面跟一个函数名。缺函数名的步骤不算完成。额外写三行观察：

- 记忆查询发生在去噪循环之外，每个 clip 一次，不是每一步去噪一次。
- 同一 clip 内部动作是常数（交互路径）或预录曲线（非交互路径），没有「去噪到一半改键」。
- 论文的 40 FPS 量的是这一整条流水线在 8+1 卡上的吞吐，不是这十步在 24GB 单卡上的帧率。

### Step 5: 设计转身再回头的动作剧本

即使 3.0 跑不起来，剧本也要设计好，后面有卡才能原样执行。对照第 12 课 oasis 的 `spin` / `walk_and_return`，这里动作用 3.0 交互接口的离散键。

剧本 A，原地转身再回头。目标：让出发方向的景物离开当前 56 帧窗口，再转回。交互模式下每 clip 只能提交一个常值动作，所以转身必须拆成多个 clip。例如：连续若干段只交鼠标 `L`（右视）或 `J`（左视），`Q` 不平移；再连续同样段数交反向鼠标；最后两段交 `U`+`Q` 站定观察。把「你估计转了大约 180 度 / 360 度」写在笔记里，依据是每段 40 帧乘鼠标值 `0.1` 的积分，不要假装知道一度对应多少。

剧本 B，走出去再走回来。若干段 `W`+`U` 前进，一段 `L` 掉头，同样段数 `W` 走回，再掉头站定。

对照第 12 课时只比量级：oasis500m 在约 32 帧离窗后结构不保。3.0 若取回索引在回头 clip 跳到出发附近，机制上具备赢的资格；是否真赢，要看视频。没有视频就停在「机制具备资格，结果未测」。

若你走 2.0 的 24GB 路径，不要用交互标准输入，改用 `conditions.py` 的键鼠张量思路：在笔记本里写明 `camera_r` 持续多少帧、再 `camera_l` 多少帧。2.0 流式脚本默认会用 `Bench_actions_universal` 的随机测试表，那张表不是转身剧本。没有改代码就测不了回头，如实写「默认脚本不是回头实验」。

### Step 6: 24GB 路径（可选 Matrix-Game 2.0）

仅当你的机器满足 2.0 README 的 24GB 声明时做。命令按 2.0 README / Hugging Face 模型卡拆开写。

```bash
conda create -n matrix-game-2.0 python=3.10 -y
```

```bash
conda activate matrix-game-2.0
```

```bash
cd Matrix-Game-2
```

```bash
pip install -r requirements.txt
```

```bash
python setup.py develop
```

```bash
huggingface-cli download Skywork/Matrix-Game-2.0 --local-dir Matrix-Game-2.0
```

冒烟（把 checkpoint、配置、图片换成你本机的真实路径，下面保持 README 的占位写法）：

```bash
python inference.py --config_path configs/inference_yaml/{your-config}.yaml --checkpoint_path {path-to-the-checkpoint} --img_path {path-to-the-input-image} --output_folder outputs --num_output_frames 150 --seed 42 --pretrained_model_path {path-to-the-vae-folder}
```

预期：一段能认出场景的短视频，不是黑屏。不要在笔记里写 25 fps，除非你用墙钟时间除总帧数自己算，并写明分辨率是脚本里的 $352\times 640$，不是 720p。跑不起来就停，把报错抄进第 10 节对应行，主线仍以 Step 4 的时序为准。

3.0 的 conda 行若你要试，README 原文是：

```bash
conda create -n matrix-game-3.0 python=3.12 -y
```

没有多卡、没有改好的权重路径，不要进入 `torchrun`。

### Step 7: 填可玩性检查表

新建 `playable_wm_checklist.md`，四列必须同时出现：帧率、动作延迟、回头一致性、动作对换。系统至少覆盖下面七行。

| 系统 | 帧率 | 动作延迟 | 回头一致性 | 动作对换 |
|---|---|---|---|---|
| open-oasis 500M | 第 12 课离线生成，默认 20 fps 是导出参数 | 预录动作，无在线键 | 第 12 课你的崩坏帧号，32 帧窗口 | 有动作切片，第 12 课确认过端口 |
| Oasis 产品网页 | 演示，未测 | 网页键盘，未测 | 未测 | 未测 |
| GameNGen | 论文：单 TPU 20 fps | 论文：可玩 DOOM | 论文：约 5 分钟短片段人眼难分 | 动作条件成立；无公开权重 |
| Matrix-Game 2.0 | README 宣称 25 fps；你的墙钟或写未测 | 开源流式脚本默认预置动作表 | 无外挂记忆，受 KV 半径限制；未测则写未测 | 键鼠作为条件；对换未测则写未测 |
| Matrix-Game 3.0 开源推理 | 论文 40 FPS 为 8+1 卡；单卡未测 | 每 clip 一次标准输入，动作重复 40 或 57 帧 | 有相机取回；结果未测则写机制已读、视频未跑 | 键鼠进模型；对换未测则写未测 |
| Genie 3 | 博客 24 fps，模型页 20-24 fps，宣称 | 宣称实时；外界未测 | 宣称数分钟，交互记忆约一分钟 | 宣称可导航；无权重不能对换 |
| HY-World 2.0 | 引擎渲染，不是每帧扩散 | 资产内导航 | 由 3D 结构保证 | 若动作只是相机，则不适用 |

互动做法：先自己猜「缺哪一项就不能叫实时世界模型」，再看表。标准答案是四项都要。缺帧率只是慢模拟器；缺动作延迟只是离线渲染器；缺回头只是会听键的幻灯片；缺对换只是外推器。Genie 3 四项都是宣称，所以只讲。

### Step 8: 填 Genie 3 宣称与可验证事实表

| 条目 | 官方文本（2025-08-05 博客或模型页） | 本课能验证吗 | 你的填写 |
|---|---|---|---|
| 分辨率 | 720p | 否，无权重 | 宣称 |
| 帧率 | 博客 24 fps；模型页 20-24 fps | 否 | 宣称，且两页不完全相同 |
| 一致性 | 数分钟量级大体保持 | 否 | 宣称 |
| 交互记忆 | 约一分钟 | 否 | 宣称 |
| 发布形态 | limited research preview；模型页指向 Project Genie | 只能确认无公开权重 | 只讲，不能练 |
| 动作空间 | 限制条款：智能体可直接执行的动作有限 | 否 | 宣称加限制 |
| 和开源的关系 | 论文与博客均未给配方 | 开源最接近的实时线是 Matrix-Game 2.0/3.0 | 机制不同，不可把 3.0 视频当成 Genie 3 复现 |

把这张表和 Step 7 的表一起交。任何一格把「博客写了 24 fps」写成「本课测到 24 fps」，本课不及格。

## 8. 配置与预算

本课没有训练。预算全在读代码、可选下载和可选推理。

| 环节 | 硬件 | 说明 |
|---|---|---|
| Step 1-5、7-8 | CPU 即可 | 主线。写这课的环境走这条 |
| 2.0 权重下载 | 磁盘数十 GB 量级 | 以 `Skywork/Matrix-Game-2.0` 页面为准 |
| 2.0 `inference.py` 冒烟 | README：至少 24GB 显存 | 150 帧量级，分辨率脚本内为 $352\times 640$ |
| 3.0 权重下载 | 5B 基座加蒸馏加 T5 加 VAE | 以 `Skywork/Matrix-Game-3.0` 为准；28B 当前 README 写尚未放出 |
| 3.0 蒸馏推理 | 论文：8+1 GPU 才报 40 FPS | README 示例 `--num_inference_steps 3`、`--num_iterations 12`，约 497 帧 |
| 3.0 基座推理 | 同硬件上更慢 | `--use_base_model --num_inference_steps 50` |

超参以仓库为准，不要背论文里未在 README 出现的学习率。需要记的只有和检查表有关的几个：2.0 流式 `num_frame_per_block: 3`、去噪时间戳列表 `[1000, 666, 333]`；3.0 第一段 57 帧、此后每段 40 帧、记忆槽 5 个、蒸馏示例 3 步、VAE 选项 `mg_lightvae` 剪枝率 0.5 或 `mg_lightvae_v2` 剪枝率 0.75。

Mac / CPU：第 7 节除读文件和填表之外的命令都可以跳。不要改装 PyTorch MPS 去跑 Wan 系列代码，仓库没有这条路径。

## 9. 验收

- [ ] 笔记里有开源三档划分：2.0 按 README 可作 24GB 体验；3.0 以当前 README 为准，本课环境未跑通则交推理时序；Genie 3 与产品 Oasis 只讲；
- [ ] Step 4 的时序每一步都能指到 `generate.py` / `inference_pipeline.py` / `inference_interactive_pipeline.py` / `cam_utils.py` 中的函数名；
- [ ] 写清 3.0 开源 `--interactive` 是每 clip 一次标准输入、动作重复 40 或 57 帧，没有把它写成逐帧游戏循环；
- [ ] 写清论文 40 FPS 的硬件前提是多卡流水线，没有写进「我的 24GB 卡」；
- [ ] `playable_wm_checklist.md` 七行四列无空格，未测的格子写未测，宣称的格子写宣称；
- [ ] Genie 3 对照表里 720p、24 fps、数分钟一致全部带「宣称」，且注明博客与模型页帧率表述不完全相同；
- [ ] open-oasis 回头一致性引用第 12 课崩坏记录，本课没有重跑 oasis 的 `generate.py`；
- [ ] HY-World 2.0 出现在对照行，接口写成 3D 资产而不是视频交互世界模型；
- [ ] 若跑了 2.0，记录实际命令、分辨率、墙钟时间和是否改过动作表；没跑则写硬件原因；
- [ ] 能口头回答：四项里缺哪项不能叫实时世界模型；桌宠控制回路最不能缺的是哪一项（动作延迟）。

## 10. 排错

| 症状 | 最可能的原因 | 怎么确认 | 怎么修 |
|---|---|---|---|
| `git clone .../Matrix-Game-3.0.git` 失败 | 独立仓库不存在，代码在单体仓库子目录 | 浏览器打开该 URL | clone `SkyworkAI/Matrix-Game`，进入 `Matrix-Game-3` |
| `huggingface-cli download Matrix-Game-3.0` 找不到模型 | README 里的 ID 不带组织名 | 打开 Hugging Face 搜索 | 改成 `Skywork/Matrix-Game-3.0` |
| `generate.py` 在 import 处报 CUDA / no driver | 没 GPU，或没装对应 PyTorch | `python -c "import torch; print(torch.cuda.is_available())"` | 无卡走 Step 4，不要改设备字符串硬跑 |
| 单卡跑官方 `torchrun ... --dit_fsdp` | FSDP 需要分布式 | 看 `_validate_args` 是否把 FSDP 关掉 | 单卡直接 `python generate.py`，不要抄 `test.sh` 的 7/8 进程 |
| 显存爆掉 | 把 3.0 5B 720p 塞进 24GB | nvidia-smi | 停。3.0 README 未声明 24GB。改走 2.0 或只读 |
| `--interactive` 一直停在 Please input | 这是设计：每个 clip 问一次 | 对照 `get_current_action` | 先用非交互随机动作打通，再谈剧本 |
| 交互时画面几乎不转 | 只交了 `U`/`Q`，或鼠标值太小 | 打印 `mouse_condition_curr` | 连续多个 clip 交 `L` 或 `J`；单 clip 内动作是常数 |
| 2.0 `inference_streaming.py` 不像在玩 | 默认动作来自 `Bench_actions_universal` | 读 `generate_videos` | 要回头实验必须改动作表；没改就不要声称测了回头 |
| 2.0 画面比例怪 | 脚本把图 crop/resize 到 $352\times 640$ | 读 `frame_process` | 这是 2.0 开源推理分辨率，不是 720p |
| FlashAttention 编译失败 | 缺对应 CUDA 或 GPU 代数 | 编译日志 | 按仓库注释安装；装不上则本课停在阅读 |
| 把网页 demo 的流畅写进帧率格 | 检查表纪律破了 | 看格子是否出现「我测」 | 改回宣称或未测 |
| 想重测 oasis 崩坏 | 与课纲冲突 | 是否又 clone 了 open-oasis | 停。抄第 12 课记录 |
| 28B 权重 404 | README 写即将发布 | Hugging Face 文件列表 | 只用 5B 的 `base_model` / `base_distilled_model` |

还有一类「失败」要单独说：2.0 在你的 24GB 卡上很慢，远低于 25 fps。那不推翻 README 的硬件下限，只说明「至少 24GB」不是「24GB 达到论文帧率」。把墙钟写下来，帧率格填你的测量，附分辨率。

## 11. 前沿与改造

同一问题，公开系统分成三条，不要混成一句「2026 年已经实时」。

第一条，封闭实时交互。Genie 3 把分辨率、帧率、数分钟一致和可提示世界事件叠在一个 research preview 里，配方不公开。产品 Oasis 是网页或 API 上的实时演示，开源缩小版仍是第 12 课那份 500M。外界能做的只有看演示和读限制条款，不能做动作对换。

第二条，开源流式像素世界。GameNGen 证明扩散可以当 DOOM 引擎，无权重。Matrix-Game 1.0 把 Minecraft 可控性做成 GameWorld Score。2.0 把因果少步和键鼠条件开源到 24GB 能尝试的量级，分辨率是 540p 一类而不是 720p，记忆主要在 KV 局部窗。3.0 把双向窗口、相机取回、DMD 少步和量化写进可克隆的仓库，实时数字绑在多卡流水线上；当前放出的权重是虚幻场景第一人称 5B，不是论文里的 28B。WorldMem（Xiao 等，arXiv:2504.12369）把帧、位姿、时间戳存进记忆库，第 39 课会把它和滑窗、隐状态对在一起。本课只要求你能指出 3.0 的查询键是相机外参和视场重叠，存的是已经生成的潜帧。

第三条，取消视频预测、改存 3D。HY-World 2.0 和 Marble 用资产换一致性。检查表上回头几乎满分，动力学往往不在接口里。桌宠若要把杯子留下，[第 21 课](21_persistent_4d.md) 的 4D 状态比再做一个 720p 游戏生成器更近。

我们差在哪。规模差：Genie 3 的数据和算力外界不知道；Matrix-Game 的数据引擎是虚幻、3A 录制加真实视频，24GB 主线买不起。机制差有三处，钱解决不了。一处是动作接口粒度：开源 3.0 把交互做成 clip 级常数动作，和「每一帧一个 $a_t$」不是同一延迟。二处是记忆查询键：按相机重叠取回，擅长「转回头还是那栋楼」，不保证「那杯水被端走了」。三处是验证者：这些系统的裁判几乎都是人眼和自己的 GameWorld Score，没有人用它们在真环境里选动作再验收，第三问空着。

动手改造清单（推理级，不新训 5B）：

1. 时序对照表。把 Step 4 的十步画成三列：输入张量形状、发生文件、是否在去噪循环内。预算：半天阅读。预期：记忆取回在循环外，VAE 在循环后。失败判据：说不清 `x_memory` 从哪份列表切片，重读 `inference_pipeline.py` 里 `src = torch.cat(all_latents_list, dim=2)` 那几行。
2. 假想对换。同一段 `extrinsics_all` 前缀，构造 `W` 前进和 `S` 后退两份 `keyboard_condition`，写出你预期的第一问：分岔应出现在新 clip 的前几帧，而不是等记忆取回。预算：纸面一小时。失败判据：你预期「记忆会代替动作」，那是把查询键和动作端口弄混了。动作进的是 `keyboard_cond` / `mouse_cond`，记忆进的是 `x_memory`。
3. 延迟预算搬回桌子。假设桌宠控制周期 100 ms。用 3.0 的 clip 长度 40 帧，分别按 10 fps、24 fps、40 fps 算出一次动作提交锁死多少毫秒。预算：一页草稿。预期：即使 40 fps，40 帧仍是 1 秒的动作延迟，超过 100 ms 周期。失败判据：只比较帧率、不乘 clip 长度。这一条直接接到第 32 课：查询若要先想再做，块长度必须短于安全层能忍受的时间。
4. 加餐，24GB 卡上的 2.0 对换。改 `conditions.py`，固定一段历史后只翻转 `camera_l` / `camera_r`，各生成一次。预算：装环境半天加两次推理。预期：视角分岔。失败判据：两段视频几乎一样，先查动作是否真的写进 `conditional_dict['mouse_cond']`。不要把这次结果写成 3.0 或 Genie 3 的测量。

论文结论「相机感知取回能支持场景再访问」对应改造 1 和 2 的机制阅读，缩小设置预期能指出查询键，不能复现论文图 9 的分钟级视频。论文结论「少步蒸馏可到 40 FPS」对应改造 3：能复现的是「吞吐来自多卡加 3 步」，不能复现的是你把 40 填进 24GB 笔记。GameNGen「扩散可以当游戏引擎」对应 2.0 加餐：能复现的是动作条件端口存在，不能复现的是 20 fps DOOM。

## 12. 论文与延伸

1. Genie: Generative Interactive Environments（Bruce 等，[arXiv:2402.15391](https://arxiv.org/abs/2402.15391)，[第 11 课](11_genie_latent_actions.md) 主论文）。带着问题读：没有动作标签时第一问如何验？LAM 的编码器推理时为什么可以扔掉？Genie 3 若仍叫 Genie，你有什么证据说它还在用 latent action，而不是键鼠标签？
2. DeepMind 博客 Genie 3: A new frontier for world models（2025-08-05，[官方页](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/)）与模型页 [deepmind.google/models/genie](https://deepmind.google/models/genie/)。带着问题读：720p、24 fps、数分钟、约一分钟记忆，分别出现在哪一句？模型页为什么写成 20-24 fps？限制清单里哪一条直接否定「已经可以当通用智能体训练场」？
3. GameNGen: Diffusion Models Are Real-Time Game Engines（Valevski 等，[arXiv:2408.14837](https://arxiv.org/abs/2408.14837)）。带着问题读：RL 录数据给动作条件提供了什么保证？20 fps 的硬件是什么？第三问为什么仍算未展示？
4. Matrix-Game: Interactive World Foundation Model（[arXiv:2506.18701](https://arxiv.org/abs/2506.18701)）。带着问题读：GameWorld Score 的四类指标覆盖了检查表的哪几项、漏了哪项？17B Minecraft 模型和 2.0 的实时线是不是同一个产品问题？
5. Matrix-Game 2.0: An Open-Source, Real-Time, and Streaming Interactive World Model（He 等，[arXiv:2508.13009](https://arxiv.org/abs/2508.13009)）。带着问题读：因果少步解决的是帧率还是回头？局部 KV 和 3.0 的外挂记忆差在查询键？README 的 24GB 句和论文 25 fps 句是不是同一测量？
6. Matrix-Game 3.0: Real-Time and Streaming Interactive World Model with Long-Horizon Memory（Wang 等，[arXiv:2604.08995](https://arxiv.org/abs/2604.08995)）。带着问题读：第 3.2 节为什么不用像素相似度取回？误差注入的 $\delta$ 来自哪一个预测？第 3.4 节 40 FPS 的卡数写在哪一段？开源 `selected_index[-1] = 4` 和文字里的 sink latent 是否同一实现？
7. 对照：HY-World 2.0 仓库 [Tencent-Hunyuan/HY-World-2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) 与技术报告（仓库 README 指向的 arXiv）。带着问题读：输出是网格 / 3DGS 还是视频？「可玩时长无限」这句话依赖哪一种数据结构？把它放进检查表时，动作对换一列为什么常常填不适用？
8. 回访 [第 12 课](12_frontier_landscape.md) 的崩坏记录和三问，[第 17 课](17_evaluating_world_models.md) 的三分评测。带着问题读：本课检查表的四项，分别落在预测准、生成真、规划好的哪一根尺子上？规划好为什么在整张游戏世界模型表里几乎全空？

现在系统会：在开源游戏世界模型上区分实时、流式、记忆和可玩，并且能指出 24GB 实际能碰的是哪一版。下一课换一把不靠观感的尺子。画面可以继续真，续写仍可能违守恒。第 44 课用 Physics-IQ 把「懂物理」写成可操作的续写协议，桌面上那只杯子会不会下桌，不能再用生成视频好不好看来代替。
