---
id: 37_training_recipes
title: "flow matching 与 self-forcing"
summary: "同样的 DiT，换训练协议为什么能从整段往后编变成边看边播？"
unit: frontier
play_tools: []
checkpoints:
  - "一张训练配方对照表。"
  - "一条 teacher forcing 对自由 rollout 的漂移曲线。"
---

# 第 37 课：同一副骨架，训练协议决定能不能边看边播

> 类型：复现（第 03 课曝光偏差）+ 体验（Self-Forcing / Wan 1.3B 推理）+ 只讲（大视频模型蒸馏与后训练）<br>
> 建议周期：2-3 天<br>
> 硬件：曝光偏差与玩具开关 Mac / CPU 可完成；Self-Forcing README 写明至少 24GB NVIDIA；Wan2.1 T2V-1.3B README 写峰值约 8.19GB。没有 CUDA 就停在核对 README，写失败原因<br>
> 锚定仓库：[guandeh17/Self-Forcing](https://github.com/guandeh17/Self-Forcing)，开源教师 [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1)；对照 [ctallec/world-models](https://github.com/ctallec/world-models) 第 03 课探针<br>
> 产物：曝光偏差曲线、两条训练开关的玩具对照、一张配方表、一次官方推理记录或诚实失败记录

## 1. 这一课做什么

第九幕读的是 2025-2026 年刷榜系统、声音、训练配方和架构。毕业标准仍在
[第 32 课](32_ship_desk_pet.md) 的桌宠总装和 [第 33 课](33_embodiment_degrees.md)
的 E0 到 E5。上一课把声音拆成两条：麦克风是观察，喇叭可以是动作，配乐不是动力学。
本课不加新骨架，只换训练协议。零件是：训练时到底喂真实历史，还是喂模型自己刚吐出
的帧。

贯穿全课的循环还是这一条：

```text
观察 先压成状态 再按动作预测下一状态 然后展开多条未来 给未来打分 最后选动作
```

大视频模型通常停在「生成真」。同一副 Diffusion Transformer（扩散变换器：用
Transformer 做去噪或流匹配的骨干，[第 10 课](10_diamond_diffusion_world_model.md) 的 U-Net 换成了这块），双向注意力
可以把整段视频一起雕得很像，却必须等整段去噪完才能看第一帧。桌宠和交互世界模型
要的是边看边播：新观察进来，立刻出下一帧，还得听动作。CausVid、Self Forcing
这条线证明，把双向教师蒸馏成少步因果学生，再在训练里用 KV cache 做自回归
rollout，同一副 Wan 骨架就能从「整段往后编」变成「流式往外吐」。

本课能量到的，仍是 [第 03 课](03_mdn_rnn_action_conditioned.md) 那个 38 万参数
的 MDN-RNN。它在 CarRacing 的 $z$ 空间里已经把 teacher forcing 和自由 rollout
画成两条曲线。本课把那张图当成宪法，再读 2025-2026 的配方如何治同一种病。
大模型蒸馏标成只讲：Self-Forcing README 写训练用 64 张 H100、600 步、不到 2 小时，
24GB 主线训不动。24GB 能做的是：核对 README、加载公开权重跑一段推理；显存不够
就停，把报错写进笔记。

做完你手里有四样东西：第 03 课那张曝光偏差图（或重跑的一份）、一张玩具开关表
（数字是 03 课量级的示意，不是 Wan 的损失）、一张配方对照表、一次
Self-Forcing 或 Wan 1.3B 的官方推理记录。缺最后一项时，失败原因也算产物。

术语速查：

| 术语 | 一句人话 |
|---|---|
| teacher forcing | 训练每一步都喂真实历史，模型只需猜下一步；第 03 课 `trainmdrnn.py::get_loss` 就是它 |
| 曝光偏差 | 训练时见的是真历史，推理时吃的是自己的错；误差按复利涨 |
| 自由 rollout | 把模型自己的预测接回输入，滚出一段想象；交互时只能这么播 |
| 流匹配 | 学一条从噪声到数据的速度场，积分出样本；和扩散同属连续生成目标 |
| 双向注意力 | 每一帧可以看见未来帧；画质高，不能流式 |
| 因果注意力 | 每一帧只看过去和自己；能边生成边播，也更容易漂 |
| KV cache | 把已经算过的注意力键值存下来，新帧不用重算整段历史 |
| 蒸馏 | 用强教师的分布去教一个更快、通常还改成因果的学生 |
| DMD | 分布匹配蒸馏：让学生生成分布去贴教师分布，不必逐步模仿去噪轨迹 |
| 后训练 | 预训练骨架不动或只动一小部分，换协议、加动作、加 LoRA，把它改造成能交互的东西 |

## 2. 问题

同一副骨架，换训练协议，为什么有的只能整段往后编，有的可以边看边播？拆成四个
具体问题。

1. 一步准和多步稳不是一回事。[第 03 课](03_mdn_rnn_action_conditioned.md) 已经
   量过：teacher forcing 的一步误差可以压在偷懒基线下面，自由 rollout 仍一路上扬。
   [第 30 课](30_desk_world_model.md) 的桌面小模型会得同样的病。大视频模型把 $z$ 换成潜空间帧，病没变。
2. 连续生成目标被说成两个宇宙。Wan2.1、Cosmos-Predict2.5、以及
   [第 26 课](26_vla_vs_world_model.md) 讲过的 $\pi_0$，公开材料都写
   flow matching。[第 10 课](10_diamond_diffusion_world_model.md) 的 DIAMOND
   写扩散。二者都是「从噪声沿一条连续路径走到数据」。本课把它们写成同一家族，
   差别在路径和网络预测的量，不在世界观。
3. 双向教师画质高，却当不了交互引擎。生成第 $t$ 帧时如果注意力能看见第 $t+k$
   帧，你就没法在第 $t$ 帧到达时立刻播出。CausVid（Yin 等，arXiv:2412.07772）
   把多步双向教师蒸馏成少步因果生成器。Self Forcing（Huang 等，arXiv:2506.08009）
   再让训练时的 rollout 长得和推理一模一样。
4. 后训练有一张清单，不要和换骨架混在一起。渐进分辨率、分模态噪声、CFG、LoRA
   都是协议旋钮。Cosmos 3 的 Generator 默认双向去噪，所以还要再走一遍因果蒸馏
   才能当交互世界模型。Causal-rCM（Zheng 等，arXiv:2606.25473）摘要写明把这套
   配方接到了 Cosmos 3 上。那是后训练，不是换了一副新骨架。

一条界限：本课不编造蒸馏步数，也不编造刷榜分数，更不把论文摘要里的少步数写成你
复现出来的速度。配方表只填机制：目标函数、是否因果、训练时吃不吃自己的输出、
适不适合交互。VBench 一类数字留给论文作者。

## 3. 准备

- 第 03 课的环境和产物：独立虚拟环境、`exp_dir/vae/best.tar`、
  `exp_dir/mdrnn/best.tar`、`datasets/carracing` 里至少一条轨迹。没有它们，
  Step 2 的曝光偏差重跑做不了。跳过第 03 课的人先回去把 MDN-RNN 训出来，
  本课不重训那个 38 万参数的 M。
- [第 03 课](03_mdn_rnn_action_conditioned.md) 第 7 节的 `probe_mdrnn.py`。
  本课 Step 2 直接调用它的 `--probe drift`，不再贴一份 500 行的拷贝。
- [第 10 课](10_diamond_diffusion_world_model.md) 对扩散的加噪-去噪和概率流 ODE 有一页讲法。本课 5.2 节从那里接
  flow matching，不重讲 EDM 预条件。
- 大模型体验需要 NVIDIA GPU 和 CUDA 版 PyTorch。先读完 Self-Forcing README
  再决定装不装。仓库写明测试过的最低配置是 24GB 显存、Linux、64GB 内存。
  Mac / 纯 CPU 在 Step 4 核对完 README 即可停。
- 磁盘：Wan2.1 T2V-1.3B 加 Self-Forcing 的 `self_forcing_dmd.pt` 预留 20GB
  以上。14B 教师只在他们的蒸馏训练里出现，本课不下载。
- [第 30 课](30_desk_world_model.md) 若已有桌面世界模型，把那份 `drift_curve.png` 也放到手边。本课
  5.8 节会把它和视频配方对上号。

## 4. 学习目标

1. 指着第 03 课的漂移图，说出 teacher forcing 曲线为什么走平、自由 rollout
   为什么上扬，以及两条曲线第 1 步必须重合的原因；
2. 在白板上写出流匹配的条件速度场和回归损失，并说明它和扩散的 score 匹配
   为什么属于同一家族；
3. 解释双向 DiT 为什么不能边看边播，因果注意力加上 KV cache 解决了哪一截、
   没解决哪一截；
4. 按「教师 / 学生 / 训练时是否吃自己的输出」三列，口述 CausVid 与 Self Forcing
   的差别，能指到 Self-Forcing 仓库里的类名；
5. 填完一张配方表：目标、是否因果、训练是否吃自己的输出、适合交互吗；
6. 解释 Cosmos 3 的 Generator 为什么还要后训练才能当交互世界模型，并指出
   第 30 课小模型能搬回去的最小补丁。

## 5. 原理

八个机制。每个仍按直觉、机制、数学、代码落点、验证来走。大模型段落的代码
落在 Self-Forcing 仓库；小模型段落落在 ctallec。

### 5.1 训练协议才是「能不能交互」的开关

第 03 课把这件事写成了实验定义，本课只换一个说法。训练时，模型每一步看到的
输入来自真实录像，记作 $z_t$。它只负责猜 $\hat{z}_{t+1}$。损失是一步负对数
似然。这叫 teacher forcing。推理时，交互引擎没有真实未来可抄：第 1 步的
$\hat{z}_{t+1}$ 会变成第 2 步的输入，误差进入隐状态，再进入下一步。这叫
曝光偏差。

两条误差曲线的定义原样沿用第 03 课。teacher forcing 误差：每步都喂真实
$(z_t, h_t)$，记 $\lVert \hat{z}_{t+1}-z_{t+1}\rVert$，随预测时刻基本走平。
自由 rollout 误差：从 $t_0$ 出发，此后每步吃自己的预测（动作仍用真实动作，
保证两条曲线只差在状态来源）。$k=1$ 处必须重合，之后的裂口就是曝光偏差。

ctallec 的训练里没有任何自由 rollout。`trainmdrnn.py::get_loss` 一次吃完整段
真实 `latent_obs`，对着真实 `latent_next_obs` 算损失。训练序列长度
`SEQ_LEN = 32`。[第 04 课](04_controller_dream_training.md) 的梦、[第 30 课](30_desk_world_model.md) 的 2 秒想象、视频模型的流式播放，全部
发生在这条线右边。

2025-2026 的视频配方没有发明新病。它们发明的是：在更大的骨架上，用蒸馏和
self-forcing 让训练分布贴近推理分布。本课后面所有「forcing」名字，都是这一
段的变体。

验证就是 Step 2 的漂移图。没有这张图，后面读论文只是在换形容词。

### 5.2 流匹配和扩散是同一家族的连续生成目标

第 10 课把扩散写成：给干净样本 $x$ 加上深度 $\sigma$ 的噪声，网络学会从
加噪版走回数据；生成是沿噪声水平从大到小积分一条常微分方程。流匹配
（Lipman、Chen、Ben-Hamu、Nickel、Le，arXiv:2210.02747）换了一个问法：先指定
一条从噪声 $x_0$ 到数据 $x_1$ 的概率路径，再让网络直接回归这条路径上的
速度场。最常用的直线路径是

$$
x_t = (1-t)\, x_0 + t\, x_1, \qquad t \in [0,1]
$$

条件速度场不依赖于当前位置的复杂公式，就是终点减起点：

$$
u_t(x_t \mid x_1) = x_1 - x_0
$$

条件流匹配损失是速度回归：

$$
\mathcal{L}_{\mathrm{CFM}} = \mathbb{E}_{t,x_0,x_1}\big\lVert v_\theta(x_t, t) - (x_1 - x_0) \big\rVert^2
$$

生成时从 $x_0\sim\mathcal{N}(0,I)$ 出发，用 Euler 或更高阶求解器积分
$\mathrm{d}x = v_\theta(x,t)\,\mathrm{d}t$，走到 $t=1$。去噪步数就是求解器
步数。少步能稳，靠的是路径够直、网络预测的量在整段 $t$ 上尺度正常。这和第 10
课 EDM 预条件解决的是同一类数值问题。

高斯噪声路径下，扩散的 score 匹配和流匹配可以互相改写：score 指向「该往数据
走多少」，速度场是同一条路径上的时间导数。Wan2.1 技术报告（Team Wan，
arXiv:2503.20314）写自己「designed using the Flow Matching framework within
the paradigm of mainstream Diffusion Transformers」。Self-Forcing 的
`configs/self_forcing_dmd.yaml` 把 `denoising_loss_type` 写成 `flow`。
Cosmos-Predict2.5 和 [第 26 课](26_vla_vs_world_model.md) 的 $\pi_0$ 也用流匹配，一个生成视频，一个生成
动作块。目标同类，输出的随机变量不同。

类比失效处：把「流」想成水管里的水，容易以为只有一条确定轨迹。模型学的是
整团概率质量怎么搬，积分一次只是从这团分布里抽一个样本。世界模型要的正是
这种抽样，同一状态同一动作可以通向几种未来。

代码落点：Self-Forcing 的 `model/base.py` 从 `utils.loss` 引入
`get_denoising_loss`，配置键是 `denoising_loss_type: flow`。学生生成器是
`WanDiffusionWrapper(..., is_causal=True)`，教师分数网络
`real_score` 是 `is_causal=False` 的双向 Wan。同一份 wrapper，因果开关和
损失类型是协议，不是另一套宇宙。

验证：打开 `configs/self_forcing_dmd.yaml`，确认同时出现 `flow` 和 `dmd`。
前者是连续生成目标，后者是蒸馏目标。两者叠在同一套训练里。

### 5.3 双向去噪画质高，却不能边看边播

双向视频扩散或流匹配在整段潜空间视频上一起去噪。第 $t$ 帧的每一个 token
可以注意第 $t+k$ 帧。未来信息提高了时间一致性，也锁死了延迟：第一帧出图
之前，整段噪声都要走完求解器。交互世界模型做不到这件事。新观察或新动作
是在线到达的，你必须在它到达之后立刻吐下一帧，而且不能回头改已经播出去的
画面。

因果注意力切断对未来的可见性。生成第 $t$ 帧时只看 $1,\ldots,t$。KV cache
把已经算过的键值留下来，新帧只算自己那一截。Self-Forcing 的
`pipeline/self_forcing_training.py` 把 Wan 1.3B 的块数写死为 30，每帧序列
长度 `frame_seq_length = 1560`，cache 形状是
`[batch, kv_cache_size, 12, 128]`，12 和 128 对上 Wan2.1 README 里 1.3B
的头数和头维度。`default_config.yaml` 里 `num_frames: 81`、
`height: 480`、`width: 832`，和 Wan 官方 480P 推理一致。

因果只解决了「能不能流式」。它不自动解决曝光偏差。每帧一旦写进 cache，后面
所有帧都会吃它的误差。CausVid 项目页写：学生初始化用教师 ODE 轨迹，再做
不对称蒸馏（双向教师监督因果学生）。这是在用教师的未来信息，补偿学生推理时
看不见未来。补偿发生在训练，不发生在已经播出的那一帧上。

MAGI-1（Sand.ai 等，arXiv:2505.13211）走另一条对照：按 chunk（固定长度的
连续帧块）自回归，块内去噪、块间因果，支持流式。本课主线是配方，不精读
MAGI。记住它，是为了第 38 课把「AR token / 双向 DiT / 因果 AR-扩散」拆开时
有一个 chunk 因果的例子。

验证：Self-Forcing 的 `inference.py` 在配置里有 `denoising_step_list` 时走
`CausalInferencePipeline`，没有该字段时走多步的
`CausalDiffusionInferencePipeline`。少步加上因果，才是他们说的实时流式。
本课不报 FPS。

### 5.4 蒸馏：把慢的双向教师压成少步因果学生

直接从零训一个少步因果视频模型很难。公开配方几乎都从开源教师出发。Wan2.1
是这条线最常用的教师：T2V-1.3B 和 14B，Apache-2.0，README 写 T2V-1.3B 峰值
约 8.19GB，5 秒 480P 在 RTX 4090 上大约 4 分钟（未量化）。它是双向流匹配
DiT，适合当画质教师，不适合当交互引擎。

CausVid 的摘要写了两步。第一步：把预训练双向扩散变换器改成自回归变换器，
按帧（或按块）往外吐。第二步：把分布匹配蒸馏（DMD，Yin 等，CVPR 2024 /
NeurIPS 2024 的图像版延伸到视频）用到视频上。论文摘要原句是把 50 步扩散
教师蒸馏成 4 步生成器。仓库 `tianweiy/CausVid` 的推理示例写的是
Autoregressive 3-step。数字以仓库当前脚本为准，本课不把它们写成你机器上的
测量。

DMD 不要求学生逐步模仿教师的去噪轨迹。它让学生生成样本，再用一个「真分数
网络」（通常就是冻住的教师）和一个「假分数网络」（跟着学生更新的评论家）
估计学生分布和数据分布之间的差异，回传给学生。Self-Forcing 的 `model/dmd.py`
把这件事写成 `_compute_kl_grad`，注释指向 DMD 原文 arXiv:2311.18828。
`model/base.py` 里 `real_score` 冻住、`fake_score` 可训、`generator` 可训，
三套 Wan wrapper 对号入座。

不对称蒸馏是关键补丁。学生是因果的，推理时看不见未来；教师是双向的，训练时
看整段。用教师分布去拉学生，等于把「未来特权」蒸馏进一个推理时没有未来特权
的网络。CausVid 还加了 ODE 初始化：先让学生回归教师 ODE 轨迹上的配对，稳定
之后再开 DMD。Self-Forcing README 写：他们直接提供 `ode_init.pt`，ODE
初始化过程和 CausVid 仓库相同。

验证：打开 `model/base.py::_initialize_models`。生成器 `is_causal=True`，
教师 `is_causal=False`。再打开 `configs/self_forcing_dmd.yaml` 的
`real_name: Wan2.1-T2V-14B`。学生从 1.3B 因果骨架出发，教师分数可以是
更大的双向模型。本课 24GB 不训这一步。

### 5.5 Self Forcing：训练时就按推理来 rollout

CausVid 把学生变成因果的，训练时仍可能在教师轨迹或真实潜变量上做
teacher forcing。推理时学生吃的是自己刚生成的帧。分布再次错位。Self Forcing
的项目页原句是：训练时用 KV cache 做自回归 rollout，模拟推理过程。

机制可以缩成四步，全部发生在
`pipeline/self_forcing_training.py::SelfForcingTrainingPipeline.inference_with_trajectory`。

1. 把 KV cache 和 cross-attn cache 清零。新视频从空记忆开始，和推理一致。
2. 按块生成。配置 `num_frame_per_block: 3`，潜空间一共 21 帧
   （`image_or_video_shape: [1, 21, 16, 60, 104]`），对应像素约 81 帧。每一块
   走 `denoising_step_list: [1000, 750, 500, 250]` 这一串少步时间戳。
3. 随机挑一个去噪步回传梯度，其余步 `torch.no_grad()`。这是随机截断：整条
   rollout 太长，不能逐步 backward，就在每块里抽一步当真。注释写
   `Only backprop at the randomly selected timestep`。
4. 块生成完，用 `context_noise`（默认 0，即干净潜变量）再跑一次前向，把这块
   的 KV 写进 cache，下一块当历史用。历史来自模型自己，不是真实视频。

损失不再是逐步去噪的 teacher forcing。配置 `distribution_loss: dmd`，
`trainer: score_distillation`。`train.py` 按这个字段选择
`ScoreDistillationTrainer`。DMD 看的是整段 rollout 像不像教师分布，所以学生
必须先把视频自己滚完。滚的时候用 KV cache，推理时也用 KV cache。训练分布和
推理分布在「吃自己的输出」这件事上对齐了。

代价写在 README 里：DMD 版训练号称 data-free，不需要视频数据，但要用 64 张
H100 跑 600 步。`guidance_scale: 3.0` 是蒸馏时教师侧的 CFG
（无分类器引导：有条件和无条件预测的差值，用来加强文本条件）。这些超参以
仓库配置为准，不要抄到桌宠小模型上当魔法数字。

类比失效处：self-forcing 听起来像第 03 课「把预测喂回去」的放大版，差不多，
但多了两样第 03 课没有的东西。一样是少步蒸馏学生，不先蒸馏就 unroll 不起
扩散模型；另一样是分布级损失，不是逐步 $z$ 空间 L2。本课玩具开关用的是
第 03 课量级的逐步误差，用来看机制方向，不许写成 Wan 的损失。

验证：读 `inference_with_trajectory` 里 `exit_flag` 那一分支。没抽中的步在
`no_grad` 里走，抽中的步才对 `generator` 求梯度。再看块结束后
`context_timestep = ... * self.context_noise` 那次前向，那是在写 cache，不是
在算损失。

### 5.6 同一条线上的修补

Self Forcing 之后一年，公开论文基本在修三件事：更长、更稳、初始化别用错教师。
本课只记机制，不记榜。

Self-Forcing++（Cui 等，arXiv:2510.02283）针对短教师训不出长视频。教师自己
不会合成分钟级画面，学生外推时潜空间误差复利。做法是从学生自己生成的长视频
里采样片段，再请教师给指导。摘要宣称可把长度扩到教师能力的约 20 倍，算力
加大时演示过 4 分 15 秒。那是论文声明，不是本课测量。

Rolling Forcing（Liu 等，arXiv:2509.25161）针对严格逐帧因果把误差锁死。它
同时去噪一个滚动窗口，窗口内噪声水平对未来逐渐升高，帧与帧可以互相改；
每向前走一步，窗口吐出一帧干净画面。另外把起始帧的 KV 当成 attention sink
（注意力锚：始终留在 cache 里的全局上下文）。训练仍在「自己生成的历史」上
做少步蒸馏，窗口不重叠以省算力。Causal Forcing 的 README 写他们的分钟级
长视频扩展建在 Rolling Forcing 上。

Causal Forcing（Zhu 等，arXiv:2602.02214）针对 ODE 初始化用错教师。项目页
的论点是：用双向教师监督自回归学生的 ODE 轨迹，在帧级上不满足函数的单值性。
他们先把双向底模微调成自回归扩散模型，用这个因果教师做 ODE 初始化，然后再
走和 Self Forcing 相同的 DMD。仓库 [thu-ml/Causal-Forcing](https://github.com/thu-ml/Causal-Forcing)
写明推理环境和 Self Forcing 相同。Causal Forcing++（arXiv:2605.15141）把
ODE 换成因果一致性蒸馏，省掉离线 ODE 配对数据。本课不跑它。

Causal-rCM（Zheng 等，arXiv:2606.25473）把 teacher forcing 和 self-forcing
写成一对互补散度。一致性模型一侧接近前向、离线、teacher forcing；DMD 一侧
接近反向、对着自己生成的样本、self-forcing。摘要的实验结论是：teacher-forcing 的
一致性模型目前是 self-forcing DMD 最好的初始化。同一篇报告写把配方接到
Cosmos 3 上，用来做带动作条件的交互世界模型。这是第 2 节第 4 问的文献出处。

Diffusion Forcing（Chen 等，arXiv:2407.01392）更早，机制不同：给序列里每个
token 独立的噪声水平，让因果模型既能逐步吐新 token，又能对过去做不同程度的
去噪。它把「下一步预测」和「整段扩散」焊在同一套训练里，是 per-token 噪声
这个旋钮的来源。MAGI-1 的「块内噪声随时间单调升高」和 Rolling Forcing 的
窗口内递增噪声，都和这个想法亲戚。

### 5.7 其余旋钮：分辨率、分模态噪声、CFG、LoRA

配方不只是 forcing 名字。下面四个旋钮会在同一副骨架上反复出现。不确定的
数字不写。

渐进分辨率。先在低分辨率或短时程上把动力学训稳，再升到 480P / 720P。
Wan2.1 README 写 1.3B 能出 720P，但该分辨率训练有限，官方建议 480P。升分辨率
是后训练，不是换骨干。

分模态噪声。Diffusion Forcing 的 per-token 噪声是时间轴版本：同一段里，
有的 token 几乎干净，有的还很吵。全模态系统（第 35 课读 Cosmos 3）把图像、
视频、声音、动作放进不同通路；噪声是否按模态独立采样，以当时技术报告为准，
本课不写死。需要记住的只有：噪声时间表是训练协议，不是新模态本身。

CFG。训练时随机丢掉条件（文本、动作、首帧），推理时用有条件和无条件预测的
差值加强条件。扩散写法是
$\hat{\varepsilon}_{\text{uncond}} + s(\hat{\varepsilon}_{\text{cond}}-\hat{\varepsilon}_{\text{uncond}})$，
流匹配把 $\varepsilon$ 换成速度场即可。Wan2.1 README 对 T2V-1.3B 建议 `--sample_guide_scale 6`。
Self-Forcing 蒸馏配置是 `guidance_scale: 3.0`。两套数服务于不同阶段，不要
混抄。$s$ 太大，运动会僵、颜色会过饱和；太小，文本或动作端口变弱。

LoRA 后训练。24GB 上动不动 14B 全参数，通常只训低秩适配器：冻住骨干，插入
低秩矩阵去拟合新条件（动作、自己的桌子、自己的相机）。它改的是「在已有生成器
上加一个口」，不自动变成世界模型。加完 LoRA，仍要做动作对换。没有对换，只是
一个会跟着提示微调画风的生成器。

### 5.8 桌宠和小模型能搬回去的最小补丁

[第 30 课](30_desk_world_model.md) 的桌面世界模型参数量和 Wan 差四个数量级，病一样：teacher forcing
一步很漂亮，2 秒自由滚动就开始漂。视频配方里真正搬得动的只有一句。训练时
混入自己的预测。

最小补丁，对着第 03 课或第 30 课的逐步预测器：以概率 $p$ 把真实 $s_t$ 换成
上一步的 $\hat{s}_t$，再算下一步损失。$p=0$ 就是 teacher forcing；$p=1$
就是纯 self-forcing。第 03 课第 11 节提过 Scheduled Sampling（Bengio 等，
2015，arXiv:1506.03099），就是这个旋钮的 RNN 年代名字。视频这边后来加了
KV cache、少步蒸馏和 DMD，那三样 24GB 桌宠默认买不起。

Cosmos 3 的 Generator 还要后训练，原因现在可以一句话说完。它的生成通路用
双向注意力对噪声 token 去噪，画质和多模态对齐是按「整段一起看」训出来的。
交互世界模型要因果、要少步、要在自己的输出上稳住。Causal-rCM 做的就是把
TF 初始化和 SF 精炼接到这副骨架上。没有这步后训练，Generator 仍是高画质
视频生成器，过不了「边看边播」。

验证分两档。小模型：Step 3 的玩具开关，看训练损失变差的同时多步裂口有没有
变窄。大模型：Step 6 能跑就看它是否真的按块往外写视频；跑不了，配方表仍然
要填，失败写进 NOTES。

## 6. 源码导读

两处仓库，带着问题读。先小后大。

ctallec 这边只重读三个落点，细节见第 03 课：

| 文件 | 是哪个零件 | 带着什么问题读 |
|---|---|---|
| `trainmdrnn.py::get_loss` | teacher forcing | `latent_obs` 是不是整段真实 $z$？有没有把预测喂回去的分支？ |
| `models/mdrnn.py::MDRNN.forward` | 一步动力学 | 动作在哪一行拼进输入？ |
| `probe_mdrnn.py::drift_probe`（第 03 课胶水） | 曝光偏差探针 | 自由分支有没有误把真实 $z$ 喂回去？第 1 步两条曲线相等吗？ |

Self-Forcing 克隆之后按这个顺序读：

| 文件 | 是哪个零件 | 带着什么问题读 |
|---|---|---|
| `configs/self_forcing_dmd.yaml` | 蒸馏配方 | `denoising_loss_type`、`distribution_loss`、`num_frame_per_block`、`real_name` 各写了什么？ |
| `configs/default_config.yaml` | 推理形状 | `num_frames`、`height`、`width`、`causal` 是否对上 Wan 1.3B 的 480P？ |
| `model/base.py` | 三套 Wan | 谁 `is_causal=True`？谁冻住？`real_name` 默认值和 yaml 覆盖值差在哪？ |
| `model/dmd.py` | DMD 损失 | `_compute_kl_grad` 里 fake score 和 real score 怎么组合？CFG 加在哪一侧？ |
| `pipeline/self_forcing_training.py` | 训练时 rollout | KV cache 何时清零、何时写入？哪一步 `no_grad`，哪一步回传？ |
| `pipeline/causal_inference.py` | 少步因果推理 | 和训练 pipeline 是否共用同一套 cache 形状？ |
| `trainer/distillation.py` | 训练循环 | `ScoreDistillationTrainer` 一次 step 调谁的 `forward`？ |
| `train.py` | 入口 | `config.trainer` 四个取值分别对应哪个 Trainer 类？ |
| `inference.py` | 体验入口 | 没有 CUDA 会在哪一行失败？`low_memory` 的阈值是多少 GB？ |
| `demo.py` | GUI | README 推荐的交互入口；本课 CLI 优先，GUI 当加餐 |

三处最要紧的细节。

第一，教师默认值和配置覆盖不一致。`model/base.py` 里
`real_name` 默认 `Wan2.1-T2V-1.3B`，`self_forcing_dmd.yaml` 改成
`Wan2.1-T2V-14B`。读代码时以合并后的 OmegaConf 为准，`train.py` 先加载
`configs/default_config.yaml` 再合并用户配置。本课推理不加载 14B 教师，
只加载 `checkpoints/self_forcing_dmd.pt` 里的学生。

第二，`inference.py` 把设备写死成 CUDA。单卡、无 `LOCAL_RANK` 时是
`device = torch.device("cuda")`。免费显存低于 40GB 时走
`DynamicSwapInstaller` 给 text encoder 做动态换入。24GB 卡落在 low_memory
分支，不等于一定 OOM，也不等于 Mac 能跑。

第三，块大小 3 和 21 帧潜空间是写进配置的协议，不是网络结构的必然。改
`num_frame_per_block` 必须和学生权重一起改，本课不要动。

Wan2.1 作为教师仓库，本课只读两处：README 顶部的 1.3B 显存数字，以及
T2V-1.3B 那条 `generate.py` 命令。它证明「开源双向教师在 24GB 上能推理」；
它不能证明「这个教师已经是交互世界模型」。没有动作端口的文本生成视频，
过不了 [第 12 课](12_frontier_landscape.md) 的第一问。

## 7. 实验

三档不要混。Step 1 到 Step 3 是复现方向：CPU 就能做。Step 4 到 Step 6 是
体验：24GB NVIDIA 才尝试，失败就停。Step 7 是 Wan 1.3B 备选体验。Step 8
填表，不依赖 GPU。

### Step 0: 核对本课锚定 README

先把 Self-Forcing 的 README 拉下来读，再决定后面装不装。不要凭记忆写命令。

```bash
curl -sL https://raw.githubusercontent.com/guandeh17/Self-Forcing/main/README.md
```

预期：标题是 Self Forcing，作者含 Huang、Li、He、Zhou、Shechtman；Requirements
写 Nvidia GPU with at least 24 GB memory；Quick Start 里有
`huggingface-cli download` 两行和 `python inference.py`。把显存要求和推理
命令抄进笔记。curl 失败就改用浏览器打开
https://github.com/guandeh17/Self-Forcing ，以页面 README 为准。

### Step 1: 确认第 03 课产物还在

```bash
ls exp_dir/mdrnn/best.tar exp_dir/vae/best.tar
```

在 ctallec 仓库根目录执行，`exp_dir` 换成你第 01-03 课的实验目录。缺文件
就回到第 03 课把 M 训完。本课不重训。

### Step 2: 重跑曝光偏差曲线

用第 03 课的 `probe_mdrnn.py`。文件应和 `trainmdrnn.py` 同级。

```bash
python probe_mdrnn.py --logdir exp_dir --probe drift
```

预期仍按第 03 课四条来查：第 1 步两条曲线重合；teacher forcing 走平且低于
偷懒基线；自由 rollout 上扬；越过 32 步竖线之后继续漂。把
`exp_dir/mdrnn/drift_curve.png` 复制一份到本课笔记目录，注明「第 37 课重跑，
权重未改」。

这张图是后面所有 forcing 名字的比例尺。大模型论文用 VBench 说话，你用这张
图说话。

### Step 3: 两条训练开关（玩具数字）

下面这份脚本不是 Wan，也不是你的 MDN-RNN 实测。它在一条带立方项的一维
动力学上，用线性模型对比两种喂法。量级故意做成和第 03 课 $z$ 空间 L2 一个
数量级的示意：一步误差零点几，几十步自由滚动到 1 以上。打开
`--mode tf` 时，训练损失好看，滚动误差涨得快；改成 `--mode sf` 时，训练
损失变差，滚动误差涨得慢。若你这台机器上两条自由滚动曲线几乎重合，把该
结果写进笔记：线性模型对这条玩具动力学的曝光偏差不明显。方向仍以 Step 2
的真图为准。

把脚本存成 `recipe_switch_demo.py`，和探针放在同一目录。

```python
import argparse
import numpy as np

parser = argparse.ArgumentParser(
    description="Toy TF vs SF switch. Not Wan loss, not MDN-RNN.")
parser.add_argument("--mode", choices=["tf", "sf"], required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--unroll", type=int, default=8)
args = parser.parse_args()

rng = np.random.default_rng(args.seed)
T, N = 40, 256
x = np.zeros((N, T))
a = rng.normal(0.0, 1.0, size=(N, T))
x[:, 0] = rng.normal(0.0, 0.4, size=N)
for t in range(T - 1):
    x[:, t + 1] = (
        0.85 * x[:, t]
        + 0.35 * a[:, t]
        - 0.20 * (x[:, t] ** 3)
        + rng.normal(0.0, 0.03, size=N)
    )

w = np.array([0.0, 0.0])
b = 0.0
lr = 0.04 if args.mode == "tf" else 0.02
losses = []
steps = 600
for _ in range(steps):
    t0 = int(rng.integers(0, T - args.unroll - 1))
    idx = rng.integers(0, N, size=64)
    z = x[idx, t0].copy()
    acc = 0.0
    for k in range(args.unroll if args.mode == "sf" else 1):
        at = a[idx, t0 + k]
        yt = x[idx, t0 + k + 1]
        xin = x[idx, t0 + k] if args.mode == "tf" else z
        pred = w[0] * xin + w[1] * at + b
        err = pred - yt
        acc += float(np.mean(err ** 2))
        w[0] -= lr * 2.0 * np.mean(err * xin)
        w[1] -= lr * 2.0 * np.mean(err * at)
        b -= lr * 2.0 * np.mean(err)
        z = pred
    losses.append(acc / (args.unroll if args.mode == "sf" else 1))

def curve(feed_self):
    z = x[:, 0].copy()
    out = []
    for t in range(T - 1):
        xin = z if feed_self else x[:, t]
        pred = w[0] * xin + w[1] * a[:, t] + b
        out.append(float(np.mean((pred - x[:, t + 1]) ** 2) ** 0.5))
        z = pred
    return out

one = curve(False)
free = curve(True)
print("mode=%s train_first=%.4f train_last=%.4f" % (
    args.mode, float(np.mean(losses[:30])), float(np.mean(losses[-30:]))))
print("step one_step free_rollout")
for k in [0, 7, 15, 23, 31, 38]:
    print("%d %.4f %.4f" % (k + 1, one[k], free[k]))
```

先开 teacher forcing：

```bash
python recipe_switch_demo.py --mode tf
```

再开 self forcing：

```bash
python recipe_switch_demo.py --mode sf
```

示意读数（你的种子会不同，只看形状）。TF：训练损失可以从 0.1 落到 0.01
量级，一步误差约 0.3，第 32 步自由滚动到 1 以上。SF：训练损失通常更高，
一步误差更大，自由滚动的斜率应当更平。若 SF 的自由滚动反而更差，记一次
失败：这条线性玩具太瘦，说明不了 Wan 级 self-forcing，只能说明「训练损失
不是多步质量」。真正的多步证据仍然是 Step 2。

互动就这两条开关。不要把打印出来的小数写进配方表当成 Wan 或 Cosmos 的
指标。

### Step 4: 克隆 Self-Forcing 并确认有 CUDA

没有 NVIDIA 卡的人到这里停。把 Step 0 读到的「at least 24 GB」和
`inference.py` 里的 `torch.device("cuda")` 写进 NOTES，本课体验档记失败，
原因是无 CUDA。有卡继续。

```bash
git clone https://github.com/guandeh17/Self-Forcing.git
```

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-cuda')"
```

第二行若打出 `False`，停。不要强行改成 `mps`：官方脚本没有这条路径。

### Step 5: 按 README 装环境并拉权重

在 `Self-Forcing` 根目录。下面每条都是 README 原命令拆开，一条围栏一条
命令。conda 若没有，改用你自己的 venv，但 Python 版本跟 README 写 3.10。

```bash
conda create -n self_forcing python=3.10 -y
```

```bash
conda activate self_forcing
```

```bash
pip install -r requirements.txt
```

```bash
pip install flash-attn --no-build-isolation
```

`flash-attn` 编译失败时停，不要抄来路不明的预编译包。把完整报错贴进
NOTES，体验档可以改走 Step 7 的 Wan 1.3B（它不依赖 flash-attn）。

```bash
python setup.py develop
```

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B
```

```bash
huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
```

预期：`wan_models/Wan2.1-T2V-1.3B` 下有官方权重目录，仓库根目录出现
`checkpoints/self_forcing_dmd.pt`。磁盘不够或 Hugging Face 超时，写失败
原因，不要改用第三方镜像除非你清楚许可证仍是原仓库的。

### Step 6: 官方少步因果推理

README 的 CLI 示例如下。提示文件若仓库 `prompts/` 里没有
`MovieGenVideoBench_extended.txt`，改成该目录下一个实际存在的 `.txt`
（先 `ls prompts`），或自己写一行详细英文提示存成 `prompts/one.txt`。
官方说明：模型更吃长而具体的提示。

```bash
python inference.py --config_path configs/self_forcing_dmd.yaml --output_folder videos/self_forcing_dmd --checkpoint_path checkpoints/self_forcing_dmd.pt --data_path prompts/MovieGenVideoBench_extended.txt --use_ema
```

预期：打印 `Free VRAM ... GB`；24GB 卡上应走 low_memory 分支（阈值写在
`inference.py`，免费显存小于 40GB）；结束后 `videos/self_forcing_dmd/`
里出现 mp4。OOM、CUDA 初始化失败、缺提示文件，都算诚实失败。失败时复制
最后 30 行日志，对照第 10 节。不要为了跑通去改 `num_frames` 再声称复现。

GUI 加餐（可选，需要显示器）：

```bash
python demo.py
```

本课验收不依赖 GUI。

### Step 7: 备选体验，Wan 1.3B 双向教师

Self-Forcing 跑不通时，退到教师本人。这只能证明 24GB 推得动双向流匹配
DiT，不能证明流式交互。先克隆教师仓库。

```bash
git clone https://github.com/Wan-Video/Wan2.1.git
```

在 `Wan2.1` 根目录装依赖（README 要求 `torch >= 2.4.0`）：

```bash
pip install -r requirements.txt
```

权重若 Step 5 已经下过 1.3B，把它拷到 `./Wan2.1-T2V-1.3B`，或按官方
Hugging Face 卡再下一次。然后跑 README 的 1.3B 单卡示例（官方为 4090
写了 `--offload_model True --t5_cpu`）：

```bash
python generate.py --task t2v-1.3B --size 832*480 --ckpt_dir ./Wan2.1-T2V-1.3B --offload_model True --t5_cpu --sample_shift 8 --sample_guide_scale 6 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

预期：数分钟到十几分钟后得到一段 480P 视频。README 写未优化时 RTX 4090
约 4 分钟、峰值约 8.19GB。你的卡若仍 OOM，停。这段视频没有动作端口，
不要拿去填「适合交互吗」那一列的「是」。

### Step 8: 填配方表并归档

在笔记目录写 `NOTES.md`。表格列必须与第 9 节验收表一致。模板：

```text
日期与机器（有无 CUDA、显存、是否 low_memory）
Step 0 README 摘录：最低显存、推理入口文件名
Step 2 漂移图路径：第 1 步误差、第 40 步 TF / free
Step 3 玩具开关：tf 与 sf 各自 train_last、第 32 步 free
Step 6 或 Step 7：成功则视频路径；失败则最后几行报错
配方表见下
本实验只说明训练协议如何改变一步/多步行为；不说明任何 VBench 分数
```

## 8. 配置与预算

| 档位 | 做什么 | 硬件 | 耗时（参考） | 三档 |
|---|---|---|---|---|
| 复现方向 | 重跑第 03 课 drift 探针 + 玩具开关 | Mac / CPU 即可 | 数分钟 | 复现（机制方向，不是论文分数） |
| 体验 A | Self-Forcing 官方推理 | README：至少 24GB NVIDIA、Linux、64GB 内存 | 装环境加下载数小时，推理以机器为准 | 体验 |
| 体验 B | Wan2.1 T2V-1.3B `generate.py` | README：约 8.19GB；4090 上约 4 分钟 / 5 秒 480P | 下载加一次生成 | 体验 |
| 只讲 | Self-Forcing / CausVid / Causal-rCM 训练 | README：64×H100、600 步、不到 2 小时（Self-Forcing DMD） | 本课不跑 | 只讲 |

Self-Forcing 推理相关、以仓库为准的字段：

| 项 | 仓库里的值 | 说明 |
|---|---|---|
| 学生骨架 | Wan 1.3B 因果包装 | `WanDiffusionWrapper(..., is_causal=True)` |
| 教师分数 | yaml 写 `Wan2.1-T2V-14B` | 仅蒸馏训练用，本课推理不加载 |
| 潜空间形状 | `[1, 21, 16, 60, 104]` | 21 潜帧，约对应 81 像素帧 |
| 块大小 | `num_frame_per_block: 3` | 训练和推理协议的一部分 |
| 少步时间戳 | `1000, 750, 500, 250` | `warp_denoising_step: true` |
| 连续目标 | `denoising_loss_type: flow` | 流匹配，不是另一套模型 |
| 蒸馏目标 | `distribution_loss: dmd` | 分布匹配，不是逐步 L2 |
| 学习率 | `lr: 2.0e-06`，评论家 `4.0e-07` | 只读，24GB 不要拿去扫 |
| EMA | `ema_weight: 0.99`，`ema_start_step: 200` | 推理加 `--use_ema` 才加载 `generator_ema` |
| low_memory | 免费显存小于 40GB | 24GB 主线会走动态换入 |

Wan 1.3B 推理：`--sample_guide_scale 6`、`--sample_shift` 官方建议 8 到 12、
分辨率 `832*480`。720P 官方说能出但不稳，本课不要用来验收。

预算纪律：不要为了「体验」去下 14B 教师或 MixKit ODE 配对。那是蒸馏训练的
材料，属于只讲。桌宠侧的预算仍是第 30 课那张小模型：本课只多一个
$p\in[0,1]$ 的喂自己预测的开关。

## 9. 验收

验收清单：

- [ ] 能指着 `trainmdrnn.py::get_loss` 说出 teacher forcing 发生在哪，指着
      `probe_mdrnn.py::drift_probe` 说出自由 rollout 发生在哪；
- [ ] 漂移图第 1 步两条曲线重合，自由 rollout 在 32 步之后明显高于 teacher
      forcing；
- [ ] 玩具开关跑过 `--mode tf` 和 `--mode sf`，笔记里写明这些数字不是 Wan
      损失；
- [ ] 一张配方表，列与下行模板完全一致，每一行能口头解释「适合交互吗」；
- [ ] 能用一句话说清：Cosmos 3 的 Generator 默认双向去噪，所以还要因果蒸馏
      加 self-forcing 才能当交互世界模型；
- [ ] Self-Forcing 或 Wan 1.3B 要么有一段官方脚本产出的视频，要么有带原因的
      失败记录（无 CUDA / 低于 24GB / flash-attn 编译失败 / OOM）；
- [ ] 没有把任何 VBench 数字或论文摘要里的 FPS 写成自己的测量。

配方表模板（五列，勿增删）：

| 配方 | 目标 | 是否因果 | 训练是否吃自己的输出 | 适合交互吗 |
|---|---|---|---|---|
| MDN-RNN teacher forcing（第 03 课） | 下一步 $z$ 的 NLL | 是（LSTM 逐步） | 否 | 能滚，但不稳 |
| 双向流匹配 / 扩散（Wan2.1 教师、DIAMOND 整段） | 速度场或 score | 否 | 否（整段真实或整段噪声） | 否，要等整段 |
| Diffusion Forcing | 独立噪声水平下的去噪 | 可因果 | 不一定 | 可以逐步吐，仍看实现 |
| CausVid 蒸馏 | DMD + ODE 初始化 | 学生是 | 主要吃教师轨迹 | 能流式，仍可能漂 |
| Self Forcing | 少步 flow + DMD | 是 | 是，KV cache rollout | 为交互设计 |
| Rolling Forcing | 窗口联合去噪 + sink | 窗口内放松 | 是，自己的历史 | 为长时流式设计 |
| Causal Forcing | 因果教师 ODE + DMD | 是 | DMD 段是 | 为交互设计 |
| Causal-rCM | TF 一致性 + SF DMD | 是 | 精炼段是 | 摘要接到 Cosmos 3 |
| MAGI-1（对照） | 块内去噪、块间自回归 | 块间是 | 按论文设定 | 支持流式 |
| 桌宠最小补丁 | 逐步预测 + 以 $p$ 混入 $\hat{s}$ | 是 | $p>0$ 时是 | 24GB 默认用这个 |

「适合交互吗」这一列填的是机制资格，不是你的 FPS。双向教师那一行必须填「否」。

## 10. 排错

| 症状 | 最可能的原因 | 怎么确认 | 怎么修 |
|---|---|---|---|
| `probe_mdrnn.py` 找不到 | 没把第 03 课胶水放到仓库根目录 | `ls probe_mdrnn.py` | 从第 03 课正文复制，不要改 `drift_probe` 的喂法 |
| 漂移图两条曲线重合 | 自由分支误喂了真实 $z$ | 第 1 步之后仍处处相等 | 对照第 03 课原版，自由分支必须把 `z` 写成模型输出 |
| 玩具脚本 tf / sf 曲线几乎一样 | 线性模型对这条立方动力学曝光偏差弱 | 看 `train_last` 是否都已经到 0.01 以下 | 记失败；机制证据改用 Step 2 真图 |
| `torch.device("cuda")` 报错 | 没有 CUDA | Step 4 的 `is_available()` | 体验档停，写无 CUDA。不要改 MPS |
| `flash-attn` 编译失败 | 缺 NVIDIA 工具链或 CUDA 版本不匹配 | 报错含 nvcc / sm_ | 改走 Step 7；不要降级乱装 |
| Hugging Face 下载中断 | 网络或磁盘 | 目录残缺 | 重跑同一条 `huggingface-cli download`，它支持续传 |
| Self-Forcing OOM | 24GB 卡上 text encoder + DiT + VAE 同时驻留 | 日志里的 `Free VRAM` | 确认走了 low_memory；仍 OOM 就停，改 Step 7 |
| 提示文件不存在 | README 示例文件名与仓库 `prompts/` 不一致 | `ls prompts` | 换成实际存在的 txt，或自写一行长提示 |
| 推理出来画面崩、颜色过曝 | 没用 EMA，或提示太短 | 是否加了 `--use_ema` | 加 `--use_ema`；提示写具体场景，不要单词语 |
| Wan 1.3B 很慢 | 官方就是多步双向采样 | 是否误用了 14B 或 720P | 确认 `--task t2v-1.3B --size 832*480` |
| 把 Wan 视频当成交互成功 | 没有动作端口 | `generate.py` 只有 `--prompt` | 配方表「适合交互吗」填否 |

## 11. 前沿与改造

前沿怎么做。2024 到 2026 年，公开系统把「交互视频」拆成同一条流水线：开源
双向教师（Wan2.1）→ 改因果注意力 → ODE 或一致性初始化 → DMD / 一致性蒸馏
→ 训练时 self-rollout → 长时再打补丁（Rolling、Self-Forcing++）。Cosmos 3
把 Generator 做成全模态双向去噪，再单独后训练才能当世界模拟器。动作端口
往往比因果开关更晚才加上，那是第 40 课的工业流水线。

我们差在哪。规模差是钱：64 张 H100 的蒸馏、14B 教师、分钟级长视频，24GB
买不起。机制差本课已经补上：训练分布要不要对齐推理分布、教师和学生对不对
齐因果结构、损失看一步还是看整段分布。桌宠差的是动作条件和安全过滤，不是
480P 画质。

动手改造清单（前两个建议做，后两个选做）：

1. 第 03 课 Scheduled Sampling。在 `trainmdrnn.py::get_loss` 里，以概率
   $p$ 把 `latent_obs[:, t]` 换成上一步模型给出的混合均值。$p$ 取 0、0.3、
   0.7 各训一版，logdir 分开。预算：每版数小时，CPU 可做。预期：测试 NLL
   随 $p$ 变差，`drift_probe` 的自由滚动斜率变缓。失败：三条漂移曲线分不开，
   说明这份 CarRacing 数据上惯性太大，把该结论写进笔记。
2. [第 30 课](30_desk_world_model.md) 桌面模型加同一个 $p$。预算：重训数小时。预期：2 秒想象里杯子
   边缘漂得更慢；动作对换的分岔不应当消失。失败：分岔塌掉，说明混入自身
   预测把动作信号冲掉了，把 $p$ 降到 0.3 再试。
3. Self-Forcing 推理消融（仅 24GB 体验成功的人）。同一提示，分别加和不加
   `--use_ema`，各存一段视频。预算：两次推理。预期：EMA 更稳。失败：看不
   出差别，记「该提示上不明显」，不要编差异。
4. 因果对照阅读。打开 `pipeline/bidirectional_inference.py` 和
   `pipeline/causal_inference.py`，写五句话：谁在每一步看见未来、谁写 KV
   cache、谁能在第一块生成完后就存盘。预算：读代码一小时。预期：你能向同事
   讲清「同一份 Wan wrapper，协议不同，产品不同」。

顺手复现的映射。论文结论「teacher forcing 和推理脱节」对应 Step 2：缩小版
必须看得到裂口，否则第 03 课权重或探针坏了。论文结论「训练时吃自己的输出
能减缓漂移」对应改造 1，预期同方向，不预期同幅度。论文结论「双向教师不能
直接当交互引擎」对应 Step 7：你能出视频，却给不出动作对换。Causal-rCM 接到
Cosmos 3 这一条，本课只读摘要，不复现。

## 12. 论文与延伸

1. Flow Matching for Generative Modeling（Lipman 等，2023，
   [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)）。带着三个问题读：
   条件速度场为什么可以写成 $x_1-x_0$？它和扩散 score 在什么假设下等价？
   少步积分时误差来自路径弯曲还是网络拟合？
2. Diffusion Forcing（Chen 等，2024，
   [arXiv:2407.01392](https://arxiv.org/abs/2407.01392)）。问：独立 per-token
   噪声怎样让同一个因果网络既做下一步预测，又做整段去噪？它和后来的
   Rolling Forcing 窗口噪声是什么关系？
3. CausVid（Yin 等，CVPR 2025，
   [arXiv:2412.07772](https://arxiv.org/abs/2412.07772)，
   [项目页](https://causvid.github.io/)）。问：不对称蒸馏里，教师看见未来、
   学生看不见，梯度到底在对齐什么？ODE 初始化缺了会怎样？仓库推理脚本写
   3-step，摘要写 4-step，以哪一份为准？
4. Self Forcing（Huang 等，2025，
   [arXiv:2506.08009](https://arxiv.org/abs/2506.08009)，
   [项目页](https://self-forcing.github.io/)，
   [代码](https://github.com/guandeh17/Self-Forcing)）。问：
   `inference_with_trajectory` 为什么必须在训练图里建 KV cache？随机截断
   一步梯度，还算不算「整段 rollout」？
5. Wan2.1（Team Wan，2025，
   [arXiv:2503.20314](https://arxiv.org/abs/2503.20314)，
   [代码](https://github.com/Wan-Video/Wan2.1)）。问：README 为什么同时说
   Flow Matching 和 Diffusion Transformer？1.3B 的 8.19GB 数字对应哪条命令？
   它缺了动作端口之后，[第 12 课](12_frontier_landscape.md) 三问卡在哪一问？
6. Self-Forcing++（Cui 等，
   [arXiv:2510.02283](https://arxiv.org/abs/2510.02283)）；
   Rolling Forcing（Liu 等，
   [arXiv:2509.25161](https://arxiv.org/abs/2509.25161)）；
   Causal Forcing（Zhu 等，
   [arXiv:2602.02214](https://arxiv.org/abs/2602.02214)，
   [代码](https://github.com/thu-ml/Causal-Forcing)）；
   Causal-rCM（Zheng 等，
   [arXiv:2606.25473](https://arxiv.org/abs/2606.25473)）。四篇当修补清单读。
   每篇只追问一个问题：它修的是长时漂、初始化用错教师，还是 TF/SF 两种散度
   怎么叠？不要抄它们的榜。
7. MAGI-1（Sand.ai 等，
   [arXiv:2505.13211](https://arxiv.org/abs/2505.13211)）。对照读：chunk
   因果和 Self Forcing 的 3 帧一块，差在训练目标还是注意力掩码？精读留给
   第 38 课。
8. 回访 [第 03 课](03_mdn_rnn_action_conditioned.md) 5.4 节、
   [第 10 课](10_diamond_diffusion_world_model.md) 5.1 节、
   [第 22 课](22_foundation_video_wm.md) 的 Cosmos-Predict2.5 段落。问：
   本课的配方表，哪一行已经在那三课出现过，哪一行是第九幕新加的？

第 38 课把骨架从协议里抽出来：AR token、双向 DiT、因果 AR-扩散、MoT、JEPA、
latent action、RSSM 各自把状态放在哪、动作从哪进。本课填完的配方表会变成
那一课的输入。桌宠不必换骨架，但要说得出为什么不换，以及第 30 课那个
$p$ 开关要不要拧开。
