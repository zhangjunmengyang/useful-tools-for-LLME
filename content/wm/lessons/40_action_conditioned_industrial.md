---
id: 40_action_conditioned_industrial
title: "工业级动作条件"
summary: "把视频基础模型后训练成会听动作的模拟器，缺哪几步？"
unit: frontier
play_tools: []
checkpoints:
  - "后训练流水线检查表。"
  - "能说出谁在学 P(s'|s,a)，谁在学 π(a|s)。"
---

# 第 40 课：把视频生成器后训练成会听低层动作的世界模拟器

> 类型：实战（DINO-WM 与 1xgpt 的动作通道文件核对、工业流水线检查表）+ 只讲（Cosmos 后训练、Cosmos Policy、Cosmos-Transfer1、UniSim、IRASim）<br>
> 建议周期：1-2 天<br>
> 硬件：Mac / 纯 CPU 可完成全部必做（读文件、核路径、填检查表）。单张 24GB 卡可复用第 22 课的 DINO-WM 环境做对照，本课不重跑 PushT 全流程。Cosmos 后训练官方按多卡 H100 / GB200 写，主线标成只讲<br>
> 锚定仓库：[gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm)（动作怎么编码、损失怎么加、规划器怎么查），[1x-technologies/1xgpt](https://github.com/1x-technologies/1xgpt)（数据里有动作、基线没用动作），[nvidia/cosmos](https://github.com/nvidia/cosmos) 当前 post-train cookbook；对照文档 [nvidia-cosmos/cosmos-predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) 的 `robot/action-cond` 与 `robot/policy`<br>
> 产物：一条后训练流水线、一张动作通道文件名表、一份工业检查表、Cosmos Policy 与 DINO-WM 的公式对照（谁在学 P(s'|s,a)，谁在学 π(a|s)）

## 1. 这一课做什么

整门课的主干循环没变：

```text
观察 先压成状态 再按动作预测下一状态 然后展开多条未来 给未来打分 最后选动作
```

你现在在第九幕。这一幕读 2025-2026 年的刷榜系统、声音、训练配方和架构，不改[第 32 课](32_ship_desk_pet.md)、[第 33 课](33_embodiment_degrees.md) 的毕业标准。[第 39 课](39_long_horizon_memory.md) 处理的是长时记忆：滑窗会切断 $K$ 帧前的杯子，隐状态会把杯子融进噪声，记忆库要按位姿和时刻取回。杯子还在，只说明状态没丢。桌宠下一步要问的是另一句话：如果我现在伸手，两秒后杯子会不会过沿。没有低层动作端口，记忆再长也只是会回忆的生成器。

本课加的零件是工业级动作条件：互联网视频预训练出来的生成器，怎样后训练成会听低层动作的世界模拟器。基础视频模型通常吃文本或首帧，输出好看的未来。工业世界-动作模型要额外吃动作轨迹，输出未来视频或未来状态，供策略查询。这条流水线在桌面上的缩小版就是[第 30 课](30_desk_world_model.md)：冻住眼睛、接上动作、对换必须分岔。工业侧换的是检查点规模和后训练菜单，不是换公式。

有一件事必须从第一段就拆开。后训练成策略，和后训练成世界模型，是两条作业，不是同一步。Cosmos Policy（Kim 等，arXiv:2601.16163）把 Cosmos-Predict2 后训练成机器人策略，部署默认更接近 $\pi(a \mid s)$。同一家平台上的 `robot/action-cond` 才是按动作生成未来视频。把两条菜单合成一句「Cosmos 已经是世界模型了」，后面的规划器和安全层都会接错头。

[第 22 课](22_foundation_video_wm.md) 已经在 DINO-WM 的 PushT 上做过动作对换和一次缩小规划。本课不重复那套安装、解压、滑杆和 `n_evals=5`。本课只抽三件工业上真正要写进流水线的事：动作通道加在哪一层（落到真实文件名）、训练损失加在哪一段、规划器查询的是哪一个张量。1xgpt 用来当反例：仓库和数据卡里都能看到动作文件，公开 GENIE 基线却只在视频序列上训练。

Cosmos 后训练、Cosmos-Transfer1、UniSim、IRASim 标成只讲。能跑的对照是你已经 clone 过的 `dino_wm` 和 `1xgpt`。24GB 主线假设下，不要开始写 8×H100 的 `torchrun`。

术语速查：

| 术语 | 一句人话 |
|---|---|
| 后训练 | 在预训练检查点上用新数据继续训，改条件或输出，不从零堆参数 |
| 动作通道 | 把低层动作写进计算图的那条输入；配置和源码里必须找得到落点 |
| 世界模拟器 | 吃当前状态和打算做的动作，吐下一状态或未来视频，学 $P(s_{t+1}\mid s_t,a_t)$ |
| 策略后训练 | 把生成器改成吐动作（常常顺带吐未来），学的主头是 $\pi(a_t\mid s_t)$ |
| 潜帧注入 | Cosmos Policy 把本体感觉、动作块、价值写成和图像一样的 latent 帧，塞进扩散序列 |
| 帧级动作条件 | IRASim 在每个 transformer 块里让第 $t$ 帧对齐第 $t$ 个动作，而不是整段视频共用一个向量 |
| 前向动力学 | Cosmos 3 Generator 的 `fd` 模式：首帧加动作轨迹进，未来视频出 |
| 逆向动力学 | Cosmos 3 Generator 的 `id` 模式：视频进，动作轨迹出 |
| 动作对换 | 钉死观察只换动作，预测必须分岔；[第 03 课](03_mdn_rnn_action_conditioned.md) 立过，本课当工业闸门 |
| 安全过滤 | 动作出口上的拦截器：模拟器说会出事就截断，[第 27 课](27_sim_to_real.md) 装过桌沿闸 |

## 2. 问题

互联网视频预训练解决的是「世界看起来会怎么自己滚」。电影、手机录像、驾驶记录仪里，镜头在动、物体在动，但逐步的关节命令通常没有和每一帧对齐。模型学会的是文本或首帧条件下的 $P(\text{future video}\mid\text{past},\text{text})$。画面可以极真，[第 12 课](12_frontier_landscape.md) 的第一问却仍可能失败：同一观察换动作，预测不分岔。

工业系统要的是另一件事。机械臂每 10 到 50 毫秒出一个末端位移或关节增量；规划器要问「如果执行这条轨迹，画面或状态会怎样」。条件里必须出现逐步动作 $a_t$，输出必须随 $a_t$ 变。这就是动作条件世界模拟器：

$$
P(s_{t+1} \mid s_t, a_t)
$$

$s$ 可以是像素、DINOv2 patch、token 或本体感觉加多相机图像。名称不重要，接口重要。

后训练看起来像「再训一会儿就行」。菜单上至少有两道菜，不能点成一套餐。

第一道：后训练成世界模型。在预训练视频检查点上加动作通道，用带动作标签的机器人轨迹继续训，让模型学 $P(\text{future}\mid s,a)$。Predict2.5 的文档把这条写成 `docs/post-training_video2world_action.md`，检查点名字叫 `robot/action-cond`。Cosmos 3 Generator 把同一接口叫前向动力学 `fd`。

第二道：后训练成策略。还是那个视频检查点，改成吐动作块，学 $\pi(a\mid s)$ 或 $p(a,s',V(s')\mid s)$。Cosmos Policy 论文和 cookbook、Cosmos 3 的 `Policy-DROID` SFT，走的是这条。它也可以同时预测未来图像，那是附加能力，不能把「会出未来图」自动翻译成「已经通过动作对换」。

混成一步的典型后果有三个。用任务成功率证明动力学（VLA 和 Cosmos Policy 的 LIBERO 数字量的是规划好，[第 17 课](17_evaluating_world_models.md) 的预测准是另一根尺子）。用画质分数放行规划器（生成真过关，动作通道可以仍是摆设）。缺安全层就上真机（模拟器还没听动作，过滤器查询的是一个惯性外推器）。

本课要你当场核三处落点，再交出一张工业检查表：

1. 动作怎么编码，加在哪一层，文件名是什么。
2. 训练损失加在哪一段，哪些维被故意排除。
3. 规划器查询的是哪个前向接口，查询之前有没有动作对换。

UniSim（Yang 等，arXiv:2310.06114）证明过「互联网生成器可以变成交互模拟器」这条志向，无本课可跑的官方权重，只讲。IRASim（Zhu 等，arXiv:2406.14540）把轨迹到视频写成帧级对齐，用来对照「整段视频共用一个动作向量」为什么不够。Cosmos-Transfer1（arXiv:2503.14492）是空间可控的世界生成（分割、深度、边缘），控制信号不是低层关节动作，不要填进动作通道那一格。

## 3. 准备

- [第 22 课](22_foundation_video_wm.md) 的 DINO-WM 笔记和（若当时装过）conda 环境。本课不要求再解压 `pusht_noise`，也不要求再跑 `plan.py`。没做过第 22 课也可以：clone 仓库，按第 7 节 grep 文件即可。
- [第 11 课](11_genie_latent_actions.md) 对 1xgpt 的印象：数据有 token 和原始动作，公开 GENIE 基线「only trains on video sequences, not actions」。本课把这句话对到 `data.py` 里被注释掉的 `actions.bin`。
- [第 24 课](24_visual_foresight.md) 的 visual MPC：预测器、代价、CEM 是三件，换代价不换 $P$。本课规划器查询段会用到。
- [第 26 课](26_vla_vs_world_model.md) 的接口表：VLA 吐 $a$，世界模型吐 $s'$。Cosmos Policy 默认站在 VLA 那一侧。
- [第 33 课](33_embodiment_degrees.md) 的是否题。本课结束时要能口头打分：Cosmos Policy 默认近 E3，DINO-WM 规划探针近 E2。
- [第 35 课](35_cosmos3_omnimodal.md) 若已读：Cosmos 3 的 Reasoner / Generator 和输入输出配置表。本课只取动作这一列，把前向动力学和策略后训练写成两道作业。
- [第 39 课](39_long_horizon_memory.md) 的三种忘法对照。本课不重测滑窗，只问：记住杯子之后，动作通道有没有接到预测器上。
- 磁盘：只 clone 三个仓库的话，几个 GB 够。不要为本课去下 Cosmos3-DROID 全集，也不要去拉 Predict2.5 的 2B/14B 权重。
- 网络：写课和你做实验都要用 `curl` 对当前 GitHub raw 路径，不要抄过期截图。`nvidia/cosmos` 的 cookbook 目录以仓库 `main` 为准。
- 硬件：必做步骤全部是读文件和填表。有 GPU 的人若第 22 课留下了对换日志，第 7 节允许引用，不允许把本课验收写成「再刷一次 PushT 成功率」。
- Python 3.10 左右即可。本课必做不新装 Cosmos Framework，不跑 `uv sync --group=cu130-train`。

## 4. 学习目标

1. 白纸画出后训练流水线：预训练视频模型、加动作通道、动作对换验收、再接到 MPC 或安全过滤；并标出「策略后训练」从哪一步分岔出去，不能画成同一条箭头。
2. 对照 DINO-WM 的真实配置，指出动作编码器类名、拼接维、损失排除了哪一段、规划器调用的是哪个方法。
3. 对照 1xgpt 的 `data.py` 与 README，说出数据里动作文件叫什么、基线有没有把它读进模型。
4. 读当前 `nvidia/cosmos` cookbook 和 Predict2.5 模型表，把 `fd` / `action-cond` 与 `policy` / `Policy-DROID` 分成两行，各写输入、输出、公式。
5. 用第 33 课是否题给 Cosmos Policy（默认直接出动作）和 DINO-WM 规划探针各打一档，并写清证据来自论文、仓库还是你自己的实验。
6. 交出工业检查表：缺动作对换不能进规划；缺安全层不能上真机；只有画质分数就停在生成器。

## 5. 原理

六个机制。每个仍走直觉、运转、数学、代码落点、验证。工业系统名字更响，尺子还是这三根：[第 12 课](12_frontier_landscape.md) 的三问，[第 17 课](17_evaluating_world_models.md) 的预测准 / 生成真 / 规划好。

### 5.1 后训练有两个终点，不能合成一步

把一台已经会画未来的视频模型拿来做机器人，实验室里最省事的说法是「再训一下」。再训一下，终点至少有两个。

终点 A 是世界模拟器。你希望模型继续吐未来，但条件里多了逐步动作。训练数据是 $(s_t, a_t, s_{t+1})$，损失落在未来状态或未来视频上。测的时候钉死 $s$，换 $a$，看 $s'$ 分不分岔。过关之后，规划器或安全层才能查询它。

终点 B 是策略。你希望模型吐出下一步该做的动作，条件是当前观察，常常再加一句语言指令。训练数据是示范里的 $(s_t, a_t)$，损失落在动作上。测的时候看任务成功率。这和 [第 25 课](25_lerobot_imitation.md) 的 ACT、[第 26 课](26_vla_vs_world_model.md) 的 OpenVLA 是同一类接口，只是骨干从 VLM 换成了视频生成器。

两个终点可以共用同一个预训练检查点，甚至可以焊在同一个网络里。Cosmos Policy 论文第 4.2 节就把一个 batch 拆成三份：大约 50% 学 $p(a,s',V(s')\mid s)$，25% 学 $p(s',V(s')\mid s,a)$，25% 学 $p(V(s')\mid s,a,s')$。这仍然是三个条件方案，不是「训一次就同时变成了策略和世界模型」。官方 cookbook 写得很直白：这篇菜谱默认把模型当直接策略部署；带未来状态和价值的 best-of-N 规划另算，更慢，论文里才展开。

类比：同一台飞机模拟器，可以改成教人开（策略），也可以改成预报气流（动力学）。座舱看起来一样，你考的执照不是同一张。类比失效处：Cosmos Policy 真的会在同一串 latent 里同时出动作、未来图和价值，比飞机模拟器更缠在一起。所以验收必须看条件掩码和返回值，不能看网络名字。

记两式，本课后面不再混用：

$$
\pi_\theta(a_t \mid s_t),\qquad P_\phi(s_{t+1} \mid s_t, a_t)
$$

验证方法本课就能做：打开 Predict2.5 当前 README 的模型表。`Cosmos-Predict2.5-2B/robot/action-cond` 的输入列写的是 action，这是终点 A。`Cosmos-Predict2.5-2B/robot/policy` 写的是 action + image、后训练于 LIBERO 和 RoboCasa，这是终点 B。同一张表、两行、两道菜。

### 5.2 预训练生成器缺的是逐步动作，不是分辨率

互联网视频模型已经很会画。缺的是和每一帧对齐的低层动作。文本「把杯子推向左边」是一条语义指令，对应无数条关节轨迹。首帧条件告诉模型桌子长什么样，不告诉它你这 200 毫秒末端往哪移了 1 厘米。

所以后训练的第一刀不是换骨干，是给计算图开一条动作通道。通道要满足四件事：

1. 每个时间步有一个动作向量，维数对得上身体（PushT 是 2 维位移，Bridge 常见 7 维末端加夹爪，DROID 在 Cosmos 3 cookbook 里写成 9D 位姿加 1D 夹爪）。
2. 编码之后进入预测器的每一层或每一帧，不能只在第 0 层看一眼再丢掉。
3. 训练损失必须让「换动作」改变输出。只重建像素、不看动作维，模型会学成惯性外推器。
4. 推理时规划器能改这条通道再前向。改不了，就只是回放器。

工业上常见四种接法，本课只采用已经核到公开材料的。

特征拼接。DINO-WM 把动作经小 MLP 编成 $d_a$ 维，按 `concat_dim: 1` 拼到每个视觉 patch 的特征维上。每个格子都带着「这一步准备怎么动」。

加法嵌入。1xgpt 的 README 写，当前 GENIE 实现只在视频序列上训练，加一个 additive embedding 就能接动作。仓库里这条嵌入没有接到 `STTransformerDecoder`。这是「文件名有动作、计算图没有动作」的标准反例。

潜帧注入。Cosmos Policy 把归一化后的本体感觉、动作块、价值复制填充成和图像 latent 一样的 $H'\times W'\times C'$ 体，插进扩散序列。顺序按论文图 2 是 $(s, a, s', V(s'))$。不改 transformer 结构，改的是序列里多了几帧「假图像」。

帧级自适应。IRASim 认为「整段视频共用一个动作向量」对不上动作块：第 $t$ 帧应该听第 $t$ 个动作。它在每个 DiT 块里做 Frame-Ada，显式对齐动作和帧。Predict2.5 的 action-cond 后训练用的 Bridge 数据，文档写明来自 IRASim 的划分，动作是末端位移加夹爪。

Cosmos-Transfer1 不在这个名单里当动作通道。它吃的是分割、深度、边缘一类空间控制图，用来做世界到世界的迁移和仿真变真。控制信号改变的是画面布局，不是 $a_t$。下一课驾驶世界模型会再碰到「条件里哪些算动作、哪些算外生、哪些算风格」。本课先把它从表上拿开，免得「可控」三个字把空间控制和关节控制混成一格。

### 5.3 DINO-WM：动作通道、损失、规划器查询

DINO-WM 是本课能指到行的缩小样板。[第 22 课](22_foundation_video_wm.md) 讲过冻住 DINOv2、只在 patch 上预测。这里只留工业流水线要抄的三处。

动作怎么编码。`conf/train.yaml` 写着

```text
action_encoder: proprio
action_emb_dim: 10
num_action_repeat: 1
concat_dim: 1
```

Hydra 会去读 `conf/action_encoder/proprio.yaml`，目标类是 `models.proprio.ProprioceptiveEmbedding`。这个类用 `nn.Conv1d` 把形状为 $(B,T,D_a)$ 的动作编成 $(B,T,d_a)$，默认 $d_a=10$。它和本体感觉编码器共用同一份 `models/proprio.py`，只是配置里的 `in_chans` 分别接动作维和本体感觉维。

拼到哪一层。`models/visual_world_model.py::VWorldModel.encode` 在 `concat_dim == 1` 时，把动作嵌入在时间维上对齐后，沿最后一维拼到每个 patch：

$$
z_t^{(n)} = \bigl[\,\mathrm{enc}(o_t)^{(n)};\; \phi_{\mathrm{prop}}(p_t);\; \phi_a(a_t)\,\bigr]
$$

$n$ 是 patch 下标。`concat_dim == 0` 是另一种接法：动作变成额外一个 token，不拼进每个格子。默认训练配置走的是维拼接。规划时换动作，靠的是 `replace_actions_from_z`：只改 $z$ 里属于动作的那几维，视觉历史不动。

损失怎么加。`VWorldModel.forward` 先 `encode` 整段，取前 `num_hist` 帧当源、后移一帧当目标，预测器输出 $\hat z$，再算 MSE。关键一行是：比较时丢掉动作维。

$$
\mathcal{L}_z = \bigl\| \hat z_{\neg a} - \mathrm{sg}\bigl(z^{\mathrm{tgt}}_{\neg a}\bigr) \bigr\|^2
$$

`concat_dim == 1` 时，源码把 `z_pred[..., :-self.action_dim]` 去对 `z_tgt` 的同一切片。动作通道是条件，不是预测目标。解码器若开着，对 `z_pred.detach()` 再重建像素，梯度到不了预测器。论文附录写过：把重建损失反传到预测器会伤规划。工业翻译：生成真的损失不要回流到动力学头。

规划器怎么查。`planning/cem.py::CEMPlanner.plan` 从当前均值方差采样 `num_samples` 条动作（`conf/plan_pusht.yaml` 里是 300），每条调用 `wm.rollout(obs_0=..., act=...)`，用 `planning/objectives.py::create_objective_fn` 比较想象终点和目标图特征。默认 `mode: last`、`alpha: 1`：

```text
loss = loss_visual + alpha * loss_proprio
```

CEM 外面包着 `planning/mpc.py::MPCPlanner`，一次执行 `n_taken_actions: 5` 步再重规划。查询接口是 `rollout`，不是解码器视频，也不是策略头。`rollout` 内部每步 `predict` 完都 `replace_actions_from_z`，所以你换动作序列，梦必须改道。

验证分两层。文件层：第 7 节 grep，必须能指到 `encode_act`、`replace_actions_from_z`、`z_loss`、`wm.rollout`。实验层：第 22 课已经做过的对换。本课不重做，但流水线把对换画在规划前面。对换失败，CEM 只是在一个动作盲模型上抽随机数。

### 5.4 1xgpt：动作在磁盘上，不在计算图里

1xgpt 是工业流水线的负对照。Hugging Face 数据卡 `1x-technologies/worldmodel` 提供 100 小时以上的 EVE 第一人称 token，并写了 `actions/` 目录。仓库 README 的 Evaluation Challenge 把终极目标写成：给定一群策略 $\pi_i(a_t\mid z_t)$，用世界模型 $p(z_{t+1}\mid z_t,a_t)$ 在脑子里排序。志向是终点 A。

当前公开实现停在生成器。README 原句：这份 GENIE 实现 only trains on video sequences, not actions，加一个 additive embedding 就能接上。`data.py::RawTokenDataset` 把三个路径拼成 `video.bin`、`segment_ids.bin`、`actions.bin`，紧接着把

```text
self.actions = np.memmap(action_tokens_path, ...)
```

整行注释掉。`__getitem__` 只返回 `input_ids`、`labels`、`attention_mask`，全部来自视频 token。`genie/st_transformer.py::STTransformerDecoder` 的输入是 `tgt`，没有动作张量。`genie/configs/magvit_n32_h8_d256.json` 里是层数、头数、词表，没有 `action_dim`。

所以 1xgpt 基线的公式是 $P(z_{t+1}\mid z_{\le t})$。数据里有原始动作，计算图里没有动作通道。压缩挑战和采样挑战可以刷得很好看，第一问仍没资格过关。想把它推进流水线，最少要做三件事：取消注释并真正读动作、在 ST 块上加嵌入、用对换证明换动作后天生视频分岔。那是改造清单，不是本课必做训练。

验证：第 7 节打开 `data.py`，确认注释还在。不要凭记忆写「1xgpt 已经是动作条件世界模型」。

### 5.5 Cosmos：同一平台上的两条后训练菜单

NVIDIA 把世界基础模型拆成预测、迁移、推理三家。[第 35 课](35_cosmos3_omnimodal.md) 已经把 Cosmos 3 的 MoT 和输入输出组合拆开。本课只关心后训练菜单里和动作有关的两行，以及 Transfer 为什么不算动作通道。

世界模型后训练。`nvidia-cosmos/cosmos-predict2.5` 的 `docs/post-training_video2world_action.md` 标题就是 Video2World Post-training for Action-conditioned Video Prediction。数据用 IRASim 划分过的 Bridge：`annotations/*.json` 里有 `state`（末端位姿）、`continuous_gripper_state`、`action`（夹爪坐标系下的 6 维位移加开合）。训练入口指向

```text
cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py
experiment=ac_reason_embeddings_rectified_flow_2b_256_320
```

新 conditioner 名叫 `action_conditioned_video_conditioner`，网络覆盖项是 `cosmos_v1_2B_action_conditioned`。推理脚本是 `examples/action_conditioned.py`，文档写明暂不支持多卡，`context_parallel_size` 必须为 1。这是终点 A：首帧或短历史加动作块，出未来视频。

策略后训练。Cosmos Policy 论文把 Cosmos-Predict2-2B 单阶段后训练成策略，不改结构，靠潜帧注入。官方 cookbook 把 LIBERO / RoboCasa / ALOHA 的训练写成多节点 `torchrun`：LIBERO 参考是 64 张 H100、48 小时，RoboCasa 32 张，ALOHA 约 185 条示范用 8 张。推理显存官方写 LIBERO 约 6.8GB、RoboCasa 约 8.9GB、ALOHA 约 6.0GB。数字是论文和 cookbook 的报告，本课未跑。部署默认是直接执行动作块；用未来状态和价值做 best-of-N，是可选的模型内规划，不是「已经验收了独立的 $P(s'|s,a)$」。

Cosmos 3 把分叉写进了同一份 cookbook。当前 `nvidia/cosmos` 的 `cookbooks/cosmos3/generator/action/README.md` 开篇列出三种任务：

| 任务 | 仓库缩写 | 输入 | 输出 | 公式 |
|---|---|---|---|---|
| 前向动力学 | `fd` | 首帧 + 动作轨迹 | 未来视频 | $P(\text{video}\mid s,a)$ |
| 逆向动力学 | `id` | 视频 | 动作轨迹 | $P(a\mid\text{video})$ |
| 策略 | `policy` | 首帧 + 指令 + 状态 | 未来视频和动作 | 近 $\pi(a\mid s)$，可附带 $s'$ |

动作定义按身体分：自动驾驶自车位姿 9 维，DROID 末端 9 维加夹爪 1 维，人手可达 57 维。后训练成 DROID 策略的脚本在

```text
cookbooks/cosmos3/generator/action/finetune/launch_sft_action_policy_droid_nano.sh
```

对应 TOML `toml/sft_config/action_policy_droid_nano.toml`，实验名 `action_policy_droid_nano`，参考形状是 HSDP 32×8、256 卡、全局 batch 8192。脚本开头的注释写的是 DROID action-policy SFT。这是终点 B。公开检查点名字叫 `Cosmos3-Nano-Policy-DROID` 和 `Cosmos3-Edge-Policy-DROID`，词是 Policy。

Transfer1 是第三条产品线：按空间控制图改视频，仓库 `nvidia-cosmos/cosmos-transfer1`，论文 arXiv:2503.14492。它能做仿真变真、驾驶数据增广。本课检查表里它停在「可控生成器」，进不了「会听低层动作的模拟器」。

硬件诚实性。Predict2.5 的 2B Video2World，Hugging Face 卡片写过约 32.54GB 显存，第 22 课已经据此把 Cosmos 标成只讲。动作条件推理官方暂不支持多卡。Cosmos 3 Nano 推荐工作站级卡，完整 SFT 按 8×H100 写。本课对 Cosmos 只读当前 cookbook 和模型表，不把任何 `torchrun` 写成你该交的作业。

### 5.6 对换闸门、规划器查询、安全层、桌宠缩小版

工业流水线按时间顺序只有四拍。少一拍就停在当前档，不许跳。

第一拍，预训练视频模型。输入是文本、首帧或短视频。输出是未来视频。验收是生成真：FID、人评、WorldScore 一类。到这里你有一台生成器。

第二拍，加动作通道并在带动作标签的轨迹上后训练。验收不是画质再涨两个点，是动作对换：同一 $s$，换 $a$ 与 $a'$，预测距离必须明显大于同一动作重复前向。对换协议和第 03、22、30 课相同，只是 $s$ 的空间从 32 维 $z$ 换成 patch、token 或 latent 帧。不过关，模型仍是惯性外推器。

第三拍，接到规划器。规划器查询的必须是 $P(s'|s,a)$ 或它的特征版，不是 $\pi(a|s)$。Visual Foresight 在像素或指定像素上打分，DINO-WM 在目标特征距离上打分，IRASim 论文把模型当视觉动力学去做模型基规划，Push-T 上把一个扩散策略的 IoU 从 0.637 报到 0.961（论文数字，本课未复现）。查询接口要暴露「换动作再 rollout」。只暴露 `predict_action` 的系统，规划器无对象可查。

第四拍，安全过滤之后才能上真机。过滤器问的是「这条候选动作的想象轨迹会不会越界、碰杯、扫落」，问的对象还是第二拍验收过的模拟器。[第 27 课](27_sim_to_real.md) 的桌沿 5 cm 闸、[第 26 课](26_vla_vs_world_model.md) 的 VLA 出口过滤器，用的都是这一拍。缺这一拍，真机上的 Cosmos Policy 或 OpenVLA 再成功，也只是开环策略。

[第 33 课](33_embodiment_degrees.md) 的档在这里对得上号。DINO-WM 的规划探针：动作对换能分岔（Q1 是），选动作时查询 `rollout`（Q2 是），环境是 gym、可重置（Q3 否、Q4 是），近 E2。Cosmos Policy 默认直接执行动作块：有身体或真机时 Q3 是，选动作不查询独立前向模型则 Q2 否，近 E3。若部署改成 best-of-N，用模型自己出的 $s'$ 和 $V$ 挑动作，Q2 可以改成是，档才离开 E3。论文里那套规划是可选模式，不是默认产品形态。

[第 30 课](30_desk_world_model.md) 是这条流水线的桌面缩小版。预训练眼睛是冻住的 DINOv2，不是 20 亿段视频；动作通道是四个离散键，不是 7 维末端；对换是看左对伸手；规划视野由你自己的漂移曲线决定；安全层留给第 32 课的克制。零件一一对应，只是卡时从 64 张 H100 降到一张 24GB 卡加一个下午。

UniSim 只讲。它把互联网图像、机器人密集动作、导航运动拼成一个交互模拟器，用来在模拟里训高层语言策略和低层 RL，再零样本搬到真机。志向是完整流水线。本课没有权重可核，不把它写成能练，也不用它的宣传图代替你自己的对换记录。

## 6. 源码导读

先读能跑的两个小仓库，再读 Cosmos 当前文档。每个文件带着问题进去。路径全部以写课时 `curl` 到的 `main` 为准；若你 clone 后对不上，以你本地 `git log -1` 的树为准，并在笔记里记下差异。

### 6.1 DINO-WM：动作通道标到文件名

克隆 [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm) 之后按这个顺序读：

| 文件 | 是哪个零件 | 带着什么问题读 |
|---|---|---|
| `conf/train.yaml` | 训练默认 | `action_encoder`、`action_emb_dim`、`concat_dim`、`num_hist` 各控制什么 |
| `conf/action_encoder/proprio.yaml` | 动作编码器配置 | `_target_` 指向哪个类 |
| `models/proprio.py` | `ProprioceptiveEmbedding` | `Conv1d` 的 `in_chans` 是动作维还是写死的 8 |
| `models/visual_world_model.py` | 世界模型本体 | `encode_act`、`encode`、`replace_actions_from_z`、`rollout`、`forward` 里的 `z_loss` |
| `conf/predictor/vit.yaml` | 转移模型 | 默认 6 层、16 头；它吃的是已经拼好动作的 $z$ |
| `planning/cem.py` | CEM | `wm.rollout` 在哪一行被调用 |
| `planning/objectives.py` | 规划代价 | `mode: last` 比较的是哪一帧 |
| `conf/plan_pusht.yaml` | PushT 规划 | `num_samples`、`opt_steps`、`alpha`、`n_taken_actions` |
| `conf/env/pusht.yaml` | 数据 | `rel_actions.pth` 从哪来（`data_path` 拼 `DATASET_DIR`） |

三处最要紧。

第一，动作通道的落点是「每个 patch 的最后几维」，不是单独一个动作 transformer。`encode` 在 `concat_dim == 1` 时把 `act_emb` tile 到所有格子再 `torch.cat(..., dim=3)`。改动作却不走 `replace_actions_from_z`，你改的是磁盘上的 `rel_actions.pth`，模型里的 $z$ 还是旧的。

第二，损失故意不算动作维。`forward` 里 `z_loss` 用 `z_pred[:, :, :, :-self.action_dim]`。有人会觉得「预测里也该把动作复原，当一致性约束」。官方实现没有这样做。动作是给定的条件，复原它等于让网络抄输入。

第三，规划器从来不问解码器。`CEMPlanner.plan` 全程 `torch.no_grad()`，代价在特征上。你看到的视频是可视化，不是搜索用的量。工业上很容易把「生成的想象视频很好看」写成「规划器已经在用世界模型」。DINO-WM 把这两件事拆在两个文件里：`models/visual_world_model.py` 的 `decode`，和 `planning/objectives.py` 的 MSE。

`conf/train.yaml` 的 Hydra launcher 仍写着 `gres: "gpu:h100:1"`。那是他们集群扫参用的。单机 `python train.py ...` 不会真去要 H100。本课连这条训练命令都不作为必做。

### 6.2 1xgpt：动作文件名在，读取被关掉

克隆 [1x-technologies/1xgpt](https://github.com/1x-technologies/1xgpt) 之后读这几个文件：

| 文件 | 是哪个零件 | 带着什么问题读 |
|---|---|---|
| `README.md` 的 GENIE 节 | 官方声明 | 原句是否仍是 only trains on video sequences, not actions |
| `data.py` | 数据集 | `actions.bin` 路径有没有拼出来，`memmap` 有没有被注释 |
| `genie/st_transformer.py` | 动力学 | `STBlock.forward` 的参数列表有没有动作 |
| `genie/configs/magvit_n32_h8_d256.json` | 基线配置 | 有没有 `action_dim` 一类键 |
| `train.py` | 训练入口 | `--genie_config` 之后有没有动作相关开关 |

`RawTokenDataset.__init__` 里这三行要并排看：

```text
video_tokens_path, segment_ids_path, action_tokens_path
    = [data_dir / f"{name}.bin" for name in ["video", "segment_ids", "actions"]]
# self.actions = np.memmap(action_tokens_path, dtype=np.uint16, mode="r", shape=(self.metadata["num_images"],))
```

文件名 `actions.bin` 被拼出来了，对象没被创建。`__getitem__` 因此不可能返回动作。Evaluation Challenge 写在 README 里的 $p(z_{t+1}\mid z_t,a_t)$ 是未来目标，不是当前基线。

Hugging Face 数据卡另写了 `actions/` 目录，里面是多个 `.bin`。磁盘布局和 `data.py` 假设的单文件 `actions.bin` 不必相同。本课验收只要求你看清：当前训练循环没有把任何动作张量送进 `STTransformerDecoder`。

### 6.3 nvidia/cosmos：当前 cookbook，不要抄过期路径

克隆 [nvidia/cosmos](https://github.com/nvidia/cosmos) 之后，本课只读、不训。写课时 `main` 上和动作有关的路径是：

| 路径 | 读什么 |
|---|---|
| 仓库根 `README.md` 的 Finetune 表 | 哪几条是 Vision generator SFT，哪一条是 Policy-DROID SFT |
| `cookbooks/cosmos3/generator/action/README.md` | `fd` / `id` / `policy` 三种任务的输入输出 |
| `cookbooks/cosmos3/generator/action/finetune/README.md` | Policy-DROID / LIBERO 的 SFT，标题就写 Action-Policy |
| `cookbooks/cosmos3/generator/action/finetune/launch_sft_action_policy_droid_nano.sh` | 脚本注释：DROID action-policy SFT；默认 TOML 按 256 卡写 |
| `cookbooks/cosmos3/generator/action/run_policy_with_cosmos_framework.md` | 推理服务的是 Policy 检查点，不是 `fd` |

对照仓库 [nvidia-cosmos/cosmos-predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) 只读文档：

| 路径 | 读什么 |
|---|---|
| 根 `README.md` 模型表 | `robot/action-cond` 与 `robot/policy` 两行 |
| `docs/post-training_video2world_action.md` | 终点 A：Bridge 数据、conditioner 名、experiment 名 |
| `docs/inference_robot_action_cond.md` | `examples/action_conditioned.py`，暂不支持多卡 |
| cookbook 页 Cosmos Policy | 终点 B：潜帧注入、50/25/25 的 batch 划分、直接策略部署 |

读的时候做一张两列表，左列「世界模型后训练」，右列「策略后训练」。任何一行如果左右都想填，停下来看它的输入输出：条件里有没有逐步 $a$、输出主头是 $s'$ 还是 $a$。

Cosmos Policy 的训练代码当前分两处：Predict2 在 [NVlabs/cosmos-policy](https://github.com/NVlabs/cosmos-policy)，Predict2.5 在 Predict2.5 仓库的 `cosmos_predict2/_src/predict2/cosmos_policy/`。本课不进这两处训练脚本，避免和「世界模型后训练」的 `action_conditioned` 配置抄串。

## 7. 实验

主线顺序：先勾检查表（猜），再 grep 小仓库把动作通道标到文件名，再读 Cosmos 当前文档把两条后训练拆开，最后用第 33 课是否题打档。对换实验不重跑；第 22 课做过的人把当时的分岔记录抄进检查表「对换」那一格，没做过的人用第 5.3 节的协议在纸上写「若我有检查点，该怎么测」，不要把空格子填成论文 0.90。

### Step 0: 工业流水线检查表，先猜再揭晓

假设你是产线守门员。面前送来四个系统。每一行只许打三种章：停在生成器、可以进规划、可以上真机。先自己填，再往下读参考。

| 系统 | 你的章 | 一句话理由 |
|---|---|---|
| 只有 FVD / 人评很高的互联网视频模型 |  |  |
| DINO-WM，对换已分岔，尚未接 CEM |  |  |
| Cosmos Policy，默认直接出动作块，真机示范上成功率很高 |  |  |
| 桌宠小模型，对换分岔，安全层会截断伸手 |  |  |

参考答案（先自己勾完再看）：

| 系统 | 章 | 理由 |
|---|---|---|
| 高画质视频模型 | 停在生成器 | 缺动作通道，更缺对换。生成真不是进规划的门票 |
| DINO-WM 已对换、未规划 | 可以准备进规划，还不能上真机 | 第一问过了，第三问还没做；没有安全层 |
| Cosmos Policy 直接策略 | 停在策略，不能当成模拟器进规划 | 学的是 $\pi(a\mid s)$。真机成功是 E3 的规划好，不是 $P(s'\mid s,a)$ 过关 |
| 桌宠小模型加安全层 | 摄像头档可以在桌子上跑克制 | 流水线四拍齐。真机接触仍要第 27 课的闸，不能因为「有模型」就伸手 |

常见错法：看见 Cosmos 就盖「可以进规划」（把平台名字当成动力学验收）；看见 DINO-WM 就盖「可以上真机」（把仿真规划当成安全层）；四行全盖「停在生成器」（把诚实变成虚无）。笔记里写你当初盖错了哪一枚，这比抄对表有用。

三条硬规则，本课后面反复用：

```text
1. 缺动作对换，不能进规划
2. 缺安全层，不能上真机
3. 只有画质分数，停在生成器
```

### Step 1: 克隆或复用 DINO-WM

第 22 课已经 clone 过的人，进入那个目录即可。新开一份：

```bash
git clone https://github.com/gaoyuezhou/dino_wm.git
```

```bash
cd dino_wm
```

```bash
git log -1 --oneline
```

把短哈希抄进笔记。后面的行号以你这次检出为准。

### Step 2: 把动作通道标到文件名

在仓库根目录执行，一次只查一组文件：

```bash
grep -n "action_encoder\|action_emb_dim\|concat_dim" conf/train.yaml
```

预期能看到 `action_encoder: proprio`、`action_emb_dim: 10`、`concat_dim: 1`。然后：

```bash
grep -n "_target_\|ProprioceptiveEmbedding" conf/action_encoder/proprio.yaml
```

```bash
grep -n "class ProprioceptiveEmbedding\|Conv1d\|def forward" models/proprio.py
```

```bash
grep -n "def encode_act\|def replace_actions_from_z\|def rollout\|z_loss\|concat_dim" models/visual_world_model.py
```

```bash
grep -n "wm.rollout\|objective_fn\|num_samples" planning/cem.py
```

```bash
grep -n "loss_visual\|mode == .last\|alpha" planning/objectives.py
```

把结果填进这张表。文件名必须是你 grep 到的，不许凭记忆写一个「动作模块.py」。

| 零件 | 文件 | 符号或键 | 你看到的值 |
|---|---|---|---|
| 动作编码器配置 |  |  |  |
| 动作编码器类 |  |  |  |
| 拼接到 $z$ 的哪一维 |  |  |  |
| 训练损失排除的维 |  |  |  |
| 规划时改动作 |  |  |  |
| 规划器前向 |  |  |  |
| 规划代价 |  |  |  |

参考填法（先自己填完再对）：动作编码器配置在 `conf/train.yaml` 加 `conf/action_encoder/proprio.yaml`；类是 `models/proprio.py::ProprioceptiveEmbedding`；拼接是 `concat_dim: 1`，落在 `VWorldModel.encode`；损失排除动作维，落在 `forward` 的 `z_loss`；改动作是 `replace_actions_from_z`；规划器前向是 `CEMPlanner.plan` 里的 `wm.rollout`；代价是 `create_objective_fn` 的 `loss_visual + alpha * loss_proprio`。

### Step 3: 1xgpt 的负对照

```bash
git clone https://github.com/1x-technologies/1xgpt.git
```

```bash
cd 1xgpt
```

不要跑 `./build.sh`。本课验收不依赖下载 100 小时 token。打开 README 的 GENIE 节，把「only trains on video sequences, not actions」抄进笔记，然后：

```bash
grep -n "actions\|self.actions\|action_tokens" data.py
```

```bash
grep -n "action" genie/configs/magvit_n32_h8_d256.json
```

```bash
grep -n "def forward" genie/st_transformer.py
```

预期：`data.py` 拼出 `actions.bin`，`self.actions = np.memmap` 被注释；JSON 配置没有动作键；`STBlock.forward` 只接收 `x_TSC`。把三行证据写进检查表「1xgpt 基线」那一格：数据有动作文件名，计算图没有动作通道，停在生成器。

### Step 4: 用 curl 核 Cosmos 当前路径

不要凭旧博客写路径。在任意能上网的目录执行：

```bash
curl -sL https://raw.githubusercontent.com/nvidia/cosmos/main/cookbooks/cosmos3/generator/action/README.md
```

在输出里找到 `fd`、`id`、`policy` 三行定义，抄进笔记。再核策略 SFT 脚本是否仍在这个位置：

```bash
curl -sL -o /dev/null -w "%{http_code}\n" https://raw.githubusercontent.com/nvidia/cosmos/main/cookbooks/cosmos3/generator/action/finetune/launch_sft_action_policy_droid_nano.sh
```

预期打印 `200`。不是 200，就回到仓库根 README 的 Finetune 表，按当前树改你的笔记，不要沿用本课打印的过期路径。然后核 Predict2.5 的世界模型后训练文档：

```bash
curl -sL -o /dev/null -w "%{http_code}\n" https://raw.githubusercontent.com/nvidia-cosmos/cosmos-predict2.5/main/docs/post-training_video2world_action.md
```

```bash
curl -sL -o /dev/null -w "%{http_code}\n" https://raw.githubusercontent.com/nvidia-cosmos/cosmos-predict2.5/main/docs/inference_robot_action_cond.md
```

两条都应是 `200`。打开 `post-training_video2world_action.md`，把 experiment 名 `ac_reason_embeddings_rectified_flow_2b_256_320` 和 conditioner 名 `action_conditioned_video_conditioner` 抄下来。这两行属于终点 A。`launch_sft_action_policy_droid_nano.sh` 属于终点 B。

### Step 5: 画出后训练流水线

在笔记里画四拍，用编号，不要用箭头符号。旁边另开一条「策略分岔」支路。下面是最低限度的合法画法，你可以改措辞，不能把两条终点画成一个方块。

```text
拍1  预训练视频模型（文本或首帧条件，验收生成真）
拍2  加动作通道，在带动作标签的轨迹上后训练动力学
拍3  动作对换。不过关就停。过关才能让规划器查询 rollout
拍4  安全过滤之后，才允许把选出的动作送到真机或桌宠出口

支路B  同一预训练检查点，后训练成策略头（Cosmos Policy / Policy-DROID）
        验收是任务成功率。要进规划，仍须单独过拍2和拍3
```

在拍 2 上标注你 Step 2 写下的文件名。DINO-WM 写 `models/proprio.py` 加 `VWorldModel.encode`。1xgpt 写「本应是 `data.py` 的 `actions.bin`，当前未接」。Cosmos 终点 A 写 `action_conditioned_video_conditioner`。Cosmos 终点 B 写 `launch_sft_action_policy_droid_nano.sh`，并且明确它不替代拍 2。

### Step 6: 用第 33 课是否题打两档

只答七条里本课用得上的前四条，其余写「本课未测」。系统 A：DINO-WM 在 PushT 上的规划探针（第 22 课做过，或仅根据源码与论文）。系统 B：Cosmos Policy 默认直接策略（根据论文第 4 节和 cookbook「deploying as a direct policy」）。

| 是否题 | DINO-WM 规划探针 | Cosmos Policy 默认 |
|---|---|---|
| Q1 动作对换是否分岔 | 第 22 课实验或源码上 `replace_actions_from_z` 可做 | 论文未把对换写成默认验收；本课记未测 |
| Q2 预测是否用于选动作 | 是，CEM 查 `rollout` | 默认否；best-of-N 才是 |
| Q3 是否同一物理世界 | 否，gym | 真机实验是；仿真基准否 |
| Q4 失败是否可无限重置 | 是 | 仿真是，真机否 |
| 档 | 近 E2 | 真机默认近 E3 |

「近」是因为你可能没有亲自跑完所有是否题。不许因为 Cosmos 更大就把 E3 写成 E4。Q2 的「是」必须指向一次查询：DINO-WM 是 `wm.rollout`，Cosmos Policy 默认部署没有这个查询。

### Step 7: 回填 Step 0，写一句桌宠用途

回到 Step 0 的四行章。若你改了主意，记下改了哪一枚、因为第几步的哪条证据。最后写一句给第 30 课的对照：

```text
第 30 课 = 拍1用冻住的 DINOv2，拍2用四键动作通道，拍3强制对换，拍4留给第 32 课克制
```

有第 22 课对换日志的人，把当时的 $d(a,a')$ 抄进拍 3。没有日志的人不要编造数字。

## 8. 配置与预算

本课必做是读文件和填表，预算按小时计，不按卡时计。

| 步骤 | 硬件 | 时间 | 磁盘 |
|---|---|---|---|
| Step 0 检查表 | 纸或编辑器 | 20 分钟 | 0 |
| clone DINO-WM + grep | CPU / Mac | 10 分钟 | 仓库本身，百 MB 级 |
| clone 1xgpt + grep | CPU / Mac | 10 分钟 | 同上；不跑 `./build.sh` 则不下数据 |
| curl Cosmos 文档 | 能上网即可 | 30-60 分钟读完并填两列表 | 0 |
| 打档 + 回填 | 纸 | 30 分钟 | 0 |

对照系统的真实训练预算，只用来建立数量级，禁止当本课作业：

| 作业 | 官方口径 | 本课档位 |
|---|---|---|
| DINO-WM 在 PushT 上训预测器 | `conf/train.yaml`：`epochs: 100`，`batch_size: 32`，`predictor_lr: 5e-4`，集群 launcher 写过 H100 | 第 22 课实战过推理与对换；本课不重训 |
| 1xgpt GENIE_138M | README 提供现成检查点；训练脚本 `python train.py --genie_config genie/configs/magvit_n32_h8_d256.json` | 第 11 课实战过基线；本课只读动作通道 |
| Predict2.5 action-cond 后训练 | `docs/post-training_video2world_action.md` 示例是单进程 `torchrun --nproc_per_node=1`，数据是 IRASim 划分的 Bridge | 只讲。24GB 也不把这条写成必做 |
| Cosmos Policy LIBERO | cookbook：64×H100，48 小时，全局 batch 1920，40K step，动作块 16 | 只讲 |
| Cosmos Policy RoboCasa | 32×H100，48 小时，动作块 32 | 只讲 |
| Cosmos Policy ALOHA | 8×H100，48 小时，约 185 条示范，动作块 50 | 只讲 |
| Cosmos 3 Policy-DROID SFT | `launch_sft_action_policy_droid_nano.sh` 的 TOML 钉 GB200、HSDP 32×8、256 卡、全局 batch 8192、10000 iter、lr $2\times 10^{-4}$ | 只讲。单机 8 卡只能按 README 改 `data_parallel_replicate_degree` 做冒烟，仍不是本课作业 |

超参只抄你用得上的。DINO-WM 动作侧：`action_emb_dim: 10`、`concat_dim: 1`、`num_hist: 3`、`num_pred: 1`、`frameskip: 5`、`normalize_action: True`。规划侧：`horizon: 5`、`num_samples: 300`、`topk: 30`、`opt_steps: 30`、`objective.mode: last`、`alpha: 1`。Cosmos Policy 推理若有人加餐，cookbook 写过把 $\sigma_{\max}$ 收到 80、$\sigma_{\min}$ 收到 4（Predict2）；Predict2.5 用 logit-normal、shift 5。那些数属于只讲，不要写进你的桌宠配置。

检查点。DINO-WM 官方 PushT 权重在 OSF，路径必须能让 `plan.py` 拼出 `outputs/pusht/checkpoints/model_latest.pth` 加 `hydra.yaml`。本课不强制下载。Cosmos 3 训练产物按根 README 写在 `outputs/train/<project>/<group>/<name>/checkpoints/`，导出要先 `cosmos_framework.scripts.export_model` 再 `convert_model_to_diffusers`。本课不跑这两条。

Mac / CPU。grep、curl、填表全部可做。DINO-WM 官方 `environment.yaml` 锁 Linux 上的 CUDA 轮子，本课本来就不训。1xgpt 的 `build.sh` 会下大数据，跳过。

## 9. 验收

量化线很少，因为本课主实验不是拟合。合格看的是分流有没有画错、文件名有没有标对。

- [ ] 流水线四拍加一条策略支路，两终点没有画在同一个方块里。
- [ ] DINO-WM 动作通道表至少七格都指向真实文件：`conf/train.yaml`、`conf/action_encoder/proprio.yaml`、`models/proprio.py`、`models/visual_world_model.py` 的 `encode` / `replace_actions_from_z` / `forward`、`planning/cem.py`、`planning/objectives.py`。
- [ ] 1xgpt 笔记写清：`data.py` 拼了 `actions.bin`，`self.actions` 的 `memmap` 被注释，`STTransformerDecoder` 不吃动作。
- [ ] Cosmos 两列表：左列 `fd` 或 `robot/action-cond` 或 `post-training_video2world_action.md`，右列 `policy` 或 `launch_sft_action_policy_droid_nano.sh` 或 Cosmos Policy cookbook。curl 状态码记下来。
- [ ] 检查表三条硬规则原文能背：缺对换不进规划，缺安全层不上真机，只有画质停在生成器。
- [ ] 口头一句：DINO-WM 学 $P(s'|s,a)$，Cosmos Policy 默认学 $\pi(a|s)$（可附带 $s'$ 和 $V$）。
- [ ] 档：DINO-WM 规划探针近 E2，Cosmos Policy 默认近 E3。证据写成「论文 / 仓库 / 我的实验」之一，未测的格子写未测。
- [ ] 桌宠对照句写了：第 30 课是这条流水线的缩小版。

可视检查。把 Step 5 的四拍给另一个人看，问「Cosmos Policy 该进哪一拍」。若对方指到拍 2 或拍 3 当默认，你的图还不够拆。把 grep 输出的截图或粘贴和文件名表并排，行号对不上就以本地仓库为准改表，不要改仓库来迁就课文。

不合格的典型交卷：只有 Cosmos 宣传摘要；把 Transfer1 填进动作通道；用 LIBERO 98.5%（论文报告，本课未跑）证明动力学；把第 22 课的 PushT 规划成功率抄过来冒充本课新实验。

## 10. 排错

| 症状 | 原因 | 怎么验证 | 怎么修 |
|---|---|---|---|
| grep 找不到 `encode_act` | 不在仓库根，或检出了不含该符号的 fork | `pwd` 和 `git remote -v` | 回到 `dino_wm` 根再 grep |
| `conf/action_encoder/proprio.yaml` 不存在 | 目录名记错成 `conf/encoder` | `ls conf` | 动作编码器在 `conf/action_encoder/`，视觉编码器在 `conf/encoder/` |
| 以为 DINO-WM 的损失也预测动作 | 没读 `z_loss` 的切片 | 看 `forward` 里 `:-self.action_dim` | 动作是条件。不要把动作维加回损失再和论文比规划 |
| 1xgpt 的 `grep action data.py` 有命中，就写成已经接上 | 命中的是被注释的 `memmap` 和路径字符串 | 看该行是否以 `#` 开头 | 注释掉的代码不是计算图。流水线记「未接」 |
| `curl` 返回 404 | 路径已搬家 | 打开仓库根 README 的 Finetune 表 | 按当前 README 改笔记。本课写过「不要抄过期路径」就是为这一行 |
| 想跑 Cosmos 3 SFT，8 卡立刻 OOM 或拒启动 | TOML 钉的是 256 卡全局 batch 8192 | 读 `launch_sft_action_policy_droid_nano.sh` 顶部注释 | 本课不要跑。若加餐，按该 README 改 `data_parallel_replicate_degree` 并降 `max_iter`，结论单独标加餐 |
| 把 `robot/policy` 检查点拿去对换 | 策略头条件是 $s$，不是「给定 $a$ 出 $s'$」 | README 模型表的输入列 | 对换必须用 `action-cond` 或 DINO-WM 这种吃 $a$ 的前向 |
| Transfer1 生成很听分割图，就当动作条件过关 | 空间控制不是 $a_t$ | 论文摘要：segmentation, depth, edge | 检查表记「可控生成器」，进下一课再拆驾驶条件 |
| 想重跑第 22 课 PushT 来凑本课篇幅 | 验收目标理解错 | 看本节清单有没有「成功率」 | 本课不收规划数字。把时间花在两列表和检查表 |
| Cosmos Policy 论文写了 world model 目标，就当成拍 2 已验收 | 50/25/25 是训练时的条件掩码，默认部署仍是直接策略 | cookbook 原句：focuses on deploying as a direct policy | 默认记 $\pi$。只有你真用 $s,a$ 条件出 $s'$ 并做了对换，才把拍 2 勾上 |
| 桌宠小模型画质差，不敢接规划 | 把生成真当成闸门 | 第 30 课验收是对换分岔，不是重建好看 | 画质差可以不开解码器。分岔不过才停 |
| 档打成 E4 | 把「模型很大 / 在真机上成功」当成查询了前向模型 | 第 33 课 Q2 | 没有 `rollout` 查询就停在 E3 |

## 11. 前沿与改造

### 前沿怎么做

2024 到 2026 年，公开系统把「视频生成器变机器人脑子」拆成了至少四条产品线，本课已经对上号。

动作条件视频世界模型。IRASim 用 DiT 加帧级动作条件，在 RT-1、Bridge、Language-Table、RoboNet 上做轨迹到视频，并把同一模型接到策略评估和模型基规划。Predict2.5 的 `robot/action-cond` 后训练直接吃 IRASim 划分的 Bridge。Cosmos 3 Generator 的 `fd` 把前向动力学收成一种官方任务，动作维按身体切换（自车 9 维、DROID 10 维、人手 57 维）。

策略后训练。Cosmos Policy 把视频扩散的学习算法直接用在动作块上，潜帧注入避免另接一个动作扩散头。论文报告 LIBERO 平均成功率 98.5%、RoboCasa 67.1%、真机双臂 93.6%，带规划时两项困难真机任务平均再高 12.5 个百分点。这些是规划好，本课未复现。Cosmos 3 的 `Cosmos3-Nano-Policy-DROID` 把同一思路收成可下载检查点，SFT cookbook 与 `fd` 笔记本分目录存放。

空间可控生成。Transfer1 用分割、深度、边缘做世界到世界迁移。仿真变真、驾驶数据增广走这条，不走 $a_t$。

无权重的志向稿。UniSim 把多种数据集编排成交互模拟器，用来在模拟里训策略再搬到真机。Genie 3 是订阅演示。二者都不能练。

和本课缩小版的差距，规模能买一部分，机制买不来。规模：Cosmos 预训练的视频量和 64 张 H100 的 SFT，24GB 卡买不起。机制：动作通道、对换闸门、规划器查询接口、安全层，第 30 课已经在桌子上走通。缺的往往是把四拍画成两步，或把策略成功率当成动力学验收。

### 动手改造清单

四个都可以在本课仓库里做。每个写清改哪里、预算、看到什么算成功、什么算失败。

1. 1xgpt 接上加法动作嵌入。改 `data.py`：取消 `self.actions` 的注释，按数据卡实际布局读 `actions.bin` 或 `actions/` 下的文件，让 `__getitem__` 返回动作。改 `genie/st_transformer.py`：在 `STBlock` 或 Decoder 入口加 `nn.Embedding` 或线性层，加到时间维。预算：第 11 课已经能训的那档 GENIE 小模型，单卡 24GB 数小时到一天量级，以你当时的墙钟为准。预期：同一 prompt 帧、两个不同动作，生成 token 的汉明距离或解码图像的像素差明显大于同动作重复采样。失败：损失下降但两动作输出几乎重合。那只是又训了一个视频生成器。

2. DINO-WM `concat_dim` 从 1 改到 0。`conf/train.yaml` 一处，`VWorldModel.encode` 会把动作改成额外 token。预算：第 22 课的 PushT 缩小训练，24GB 卡数小时冒烟即可，不必满 100 epoch。预期：对换仍分岔，但分岔可能变弱或变慢，因为动作不再出现在每个格子上。失败：`concat_dim=0` 时对换塌掉。那说明默认的「每个 patch 都带动作」不是装饰。

3. 给桌宠补查询接口，不换骨干。在第 30 课胶水脚本里，模仿 `replace_actions_from_z` 写一个 `swap_and_rollout(obs, act)`，返回特征序列。预算：CPU 可做，一晚。预期：四键对换表的对角线近 0，看左对看右随步数增大。失败：接口接上了，四键想象仍重合。先查动作有没有被写成常数，再查是不是只在第一步看了动作。

4. 安全层查询的必须是拍 2 过关的模型。把第 27 课桌沿闸的输入，从几何估计换成 `swap_and_rollout` 的最后一帧特征或解码位置。预算：第 30 课模型加一天。预期：伸手键的想象若把杯子送到沿内 5 cm，出口改成停。失败：过滤器在动作盲模型上也能「工作」（两键都停或两键都不停）。那是规则，不是查询。

顺手复现的方向：IRASim 论文说帧级条件强于视频级共用一个动作向量。缩小版对应改造 2 和改造 3：动作只出现一次，对换应当变差。预期能看到同方向趋势，不能复现他们在 RT-1 上的 FVD 表。Cosmos Policy 论文说直接策略已经很强、规划再加点。缩小版不要去复现 98.5%；要复现的是「默认部署可以不查询 $P$」。打开 cookbook，确认主路径是出动作块而不是 `fd`，这一方向就算对上了。

## 12. 论文与延伸

读之前先带着本课的两终点。任何一篇把 $\pi$ 和 $P$ 写在同一段落里的，都要自己拆成两行再往下看。

1. Kim, M. J., et al. Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning. arXiv:2601.16163, 2026. 问：潜帧注入之后，训练时 50/25/25 的条件掩码分别对应哪个公式？cookbook 默认部署用的是哪一份？你能不能在不看成功率表的情况下，说出它为什么默认近 E3。

2. NVIDIA et al. Cosmos-Transfer1: Conditional World Generation with Adaptive Multimodal Control. arXiv:2503.14492, 2025. 问：控制输入是分割、深度、边缘。若有人把 Transfer 填进「动作通道」那一格，你用哪一句话把它拿出来？它和下一课驾驶条件板的关系是什么。

3. Agarwal, et al. Cosmos 3 技术报告. arXiv:2606.02800, 2026. 仓库 [nvidia/cosmos](https://github.com/nvidia/cosmos) 同步。问：Reasoner 和 Generator 哪条通路在做去噪？`fd`、`id`、`policy` 三种任务各自缺第 12 课的哪一问。公开检查点名字里带 Policy 的，默认不能当 $P(s'|s,a)$ 用。

4. Zhou, G., Pan, H., LeCun, Y., Pinto, L. DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning. arXiv:2411.04983, 2024. 问：损失为什么不含动作维？规划代价为什么可以不看解码器？把第 3.2 节的 $\mathcal{C}=\|\hat z_T-z_g\|^2$ 对到 `planning/objectives.py` 的哪一个 `mode`。

5. Zhu, F., et al. IRASim: A Fine-Grained World Model for Robot Manipulation. arXiv:2406.14540, 2024. 项目页 [gen-irasim.github.io](https://gen-irasim.github.io/)，代码 [bytedance/IRASim](https://github.com/bytedance/IRASim)。问：轨迹到视频和「整段视频一个动作向量」差在哪一帧？论文把 Push-T 上扩散策略的 IoU 从 0.637 报到 0.961，这是规划好还是生成真。本课未复现该数字。

6. Yang, S., et al. Learning Interactive Real-World Simulators. arXiv:2310.06114, 2023. 问：UniSim 用哪些类型的数据补哪些能力？它训出来的高层策略和低层 RL 是 $\pi$ 还是 $P$。为什么本课把它标成只讲。

7. 回访 Finn, Levine 与 Ebert 等人的 Visual Foresight（arXiv:1812.00568）和[第 24 课](24_visual_foresight.md)。问：2018 年已经把动作条件视频预测接到 CEM。2026 年工业后训练多出来的是规模和预训练，还是新的闸门。你的检查表里哪一条 2018 年就有。

延伸阅读按需，不要求做笔记：Predict2.5 文档 `docs/post-training_video2world_action.md` 与 `docs/inference_robot_action_cond.md`；Cosmos Cookbook 的 Cosmos Policy 页；1X World Model Challenge 的 Evaluation Challenge 段落；[第 17 课](17_evaluating_world_models.md) 三分评测，用来给任何一份「SOTA 世界模型」广告拆尺子。

[第 39 课](39_long_horizon_memory.md) 把杯子留在记忆里。本课给记忆接上动作端口，并规定：没通过对换的端口不准接到规划器，没装安全层的规划器不准上真机。[下一课](41_driving_world_models.md) 换验证者：路，不是桌子。驾驶世界模型会把条件拆成自车动作、他车与行人、天气与风格。高清多相机仍然可能只是 E0 或 E1。桌宠用不到车，用得到那张条件分类表。毕业标准仍在[第 32 课](32_ship_desk_pet.md)、[第 33 课](33_embodiment_degrees.md)：会看、会想、会克制，以及是否题打出来的档。
