---
id: 25_lerobot_imitation
title: "先模仿，再谈世界模型"
summary: "ACT 和 Diffusion Policy 没有显式动力学，桌宠能走多远？缺了什么？"
unit: embodied
play_tools: []
checkpoints:
  - "一条能回放的数据集加训练记录（论文复现 #8）。"
  - "模仿 vs 动力学对照笔记。"
---

# 第 25 课：先模仿专家动作，再谈世界模型

> 类型：复现（论文复现 #8：公开桌面臂数据集上 ACT 或 Diffusion Policy 的方向性复现）<br>
> 建议周期：3-5 天（安装与回放半天，小配置训练挂机 0.5-4 小时，对照笔记 1 天）<br>
> 硬件：单张 24GB 卡训 ACT 小配置约 0.5-2 小时；Mac（MPS）能训但更慢；纯 CPU 只建议做回放、钢琴卷帘和离线评测。无真机。<br>
> 锚定仓库：[huggingface/lerobot](https://github.com/huggingface/lerobot)，文档 [huggingface.co/docs/lerobot](https://huggingface.co/docs/lerobot)；机器人页 [SO-101](https://huggingface.co/docs/lerobot/en/so101)<br>
> 产物：一条能回放的公开桌面抓放轨迹、一份 ACT 或 Diffusion Policy 小配置训练记录、一张模仿策略 vs 世界模型对照表

## 1. 这一课做什么

整门课的主干还是那句：

```text
观察 先压成状态 再按动作预测下一状态 然后展开多条未来 给未来打分 最后选动作
```

八幕里你现在在第七幕「接到身体上」（24-27）。按第 33 课的尺子，本课默认停在 E3：有身体或身体数据集，选动作时不查询前向模型。升到 E4 要等到把第 30 课的世界模型接到动作出口。

| 幕 | 课 | 一句话 |
|---|---|---|
| 第一幕：第一个世界模型 | 01-04 | 完整复现 World Models：V-M-C 三件套 |
| 第二幕：潜空间方法 | 05-08 | RSSM、Dreamer、MuZero、TD-MPC2 |
| 第三幕：生成式世界模型 | 09-12 | token、扩散、latent action |
| 第四幕：JEPA 路线 | 13-16 | 不生成像素只预测表征 |
| 第五幕：评测与改造 | 17-19 | 评测、缩放、消融、改结构 |
| 第六幕：空间与物体 | 20-23 | 持久 3D/4D、视频基础模型、物体中心 |
| 第七幕：接到身体上 | 24-27 | 视觉预测控制、模仿策略、VLA 与仿真到真机 |
| 第八幕：桌宠毕业设计 | 28-32 | 在一张桌子上做出会看、会想、会克制的小机器人 |

第 24 课把预测变成了规划：先在模型里推几步，再决定真不真动手。这一课先把「手」本身装上。桌上挥手、点头、把笔推正，今天的机器人项目几乎都先训一个模仿策略：看专家怎么动，照着出动作。它不预测杯子会不会倒，只预测专家接下来会怎么动关节。

这和主干里的世界模型职责不同。世界模型吃当前状态和打算做的动作，吐下一状态的分布 $P(s_{t+1} \mid s_t, a_t)$。模仿策略吃当前观察，直接吐动作，学的是 $\pi(a_{t:t+k} \mid o_t)$。没有显式动力学，却能在几十条示教之后把抓放做得像样。桌宠要先会「像人那样动」，再谈「别把水推倒」。

为什么人人先训它们：数据采集是遥操作，损失是监督，单卡几小时能看到关节轨迹贴上专家；世界模型规划还要学动力学、要想象展开、要在真机上处理接触和延迟。模仿先给一条能用的动作出口。它走不远的地方也清楚：物体换个位置、人手伸进画面、杯子比训练时更靠桌沿，策略仍会按记忆里的动作块往前走，因为它没有「如果我反过来抓」这条通道。

本课锚定 [huggingface/lerobot](https://github.com/huggingface/lerobot)。实验全部用 Hub 上的公开数据集，不把买手臂写成及格条件。有 SO-101 的人可以把同一套权重接到真机；没有的人把回放、训练和离线评测跑完，照样交复现 #8。

做完你能验证三件事：公开抓放轨迹能在本地一帧帧看完；ACT 或 Diffusion Policy 的训练损失相对均值动作基线下降；你能用一张表说出模仿策略补了哪一截、世界模型还要补哪一截。第 26 课才会碰到带语言指令的视觉-语言-动作模型（VLA）。ACT 没有语言骨干，不要把两者写成同一种东西。

术语速查：

| 术语 | 一句人话 |
|---|---|
| 模仿学习 / 行为克隆 | 把专家的（观察, 动作）当成监督学习：看见类似画面，就出类似动作 |
| 策略 $\pi$ | 从观察到动作的映射；本课的 ACT 和 Diffusion Policy 都是策略 |
| 动作块（action chunk） | 一次预测未来连续 $k$ 步动作，而不是只出当前这一拍 |
| ACT | Action Chunking with Transformers：用 Transformer 一次吐一截关节轨迹，可选 VAE 风格变量 |
| Diffusion Policy | 把整段动作轨迹当成要去噪的样本，观察当条件 |
| 遥操作 | 人动主臂或手柄，从臂跟着动，用来采示教 |
| 复合误差 | 每步动作稍偏一点，观察渐渐离开训练分布，后面越错越远 |
| 离线评测 | 不接真机：在留出的示教上比预测动作和专家动作的误差，或在仿真里测成功率 |
| SO-101 | LeRobot 文档里的旗舰低成本桌面臂，6 个舵机；本课用公开数据，不要求你组装 |
| VLA | 观察加语言指令映射成动作的大模型，第 26 课的主角，本课只划界 |

## 2. 问题

桌面操作看起来像「看见了就伸手」。真做起来有三处会把朴素行为克隆打穿，也有一处必须先划开。

第一处是时序。人抓杯子时，手指合拢、手腕旋转、肘部后撤是一段连贯轨迹。若模型每 33 毫秒独立猜一个关节角，相邻两拍会对不上，夹爪在接触面上发抖。ACT 的回答是一次预测 $k$ 步（LeRobot 默认 `chunk_size=100`），执行队列里的动作，队列空了再问网络。Diffusion Policy 把同一段轨迹当成一条要去噪的样本，多峰动作分布（绕左边或绕右边）可以同时存在。

第二处是误差滚动。示教里杯子总在垫子中央。部署时杯子偏了两厘米，第一步动作仍按「中央」去够，画面变得更陌生，第二步更偏。这叫复合误差，是模仿策略的慢性病。动作块能把「一段内的自洽」做强，不能凭空补上没见过的物体布局。世界模型在这里的用法不同：它可以问「如果夹爪按这个角度合上，杯子会不会翻」，在执行前拒绝一条会闯祸的动作。

第三处是数据口径。公开的 SO-101 / ALOHA 数据集是人遥操作录的，相机位、桌布、物块颜色都写进了分布。你在 Hub 上训出来的损失下降，只说明模型在拟合这份示教，不说明换一张桌子还能抓。本课的复现档是方向性的：损失相对均值动作基线下降，钢琴卷帘上未来 0.5 秒的动作块看起来像专家，不冒充论文里那台 ALOHA 双臂的 80% 至 90% 真机成功率。

必须划开的界限：VLA（OpenVLA、$\pi_0$、SmolVLA）也是「观察进、动作出」，但第一公民是语言指令，骨干是预训练视觉-语言模型，训练预算和评测协议都不是本课这套。ACT 的输入是多路相机加关节角，没有指令编码器。LeRobot 文档把 ACT 放在 Imitation Learning，把 $\pi_0$ 和 SmolVLA 放在 VLA。下一课再管语言。本课谁把 ACT 写成 VLA，验收直接不合格。

另一条界限：买不买 SO-101 都不影响及格。`lerobot-replay` 会把示教动作打到真机上，那是加分项。主线用 Hub 数据集做可视化回放和离线评测。

## 3. 准备

- 第 24 课的概念存货：你会用动作条件预测做规划，知道「动作盲」基线是什么。本课不调用那份规划代码，但对照表要写规划那一列。若第 24 课还没写完，用 [第 05 课](05_rssm_planet.md) PlaNet / [第 06 课](06_dreamerv3_imagination.md) Dreamer 的「在想象里选动作」即可填表。动作条件本身是 [第 03 课](03_mdn_rnn_action_conditioned.md) 的考试题。
- Python 3.12 和环境隔离。官方安装页推荐 miniforge 的 `conda`，也支持 `uv` / `venv`。给 LeRobot 单开一个环境，不要和 [第 01 课](01_world_models_hands_on.md) 的老 gym、[第 09 课](09_iris_token_world_model.md) 的 IRIS 混装。
- 磁盘：本课主数据集 `lerobot/svla_so101_pickplace` 约几十 MB；`lerobot/pusht` 同量级。预留 5GB 给缓存、checkpoint 和可视化即可。
- 账号：从 Hub 拉公开数据一般不需要写权限。不想把模型推上去，训练时加 `--policy.push_to_hub=false`。要用 Weights & Biases 再 `wandb login`，本课默认关掉。
- 真机可选：有 SO-101 的人按 [SO-101 组装页](https://huggingface.co/docs/lerobot/en/so101) 校准，本课不把这一步写入验收。文档写从臂用 6 个 STS3215（减速比 1/345），主臂六个关节的减速比不同，为的是人拖得动、又能撑住自重。Feetech 额外依赖是 `pip install -e ".[feetech]"`，只在接手臂时才需要。标定文件按 `--robot.id` 存放，遥操作、录数据和部署必须用同一个 id。
- 对照阅读：先打开 [LeRobot 文档首页](https://huggingface.co/docs/lerobot) 和 [ACT 页](https://huggingface.co/docs/lerobot/act)。文档版本以你打开当天的 `main` / `v0.6.x` 为准；本课命令按 2026-08 的当前页核对。

## 4. 学习目标

1. 在白纸上写出模仿策略 $\pi(a_{t:t+k} \mid o_t)$ 和世界模型 $P(s_{t+1} \mid s_t, a_t)$ 的输入输出，并指出桌宠的哪类行为该走哪一条。
2. 解释 ACT 为什么一次吐一截动作：动作块解决什么、解决不了什么；推理时潜变量为什么置零。
3. 解释 Diffusion Policy 为什么把动作轨迹当成去噪对象：多峰动作分布在这里怎么被表达，执行时为什么只取 horizon 里的一段。
4. 独立安装当前版 LeRobot，回放 `lerobot/svla_so101_pickplace` 的一条抓放，画出该条轨迹未来 0.5 秒的钢琴卷帘。
5. 在公开数据上训完一个 ACT 或 Diffusion Policy 小配置，离线评测优于均值动作基线，并留下命令、commit、种子和曲线。
6. 填完模仿 vs 世界模型对照表，口头说明为什么 VLA 和 ACT 不能混称。

## 5. 原理

五个机制。每个都落到：为什么需要、怎么运转、公式、LeRobot 里的类、怎么验收。

### 5.1 策略学「下一步怎么动」，世界模型学「这么动之后会怎样」

学骑自行车有两种笔记。一种记下教练的手脚：看见车把往左倒，就往左拧。另一种记下车和地面：车把这样拧，前轮会朝哪、人会不会摔。第一种是策略，第二种是世界模型。

策略的最小定义：

$$
\pi(a_t \mid o_t)
\quad\text{或一次多步}\quad
\pi(a_{t:t+k} \mid o_t)
$$

$o_t$ 是此刻的观察（相机画面加关节角），$a$ 是关节位置或末端增量。训练信号来自专家轨迹，损失是预测动作和示教动作之间的距离。没有奖励，也没有下一帧预测。

世界模型的最小定义仍是第 01 课那条：

$$
P(s_{t+1} \mid s_t, a_t)
$$

它必须随「打算做的动作」分岔。第 03 课的动作对换实验考的就是这件事。策略没有这个接口：你把「反过来抓」喂给 ACT，它不会返回杯子的未来姿态，它只会再吐一截它觉得专家会做的动作。

类比失效处：教练笔记在教练去过的路上很强；路况一变，笔记不会自己推演物理。桌宠「挥手、点头、把笔推正」走策略就够；「手伸过去会不会碰到杯子」必须有动力学，哪怕是缩小版。

LeRobot 把两类东西都收进同一个仓库。README 的模型表把 ACT、Diffusion、VQ-BeT 列在 Imitation Learning，把 VLA-JEPA、LingBot-VA、FastWAM 列在 World Models。本课只练前一类。你在 [第 17 课](17_evaluating_world_models.md) 用过的尺子在这里仍然适用：预测准、生成真、规划好是三件事；模仿策略连「预测下一状态」这一项都没有，它的分数只能叫「像不像专家」，不能叫「懂不懂桌子」。

### 5.2 ACT：一次生成一截动作，用潜变量消化示教风格

Zhao、Kumar、Levine、Finn 的 ALOHA 论文（arXiv:2304.13705）要在低成本双臂上做插电池、开酱料杯盖这类细动作。人的示教又快又抖，同一种抓法每次关节轨迹都不一样。若每步独立回归平均动作，模型会在两种合法抓法中间走出一条谁也不像的轨迹，夹爪在杯沿上打滑。

ACT 做了两件事。

第一件是动作分块。网络一次输出 $k$ 步关节目标。执行端维护一个队列：队列空了才再次前向。一块之内的动作由同一个前向产生，时间上自洽，复合误差的入口从「每一步」变成「每一块」。LeRobot 默认 `chunk_size=100`、`n_action_steps=100`，30 Hz 的 SO-101 数据上大约是 3.3 秒。你可以改成预测 100 步、只执行 50 步，后半截丢掉，用新观察重算。

用数字看一次。画面 30 Hz，朴素行为克隆每 33 毫秒问一次网络，1 秒问 30 次，每次都可能和上一拍对不齐。ACT 默认 3.3 秒才问一次，这一段里的 100 个关节向量来自同一次前向，夹爪合拢和手腕旋转被绑在同一条轨迹上。代价是：这 3.3 秒里杯子若被人手挪开，策略仍在执行旧块。缩短 `n_action_steps` 就是在「自洽」和「勤看」之间挪指针。

第二件是条件 VAE。训练时，VAE 编码器看见整段专家动作和当前关节，吐出潜变量 $z$ 的均值和方差；解码器（Transformer）在 $z$、关节、多路图像条件下重建这段动作。损失是重建的 L1 加上 KL：

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{L1}}
+
\lambda\,
D_{\mathrm{KL}}
\bigl(q(z \mid a_{t:t+k}, s_t)\,\|\,\mathcal{N}(0, I)\bigr)
$$

`configuration_act.py` 里 $\lambda$ 是 `kl_weight=10.0`，`latent_dim=32`。推理时没有专家动作可编码，$z$ 被置成全零，见 `modeling_act.py` 里 `ACT.forward` 的 `else` 分支。$z$ 在训练时吸收「这一条示教偏快还是偏慢」这类风格，推理时走先验均值。

图像走 ImageNet 预训练的 ResNet-18，多机位特征图展平后和 $z$、关节一起进 Transformer 编码器。解码器带 `chunk_size` 个可学习查询，每个查询对应未来一拍动作。这里有一处必须照仓库写：原版 ACT 宣称 7 层解码器，实现里只有第一层真正被用到（`tonyzhaozh/act` 的 issue 25）。LeRobot 把 `n_decoder_layers` 设成 1，明确对齐那个实现，而不是对齐论文句子。

时间集成（temporal ensembling）是论文算法 2：每步都重新预测一块，对重叠部分做指数加权。LeRobot 默认 `temporal_ensemble_coeff=None`，走动作队列，不走集成。要复现论文那条路径，需要 `n_action_steps=1` 且给出系数，配置类里写了这个约束。

ACT 是策略。VAE 里的 $z$ 是动作风格，不是下一帧画面，也不是物体状态。不要把它和第 02 课 VAE 的图像潜向量当成同一个零件。

### 5.3 Diffusion Policy：把动作轨迹当成一张要洗干净的「图」

Chi 等人的 Diffusion Policy（arXiv:2303.04137，RSS 2023，IJRR 扩展版 2024）把第 10 课你见过的扩散引擎从像素搬到了动作。第 10 课 DIAMOND 去噪的是下一帧画面，条件是历史和动作，它是世界模型引擎。本课去噪的是未来动作轨迹，条件是观察，它是策略。

训练时，取专家轨迹 $a^{0}$（长度 `horizon`，默认 64），抽一个扩散步 $\tau$，按调度把噪声加进去，得到 $a^{\tau}$。一维条件 U-Net 吃 $a^{\tau}$、$\tau$ 和观察编码，默认预测噪声 $\epsilon$（`prediction_type="epsilon"`）。损失是逐元素 MSE。观察编码是：过去 `n_obs_steps`（默认 2）步的关节，加上每路相机一个 ResNet，拼成 `global_cond`。

推理时从纯噪声出发，按 DDPM 或 DDIM 一步步撤噪声，得到一条完整 horizon 的动作。真正执行的只是其中 `n_action_steps` 步（默认 32），从「当前观察对应的那一拍」切出来。这是收缩视野控制：预测看较长的未来，执行只走前一小段，走完用新观察再生成。配置里写明需要

$$
n_{\text{action}} \le \text{horizon} - n_{\text{obs}} + 1
$$

默认 32、64、2 满足这个不等式。`drop_n_last_frames=7` 避免在轨迹末尾过度填充，注释写这是对齐原实现后训练更稳的选择。

扩散对策略有用的地方，是多峰。绕左侧抓和绕右侧抓都合法，L1 / MSE 回归会平均出一条撞杯子的中间路径。去噪过程可以从噪声里采样出其中一条，而不是两条的均值。训练稳定性也是论文强调的点：同一套配方在 4 个基准 12 个任务上平均相对提升 46.9%（论文摘要原句，那是他们的协议和他们的对照，不是你这张 24GB 卡上的数字）。

LeRobot 实现里，EMA 影子权重要显式打开：`--ema.enable=true`，评测时加载 `pretrained_model_ema`。这是仓库 `docs/source/policy_diffusion_README.md` 的写法，默认训练命令不会替你开。

和 [第 10 课](10_diamond_diffusion_world_model.md) 的对照只需要记一句：DIAMOND 的条件里有动作、输出是画面；Diffusion Policy 的条件里有画面、输出是动作。前者能回答「如果我反过来抓」；后者只回答「专家接下来大概怎么抓」。DIAMOND 用 EDM、几步就能交互；本课的 DP 默认 DDPM、训练 100 步扩散、推理默认同样步数。不要把两个仓库的步数抄来抄去。

### 5.4 为什么现在的项目先训模仿

三条工程原因，不是口号。

数据采集便宜。遥操作录 50 条抓放，按官方硬件指南的算法：50 条 × 30 Hz × 30 秒约 45000 帧。`lerobot/svla_so101_pickplace` 实际是 50 条、11939 帧、30 Hz，平均每条约 8 秒。ACT 文档写「大约 50 条示教常常就能看到高成功率」；ALOHA 原论文四项任务在各自 50 条示教上报告过 96%、84%、64%、92%。区间已经很大，本课不得把「50 条」写成万能常数。

优化问题简单。监督 L1 或噪声 MSE，没有奖励设计，没有想象展开，没有 actor-critic。硬件指南把 ACT / VQ-BeT / TD-MPC 归在 Light BC，峰值显存约 2 至 6 GB（batch 8，AdamW）。单卡 24GB 训 ACT，50 条、5 个 epoch，指南给的墙钟是 30 至 60 分钟量级。

动作出口现成。LeRobot 的部署命令 `lerobot-rollout` 直接吃 `--policy.path`。世界模型规划还要在输出端接 MPC 或 CEM，还要处理模型钻空子。桌宠第一周能挥手，靠的是这条捷径。

模仿补不到的那一截，正好是第七幕后半和第八幕要补的：分布外的接触、反事实、安全约束。「别把水推倒」在策略里只能靠「示教里从来没人把水推倒」，在世界模型里可以靠「预测到杯子会过桌沿就停」。

### 5.5 钢琴卷帘：当前帧往后 0.5 秒

把一条轨迹的 6 个关节角画成 6 行随时间变化的色带，横轴是帧，纵轴是关节。这就是钢琴卷帘：每一行是一个「键」，颜色深浅或曲线高度是该关节此刻的目标位置。

30 Hz 下，0.5 秒是 15 帧。你拖动「当前帧」$t$，高亮 $[t, t+15)$。ACT 在 $t$ 时刻实际上预测了远长于 15 帧的一块（默认 100 步），卷帘上那 0.5 秒只是人眼能对齐画面的窗口。Diffusion Policy 默认 horizon 64、执行 32，10 Hz 的 PushT 上 0.5 秒是 5 帧。

这一眼能看出三件事。专家在接触前后会不会把某几个关节一起停住；夹爪通道是不是在接触瞬间才合拢；模型预测的未来 0.5 秒是贴着专家，还是提前合拢、或在两种抓法之间抖动。

世界模型的对照在同一张图上只能先画成示意：模仿卷帘回答「接下来 0.5 秒专家会怎么动」；世界模型要另开一条「如果我现在把夹爪通道反号，画面里的杯子会怎样」。后一条本课不做实验，第 26 课之后才会变成真的前向模拟。本课验收只要求你能指着卷帘说出两者问的问题不一样。

### 5.6 验收这条原理的最小实验

不用真机。加载一条公开抓放，确认 6 维动作和两路相机对得上时间戳。用均值动作当基线：每个关节预测训练集上该关节的平均位置。训完的 ACT 在留出片段上的 L1 必须低于这条基线。卷帘上，接触瞬间的夹爪通道要和专家同一拍合拢，不允许提前 0.5 秒就夹死，也不允许接触后还张着。

这条协议量的是「策略是否在拟合示教」，不是「真机成功率」。真机成功率是有手臂的人加做的，不进复现 #8 的及格线。

## 6. 源码导读

克隆后先看 `src/lerobot/`。当前仓库把策略、数据、脚本都收在这个包里。带问题读，不要按文件名从头扫。

| 文件 | 零件 | 带着什么问题读 |
|---|---|---|
| `src/lerobot/policies/act/configuration_act.py` | ACT 超参 | `chunk_size`、`n_action_steps`、`n_decoder_layers`、`kl_weight` 的默认值是多少，注释如何解释解码器层数 |
| `src/lerobot/policies/act/modeling_act.py` | ACT 本体 | `ACTPolicy.select_action` 的队列；`forward` 的 L1 和 KL；推理时 $z$ 在哪一行变成零 |
| `src/lerobot/policies/act/processor_act.py` | ACT 预处理 | 图像和状态怎样被规范到网络入口 |
| `src/lerobot/policies/diffusion/configuration_diffusion.py` | DP 超参 | `horizon=64`、`n_action_steps=32`、`n_obs_steps=2` 是否满足不等式 |
| `src/lerobot/policies/diffusion/modeling_diffusion.py` | DP 本体 | `compute_loss` 预测的是噪声还是样本；`generate_actions` 从哪一拍切到哪一拍 |
| `src/lerobot/policies/factory.py` | 注册表 | `--policy.type=act` 和 `diffusion` 怎样变成类 |
| `src/lerobot/datasets/` | 数据 | `LeRobotDataset` 按帧索引还是按条索引；`episodes=[0]` 做什么 |
| `src/lerobot/scripts/lerobot_train.py` | 训练入口 | CLI `lerobot-train` 读哪些嵌套配置 |
| `src/lerobot/configs/` 里的训练配置 | 预算 | 默认 `steps=100000`、`batch_size=8`、`save_freq=20000`；`dataset.eval_split` 默认是 0 |

读 ACT 时按调用链走三跳。

第一跳是 `ACTPolicy.select_action`。它在 `eval()` 下工作。若开了时间集成，每步都调用 `predict_action_chunk` 再交给 `ACTTemporalEnsembler.update`。默认没开，就往 `_action_queue` 里倒动作：队列空时取整块的前 `n_action_steps` 步，转置成「时间在前」再 `popleft`。这就是 5.2 的队列。

第二跳是 `ACTPolicy.forward`。多路图像被收成 `OBS_IMAGES` 列表，送进 `ACT`。返回的 `actions_hat` 和示教做带 `action_is_pad` 掩码的 L1；若 `use_vae`，再加上对标准正态的 KL，权重 `kl_weight`。把 `action_is_pad` 漏掉，轨迹末端的填充会污染损失。

第三跳是 `ACT.forward` 的潜变量分支。训练且提供了动作时，VAE 编码器的 token 顺序是 `[cls, 关节, 动作序列]`，cls 的输出投影成 $(\mu, 2\log\sigma)$，再参数化采样 $z$。否则 $z$ 是零向量。之后 $z$、关节、环境状态、图像特征图一起进 Transformer 编码器，解码器的 `chunk_size` 个查询经 `action_head` 变成关节维。

读 Diffusion Policy 时盯两个函数。

`DiffusionModel.compute_loss`：从 batch 取 `action` 当作干净轨迹，抽 $\epsilon$ 和 $\tau$，`noise_scheduler.add_noise` 得到带噪轨迹，U-Net 输出 `pred`。`prediction_type=="epsilon"` 时 target 是 $\epsilon$。这是标准 DDPM 训练，只是数据在动作空间。

`DiffusionModel.generate_actions`：先 `_prepare_global_conditioning` 得到 $(B, \cdot)$ 的条件，再 `conditional_sample` 从噪声走到干净轨迹。切片 `start = n_obs_steps - 1`，`end = start + n_action_steps`。默认 start=1、end=33，也就是 64 步里丢掉「对应当前观察之前」的那一拍，再取 32 步执行。

数据层注意两件事。

其一，`LeRobotDataset` 的 `__getitem__` 是按帧，不是按条。README 示例里 `episode_index=0` 再 `dataset[episode_index]`，取到的是第 0 帧。要整条轨迹，用 `LeRobotDataset(repo_id, episodes=[0])`，再按 `num_frames` 遍历，这是官方 `lerobot-replay` API 示例的写法。

其二，v3 数据集把很多条轨迹打进同一个 Parquet / MP4，靠 `meta/episodes/` 还原边界。`lerobot/svla_so101_pickplace` 的 `meta/info.json` 是 `codebase_version: v3.0`，`robot_type` 写成 `so100_follower`，动作 6 维，名字是 `shoulder_pan.pos` 到 `gripper.pos`，相机键是 `observation.images.up` 和 `observation.images.side`，480×640，30 Hz，50 条、11939 帧。名字带 so101，元数据写 so100：SO-100 和 SO-101 的动作维相同，本课把它当桌面抓放公开集用，不把元数据里的类型字样改写成你手里那台硬件的合格证明。

`lerobot/pusht` 目前仍是 v2.0 布局（一条一个文件），206 条、25650 帧、10 Hz、动作 2 维、画面 96×96，带 `next.reward` / `next.success`。它是 Diffusion Policy 论文的官方任务之一。若当前库加载 v2 失败，先看 [Backward compatibility](https://huggingface.co/docs/lerobot/backwardcomp)，不要手改 Parquet。

预处理不在 `modeling_act.py` 里完成。`processor_act.py` 负责把原始关节和图像变成网络入口的规范化张量，checkpoint 目录里会留下 preprocessor / postprocessor 的 json。离线算均值基线时，我们用 `dataset.meta.stats` 做同样的 MEAN_STD，才能和日志里的 L1 同一量纲。部署时必须成对加载模型和 processor，只 `from_pretrained` 权重、却用未规范化的像素直接 `select_action`，动作会在错误的数值范围里。

工厂与 CLI：`--policy.type=act` 走 `ACTConfig` 注册子类（`@PreTrainedConfig.register_subclass("act")`），训练脚本在 `src/lerobot/scripts/lerobot_train.py`。老文档和部分模型卡还写 `python lerobot/scripts/train.py`，那是 0.5 时代的路径，本课一律用 `lerobot-train`。部署入口从 0.5 的 `lerobot-record` 换成了当前的 `lerobot-rollout`，cheat sheet 写明了这条兼容提示。

## 7. 实验

目标不是复现 ALOHA 论文的真机百分比，而是在公开桌面抓放数据上跑通「回放、训练、离线对照」三件事。后续命令默认在 `lerobot` 仓库根目录、`lerobot` conda 环境已激活。把 `cuda` 换成 `mps` 或 `cpu` 即可改设备。

### Step 1: 建环境并克隆仓库

官方安装页要求 Python 3.12。先装 miniforge（已有 conda 可跳过），再建模环境：

```bash
conda create -y -n lerobot python=3.12
```

```bash
conda activate lerobot
```

Linux 且要用 TorchCodec 解视频时，在该环境里装 ffmpeg：

```bash
conda install ffmpeg -c conda-forge
```

macOS Intel、部分 ARM Linux 会自动退回 pyav，官方写明那时可以跳过 ffmpeg。克隆源码，后面读文件才对得上路径：

```bash
git clone https://github.com/huggingface/lerobot.git
```

```bash
cd lerobot
```

### Step 2: 安装训练与可视化依赖

源码可编辑安装。ACT 在基础包里，不必再装 policy extra。本课需要训练和数据集可视化：

```bash
pip install -e ".[training,dataset_viz]"
```

若你后面要走 Diffusion Policy 和 PushT 仿真，再加（现在不加也不影响 Step 3 到 Step 8）：

```bash
pip install -e ".[diffusion,pusht]"
```

只想用 PyPI、不读源码，官方等价写法是 `pip install 'lerobot[training,dataset_viz]'`。本课源码导读按仓库路径写，建议走源码安装。装完自检：

```bash
lerobot-info
```

预期：打印当前 `lerobot` 版本、PyTorch 设备和可选组件。报找不到命令，说明入口脚本没进 PATH，检查是否装进了刚刚激活的环境。

### Step 3: 加载一条公开桌面抓放

主数据集是 Hugging Face 组织下的 `lerobot/svla_so101_pickplace`：50 条、11939 帧、30 Hz、6 维关节、两路 480×640 相机（`up`、`side`），任务是桌面抓放。第一次运行会下载到 Hub 缓存（大约几十 MB）。

把下面存成 `inspect_so101.py`，在仓库根目录跑。

```python
from lerobot.datasets import LeRobotDataset

repo_id = "lerobot/svla_so101_pickplace"
ds = LeRobotDataset(repo_id, episodes=[0])
print("frames", ds.num_frames, "fps", ds.fps)
print("action names", ds.features["action"]["names"])
print("keys", sorted(ds[0].keys()))
s0 = ds[0]
print("action", tuple(s0["action"].shape))
print("state", tuple(s0["observation.state"].shape))
print("up", tuple(s0["observation.images.up"].shape))
print("side", tuple(s0["observation.images.side"].shape))
```

```bash
python inspect_so101.py
```

预期：`frames` 大约一两百（第 0 条不是全集 11939），`fps` 为 30，动作名六个关节，图像张量是通道在前的 `(3, 480, 640)` 或库当前返回的等价布局。若 `episodes=[0]` 仍给出 11939，说明这次调用按全集索引了，改用 `ds.meta.episodes` 或打印 `episode_index` 列核对；以你这版 `LeRobotDataset` 的 `__len__` 为准，不要和 README 把帧索引叫成 `episode_index` 的示例混为一谈。

备选公开集：README 示例 `lerobot/aloha_mobile_cabinet`（双臂移动柜门，动作维不是 6）；仿真对照 `lerobot/pusht`（2 维、10 Hz、带 `next.success`）。主线用 SO-101 抓放。

### Step 4: 可视化回放第 0 条

不接手臂的回放有两条路，做一条即可。

本地 Rerun 查看器（安装了 `dataset_viz` extra）：

```bash
lerobot-dataset-viz --repo-id=lerobot/svla_so101_pickplace --episode-index=0
```

预期：打开查看器，相机画面和关节时间序列能 scrub。命令参数名以 `--help` 为准；当前文档和 issue 使用 `--repo-id` 与 `--episode-index`，不要写成 `lerobot-train` 那种 `--dataset.repo_id`。

第二条路：打开官方 Space [lerobot/visualize_dataset](https://huggingface.co/spaces/lerobot/visualize_dataset)，粘贴 `lerobot/svla_so101_pickplace`。文档「Visualize a dataset」一节写的就是这条在线路径。

有真机的人可以额外跑 `lerobot-replay`（官方模仿学习教程「Replay an episode」）。它把示教动作打到从臂上，用来测机械重复性，不是本课及格项。不要在没校准的手臂上盲打。

回放时眼睛盯三处：夹爪合拢是不是发生在接触画面出现的那几帧；两条相机的时间戳有没有明显错位；有没有整段静止（那条示教几乎没信息）。把「看得过去的一条」的编号写进 `NOTES.md`。

### Step 5: 钢琴卷帘，拖当前帧看未来 0.5 秒

30 Hz 下 0.5 秒是 15 帧。下面的脚本画出 6 个关节随时间的曲线，滑块改变当前帧，阴影盖住接下来 15 帧。需要 matplotlib：

```bash
pip install matplotlib
```

存成 `piano_roll.py`。

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from lerobot.datasets import LeRobotDataset

repo_id = "lerobot/svla_so101_pickplace"
episode = 0
horizon = 15  # 0.5 s at 30 Hz
ds = LeRobotDataset(repo_id, episodes=[episode])
n = ds.num_frames
names = list(ds.features["action"]["names"])
acts = np.stack([np.asarray(ds[i]["action"]) for i in range(n)], axis=0)
t = np.arange(n) / float(ds.fps)

fig, axes = plt.subplots(len(names), 1, sharex=True, figsize=(10, 8))
spans = []
for i, ax in enumerate(axes):
    ax.plot(t, acts[:, i], color="0.15", linewidth=1.0)
    span = ax.axvspan(0.0, horizon / float(ds.fps), color="0.75", alpha=0.5)
    spans.append(span)
    ax.set_ylabel(names[i].replace(".pos", ""), fontsize=8)
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("time (s)")
fig.suptitle("episode %d  drag t to see next 0.5 s" % episode)
plt.subplots_adjust(bottom=0.12, hspace=0.15)
ax_sl = fig.add_axes([0.15, 0.04, 0.7, 0.03])
slider = Slider(ax_sl, "t (frame)", 0, max(n - 1, 1), valinit=0, valstep=1)

def on_change(val):
    f = int(val)
    t0 = f / float(ds.fps)
    t1 = min(n - 1, f + horizon) / float(ds.fps)
    for span in spans:
        span.set_xy([[t0, 0], [t0, 1], [t1, 1], [t1, 0], [t0, 0]])
    fig.canvas.draw_idle()

slider.on_changed(on_change)
on_change(0)
plt.show()
```

```bash
python piano_roll.py
```

预期：六行曲线，拖滑块时灰色窗口宽度约 0.5 秒。接触附近夹爪那一行应出现一次明显的合拢或张开。把窗口停在合拢前一拍，问自己：若此刻把夹爪通道反号，杯子会怎样。本课没有模型能回答，把这个问题写进对照笔记即可。

读卷帘时按三个时间尺度看。小于 0.1 秒：专家会不会在接触前把腕部停住，这是细动作；0.5 秒窗口：肩、肘、夹爪是否一起朝杯子走，这是动作块该保住的协同；整条 8 秒：有没有「伸手、合拢、抬起、放下、松开」五段，缺一段说明这条示教不适合当主回放。世界模型若在场，它关心的是 0.5 秒窗口里杯子像素会不会过桌沿，而不是肩关节曲线漂不漂亮。

### Step 6: 训练 ACT 小配置

官方 ACT 页和 cheat sheet 的骨架命令如下，本课加上小步数、关掉推送和 wandb。默认 `steps=100000` 对 50 条数据过长。按硬件指南的算法，约 12000 帧、batch 8，一个 epoch 大约 1500 步；8000 步大约 5 个 epoch，和指南「50 条、5 epoch、4090 上 30 至 60 分钟」同一量级。冒烟可先改成 `--steps=2000`。

```bash
lerobot-train \
  --dataset.repo_id=lerobot/svla_so101_pickplace \
  --policy.type=act \
  --output_dir=outputs/train/act_so101_pickplace_smoke \
  --job_name=act_so101_pickplace_smoke \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.push_to_hub=false \
  --steps=8000 \
  --batch_size=8 \
  --save_freq=4000 \
  --dataset.eval_split=0.1 \
  --eval_steps=2000
```

日志里每 `log_freq=200` 步打一次 train `l1_loss`，每 2000 步打一次 eval。ACT 的 `forward` 还可能打印 `kld_loss`。你要看的是 L1 的趋势，不是绝对值漂亮不漂亮：不同规范化、不同任务的 L1 不能横比。开头几百步 L1 接近均值基线很正常；若 2000 步后仍贴着基线不动，再查数据键名和设备。`save_freq=4000` 会在 4000 和 8000 各留一个 checkpoint，目录结构是 `checkpoints/<step>/pretrained_model/`。

Mac 把 `--policy.device=cuda` 换成 `--policy.device=mps`，并把 `--batch_size=8` 改成 `4`。没有本地 GPU 时，官方支持把同一条命令加上 `--job.target=a10g-small` 丢到 Hugging Face Jobs，那是可选项，需要 `hf auth login` 和按秒计费，不进及格线。Colab 笔记本见文档 [Notebooks / Training ACT](https://huggingface.co/docs/lerobot/notebooks)。

预期：日志里 `l1_loss` 下降；每隔 2000 步出现一次留出集上的 eval `l1_loss`（`eval_split=0.1` 在 50 条里留出约 5 条）。checkpoint 写在 `outputs/train/act_so101_pickplace_smoke/checkpoints/`。ACT 没有默认学习率衰减（`get_scheduler_preset` 返回 `None`），缩短步数不必改 scheduler。若你改训 Diffusion Policy，硬件指南要求同时把 `--policy.scheduler_decay_steps` 收到和 `--steps` 同量级。

社区里有人报告过「训练时用了 validation split，部署时机器人不动」。本课只做离线损失，不部署。以后要上真机，用全数据再训一版，不要直接拿这条带 `eval_split` 的权重当唯一部署对象。

### Step 7: 离线评测与均值动作基线

训练日志里的 L1 已经在 `MEAN_STD` 规范化后的动作空间里。均值策略在同一空间里就是永远预测 0，它的 L1 等于规范化动作的平均绝对偏差。把下面存成 `mean_baseline.py`。

```python
import numpy as np
from lerobot.datasets import LeRobotDataset

repo_id = "lerobot/svla_so101_pickplace"
ds = LeRobotDataset(repo_id)
stats = ds.meta.stats["action"]
mean = np.asarray(stats["mean"], dtype=np.float32)
std = np.asarray(stats["std"], dtype=np.float32)
std = np.where(std < 1e-6, 1.0, std)

n = min(len(ds), 4000)
absdev = []
for i in range(n):
    a = np.asarray(ds[i]["action"], dtype=np.float32)
    z = (a - mean) / std
    absdev.append(np.abs(z).mean())
print("n", n, "mean-policy L1 (normalized)", float(np.mean(absdev)))
```

```bash
python mean_baseline.py
```

把打印出来的数写进 `NOTES.md`，再和训练日志末尾的 train / eval `l1_loss` 并排。方向性通过：训练结束时的 eval L1 明显低于这条均值基线，并且低于训练开头的 L1。单点损失没有资格当复现结论，报的时候写步数、batch、是否 `eval_split`。

这一步不证明真机抓放成功率。它证明网络在拟合这 50 条示教，而不是输出数据集均值。

### Step 8: （可选）Diffusion Policy 小配置或 PushT 仿真

想对照 5.3，在同一套 LeRobot 命令里把 `--policy.type=act` 换成 `--policy.type=diffusion`，并确保已装 `diffusion` extra。DP 更吃显存（指南 8 至 14 GB），墙钟大约是 ACT 的数倍。记得 `--ema.enable=true`，评测加载 `pretrained_model_ema`。

想看带成功率的仿真，改用 `lerobot/pusht` 和 `--env.type=pusht`。老模型卡 `lerobot/diffusion_pusht` 仍写 `python lerobot/scripts/train.py`；当前入口是 `lerobot-train` 与 `lerobot-eval`。官方 README 的评测示例是：

```bash
lerobot-eval \
  --policy.path=lerobot/pi0_libero_finetuned \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=10
```

那是 LIBERO 上的 VLA 例子，不是本课必跑。PushT 的 `env.type` 以 `lerobot-eval --help` 和当前 `src/lerobot/envs/` 为准，不要从 2024 年的博客抄。本课及格不依赖这一步。

### Step 9: 填对照表并留证据

在实验目录写 `NOTES.md`，至少六行：

```text
日期与机器（cuda / mps / cpu，显存）
仓库 git rev-parse HEAD 与 pip 里的 lerobot 版本
数据集 repo id、条数、fps、动作维
完整训练命令
均值基线 L1 与最终 train/eval L1（含步数）
对照表：模仿补了什么，世界模型还缺什么
```

对照表模板见第 9 节。三个月后只看这份笔记，要能复述你训的是策略还是世界模型。

## 8. 配置与预算

数字来自 LeRobot 当前文档和源码默认值。墙钟是指南里的量级，不是你机器上的 SLA。

| 档位 | 数据 | 策略与步数 | 显存 | 墙钟（量级） | 用途 |
|---|---|---|---|---|---|
| 回放档 | `svla_so101_pickplace` 第 0 条 | 不训练 | CPU / Mac 即可 | 十几分钟（含下载） | 看懂示教，画钢琴卷帘 |
| 冒烟档 | 同上全集 50 条 | ACT，2000 步，batch 8 | 2 至 6 GB | 十几到几十分钟 | 确认损失在降 |
| 本课小配置 | 同上 | ACT，8000 步，batch 8，`eval_split=0.1` | 2 至 6 GB | 24GB 卡约 0.5 至 2 小时；MPS 数小时 | 方向性复现 #8 |
| 指南 5 epoch | 约 50 条、640×480 | ACT batch 8 | 同上 | 4090 / 3090 约 30 至 60 分钟 | 对照官方硬件指南 |
| DP 选做 | 同一集或 `lerobot/pusht` | `diffusion`，步数自定，开 EMA | 8 至 14 GB | 约为 ACT 的数倍 | 对照 5.3 |
| 论文原配置 | ALOHA 双臂、每任务约 50 条示教 | 原 ACT 仓库，50 Hz | 以论文为准 | 本课不要求 | 只读 |

ACT 源码默认：`chunk_size=100`，`n_action_steps=100`，`dim_model=512`，`n_encoder_layers=4`，`n_decoder_layers=1`，`use_vae=True`，`latent_dim=32`，`kl_weight=10.0`，`vision_backbone=resnet18`，学习率 `1e-5`。文档称大约 8000 万参数。训练预设 `steps=100000`、`batch_size=8`、`save_freq=20000`，本课小配置必须改步数和存盘间隔。

Diffusion Policy 源码默认：`horizon=64`，`n_action_steps=32`，`n_obs_steps=2`，`num_train_timesteps=100`，`noise_scheduler_type=DDPM`，`prediction_type=epsilon`，学习率 `1e-4`。

磁盘：数据集缓存很小；checkpoint 按步数涨，预留 5GB 足够。CPU 训练官方硬件指南写「不要训，用 Colab 或租卡」。Mac MPS 可以训 ACT，指南给 5 epoch、batch 4、约 6 至 14 小时。

有 SO-101 的人额外预算在组装和校准，不计入本课墙钟。

## 9. 验收

- [ ] 能在白纸上画出 $\pi(a_{t:t+k} \mid o_t)$ 和 $P(s_{t+1} \mid s_t, a_t)$ 的输入输出，并各举一个桌宠行为。
- [ ] 回放过 `lerobot/svla_so101_pickplace` 至少一条，能指出接触瞬间夹爪通道的变化。
- [ ] 钢琴卷帘能拖动当前帧，阴影覆盖约 0.5 秒，能口头对比「模仿问什么、世界模型问什么」。
- [ ] ACT 或 DP 小配置跑完，日志里的 L1 相对训练初期下降；eval L1（若开了 `eval_split`）低于 `mean_baseline.py` 打出的均值策略 L1。
- [ ] `NOTES.md` 含命令、版本、数据、两个 L1 数字和对照表。
- [ ] 对照表至少覆盖：学什么、数据效率、泛化、失败模式、桌宠用途。下面是空白模板，用你的实验填右两列的「本课观察到」；机制列可以先抄，观察列必须来自你的回放和曲线。

| 维度 | 模仿策略（ACT / DP） | 世界模型规划 |
|---|---|---|
| 学的对象 | $\pi(a \mid o)$，专家动作 | $P(s' \mid s, a)$，下一状态 |
| 输入 | 相机 + 关节（ACT 无语言） | 状态 + 候选动作 |
| 输出 | 一截未来动作 | 未来状态（再拿去打分选动作） |
| 数据 | 示教轨迹，几十条就能像样 | 要覆盖动作条件，通常更贪心 |
| 泛化 | 布局一变就复合误差 | 见过的动力学可以迁移到新目标 |
| 失败 | 平均掉多峰、分布外瞎编专家动作 | 模型钻空子、长程漂移 |
| 反事实 | 不会回答「反过来抓会怎样」 | 正是它的接口 |
| 桌宠 | 挥手、点头、把笔推正 | 别把水推倒、别扫落杯子 |
| 本课证据 | 回放 + 损失下降 | 第 24 课规划或本课示意 |

- [ ] 能向别人讲清：VLA 有语言指令和预训练视觉-语言模型，ACT 没有；二者都吐动作，不是同一种模型。
- [ ] 没有把「我没买手臂」写成实验失败。

## 10. 排错

| 症状 | 最可能的原因 | 怎么确认 | 怎么修 |
|---|---|---|---|
| `lerobot-info` 找不到 | 装进了别的环境，或只用了基础 `pip install lerobot` 却期望 CLI 全套 | `which lerobot-info`，`conda env list` | 激活 `lerobot` 环境，按 Step 2 重装 extras |
| 编译 / `cmake` / ffmpeg 报错 | 系统缺头文件，或 ffmpeg 与 torchcodec 版本打架 | 日志里搜 `avformat`、`libsvtav1` | 官方 Installation 的 `apt-get` 列表；conda 可钉 `ffmpeg=7.1.1` |
| 数据集下载停住或权限错 | Hub 网络，或把公开集写成了私有 repo | 浏览器打开数据集页 | 换网络；确认 repo id 是 `lerobot/svla_so101_pickplace` |
| `dataset[0]` 图像键报 KeyError | 键名不是 `up` / `side` | 打印 `ds.features` | 以该数据集 `meta/info.json` 为准，不要抄别的任务 |
| `lerobot-dataset-viz` 无窗口 | 缺 `rerun-sdk`，或远程无显示 | `pip show rerun-sdk` | 重装 `.[dataset_viz]`；或改走在线 Space |
| 训练立刻要 `repo_id` / 推送失败 | 默认 `push_to_hub` 为真且没给模型仓库 | 报错原文 | `--policy.push_to_hub=false` |
| 损失不降或 NaN | 步数太少还没看见趋势，或学习率被改乱 | 看前 200 步是否有下降趋势 | 冒烟先看到下降再加到 8000；ACT 保持默认 `1e-5` |
| eval 报需要 `eval_split` | `--eval_steps>0` 但留出比例仍是 0 | 对照 `TrainPipelineConfig` 校验 | 加上 `--dataset.eval_split=0.1` |
| 显存不够 | batch 太大，或误开了 VLA | `nvidia-smi` | ACT 把 batch 降到 4 或 2；不要在本课加载 $\pi_0$ |
| MPS 慢或不稳 | 指南就写 MPS 上 ACT 要数小时 | 墙钟对比指南 6 至 14 小时 | 缩小 `--steps`；或 Jobs / Colab |
| 加载 `lerobot/pusht` 失败 | 该集仍是 v2.0 布局 | `meta/info.json` 的 `codebase_version` | 读 backward compatibility 页，不要手改文件 |
| 真机 replay 不动或乱甩 | 未校准、`id` 不一致、限位 | 先 `lerobot-calibrate` | 本课可跳过真机；校准后再 replay |
| 把 ACT 当 VLA 去加 prompt | 概念混了 | 训练命令里有没有语言相关 flag | 删掉。ACT 页写明 `--task` 可跳过 |

## 11. 前沿与改造

2024 到 2026 年，公开系统把「一次吐一截动作」留了下来，换的是生成器和条件。Physical Intelligence 的 $\pi_0$ 用 flow matching 生成连续动作块，条件里多了语言；LeRobot 文档把它和 SmolVLA、$\pi_{0.5}$ 列在 VLA，不列在 ACT 旁边。Real-Time Chunking（RTC）解决的是「动作块还没执行完、观察已经变了」时如何平滑衔接，部署页 `lerobot-rollout --inference.type=rtc` 面向的是这类较慢的大策略。这些工作默认你已经接受 5.2 的动作块，不默认你有动力学。

同一仓库里也开始出现世界模型条目：README 把 VLA-JEPA、LingBot-VA、FastWAM 单独列成 World Models。那是「观察加动作预测未来」这条线，和本课练的策略不是同一张计算图。本课不训它们。第 26 课只比较 VLA 和世界模型的接口，仍然不会把 ACT 并进去。

缩小版和前沿版的差距，一半是规模：示教从 50 条变成跨机器人、跨场景的大混合；骨干从 ResNet-18 变成预训练视觉-语言模型。钱和数据能缩小这一半。另一半是机制：动作块之间的实时衔接（RTC）、语言条件（VLA）、以及把动力学接在动作出口之前当过滤器。后两件是第 26 课和第 30 课的零件，本课只要求你把过滤器该接在哪画出来。

动手改造（选做，每个只改一个旋钮）：

1. 动作块长度。在 `ACTConfig` 默认 100 的基础上，跑 `--policy.chunk_size=50` 和 `--policy.n_action_steps=50`，再跑一组 100。数据仍是 `svla_so101_pickplace`，步数 8000，batch 8。预期：50 步的 eval L1 不一定更好，钢琴卷帘上接触附近可能更跟手（更勤地看新观察），也可能在块边界出现一次关节跳变。失败判据：两组 eval L1 都高于均值基线，说明改块长之前网络还没拟合示教，先退回默认。
2. 关掉 VAE。`--policy.use_vae=false`，其余同小配置。预期：训练更像纯 L1 回归，日志不再有 `kld_loss`；若示教多峰（同一抓放有两种绕法），eval L1 可能看起来不错，但卷帘在接触点出现「介于两种夹爪开度之间」的中间值。失败判据：关 VAE 之后 eval L1 爆炸，先检查是否有键名或维度报错，而不是下「VAE 无用」的结论。
3. 数据量。用 `--dataset.episodes` 只喂前 10 条，对照 50 条，步数都用 8000。预期：10 条组的训练 L1 可以降得很低（记熟了），留出条上的 eval L1 明显高于 50 条组。这是在缩小预算上走 ALOHA 论文「约 50 条示教」那个方向，不是在复现他们的真机百分比。失败判据：两组 eval 分不出差别，先确认 10 条组真的只看到了 10 条（打印 `num_episodes`）。
4. 同一数据上的 ACT vs Diffusion Policy。DP 开 `--ema.enable=true`，步数可减到 4000 以免墙钟翻倍。预期：两条曲线都能低于均值基线；DP 更慢、显存更高。不要在失败时宣称「DP 更强」或「ACT 更强」，样本量和超参都不够给路线排名。失败判据：DP 装不上 `diffusers`（漏了 `.[diffusion]`），或 EMA 没开却去加载 `pretrained_model_ema`。

顺手复现映射：

| 论文结论 | 本课缩小实验 | 预期 |
|---|---|---|
| ACT：动作分块减轻复合误差，约 50 条示教可学细操作 | 改造 1 和 3 | 能看到块长和条数影响 eval L1 的方向；不能得到 80% 至 90% 真机成功率 |
| ACT：推理时 $z=0$ | 读 `ACT.forward` 的 else 分支 | 能在代码里指出来即通过 |
| Diffusion Policy：动作空间扩散能表达多峰、训练稳 | 选做 DP，损失下降且低于均值基线 | 方向性；12 任务平均 +46.9% 不能在本课复现 |
| IBC：隐式能量模型优于显式回归 | 本课不实现 EBM | 只读，用来理解 DP 为何不走 L1 平均 |

## 12. 论文与延伸

1. Chi et al.，Diffusion Policy: Visuomotor Policy Learning via Action Diffusion，arXiv:[2303.04137](https://arxiv.org/abs/2303.04137)。带着三个问题读：动作扩散的条件是观察而不是「上一动作加状态」意味着它还是不是世界模型？论文如何用收缩视野同时要长 horizon 和短执行？46.9% 的平均提升是相对哪些基线、在哪些任务上平均的？
2. Zhao, Kumar, Levine, Finn，Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware，arXiv:[2304.13705](https://arxiv.org/abs/2304.13705)。带着三个问题读：动作分块解决的是抖动还是复合误差，还是两者？VAE 的 $z$ 在推理时为什么可以扔掉？50 条示教在四个任务上成功率从 64% 到 96%，这个区间对「ACT 很数据高效」这句话意味着什么？原实现 [tonyzhaozh/act](https://github.com/tonyzhaozh/act) 和项目页 [tonyzhaozh.github.io/aloha](https://tonyzhaozh.github.io/aloha) 可对照。LeRobot 把解码器层数写成 1 以对齐该实现的已知行为，读 issue 25。
3. Fu, Zhao, Finn，Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation，arXiv:[2401.02117](https://arxiv.org/abs/2401.02117)。带着两个问题读：静态 ALOHA 数据和移动任务共训，成功率为何能升？这还是纯模仿，还是已经需要世界模型？本课的 SO-101 桌面抓放相当于他们的「桌面那一段」，没有底盘。
4. Florence et al.，Implicit Behavioral Cloning，arXiv:[2109.00137](https://arxiv.org/abs/2109.00137)。带着两个问题读：显式回归为什么会在多峰动作上失败？能量模型、ACT 的 VAE、Diffusion Policy 的去噪，三者各自怎样避免「平均出非法动作」？本课不实现 IBC，它是 5.3 的理论前史。
5. LeRobot 文档以当天页面为准：[Installation](https://huggingface.co/docs/lerobot/installation)、[ACT](https://huggingface.co/docs/lerobot/act)、[Imitation Learning](https://huggingface.co/docs/lerobot/il_robots)、[Dataset v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)、[Compute Hardware Guide](https://huggingface.co/docs/lerobot/hardware_guide)、[SO-101](https://huggingface.co/docs/lerobot/en/so101)。仓库 README 的训练示例是 `lerobot-train --policy.type=act --dataset.repo_id=lerobot/aloha_mobile_cabinet`。原版 Diffusion Policy 代码在 [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy)，项目页 [diffusion-policy.cs.columbia.edu](https://diffusion-policy.cs.columbia.edu)。
6. 选读：第 24 课将精读的 Visual Foresight（Finn & Levine，arXiv:[1812.00568](https://arxiv.org/abs/1812.00568)）和 DayDreamer（Wu, Escontrela, Hafner et al.，arXiv:[2206.14176](https://arxiv.org/abs/2206.14176)）。读的时候只问一句：它们的模型输出是未来状态还是未来动作？若是状态，它和本课的策略差在哪个接口。[第 10 课](10_diamond_diffusion_world_model.md) 的 DIAMOND 是像素世界模型，不要和 Diffusion Policy 共用同一张「扩散」标签完事。[第 19 课](19_surgery_experiments.md) 的手术单写法本课改造实验直接沿用：一次只改一个旋钮，写预算和失败判据。

现在系统里多了一只会模仿的手：观察进，动作块出。下一课要把语言接进这条出口，并写清 VLA 成功不等于理解物理；世界模型仍负责「这么动会不会闯祸」那一截。桌宠的挥手可以交给本课的策略，倒水前的停手还不能。
