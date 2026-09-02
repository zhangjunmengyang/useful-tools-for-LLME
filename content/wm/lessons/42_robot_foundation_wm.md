---
id: 42_robot_foundation_wm
title: "机器人基础世界模型 vs VLA"
summary: "Reasoner、Generator、Policy 三套头，哪一个才是 P(s'|s,a)？"
unit: frontier
play_tools: []
checkpoints:
  - "三列表：理解头、生成头、动作头。"
  - "桌宠最小接入：策略提议，世界模型过滤扫杯。"
---

# 第 42 课：拆开机器人基础模型上的理解、生成和动作三个头

> 类型：研究（对照公开报告与仓库接口，不新训大模型）<br>
> 建议周期：1-2 天<br>
> 硬件：主线不需要 GPU；Mac / 纯 CPU 可完成三列表、广告词阅读笔记和桌宠过滤器设计。单张 24GB 可选做第 26 课已写过的 OpenVLA 推理，或按 [nvlabs/cosmos-policy](https://github.com/nvlabs/cosmos-policy) README 做 Cosmos Policy 2B 推理（仓库写 LIBERO 约 6.8GB）。Cosmos 3 的 16B / 64B 标成只讲<br>
> 锚定仓库：[nvidia/cosmos](https://github.com/nvidia/cosmos)（Cosmos 3 文档与 cookbook），对照 [openvla/openvla](https://github.com/openvla/openvla)、[Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)、[gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm)、[nvlabs/cosmos-policy](https://github.com/nvlabs/cosmos-policy)<br>
> 产物：一张理解头 / 生成头 / 动作头三列表、一段「广告词里的 world model 对应哪一头」阅读笔记、一份桌宠最小接入草稿（ACT 或 VLA 提议动作，第 30 课模型过滤扫杯）

## 1. 这一课做什么

整门课的主干循环没变：

```text
观察 先压成状态 再按动作预测下一状态 然后展开多条未来 给未来打分 最后选动作
```

九幕里你现在在最后一幕，读榜和选零件，不改毕业标准：

| 幕 | 课 | 一句话 |
|---|---|---|
| 第一幕：第一个世界模型 | 01-04 | 在 CarRacing 上复现 World Models |
| 第二幕：潜空间方法 | 05-08 | RSSM、Dreamer、MuZero、TD-MPC2 |
| 第三幕：生成式世界模型 | 09-12 | token、扩散、latent action |
| 第四幕：JEPA 路线 | 13-16 | 不生成像素只预测表征 |
| 第五幕：评测与改造 | 17-19 | 评测、缩放、消融、改结构 |
| 第六幕：空间与物体 | 20-23 | 持久 3D/4D、视频基础模型、物体中心 |
| 第七幕：接到身体上 | 24-27，理论 33 | 视觉预测控制、模仿、VLA；E0 到 E5 |
| 第八幕：桌宠毕业设计 | 28-32 | 会看、会想、会克制的小机器人 |
| 第九幕：刷榜、声音与配方 | 34-45 | 读 2025-2026 系统，24GB 能跑的才动手 |

毕业标准仍在[第 32 课](32_ship_desk_pet.md)的五件行为和[第 33 课](33_embodiment_degrees.md)的 E0 到 E5。第 41 课刚把驾驶世界模型的条件拆成动作、外生事件和风格；验证者是路，不是桌子。这一课换到机器人基础模型：同一套权重上常常同时挂着理解、生成和动作三个头，广告词一律叫 world model。要会拆。

[第 26 课](26_vla_vs_world_model.md) 已经钉过一次分工。VLA（视觉-语言-动作模型）学

$$
\pi(a \mid o, \text{instruction})
$$

世界模型学

$$
P(s_{t+1} \mid s_t, a_t)
$$

本课不把那一课的 256 箱、LIBERO 四套件表和 $\pi_0$ 的 10 步欧拉再抄一遍。升级点是 2026 年的基础模型清单：Cosmos 3 用同一套 Mixture-of-Transformers（MoT：按通路分参数的 Transformer，Liang 等，arXiv:2411.04996）同时跑 Reasoner 和 Generator，动作既可以当条件，也可以当要去噪的目标；Cosmos Policy 把更早的 Cosmos-Predict2 后训练成同时吐动作、未来状态和价值的策略。焊在一起之后，三个头的接口仍然不同。桌宠要问「伸手拿杯，杯子会不会倒」，只有前向模型这一头答得上。

[第 33 课](33_embodiment_degrees.md) 的尺子原样生效。VLA、[第 25 课](25_lerobot_imitation.md) 的 ACT、以及只出动作的 Cosmos 后训练检查点，默认是 E3：真机上可以很成功，选动作时不查询前向模型。查询了 $P(s_{t+1}\mid s_t,a_t)$，日志里看得到这次查询改写了动作，才进 E4。RoboArena 上拿过第一，不自动把你家书桌升到 E4，也不证明扫杯过滤器已经可迁移。

本课不新训。24GB 主线假设下，16B / 64B 的 Cosmos 3 不能装成你训过；作业是读文档、填三列表、把三个头接到「伸手拿杯」上，再写一段第 30 课小模型当过滤器的胶水。毕业设计可以继续用那只小模型，不必换上 64B。

术语速查：

| 术语 | 一句人话 |
|---|---|
| 理解头 / Reasoner | 吃图和字，吐字：说明桌上有什么、指令是什么、下一步计划用哪句话 |
| 生成头 / Generator | 吃条件，吐画面、声音或未来表征：把「接下来长什么样」播出来 |
| 动作头 / Policy | 吃观察和指令，吐关节命令或动作块：回答现在该怎么动 |
| 前向动力学 | 给定当前观察和打算执行的动作，预测下一观察；本课要找的 $P(s_{t+1}\mid s_t,a_t)$ |
| 逆动力学 | 给定一段画面变化，反推中间发生过什么动作；更接近策略，不是世界模型 |
| 世界-动作模型 | 同一套生成器既能播未来视频，也能吐动作；配置不同，职责不同 |
| MoT | Mixture-of-Transformers：Reasoner 通路和 Generator 通路各有一套层参数，在注意力里会合 |
| Cosmos Policy | 把 Cosmos-Predict2-2B 后训练成同时生成动作、未来状态和价值的策略（Kim 等，arXiv:2601.16163） |
| RoboArena | 社区在真机 DROID 上做双盲两两比较的策略榜；量的是规划好，不是预测准 |
| E3 / E4 | 第 33 课档位：开环策略停在 E3，查询前向模型并改写动作才进 E4 |

## 2. 问题

2026 年的机器人基础模型喜欢把三件事焊进同一张图。一张图读起来很省事，用起来会把接口用错。本课要同时堵住四条交白卷的路。

第一，把「会回答杯子在哪」当成「会预测杯子倒不倒」。Reasoner 看一张图，用中文写出「杯子在桌沿，伸手可能碰到」。这是文本理解。它没有一条你可以替换 $a_t$、再前向一次、得到两段分岔未来的计算图。[第 03 课](03_mdn_rnn_action_conditioned.md) 的动作对换对它没有接口。

第二，把「会播未来视频」当成已经有动力学。Generator 可以做文本生成视频、首帧生成视频，画面可以极真。[第 12 课](12_frontier_landscape.md) 的三问里，动作起不起作用、有没有人用它选过动作，这两问仍然可能全否。只有把低层动作当成条件、并且对换分岔，这一头才是前向模型。

第三，把「会出动作」当成已经在想完再动。OpenVLA 的 `predict_action`、openpi 的 `policy.infer(...)["actions"]`、Cosmos 3 后训练出来的 Policy 检查点，返回值都是动作。任务成功率是第 17 课的规划好。规划好很高，前向模型仍可以缺席。第 26 课已经在 LIBERO 上演示过这件事；本课要你把同一把尺子伸到 Cosmos 3 和 RoboArena 的广告词上。

第四，把榜上的第一写成桌宠已经可迁移。Cosmos 3 技术报告写过：后训练模型在报告撰写时被 RoboArena 列为当时最好的策略模型，图 26 的截图日期是 2026-05-30。榜会变，身体是 DROID 上的 Franka，任务分布不是你家 60 cm × 40 cm 的桌子。本课禁止把那一格第一抄进第 32 课的毕业日志。

档位先写死。精读 Cosmos 3 报告、填三列表、设计过滤器，是研究。跑 OpenVLA 7B 或 Cosmos Policy 2B 的官方推理，是可选体验。从头训 Cosmos 3、复现 RoboArena、复现 $\pi_{0.5}$ 共训配方，本课不做。

## 3. 准备

- [第 26 课](26_vla_vs_world_model.md) 的接口对照表在手边：VLA 吐动作，世界模型吐未来。本课在那张表上加第三列「理解头」，并换成 2026 年的基础模型行。
- [第 33 课](33_embodiment_degrees.md) 的七条是否题能口头复述。本课会给 Cosmos 3 的三种配置、$\pi_0$、OpenVLA、DINO-WM、第 30 课桌面模型各打一档。
- [第 30 课](30_desk_world_model.md) 若已经留下 `runs/desk1/best.pt` 和 `desk_wm.py` 的 `rollout`，本课过滤器直接调用它。没训过也可以读接口，用纸面草稿验收。
- [第 17 课](17_evaluating_world_models.md) 的三分评测：预测准、生成真、规划好，分开写。RoboArena 只进规划好这一列。
- [第 22 课](22_foundation_video_wm.md) 的资格审查还在：Cosmos-Predict2.5 是视频基础模型，不是全模态（omnimodal：同一套权重按配置切换语言、图像、视频、声音、动作）；本课的 Cosmos 3 才把这些模态放进同一套 MoT。
- 一台能打开 GitHub 和 arXiv 的机器。要 clone 对照仓库，不要求装齐推理依赖。磁盘预留 2GB 给五个浅克隆足够；不下载 16B / 64B 权重。
- 不需要真机，不需要 8 卡，不需要 RoboArena 的 DROID 平台。有第 25 课 ACT 检查点或第 26 课 OpenVLA 环境的人，可以把过滤器接到已有动作出口上，那是加分，不是及格线。

## 4. 学习目标

1. 在白纸上画出三个头：理解头吃什么吐什么，生成头在什么配置下才是 $P(s_{t+1}\mid s_t,a_t)$，动作头为什么可以没有未来分支。
2. 对照 Cosmos 3 报告第 2 节和仓库 README 的 Reasoner / Generator 表，说出两条通路的注意力差在哪，以及前向动力学、逆动力学、策略三种动作模式各自干净的是哪一段 token。
3. 说出 Cosmos Policy 和 Cosmos3-Nano-Policy-DROID 不是同一个检查点：前者后训练的是 Predict2-2B，后者后训练的是 Cosmos 3 Nano；两者都更接近策略，只有显式查询未来状态时才碰到前向模型。
4. 给 Cosmos 3、$\pi_0$、OpenVLA、DINO-WM、第 30 课桌面模型各填一格理解 / 生成 / 动作，空格写「无」，不许写「差不多」。
5. 把一句基础模型广告词拆开：world model 三个字落在哪一头，第 17 课的哪根尺子，第 33 课的哪一档。
6. 写一份桌宠最小接入：ACT 或 VLA 出候选，$H$ 步调用第 30 课 `rollout`，预测扫杯则改成停下。不新训大模型。
7. 给「伸手拿杯」接三次线，明确指出谁在回答「杯子会不会倒」。

## 5. 原理

六个机制。每个都落到同一句验收语：三个头焊在一起，并不等于三个问题被同一个张量回答。

### 5.1 三个头，三句不同的话

把「伸手拿杯」拆成三句，接口立刻分开。

理解头回答：画面里哪一个是杯子，人说的「拿」是抓还是推，下一步语言计划是什么。形式是

$$
\text{text} = R_\theta(I_t, \ell)
$$

$I_t$ 是图像或短视频，$\ell$ 是指令或问题，$R_\theta$ 输出 token。Cosmos 3 的 Reasoner、普通 VLM、GR00T 里读图读字的那一段，都落在这里。它可以被训得很会说物理，仍然没有动作对换。

生成头回答：接下来画面或表征长什么样。最弱的形式是无动作视频生成 $P(\text{video}\mid \text{text})$，第 12 课把它记成 E0。加上动作条件之后，才变成

$$
P(o_{t+1:t+H} \mid o_{\leq t}, a_{t:t+H-1})
$$

这是前向动力学，也是本课唯一承认的世界模型头。DINO-WM 的 ViT 预测器、第 30 课的 `rollout`、Cosmos 3 Generator 的 `forward_dynamics` 模式，落在这里。

动作头回答：现在该怎么动。

$$
a_{t:t+H-1} = \pi_\theta(o_t, \ell)
$$

OpenVLA、$\pi_0$、ACT、Cosmos 3 的 `policy` 模式、Cosmos Policy 直接部署时用的动作块，都是这一头。它可以顺带吐一段未来视频当辅助目标，那是训练技巧；部署时如果把未来丢掉、只执行 $a$，回路里就没有查询。

类比：理解头是解说员，生成头是能换镜头的预演，动作头是伸手的学徒。解说员说「杯子会倒」和预演里杯子真的过沿，不是同一条证据。类比失效处：2026 年的大模型常把三件事放进同一套权重，换一个输入输出配置就换一种产品。本课要你看的是配置和返回值，不是模型家族名。

验证方法本课就能做：打开源码或 cookbook，看返回值字段。没有 `next_state` / 未来视频 / 特征 $\hat{z}$，就没有前向模型可查询。

### 5.2 Cosmos 3：一条 MoT，两种注意力，三种动作模式

Cosmos 3（NVIDIA，arXiv:2606.02800，本课按 v4，2026-06-23）自称 omnimodal world model：语言、图像、视频、声音、动作进同一套 MoT。仓库 README 把它暴露成两个运行面：

| 运行面 | 输入 | 输出 | 官方举例 |
|---|---|---|---|
| Reasoner | 文本、视觉 | 文本 | 理解、框出物体、物理判断、任务规划、动作预报（文字） |
| Generator | 文本、视觉、声音、动作 | 视觉、声音、动作 | 世界生成、世界模拟、未来预测、合成数据、策略学习 |

架构要点在报告第 2 节和 `cookbooks/cosmos3/cosmos3-model-architecture.png`。每个 decoder 层有两套参数。Reasoner 通路吃自回归子序列（语言，以及 ViT 编码的理解用视觉 token），因果自注意力，做下一个 token。Generator 通路吃扩散子序列（VAE 编码的图像 / 视频、音频、动作），双向注意力，用 flow matching 去噪。Generator 的 query 可以看全部 Reasoner 的 key / value；Reasoner 不看 Generator。两条通路共享 3D mRoPE，用来把视频、音频、动作对齐到同一条物理时间轴。

这张图很容易读成「已经是世界模型」。本课只承认配置，不承认家族名。报告 2.2.2 节把动作相关的生成写成三种局部窗口：

1. 前向动力学：动作 token 保持干净，去噪的是视觉 token。吃 $a$，吐未来画面。这是 $P(o'\mid o,a)$。
2. 逆动力学：视觉 token 保持干净，去噪的是动作 token。吃一段画面变化，吐中间的动作。这是从结果反推原因，接近逆向模型。
3. 策略（联合视频-动作）：视觉和动作一起去噪。吃当前画面和指令，同时吐动作块和未来视频。部署时可以只留下动作。

仓库 cookbook `cookbooks/cosmos3/generator/action/README.md` 把三个模式写成 `fd` / `id` / `policy`，并给出 DROID 上的动作定义：末端位姿 9 维（3 维平移加 6 维旋转）再加 1 维夹爪，合计 10 维，16 帧、15 FPS。Diffusers 路径用 `CosmosActionCondition(mode="forward_dynamics", ...)` 或 `mode="policy"`。vLLM-Omni 路径用 `extra_params` 里的 `action_mode`。

型号三档，README 写的是 Cosmos3-Super 64B、Cosmos3-Nano 16B、Cosmos3-Edge 4B。推荐硬件分别是数据中心卡、工作站 / 数据中心卡、Jetson 或 RTX Pro 6000。本课不编 24GB 上的显存数字。16B / 64B 只讲。Edge 官方写给边缘部署和实时策略，仓库有 `Cosmos3-Edge-Policy-DROID`，24GB 能否装下以你当时的 Hugging Face 卡片为准，装不下就停，不要降精度硬撑再写成体验成功。

后训练成策略的那条线在报告 4.2.5 节。他们从已经做过中间训练（mid-training：在通用动作和视频上先训过一遍，还没对准某个机器人）的 Cosmos3-Nano 接着训 `Cosmos3-Nano-Policy-DROID`：新初始化动作编码器、动作解码 MLP 和动作 embedding，动作相关参数学习率乘 5，输入是本体感觉加三路相机拼成的 540×640 画布，预测 32 步绝对关节位置，另外出辅助 RGB，15 Hz。推理用 4 步扩散、无分类器引导强度 3，并且跳过视频 latent 解码来换速度。策略服务器写的是 2 张 NVIDIA RTX Pro 6000。数据是 DROID：报告写 76k 条轨迹、350 小时、86 个任务、564 个场景。

这一段必须拆开读。辅助 RGB 说明 Generator 还在；跳过视频解码说明上场时他们优先保动作延迟。榜上的第一来自这个 Policy 检查点，不是来自未后训练的 Reasoner，也不是来自随便一段文本生成视频。

### 5.3 Cosmos Policy：同一套扩散序列里插入动作、未来和价值

Cosmos Policy（Kim, Gao, Lin 等，arXiv:2601.16163，2026-01-22）和上一节不是同一个检查点。它后训练的是 Cosmos-Predict2-2B-Video2World，不是 Cosmos 3 的 MoT。做法刻意不加新模块：在视频扩散的 latent 帧序列里插入新帧，把本体感觉、动作块、未来本体、未来图像、未来价值写成和图像同一形状的 latent，再按原来的去噪目标训练。论文把这叫做 latent frame injection。

训练时一个 batch 被切成三种条件掩码：一部分学 $p(a,s',V(s')\mid s)$，一部分学 $p(s',V(s')\mid s,a)$，一部分学 $p(V(s')\mid s,a,s')$。于是同一个网络可以当策略、当世界模型、当价值函数。消融里他们一路剥到只剩 $\pi(a\mid s)$，最大的掉点发生在「不再预测未来状态」那一步。这句话对桌宠有用：辅助地预测 $s'$，能抬策略成功率；但若推理时你只取 `actions`、从不对 $s'$ 设阈值，动作出口上仍然没有过滤器。

论文报告的数字按规划好来读：LIBERO 四套件平均成功率 98.5%（每套件 500 次 × 3 种子），RoboCasa 24 个厨房任务平均 67.1%（每任务 50 次 × 3 种子，只用每任务 50 条人工演示），ALOHA 真机四任务平均完成度 93.6%。加上基于模型的 best-of-N 规划之后，两个更难的 ALOHA 任务平均再高 12.5 个百分点。规划贵：论文写 8 路并行、每路一张 H100 时大约 4.9 秒才吐出一个动作块。仓库 README 写直接策略推理在 LIBERO 上约 6.8GB 显存，规划模式单卡约 10GB。这些是体验档的硬件依据，不是你复现出来的分数。

和 Cosmos 3 的对照只有一句：Predict2 后训练出来的 Cosmos Policy，三样输出从一开始就写在扩散序列里；Cosmos 3 用配置在前向 / 逆向 / 策略之间切换，后训练的 DROID 策略还可以选择不上视频解码。问「哪一个才是 $P(s_{t+1}\mid s_t,a_t)$」，两边都只有在你显式走前向条件、并且检查对换的时候，答案才是是。

### 5.4 $\pi_0$ 和 OpenVLA：两个动作头，没有未来分支

这一节只补第 26 课之后还要用来填三列表的部分。细节仍以那一课为准。

OpenVLA（Kim 等，arXiv:2406.09246）的理解侧是 Prismatic VLM：SigLIP 加 DINOv2 的图像 token 送进 Llama 2，语言指令是普通文本。生成侧不是视频，是 256 箱动作 token 的自回归。动作侧就是策略本身。仓库里离开模型的张量是 `predict_action` 的返回值。没有 `next_state`。

$\pi_0$（Black 等，arXiv:2410.24164）的理解侧是 PaliGemma，大约 30 亿参数，另加大约 3 亿参数的 action expert。生成侧是流匹配，推理按论文写的前向欧拉 10 步、步长 $0.1$，积出一个动作块。离开模型的张量是 `policy.infer(...)["actions"]`。FAST 版换成 DCT 动作分词再自回归，解决的是动作序列太长，不是补动力学。

$\pi_{0.5}$（Physical Intelligence，arXiv:2504.16054）有公开技术报告，openpi 后来也挂了 `pi05_base` / `pi05_libero` / `pi05_droid` 权重。它加的是异构共训和开放家庭环境里的打扫，不是前向模型。本课把它写成动作头的加强版，完整共训配方仍只讲，作业里不要出现 `scripts/train.py pi05_libero`。

GR00T 也能写，因为有公开技术报告。GR00T N1（NVIDIA，arXiv:2503.14734）是双系统 VLA：慢系统用 VLM 做语言和场景理解，快系统用扩散 Transformer 去噪连续动作。官方仓库 [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) 当前主分支是 N1.7，README 写骨干换成 Cosmos-Reason2-2B / Qwen3-VL，动作头仍是扩散 Transformer，许可证 Apache 2.0。N1.7 没有另立一篇与 N1 同级别、本课已核过全文的新报告；本课只引用 N1 论文和当前 README 能核到的接口，不编 N1.7 的成功率。三个头怎么填：理解头有（VLM），生成头若存在也是在为动作去噪，动作头就是策略。默认 E3。

这四家对桌宠的共同用途没变：当嘴，把「看我」「把笔推过来」变成候选动作。它们都不回答「同一时刻反方向伸手，杯子会不会倒」。

### 5.5 DINO-WM 和第 30 课：真的在学动力学

DINO-WM（Zhou, Pan, LeCun, Pinto，arXiv:2411.04983）几乎没有理解头，也没有语言动作头。冻结的 DINOv2 只当编码器，取出 `x_norm_patchtokens`。`models/visual_world_model.py` 把动作拼到每个 patch 上，`ViTPredictor` 预测下一组特征，损失是特征空间 MSE。解码器接在 `z_pred.detach()` 后面，重建不许回流进预测器。规划时用 CEM 在特征里搜动作，不需要专家演示。第 22、24 课已经在 PushT 上用过它。

第 30 课把同一张图纸缩到你的桌子上。状态是冻结 DINOv2 的 patch，动作是四个离散键：看左、看右、伸手、不动。胶水脚本的 `rollout(model, z0, a_seq, device)` 固定历史、只换动作序列，返回 $n$ 步预测特征。验收不看重建，看对换分岔、自由滚动漂移、负对照。这一行在三列表里应当是：

| 头 | 填什么 |
|---|---|
| 理解头 | 无（编码器不是 Reasoner） |
| 生成头 | 有，而且就是 $P(z_{t+1}\mid z_{\leq t}, a_{\leq t})$ |
| 动作头 | 无（规划器另接，默认不用 VLA） |

桌宠最小接入吃的就是这一行，不是 16B。

### 5.6 默认 E3，查询了前向模型才进 E4

第 33 课的两条主轴：预测有没有进动作回路（是否题 2），传感器和执行器是否共享物理世界（是否题 3）。套到本课的清单：

| 系统 | 是否题 1 对换 | 是否题 2 查询 | 是否题 3 真身体 | 主档 |
|---|---|---|---|---|
| Cosmos 3 Reasoner 只出文本 | 否 | 否 | 否 | E0 |
| Cosmos 3 Generator，无动作的文生视频 | 否 | 否 | 否 | E0 |
| Cosmos 3 `forward_dynamics`，只在模型里滚 | 应当是 | 若只观看来否 | 否 | E1 |
| Cosmos 3 / Cosmos Policy 直接出动作，真机执行 | 不必需 | 否 | 是 | E3 |
| 同上，执行前用未来状态或第 30 课模型否决 | 是 | 是 | 是或部分 | E4 |
| OpenVLA / $\pi_0$ / GR00T / ACT 真机 | 不必需 | 否 | 是 | E3 |
| DINO-WM 在 PushT 上 CEM | 是 | 是 | 否 | E2 |
| 第 30 课模型已对换，尚未接身体 | 是 | 否 | 部分（摄像头） | 先记 E1，接上第 32 课克制再升 |

「部分」必须写清：摄像头档的身体是屏幕和音效，桌子是真的；真机档把执行器补实。语言接口不升档。RoboArena 第一不升档。跳过视频解码、只出动作的 Cosmos 3 策略，即使辅助训练过 RGB，上场仍按 E3 打，除非日志里留下对未来状态的查询。

### 5.7 广告词里的 world model，以及 RoboArena 量的是哪一种好

把一句常见广告拆开：「Cosmos 3 是开源的 Physical AI（面向机器人、车辆等会动手的系统的基础模型口号）世界模型，Reasoner、Generator、动作一体，RoboArena 第一。」

1. world model 三个字：若说话人指 Reasoner，落点是理解头，尺子最多是问答准确率，不是 $P(s'\mid s,a)$。
2. 若指 Generator 的文生视频，落点是生成真，第 17 课的 FVD / 人工偏好那一列，第 12 课三问通常过不了动作这一问。
3. 若指 `forward_dynamics`，落点才是预测准，应当补动作对换，本课作业会要求你在笔记里写「未跑 / 已跑」。
4. 若指 `Cosmos3-Nano-Policy-DROID` 或 RoboArena，落点是规划好。报告原文把第一的时间钉在技术报告撰写时，图 26 标注 2026-05-30。后来的榜是否还第一，本课不追，也不需要追：身体是 DROID，评测是人群双盲两两比较，和桌宠的扫杯过滤器没有迁移定理。

Artificial Analysis 上的文生图 / 图生视频第一，同样只进生成真。Physics-IQ 一类续写榜进第 44 课，本课只提醒：续写对，不等于选过动作。

禁止写进笔记的句子：「RoboArena 第一，所以桌宠可以换上这个基础模型。」允许写的句子：「报告撰写时该策略检查点在 DROID 真机两两比较里排过第一；我家桌子没有 DROID，毕业设计仍用第 30 课模型过滤扫杯。」

## 6. 源码导读

五个仓库都按「先看返回值，再看动作从哪进，最后看官方把哪一档写成可跑」这条线读。路径以当前主分支为准，读之前用 `git log -1 --oneline` 记 commit。本课不要求你把推理依赖装齐。

### 6.1 nvidia/cosmos：先读两张表，再进 cookbook

```bash
git clone https://github.com/nvidia/cosmos.git
```

| 路径 | 读什么 |
|---|---|
| `README.md` 的 Cosmos 3 节 | Reasoner / Generator 两行输入输出；Super 64B、Nano 16B、Edge 4B 的硬件表 |
| `cookbooks/cosmos3/cosmos3-model-architecture.png` | MoT 双通路，对照报告图 5 |
| `cookbooks/cosmos3/README.md` | 共享环境；Reasoner 和 Generator 各链到哪一个后端 |
| `cookbooks/cosmos3/reasoner/README.md` | 理解头怎么跑；官方写明统一 omni 检查点会把 Generator 权重也装进来，后端不同显存不同 |
| `cookbooks/cosmos3/generator/action/README.md` | `fd` / `id` / `policy` 三模式、DROID 10 维动作、15 FPS、16 帧 |
| `cookbooks/cosmos3/generator/action/run_fd_with_diffusers.ipynb` | `CosmosActionCondition(mode="forward_dynamics")`，这是前向模型接口 |
| `cookbooks/cosmos3/generator/action/run_policy_with_diffusers.ipynb` | `mode="policy"`，返回视频加动作块 |
| `cookbooks/cosmos3/generator/action/run_policy_with_cosmos_framework.md` | Nano-Policy-DROID 与 Edge-Policy-DROID 的框架入口 |
| `cookbooks/cosmos3/generator/action/finetune/README.md` | 把 Nano 后训练成动作策略的 SFT 说明，本课不跑 |

README 的 Diffusers 最小例子用 `Cosmos3OmniPipeline.from_pretrained("nvidia/Cosmos3-Nano")` 做文本生成视频，`num_frames=189`、24 FPS、35 步。那是生成头的无动作配置，不要把它写成前向模型。动作模式另走 `CosmosActionCondition`。Generator 还要求申请 `nvidia/Cosmos-1.0-Guardrail`；只读文档可以不管，真要跑再按 README 关 `enable_safety_checker=False`。

Reasoner 的消息格式跟 Qwen3-VL 兼容，返回值是文本。把「杯子会不会倒」写成 prompt，得到的仍是一段话。把它和 `forward_dynamics` 的视频并排，才能看出哪一头在做对换。

### 6.2 nvlabs/cosmos-policy：看 `get_action` 一次返回三样

```bash
git clone https://github.com/nvlabs/cosmos-policy.git
```

| 路径 | 读什么 |
|---|---|
| `README.md` | 推理显存（LIBERO 约 6.8GB，规划约 10GB）和训练卡数（论文实验是 8 / 32 / 64 张 80GB） |
| `SETUP.md` | Docker 安装，本课不强制 |
| `cosmos_policy/experiments/robot/cosmos_utils.py` | `get_model`、`get_action` |
| `cosmos_policy/experiments/robot/libero/run_libero_eval.py` | `PolicyEvalConfig`，检查点键 `nvidia/Cosmos-Policy-LIBERO-Predict2-2B` |
| `LIBERO.md`、`ROBOCASA.md`、`ALOHA.md` | 三条评测线，本课只读协议 |

README 的最小 Python 示例一次取出三样：`action_return_dict['actions']`、`future_image_predictions`、`value_prediction`。这是本课最干净的「焊在一起」现场。问自己：若桌宠只把 `actions` 送给电机，另外两样存盘却从不设阈值，档位仍是 E3。

### 6.3 OpenVLA 与 openpi：回访返回值，不重做第 26 课全流程

OpenVLA 仍然看这些文件（第 26 课已经列过，本课只核返回值）：

| 路径 | 带着什么问题读 |
|---|---|
| `prismatic/models/vlas/openvla.py` | 动作 token 接在 VLM 哪一段后面 |
| `experiments/robot/openvla_utils.py` | `predict_action` 卸出来的是哪几个数 |
| `vla-scripts/finetune.py` | 本课不跑；确认没有 world model 开关 |
| `experiments/robot/libero/run_libero_eval.py` | 评的是任务成功，不是未来状态误差 |

openpi：

| 路径 | 带着什么问题读 |
|---|---|
| `src/openpi/models/pi0.py`、`pi0_config.py` | 流匹配动作头 |
| `src/openpi/models/pi0_fast.py` | FAST 仍吐动作 |
| `src/openpi/policies/policy_config.py` | `create_trained_policy`，`infer` 的入口 |
| `scripts/serve_policy.py` | 策略服务器，默认 8000 端口 |
| `examples/simple_client/main.py` | 无机器人冒烟，观察是随机的 |

对照着读，三件事必须能指到行：动作在哪个键里离开；有没有未来状态字段；评测脚本默认报的是成功率还是预测误差。

### 6.4 DINO-WM 和第 30 课胶水：前向模型的落点

```bash
git clone https://github.com/gaoyuezhou/dino_wm.git
```

| 路径 | 带着什么问题读 |
|---|---|
| `models/dino.py::DinoV2Encoder` | 状态是 patch 还是 CLS |
| `models/visual_world_model.py::encode` | 动作拼在 token 维还是特征维 |
| `models/visual_world_model.py::forward` | `z_loss` 和 decoder 损失谁对预测器有梯度 |
| `models/visual_world_model.py::replace_actions_from_z` / `rollout` | 对换时覆盖哪几维 |
| `models/vit.py::ViTPredictor` | 官方注意力掩码是否写死 CUDA |

第 30 课自己的脚本里，过滤器只需要 `rollout`。输入是历史特征 `z0` 和动作序列 `a_seq`，输出是 $n$ 步 $\hat{z}$。把伸手键写成一条长度为 10 的序列（5 帧/秒下的 2 秒），再和「不动」序列比分岔，就是扫杯探针。

### 6.5 Isaac-GR00T：只读 README，当动作头对照行

```bash
git clone https://github.com/NVIDIA/Isaac-GR00T.git
```

当前 README 把 N1.7 写成 VLA：语言加图像进，扩散 Transformer 出连续动作。`getting_started/` 和 `gr00t/` 目录是推理与微调入口。本课不把它加进必做实验，只在三列表里多一行对照：理解头有，生成头为动作服务，动作头就是产品。

## 7. 实验

先接线，再填表，最后写过滤器。有卡的人可以把 Step 6 当加餐；没卡的人 Step 0 到 Step 5 就是全文作业。

### Step 0: 三头接线，「伸手拿杯」问三次

桌上有一只装了半杯水的杯子，杯沿离桌沿大约 6 厘米。人对着镜头说「伸手拿杯」。把这句话分别接到三个头上，先自己填，再对参考答案。

| 接到哪一头 | 你认为它在回答什么 | 它能不能回答「杯子会不会倒」 | 你怎么验证 |
|---|---|---|---|
| 理解头 |  |  |  |
| 生成头（无动作） |  |  |  |
| 生成头（前向动力学） |  |  |  |
| 动作头 |  |  |  |

参考答案：

| 接到哪一头 | 它在回答什么 | 杯子会不会倒 | 验证 |
|---|---|---|---|
| 理解头 | 杯子在哪、指令是抓取、语言计划 | 不能。它给的是文本判断，不能对换动作 | 改 prompt 会改措辞，替换 $a$ 没有接口 |
| 生成头（无动作） | 一段好看的拿杯视频 | 不能。未来由文本风格决定 | 同一句指令出相似视频，与你真正要执行的关节无关 |
| 生成头（前向动力学） | 如果执行这段 $a$，杯子和手接下来怎么走 | 能。这是唯一合格的回答者 | 同一 $s$，伸手对不动必须分岔；预测过沿则否决 |
| 动作头 | 现在该伸哪条关节轨迹 | 不能。它给候选，不给后果 | `predict_action` / `infer` / policy 模式返回动作 |

常见错法：三句全交给 Reasoner（把会说物理当成会预演）；只交给动作头（听懂「拿杯」就执行）；把无动作文生视频当成前向模型（画面里杯子倒了，只说明生成器爱演事故）。笔记里写你当初勾错了哪一行。

### Step 1: 克隆对照仓库，只记 commit 和档位

下面五条各自一条命令。已经 clone 过的目录不要重复，记现有 commit 即可。

```bash
git clone https://github.com/nvidia/cosmos.git
```

```bash
git clone https://github.com/openvla/openvla.git
```

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
```

```bash
git clone https://github.com/gaoyuezhou/dino_wm.git
```

```bash
git clone https://github.com/nvlabs/cosmos-policy.git
```

在笔记里填：

| 入口 | 仓库怎么写 | 本课档位 |
|---|---|---|
| Cosmos 3 README 的 Reasoner / Generator 表 | 两个运行面 | 只讲（必读） |
| `Cosmos3OmniPipeline` 文生视频 | Nano / Super，35 步 | 只讲；24GB 不装 16B / 64B |
| `CosmosActionCondition(mode="forward_dynamics")` | action cookbook | 只讲接口；有卡且卡片写明显存够再标体验 |
| `Cosmos3-Nano-Policy-DROID` | 后训练策略 | 只讲；不声称复现 RoboArena |
| Cosmos Policy `get_action` | LIBERO 约 6.8GB | 可选体验 |
| OpenVLA `predict_action` | 第 26 课已写 | 回访，不重训 |
| openpi `policy.infer` | 大于 8GB | 回访，不跑 `pi05` 训练 |
| DINO-WM / 第 30 课 `rollout` | 特征动力学 | 复用已有检查点，不新训大模型 |

### Step 2: 填三列表（本课主证据）

同一件桌面小事：伸手拿杯，路上可能把水扫下桌。五行星，三列，空格写「无」。

| 系统 | 理解头 | 生成头 | 动作头 |
|---|---|---|---|
| Cosmos 3 | Reasoner：图+字进，字出；规划是文本 | Generator：可文生视频，也可 `forward_dynamics` | 后训练 Policy，或 `mode="policy"` 联合出动作和视频 |
| $\pi_0$ / openpi | PaliGemma 读图读字 | 流匹配生成的是动作块，不是下一状态 | 整个模型就是 $\pi(a\mid o,\ell)$ |
| OpenVLA | Prismatic VLM | 自回归动作 token，不是视频 | `predict_action` |
| DINO-WM | 无（冻结 DINOv2 只编码） | ViT 预测 $z_{t+1}$，这是动力学 | 无；CEM 另接 |
| 第 30 课桌面模型 | 无 | `rollout` 预测 patch 特征 | 无；第 32 课规划器 / 安全层另接 |

表底下再加两行对照，不占主验收，但能挡住张冠李戴：

| 系统 | 理解头 | 生成头 | 动作头 |
|---|---|---|---|
| Cosmos Policy（Predict2-2B） | 文本条件走 T5，不是 Cosmos 3 Reasoner | 可解码未来图像 | `get_action` 的 `actions`；规划时才用未来和价值 |
| GR00T N1 / N1.7 | VLM（N1 论文；N1.7 README 写 Cosmos-Reason2-2B / Qwen3-VL） | 扩散头去噪的是动作 | VLA，默认 E3 |

### Step 3: 读一句广告词，写阅读笔记

任选下面一句，或换一条你此刻在 Cosmos 项目页、Hugging Face 卡片上看到的原文。把原句抄进笔记，再按四栏拆。

示例原句（来自 arXiv:2606.02800 摘要口径）：后训练的 Cosmos 3 模型在技术报告撰写时，被 Artificial Analysis 列为开源文生图 / 图生视频最优，被 RoboArena 列为当时最优策略模型。

| 广告词片段 | 落在哪一头 | 第 17 课哪根尺子 | 第 33 课哪一档 | 能不能迁到桌宠扫杯 |
|---|---|---|---|---|
| 开源文生图 / 图生视频最优 | 生成头，无动作 | 生成真 | E0 | 不能 |
| RoboArena 最优策略 | 动作头 | 规划好 | 真机开环则 E3 | 不能。DROID 不是书桌 |
| omnimodal world model | 三个头的总称 | 未指定 | 未指定 | 不能。要看配置 |
| `forward_dynamics` cookbook | 生成头，有动作 | 预测准（需对换） | 离线则 E1 | 接口可借鉴，权重不必换 |

笔记结尾写一句本课验收要检查的话：基础模型广告里的 world model，多数时候不是 $P(s_{t+1}\mid s_t,a_t)$。

### Step 4: 桌宠最小接入（不新训大模型）

不写新训练代码。用第 30 课已经有的 `rollout`，加上第 25 课 ACT 或第 26 课 VLA 的候选动作。没训过第 30 课的人，把函数签名抄对，用伪代码交差。

数据流：

1. 观察 $I_t$ 经第 29 课感知得到杯子框、桌沿线；同时经第 30 课编码器得到 $z_t$。
2. 策略给出 $a^{\mathrm{prop}}$。语言指令走 VLA；固定技能（挥手、点头）走 ACT。本课不在 24GB 上新训它们。
3. 把 $a^{\mathrm{prop}}$ 量化到第 30 课的四键。末端朝杯子且速度向外，记为伸手；幅度低于阈值，记为不动。量化规则写进笔记，允许粗糙。
4. 调用 `rollout(model, z0, a_seq, device)`，`a_seq` 为长度 10 的伸手序列，再滚一条不动序列。
5. 三条否决：伸手分支与不动分支在 2 秒处的特征距离低于你第 30 课对换表的健康阈值（模型动作盲，过滤器没有依据，直接拒伸手）；把 $\hat{z}$ 上的杯子位置（或 PCA 伪彩里杯对应的块）外推过桌沿 5 厘米；预测器对自己的未来没把握（自由滚动已越过第 30 课漂移曲线的偷懒基线）。
6. 否决则 $a^{\mathrm{exec}}=$ 停下，屏幕提示「会碰到杯子」，日志写 `wm_forward`、候选动作、哪一条否决。放行则执行 $a^{\mathrm{prop}}$。
7. 「把杯子推到桌沿」这种指令在过滤器之前就拒，沿用第 26 课 Step 0。

类型要具体：

$$
a^{\mathrm{exec}} =
\begin{cases}
a^{\mathrm{prop}} & \text{对换分岔，且 }\hat{r}(z_t,a^{\mathrm{prop}})\ \text{全部低于阈值} \\
a^{\mathrm{stop}} & \text{否则}
\end{cases}
$$

$\hat{r}$ 来自第 30 课模型，不来自 VLA 的 softmax，也不来自 Reasoner 的一句「我觉得危险」。

若你有 `runs/desk1/best.pt`，用真实检查点跑一次伸手对不动，把 `action_swap.png` 或分岔指数抄进本课笔记。没有检查点，就停在接口，档位标「设计稿，未接权重」。

### Step 5: 给五个系统打 E 档

用第 33 课七条是否题，给 Step 2 主表的五行各打一档。答案只能是是、否、部分。部分必须写清。不要因为 Cosmos 3 参数多就把 Reasoner 写成 E4。

最低要求：OpenVLA 或 $\pi_0$ 写成 E3；第 30 课未接身体的模型不要写成 E4；接上 Step 4 过滤器并且日志有查询，才能把桌宠设计档写成 E4。

### Step 6: 可选体验，24GB 够哪条做哪条

本步失败标体验失败，不影响及格。

有第 26 课环境的人，用灰图再跑一次 `predict_action`，确认返回值仍是动作向量、没有拒绝通道。命令和代码以第 26 课 Step 2 为准，本课不重复贴。

有 Ubuntu、Docker、大约 8GB 以上显存的人，可以按 cosmos-policy 的 `SETUP.md` 进容器，跑 README 那段 `get_action` 示例。预期打印动作块、两张未来图、一个价值标量。把三样输出的键名抄进笔记，并写一句：若只用动作键，过滤器等于没装。

不要下载 Cosmos3-Nano 或 Super 来「顺便看看」。那不是本课作业。

## 8. 配置与预算

本课几乎没有你自己的训练超参。预算按研究档来，把官方数字和你不该花的钱分开。

| 步骤 | 硬件 | 时间（参考） | 说明 |
|---|---|---|---|
| Step 0 三头接线 | 纸或编辑器 | 20 分钟 | 必做 |
| Step 1 克隆与分档 | 任意机器 | 30 分钟 | 必做；五个浅克隆约 1 到 2GB |
| Step 2 三列表 | 任意 | 1 小时 | 必做 |
| Step 3 广告词笔记 | 任意 | 40 分钟 | 必做 |
| Step 4 过滤器草稿 | 任意；有第 30 课权重更佳 | 1 到 2 小时 | 必做；不新训 |
| Step 5 E 档 | 任意 | 30 分钟 | 必做 |
| Step 6 OpenVLA 回访 | 单张 24GB | 分钟级推理 | 可选 |
| Step 6 Cosmos Policy `get_action` | Docker + 约 8GB | 下载检查点为主 | 可选 |
| Cosmos 3 Nano / Super 推理 | 官方写工作站 / 数据中心卡 |  | 只讲 |
| Cosmos 3 或 Cosmos Policy 后训练 | 论文写 8 到 64 张 80GB | 数十小时 | 禁止当本课作业 |
| $\pi_{0.5}$ 共训 / GR00T 全量微调 | 多卡 |  | 只讲 |

第 30 课若还没训，不要在本课补训一个「看起来像」的大模型。过滤器可以先交设计稿。真要补动力学，回到第 30 课的缩小档：冻结 DINOv2、2 层预测器、自己桌子上的四键，单张 24GB 数十分钟，Mac 按一小时准备。

超参凡涉及 Cosmos 3 Policy 后训练（学习率 $2\times 10^{-4}$、动作参数 5 倍、32 步关节、4 步扩散、CFG 3），一律当作报告数字，不要抄进你的训练脚本。本课没有训练脚本。

## 9. 验收

- [ ] Step 0 四行都勾过，并且能指出「杯子会不会倒」只应由前向动力学回答。
- [ ] 三列表五行填完，DINO-WM 和第 30 课的动作头写的是「无」，OpenVLA / $\pi_0$ 的生成头没有被写成下一状态。
- [ ] 广告词笔记四栏齐全，RoboArena 被标成规划好，且写明不能迁到桌宠扫杯。
- [ ] 过滤器七条接口齐全：候选从哪来、四键怎么量化、`rollout` 谁来调、三条否决、否决后做什么、第三句危险指令为何在过滤器之前就拒、日志里哪个字段证明查询发生过。
- [ ] 五个系统的 E 档指向是否题，OpenVLA / $\pi_0$ 默认 E3，没有因为参数量或榜上升档。
- [ ] 分档表里，16B / 64B 训练和 RoboArena 复现被标成不跑。
- [ ] 能口头说出 Cosmos Policy 和 Cosmos3-Nano-Policy-DROID 后训练的是哪一个基座。
- [ ] 证据目录最小集：五个仓库的 commit（或「未克隆，读网页」）、三列表、接线题答案、广告词笔记、过滤器草稿。做了 Step 6 的人加上返回值键名。

口头关：向没上过第九幕的人讲清「基础模型上的三个头怎么拆」。听完的人应当能指出：听懂「拿杯」靠理解头或 VLA；决定伸不伸手靠前向模型；RoboArena 第一不是书桌验收。

## 10. 排错

| 症状 | 最可能的原因 | 怎么确认 | 怎么修 |
|---|---|---|---|
| 把 Reasoner 的「会倒」当成过滤器 | 把文本判断当成 $P(s'\mid s,a)$ | 问自己能不能替换 $a$ 再前向一次 | 改接 `forward_dynamics` 或第 30 课 `rollout` |
| 三列表里 OpenVLA 的生成头写成下一帧 | 把动作 token 生成当成世界生成 | 看 `predict_action` 返回值 | 改成「动作 token，无未来状态」 |
| 把 Cosmos Policy 写成 Cosmos 3 | 名字都带 Cosmos | 看基座：Predict2-2B 还是 Cosmos3-Nano | 拆成两行 |
| 16B 一加载就 OOM | 主线 24GB 不够 Nano / Super | 官方硬件表 | 停。改读 cookbook，标只讲 |
| Cosmos Policy 示例缺键 | 没走 Docker / 检查点路径不对 | README 的 `ckpt_path` | 按 `SETUP.md`；失败则只读示例 |
| 过滤器从不触发 | 四键量化把所有动作映成不动，或模型动作盲 | 先跑第 30 课对换表 | 对换塌了回去补数据；量化过粗就先只过滤「伸手」 |
| 伸手和不动的 $\hat{z}$ 几乎重合 | 第 30 课模型动作盲 | 负对照比值贴 1 | 不要上真机。先拒伸手，再补伸手轨迹 |
| 想用 RoboArena 第一代替 Step 5 | 把规划好当成具身档 | 是否题 2 有没有查询 | 删掉榜名字，重打是否题 |
| 在本课启动 Cosmos 3 SFT | 把 finetune cookbook 当作业 | `cookbooks/cosmos3/generator/action/finetune/` | 删掉这条命令 |
| Mac 上装 openpi 或 cosmos-policy | 官方按 Ubuntu / Docker 写 | 编译失败 | 精读源码，不装也算 Step 1 到 5 完成 |
| 第 30 课还没有 `best.pt` | 想在本课从头训桌面模型 | 本课档位是研究 | 交设计稿，回去第 30 课补权重 |

## 11. 前沿与改造

前沿怎么做。2026 年的公开系统还在把三个头往同一套权重里焊。Cosmos 3 用 MoT 把 Reasoner 和 Generator 做成可切换配置，再后训练出 DROID 策略。Cosmos Policy 走另一条更窄的路：不改 Predict2 的骨架，只在 latent 帧里插入动作和价值，必要时用 rollout 数据把世界模型和价值函数再训一遍，做 best-of-N。VLA 一侧，$\pi_{0.5}$ 用异构共训换开放家庭，GR00T N1.7 用更大的人形数据和更强的 VLM 骨干换跨身体，动作头仍是扩散或流匹配。世界模型一侧，DINO-WM、第 24 课的视觉 MPC、第 30 课的桌面模型继续走「先预测再选动作」。两条线都在变强，接口差异没变。

我们差在哪。规模差：DROID 350 小时、互联网视频、64B 卡时，不是一张 24GB 卡能补的。机制差：桌宠真正缺的是动作出口上那一次可对换的查询，不是再大一个 Reasoner。钱能买到更好的 VLA 微调和更漂亮的生成真，买不到「伸手会不会扫杯」这条查询，除非另外加载或训练一个前向模型。第 25、26 课已经证明模仿策略和 VLA 同样缺它。本课只是把缺的那截画在 2026 年的基础模型后面。

动手改造清单，全部允许在缩小设置里失败，全部不新训 16B。

1. 给第 25 课 ACT 或第 26 课 OpenVLA 的动作出口包一层第 30 课 `rollout`。改的是你自己的胶水，不改 `prismatic/`，不改 `src/openpi/`。预算：半天。预期：日志出现 `wm_forward` 和至少一次拒绝。失败：过滤器从未触发，或误伤「看我」。记录误伤次数，不要用规则补丁冒充动力学。
2. 同一张桌面照片，三问对照。问 Reasoner「伸手会不会倒」（若你跑得动任何 VLM，用现成的即可，不必是 Cosmos 3）；问第 30 课模型伸手对不动是否分岔；把 OpenVLA 的指令改成 `push the cup to the edge` 看它是否仍吐动作。预算：三次推理或两次纸面加一次本地 `rollout`。预期：三者答案可以互相打架。失败：你把三个答案合成一个「模型理解物理」的总分。
3. 动作对换探针只打生成头。Cosmos 3 的 `forward_dynamics` 若显存不够，就用 DINO-WM 官方 PushT 或第 30 课模型：同一 $z$，伸手对不动。预算：推理级。预期：你能指出哪一个返回值里没有 $a$。失败：把两次不同语言指令的 VLA 输出，当成两条想象。
4. 若有 8 卡短租：按 cosmos-policy 的 `LIBERO.md` 在**小**演示集上微调 2B，只为看 `get_action` 是否仍同时出未来图。预期：未来图对 LIBERO 桌面还有物体形状；换到你的书桌照片会漂。失败：把 LIBERO 成功率写成桌宠已经可迁移。这是加餐，笔记标可选。

顺手复现（方向性，不是数字）：

| 论文或报告结论 | 缩小版 | 预期 |
|---|---|---|
| VLA 是 $\pi(a\mid o,\ell)$ | 灰图仍吐动作，无 `next_state` | 与第 26 课同方向，本课必现 |
| 去掉未来状态预测，Cosmos Policy 掉点最大 | 读论文消融，24GB 做不到公平对照 | 只读，不装复现 |
| Cosmos 3 三种动作模式接口不同 | 对照 cookbook 的 `fd` / `id` / `policy` | 能指到 `CosmosActionCondition.mode` |
| RoboArena 第一可迁到桌宠 | 无缩小版 | 不能复现，也不该复现 |
| 第 30 课对换分岔后才能当过滤器 | 已有 `best.pt` 则跑伸手对不动 | 分岔失败则过滤器只能拒，不能放行 |

## 12. 论文与延伸

每篇只列你要用的读法。数字以你打开的版本为准。

1. NVIDIA. *Cosmos 3: Omnimodal World Models for Physical AI*. arXiv:2606.02800（本课按 v4, 2026-06-23）。读第 2 节 MoT、token 排列、三种动作模式；读 4.2.5 节 DROID 后训练；读 6.2.5 节他们自己怎么写 RoboArena。阅读问题：Reasoner 的因果注意和 Generator 的双向注意，谁允许对换 $a$？跳过视频解码之后，上场还剩哪一头？图 26 的日期为什么必须抄进笔记？

2. Kim, Gao, Lin et al. *Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning*. arXiv:2601.16163。读 latent frame injection、三种掩码、best-of-N。阅读问题：同一网络何时是 $\pi(a\mid s)$，何时是 $P(s'\mid s,a)$？消融里「不再预测未来」为什么掉点最大，却仍不能单独证明部署时查询了世界模型？

3. 回访 Kim et al. *OpenVLA*. arXiv:2406.09246；Black et al. *$\pi_0$*. arXiv:2410.24164。阅读问题：两个返回值里哪一个最接近「杯子下一秒的位置」？若答案是「都没有」，三列表该怎么填？

4. Zhou, Pan, LeCun, Pinto. *DINO-WM*. arXiv:2411.04983。阅读问题：这篇几乎没有理解头和动作头，为什么反而最接近本课的世界模型定义？零样本规划的「零」指没在目标任务上训策略，还是没在你的桌子上训动力学？

5. Physical Intelligence. *$\pi_{0.5}$: a Vision-Language-Action Model with Open-World Generalization*. arXiv:2504.16054。只讲。阅读问题：异构共训转移的是语义还是接触力学？openpi 能下 `pi05_libero`，和「复现 $\pi_{0.5}$ 论文」差了哪几块？

6. NVIDIA. *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots*. arXiv:2503.14734。对照仓库 README 的 N1.7 说明。阅读问题：双系统里哪一块是理解头，哪一块是动作头？N1.7 换骨干有没有改变「默认 E3」？

7. Liang, Yu, Luo et al. *Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models*. arXiv:2411.04996。选读。阅读问题：原文的模态分参数，和 Cosmos 3 的 Reasoner / Generator 双通路，差在分的是模态还是任务？

8. Atreya, Nasiriany 等. *RoboArena: Distributed Real-World Evaluation of Generalist Robot Policies*. arXiv:2506.18123。读双盲两两比较和 DROID 平台假设。阅读问题：这个验证者和第 32 课的桌子差在哪三件事（身体、任务分布、能否 `reset`）？

延伸不必做：Cosmos 3 的声音通道留给第 36 课的口径；工业级动作条件后训练流水线是第 40 课的主题；「懂物理」的操作化是第 44 课。下一课的麻烦换成可玩的世界：Genie 3、Oasis、Matrix-Game 会把实时交互和数分钟一致性焊在另一句广告里。三个头的拆法仍然够用。帧率再高，不问动作对换，桌宠也不能把它当成小脑。
