---
id: 35_cosmos3_omnimodal
title: "Cosmos 3 的全模态 MoT"
summary: "语言、图像、视频、声音、动作进同一套权重，哪些配置是 VLM，哪些才是世界模拟器？"
unit: frontier
play_tools: []
checkpoints:
  - "一张输入输出配置分类表。"
  - "一段 Predict2.5 与 Cosmos 3 的对照笔记。"
---

# 第 35 课：同一套 MoT，配置决定它是 VLM 还是世界模拟器

> 类型：研究（文档精读必做）+ 只讲（Nano 16B / Super 64B 的推理、完整后训练、官方排行榜）<br>
> 建议周期：2-3 天<br>
> 硬件：主线单张 24GB 只读报告、仓库文档、官方样例视频并填表。NVIDIA/cosmos-framework 的 FAQ 写明 Cosmos3-Nano（按单塔 8B 计）推理约 32 GB，Cosmos3-Super（按单塔 32B 计）约 128 GB，24GB 不够。Cosmos3-Edge（4B，单塔 2B）官方标单卡可跑，FAQ 未给 24GB 消费卡数字，本课不把它写成必做体验<br>
> 锚定仓库：[NVIDIA/cosmos](https://github.com/NVIDIA/cosmos)（产品 README、cookbook、样例），训练与推理代码对照 [NVIDIA/cosmos-framework](https://github.com/NVIDIA/cosmos-framework)；权重见 [Hugging Face collections/nvidia/cosmos3](https://huggingface.co/collections/nvidia/cosmos3)；技术报告 [arXiv:2606.02800](https://arxiv.org/abs/2606.02800)<br>
> 产物：一张官方输入-输出配置分类表、一段「第 22 课 Predict2.5 和第 35 课 Cosmos 3 差在哪」的对照笔记、一份模态路由板答案、按第 33 课口径给无动作生成和有动作离线播各打一档

## 1. 这一课做什么

第八幕已经毕业：第 32 课把桌宠装上身体，第 33 课用 E0 到 E5 打具身程度。第九幕不改那把尺子。第 34 课把 2025-2026 年的榜还原成输入、输出和评分器，告诉你 WorldScore 第一名不必能给桌宠做 MPC。这一课把其中一块最大的开源系统拆开：NVIDIA Cosmos 3。

贯穿主干没变：

```text
观察 先压成状态 再按动作预测下一状态 然后展开多条未来 给未来打分 最后选动作
```

Cosmos 3 换的不是桌宠上的小模型，是同一条循环在工业级全模态系统里长什么样。它把语言、图像、视频、声音、动作收进一套 Mixture-of-Transformers（混合 Transformer：两套参数共用注意力）。Reasoner 通路用因果自注意力吐文字；Generator 通路用双向注意力给图像、视频、声音、动作的噪声 token 去噪。两条通路共享 3D mRoPE（多维旋转位置编码：给 token 标上时间、高、宽）。产品形态不靠换骨架，靠换输入输出：同一份权重，可以当视觉语言模型（VLM），可以当视频生成器，可以当世界模拟器，也可以当世界-动作模型。

本课必须和 [第 22 课](22_foundation_video_wm.md) 分开。那课的 Cosmos 是 Cosmos-Predict2.5：Text / Image / Video2World 的视频基础模型，动作端口在后训练分支 `robot/action-cond`。本课的 Cosmos 3 是 2026 年 5 月 31 日公开的 omnimodal（全模态：多种观察和动作在同一套序列里进出）世界模型，报告编号 arXiv:2606.02800。仓库 README 自己写着 Predict2.5 不再活跃开发。不要把两课的检查点、显存数字、命令抄串。

24GB 主线假设下，本课不跑 Nano，不训 16B/64B。官方 FAQ 的显存表把 Nano 写成 32 GB、Super 写成 128 GB。权重按双塔合计是 Nano 16B、Super 64B、Edge 4B。桌宠默认仍用 [第 30 课](30_desk_world_model.md) 的小模型。麦克风、喇叭、关节可以进同一套大模型，那是以后短租 80GB 级卡才考虑的零件，不是毕业标准。

必做的动手是读文档、列配置、对照官方样例视频。有 80GB 或以上再走官方推理命令，结论标成加餐，不是本课验收。Artificial Analysis 的文生图 / 图生视频排名、RoboArena 的策略排名，报告写的是他们提交时的宣称，你没有复现。

术语速查：

| 术语 | 一句人话 |
|---|---|
| omnimodal | 语言、图像、视频、声音、动作在同一套模型里处理；产品形态由输入输出组合决定 |
| MoT | Mixture-of-Transformers：层里放两套（或多套）参数，在共享注意力里会合 |
| Reasoner 通路 | 因果自注意力那条塔，吃文字和 ViT 视觉 token，吐下一个词 |
| Generator 通路 | 双向去噪那条塔，吃 VAE / 音频 / 动作的噪声 token，吐连续模态 |
| mRoPE | 给每个 token 一个 $(t,h,w)$ 坐标，让不同帧率的视频、声音、动作对齐到同一条物理时间轴 |
| 世界模拟器 | 条件里有观察和动作，输出是未来画面；对应 $P(o_{t+1}\mid o_t,a_t)$ 的像素版 |
| 世界-动作模型 | 同一段序列里同时去噪动作和视频，既预测怎么动，也预测动完画面怎样 |
| 配置 | 哪些 token 干净、哪些带噪声；换配置等于换产品，不必换网络 |
| 只讲 | 本课不跑权重、不训练；读报告和当前 README，把宣称和可验证事实分开 |
| E0 / E1 | [第 33 课](33_embodiment_degrees.md) 的头两档：无动作是 E0，有动作但只离线播是 E1 |

## 2. 问题

第 22 课已经证明过：名字叫世界模型，输入输出可以完全不同。Predict2.5 的基础检查点吃文本加图像或视频，吐未来视频；DINO-WM 吃观察和低层动作，吐下一时刻的 patch 特征。Cosmos 3 把这件事推到更滑：同一份 16B 或 64B 权重，官方 README 同时列出 Reasoner 和 Generator 两张表面。你若只看发布会，会以为「全模态」自动等于「桌宠要的动力学」。本课要拆开三件具体的事。

第一，全模态是观察变多了，还是真的把世界模拟器和动作模型做成同一套权重。语言、图像、视频、声音都可以只是生成通道。动作必须作为因果变量进序列，预测才随 $a_t$ 分岔。报告把动作定义成「诱导世界状态改变的因果变量」：相邻视频 token $v_{t-1}$ 到 $v_t$ 之间夹一个 $a_t$。这和 [第 03 课](03_mdn_rnn_action_conditioned.md) 的动作条件是同一句话，只是 token 换成了 VAE latent 和相对位姿。

第二，Reasoner 和 Generator 为什么要两套参数还共享注意力。纯 VLM 只会写字，纯视频扩散只会画。Physical AI 要两件事同时成立：看懂桌上有什么，以及按这个动作把未来播出来。报告的做法是双塔：AR 子序列走 Reasoner，扩散子序列走 Generator，扩散 query 可以看全部 AR 的 key/value，AR 绝不回看扩散。共享的是块结构和 mRoPE，不是把理解头和生成头焊成一个前向。

第三，官方排行榜测的是哪一种「好」。报告摘要写：后训练过的 Cosmos 3 在技术报告撰写时被 Artificial Analysis 列为最佳开源文生图和图生视频模型，并被 RoboArena 列为最佳策略模型。那是生成真和下游策略成功率的宣称，对应 [第 12 课](12_frontier_landscape.md) 三分评测里的两根尺子，不是你在 24GB 上测过的预测准，更不是桌宠的规划好。本课验收看配置表和对照笔记，不看你有没有刷过那两份榜。

一条界限先划清。无动作的文生图、文生视频、图生视频，按第 33 课记 E0。有动作条件、只在录好的轨迹或官方样例上离线播，记 E1。接到会拒绝你的身体、在回路里选动作，本课不验收，那是以后短租加餐或第 40、42 课的事。桌宠 24GB 默认仍用第 30 课小模型。

## 3. 准备

- 读过 [第 12 课](12_frontier_landscape.md) 的三问和三分评测，[第 17 课](17_evaluating_world_models.md) 的预测准 / 生成真 / 规划好，[第 22 课](22_foundation_video_wm.md) 的 Predict2.5 资格审查，[第 26 课](26_vla_vs_world_model.md) 的 VLA 与世界模型分工，[第 33 课](33_embodiment_degrees.md) 的 E0 到 E5。第 34 课若已写完，把它的地图放在手边：Cosmos 3 那一行不要填你没测过的数字。
- 一台能上网的电脑。必做步骤不需要 GPU。Linux 最顺，因为两个锚定仓库都按 Linux 写；Mac / 纯 CPU 可以完成克隆、读 README、下载官方样例视频、填表。
- Hugging Face 账号可选。Generator 路径会要求申请 [nvidia/Cosmos-1.0-Guardrail](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail) 或框架文档里的 [nvidia/Cosmos-Guardrail1](https://huggingface.co/nvidia/Cosmos-Guardrail1)；本课必做不加载 Generator，可以先不申请。
- 磁盘：只读文档大约几百 MB。若加餐跑推理，cosmos-framework 的 setup 文档建议预留约 150 GiB（Hugging Face 缓存约 90 GiB，uv 缓存约 20 GiB，一次运行输出约 30 GiB）。主线不要为 Nano 权重清盘。
- 论文 PDF 或 HTML： [arXiv:2606.02800](https://arxiv.org/abs/2606.02800)，当前 HTML 版本是 v4（2026-06-23）。项目页 [research.nvidia.com/labs/cosmos-lab/cosmos3](https://research.nvidia.com/labs/cosmos-lab/cosmos3/)。许可证是 Linux Foundation 的 OpenMDW-1.1，以仓库 LICENSE 为准。

## 4. 学习目标

1. 画出 Cosmos 3 一条序列里 AR 子序列和扩散子序列的前后关系，并说出 Reasoner 的注意力是因果的、Generator 的注意力是双向的、AR 为什么不许回看扩散。
2. 用官方 README 的 Use Cases 表，把每一种输入-输出组合标成 VLM / 视频生成器 / 世界模拟器 / 世界-动作模型 / 策略头 之一，并写出第 12 课三问里哪一问还没回答。
3. 口头说明第 22 课 Predict2.5 和第 35 课 Cosmos 3 在产品切分、动作端口、模态、显存上的四处差别，不把两套检查点当成同一个东西。
4. 写出动作 token 的几何定义：相邻位姿的相对变换 $\Delta\mathbf{T}_t=\mathbf{T}_{t-1}^{-1}\mathbf{T}_t$，以及不同身体用域相关投影接到同一套 MoT。
5. 写出 Generator 的 rectified flow matching 目标：噪声 latent 是干净目标和高斯噪声的直线插值，网络预测速度 $v^{\ast}=\epsilon-x_0$。
6. 按第 33 课口径，给「无动作的 T2V」打 E0，给「forward dynamics 离线播」打 E1，并说明为什么官方 RoboArena 排名不能把档升到 E4。
7. 交出三份产物：配置分类表、Predict2.5 对照笔记、模态路由板答案。24GB 上没有 Nano 推理日志，不算缺作业。

## 5. 原理

五个机制。每个按同一节奏走：为什么需要它、它怎么工作、精确定义、在源码哪里、怎么证明你没看错。

### 5.1 输入输出是配置，不是魔法

家用打印机有扫描、复印、打印三个按钮，里面可能是同一组滚筒。Cosmos 3 的发布会也爱说「一个模型，多种应用」。类比失效处：打印机按钮不会把「复印」变成「自动驾驶策略」；Cosmos 3 的按钮是 token 布局。哪些位置放干净条件、哪些位置放噪声目标，决定这次前向是 VLM、文生视频，还是按动作滚未来。

报告 2.2 节把一条输入拆成两段。前面是 AR 子序列：语言 token，加上 ViT 编码的图像或视频，末尾接 `<EOS>` 和开始生成的 `<BOG>`。后面是扩散子序列：VAE 编码的图像或视频、音频、动作。三条硬规则对所有任务成立：(1) AR 永远在扩散前面；(2) 扩散段里，每种模态的干净条件在噪声目标前面；(3) 条件段和扩散段内部，顺序都是视觉、音频、动作。

于是官方支持的生成模式可以直接写成序列。下面 $l_{1:n}$ 是语言，$\tilde{v}$ 是带噪声的视觉 token，$v_{1:P}$ 是干净的条件帧，$\tilde{s}$ 是带噪声的音频。文生图是

$$
\mathbf{S}_{\mathrm{T2I}}=[\mathbf{S}_{\mathrm{AR}},\;\tilde{v}_{1}]
$$

文生视频（可选音频）是

$$
\mathbf{S}_{\mathrm{T2V+Audio}}=[\mathbf{S}_{\mathrm{AR}},\;\tilde{v}_{1:N},\;\tilde{s}]
$$

图生视频或视频续写是

$$
\mathbf{S}_{\mathrm{V2V}}=[\mathbf{S}_{\mathrm{AR}},\;v_{1:P},\;\tilde{v}_{P+1:N}]
$$

$P=1$ 就是图生视频，$P>1$ 就是视频续写。控制视频迁移把干净的深度或边缘图当作 $v^{\mathrm{ctrl}}$，噪声目标是 RGB。

动作有三种布局，报告图 4 画得很清楚。前向动力学：动作 token 干净，去噪的是未来视觉。逆动力学：视觉干净，去噪的是动作。策略 / 世界-动作：视觉和动作一起去噪。语言和特殊 token 在图里省略，实际推理时仍在 AR 前缀里。

验证：打开 cosmos-framework 的 `docs/inference.md` 的 Modes 表。`model_mode` 取 `text2image`、`text2video`、`image2video`、`video2video`、`forward_dynamics`、`inverse_dynamics`、`wam`，每个模式对应一份 `inputs/omni/*.json`。JSON 里没有神经网络结构字段，只有 `prompt`、`vision_path`、`action_path`、`domain_name`。换文件等于换配置。Reasoner 单独是另一张表面：只走 AR，扩散参数不激活，官方把它叫 VLM。

### 5.2 双塔 MoT：因果通路和去噪通路

Liang 等人 2024 年的 Mixture-of-Transformers（arXiv:2411.04996）把非嵌入参数按模态拆开：文本、图像、语音各有自己的 FFN、注意力投影和 LayerNorm，再用全局自注意力看整段序列。目的是省预训练 FLOPs。Chameleon 7B 设定下，他们宣称用约 55.8% 的 FLOPs 追上稠密基线。

Cosmos 3 借用了「一层里多套参数、共享注意力」这个名字，切分轴不一样。它不是按语言 / 图像 / 语音拆专家，而是按任务通路拆：Reasoner 一套，Generator 一套。报告 2.3 节写，两套都从预训练 VLM 初始化（Nano 用 Qwen3-VL 8B，Super 用 Qwen3-VL 32B，Edge 用他们自己训的 2B 稠密 Transformer），然后 AR token 进 Reasoner 塔，扩散 token 进 Generator 塔。

注意力是非对称的。记 AR 的 query/key/value 为 $\mathbf{Q}_{\mathrm{AR}},\mathbf{K}_{\mathrm{AR}},\mathbf{V}_{\mathrm{AR}}$，扩散侧为 $\mathbf{Q}_{\mathrm{DM}},\mathbf{K}_{\mathrm{DM}},\mathbf{V}_{\mathrm{DM}}$。Reasoner 只在自己内部做因果注意：

$$
\mathbf{O}_{\mathrm{AR}}=\operatorname{Attn}_{\mathrm{causal}}(\mathbf{Q}_{\mathrm{AR}},\mathbf{K}_{\mathrm{AR}},\mathbf{V}_{\mathrm{AR}})
$$

Generator 做双向注意，key/value 是 AR 和扩散的拼接：

$$
\mathbf{O}_{\mathrm{DM}}=\operatorname{Attn}_{\mathrm{full}}(\mathbf{Q}_{\mathrm{DM}},[\mathbf{K}_{\mathrm{AR}};\mathbf{K}_{\mathrm{DM}}],[\mathbf{V}_{\mathrm{AR}};\mathbf{V}_{\mathrm{DM}}])
$$

扩散因此能读到提示词和 ViT 理解 token；文字生成保持自回归，不被尚未去噪完的视频污染。类比：编剧（Reasoner）按时间顺序写台词，摄影师（Generator）拍整场戏时可以回头看剧本，但不能改已经写下的字。类比失效处：这里没有两个独立模型在对话，只是同一层里两套投影矩阵。

代码落点在 cosmos-framework。`OmniMoTModel`（`cosmos_framework/model/generator/omni_mot_model.py`）是训练用的外壳，损失走 `compute_flow_matching_loss`。真正的双塔网络是 `Cosmos3VFMNetwork`（`cosmos_framework/model/generator/mot/cosmos3_vfm_network.py`）。配置字段 `joint_attn_implementation` 默认 `"two_way"`，和报告的双流联合注意力对应。打包时 `get_causal_seq` 取出因果段，`get_full_only_seq` 取出只做全注意的扩散段。Reasoner 单独解码走 `generate_reasoner_text`。

验证：读 `Cosmos3VFMNetworkConfig` 的开关。`vision_gen`、`action_gen`、`sound_gen` 决定扩散段有没有对应模态；Reasoner 推理可以 `load_vision_tokenizer=False`，不装 VAE。官方 README 也把运行时分成两张表面：Reasoner 输入文本和视觉、输出文本；Generator 输入文本、视觉、声音、动作，输出视觉、声音、动作。两张表面共享检查点，不共享这次前向激活的塔。

### 5.3 五种模态怎样变成同一条序列

眼睛、耳朵、关节量纲完全不同。硬拼进一个 Transformer 之前，要先投影到同一隐藏维，再告诉位置编码「这是第几秒、画面的哪一块」。

视觉用两套编码器，职责分开。理解走 ViT：patch $16\times 16$，再经两层 MLP 把 $2\times 2$ token 合成一个，投影进 Transformer；训练时 ViT 和主干一起更新。生成走冻结的 Wan2.2-TI2V-5B 视频 VAE：时间压缩 $4\times$，空间压缩 $32\times$（先 $16\times 16$ 再 $2\times 2$ patch merge）。Reasoner 看的是语义对齐过的格子，Generator 去噪的是可解码回像素的 latent。不要把两者当成同一个 $z$。

音频只服务生成。立体声 48 kHz，hop 1920 样本，大约每秒 25 个 token。音频 VAE 冻结。官方 README 有脚注：声音跟视频一起出，不是独立的文生音频模型。Edge 检查点没有声音分词器，`enable_sound` 那几份 JSON 对 Edge 不适用。

动作被写成身体无关的几何。报告 2.1.3 节：相机和车只用自车位姿；第一人称数据用头部位姿差当自车、手腕位姿差当执行器、指尖在腕坐标系里的位置当抓取状态；机器人用相机位姿差、末端法兰位姿差、夹爪开合。相邻 $\mathrm{SE}(3)$ 位姿之间的伪动作是相对变换

$$
\Delta\mathbf{T}_t=\mathbf{T}_{t-1}^{-1}\mathbf{T}_t
$$

旋转用 Zhou 等人的 6D 表示（旋转自由度是 3，6D 是过参数化，解码时 SVD 收回 $\mathrm{SO}(3)$）。抓取状态不取差分，直接写当前开合或指尖。不同身体的向量长度不同，用域相关的输入输出投影接到同一隐藏维：

$$
\mathbf{z}=\mathbf{W}_{\mathrm{in}}^{(k)}\mathbf{x}+\mathbf{b}_{\mathrm{in}}^{(k)},\qquad
\mathbf{x}=\mathbf{W}_{\mathrm{out}}^{(k)}\mathbf{z}+\mathbf{b}_{\mathrm{out}}^{(k)}
$$

$k$ 是身体编号。代码里对应 `DomainAwareLinear`（`cosmos_framework/model/generator/mot/domain_aware_linear.py`），以及 `Cosmos3VFMNetwork.pack_action` / `unpack_action`。Hugging Face Nano 卡片列出的维数包括：相机运动 9D、自动驾驶 9D、第一人称 57D、单臂 Franka+RobotiQ 10D、双臂 20D、Agibot 29D，以及 UR / Google robot / WidowX / UMI 等 9D 或 10D 变体。桌宠的「看左 / 伸手 / 不动」四个离散符号不在这张表里，不要幻想把第 30 课的按键直接喂给 Cosmos 3。

位置编码是 3D mRoPE，再加绝对时间调制。语言 token 的 $t=h=w$ 同步递增，退化成普通 1D RoPE。视频 token 一帧共享 $t$，格子走 $h,w$。音频和动作只走时间，$h=w=0$。不同 FPS 会对不齐：24 fps 和 60 fps 的「加 1」不是同一段物理时间。报告定义每秒时间步 $\mathrm{TPS}$，视频是帧率除以 VAE 的时间压缩 4，音频约为 $48000/1920\approx 25$，动作就是采样频率。基准取 24 fps 视频的 $\mathrm{TPS}_{\mathrm{base}}=24/4=6$，时间增量

$$
\delta t=\frac{\mathrm{TPS}_{\mathrm{base}}}{\mathrm{TPS}}
$$

另外，扩散段的时间下标相对 AR 最后一项平移 15000，避免第一个视频帧和最后一个字挤在几乎相同的时间嵌入上（大模型上会出现过饱和和棋盘格）。`Cosmos3VFMNetworkConfig.enable_fps_modulation` 和 `base_fps=24` 就是这套开关。

验证：官方 Input and Output 表写视觉条件分辨率（720p 用 1280×720，480p 用 832×480，256p 用 320×192），视频条件默认 5 帧。框架 FAQ 还提醒：`resolution: "256"` 加默认宽高比 16:9 得到的是 320×192，不是高 256。这是配置陷阱，不是模型坏了。

### 5.4 Generator 学的是速度场，Reasoner 学的是下一个词

Reasoner 是标准的多模态下一词预测。预训练约 22.0M 样本，SFT 约 2.2M，SFT 阶段视频-文本占比提到约 50%，面向机器人、驾驶、空间智能。预训练所有部件一起训，没有单独冻住主干只训投影的对齐阶段。序列上限 16k token，图像最多 2048 token，视频最多 8192。优化器 AdamW，预训练峰值学习率语言模型和投影 $5\times 10^{-5}$，ViT $5\times 10^{-6}$。这些数字来自报告 4.1 节，用来理解「Reasoner 先成为 VLM，再克隆给 Generator」，不是让你在 24GB 上复现。

Generator 的目标是 rectified flow matching（直线流匹配：在干净数据和噪声之间拉一条直线，网络学沿这条线的速度）。对任意模态的干净 latent $x_0$，噪声水平 $\sigma\in[0,1]$，构造

$$
x_{\sigma}=\sigma\cdot\epsilon+(1-\sigma)\cdot x_0,\qquad \epsilon\sim\mathcal{N}(0,I)
$$

网络 $v_{\theta}(x_{\sigma},\sigma,c)$ 预测恒定速度 $v^{\ast}=\epsilon-x_0$，损失是掩码后的均方误差。干净的条件帧不进损失。图像、音频、动作的 $\sigma$ 用 logit-normal，视频用 mode sampling。再经过 shift 重参数把边际推向高噪声。预训练三档分辨率 256p / 480p / 720p 的 shift 是 1 / 3 / 5，中训提到 3 / 5 / 10。动作损失额外乘 10，因为归一化后的动作 MSE 比视觉小一个数量级。

这和 [第 10 课](10_diamond_diffusion_world_model.md) 的像素扩散、[第 26 课](26_vla_vs_world_model.md) 的 $\pi_0$ 动作流匹配是同一家族：连续量用速度场，离散词用交叉熵。Cosmos 3 的特别处是同一套去噪器覆盖视觉、声音、动作，条件 $c$ 里可以有文本、干净视频、干净动作。

训练分阶段，报告 4.2 节。预训练：图像、视频、音频，任务包括 T2I、T2V、I2V、V2V，分辨率三档，上下文打包约 74k token。中训：加入动作（前向、逆向、策略合计约 25%）和视频迁移（边缘、模糊、深度、分割约 20%，驾驶场景图约 5%），视频+音频约 8%。后训练再切产品：Super 的 T2I 检查点、I2V 检查点、Nano 在 DROID 上的策略检查点。DROID 后训练吃 76k 条轨迹、约 350 小时、86 个任务、564 个场景；输入是本体感觉加三机位拼图（540×640），输出 32 步绝对关节位置加辅助 RGB，15 Hz；推理 4 步扩散、shift 5、CFG 3，跳过视频 latent 解码。官方写部署在 2 张 RTX Pro 6000 上。完整后训练标成只讲。

推理侧，Diffusers 路径用 `Cosmos3OmniPipeline` 加 `UniPCMultistepScheduler`，README 示例 `flow_shift=10.0`、`num_inference_steps=35`、`guidance_scale=6.0`。这是采样器超参，不是你要扫的网格。

验证：`OmniMoTModel` 文档字符串写明「MoT model to be trained with the flow matching objective for visual / sound / action generation」。损失文件就叫 `flow_matching.py`。不要把 Generator 的训练目标写成下一词预测。

### 5.5 同一套权重，四类产品，三问怎么答

把配置表和 [第 12 课](12_frontier_landscape.md) 三问叠在一起，答案会变得难听，但这正是本课要的。

只走 Reasoner：输入文本加图像或视频，输出文本。这是 VLM。第一问：没有逐步动作端口，动作对换没有定义。第二问：VLM 可以描述「杯子还在桌上」，那是语言判断，不是内部状态在滚。第三问：规划如果存在，也是把轨迹写成文字，不是在模型里 `env.step`。记 E0。

T2I / T2V / I2V / V2V / 带声音的视频：视频或图像生成器。条件是文本和可选首帧，没有 $a_t$。第一问失败。生成真可以很高，报告把后训练 T2I / I2V 送去 Artificial Analysis，那是宣称。记 E0。

forward dynamics：世界模拟器。条件里有观察和动作，输出未来视频。第一问在接口上成立。你在 24GB 上没有做对换，所以「成立」来自官方 JSON 的 `action_path` 字段，不是你的 L2 曲线。第二问：扩散段双向看完整窗口，不是 RSSM 那种压缩信念；长时程物体恒常报告自己在 Limitations 里承认会漂、会变形。第三问：官方样例是离线播，不是在真机上搜动作。记 E1。

inverse dynamics：从视频反推动作。它学的是 $P(a_t\mid v_{t-1},v_t)$，不是 $P(v_t\mid v_{t-1},a_t)$。对规划有用，但单独不是世界模型。不要把它填进「世界模拟器」那一格。

`wam` / 策略：世界-动作模型或策略头。联合去噪动作和视频时，接口上同时有 $a$ 和未来画面。若只保存 `sample_outputs.json` 和 `vision.mp4`，仍是 E1。DROID 后训练检查点 `Cosmos3-Nano-Policy-DROID` 在真机上吐关节位置，选动作时跳过视频解码，行为更接近 [第 26 课](26_vla_vs_world_model.md) 的 VLA：输出是 $a$，不是查询 $P(s_{t+1}\mid s_t,a_t)$ 再决定。按第 33 课，开环真机策略记 E3，不能因为 RoboArena 第一（宣称）就升到 E4。E4 要日志里看得到「查询世界模型之后动作被改写」。

Edge / Nano / Super 是规模，不是档。4B 的 E0 仍是 E0。64B 的离线前向动力学仍是 E1。

下一课会专门问声音：模型会出声，不等于它把杯子碰桌沿当成观测通道。本课只需要记住官方脚注：声音跟视频走，Edge 没有这条通道。

## 6. 源码导读

两个仓库分工不同。`NVIDIA/cosmos` 是产品入口：README 的模型家族表、Quickstart、cookbook 笔记本、推理延迟表。`NVIDIA/cosmos-framework` 是可跑的训练和推理框架：Python 包 `cosmos_framework/`。本课导读以 framework 为主，cosmos 仓库用来核对命令和显存。不要引用你没打开过的路径。下面这些是当前 main 分支里真实存在的文件。

先按这个顺序读，每个文件带着问题进去：

| 文件 | 是哪个零件 | 带着什么问题读 |
|---|---|---|
| `NVIDIA/cosmos` 的 `README.md` | 产品说明书 | Reasoner / Generator 两张表面的输入输出？Nano 推荐硬件写的是哪几张卡？声音脚注怎么写？ |
| `cosmos-framework/docs/inference.md` | 推理协议 | `model_mode` 有哪几种？`forward_dynamics` 必填哪些字段？Edge 缺哪一种模式？ |
| `cosmos-framework/docs/faq.md` | 显存与陷阱 | Nano 和 Super 的 GPU Memory 各写多少？`resolution: "256"` 实际像素是多少？ |
| `inputs/omni/t2v.json` | 无动作配置 | 有没有 `action_path`？prompt 是短句还是结构化长描述？ |
| `inputs/omni/action_forward_dynamics_robot.json` | 有动作配置 | `model_mode`、`domain_name`、`action_path`、`vision_path` 分别是什么？ |
| `inputs/omni/action_policy_robot.json` | 世界-动作配置 | `model_mode` 为什么叫 `wam`？和前向动力学比多了什么、少了什么？ |
| `cosmos_framework/scripts/inference.py` | CLI 入口 | `-i`、`--checkpoint-path`、`--parallelism-preset` 怎么进前向？ |
| `cosmos_framework/inference/model.py` | 权重装载 | `OmniMoTModel` 从哪 import？Diffusers 键名怎样映射到 `action2llm` / `vae2llm`？ |
| `cosmos_framework/model/generator/omni_mot_model.py` | 训练外壳 | 分词器为什么分成 `tokenizer_vision_gen` 和 VLM processor？损失调用哪一个函数？ |
| `cosmos_framework/model/generator/mot/cosmos3_vfm_network.py` | 双塔网络 | `pack_action` 和 `generate_reasoner_text` 是否走同一条 `forward`？`joint_attn_implementation` 默认值？ |
| `cosmos_framework/model/generator/algorithm/loss/flow_matching.py` | Generator 损失 | 速度目标怎样从 $x_0$ 和 $\epsilon$ 构造？条件 token 如何被 mask？ |
| `cosmos_framework/model/generator/mot/domain_aware_linear.py` | 身体投影 | 域编号 $k$ 怎样选中 $\mathbf{W}_{\mathrm{in}}^{(k)}$？ |
| `cosmos_framework/inference/defaults/` 下各 `sample_args.json` | 每模式默认采样 | text2video 和 forward_dynamics 的默认步数、shift 是否相同？ |
| `NVIDIA/cosmos` 的 `cookbooks/cosmos3/` | 官方 notebook | Diffusers、vLLM-Omni、Reasoner、SFT、蒸馏各在哪个子目录？ |

三处最要紧的细节，读的时候记下来。

第一，检查点名字按双塔合计数，FAQ 按单塔数。Hugging Face 卡片写 Cosmos3-Nano 16B、Super 64B、Edge 4B。framework FAQ 写 Nano (8B) 需要 32 GB，Super (32B) 需要 128 GB。报告表 2 写 Nano 建在 8B 稠密 Transformer 上、Super 建在 32B 上、Edge 建在 2B 上，每层两套参数。三份文档说的是同一件事：塔的宽度按 Qwen3-VL 骨干计，总参数大约翻倍。写笔记时两套数字都保留，不要把「8B 就能 24GB」从 FAQ 的括号里读出来。

第二，前向动力学的样例不是桌宠动作。`inputs/omni/action_forward_dynamics_robot.json` 当前内容是：`model_mode` 为 `forward_dynamics`，`domain_name` 为 `bridge_orig_lerobot`，`action_chunk_size` 为 16，`fps` 为 5，提示语 `Put the pot to the left of the purple item.`，`vision_path` 和 `action_path` 指向 `nvidia-cosmos/cosmos-dependencies` 上的 Bridge 样例。这是官方可复述的动作条件接口，身体是桌面臂，不是第 30 课的四个按键。

第三，Diffusers 和 framework 装的是同一份 Omni 检查点，键名不同。`inference/model.py` 里有一长串正则：`action_proj_in` 改成 `action2llm`，`proj_in` 改成 `vae2llm`，`audio_proj_in` 改成 `sound2llm`。看到 `moe_gen` 后缀的 q/k/v 投影，那是 Generator 塔的注意力矩阵，不是 Mixture-of-Experts 路由。读到这里就能把报告图 5 的双塔对上权重文件。

对照第 22 课只读、不跑的那份仓库：`nvidia-cosmos/cosmos-predict2.5` 的 `docs/inference.md` 和 `docs/inference_robot_action_cond.md`。Predict2.5 的基础入口是 `examples/inference.py`，条件是提示文本加图像或视频；动作在后训练脚本 `examples/action_conditioned.py`。Cosmos 3 把动作模式写进同一份 `scripts/inference.py` 的 `model_mode`。这是本课对照笔记的代码证据，不是「Cosmos 升级了所以第一问自动过关」。

SFT 和蒸馏 cookbook 在 cosmos README 的 Finetune / Distill 两节。Vision SFT 的 Nano 脚本是 `cookbooks/cosmos3/generator/audiovisual/finetune/launch_sft_vision_nano.sh`，官方写明 `bash launch_sft_<recipe>.sh` 准备数据并在 8×H100 上训练。Policy-DROID 对应 `launch_sft_action_policy_droid_nano.sh`。这些命令本课只抄进预算表，不执行。

## 7. 实验

主线没有 GPU 训练，也没有 24GB 上的 Nano 推理。实验要证明的是：你能根据官方配置判断产品类型，能用官方样例视频看出「有动作条件」和「无动作生成」不是同一份输出，能按第 12 课三问和第 33 课档位打分。加餐推理单独放在 Step 6，硬件不够就跳过，验收不扣分。

建议在一个空目录做，例如 `~/learn-wm/l35-cosmos3`。下面每个 bash 围栏一条命令。

### Step 0: 克隆两个锚定仓库，核对 README 表

```bash
git clone https://github.com/NVIDIA/cosmos.git
```

```bash
git clone https://github.com/NVIDIA/cosmos-framework.git
```

```bash
cd cosmos
```

打开 `README.md` 的 Cosmos 3 一节，抄下三张表：Reasoner / Generator 表面、Model Family（Super 64B / Nano 16B / Edge 4B）、Supported Generation Settings、Input and Output。再打开 `inference_benchmarks.md` 的目录，确认 Nano Generator 的延迟表里出现的卡是 RTX PRO 6000、H20、H100、H200、B200 这一档，没有 24GB 消费卡。

```bash
cd ../cosmos-framework
```

打开 `docs/faq.md`，找到「How much GPU memory do the models need?」。把 Nano 32 GB、Super 128 GB 抄进笔记。打开 `docs/inference.md` 的 Modes 表，列出全部 `model_mode`。打开 `docs/setup.md` 的 System Requirements：Ampere 或更新、CUDA >= 12.8、Linux、建议约 150 GiB 磁盘。这些是本课硬件判断的依据，不要用社交媒体上的「Nano 能上 4090」替换官方表。

预期：两个仓库都在，README 和 FAQ 的显存数字互相吻合（Nano 按单塔 8B 计约 32 GB）。若 `git clone` 失败，改用浏览器打开同一 URL 读，把「未克隆、已读网页」写进笔记即可继续 Step 1。

### Step 1: 列出官方输入-输出组合并分类

在笔记里建表 `cosmos3_io_config.md`，按下表抄官方组合，第三列由你填。依据只能是 README Use Cases、inference.md Modes、报告 2.2 节，不要发明官方没写的「文本进、关节出、同时出声音」之类组合。

| 官方工作流 | 输入 | 输出 | 你标的产品类 | 三问（动作 / 状态 / 规划） | 第 33 课档 |
|---|---|---|---|---|---|
| Reasoner Caption / Grounding / CoT | 文本 + 图像或视频 | 文本 |  |  |  |
| Text-to-image | 文本 | 图像 |  |  |  |
| Text-to-video | 文本 | 视频 |  |  |  |
| Text-to-video with sound | 文本 | 视频 + 声音 |  |  |  |
| Image-to-video | 文本 + 图像 | 视频 |  |  |  |
| Video-to-video | 文本 + 视频 | 视频 |  |  |  |
| Video transfer（深度 / 边缘等） | 文本 + 控制视频 | RGB 视频 |  |  |  |
| Forward dynamics | 文本 + 视觉 + 动作 | 视频 |  |  |  |
| Inverse dynamics | 文本 + 视频 | 动作 |  |  |  |
| Action policy / wam | 文本 + 视觉 | 动作 + 视频 |  |  |  |
| Nano-Policy-DROID（后训练） | 指令 + 多机位 + 本体感觉 | 关节动作（辅助视频可关） |  |  |  |

参考答案（先自己填，再对）：Reasoner 和所有无动作生成是 VLM 或视频生成器，三问第一问否，档 E0。Forward dynamics 是世界模拟器，第一问接口上是，第二问未测，第三问离线，档 E1。Inverse dynamics 是逆动力学头，不要标成世界模拟器。wam 在离线 JSON 上是世界-动作模型、E1；DROID 策略检查点按开环真机策略更接近 E3，RoboArena 排名写「宣称」。Edge 没有声音、没有 video-to-video transfer，这两行对 Edge 标「该检查点不支持」。

把 `inputs/omni/` 里实际文件名对上表。当前仓库里前向动力学机器人样例是 `action_forward_dynamics_robot.json`，策略样例是 `action_policy_robot.json`（`model_mode` 为 `wam`），文生视频是 `t2v.json`。文件名以你 clone 到的版本为准。

### Step 2: 有动作条件 vs 无动作条件，对照官方样例，不跑权重

Nano 在 24GB 上不够。对照改用官方已经生成好的样例，证明两套配置的输出接口不同。Hugging Face 卡片 `nvidia/Cosmos3-Nano` 在 `assets/` 下挂了无动作视频和动作前向动力学视频。新建目录并下载（每条命令只取一个文件）：

```bash
mkdir -p ~/learn-wm/l35-cosmos3/assets
```

```bash
curl -L -o ~/learn-wm/l35-cosmos3/assets/example_t2v_diffusers_output.mp4 https://huggingface.co/nvidia/Cosmos3-Nano/resolve/main/assets/example_t2v_diffusers_output.mp4
```

```bash
curl -L -o ~/learn-wm/l35-cosmos3/assets/example_i2v_output.mp4 https://huggingface.co/nvidia/Cosmos3-Nano/resolve/main/assets/example_i2v_output.mp4
```

```bash
curl -L -o ~/learn-wm/l35-cosmos3/assets/example_action_fd_agibotworld_4chunk_output.mp4 https://huggingface.co/nvidia/Cosmos3-Nano/resolve/main/assets/example_action_fd_agibotworld_4chunk_output.mp4
```

```bash
curl -L -o ~/learn-wm/l35-cosmos3/assets/example_action_fd_agibotworld_first_frame.png https://huggingface.co/nvidia/Cosmos3-Nano/resolve/main/assets/example_action_fd_agibotworld_first_frame.png
```

```bash
curl -L -o ~/learn-wm/l35-cosmos3/assets/example_action_fd_agibotworld_action_chunks.json https://huggingface.co/nvidia/Cosmos3-Nano/resolve/main/assets/example_action_fd_agibotworld_action_chunks.json
```

用系统播放器打开两段视频，把 JSON 里的动作块打开看形状。笔记只要求三句话：

1. T2V / I2V 样例的条件是文本或首帧，资产目录里没有对应的动作 JSON。
2. 前向动力学样例同时有首帧、动作 JSON、输出视频；动作是按时间展开的数组，不是一句自然语言。
3. 你没有在同一随机种子下自己换动作重生成，因此不能填写对换 L2，只能填写「官方接口分岔，本课未测数值」。

这就是本课允许的「有 / 无动作对照」。不要把官方 demo 的观感写成第 03 课那种 `flip_l2`。Edge 卡片同样有 `assets/edge_action_fd_umi_2chunk_output.mp4` 和 I2V 样例，可以当第二组对照，不是必须。

若 `curl` 被墙，改用浏览器从模型卡页面下载同一路径，或只看卡片里已经嵌入的预览，把「未落盘、已看预览」写进笔记。

### Step 3: 模态路由板

下面这张板就是本课的互动实验。先勾选输入，再勾选输出，再判断产品类、三问缺口、档位。不要看第 5.5 节的现成答案直接抄。系统没有网页组件时，用纸或笔记完成。每一行只改勾选，其他列自己写。

| 题 | 输入（勾选） | 输出（勾选） | 更像哪类产品 | 三问哪一问没回答 | 档 |
|---|---|---|---|---|---|
| A | 语言 + 图像 | 语言 |  |  |  |
| B | 语言 | 视频 |  |  |  |
| C | 语言 + 图像 | 视频 |  |  |  |
| D | 语言 + 视频 + 声音 | 视频 + 声音 |  |  |  |
| E | 语言 + 图像 + 动作 | 视频 |  |  |  |
| F | 语言 + 视频 | 动作 |  |  |  |
| G | 语言 + 图像 | 动作 + 视频 |  |  |  |
| H | 语言 + 多机位 + 本体感觉 | 关节动作 |  |  |  |

标准口径：

- A：VLM。第一问没回答（没有逐步动作）。E0。
- B：视频生成器。第一问没回答。E0。
- C：图生视频，仍是视频生成器。第一问没回答。E0。第二问若有人用它测物体恒常，那是续写实验，不是本课测过。
- D：音画生成器。第一问没回答。声音是配乐通道还是观测通道，留给第 36 课。E0。
- E：世界模拟器。第一问接口上回答了。第二问（离开视野还在吗）没回答。第三问（有没有拿它选过动作并回真环境）没回答。E1。
- F：逆动力学。三问的第一问问的是预测是否随动作变，这里动作是输出不是条件，第一问仍没按世界模型的方式回答。不要标 E1 世界模拟器。
- G：世界-动作模型。第一问接口上回答了。若只离线保存视频和 JSON，第三问没回答，E1。
- H：策略头。更像 VLA。若执行时不查询前向动力学，第三问按「用世界模型选动作」来算是否。档最多 E3，除非日志证明先查询再改动作。

故意挖的坑：D 看起来「全模态」，仍可能是 E0。G 和 H 的输出都含动作，一个还在播世界，一个已经在当手。桌宠若直接部署 H，杯子会不会倒仍然没人问。

### Step 4: 写 Predict2.5 对照笔记

新建 `predict25_vs_cosmos3.md`，按下面六行写，每行给证据来源（第 22 课笔记、Predict2.5 仓库文档、本课 README / 报告）。

| 维度 | Cosmos-Predict2.5（第 22 课） | Cosmos 3（本课） |
|---|---|---|
| 论文 | arXiv:2511.00062；平台综述 arXiv:2501.03575 | arXiv:2606.02800 v4 |
| 产品切分 | predict / transfer / reason 三家仓库 | 同一套 MoT，Reasoner 与 Generator 是两张表面 |
| 基础检查点条件 | 文本 + 图像或视频 | 文本、视觉、声音、动作按 `model_mode` 组合 |
| 动作端口 | 后训练 `robot/action-cond`，另有 `robot/policy` | 中训起就有前向 / 逆向 / 策略三种扩散布局 |
| 声音 | 基础 Predict2.5 不当作一等公民生成通道 | Nano / Super 可随视频出 48 kHz 立体声；Edge 无 |
| 主线 24GB | HF 卡片 Video2World 约 32.54GB，只讲 | framework FAQ Nano 约 32GB，只讲 |

再补一段话：Predict2.5 把 Cosmos-Reason 当文本编码器；Cosmos 3 把 Reasoner 权重克隆成 Generator 的初始化，然后用 flow matching 训去噪塔。两者都用流匹配，但 Cosmos 3 的流匹配覆盖视觉、声音、动作，不只是 Video2World。

### Step 5: 给两种配置打 E0 / E1

打开 [第 33 课](33_embodiment_degrees.md) 第 5.6 节。对「官方 T2V 样例」和「官方 forward dynamics 样例」各答七条是否题里用得上的 Q1、Q2。本课没有把模型接到 `env.step`，Q2（模型是否在回路）一律否。

预期填写：T2V 的 Q1 否，档 E0。forward dynamics 的 Q1 在接口上是、Q2 否、Q3（物理世界）否或未接，档 E1。DROID 策略检查点若只读模型卡、没有真机日志，不要写成 E4。把「Artificial Analysis / RoboArena 是宣称」写在表下。

### Step 6: 加餐推理（80GB 级卡才走，24GB 停）

本步不是验收。官方 Cosmos3-Nano Diffusers 示例如下，来自 `NVIDIA/cosmos` README 的 Generator with Diffusers。先建 Python 3.13 环境并安装（命令以 README 为准，版本会变）：

```bash
uv venv --python 3.13 --seed --managed-python
```

激活虚拟环境之后再装依赖。下一围栏是一条长 `uv pip install`，中间不要拆成多条以免和官方不一致：

```bash
uv pip install --torch-backend=auto "diffusers @ git+https://github.com/huggingface/diffusers.git" accelerate av cosmos_guardrail huggingface_hub imageio imageio-ffmpeg torch torchvision transformers
```

Python 推理脚本不要一次性塞进 bash。按 README 用 `Cosmos3OmniPipeline.from_pretrained("nvidia/Cosmos3-Nano", torch_dtype=torch.bfloat16, device_map="cuda")`，调度器 `UniPCMultistepScheduler`，`flow_shift=10.0`，`num_frames=189`，`num_inference_steps=35`，`guidance_scale=6.0`。Guardrail 默认开启；没有申请 gated 仓库会在这一步失败。

framework 单卡入口（同样按 FAQ 需要约 32 GB，不是 24GB）：

```bash
python -m cosmos_framework.scripts.inference --parallelism-preset=latency -i "inputs/omni/t2v.json" -o outputs/omni_nano --checkpoint-path Cosmos3-Nano --seed=0
```

有动作对照若硬件够，把 `-i` 换成 `inputs/omni/action_forward_dynamics_robot.json`，再跑一次无动作的 `t2v.json`，只比较接口和输出文件类型（`vision.mp4` vs 带 `sample_outputs.json`），不报 RoboArena 数字。OOM 就停，对照 FAQ 的 32 GB 一行，不要关 guardrail 硬挤。

Edge 的 framework 文档写「compact 2B omni model. It fits comfortably on a single GPU」，示例是 `--checkpoint-path Cosmos3-Edge` 和 `inputs/omni/t2i.json`。官方延迟表包含 Jetson 16GB 统一内存上的图生视频，不包含 RTX 4090 24GB。主线不要把 Edge 写成「24GB 必做体验」。

## 8. 配置与预算

数字一律来自当前 README、FAQ、报告或 Hugging Face 卡片。不确定的超参不写。本课主线不训练。

模型家族（报告表 2 与 Hugging Face 卡片一致，FAQ 用单塔宽度称呼）：

| 变体 | 双塔合计 | 单塔骨干 | LLM 层数 | Hidden | 注意力头 / KV 头 | 初始化 | 本课档位 |
|---|---|---|---|---|---|---|---|
| Cosmos3-Edge | 4B | 2B | 28 | 2048 | 16 / 8 | 自训稠密 LLM（报告附录 D） | 文档必读；单卡推理官方未给 24GB 数字，不作为必做体验 |
| Cosmos3-Nano | 16B | 8B（Qwen3-VL 8B） | 36 | 4096 | 32 / 8 | Qwen3-VL 8B | 只讲。FAQ：推理约 32 GB |
| Cosmos3-Super | 64B | 32B（Qwen3-VL 32B） | 64 | 5120 | 64 / 8 | Qwen3-VL 32B | 只讲。FAQ：推理约 128 GB；单卡 80GB H100 装不下完整 Super |

公开检查点（Hugging Face collections/nvidia/cosmos3，写课时集合页列出 Super / Nano / Edge / Action-Viewer；各变体卡片另列后训练和蒸馏权重）：

| 名字 | 角色 | 本课 |
|---|---|---|
| `nvidia/Cosmos3-Nano` | 通用 Omni | 只讲 |
| `nvidia/Cosmos3-Super` | 通用 Omni，偏数据中心 | 只讲 |
| `nvidia/Cosmos3-Edge` | 端侧，无声音，无 V2V transfer | 只讲 |
| `nvidia/Cosmos3-Nano-Policy-DROID` | DROID 策略后训练 | 只讲 |
| `nvidia/Cosmos3-Super-Text2Image` 与 `...-4Step` | T2I 后训练 / DMD2 四步蒸馏 | 只讲 |
| `nvidia/Cosmos3-Super-Image2Video` 与 `...-4Step` | I2V 后训练 / 四步蒸馏 | 只讲 |
| `nvidia/Cosmos3-Edge-Policy-DROID` | Edge 的 DROID 策略 | 只讲 |

生成设置（Nano / Super 的 README 表；Edge 另有限制）：

| 项 | 官方支持 | 默认 | 本课 |
|---|---|---|---|
| 分辨率档 | 256p、480p、720p | 480p | 不跑 |
| 宽高比 | 16:9、4:3、1:1、3:4、9:16 | 16:9 | 256 档 16:9 实际 320×192 |
| 帧率 | 10、16、24、30 | 24 | 不跑 |
| 帧数 | 5 到 300 | 189（约 7.9 秒 @ 24 fps） | Edge 默认 121，支持 50–150 帧、256p 与 480p |
| 精度 | 官方测 BF16 | BF16 | FP8 / NVFP4 卡片写 coming soon 或 NIM 路径，本课不写 |
| Diffusers 示例步数 | 35 | `guidance_scale=6.0`，`flow_shift=10.0` | 加餐才用 |
| DROID 策略推理 | 4 步，shift 5，CFG 3 | 跳过视频解码 | 只讲 |
| 操作系统 | Linux |  | Mac 走阅读路径 |
| GPU 架构 | Ampere、Hopper、Blackwell |  | 主线 24GB Ampere 不够 Nano |

训练预算（报告，只读）：Reasoner 预训练约 22.0M 样本、SFT 约 2.2M。Generator 中训 Nano 约 2.4T token、1024 张 GB200；Super 约 1.9T token、2048 张 GB200。DROID 后训练 76k 轨迹。SFT cookbook 官方写 8×H100。课程硬件分档把「35 课 Cosmos 3 Nano 后训练」放在 8×A100 短租加餐，且必须官方 cookbook 声明单机可做才写命令；当前 cookbook 是 8×H100，本课不写你能在 24GB 上 SFT 16B。

时间量级（主线，无训练）：

| 环节 | 大概耗时 | 说明 |
|---|---|---|
| 克隆两个仓库 | 数分钟 | 只要 README 和 docs，不必拉 LFS 权重 |
| 读报告 2、4、6.2.5 节 | 3-5 小时 | 重点是 token 布局、MoT、动作表示、DROID 后训练 |
| 下载官方样例视频 | 视网速 | 几个 MP4，远小于 90 GiB 的权重缓存 |
| 填配置表 + 路由板 + 对照笔记 | 2-4 小时 | 本课真正的作业 |
| 加餐 Nano 推理 | FAQ 约 32 GB 起 | 24GB 不要试；首次还要下权重和 Guardrail |
| 加餐 8×H100 SFT | 按 cookbook | 只讲 |

磁盘：主线几百 MB。加餐按 setup 文档约 150 GiB。许可证 OpenMDW-1.1。Generator 还依赖 gated Guardrail 仓库，名称在 cosmos README 里是 `nvidia/Cosmos-1.0-Guardrail`，在 framework 文档里是 `nvidia/Cosmos-Guardrail1`，申请时按你走的那条路径打开对应卡片。

## 9. 验收

- [ ] 能不看笔记画出 AR 子序列在前、扩散子序列在后，并说出 Reasoner 因果、Generator 双向、AR 不回看扩散。
- [ ] `cosmos3_io_config.md` 覆盖 README 列出的工作流；每一行有产品类、三问、档位；Edge 不支持的声音和 V2V transfer 有标注。
- [ ] 无动作生成标 E0，forward dynamics 离线播标 E1，没有把 RoboArena 或 Artificial Analysis 写成你复现的结果。
- [ ] `predict25_vs_cosmos3.md` 至少六行：论文号、产品切分、基础条件、动作端口、声音、24GB 显存；Predict2.5 和 Cosmos 3 没有抄成同一个检查点。
- [ ] 模态路由板 A-H 都有答案；D 没有因为「全模态」升档；H 没有因为「会吐动作」升到 E4。
- [ ] 官方样例对照写明：T2V 资产无动作 JSON，前向动力学资产有动作数组；未测对换 L2。
- [ ] 能指出动作表示是相对位姿加域相关投影，桌宠四键不能直接当 `action_path`。
- [ ] 能指出 Generator 损失在 `flow_matching.py`，Reasoner 是下一词预测，两套目标不要写反。
- [ ] 24GB 上没有伪造的 Nano 推理日志；若加餐 OOM，笔记引用 FAQ 的 32 GB，而不是关安全检查硬跑。
- [ ] 桌宠结论写明：24GB 默认仍用第 30 课小模型。麦克风和喇叭可以进 Cosmos 3，那不是本课毕业条件。

## 10. 排错

| 症状 | 原因 | 验证 | 修法 |
|---|---|---|---|
| 想在 24GB 上 `from_pretrained("nvidia/Cosmos3-Nano")` 立刻 OOM | FAQ 写 Nano 约 32 GB；16B BF16 权重大约就占满一张 24GB | 对照 `docs/faq.md` 显存表 | 停。标只讲。不要 `--enable-layerwise-offload` 硬挤当成本课体验 |
| Diffusers 报 Guardrail 无权限 | Generator 默认安全检查，仓库 gated | Hugging Face 该模型卡是否 Accepted | 申请对应 Guardrail 卡，或只走本课 Step 0-5 |
| `resolution: "256"` 得到 320×192 | 档名不是像素高，默认宽高比 16:9 | 对照 `docs/inference.md` Resolution tiers | 需要正方形时设宽高比 `1,1` |
| 把 Predict2.5 的 `examples/inference.py` 当成本课命令 | 两套仓库、两套检查点 | URL 是 `nvidia-cosmos/cosmos-predict2.5` 还是 `NVIDIA/cosmos` | 回到本课 README Quickstart |
| `model_mode=wam` 却当成纯 VLA | `action_policy_robot.json` 仍要出 `vision.mp4` | 读 `docs/inference.md` 的 Outputs 列 | 离线 wam 是世界-动作，DROID 后训练关视频解码才更像策略头 |
| 用桌宠四键填 `action_path` | 官方动作维按身体注册，Bridge 样例是连续向量 | 打开 `action_forward_dynamics_robot.json` 的 `domain_name` | 不要改 JSON 硬接第 30 课按键 |
| Edge 开 `enable_sound` 失败 | Edge 无声音分词器 | `docs/inference.md` Cosmos3-Edge 段 | 声音行改走 Nano 文档，或等第 36 课 |
| Super 单卡 80GB OOM | 文档写 Super 不拟合单卡 80GB H100 | `docs/inference.md` Multi-GPU | 4 或 8 卡 FSDP，本课不要求 |
| `git clone` 很慢 | 仓库含 cookbook 资源 | `du -sh` | 只需 README 和 docs 时，用浏览器读亦可 |
| 把报告表 1 的 91.36 抄进自己的复现报告 | 那是后训练 T2I 的 UniGenBench，带星号 | 表注 `*` 表示后训练变体 | 写成「报告宣称」，不要写成「我测得」 |
| vLLM 只起 Reasoner 却期望出视频 | Reasoner 表面只出文本 | README：理解任务用 Reasoner with vLLM | 视频走 Diffusers / vLLM-Omni / framework inference |
| CUDA 驱动过旧，`torch.cuda.is_available()` 为 False | cosmos README Troubleshooting：uv 默认可能装到过新的 CUDA 轮子 | `nvidia-smi` 与 `python -c "import torch; print(torch.version.cuda)"` | 加餐才处理；按 README 用 `--torch-backend=cu128` 等匹配项 |

## 11. 前沿与改造

同一问题，2024-2026 年公开系统给出了三种不同的「把理解和生成放在一起」的做法。本课只引用核对过的材料。

Liang 等人的 MoT（arXiv:2411.04996）按模态拆非嵌入参数，共享全局自注意力，省的是预训练 FLOPs。Cosmos 3 按 Reasoner / Generator 两条通路拆，共享 mRoPE 和联合注意力，省的是「理解完再另训一个视频模型」的产品碎片。切分轴不同，不要把 Cosmos 3 写成 Liang 论文的官方实现。

Causal-rCM（Zheng 等人，arXiv:2606.25473，仓库 [NVlabs/rcm](https://github.com/NVlabs/rcm)）把 teacher forcing 和 self-forcing 收成一套自回归扩散蒸馏配方，宣称把 Cosmos 3 这种带动作条件的 omnimodal 基础模型蒸馏成可交互世界模型。双向 Teacher 画质高，不能边看边播；交互要因果、少步。这正好是第 37 课训练配方的主题。本课只记一句：Generator 预训练是整段双向去噪，官方自己也承认还要后训练或蒸馏才能当流式世界模型。Cosmos README 的 Distill 节目前展示的是 Super T2I / I2V 的 DMD2 四步学生（Improved Distribution Matching Distillation，arXiv:2405.14867），用公开样例演示流程，并写明不是生产级复现配方。

第 22 课的 Predict2.5、UniSim、GAIA-1 仍是对照行。Predict2.5 把动作放后训练；UniSim 从一开始就把语言和连续控制编进动作嵌入，历史却是滑窗；GAIA-1 的动作是速度和曲率两个 token。Cosmos 3 的动作是多身体几何，中训占比约 25%。规模上，报告写 Generator 中训在 GB200 集群上吃 T 级 token，DINO-WM 和你第 30 课的桌子不是同一个量级。钱能买规模，买不来「24GB 桌宠已经在回路里查询它」。

机制差距有三处。一处是接口：全模态权重默认仍可当 E0 生成器，必须显式走到 `forward_dynamics` 或 `wam`，第一问才在接口上成立。二处是推理成本：像素去噪一步很贵，测试时做 CEM 300 次 rollout 在本课硬件上不成立，这和第 22 课读 Predict2.5 时的结论同方向。三处是验证者：Artificial Analysis 和 RoboArena 测生成真与策略成功率，桌宠要的是动作对换和碰撞截断。

**动手改造清单**（24GB 主线只做 1 和 2；3、4 是短租加餐）：

1. 配置消融（无训练）。把 Step 1 的表按「去掉动作列」重填一遍，看有多少行从世界模拟器掉回视频生成器。预算：一小时。预期：T2V、I2V、Reasoner 不变，forward dynamics 和 wam 降档。失败判据：去掉动作之后你仍把 T2V 写成 E1，说明档位在看画质。
2. 样例对照加密。除 AgibotWorld 前向动力学外，再看 Nano 卡片里的逆动力学驾驶样例（`example_action_id_av_0_input.mp4` 与 `example_action_id_av_0_output.json`）。预算：半小时。预期：逆动力学输出是 JSON 动作，不是未来视频；产品类必须和前向动力学分开。失败判据：把两段都标成世界模拟器。
3. 加餐：同一份 Bridge 样例上跑 `forward_dynamics` 两次，第二次把 `action_path` 换成全零或打乱（需 32GB 以上）。预算：两次推理。预期：输出视频在机械臂轨迹上可见分岔。失败也是结果：若 4 步或默认 seed 下看不出差别，写「本硬件未观察到分岔」，不要声称 Cosmos 3 动作盲。
4. 加餐蒸馏。按 cosmos README Distill 节跑 `cookbooks/cosmos3/generator/audiovisual/distill/launch_distillation_t2i.sh` 的短流程（官方 8×H100 级）。预算：短租。预期：学生检查点步数变成 4，画质相对教师下降或接近，以你自己的两张图为准。失败判据：把脚本打印的 loss 当成 Artificial Analysis 排名。

论文结论「灵活的输入输出让一个骨架覆盖 VLM、视频生成器、世界模拟器、世界-动作模型」对应改造 1，缩小设置能复现的是分类变化，不能复现表 1 分数。论文结论「后训练可把同一骨架切成 T2I / I2V / 机器人策略」对应只读 4.2.3-4.2.5 节，24GB 不能复现。论文结论「RoboArena 最佳开源策略」对应宣称栏，本课映射为「规划好这根尺子上他们提交过榜，你没有跑」。

## 12. 论文与延伸

1. Cosmos 3（NVIDIA，[arXiv:2606.02800](https://arxiv.org/abs/2606.02800)，v4）。带着问题读：2.2 节三种动作布局怎样对应 `forward_dynamics` / `inverse_dynamics` / `wam`？2.3 节为什么 AR 不许看扩散 token？2.4 节时间平移 15000 和 FPS 调制各解决什么伪影？4.2.5 节 DROID 后训练关视频解码之后，还算不算世界-动作模型？表 1 带星号的列是基础模型还是后训练变体？Limitations 承认了哪些物理失败，和桌宠「碰杯就停」差在哪？
2. Mixture-of-Transformers（Liang 等，[arXiv:2411.04996](https://arxiv.org/abs/2411.04996)）。带着问题读：他们按模态拆参数，Cosmos 3 按通路拆参数，共享注意力这一点是否同构？Chameleon 与 Transfusion 两个设定里省下的 FLOPs，能不能直接拿来解释 Cosmos 3 为什么要双塔？
3. Cosmos-Predict2.5（[arXiv:2511.00062](https://arxiv.org/abs/2511.00062)）与 Cosmos 平台综述（[arXiv:2501.03575](https://arxiv.org/abs/2501.03575)）。带着问题重读第 22 课列过的问题：Predict / Transfer / Reason 三家如何被 Cosmos 3 收进 MoT？基础 Video2World 的 32.54GB 和本课 FAQ 的 32 GB 是不是同一张卡的故事？
4. Causal-rCM（Zheng 等，[arXiv:2606.25473](https://arxiv.org/abs/2606.25473)）。带着问题读：摘要里「apply Causal-rCM to Cosmos 3 … enabling an interactive world model」具体改了训练时看真帧还是看自己刚生成的帧？这和第 03 课曝光偏差、第 37 课 self-forcing 是不是同一类修补？仓库 [NVlabs/rcm](https://github.com/NVlabs/rcm) 当前 README 把 Cosmo 3 交互放在哪一节？
5. Flow matching（Lipman 等，[arXiv:2210.02747](https://arxiv.org/abs/2210.02747)）。带着问题读：报告 4.2 节的 $x_{\sigma}=\sigma\epsilon+(1-\sigma)x_0$ 和原文的线性插值是否同一条直线？为什么视频用 mode sampling、动作却继承视觉的噪声日程？
6. 6D 旋转表示（Zhou 等，CVPR 2019，常用引用 [arXiv:1812.07035](https://arxiv.org/abs/1812.07035)）。带着问题读：为什么连续旋转不用欧拉角？Cosmos 3 解码时 SVD 收回 $\mathrm{SO}(3)$，少了这一步会怎样？
7. DROID（Khazatsky 等，[arXiv:2403.12945](https://arxiv.org/abs/2403.12945)）。带着问题读：76k 轨迹的任务分布和你桌子上的杯子差在哪？Cosmos3-Nano-Policy-DROID 用短指令、三机位、绝对关节位置，和 [第 26 课](26_vla_vs_world_model.md) OpenVLA 的离散动作 token 比，世界模型零件还在不在？
8. Qwen3-VL（报告引用为 Nano / Super 的初始化，卡片与报告写 8B / 32B）。带着问题读：DeepStack 和交错时间戳进了 Reasoner 的哪一段？Generator 为什么还要另装 Wan2.2 VAE，而不是复用 ViT token 去解码像素？
9. Wan2.2 与 Wan2.1（视频 VAE / 开源教师；Wan2.1 见 [arXiv:2503.20314](https://arxiv.org/abs/2503.20314)）。带着问题读：时间压缩 4×、空间 32×，对 24 fps 视频的 $\mathrm{TPS}$ 意味着什么？第 37 课若用 Wan 1.3B 做 self-forcing，和 Cosmos 3 Generator 的双向预训练差在因果性。
10. 第 30 课自己的桌面世界模型。带着问题回头看：四个离散动作、冻结 DINOv2、特征空间对换。Cosmos 3 哪一块（域相关动作投影、flow matching、双塔）搬得回 24GB？哪一块搬回去只会把毕业设计拖死？下一课先问声音：喇叭是动作还是配乐，麦克风是观察还是装饰。

现在整个系统长这样：第八幕的桌宠仍在桌子上，第九幕多了一张「工业级全模态骨架」的说明书。你知道 Cosmos 3 怎样用配置冒充四种产品，也知道 24GB 上它仍然不是第 32 课的身体。下一课把声音从「视频附带的 AAC 音轨」里拆出来，看它到底是 $o_t$ 的通道，还是生成器的配乐。
