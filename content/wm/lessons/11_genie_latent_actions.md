---
id: 11_genie_latent_actions
title: "Genie 从视频中发现动作"
summary: "只有视频、没有按键记录，模型怎么自己发现“世界上存在 8 种动作”？"
unit: generative
play_tools: []
checkpoints:
  - "latent action 可解释性分析。"
  - "真实机器人数据上的 GENIE-style baseline 训练记录。"
---

# 第 11 课：没有动作标签，Genie 怎么从视频中发现动作

> 类型：复现（tinyworlds 三件套小规模训练 + latent action 可解释性检查）+ 实战（1xgpt 真实机器人数据上的 GENIE-style baseline）+ 只讲（Genie 本尊 11B，无权重）<br>
> 建议周期：3-5 天<br>
> 硬件：单张 24GB 卡全程够用；tinyworlds 的推理配置默认走 Mac 的 MPS，交互游玩在笔记本上就能做；1xgpt 的数据已预 token 化，训练比像素级世界模型省算力得多<br>
> 锚定仓库：[AlmondGod/tinyworlds](https://github.com/AlmondGod/tinyworlds)（最小 Genie 教学实现：FSQ tokenizer + 无监督动作 tokenizer + MaskGIT 动力学）+ [1x-technologies/1xgpt](https://github.com/1x-technologies/1xgpt)（100+ 小时真实人形机器人第一人称数据，预 token 化，附 GENIE-style baseline）<br>
> 产物：latent action 可解释性分析报告 + 真实机器人数据上的 baseline 训练与评测记录

## 1. 这一课做什么

IRIS 和 DIAMOND 都假设数据中已经记录了动作：CarRacing 有方向盘数值，Atari 有手柄
按键。但互联网视频、机器人第一人称录像和行车记录仪通常只有画面，没有同时保存的
控制指令。要利用这些数据，模型必须先从前后帧的变化中推断可能的动作。

DeepMind 2024 年的 Genie 使用一个很窄的离散瓶颈来完成这件事。在"上一帧"
和"下一帧"之间只留少量动作码：编码器可以看下一帧，并把历史无法解释的变化压进码里；
解码器则根据历史和动作码重建下一帧。因为通道很窄，码更可能捕获外部干预，而不是复制
整幅画面。训练完成后，每个码的含义仍需通过生成结果逐一解释。

本课使用 tinyworlds 训练一个最小版 Genie，包括三个部分：FSQ 视频 tokenizer、无监督
动作 tokenizer，以及 MaskGIT 动力学模型。与逐 token 自回归不同，MaskGIT 会并行猜测
被遮住的 token，再分轮修正。

训完后会固定每个离散动作码，从多个起点生成视频，检查它是否稳定对应上移、下移或其他
操作；若两个码效果无法区分，也要明确记为退化。随后把相同方法用到 1X 公司 EVE 机器人
的第一人称录像，比较游戏画面与真实办公室数据上的生成质量。

动作标签不再是必需条件后，训练数据可以从专门采集的轨迹扩展到普通视频。但“码是否真
学到了可控动作”不能由训练损失代替，必须通过干预和跨起点一致性验证。这套检查方法还会
用于后面的 JEPA 坍缩审计和社区模型条件轴测试。

术语速查：

| 术语 | 一句人话 |
|---|---|
| latent action / 潜动作 | 模型自己从视频里挖出来的离散动作变量，没有任何人工标签，含义要靠事后检查来"翻译" |
| LAM | latent action model，挖潜动作的那个零件：编码器偷看下一帧压出动作码，解码器拿历史加码重建下一帧，训完只留码本 |
| 信息瓶颈 | 故意把通道做窄，逼模型只传最要命的信息；LAM 的码本小到 8 个码，挤出来的就是"动作" |
| FSQ | 有限标量量化：每一维压到 [-1,1] 后四舍五入到少数几个格点，没有可学的码本，天生不坍缩 |
| 码本坍缩 | VQ 的常见问题：大量码没人用、活跃码越来越少，参见第 09 课 |
| MaskGIT | 掩码生成：训练时随机遮住一部分 token 学着补全，生成时全部一起猜、按置信度逐轮揭开 |
| ST-Transformer | 时空注意力交替的 transformer：空间层看同帧内的 token，时间层看同位置的历史，视频模型的省算力标配 |
| FiLM | 一种条件注入方式：把动作码变成缩放和平移系数，乘加在归一化后的特征上 |
| teacher forcing 三档 | 评测生成模型时给多少真实上下文：全自回归、时序 teacher-forced（TTF）、全 teacher-forced，1xgpt 的评测按这三档分 |
| $\Delta_t$ PSNR | Genie 的可控性度量：用推断动作生成和用随机动作生成，两者画质差多少，差得越多说明动作越管用 |

## 2. 问题

需要回答三个问题：

1. 机制问题：没有标签，"动作"这个变量凭什么会自己长出来？直觉上这像无中生有。
   第 5 节会把它拆成一条信息论的账：两帧之间的变化分两种，世界自己会滚的部分
   （历史能预测）和外来干预的部分（历史预测不了）；一个只有几比特的瓶颈，装不下
   前者，只装得下后者。账算清之后你会发现这不是魔法，是压缩的必然结果，顺带
   理解为什么这一路的量化器要从 VQ 换成 FSQ。
2. 度量问题：挖出来的"动作"，怎么验收？这是重点。一个无监督变量最容易
   骗人：训练损失在降，动作码在变，但它可能只是噪声开关。合格的验收要过两关，
   分岔关（同一起点、不同码，生成必须不同；这是第 03 课动作对换实验的直系后代）
   和一致关（同一个码、不同起点，效果必须是同一种操作）。第 5 节讲 Genie 论文的
   量化版本 $\Delta_t$ PSNR，第 7 节你在 tinyworlds 上做定性加定量两套。
3. 现实问题：游戏玩具上灵的配方，搬到真实世界还剩多少？tinyworlds 的五个
   数据集全是像素游戏：画面干净、动作离散、变化集中。EVE 机器人的第一人称录像
   是另一个世界：连续全身控制、镜头晃动、光照与场景千变万化。1xgpt 的 baseline
   甚至干脆没接动作。这个落差不是本课的遗憾，是本课的教学内容，你要如实记录它，
   第 12 课判断前沿系统时要靠这份手感。

## 3. 准备

- 前课的知识：第 09 课的"图像变 token"与 VQ 码本坍缩（本课 FSQ 是冲着它来的）、
  第 03 课的动作对换实验（本课可解释性检查的思想源头）、第 08 课的 checkpoint 评测
  协议（1xgpt 的预训练 baseline 评测直接复用这个习惯）。
- 两个互相隔离的虚拟环境。tinyworlds 是常规 `pip install -r requirements.txt`；
  1xgpt 自带 `build.sh`，会创建自己的 venv 并下载数据，要求 Python 3.10 以上
  （官方在 3.10.12 上测试过）。两个项目依赖不同，别混装。
- 一个 wandb 账号。tinyworlds 的训练配置默认 `use_wandb: true`，曲线都往那儿打；
  不想注册就把 `configs/training.yaml` 里这项改成 false。
- 磁盘：tinyworlds 的数据集是处理好的 `.h5` 文件，单个游戏在 GB 量级；1xgpt 的
  token 数据集比原始视频小得多，但 100+ 小时也不轻，预留几十 GB，具体体积以
  HuggingFace 数据卡（`1x-technologies/worldmodel`）为准。
- 硬件分工：tinyworlds 训练用单张 NVIDIA 卡（配置里 AMP/BF16、TF32、compile
  默认全开）；推理和交互游玩默认 `device: mps`，Mac 直接能跑。1xgpt 训练与评测
  在单卡上进行，官方榜单的生成速度就是在一张 RTX 4090 上测的。

## 4. 学习目标

1. 用信息瓶颈的语言讲清 LAM 为什么能从无标签视频里挖出动作，并回答一个好问题：
   编码器训练时偷看了下一帧，为什么推理时可以把它整个扔掉；
2. 说清 FSQ 和 VQ 的机制差异，以及"没有可学码本"为什么从根上消掉了码本坍缩；
3. 白纸写出 MaskGIT 的训练目标和解码循环，并解释它比逐 token 自回归快在哪、
   可能挂在哪；
4. 独立设计并执行一次 latent action 可解释性检查：分岔关、一致关、量化对照三件套，
   并把结论写得可复核（哪个码、哪些起点、看到什么、判定为什么）；
5. 拿到一份新的视频数据集时，能预估"Genie 配方在它上面能走多远"，并说出依据
   （动作离散度、画面稳定性、变化稀疏性）；
6. 说清教学缩小版与 Genie 本尊之间差什么：哪些是规模（参数、数据、算力），哪些是
   机制上完全一样的。

## 5. 原理

四个机制加一个现实核对，每个照旧走五步：直觉、机制、数学、代码落点、验证。

### 5.1 LAM：把"动作"从两帧的夹缝里挤出来

盯着游戏录像里相邻的两帧看，画面的变化来自两股力量：一股是世界自己在滚
，云在飘、敌人按脚本巡逻、跳到半空的角色继续沿抛物线下落；另一股是玩家的干预，
按了左键，角色开始左移。第一股力量是可预测的：看足够多的历史，模型自己就能推出来。
第二股不行：玩家下一步按什么键，历史里没有答案。现在做一个思想实验：让一个助手
（编码器）偷看下一帧，允许它给你（解码器）递一张小纸条，你只凭历史帧和纸条重建
下一帧。纸条要是够大，助手会把整张下一帧抄上去，学不到任何结构。但把纸条缩到
只写得下 3 个比特呢？助手会怎么用这 3 比特？写"云往左飘了一格"是浪费，你自己
能推出来；唯一值得写的是你猜不到、写了又最能救重建损失的东西：**玩家干预**。
纸条就是 latent action。

Genie 的 LAM 是一对不对称的网络加一个极小的 VQ 码本。编码器吃全部历史帧
加下一帧 $x_{t+1}$，输出连续向量，量化到 8 个码之一；解码器吃历史帧加这个码，重建
$x_{t+1}$。关键设计有三处。其一，码本小：$|A|=8$，论文明说小码本既是信息瓶颈也是
为了人类玩得动。其二，训练完成后编码器和解码器**整个扔掉**，只留 8 个码的码本，
推理时没有"下一帧"可偷看，动作码由玩家直接选（输入 0 到 7 的整数）。解码器从头到
尾只是给训练供梯度的脚手架。其三，Genie 的消融发现 LAM 吃原始像素比吃 token 效果
好：动作往往藏在细微的像素变化里，tokenizer 有可能把它压丢，这和第 10 课
DIAMOND 的立课之本（小细节恰恰最重要）是同一个观察。

tinyworlds 的实现同构而更小：编码器把每帧 patch 化后过因果 ST-Transformer，帧内
均值池化，再把第 $t$ 帧和第 $t+1$ 帧的特征拼接过一个小 MLP，得到每个转移的动作
向量；量化用 FSQ 而非 VQ（下一小节讲为什么）；解码器用 FiLM 把动作码注入每层。
它还多加了两道保险，防止解码器绕开动作走捷径：训练时把**除第一帧外所有帧的 patch
全部换成 mask token**（源码里 `keep_rate = 0.0`，注释原话是 strongly forces actions
to contain most useful info），历史被遮到只剩锚点，重建信号几乎只能从动作序列里
来；再对编码器输出加一个 batch 内方差的辅助损失（`var_target`、`var_lambda`），
不许所有样本的动作码挤成同一个值。

码本 8 个码，每个转移 $\log_2 8 = 3$ 比特。记两帧间全部变化的信息量为
$H(x_{t+1} \mid x_{\le t})$，远大于 3 比特。编码器要在 3 比特预算内最小化重建
损失，最优策略必然是优先编码"条件熵里贡献最大、又无法由历史推出"的成分。世界
自转的部分对 $H(x_{t+1} \mid x_{\le t})$ 贡献小（历史可推），外生干预贡献大且不可
推，瓶颈越窄，码本越被迫向干预对齐。这个论证不是严格定理（如果环境里有强随机
事件，比如随机刷怪，码本也可能被它抢走），但它给出明确的可检验预言：码本容量
接近真实动作数时，码应当与动作一一对应；预言对不对，第 7 节验。

`models/latent_actions.py`：`LatentActionsEncoder`（看 `action_head`
前面那段相邻帧特征拼接）、`LatentActionsDecoder`（看 `keep_rate` 那几行和
`conditioning=actions`）、`LatentActionModel`（看 `NUM_LATENT_ACTIONS_BINS = 2` 和
`action_dim = log2(n_actions)` 的断言：动作数必须是 2 的幂）。配置在
`configs/training.yaml` 的 `n_actions`（默认 4）和 `configs/latent_actions.yaml`。
Genie 本尊的对应物：LAM 约 300M 参数、patch 16、8 个码。

两条：训练侧看 LAM 的重建损失显著低于"无动作纯外推"的水平（说明码
确实在传信息）；使用侧做第 7 节的可解释性检查（说明传的是"动作"而不是别的）。
两条缺一不可，损失降只证明纸条有用，不证明纸条上写的是按键。

### 5.2 FSQ：把码本这个病灶整个切掉

第 09 课你见过 VQ 的老毛病：码本是一组可学习的向量，编码器输出找最近邻，
赢者通吃，被选中的码有梯度、越练越好，没被选中的码原地腐烂、越来越没人选。
坍缩起来，4096 个码只有几十个在干活。社区的补法是一堆护理措施：commitment 损失、
码本重播种、code splitting、熵惩罚……FSQ 的思路是釜底抽薪：**病灶是"可学习的
码本"，那就不要码本**。把连续向量的每一维独立地压进固定区间、四舍五入到固定的
几个格点，格点是写死的，不学习、不查表、没有"没人用就腐烂"的对象。

四步：对每维做 $\tanh$ 压到 $[-1,1]$；线性缩放到 $[0, L-1]$（$L$ 是每维
格点数）；四舍五入取整；缩放回 $[-1,1]$。词表隐式存在，大小是 $L^D$（$D$ 是维数）
，tinyworlds 的视频 tokenizer 用 $D=5$、$L=4$，词表 $4^5 = 1024$；动作 tokenizer
每维固定 2 个格点，$D = \log_2(\texttt{n\_actions})$，默认配置 4 个动作就是 2 维
各 2 格。取整不可导，用直通估计把梯度原样传回去，源码就一行：
`quantized = bounded + (rounded - bounded).detach()`。

FSQ 论文（arXiv:2309.15505）的主张：把 VQ-VAE 的向量量化换成这套逐标量
取整，下游的自回归或掩码 transformer 不用改一个字，性能打平，而 commitment 损失、
重播种、splitting、熵惩罚全部删掉，码本利用率不再是需要监控的病人。代价是表达
结构变了：VQ 的码是自由分布在高维空间的 4096 个点，FSQ 是规整网格的格点，
论文的经验是把维数选小（典型小于 10）、格点选少，效果就能对齐。

`models/fsq.py::FiniteScalarQuantizer`，全文不到百行，文件头注释
原话就是 prevents token collapse and no auxiliary losses necessary。看三个方法：
`forward`（四步量化加直通）、`get_indices_from_latents`（格点坐标按 $L$ 进制拼成
token 序号，这是"隐式码本"的实现）、`get_codebook_usage`（利用率统计，
第 09 课你监控码本坍缩用的就是这个指标，这里拿来验证"不坍缩"）。

训练视频 tokenizer 时顺手打印码本利用率：VQ 时代这是重点监护对象，FSQ
下它应当自然地高且稳。另一条验证在结构上：这里从头到尾没有任何码本护理超参，
你在 configs 里找不到 commitment 系数，因为不需要。

### 5.3 MaskGIT 动力学：全部一起猜，再逐轮改口

第 09 课的 IRIS 按光栅顺序逐个预测下一帧的 token：一帧几百个 token 就是
几百次前向，生成慢得没法交互。可图像不是句子，左上角的天空和右下角的地面几乎
互不依赖，凭什么要排队生成？MaskGIT 把生成当成填字游戏：先让模型对**所有**空格
同时出牌，只把它最有把握的几格落定；下一轮，已落定的格子变成线索，再落定次有把握
的一批；十来轮之内整张填完。把握大的先写、把握小的等线索，这比从左往右硬写合理
得多，也快得多，MaskGIT 论文摘要的数字是最高把自回归解码加速 64 倍。

训练是 BERT 式的：随机遮住一部分 token，让模型看着没遮的部分（以及
动作码）把遮住的补出来。tinyworlds 和 Genie 用同一个训练掩码率区间，每个 batch
均匀采一个 $[0.5, 1)$ 之间的比例（源码 `mask_ratio = 0.5 + torch.rand(()) * 0.5`），
再按它随机选位置遮盖；掩码率必须练到接近 1，因为推理起点就是"整帧全遮"。推理时
往上下文后面追加一整帧 mask token，然后循环：预测所有掩码位的分布，按置信度
（最大概率）挑出 $k$ 个位置揭开，其余保持掩码，进下一轮。每轮揭开多少由调度函数
定，tinyworlds 用指数调度（`exp_schedule_torch`，默认 `schedule_k=5.0`）：前几轮
只揭一两个、后几轮成片揭开，README 给的示意是每步约 1、2、5、20、50 个。Genie
本尊每帧走 25 步 MaskGIT，采样温度 2；动作码不是拼接进序列，而是加性嵌入直接加在
token 表征上，论文说这个选择对可控性更好。

训练目标是掩码位置上的交叉熵：只对被遮的位置算损失（tinyworlds 源码里
用 `mask_flat` 加权后除以掩码数）。推理的调度函数为 $n(t) = P \cdot \frac{1 - e^{k t / T}}{1 - e^{k}}$
的相邻差分（$P$ 是总掩码数、$T$ 是总步数），温度 0 时对每个位置取 argmax，大于 0
时按分布采样但置信度仍用最大概率衡量。

`models/dynamics.py`：训练看 `forward`（掩码率采样、每个空间位置
保底一个不掩的时间锚点、掩码位交叉熵），推理看 `forward_inference`（追加掩码帧、
指数调度、按置信度逐轮揭开）。1xgpt 那边的同类实现在 `genie/st_mask_git.py`，
评测入口 `genie/evaluate.py` 的 `--maskgit_steps` 参数直接控制解码轮数，官方榜单
的数字是 2 步解出来的，这个"步数换质量"的旋钮第 7 节你会拧。

两个方向：把解码步数从 2 加到更多，看质量指标和耗时怎么走（1xgpt 上做）；
把温度从 0 调大，看生成从"保守清晰"滑向"多样但易穿帮"（tinyworlds 上做）。
能预言旋钮方向、再被实验确认，才算真懂了这个循环。

### 5.4 可控性怎么度量：动作对换的直系后代

第 03 课你给 MDN-RNN 做过动作对换：同一状态喂不同动作，预测必须分岔，
不分岔就是"动作盲"。本课的变量是无监督挖出来的，验收要加码到两关：**分岔关**，
同一起点、不同码，生成的未来必须不同（否则码没被用上）；**一致关**，同一个码、
不同起点，效果必须是同一种操作（左移码在哪儿都该左移；否则那只是每个状态各自
含义不同的噪声开关，人没法拿它当按钮）。Genie 论文对第二关的表述很形象：每个
潜动作的含义事先未知，但跨输入保持一致，学会玩它就像拿到一个没贴标签的新手柄，
试几下就知道哪个键是跳。

Genie 把这套检查做成了量化指标 $\Delta_t$ PSNR：取一段真实视频，
用 LAM 从真帧序列推断出动作码序列生成一遍（得 $\hat{x}_t$），再用随机抽的动作码
生成一遍（得 $\hat{x}'_t$），比较两者对真帧的 PSNR：

$$
\Delta_t \text{PSNR} = \text{PSNR}(x_t, \hat{x}_t) - \text{PSNR}(x_t, \hat{x}'_t)
$$

差值越大，说明换成随机动作后未来偏得越远，动作真的在驾驶世界；差值接近 0，
说明动作码形同虚设。论文在 $t=4$ 处报告这个数，并用它做了那组"LAM 吃像素还是吃
token"的消融：像素版 1.91、token 版 1.33（Platformers 数据），可控性站在像素一边。

tinyworlds 没有内置 $\Delta_t$ PSNR，但推理配置里天然备好了两个
条件组：`configs/inference.yaml` 的 `use_gt_actions: true` 就是"用 LAM 从真视频
推断的动作生成"，`use_actions: true` 就是"用随机动作生成"，且
`utils/inference_utils.py::visualize_inference` 每次运行都打印生成帧对真帧的 MSE。
两种模式各跑一遍、比 MSE，就是一个穷人版 $\Delta_t$ PSNR，方向完全相同，只是
把 PSNR 换成了它的单调变换。第 7 节 Step 6 照此执行。

这一小节本身就是验证方法，留一个思考校验：如果你的模型分岔关过了、
一致关挂了（每个码都有效果但含义乱跳），按 5.1 的信息瓶颈论证，最该先查的是什么？
（提示：码本容量和数据里真实操作数的关系，答案在第 10 节症状表。）

### 5.5 从 Pong 到 EVE：同一配方在真实世界还剩多少

tinyworlds 的五个数据集（PicoDoom、Pong、Zelda、Pole Position、Sonic）
全是理想化的世界：镜头固定或平滑滚动、调色板干净、两帧之间的变化几乎全部由少数
几个离散按键解释。EVE 机器人的第一人称录像每一条假设都不成立：动作是连续的全身
控制而非 8 个按键，摄像头长在会走路的躯干上（世界"自转"部分暴涨），光照、行人、
物体遮挡千变万化，且数据以 2Hz 采样，相邻两帧隔了半秒，变化大得多。信息瓶颈
论证依赖"干预是两帧间变化的主要不可预测成分"，这些条件一松，挤出来的码就未必
是干净的"动作"了。

1xgpt 的应对很诚实：它的 GENIE-style baseline（时空 transformer 加
MaskGIT 采样器）**只在视频 token 序列上训练，不接动作**，README 原话说加上动作
via an additive embedding 是 trivial 的，但基线版就是纯视频预测。数据侧的功课做得
很足：100+ 小时录像用一个帧级 MAGVIT2 自编码器预先 token 化成 16×16 的 token 图
（词表 $2^{18}$），你训练时碰的全是 token，算力省了一个量级；配套发布 tokenizer
权重 `magvit2.ckpt`，想拿外部视频加练也能自己编码。评测按 teacher forcing 三档
分场景，主指标是时序 teacher-forced（TTF）交叉熵，注意它的账法：$2^{18}$ 的
词表太大，logit 张量存不下，官方把每个 token 拆成两个 $2^9$ 类的因子化预测、
交叉熵求和，所以这个 loss 数值不能和普通 LLM 的逐 token loss 直接比大小。

因子化即 $p(x_1, x_2) = p(x_1)\,p(x_2)$：一个 18 比特 token 拆成两个
9 比特半票。官方规则明确评测固定用这个因子化形式，不接受用联合分布另开赛道。

因子化的实现在 `genie/factorization_utils.py`，损失与指标在
`eval_utils.py` 和 `genie/evaluate.py`；MAGVIT2 的编解码器在 `magvit2/` 目录
（`models/lfqgan.py`）。对照读一眼 tinyworlds 的 `models/video_tokenizer.py`，
你会看清两个项目在"帧变 token"这一步的分工完全同构，只是规模差两个量级。

落差要量出来而不是感慨出来。第 7 节 Step 8 的对照笔记里，你至少要给出
三条具体差异（例如：Pong 上固定动作码画面响应肉眼可辨，EVE 生成里机器人手臂
动作模糊化；Pong 的 tokenizer 重建近乎无损，EVE 的 MAGVIT2 重建在细小物体上
糊掉；同等训练步数下两者 loss 曲线形态不同），每条都注明你在哪个输出文件里看到。

## 6. 源码导读

两个仓库各一张导览表。tinyworlds（结构极小，一晚上能通读）：

| 文件 | 是哪个零件 | 带着什么问题读 |
|---|---|---|
| `models/fsq.py` | FSQ 量化器 | 直通估计那一行在哪？`get_indices_from_latents` 怎么把格点坐标拼成 token 序号？|
| `models/video_tokenizer.py` | 视频 tokenizer | 编码器输出几维、几个格点？词表 1024 是怎么由配置推出来的？|
| `models/latent_actions.py` | LAM | 动作数为什么必须是 2 的幂？`keep_rate = 0.0` 遮掉了什么、留下了什么？方差辅助损失在防哪种坍缩？|
| `models/dynamics.py` | MaskGIT 动力学 | 训练掩码率是多少到多少？推理的指数调度前紧后松还是前松后紧？|
| `models/st_transformer.py` | 时空骨干 | 空间注意力和时间注意力各自的注意范围是什么？FiLM 条件从哪里注入？|
| `scripts/full_train.py` | 三阶段总控 | 三个 `run_*` 开关怎么串起三个阶段？各阶段分别依赖谁的 checkpoint？|
| `scripts/run_inference.py` | 推理入口 | 三种动作模式（随机/推断/交互）互斥关系怎么处理？`n_actions` 从哪个对象读出来？|
| `utils/inference_utils.py` | 推理胶水 | `get_action_latent` 的交互输入长什么样？`visualize_inference` 把结果存到哪、打印什么指标？|
| `configs/training.yaml` | 总配置 | `n_actions`、`latent_dim`、`num_bins`、`context_length`、`frame_size` 各控制什么？|
| `datasets/datasets.py` | 数据注册 | 加一个新游戏数据集要写哪几行？|

1xgpt（工程更工业化，挑主干读）：

| 文件 | 是哪个零件 | 带着什么问题读 |
|---|---|---|
| `build.sh` | 环境与数据 | 它建了什么 venv、把数据下到哪个目录？|
| `train.py` | 训练入口 | `--genie_config` 吃的 json 里定义了哪些结构超参？|
| `genie/st_mask_git.py` | GENIE 主模型 | 和 tinyworlds 的 `dynamics.py` 对照：掩码训练和迭代解码各自怎么写？|
| `genie/configs/magvit_n32_h8_d256.json` | 模型配置 | 仓库自带的唯一一份训练配置，层数、头数、宽度各是多少？|
| `genie/generate.py` / `genie/evaluate.py` | 生成与评测 | `--maskgit_steps` 和 `--temperature` 各在哪里生效？评测算哪几个指标？|
| `genie/factorization_utils.py` | 词表因子化 | $2^{18}$ 拆成 $2 \times 2^9$ 的具体张量操作是什么？|
| `magvit2/models/lfqgan.py` | MAGVIT2 tokenizer | 只读接口：encode 和 decode 的输入输出形状？|
| `visualize.py` | 可视化 | 它吃哪个目录的 token、吐什么格式的动图？|

## 7. 实验

前六步在 tinyworlds（训练、重点），后三步在 1xgpt（真实数据）。每一步先写预期、
再跑、再对照。

### Step 1: 安装 tinyworlds

```bash
git clone https://github.com/AlmondGod/tinyworlds.git
```

```bash
pip install -r requirements.txt
```

进入仓库目录后装依赖，照旧开独立虚拟环境。然后设两个环境变量（README 的原文示例
路径是 `/workspace/tinyworlds`，改成你自己的克隆路径）：

```bash
export WANDB_API_KEY=<你的wandb密钥>
```

```bash
export PYTHONPATH="/你的路径/tinyworlds:$PYTHONPATH"
```

不想用 wandb 就把 `configs/training.yaml` 里 `use_wandb` 改成 false，密钥那步跳过。

### Step 2: 先钻进预训练模型玩一次

和第 09 课先玩 Breakout 一个道理：先建立体感，再自己训练。下载作者放在
HuggingFace 上的 Sonic 预训练三件套，然后启动交互推理：

```bash
python scripts/download_assets.py models --suite-name sonic
```

```bash
python scripts/run_inference.py --config configs/inference.yaml -- use_latest_checkpoints=true dataset=SONIC
```

`configs/inference.yaml` 默认 `use_interactive_mode: true`、`device: mps`（Mac 直接
跑；NVIDIA 机器把 device 改成 cuda）。终端会逐步提示
`Enter action id [0..N-1]`，你输一个数字它生成一帧。跑完在 `inference_results/`
里找两样东西：一张"真实帧对生成帧"的对比图（每帧标着你输的动作号）和一段 MP4。
预期：画面是认得出的 Sonic 世界，会响应你的输入，但 64×64 的教学模型穿帮是常态
，记下你看到的第一处穿帮，第 12 课谈长时程一致性时用得上。

### Step 3: 下载一个小游戏数据集

```bash
python scripts/download_assets.py datasets --pattern "pong_frames.h5"
```

README 的演示用的是 `zelda_frames.h5`，五个数据集（picodoom、pong、zelda、
pole_position、sonic）任选。本课主线选 **Pong**：它的真实操作结构最简单（球拍
上下移动），后面给动作码"当翻译"时最容易读出结论。数据是预处理好的 `.h5`，
来自 HuggingFace 的 `AlmondGod/tinyworlds`。

### Step 4: 三阶段训练

```bash
python scripts/full_train.py --config configs/training.yaml -- --dataset=PONG
```

`full_train.py` 按 `configs/training.yaml` 里三个 `run_*` 开关顺序执行三个阶段：
视频 tokenizer、动作 tokenizer（LAM）、动力学，也可以用 `scripts/train_video_tokenizer.py`
等三个脚本分开跑。先别急着跑满：默认配置里动力学阶段 `n_updates` 是 300000 步，
这是作者的完整档；冒烟档建议先把 `configs/dynamics.yaml` 的 `n_updates` 改到
两三万步、把 `configs/video_tokenizer.yaml` 保持 40000 步不动，量力再加。预期：
wandb 上三条损失曲线依次出现并下降。**中途验收两处**：视频 tokenizer 训完看重建
（球、球拍、比分认得出才继续）；LAM 训练时留意码本利用情况，FSQ 之下它应当
高且稳，这是 5.2 节论断的现场验证。三个阶段各自往 checkpoint 目录存档，动力学
阶段会自动去找前两个阶段最新的 checkpoint。

### Step 5: 重点：latent action 可解释性检查

现在你手里有一个从没见过动作标签的模型，和 4 个它自己发明的动作码（默认
`n_actions: 4`）。任务：给每个码找出它的含义，或诚实地判定它没有稳定含义。

先加两行胶水固定起点。`run_inference.py` 每次运行会从数据集里**随机**抽一段起始
context（源码 `random.randint` 那行），这对"一致关"是天然便利（重跑即换起点），
但"分岔关"需要同起点对照。打开 `scripts/run_inference.py`，在 `main()` 开头加一行
`random.seed(0)`（脚本已 `import random`）。改动两行以内，属于本课允许的胶水。

然后把 `configs/inference.yaml` 设置为：`dataset: PONG`、`temperature: 0`（argmax，
消掉采样随机性）、`generation_steps: 8`、`use_interactive_mode: true`（其余两个
动作开关保持 false）。执行协议：

1. 分岔关：种子固定为 0，跑 4 次推理，第 $i$ 次从头到尾每步都输入动作码 $i$
   （$i = 0,1,2,3$）。每次跑完立刻把 `inference_results/` 里新生成的 PNG 和 MP4
   改名归档（如 `pong_seed0_action2.mp4`）。四段视频起点相同、码不同，逐对比较：
   生成是否分岔？
2. 一致关：把种子改成 1 和 2，各重复一遍上面 4 次，共 12 段视频。对每个码，
   横向看它在三个不同起点下的效果是否为同一种操作。
3. 填矩阵：4 行（码）× 3 列（起点），每格写一句客观描述（"球拍上移两格"、
   "与码 0 无可辨差异"），最后每行给一个判定，三选一：**一致可解释**（跨起点
   同一效果，且能用人话命名）、**有效果但不可解释**（分岔了，但含义跨起点漂移）、
   无效果（与其他码或无动作生成无可辨差异）。

预期怎么设才诚实：这是一个几十万参数级的教学模型，Genie 论文里"左右移动、跳、
无操作各占一码"的干净结果不保证在你这里复现。常见的合格结果长这样：一两个码
稳定对应球拍方向，一个码近似无操作，剩下的码效果微弱或与他码重合。三种判定都是
合法发现，这份矩阵的价值在于它可复核，任何人拿你的种子和码号能重看出同样的画面。
全部判为"无效果"才说明系统有病，去第 10 节查。

### Step 6: 穷人版 $\Delta_t$ PSNR：把可控性变成数字

矩阵是定性的，再补一个定量对照（5.4 节的机制）。保持种子和 temperature 不变，
改 `configs/inference.yaml` 跑两组：

1. `use_gt_actions: true`（另两个动作开关 false）：LAM 从真实视频推断动作序列，
   按它生成。跑完记下终端打印的 `Mean Squared Error (GT vs Pred)`。
2. `use_actions: true`（另两个 false）：随机动作生成，同样记 MSE。

各换 3 个种子重复，报均值。预期：推断动作组的 MSE 系统性低于随机动作组，真动作
把生成拉向真实未来，随机动作把它带偏，这个差距就是可控性。两组差距若在噪声以内，
说明动力学模型在无视动作码（动作盲，第 03 课的老病在无监督设定下复发），先查
Step 4 里 LAM 的损失，再查第 10 节。把两组数字连同矩阵一起写进可解释性报告，
本课交付物之一完成。

### Step 7: 换真实世界：安装 1xgpt 并拉数据

```bash
git clone https://github.com/1x-technologies/1xgpt.git
```

```bash
./build.sh
```

```bash
source venv/bin/activate
```

`build.sh` 会装依赖并把 100+ 小时的预 token 化数据下到 `data/train_v1.1` 与
`data/val_v1.1`。要求 Python 3.10+。这批数据是 EVE 人形机器人在 1X 办公室作业的
第一人称录像，MAGVIT2 已经替你把每帧压成 16×16 个 token，你接下来训练的每一步
都在 token 上进行，这是"预 token 化很省算力"的含义：像素级的重活别人一次性
干完了。

### Step 8: 评测官方预训练 baseline：协议对齐练习

先不训练，把官方发布的 GENIE_138M 权重按官方协议评一遍（第 08 课 checkpoint
评测手艺的复用：对不上数字时先怀疑协议，不怀疑权重）：

```bash
python genie/evaluate.py --checkpoint_dir 1x-technologies/GENIE_138M --maskgit_steps 2
```

预期：验证集上时序 teacher-forced 交叉熵约 8.79、LPIPS 约 0.207，与仓库 README
榜单一致（榜单同时列了 GENIE_35M：8.99 与 0.217）。注意两点协议细节：这个 loss
是 $2 \times 2^9$ 因子化交叉熵之和，量纲与常见 LLM loss 不可比；榜单速度
（138M 每帧 0.075 秒、35M 每帧 0.030 秒）是 RTX 4090 上纯 latent 生成的耗时，
不含 token 解码回图像。再生成一段看看：

```bash
python genie/generate.py --checkpoint_dir 1x-technologies/GENIE_138M --output_dir data/genie_baseline_generated --example_ind 0 --maskgit_steps 2 --temperature 0
```

```bash
python visualize.py --token_dir data/genie_baseline_generated
```

把 `--maskgit_steps` 加大再跑一遍，量一下质量与耗时怎么变，这是 5.3 节"步数
换质量"旋钮的实测。

### Step 9: 训练你自己的 GENIE-style baseline，写落差笔记

```bash
python train.py --genie_config genie/configs/magvit_n32_h8_d256.json --output_dir data/genie_model --max_eval_steps 10
```

用仓库自带的唯一一份配置训练（README 原文命令，`--max_eval_steps 10` 让训练途中
的评估只抽 10 个 batch，省时间）。训练记录照第 01 课的规矩留四件套：完整命令、
配置、随机性说明、loss 曲线。训完依次评测、生成、可视化：

```bash
python genie/evaluate.py --checkpoint_dir data/genie_model/final_checkpt
```

```bash
python genie/generate.py --checkpoint_dir data/genie_model/final_checkpt
```

```bash
python visualize.py --token_dir data/genie_generated
```

最后写**落差对照笔记**（5.5 节的验证）：拿你 Pong 上的可解释性矩阵和这里的生成
动图并排，至少写出三条具体差异，每条注明出处文件。提醒一句：1xgpt 的 baseline
不接动作，所以这里没有"按钮"可按，纯视频预测。这本身就是最大的一条落差：
真实机器人的动作是连续全身控制，Genie 式"8 个离散按键"的假设在这里对不上口径，
官方宁可先做纯预测。数据集里带着原始动作流，README 说加性嵌入接入动作是
trivial 的，第 11 节把它列成了本课含金量最高的改造实验。

## 8. 配置与预算

tinyworlds 三个阶段的默认配置（均以仓库 `configs/` 当前内容为准）：

| 阶段 | 配置文件 | batch/GPU | 步数（默认） | 学习率 | 冒烟档建议 |
|---|---|---|---|---|---|
| 视频 tokenizer | `configs/video_tokenizer.yaml` | 350 | 40000 | 0.001 | 照默认跑完，这是地基 |
| 动作 tokenizer（LAM） | `configs/latent_actions.yaml` | 350 | 10000 | 0.0001 | 照默认 |
| 动力学（MaskGIT） | `configs/dynamics.yaml` | 500 | 300000 | 0.01 | 先砍到 2 万-3 万步看效果，量力加 |

几笔账：模型本身极小（三个零件的 embed 维度都是 32，帧只有 64×64），单张 24GB
卡显存绰绰有余，瓶颈更多在步数和数据吞吐；`preload_ratio` 控制预载进内存的数据
比例，内存紧就调小。AMP（BF16）、TF32、torch compile 在 `configs/training.yaml`
里默认全开；多卡可切 `use_ddp: true` 用 torchrun 起，DDP 是"同一模型、各卡吃
不同数据"的标准数据并行，单卡完全够这里用，这两项知道即可。交互推理的 MaskGIT
步数在 `run_inference.py` 里固定为 10 步，属于体验档配置。

1xgpt 侧：官方两个 baseline 规模是 35M 和 138M。仓库自带的训练配置是
`genie/configs/magvit_n32_h8_d256.json`（结构超参以该文件为准）；数据已预 token 化，
训练是在 16×16 token 图上的 transformer，单卡可行，官方榜单本身就在一张
RTX 4090 上量的生成速度（35M 每帧 0.030 秒、138M 每帧 0.075 秒，纯 latent 生成
不含解码）。你的训练步数和 wall-clock 取决于卡和配置，先跑短程看 loss 斜率，
再决定投入；评测协议（`--maskgit_steps 2`、v1.1 验证集）保持与榜单一致，数字
才有对照意义。

## 9. 验收

验收清单：

- [ ] Sonic 预训练模型交互体验完成，笔记里记了至少一处穿帮及其出现步数；
- [ ] tinyworlds 三阶段损失曲线齐全；视频 tokenizer 重建里球、球拍、比分认得出；
- [ ] 可解释性矩阵完成：4 码 × 3 起点，每格一句客观描述，每行一个三选一判定，
      种子与文件名可复核；
- [ ] 穷人版 $\Delta_t$ PSNR 完成：推断动作组与随机动作组的 MSE 各报 3 个种子的
      均值，方向正确（推断组更低），两组数字写进报告；
- [ ] 1xgpt 官方 GENIE_138M 评测数字与 README 榜单同量级（TTF 交叉熵 8.79、
      LPIPS 0.207 附近），对不上时能说出自己排查了哪些协议差异；
- [ ] 自训 baseline 的四件套证据 + 评测数字 + 至少一段生成动图；
- [ ] 落差对照笔记至少三条具体差异，每条注明出处文件；
- [ ] 能口头回答：LAM 的编码器和解码器为什么训完必须扔掉？（编码器要偷看下一帧，
      推理时下一帧还不存在；它的历史使命只是把码本训出来，之后动作码由玩家的
      整数输入直接查表。）
- [ ] 能口头回答：FSQ 为什么天生不坍缩？（没有可学习的码本向量，格点写死，不存在
      "没被选中就得不到梯度"的对象。）

## 10. 排错

| 症状 | 最可能的原因 | 怎么确认 | 怎么修 |
|---|---|---|---|
| 所有动作码生成无可辨差异 | LAM 退化（动作没编进码）或动力学无视条件 | 看 Step 6 两组 MSE 差距是否为零；查 LAM 训练损失是否明显低于纯外推 | 确认 `configs/dynamics.yaml` 的 `use_actions: true`；确认没动过 `keep_rate` 和方差辅助损失；重训 LAM |
| 码有分岔但含义跨起点乱跳 | 码本容量与数据真实操作数不匹配，或动力学训练不足 | 数一数该游戏的真实操作有几种，对照 `n_actions` | 把 `n_actions` 降一档（必须是 2 的幂）重训 LAM 与动力学；或先加动力学步数 |
| 视频 tokenizer 重建一团糊 | 训练步数或数据不足 | 看重建损失是否仍在降 | 加步数；换更小的数据集先验证流程 |
| 训练一启动就 import 报错 | PYTHONPATH 没指到你的克隆路径 | 报错信息里是 `models.` 或 `utils.` 找不到 | 重新 export（Step 1），注意 README 示例路径是 `/workspace/tinyworlds` |
| 启动时卡在 wandb 登录 | 没设 `WANDB_API_KEY` | 终端提示 wandb 认证 | 设密钥，或 `use_wandb: false` |
| Mac 上训练极慢或算子报错 | 训练不适合 MPS | 报错含 mps 字样 | 训练放 NVIDIA 卡；Mac 只承担推理与交互（本课设计如此）；`compile` 关掉再试 |
| 每次推理起点一模一样 | Step 5 加的 `random.seed` 胶水忘了删 | 看脚本开头 | 一致关实验做完就删掉或换种子 |
| 1xgpt `build.sh` 失败 | Python 低于 3.10 或磁盘不够 | `python --version`；看下载报错 | 升级 Python；清磁盘重跑 |
| 官方 checkpoint 评测数字对不上榜单 | 解码步数或数据版本不一致 | 命令里 `--maskgit_steps` 是不是 2；数据目录是不是 v1.1 | 严格用 Step 8 的原文命令重跑 |
| 自训 baseline 交叉熵比 35M 榜单还高不少 | 规模与步数不够，正常现象 | 对照训练步数与官方投入 | 如实记录，别硬调；这条差距本身写进落差笔记 |
| 看到 8 点几的 loss 心里发慌 | 拿它和 LLM 的 loss 比了 | 回看 5.5 节的因子化账法 | 量纲不同，不可比；只和同协议的榜单数字比 |

## 11. 前沿与改造

Genie 本尊的三件套和你训的 tinyworlds 逐件对应，差的是三个量级的
规模：视频 tokenizer 约 200M 参数（patch 4，码本 1024，码本大小和 tinyworlds 的
1024 一样，巧合但好记）、LAM 约 300M（patch 16，8 个码）、动力学 10.1B，合计约
11B；数据是从约 24.4 万小时的公开游戏视频里用质量分类器筛出的 3 万小时 2D 平台
跳跃类实况；训练吃掉 942B token，用了 256 块 TPUv5p。推理每帧 25 步 MaskGIT、
温度 2。论文还有一个机器人版：2.5B 模型把 RT-1 系的机器人数据**当无标签视频**训练，
同一套码本长出了一致的手臂控制，这正是 1xgpt 那批数据想让社区接着走的方向。
再往后是两代只有博客没有权重的系统：Genie 2（2024 年 12 月，DeepMind 博客展示
单张图片生成可交互 3D 世界）和 Genie 3（2025 年 8 月，博客宣称实时 720p、24 帧
每秒、一致性维持数分钟、支持用文字提示改变世界事件）。两者均无论文细节、无开源
权重，按本课程的分档纪律属于**只讲，不能练**；它们在前沿地图上的位置留给第 12 课。

规模那半（参数、数据小时数、TPU pod）钱能解决，不是本课的事。
机制那半你已经全部摸过：三件套结构、$[0.5, 1)$ 的训练掩码率、极小动作码本、
动作走加性或 FiLM 注入，缩小版与本尊在这些点上是同一张图纸。真正还没解决的
机制问题是真实世界那关：连续动作对不上离散码本、镜头自运动抢走信息瓶颈的带宽，
这些在 1xgpt 的落差笔记里你已经见过，目前公开文献也没有定论。

动手改造清单（选做，预算按你的冒烟档折算）：

1. 码本容量扫描：`configs/training.yaml` 的 `n_actions` 从 4 改到 8、16（必须
   2 的幂），视频 tokenizer 复用，只重训 LAM（每档 10000 步，很快）和冒烟档动力学。
   预期：容量贴近 Pong 真实操作数时矩阵最干净；16 码时"无效果"与"含义重合"的
   行明显增多，码本容量是设计出来的先验，这个实验让你对"Genie 为什么选 8"有
   自己的数据。失败判据：三档矩阵无系统差异，通常说明动力学步数不足，先加练再
   下结论。
2. LAM 吃像素还是吃 token：复现 Genie 论文的消融方向。tinyworlds 的 LAM 本来
   吃原始帧（`models/latent_actions.py` 的 patch embedding 直接作用在像素上），改一版
   吃视频 tokenizer 的量化 latent，对比穷人版 $\Delta_t$ PSNR 和矩阵质量。预期方向
   与论文一致：像素版可控性更好（论文数字 1.91 对 1.33）。失败判据：两版无差，
   64×64 的分辨率下动作信号可能没有论文设定里那么容易被 tokenizer 压丢，如实记录，
   这不算复现失败，算边界条件笔记。
3. 给 1xgpt baseline 接上动作：数据集里带着原始动作流，README 明说用加性嵌入
   接入动作是顺手的事，但 baseline 没做，这是留给你的作业。改 `genie/st_mask_git.py`，
   把动作嵌入加到对应时间步的 token 表征上，35M 档配置单卡训练。预期：TTF 交叉熵
   相对不接动作的对照组下降（动作是未来帧的强预测信号）。失败判据：交叉熵不动，
   先查动作与帧的时间对齐和归一化，再考虑"2Hz 下动作信息量本来就有限"这个假设。
   这是本课含金量最高的改造：做完你就有了一个真实机器人数据上的动作条件世界模型。

Genie 论文两个核心结论在你的缩小版设置里都有对应物：其一，"同一
潜动作跨输入语义一致"，对应你的可解释性矩阵，哪怕只有一个码过了一致关，方向就
复现了；其二，"随机潜动作生成显著偏离真实未来"（$\Delta_t$ PSNR 为正），对应
Step 6 的两组 MSE 之差。另外，想在更接近论文的设置（CoinRun）上完整复现 Genie 的，
社区有 JAX 实现 [FLAIROx/jafar](https://github.com/FLAIROx/jafar)，训练预算超出
单卡主线，属于 8×A100 短租级别的加餐，课程蓝图把它列为选做。

## 12. 论文与延伸

1. Genie: Generative Interactive Environments（Bruce, Dennis, Edwards et al.,
   2024，[arXiv:2402.15391](https://arxiv.org/abs/2402.15391)），本课主论文。
   带着三个问题读：LAM 的解码器训练完就被扔掉，为什么还值得花约 300M 参数去训它
   （提示：它卖的不是重建，是给码本供梯度）？8 个动作是"发现"还是"设计"，
   论文在哪些段落把可玩性当成了约束条件？3.4 节的消融里，LAM 吃像素赢在可控性、
   吃 token 赢在 FVD，如果你的应用只看视频质量不要交互，你会怎么选？
2. MaskGIT: Masked Generative Image Transformer（Chang, Zhang, Jiang, Liu &
   Freeman, 2022，[arXiv:2202.04200](https://arxiv.org/abs/2202.04200)），带着
   两个问题读：摘要里"最高加速 64 倍"是怎么来的，步数从多少压到多少？按置信度
   先落定的贪心策略会不会"一步错、步步错"，论文的掩码调度设计怎么缓解这件事？
   读完回头看 tinyworlds 的指数调度：前几轮只揭一两个 token 的保守劲儿，就是对
   这个风险的回应。
3. Finite Scalar Quantization: VQ-VAE Made Simple（Mentzer, Minnen, Agustsson
   & Tschannen, 2023，[arXiv:2309.15505](https://arxiv.org/abs/2309.15505)），
   带着两个问题读：它把 VQ 生态里哪四类补丁（commitment 损失、码本重播种、code
   splitting、熵惩罚）全部删掉还能打平？隐式码本 $\prod L_i$ 的维数和格点数怎么
   选才能对齐一个给定的 VQ 词表？对照第 09 课你在 IRIS 里监控码本利用率的经历读，
   体会"换机制"和"打补丁"是两种不同段位的解法。
4. tinyworlds 的 README，本课锚定仓库的架构文档，图文并茂。带着问题读：
   动作 tokenizer 一节列的两道保险（除首帧全掩、batch 方差辅助损失）分别堵的是
   哪条退化路径？动力学一节的逐步揭开示意（约 1、2、5、20、50）对应源码里哪个
   函数？
5. 1xgpt 的 README 与 HuggingFace 数据卡（`1x-technologies/worldmodel`），
   带着问题读：三档 teacher forcing 场景（全自回归、时序 TF、全 TF）各自考察模型
   的什么能力，为什么压缩指标选了中间档？词表因子化 $p(x_1, x_2) = p(x_1)p(x_2)$
   牺牲了什么、换来了什么？
6. 选读：**Genie 2 与 Genie 3 的 DeepMind 官方博客**，不是论文，无技术细节可
   核对，当"前沿在往哪走"的路标读，具体判断标准等第 12 课的三分评测观建立后再下。

第 12 课把整个第三幕收官：GameNGen、Oasis、Genie 3、Cosmos、Sora 一类
系统全都自称世界模型，你会带着本课练出的手感，什么叫可控、怎么量落差、权重开不
开源意味着什么，建立"预测、生成、规划"三分的评测观，把这些系统一个个摆到地图
上，分清谁真的能支撑决策，谁只是一台漂亮的视频播放器。
