---
id: 26_vla_vs_world_model
title: "VLA 和世界模型各管一段"
summary: "OpenVLA / π0 直接吐动作，世界模型吐未来。桌宠什么时候该用哪一个？"
unit: embodied
play_tools: []
checkpoints:
  - "VLA 与世界模型的接口对照表。"
  - "至少一处写明 VLA 成功不等于理解物理。"
---

# 第 26 课：语言进、动作出之后，还要不要预测未来

> 类型：体验 + 研究比较（跑公开权重或精读评测协议，**不从头训练**，也不把 LoRA 微调写成 24GB 必做）<br>
> 建议周期：2-3 天<br>
> 硬件：单张 24GB 卡可做 OpenVLA 推理和 openpi 无机器人冒烟；OpenVLA LoRA 官方写明至少约 27GB，默认 batch 更要到约 72GB，24GB 只读说明不硬训；Mac/CPU 可完成全部精读、接口对照和三句指令分流<br>
> 锚定仓库：[openvla/openvla](https://github.com/openvla/openvla)（OpenVLA，Hugging Face 权重 `openvla/openvla-7b`），[Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)（pi0 / pi0-FAST，以及后来补进仓库的 pi0.5 权重卡）<br>
> 产物：一张 VLA 与世界模型的接口对照表、一份 LIBERO 协议精读笔记、三句桌宠指令的分流答案、一份「VLA 成功不等于理解物理」的书面说明

## 1. 这一课做什么

整门课的主干循环没变：观察先压成状态，再按动作预测下一状态，然后展开多条未来，给未来打分，最后选动作。第七幕在把这条循环接到身体上。第 24 课用视觉预测做「先想再推」；第 25 课用 ACT 和 Diffusion Policy 证明，没有显式动力学，只模仿专家轨迹，挥手和推笔也能很快像样。这一课上场的是另一类更大的策略：视觉-语言-动作模型（VLA：Vision-Language-Action model，吃图像和自然语言，直接吐机器人动作）。

VLA 的卖点很好懂。人对着桌宠说「把笔推过来」，不必再手写状态机，也不必先采集几百条「推笔」示范再训一个 ACT。模型把这句话和当前画面一起读进去，下一步关节命令就出来了。OpenVLA 把连续动作切成离散 token，像写句子一样自回归吐出；Physical Intelligence 的 $\pi_0$ 用流匹配（flow matching：从噪声积分出一条连续动作轨迹的生成方法）一次吐出一个动作块。两者都是策略，输出是 $a$，不是下一状态 $s_{t+1}$。

世界模型多走一步。它要回答的是：如果真的执行这个 $a$，杯子会不会倒、手会不会擦到显示器。桌宠毕业设计里，这两件事必须同时成立。语言任务（看我、把笔给我）适合交给 VLA 或第 25 课那种模仿策略；「伸手会不会碰到杯子」必须问一个会按动作分岔的世界模型。本课的零件就是这道闸：学会看清 VLA 的输入输出，再学会把世界模型接在它后面当安全过滤器（safety filter：动作出口上的拦截器，预测到危险就截断或改成停下）。

24GB 主线假设下，本课不复现 OpenVLA 的 970k 轨迹预训练，也不复现 $\pi_0$ 的一万小时配方。能跑就跑官方小推理；跑不动就把 LIBERO 评测协议读透，用第 17 课的尺子给公开数字打分。$\pi_{0.5}$ 的论文训练配方（异构共训、knowledge insulation）不是本课作业；openpi 后来虽然挂上了 $\pi_{0.5}$ 权重，本课仍把它标成只讲，不写训练命令。

术语速查：

| 术语 | 一句人话 |
|---|---|
| VLA | 视觉-语言-动作模型：图像加一句话进去，动作出来；是策略，不是 $P(s_{t+1}\mid s_t,a_t)$ |
| VLM | 视觉语言模型：图像加文字进去，文字出来；VLA 通常从它改头而来 |
| 动作离散化 | 把每个连续关节量切成固定个箱子，每个箱子变成一个「词」，好让语言模型吐动作 |
| 动作块（action chunk） | 一次推理吐出未来一小段动作，第 25 课 ACT 已经用过这个想法 |
| 流匹配 | 学一个从噪声指向真实动作块的速度场，推理时按欧拉法积出来 |
| action expert | $\pi_0$ 里专管本体感觉和动作 token 的一小撮权重，和读图读字的主干分开 |
| FAST | 一种把动作块压成更短离散序列的分词器，用来训自回归版 $\pi_0$-FAST |
| LIBERO | 语言条件的桌面操作仿真基准，四个套件分别考空间、物体、目标和长程 |
| 安全过滤器 | 世界模型先看一眼候选动作的未来，超标就拒绝执行 |
| 未归一化动作 | OpenVLA 先在分箱空间里吐整数，再按数据集的分位数区间还原成真实关节量 |

## 2. 问题

第 25 课结束时，你手里有一条能回放的示范轨迹，和一张「模仿策略对世界模型」的对照表。模仿很快，失败也典型：没见过的摆法会僵，杯子在轨迹外会扫下去，语言指令稍微换词就不认。VLA 看起来像是把「换词就不认」治好了。它在互联网图文上预训练过，又在几十万条真机示范上继续训，论文数字也确实比从零训的 Diffusion Policy 好看。于是一个很自然的问题会冒出来：语言进、动作出的大模型，还要不要动力学？

要拆开问，否则会被成功率带着跑。

第一，VLA 到底输出什么。OpenVLA 的 `predict_action` 返回的是 7 维末端增量加夹爪（BridgeData V2 / WidowX 设定下），$\pi_0$ 的 `policy.infer` 返回的是一个动作块。两边都没有「下一帧杯子在哪」这个张量。任务成功，只说明在评测分布里，这些动作常常够用。它不说明模型内部有一张可查询的桌子。

第二，公开数字是按什么协议量的。LIBERO 的四个套件（Spatial / Object / Goal / Long）各自固定 10 个语言条件任务，OpenVLA 论文附录和仓库 README 用的是 3 个种子 × 每任务 50 次、共 500 次 rollout。$\pi_{0.5}$ 在 openpi 的 LIBERO README 里另报了一张表。两张表的模型、微调数据和是否改过演示集都不一样，不能横着减出「谁强 20 个点」。本课要你自己用第 17 课的口径把这些数字重新贴标签：这是规划好（下游成功率），不是预测准，更不是「理解物理」。

第三，桌宠上的分工一旦画错，第八幕会直接把杯子扫下桌。人对着镜头说「把杯子推到桌沿」，一个只会跟指令的 VLA 可能真去推；一个只会上滤镜的世界模型又听不懂这句话。正确接法是：VLA 或模仿策略负责把语言变成候选动作，世界模型负责说「这段动作接下来两秒会不会越界」，越界就拒绝。本课的互动题就是逼你当场勾选，而不是课后再补一句「还是要安全层」。

档位先写死，避免后面实验段口气滑掉。跑 `openvla/openvla-7b` 或 openpi 官方检查点的推理，是体验。精读 LIBERO 协议、对照两篇论文的表、设计安全过滤器，是研究比较。从头训 OpenVLA 或 $\pi_0$ 基座，本课不做，24GB 也做不了。$\pi_{0.5}$ 的完整共训配方按只讲处理。

## 3. 准备

- 第 25 课的对照笔记在手边：ACT / Diffusion Policy 的输入输出、它们为什么不是世界模型。第 24 课的「先在想象里推一把再动手」和第 03 课的动作对换，本课会当对照尺用。没写完那两课也可以读本课，但接口表会空一列。
- 第 17 课的三分评测习惯：预测准、生成真、规划好，分开写，不许合成一个总分。
- 一台能上 Hugging Face 的机器。OpenVLA 7B 的 bf16 权重大约是「70 亿参数 × 2 字节」这个量级，再加激活，单张 24GB 卡做推理通常够；Mac 只做代码阅读和协议精读，不假装能实时控臂。
- Python 3.10 左右的环境（OpenVLA README 写的是 3.10，并钉过 PyTorch 2.2.*、transformers 4.40.1、tokenizers 0.19.1、timm 0.9.10）。openpi README 要求 Ubuntu 22.04 加 NVIDIA GPU，用 uv 管依赖，官方写明暂不支持其他操作系统。
- 磁盘：OpenVLA 权重加依赖预留 20GB 以上；若只读 LIBERO 协议、不下载权重，几百 MB 的仓库克隆就够。
- Hugging Face 账号。`openvla/openvla-7b` 基于 Llama 2，权重受 Llama Community License 约束；仓库代码是 MIT。openpi 的检查点默认从 `gs://openpi-assets` 拉到 `~/.cache/openpi`，可用环境变量 `OPENPI_DATA_HOME` 改缓存目录。
- 不需要真机，也不需要买 SO-101。有臂的人可以把本课的安全过滤器接到自己的动作出口上，那是加分，不是及格线。

## 4. 学习目标

1. 在白纸上画出三类接口：VLA、第 25 课的模仿策略、世界模型，写清各自吃什么、吐什么、缺什么。
2. 说出 OpenVLA 怎样把连续动作变成 Llama 词表里的 token（256 箱、用 1% 和 99% 分位数而不是最小最大值、覆盖词表末尾 256 个罕用 token），以及为什么这和「下一状态」不是一回事。
3. 说出 $\pi_0$ 的流匹配推理在做什么：从噪声出发，按论文里的前向欧拉、10 步、步长 $0.1$，积出一个动作块；action expert 大约 3 亿参数，主干是 PaliGemma 3B。
4. 独立读完 OpenVLA 与 openpi 当前 README 的硬件表，标出哪些命令是推理、哪些是 LoRA、哪些是满血微调，并明确 24GB 能做哪几条。
5. 用 LIBERO 四个套件的公开成功率，按第 17 课的尺子打分：这是规划好，不是预测准；并写下一句「VLA 成功不等于理解物理」。
6. 给三句桌宠指令勾选「走 VLA / 走世界模型 / 拒绝」，书面解释为什么「把杯子推到桌沿」不能直接执行。
7. 设计一个最小安全过滤器：VLA 出候选动作，世界模型展开 1-2 秒，越界则截断。不要求这课把过滤器训出来，要求接口画对。

## 5. 原理

五个机制。每个都落到「桌宠到底该问谁」这一句上。

### 5.1 VLA 是策略，世界模型是未来

把一张桌子上的问题写成两句，差别立刻出现。

人对着镜头说「把笔推过来」。VLA 做的是：

$$
a_t = \pi_\theta(I_t, \ell)
$$

$I_t$ 是当前图像，$\ell$ 是指令，$a_t$ 是此刻该执行的动作（或一小段动作块）。它回答「现在该怎么动」。第 25 课的 ACT 和 Diffusion Policy 也是这条式子，只是 $\ell$ 常常缺席，换成「当前观察进、动作块出」的条件生成。

世界模型做的是：

$$
P(s_{t+1} \mid s_t, a_t)
$$

它回答「如果这么动，下一刻世界怎样」。同一起点换一个 $a$，预测必须分岔，这是第 03 课就钉死的试金石。VLA 再强，默认也不提供这个条件分布。你可以事后从它吐出的动作里反推「它大概想去哪」，那是你在读策略，不是模型在报状态。

类比：VLA 像一个听得懂人话的熟练学徒，看一眼桌子就能伸手；世界模型像学徒脑子里的那张桌子。学徒出手很快，但问他「杯子会不会倒」，他只能用过去见过的类似场面来猜，不会在脑子里把杯子的位置往前滚两秒。类比失效处：实验室里的 VLA 见过的场面远比你家书桌多，成功率可以很高；你家书桌上多出来的那杯水，不在它的示范分布里，它仍然没有一张可查询的桌子。

验证方法本课就能做：看源码返回值。OpenVLA 的 `predict_action` 和 openpi 的 `policy.infer(...)["actions"]` 都是动作。没有 `next_state`，没有想象视频，没有碰撞概率。

### 5.2 历史对照：RT-2 把动作写成句子

VLA 这个名字来自 RT-2（Brohan et al., arXiv:2307.15818）。做法刻意简单：已有的视觉语言模型本来就会根据图像生成文字；把机器人动作也编成文字 token，和互联网图文任务混在一起微调。推理时把吐出来的 token 解回末端位置、旋转和夹爪，再送给机器人逐步执行。论文报告了约 6000 次真机评测，并展示一些互联网预训练带来的「额外能力」，例如按语义关系选物体、用思维链处理多步语言。权重未开源。本课只把它当家族史：后来的 OpenVLA 几乎是同一张图纸的开源落地，$\pi_0$ 则把「动作必须是词」这条换成了连续生成。

RT-2 留下的一个教训，本课要一直用：能把「最小的那个」或「靠近杯子的那个」说对，仍然只是语言侧的泛化。它不自动等于接触力学、摩擦和「推到桌沿会掉」。评测如果只报任务成功率，这两种能力会被混成一个数。

### 5.3 OpenVLA：双视觉编码器加 256 箱动作词

OpenVLA（Kim et al., arXiv:2406.09246）是 70 亿参数的开源 VLA，主干来自 Prismatic VLM。图像同时过 SigLIP（语义对齐）和 DINOv2（空间细节），拼接成 patch token，经投影层送进 Llama 2。语言指令是普通文本。动作走离散化：每个连续维度单独切成 256 个箱子，箱宽由训练数据该维度的 1% 和 99% 分位数均匀划分。用分位数而不是最小最大值，是为了不让个别极端样本把整个区间撑宽、把有效分辨率压没。得到 $N$ 个 0 到 255 的整数之后，论文的做法是覆盖 Llama 词表里最少被用到的末尾 256 个 token（Llama 预留的 special token 只有 100 个，装不下 256 个动作词）。训练目标就是普通的自回归交叉熵，只是监督信号是这些动作 token，不是自然语言。

记第 $i$ 维连续动作为 $a_i$，分位数边界为 $q_{0.01}$、$q_{0.99}$：

$$
t_i = \mathrm{clip}\!\left(\left\lfloor 256\cdot\frac{a_i-q_{0.01}}{q_{0.99}-q_{0.01}}\right\rfloor,\,0,\,255\right)
$$

模型学的是

$$
\mathcal{L} = -\sum_{i=1}^{N}\log p_\theta(t_i\mid I,\ell,t_{<i})
$$

推理时把 $t_i$ 按同一套分位数区间反解回连续值。仓库 README 的最小例子里，这一步被包进 `predict_action(..., unnorm_key="bridge_orig")`：`unnorm_key` 指定用哪份数据集的统计量来还原，BridgeData V2 这条线对应的键是 `bridge_orig`。键选错，动作数值会对，物理单位会错，真机上就是乱伸。

论文主结果（摘要口径）：在 29 个任务、多种身体上，7B 的 OpenVLA 绝对成功率比 55B 的闭源 RT-2-X 高 16.5%；微调到新场景时，比从零训的 Diffusion Policy 高 20.4%。这些数字是任务成功率，量的是规划好。数据是 Open X-Embodiment 里约 97 万条真机演示，混合物在 `prismatic/vla/datasets/rlds/oxe/mixtures.py` 的 `Open-X Magic Soup++`。预训练算力不在 24GB 范围：仓库另有 `vla-scripts/train.py` 做从零训 VLA，README 的示例是单机 8 卡。本课禁止把那条命令写成你该跑的作业。

LoRA 是论文和仓库共同推销的消费级路径。论文表里 LoRA rank=32 只训约 1.4% 参数，在一组 Franka 桌面任务上与全量微调成功率相当，单卡 A100 大约 10-15 小时。仓库 README 写得更工程：单卡至少约 27GB；`--batch_size 16` 且 `--grad_accumulation_steps 1` 时约 72GB。24GB 卡低于 27GB 这条线，本课把 LoRA 当说明，不当必做。4-bit 量化推理是论文的另一项贡献：与 bfloat16 的下游成功率相当，显存明显下降。真要在更小的卡上只看 `predict_action` 能不能吐出 7 个数，优先走量化，而不是开始写训练脚本。

LIBERO 附录（论文 v2，仓库 README 同步抄了表）把四个套件的成功率写成：

| 方法 | Spatial | Object | Goal | Long | 平均 |
|---|---|---|---|---|---|
| 从零训 Diffusion Policy | $78.3\pm 1.1\%$ | $92.5\pm 0.7\%$ | $68.3\pm 1.2\%$ | $50.5\pm 1.3\%$ | $72.4\pm 0.7\%$ |
| 微调 Octo | $78.9\pm 1.0\%$ | $85.7\pm 0.9\%$ | $84.6\pm 0.9\%$ | $51.1\pm 1.3\%$ | $75.1\pm 0.6\%$ |
| 微调 OpenVLA | $84.7\pm 0.9\%$ | $88.4\pm 0.8\%$ | $79.2\pm 1.0\%$ | $53.7\pm 1.3\%$ | $76.5\pm 0.6\%$ |

每个数是 3 种子 × 500 次 rollout（10 任务 × 50 次）。仓库强调他们改过官方演示集再微调，改法见 `experiments/robot/libero/regenerate_libero_dataset.py`。评测时必须 `--center_crop True`，因为训练用了 90% 面积的随机裁剪，测试取中心 90%。协议细节比百分点更值钱：换一套裁剪，表上那些「高 1 个点」就会换人。

### 5.4 $\pi_0$：连续动作块，流匹配，action expert

$\pi_0$（Black et al., arXiv:2410.24164，RSS 2025）面对的是另一头的问题。洗衣折叠、装盒子、50Hz 双臂，动作轨迹又快又弯，256 箱的自回归一步一维，既慢又难拟合。它的解法是：主干仍用预训练 VLM（PaliGemma，约 30 亿参数），另外加约 3 亿参数的 action expert，两者在注意力里会合，但权重不共享。观察 $\mathbf{o}_t$ 含多路 RGB、语言指令和本体感觉；输出是未来 $H$ 步的动作块 $\mathbf{A}_t=[\mathbf{a}_t,\ldots,\mathbf{a}_{t+H-1}]$，论文里 dexterous 任务取 $H=50$。

生成方法是流匹配。推理时从标准正态噪声出发，按前向欧拉把学到的向量场积到 $\tau=1$：

$$
\mathbf{A}_t^{\tau+\delta}=\mathbf{A}_t^{\tau}+\delta\,\mathbf{v}_\theta(\mathbf{A}_t^{\tau},\mathbf{o}_t)
$$

论文写明实验用 10 步，对应 $\delta=0.1$。观察侧的键值可以缓存，每一步只重算动作侧。这和第 25 课 Diffusion Policy「把动作轨迹当要去噪的样本」是同一类连续生成，差别是条件里多了互联网预训练过的图文主干，以及专门的 action expert。

openpi 仓库把这套东西做成可下的检查点。README 的硬件表：

| 模式 | 显存 | 例子 |
|---|---|---|
| 推理 | 大于 8 GB | RTX 4090 |
| LoRA 微调 | 大于 22.5 GB | RTX 4090 |
| 全量微调 | 大于 70 GB | A100 80GB / H100 |

基座权重在 `gs://openpi-assets/checkpoints/pi0_base` 和 `pi0_fast_base`，预训练口径是 1 万小时以上的机器人数据。微调专家检查点包括 DROID、ALOHA 折毛巾 / 开保鲜盒 / 拔笔帽。仓库自己写得很直：$\pi_0$ 是为他们自己的机器人发育的，拿到 ALOHA 或 DROID 上「可能有用，也可能没用」。本课把这句话当体验档的使用说明，不把它改成「你的桌宠装上就能折毛巾」。

$\pi_0$-FAST（Pertsch, Stachowicz et al., arXiv:2501.09747）把动作块先做离散余弦变换（DCT）再分词，重新走自回归。论文报告在同一套数据上训练比流匹配版快约 5 倍，灵巧任务上能对上 $\pi_0$。openpi 同时提供 `pi0_fast_base` 和 `pi0_fast_droid`。FAST 解决的是「按维分箱的动作词太长、高频灵巧任务拟合差」，不是「模型开始预测未来」。它仍然吐动作。

$\pi_{0.5}$（arXiv:2504.16054）在 $\pi_0$ 上加异构共训：多机器人、高层语义、网页数据等，论文展示在从未见过的家庭环境做打扫。2025 年 9 月起，openpi README 列出了 `pi05_base`、`pi05_libero`、`pi05_droid`，并写明仓库里训练和推理目前只接流匹配头。LIBERO README 给出微调检查点 `gs://openpi-assets/checkpoints/pi05_libero/` 的表：Spatial 98.8、Object 98.2、Goal 98.0、Libero-10 92.4，平均 96.85。这些权重可以下，不代表本课要你复现 knowledge insulation。完整共训配方按只讲处理，作业里不要出现 `scripts/train.py pi05_libero`。

### 5.5 同一张桌子上的接法：VLA 提议，世界模型否决

把三件套接到桌宠动作出口，数据流是这样的。

摄像头给出 $I_t$，人对着麦克风说 $\ell$。VLA（或第 25 课的模仿策略，若指令很固定）给出候选 $a^{\text{prop}}$。世界模型吃当前桌面状态 $s_t$ 和这个 $a^{\text{prop}}$，展开 $H$ 步，得到杯子位置、手与杯的距离、物体是否越过桌沿。打分函数只需要几条硬规则：预测到桌沿 5 厘米内、预测到手杯距离小于安全阈值、预测置信度过低。任一触发，执行层改成停止或改用表情/语言提示，不把 $a^{\text{prop}}$ 发给电机。

$$
a^{\text{exec}} =
\begin{cases}
a^{\text{prop}} & \hat{r}(s_t,a^{\text{prop}})\ \text{全部低于阈值} \\
a^{\text{stop}} & \text{否则}
\end{cases}
$$

$\hat{r}$ 是世界模型给出的风险向量，不是 VLA 的 softmax。VLA 可以继续当「嘴」：把「看我」「把笔推过来」映射成转头或伸手。世界模型当「克制」：伸手这条未来如果会碰到杯子，就否决。第 32 课五件行为里的「克制」和「失败承认」，用的就是这道闸。本课只要求你把接口画对，并在三句指令上勾选正确的出口；真把 $\hat{r}$ 训出来，是第 30 课的事。

有一个容易画错的变体：让 VLA 自己「想象」失败。OpenVLA 没有未来分支，问它「如果反方向推呢」它只会再吐一个动作，不会给你两段分岔的桌子。要分岔，必须有一个显式吃 $a$ 的预测器。这是本课和第 25 课共同的结论，只是本课的策略更大、更会听人话，更容易让人误以为内部已经有世界。

## 6. 源码导读

两个仓库都按「先看入口，再看动作怎么变成数字，最后看评测脚本假定了什么」这条线读。路径以当前主分支为准，读之前用 `git log -1 --oneline` 记 commit。

### 6.1 OpenVLA

克隆后先看 README 的 Getting Started，再进这些文件：

| 路径 | 读什么 |
|---|---|
| `prismatic/models/vlas/openvla.py` | 训练用的 OpenVLA 类，动作如何接在 VLM 后面 |
| `prismatic/models/load.py`、`materialize.py`、`registry.py` | 权重和架构怎么被叫出来 |
| `prismatic/models/vlms/`、`prismatic/models/backbones/` | Prismatic VLM、视觉骨干（SigLIP / DINOv2） |
| `prismatic/vla/datasets/rlds/oxe/mixtures.py` | `Open-X Magic Soup++` 混合物，论文 97 万条轨迹的清单 |
| `prismatic/vla/datasets/rlds/oxe/configs.py`、`transforms.py` | 新数据集要登记的配置和变换 |
| `vla-scripts/finetune.py` | LoRA / 量化 LoRA / 全量微调的同一入口，24GB 只读参数，不跑 |
| `vla-scripts/train.py` | 从零训 VLA，多卡作业，本课禁止当 24GB 命令抄 |
| `vla-scripts/deploy.py` | REST 服务，把推理从机器人本机拆出去 |
| `experiments/robot/openvla_utils.py`、`robot_utils.py` | 评测时怎么加载模型和卸动作 |
| `experiments/robot/libero/run_libero_eval.py` | LIBERO 评测入口，命令行参数以 README 为准 |
| `experiments/robot/libero/regenerate_libero_dataset.py` | 他们改官方演示集的脚本，协议的一部分 |
| `experiments/robot/libero/libero_requirements.txt` | 仿真评测额外依赖 |
| `requirements-min.txt` | 只推理时的轻量依赖 |

Hugging Face 上的 `openvla/openvla-7b` 走 `transformers` 的 `AutoModelForVision2Seq` / `AutoProcessor`，`trust_remote_code=True` 会拉模型卡里的远程代码。最小推理在 README 里已经写全，本课 Step 2 几乎原样使用，只是把摄像头换成一张纯色占位图，用来确认返回的是动作向量。

读 `finetune.py` 时盯三个开关：`--lora_rank`（官方示例 32）、`--batch_size` 与 `--grad_accumulation_steps` 的乘积（有效 batch）、`--image_aug`。README 写过：在已经包含 BridgeData V2 的预训练模型上把 `--image_aug` 设成 False，训练日志里的 `action_accuracy` 会接近 100%，因为模型见过这份数据。这个数字不是「新任务已经学会」。

### 6.2 openpi

目录比 OpenVLA 更「库」一些：

| 路径 | 读什么 |
|---|---|
| `src/openpi/models/pi0.py`、`pi0_config.py` | 流匹配 $\pi_0$ |
| `src/openpi/models/pi0_fast.py`、`gemma_fast.py` | 自回归 FAST 版 |
| `src/openpi/models/gemma.py`、`siglip.py`、`vit.py` | PaliGemma / 视觉侧 |
| `src/openpi/models/lora.py` | LoRA 实现（JAX 路径；PyTorch 路径 README 写明暂不支持 LoRA） |
| `src/openpi/models/tokenizer.py` | 文本与动作相关的分词 |
| `src/openpi/policies/policy_config.py` | `create_trained_policy`，README 推理示例的入口 |
| `src/openpi/policies/libero_policy.py` | `LiberoInputs` / `LiberoOutputs`，仿真观察和模型输入的映射 |
| `src/openpi/training/config.py` | `TrainConfig`、LIBERO 数据配置、各实验名 |
| `src/openpi/shared/` | 含检查点下载 `download.maybe_download` |
| `scripts/serve_policy.py` | 策略服务器，默认听 8000 端口 |
| `scripts/compute_norm_stats.py`、`scripts/train.py` | 微调前统计量和 JAX 训练入口，本课不跑 $\pi_{0.5}$ 训练 |
| `examples/simple_client/main.py` | 无机器人冒烟：随机观察，打推理频率 |
| `examples/libero/README.md`、`examples/libero/main.py` | LIBERO 评测，官方推荐 Docker |
| `examples/inference.ipynb` | README 指向的推理笔记本 |
| `docs/remote_inference.md` | 机器人本机和 GPU 服务器拆开 |

README 的推理示例有一处要对笔记：注释写「这里用我们的 $\pi_0$-FAST-DROID」，紧接着的代码却是 `get_config("pi05_droid")` 加 `gs://openpi-assets/checkpoints/pi05_droid`。本课体验档如果做无机器人冒烟，优先走 `examples/simple_client` 的 `--help` 和官方成对命令，避免把注释里的模型名和代码里的检查点抄串。PyTorch 路径是 2025 年 9 月补的，README 写明暂不支持 $\pi_0$-FAST、混合精度训练、FSDP、LoRA、训练期 EMA。

### 6.3 对照着读，三件事必须能指到行

1. 动作在哪一个张量里离开模型。OpenVLA 是 `predict_action` 的返回值；openpi 是 `infer` 字典的 `"actions"`。
2. 数据集统计量在哪用。OpenVLA 的 `unnorm_key`，openpi 的 `compute_norm_stats.py` 和 norm stats 重载文档 `docs/norm_stats.md`。
3. 评测脚本默认跑多少试次、裁不裁剪、用哪份检查点。OpenVLA 是 500 次加中心裁剪；openpi LIBERO 默认检查点是 `pi05_libero`，本课只读数字，不把它写成你训出来的。

## 7. 实验

先做分流题，再碰仓库。题目做错了，后面的接口表才有东西可改。

### Step 0: 三句指令，先勾选再揭晓

假设桌宠的工作空间是 60 cm × 40 cm 的桌子。桌上有一支笔、一只装了半杯水的杯子、一台笔记本。人对着镜头说下面三句话。每一句在三个出口里只选一个：走 VLA（或同等的语言条件策略）、走世界模型（问未来再决定）、拒绝执行。

把选择抄进笔记，再往下读。

| 指令 | 你的选择 | 一句话理由 |
|---|---|---|
| 看我 |  |  |
| 把笔推过来 |  |  |
| 把杯子推到桌沿 |  |  |

参考答案（先自己勾完再看）：

| 指令 | 出口 | 理由 |
|---|---|---|
| 看我 | 走 VLA | 语言任务、动作幅度小、碰撞风险低；模仿策略也能做，但本课的语言入口是 VLA |
| 把笔推过来 | 先走 VLA 出候选，必须再走世界模型过滤 | 听得懂「笔」和「过来」是 VLA 的活；路上有没有杯子、会不会带倒，是 $P(s_{t+1}\mid s_t,a)$ 的活 |
| 把杯子推到桌沿 | 拒绝 | 这正是第八幕要避免的失败。世界模型如果预测液体过沿，过滤器应否决；即使预测器暂时还没有，安全策略也是直接拒绝，不能「先推看看」 |

常见错法：三句全交给 VLA（把听懂指令当成已经安全）；三句全交给世界模型（世界模型不解析开放语言）；第二句只走 VLA、第三句走世界模型但不拒绝（预测到过沿仍执行，过滤器等于没装）。笔记里写你当初勾错了哪一句，这比抄对表更有用。

### Step 1: 核对两份 README 的分档

```bash
git clone https://github.com/openvla/openvla.git
```

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
```

openpi 官方 README 写的是 SSH 地址 `git@github.com:Physical-Intelligence/openpi.git`，上面这条 HTTPS 等价；已经克隆过的目录要补子模块就用仓库写明的 `git submodule update --init --recursive`。记两个仓库的 `git log -1 --oneline`。然后只读、不动手训，在笔记里填：

| 命令或入口 | 仓库怎么写 | 本课档位 |
|---|---|---|
| OpenVLA `AutoModelForVision2Seq` + `predict_action` | Getting Started，轻量依赖见 `requirements-min.txt` | 体验，24GB 可试 |
| `vla-scripts/finetune.py` LoRA | 至少约 27GB；batch 16 约 72GB | 只讲说明，24GB 不跑 |
| `vla-scripts/train.py` 从零训 | 示例按单机 8 卡写 | 禁止当本课作业 |
| `experiments/robot/libero/run_libero_eval.py` | 四份微调检查点，默认 500 次 | 体验（有仿真）或只讲协议 |
| openpi 推理 `policy.infer` | 大于 8GB | 体验 |
| `examples/simple_client` | 随机观察测频率 | 体验，无机器人 |
| openpi LoRA 微调 | 大于 22.5GB | 可选加餐，非必做 |
| openpi 全量微调 | 大于 70GB | 只讲 |
| `scripts/train.py pi05_libero` | README 的微调示例 | 本课不跑 |

### Step 2: OpenVLA 最小推理（有 24GB 就做，没有就跳到 Step 3）

按 README 的轻量依赖装环境。仓库把安装命令写在注释里，本课拆成一条可复制的命令：

```bash
pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
```

若这台机器访问 raw.githubusercontent.com 不稳定，改成本地文件：先进入克隆目录，再 `pip install -r requirements-min.txt`。PyTorch 仍按你自己的 CUDA / MPS 在 [pytorch.org](https://pytorch.org/get-started/locally/) 另装，不要用仓库里那条写着 `UPDATE ME` 的示例 CUDA 版本。

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch

processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to("cuda:0")

image = Image.new("RGB", (224, 224), color=(180, 180, 180))
prompt = "In: What action should the robot take to pick up the pen?\nOut:"
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
print(type(action), getattr(action, "shape", None))
print(action)
```

预期：返回一个长度为 7 左右的向量（BridgeData V2 / WidowX 的末端增量加夹爪），数值已经按 `bridge_orig` 反归一化。画面是纯灰，动作不会「抓到笔」，这正是要点。把同一张灰图的指令改成 `push the cup to the edge of the table`，再跑一次，你会再得到 7 个数。模型没有拒绝通道，也没有「杯子会掉」这个输出。把它写进笔记：VLA 在垃圾观察上仍然吐动作，成功接口和安全接口不是同一个。

显存不够时，先去掉不存在于本段代码里的 flash-attn 选项（README 里那项是可选的，本段已经没写）。再不行，查阅论文的 4-bit 量化推理，不要开始改训练脚本。Mac 没有 CUDA 就停在「代码能读、权重能下到 CPU 但推理极慢」，不要为了出 7 个数把风扇转化。

### Step 3: LIBERO 协议精读（必做，不依赖真机）

打开 OpenVLA README 的 LIBERO 节和 LIBERO 论文（Liu et al., arXiv:2306.03310）。在笔记里回答，不许只抄表：

1. 四个套件各在转移什么？Spatial 换空间关系，Object 换物体，Goal 换目标，Long / LIBERO-10 是更长的 10 个任务。LIBERO-100 是 100 任务的纠缠套件，OpenVLA 仓库这条评测线没用它。
2. OpenVLA 的 500 次是怎么来的？10 任务 × 50 次 × 3 种子。单次成功率没有误差条资格。
3. 他们是否改了官方演示？是，见 `regenerate_libero_dataset.py`。和「在原始 LIBERO 演示上微调」不是同一协议。
4. 为什么必须中心裁剪？训练 90% 随机裁，测试取中心 90%。关掉 `--center_crop` 就是换协议。
5. openpi LIBERO README 的 96.85% 平均，用的是 `pi05_libero` 微调检查点，和 OpenVLA 那张 76.5% 的表不能减。把两张表并排时，每一列旁边写模型、是否改数据、评测脚本路径。

有仿真和 24GB 的人，可以按 README 跑一个套件。先装 LIBERO 再装评测依赖：

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
```

进入 `LIBERO` 目录后：

```bash
pip install -e .
```

回到 `openvla` 目录后：

```bash
pip install -r experiments/robot/libero/libero_requirements.txt
```

只跑 Spatial、并把每任务次数降到 2，用来确认脚本能进仿真，不用来报成功率：

```bash
python experiments/robot/libero/run_libero_eval.py --model_family openvla --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial --task_suite_name libero_spatial --center_crop True --num_trials_per_task 2
```

默认 500 次是论文口径，单卡要很久。2 次只证明管道通。想对表，再改回默认 50 次并固定 `--seed`。跑不起来的人停在上面 5 问，档位改成只讲，不算不及格。

### Step 4: openpi 无机器人冒烟（能装 uv 就做）

openpi README 要求 Ubuntu 22.04 加 NVIDIA，用 uv。克隆时已带 `--recurse-submodules` 的人，在仓库根目录：

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

```bash
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

`GIT_LFS_SKIP_SMUDGE=1` 是 README 原话，为了拉 LeRobot 依赖时跳过 LFS 大文件。先看客户端支持哪些环境：

```bash
uv run examples/simple_client/main.py --help
```

然后按 README 的无 Docker 双终端示例。终端 1：

```bash
uv run scripts/serve_policy.py --env DROID
```

终端 2：

```bash
uv run examples/simple_client/main.py --env DROID
```

预期：服务器开始下检查点（默认进 `~/.cache/openpi`），客户端打印推理频率。观察是随机的，动作没有任务意义。你要记录的是：返回字典里有没有未来状态、一次推理吐出的是一步还是一块、墙钟延迟大概多少。装不上 uv、不在 Ubuntu、或下不动 `gs://openpi-assets` 的人，改读 `examples/inference.ipynb` 和 `src/openpi/policies/policy_config.py`，把 `infer` 的输入键（外视、腕部相机、`prompt`）抄进接口表，档位标体验失败 / 只讲。

不要在这一步启动 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero`。那是 README 的微调示例，不是本课作业。

### Step 5: 填接口对照表

同一件桌面小事：把笔从桌子左侧推到人面前，路上可能挡着杯子。三列都填，空格用「无」不要用「差不多」。

| 项目 | OpenVLA / $\pi_0$ | 第 25 课 ACT 或 Diffusion Policy | 世界模型（第 03 / 24 / 30 课那种） |
|---|---|---|---|
| 主输入 | 图像 + 自然语言（$\pi_0$ 还吃本体感觉） | 图像 + 本体，语言通常很弱或没有 | 当前状态 + 候选动作 |
| 主输出 | 动作或动作块 | 动作块 | 下一状态或一段未来 |
| 训练信号 | 动作 token 交叉熵，或流匹配向量场 | 模仿专家动作 | 下一状态 / 表征 / 像素 |
| 换一句指令 | 设计上应当改动作 | 常常不理 | 本身不解析开放语言 |
| 换一个动作问未来 | 不会分岔出两张桌子 | 不会 | 必须分岔，否则模型是盲的 |
| 失败时你看得到什么 | 任务失败，内部没有「杯子掉了」这个变量 | 轨迹偏离示范 | 想象里的碰撞、过沿、漂移 |
| 桌宠用途 | 嘴：把「看我」「把笔给我」变成候选动作 | 固定技能：挥手、点头、推正 | 克制：否决危险候选 |
| 本课档位 | 体验推理 / 精读协议 | 第 25 课已做 | 本课只接接口，不新训 |

表底下写一句本课验收要检查的话：VLA 在 LIBERO 上的成功，只说明在该协议的语言条件任务里动作常常够用，不说明它有一张可查询的桌子，也不说明它理解接触和液体。

### Step 6: 在纸上设计安全过滤器

不写新训练代码。用伪代码级别即可，但输入输出类型要具体。

1. 输入：VLA 的 $a^{\text{prop}}$（OpenVLA 的 7 维或 $\pi_0$ 的动作块）、第 29 课那种桌面状态（本课没有真状态就用「杯子像素位置 + 桌沿线」的示意）。
2. 世界模型展开 1-2 秒（或 $H$ 步）。
3. 三条否决：预测杯沿越过桌沿 5 cm 内；预测手与杯距离小于你定的阈值；预测器对自己的未来不确定（例如多步漂移超过第 03 课那种偷懒基线）。
4. 否决时的执行：速度清零，屏幕或喇叭提示「会碰到杯子」，把事件写进日志。
5. 对照 Step 0：第三句指令应当在第 3 条之前就被产品策略拒绝；即使世界模型说「这次刚好不会掉」，「把杯子推到桌沿」仍然不该成为桌宠默认技能。

把这五条收进笔记，第 27 课会把它接到仿真里的截断，第 32 课会变成毕业行为「克制」。按第 33 课，VLA 默认 E3；过滤器真正改写了动作，才进入 E4。

## 8. 配置与预算

本课几乎没有「你自己的训练超参」。预算表按体验档来，把官方数字和你不该花的钱分开。

| 步骤 | 硬件 | 时间（参考） | 说明 |
|---|---|---|---|
| Step 0 三句分流 | 纸或编辑器 | 20 分钟 | 必做 |
| Step 1 克隆与分档表 | 任意机器 | 30 分钟 | 必做 |
| Step 2 OpenVLA 推理 | 单张 24GB | 权重下载数十分钟，推理一次分钟级 | 7B bf16 权重量级约 14GB，激活另计；失败则只讲 |
| Step 3 LIBERO 精读 | 任意 | 2-3 小时 | 必做；仿真评测可选 |
| Step 3 可选 2 次/任务冒烟 | 24GB + MuJoCo 显示 | 一小时内 | 不报成功率 |
| Step 3 可选 50 次/任务对表 | 24GB | 数小时到一天 | 仅当你要核对官方检查点 |
| Step 4 openpi simple_client | Ubuntu + GPU，大于 8GB | 下载检查点为主 | 非 Ubuntu 标体验失败 |
| Step 5-6 对照表和过滤器 | 任意 | 1-2 小时 | 必做 |
| OpenVLA LoRA（官方） | 至少约 27GB，示例 batch 约 72GB | 论文口径单卡 A100 10-15 小时 | 非必做 |
| OpenVLA 从零训 | 多卡，README 示例 8 卡 | 论文级 | 本课禁止 |
| openpi LoRA | 大于 22.5GB | 视数据 | 8 卡短租加餐，非本课及格线 |
| openpi 全量微调 | 大于 70GB | 视数据 | 只讲 |
| $\pi_{0.5}$ 共训配方 | 未作为本课作业开放 |  | 只讲 |

OpenVLA LoRA 若你以后自己加餐，超参以仓库示例为准，不要发明：`--lora_rank 32`、`--learning_rate 5e-4`、有效 batch 用 `--batch_size` 乘 `--grad_accumulation_steps` 维持稳定。24GB 上把 batch 降到 1 仍可能低于 README 写的 27GB 门槛，那时停，不要改学习率硬撑。LIBERO 微调检查点已经公开：

- `openvla/openvla-7b-finetuned-libero-spatial`
- `openvla/openvla-7b-finetuned-libero-object`
- `openvla/openvla-7b-finetuned-libero-goal`
- `openvla/openvla-7b-finetuned-libero-10`

评测吃这些权重，不要在 24GB 上从 `openvla-7b` 再 LoRA 一遍 LIBERO 来「复现附录表」。

数据体积：BridgeData V2 全量 TFDS 官方写约 124GB，本课不下载。修改后的 LIBERO RLDS 约 10GB，仅当你真的要微调时才按

```text
Hugging Face 数据集 openvla/modified_libero_rlds
```

去取。评测官方检查点不需要这份数据。

## 9. 验收

- [ ] Step 0 三句都勾过，并且能解释第三句为什么是拒绝而不是「让世界模型试试看」。
- [ ] 接口对照表三列填完，世界模型那一列的输出写的是未来或下一状态，没有误写成动作。
- [ ] 至少一处用自己的句子写明：VLA 成功不等于理解物理。建议写在 LIBERO 表旁边，带上协议（500 次、中心裁剪、是否改演示）。
- [ ] 分档表里，从零训练和 $\pi_{0.5}$ 微调示例被标成不跑。
- [ ] 能口头说出 OpenVLA 的 256 箱和 $\pi_0$ 的 10 步欧拉各自在解决什么，以及两者为何仍不是 $P(s_{t+1}\mid s_t,a_t)$。
- [ ] 安全过滤器五条接口齐全：候选从哪来、展开谁来做、三条否决、否决后做什么、第三句指令为何在过滤器之前就能拒。
- [ ] 若做了 Step 2：打印了动作向量形状，并记录「灰图仍吐动作、没有拒绝通道」。
- [ ] 若做了 Step 4：记录一次推理的输出键名和是否含状态。
- [ ] 证据目录最小集：两个仓库的 commit、分档表、对照表、三句答案、过滤器草稿。做了推理的人加上完整命令和返回值打印。

口头关：向没上过这门课的人讲清「桌宠的嘴和手怎么分工」。听完的人应当能指出：听懂「把笔给我」靠 VLA 或模仿；决定伸不伸手靠世界模型；「把杯子推到桌沿」应当拒绝。

## 10. 排错

| 症状 | 最可能的原因 | 怎么确认 | 怎么修 |
|---|---|---|---|
| `predict_action` 报没有 `unnorm_key` 或数值离谱 | 键和训练统计对不上 | 打印模型配置里的 dataset statistics 键名 | BridgeData 示例用 `bridge_orig`；LIBERO 检查点不要套 Bridge 的键 |
| 7B 一加载就 OOM | bf16 权重大约 14GB，再加激活和碎片 | `nvidia-smi` | 先保证没开无关进程；去掉 flash-attn；再考虑论文里的 4-bit 量化推理 |
| `trust_remote_code=True` 被拒 | transformers 安全策略或离线 | 报错文本 | 允许远程代码或改用仓库本地 `prismatic` 加载路径 |
| 依赖版本冲突 | README 钉过 transformers 4.40.1、timm 0.9.10 等 | `pip show transformers timm tokenizers` | 按 README 5/21/24 注记降回钉版本，不要擅自升 |
| LIBERO 评测明显差于表 | 没开中心裁剪，或检查点与套件不配 | 命令里是否有 `--center_crop True`，检查点名是否含对应套件 | 四个套件用四份检查点，不要拿 Spatial 的权重评 Object |
| `run_libero_eval.py` 缺模块 | 只装了 openvla，没装 LIBERO 和 `libero_requirements.txt` | `python -c "import libero"` | 按 Step 3 的顺序装 |
| openpi `uv sync` 卡在 git-lfs | 拉 LeRobot 子模块时砸了大文件 | 日志里的 LFS 报错 | 按 README 加 `GIT_LFS_SKIP_SMUDGE=1` |
| `serve_policy.py` 一直下权重 | `gs://openpi-assets` 网络或缓存盘满 | `~/.cache/openpi` 体积 | 设 `OPENPI_DATA_HOME` 到大盘；下不动就改读笔记本，标体验失败 |
| simple_client 有动作无意义 | 观察是随机的 | README 写明这是无机器人测试 | 预期如此，记录的是接口和延迟，不是任务成功 |
| 在 24GB 上跑 `finetune.py` batch 16 | README 写约 72GB | 立刻 OOM | 停。不要把它改成从零训。24GB 低于约 27GB 的 LoRA 门槛 |
| 想「复现」OpenVLA 预训练 | 把 `vla-scripts/train.py` 当本课作业 | 示例是 8 卡 | 删掉这条命令。本课没有复现档 |
| Mac 上装 openpi | README：测试于 Ubuntu 22.04，暂不支持其他系统 | 编译或运行失败 | 精读源码和协议，不装也算 Step 1/3/5/6 完成 |
| 两张 LIBERO 表直接相减 | 模型、数据、脚本都不同 | 一边是 OpenVLA 7B 四套件微调，一边是 $\pi_{0.5}$ 检查点 | 并排时只比协议，不比绝对值排名 |

## 11. 前沿与改造

前沿怎么做。2024 到 2026 年，开源 VLA 主要在改动作头，不在补世界模型。OpenVLA-OFT（Kim et al., 项目页 openvla-oft.github.io）把微调配方换成连续动作、动作块、多图输入，官方报告 LIBERO 四套件平均成功率 97.1%，推理比原版快数十倍；训练仍按 8 张 80GB 卡写，不是 24GB 作业。FAST 把离散动作块压短，给 $\pi_0$-FAST 和后来的自回归 VLA 提速。$\pi_{0.5}$ 用异构共训换开放世界里的打扫，openpi 后来挂了权重，但仓库注明只接流匹配头，knowledge insulation 的完整配方仍按论文阅读。另一头，第 24 课那种视觉预测控制和第 15 课 V-JEPA 2-AC 继续走「先预测再选动作」。两条线都在变强，接口差异没变：一边出 $a$，一边出未来。

我们差在哪。规模差：97 万条真机演示和一万小时数据，不是一张 24GB 卡能补的。机制差：桌宠真正缺的是动作出口上的未来，不是再大一个语言模型。钱能买到更好的 VLA 微调，买不到「伸手会不会碰到杯子」这条查询，除非另外训或加载一个世界模型。第 25 课已经证明模仿策略同样缺这条查询。本课只是把缺的那截画在更大的策略后面。

动手改造清单，全部允许在缩小设置里失败。

1. 给 `predict_action` 外包一层拒绝。改的是你自己的胶水，不改 `prismatic/`。输入加桌沿像素坐标和杯子框；若候选动作的 xy 增量指向杯且指令含「桌沿」「edge」「drop」，直接返回全零动作。预算：零训练，小时级。预期：Step 0 第三句被拒，第一句不受影响。失败：规则误伤「把笔推过来」（笔和杯框重叠时）。记录误伤率，不要用这条规则冒充世界模型。
2. 灰图对照。同一套 OpenVLA 权重，喂纯灰图、喂一张你桌子的照片、喂一张 LIBERO 官方图（若你下了评测环境）。指令固定为 `pick up the pen`。预算：三次推理。预期：三种观察下动作范数和方向都变；若几乎不变，说明这份权重在这个观察域上接近常数策略，体验结论要下调。失败：三种一模一样，仍把它写成「理解桌子」。
3. 动作对换探针（机制，不训 VLA）。把世界模型（第 03 课 MDN-RNN、第 24 课的动作条件预测，或第 30 课将训的桌面模型）和 VLA 并排：同一帧，VLA 给一个 $a$，你再手写一个反向 $a'$。世界模型必须分岔；VLA 对 $a'$ 没有接口，你只能再问它一次「反过来推」，它会另吐一个动作而不是两段未来。预算：推理级。预期：你能指出哪边缺 $P(s_{t+1}\mid s_t,a)$。失败：把 VLA 两次不同指令的输出，当成世界模型的两条想象。
4. 若有 8 卡短租：按 OpenVLA README 在**小**自定义 RLDS 上 LoRA，rank 32，有效 batch 靠梯度累积凑。预期：新任务的 `action_accuracy` 在有图像增广时从低处爬，而不是一上来 100%。失败：无增广、数据已在预训练混合物里、日志 100%，还把它写成新技能。预算按论文 10-15 小时单卡 A100 外推，8 卡只是墙钟变短。这是加餐，写进笔记时标可选。

顺手复现（方向性，不是数字）：

| 论文结论 | 缩小版 | 预期 |
|---|---|---|
| OpenVLA 微调后 LIBERO 平均约 76.5% | 跑官方 `openvla-7b-finetuned-libero-*`，协议对齐 500 次和中心裁剪 | 能接近表；2 次/任务冒烟不能复现这个数 |
| LoRA 与全量微调成功率相当 | 24GB 做不到公平对照 | 只读论文表，不装复现 |
| $\pi_0$ 流匹配可在大于 8GB 上推理 | simple_client | 能出动作块和延迟；不能验证洗衣折叠 |
| VLA 成功不等于有动力学 | 灰图仍吐动作 + 无 `next_state` 字段 | 本课必现，方向应与论文「VLA 是策略」一致 |

## 12. 论文与延伸

每篇只列你要用的读法。数字以你打开的版本为准，本课核对的是 HTML / abs 页。

1. Kim, Pertsch, Karamcheti et al. *OpenVLA: An Open-Source Vision-Language-Action Model*. arXiv:2406.09246（本课按 v3, 2024-09-05）。读第 3 节动作分箱、视觉融合和词表覆盖；读微调与量化；读附录 E 与仓库同步的 LIBERO 表。阅读问题：256 箱用 1% / 99% 分位数，比最小最大值少了哪种误差？LoRA rank=32 训了百分之几的参数，和全量微调比的是哪组 Franka 任务？LIBERO 表的 ± 是 3×500 次，还是单次 rollout？

2. Black, Brown, Driess et al. *$\pi_0$: A Vision-Language-Action Flow Model for General Robot Control*. arXiv:2410.24164（本课按 v4, 2026-01-08；RSS 2025）。读 action expert、动作块 $H=50$、欧拉 10 步 $\delta=0.1$、PaliGemma 3B 加约 3 亿动作头。阅读问题：流匹配解决了自回归 VLA 的哪两个具体困难（频率、轨迹形状）？观察侧 KV 缓存省的是哪一段计算？论文强调的预训练 / 后训练分工，和只在高质量演示上模仿，差在哪类失败（不会恢复）？

3. Brohan, Brown, Carbajal et al. *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*. arXiv:2307.15818。闭源，只讲。读「动作写成文本 token」这条家族配方，以及他们所谓的涌现语义能力依赖什么评测。阅读问题：RT-2 把 VLM 用在低层控制，和「VLM 只做高层状态机、底层另接控制器」差在数据流的哪一跳？论文里的成功仍然量的是任务，还是未来状态误差？

4. Liu, Zhu, Gao et al. *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*. arXiv:2306.03310。读四个套件各自转移的知识类型，以及他们把问题写成终身模仿而非稀疏奖励 RL 的原因。阅读问题：OpenVLA 仓库用的 Spatial / Object / Goal / 10，对应论文里的哪一种知识转移？为什么改演示集之后，你就不能把 76.5% 写进「官方 LIBERO 排行」而不加注？

5. Pertsch, Stachowicz, Ichter et al. *FAST: Efficient Action Tokenization for Vision-Language-Action Models*. arXiv:2501.09747。选读。阅读问题：FAST 用 DCT 压缩的是动作块还是图像？训练快 5 倍之后，模型有没有因此获得 $P(s_{t+1}\mid s_t,a_t)$？

6. Physical Intelligence. *$\pi_{0.5}$: a Vision-Language-Action Model with Open-World Generalization*. arXiv:2504.16054。只讲。阅读问题：异构共训转移的是语义、高层子任务还是接触力学？openpi 现在能下 `pi05_libero` 权重，和「复现 $\pi_{0.5}$ 论文」差了哪几块数据与损失？

延伸不必做：OpenVLA-OFT 的连续动作微调、LeRobot 文档里的 $\pi_0$ 封装、第 27 课要把本课的过滤器接到延迟和接触上。下一课的麻烦是：仿真里过滤器亮了绿灯，真桌子上第一小时仍可能摔，原因通常不在 VLA 的词表，而在标定、延迟和接触。
