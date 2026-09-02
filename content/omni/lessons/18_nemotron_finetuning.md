---
id: 18_nemotron_finetuning
title: "Nemotron 官方微调"
summary: "把权重、AutoModel、CORD-v2 数据和硬件版本全部锁死之后，官方的 LoRA recipe 能一步不差地复现出来吗？之后只换 dataset/collator，能把它迁到你自己的任务上吗？"
unit: alignment
play_tools: []
checkpoints:
  - "讲明白 30B total / 约 3B active 是怎么回事，还有 hybrid backbone 和 EP=8。"
  - "把 base→LoRA→save→新进程 load→resume→inference 这条链完整走一遍。"
  - "读懂 processor、collator、image flags 和 label mask，然后迁移到自定义监督任务。"
  - "分清权重、代码、数据、recipe、license 各开放到哪一步，别混为一谈。"
---

# 第 18 课：复现 Nemotron Omni LoRA 微调

本课复现 NVIDIA 公开的 CORD-v2 LoRA 微调流程。实验从固定模型、代码和数据版本开始，依次验证训练、保存、恢复、合并与推理，最后只替换数据接口，把同一方法迁移到一个小型自定义多模态任务。

## 1. 先复现公开的 Nemotron recipe

前 17 课使用 26M 可训练参数的 MiniMind-O 完成了流式交互、视觉与视频、MoE、混合架构、分布式训练、长上下文和三类后训练实验。这个规模适合验证机制，但不足以评估大型模型的知识、推理和部署约束。

第 19 课将使用 NVIDIA 开源的 Nemotron 3 Nano Omni 作为 Thinker。它有 30B 总参数，支持文字、图像、视频和音频输入，输出文字。本课先按照 NVIDIA 公开的微调 recipe，在 8 卡环境中完成 CORD-v2 收据字段抽取的 LoRA 微调，再保持训练方法不变，只替换数据接口迁移到自定义任务。实验会复用前面课程中的 hash 与 manifest、MoE 路由与专家并行、多模态数据 contract 和预注册评测方法。

资源约束需要单独核算。MiniMind mini 可以在一张 24GB 消费卡上约两小时完成；Nemotron 全参数微调需要 8×H100 80GB，官方实测每卡约 49 GiB，可训练参数约 31.5B。LoRA 冻结底座权重，只训练约 55M 个低秩参数，约占全参数的 0.2%，每卡显存约 30 GiB。它减少的是梯度和优化器状态，不会消除 30B 底座权重的驻留成本。第 8 节会逐项计算这些资源。

本课还验证 Nemotron 在目标 8 卡环境中的加载方式、专家分布、多模态 processor 输入，以及 adapter 的保存与合并。这些结果是第 19 课集成 Thinker 与 Talker 的前置条件。

跑通官方 400 步 LoRA 后，用同一张收据图片比较 base 和 LoRA 模型。目标是让微调模型输出严格的 XML 字段序列，并生成可在新机器上恢复的复现与迁移报告。

本课术语：

| 术语 | 简要解释 |
|---|---|
| Nemotron 3 Nano Omni | NVIDIA 开源的 30B 全模态模型：吃文字、图、视频、音频，只输出文字；第 19 课的 Thinker |
| 30B-A3B | 总参数 30B、每个 token 只激活约 3B：MoE 稀疏结构（忘了回第 11 课） |
| LoRA / adapter | 冻结原权重、只训旁路低秩小矩阵的省钱微调法；adapter 就是这组小矩阵加配置 |
| rank ($r$) / alpha | adapter 的"宽度"和缩放系数：$r$ 越大能学的花样越多，参数也越多 |
| full SFT | 全参数监督微调：约 31.5B 参数全部更新，只有 8×H100 80GB 跑得动 |
| EP=8 | 专家并行：MoE 的专家参数分摊到 8 张卡，token 被路由到持有目标专家的卡上算 |
| HBM | GPU 卡上的本地显存；8 张 24GB 卡不等于一张 192GB 卡 |
| CORD-v2 | 官方教程用的收据理解数据集：输入收据照片，输出结构化字段 |
| processor / collator | 前者把文字和媒体转成模型输入张量；后者把一条对话样本装配成 token、图像张量和 labels |
| revision / source lock | 权重、代码、数据的精确版本指纹；全部锁死，两次实验才可比 |

## 2. 本课复现什么

先把实验边界固定并写入报告首页——微调大模型这件事最容易吹牛，边界不清的报告会把"跑通了官方示例"说成"复现了 Nemotron"。本课覆盖以下公开可执行部分：

- 下载公开权重并核验许可；
- 运行官方推理路径；
- 按 NeMo AutoModel 官方 CORD-v2 示例跑 LoRA；
- 仅在满足 8×H100 80GB 硬件条件时，把 full SFT 当作可选参考；
- 将同一接口迁移到一个自定义监督任务；
- 核验保存、恢复、合并 adapter 与部署。

以下内容不在课程范围内，实验结果也不能据此扩展表述：

- 复现 30B-A3B 基座预训练；
- 复现论文全部 adapter/encoder training；
- 复现完整 `SFT → MPO → Text GRPO → Vision GRPO` 最终模型；
- 获得论文使用的全部私有、第三方和内部合成数据；
- 获得全部过滤器、teacher、reward 环境和训练基础设施；
- 得到原生 speech output 或 full-duplex 能力。

Nemotron 3 Nano Omni 接受 text、image、video 和 audio，**输出是 text**。模型不包含 MiniMind-O/Moshi 式 Talker，因此不具备原生语音输出或 full-duplex 交互能力，也不能称为完全开源的 GPT-4o 复刻。给它接上嘴是第 19 课的活。

先从模型卡逐项记录输入、输出、许可和 checkpoint 标识，并为每项附上原始链接。随后分别用文本、图像、视频和音频样例验证输入路径，并确认输出都是文本。模型卡声明与本地运行结果不一致时，先停止训练并记录版本差异。

## 3. 运行顺序与交付物

这是一项从教学模型迁移到大规模 hybrid MoE Omni 的工程实验，可以独立执行，不要求完成实验 1–17。前序课程中的 connector、MoE、Mamba、数据、SFT 和 post-training 原理用于解释现象，不会改变官方 checkpoint——你在玩具上拆过的每个零件，这里都能对号入座。

课程产物是一份带测量证据的 LoRA recipe 复现与迁移报告。先固定代码、数据和权重，再测 base，然后训练 LoRA；LoRA 通过后依次验证保存、恢复和合并，最后才迁移到自定义任务。每一步都读取上一步已经验收并锁定的产物。

full SFT 不属于课程完成条件。只有硬件满足 8×H100 80GB 要求时，才把它作为独立参考，用于理解官方显存与吞吐数据。

开始前，为上述六个阶段建立产物索引，给每个阶段指定输入、输出、hash 和通过条件。完成后，最终报告中的任意一项结果都应能追溯到对应 checkpoint、配置和数据版本。

## 4. 要验证的结论与失败条件

研究问题需要同时覆盖可执行性与能力增量：在锁定 AutoModel 代码、CORD-v2 revision、模型权重、配置和八路专家并行环境后，官方 LoRA recipe 能否从干净环境重复运行；相对 base，LoRA 能否提高 CORD-v2 结构化抽取；仅替换 dataset/collator 后，同一方法能否迁移到自定义任务。

开训前预注册以下假设：

- 官方 LoRA 配置能在 CORD-v2 上快速学会结构化字段输出；
- 两次独立 run 的安装、数据、训练动力学、adapter 保存/恢复和固定推理结果可对齐；
- validation 只用于选择 checkpoint；锁定后再在 test 上一次性测量 base→LoRA 增量；
- wrong-image 对照能检查 LoRA 的结果是否依赖输入图像；
- 自定义任务是否成功主要取决于 processor/collator/label contract，adapter rank 只是其中一个变量。

训练集只用于更新参数，validation 只用于选择 checkpoint，test 在此期间保持封存。选好 checkpoint 后，先冻结其文件 hash、processor、生成参数和解析器，再对 base 与 LoRA 各运行一次 test。看过 test 后不能换 checkpoint、改阈值或再次训练；如果确需修改，必须新建实验编号，并换用一份尚未看过的测试集。test 是考卷，只许拆封一次，拆过封还改答案的成绩不算数。

官方教程的 full SFT 数字只作受硬件条件限制的参考。它与 LoRA 的学习率、可训练参数量和 checkpoint 体积不同，不进入本课主要对照，也不是 LoRA recipe 可复现性的必要条件。

在 test 解封前，把主指标、最小有意义效应、回归容许范围、95% 置信区间算法、seed 和失败条件写入只读配置并生成 hash。置信区间采用成对重采样（bootstrap：反复重抽样本来估计结果波动范围的方法）：base 与 LoRA 每次抽到相同的样本编号，避免两组样本差异干扰比较。评测完成后逐条给出支持或反驳证据；不能用训练 loss 或 validation 结果代替一次性的 test 结果。

## 5. 官方数据：硬件报告的使用范围

官方教程的数字描述一个明确的硬件和配置组合。以当前 [NeMo AutoModel Nemotron-Omni 官方教程](https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/nemotron-omni)为准。

下面这些缩写会在本课反复出现：

- **EP（Expert Parallelism，专家并行）**：把 MoE 的不同专家参数放到不同 GPU 上。
  `EP=8` 表示专家参数分布在 8 个训练进程上，token 会被发送到持有目标专家的进程；
- **HBM（High Bandwidth Memory，高带宽显存）**：GPU 卡上的本地显存。总显存不能代替
  单卡 HBM，因为一张卡放不下的对象不会自动分散到其他卡；
- **BF16**：16 位浮点格式，通常用于训练权重和计算；**FP8**：8 位浮点格式，占用更少，
  但是否可用取决于 GPU、算子和训练配置；
- **NVLink**：GPU 之间的高速直连；**NVSwitch**：把多张 GPU 的 NVLink 连接组织成交换
  网络。二者都是硬件互联，不是训练算法；
- **NCCL**：NVIDIA 的多 GPU 通信库，负责 all-reduce、all-to-all 等集体通信。NCCL 测试
  检查的是多卡通信是否正确、是否达到合理速度。

官方本次观测使用的条件和结果如下：

- **要求：8×H100 80GB**；
- **MoE expert parallel：EP=8**；
- full SFT 观测约 **49 GiB/GPU**；
- LoRA 观测约 **30 GiB/GPU**；
- 教程 workload：CORD-v2 800 train samples、400 steps；
- 官方给出的训练时间约 full SFT 10 分钟、LoRA 6 分钟；
- full SFT trainable params 约 31.5B；
- LoRA trainable params 约 55M；
- vision tower 和 audio tower 在示例里冻结。

这些是官方特定软件、硬件和 config 下的观测，不能直接外推到其他 8 卡机器。开始前必须确认 GPU 型号、每卡 HBM、BF16/FP8 支持、GPU 互联方式、CUDA 与 driver 版本、NCCL topology，以及本地或共享存储的实测吞吐。任何一项未知时，硬件检查都不能标记为通过。

8×24GB 即使总显存达到 192GB，也不等价于 8×H100 80GB。EP 仍受单卡峰值、激活、通信和基础权重驻留限制——放不进一张卡的张量不会自己搬去别的卡。

运行 `nvidia-smi -L`、`nvidia-smi topo -m`、显存查询和 NCCL collective 小测，并保存完整输出。再用官方配置做不更新参数的初始化或短启动测试，记录每个进程的峰值 HBM。只有 GPU 型号、每卡 HBM、EP=8 和通信测试同时满足要求，完整 SFT 的硬件检查才能标记为通过。

## 6. 固定训练起点：模型权重与 processor

```text
checkpoints/exp18_start/
├── model/      # nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
├── revision.txt
├── files.sha256
├── license.txt
└── baseline_outputs/
```

本课默认模型 snapshot（某一精确版本下载到本地的完整文件副本）固定如下：

| 字段 | 固定值 |
|---|---|
| `model_id` | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` |
| `revision` | `24e67ea000b7c2837fc8f9488aa2008524fac8ba` |
| `verified_at` | `2026-07-23` |

若 NVIDIA 发布更新版，先新建实验 ID；同一报告中的两次 run 不能分别解析浮动 `main`——`main` 会移动，指纹才可比，这是第 1 课就立下的规矩。

必须从官方模型卡记录：

- 精确 model ID；
- revision/commit；
- config、processor 和 custom code revision；
- license：`nvidia-open-model-agreement`；
- 下载日期；
- BF16/FP8/NVFP4 具体变体；
- 是否启用 `trust_remote_code`。

不要混用不同变体 checkpoint。训练课默认 BF16；量化权重先用于推理 profile，不默认作为 QLoRA 起点。

下载 snapshot 后，生成按路径排序的逐文件 SHA-256 清单，并在 `baseline_outputs/` 保存四种模态的固定样例。清除进程缓存后重新加载，模型版本、文件 hash、processor 配置和固定样例的生成条件必须一致。任一文件发生变化，都要创建新的实验记录。

## 7. 学完后应能完成

本课结束时，应能根据配置、模块名称、批次导出和性能记录解释下列内容：

- 解释 30B total / 约 3B active 的 MoE 含义；
- 解释 hybrid Mamba/Attention、vision/audio tower、projector 与语言骨干的边界；
- 理解 EP=8 下 expert 参数和 token all-to-all；
- 区分 full SFT、LoRA、冻结 tower 和 adapter target modules；
- 读懂 Nemotron processor、collator、image flags 与 label mask；
- 在 8 卡上测 HBM、吞吐、通信和 checkpoint；
- 将官方 image→text recipe 迁移到一个自定义任务；
- 用 exact dataset/code revision 与逐文件 hash 重建输入；
- 准确说明权重、代码、数据、recipe 和 license 的开放程度。

验收时随机抽取一个 batch，说明 token、媒体 embedding、label mask 和 loss 的形状；再随机抽取一个 checkpoint，说明基础模型、adapter 与 optimizer state 的保存关系。解释必须与实际导出和文件清单一致。

## 8. 原理:边造边讲

本节把原"显存占用与 LoRA target modules"的四个机制按同一节奏展开：为什么需要（直觉）、怎么运转（机制）、精确定义（数学）、去哪里核对（代码落点）、怎么证明做对了（验证）。这四个机制合起来回答一个问题：30B 的模型为什么塞得进 8 张卡，塞进去之后哪些参数在动。

### 8.1 Sparse MoE：账面 30B，每次只付 3B

MoE 像一家有很多位专科医生的医院：每个病人（token）只挂 top-k 个号，账面医生（总参数）很多，每次看病只付出诊那几位的算力。所以"30B 总参数"和"每 token 激活约 3B"是两项不同指标：前者决定权重驻留显存，后者决定每步计算量。类比失效处：医院里病人不用换楼，MoE 在 EP 下 token 真的要跨卡搬运，搬运费直接算进 step time。

对 hidden state `h`，router 为每个 token 计算 expert 分数，再按模型配置选择 top-k routed experts；shared experts 则按模型定义参与计算。EP=8 将 expert 参数分布到 8 个 rank。token 按 router 结果经 all-to-all 发往持有目标 expert 的 rank，计算后再返回原序列位置。互联带宽和 expert 负载不均会直接影响 step time。第 11 课在玩具上手搓过同款路由，只是那时所有专家都挤在一张卡上。

`h` 的 shape 可写为 `[B, S, d_model]`；router 分数为 `[B, S, E]`。实际 `E`、`k` 和 shared expert 数必须从固定 revision 的 config 读取，不能凭模型名称推断——"A3B"只是命名口径，不是配置文件。

从固定 revision 的 config 与 `named_modules()` 打印实际的路由器、共享专家、路由专家和 adapter 目标模块名称。LoRA 参数即使只有约 55M，base weights、激活和 dispatch buffer 仍需占用 HBM。

保存一个批次的路由 logits、top-k 专家编号、每个专家接收的 token 数和 all-to-all 耗时。验收时确认 EP=8 的专家分片符合配置，所有 token 在路由前后数量守恒，并列出 adapter 覆盖了哪些共享专家、路由专家和路由器参数。

### 8.2 Hybrid Mamba/Attention：两种层混排，别拿纯 Transformer 的地图找路

attention 层回看全场，序列越长越贵；Mamba 类层维护固定大小的内部状态，顺序扫过序列。把两种层混排，长序列的显存和吞吐才压得住——第 12 课在玩具上装过同款混合内核，现在看真家伙的排布。

Hybrid 骨干在同一序列中组合 Mamba 类状态空间层与 attention 层。两类层都接收序列 hidden state，但内部参数命名和缓存状态不同：Mamba 类层执行状态更新，attention 层计算显式 token 交互。微调时不能把所有 `Linear` 都按普通 Transformer 的 q/k/v/o 理解——LoRA 的目标选择要是按错误的地图匹配名称，就会命中不该动的模块。

从固定模型版本导出完整 `named_modules()`，按层类型、参数形状、是否冻结和是否命中 LoRA 分组。

分别用一个短序列和一个长序列运行 forward，保存每类层的输入输出形状与峰值 HBM。目标清单必须能逐项对应到实际模块，并且不能因为名称匹配而误把 vision/audio tower 或 `lm_head` 纳入训练。

### 8.3 LoRA：只刷薄板，不重建承重墙

全参微调等于把整栋楼重新装修；LoRA 是在每面被选中的承重墙（命中的线性层）上贴一块薄板，只刷薄板。原墙 $W$ 一动不动，梯度和优化器状态只对薄板存在——这正是省钱的来源。类比失效处：薄板可以在交付时抹进墙里（merge 回 $W$），合并前后模型输出必须在容差内一致，这是 Step 9 的验收项。

对 $W\in\mathbb R^{d_{\text{out}}\times d_{\text{in}}}$，LoRA 冻结 $W$ 本身，另学两个低秩矩阵：

$$
\begin{aligned}
W' &= W+\Delta W, \\
\Delta W &= \frac{\alpha}{r}BA, \\
A &\in \mathbb R^{r\times d_{\text{in}}},
\qquad
B \in \mathbb R^{d_{\text{out}}\times r}, \\
\operatorname{rank}(\Delta W) &\le r.
\end{aligned}
$$

这里 $r$ 是 adapter rank，$\alpha/r$ 是缩放因子。矩阵 $A$ 的 rank 与矩阵 $B$ 的 rank 分别定义，因此不能写成 `rank(A,B)=r`。低秩意味着 $\Delta W$ 的表达能力受限：学 CORD 这类"换一种输出格式"的任务通常够用；要大改模型内在能力（官方 post-training 那种四阶段全参训练），低秩就不一定装得下。

**显存账。** LoRA 之后显存仍包括：

- base BF16 权重；
- LoRA 参数、梯度、optimizer state；
- 激活；
- media tower 输出；
- expert dispatch buffer；
- dataloader/processor 暂存。

第一项是大头：30B 参数的 BF16 权重本身就在 60 GB 量级，靠 EP 等手段分片后仍是每卡驻留的基本盘。这解释了为什么可训练参数从 31.5B 砍到 55M（约 570 分之一），每卡显存却只从约 49 GiB 降到约 30 GiB——LoRA 砍掉的是梯度和优化器状态，砍不掉底座驻留和激活。

对于一个被命中的线性层，LoRA 新增参数量为 `r(d_in+d_out)`，而基础矩阵 `W` 保持冻结。根据 named modules 逐层计算理论 adapter 参数量，再与框架报告的约 55M 可训练参数对照。随机抽取一个目标层，检查 `W.grad` 为空、`A.grad` 与 `B.grad` 非空；保存并重新加载后，合并前后的固定输入 logits 应在设定容差内一致。

### 8.4 Full SFT 显存账：31.5B 参数的完整账单

全参训练时每个参数背后跟着一串影子：梯度、Adam 的一阶与二阶矩、master weights——加起来是参数本体字节数的好几倍。31.5B 可训练参数的这串影子，单卡无论如何放不下，所以官方配置同时上 FSDP2（把参数、梯度、优化器状态切片摊到多卡的并行方式）和 EP=8。

full SFT 的显存账要同时包含参数、梯度、Adam 状态、master weights、activation 和通信 buffer。FSDP2/EP 可以分片部分状态，但媒体 tower 输出、局部激活、通信临时区和未分片对象不会统一除以 8——这是"总显存 192GB 不等于 8×80GB"的根本原因。

用 profiler 分别记录初始化后、forward、backward 和 optimizer step 的显存峰值，并按进程保存。将实测曲线与官方约 49 GiB/GPU 对照；出现差异时，从软件版本、序列长度、batch、精度、activation checkpointing 和通信后端中定位原因，不能只比较一个最终数字。

## 9. 数据字段：固定使用 CORD-v2

官方教程使用 CORD-v2。数据规模与任务形式为：

- 800 train；
- 100 validation；
- 100 test；
- 扫描收据图像；
- structured ground-truth JSON；
- 输出 XML-like 字段序列。

本课固定以下数据身份：

- dataset ID：`naver-clova-ix/cord-v2`；
- revision：`7f0115a4b758a71d6473b8d085751692da2fef98`；
- license：`CC-BY-4.0`；
- 下载后对 snapshot 内的每个文件生成排序后的 SHA-256 manifest；
- 保存 dataset card、split 行数、下载时间和本地 snapshot 路径。

浮动 `main` 不能代替上述 revision。不要复制带有 `/absolute/path` 占位符的命令；它很容易对错误目录生成一份看似正常的清单。数据准备程序必须接收已经解析并存在的 snapshot 目录，在该目录内逐文件计算 SHA-256，按相对路径排序，然后写出 `cord-v2.files.sha256`。程序还要在目录不存在、文件数为 0 或清单含绝对路径时直接失败。

`cord-v2.files.sha256` 必须随实验产物保存；复现实验前逐文件重新校验。CORD-v2 的 `CC-BY-4.0` 与模型权重的 NVIDIA Open Model Agreement 是两套不同许可，报告和再分发数据时必须分别记录并满足署名要求。

统一转换后的样本保存 conversation、asset、结构化 gold、输出 schema、来源与 split：

```json
{
  "id": "cord_train_0001",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "asset_id": "img0"},
        {"type": "text", "text": "Extract all receipt fields."}
      ]
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "<s_total>...</s_total>"}]
    }
  ],
  "assets": [{"asset_id": "img0", "uri": "dataset://cord-v2/train/0"}],
  "ground_truth_json": {"menu": [], "sub_total": {}, "total": {}},
  "output_schema": "cord_xml_v1",
  "source": "naver-clova-ix/cord-v2",
  "split": "train"
}
```

先核对 800/100/100 三个数据集的行数，再对所有文件执行 hash 校验。每个数据集随机抽取 20 条，确认图像可解码、标准答案可解析、对话与图像对应。验证程序还要检查训练集、验证集和测试集之间的文档族重复（同一张收据的不同裁剪、同一模板的近亲，散布在不同 split 就是泄漏）；任何泄漏都要在训练前处理。

### 9.1 Collator 输入输出约定

collator 把一条 conversation 转为模型 forward 所需的 token、媒体 tensor 和 labels。官方教程说明：

- 对 conversation 应用 chat template（把多轮对话拼成模型实际读到的一整段 token 的模板）；
- assistant turn 含 `<think></think>` 前缀语义；
- 处理图像；
- 建立 `image_flags`；
- 构造 training labels；
- 每个 `<image>` 在 forward 中由 256 个 vision embeddings 替换。

每个 `<image>` 占位符在 forward 中由 256 个 vision embeddings 替换，因此模板长度与替换后的序列长度不同。assistant token 承担监督 loss；system/user、padding 和不应学习的媒体位置需要按实现设为 `-100`——这正是第 1 课 loss mask 的同款约定，只是序列里多了媒体占位区。

必须从 10 条样本对应的 batch 中导出 raw messages、rendered template、`input_ids`、image tensor shape、`image_flags`、`labels`、承担 loss 的 token 区间，以及图像 embedding 替换后的序列长度。这些字段必须属于同一条样本 trace，不能从不同 batch 拼接。

逐条验证 `<image>` 数量、`image_flags`、图像 tensor 与 conversation 一致；检查 loss-bearing spans 只覆盖目标 assistant 内容。做两个故障注入：移除图像和把 labels 全设为 `-100`。前者应触发明确的数据错误或错误条件结果，后者必须被训练前检查拦截，不能静默产生零 loss。

## 10. 自定义任务选择：只迁移一个任务

自定义任务只选一个，以便把迁移失败定位到一套 dataset/collator/label contract：

- 图像：发票/表单字段抽取；
- 视频：固定 schema 的事件清单；
- 音频：ASR + 说话人/时间戳 JSON；
- 音视频：事件与声音 offset JSON。

首轮推荐图像字段抽取，因为它与官方 CORD-v2 的输入输出形式最接近，可以减少模型路径变更。完成图像任务后，再把视频、音频或音视频任务作为扩展。

自定义 schema：

```json
{
  "id": "custom_00001",
  "messages": [],
  "assets": [],
  "target": {
    "schema_id": "invoice_json_v2",
    "value": {"vendor": "...", "total": 37.5, "date": "2026-07-23"}
  },
  "source": "owned-or-licensed",
  "license": "explicit",
  "split_group": "document-family-id",
  "quality": {"human_verified": true}
}
```

同一模板或同一文档的裁剪必须按 `split_group` 划分，不能跨越 train、validation 和 test。实现字段校验、文档族去重和错媒体对照，并在 10 条训练样本上打印完整的 processor/collator 输出。validation 只选择 checkpoint；锁定后，基础模型与 LoRA 才在同一 test 上各运行一次。迁移是否成功按预注册的主指标、最小效应量和置信区间判断。

### 10.1 三档数据规模

| 档位 | CORD-v2 | 自定义任务 | 目的 |
|---|---:|---:|---|
| pilot | 10–100 train | 10–200 train / 50 held-out | overfit、collator、恢复训练 |
| standard | 官方 800/100/100 | 1k–20k / 至少 500 held-out | LoRA 主结论 |
| full | 不扩写官方 CORD | 20k–200k+ | 只有 standard 通过后 |

pilot 可以重复采样做 10-sample overfit，但 standard/full 必须按原始文档或媒体族去重。自定义 full 只表示本课监督微调数据的规模档，与 Nemotron 官方全量数据没有等价关系。

按"小规模预实验、标准实验、扩展实验"的顺序推进。先用 10 条样本证明 loss 能下降且输出能复现目标——第 1 课的 128 样本过拟合测试在这里缩水成 10 条，道理相同：小数据都背不下来，说明装配线有错。标准实验通过未见数据验收后，才允许扩展规模。每次扩容都保存去重报告、数据集统计和训练 token 数，避免把重复数据带来的收益误判为规模收益。

## 11. 与其他课程的边界：公开与缺失资产

官方模型卡披露了数据统计、公开数据名、私有数据类别和合成数据说明。披露信息仍不代表全部训练资产可下载并重放：

- 模型卡明确列出 public、private 和 self-sourced synthetic data；
- 部分第三方/私有数据不可由你获得；
- teacher、过滤器、质量/安全 judge 和 identity-fix 流水线不一定完整公开；
- 全部 717B-token mixture 与 124M curated post-training examples 不能仅凭列表精确重建；
- 20 RL datasets/25 environments 的完整状态不应假设全部可运行。

因此报告应写明：权重、模型卡、技术报告、推理代码与一个官方微调示例已经发布；训练数据披露较丰富，但全量数据与端到端训练资产没有完整开放。

许可遵循 NVIDIA Open Model Agreement。商业使用条款不能自动改写为 Apache-2.0、MIT 或 OSI 定义的完全开源。

建立开放程度表，把权重、代码、recipe、公开数据、私有数据、teacher、过滤器和 RL 环境分别标为已发布、部分披露或不可获得，并为每行附证据链接。请另一位读者只根据该表判断哪些实验可以重放；如果仍需猜测，说明状态或证据写得不够明确。

## 12. 实验矩阵：base vs LoRA

| 臂 | 训练 | 目的 |
|---|---|---|
| A：base | 不训练 | 零样本与格式基线 |
| B：official LoRA | rank 64、LM linear targets | 检验官方 recipe 的可复现性与增量 |

本课的主要对照固定为 A/B 两组。这是在实际训练条件下比较能力，不是严格控制计算量相同的消融实验。A 保留原始 base，B 从同一 revision 加载 rank 64 LoRA，并按官方 LM linear targets 训练。

比较时固定：

- 同一 BF16 base revision；
- 同一 CORD train/validation；
- 同一份封存的 CORD test；
- 同一 processor/collator；
- 固定训练集 5 条调试样例，以及 50/100 条 validation 检查点；
- 同一生成参数；
- 同一 checkpoint selection 规则；
- 报告不同 LR、trainable params 与实际 token，禁止隐藏。

为 A/B 两组生成同一份验证清单。先用训练集中的 5 个样例调通，再用 50 条 validation 做中期检查，最后用完整 100 条 validation 选择 checkpoint。checkpoint hash 锁定后，再为 A/B 生成一份独立的 test 清单，并各运行一次。学习率、可训练参数量和实际训练 token 数必须同时列出。自定义任务也只比较 A/B。验收时逐项确认两组的样例编号、媒体 hash、生成参数和解析器版本一致。

### 12.1 可选 full SFT 硬件参考

full SFT 不属于主要对照，不承担主假设验证、模型选择或课程通过条件。只有严格满足 **8×H100 80GB、EP=8** 的环境，才可按官方配置运行独立参考实验。跳过、显存不足或主动停止都不影响课程结论，但必须记录硬件检查结果。

这项参考实验只对照官方在特定环境下报告的约 49 GiB/GPU、约 31.5B 可训练参数和约 10 分钟运行时间，不能混入基础模型与 LoRA 的主表。把配置、指标和停止原因写入 `optional_full_sft_reference/`；主报告只引用该目录，不合并曲线或排名。

## 13. 目标 LoRA recipe

### 13.1 本课必须复现的 LoRA 配置

官方 LoRA 的 source lock、目标排除规则、rank、缩放因子和学习率如下：

```yaml
source_lock:
  automodel_commit: 81e7f01f431c31a60d607b9245f1337d8dcf9e1b
  model_id: nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
  model_revision: 24e67ea000b7c2837fc8f9488aa2008524fac8ba
  dataset_id: naver-clova-ix/cord-v2
  dataset_revision: 7f0115a4b758a71d6473b8d085751692da2fef98
peft:
  match_all_linear: false
  exclude_modules:
    - "*vision_tower*"
    - "*vision_model*"
    - "*audio*"
    - "*sound*"
    - "*lm_head*"
    - "*mlp1*"
  dim: 64
  alpha: 128
  use_triton: true
optimizer:
  lr: 1.0e-3
```

`match_all_linear: false` 表示目标模块由 recipe 中的明确规则选择；排除项用于避免 vision/audio tower、`lm_head` 和 `mlp1` 被 LoRA 命中。从固定 commit 解析最终目标列表，统计每类模块数量、参数形状与总可训练参数。列表中出现任一排除模块，或总量无法解释约 55M 参数时，先停止训练。

### 13.2 仅供 8×H100 80GB 参考的 full SFT 配置

```yaml
distributed:
  strategy: fsdp2
  ep_size: 8
freeze_config:
  freeze_embeddings: true
  freeze_vision_tower: true
  freeze_audio_tower: true
  freeze_language_model: false
dataset:
  path_or_dataset: naver-clova-ix/cord-v2
  split: train
dataloader:
  max_length: 4096
optimizer:
  lr: 1.0e-4
  weight_decay: 0.01
  betas: [0.9, 0.95]
```

最终配置必须从固定的 AutoModel commit `81e7f01f431c31a60d607b9245f1337d8dcf9e1b` 读取；本文片段用于阅读，不代替该 revision 的源码。

完整 SFT 配置将 language model 设为可训练，同时冻结 embeddings、vision tower 和 audio tower，并使用 FSDP2、EP=8、`max_length=4096`。只有硬件验收通过后才解析并运行该配置。验收时保存源码路径、commit、最终生效配置和 SHA-256；文中片段与源码冲突时，以固定 commit 为准并在报告中标明差异。

## 14. 实验步骤：从硬件检查到 adapter 交付

### Step 1：硬件与拓扑验收

先确认机器是否满足官方训练硬件要求。下面三个检查分别执行并分别保存输出。先列出 GPU：

```bash
nvidia-smi -L
```

再查看 GPU 拓扑：

```bash
nvidia-smi topo -m
```

最后记录型号、单卡显存和 driver：

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

再运行 NCCL all-reduce 与 all-to-all 小测，记录每个进程的带宽、延迟和错误。若 GPU 型号或每卡 HBM 不符合官方配置，只检查推理和 LoRA 是否可运行，不承诺完整 SFT。将结果写入 `hardware_check.md`，由脚本明确输出 `pass` 或 `not eligible`；不能用总显存估算代替单卡显存和互联拓扑的实测证据。

### Step 2：锁定环境

官方教程要求 NeMo AutoModel 容器或源码环境，并指出 Nemotron Omni 依赖 `mamba_ssm`、`causal_conv1d` 和媒体读取依赖。环境锁定发生在任何模型下载和训练之前。

记录：

- container digest；
- AutoModel commit：`81e7f01f431c31a60d607b9245f1337d8dcf9e1b`；
- Python/CUDA/PyTorch；
- `mamba_ssm`、`causal_conv1d`；
- config SHA；
- model revision。

必须 detached checkout（把工作树固定到某个 commit，不跟随分支移动）到该 commit，并把 `git status --short`、`git rev-parse HEAD` 与依赖 lock 一起保存。本地补丁必须单列；存在补丁时，实验名称要标记为基于官方 recipe 的修改版。

在干净容器中重新执行安装，并运行导入测试、模型配置加载和一次无梯度 forward。两次环境的容器摘要、commit、依赖版本和配置 SHA 应一致；否则不能进入能力比较。

### Step 3：跑 base inference

base inference 用于验证 processor、四模态入口和结构化输出基线。准备 10–20 个固定样例，覆盖：

- text；
- image；
- video；
- audio；
- JSON/结构化输出。

每条样例保存 messages、媒体 hash、processor 输出 shape、生成参数和原始 text 输出。验证四类输入都能被正确读取，JSON/结构化请求的 parse 结果可记录。模型支持音频输入并不表示支持音频输出；本步骤的输出文件类型应全部为 text。

### Step 4：复现 CORD processor/collator

以 `revision="7f0115a4b758a71d6473b8d085751692da2fef98"` 下载 `naver-clova-ix/cord-v2`，核验 license 为 `CC-BY-4.0` 和 800/100/100 split。缓存完成后生成并保存逐文件 `cord-v2.files.sha256`；后续 run 只使用通过该 manifest 校验的 snapshot，不得重新解析浮动 `main`。

逐 token 检查 10 条样本，确认图像、模板、labels 和 loss mask，并对长字段与空字段单独测试。验证表要列出 `<image>` 替换前后的序列长度、256 个 vision embeddings、`image_flags` 和参与 loss 的 token 区间。hash、数据划分、许可或 label mask 任一不符合预期，都要在 10 步训练前修复。

### Step 5：100-step LoRA smoke

100-step smoke 主要检查状态能否完整保存与恢复。按以下时间线执行：

- train 10 steps；
- save；
- 新进程 load；
- 继续到 100 steps；
- inference；
- adapter 文件清单；
- 训练前后固定样例。

第 10 步保存后必须结束原进程，再从全新进程加载 model、adapter、optimizer、scheduler 与 dataloader 状态，继续到第 100 步。比较连续运行与恢复运行的 optimizer step、LR、loss 区间和固定样例输出。状态不连续或 adapter 文件缺失时，不进入 400 步训练。

### Step 6：完整官方 400-step LoRA

完整 LoRA 从通过 smoke 的同一 source lock 开始，运行到 400 steps。每 10 steps 保存：

- loss、grad norm、LR；
- HBM；
- tokens/s；
- expert dispatch；
- validation；
- checkpoint 时间。

在预先登记的保存点运行 validation，并按固定规则选择 checkpoint。完整的 100 条 validation 都用于选择，不能用官方 5 个示例替代。选择完成后冻结 checkpoint hash、processor、生成参数与解析器，再解封 100 条 test；base 与选中的 LoRA 各运行一次，最终报告只使用这次 test 计算 base→LoRA 增量和 95% 置信区间。

第二次独立训练使用相同锁定项和不同运行编号，也只能用 validation 选择 checkpoint。训练曲线、adapter 保存与恢复、固定推理结果需要在报告定义的容差内对齐；不能因为第二次训练看起来更好，就在 test 解封后替换已经锁定的主实验 checkpoint。

### Step 7（可选）：8×H100 full SFT reference

本步骤不进入课程完成条件。仅在 **8×H100 80GB、EP=8** 且下列检查全部通过后执行：

- 每卡至少满足实测峰值 + 15% 余量；
- EP=8 all-to-all 稳定；
- 100-step LoRA 已可恢复；
- checkpoint 空间至少 2×预期；
- 没有并行用户任务争抢 HBM。

先运行 10–50 步，比较每个进程的 HBM、吞吐、专家路由和官方观测，再决定是否继续到 400 步。跳过、显存不足或只完成硬件检查都不影响课程验收；在报告中单独标为 `optional full-SFT reference`，不能并入主要对照表。保存停止前最后一个正常 checkpoint，并确认它能在新进程中加载。

### Step 8：迁移自定义数据

迁移实验先只替换 dataset/collator，不改模型、LoRA rank 或 target modules：

1. 10 samples overfit；
2. 100–1000 pilot；
3. 用固定 validation 选择 checkpoint；
4. base vs LoRA；
5. JSON/schema verifier；
6. wrong-media 对照。

10 条样本过拟合用于验证 loss、目标重现和 checkpoint 恢复；100–1000 条预实验用于发现字段、长度和数据划分问题；固定 validation 只用于选择 checkpoint。选择完成后锁定全部评测条件，在自定义 test 上对基础模型和 LoRA 各运行一次。使用 JSON/字段校验器与错媒体对照，检查模型是否读取媒体。迁移任务在未见数据上没有改善时，先检查 processor、collator 和 label 约定，不要同时调整 rank、学习率和数据。

### Step 9：合并/加载 adapter

adapter 有三种交付状态：训练框架内加载、adapter-only checkpoint、合并后的 HF base。分别验证：

- AutoModel 内加载；
- adapter-only checkpoint；
- merge 到 HF base；
- 新进程推理；
- 合并前后 logits/output；
- 部署引擎兼容性。

`adapter-only checkpoint` 只保存 LoRA 权重和必要的 adapter 配置，用于加载或分发，不是完整的训练恢复点。要从中断处继续训练，还必须保存 base revision、optimizer、scheduler、scaler、global step、dataloader/sampler cursor 和 RNG state。缺少任一项时，只能重新开始一个训练 run，不能把 adapter-only 加载写成"断点恢复成功"。

在全新进程中，对同一固定 batch 比较未合并与合并模型的 logits，再比较确定性生成输出。差异超过预设容差时，检查参数键映射、LoRA scale 和 dtype。只有文件清单、加载路径、logits 与部署启动测试都通过，adapter 才算可交付。

### Step 10：与 MiniMind 教学模型对照

为了把基座能力与微调方法区分开，在同一简化任务、同一 held-out 数据和同一生成协议下比较：

- base capacity；
- LoRA 增量；
- trainable params；
- GPU-hours；
- 峰值 HBM；
- 延迟。

分别报告基础模型能力，以及 LoRA 相对各自基础模型的增量。Nemotron 的绝对分数更高只能说明基础模型不同；讨论 recipe 效率时，要同时比较归一化后的 adapter 增量、成本和回归结果。玩具和真家伙同台，比的是方法，别让底座抢了戏。

## 15. AutoModel 配置入口：固定 commit

LoRA 是本课主实验。先确认当前工作树已 detached checkout 到 `81e7f01f431c31a60d607b9245f1337d8dcf9e1b`，再使用该 commit 提供的 `automodel` 配置入口：

```bash
uv run automodel examples/vlm_finetune/nemotron_omni/nemotron_omni_cord_v2_peft.yaml --nproc-per-node 8
```

full SFT 只用于 8×H100 80GB 的可选 reference：

```bash
uv run automodel examples/vlm_finetune/nemotron_omni/nemotron_omni_cord_v2.yaml --nproc-per-node 8
```

旧教程中的脚本入口只是兼容 shim，不再作为本课入口。执行时必须使用本课固定的 AutoModel commit，不能解析 nightly/beta 浮动路径。把实际命令、工作目录、环境变量、最终生效配置和退出码写入启动记录。8 个进程均完成初始化且配置 hash 一致后，启动检查才算通过。

## 16. 训练预算与 8 卡停止条件

| 硬件档位 | 允许动作 |
|---|---|
| 非 H100、<30GB | inference/量化推理；不照搬官方训练 |
| 8×40/48GB | LoRA feasibility，缩短长度/batch 需另记 config |
| 8×80GB 非 H100 | LoRA；不执行本课的 full SFT reference |
| 8×H100 80GB | 官方 LoRA；可选 full SFT reference |

硬件表先限定允许动作，训练监控再执行以下立即停止条件：

- 某卡 HBM > 95% 持续；
- EP rank 死锁/NCCL error；
- loss NaN 连续；
- 数据 label 全 `-100` 或 media flags 异常；
- expert load 极端失衡；
- checkpoint 无法恢复；
- 修改配置后仍标记为官方原配置复现。

任何降低 batch、长度、精度、冻结范围或 target modules 的改动，都创建新 experiment ID。

把停止条件实现为监控告警，并保存触发前后的 HBM、loss、NCCL 与专家负载。停止后先归类原因并检查 checkpoint 完整性；修改配置时新建实验编号，旧日志保持只读，验证修复不能覆盖原始失败证据。

## 17. 指标：能力、系统与回归

### 能力

能力指标要区分结构合法性、字段正确性和媒体依赖。最终主指标是 test 上的字段 F1 增量：

$$
\Delta_{\text{CORD}}
=
F1_{\text{test}}(\text{LoRA})
-
F1_{\text{test}}(\text{base}).
$$

validation 分数只留下 checkpoint 选择记录，不能放进这个公式。test 上还要计算：

$$
G_{\text{media}}
=
F1_{\text{correct image}}(\text{LoRA})
-
F1_{\text{wrong image}}(\text{LoRA}),
$$

其中 wrong-image 使用测试前固定的错图映射，并与正常图像逐样本配对。这个指标回答"模型是真在看图，还是背下了收据的套路"。自定义任务只预注册一个主要 test 指标，记其 LoRA−base 差值为 $\Delta_{\text{custom}}$，不能看完多个指标后再挑最大的一个。

在解封 test 前，分别写下最小有意义效应 $\delta_{\text{CORD}}$、$\delta_{\text{media}}$ 和 $\delta_{\text{custom}}$。这些数值根据字段标注误差、任务用途和采用 LoRA 的成本确定，不在教程中凭空指定。三个差值都报告成对 bootstrap 的 95% 置信区间。

同时保存以下明细指标：

- XML/JSON parse rate；
- exact match；
- field-level precision/recall/F1；
- numeric/date normalized accuracy；
- test 上的 base→LoRA 差值；
- wrong-image condition gap；
- 自定义任务 test。

若实际执行 full SFT，其能力与系统指标必须放入独立 `optional_full_sft_reference` 表，不参与 primary 结论。

先用手工构造的样例验证解析器、数值与日期归一化、字段 F1，再对基础模型与 LoRA 使用同一份冻结的 test 清单。错图条件的差值必须与正常图像结果配对。若实际执行完整 SFT，其能力与系统指标只进入独立的 `optional_full_sft_reference` 表。

### 系统

系统指标按 rank 与 step 时间线采集：

- peak HBM per rank；
- tokens/s/GPU 与 global tokens/s；
- step P50/P95；
- all-to-all/collective 占比；
- GPU utilization；
- checkpoint size/save/load；
- adapter merge 时间；
- 训练 GPU-hours。

保存原始性能记录，并报告 P50/P95 的采样区间、预热丢弃规则和全局 batch。用 checkpoint 文件大小与实际保存、加载时间复核日志；逐进程 HBM 取最大值，不能只报告平均显存。

### 回归

回归集用于检查 CORD 或自定义任务训练是否破坏已有能力——微调最常见的暗亏是新任务学会了、旧本事丢了：

- 固定 text/image/video/audio probes；
- general instruction following；
- JSON/tool calling；
- 原 CORD 与自定义任务交叉回归；
- long context smoke；
- reasoning/non-reasoning template。

每个回归指标 `m` 都在 test 解封前登记一个最大可接受下降 $\delta_{\text{reg},m}$。基础模型、LoRA 和合并 adapter 使用相同输入与生成参数，逐项报告 $\Delta_{\text{reg},m}=\operatorname{score}_m(\text{LoRA})-\operatorname{score}_m(\text{base})$ 及其 95% 置信区间。是否通过按预注册的非劣条件判断，不能用"看起来变化不大"代替数值。

## 18. 验收条件

### 18.1 先检查 recipe 是否完整跑通

以下是不能用统计波动解释的硬条件：

- 8-rank 训练、保存、全新进程加载、继续训练全部成功；
- 100 条 validation 只用于选择 checkpoint，选择结果及文件 hash 已冻结；
- 100 条 test 在冻结后只运行一次，base 与 LoRA 的样本编号完全一致；
- test structured parse rate 的点估计不低于 95%；
- 文本、图像、视频和音频四种固定输入均能完成加载与生成，没有崩溃、NaN 或空输出文件；
- adapter 保存、恢复和合并后的固定输入 logits 在预注册的数值容差内一致；
- HBM/吞吐和官方数值差异有解释；
- adapter target module 清单可审计；

自动验收程序要逐条输出状态和证据路径。上述条件全部满足，才能说公开 recipe 已经从训练走到可加载产物；训练 loss 下降不能替代这些证据。

### 18.2 再判断能力结论

只有满足下面的预注册规则，才能说 LoRA 在对应任务上带来了有效增量：

- $\Delta_{\text{CORD}}$ 的点估计不小于 $\delta_{\text{CORD}}$，且 95% 置信区间下界大于 0；
- $G_{\text{media}}$ 的点估计不小于 $\delta_{\text{media}}$，且 95% 置信区间下界大于 0；
- 若声称自定义任务迁移成功，$\Delta_{\text{custom}}$ 的点估计不小于 $\delta_{\text{custom}}$，且 95% 置信区间下界大于 0；
- 对每个必须保留的回归指标，$\Delta_{\text{reg},m}$ 的 95% 置信区间下界不低于 $-\delta_{\text{reg},m}$。

如果某个效应没有达到最小效应量，或者其区间包含 0，报告应直接写"这次 test 没有证明该增量"。这不等于 recipe 没有跑通，但不能发布能力提升结论，也不能回到 validation 重新挑 checkpoint。报告必须保留点估计、置信区间和逐样本结果。

### Full SFT 可选 reference 报告规则

- 仅 8×H100 80GB 官方硬件组合可把结果称为官方硬件条件复现；
- 其他硬件不执行本课的 full SFT reference；
- 峰值 HBM 需保留至少 10% 安全余量；
- checkpoint 可恢复；
- 若跑 400 steps，报告完整 validation，不照抄官方 5-case 表。

未尝试、未完成或硬件验收判定不适合完整 SFT，均不影响本课通过；只需保存硬件检查证据。完整 SFT 结果不能替代基础模型与 LoRA 的必做评测。该参考实验必须拥有独立目录、配置和指标表；任何记录混入主要对照表，都要在发布前移除。

### 报告措辞检查

报告必须明确保留以下范围声明：

> 本实验复现的是公开的 NeMo AutoModel 微调示例，不是 Nemotron 3 Nano Omni 的完整预训练与 post-training。

范围声明还要链接 source lock 与 openness matrix。审阅者应能据此区分公开微调示例、可选 full SFT reference 和未复现的完整训练阶段。

## 19. 失败诊断：定位失败层

诊断顺序从输入 contract 开始，再检查状态保存、并行通信与训练数值。不要同时修改多个变量——8 卡大模型的失败现场比玩具贵得多，每次只动一个旋钮：

| 症状 | 可能原因 | 检查 | 修复 |
|---|---|---|---|
| 8 卡仍 OOM | 单卡权重/激活峰值 | rank HBM timeline | LoRA、短长度 |
| EP 卡住 | 拓扑/NCCL/dispatcher | all-to-all test | 修通信环境 |
| loss 很低但输出不可解析 | mask/teacher forcing | label dump | 修 template/schema |
| LoRA 无变化 | target 未命中 | trainable names | 对照官方 exclusions |
| 输出字段缺失 | truncation/schema | target token 长度 | 调 max length |
| validation 近 100%、test 差 | 文档模板泄漏 | group split | 文档族去重 |
| 错图也答对 | prompt/数据捷径 | wrong-image | hard negatives |
| 可选 full reference 比 LoRA 差 | LR/小数据过拟合 | train/val curve | 降 LR/早停 |
| 恢复后 loss 跳变 | optimizer/EP state | state dict | 修 checkpoint |
| merge 后输出不同 | key mapping/scale | logits compare | 检查 A/B 与 alpha |
| HBM 比官方高 | 版本/backend 差 | config diff | pin 官方环境 |
| HBM 低但巨慢 | PCIe/CPU offload | profiler | 改拓扑/数据 |
| 音频输入不可用 | media deps/processor | base smoke | 安装官方 extras |
| 误称 speech output | 能力边界混淆 | model card | 接 Talker 才可说话 |

每次只选择表中的一个症状，保存原始证据、最小检查、单项修复和修复后的启动测试结果。只有相同输入下症状消失且其他验收项没有退化，修复才可进入主实验。

## 20. 逐个样例检查：LoRA 改变了什么

逐例审计用于判断 LoRA 改变了字段事实、输出格式，还是两者都改变。CORD 至少审计：

- 20 个 exact match；
- 20 个 partial match；
- 全部 parse failure；
- 20 个 base→LoRA 改变；
- 20 个 wrong-image；
- 20 个自定义任务。

每条记录使用统一 schema：

```yaml
case_id: cord_validation_0037
asset_hash: sha256:...
gold_fields:
  total: 37.50
base_output: "The receipt total appears to be..."
lora_output: "<s_total><s_total_price>37.50</s_total_price></s_total>"
parsed:
  base: null
  lora: {total_price: 37.50}
field_f1: 1.0
wrong_image_output: "<s_total>...</s_total>"
condition_dependent: true
failure_layer: null
checkpoint: step_399
```

审计者逐项记录：

1. gold 与图像是否一致；
2. processor 是否保留了关键区域；
3. labels 是否覆盖正确 target；
4. 输出错在 perception、reasoning 还是 serialization；
5. LoRA 是否改变了事实或只改变格式；
6. wrong-image 是否合理改变答案；
7. case 是否与 train 同文档族；
8. merge 前后是否一致。

从固定 checkpoint 重新生成选中的样例，并重跑解析器、字段 F1 与错图条件。两名检查者对失败环节判断不一致时保留各自的原始标注。最终统计感知、推理、序列化、数据泄漏和合并五类失败的数量，确保聚合指标能回到具体样本。

## 21. 交付物目录

交付目录把环境、模型、代码、数据、checkpoint、指标和逐例证据分开，便于在新机器逐层恢复：

```text
artifacts/exp18/
├── environment/
│   ├── hardware.txt
│   ├── topology.txt
│   ├── versions.lock
│   └── container_digest.txt
├── model/{revision.txt,files.sha256,license.txt}
├── code/{automodel_commit.txt,working_tree.patch,config.sha256}
├── configs/{base,lora,custom_lora}.yaml
├── optional_full_sft_reference/   # 仅在实际尝试时创建
│   ├── hardware_check.md
│   ├── config.yaml
│   └── metrics.jsonl
├── data/
│   ├── cord-v2.revision.txt
│   ├── cord-v2.license.txt
│   ├── cord-v2.files.sha256
│   ├── cord_split_counts.json
│   ├── cord_manifest.jsonl
│   ├── custom_manifest.jsonl
│   └── asset_registry.jsonl
├── checkpoints/index.json
├── metrics/{cord,custom,regression,system}.jsonl
├── traces/{hbm,nccl,profile}/
├── cases/audit.md
├── openness_matrix.md
└── report.md
```

`openness_matrix.md`：

| 项目 | 状态 | 证据 | 能否完整重放 |
|---|---|---|---|
| 权重 | released | model card | 是，受许可约束 |
| 推理代码 | released | official repo | pin 版本后可 |
| 微调示例 | released | AutoModel guide | 指定硬件可 |
| 全量训练数据 | partially disclosed | model card | 否 |
| 完整过滤/teacher | partially disclosed | paper/card | 否 |
| 全部 RL 环境 | partially disclosed | recipe/card | 否 |

在干净目录运行产物校验器：逐文件检查 hash，验证 JSON/JSONL 字段，确认 checkpoint 索引指向存在的文件，并从 `cord_manifest.jsonl` 随机重载媒体。`optional_full_sft_reference/` 只在实际尝试时创建；空目录或复制主实验指标都会造成误读。

## 22. 复现清单：最终检查

清单按执行顺序核对，每项都要附文件或日志路径：

- [ ] 官方模型 revision 和文件 hash 已固定；
- [ ] NVIDIA Open Model Agreement 已保存/阅读；
- [ ] 明确输出只有 text；
- [ ] GPU/HBM/topology 已记录；
- [ ] AutoModel 已锁定到 `81e7f01f431c31a60d607b9245f1337d8dcf9e1b`；
- [ ] container digest 与 config SHA 已锁定；
- [ ] CORD-v2 已锁定到 `7f0115a4b758a71d6473b8d085751692da2fef98`；
- [ ] CORD-v2 `CC-BY-4.0` 已保存并满足署名要求；
- [ ] CORD-v2 逐文件 SHA-256 manifest 已生成并复核；
- [ ] CORD 800/100/100 已核验；
- [ ] processor/collator/labels 已逐 token 审计；
- [ ] base 固定样例已保存；
- [ ] LoRA 10-step save/load/resume 通过；
- [ ] LoRA 400-step 或清楚记录停止原因；
- [ ] 三个最小效应量、回归容许范围和置信区间算法在 test 前预注册；
- [ ] 100 条 validation 只用于 checkpoint 选择；
- [ ] checkpoint 与评测配置锁定后，100 条 test 只运行一次；
- [ ] test 上的 base→LoRA 差值与 95% 置信区间已报告；
- [ ] wrong-image 的配对差值与 95% 置信区间已报告；
- [ ] 自定义任务 pilot、validation 与一次性 test 已完成；
- [ ] adapter merge/load 输出一致；
- [ ] 四模态回归完成；
- [ ] HBM/吞吐/GPU-hours 已报告；
- [ ] openness matrix 完成；
- [ ] 未声称完整官方训练复现。

以下仅在选择执行可选 reference 时检查，不属于完成清单：

- [ ] full SFT 确认是 8×H100 80GB、EP=8；
- [ ] full SFT 硬件检查、配置、停止原因和结果已单独归档；

由未参与训练的人在新进程完成三项抽查：加载固定模型与 adapter、重放 10 条 CORD 推理、恢复一个训练 checkpoint。抽查通过后再签署清单；口头确认不能替代日志。

## 23. 前沿对照与改造方向

本课回答的问题——怎么把一个大 omni 模型驯到自己的任务上——前沿系统给出了两层答案。第一层是官方自己怎么训：按 [Nemotron 3 Nano Omni 技术报告](https://arxiv.org/abs/2604.24954)与[官方 training recipe](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/omni3/README.md)，最终模型经过 `SFT → MPO → Text GRPO → Vision GRPO` 四阶段全参 post-training，上下文按 `16K→49K→256K` 分阶段拉长，背后是 717B-token mixture、124M 条精选 post-training 样本和 20 个 RL 数据集/25 个环境——这条路线和你在第 15-17 课的玩具上走过的 SFT、偏好优化、GRPO 是同一套方法学，只是每一步都贵好几个数量级。第二层是官方建议你怎么微调：对外发布的下游适配路径是 NeMo AutoModel 的 LoRA 示例，冻结 vision/audio tower、只在语言骨干的线性层上挂低秩 adapter。[Qwen2.5-Omni 技术报告](https://arxiv.org/abs/2503.20215)（第 1 课引用过）同样只公开底座能力与按模态分桶的评测，下游任务怎么微调，其技术报告未展开——把通用能力训进底座、把任务适配留给参数高效微调，是 2024-2026 年开源大模型的通行分工，NVIDIA 只是把这条路直接写成了官方教程。

差距要拆成三类。规模差距（钱能解决）：8×H100、私有与合成数据、RL 环境、四阶段全参 post-training——openness matrix 里已经写明哪些拿不到、哪些跑不起，砸钱也只能追到"公开资产"这条线。机制差距（本课教的东西能解决）：label contract 是否正确、LoRA target 是否命中、评测是否预注册、adapter 能否保存/恢复/合并——这些不看钱，看纪律；做完本课，你在微调工程这件事上和官方教程作者用的是同一套动作。能力边界差距：Nemotron 输出只有文本，离"会说话的 GPT-4o 类系统"还差一张嘴，这张嘴恰好是我们在玩具上训好的 Talker——第 19 课就干这个。

每个改造都按第 4 节的纪律新建 experiment ID 并重新预注册；同一份 100 条 test 每复用一次都要在报告里注明，复用多了它就不再是"未见数据"。

1. **rank 16 对照 rank 64。** 改 13.1 配置里的 `peft.dim: 64` 为 16（`alpha` 是同比例改 32 还是保持 128，作为配置差异另记），其余 source lock 不动。预算：官方观测 400 步 LoRA 约 6 分钟，一次训练 8 卡合计约 1 GPU-hour 内，两臂加评测 4 GPU-hour 内。预期：可训练参数按 `r(d_in+d_out)` 线性缩到约四分之一；若 rank 16 的 $\Delta_{\text{CORD}}$ 仍达 $\delta_{\text{CORD}}$ 且与 rank 64 的置信区间重叠，说明 CORD 这类格式任务用不满 rank 64 的容量。失败判定：rank 16 的 parse rate 点估计跌破 95%，记录为该任务的秩下限证据；两臂都不过 $\delta_{\text{CORD}}$ 则先回查 collator，与 rank 无关。
2. **把 LoRA target 扩到 connector。** 把 `*mlp1*` 从 13.1 的 `exclude_modules` 移出，先按 8.2 节用 `named_modules()` 导出并审计新命中清单，确认 `*vision_tower*`/`*vision_model*` 仍被排除，再训练。预算：一次 400 步 LoRA 加 test 评测，4 GPU-hour 内。预期：connector 参与训练后 $G_{\text{media}}$（错图配对差值）应不降；可训练参数的增加量要能用逐层公式解释。失败判定：审计清单命中任何 vision tower 本体模块（立即停止，实验作废），或回归集图像 probe 的下降超出 $\delta_{\text{reg},m}$。
3. **显存下限探测。** 固定 LoRA 配置，把 dataloader `max_length` 从 4096 依次降到 2048、1024，每档只跑 10–50 步，按 17 节系统指标记录逐 rank 峰值 HBM 与 tokens/s。预算：三档共 2 GPU-hour 内；若在 8×40/48GB 档位机器执行，按 16 节的表另记 config，不得标为官方复现。预期：激活相关显存随长度下降，但曲线会露出一块不随长度动的"地板"——那就是 8.3 节算过的底座权重驻留。失败判定：1024 长度下仍触发 HBM>95% 停止条件，结论写"该硬件档位不适合本模型 LoRA"，不要继续硬调。
4. **数据效率曲线。** 从 800 条训练集按文档族抽 100/200/400 条（复用第 9 节的去重结果），各训一次 LoRA，在同一封存 test 上各测一次。预算：三次训练加评测 8 GPU-hour 内。预期：$\Delta_{\text{CORD}}$ 随数据量单调上升；若 100 条已接近 800 条的水平，结合第 20 节的逐例审计判断 LoRA 学到的主要是输出格式还是字段事实。失败判定：曲线无序波动且各点置信区间互相包含——先怀疑 100 条 test 的区间本来就宽，扩大评测集后再下结论，不急着谈数据效率。

三条"官方结论、本课缩小版对应实验、预期方向"的映射：

- 官方教程结论"55M 的 LoRA 就能让 30B 模型学会 CORD 抽取，且只要约 6 分钟"，对应本课主实验 B：预期能复现同方向，$\Delta_{\text{CORD}}$ 置信区间下界大于 0；训练时间与显存和官方数字的偏差按 18.1 节要求给出解释。
- 官方示例设定"冻结 vision/audio tower、只调语言骨干线性层，也能学会看图抽取"，对应 B 组加 $G_{\text{media}}$：预期 $G_{\text{media}}$ 点估计大于 0，证明增量确实依赖图像；若错图也答对，失败原因大概率是 19 节表里的数据捷径，与冻结策略无关。
- 技术报告的架构主张"hybrid Mamba-Attention 控制长序列成本"，对应 8.2 节的短/长序列 forward 对比：预期两类层的峰值 HBM 随长度增长的斜率不同，方向可测；但这只是推理侧单次 forward 的缩小版观察，复现不了报告里训练侧的完整吞吐结论，报告里要写明这个边界。

## 24. 必读论文与官方材料

每份材料都带着能在自己产物里指认答案的问题去读；答不上来就回去重读。

### [Nemotron 3 Nano Omni 技术报告](https://arxiv.org/abs/2604.24954)

先读架构图、模型配置和训练阶段，不要从摘要开始抄结论。带着问题：四种模态各自沿什么路径进入语言骨干？总参数和激活参数分别按什么口径统计？读完后提交两张图和一张表：

1. 画出图像、视频、音频进入语言骨干的实际路径，标出 C-RADIO、音频编码器、Conv3D/EVS 和 projector；
2. 画出一层 hybrid Mamba–Attention MoE，并分别写出总参数和每个 token 激活参数的口径；
3. 用表格列出 `16K→49K→256K`、SFT、MPO、Text GRPO 和 Vision GRPO 各自改变的数据、目标和参数。

每一项都附论文页码。论文没有给出的 batch、数据混合比例、过滤器或内部环境直接标为"未公开"，不能自行补齐。

### [官方 Hugging Face 模型卡](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16)

模型卡用于确定"能用什么"和"能声称什么"。带着问题：哪些声明你能在本地验证，哪些只能引用？把下列内容逐项抄入 `openness_matrix.md`，并保留原字段链接：

- 权重许可与商业使用条件；
- 支持的文本、图像、视频和音频输入；
- 输出类型为何只有文本；
- 公开、私有和合成数据的披露状态；
- BF16、FP8、NVFP4 各变体的用途与限制。

随后分别运行一个文本、图像、视频和音频样例。模型卡字段与本地输出不一致时，记录使用的 revision 和差异，不要继续引用浮动的 `main`。

### [NeMo AutoModel 官方微调教程](https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/nemotron-omni)

这份教程要跟着执行，边跑边对照本课 Step 1-6。按"环境、CORD-v2、配置、collator、启动、推理"的顺序，为每一步保存命令、输入文件、输出文件和退出码。读完后应能直接回答：

1. 为什么官方配置要求 8×H100 80GB 和 EP=8；
2. 完整 SFT 与 LoRA 分别训练哪些参数、占用多少 HBM、保存哪些文件；
3. 哪些视觉和音频模块被冻结；
4. LoRA 的排除规则最终命中了哪些 Mamba/MoE 线性层；
5. 为什么教程中的 5 个样例只能做启动检查，不能代替 100 条验证集。

### [Nemotron Omni 官方 training recipe](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/omni3/README.md)

沿着 SFT、MPO、Text GRPO、Vision GRPO 四个阶段逐项检查仓库，带着问题：每个阶段你能拿到什么、缺什么？每个阶段记录三列："公开命令与配置""公开 checkpoint""缺少的数据或环境"。只有三列中的实际链接和路径都齐全，才标记为可重放。文档描述了方法但缺少数据、过滤器或奖励环境时，结论只能写"公开了 recipe，未公开完整复现资产"。

### [NeMo AutoModel 官方仓库](https://github.com/NVIDIA-NeMo/Automodel)

从本课锁定的 commit 出发，沿一次训练 batch 的调用顺序定位五处代码：processor、multimodal collator、模型 forward、LoRA 目标选择、checkpoint 保存与加载。带着问题：一条 CORD 样本变成一个可训练 batch，中间经过哪几双手？最终提交：

- 一条样本从 CORD 字段到 label mask 的调用路径；
- HF 参数名到 AutoModel 参数名的映射样例；
- EP=8 device mesh 的创建位置；
- 一个 LoRA 目标层为何被选中、一个排除层为何没有被选中；
- 合并 adapter 时 key、scale 和 dtype 的一致性测试。

每个结论至少引用一个论文页码、模型卡字段或固定 commit 下的源码路径，并指向本课生成的配置、日志或测试。只复述摘要不算完成阅读。

## 25. 扩展题：后续研究

扩展实验从已经完成验收的主实验分叉，每次只改变一个变量：

1. 比较 LoRA rank 16/64，但保持总 token 和 eval 不变；
2. 将 LoRA target 从 language linear 扩展到 connector，先审计参数名；
3. 对 video 或 audio JSON extraction 做迁移；
4. 比较 BF16、FP8、NVFP4 的推理质量/显存，不与训练主结论混合；
5. 使用实验 15 的数据 contract 接入小型 joint SFT；
6. 用实验 16/17 的公开小数据做 adapter-level MPO/GRPO，但不得称复现官方 final model；
7. 将 Nemotron text 输出连接 MiniMind-O Talker，形成可说话 cascade；
8. 再训练 hidden-state adapter，比较 text bridge 与 state bridge。

每项扩展都使用新的 experiment ID，保存变量、预算、held-out 指标和回归。完成条件是给出可反驳结论与对应证据；单张 loss 曲线不足以支持结论。

## 26. 最终版本与复现声明

通过本课后产出两个明确版本：`nemotron-cord-lora-reproduced` 对应固定 CORD-v2 recipe 的复现，`nemotron-custom-lora-v1` 对应自定义任务迁移。

若 full SFT 成功，另标为 `nemotron-cord-full-sft-reference`。

完成 `nemotron-cord-lora-reproduced` 与 `nemotron-custom-lora-v1` 两个必做产物后，本课即可通过；未尝试或未跑通 full SFT 不影响结课。

只有同时固定模型 revision、AutoModel commit `81e7f01f431c31a60d607b9245f1337d8dcf9e1b`、CORD-v2 revision `7f0115a4b758a71d6473b8d085751692da2fef98`、原始 LoRA config，并匹配 8×H100 80GB、EP=8 环境，才可标记为官方 LoRA recipe 复现。任何配置、代码、数据或硬件改动都标记为基于官方 recipe 的修改版。可选 full SFT 只能单独标记为官方硬件条件下的 full SFT reference，不能扩大为完整训练复现。

最终报告需要准确说明：

- 这套权重是否可用、受什么许可；
- 官方 LoRA recipe 能否从干净环境重复，并能否迁移到自定义任务；
- 你的 8 卡能否稳定运行 LoRA；若恰为 8×H100 80GB，可选 full SFT 参考实验的硬件检查结果是什么；
- LoRA 学到了事实、格式还是两者；
- 哪些能力来自 base，哪些来自 adapter；
- 哪些训练资产仍不开放；
- 它适合深入学习的原因，以及它与完全可复现 GPT-4o clone 的差距。

发布前，在新环境加载两个必做版本，运行固定 CORD 与自定义样例，并核对源码锁定文件、开放程度表和报告措辞。三个部分一致后，本课交付才完成。

回头看整个系统：你手里现在有两样东西——一台被改造过 17 轮、会听会说会看的 26M 玩具，和一颗刚被你驯服、能看图看视频听音频但只会打字的 30B 脑子。它们还没见过面。下一课（[第 19 课](19_capstone_thinker_talker.md)）就是毕业设计本体：把 Nemotron 当 Thinker、把 MiniMind 的 Talker 当嘴，用 bridge 把两者的时间轴和 hidden size 对齐，做出能监听、能发声、能被打断的双工系统。
