---
id: 11_tiny_moe
title: "现代 Tiny MoE"
summary: "每个 token 实际动用的 FFN 计算量差不多时，routed 加 shared experts（按需派活的专家＋人人都过的公共专家）带来的提升，是 dense 对照解释不了的真容量收益吗？"
unit: backbone
play_tools: []
checkpoints:
  - "分清三笔账：total parameters、active parameters、真实 active FLOPs。"
  - "搭起三组公平对照：dense-iso-active、dense-iso-total、routed+shared。"
  - "盯住 expert load、entropy、overflow、dead expert 和模态×专家的分布。"
  - "单卡 reference 验证通过后，再搬到 8 卡 Expert Parallel 上。"
---

# 第 11 课：实现稀疏专家 Thinker

> 内容：稀疏 FFN、路由观测与 Expert Parallel；建议时长：6–10 天
> 最小硬件：1×24GB 完成功能和 iso 对照；8 卡完成 Expert Parallel  
> 独立起点：preflight 锁定的官方 `jingyaogong/minimind-3o` dense checkpoint，不依赖视觉/视频课程

## 1. 用稀疏专家扩大容量

前三幕造完，系统已经像模像样：第一幕（01-04 课）把 MiniMind-O 跑起来、拆明白，声音怎么变 token、token 怎么变声音都有证据链；第二幕（05-07 课）教会它边听边想边说；第三幕（08-10 课）给它长出眼睛，一路做到原生视频和视频 token 压缩。唯独心脏没换过：Thinker 还是 8 层、768 宽的密集小模型（dense：每个 token 都要过全部参数）。

密集小模型的天花板在哪？想更聪明只能加宽加深，而每加一分参数，**每个 token 的计算量**同步涨一分。对一个要实时插话、prefill 里还塞着几百个视频 token 的系统，每 token 计算预算是硬约束；第 5-7 课好不容易压下去的延迟，会被更胖的 dense 模型原样吃回来。第四幕"换强心脏"就是冲这个来的：11-14 课依次上 MoE 稀疏专家、Mamba 混合架构、8 卡并行、长上下文。

 MoE（Mixture of Experts，稀疏专家混合）的主意很直白：与其造一个大 FFN（前馈网络：Transformer 每层里负责"加工"信息的全连接模块，参数大头），不如造 8 个小 FFN 当专家（expert），每个 token 只走其中 2 个；去哪由 router（路由器：一个小打分层，看一眼 token 向量给 8 个专家打分，取最高的 k 个）决定。总参数翻三倍，每 token 实际过的参数量不变；容量和计算量解耦了。仓库里已有教学版 `MOEFeedForward`，但缺 shared expert（人人都过的公共专家）、完整路由统计和多卡 Expert Parallel，本课补齐它们，并回答一个更严格的问题：MoE 的收益来自稀疏，还是只是参数多？为此造两个 dense 对照陪跑，拼同一份数据、同一个训练预算。

不做这课，第 13 课 8 卡并行里的 Expert Parallel 没有对象；第六幕要接的现代大脑几乎清一色是 MoE 心脏，看不懂路由就没法接。而"负载均衡"这个 MoE 特有的坑；专家有人过劳有人失业；只有在能逐 token 看路由的小规模上才练得出手感。

做完这课，你可以打开路由 trace，亲眼看到某个视频 token 在第 5 层被送进 2 号和 7 号专家、权重各多少；能指着三臂曲线说清收益是稀疏带来的还是参数多带来的；还能在 8 卡上看到 token 靠 all-to-all 通信各自找到专家所在的卡。

本课术语：

| 术语 | 简要解释 |
|---|---|
| MoE | 稀疏专家混合：多个小 FFN 当专家，每个 token 只走少数几个，容量大、单 token 计算少 |
| expert（专家） | 一个独立的小 FFN；本课每层 8 个 routed 专家外加 1 个 shared 专家 |
| router（路由器） | 给每个 token 的 8 个专家打分、选 top-k（进几个专家）的小线性层 |
| shared expert | 不经过路由、每个 token 都走的公共专家，兜底公共知识 |
| 负载均衡（load balancing） | 防止专家旱涝不均的手段：有人挤爆、有人常年没活干都算失败 |
| aux loss（辅助损失） | 负载均衡损失：把"分配不均"变成一项额外 loss 加进训练目标 |
| expert bias | 不加损失项的均衡法：给欠载专家的选择分数加一点偏置，只影响选谁 |
| dead expert | 长期没有 token 被路由过去的"失业"专家 |
| EP（Expert Parallel） | 把专家分到不同 GPU，token 靠 all-to-all（每张卡互发数据的通信原语）找专家 |
| iso-active / iso-total | 两个 dense 对照：一个匹配 MoE 的每 token 激活参数，一个匹配总参数 |

## 2. 本课比较哪些 MoE 结构

MiniMind-O 已有教学版 `MOEFeedForward`，但它缺少 shared expert、完整路由统计和 Expert Parallel（EP）。本课补齐这些能力，并把 MoE 与两个参数量匹配的 dense 模型放在同一实验中比较。

实验需要验证以下结论：

> 在每 token active FFN 计算近似相同的条件下，routed + shared experts 能用更大总容量改善多模态任务；收益不能被 dense-iso-active 或 dense-iso-total 解释，且不会靠牺牲少数模态或专家负载稳定性取得。

以下结果都不能支持该结论：MoE 只优于原始小 dense，却不如 dense-iso-active；收益只能由总参数更多解释；出现长期 dead expert 或单专家垄断；某一模态被系统性丢弃；启用 EP 后数值不一致或吞吐没有改善。MoE 天生占两个便宜——比小 dense 参数多，比大 dense 算得少；三臂对照逼你两头都比。

## 3. 进入条件与独立起点

- 已通过下述 preflight 的 MiniMind-O dense checkpoint、固定 tokenizer 和三模态回归集（preflight：开工前一次性解析并锁死全部外部依赖的检查，忘了的话回第 01 课的"冻结环境"）。
- 能记录每个 token 的 modality id 与 loss mask。
- 已有总训练 token、global batch、optimizer 可复现配置。
- 单卡先完成路由正确性；未通过不得直接做 EP。
- 8 卡硬件型号、显存、NVLink/PCIe 拓扑已记录。
- 本课不更换 attention、connector、codec 或数据 mixture。

### 3.1 不可变 checkpoint 契约

正式实验禁止使用模型别名、浮动分支或本地目录名识别模型。preflight 只解析一次以下官方发布物：

```yaml
# manifests/base_dense.lock.json
schema_version: 1
code:
  git_url: https://github.com/jingyaogong/minimind-o.git
  git_commit: a10fa6c148ed274d66f96dc119689e93e01be823
checkpoint:
  hub: huggingface
  repo_id: jingyaogong/minimind-3o
  repo_revision: ee3febbd08cc5b2bd41c039c825a8934232fee33
  filename: pytorch_model.bin
  sha256: 21530f9bbc540f461e2c0e29292ad359781d4d984d1e0c994510945f9b0edaab
tokenizer_revision: ee3febbd08cc5b2bd41c039c825a8934232fee33
required_source_config:
  hidden_size: 768
  thinker_layers: 8
  talker_layers: 4
  intermediate_size: 2432
  use_moe: false
```

preflight 以完整 commit/revision 下载，重算 checkpoint SHA-256，逐字段核对 `config.json`，并 strict-load 官方 dense 模型。它还要记录每个 source tensor 的 `name/shape/dtype/sha256`，生成只读 `base_dense.lock.json` 与 `run.lock.json`；训练阶段只读取 lock，不再联网解析版本。

## 4. 完成本课需要掌握的操作

1. 正确区分 active parameters、total parameters 与 active FLOPs；
2. 构造 dense-iso-active 和 dense-iso-total 两个容量对照；
3. 理解 top-k、capacity、dropless 和 load balancing；
4. 从路由 trace 中发现 dead expert 与 token drop；
5. 分析 `modality × expert`，但不把相关性误称为因果；
6. 实现 shared expert 与 aux-loss-free bias；
7. 在 8 卡上验证 EP all-to-all 和数值一致性。

## 5. 原理:边造边讲

四个机制，每个按同一节奏走五步：直觉、机制、数学、代码、验证。

### 5.1 参数账本：active 和 total 必须分开记

dense 模型只有一个参数量，MoE 有两个：total（仓库里放着多少）和 active（每个 token 实际用到多少）。质量大体跟着 total 走，计算成本和延迟跟着 active 走；两本账混着记，"MoE 更好"就无从审计——动手写模型之前先学会算账。

每层 FFN 参数近似正比于隐藏宽度。MoE 的 active 宽度是 shared 宽度加 k 个 routed 宽度，total 宽度是 shared 加全部 E 个 routed；两个 dense 对照分别按这两个宽度造。

忽略 bias，SwiGLU FFN（本模型用的 FFN 变体：三个投影矩阵、门控乘法激活）参数近似：

$$
P_{\mathrm{FFN}}(h) \approx 3d_{\text{model}}h
$$

含 shared expert 的 MoE：

$$
\begin{aligned}
h_{\text{active}} &= h_{\text{shared}} + k h_{\text{routed}}, \\
h_{\text{total}}  &= h_{\text{shared}} + E h_{\text{routed}},
\end{aligned}
$$

公平对照：

$$
\begin{aligned}
h_{\text{dense-iso-active}} &\approx h_{\text{active}}, \\
h_{\text{dense-iso-total}}  &\approx h_{\text{total}}.
\end{aligned}
$$

参数账本在步骤 3 的审计脚本里落地，对照第 11 节配置里的 `dense_iso_active_hidden: 768` 与 `dense_iso_total_hidden: 2304`。

公式只估算 FFN 参数量。实际比较还要用 profiler 记录 router、dispatch、expert GEMM（矩阵乘法运算）和通信成本，并检查 A/C 的 active FLOPs 误差是否在 ±5% 内。router 本身也有参数和计算，小模型上不可忽略，都要进账本。

### 5.2 路由:router 怎么给 token 找专家

路由是 MoE 的中枢：token 进来，router 打分，分高的 k 个专家干活，输出按分数加权合并。麻烦在于分数一身二职：既决定"去哪"（离散选择），又决定"听谁的多"（连续权重），负载均衡的所有纠结都源于此。

本课正式的 expert-bias 路径采用论文中的 sigmoid affinity（亲和度：router 给"这个 token 配这个专家"打的原始分；sigmoid 把每个分独立压到 0 到 1 之间，不像 softmax 强制专家间竞争）。

对 token $\mathbf x_t$：

$$
\begin{aligned}
\mathbf s_t &= \sigma(W_{\text{router}}\mathbf x_t), \\
\mathcal E_k(t) &= \operatorname{TopK}(\mathbf s_t+\mathbf b,k), \\
\widetilde s_{t,i}
&= \frac{s_{t,i}}{\sum_{j\in\mathcal E_k(t)}s_{t,j}},
\qquad i\in\mathcal E_k(t), \\
\mathbf y_t
&= \sum_{i\in\mathcal E_k(t)}\widetilde s_{t,i}E_i(\mathbf x_t).
\end{aligned}
$$

偏置 $\mathbf b$ 只改变 Top-k 选择，combine 权重仍由未加偏置的原始 affinity $\mathbf s_t$ 计算。若另测 softmax router，必须新开 gate-specific 实验，不能沿用本课 expert-bias 主臂的配置后把差异只归因于负载策略。

Top-1 每个 token 只进入一个 expert，active FLOPs 和通信较低。Top-2 进入两个 expert 并按归一化 affinity 合并输出，通常更稳定，但 active FLOPs 和通信都更高。

教学版实现在 `model/model_minimind.py::MOEFeedForward`（softmax 加 topk，第 6 节详拆）；本课的 sigmoid affinity 和 bias 路径在步骤 2 的单卡 reference 里重写。

单测必须确认每个 token 恰好出现 `k` 次，且选中权重之和为 1；这两条不成立，所有路由统计都是废数。

### 5.3 Capacity 与 dropless:专家的接客上限

router 自由分配，没人保证均匀，热门专家可能被塞进一半的 token。capacity limit 像餐厅限流：每个专家最多接这么多客，多的要么转桌（overflow）要么请走（drop）。类比失效处：被"请走"的 token 不是延后处理，而是这一层 routed 输出直接归零，信息真的丢了。

capacity limit 规定每个 expert 最多接收多少 token。超过容量的 token 会 overflow 或被丢弃，少数模态 token 可能先受影响——视频 token 挤在同一段序列里，专家瞬时爆满时最先被丢的就是它们。Dropless 模式处理所有 token，但负载最重的 expert 会决定本步完成时间，可能提高 all-to-all 的尾延迟。

本课主臂选 dropless，硬指标是 `drop_rate = 0`；代价记录在 step p95（第 95 百分位的单步耗时）里。

教学版 `MOEFeedForward` 没有 capacity/dropless 配置；本课在单卡 reference 中显式加上，8 卡用 Megatron Core 的 dropless 实现（步骤 7）。

需要同时记录任务正确率、每个 expert 的 token 数、drop rate 和 step p95，不能只看平均吞吐——drop 的代价在质量表上，dropless 的代价在延迟尾巴上。

### 5.4 Load balancing:治专家旱涝不均的两种药

路由有个自我强化的坏循环：某专家初始化稍好，接的 token 多、学得快，分数更高、接得更多——最后一家垄断，其他专家失业（dead expert），总容量名存实亡。干预是必须的，但干预有副作用。

药方一是辅助 loss：把负载均衡项加进训练目标；系数过大会迫使 router 接近均匀，干扰主任务优化。药方二是 aux-loss-free expert bias，它是一条完整的离散更新规则：不碰梯度，只在每个 microbatch 结束后，给欠载专家的选择分数加一点偏置、给过载的减一点。

对第 $\ell$ 层刚完成的一个 global microbatch，令 $\mathcal U_\ell$ 包含每个 non-padding 逻辑 token 恰好一次：

$$
\begin{aligned}
c_{\ell,i}
&= \sum_{t\in\mathcal U_\ell}
   \mathbf 1[i\in\mathcal E_k(t)], \\
\bar c_\ell
&= \frac{1}{E}\sum_{i=1}^{E}c_{\ell,i}, \\
e_{\ell,i}
&= \bar c_\ell-c_{\ell,i}, \\
b_{\ell,i}
&\leftarrow b_{\ell,i}
 + u\,\operatorname{sign}(e_{\ell,i}).
\end{aligned}
$$

其中 $\operatorname{sign}(0)=0$。低于平均负载的 expert 得到正向偏置，过载 expert 得到负向偏置；Top-k 时会更偏向前者。所有 bias 从 0 开始，更新速度 $u$ 只在 pilot 中选择一次，随后写入 run lock。

**代码落点与分布式细节。** 更新必须发生在该 global microbatch 完成之后，只对下一个 microbatch 生效。分布式实现先在 router-stat process group 上 all-reduce $c_{\ell,i}$；若 CP/EP 布局复制了 token，必须用 token ownership mask 去重。每个复制 router 的 rank 得到完全相同的 $\mathbf b_\ell$。bias 是不参与反向传播和 optimizer 的持久 buffer，但必须进入 checkpoint。

实现后分别记录无偏 affinity、加偏选择分数、combine 权重、全局 count 与 bias 轨迹。三处最易翻车：更新时机提前到本 microbatch（用未来信息路由当前 token）；combine 权重误用加偏分数（bias 只准影响选谁）；跨 rank 的 bias 不一致（同一 token 在不同卡走了不同专家）。

## 6. MiniMind-O 当前机制与代码落点

- `model/model_minimind.py`
  - `MiniMindConfig` 已有 `use_moe, num_experts=4, num_experts_per_tok=1, moe_intermediate_size, norm_topk_prob, router_aux_loss_coef`；
  - `MOEFeedForward` 用 `softmax → topk → Python for experts → index_add_`；
  - 当前实现单进程、无 shared expert、无 capacity/dropless 配置、无 grouped GEMM（把多个专家的小矩阵乘打包成一批算的加速手段）；
  - load balance 只有辅助 loss；
  - `MiniMindBlock` 在 `use_moe=True` 时把所有 FFN 替换为 MoE。
- `model/model_omni.py`
  - `TalkerModule` 继承 `use_moe`，可能让 Thinker 和 Talker 同时变 MoE；
  - `forward()` 汇总 Thinker/Talker 的 `aux_loss`。
- `trainer/train_sft_omni.py`
  - 只有 `--use_moe` 开关；
  - 训练器使用 DDP（多卡数据并行，忘了回第 01 课），不支持 expert process group；
  - 日志没有 expert load、entropy、overflow 或通信。

教学版的 Python for 循环逐专家跑一遍再 `index_add_` 拼回去，结果正确但 GPU 大部分时间在等——这正是它只能当 reference、不能当 production 的原因。

本课首先把 `moe_layer_pattern` 与 Thinker/Talker 配置解耦；默认只在 Thinker 每隔一层放 MoE，Talker 保持 dense，以限制变量——一次只动一个器官，语音输出的变化才不会污染 Thinker 的结论。

## 7. 目标架构

RMSNorm 后的 hidden 同时进入两条路径。shared expert 处理每个 token；router 为每个 token 选择 top-k routed experts，dispatch（把 token 按路由结果分发到各专家）后按归一化权重合并。最终输出是 `y_shared + y_routed`。

统一 trace：

```yaml
layer: 5
token_modality: video
expert_ids: [2, 7]
router_probs: [0.61, 0.24]
dropped: false
ep_rank: 1
```

现代实现路径分两层：

1. 单卡 reference：向量化 dispatch，结果易检查。
2. 8 卡 production：Megatron Core dropless MoE / grouped GEMM / EP。

单卡 reference 用于验证数值。8 卡版本使用 Megatron Core 的 dropless MoE、grouped GEMM 和 EP；本课不把自行实现的跨机 all-to-all 作为正式结果。

### 7.1 官方 8-layer 到三臂的确定初始化

三臂对照要公平，起点必须逐 tensor 相同——初始化不公平的实验，训练做得再干净也白搭。

本课固定使用官方 **8-layer Thinker**，不增加、复制或随机插入层。三个 arm 都使用恒等 source→target mapping：source 的第 0–7 层分别映射到 target 的第 0–7 层。

加载规则固定如下：

- tokenizer、embedding、LM head、全部 attention、RMSNorm、vision/audio projector、Talker 均从同一个 source tensor 原样加载。
- Thinker 的 `0/2/4/6` 层保留官方 dense FFN，三个 arm 的 tensor hash 必须相同。
- Thinker 的 `1/3/5/7` 层是受控替换位置；三个 arm 都丢弃这些层原有的 2432-width FFN，禁止只让 dense 臂继承预训练 FFN。
- 不允许新增第 9–12 层，也不允许用循环复制官方层。

为什么三臂都丢弃这四层的预训练 FFN？若 dense 臂继承预训练权重而 MoE 臂随机初始化，比较测的就成了"预训练对随机"，与结构无关。

四个替换位置各由 preflight 生成一个共享的 2304-channel SwiGLU 参数库：

| 项目 | 固定值或切片 |
|---|---|
| gate/up bank | `[2304,768]` |
| down bank | `[768,2304]` |
| initializer | `Normal(mean=0,std=0.02)` |
| layer seed | `uint64(SHA256("exp11-ffn-bank-v1:{layer}")[:8])` |
| A dense-iso-active | `bank[0:768]` |
| B dense-iso-total | `bank[0:2304]` |
| C shared expert | `bank[0:256]` |
| C routed expert `e` | `bank[256+256e : 512+256e]`，`e=0..7` |

C 的 router 使用独立固定种子 `SHA256("exp11-router-v1:{layer}")` 和同一 `Normal(0,0.02)`；expert-bias 全零。A/C 每 token active width 均为 `768`，B/C total width 均为 `2304`。所有新 FFN 权重都从同一参数库切片，初始化结果不受模块构造顺序影响。

preflight 保存 `manifests/ffn_init_bank.safetensors`、逐 tensor SHA-256 和三份初始化 state-dict hash。加载时允许缺失/新增的 key 必须精确等于四个替换 FFN 及 C 的 router/bias allowlist；出现其他 missing/unexpected key 立即失败。

## 8. 数据 recipe

### 8.1 Schema

沿用统一 Omni sample，并新增训练时派生字段：

```yaml
id: sample_001
messages: [...]
modalities: [text, image]
source: owned_or_dataset
license:
  media: verified
  annotation: verified
  redistributable: true
split: train
task: image_qa
token_accounting:
  text: 180
  image: 64
```

token modality 必须由 packer 显式生成。视觉和音频 embedding 可能共用占位 token id，因此不能根据 token id 范围推断模态——这里偷懒，第 13 节的 `modality × expert` 分析整张表作废。

### 8.2 Mixture

- 40% text tokens；20% image tokens；20% audio-input tokens；20% audio-output/other tokens。
- 使用 token-balanced sampler，不用样本-balanced。
- 每个 batch 至少包含两个模态，另保留单模态诊断 batch。
- license 对 media 和 annotation 分开；不清楚的数据不得进入可发布 full recipe。

为什么按 token 配平？一条视频样本的 token 数可能是纯文本样本的十倍；按样本配平，router 见到的 token 流仍被多数模态支配，少数模态的专家分配就会失真。

### 8.3 切分与规模

- 按原媒体/对话 identity 分组切分；所有实验臂共用 exact sample order。
- pilot：20M tokens；standard：300M–1B tokens；full：按参数规模至少 5B tokens 才讨论容量趋势。
- router 诊断集固定 10k samples，覆盖模态、任务、语言、长度。

## 9. 实施步骤

### 步骤 1：修正配置边界

新增 `thinker_use_moe`、`talker_use_moe`、`moe_layer_pattern`、`shared_expert_width`、`router_type`。默认 Talker dense，避免语音输出变化污染 Thinker 结论。

### 步骤 2：写单卡 reference MoE

验证 top-1/top-2、权重归一化、每个 token 恰好路由 k 次、无 token 静默丢弃。小 tensor 下与逐 token naive 实现逐元素比较。再用手算 count 向量验证 5.4 的 bias 更新方向、`sign(0)=0`、当前 microbatch 输出不变，以及保存/恢复后下一 microbatch 的选择一致。

### 步骤 3：构造 iso 对照

以固定的 `d_model=768, h_routed=256, h_shared=256, E=8, k=2` 审计三臂宽度，并分别打印 `total_params`、`active_params` 和 `estimated_active_flops`。

宽度不得在训练入口自动调整；若参数或 profiler active FLOPs 误差超过 ±5%，本次 preflight 失败并要求另建实验配置。

### 步骤 4：加入 shared expert

shared expert 对每个 token 都执行，routed experts 只处理 router 选中的 token。计算 dense-iso-active 宽度时必须把 shared expert 计入 active width——漏记它，A 臂就凭空少了 256 宽度，对照从根上不公平。

### 步骤 5：加入路由监控

每层记录 load histogram、CV、Gini、entropy、top-k probability、dead steps、overflow/drop、按模态条件分布、route churn（相邻 checkpoint 之间同一 token 换专家的比例）。

### 步骤 6：做负载策略预检

在 pilot 内前 2k steps 比较 aux loss 与 expert bias，两个分支保持同一 sigmoid affinity，只改变负载策略；选一个固定给主 MoE 臂。该预检不作为主要质量结论，也不能额外增加 MoE 总训练 token。

### 步骤 7：迁移 Megatron Core MoE

用相同权重做单 batch forward parity，再启用 grouped GEMM、dropless dispatch 和 EP。router 和 expert checkpoint mapping 要有显式测试。

### 步骤 8：做路由因果检查

评测时分别随机置换 expert 权重、固定路由和使用均匀路由，并记录任务分数变化。`modality × expert` 热图只能说明相关性；只有这些反事实测试能检查路由选择是否影响输出。热图说"视频 token 常去 3 号专家"，置换测试才回答"换掉 3 号，视频任务是否真的变差"。

## 10. 三个对照实验组

| 臂 | 结构 | 匹配目标 |
|---|---|---|
| A | dense-iso-active | 每 token FFN 参数/FLOPs 接近 MoE |
| B | dense-iso-total | 总 FFN 参数接近 MoE |
| C | shared + routed MoE | top-2，dropless |

锁定的官方 dense checkpoint 仅作历史质量参考，不获得额外训练预算。C 赢 A 说明同等每 token 计算下稀疏容量有用，C 追平 B 说明稀疏没漏掉多少总容量收益；两个都成立，MoE 才算真赢。

公平控制：同一个 `base_dense.lock.json`、恒等 layer mapping、共享 FFN init bank、8 层、`d_model`、attention、embedding、数据顺序、训练 token、optimizer、global token batch 和 seed；A/C active FFN 参数与 profiler FLOPs ±5%；B/C total FFN 参数 ±5%；替换位置固定为 `1/3/5/7`；能力按 token macro-average。

## 11. 配置与公式

```yaml
experiment: exp11_tiny_moe
model_lock: manifests/base_dense.lock.json
run_lock: manifests/run.lock.json
ffn_init_lock: manifests/ffn_init_bank.lock.json
backbone:
  d_model: 768
  layers: 8
  replaced_ffn_layers: [1, 3, 5, 7]
moe:
  layers: [1, 3, 5, 7]
  experts: 8
  top_k: 2
  routed_hidden: 256
  shared_hidden: 256
  dropless: true
  load_balance: expert_bias
  router_gate: sigmoid
  bias_init: 0.0
  bias_update_speed: 0.001  # 只作为 pilot 起点，选定后写入 run lock
  bias_update_scope: global_microbatch
  grouped_gemm: true
  expert_parallel: 1
controls:
  dense_iso_active_hidden: 768   # 256 + 2*256
  dense_iso_total_hidden: 2304   # 256 + 8*256
  runtime_width_search: false
train:
  tokens: 500_000_000
  seeds: [17, 23, 41]
```

```python
active_width = h_shared + k * h_routed
total_width = h_shared + E * h_routed
assert rel_error(params(dense_active), active_params(moe)) < 0.05
assert rel_error(params(dense_total), total_params(moe)) < 0.05
```

## 12. 训练预算与 8 卡执行

- **单卡 pilot**：4 experts、top-1/2，20M tokens；重点是 parity、route trace、无 dead expert。
- **标准质量实验**：2–4×24GB，8 experts、300M–1B tokens，三臂三 seed；优先并行臂而非过早 EP。
- **8 卡 EP**：`E=8, EP=8` 每卡一个 expert；DP=1。若专家太小，all-to-all 会压倒 GEMM，先加大 hidden 或每卡多个 expert。
- `num_experts % EP == 0`；检查进程组与数据并行组正交。
- 8×PCIe：先测 all-to-all microbenchmark；可能 EP=2/4 比 EP=8 更快。
- 8×NVLink/H100：启用 grouped GEMM、communication overlap；记录 token permutation 开销。
- 全程固定 global useful tokens/step；EP 不能改变数据 batch。

别把 EP 当免费加速：26M 模型的专家只有 256 宽，expert GEMM 太小时，token 卡间搬运时间反而超过计算——小模型上 EP 常是负优化，这正是本课要测的结论之一。

## 13. 指标与测量方法

### 能力

文本 PPL（perplexity，困惑度：模型对下一个 token 有多"吃惊"，越低越好）/固定 QA；图像 QA；音频 QA/ASR；语音输出回归；按模态 macro average；少数模态最差分数。

### 路由

$$
\operatorname{CV}_{\text{load}}
=
\frac{\operatorname{std}(n_1,\ldots,n_E)}
     {\operatorname{mean}(n_1,\ldots,n_E)},
$$

其中 $n_i$ 是分配给专家 $i$ 的 token 数。CV（变异系数）为 0 表示绝对均匀，越大越旱涝不均。还要同时记录：

- `Gini(load)`（基尼系数：不平等程度的另一个量法，对极端垄断更敏感）；
- `routing_entropy`（路由熵：路由分布的混乱度，塌缩成单专家时趋近 0）；
- `dead_expert_rate`；
- overflow/drop rate；
- modality–expert mutual information（互信息：知道 token 模态后，对它去哪个专家能多猜准多少）；
- checkpoint 间的 `route_churn`。

互信息需与保持边际分布的 permutation baseline 比较——两个边际分布不均匀的变量，即使独立也会算出非零互信息，不做置换基线就会把噪声当发现。

### 系统

tokens/s/GPU、step p50/p95、expert GEMM time、all-to-all time、communication overlap、peak HBM（显存）、MFU（实际算力利用率占硬件峰值的比例）、checkpoint size/load time。

### 测量

先 100 step warmup；再 500 step profile；同 global batch；A/C 同 active profiler FLOPs；3 seed；输出均值/标准差；按 token 类型与长度分桶。

## 14. 验收条件

1. 三臂的参数账本和训练 token 审计通过；
2. C 相对 A 的变化在多 seed 下稳定，或得到明确的负结果；
3. C 不只与原始 dense 比，也与 B 比容量边界；
4. dead expert 和连续空闲时长按专家完整报告；
5. dropless 模式下 `drop_rate=0`；
6. 少数模态的回退逐模态报告；
7. EP 与单卡 reference 的 forward/loss 在预注册容差内；
8. EP 吞吐与单卡、复制专家路径做同 batch 的比较；
9. 专家置换与固定路由反事实检查已经完成并保存。

实验有效后，再按下面的预注册判据决定能否写"Tiny MoE 胜出"。主统计单位是同一 held-out case；三个 seed 做两层配对 bootstrap（自助重采样：反复有放回抽样估计波动区间；CI 即置信区间），先重采样 seed，再重采样 case，共 10,000 次：

1. C 相对 A 的 text/image/audio/跨模态 macro accuracy 至少提高 2 个百分点，且配对 95% CI 下界高于 0；
2. C 相对 B 的 macro accuracy 非劣，配对 95% CI 下界不低于 -1.5 个百分点；
3. 在预注册的 standard profile 长度桶和 global token batch 下，C 相对 B 的吞吐至少提高 20%，且对相同 profile run 配对后的 95% CI 下界高于 10%；
4. C 相对 A 的任一单独模态回退，其 95% CI 下界都不低于 -2 个百分点；
5. dropless、dead-expert、数值 parity 和 checkpoint 恢复硬门槛全部通过。

若 C 只胜 A、但对 B 的容量边界不满足质量非劣，只能写"稀疏容量可能有收益"，不能写成 MoE 质量—效率同时胜出。`dead_expert_rate` 与连续空闲 step 的上限仍需根据 pilot 分布在打开 test 前写入 `stats_plan.yaml`；上面的最小效应、非劣界、case 列表和 bootstrap 脚本 hash 也必须同时冻结。

## 15. 失败诊断表

调试从最便宜的观测开始：先看路由 trace 和参数账本，再动训练配置，最后才怀疑结构本身。

| 现象 | 原因 | 修复 |
|---|---|---|
| MoE 赢原 dense 输 A | active 计算不公平 | 重算 iso-active |
| 只赢 A 不赢 B | 可能只是总容量 | 如实陈述容量/稀疏权衡 |
| dead expert | router 初始化/偏置 | 降 router LR、调 bias |
| 单专家垄断 | aux/bias 太弱 | 看分层 load，渐进调节 |
| load 均匀但质量差 | 平衡约束过强 | 降 aux，改 bias |
| 少数模态下降 | batch/路由被多数模态支配 | token-balanced mixture |
| EP 变慢 | expert 太小、all-to-all 主导 | 降 EP 或增 expert GEMM |
| EP loss 不一致 | dispatch/weight mapping 错 | 单 batch parity |
| top-2 token 重复错误 | combine index 错 | naive reference 对照 |
| 热图有明显聚类但消融无效 | 相关性非因果 | expert permutation |

## 16. 逐 case 要求

固定 120 cases：text/image/audio 各 30，语音输出 15，跨模态 15。每 case 输出答案、三个主臂结果、每层 expert id/probability、是否经过 shared expert、loss、延迟。另选 20 个错例做 route timeline。

错误标签：`capacity_gap`、`routing_collapse`、`minority_modality_starvation`、`expert_semantics_mismatch`、`distributed_dispatch_bug`、`reasoning_miss`。不得仅展示专家热图。

## 17. 交付物

```text
exp11/
  configs/{dense_iso_active,dense_iso_total,moe}.yaml
  manifests/base_dense.lock.json
  manifests/ffn_init_bank.{lock.json,safetensors}
  manifests/run.lock.json
  scripts/parameter_accounting.md
  data/manifest.jsonl
  metrics/{quality,routing,system}.jsonl
  traces/router/
  checkpoints/
  plots/load_and_modality_expert.png
  cases/index.md
  report.md
```

报告必须给出 actual/active/total 参数、实测 FLOPs、路由稳定性、少数模态、EP 成本和结论边界。

## 18. 复现清单

- [ ] preflight 已核验 exact Git commit、Hub revision 与 checkpoint SHA-256；
- [ ] source/target 均为 8 层，layer mapping 恒等；
- [ ] 三臂所有未替换 tensor 的 SHA-256 相同；
- [ ] 四层共享 FFN init bank 已锁定；
- [ ] config 不含 checkpoint 别名或运行时宽度搜索；
- [ ] Thinker/Talker MoE 解耦；
- [ ] 三臂参数账本通过自动审计；
- [ ] 数据顺序与有效 token 相同；
- [ ] naive/reference parity 通过；
- [ ] top-k 权重与索引单测通过；
- [ ] expert bias 按全局 count 更新，跨 rank 一致且只对下一 microbatch 生效；
- [ ] expert bias 未进入 optimizer，但 checkpoint 保存/恢复通过；
- [ ] drop/overflow、route trace 与 dead expert 监控已保存；
- [ ] EP 权重 mapping 已核对；
- [ ] 单卡/EP 数值 parity 通过；
- [ ] all-to-all profile 已保存；
- [ ] 三个 seed 均已完成；
- [ ] 逐 case 与因果路由消融完整。

## 19. 前沿对照与改造方向

### 19.1 公开系统采用的方案

同一个"容量与计算解耦"的问题，前沿是这样一路走过来的。[Switch Transformer](https://arxiv.org/abs/2101.03961) 把路由简化到 top-1，用 capacity factor 控制专家接客上限，证明同 FLOPs 下稀疏比 dense 收敛更快，也第一次系统写清 token dropping、路由塌缩这些坑。[Mixtral of Experts](https://arxiv.org/abs/2401.04088) 用 8 专家 top-2 做出公开权重的强 MoE，总参数约 47B、每 token 激活约 13B——active 和 total 两本账在其报告里就是分开报的；其专家分工分析还发现路由偏向 token 的表层特征聚类，专家并没有长成"数学专家""生物专家"。[DeepSeekMoE](https://arxiv.org/abs/2401.06066) 走细粒度路线：专家切小、多选几个，配 shared expert 承担公共计算，主张组合空间更大、专家更专业化——本课 C 臂的 shared + routed 结构直接来自这条线。[Auxiliary-Loss-Free Load Balancing](https://arxiv.org/abs/2408.15664) 提出的 expert bias 更新规则，就是 5.4 节逐式实现的那条；DeepSeek-V3（有公开技术报告，总参数 671B、每 token 激活约 37B）把 sigmoid affinity 加 bias 均衡用到了生产规模。多模态侧，Qwen3-Omni（有公开技术报告，30B-A3B 即总参数约 30B、每 token 激活约 3B）把 Thinker 和 Talker 都换成了 MoE；本课 Talker 保持 dense 只是为了控制变量。

### 19.2 教学设置与公开系统的差距

先分清哪些差距是钱的问题。专家数（8 对上百）、routed 宽度、训练 token（500M 对上万亿级）、支撑 EP 的 NVLink 集群，都是规模问题，机制上本课已对齐。剩下是机制差距：其一，前沿的专家粒度细得多，组合空间是 $\binom{E}{k}$ 量级，E=8、k=2 只有 28 种组合，专家专业化天然受限——19.3 第一个实验就动这里。其二，前沿在 EP 之上还叠通信域限制（按节点限制路由范围，控制跨机 all-to-all 流量），8 卡单机测不出这层，扩展题 5 是入口。其三，前沿的 MoE 心脏与多模态是联合预训练的，我们是把 MoE 移植到 dense 预训练模型上短训——路由在已有表征上学，趋势方向可参考，绝对收益不可外推。

### 19.3 动手改造清单

三个实验都是"重写模型结构"级，全部继承第 10 节的公平控制（同 lock、同数据顺序、同训练 token），各自新开实验目录，不得复用主臂 run lock。

1. **细粒度专家：E=8 拆成 E=16。** 验证 DeepSeekMoE 的核心主张在 26M 上有没有方向性信号。改法：`MiniMindConfig` 设 `num_experts=16, num_experts_per_tok=4, moe_intermediate_size=128`；7.1 节的 init bank 不变、只改切片规则（shared 仍取 `bank[0:256]`，routed expert `e` 取 `bank[256+128e : 384+128e]`，`e=0..15`），active width 仍为 $256+4\times128=768$、total 仍为 2304，与三臂完全可比。预算：与 C 臂同款 500M tokens、3 seed，2–4×24GB 约 3-5 天；先跑 20M pilot 确认无 dead expert。预期：macro accuracy 不低于 C（CI 下界不低于 -1.5 个百分点），modality-expert 互信息与 routing entropy 高于 C。失败判定：dead_expert_rate 明显高于 C 且调 $u$ 无效，或质量 CI 下界低于 -1.5 个百分点——如实写"该规模下细粒度收益不可见"，这也是有效结论。
2. **给 Talker 换稀疏心脏。** 把扩展题 4 做成完整实验：Thinker 保持 dense（A 臂结构），Talker 4 层中的第 1/3 层换成 shared+routed MoE，`talker_use_moe=true`、`thinker_use_moe=false`，宽度按 Talker 的 FFN 宽度等比缩放，init bank 照 7.1 节规则另生成 `exp11-talker-bank-v1`。预算：300M tokens、2 seed，2×24GB 约 2-3 天。预期：WER 与 speaker similarity（量尺定义回第 01 课）持平或改善，8 路 codebook loss 分项不恶化，路由出现 codebook 位置或音素相关的聚类。失败判定：WER 回退超过预注册非劣界，或 Talker 路由塌缩成单专家——说明语音 token 分布太均匀，撑不起路由分工。
3. **模态感知路由。** 检验"router 只看 hidden 够不够"：把 packer 生成的 modality id 做成小 embedding，与 hidden 拼接后送入 router，专家计算不变。改动位置：单卡 reference 的 router 构造与 `forward`，trace schema 加一列 `router_saw_modality`。预算：20M pilot 加 300M standard、2 seed，2×24GB 约 2-3 天。预期：modality-expert 互信息显著上升、少数模态最差分数改善，总 macro accuracy 不降。失败判定：互信息升了但少数模态分数不动甚至降——"分得开"不等于"学得好"，模态聚类是表象不是机制，这个负结果直接回应步骤 8 的因果检查。

### 19.4 顺手复现:论文结论到 26M 缩小版的映射

| 论文结论 | 26M 缩小版对应实验 | 预期能否复现同方向趋势 |
|---|---|---|
| Switch Transformer：同 active FLOPs 下稀疏比 dense 收敛快 | 主实验 C 对 A 的 loss 曲线与 macro accuracy | 能，方向可复现；幅度小于论文，E=8 的容量差距远小于论文设置 |
| Switch Transformer：capacity factor 越小 drop 越多、质量越差 | C 臂关 dropless，capacity factor 取 1.0/1.25/2.0，看 drop rate 与少数模态分数 | 能，且预期少数模态先受损——论文没细分的多模态特有现象 |
| Mixtral：专家不按领域分工，路由偏表层特征聚类 | 步骤 5 的 modality×expert 热图，限定纯文本桶内按任务分析 | 文本桶内预期复现"不按领域聚类"；跨模态整体可能模态聚类，设置不同，不算矛盾 |
| DeepSeekMoE：细粒度加 shared expert 提高专家专业化 | 19.3 实验 1（E=16）对 C 臂（E=8）的互信息与质量 | 方向大概率可见但信号弱；500M tokens 可能不够体现在终点质量 |
| Aux-loss-free：expert bias 均衡不逊于 aux loss 且不干扰主目标 | 步骤 6 的负载策略预检本身就是缩小版复现 | 能，预期 bias 分支 load CV 相当或更低、主 loss 不升；塌缩先查 $u$ 与更新时机 |

## 20. 论文精读与问题

1. [Switch Transformer](https://arxiv.org/abs/2101.03961)：路由、capacity、稳定训练；阅读检查：说明 token dropping 对少数模态的影响，并回答他们为什么敢用 top-1——答案要对上 5.2 节的权衡和本课选 top-2 的理由。
2. [Mixtral of Experts](https://arxiv.org/abs/2401.04088)：top-2 sparse MoE；阅读检查：分别计算并报告 active 与 total 参数——用 5.1 节公式验算，再对照其专家分工分析预判 19.4 表第三行的结果。
3. [DeepSeekMoE](https://arxiv.org/abs/2401.06066)：细粒度 experts 与 shared experts；阅读检查：说明 shared expert 承担的公共计算——没有它，routed 专家被迫各自复制哪些能力？对照 19.3 实验 1 作答。
4. [Auxiliary-Loss-Free Load Balancing](https://arxiv.org/abs/2408.15664)：expert bias；阅读检查：写出 count、平均负载与 bias 的完整更新式，并说明为何 bias 只参与选择且下一 batch 才生效——对照 5.4 节的三处翻车点，在论文里找到对应的设计决定。
5. [Megatron Core MoE 官方文档](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)：精读 EP、dropless、grouped GEMM、overlap 配置——哪些配置项是为第 12 节"小专家 all-to-all 压倒 GEMM"的问题准备的？
6. [MiniMind-O 官方仓库](https://github.com/jingyaogong/minimind-o)：逐行读 `MOEFeedForward`；阅读检查：列出需要替换的教学实现和必须保留的数值语义——for 循环换成向量化 dispatch 后，哪些数值行为必须逐元素不变？答案就是步骤 2 的 parity 清单。

## 21. 扩展题

1. 只让视频/音频 token 进入 routed experts，文本走 shared expert，做严格对照。  
2. 比较 router 在 SFT/RL 阶段冻结或继续训练。  
3. 做 expert pruning：删掉低利用专家后是否保留能力。  
4. 将 MoE 放在 Talker，但必须新开实验，不能污染本课（完整设计见 19.3 实验 2）。
5. 研究 device-limited routing，但先保持任务质量与通信变量分离。
到这里，第四幕第一刀落下了：Thinker 一半的 FFN 层换成稀疏专家，总容量翻三倍而每 token 计算量不变，路由的每次选择都有 trace 可查，"稀疏是否真有收益"有三臂对照和预注册判据兜底。但注意力还是老样子——序列每长一倍，KV cache 和注意力计算照涨。下一课[第 12 课](12_mamba_attention_hybrid.md)动第二刀：把部分注意力层换成 Mamba-2，做一场同样严格的长序列骨干对照实验。
