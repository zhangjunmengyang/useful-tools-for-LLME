---
id: 13_distributed_8gpu
title: "八卡训练系统"
summary: "FSDP2、EP、CP 这几种并行切法，能在不动 global batch、不改优化语义的前提下，把单卡显存压下来、把 useful-token 吞吐提上去吗？"
unit: backbone
play_tools: []
checkpoints:
  - "分清 DP、FSDP、TP、PP、EP、CP 六种并行各切什么东西、走哪种 collective 通信。"
  - "用 global useful tokens 固定每次更新的语义，别拿 micro batch 凑合近似。"
  - "先用 FP32 微型模型把计算算对，再检查 BF16 正式模型的数值是否一致。"
  - "测 strong scaling、MFU、通信占比、p95 step time，还有换并行度后能否照常恢复训练。"
---

# 第 13 课：实现可验证的八卡分布式训练

> 内容：参数、专家和上下文三种并行；建议时长：5–10 天；硬件：必须 8 张 CUDA GPU
> 独立起点：preflight 锁定的官方 `jingyaogong/minimind-3o-moe`；第 11 课只作原理背景，不是运行依赖

## 1. 八张卡怎样保持同一份训练语义

第四幕第三站。第 11 课把 Thinker 的 FFN 换成了稀疏专家（MoE），第 12 课比较了 Mamba-2 和 Attention 混合骨干。至今所有训练都在单卡上跑，但第 12 课已碰到 32K 长序列，下一课要冲 128K，本课标准档模型也涨到 300M–3B；一张卡既装不下训练状态，也算不完一条长序列。这课把"多卡怎么分工"练熟。

把同一个 MoE 模型的训练摊到 8 张卡上，三种切法各造一遍：把训练状态（参数、梯度、优化器）切成 8 份分头保管；让 4 个专家各住一张卡，token 该找谁就坐"班车"过去；把一条长序列横切两段，两张卡各算一半。三种切法共享一条铁律：**切开后算出的 loss、梯度、参数更新必须和单卡一字不差，或落在事先声明的浮点容差内。** 所以本课主线不是"跑得多快"，而是先建三层数值对账（forward、gradient、一步更新），全部通过才许谈吞吐。

先算一笔显存明白账。用 Adam 做混合精度训练，每个参数的显存开销约为：BF16 权重 2 字节 + BF16 梯度 2 字节 + FP32 主权重 4 字节 + Adam 一阶动量 4 字节 + 二阶动量 4 字节，合计约 16 字节。26M 参数的玩具模型只要约 0.4GB；标准档上限 3B 就是约 48GB；光训练状态就塞不进任何一张 24GB 卡，激活值还没算。这就是"切开就省"的全部道理：**这些字节本来是 8 张卡各存一份完整拷贝，纯属重复；切片后每卡只保管八分之一（48GB 变每卡 6GB），用时再临时找别人借。** 借要花通信，那笔账 5.3 细算。不练这手，第 14 课的 128K 长序列和第五幕的 RL 训练都会在 OOM（显存耗尽报错）面前卡死。

做完这课，你能把同一个官方 MoE checkpoint 按三种切法启动，看到三份 loss 曲线在容差内重合；在 profiler 时间线上指着空隙说"这是 all-to-all 在等 PCIe"；再把 A 布局的 checkpoint 搬进 C 布局继续训，逐 tensor hash 对得上。

本课术语：

| 术语 | 简要解释 |
|---|---|
| rank | 每个 GPU 进程的编号，8 卡就是 rank 0 到 7 |
| DP / DDP | 数据并行（data parallel）：每卡一份完整模型、各吃不同数据，梯度求平均；DDP 是 PyTorch 的标准实现 |
| ZeRO / FSDP | 状态分片：参数、梯度、优化器状态切成 N 份分卡保管，用时临时凑齐；ZeRO（zero redundancy optimizer）是方案，FSDP（fully sharded data parallel）是 PyTorch 实现，FSDP2 是新接口 |
| TP | 张量并行（tensor parallel）：把一层内部的大矩阵切开，几张卡合算一层 |
| PP | 流水并行（pipeline parallel）：按层切段，卡与卡像流水线工位 |
| EP | 专家并行（expert parallel）：MoE 的不同专家住不同的卡，token 路由到谁就发给谁 |
| CP | 上下文并行（context parallel）：一条长序列横切几段，各卡算各段，attention 时交换必要信息 |
| collective | 集合通信：一组卡同时参与的通信操作（all-reduce、all-gather、reduce-scatter、all-to-all） |
| NCCL | NVIDIA 的 GPU 集合通信库，上面那些操作的底层执行者 |
| MFU | 模型算力利用率：实际有效计算量占 GPU 理论峰值算力的比例 |
| parity | 数值对账：两种实现在同一输入上逐项比较输出，证明计算语义没变 |

## 2. 先验证数值，再比较性能

一个训练任务能在 8 卡上启动，只能说明进程和基本通信可用。它不能证明 loss 归一化、样本顺序、专家派发和参数更新仍与单卡一致。最容易翻车处：并行改错往往不报错，loss 照降，只是降的是另一个目标函数。本课先比较单卡、FSDP2、EP 和 CP 的 forward、gradient 与一次 optimizer update，对上账之后再测吞吐和显存。

实验需要验证以下结论：

> 根据模型稀疏性、序列长度和互联拓扑选择 FSDP2/EP/CP，可在不改变优化语义的前提下降低单卡 HBM（GPU 板载高带宽显存）占用，并提高有效 token 吞吐。

以下结果不能支持该结论：并行后的 loss 或 gradient 超出容差；global batch 或样本顺序改变；吞吐增长来自 padding 或少算 token；EP 的 all-to-all 或 CP 通信开销抵消计算收益；checkpoint 不能跨并行度恢复；p95 step time 出现明显长尾。

## 3. 进入条件与独立起点

- 8 卡位于同一可控节点；记录型号、显存、SM、互联。
- 官方 MoE checkpoint 可在单卡小 batch 完成 forward/backward/update。
- 本课内嵌的 4-expert、top-1 MoE 单卡 reference 已通过 top-k/dispatch gate（路由和派发忘了就回[第 11 课](11_tiny_moe.md)），才可做 EP。
- 具备 NCCL、PyTorch profiler/nsys 权限。
- 数据顺序可由 `sample_id` trace；global useful tokens 可精确计数。
- 先使用两层 micro model 做 FP32 parity，再用正式 BF16 模型。
- 不在本课改变模型结构、数据 mixture、optimizer 或训练目标。

## 4. 完成本课需要掌握的操作

完成本课后应能：

1. 区分 DP/DDP、FSDP、TP、PP、EP 与 CP 分别切分什么；
2. 说明 GPU 拓扑如何影响 collective 延迟和带宽；
3. 配置 FSDP2 的参数、梯度和 optimizer state 分片；
4. 配置 MoE expert process group 与 sequence/context parallel；
5. 建立 forward、backward、update 三层数值一致性检查；
6. 为给定机器设计 topology-aware rank mapping；
7. 测量 MFU、strong scaling、通信重叠和 checkpoint 开销；
8. 区分"算得对但不够快"与"吞吐高但改变了训练语义"。

## 5. 原理:边造边讲

四个机制，每个按同一节奏走：直觉、机制、数学、代码落点、验证。

### 5.1 并行维度:六种切法各切什么

训练时显存里躺着四类东西：参数、梯度、优化器状态、激活值（forward 时留给 backward 用的中间结果）。每种并行都在回答同一个问题：**这四类里哪一类在多卡之间是重复的、可以切开只存一份？** 数据并行把前三类复制了 8 遍；MoE 的全部专家挤在每张卡上也是复制；长序列的激活值则是单卡装不下，不切不行。

六种切法（第 1 节速查表已露面）：

- **DP/DDP**：每个 rank 持有完整模型，读取不同数据，梯度做 all-reduce（所有卡把各自的数加起来，人手一份总和）。
- **FSDP/ZeRO**：参数、梯度和 optimizer state 沿 DP ranks 分片，每卡只保管 1/N，算到哪层临时凑齐哪层。
- **EP**：不同 ranks 持有不同 experts，token 经 all-to-all（每张卡给每张卡各发一包数据）派发到专家所在的卡。
- **CP**：同一长序列沿 context 维切分，attention 需要交换 K/V 或等价通信——每个 token 要看前文，切开后前文有一半在别的卡上。
- **TP**：单层内部的权重矩阵按张量维切开，几张卡合算一层；本课不设主实验臂。
- **PP**：按层切成几段的层级流水；本课仅作扩展。

把第 1 节那笔 16 字节的账写精确。设参数量 $P$、分片度 $N$，BF16 混合精度加 Adam 下，单卡训练状态字节数约为

$$
M_{\text{state}} \approx \frac{(2+2+12)\,P}{N} = \frac{16P}{N},
$$

DDP 相当于 $N=1$（人人全存）；FSDP 全分片是 $N=\text{world size}$。激活值另算：它随 batch、序列长、层数走，CP 把它按序列维除以 CP 度，activation checkpoint（只存少数关键激活、backward 时重算其余，用计算换显存）再砍一刀。

DDP 是官方现状，在 `trainer/train_sft_omni.py`；FSDP2 的 `fully_shard` 与 mesh 在本课步骤 2 引入；EP 的专家归属改造对象是 `model/model_minimind.py::MOEFeedForward`；CP 的序列切分动 attention 的输入输出两端。

每种切法启用后，先量 `torch.cuda.max_memory_allocated()`，对照上式预测的量级；对不上就是有东西没被切到（常见嫌疑：frozen encoder 每卡一份，见 14 节）。

### 5.2 Global batch 语义:吞吐的分母不许注水

多卡最隐蔽的走样，是悄悄改了"一次参数更新见了多少有效 token"。padding（凑长度的填充位）不产生学习信号，多模态样本长短不一，各卡 padding 比例不同；用张量尺寸数 token，吞吐会虚高，且三臂虚高得不一样，比较就失效了。

必须固定每次更新的全局有效 token 数：

$$
T_{\text{global/update}}
=
\sum_{a=1}^{g}
\sum_{r=1}^{N_{\text{rank}}}
T_{\text{nonpad}}^{(a,r)},
$$

其中 $g$ 是 gradient accumulation（梯度累积：先攒几个小 batch 的梯度再更新一次参数）次数，$a$ 和 $r$ 分别索引 micro step 与 rank。不能用 `micro_batch × world_size` 代替，因为多模态和长序列的 padding 各卡不同。

`dataset/omni_dataset.py` 记录每样本 useful token；loss 归一化在步骤 3 改为全局 sum/count。

三个并行臂在同一 serialized batch 上数出的 $T_{\text{global/update}}$ 必须逐 update 相等；不等就先查 sampler 和 padding，谁也不许开始测吞吐。

### 5.3 通信:每种切法坐什么车、花多少运费

分片省下的显存不是白来的，代价是把"本地读显存"换成"跨卡发快递"。记清每种并行用哪种快递：

- 梯度同步：all-reduce/reduce-scatter（后者加完总和每卡只留自己那段，恰好配合分片）；
- FSDP 参数：all-gather（每卡把自己保管的 1/N 广播出去，人人凑齐整层）；
- EP token：all-to-all；
- CP context：all-gather/reduce-scatter 或 ring/p2p（环形逐跳传递）。

再算一笔通信明白账。DDP 每步只寄一次快递：全部梯度 all-reduce，环形算法下每卡收发约 $2P$ 个元素。FSDP 全分片要寄三次：forward 逐层 all-gather 参数、backward 再 all-gather 一次、最后 reduce-scatter 梯度，各约 $P$，合计约 $3P$，是 DDP 的 1.5 倍。**多付的这 0.5 倍运费，买到的是每参数 16 字节的仓储费降到 16/N 字节**——模型越大越划算；小到本来就装得下时，纯亏运费。EP 的 all-to-all 运的不是参数是激活（每层被路由 token 的 hidden state），量随 batch 和序列长走；CP 运的是 K/V 或等价物，量随序列长走。

单次 collective 耗时近似 $t \approx \alpha + B/\beta$：$\alpha$ 是启动延迟（消息再小也要付的固定成本），$\beta$ 是链路带宽，$B$ 是消息字节数；小消息被 $\alpha$ 支配，大消息被 $\beta$ 支配。耗时还随参与 rank 数和互联拓扑变化：同一配置在 NVLink/NVSwitch（卡间直连高速通道）与 PCIe 上结果可能截然不同。

collective microbenchmark 在第 8 节硬件画像先跑；正式训练的通信全部由 FSDP2/EP/CP 的 process group（进程分组：哪几张卡组成一个通信小组）触发。

先用 microbenchmark 扫出本机各消息尺寸、各 rank 数的实测带宽，再决定 EP/CP 的度；跳过这步，等于闭眼选快递公司。

### 5.4 数值一致性:浮点世界里"一样"怎么定义

浮点加法不满足结合律：`(a+b)+c` 和 `a+(b+c)` 的最后几位可能不同。collective 改变了求和顺序，并行版和单卡版注定做不到逐位相同——但"必然有小差异"不是"差多少都正常"的借口。要事先划线：线内是浮点噪声，线外是 bug。

分两级 gate：

1. 小模型 FP32、无 dropout：严格 gate。FP32 噪声极小，容差收得紧，实现错误几乎都会撞线。
2. 正式 BF16：趋势/容差 gate。BF16 只有约 8 个二进制有效位，按第 10 节预注册的容差执行。

精确阈值见 10.4 的验收表；比较对象是三层：forward 的 logits、backward 的 gradient（逐元素误差加方向余弦）、一步 optimizer update 后的参数。

parity 脚本在步骤 1 生成单卡 oracle，步骤 2/4/5/6 各自对账。

必须在固定 batch 上逐项比较 forward、gradient 和 update。只看最终指标接近不算数。比如 aux loss 语义变了，20 步内看不出来，2000 步后路由塌方。

## 6. MiniMind-O 当前机制与代码落点

- `trainer/train_sft_omni.py`
  - 使用 `DistributedDataParallel`；
  - `DistributedSampler` + 自定义 `SkipBatchSampler`；
  - 每 rank 的 seed 包含 rank，且每个 epoch 重置 seed；
  - optimizer 持有完整状态（没有任何分片）；
  - checkpoint 主要由主 rank 保存；
  - 没有 FSDP/EP/CP process group。
- `trainer/trainer_utils.py`
  - 负责 distributed init、seed、checkpoint；需要把 topology、world-size、parallel degrees 写入 manifest。
- `model/model_minimind.py`
  - `MOEFeedForward` 的专家全部复制在每个 rank 上；
  - Python loop dispatch，不支持 expert sharding；
  - Attention 没有 context partition contract。
- `model/model_omni.py`
  - vision/audio encoder 通过 `object.__setattr__` 持有；
  - frozen towers 与 FSDP wrapping 边界必须显式处理；
  - Thinker/Talker cache 和模块层次影响 auto-wrap。
- `dataset/omni_dataset.py`
  - 变长音频/图像 padding；需要记录 useful token，不能用 tensor size 计算吞吐。

DDP 用作已知正确的分布式起点。本课在此基础上增加状态分片、expert ownership 和 context partition，并分别验证每一项。

## 7. 目标系统

教学路径固定为三个主臂：

| 臂 | 同一 MoE 上启用的并行机制 |
|---|---|
| A | FSDP2-only |
| B | FSDP2 + EP |
| C | EP + CP，FSDP 关闭 |

规模路径使用 Megatron Core / Megatron Bridge；在本课的 8 卡设置中：

$$
N_{\mathrm{DP}}N_{\mathrm{EP}}N_{\mathrm{CP}}=8
$$

统一逻辑 rank 公式：

```yaml
world_size: 8
axis_names: [dp, cp, ep]
rank_formula: "rank = ((dp * CP) + cp) * EP + ep"
tp: 1
pp: 1
```

断言 `dp × ep × cp × tp × pp == world_size`，且 process groups 正交（每张卡在每个并行轴上有且只有一个分组，组与组不许串线）。

### 7.1 Canonical MoE checkpoint 与精确结构

三个主臂只改变 state-dict 的布局和 collective，不改变一个参数值。正式模型固定为以下官方发布物：

```yaml
# manifests/base_moe.lock.json 的人类可读表示
schema_version: 1
code:
  git_url: https://github.com/jingyaogong/minimind-o.git
  git_commit: a10fa6c148ed274d66f96dc119689e93e01be823
checkpoint:
  hub: huggingface
  repo_id: jingyaogong/minimind-3o-moe
  repo_revision: ae90b1f02858ed2f7ba7c7abb0881af30618b325
  filename: pytorch_model.bin
  sha256: 8a08e34b2c90212a65de93ac31547f0e2666819bb48c47979a31c365f8be30d3
tokenizer_revision: ae90b1f02858ed2f7ba7c7abb0881af30618b325
published_config:
  dtype: bfloat16
  hidden_size: 768
  intermediate_size: 2432
  num_hidden_layers: 8
  num_talker_hidden_layers: 4
  num_attention_heads: 8
  num_key_value_heads: 4
  head_dim: 96
  max_position_embeddings: 32768
  use_moe: true
  num_experts: 4
  num_experts_per_tok: 1
  moe_intermediate_size: 2432
  norm_topk_prob: true
  router_aux_loss_coef: 0.0005
runtime_assertions:
  thinker_moe_blocks: 8
  talker_moe_blocks: 4
  router: softmax_top1
  dispatch: python_expert_loop_capacity_free_no_drop
  aux_loss_scope_upstream: per_rank_per_layer_local_tokens
```

preflight 必须：

1. 以完整 commit checkout 代码，以完整 Hub revision 下载模型；本地重算 `pytorch_model.bin` 的 SHA-256。
2. 只把 `published_config` 逐字段与固定 revision 的 `config.json` 比对；字段缺失就是失败，不能用源码推断补成"config 已验证"。
3. 对固定 commit 实例化模型并运行小 batch，逐项执行 `runtime_assertions`：统计 Thinker/Talker 的 MoE block 数，hook router 与 expert dispatch，确认 softmax top-1、无 capacity 分支且每个逻辑 token 恰好处理一次。它们是源码/运行时断言，不是发布配置字段。
4. 以 strict load 检查 missing/unexpected keys 都为空。
5. 为 canonical state-dict 中每个 tensor 保存 `name/shape/dtype/sha256`，按 name 排序后计算 `state_dict_tree_sha256`。
6. 把 tokenizer、optimizer 初始化、数据 manifest、依赖 lock 和 cached modality embeddings 的 SHA-256 一并写入只读 run manifest。
7. 从 canonical state-dict 为 A/B/C 生成不同 layout；每个 layout 在第一次 forward 前重组抽检 tensor，并与同一 tensor-hash manifest 比对。

特别注意：上游当前的 router aux（路由均衡辅助损失，防止 token 挤向同一专家；细节回[第 11 课](11_tiny_moe.md)）是"每 rank、每层、局部 token"计算；这只是需要被 parity test 捕获的运行时事实。进入本课分布式主臂后，必须按 10.2 改成全局 sufficient-statistics 聚合，不能把局部 aux scalar 直接当作 canonical 并行语义。

正式训练入口只接受这份只读 manifest。不得使用第 11 课的临时 checkpoint、`latest`、目录别名或运行时自动选择某个 checkpoint。若更新版本，必须创建新的 manifest 和实验 ID。

### 7.2 三个固定 mesh 与 sharding axis

mesh（进程网格：把 8 张卡摆成一个多维表格，每个并行轴占一维）固定如下；物理 rank 可按 NVLink/PCIe 拓扑重排，但 `logical_to_physical` 映射必须进入 run manifest：

| 臂 | mesh `(DP, CP, EP)` | 逻辑 rank groups |
|---|---|---|
| A | `(8, 1, 1)` | `dp=[0,1,2,3,4,5,6,7]` |
| B | `(2, 1, 4)` | `ep groups={0,1,2,3},{4,5,6,7}`；`dp groups={0,4},{1,5},{2,6},{3,7}` |
| C | `(1, 2, 4)` | `ep groups={0,1,2,3},{4,5,6,7}`；`cp groups={0,4},{1,5},{2,6},{3,7}` |

参数布局是课程契约，不允许实现自行猜测：

- **A**：所有参数只在 `dp` 轴执行 FSDP2 `fully_shard`；`ep/cp` 均不存在。
- **B non-expert**：在 `ep` 轴逻辑复制，在同一 `ep` coordinate 的 `dp` 子 mesh 上 FSDP2；复制产生的 dense/router 梯度在 `ep` 轴同步。
- **B expert**：global expert `e` 只归 `ep=e` 所有，再在相同 `ep=e` 的 `dp` 子 mesh 上 FSDP2；禁止复制四个 experts 到每个 EP rank。
- **C non-expert**：在 `cp` 与 `ep` 轴复制，按 CP 算法交换 activation/KV 并同步参数梯度；因为 `dp=1`，FSDP2 必须显式关闭而不是挂一个无意义 mesh。
- **C expert**：global expert `e` 只归 `ep=e`，在两个 `cp` coordinate 上复制并跨 `cp` 同步梯度；不在 `ep` 轴同步不同 expert 的梯度。

因此 FSDP 的唯一合法 sharding axis 是 `dp`。EP 只表达 expert ownership/dispatch，CP 只表达 context partition；三者不可复用同一 process group。

## 8. 硬件画像

训练前先列出节点上的 GPU：

```bash
nvidia-smi -L
```

再保存 GPU 拓扑：

```bash
nvidia-smi topo -m
```

硬件报告还要记录 GPU memory、compute capability、NVLink、PCIe generation/width、CPU sockets/NUMA、RAM、local NVMe throughput、NCCL、CUDA driver/runtime 和 PyTorch build。

跑 collectives microbench：

- all-reduce；
- all-gather；
- reduce-scatter；
- all-to-all；
- 1MB、16MB、128MB、1GB；
- 2/4/8 ranks。

先找到拓扑带宽边界，再选 EP/CP degree。

## 9. 数据 recipe

### 9.1 Schema

```yaml
id: packed_000001
sample_ids: [a17, b92, c03]
source_shards: [train-0004.parquet]
license_summary:
  redistributable: true
  unresolved_items: 0
effective_tokens: 8192
padded_tokens: 8192
modality_tokens:
  text: 6016
  image: 512
  audio: 1664
split: train
pack_seed: 17
pack_version: v1
```

所有 rank 写 `sample_ids`，用于确认不同并行臂看到完全相同的 global sample order。

### 9.2 数据组成

先用两种：

- 短序列 2K：测 FSDP/EP；
- 长序列 16K/32K：测 CP。

固定 token mixture：60% text、15% image、15% audio、10% video/synthetic multimodal。媒体与 annotation license 分开汇总；本课不改变数据 recipe。

### 9.3 三档规模

| 档位 | 模型/序列 | updates | 目的 |
|---|---|---:|---|
| micro | 2 层、FP32、128 token | 5 | 数值严格 gate |
| pilot | 100–300M、2K/16K | 200 | profile 与配置筛选 |
| standard | 300M–3B、2K/32K | 2k+ | strong scaling/稳定性 |

## 10. 数值一致性协议

### 10.1 固定输入

- rank 0 生成全局 batch 并保存；
- 每个并行配置读取同一个 serialized batch；
- dropout=0；
- 同一 checkpoint；
- deterministic 算法能开则开；
- loss reduction 按 global 有效 token 求和再除，不先求 rank 均值。

最后一条展开：各卡 padding 比例不同，"先各卡平均、再求平均"会给 padding 少的卡更高权重，等价于换了 loss 函数；正确做法是分子分母各自全局相加，最后除一次。

### 10.2 Router aux 的全局聚合

主任务 loss 做全局 sum/count 还不够；router aux 也必须保持与未分片逻辑 batch 相同的统计语义。本课把统计范围固定为**每个 serialized global microbatch**。对第 $\ell$ 层，令 $\mathcal U_\ell$ 含每个 non-padding 逻辑 router token 恰好一次，$p_{\ell,t,i}$ 是 softmax 后第 $i$ 个 expert 的概率：

$$
\begin{aligned}
C_{\ell,i}
&= \sum_{t\in\mathcal U_\ell}
   \mathbf 1[i=\operatorname{Top1}(\mathbf p_{\ell,t})], \\
S_{\ell,i}
&= \sum_{t\in\mathcal U_\ell} p_{\ell,t,i}, \\
T_\ell
&= |\mathcal U_\ell|, \\
\mathcal L_{\text{aux},\ell}
&= \alpha E\sum_{i=1}^{E}
   \left(\frac{C_{\ell,i}}{T_\ell}\right)
   \left(\frac{S_{\ell,i}}{T_\ell}\right), \\
\mathcal L_{\text{aux}}
&= \sum_\ell \mathcal L_{\text{aux},\ell}.
\end{aligned}
$$

实现时在 `router_stats_group` 上 all-reduce sufficient statistics $(\mathbf C_\ell,\mathbf S_\ell,T_\ell)$。这个 group 与 token ownership mask 的组合必须让 group 并集覆盖 global microbatch，并且每个逻辑 token 只计一次；EP/CP 中复制的 activation 不能重复计数。$\mathbf S_\ell$ 是梯度路径，必须使用支持 autograd 的 collective，或数学等价且经过梯度 parity 证明的缩放；只做 detached all-reduce 供日志使用不合格。

不能先在各 rank 计算 aux scalar 再求平均，因为 `mean(local load × local probability)` 不等于"全局 load 均值 × 全局 probability 均值"——乘积的均值和均值的乘积是两回事。gradient accumulation 时，各 arm 必须读取相同的 serialized microbatches，并用相同的预注册权重组合各 microbatch loss。

### 10.3 比较点

每个配置都比较 input ids、modality tensors、选定层 activation、logits、total/text/audio/aux loss、选定参数 gradient、global grad norm、optimizer state 和一步更新后的参数。

### 10.4 数值验收

| 口径 | relative loss error | logit error | gradient cosine | 更新或漂移要求 |
|---|---:|---:|---:|---|
| FP32 micro | ≤ `1e-6` | max abs ≤ `1e-5` | ≥ `0.99999` | updated parameter max abs ≤ `1e-5` |
| BF16 formal | ≤ `3e-3` | 单独报告分布 | ≥ `0.999` | 20-step loss relative drift ≤ 1%，且无系统性 modality-loss drift |

若 kernel 非确定性导致略超阈值，必须定位原因并给出分布，不能直接放宽阈值。

## 11. 实施步骤

### 步骤 1：单卡 reference

从 7.1 的 canonical checkpoint strict-load 后，保存 micro batch、逐 tensor hash、forward activations、gradients、optimizer update 和 RNG state。它是所有并行配置的 oracle（对账时的标准答案）；不得从某个分布式 arm 的 shard 反向生成 reference。

### 步骤 2：FSDP2 改造

选择 auto-wrap：Thinker/Talker 按 block 级包裹；frozen vision/audio tower 不分片或单独管理。使用 mixed precision policy、activation checkpoint 和 sharded optimizer。这里只建立 A 臂的 FSDP scaling 路径：先 world-size=1，再 2/4/8。

### 步骤 3：修正 global loss

text/audio loss 分别累加 `loss_sum` 与 `valid_count`，跨所有相关 ranks 做 all-reduce，再除全局 count。router aux 则按 10.2 聚合每层 $(\mathbf C,\mathbf S,T)$ 后重新计算，禁止 all-reduce rank-local aux scalar。padding 分布或 EP/CP token ownership 不同，也应得到同一梯度语义。

### 步骤 4：CP-only micro validation

在 dense 2 层 Attention 上切 context，比较完整 Attention 输出/梯度。测试 causal mask、position、跨 shard 边界 needle（把关键信息埋在切分边界两侧的检索探针）。此项是数值预检，不算主要训练臂。

### 步骤 5：EP-only 主路径

使用 7.1 的 canonical 4-expert、top-1 MoE；先 EP=2 做实现预检，再运行正式 EP=4。EP=2 不是 B 臂 strong-scaling 测点。验证 token dispatch/combine、top-k 权重、10.2 的 global router aux 和 no-drop semantics。每个逻辑 token 全局恰好路由到 1 个 expert。

### 步骤 6：组合 EP+CP

建立正交 process groups；例如 EP=4、CP=2。逐样本确认 context shard 后路由 token 总数与未切分相同——两半各自路由，加起来不许多也不许少一个。

### 步骤 7：性能开关 sweep

依次而非同时打开：activation checkpoint、grouped GEMM（把多个专家的小矩阵乘合并成一次大的核函数调用）、communication overlap（通信与计算重叠：趁 GPU 算当前层时提前收发下一层的数据）、prefetch、mixed precision。每次只变一个开关并保留 profile——一次全开，快慢都说不清归谁。

### 步骤 8：checkpoint/reshard

保存 sharded checkpoint；在相同并行度恢复；再让同一个 canonical state-dict 在 A `(8,1,1)`、B `(2,1,4)`、C `(1,2,4)` 之间 reshard（换一种切法重新摆放同一批参数），比较逐 tensor hash 与 logits。

### 步骤 9：稳定性运行

至少 2k updates；记录 step p50/p95、straggler rank（拖后腿的慢卡，最慢者决定全队速度）、NCCL timeout、data wait、checkpoint 时间和恢复后 loss 连续性。

## 12. 三个对照实验组

| 臂 | 并行配置 | 用途 |
|---|---|---|
| A | FSDP2：DP=8，EP=1，CP=1 | 同一 MoE 的非 expert-aware 分片基线 |
| B | FSDP2+EP：DP=2，EP=4，CP=1 | 测量 expert ownership 与 all-to-all dispatch |
| C | EP+CP：DP=1，EP=4，CP=2，FSDP off | 测量 expert/context 组合布局 |

CP-only 只做强制 micro/pilot parity，不作为第四个完整质量臂。

公平控制：同一个 `base_moe.lock.json`、同一个 `state_dict_tree_sha256`、同模型结构、global useful tokens/update、serialized batch、样本顺序、optimizer、LR、dtype、训练 updates、loss 归一化和 seed。每个 2K/16K/32K 测点都让 A/B/C 读取同一 batch；C 不能因使用 CP 而独占更长序列。A 保持完整逻辑 MoE，只是不做 expert-aware ownership。

## 13. 配置示例

```yaml
experiment: exp13_ep4_cp2
arm: C
model_lock: manifests/base_moe.lock.json
hardware:
  world_size: 8
parallel:
  axis_names: [dp, cp, ep]
  rank_formula: "rank = ((dp * CP) + cp) * EP + ep"
  dp: 1
  cp: 2
  ep: 4
  tp: 1
  pp: 1
  sequence_parallel: true
fsdp:
  enabled: false
  shard_axis: null
expert_layout:
  owner_axis: ep
  experts: 4
  expert_id_equals_ep_coordinate: true
  replicated_axes: [cp]
  gradient_sync_axes: [cp]
non_expert_layout:
  replicated_axes: [cp, ep]
  gradient_sync_axes: [cp, ep]
precision:
  param: bf16
  reduce: fp32
  grad_scaler: false
train:
  global_useful_tokens_per_update: 262144
  grad_clip: 1.0
  loss_normalization: global_valid_tokens
  router_aux_aggregation: global_c_s_t_per_microbatch
  router_stats_unique_token_ownership: true
checkpoint:
  format: distributed
  async_save: false
profile:
  warmup_steps: 100
  capture_steps: [120, 121, 122]
```

进程组伪代码：

```python
assert DP * CP * EP == world_size
mesh = init_device_mesh(
    "cuda",
    mesh_shape=(DP, CP, EP),
    mesh_dim_names=("dp", "cp", "ep"),
)
dp_group = mesh["dp"].get_group() if DP > 1 else None
cp_group = mesh["cp"].get_group() if CP > 1 else None
ep_group = mesh["ep"].get_group() if EP > 1 else None
assert groups_are_orthogonal(ep_group, cp_group, dp_group)
if DP > 1:
    fully_shard(non_expert_blocks, mesh=mesh["dp"], reshard_after_forward=True)
    fully_shard(local_owned_experts, mesh=mesh["dp"], reshard_after_forward=True)
else:
    assert not fsdp.enabled
```

A/B/C 各有一份完整配置，不能靠运行时修改上例的 degree。preflight 逐份检查 mesh 乘积、专家数可整除 EP、FSDP 只引用 `dp`、三个 arm 的 model/state-dict/data hash 完全一致。

## 14. 8 卡执行建议

### 8×24GB PCIe

- 优先 FSDP2、activation checkpoint 和短序列；
- EP=2 只做实现预检，再运行正式 EP=4；本课 canonical checkpoint 只有 4 个 experts，不允许 EP=8；
- CP=2 优先，避免过多小通信（回 5.3：小消息被启动延迟支配）；
- 数据与 checkpoint 放本地 NVMe；
- 不要让 frozen vision encoder 每 rank 重复占满显存，可离线缓存。

### 8×48GB，部分 NVLink

- 按 NVLink island 建立 EP group（all-to-all 最重，让它走最快的路）；
- 跨 island 尽量放 DP/CP 中通信较可控的维度；
- 测 EP=4×DP=2 与 EP=4×CP=2 的边界，但主课仍保留三臂。

### 8×80GB NVSwitch/H100

- canonical checkpoint 只有 4 个 experts；在 8 卡内固定 EP=4，不硬凑 EP=8；
- `EP=4,CP=2` 适合主组合；
- grouped GEMM + overlap；
- FP8 属于扩展变量，不进入主实验。

无论 GPU 型号如何，都先测 topology 和 collectives，再判断三套固定 mesh 是否适合当前机器；不适合时报告负结果，不能静默改主臂并行度。GPU 数量本身不能说明 all-to-all 或 all-gather 的带宽。

## 15. 指标与测量方法

### 正确性

loss/logit/gradient/update 误差；CP 边界 needle；EP token conservation（派发出去的 token 数等于收回来的）；checkpoint 恢复连续性；reshard 前后 logits。

### 性能

- useful tokens/s/GPU 与总 tokens/s；
- MFU（见第 1 节速查表）。统计窗口内按下式计算：

  $$
  MFU=
  \frac{F_{\text{model}}}
       {\Delta t\;N_{\text{GPU}}\;P_{\text{peak}}}
  \approx
  \frac{R_{\text{token}}F_{\text{train/token}}}
       {N_{\text{GPU}}P_{\text{peak}}}
  $$

  分子 $F_{\text{model}}$ 是窗口内实际执行的模型计算量；近似式中的 $R_{\text{token}}$ 是全局每秒参与模型计算的 token 数，$F_{\text{train/token}}$ 是一次前向和反向的模型 FLOPs/token。统一规定一次乘加算 2 FLOPs；MoE 只计算每个 token 实际路由到的 top-k experts，并计入 attention 等非 expert 模块，不把所有 experts 的总参数量都算进去。通信、optimizer、数据读取不计入分子，它们造成的等待会通过 $\Delta t$ 降低 MFU。分母中的 $P_{\text{peak}}$ 是该训练数据类型下单卡理论峰值 FLOPs/s，再乘 GPU 数；必须报告采用的硬件和 BF16/FP16/FP8 口径。peak HBM 是字节数，只能衡量显存容量，不能代替 $P_{\text{peak}}$。若 padding 仍执行了 kernel，MFU 的 token 数要包含它，同时另报 useful-token 比例；
- strong-scaling efficiency（加卡效率：卡数翻倍、吞吐是否跟着翻倍）只在同一 arm 内、EP/CP degree 与 global workload 保持不变时定义：

$$
\eta_a(N;N_0)
=
\frac{\operatorname{throughput}_a(N)}
     {(N/N_0)\operatorname{throughput}_a(N_0)}
$$

- 合法 GPU 数与基线：

| arm family | 固定维度 | 合法 $N$ | $N_0$ | 可报告的 strong scaling |
|---|---|---|---:|---|
| A | `EP=1, CP=1, DP=N` | `{1,2,4,8}` | 1 | $\eta_A(2),\eta_A(4),\eta_A(8)$ |
| B | `EP=4, CP=1, DP=N/4` | `{4,8}` | 4 | $\eta_B(8;4)$ |
| C | `EP=4, CP=2, DP=1` | `{8}` | — | 无合法曲线 |

B 的 4 卡基线是 `DP=1, EP=4`，8 卡点才是正式 `DP=2, EP=4`；EP=2 预检不属于这条曲线。C 在本课单节点只有 8 卡这一个合法点，不能拿不存在的 1 卡配置计算 $\eta_C(8)$；只在 8 卡上与 A/B 横向比较，并单列 CP 通信、HBM 与长序列收益。

- step p50/p95；peak HBM；all-gather/reduce-scatter/all-to-all 时间；overlap 比例；data wait；checkpoint save/load。

### 测量纪律

- 100 step warmup、500 step 统计；
- 同一 global batch；
- 分短 2K 和长 32K；
- 至少重复 3 次 profile；
- 记录所有 rank，不能只看 rank 0；
- 系统结果报告硬件拓扑；
- 能力在 2k updates 后与单卡/历史趋势比较。

## 16. 验收条件

正确性是硬门槛：

1. FP32 micro 的四项数值检查全部通过；
2. BF16 forward、backward、update 的误差都在预先声明的容差内；
3. CP 跨 shard 边界的 needle case 与完整序列结果一致；
4. EP 没有 token 丢失或重复；`runtime_assertions` 的 no-drop 语义下 `drop=0`；
5. 三臂的 global useful tokens 和样本顺序一致；
6. checkpoint 可以恢复，恢复前后 loss 没有无法解释的跳变；
7. reshard 后的 logits 通过同一套 BF16 容差检查。

性能没有跨机器通用的及格线。开始正式测量前，按 15 节合法范围预注册目标：A 测 1/2/4/8 卡，B 只测 4/8 卡，C 只测 8 卡横向点。随后完整报告：

- A/B 的合法 strong-scaling efficiency；C 不伪造曲线；
- 每个合法测点的 peak HBM 绝对值；A 另报相对单卡变化，B 相对 4 卡基线；
- useful-token throughput 的绝对值和相对变化；
- step p50、p95 以及每个 rank 的长尾来源。

如果正确性全部通过，但性能没有超过预注册基线，这仍然是合格的负结论：当前硬件拓扑、模型粒度或序列长度不适合该并行度。阈值必须在查看 test 结果前冻结，不能根据结果重新调整。

## 17. 失败诊断表

调试从最便宜的一层查起：loss 不对先查归一化，再怀疑 NCCL。

| 现象 | 原因候选 | 诊断/修复 |
|---|---|---|
| loss 差一倍 | rank 均值再平均 | global loss sum/count |
| aux loss 随并行度漂移 | 对 rank-local aux scalar 求平均或 token 重复计数 | all-reduce 全局 $(C,S,T)$，审计 ownership mask |
| CP 边界答案错 | causal mask/position 分片错 | micro needle trace |
| EP 输出错 | token permutation/combine 错 | token conservation |
| EP 慢于复制 | expert 太小/all-to-all 太大 | 降 EP、grouped GEMM |
| FSDP 频繁 OOM | all-gather 峰值/错误 wrap | block wrap、reshard |
| p95 尖峰 | straggler/data/NCCL | per-rank timeline |
| 吞吐虚高 | padding/少算 token | useful token 计数 |
| resume loss 跳 | optimizer/RNG/data cursor 丢失 | 完整状态 checkpoint |
| reshard 失败 | expert id mapping 不稳定 | global expert registry |
| overlap 无收益 | stream 依赖或通信太短 | profiler 检查空隙 |

## 18. 逐 case 与系统逐步要求

能力 case 至少 60 个：20 text、10 image、10 audio、10 个跨 CP 边界长检索、10 个 MoE 路由。每个 case 保存单卡/A/B/C logits 摘要、loss、expert ids、context shard 边界和误差。

系统分析以 step 为单位：至少保存 30 个完整 step trace，包含每个 rank 的 useful tokens、data wait、forward/backward/optimizer、collectives、HBM 和 straggler。错误标签：`loss_normalization_bug`、`cp_mask_bug`、`ep_dispatch_bug`、`topology_bottleneck`、`data_stall`、`checkpoint_state_gap`。

## 19. 交付物

```text
exp13/
  hardware/topology.md
  hardware/collectives.json
  configs/{fsdp8,fsdp2_ep4,ep4_cp2}.yaml
  manifests/base_moe.lock.json
  manifests/tensor_hashes.json
  manifests/run.lock.json
  data/manifest.jsonl
  parity/{fp32,bf16}/
  profiles/nsys/
  metrics/{correctness,throughput,memory,checkpoint}.jsonl
  checkpoints/
  cases/index.md
  report.md
```

报告必须写明瓶颈位于 HBM、GEMM、all-to-all、all-gather 还是 data loader，并用 topology 和 profile 解释所选并行度。数值阈值、超限点和例外也要逐项列出。

## 20. 复现清单

- [ ] preflight 已核验 Git commit、Hub revision、checkpoint SHA-256 与 `published_config`；
- [ ] Thinker/Talker MoE、router、dispatch 与 upstream aux scope 已作为独立 runtime assertions 执行；
- [ ] 三臂的 `state_dict_tree_sha256` 一致；
- [ ] A/B/C mesh 分别为 `(8,1,1)`、`(2,1,4)`、`(1,2,4)`；
- [ ] FSDP 只使用 `dp` 轴，C 臂明确关闭；
- [ ] expert ownership 与复制/梯度同步轴通过断言；
- [ ] GPU、topology、NCCL 信息已记录；
- [ ] collectives microbench 已保存；
- [ ] serialized global batch 已固定；
- [ ] FP32 micro gate 与 BF16 gate 均通过；
- [ ] global loss 使用全局 sum/count；
- [ ] router aux 按层聚合全局 $(C,S,T)$，且每个逻辑 token 恰好计数一次；
- [ ] $\mathbf S$ 的跨 rank 聚合保留 autograd，aux gradient parity 已通过；
- [ ] useful token 计数已核对；
- [ ] CP-only 边界测试与 EP token conservation 均通过；
- [ ] process groups 两两正交；
- [ ] per-rank profile 已保存；
- [ ] 100-step warmup 与 500-step 统计已完成；
- [ ] scaling 只使用 A `{1,2,4,8}`、B `{4,8}` 与 C `{8}` 的合法范围；
- [ ] checkpoint 恢复和 A/B/C reshard 已验证；
- [ ] 2k-step 稳定性已检查；
- [ ] 能力 case 与系统逐 case artifact 完整。

## 21. 前沿对照与改造方向

本课的三种切法在前沿系统里同时上阵，还叠着本课没开的维度。[Megatron Core 并行指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)是工业界标准做法：TP、PP、DP、CP、EP 组成多维 mesh，rank 按"通信最重的维度放最快互联轴"排布——与 14 节"按 NVLink island 建 EP group"同一原则，放大到几千卡。[ZeRO](https://arxiv.org/abs/1910.02054)是 5.1 那笔 16 字节账的原始出处：优化器状态、梯度、参数三级分片，逐级给出显存与通信公式；FSDP2 是它的 PyTorch 原生化。长序列方向，[DeepSpeed-Ulysses](https://arxiv.org/abs/2309.14509)用 all-to-all 把"按序列切"临时倒换成"按注意力头切"，attention 算完再换回；Megatron 的 [CP 实现](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)用 ring 式 K/V 交换配合 GQA 减少传输量。MoE 方向，DeepSeek-V3（公开技术报告，671B 总参数、37B 激活）把 EP 推到跨节点规模，用计算与通信重叠掩盖 all-to-all，并用无辅助损失的负载均衡绕开 10.2 那类 aux 聚合难题（见第 11 课引用的 [Auxiliary-Loss-Free Load Balancing](https://arxiv.org/abs/2408.15664)）。多模态侧，[VeOmni](https://github.com/ByteDance-Seed/VeOmni)专门处理第 6 节的难题：frozen encoder、变长多模态 padding 与 FSDP/序列并行的组合封装。

规模问题（钱能解决）：卡数、互联（PCIe 对 NVSwitch 与跨节点网络）、并行维度数（3 维对 5 维）、FP8 低精度。机制问题（本课能解决）：官方 DDP 起点没有状态分片、专家复制在每张卡、attention 没有序列切分接口，这三刀本课补上；数值对账、useful-token 口径、预注册阈值这些纪律与前沿同款。剩下的差在工程深度：前沿的 dispatch 是融合 kernel 加 grouped GEMM，我们是 Python 循环；前沿的 overlap 是手工编排的通信调度，我们靠 FSDP2 默认 prefetch——下面清单逐个丈量。



1. **换分片策略：`reshard_after_forward=False`（ZeRO-2 语义）对 `True`（ZeRO-3 语义）。** 位置：步骤 2 引入的 `fully_shard` 调用参数（`trainer/train_sft_omni.py`），A 臂各跑一份。对照 5.3 的通信账：forward 后不放掉凑齐的参数，backward 省一次 all-gather，通信从约 $3P$ 降到约 $2P$，代价是常驻显存变大。预算：pilot 档 100–300M、2K 序列、200 updates × 2 份，约 8–16 GPU-hour。预期：tokens/s 与 MFU 上升（PCIe 上更明显），peak HBM 上升，两份 BF16 parity 都过。失败判定：吞吐无可测差异（all-gather 本就被 overlap 掩盖，记负结果）或 parity 超差（碰了语义，回修）。
2. **换通信重叠方式：FSDP2 显式 prefetch 编排对默认行为。** 位置：步骤 2 wrap 完成后，用 FSDP2 的 forward/backward prefetch 接口按 Thinker 8 层加 Talker 4 层的 block 顺序显式登记预取。预算：A 臂 pilot，2K 与 16K 各 200 updates，约 10 GPU-hour。预期：profiler 上 all-gather 与计算的重叠比例上升，step p50 下降，MFU 升 1 个百分点量级即算有效；16K 收益应大于 2K（计算更厚，更容易盖住通信）。失败判定：重叠上升但 step time 不降（通信不在关键路径，负结果照报）；或 HBM 峰值超限、OOM——预取本质是提前占显存，正好复现 17 节"FSDP 频繁 OOM"一行。



- ZeRO 的"训练状态显存随分片度近线性下降"：A 臂 world-size 1/2/4/8 各测 peak HBM（15 节本就要求），对照 5.1 的 $16P/N$。预期：状态部分近线性降，总显存不按 $1/N$ 走，因为激活值和 frozen encoder 不随 FSDP 分片——偏离量就是激活占比的直接测量。
- ZeRO 的"全分片通信量约为 DDP 的 1.5 倍"：profiler 统计 A 臂一个 step 的 all-gather 加 reduce-scatter 总字节，除以 DDP 基线的 all-reduce 字节。预期比值落在 1.5 附近；明显偏高通常是 frozen tower 被错误 wrap 进了分片（17 节"错误 wrap"一行）。

## 22. 论文与官方文档精读

每篇对应一个能在本课产物里验证的问题，答案写进报告；答不上来就回去重读。

1. [ZeRO](https://arxiv.org/abs/1910.02054)：精读三类状态分片和通信。带着问题读：论文的 stage 1/2/3 各切掉哪类状态、各多付多少通信？读完对照自己步骤 2 的配置，列出 FSDP2 与 ZeRO-3 分别分片的训练状态，并指认 5.3 那笔 $3P$ 通信账在论文哪一节。
2. [DeepSpeed-Ulysses](https://arxiv.org/abs/2309.14509)：精读 sequence parallel 的 all-to-all。带着问题读：它把"按序列切"倒换成"按头切"发生在计算图哪个位置？对照本课 CP 实现，写出两者在 GQA 下各自的通信量表达式。
3. [PyTorch FSDP2 官方文档](https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)：精读 `fully_shard`、mesh、reshard。带着问题读：`reshard_after_forward` 的两个取值分别对应 ZeRO 哪个 stage？改造清单第 1 项的预期就从这里来。
4. [Megatron Core Parallelism Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)：精读并行维度组合和 rank order。带着问题读：它建议把哪个维度放在最内层、为什么？对照 7.2 自己的 `rank_formula`，解释 ep 放最内层的拓扑理由。
5. [Megatron Core Context Parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)：精读 CP 通信与 GQA 的关系。带着问题读：KV head 少于 Q head 时，CP 交换的字节数怎么变？本课 canonical 配置 `num_key_value_heads: 4`，算出它对 CP 通信量的影响。
6. [Megatron Core MoE](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)：精读 EP、token dispatcher、grouped GEMM、overlap。带着问题读：它的 dispatcher 有哪几种实现、各适合什么专家规模？判断本课 4-expert、768×2432 的小专家落在哪个适用区（提示：5.3 的 $t \approx \alpha + B/\beta$，小消息全在付启动延迟）。
7. [VeOmni 官方仓库](https://github.com/ByteDance-Seed/VeOmni)：精读 FSDP/长序列训练抽象。带着问题读：它怎么处理 frozen encoder 与分片边界？对照 6 节 `object.__setattr__` 持有 encoder 的写法，写出本课 wrap 边界的对应处理。
8. [MiniMind-O 官方仓库](https://github.com/jingyaogong/minimind-o)：精读 DDP、sampler、checkpoint。阅读检查：列出恢复训练时保持 data cursor 一致所需的全部状态（sampler 位置、RNG、epoch、step），对照 17 节"resume loss 跳"一行。

## 23. 扩展题

1. 加 TP=2，在 8 卡上比较 `TP2×EP4`，重新做全部数值 gate。
2. 加 PP，研究异构 Mamba/Attention 层（见[第 12 课](12_mamba_attention_hybrid.md)）的 stage 负载均衡。
3. 异步 checkpoint，但先证明失败恢复的一致性。
4. FP8 训练另开实验，不能与并行收益混合归因。
5. 两节点扩展时先跑跨节点 collectives，再决定将哪一维跨节点。

读完这课回头看：模型一个参数没动，但同一份权重能以三种布局在 8 张卡上训练，每种布局都有对账凭证，checkpoint 能在布局间自由搬家。[第 14 课](14_long_context_curriculum.md)就要用上这身本事：把上下文从 32K 渐进拉到 128K，检验模型会不会用远处的证据——长序列训练的每一步，都跑在这课搭好的多卡地基上。
