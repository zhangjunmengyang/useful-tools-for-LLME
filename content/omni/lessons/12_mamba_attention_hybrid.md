---
id: 12_mamba_attention_hybrid
title: "Mamba-2 × Attention"
summary: "参数和训练 token 都配平之后，hybrid（大部分层换 Mamba、留几层 Attention）真能在长序列上省 cache/HBM 或提吞吐，还不丢 Attention 那种精确翻旧账的本事吗？"
unit: backbone
play_tools: []
checkpoints:
  - "能讲明白 selective SSM、scan/recurrent 两种等价算法和 Mamba-2 的 SSD。"
  - "实现可配置的 attention/Mamba 层排布，以及两种层各自不同的 inference state。"
  - "把非 embedding 参数、训练 token 和长度分布配平到可比。"
  - "真实 QA、needle/copy 检索、prefill、decode、KV cache、SSM state 一起测，一个不落。"
---

# 第 12 课：比较 Mamba-2 与 Attention 混合架构

> 内容：Mamba-2、GQA Attention 与混合序列层；建议时长：7–12 天；最小硬件：1×24GB 做 8-layer pilot，4–8 卡做标准长序列实验
> 独立起点：preflight 锁定的官方 `jingyaogong/minimind-3o` dense checkpoint + 已冻结的模态输入 contract；不依赖 MoE

## 1. 为什么要混合 Mamba 和 Attention

第 11 课只替换了 FFN，本课比较负责序列信息交换的 mixer。实验关闭 MoE，单独测量 Attention、Mamba-2 和混合骨干，避免容量变化干扰架构结论。

Attention 将历史 token 的 key/value 保存在 KV cache 中，因此缓存随序列长度线性增长，标准 prefill 的计算量随长度平方增长。它的优势是能够按内容精确访问历史位置。视频和多轮对话很容易超过一万 token，这使长上下文的显存与吞吐成本迅速增加。

状态空间模型（SSM）把历史压缩到固定大小的状态向量，每读一个 token 更新一次，因此状态内存和单步计算不随上下文长度增长。压缩也可能丢失需要精确回查的细节。混合方案在 8 层 Thinker 中使用多数 Mamba-2 层，并保留少量 Attention 层处理内容寻址。

两个对照实验分别回答不同问题。12A 从随机初始化出发并配平总参数，比较同预算下的架构；12B 从官方 checkpoint 出发，测试结构迁移能否保留已有能力。

三个下游压着这课：第 13 课做 8 卡并行，混合模型各层成本不均匀，流水线切分要按本课实测的每层成本来；第 14 课长上下文直接消费本课的 crossover 结论；第 18 课要接的 Nemotron 大脑本身就是 Mamba-2/Transformer/MoE 混合骨干，这课不亲手造一遍小的，到时候接的就是黑盒。

2K、8K 和 32K sweep 同时报告显存、吞吐和 needle 检索结果，用于定位 hybrid 开始节省资源的长度，并量化状态压缩带来的检索损失。

本课术语：

| 术语 | 简要解释 |
|---|---|
| SSM（状态空间模型） | 不存历史原文，只维护一个固定大小的状态向量，每读一个 token 更新一次 |
| selective（选择性） | 状态怎么更新由当前输入决定：重要的多写进小抄，废话就多保留旧内容 |
| Mamba-2 | 本课采用的 selective SSM 实现，带官方 CUDA kernel |
| SSD（状态空间对偶） | 把 SSM 计算改写成分块矩阵乘法，吃满 GPU 最快的矩阵乘单元 |
| KV cache | 注意力的笔记本：存下所有历史 token 的 key/value，随长度线性涨 |
| GQA | 分组查询注意力：多个 query head 共用一组 K/V，笔记本变薄，机制不变 |
| hybrid pattern | 8 层里哪几层用注意力、哪几层用 Mamba 的排布，如 `M M M A M M M A` |
| packing / cu_seqlens | 把多条短样本拼进同一条长张量省算力；`cu_seqlens` 记录每条样本的边界 |
| crossover length | 序列长过这个值，hybrid 才在延迟或显存上真正反超纯注意力 |
| KD（知识蒸馏） | 让学生模型拟合教师模型的输出分布；12B 用它帮换过心脏的模型恢复 |

## 2. 比较范围与需要验证的结论

先把最容易翻车的一件事说清楚："换架构好不好"其实是两个不同的问题，混着答，答出来的必然是错的。本课拆成两个实验。

**12A 是从相同随机 Thinker 起点开始的 iso-parameter 架构比较**（iso-parameter：参数配平，所有臂的总参数拉到同一水平，赢了才算架构的功劳）。三臂都不加载官方 checkpoint 的 Thinker mixer 和 FFN，只复用 tokenizer、embedding、LM head、模态 encoder/connector 和数据 contract。B/C 使用较窄 FFN，把三种 Thinker backbone 的总参数控制在同一范围。12A 只能回答教学规模下哪种架构更合适，不能写成"把官方 MiniMind-O checkpoint 升级成了 Mamba"。

**12B 是从官方 checkpoint 开始的实用迁移。** 三臂保留原来的 2432-width FFN；控制臂继续使用原 Attention，两个迁移臂只在预注册层位用 Mamba-2 替换 Attention。三臂使用同一批蒸馏/监督数据和相同更新数。12B 不做总参数配平，必须单独报告新增参数和迁移训练成本；它回答的是"现有 checkpoint 能否迁移"，不是纯架构因果问题。

两项实验分别检验：

> 12A：在 Thinker backbone 总参数匹配、训练 token 和长度分布相同的条件下，
> hybrid 能否降低长序列 cache/HBM 或提高吞吐，同时保住内容寻址与多模态绑定。
>
> 12B：从同一个官方 checkpoint 出发，使用同一迁移目标和 token 预算后，hybrid
> 能否达到预注册的质量保留率，并在长序列上获得可测的系统收益。

12A 的 FFN 宽度随 arm 改变，所以差异只能归因于整套配平后的 backbone design，不能全部归因于 mixer。12B 的参数量和初始化恢复难度不相同，也不能用来宣称 Mamba mixer 的纯因果收益。

以下结果不能支持对应结论：

- 12A 某臂使用了更多参数或训练 token；
- 把 12A 写成 checkpoint 升级；
- 12B 隐藏新增参数或蒸馏成本；
- 复杂度更低但实测 kernel 更慢；
- needle/copy 提升但真实 QA 退化；
- packed sample 之间发生状态串扰；
- SSM 没有收到图像二维位置或音频时间；
- 短序列吞吐或生成延迟明显变差。

最终报告需要给出"质量—吞吐—上下文长度—cache"Pareto（多指标下不被任何对手全面压制的配置集合），并标明 hybrid 开始省时或省内存的上下文长度。若没有出现 crossover，也要按实测结果报告。

## 3. 进入条件与独立起点

preflight（正式实验前一次性完成的检查与落锁程序）通过之前不开工：

- 下述官方 dense checkpoint 与 MiniMind-O 模态 token 注入路径已通过 preflight 和 golden cases。12A 只复用固定的非 Thinker 资产；12B 才把它作为迁移起点。
- CUDA 环境可编译/安装官方 `state-spaces/mamba` selective scan kernel。
- 固定 tokenizer、数据 manifest、训练 token、context length buckets。
- 能分别测 prefill 和单 token decode；能报告 KV cache 与 SSM state bytes。
- 至少有 2K/8K/32K 的合成检索集；12A、12B 都不越过 32K。
- 本课关闭 MoE。12A 的随机 Thinker 初始化和 12B 的 exact checkpoint 迁移分别写入不同 run lock，结果不得混算。

### 3.1 Base model 与 Mamba 实现锁

正式实验不接受 checkpoint 别名、浮动分支或未记录的 pip wheel。preflight 固定两份上游：

```yaml
# manifests/sources.lock.json
base_model:
  code_git_url: https://github.com/jingyaogong/minimind-o.git
  code_git_commit: a10fa6c148ed274d66f96dc119689e93e01be823
  checkpoint_repo: jingyaogong/minimind-3o
  checkpoint_revision: ee3febbd08cc5b2bd41c039c825a8934232fee33
  checkpoint_file: pytorch_model.bin
  checkpoint_sha256: 21530f9bbc540f461e2c0e29292ad359781d4d984d1e0c994510945f9b0edaab
  tokenizer_revision: ee3febbd08cc5b2bd41c039c825a8934232fee33
  required_source_config:
    hidden_size: 768
    thinker_layers: 8
    talker_layers: 4
    intermediate_size: 2432
    max_position_embeddings: 32768
    use_moe: false
mamba2:
  git_url: https://github.com/state-spaces/mamba.git
  git_commit: e9594ce1c732d97440f0332fdc43170a2294dbfa
  class: mamba_ssm.modules.mamba2.Mamba2
```

preflight 用完整 commit/revision 获取源码和 checkpoint，重算 `pytorch_model.bin` 的 SHA-256，逐字段检查 source config，并记录 Mamba、causal-conv1d、CUDA、PyTorch 的 file-tree/package hash。随后写只读 `sources.lock.json`、`position_contract.lock.json`、`run_12a.lock.json` 与 `run_12b.lock.json`；训练阶段禁止重新联网解析版本。这套流程和第 01 课冻结环境是同一门手艺，只是这次要锁的上游多了一个 kernel 仓库——kernel 版本不同，速度结论就不可比。

## 4. 完成本课需要掌握的操作

1. 理解 selective SSM、scan/recurrent duality（同一组参数的两条计算路径）和 SSD；
2. 区分 Attention 的内容寻址与 SSM 的压缩状态；
3. 区分 12A 的从头架构比较与 12B 的 checkpoint 迁移；
4. 实现可配置 block pattern，并对 12A 的 Thinker 总参数做公平控制；
5. 用 `seq_idx/cu_seqlens` 隔开 packed samples，正确重置 Mamba state；
6. 让图像二维位置和音频时间通过同一输入位置 contract 进入所有 mixer；
7. 正确管理 Mamba conv/SSM inference state；
8. 实测 prefill、decode 与 cache，并把结果和复杂度分析分开；
9. 判断哪些多模态任务仍然需要 attention 层。

## 5. 原理:边造边讲

四个机制，每个按同一节奏走：为什么需要（直觉）、怎么运转（机制）、精确定义（数学）、在源码哪里（代码）、怎么证明做对了（验证）。

### 5.1 Attention：开卷翻笔记，精确但越来越贵

注意力强在内容寻址：拿当前问题（query）去和笔记本里每一条记录（key）比对相关度，按相关度把对应内容（value）加权取出来。"把第 517 个 token 原样抄出来"、"把这张图左上角的物体和三段之后的代词绑在一起"，这类活它天生擅长——任何历史位置都能被直接照亮。但翻笔记的动作本身不便宜，而且笔记一条都不能扔。

Attention 用当前 query 与所有历史 key 计算相关度，再对 value 加权求和，因此适合复制、精确关联和跨模态绑定。训练和 prefill 的计算量随序列长度近似二次增长；自回归生成还要保存随历史长度线性增长的 KV cache。GQA 通过让多个 query head 共享 K/V heads 来压缩 cache——笔记本变薄，机制不变。

每 token 的 KV cache 字节数约为

$$
B_{\text{KV/token}}
\approx
2L_{\text{attn}}n_{\text{KV-head}}d_{\text{head}}b_{\text{element}}
$$

其中 $L_{\text{attn}}$ 是注意力层数，$n_{\text{KV-head}}$ 是共享后的 K/V head 数，$d_{\text{head}}$ 是每 head 维度，$b_{\text{element}}$ 是每个元素的字节数，系数 2 对应 K 和 V 各存一份。注意这个式子只对 $L_{\text{attn}}$ 求和：把 8 层里的 6 层换成 Mamba，cache 直接砍到四分之一，这就是 hybrid 省显存的算术来源。

`model/model_minimind.py::Attention`，带 Q/K norm 的 GQA（第 6 节有完整定位表）。

用上面的公式手算 32K 序列的 KV 字节数，与步骤 10 实测的 cache 占用对账；对不上就是实现或统计口径有错。另外别指望 GQA 解决长序列——它只改常数项，复杂度阶数不变，32K sweep 会给出证据。

**类比失效处。** "翻笔记"听起来是逐条精确查，attention 实际是对全部 value 的软性加权平均，"翻到哪条"是可微的分布；并且 8 个 query head 并行翻 4 组共享笔记（GQA），压根没有"一个人一本"这回事。

### 5.2 Selective SSM：一张边听边改的小抄

换个极端：闭卷，只许带一张固定大小的小抄。每读一个 token，就决定小抄上哪些旧内容淡一点、腾出的地方写什么新内容；答题时只看小抄。存储和每步计算都与历史长度无关——这就是 SSM。早期 SSM（线性时不变，LTI）的毛病是写小抄的规则是死的：同一套保留和写入系数对所有 token 一视同仁，"下面是重点"和一句语气词享受同等待遇。Mamba 的 selective 机制让这些系数全部由当前输入现算：模型自己学会"这个 token 值得记，多写；那个是废话，让旧状态多留一会儿"。

**类比失效处，两条都要记住。** 第一，真小抄是追加式的，写满为止；SSM 每一步都把整个旧状态乘上一个衰减量，像全张小抄同时变淡再叠新字，没有"单独擦掉某一条"的操作。第二，"小抄容量"不能按条数直觉理解：状态是连续向量，能叠加存下远超"几条记录"的信息，但读取是有损的——它擅长回答"前文大致说了什么"，很难保证"第 3000 个 token 恰好是哪个词"。后面步骤 11 的内容寻址负对照，就是专门去戳这第二条软肋的。

训练时用 parallel scan 一次算完整段序列（像把整本卷子摊开批改），生成时用 recurrent state 逐 token 更新（像逐题作答）。同一组参数、两条计算路径，数学上等价，数值上不完全等价。state 大小不随历史长度增长，但它把历史压缩到固定空间，因此不保证任意位置的精确检索。

Mamba 用输入依赖的参数选择性更新状态：

$$
\begin{aligned}
\mathbf h_t
&= A(\mathbf x_t)\mathbf h_{t-1}
 + B(\mathbf x_t)\mathbf x_t, \\
\mathbf y_t
&= C(\mathbf x_t)\mathbf h_t + D\mathbf x_t.
\end{aligned}
$$

对着小抄读：$\mathbf h_t$ 是小抄本身；$A(\mathbf x_t)$ 决定旧内容保留多少，$B(\mathbf x_t)$ 决定当前 token 写入多少，$C(\mathbf x_t)$ 决定答题时从小抄读出什么，$D\mathbf x_t$ 是绕过小抄的直连项。三个系数都带 $(\mathbf x_t)$，这就是 selective 的全部含义——LTI SSM 里它们是常量。

本课不手写 kernel，直接用 3.1 节锁定 commit 的 `mamba_ssm.modules.mamba2.Mamba2`；包装进步骤 2 的 `SequenceMixer` 统一接口。

必须同时运行 full-sequence 与 recurrent 两条路径，并比较输出误差（步骤 3）。两条路径不一致时，训练指标再漂亮也不可信——你训练的和你部署的是两个模型。

### 5.3 Mamba-2 与 SSD：让小抄更新跑上矩阵乘法快车道

复杂度是线性的，不代表实际就快。Mamba-1 的 selective scan 顺序性强，GPU 上最猛的硬件单元（专吃大矩阵乘的 tensor core）使不上劲。SSD 的核心发现是：一类 SSM 的整段计算可以改写成结构化矩阵乘法，让"小抄更新"也吃到矩阵乘的硬件红利。

State Space Duality（SSD）把这类 SSM 计算写成结构化矩阵乘法。Mamba-2 将序列分块：块内用矩阵乘法一次算完，块间只传递状态。

**数学与账本。** 复杂度对长度是线性的，但线性只有在 fused kernel 正常启用且序列足够长时，才可能转化为实际加速——常数项和 kernel 启动开销在短序列上完全可能让"线性"输给"二次"。因此必须记录 kernel fallback 次数和 crossover length，理论分析和实测速度分开报告（第 2 节已声明：复杂度更低但实测更慢，不算收益）。

同上 `mamba_ssm`；kernel 相关版本（mamba、causal-conv1d、CUDA、PyTorch）全部记录在 3.1 节的 lock 里。

步骤 10 的 2K/8K/32K sweep 给出实测曲线；第 14 节要求把 kernel fallback 计入系统指标。

### 5.4 Hybrid：小抄跑日常，留几页笔记本做精确检索

纯小抄丢精确检索，纯笔记本账单太贵。混合骨干的赌注是：大部分层用小抄处理日常语言建模，留少数注意力层负责精确查找和跨模态绑定，就能同时拿到两头的大头。

这几项配比全部要显式写下，没有默认值：

- attention 比例；
- attention 所在深度；
- 连续 Mamba 层最长跨度；
- 是否保留首层/末层 attention；
- 多模态边界附近是否需要 attention。

本课只比较固定规律 pattern，不做大规模 architecture search。

没有新公式：hybrid 的 cache 账就是 5.1 的公式只对留下的 attention 层数求和，再加每个 Mamba 层一份固定大小的 conv/SSM state——后者与序列长度无关，是常数。

pattern 写在配置的 `mixer_pattern` 字段（第 12 节），块结构见第 7 节的 HybridBlock。

留几层、摆在哪，由第 11 节的三臂对照和步骤 11 的内容寻址负对照回答，不靠直觉拍板。

## 6. MiniMind-O 当前机制与代码落点

动刀之前，先看清原装结构里哪些隐含假设会被异构 mixer 打破：

- `model/model_minimind.py`
  - `Attention` 是带 Q/K norm 的 GQA；
  - `MiniMindBlock` 固定 `self_attn + dense/MoE FFN`；
  - `MiniMindModel.layers` 全部同构；
  - `precompute_freqs_cis()` 与 `apply_rotary_pos_emb()` 只服务 Attention；
  - `past_key_values` 假设每层都是 `(K,V)`。
- `model/model_omni.py`
  - Thinker 逐层调用同一 `MiniMindBlock`；
  - `bridge_layer` 抽取 Thinker 中间状态供 Talker；
  - `past_key_values` 把 Thinker/Talker cache 拼在一个列表；
  - 图像/音频 embedding 在 Thinker 第一层前注入。
- `trainer/train_sft_omni.py`
  - 不区分 mixer 类型、没有 sequence-length bucket 指标；
  - checkpoint 没有 architecture manifest。

改造时把 block mixer、cache 类型、位置输入和 bridge 层语义写入显式接口。Mamba 层使用 `MambaCache`，不能用空 K/V 占位，否则保存、恢复和逐 token 推理都会产生歧义。

## 7. 目标架构

设计思路是"零件互换"：block 的外壳（norm、residual、FFN）不动，只把序列混合器做成可插拔。token/media embedding 先与共享的 `InputPositionEncoder(position_contract)` 输出相加。HybridBlock 经过 RMSNorm 后，按固定 layer pattern 调用 `GQAAttentionMixer(position, kv_cache)` 或 `Mamba2Mixer(ssm_state, conv_state)`。两种 mixer 都返回同宽 hidden，再依次经过 residual、RMSNorm、Dense SwiGLU（带门控的 FFN 结构）和第二个 residual。

统一 cache：

```python
LayerCache = Union[
    AttentionCache(k, v),
    MambaCache(conv_state, ssm_state)
]
```

### 7.1 所有 mixer 共用的输入位置 contract

先说为什么要有这一节。Attention 的位置信息走 RoPE（旋转位置编码，注意力靠它感知相对位置）这扇侧门进模型，Mamba 根本不调用 RoPE。不修门就直接换层，图像的二维坐标和音频的物理时间就只有 attention 层看得见，Mamba 层两眼一抹黑——这不叫架构差距，叫喂料不公。解决办法是开一扇所有 mixer 共用的正门：把位置信息编码后直接加在 content embedding 上。

packer 必须为每个 non-padding token 生成 `global_sequence_position`、`position_in_sample`、`modality_id`、`segment_id`、`image_y`、`image_x`、`image_xy_valid`、`time_bucket` 和 `time_valid`。

`image_y/image_x` 使用归一化原图坐标；非图像 token 的值为 0 且 `image_xy_valid=false`。`time_bucket` 由 source-absolute 毫秒按锁定桶宽换算；没有物理时间的 token 设为 0 且 `time_valid=false`。正式位置编码器固定为：

| 分量 | 宽度 | 生效条件 |
|---|---:|---|
| `sincos(position_in_sample)` | 64 | 所有 token |
| `sincos(image_y,image_x)` | 128 | `image_xy_valid` |
| `sincos(time_bucket)` | 64 | `time_valid` |
| modality embedding | 64 | 所有 token |
| `sincos(segment_id)` | 64 | 所有 token |

五个分量拼成 384 维向量，再经过 `Linear(384,D)` 和 scalar gate（一个标量门，控制位置信号注入强度），得到 `position_embed[D]`。

12A 的 linear 使用锁定 seed 的 `Normal(0,0.02)`，gate 固定从 1 开始。12B 为了在 step 0 保持 source 行为，linear 使用同一确定性初始化，三臂 gate 都从 0 开始；同一轨道 A/B/C 的 position encoder state-dict hash 完全相同。位置编码器执行：

```python
position_embed = position_encoder(
    position_in_sample,
    modality_id,
    segment_id,
    image_xy,
    image_xy_valid,
    time_bucket,
    time_valid,
)                                      # [T, D]
hidden = content_embed + position_embed # [T, D]
```

`position_encoder` 的结构、初始化、state-dict hash 和是否训练在 12A/12B 各自的所有臂中完全相同。Attention 的 RoPE/M-RoPE（多模态版 RoPE）ids 也由这份 contract 生成；Mamba 不使用 rotary phase，但从 `hidden` 读取相同的二维和时间信息。不得让 Attention 臂读取 `image_y/x`，却只给 Mamba 一个扁平序号。

packed batch 中，Attention rotary 与 Mamba position embedding 都使用 `position_in_sample`，不能使用跨样本连续增长的 packed offset。

单测固定 content embedding，只改变 `image_x`、`image_y` 或 `time_bucket`，确认三臂的第一层输入按同一规则变化；再把字段恢复，输出必须恢复。padding token 的 `position_embed` 必须为 0。

### 7.2 Packed sequence 与 Mamba state reset

packing 是把几条短样本首尾相接塞进一条长张量省算力。Attention 靠 block-diagonal mask 挡住跨样本偷看；SSM 的小抄却是一路写下来的，不在样本边界擦掉，上一条样本的内容就会渗进下一条——这是 hybrid 训练最隐蔽的坑，也是本课 parity 检查最重的一处。

训练 batch 使用下列边界 contract：

| 字段 | shape / dtype | 语义 |
|---|---|---|
| `hidden` | `[T,D]` | 所有有效 token 拼接 |
| `seq_idx` | `[T] int32` | 每个 token 所属的逻辑样本 |
| `cu_seqlens` | `[N+1] int32` | 每条样本在 packed tensor 中的边界 |
| `position_in_sample` | `[T]` | 每条样本都从 0 重新开始 |

Attention 使用由 `cu_seqlens` 生成的 block-diagonal causal mask。Mamba wrapper 必须在 `seq_idx` 改变时把 conv state 与 SSM state 清零。若锁定的 fused kernel 不能消费 `seq_idx/cu_seqlens`，先按 `cu_seqlens` 逐样本调用官方 kernel；不能把整条 packed tensor 当作一个序列。padding 不进入 `[T,D]` varlen（变长拼接）路径；使用 padded fallback 时，padding 位置不得更新 state，loss 也必须屏蔽。

训练时 state 不跨 batch 保留。自回归推理的 cache 按 request/sample id 隔离，新请求或新样本的第一个 token 必须从零 state 开始。

必须通过以下 parity（一致性校验）：

1. 单独运行样本 B，保存 logits 和最终 state；
2. 将 A、B packing 后运行，按 `cu_seqlens` 取出 B，结果与步骤 1 在预注册容差内一致；
3. 改成 B、A 顺序再测一次；
4. 在 A 与 B 之间加入 padding，B 的结果仍不变；
5. full-sequence 与逐 token recurrent 路径在每个样本内一致。

pattern 示例：

| 配置 | 八层 pattern |
|---|---|
| pure GQA | `A A A A A A A A` |
| 3:1 hybrid | `M M M A M M M A` |
| 7:1 hybrid | `M M M M M M M A` |

Talker 保持原纯 Attention。`bridge_layer` 用归一化后的相同深度输出，三臂索引一致。

### 7.3 12A：8-layer mapping 与公平随机初始化

12A 保持官方 Thinker 的 8 层深度，source→target 为 `0→0, …, 7→7`；不复制层、不插入第 9–24 层。三臂都原样加载 tokenizer、embedding、LM head、最终 RMSNorm、vision/audio projector 和完整 Talker，但不加载 Thinker block 内的 mixer、FFN 或 block norm。它们全部按下面规则重新初始化。

为什么这么苛刻？如果 A 臂继承八层预训练 Attention，而 B/C 的 Mamba 从随机开始，比出来的只是"谁的起跑线好"。所以 12A 三个 arm 都不加载任何 Thinker mixer 或 Thinker FFN tensor。preflight 按以下规则重新初始化：

- 对每层生成一个 2432-channel SwiGLU bank，`Normal(0,0.02)`；A/B/C 分别取前 `2432/1792/1728` channels。
- Attention 使用 MiniMind GQA 结构和按层固定种子；B/C 保留的 Attention 与 A 同层 tensor hash 完全一致。
- Mamba-2 使用锁定 commit 的官方 constructor 和按层固定种子；B/C 同层 Mamba tensor hash 完全一致。`dt_bias/A_log/D` 保留官方专用初始化，不改成普通正态。
- seed 为 `uint64(big_endian(SHA256("exp12-init-v1:{module}:{layer}")[:8]))`，与 Python 模块创建顺序无关。
- 三臂 bridge 都固定取 target layer 3 的 post-block、pre-final-norm hidden；Talker 不改变。

初始化结果分别保存为 `manifests/12a_init_{a,b,c}.safetensors`，逐 tensor hash 与 source 中实际加载/丢弃的 key allowlist 写入 `architecture_12a.lock.json`。出现 allowlist 之外的 missing/unexpected key 必须终止。12A 报告标题必须包含 `random-thinker-architecture-comparison`。

### 7.4 12B：官方 checkpoint 的实用迁移

12B 的 A 臂 strict-load（严格模式加载：键不匹配就报错）完整官方 Thinker。B/C 同样 strict-load 后，只替换 pattern 指定的 Attention mixer；所有 FFN、norm、embedding、projector、LM head 和 Talker 保留 source tensor。Mamba 使用锁定官方初始化，新增参数单列。

换过心脏的模型一开始必然虚弱，恢复靠蒸馏：让学生对着换心前的自己（冻结的 source teacher）学。三臂都读取同一批监督 target 和冻结 source teacher logits，使用相同目标：

$$
\mathcal L_{\text{12B}}
=
\mathcal L_{\text{LM}}
+\lambda_{\text{KD}}\tau^2
\operatorname{KL}\!\left(
p_{\text{source}}^\tau
\;\|\;
p_{\text{student}}^\tau
\right).
$$

其中 $p_{\text{source}}^\tau=\operatorname{softmax}(z_{\text{source}}/\tau)$，$p_{\text{student}}^\tau=\operatorname{softmax}(z_{\text{student}}/\tau)$。温度 $\tau$ 把 teacher 分布"调软"，让第二名、第三名候选的概率也携带监督信息；KL 的方向固定为冻结 source teacher 到 student，也就是让 student 拟合 teacher 分布。前面的 $\tau^2$ 用来补偿温度造成的梯度缩小。若实现中不乘 $\tau^2$，必须把它写成另一套目标并重新锁定 `lambda_kd`，不能与本课结果混用。

`lambda_kd`、温度、蒸馏 token 范围、训练 token 和 update 数在第一次训练前锁定。A 也执行完全相同的迁移训练，不能让 A 直接使用未经训练的 source 分数与训练后的 B/C 比较。三臂都训练完整 Thinker 与共享 position encoder，冻结 modality encoders 和 Talker；B/C 因新增 Mamba 而拥有更多可训练参数，必须如实报告。12B 保留原 FFN 宽度，因此不满足 12A 的总参数配平；报告必须同时给出 source、A、B、C 的总参数、可训练参数、训练 GPU-hours 和最终系统指标。

## 8. 公平匹配方法

架构比较最常见的作弊方式有两种：参数不对等（大的赢）和数据不对等（吃得多的赢）。这一节把两条路都堵死。

### 8.1 主控制

两项实验共同固定 tokenizer、样本顺序、context bucket、`d_model`、输入/输出 embedding、位置 contract、optimizer、LR schedule、global token batch、seed 和有效训练 token。所有臂都使用同一 packed boundaries；不能为 Mamba 单独减少长样本或关闭多模态位置。

12A 额外要求：

- Thinker backbone 总参数误差 ≤2%；
- 三臂 Thinker block 都从锁定随机初始化开始；
- FFN 只使用 8.2 预注册的固定宽度；
- 不强行让理论 FLOPs 相同，但必须实测每 token FLOPs/时间。

12B 额外要求：

- 三臂从同一 source checkpoint 出发；
- FFN 保持 source 的 2432 width，未替换 tensor hash 相同；
- 蒸馏 teacher、LM/KD loss、训练 token、update 和可训练范围规则相同；
- 不宣称参数匹配，逐臂报告新增参数、总参数和迁移训练成本。

### 8.2 参数配平

本节只适用于 12A。Mamba mixer 比 GQA mixer 参数多，直接换层总参数就超了；配平的手段是把 B/C 的 FFN 收窄。三臂固定 `d_model=768`、8 blocks 和下表宽度；preflight 只复算、验证并落锁，不允许运行时搜索 FFN width：

| 臂 | 8-layer pattern | FFN width/层 | mixer params | FFN params | mixer+FFN 小计 | 相对 A |
|---|---|---:|---:|---:|---:|---:|
| A | `A A A A A A A A` | 2432 | 14,157,312 | 44,826,624 | 58,983,936 | 0 |
| B | `M M M A M M M A` | 1792 | 25,532,976 | 33,030,144 | 58,563,120 | -0.71% |
| C | `M M M M M M M A` | 1728 | 27,428,920 | 31,850,496 | 59,279,416 | +0.50% |

计数对应与 source/Talker 兼容的 GQA `heads=8, kv_heads=4, head_dim=96`，以及 Mamba-2 `d_state=64, d_conv=4, expand=2, headdim=64, ngroups=1, D_has_hdim=false, rmsnorm=true, bias=false, conv_bias=true`。若实例化后的精确计数与表格不符，preflight 失败；不得静默改宽度。不允许只给某几层加隐蔽 adapter。

正式匹配口径为

$$
P_{\text{Thinker-backbone}}
= P_{\text{mixer}} + P_{\text{FFN}} + P_{\text{norm}}.
$$

表中只列会随 arm 变化的 mixer 与 FFN 小计；结构相同的 block norm 也必须由脚本计入 `thinker_backbone_total_params`。embedding、LM head、connector 和 Talker 不进入这一匹配口径，因为三臂逐 tensor 相同。报告不得把 FFN 宽度变化藏在"mixer 对比"这个名字下面。

配置生成器分别输出 `embedding_params`、`mixer_params`、`ffn_params`、`norm_params`、`thinker_backbone_total_params`、`fixed_non_backbone_params` 和 `trainable_params`。

### 8.3 Token 公平

训练预算按有效 token 计算。有效 token 指 non-padding、进入 loss 或上下文的 token。12A 和 12B 各自在自己的三臂内保持每个 update 的 global token batch 相同，并在累计 token 达到同一阈值时停止；不能只比较 step 数，也不能把 12A 与 12B 的 token 合并成一个胜负表。

## 9. 数据 recipe

### 9.1 Schema

```yaml
id: seq_000001
messages: [...]
modalities:
  - type: text
    span: [0, 4000]
  - type: image
    span: [4000, 4064]
source: synthetic_retrieval
license:
  content: CC0-1.0
  media: none
  redistributable: true
split: train
sequence:
  effective_tokens: 8192
  bucket: 8k
  answer_evidence_spans: [[517, 530], [7900, 7912]]
packing:
  source_item_id: seq_000001
  position_resets_at_start: true
task: multi_needle
```

`answer_evidence_spans` 记录证据在序列中的位置，后面按"证据在前/中/后"分桶评测全靠它。JSONL 不预先写死 batch 内的 `seq_idx`；manifest compiler 按实际 packing 顺序生成 `seq_idx/cu_seqlens`，同时保存每个 source item 的 token 起止位置。编译器还要输出第 7.1 节的位置字段，并验证拆包后每条样本的 `position_in_sample` 从 0 开始。

### 9.2 Mixture

- 55% 普通 text LM/instruction tokens；
- 15% copy/associative recall/multi-query retrieval；
- 10% long document QA；
- 8% image/text；
- 7% audio/text；
- 5% video/text 或 interleaved synthetic multimodal。

所有臂使用相同 modality token accounting。媒体与 annotation license 分开；合成长序列优先使用自生成 CC0 内容，避免不可追踪语料。

### 9.3 长度课程

standard 长度课程按训练 token 分配：40% 为 2K、35% 为 8K、20% 为 16K、5% 为 32K。

评测固定 2K/8K/32K。训练不因某臂更快而额外增加长序列 token。

### 9.4 切分

- 文档 identity、合成 key/value space、模板 family 分组切分；
- needle 的值域和位置分布 train/test 分离；
- 多模态媒体 identity 不跨 split；
- golden 200 cases 永不参与 architecture 选择。

### 9.5 三档规模

| 档位 | 模型 | 每臂训练 token | 目的 |
|---|---:|---:|---|
| common pilot | 80–120M | ≤100M | position、packing、kernel、cache parity |
| 12A standard | 100–150M 随机 Thinker | 1–3B | iso-parameter 架构比较 |
| 12B migration | 官方 checkpoint | 0.5–2B | LM+KD 实用迁移 |
| full | 300M–1B | 10B+ | 另开深度扩展，不与 12A/12B 主结论混算 |

## 10. 实施步骤

顺序有讲究：先修门（位置与边界），再换心（mixer），最后才谈成绩。前一步的 parity 不过，后一步的一切数字都是废纸。

### 步骤 1：先实现位置与 packed-boundary contract

让 packer 输出第 7.1、7.2 节定义的 position 字段、`seq_idx` 和 `cu_seqlens`。先用两条短文本、一张图和一段带时间戳音频验证位置，再运行 A+B、B+A 和插入 padding 三组 packing parity。未通过时不要接 Mamba kernel。

### 步骤 2：把 mixer 抽象出来

新增统一 `SequenceMixer.forward(hidden, position_contract, seq_idx, cu_seqlens, cache)`。先用 Attention wrapper 替换旧调用，断言 logits 与旧 baseline 一致——重构不改行为，是第 01 课 trace 零扰动的同款纪律。

### 步骤 3：接入官方 Mamba-2 kernel

使用官方 `state-spaces/mamba` 实现；先跑 CPU/小 CUDA tensor shape test，再比较 full-sequence 和 step-by-step recurrent 输出。

### 步骤 4：实现异构 cache 与每样本 state reset

保存/恢复 Attention K/V、Mamba conv state、Mamba SSM state。训练 wrapper 在每个 `cu_seqlens` 边界清零 state，推理 cache 按 request id 隔离。验证一次性 forward 与逐 token decode、单独 B 与 packed A+B 的结果都在容差内一致；beam/search 若未支持 cache reorder，必须显式禁用。

### 步骤 5：生成 12A 的三套总参数匹配配置

按 8.2 的三个固定宽度物化完整配置，复算 `thinker_backbone_total_params`；误差超过 2% 或与锁定计数不符就终止，禁止自动调整 FFN intermediate。再用 profiler 记录真实 FLOPs。

### 步骤 6：运行 12A 从头架构实验

先用 100M tokens 纯文本检查 loss、梯度、吞吐、copy 和 needle，再接入相同的多模态位置 contract。任何臂出现 NaN、scan/recurrent 不一致、packed sample 串扰或数据顺序差异时先修复。正式结果只写 `random-thinker-architecture-comparison`，不使用"官方 checkpoint 升级"措辞。

### 步骤 7：运行 12B checkpoint 迁移

从同一 source checkpoint 物化 A/B/C。A 保持 Attention，B/C 只按 pattern 替换 mixer；FFN width 都保持 2432。三臂读取同一份 teacher logits，按第 7.4 节目标训练相同 token。保存新增参数、未替换 tensor hash、KD/LM loss 和 GPU-hours。

### 步骤 8：验证多模态位置确实进入两类 mixer

固定 content embedding，分别改变图像 `x/y` 和音频 `time_bucket`。Attention 与 Mamba 第一层输入都必须变化；把位置恢复后输出也要恢复。再对 text/image/audio/video 分别做 packed A+B parity。只打印 metadata、却没有进入 `position_encoder` 的实现不合格。

### 步骤 9：运行统一训练与 checkpoint

12A、12B 分别按累计有效 token 同步停止；每 50M token 只跑统一 dev，保存相同 milestone checkpoint。最终 test 在配置和 checkpoint 选择冻结后只运行一次。

### 步骤 10：长序列系统 sweep

在 2K/8K/32K 上分别测 prefill、decode、cache 和 HBM（显卡的高带宽显存）。batch=1 与吞吐 batch 分开报告。crossover length 是 hybrid 首次在延迟或显存上优于各自 A 臂的长度；12A、12B 分开画图，不得只给最长序列结果。

### 步骤 11：内容寻址负对照

运行 exact copy、multiple needles、key collision（多个相似 key 互相干扰）、证据位置打乱。这些任务专戳 5.2 节讲的状态压缩软肋：hybrid 若只记住局部统计而不能精确检索，应在报告中暴露，而非被平均分掩盖。

## 11. 对照实验组

### 11.1 12A：随机 Thinker 的 iso-parameter 架构比较

| 臂 | Pattern | FFN width | 起点 | 目标 |
|---|---|---:|---|---|
| 12A-A | `A A A A A A A A` | 2432 | 锁定随机 Thinker | 质量 reference |
| 12A-B | `M M M A M M M A` | 1792 | 锁定随机 Thinker | 固定 3:1 hybrid |
| 12A-C | `M M M M M M M A` | 1728 | 锁定随机 Thinker | 固定 7:1 hybrid |

12A 三臂 Thinker backbone 总参数误差 ≤2%，训练 token 严格相同。Attention 层位置固定为每组末层。12A 不加载 source Thinker block，也不和 source checkpoint 质量直接做"升级成功"比较。

### 11.2 12B：官方 checkpoint 的实用迁移

| 臂 | Pattern | FFN width | 起点 | 参数口径 |
|---|---|---:|---|---|
| 12B-A | `A A A A A A A A` | 2432 | 完整 source checkpoint | source 参数量 |
| 12B-B | `M M M A M M M A` | 2432 | source + 新 Mamba mixer | 新增参数单列 |
| 12B-C | `M M M M M M M A` | 2432 | source + 新 Mamba mixer | 新增参数单列 |

12B 保留未替换的 source tensor，不缩窄 FFN。三臂使用相同 LM+KD 目标、位置 contract、packed boundaries、数据顺序、训练 token 和 seed。12B 不满足总参数匹配，结论只能写成"在这套迁移 recipe 下的质量—成本结果"。

两项实验都不加入纯 Mamba 第四臂；可在第 22 节扩展题预研。12A 与 12B 使用不同 experiment ID、run lock、checkpoint 目录和统计表。

## 12. 配置与伪代码

```yaml
experiment: exp12a_architecture
track: 12A
arm: 12A-B
sources_lock: manifests/sources.lock.json
architecture_lock: manifests/architecture_12a.lock.json
position_contract_lock: manifests/position_contract.lock.json
run_lock: manifests/run_12a.lock.json
model:
  d_model: 768
  layers: 8
  mixer_pattern: [mamba2, mamba2, mamba2, attention, mamba2, mamba2, mamba2, attention]
  attention:
    heads: 8
    kv_heads: 4
    head_dim: 96
  mamba2:
    d_state: 64
    d_conv: 4
    expand: 2
    headdim: 64
    ngroups: 1
    D_has_hdim: false
    rmsnorm: true
    bias: false
    conv_bias: true
  ffn_intermediate: 1792
  use_moe: false
position_contract:
  input_fields: [position_in_sample, modality_id, segment_id, image_xy, image_xy_valid, time_bucket, time_valid]
  image_xy: normalized_original_coordinates
  time_origin: source_media_start
  time_bucket_ms: 80
  add_to_content_embedding: true
  shared_state_hash_across_arms: true
packing:
  format: varlen
  boundary_fields: [seq_idx, cu_seqlens]
  attention_mask: block_diagonal_causal
  mamba_reset_on_seq_change: true
  padding_updates_state: false
train:
  effective_tokens: 2_000_000_000
  global_tokens_per_step: 262144
  precision: bf16
  seeds: [17, 23, 41]
eval:
  context_lengths: [2048, 8192, 32768]

---
experiment: exp12b_checkpoint_migration
track: 12B
arm: 12B-B
sources_lock: manifests/sources.lock.json
source_checkpoint: jingyaogong/minimind-3o@ee3febbd08cc5b2bd41c039c825a8934232fee33
migration_lock: manifests/migration_12b.lock.json
position_contract_lock: manifests/position_contract.lock.json
run_lock: manifests/run_12b.lock.json
model:
  d_model: 768
  layers: 8
  mixer_pattern: [mamba2, mamba2, mamba2, attention, mamba2, mamba2, mamba2, attention]
  ffn_intermediate: 2432
  keep_unreplaced_source_tensors: true
packing:
  boundary_fields: [seq_idx, cu_seqlens]
  mamba_reset_on_seq_change: true
migration:
  objective: lm_plus_teacher_kl
  teacher: frozen_source_checkpoint
  lambda_kd: 0.5
  temperature: 2.0
  distill_tokens: assistant_nonpadding
  trainable: [thinker, position_encoder]
  frozen: [vision_encoder, audio_encoder, talker]
train:
  effective_tokens: 2_000_000_000
  global_tokens_per_step: 262144
  precision: bf16
  seeds: [17, 23, 41]
```

前向循环的骨架——注意两种 mixer 拿到的边界信息同源：

```python
position_embed, rotary_ids = position_encoder(position_contract)
h = content_embed + position_embed
for block, cache in zip(blocks, caches):
    if block.kind == "attention":
        h, cache = block(
            h,
            rotary=rotary_ids,
            cu_seqlens=cu_seqlens,
            kv_cache=cache,
        )
    else:
        h, cache = block(
            h,
            seq_idx=seq_idx,
            cu_seqlens=cu_seqlens,
            inference_state=cache,
            reset_state_on_seq_change=True,
        )
```

12B 的蒸馏目标，落到代码就是十来行：

```python
with torch.no_grad():
    teacher_logits = source_teacher(batch).logits

student_logits = lm_head(h)
mask = assistant_nonpadding_mask
log_p_student = F.log_softmax(student_logits[mask] / temperature, dim=-1)
p_teacher = F.softmax(teacher_logits[mask] / temperature, dim=-1)

# F.kl_div(input_log_probs, target_probs) 计算 KL(target || input)
kd_loss = F.kl_div(
    log_p_student,
    p_teacher,
    reduction="batchmean",
) * temperature**2
loss = lm_loss + lambda_kd * kd_loss
```

`lambda_kd=0.5` 和 `temperature=2.0` 只是课程默认值；可以在 12B pilot 的 dev 集上选择一次，随后写入 migration lock。A/B/C 必须使用同一组锁定值。

## 13. 训练预算与 8 卡执行

- **共同接口 pilot**：1×24GB，2K/8K、最多 100M tokens；先完成位置输入、packed state reset、full/recurrent parity，不做能力结论。
- **12A standard**：4×24–48GB，100–150M 级随机 Thinker、每臂 1–3B tokens。这是教学规模的从头架构比较，不是官方 checkpoint 升级。
- **12B standard**：4×24–48GB，从官方 checkpoint 迁移、每臂相同的 0.5–2B LM+KD tokens；source teacher 推理成本和训练 GPU-hours 单列。
- **8×24GB**：每臂 2 卡并行 + 余卡评测；32K 用 activation checkpointing（用重算换显存）、Flash Attention/官方 Mamba kernel。
- **8×48/80GB**：12A、12B 分别完成后再做深度扩展；1B 模型另开实验 ID，且用 FSDP/sequence parallel（下一课的主角，先混个脸熟）前先通过第 13 课的数值 gate。
- 编译 kernel 前记录 CUDA、PyTorch、causal-conv1d、mamba-ssm 版本；不同 kernel 不能直接比较系统结果。
- 混合架构的 pipeline stage 必须按实测层成本平衡，不能按层数平均切分——Mamba 层和 Attention 层的每层成本差异很大，这份实测数据第 13 课直接要用。

## 14. 指标与测量方法

### 质量

- validation NLL/PPL（困惑度）；copy exact；associative recall（联想召回：给 key 找配对的 value）；multi-needle accuracy；RULER 子任务（一套长上下文合成评测）；长文档 QA；图像/音频/视频回归；Talker 输出回归。
- 12A 只在随机 Thinker 三臂内比较。
- 12B 另外报告相对冻结 source checkpoint 的质量保留率、LM/KD loss 与恢复曲线。

### 系统

- prefill p50/p95；每 token decode latency；tokens/s；peak HBM；KV bytes/token；SSM+conv fixed state bytes/layer/batch；compile time；kernel fallback 次数。

### 分层报告

- context：2K/8K/32K；
- evidence position：前/中/后；
- needle 数量与 key collision；
- modality：text/image/audio/video；
- batch=1 与 throughput batch。

### 测量纪律

预热 20、正式 100；同步 CUDA；同 dtype/硬件/software；训练按有效 token 停止；3 seed；参数和 FLOPs 由脚本生成；报告 crossover context length。12A 与 12B 分别统计，不把两条轨道的均值放进同一显著性检验。

## 15. 验收条件

共同硬门槛：

1. 单独 B、packed A+B、packed B+A 和插入 padding 后的 B logits/state 都在预注册容差内一致；
2. 每个 `cu_seqlens` 边界都重置 Mamba conv/SSM state，训练 state 不跨 batch；
3. full-sequence 与 recurrent Mamba 输出通过数值容差；
4. 图像 `x/y` 和音频 `time_bucket` 通过同一个 position contract 进入所有臂，padding 的 position embedding 为 0；
5. 累计训练有效 token、packed boundaries 和样本顺序在同一轨道的三臂间一致；
6. 在 2K/8K/32K 上完整报告吞吐、峰值 HBM 和实际 cache bytes。

12A 额外门槛：

1. 三臂 `thinker_backbone_total_params` 在 ±2% 内；
2. 三臂 Thinker block 均未加载 source mixer/FFN/norm；
3. 报告明确标为随机 Thinker 架构比较，质量、内容寻址和多模态结果只相对 12A-A。

12B 额外门槛：

1. 三臂 source checkpoint、position contract、teacher、LM/KD 目标和训练 token 相同；
2. 未替换 source tensor hash 一致，B/C 的新增 Mamba 参数和 GPU-hours 完整报告；
3. 质量保留率与系统收益相对 12B-A 和冻结 source 分别报告；
4. 不使用"iso-parameter"或"纯 mixer 因果收益"描述 12B。

12B 的质量保留和两项实验的 HBM 收益目标要在各自 pilot 后、查看 test 前声明。本课的 canonical 结论使用下面一组固定判据。质量按同一 held-out case 配对；三个 seed 做 10,000 次两层 bootstrap（重采样统计：先重采样 seed，再重采样 case，估计结论对随机性的敏感度）。系统指标在同一机器上按相同 case 和重复轮次配对：

### 12A 胜出条件

在 dev 上按预注册顺序从 B/C 选择一个 hybrid，冻结后只在 test 运行一次。只有同时满足以下条件，才能写"随机 Thinker 中 hybrid 优于纯 Attention"：

1. 32K multi-needle/long-QA macro accuracy 相对 12A-A 的 95% CI（置信区间）下界不低于 -2 个百分点；
2. 2K 综合能力相对 12A-A 的 95% CI 下界不低于 -1 个百分点；
3. 32K peak HBM 至少降低 30%，且配对 95% CI 下界高于 20%；
4. 32K throughput 至少提高 20%，且配对 95% CI 下界高于 10%。

### 12B 胜出条件

只有同时满足以下条件，才能写"checkpoint 迁移后的 hybrid 是可用升级"：

1. 相对 12B-A，32K 内容寻址 macro accuracy 的 95% CI 下界不低于 -2 个百分点；
2. 相对冻结 source，2K text/image/audio macro accuracy 的 95% CI 下界不低于 -1 个百分点；
3. 相对 12B-A，32K peak HBM 至少降低 30%，且配对 95% CI 下界高于 20%；
4. 相对 12B-A，32K p95 prefill latency 至少降低 20%，且配对 95% CI 下界高于 10%。

任一质量非劣条件未通过时，即使系统更快，也只能报告质量—成本权衡。不同 kernel 或 GPU 需要新建系统实验 ID，不得沿用这组系统结论。case 列表、重复轮次、选择顺序和统计脚本 hash 必须在打开 test 前写入 `stats_plan.yaml`。

负结果合格条件：明确 crossover 不出现的原因是 kernel、模型过小、序列过短还是内容寻址损失，并保留公平数据。

## 16. 失败诊断表

调试从最便宜的检查开始：先查边界和 state，再怀疑架构本身。

| 现象 | 原因候选 | 修复 |
|---|---|---|
| Mamba 比 Attention 慢 | 序列短/kernel fallback | 检查 fused kernel 与 crossover |
| train 快、decode 错 | recurrent state 更新错误 | whole vs step parity |
| packed B 与单独 B 不一致 | A 的 state 泄漏进 B | 检查 `seq_idx/cu_seqlens` 与边界 reset |
| 加 padding 后结果变化 | padding 更新了 SSM state | 改 varlen 路径或显式屏蔽 state update |
| 图像空间题只在 Attention 臂正常 | Mamba 没收到二维位置 | trace `position_encoder` 输入与第一层 hidden |
| 音频时间题在 Mamba 层退化 | time bucket 只进入了 RoPE | 把时间字段接入共享 position contract |
| 12A 被写成 checkpoint 升级 | 实验身份混淆 | 改为随机 Thinker 架构结论 |
| 12B 看似更强但参数更多 | 未报告迁移参数/成本 | 补参数与 GPU-hours ledger |
| needle 位于序列尾部时明显退化 | 状态压缩/attention 太少 | 比较 3:1，分析位置 |
| 多模态绑定明显退化 | 连续 Mamba 层过多 | 检查 attention placement |
| 三臂 loss 不可比 | 参数/token/样本顺序不同 | 自动审计并重跑 |
| OOM 随 context 增长 | cache 未按 layer type 分配 | 异构 cache |
| checkpoint 恢复变差 | conv/SSM state 或 config 丢失 | architecture manifest |
| 训练 NaN | dt/A 参数或精度不稳 | 官方初始化、局部 FP32 |
| Talker 变化 | 错误替换 Talker block | Talker 固定纯 Attention |
| 理论省内存但实测无 | activation/FFN 占主导 | 分项 memory snapshot |

## 17. 逐 case 要求

至少 200 cases：50 exact copy/association、50 multi-needle、30 长文档、25 图像、20 音频、15 视频、10 Thinker→Talker。每 case 分别输出 12A、12B 三臂答案、证据位置、context 长度、token 数、正确性、prefill/decode/cache、attention 层 pattern；12B 另附 source 答案和质量保留。

错误标签增加 `packed_state_leak`、`position_contract_gap`、`migration_recovery_gap`；其余为 `state_compression_loss`、`content_addressing_failure`、`modality_binding_failure`、`cache_bug`、`kernel_no_speedup`、`reasoning_failure`。必须展示按证据位置和实验轨道分开的失败树。

## 18. 交付物

```text
exp12/
  configs/12a/{pure_gqa,hybrid_3to1,hybrid_7to1}.yaml
  configs/12b/{continue_gqa,migrate_3to1,migrate_7to1}.yaml
  manifests/sources.lock.json
  manifests/position_contract.lock.json
  manifests/architecture_12a.lock.json
  manifests/12a_init_{a,b,c}.safetensors
  manifests/migration_12b.lock.json
  manifests/{run_12a,run_12b}.lock.json
  manifests/{12a,12b}_model_parameter_report.json
  data/manifest.jsonl
  data/license_report.md
  checkpoints/{12a,12b}/
  metrics/{12a,12b}/{quality,long_context,system}.jsonl
  traces/{packed_parity,position_contract,cache_and_kernel}.jsonl
  plots/{12a,12b}_quality_throughput_context_cache.png
  cases/index.md
  reports/{12a_architecture,12b_migration}.md
```

两份报告都必须写明 crossover 所在的 context length、attention 比例对精确检索的影响、实际 cache bytes，以及需要更多 attention 的模态和任务。12A 明确写"随机 Thinker 架构比较"；12B 明确写新增参数、迁移成本和相对 source 的质量保留。

## 19. 复现清单

- [ ] exact dense checkpoint revision/SHA-256 已锁定；
- [ ] exact Mamba Git commit 已锁定；
- [ ] CUDA、PyTorch、kernel file-tree 与 package hash 已记录；
- [ ] 共享 position contract 已锁定，图像二维位置与音频时间进入所有 mixer；
- [ ] `seq_idx/cu_seqlens`、padding 和每样本 state reset 单测通过；
- [ ] 单独 B、packed A+B、packed B+A 的 logits/state parity 通过；
- [ ] full/recurrent parity 通过；
- [ ] cache 保存与恢复通过；
- [ ] 12A source/target 都是 8 层恒等 mapping；
- [ ] 12A 三臂 Thinker mixer/FFN/norm 都未加载 pretrained tensor；
- [ ] 12A FFN widths 固定为 `2432/1792/1728`，没有运行时宽度搜索；
- [ ] 12A `thinker_backbone_total_params` 精确计数通过；
- [ ] 12A 报告没有声称 checkpoint 升级；
- [ ] 12B 三臂从同一 source checkpoint 开始；
- [ ] 12B 未替换 source tensor hash 相同，FFN 都保持 2432；
- [ ] 12B 三臂 LM/KD 目标、teacher、token 和 update 相同；
- [ ] 12B 新增参数、总参数、teacher 成本和 GPU-hours 已报告；
- [ ] 每条轨道内部 tokenizer、数据顺序、有效训练 token 与长度 mixture 相同；
- [ ] Talker 未替换；
- [ ] 2K/8K/32K sweep 已完成；
- [ ] batch=1 与吞吐 batch 均已测量；
- [ ] 多模态回归、三个 seed 与逐 case 分析完整；
- [ ] Pareto 已输出。

## 20. 前沿对照与改造方向

这条"小抄换笔记本"的路线，公开文献里已经走完了从原理到生产的全程。[Mamba](https://arxiv.org/abs/2312.00752) 证明 selective 机制让 SSM 第一次在语言建模上与同规模 Transformer 正面掰手腕，并用 selective copying、induction head 这类合成任务精确定位了固定系数 SSM 的短板——和本课步骤 11 的负对照是同一套探针。[Mamba-2/SSD](https://arxiv.org/abs/2405.21060) 把一类 SSM 与一类 attention 统一进结构化矩阵框架，换来分块矩阵乘的快 kernel，这是本课敢在 24GB 卡上跑 32K 的底气。[Jamba](https://arxiv.org/abs/2403.19887) 把路线推到生产规模：attention 与 Mamba 层比例压到 1:7，再叠 MoE，支撑十万 token 级上下文；它的消融给出一个与本课直接相关的定性结论——纯 Mamba 模型在 in-context learning 的格式跟随上不稳，插回少量 attention 层就恢复，印证了"内容寻址是 attention 层保留的核心能力"。[Nemotron 3 Nano Omni](https://arxiv.org/abs/2604.24954) 则把 hybrid Mamba-2/Transformer/MoE 骨干直接用进 omni 模型，是第 18 课的主角。另一条路线仍然全 attention：[Qwen2.5-Omni 技术报告](https://arxiv.org/abs/2503.20215)（第 01 课引用过）的 Thinker 就是标准 Transformer 加长上下文工程。两条路线并存说明这是账单结构的取舍，谈不上谁淘汰谁——本课的 Pareto 图就是这笔账的缩小版。

规模差距一目了然：前沿 hybrid 是几十亿参数、万亿级 token，12A standard 是 100–150M 随机 Thinker、每臂 1–3B token，结论只在教学规模成立（Nemotron 论文的规模化结论不能外推到这里，第 21 节第 4 篇的阅读检查就是这件事）；上下文上 Jamba 级系统跑十万 token，本课封顶 32K，source 的 `max_position_embeddings=32768` 也卡在那里，越界要按第 22 节第 5 题另开实验。这些都是钱和卡的问题。机制上本课教的与前沿是同一套，一件不缺：packed 边界的 state reset、full/recurrent parity、异构 cache、共享位置 contract，任何规模都绕不开。真正可能追不平的是 kernel 效率——26M 到 150M 的模型、32K 以内的序列，crossover 可能根本测不出来；这不丢人，按第 15 节的负结果条款如实写，就是合格结论。

三个实验都遵守本课纪律：同轨道内 token、样本顺序、位置 contract 相同；新臂另开实验 ID；先过 parity 再谈结论。

1. **小抄容量扫描（d_state）。** 本课把 Mamba-2 锁在 `d_state=64`。复制 12A-C 配置三份，`model.mamba2.d_state` 分别设 32/64/128；d_state 变化会改 mixer 参数量，必须按 8.2 的口径重算 FFN 宽度并重新落锁，不能沿用 1728。预算：common pilot 档，三档各 ≤100M token（含 15% 检索任务），1×24GB 单档数小时、三档一天出头。预期：multi-needle 准确率随 d_state 单调上升，SSM state bytes 同步上升——这条曲线就是"小抄页数买检索力"的实测版。失败判据：三档 needle 准确率无差异，先怀疑任务太容易（加 key collision 和 needle 数）或 pilot token 太少，排除这两条再下结论。
2. **另一条省 cache 的路：滑动窗口注意力第四臂。** hybrid 靠"小抄 + 少量全量笔记"；另一条常见路线是每层都翻笔记、但只翻最近 W 页（sliding window attention）。给 12A 加 D 臂：8 层全 attention、窗口 W=1024，cache 封顶在 W。改动位置：`GQAAttentionMixer` 的 mask 生成加窗宽参数（Flash Attention 系 kernel 原生支持窗口），FFN 保持 2432（参数量与 A 臂相同，不用重新配平）。预算：pilot 先行（≤100M token，数小时），standard 每臂 1–3B token、4×24–48GB，与其他臂同规格。预期：32K 显存收益与 hybrid 同量级，但 multi-needle 在证据距离超过 W 时断崖式下跌，hybrid 则是缓降——两种省钱方案的失效形状不同，这正是要画出来的图。失败判据：D 臂在超窗距离的检索上不掉点，基本可断定 needle 数据的证据距离没超过 W，回去检查数据生成。
3. **bridge 取层实验：Talker 需要哪种表征。** 本课三臂 bridge 都固定取 layer 3——在 hybrid pattern 里恰好是 attention 层。在 12B-B 上另训一组，把 `model/model_omni.py` 的 `bridge_layer` 改为 2（Mamba 层），其余不动。预算：12B pilot 档（≤100M token 的迁移预演）两组，4×24GB 各数小时；评测用第 14 节的 Talker 输出回归加第 17 节的 10 个 Thinker→Talker case。预期：bridge 取 Mamba 层时 Talker 回归退化更明显，说明 Talker 依赖经过内容寻址绑定后的表征——这个结论直接影响第 18 课接 Nemotron 时的 bridge 选层；若无差异，同样记录在案。失败判据：两组连文本侧 dev NLL 都拉开差距，说明改动影响的是训练稳定性而非 bridge 语义，查初始化与学习率后重跑。

三条论文结论能在本课缩小版设置里验方向：

| 论文结论 | 缩小版对应实验 | 预期 |
|---|---|---|
| Mamba：selective 机制补上固定 SSM 做不了的 selective copying | 步骤 11 的 exact copy 与 key collision 负对照 | 部分可复现：含 attention 层的 hybrid 应保住 copy；若做第 22 节第 1 题的纯 Mamba 臂，copy 与 multi-needle 应可见退化，方向与论文一致 |
| Jamba：纯 Mamba 缺内容寻址，少量 attention 层即可恢复 | 12A-B（3:1）与 12A-C（7:1）在 32K multi-needle 上的差距 | 同方向可复现：attention 层越少、证据位置越刁钻，退化越明显；7:1 不掉点则先怀疑任务太容易 |
| Mamba-2：线性复杂度在长序列兑现为实际加速 | 步骤 10 的 2K/8K/32K sweep 与 crossover length | 方向可复现但不保证：教学规模加 32K 封顶，crossover 可能不出现；此时结论写"本设置未观察到"，而非"论文有错" |

## 21. 论文精读与问题

每篇带着能在自己实验里对答案的问题去读；答不上来就回去重读。

1. [Mamba](https://arxiv.org/abs/2312.00752)：精读 selective mechanism、hardware-aware scan。带着问题进去：5.2 节说 LTI SSM 对所有 token 一视同仁，论文用哪个合成任务暴露这一点？阅读检查：说明选择性参数相对固定 SSM 改变了什么——把 $A,B,C$ 各自从常量变成了什么的函数写出来，逐个对应小抄的哪个动作（保留/写入/读出）。
2. [Transformers are SSMs: Mamba-2 / SSD](https://arxiv.org/abs/2405.21060)：精读 SSD 与 chunk algorithm。带着问题：块内矩阵乘和块间状态传递的分界线在哪，块大小影响什么？阅读检查：画出 chunk 内矩阵计算和 chunk 间状态传递——步骤 3 的 full/recurrent parity 验证的正是这两条路径的等价性。
3. [Jamba](https://arxiv.org/abs/2403.19887)：精读 Mamba–Attention–MoE hybrid 与层比例。带着问题：它的 1:7 与本课 12A-C 的 7:1 是同一档配比，作者凭什么敢把 attention 压这么低？阅读检查：列出 Attention 层在 hybrid 中保留的能力，和步骤 11 的负对照清单逐条对照，缺的补进你的 case 集。
4. [Nemotron 3 Nano Omni](https://arxiv.org/abs/2604.24954)：精读 hybrid Mamba-2/Transformer/MoE backbone。带着问题：它在多模态输入边界附近怎么摆 attention，与 5.4 节的第五条结构参数对照。阅读检查：列出规模化结论不能直接外推到 300M 的条件。
5. [state-spaces/mamba 官方仓库](https://github.com/state-spaces/mamba)：精读 `Mamba2` 与 inference cache API。带着问题：conv_state 和 ssm_state 各存什么、各多大？对照第 7 节 `LayerCache` 的两个字段，步骤 4 的保存/恢复代码照这里写。
6. [MiniMind-O 官方仓库](https://github.com/jingyaogong/minimind-o)：精读 `Attention`、`MiniMindBlock`、`past_key_values`。阅读检查：列出异构 mixer 需要修改的 cache 和位置假设，与第 6 节的清单互相验证，每一条都在源码里指到行。

## 22. 扩展题

1. 加纯 Mamba-2 臂，但另开实验并重新做参数匹配。
2. 固定 3:1 比例，比较 attention 位于组首/组末/均匀散布。
3. 让视频边界附近动态启用 attention；需防止内容泄露。
4. 组合 MoE，但必须在第 11 课与本课结论稳定后做 2×2 因子实验。
5. 64K/128K 必须另开实验 ID：先选择并锁定位置扩展方法，再重新训练或校准，重做 full-sequence/recurrent parity 与位置外推评测；不能把越过 source `max_position_embeddings=32768` 的结果混入本课主结论。

到这课为止，Thinker 的心脏可以在纯注意力和 Mamba 混合之间切换，你手里握着同一预算下三种骨干的完整账本：质量—吞吐—上下文—cache 的 Pareto，以及 hybrid 开始回本的长度。下一个麻烦是规模本身：模型再大一点、序列再长一点，单卡既装不下也等不起。[第 13 课](13_distributed_8gpu.md)把训练摊到 8 张卡上——先证明并行没有偷偷改变优化语义，再谈吞吐；本课"先 parity、后性能"的纪律，在那里会原样重演一遍。
