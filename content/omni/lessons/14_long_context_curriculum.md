---
id: 14_long_context_curriculum
title: "长上下文课程"
summary: "按 8K→32K→128K 一步步把窗口拉长，能不能比一上来就 128K 更稳地学会“用上远处的跨模态证据”，还不把短上下文的本事忘掉？"
unit: backbone
play_tools: []
checkpoints:
  - "把三件事分开验证：窗口能跑、位置能外推、证据能被用上。"
  - "弄清 RoPE、YaRN 各管到哪，以及训练长度 provenance（这模型到底见过多长的序列）的边界。"
  - "构造防抄近路的长距离检索、跨段推理和模态 ablation 任务。"
  - "用相同训练 token 比 direct 和 progressive 两种扩窗课程，并检查短任务有没有退步。"
---

# 第 14 课：渐进式扩展到 128K 上下文

> 本课产出：在相同训练 token 预算下，对比直接扩展与渐进式扩展两种训练课程，并用证据删除实验判断模型是否真的使用了远距离的文本、图像和音频信息。视频只作为显式扩展。

## 1. 长上下文先解决训练顺序

 第四幕"换强心脏"已经走了三课：第 11 课给 FFN 换上 MoE 稀疏专家（参数变多、每个 token 的计算量不涨），第 12 课把大部分注意力层换成 Mamba-2（长序列不再每步全量回看），第 13 课把训练搬上 8 卡、并证明数值没有变味。心脏换到现在，还剩最后一面墙：上下文窗口。模型一次能看进去的 token 数还停留在几千到 32K 的配置容量，而一场半小时的对话、一份长文档、一段带音频的长视频，动辄就是几万到十几万 token。窗口不够长，前三课换来的"算得省、算得快、装得下"就使不上力气。

 第一件是把窗口拉开：用 YaRN（一种按频率区别缩放位置编码的方法，第 6 节细讲）把 RoPE 改造到 128K，再把训练排成 8K→32K→128K 三级台阶，与"一步跨到 128K"做同预算对照。第二件更重要：证明模型真的在用远处的信息。窗口开得大和真的会用是两回事；一个 128K 的模型完全可能只靠看结尾几段把评测糊弄过去。所以本课给评测样本准备了反事实版本：把答案依赖的证据删掉，看模型会不会老实说"无法判断"。

直接把 `max_position_embeddings` 改大就开训，位置编码会在训练时没见过的位置上失灵（为什么会崩，第 6.1 节把直觉讲透），轻则 loss 尖峰，重则长文答非所问；而不做证据删除评测，你手里只有一个"能吃进 128K 输入"的模型，说不出它到底会不会用。往后看，第六幕的双工系统要维持长对话历史，第 9、10 课的视频 token 也只有在长窗口里才放得下。

**做完这课你能亲手做到：** 把一条 30 分钟的音频塞进模型，问"关键词 alpha 和 beta 谁先出现"；再把含 alpha 的那几秒删掉重问，看模型是否改口"无法判断"、原答案的概率是否明显下降。这套流程走通，第四幕收官；"现代内核"四件套（稀疏专家、混合序列层、多卡并行、长上下文）配齐。

本课术语：

| 术语 | 简要解释 |
|---|---|
| RoPE | 旋转位置编码：把每对通道当成一根按位置旋转的表针，注意力从表针夹角读出两个 token 隔多远 |
| 长度外推 | 让模型处理比训练时更长的输入；不加处理时，位置编码转进训练时没见过的角度区间，注意力读数失灵 |
| position interpolation | 位置插值：把大位置等比压回训练见过的范围；副作用是相邻 token 变得难分 |
| YaRN | 按频率区别对待的插值：高频通道少动、低频通道多压，论文还配一个注意力温度修正 |
| 渐进式课程 | 8K→32K→128K 分台阶训练，阶段切换按累计有效 token 计，不按 step |
| text anchor | 占训练 token 10% 的短文本回放，用来防止"学会长的、忘了短的" |
| needle / RULER | 长文里藏一根"针"再让模型找的检索测试；RULER 是把这类测试系统化的评测集 |
| evidence removal | 证据删除：删掉答案依赖的片段后重问，合格的模型应改答"无法判断" |
| non-padding token | 真正进模型的有效 token（不含 padding）；本课的预算、进度、吞吐全按它计 |
| CP（context parallelism） | 上下文并行：把同一条长序列切到多张卡上；本课主实验固定 CP=1，不用它 |

## 2. 本课要解决的问题

这是一门可以独立执行的长上下文训练课，不要求先完成第 11–13 课。默认主实验只使用 MiniMind-O 已有的文本、图像和音频入口。视频属于显式扩展，不能成为未声明的依赖。主实验采用 `torchrun + DDP + CP=1`：DDP（多卡数据并行：每张卡持有完整模型副本、各读各的数据、梯度做全局同步）下八张卡各持有完整模型副本，`CP=1` 表示不沿序列维切分。context parallelism（上下文并行，CP）只在主实验之外评估。

开始改配置前，先区分三个可独立失败的目标：

1. **窗口可运行**：模型和训练系统能处理目标长度，不 OOM（显存不够直接崩）；
2. **位置可外推**：位置编码在超过原训练长度后不崩；
3. **证据可使用**：模型能找到远处、跨段、跨模态的必要证据并完成推理。

拆成三项，是因为它们各有各的失败方式：第一项是系统工程问题，第二项是位置编码问题，第三项才是能力问题。混在一起，你就说不清 128K 的分数到底在测什么。

本课以第三项作为能力结论。修改 `max_position_embeddings=131072` 只说明配置允许更长输入。完成训练后，要分别检查原始输入是否答对、删除必要证据后是否正确回答"无法判断"，以及原答案概率是否下降。三项都使用各自有效的标准答案；不能继续用原答案给反事实输入打分。

## 3. 要验证的结论与失败条件

**研究问题：** 相比直接在 128K 上继续训练，`8K → 32K → 128K` 的渐进式课程能否减少训练不稳定，并改善长距离证据使用能力，同时把短上下文回退控制在 2% 内？

本课把占训练 token 10% 的短文本回放数据称为 `text anchor`。它用于检查模型学习长序列后是否忘记短文本能力。测试集中的样本不参加训练，下面称为未见测试集。

**预期结果：** 渐进式长度扩展加入 10% 短文本回放后，在相同训练 token 预算下，应当：

- 提高未见测试集上的长距离检索与跨段推理；
- 减少 loss 突然升高和梯度异常；
- 降低短文本、单图和短音频能力遗忘。

**拒绝标准：** 如果直接 128K 臂在相同 token、数据与优化预算下同时达到更高长上下文分数和相同回归水平，则渐进式课程没有获得支持。

## 4. 固定训练起点

推荐起点：

```text
checkpoints/exp14_start/
├── model.safetensors      # jingyaogong/minimind-3o
├── config.json            # max_position_embeddings=32768
├── tokenizer/
├── modality_connectors/
└── provenance.json
```

独立默认起点固定为：

| 字段 | 固定值 |
|---|---|
| model | `jingyaogong/minimind-3o` |
| revision | `ee3febbd08cc5b2bd41c039c825a8934232fee33` |
| code | `jingyaogong/minimind-o@a10fa6c148ed274d66f96dc119689e93e01be823` |
| hidden / layers | `768 / 8` |
| `rope_theta` | `1000000` |
| checkpoint config capacity | `32768` |
| reported pretraining context | `unknown` |
| 本课 scaling reference | `32768`，是预注册实验选择，不是历史训练长度声明 |

常见错误是把配置当历史：`max_position_embeddings=32768` 只证明配置容量，不证明 checkpoint 的权重确实在 32K 上训练过。本课将 32768 固定为三臂共同的 YaRN scaling reference，但必须把它标成实验选择，不能写成官方训练长度。若后来获得可引用的训练长度证据，必须新建 experiment ID 并重跑三臂，不能在中途替换。

起点必须满足：

- 文本 perplexity（困惑度：模型对文本的"惊讶程度"，越低越好）、单图问答、短音频 ASR 已有固定基线；
- 所有模态序列都能导出 `token_type`、`time_index` 和 `loss_mask`；
- KV cache 与 attention mask 有单元测试；
- checkpoint 没有混入本课的长上下文训练数据。

若从[第 13 课](13_distributed_8gpu.md)的 8 卡 checkpoint 进入，复制一份并写明 SHA。主课程始终用 DDP、`context_parallel_size=1`；128K 是否可行由本课的 50-step HBM（显卡上的高带宽显存）测试决定，不强制依赖第 13 课。若 HBM 测试不通过，正式结论止于 32K，并把 128K 记为未完成而非伪造结果。32K 结果只能作为工程部分结果，不能回答本课"8K→32K→128K 是否优于直接 128K"的主问题，也不能发布 `long-context-progressive-v1`。

### 4.1 记录训练长度的来源

在任何 RoPE patch 前先生成：

```yaml
checkpoint_id: jingyaogong/minimind-3o
checkpoint_revision: ee3febbd08cc5b2bd41c039c825a8934232fee33
checkpoint_files_sha256: artifacts/exp14/provenance/checkpoint.files.sha256
checkpoint_config_max_position: 32768
reported_pretraining_context:
  value: null
  status: not_disclosed
  evidence_url: null
rope_scaling_reference:
  value: 32768
  status: preregistered_experimental_choice
  selection_basis: checkpoint_config_capacity
claim_allowed: "本实验以 32768 作为 YaRN scaling reference"
claim_forbidden: "该 checkpoint 已由官方在 32768 token 上训练"
```

这里要防的是一个常见归因错误：配置上限、公开披露的训练长度和本实验选择的缩放基准是三个不同的事实，并不等价。请分别填写 `checkpoint_config_max_position`、`reported_pretraining_context` 与 `rope_scaling_reference`，并在 `provenance.json` 保存原始 `config.json`、模型文件 SHA-256、下载时间和证据 URL。验收时，任一字段缺失或被另一个字段代填，都视为 provenance（来源证据链）不合格。

### 4.2 正式 GPU 代码库需要的源码映射

先确认文档中的参数能抵达运行时——配置文件写得再漂亮，trainer 不消费它就等于没写。锁定 commit 的代码存在两项限制：

- `MiniMindConfig` 只接受布尔量 `inference_rope_scaling`；打开后使用硬编码的 `original_max_position_embeddings=2048`、`factor=16`；
- `train_sft_omni.py` 只把 hidden size、layer 数和 MoE 开关传入 `OmniConfig`，并使用 DDP；本文后面的 `position.*` YAML 不会自动生效。

因此正式 GPU companion repository 必须提供经过测试的独立源码改造，并把实际 diff 与 SHA-256 写入运行 provenance。改造至少完成以下源码映射：

```diff
--- a/model/model_minimind.py
+++ b/model/model_minimind.py
@@
-self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
-self.rope_scaling = {
-    "beta_fast": 32,
-    "beta_slow": 1,
-    "factor": 16,
-    "original_max_position_embeddings": 2048,
-    "attention_factor": 1.0,
-    "type": "yarn",
-} if self.inference_rope_scaling else None
+self.rope_scaling = kwargs.get("rope_scaling")
+if self.rope_scaling is not None:
+    assert self.rope_scaling["type"] == "yarn"
+    assert self.rope_scaling["original_max_position_embeddings"] > 0
+    assert self.rope_scaling["factor"] >= 1.0
```

同时对 `trainer/train_sft_omni.py` 增加并实际消费：

```python
parser.add_argument("--max_position_embeddings", type=int, required=True)
parser.add_argument("--rope_original_max_position_embeddings", type=int, required=True)
parser.add_argument("--rope_factor", type=float, required=True)
parser.add_argument("--rope_attention_factor", type=float, default=1.0)
parser.add_argument("--context_parallel_size", type=int, choices=[1], default=1)
parser.add_argument("--target_nonpad_tokens", type=int, required=True)
parser.add_argument("--schedule_total_nonpad_tokens", type=int, required=True)
parser.add_argument("--init_checkpoint", type=str, required=True)
parser.add_argument("--checkpoint_sha256_manifest", type=str, required=True)
parser.add_argument("--resume_checkpoint", type=str, default=None)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--warmup_ratio", type=float, default=0.03)
parser.add_argument("--flash_attn", type=int, choices=[0, 1], default=1)

rope_scaling = {
    "type": "yarn",
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "factor": args.rope_factor,
    "original_max_position_embeddings":
        args.rope_original_max_position_embeddings,
    "attention_factor": args.rope_attention_factor,
}
omni_config = OmniConfig(
    hidden_size=args.hidden_size,
    num_hidden_layers=args.num_hidden_layers,
    use_moe=bool(args.use_moe),
    max_position_embeddings=args.max_position_embeddings,
    rope_scaling=rope_scaling,
    flash_attn=bool(args.flash_attn),
)
assert args.context_parallel_size == 1
assert args.max_seq_len <= omni_config.max_position_embeddings
```

trainer 还必须以 `input_ids != pad_token_id` 统计每个 rank 的 non-padding input token，经 `dist.all_reduce(..., SUM)` 得到全局累计量，并把 `consumed_nonpad_tokens` 写入 checkpoint。恢复训练后该计数不能清零；`target_nonpad_tokens` 表示本阶段结束时的**累计停止点**。只按 epoch 或 step 停止的运行不能进入主比较。

`--init_checkpoint` 必须解析到显式文件，先按 `--checkpoint_sha256_manifest` 中与该文件相同相对路径的记录比对，再加载 state dict；manifest 每行使用 64 位十六进制 SHA256、两个空格、相对路径的标准格式，且自身也写入 `provenance.json`。不得使用只提供 `llm/sft_omni` 别名、不能解析到固定文件的 `--from_weight` 查找。`--resume_checkpoint` 非空时只允许加载上一阶段完整 checkpoint，并同时恢复 optimizer、scaler、sampler cursor、LR scheduler 与 `consumed_nonpad_tokens`。

patch 还必须把上游硬编码的 `setup_seed(42 + rank)` 改为 `setup_seed(args.seed + rank)`，并把按 epoch/step 调用的 `get_lr(...)` 改成以 `consumed_nonpad_tokens / schedule_total_nonpad_tokens` 为进度的 warmup/decay。三个阶段的 `schedule_total_nonpad_tokens` 都固定为 800M。否则 `experiment.seed`、`schedule_by: consumed_tokens` 和 `warmup_ratio` 仍只是未生效的文档字段。

当前学习仓库没有交付这一正式 GPU 改造，因此这里不提供会切换当前工作区或引用不存在文件的 shell 命令。后续 companion repository 必须在隔离的 upstream worktree 中锁定 commit、应用实际存在的改造、运行单测，再把最终 diff 和 SHA-256 写入 provenance。缺少这条证据链时，报告只能写"只修改了配置"，不能标记为 `YaRN 32K→128K` 或"实验 14 默认主实验"。

## 5. 学完后应能完成

完成后你应能：

- 解释 RoPE 相位、position interpolation、YaRN 和 LongRoPE（逐维搜索非均匀缩放系数的扩窗方法）的差异；
- 为文本、图像 patch、视频帧和音频帧定义统一而不冲突的位置语义；
- 构造不会被模板位置、长度或答案格式泄漏的 long-context 数据；
- 计算 sample-balanced 与 token-balanced mixture（按样本数配比与按 token 数配比）的实际采样概率；
- 理解上下文并行、序列打包和重计算的系统取舍，并用默认的 DDP+CP1 性能测试判断 8 卡可行长度；
- 将失败归因到位置外推、检索、推理、数据捷径或系统吞吐，并避免使用"上下文太长"这类不可检验描述。

## 6. 原理:边造边讲

三个机制，每个按同一节奏走：直觉（为什么需要）、机制（怎么运转）、数学（精确定义）、代码落点（在哪里）、验证（怎么证明做对了）。

### 6.1 RoPE 与长度外推

先回答本课最核心的一个"为什么"：在 32K 上训练的模型，直接喂 128K，位置编码为什么会崩？RoPE（rotary position embedding，旋转位置编码）把注意力 head 的通道两两配对，每一对像一根按自己转速走的表针：token 排在第几位，表针就转多大角度。query 和 key 都这样转过之后，两根表针的夹角恰好编码"这两个 token 隔多远"，注意力从夹角里读出相对距离。问题在于，注意力权重是训出来的：模型只学会了解读训练时见过的那些角度组合。转速快的高频表针在 32K 之内已经转了几千圈，什么角度都见过；转速慢的低频表针在 32K 之内连一圈都没转完，只扫过一小段弧。位置一旦超过训练长度，低频表针就转进了训练时从未到达的角度区间——模型面对的是分布外输入，注意力读数不是缓慢变差，而是可能直接失灵：logits 异常、困惑度爆炸、长文答非所问。一句话：旋转公式本身算多远都行，崩的是"模型没见过那么远的位置对应的输入长什么样"。类比失效处：真表针转满一圈就回到原点、与起点无异，而这里恰恰是"没转完一圈"的低频通道才会闯进新角度——"按转过的圈数区别对待每个通道"，正是 YaRN 的出发点。

应对办法按"动谁的转速"分三类。线性位置插值（position interpolation，PI）把位置 $m$ 换成 $m/s$：所有表针一起减速，128K 的相位被压回 32K 见过的范围，低频不再越界；代价是高频表针也被拖慢，相邻 token 的夹角缩小 $s$ 倍，短距离分辨率受损。NTK-aware 一族（YaRN 的 NTK-by-parts 是代表）按通道区别对待：在原训练长度内转过很多圈的高频通道不动（它们负责短距离分辨，而且本来就不越界），几乎没转完一圈的低频通道按 $s$ 全额插值，中间的通道线性过渡；过渡边界由 `beta_fast=32` 和 `beta_slow=1` 两个"圈数阈值"决定。YaRN 还做第三件事：序列变长后注意力分布趋平，论文为此给 attention logits 配了一个随缩放倍数缓慢增长的温度修正，常见参考实现取 $\sqrt{1/t}=0.1\ln s+1$（$s=4$ 时约 1.14）。本课默认 config 把 `attention_factor` 固定为 1.0；若数值对照证实这确实关闭了论文的温度缩放，Step 2 会要求把产物改名为 `yarn-frequency-only`——名字必须和数值实现对得上。

第 $i$ 对通道的角速度为

$$
\theta_i = b^{-2i/d}
$$

其中 $b$ 是 `rope_theta`（本课起点为 1000000），$d$ 是 head 维度。位置 $m$ 的向量在第 $i$ 对通道上旋转 $m\theta_i$；记二维旋转为 $R(\cdot)$，则

$$
\langle R(m\theta_i)\,q,\ R(n\theta_i)\,k\rangle=\langle R((m-n)\theta_i)\,q,\ k\rangle
$$

内积只依赖相对位置 $m-n$，这就是 RoPE 编码相对距离的方式。每对通道的波长 $\lambda_i = 2\pi/\theta_i$，在训练长度 $L$ 内转过的圈数 $r_i = L/\lambda_i$；YaRN 用 $r_i$ 与 $\beta_{fast}$、$\beta_{slow}$ 比较，决定该通道插值多少。线性 PI 则等价于把所有 $\theta_i$ 统一除以 $s$。

锁定 commit 的 `model/model_minimind.py` 里，`rope_scaling` 由布尔量 `inference_rope_scaling` 触发一组硬编码参数（original=2048、factor=16）；第 4.2 节的 patch 把它改成显式传入并断言的 YaRN 字典。RoPE buffer（预先算好的 cos/sin 查找表）在模型初始化时生成，Step 2 会检查它的覆盖范围和前几个位置。

先用 4 维向量手算位置 0、1、4 的旋转角，再把同一组 query/key 的内积填成表格。随后回答：

- 插值改变的是位置还是频率？
- 为什么简单线性插值可能损害短上下文分辨率？
- YaRN 的 NTK-aware scaling 与温度修正分别解决什么？

如果手算结果无法复现同一相对距离对应的旋转差，先不要修改模型代码。本节通过标准是：公式计算、一个 PyTorch 小脚本和模型 RoPE buffer 的前三个位置在预设容差内一致。

### 6.2 多模态位置

文本 token 只有先后顺序；图像 patch 还有行列坐标；视频帧和音频帧还对应物理时间。把这些位置都压成一个整数不会立即报错，却可能让相同序号承担互不相容的语义：第 3000 个音频 token 与第 3000 个文本 token 共享同一个全局序号，但前者背后是"第几秒"，后者只是"第几个词"。问"第 37 秒说了什么"时，模型需要的是时间轴，不是全局序号。

M-RoPE 指为不同位置轴分配旋转维度的多轴 RoPE。本课不要求实现新 M-RoPE，只要求 collator（把样本拼成 batch 的数据整理器）先把每种位置显式导出为字段：`global_sequence_position`、`modality_local_position`、`time_seconds`、`segment_id` 和 `source_item_id`。位置语义先记录、后建模——字段导对了，第 18 节的 M-RoPE 改造实验才有输入。

本节没有新公式，守的是一条不变量：packing（把多条短样本拼进同一窗口以减少 padding）与截断可以改变 token 的全局位置，但不得改变它的 `(source_item_id, time_seconds)` 归属。全局序号是排版，时间和来源是事实；排版可变，事实不可串。

collator 的字段导出与 packing 逻辑；下游消费方是第 8 节的 manifest 编译器和第 10.1 节的 ledger。

用一条图文样本和一条 30 秒音频样本打印字段，检查 padding、packing 和截断前后的对应关系。第 3000 个音频 token 与第 3000 个文本 token 只共享全局序列索引，不共享物理时间。若 `time_seconds` 或 `source_item_id` 在 packing 后发生串样，本节不通过。

### 6.3 长序列训练系统

窗口从 32K 拉到 128K，长度乘 4，注意力矩阵的元素数乘 16——二次增长的账，显存和算力都要还。系统层的所有手段（FlashAttention、CP、packing）都在跟这笔账讨价还价，没有一个能把二次变成免费。

长序列首先受 attention 激活、算力和通信约束。训练前请对 8K、32K、128K 分别估算注意力矩阵元素数（正比于 $T^2$），并在一张卡上采集峰值 HBM 与 step time。阅读和实测应能解释：

- attention 激活为何随序列长度近似二次增长；
- FlashAttention（分块计算注意力、不在显存里落下完整矩阵的 kernel）降低的是显存 IO，不会消除全部二次计算；
- context parallelism 在序列维切分什么、需要哪些通信；
- packing 为什么能减少 padding，却可能制造跨样本泄漏。

attention 走锁定源码的 SDPA 路径（`--flash_attn`，见第 10.1 节映射表）；packing 与 mask 在数据编译器和 collator 层。

把估算值与 profiler 结果并列。FlashAttention 开启后显存下降但算力仍随长度快速增长，属于正常现象；若 packing 后两个样本可以互相注意，则是 mask 错误，不能用吞吐提升掩盖。

## 7. 目标 recipe

实验变量只有长度课程和 replay。使用同一 checkpoint、数据池、token 预算与优化器，只比较三个主要实验臂：

| 臂 | 长度课程 | 位置方案 | 各数据桶占总训练 token 的比例 |
|---|---|---|---|
| A：control | 直接 128K | YaRN | 长文本 50% / 图文 20% / 分段音频 20% / text anchor 10% |
| B：progressive | 8K→32K→128K | YaRN | 长文本 50% / 图文 20% / 分段音频 20% / text anchor 10% |
| C：no-anchor | 8K→32K→128K | YaRN | 长文本 60% / 图文 20% / 分段音频 20% / text anchor 0% |

先核对三个 resolved config 的 diff：A/B 只能在长度调度上不同。B/C 的唯一数据差异是把占总预算 10% 的短文本回放替换成同一训练池中的长文本；这 10% 不能丢弃，也不能平均分到图文和音频。这样 C 的总 token 仍与 B 相同，并且可以直接解释为"去掉短文本回放，增加等量长文本"。主要结论比较 A/B；B/C 只估计这项替换对短上下文保留的影响。LongRoPE 只能作为扩展题。若 config diff 出现学习率、可训练参数或数据总量差异，三臂比较无效。

公平控制：

- 三臂总训练 token 相同；
- A/B 从同一份 `semantic_sample_ledger`（账本：逐条记录样本来源与 token 范围的只读文件）读取相同的来源顺序和 token offset；
- 8K、32K、128K 只生成不同 window/packing view，不改变 A/B 实际看到的 token identity 或来源曝光次数；
- evidence distance 和 packed position 是长度课程直接改变的量，不能强行要求 A/B 相同；按 arm/stage 完整报告，并检查它们只由已登记的 window compiler 产生；
- C 只有已声明的 10% text-anchor → long-text 替换可以改变 token identity；
- 同一 dev/test；
- 相同 active parameters、optimizer、峰值学习率与 warmup token；
- 相同有效 batch token；
- 相同 3 个 seed；
- 每个长度阶段按 token 而不是 step 对齐；
- 评测使用完全相同 prompt 和生成参数。

dev 用于训练中的 checkpoint 检查、阈值预注册和故障修复；test 在代码、checkpoint、解码配置、统计脚本和所有阈值冻结后只运行一次。任何按 20M/50M token 触发的评测都只能读取 dev。test 结果不得用于延长训练、选择 checkpoint 或修改长度课程。

## 8. 数据字段与检查规则

长上下文样本必须同时记录媒体、证据位置、干扰项距离和 loss 范围，否则无法判断模型答对是检索成功还是数据泄漏。每条样本使用以下 JSONL（每行一条 JSON，忘了的话回[第 1 课](01_baseline_reproduction.md)）：

```json
{
  "id": "lc_v1_000001",
  "messages": [{"role": "user", "content": "<audio>...长上下文问题..."}],
  "assets": [
    {"asset_id": "a0", "type": "audio", "uri": "sha256://...", "duration_s": 1820.4}
  ],
  "segments": [
    {"id": "a0@37.2:45.0", "asset_id": "a0", "start_s": 37.2, "end_s": 45.0, "role": "evidence"}
  ],
  "answer": {"text": "关键词在第 37.2 秒出现", "evidence_ids": ["a0@37.2:45.0"]},
  "length_bucket": "32k",
  "difficulty": {"hops": 2, "distractors": 12, "distance_tokens": 24110},
  "source": "public_or_self_generated",
  "license": "SPDX-or-explicit",
  "split": "train",
  "template_id": "temporal_order_v3",
  "loss_mask": "assistant_only"
}
```

批量生成前：先手工构造一条 32K 音频样本，通过 schema 校验器后再放开产线。校验器要确认 `evidence_ids` 能解析到 `segments`、`distance_tokens` 与实际 token 化结果一致、媒体 hash 可访问、train/test 不共享源条目。任一断言失败时停止编译 Parquet（列式数据表文件）。

### 8.1 数据来源与开放边界

本节解决复现实验时容易遗漏的两个问题：资产许可和评测污染。建立 asset registry 后，再允许下列来源进入数据池：

- 自生成的 passkey（在长文里埋一串口令让模型找回的检索测试）、变量绑定、列表追踪与多跳文本；
- 具有明确许可的长文档；
- 自己采集或有再分发许可的视频、音频；
- 从公开资产生成的问题，但必须保留原始 license 与 provenance。

默认 token mixture 固定为：

| 数据桶 | A/B | C |
|---|---:|---:|
| long text | 50% | 60% |
| multi-image text | 20% | 20% |
| segmented audio text | 20% | 20% |
| short-context anchor | 10% | 0% |

四个比例都按实际进入模型的 non-padding token 计算。manifest 编译器必须分别输出 A、B、C 的桶计数；每个桶与目标比例的绝对误差不超过 0.5 个百分点。C 中增加的 10% 长文本从与 A/B 相同的 `long_text` 训练池采样，不得使用新来源或复制同一条样本凑 token。B/C 的 8K、32K 和 128K 三个阶段都分别使用对应臂的同一组比例，不能把 text anchor 集中到某个阶段。

先为 A/B 冻结一份按 `(source_item_id, source_token_start, source_token_end)` 排序的 `semantic_sample_ledger.jsonl`。长度臂只能把 ledger 条目切成不同窗口或重新 packing，不能丢弃长样本尾部、重复短样本或改变来源次序。编译后再导出 `token_identity_ledger.parquet`，至少保存 source、offset、token id、evidence distance、arm、stage 和 packed position。A/B 的 source-offset-token 多重集合必须完全相同；来源曝光次数也必须一致。evidence-distance 直方图应分别按 arm/stage 保存，因为 B 的 8K 和 32K 阶段本来就会改变可见距离；它是 manipulation check（操纵检查：确认实验变量确实动了该动的东西），不是相等约束。最终 128K dev/test 使用同一 manifest，保证能力比较读取完全相同的 evidence-distance 分布。只比较"来自同一数据池"而不核对 token ledger，仍可能把样本组成差异误认为长度课程收益。

视频不进入默认主矩阵。只有提供 `video_frontend.lock.json`（含代码 commit、checkpoint SHA、frame/tubelet contract、每帧 token 数与离线/在线一致性测试）后，才能从前三桶等比例挪出 10% token 建立 video 扩展；三臂必须共用同一冻结 frontend。

禁止：

- 把 RULER 测试模板原样混入训练；
- 用 closed API 生成答案却不记录模型、时间和过滤规则；
- 将无再分发权的媒体打包进课程数据；
- 使用靠近结尾、固定选项位置等标签捷径；
- 把同一文档切片同时放进 train 和 test。

逐条运行 license、来源和 split-group 检查，并抽样回放媒体。公开论文给出方法，不代表其全部训练数据可获得。本课报告分别标记 `method-open`、`code-open`、`data-open` 和 `weights-open`；四项中任一项没有证据时填写 `unknown`，不能由其他项推断。

### 8.2 三档数据规模

数据档位用于把调试、比较和规模化结论分开。先在 pilot 验证数据与恢复训练，再在 standard 做三臂三 seed；full 只用于验证已出现的趋势。

| 档位 | 训练 token | 最大长度 | 用途 |
|---|---:|---:|---|
| pilot | 30M–60M | 8K/32K | 验证 mask、RoPE、恢复训练 |
| standard | 0.5B–1B | 8K/32K/128K | 三臂正式比较 |
| full | 3B–10B | 8K/32K/128K+ | 仅趋势明确后扩展 |

canonical standard 的 B/C 阶段 token 比例固定为 8K:32K:128K = `35:35:30`，这是 token 比例，不是样本比例；A 在相同总 token 内始终使用 128K 最大窗口。任何不同分配都创建新 experiment ID。

编译完成后，用 non-padding token 重新统计三段比例。每段与目标的绝对误差应小于 0.5 个百分点；这里比较的是 token，不是 JSONL 行数。

## 9. 实验步骤

八个 Step 按依赖顺序执行：基线没冻结不改位置编码，位置单测没过不生成数据，pilot 没干净不进 standard。

### Step 1：冻结基线与构造短上下文回归集

先回答扩长前模型会什么。默认保存 200 个文本、100 个单图、100 个短音频 case；只有启用已锁定的 video frontend 时再加 50 个短视频 case。每个 case 保存原始输入与解码配置，训练后原样回放。

记录：

- 输入 token 数；
- 输出；
- log-prob；
- 能力指标；
- TTFT（首个输出 token 的延迟，定义回第 1 课）、峰值 HBM、tokens/s。

随机重跑 20 个 case，确认相同 seed 下输出与 log-prob 可复现。无法复现的基线不能用于计算 2% 回退。

### Step 2：位置编码单元测试

这一阶段只验证位置与 mask，不启动正式训练。对 8K、32K、128K 分别构造边界样本并验证：

- position id 单调且无越界；
- padding/packing 后 segment 不串扰；
- 图像局部坐标可复现；
- 只有启用已锁定 video frontend 时才测试视频局部/时间坐标；
- 从 checkpoint 恢复后 scaling 参数一致；
- 8K 输入在扩窗前后 logits 漂移可解释。

每个长度至少保存一份 `position_ids`、RoPE buffer 范围和 attention mask 快照。断言失败时输出首个错误 token 的样本 ID、segment 和索引，便于定位 collator 或模型层。

还要核对"YaRN"这个名称是否与数值实现相符。锁定一份可信参考实现和版本，在 `head_dim=8`、位置 `[0, 2047, 32767, 131071]` 上分别导出 inverse frequency、cos/sin 和 attention scaling，与本课 patch 逐元素比较。若 `attention_factor=1.0` 实际关闭了论文/参考实现中的幅度或温度缩放，本课产物必须改名为 `yarn-frequency-only`，不能继续写"完整 YaRN 复现"；另一条合法路径是使用参考公式计算出的 attention factor，并重新生成三臂共同 config。无论选择哪条，A/B/C 必须读取同一实现和同一数值测试快照。

### Step 3：构造反捷径 long-context 数据

模型可能从证据位置、答案格式或模板词猜答案。为隔离这些捷径，每种模板至少生成：

- evidence 位于头部、中部、尾部的均衡版本；
- 相同答案但不同 evidence 位置；
- 相同位置但不同答案；
- 无答案样本；
- 冲突证据与时间顺序样本。

对生成结果做位置与答案的互信息检查，并人工审阅每个模板 20 条。证据位置分布明显偏斜，或仅凭问题文本即可超过随机水平时，重写生成器。

### Step 4：跑 pilot 并排除系统错误

pilot 用于暴露 mask、数值和吞吐问题，不用于宣称能力提升。仅训练 1k–3k steps：

- A 从 128K 起跑；
- B/C 从 8K 起跑；
- 每 100 steps 记录 loss、梯度范数、HBM、step time；
- 每 500 steps 做一次 8K/32K `needle_dev`；pilot 进程没有 test 读取权限。

结束后从中途 checkpoint 启动新进程，比较接续 20 steps 的 token 计数与学习率。若 mask 泄漏、NaN、恢复不一致或有效 token 利用率低于 65%，不得进入 standard。

### Step 5：执行三臂标准训练

B/C 的阶段切换以全局累计 non-padding token 为准：

| 累计训练 token 区间 | 最大序列长度 |
|---|---:|
| 0%–35% | 8K |
| 35%–70% | 32K |
| 70%–100% | 128K |

每次切换前后保存 checkpoint，并用 dev 中相同的 50 个 probes（探针题：固定的小测试题）测即时退化和恢复速度。这些 probe 只能用于检查训练状态，不能充当最终结果。阶段日志中的累计 token 应分别停在 35%、70% 和 100%；不能用 step 数替代。

### Step 6：在 dev 上建立跨模态远距离评测

能力评测要同时改变距离、证据模态和推理跳数。主实验必须覆盖：

- 文本：多跳、变量追踪、干扰段；
- 图像：多图引用和跨图比较；
- 音频：跨段说话人/关键词关联。

只有启用并锁定 `video_frontend.lock.json` 的扩展实验，才额外覆盖视频远距离事件顺序与音视频错配；这些结果进入 `exp14x_video` 独立报告，不参与 canonical 三臂验收。

先在 dev 上完成指标实现、长度分桶和失败定位。对每个长度桶同时报告总体分数和按证据距离分桶的分数。只有平均分、没有距离曲线时，不能声称模型学会远距离使用。此时不打开 test。

### Step 7：做 modality ablation

modality ablation（模态消融）用反事实输入检验答案是否依赖媒体。对每个 dev/test case 预先生成四个变体，并为每个变体保存自己的标准答案：

1. 原始；
2. 删除必要证据，标准答案改为"无法判断"；
3. 只改变 evidence 与 distractor 的位置，内容不变时标准答案保持不变；
4. 保留文本问题但换错媒体；若新媒体有明确答案则重新标注，否则标准答案为"无法判断"。

保持问题文本和解码参数不变，只替换证据条件。不能拿原始答案继续给删除证据或错媒体打分。证据使用至少报告三项：原始输入正确率、删除证据后的正确拒答率、原始答案概率从原始输入到删除证据输入的下降量。预测改变但没有变成新的正确答案，只能说明模型对输入敏感，不能说明证据使用正确。

### Step 8：做三 seed 汇总与 Pareto 分析

先冻结 checkpoint、processor、解码参数、`stats_plan.yaml`、test manifest hash 和评测脚本 SHA。确认训练进程无法读取 test 后，三个 seed 各运行一次 final test。任何补跑都使用新的 run ID 并说明原因，不以补跑结果替换原始结果。

随后按预注册脚本汇总三个 seed 的均值、标准差和 GPU-hours，并画（Pareto 分析：同时看多个目标，找出没被任何配置全面压制的点）：

- 长度—准确率；
- 距离—准确率；
- 训练 token—能力；
- HBM—吞吐；
- 长能力—短能力回归。

在图中标出每个 seed，不只画均值。若 B 只在一个 seed 胜出，或收益伴随超过 2% 的短上下文回退，则渐进式课程未通过。

## 10. 配置示例

```yaml
experiment:
  name: exp14_progressive_yarn
  seed: [17, 23, 41]
model:
  init_checkpoint: jingyaogong/minimind-3o
  revision: ee3febbd08cc5b2bd41c039c825a8934232fee33
  position:
    type: yarn
    original_max_position: 32768
    target_max_position: 131072
    factor: 4.0
    attention_factor: 1.0
  flash_attn: true
data:
  manifest: data/exp14/train.jsonl
  compiled_stage_dir: data/exp14/compiled_parquet
  token_balanced: true
  mixture:
    long_text: 0.50
    multi_image_text: 0.20
    segmented_audio_text: 0.20
    short_context_anchor: 0.10
  arm_c_mixture:
    long_text: 0.60
    multi_image_text: 0.20
    segmented_audio_text: 0.20
    short_context_anchor: 0.00
  buckets:
    - {max_length: 8192, token_fraction: 0.35}
    - {max_length: 32768, token_fraction: 0.35}
    - {max_length: 131072, token_fraction: 0.30}
train:
  schedule_by: consumed_tokens
  total_tokens: 800000000
  learning_rate: 1.0e-5
  warmup_ratio: 0.03
  bf16: true
distributed:
  strategy: ddp
  world_size: 8
  context_parallel_size: 1
  data_parallel_size: 8
eval:
  dev_every_tokens: 20000000
  dev_suites: [short_regression_dev, ruler_clean_dev, long_text_dev, multi_image_dev, segmented_audio_dev]
  final_test:
    run_once_after_freeze: true
    suites: [short_regression_test, ruler_clean_test, long_text_test, multi_image_test, segmented_audio_test]
```

### 10.1 配置到运行时的唯一 mapping

这一节防止配置文件看似完整、训练进程却没有消费字段。上面的 YAML 是正式 GPU companion launcher 的输入，不是上游 trainer 原生格式。companion repository 交付 launcher 后，它必须输出 `resolved_config.yaml` 和完整 argv；以下字段必须一一映射，不能静默忽略：

| 文档字段 | patched trainer/runtime | 断言 |
|---|---|---|
| `model.position.original_max_position` | `--rope_original_max_position_embeddings` | `32768` |
| `model.position.target_max_position` | `--max_position_embeddings` | `131072` |
| `model.position.factor` | `--rope_factor` | `4.0` |
| `model.position.attention_factor` | `--rope_attention_factor` | `1.0` |
| `model.flash_attn` | `--flash_attn` | `1`，调用锁定源码的 SDPA 路径 |
| `model.init_checkpoint/revision` | `--init_checkpoint` + `--checkpoint_sha256_manifest` | 精确匹配 provenance |
| `experiment.seed` | `--seed` | 分别为 17/23/41 |
| `data.manifest` | manifest compiler 的输入 | SHA 匹配 |
| `semantic_sample_ledger/token_identity_ledger` | length-view compiler | A/B token identity 多重集合与来源曝光相同；距离分布按 arm/stage 报告；C 只允许 10% 白名单替换 |
| `token_balanced/mixture/arm_c_mixture` | stage manifest compiler | A/B 为 50/20/20/10，C 为 60/20/20/0；编译后重算实际比例 |
| `data.buckets[*].max_length` | 每阶段 `--max_seq_len` | 不超过 131072 |
| 各阶段累计停止点 | `--target_nonpad_tokens` | B/C 为 280M、560M、800M；A 为 800M |
| `train.total_tokens` | `--schedule_total_nonpad_tokens` | 四次启动都为 800M |
| `train.learning_rate/warmup_ratio` | token-based optimizer scheduler | 以 consumed token 为横轴 |
| `train.bf16` | `--dtype bfloat16` | 八个 rank 一致 |
| `distributed.strategy` | `torchrun` DDP | 只能是 `ddp` |
| `world_size/data_parallel_size` | `--nproc-per-node=8` | 均为 8 |
| `distributed.context_parallel_size` | `--context_parallel_size` | 必须等于 1 |
| `eval.dev_every_tokens/dev_suites` | 独立 dev evaluator argv | 不传给 trainer；运行中不可解析 test 路径 |
| `eval.final_test` | 冻结后的独立 final evaluator | 每个 seed 一次；run ID、manifest 与脚本 SHA 写入账本 |

上游 `OmniDataset` 接受 Parquet，不直接接受本文 JSONL manifest。因此预处理器必须把每个 arm/stage 的 JSONL 编译成 Parquet，并保存 `source_manifest_sha256`、`semantic_sample_ledger_sha256`、`compiled_parquet_sha256`、`token_identity_ledger_sha256`、`row_count`、`nonpad_token_count`、`tokenizer_sha256`、`max_seq_len` 和 `compiler_commit`。

launcher 遇到未映射键、manifest hash 不一致、video 样本出现在 canonical 数据中、或 `strategy != ddp / CP != 1` 时必须 fail closed（一有异常就拒绝启动，不带病运行）。

运行前故意拼错一个键，确认 launcher 会报错；随后比较 YAML、resolved config 和 argv。三处值逐项一致，且运行时断言通过，才算完成映射验收。

### 10.2 canonical torchrun launcher

这一节把表格中的课程转换成可审计进程。B 臂三个阶段分别执行；A 只执行 128K、总 token 为 800M，C 与 B 使用相同阶段长度和 token，只替换 replay mixture：

| stage | `max_seq_len` | 本阶段 token | `target_nonpad_tokens` 累计停止点 | resume |
|---|---:|---:|---:|---|
| B/C-1 | 8192 | 280000000 | 280000000 | 起点 checkpoint |
| B/C-2 | 32768 | 280000000 | 560000000 | 上一阶段 |
| B/C-3 | 131072 | 240000000 | 800000000 | 上一阶段 |
| A-1 | 131072 | 800000000 | 800000000 | 起点 checkpoint |

launcher 必须把表中每一行解析为一条完整且可审计的 `torchrun` argv。当前学习仓库没有提供正式 GPU trainer、checkpoint 或 compiled Parquet，因此不展示一条表面可复制、实际必然缺文件的启动命令。companion repository 交付这些文件后，网页才应展示由 `--dry-run` 生成并验证过的实际命令。

stage 2/3 必须以 `--resume_checkpoint` 从上一阶段完整 checkpoint 恢复 model、optimizer、scaler、sampler cursor 和累计 token；不得只加载权重后重新开始 optimizer。launcher 必须保存 `launch.argv.txt`、环境变量白名单、每 rank 日志与退出码。

先用 1000 个目标 token 做 dry run，检查八个 rank 的参数、输入分片和退出码。正式运行目录中缺少任一 argv、日志或 checkpoint 状态文件时，不得进入下一阶段。

### 10.3 启动单测与 50 步验收

这一节判断实现是否正确，以及 8 卡能否承担目标长度。当前学习仓库只提供 CPU 机制实验；正式 GPU companion repository 必须分别提供可运行的 config/RoPE/manifest/token-resume 单测和独立的 8-rank DDP startup smoke。两者是两个执行步骤，不能合并成一个代码块，也不能在对应文件尚未交付时发布复制命令。

页面顶部的 CPU toy 只检查一组小向量上的位置插值数学、长度调度和恢复计数。它没有实现 YaRN 的 `beta_fast/beta_slow` 分频插值、attention scaling，也没有运行 128K attention。CPU toy 通过只能说明这些基础算式和状态转换可复现，不能写成"完整 YaRN 已复现"或"128K 已可训练"。完整 YaRN 必须通过 Step 2 的锁定参考实现逐元素对照。

测试必须断言：

1. runtime 中 `max_position_embeddings=131072`；
2. `rope_scaling` 精确为 YaRN、original=32768、factor=4，而不是上游 legacy 的 2048/16；
3. RoPE buffer 至少覆盖 131072，8K/32K/128K position slice 不为空且无越界；
4. 八个 rank 的 config、checkpoint、tokenizer 与 manifest hash 完全一致；
5. canonical wrapper 是 DDP，`context_parallel_size=1`，不存在 FSDP/CP shard；
6. 一次 forward/backward/optimizer/save/new-process-resume 成功；
7. resume 前后 `consumed_nonpad_tokens`、LR 与 sampler cursor 连续；
8. canonical batch 中只有 text/image/audio，没有 video/AV 样本。

单测通过后再分别做 8K、32K、128K 的 50-step HBM/吞吐测试。128K OOM 时记录为不可行并止于 32K，禁止偷偷启用未验证的 CP/FSDP 后仍沿用 canonical ID。此时报告标题必须标明"32K 工程部分结果，128K 主实验未完成"，不得计算或解释 A/B 的 128K 主效应。

每项测试重复两次，记录峰值 HBM、P50/P95 step time 和 non-padding tokens/s。第二次运行未出现 OOM/NaN，且指标波动在预注册范围内，才将该长度标记为可训练。

## 11. 训练预算与 8 卡运行方案

| lane | 硬件 | 目标 | 预计工作量 |
|---|---|---|---|
| debug | 1×24–48GB | 8K、batch=1、mask 单测 | 2–6 GPUh |
| pilot | 8×24–48GB | 32K，三臂短跑 | 80–200 GPUh |
| standard | 8×80GB，优先 NVLink | 128K、DDP+CP1 | 800–2400 GPUh |
| full | 多节点 80GB | 128K+ 与更大模型 | 单独立项 |

canonical 只接受 DDP+CP1。若另做 `CP=2/4/8`，命名为 `exp14x_cp`，先验证：

1. 同一 2K batch 在 CP=1 与候选 CP 下 logits/loss 满足预注册 tolerance；
2. backward 梯度与 CP=1 对齐；
3. media span all-gather 后顺序与原 `position_ids` 相同；
4. checkpoint 能回到 CP=1 恢复；
5. device mesh 和通信 trace 可审计。

CP 扩展结果不得替换或混入 canonical A/B/C 主表。

每次 profile 保存 NCCL trace（NCCL：多卡集合通信库）、每 GPU HBM、算子时间和 padding 比例。

## 12. 指标

### 12.1 能力指标

能力指标回答模型是否找到并使用了证据。按长度、证据位置、模态和跳数切片后记录：

- RULER 各任务、各长度准确率；
- clean needle 与 adversarial needle（换了措辞和位置的对抗版藏针题）；
- 多跳 exact match/F1；
- 视频扩展启用时：事件顺序 accuracy；
- 音频跨段 speaker/keyword F1；
- 删除证据后的 performance drop。

每个汇总分都应能下钻到 case ID；缺少 per-case 记录的指标不进入主结论。

### 12.2 系统指标

系统指标回答结果需要多少显存和时间。用相同 profiler 配置采集：

- peak allocated/reserved HBM；
- 有效非 padding tokens/s；
- step time P50/P95；
- checkpoint save/load 时间；
- NCCL 通信占比；
- OOM、NaN、loss spike 次数。

所有吞吐使用有效 non-padding token 作为分子。仅报告 raw tokens/s 会把 padding 误计为有效工作。

### 12.3 回归指标

回归指标用于发现扩长训练对已有能力的损害。训练过程中只运行 dev 版本；最终在同一冻结测试集和解码配置上比较：

- 原生 8K 文本综合；
- 单图 QA；
- 30 秒内 ASR；
- video extension 启用时：30 秒内视频理解；
- 原 checkpoint 的 perplexity 与生成格式有效率。

报告绝对差与 bootstrap 置信区间。任何一项超过预注册回退阈值，都要在能力收益旁并列显示。

### 12.4 统计口径

主统计单位是未见测试集中的 `source_item_id`，不是 token、窗口或同一来源切出的多个问题。同一个来源在 A/B/C、正确媒体和删除证据条件下使用相同 case ID，因此差值按 case 配对。若一个来源生成多个问题，先在来源内求平均，再进入总体统计，避免把同一材料重复计数。

每个主效应使用 3 个 seed，并做 10,000 次分层配对 bootstrap（有放回重采样：用重抽样分布估计置信区间）：先对 seed 有放回抽样，再在长度、模态和难度层内对 `source_item_id` 有放回抽样。报告三 seed 平均绝对差、95% percentile CI 和每个 seed 的单独差值。测试集、分层字段、最小效应量和 bootstrap 脚本 SHA 必须在第一次查看 test 结果前写入 `stats_plan.yaml`。事后更换指标或只报告有利切片，不进入主结论。

预注册的最小效应量如下：

- 渐进课程收益：B 相对 A 的 32K/128K 长上下文 macro accuracy 至少提高 3 个百分点，且配对 95% CI 下界高于 0；
- 证据使用：在原始输入答对的 evidence-required case 中，B 删除证据后的正确拒答率至少为 70%，且配对 95% CI 下界至少为 60%；同时原始答案概率的平均下降量必须大于 0，且其配对 95% CI 下界高于 0；
- 短能力非劣：B 相对起点的短上下文综合差值，其 95% CI 下界不低于 -2 个百分点；
- anchor 效应：只有 B 相对 C 的短上下文综合至少提高 2 个百分点，且 95% CI 下界高于 0，才可以写"10% text anchor 减少了遗忘"。

这些阈值用于决定允许写出的结论。只要数据、训练和统计协议完整，未达到阈值仍是有效的负结果，不得改阈值后重算。

## 13. 验收条件

先检查实验是否有效。以下条件必须同时满足：

- 三臂达到相同的总 non-padding token，误差不超过 0.5%；
- A/B 的 token identity 多重集合和来源曝光次数一致；各阶段 evidence-distance 分布与已登记的 window compiler 输出一致；
- A/B 和 B/C 的 resolved config 差异分别符合第 7 节白名单；
- 测试集按 `source_item_id` 与训练集隔离，所有主指标保留 per-case 记录；
- 训练日志中的周期评测只读取 dev，final test 在全部配置冻结后每个 seed 只运行一次；
- 三个 seed 都完成，且使用预注册的统计脚本；
- 128K 训练无持续 NaN，P95 step time 不超过 P50 的 1.5 倍；
- 有效 token 利用率至少 70%；
- 每个结论可追到 manifest、config、checkpoint 和 case。

实验有效后，再按第 12.4 节判断结论：

- 渐进课程、证据使用和短能力非劣三个阈值全部满足，才可发布 `long-context-progressive-v1`；
- adversarial 模板准确率还必须达到 clean 的 80%，用于排除固定模板捷径；
- anchor 效应单独由 B/C 阈值决定，不影响 A/B 主比较是否完成。

上面的"实验有效"还要求 A/B/C 三臂都完成 128K 阶段和冻结的 128K test。只完成 8K/32K 时可交付实现、显存和吞吐记录，但课程主问题仍标为未完成，不进入 `long-context-progressive-v1` 判定。

若窗口可运行但证据使用阈值未通过，结论写为：系统支持目标窗口，但本实验未验证模型会使用远距离证据。若 A/B 差值的置信区间包含 0，则写为：本次实验没有证明渐进式课程优于直接扩展。两种情况都保留完整结果，不删除失败 seed。

## 14. 失败诊断表

调试时先从最便宜的假设查起：先怀疑模板和数据，再怀疑位置编码，最后才动训练配置。

| 症状 | 优先假设 | 检查 | 修复实验 |
|---|---|---|---|
| clean needle 高、改模板后崩 | 模板泄漏 | 换 key、位置、措辞 | adversarial template split |
| 头尾高、中部低 | lost-in-the-middle | 按位置分桶 | 位置均衡训练 |
| 128K 可跑但答非所问 | 检索未学会 | evidence removal | 增加跨段监督 |
| 长能力升、8K 降 | 遗忘 | anchor 消融 | 恢复 10% text replay |
| video extension 远距差、文本正常 | 时间位置不一致 | time index trace | 修复局部/全局位置 |
| loss 在扩长时尖峰 | RoPE/优化切换 | 相位与 LR 日志 | 缓慢 length ramp |
| tokens/s 很低 | padding/通信 | profiler | length packing、降低 canonical 可行长度 |
| 无证据也答对 | 标签或常识捷径 | 错媒体/无答案 | 重写数据生成器 |
| 只会找最后一段 | answer-position bias | 位置条件统计 | 均衡 evidence 位置 |
| 长答案变得冗长 | 长度作为 reward proxy | 输出长度分布 | 固定格式与短答评测 |

## 15. 逐个样例检查

canonical 至少审计 60 个 case，每个长度桶 20 个；text、image、audio 三种默认模态各不少于 10 个。启用 video extension 时另加至少 10 个 video/AV case，不挤占默认 60 个。

每个 case 保存：

```yaml
case_id: lc_audio_eval_0037
input_assets: ["sha256:..."]
required_evidence:
  - {segment: "a0@37.2:45.0", fact: "关键词 alpha 出现"}
  - {segment: "a0@1712.0:1718.5", fact: "关键词 beta 后出现"}
model_answer: "alpha 先出现"
answer_correct: true
evidence_present: true
counterfactual_answer: "无法判断"
counterfactual_correct: true
trace:
  input_tokens: 126401
  output_tokens: 18
  latency_ms: 8420
failure_layer: null
```

审计时逐项判断：

- 所需事实是否真实存在于输入；
- connector 是否生产了对应 token；
- 位置/时间索引是否正确；
- 模型是否引用正确证据而非模板猜测；
- 删除证据后是否改变答案；
- 错误属于 coverage、representation、retrieval 还是 reasoning。

## 16. 交付物

```text
artifacts/exp14/
├── README.md
├── configs/{arm_a,arm_b,arm_c}.yaml
├── manifests/{train,dev,test}.jsonl
├── provenance/
│   ├── checkpoint.files.sha256
│   ├── position.yaml
│   ├── applied.patch
│   ├── applied.patch.sha256
│   ├── source_and_compiled_data.sha256
│   └── licenses.csv
├── checkpoints/index.json
├── metrics/{summary,per_length,per_case}.jsonl
├── runtime/{resolved_config.yaml,launch.argv.txt,startup_report.json}
├── traces/profile_ddp_cp1.json
├── cases/audit.md
├── plots/
└── report.md
```

`report.md` 必须明确写渐进式课程是否获得实验支持，并给出 GPU-hours。

## 17. 复现清单

- [ ] 起点 checkpoint SHA 和 tokenizer hash 已记录；
- [ ] 配置容量、披露训练长度、scaling reference 已分字段记录；
- [ ] 正式 GPU 代码库中的源码改造已通过单测，实际 diff 与 SHA-256 已进入 provenance；
- [ ] 每个 YAML 字段均有 launcher/runtime mapping，无静默忽略项；
- [ ] canonical runtime 为 torchrun DDP+CP1；
- [ ] config/RoPE/manifest/token-resume/DDP 启动单测通过；
- [ ] RoPE 固定向量与锁定参考实现一致；若关闭 attention scaling，产物已标为 YaRN frequency-only 变体；
- [ ] 三臂只差课程/replay 变量；
- [ ] 总训练 token 和有效 batch token 对齐；
- [ ] 数据去重跨原始文档/媒体执行；
- [ ] test 模板未进入训练；
- [ ] position、mask、packing 单测通过；
- [ ] 3 seed 完成或报告为何不足；
- [ ] 8K、32K、128K 都有能力与系统指标；若 128K 未完成，报告已明确降级为 32K 工程部分结果且未声称回答主问题；
- [ ] evidence removal、shuffle、wrong-media 都已跑；
- [ ] canonical 必做数据、评测和 case 不含 video/AV；
- [ ] 短上下文回归已跑；
- [ ] checkpoint 可恢复；
- [ ] 逐 case 审计可回放；
- [ ] license 与生成来源可追溯。

## 18. 前沿对照与改造方向

本课的三件套——位置缩放、渐进课程、"有效长度"评测——都能在 2024–2026 年的公开系统里找到放大版。位置缩放这条线，[YaRN](https://arxiv.org/abs/2309.00071) 的按频率区别插值已经成为开源模型扩窗的常用起点，[LongRoPE](https://arxiv.org/abs/2402.13753) 更进一步，用搜索为每个维度找非均匀缩放系数，并同样采用渐进扩展加短上下文恢复的两段式流程。课程这条线，[Nemotron 3 Nano Omni](https://arxiv.org/abs/2604.24954) 把长上下文训练排成 `16K → 49K → 256K` 的台阶——和本课 B 臂同一形状，只是每级台阶高一个量级；它同样把"位置扩展"和"数据课程"当两件事分别处理。评测这条线，[RULER](https://arxiv.org/abs/2404.06654) 的核心发现是：很多模型的有效上下文长度明显短于其声称长度——"能吃进去"和"能用起来"的差距，前沿模型一样存在，本课的距离-准确率曲线就是这套测量的缩小版。多模态位置上，[Qwen2.5-Omni](https://arxiv.org/abs/2503.20215) 用时间对齐的多轴位置编码（TMRoPE）把音频帧和视频帧按绝对时间对齐到同一位置轴——本课第 6.2 节只导出 `time_seconds` 字段而不改旋转，前沿是把时间轴直接编进位置编码。长上下文对 omni 系统的压力也比纯文本来得早：按 [Moshi](https://arxiv.org/abs/2410.00037) 的 12.5 Hz 帧率折算，半小时对话单条音频流就是 22500 帧，还要乘上双流和文本流。

规模差距：hidden 768、8 层、800M token 预算，对前沿的几十亿参数和大得多的长文本预算；窗口 128K 对 256K 以上。这些砸资源能缩小。机制差距才是本课教的：YaRN 缩放、按累计 token 切换的课程、text anchor 防遗忘、evidence removal 评测，这套流程和前沿是同构的，做完本课你缺的只是钱。真正还没做的机制有两个：时间对齐的多轴位置编码（TMRoPE 一类，本课只导出了字段），以及 CP/序列并行级别的系统支持（本课刻意 CP=1，把系统变量摁住）——前者是下面改造清单的第 2 项，后者留给 `exp14x_cp`。



1. **位置编码变体三臂对比（结构级）。** 把第 4.2 节 patch 的 `rope_scaling` 分支扩成三种：线性 PI（把所有 inverse frequency 统一除以 factor=4）、`yarn-frequency-only`（本课默认，`attention_factor=1.0`）、YaRN 加温度（`attention_factor` 按参考公式 $0.1\ln s+1$ 计算）。改动位置：`model/model_minimind.py` 的 rope_scaling 构造处，加 `type` 分支；Step 2 的数值快照对三种实现各存一份。预算：pilot 档（30M–60M token、8K/32K），三臂各一 seed，约 60–150 GPUh。预期：PI 臂的短上下文回归（8K 文本综合、perplexity）最大；两个 YaRN 臂长上下文 needle 相近、短回归更小；温度臂与 frequency-only 的差异在 pilot 档可能小于 seed 噪声，分不出就如实写"未分辨"。失败判定：三臂所有指标都在噪声内不可分——说明 pilot 长度或数据不足以暴露差异，升 standard 之前不下任何结论。
2. **简化 M-RoPE（结构级）。** 把 head 维度按位置轴分组：文本用全局序号，图像 patch 分行、列两轴，音频帧用时间轴；各轴用第 6.2 节 collator 已导出的 `modality_local_position` 和 `time_seconds` 旋转。改动位置：`model_minimind.py` 的 rotary 应用处（按通道段选择位置源），加上 collator 到模型的 position 传递；三臂共用同一实现。预算：pilot 档 32K，多图与分段音频桶为主，约 80–200 GPUh。预期：多图跨图比较与音频跨段关联的距离-准确率曲线整体上移，文本回归不超过 2%。失败判定：文本回归超阈值，或多图指标毫无变化——先查轴分配是不是吃掉了太多高频通道，再查局部坐标有没有在 packing 时串样。
3. **sliding-window 加 attention sink 的混合注意力（结构级）。** 把大部分层换成固定窗口（如 4K）的局部注意力，保留少数全局层或若干 sink token（始终可被注意的开头位置），对照全注意力的显存、吞吐与距离-准确率。改动位置：attention mask 构造与 SDPA 调用处；先跑第 10.3 节式的 50-step HBM 测试，再做 pilot 微调。预算：50-step 测试数 GPUh，加 pilot 一臂 40–100 GPUh。预期：128K 峰值 HBM 明显下降、tokens/s 上升；超出窗口距离的 needle 准确率明显下降——这个"失败"本身就是教学结论：省掉的计算正是远距离证据的通路。失败判定：窗口内的近距离任务也崩，那是 mask 实现错了，回 Step 2 查，别把 bug 当取舍。
4. **text anchor 剂量实验（数据级）。** 把 B/C 对比推广成 anchor 比例 0/5/10/20% 四个点（新 experiment ID，模型零改动），画短上下文回退对长上下文收益的 tradeoff 曲线。改动位置：mixture 配置与 manifest 编译器。预算：pilot 档四臂，60–120 GPUh。预期：短回退随 anchor 比例单调减小，长上下文收益在高 anchor 端被稀释。失败判定：曲线无单调趋势且跨 seed 不稳定，说明 pilot 预算下遗忘信号太弱，只能升档再试。

本课的缩小版设置能按方向复现四篇材料的核心结论：

| 论文结论 | 缩小版对应实验 | 预期 |
|---|---|---|
| YaRN：按频率区别处理比均匀线性插值更少损害短上下文 | 改造 1 的 PI 臂对 `yarn-frequency-only` 臂 | 方向可复现：PI 臂短回归更大；幅度不必对齐论文 |
| LongRoPE：渐进扩展加短文本恢复优于一步扩到位 | 主实验 A/B 臂加 text anchor（B/C）| 方向有望复现；若 A/B 差值 CI 含 0，按第 13 节如实报负结果 |
| RULER：有效上下文长度普遍短于声称长度 | 距离-准确率曲线与 `validated_effective_length` 字段 | 几乎必然复现：可运行长度大于能力验收长度 |
| Nemotron 3 Nano Omni：分级长度课程可行 | B 臂 8K→32K→128K | 只复现"课程形状"；论文未完全公开数据与工程字段，不做数字对齐 |

## 19. 原始论文与官方实现精读

每篇材料带着能在本课产物里核对答案的问题去读；答不上来就回去重读。

### 必读 1：[RoFormer](https://arxiv.org/abs/2104.09864)

精读：RoPE 定义、相对位置性质、实验。

阅读任务：推导旋转后的 query-key 内积如何编码相对距离，与第 6.1 节的内积公式对上；再标出论文中不能直接外推到 128K 的假设——注意论文证明的是"公式只依赖相对位置"，没有承诺"没见过的相对位置也能读懂"。把推导与 6.1 的 4 维手算互相核对。

### 必读 2：[YaRN](https://arxiv.org/abs/2309.00071)

精读：Method 中的 interpolation/extrapolation 讨论、NTK-by-parts、attention scaling、长上下文实验。

阅读任务：解释不同频率维度为何采用不同处理（对照 6.1 的"圈数" $r_i$），并从论文消融中定位短上下文损失来源。把论文参数与本课 `rope_scaling` 字段逐项对应，特别回答：`attention_factor=1.0` 关掉了论文的哪一部分？这决定 Step 2 里你的产物叫 YaRN 还是 `yarn-frequency-only`。

### 必读 3：[LongRoPE](https://arxiv.org/abs/2402.13753)

精读：non-uniform interpolation、progressive extension、短上下文恢复。

阅读任务：判断搜索得到的缩放参数能否迁移到不同模型，并区分位置扩展与数据课程各自解决的问题——本课把两者拆成 Step 2 和 Step 5，论文把哪些混在一起了？用一页表格比较 LongRoPE 与本课 B 臂。

### 必读 4：[RULER 论文](https://arxiv.org/abs/2404.06654) 与 [NVIDIA RULER 官方仓库](https://github.com/NVIDIA/RULER)

精读：任务设计、effective context length、各任务生成器与评测脚本。

阅读任务：找出单 needle 高估能力的原因（对照 Step 3 的反捷径设计），并从官方生成脚本定位模板污染风险。选一个 RULER 任务生成 20 条样本，逐条检查与你的训练模板是否重合——第 8.1 节禁止清单的第一条就是为它写的。

### 必读 5：[Nemotron 3 Nano Omni](https://arxiv.org/abs/2604.24954)

精读：architecture、training curriculum、long-context 与 modality mixture 相关章节。

阅读任务：把论文中的 `16K → 49K → 256K` 拆成位置扩展与数据课程两部分，逐项对照本课 B 臂的 8K→32K→128K；再列出论文未公开的数据和工程字段。只有能追溯到论文或代码的内容才能写入复现声明——这正是 4.1 节 provenance 纪律的论文版。

## 20. 扩展题

1. 用 LongRoPE 替换 YaRN，保持训练 token 不变，只作为 B 臂的二级实验；
2. 比较 1D RoPE 与简化 M-RoPE 在多图/视频上的位置归纳偏置；
3. 加入 memory token 或检索器，比较扩大窗口与外部检索；
4. 用 attention sink 或 sliding-window hybrid 降低 128K 成本；
5. 构造跨音视频的 3-hop benchmark，要求同时引用两个时间段和一条文本约束；
6. 将 context parallel profile 迁移到 1B–3B，验证教学模型的系统结论是否成立。

## 21. 最终 checkpoint 的名称与字段

只有满足验收条件的 checkpoint 才可标记为 `long-context-progressive-v1`。

它应同时提供：

- `supports_context_length`：系统可运行长度；
- `validated_effective_length`：通过能力验收的长度；
- `position_scheme` 与缩放参数；
- 每种模态的最大 token/time；
- 短上下文回归结果。

前两个字段的差距，就是本课反复强调的"能吃进去"与"能用起来"的差距，公开写出来，不藏在平均分里。

**第四幕到此收官。** 盘点这四课换上的"现代内核"四件套：[第 11 课](11_tiny_moe.md)的 MoE 让参数规模涨上去而每 token 计算量不涨；[第 12 课](12_mamba_attention_hybrid.md)的 Mamba-2 混合层把长序列的全量回看换成固定大小的状态；[第 13 课](13_distributed_8gpu.md)的 8 卡并行在不改变优化语义的前提下让系统装得下、训得动；本课把窗口拉到 128K，并立下"窗口能跑、位置不崩、证据会用"的三级判定。结构这条线暂告一段落——模型的身体已经是现代的了，接下来轮到教法。第五幕训练方法升级：[第 15 课](15_joint_multimodal_sft.md)先做多模态联合 SFT，把文本、图像、音频理解与语音生成放进同一次训练交错采样，回答"怎么混不冲突、怎么防遗忘"；随后第 16、17 课接上偏好优化与 GRPO 可验证奖励强化学习。下一课可以从本课的 checkpoint 开始 joint SFT；也可以按它自己的独立起点重新开始，避免把本课的负结果带进后续实验。
