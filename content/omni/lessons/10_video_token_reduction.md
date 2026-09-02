---
id: 10_video_token_reduction
title: "视觉 Token Reduction"
summary: "保留率相同的前提下，按画面变化或相似度来压缩视觉 token，能比随机丢和均匀丢更好地保住 OCR、小目标和短暂事件，还真把端到端延迟压下来吗？"
unit: vision
play_tools: []
checkpoints:
  - "搞清 pooling、merging、pruning、sampling 四种压法各把成本花在哪个环节。"
  - "实现 EVS 和 similarity merge（按相似度合并），并保住每个 token 原来的时空 position id。"
  - "按 100/75/50/25% 四档保留率做扫描，配上随机、均匀和内容感知三组对照。"
  - "把 decode、vision、reducer、prefill、generation 的延迟一段段拆开计。"
---

# 第 10 课：视频 Token 压缩与质量延迟权衡

> 内容：视觉 token 采样、合并、剪枝与系统测量；建议时长：5–8 天；最小硬件：1×24GB 做推理剪枝，2–8 卡做 uptraining
> 独立起点：由 preflight 锁定的官方 `jingyaogong/minimind-3o` dense checkpoint；第 09 课的产物只能通过另一份不可变 frontend manifest 进入迁移扩展

## 1. 视频 token 不能无限增长

第三幕走到最后一站。[第 08 课](08_dynamic_vision.md)把"看图"从固定小图升级成任意分辨率;[第 09 课](09_native_video.md)把视频变成带统一时间轴的原生模态:帧有时间戳、声画按真实时间交错。眼睛长出来了,但第 09 课留下一笔没算的账,这课来算:**视频 token 太多,不压缩就训不动、跑不快。**

**先把账算出来。** MiniMind-O 一张图固定占 64 个视觉 token(`image_token_len=64`)。视频逐帧过 encoder,每帧还是 64 个:pilot 里的 4–8 帧,合计 256–512 个,勉强扛得住;可按每秒抽 1 帧看一段 5 分钟的视频,就是 300 帧、300 × 64 = 19,200 个视觉 token,单图的 300 倍。要命的是 Thinker 的 attention 计算量随序列长度平方增长:token 多 300 倍,prefill(生成第一个字之前,把整段输入一次算完的阶段)的计算量翻 9 万倍;KV cache 显存另按 token 数线性涨;换成第 08 课的多 tile 高分辨率输入,每帧再乘一个 tile 数。这笔账不分模型大小,都得还。

好消息是视频 token 冗余得厉害:相邻两帧多数 patch 几乎没变,盯 5 分钟监控画面,值得记的可能只有红灯亮起那 0.4 秒。这课在 vision encoder 和 Thinker 之间插一个 reducer(视觉 token 压缩器),按整数预算删掉或合并冗余 token,只把值钱的送进 LLM。三种正式策略同台:均匀分层采样(A 臂)、相似度合并(B 臂)、按帧间变化打分的 EVS 剪枝(C 臂),再加同预算随机删除当负对照；赢不过随机的算法等于没用。

**取舍必须是明白账。** 每种压缩都在丢信息,这课的纪律是丢得明码标价:pooling 丢文字笔画和小目标,换确定的 token 数;merge 丢相似 token 之间的区分度,换部分保留的信息;prune 永久丢掉被删 token,换真正变短的序列;sampling 丢掉没进 encoder 的帧,换唯一能省下 encoder 计算的机会。所以验收从不只看"删了多少 token",而要同时报质量、token、延迟、显存四维,画 Pareto,还要单独盯 OCR、小目标、短暂事件这些最先受伤的桶。

跳过这课,第四幕的长上下文课([第 14 课](14_long_context_curriculum.md))一碰长视频就被 prefill 和显存卡死,第六幕接真正的大模型时更付不起。做错这课；只看 FLOPs 不看延迟、只看均分不看困难桶；会得到纸面变快、实际变笨的系统。

做完这课,你能指着 Pareto 图上 50% 保留率的点,说出它相对无压缩掉了几个点、prefill p95 降了百分之几;也能调出任一 OCR case 的 keep-mask 热图,亲眼看到是谁把字幕删了。

本课术语:

| 术语 | 简要解释 |
|---|---|
| token budget / 保留率 | 允许进 Thinker 的视觉 token 整数配额;保留率 50% 就是删一半 |
| pooling / merging / pruning / sampling | 四种减法:邻域求平均、相似的合成一个、直接删、少送帧进 encoder |
| EVS | 按帧间变化打分的剪枝:同一位置变化越大越该留(出处见论文节第 3 篇) |
| token mass | 合并 token 的"体重":它代表几个原始 token;没合并过的都是 1 |
| proportional attention | 按体重补权重:视觉 key 的 attention 得分加上 log(mass) |
| prefill | 生成第一个字前把整段输入一次算完;视觉 token 的延迟大头付在这里 |
| HBM | 显卡的高带宽显存;峰值占用决定能塞进多长的视频 |
| Pareto frontier | 质量、token、延迟三头都没被别的配置全面压过的配置集合 |
| uptraining | 压缩后再训一小段让模型适应缺 token;收益和算法本身分开记账 |
| feature cache | 把 encoder 输出存盘反复用,单卡也能扫几十个配置 |

## 2. 压缩位置与实验范围

本课解决视觉 token 过多导致的 LLM prefill 延迟和显存开销,刀只落在一处:**进入 LLM 的视觉 token 集合**。B 臂会额外把 token mass 作为固定 attention bias 传入 Thinker,但不改 Thinker 权重。抽帧、vision encoder、训练数据和答案生成保持不变——变量只留一个,收益才能记到 reducer 头上。

实验需要验证以下结论:

> 视频 token 存在可预测冗余;在相同保留率下,基于变化或相似度的压缩比随机剪枝、均匀降采样更能保住 OCR、小目标和短暂事件,并降低真实 prefill 延迟。

以下结果不能支持该结论,先写下来防止事后找补:

- 同 token budget 下不优于随机或均匀方法;
- FLOPs(浮点运算次数)下降,但端到端延迟不降;
- 只在静态视频上有效;
- 训练期收益其实来自额外训练 token;
- 总体均分保住了,但 OCR、短事件或低频模态明显退化。

验收时必须同时报告质量、LLM 输入 token、端到端延迟和峰值 HBM,并绘制四维 Pareto。单独报告 token 删除比例,既说明不了系统更快,也说明不了质量丢了多少。

## 3. 进入条件与独立起点

开始前要有:

- 已冻结的图像/视频输入 pipeline 和确定性评测抽帧;
- 每个视觉 token 都记录了 `frame_id, x, y, original_position_id`;
- 至少 100 个逐 case:OCR、小目标、空间、时序、短暂事件、静态视频;
- 能分别测 video decode、vision encoder、reducer、LLM prefill 四段耗时;
- 没做第 09 课时,用逐帧 SigLIP2 + concat 当前端,所有实验臂共享这一前端;
- 先跑无压缩 baseline 并保存逐样本 logits;否则无法检查压缩实现是否改坏了非视觉路径。

### 3.1 独立起点与 preflight 锁

主实验固定使用 `canonical-independent` profile:官方 dense checkpoint + 逐帧 SigLIP2 + 第 09 课定义的 "uniform spatial sampling → chronological concat" 前端。该前端在 preflight(正式实验前的一次性解析与锁定)时确定,运行时不得根据第 09 课的结果自动切换——条件逻辑会让两次运行的起点悄悄不同。

```yaml
# manifests/base_model.lock.json
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
```

preflight 以完整 commit/revision 下载并重新计算 checkpoint SHA-256;同时冻结 vision processor/encoder 的 resolved revision、逐文件 SHA-256、file-tree SHA-256,以及抽帧和 frontend 配置 hash。随后写出只读的 `base_model.lock.json` 和 `video_frontend.lock.json`,三个 reducer 臂只引用这两个 manifest 及其实际 SHA-256,训练阶段禁止重新解析浮动 ref。

若要研究第 09 课选定的最佳臂,另开 `transfer-exp09` 实验:preflight 必须接收一个已存在的实验 09 checkpoint 文件,计算其 SHA-256,并把其父 manifest SHA-256、arm 名、训练数据 manifest SHA-256 写入新的 `video_frontend.lock.json`。不得在同一个 config 中写 `winner-or-baseline`,也不得把迁移结果与独立主实验混算。

## 4. 完成本课需要掌握的操作

1. 区分 pooling、merging、pruning 和 sampling;
2. 说明 attention score 与信息价值的区别,并设计负对照验证;
3. 实现保留原 position id 的 EVS;
4. 设计相同 budget 的随机与均匀负对照;
5. 区分训练期 uptraining 与推理期即插即用;
6. 判断系统是 vision-encoder-bound、prefill-bound 还是 decode-bound(瓶颈卡在哪一段);
7. 按运动强度和事件持续时间解释退化。

## 5. 原理:边造边讲

五个机制,每个按同一节奏走:直觉、机制、数学、代码落点、验证。

### 5.1 四类减法:各丢什么,各换来什么

四类方法的差别,就是丢的信息种类和省钱的位置不同。先看价签:

- **Pooling**:固定邻域求均值/卷积。丢的是邻域内的空间细节——文字笔画、小目标轮廓会被平均抹平;换来的是 token 数完全确定、实现最便宜。
- **Merging**:把相似 token 合成一个表示。丢的是相似 token 之间的区分度;换来的是信息以加权形式部分保留。每个 token 同时保存一个 `token_mass`,表示它代表多少个原始 token;原始 token 的 mass 为 1。这本账在 5.4 节要用来在 attention 里补权重。
- **Pruning**:直接删除部分 token。丢的是被删 token 的全部信息,后续层无法恢复;换来的是序列真的变短、实现直接。
- **Sampling**:减少送入 encoder 的帧或 patch,通常发生在 encoder 前。丢的是没被编码的内容——短暂事件可能整个错过;换来的是唯一能省下 vision encoder 计算的机会。它与 LLM 前 token pruning 节省的计算阶段不同,报告里不能混记。

四类方法没有全能冠军:哪类划算,取决于任务里 OCR、小目标、短事件的占比和系统瓶颈在哪一段——后面实验矩阵和分段计时就量这个。

### 5.2 成本位置:在哪儿删,决定省哪段钱

删 token 像退货:货过了哪道工序,那道工序的钱就退不回来了。

若在 vision encoder 后剪枝,只节省 connector 和 LLM prefill;视频解码和 vision tower 已经算完,相关耗时一毫秒都不会降。若在 patch embedding 前降帧/降 patch,能节省更多计算,却可能错过短事件。

报告必须分别列出 `decode_ms`、`vision_ms`、`reducer_ms`、`llm_prefill_ms` 和 `decode_generation_ms`,不能只给端到端总时长。分段一摆账就清楚:只在 LLM 前剪枝的方法,`vision_ms` 若也降了,是测量或实现有错。

### 5.3 EVS:按"变没变"给 token 打分

静止画面里,逐帧编码得到的大多数 token 在重复说同一句话。EVS 的假设:同一空间位置,特征相对上一帧变化越大,越可能携带新信息。

对相邻帧同一空间位置,变化分数定义为

`change[t,p] = 1 - cosine(z[t,p], z[t-1,p])`

`change` 越大,表示同一位置的特征相对上一帧变化越明显。每个时间窗先保留固定数量的 anchor(锚点:不参与打分、无条件保留的 token),再按 `change` 从高到低选择剩余 token。第一帧没有前驱,必须使用单独的 anchor 规则。

**位置纪律。** 保留 token 必须携带原始 `(t,y,x)` position。gather(按索引把保留 token 收拢成短序列)后只允许按原始序列位置排序,不能把稀疏时间重新编号为连续时间——删掉第 3、4 帧后,第 5 帧还得叫第 5 帧,否则模型看到的是被悄悄加速的假视频,第 09 课建立的真实时间轴就毁在 reducer 手里。

实现见步骤 4 和第 11 节伪代码。单测应检查删除中间帧后,后续 token 的时间坐标保持不变。类比失效处:EVS 打的是"变化"分,对"重要"一无所知——匀速摇镜头让所有位置都高分(运动噪声),静止的字幕反而低分;失败诊断表里"EVS 只在静态视频好"的根源就在这。

### 5.4 token mass 与 proportional attention:合并必须留账本

四个 token 合成一个后,softmax 归一化只把它当一票,被合并区域的影响力就被稀释了。

解法是给每个视觉 key 的 attention 得分加 $\log m_k$($m_k$ 是该 key 的 token mass)。因为 $\exp(s+\log m)=m\cdot\exp(s)$,这等价于在 softmax 归一化里把该 key 的权重乘上 $m$:代表 4 个原始 token 的合并 token 恰好拿回 4 份权重。文本和音频 key 的 mass 为 1、附加项为 0,非视觉路径不受影响。叫法沿用 ToMe,称 proportional attention;精确公式在步骤 3。

mass 由 reducer 输出(第 7 节接口),经固定 connector 传入 Thinker attention(步骤 3)。B 臂若在低保留率突然退化,第一嫌疑就是 mass 没传进 attention——合并越狠、mass 越大,漏这一项伤得越重。

### 5.5 Pareto:三个目标冲突时怎么选默认配置

质量、token 数、延迟互相牵制,单指标排名没有意义;要找的是没被任何配置全面压过的那批点。

**机制(支配的精确定义)。** 配置 X 被 Y 支配,当 Y 的质量不低、token 不多、延迟不高,且至少一项严格更好。

**操作与验证。** 至少扫描 `100/75/50/25%` 四个保留率。对每个点计算是否被其他配置支配,再从 non-dominated frontier(未被支配点组成的边界)上选择默认策略;选哪个点还要过第 14 节的统计门槛。

## 6. MiniMind-O 当前机制与代码落点

当前实现里没有 reducer 的位置,下面每条都是要动的边界。

- `model/model_omni.py`:`MMVisionProjector` 对全部 encoder token 做同一 MLP;`count_vision_proj()` 将投影结果顺序注入占位符;没有 token score、merge weight 或 position-preserving gather。
- `dataset/omni_dataset.py`:`image_token_len=64` 决定固定占位数;没有变长视觉 token mask,也没有 frame/patch metadata。
- `model/model_minimind.py`:attention 接收完整输入序列;原生 1D position 依赖 token 顺序。
- `trainer/train_sft_omni.py`:collate 按 tensor 拼接,尚未按压缩后的变长 token packing;loss 也未记录 reducer 统计。
- `eval_omni.py`:需要新增逐样本 token、各阶段 latency 和保留热图。

当前单图固定为 64 token,压缩空间很小,还容易得出"压缩无害"的假象。主结果必须来自高分辨率多 tile 或多帧视频;只在 64-token 单图上得到的结果不得外推到视频。

## 7. Reducer 的输入、输出和限制

reducer 接收 encoder features `z[B,F,P,D]`(batch、帧、每帧 patch、特征维)、`position[B,F,P,3]` 和 `valid_mask`。它按 `mode` 和整数 `budget` 输出 `reduced_z`、`original_position_ids`、`token_mass`、`keep_mask` 和 trace。固定 connector 随后把结果送入 Thinker;合并臂还对视觉 key 使用 `log(token_mass)`。

统一接口:

```python
class VisualTokenReducer:
    def forward(self, z, position_ids, valid_mask, budget, training):
        return {
            "embeds": reduced,
            "position_ids": original_pos,
            "token_mass": token_mass,
            "keep_mask": keep_mask,
            "trace": {
                "score": score,
                "keep_indices": idx,
                "merge_pairs": merge_pairs,
            },
        }
```

A 和 C 没有发生合并,所有保留 token 的 `token_mass=1`。B 必须把 mass 继续传到 Thinker,不能算完合并后丢弃(原因见 5.4 节)。A、C 的 `merge_pairs` 是空列表。

reducer 的输入仅限视觉特征、位置、有效 mask 和预算。若读取答案 token、future labels 或 decoder attention,训练或评测就发生 label leakage(标签泄漏:算法偷看了不该看的答案信息)。接口测试应主动检查这些字段没有传入 reducer。

## 8. 数据 recipe

### 8.1 Schema

```yaml
id: video_001
media_ref: videos/001.mp4
media_sha256: "..."
question: "红灯亮起后，人物做了什么？"
answer: "停下"
task: transient_temporal
events:
  - span_ms: [4120, 4480]
    label: red_light_on
motion_score: 0.37
ocr_regions:
  - frame_ms: 4200
    bbox_xyxy: [120, 80, 240, 120]
source: owned
license:
  media: CC-BY-4.0
  annotation: CC-BY-4.0
  redistributable: true
split: test
```

对外部视频分别记录 media/annotation license;不清楚时标 `unknown`、禁止再分发,只发布 URL/hash/脚本。

### 8.2 组成与切分

- 20% 静态/低运动;20% 中运动;20% 高运动;15% 短暂事件;15% OCR/小目标;10% 音视频任务。
- 按原视频 identity 分组切分;同视频不同 clip 不跨 split;运动强度分层后切分;test 保留未见时长和 fps。
- 生成随机、均匀、EVS 的 keep mask 时,使用同一原始 feature cache 和同一整数 budget。

### 8.3 三档规模

| 档位 | clips/images | 作用 | 训练 |
|---|---:|---|---|
| pilot | 2k | 纯推理 reduction、验证 position | 0 或 500 steps |
| standard | 50k | stochastic uptraining | 10–20k steps |
| full | 300k+ | 多时长、多分辨率 | 定额 200M+ tokens |

## 9. 实施步骤

### 步骤 1：建立无压缩 trace

先证明"不删"这条路径一个数都不差。保存每个 case 的 encoder token、原位置、attention mask、答案、各阶段 latency。对 `budget=100%` 断言:输出 embedding 和 position 原样通过、`token_mass` 全为 1、attention 附加项全为 0,并确认 reducer 输出 logits 与原路径在混合精度容差内一致。这一步不过,后面分不清退化是压缩的锅还是插 reducer 本身的锅。

### 步骤 2：实现 uniform stratified sampling 基线

把每个时间窗和空间象限视为 stratum(分层抽样里的"层"),以 largest-remainder 规则(最大余数法:整数部分先分,剩的名额按余数大小发)分配精确整数 budget;每个 stratum 按原序列等距取样,最终按 `original_position_id` 排序。它是唯一的 A 臂算法。`2×2 spatial pooling` 和 temporal mean pooling 仅可在扩展实验中预研,不得进入 A。

### 步骤 3：实现 similarity merge

正式 B 臂使用下面这套固定规则,不允许在不同 seed 或保留率下临时换 matching 算法。规则连浮点误差和平手排序都写死,为的是 merge trace 能逐步重放;排序抖一下,B 臂就不可复现。

先用原始离散坐标划分局部窗口:
`window_id = (floor(t / 2), floor(y / 2), floor(x / 2))`。

也就是一个窗口最多覆盖 `2 帧 × 2 行 × 2 列` 的 8 个 token。不同窗口之间禁止配对,防止把画面两端"长得像"的远距离 token 捏在一起。设全样本目标 token 数为 `budget`,先给每个非空窗口保留 1 个名额,剩余名额按各窗口的 `token_count - 1` 使用 largest-remainder 分配;余数相同时按 `window_id` 从小到大分配。这样每个窗口都得到确定的目标数 `budget_w`,且所有窗口的目标数之和严格等于 `budget`。若 `budget` 小于非空窗口数,配置无效并在 preflight 失败,不能静默跨窗口合并。

每个窗口独立执行迭代配对:

1. 对当前 token embedding 做 L2 normalize(缩放到单位长度),计算所有无序 token pair 的 cosine similarity;
2. 把 similarity 乘 `1e6` 后四舍五入为整数 `sim_q`,避免极小浮点误差改变排序;
3. 候选 pair 按 `(-sim_q, min(original_position_id), max(original_position_id))` 排序;
4. 从前往后选择互不共享 token 的 pair。本轮最多选择 `min(current_count - budget_w, floor(current_count / 2))` 对;
5. 合并选中的 pair,未选 token 原样进入下一轮;重复到 `current_count == budget_w`。

若 token `i、j` 的 mass 分别为 `m_i、m_j`,合并结果为:

$$
m_{ij}=m_i+m_j,\qquad
z_{ij}=\frac{m_i z_i+m_j z_j}{m_{ij}}.
$$

位置不取平均。mass 较大的输入贡献 `original_position_id`;mass 相同时取较小的原始位置。最终所有窗口输出再按该位置升序排列。trace 必须保存每轮候选排序、选中的 pair、成员原始位置和合并后的 mass。

Thinker attention 对视觉 key 加上 `log(token_mass)`:

$$
a_{qk}=\frac{q^\top k}{\sqrt d}+\log m_k.
$$

文本和音频 key 的 mass 固定为 1,因此附加项为 0。这个处理称为 proportional attention:一个代表四个原始 token 的合并 token,在 attention 归一化时仍保留相应权重(推导见 5.4)。若没有实现这项传递,B 臂不得标记为完成。配对索引在 `no_grad` 下计算,mass-weighted mean 仍参与反向传播——选谁配对是离散决定不回传梯度,加权平均是连续计算照常回传。

### 步骤 4：实现 EVS

每帧保留:

- 1 个 global/CLS anchor(encoder 的全局汇总 token);
- 每个 `K` 帧窗口的最低配额;
- change score top-k;
- OCR/边缘保护可作为扩展,不进入主臂。

第一帧无前驱,按 anchor 策略保留。窗口最低配额是保险:哪怕画面完全静止,每个时间窗也留底,不至于整段消失。

### 步骤 5：加入同 budget 负对照

`random` 使用固定 seed;`uniform` 在时间和空间均匀采样;三者 keep count 必须逐样本完全相等,而非仅平均相同——平均相同但逐样本不同,算法就能在难样本上偷偷多留 token。

### 步骤 6：区分 inference-only 与 uptraining

先把相同 pretrained checkpoint 直接压缩(即插即用);再用随机保留率 `[0.25,1.0]` uptrain。两组结果分别报告,不能把训练补偿归因于算法本身。

### 步骤 7：实现动态 packing

collate 返回 offsets/lengths,避免删掉的 token 又以 padding 形式送入 LLM——删 75% 再补 75% 的 padding,一分钱没省。记录 `useful_tokens / allocated_tokens`。

### 步骤 8：做分层评测与 Pareto

对保留率 `100/75/50/25%`、运动强度 quartile(四分位分桶)、事件时长 bucket 分别输出质量和系统指标,计算 non-dominated frontier。

## 10. 三个对照实验组与公平控制

| 臂 | 方法 | 训练 |
|---|---|---|
| A | uniform stratified sampling | inference-only + 同预算 uptrain |
| B | similarity merge | inference-only + 同预算 uptrain |
| C | EVS change pruning | inference-only + 同预算 uptrain |

无压缩 100% 是共同 reference;random pruning 是负对照,不作为额外训练臂。

公平控制:同 encoder features、抽帧、原始 position、最终 token 整数 budget、训练 token、optimizer、seed、backbone 和生成参数;uptraining 参数量相同;不能给某臂额外 OCR loss。任何一项不同,臂间差异就有第二种解释。

## 11. 配置与伪代码

```yaml
experiment: exp10_token_reduction
model_lock: manifests/base_model.lock.json
video_frontend_lock: manifests/video_frontend.lock.json
run_lock: manifests/run.lock.json
reducer:
  keep_ratios: [1.0, 0.75, 0.5, 0.25]
  preserve_original_position: true
  stochastic_uptraining: true
  arms:
    A:
      type: uniform_stratified_sampling
    B:
      type: deterministic_similarity_merge
      local_window_tyx: [2, 2, 2]
      similarity: cosine
      similarity_quantization: 1000000
      pair_selection: iterative_disjoint_greedy
      pair_tie_break: [similarity_desc, min_position_asc, max_position_asc]
      representation: mass_weighted_mean
      position_rule: larger_mass_then_lower_position
      attention_mass_bias: log_mass
    C:
      type: evs
      score: adjacent_frame_cosine_delta
      window_frames: 4
      min_tokens_per_window: 8
train:
  tokens: 120_000_000
  trainable: [reducer, thinker_lora]
  seeds: [17, 23, 41]
system_eval:
  batch_sizes: [1, 8]
  warmup: 20
  repetitions: 100
```

preflight 在 `run.lock.json` 中写入 model/frontend lock 与三个 arm config 的实际 SHA-256;训练入口逐项重算,发现不一致、浮动 Hub ref 或 checkpoint 别名时必须拒绝启动。A/B/C 必须引用同一份两个 lock 文件。

Similarity merge 的参考伪代码:

```python
targets = largest_remainder_window_budgets(
    position_ids,
    total_budget=budget,
    window_shape=(2, 2, 2),
    reserve_one_per_nonempty_window=True,
)
for window_id in sorted(targets):
    tokens = tokens_in_window(z, position_ids, window_id)
    while len(tokens) > targets[window_id]:
        pairs = all_unordered_pairs(tokens)
        pairs = sort_pairs(
            pairs,
            key=lambda p: (
                -round(cosine(p.left.z, p.right.z) * 1_000_000),
                min(p.left.position, p.right.position),
                max(p.left.position, p.right.position),
            ),
        )
        count = min(
            len(tokens) - targets[window_id],
            len(tokens) // 2,
        )
        chosen = first_disjoint_pairs(pairs, count)
        tokens = mass_weighted_merge(tokens, chosen)

merged = sort_by_representative_position(concat_all_windows())
visual_key_bias = log(merged.token_mass)
```

EVS 的参考伪代码:

```python
delta = 1 - cosine(z[:, 1:], z[:, :-1])
score = pad_first_frame(delta)
score = apply_valid_mask(score)
anchor = choose_window_anchors(position_ids)
idx = exact_budget_topk(score, budget, force_keep=anchor)
idx = sort_by_original_sequence_position(idx)
return gather(z, idx), gather(position_ids, idx), idx
```

## 12. 训练预算与 8 卡执行

- **Pilot**:1×24GB,feature cache,2k cases,先做 inference-only;确保 100% 路径数值一致。
- **Standard**:2–4×24GB,50k clips、120M tokens,reducer + LoRA(低秩适配:只训小矩阵,省显存),三 seed。
- **8 卡并行建议**:每两卡一个保留率/seed,优先并行 sweep;若跨 8 卡,固定 global useful tokens 而非 padded tokens。
- **8×48/80GB**:可以在线跑 vision encoder,并同时测 encoder 前 sampling;仍需与 LLM 前 reduction 分开报告。
- PCIe-only 节点优先本地 NVMe feature cache;cache key 必须含媒体 hash、抽帧配置、encoder revision、dtype,缺一项就可能命中别的实验的特征。
- 记录 reducer kernel 是否引入 CPU/GPU 同步;高理论 FLOPs 节省可能被 gather/sort 的开销抵消——"FLOPs 降了延迟不降"的常见来源。

## 13. 指标与测量方法

### 能力指标

- 自然图 VQA;OCR ANLS(基于编辑距离的相似度得分)/字符召回;小目标 recall;空间关系 accuracy;时序 QA;transient-event recall;AV QA;文本/音频回归。
- 报 `retained_quality = compressed_score / baseline_score`,同时保留绝对分数,防止低基线掩盖问题。

### 系统指标

- 原始/保留/实际分配 token;vision FLOPs 与 LLM prefill FLOPs;decode、vision、reducer、prefill、generation 分段 p50/p95;peak HBM;tokens/s;padding ratio。

### 测量纪律

- CUDA event + 同步;预热 20、测 100;batch=1 和吞吐 batch 都测;固定 GPU clocks/软件版本(能固定时);按视频长度和运动 quartile 分桶;3 seed,报告 bootstrap CI(自助重采样置信区间)。

## 14. 验收条件

1. 在预注册的保留率上报告综合能力相对 baseline 的变化;
2. 相同 budget 下与 random、uniform 做逐类比较,特别检查短事件和 OCR;
3. 报告 LLM 实际输入 token 的绝对值与下降比例;
4. 分别报告端到端、vision encoder、prefill 和 decode 延迟;
5. 单独报告高运动 quartile 与短事件召回,不用宏平均遮蔽;
6. 检查文本与音频回归;
7. 证明 100% reducer 路径与原路径 logits 一致;
8. 输出 non-dominated Pareto 与逐类退化点。

主结论固定在 50% 保留率判断,其他保留率用于画曲线。对三个 seed 的同一批 case 做 10,000 次两层配对 bootstrap:先重采样 seed,再重采样 case;系统指标则对同一 case、同一重复轮次配对。只有一个候选臂同时满足以下条件,才能把它设为默认 reducer:

1. 相对 100% 无压缩 reference,综合能力差值的 95% CI 下界不低于 -2 个百分点;
2. 相对 A 的 uniform sampling,OCR 与短暂事件 macro accuracy 至少提高 2 个百分点,且逐 case 配对 95% CI 下界高于 0;
3. 相对 100% reference,LLM prefill p95 至少降低 25%,且配对 95% CI 下界高于 15%;
4. text/audio golden 相对 100% reference 的 95% CI 下界不低于 -1 个百分点;
5. 该点位于质量—token—延迟的 non-dominated frontier。

B 与 C 都通过时,选择预注册顺序中系统成本更低的一臂;两者都未通过时,保留 A 或 100% reference,并报告有效负结果——"聪明方法没赢"也是合格产出。阈值、case、重复轮次和统计脚本 hash 必须在打开 test 前写入 `stats_plan.yaml`,不能根据 test 修改。

若未达到目标,报告仍需指出瓶颈所在阶段、不能压缩的 token 类型,以及不适合默认 reduction 的视频条件。

## 15. 失败诊断表

调试从最便宜的检查开始:先看分段计时和 keep mask。

| 现象 | 原因候选 | 诊断与修复 |
|---|---|---|
| token 降、延迟不降 | vision/decode-bound 或 gather 开销 | 分段 profile,融合 kernel |
| EVS 只在静态视频好 | change score 偏向运动噪声 | motion quartile 审计、加窗口配额 |
| OCR 首先明显退化 | similarity merge 删除文字 token | OCR case heatmap,保留局部 anchor |
| B 在低保留率突然退化 | token mass 未传入 attention,或远距离 token 被误合并 | 检查 `log_mass` bias 与 window trace |
| 短事件消失 | 均匀窗口过大 | 降窗口、event-aware sampling |
| position 错乱 | gather 后重新编号 | 保留 original position id |
| 100% logits 不一致 | reducer 改顺序/dtype | identity 单测 |
| uptrain 才有效 | checkpoint 未适应缺 token | 分开陈述算法与适应收益 |
| padding 未下降 | collate 仍补回最大长度 | packed offsets/varlen attention |
| 总均分高但困难集明显退化 | 静态样本占比过大 | 分任务 macro average |
| random 偶尔更好 | seed/整数 budget 不一致 | 固定逐样本 keep count |

## 16. 逐 case 要求

至少 100 case:15 OCR、15 小目标、15 空间、15 静态、15 高运动、15 短事件、10 AV。每个 case 展示原视频 contact sheet(缩略帧拼板)、score heatmap、三臂 keep mask、B 臂每轮 merge pair 与最终 token mass、保留 token 数、正常答案、各臂答案、分段 latency、错误标签。

错误标签固定为:`sampling_miss`、`pruning_information_loss`、`merge_collision`、`position_corruption`、`reasoning_miss`、`system_no_speedup`。至少展示 20 个失败 case,不得只挑成功例。

## 17. 交付物

```text
exp10/
  configs/{uniform,merge,evs}.yaml
  manifests/base_model.lock.json
  manifests/video_frontend.lock.json
  manifests/run.lock.json
  manifests/arm_config_hashes.json
  data/manifest.jsonl
  data/license_report.md
  caches/cache_manifest.jsonl
  metrics/{aggregate,per_case,latency}.jsonl
  traces/keep_masks/
  plots/pareto_quality_token_latency.png
  cases/index.md
  report.md
```

报告必须写明 token 在 encoder 前还是 LLM 前被删除、实际下降的是哪一段耗时、最先退化的 case 类型,以及推荐的默认保留率和 fallback 条件。

## 18. 复现清单

- [ ] preflight 已冻结 Git commit、checkpoint repo revision 与 checkpoint SHA-256;
- [ ] 三臂引用完全相同的 model/frontend lock SHA-256;
- [ ] config 内没有 `winner-or-baseline`、浮动 ref 或未展开 SHA;
- [ ] 媒体、feature hash 与抽帧方式已固定;
- [ ] 逐样本整数 budget 已固定;
- [ ] A 仅使用 uniform stratified sampling;
- [ ] reducer 保留原 position;
- [ ] B 的 local window、similarity 量化、pair 排序和 tie-break 已写入 config hash;
- [ ] B 的最终 token 数逐样本精确等于 budget,merge trace 可重放;
- [ ] B 的 token mass 已进入 Thinker attention,文本/音频的 mass 固定为 1;
- [ ] 100% identity test 通过;
- [ ] random seed 已固定;
- [ ] inference-only 与 uptrain 结论分开;
- [ ] 三臂训练 token 相同;
- [ ] batch=1 与吞吐 batch 均已测量;
- [ ] latency 已分段;
- [ ] 结果已按运动 quartile 与事件时长分桶;
- [ ] 困难 case 没有被宏平均遮蔽;
- [ ] Pareto 已输出;
- [ ] checkpoint 恢复后已复测。

## 19. 前沿对照与改造方向

本课三个臂各有前沿出处。C 臂的 EVS 来自 [Nemotron 3 Nano Omni](https://arxiv.org/abs/2604.24954):生产级系统同样按帧间变化剪视频 token,同样强调 position 保留和随机保留率 uptraining——本课是它的缩小复刻。B 臂的合并加 proportional attention 来自 [Token Merging](https://arxiv.org/abs/2210.09461):它证明训练好的 ViT 不重训也能合并相似 token,log-mass 补偿就是它提出的。[FastV](https://arxiv.org/abs/2403.06764) 从另一个位置下刀:它发现视觉 token 的 attention 在 LLM 深层迅速变稀,于是在模型内部按 attention score 即插即用剪枝——成本位置和本课"LLM 前"不同,正好对照 5.2 节的账。而这一切的前提"视频高度冗余",[VideoMAE](https://arxiv.org/abs/2203.12602) 用极高 mask ratio 的预训练早就给出了证据。

帧数、分辨率、encoder 大小、数据时长是规模问题,加钱能解决。机制差距才是本课要补的:上游实现固定 64 占位、没有变长 packing、没有 position-preserving gather、attention 里没有 mass 的位置——这些不改,给再多卡也压不了视频。另一个机制差距是压缩位置:本课主臂只在 LLM 前动手,前沿系统还会在 encoder 内部逐层做(ToMe 的原始形态),那部分收益要靠 8×48/80GB 的扩展实验去摸。



1. **FastV 式深层剪枝对照臂。** 在 `model/model_minimind.py` 的 Thinker 第 k 层后按 attention score 剪视觉 token,与 LLM 前剪枝的 C 臂做同预算对比。改动位置:attention forward 加一个可选的层内 keep mask,reducer 接口不动。预算:inference-only,pilot 2k cases + feature cache,1×24GB 数个 GPU-hour。预期:深层剪枝质量可能更好(打分看过视觉与文本的浅层交互),但 `llm_prefill_ms` 降幅小于 C 臂——前 k 层计算照付。失败判定:两者 prefill 降幅相同,说明分段计时没测对,回步骤 1 查打点。
2. **pooling 预研臂。** 把步骤 2 里禁止进 A 臂的 `2×2 spatial pooling` 和 temporal mean pooling 做成扩展臂,与 A 同预算对比。改动位置:reducer 新增 mode,接口与 trace 复用。预算:inference-only,pilot 2k cases,1×24GB 数个 GPU-hour。预期:总均分与 A 接近,但 OCR 和小目标桶明显更差(5.1 节 pooling 价签的实证)。失败判定:pooling 在 OCR 桶不输 A——先怀疑 case 太简单,回数据 recipe 查 `ocr_regions` 覆盖和难度。

ToMe 的核心结论"同预算下 merge 比 prune 丢的信息少,且可免训练使用"可以映射到本课:B 臂对 C 臂(以及 random 负对照)在 50% 保留率 inference-only 的逐桶比较,重点看 OCR 和小目标。预期能复现同方向趋势(B 的 `retained_quality` 更高)。注意口径:本课 C 臂按帧间变化打分,和 ToMe 对照的朴素 prune 不同;若 C 在高运动桶反超 B,不算推翻原结论,只是打分信号和数据分布换了。

**更多扩展题:**

1. 学习式 budget controller:按问题和视频动态决定保留率;
2. OCR-safe EVS:把局部文字置信作为保护信号,但必须另设对照;
3. 在 vision encoder 中层剪枝,比较真实 encoder 节省;
4. 为实时视频建立 streaming reducer,只使用过去帧;
5. 将 winner 固定后进入长视频/长上下文课([第 14 课](14_long_context_curriculum.md)),不再联合调参。

## 20. 论文精读与问题

1. [Token Merging: Your ViT But Faster](https://arxiv.org/abs/2210.09461):精读 matching、proportional attention、训练与推理迁移。带着问题:它的 bipartite matching 和本课步骤 3 的确定性窗口配对差在哪,本课为什么宁可牺牲一点合并质量也要换可重放的 trace?阅读检查:比较 merge 与 prune 丢失信息的方式,对照你自己 B/C 臂在 OCR 桶的实测差距。
2. [FastV](https://arxiv.org/abs/2403.06764):精读视觉 token 冗余出现层位与即插即用剪枝。带着问题:在 LLM 第 k 层剪枝,哪些计算已经付掉了?阅读检查:列出在 LLM 深层剪枝仍需支付的 prefill 计算,和 5.2 节的成本位置账对上。
3. [Nemotron 3 Nano Omni](https://arxiv.org/abs/2604.24954):精读 EVS、position 保留和 stochastic uptraining。带着问题:为什么它坚持保留原始 position、坚持用随机保留率训练?阅读检查:列出 EVS 的静态 patch 假设及失效条件,对照失败诊断表里"EVS 只在静态视频好"那一行。
4. [VideoMAE](https://arxiv.org/abs/2203.12602):精读视频冗余与高 mask ratio。带着问题:预训练时能 mask 掉绝大部分内容还学得好,是否意味着推理时也能删同样多?阅读检查:区分预训练 masking 与推理 pruning 的目标和信息条件——前者有重建 loss 逼模型脑补,后者删了就真没了。
5. [MiniMind-O 官方仓库](https://github.com/jingyaogong/minimind-o):找出固定 64-token 假设及注入顺序。阅读检查:列出支持变长 token 需要修改的数据、模型和推理边界,和第 6 节的代码落点表逐格对上。

读完材料回头看,第三幕在这里收官。系统的眼睛完整了:第 08 课按图片本来的分辨率看图,第 09 课把视频当带时间轴的整体来理解、声画对得上,本课给这双眼睛装上预算意识——看视频先算 token 账,50% 保留率下质量掉多少、prefill 快多少,Pareto 图上白纸黑字。加上第一幕搞懂的听说链路、第二幕的流式与插话,这个 26M 的小系统已经能听、能说、能被打断、能看图看视频。但内核还是第 01 课那套 8 层 dense Thinker:每个 token 都要走全部参数,attention、FFN、位置编码全是教科书原型。第四幕换现代内核:[第 11 课](11_tiny_moe.md)先把 FFN 换成稀疏专家(MoE)——总参数变大、每 token 计算量不变,用同样的算力买更大容量;它从官方 dense checkpoint 独立起步,而你在本课练出的"同预算对照 + 分段计时 + 预注册验收"这套手法,会原样跟着你进下一幕。
