---
id: 49_video_generation
title: "视频生成拆账"
summary: "生成未来帧的训练目标和理解 caption 的交叉熵能否共用有效位置？物体永久性、相机轨迹与 Video-MME 为什么不是同一张表？"
unit: gen-native
play_tools: []
checkpoints:
  - "写出理解 CE 与生成帧差（或 v-prediction / flow）的有效位置，并证明两张 mask 不相交。"
  - "造出理解答对、生成帧物体消失的夹具，且不把 Video-MME 准确率写成生成质量。"
  - "对照 CogVideoX 的 3D VAE 与专家 Transformer、HunyuanVideo 的双流到单流与 14 类相机标注，不编造 Sora 层数或参数量。"
  - "把物体永久性、相机可控、VBench 动态度与 Video-MME 分列记账。"
---

# 第 49 课：把视频生成和视频理解拆开记账

> 内容：视频理解交叉熵与视频生成帧差分账、物体永久性探针、CogVideoX 专家 Transformer 与 HunyuanVideo 公开架构<br>
> 建议周期：阅读约 80 分钟；浏览器 Lab 一次验收约 12 分钟；CPU 夹具数分钟<br>
> 硬件：无 GPU 可完成本课阅读、教学模拟与 CPU 机制实验。若要加载公开权重：CogVideoX-2B 论文表 7 在 H800、50 步、480×720、6 秒约 18GB；5B 同规格约 26GB。HunyuanVideo 1.5 在卸载后 720p、121 帧峰值约 13.6GB。本课不要求加载这些权重<br>
> 产物：两张不相交的 loss mask、理解答对且生成帧杯子消失的夹具、生成账与 Video-MME 分列的协议卡<br>
> 独立性：需要[第 09 课](09_native_video.md)的时间轴概念，不重做 TMRoPE；需要[第 10 课](10_video_token_reduction.md)知道理解侧 token 会爆，不重做压缩 Pareto；需要[第 20 课](20_unified_understanding_generation.md)知道图像理解和生成已经分路过；需要[第 33 课](33_world_model_latent.md)知道像素 $L_2$ 会吞接触，不重做 JEPA；需要[第 47 课](47_eval_taxonomy.md)把 Video-MME 留在 C2。无 GPU 时不要把 CPU 数字写成 VBench 或 Sora 能力

## 1. 生成下一帧和看懂这一段差在哪

桌上有一只杯子。五帧监控里它一直在。问模型“杯子还在不在”，答“还在”，交叉熵（cross-entropy，CE：对正确答案编号打的负对数概率）可以很低。再让同一段视频去生成第六帧：把每个格子往 0.5 的均值里抹，杯子占用掉到 0，整帧 $L_2$ 却可以比“把第五帧抄过来”更低。理解过关，物体在下一帧消失。这不是同一笔账算错了小数，是两笔账写在序列的不同位置上。

[第 09 课](09_native_video.md)把视频建成带真实毫秒的对象，抽帧、时序融合、音视频交错都为了**看懂**这一段。[第 10 课](10_video_token_reduction.md)把进 Thinker 的视觉 token 压到算力扛得住，刀落在理解前缀，不落在要画出来的未来帧。[第 20 课](20_unified_understanding_generation.md)在图像上已经分过语义路径和 VAE 路径；本课把那条纪律接到视频，样本从一张图换成一段短序列。[第 33 课](33_world_model_latent.md)比较像素 $L_2$ 和表征回归，本课不改 JEPA，不训 EMA 目标编码器，不把接触探针再当主验收。[第 34 课](34_world_model_platform.md)把物体永久性写成数据引擎的拒收条件；本课把它写成**生成账**上的探针，并且禁止用理解 CE 替这笔账。[第 47 课](47_eval_taxonomy.md)规定 Video-MME 是 C2 视频时序理解。本课要改的是记账：生成未来帧的训练目标，和看懂这一段的 caption CE，能不能共用有效位置。

答案先写死。不能共用。理解 CE 的有效集合 $M_{\mathrm{und}}$ 是答案文本位置。生成帧差、v-prediction 或 flow matching 的有效集合 $M_{\mathrm{gen}}$ 是未来视觉格子（或对应的 3D VAE latent）。两集合交必须为空。把它们 OR 成一张 mask，梯度会同时改“该答什么字”和“下一帧每个像素该多亮”，杯子消失可以藏进均值里，CE 还在降。

类比只服务这一处机制。值班员交的作业是填表：杯子还在桌上吗。画师交的作业是画出下一秒。填对表不保证画里还有杯子。类比失效处：值班员也会看错，画师也可能碰巧把杯子留下来。所以两张表都要留，位置还不能共用。失效处再补一句：本课的“画师”在夹具里是均值填充，不是 CogVideoX-5B；论文里的生成器会做 3D 注意力和专家 AdaLN，失败模式仍然要单独测物体永久，不能用值班员的准确率代替。

把 CPU 夹具的黄金样例抄进笔记。网格高 4、宽 8，可见段 $t=0,\ldots,4$，杯子在列 5、行 1 和行 2，杯子格子取值 0.08，背景纹理 $(17t+13i+7j)\bmod 10$。生成目标是 $t=5$。遗忘系数 $\mathrm{forget}=1$ 把下一帧填成 0.5。联合序列长度 197：历史像素 160，问句 4，答案 1，未来格子 32。CE 有效位置只有下标 164，帧差有效位置是 165 到 196。交为 0。历史占用 1.0，理解 $p(\text{还在})=0.916827$，caption CE $0.086836$。遗忘生成占用 0.0，帧差 $L_2=0.109598$，低于抄上一帧的 $0.243056$。谁改了杯子列却不改这组数，后面的 mask 对不上。

把六步再写成表，后面所有公式都代这组数。杯子列固定 5。可见段每一帧占用都是 1，因为杯子格子被写成 0.08。$t=5$ 的真值仍然占用 1，纹理已经换成 $t=5$ 的图案，所以抄 $t=4$ 会在 30 个非杯子格子上付出时间差。均值填充把 32 格都写成 0.5：两个杯子格误差 $(0.5-0.08)^2$，三十个背景格则靠近“不可预测纹理的均值”。

| $t$ | 角色 | 杯子格取值 | 占用 $o$ | 进哪张 mask |
|---|---|---|---|---|
| 0 | 可见历史 | 0.08 | 1 | 条件，不进损失 |
| 1 | 可见历史 | 0.08 | 1 | 条件，不进损失 |
| 2 | 可见历史 | 0.08 | 1 | 条件，不进损失 |
| 3 | 可见历史 | 0.08 | 1 | 条件，不进损失 |
| 4 | 最后可见 | 0.08 | 1 | 理解只读占用；生成当抄帧对照 |
| 5 真值 | 生成目标 | 0.08 | 1 | $M_{\mathrm{gen}}$ 的真值 |
| 5 均值 | 遗忘生成 | 0.50 | 0 | $M_{\mathrm{gen}}$ 的预测 |
| 5 抄帧 | 负对照 | 0.08 | 1 | 占用对、纹理错，$L_2$ 反而更高 |

再把同一表读成“理解对、生成消失”的最小版本。理解头不看 $t=5$，只看 $t=4$ 的占用。占用 1，argmax 为 YES，$p_1\approx 0.917$。生成头必须交 $t=5$。交均值时占用 0，交抄帧时占用 1 但 $L_2=0.243$，交均值 $L_2=0.110$。若你的验收只看 $L_2$，会给消失发奖。若你的验收只看 CE，会给已经没杯子的生成器发“看懂了”的奖。两本账必须同时写。Lab 把网格改成 $8\times 8$ 方便看，占用公式不变；CPU 报告只许抄 $4\times 8$ 的数。有人把 Lab 的 64 格写成 CPU 的 32 格，帧差位置数会对不上 `gen_positions=32`。

纹理函数不要改成常数。常数背景会让抄帧 $L_2$ 接近 0，均值再也赢不了抄帧，`mean_fill_l2_beats_copy_while_cup_vanishes` 会翻成假。那不是生成器突然学会了物体永久，是探针被关掉了。第 33 课用纹理强度 0.55 当门槛，本课把纹理写死在取模函数里，不提供滑条去关它。Lab 的 forget 滑条只混合最后一帧和 0.5，关不掉时间纹理；把“可见历史里有没有杯子”拨到没有，理解会先错，Gate 仍不亮。两条开关分工：杯子开关管理解是否该对，forget 管生成是否消失。验收要的是前者开、后者大。

本课术语：

| 术语 | 简要解释 |
|---|---|
| 理解 CE | 只对答案文本位置算的负对数似然，历史像素和问句通常标成跳过 |
| 生成帧差 | 预测下一帧与真帧在有效格子上的均方误差；教学夹具用它代表像素生成账 |
| v-prediction | CogVideoX 采用的扩散目标：预测速度混合 $v$，而不是直接预测噪声 $\epsilon$ |
| flow matching | HunyuanVideo 采用的速度场回归：在噪声与数据的直线路径上拟合 $u_t$ |
| 3D causal VAE | 在时间、高、宽上压缩视频的变分自编码器，卷积填充只落在过去，避免未来泄漏 |
| 专家 AdaLN | CogVideoX 对文本和视觉各用一套自适应 LayerNorm，时间步分别调制 |
| 物体永久性 | 已出现的物体不应无故从后续帧消失；本课用杯子占用 $o$ 测量 |
| 相机轨迹可控 | 生成视频是否服从指定的运镜；HunyuanVideo 在结构化 caption 里标了 14 类 |
| Video-MME | 短中长视频多项选择，第 47 课的 C2，测理解不测生成 |
| VBench | CogVideoX 用来报 Human Action、Dynamic Degree 等的生成基准 |

改对了的判据看四件可记录的事。$M_{\mathrm{und}}\cap M_{\mathrm{gen}}=\emptyset$。理解在杯子始终可见时答“还在”。生成帧在遗忘填充后占用低于 0.15。Video-MME 准确率不得写入生成账。浏览器 Lab 和 CPU 都是教学夹具。它们证明分账可复核，不能写成“我们复现了 CogVideoX-5B”或“Sora 已经解决物体永久”。Sora 的公开材料停留在 spacetime patch 与定性失败，其技术报告未公开层数、参数量和完整损失。

## 2. 本课解决的问题

当前系统（以及大量把“视频模型”写进同一张幻灯片的报告）默认验收是一个视频分。它解决不了六类失败，而且这六类会在总平均里被兑掉：

1. 把第 09 课的时序问答准确率写成“已经会生成下一帧”。
2. 把第 10 课的 token 保留率写成生成压缩比，刀其实落在理解前缀。
3. 把第 20 课的图像 flow 实验改个标题，样本仍是单张 VAE latent。
4. 把第 33 课的 JEPA 接触探针当成已经交了生成账。
5. 把 Video-MME 无字幕 75.0% 填进 VBench Dynamic Degree。
6. 把物体消失、相机不听指令、闪烁，全部兑成“画质还行”。

本课的改造范围只包括损失位置、两本账的协议卡，以及一小段可手算的杯子占用。不更换 MiniMind-O 的 SigLIP2，不重做第 10 课的 EVS 剪枝，不比较 2D+1D 注意力的工程加速是否值得（CogVideoX 的消融已经给出方向，本课只引用，不重跑），不把离散视频 token 的 next-token 再讲一遍（那是[第 41 课](41_discrete_any_to_any.md)的 Emu3 路线）。

以下结果不能支持“理解和生成已经统一到视频”：

- 只报一个联合 loss，不打印 $M_{\mathrm{und}}$ 和 $M_{\mathrm{gen}}$ 的下标；
- 理解答对，不报生成帧占用；
- 生成 $L_2$ 下降，不报占用是否掉到 0；
- 用 Video-MME 准确率给生成器打分；
- 把 HunyuanVideo 的 Text Alignment 61.8% 减 Video-MME 的 75.0%，得到“理解比生成强”；
- 为 Sora 补一层数或一个 VAE 压缩比。

执行顺序固定：先在 CPU 上证明两张 mask 不相交、杯子可以在 CE 过关时消失，再在浏览器里先预测再揭晓，最后如果有 GPU，才把同一套字段接到公开权重的日志上。没有 GPU 时，前两步已经构成完整交付。

和第 09、10、33、47 课的边界再钉死一次，避免四课抢题。第 09 课管理解侧时间轴：帧有毫秒、倒放必须改答案。本课不重做反事实考卷；倒放会改变“杯子是在门前还是门后掉”，那是理解题，生成题是“下一帧杯子还在不在格子里”。第 10 课管理解侧预算：进 Thinker 的 token 太多。本课的 8×8×4 是生成 VAE，删的是像素体积，不是 EVS 的 keep-mask。第 33 课管世界模型监督空间：像素 $L_2$ 对表征。本课的均值填充也会让 $L_2$ 变好、物体变差，但验收对象是 mask 与账本，不是 EMA 防塌缩。第 47 课管六类评测数字。本课在 C2 之外把生成账切开，不新增第七类去和 LIBERO 横比。四课的检查票要一起用：先问这是理解还是生成，再问有效位置，再问占用，再问类标签。

边界还有两处容易漏。[第 34 课](34_world_model_platform.md)已经把物体永久写成数据引擎拒收。本课若再把 ID 计数器当主实验，就抢 34 的题。本课只保留一个杯子占用，并且必须和理解 CE 同时报。[第 41 课](41_discrete_any_to_any.md)已经把理解 mask 看全图、生成 mask 挡住未来像素写成离散 token 的作业。本课生成侧是连续格子或 latent，没有码本偏移。谁在本课 CPU 里实现 VQ，属于把 41 的代码贴错课。

本课固定五个可证伪命题。它们是协议命题。协议命题可以用 CPU 夹具证伪；模型命题需要权重和数据，本课不做。

1. 存在一张联合序列，使得 $M_{\mathrm{und}}$ 与 $M_{\mathrm{gen}}$ 交为空，问句和历史像素都不进 CE，历史像素和答案 token 都不进帧差。
2. 历史 5 帧杯子占用为 1 时，理解 argmax 为“还在”，$p(\text{还在})>0.9$。
3. $\mathrm{forget}=1$ 的生成帧占用为 0，且其 $L_2$ 可以低于抄上一帧。
4. 把 $M_{\mathrm{und}}\cup M_{\mathrm{gen}}$ 当成共享 mask，有效位置数等于 1+32=33，必须判为非法合并。
5. `videomme_accuracy` 不得标成生成分数；`camera_type_match` 与 `videomme_accuracy` 不得记入同一本账。

五条都通过，只能说明分账被写成了可审计协议。它们不能说明 CogVideoX-5B 的 Multiple Objects 70.95 可以搬到你的监控摄像头，也不能说明 HunyuanVideo 的 Overall 41.3% 等于“已经会物理”。

## 3. 开始前需要准备什么

本课没有 MiniMind-O 训练步骤。开始前把上游事实和本课约定分开写进实验记录。

**上游事实（打开过的页面，不是口口相传）：**

- CogVideoX：[arXiv:2408.06072](https://arxiv.org/abs/2408.06072)，HTML：[v3](https://arxiv.org/html/2408.06072v3)。10 秒、16 fps、768×1360；3D causal VAE；专家 Transformer；2B 与 5B。
- HunyuanVideo：[arXiv:2412.03603](https://arxiv.org/abs/2412.03603)，HTML：[v6](https://arxiv.org/html/2412.03603v6)。13B；Causal 3D VAE；$c_t=4,c_s=8,C=16$；Flow Matching；60 名评估员。
- HunyuanVideo 1.5：[arXiv:2511.18870](https://arxiv.org/abs/2511.18870)，HTML：[v2](https://arxiv.org/html/2511.18870v2)。8.3B；空间 16×、时间 4×、通道 32；SSTA；两阶段超分到 1080p。
- Video-MME：[arXiv:2405.21075](https://arxiv.org/abs/2405.21075)。900 视频、2700 题；数字沿用[第 47 课](47_eval_taxonomy.md)。
- Sora 公开技术叙述：[Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/)。spacetime patch、压缩网络、扩散 Transformer、定性的物体永久与失败。层数未公开。

**课程约定：**

- CPU 网格 $4\times 8$，Lab 为了可读用 $8\times 8$，占用公式相同，不要把 Lab 的格子数抄进 CPU 报告；
- 答案词表大小为 2：`NO=0`，`YES=1`；夹具 logits 固定为 $(0.1,2.5)$；
- 生成教学损失用帧差 $L_2$，论文损失另行列出，禁止把夹具 $L_2$ 写成 CogVideoX 的 $L_{\mathrm{simple}}$；
- 隔离实现位于 `experiments/src/learn_omni_experiments/lessons/lesson_49.py`。登记前不要改 `lesson_id`。

硬件分层写进记录。读课文和 CPU：笔记本即可。跑 Lab：现代浏览器。复现 CogVideoX 论文表 7 的推理时间：H800、50 步。复现 HunyuanVideo 1.5 的 13.6GB：需要官方卸载与 tiling。没有这些卡时，把对应行标 `skipped-no-gpu`。空行表示范围已经声明。把空行填上别人的表值却不写对照来源，才算交付失败。

前置阅读若只选一篇，选 CogVideoX 第 2 节：VAE、专家 Transformer、3D 注意力三张图够本课机制。第二篇选 HunyuanVideo Table 3，用来练习生成账内部也不能只报 Visual。第三篇选 Video-MME 的协议，用来练习跨账拒绝。Sora 页面放在最后读，读完立刻把空格子画叉，防止印象里的“世界模拟器”把占用探针抬成定理。公开叙述自己写了 limitations，本课跟着写 limitations，不替它补架构。

依赖的代码能力只有三项：会构造 0/1 mask，会写 softmax，会算均方误差。不会 3D 卷积也可以完成本课。不会 FlashAttention 也可以。不会跑 `diffusers` 也可以。会跑的人把 Step 6 做了，不会跑的人把 Step 6 标跳过。跳过不是扣分，填假数字才是。

## 4. 完成后应具备的能力

完成后，拿到任意一段“视频模型很好”的声明，你应能完成以下检查：

1. 画出联合序列：历史像素、问句、答案、未来格子，标出 $M_{\mathrm{und}}$ 和 $M_{\mathrm{gen}}$；
2. 手算一组 logits 的 softmax 与 CE，指出问句位置为什么必须是 `-100`；
3. 用占用 $o$ 判断生成帧杯子还在不在，不把整帧 $L_2$ 当成占用；
4. 说明 CogVideoX 专家 AdaLN 解决的是模态尺度，不是理解 CE；
5. 说明 HunyuanVideo 的 14 类相机标注属于生成可控，不是 Video-MME 的选项命中；
6. 引用 Sora 时只写公开叙述，空格子写“未公开”；
7. 把 Video-MME、VBench、物体消失率、相机命中分成至少四列；
8. 在 Lab 里造出理解对、生成帧消失，并且先预测再揭晓；
9. 指出 2D+1D 注意力的裂头问题属于生成账内部，解决了也不自动过 CE；
10. 把 CogVideoX 密 caption 管线写成“理解服务于生成数据”，不写成“已经统一”。

十条里前八条没有 GPU 也能做完。后两条需要你打开论文图和附录，仍然不需要训练。有人把第 9 条理解成“必须实现 FlashAttention”，属于把生成工程问题当成开课条件。本课开课条件是会写 Python、知道 CE，以及读过第 09 课“视频不是一叠图片”那一句。没读第 10 课也可以：只要你答应不把 8×8×4 写成保留率。没读第 33 课也可以：只要你答应不把占用探针升级成 JEPA。没读第 47 课会比较痛，因为 Video-MME 的 75.0% 太容易抄进生成表；建议至少读第 47 课 C2 那一节。

## 5. 原理：边造边讲

十一个机制。每个按同一节奏：为什么需要、怎么运转、精确定义、公开落点、怎么证明做对了。前四节把账本和夹具钉死，中间四节对照公开生成器，后三节处理评测、Sora 空格子和梯度尺度。顺序不要倒：先会画 mask，再打开 13B。

### 5.1 联合序列上的两张 mask

为什么需要。视频理解的标准作业是：看见一段帧，写出或选出文字。视频生成的标准作业是：给定文本或首帧，在视觉格子上吐出未来。两件事可以读同一段媒体。训练时若共用一张 0/1 表，损失会把“该答哪一个字”和“每个像素差多少”加在一起。像素项通常多两个数量级（夹具里 32 对 1），CE 会被帧差淹没，或者反过来把未来格子拿去当分类目标。

怎么运转。把一条样本排成四段。

| 段 | 长度（夹具） | 下标 | 进 CE | 进帧差 |
|---|---|---|---|---|
| 历史像素 | 160 | 0–159 | 否 | 否 |
| 问句 | 4 | 160–163 | 否 | 否 |
| 答案 | 1 | 164 | 是 | 否 |
| 未来格子 | 32 | 165–196 | 否 | 是 |

历史像素给理解当条件，也给生成当条件，但历史本身不是要学的目标：抄历史会让模型变成帧延迟器。问句是指令，学模仿提问没有用，见[第 01 课](01_baseline_reproduction.md)的 loss mask。答案是理解目标。未来格子是生成目标。

数学。记联合下标集合为 $\{0,\ldots,L-1\}$，$L=197$。

$$
M_{\mathrm{und}}=\{164\},\qquad M_{\mathrm{gen}}=\{165,166,\ldots,196\}
$$

$$
M_{\mathrm{und}}\cap M_{\mathrm{gen}}=\emptyset
$$

非法合并写成

$$
M_{\mathrm{or}}=M_{\mathrm{und}}\cup M_{\mathrm{gen}}
$$

$|M_{\mathrm{or}}|=33$。CPU 里 `illegal_or_mask_covers_both_ledgers` 要求 OR 之后恰好盖住这两段，用来证明你没有漏算。

公开落点。Show-o / Emu3 在[第 41 课](41_discrete_any_to_any.md)已经把理解 mask 和生成 mask 分开；那是离散 token。本课的生成侧可以是连续格子。CogVideoX 的文本与视觉在序列维拼接，损失却写在扩散的视觉 latent 上，T5 文本是条件，不是 CE 目标。HunyuanVideo 同样：MLLM 文本是条件，Flow Matching 写在 3D VAE latent 上。不要看见“拼在同一序列”就以为 CE 和帧差共享有效位置。

把 $L=12$ 的玩具表先画出来，再放大到 197。玩具：历史 4 格、问句 2、答案 1、未来 5。行是损失名，列是下标，`1` 表示计损失。

| 损失 \ 下标 | 0–3 史 | 4–5 问 | 6 答 | 7–11 未来 |
|---|---|---|---|---|
| 理解 CE | 0000 | 00 | 1 | 00000 |
| 生成帧差 | 0000 | 00 | 0 | 11111 |
| 非法 OR | 0000 | 00 | 1 | 11111 |

三句断言。理解行只有答案是 1。生成行只有未来是 1。OR 行有 6 个 1，等于 1+5，不再是任何单一任务的有效位置。放大到夹具：史 160、问 4、答 1、未来 32，OR 为 33。`illegal_or_mask_covers_both_ledgers` 检查的就是第三行。有人用 `loss = ce + mse` 却把 `mse` 写在整段序列上，第三行会在历史上也出现 1，OR 长度不再是 33，夹具同样失败。失败有两种，不要混：交非空是串账；OR 比 33 大是把条件当目标。

验证。打印两个 mask 的非零下标。交必须为空。把答案下标 164 写进生成 mask，或把 165 写进 CE mask，夹具必须失败。Lab 揭晓后的两条色带：上面一条只在答案段亮，下面一条只在未来段亮。两条同时亮在同一段，就是 $M_{\mathrm{or}}$。色带宽度按 160:4:1:32 画，不要画成四段等宽，否则眼睛会以为答案和未来一样长，梯度尺度那一节就读不进去。

### 5.2 理解 CE：只问杯子还在不在

为什么需要。“看懂这一段”在本课收成一个可手算的问题，避免和 Video-MME 的 12 种题型抢篇幅。物体永久性的理解版是：可见历史上杯子占用高，答案应为“还在”。生成版是：下一帧占用是否还高。两版必须分开，否则你会用答对字来证明画出了杯子。

怎么运转。夹具不训练网络。它用最后一帧占用是否 $\ge 0.5$ 决定 argmax，再用固定 logits $(0.1,2.5)$ 算概率。这是教学用的理解头：条件已经足够，头不该犹豫。真实 VLM 会有幻觉，那是[第 23 课](23_grounding_ocr_spatial.md)的 POPE，本课不把幻觉率写成生成占用。

数学。词表 $\{0,1\}$，目标 $y=\mathrm{YES}=1$。

$$
p_k=\frac{\exp(z_k-\max_j z_j)}{\sum_j\exp(z_j-\max_j z_j)}
$$

$$
\mathcal{L}_{\mathrm{und}}=-\log p_{y}
$$

代入 $z=(0.1,2.5)$：$\exp(0.1-2.5)\approx 0.090717$，$\exp(0)=1$，分母 $\approx 1.090717$，$p_1\approx 0.916827$，

$$
\mathcal{L}_{\mathrm{und}}\approx 0.086836
$$

占用

$$
o(I)=\mathrm{clip}_{[0,1]}\frac{0.5-\mathrm{mean}(I_{\mathrm{cup}})}{0.5-0.08}
$$

杯子格子为 0.08 时 $o=1$。理解规则：$o(I_4)\ge 0.5$ 则答 YES。

公开落点。Video-MME 的 1 次正确是四选一命中，见第 47 课 C2，协议卡要写短中长和有无字幕。本课这道题连四选一都不是，它是二分类占用。不要把 $p_1=0.917$ 抄进 Video-MME 列。CogVLM2-Caption 在 CogVideoX 论文里是给**训练视频写密 caption** 的理解模型，用来改善生成数据，不是本课的验收模型。论文附录甚至把 CogVideoX 和 CogVLM2-Caption 串成 video-to-video。那是数据管线，不是“理解和生成已经共用 CE”。

手算占用，避免把均值亮度当成“还在”。杯子两格取值 0.08，背景约定 0.5：

$$
o=\frac{0.5-0.08}{0.5-0.08}=1
$$

均值填充后两格都是 0.5，$o=0$。若有人把占用改成“杯子格是否小于 0.5”，均值填充时 $0.5<0.5$ 为假，碰巧还能判消失；一旦 forget=0.4，杯子格 $0.08\times 0.6+0.5\times 0.4=0.248$，小于 0.5，会被判成还在，而连续占用

$$
o=\frac{0.5-0.248}{0.42}\approx 0.60
$$

仍明显高于 0.15 的消失门槛。夹具用连续占用，就是防止门槛和混合系数打架。Lab 的 Gate 用 0.25，CPU 用 0.15，方向相同、数值不同，报告里要写明哪一条门槛。不要把 0.25 抄进 `result.json`。

验证。`understand_answers_cup_still_there` 为真。把历史杯子关掉，$o$ 接近 0，argmax 必须变成 NO，Lab 的 Gate 不得通过：本课验收要的是理解对、生成错，两边都错或两边都对都算没造出分裂。logits 也不可改。改成 $(2.5,0.1)$ 会让 $p_1$ 掉到约 0.083，CE 升到约 2.49，理解不再“对”，命题 2 失败。改成 $(0,0)$ 则 $p_1=0.5$，也不满足 $>0.9$。夹具把 $(0.1,2.5)$ 写死，是为了让 CE 的数字可以手算，不是因为真实 VLM 有 0.917 的把握。

### 5.3 生成帧差：均值可以赢抄帧，同时把杯子抹掉

为什么需要。生成账最常见的偷换是：整帧 $L_2$ 降了，就说下一帧更像真的。桌布纹理每一帧都在变，夹具用 $(17t+13i+7j)\bmod 10$。抄上一帧会在纹理上付出大误差；把整帧涂成 0.5，纹理误差变小，杯子格子从 0.08 被拉到 0.5，占用掉到 0。最小化 $L_2$ 并不最大化 $o$。

怎么运转。教学生成器只有一个旋钮 $\mathrm{forget}\in[0,1]$：

$$
\hat I_5=(1-\mathrm{forget})\,I_4+\mathrm{forget}\cdot 0.5
$$

$\mathrm{forget}=0$ 抄最后可见帧，$o=1$。$\mathrm{forget}=1$ 纯均值，$o=0$。真值 $I_5$ 仍有杯子，纹理已经换成 $t=5$ 的图案。

数学。

$$
\mathcal{L}_{\mathrm{frame}}=\frac{1}{HW}\sum_{i,j}\bigl(\hat I_5(i,j)-I_5(i,j)\bigr)^{2}
$$

CPU 输出：抄帧 $L_2=0.243056$，均值 $L_2=0.109598$。均值更低，占用从 1 到 0。`mean_fill_l2_beats_copy_while_cup_vanishes` 把这三件事绑在一起：更低的 $L_2$、消失的杯子、抄帧仍然保住占用。

这一条和[第 33 课](33_world_model_latent.md)的像素 $L_2$ 吞接触是亲戚，不是同一张作业。第 33 课比较的是监督空间：像素对表征。本课比较的是任务账本：文字 CE 对未来格子。不要把 EMA 目标编码器抄进本课的 `run()`。

公开落点。CogVideoX 训练用 v-prediction 与 zero SNR，损失在 3D VAE latent 上，不是本课的像素 $L_2$。HunyuanVideo 用 Flow Matching 的速度 MSE，同样在 latent 上。夹具用像素 $L_2$ 是为了手算占用。把 $0.109598$ 写进论文表，属于第 01 课禁止的口径错配。

把 $L_2$ 拆成杯子格和背景格，避免只看一个标量。32 格里 2 个杯子、30 个背景。均值相对真值：杯子项 $2\times(0.42)^2/32\approx 0.0110$。背景项取决于 $t=5$ 纹理离 0.5 有多远，夹具给出总值 0.109598，所以背景大约贡献 0.098。抄帧相对真值：杯子项为 0（杯子格仍是 0.08），总值 0.243056 几乎全是 30 个背景格的时间纹理差。结论是：纹理差把抄帧罚重了，均值靠“猜平均纹理”拿回分数，顺便把杯子项的 0.011 也付掉。0.011 相对 0.243 可以忽略，这就是物体永久在裸 $L_2$ 里看不见的原因。若只监督杯子格，均值会立刻变差；本课不把生成损失改成只监督杯子，那是改造清单第 1 条。主课要先证明默认 $L_2$ 看不见杯子。

forget 连续变化时占用是直线 $o=1-\mathrm{forget}$（在杯子格与 0.5 的线性混合下）。0.75 对应占用 0.25，正好踩在 Lab 门槛上；0.80 对应 0.20，更稳。不要停在 0.70：占用 0.30，Gate 不亮。CPU 直接用 1.0，占用 0，省掉滑条。两边门槛不同必须写进记录，否则有人会说“CPU 和 Lab 数字打架”。

验证。Lab 把 $\mathrm{forget}$ 拖到 0.75 以上，生成占用低于 0.25，理解仍答“还在”。揭晓前生成帧是斜纹遮罩，防止先看消失再选预测。四个预测项里，“帧差更低就等于杯子还在”是专门打这一小节的。你若已经看见 0.110 对 0.243 仍选这一项，说明数字没有改信念，Gate 应当拒绝。信念改过来的标志是：承认更低的 $L_2$ 可以和更低的占用同时出现。

### 5.4 扩散、v-prediction 与 flow：生成账的三种写法

为什么需要。公开视频生成器几乎都不在像素上做裸 $L_2$。你仍要把它们记进**生成账**：有效位置是视觉 latent，条件是文本，不是 caption CE。三种写法的公式不同，账本相同。

怎么运转。CogVideoX 第 3.3 节先抄 DDPM 的噪声目标

$$
L_{\mathrm{simple}}(\theta):=\mathbf{E}_{t,x_0,\epsilon}\bigl\lVert\epsilon-\epsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,t)\bigr\rVert^{2}
$$

随后写明采用 v-prediction 与 zero SNR，噪声日程跟随 LDM。v-prediction 把目标从 $\epsilon$ 换成速度混合，训练更稳，它仍然对带噪视觉 latent 回归，不对答案 token 做分类。HunyuanVideo 第 4.5.1 节：数据 $x_1$，噪声 $x_0\sim\mathcal{N}(0,I)$，$t$ 从 logit-normal 抽样，直线插值得到 $x_t$，回归 $u_t=\mathrm{d}x_t/\mathrm{d}t$，

$$
\mathcal{L}_{\mathrm{generation}}=\mathbb{E}_{t,x_0,x_1}\lVert v_t-u_t\rVert^{2}
$$

推理用一阶 Euler。这和[第 20 课](20_unified_understanding_generation.md)、[第 28 课](28_flow_matching_vla.md)是同一族速度场，样本分别是图像 latent、动作块、视频 latent。公式眼熟，对象不能调包。

数学上还要钉住条件。设文本嵌入为 $c$。CogVideoX 把 $z_{\mathrm{text}}$ 与 $z_{\mathrm{vision}}$ 在序列维拼接，损失只回传到视觉输出再 unpatchify。HunyuanVideo 双流阶段文本和视频各走各的块，单流阶段拼接后融合；损失仍是速度场，不是文本 CE。若有人把 T5 或 MLLM 的 next-token CE 加进总损失，必须另开 $M_{\mathrm{text}}$，并且声明它是 captioner 或 prompt 改写器，不是视频生成器本体。

公开落点。CogVideoX 2B：30 层、隐维 1920；5B：42 层、隐维 3072（附录 Table 5）。T5 文本长度 226 出现在附录超参表。HunyuanVideo 13B：双流 20 块、单流 40 块、维 3072、FFN 12288、24 头、头维 128、3D-RoPE 通道 $(16,56,56)$（Table 2）。HunyuanVideo 1.5：8.3B，双流 54 块、维 2048、FFN 8192、16 头（Table 1）。这些数字只说明生成器有多大，不说明理解 CE 写在哪。

v 和 $\epsilon$ 的关系只写公开定义，不把 CogVideoX 没写的系数补上。Salimans & Ho 的 v-prediction 把目标写成噪声和数据的线性混合，CogVideoX 说“采用 v-prediction 与 zero SNR，噪声日程跟随 LDM”。zero SNR 指 $t=T$ 时信号噪声比为 0，避免最后一步仍残留信号。这些都改变**生成账内部**的回归目标，不改变有效位置：被回归的还是带噪视觉 latent。有人把 v-prediction 理解成“预测下一帧像素的速度”，会和 flow 的 $u_t$ 搅在一起。夹具不实现任何一种采样器，避免你用 Euler 步数去解释占用。占用在 $\mathrm{forget}$ 里已经决定了，和 50 步还是 10 步无关。

HunyuanVideo 的 $t$ 来自 logit-normal，不是均匀。CogVideoX 的 Explicit Uniform Sampling 则是把均匀区间切给各 rank。两种抽样都是生成账的训练稳定性技巧。理解 CE 没有时间步 $t$，不要给答案 token 也喂扩散时刻。给答案 token 喂 $t$ 等于发明一种“带噪文字分类”，两篇论文都没有这一项。

验证。读任何开源训练脚本，断言 `loss` 的 `reduction` 只发生在视觉 token 或视觉 latent。若 `labels` 把 prompt 也包括进去，先改 mask，再谈 FVD。本课 CPU 不实现扩散采样，只实现位置契约：未来 32 格进帧差，答案 1 格进 CE。见到 `guidance_scale` 或 CFG，把它记进推理超参，不要记进 $M_{\mathrm{und}}$。CFG 是生成时用无条件分支减一下，理解头没有无条件分支这一说。

### 5.5 3D causal VAE：压缩的是生成对象，不是理解 token 预算

为什么需要。视频像素体积比图大一个时间维。生成器若在像素上做注意力，序列长不可训。3D VAE 把时间、高、宽一起压进 latent，后面的 DiT 只看见压缩后的格子。这和[第 10 课](10_video_token_reduction.md)的 reducer 不是一把刀。reducer 删的是**进 LLM 的理解 token**；VAE 压的是**要被扩散或 flow 重建的生成对象**。把 50% 保留率写成 CogVideoX 的 8×8×4，口径错了。

怎么运转。CogVideoX 的 3D causal VAE：空间 8×8、时间 4，像素到 latent。因果卷积把 padding 放在时间轴开头，未来帧不进当前预测，用来减轻闪烁。论文 Table 2 在 WebVid 验证集、256×256、17 帧上，闪烁（相邻帧 $L_1$）Open-Sora 92.4、Open-Sora-Plan 90.2、他们 85.5，并写 PSNR 最好；HTML 转换丢失了 PSNR 列的对照数字，本课只引用闪烁三个数，不编 PSNR。训练先 17 帧再 context parallel 到 161 帧；损失是加权 $L_1$、LPIPS、KL，后期加 3D 判别器 GAN。16×16×8 过猛时即使加通道也难收敛，他们预训练用的是文中的 variant B。

HunyuanVideo：输入 $(T+1)\times 3\times H\times W$，

$$
\Bigl(\frac{T}{c_t}+1\Bigr)\times C\times\frac{H}{c_s}\times\frac{W}{c_s},\quad c_t=4,\;c_s=8,\;C=16
$$

从零训，图视频 4:1 混合。完整损失

$$
\mathrm{Loss}=L_1+0.1 L_{\mathrm{lpips}}+0.05 L_{\mathrm{adv}}+10^{-6} L_{\mathrm{kl}}
$$

Table 1：ImageNet 256 PSNR 33.14，MCL-JCV 33×360×640 为 35.39；CogVideoX-1.5 同表为 31.73 / 33.22。推理用空间-时间 tiling，重叠线性混合；另做一次随机开关 tiling 的微调，避免训推不一致。

HunyuanVideo 1.5 把空间压缩加到 16×，时间仍 4×，通道 32。压缩更狠，DiT 才能在 8.3B 上做 720p。这仍然是生成对象的体积，不是 Video-MME 的抽帧数 $F$。

手算一条 HunyuanVideo 的压缩，避免把 16 通道抄成 16 帧。取 $T+1=65$ 帧、720×1280 的像素（他们低分辨率短视频阶段写过 256×256×65 这一档，这里用 65 帧只为整除）。$c_t=4,c_s=8,C=16$：

$$
\Bigl(\frac{64}{4}+1\Bigr)\times 16\times\frac{H}{8}\times\frac{W}{8}=17\times 16\times\frac{H}{8}\times\frac{W}{8}
$$

时间维是 17 不是 16，因为因果 VAE 对第一帧单独留了一格。CogVideoX 的 $q>1$ 时在序列开头重复第一帧，动机同类：图像和视频要能一起训，第一帧不能被时间下采样吃掉。理解模型抽 8 帧进 LLM，没有“第一帧多一格”这种契约。两套 17 对 8 不能比谁更密。

再算 token 数的数量级。设 $H=W=256$，则空间 $32\times 32$，再乘时间 17、通道不进 token（通道在每个 token 的特征里）。patchify 之后还要除 $p$。CogVideoX 写长度 $\frac{T}{q}\cdot\frac{H}{p}\cdot\frac{W}{p}$。$p$ 的具体整数以仓库配置为准，论文正文没有把 $p$ 写成唯一一张表；不确定就不写。本课只锁：生成 token 数随 $T,H,W$ 涨，理解 token 数随抽帧 $F$ 和每帧 patch 涨，两条曲线自变量不同。第 10 课的 300×64=19200 是理解侧 5 分钟、每秒 1 帧、每帧 64 token。CogVideoX 10 秒 16 fps 是 160 帧像素，进 VAE 之后远小于 160×像素，但那是生成器自己的序列，不是 Thinker 的 prefill。

验证。不要在本课 CPU 里实现 3D 卷积。验收句只有：生成损失的格子数等于 VAE latent 展开后的 token 数，不等于理解前缀长度。夹具没有 VAE，用 32 个像素格子代替 32 个 latent，为的是占用可看。谁把 32 写成“CogVideoX token 数”，报告作废。谁把 17 格时间维写成“理解抽了 17 帧”，同样作废。

### 5.6 CogVideoX 的专家 Transformer：融合文本与视频，不融合两本账

为什么需要。文本和视频数值尺度不同，硬拼进同一序列会让 LayerNorm 被一侧主导。CogVideoX 用专家自适应 LayerNorm：时间步 $t$ 进调制模块，Vision Expert AdaLN 和 Text Expert AdaLN 分别作用于两侧 hidden。论文消融：同参数量下专家 AdaLN 的 FVD、CLIP4Clip 优于无专家 AdaLN，也优于同参数 MMDiT；MMDiT 再加倍参数才有得打。他们的判断是：两套独立 Transformer 不是必须的，专家 AdaLN 够用来对齐特征尺度。

怎么运转。3D VAE latent 形状 $T\times H\times W\times C$，patchify 成长度 $\frac{T}{q}\cdot\frac{H}{p}\cdot\frac{W}{p}$ 的 $z_{\mathrm{vision}}$。$q>1$ 时在序列开头重复第一帧，以便图和视频联合训。T5 得到 $z_{\mathrm{text}}$，与视觉在序列维拼接，进入一叠专家 Transformer。3D-RoPE：每个 latent 有坐标 $(x,y,t)$，三轴独立 1D-RoPE，通道占 $3/8$、$3/8$、$2/8$ 再拼接。3D 全注意力：论文 Figure 5 说明 2D+1D 时，第 $i+1$ 帧的人头不能直接看见第 $i$ 帧的人头，只能经背景补丁隐式传递，大运动容易裂。他们改用 3D 混合注意力，并借助 FlashAttention。消融里 2D+1D 的 FVD 早期明显更高，且更不稳。

数据与日程。约 3500 万单镜头片段，平均约 6 秒；另用 LAION-5B 与 COYO-700M 滤出的约 20 亿张图。负标签包括剪辑特效、运动不连、低质量、讲座、文字主导、屏幕录像。密 caption 管线：Panda70M 短句、CogVLM 逐帧密描述、GPT-4 汇总，再微调 LLaMA2 / CogVLM2-Caption。Multi-Resolution Frame Pack 把不同时长、分辨率打进同一 batch。分辨率从 256px 逐步到 512、768。Explicit Uniform Sampling：把 $1\ldots T$ 按 data parallel rank 切成 $n$ 段，每卡只在自己那段均匀抽 $t$，损失曲线更稳。高质量微调用总数据约 20% 的更干净子集，字幕和水印减少，论文也写语义能力略降。

评测数字只进生成账。VBench 上 CogVideoX-5B：Human Action 96.8、Scene 55.44、Dynamic Degree 62.22、Multiple Objects 70.95、Appearance Style 24.44、Dynamic Quality 69.5、GPT4o-MTScore 3.36。2B 的 Dynamic Degree 66.39 高于 5B 的 62.22，Multiple Objects 则 57.68 对 70.95：同一模型家族内部也不能用一个数代表全部。人工四项对 Kling（2024.7）：Sensory 0.722 对 0.638，Instruction 0.495 对 0.367，Physics 0.667 对 0.561，Cover 0.712 对 0.668，总分 2.74 对 2.17。Physics Simulation 仍是人工 0/0.5/1，不是杯子占用 $o$，更不是 Video-MME。

推理成本写进硬件行，不写进准确率行。H800、50 步：5B、480×720、6 秒，113 秒、26GB；5B、768×1360、5 秒，500 秒、76GB；2B、480×720、6 秒，49 秒、18GB。

RoPE 的外推与插值写在附录，生成分辨率变了才会碰到。他们比较：分辨率升高时，外推保留局部细节、容易出现许多张清晰小图；插值保留全局、容易一张大糊图。因为 RoPE 是相对位置，他们选外推。这和[第 14 课](14_long_context_curriculum.md)的文本 RoPE 外推是亲戚，对象是视频格子不是字。本课不重做课程式加长上下文，只引用：位置编码属于生成序列几何，改它不会把 Video-MME 变好，除非你另外训理解模型。

Frame Pack 解决的是 batch 里时长和分辨率不一致。固定帧数会丢掉短片、截断长片，还会让单帧图像和几十帧视频在双向注意力下裂成两种模式。Pack 把不同时长、分辨率塞进同一形状。3D-RoPE 按每个样本自己的 $(x,y,t)$ 取编码表的前缀（外推），不把表缩放到当前分辨率（插值）。理解模型的动态分辨率是[第 08 课](08_dynamic_vision.md)的 tile 与 M-RoPE，刀落在进 LLM 的图。两边都叫“可变分辨率”，一张管生成 batch，一张管理解前缀。报告里不要写成同一个开关。

Explicit Uniform Sampling 的定量对照在附录 Table 9：训练 40k 步时，五个扩散时刻 100/300/500/700/900 的验证损失，有显式均匀都低于无显式均匀，例如时刻 100 为 0.216 对 0.222，时刻 500 为 0.116 对 0.119。方向是“每个时刻都低一点”，不是只修某一个 $t$。这仍然全在生成账。不要把 0.116 当成 caption CE。

验证。引用 CogVideoX 时必须带任务：文生视频或图生视频。I2V 是从 T2V 微调，图像经 3D VAE 后与带噪输入拼接。不要把 I2V 的首帧条件写成理解 CE 的答案位置。首帧是生成条件，答案 token 是理解目标。图生视频会让人误以为“模型看见了第一帧所以理解过关”。看见第一帧只是条件，问“杯子还在不在”仍要单独的文字头和 $M_{\mathrm{und}}$。没有文字头，I2V 再稳也交不了理解账。

### 5.7 HunyuanVideo：13B 双流到单流，相机是生成标签

为什么需要。大纲要求写前核 arXiv。2412.03603 给出可引用的架构和人工表；2511.18870 给出缩小参数后的 8.3B 与稀疏注意力。两篇都是生成系统。结构化 caption 里的镜头类型、运镜，是为了让生成器听指令，不是为了答 Video-MME。

怎么运转。骨干：Dual-stream to Single-stream。双流里视频和文本各走多块 Transformer，互不抢调制；单流里拼接后融合。文本侧用视觉指令微调过的 MLLM，外加双向 token refiner；CLIP-Large 最后一个非 padding token 当全局引导，加进时间步嵌入。位置：3D-RoPE，按 $T,H,W$ 分通道。训练：先 256px / 512px 图像预热，再图视频联合，分辨率和时长课程式加长。视频按时长桶 $B_T$ 和宽高比桶 $B_{AR}$ 分桶，每桶最大 batch 卡在显存上限。Prompt 改写用 Hunyuan-Large，训练免费的 in-context 加自修订，另有 LoRA 加速版。

相机。他们训了一个运镜分类器，14 类：zoom in/out，pan 上下左右，tilt 上下左右，around 左右，static，handheld。高置信写入 JSON caption，意图是让生成可控。这是生成账的条件字段。Video-MME 没有“请向左摇镜头再四选一”这种成功定义。把 14 类命中率填进 C2，属于第 47 课禁止的类错。

人工评测 Table 3。60 名专业评估员，1533 条 prompt，公开 600 条。三项：Text Alignment、Motion Quality、Visual Quality，再汇总 Overall 与排名。HunyuanVideo 5 秒：61.8%、66.5%、95.7%、Overall 41.3%，第 1。CNTopA：62.6%、61.7%、95.6%、37.7%。Gen-3 alpha 6 秒：47.7%、54.7%、97.5%、27.4%。Luma 1.6：57.6%、44.2%、94.1%、24.8%。读表时盯两处。Visual Quality 他们 95.7，Gen-3 97.5，Gen-3 更高；Overall 仍是他们第一，因为 Motion 66.5 对 54.7。生成账内部已经不能只报画质。Text Alignment 61.8% 是人对“是否跟 prompt”的满意率，单位不是 Video-MME 的准确率。

HunyuanVideo 1.5 补的是效率与后训练，不是理解基准。8.3B DiT 出 480p–720p、5–10 秒，再 VSR 到 1080p。SSTA：块划分、选择性 Top-k、滑动 tile 窗口，两 mask 求交后做 block-sparse attention；10 秒 720p 相对 FlashAttention-3 端到端 1.87×。Muon 优化器：论文写在一半步数内训练损失低于 AdamW。数据：超过 1000 万小时原始视频切成 2–10 秒，过滤后约 8 亿片段；图像 50 亿进预训练。任务比 T2I:T2V:I2V $=1:6:3$。后训练分 CT / SFT / RLHF；I2V 用在线 RL，T2V 先 DPO 再在线。T2V Rating 720p：指令 61.57、审美 63.30、视觉 57.35、结构稳定 79.75、运动 57.67。GSB、300 条、100 余名评估：对 Veo3 的 T2V win rate 为 $-10.32\%$（Better 24.64%，Other 34.96%）。消费卡路径：卸载加 VAE tiling，720p 121 帧峰值 13.6GB。这些数字全部进生成账。没有一行是 Video-MME。

缩放律只用来解释他们为什么选 13B，不要当成理解模型的参数配方。他们先在 DiT-T2X 图像家族上拟合 $N_{\mathrm{opt}}=a_1 C^{b_1}$、$D_{\mathrm{opt}}=a_2 C^{b_2}$，得到 $a_1=5.48\times 10^{-4}$、$b_1=0.5634$、$a_2=0.324$、$b_2=0.4325$（$N,D$ 以十亿计，$C$ 以 Peta FLOP 计）。再把图像包络上的检查点当视频实验初始化，视频侧 $a_1=0.0189$、$b_1=0.3618$、$a_2=0.0108$、$b_2=0.6289$。结合训练消耗和推理成本，模型定在 13B。原文写明：由此算出的 token 数只覆盖图像和视频各自的**第一阶段**，从低分辨率到高分辨率的课程式缩放留待将来。本课更不把这些幂律抄进 MiniMind-O。幂律描述的是生成 DiT 的计算–参数–数据关系，Video-MME 的 2700 题不在自变量里。

1.5 的推理表用来填硬件行。无工程加速、8×H800：720p、241 帧，无稀疏 5.5070±0.0284 秒/步，有 SSTA 2.9475±0.0206 秒/步。有 SageAttention、`torch.compile`、特征缓存时，50 步总计：同规格无稀疏 96.78 秒，有稀疏 58.39 秒。这些是生成一步扩散的墙钟，不是理解 prefill。第 10 课的 prefill 针对进 LLM 的视觉 token；这里针对 DiT 的 latent 序列。两套延迟表禁止合并。

验证。协议卡加四列：Text Alignment、Motion、Visual、Overall。缺 Motion 只报 Visual，本课视为未完成。相机列单独写 14 类，不要和 Text Alignment 兑。1.5 的 GSB 负值必须原样保留，禁止改写成“已经超过 Veo3”。13B 与 8.3B 是两篇报告、两套数字，引用时写版本。把 1.5 的结构稳定 79.75 填进 13B 的 Overall 41.3 那一行，属于第 01 课禁止的版本错配。

### 5.8 物体永久性：理解对、生成可以消失

为什么需要。大纲验收：理解答“杯子还在不在”对，生成下一帧物体消失，必须能造出。这是分账的存在性证明。若你的夹具里生成器永远抄上一帧，两本账看起来永远一致，协议就测不出来。

怎么运转。Sora 的公开技术叙述写：模型**常常、但并不总是**能在遮挡或出画后保持人或物；失败模式包括长视频失谐、物体凭空出现。其技术报告未公开占用探针的公式或成功率。Cosmos 在[第 34 课](34_world_model_platform.md)已引用的 Limitations 里，把缺少物体永久性列为当前生成器问题。本课不重做 Cosmos 的 ID 计数器，只用一个杯子占用。理解读 $o(I_4)$，生成读 $o(\hat I_5)$。两者可以反向。

数学。历史全有杯子：

$$
o(I_t)=1\quad t=0,\ldots,4
$$

均值生成：

$$
o(\hat I_5)=0,\qquad o(I_5)=1
$$

理解仍输出 YES。这组不等式就是 Lab 的 Gate。$\mathrm{forget}\ge 0.75$ 时教学 $8\times 8$ 上占用低于 0.25，与 CPU 的 $\mathrm{forget}=1$、$o=0$ 同方向。

公开落点。CogVideoX 人工项有 Physics Simulation 0.667，不是 $o$。VBench Multiple Objects 70.95 测多物体是否按 prompt 出现，不测跨帧同一物体是否还在。HunyuanVideo Motion 66.5% 是人对运动质量的满意，不拆“消失”。谁把这三项平均成“物理 66%”，本课与第 47 课一起拒收。

遮挡版和遗忘版不要混。第 33 课从 $t=5$ 起遮住未来，像素预测器看不见纹理只能填 0.5，接触边糊掉。本课不遮历史：五帧杯子都看得见，理解没有任何借口答“不在”。生成器仍可以主动把下一帧涂成均值。一个是看不见所以糊，一个是看见了仍抹掉。Sora 公开叙述里的遮挡保持，对应第 33 课那类探针；凭空出现和自发消失，对应本课 $o(\hat I_5)=0$。本课 Gate 用的是后者。有人把 Lab 理解成“先遮住杯子再问还在不在”，会把历史开关拨到没有，理解先错，分裂造不出来。历史必须看见杯子，生成才算“明知还在却画没了”。

验证。CPU：`generated_frame_cup_vanished` 与 `understand_answers_cup_still_there` 同时为真。Lab：预测必须选“理解答还在，生成下一帧杯子消失”，历史必须有杯子，forget 必须够大。选“帧差更低就等于杯子还在”不能过 Gate，即使你已经看见 $L_2$ 数字。选“理解会先答错”同样不能过：那是把第 23 课幻觉探针拿来交差。本课允许幻觉存在于真实 VLM，夹具却把理解头写成占用分类器，为的是隔离变量。真实幻觉要另开 POPE 行，不要和 $o(\hat I_5)$ 兑。

### 5.9 相机轨迹与 Video-MME 不是同一张表

为什么需要。生成器爱报“跟得上运镜”，理解器爱报“看懂长视频”。两个百分数并排，读者会以为都是视频能力。第 47 课已经禁止六类横比。本课在生成账内部再切一刀：相机可控、VBench 动态度、物体永久、闪烁，也不能兑成一列。

怎么运转。四列最低配置：

| 列 | 测什么 | 本课引用 | 不得写成 |
|---|---|---|---|
| C2 理解 | 短中长 MCQ | Video-MME 无字幕 75.0%（Gemini 1.5 Pro，2700 题） | 生成质量、运镜、占用 |
| 生成动态 | 运动是否丰富 | CogVideoX-5B Dynamic Degree 62.22 | 理解准确率 |
| 生成可控 | 是否服从运镜标签 | HunyuanVideo 14 类写入 caption | Video-MME 选项 |
| 生成永久 | 物体是否无故消失 | 夹具 $o(\hat I_5)$ | CE 或 Video-MME |

数学。账本函数 `LEDGERS`：`videomme_accuracy` 映射到 `understand_c2`，`camera_type_match` 映射到 `generation_camera`，`object_permanence_miss` 映射到 `generation_physics`，`vbench_dynamic_degree` 映射到 `generation_vbench`。`same_ledger(a,b)` 为假时禁止做差。`may_post_as_generation_score("videomme_accuracy")` 必须为假。

公开落点。Video-MME 900 视频、2700 题，短 / 中 / 长，字幕可选。Gemini 1.5 Pro 无字幕 75.0%、有字幕 81.3%、长视频无字幕 67.4%。这些数字第 47 课已经打开原文。本课只借用：它们进 C2。CogVideoX 不用 Video-MME 当主表，它用 VBench 与人工四项。HunyuanVideo 不用 Video-MME，它用 60 人三项满意率。三套协议并排可以，相减不行。

验证。CPU 四条：`videomme_accuracy_rejected_as_generation_score`、`object_permanence_stays_on_generation_ledger`、`camera_match_not_same_ledger_as_videomme`、`vbench_not_same_ledger_as_videomme`。缺一条，分账没写完。

### 5.10 Sora：公开叙述到此为止

为什么需要。任务写明：Sora 无公开技术报告则不编架构。OpenAI 页面 *Video generation models as world simulators* 给出高水平叙述：先把视频压进低维 latent，再切成 spacetime patch 当 Transformer token；Sora 是扩散 Transformer，给定带噪 patch（以及文本等条件）预测干净 patch；可变时长、分辨率、宽高比；用了 DALL·E 3 那类 recaption。定性能力包括遮挡后仍常能保持物体，定性失败包括玻璃破碎、进食后状态、长样本失谐、物体凭空出现。参数量、层数、VAE 倍数、训练步数、完整损失，其技术报告未公开。

怎么运转。本课引用规则：

- 可以写：Sora 把视频当成 spacetime patch 上的扩散 Transformer。
- 可以写：公开叙述承认物体永久不是总成立。
- 不可以写：Sora 有 $N$ 层、$C$ 通道、某一种 AdaLN。
- 不可以把 2402.17177 这类综述的反向工程当成 Sora 官方结构。

验证。实验记录里给 Sora 留一行，字段全是 `unspecified`，`forbidden=architecture_invented`。谁填了层数，本课与第 01 课一起判不可追溯。

### 5.11 梯度尺度：32 格对 1 个 token

为什么需要。即便你已经把 mask 写成不相交，若把两项直接相加成一个标量再 `backward`，默认均值会让 32 项帧差压过 1 项 CE。看起来“已经分账”，优化路径仍被生成主导。分账要写到日志和权重，不只写到注释。

怎么运转。记 $\ell_{\mathrm{und}}=\mathcal{L}_{\mathrm{und}}$，$\ell_{\mathrm{gen}}=\mathcal{L}_{\mathrm{frame}}$。夹具量级：$\ell_{\mathrm{und}}\approx 0.087$，$\ell_{\mathrm{gen}}\approx 0.110$。看起来同阶，是因为 CE 只有 1 项、帧差已经对 32 格取了平均。若有人把帧差改成 sum 而不是 mean，生成项会大约乘 32，变成 3.5 对 0.087，CE 几乎消失。反向操作同样危险：有人按 token 数把 CE 乘上文本长度，问句 4 加答案 1 再平均错了，理解项会漂。纪律是：每项先在自己的有效位置上取平均，再以显式 $\lambda$ 相加，并且每步打印两项。

数学。合法总损失

$$
\mathcal{L}=\lambda_{\mathrm{und}}\ell_{\mathrm{und}}+\lambda_{\mathrm{gen}}\ell_{\mathrm{gen}}
$$

$\lambda$ 必须出现在配置里。本课夹具不训练，所以不选 $\lambda$。夹具只证明：没有两项日志时，你无法知道是 CE 在降还是 $L_2$ 在降。`total_loss` 一个键等于没分账。

公开落点。Emu3 预训练对视觉 token 乘 0.5，那是离散统一里的 $\lambda$，见第 41 课。CogVideoX 与 HunyuanVideo 的主损失没有理解 CE 这一项，它们的 $\lambda_{\mathrm{und}}=0$。想在同一 checkpoint 上加 VQA，必须新增 $M_{\mathrm{und}}$ 和 $\lambda$，不能指望扩散损失“顺便”学会答杯子。第 20 课在图像上比较过干扰（interference）：一个任务的更新拖偏另一个。视频上同样会发生。本课不测干扰幅度，只要求你有两个键，否则干扰发生时你看不见。

验证。改造清单第 3 条就是双项日志。CPU 主路径不训练，所以这一条不进 16 个 `checks`。实验记录仍要写：“本课未训联合权重，故未测 $\lambda$。”空着比填一个编出来的 0.5 好。Emu3 的 0.5 不能借来当 CogVideoX 的 $\lambda$。

## 6. 在公开实现中定位这些机制

本课没有 MiniMind-O 必改文件。对照公开仓库时，只核对论文已经写明的模块名，不根据博客补层数。

CogVideoX 官方代码与权重入口见论文给出的 [THUDM/CogVideo](https://github.com/THUDM/CogVideo)。打开仓库后按三块搜：3D causal VAE（时间因果卷积、压缩比）、Transformer（专家 AdaLN、3D 注意力、RoPE）、pipeline（T2V / I2V、50 步默认）。Hugging Face `diffusers` 的 `CogVideoXPipeline` / `CogVideoXImageToVideoPipeline` 是推理封装，训练 mask 仍以论文第 2–3 节为准。见到 `prompt_embeds` 与 `latents` 分张量，就是条件与生成对象分开；不要把 `prompt_embeds` 的 CE 臆造进去。

HunyuanVideo：[Tencent/HunyuanVideo](https://github.com/Tencent/HunyuanVideo)。对照 Figure 5：Causal 3D VAE、双流块、单流块、MLLM 文本、CLIP 全局向量。14 类运镜若出现在 caption JSON 的字段名里，把它抄进生成可控列。HunyuanVideo 1.5：[Tencent-Hunyuan/HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5)。对照 SSTA 与 VSR：稀疏注意力是生成序列上的加速，不是第 10 课的理解 token 剪枝。峰值 13.6GB 只在官方卸载配置下成立，不要写成“8.3B 随手 13.6GB”。

对照时不要被类名骗。`CogVideoXTransformer3DModel` 里有 attention，不等于第 09 课的时序融合模块。`AutoencoderKLCogVideoX` 里有 KL，不等于理解模型的 VAE 可视化。`prompt_embeds` 形状若是 `[B, L_text, C]`，它是条件；`hidden_states` 或 `latents` 形状若是 `[B, C, T, H, W]` 或 packed 序列，它是生成对象。打印一次 `.shape` 比读三篇博客有用。HunyuanVideo 的 dual-stream block 数量 20 必须和 single-stream 40 一起出现；只抄 60 层会把双流阶段的“各走各的”抹掉。1.5 改成 54 个 dual-stream、没有沿用 20+40，抄混版本就是抄错。

定位时带一张空的双列表。左栏只准填理解：Video-MME、caption CE、POPE。右栏只准填生成：VBench、人工 Motion、相机、占用 $o$、闪烁。仓库里搜到 `total_loss` 这种键，应当拆成两个键，而不是给它补一个权重。搜到 `guidance_scale` 填右栏推理超参。搜到 `num_frames` 填右栏生成长度，不要填 Video-MME 的短中长。短中长是评测桶，`num_frames` 是采样长度，单位都和秒有关，成功定义不同。

## 7. 数据与训练 recipe

本课不训 13B。recipe 写成“若你要在缩小版上复现分账，数据该长什么样”，以及论文自己的数据事实。

CogVideoX 的生成预训练：过滤后约 35M 片段，均长约 6 秒；约 2B 张图辅助。负例过滤用 Video-LLaMA 上训的 6 个分类器，再加光流与美学阈值。密 caption 不是原始字幕。高质量微调约 20% 子集。这些数字只说明生成器吃什么。它们不能解释 Video-MME 的 900 条评测视频。

HunyuanVideo：层级过滤做出 256p / 360p / 540p / 720p 四档，每档丢掉上一档的一半到五分之一量级（原文 Figure 4 的定性描述）。终档 SFT 约 100 万条人工挑选。结构化 caption 的 JSON 字段包括短描述、密描述、背景、风格、镜头、灯光、气氛，再加元数据标签，用 dropout 与排列拼出长度不同的 prompt。1.5 进一步：原始视频超过 1000 万小时，过滤后约 8 亿片段；图像 50 亿 + 10 亿后续。任务比 1:6:3。后训练 T2V / I2V 各约 100 万 CT。

缩小版 recipe（教学，不是论文规模）：

| 字段 | 理解样本 | 生成样本 |
|---|---|---|
| 媒体 | 5 帧桌面，杯子在或不在 | 同一 5 帧作条件，第 6 帧作目标 |
| 文本 | “杯子还在桌上吗？” + YES/NO | 可选：“杯子留在列 5”作条件，不算 CE |
| mask | 只有 YES/NO | 只有第 6 帧格子 |
| 禁止 | 把 YES 的 CE 加到像素平均里 | 把第 6 帧 $L_2$ 当成准确率 |

batch 里两种样本可以混，**损失必须分项打印**。混合权重 $\lambda$ 是超参，本课不指定数值；没有分项的 $\lambda$ 等于没分账。混 batch 时还要打印每项的有效 token 数：理解项应接近答案长度之和，生成项应接近未来格子之和。若理解项有效数突然变成 197，说明 mask 被广播成全 1。若生成项有效数变成 160，说明在重建历史。第 01 课要求按模态统计有效数量，本课按任务统计。任务不是模态：同一视觉格子，在历史段是条件，在未来段是目标。按模态加总会把两段加在一起，看起来“视觉 loss 很健康”，其实在抄条件。

缩小版不要用 Video-MME 的 900 条当训练集。那是评测。也不要用 CogVideoX 的 35M 当理解 SFT：那些密 caption 是给生成器对齐语义的，题型是描述，不是“还在不在”。若你只想交本课主路径，128 条桌面短序列足够让 CPU 和 Lab 之外的可选 LoRA 过拟合到占用探针上。128 条证明不了 VBench。写进记录：`n_train=128`，`eval=fixture`，`forbidden=vbench_sota`。

过滤负例的标签也要分账。CogVideoX 把“讲座、文字主导、静止”当成生成数据的负例，因为生成器要学运动。理解模型可能恰恰需要讲座和字幕来答 Video-MME 的推理题。同一条视频，生成账可能丢，理解账可能留。不要把生成过滤脚本直接套到理解数据上，否则你会训出一个很会动、很不会读字幕的 Omni，再把 Video-MME 有字幕的 +6.2 个点当成已经修好。+6.2 是第 47 课的可选增益，修的是评测输入，不是生成过滤。

推荐配置：CPU 用 `FORGET=1.0`、logits $(0.1,2.5)$。Lab 默认 $\mathrm{forget}=0.30$，强迫你自己拉到 0.75。不要改 CPU 的 4×8 去迁就 Lab 的 8×8。

## 8. 按依赖顺序执行实验

实验分两层。CPU 层证明 mask 与占用。教具层证明你能先预测再造出分裂。没有 GPU 的读者把 Step 0 到 Step 5 做完即完成本课主路径。Step 6 标 `skipped-no-gpu`。

CPU 的 `checks` 是验收清单的机器版：

- `understand_ce_and_generate_masks_disjoint` 锁交为空；
- `prompt_tokens_excluded_from_ce` 与 `history_pixels_excluded_from_ce` 锁理解侧；
- `history_pixels_excluded_from_generate`、`future_pixels_excluded_from_ce`、`answer_token_excluded_from_generate` 锁越界；
- `illegal_or_mask_covers_both_ledgers` 锁反例；
- `understand_answers_cup_still_there` 与 `generated_frame_cup_vanished` 锁分裂；
- `copy_last_frame_keeps_cup` 与 `truth_next_frame_keeps_cup` 锁对照；
- `mean_fill_l2_beats_copy_while_cup_vanishes` 锁 $L_2$ 不可代替占用；
- 四条账本检查锁 Video-MME / 相机 / VBench / 物体永久。

网页练习只展示 `metrics` 里的有限数字：下标、占用、$p(\text{还在})$、两段 $L_2$。不要把完整网格当成功率。`intersection_size` 和 `gen_occupancy` 必须是 0。`understand_answer` 必须是 1。`caption_ce` 必须能对到手算的 0.086836，允许打印六位小数的四舍五入差，不允许改成 0。`copy_l2` 必须大于 `gen_l2`。若你本地改了纹理函数导致这一条翻掉，先还原 `texture()`，不要改 `checks` 去迁就。

默认参数下，把 164 附近的序列切片抄进记录，防止只看布尔量。下标 160–163 是问句，四位全 0。164 是 1。165–168 是未来帧的前四个格子，应全是生成 1、理解 0。若 160 变成 1，说明问句进了 CE，模型在学“杯子还在桌上吗”这六个字的写法。若 159 变成 1，说明最后一帧历史像素进了 CE 或进了帧差：进 CE 会把理解变成“复读最后一帧的亮度分类”，进帧差会把生成变成“重建已经看见的帧”。两种都是条件当目标。159 属于历史段末尾，夹具里它对应 $t=4$ 的最后一个空间格，最容易被 off-by-one 写进未来段。写代码时先断言 `FUTURE_START == ANSWER_INDEX + 1`。

Lab 与 CPU 的 forget 默认值不同，不要因此认为协议不一致。CPU 用 1.0 一次造出占用 0。Lab 默认 0.30，是为了强迫你动手把分裂造出来：不拖滑条就过不了。有人把 Lab 默认改成 1.0 再交截图，Gate 可能仍亮，但“必须造出”这条验收被短路了。本课要求截图或进度里出现 $\mathrm{forget}\ge 0.75$ 的状态值，进度字段由 `useCompletionGate` 写入。重置会把 forget 送回 0.30 并清空预测，必须重选。调历史开关同样清揭晓。这些清场规则是为了先预测再揭晓，不是为了找茬。

### Step 0：冻结符号

在记录里写死 $H=4,W=8,T=5$，答案下标 164，未来 165–196，logits $(0.1,2.5)$。

```bash
PYTHONPATH=experiments/src python3 -c "from learn_omni_experiments.lessons.lesson_49 import ANSWER_INDEX, FUTURE_START, SEQ_LEN; assert (ANSWER_INDEX, FUTURE_START, SEQ_LEN)==(164,165,197)"
```

### Step 1：跑分账夹具

登记前从仓库根目录：

```bash
PYTHONPATH=experiments/src python3 -c "from learn_omni_experiments.lessons.lesson_49 import run; r=run(); print(r['checks']); print(r['metrics'])"
```

登记之后改用仓库统一入口：

```bash
python3 run.py run 49
```

确认 16 条 `checks` 为真。重点抄：`intersection_size=0`，`gen_occupancy=0.0`，`caption_ce=0.086836`，`gen_l2=0.109598`，`copy_l2=0.243056`。

### Step 2：手算 softmax

用计算器核对 $p_1$ 与 CE。对不上先改你的指数运算，不要改夹具 logits。

### Step 3：非法 OR

在笔记里把 CE 位与未来 32 位画成一条 197 的 0/1 串。OR 之后 1 的个数必须是 33。这就是共享 mask 的罪状。

### Step 4：占用对照

抄帧占用 1，均值占用 0，真值占用 1。若你改纹理函数导致均值占用不再为 0，先回到 `BG_FILL=0.5`。

### Step 5：浏览器教具

打开本课 Lab。先在四个选项里预测哪句话会出现。保留“可见历史里有杯子”。把 forget 拉到 0.75 或以上。点揭晓。理解侧必须显示“还在”，生成帧杯子格子变成灰均值，mask 交为 0。不要点“两边都会保住杯子”再指望 Gate 亮。教具标明教学模拟。

推荐顺序：预测选分裂；确认有杯子；forget 从 0.30 拖到 0.80；揭晓；看占用从 1 附近掉到 0.20 以下。若先揭晓再改预测，进度会清掉，必须重选。页面在未揭晓时生成帧用斜纹占位。

教具左侧五帧是条件，人人可见，不算泄漏。泄漏指的是生成帧和 $p(\text{还在})$、$L_2$、占用、mask 交。四格数字在揭晓前显示破折号。理解面板的大字是“揭晓后”。预测区四个选项必须在点揭晓之前选好；未选时按钮禁用。禁用不是故障。有人用开发者工具强行揭晓，进度仍要求 `prediction === "understand_ok_gen_vanish"`，捷径过不了 Gate。Gate 文案写明：先选分裂，保留历史杯子，forget 到 0.75 以上。三条缺一，文案不会变成“验收已通过”。

把教具滑条和 CPU 默认值对一下。CPU：$\mathrm{forget}=1$，$o=0$，$L_2=0.109598$。Lab：$8\times 8$ 的 $L_2$ 数值会变，因为格子多了、纹理项占比不同，**不要**要求 Lab 的 $L_2$ 等于 0.109598。Lab 只锁方向：forget 大时占用低，理解仍答还在，交为 0。方向对、绝对值不同，是网格尺寸不同，不是协议分裂。谁为了对齐绝对值去改 CPU 的 `HEIGHT`，会把 `gen_positions` 从 32 改成 64，Step 0 的断言会炸。炸是对的。不要改断言去迁就 Lab。

### Step 6：可选，公开权重只填协议卡

若有卡，用官方 prompt 各抽 8 个视频，人工看杯子是否消失、运镜是否听从，**不要**把这 8 个数写成 VBench。测不到就引用论文表，并写“未在本课复测”。不要用 Sora 宣传片估物体永久成功率。

## 9. 评测与测量

主指标不是一个视频分，是两本账是否自洽。

| 指标 | 定义 | 通过条件 |
|---|---|---|
| mask 交 | $M_{\mathrm{und}}$ 与 $M_{\mathrm{gen}}$ 的交集大小 | 必须为 0 |
| CE 位置数 | $M_{\mathrm{und}}$ 的元素个数 | 夹具为 1 |
| 帧差位置数 | $M_{\mathrm{gen}}$ 的元素个数 | 夹具为 32 |
| 理解 $p(\text{还在})$ | softmax 在 YES 上的质量 | 有杯子时 $>0.9$ |
| 历史占用 | $o(I_4)$ | 有杯子时 $=1$ |
| 生成占用 | $o(\hat I_5)$ | 遗忘夹具 $<0.15$ |
| $L_2$ 对照 | 均值对抄帧 | 均值可以更低 |
| 账本 | Video-MME 是否当生成分 | 必须拒绝 |

CogVideoX 与 HunyuanVideo 的公开表可以抄进“对照行”，必须另写 `run_id=paper`。自测行 `run_id=lesson49-cpu` 只有占用与 CE。两行禁止平均。对照行抄的是别人的生成账，自测行抄的是你的协议账。两行并排时，读者仍可能把 96.8 和 0.917 看成同类。并排不等于可减。表头必须写出 ledger 名字，名字不同就停手。名字写了、单位不同，同样停手。单位才是尺子，名字只是抽屉。抽屉贴错可以撕掉重贴，尺子用错会把 75.0 减成 62.2。

区间：本课 $N=1$ 条教学序列，不准算 Wilson 去跟 Video-MME 的 2700 题比宽窄。第 47 课的区间纪律在这里仍然有效：分母不可借。

对照行怎么抄，写一张样板，避免交作业时把表号写丢。

`run_id=cogvideox-5b-paper-table3`，`ledger=generation_vbench`，`human_action=96.8`，`scene=55.44`，`dynamic_degree=62.22`，`multiple_objects=70.95`，`appearance_style=24.44`，`dynamic_quality=69.5`，`gpt4o_mt=3.36`，`forbidden=videomme_accuracy`。旁边必须另有人工四项：`run_id=cogvideox-5b-paper-table4`，`sensory=0.722`，`instruction=0.495`，`physics=0.667`，`cover=0.712`，`total=2.74`，对照 Kling `total=2.17`。Hunyuan 行：`run_id=hunyuanvideo-paper-table3`，`text_alignment=61.8`，`motion=66.5`，`visual=95.7`，`overall=41.3`，`n_prompts=1533`，`n_raters=60`，`forbidden=videomme_accuracy`。自测行：`run_id=lesson49-cpu`，`intersection=0`，`o_hist=1.0`，`o_gen=0.0`，`p_yes=0.916827`，`ce=0.086836`，`l2_mean=0.109598`，`l2_copy=0.243056`。三行可以并排，禁止平均成“视频分 80”。

单位再钉一次。VBench 分项不是百分数准确率，有的像 96.8 看起来像准确率，Appearance Style 24.44 明显不是同一把尺。人工 0/0.5/1 的均值 0.667 也不是 66.7% 的试题命中。Hunyuan Overall 41.3% 是三项满意率合成的汇总，原文没有把公式写成和 Video-MME 相同的微观准确率。看见百分号不要自动允许做差。本课 CPU 的占用是 $[0,1]$ 的连续量，打印成 0.0 或 1.0 时尤其容易被抄成“成功率 100%”。占用 1 表示杯子格子够暗，不表示生成任务成功。

## 10. 验收条件

验收不看你是否喜欢生成的视频。夹具的均值帧很难看，这是故意的。好看是 VBench 和人工 Sensory 的事，本课主路径没有这两项的自测。有人把 Lab 生成帧调到 forget=0 因为“杯子还在比较舒服”，Gate 不会亮。舒服的那一帧是抄历史，占用 1，$L_2$ 却可能更高，恰好是 5.3 节要你承认的那件事。验收要你亲手造出难看的消失，并承认理解 CE 对此无动于衷。难看不是失败，看不见消失才是失败。

同时满足才算本课通过：

1. CPU 16 条 `checks` 全为真。
2. Lab 预测选分裂，历史有杯子，$\mathrm{forget}\ge 0.75$，揭晓后理解“还在”、生成占用 $<0.25$。
3. 实验记录里 $M_{\mathrm{und}}$、$M_{\mathrm{gen}}$ 下标与 197 长度一致。
4. 论文数字带 arXiv 与表号；Sora 行没有编造的层数。
5. Video-MME、VBench、相机、占用四列分开，没有总平均。
6. 没有把第 10 课的保留率或第 33 课的 JEPA 接触差写成生成器已经交卷。

任何一条失败，先改协议，不要改公开数字。

## 11. 根据症状定位失败环节

| 症状 | 先查 | 不要先做 |
|---|---|---|
| CE 很低但问句也被学习 | `prompt_tokens_excluded_from_ce` | 加大生成权重 |
| 生成 $L_2$ 降、杯子没了 | 占用 $o$，不是再降 $L_2$ | 宣称物理更好 |
| 理解也答“不在” | 历史是否真的有杯子 | 怪专家 AdaLN |
| mask 交不为 0 | 答案下标是否落入未来段 | 换 3D 注意力 |
| 想用 75.0% 证明生成强 | 第 47 课 C2 协议卡 | 和 62.22 做差 |
| 报告出现 Sora 36 层 | 公开页面有没有这一句 | 从综述抄反向工程 |
| Lab Gate 不亮 | 预测选项、forget、历史开关 | 改 CSS |
| 抄帧 $L_2$ 低于均值 | 纹理是否被改成常数 | 改占用阈值 |
| I2V 首帧被当成答案 | 条件与目标是否分张量 | 加 caption CE |
| 闪烁很低但物体乱换 | 相邻帧 $L_1$ 不测身份 | 用闪烁代替 $o$ |
| 总 loss 降、分项一升一降 | 两项日志与 $\lambda$ | 只报 total |
| 5B Dynamic Degree 低于 2B | Table 3 分列是否还在 | 用 5B 覆盖 2B |
| Hunyuan Visual 低于 Gen-3 | Overall 与 Motion 是否同表 | 只截 Visual 第一 |
| 1.5 结构稳定 79.75 进 13B 行 | 版本字段 | 两篇报告兑一列 |
| Lab 占用 0.30 过不了 | forget 是否 $\ge 0.75$ | 改 CPU 门槛 |
| $p(\text{还在})$ 对不上 0.917 | logits 是否仍是 $(0.1,2.5)$ | 换优化器 |
| 交为 0 但 OR 是 197 | 历史是否被写进 OR | 只看交 |
| 相机命中 90%、杯子仍消失 | 14 类是条件不是占用 | 用可控代替永久 |
| Sora 演示杯子还在 | 公开叙述写“并不总是” | 把演示写成定理 |
| 把 8×8×4 写成保留率 50% | 第 10 课 reducer 位置 | 改 EVS 阈值 |
| JEPA 接触差很好 | 本课有没有 CE mask | 用第 33 课交差 |

定位时带四个问题，按顺序问，不要跳。第一问：这是理解数字还是生成数字？答不出，停在第 47 课。第二问：有效位置是答案 token 还是未来格子？答不出，停在 5.1 节。第三问：占用是多少，$L_2$ 是多少，两者是否同向？答不出，停在 5.3 节。第四问：Sora 这一行有没有编造的层数？有，删掉再继续。四个问题都答完，才允许打开改造清单。改造清单会加 $\beta\mathcal{L}_{\mathrm{occ}}$，占用将进入损失；若你还没证明默认 $L_2$ 看不见占用，加 $\beta$ 会让你以为问题从来都不存在。

off-by-one 单独说一次。未来段起点写错一位，交仍可能为空：164 进 CE，166–196 进帧差，165 谁都不管。交为空，OR 变成 32，夹具 `gen_positions` 不再是 32，`illegal_or_mask_covers_both_ledgers` 仍可能因长度变化而失败。所以交为空是必要条，不是充分条。必须同时核对两个集合的大小和端点。Lab 色带看不出 165 缺了一格，CPU 的 `future_start=165` 与 `gen_positions=32` 才能看出来。视觉验收和机器验收要一起用。

定位顺序固定：先 mask，再占用，再账本标签，最后才是模型结构。结构问题（2D+1D 裂大运动、VAE 闪烁）属于生成账内部，解决了也不自动过理解 CE。

## 12. 交付物

1. 一份协议卡：两张 mask 的下标、占用公式、四列评测。
2. CPU `result.json`：`intersection_size=0`，`gen_occupancy=0.0`，`understand_answer=1`。
3. Lab 一次通过的截图或进度状态：预测、forget、占用。
4. CogVideoX Table 3 / 4 与 HunyuanVideo Table 3 的对照抄本，带表号。
5. Sora 行：公开叙述三句 + `unspecified` 字段。
6. 明确声明：未训练 5B/13B，未跑 VBench，未跑 Video-MME。

缺第 6 条的交付物，本课视为把夹具能力写成了产品能力。

交付物自检用这张短表，交之前勾一遍。

| 项 | 有 | 没有时的症状 |
|---|---|---|
| 两张 mask 下标 | 164 与 165–196 | 只写“我们已经分开了” |
| 占用两个数 | 1.0 与 0.0 | 只写 $L_2$ |
| 四列评测 | C2 / VBench / 相机 / 永久 | 一个视频分 |
| 表号 | Table 3、Table 4、Table 3 | “论文里很高” |
| Sora 空行 | unspecified | 36 层或 3D U-Net |
| 否定句 | 未训 5B、未跑 VBench | 默认为已经复现 |

六行都有，才允许把本课标成完成。五行为有、Sora 行写了层数，整份交付作废，因为不可追溯比缺一张表更糟。第 01 课的指纹纪律在这里变成：没有公开来源的架构数字，等同没有 hash 的 checkpoint。

## 13. 前沿对照与改造方向

**公开方案。** 2024–2025 年把视频生成写成可训练系统的公开材料，按“损失写在哪 / 文本怎么进 / 评什么”分成三组。第一组扩散 Transformer：CogVideoX 用 T5 + 专家 AdaLN + 3D 全注意力 + v-prediction。第二组 flow + 双流：HunyuanVideo 13B，1.5 改 8.3B、SSTA、超分、Muon、RLHF。第三组只有定性叙述：Sora 的 spacetime patch 与失败列表。第一、二组可以对照公式。第三组只能对照失败模式的名字，不能对照层数。理解侧对照仍是第 09、10、47 课：TMRoPE、token 压缩、Video-MME。不要把 Qwen2.5-Omni 的 Video-MME 64.3% 填进 CogVideoX 的 Dynamic Degree。

[第 20 课](20_unified_understanding_generation.md)的图像统一、[第 41 课](41_discrete_any_to_any.md)的离散统一，都在问共享骨干时 mask 怎么写。本课在视频连续生成上把同一问再问一次。共享的可以是 3D VAE 或文本编码器；共享不了的是有效位置。Janus 把理解和生成分视觉编码器，那是图像课的结论。视频上 CogVideoX 并没有再训一个 SigLIP 当生成器眼睛，它的眼睛是 3D VAE。不要把第 20 课的双路径强行安在 CogVideoX 论文上。

**差距。** 缩小版没有 35M 片段，没有 13B，没有 60 名评估员。规模差距：数据、卡时、人工，钱可以补。机制差距：两张 mask、占用探针、四列账本，不买卡也必须做。缺这三条，把 CogVideoX 的 VAE 接到 MiniMind 的 Thinker 上，仍会出现一种假统一：总 loss 在降，杯子在下一帧消失，Video-MME 还没测就被写成已经会看视频。

缩小版也缺 3D 全注意力的稳定性问题。CogVideoX 写 2D+1D 在 5B 上更不稳。夹具没有注意力，不能复现 FVD 曲线。能复现的是：大运动需要时空直接可见，这个命题在 Figure 5 的示意图里，不在 CPU 的 32 个格子里。

**动手改造清单。**

1. **占用进入生成损失。** 在 `lesson_49.py` 增加 $\mathcal{L}_{\mathrm{occ}}=(o(\hat I)-o(I))^2$，总生成损失 $\mathcal{L}_{\mathrm{frame}}+\beta\mathcal{L}_{\mathrm{occ}}$。预算：CPU，小于 1 人日。预期：$\mathrm{forget}=1$ 时 $\mathcal{L}_{\mathrm{occ}}=1$，不再被纹理平均藏住。失败判定：$\beta>0$ 但 `gen_occupancy` 仍被当成不打印的量。
2. **相机标签当条件、不当 CE。** 给未来帧加一个离散运镜 id，生成器读它，理解 CE 仍然只问杯子。预算：CPU 或浏览器，0.5 人日。预期：改运镜 id 不改变 $M_{\mathrm{und}}$；理解答案不变。失败判定：运镜 id 被写进答案词表，CE 跟着运镜变。
3. **双项日志。** 训练脚本（若你有 1×24GB 跑 CogVideoX-2B LoRA）每步打印 `loss_und`、`loss_gen`、`o_hat`。预算：单卡数小时，数据用 128 条桌面片段即可，不追求 FVD。预期：两项可以反向运动。失败判定：只剩 `total_loss` 一个标量。
4. **跨账拒绝器。** 把第 47 课的 `may_compare` 扩到本课四列。预算：CPU，0.5 人日。预期：Video-MME 对 Dynamic Degree 返回假。失败判定：返回真却把差写成“生成进步”。

缩小版若要接到 MiniMind-O，最小增量是：在现有理解序列后面**不要**直接接未来像素。理解序列已经有视觉前缀和答案 CE。生成应另开一条 `GENERATE` 路径，条件可以共享冻结视觉或共享文本编码器，损失张量不要广播到前缀。最小增量不是：再训一个联合总 loss。联合总 loss 会让第 5.11 节的尺度问题立刻出现。路径开关可以继续用第 20 课的 `UNDERSTAND` / `GENERATE`，外面套一层视频帧缓冲。缓冲按任务打标签，比重新标注“视频 token 既是答案又是未来”更便宜。有卡再做 2B LoRA；没卡时路径开关加规则即可：问句以“还在吗”结尾走理解列；以“下一秒”结尾走生成列；两种问句同时出现则写两条样本，不要写一条双头样本却共用 mask。Lab 的两块面板就是这条规则的人肉版。

CogVideoX 的密 caption 管线还可以当反例读。他们用理解模型给生成数据写字，目的是让生成器更听 prompt。这是理解**服务于**生成数据，不是理解指标可以填进生成表。反过来，用生成视频去增广 Video-MME 训练，是生成服务于理解数据，第 54 课才会问分布。本课两边都不做增广，只做分账。读到 “we can perform video-to-video generation by connecting CogVideoX and CogVLM2-Caption” 时，把它记成数据管线，记成“已经统一”就失败。连接两个模型不是共用 $M_{\mathrm{und}}$。两个模型各有各的有效位置，中间是文字。文字当桥，mask 仍是两张。

**顺手复现。** 第 01 课“答案才进 CE”应能在 $M_{\mathrm{und}}$ 再现。第 20 课“理解前缀与生成 latent 不是同一种 token”应能在本课四段序列再现。第 33 课“像素 $L_2$ 会被不可预测纹理主导”应能在 `copy_l2 > gen_l2` 再现，方向相同，对象从接触边换成杯子占用。第 47 课把 Video-MME 留在 C2，本课把它继续挡在生成账门外。CogVideoX 的 2.74 对 Kling 2.17 不能在 CPU 上再现。HunyuanVideo 的 41.3% Overall 不能在 32 个格子上再现。能复现的是协议方向。

对照时还容易把第 28 课的动作 flow 拉进来。可以拉，但只拉“速度场写在哪一套坐标”这一句。第 28 课的样本是动作块 $A\in\mathbb{R}^{H\times d}$，本课是视频 latent。HunyuanVideo 的 Euler 积分步属于生成器内部迭代，记进推理步数，不记进理解 CE。谁把 50 步 Euler 写成 Video-MME 的抽帧 $F$，评测输入和生成采样就被兑在一起。CogVideoX 论文表 7 的 50 步同理：那是扩散步，不是理解模型看了 50 帧。步数、帧数、token 数三个词看起来都能当长度，单位全不同。本课改造清单第 2 条的运镜 id，也必须写进生成条件，用 $M_{\mathrm{gen}}$ 过滤，而不是因为“已经是离散编号了”就准许进 CE。编号进 CE 是第 41 课的作业，本课没有码本。

若要把本课分账接到真实 CogVideoX 微调，最小增量是：现有 latent 损失保持不动，另建一个占用头或检测器，只在评测时读 $o$，训练初期 $\beta=0$。最小增量不是：把 Video-MME 题拼进扩散 batch。评测头可以继续用占用公式，外面套一层阈值。阈值 0.15 属于夹具，真实视频要重新标定，标定过程写进协议卡，不要把 0.15 写进模型卡当物理精度。没卡时占用头加规则即可：均值亮度靠近背景则 $o$ 低。Lab 的 forget 滑条就是这条规则的人肉版。

下一课规格是三维资产生成：网格、高斯、辐射场要分解码器。本课停在视频的两本账。不要为了给第十六幕一个更响的开头，把杯子占用升级成“已经会物理世界”。占用只证明生成器可以在理解过关时把物体抹掉。抹掉被测到了，账才分开。三维课会问解码器契约，不再问 Video-MME。见[第 50 课](50_3d_generation.md)。有人提前把高斯球半径写进本课 Lab，属于抢题。本课 Lab 只有占用和 mask，没有球，没有网格，没有辐射场。规格在 COURSE_EXPANSION 里，本课交付以五个隔离文件为准。第 50 课接三维解码器，本课不预支那张契约，也不把占用写成网格。

还差一张“什么时候可以开始训 2B”的门禁。四条都勾上才允许碰 GPU：CPU 16 条为真；Lab Gate 亮过一次；协议卡四列都有名字；Sora 行没有层数。四条缺一，2B 的显存占用再低，也只是在未分账的总 loss 上烧卡。烧卡改变不了 164 和 165 的定义。定义先于权重。权重可以明天再下，定义必须今天写在纸上。纸上没有下标，仓库里的 `lesson_49.py` 就是那张纸的机器版。先跑机器版，再决定要不要下载 5B。下载之前把 `skipped-no-gpu` 写进记录，比先下载再假装已经评了 VBench 更接近本课的验收口径。

## 14. 论文与必读材料

1. [CogVideoX](https://arxiv.org/abs/2408.06072)。带着问题读：3D causal VAE 的 8×8×4 压的是生成对象还是理解 token？专家 AdaLN 调制的是哪一侧 hidden？3D 全注意力相对 2D+1D 解决的是大运动还是 CE？Table 3 的 Dynamic Degree 与 Multiple Objects 为什么不能兑成一个生成分？人工 Physics 0.667 和本课 $o$ 差在成功定义？附录 Table 5 的 30/42 层、1920/3072 维，写进报告时如何避免说成 Sora 的层数？高质量微调 20% 子集让语义略降，说明生成账内部还要再切。

2. [HunyuanVideo](https://arxiv.org/abs/2412.03603)。带着问题读：Flow Matching 的 $x_1$ 是数据还是噪声？双流到单流和 CogVideoX 专家 AdaLN 各保住什么？Table 2 的 $(d_t,d_h,d_w)=(16,56,56)$ 是位置编码通道，不是理解词表。Table 3 里 Visual 低于 Gen-3、Overall 仍第一，幻灯片若只截 Visual 会发生什么？14 类运镜写进 JSON 之后，评测应记可控列还是 Video-MME？VAE Table 1 的 PSNR 是重建，不是生成器听 prompt。

3. [HunyuanVideo 1.5](https://arxiv.org/abs/2511.18870)。带着问题读：8.3B 加 16× 空间压缩，省的是 DiT token 还是理解前缀？SSTA 的 1.87× 相对的是 FlashAttention-3，不是第 10 课 EVS。T2V Rating 五维为什么必须分列？对 Veo3 的 GSB win rate $-10.32\%$ 如何原样入账？13.6GB 的前提是卸载和 tiling。Muon 与 RLHF 改变的是生成分布，不改变 $M_{\mathrm{und}}$ 的定义。

4. [Video-MME](https://arxiv.org/abs/2405.21075) 与[第 47 课](47_eval_taxonomy.md)。带着问题读：900 / 2700 / 短中长 / 有无字幕，哪一项进本课协议卡？75.0% 为什么进不了 VBench 列？长视频 67.4% 掉点的三条原因里，哪一条是抽帧 $\Delta t$，哪一条不该被生成器的 10 秒时长救？

5. OpenAI 公开叙述 [Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/)。带着问题读：哪些句子可以引用（spacetime patch、压缩再切、扩散 Transformer、recaption、物体永久“常常但并不总是”、玻璃与进食失败）？哪些格子必须写未公开（层数、参数量、VAE 比、完整损失、训练集大小）？为何 2402.17177 这类综述不能当官方结构？读完用一句话说明：本课为什么不声称复现 Sora。

6. [第 09 课](09_native_video.md)、[第 10 课](10_video_token_reduction.md)、[第 20 课](20_unified_understanding_generation.md)、[第 33 课](33_world_model_latent.md)、[第 41 课](41_discrete_any_to_any.md)。带着问题读：时间轴、压缩、图像 flow、JEPA、离散 mask，各停在哪一句，本课从哪一句接着写？哪一句若被本课重做，算抢题？

读这六组时带一张空白的双列表。左栏只准填理解，右栏只准填生成。读完仍空着的格子写“未给出”，不要用另一篇的数字填上。CogVideoX 的 96.8 Human Action 不要填进 HunyuanVideo 的 Motion 66.5%。Video-MME 的 75.0% 不要填进夹具的 $p(\text{还在})=0.917$。夹具的 0.917 只属于 logits $(0.1,2.5)$。HunyuanVideo 1.5 的指令跟随 61.57 不要填进 13B 的 Text Alignment 61.8：两个 61 看起来像同一格，版本、评测方法和样本都不同。看起来像，是最危险的抄法。

结构化 caption 的字段清单建议抄一遍当默写。短描述、密描述、背景、风格、镜头类型、灯光、气氛，再加元数据标签。七个字段里，镜头类型进相机列；密描述进生成条件；没有任何一个字段等于 Video-MME 的选项字母。字段越多，越容易误以为“已经很理解了”。理解的成功定义是答题，生成的成功定义是像素或 latent 上的回归加人工观感。字段是条件的结构，不是分数的结构。HunyuanVideo 1.5 还加了 I2V 教学式 caption：只描述相对首帧的变化。那是生成条件的另一种写法，仍然不是 CE 答案。把它误抄进 $M_{\mathrm{und}}$，I2V 会开始“回答”运动描述，而不是画出运动。

读完本课，手里应有两张能执行的 mask、一个会消失的杯子、四列不许横比的数字。第十六幕从这里开始：生成是另一条时间轴。轴可以和理解轴读同一段视频，账必须分开写。结尾只值一句事实：答对“还在”，填不成下一帧的杯子。

阅读时建议按四天拆开。第一天只手算 197 下标和 softmax，抄占用公式。第二天打开 CogVideoX 第 2 节和第 4 节表，把 VAE、专家 AdaLN、VBench、人工四项填进右栏。第三天打开 HunyuanVideo Table 2、Table 3 和 1.5 的 Rating / GSB，相机 14 类单独一列。第四天跑 CPU 与 Lab，把 5.1 节的表对到 `metrics`。任何一天都不允许给 Sora 填层数。四天结束时，若还无法在白板上面出四段序列，就从 Step 0 的断言再抄一遍下标，不要从宣传视频倒推字段。

若只剩半小时，不要从 Sora 页面倒着读。打开 CPU 文件，从 `ANSWER_INDEX=164` 与 `FUTURE_START=165` 往回追：哪一段进 CE，哪一段进帧差，占用怎样从 1 变 0。能把这三问答给另一个人听，本课的分账就算落地。宣传视频留给有余力的晚上，而且只能用来核对论文里已经写明的分辨率和秒数，不能用来发明本课没写的模块。白板上一行四段画不出来时，先把 `M_und` 和 `M_gen` 写成两个方框，中间只连“同一段视频”，不要先画 DiT。DiT 可以后填，方框不能省。方框画好后再标 164 和 165，数字必须来自夹具。数字丢了，第二天你自己也会把两张 mask 加回去。这半小时练习的产出是一张纸，不是一段观后感。纸上有下标、有 0.0868 和 0.1096 两个损失、有“交为空”三个字，就算交了短作业。若连纸也懒得画，至少把 CPU 的 `metrics` 抄下来：`intersection_size` 必须是 0，`understand_answer` 必须是 1，`gen_occupancy` 必须是 0。三个数对不上，先不要读第 13 节的改造清单。改造清单是给分账已经站住的人用的；分账没站住时，加 $\beta\mathcal{L}_{\mathrm{occ}}$ 只会让错误长出一只新脚。三个数里最容易被忽略的是 `intersection_size=0`：有人把 CE 和帧差写进同一次 `loss.backward()`，两项标量仍可能分开打印，有效位置却已经 OR 在一起。本课验的是下标集合，不是打印了几个标量。标量可以有两项，下标必须不相交。Lab 的两条色带就是给人眼看这两段用的：答案段和未来段同时亮，就回去改 mask。看的时候看位置，不要只看 forget 滑条。滑条决定杯子消不消失，位置决定账算不算对。消失是探针，位置是协议。把这一句贴在 Lab 旁边，拖滑条之前先念一遍，比把 13B 参数背下来更接近本课要教的东西。念完再看四格揭晓：理解答案、生成占用、两段 $L_2$、mask 交。四格齐了，预测也选对了，Gate 才会有机会亮。预测选错时四格再漂亮也过不了，因为你还没承认分裂是协议而不是界面巧合。先改预测，再揭晓，不要指望把 forget 拖到 1 去迁就错误选项。选项是命题，滑条是操作。
