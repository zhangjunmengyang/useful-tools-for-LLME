---
id: 20_unified_understanding_generation
title: "统一理解与图像生成"
summary: "骨干、数据、更新预算全都固定后，“语义视觉路径＋低层 VAE 路径”双管齐下，真能比单独一条路更好地兼顾看懂图和画出图这两件事吗？"
unit: frontier
play_tools: []
checkpoints:
  - "分清三种视觉表示装了多少信息：VQ token、VAE latent、语义视觉 feature。"
  - "亲手实现一个最小的 flow matching（学“噪声到图像”的变形路径）、采样和 classifier-free guidance。"
  - "公平比较三臂：只用 VAE、双路径融合、Janus 式理解生成分家。"
  - "用 specialist gap 和 interference delta 两个量，测共享训练到底是两头都帮，还是互相拖后腿。"
---

# 第 20 课：统一视觉理解与图像生成（选修）

> 类型：高阶选修，不属于语音双工主线的必要依赖  
> 建议周期：2–4 周  
> 推荐硬件：8×80GB 做缩小版机制实验；复现论文规模通常需要更多资源  
> 独立性：从本课冻结的完整 MiniMind-O、SigLIP2 与 SD VAE checkpoint 开始，不要求完成实验 2–19  

## 1. 理解和生成为什么会冲突

前 19 课覆盖了语音、视觉理解、实时交互、现代骨干和后训练，但系统不能生成图像。第 20 课研究一个独立问题：同一组核心 Transformer 参数能否同时支持视觉理解与图像生成。

视觉理解通常压缩图像：CLIP、SigLIP 和 VLM 需要保留对象、位置与关系，同时忽略不影响语义的光照和纹理变化。图像生成则需要重建颜色、纹理和空间细节。两类任务不仅需要不同的信息，训练目标也不同：理解通常使用离散的 next-token 交叉熵和因果注意力，主流生成使用连续的 diffusion 或 flow 目标和双向注意力。共享参数还会带来 interference，一个任务的更新可能使另一个任务退化。Show-o、Janus、BAGEL 和 GPT-4o 的原生图像生成采用了不同的共享与解耦方案，本课在缩小设置中比较这些路线。

实验冻结 MiniMind-O 核心，并加入三个模块。理解侧使用 3-block adapter，将 SigLIP2 或 VAE 特征压成 32 个前缀 token；生成侧使用 flow head，在 VAE latent 上学习速度场；顶部 4 层加入共享 residual adapter。预注册实验比较 VAE-only、语义与低层特征双路融合，以及理解、生成分别编码三种方案。

最终 checkpoint 同时接受图像问答和文本生成提示。图像生成使用 32 步 Euler 积分从噪声得到 256×256 图像，不调用外部绘图 API。报告记录三臂的参数量、训练预算、理解与生成指标及置信区间。本课是选修，不影响语音双工主线。

本课术语：

| 术语 | 简要解释 |
|---|---|
| VQ tokenizer | 把图切块后查"视觉字典"变成离散编号的压缩器，像素版码本 |
| VAE latent | 变分自编码器把图压成的连续小张量（本课 4×32×32），能近乎无损解回像素 |
| flow matching | 训练一个速度场：告诉每个噪声点往哪走、走多快，才能一路流回真实图像 |
| velocity（速度目标） | 直线路径下就是"噪声减干净图"，网络要回归的监督量 |
| DiT | 把扩散模型的 U-Net 骨干换成 Transformer 的做法 |
| adaLN-Zero | 用时间步信号调制每层归一化的缩放/平移，门控零初始化，接上冻结骨干时零扰动 |
| CFG | classifier-free guidance：有条件预测减无条件预测再放大，换取更听 prompt 的图 |
| 语义路径 / 低层路径 | SigLIP2 特征管"有什么"，VAE latent 管"长什么样" |
| interference | 联合训练时一个任务把共享参数拽偏，另一个任务掉点 |
| retention（保留率） | 统一模型对单任务专用模型的能力比值，衡量统一有没有白拿 |
| iso-data / iso-update | 三臂吃同样数据、走同样更新步数，但不承诺总算力相等 |

## 2. 本课要解决的问题

MiniMind-O 主要覆盖多模态输入、文本推理和语音输出；Nemotron Omni 重点覆盖多模态感知与文本推理。两者都只"进"不"画"。本课研究缺的那块：让同一组核心 Transformer 参数同时处理视觉理解和图像生成。

规矩先立死：图像必须由模型内部的 visual head 生成，不能调用外部 Stable Diffusion API 代替——外挂工具也能出图，但那验证不了"同一组参数干两种活"这件事。实验比较以下机制：

- 语义理解表征与像素生成表征能否共享；
- autoregressive language modeling 与 flow/diffusion objective 如何共存；
- 为什么理解和生成需要不同粒度的视觉编码；
- 联合训练怎样避免两种能力互相伤害。

页面机制图先画最清楚的 decoupled 执行链，把两条路各自吃什么、吐什么固定。`UNDERSTAND` 把图像依次送入冻结 SigLIP2 和 3-block understanding adapter，最终只给 core 32 个 visual prefix。`GENERATE` 不经过这条 adapter；prompt、256 个 noisy-latent token $z_t$ 直接进入 core，时间 $t$ 只通过 adaLN 条件进入。训练图像的 VAE latent $x_0$ 只位于训练目标一侧，用来构造 $z_t$ 和速度目标；推理时没有 $x_0$。32 个理解前缀与 256 个生成 latent 不是同一种 semantic token，别把它们混为一谈。

主线只实现**图像版缩小实验**。视频生成保留为扩展，不进入最低交付或主矩阵。

## 3. 要验证的结论与失败条件

先定义本节使用的三个名称：

- **语义视觉路径**：读取冻结视觉编码器的特征，主要保留对象、文字和空间关系；
- **VAE 低层路径**：读取 VAE 的连续 latent，主要保留颜色、纹理和像素布局；
- **专用模型**：只训练图像理解或只训练图像生成的模型，用作能力上限对照。

主假设是：固定语言骨干、训练数据和更新预算，并单独报告实际计算成本时，同时使用语义视觉路径和 VAE 低层路径，比只使用 VAE 低层路径更能兼顾图像理解与图像生成。

为什么要在看结果前把尺子固定？因为人挑尺子的手是不自觉偏的：先看到结果再选指标，总能选出一把对自己有利的。所以在查看 test 结果前，先固定两个主指标。`U_primary` 是四项理解分数的宏平均：caption scene-graph F1（scene graph：把图描述成"对象-属性-关系"三元组的结构化表格，F1 衡量抽取得全不全、对不对）、属性问答准确率、计数问答准确率和关系问答准确率。`G_primary` 是固定生成 prompt 上对象、颜色、计数和空间关系准确率的宏平均。各分项都在 $[0,1]$ 内，数值越高越好。主比较为：

$$
\begin{aligned}
\Delta U_{B-A} &= U_{\text{primary},B}-U_{\text{primary},A}, &
\Delta U_{B-C} &= U_{\text{primary},B}-U_{\text{primary},C},\\
\Delta G_{B-A} &= G_{\text{primary},B}-G_{\text{primary},A}, &
\Delta G_{B-C} &= G_{\text{primary},B}-G_{\text{primary},C}.
\end{aligned}
$$

生成非劣阈值预先固定为绝对下降 0.02。"非劣"的意思是：双路径 B 的卖点是理解更好，生成允许比 A/C 略差，但差距超过 0.02 就算塌了。对三个 seed 的同一批 test scene 做两层配对 bootstrap（bootstrap：反复有放回地重抽样本、重算指标，用抽出来的分布估计不确定度）：先重采样 seed，再重采样 scene id，共 10,000 次，报告 95% 置信区间。只有 $\Delta U_{B-A}$ 和 $\Delta U_{B-C}$ 的置信区间下界都大于 0，同时 $\Delta G_{B-A}$ 和 $\Delta G_{B-C}$ 的置信区间下界都大于 $-0.02$，才记为主假设得到支持。

实验是否有效与主假设是否得到支持分开判断，这是两个问题。三臂数据顺序、更新数、可训练参数或采样设置不一致，属于实验无效；这些条件都满足并完成预注册置信区间后，即使结果没有达到上述阈值，也是一项有效的负结果，不能在 test 后改指标或阈值。

实验还要比较：

1. 理解与生成应共享输入投影层、共享 Transformer，还是只共享高层语义？
2. 文本用自回归 head、图像用 flow head，是否比把所有视觉 token 都离散化后统一自回归更适合小模型？

主假设是否得到支持只按上面的 B−A、B−C 和 0.02 非劣规则判断。若 Arm B 明显弱于两个固定的专用模型，则另记为绝对能力目标未达到，并报告冲突发生在哪个任务、哪一层和哪个训练阶段。不能只因为同一个 checkpoint 可以切换 head，就判定统一训练成功。

## 4. 与其他课程的边界

选修课最容易犯的错是什么都想碰一点。下列能力不进入本课的训练变量或验收：

- 用户/助手语音双工；
- 神经音频 codec；
- Nemotron 的 text/image/video/audio 感知；
- 大规模视频生成；
- 仅通过工具调用外部生成模型。

本课只研究模型内部的理解 CE、visual flow objective、共享参数和表示冲突。

## 5. 三种架构路线

统一理解与生成，公开文献里主要走三条路。三条路的差别是：视觉信息用什么形式进模型、用什么目标出模型。

### 路线 A：统一离散 token

该路线先用 VQ tokenizer 把图像量化成离散视觉 id。文本 token 和视觉 token 随后进入同一个自回归 Transformer；模型根据任务标记继续预测文本，或继续预测视觉 token。一句话：把图也当"文字"，一个 next-token 目标通吃。Janus 的生成侧走的就是这条离散路线。

适用优势：

- 训练目标统一；
- 序列接口简单；
- 任意模态交错自然。

主要代价：

- 视觉序列长；
- 离散 tokenizer 重建质量和语义能力冲突；
- 图像生成速度慢。

### 路线 B：AR + discrete diffusion

该路线让文本理解使用 autoregressive（AR）目标，让图像生成使用视觉 token 上的 discrete diffusion 目标（在离散 id 上做"遮住再猜"式的迭代去噪，而非逐个 token 自回归）。两种任务共享 Transformer trunk，但 task token、attention mask 和 loss-bearing position 分开定义。

Show-o 是该路线的代表。阅读实现时需要核对 task token、attention mask 和两种 loss 分别作用在哪些位置。

### 路线 C：语义/低层双路径 + flow head

该路线用 semantic encoder 保留对象与关系信息，用 VAE latent 保留颜色、纹理和空间细节。两路经过各自 projector 后融合成视觉表示，再送入共享 LLM。language head 对文本做 AR 预测，flow head 则预测 VAE latent 的 velocity；两个 head 共享 core，但目标和输出空间不同。

本课以 Show-o2 类双路径为主线，以 Janus 类理解/生成编码解耦作为对照。实现后用 tensor trace 检查理解路径和生成路径各自读取了哪一路视觉表示——这一步不做，后面出了问题你分不清是路线错了还是接线错了。

## 6. 学习目标

完成后应能：

1. 解释 VQ tokenizer、VAE latent 和语义视觉特征的差别；
2. 实现最小 flow matching 训练与采样；
3. 把视觉理解 head 与生成 head 接到同一骨干；
4. 设计 VAE single-path、dual-path fused、Janus-style decoupled 三臂公平实验；
5. 测量 joint training 的 interference delta；
6. 区分"模型原生生成"与"调用外部生成工具"；
7. 准确说明 Show-o2/BAGEL 论文与公开 recipe 的边界。

## 7. 原理:边造边讲

四个机制，每个按同一节奏走：为什么需要（直觉）、怎么运转（机制）、精确定义（数学）、在哪个文件（代码落点）、怎么证明做对了（验证）。

### 7.1 VQ 与 VAE：图像的两种"进出口格式"

模型只会处理 token 或向量，图像必须先换算。两种换法：一种是查字典——把图切块，每块在有限词表里找最像的编号，得到离散 id，这就是 VQ；另一种是打压缩包——把图压成一个连续的小张量，要用时再解压回像素，这就是 VAE。离散 id 可以直接套语言模型的交叉熵；连续 latent 没有"词表"可查，需要 diffusion 或 flow 这类为连续量身定做的目标。

VQ tokenizer 将图像映射为有限词表中的离散 id，可以直接使用 token CE。VAE 将图像映射为连续 latent，适合用 diffusion 或 flow 预测连续变化。本课的 VAE 输出 `[4, 32, 32]`：4 个通道、空间下采样 8 倍。

`tokenizers/semantic_encoder.py` 与 `tokenizers/vae_adapter.py`（见第 9 节目录）；VAE 用冻结的 `stabilityai/sd-vae-ft-mse`。

实现前分别运行 VQ/VAE encode-decode，并记录：

- reconstruction loss；
- KL regularization；
- codebook utilization（码本利用率：字典里的词有多少真被用上；利用率崩了叫码本坍缩）；
- latent spatial downsampling；
- tokenizer reconstruction upper bound。

最后一项最重要：tokenizer 自己 encode 再 decode 都救不回来的细节，后面的模型再强也救不回来。这个上限必须先量出来，否则生成模糊时你会冤枉错模块。

### 7.2 Diffusion 与 flow matching：学一张"导航场"

生成连续图像，逐 token 查字典行不通，换一种想法：把"从噪声变成图"看成一场流动。flow matching 训练的是一张导航场——空间中每个位置、每个时刻，告诉你往哪个方向走、走多快；照着走，随机噪声就能一路开到真实图像。类比失效处：真实导航的目的地是事先指定的一个点，这里是把整个噪声分布运输到整个数据分布，单个样本最终落在哪，由起点和整张场共同决定。

Flow matching 训练向量场 $v_\theta$，使采样 ODE（常微分方程：按速度场一小步一小步积分走轨迹）能把噪声分布运输到数据分布。本课采用从 data endpoint 到 noise endpoint 的直线路径。

令数据 latent 为 $x_0$，噪声 $\varepsilon\sim\mathcal N(0,I)$，时间 $t\sim\mathcal U(0,1)$。对直线路径：

$$
\begin{aligned}
x_t &= (1-t)x_0+t\varepsilon, \\
u_t &= \frac{\mathrm d x_t}{\mathrm dt}
     = \varepsilon-x_0, \\
\mathcal L_{\text{flow}}(\theta)
&=
\mathbb E_{x_0,\varepsilon,t}
\left[
\left\|
v_\theta(x_t,t,c)-u_t
\right\|_2^2
\right].
\end{aligned}
$$

训练时网络输入为 $(x_t,t,c)$，监督目标是速度 $u_t=\varepsilon-x_0$。直线路径的速度处处相同，所以监督量简单到就是"噪声减干净图"。采样时从 $t=1$ 的噪声反向积分到 $t=0$ 的数据。

`training/generation_step.py` 与 `sampling/flow_sampler.py`；训练与采样两段参考实现见 Step 2。

修改 path、prediction target 或积分方向时，三处必须一起修改，并用二维 toy distribution 验证。这三处是同一份约定的三个副本，改一漏二是本课最常见的翻车方式：loss 照降，采出来全是噪声。

### 7.3 DiT/adaLN：告诉网络"现在是第几步"

同一个网络在 $t=0.9$（几乎全是噪声）和 $t=0.1$（几乎成图）要干完全不同的活，必须知道当前时刻。DiT 在 Transformer 中处理 noisy latent token；adaLN 根据 timestep 和 condition 调整每层归一化后的 scale/shift，等于在每一层拧一个随时刻变化的旋钮。adaLN-Zero 将残差门控初始化为接近零——刚接上时整个生成外挂对冻结 backbone 的输出零贡献，训练再慢慢拧开阀门；这是把新目标接到预训练模型上又不毁掉它的关键技巧，减少接入预训练 backbone 时的扰动。

精确的 modulation 公式、作用范围和逐层伪代码在 11.4 节给出；本课的 adaLN-Zero 只调制 256 个 latent query，prompt token 走冻结 block 的原路径。

11.4 的 `generation_block` wrapper；timestep MLP 与 4×adaLN-Zero 的参数账本在 11.6。

实现后分别检查：

- timestep embedding；
- condition injection；
- adaLN/adaLN-Zero；
- classifier-free guidance；
- sampling steps 与质量/速度。

三臂所有 generation batch 从训练开始就使用相同的 `condition_dropout=0.10`（训练时按 10% 概率把 prompt 抹掉，逼模型也学会无条件生成，这是推理时做 CFG 的前提）。这个值同时用于 Stage 1 generation warmup 和 Stage 2 joint training，主矩阵结束前不扫描也不修改。

### 7.4 多任务冲突：一副身子两份工

理解任务常希望表征对无关颜色微扰和纹理变化保持稳定；生成任务则需要重建这些细节。同一组共享参数被两股梯度往相反方向拉，就可能产生梯度冲突——这正是第 1 节说的第三道坎，也是本课要拿数据说话的地方。

训练中记录两项 loss、任务梯度 cosine（两个任务梯度方向的夹角余弦，负值代表在拔河）和分任务 validation，判断冲突出现在哪个阶段，而不能只看 joint loss——两项 loss 的加权和平滑下降时，完全可能一项在涨一项在跌，joint 曲线把冲突盖住了。逐步的审计流程在 Step 9。

## 8. 独立起点

为保证三臂可以复现，默认起点固定为下列 checkpoint，不使用未指定的"任意 0.5B–1.5B VLM"：

| 组件 | repo / checkpoint | revision（核验于 2026-07-23） | 本课维度与许可 |
|---|---|---|---|
| code | [`jingyaogong/minimind-o`](https://github.com/jingyaogong/minimind-o) | `a10fa6c148ed274d66f96dc119689e93e01be823` | Apache-2.0 |
| shared backbone + tokenizer + Talker | [`jingyaogong/minimind-3o`](https://huggingface.co/jingyaogong/minimind-3o) | `ee3febbd08cc5b2bd41c039c825a8934232fee33` | hidden size 768；Apache-2.0 |
| semantic encoder | [`jingyaogong/siglip2-base-p32-256-ve`](https://huggingface.co/jingyaogong/siglip2-base-p32-256-ve) | `9465d1dc89db6bc6227c5b6b0e0ca9b940325d62` | 256×256、64 tokens、hidden 768；Apache-2.0 |
| continuous image tokenizer | [`stabilityai/sd-vae-ft-mse`](https://huggingface.co/stabilityai/sd-vae-ft-mse) | `31f26fdeee1355a5c34592e401dd41e45d25a493` | `AutoencoderKL`、4 latent channels、8× spatial downsample；MIT |

运行前生成 `artifacts.lock.json`，记录 repo revision、每个下载文件 SHA256、tokenizer files 和模型卡许可。三臂从同一份完整 MiniMind-O 权重复制。Talker 始终冻结，但 shared core adapter 仍可能影响其他路由——冻结的嘴接在被外挂改过的身子上，声音照样可能变——因此每次 checkpoint 都要运行 speech regression。

更换更大的 backbone 时创建新实验编号，并重新生成三臂配置与 reference。主矩阵中途不能替换 backbone。

## 9. 推荐代码边界

目录按视觉表示、共享模块、训练步骤、采样和评测分离。理解与生成可以调用相同 core，但不得通过全局状态隐式切换 mask 或 head——隐式切换是这类双任务代码里最难查的 bug 来源，路由必须显式：

```text
unified_generation/
  tokenizers/
    semantic_encoder.py
    vae_adapter.py
  fusion/
    dual_path_projector.py
    single_path_projector.py
  heads/
    language_head.py
    flow_head.py
  training/
    understanding_step.py
    generation_step.py
    joint_sampler.py
  sampling/
    flow_sampler.py
    cfg.py
  eval/
    understanding_eval.py
    generation_eval.py
    interference_report.py
```

Talker 保持冻结且不参与 loss。`speech regression` 通过独立 eval 路由运行。

## 10. 数据 recipe

### 10.1 统一数据字段

同一 schema 同时容纳理解、生成和可选 editing。`task` 决定哪些 target 字段必须存在；loader 对缺失的必需字段直接报错，不创建空 tensor——静默补零的 loader 会把数据 bug 伪装成模型 bug：

```json
{
  "id": "unified-00001",
  "task": "caption|vqa|t2i|editing",
  "input_text": "a red chair in a white room",
  "input_image": "optional/source.jpg",
  "target_text": "optional answer",
  "target_image": "optional/target.jpg",
  "mask": "optional/edit_mask.png",
  "source": "dataset",
  "source_revision": "immutable-revision",
  "asset_origin": "generated|source-url|local-consented",
  "license": "license-note",
  "redistributable": false,
  "split": "train",
  "quality": {
    "resolution": [512, 512],
    "aesthetic": 0.0,
    "watermark": false
  }
}
```

### 10.2 数据类型

三类数据分别决定 language head、flow head 或两者的监督范围：

#### 理解数据

- image caption；
- VQA；
- OCR/document；
- spatial relation。

#### 生成数据

- text-to-image；
- image reconstruction；
- caption-conditioned image；
- 可选 editing/inpainting。

#### 联合/交错数据

- image → description → edited image；
- multiple images + text；
- question → answer + visual output。

主矩阵不使用联合/交错记录；它们只在单任务路径分别通过后进入扩展实验。

### 10.3 主实验数据：`synthetic_shapes_v1`

主矩阵使用本地确定性图形数据，以控制 scene graph、文本答案和图像 target。为什么用合成图形不用真实照片？因为主矩阵比的是架构：合成数据的每个对象、颜色、关系都是程序生成的，理解答案可以程序判分，生成结果可以程序验收，还没有网页图片许可和未清洗 caption 的干扰。真实图像另开迁移实验（10.4）：

```yaml
recipe: synthetic_shapes_v1
version: 1.0.0
seed: 20260723
canvas: [256, 256]
scenes:
  train: 100000
  dev: 5000
  test: 10000
objects_per_image: [1, 6]
vocab:
  shapes: [circle, square, triangle, star, hexagon]
  colors: [red, green, blue, yellow, purple, orange, black, white]
  sizes: [small, medium, large]
relations: [left_of, right_of, above, below, inside, overlap]
records_per_scene:
  caption: 1
  vqa_attribute: 1
  vqa_count: 1
  vqa_relation: 1
  text_to_image: 1
split_key: scene_graph_hash
license: CC0-1.0
```

生成器先采样 canonical scene graph，再为每个 scene 确定性派生 4 条理解记录和 1 条生成记录。因此 train 的 100k scene 对应 `400k understanding records + 100k unique generation records`。train/dev/test 的 scene graph 不重复；测试集另含 20% 未见过的颜色—形状组合，用于测 compositionality（组合泛化：见过红圆和蓝方，能不能对付蓝圆）。交付物包含生成器代码、字体、图形库版本和输出 manifest hash。

主矩阵固定 sampler 顺序，不根据运行时 loss 临时选择任务：

| 单位 | 固定定义 |
|---|---|
| paired update unit | 先跑 1 个 256 条的 understanding global batch，再跑 1 个 256 条的 generation global batch；两个 loss 分别按有效 answer token 和 latent element 求均值，累积梯度后只调用 1 次 `optimizer.step()` |
| canonical data epoch | understanding 的 400k records 无放回遍历 1 次；generation 的 100k records 按固定 seed 洗牌后遍历 4 次 |

四类 understanding record 使用 round-robin 分层队列；不足一个 global batch 的尾部采用确定性 rollover。每个 epoch 的种子为 `20260723 + epoch_id`，三臂共享 `sampler_order.arrow`。每个 canonical epoch 两侧各暴露 400k records；`0.5/0.5` 表示 paired unit 内两次前向的 loss 权重，不表示原始 scene 数量比例。

### 10.4 真实图像迁移实验

主矩阵冻结配置后，再增加 COCO Captions + VQAv2 transfer lane。执行规则：

- 固定 COCO 2017 image/caption annotation 包和 VQAv2 annotation 文件 SHA256；
- 每张图片记录原始 COCO id、Flickr/source URL、来源字段和下载时间；
- 没有逐资产许可证据时一律 `redistributable=false`；
- 只发布 id/annotation/派生指标，不重新分发图片；
- CC3M 不进入默认 recipe；"许可可用部分"不是可执行定义；
- GenEval prompts 只用于评测，绝不进入训练。

### 10.5 规模档位

规模只改变数据量、分辨率和可训练范围；每档使用独立 experiment id：

| 档位 | 理解样本 | 生成图文对 | 分辨率 | 目的 |
|---|---:|---:|---|---|
| Toy | 10k | 10k | 256 | 验证 flow 与接口 |
| Pilot | 100k scenes→400k records | 100k unique、每 epoch 重复 4 次 | 256 | 三臂机制比较 |
| Standard | 1M–5M | 5M–20M | 256→512 | 可信统一实验 |
| Paper-scale | 数千万以上 | 数千万以上 | 图像+视频 | 不适合默认 8 卡 |

### 10.6 切分

切分后用 image hash、scene graph hash 和 caption 近重复检测验证无泄漏：

- 相同图像的 resize/crop 不得跨 split；
- caption 近重复去重；
- 理解和生成 test image 都不得进入训练；
- 人工 prompt benchmark 不用于训练；
- 数据来源分层报告，防止某一来源支配结果。

## 11. 目标架构

### 11.0 先固定两条执行链

机制图展示 Arm C 的 decoupled route，因为它能把边界画得最清楚：

- `UNDERSTAND`：image → frozen SigLIP2 `[B,64,768]` → 3-block adapter →
  `[B,32,768]` visual prefix → shared core → language head；
- `GENERATE`：prompt + 256 个 $z_t$ token → shared core → flow head；$t$ 通过
  adaLN-Zero 调制 latent query，不作为 visual prefix，也不经过 understanding
  adapter；
- $x_0$：只在训练侧由目标图像的 frozen VAE 得到，用来形成
  $z_t=(1-t)x_0+t\varepsilon$ 和监督 $u_t=\varepsilon-x_0$；它不是干净条件 token，
  推理时不可用。

后面的 A/B 只改变 understanding route 的输入来源，用于受控比较；三臂的 GENERATE 链都遵守第二条。机制图不是把 A/B/C 三条 understanding 消融叠在一张图里。

### 11.1 Semantic path

Semantic path 先把图像送入冻结的 SigLIP2，得到侧重对象、关系和文字的 semantic tokens，再由 3 个 cross-attention block 压成 32 个 visual prefix token，随后进入共享 core。该 adapter 只在 `UNDERSTAND` route 执行。

输出 shape 和 mask 必须在 trace 中保存；用对象、关系和 OCR probe 检查其信息——probe 探不出对象和关系，说明压缩把语义压丢了，先修 adapter 再谈联合训练。

### 11.2 Low-level path

生成训练把目标图像送入冻结 VAE encoder，得到 $x_0$。它先与噪声按时间 $t$ 混合成 $z_t$，再 patchify（把空间网格切成小块拼成 token 序列）成 256 个 token 并投影到 shared core hidden size。core 的输入是 prompt、$z_t$ token 和通过 adaLN 注入的 $t$；干净的 $x_0$ 不作为额外条件输入——把 $x_0$ 喂给模型等于把答案印在考卷上，训练指标会好看得毫无意义。

记录 VAE latent 的 `[B,C,H,W]`、scale 和 patchify 后 shape；用 reconstruction 指标检查颜色、纹理和局部结构的上限。

### 11.3 Fusion

Fusion 只属于 Arm B 的 understanding 消融。它把 SigLIP2 token 与目标图像的 VAE summary token 转成固定数量的 visual prefix，再交给 `UNDERSTAND` core。它不参与 `GENERATE`：256 个 $z_t$ token 仍直接进入 core，不能先压成 32 个理解前缀。

三臂必须输出相同的 $K$ 和 hidden size。启动时用 shape assertion 检查，不能依赖 padding 隐式补齐。

### 11.4 共享 core 的张量形状与注意力规则

固定 256×256 输入时，各路径 shape 为：

| 路径 | 输入与输出 shape |
|---|---|
| SigLIP2 | image → `[B,64,768]` |
| VAE encoder | image → `x0: [B,4,32,32]` |
| VAE patchify | patch=2 后为 `[B,256,16]`，Linear 后为 `[B,256,768]` |
| 机制图 / Arm C understanding 序列 | 32 个 SigLIP2-derived visual prefix + text |
| Arm B fusion 内部 | 32 个 semantic summary + 32 个 VAE summary → 32 个 visual prefix |
| generation 序列 | prompt text + 256 个 noisy-latent token $z_t$ |
| generation 时间条件 | $t$ → timestep MLP → adaLN-Zero，不占序列 token |

上面的 64 个 source summary 只存在于 Arm B 的 fusion 内部。第三个 cross-attention block 用 32 个 learned queries 读取这 64 个 source token，输出 32 个 unified visual prefix。Arm A/C 也各输出 32 个，因此三臂送入 MiniMind core 的 understanding prefix 均为 `[B, 32, 768]`。

这段 3-block 计算全部属于 understanding stack。生成时不复用它，也不把 256 个 $z_t$ token 先压成 32 个 token；否则生成路径、参数账本和机制图都会改变。

注意力的可见性规则是本节真正的骨架：理解侧文本因果、前缀双向；生成侧 latent 之间双向互看（一张图的左上角和右下角要同时商量），prompt 只被看、不看回。MiniMind attention 使用两种显式 block mask：

| 任务 | query | key | 可见性 |
|---|---|---|---|
| UNDERSTAND | visual prefix | visual prefix | 双向可见 |
| UNDERSTAND | text | visual prefix | 可见 |
| UNDERSTAND | text | text | causal |
| UNDERSTAND | visual prefix | text | 禁止 |
| GENERATE | prompt text | prompt text | causal |
| GENERATE | latent tokens | 全部 prompt text | 可见 |
| GENERATE | latent tokens | latent tokens | 双向可见 |
| GENERATE | prompt text | latent tokens | 禁止 |

同一个 MiniMind block 参数在两种 mask 下复用。timestep 不加到 prompt 或 latent token embedding；它先被编码成 `time_cond`，再通过 top-4 shared blocks 外挂的 gated adaLN-Zero adapter 只调制 latent query。原 MiniMind block 权重保持冻结。用构造好的全 1 attention probe 验证每个允许/禁止方向，并断言三臂的 core、hidden size、latent token 数和 head 参数量相同。

"外挂"在本课中只有一种执行顺序。设冻结 MiniMind block 的输入为 `h: [B,S,768]`，`latent_mask: [B,S,1]` 只在 256 个 noisy-latent 位置为 1。`time_cond: [B,768]` 等于 timestep MLP 输出加冻结的 `<|generate|>` task embedding，只作为 adaLN-Zero 的输入。文本条件通过 prompt token 和 block attention 进入 latent，不另建未计入参数账本的 condition projector。每层 modulation 计算：

$$
(\Delta_{\mathrm{msa}},\Gamma_{\mathrm{msa}},G_{\mathrm{msa}},
\Delta_{\mathrm{mlp}},\Gamma_{\mathrm{mlp}},G_{\mathrm{mlp}})
=W_{\mathrm{mod}}c_t+b_{\mathrm{mod}},
$$

六个输出的 shape 都是 `[B,768]`。`shift` 写作 $\Delta$，`scale` 写作 $\Gamma$，`gate` 写作 $G$。它们只作用于 latent query；prompt query 执行冻结 block 原本的 residual 路径。单层伪代码固定为：

```python
def generation_block(h, block, shared_adapter, time_cond, attn_mask, latent_mask):
    # h: [B,S,D], latent_mask: [B,S,1], time_cond: [B,D]
    d_msa, g_msa, gate_msa, d_mlp, g_mlp, gate_mlp = (
        block.adaln(time_cond).chunk(6, dim=-1)
    )

    n1_base = block.ln1(h)
    n1_latent = n1_base * (1 + g_msa[:, None, :]) + d_msa[:, None, :]
    n1 = where(latent_mask, n1_latent, n1_base)
    attn_delta = block.attn(n1, mask=attn_mask)
    attn_scale = where(latent_mask, gate_msa[:, None, :], ones_like(h))
    h = h + attn_scale * attn_delta

    n2_base = block.ln2(h)
    n2_latent = n2_base * (1 + g_mlp[:, None, :]) + d_mlp[:, None, :]
    n2 = where(latent_mask, n2_latent, n2_base)
    mlp_delta = block.mlp(n2)
    mlp_scale = where(latent_mask, gate_mlp[:, None, :], ones_like(h))
    h = h + mlp_scale * mlp_delta

    # UNDERSTAND 与 GENERATE 共用；纯文本和 Talker route 不调用这个 wrapper。
    h = h + shared_adapter(h)
    return h
```

`where` 的 mask 必须广播为 `[B,S,768]`。四个 `W_mod/b_mod` 全部零初始化；四个 shared adapter 的上投影也零初始化。初始化单测应证明：prompt token 与原冻结 block 的输出在公差内一致；latent 位置的 attention/MLP residual 增量为 0；纯文本和 Talker route 的 logits 与未接 wrapper 时一致。完成一次更新后，再检查 timestep 改变只会通过 modulation 改变 latent query，而不会改写 prompt 的 position 或 mask。

### 11.5 Heads

Head 根据显式 task token 读取对应位置：

- language head：next-token CE；
- flow head：将 256 个 hidden token 投影回 `[B, 4, 32, 32]` velocity；
- task router：由明确的 `<|understand|>` / `<|generate|>` token 选择 mask 和 head，不做隐式猜测。

第一版固定 dense MiniMind block 和现有 tokenizer，不同时加入 MoE、Mamba 或新视觉 tokenizer，避免增加未受控变量——一次只动一个东西，这条纪律从第 1 课贯穿到这里。

### 11.6 三臂逐模块定义与参数账本

本节固定三臂的可训练模块，避免表示路径和参数量同时变化——否则赢的那臂到底赢在路线还是赢在参数多，说不清。三个 arm 都实例化 `3 × CrossAttentionBlock`。每块使用 `d=768, heads=12, MLP=3072, bias=true`，并带独立的 `32×768` learned query bank：

| Arm | block 1 | block 2 | block 3 | core 最终视觉 token |
|---|---|---|---|---:|
| A | 32 queries 读取 256 个 VAE patch | 32 queries refine block 1 | 32 queries refine block 2 | 32 |
| B | 32 queries 读取 64 个 SigLIP2 token | 32 queries 读取 256 个 VAE patch | 32 queries 读取前两者的 64-token concat | 32 |
| C | 32 queries 读取 64 个 SigLIP2 token | 32 queries refine block 1 | 32 queries refine block 2 | 32 |

`refine` block 的输出进入下一个 block，第三块输出直接进入 core，因此三块都参与 forward。一个 cross-attention block 的参数计数为：

$$
\begin{aligned}
P_{\text{block}}
&= P_{\text{Q/K/V/O}}+P_{\text{MLP}}+P_{\text{2 LayerNorm}} \\
&= 12d^2+13d \\
&= 7{,}087{,}872, \\
P_{\text{query bank}}
&= 32d = 24{,}576, \\
P_{\text{understanding stack}}
&= 3\left(P_{\text{block}}+P_{\text{query bank}}\right) \\
&= 21{,}337{,}344.
\end{aligned}
$$

三个 arm 还共享完全相同的生成模块：

| 模块 | 固定定义 | trainable params |
|---|---|---:|
| latent input | `Linear(16,768,bias=true)` | 13,056 |
| timestep MLP | sinusoidal-256 → Linear 768 → SiLU → Linear 768 | 787,968 |
| 4× adaLN-Zero | 每个 top block 为 `Linear(768,6×768,bias=true)`；按 11.4 的顺序作用于 latent query | 14,174,208 |
| flow output | `Linear(768,16,bias=true)` | 12,304 |
| **generation subtotal** |  | **14,987,536** |

为了让两项任务共享一组可训练参数，top-4 MiniMind block 的输出各接一个两层 residual adapter：`Linear(768,64,bias=true) → SiLU → Linear(64,768,bias=true)`，四层合计 `4 × 99,136 = 396,544` 参数。UNDERSTAND 与 GENERATE 路由共用该 adapter；纯文本和 Talker 路由显式 bypass。测试用相同文本/语音输入比较 adapter 加载前后输出，确认 bypass 路由未改变。每个 adapter 的第二个 Linear 使用零初始化，使接入时 residual 增量为 0；不能用随机初始化破坏冻结 backbone 的起点。

所以每个 arm 的规范 trainable 总数必须是：

$$
P_{\text{trainable}}
=
21{,}337{,}344
+14{,}987{,}536
+396{,}544
=
36{,}721{,}424
$$

启动器逐模块执行 `sum(p.numel() for p in module.parameters() if p.requires_grad)`，写出 `param_ledger.json`。任一 arm 不等于 `36,721,424` 时中止。SigLIP2、VAE、MiniMind base weights、LM head 和 Talker 不计入 trainable count；报告另列 total、resident 和 actually-executed 参数。

主矩阵采用 **iso-data / iso-update**，不声称 iso-compute。三臂读取相同的 record 顺序、分辨率、paired update 数和 optimizer step；B 在 understanding batch 多执行一个冻结 encoder，因此 FLOPs 和 GPU-seconds 更高。报告同时给质量、GPU-seconds、峰值 HBM 和 images/s。不能通过多训 A/C 或少训 B 调整算力后仍称其为单变量比较。

## 12. 逐步实验

### Step 0：复现两个固定 specialist reference

Retention 的分母必须可复现——"统一模型保住了专用模型九成能力"这句话里，那个"专用模型"是什么、怎么训的，得先固定。因此固定两个 specialist reference：

| Reference | 唯一架构 | 冻结项 | 可训练项 | 数据与更新预算 |
|---|---|---|---|---|
| U-ref | frozen SigLIP2 → 本课同款 3-block adapter（C routing）→ frozen MiniMind + shared adapters | SigLIP2、MiniMind base weights、LM head、VAE、Talker | understanding stack + shared adapters，精确 `21,733,888` 参数 | unified 每臂实际看到的同一 understanding 顺序，12,000 updates |
| G-ref | frozen VAE → generation stack → frozen MiniMind + shared adapters | VAE、MiniMind base weights、LM head、SigLIP2、Talker | generation stack + shared adapters，精确 `15,384,080` 参数 | unified 每臂实际看到的同一 generation 顺序，12,000 updates |

G-ref 只使用本课定义的 rectified-flow velocity objective。两者从相同锁定 checkpoint 开始，使用与 unified 相同的 optimizer、LR、global batch、分辨率、seed 和预处理。固定取第 12,000 个 update 的 checkpoint 作为 denominator，不按 test 选择 best。运行验收 3 前，先检查 `specialists/{u_ref,g_ref}.yaml` 和逐模块的 `param_ledger.json`。

它们是本缩小实验的 per-task reference，不代表全领域绝对上界。Retention 先在每个 seed 内与同 seed reference 计算，再汇总均值与置信区间。

### Step 1：审计 VAE reconstruction upper bound

该步骤测量 frozen VAE 对生成质量施加的上限。对 held-out 图像：

- encode/decode；
- 计算 LPIPS/PSNR/SSIM（三种重建质量指标：LPIPS 用神经网络特征算感知差异，越低越好；PSNR/SSIM 是像素级信噪比与结构相似度，越高越好）；
- 人眼检查文字、人脸、细纹理；
- 记录 latent shape 与压缩率。

保存最差 case 的原图和 reconstruction。VAE 已经丢失的细节不能归因于 shared core 或 flow head。

### Step 2：完成 toy flow matching

接入 LLM 前。先在不接 LLM 的条件下验证 flow 定义：

- 在小图或 VAE latent 上训练；
- loss 能下降；
- 采样能恢复数据分布；
- 验证 timestep 与 target 定义。

正式实现包含训练与采样两条路径。该 VAE 配置没有可靠的 `scaling_factor` 字段，因此用 10k 个 train image 的 posterior mode（VAE 编码分布的众数，即不加噪的确定性编码）计算 `latent_scale = 1 / std(z)`，写入 manifest 后冻结；训练和采样读取同一个值。

```python
def generation_training_step(
    batch, core, vae, latent_scale, timestep_mlp, generate_task_embedding
):
    # x0 是 data endpoint，noise 是 t=1 endpoint。
    with torch.no_grad():
        x0 = vae.encode(batch.image).latent_dist.mode() * latent_scale

    noise = torch.randn_like(x0)
    t = torch.rand(x0.shape[0], device=x0.device)
    t4 = t[:, None, None, None]
    xt = (1.0 - t4) * x0 + t4 * noise
    target_velocity = noise - x0

    prompt_ids = drop_condition_for_cfg(batch.prompt_ids, p=0.10)
    latent_tokens = patchify_2x2(xt)             # [B, 256, 16]
    latent_tokens = latent_in(latent_tokens)     # [B, 256, 768]
    time_cond = timestep_mlp(t) + generate_task_embedding[None, :]

    h = core.forward_generation(
        prompt_ids=prompt_ids,
        latent_tokens=latent_tokens,
        time_cond=time_cond,                     # [B, 768]，只供 adaLN-Zero 使用
        attention_mask=build_generation_block_mask(),
    )
    velocity = unpatchify_2x2(flow_out(h.latent))  # [B, 4, 32, 32]
    return ((velocity - target_velocity) ** 2).mean()
```

```python
@torch.no_grad()
def sample(prompt_ids, seed, steps, cfg, core, vae, latent_scale):
    g = torch.Generator(device=prompt_ids.device).manual_seed(seed)
    z = torch.randn((1, 4, 32, 32), generator=g, device=prompt_ids.device)
    grid = torch.linspace(1.0, 0.0, steps + 1, device=z.device)

    for t_now, t_next in zip(grid[:-1], grid[1:]):
        # predict_velocity 内部用 t 构造 time_cond，并显式传给 forward_generation；
        # t 不与 latent token embedding 相加。conditional/null 共用同一个 t。
        v_cond = predict_velocity(z=z, t=t_now, prompt_ids=prompt_ids, core=core)
        v_null = predict_velocity(z=z, t=t_now, prompt_ids=null_prompt_ids(), core=core)
        velocity = v_null + cfg * (v_cond - v_null)
        z = z + (t_next - t_now) * velocity  # 从 noise t=1 反向积分到 data t=0

    image = vae.decode(z / latent_scale).sample
    return image.clamp(-1, 1)
```

第一版固定 Euler 和 32 steps。Heun 或自适应 solver 作为扩展实验，不能在三臂之间使用不同 solver。

### Step 3：实现 single-path unified baseline

Arm A 固定为 **VAE-only single path**：

- 理解时只将 VAE latent resample 成 32 个 prefix tokens；
- 生成时预测同一 VAE latent 的 velocity；
- 不接 SigLIP2；
- 接同一个 language head 与 flow head。

主矩阵不在运行时切换 semantic-only 或 VQ-only 定义；VQ/discrete 路线留到课后扩展。

### Step 4：实现 dual-path

Arm B 接入两种冻结视觉表示，并保持输出 token 数与 Arm A/C 相同：

- semantic encoder 与 VAE 均冻结；
- 按 11.6 的 B routing 训练 understanding 侧的 3-block adapter，以及三臂共同的
  generation stack；生成 token 不经过 understanding adapter；
- MiniMind core、LM head 和 Talker 在 canonical Pilot 的两个 stage 始终冻结；
- 确认理解与生成 forward 可以分别执行。

### Step 5：Stage 1 warmup

Stage 1 分别确认两条任务路径可以优化——先各自单独能训，再谈一起训；跳过这步直接 joint，出问题时你不知道该怪哪条路。三臂都先执行 2,000 个 understanding update，再执行 2,000 个 generation update：

| warmup | 可训练模块 | loss 与固定项 |
|---|---|---|
| understanding | 3-block adapter + shared top-4 residual adapters | assistant-only、按有效 token 归一化的 CE |
| generation | `latent_in` + timestep MLP + 4×adaLN-Zero + `flow_out` + 同一组 shared top-4 residual adapters | 按 latent element 归一化的 velocity MSE；condition dropout 固定为 0.10 |

冻结项不因 stage 改变。A/B/C 的 generation warmup 都使用同一份 null-condition mask。Stage 1 每个 task 的 sampler 各消费前 2,000 个固定 global batch；不得通过只给某一 arm 更长 warmup 改变结果。

### Step 6：Stage 2 joint training

Stage 2 训练 10,000 个 paired update unit。每个 unit 包含一次 understanding global batch 和一次 generation global batch：

$$
\begin{aligned}
\mathcal L_{\text{text}}
&=
\frac{1}{N_{\text{answer}}}
\sum_{i=1}^{N_{\text{answer}}}\operatorname{CE}_i, \\
\mathcal L_{\text{flow}}
&=
\frac{1}{N_{\text{latent}}}
\sum_{j=1}^{N_{\text{latent}}}
\left(v_{\theta,j}-u_j\right)^2, \\
\mathcal L_{\text{paired}}
&=
0.5\,\mathcal L_{\text{text}}
+0.5\,\mathcal L_{\text{flow}}.
\end{aligned}
$$

一个 paired unit 先累积这两次前向的梯度，再执行一次 `optimizer.step()`。

canonical lane 不使用未定义的 `mixed batch`，也不加入 reconstruction auxiliary loss。三臂 generation batch 继续使用 `condition_dropout=0.10`，并读取按 sample id 固定的同一份 dropout mask。每个 task 暴露 `2,000 + 10,000 = 12,000` 个 global batch，与对应 specialist 一致。日志记录有效 answer token、latent scalar、image pixel、FLOPs 和 GPU-seconds，不只记录 sample 数。

### Step 7：比较 shared 与 decoupled encoding

主矩阵固定为：

1. VAE-only single path；
2. SigLIP2 + VAE dual-path fused；
3. Janus-style decoupled：理解只用 SigLIP2，生成只用 VAE，但共享 MiniMind core。

两个 specialist 只作为固定 per-task reference，不纳入三臂架构检验。

### Step 8：加入 classifier-free guidance

CFG 通过比较 conditional 与 null-condition velocity 增强 prompt 条件。训练时的 condition dropout 已经从 Stage 1 起固定为 0.10，Step 8 不再训练，也不修改 dropout。主矩阵的 A/B/C 全部使用 `cfg_scale=3.0`。主矩阵封存后，只在 Arm B 的 dev prompts 上扫描 `cfg_scale ∈ {1.0, 2.0, 3.0, 4.0, 5.0}`，记录 prompt adherence、过饱和和模式坍缩。该扫描使用独立实验 id，不改主矩阵结论；发布固定 prompt 和全部 seed 的样例，不能只展示筛选后的图。

### Step 9：做 interference audit

`interference` 指一个任务的训练使另一个任务性能下降。每 N steps 同时运行：

- VQA/OCR；
- generation prompts；
- text-only regression；
- Talker regression（本课固定从完整 MiniMind-O 起步，因此必做）。

绘制每项能力随 update 的曲线，并与 U-ref/G-ref 同步对齐。只比较最终 checkpoint 会遗漏先升后降的干扰过程——中途被拽垮又部分恢复的能力，在终点对比里是隐形的。

### Step 10：可选 editing

只有 text-to-image 在固定 prompt 集上稳定后才加入 editing：

- source image latent；
- edit mask；
- instruction；
- target image。

## 13. 三个对照实验组

### 主矩阵

表中只改变 understanding encoding route。生成 latent、shared core、head 和训练顺序保持一致：

| Arm | 理解编码 | 生成编码 | 核心骨干 | Heads |
|---|---|---|---|---|
| A VAE-only single | VAE → 3 blocks → 32 prefix | prompt + 256 $z_t$；$t$ 经 adaLN | 同一 MiniMind core | text + flow |
| B dual-path fused | SigLIP2/VAE summary → 3 blocks → 32 prefix | prompt + 256 $z_t$；$t$ 经 adaLN | 同一 MiniMind core | text + flow |
| C Janus-style decoupled | SigLIP2 → 3 blocks → 32 prefix | prompt + 256 $z_t$；$t$ 经 adaLN | 同一 MiniMind core；projector 解耦 | text + flow |

三臂都只在训练时读取目标图像的 $x_0$，用它构造 $z_t$ 和 $u_t=\varepsilon-x_0$。没有任何一臂把干净 $x_0$ 作为生成条件；推理统一从噪声开始。

运行前检查以下公平条件：

- 同一初始 MiniMind-O revision；
- 同一 `sampler_order.arrow`、record/pixel 暴露和 paired update 数；
- 同一分辨率；
- 同一 generation sampler；
- 相同的训练期 `condition_dropout=0.10`、dropout mask 和主评测 `cfg_scale=3.0`；
- 三臂 core、flow head、最终 visual prefix、latent token 数和可训练参数都按 11.6 固定；
- 主检验是 `iso-data / iso-update`，不是 `iso-total-compute`；每臂实测 compute 单列；
- 两个 specialist 另报 `per-task reference`；两次 specialist 训练的总计算不能计入一个
  unified arm 的预算。

### 数据配比子实验

在 dev 上按预注册规则固定架构后，只比较数据/梯度平衡策略：

1. understanding:generation = 1:1 sample；
2. 按有效 token/pixel 平衡；
3. 动态 loss/gradient balance。

数据配比实验使用独立 experiment id，不能与架构主矩阵同时改变。

## 14. 参考配置

下面给出 Arm B 的 resolved config。Arm A/C 只改变 `arm.id` 和 `block_routes`：

```yaml
experiment: exp20_dual_path_joint

backbone:
  checkpoint: jingyaogong/minimind-3o
  revision: ee3febbd08cc5b2bd41c039c825a8934232fee33
  hidden_dim: 768
  core_frozen_all_stages: true
  lm_head_frozen_all_stages: true
  talker_frozen_all_stages: true

vision:
  semantic_encoder: jingyaogong/siglip2-base-p32-256-ve
  semantic_revision: 9465d1dc89db6bc6227c5b6b0e0ca9b940325d62
  semantic_frozen: true
  semantic_tokens: 64
  vae: stabilityai/sd-vae-ft-mse
  vae_revision: 31f26fdeee1355a5c34592e401dd41e45d25a493
  vae_frozen: true
  image_size: 256
  vae_latent_shape: [4, 32, 32]
  vae_patch: 2
  latent_tokens: 256
  latent_scale: calibrated_on_train_10k

arm:
  id: B_dual_path_fused
  block_routes: [siglip64_to_32, vae256_to_32, concat64_to_32]

understanding_adapter:
  type: cross_attention
  hidden_dim: 768
  heads: 12
  mlp_dim: 3072
  bias: true
  blocks: 3
  learned_queries_per_block: 32
  output_tokens: 32
  expected_trainable_params: 21337344

shared_core_adapter:
  layers: [4, 5, 6, 7]
  bottleneck_dim: 64
  activation: silu
  bias: true
  routes: [understand, generate]
  bypass_routes: [text, talker]
  expected_trainable_params: 396544

flow_head:
  shared_top_layers_with_adaln: 4
  hidden_dim: 768
  prediction: velocity
  path: linear_data_to_noise
  train_time: uniform_0_1
  sampler: euler_reverse
  sampling_steps: 32
  cfg_dropout: 0.1
  cfg_scale_main: 3.0
  expected_trainable_params: 14987536

loss:
  text_weight: 0.5
  flow_weight: 0.5
  text_normalization: valid_answer_tokens
  flow_normalization: valid_latent_scalars

mixture:
  understanding: 0.5
  generation: 0.5
  unit: paired_update
  compute_policy: iso_data_not_iso_compute

data:
  recipe: synthetic_shapes_v1
  recipe_version: 1.0.0
  seed: 20260723

training:
  stage1_understanding_updates: 2000
  stage1_generation_updates: 2000
  stage2_paired_updates: 10000
  global_batch_size: 256
  micro_batch_size_per_gpu: 4
  gradient_accumulation_per_task: 8
  optimizer: adamw
  learning_rate: 0.0001
  weight_decay: 0.01
  warmup_updates: 500
  bf16: true
  seed: 42
  world_size: 8
  backend: ddp
  expected_total_trainable_params: 36721424
```

`arm_a.yaml` 将 `block_routes` 设为 `[vae256_to_32, refine32, refine32]`；`arm_c.yaml` 设为 `[siglip64_to_32, refine32, refine32]`。除 `arm.id/block_routes` 外，三份 resolved config 必须 byte-for-byte 相同。8 卡下每个 task 的 global batch 为 `4 × 8 GPUs × 8 accumulation = 256`。Stage 2 先累计 8 个 understanding microbatch，再累计 8 个 generation microbatch；两个 loss 分别归一化后按 `0.5/0.5` 加权，执行一次 update。

正式 launcher 必须调用已经交付并通过 smoke test 的 joint trainer，同时解析 arm config、固定 sampler order 和 specialist config。当前伴随代码库只实现页面顶部的 CPU 机制实验，尚未交付这个八卡 trainer、resolved config 和 sampler manifest，所以这里不发布一条无法在仓库中解析的复制命令。

正式代码交付后，launcher 在启动前必须验证 artifact SHA、冻结 allowlist、参数总数、32-token understanding 输出、256-token generation 输入和 sampler hash；任一项不符即退出。U-ref 与 G-ref 也必须调用同一入口，只替换各自的 resolved specialist config。

## 15. 训练预算

### Toy

Toy 只验证 shape、loss 和过拟合，不用于架构结论：

- 1–2×24GB；
- 256 分辨率；
- 10k+10k 样本；
- 只验证代码和过拟合。

### Pilot

Pilot 是本课主结论的默认规模：

- 8×48–80GB；
- 100k scene，派生 400k 理解记录与 100k unique 生成记录；
- encoder、MiniMind core、LM head、Talker 全程冻结；
- 3 arms 各 `2k U warmup + 2k G warmup + 10k paired updates`；
- 适合课程结论。

### Standard

Standard 扩大数据和分辨率，需要新实验 id，并重新冻结 specialist reference：

- 8–32×80GB；
- 百万到千万图文对；
- 渐进 256→512；
- 部分解冻 LLM；
- 多 seed 成本较高。

### Paper-scale 边界

Show-o2/BAGEL 的正式结果依赖更大模型、数千万到万亿级多模态 token 和大规模集群。8 卡实验只验证缩小后的机制与干扰趋势，不能声称复现论文绝对性能。

## 16. 评测

### 16.1 理解

理解指标分别测通用问答、科学图像、OCR 和幻觉；text-only regression 检查共享 adapter 是否影响语言能力：

- VQAv2；
- ScienceQA-IMG；
- TextVQA/OCRBench 小切片；
- POPE/幻觉；
- text-only regression。

### 16.2 生成

生成评测同时覆盖 prompt 条件遵循、视觉质量和组合关系：

- GenEval；
- DPG-Bench；
- CLIPScore（用 CLIP 打图文匹配分，衡量图像是否符合文字描述）；
- 语义对象计数/颜色/空间关系；
- 足够样本时再用 FID（生成分布与真实分布在特征空间的距离，越低越好，小样本下不可靠）；
- 盲评 prompt adherence 与视觉质量。

### 16.3 Editing（选做）

Editing 只在 Step 10 执行后报告，并分别测指令遵循与源图保持：

- instruction adherence；
- source preservation；
- masked-region quality；
- identity/布局保持。

### 16.4 系统

系统指标解释模型质量对应的计算成本。每项按 arm 和 sampling step 数报告：

- 视觉 token；
- trainable/total parameters；
- train throughput；
- generation latency；
- sampling steps；
- peak HBM；
- 每张图 GPU-seconds。

### 16.5 能力保留率

先在查看测试集结果前登记理解指标集合 $\mathcal M_U$、生成指标集合 $\mathcal M_G$，并写明每项指标是越高越好还是越低越好。对任意指标 $m$，统一模型为 $U_m$，对应的同 seed 专用模型为 $S_m$。能力保留率按指标方向计算：

$$
R_m =
\begin{cases}
U_m/S_m, & \text{指标越高越好},\\
S_m/U_m, & \text{指标越低越好}.
\end{cases}
$$

例如 CLIPScore 使用 $U/S$，FID 使用 $S/U$，不能把原始 FID 直接代入 $U/S$。两项总保留率取各自预先登记指标中的最小值：

$$
\operatorname{UnderstandingRetention}=\min_{m\in\mathcal M_U}R_m,\qquad
\operatorname{GenerationRetention}=\min_{m\in\mathcal M_G}R_m.
$$

取最小值的用意：保留率是短板逻辑，九项指标保住、一项塌掉，不能靠平均把塌掉的那项藏起来。只有样本数和计算方法在评测前固定时，FID 才能放入 $\mathcal M_G$。所有指标还要同时报告原始分数、绝对差和置信区间，避免比值掩盖分母过小的问题。

## 17. 验收条件

### 验收 1：生成链路正确

验收 1 检查 VAE、流匹配训练和反向采样的定义是否一致：

- VAE reconstruction upper bound 已报告；
- toy flow 能生成非噪声样本；
- 训练/采样定义一致；
- 无 test leakage。

### 验收 2：三个实验组可公平比较

验收 2 检查三个实验组是否只改变了预先声明的编码路径：

- A/B/C 使用同一初始骨干、数据顺序、record/pixel 暴露、更新数和 `36,721,424` 可训练参数；
- 三臂配置可复现；
- 三臂 understanding core 输入都是 32 tokens，generation 输入都是 256 latent tokens；
- 三臂使用相同的 `condition_dropout=0.10`、dropout mask、32-step Euler sampler 和
  `cfg_scale=3.0`；
- 不强行宣称 compute 相等，实测 FLOPs、GPU-seconds、吞吐和 HBM 均已逐臂报告；
- 不靠不同 CFG/sampling steps 偏袒某一臂；
- 至少 3 seed 的 pilot；
- specialist 只作锁定的 per-task reference；另报其总计算，不写进 A/B/C 胜负检验。

### 验收 3：主结果可以按预注册规则判定

先判定实验有效性。只有 A/B/C 通过验收 1 和验收 2、完成三个 seed，并按第 3 节生成四个差值的 95% 配对置信区间，主结果才有效。

有效实验再按两个层次报告：

1. **主假设得到支持**：$\Delta U_{B-A}$、$\Delta U_{B-C}$ 的置信区间下界均大于 0，
   且 $\Delta G_{B-A}$、$\Delta G_{B-C}$ 的置信区间下界均大于 $-0.02$；
2. **绝对能力目标达到**：Arm B 的理解保留率至少 95%，生成保留率至少 90%，同一个
   核心 checkpoint 能直接切换两种任务，且没有调用外部生成模型。

Retention 的 denominator 固定为 Step 0 中同 seed、第 12,000 update 的 U-ref/G-ref。主假设或绝对能力目标未达到时，报告有效的负结果和对应任务，不把实验写成无效，也不能临时换模型、按 test 选择 checkpoint 或修改 0.02 非劣阈值。

### 验收 4：能解释任务之间的干扰

验收 4 要求根据训练曲线、表征和梯度记录定位任务干扰：

- 明确损失发生在哪些任务；
- 有 gradient/loss/representation 证据；
- 能指出 single-path 与 dual-path 的差异；
- 提供不少于 50 个无筛选生成样例。

## 18. 失败诊断

先按症状定位数据、VAE、flow、共享表示或系统问题，再修改单一变量。和第 1 课的纪律一样：从最便宜、最可观测的一层查起——生成全是噪声时先跑 toy distribution，别急着怀疑共享参数：

| 症状 | 可能原因 | 诊断 | 修复 |
|---|---|---|---|
| 生成全是噪声 | flow target/采样 ODE 不一致 | toy distribution | 先回到无条件 toy |
| 图像模糊 | VAE upper bound 或 latent 过小 | VAE recon | 换 VAE/提高分辨率 |
| prompt 不跟随 | condition 注入弱 | condition dropout/attention | 调 fusion/CFG |
| VQA 明显下降 | 低层特征污染语义 | semantic-only probe | gated fusion/路径解耦 |
| 生成好但文字错 | VAE/数据不擅长文本 | OCR 分组 | OCR 数据/更高分辨率 |
| joint loss 正常、能力崩 | loss scale 隐藏梯度冲突 | gradient cosine | task-balanced sampling |
| single-path 看似最好 | 参数或数据不公平 | 参数/compute ledger | iso-budget 重跑 |
| FID 好但 prompt 错 | 分布质量不等于条件遵循 | GenEval/人评 | 多指标 |
| 样例漂亮、总体差 | cherry-picking | 固定 prompt 列表 | 发布全量网格 |
| Talker 退化 | 联合训练改写共享骨干 | speech regression | 冻结/能力 replay |
| OOM | VAE latent + semantic token 太多 | activation profile | resample/checkpoint |
| 8 卡吞吐低 | 图像解码/数据 IO | profiler | latent cache/webdataset |

## 19. 逐个样例检查

### 理解 case

每个理解 case 保存 `case_id`、input image、question、specialist answer、A 的 VAE-only answer、B 的 dual-path answer、C 的 decoupled answer、evidence region 和 error type。三臂必须引用同一个输入文件 hash。

### 生成 case

每个生成 case 固定 prompt、seed、sampler、step 数和 CFG，并保存 specialist image、A 的 VAE-only image、B 的 dual-path image、C 的 decoupled image，以及 object/color/count/spatial 分项分数和 human notes。

所有模型使用相同 prompt、seed policy、采样步数和后处理。人工备注要指出具体对象、颜色、计数或空间错误，不能只写"更好看"。

## 20. 交付物

```text
artifacts/exp20/
├── artifacts.lock.json
├── configs/{arm_a,arm_b,arm_c}.yaml
├── specialists/{u_ref,g_ref}.yaml
├── manifests/
│   ├── {train,dev,test}.jsonl
│   ├── sampler_order.arrow
│   ├── sampler_order.sha256
│   └── latent_scale.json
├── ledgers/
│   ├── param_ledger_{a,b,c,u_ref,g_ref}.json
│   └── compute_{a,b,c,u_ref,g_ref}.jsonl
├── checkpoints/index.json
├── eval/{understanding,generation,text,speech,retention}.jsonl
├── flow_toy/
├── vae_reconstruction/
├── samples/all_fixed_prompts/
├── cases/failure_tree.md
├── plots/interference/
└── report.md
```

`report.md` 必须同时给三臂质量、参数、FLOPs、GPU-hours 和吞吐，并明确接受或拒绝主假设。

## 21. 复现清单

- [ ] 固定 VAE revision；
- [ ] 固定 semantic encoder；
- [ ] 固定 LLM checkpoint；
- [ ] 保存图像预处理；
- [ ] 记录 latent scaling；
- [ ] 记录 flow path 和 prediction target；
- [ ] 固定 sampler/steps/CFG；
- [ ] A/B/C 从 Stage 1 起使用同一 `condition_dropout=0.10` 与 dropout mask；
- [ ] 固定 prompt benchmark；
- [ ] 去重 train/test 图像；
- [ ] A/B/C 主矩阵按 iso-data/iso-update 与相同 trainable 参数比较；
- [ ] A/B/C 的实际 compute 单列，未伪称 iso-total-compute；
- [ ] specialist 使用固定 reference recipe，并单列两次训练总成本；
- [ ] 预注册 `U_primary`、`G_primary`、B−A/B−C 差值、0.02 生成非劣阈值和 95% CI；
- [ ] `param_ledger.json` 断言三臂各为 36,721,424；
- [ ] 发布未筛选样例；
- [ ] 理解/生成/text/speech 回归齐全；
- [ ] 数据 license/provenance 可审计。

## 22. 前沿对照与改造方向

统一理解与生成这条路，2024–2026 年的公开系统正好把第 5 节的三条路线各走了一遍。[Show-o](https://arxiv.org/abs/2408.12528) 是"一个 Transformer、两种目标"的代表：文本走自回归 CE，视觉离散 token 走 discrete diffusion，靠 task token 和不同 attention mask 在同一组参数里分流——本课 11.4 的双 mask 表就是这个思路的连续版。[Janus](https://arxiv.org/abs/2410.13848) 的出发点与本课主矩阵同构：它指出用单一视觉编码器同时伺候理解和生成会两头受气——理解要高层语义、生成要低层细节——于是把两条编码路径解耦、只共享 Transformer 主干，这正是 Arm C 的原型。[Show-o2](https://arxiv.org/abs/2506.15564) 走双路融合：在 3D causal VAE 空间上同时建语义与低层特征，language head 管文本、flow head 管视觉，两阶段 recipe 先接通生成再联合训练——Arm B 和本课的 Stage 1/Stage 2 就是它的缩小版。[BAGEL](https://arxiv.org/abs/2505.14683) 把规模拉满：decoder-only 统一模型加 Mixture-of-Transformers，在大规模 interleaved 多模态数据上训练，并报告随规模出现的组合能力。GPT-4o 在 2025 年公开的原生图像生成也属于"同一模型原生出图"路线，但没有公开技术报告可查其架构细节。工程侧，[vLLM-Omni](https://arxiv.org/abs/2602.02204) 把 understanding、language generation 与 flow sampling 拆成 stage graph 分别调度——本课的三段式路由（understanding adapter、shared core、flow head）天然对得上它的切法。

先分清钱能解决的和钱解决不了的。规模问题：骨干是 hidden 768 的冻结 MiniMind，前沿是几十亿参数全量训练；数据是 10 万合成 scene，前沿是数千万到万亿级真实图文与交错 token；分辨率 256 对高分辨率乃至视频（Show-o2 的 3D causal VAE 天生为视频准备，本课的 SD VAE 只管单图）；BAGEL 报告的组合能力是数据和参数规模的产物，8 卡上不用指望。这些砸资源就能缩小。机制问题：表示路由的选择（三臂矩阵）、两种目标在同一组参数上的共存（CE 加 velocity MSE）、干扰的测量与归因（Step 9）、公平比较的账本纪律（11.6）——这些和前沿是同一套，任何规模都绕不开。本课刻意留下的机制缺口有两个：一是 core 全程冻结、只训 36.7M 外挂，前沿是全量或大比例解冻；二是没有 interleaved 数据和 editing 主线，"边看边画边改"的链路只在扩展里。下面的改造清单直接碰这两个缺口。



1. **第四臂 Arm D：统一离散自回归（路线 A 落地）。** 把生成侧换成离散路线，亲手量一次路线 A 的代价。改动位置：先在 `synthetic_shapes_v1` 上训一个小 VQ tokenizer（16×16 网格、码本 1024，`tokenizers/` 下新增模块）；`heads/flow_head.py` 换成扩展视觉词表的 AR head；11.4 的 GENERATE mask 里 latent 双向可见改为 causal；`sampling/flow_sampler.py` 换成逐 token 采样；训练目标从 velocity MSE 换成视觉 id 的 CE。参数账本重算并写入独立 ledger，用独立实验 id。预算：VQ tokenizer 单卡数小时；D 臂按 Pilot 同规格 `2k+2k+10k` updates，8 卡成本与现有单臂同量级。预期：每张图从 32 步 Euler（含 CFG 共 64 次前向）变成 256 步逐 token 解码，generation latency 明显上升；G_primary 被 VQ 重建上限压住；理解侧与 Arm A 同量级。失败判定：VQ codebook utilization 掉到个位数百分比（码本坍缩），此时先修 tokenizer 再谈架构结论，D 臂数据作废。
2. **adaLN-Zero 换 in-context 时间 token（DiT 消融的缩小版）。** [DiT](https://arxiv.org/abs/2212.09748) 比较过多种条件注入方式，adaLN-Zero 效果最好。改造：去掉 4×adaLN-Zero，把 timestep embedding 作为 1 个额外序列 token 拼进 generation 序列，latent 对它可见；改动位置是 11.4 的 `generation_block` wrapper（删 modulation 分支）和 mask 表（加一行时间 token 的可见性）；参数账本随之变化，必须重算 ledger 并另开实验 id。预算：Pilot 单臂重跑，预算与 Arm B 相同。预期：同 update 数下 flow loss 收敛更慢、G_primary 更低，与 DiT 结论同方向。失败判定：两种注入方式无差异——先打印 adaLN 的 gate 范数确认门控真的在工作，再检查时间 token 是否被 mask 挡住；两处都正常还无差异，就如实报告"该结论在本规模未复现"。
3. **解冻 top-4 MiniMind block：验证"为什么本课要冻 core"。** 把 shared residual adapter 换成直接解冻 top-4 block，学习率比外挂低一个数量级，其余配置与 Arm B 相同；可训练参数大幅上升，ledger 重算，独立实验 id。预算：Pilot 单臂加三项回归（text-only、speech、理解/生成主指标），8 卡与单臂同量级。预期：U_primary/G_primary 上升，但 text-only regression 和 Talker speech regression 掉点——干扰从"外挂内部冲突"升级成"改写共享大脑"，这正是 BAGEL 这类全量训练系统需要海量 interleaved 数据来撑住所有旧能力的原因。失败判定：speech regression 不掉、质量也不涨，先检查解冻名单和学习率是否真的生效（打印逐模块 grad norm），再下结论。

四条论文结论能在本课缩小版设置里验方向：

| 论文结论 | 缩小版对应实验 | 预期 |
|---|---|---|
| Janus：理解/生成解耦编码，理解质量优于单一低层编码 | Arm C 对 Arm A：理解编码从 VAE 换成 SigLIP2，生成路径逐位相同 | 同方向可复现：U_primary(C) 高于 U_primary(A)，ΔG 接近 0；若 U 无差异，先怀疑 synthetic_shapes 的理解任务太容易，加难 relation/count 再测 |
| Show-o2：语义+低层双路径能兼顾两种任务 | 主矩阵 B−A/B−C 的预注册检验 | 方向可期但不保证：主假设本身就是它的缩小版；按第 3 节规则判定，负结果照实发表，不改 0.02 阈值 |
| DiT：adaLN-Zero 优于 in-context 条件注入 | 改造实验 2 | 同方向可复现：in-context 版收敛更慢、G_primary 更低 |
| Flow Matching：直线路径的速度回归可训练出可采样的生成模型 | Step 2 的 toy flow 与主矩阵任一臂 | 完全可复现：toy 分布能回收，256 分辨率出非噪声图 |

BAGEL 的组合能力结论不在此列：那是数据和模型规模的产物，8 卡只能验证"统一训练不塌"的机制层，验证不了规模效应，报告里别越界。

## 23. 论文精读

### 必读 1：[Show-o](https://arxiv.org/abs/2408.12528)

重点：

- AR 与 discrete diffusion 的统一；
- attention mask；
- multimodal mixed generation；
- 训练任务组合。

带着三个问题读，答案要能在自己的产物里指认：为什么视觉 token 不全部用 AR（计算原因，对照改造实验 1 里 Arm D 的 256 步解码账单）；task token、attention mask 和两种 loss 各作用在哪些位置（对照 11.4 的双 mask 表，逐行找 Show-o 的对应物）；统一训练相对 specialist 的收益和退化各是什么（对照 16.5 的 retention 定义，看它有没有报等价的量）。

### 必读 2：[Show-o2](https://arxiv.org/abs/2506.15564)

重点：

- 3D causal VAE space；
- semantic/low-level dual path；
- language head 与 flow head；
- two-stage recipe。

带着问题读：双路径在哪一层融合、representation conflict 怎么处理（对照 Arm B 的 fusion 只进理解路由这条边界）；Stage 1 冻结了什么、训练目标是什么（对照 Step 5 的 warmup 表逐格比）；哪些机制公开脚本可复现、哪些大规模数据组件未公开（这决定你的报告里哪些话能说）。

### 必读 3：[Janus](https://arxiv.org/abs/2410.13848)

重点：

- decoupled visual encoding；
- shared transformer；
- understanding/generation tokenizer 选择。

带着问题读：理解与生成各用什么视觉 encoder、为什么单一 encoder 两头受气（对照第 1 节的表示冲突和 Arm A 的预期短板）；解耦之后仍共享的 Transformer 参数、token space 和训练接口是什么（对照 Arm C：解耦的是编码，共享的是 core）。

### 必读 4：[BAGEL](https://arxiv.org/abs/2505.14683)

重点：

- decoder-only unified model；
- interleaved multimodal data；
- Mixture-of-Transformers；
- emerging capabilities。

带着问题读：论文报告的每项能力，证据来自架构、数据规模还是训练规模（逐条标注来源）；8 卡缩小实验能验证其中哪些机制结论、不能验证哪些绝对性能结论（写进报告的"不能外推"清单）。

### 基础论文

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)；
- [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748)；
- [Neural Discrete Representation Learning (VQ-VAE)](https://arxiv.org/abs/1711.00937)。

基础论文阅读后完成三项检查：

- 自己推导 `x_t`、velocity target 和 sampling ODE；
- 画出 VQ 与连续 VAE 两条生成路径；
- 解释为什么 FID 不能衡量 prompt adherence。

## 24. 课后扩展

### 低成本

- 加 image editing/inpainting；
- 比较 16/32/64 unified visual queries；
- 加 task-specific LoRA，测共享 trunk 是否仍有价值；
- 用 frozen Show-o2 teacher 蒸馏小模型。

### 研究级

- image/video 共用 3D causal VAE；
- flow head 与 speech Talker 同时存在；
- Mixture-of-Transformers 按任务路由；
- multimodal preference optimization 覆盖图像输出；
- any-to-any interleaved generation；
- world model/future frame prediction。

### 系统级

- latent cache；
- flow sampling 并行化；
- 量化 shared LLM；
- understanding 与 generation 动态 batching；
- 用 [vLLM-Omni](https://arxiv.org/abs/2602.02204) 的 stage graph 分别表示
  understanding、language generation 与 flow sampling，再检查跨 stage tensor、
  dtype、batch key 和 GPU placement。

## 25. 最终报告必须回答的问题

最终报告应包含：

- understanding route 读取的 encoder、token shape 和 attention mask；
- generation route 读取的 VAE latent、timestep condition 和 flow head；
- 两项任务共享与独立的参数清单；
- joint training 冲突出现的 task、update 区间和梯度/指标证据；
- dual-path 相对 single/decoupled path 的逐任务差异；
- VAE reconstruction upper bound 与生成错误的分离分析；
- 相对 U-ref/G-ref 的 retention、绝对差和置信区间；
- 8 卡缩小实验支持的机制结论及不能外推的 paper-scale 结论。

报告需要同时说明完成了什么，以及哪些结论仍受模型和数据规模限制。

前四课建立可复现基线，并拆解 connector、codec 和多码本 Talker。第 05 至 07 课加入流式 listener、turn policy 与全双工调度。第 08 至 10 课处理动态分辨率、原生视频和视觉 token 预算。第 11 至 14 课比较 MoE、Mamba 混合骨干、八卡并行和长上下文训练。第 15 至 17 课覆盖联合 SFT、偏好优化和 GRPO。最后三课复现 Nemotron 微调、连接 Thinker 与 Talker，并比较统一理解和生成的方案。所有阶段共用预注册实验、参数与算力账本、回归门禁和负结果报告规范。

**离 2026 年的前沿还差什么。** 差距分三层，看清各层才知道哪些值得追。规模层：26M 到 36.7M 量级的可训练参数对前沿的几十上百亿，几百小时语音和十万合成 scene 对万亿级 token——钱和卡的差距，方法不变。训练形态层：这门课的系统是模块化拼接，冻结的耳朵眼睛、外挂的 adapter、分阶段的训练；前沿在走大规模 interleaved 数据的端到端统一训练，BAGEL 报告的能力就长在那里——这一层光加卡不够，数据和训练组织方式都得换。资产层：前沿系统的高质量语料、RL 环境和安全训练不公开，公开社区只能用缩小版逼近。但机制层没有秘密：双工调度、视觉 token 压缩、混合架构、可验证奖励、表示路由——论文里能读到的每一个机制，你都造过它的缩小版，也量过它的账单。

后续研究可以选择一个已预注册的问题继续扩展，例如让 flow head 与 speech Talker 共用 core，或把第 17 课的可验证奖励用于图像生成。工程方向可以继续优化 19C 的会话层，包括量化、动态 batching 和 vLLM-Omni 式 stage graph 调度。跟踪新系统时，仍可按输入表示、双工调度、视觉 token、骨干、训练阶段和生成路径逐项核对。课程提供的是一套可重复使用的分析和实验方法，而不是对某个固定架构的最终答案。
