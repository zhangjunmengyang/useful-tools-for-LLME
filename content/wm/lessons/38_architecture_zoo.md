---
id: 38_architecture_zoo
title: "架构动物园"
summary: "AR、DiT、MoT、JEPA、latent action 各自把状态放在哪，桌宠该借哪一块？"
unit: frontier
play_tools: []
checkpoints:
  - "一份选型说明书，每个选择写清赌什么和 24GB 能不能跑。"
---

# 第 38 课：给 24GB 桌宠挑一副还在用的世界模型骨架

> 类型：研究（对照已有仓库的模块图做零件选型，不新训大模型）<br>
> 建议周期：2-3 天<br>
> 硬件：单张 24GB 卡只做读代码和可选的小仓库推理；Mac / 纯 CPU 可完成全部选型作业。MAGI-1 4.5B 官方写 24GB 够推理，本课不当必做。Cosmos 3 的 16B / 64B 标成只讲<br>
> 锚定仓库：[eloialonso/iris](https://github.com/eloialonso/iris) 的 `src/models/world_model.py`、[eloialonso/diamond](https://github.com/eloialonso/diamond) 的去噪器、[AlmondGod/tinyworlds](https://github.com/AlmondGod/tinyworlds) 的 latent action、[NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch) 的 RSSM、[nvidia/cosmos](https://github.com/nvidia/cosmos) 与 [NVIDIA/cosmos-framework](https://github.com/NVIDIA/cosmos-framework) 文档里的 MoT 图<br>
> 产物：一份桌宠零件选型说明书、一张八行骨架对照表、一次第 33 课是否题打档、一份架构拼板记录

## 1. 这一课做什么

第九幕读到这里，配方已经拆过了。第 37 课讲的是同一副骨架怎么训：teacher forcing 看真历史，self forcing 吃自己刚吐的帧，双向教师可以蒸馏成少步因果生成器。配方决定推理时会不会崩。骨架决定状态住在哪、动作从哪进、24GB 桌宠到底借得动哪一块。两课不要并成一句「换个大模型」。

贯穿主干没变：

```text
观察 先压成状态 再按动作预测下一状态 然后展开多条未来 给未来打分 最后选动作
```

本课只动中间两截的**实现形态**。不引进第四条路线。[第 16 课](16_three_roads_debate.md)已经用重建、表征预测、物体槽三根目标轴做过一次站队；那是「预测什么、拿什么当裁判」。今天的笼子更窄：在已经决定要生成或已经决定不生成的前提下，2026 年还在用的骨架把未来写成什么单位，注意力朝哪边看，动作是标签、是词，还是从两帧夹缝里挤出来的码。

八个笼子，每个核一篇代表论文再写机制：

1. AR token：IRIS、VideoPoet、MAGI-1（arXiv:2505.13211）把未来写成下一个 token 或下一个 chunk。
2. 双向去噪：Sora 类、Cosmos-Predict、DIAMOND 一次看整段（或整帧）再去噪。DIAMOND 的骨干是 U-Net，不是 DiT，它进这个笼子是因为「先看再凿」，不是因为名字里有 Transformer。
3. 因果 AR-扩散：CausVid、MAGI-1、Cosmos 3 Generator 的后训练形态，想兼得画质和流式。
4. MoT：Cosmos 3。Liang 等 Mixture-of-Transformers（arXiv:2411.04996）给不同模态分专家、共享注意力；Cosmos 3 把这条用成 Reasoner 与 Generator 两条通路。
5. JEPA：V-JEPA 2（arXiv:2506.09985）不生成像素，状态住在表征里。
6. latent action：Genie（arXiv:2402.15391）把动作藏进离散码。
7. RSSM：Dreamer 把状态拆成确定通道和随机通道。
8. 空间状态：VGGT / CUT3R 提供可查询的几何，不是 $P(s_{t+1}\mid s_t,a_t)$。

Transformer 不是一种世界模型。它是注意力层的排法。3D 高斯泼溅不是动力学。它是给定已有照片或视频之后的可渲染场，没有桌宠的 $a_t$。这两条写进第 5.1 节，后面不再回头辩。

做完你手里会有四样东西：一张「状态放哪 / 动作从哪进 / 24GB 借不借得动」的对照表；一次四个槽的架构拼板，缺哪一口、它更像谁；一份给桌宠写的零件选型（编码器、状态、动力学、动作编码、规划器），每块写清赌什么、失败模式；用[第 33 课](33_embodiment_degrees.md)七条是否题给这个选型打一档。毕业设计不换骨架也可以，但要说得出为什么不换。

术语速查：

| 术语 | 一句人话 |
|---|---|
| 骨架 | 状态住在哪、未来按什么单位往前走、动作从哪个口子进；不是骨干网络的商品名 |
| 预测单位 | 一次前向吐出的未来块：一个 token、一帧、一个 24 帧 chunk、一个潜状态 |
| 因果 / 双向 | 生成第 $t$ 步时能不能看见 $t$ 之后；看不见才能边看边播 |
| chunk 自回归 | 一次吐出固定长度的一段，再以这段为条件吐下一段；MAGI-1 的 24 帧就是这种单位 |
| DiT | 用 transformer 块做去噪或流匹配的视频骨干；它是网络，不是世界模型种类 |
| MoT | 同一套块里给不同通路（或不同模态）各留一套非嵌入参数，注意力仍可看整段 |
| 潜动作 | 没有人工按键时，模型自己从相邻帧变化里挤出的离散码 |
| 空间状态 | 点图、位姿这类「现在世界长什么样」；缺动作就还不是动力学 |
| 选型说明书 | 本课主交付：五件零件各选一块，写赌什么、24GB、桌宠哪条行为用得上 |
| 设计档 | 第 33 课的打分口径：计划达到的档，不是已经测到的档 |

## 2. 问题

四个具体问题，从读论文到写选型。

第一，2026 年刷榜系统共用一批词：DiT、自回归、世界模型、foundation。词帮不上忙。同一套 transformer 块，可以是 IRIS 那种因果续写，可以是 Sora 那种整段去噪，可以是 Cosmos 3 那种 MoT 双通路。本课用三个槽定位任何一篇新论文：状态的数据结构、预测单位、动作入口。填不满三个槽，就还只是视频生成器或几何重建器。

第二，八个笼子会重叠。MAGI-1 既是 chunk 自回归，又是因果扩散：每个 chunk 内部在做去噪，chunk 之间严格从左到右。Cosmos 3 的基础 Generator 在官方 README 里走双向去噪，想当交互世界模型还要后训练。重叠不是分类失败，是机制本身。对照表允许一行系统出现在两列，但必须写清「哪一段是哪一种」。

第三，24GB 桌宠该借哪一块。借错的典型症状有三种：把双向大视频模型塞进克制回路，两秒想象变成二十秒等待；把滑窗 AR 当成持久记忆，转头杯子被重新发明；把 VGGT 的点图叫做动力学，伸手之前问它「会不会倒」，它答的是「刚才那几张照片里杯子在哪」。本课把这三种错写成失败模式，写进选型说明书。

第四，打档。骨架选完不等于具身升档。第 33 课的尺子看是否题，不看参数量和官方产品名。一套很新的 MoT 若动作对换没做，仍可以是 E0。一套 2018 年的 RSSM 若已经接到安全层，可以是 E4。本课作业是给**你的选型**打一档，不是给论文打一档。

界限先划清。本课是研究档。动手是对照和选择，不是训练。Genie 本尊 11B、Sora、VideoPoet、Genie 3 无权重，只讲。Cosmos 3 Nano 16B / Super 64B 和 MAGI-1 24B 只讲。MAGI-1 4.5B 官方写「至少 24GB 的 GPU 就够」，列为可选体验，显存不够就停，写清失败原因。第 30 课已经训过的桌面小模型，本课复用它的接口，不重训。

## 3. 准备

- 概念：能口头复述主干循环，能做第 03 课那种动作对换。第 16 课的三条路线当作背景知识，本课不再重审「该不该重建」。第 37 课的配方表（目标函数、是否因果、训练时是否吃自己的输出）放在手边，本课只引用，不重做曝光偏差实验。
- 已经 clone 过、本课只读不训的仓库：第 09 课的 iris，第 10 课的 diamond，第 11 课的 tinyworlds，第 05 课的 dreamerv3-torch。缺哪一个，按对应课的 README 补 clone，不要按其训练命令开跑。
- 文档侧：打开 [nvidia/cosmos](https://github.com/nvidia/cosmos) 的 README「Model Architecture」一节，和 `cookbooks/cosmos3/cosmos3-model-architecture.png`。代码侧若要核对 MoT 类名，再 clone [NVIDIA/cosmos-framework](https://github.com/NVIDIA/cosmos-framework)，读 `cosmos_framework/model/generator/omni_mot_model.py` 与 `cosmos_framework/model/generator/mot/unified_mot.py`。不要按其训练入口提交作业。
- 空间状态对照：第 20 课 VGGT、第 21 课 CUT3R 的笔记。本课不重跑重建。
- 打档：第 33 课七条是否题，函数在 `web/lib/embodiment-score.ts` 的 `inferRung`。作业最小字段与第 33 课相同：`system`、`answers`、`envIsModelOnly`、`kind`、`evidence`。
- 硬件：读代码和填表不需要 GPU。可选的 MAGI-1 4.5B 推理按官方 `example/4.5B/run.sh`，机器不够就跳过。Cosmos 3 官方推理基准表从 H100 80GB、RTX PRO 6000 一类卡起跳，主线 24GB 不假装跑过。
- 阅读：MAGI-1（arXiv:2505.13211）第 2 节；Genie（arXiv:2402.15391）第 2 节；V-JEPA 2（arXiv:2506.09985）方法与 AC 规划节；Liang 等 MoT（arXiv:2411.04996）第 2 节；CausVid（arXiv:2412.07772）摘要与方法；Cosmos 3（arXiv:2606.02800）架构节。Sora、VideoPoet 只读公开技术报告或博客，标成只讲。

## 4. 学习目标

1. 拿到一篇 2026 年自称世界模型的论文，能在三分钟内填三个槽：状态的数据结构、一次前向的预测单位、动作从哪进；填不满就标「还不是动力学」。
2. 口头区分 AR token、双向去噪、因果 AR-扩散三家，并指出 MAGI-1 为什么同时出现在第一家和第三家。
3. 对照 Liang 的 MoT 定义和 Cosmos 3 文档里的双通路，说出「分专家」分的是模态还是 Reasoner / Generator，共享的是什么。
4. 解释为什么 VGGT / CUT3R 的点图不能当作 $P(s_{t+1}\mid s_t,a_t)$，以及 3D 高斯泼溅为什么进不了本课的动力学栏。
5. 给桌宠写出五件零件的选型，每件写「赌什么 / 24GB 能不能跑 / 哪条行为用得上 / 失败模式」。
6. 用第 33 课是否题给这份选型打一档，分清设计档和日志档。
7. 做一次架构拼板：四个槽缺一口时，能指出它更像 IRIS、Dreamer、Genie、Cosmos 3 还是 V-JEPA 2。

## 5. 原理

十个机制。每个仍走老节奏：为什么需要、怎么运转、精确定义、代码落点、怎么证明做对了。第 16 课的目标轴这里只当背景：重建路线内部今天再拆骨架，表征路线和空间状态各占一节，不重开三条路线的法庭。

### 5.1 三个槽，外加两个不许进栏的东西

先把「骨架」从商品名里抽出来。一篇论文写「我们用了 Transformer」或「我们用了 DiT」，等于告诉你厨房里有一口锅。桌宠要问的是：锅里煮的是下一帧像素、下一个离散词、下一个表征，还是下一份点图？动作是进锅里当调料，还是只写在菜单上？

三个槽：

1. **状态的数据结构。** 连续向量、离散 token 序列、表征 token、确定加随机双通道、点图加位姿、物体槽。状态是模型根据历史估出来的当前世界，不是摄像头当时吐出的像素。
2. **预测单位。** 一次前向吐出多长的未来。IRIS 一次一个画面词；DIAMOND 一次去噪整帧；MAGI-1 一次处理 24 帧的 chunk；RSSM 一次滚一步潜状态。单位越大，单步画质往往越好，流式和动作插入越难。
3. **动作入口。** 拼进 token 序列、调制归一化层、作为预测器的额外 token、从两帧变化里推断、根本没有。没有入口，或入口在但对换不分岔，第 12 课第一问就否决。

最小公式没有换过。记状态 $s$、动作 $a$，世界模型学

$$
P(s_{t+1} \mid s_t, a_t)
$$

骨架之争是 $s$ 的类型和 $P$ 的计算图。验证仍是[第 03 课](03_mdn_rnn_action_conditioned.md)的动作对换：同一 $s_t$，换 $a_t$，预测必须分岔。

两个不许进栏的东西，写在这里以免后面走神。第一，Transformer 是一层注意力加一层前馈的重复，Sora、IRIS、V-JEPA 2、VGGT 都在用它，职责完全不同。第二，3D 高斯泼溅（3DGS）是对已有照片拟合一份可渲染的点云场，相机拖到另一侧能出好看的新视图。桌宠的动作是转头、伸手、出声，不是拖虚拟相机。渲染器没有 $a_t$，也没有 $s_{t+1}$。第 20 课已经把重建和新视角合成从动力学里划出去，本课遵守那条线。

### 5.2 AR token：未来写成下一个词，或下一段 chunk

把观察切成离散词之后，预测下一刻变成续写。这是语言模型十年基础设施的一次搬家：交叉熵、温度采样、KV cache，原样能用。

IRIS（Micheli, Alonso, Fleuret, ICLR 2023）是课内可跑的最小标本。VQ tokenizer 把 64×64 的 Atari 帧压成 16 个词，词表 512。世界模型是因果 transformer：每步交互打成 17 个 token 的块（16 个画面词加 1 个动作词），窗口 20 块。观察头在每个画面位置预测下一个画面词；奖励头和终止头只在动作词位置发言。动作从序列里进，是一个和画面分表嵌入的离散词。状态就是窗口里那串 token 加 KV cache。预测单位是**一个 token**。

$$
p(z_{t,k} \mid z_{t,1:k-1},\, z_{<t},\, a_{\le t})
$$

$z_{t,k}$ 是第 $t$ 帧的第 $k$ 个画面词。生成一帧要采样 16 次，靠 KV cache 只算增量。记忆是硬窗口：第 21 帧起，第 1 帧从缓存里消失。这和第 05 课 RSSM「理论上无限、实际上漂」是两种忘法，第 39 课会专门拆。

代码落点：`src/models/world_model.py::WorldModel`，三个头的模式串在构造函数里，`src/models/slicer.py` 按模式切片，`src/models/kv_caching.py` 预分配缓存。配置在 `config/world_model/default.yaml`。你[第 09 课](09_iris_token_world_model.md)已经进梦里玩过 Breakout。

VideoPoet（Kondratyuk 等，arXiv:2312.14125）把同一张图纸放大到多模态。MAGVIT v2 把图像和视频切成离散码，SoundStream 把音频切成离散码，全部倒进一个 decoder-only 语言模型的统一词表。训练目标仍是下一个 token。它能做文本生成视频、图像生成视频、视频续写、视频配音，公开材料里没有桌宠能用的低层动作端口。无权重，只讲。它证明的是：AR token 这条骨架不依赖 Atari，也不依赖「必须叫世界模型」才能续写视频。

MAGI-1（Teng 等，arXiv:2505.13211）把预测单位从「一个词」改成「一段 chunk」。论文定义 chunk 为固定长度的连续帧，实现里每段 24 帧原始画面，约 1 秒（24 FPS）。VAE 先把视频压到潜空间（空间 8 倍、时间 4 倍），去噪在潜空间做。时间上因果：当前 chunk 可以看已经去噪过的过去 chunk，不能看未来。chunk 内部是整体去噪，所以它又滑进 5.4 节的因果扩散笼子。训练目标是 flow matching：对第 $i$ 个 chunk，在噪声与干净潜变量之间做线性插值

$$
x_i^{t} = (1-t)\, x_i^{0} + t\, x_i^{1}
$$

网络预测速度场。推理时不必等当前 chunk 完全干净，去到一定程度就可以开下一段，最多四段流水线并行。峰值显存与视频总长无关，这是它相对「一次看完整段视频」的工程卖点。动作入口在公开推理脚本里主要是文本、首帧或前缀视频，不是关节力矩。仓库 `example/4.5B/4.5B_base_config.json` 里 `chunk_width` 为 6，对应时间下采样 4 倍之后的潜帧数，$6\times 4=24$，和论文一致。`model_name` 字段写着 `videodit_ardf`。

代码落点：`inference/pipeline/pipeline.py::MagiPipeline`，按 `t2v` / `i2v` / `v2v` 三个模式跑；真正按段吐帧的是 `inference/pipeline/video_generate.py::generate_per_chunk`；骨干是 `inference/model/dit/dit_model.py::VideoDiTModel`。4.5B 入口：

```bash
bash example/4.5B/run.sh
```

官方 README 写：4.5B「任何至少 24GB 显存的机器都够」；更紧则用 distill+fp8 并把 `window_size` 调到 1，宣称 12GB 可跑。24B 要 H100/H800 × 8，只讲。本课不把 MAGI 当桌宠小脑：它能流式出画面，缺的是第 30 课那种低层动作对换。

验证。IRIS：同一历史换动作词，下一帧的 token 分布必须分岔，你第 09 课做过。MAGI-1：官方在 Physics-IQ 上报告 V2V 的 Phys. IQ 分数（24B 为 56.02，4.5B 为 42.44，表里还有 VideoPoet、Sora、Wan2.1 对照）。那是生成真加一点物理探针，不是规划好。桌宠借 AR token，借的是「KV cache 流式」和「softmax 天生多峰」，不是那张榜。

### 5.3 双向去噪：一次看整段，再把噪声凿回去

扩散或流匹配把生成写成：从噪声出发，沿着学到的速度场或 score 走到数据。若注意力在时间上是双向的，每一步去噪都看得到整段的过去和未来。画质高，因为每一帧都能拿未来当上下文来自洽。代价是不能边看边播：第 1 帧要等整段凿完才干净。交互世界模型要的是「我现在按左，马上看到左」，双向骨架天生别扭。第 37 课把这个别扭写成配方问题；本课把它写成骨架问题。

Sora 类（OpenAI 2024 年技术报告《Video generation models as world simulators》）把视频切成时空 patch，用 DiT 做整段去噪。公开材料以文本和首帧为条件。无权重，不能练。[第 12 课](12_frontier_landscape.md)三问的第一问，Sora 类通常过不去：没有逐步动作端口。把它写进世界模型动物园，是因为它定义了 2024 到 2026 年「生成真」的默认骨架：双向 DiT，一次看完整段。

Cosmos-Predict 走同一条骨架的开源分支。第 22 课读过 Predict2.5（arXiv:2511.00062）：flow matching，Text / Image / Video2World 收成一个模型。基础检查点的条件是文本加图像或视频。动作条件在后训练分支 `Cosmos-Predict2.5-2B/robot/action-cond`，入口 `examples/action_conditioned.py`。Hugging Face 卡片写 2B 的 Video2World（720p，16FPS）约需 32.54GB，主线 24GB 不够，本课继续只讲。仓库谱系已指向 Cosmos 3，Predict2.5 不再活跃开发；骨架没换，换的是产品名和 MoT 包装。

DIAMOND（Alonso 等，NeurIPS 2024，arXiv:2405.12399）证明「一次看再去噪」不必是 20 亿参数的 DiT，也不必一次看 8 秒。它在像素空间对**下一帧**做条件扩散，历史 4 帧和动作从通道与自适应组归一化进 U-Net。EDM 预条件让 3 步去噪就能稳，1 步不崩只是糊。它进这个笼子，是因为生成第 $t+1$ 帧时，去噪器在这一帧的空间里双向看，并且把「整帧」当作预测单位；不是因为骨干叫 Transformer。把 DIAMOND 写成「DiT 世界模型」是把锅的品牌当成菜名。

代码落点：`src/models/diffusion/denoiser.py::Denoiser` 算四个预条件系数；`src/models/diffusion/inner_model.py` 把 4 帧历史拼进通道、用动作嵌入调制残差块；`src/models/diffusion/diffusion_sampler.py` 用 Euler 按 `num_steps_denoising` 走完 $\sigma$ 序列。配置 `config/trainer.yaml` 默认 3 步。你[第 10 课](10_diamond_diffusion_world_model.md)已经玩过预训练世界，并把 IRIS 和 DIAMOND 并排过。

动作入口。DIAMOND 的动作是真手柄，对换成立，所以它是世界模型引擎。Sora 类和 Cosmos-Predict 基础检查点的动作经常缺席或被文本代替。同一骨架，动作口子不同，第 12 课第一问的答案就不同。

验证。DIAMOND：1 步对 3 步，糊不崩；动作对换必须分岔。Sora / Predict：本课不跑，只把「整段双向」和「有没有逐步动作」两句话写进对照表。桌宠若把双向大视频模型塞进克制回路，失败模式是延迟：人伸手的时间尺度是几百毫秒到两秒，等整段去噪完，杯子已经过沿。

### 5.4 因果 AR-扩散：想兼得画质和流式

双向教师画质好但不能流；逐 token 自回归能流但早期 token 视频往往糊、且长了会漂。2024 到 2026 年的修补是：把去噪留着，把注意力改成因果，再拿双向教师蒸馏少步学生。

CausVid（Yin 等，CVPR 2025，arXiv:2412.07772）把这件事写清楚。教师是多步双向视频扩散（实现里嫁在 Wan 上）；学生是 4 步因果生成器，用 KV cache 流式出帧。蒸馏用 distribution matching distillation（DMD）的视频版，外加两味药：用教师的 ODE 轨迹初始化学生，以及「双向教师监督因果学生」的不对称蒸馏，用来压自回归误差累积。论文报告初始延迟约 1.3 秒，之后约 9.4 FPS 流式，VBench-Long 总分 84.27。这些数字是论文结果，本课不复现。

代码落点：`causvid/models/wan/causal_model.py::CausalWanModel`（因果自注意力 `CausalWanSelfAttention`）；`causvid/models/wan/causal_inference.py` 是推理循环；`causvid/dmd.py` 和 `causvid/train_distillation.py` 是蒸馏；最小入口 `minimal_inference/autoregressive_inference.py`。教师 Wan 14B 的显存官方没按 24GB 主线公布，本课只读因果注意力怎么改，不装教师。

MAGI-1 在 5.2 节已经出现。它和 CausVid 的差别：CausVid 是「先有一个双向教师，再蒸成因果学生」；MAGI-1 从预训练起就按 chunk 因果去噪，chunk 内部双向、chunk 之间因果。两者预测单位都大于一帧，所以都能流水线。MAGI 的动作口子仍然偏文本和前缀视频。

Cosmos 3 的 Generator 要分两句话写，否则会和第 35 课、第 37 课打架。官方 README「Model Architecture」写：Generator 模式对图像、视频、声音、动作的噪声 token 做去噪，走**全注意力**；Reasoner 模式走因果自注意力做下一 token。也就是说，出厂的 Generator 通路更接近 5.3 的双向去噪。第 37 课的验收之一正是：解释为什么这台 Generator 还要后训练才能当交互世界模型。后训练的目标，就是 5.4 这种因果 AR-扩散：少步、可 KV cache、能插动作。仓库里 `Cosmos3VFMNetworkConfig` 有 `video_temporal_causal` 开关（`cosmos_framework/model/generator/mot/cosmos3_vfm_network.py`），`OmniMoTModel` 把它传进网络。本课把 Cosmos 3 Generator 同时记在 5.3 和 5.4：预训练骨架双向，交互形态靠配方改因果。不要写成「Cosmos 3 天生就是因果世界模型」。

验证。CausVid：公开材料看能不能流式、动态改 prompt；本课不测 VBench。桌宠借这条骨架，借的是「少步 + KV cache」这个配方，不是 14B 教师。24GB 上能落地的，仍是第 03 课那种「训练时混入自己的预测」，以及第 30 课小模型的自由 rollout 曲线。

### 5.5 MoT：分专家的是通路，共享的是注意力

Mixture-of-Transformers 这个名字在 2024 年底和 2026 年的 Cosmos 3 里各用了一次，分的对象不一样，共享的东西同族。先核原文，再看 Cosmos 怎么改。

Liang, Yu, Luo 等（FAIR / Stanford，arXiv:2411.04996）《Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models》。要解决的病是：早期融合的多模态大模型把文本、图像、语音都当成 token 丢进同一套稠密块，模态在特征空间里自己分堆，训练动态互相打架，算力却按最贵的那个模态付。MoT 的修法：按模态解开所有非嵌入参数，前馈、注意力投影、LayerNorm 各模态一套；自注意力仍然打在整段交错序列上。于是每个 token 走自己的专家，又能看见别的模态。论文在 Chameleon 设定（文本和图像都自回归）里报告 7B MoT 用约 55.8% 的 FLOPs 追上稠密基线；加上语音后，语音模态约 37.2% FLOPs；在 Transfusion 设定（文本自回归、图像扩散）里，同一套块可以给不同模态接不同损失。系统剖析在 AWS p4de.24xlarge / A100 上测过墙钟，图像质量约 47.2% 时间追上稠密基线。这些数字是原文实验，本课不复现。

注意它和 Mixture-of-Experts 的差别。MoE 用可学习路由器挑 MLP，负载不均、训练不稳是老问题。MoT 的路由是规则：看这个 token 属于哪个模态。没有「专家没人用就腐烂」的码本病，也没有双层优化。

Cosmos 3（Agarwal 等，arXiv:2606.02800）把 MoT 用成两条**通路**，不是三个模态各一套那么简单。官方 README 原话：统一的 Mixture-of-Transformers，一边是自回归 transformer 做推理（Reasoner），一边是扩散 transformer 做多模态生成（Generator）；Reasoner 对语言和视觉理解 token 做因果自注意力，Generator 对噪声 token 做全注意力去噪；两套共享同一套块结构、多模态注意力和统一的三维 mRoPE。输入输出组合是配置：只跑 Reasoner 就是 VLM；Generator 出视频就是生成器；再接上动作 token，才谈得上世界模拟器或世界-动作模型。第 35 课专门拆这些配置。本课只记骨架。

代码把「两套非嵌入参数、共享注意力」写得很死。`cosmos_framework/model/generator/omni_mot_model.py::OmniMoTModel` 的文档字符串写：MoT 模型，用 flow matching 目标训视觉 / 声音 / 动作生成。注释里写明两组权重：理解 / Reasoner 通路（`language_model` 骨干，例如 Qwen3-VL）和生成通路（扩散专家）；新鲜初始化时从预训练理解权重拷进生成通路。`cosmos_framework/model/generator/mot/unified_mot.py::MoTDecoderLayer` 的文档字符串写 dual-pathway attention。同一层里，理解侧是 `mlp` / `input_layernorm`，生成侧是带 `_moe_gen` 后缀的孪生模块；`PackedAttentionMoT` 仍在打包后的 und+gen 序列上做注意力。文件后半段注释写得更白：两条通路结构相同、权重不相交；Reasoner 塔没有 `_moe_gen` 后缀，生成塔有。官方图在 `nvidia/cosmos` 的 `cookbooks/cosmos3/cosmos3-model-architecture.png`。

动作从哪进。Generator 可以把动作当成要去噪的 token，也可以当条件。仓库 `action_gen` 与视觉生成绑在一起：`Cosmos3VFMNetworkConfig` 里若 `action_gen` 为真，必须 `vision_gen` 也为真，注释写「We do NOT support action only training」。桌宠若只想要一条「听关节、吐下一状态」的小脑，上 MoT 是把 VLM、视频生成器和动作头焊在同一套 4B 到 64B 的块里。官方型号：Edge 4B、Nano 16B、Super 64B。推理基准表从 H100 80GB、H200、RTX PRO 6000、Jetson AGX Thor 起跳，没有一张消费级 24GB 卡的官方数字。本课对 Cosmos 3 权重只讲。

验证。读代码确认两件事：理解 token 和生成 token 是不是走两套 LN / MLP；注意力是不是打在打包序列上。确认了，你就核过 Cosmos 3 文档里的 MoT 定义，也核过 Liang 原文「分专家、共享注意力」这句话在 Cosmos 里落成了什么。不要把 RoboArena 或 Artificial Analysis 的官方排名写成你复现的结果。

### 5.6 JEPA：状态住在表征里，像素可以丢掉

V-JEPA 2（Assran 等，arXiv:2506.09985）把第 13 课的配方做到视频和规划。状态不是能解码回画面的 token，是 ViT 对 tubelet（2 帧 × 16 × 16）编出来的表征。预训练目标与 I-JEPA 同构：context 编码器看未被遮的时空块，预测器猜被遮块的表征，target 来自 EMA 教师，L1 只算被遮位置。规模是新东西：论文口径超过 100 万小时视频。动作在预训练里不出现。

$$
\mathcal{L} = \frac{1}{|M|} \sum_{i \in M} \bigl\| P_\phi\bigl(E_\theta(x_{\setminus M})\bigr)_i - \bar{E}_{\bar\theta}(x)_i \bigr\|_1
$$

后训练才接上动作。V-JEPA 2-AC 冻住编码器，训一个 block-causal 的动作条件预测器。规划时在表征空间滚 $T$ 步，能量是预测终点与目标图像编码的 L1，CEM 搜动作。论文用不到 62 小时无标注机器人视频做后训练，在两间实验室的 Franka 上零样本抓放。那是论文结果，第 15 课未复现训练。

代码落点：编码器 `src/models/vision_transformer.py`，预训练预测器 `src/models/predictor.py`，AC 预测器 `src/models/ac_predictor.py::VisionTransformerPredictorAC`。加载入口在 `hubconf.py` 的 `vjepa2_ac_vit_giant`。你[第 15 课](15_vjepa2_in_practice.md)做过探针和能量景观。

动作入口：AC 预测器把 7 维动作当作序列里的 token，block-causal 掩码让预测看过去的表征和动作，不看未来。没有 AC 头的 V-JEPA 2 预训练权重，状态很强，仍不是世界模型。

桌宠借 JEPA，借的是「不养纹理」和「目标图就能规划」。失败模式：没有解码器，第 32 课没法给人看「我想象里杯子会倒」；62 小时机械臂数据和一张挤满杯子、手机、手的桌子也不是同一个世界域，副驾要在你自己的轨迹上重训。24GB 跑 ViT-L 探针第 15 课做过；ViT-g 加 AC 规划的显存以仓库配置为准，本课不新开训练。

### 5.7 latent action：动作可以是挤出来的码

Genie（Bruce 等，arXiv:2402.15391）面对的数据没有按键：互联网游戏视频只有画面。它拆成三件：时空视频 tokenizer、自回归动力学、latent action model（LAM）。LAM 的编码器偷看下一帧，把历史解释不了的变化压进很小的码本（论文用 8 个码）；解码器用历史加码重建下一帧。训完扔掉编码器和解码器，只留码本。玩家推理时直接选 0 到 7。11B 的 Genie 无权重，不能练。

信息论的账[第 11 课](11_genie_latent_actions.md)算过：码本只有几比特，装不下整帧，只能优先装「历史推不出来、对重建又最值钱」的成分，通常就是干预。预言：码本容量接近真实动作数时，码应当和按键对齐。预言靠分岔关和一致关验收，不靠训练损失。

tinyworlds 是课内可跑的最小实现。量化换成 FSQ（没有可学码本）。`models/latent_actions.py` 里 `LatentActionsEncoder` 把相邻帧特征拼接后出动作向量，`LatentActionsDecoder` 用 FiLM 把码注入，`LatentActionModel` 断言动作数必须是 2 的幂。默认 `n_actions` 在 `configs/training.yaml` 里是 4。动力学是 `models/dynamics.py` 的 MaskGIT，骨干 `models/st_transformer.py`。

桌宠什么时候借这一块：摄像头档你按键的同时自己转头，其实已经有标签，不必上 LAM。真没有同步关节日志、只有桌面录像时，LAM 才是动作入口的候选。失败模式：码学成噪声开关，或两个码效果无法区分。验收必须做第 11 课那套分岔加一致，不能只看重建 PSNR。

### 5.8 RSSM：状态拆成记牢和押注两条通道

PlaNet / Dreamer 的循环状态空间模型把压缩、记忆、预测焊进同一个损失。确定通道 $h_t$ 是 GRU 滚出来的历史，负责记牢；随机通道 $z_t$ 每步从分布里抽，负责押注。纯确定会把多峰未来平均成鬼影，纯随机又记不牢长历史。单步是

$$
h_t = f_{\mathrm{GRU}}\big(h_{t-1},\ [z_{t-1};\ a_{t-1}]\big), \qquad z_t \sim p(z_t \mid h_t)
$$

训练用后验 $q(z_t\mid h_t,o_t)$，想象和规划用先验 $p(z_t\mid h_t)$。KL 把两者拉近，先验就是潜空间里的动力学。DreamerV3 把 $z$ 换成 32×32 的 categorical。动作拼进 GRU 的输入，对换在 `img_step` 上做：同一 $h$，换 $a$，先验必须分岔。

代码落点：`networks.py::RSSM`。`img_step` 是先验一步（拼 $z$ 和动作、GRU 滚 `deter`、出 `stoch`），`obs_step` 是后验（内部先调 `img_step`，再看观察），`imagine_with_action` 闭眼推演，`kl_loss` 带 free bits 和两侧缩放。雇主是 `models.py::WorldModel`。仓库已归档，README 指向更新的 r2dreamer；本课精读这份定稿教材，不按其 `dreamer.py` 开训。你[第 05 课](05_rssm_planet.md)、[第 06 课](06_dreamerv3_imagination.md)已经拆过。

桌宠借 RSSM，借的是「一步一个潜状态、CEM 搜得动、24GB 训得动」。失败模式：隐状态会漂，转一圈回来杯子坐标对不上。第 21 课的点图和第 39 课的记忆库治的是另一种病，不要指望把 `deter` 加宽就能当空间记忆。

### 5.9 空间状态不是动力学

VGGT（Wang 等，CVPR 2025 Best Paper）一次前向吃多张 RGB，吐相机位姿和点图。状态是「这几张图里的桌子长什么样」，钉在第一张图的坐标系。它不吃动作，也不预测下一秒。代码：`vggt/models/vggt.py::VGGT`，头在 `vggt/heads/`。[第 20 课](20_spatial_3d_state.md)跑过公开权重。

CUT3R（Wang 等，CVPR 2025 Oral，arXiv:2501.12387）维护一份持续更新的 3D 状态，新帧进来就改：

$$
s_t = f(s_{t-1}, I_t)
$$

这是感知滤波，不是 $P(s_{t+1}\mid s_t,a_t)$。差一个动作，还差一步对未来的预测。代码：`src/dust3r/model.py::ARCroco3DStereo`，在线循环在 `src/dust3r/inference.py`。[第 21 课](21_persistent_4d.md)量过挡住 / 移走 / 转头。

两份几何对桌宠极有用：转头之后杯子的坐标还在，查询比重新发明便宜。把它们写进「编码器」或「可查询记忆」那一栏。写进「动力学」那一栏，就是用昨天的照片回答明天的推。3DGS / 4DGS 同理：能围着桌子转圈看，转圈的是虚拟相机。

### 5.10 一张表看完，再决定借哪一块

八行允许重叠。MAGI-1 和 Cosmos 3 Generator 各占两格，格里写的是机制，不是站队。

| 骨架 | 状态放哪 | 预测单位 | 动作从哪进 | 24GB | 桌宠借什么 |
|---|---|---|---|---|---|
| AR token（IRIS） | 窗口里的离散词 + KV cache | 一个画面词 | 序列里的动作词 | 预训练可玩，单游戏可训 | 流式和多峰采样；窗口会忘 |
| AR token（VideoPoet） | 多模态离散码 | 一个 token | 公开材料以文本为主 | 无权重，只讲 | 不借权重，借「词表可拼」 |
| chunk AR / 因果扩散（MAGI-1） | 潜空间视频 + 已去噪 chunk | 24 帧 chunk | 文本、首帧、前缀视频 | 4.5B 官方写 24GB 够推理 | 借流水线，不借当小脑 |
| 双向去噪（Sora / Predict） | 整段时空 token | 整段视频 | 文本 / 首帧；动作靠后训练 | Predict2.5-2B 约 32.54GB，只讲 | 不塞进克制回路 |
| 双向去噪（DIAMOND） | 像素帧本身 | 下一帧，3 步凿完 | 通道拼接 + 组归一化调制 | 官方训练约 12GB，可玩 | 借少步去噪和动作调制 |
| 因果 AR-扩散（CausVid） | 因果 DiT 的潜帧 + KV | 少步因果帧 | 实现跟教师，桌宠级动作未接 | 教师显存未按 24GB 公布，只讲 | 借蒸馏配方 |
| MoT（Cosmos 3） | 打包的理解 token + 噪声 token | 去噪整段或 AR 出词 | 动作可当生成 token | Edge/Nano 无 24GB 官方数，只讲 | 借双通路图，不借 16B |
| JEPA（V-JEPA 2） | 表征 token | 被遮块或下一步表征 | AC 预测器的动作 token | ViT-L 探针可做；AC 规划按第 15 课 | 借不重建、目标图规划 |
| latent action（Genie） | 视频 token + 动作码 | 下一帧 token | 码本里的离散码 | tinyworlds 可训可玩；11B 只讲 | 无标签录像才借 LAM |
| RSSM（Dreamer） | `deter` + `stoch` | 一步潜状态 | 拼进 GRU 输入 | 单任务 24GB 很宽 | 借双通道和想象规划 |
| 空间状态（VGGT/CUT3R） | 点图 + 位姿 / 持久 3D token | 不预测未来 | 无 | 8-20 张图或 512 档推理可做 | 借查询，不借动力学 |

桌宠默认选型写在第 7 节。原则一句话：24GB 上要实时回答「伸手会不会碰杯」，预测单位必须是一步或短窗，动作必须是你身体真能发的符号，状态必须能被第 29 课的杯子框查询。大视频骨架用来读论文和可选体验，不换掉第 30 课的小脑。

## 6. 源码导读

五个已经跑过或读过的仓库，外加 Cosmos 3 的文档和两个 MoT 文件。每个文件带着问题进去。路径写课前用 GitHub API、raw.githubusercontent.com 和 jsDelivr 包目录核对过；你本地若对不上，以仓库当前默认分支为准，把差异写进选型说明书，不要改课文去迁就一份过期 clone。

先确认类名还在。分别进入对应仓库根目录。

IRIS，世界模型本体：

```bash
grep -n "class WorldModel" src/models/world_model.py
```

预期：`src/models/world_model.py` 里有 `class WorldModel(nn.Module):`。接着打开同一个文件，找三个 `Head` 和观察头模式串里倒数第二位置零的那一行。动作词和画面词分表，在 `src/models/slicer.py` 的 `Embedder`。KV cache 尺寸和 `max_tokens` 的关系在 `src/models/kv_caching.py`。

DIAMOND，去噪器：

```bash
grep -n "class Denoiser" src/models/diffusion/denoiser.py
```

预期：`class Denoiser(nn.Module):`。四个预条件系数在这个文件；4 帧历史和动作调制在 `src/models/diffusion/inner_model.py`；Euler 一步在 `src/models/diffusion/diffusion_sampler.py`。读的时候盯住：这里没有 DiT 类名。骨架是「下一帧去噪」，骨干是 U-Net。

tinyworlds，潜动作：

```bash
grep -n "class LatentAction" models/latent_actions.py
```

预期三条：`LatentActionsEncoder`、`LatentActionsDecoder`、`LatentActionModel`。看 `keep_rate` 和 `action_dim = log2(n_actions)` 的断言。动力学对照 `models/dynamics.py`，骨干 `models/st_transformer.py`。

dreamerv3-torch，RSSM（仓库已归档，当教材读）：

```bash
grep -n "class RSSM" networks.py
```

预期：`networks.py` 第 13 行附近 `class RSSM(nn.Module):`。四个方法对 5.8 节：`img_step`、`obs_step`、`imagine_with_action`、`kl_loss`。`models.py::WorldModel` 是雇主，本课不跑 `dreamer.py`。

Cosmos 3 的 MoT，先读文档再读类。README 的「Model Architecture」和 `cookbooks/cosmos3/cosmos3-model-architecture.png` 是官方图。代码：

```bash
grep -n "class OmniMoTModel" cosmos_framework/model/generator/omni_mot_model.py
```

```bash
grep -n "class MoTDecoderLayer" cosmos_framework/model/generator/mot/unified_mot.py
```

预期：`OmniMoTModel` 的文档字符串写 flow matching 与 visual / sound / action；`MoTDecoderLayer` 写 dual-pathway attention。在 `unified_mot.py` 后半段搜索 `_moe_gen`，应看到「两条通路结构相同、权重不相交」那段注释。`cosmos3_vfm_network.py` 里搜索 `video_temporal_causal` 和 `action_gen`，核对 5.4 节那两句话。

对照读，不要求本课新 clone 的：

| 文件 | 零件 | 带着什么问题读 |
|---|---|---|
| iris `src/models/world_model.py::WorldModel` | AR token 动力学 | 动作词在序列的哪一格？观察头为什么不预测动作词？ |
| diamond `src/models/diffusion/denoiser.py::Denoiser` | 下一帧去噪 | $c_{\mathrm{skip}}$ 随 $\sigma$ 怎么变？这里有没有因果掩码？ |
| tinyworlds `models/latent_actions.py::LatentActionModel` | 动作入口 | 推理时编码器还在不在？码是标签还是推断出来的？ |
| dreamerv3-torch `networks.py::RSSM.img_step` | 潜状态一步 | 动作拼进哪一层？先验和后验谁在喂下一步？ |
| cosmos-framework `omni_mot_model.py::OmniMoTModel` | MoT 外壳 | 理解权重和生成权重哪一组先从 HF 加载？ |
| cosmos-framework `unified_mot.py::MoTDecoderLayer` | MoT 一层 | und 和 gen 是不是各有 LN 和 MLP？注意力看不看打包序列？ |
| vjepa2 `src/models/ac_predictor.py::VisionTransformerPredictorAC` | 表征动力学 | 7 维动作在哪一层进入序列？ |
| vggt `vggt/models/vggt.py::VGGT` | 空间编码器 | `forward` 的输入列表里有没有动作？ |
| CUT3R `src/dust3r/model.py::ARCroco3DStereo` | 在线 3D 状态 | 更新公式是 $s_t=f(s_{t-1},I_t)$ 还是 $P(s_{t+1}\mid s_t,a_t)$？ |
| MAGI-1 `inference/pipeline/video_generate.py::generate_per_chunk` | chunk 自回归 | `chunk_width` 和 24 帧原始画面怎么对上？ |

MAGI-1 若要对照配置，打开 `example/4.5B/4.5B_base_config.json`：`model_name` 为 `videodit_ardf`，`chunk_width` 为 6，`temporal_downsample_factor` 为 4，`window_size` 为 4，`num_frames` 为 96。CausVid 若只读一篇，读 `causvid/models/wan/causal_model.py::CausalWanModel`，看因果自注意力和 Wan 双向教师的差别。

读完应能在白纸上画出五张数据流，每张只标三个槽。画不出第三张槽的，回到第 5 节对应小节，不要用「都是 Transformer」糊过去。

## 7. 实验

不新训大模型。实验是对照、拼板、选型、打档。产物目录建议 `runs/L38/`，纯文本即可。第 30 课若已有桌面世界模型，接口笔记放在手边；没有也不挡，本课写的是设计档。

### Step 0: 架构拼板，四个槽缺哪一口

纸面或本课网页实验。四个槽：观察编码器、状态、动力学、动作编码。规划器先空着，第五件留到 Step 3。每一槽从下面选一块，或写「空」。

观察编码器候选：VQ tokenizer（IRIS）、像素本身（DIAMOND）、FSQ 视频 tokenizer（tinyworlds）、VAE 潜空间（MAGI / Sora 类）、冻结 DINOv2（第 30 课）、V-JEPA 2 ViT、VGGT 点图。

状态候选：token 窗口、像素帧、表征 token、`deter`+`stoch`、点图加位姿、空。

动力学候选：因果 transformer 续写、下一帧去噪、chunk 因果去噪、RSSM 先验、JEPA 预测器、CUT3R 状态更新、空。

动作编码候选：动作词、组归一化调制、latent 码、拼进 GRU、AC 动作 token、文本提示、无。

填完对照 5.10 的表，用下面规则判定「更像谁」。规则是机制，不是像不像宣传图。

| 你填的组合 | 更像谁 | 缺哪一口 |
|---|---|---|
| token 窗口 + 因果续写 + 动作词 | IRIS | 若编码器不是 VQ，写清差异 |
| 像素 + 下一帧去噪 + 动作调制 | DIAMOND | 若无动作调制，更像无动作视频扩散，E0 |
| 表征 + JEPA 预测器 + 无动作 | V-JEPA 2 预训练 | 缺动作，还不是世界模型 |
| 表征 + AC 预测器 + 动作 token | V-JEPA 2-AC | 缺解码器，不能播给人对看 |
| `deter`+`stoch` + 先验 + 动作拼进 GRU | Dreamer | 缺空间恒常 |
| 视频 tokenizer + MaskGIT + latent 码 | Genie / tinyworlds | 有标签时这一口多余 |
| 潜视频 + chunk 去噪 + 文本 | MAGI-1 | 缺低层动作 |
| 打包 token + MoT 双通路 + 动作可生成 | Cosmos 3 | 24GB 装不下默认权重 |
| 点图 + CUT3R 更新 + 无动作 | CUT3R | 缺动力学，不是世界模型 |
| 点图 + 3DGS 渲染 + 相机拖动 | 新视角合成 | 相机不是桌宠动作 |

故意留空做四次，各写一句「缺哪一口」：

1. 只填编码器和状态，动力学空、动作空。更像 VGGT 或普通视觉骨干。
2. 填编码器、状态、动力学，动作空。更像 Sora / VideoPoet / 无动作 V-JEPA 2。
3. 填四槽但动力学选 CUT3R 更新。指出公式里没有 $a_t$。
4. 填四槽且动作是文本。更像 Cosmos-Predict 基础检查点；写清它过不了逐步动作对换。

把四次记录存成 `runs/L38/board.md`。网页实验若已上线，以网页为准，纸面结果抄进去。

### Step 1: 核五处类名，把路径抄进笔记

在已经 clone 的仓库里跑第 6 节那五条 `grep`（IRIS、DIAMOND、tinyworlds、dreamerv3-torch、以及你若 clone 了的 cosmos-framework）。每条命令单独跑。把「文件::类」抄进 `runs/L38/paths.md`，后面选型只许引用这份笔记里出现过的路径。

Cosmos 3 若没 clone framework，允许只引用 `nvidia/cosmos` README 的 Model Architecture 段和 `cookbooks/cosmos3/cosmos3-model-architecture.png`，在笔记里写「文档核过、代码未 clone」。

可选体验，不是必做。机器有 24GB 且已按 MAGI-1 README 装好环境时，在仓库根目录跑：

```bash
bash example/4.5B/run.sh
```

脚本默认 `--mode t2v`、`--prompt "Good Boy"`、输出 `example/assets/output_t2v.mp4`。它会打出 `max memory allocated`。把这行数字抄下来。OOM 或装环境失败，写原因后停，不算本课不及格。不要改去跑 24B。

### Step 2: 填八行对照表

复制 5.10 的表，删掉你没读过论文摘要的行之前，先保证这八行都在：IRIS、DIAMOND、MAGI-1、Cosmos 3、V-JEPA 2、Genie / tinyworlds、RSSM、VGGT/CUT3R。每行三句话：状态放哪、动作从哪进、本课档位（复现 / 体验 / 只讲）。数字只许抄课文已经给出的（Predict2.5 的 32.54GB、MAGI 4.5B 的 24GB 声明、DIAMOND 训练约 12GB、IRIS 预训练可玩）。不许编 VBench、RoboArena、star 数。

存 `runs/L38/zoo.md`。

### Step 3: 给桌宠写五件零件

默认推荐如下。你可以改，但改每一件都要在失败模式里加一行。推荐不是新发明，零件全部来自你已经做过的课。

1. **观察编码器。** 冻结 DINOv2 ViT-S/14，接口对[第 30 课](30_desk_world_model.md)。需要「转头杯子还在」时，VGGT 或 CUT3R 当查询器，输出点图或世界系坐标，**不**进预测损失。赌的是：桌面纹理不必从零学；几何一致性由数据结构保证。
2. **状态。** DINOv2 的 patch 特征，拼上第 29 课的杯子框、人脸朝向、头朝向。不要用 IRIS 那种 20 帧硬窗口当唯一记忆，也不要把 `deter` 幻想成点图。
3. **动力学。** 第 30 课的特征空间动作条件预测器（DINO-WM 式 `predict` / `rollout`）。24GB 训得动、一步一个状态、对换测得了。不要换 MAGI / Sora / Cosmos 3 Generator 当小脑。
4. **动作编码。** 四个离散符号（看左、看右、伸手、不动）或真机关节命令，拼进预测器。有标签就不要上 Genie LAM。无标签桌面录像才允许把 tinyworlds 的 LAM 当加餐，必须做分岔关。
5. **规划器。** 短时程 CEM 或第 32 课那种安全过滤：先在模型里展开 1 到 2 秒，过桌沿则拒绝。不要用整段双向视频生成当「想一想」。

每件按四列写进 `runs/L38/parts.md`：

| 零件 | 选择 | 赌什么 | 24GB | 哪条行为 | 失败模式 |
|---|---|---|---|---|---|
| 编码器 | （你的选择） | | 训 / 推理 / 装不下 | 看、转头 | |
| 状态 | | | | 杯子恒常 | |
| 动力学 | | | | 克制前的想象 | |
| 动作 | | | | 动作对换 | |
| 规划器 | | | | 克制、停下 | |

推荐行的参考填法（可抄再改）：动力学赌「同一张桌子、四键必须分岔」，24GB 能训，行为是克制；失败模式是自由 rollout 超过 2 秒误差爆掉，第 32 课规划视野必须按第 30 课曲线截断。编码器若改成 VGGT 当动力学，失败模式直接触发 5.9：它不吃 $a_t$。

### Step 4: 写清 24GB 预算，点名三种借错

在 `parts.md` 下面追加「预算」一节，只写本课真实能发生的花费：

| 动作 | 卡时 | 备注 |
|---|---|---|
| 读五个仓库 + 填表 | 0 GPU | 半天到一天 |
| 第 33 课打档 | 0 GPU | 一小时 |
| 可选 MAGI 4.5B 推理 | 一次前向，官方宣称 24GB | 失败就停 |
| 重训第 30 课小模型 | 不在本课 | 需要回去第 30 课 |

三种借错，必须各写一句你自己的话，不许只抄：

1. 双向大视频模型进克制回路：延迟。
2. 滑窗 AR 当空间记忆：转头丢杯子。第 12 课 open-oasis 量过崩坏，本课不重测，引用那份帧号即可。
3. 点图或 3DGS 当动力学：没有 $a_t$。

### Step 5: 用第 33 课是否题打一档

对 Step 3 的选型打设计档。七条只能答是 / 否 / 部分。部分必须写清哪一部分。`kind` 填 `design`。若第 32 课已经有日志，另打一份 `kind: log`，两份都留。

推荐选型的参考答案（按摄像头档、预测已进克制来写；你的实现不同就改，不要为了好看抄满）：

1. 动作对换是否分岔？是（第 30 课必须过；没过就停在否，档掉到 E0 或 E3）。
2. 预测是否用于选动作或过滤动作？是（规划器查询动力学）；若你把规划器写成「只模仿」，改否。
3. 传感器与执行器是否共享同一物理世界？部分（摄像头真桌子，屏幕脸假身体）或是（真机）。
4. 失败之后是否可以无限重置？否（桌子不会 `reset()`）。
5. 接触或碰撞是否进入代价或安全层？部分（想象里截断）或是（真机接触）。
6. 是否存在控制不了、却必须预测的他者？是（对面的人，第 31 课）。
7. 不确定时停下是否算成功？是（第 32 课失败承认）。

`envIsModelOnly`：动作若只在模型里 rollout、从不驱动相机或关节，填 true，档会停在 E1。摄像头档或真机填 false。

用 `inferRung` 的口径自检：Q2 是且 Q3 部分，设计档是 E4。Q2 否则最多 E3。Q1 否且没有开环真机策略，E0。不要因为选型里写了 Cosmos 3 或 MAGI 就升档。

存 `runs/L38/rung.json`，字段与第 33 课作业相同。

### Step 6: 写选型说明书

`runs/L38/SPEC.md`，固定六节：

```text
一、三个槽：我的状态、预测单位、动作入口各是什么
二、五件零件表（Step 3 原样贴）
三、为什么不换骨架：对照 5.10，逐行写「不选它是因为」
四、24GB 与三种借错（Step 4）
五、第 33 课档：设计档或日志档，七条答案，依据路径
六、若以后要换一块：只许换一件，写预算和失败判据
```

第三节禁止出现「XX 骨架全面更好」。合格形态是条件句：在世界域为桌面、验证者为碰杯、硬件为单张 24GB 时，特征动力学比双向 DiT 合适，因为延迟和动作口子。第六节换一块的合法例子：把查询器从无换成 CUT3R；或把 LAM 加到无标签录像上。不合法：一次换成 Cosmos 3 16B 并声称桌宠升到 E5。

## 8. 配置与预算

| 环节 | 规模 | 24GB 卡 | Mac / CPU | 产物 |
|---|---|---|---|---|
| Step 0 拼板 | 纸面四次 | 不用 | 同左 | `board.md` |
| Step 1 核路径 | 五条 grep | 不用 | 同左 | `paths.md` |
| Step 1 可选 MAGI | 4.5B 一次 t2v | 官方宣称够；OOM 则停 | 不跑 | 显存日志或失败原因 |
| Step 2 对照表 | 八行 | 不用 | 同左 | `zoo.md` |
| Step 3-4 选型 | 五件零件 | 不用 | 同左 | `parts.md` |
| Step 5 打档 | 七条是否题 | 不用 | 同左 | `rung.json` |
| Step 6 说明书 | 六节 | 不用 | 同左 | `SPEC.md` |

全课 2-3 天，机器时间是零头。时间应花在「不选它是因为」那一节，不要花在装 Cosmos 3。dreamerv3-torch 已归档，读 `networks.py` 即可；训练命令以 r2dreamer 为准，本课不跟。iris / diamond / tinyworlds 的训练预算见第 09、10、11 课，本课不重开。

## 9. 验收

- [ ] 能不看笔记给一篇新论文填三个槽：状态数据结构、预测单位、动作入口；填不满标「还不是动力学」。
- [ ] 口头区分 AR token、双向去噪、因果 AR-扩散，并说出 MAGI-1 为什么同时落在第一家和第三家。
- [ ] 说出 Transformer 为什么不是一种世界模型，3DGS 为什么不是动力学。
- [ ] `paths.md` 里至少五条「文件::类」来自你本机 grep，或 Cosmos 一侧写明「只核文档」。
- [ ] `zoo.md` 八行齐全，档位没有把只讲写成复现，没有编造未给出的分数。
- [ ] `board.md` 有四次留空，每次写清更像谁、缺哪一口。
- [ ] `parts.md` 五件零件都有赌什么 / 24GB / 行为 / 失败模式。
- [ ] 三种借错各有一句自己的话。
- [ ] `rung.json` 七条是否题齐全，`kind` 标明 design 或 log，档与 `inferRung` 口径一致。
- [ ] `SPEC.md` 六节齐全，第三节全是条件句，第六节只换一件。
- [ ] 毕业设计若沿用第 30 课骨架，说明书里写得出为什么不换。

档位再划一次。本课主线是**研究档**。IRIS、DIAMOND、tinyworlds、RSSM、VGGT、CUT3R、V-JEPA 2 探针是你在前课已经做过的复现或体验，本课只引用。MAGI-1 4.5B 是可选体验。VideoPoet、Sora、Genie 11B、Genie 3、CausVid 训练、Cosmos 3 16B/64B、MAGI-1 24B 是只讲。

## 10. 排错

| 症状 | 最可能的原因 | 怎么确认 | 怎么修 |
|---|---|---|---|
| 把「用了 Transformer」写成一种骨架 | 把骨干商品名当成三个槽 | 问自己状态和预测单位是什么 | 回到 5.1，重填表 |
| 把 DIAMOND 写成 DiT | 课文笼子名叫「双向 DiT」 | 打开 `inner_model.py` 看有没有 U-Net / 残差块 | 写成「下一帧去噪，骨干 U-Net」 |
| 把 VGGT 填进动力学 | 点图看起来像「世界」 | 公式里有没有 $a_t$ | 挪到编码器或记忆栏 |
| 把 3DGS 当世界模型 | 新视角很真 | 动作是不是桌宠能发的符号 | 划掉，写进「不许进栏」 |
| MAGI 4.5B OOM | `window_size` 太大或还有别的进程 | `nvidia-smi`，看脚本打印的 allocated | 按 README 改 distill+fp8 且 `window_size` 为 1；仍不够就停 |
| cosmos-framework 的 `grep` 找不到类 | 没 clone 或目录不在仓库根 | `ls cosmos_framework/model/generator` | 只核 README 图，笔记写「代码未 clone」 |
| dreamerv3-torch 训练入口报过时 | 仓库已归档 | README 顶部 Notice | 本课只读 `networks.py`，不要跟训练 |
| 选型抄了 Cosmos 3 却打成 E4 | 把官方名称当成是否题 | Q1、Q2 你有没有日志 | 没有对换就改否，档降下来 |
| 拼板「四槽齐全」却更像 Sora | 动作填了文本 | 对换能不能做 | 标「文本条件生成器」 |
| 想一次换掉全部五件 | 把本课当成训练课 | 看第六节是不是只换一件 | 删掉，只留一个改造 |

## 11. 前沿与改造

前沿怎么做。2025 到 2026 年公开系统仍围着这八个笼子打补丁，很少再发明第九种状态。MAGI-1 把 AR 的流式和扩散的画质焊在 chunk 上，并开源 4.5B / 24B 推理。CausVid 把双向教师蒸成 4 步因果学生。Cosmos 3 用 MoT 把 VLM 和生成器焊进同一套块，输入输出靠配置切换；Edge 4B 瞄准端侧策略，Nano 16B 瞄准工作站，Super 64B 瞄准数据中心，官方推理基准没有消费级 24GB 这一档。V-JEPA 2-AC 继续走「不生成像素、目标图规划」。Genie 3 官方博客宣称数分钟一致性，无权重，只讲。WorldMem、Matrix-Game 3.0 把记忆库接进生成式骨架，那是第 39 课的题目：滑窗会忘、隐状态会漂、记忆库要按位姿取回。本课不提前把「加 context」写成已经解决物体恒常。

我们差在哪。规模占一半：16B MoT、24B chunk 扩散、100 万小时 JEPA，钱和数据能堆。机制占一半，而且本课能动手的全在机制：三个槽填对、动作口子真接上、预测单位匹配桌子的时间尺度、空间状态不要冒充动力学。第 30 课的小模型和 MAGI-1 4.5B 的差别，主要不是「会不会续写视频」，是「会不会在 200 毫秒量级回答伸手」。

动手改造清单（选做，都不新训大模型）：

1. 只换动作入口：在第 30 课预测器上把离散四键改成「文本描述动作」。预算：改胶水半天，不重训编码器。预期：对换分岔变弱或消失。失败：你没法量化分岔，因为没留特征空间距离。这是「文本条件」和「低层动作」的最小对照。
2. 只换查询器：VGGT 或 CUT3R 的杯子坐标作为额外状态通道，动力学仍用第 30 课预测器。预算：推理一次点图加改拼接，数小时。预期：转头后再查询，杯子 ID 还在；预测「会不会倒」仍靠动力学，不靠点图。失败：你用两点图的差冒充 $s_{t+1}$，等于没做这一题。
3. 只换预测单位：把第 30 课一步预测改成一次吐 5 步，再在自由 rollout 上比一步五次。预算：改头和小训，单卡数小时，数据仍用第 30 课那批。预期：短了可能稳，过了你的 2 秒曲线一样漂。失败：没有画两条曲线。
4. 纸面 MoT：按 `MoTDecoderLayer` 的两套 LN / MLP，画出桌宠若只有「理解人脸」和「生成下一帧」两个专家、共享注意力，动作 token 应该走哪一套。预算：一小时。预期：你能指出动作若只走 Reasoner，出的是字；若走 Generator，出的是要去噪的轨迹。不训练。

顺手复现。IRIS 论文「动力学可以写成下一个 token」，方向已在第 09 课兑现。DIAMOND「少步去噪可当世界引擎」，方向已在第 10 课兑现。Genie「窄瓶颈能挤出可控码」，方向已在第 11 课兑现。MAGI-1「chunk 因果可流式」，本课最多在 4.5B 上体验一次出片，不构成对 Physics-IQ 表的复现。Liang MoT「按模态分非嵌入参数、共享注意力」，在 Cosmos 3 代码里对应 und / gen 两套权重加 `PackedAttentionMoT`，读到那两处就算核对，不是复现 55.8% FLOPs。

## 12. 论文与延伸

1. MAGI-1: Autoregressive Video Generation at Scale（Teng 等，2025，[arXiv:2505.13211](https://arxiv.org/abs/2505.13211)）。带着问题读：chunk 为什么定为 24 帧？块间因果、块内去噪，和「逐 token AR」以及「整段双向扩散」各差在哪一口？推理峰值与视频长度无关，省下的是显存还是延迟？动作口子在论文里是什么？
2. Genie: Generative Interactive Environments（Bruce 等，2024，[arXiv:2402.15391](https://arxiv.org/abs/2402.15391)）。带着问题读：LAM 训练时偷看下一帧，推理时为什么可以扔掉编码器？8 个码的预算逼出的一定是按键吗，随机刷怪会不会抢走码本？$\Delta_t$ PSNR 量的是可控性还是画质？
3. Mixture-of-Transformers（Liang, Yu, Luo 等，2024，[arXiv:2411.04996](https://arxiv.org/abs/2411.04996)）。带着问题读：分的是哪些非嵌入参数？共享的注意力看见什么？它和 MoE 路由器差在哪？Transfusion 设定里，同一套 MoT 如何同时接自回归损失和扩散损失？
4. Cosmos 3: Omnimodal World Models for Physical AI（Agarwal 等，2026，[arXiv:2606.02800](https://arxiv.org/abs/2606.02800)），对照 [nvidia/cosmos README](https://github.com/nvidia/cosmos) 的 Model Architecture。带着问题读：Reasoner 和 Generator 的注意力掩码差在哪？输入输出组合里，哪一种才有 $P(s_{t+1}\mid s_t,a_t)$？官方排名哪些你不能写成自己的结果？
5. V-JEPA 2（Assran 等，2025，[arXiv:2506.09985](https://arxiv.org/abs/2506.09985)）。回读 AC 规划节。带着问题读：状态里没有像素，规划还算不算世界模型？能量和 RSSM 奖励头各要求什么标签？桌面多物体时，表征里的杯子和手会不会糊成一个残差？
6. From Slow Bidirectional to Fast Autoregressive Video Diffusion Models（Yin 等，CVPR 2025，[arXiv:2412.07772](https://arxiv.org/abs/2412.07772)）。带着问题读：学生为什么必须因果？DMD 蒸的是分布还是单条轨迹？不对称蒸馏防的是哪种误差累积？这和第 37 课 self forcing 是不是同一张病历？
7. VideoPoet（Kondratyuk 等，2023，[arXiv:2312.14125](https://arxiv.org/abs/2312.14125)）。带着问题读：统一词表之后，视频和音频的「下一个 token」还是不是世界的下一步？公开任务列表里有没有低层动作？只讲。
8. Sora 技术报告（OpenAI，2024，《Video generation models as world simulators》）。带着问题读：时空 patch 的状态是什么数据结构？双向去噪为什么几乎必然没有逐步动作端口？第 12 课三问它卡在第几问？只讲。
9. DIAMOND（Alonso 等，2024，[arXiv:2405.12399](https://arxiv.org/abs/2405.12399)）与 IRIS（Micheli 等，ICLR 2023）。对照读：预测单位差在哪？谁在用 U-Net，谁在用因果 transformer？同一作者、同一 Atari 100k，差别能不能归因到骨架而不是路线？
10. PlaNet（Hafner 等，2019，[arXiv:1811.04551](https://arxiv.org/abs/1811.04551)）与 DreamerV3（Hafner 等，[arXiv:2301.04104](https://arxiv.org/abs/2301.04104)）。带着问题读：`deter` 和 `stoch` 各治哪一病？先验作为动力学，和双向 DiT 的「一次看整段」在时间结构上差在哪？
11. VGGT 与 CUT3R（[arXiv:2501.12387](https://arxiv.org/abs/2501.12387)）。回读第 20、21 课。带着问题读：把点图接进第 30 课预测器，应该拼在状态里还是代替动力学？4DGS 若出现在 2026 年的新论文标题里，你用 5.1 的哪一条把它挡在动力学栏外？
12. 选读：Cosmos-Predict2.5（[arXiv:2511.00062](https://arxiv.org/abs/2511.00062)），看基础检查点和 `robot/action-cond` 差在动作入口。选读 DINO-WM（[arXiv:2411.04983](https://arxiv.org/abs/2411.04983)），对照你 Step 3 的默认动力学。

现在整个系统的零件表更新了一轮：配方在第 37 课，骨架在本课，记忆还没装。滑窗会切断 $K$ 帧前的杯子，RSSM 的 $h_t$ 会漂，记忆库要按位姿和时刻取回。第 39 课专做这件事。毕业标准仍在第 32 课和第 33 课：会看、会想、会克制，档靠是否题，不靠换了一副更新的骨架。
