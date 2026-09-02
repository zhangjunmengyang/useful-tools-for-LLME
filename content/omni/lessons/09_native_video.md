---
id: 09_native_video
title: "原生视频与 AV 对齐"
summary: "帧数和 LLM token 预算都一样时，模型是真用上了时间先后和声画对应，还是只在挨个认单帧画面？"
unit: vision
play_tools: []
checkpoints:
  - "分清三种视频前端：直接拼帧、后接时序 adapter、一开始就用 tubelet/Conv3D 看时空块。"
  - "给每一帧和每段音频 chunk 建一条共享的毫秒时间轴。"
  - "构造乱序、倒放、错配音轨这类不泄露答案的负对照。"
  - "画出 fps、token、质量、短暂事件召回和延迟之间的 Pareto 取舍图。"
---

# 第 09 课：建立原生视频时间轴

> 内容：视频时序融合、真实时间位置与音视频对齐
> 建议时长：5–9 天  
> 最小硬件：1×24GB 做 4–8 帧 pilot；4–8 卡做标准实验  
> 独立起点：由 preflight 锁定的官方 `jingyaogong/minimind-3o` dense checkpoint + 固定视觉 connector

## 1. 视频不是一叠图片

第三幕"长出眼睛"走到第二站。[第 08 课](08_dynamic_vision.md)把模型的眼睛从"只能看固定尺寸"升级成任意分辨率、多图身份、二维位置编码。但它看"视频"的方式仍是：抽几帧，当几张互不相干的图片塞进去。模型眼里没有先后，没有变化，更没有声音和画面的对应关系；它看的是一叠打乱了也没人发现的照片。

一句话直觉：视频就是带时间轴的图片序列。这句话的重点全在"带时间轴"四个字上，它正是不能把视频简单当多张图处理的原因，拆开是三条。

第一条，时序位置。多张独立图片之间没有先后承诺，视频帧有。"杯子是在门打开之前还是之后掉落"这类问题，答案完全藏在顺序里；把帧当无序图集送进去，顺序信息根本进不了模型。而且"顺序"还不能拿帧号（frame index，抽帧之后的序号）充数：同一个帧号，在 30fps 的视频里是一个时刻，在 15fps 的视频里是另一个时刻。必须保存真实毫秒。

第二条，帧间冗余。相邻两帧的绝大部分内容一模一样。逐帧独立编码，token 预算全花在重复内容上；而真正的信息；"什么变了"；模型得自己从两份几乎相同的特征里减出来，既难又浪费。时序融合模块的职责就是把"变化"显式算出来，让它直接成为特征。这份冗余同时也是下一课压缩 token 的本钱。

第三条，与音频的时间对齐。视频天生带声轨。"哪个声音对应哪个画面""说话的是左边还是右边那个人"，要求音频 token 和视频 token 落在同一条时间线上。图片从来没有这个问题；图片没有时间。如果音频按自己的 chunk 序号计时、视频按帧号计时，两套时钟对不上，声画绑定永远学不会。

所以本课干三件事：给每一帧保存真实毫秒时间戳；在冻结的图像 encoder 之后加轻量时序融合模块（三种方案公平对照）；把音视频 token 按统一的时间格子交错打包进 Thinker。另配一套"反事实"考卷：正放、倒放、打乱、换音轨，每个版本按新事实重新标注答案，逼模型证明它真的在用时间，防止靠背物体清单蒙分。

下一课要把视频 token 压到算力扛得住，而压缩的前提是知道时间信息存在哪、冗余长什么样；第 10 课的独立前端直接引用本课定义的抽帧协议和数据 contract，本课选出的最佳臂还能通过迁移实验进入第 10 课。这一步跳过，后面的"视频"永远是幻灯片，第四幕的长上下文、第六幕的毕业设计都没有真视频可用。

拿一段"门开了，然后杯子掉了"的视频：正放问"之前还是之后"，模型答"之后"；倒放，答"之前"；打乱帧序，答"无法判断"；换上错配的音轨，答案跟着新音轨走。你能在逐 case 报告里亲眼看到答案随事实翻转；打开 trace（结构化日志，忘了回[第 01 课](01_baseline_reproduction.md)），能看到每个 token 带着自己的时间 id 和模态标签。

本课术语：

| 术语 | 简要解释 |
|---|---|
| 时序融合（temporal fusion） | 把相邻帧的特征算到一起，让"帧与帧之间变了什么"直接成为特征 |
| frame-wise 基线 | 每帧各编各的、帧间互不通气的对照臂，用来证明时序模块真有用 |
| DW-Conv（depthwise 卷积） | 每个通道只卷自己、参数很省的卷积；后面用 1×1 线性层再混合通道 |
| Conv3D | 在时间、高、宽三个维度上一起做卷积，天生能看到相邻帧 |
| 时间格点（time slot） | 融合后特征在时间轴上的落脚点，本课三臂统一输出 8 个 |
| time bucket | 把连续毫秒切成的小格子（如 80ms 一格），音视频 token 都按格对号入座 |
| modality_id | 与 token 序列等长的整数标签，标明每个 token 是画面还是声音 |
| 反事实（counterfactual） | 倒放、打乱、换音轨后的测试样本，答案按变换后的新事实重新标注 |
| preflight / model lock | 正式训练前把代码、权重、encoder 全部下载校验、冻结指纹的检查 |
| oracle | 偷看了事件位置的作弊上界，只用来估天花板，不计入成绩 |
| PTS | 视频容器里记录的每帧显示时刻；时间基各不相同，用前要换算成毫秒 |
| 感受野（receptive field） | 一个输出位置能"看到"的输入范围；时间感受野就是能看到前后几帧 |

## 2. 静态帧基线缺少什么

本课要检验的能力只有一个：模型是否真的使用了时间顺序和声画对应关系。这件事必须靠专门的负对照来证明——只认静态帧的模型，在普通视频问答上也可能拿到不错的分数，因为很多问题靠单帧线索（出现了什么物体、什么场景）就能蒙对。

"均匀抽帧后拼接"仍然要做，它用于建立基线。原生视频建模在此之上加三样东西：显式的帧时间、跨帧计算和音视频对齐。

本课把视频建成一个带统一时间轴的模态对象：视频帧保留时间戳并经过时序模块；音频 chunk 保留自己的中心时间并经过 audio encoder。两路表示按真实时间交错打包后再送入 Thinker。

评测时分别构造反转、打乱和错配音轨的反事实，并且给每个反事实配上合法的新标签：反转视频必须重新标注正确顺序；打乱后顺序证据被破坏、不可回答的问题，以"无法判断"为标准答案；错配音轨必须提供配对后的新答案，或只报告原答案 log-prob（模型给该答案打的对数概率）的变化。常见错误是把变换前的答案直接套到已经改变事实的媒体上——那考的是模型的固执，考不出理解。

## 3. 实验要验证的结论

实验比较以下结论：

> 在相同输入帧、相同输出时间格点、相同空间 token 配额、相同 LLM token budget
> 和相同训练 token 下，轻量时序融合能提高顺序、状态变化和声画绑定任务，并能
> 对反转、打乱和错配音轨给出符合新输入事实的答案。

句子前半连着五个"相同"，它们是结论成立的前提。以下结果不能支持该结论：

- 原生时序臂不优于 matched frame-wise baseline（输入输出规格完全对齐的逐帧基线）；
- 反转后仍机械输出原顺序，没有跟随新事实改变答案；
- 打乱后仍给出确定顺序，没有在证据不足时拒答；
- 错配音轨后仍机械输出原答案，或错配样本没有合法的新标签；
- 提升只来自更多帧或更多 token；
- 静态图能力明显回退。

前四条防"假时序"，第五条防预算不公平，第六条防学了新的忘了旧的。

## 4. 进入条件与独立起点

开工前先清点家底。必须先有：

- 可复现的 MiniMind-O 图像基线；
- 一个冻结 vision encoder；
- 一个冻结或固定配置的 audio encoder；
- 能保存 `frame_time_ms` 与 `audio_span_ms` 的数据层；
- 至少 100 个带可验证时序答案的 held-out case（held-out：从不参与训练、只用来考试的样本）；
- 能测视觉 encoder、prefill（整段输入一次算完的预填充阶段，忘了回第 01 课）和总延迟。

本课是独立起点，不依赖[第 08 课](08_dynamic_vision.md)的动态分辨率产物。若第 08 课未完成：

- 使用固定 256×256 帧；
- 每帧固定 64 encoder token；
- 统一压到固定 LLM token cap；
- 不在本课调动态图像策略。

### 4.1 不可变模型锁

三臂要比出胜负，起点权重、encoder、tokenizer 只要有一个字节不同，结论就说不清来源。所以正式训练前运行一次 preflight，并生成只读 `manifests/base_model.lock.json`：

```yaml
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

preflight 必须用完整 commit/revision 下载，重新计算 checkpoint 的 SHA-256（文件指纹，第 01 课的老规矩），并把 vision/audio encoder、processor、decoder 的 `resolved_revision`、逐文件 SHA-256 和排序 file-tree SHA-256 一并冻结。三个 arm config 只允许引用该 manifest 及其实际 SHA-256；训练阶段不得联网解析 `main/master/latest` 这类会移动的引用，也不得接受 `baseline-v1` 之类的别名。升级任何资产必须新建 manifest 和实验 ID。

## 5. 完成本课需要掌握的操作

完成后应能独立完成以下工作：

1. 区分 frame-wise、stride-1 temporal adapter 与 feature-space strided
   Conv3D 三种时序方案；
2. 设计共享毫秒时间轴；
3. 构造不会泄露顺序答案的时序负对照；
4. 在相同帧/token 预算下比较时序结构；
5. 解释时间 receptive field 与短暂事件召回（该找到的短事件找到了多少）的关系；
6. 分离视频理解失败和音视频绑定失败；
7. 给出 fps、token、质量、延迟的 Pareto（一组"再改进一项就必须牺牲另一项"的最优权衡点）。

## 6. 时序融合与共享时间位置

本节是原理部分：三个机制，各按直觉、机制、代码落点、验证的节奏展开。参数量的数学集中在第 11 节，接口形状集中在第 8 节，这里先把"为什么这样设计"讲清。

### 6.1 视频不是图片列表

视频像翻页动画书：单看每一页是静态图，页与页连起来才有"动作"。类比失效处有两个，恰好对应本课两个零件：动画书页与页之间等间隔，真实视频的采样 fps 会变，所以不能拿页码当时间，要记真实毫秒（6.3 节）；动画书没有声音，真实视频的声轨必须和画面钉在同一条时间线上（6.4 节）。

先划清两类问题的边界。静态帧可以回答：

- 出现了什么物体；
- 场景在哪里；
- 人物大致在做什么。

以下问题需要时间结构：

- A 在 B 之前还是之后；
- 物体从什么状态变为什么状态；
- 一个动作发生了几次；
- 哪个声音对应哪个画面；
- 短暂事件出现在何时。

第 1 节的三个缺口各对应一个零件：跨帧计算由 6.2 的时序融合补上，时序位置由 6.3 的真实毫秒时间戳补上，声画对齐由 6.4 的统一时间轴打包补上。

判断模型是否真用了时间，可靠的办法是反事实四件套：正放、倒放、打乱、换音轨（9.3 节造数据，第 14 节算指标）。模型若只在正放上得分、倒放后答案不翻转，说明它靠的还是物体共现。

### 6.2 本课比较的三类时序融合

A 臂是"帧间完全不通气"的对照组；B 臂是"先各编各的，事后在特征上补一轮相邻帧沟通"；C 臂是"在特征空间做时间步长为 2 的 3D 卷积，一边沟通一边把时间长度减半"。三者都不动那个冻结的图像 encoder，这是公平比较的底线。


1. **Frame-wise matched baseline**
   - 每帧独立编码；
   - 相邻两帧做固定平均，得到与 B/C 相同的 8 个时间格点；
   - 没有可训练时序模块，是主实验的必需基线；
   - 纯 16 帧 chronological concat（按时间顺序直接拼接、不做任何融合）另作
     冻结历史 reference，不进入三臂胜负。
2. **Stride-1 temporal adapter**
   - 先逐帧编码；
   - 再在同一空间 patch 的视觉 embedding 上融合相邻帧；
   - adapter 后使用与 A 相同的固定相邻帧平均，得到 8 个时间格点；
   - 容易复用冻结图像 encoder。
3. **Feature-space strided Conv3D**
   - 仍先使用同一个冻结 2D image encoder；
   - 把 patch grid 恢复成 `[time, height, width]`，再做时序 stride=2 的
     depthwise Conv3D；
   - 空间 kernel 固定为 `1×1`，只合并相邻时间步；
   - 输出 8 个与 A/B 相同的时间格点，但不改 vision encoder 的输入。

本课不把 raw-pixel tubelet stem（直接在像素上切"时空小方块"当卷积输入）作为正式 C 臂。raw-pixel tubelet 会替换 SigLIP2 的 patch embedding，无法再声称三臂使用同一个冻结 vision encoder。要研究它，放到单独的扩展实验里。

形状与接口锁在第 8 节，公平控制与参数量公式在第 11 节；验证靠 preflight 的 shape 检查和逐参数明细。

### 6.3 时间位置

frame index 只表示采样后的次序，好比页码：两本厚薄不同的书，同一页码讲的内容完全不同。不同视频或不同 fps 下，同一个 frame index 可能对应不同的真实时刻，因此训练和推理还要保存毫秒时间。

统一位置至少保留 `absolute_time_ms`、`frame_index`、`source_fps`、
`sampled_fps`、`clip_start_ms` 和 `clip_end_ms`。打包 token 时从 `frame_time_ms` 计算时间位置，从帧号算是错的。

对同一视频使用两种采样 fps，检查对应同一真实时刻的帧得到相同的时间 bucket。这个单测过不了，后面所有时序指标都是空中楼阁。

### 6.4 声画对齐

视频帧 token 和音频 token 必须映射到同一个时间单位，"哪个声音对应哪个画面"才有讨论的基础。

可选择的做法有三种：

- 交错打包（两路 token 按时间排进同一条序列）；
- 分流编码后 cross-attention（交叉注意力：两路各自成序列，靠注意力互查）；
- 对齐窗口内附加 time embedding。

本课固定使用"统一时间 bucket + segment id"。time bucket 表示真实时间，segment id（在代码里就是 `modality_id`）区分音频与视频；二者都必须进入模型，并在 trace 中可见。固定成一种做法的理由：本课的实验变量是时序融合，声画打包若同时开放成变量，臂间差异就说不清来源。交错与 cross-attention 的正面比较留给扩展题 2。

打包接口是第 8 节的 `AVTokenPacker`；单测要求构造同一时刻的音频、视频 token，确认时间位置相同且 `modality_id` 不同（步骤 6）。

## 7. MiniMind-O 当前机制与代码落点

动手前先搞清官方代码有什么、缺什么。当前官方代码主要支持图像和整段音频：

- `dataset/omni_dataset.py`
  - schema 有 `image_bytes`、`question_audios`；
  - 默认只读最后一轮 user 的第一张图；
  - 没有 `video_bytes/path`、frame timestamp 或 AV sync 字段。
- `model/model_omni.py`
  - `forward()` 能处理某些多图 tensor 形状；
  - 每张图仍经相同 `vision_encoder → vision_proj`；
  - 没有 temporal module；
  - 图像与音频分别注入，没有共享时间位置。
- `model/model_minimind.py`
  - 统一 1D causal sequence；
  - 没有显式的视觉时间轴。
- `trainer/train_sft_omni.py`
  - collate（把一批样本拼成 batch 的函数）不处理变长帧、fps、timestamp
    或音视频配对 mask。
- `eval_omni.py`
  - 没有视频抽帧协议和负对照评测。

所以本课新增独立的视频 contract（数据侧的字段契约），明确保存帧、时间戳、fps 和音视频配对关系。现有多图字段不能代替这些信息。

## 8. 视频编码和 AV 打包接口

推荐接口：

| 接口 | 输入 | 输出 |
|---|---|---|
| `VideoProcessor.decode` | `video_ref`、sampling spec | frames、`frame_times_ms`、clip metadata |
| `TemporalVisionEncoder` | frames、frame mask | video features、video times |
| `AVTokenPacker` | video/audio features、timestamps | embeds、`modality_id`、time ids、attention mask |

B 臂先用冻结 2D vision encoder 把 `[B,F,C,H,W]`（batch、帧数、通道、高、宽）编码为 `[B,F,P,D]`（P 是每帧 patch 数，D 是特征维），再对每个空间 patch 执行两层 gated temporal DW-Conv1D。输出仍为 `[B,F,P,D]`；相邻两帧固定平均后得到 `[B,F/2,P,D]`，最后进入共同 connector 和固定空间 reducer。

对照臂（C）使用同一个冻结 2D vision encoder，先得到 `[B,F,H_p,W_p,D]`，再转为 `[B,D,F,H_p,W_p]`。第一层 gated depthwise Conv3D 使用 `kernel=(2,1,1)`、`stride=(2,1,1)`；第二层使用 `kernel=(1,1,1)`、`stride=(1,1,1)`。输出恢复为 `[B,F/2,H_p,W_p,D]`，然后进入与 B 臂相同的 connector 和 reducer。

`DW-Conv` 是 depthwise convolution：每个通道单独做卷积；后面的 `1×1` pointwise linear 再混合通道。正式配置使用 `F=16`，A/B/C 都输出 `F'=8` 个时间格点。第 `j` 个输出格点的时间戳固定为输入帧 `2j` 与 `2j+1` 的真实时间中点——时间减半之后，时间戳必须跟着重新计算，不能沿用旧帧的。

三个臂随后使用同一个确定性空间 reducer。reducer 用死规则的原因：内容相关的挑选会成为混杂变量，而且那是第 10 课的主题。正式配置固定 `Q=256`，因此每个时间格点保留 `Q/F'=32` 个视觉 token。对 `8×8` patch grid，先切成 `4×4` 个 `2×2` strata（分层抽样的"层"，保证保留的 patch 空间上均匀铺开），每个 stratum 保留 2 个 patch；候选按 `(distance_to_stratum_center, y, x, original_position_id)` 排序。三臂使用同一 tie-break（并列时的裁决顺序），最终 token 依次按 `(time_ms, y, x)` 排列。任何臂都不得改用 attention score、随机 top-k 或另一套 reducer。

## 9. 数据 recipe

### 9.1 Schema

一条带反事实标注的样本长这样：

```yaml
id: av_000001
messages:
  - role: user
    content:
      - type: video
        video_ref: videos/000001.mp4
        start_ms: 1200
        end_ms: 9200
      - type: audio
        audio_ref: videos/000001.mp4#audio
        start_ms: 1200
        end_ms: 9200
      - type: text
        text: "杯子是在门打开之前还是之后掉落？"
  - role: assistant
    content:
      - type: text
        text: "之后"
video:
  sha256: "..."
  source_fps: 30.0
  duration_ms: 10000
  has_audio: true
timeline:
  origin: source_media_start
  time_base: milliseconds
  frame_times_are: source_absolute
  audio_times_are: source_absolute
events:
  - label: door_open
    span_ms: [2100, 2600]
  - label: cup_fall
    span_ms: [4300, 4700]
source: owned_or_dataset_name
license:
  annotation: verified
  media: verified_or_source_dependent
  allowed_use: research
  redistributable: false
split: test
task: temporal_order
counterfactuals:
  reverse:
    transform_id: reverse_v1
    answer: "之前"
    label_status: relabeled
  shuffle:
    transform_id: shuffle_seed_17
    answer: "无法判断"
    answerable: false
```

`start_ms/end_ms`、frame timestamp 与 audio timestamp 都以源媒体起点为 0。若解码器返回 PTS（容器里的显示时间戳），processor 必须用记录在 manifest 中的 time base 换算成 source-absolute 毫秒；不能把视频用 clip-relative 时间、音频用 source-absolute 时间后直接分桶——两套原点各差一个 `clip_start_ms`，声画就这样错开了。

### 9.2 任务配比

配比向"必须用时间才能答对"的任务倾斜：

- 25% before/after；
- 15% 状态变化；
- 15% 动作计数；
- 15% 短暂事件定位；
- 15% 声音来源；
- 10% 唇动/说话人绑定；
- 5% 静态内容回放。

### 9.3 负例

对每个正例预生成带标签的反事实：

- `reverse`：重新计算 before/after、计数或状态答案，保存 `counterfactual_answer`；
- `shuffle`：仅用于顺序证据被破坏的样本，标准答案固定为"无法判断"；
- `audio_mismatch`：优先选同任务、同长度、同答案分布的配对音轨，并提供错配后的
  新答案；
- `audio_shift_ms = ±500/1000/2000`：只有能从事件标注重新计算答案时才进入
  accuracy；否则只进入原答案 log-prob sensitivity；
- `same_objects_different_order`：原始与反事实各自有独立答案。

问题文字尽量保持不变，但媒体事实改变时答案必须随事实重标注。无法获得合法新答案的样本不得进入 accuracy，只能进入明确标为 diagnostic 的 log-prob 分析。

### 9.4 切分

切分的敌人是泄漏——同一段素材换个裁剪出现在训练和测试两边，分数就是假的：

- 按原视频 identity 切分；
- 同一视频的裁剪不得跨 split；
- 同一场景连续拍摄视为同一 group；
- 模板化合成视频按动作脚本 family 切分；
- test 中保留未见 fps、时长和事件密度。

### 9.5 规模

| 档位 | clips | 平均时长 | 用途 |
|---|---:|---:|---|
| pilot | 5k | 4–8s | 4–8 帧、shape 与负例 |
| standard | 80k | 8–30s | 三臂公平训练 |
| full | 500k+ | 5s–3min | 多尺度视频课程 |

license 不明确的公开视频：

- 可保存 URL 与 hash；
- 不默认允许再分发；
- 发布 manifest 时移除受限媒体。

## 10. 实施步骤

八个步骤按依赖顺序排：先锁协议和数据，再搭三臂，最后统一时间轴、训练、上负对照考卷。

### 步骤 1：定义抽帧协议

固定：

```yaml
sampling:
  strategy: uniform_fixed_count
  frames: 16
  deterministic_eval: true
  reads_event_annotations: false
```

正式训练和评测的 sampler 都不得读取 `events`、答案、evidence span 或其他标注字段——sampler 偷看了事件位置，等于考试前告诉模型答案在第几页。训练可以在均匀格点内做带 seed 的 jitter（小幅随机偏移）；评测固定确定性 uniform sampling。三臂读取同一份 frame manifest。

event-aware sampling 只作为单独的 oracle upper bound：它可以读取事件 span，但结果必须写入 `metrics/oracle_event_sampling.jsonl`，不得用于选择主臂、计算主指标或声称模型在未知事件位置上具备同等能力。

### 步骤 2：扩展数据字段

数据层返回 `frames`、`frame_valid_mask`、`frame_times_ms`、`audio_features`、
`audio_times_ms` 和 `event_spans`。

`event_spans` 只供标签生成和离线评测使用，不得传给 sampler、temporal module 或 Thinker。解码失败必须有计数和原因，禁止静默替换黑帧——黑帧混进训练只会教坏模型。

### 步骤 3：建立 frame-wise 基线

每帧独立编码后，把相邻两帧同一 patch 的 feature 做固定算术平均，时间戳取两帧中点。该臂没有可训练时序参数。它与 B/C 一样得到 `[B,8,P,D]`，再使用第 8 节锁定的每时间格点 32-token reducer。纯 16 帧 chronological concat 只作历史 reference，不进入主三臂。

### 步骤 4：实现 late temporal adapter

正式 B 臂固定为两层 gated temporal depthwise Conv1D adapter：kernel=3、stride=1。vision encoder 先输出 `[B,F,P,D]`，再把每个空间 patch 位置独立整理成 `[B×P,D,F]`——时间轴放到最后一维，卷积只沿时间滑动。每层执行 `DWConv1D → SiLU → pointwise linear → scalar gate → residual`（SiLU 是激活函数；scalar gate 是可学的标量门，乘在模块输出上；residual 即输出加回输入），并使用有效 frame mask。局部 temporal attention 只能作为扩展实验，不能混进 B。

把 gate 初始化为接近零，使 adapter 初始输出接近原图 checkpoint——模型一开始"几乎还是原来的模型"，避免刚加载就把图像能力砸了。加载后先运行逐帧输出对比。adapter 输出后执行与 A 相同的固定相邻帧平均，得到 `[B,8,P,D]`，再开始训练。

### 步骤 5：实现 feature-space strided Conv3D 对照

正式 C 臂也从冻结 image encoder 的 `[B,F,P,D]` 输出开始。先按 patch grid 恢复为 `[B,D,F,H_p,W_p]`，再执行两个 block：

1. 第一层 depthwise Conv3D 使用 kernel=`(2,1,1)`、stride=`(2,1,1)`；
   residual 分支使用相同 stride 的时间平均池化；
2. 第二层使用 kernel=`(1,1,1)`、stride=`(1,1,1)`，直接 residual；
3. 两层都使用与 B 相同的 `SiLU → D×D pointwise linear → scalar gate`，
   所有卷积和 linear 都不使用 bias。

C 的输入帧数必须为偶数；正式实验固定 `F=16`，不在运行时静默补帧。第一层输出的时间戳取两帧真实时间的中点。C 后接与 A、B 相同的 connector 和 token reducer，不再加入 tubelet stem 或 temporal attention。

最终进入 LLM 的 token 数必须与其他臂相同。A/B/C 都使用 8 个中点时间格点、每格 32 个空间 token 和同一 tie-break。C 与 A、B 的 vision encoder FLOPs（浮点运算次数）应相同，因为 stride=2 发生在 encoder 之后；它只能减少后续 connector、reducer 和 temporal module 第二层处理的中间 token。若日志显示 C 的 encoder FLOPs 更低，说明三臂实际走了不同 encoder 路径——这是公平性事故，先修再跑。

### 步骤 6：统一 AV 时间位置

时间 bucket 例如 40ms 或 80ms。

图像 token 继承 frame timestamp。

音频 token 继承 chunk 中心 timestamp。

相同 bucket 内可以同时出现图像和音频 token，因此仍需 `modality_id`。该字段是与 token 序列等长的整数 tensor，不再使用其他复数命名。单测应构造同一时刻的音频、视频 token，确认时间位置相同且 `modality_id` 不同。

### 步骤 7：训练 curriculum

curriculum（课程表式训练：由易到难分阶段喂数据）依次进行四个阶段：

1. 静态图 replay 与 video caption；
2. temporal order 与 state change；
3. AV sync 与 source binding；
4. joint LoRA（低秩适配：冻住主干，只训练插在权重旁的小矩阵）与 capability replay。

每阶段固定训练 token，不用样本数代替——clip 长短不一，按样本数分配预算会被长视频偷走。

### 步骤 8：建立负对照评测

对同一模型、同一问题运行并使用各自合法标签：

- normal：原答案；
- reverse：重新标注后的顺序/状态答案；
- shuffle：证据不足时的"无法判断"；
- audio mismatch：配对音轨对应的新答案；
- audio time shift：可重标注时测 accuracy，否则只测原答案 log-prob 变化。

输出逐 case 的正常正确性、反事实正确性、预测是否随事实改变，以及原答案 log-prob 变化。不得只输出"变换后分数下降"——分数下降说明不了模型读懂了时间。

## 11. 三个对照实验组

| 臂 | 时序机制 | 输出时间格点 | 每格空间 token | 最终视频 token |
|---|---|---:|---:|---:|
| A | frame-wise feature + 固定 adjacent-pair mean | 8 | 32 | 256 |
| B | 两层 gated DW-Conv1D + 同一 adjacent-pair mean | 8 | 32 | 256 |
| C | 两层 feature-space DW-Conv3D，首层 temporal stride=`2` | 8 | 32 | 256 |

公平控制逐条锁死：

- 同一抽帧结果；
- 同一 16 帧输入、同一 8 个中点时间格点；
- 同一每格 32-token 空间预算、同一 reducer 与 tie-break；
- 同一最终 256 个 LLM video tokens；
- 同一 backbone、connector 与训练 token；
- A 没有可训练 temporal module，新增时序参数记为 0；
- B、C 的新增 temporal 参数量控制在 ±10%，并在 preflight 导出逐参数明细；
- 三臂分别报告总可训练参数和实测 FLOPs，不能用 B/C 的参数匹配描述 A；
- 同一 AV packer 和时间 bucket；
- 同一生成参数与 seed。

A/B/C 读取完全相同的采样帧。三臂 reducer 输入都必须是 `[B,8,8,8,D]`，输出都是 `[B,8,32,D]`；若 shape、时间戳或逐格 keep count 不同，preflight 直接失败，不能在进入 LLM 前用另一套临时 top-k 补齐。

设 feature dimension 为 `D`，忽略共同的 connector 后，两个正式 temporal module 的参数数目为：

$$
P_B=2(D^2+3D+1),\qquad
P_C=(D^2+2D+1)+(D^2+D+1).
$$

最后的 `1` 是每层的 scalar gate。两者只差 `3D` 个参数，天然满足 ±10% 的控制；但 preflight 仍以实际 `requires_grad` 列表为准，不用公式代替检查——公式算的是设计，检查的是实现。

## 12. 配置与伪代码

```yaml
experiment: exp09_native_video
model_lock: manifests/base_model.lock.json
run_lock: manifests/run.lock.json
video:
  frames: 16
  sampling: uniform_fixed_count
  image_size: 256
  encoder: frozen_siglip2
  llm_token_cap: 256
  formal_sampling: uniform
  reads_event_annotations: false
matched_output:
  time_slots: 8
  timestamp: adjacent_pair_midpoint
  tokens_per_time_slot: 32
  spatial_reducer: uniform_4x4_strata
  tokens_per_stratum: 2
  tie_break: [distance_to_stratum_center, y, x, original_position_id]
oracle:
  event_aware_sampling:
    enabled_for_main_metrics: false
temporal_modules:
  gate_init: 0.0
  A:
    type: none
  B:
    type: gated_dwconv1d
    layers: 2
    kernels: [3, 3]
    strides: [1, 1]
    pointwise_dim: model_dim
    bias: false
  C:
    type: gated_feature_dwconv3d
    layers: 2
    kernels: [[2, 1, 1], [1, 1, 1]]
    strides: [[2, 1, 1], [1, 1, 1]]
    residual: [temporal_avg_pool, identity]
    pointwise_dim: model_dim
    bias: false
parameter_control:
  compare_new_temporal_params: [B, C]
  max_relative_difference: 0.10
  report_total_trainable_params_for: [A, B, C]
  report_measured_flops_for: [A, B, C]
audio:
  encoder: frozen_sensevoice
  time_bucket_ms: 80
packing:
  order: timestamp_then_modality
  preserve_modality_id: true
train:
  total_tokens: 150_000_000
  seeds: [17, 23, 41]
  static_image_replay: 0.15
```

preflight 在 `run.lock.json` 中写入 base-model lock、三个 arm config、encoder file-tree 和抽帧 manifest 的实际 SHA-256；训练入口逐项重算，不匹配即终止。

三臂的前向骨架用伪代码写出来是：

```python
frames, t_v = decode_and_sample(video, spec)
v = vision_encoder(frames)              # [B,F,P,D]，三臂相同
if arm == "B":
    v = temporal_dwconv1d(v, frame_mask)
    v, t_v = adjacent_pair_mean(v, t_v)  # [B,8,P,D]，时间取中点
elif arm == "C":
    assert v.shape[1] % 2 == 0
    v = to_spatiotemporal_grid(v)        # [B,D,F,H_p,W_p]
    v = temporal_dwconv3d_stride2(v, frame_mask)
    v = from_spatiotemporal_grid(v)
    t_v = pairwise_midpoint(t_v)
else:
    v, t_v = adjacent_pair_mean(v, t_v)  # A 与 B/C 使用相同时间格点
v = connector(v)
v, t_v = fixed_uniform_spatial_reduce(
    v,
    t_v,
    tokens_per_slot=32,
    strata=(4, 4),
    tie_break=("center_distance", "y", "x", "original_position_id"),
)
assert v.shape[1:3] == (8, 32)
a, t_a = audio_encoder(audio)
embeds, time_ids, modality_id = av_pack(v, t_v, a, t_a)
logits = thinker(embeds, time_ids=time_ids, modality_id=modality_id)
```

## 13. 训练预算与 8 卡建议

### Pilot

- 1×24GB；
- 4–8 帧；
- 冻结两个 encoder；
- 离线缓存 frame feature；
- 5k clips，2k steps。

### Standard

- 4×24–48GB；
- 16 帧、256 LLM tokens；
- 80k clips；
- 150M–400M 训练 token；
- temporal module + Thinker LoRA。

### 8 卡

- 8×24GB：四卡一臂或 2 卡×3 seed 并行；
- 8×48/80GB：32–64 帧、在线 encoder；
- NVLink 好时可跨卡训练，PCIe-only 优先 feature cache；
- 保持 global tokens/step 一致；
- cache key 包含视频 hash、fps、sampling seed、encoder revision；
- 音视频解码放 CPU worker，监控 data wait；
- 不用 padding 帧占训练 token 预算。

## 14. 指标与测量

### 能力

- temporal order accuracy；
- state-change accuracy；
- action count exact；
- transient-event recall；
- temporal grounding IoU / Recall@IoU（IoU 即交并比：预测时间段与真实时间段的重叠占比）；
- AV source binding accuracy；
- AV sync classification AUC（AUC：随机抽一正一负，模型把正例排在前面的概率）；
- 静态图回归；
- 文本与音频回归。

### 负对照敏感度

反事实首先测"在新输入上答对"；只看分数是否机械下降，说明不了模型用没用时间：

- `reverse_counterfactual_accuracy`：反转媒体上对重新标注答案的准确率；
- `reverse_flip_consistency`：normal 与 reverse 两边都答对，且答案按事实改变的
  配对比例；
- `shuffle_abstention_accuracy`：顺序证据被破坏后回答"无法判断"的准确率；
- `audio_counterfactual_accuracy`：错配音轨上对配对新答案的准确率。

没有合法新答案的 audio shift/mismatch 只计算诊断量：

$$
\Delta\log p_{\text{original}}
=
\log p(y_{\text{original}}\mid x_{\text{normal}})
-
\log p(y_{\text{original}}\mid x_{\text{counterfactual}}).
$$

这个差值说明模型对媒体变化是否敏感，不等于反事实回答正确。accuracy、拒答率和 log-prob sensitivity 必须按任务分开报告，不能合成一个"退化分"。

### 系统

- decode ms/video；
- vision encode ms/frame；
- temporal module ms；
- LLM prefill p50/p95；
- total visual tokens；
- peak HBM（显卡上的高带宽显存峰值占用）；
- clips/s 和 effective tokens/s；
- data-loader idle ratio；
- 不同时长 bucket 的 OOM 率。

### 测量方法

- 固定 sampled frames，不在模型臂间重新抽样；
- latency 预热 20、测 100；
- 能力至少 3 seed；
- 按运动强度、时长、fps、事件持续时间分桶；
- 为短暂事件单独报告，防止被大量静态视频稀释。

主差值按原视频 identity 配对，使用 10,000 次分层 bootstrap（有放回重抽样估计置信区间）；分层字段为任务和事件时长。正文中的"显著高于"统一表示配对 95% CI（置信区间）下界高于 0。

## 15. 验收条件

1. B 或 C 在时序综合指标上比 A 提升至少 5 个百分点；
2. 最佳臂的 `reverse_counterfactual_accuracy` 和
   `reverse_flip_consistency` 都显著高于 A；
3. `shuffle_abstention_accuracy` 显著高于"始终猜一个顺序答案"的基线；
4. AV 任务必须报告 `audio_counterfactual_accuracy`；无法重标注的样本只能报告
   $\Delta\log p_{\text{original}}$，不得计入 accuracy；
5. 同 token budget 下，最佳臂 prefill 不劣于 A 超过 15%；
6. 静态图能力保留 baseline 的 ≥97%；
7. 文本、音频回归 ≤2%；
8. 至少一个挑战集证明模型处理了真实顺序而非物体共现；
9. 输出质量—帧数—token—延迟 Pareto；
10. event-aware oracle 单列，不参与以上任何阈值。

## 16. 失败诊断表

先核对标签和时间戳语义，再 trace 张量，最后才怀疑训练本身。

| 现象 | 可能原因 | 检查/修复 |
|---|---|---|
| reverse 后仍答原标签 | 时间位置未生效或反事实未重标注 | 先核对新答案，再 trace time ids |
| shuffle 后不拒答 | 数据仍可由单帧回答或未训练拒答 | 重做 paired-order 数据与不可回答标签 |
| AV mismatch 后仍答原标签 | 模型忽略音频或配对标签错误 | 核对新答案，做 audio-only 可回答性审计 |
| 训练 loss 降但 QA 不升 | caption 占比过高 | 增加可验证 temporal QA |
| 短事件漏掉 | 正式 uniform fps 太低 | 提高统一 fps 或缩短 clip；event-aware 仍只作 oracle |
| C 显存过高 | stride=2 后仍保留旧张量，或第二层仍处理原帧数 | trace 每层 shape 与显存 |
| B 只在长视频好 | temporal kernel/receptive field | 分时长调层数 |
| 静态图回退 | video token 占比过高 | 加 image replay |
| 数据加载卡住 | 在线解码瓶颈 | 缓存、分片、本地盘 |
| 声画错位 | 起止时间基准不同 | 全部换绝对毫秒 |

## 17. 逐 case 要求

至少 100 个 case：

- 20 before/after；
- 15 reverse-sensitive；
- 15 状态变化；
- 15 计数；
- 15 transient event；
- 10 声音来源；
- 10 唇动/说话人绑定。

每个 case 输出：

- 关键帧 contact sheet（关键帧拼成的一张缩略图总表）、事件时间线和音频波形或时间段；
- 正常输出与原答案；
- reverse 输出与重标注答案；
- shuffle 输出与"无法判断"标签；
- mismatch 输出与配对新答案，或原答案 log-prob 变化；
- 各臂答案、保留帧与 token、延迟、显存和错误归因。

错误标签：

- `frame_sampling_miss`；
- `temporal_order_miss`；
- `state_tracking_miss`；
- `av_alignment_miss`；
- `transient_event_miss`；
- `reasoning_miss`。

## 18. 交付物

```text
exp09/
  configs/{framewise,late_adapter,conv3d}.yaml
  manifests/base_model.lock.json
  manifests/run.lock.json
  manifests/arm_config_hashes.json
  data/manifest.jsonl
  data/license_report.md
  data/negative_pairs.jsonl
  checkpoints/
  metrics/aggregate.json
  metrics/per_case.jsonl
  traces/timeline.jsonl
  metrics/oracle_event_sampling.jsonl
  plots/quality_frames_tokens_latency.png
  cases/index.md
  report.md
```

## 19. 复现清单

- [ ] 固定视频 decoder 版本；
- [ ] preflight 已冻结代码 commit、checkpoint repo revision、checkpoint SHA-256 和所有 encoder file-tree hash；
- [ ] 三臂 model-lock SHA-256 完全一致且配置内无浮动 ref/别名；
- [ ] 固定抽帧协议与 seed；
- [ ] 正式 sampler 不读取 event/evidence/answer 字段；
- [ ] event-aware sampling 只存在于单列 oracle 报告；
- [ ] 保存原始/采样 fps；
- [ ] 保存 frame timestamps；
- [ ] 保存音频 timestamps；
- [ ] 三臂使用相同帧；
- [ ] 三臂最终 token cap 相同；
- [ ] A/B/C 都输出 8 个相同时间格点，每格保留 32 个空间 token；
- [ ] 三臂使用同一固定 reducer 与 tie-break；
- [ ] A 只使用 frame-wise feature + 固定 adjacent-pair mean；
- [ ] B/C 的模块结构与 kernel/stride 已按主臂表冻结；
- [ ] reverse 已重新标注，shuffle 已标为不可回答；
- [ ] mismatch/time-shift 有配对新答案，或只进入 log-prob diagnostic；
- [ ] 按运动强度分桶；
- [ ] 按事件持续时间分桶；
- [ ] 跑静态图/文本/音频回归；
- [ ] 输出逐 case；
- [ ] 验证 checkpoint 恢复。

## 20. 前沿对照与改造方向

本课的三个设计决定——真实时间位置、特征侧时序融合、音视频按时间交错——在 2024-2026 年的公开系统里都能找到放大版。[Qwen2-VL](https://arxiv.org/abs/2409.12191) 把位置编码拆成时间、高、宽三路（M-RoPE，第 08 课已经用过它的二维部分），视频帧的时间位置来自真实时间而非帧号，6.3 节"两种 fps、同一时刻、同一 bucket"的单测就是对着这个设计做的；它还在 patch embedding 阶段用时间深度为 2 的 3D 卷积把相邻两帧并成一组——与本课 C 臂"首层时间 stride=2"同方向，但做在像素侧，正是本课为保住"三臂同一冻结 encoder"而排除的 raw-pixel tubelet 路线。[Qwen2.5-Omni 技术报告](https://arxiv.org/abs/2503.20215)（第 01 课引过）用 TMRoPE 把音频与视频 token 按真实时间交错进同一序列；本课的"统一时间 bucket + segment id"是它的手工简化版：我们把时间做成显式的 bucket id，它把时间编进位置编码本身。[Nemotron 3 Nano Omni](https://arxiv.org/abs/2604.24954) 在 video encoder 里用 Conv3D 做早期时序融合，可以看作 C 臂思路的工业版；[MiniCPM-V 4.5](https://arxiv.org/abs/2509.18154) 用统一 3D-Resampler 把多帧联合压成极少的 token，那是第 10 课的主战场。[VideoMAE](https://arxiv.org/abs/2203.12602) 则证明视频冗余大到遮掉 90%-95% 的时空 patch 仍能有效预训练——这正是本课"逐帧独立编码浪费预算"的实验依据。

规模问题：参数量（26M 级对几十亿）、帧数（16 帧对前沿的长视频输入）、数据量（80k clips 对海量语料），这些砸资源就能缩小。机制问题里，本课解决三个：真实毫秒时间位置、轻量时序融合、音视频统一时间轴。本课解决不了的还有两个：冻结的逐帧 encoder 看不到像素级运动（raw-pixel tubelet 被公平性要求排除在三臂外），以及分钟级以上长视频的记忆结构（扩展题 4）。



1. **time bucket 宽度消融。** 把第 12 节 config 的 `audio.time_bucket_ms` 从 80 分别改成 40 和 160，各训一个 pilot 档模型（5k clips、2k steps、1×24GB、冻结 encoder + 离线特征缓存，单个配置数个 GPU-hour）。预期：AV sync AUC 与声源绑定 accuracy 随 bucket 变宽而下降，40 与 80 接近。失败判定：三个宽度指标无差异——先跑步骤 6 的 time-id 单测和 trace，确认时间位置真的进了模型，再审计 AV 任务是否能被单模态回答。
2. **B 臂时间感受野实验。** 把 `temporal_modules.B.kernels` 从 `[3, 3]` 改成 `[5, 5]`，或层数 2 改 3。注意这会打破与 C 的 ±10% 新增参数控制，必须另开实验 ID，不进主三臂。预算：standard 档单臂一份（80k clips、与主臂相同的训练 token，4×24–48GB）。预期：长时长桶与慢事件桶的 temporal order、state-change 提升，短 clip 桶持平——诊断表"B 只在长视频好"一行就是这个实验的先验。失败判定：所有时长桶都无提升，说明感受野不是瓶颈，回数据里检查长程依赖任务的占比。
3. **把时间写进位置编码。** 参照 TMRoPE 的方向，把"time bucket 加性 id + segment id"改成在 Thinker 的 RoPE 里加一路时间维（第 08 课已把位置编码拆成多路，这里再加一路）。改动位置：`AVTokenPacker` 输出 time id 的消费方式、Thinker 的位置编码实现。预算：先 pilot 验证 shape 与静态图回归，再 standard 单臂重训。预期：temporal grounding 与 AV 绑定持平或提升，对 test 集保留的未见 fps/时长外推更稳。失败判定：静态图回归掉出 97% 红线，说明时间维挤占了原有位置编码容量，回退设计。

VideoMAE 的"高冗余"结论能在本课设置里复现方向：对训好的最佳臂，评测时在 reducer 之后随机丢弃 30%/60%/90% 的视频 token（固定 seed，只推理不训练，单卡 2–4 GPU-hour）。预期同方向：静态内容回放与 caption 类指标掉得很慢，transient-event recall 与动作计数最先崩——冗余集中在慢变化内容上。这条退化曲线顺手成为第 10 课设计压缩策略的第一张参考图。若所有任务同步崩掉，说明 256 个 token 已经没有冗余，第 10 课的压缩空间要重新估计。

## 21. 论文精读

每篇带着一个能落到本课产物上的问题读；读完把答案写进 report，答不上来就回去重读。

1. [VideoMAE](https://arxiv.org/abs/2203.12602)
   - 精读：tube masking、视频冗余、预训练结构。
   - 带着问题读：为什么视频能承受 90%-95% 的遮挡？tube masking 为什么要沿时间轴贯穿遮住同一空间位置——如果逐帧独立随机遮，相邻帧的冗余会怎样把答案泄漏回来？
   - 阅读检查：说明视频冗余如何支持较高的 mask ratio；再对照本课想一层：同一份冗余，如何既让压缩可行（第 10 课），又让短暂事件变难（第 14 节要求单独报告 transient-event recall 的原因）。
2. [Qwen2-VL](https://arxiv.org/abs/2409.12191)
   - 精读：图像/视频统一表示与 M-RoPE。
   - 带着问题读：它的时间位置来自帧号还是真实时间？
   - 阅读检查：写出不同 fps 下保持真实时间位置一致的计算方法，与 6.3 节"两种 fps 同一 bucket"的单测互相印证。
3. [MiniCPM-V 4.5](https://arxiv.org/abs/2509.18154)
   - 精读：统一 3D Resampler 与高密度视频压缩。
   - 带着问题读：时序压缩做进 resampler 之后，每个输出 token 的时间戳还剩下什么？
   - 阅读检查：标出 3D Resampler 相对 vision tower 的位置和输入输出形状，并写出它与本课"encoder 之后、reducer 之前融合"方案在时间戳保留上的差别。
4. [Nemotron 3 Nano Omni](https://arxiv.org/abs/2604.24954)
   - 精读：video encoder、Conv3D 与时序数据 recipe。
   - 带着问题读：时序融合做进 encoder（像素侧）与做在 encoder 之后（本课 C 臂），省下的算力各在哪一段？
   - 阅读检查：分别计算早期时序融合对 encoder 和 LLM 成本的影响；对照步骤 5 里"C 的 encoder FLOPs 必须与 A/B 相同"的检查，想清楚同样的 stride=2 放在不同位置，省的算力完全不同。
5. [Qwen3-Omni](https://arxiv.org/abs/2509.17765)
   - 精读：原生音视频输入、Thinker–Talker。
   - 带着问题读：音频与视频各自编码后，在哪一层、带着什么位置信息汇合？
   - 阅读检查：标出音频与视频信息在模型中的汇合位置，与本课 AVTokenPacker"进 Thinker 前按时间交错"对照，写出两种汇合点对时延和对齐精度的影响方向。
6. [MiniMind-O 官方仓库](https://github.com/jingyaogong/minimind-o)
   - 精读：图像和音频注入路径。
   - 带着问题读：第 7 节列的每条缺口，在代码里的确切位置是哪一行？
   - 阅读检查：列出现有多图 tensor 缺少的视频时间信息，逐条对应到本课 contract 新增的字段。

## 22. 扩展题

1. 加入 event-aware adaptive fps（按内容自适应抽帧），把它从 oracle 变成不读标注也能用的正式策略——难点在于"检测哪里有事件"这一步本身不能偷看事件标注。
2. 比较时间交错与双流 cross-attention 两种声画汇合方式（6.4 节被固定掉的那个变量，在这里放开）。
3. 训练可学习 AV sync head（判断声画是否同步的输出头）并将其作为辅助损失。
4. 扩展到 5–30 分钟视频，研究分层 temporal memory。
5. 将本课选定的最佳臂固定为[第 10 课](10_video_token_reduction.md)的输入，不再修改 temporal module。

最后盘点系统现状：耳朵会听（第一、二幕），眼睛在第 08 课学会看任意分辨率的图，这一课又学会了看真视频——帧带真实时间戳，跨帧变化被显式建模，声音和画面钉在同一条时间线上，并有反事实考卷作证。新麻烦也随之而来：16 帧的短 clip 就要吃掉 256 个视频 token，帧数一多，prefill 延迟和显存按倍数涨，可视频里明明一大半内容是重复的。下一课（[第 10 课](10_video_token_reduction.md)）就冲着这份冗余去：比较采样、合并、剪枝几类 token 压缩策略，画出质量—token—延迟的 Pareto，把视频 token 压到算力真正扛得住。
