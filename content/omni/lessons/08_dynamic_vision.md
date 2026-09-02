---
id: 08_dynamic_vision
title: "动态分辨率与 M-RoPE"
summary: "保住宽高比的动态切片，配上二维位置编码，在同样的 token 预算下，真能把 OCR、小目标和多图理解做得更好吗？"
unit: vision
play_tools: []
checkpoints:
  - "实现按像素预算规划的动态 tile planner（切块规划器）加一张全局缩略图。"
  - "给整图、切块和二维坐标定好明确的位置字段，接进 M-RoPE（二维旋转位置编码）。"
  - "在动态 batch 里把变长 mask 和 position ids 拼对。"
  - "用坐标打乱实验和同 token 预算对照，分清收益来自分辨率还是来自位置编码。"
---

# 第 08 课：动态分辨率、多图身份与 M-RoPE

> 内容：动态图像切片、多图身份与二维位置编码
> 建议时长：4–7 天  
> 最小硬件：1×24GB 做功能验证；2–4×24GB 做标准实验  
> 独立起点：由 preflight 锁定的官方 `jingyaogong/minimind-3o` dense checkpoint，不依赖实验 02–07

## 1. 一张大图怎么送进模型

前两幕的活已经干完：第一幕（01-04 课）把 MiniMind-O 拆开，搞懂了声音怎么变 token、8 路码本怎么错位、Talker 怎么把想法念出来；第二幕（05-07 课）把"你说完我再说"改造成"边听边想边说"；流式听、判断该不该开口，上一课又装上了全双工 routing：生成期间继续听，能 CONTINUE、PAUSE、REPLAN。到今天，这套系统有了一对能实时对话的耳朵和嘴巴。

但眼睛还停留在第 01 课的状态。SigLIP2（冻结的视觉编码器）只会看固定尺寸的图：不管你给它 2480×3508 的发票扫描件、细长的聊天记录截图，还是一张全景照片，预处理都先把它硬压成 256×256 的正方形，切成 8×8 共 64 个 patch（图像小方块：vision encoder 把图切成 32×32 像素的小块逐块编码，这是它看图的最小单位），最后变成 64 个视觉 token 交给 Thinker。小字糊成一团，长文档被竖向压扁，两张图一起送进去还分不清谁是谁。

第三幕"长出眼睛"从这里开幕：本课让静态图适应任意分辨率，第 09 课上原生视频，第 10 课把视频 token 压到算力扛得住。本课给眼睛装三个零件：一个按宽高比切图的动态切片器（大图切成若干 256×256 的 tile，外加一张看全局的缩略图）；一个位置感知 resampler（不管 encoder 这次看了多少 patch，都压成固定 64 条摘要，每条带二维坐标交给 Thinker）；一套 M-RoPE（让 Thinker 用"第几步、第几行、第几列"三个轴理解位置）。实验拆成 08A 和 08B 两段：08A 只比图像前端，08B 锁死前端只比位置编码，一次只动一个变量。主线剧情承接第二幕，但工程上本课从 preflight 锁定的官方 checkpoint 另起炉灶，不依赖 02–07 的产物；图像链路和音频链路互不纠缠，单卡也能独立开工。

第 09 课的视频就是"一串带时间戳的帧"：单帧都看不清、二维位置都说不明白，时序建模无从下手。第 10 课要把视频 token 往下压，预算记账用的正是本课"encoder patch token 和 LLM visual token 分开数"的口径。这一步含糊，后两课的结论全是糊账。

做完这课，你可以拿同一张发票问"总金额是多少"：fixed-256 臂把小字糊掉答错，dynamic 臂能读出来。再跑负对照：把 M-RoPE 臂的 64 个二维坐标随机打乱，空间题分数应声下跌；跌得动，才证明位置信息真被用上了。

本课术语：

| 术语 | 简要解释 |
|---|---|
| patch | vision encoder 把图切成的 32×32 像素小方块，每块编码成一个向量 |
| tile（切片） | 按宽高比从原图切出的 256×256 子图，每张单独过 encoder |
| global thumbnail | 整图缩成一张 tile 当全局缩略图，保布局；细节靠各 tile 补 |
| pixel budget | 像素预算：一张图最多送进 encoder 多少像素，决定 tile 数上限 |
| resampler / query slot | 用 64 个可学习 query 对任意数量的 patch 特征做 cross-attention（交叉注意力：query 来自一组向量，key/value 来自另一组），压成固定 64 条输出；每个输出位置叫一个 query slot |
| RoPE | 旋转位置编码：把"第几个位置"编码成向量的旋转角度，注意力算内积时自然带上相对距离；Thinker 现用一维版 |
| M-RoPE | 多轴 RoPE：文本 token 用 (t,t,t)，图像 token 用 (global_t, y, x) 三个轴 |
| 位置插值（position interpolation） | encoder 的位置编码只学过 8×8 网格，遇到 14×14 网格时按比例拉伸复用旧编码 |
| 分辨率桶（bucket） | 组 batch 时按 encoder patch token 总数把样本分组，大图小图不混桶，少浪费显存 |
| ANLS | OCR 类答案的宽容打分：按编辑距离给部分分，错一两个字符不算全错 |

## 2. 固定分辨率输入的限制

现在的图像链路对任何输入一视同仁：先硬压成固定正方形，再切固定数量的 patch。本课解决它带来的三个问题：强制缩放会损失细节；vision encoder 只能看到固定数量的 patch；一维位置编码不能完整表示二维布局。改造范围只包括图像预处理、视觉 token 打包和位置编码。

先分清两种 token，本课所有记账都建立在这个区分上：

- **encoder patch token**：vision encoder 对图片或 tile 产生的 patch 表示。
  动态切片后，它的数量随图片内容和 pixel budget 变化；
- **LLM visual token**：resampler 输出并送入 Thinker 的视觉表示。为公平比较，
  本课固定为每张图片 64 个。

动态分辨率增加的是 encoder 侧可见的细节，不是让动态臂向 Thinker 输入更多 LLM visual tokens。眼睛允许看得更细，但递给脑子的摘要永远是 64 条。少了这条约束，dynamic 臂赢了你也说不清是赢在"看得细"还是赢在"塞得多"。

本课不更换语言模型，不加入视频，不做视觉 token 剪枝，也不同时比较多种 connector（连接器：把编码器输出接进 Thinker 的小模块，第 02 课的主角）。

本课拆成两个连续的小实验。08A 只比较图像前端，所有臂都使用 1D RoPE；08B 锁定 08A 的 dynamic frontend 后，才比较 1D RoPE 与 M-RoPE。这样可以分别回答"动态切片是否有效"和"M-RoPE 是否额外有效"，不会把两个改动混成一个结论——混着改是对照实验里最常见的自欺。

两项实验共用同一条处理链。原图先经过 fixed resize 或保持宽高比的 dynamic tile；dynamic 配置额外生成一张 global thumbnail，并为每个 patch 保存 image、tile、x、y。冻结的 SigLIP2 和固定 connector 产生 patch feature；带二维位置的 64-query resampler 再把每张图压成 64 个视觉表示和 64 个 query 坐标。08A 的所有臂继续使用 1D RoPE；08B 才在冻结图像前端后比较 1D RoPE 与 M-RoPE，最后把结果送入 MiniMind Thinker。

验收时同时报告任务质量、encoder/LLM token 数、prefill 延迟和峰值显存。将四个量绘制为 Pareto 曲线（同一张图上同时看质量和成本：曲线上的点想再提质量，就必须多付 token、延迟或显存），用于判断质量提升是否值得额外的 token、延迟和显存。

## 3. 实验要验证的结论

实验分别检验两个结论：

> 08A：在相同训练 token、相同 backbone 和相同 Thinker 位置编码下，保宽高比的
> 动态切片能否提高 OCR、文档和小目标能力。
>
> 08B：锁定同一个 dynamic frontend、同一个起点 checkpoint 和同一批 64 个
> resampler 输出后，M-RoPE 能否比 1D RoPE 更好地利用二维位置。

以下结果不能支持该结论：

- 08A 的动态方案只在总体均分上升，但同 LLM token budget 下不优于固定分辨率；
- 08B 的 M-RoPE 不优于同一 dynamic frontend 上的 1D RoPE；
- 08B 打乱二维坐标后结果几乎不变，说明模型没有使用 M-RoPE；
- 把 08A 与 08B 的增益相加后，反过来宣称它们来自某一个单独改动；
- 图像能力提升伴随文本、音频回归超出预设阈值；
- 多图问题中模型无法区分图片身份。

这张"不算数清单"提前写好，是为了防止事后给自己找台阶。负结果照样是合法交付（见第 15 节）。

## 4. 进入条件与独立起点

开始前必须具备：

- 已通过下述 preflight、可加载且哈希一致的 MiniMind-O Thinker checkpoint；
- 固定的图像问答评测集和 50 个逐 case golden cases（冻结测试样例，忘了回[第 01 课](01_baseline_reproduction.md)）；
- 能分别记录每个样本的 encoder patch token、LLM visual token、prefill 时间和
  峰值显存；
- 能冻结 vision encoder、固定 connector 和 backbone；
- 数据 manifest 中已经记录来源与 license 状态。

如果实验 02 尚未选定 connector，统一使用原始 `MMVisionProjector`。

不要临时换 Q-Former（用一组可学习 query 做 cross-attention 压缩的另一类 connector），否则无法判断收益来自动态分辨率还是 connector。本课的 resampler 已经是 query 式压缩器，connector 再变，两个变量就又搅在一起了。

### 4.1 不可变模型锁

正式实验不接受 `baseline-v1`、`latest`、本地目录名或浮动分支作为模型身份。只允许 preflight 解析一次，随后 08A、08B 的所有臂都读取同一个只读 manifest：

```yaml
# manifests/base_model.lock.json 的人类可读表示
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

preflight 必须完成并原子写入 `manifests/base_model.lock.json`：

1. 以完整 40 位 commit checkout 代码，以完整 Hub revision 下载 checkpoint/tokenizer；禁止训练阶段联网重新解析。
2. 对本地 `pytorch_model.bin` 重新计算 SHA-256，并同时校验 Hub 返回的 repo revision；任一不一致立即终止。
3. 对 SigLIP2、processor 和其他外部资产记录 `provider/repo_id/resolved_revision/files[{path,sha256}]`，再计算排序后的 file-tree SHA-256。
4. 保存 clean/dirty worktree、依赖 lock hash、配置 hash 和生成时间；manifest 写成只读文件。
5. preflight 计算 `base_model.lock.json` 自身 SHA-256。08A 的三个 arm config 写入
   `manifests/run_08a.lock.json`；08B 另把锁定的 dynamic frontend、共同起点
   checkpoint 和两个 arm config 写入 `manifests/run_08b.lock.json`。训练入口只
   接受对应的 run lock，不再各自选择 checkpoint。

若未来要升级官方版本，必须生成一个新的 lock manifest 和新的实验 ID，不能原地改写本课结果。

## 5. 完成本课需要掌握的操作

完成后应能独立完成以下工作：

1. 说明固定缩放对 OCR、长图和极端宽高比图像的影响；
2. 实现由 pixel budget 驱动的动态 tile 选择；
3. 给多图、tile、二维 patch 建立无歧义 position schema；
4. 实现文本 1D、图像 2D 的 M-RoPE；
5. 在动态 batch 中正确生成 attention mask 和 position ids；
6. 先隔离动态切片，再隔离 M-RoPE，并为两项实验分别做同预算对照；
7. 从逐 case 结果区分"分辨率不足"和"推理失败"。

## 6. Patch、预算与二维位置

本课的原理层就四件事：图怎么切（6.1）、切多细谁说了算（6.2）、位置怎么告诉 Thinker（6.3）、多张图怎么不串号（6.4）。每一件都按"为什么、怎么做、怎么验证"走。

### 6.1 Patch 与 tile

encoder 的"视网膜"是死的。SigLIP2-P32 表示 patch stride 为 32：每 32×32 像素切一块。若单个 tile 为 `256×256`，空间网格为 `8×8`，即 64 个 patch token。把一张 2480×3508 的发票塞进 256×256，一个汉字摊不到几个像素，读小字等于让人隔着毛玻璃认账单。

把原图直接压到正方形会：

- 改变物体形状；
- 抹掉小字；
- 让长文档高度信息压缩；
- 让 panoramic 图像两端细节消失。

动态切片不缩放整图硬凑正方形：先根据原图宽高比选择 tile 网格（比如竖长的文档切成 2 列 4 行），再在 pixel/token budget 内限制 tile 数量。每张 tile 都是 encoder 熟悉的 256×256，细节按原始比例保留。

对同一张图输出缩放后的尺寸、网格、每个 tile 的坐标和最终 token 数，确认没有在切片前把原图压成正方形。常见错误是：预处理库的默认 resize 悄悄把图先压方了，后面切得再漂亮也是白切——诊断表（第 16 节）第一行就是它。

### 6.2 Pixel budget 与 token budget

tile 切得越多看得越细，token、显存、prefill 时间也一起涨。没有预算约束的动态切片会被一张超长网页图直接打爆显存，所以"看多细"必须由预算说了算。

视觉 token 数的近似关系：

$$
N_{\text{visual}}
\approx
N_{\text{tiles}}
\frac{H_{\text{tile}}}{H_{\text{patch}}}
\frac{W_{\text{tile}}}{W_{\text{patch}}}
+N_{\text{global}}
$$

其中 $N_{\text{tiles}}$ 是 tile 数，$H_{\text{tile}}/H_{\text{patch}}$ 与 $W_{\text{tile}}/W_{\text{patch}}$ 是每张 tile 纵横两个方向的 patch 数（本课 256/32=8），$N_{\text{global}}$ 是 global thumbnail 贡献的 64 个 patch。

**记账纪律。** 训练配置必须同时保存：

- 输入像素数；
- tile 数；
- encoder 输出 token 数；
- connector 输出 token 数；
- 注入 LLM 的最终 token 数。

报告必须同时给出这五个量。"448 分辨率"无法说明宽高比、tile 数和进入 LLM 的 token 数，不能作为完整配置——论文里只写一个分辨率数字的配置，你复现时会发现根本不唯一。

### 6.3 2D RoPE 与 M-RoPE

文本只需要序列轴 `t`：一条街上报门牌号就够了。图像至少需要 `x, y`，多图还需要 `image_id` 或 segment embedding（给每段 token 加一个"我属于第几张图"的可学习标记向量）。把 8×8 的格子拉直成一条线后，"正上方"变成"往前数 8 个"，这种关系模型只能从数据里硬猜；M-RoPE 直接把二维坐标编进位置相位，省掉这层猜。

本课中的 `query slot` 指 resampler 的一个输出位置。每张图固定有 64 个 query slots，按 `8×8` 网格覆盖原图。每个 slot 保存 `image_id`、`query_row`、`query_column`、`normalized_y` 和 `normalized_x`。

tile 内 patch 的坐标先换算回原图坐标，再送入 resampler。resampler 的 key/value 加固定二维正弦位置编码，因此它知道某个 patch 来自原图的哪个位置。resampler 输出后，Thinker 不再看到原始 patch 坐标；它看到的是 64 个 query slots 的坐标。08B-B 的 M-RoPE 必须使用这些输出坐标，不能继续传已经被压缩掉的 patch 坐标——patch 坐标在 resampler 之后就没有对应的 token 了，传下去只会张冠李戴。

**实现。** 一种教学实现是把 head dimension 划为三组：文本 token 使用 `(t, t, t)`，图像 token 使用 `(global_t, y, x)`。M-RoPE 只改变 token 的位置相位，不改变 token 的内容和排列顺序。

它只进入 08B；08A 三臂都继续使用相同的 1D RoPE。08B 单测应固定输入 embedding，只替换位置坐标，确认输出变化来自位置编码，而非 embedding 或顺序的偷换。

### 6.4 多图身份

两张图具有相同局部 `(x, y)`：两栋楼都有"三层东侧那间"，只报房间号快递会送错楼。

因此必须额外保留：

- `image_id`；
- `tile_id`；
- 或明确的 `<image_start_n> ... <image_end_n>` segment。

**判断与验证。** 拼接顺序不会自动形成稳定的图片身份——模型可能碰巧用顺序区分了两张图，也可能根本没区分。必须选择一种显式表示，并通过交换 `image_id` 的负对照确认模型确实使用了该信息：交换后答案跟着变，才说明身份信息在起作用。

## 7. MiniMind-O 当前机制与代码落点

动刀前先摸清现状。以官方仓库当前结构为准：

- `model/model_omni.py`
  - `OmniConfig.image_token_len` 默认 64；
  - `MMVisionProjector` 是 `LayerNorm → Linear → GELU → Linear`；
  - `load_vision()` 加载 SigLIP vision tower；
  - `get_image_embeddings()` / `encode_image_inputs()` 编码图像；
  - `count_vision_proj()` 把视觉表示写入 `<|image_pad|>` 位置；
  - `forward()` 已有有限的多图 tensor 形状兼容，但位置仍走统一序列。
- `dataset/omni_dataset.py`
  - 构造固定长度的 64 个 `<|image_pad|>`；
  - `load_image_inputs()` 只处理单张 PIL 图像；
  - `__getitem__()` 默认只读取 `image_bytes[0]`；
  - collate 前没有 tile metadata 和二维 position。
- `model/model_minimind.py`
  - `precompute_freqs_cis()` 只生成一维 RoPE；
  - `Attention.forward()` 接收统一 `(cos, sin)`；
  - `MiniMindModel.forward()` 根据序列位置切一段 RoPE。
- `trainer/train_sft_omni.py`
  - `omni_collate_fn()` 只拼接已有 pixel tensor；
  - `--mode vision_proj` 可作为第一阶段冻结训练入口。
- `eval_omni.py`、`webui/web_demo.py`
  - 需要同步支持动态预处理和多图输入，不能只改训练。

这张清单也解释了"固定 64"假设埋得有多深：数据侧写死 64 个占位符，模型侧写死 `image_token_len`，位置侧只有一维 RoPE。先为这些点写 smoke test（冒烟测试：最基本的"能跑通、形状对"检查），再开始结构改造；否则改到一半才发现推理路径还有第四处写死。

## 8. 图像处理、打包和位置编码接口

实现分为四个接口：

| 接口 | 输入 | 输出或职责 |
|---|---|---|
| `DynamicImageProcessor.encode_layout` | image、pixel budget | tiles、thumbnail、`patch_xy`、`layout_meta` |
| `PositionAwareVisualResampler.resample` | features、`patch_xy`、`image_id` | `visual_embeds[64,D]`、`query_xy[64,2]`、`segment_ids[64]` |
| `VisionTokenPacker.pack` | text/visual embeds、query 坐标、segment | packed embeds、模态位置和 segment |
| `MultimodalRotaryEmbedding` | 文本 1D 位置或图像的 global/y/x | attention 使用的 cos、sin |

四个接口各管一段，边界清楚才好单测：切片归 processor，压缩归 resampler，排队归 packer，相位归 rotary。

多图输入对每张图分别运行一次同一个 resampler，再按消息中的图片顺序拼接。因此两张图产生 128 个 LLM visual tokens，但每张图仍各有 64 个；`image_id` 和 segment 边界不能复用。08A、08B 的所有臂读取同一条样本时图片数相同，所以 LLM token budget 仍然可比。

`layout_meta` 至少包含：

```yaml
image_id: 0
original_hw: [1440, 2560]
tile_grid: [2, 4]
tile_hw: [256, 256]
thumbnail: true
patch_grid_per_tile: [8, 8]
valid_patch_mask: ...
```

保持 connector 唯一且固定。

## 9. 数据 recipe

### 9.1 统一数据字段

每行 JSONL 或 Parquet：

```yaml
id: docvqa_000001
messages:
  - role: user
    content:
      - type: image
        image_ref: images/a.png
      - type: text
        text: "发票总金额是多少？"
  - role: assistant
    content:
      - type: text
        text: "128.50 元"
images:
  - image_ref: images/a.png
    sha256: "..."
    width: 2480
    height: 3508
source: docvqa
license:
  annotation: source-declared
  media: source-dependent
  redistributable: false
split: train
task: document_qa
answer_type: exact
```

必须保留媒体与标注的两层 license：图片本身和问答标注可能来自不同授权方。

网页可下载不等于允许再分发。

### 9.2 数据组成

- 30% 自建合成文档/OCR：自己生成，建议 CC BY 4.0；
- 25% 自然图 VQA；
- 20% 文档、表格、票据；
- 15% 小目标、计数和空间关系；
- 10% 多图比较与跨图指代。

配比向 OCR/文档倾斜是有意的：这正是固定分辨率最疼、动态切片最该赢的地方；自然图占比保底，用来监控"专项涨、通用跌"。

### 9.3 切分纪律

- 按原始 document/image identity 去重后切分；
- 同一模板不同数字不能跨 train/test；
- 合成字体、背景、版式按 family 切分；
- 多图组合中的任一图出现在训练集，则组合不得进入测试集；
- golden 50 cases 永不参与训练和调参。

前三条都在防同一件事：模型背下版式而非学会看图。合成数据尤其危险——同一套字体加背景生成一万张，train/test 随机切等于开卷考试。

### 9.4 三档规模

| 档位 | 样本数 | 用途 | 预计训练 |
|---|---:|---|---|
| pilot | 10k | 验证 shape、mask、M-RoPE | 1–3k steps |
| standard | 150k | 完整 08A 与 08B 比较 | 每个子实验 10–30k steps |
| full | 1M+ | 验证规模趋势 | 1 epoch 或定 token |

每档都按 token 数而非 epoch 做公平预算：动态臂每样本 token 数不同，"跑一个 epoch"对各臂不是同一个算力量。

## 10. 实施步骤

### 步骤 1：冻结基线测量

先量出旧眼睛的底细，改完才有对照。记录固定 256 输入下每类任务的：

- 准确率；
- OCR 字符召回；
- 视觉 token；
- prefill p50/p95；
- 峰值 HBM（显卡上的高带宽显存，下同）；
- 文本与音频回归。

### 步骤 2：实现动态 tile planner

枚举允许网格，例如 `(1,1)`、`(1,2)`、`(2,1)`、`(2,2)`、`(1,3)`、
`(3,1)`、`(2,3)`、`(3,2)` 和 `(3,3)`。

在 `max_tiles` 和 aspect-ratio error（候选网格宽高比与原图宽高比的偏差）下选择最优网格。

训练时对 pixel budget 做 sampling（随机采不同预算，让模型见过各种切法），评测时固定确定性策略——评测再随机，数字就没法复比。

### 步骤 3：加入 global thumbnail

global thumbnail 保留整张图的布局，tile 保留局部细节；只有 tile 没有 thumbnail，模型看得清每块砖却拼不出楼长什么样。为 thumbnail 和每个 tile 分配不同的来源标记，并检查它们的 `image_id`、`tile_id` 和二维坐标没有冲突。

### 步骤 4：改造数据与 collate

collate 返回 `pixel_values`、`tile_valid_mask`、`image_ids`、`tile_ids`、
`patch_xy`、`encoder_patch_token_count` 和 `llm_visual_token_count`。

动态 batch 先按总 encoder patch token 分桶（bucketing：把 token 数相近的样本编进同一个 batch，一张 9-tile 长图和一堆单 tile 小图混在一起，padding 全浪费在小图上），避免 vision encoder 侧显存长尾；同时记录每个样本的图片数，因为 LLM 侧 token 数为 `64 × 图片数`。

### 步骤 5：实现 M-RoPE，但先不要接入 08A

先写纯函数单测：

- 文本位置与旧实现一致；
- 相同图像布局产生完全相同的 64 个 query 坐标；
- 每个 resampler 输出与一个 query 坐标一一对应；
- padding token 不获得有效二维位置；
- 多图之间 image segment 不混淆；
- 保存/恢复 checkpoint 后 position 相同。

主负对照固定 visual embeddings 和 token 顺序，只打乱 08B M-RoPE 臂 resampler 输出的
`query_xy`。如果改的是 resampler 输入 `patch_xy`，测到的是 resampler 是否
使用 patch 位置，不是 Thinker 是否使用 M-RoPE；两项诊断要分开报告。打乱错了对象，等于给错误的结论盖了章。

### 步骤 6：改造 token 注入

删除"vision encoder 恰好输出 64 个 patch"这一固定假设，但保留"每张图经
resampler 后恰好输出 64 个 LLM visual tokens"的正式实验契约。前者是历史写死，后者是本课公平比较的地基，两者别混。

packer 为每张图生成 64 个 `<|image_pad|>`，或直接传
`inputs_embeds + position_ids`。注入前必须逐图断言 resampler 输出数、
`query_xy` 数和 placeholder 数都等于 64。

对超 budget 样本要显式拒绝或降采样，不得静默截断答案。

### 步骤 7：先运行 08A，再锁定 08B 起点

08A 的 fixed-256、fixed-448 和 dynamic 三臂都使用 1D RoPE。第一阶段只训练
projector 与 resampler；第二阶段再给三臂同时解冻相同范围的 Thinker LoRA（低秩适配：不动原权重，加一对小矩阵学增量，省显存也便于对齐可训练范围）。
完成后，把 08A dynamic 臂的 processor、tile planner、resampler、checkpoint 和
逐 tensor SHA-256 写入 `dynamic_frontend.lock.json`。

08B 的两个臂都从这一个 checkpoint 开始，读取同一份 dynamic feature cache 和
同一批 64 个 query slots。控制臂继续使用 1D RoPE，实验臂只切换到 M-RoPE；
两臂再训练相同 token，且可训练参数范围、optimizer、样本顺序和 seed 完全相同。
08B 的 processor、vision encoder、connector 和 resampler 全部冻结，不得重新
选择 tile、pixel budget 或 resampler。08B 里任何一处前端差异，都会让"位置编码带来的增益"这句话失去依据。

### 步骤 8：同步推理路径

训练、`eval_omni.py`、WebUI 必须调用同一个 processor。训练和推理各养一套预处理，是"训练分数高、上线就翻车"的经典来源。

trace 按 `original_hw`、`tile_grid`、`encoder_tokens`、`llm_tokens` 的顺序记录，
使每个样本的分辨率决策和 token 成本都能追溯。

## 11. 对照实验组

所有臂都使用同一个 **position-aware 64-query cross-attention resampler**。
它把每张图的 encoder patch 序列压到精确 `Q=64` 个 LLM visual tokens。每个
query 对应原图上的一个 `8×8` 网格位置；patch 原图坐标进入 resampler 的
key/value 位置编码，query 坐标随输出进入 packer。

### 11.1 08A：只比较图像前端

| 臂 | 唯一输入算法 | Thinker 位置编码 | 每张图进入 LLM 的 token |
|---|---|---|---:|
| 08A-A | fixed 256，`8×8` patch grid | 1D RoPE | 64-query resampler → 64 |
| 08A-B | fixed 448，`14×14` patch grid，vision position interpolation | 1D RoPE | 同一 resampler → 64 |
| 08A-C | aspect-ratio dynamic 256 tiles + 256 thumbnail | 1D RoPE | 同一 resampler → 64 |

08A-B 的 vision position interpolation（位置插值）补一句：SigLIP2 的位置编码只学过 8×8 网格，448 输入产生 14×14 网格，多出来的位置没有现成编码，只能把 8×8 的编码按比例拉伸到 14×14 复用。这是"只提分辨率、不切片"路线的标准做法，也是它的天花板所在。

resampler 的结构、初始化 seed、参数量、训练 token 和初始权重逐字节相同；
08A-A 即使输入恰好为 64 patch tokens，也必须经过它。原始
`MMVisionProjector` 直通结果只作冻结历史 reference，不是第四个训练臂。

08A 的公平控制：

- 相同 vision encoder、connector、resampler 与 1D RoPE；
- 相同训练样本、样本顺序和总训练 token；
- 每张图进入 LLM 的有效视觉 token 精确等于 64；
- 相同 optimizer、warmup、seed、生成参数；
- 至少 3 个 seed，pilot 可先 1 个。

08A 的额外诊断只改变输入，不参与训练臂计数：随机选相同数量的 tile，以及交换
多图 `image_id`。

### 11.2 08B：锁定 dynamic frontend 后比较位置编码

| 臂 | 图像前端与输入 embedding | Thinker 位置编码 | 训练起点 |
|---|---|---|---|
| 08B-A | 同一锁定 dynamic frontend、同一 64-query 输出 | 1D RoPE | 同一个 08A-C checkpoint |
| 08B-B | 同一锁定 dynamic frontend、同一 64-query 输出 | M-RoPE | 同一个 08A-C checkpoint |

08B 的 `dynamic_frontend.lock.json` 必须锁定 processor、tile layout、feature
cache、resampler state 和 query 坐标。两个臂逐样本读取完全相同的
`visual_embeds/query_xy/segment_ids`；唯一允许的 config diff 是 Thinker 的位置
编码。两臂使用相同训练 token、LoRA 范围、optimizer、样本顺序和 seed。

08B 的位置负对照固定 visual embeddings 与 token 顺序，只打乱
`query_xy`。另一个诊断固定 query 与 M-RoPE，只打乱 resampler 输入 patch
坐标；前者检查 Thinker M-RoPE，后者检查 resampler 的 patch 位置，不能混写。

## 12. 配置示例

```yaml
experiment: exp08_dynamic_vision
model_lock: manifests/base_model.lock.json
run_locks:
  exp08a: manifests/run_08a.lock.json
  exp08b: manifests/run_08b.lock.json
vision:
  encoder: siglip2-base-p32-256
  frozen: true
  tile_size: 256
  patch_size: 32
  max_tiles: by_arm
  add_thumbnail: by_arm
  llm_visual_tokens_per_image: 64
  aspect_ratio_preserving: true
resampler:
  type: position_aware_cross_attention
  queries: 64
  query_layout: [8, 8]
  layers: 1
  heads: 8
  patch_position_encoding: fixed_2d_sincos
  output_position: query_layout_xy
  initial_state_lock: manifests/resampler_init.lock.json
position:
  axes: [temporal, height, width]
  visual_coordinate_source: resampler_query_xy
  image_segment_embedding: true
subexperiments:
  exp08a_frontend:
    arms:
      A: {input: fixed_256, max_tiles: 1, add_thumbnail: false, position: rope_1d}
      B: {input: fixed_448, max_tiles: 1, add_thumbnail: false, position: rope_1d}
      C: {input: dynamic_tiles, max_tiles: 6, add_thumbnail: true, position: rope_1d}
  exp08b_position:
    frontend_lock: manifests/dynamic_frontend.lock.json
    init_checkpoint: checkpoints/exp08a_dynamic.safetensors
    arms:
      A: {input: locked_dynamic_cache, position: rope_1d}
      B: {input: locked_dynamic_cache, position: mrope}
connector:
  type: original_mlp
train:
  train_tokens: 80_000_000
  precision: bf16
  seeds: [17, 23, 41]
  freeze_vision: true
  thinker_lora_rank: 16
eval:
  llm_visual_tokens_per_image: 64
  encoder_pixel_budgets: [65536, 200704, 458752]
```

preflight 在 `run_08a.lock.json` 中写入 model lock、08A 三个 arm config 和
resampler 初始 state-dict 的实际 SHA-256。08A 完成后再生成
`dynamic_frontend.lock.json`；`run_08b.lock.json` 同时锁定它、共同起点
checkpoint 和 08B 两个 arm config。训练入口逐项重算，不匹配即终止。
Pixel-budget sweep 只属于 08A；08B 不再扫描 pixel budget，LLM 侧始终为每张图
64 token。

参考伪代码：

```python
tiles, meta = planner(image, pixel_budget)
patch_features = connector(vision_encoder(tiles))
patch_xy = map_tile_patches_to_original_image(meta)
visual, query_xy, segments = resampler(
    patch_features,
    patch_xy=patch_xy,
    query_layout=(8, 8),
    image_id=meta.image_id,
)
assert len(visual) == len(query_xy) == 64
hidden, pos3d = packer(text_embeds, visual, query_xy, segments)
cos, sin = mrope(pos3d) if experiment == "08B" and arm == "B" else rope_1d(hidden)
logits = thinker(hidden, rotary=(cos, sin), segments=segments)
```

## 13. 训练预算与 8 卡建议

### Pilot

- 1×24GB；
- backbone 冻结；
- 10k 样本，1–3k steps；
- `max_tiles=2`；
- 目标是跑通 shape、loss、保存/恢复和负对照。

### Standard

- 2–4×24GB；
- 150k 样本；
- 80M–200M 有效训练 token；
- gradient checkpointing（省显存的技巧：前向少存中间结果，反向时重算）；
- 按 encoder patch token bucket；
- 08A 三臂与 08B 两臂各 3 seed。

### 8 卡

- 不要为了"用满 8 卡"扩大 batch；
- 固定 global token batch，再调整 micro-batch；
- 8 卡优先并行不同实验臂和 seed；
- 如果单次训练跨卡，报告 DDP 通信占比；
- 无 NVLink 时，冻结 vision encoder 并离线缓存特征可更高效；
- 缓存必须包含 processor/version/layout hash，防止错配。

最后一条再啰嗦一句：特征缓存是本课最值钱的加速，也是最阴的坑——processor 改了一行、缓存没重建，你会训练一个星期都不知道喂的是旧切法的特征。

## 14. 指标与测量方法

### 能力指标

- TextVQA/自建 OCR：ANLS、exact match、字符召回；
- 文档 QA：ANLS、字段级 F1；
- 计数：exact accuracy；
- 空间关系：分类准确率；
- 多图：跨图指代准确率；
- 自然图：固定 VQA 小切片；
- 文本、音频 golden regression。

ANLS 用来给"基本正确但有少量字符错误"的答案计部分分。对预测字符串 $a$ 和
参考答案 $g$，先算 Levenshtein 编辑距离 $d(a,g)$（把一个串改成另一个串最少要增删改几个字符），再得到归一化相似度：

$$
s(a,g)=
\max\left(0,\ 1-\frac{d(a,g)}{\max(|a|,|g|)}\right).
$$

两者都为空时单独记 $s=1$；只有一个为空时记 $s=0$。一条样本有多个参考答案时，
取最大的 $s(a,g)$。本课固定阈值 $\tau_{\text{ANLS}}=0.5$：相似度不低于 0.5
时保留该相似度，否则记 0；最后对所有样本取平均。Unicode 归一化、大小写和空白
处理规则也要在第一次看 test 前写入评测配置，之后不能更改——量尺一换，数字变化就说不清来源，这条纪律第 01 课讲 WER 时立过。

### 系统指标

- 每样本 encoder token 与 LLM visual token；
- vision encode time；
- prefill p50/p95；
- end-to-end latency；
- tokens/s；
- peak allocated/reserved HBM；
- padding ratio；
- OOM 与超 budget 比例。

### 测量纪律

- 预热 20 次，正式测 100 次；
- 固定 batch size 1 和吞吐 batch 两组；
- 同 GPU、dtype、软件版本；
- latency 同步 CUDA；
- 能力按 token-budget bucket 分层；
- 报均值、标准差和 95% bootstrap CI（自助置信区间：反复重采样估计指标波动范围）。

## 15. 验收条件

先判断实验是否有效，再判断方法是否胜出。两项实验分别验收，不能用一个实验的
结果替另一个补分。

### 15.1 实验有效性

08A 的有效性要求：

1. 三臂都使用 1D RoPE，且每张图进入 LLM 的 token 都精确为 64；
2. 三臂按锁定协议完成质量、encoder patch token、prefill latency、HBM 和三 seed
   配对置信区间的报告；
3. 多图身份测试、质量—token—延迟 Pareto 数据和逐 case 失败分析完整。

08B 的有效性要求：

1. 两臂的 dynamic frontend lock、输入 embedding、query 坐标、checkpoint 和训练
   数据 hash 一致；
2. 按预注册方案完成 `query_xy` 打乱实验，报告打乱前后的逐 case 差值和 95% CI；
3. 完整报告空间/OCR、文本与音频 golden、多图身份测试，以及逐 case 失败分析。

只要以上控制和报告完整，即使没有观察到增益，实验仍然有效，本课也算完成。不能
因为结果为负而修改阈值、删除 seed 或更换 test 集。

### 15.2 方法胜出条件

08A 只有同时满足以下条件，才能声称动态分辨率前端优于固定 256：

1. 08A-C 相对 08A-A 的 OCR/文档 macro accuracy 至少提高 5 个百分点，三 seed
   配对 95% CI 下界高于 0；
2. 08A-C 的自然图能力不低于 08A-A 的 98%；
3. 08A-C 与 fixed-448 在质量、encoder patch token、prefill latency 和 HBM 上
   形成更好的 Pareto，只提高一个均分不算数。

08B 只有同时满足以下条件，才能声称 M-RoPE 带来额外收益：

1. 08B-B 相对 08B-A 的空间/OCR macro accuracy 至少提高 2 个百分点，且三 seed
   配对 95% CI 下界高于 0；
2. 打乱 08B-B 的 `query_xy` 后，空间/OCR macro accuracy 至少下降 2 个百分点，
   且"未打乱减打乱"的逐 case 配对 95% CI 下界高于 0；
3. 两臂文本、音频 golden 综合回退都不超过 2%。

三项差值都使用同一批 case 和 seed 做 10,000 次两层配对 bootstrap：先重采样 seed，
再重采样 case。阈值、case 列表和统计脚本 hash 在打开 test 前写入
`stats_plan.yaml`。

若方法没有胜出，需要明确它在哪些任务、token budget 或延迟范围内失效，并把结果
记录为有效负结果；不能写成支持 H1。

## 16. 失败诊断表

按"先查便宜的"排序使用：先看预处理和记账，再怀疑模型。

| 现象 | 优先检查 | 修复方向 |
|---|---|---|
| OCR 不提升 | 原图是否先被缩小 | tile 前不要全局 resize |
| token 数超过预算 | planner 是否只看宽高比 | 加 pixel/token hard cap |
| M-RoPE 无效果 | position 是否传入每层 | trace 前三层 cos/sin |
| 多图身份混淆 | image segment 丢失 | 加 image id 与边界 token |
| loss 突升 | 新视觉占位数与 embedding 数不一致 | assert 一一对应 |
| padding 比例高 | batch 按样本数分组 | 改为按总 encoder patch token bucket |
| fixed-448 反而更好 | 动态 tile 缺整体上下文 | 检查 thumbnail 与重叠 |
| 文本能力下降 | 图像样本比例过高 | 加 text replay，固定 token mix |
| 训练快、推理错 | processor 分叉 | 训练评测共用同一实现 |
| OOM 随机发生 | tile 数长尾 | 保存每批 token histogram |

## 17. 逐 case 要求

至少维护以下 50 个 case：

- 10 个小字/OCR；
- 8 个长文档；
- 8 个极端宽高比；
- 8 个小目标/计数；
- 8 个空间关系；
- 8 个多图比较。

每个 case 页面必须展示：

- 原图、tile 布局和保留的 patch/token；
- 问题与标准答案；
- 08A-A/B/C 和 08B-A/B 的输出；
- 置信度或 token log-prob；
- 延迟、显存和错误分类。

错误标签至少分为：

- `resolution_miss`；
- `layout_position_miss`；
- `ocr_decode_miss`；
- `cross_image_binding_miss`；
- `reasoning_miss`；
- `answer_format_miss`。

不得只展示成功案例。逐 case 页面的价值恰恰在失败样例：`resolution_miss` 多，说明预算不够；`layout_position_miss` 多，该回头查位置编码；`reasoning_miss` 多，那是分辨率救不了的，别再加 tile 了。

## 18. 交付物

```text
exp08/
  configs/08a/{fixed256,fixed448,dynamic}.yaml
  configs/08b/{rope1d,mrope}.yaml
  manifests/base_model.lock.json
  manifests/resampler_init.lock.json
  manifests/dynamic_frontend.lock.json
  manifests/{run_08a,run_08b}.lock.json
  manifests/{08a,08b}_arm_config_hashes.json
  data/manifest.jsonl
  data/license_report.md
  checkpoints/
  metrics/aggregate.json
  metrics/per_case.jsonl
  traces/layout_trace.jsonl
  plots/quality_token_latency.png
  cases/index.md
  report.md
```

`report.md` 必须回答：

- 08A 中动态分辨率相对 fixed-256/fixed-448 的收益和代价；
- 08B 中 M-RoPE 相对同一 dynamic frontend 上 1D RoPE 的增量；
- 哪类图片最先达到 token budget；
- 最终推荐哪一个默认 pixel budget。

## 19. 复现清单

- [ ] preflight 已冻结 MiniMind-O Git commit、Hub revision 与 checkpoint SHA-256；
- [ ] 08A 三臂引用同一只读 `base_model.lock.json`，且 arm config 中已展开该 manifest 的具体 SHA-256；
- [ ] 记录 vision encoder 精确 revision、逐文件 SHA-256 与 file-tree SHA-256；
- [ ] 记录 processor 参数与 hash；
- [ ] 固定 08A/08B 的 train/dev/test split；
- [ ] 固定总训练 token；
- [ ] 08A 固定 connector、backbone 与 1D RoPE；
- [ ] 08A 三臂 resampler 初始 state-dict SHA-256 相同；
- [ ] 08A 三臂每张图进入 LLM 的视觉 token 精确为 64，同一样本图片数一致；
- [ ] 08A dynamic frontend 与 checkpoint 已单独锁定；
- [ ] 08B 两臂逐样本读取相同 visual embeddings、query 坐标和 segment ids；
- [ ] 08B 唯一 config diff 是 1D RoPE 与 M-RoPE；
- [ ] 保存 tile planner 版本；
- [ ] 保存 seed 和 CUDA/PyTorch 版本；
- [ ] 运行二维位置负对照；
- [ ] 运行多图身份负对照；
- [ ] 运行文本、语音回归；
- [ ] 输出逐 case；
- [ ] 生成 Pareto 图；
- [ ] 从 checkpoint 恢复并复测。

## 20. 前沿对照与改造方向

本课的两条技术路线在前沿系统里都有原型。[Qwen2-VL](https://arxiv.org/abs/2409.12191) 把动态分辨率做成原生：图片按接近原始的分辨率直接进 ViT，patch token 数随图变化，进入 LLM 的视觉 token 也是变长的；位置用 M-RoPE 表示，temporal、height、width 三个轴分别承载时序、行、列——本课 08B 的教学实现就是它的缩小版，第 09 课还会把 temporal 轴真正用起来。[LLaVA-UHD](https://arxiv.org/abs/2403.11703) 与 08A-C 同一路线：按宽高比把图模块化成变尺寸切片，配 overview 图保布局，再用压缩层控制进入 LLM 的 token 数。[MiniCPM-V 2.6](https://arxiv.org/abs/2408.01800) 把这条切片加压缩的路线做到端侧可跑：高分辨率图切片后，每片经 query 式压缩模块变成少量 LLM token，思路与本课 resampler 相同。再往 omni 方向看一步，[Qwen2.5-Omni](https://arxiv.org/abs/2503.20215)（第 01 课引过）把 M-RoPE 的时间轴与真实时间对齐（TMRoPE），让视频帧和音频块共享同一条时间轴——那是下一课的正题。

规模问题：前沿的 vision encoder 本身在多分辨率数据上训过，原生支持变尺寸输入；我们的 SigLIP2 冻结且预训练分辨率固定为 256，只能靠切 tile 和位置插值绕过去。训练数据 150k 样本对前沿的海量 OCR/文档语料，也是钱能缩小的差距。机制问题（本课能解决的）：固定 64 token 是我们为公平比较自定的实验契约，前沿走变长 token；而"切片保宽高比、显式二维位置、多图显式身份、encoder 与 LLM token 分开记账"这四件事，做完本课你的做法与前沿同构，差的只是尺寸。



1. **变长 visual token 臂。** 打破 64 契约，学 Qwen2-VL 让进入 Thinker 的视觉 token 数随图变化：把 `PositionAwareVisualResampler` 换成相邻 patch 合并式压缩，packer 取消步骤 6 的 64 断言，`query_xy` 改为合并后 token 的原图坐标。必须另开实验 ID 和新的 run lock——64 契约是 08A/08B 公平性的地基，不能原地改。预算：standard 档 150k 样本、80M 训练 token、2–4×24GB，与 08A 单臂相当。预期：OCR/文档桶继续上涨，prefill 延迟与显存同步上涨，在 Pareto 图上检查它是否优于 08A-C。失败判定：质量不涨只有 token 涨，或文本/音频 golden 回退超过 15.2 节的 2% 阈值。
2. **第四轴 RoPE 多图身份。**（对应扩展题 3）把 image_id 从 segment embedding 挪进 RoPE 的第四个轴：`MultimodalRotaryEmbedding` 加一轴，packer 把 `image_id` 写进 pos 张量，与 segment embedding 臂做对照。预算：pilot 档 10k 样本、1–3k steps，1×24GB 即可。预期：多图跨图指代准确率不低于 segment embedding 臂，交换 `image_id` 负对照的答案变化幅度保持或扩大。失败判定：交换 `image_id` 后输出几乎不变，说明第四轴没被使用，退回检查该轴的频率分配。



- LLaVA-UHD 的"整图压成固定正方形损伤高分辨率任务"这一方向性结论，08A-A 对 08A-C 就是缩小版复现。预期同方向：OCR/文档/极端宽高比桶差距最大，自然图桶差距最小；若自然图桶差距反而最大，先查 9.2 的数据配比是否失衡。
- Qwen2-VL 的"二维位置表示优于一维展开"结论，对应 08B-A 对 08B-B 加 `query_xy` 打乱负对照。预期能复现方向，但幅度受小尺寸 Thinker 和 64 token 摘要瓶颈限制；若 CI 下界不过 0，按 15.2 记有效负结果，并注明与原论文设置的规模差异，不算复现失败。

## 21. 论文精读

只读原始论文和官方实现：

1. [Qwen2-VL](https://arxiv.org/abs/2409.12191)
   - 精读：Naive Dynamic Resolution、M-RoPE、训练数据部分。
   - 阅读检查：说明动态 token 如何改变 batch 构造和位置表示——对照你在步骤 4 写的分桶 collate，指出它要解决的是同一个 padding 长尾问题。
   - 阅读检查：列出 M-RoPE 三个轴各自承载的信息，与 6.3 节 `(global_t, y, x)` 逐轴对应；再回答：本课的静态图为什么 temporal 轴退化成常数，第 09 课它会变成什么。
2. [LLaVA-UHD](https://arxiv.org/abs/2403.11703)
   - 精读：image modularization、compression、spatial schema。
   - 阅读检查：说明 overview image 与局部 tile 分别保留哪些信息；对照步骤 3，回答如果删掉 global thumbnail，第 16 节诊断表哪一行症状会出现。
3. [MiniCPM-V 2.6](https://arxiv.org/abs/2408.01800)
   - 精读：高分辨率与 token 压缩设计。
   - 阅读检查：写出高像素输入被压缩为少量 LLM token 的具体位置；对照本课 resampler，指出两者都在哪一层把"encoder 看多少"与"LLM 收多少"解耦。
4. [Flamingo](https://arxiv.org/abs/2204.14198)
   - 精读：多图交错输入与 media conditioning。
   - 阅读检查：标出多图身份在输入和模型中的保留位置；对照 6.4 节，回答你的 `image_id`/segment 方案与它的机制哪个更接近，交换身份的负对照在两种方案下分别应该测出什么。
5. [MiniMind-O 官方仓库](https://github.com/jingyaogong/minimind-o)
   - 精读：`model/model_omni.py`、`dataset/omni_dataset.py`。
   - 阅读检查：列出当前固定 64-token 假设涉及的数据、模型和推理边界，拿第 7 节的清单逐格打勾，确认没有第四处写死的地方漏网。

## 22. 扩展题

1. 让模型按问题内容动态申请 pixel budget，而不只看图像宽高比。
2. 对 OCR tile 使用更高分辨率，对自然图区域使用低分辨率。
3. 比较 segment embedding、边界 token 与第四轴 RoPE 的多图身份方案。
4. 实现 packing：不同样本图片数不同，LLM visual token 数也不同时，减少
   padding。
5. 将本课选定的最佳臂作为实验 09 的唯一图像前端，禁止再次调参。

到这课收尾，系统的眼睛不再要求世界方方正正：一张图先按自己的宽高比切成 tile，encoder 尽量看全细节，resampler 压成 64 条带二维坐标的摘要交给 Thinker，两张图也不会串成一张。但它看到的仍是静止的世界。下一课[第 09 课](09_native_video.md)把"图"变成"带时间戳的帧序列"：帧要有真实时间位置，声音和画面要在同一条时间轴上对齐；本课锁定的图像前端会原封不动带过去，禁止再调参——扩展题 5 说的就是这件事。
