---
id: 45_objectives_and_takeaway
title: "目标函数动物园，带回桌子"
summary: "重建、JEPA、价值等价、flow、蒸馏各自在优化什么？24GB 该改第 30 课的哪一块？"
unit: frontier
play_tools: []
checkpoints:
  - "第九幕带回桌子的改造记录或失败记录。"
  - "重申毕业在第 32 课，档在第 33 课。"
---

# 第 45 课：目标函数动物园，只把一块搬回桌子

> 类型：研究 + 小改造（第九幕收官；不改第 32/33 课毕业标准）<br>
> 建议周期：2-3 天（盘点与选择半天，训一小截加探针一天，写改造或失败报告半天）<br>
> 硬件：单张 24GB 卡训一小截够用；Mac / 纯 CPU 可完成盘点、选择器和失败报告，训练慢数倍<br>
> 锚定：第 30 课自己的 `~/desk_wm/desk_wm.py` 与 `runs/desk1/best.pt`（或 Genesis 档 `runs/gen1`）；对照 [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm) 已读过的三个文件。不新训、不后训练 Cosmos 3<br>
> 产物：第 30 课模型现状表、目标函数选择器填写表、一次只选一个的改造记录或失败报告。毕业标准仍在第 32/33 课

## 1. 这一课做什么

第九幕读完了。第 34 课把 WorldScore、WorldModelBench、Physics-IQ 拆成三根不同的尺子；第 37 课把 teacher forcing、flow matching、self-forcing、蒸馏摊开；第 38 课把骨架再拆一遍。上一课（第 44 课）把「画面真」和「物理对」拆开，并且写明物理续写对了也不等于规划好。本课是第九幕收官，不是第二场毕业。刷榜知识要压回主线：你已经有一张桌子、一个会听动作的小世界模型、一套第 32 课的总装循环。现在只问一件事：优化目标决定模型在哪根尺子上好看。读完动物园之后，从第九幕只选一个 24GB 搬得动的改造，带回 [第 30 课](30_desk_world_model.md) 的小模型。做不成就写失败报告，仍然算完成本课。

整门课仍是同一个大项目。九幕走到这里：

| 幕 | 课 | 一句话 |
|---|---|---|
| 第一幕：第一个世界模型 | 01-04 | 在 CarRacing 上复现 World Models，理解 V-M-C |
| 第二幕：潜空间方法 | 05-08 | RSSM、Dreamer、MuZero、TD-MPC2 |
| 第三幕：生成式世界模型 | 09-12 | token、扩散、latent action |
| 第四幕：JEPA 路线 | 13-16 | 不生成像素，预测表征；三条路收官 |
| 第五幕：评测与改造 | 17-19 | 预测准、生成真、规划好必须拆开 |
| 第六幕：空间与物体 | 20-23 | 3D/4D 状态、视频基础模型、物体槽 |
| 第七幕：接到身体上 | 24-27，理论 33 | 视觉预测控制、模仿、VLA；具身程度 E0 到 E5 |
| 第八幕：桌宠毕业设计 | 28-32 | 会看、会想、会克制的最小系统 |
| 第九幕：刷榜、声音与配方 | 34-45 | 读榜和选零件；24GB 能跑的才动手 |

贯穿主干没有换：观察先压成状态，再按动作预测下一状态，然后展开多条未来，给未来打分，最后选动作。本课动的是「预测下一状态」那一截的**训练目标**，不是身体，也不是 E 档。第九幕唯一必须动自己代码的课就是这一课。对象写死：`~/desk_wm/desk_wm.py` 里的 `DeskWM`，损失现在是特征空间的一步 MSE，历史窗口 2 帧，动作是 4 类 embedding 拼到每个 patch，没有声音通道。改造只许三选一：混入 self-forcing、加很粗的声音能量通道、或加离开视野的记忆槽。Cosmos 3 的 Generator 后训练不在必做清单里，24GB 主线也不靠它毕业。

[第 32 课](32_ship_desk_pet.md) 结尾写过：离每天能开的桌宠还差的是你 `NEXT.md` 里那一个真瓶颈，通常是接触盲、人的下一秒飘、记不住昨天那杯水、或真机延迟把克制变成马后炮。本课不把那一页提案作废，也不另开一场毕业答辩。你带回桌子的若是 self-forcing，针对的是第 30 课漂移曲线上自由 rollout 越过偷懒基线太早；若是声音能量，针对的是第 36 课那句「麦克风是观察」；若是记忆槽，针对的是 [第 21 课](21_persistent_4d.md) 的转头就忘。三件一起做，预算会把对照搅浑。只做一件。

术语速查：

| 术语 | 一句人话 |
|---|---|
| 目标函数 | 训练时真正被梯度下降的那个标量；它规定模型在哪根尺子上好看 |
| 像素重建 / ELBO | 赌「画得像」：VAE、Dreamer 的解码项，外加把编码分布按住的 KL |
| token 交叉熵 | 赌「下一个符号对」：IRIS 把画面切成词，用语言模型的损失续写 |
| 扩散 / flow matching | 赌「噪声到数据的路径对」：去噪或学一条速度场，画质尺子上常占便宜 |
| JEPA | 赌「表征对」：不解码像素，预测未来的抽象向量 |
| 价值等价 | 赌「决策对」：隐状态只要把价值、奖励、策略算对，外观可以错 |
| 对比 / 能量 | 赌「槽或状态可分」：真转移能量低，假转移能量高 |
| DMD / self-forcing | 赌「推理分布对」：训练时按推理方式滚动，或让学生分布贴上教师分布 |
| 24GB 能带走 | 单卡能训一小截、能用第 30 课探针验收的改造；不是榜上的第一名 |
| 失败报告 | 改造做了、尺子没动或变差，把命令、曲线、原因写下来，本课照样过 |

## 2. 问题

第九幕最容易犯的错，是把「我读过 Cosmos 3 / Self-Forcing / Physics-IQ」写成「我的桌宠升级了」。读榜和毕业是两件事。[第 12 课](12_frontier_landscape.md) 的三问（动作起作用吗、离开视野的东西还在吗、能不能拿去选动作）和第 17 课的三分评测（预测准、生成真、规划好）仍然有效。WorldScore 高，说的是世界生成任务上画面和三维一致性；Physics-IQ 高，说的是真实物理实验续写对得上；桌宠克制要的是：同一段历史，伸手和不动的想象分岔，并且碰杯概率能挡住一次真实的伸手。三根尺子可以互不相让。

本课要同时处理四个具体问题。

第一，目标函数会被误当成架构。同一副 ViT，换成重建损失就在养桌面纹理，换成特征 MSE 就在跟 DINOv2 的 patch 走，换成对比能量就在把杯子和手推开。第 16 课的四轴里，这是「目标」那一根，不是「架构」。第 38 课的骨架动物园回答状态放在哪、动作从哪进；本课回答损失在惩罚什么。两课不要并成一句「我用了 Transformer 世界模型」。

第二，榜上的第一通常优化了生成真。像素重建、扩散、flow matching、token 交叉熵，都会把容量喂给「下一帧看起来像」。[第 02 课](02_vae_visual_compression.md) 已经见过：重建损失低，弯道探针不一定高。[第 07 课](07_muzero_value_equivalence.md) 从反面证明：不重建也可以决策。第 44 课又补了一刀：视觉质量和 Physics-IQ 不必同序。所以「换一个更现代的生成目标」不自动让动作对换变好，更不自动让克制变好。

第三，第 30 课的小模型有四件现状必须先写下来，再谈改造。目标是什么、记忆有多长、动作怎么进网络、有没有声音通道。不写现状，改造没有基线。本课默认那份胶水脚本的答案其实已经钉死，第 6 节对着文件再对一次，防止你改过 `hist` 却还按 2 帧在做笔记。

第四，24GB 主线只能带走一块。混入 self-forcing、加很粗的声音能量、加离开视野的记忆槽，三件都能在第 30 课代码上动几行，也都能用已有探针验收。后训练 Cosmos 3、在桌面视频上重训 DiT、把 WorldScore 全套样本跑一遍，都不是本课动作。选一块、训一小截、对换加恒常各测一次。数字没动，就写失败报告。把没跑的命令写成「已完成」，本课判负。

一条界限先划清。第 32 课仍是桌宠总装：五件行为必须查询世界模型。第 33 课仍是 E0 到 E5：没有查询就不要写 E4。本课既不把第九幕的阅读笔记升级成 E 档，也不因为你混入了 self-forcing 就宣布桌宠毕业。第九幕是读榜和选零件。零件装上了，回去第 32 课的对照表看克制那一行有没有新的 `safety_rewrite`；没装上，对照表保持原样，诚实记「未改」。

## 3. 准备

本课不从零采集，也不新 clone 训练用的大仓库。缺第 30 课产物就回到那一课补，不要用随机初始化的 ViT 冒充基线。

- [第 30 课](30_desk_world_model.md) 的工作目录 `~/desk_wm`。至少要有：`desk_wm.py`、`collect_desk.py`（或 `collect_genesis.py`）、一份带动作的 `data/ep_*.npz`（或 `data_genesis/`）、`runs/desk1/cache/ep_*.pt`、`runs/desk1/best.pt`。Genesis 档把 `desk1` 换成 `gen1` 即可。`NOTES.md` 里应已有对换表和漂移曲线；没有的话先补跑 `python desk_wm.py --mode swap` 与 `--mode drift`，本课的对照才有分母。
- 第 30 课的动作对换协议原样有效：同一历史，四键分岔，对角线为 0，看左对看右随步数增大。物体恒常的操作化沿用 [第 21 课](21_persistent_4d.md) 和 [第 32 课](32_ship_desk_pet.md)：杯子被挡或离开视野，状态不该立刻当它不存在。本课用「遮住中心 patch 再预测」做最小探针，不要求再跑 CUT3R。
- 回访这些课的观念，不必重训：第 02 课 ELBO 两项、第 07 课价值等价探针、第 10 课去噪目标、第 13 课坍缩、第 16 课四轴、第 17 课三分评测。第 34、37、38、44 课若正文已读，把地图、配方表、骨架选型、三列尺子笔记放在手边；还没读完也可以先做本课，选择器里用到的榜名以本课第 5.9 节为准，不编造分数。
- Python 环境沿用第 30 课。本课默认不再下载 DINOv2：编码缓存已经在 `runs/*/cache`。选声音通道的人额外 `pip install sounddevice`，并准备补采若干段带麦克风的轨迹；不选这一档就不要装。
- 磁盘再留 1GB 给新的 `runs/desk45_*`。基线检查点只读，不要覆盖 `runs/desk1/best.pt`。
- 硬件预期：24GB 卡上，10 个 epoch、horizon 3 的 self-forcing 通常十几分钟；Mac / CPU 按一到两小时准备。声音档的耗时在采集，不在预测器。记忆槽档和基线同量级。

没有第 30 课对换表的人，本课没有基线可减。先回去把复现 #9 做到「看左对看右第 10 步明显大于 0」，再来选零件。

## 4. 学习目标

1. 对着第 30 课的 `desk_wm.py` 填出现状四格：主损失是什么、记忆存在哪、动作怎么编码、有没有声音通道。不靠记忆，靠文件。
2. 用已授课或已核论文，说出七种目标函数各自在赌什么，并指出每种在 WorldScore、Physics-IQ、动作对换、桌宠克制四根尺子上哪根可能动、哪根通常不动。
3. 完成目标函数选择器：先自己猜一列，再对照 5.9 节揭晓表，把猜错的格子写成一句「我把生成真当成了 X」。
4. 从三块零件里只选一个，在 `~/desk_wm` 里改代码、训一小截，用动作对换和物体恒常（中心遮挡探针）对照基线。
5. 无论数字变好、不动还是变差，都写出改造记录或失败报告；报告里必须有完整命令，禁止把没跑的改造成绩写进去。
6. 口头说清三句话：毕业标准仍在第 32 课；具身程度仍在第 33 课；第九幕是读榜和选零件。

## 5. 原理

十个小节。前一节把「目标等于尺子」钉死，中间七节是动物园，每种都走完直觉、机制、数学、已有代码落点、验证。选择器在 5.9。5.10 说明为什么只能带走一块，以及它为什么不是毕业。

### 5.1 优化目标决定你在哪根尺子上好看

同一张桌子、同一段伸手录像，可以拿去训七种完全不同的模型。差别往往不在网络层数，在损失问的那句话。问「下一帧每个像素像不像」，模型会把容量花在木纹和反光上。问「下一个 token 对不对」，模型会把容量花在码本里的高频符号上。问「两个表征贴不贴」，模型可以完全不画画。问「价值对不对」，模型可以不认识杯子长什么样，只认识碰杯之后奖励掉了。你之后在哪张榜、哪张探针表上好看，几乎在写损失那一行就已经定了。

[第 17 课](17_evaluating_world_models.md) 把「好」拆成预测准、生成真、规划好。第九幕又补了两根公开尺子：WorldScore 测世界生成（下一场景、三维与四维一致性、文生/图生视频），Physics-IQ 测真实物理实验的续写是否落在真录像的空间/时空重叠上。桌宠另有两根课内尺子：动作对换（[第 03 课](03_mdn_rnn_action_conditioned.md) 发明，第 30 课搬到桌上），以及克制（第 32 课：想象会碰杯就不执行）。四根公开或课内尺子互不背书。本课的动物园就是一张「赌什么 / 哪根可能动」的对照，不是一张「谁是最好的世界模型」的总排名。

类比可以走到这里：考试大纲决定你会刷哪一科。失效处也很硬：世界模型的「考试」有很多份试卷，出题人还经常把试卷名字都印成 World Model。你的工作是先看卷子考什么，再决定改自己的哪一行损失。

### 5.2 像素重建与 ELBO：赌画得像

[第 02 课](02_vae_visual_compression.md) 的 VAE 把 64×64 赛道压成 32 个数，再解回来。训练最大化 ELBO（证据下界）：一项逼解码器把像素还原，一项用 KL 把编码分布按在标准正态附近，免得后验乱跑。仓库里没有单独的 β 旋钮，等于 β=1：

$$
\mathcal{L}_{\mathrm{ELBO}} = \mathbb{E}_{q(z\mid x)}[-\log p(x\mid z)] + \mathrm{KL}\bigl(q(z\mid x)\,\|\,p(z)\bigr)
$$

前一项是「画得像」，后一项是「编码要规整」。两项会打架：KL 太大就后验塌缩，解码器无视 $z$；重建项独大就把容量花在草纹理上，弯道探针读不出来。第 02 课并排重建损失和弯道探针，就是在演示：同一目标函数内部，生成真和「对决策有用」已经可能分家。

Dreamer / RSSM（[第 05 课](05_rssm_planet.md)、[第 06 课](06_dreamerv3_imagination.md)）把这件事做成视频版。观察损失仍是重建像素（或等价的似然），再加先验与后验之间的 KL，以及奖励头、继续头。想象训练用的是这个世界模型滚动出来的潜状态，不是像素本身。所以 Dreamer 可以重建糊、walker 照样跑，第 17 课把这当作「生成真和规划好不同步」的标准例。赌的仍是「能从状态解回观察」，外加「先验跟得上后验」。桌面上若把第 30 课的特征 MSE 改回像素重建，你多半会看到木纹变清楚、对换表不一定动：梯度在为面积大的背景服务，杯沿那一圈几乎没票。第 16 课「一杯水被手碰到」的重建写法，就是这个陷阱的桌面版。

代码落点：`ctallec/world-models` 的 `models/vae.py` 与 `trainvae.py`；`dreamerv3-torch` 的 `models.py::WorldModel._train` 里 `observation_loss` 与 `kl_loss`。验证：重建图、KL 曲线、以及第 02 课那种「重建好看但探针读不出弯道」的并排。不要用 ELBO 下降代替动作对换。

### 5.3 离散 token 交叉熵：赌下一个符号对

[第 09 课](09_iris_token_world_model.md) 的 IRIS 先用 VQ 把一帧切成 16 个码本下标，再把动作也当成符号，用 Transformer 做下一 token 预测。损失是分类交叉熵。词表有限，工程上借用了语言模型全套采样、温度、teacher forcing。赌的是：离散符号序列上的下一步分类，足够支撑「交互式地播放未来」。

$$
\mathcal{L}_{\mathrm{CE}} = -\sum_{i}\log p_\theta(w_i \mid w_{<i})
$$

$w_i$ 有时是图像 token，有时是动作 token。动作若进了序列，理论上对换有门；动作若只写在条件里却很少被用到，交叉熵照样可以靠「复读上一帧的码字」刷得很低。这和第 03 课 MDN 的惯性外推是同一类病，只是空间从 32 维连续 $z$ 换成了 512 词的码本。码本坍缩（死码）是额外的病：词表没用满，未来能说的话变少，梦里就会出现第 09 课那种穿帮镜头。

验证：token 准确率、码本使用率、以及必须另做的动作对换。WorldScore 一类生成榜上，token 引擎可以因为「看起来像游戏」拿分；Physics-IQ 不保证，因为它的答案是真实实验录像的空间重叠，不是码本里的高频符号。桌宠克制需要你能从 token 解回「杯子还在不在」或至少解回会碰杯的特征。只盯交叉熵，解不回这句。

### 5.4 扩散与 flow matching：赌噪声到数据的路径对

[第 10 课](10_diamond_diffusion_world_model.md) 把下一帧写成逐步去噪。训练：对干净帧加噪声，让网络从加噪版恢复。生成：从纯噪声沿噪声水平积分到 0。最优去噪器等价于 score（对数密度的梯度）。DIAMOND 用 EDM 预条件，少步去噪才能在想象 rollout 里撑住。赌的是：学会噪声到数据的路径，就能从分布里抽出「像真的」下一帧。

Flow matching（Lipman 等，arXiv:2210.02747，第 37 课配方表里的连续生成目标）把离散去噪步换成一条连续速度场。线性路径上 $x_t=(1-t)x_0+t x_1$，网络学 $v_\theta(x_t,t)$ 去贴 $x_1-x_0$：

$$
\mathcal{L}_{\mathrm{FM}}=\mathbb{E}_{t,x_0,x_1}\bigl\lVert v_\theta(x_t,t)-(x_1-x_0)\bigr\rVert_2^2
$$

Cosmos 3 的 Generator、π0 一类连续策略头，公开材料里走的是这一族，不是另一套物理。第 37 课禁止把 flow matching 和扩散写成互不相干的两个宇宙：都在学一条从噪声到数据的路径，参数化不同。双向 DiT 在整段视频上一起去噪，画质高，不能边看边播；因果 AR-扩散才接近交互。世界模型要的若是第 32 课那种 200 ms 一拍的查询，双向少步高清往往用不上。

验证：人眼、FVD、WorldScore 这类生成真尺子上，扩散 / flow 经常好看。Physics-IQ 论文自己的结论是：视觉真实感和物理续写关系不大。动作对换只在动作被写进条件并且滚动时能替换时成立；许多文生视频系统连动作端口都没有，第 12 课三问第一问直接出局。桌宠克制：24GB 上不要指望把第 30 课的 2 层 ViT 换成桌面 DiT。你要的若是路径对，self-forcing 那一档是把「推理时的路径」对准，仍然在特征 MSE 骨架上，搬得动。

### 5.5 JEPA：赌表征对

[第 13 课](13_ijepa_from_scratch.md) 的 I-JEPA 不解码像素。学生网络看被挡住的图，预测目标网络在遮挡位置上的表征。损失是表征空间的距离。常数解会让损失严格为零，所以必须靠 EMA 目标、停梯度、出题设计把坍缩堵住。赌的是：未来只要在抽象空间里对得上，纹理可以扔。

[第 14 课](14_eb_jepa_action_conditioned.md) 把动作接进预测器，并在 Two Rooms 里用表征距离做规划。V-JEPA 2（第 15 课）把这条路做到视频骨干和机械臂规划。第 16 课把它定为三条路线里的「预测表征」。和第 30 课默认的 DINO-WM 式特征 MSE 是近亲：都不重建像素，都在预训练视觉特征上做下一步。差别是 JEPA 的目标来自另一个（常为 EMA）编码器，DINO-WM 的目标来自冻结 DINOv2 对下一帧真实画面的输出。后者有外部锚，坍缩没那么容易；前者更省像素税，也更要靠防坍缩零件。

验证：线性探针、规划成功率、动作对换（AC-JEPA 的 IDM 就是在逼「不同动作必须留下不同痕迹」）。WorldScore 通常无意义：根本不生成给榜看的视频。Physics-IQ 默认也不是这把尺子；V-JEPA 2 有自己的世界模型基准，那是表征路线的对照，不要和 Physics-IQ 混成一个总分。桌宠克制：特征空间上的分岔，必须再映到杯子坐标或碰杯头上，安全层才读得懂。第 30 课已经走在这条路上，本课若再「改成 JEPA」而不改验收，多半是把名字换了。

### 5.6 价值等价：赌决策对

[第 07 课](07_muzero_value_equivalence.md) 的 MuZero 不重建观察。隐状态 $s$ 被三个头盯着：策略、价值、奖励。动力学 $g$ 在隐空间里走，MCTS 用这三个头打分。Grimm 等人把立场写成价值等价：两个模型只要对给定的策略类和价值函数诱导相同的贝尔曼更新，就等价。外观不在订单上。

$$
\mathcal{L} = \ell_p(\pi_\theta(s),\,\pi^{\mathrm{MCTS}}) + \ell_v(v_\theta(s),\,z) + \ell_r(r_\theta(s),\,u)
$$

三项都来自真实对局或自博弈的统计量，不来自像素。常数隐状态喂不饱它们，所以免重建却不一定坍缩。第 07 课的线性探针是验收：CartPole 上哪些物理量还读得出，哪些被扔掉。TD-MPC2（第 08 课）把同一立场搬进连续控制，防坍缩改靠 latent 一致性。

验证：下游回报、价值误差、探针 $R^2$。WorldScore、Physics-IQ 基本不动，因为模型不产出给这两份榜看的视频。动作对换：应在奖励/价值头上分岔，不必在画面上分岔。桌宠克制：只有当你把「碰杯」写进奖励或价值，这项目标才会保护杯子。第 30 课没有奖励信号，桌面上也没有可无限重置的自博弈，纯血 MuZero 目标搬不回去。不要为了「更现代」硬接一个假奖励。

### 5.7 对比与能量：赌槽或状态可分

[第 23 课](23_object_centric_wm.md) 的 C-SWM 把状态拆成 $K$ 个槽，转移走图网络，损失是对比能量：真的下一步要近，从经验池打乱抽来的假状态要远。能量是槽上的平方距离，损失是铰链：

$$
\mathcal{L}=H+\max(0,\gamma-\tilde H)
$$

$H$ 是真转移能量，$\tilde H$ 是假状态能量，论文 $\gamma=1$。赌的是：杯子、手、背景应该可分，关系应该走边，不必为桌面纹理养解码器。打开 `--decoder` 就退回重建，论文用这张消融说明「赢在目标，不是赢在多几个卷积」。

[第 14 课](14_eb_jepa_action_conditioned.md) 把能量语言用在 JEPA 上：真搭配能量低，面要靠 VICReg 一类正则撑住，否则全部打零分。对比式采负样本、正则式不许压平、EMA 式靠不对称，是同一张打分面的三种撑法。第 13 课已把坍缩算成目标函数里的合法满分答案。

验证：槽掩码是否绑住物体、交换槽之后能量是否升高、`ignore_action` 打开后对换是否塌掉。WorldScore / Physics-IQ 仍然不是主场。动作对换是主场。桌宠克制：槽若真把杯子和手分开，「手的槽靠近杯子的槽」比「整图特征动了一点」更像安全层能用的句子。第 30 课把 C-SWM 列为备选架构，真实桌面的槽发现很难；本课若选记忆槽，做的是「一个离开视野的寄存器」，不是把 C-SWM 的 $K=4$ 重训一遍。不要把没跑通的槽发现写成已完成。

### 5.8 DMD 与 self-forcing：赌推理分布对

第 03 课已经量过曝光偏差：训练逐步喂真 $z$（teacher forcing），推理却吃自己的预测，自由 rollout 的误差会先于 teacher forcing 越过偷懒基线。第 30 课的 `drift` 模式把同一张图搬到了桌上。第 37 课把现代修补写成一条线：CausVid（Yin 等，arXiv:2412.07772）把双向教师蒸馏成少步因果生成器；Self Forcing（Huang 等，arXiv:2506.08009）在训练里按推理配方做自回归 rollout，用 KV cache 对齐推理分布。

DMD（Distribution Matching Distillation，Yin 等，arXiv:2311.18828）是更早的分布层蒸馏：让一步（或少步）学生生成器的输出分布贴上多步扩散教师的分布，梯度来自两个 score 的差，不要求逐步轨迹一一对应。CausVid 把这类想法用到因果视频上。Self-Forcing 更进一步：学生训练时看到的历史，就是它自己刚刚生成的那一截，和上场时同分布。赌的都是「推理时会见到的分布」，不是「教师 forcing 时见到的真帧分布」。

第 30 课没有扩散教师，也没有 KV cache 的 DiT。能搬的是这句话的最小实现：训练窗口里以概率 $p$ 把真实特征换成模型自己刚预测的特征，再算下一步 MSE。这是 scheduled sampling 量级的 self-forcing 混入，不是 Huang 等人论文的复现。DMD 本身 24GB 桌上做不了：你没有教师视频扩散模型，也不该去后训练 Cosmos 3 来充当教师。选择器里 DMD / self-forcing 算同一格「推理分布对」；动手只做混入，不写「我完成了 DMD」。

验证：同一检查点上的 teacher forcing 对自由 rollout 裂口应缩小；动作对换不应塌掉（动作通道还在）；物体恒常不保证变好，因为曝光偏差和「离开视野」是两件事。WorldScore 上，对准推理分布可能让长视频更稳，那是生成真的时间轴，不是桌宠克制。

### 5.9 互动：目标函数选择器

先不要看揭晓表。选一个目标，只凭你现在的直觉，给四根尺子各打一格：可能变好 / 通常不动 / 可能变差。四根尺子的定义固定：

| 尺子 | 它实际在问 |
|---|---|
| WorldScore | 世界生成：下一场景、三维/四维、文生或图生视频好不好看、空间是否自洽 |
| Physics-IQ | 真实物理实验录像的续写，和真后续比空间/时空重叠，不是规划成功率 |
| 动作对换 | 同一历史只换动作，想象分不分岔（第 03、30 课） |
| 桌宠克制 | 伸手会碰杯时，安全层能不能改写动作（第 32 课） |

你的选择（七选一，只圈一个）：像素重建 / ELBO；离散 token 交叉熵；扩散 / flow matching；JEPA 表征；价值等价；对比 / 能量；DMD / self-forcing。把四格写进笔记，写完再看下面这张揭晓表。

| 目标 | WorldScore | Physics-IQ | 动作对换 | 桌宠克制 |
|---|---|---|---|---|
| 像素重建 / ELBO | 可能变好（画面更像） | 通常不动（第 44 课：好看不等于续写对） | 不保证（惯性外推也能把像素误差刷低） | 不保证（杯沿票数太少） |
| 离散 token 交叉熵 | 看 tokenizer 和是否像视频生成；可能变好 | 通常不动 | 仅当动作 token 真被用；否则与重建同类 | 仅当能解回碰杯相关状态 |
| 扩散 / flow matching | 经常变好（生成真主场） | 不随画质同序 | 仅当有可替换的动作条件且能逐步滚 | 24GB 桌宠默认搬不动整段 DiT |
| JEPA 表征 | 通常不动（不生成榜要的视频） | 通常不动（尺子不是表征基准） | 动作条件版本可能变好 | 可能变好，若分岔映到杯子/碰杯头 |
| 价值等价 | 通常不动 | 通常不动 | 在价值/奖励头上可能变好，画面上不必 | 仅当碰杯写进了奖励或价值 |
| 对比 / 能量 | 通常不动 | 通常不动 | 可能变好（真假转移必须可分） | 可能变好（槽把杯和手分开时） |
| DMD / self-forcing | 长视频一致性上可能变好 | 不保证（在分布上不等于物理对） | 不应塌；多步对换可能略好 | 仅当 2 秒梦以前死于曝光偏差 |

判读规则。猜对三格以上，说明你没把生成真当成万能。把「扩散」四个格子全写成可能变好，回去重读第 17 课和第 44 课。把「价值等价」写成 WorldScore 会变好，说明仍在用视频生成器理解 MuZero。选择器不产生分数，只产生「我接下来改第 30 课哪一块才跟我在乎的尺子有关」。

第 7 节动手时还要再圈一次：三块可搬零件里只许一块。选择器的七格是观念；三块零件是 24GB 动作。七格里的 DMD / self-forcing 对应零件「混入 self-forcing」；声音能量并不对应动物园里单独一种目标，它是第 36 课的观察通道，改的是输入，损失仍可以是特征 MSE；记忆槽对应「离开视野还在」，更接近第 21 课和第 39 课的记忆，也不强制换损失。输入、记忆、训练时看谁的输出，三件事不要并成「我换了目标函数」一句。

### 5.10 24GB 只能带走一块，这不是毕业

可搬的三块，对应第九幕三句不同的话。

混入 self-forcing：第 37 课的配方。第 30 课 `train_loop` 现在每步喂真 $z$（`EpisodeCache` 返回 `z[:-1], a, z[-1]`，`one_step` 只看真实窗口）。推理时 `rollout` 把 `pred` 接回 `z_win`。训练分布和推理分布从第二步起就不是同一个。混入之后，训练也开始吃自己的预测。预期优先动漂移曲线，对换表应大体保住，恒常不保证。

加很粗的声音能量通道：第 36 课的观察。`collect_desk.py` 只存 `frames`、`actions`、`fps`，麦克风从未进 npz。杯子碰桌沿和键盘敲击可以画面几乎一样。能量通道赌的是：一个标量 RMS 当额外观察，模型能把「下一秒会不会有撞击」和画面分开。损失不必换。预期：有撞击的片段上，对换或恒常可能间接受益；没有撞击标签的旧数据，通道是死的。必须补采。

加离开视野的记忆槽：第 21 课的恒常和第 32 课行为 1。第 30 课的记忆就是 `hist=2` 的滑窗，杯子离开这两帧就被忘。一个 GRU 槽吃平均 patch，训练时随机挡住中心，逼槽在看不见时仍贴住被挡前的均值。预期优先动遮挡探针，对换不应塌成全 0。

为什么只许一块。两块一起改，对换表动了你不知道是谁的功劳，失败了也不知道该回滚哪一行。[第 19 课](19_surgery_experiments.md) 的手术纪律在这里仍然有效：一次改一个机制。为什么不是毕业。毕业要的是第 32 课五件行为都查询了世界模型，以及第 33 课按是否题打出的档。一块零件最多缩短漂移、补一个观察、或让遮挡后少忘一点。它不自动产生安全层，不自动产生失败承认，不自动把 E3 变成 E4。第九幕读榜和选零件；第八幕的总装标准一字不改。

## 6. 源码导读

锚定就是你自己的第 30 课目录，外加已经 clone 过的 `dino_wm` 对照。本课不改官方 `train.py`，也不把它改造成摄像头管道。先把现状从文件里读出来，再谈第 7 节往哪几行动刀。

建议工作目录仍是 `~/desk_wm`。下面每条命令都在该目录执行。先确认基线文件还在：

```bash
ls desk_wm.py collect_desk.py runs/desk1/best.pt
```

若走 Genesis 档：

```bash
ls desk_wm.py collect_genesis.py runs/gen1/best.pt
```

`ls` 报缺哪个补哪个。`best.pt` 是第 30 课 `train_loop` 按验证 MSE 写下的字典，键包括 `model`、`hist`、`best_val`。探针脚本 `load_model` 只认这个结构，本课新检查点要另存目录，不要覆盖。

| 文件与位置 | 是哪个零件 | 带着什么问题读 |
|---|---|---|
| `collect_desk.py` 里 `np.savez_compressed` | 观察和动作怎么落盘 | 存了哪些键？有没有麦克风？`actions` 为什么比 `frames` 短 1？ |
| `desk_wm.py::DinoEncoder` | 冻结视觉编码器 | 和官方 `models/dino.py` 一样取 `x_norm_patchtokens` 吗？本课改目标时动不动它？ |
| `desk_wm.py::DeskWM.fuse` | 动作编码 | `nn.Embedding(4, 32)` 拼到每个 patch 最后一维，对应官方 `concat_dim: 1`。声音能量若要加，最省事的位置在哪？ |
| `desk_wm.py::DeskWM.one_step` | 一步预测 | 输入窗口长度等于 `self.hist`，默认 2。这就是全部记忆吗？ |
| `desk_wm.py::EpisodeCache.__getitem__` | 训练样本 | 只返回一步目标 `z[-1]`。self-forcing 为什么必须另写一个更长窗口的 Dataset？ |
| `desk_wm.py::train_loop` 里 `F.mse_loss(pred, z_next)` | 现行目标 | 这是特征 MSE、teacher forcing、无像素项、无 KL。和 5.2 到 5.8 哪一格对应？ |
| `desk_wm.py::rollout` | 推理分布 | `z_win` 接的是 `pred`。和 `train_loop` 喂的真 $z$ 从第几步开始分家？ |
| `desk_wm.py::swap_probe` / `drift_probe` / `neg_probe` | 验收 | 本课改造后仍调用它们。`load_model` 若遇到多出来的权重会怎样？ |
| 官方 `models/visual_world_model.py::forward` | 对照 | `emb_criterion` 也是 MSE；解码器接在 `z_pred.detach()` 后。本课有没有解码器主损失？ |
| 官方 `models/visual_world_model.py::replace_actions_from_z` | 对照 | 滚动时覆盖动作维。本课 `rollout` 每步把新动作接到 `a_win` 尾部，是同一件事的缩小版吗？ |

现状的标准答案，用来填 Step 1 的表，允许你的目录和默认值有出入，但要写明。

目标。`train_loop` 第 580 行附近：`pred = model.one_step(z_hist, a_hist)`，`loss = F.mse_loss(pred, z_next)`。`z_next` 来自冻结 DINOv2 对下一帧的 patch。没有像素重建，没有 ELBO 的 KL，没有 token 交叉熵，没有扩散噪声水平，没有 JEPA 的 EMA 目标，没有 MuZero 三个头，没有 C-SWM 铰链。`--blind` 只在负对照重训时把动作置零，主训练不加。这是「特征空间的动作条件一步 MSE + teacher forcing」。第 16 课三条路线里它靠表征预测最近，和第 14 课 AC-JEPA 的差别是目标锚在冻结 DINO 上。

记忆。`DeskWM.__init__` 默认 `hist=2`。没有 RNN 隐状态，没有槽，没有按位姿取回的记忆库。窗口里的 $2\times 256$ 个 patch 就是全部历史。转头超过 2 帧，离开视野的杯子从输入里消失。[第 31 课](31_interaction_memory.md) 的环形短时记忆是另一份头，不在 `desk_wm.py` 里；本课不要假装已经接上。

动作编码。`self.action_embed = nn.Embedding(action_dim, act_emb)`，`act_emb=32`，`fuse` 里 `unsqueeze(2).expand` 到每个 patch 再 `cat`。四个符号：`NAMES = ["look_left", "look_right", "reach", "stay"]`，与 `collect_desk.py` 的 `a/d/w/s` 一致。摄像头档没有本体感觉通道，官方 DINO-WM 配置里的 proprio 被第 30 课故意拿掉。

声音通道。`collect_desk.py` 的 `np.savez_compressed(path, frames=arr_f, actions=arr_a, fps=args.fps)` 只有三键。`encode_split` 只把 `frames` 送进 DINO、把 `actions` 存成 `a`。麦克风、RMS、频谱都不存在。Reachy Mini 的麦阵列是第 32 课身体的事，没有自动流进第 30 课模型。

官方仓库只当对照，本课命令仍走胶水脚本。若要再看一眼官方损失隔离：

```bash
python -c "import inspect,sys; sys.path.insert(0,'dino_wm'); print('skip if not cloned')"
```

没有 clone 也不扣分，第 30 课第 6 节已经读过 `VWorldModel.forward`。本课动刀位置是 `~/desk_wm/desk_takeaway.py`（新建），基线脚本只读。

## 7. 实验

以下命令都在 `~/desk_wm` 执行。基线检查点只读。新运行目录用 `runs/desk45_sf`、`runs/desk45_audio` 或 `runs/desk45_slot`，三选一。做不完、数字不动、对换塌掉，都把命令和日志抄进报告，本课仍算完成。不要后训练 Cosmos 3，不要同时开两个改造。

### Step 0: 确认第 30 课产物和设备

```bash
python -c "from pathlib import Path; import torch; p=Path('runs/desk1/best.pt'); print('best', p.exists(), 'cache', len(list(Path('runs/desk1/cache').glob('ep_*.pt')))); print('cuda', torch.cuda.is_available())"
```

预期：`best True`，cache 段数不少于你第 30 课训过的数量（摄像头档通常二十以上）。`cuda True` 表示 24GB 主线；Mac 上是 `False`，脚本会走 MPS 或 CPU。若你当时用的是 Genesis 档，把命令里的 `desk1` 改成 `gen1`，后面所有 `desk1` 同样替换。没有 `best.pt` 就不要往下写改造，先回到第 30 课 Step 4。

再确认可以从当前目录 import 胶水脚本：

```bash
python -c "import desk_wm; print(desk_wm.NAMES, desk_wm.DeskWM().hist)"
```

预期打印 `['look_left', 'look_right', 'reach', 'stay'] 2`。报 `No module named desk_wm`，说明当前目录不是 `~/desk_wm`。

### Step 1: 填现状表（必做，先于改代码）

对着源码，不要凭印象。把这张表抄进 `runs/desk45_NOTES.md`：

```text
目标：train_loop 里的损失是 ________。有无像素项 / KL / 交叉熵 / 扩散噪声。
记忆：hist = ____。有无 RNN / 槽 / 记忆库。离开视野的杯子还在不在输入里。
动作编码：Embedding 维数 ____，拼在 patch 的哪一维。四键名称。
声音通道：npz 的键有哪些。encode_split 有没有读麦克风。
基线对换：look_left 对 look_right 第 1 步 ____，第 10 步 ____，比例尺 ____。
基线漂移：第 1 步 teacher/free ____，越过偷懒基线大约第 ____ 步。
基线盲区：哪类动作在想象里几乎不动（抄第 30 课 NOTES）。
```

默认答案在第 6 节。你的 `hist` 若改过，以检查点里的 `hist` 为准：

```bash
python -c "import torch; c=torch.load('runs/desk1/best.pt', map_location='cpu', weights_only=False); print(c.keys()); print('hist', c.get('hist'), 'best_val', c.get('best_val'))"
```

### Step 2: 做完选择器再动手

打开第 5.9 节。先猜后看揭晓表。把「我圈的目标」和「四根尺子的猜测 / 揭晓」写进同一份 NOTES。揭晓之后，用一句话写你真正在乎的尺子：若是 2 秒梦漂得太快，零件应选 self-forcing；若是杯子离开画面就忘，选记忆槽；若是撞击只听得见看不见，选声音能量。WorldScore 第一不是选项。把这句话写在 NOTES 的「只选一个」下面，后面不准改口同时做三件。

### Step 3: 只圈一个改造

三块零件的预期和失败，抄到 NOTES 里圈中的那一行：

| 零件 | `--mod` | 需要新数据吗 | 预期可能动的尺子 | 失败长什么样 |
|---|---|---|---|---|
| 混入 self-forcing | `sf` | 否，复用 `runs/desk1/cache` | 漂移裂口缩小；对换应保住；恒常不保证 | 对换塌成全 0；或自由曲线比基线更早越过偷懒线 |
| 粗声音能量 | `audio` | 是，补采带 RMS 的 npz 并重新 encode | 有撞击的片段上，遮挡或对换可能间接受益 | 没采到撞击，能量几乎常数，数字与基线难分 |
| 离开视野的记忆槽 | `slot` | 否，复用旧 cache，训练时遮中心 patch | 中心遮挡探针的误差下降；对换应保住 | 对换塌掉；或遮挡误差与基线差不到 5% |

推荐默认：`sf`。它不新采数据，直接打第 30 课已经画过的漂移图，最接近第 37 课那句「训练时混入自己的预测」。下面 Step 4 的脚本三个 `--mod` 都实现了，但你只许把其中一个训完并写入报告。

### Step 4: 写入 `desk_takeaway.py`

把下面存成 `~/desk_wm/desk_takeaway.py`。它 import 第 30 课的 `DeskWM`、`CausalViT`、`rollout`、`pca_map`，不复制编码器，不覆盖 `desk_wm.py`。self-forcing 换的是 Dataset 和训练循环；声音档换的是 `fuse` 多一截能量；槽档在特征 MSE 之外加 GRU 槽，并用中心遮挡当辅助。三种权重互不兼容，所以检查点里必须写下 `mod`。

```python
"""desk_takeaway.py  第 45 课：三选一改造。禁止一次开两个 mod。"""
import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import desk_wm as D

NAMES = D.NAMES


def center_idx(n_patches, side=None):
    if side is None:
        side = int(round(n_patches ** 0.5))
    idx = []
    for r in range(side // 4, side - side // 4):
        for c in range(side // 4, side - side // 4):
            idx.append(r * side + c)
    return torch.tensor(idx, dtype=torch.long)


def mask_center(z, idx):
    z2 = z.clone()
    z2[..., idx, :] = 0
    return z2


class LongCache(Dataset):
    """多步窗口：给 self-forcing 用。z_future 长度 = horizon。"""

    def __init__(self, cache_dir, hist=2, horizon=3):
        self.files = sorted(glob.glob(str(Path(cache_dir) / "ep_*.pt")))
        self.hist = hist
        self.horizon = horizon
        self.index = []
        self.data = []
        for i, f in enumerate(self.files):
            pack = torch.load(f, map_location="cpu", weights_only=False)
            self.data.append(pack)
            t = pack["z"].shape[0]
            last = t - hist - horizon + 1
            for s in range(0, max(0, last)):
                self.index.append((i, s))
        if not self.index:
            raise SystemExit("cache 太短，不够 hist+horizon。减 --horizon 或检查 --cache")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        i, s = self.index[idx]
        pack = self.data[i]
        h, k = self.hist, self.horizon
        z = pack["z"]
        a = pack["a"]
        z_hist = z[s: s + h]
        z_fut = z[s + h: s + h + k]
        a_full = a[s: s + h + k - 1]
        extra = {}
        if "e" in pack:
            extra["e"] = pack["e"][s: s + h + k - 1]
        return z_hist, a_full, z_fut, extra


class DeskWMAudio(D.DeskWM):
    def __init__(self, visual_dim=384, action_dim=4, act_emb=32, hist=2, e_dim=8):
        nn.Module.__init__(self)
        self.hist = hist
        self.act_emb_dim = act_emb
        self.visual_dim = visual_dim
        self.e_dim = e_dim
        self.action_embed = nn.Embedding(action_dim, act_emb)
        self.energy_head = nn.Linear(1, e_dim)
        self.dim = visual_dim + act_emb + e_dim
        self.predictor = D.CausalViT(self.dim)

    def fuse(self, z, a, e=None):
        ae = self.action_embed(a)
        ae = ae.unsqueeze(2).expand(-1, -1, z.shape[2], -1)
        if e is None:
            e = torch.zeros(z.size(0), z.size(1), 1, device=z.device, dtype=z.dtype)
        else:
            e = e.unsqueeze(-1)
        ee = self.energy_head(e).unsqueeze(2).expand(-1, -1, z.shape[2], -1)
        return torch.cat([z, ae, ee], dim=-1)

    def predict_window(self, z, a, e=None):
        h = self.fuse(z, a, e)
        b, t, p, d = h.shape
        y = self.predictor(h.reshape(b, t * p, d)).reshape(b, t, p, d)
        z_hat, _ = self.split(y)
        return z_hat

    def one_step(self, z_hist, a_hist, e_hist=None):
        z_hat = self.predict_window(z_hist, a_hist, e_hist)
        return z_hat[:, -1]


class DeskWMSlot(D.DeskWM):
    def __init__(self, visual_dim=384, action_dim=4, act_emb=32, hist=2):
        super().__init__(visual_dim, action_dim, act_emb, hist)
        self.gru = nn.GRUCell(visual_dim, visual_dim)

    def slot_of(self, z_hist):
        pooled = z_hist[:, -1].mean(dim=1)
        init = pooled.detach()
        return self.gru(pooled, init)


def build_model(mod, visual_dim, hist, device):
    if mod == "audio":
        m = DeskWMAudio(visual_dim=visual_dim, hist=hist)
    elif mod == "slot":
        m = DeskWMSlot(visual_dim=visual_dim, hist=hist)
    else:
        m = D.DeskWM(visual_dim=visual_dim, hist=hist)
    return m.to(device)


def sf_loss(model, z_hist, a_full, z_fut, p, device):
    hist = model.hist
    b = z_hist.size(0)
    z_win = z_hist
    a_win = a_full[:, :hist]
    acc = []
    for k in range(z_fut.size(1)):
        pred = model.one_step(z_win, a_win)
        acc.append(F.mse_loss(pred, z_fut[:, k]))
        if k + 1 == z_fut.size(1):
            break
        coin = (torch.rand(b, 1, 1, device=device) < p).to(z_win.dtype)
        nxt = coin * pred.detach() + (1.0 - coin) * z_fut[:, k]
        z_win = torch.cat([z_win[:, 1:], nxt.unsqueeze(1)], dim=1)
        a_win = torch.cat([a_win[:, 1:], a_full[:, hist + k: hist + k + 1]], dim=1)
    return sum(acc) / len(acc)


def train_loop(args, device):
    ds = LongCache(args.cache, hist=args.hist, horizon=args.horizon)
    n_val = max(1, len(ds) // 8)
    n_tr = len(ds) - n_val
    tr, va = torch.utils.data.random_split(
        ds, [n_tr, n_val], generator=torch.Generator().manual_seed(0)
    )
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, drop_last=True)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False)
    pack0 = ds.data[0]
    if args.mod == "audio" and "e" not in pack0:
        raise SystemExit("audio 档需要 cache 里有键 e。先按 Step 4B 补采并 encode")
    model = build_model(args.mod, pack0["z"].shape[-1], args.hist, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = 1e9
    out = Path(args.run)
    out.mkdir(parents=True, exist_ok=True)
    p_idx = center_idx(pack0["z"].shape[1]).to(device)
    print("mod", args.mod, "n", len(ds), "sf_prob", args.sf_prob)
    for epoch in range(args.epochs):
        model.train()
        tr_loss = []
        for z_hist, a_full, z_fut, extra in tl:
            z_hist = z_hist.to(device)
            a_full = a_full.to(device)
            z_fut = z_fut.to(device)
            if args.mod == "sf":
                loss = sf_loss(model, z_hist, a_full, z_fut, args.sf_prob, device)
            elif args.mod == "audio":
                e = extra["e"].to(device)
                pred = model.one_step(
                    z_hist, a_full[:, : model.hist], e[:, : model.hist]
                )
                loss = F.mse_loss(pred, z_fut[:, 0])
            else:
                z_in = mask_center(z_hist, p_idx)
                pred = model.one_step(z_in, a_full[:, : model.hist])
                slot = model.slot_of(z_in)
                tgt = z_fut[:, 0]
                loss = F.mse_loss(pred, tgt) + 0.3 * F.mse_loss(slot, tgt.mean(dim=1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss.append(loss.item())
        model.eval()
        va_loss = []
        with torch.no_grad():
            for z_hist, a_full, z_fut, extra in vl:
                z_hist = z_hist.to(device)
                a_full = a_full.to(device)
                z_fut = z_fut.to(device)
                if args.mod == "sf":
                    va_loss.append(
                        sf_loss(model, z_hist, a_full, z_fut, 0.0, device).item()
                    )
                elif args.mod == "audio":
                    e = extra["e"].to(device)
                    pred = model.one_step(
                        z_hist, a_full[:, : model.hist], e[:, : model.hist]
                    )
                    va_loss.append(F.mse_loss(pred, z_fut[:, 0]).item())
                else:
                    pred = model.one_step(z_hist, a_full[:, : model.hist])
                    va_loss.append(F.mse_loss(pred, z_fut[:, 0]).item())
        tr_m, va_m = float(np.mean(tr_loss)), float(np.mean(va_loss))
        print("epoch %03d  train %.5f  val %.5f" % (epoch, tr_m, va_m))
        if va_m < best:
            best = va_m
            torch.save(
                {
                    "model": model.state_dict(),
                    "hist": args.hist,
                    "best_val": best,
                    "mod": args.mod,
                    "sf_prob": args.sf_prob,
                    "horizon": args.horizon,
                },
                out / "best.pt",
            )
    print("best val", best, "->", out / "best.pt")


def load_takeaway(run, device, cache_dir=""):
    ckpt = torch.load(Path(run) / "best.pt", map_location=device, weights_only=False)
    cache = Path(cache_dir) if cache_dir else Path(run) / "cache"
    files = sorted(cache.glob("ep_*.pt"))
    sample = torch.load(files[0], map_location="cpu", weights_only=False)
    mod = ckpt.get("mod", "sf")
    model = build_model(mod, sample["z"].shape[-1], ckpt["hist"], device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt, cache


@torch.no_grad()
def rollout_any(model, z0, a_seq, device, e0=None):
    if not isinstance(model, DeskWMAudio):
        return D.rollout(model, z0, a_seq, device)
    hist = model.hist
    z_win = z0.clone().to(device)
    a_win = a_seq[:1].repeat(hist).to(device)
    if e0 is None:
        e_win = torch.zeros(hist, device=device)
    else:
        e_win = e0.to(device)
        if e_win.dim() == 0:
            e_win = e_win.repeat(hist)
        elif e_win.numel() == 1:
            e_win = e_win.reshape(()).repeat(hist)
        elif e_win.numel() < hist:
            e_win = e_win[-1].repeat(hist)
        else:
            e_win = e_win[-hist:]
    preds = []
    for k in range(len(a_seq)):
        pred = model.one_step(
            z_win.unsqueeze(0), a_win.unsqueeze(0), e_win.unsqueeze(0)
        )[0]
        preds.append(pred.cpu())
        z_win = torch.cat([z_win[1:], pred.unsqueeze(0)], dim=0)
        a_win = torch.cat([a_win[1:], a_seq[k: k + 1].to(device)], dim=0)
        e_win = torch.cat([e_win[1:], e_win[-1:]], dim=0)
    return torch.stack(preds)


def swap_probe(args, device):
    model, ckpt, cache = load_takeaway(args.run, device, args.cache)
    pack, src = D.pick_episode(cache, -1)
    z, a = pack["z"], pack["a"]
    t0 = args.t0
    hist = ckpt["hist"]
    assert t0 >= hist and t0 + args.branch < len(z)
    scale = D.scale_of(z)
    z0 = z[t0 - hist: t0]
    e0 = pack["e"][t0 - hist: t0] if "e" in pack else None
    trajs = []
    for aid, name in enumerate(NAMES):
        acts = torch.full((args.branch,), aid, dtype=torch.long)
        pred = rollout_any(model, z0, acts, device, e0)
        trajs.append(pred)
        print(name, "steps", pred.shape[0])
    print("比例尺 %.4f  cache %s t0=%d mod=%s" % (scale, src, t0, ckpt.get("mod")))
    for step_i, label in [(0, "第 1 步"), (args.branch - 1, "第 %d 步" % args.branch)]:
        print("\n[%s] 两两距离 / 比例尺" % label)
        print("%12s" % "" + "".join("%12s" % n for n in NAMES))
        for i, ni in enumerate(NAMES):
            row = "%12s" % ni
            for j in range(4):
                d = torch.norm(trajs[i][step_i] - trajs[j][step_i]).item() / scale
                row += "%12.2f" % d
            print(row)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    raw_files = sorted(Path(args.data).glob("ep_*.npz"))
    raw = np.load(raw_files[-1])
    axes[0, 0].imshow(raw["frames"][t0])
    axes[0, 0].set_title("t0 real")
    axes[1, 0].imshow(raw["frames"][min(t0 + args.branch, len(raw["frames"]) - 1)])
    axes[1, 0].set_title("real +%d" % args.branch)
    for i, name in enumerate(NAMES):
        axes[0, i + 1].imshow(D.pca_map(trajs[i][0].numpy()))
        axes[0, i + 1].set_title("%s +1" % name)
        axes[1, i + 1].imshow(D.pca_map(trajs[i][-1].numpy()))
        axes[1, i + 1].set_title("%s +%d" % (name, args.branch))
    for ax in axes.ravel():
        ax.axis("off")
    out = Path(args.run) / "action_swap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close()
    print("对换图", out)


def drift_probe(args, device):
    model, ckpt, cache = load_takeaway(args.run, device, args.cache)
    hist = ckpt["hist"]
    files = sorted(cache.glob("ep_*.pt"))
    horizon = args.dhorizon
    tf = np.zeros(horizon)
    fr = np.zeros(horizon)
    nwin = 0
    scale_acc = []
    for f in files[: args.windows]:
        pack = torch.load(f, map_location="cpu", weights_only=False)
        z, a = pack["z"], pack["a"]
        e = pack["e"] if "e" in pack else None
        if len(z) < hist + horizon + 2:
            continue
        scale_acc.append(D.scale_of(z))
        starts = np.linspace(hist, len(z) - horizon - 2, num=2, dtype=int)
        for t0 in starts:
            with torch.no_grad():
                for k in range(horizon):
                    z_hist = z[t0 + k - hist: t0 + k].unsqueeze(0).to(device)
                    a_hist = a[t0 + k - hist: t0 + k].unsqueeze(0).to(device)
                    if isinstance(model, DeskWMAudio):
                        e_hist = e[t0 + k - hist: t0 + k].unsqueeze(0).to(device)
                        pred = model.one_step(z_hist, a_hist, e_hist)[0].cpu()
                    else:
                        pred = model.one_step(z_hist, a_hist)[0].cpu()
                    tf[k] += torch.norm(pred - z[t0 + k]).item()
                z0 = z[t0 - hist: t0]
                acts = a[t0: t0 + horizon]
                e0 = e[t0 - hist: t0] if e is not None else None
                pred_fr = rollout_any(model, z0, acts, device, e0)
                for k in range(horizon):
                    fr[k] += torch.norm(pred_fr[k] - z[t0 + k]).item()
            nwin += 1
    assert nwin > 0, "轨迹太短，调小 --dhorizon"
    tf /= nwin
    fr /= nwin
    scale = float(np.mean(scale_acc))
    print("窗口数", nwin, "偷懒基线 %.4f" % scale)
    print("第 1 步 teacher=%.4f free=%.4f" % (tf[0], fr[0]))
    print("第 %d 步 teacher=%.4f free=%.4f" % (horizon, tf[-1], fr[-1]))
    plt.figure(figsize=(7, 4))
    ks = np.arange(1, horizon + 1)
    plt.plot(ks, tf, label="teacher forcing")
    plt.plot(ks, fr, label="free rollout")
    plt.axhline(scale, linestyle="--", linewidth=1, label="lazy copy-z")
    plt.xlabel("steps ahead")
    plt.ylabel("L2 in DINO patch space")
    plt.legend()
    out = Path(args.run) / "drift_curve.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print("漂移图", out)


def perm_probe(args, device):
    model, ckpt, cache = load_takeaway(args.run, device, args.cache)
    hist = ckpt["hist"]
    files = sorted(cache.glob("ep_*.pt"))
    p_idx = None
    open_err, mask_err, n = 0.0, 0.0, 0
    for f in files[: max(4, args.windows)]:
        pack = torch.load(f, map_location="cpu", weights_only=False)
        z, a = pack["z"], pack["a"]
        e = pack["e"] if "e" in pack else None
        if p_idx is None:
            p_idx = center_idx(z.shape[1])
        if len(z) < hist + 6:
            continue
        starts = np.linspace(hist, len(z) - 4, num=4, dtype=int)
        for t0 in starts:
            z_hist = z[t0 - hist: t0].unsqueeze(0).to(device)
            a_hist = a[t0 - hist: t0].unsqueeze(0).to(device)
            z_next = z[t0]
            z_masked = mask_center(z_hist, p_idx.to(device))
            with torch.no_grad():
                if isinstance(model, DeskWMAudio):
                    e_hist = e[t0 - hist: t0].unsqueeze(0).to(device)
                    pred_o = model.one_step(z_hist, a_hist, e_hist)[0].cpu()
                    pred_m = model.one_step(z_masked, a_hist, e_hist)[0].cpu()
                else:
                    pred_o = model.one_step(z_hist, a_hist)[0].cpu()
                    pred_m = model.one_step(z_masked, a_hist)[0].cpu()
            cen = z_next[p_idx]
            open_err += torch.norm(pred_o[p_idx] - cen).item()
            mask_err += torch.norm(pred_m[p_idx] - cen).item()
            n += 1
    assert n > 0
    open_err /= n
    mask_err /= n
    print("恒常探针 窗口", n, "mod", ckpt.get("mod"))
    print("中心 patch L2  不遮挡 %.4f  遮中心 %.4f  比值 %.3f" % (
        open_err, mask_err, mask_err / max(open_err, 1e-8)))
    print("基线对照：对 runs/desk1 再跑一次本命令，把 --run 换成基线目录")
    print("slot 档预期：遮挡比值低于基线。sf / audio 不保证下降。")


def encode_audio(args, device):
    enc = D.DinoEncoder().to(device)
    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.data).glob("ep_*.npz"))
    assert files, "没有 npz"
    for f in files:
        dest = cache_dir / (f.stem + ".pt")
        if dest.exists():
            print("跳过", dest)
            continue
        raw = np.load(f)
        if "energy" not in raw.files:
            raise SystemExit("%s 没有 energy 键，先跑 collect_desk_audio.py" % f)
        frames = raw["frames"]
        acts = raw["actions"]
        energy = raw["energy"]
        zs = []
        for i in range(0, len(frames), 8):
            chunk = frames[i: i + 8]
            xs = torch.stack([D.preprocess(im) for im in chunk]).to(device)
            zs.append(enc(xs).cpu())
        z = torch.cat(zs, dim=0)
        a = torch.as_tensor(acts, dtype=torch.long)
        e = torch.as_tensor(energy, dtype=torch.float32)
        T = min(z.shape[0] - 1, a.shape[0], e.shape[0])
        torch.save(
            {"z": z[: T + 1], "a": a[:T], "e": e[:T], "src": str(f)}, dest
        )
        print("编码", f.name, "z", tuple(z.shape), "e_mean", float(e[:T].mean()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mod", required=True, choices=["sf", "audio", "slot"])
    p.add_argument(
        "--mode", required=True,
        choices=["encode", "train", "swap", "drift", "perm"],
    )
    p.add_argument("--data", type=str, default="data")
    p.add_argument("--run", type=str, default="runs/desk45_sf")
    p.add_argument("--cache", type=str, default="")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hist", type=int, default=2)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--sf-prob", type=float, default=0.5, dest="sf_prob")
    p.add_argument("--t0", type=int, default=40)
    p.add_argument("--branch", type=int, default=10)
    p.add_argument("--dhorizon", type=int, default=15)
    p.add_argument("--windows", type=int, default=6)
    args = p.parse_args()
    if not args.cache:
        args.cache = str(Path(args.run) / "cache")
    device = D.device_of()
    print("device", device, "mod", args.mod, "mode", args.mode)
    print("不后训练 Cosmos 3；本脚本只改第 30 课小模型")
    if args.mode == "encode":
        if args.mod != "audio":
            raise SystemExit("encode 只给 audio 档。sf/slot 复用第 30 课 cache")
        encode_audio(args, device)
    elif args.mode == "train":
        train_loop(args, device)
    elif args.mode == "swap":
        swap_probe(args, device)
    elif args.mode == "drift":
        drift_probe(args, device)
    else:
        perm_probe(args, device)


if __name__ == "__main__":
    main()
```

只选声音档的人，还要存采集脚本（sf / slot 跳过 4B）。把下面存成 `~/desk_wm/collect_desk_audio.py`。它在第 30 课 `collect_desk.py` 上多写一个 `energy`：每一帧对麦克风做一段 RMS。按键协议不变。桌上要有能发出撞击的杯子，故意录几段「伸手碰到杯」和几段「只敲键盘、杯子不动」。

```python
"""collect_desk_audio.py  第 45 课声音档：在 collect_desk 上加每帧 RMS。"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import sounddevice as sd

ACTIONS = {ord("a"): 0, ord("d"): 1, ord("w"): 2, ord("s"): 3}
NAMES = {0: "look_left", 1: "look_right", 2: "reach", 3: "stay"}


def chunk_rms(sr, fps):
    n = max(1, int(sr / fps))
    buf = sd.rec(n, samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return float(np.sqrt(np.mean(buf ** 2) + 1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="data_audio")
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--sr", type=int, default=16000)
    args = p.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit("打不开摄像头")
    print("a/d/w/s 粘滞，q 结束本段。碰杯时能量应明显高于发呆。")
    ep = 0
    interval = 1.0 / args.fps
    try:
        while True:
            frames, actions, energies, current = [], [], [], 3
            n_target = int(args.seconds * args.fps)
            t0 = time.time()
            next_t = t0
            while len(frames) < n_target:
                ok, frame = cap.read()
                if not ok:
                    break
                now = time.time()
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key in ACTIONS:
                    current = ACTIONS[key]
                if now < next_t:
                    vis = frame.copy()
                    cv2.putText(
                        vis, NAMES[current], (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
                    )
                    cv2.imshow("desk audio", vis)
                    continue
                rms = chunk_rms(args.sr, args.fps)
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                actions.append(current)
                energies.append(np.log1p(100.0 * rms))
                next_t += interval
                vis = frame.copy()
                cv2.putText(
                    vis, "%s e=%.2f %d/%d" % (
                        NAMES[current], energies[-1], len(frames), n_target),
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )
                cv2.imshow("desk audio", vis)
            if len(frames) < args.fps * 5:
                print("本段太短，丢弃")
                continue
            arr_f = np.stack(frames).astype(np.uint8)
            arr_a = np.array(actions[:-1], dtype=np.int64)
            arr_e = np.array(energies[:-1], dtype=np.float32)
            path = out_dir / ("ep_%03d.npz" % ep)
            np.savez_compressed(
                path, frames=arr_f, actions=arr_a, energy=arr_e, fps=args.fps,
            )
            print("写入", path, "e_mean", float(arr_e.mean()), "e_max", float(arr_e.max()))
            ep += 1
    except KeyboardInterrupt:
        print("停止采集")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

声音档依赖：

```bash
pip install sounddevice
```

### Step 4B: 声音档补采（仅 `--mod audio`）

```bash
python collect_desk_audio.py --out data_audio --fps 5 --seconds 30
```

至少 8 段，其中不少于 3 段包含清晰碰杯。打印的 `e_max` 应明显高于发呆段的 `e_mean`。分不出，麦克风没拾到撞击，后面的通道是死的，可以直接写失败报告，不必硬训。

### Step 5: 训一小截

self-forcing 默认路径。先把第 30 课的 cache 链到新目录，避免 `load` 找不到 `ep_*.pt`，也避免覆盖基线：

```bash
mkdir -p runs/desk45_sf
```

```bash
ln -sfn ../desk1/cache runs/desk45_sf/cache
```

```bash
python desk_takeaway.py --mod sf --mode train --run runs/desk45_sf --cache runs/desk1/cache --epochs 10 --horizon 3 --sf-prob 0.5 --batch 8
```

预期：每个 epoch 打出 `train` / `val`；`val` 是 $p=0$ 的多步 teacher forcing，便于和基线一步 MSE 量级对照，不必相等。`runs/desk45_sf/best.pt` 出现。24GB 卡上 10 epoch 通常十几分钟；Mac 按一到两小时。`sf_prob=0.5` 表示一半时间把真实特征换成自己刚预测的特征再往下走，这是混入，不是论文里的 KV cache 配方。

记忆槽档把目录改成 `runs/desk45_slot`，命令改 `--mod slot`，同样可以链到旧 cache。声音档不要链旧 cache，先编码：

```bash
python desk_takeaway.py --mod audio --mode encode --data data_audio --run runs/desk45_audio --cache runs/desk45_audio/cache
```

```bash
python desk_takeaway.py --mod audio --mode train --run runs/desk45_audio --cache runs/desk45_audio/cache --data data_audio --epochs 10 --batch 8
```

三种命令你只跑一种训练。训练中途验证损失爆炸、或对换还没跑就宣称成功，都不算完成。

### Step 6: 动作对换（基线和新模型各一次）

基线（第 30 课脚本，数字应与 NOTES 里已有表同方向）：

```bash
python desk_wm.py --mode swap --run runs/desk1 --data data --t0 40 --branch 10
```

新模型：

```bash
python desk_takeaway.py --mod sf --mode swap --run runs/desk45_sf --cache runs/desk1/cache --data data --t0 40 --branch 10
```

声音档把 `--mod audio --run runs/desk45_audio --cache runs/desk45_audio/cache --data data_audio` 换上；槽档同理。把第 1 步和第 10 步 `look_left` 对 `look_right` 抄进 NOTES 并排。及格方向：对角线仍为 0；第 10 步分岔不明显小于基线。塌成全 0，改造失败，进入 Step 9 写失败原因，不要再加大 epoch 假装没塌。

### Step 7: 物体恒常（中心遮挡探针）

基线没有 `perm` 模式。`desk_takeaway.py` 加载第 30 课的 `best.pt` 时，若没有 `mod` 键，会按 `sf` 去建 `DeskWM`，和基线权重形状一致，可以直接测基线：

```bash
python desk_takeaway.py --mod sf --mode perm --run runs/desk1 --cache runs/desk1/cache --windows 6
```

新模型：

```bash
python desk_takeaway.py --mod sf --mode perm --run runs/desk45_sf --cache runs/desk1/cache --windows 6
```

你圈中的若是 slot / audio，把 `--mod` 和 `--run`、`--cache` 换成 Step 5 那一档。基线那条仍用 `--mod sf --run runs/desk1`。

打印三行：不遮挡的中心 L2、遮中心的中心 L2、比值。比值接近 1，说明挡住中间几乎不影响预测（模型本来就不看中心，或槽把中心记住了）。比值明显大于 1，说明中心被挡之后预测崩了。slot 档预期新模型比值低于基线。sf / audio 不保证，写「未动」即可，不算这档失败。探针是「挡住输入里的中心 patch」，不是真的用手挡住杯子；真遮挡仍以第 32 课行为 1 的日志为准。本课这个探针只回答：改造有没有让模型在看不见中间时少崩一点。

### Step 8: 漂移曲线（sf 必做，另外两档建议做）

```bash
python desk_wm.py --mode drift --run runs/desk1 --horizon 15 --windows 6
```

```bash
python desk_takeaway.py --mod sf --mode drift --run runs/desk45_sf --cache runs/desk1/cache --dhorizon 15 --windows 6
```

把第 1 步、第 10 步、第 15 步的 teacher / free、以及越过偷懒基线的步数并排。sf 档的主验收在这里：自由曲线若比基线更晚越过偷懒线，或第 10 步 free 误差下降，记「漂移改善」。若第 1 步已经分家且 free 更差，检查 `sf_loss` 是不是把 `pred` 在 `p=1` 时训崩了，把 `--sf-prob` 降到 0.3 再训一小截；仍更差，写失败。audio / slot 的漂移不是主验收，变差超过 20% 则记副作用。

### Step 9: 改造记录或失败报告

在 `runs/desk45_NOTES.md` 末尾按这个骨架写，缺命令的报告不算完成：

```text
选择器：我圈的目标；四根尺子猜测与揭晓；我真正在乎的尺子。
只选一个：sf / audio / slot（圈中的那一个）。另外两个写「未做」。
现状表：Step 1 已填。
命令：训练、swap、perm、drift 的完整命令，含 --run 与 --cache。
基线数字：对换第 10 步 look_left-look_right；漂移越过偷懒的步数；遮挡比值。
新模型数字：同上三行。
结论：哪根尺子动了，哪根没动。动了也不写「已毕业」。
失败（若发生）：对换塌 / 漂移更差 / 能量常数 / 遮挡比值差不到 5%。可能原因与下一步。
不是本课产物：Cosmos 3 后训练、WorldScore 分数、没跑的另外两个改造。
毕业标准：仍在第 32 课五件行为与第 33 课 E 档。第九幕是读榜和选零件。
```

数字没动，把「失败」那一段写完整，本课通过。把未做的 audio 写成「已加入声音通道」，本课不通过。

## 8. 配置与预算

| 项 | 本课默认 | 说明 |
|---|---|---|
| 基线 | 第 30 课 `runs/desk1/best.pt` 只读 | Genesis 档换 `gen1`；不覆盖 |
| 改造数量 | 恰好 1 | sf / audio / slot 三选一 |
| self-forcing | `--sf-prob 0.5`，`--horizon 3`，10 epoch | 验证损失用 $p=0$ 的多步 TF，不和基线一步 MSE 强行比绝对值 |
| 记忆槽 | 中心遮挡 + $0.3$ 的槽辅助损失，10 epoch | 主损失仍是特征 MSE |
| 声音能量 | 补采 ≥8 段，RMS 经 `log1p(100 e)` | 旧 `data/ep_*.npz` 没有 `energy`，不能直接训 audio |
| 优化 | AdamW，lr $3\times 10^{-4}$，batch 8 | 与第 30 课相同，便于对照 |
| 编码 | sf / slot 不重做 encode | audio 必须 `--mode encode` 写 `e` |
| 24GB 卡 | 10 epoch 约 10 到 20 分钟 | 含探针一共半天 |
| Mac / CPU | 10 epoch 约 1 到 2 小时 | 选择器和失败报告不需要 GPU |
| 新目录磁盘 | <1GB | cache 用符号链接，不要复制 DINOv2 特征 |
| 明确不做 | Cosmos 3 后训练、桌面 DiT、完整 WorldScore / Physics-IQ | 24GB 主线带不走 |

`hist` 必须与基线检查点一致。第 30 课默认 2；你若当时改成 3，本课 `--hist 3`，否则 `load_state_dict` 会在位置编码或窗口上对不上。`best.pt` 里的 `hist` 在 Step 1 已经打印过。

预算优先花在对照上，不花在加层。10 个 epoch 叫一小截，就是为了防止你把第 30 课整段重训当成「第九幕升级」。还想加钱时，先把 sf_prob 做成 0.3 / 0.5 / 0.7 三格，而不是把 ViT 加到 6 层。官方 Self-Forcing 仓库和 Wan 教师都不是本课训练对象；第 37 课若跑过它们的推理命令，把失败或显存记录链到 NOTES，不要把那次推理写成第 30 课改造。

## 9. 验收

及格看三样东西：现状表、选择器、一份带命令的改造记录或失败记录。生成视频好看、Cosmos 3 的广告词、没跑的另外两个零件，都不加分。

- [ ] Step 1 现状四格能指向 `desk_wm.py` 的具体类或行：目标、记忆、动作编码、声音通道。
- [ ] 选择器先猜后揭晓，四根尺子没有全部写成「可能变好」。
- [ ] 只训了一个 `--mod`。NOTES 里另外两个写「未做」。
- [ ] 新检查点在 `runs/desk45_*`，第 30 课 `runs/desk1/best.pt` 仍在且未被覆盖。
- [ ] 动作对换：基线和新模型各有一张表；新模型对角线为 0；若塌成全 0，失败报告写了原因，没有改口称成功。
- [ ] 物体恒常：基线和新模型各有遮挡比值；slot 档比值应低于基线，否则写入失败；sf / audio 写「未要求下降」也可以过。
- [ ] sf 档交了漂移对照；自由曲线若变差，有失败段。
- [ ] 报告里有完整命令，没有 Cosmos 3 后训练，没有编造 WorldScore 分数。
- [ ] 能口头说：毕业在第 32 课，档在第 33 课，第九幕是读榜和选零件。

口头抽查三句。第一句：第 30 课现在的损失是什么，不是「世界模型损失」这种空话。第二句：你选的改造预期动哪根尺子，哪根你事先就说了不动。第三句：为什么 WorldScore 第一不能代替动作对换。答不上，回去第 5 节，不要改 NOTES 里的数字。

## 10. 排错

| 症状 | 最可能的原因 | 怎么确认 | 怎么修 |
|---|---|---|---|
| `No module named desk_wm` | 当前目录不是 `~/desk_wm` | `pwd` | 进入该目录再跑 |
| `best.pt` 的 `load_state_dict` 报 unexpected key `gru` 或 `energy_head` | 用错了 `--mod`，或拿槽/声音的权重去建 `DeskWM` | 打印 `ckpt['mod']` | `--mod` 与训练时一致；基线 perm 用 Step 7 那份拷了 `mod=sf` 的副本 |
| cache 里找不到 `ep_*.pt` | 新 `run` 目录没有链到旧 cache | `ls runs/desk45_sf/cache` | 重做 Step 5 的 `ln -sfn` |
| `audio 档需要 cache 里有键 e` | 用旧 cache 训了 `--mod audio` | `python -c` 里 `torch.load` 打印 keys | 走 Step 4B 和 `--mode encode` |
| `sounddevice` 报无输入设备 | 麦克风权限或没有默认输入 | 系统设置里看终端/Python 是否被允许 | Mac 给终端麦克风权限；仍不行就改选 sf，把 audio 记未做 |
| 能量几乎常数 | 没采到撞击，或 RMS 太小被 `log1p` 压平 | 对比发呆段和碰杯段的 `e_max` | 把麦靠近杯子；分不出就写失败，不要硬训 |
| 对换全表贴 0 | `sf_prob` 太大把动作信息冲掉，或训崩 | 看 val 是否从第一个 epoch 就爆炸 | `--sf-prob 0.3` 重训；仍塌则失败报告，回滚基线 |
| 自由曲线比基线更差 | 混入过早、horizon 过长、数据太懒 | 对比第 1 步 teacher 是否也变差 | 降 `sf_prob` 或 `horizon`；第 1 步也差说明连一步都毁了 |
| 遮挡比值与基线差不到 5% | 槽没学到东西，或中心 patch 本来就不含杯子 | 看 PCA 里杯子是否在画面中间 | 换 `t0` 到杯子居中的帧；仍无差则 slot 档失败 |
| `LongCache` 报太短 | 某段帧数 < hist+horizon | 打印各 `ep_*.pt` 的 `z.shape[0]` | `--horizon 2`；或丢掉过短的段 |
| MPS 上 ViT 报错 | 与第 30 课相同的设备坑 | 同一脚本设 CPU 对比 | 训练走 CPU；本课不改 CausalViT |
| 符号链接在 Windows 失败 | 本课默认 POSIX | `ls -l cache` | 把 `--cache` 显式指到 `runs/desk1/cache`，不必链接 |
| 把 Cosmos 3 推理当本课产物 | 读错第九幕收官 | NOTES 里出现 16B/64B 或后训练 | 删掉，只保留第 30 课小模型的数字 |

真机额外一条：本课不要把 Reachy Mini 的麦阵列「顺便」接进训练，除非你选了 audio 并且日志里的 `energy` 来自那只麦。身体接口仍归第 32 课。

## 11. 前沿与改造

前沿怎么做。同一问题（训练目标和推理分布、观察通道、离开视野），2025-2026 年公开系统分三摊。生成式视频世界模型用 DMD、CausVid、Self-Forcing 把双向教师变成可流式的因果学生，WorldScore 上比的是生成真。Physics-IQ 用真实验续写打分，结论是画面真和物理对可以分家。机器人或桌面尺度上，DINO-WM 仍然是特征 MSE 加动作拼接；V-JEPA 2-AC 是表征距离规划；C-SWM 是对比能量。没有哪个前沿系统因为换了目标函数，就自动通过第 12 课的三问。

我们差在哪。规模差：没有教师 DiT，没有小时级桌面撞击数据，没有按位姿取回的记忆库。钱能买卡和数据，买不来第 30 课这份已经能对换的小模型之外的另一份毕业。机制差：曝光偏差、没有声音观察、滑窗会忘，三件事本课都给了最小补丁。补丁的机制来自第九幕，规模故意保持在 2 层 ViT。

动手改造清单（选做，且仍遵守「一次一块」；本课主实验已经用掉一块配额）：

1. sf_prob 扫描。在已经做完 `sf` 的前提下，把 `--sf-prob` 设成 0.3 和 0.7 再各训 10 epoch，目录分开。预算：24GB 各十几分钟。预期：0.7 更伤对换、更利多步漂移，或两头都伤。失败：三格数字无差异，说明混入没生效，查 `coin * pred.detach()` 是否真的接到 `z_win`。
2. 真遮挡对照。槽档若遮挡探针下降，补采 5 段「手挡杯 2 秒再拿开」，用第 32 课行为 1 的日志格式看头是否还朝估计位置。预算：半天采集加前向，不重训。预期：探针下降但真遮挡仍失败，因为中心 patch 不等于杯子槽。失败：连探针都没降还去采真遮挡，先停。
3. 声音负对照。audio 档若能量有变化，把测试时的 `e` 全部置零再跑 swap / perm。预算：一次前向。预期：置零后若数字几乎不动，模型没听能量。失败：置零更差但训练数据里能量与动作完全共线（只在伸手时出声），那是标签泄漏，写进报告。
4. 顺手复现映射。Self-Forcing 论文结论「训练按推理配方滚动，能缩小 train-test 分布差」。缩小版对应 Step 8 的自由曲线。预期能看到同方向（裂口缩小），不能预期看到他们的视频 FVD。DINO-WM「特征空间动作条件预测足以让动作分岔」对应 Step 6，本课改造后应仍成立；若不成立，优先判改造失败，不要判第 30 课复现作废。

不要做的改造：后训练 Cosmos 3；把 `CausalViT` 换成公开 DiT 权重再声称桌面世界模型；把 WorldScore 官方排行抄进 NOTES 当自己的结果。[第 19 课](19_surgery_experiments.md) 的手术纪律在第九幕收官仍然有效。

## 12. 论文与延伸

只回访已授课和本课动物园用过的文献。不新开未核论文。读的时候带着「赌什么 / 哪根尺子」这个问题，不要当第九幕书单复读。

1. [第 02 课](02_vae_visual_compression.md) 与 World Models（Ha & Schmidhuber, 2018，[arXiv:1803.10122](https://arxiv.org/abs/1803.10122)）。带着问题读：ELBO 的重建项和 KL 项各在保护哪根尺子？弯道探针和重建损失分家，对应本课选择器里像素重建的哪一格？
2. [第 07 课](07_muzero_value_equivalence.md) 与 MuZero（Schrittwieser 等，[arXiv:1911.08265](https://arxiv.org/abs/1911.08265)）。带着问题读：价值等价探针若读不出杯子外观，克制还能不能做？桌面没有自博弈奖励时，这一格为什么搬不回第 30 课？
3. [第 09 课](09_iris_token_world_model.md) 与 IRIS（Micheli, Alonso, Fleuret，[arXiv:2209.00588](https://arxiv.org/abs/2209.00588)）。带着问题读：交叉熵下降和动作对换哪一个先坏？码本死码会让选择器里「可能变好」的 WorldScore 变成什么？
4. [第 10 课](10_diamond_diffusion_world_model.md) 与 DIAMOND / EDM。带着问题读：少步去噪在想象 rollout 里崩，和 teacher forcing 的曝光偏差是不是同一件事？若不是，本课 sf 档治的是哪一件？
5. [第 13 课](13_ijepa_from_scratch.md) 与 I-JEPA。带着问题读：常数解是目标函数里的合法满分。第 30 课冻结 DINO 的 MSE 有没有这条满分通道？为什么本课没有把目标改成「纯 JEPA」？
6. [第 16 课](16_three_roads_debate.md) 与 [第 17 课](17_evaluating_world_models.md)。带着问题读：四轴里本课动的是哪一根？三分评测里 WorldScore 属于哪一种「好」？动作对换属于哪一种？
7. [第 21 课](21_persistent_4d.md) 与 [第 32 课](32_ship_desk_pet.md)。带着问题读：CUT3R 的持久状态和第 30 课 `hist=2` 差在有没有动作？本课记忆槽若只降低了中心遮挡 L2，能不能写进第 32 课行为 1 的对照表？
8. [第 23 课](23_object_centric_wm.md) 与 C-SWM（Kipf 等，[arXiv:1911.12247](https://arxiv.org/abs/1911.12247)）。带着问题读：对比能量和本课槽档的 GRU 辅助损失是不是同一件事？差在负样本和物体身份。不要把 GRU 槽写成已经做了 C-SWM。
9. DINO-WM（Zhou 等，[arXiv:2411.04983](https://arxiv.org/abs/2411.04983)）。第 30 课架构出处。带着问题读：他们把解码器隔离在 `detach` 之后，本课有没有偷偷把像素项加回去？
10. Flow matching（Lipman 等，[arXiv:2210.02747](https://arxiv.org/abs/2210.02747)）。第 37 课配方。带着问题读：速度场目标和特征 MSE 各在惩罚什么？24GB 桌上为什么不把第 30 课改成 flow matching？
11. One-step Diffusion with Distribution Matching Distillation（Yin 等，[arXiv:2311.18828](https://arxiv.org/abs/2311.18828)）。带着问题读：DMD 要教师 score。你的桌子上教师是谁？没有教师时，本课为什么只混入 self-forcing 而不写「完成了 DMD」？
12. Self Forcing（Huang 等，[arXiv:2506.08009](https://arxiv.org/abs/2506.08009)）。带着问题读：论文的 KV cache 自回归和本课 `sf_loss` 里 `pred.detach()` 混入，相同的是哪一句，不同的是哪一句？把 FVD 数字抄进桌宠 NOTES 犯了第 17 课哪条纪律？
13. Physics-IQ（Motamed 等，[arXiv:2501.09038](https://arxiv.org/abs/2501.09038)）与 WorldScore（Duan 等，[arXiv:2504.00983](https://arxiv.org/abs/2504.00983)）。第 34、44 课的尺子。带着问题读：两份榜的输入输出和评分器各是什么？为什么第一名不必能给桌宠做 MPC？分数只许抄论文表或仓库 README，不许编。

[第 33 课](33_embodiment_degrees.md) 若还没打分，做完本课仍按第 32 课日志打，不要因为混入了 self-forcing 就加一档。没有新的查询记录，E4 不成立。升一档只选一个实验，那是第 33 课的规矩，本课的一块零件最多成为那个实验的候选，不是自动升级。

零件盘点。观察进状态是第 29 课；动作条件动力学是第 30 课；人作为外生过程是第 31 课；身体、安全层、五件行为是第 32 课；E 档是第 33 课。第九幕从第 34 课读到本课，往这台机器上最多拧进一颗螺丝：训练时吃自己的预测、或一个很粗的声音能量、或一个离开视野的槽。刷榜知识压回主线之后，尺子仍是动作对换、物体恒常、以及第 32 课对照表上的克制。WorldScore 和 Physics-IQ 用来防止你被生成真收买，不用来给桌宠发毕业证。

下一课不再把毕业标准改写一遍。若还要做，回到你第 32 课的 `NEXT.md`：接触、人预测、长时记忆、真机延迟，四选一。本课若已经用记忆槽碰过长时记忆，就不要在同一周再把「长时记忆」写成新的真瓶颈。第九幕收官。毕业标准仍在第 32 课。具身程度仍在第 33 课。第九幕是读榜和选零件。




