---
id: 30_desk_world_model
title: "给自己的桌子训一个会听动作的世界模型"
summary: "同一张桌子，头左转和手往杯方向伸，未来必须不一样。"
unit: deskpet
play_tools: []
checkpoints:
  - "动作对换报告（论文复现 #9）。"
  - "多步漂移图和动作盲负对照。"
---

# 第 30 课：给自己的桌子训一个会听动作的世界模型

> 类型：复现（论文复现 #9：自制桌面世界模型，动作对换必须分岔）<br>
> 建议周期：3-5 天（采集半天到一天，训练数小时，探针与互动一天）<br>
> 硬件：单张 24GB 卡全程够用；Mac / 纯 CPU 可完成采集、缩小档训练和全部探针，训练慢数倍<br>
> 锚定仓库：[gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm)（DINO-WM 架构复用，不新发明），对照 [NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch) 的 `networks.py::RSSM` 与 [tkipf/c-swm](https://github.com/tkipf/c-swm) 的 `modules.py::ContrastiveSWM`；数据来自你自己的桌面摄像头，或 [Genesis-Embodied-AI/genesis-world](https://github.com/Genesis-Embodied-AI/genesis-world) 桌面场景<br>
> 产物：带动作标签的桌面轨迹、训好的动作条件世界模型、动作对换对照报告、多步漂移曲线、至少一种负对照、同一历史帧上四键 2 秒想象与真实后续的并排记录

## 1. 这一课做什么

第八幕是毕业设计：在一张桌子上做出会看、会想、会克制的桌宠。第 28 课把桌子写成一个有界的决策问题，第 29 课把乱像素收成状态（杯子在不在、人看不看镜头、头朝哪）。状态还只是现在。桌宠要克制，必须先能回答另一句话：如果我现在转头看左，两秒后桌上会怎样；如果我伸手去够杯子，又会怎样。两份未来必须不一样。这一课给桌子装上小脑，学的就是条件分布

$$
P(s_{t+1} \mid s_t, a_t)
$$

贯穿 32 课的循环里，本课换的是中间那一截：观察已经压成状态，现在要按动作预测下一状态。没有它，第 32 课的五件行为只能靠规则硬编；有了它，克制才是查询模型之后的决定，不是 `if` 写死的表情包。

架构不许新发明。从你第 05、22、23 课真正跑通过的三条里选一条：RSSM（循环状态空间模型，第 05 课）、DINO-WM 式特征动力学（冻结 DINOv2 的空间 patch，在特征上预测下一步）、或 C-SWM 式槽动力学（每个物体一个槽，图网络做转移）。本课默认走 DINO-WM 式，原因很具体：你自己桌子上的轨迹通常只有几十分钟，从零训编码器会把容量浪费在重建桌面纹理上；DINOv2 的 patch 已经带了空间结构，官方 DINO-WM（Zhou, Pan, LeCun, Pinto, arXiv:2411.04983）证明可以在离线轨迹上训动作条件预测，再拿去规划。官方仓库 `gaoyuezhou/dino_wm` 的训练入口绑死了他们自己的 Hydra 环境与 PushT / PointMaze 数据格式，不能直接吞你的摄像头录像。本课给出一份缩小版胶水脚本，零件一一对应他们的 `models/dino.py`、`models/vit.py`、`models/visual_world_model.py`，损失仍然是特征空间的动作条件预测，不是像素重建。

论文复现 #9 的验收不看你是否刷出 DINO-WM 论文里六个环境的规划成功率。它看三件事：同一段历史只换动作，想象必须分岔；自由滚动若干步，误差曲线必须画得出来；打乱动作或把动作置零之后，预测必须变差或分岔塌掉。三者缺一，模型就还只是个惯性外推器。

摄像头档没有手臂也能做完：键盘或鼠标充当假身体，你按「看左 / 看右 / 伸手 / 不动」的同时，自己转镜头或伸手。真机档把按键换成关节命令。没有桌子可拍、或想先在可复现的物理里练手，用 Genesis 搭一张桌子加一个方块，脚本策略自己采。三条数据通路训出来的是同一个公式。

术语速查：

| 术语 | 一句人话 |
|---|---|
| 桌面世界模型 | 吃当前桌面状态和你打算做的动作，吐出下一步状态的分布或点估计 |
| 动作条件 | 预测必须随动作变；同一现在，看左和伸手不能给出同一份未来 |
| 动作盲 | 网络表面上接收动作，计算里没给它实权；平均误差看不出来，对换一测一个准 |
| 特征动力学 | 不预测像素，预测冻结视觉骨干吐出的 patch 特征；DINO-WM 的做法 |
| 分岔指数 | 同一起点换动作后，两份想象在特征空间里离开多远，再用真实一步位移当比例尺 |
| 自由 rollout | 把模型自己的预测喂回去继续滚，用来量误差雪球和第 32 课能信几步梦 |
| 负对照 | 故意破坏动作信息（打乱、置零）再测；对照不过关，主实验的分岔不能算数 |
| 假身体 | 摄像头档用键盘或鼠标冒充头和手；标签是按键，物理后果是你自己做出来的 |
| DINOv2 | 无需标注训出来的视觉骨干，本课冻结它，只当编码器（Oquab et al., arXiv:2304.07193） |
| 论文复现 #9 | 课程承诺的第九项复现：自制桌面设置上，动作对换必须分岔，不追论文表里的规划数字 |

## 2. 问题

第 03 课在 CarRacing 上已经证明过一件让人不舒服的事：赛道惯性很大，一个完全无视方向盘的模型，靠「下一帧约等于这一帧」也能把平均误差刷得很体面。桌子比赛道更懒。你盯着杯子坐着，连续几十帧几乎不动；手伸过去的那半秒，才是动作真正改写画面的窗口。平均一步误差会把这半秒稀释掉。所以本课的第一个问题不是「损失降了没有」，而是「同一段历史，看左和伸手，想象分不分叉」。

第二个问题是动作标签从哪来。游戏里环境每步都给你一个向量。桌子上，摄像头档没有关节编码器。本课把动作收成四个离散符号：看左、看右、伸手、不动。你按键的同时必须真的转镜头或伸手，键是标签，身体是执行器。标签和物理对不上，模型学到的就是噪声。Genesis 档没有这个对账问题，相机位姿和方块位移由仿真器写进日志，适合做可复现的对照。

第三个问题是一步准和两秒准不是一回事。互动实验要播 2 秒想象。5 帧/秒采的话，那是 10 步自由滚动；10 帧/秒就是 20 步。第 03 课的 teacher forcing 损失只保证一步。本课必须把两条曲线画在同一张图上：每步喂真实历史，和持续吃自己的预测。裂口就是第 32 课规划视野的上限。

第四个问题是负对照。分岔图很好看，仍可能是「不同动作对应了训练里不同的时间段」，模型其实在认时钟，没在认动作。把测试时的动作打乱、或全部置零，分岔应当塌掉、多步误差应当变差。塌不了，说明你测到的不是动作条件。

界限先划清。本课是复现档，对象是你自己的桌子或 Genesis 桌面场景，对齐的是「动作条件动力学」这个方向，不是 DINO-WM 论文表 1 的零样本规划成功率。官方 `dino_wm` 可以 clone 来对照文件，他们的 `train.py` 依赖 Hydra、Accelerate、WandB 和自己的数据集目录，本课不把它改造成摄像头管道，也不假装在 H100 上复现了原文。解码器如果要训，只许接在特征预测之后当可视化，不许用「重建很像」代替动作对换。禁止把一段桌面视频丢给 GPT-4o 让它用文字描述未来，再把这段文字叫做动力学。

## 3. 准备

- 第 03 课的动作对换和漂移曲线画法会原样搬过来，只是空间从 32 维 $z$ 换成 DINOv2 的 patch 特征。第 05 课 RSSM 的先验 / 后验分工，本课闭眼滚动时用得到。第 29 课如果已经留下桌面视频和状态日志，本课复用画面，补上同步的动作通道即可；没做过第 29 课，按第 7 节 Step 1 现采。
- Python 3.10 左右，已装 PyTorch。再装 `opencv-python` 和 `matplotlib`。DINOv2 权重走 `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")`，首次运行会下载 ViT-S/14，大约几十到一百兆字节量级，以 Hub 实际文件为准。
- 一块能看全桌面的摄像头：笔记本自带即可。桌上固定放一只杯子（或水瓶）和一部手机，给伸手动作一个明确目标。灯光尽量稳定，不要让自动曝光把「转头」学成「画面变亮」。
- 磁盘留 5GB：5 帧/秒、640×480、40 段各 30 秒，未压缩会到数 GB；脚本按 JPEG 或缩小后的数组存，通常小得多。
- 可选：`pip install genesis-world`（官方要求 Python ≥3.10 且 <3.14，先按 [pytorch.org](https://pytorch.org/get-started/locally/) 装好 PyTorch）。Genesis 用来生成可复现的桌面轨迹，不替代你自己的桌子；有真机（Reachy Mini 或 SO-101）的人把按键换成关节命令，数据格式与摄像头档相同。
- 读代码用的对照仓库本课只要求 clone，不要求按其 README 在 Mujoco 上训完 PointMaze。想顺手跑官方训练命令的人，硬件和数据按他们 README 自行准备，那是加餐，不是复现 #9 的必做项。

## 4. 学习目标

1. 白纸写出桌面世界模型的接口：状态从哪来、四个离散动作各自对应什么物理事件、训练损失在哪个空间里算、解码器（如果有）为什么不许当主损失。
2. 对照 `dino_wm` 的三个文件，指出胶水脚本里编码器、动作拼接、预测器各抄了哪一段，以及官方 `models/vit.py` 把注意力掩码写死到 CUDA 为什么不能原样搬到 Mac。
3. 独立跑通动作对换：同一历史、四键分岔，对角线为 0，看左对看右的距离随步数增大；说得出为什么平均特征误差查不出动作盲。
4. 画出 teacher forcing 与自由 rollout 的漂移曲线，读出「2 秒想象」落在曲线的哪一段，并据此写下第 32 课不该超过的规划视野。
5. 完成至少一种负对照（测试时打乱动作、测试时动作置零，或重训一个动作置零模型），解释对照和主模型之间必须出现的差异。
6. 在同一历史帧上按四键各播 2 秒想象，与真实后续并排，给「会不会碰杯」打分，并写明模型对哪一类动作是盲的。

## 5. 原理

五个机制。每个仍按第 01 课的节奏走：为什么需要、怎么运转、精确定义、代码落在哪、怎么证明做对了。

### 5.1 同一张桌子，两个动作必须通向两个未来

倒车前你会在脑子里分两条路：方向盘左打，车尾往哪；右打，又往哪。桌宠的「倒车」是伸手。杯子离桌沿 8 厘米，手再往前 5 厘米，下一秒可能没事，也可能把水扫下去。世界模型要做的，就是在真的伸手之前把这两条未来分开。如果模型无论你按哪一个键都播同一段「杯子还在原处」的录像，第 32 课的克制按钮就没有查询对象。

最小定义没有变，和第 01、03 课同一句话：吃当前状态和打算执行的动作，吐出对下一状态的预测。变的是世界域。CarRacing 的动作是连续的转向、油门、刹车；桌子上我们把身体收成四个离散符号，记 $a\in\{0,1,2,3\}$，对应看左、看右、伸手、不动。状态 $s_t$ 不再是 VAE 的 32 个数，默认是冻结 DINOv2 对当前帧算出的 patch 特征 $z_t\in\mathbb{R}^{N\times D}$（ViT-S/14、输入 224×224 时 $N=256$，$D=384$）。模型学

$$
\hat{z}_{t+1} = f_\theta(z_{t-H+1:t},\, a_{t-H+1:t})
$$

$H$ 是历史帧数，本课缩小档取 $H=2$。训练目标是特征空间的均方误差，不是像素：

$$
\mathcal{L}_{\mathrm{dyn}} = \lVert \hat{z}_{t+1} - z_{t+1} \rVert_2^2
$$

$z_{t+1}$ 来自下一帧真实画面过同一套冻结编码器，对预测器来说是常数。官方 DINO-WM 在 `models/visual_world_model.py::VWorldModel.forward` 里把这项写成 `emb_criterion`（也是 MSE），并且明确把解码器损失接在 `z_pred.detach()` 后面，重建梯度不许回流进预测器。本课沿用这个隔离。

类比到此失效的地方要说清。飞行员的模拟器是工程师按物理造的，误差有界；这个 $f_\theta$ 是从你几十分钟录像里拟合出来的。数据里没出现过「伸手把杯推到桌沿」，模型就不会诚实地说「我会把水扫下去」，它更可能继续播杯子静止的平均未来。第 04 课策略专挑模型学错的地方钻，第 32 课的安全层必须假定梦会撒谎。

验证这件事的方法不是看 $\mathcal{L}_{\mathrm{dyn}}$ 降了多少，是 5.3 节的对换。损失下降只说明模型在训练分布上能跟手。

### 5.2 三条旧架构，选你跑通过的那一条

本课禁止为桌子新发明第四种世界模型。三条候选的赌法不同，数据胃口也不同。

DINO-WM 式特征动力学（默认）。赌的是：一个在海量无标签图上预训练的空间特征，已经够当桌面的状态，只要再学「特征怎么随动作走」。官方实现分四段：`models/dino.py::DinoV2Encoder` 调 `torch.hub` 加载 `dinov2_vits14`，取出 `x_norm_patchtokens`；`models/visual_world_model.py::encode` 把动作和本体感觉沿特征维拼到每个 patch 上（配置里 `concat_dim: 1`）；`models/vit.py::ViTPredictor` 是带因果掩码的 ViT，把 $H$ 帧的所有 patch 展平后预测下一组特征；解码器可选，只为看图。论文强调离线轨迹就能训、测试时用 CEM 在特征里规划，不需要奖励模型和专家演示。本课缩小档把 ViT 从官方配置的 6 层 16 头（`conf/predictor/vit.yaml`）收到 2 层 4 头，历史从 3 帧收到 2 帧，去掉本体感觉通道（摄像头档没有关节角）。损失、动作拼接、冻结编码器这三件事保持原样。

RSSM 式双通道（备选）。赌的是：桌子也有「记不住」和「赌不准」两个病，确定通道 $h_t$ 记历史，随机通道 $z_t$ 处理「手会不会真的碰到杯」这种分叉。代码落点是第 05 课精读过的 `networks.py::RSSM.img_step` / `obs_step`。它更适合你已经有数小时带动作轨迹、并愿意训编码器的情况。桌面缩小数据上，随机通道很容易被 KL 压死，free nats 和 KL balancing 的调法要原样搬过来，本课不把这条当默认，避免你把时间花在重调 PlaNet 超参上。

C-SWM 式槽动力学（备选）。赌的是：杯子、手、手机应该是不同的槽，转移走图网络。代码落点是 `modules.py::ContrastiveSWM` 和 `TransitionGNN`：每个物体一个向量，边 MLP 建模关系，动作拼到节点上，损失是对比能量而不是重建。它对「杯子倒了、手机没动」这种问题比一条向量诚实。真实桌面的槽发现比 2D Shapes 难一个数量级，官方 `train.py` 的小环境设定不能直接吃 224 的 RGB 录像。第 23 课如果已经在 C-SWM 的 Shapes 上跑通过，可以把槽数设成 4（杯、手、手机、背景）试一次；失败了如实写，回到 DINO-WM 式。

三条路的验证协议相同：对换、漂移、负对照。换架构不换尺子。

### 5.3 动作对换：平均误差是钝器，分岔才是利器

第 03 课的审讯工具原样有效，只改空间。固定历史特征 $z_{\leq t_0}$ 和隐状态（ViT 没有 RNN 隐状态，历史窗口本身就是记忆），只替换从 $t_0$ 起的动作，各自向前滚 $n$ 步。前向是确定性的（DINOv2 冻结、预测器 `eval()`、不用 dropout），输出有差异只可能来自动作。

四个探针动作就是互动实验的四键。看左持续 $n$ 步，画面里的杯子应该往画面右边走（镜头左转，物体相对右移）；看右相反；伸手应该让「手」对应的 patch 发生变化；不动应该几乎贴着惯性。一步之内桌面几乎不动，所以必须按住同一动作滚够 $n$ 步。本课取 $n=10$，对应 5 帧/秒下的 2 秒，正好覆盖互动实验的时程。

分岔指数沿用第 03 课的比例尺。记 $\hat{z}^{(i)}_n$ 为动作 $a^{(i)}$ 分支第 $n$ 步的预测特征（把 $N\times D$ 展平），分母是这条轨迹上真实相邻帧的平均特征位移：

$$
D_n = \frac{\mathrm{mean}_{i \neq j} \,\lVert \hat{z}^{(i)}_n - \hat{z}^{(j)}_n \rVert_2}{\mathrm{mean}_t \,\lVert z_{t+1}-z_t \rVert_2}
$$

动作盲模型的 $D_n$ 贴着 0。健康模型的 $D_n$ 随 $n$ 增长，2 秒后看左对看右应是全表最大的一格，量级至少和 1 可比：换一个动作造成的未来差异，赶上世界自己走一步。对角线必须恰好为 0，这一格不为 0，先修探针。

官方仓库里对应的操作是 `VWorldModel.replace_actions_from_z`：滚动时每步用新动作的嵌入覆盖拼接在 patch 尾部的那几维，再送进 `predict`。胶水脚本的 `rollout` 按同一方式写。数据对齐也和第 03 课同一个坑：`actions[t]` 必须是作用在 `frames[t]` 上、产生 `frames[t+1]` 的那个键。摄像头脚本按「先读键、再抓帧」存盘；自己改采集循环时错一位，对换会变成对「上一帧的键」做实验，分岔会被稀释。

### 5.4 负对照：打乱和置零各打一种假

对换表好看，仍有两种假阳性。

第一种：模型没听动作，但听时间。训练里你总是「先看左 10 秒，再伸手 10 秒」，模型学会了「第 80 帧该伸手」。测试时如果你仍按时间顺序喂动作，它会碰巧分岔。打乱动作把 $(s_t, a_t, s_{t+1})$ 的动作维换成同 batch 里另一个时刻的 $a$，时间结构还在，动作对齐没了。健康模型的一步误差应明显上升；动作盲模型几乎不动。

第二种：模型把「有一个非零动作向量」当成开关，并不区分看左和看右。动作置零（或全部喂「不动」）之后，四个分支应塌成同一条；若看左和看右在零动作下仍分岔，分岔来自随机种子或历史本身，不是动作。

两种对照都可以只在测试时做，不必重训。更狠的一版和第 03 课改造清单相同：重训一个 `a = 0` 的模型，主损失可能只差一点点（桌子太懒），对换表却必须塌。本课把测试时打乱和置零列为必做，重训列为加分。只做重建、不做这些对照，验收直接判负。

### 5.5 多步漂移，以及为什么重建不是世界模型

训练损失是 teacher forcing：每一步的输入特征来自真实画面。上场播 2 秒想象时，第 2 步起输入来自自己的 $\hat{z}$。误差有两个通道：特征本身偏了，以及窗口里的历史被污染。定义两条曲线，窗口从多段轨迹平均：

- teacher forcing：每步喂真实 $z_t$，记 $\lVert \hat{z}_{t+1}-z_{t+1}\rVert$；
- 自由 rollout：从 $t_0$ 起喂自己的预测，动作仍用真实日志里的动作，只换状态来源。

$k=1$ 处两条曲线必须重合。之后自由曲线上扬。再画一条偷懒基线：永远猜 $z_{t+1}=z_t$，其误差等于比例尺分母。teacher forcing 若高于这条线，模型白训了。自由曲线穿过基线的步数，就是「梦还能当尺子」的有效长度。2 秒想象落在这条线右边，第 32 课就不得把规划视野开到 2 秒以外还装成有把握。

解码器在这里只是显示器。你可以在冻结的 $\hat{z}$ 上训一个线性头，把每个 patch 映回 14×14 的 RGB 小块再拼成图，方便肉眼看对换。也可以不做解码，把 384 维 patch 做 PCA 取前三个分量当伪彩图。两种图糊都不要紧，方向要分得开。如果你发现自己在加大解码器、调感知损失、却还没跑对换，停下来。那是在训一个桌面自动编码器，公式里的 $a_t$ 被你关掉了。

用 GPT-4o 一类多模态语言模型看当前帧、写一段「接下来手会碰到杯子」，再把这段话叫做 $P(s_{t+1}\mid s_t,a_t)$，同样不算。语言模型没有和你的动作通道对过账，也没有一条可以替换 $a_t$ 再前向一次的计算图。它给的是文本故事，故事不会在你把动作置零之后自动坍缩。世界模型必须是一个你能对换、能置零、能画出漂移曲线的函数。

## 6. 源码导读

先 clone 对照仓库，按问题读，不要顺着 `train.py` 的 Hydra 迷宫往下走。官方训练脚本默认写给 SLURM 和 H100（`conf/train.yaml` 里 `gres: "gpu:h100:1"`），那是他们的集群配置，不是本课的硬件假设。

```bash
git clone https://github.com/gaoyuezhou/dino_wm.git
```

| 文件与位置 | 是哪个零件 | 带着什么问题读 |
|---|---|---|
| `models/dino.py::DinoV2Encoder` | 冻结视觉编码器 | `feature_key` 取 `x_norm_patchtokens` 还是 `x_norm_clstoken` 时 `latent_ndim` 有何不同？本课为什么必须用 patch 而不是一个 CLS 向量？ |
| `models/visual_world_model.py::encode` | 动作如何进状态 | `concat_dim==0` 把动作当成额外 token，`==1` 把动作维拼进每个 patch，两种拼接对对换实验意味着什么？ |
| `models/visual_world_model.py::forward` | 损失隔离 | `z_loss` 和 `decoder_loss_pred` 谁对预测器有梯度？`z_tgt` 为什么 `.detach()`？ |
| `models/visual_world_model.py::replace_actions_from_z` / `rollout` | 想象循环 | 滚动时新动作覆盖的是拼接后的哪几维？`num_hist` 窗口怎么切？ |
| `models/vit.py::ViTPredictor` / `Attention` | 预测器 | `generate_mask_matrix` 在做什么？第 58 行 `self.bias = ... .to('cuda')` 在 CPU 或 MPS 上会怎样？ |
| `conf/encoder/dino.yaml` | 官方默认骨干 | 确认 `name: "dinov2_vits14"` 与胶水脚本一致 |
| `conf/predictor/vit.yaml` | 官方预测器规模 | depth 6、heads 16、mlp_dim 2048；对照本课缩小档，说出你砍掉的是规模还是机制 |
| `conf/train.yaml` | 官方训练开关 | `model.train_encoder: False`、`has_predictor: True`、`frameskip: 5`、`num_hist: 3`、`num_pred: 1` 各对应 5.1 节哪句话？ |
| `train.py` | 官方入口 | README 示例是 `python train.py --config-name train.yaml env=point_maze frameskip=5 num_hist=3`。本课不改这个文件去吞摄像头数据 |

RSSM 备选只复习两处：`networks.py::RSSM.img_step`（闭眼，对换时喂不同动作）和 `imagine_with_action`。C-SWM 备选读 `modules.py::ContrastiveSWM.contrastive_loss` 和 `TransitionGNN.forward` 里动作拼到节点的那几行；官方 `train.py --dataset data/shapes_train.h5 --encoder small --name shapes` 是第 23 课的命令，不是本课桌面数据的命令。

Genesis 只当数据源。最小程序是官方 `examples/tutorials/hello_genesis.py`：`gs.init`、`Scene`、`morphs.Plane`、`morphs.MJCF`、`scene.build`、`scene.step`。相机在 `examples/rendering/moving_camera.py`：`scene.add_camera`、循环里 `cam_0.set_pose`、`cam_0.render(rgb=True)`。桌面采集脚本按这两份文件搭，不另写物理引擎。

## 7. 实验

建议工作目录 `~/desk_wm`，与对照仓库分开，避免 Genesis「在源码目录里 import 自己」那种坑。Step 1A 是摄像头档，Step 1B 是 Genesis 档，至少做一档；两档都做的人，对换表可以并排。Step 2 先用「复制最后一帧」冒充想象，让你看到失败，再训模型。

### Step 0: 建目录并确认 PyTorch

```bash
mkdir -p ~/desk_wm/data ~/desk_wm/runs
```

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

预期打印出版本号。`True` 表示有 CUDA；Mac 上应看到 `False`，后面脚本会走 MPS 或 CPU。再装采集和画图依赖（已有就跳过）：

```bash
pip install opencv-python matplotlib
```

DINOv2 第一次前向会从 `torch.hub` 拉权重。网络不好时，可先手动 clone [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) 再改 Hub 缓存，本课不把这当成必做。

### Step 1A: 摄像头档采集（假身体）

把下面存成 `~/desk_wm/collect_desk.py`。协议先读完再按：摄像头能同时看到杯子和你的手；键是粘滞的，按一次保持到按下一个，方便腾出手转镜头；`a` 看左（镜头或整台笔记本向左转）、`d` 看右、`w` 伸手向杯、`s` 不动、`q` 结束本段。按键的同时必须做对应的事，否则标签是假的。每段 20 到 40 秒，目标至少 20 段，四类动作的帧数不要让任何一类超过 60%。

```python
"""collect_desk.py  摄像头档：键盘当假身体，粘滞四键采桌面轨迹。"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

ACTIONS = {ord("a"): 0, ord("d"): 1, ord("w"): 2, ord("s"): 3}
NAMES = {0: "look_left", 1: "look_right", 2: "reach", 3: "stay"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="data")
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--seconds", type=float, default=30.0)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit("打不开摄像头，换 --camera 编号或检查权限")

    print("按 a/d/w/s 选择动作并同时做出来，q 结束本段，Ctrl-C 停采")
    ep = 0
    interval = 1.0 / args.fps
    try:
        while True:
            frames, actions, current = [], [], 3
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
                    cv2.imshow("desk collect", vis)
                    continue
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                actions.append(current)
                next_t += interval
                vis = frame.copy()
                cv2.putText(
                    vis, "%s %d/%d" % (NAMES[current], len(frames), n_target),
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                )
                cv2.imshow("desk collect", vis)
            if len(frames) < args.fps * 5:
                print("本段太短，丢弃")
                if (cv2.waitKey(0) & 0xFF) == ord("q"):
                    break
                continue
            arr_f = np.stack(frames).astype(np.uint8)
            arr_a = np.array(actions[:-1], dtype=np.int64)
            path = out_dir / ("ep_%03d.npz" % ep)
            np.savez_compressed(path, frames=arr_f, actions=arr_a, fps=args.fps)
            hist = np.bincount(arr_a, minlength=4)
            print("写入", path, "T=", len(arr_f), "hist=", hist.tolist())
            ep += 1
    except KeyboardInterrupt:
        print("停止采集")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

开始采集：

```bash
python collect_desk.py --out data --fps 5 --seconds 30
```

采满 20 段以上再进入 Step 2。Mac 第一次开摄像头会弹权限，拒绝的话脚本会直接退出。

### Step 1B: Genesis 档采集（可选，可复现对照）

没有合适的桌子、或想先在仿真里拿到可复现数字，用官方 Genesis。安装按当前文档：先装 PyTorch，再

```bash
pip install genesis-world
```

把下面存成 `~/desk_wm/collect_genesis.py`。场景是平面、一个方块当杯子、一台可平移的相机，动作用随机粘滞策略从「看左 / 看右 / 伸手 / 不动」里抽，相机位姿写法抄 `examples/rendering/moving_camera.py` 的 `set_pose`。

```python
"""collect_genesis.py  用 Genesis 官方 API 采桌面级带动作轨迹。"""
import argparse
from pathlib import Path

import numpy as np
import genesis as gs


def grab_rgb(cam):
    out = cam.render(rgb=True)
    rgb = out[0] if isinstance(out, (tuple, list)) else out
    return np.asarray(rgb)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="data_genesis")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--horizon", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    gs.init(backend=gs.cpu)
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    cube = scene.add_entity(
        gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.2, 0.0, 0.05))
    )
    cam = scene.add_camera(
        res=(320, 240), pos=(1.4, 0.0, 0.9), lookat=(0.2, 0.0, 0.05), fov=40
    )
    scene.build()

    for ep in range(args.episodes):
        cube.set_pos((0.2, 0.0, 0.05))
        yaw = 0.0
        frames, actions = [], []
        action = 3
        hold = 0
        for t in range(args.horizon):
            if hold <= 0:
                action = int(rng.integers(0, 4))
                hold = int(rng.integers(4, 10))
            hold -= 1
            if action == 0:
                yaw -= 0.04
            elif action == 1:
                yaw += 0.04
            elif action == 2:
                pos = np.array(cube.get_pos())
                pos[0] -= 0.01
                cube.set_pos(tuple(pos))
            cam.set_pose(
                pos=(1.4 * np.cos(yaw), 1.4 * np.sin(yaw), 0.9),
                lookat=(0.2, 0.0, 0.05),
            )
            scene.step()
            frames.append(grab_rgb(cam))
            actions.append(action)
        arr_f = np.stack(frames).astype(np.uint8)
        arr_a = np.array(actions[:-1], dtype=np.int64)
        path = out_dir / ("ep_%03d.npz" % ep)
        np.savez_compressed(path, frames=arr_f, actions=arr_a, fps=10.0)
        print("写入", path, "hist=", np.bincount(arr_a, minlength=4).tolist())


if __name__ == "__main__":
    main()
```

`cube.get_pos` / `set_pos` 若与你安装的 Genesis 版本方法名不一致，对照该版本的 Box 实体文档改两行，不要为此换仿真器。采数：

```bash
python collect_genesis.py --out data_genesis --episodes 30 --horizon 80
```

首次 `scene.build()` 会编译内核，可能要几分钟；同一场景再次运行会走缓存。

### Step 2: 体检数据，并用「复制最后一帧」先玩一次失败

```bash
python -c "from pathlib import Path; import numpy as np; fs=sorted(Path('data').glob('ep_*.npz')); assert fs, 'data/ empty'; packs=[np.load(f) for f in fs]; h=sum((np.bincount(p['actions'], minlength=4) for p in packs), start=np.zeros(4, dtype=np.int64)); T=sum(len(p['frames']) for p in packs); print('n', len(fs), 'frames', T, 'hist', h.tolist(), 'frac', (h/max(h.sum(),1)).round(3))"
```

预期：至少约 20 个 npz；`actions` 比 `frames` 短 1；四个动作比例没有某一档超过 0.6。若 `stay` 占 90%，回去补采看左、看右、伸手，否则对换实验没有材料。需要逐段看形状时，把 `np.load(f)['frames'].shape` 打进上面那条命令即可。

接着用惯性冒充世界模型。任取一段，把最后一帧复制 10 次当作「2 秒想象」，和真实后续并排看：转头的段落里，复制帧里的杯子不会移动，真后续会。这就是动作盲的样子。真正的模型必须比这个复制基线分得出看左和看右。把这一眼记进笔记，后面 Step 8 还要回来对比。

### Step 3: 写入 DINO-WM 式胶水脚本

把下面存成 `~/desk_wm/desk_wm.py`。它不是新架构：编码器调用方式与 `models/dino.py` 相同，动作沿特征维拼到每个 patch 上，对应 `concat_dim=1`；预测器是缩小的因果 ViT，掩码按设备创建，避开官方 `models/vit.py` 写死 `.to('cuda')` 的那一行；滚动时每步用探针动作覆盖动作维，对应 `replace_actions_from_z`。主损失只有特征 MSE。肉眼看对换时用 patch 的 PCA 伪彩，不把像素重建写进训练目标。

```python
"""desk_wm.py  第 30 课：DINO-WM 式桌面特征动力学 + 对换 / 漂移 / 负对照 / 想象。"""
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
from PIL import Image
from torch.utils.data import DataLoader, Dataset

NAMES = ["look_left", "look_right", "reach", "stay"]
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)


def device_of():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DinoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.emb_dim = self.backbone.num_features
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        return self.backbone.forward_features(x)["x_norm_patchtokens"]


class CausalViT(nn.Module):
    def __init__(self, dim, depth=2, heads=4, mlp_ratio=2.0, max_tokens=512):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, max_tokens, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            batch_first=True, activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        x = x + self.pos[:, :n]
        mask = torch.triu(torch.ones(n, n, device=x.device), diagonal=1).bool()
        x = self.enc(x, mask=mask)
        return self.head(self.norm(x))


class DeskWM(nn.Module):
    def __init__(self, visual_dim=384, action_dim=4, act_emb=32, hist=2):
        super().__init__()
        self.hist = hist
        self.act_emb_dim = act_emb
        self.visual_dim = visual_dim
        self.action_embed = nn.Embedding(action_dim, act_emb)
        self.dim = visual_dim + act_emb
        self.predictor = CausalViT(self.dim)

    def fuse(self, z, a):
        # z: (b, t, p, dv)  a: (b, t) long
        ae = self.action_embed(a)
        ae = ae.unsqueeze(2).expand(-1, -1, z.shape[2], -1)
        return torch.cat([z, ae], dim=-1)

    def split(self, h):
        return h[..., : self.visual_dim], h[..., self.visual_dim :]

    def predict_window(self, z, a):
        h = self.fuse(z, a)
        b, t, p, d = h.shape
        y = self.predictor(h.reshape(b, t * p, d)).reshape(b, t, p, d)
        z_hat, _ = self.split(y)
        return z_hat

    def one_step(self, z_hist, a_hist):
        z_hat = self.predict_window(z_hist, a_hist)
        return z_hat[:, -1]


class EpisodeCache(Dataset):
    def __init__(self, cache_dir, hist=2):
        self.files = sorted(glob.glob(str(Path(cache_dir) / "ep_*.pt")))
        self.hist = hist
        self.index = []
        self.data = []
        for i, f in enumerate(self.files):
            pack = torch.load(f, map_location="cpu", weights_only=False)
            self.data.append(pack)
            t = pack["z"].shape[0]
            for s in range(0, t - hist):
                self.index.append((i, s))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        i, s = self.index[idx]
        pack = self.data[i]
        h = self.hist
        z = pack["z"][s : s + h + 1]
        a = pack["a"][s : s + h]
        return z[:-1], a, z[-1]


def preprocess(frame, size=224):
    img = Image.fromarray(frame).resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img)).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor(IMNET_MEAN)[:, None, None]
    std = torch.tensor(IMNET_STD)[:, None, None]
    return (x - mean) / std


@torch.no_grad()
def encode_split(enc, data_dir, cache_dir, device, batch=8):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(data_dir).glob("ep_*.npz"))
    assert files, "在 %s 下没有 ep_*.npz" % data_dir
    for f in files:
        dest = cache_dir / (f.stem + ".pt")
        if dest.exists():
            print("跳过", dest)
            continue
        raw = np.load(f)
        frames = raw["frames"]
        acts = raw["actions"]
        zs = []
        for i in range(0, len(frames), batch):
            chunk = frames[i : i + batch]
            xs = torch.stack([preprocess(im) for im in chunk]).to(device)
            zs.append(enc(xs).cpu())
        z = torch.cat(zs, dim=0)
        a = torch.as_tensor(acts, dtype=torch.long)
        T = min(z.shape[0] - 1, a.shape[0])
        torch.save({"z": z[: T + 1], "a": a[:T], "src": str(f)}, dest)
        print("编码", f.name, "z", tuple(z.shape))


def train_loop(args, device):
    ds = EpisodeCache(args.cache, hist=args.hist)
    n_val = max(1, len(ds) // 8)
    n_tr = len(ds) - n_val
    tr, va = torch.utils.data.random_split(
        ds, [n_tr, n_val], generator=torch.Generator().manual_seed(0)
    )
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, drop_last=True)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False)
    pack0 = ds.data[0]
    model = DeskWM(visual_dim=pack0["z"].shape[-1], hist=args.hist).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = 1e9
    out = Path(args.run)
    out.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        tr_loss = []
        for z_hist, a_hist, z_next in tl:
            z_hist = z_hist.to(device)
            a_hist = a_hist.to(device)
            z_next = z_next.to(device)
            if args.blind:
                a_hist = torch.zeros_like(a_hist)
            pred = model.one_step(z_hist, a_hist)
            loss = F.mse_loss(pred, z_next)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss.append(loss.item())
        model.eval()
        va_loss = []
        with torch.no_grad():
            for z_hist, a_hist, z_next in vl:
                pred = model.one_step(z_hist.to(device), a_hist.to(device))
                va_loss.append(F.mse_loss(pred, z_next.to(device)).item())
        tr_m, va_m = float(np.mean(tr_loss)), float(np.mean(va_loss))
        print("epoch %03d  train %.5f  val %.5f" % (epoch, tr_m, va_m))
        if va_m < best:
            best = va_m
            torch.save(
                {"model": model.state_dict(), "hist": args.hist, "best_val": best},
                out / "best.pt",
            )
    print("best val", best, "->", out / "best.pt")


def load_model(run, device):
    ckpt = torch.load(Path(run) / "best.pt", map_location=device, weights_only=False)
    cache = sorted(Path(run).glob("cache/ep_*.pt"))
    if not cache:
        cache = sorted((Path(run) / "cache").glob("ep_*.pt"))
    sample = torch.load(cache[0], map_location="cpu", weights_only=False)
    model = DeskWM(visual_dim=sample["z"].shape[-1], hist=ckpt["hist"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def pca_map(z, k=3):
    # z: (p, d) -> (16, 16, 3) 近似
    x = z.astype(np.float64)
    x = x - x.mean(0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    comp = x @ vt[:k].T
    side = int(np.sqrt(comp.shape[0]))
    img = comp.reshape(side, side, k)
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    return img


@torch.no_grad()
def rollout(model, z0, a_seq, device):
    """z0: (hist, p, d); a_seq: (n,) 每步要执行的动作，返回 n 个预测特征。"""
    hist = model.hist
    z_win = z0.clone().to(device)
    a_win = a_seq[:1].repeat(hist).to(device)
    preds = []
    for k in range(len(a_seq)):
        pred = model.one_step(z_win.unsqueeze(0), a_win.unsqueeze(0))[0]
        preds.append(pred.cpu())
        z_win = torch.cat([z_win[1:], pred.unsqueeze(0)], dim=0)
        a_win = torch.cat([a_win[1:], a_seq[k:k + 1].to(device)], dim=0)
    return torch.stack(preds)


def pick_episode(cache_dir, which=-1):
    files = sorted(Path(cache_dir).glob("ep_*.pt"))
    pack = torch.load(files[which], map_location="cpu", weights_only=False)
    return pack, files[which]


def scale_of(z):
    return torch.norm(z[1:] - z[:-1], dim=(1, 2)).mean().item()


def swap_probe(args, device):
    model, ckpt = load_model(args.run, device)
    pack, src = pick_episode(Path(args.run) / "cache", -1)
    z, a = pack["z"], pack["a"]
    t0 = args.t0
    hist = ckpt["hist"]
    assert t0 >= hist and t0 + args.branch < len(z)
    scale = scale_of(z)
    z0 = z[t0 - hist: t0]
    names = NAMES
    trajs = []
    for aid, name in enumerate(names):
        acts = torch.full((args.branch,), aid, dtype=torch.long)
        pred = rollout(model, z0, acts, device)
        trajs.append(pred)
        print(name, "steps", pred.shape[0])
    print("比例尺（真实一步特征位移）: %.4f" % scale)
    print("使用缓存:", src, "t0=", t0)
    for step_i, label in [(0, "第 1 步"), (args.branch - 1, "第 %d 步" % args.branch)]:
        print("\n[%s] 两两距离 / 比例尺（对角线应为 0）" % label)
        print("%12s" % "" + "".join("%12s" % n for n in names))
        for i, ni in enumerate(names):
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
    for i, name in enumerate(names):
        axes[0, i + 1].imshow(pca_map(trajs[i][0].numpy()))
        axes[0, i + 1].set_title("%s +1" % name)
        axes[1, i + 1].imshow(pca_map(trajs[i][-1].numpy()))
        axes[1, i + 1].set_title("%s +%d" % (name, args.branch))
    for ax in axes.ravel():
        ax.axis("off")
    out = Path(args.run) / "action_swap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close()
    print("对换图", out)


def drift_probe(args, device):
    model, ckpt = load_model(args.run, device)
    hist = ckpt["hist"]
    files = sorted((Path(args.run) / "cache").glob("ep_*.pt"))
    horizon = args.horizon
    tf = np.zeros(horizon)
    fr = np.zeros(horizon)
    nwin = 0
    scale_acc = []
    for f in files[: args.windows]:
        pack = torch.load(f, map_location="cpu", weights_only=False)
        z, a = pack["z"], pack["a"]
        if len(z) < hist + horizon + 2:
            continue
        scale_acc.append(scale_of(z))
        starts = np.linspace(hist, len(z) - horizon - 2, num=2, dtype=int)
        for t0 in starts:
            with torch.no_grad():
                for k in range(horizon):
                    z_hist = z[t0 + k - hist: t0 + k].unsqueeze(0).to(device)
                    a_hist = a[t0 + k - hist: t0 + k].unsqueeze(0).to(device)
                    pred = model.one_step(z_hist, a_hist)[0].cpu()
                    tf[k] += torch.norm(pred - z[t0 + k]).item()
                z_win = z[t0 - hist: t0].clone()
                a_win = a[t0 - hist: t0].clone()
                for k in range(horizon):
                    pred = model.one_step(
                        z_win.unsqueeze(0).to(device),
                        a_win.unsqueeze(0).to(device),
                    )[0].cpu()
                    fr[k] += torch.norm(pred - z[t0 + k]).item()
                    z_win = torch.cat([z_win[1:], pred.unsqueeze(0)], 0)
                    nxt = a[min(t0 + k, len(a) - 1): min(t0 + k, len(a) - 1) + 1]
                    a_win = torch.cat([a_win[1:], nxt], 0)
            nwin += 1
    assert nwin > 0, "轨迹太短，调小 --horizon"
    tf /= nwin
    fr /= nwin
    scale = float(np.mean(scale_acc))
    print("窗口数", nwin, "偷懒基线", "%.4f" % scale)
    print("第 1 步 teacher=%.4f free=%.4f（应接近）" % (tf[0], fr[0]))
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


def neg_probe(args, device):
    model, ckpt = load_model(args.run, device)
    hist = ckpt["hist"]
    pack, _ = pick_episode(Path(args.run) / "cache", -1)
    z, a = pack["z"], pack["a"]
    t0 = min(max(args.t0, hist), len(z) - 8)
    z_hist = z[t0 - hist: t0].unsqueeze(0).to(device)
    a_true = a[t0 - hist: t0].unsqueeze(0).to(device)
    z_next = z[t0].to(device)
    with torch.no_grad():
        err_true = F.mse_loss(model.one_step(z_hist, a_true)[0], z_next).item()
        a_zero = torch.full_like(a_true, 3)
        err_zero = F.mse_loss(model.one_step(z_hist, a_zero)[0], z_next).item()
        perm = a.clone()
        perm = perm[torch.randperm(len(perm))]
        a_shuf = perm[t0 - hist: t0].unsqueeze(0).to(device)
        err_shuf = F.mse_loss(model.one_step(z_hist, a_shuf)[0], z_next).item()
    print("一步 MSE  真实动作 %.5f" % err_true)
    print("一步 MSE  动作置零(stay) %.5f" % err_zero)
    print("一步 MSE  打乱动作 %.5f" % err_shuf)
    print("置零 / 真实 = %.3f   打乱 / 真实 = %.3f" % (
        err_zero / max(err_true, 1e-8), err_shuf / max(err_true, 1e-8)))
    if err_zero <= err_true * 1.02 and err_shuf <= err_true * 1.02:
        print("负对照未拉开：模型很可能动作盲，或这段窗口本身接近不动")


def imagine_probe(args, device):
    model, ckpt = load_model(args.run, device)
    hist = ckpt["hist"]
    pack, src = pick_episode(Path(args.run) / "cache", args.episode)
    z = pack["z"]
    t0 = args.t0
    raw_files = sorted(Path(args.data).glob("ep_*.npz"))
    raw = np.load(raw_files[args.episode])
    frames = raw["frames"]
    fps = float(raw["fps"]) if "fps" in raw.files else 5.0
    steps = int(round(args.seconds * fps))
    z0 = z[t0 - hist: t0]
    fig, axes = plt.subplots(5, steps + 1, figsize=(1.6 * (steps + 1), 8))
    axes[0, 0].imshow(frames[t0])
    axes[0, 0].set_title("t0")
    for k in range(steps):
        idx = min(t0 + 1 + k, len(frames) - 1)
        axes[0, k + 1].imshow(frames[idx])
        axes[0, k + 1].set_title("real +%d" % (k + 1))
    for aid, name in enumerate(NAMES):
        acts = torch.full((steps,), aid, dtype=torch.long)
        pred = rollout(model, z0, acts, device)
        axes[aid + 1, 0].imshow(pca_map(z0[-1].numpy()))
        axes[aid + 1, 0].set_title(name)
        for k in range(steps):
            axes[aid + 1, k + 1].imshow(pca_map(pred[k].numpy()))
    for ax in axes.ravel():
        ax.axis("off")
    out = Path(args.run) / "imagine_%d_t%d.png" % (args.episode, t0)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close()
    print("想象条", out, "源", src)
    print("给四条分支打分：会不会在 +%.1f 秒碰到杯子？写进 NOTES.md" % args.seconds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True,
                   choices=["encode", "train", "swap", "drift", "neg", "imagine"])
    p.add_argument("--data", type=str, default="data")
    p.add_argument("--run", type=str, default="runs/desk1")
    p.add_argument("--cache", type=str, default="")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hist", type=int, default=2)
    p.add_argument("--t0", type=int, default=40)
    p.add_argument("--branch", type=int, default=10)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--windows", type=int, default=6)
    p.add_argument("--episode", type=int, default=-1)
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--blind", action="store_true")
    args = p.parse_args()
    if not args.cache:
        args.cache = str(Path(args.run) / "cache")
    device = device_of()
    print("device", device)
    if args.mode == "encode":
        enc = DinoEncoder().to(device)
        encode_split(enc, args.data, args.cache, device)
    elif args.mode == "train":
        train_loop(args, device)
    elif args.mode == "swap":
        swap_probe(args, device)
    elif args.mode == "drift":
        drift_probe(args, device)
    elif args.mode == "neg":
        neg_probe(args, device)
    else:
        imagine_probe(args, device)


if __name__ == "__main__":
    main()
```

脚本里有意没有像素重建主损失。`--blind` 是 5.4 节的重训负对照开关，主训练不要加它。

### Step 4: 编码并训练

先把每段录像压成 patch 特征，这一步最耗时，但只做一次：

```bash
python desk_wm.py --mode encode --data data --run runs/desk1
```

预期：`runs/desk1/cache/ep_000.pt` 陆续出现，打印 `z (T, 256, 384)`。Mac / CPU 上每段 150 帧可能要一两分钟；24GB 卡上是秒级。中途断了再跑会跳过已有缓存。

然后训练预测器：

```bash
python desk_wm.py --mode train --data data --run runs/desk1 --epochs 30 --batch 8
```

预期：`train` 和 `val` 的 MSE 前几个 epoch 明显下降，然后变缓；`runs/desk1/best.pt` 按验证损失保存。验证损失若从第一个 epoch 起就不降，去第 10 节。24GB 卡上 30 epoch 通常十几分钟量级；Mac 上按一个小时准备。不要在这一步根据重建图调参，你还没有解码器。

Genesis 档把 `--data data_genesis` 和 `--run runs/gen1` 换掉即可，其余命令相同。

### Step 5: 动作对换（复现 #9 的主证据）

```bash
python desk_wm.py --mode swap --run runs/desk1 --data data --t0 40 --branch 10
```

按顺序检查四件事：

1. 对角线全是 0.00。不为 0，探针混进了随机性，先修再谈分岔。
2. 第 1 步的看左对看右大于 0，但可以很小。桌面惯性决定了一步几乎不动。
3. 第 10 步（2 秒）看左对看右显著大于第 1 步，并且通常是全表最大。健康模型这一格应到 1 的量级附近。全表贴 0，宣布动作盲，去做负对照和排错。
4. 打开 `runs/desk1/action_swap.png`：第一列是真实 t0 和真实 +10 帧，其余四列是四键想象的 PCA 伪彩。看左和看右的伪彩在第 10 步应能分出空间结构在挪；伸手应和不动不同。图会糊，糊不是失败，方向分不开才是失败。

`--t0` 要落在这段录像里你确实做过转头或伸手的附近。选在「全程坐着不动」的窗口，四个动作的真值未来几乎一样，模型分不开是数据的问题，不是探针的问题。

### Step 6: 多步漂移

```bash
python desk_wm.py --mode drift --run runs/desk1 --horizon 15 --windows 6
```

预期读数：

1. 第 1 步 teacher 与 free 接近（实现上窗口输入相同）。差一个数量级，先查 `rollout` 有没有误喂真实 $z$。
2. teacher forcing 全程低于图中虚线偷懒基线。高于基线，预测器没有超过「复制上一帧」。
3. 自由 rollout 上扬，与 teacher 的裂口随步数变宽。这是暴露偏差在桌面上的宽度。
4. 看自由曲线在第几步越过偷懒基线。5 帧/秒下第 10 步就是 2 秒。若第 4 步已经越过，Step 8 的 2 秒想象只能当演示，不能当第 32 课的规划视野。

把 `drift_curve.png` 存进证据目录。自由曲线涨过基线并不稀奇：两个都合理的未来，逐点 L2 也会很大。第 17 课讨论过「多步之后逐点误差不再公道」；本课仍然画它，因为它给规划视野一个保守上限。

### Step 7: 负对照（必做）

```bash
python desk_wm.py --mode neg --run runs/desk1 --t0 40
```

预期：真实动作的一步 MSE 低于动作置零，也低于打乱动作。比值「置零 / 真实」和「打乱 / 真实」应明显大于 1。若两档比值都在 1.02 以内，模型在这段窗口里没用动作。换一个有转头的 `t0` 再测一次；仍然塌，去排错。

加分对照：重训一个致盲模型。把 `best.pt` 先复制走，以免覆盖。

```bash
python desk_wm.py --mode train --run runs/desk1_blind --data data --epochs 30 --blind
```

致盲模型要先有自己的 cache。最省事的办法是把主实验的缓存拷过去：

```bash
cp -R runs/desk1/cache runs/desk1_blind/cache
```

然后对致盲模型跑 Step 5。预期：主模型第 10 步看左-看右距离是致盲模型的数倍；两边的验证 MSE 可能只差一点点。这就是第 03 课「钝器与利器」在桌子上的再版。致盲版验证损失若反而低很多，说明你的动作标签噪声太大，模型靠忽略动作在拟合惯性，把这件事写进报告。

### Step 8: 互动，同一历史帧播 2 秒想象

挑一段你记得「后面真的伸过手」的录像，把 `--episode` 指到它（`-1` 是最后一段），`--t0` 指到伸手前几帧：

```bash
python desk_wm.py --mode imagine --run runs/desk1 --data data --episode -1 --t0 40 --seconds 2
```

打开 `runs/desk1/imagine_-1_t40.png`（文件名按你的参数变）。第一行是真实后续，下面四行是看左、看右、伸手、不动的 PCA 想象。按下面这张表给「会不会碰杯」打分，分数是你的判断，不是模型吐出的概率。

| 分支 | 想象里手/杯子是否靠近 | 你会不会执行 | 真实后续是否碰杯 |
|---|---|---|---|
| 看左 | 填 是 / 否 / 看不清 | 填 | 填 |
| 看右 | 填 | 填 | 填 |
| 伸手 | 填 | 填 | 填 |
| 不动 | 填 | 填 | 填 |

和 Step 2 的复制帧并排看一眼：复制帧四行应几乎一样；现在四行必须能分出至少「转头 vs 伸手」或「看左 vs 看右」。分不出，模型还不能进第 32 课。伸手分支若把杯子「溶进」桌面纹理，记下「接触动作是盲的」，这是第 11 节改造清单的入口，不是把解码器加大就能消失的问题。

有真机的人把四键映射成 Reachy Mini 的头偏航或 SO-101 的末端位移，同一套 `desk_wm.py` 不用改预测器，只改采集端写入的 `actions` 含义。

### Step 9: 写对照报告

在 `runs/desk1/NOTES.md` 里按这个骨架写，写完才算复现 #9 交卷：

```text
数据：来源（摄像头 / Genesis / 真机），段数，总帧，fps，四类动作计数
命令：encode / train / swap / drift / neg / imagine 的完整命令
训练：best val MSE，是否用 --blind
动作对换：比例尺，第 1 步与第 10 步 look_left-look_right，是否动作盲
漂移：第 1 步 teacher/free，第 10 步 / 第 15 步，越过偷懒基线的步数
负对照：真实 / 置零 / 打乱 的一步 MSE 与比值
想象：所用 episode 与 t0，四键碰杯打分表，哪类动作是盲的
本实验只说明：在这份桌面（或 Genesis）数据分布上，模型是否听动作、2 秒梦漂多远。
不说明：能在真机上规划成功，也不说明复现了 DINO-WM 论文的六环境数字。
```

## 8. 配置与预算

| 项 | 本课默认 | 说明 |
|---|---|---|
| 数据（摄像头） | ≥20 段 × 30 秒 × 5 fps，约 3000 帧 | 四类动作任一类占比不超过 0.6；少了就补采，不要先加模型 |
| 数据（Genesis） | 30 段 × 80 步 | 官方 `gs.init` + `add_camera`，可复现，作对照不替代真桌子 |
| 编码器 | 冻结 `dinov2_vits14`，patch 256×384 | 与 `conf/encoder/dino.yaml` 同骨干；不训练编码器 |
| 预测器 | 2 层、4 头、历史 2 帧 | 官方 `conf/predictor/vit.yaml` 是 6 层 16 头、历史 3 帧，本课砍规模不砍机制 |
| 动作 | 4 类 embedding，拼到每个 patch | 对应官方 `concat_dim: 1`，无本体感觉通道 |
| 优化 | AdamW，lr 3e-4，30 epoch，batch 8 | 按验证 MSE 存 `best.pt` |
| 主损失 | 特征 MSE | 禁止把像素重建当主损失 |
| 编码耗时 | 24GB 卡数分钟；Mac 数十分钟 | 缓存到 `runs/*/cache`，断点可续 |
| 训练耗时 | 24GB 卡约 10-30 分钟；Mac 约 1 小时 | 致盲重训再加一份同样预算 |
| 探针耗时 | 数分钟，CPU 可做 | swap / drift / neg / imagine 都是纯前向 |
| 检查点 | `runs/desk1/best.pt` | 探针只认这个文件 |

预算上的大头是采集覆盖，不是预测器参数量。桌面世界模型最常见的失败是「20 段里 15 段在发呆」，不是「2 层 ViT 太小」。想加钱，先把伸手和转头的帧数加到和不动同一量级。官方 DINO-WM 在 PushT 一类环境上用 `frameskip=5`，等于他们的一步对应环境 5 帧；你如果把摄像头提到 15 fps 却不抽帧，模型会把容量浪费在几乎不变的相邻帧上。5 fps 是有意选的。

## 9. 验收

- [ ] 至少一档带动作的轨迹：摄像头或 Genesis。动作直方图打印出来，没有单类超过 60%。
- [ ] `best.pt` 存在，训练日志里验证 MSE 相对第 0 个 epoch 下降。
- [ ] 动作对换表：对角线为 0；第 10 步看左对看右大于第 1 步，且明显大于 0。`action_swap.png` 里至少两个动作的 PCA 能分方向。
- [ ] 漂移图：teacher forcing 低于偷懒基线；自由 rollout 上扬；笔记里写了 2 秒落在曲线哪一段。
- [ ] 负对照：置零或打乱至少一种使一步 MSE 变差；或致盲重训的对换表塌掉。
- [ ] 想象条：同一 `t0` 四键 2 秒与真实后续并排，碰杯打分表填完，并写明哪类动作是盲的。
- [ ] `NOTES.md` 按 Step 9 写完，含「只说明什么 / 不说明什么」那两句。
- [ ] 能口头回答：为什么平均特征误差查不出动作盲？为什么只训重建不能毕业？为什么 GPT-4o 描述未来视频不算世界模型？

过线的标准是方向：动作改写了未来，对照改写了结论。不过线的典型方式是损失好看、图很糊、四键想象像复制。那种结果可以当失败分析交，不能当复现 #9 通过。

## 10. 排错

| 症状 | 最可能的原因 | 怎么确认 | 怎么修 |
|---|---|---|---|
| `torch.hub.load` 失败或卡住 | 网络、Hub 校验、或公司代理 | 单独跑一句加载 DINOv2 的 Python | 设好代理；或与官方 `dino.py` 一样保留 `_validate_not_a_forked_repo` |
| 编码时显存爆 | batch 太大或分辨率被改大 | 看报错是否 OOM | `encode_split` 里默认 batch=8，改成 2；不要把 224 改成 448 |
| MPS 上 ViT 报错 | 个别 PyTorch 版本的 MPS 对 `TransformerEncoder` 支持不完整 | 同一命令设环境走 CPU 对比 | 训练改 CPU，或把 CausalViT 换成两层 `nn.Linear` 做冒烟，确认数据管道后再回 ViT |
| 验证损失不降 | 缓存与原始帧错位，或几乎全是不动 | 打印一段 `a` 的直方图；抽一帧 cache 与 npz 对长度 | 重做 encode；补采动作；先在 Genesis 档上看损失是否会降 |
| 对换表对角线非 0 | 模型仍在 train()，或 Hub 编码器被设成训练模式 | 查 `model.eval()` 与 `DinoEncoder.eval()` | 恢复脚本，探针全程 `no_grad` |
| 对换全表贴 0 | 动作盲：标签错位、数据太懒、或训练不足 | 先跑 neg；再人为把某段 `actions` 改成全 0 重训对比 | 检查 `actions` 比 `frames` 短 1；补采转头和伸手；加 epoch |
| 数字分岔但 PCA 四张一样 | 差异在少数 patch 上，PCA 前三分量吃不到 | 看第 10 步距离是否其实偏小 | 把 `--branch` 加到 15；或只可视化变化最大的 32 个 patch |
| 自由曲线与 teacher 重合 | 自由滚动误喂了真实 $z$ | 两曲线处处相等就是证据 | 对照 `rollout` 里 `z_win` 必须接 `pred` |
| Genesis `get_pos` / `set_pos` 报错 | 你安装的版本方法名不同 | 在解释器里 `print(dir(cube))` | 按当前 Genesis 文档改采集脚本那两行，预测器不用动 |
| 摄像头全黑或权限拒绝 | macOS 摄像头权限、或 `--camera` 编号不对 | 系统设置里看终端是否被允许 | 换 `--camera 1`；用系统相机应用确认硬件可用 |
| 想象里伸手完全无变化 | 数据里伸手帧太少，或伸手时杯子被身体挡住 | 回放含 `w` 的片段 | 换机位让手和杯同时入画；单独加采 10 段伸手 |

官方 `dino_wm` 的 `train.py` 若你想加餐跑 PointMaze：先按他们 README 装 `environment.yaml`、下 OSF 数据、设 `DATASET_DIR`，再使用文档里的 `python train.py --config-name train.yaml env=point_maze frameskip=5 num_hist=3`。跑不通不扣本课分。那是另一份数据上的官方训练，替代不了你桌子上的对换表。

## 11. 前沿与改造

同一问题，2024-2025 年公开系统怎么做。DINO-WM 把动力学放在冻结 DINOv2 patch 上，测试时用 CEM 或梯度规划去够一张目标图，六个环境（PushT、PointMaze、Wall、Rope、Granular、Reacher）上报告零样本目标到达，不需要奖励模型和专家演示。Visual Foresight（Ebert, Finn, Dasari, Xie, Lee, Levine, arXiv:1812.00568）更早一步：直接在像素里学视频预测，再用视觉 MPC 推物体，数据是机械臂自己乱抓来的。PlaNet / RSSM 把随机性塞进状态，在潜空间做 CEM。C-SWM 把状态拆成槽，对比学习代替重建。V-JEPA 2-AC（第 15 课精读过接口）走的是冻结视频骨干再接动作条件预测，规划时压能量，和本课默认路线同构，只是骨干从图像 DINOv2 换成了视频 V-JEPA 2，数据从你的 3000 帧换成了论文里的机器人小时级轨迹。

我们差在两处，要分开说。规模差：官方预测器更深、数据是受控环境里覆盖过的随机策略、规划时滚 CEM 几十轮。钱能补数据量和层数，补不齐你桌子上「伸手碰杯」这种接触事件的稀疏。机制差：本课没有规划器、没有目标图像接口、没有把第 29 课的杯子框和人脸朝向拼进状态。第 32 课会把展开接到安全过滤上；本课若把 CEM 提前做了，也只许当探针，不许写成「我复现了 DINO-WM 的规划表」。

动手改造清单，每个都写在锚定文件或本课脚本上：

1. 把官方拼接从 `concat_dim=1` 改成 `concat_dim=0`（动作当额外 token）。改 `DeskWM.fuse`，让动作嵌入成为第 $N+1$ 个 token，预测后再丢掉。预算：同数据重训 30 epoch。预期：对换仍应分岔；若塌掉，说明你的缩小 ViT 在「额外 token」设定下容量不够，记成规模问题。失败：两种拼接对换表都贴 0，回到数据和标签。
2. 加第 29 课的低维状态当辅助目标：在特征 MSE 之外，用一个线性头从 $\hat{z}$ 回归杯子是否在画面里、人脸是否看镜头。头的损失权重取 0.1 量级试起。预算：同数据再训一份。预期：伸手分支对「杯子还在不在」的判断比纯特征 MSE 稳。失败：辅助头过拟合到「永远在」，把权重降到 0 对比。
3. 接触专项数据：只补采伸手，目标是让 `reach` 帧数与 `stay` 同量级，不改模型。预算：半天采集加一次重训。预期：Step 8 伸手分支开始和不动分开。失败：仍分不开，检查伸手时手是否挡住杯子导致编码器看见的是「一团运动纹理」。
4. 选做 RSSM：把 `DeskWM` 换成第 05 课 `RSSM.img_step` 的最小移植，编码器仍冻结 DINOv2，只把 patch 做平均池化成一条向量再进 GRU。预算：数小时。预期：短时漂移可能更好，对换仍要过。失败：随机通道被 KL 压死，先验不管动作，对换塌回 0。

顺手复现映射。DINO-WM 的核心论断「预训练空间特征上的动作条件预测，足以让不同动作的想象分开」，在本课对应 Step 5 的分岔表，预期能看到同方向趋势（分岔随步数增大），不能预期看到他们的规划成功率。PlaNet「确定加随机两条通道」在桌面缩小数据上不一定显形，对应改造 4，预期可能复现不出差异，那是任务随机性不够，不是你没读懂论文。Visual Foresight「随机探索数据就能训可用的视觉预测」对应 Step 1 的粘滞随机（摄像头档是你自己当执行器），预期：覆盖不够时伸手是盲的，正好解释他们为什么要让机器人抓几百个物体。

## 12. 论文与延伸

1. DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning（Zhou, Pan, LeCun, Pinto, 2024/2025，[arXiv:2411.04983](https://arxiv.org/abs/2411.04983)）。本课架构的出处。带着四个问题读：他们为什么坚持预测 patch 而不是 CLS？解码器在方法节里被标成 optional 的根据是什么？测试时规划的目标是目标图像的特征，和本课四键想象差在哪一截？CLEVRER 上「1 帧条件 vs 3 帧条件」的开放预测，对你把 `hist` 设成 2 有什么提醒？
2. Learning Latent Dynamics for Planning from Pixels（PlaNet / RSSM，Hafner et al., 2019，[arXiv:1811.04551](https://arxiv.org/abs/1811.04551)）。第 05 课读过，本课只重读动力学与规划分工。带着两个问题：`img_step` 喂不同动作比较先验，怎样改写成 Step 5 那种对换表？他们用多步潜变量目标（latent overshooting）对付暴露偏差，你只有单步 MSE，漂移曲线里缺的是哪一味药？
3. Contrastive Learning of Structured World Models（C-SWM，Kipf, van der Pol, Welling, ICLR 2020，[arXiv:1911.12247](https://arxiv.org/abs/1911.12247)）。槽动力学的对照。带着三个问题：`TransitionGNN` 把动作拼到节点上，和 DINO-WM 拼到 patch 上，谁更适合「只有手在动、杯子不该动」？对比损失为什么能不用解码器？官方 `ignore_action` 在三体重力实验里打开，那是动作盲的合法用途还是反例？
4. Visual Foresight: Model-Based Deep Reinforcement Learning for Vision-Based Robotic Control（Ebert, Finn, Dasari, Xie, Lee, Levine, 2018，[arXiv:1812.00568](https://arxiv.org/abs/1812.00568)）。像素世界模型做操作的祖先。带着三个问题：他们的数据是机器人自己采集的，和你按键同时伸手，哪一种更接近「随机策略覆盖」？目标像素、目标图像、分类器三种指定目标的方法，哪一种能接到第 32 课的克制？论文里泛化到未见物体，你桌子上换一只杯子，对换表还能不能过？
5. DINOv2: Learning Robust Visual Features without Supervision（Oquab et al., 2023，[arXiv:2304.07193](https://arxiv.org/abs/2304.07193)）。编码器出生证，只读「patch 特征有空间结构」那部分。带着一个问题：若把本课骨干换成随机初始化的 ViT-S，30 epoch 的桌面数据撑不撑得住对换？不要猜，改造清单里可以加第五项，预算是一次重训。

选读官方项目页 [dino-wm.github.io](https://dino-wm.github.io) 上的开环想象与规划视频，只当对照，不要把页面上的成功率抄进你的 `NOTES.md`。World Models（Ha & Schmidhuber, [arXiv:1803.10122](https://arxiv.org/abs/1803.10122)）若还要翻，只看 M 必须吃动作那一段，当作本课公式的 2018 年写法。

零件盘点。观察进状态这件事第 29 课做了；本课给状态接上动作，桌子第一次有了可以查询的未来。系统现在能对同一段历史播出看左、看右、伸手、不动四条 2 秒的梦，并且你知道梦在第几步开始不能当尺子。还缺两块：人不是你能选的动作，却是桌上最大的外生过程，第 31 课要把「人会不会看过来、手会不会靠近杯子」当成预测目标，并加一段短时记忆。第 32 课再把感知、本课的展开、第 27 课的安全过滤和身体接在一起，做出五件先查询世界模型再动的行为。没有本课这份对换表，那五件里的克制和失败承认都没有根据。
