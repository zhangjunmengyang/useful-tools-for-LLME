---
id: 20_spatial_3d_state
title: "从多张图得到持久的 3D 状态"
summary: "一帧画面不是世界。换个视角，房间和物体为什么不该跟着漂？"
unit: spatial
play_tools: []
checkpoints:
  - "一份自己桌子的点图和误差表。"
  - "书面区分：重建不是 $P(s_{t+1}|s_t,a_t)$。"
---

# 第 20 课：从多张照片得到一份持久的 3D 状态

> 类型：实战/体验（跑公开权重，**不从头训练 VGGT**）<br>
> 建议周期：2-3 天（拍桌子半天，装环境与第一次推理半天，量误差和写表 1 天）<br>
> 硬件：单张 24GB 卡跑 8-20 张图足够；论文在 H100 上测过 20 张约 5.6GB 骨干显存，消费卡会更高，8-12 张在 12GB 卡上通常能过；Mac/CPU 可装环境和读代码，本地推理会很慢，可视化可改走官方 [Hugging Face Space](https://huggingface.co/spaces/facebook/vggt)<br>
> 锚定仓库：[facebookresearch/vggt](https://github.com/facebookresearch/vggt)（CVPR 2025 Best Paper，权重 `facebook/VGGT-1B`）；对照 nerfstudio / gsplat 只讲，不作为主实验<br>
> 产物：自己桌子的点图与相机位姿、一张误差表（轨迹自洽 / 同物跨视角 3D 误差 / 拿掉侧面图后遮挡面还在不在），以及一句能当面讲清的话：重建不是世界模型，缺动作条件和时间

## 1. 这一课做什么

[第 19 课](19_surgery_experiments.md) 把第五幕收官了。你手里现在有一套能在有 gym、有分数的领域里自转的研究循环：复现、换引擎、对照、测量、缩放、改结构。第 19 课结尾也把三根拐杖点名了：环境可以无限 reset，分数现成，验证者是能重跑的代码。从这一课起，这三根都要松。桌子不会给你重开一局；杯子会不会倒，得你自己定义什么叫准。

第六幕要补的是像素世界模型一直缺的两样东西：离开视野的物体还在不在，桌上的东西是不是分开的。没有这两样，后面的桌宠会把杯子忘掉，或把杯子和手揉成一团纹理。本课只做第一样。零件很具体：把多张 RGB 变成相机位姿加一份可查询的点图（point map：每个像素对应一个三维坐标）。这是编码器，不是完整世界模型。它默认不吃动作，也不预测下一秒。

整门课的主干还是那条：

```text
观察 先压成状态 再按动作预测下一状态 然后展开多条未来 给未来打分 最后选动作
```

前 19 课的"状态"几乎都是一帧画面压出来的向量，或一段滑窗里的 token。[第 12 课](12_frontier_landscape.md) 你已经在 open-oasis 里量过：转一圈再回头，窗外的房子会被重新发明，因为模型没有一份独立于当前画面的空间记忆。本课换上的零件，就是把"状态"从"当前看见的像素"换成"这张桌子的三维几何"。头转动时，身后那只杯子必须还在这份状态里，哪怕此刻镜头里没有它。

做完你能验证三件事。第一，多张桌子照片能前向一次吐出相机和点图，不必自己写光束法平差。第二，同一只杯子在不同视角下的三维位置对得上，误差能写成表。第三，你能把这份点图和 $P(s_{t+1} \mid s_t, a_t)$ 分开：前者回答"现在世界长什么样"，后者才回答"我伸手之后会怎样"。第 21 课才会给这份状态加上时间更新。

术语速查：

| 术语 | 一句人话 |
|---|---|
| 点图（point map） | 一张和输入图对齐的三维坐标图：像素 $(u, v)$ 对应场景里的一个点 $P(u,v)\in\mathbb{R}^3$ |
| 外参 / 内参 | 外参是相机在世界里的位置和朝向；内参是焦距、主点这类镜头几何 |
| 参考系（第一相机） | VGGT 把第一张图的相机当作世界原点，后面所有点都写在这个坐标系里 |
| 前向重建 | 一次神经网络前向就出几何，不再先匹配再三角化再迭代优化 |
| 光束法平差（BA） | 传统视觉里反复调相机和三维点、让重投影误差变小的后处理；VGGT 可选，不是必须 |
| 新视角合成 | 给定已有照片，画出没拍过的那个角度看过去长什么样；3DGS / NeRF 擅长这个 |
| 持久 3D 状态 | 一份不随当前画面消失的空间记忆：转头之后，刚才身后的杯子坐标还在 |
| 遮挡面 | 当前镜头看不见、但别的照片看见过的那一面；桌宠转头时最容易丢的就是它 |
| 动作条件 | 预测必须随打算做的动作变化；本课的模型没有这个输入 |
| 实战/体验 | 跑别人训好的权重做探针和游玩，不从头训练；见课程分档 |

## 2. 问题

一帧画面不是世界。你盯着杯子的正面拍一张，背面、桌腿内侧、显示器后面那一叠纸，全部不在这张图里。如果桌宠的状态就是当前帧，它转一下头，杯子就从状态里蒸发。人不会这样。人转头再转回来，杯子还在原处，因为脑子里存的是房间，不是视网膜上此刻的光。

前几幕的像素世界模型用另一种方式丢杯子。第 12 课的 Oasis 500M 把记忆钉死在 32 帧滑窗里：窗外的东西只能靠权重先验重新发明。转 360 度回来房子还在，对那个架构是结构性做不到的。第 12 课已经把第三条出路写在纸上：把世界存成三维资产，让一致性由数据结构保证，而不是由网络的记性保证。本课就走这条路的感知半截。

这条路也有自己的坑，必须先划清：

1. 单目深度看起来像三维，其实是一张"每个像素离相机多远"的图。换一个机位，你没有一份公共坐标系，杯子会在两套深度图里各活一次，对不上。桌宠要的是"同一只杯子一个坐标"，不是"每张照片自己猜一次远近"。
2. 多视图重建能给出公共坐标，但默认假设场景是静的。人伸手把杯子挪了，或者杯子自己倒了，一份冻住的点图不会自己改。它回答"刚才那几张照片里的桌子长什么样"，不回答"下一秒会怎样"。
3. 好看的新视角渲染（NeRF、3D 高斯泼溅）经常被宣传成世界模型。它们能围着一张桌子转圈看，画面可以很真。相机移动不是桌宠的动作：桌宠的动作是推、抓、让开，不是把虚拟相机拖到另一侧。渲染器没有 $a_t$，也没有 $s_{t+1}$。

所以本课要解决的具体问题是：给你 8 到 20 张自己桌子的照片（或一份公开室内序列），你要得到一份可查询的三维状态，并量出它在三件事上的表现：相机轨迹是否自洽、同一物体在不同视角下的三维误差、故意拿掉一张侧面图之后被遮挡的那一面还在不在。量完之后，你还得能当面说清：这份状态缺什么，才到不了世界模型。

## 3. 准备

- 第 19 课的手艺带着：一次只改一个条件、对照要写下来、数字带样本量和口径。本课没有训练曲线，但误差表按同一纪律填。第 12 课的长时程一致性实验最好还记得，本课的"转头杯子还在"就是它的三维版。
- 一台有 NVIDIA 显卡的 Linux 机器最顺。24GB 卡跑 20 张图很宽裕。12GB 卡先用 8 到 12 张，显存不够就降到仓库 demo 默认的厨房序列里抽 8 张。没有 CUDA 也能克隆仓库、读源码、在 [官方 Space](https://huggingface.co/spaces/facebook/vggt) 上传照片看点云；本地 CPU 推理能跑，只是慢到不适合当主路径。
- Python 虚拟环境。仓库 `requirements.txt` 钉死了 `torch==2.3.1` 和 `torchvision==0.18.1`，不要装进你前几课的 Dreamer 环境里，单独建一个。
- 磁盘：权重第一次从 Hugging Face 拉 `facebook/VGGT-1B`，按 fp32 估大约 5GB 量级；自己的桌子照片几乎不占空间。需要能访问 Hugging Face。
- 8 到 20 张桌子照片。用手机即可，协议写死：围着桌子走半圈到一圈，每一步大概 15 到 30 度；每张都要看见同一只杯子（或同一个马克杯）的一部分；至少两张拍到杯子侧面或背面；关掉美颜和超广角畸变（普通广角可以）；手尽量稳，不要边走边拍糊成一团。拍完按拍摄顺序编号，`00.jpg` 到 `19.jpg`。没有桌子或不方便拍，用仓库自带的 `examples/kitchen/images/`（25 张室内厨房图）或 `examples/room/images/` 顶上，主实验数字改成"公开室内序列"，桌子那一行留空并写明。
- 许可：教学用的 `facebook/VGGT-1B` 在 Hugging Face 上标的是 CC-BY-NC-4.0，非商业。仓库 2025 年 7 月另发了 `VGGT-1B-Commercial`，要申请，本课不需要。读代码、跑推理、写作业都按非商业权重走。
- 心理准备：本课**禁止从头训练 VGGT**。论文原训练是 64 张 A100、9 天、16 万步，那不是本课预算。仓库 `training/` 里后来补了微调入口，那是给以后有数据的人用的，本课所有命令都停在推理和可视化。

## 4. 学习目标

1. 白纸画出 VGGT 的数据流：多张 RGB 怎么进、交替注意力在哪两层之间切换、四个头各自吐什么，并标出参考系钉在第一张图上；
2. 用点图的定义解释：为什么同一只杯子在第 3 张和第 11 张里的像素，应该落在同一个三维坐标附近；
3. 对照 DUSt3R 的成对点图加全局对齐、MASt3R 的匹配头、VGGT 的一次前向，说出"两张以上怎么对齐"这件事各自交给谁；
4. 在自己的桌子（或厨房序列）上跑通官方推理，导出相机和点图，拖虚拟相机找出至少一处"转过去杯子变薄或漂走"的失败；
5. 填完三列表：相机轨迹是否自洽、同一物体跨视角三维误差、拿掉一张侧面图后遮挡面还在不在；
6. 口头把重建、新视角渲染、世界模型三个词拆开，说清桌宠缺的是哪两样（动作条件、时间更新）。

## 5. 原理

六个机制。每个仍按老节奏：为什么需要、怎么运转、精确定义、代码落在哪、怎么证明做对了。第 7 节会先让你玩到失败，再把失败扣回这里的公式。

### 5.1 一帧不是世界：持久性从数据结构来

桌宠转头，是相机在动，不是杯子在蒸发。如果状态等于当前观察 $o_t$，那么任何此刻不在画面里的东西，数学上已经不存在。第 12 课的滑窗生成器把这个问题推到了极限：条件里只有最近 $K$ 帧，窗外的一致性只能靠先验撞大运。把世界写成三维点，等于把记忆从"网络权重里的印象"搬进一份可以索引的几何。转头变成一次查询：新相机射线去点图里取样，杯子的坐标不需要被重新发明。

类比：一张纸地图对一座城市。你面向东看，西边的火车站不在视野里，但地图上还在。类比失效处也很硬：纸地图是人测过的，误差有界；本课这份"地图"是网络从照片里估出来的，没拍到的那一面可能被补全，也可能被糊成一层薄膜。第 7 节的遮挡消融就是来量这层薄膜。

持久性和预测是两件事。持久性问的是 $s$ 里有没有刚才看见过的杯子；预测问的是给了动作 $a$ 之后 $s$ 怎么变。本课只做前一件。写成条件分布，世界模型要的是

$$
P(s_{t+1} \mid s_t, a_t)
$$

VGGT 做的是

$$
(I_1,\ldots,I_N) \mapsto (\mathbf{g}_i, D_i, P_i, T_i)_{i=1}^{N}
$$

右边没有 $a$，也没有 $t+1$。输入是一组无顺序要求的照片（除了第一张当原点），输出是这组照片共享的静态几何。你在第 7 节拖虚拟相机时，动的是查询视角，不是桌子上的物体。

验证这件事不看损失曲线，看一个操作：把输入里那张拍杯子背面的照片拿掉，再推理一次，问背面的点还在不在。还在，说明模型用别的视图和先验补上了；不在，说明这份状态的覆盖率就是"被拍到的那些面"。两种结果都合格，只要你写进表里。

### 5.2 点图：把每个像素写成同一个坐标系里的点

传统多视图立体（MVS）先要相机，再对每个像素三角化。DUSt3R（Wang 等，[arXiv:2312.14132](https://arxiv.org/abs/2312.14132)）把这件事反过来：直接回归点图，相机事后再从点图里解。点图 $P$ 和输入图一样大，位置 $\mathbf{y}$ 上写的是该像素对应的三维点 $P(\mathbf{y})\in\mathbb{R}^3$。关键约束是**视点不变**：两张图的点图都写在第一张相机的坐标系里。于是两张图里那只杯子，应对到同一坐标系里彼此靠近的两个三维点。

VGGT 沿用这个定义，并同时预测深度图 $D$ 和相机 $\mathbf{g}$。深度 $D_i(\mathbf{y})$ 是第 $i$ 个相机下该像素的正深度；点图 $P_i(\mathbf{y})$ 仍在第一相机坐标系。二者由针孔几何连着：已知 $D_i$、内参、外参，可以把像素反投影到第一相机坐标系，得到另一份点图。论文在 ETH3D 上报告，推理时用"深度头加相机头再反投影"比直接用点图头更准（Overall Chamfer：0.677 对 0.709；DUSt3R 全局对齐后是 1.005，MASt3R 是 0.826）。训练时三个量都监督，推理时选更准的那条组合。仓库 README 把这件事写进了推荐用法。

相机 $\mathbf{g}$ 用 9 个数表示，沿自 VGGSfM：

$$
\mathbf{g} = [\mathbf{q},\mathbf{t},\mathbf{f}]
$$

$\mathbf{q}\in\mathbb{R}^4$ 是旋转四元数，$\mathbf{t}\in\mathbb{R}^3$ 是平移，$\mathbf{f}\in\mathbb{R}^2$ 是垂直和水平视场角。主点假定在图像中心。第一张图的外参被钉成恒等：$\mathbf{q}_1 = [0,0,0,1]$，$\mathbf{t}_1 = [0,0,0]$。尺度也钉死：所有三维点到原点的平均欧氏距离归一化成 1，平移和深度跟着缩。没有这步，同一组照片可以整体放大缩小，网络不知道该吐哪一个合法解。读代码时注意：论文把这 9 维写作 $[\mathbf{q},\mathbf{t},\mathbf{f}]$，仓库 `pose_enc.py` 的实际拼接顺序是绝对平移、四元数、视场角（类型名 `absT_quaR_FoV`）。拆编码以代码为准。

代码落点：`vggt/utils/pose_enc.py::pose_encoding_to_extri_intri` 把 9 维编码拆回 OpenCV 约定的 camera-from-world 外参矩阵和 $3\times 3$ 内参；`vggt/utils/geometry.py::unproject_depth_map_to_point_map` 做反投影。第 7 节量"同一杯子跨视角误差"时，用的就是这两步的输出。

验证：第一张图解出来的外参应当接近恒等；同一只杯子在两张图上点出来的像素，反投影之后的三维距离应当远小于杯子本身的尺度。桌面上一只 8 厘米口径的杯子，两个视角给出的中心如果差出十几厘米，这份状态就还不能用来伸手。

### 5.3 从成对拼图到一次看完全部

DUSt3R 一次只吃两张图，吐一对点图，多于两张时要做全局对齐：把每对点图变到同一个参考系，迭代到互相咬合。MASt3R（Leroy 等，[arXiv:2406.09756](https://arxiv.org/abs/2406.09756)）在 DUSt3R 上加了一个稠密局部特征头和匹配损失，并给出一种快速互逆匹配，把"配对"从二维特征近似拉回三维。它在 Map-free 定位上把当时最好方法的 VCRE AUC 绝对提高了约 30 个百分点（论文摘要口径）。多图场景里，它仍然要靠成对结果再拼。

VGGT 的赌注是：把 $N$ 张图一次送进同一个 Transformer，让网络自己在全局注意力里完成对齐，测试时不再必须做全局优化。论文 Table 1 在 RealEstate10K（训练没见过）和 CO3Dv2 上用 10 张随机帧比相机位姿，前向模式 AUC@30 分别为 85.3 和 88.2，耗时约 0.2 秒（单卡 H100）；加上 BA 后到 93.5 和 91.8，约 1.8 秒。DUSt3R / MASt3R 同表里是 67.7 / 76.4（Re10K）和 76.7 / 81.8（CO3Dv2），耗时约 7 到 9 秒，因为它们还要跑对齐。这些数字是论文在标准基准上的结果，不是你桌子上的验收线。你桌子没有真值位姿，第 7 节改用量相对误差。

代价也清楚。全局自注意力的显存随帧数和 token 数涨。论文 Table 9 只计骨干、分辨率 $336\times 518$、H100 加 FlashAttention v3：8 张 0.11 秒 / 3.23GB，20 张 0.31 秒 / 5.58GB，50 张 1.04 秒 / 11.41GB，200 张 8.75 秒 / 40.63GB。相机头大约再加骨干 5% 时间和 2% 显存；每个 DPT 头平均每帧约 0.03 秒、0.2GB。消费卡没有 H100 那么省，同一张数请按 2 到 3 倍显存预留。2026 年 5 月仓库修过一处中间张量没释放的实现问题，同一显存预算大约能多塞 2 到 3 倍帧，细节以仓库 Updates 为准。

代码落点：`vggt/models/vggt.py::VGGT.forward` 先走 `aggregator`，再分头预测。`demo_colmap.py` 里的 `--use_ba` 是可选后处理，默认关。本课主路径不升 BA，先看前向结果有多能用。

验证：同一组照片，打乱第 2 张以后的顺序再跑一次，第一张固定。对齐到同一参考系之后，相机轨迹应当几乎重合。若顺序一变杯子飞走，说明你其实在看网络对输入排列的敏感，不是在看桌子。

### 5.4 VGGT 怎么一次吐出相机、深度、点图和轨迹

网络本身几乎没有三维归纳偏置。论文原话是：除了在"逐帧注意力"和"全局注意力"之间交替，它就是一个普通的大 Transformer，靠大量带三维标注的公开数据学会几何。总参数约 12 亿。输入每张图先被 DINOv2（默认 `dinov2_vitl14_reg`）切成 patch token，分辨率默认最长边 518、patch 14，所以一张图大约是 $37\times 37$ 量级的 token。每张图再拼上 1 个相机 token 和 4 个 register token。第一张图的相机 token、register token 和其余帧用的是两套可学习向量，网络靠这个认出"谁是原点"。

然后是 24 层逐帧自注意力和 24 层全局自注意力交替（`aa_order = ["frame", "global"]`）。逐帧层只在一张图内部混合，全局层让所有图的 token 互相看见。没有 cross-attention。消融（论文 Table 5，ETH3D 点图）显示：纯全局、或改成每帧去看所有其他帧的 cross-attention，都不如这种交替。直观上，逐帧层负责把一张图内部的几何理顺，全局层负责把多张图对到同一个房间。

四个头挂在聚合后的 token 上：

- 相机头：从每张图的相机 token 再走 4 层自注意力加线性层，吐 9 维 $\mathbf{g}$。实现是 `vggt/heads/camera_head.py`。
- 深度头、点图头：都是 DPT（密集预测 Transformer 头），把 token 还原成和输入对齐的图。深度头输出通道 2（深度加置信度），点图头输出通道 4（XYZ 加置信度）。实现是同一个 `vggt/heads/dpt_head.py`，用 `output_dim` 区分。
- 轨迹头：在稠密特征上跑一个 CoTracker2 风格的跟踪器，给定查询像素，给出它在所有图里的二维对应。实现是 `vggt/heads/track_head.py`。训练时查询默认落在第一张图，推理时可以点任何一张。

训练损失是四项相加：

$$
\mathcal{L} = \mathcal{L}_{\mathrm{camera}} + \mathcal{L}_{\mathrm{depth}} + \mathcal{L}_{\mathrm{pmap}} + \lambda \mathcal{L}_{\mathrm{track}}
$$

$\lambda = 0.05$。相机项是预测 $\hat{\mathbf{g}}$ 对真值的 Huber；深度和点图带不确定性加权，并加了梯度项（让边缘别糊）；轨迹项是对应点的欧氏误差，外加可见性的二元交叉熵。论文 Table 6 的多任务消融：拿掉相机监督、深度监督或轨迹监督，ETH3D 点图都会变差。训练时让网络同时说这几件互相能换算的事，比只说一件更准。推理时你仍然可以只用深度加相机。

这些损失和超参是论文训练设定，不是你要跑的命令。本课加载的是已经训好的 `facebook/VGGT-1B`。

验证分两层。层一：`predictions` 字典里该有的键都在，`pose_enc` 形状是 `[1, S, 9]`，`world_points` 是 `[1, S, H, W, 3]`，`depth` 是 `[1, S, H, W, 1]`。层二：viser 里相机椎体围着桌子转，点云不是一片墙，杯子大致是个圆柱而不是一张纸。层二过了再进入第 7 节的量化。

### 5.5 重建、渲染、世界模型：三个词不要混

3D 高斯泼溅（Kerbl 等，[arXiv:2308.04079](https://arxiv.org/abs/2308.04079)，TOG 2023）把场景写成一堆三维高斯：中心、各向异性协方差、不透明度、球谐颜色。渲染是把高斯泼到图像平面上，1080p 能到实时。它需要先有相机（经典流程用 COLMAP）。VGGT 的 `demo_colmap.py` 可以把前向结果存成 COLMAP 的 `sparse/`，README 写明这份文件能直接喂给 [gsplat](https://github.com/nerfstudio-project/gsplat)。那是"用更好的几何初始化去训一个渲染器"，不是"世界模型训好了"。

三件东西的输入输出不同：

| 系统 | 吃什么 | 吐什么 | 动的是什么 | 桌宠能不能直接用 |
|---|---|---|---|---|
| VGGT | 一组照片 | 相机、深度、点图、轨迹 | 无；场景默认静 | 能当空间记忆的编码器 |
| 3DGS / NeRF | 照片加相机（或从 SfM 来） | 任意新视角的像素 | 虚拟相机 | 能看，不能推杯子 |
| 世界模型 | 状态加动作 | 下一状态（或下一观察）的分布 | 智能体的动作 | 能在脑子里先推一把再动手 |

第 12 课写 Marble 时已经下过判断：它把生成结果存成显式三维资产，转身一万度世界也不变，因为一致性由数据结构保证；代价是动作退化成相机移动，几乎没有动力学。本课的 VGGT 点图是同一条路上更瘦的一截：连"生成一张好看的新视图"都不是主任务，主任务是把几何估出来。你在 viser 里拖相机，看到的是点被换了个角度看，不是模型在模拟"如果我把手伸过去"。

类比失效写清楚。点图像一张钉在桌上的针板，每根针是一个点。你围着桌子转，针还在。你用手把杯子挪开，针板不会自己拔掉旧针、在新位置插一排。第 21 课的 4D 状态才处理"人把杯子挪了，模型要更新"。本课验收口试就一句：重建不是世界模型，缺动作条件和时间。

### 5.6 没有真值时怎么量三件事

你的桌子没有 CO3D 那种标注。论文的 AUC@30、Chamfer 不能直接抄到作业里。本课用量相对、可复现的三个量，口径固定如下。

相机轨迹自洽。第一张图的外参应当接近恒等，平移范数接近 0，四元数接近 $[0,0,0,1]$。如果你是围着桌子匀速走的，相邻帧相机中心的间距应当大致平稳，不应出现某一跳突然飞出桌子尺度的十倍。把第 2 张以后打乱再跑，用 Umeyama（论文评 ETH3D 点云时用的同一类相似变换）把两次轨迹对齐，报告对齐后相机中心的中位距离。这个量没有"小于多少算论文级"的官方线；本课验收只要求你写得出数字、拍法、以及第一帧是否钉住。

同一物体跨视角三维误差。在至少两个视角里标出杯子口沿或杯把上同一个可辨认的点（像素坐标即可），用深度加相机反投影到第一相机坐标系，报告两点的欧氏距离。再标杯底一个点，用杯口到杯底的三维距离当尺子，把误差除以这根尺子，得到相对误差。绝对厘米数取决于你有没有量过杯子；相对误差不需要尺子。同一只杯子两个视角若相对误差经常大于 0.5（差出半个杯子），伸手会抓空。

遮挡面还在不在。先用全套照片跑一次，在点图里找到只被侧面图看见的那一块（杯子背面、显示器侧面）。再把那张侧面图从输入里删掉，重跑。比较两份点图：背面还在，记"补全了"；背面空了或置信度掉到滑条滤掉的水平，记"覆盖率等于拍到的面"。两种结果都要配一张截图。这个实验测的是状态的完备性，不是生成是否好看。

单目对照。只用第一张图跑一次（仓库 README 写明模型没为单图任务专门训练，但可以直接推理，不必把图复制成一对）。单图点云往往只有朝向镜头的那一层皮。多图点云应该长出侧面。把两份点云并排截图，这就是互动里"只看当前帧的深度"对"多视图点图"的实物差。

代码落点都在第 7 节逐步给出。原理上记住三句话：第一帧是原点，点要在同一个坐标系里比，没拍到的面允许模型不会变魔术。

## 6. 源码导读

克隆后按这个顺序读，每个文件带着问题进去。路径都以仓库根目录为准。

| 文件 | 管什么 | 带着什么问题读 |
|---|---|---|
| `vggt/models/vggt.py` | `VGGT` 类，整网入口 | `forward` 在什么条件下给 `images` 补 batch 维？四个头各自往 `predictions` 里写哪些键？为什么相机、深度、点图包在 `autocast(enabled=False)` 里？ |
| `vggt/models/aggregator.py` | 交替注意力骨干 | 默认 `depth=24`、`embed_dim=1024`、`patch_embed="dinov2_vitl14_reg"`；`camera_token` 和 `register_token` 为什么第一维是 2？`aa_order` 默认是什么？ |
| `vggt/heads/camera_head.py` | 相机头 | 输入是聚合 token 列表，输出是迭代中的 pose encoding 列表，调用方为什么取 `[-1]`？ |
| `vggt/heads/dpt_head.py` | 深度头和点图头的共用实现 | `VGGT.__init__` 里点图 `output_dim=4`、深度 `output_dim=2`，多出来的通道是置信度吗？激活函数 `inv_log` / `exp` 各自防什么？ |
| `vggt/heads/track_head.py` | 轨迹头 | 没有 `query_points` 时 `forward` 会不会跑它？（答案在 `vggt.py`：不会） |
| `vggt/utils/load_fn.py` | 读图与预处理 | `load_and_preprocess_images` 默认 `mode="crop"`，宽被拉到 518，高按 14 的倍数对齐后中心裁；`pad` 模式何时该用？COLMAP 导出为什么另走 `load_and_preprocess_images_square`？ |
| `vggt/utils/pose_enc.py` | 9 维编码与外参内参互转 | 外参约定是 OpenCV 的 camera-from-world；主点怎么定？视场角和焦距哪一行互转？ |
| `vggt/utils/geometry.py` | 反投影、SE3 求逆 | `unproject_depth_map_to_point_map` 的输入形状必须对上 `demo_viser.py` 里那份 `pred_dict` |
| `demo_viser.py` | 交互点云 | 默认 `--image_folder examples/kitchen/images/`；`--use_point_map` 关掉时用深度反投影，打开时用点图头。GUI 里 Confidence Percent 滤掉的是最低多少百分比的点？ |
| `demo_gradio.py` | 浏览器上传 | 适合先看效果；本课量化不要只停在这里，数字要从自己保存的张量出 |
| `demo_colmap.py` | 导出 COLMAP | 图片必须放在 `SCENE_DIR/images/`；`--use_ba` 默认关；注释写明 BA 路径用的是 VGGSfM 跟踪器而不是 VGGT 自己的 track head |
| `visual_util.py` | 点云转 glb、可选天空分割 | Gradio 出模型预览走这里 |
| `training/` | 微调入口 | 本课不跑。README 写的是 `torchrun --nproc_per_node=4 launch.py`，且默认冻结 `aggregator`、从预训练权重 resume。知道有这扇门即可 |

`VGGT.forward` 的输入输出以源码为准。`images` 接受 `[S, 3, H, W]` 或 `[B, S, 3, H, W]`，数值范围 $[0, 1]$。一次前向返回的字典至少包括：

- `pose_enc`：`[B, S, 9]`
- `depth`、`depth_conf`：`[B, S, H, W, 1]` 和 `[B, S, H, W]`
- `world_points`、`world_points_conf`：`[B, S, H, W, 3]` 和 `[B, S, H, W]`
- 推理模式下还会带回 `images`，方便上色

`demo_viser.py` 会再调用 `pose_encoding_to_extri_intri` 得到 `extrinsic`（`S, 3, 4`）和 `intrinsic`（`S, 3, 3`），缺省用深度反投影填 `world_points`。论文和 README 都建议你默认走这条路，把 `--use_point_map` 留作对照。

Aggregator 里有一处以后改结构会碰到的设计：`camera_token` 形状是 `(1, 2, 1, embed_dim)`，`register_token` 是 `(1, 2, 4, embed_dim)`。索引 0 给第一帧，索引 1 给其余帧。这就是 5.2 节"第一张图是世界原点"在参数里的钉子。把它两套合成一套，网络就失去标记原点的通道，点图会在某个未定义的坐标系里漂。第 11 节的改造清单会回到这里。

读代码时不要被 `training/loss.py` 带跑。损失是训练用的，本课加载的权重已经最小化过它。你要会的是：推理时哪个头的输出被拿去可视化，哪个张量被拿去填误差表。

## 7. 实验

先玩到失败，再回头对 5.5 节的那张表。本节所有 bash 围栏都假定你已经在仓库根目录、且用的是本课单独的虚拟环境。命令与 2026-08 核对过的仓库 README 一致；README 里 `git clone` 写的是 SSH 地址，下面给出 HTTPS 等价写法，避免没配密钥的人卡在第一步。

### Step 0: 建环境并克隆

```bash
python3 -m venv .venv-vggt
```

```bash
source .venv-vggt/bin/activate
```

```bash
git clone https://github.com/facebookresearch/vggt.git
```

```bash
cd vggt
```

```bash
pip install -r requirements.txt
```

```bash
pip install -r requirements_demo.txt
```

预期：`requirements.txt` 装上 `torch==2.3.1`、`torchvision==0.18.1`、`numpy==1.26.1`、`Pillow`、`huggingface_hub`、`einops`、`safetensors`。`requirements_demo.txt` 再补 Gradio、viser、以及可视化依赖。若 `torch==2.3.1` 和你机器的 CUDA 对不上，不要随意改成"随便一个新版本"之后还声称在跑官方权重；先读仓库当前 README 和 Issue，换能装上的邻近版本时在笔记里写明实际版本。Mac 走 CPU 也能装，只是下一步会慢。

### Step 1: 先用仓库厨房跑通可视化

不要一上来就拍自己的桌子。仓库自带 `examples/kitchen/images/`，`00.png` 到 `24.png` 共 25 张室内图，目录结构就是 demo 要的那种"文件夹里只有图"。viser 脚本默认也指向这里。

```bash
python demo_viser.py --image_folder examples/kitchen/images/
```

第一次运行会从 Hugging Face 拉 `facebook/VGGT-1B`，需要联网，可能要几分钟。权重落地之后，终端会打印 `Starting viser server on port 8080`。本机浏览器打开 `http://127.0.0.1:8080`。

预期：几秒到几十秒内（论文说重建本身通常不到 1 秒，点云渲染可能要几十秒，图多时更慢）出现带颜色的点云和相机椎体。厨房的台面、柜门应该能认出来，不是一片噪点。GUI 上有 Show Cameras、Confidence Percent、Show Points from Frames。默认会滤掉置信度最低的一部分点。

若你更想先在浏览器里上传，等价入口是：

```bash
python demo_gradio.py
```

或直接用官方 Space，不装 demo 依赖。Gradio 路线适合确认"模型能出点云"，本课后面的误差表仍以 viser / 自己保存的张量为准。

### Step 2: 拖虚拟相机，专门找失败

点云出来之后不要截一张好看的就收工。按下面的剧本玩，每条都写进笔记，找到了就截图，找不到也写"在这组图上没找到"。

1. 围着厨房转一圈。相机椎体应当大致落在一圈拍摄轨迹上，不应有某两帧对穿到房间对面。
2. 把视角拖到输入照片里没正面拍过的方向，盯一个小物体（杯子、瓶、把手）。它是保持体积，还是被拉成一张纸、裂成两半、跟着你的虚拟相机漂？
3. 把 Confidence Percent 从默认往上加，观察先消失的是哪些点：玻璃、反光、天空、运动模糊，还是物体本体。本体大面积被滤掉，说明这组图的几何置信度整体偏低。
4. 在 Show Points from Frames 里只显示单帧的点，再切回 All。单帧应当是一层朝向该相机的皮；All 应当能补上侧面。若 All 和单帧几乎一样，多视图没有真正融合。

这一步的目标就是找到"一转头杯子变薄 / 漂走"。找到了，5.1 节的类比失效处就落在你自己的截图上。没找到也不许改口说"VGGT 没有这个问题"：换更少的图、更少重叠、更强反光再试，Step 6 会强制制造一次失败。

### Step 3: 单张对多张，同一场景两份状态

另开一次，只用厨房的第一张图。把那一张拷到临时目录（demo 假定文件夹里都是图，不要把别的文件混进去）：

```bash
mkdir -p /tmp/vggt_kitchen1
```

```bash
cp examples/kitchen/images/00.png /tmp/vggt_kitchen1/
```

```bash
python demo_viser.py --image_folder /tmp/vggt_kitchen1 --port 8081
```

预期：单图也能出点云（README 的 Zero-shot Single-view Reconstruction 一节写明不必把图复制成一对），但侧面和背面会缺，物体像纸片。回到 8080 那个多图会话，同一物体应该更鼓。两份截图并排贴进笔记，标题写成"当前帧深度皮"对"多视图点图"。这就是本课互动要求的对照。桌宠如果只吃当前帧，它的状态就是 8081 里那张皮。

### Step 4: 换成自己的桌子

按第 3 节的拍摄协议拍 8 到 20 张，放到 `desk_scene/images/`（名字自定，结构必须是"场景目录下有一个只装图片的 `images/`"，和 COLMAP 导出脚本一致）。viser 不强制子目录叫 `images`，它只要一个只装图的文件夹：

```bash
python demo_viser.py --image_folder desk_scene/images/ --port 8082
```

拍不了桌子，用 `examples/room/images/` 替换，笔记里写"公开室内序列，不是我的桌子"。预期和 Step 1 相同，只是场景换成你认识的那张桌子。你应当能指着点云说"这是杯子、这是显示器、这是桌角"。指不出来，先检查：图是不是糊的、是不是每张杯子都不在画面里、是不是美颜把纹理抹掉了、是不是只在一个点站着连拍了 12 张（基线几乎为零，三角化退化成单目）。

对照一次点图头和深度反投影：

```bash
python demo_viser.py --image_folder desk_scene/images/ --use_point_map --port 8083
```

论文说深度加相机通常更准。你不一定能在桌上复现 ETH3D 的 Chamfer 差，但可以看：哪一份杯子更圆、哪一份桌沿更直、哪一份飞点更少。把观察写进表，不要编造没量过的 Chamfer。

### Step 5: 导出张量，量相机和同物误差

可视化不能代替数字。在仓库根目录用下面这份最小脚本做一次前向，把相机和点存下来。它只调用 README Quick Start 里出现过的符号：`VGGT.from_pretrained`、`load_and_preprocess_images`、`pose_encoding_to_extri_intri`、`unproject_depth_map_to_point_map`。

```python
import json
from pathlib import Path

import numpy as np
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

folder = Path("desk_scene/images")
names = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
assert 8 <= len(names) <= 20, f"本课主实验要 8-20 张，现在是 {len(names)}"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
images = load_and_preprocess_images([str(p) for p in names]).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        pred = model(images)

extrinsic, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
ext = extrinsic[0].float().cpu().numpy()
intr = intrinsic[0].float().cpu().numpy()
depth = pred["depth"][0].float().cpu().numpy()
pts = unproject_depth_map_to_point_map(depth, ext, intr)

out = Path("desk_scene/pred")
out.mkdir(parents=True, exist_ok=True)
np.save(out / "extrinsic.npy", ext)
np.save(out / "intrinsic.npy", intr)
np.save(out / "depth.npy", depth)
np.save(out / "points.npy", pts)
np.save(out / "points_head.npy", pred["world_points"][0].float().cpu().numpy())
np.save(out / "depth_conf.npy", pred["depth_conf"][0].float().cpu().numpy())

cam_centers = []
for i in range(len(ext)):
    R, t = ext[i][:, :3], ext[i][:, 3]
    # camera-from-world: x_cam = R x_world + t, center satisfies R c + t = 0
    c = -R.T @ t
    cam_centers.append(c.tolist())
cam_centers = np.asarray(cam_centers)
step = np.linalg.norm(np.diff(cam_centers, axis=0), axis=1)
report = {
    "n_images": int(len(names)),
    "files": [p.name for p in names],
    "first_translation_norm": float(np.linalg.norm(ext[0][:, 3])),
    "cam_step_median": float(np.median(step)) if len(step) else None,
    "cam_step_max": float(np.max(step)) if len(step) else None,
    "cam_step_max_over_median": float(np.max(step) / (np.median(step) + 1e-8)) if len(step) else None,
}
(out / "camera_stats.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
print(json.dumps(report, indent=2, ensure_ascii=False))
```

公开厨房序列张数是 25，超过 20。用厨房顶替时把断言改成 `len(names) >= 8`，并在笔记写明你实际用了哪些编号。不要为了过断言去复制图片凑数。

预期：`first_translation_norm` 应当接近 0（第一帧是原点）。`cam_step_max_over_median` 若到了十几，多半是某一帧外参飞了，或拍摄时你突然走了很远。围着一张桌子匀速走，这个比值通常是个位数。把它抄进误差表，附上拍摄是"站着转"还是"走半圈"。

同物误差用手点像素。先用任意看图工具打开两张相隔较远的照片，记下杯子口沿同一处的像素 $(u, v)$（注意：脚本里的图已经被 `load_and_preprocess_images` 缩到宽 518 并可能中心裁高，像素要在预处理后的图上点，或者你自己按 `load_fn.py` 的规则把原图像素换算过去）。假设视角 `i`、`j` 上点了 `p_i`、`p_j`：

```python
import numpy as np

pts = np.load("desk_scene/pred/points.npy")  # (S, H, W, 3)
i, ui, vi = 2, 240, 180
j, uj, vj = 9, 260, 200
a = pts[i, vi, ui]
b = pts[j, vj, uj]
print(a, b, float(np.linalg.norm(a - b)))
```

再点杯底一个点，算杯口到杯底的三维长度当尺子，把上面的距离除以尺子。相对误差写进表。点不准是这个量最大的噪声源：同一处口沿偏了 10 个像素，误差表会跟着抖。每个物体至少点两对视角，取中位数。

### Step 6: 拿掉侧面图，看遮挡面还在不在

先在全套点云里认定一块"主要靠某一张侧面图才看见"的区域：杯子背面、显示器侧面、椅腿内侧。记下那张图的文件名，把它移出输入再跑一遍。

```bash
mkdir -p desk_scene_ablate/images
```

```bash
cp desk_scene/images/* desk_scene_ablate/images/
```

```bash
rm desk_scene_ablate/images/07.jpg
```

最后一条里的文件名换成你自己认定的那张侧面图。然后把 Step 5 脚本里的 `folder` 改成 `desk_scene_ablate/images`，输出改到 `desk_scene_ablate/pred`，再跑一次。两份 `points.npy` 不能直接逐像素减：输入集合变了，第一张如果没变，参考系还在，但被删的那一帧不再占一个索引。比较方法用眼睛加置信度：在 viser 里把全套和删减套用两个端口打开，拖到同一背面视角。

```bash
python demo_viser.py --image_folder desk_scene_ablate/images/ --port 8084
```

验收口径：

- 背面仍在，形状大致还是杯子：记"模型补全了未观测面"，截图。
- 背面空了，或只剩一层从正面透过去的薄膜：记"覆盖率等于拍到的面"，截图。
- 背面还在，但漂到桌子另一头：记失败，这比缺失更糟，因为桌宠会伸手去一个错误坐标。

三种都算本实验做完。禁止把"看起来挺真"写成"遮挡面重建成功"。

到这里你应该已经至少失败过一次：Step 2 的漂或变薄，或 Step 6 的背面失踪 / 漂走，或 Step 3 单图纸片。把失败扣回 5.5：你动的是查询相机，桌子上的杯子并没有收到动作；模型也没有 $t+1$。点图再完整，仍是一份静态状态。

### Step 7: 可选导出 COLMAP，以及一份误差表

想看这份几何能不能接到高斯泼溅上，用仓库提供的导出脚本。图片必须在 `SCENE_DIR/images/` 下，不要混进别的文件。

```bash
python demo_colmap.py --scene_dir=desk_scene
```

预期：`desk_scene/sparse/` 下出现 `cameras.bin`、`images.bin`、`points3D.bin`。加 BA 是可选的，命令是 README 原样：

```bash
python demo_colmap.py --scene_dir=desk_scene --use_ba
```

BA 更慢，12GB 卡上 20 张可能吃紧。本课不把 BA 后的观感当成主验收。gsplat 的训练命令在 README 的 Integration with Gaussian Splatting 一节（`examples/simple_trainer.py`），那是对照阅读，不是本课必做。就算你跑出了能围着桌子转的高斯场景，它仍然没有动作端口。

在 `desk_scene/NOTES.md` 留下这张表，空格必须填你自己的数：

```text
日期 / 机器 / CUDA 是否可用
仓库 commit
照片来源（我的桌子 / kitchen / room）与张数
拍摄协议（站转 / 走半圈 / 步距大概多少度）
first_translation_norm
cam_step_median / cam_step_max / 比值
同物相对误差（物体、视角对、中位数）
删掉的侧面图文件名
遮挡面结论（补全 / 缺失 / 漂走）+ 截图路径
单图 vs 多图：各一张截图路径
一句话：这份点图缺什么才到不了世界模型
```

三个月后只看这一页，应能复述整个实验。

## 8. 配置与预算

| 档位 | 输入 | 做什么 | 预算 | 用途 |
|---|---|---|---|---|
| 冒烟 | `examples/kitchen/images/` 全套 25 张，或从中抽 8 张 | `demo_viser.py` 出点云 | 首次下权重数分钟；之后单次前向在 24GB 卡上通常秒到十几秒 | 证明环境、权重、可视化通 |
| 本课主线 | 自己桌子 8 到 20 张，或 `examples/room` | viser 找失败 + Step 5/6 误差表 | 拍照 30 到 60 分钟；推理每次一分钟内（不含渲染） | 验收用的那张表 |
| 对照 | 同组照片加 `--use_point_map`，再加一次单图 | 两份点云并排看 | 多两次推理 | 分离"点图头"和"深度加相机" |
| 可选导出 | 同一 `desk_scene` | `demo_colmap.py`，BA 选做 | 前向很快；`--use_ba` 视张数到数分钟 | 接到 gsplat / COLMAP 工具，不计入本课必做 |
| 不跑 | 论文原训练、仓库 `training/` 微调 | 无 | 论文：64×A100、9 天、160K 步。微调示例是 4 卡 `torchrun`，默认冻骨干 | 知道存在，本课禁止当作业 |

权重与精度：`facebook/VGGT-1B` 约 12 亿参数，HF 模型卡标 F32。推理按 README 走：Ampere 及更新（compute capability 8.0 及以上）用 `bfloat16`，否则 `float16`。CPU 回退是源码里写好的，不是推荐路径。

显存经验（论文 Table 9 是 H100 骨干，消费卡请按 2 到 3 倍预留）：

| 输入张数 | 论文骨干显存（H100，336×518） | 本课建议 |
|---|---|---|
| 8 到 10 | 约 3.2 到 3.6GB | 12GB 卡的起点 |
| 20 | 约 5.6GB | 24GB 卡轻松；12GB 卡先关轨迹头、少开 viser 叠加 |
| 25（厨房全套） | 介于 20 与 50 之间 | 24GB 应当能过；OOM 就抽 00、03、06…每隔两张取一张 |

超参本课几乎没有可调的训练项。推理侧真正会改变结果的只有：输入是哪些图、第一张是哪张、预处理 `crop` 还是 `pad`、可视化滤掉多少低置信点、用点图头还是深度反投影。一次只改其中一样。

Mac / 无卡：Step 0 和读代码完整做；Step 1 改走 Hugging Face Space；Step 5 的脚本在 CPU 上能跑，20 张会慢到你去干别的事。误差表仍然要填，不能用 Space 的截图代替 `first_translation_norm`。

## 9. 验收

- [ ] 能在白纸上画出：DINOv2 切 token、第一帧专用相机 token、交替注意力、四个头、第一相机作原点；
- [ ] 独立把厨房或自己的桌子跑出 viser 点云，截图里能指出杯子（或厨房里一个可命名物体）和至少三个相机椎体；
- [ ] 找到并截下一处失败：转过去物体变薄、裂开、漂走，或单图只有一层皮；
- [ ] `camera_stats.json` 里 `first_translation_norm` 接近 0，相邻相机步长的中位数和最大值都写进表；
- [ ] 至少一对跨视角像素给出同物三维距离和相对误差（带视角编号和像素）；
- [ ] 做过一次"拿掉侧面图"，三种结论（补全 / 缺失 / 漂走）里记下一种，配删减前后截图；
- [ ] `NOTES.md` 按 Step 7 的字段填满；
- [ ] 口头验收：重建不是世界模型，缺动作条件和时间。追问"那 3DGS 呢"时，能答：它是新视角渲染器，动作退化成拖相机，不能用来规划推杯子。

口试不过，实验数字再漂亮也不算完成本课。桌宠用途就一句话：头转动时，身后的杯子必须还在状态里。你的点图要么做到了，要么你量出了它没做到、以及差在覆盖率还是精度。

## 10. 排错

| 症状 | 最可能的原因 | 怎么确认 | 怎么修 |
|---|---|---|---|
| `git clone` 用 SSH 报 Permission denied | 没配 GitHub 密钥 | 报错含 `Permission denied (publickey)` | 改用本课 Step 0 的 HTTPS 地址 |
| 装 `torch==2.3.1` 失败 | CUDA / Python 版本和钉死的轮子不匹配 | pip 报 no matching distribution 或 CUDA runtime 错 | 单独建 venv；对照官网装能用的 PyTorch 后把实际版本写进 NOTES，不要混进 Dreamer 环境 |
| 第一次 `from_pretrained` 卡住或超时 | Hugging Face 下载慢或需镜像 | 进程停在下载、无 GPU 活动 | 按 README 手动下载 `https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt`，再用 `load_state_dict_from_url` 或本地 `load_state_dict` |
| CUDA OOM | 张数太多或分辨率被 pad 成大方图 | `torch.cuda.OutOfMemoryError` | 先减到 8 张；关其他占用 GPU 的进程；viser 和推理不要叠两个 20 张会话 |
| viser 打开是空白或一直转圈 | 点太多，第三方渲染慢 | 终端已打印 server started，浏览器无点 | 把 Confidence Percent 拉高，先显示高置信点；图很多时等几十秒，README 已警告可视化比推理慢 |
| 点云是一片墙或杯子扁成纸 | 基线太小，或多视图没融合 | 相机椎体几乎叠在同一点；单帧和 All 几乎一样 | 重拍，人要真的围着桌子走；确认所有图都进了同一个文件夹且没有重复帧 |
| 点云炸开、尺度离谱 | 第一张图选得差（特写、大面积过曝）或图里有运动 | 第一帧椎体正常，其余飞到很远 | 换一张能看见整张桌子的图当 `00`；去掉有人入画、杯子被挪过的帧 |
| 同物误差大到超过杯子本身 | 像素点在预处理前的原图上，和张量对不上 | 用 `images.shape[-2:]` 核对高宽 | 在宽 518 的预处理结果上重点，或按 `load_fn.py` 把原图坐标换算过去 |
| 玻璃、屏幕全是飞点 | 反光、透明，几何先验失效 | 飞点集中在显示器和窗户 | README 允许把不想要的像素直接涂成 0 或 1 做粗掩膜，不必精确分割（见仓库 Issue 47） |
| `demo_colmap.py` 报 No images | 图不在 `scene_dir/images/` | 脚本打印的 `image_dir` 是空的 | 按 `examples/kitchen/` 的结构放：场景目录下只有 `images/` |
| CPU 上跑到半天 | 正常 | 设备打印是 `cpu` | 减到 8 张；或只在 Space 上看图，数字仍尽量在能借到的 GPU 上出 |
| 把 NeRF/3DGS 的漂亮旋转视频当成验收 | 验收对象错了 | 视频很真，但 NOTES 里没有三列误差 | 回到 Step 5/6，渲染视频最多当附录 |

## 11. 前沿与改造

同一问题，2024 到 2026 年公开系统怎么做。DUSt3R / MASt3R 把多视图几何收成"成对点图再对齐"，VGGT 收成"一次前向看完全部"。论文 Table 1 还列了几篇同期前向工作，都在尝试去掉 DUSt3R 的测试时优化。其中 CUT3R（Wang 等，CVPR 2025 Oral，[arXiv:2501.12387](https://arxiv.org/abs/2501.12387)）维护一份持续更新的三维状态，新帧进来就改，还能查询没看见的视角。那是第 21 课的主线，本课只记一句差别：VGGT 吃的是一批已经拍好的图，CUT3R 吃的是流。

仓库 Updates 在 2026 年 5 月指向 VGGT-Omega（项目页 [vggt-omega.github.io](https://vggt-omega.github.io/)），并修过中间张量占显存的问题。本课仍锚定 `facebookresearch/vggt` 和 `facebook/VGGT-1B`，Omega 当加餐，命令以它当天的 README 为准，不要把两边的权重混着比数字。

3DGS / nerfstudio / gsplat 在本课的位置是渲染层。VGGT 给出相机和点，gsplat 把点养成能实时画的高斯。World Labs 的 Marble 把这条路走到产品里：一致性满分，动力学接近零。第 12 课已经标过它的坐标。本课不要把"我训出了一团能转的高斯"写成世界模型进展。

缩小版和前沿版差在哪。规模上，你缺的是论文那 17 个带三维标注的数据集和 64 卡 9 天，钱能补。机制上，你缺的是动作条件和在线更新：这两样钱补不来，得换问题定义。桌宠要的下一块零件是"人把杯子挪了，状态跟着改"，不是"同一张静桌再多 100 张图"。

动手改造清单，全是推理级，禁止开训：

1. 第一帧选择。把最模糊、最特写、最远的一张分别设成 `00`，各跑 Step 5。预算：三次前向。预期：特写当原点时远处桌角误差变大，轨迹步长比值变差。失败判据：三份 `first_translation_norm` 都远大于 0，回头查你是不是用错了 `pose_enc` 的第一帧。
2. 张数扫描。同一条拍摄轨迹取 8、12、20 张（均匀抽）。预算：三次前向。预期：8 张时侧面更空，20 张时飞点可能变多但杯子更圆。失败判据：20 张反而比 8 张圆不起来，检查是不是后期加入了运动帧或重复帧。
3. 点图头对深度反投影。同一输入分别存 `points_head.npy` 和 `points.npy`，在杯子口沿用同一组像素比相对误差。预算：一次前向（Step 5 已经两个都存了）。预期：深度反投影的相对误差更小或相当，和论文 ETH3D 方向一致。失败判据：点图头明显更准，如实记录，这在桌上完全可能，因为 ETH3D 不是你的桌子。
4. 掩膜去反光。把显示器屏幕或窗户涂成常数 0，再跑。预算：一次涂改加一次前向。预期：飞点减少，屏幕不再被建成一道斜墙。失败判据：杯子跟着消失，说明你涂得太狠，掩膜碰到了物体。

顺手复现（方向，不是论文数字）：论文说"推理时深度加相机优于专用点图头"。改造 3 就是这句话在桌上的缩小版。能看到同方向趋势最好；看不到就写场景差异（反光桌面、少纹理墙），不要改口声称论文测错了。

## 12. 论文与延伸

下面四篇是本课必读，第五篇给第 21 课踩点。每篇带着问题读，读完要能用自己的桌子当例子回答。

1. VGGT: Visual Geometry Grounded Transformer（Wang, Chen, Karaev, Vedaldi, Rupprecht, Novotny，CVPR 2025 Best Paper，[arXiv:2503.11651](https://arxiv.org/abs/2503.11651)）。项目页 [vgg-t.github.io](https://vgg-t.github.io/)，代码 [facebookresearch/vggt](https://github.com/facebookresearch/vggt)。带着四个问题读：点图为什么定义在第一相机坐标系，而不是每张图自己的相机坐标系？交替注意力相对"纯全局"和"cross-attention"到底赢在 ETH3D 的哪一列（Table 5）？训练时三个互相能换算的量都监督、推理时却推荐深度加相机，论文怎么解释这种不一致？Table 1 里"前向已经超过带优化的 DUSt3R / MASt3R"和"再加 BA 还能再涨"同时成立，对你要不要在桌上开 `--use_ba` 意味着什么？
2. DUSt3R: Geometric 3D Vision Made Easy（Wang, Leroy, Cabon, Chidlovskii, Revaud，[arXiv:2312.14132](https://arxiv.org/abs/2312.14132)）。带着三个问题读：它如何用成对点图同时覆盖单目和双目，而不先要内参外参？多于两张图时全局对齐优化的是什么、失败时房间会怎样裂开？把 DUSt3R 的"先成对再拼"换成 VGGT 的"一次看完全部"，你丢掉的是可扩展的成对缓存，还是测试时那几秒优化？
3. Grounding Image Matching in 3D with MASt3R（Leroy, Cabon, Revaud，[arXiv:2406.09756](https://arxiv.org/abs/2406.09756)）。带着三个问题读：匹配为什么被写成三维任务而不是二维描述子？新加的稠密特征头和互逆匹配各治 DUSt3R 的哪一种不准（鲁棒有了、精度不够）？Map-free 上那 30 个百分点的绝对提升，对应的是相对位姿、还是你桌上那种稠密点图？
4. 3D Gaussian Splatting for Real-Time Radiance Field Rendering（Kerbl, Kopanas, Leimkühler, Drettakis，TOG 2023，[arXiv:2308.04079](https://arxiv.org/abs/2308.04079)）。带着三个问题读：三维高斯比 NeRF 的隐式场多了哪三样东西，才换来 1080p 实时？它的输入为什么几乎总是先有 SfM 相机？把 gsplat 接到 VGGT 的 COLMAP 导出之后，你得到的是更好看的查询，还是 $P(s_{t+1}\mid s_t, a_t)$？最后一问的答案必须是否定的后半句。

选读：CUT3R（[arXiv:2501.12387](https://arxiv.org/abs/2501.12387)），只读摘要和引言里与 VGGT 的对照，带着问题：持续状态和本课这份"一批照片冻住的点图"差在谁能处理杯子被挪走。仓库 README 的 Research Progression 把 Deep SfM Revisited、PoseDiffusion、VGGSfM、CoTracker 串成一条族谱，想知道 VGGT 从哪来的，按那张表回看即可。

现在整个系统长这样：前五幕给你的仍是"观察压成状态、按动作想未来、选动作"的循环，第六幕刚给状态换上了一个可以转头查询的三维底座。底座是静的。人把杯子挪了，它不会改；你伸手推杯子，它也不会先在脑子里滚出一个下一状态。第 21 课要解决的麻烦正是这个：空间随时间变，状态怎么连续更新，而不是每帧重造一个新场景。
