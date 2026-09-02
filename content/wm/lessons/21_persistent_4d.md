---
id: 21_persistent_4d
title: "边走边改的 4D 状态"
summary: "每帧都像真的，却不是同一个空间随时间变化，算哪门子世界模型？"
unit: spatial
play_tools: []
checkpoints:
  - "一条带时间戳的漂移曲线。"
  - "物体恒常测试记录。"
---

# 第 21 课：边走边改的 4D 状态

> 类型：实战/体验（跑 CUT3R 公开权重做在线重建与物体恒常测试，**不从头训练**）<br>
> 建议周期：2-3 天<br>
> 硬件：单张 24GB 卡可跑 512 档官方推理；224 档更省显存；无 NVIDIA 卡可完成阅读、协议设计和书面区分，官方 demo 依赖 CUDA RoPE 核，Mac/CPU 跑不通<br>
> 锚定仓库：[CUT3R/CUT3R](https://github.com/CUT3R/CUT3R)（CVPR 2025 Oral，官方实现）；对照第 20 课的 [facebookresearch/vggt](https://github.com/facebookresearch/vggt)<br>
> 产物：一条带时间戳的漂移曲线、一份「挡住 / 移走 / 转头」物体恒常灯记录、一段书面区分（感知状态更新还不是动作条件世界模型）

## 1. 这一课做什么

第六幕要给状态装上空间和物体。第 20 课用 VGGT 把一组桌面照片压成相机位姿加
点图：多张图进，一份 3D 几何出。那是 encoder。它回答「这几张图里的房间长什么样」，
不回答「下一秒我会看到什么」，也不记住「刚才那只杯子被手挡住了」。桌宠转一下头，
若系统把当前这一批图重新推一遍，身后的杯子要么被挤出输入窗口，要么被当成一张新
场景从头发明。

本课换上一个会**边走边改**的零件：CUT3R（Continuous Updating Transformer for
3D Reconstruction）。它维护一份持续的 3D 状态，新帧进来就更新，点图画在同一个
世界坐标系里，还可以用虚拟相机去问没看见的角落。你要量两件事：杯子被手挡住
两秒，状态里它还在不在；绕桌子走一圈再走回来，同一只杯子的三维位置漂了多少。

贯穿主干在这一课只动「观察先压成状态」这一截，而且压出来的状态第一次带上了时间：
同一份 3D 内容随观察流连续改写。动作条件和下一步预测都还没接上。做完你必须能
把这句话写清楚并守住：CUT3R 更新的是感知状态 $s_{t}=f(s_{t-1}, I_{t})$；世界模型
要的是 $P(s_{t+1} \mid s_t, a_t)$。两者差一个动作，还差一步对未来的预测。第 22
课才会问视频基础模型能不能补上动作，第 23 课才会把「杯子」从一袋匿名 token 里
拆成物体槽。

不做这一课，后面接到真桌子上会出现两种对称的错。一种是转头就忘：杯子离开画面
就被从状态里删掉，桌宠回头以为桌上没东西。一种是死抱第一帧：人把杯子挪走了，
点云还钉在原地。物体恒常灯就是用来同时抓住这两种错的。

术语速查：

| 术语 | 一句人话 |
|---|---|
| 点图（pointmap） | 每个像素对应一个三维点，CUT3R 同时输出相机坐标系和世界坐标系两份 |
| 持久状态（persistent state） | 模型内部一直带着走的那组 token，新图用来改它，而不是每来一组图就另起炉灶 |
| 在线重建 | 帧按时间到达，每到一帧立刻出几何和位姿，不必等整段视频收齐再优化 |
| 世界系 | 本课里取第一帧相机坐标系当世界原点，后面所有点图都画在这套坐标里 |
| 物体恒常 | 东西被挡住或暂时离开画面，状态里它仍在；真被拿走，状态才该改 |
| 漂移 | 同一物理点在后续帧里估出的三维位置慢慢走样；长序列上在线方法的老毛病 |
| raymap | 用每条射线的起点和方向做成的六通道图，拿它当虚拟相机去查询状态 |
| 度量尺度 | 点的单位是米，不是「差一个未知比例」的相对重建 |
| VGGT | 第 20 课：一组图一次前向，出全部相机和点图；记忆就是这组输入本身 |
| 4D 生成 / 4DGS | 给一段已知视频拟合可渲染的时空场，用来播画面，不是一份可在线改写的状态 |

## 2. 问题

三个具体问题，对应三种会把桌宠带沟里的混淆。

第一，每帧都像真的，不等于这是同一个空间在随时间变化。视频生成器可以连续吐出
好看的桌子，转一圈回来杯子换成了另一个，纹理还很真。第 12 课用 open-oasis 量过
长时程一致性崩坏：滑窗自回归只看最近 $K$ 帧，窗外的历史等于没发生过。本课要的是一份跨时间对齐的 3D 状态：离开再回来，还是那只杯子的那几个三维点。
更好看的下一帧解决不了这件事。

第二，VGGT 和 CUT3R 都输出点图和相机，职责完全不同。VGGT 吃的是一个集合：你把
现在手里这 $N$ 张图一次性喂进去，它一次前向估出所有位姿和几何。新来一张图，
正规用法是把 $N+1$ 张重新推一遍。CUT3R 吃的是一条流：内部带着 $s_{t-1}$，只把
$I_t$ 编进去，更新成 $s_t$，读出当前帧的世界系点图。对照实验要量的是「新信息
怎么进状态」，不是谁的点云更好看。

第三，4D 这个词被两件无关的事共用。一件是 4D Gaussian Splatting 一类方法：给定
一段已经拍好的动态视频，按场景优化出能按时间查询的高斯，用来实时渲染。另一件
是本课的可更新状态：观察还在源源进来，状态在改，杯子被挪了就要改，被挡住了就
不该删。前者是对已知录像做表示，后者是对正在发生的观察做滤波。本课只做后一件。
仓库 README 和论文标题写的是 Continuous 3D Perception，评估里才出现 3D/4D 任务；
4D 在这里的意思是「带时间的 3D 状态」，不是生成一段 4D 视频。

本课档位是实战/体验。CUT3R 从零训练的官方说明写在 `docs/train.md`，配置按
8×A100 80GB 测过，课程不训。你跑的是 Google Drive 上两份公开 checkpoint，做推理、
做恒常测试、做漂移曲线。论文表格里的 Abs Rel、ATE 你读，不作为本课复现目标。

## 3. 准备

- 第 20 课的概念已经够用：点图、多视图、相机位姿、以及「重建还不是世界模型」。
  第 20 课的点图文件如果还在，本课对照实验可以直接拿来用；没有也不挡主线，CUT3R
  仓库自带 `examples/001` 到 `examples/004` 和 `examples/lady-running.mp4`。
- 一台带 NVIDIA 显卡的 Linux 机器。官方安装写的是 Python 3.11、PyTorch 搭配
  CUDA 12.1，并要编译 `src/croco/models/curope/` 里的 RoPE CUDA 核。24GB 跑
  `cut3r_512_dpt_4_64.pth` 做几十到一两百帧的桌面序列通常够；更长的视频按 README
  提示会线性涨显存，因为编码器对输入做了并行加速。显存紧就改用
  `cut3r_224_linear_4.pth`。
- 磁盘：两份权重加依赖大约几 GB；自己拍的桌面视频再留几 GB。权重在 Google Drive，
  仓库用 `gdown --fuzzy` 下载。
- 一只杯子、一只手、一部能录 1080p 的手机或网络摄像头。三种操作都要真做：手挡住
  杯子约两秒、把杯子拿走、转头离开再转回来。背景尽量简单，桌子不要反光成一片。
- 会用 Python 写几十行分析脚本：读 npz、在一个三维盒子里数点、用 matplotlib 画
  曲线。不需要训练，不需要 SLAM 背景。
- 先花 20 分钟把论文项目页 [cut3r.github.io](https://cut3r.github.io/) 的 Method
  Overview 和「Online vs Revisiting」看完。椅子错觉那个例子会直接变成第 5 节的
  教具：先验把第一帧看成立体椅子，后续观察把状态改写成平面，最后一帧长得像第一
  帧，模型也不退回去。

## 4. 学习目标

1. 白纸画出 CUT3R 一帧的数据流：图像编码、状态更新、状态读出、self 点图、world
   点图、位姿头，并标出世界系原点取在哪一帧；
2. 口头对比 VGGT「一组图重新推」和 CUT3R「连续更新」：输入是集合还是流、记忆
   存在哪、新来一张图时算力怎么涨；
3. 写出并守住书面区分：CUT3R 是 $s_{t}=f(s_{t-1}, I_{t})$，不是
   $P(s_{t+1} \mid s_t, a_t)$，因此不能拿它规划「我推一下杯子会怎样」；
4. 用仓库 `demo.py` 在官方示例和自己拍的桌面序列上跑通在线重建，viser 里能指出
   杯子对应的那一团点；
5. 完成挡住、移走、转头三种操作的物体恒常灯：挡住和转头灯应保持「还在」，移走
   灯应收掉；失败也要记，不许改口令来迁就曲线；
6. 交出一条带时间戳的漂移曲线：绕杯子走一圈再走回来，同一只杯子的世界系质心相对
   首次观测偏了多少米。

## 5. 原理

六个机制。每个按老节奏走：为什么需要、怎么运转、精确定义、代码落点、怎么验收。

### 5.1 一帧不是世界，一组图重推也还不是时间

第 20 课已经把这句话钉死：单张 RGB 没有唯一的 3D。VGGT 的解法是把多张图
当成一个集合，用全局注意力一次读完，吐出所有相机和点图。这对「拍完一圈桌子、
现在给我重建」非常合适。桌宠面对的却是一条没完没了的流：人还在动，头还在转，
杯子可能被拿起来。你不能每来一帧就把从开机到现在的所有图重新塞进 VGGT。就算显存
扛得住，语义也不对：那是一次次重新解释「当前这一批观察」，不是在改写同一份世界。

类比：VGGT 像每次把桌上所有照片摊开重排；CUT3R 像带着一张已经画过的地图，新看到
的东西只改地图上该改的部分。类比失效处：人类地图上的「杯子」是一个物体条目；
CUT3R 的状态是 768 个没有物体身份的 token。挡住杯子时，这些 token 可能还留着杯子
附近的几何，也可能被手的观察冲掉。第 7 节的恒常灯就是在量这件事，不要提前假设
它一定绿。

CUT3R 论文把传统 SfM、SLAM、NeRF、3DGS 称为 tabula rasa：每个新场景从空白开始，
只用当前这批观察。学习型先验让它能从极少的图、甚至单张图猜出度量尺度的几何；
循环更新让这份先验能被后续观察改写。两件事叠在一起，才是「边走边改」。

### 5.2 状态更新与状态读出同时发生

机制很短。状态 $s$ 是一组可学习 token，没看过任何图之前，所有场景共用同一组
初始 token。来了一张图 $I_t$，先用共享权重的 ViT 编成图像 token：

$$
\mathbf{F}_t = \mathrm{Encoder}_i(\mathbf{I}_t)
$$

再在图像 token 前面拼一个可学习的位姿 token $\mathbf{z}$，让两套 decoder 对
$s_{t-1}$ 做双向交互：

$$
[\mathbf{z}'_t, \mathbf{F}'_t], \mathbf{s}_t = \mathrm{Decoders}([\mathbf{z}, \mathbf{F}_t], \mathbf{s}_{t-1})
$$

一边把当前图写进状态（state-update），一边从状态里把过去的上下文读到当前图上
（state-readout）。论文写明这两件事在同一组互相 cross-attend 的 decoder 里同时
完成。读出之后用三个头解析显式 3D：

$$
\hat{\mathbf{X}}_t^{\mathrm{self}}, \mathbf{C}_t^{\mathrm{self}} = \mathrm{Head}_{\mathrm{self}}(\mathbf{F}'_t)
$$

$$
\hat{\mathbf{X}}_t^{\mathrm{world}}, \mathbf{C}_t^{\mathrm{world}} = \mathrm{Head}_{\mathrm{world}}(\mathbf{F}'_t, \mathbf{z}'_t)
$$

$$
\hat{\mathbf{P}}_t = \mathrm{Head}_{\mathrm{pose}}(\mathbf{z}'_t)
$$

self 点图在当前相机系，world 点图在第一帧相机系，位姿是当前帧到世界的刚体变换
（四元数加平移，6 自由度）。三份输出看起来重复，论文的理由是：每条都能直接吃
监督，于是只有位姿标注或只有单视图深度的数据也能拿来训。点图和位姿都是度量尺度，
单位是米。

实现数字以论文第三节为准：图像编码器是 ViT-Large，用 DUSt3R 编码器权重初始化；
decoder 是 ViT-Base；patch 16×16；状态 768 个 token，每个 768 维；raymap 编码器
只有 2 个 block。512 档的几何头是 DPT，224 档中间 checkpoint 的几何头是线性层。
代码落在 `src/dust3r/model.py` 的 `ARCroco3DStereo`（`demo.py` 的文档字符串写明
推理用这个类），头在 `src/dust3r/heads/`，损失在 `src/dust3r/losses.py`。

验证这一小节只看一件事：同一段序列，中途把状态丢掉、从当前帧当第一帧重新开始，
世界系点图的坐标原点必须跳一次。原点不跳，说明你根本没在用那份持续状态。

### 5.3 用虚拟相机问没看见的地方

人走进餐厅扫一眼，能大概猜出柜台后面还有空间。CUT3R 把这件事做成：不更新状态，
只用一份 raymap 去读状态。raymap $\mathbf{R}$ 是六通道图，每个像素存射线原点和
方向，编码虚拟相机的内外参。另有一个轻量 $\mathrm{Encoder}_r$，把 $\mathbf{R}$
编成 token，再走同一套 decoder，但这一次状态保持 $s_t$ 不动，只读出该虚拟视角
的点图和颜色。

论文把它类比成 MAE：MAE 在图像内部补被遮的 patch，这里在场景尺度上补没拍到的
视角。类比失效处：这是回归，不是生成。论文 limitations 写得很直：外推太远会糊，
因为没有随机采样那些「多种可能的未见表」。糊的补全仍然可以告诉你「这个方向上
大概有桌面」，不能拿去当照片用。

训练时，只有带度量尺度 3D 标注的数据才打开 raymap 模式，并且会随机把除第一帧
以外的图像替换成对应 raymap。尺度未知的数据关掉查询，免得状态里的米和标注里的
相对单位打架。

物体恒常和「查询未见视角」是同一块肌肉。杯子被手挡住，当前帧的 self 点图里杯子
那一块变成了手；若状态没把杯子忘干净，用一台仍朝向杯子原位的虚拟相机去查，读出
的 world 点图里杯子那一团还应该在。第 7 节用更朴素的办法近似这件事：在世界系里
给杯子画一个三维盒子，数盒子里高置信点还剩多少。查询头的完整接口本课不作为必做
实验，因为官方 `demo.py` 走的是图像流重建，raymap 查询要自己从模型类里接。改造
清单第 2 条把它列为选做。

### 5.4 训练目标：点图回归、位姿、颜色

几何损失跟 MASt3R 同一家族，带置信度的点图回归：

$$
L_{\mathrm{conf}} = \sum_{(x, c)} (c \cdot d(\hat{x}/\hat{s}, x/s) - \alpha \log c)
$$

有度量真值时令 $\hat{s} := s$，逼模型直接输出米。位姿损失是四元数和平移的 L2，
平移同样做尺度归一。raymap 查询额外加颜色 MSE。置信度 $c$ 既是权重也是「这段
几何我有多敢报」的读数，后面恒常灯会用它滤点，低置信的点不许投票「杯子还在」。

训练是四段课程（论文 3.4 节，仓库 `docs/train.md` 与 `config/stage1.yaml` 到
`config/dpt_512_vary_4_64.yaml` 对应）：先在偏静态数据上用 224 分辨率加线性头训
短序列，再混入动态数据和残缺标注，再把长边升到 512 并换成 DPT 头，最后冻结编码
器、只训 decoder 和头，序列长度拉到 4 至 64 帧。本课不跑其中任何一段。你只需要
知道：512 DPT 那份 `cut3r_512_dpt_4_64.pth` 是最终档，`cut3r_224_linear_4.pth`
是中间档，README 写明了这一点。

### 5.5 这是感知状态更新，还不是世界模型

第 1 课就把世界模型写成 $P(s_{t+1} \mid s_t, a_t)$。CUT3R 的递归是：

$$
s_t = f(s_{t-1}, I_t)
$$

条件里没有动作 $a_t$。输出是对当前这一刻几何的估计，不是对下一步的预测。
它更接近带学习先验的在线感知，或带记忆的滤波器：新观察来了，修正对世界的信念。
人伸手去推杯子之前，CUT3R 不能回答「推完杯子会在哪」，因为它从没学过动作的后果。
它能回答的是：推完之后你把新的画面喂进来，状态会不会改。

第 03 课的动作对换在这里无的放矢：没有动作端口可换。第 17 课三分评测里，CUT3R
最多碰到「当前几何估得准不准」，碰不到「规划好」。第 24 课 Visual Foresight 才
把「看见未来再动手」接上。本课验收里那份书面区分，就是防止你把一份很好的 3D
感知状态误当成已经完成的世界模型。

桌宠上的分工也因此清楚。CUT3R 这类模块负责：头转动时身后的杯子还在状态里；人
把杯子挪了，状态要改。负责「我要不要伸手」的，仍是后面的预测器和规划器。

### 5.6 不要把 4D 生成视频和可更新状态混成一件事

4D Gaussian Splatting（Wu 等，CVPR 2024，arXiv:2310.08528）的问题是：已经有一段
动态视频，怎样表示成一组可随时间变形的高斯，好实时渲染任意时刻、任意视角。它
按场景优化，训练时整段视频都在，推理时查询的是时间戳，不是「新来的一张尚未见过
的图」。动态 3DGS、Deformable-3DGS 同一类。它们产出的是一段可播放的 4D 资产。

CUT3R 的问题是：图还在一张张到达，状态必须在线改，而且输入里没有相机参数。产出
的是每一步的点图和位姿，累加起来是一份正在被改写的场景。论文实验里所谓 4D 任务，
指动态场景上的深度、位姿、重建，方法仍是前馈、在线、不按视频做 test-time
optimization。

两者都在「时间」上说话，验证者完全不同。4DGS 的验证者是新视角、新时刻的渲染误差；
CUT3R 的验证者是在线点图是否与后续观察一致，以及（本课要量的）被挡住的物体还在
不在。把 4DGS 的演示视频当成「模型记住了杯子」，是把渲染一致性误读成状态持久。
本课实验协议里禁止用渲染好看当恒常灯的绿灯。

## 6. 源码导读

克隆后按这个顺序读，每个文件带着问题进去。路径以仓库 main 分支为准（写课时用
GitHub 目录核实：`src/dust3r/`、`src/croco/`、`config/`、`eval/`、`demo.py`）。

| 文件 | 是哪个零件 | 带着什么问题读 |
|---|---|---|
| `demo.py` | 官方推理入口 | 模型类是不是 `ARCroco3DStereo`？`seq_path` 既可接文件夹也可接视频吗？viser 默认端口是不是 8080？点图和位姿有没有写入 `output_dir`，写入的话文件名是什么？ |
| `demo_ga.py` | 可选的全局对齐 | 它在前馈之后还优化什么？这还算不算论文强调的 online？长序列漂移时才碰它 |
| `viser_utils.py` | 可视化 | `PointCloudViewer` 用置信度阈值丢掉哪些点？对应命令行的 `--vis_threshold` |
| `src/dust3r/model.py` | 状态与 decoder | 状态 token 怎么初始化、每帧怎么更新？raymap 查询那条分支会不会改状态？ |
| `src/dust3r/inference.py` | 推理循环 | 新帧是逐张喂给状态，还是编码器先并行再进入循环？README 警告过显存随帧数线性涨，根在这里 |
| `src/dust3r/heads/` | 三个头 | self 头、world 头、pose 头各吃什么 token？world 头是否如论文所述用位姿 token 做 modulation？ |
| `src/dust3r/losses.py` | 训练损失 | 置信度加权和 alpha log c 写在哪？度量尺度时 hat s 等于 s 的开关在哪？ |
| `src/dust3r/blocks.py` | 双向 decoder | 图像 token 和状态 token 的 cross-attention 每一层是否互相看？ |
| `src/croco/models/curope/` | RoPE CUDA 核 | 不编译这里，官方 README 的安装就不完整；Mac 没有这条路径 |
| `add_ckpt_path.py` | 权重路径补丁 | 下载完 checkpoint 若加载报路径错，先看这个脚本是不是还要跑一次 |
| `config/stage1.yaml` 至 `dpt_512_vary_4_64.yaml` | 训练课表 | 只读：四段课程各自冻了什么、序列多长；本课不 launch |
| `eval/monodepth/run.sh` 等 | 论文口径评测 | 只读：要 Sintel、KITTI、7-scenes 等数据，本课主线不跑；知道数字从哪来即可 |

`examples/` 里有四个图像序列目录和一个 `lady-running.mp4`。先用 `examples/001`
把安装跑通，再用动态示例看移动物体，最后才上自己的桌子。不要一上来就喂五分钟
长视频：README 写明编码器并行加速会导致显存随帧数近似线性增加。

论文里的 Spann3R、MonST3R、DUSt3R 对照，代码不在本仓库里。CUT3R 的
`Acknowledgements` 写明自己基于 DUSt3R、MonST3R、Spann3R 和 Viser。要读对照方法，
用第 12 节的论文链接，不要在本仓库里找它们的训练脚本。

## 7. 实验

档位是推理。目标不是复现论文表格，是在自己桌子上看到「状态会改、也会记住」。
每一步先写预期，再跑，再对照。Step 1 到 Step 4 用官方示例，Step 5 到 Step 8
用你自己拍的序列。

### Step 1: 克隆与环境

```bash
git clone https://github.com/CUT3R/CUT3R.git
```

之后所有命令都在仓库根目录的同一个终端里执行。官方 README 的安装顺序如下。

```bash
conda create -n cut3r python=3.11 cmake=3.14.0
```

```bash
conda activate cut3r
```

先按你机器的 CUDA 版本安装 PyTorch。README 示例写的是 12.1，版本不对就去
PyTorch 官网改命令，不要照抄过期的 CUDA 号：

```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

```bash
pip install -r requirements.txt
```

README 注明 dataloader 可能踩 PyTorch issue 99625，需要：

```bash
conda install "llvm-openmp<16"
```

`gsplat` 是训练日志依赖，本课不训练，跳过。`evo` 和 `open3d` 是论文评测依赖，
主线不做 Sintel/KITTI，也可以先不装；自己画漂移曲线用 matplotlib 即可。

编译 RoPE CUDA 核（CroCo v2 同款）。必须在该目录里执行 `setup.py`，相对路径
写死了：

```bash
cd src/croco/models/curope
```

```bash
python setup.py build_ext --inplace
```

```bash
cd ../../../..
```

预期：该目录下出现编译好的扩展，没有报错。编译失败不要硬跑 demo，先查第 10 节。

### Step 2: 下载公开权重

两份 checkpoint 都在 Google Drive，README 用 `gdown --fuzzy`。主线用最终档 512
DPT；显存不够再换 224 线性档。先装 gdown：

```bash
pip install gdown
```

```bash
gdown --fuzzy https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link -O src/cut3r_512_dpt_4_64.pth
```

224 档可选：

```bash
gdown --fuzzy https://drive.google.com/file/d/11dAgFkWHpaOHsR6iuitlB_v4NFFBrWjy/view?usp=drive_link -O src/cut3r_224_linear_4.pth
```

预期：`src/cut3r_512_dpt_4_64.pth` 存在且体积是 GB 量级。文件只有几 MB 说明 Drive
返回了下载页面而不是权重，换浏览器登录后手动下载，再放到 `src/`。

### Step 3: 官方示例跑通

README 的推理示例（一条命令，输入可以是文件夹或视频）：

```bash
python demo.py --model_path src/cut3r_512_dpt_4_64.pth --seq_path examples/001 --size 512 --vis_threshold 1.5 --output_dir tmp
```

预期：终端开始逐帧推理；浏览器打开 viser（README 写默认端口 8080）能看到点云和
相机。转一转视角，场景应是连贯的一份几何，而不是每帧各画各的。`tmp/` 里会落下
输出，具体文件名以 `demo.py` 为准，下一步要打开这个文件确认。

动态内容用仓库自带视频（更短、更省事的冒烟）：

```bash
python demo.py --model_path src/cut3r_512_dpt_4_64.pth --seq_path examples/lady-running.mp4 --size 512 --vis_threshold 1.5 --output_dir tmp_dyn
```

预期：移动的人在点云里是会动的一团，不是被当成固定雕塑。若人被「焊」在第一帧的
位置上，状态更新没发生，停下来读 `src/dust3r/inference.py`。

显存爆了就改 `--size 224` 并换 224 权重；序列太长就先把视频切到前 60 到 90 帧。
README 的内存警告是认真的。

### Step 4: 把点图和位姿落成可分析的 npz

恒常灯和漂移曲线不能靠「我觉得 viser 里杯子还在」。打开 `demo.py`，顺着推理循环
找到当前帧的 world 点图、置信度、相机到世界的位姿。若官方脚本已经把它们写入
`output_dir`，记下文件名和数组形状，跳到 Step 5。若只启动了 viser、没有按帧落盘，
在循环结束处按你读到的变量名保存。下面是目标格式，左边的变量名必须换成脚本里的
真名：

```python
import os
import numpy as np

os.makedirs(out_dir, exist_ok=True)
np.savez_compressed(
    os.path.join(out_dir, "stream.npz"),
    pts_world=np.stack(pts_world_list, axis=0),
    conf=np.stack(conf_list, axis=0),
    poses=np.stack(pose_list, axis=0),
    rgb=np.stack(rgb_list, axis=0),
)
```

约定：

- `pts_world` 形状 `(T, H, W, 3)`，世界系，单位米；
- `conf` 形状 `(T, H, W)`，与点图对齐；
- `poses` 形状 `(T, 4, 4)`，当前相机到世界。若你拿到的是四元数加平移，先在保存前
  自己拼成 4×4，后面脚本一律吃矩阵；
- `rgb` 形状 `(T, H, W, 3)`，与点图同一分辨率。像素框必须在这张图上读，不要在
  原始 1080p 视频上读，CUT3R 会把输入缩放到 `--size`。

用 `examples/001` 的输出做一次形状检查：

```bash
python -c "import numpy as np; z=np.load('tmp/stream.npz'); print({k: z[k].shape for k in z.files})"
```

预期：四把钥匙都在，`T` 等于输入帧数，`rgb` 的 `H,W` 与 `pts_world` 相同。`T`
对不上就说明循环里只存了最后一帧，回去改保存位置。把 `rgb[0]` 存成 png 再读杯子
像素框：

```python
import matplotlib.pyplot as plt
import numpy as np
z = np.load("tmp/stream.npz")
plt.imsave("tmp/frame0.png", z["rgb"][0])
```

### Step 5: 拍三条桌面操作

固定摄像头高度，桌子上放一只杯子，背景尽量干净。每条 10 到 25 秒，1080p、30fps
即可，CUT3R 会自己缩放。不要开超广角畸变太大的镜头。

1. `occlude.mp4`：杯子在画面中央，手伸过去挡住杯子约两秒，手拿开，杯子没动。
2. `remove.mp4`：同样的起始构图，手把杯子端走，离开桌面，空桌子再拍两三秒。
3. `turn.mp4`：先对着杯子拍两秒，整台相机转到旁边的墙或空桌面（杯子完全离开画面）
   约两秒，再转回杯子。

第四条给漂移曲线：`loop.mp4`，手持相机绕杯子走一圈再走回起点，杯子始终大致在
视野里或只短暂出画。走慢一点，避免运动模糊。

把四条视频放进 `runs/desk/`。每条附一个 `events.csv`，三列：`name,start_s,end_s`。
`occlude` 那条至少有一行 `occlude,3.0,5.0`（改成你真实的秒数）；`remove` 有
`remove`；`turn` 有 `away` 和 `back`。时间用手机录像的播放器读，精确到 0.5 秒
够用。

对每条视频跑一次 demo（把路径换成你的）：

```bash
python demo.py --model_path src/cut3r_512_dpt_4_64.pth --seq_path runs/desk/occlude.mp4 --size 512 --vis_threshold 1.5 --output_dir runs/desk/occlude
```

`remove`、`turn`、`loop` 同样各跑一次，换 `--seq_path` 和 `--output_dir`。跑完按
Step 4 的格式在每个目录留下 `stream.npz`。

### Step 6: 物体恒常灯

灯的规则写死，不允许事后改：

- 挡住：操作期间和刚结束后，杯子盒子里高置信点数应仍明显高于空桌子基线，灯绿。
- 移走：杯子离开之后，盒子里的点数应掉到基线附近，灯红（「还在」是错的）。
- 转头：杯子不在画面里的那几秒，盒子里应仍有点，灯绿；转回来后质心不应跳到桌子
  另一头。

「盒子」怎么来：打开该序列 `rgb[0]` 存出的 png，用看图工具读杯子像素框
`--pixel-box x0,y0,x1,y1`（左上为原点，坐标相对模型分辨率，不是原始视频）。脚本
用该框内、挡住之前若干帧的高置信点中位数当杯子质心，再取边长约 20 厘米的立方体
（`--half 0.10` 表示半边长 10 厘米；杯子差很多就改，改完写进笔记）。同一只像素框
和同一只三维盒子用于三条操作，不许三条各调一套来让灯变绿。

把下面存成 `runs/desk/constancy_lamp.py`，在仓库根目录、已 `conda activate cut3r`
的环境里跑。它只依赖 numpy 和 matplotlib。

```python
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_stream(path):
    z = np.load(path)
    pts = z["pts_world"]
    conf = z["conf"]
    poses = z["poses"]
    t, h, w, _ = pts.shape
    assert conf.shape == (t, h, w)
    assert poses.shape[0] == t
    return pts, conf, poses


def fps_of(t, duration_s):
    return t / max(duration_s, 1e-6)


def box_count(pts, conf, center, half, conf_thr):
    lo = center - half
    hi = center + half
    inside = np.all((pts >= lo) & (pts <= hi), axis=-1) & (conf >= conf_thr)
    xyz = pts[inside]
    n = int(inside.sum())
    centroid = xyz.mean(axis=0) if n else np.full(3, np.nan)
    return n, centroid


def read_events(csv_path):
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append((row["name"], float(row["start_s"]), float(row["end_s"])))
    return rows


def estimate_center(pts, conf, conf_thr, end_frame, pixel_box):
    sl = slice(0, max(end_frame, 1))
    mask = conf[sl] >= conf_thr
    if pixel_box is not None:
        x0, y0, x1, y1 = pixel_box
        _, h, w, _ = pts.shape
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        roi = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
        mask = mask & roi[None, :, :]
    xyz = pts[sl][mask]
    if xyz.shape[0] < 50:
        raise SystemExit("置信点太少，检查 --pixel-box 和 --conf-thr")
    return np.median(xyz, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--events", required=True)
    p.add_argument("--duration", type=float, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--conf-thr", type=float, default=1.5)
    p.add_argument("--half", type=float, default=0.10)
    p.add_argument("--pre-s", type=float, default=2.0)
    p.add_argument("--pixel-box", required=True, help="x0,y0,x1,y1 on frame 0")
    args = p.parse_args()

    pts, conf, _ = load_stream(args.npz)
    t = pts.shape[0]
    fps = fps_of(t, args.duration)
    pre_n = max(int(args.pre_s * fps), 5)
    box = tuple(int(x) for x in args.pixel_box.split(","))
    center = estimate_center(pts, conf, args.conf_thr, pre_n, box)
    half = np.array([args.half, args.half, args.half])

    counts = []
    cents = []
    times = np.arange(t) / fps
    for i in range(t):
        n, c = box_count(pts[i], conf[i], center, half, args.conf_thr)
        counts.append(n)
        cents.append(c)
    counts = np.array(counts)
    cents = np.array(cents)

    baseline = np.median(counts[:pre_n])
    events = read_events(args.events)

    lines = [
        f"center_m={center.tolist()}",
        f"fps={fps:.3f} T={t} baseline_count={baseline:.1f}",
    ]
    for name, t0, t1 in events:
        sl = (times >= t0) & (times <= t1)
        after = (times > t1) & (times <= t1 + 1.0)
        during = counts[sl].mean() if sl.any() else np.nan
        post = counts[after].mean() if after.any() else during
        if name in ("occlude", "away"):
            lamp = "green" if post >= 0.4 * baseline else "red"
            expect = "green"
        elif name in ("remove",):
            lamp = "red" if post <= 0.4 * baseline else "green"
            expect = "red"
        elif name in ("back",):
            lamp = "green" if post >= 0.4 * baseline else "red"
            expect = "green"
        else:
            lamp, expect = "n/a", "n/a"
        lines.append(
            f"{name} {t0:.1f}-{t1:.1f}s during={during:.1f} post={post:.1f} "
            f"lamp={lamp} expect={expect}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".txt").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(times, counts, color="black", lw=1.5)
    ax.axhline(0.4 * baseline, color="gray", ls=":", lw=1)
    for name, t0, t1 in events:
        ax.axvspan(t0, t1, color="0.85", lw=0)
        ax.text(t0, max(counts) * 0.95 if len(counts) else 1, name, fontsize=8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("high-conf points in cup box")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=140)
    print("\n".join(lines))
    print("wrote", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
```

挡住这条的调用（时长改成你的视频秒数）：

```bash
python runs/desk/constancy_lamp.py --npz runs/desk/occlude/stream.npz --events runs/desk/occlude_events.csv --duration 12 --pixel-box 400,220,720,720 --out runs/desk/occlude/lamp
```

`remove` 和 `turn` 各跑一遍。预期：挡住和转头的 `lamp=green` 且 `expect=green`；
移走的 `lamp=red` 且 `expect=red`。对不上不要改 `0.4 * baseline` 这个阈值来凑绿灯，
先检查盒子是不是太大（手也算进杯子）或太小（稍微一漂就空）。盒子边长和置信阈值
写进笔记，改一次就记一次。

转头那条如果灯红，常见原因是状态其实把出画的几何冲掉了，或世界系随错误位姿漂走
导致盒子套空。这两种失败正好是论文 limitations 里「长序列无全局对齐会漂」的桌面
版，记下来，第 9 节允许失败，不允许没测。

### Step 7: 漂移曲线

在 `loop.mp4` 的 `stream.npz` 上，用 Step 6 同一只杯子盒子，画出质心相对第 1 秒
内中位数位置的欧氏距离，横轴是时间。把下面存成 `runs/desk/drift_curve.py`。

```python
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--duration", type=float, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--conf-thr", type=float, default=1.5)
    p.add_argument("--half", type=float, default=0.10)
    p.add_argument("--pre-s", type=float, default=1.0)
    p.add_argument("--pixel-box", required=True, help="x0,y0,x1,y1 on frame 0")
    args = p.parse_args()

    z = np.load(args.npz)
    pts, conf = z["pts_world"], z["conf"]
    t = pts.shape[0]
    fps = t / args.duration
    pre_n = max(int(args.pre_s * fps), 5)
    x0, y0, x1, y1 = (int(x) for x in args.pixel_box.split(","))
    _, h, w, _ = pts.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    roi = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
    mask0 = (conf[:pre_n] >= args.conf_thr) & roi[None, :, :]
    center0 = np.median(pts[:pre_n][mask0], axis=0)
    half = np.array([args.half, args.half, args.half])

    times = np.arange(t) / fps
    dist = np.full(t, np.nan)
    for i in range(t):
        inside = np.all((pts[i] >= center0 - half) & (pts[i] <= center0 + half), axis=-1)
        inside &= conf[i] >= args.conf_thr
        if inside.sum() < 30:
            continue
        c = pts[i][inside].mean(axis=0)
        dist[i] = np.linalg.norm(c - center0)

    out = Path(args.out)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(times, dist, color="black", lw=1.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("cup centroid drift (m)")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=140)
    np.savetxt(
        out.with_suffix(".csv"),
        np.stack([times, dist], axis=1),
        delimiter=",",
        header="t_s,drift_m",
        comments="",
    )
    valid = dist[~np.isnan(dist)]
    print(f"T={t} fps={fps:.2f} median={np.median(valid):.4f} max={np.max(valid):.4f}")


if __name__ == "__main__":
    main()
```

```bash
python runs/desk/drift_curve.py --npz runs/desk/loop/stream.npz --duration 20 --pixel-box 400,220,720,720 --out runs/desk/loop/drift
```

预期：曲线有起伏，但绕一圈走回起点时，末端漂移应明显小于杯子本身的尺度（十几厘米
量级）。若走回起点仍有几十厘米甚至米级偏差，这是在线方法的漂移，把它写进报告，
不要改坐标去「对齐」成好看的零。可选对照：同一条 `loop.mp4` 再跑一次
`demo_ga.py`（命令把 `demo.py` 换成 `demo_ga.py`，其余参数相同），看全局对齐是否
把末端漂移压下去。`demo_ga.py` 已经不是纯在线前馈，报告里必须标明。

### Step 8: VGGT 对照（有第 20 课环境就做，没有就写书面对照）

同一组从 `loop.mp4` 抽的 8 到 16 张关键帧，分别交给两种模型。VGGT 官方 README 的
viser 入口是：

```bash
python demo_viser.py --image_folder path/to/your/images/folder
```

该命令在 [facebookresearch/vggt](https://github.com/facebookresearch/vggt) 仓库根
目录执行，不要在 CUT3R 目录里找这个脚本。对照要写的不是「谁更漂亮」，是下面这张
表：

| 问题 | VGGT | CUT3R |
|---|---|---|
| 输入 | 当前这一组图 | 当前这一帧加内部状态 |
| 新来 1 帧 | 把组扩成 N+1，整组再前向 | 只处理新帧，更新 $s$ |
| 挡住杯子时若组里没有未挡帧 | 输入里没有杯子，输出里杯子消失 | 状态里若还在，world 点图里杯子可还在 |
| 有没有动作端口 | 无 | 无 |
| 算不算 $P(s_{t+1}\mid s_t,a_t)$ | 不算 | 不算 |

若你第 20 课已经用同一张桌子跑过 VGGT，把当时的点图误差表附在本课笔记后面，标
清两次实验的帧集合是否相同。帧集合不同，不许比绝对值。

### Step 9: 书面区分和笔记

在 `runs/desk/NOTES.md` 里用自己的话写满这三段，每段不少于五句：

1. CUT3R 这一步在数学上吃什么、吐什么，状态存在哪；
2. 为什么挡住应记住、移走应更新、转头应记住，你的灯实际怎么亮，失败的那条你认为
   坏在盒子、位姿还是状态本身；
3. 为什么即使三条灯全对，仍然不能把 CUT3R 接到控制器上去推杯子。第三段必须出现
   公式 $s_{t}=f(s_{t-1}, I_{t})$ 和 $P(s_{t+1} \mid s_{t}, a_{t})$，并写明差在动作和
   「下一步」。

笔记开头再抄五行：日期与机器、仓库 commit、权重文件名、每条视频时长和事件表、
漂移曲线的中位数和最大值（米，带样本帧数）。

## 8. 配置与预算

| 档位 | 权重 | 输入规模 | 显存与时间（参考） | 用途 |
|---|---|---|---|---|
| 冒烟档 | `cut3r_224_linear_4.pth`，`--size 224` | `examples/001` 或几十帧 | 一张 8 到 12GB 卡也能试；数分钟 | 验证安装和 viser |
| 本课主档 | `cut3r_512_dpt_4_64.pth`，`--size 512` | 四条 10 到 25 秒桌面视频，建议抽到 5 到 15 fps 再跑 | 单张 24GB 通常够；每条数分钟到十几分钟 | 恒常灯与漂移曲线 |
| 长序列 | 同上 | 数百帧以上 | 显存随帧数近似线性涨，README 原话 | 观察漂移，非必做 |
| 论文评测档 | 同上，跑 `eval/*/run.sh` | Sintel、KITTI、7-scenes 等 | 另下数据集，多卡可并行 | 不构成本课验收 |
| 训练档 | `docs/train.md` 的 stage1 到 `dpt_512_vary_4_64` | 32 个数据集 | 官方按 8×A100 80GB 测过 | 本课不训 |

抽帧建议：30fps 的手机视频直接喂会又慢又占显存。先按 10fps 抽一版给 CUT3R，事件表
的秒数仍然对原视频的播放时间，两边用同一条时间轴。抽帧命令示例（一条）：

```bash
ffmpeg -i runs/desk/occlude.mp4 -vf fps=10 runs/desk/occlude_frames/%06d.jpg
```

然后 `--seq_path` 指向 `runs/desk/occlude_frames`。`duration` 仍填原视频秒数。

Mac 用户：RoPE CUDA 核编译不过，本课官方 demo 视为不可跑。你仍然要完成第 5 节
机制、第 6 节源码阅读、Step 9 书面区分，以及用项目页的 Online vs Revisiting 和
椅子错觉写一份「状态会改写先验」的观察笔记。有同学的 `stream.npz` 也可以只跑
Step 6、7 的分析脚本。不要改装 CPU 版冒充已经做了在线重建。

权重许可以仓库 `LICENSE` 为准，动手前读一遍。本课只做本地推理，不重新分发权重。

## 9. 验收

验收清单：

- [ ] 白纸数据流图上能标出 $I_{t}$、$s_{t-1}$、$s_{t}$、self 点图、world 点图、相机位姿，
      并写明世界系原点是第一帧；
- [ ] 官方 `examples/001` 的 viser 截图一张，动态示例或桌面序列截图一张，能指出杯子
      或主要物体对应的点团；
- [ ] 三条操作各有 `lamp.txt` 和 `lamp.png`：挡住、转头期望绿，移走期望红；实际颜色
      与期望一致或对失败有归因为盒子 / 位姿 / 状态三者之一；
- [ ] `loop/drift.png` 与 `loop/drift.csv` 存在，横轴是秒，纵轴是米，笔记里报告中位数
      和最大值；
- [ ] VGGT 对照表填完（跑过或书面），没有把「点云更好看」写成「更像世界模型」；
- [ ] `NOTES.md` 三段书面区分齐全，公式 $s_{t}=f(s_{t-1}, I_{t})$ 与
      $P(s_{t+1} \mid s_{t}, a_{t})$ 都出现，并明确 CUT3R 没有动作端口；
- [ ] 能口头举出 4DGS 与本课状态的差别：一个按已拍视频优化可渲染场，一个在观察流上
      在线改写几何；
- [ ] 没把论文表格的 KITTI Abs Rel 或 ATE 抄进「我复现了」一栏。

口头关：找一个没上过这课的人，只用杯子和手演示三种操作，说明灯为什么那样亮，以及
为什么绿灯仍不够让桌宠决定伸不伸手。听的人若把 CUT3R 理解成「会预测未来的世界
模型」，口头关失败，回去改 `NOTES.md` 第三段。

## 10. 排错

| 症状 | 最可能的原因 | 怎么确认 | 怎么修 |
|---|---|---|---|
| `curope` 编译失败 | 没装对应 CUDA，或在错误目录跑 `setup.py` | 报错含 nvcc / cuda | 按本机 CUDA 重装 PyTorch 后，必须在 `src/croco/models/curope` 里编译 |
| Mac / CPU 上 demo 直接崩 | 官方路径依赖 CUDA 扩展 | import curope 失败 | 本课动手视为不可跑，改走第 8 节 Mac 路线，不要乱改核 |
| gdown 下下来只有几 MB | Drive 反爬，下到了 HTML | `file` 看文件类型 | 浏览器登录后手动下载，放到 `src/cut3r_512_dpt_4_64.pth` |
| 加载权重报路径或 key 对不上 | 文件不完整，或需要 `add_ckpt_path.py` | 对照 README 的文件名；打开 `add_ckpt_path.py` | 重新下载；按该脚本说明补路径 |
| 显存 OOM | 帧太多，编码器并行导致线性涨显存 | 日志在编码阶段炸 | 抽到 10fps、切短、换 224 档、降 `--size` |
| viser 打不开 | 远程机器没转发 8080 | 浏览器超时 | SSH 端口转发；或不看 viser，只落 `stream.npz` 做分析 |
| `stream.npz` 只有一帧 | 保存写在循环外，覆盖了列表 | `T==1` | 把 append 放进每帧循环，`stack` 放循环后 |
| 恒常灯三条全绿 | 盒子太大，整张桌子都在里面 | 打印 `baseline_count`，移走后点数几乎不掉 | 缩小 `--half`，或用下面的像素框指定杯子 |
| 恒常灯三条全红 | 置信阈值太高，或世界系漂得盒子套空 | 降 `--conf-thr` 看点数是否回来；画位姿轨迹 | 先把阈值调到与 `--vis_threshold` 同量级；仍红就记录为漂移失败 |
| 移走后灯仍绿 | 状态死抱第一帧，或杯子投影的点没清 | viser 里空桌子上是否还钉着一杯点 | 这是有意义的失败：更新不足。记进笔记，不要改事件表把移走改成挡住 |
| 人像焊在原地 | 没走循环状态，或把动态场景当静态 GA | 对照 `lady-running.mp4` | 确认跑的是 `demo.py` 不是只跑了某帧；动态序列不要用静态假设的全局对齐硬套 |
| `demo_ga.py` 比 `demo.py` 好看很多 | 全局对齐在吃未来帧做后处理 | 读 `demo_ga.py` 是否在全序列上优化 | 可以当对照，验收主线仍以前馈 `demo.py` 为准 |

杯子中心必须来自 `--pixel-box`，不要用整幅图高置信点的中位数，那会落到桌面中央。
像素框写进 `NOTES.md`，三条操作共用。

## 11. 前沿与改造

同一问题，2024 到 2026 年公开系统大致分成四条路。一条是成对点图再全局对齐：
DUSt3R（Wang 等，arXiv:2312.14132）把两张图映到同一坐标系的点图，多视图靠事后的
global alignment；MASt3R（Leroy 等，arXiv:2406.09756）把点图做成度量尺度并加强
匹配；MonST3R（Zhang 等，arXiv:2410.03825）在动态数据上微调同一对式家族，动态场景
上的视频深度很强，但仍要 GA，官方对比里 KITTI 上带光流的优化比纯在线慢约 50 倍。
第二条是外挂空间记忆：Spann3R（Wang 与 Agapito，arXiv:2408.16061）给 DUSt3R 加一块
缓存观察到的 3D，在线、但不以推断未见结构为设计目标，训练偏静态。第三条是一组图
一次前向：VGGT（Wang 等，CVPR 2025 Best Paper，arXiv:2503.11651）把当前集合里的
全部相机、点图、深度、轨迹一起读出，速度快，记忆就是这组输入。第四条才是本课的
压缩循环状态：CUT3R 把观察写进 768 个 token，并能用 raymap 读未见视角。

论文表格里，在线方法这一组，CUT3R 在 KITTI 视频深度（按序列对齐尺度）Abs Rel
0.118、δ<1.25 为 88.1，A100 上 512×144 约 16.58 FPS；Spann3R 同表 13.55 FPS 且只
支持 224。这些数字是论文的，不是你的机器数字。你的验收是恒常灯和漂移曲线。

缩小版和前沿版的差距，规模占一块：32 个数据集、8×A100、四段课程，钱能堆。机制
上你已经摸到了：循环状态、world 点图、在线更新。还没摸到的机制有三件，每件都对应
下面的改造，而不是「再下一个更大的 checkpoint」。

1. 长序列漂移。论文 limitations 第一句：没有全局对齐，很长的序列会漂。TTT3R
   （Chen 等，arXiv:2509.26645）把 CUT3R 的状态更新改写成测试时训练的逐步规则，
   用来拉长记忆。本课不要求跑 TTT3R，但漂移曲线就是它想治的病。
2. 没有物体身份。768 个 token 不保证「杯子」是一个槽。挡住时灯红，可能不是忘了
   几何，而是杯子和手揉在同一批 token 里。第 23 课 C-SWM 才正式拆槽。
3. 没有动作。再好的持久几何也不能规划。第 22 课问视频基础模型动作起不起作用，
   第 24 课才把预测接上控制。

动手改造清单（选做，每个写清预算和失败判据）：

1. 滑窗 VGGT 对循环 CUT3R。同一条 `occlude.mp4`，VGGT 每次只吃最近 4 帧（模拟
   「集合里没有旧观察」），CUT3R 吃完整流。预算：已有两边环境的话半天。预期：
   挡住期间 4 帧窗口里全是手，VGGT 的杯子点应掉光；CUT3R 若状态还在，盒子里应
   还能数到点。失败判据：CUT3R 同样掉光，则本课主线的「持久」在你的桌子上没发生，
   报告如实写，并检查是不是第一帧世界系已经漂出盒子。
2. 冻结状态再读。复现论文的 revisiting：先正向跑完整段得到最终 $s_T$，再冻结
   $s_T$、重新读每一帧的点图（论文 Table 5 的设定）。预算：要改 `demo.py` 或
   `inference.py` 里状态是否继续写，数小时级改代码加推理。预期：回头看早先模糊
   的区域应变清楚，类似项目页 TV 与茶几重叠被纠正的例子。失败判据：来回两次点图
   无差别，说明你的冻结没生效，或序列太短状态没积累。
3. raymap 查询杯子原位。挡住期间不更新状态，用杯子还在时的相机 raymap 去读当前
   状态，看读出的点图里盒子计数。预算：要读 `model.py` 里查询分支，一天。预期：
   挡住时查询结果仍绿，当前帧 self 点图变红。失败判据：查询与当前帧无差别，说明
   走错了会写状态的那条路。
4. 漂移对帧间隔。`loop.mp4` 用 5fps、10fps、原 fps 各跑一次，画三条漂移曲线。
   预算：三次推理。预期：更密的帧应减轻位姿积分误差。失败判据：更密反而更漂，
   先查有没有把运动模糊帧喂进去。

顺手能看到的论文方向：论文说 revisiting 提高 accuracy，你的改造 2 在缩小桌上
应能看到「同一区域更完整」，不必追 7-scenes 的表格数字。论文说动态场景上在线方法
优于静态记忆的 Spann3R，你的 `lady-running.mp4` 和移走操作就是这个方向的定性版。

## 12. 论文与延伸

1. CUT3R（Wang, Zhang, Holynski, Efros, Kanazawa，CVPR 2025 Oral，
   [arXiv:2501.12387](https://arxiv.org/abs/2501.12387)）。本课宪法。带着四个问题
   读：状态 token 和图像 token 如何同时完成 update 与 readout？world 点图的坐标系
   钉在哪？raymap 查询为什么特意不写状态？limitations 里漂移、糊、循环训练贵，
   哪一条被你的桌面实验碰到了？项目页 [cut3r.github.io](https://cut3r.github.io/)
   的椅子错觉和 Online vs Revisiting 与正文第四节对照着看。
2. VGGT（Wang 等，CVPR 2025 Best Paper，[arXiv:2503.11651](https://arxiv.org/abs/2503.11651)）。
   第 20 课主锚。带着问题读：它的记忆是输入集合还是内部递归？新来一张图，计算如何
   增长？把摘要里的「one, a few, or hundreds of views」和 CUT3R 的「stream」放在
   同一张纸上，写出本课 Step 8 那张表的论文依据。
3. Spann3R（Wang 与 Agapito，[arXiv:2408.16061](https://arxiv.org/abs/2408.16061)，
   3DV 2025）。CUT3R 论文指定的在线对照。带着问题读：spatial memory 缓存的是已观察
   内容还是也能推断未见结构？静态假设写在哪一节？为什么动态人像会把这种记忆打坏？
4. MonST3R（Zhang 等，[arXiv:2410.03825](https://arxiv.org/abs/2410.03825)）。对式
   点图家族在动态场景上的代表。带着问题读：它相对 DUSt3R 改了数据还是改了结构？
   全局对齐假设了什么，动态物体上这个假设如何受伤？CUT3R 论文 Table 2 把谁标成
   Optim.、谁标成 Onl.，你的 `demo.py` 对哪一列？
5. 选读 DUSt3R（Wang 等，[arXiv:2312.14132](https://arxiv.org/abs/2312.14132)）与
   MASt3R（Leroy 等，[arXiv:2406.09756](https://arxiv.org/abs/2406.09756)）：点图
   这个表示从哪来、度量尺度怎么进损失。CUT3R 的置信度损失几乎
   直接搬了后者。再选读 4D Gaussian Splatting（Wu 等，CVPR 2024，
   [arXiv:2310.08528](https://arxiv.org/abs/2310.08528)），只为划清：可渲染的 4D
   资产不是可更新的感知状态。

仓库入口：[CUT3R/CUT3R](https://github.com/CUT3R/CUT3R) 的 README、`docs/eval.md`、
`docs/train.md`。评测脚本和训练配置用来核对论文数字从哪来，不用来当本课作业命令。

第 22 课要把镜头从「几何还在不在」转到「视频基础模型算不算世界模型」：同样会往后
播画面的系统，动作起不起作用、状态能不能保持、能不能拿去选动作。本课留下的那份
书面区分会直接变成第 22 课筛选器的第二问：离开视野的东西还在吗？第三问仍然空着，
因为这里还没有 $a_t$。
