---
id: 23_object_centric_wm
title: "物体中心世界模型"
summary: "状态是一条向量时，桌上两样东西怎么分开推演？"
unit: spatial
play_tools: []
checkpoints:
  - "C-SWM 方向性复现报告（论文复现 #6）。"
  - "槽与物体的对应图。"
---

# 第 23 课：杯子和手不该塞进同一个向量

> 类型：复现（论文复现 #6：C-SWM 官方小环境方向性复现 + 槽可解释性）<br>
> 建议周期：2-3 天<br>
> 硬件：Mac / 纯 CPU 可完成全部必做；单张 24GB 卡更快，本课几乎吃不满显存<br>
> 锚定仓库：[tkipf/c-swm](https://github.com/tkipf/c-swm)（官方 PyTorch，2019 年停更，课内给兼容补丁）；论文 C-SWM（Kipf, van der Pol, Welling, ICLR 2020, [arXiv:1911.12247](https://arxiv.org/abs/1911.12247)）<br>
> 产物：2D Shapes 上训好的 C-SWM、Hits@1 / MRR 记录、槽与物体对应图、交换或删除槽的互动记录、一份「向量状态做不到、槽能做」的对照笔记

## 1. 这一课做什么

整门课的循环没变：观察先压成状态，再按动作预测下一状态，然后展开未来、打分、选动作。
前几课给这条循环补了空间零件。第 20 课用多视图把桌子收成一份可查询的 3D 状态；
第 21 课让这份状态跟着时间改，人把杯子挪了要更新，头转过去再转回来杯子还在；
第 22 课用第 12 课的三问筛过 Cosmos、DINO-WM 一类会往后播的大视频模型。缺的那一块
是：状态里的东西是不是分开的。

[第 16 课](16_three_roads_debate.md) 把三条路摊开时，第三条路叫物体中心槽：状态
不是一条向量，是一组槽，每个槽绑一个物体或部位，转移在槽上走，关系用图或注意力。
当时只在表上把「一杯水被手碰到」写成杯子、手、桌子三个下一状态，没有真训。这课
把那次预告兑成实验。

本课零件落在循环的「压成状态」和「按动作预测下一状态」两格。观察仍是像素。状态
从一条 $z$ 换成 $K$ 个槽 $\{z^1,\ldots,z^K\}$。预测器从「整幅图一起挪」改成每个槽
自己走一步，再靠图网络看邻居。桌宠上这句话非常具体：杯、手机、手、自己的头必须
分开预测。揉成一条向量之后，杯子倒了和手机没动会糊成同一次纹理变化，规划器分不清
该躲开哪一样。

动手锚定 [tkipf/c-swm](https://github.com/tkipf/c-swm)。这是 C-SWM 论文的官方
PyTorch 实现，环境小，CPU 就能训。你要做四件事：按仓库跑通 2D Shapes；把物体抽取
器的特征图画出来，看槽有没有各绑一个物体；做两个消融，打乱物体身份、挡住一个物
体；在 3-body 小球或 Shapes 的两个槽上，交换槽 A / 槽 B 或删掉一个，看模型还知不
知道谁在动。论文复现 #6 的口径是方向性：1 步 Hits@1 明显高、多步掉得比像素重建基
线慢、槽图能对上物体。不要求对齐论文 Table 1 的四种子均值。

对照只讲、不训：Slot Attention（Locatello 等, NeurIPS 2020,
[arXiv:2006.15055](https://arxiv.org/abs/2006.15055)）把「怎么把像素绑到槽上」做成
可迭代模块；STEVE（Singh, Wu, Ahn, [arXiv:2205.14065](https://arxiv.org/abs/2205.14065)）
把槽接到视频上；SlotFormer（Wu 等, ICLR 2023,
[arXiv:2210.05861](https://arxiv.org/abs/2210.05861)）用 Transformer 在槽上做长程
动力学。它们是同一条路上更新的零件，本课的手还是 2019 年的 C-SWM。

诚实分档先说清。官方小环境上的训练、评测、槽可视化和两个消融是**复现**。Slot
Attention / STEVE / SlotFormer 的大规模视频实验是**只讲**，本课不训它们。自己拍的
桌面短视频上试一个最小槽编码器，是可选体验，不计入复现 #6，也赶不上论文数字。

术语速查：

| 术语 | 一句人话 |
|---|---|
| 槽 / slot | 状态里的一个格子，理想情况下一格绑一个物体或部位，整份状态是一组槽而不是一条向量 |
| 物体抽取器 | C-SWM 里那张 CNN：最后一层输出 $K$ 张特征图，每张当作一个槽的软掩码 |
| 物体编码器 | 权重共享的 MLP：把每张掩码压成 $D$ 维槽向量 $z^k$ |
| 图转移 / TransitionGNN | 槽当节点、两两连边的图网络，输出每个槽这一步该怎么动 |
| 对比能量 | 把「当前槽 + 动作推出的下一步」和「真的下一步」的距离当能量，真转移要低、假转移要高 |
| Hits@1 | 在一堆候选下一状态里，模型预测离真实下一状态最近的那一次算命中；本课的主尺 |
| MRR | 平均倒数排名：真实下一状态排第 $r$ 名就记 $1/r$，对排错更敏感 |
| 槽置换 | 交换或打乱槽的顺序；身份真绑住了，置换会把「谁被推」搞错 |
| copy-action | Atari 实验里把同一个动作复制给所有槽，因为游戏手柄不是「对某个物体动手」 |
| ignore-action | 3-body 物理没有智能体，转移模型不吃动作 |

## 2. 问题

一条向量状态在单物体、单主角的世界里够用。CarRacing 里主要是「车和路」，
[第 05 课](05_rssm_planet.md) 的 RSSM 可以把 $h_t,z_t$ 当成全世界。桌子不是这种
世界。同一帧里杯子在倒、手机没动、手伸过来、你的头在转，四件事时间尺度不同、该
不该动也不同。把它们压进同一个 $z$，预测器只能输出一次平均位移。
[第 03 课](03_mdn_rnn_action_conditioned.md) 的动作对换在这里会失效：你对杯子伸手，
手机的未来也被带着漂。

四个具体问题，本课都要当场看见：

1. 状态的形状改成一组槽之后，编码器凭什么不把五个物体写进同一个槽？C-SWM 的答
   案不是重建像素，是对比学习：真的 $(s_t,a_t,s_{t+1})$ 能量要低，随机抽来的假
   $s$ 能量要高。物体身份是这个目标逼出来的，没有框标注。
2. 动作怎么进槽？2D Shapes 的动作是「对第 $k$ 个物体往某个方向推一格」。图网络
   必须让被推的那个槽动、没被推的槽尽量不动，还要处理「目标格被别人占着」这种
   碰撞。去掉图、去掉分槽，论文 Table 1 里长程 Hits 会掉。你的消融要复现这个方向。
3. 槽绑住了，能不能被拆开检查？这是向量 RSSM 给不了的验收。交换两个槽，等于对
   调两个物体的身份再问「下一步谁撞谁」。删掉一个槽，等于假装那个物体从世界上
   消失。这两个操作是本课的互动实验，不是装饰。
4. 2019 年的物体抽取器只是「CNN 最后一层 $K$ 张图」。它分不清两个长得一样的物
   体，也扛不住自然桌面的纹理。Slot Attention 用槽之间的竞争来绑物体，STEVE /
   SlotFormer 再把时间接上。本课要分清：你练的是对比式结构化世界模型，不是 2023
   年的视频槽模型。别把只讲的系统写进复现报告。

先看一次失败，再读公式。想象两个球，红的往右、蓝的往左，下一帧它们会擦肩。向量
模型吐一张糊在一起的「两个色块都往中间走」的预测；槽模型应该有槽 A 写红球位置、
槽 B 写蓝球位置。现在把 A 和 B 对调再送进转移模型。如果身份是真的，对调等于告诉
模型红球在蓝球的位置上、蓝球在红球的位置上，预测会按错的身份走。如果两个槽其实
是同一份糊表示，对调几乎没影响。第 7 节你会在自己训出来的权重上做这件事。

边界一句话：2D Shapes 和 3-body 小球是合成环境，物体颜色不同、背景干净。这只说
明「分槽 + 对比 + 图转移」在组合结构的世界里比像素重建更站得住，不说明你的桌宠
摄像头已经能自动抠出杯子。桌面档是第 29、30 课的工作，本课只把零件造出来。

## 3. 准备

- 手艺依赖，不是产物依赖。[第 03 课](03_mdn_rnn_action_conditioned.md) 的动作对换、
  [第 05 课](05_rssm_planet.md) RSSM 的「一条向量装全世界」、
  [第 13 课](13_ijepa_from_scratch.md) 对比目标和坍缩、
  [第 16 课](16_three_roads_debate.md) 三条路的条件句，这课都会用到。没有训过那些
  模型也能跑通仓库，但对照笔记会少两根柱子。
- 硬件门槛低。CURRICULUM 把 C-SWM 列进 Mac / 纯 CPU 可完成的课。2D Shapes 的数据
  生成和 100 epoch 训练在笔记本上是小时级，不是天级。3-body 数据生成更慢，因为它
  要拒绝采样掉撞墙的轨迹。
- 仓库停在 2019 年。README 写的是 Python 3.6 或 3.7、PyTorch 1.2、gym 0.12。今天
  的 Python 3.10 会在 `train.py` 的 `.next()` 上立刻报错，新版 scikit-image 会拿掉
  `skimage.draw.circle`。第 7 节 Step 1 给兼容补丁，先打补丁再采数据。给这个仓库
  单独建虚拟环境，别和第 01 课的老 gym 或第 13 课的新 torch 混装。
- 磁盘几百 MB。h5 回放不大；3-body 会先写一份 `.npz` 再转 h5。
- 建议先读论文第 2 节（对比损失和分槽）和第 4.6 节 Table 1，再对着 `modules.py`
  里的 `ContrastiveSWM.contrastive_loss`。公式和代码差一个能量缩放 $\sigma$，第 5
  节对上。

Python 侧本课用到：`torch`、`numpy`、`h5py`、`gym`（0.21 这一档，自定义 Shapes
环境用它的注册接口）、`scikit-image`、`matplotlib`、`pillow`。Atari 两条线
（Pong / Space Invaders）需要 `atari-py`，本课必做不走 Atari，可以不装。

## 4. 学习目标

1. 白纸画出 C-SWM 三件套（物体抽取器、物体编码器、图转移）的张量形状，标出 $K$
   个槽在哪一层出现、动作在哪一层拼进去；
2. 写出对比能量 $H$ 和铰链损失，并说明负样本是怎么抽的、为什么 `no_trans=True`
   的负项不跑转移模型；
3. 解释 2D Shapes 的动作编码（物体编号 $\times$ 四方向）怎样变成每个槽自己的
   one-hot，以及 `copy-action` / `ignore-action` 各服务哪类环境；
4. 用 Hits@1 和 MRR 读一次 `eval.py` 的排名矩阵，说清它量的是潜空间里的下一状态，
   不是像素重建；
5. 独立完成槽可视化、槽交换 / 删除、打乱身份、挡住一个物体四组检查，并对每种
   结局给出一句判读；
6. 口头对比 C-SWM、Slot Attention、STEVE、SlotFormer：谁在绑槽、谁在做动力学、
   谁还吃动作，以及桌宠为什么需要「杯、手机、手、头」四条分开的预测。

## 5. 原理

五个机制。每个都落到锚定仓库的类名上，验证方法写进第 7 节。

### 5.1 一条向量装不下「杯子倒了、手机没动」

第 05 课的 RSSM 把当前世界收成 $h_t$ 和 $z_t$。对 CarRacing 这够用：画面里值得
记的主要是车和路的相对姿态。对桌子不够。设观察 $o_t$ 里同时有杯子和手机。编码器
$E(o_t)=z_t\in\mathbb{R}^D$ 只有一份。转移

$$
\hat z_{t+1}=T(z_t,a_t)
$$

无论 $a_t$ 是「伸向杯子」还是「碰都不碰」，更新都作用在整条 $z$ 上。训练目标如果
再是像素误差，杯子轮廓只占画面一小块，手的运动和桌面反光会把梯度分走。C-SWM 论文
引言举过 Atari 子弹的例子：视觉上很小、对未来却致命的东西，像素损失会忽略它。

分槽之后状态变成集合

$$
z_t=(z_t^1,\ldots,z_t^K),\qquad z_t^k\in\mathbb{R}^D
$$

杯子倒只应改 $z^{\text{杯}}$，手机那一格保持。这不是审美偏好。桌宠的安全过滤要问
的是「这条伸手会不会碰到杯」，问的对象必须是杯子那个状态，不是整张纹理。

类比：一条向量像把整张桌子拍成一张糊照片再预测；一组槽像在脑子里给每样东西各立
一张卡片。类比失效处：卡片不会自动写上「杯子」两个字。绑错、绑空、两个槽抢同一个
物体，都是真实失败模式。第 7 节的掩码图就是在查这件事。

### 5.2 物体抽取器：用 $K$ 张特征图当槽

C-SWM 把编码器拆成两段，见 `modules.py` 的 `ContrastiveSWM.__init__`。

物体抽取器 $E_{\text{ext}}$ 是 CNN，最后一层的通道数等于槽数 $K$。Shapes 用
`EncoderCNNSmall`：先 $10\times 10$、步长 10 的卷积把 $50\times 50$ 的图收成
$5\times 5$，再 $1\times 1$ 卷积吐出 $K$ 张图，经 sigmoid 压到 $(0,1)$。每张图
$m_t^k$ 被论文叫做一个物体掩码。没有框监督，梯度只来自后面的对比损失。

物体编码器 $E_{\text{enc}}$ 是 `EncoderMLP`，权重在 $K$ 个槽之间共享：把每张掩码
展平，过两层 MLP，得到 $z_t^k$。Shapes 默认 $D=2$。二维不是为了好训，是为了后面
能把槽画成平面上的点，看它跟物体坐标是不是只差一个线性变换。论文 4.5 节写过：把
$D$ 加大，结论不变。

抽取器的归纳偏置很硬：一个槽对应一张空间图。两个外观相同的物体它分不开，因为卷积
滤波器认的是颜色和形状，没有迭代推理去「再找一个同样的」。论文 4.7 节把这件事写成
限制，并指向后续该用迭代绑定。Slot Attention 做的就是那一步，5.6 节再讲。

2D Shapes 环境本身在 `envs/block_pushing.py::BlockPushing`。$5\times 5$ 格子，默认
5 个物体，动作空间大小 $4K$：先选物体，再选北南东西。碰撞开启时，目标格被占则这
步不动。训练局 `ShapesTrain-v0` 每局最多 100 步，评测局 `ShapesEval-v0` 每局 10
步。观察是 $3\times 50\times 50$ 的渲染图，不是格子状态。格子状态 `get_state()`
只在环境内部，模型看不见。

### 5.3 对比能量：真转移靠近，假状态推开

C-SWM 不重建像素。它把世界模型写成知识图谱嵌入里的 TransE：头实体加关系，应靠近
尾实体。这里头实体是 $z_t$，关系是动作条件的转移 $T(z_t,a_t)$，尾实体是 $z_{t+1}$。
能量取平方欧氏距离：

$$
H=\frac{1}{K}\sum_{k=1}^{K}d\bigl(z_t^k+T^k(z_t,a_t),\,z_{t+1}^k\bigr)
$$

负样本 $\tilde z_t$ 从经验池里随机抽另一个状态的编码（代码里是把当前 batch 打乱），
负能量不跑转移：

$$
\tilde H=\frac{1}{K}\sum_{k=1}^{K}d(\tilde z_t^k,\,z_{t+1}^k)
$$

单条样本的损失是铰链：

$$
\mathcal{L}=H+\max(0,\gamma-\tilde H)
$$

论文取 $\gamma=1$。代码在 `ContrastiveSWM.energy` 里还乘了 $1/(2\sigma^2)$，默认
$\sigma=0.5$，所以实现里的 $d$ 比论文公式多一个常数缩放，优化方向不变。
`no_trans=True` 对应负项：假状态不是「走错动作到达的地方」，是「另一个无关场面」。
正项逼转移走准，负项逼不同场面在槽空间里分开。两者缺一，槽会塌成常数。

这和 [第 13 课](13_ijepa_from_scratch.md) I-JEPA 的对比不是同一个装置。I-JEPA 比的是同一张图里不同块的表征，
靠 EMA 目标编码器防塌。C-SWM 比的是时间上的真假转移，靠铰链间隔 $\gamma$ 把假
状态推到至少 $\gamma$ 远。坍缩仍然可能：所有槽输出同一个点，正负能量一起变，铰链
两边都骗得过。第 7 节可视化若看到 $K$ 张掩码几乎一样，就是这条病。

`train.py` 提供了一条对照开关 `--decoder`。打开之后不再走 `contrastive_loss`，改
成用 `DecoderCNNSmall` 重建当前帧和预测帧，像素二元交叉熵。这就是论文 Table 1 里
「- contrastive loss」那一行。2D Shapes 上它 1 步 Hits@1 大约一半，10 步掉到百分
之几。你不一定要跑满这条对照，但要知道：本课的主模型赢在目标函数，不是赢在「多
了个 CNN」。

### 5.4 图转移：槽是节点，动作贴在节点上

`TransitionGNN` 把 $K$ 个槽当成全连接图上的节点（去掉自环）。边函数

$$
e_t^{(i,j)}=f_{\text{edge}}([z_t^i,z_t^j])
$$

节点函数把自身、动作和汇总后的边拼起来：

$$
\Delta z_t^j=f_{\text{node}}\bigl([z_t^j,\,a_t^j,\,\textstyle\sum_{i\neq j}e_t^{(i,j)}]\bigr)
$$

下一步是残差更新 $z_{t+1}^j=z_t^j+\Delta z_t^j$。残差是归纳偏置：多数动作只动一个
物体，没被点到的槽 $\Delta z$ 应接近 0。边让模型看见「格子被占」这类关系。论文把
去掉 GNN、改成每个槽独立 MLP 的变体叫做「- latent GNN」：2D Shapes 上 1 步仍接近
满分，10 步 Hits@1 掉到约 0.90。碰撞是长程才显形的交互，单步看不出来。

动作怎么变成 $a_t^j$：默认 `copy_action=False` 且 `ignore_action=False`。环境给一
个整数 $a\in\{0,\ldots,4K-1\}$。`TransitionGNN.forward` 把它做成长度为 $4K$ 的
one-hot，再 `view` 成 $K\times 4$。第 $j$ 个槽拿到 4 维方向向量，没被选中的槽是
全零。这就是「对哪个物体动手」的接口。Atari 没有这种分解，仓库用 `--copy-action`
把同一个 6 维动作复制给所有槽。3-body 没有智能体，`--ignore-action` 把动作维设为
0。三条开关对应三类世界，别混用。

评测不看像素。`eval.py` 把起点编码成 $z$，在潜空间里连加 $T$ 次 $\Delta z$，再和
所有评测样本的真实 $z_{t+T}$ 算两两距离，看真实下一状态排第几。Hits@1 是排第一的
比例，MRR 是 $1/r$ 的平均。这和 [第 17 课](17_evaluating_world_models.md) 把「预测
准」和「生成真」拆开量是同一条纪律：C-SWM 故意不生成像素，所以验收也不能改回
PSNR。论文 Table 1 在 2D Shapes 上 C-SWM 的 1 / 5 / 10 步 Hits@1 为
$100\% / 100\% / 99.9\%$（四次运行的均值）。World Model (AE) 1 步也有 $98.7\%$，
10 步只剩 $6.5\%$。短程像素模型可以蒙对，长程才暴露「状态里没有物体」。

### 5.5 槽为什么能被交换：可解释性来自分槽，不是来自语言

互动实验的合法性在这里。如果 $z_t^1$ 真的绑红球、$z_t^2$ 真的绑蓝球，交换这两格
再跑 $T$，等于把两个球的身份对调后再推一步。预测应跟着身份走：原来「红球向右」
的更新会作用到蓝球那张卡片上。如果两个槽是一份复制的全局描述，交换前后能量几乎
不变。删除一格（置零或拿掉节点）等于从世界上拿掉一个物体：碰撞项应消失，别的槽
的更新应接近「场上少了一个邻居」。

这不是给物体写人设。槽里没有「红球性格开朗」这种字段，只有 $D$ 个数。桌宠若用大
语言模型给杯子编一段独白，那是在用文字冒充动力学，本课禁止。可解释性的操作定义
是：对槽做置换、删除、遮挡，预测按你改的那一格变，没改的那一格少变。

遮挡同理。把观察里一个物体涂黑，再过抽取器。绑得对的模型应主要改对应那张掩码，
其它掩码大体不动。绑得不对的模型会五张图一起抖。这是第 21 课物体恒常在槽上的小
版本：挡住不等于从状态里删掉，除非你真的把它从世界上拿走。

### 5.6 后来的零件：Slot Attention、STEVE、SlotFormer

C-SWM 的抽取器是一次前馈。Slot Attention 把槽改成一组可交换的向量，随机初始化，
然后和 CNN 特征做 $T$ 轮点积注意力。关键设计是 softmax 沿槽这一维归一化：每个空
间位置必须把注意力分给槽，槽之间竞争谁解释这块像素。算法在论文 Algorithm 1，默
认 3 轮，更新函数是 GRU。置换等变性保证「哪个槽绑哪个物体」不取决于槽的编号。
C-SWM 没有这个竞争过程，所以两个同色物体会糊。

STEVE（Slot-TransformEr for VidEos）解决的是视频。它用 Slot Attention 一类的循环
槽编码器跟踪物体，再用 SLATE 那种自回归 Transformer 解码器按槽重建每一帧。训练
目标是重建，不是 C-SWM 的对比能量。它证明了：在更脏的视频上，解码器质量会反过来
决定槽能不能出现。它仍然主要是感知加跟踪，动作条件的世界模型不是它的主场。

SlotFormer 把动力学接到槽上：先冻住一个训好的物体中心编码器，再让 Transformer 在
多帧槽序列上自回归预测未来槽。论文把它用作视频预测、VQA 的未来推理，以及规划用
的世界模型。和 C-SWM 的差别可以写成一张零件表：C-SWM 的转移是一阶 GNN 残差，
SlotFormer 的转移是多帧注意力；C-SWM 训练编码器与转移一起对比，SlotFormer 常常
先训槽再训动力学。桌宠若物体种类少、动作是「对某个物体动手」，C-SWM 的接口更贴；
若要从杂乱视频里先抠出物体再学动力学，走 Slot Attention 再接 SlotFormer。

这三篇是对照，不是本课复现对象。报告里写「我训了物体中心世界模型」时，指的是
C-SWM 在 Shapes / 小球上的 Hits 和槽图，不是 STEVE 的分割分数。

### 5.7 桌宠：杯、手机、手、头四条预测

把本课零件接回桌子。观察是一帧桌面画面。状态至少四个槽，外加一个可选的背景槽。
动作是「对哪一个槽做什么」，不是一个含糊的「伸手」。

| 槽 | 现在该装什么 | 伸手去杯沿时下一步 | 向量 $z$ 会怎样 |
|---|---|---|---|
| 杯 | 位置、是否直立、是否近桌沿 | 可能倾、滑、掉 | 和手的运动糊在一起 |
| 手机 | 位置、是否被手碰到 | 应几乎不变 | 被杯子的梯度带着漂 |
| 手 | 位置、开合、朝向 | 沿动作方向移动 | 占画面大，会主导整条 $z$ |
| 头 | 相机朝向、是否看人 | 由转头动作更新，与杯无关 | 转头变成整张图在漂 |

第 20 课解决「头转了杯子还在不在」，第 21 课解决「人把杯子挪了要不要更新」，本课
解决「更新的时候改哪一格」。三件事叠起来，第 30 课才能在自己的桌子上训
$P(s_{t+1}\mid s_t,a_t)$。若现在就用语言模型给杯子写「我不想被碰」，那是在用文字
代替动力学，验收时一律不算物体中心世界模型。

C-SWM 的动作接口在这里是优点也是限制。Shapes 的 $a$ 自带物体编号，桌宠的关节命令
没有「我在动杯子」这个标签。你要么像 Atari 那样 `--copy-action`，把同一条手臂动作
广播给所有槽，让图网络自己学谁该动；要么在第 29 课的感知层先估计手在够哪一样，再
把这个估计写成槽索引。本课只要求你看见第二种接口存在，不必实现。

## 6. 源码导读

仓库很小：根目录四个 Python 文件，外加 `data_gen/` 和 `envs/`。克隆后按这个顺序
读，每个文件带着问题进去。

| 文件 | 是哪个零件 | 带着什么问题读 |
|---|---|---|
| `envs/__init__.py` | 环境注册 | `ShapesTrain-v0` 和 `ShapesEval-v0` 的 `max_episode_steps` 为什么不同？ |
| `envs/block_pushing.py` | 2D/3D 格子 | 动作整数怎样拆成「哪个物体 + 哪个方向」？碰撞失败时状态变不变？ |
| `data_gen/env.py` | 采数据 | 非 Atari 分支为什么存 `ob[1]`？`ob[0]` 是什么、模型看不看得到？ |
| `modules.py::EncoderCNNSmall` | 抽取器 | $50\times 50$ 怎样变成 $5\times 5\times K$？sigmoid 在哪一层？ |
| `modules.py::EncoderMLP` | 编码器 | 权重共享是怎么实现的（提示：先把 $K$ 折进 batch 维）？ |
| `modules.py::TransitionGNN` | 转移 | `copy_action` 和默认分支的 one-hot 长度差在哪？全连接边列表何时缓存？ |
| `modules.py::ContrastiveSWM` | 损失 | `energy(..., no_trans=True)` 的负样本到底减的是谁？ |
| `train.py` | 训练 | `--decoder` 打开后正负样本还在不在？checkpoint 何时写入 `model.pt`？ |
| `eval.py` | 评测 | `PathDataset` 的 `path_length` 和 `--num-steps` 什么关系？Hits@1 的「候选库」是谁？ |
| `utils.py` | 数据与距离 | `StateTransitionsDataset` 和 `PathDataset` 各返回什么？`pairwise_distance_matrix` 有没有归一化？ |
| `data_gen/physics.py` | 3-body 数据 | 动作为什么全是 0？`--eval` 只改了什么？ |
| `envs/physics_sim.py` | 三体积分 | 撞墙或两球过近为什么整段重采？ |

几处容易读错的落点。

`data_gen/env.py` 里 `env.reset()` 返回 `(格子状态, 渲染图)` 这个元组，所以
`ob[1]` 才是 $3\times 50\times 50$ 的像素。格子状态有物体身份，模型训练时故意
不用，否则「无监督发现物体」这句话不成立。评测也只用像素编码后的槽。

`TransitionGNN._get_edge_list_fully_connected` 按 batch 大小缓存边列表。Shapes 默
认 $K=5$，每个样本 $5\times 4=20$ 条有向边。边函数输入是两个 $D$ 维槽的拼接，默
认 $D=2$，所以边 MLP 的第一层是 `Linear(4, 512)`。节点 MLP 的输入维是
`hidden + embedding + action_dim`，默认 $512+2+4$。这些数字你在 `print(model)`
里对得上，才说明读到了实现而不是示意图。

`eval.py` 的排名有一个实现细节：它把对角线距离再拼到距离矩阵最左侧，然后稳定排
序，标签全是 0。这样做是为了在「预测恰好等于某个别人的真实下一状态」时，自己的
那一条仍排在并列的最前。读 `utils.pairwise_distance_matrix`：平方欧氏、没有开方、
没有按维归一化。所以 Hits 比较的是同一套平方距离，和训练时的 $d(\cdot)$ 一致。

`train.py` 用 `train_loader.__iter__().next()` 取一个 batch 推断 `input_shape`。
这是 Python 2 写法，Step 1 必须改。它还把 `print` 重绑成 `logger.info`，日志同时
进终端和 `checkpoints/<name>/log.txt`。`metadata.pkl` 存整份 `args`，`eval.py`
靠它重建模型超参。改过 `num-objects` 或 `embedding-dim` 却拿错文件夹，加载会在
`load_state_dict` 上报尺寸不匹配。

## 7. 实验

必做是 2D Shapes 的训练、评测、槽可视化、交换 / 删除槽、两个消融。3-body 小球是
互动实验的加分环境：两个球的下一帧更像「谁撞谁」，但生成慢。Atari 本课不做。进
仓库根目录后再跑下面的命令。

### Step 0: 克隆与独立环境

```bash
git clone https://github.com/tkipf/c-swm.git
```

README 钉死的版本今天装不全。建议 Python 3.9 或 3.10 的干净虚拟环境，安装：
`torch`（CPU 或 CUDA 皆可）、`numpy`、`h5py`、`gym==0.21.0`、`scikit-image`、
`matplotlib`、`pillow`。不装 `atari-py`。gym 必须低于 0.26，否则 `reset()` 返回
值从观测变成 `(obs, info)`，`data_gen/env.py` 会把元组存进 h5。

### Step 1: 打兼容补丁

把下面脚本存成仓库根目录的 `compat_patch.py` 再运行。它只改三处：Python 3 的
迭代器、`skimage.draw.circle` 到 `disk`、以及 `eval.py` 加载权重时的
`map_location`。不改算法。

```python
from pathlib import Path

root = Path(".")

for name in ("train.py", "eval.py"):
    p = root / name
    t = p.read_text()
    t = t.replace(
        "train_loader.__iter__().next()",
        "next(iter(train_loader))",
    )
    t = t.replace(
        "eval_loader.__iter__().next()",
        "next(iter(eval_loader))",
    )
    t = t.replace(
        "model.load_state_dict(torch.load(model_file))",
        "model.load_state_dict(torch.load(model_file, map_location=device))",
    )
    p.write_text(t)

block = root / "envs" / "block_pushing.py"
t = block.read_text()
t = t.replace("skimage.draw.circle(", "skimage.draw.disk(")
# old: disk(r, c, radius, shape) -> disk((r, c), radius, shape=shape)
t = t.replace(
    "pos[0]*10 + 5, pos[1]*10 + 5, 5, im.shape)",
    "(pos[0]*10 + 5, pos[1]*10 + 5), 5, shape=im.shape)",
)
block.write_text(t)

phys = root / "envs" / "physics_sim.py"
t = phys.read_text()
t = t.replace(
    "from skimage.draw import circle",
    "from skimage.draw import disk as circle_disk",
)
t = t.replace(
    "rr, cc = circle(int(pos[1] * scale), int(pos[0] * scale),\n"
    "                                    radius * scale, scaled_img_size)",
    "rr, cc = circle_disk((int(pos[1] * scale), int(pos[0] * scale)),\n"
    "                                    radius * scale, shape=tuple(scaled_img_size))",
)
phys.write_text(t)
print("patched train.py eval.py block_pushing.py physics_sim.py")
```

```bash
python compat_patch.py
```

预期打印 `patched train.py eval.py block_pushing.py physics_sim.py`。打开
`train.py`，确认取 batch 那一行已经是 `next(iter(train_loader))`。Mac 上若
DataLoader 卡死，把 `train.py` 和 `eval.py` 里 `num_workers=4` 改成 `0`。

### Step 2: 生成 2D Shapes 数据

README 正式配置：

```bash
python data_gen/env.py --env_id ShapesTrain-v0 --fname data/shapes_train.h5 --num_episodes 1000 --seed 1
```

```bash
python data_gen/env.py --env_id ShapesEval-v0 --fname data/shapes_eval.h5 --num_episodes 10000 --seed 2
```

冒烟档把评测集降到 200 局、训练集 200 局，只验证流程。评测集 10000 局是论文口径，
生成要一段时间，磁盘仍很小。预期：每 10 局打印一次 `iter N`，结束后出现
`data/shapes_train.h5` 和 `data/shapes_eval.h5`。

抽查一帧。下面脚本存成 `peek_shapes.py`：

```python
import utils
import matplotlib.pyplot as plt
import numpy as np

buf = utils.load_list_dict_h5py("data/shapes_train.h5")
obs = buf[0]["obs"][0]
print("episodes", len(buf), "obs", obs.shape, "action0", buf[0]["action"][0])
img = np.transpose(obs, (1, 2, 0))
plt.imsave("peek_shapes.png", np.clip(img, 0, 1))
```

```bash
python peek_shapes.py
```

预期：`obs` 形状 `(3, 50, 50)`，动作是 0 到 19 的整数，`peek_shapes.png` 里能数出
五个彩色形状（圆 / 三角 / 方）。数不出五个，或画面全黑，先别训。

### Step 3: 训练 C-SWM（Shapes）

README 正式命令：

```bash
python train.py --dataset data/shapes_train.h5 --encoder small --name shapes
```

默认 100 epoch、batch 1024、学习率 $5\times 10^{-4}$、$K=5$、$D=2$、`sigma=0.5`、
`hinge=1`。冒烟档加 `--epochs 20`。预期：`checkpoints/shapes/log.txt` 里
`Average loss` 下降；`model.pt` 和 `metadata.pkl` 出现。损失绝对值没有通用标准，
看趋势。若 5 个 epoch 内不降，查 Step 1 补丁和数据形状。

CPU 上正式档大约数小时；有 GPU 则明显更快。batch 1024 对 50×50 很小的 CNN 不是
显存问题，是 CPU 上 DataLoader 的问题，必要时把 `num_workers` 降到 0。

### Step 4: 评测 Hits@1 和 MRR

```bash
python eval.py --dataset data/shapes_eval.h5 --save-folder checkpoints/shapes --num-steps 1
```

```bash
python eval.py --dataset data/shapes_eval.h5 --save-folder checkpoints/shapes --num-steps 5
```

```bash
python eval.py --dataset data/shapes_eval.h5 --save-folder checkpoints/shapes --num-steps 10
```

预期打印 `Hits @ 1:` 和 `MRR:`。论文 Table 1 在 2D Shapes、四次运行上，C-SWM 的
1 / 5 / 10 步 Hits@1 为 100% / 100% / 99.9%。你是单种子、可能还是冒烟数据，不要
把 0.99 写成复现失败，也不要把冒烟档的 0.4 写成复现成功。复现档（1000 训 /
10000 评 / 100 epoch）的方向是：1 步接近 1，10 步仍高。World Model (AE) 的 10 步
Hits@1 只有 6.5%，那是对照，不是你的主模型。

`eval.py` 会跳过最后一个不满 batch 的包（`batch_size=100`）。10000 局评测集约
9900 条进入统计。冒烟 200 局则只有 100 条，方差大，只看流程。

### Step 5: 可视化槽

把下面存成 `inspect_slots.py`。它读 `checkpoints/shapes`，画出：原图、K 张掩码、
交换两个最亮槽之后的下一步能量、删除一个槽之后的能量、打乱槽顺序、挡住一块区域。

```python
import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

import modules
import utils


def to_img(tchw):
    x = tchw.detach().cpu()
    if x.size(0) > 3:
        x = x[:3]
    return np.clip(x.permute(1, 2, 0).numpy(), 0, 1)


def load_model(save_folder, device, hdf5_file):
    args = pickle.load(open(os.path.join(save_folder, "metadata.pkl"), "rb"))["args"]
    sample = utils.load_list_dict_h5py(hdf5_file)
    obs0 = utils.to_float(sample[0]["obs"][0])
    input_shape = obs0.shape
    model = modules.ContrastiveSWM(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
        input_dims=input_shape,
        num_objects=args.num_objects,
        sigma=args.sigma,
        hinge=args.hinge,
        ignore_action=args.ignore_action,
        copy_action=args.copy_action,
        encoder=args.encoder,
    ).to(device)
    state = torch.load(os.path.join(save_folder, "model.pt"), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, args


def energy_of(model, state, action, next_state):
    return model.energy(state, action, next_state).detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-folder", default="checkpoints/shapes")
    parser.add_argument("--dataset", default="data/shapes_eval.h5")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", default="slot_report")
    args_cli = parser.parse_args()
    os.makedirs(args_cli.out, exist_ok=True)
    device = torch.device("cpu")
    model, _ = load_model(args_cli.save_folder, device, args_cli.dataset)
    buf = utils.load_list_dict_h5py(args_cli.dataset)
    ep = buf[args_cli.index]
    obs = torch.from_numpy(utils.to_float(ep["obs"][0])).unsqueeze(0)
    next_obs = torch.from_numpy(utils.to_float(ep["next_obs"][0])).unsqueeze(0)
    action = torch.tensor([int(ep["action"][0])], dtype=torch.int64)

    with torch.no_grad():
        masks = model.obj_extractor(obs)
        next_masks = model.obj_extractor(next_obs)
        state = model.obj_encoder(masks)
        next_state = model.obj_encoder(next_masks)
        pred_state = state + model.transition_model(state, action)

    k = masks.size(1)
    img = to_img(obs[0])
    fig, axes = plt.subplots(2, k + 1, figsize=(3 * (k + 1), 6))
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("obs")
    axes[1, 0].imshow(to_img(next_obs[0]))
    axes[1, 0].set_title("next")
    for i in range(k):
        axes[0, i + 1].imshow(masks[0, i].numpy(), cmap="magma", vmin=0, vmax=1)
        axes[0, i + 1].set_title("slot %d" % i)
        axes[1, i + 1].imshow(next_masks[0, i].numpy(), cmap="magma", vmin=0, vmax=1)
        axes[1, i + 1].set_title("next slot %d" % i)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args_cli.out, "masks.png"), dpi=140)
    plt.close(fig)

    strength = masks[0].mean(dim=(1, 2)).numpy()
    order = np.argsort(-strength)
    a, b = int(order[0]), int(order[1])
    swapped = state.clone()
    swapped[:, [a, b]] = state[:, [b, a]]
    deleted = state.clone()
    deleted[:, a] = 0
    perm = torch.randperm(k)
    shuffled = state[:, perm]

    occluded = obs.clone()
    occluded[:, :, 10:30, 10:30] = 0
    occ_masks = model.obj_extractor(occluded)
    occ_state = model.obj_encoder(occ_masks)

    e_true = energy_of(model, state, action, next_state)[0]
    e_pred = ((pred_state - next_state) ** 2).sum(-1).mean().item()
    e_swap = energy_of(model, swapped, action, next_state)[0]
    e_del = energy_of(model, deleted, action, next_state)[0]
    e_shuf = energy_of(model, shuffled, action, next_state)[0]
    e_occ = energy_of(model, occ_state, action, next_state)[0]

    lines = [
        "index %d action %d" % (args_cli.index, int(action.item())),
        "slot mean activation: %s" % np.round(strength, 4),
        "swap pair: %d %d" % (a, b),
        "shuffle perm: %s" % perm.tolist(),
        "energy true transition: %.4f" % e_true,
        "mse pred vs encoded next: %.4f" % e_pred,
        "energy after swap A/B: %.4f" % e_swap,
        "energy after delete slot A: %.4f" % e_del,
        "energy after shuffle identity: %.4f" % e_shuf,
        "energy after occlude patch: %.4f" % e_occ,
        "mask l1 change under occlude: %s"
        % np.round((occ_masks - masks).abs().mean(dim=(2, 3))[0].numpy(), 4),
    ]
    text = "\n".join(lines)
    print(text)
    open(os.path.join(args_cli.out, "numbers.txt"), "w").write(text + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img)
    axes[0].set_title("obs")
    axes[1].imshow(to_img(occluded[0]))
    axes[1].set_title("occluded")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args_cli.out, "occlude.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
```

```bash
python inspect_slots.py --save-folder checkpoints/shapes --dataset data/shapes_eval.h5 --index 0 --out slot_report
```

预期：`slot_report/masks.png` 上多数槽各亮一块不同的形状，而不是五张相同的热力图。
`numbers.txt` 里 `energy after swap` 和 `energy after shuffle` 应高于
`energy true transition`。方向对了就记下；个别样本对调后能量差不多，换 `--index`
再试三个。五个槽全亮同一团，训练没把物体分开，回 Step 3 加 epoch 或查补丁。

判读表：

| 你看见的 | 判读 |
|---|---|
| 每张掩码对应一个形状，交换后能量明显升高 | 槽绑住了物体身份，转移在用这个身份 |
| 掩码分开，但交换后能量几乎不变 | 抽取器分开了，转移没把身份用起来（动作编码或 GNN 没学到） |
| 五张掩码几乎一样 | 槽坍缩，对比损失没撑开物体维 |
| 挡住中心方块后只有 1-2 张掩码的 L1 大变 | 遮挡局部化，符合「挡住一个物体」 |
| 挡住之后五张掩码一起翻 | 槽没有空间分工，和一条向量差不多 |

### Step 6: 两个槽的下一帧，交换或删除

Shapes 上 Step 5 已经对最亮的两个槽做了交换和删除。若要更接近「两个球」，按
README 生成 3-body 数据并训练（`--ignore-action`，没有智能体，动力学来自引力）。

```bash
python data_gen/physics.py --num-episodes 5000 --fname data/balls_train.h5 --seed 1
```

```bash
python data_gen/physics.py --num-episodes 1000 --fname data/balls_eval.h5 --eval --seed 2
```

```bash
python train.py --dataset data/balls_train.h5 --encoder medium --embedding-dim 4 --num-objects 3 --ignore-action --name balls
```

```bash
python eval.py --dataset data/balls_eval.h5 --save-folder checkpoints/balls --num-steps 1
```

冒烟档把 `--num-episodes` 降到 500 / 100、`--epochs 20`。3-body 生成会拒绝撞墙轨
迹，CPU 上正式档可能要几十分钟到两小时。论文 Table 1：C-SWM 在 3-body 上 1 步
Hits@1 为 100%，10 步 75.5%。AE 基线 1 步也是 100%，10 步 67.9%，差距主要在长程。

小球训完后：

```bash
python inspect_slots.py --save-folder checkpoints/balls --dataset data/balls_eval.h5 --index 0 --out slot_balls
```

三个槽应对应三颗不同颜色的球。交换槽 A / 槽 B：引力更新会按对调后的位置算，能量
应升高。删除槽 A：少了一个质量，另外两颗的 $\Delta z$ 应改变。这就是互动实验的
操作定义。把 `masks.png` 和 `numbers.txt` 贴进笔记，写清你交换的是哪两个槽。

时间不够时，Shapes 的 Step 5 也算完成互动：选两个形状当「两个球」，同样交换或
删除。报告里写用的是哪套环境。

### Step 7: 消融，打乱物体身份

Step 5 的 `energy after shuffle identity` 是单样本探针。下面在评测集上把打乱做成
批量协议：编码后把槽维做一次随机置换，再跑转移，和未打乱的 Hits 对照。把脚本存成
`ablate_identity.py`。

```python
import argparse
import os
import pickle

import numpy as np
import torch
from torch.utils import data

import modules
import utils


def hits_from_states(pred, nxt):
    pred_f = pred.view(pred.size(0), -1)
    nxt_f = nxt.view(nxt.size(0), -1)
    dist = utils.pairwise_distance_matrix(nxt_f, pred_f)
    dist_aug = torch.cat([torch.diag(dist).unsqueeze(-1), dist], dim=1)
    dist_np = dist_aug.numpy()
    indices = np.stack(
        [np.lexsort((np.arange(len(row)), row)) for row in dist_np], axis=0
    )
    hits = (indices[:, 0] == 0).mean()
    ranks = (indices == 0).argmax(1)
    mrr = np.mean(1.0 / (ranks + 1.0))
    return hits, mrr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-folder", default="checkpoints/shapes")
    parser.add_argument("--dataset", default="data/shapes_eval.h5")
    parser.add_argument("--max-batch", type=int, default=20)
    args_cli = parser.parse_args()
    meta = pickle.load(open(os.path.join(args_cli.save_folder, "metadata.pkl"), "rb"))
    args = meta["args"]
    device = torch.device("cpu")
    dataset = utils.PathDataset(hdf5_file=args_cli.dataset, path_length=1)
    loader = data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=0)
    obs0 = next(iter(loader))[0][0]
    model = modules.ContrastiveSWM(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
        input_dims=obs0[0].size(),
        num_objects=args.num_objects,
        sigma=args.sigma,
        hinge=args.hinge,
        ignore_action=args.ignore_action,
        copy_action=args.copy_action,
        encoder=args.encoder,
    ).to(device)
    model.load_state_dict(
        torch.load(os.path.join(args_cli.save_folder, "model.pt"), map_location=device)
    )
    model.eval()

    pred_ok, pred_bad, nxts = [], [], []
    with torch.no_grad():
        for i, (observations, actions) in enumerate(loader):
            if i >= args_cli.max_batch:
                break
            if observations[0].size(0) != 100:
                continue
            obs = observations[0]
            next_obs = observations[-1]
            state = model.obj_encoder(model.obj_extractor(obs))
            next_state = model.obj_encoder(model.obj_extractor(next_obs))
            act = actions[0]
            pred = state + model.transition_model(state, act)
            perm = torch.randperm(state.size(1))
            shuffled = state[:, perm]
            pred_shuf = shuffled + model.transition_model(shuffled, act)
            pred_ok.append(pred.cpu())
            pred_bad.append(pred_shuf.cpu())
            nxts.append(next_state.cpu())
    nxt = torch.cat(nxts)
    h0, m0 = hits_from_states(torch.cat(pred_ok), nxt)
    h1, m1 = hits_from_states(torch.cat(pred_bad), nxt)
    print("intact Hits@1 %.4f MRR %.4f" % (h0, m0))
    print("shuffled identity Hits@1 %.4f MRR %.4f" % (h1, m1))


if __name__ == "__main__":
    main()
```

```bash
python ablate_identity.py --save-folder checkpoints/shapes --dataset data/shapes_eval.h5
```

预期：打乱后 Hits@1 下降。2D Shapes 的动作是「对第 $k$ 个槽动手」，槽顺序就是物
体身份；打乱等于把推错物体。下降很少，说明转移没在用身份，或槽本来就没分开。论文
没有这一行数字，这是课内探针，只报方向和你的两个 Hits。

### Step 8: 消融，挡住一个物体

Step 5 的遮挡是画面中心硬切一块。更干净的做法：用环境自己的格子状态找到一个物体
的像素，涂黑后再编码。把脚本存成 `ablate_occlude.py`。它不经过训练好的抽取器找物
体，只在像素上挖掉一个形状，避免循环论证。

```python
import argparse
import os
import pickle

import numpy as np
import torch

import envs  # noqa: F401
import gym
import modules
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-folder", default="checkpoints/shapes")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args_cli = parser.parse_args()
    meta = pickle.load(open(os.path.join(args_cli.save_folder, "metadata.pkl"), "rb"))
    args = meta["args"]
    device = torch.device("cpu")
    env = gym.make("ShapesEval-v0")
    env.seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    sample_obs = env.reset()[1]
    model = modules.ContrastiveSWM(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
        input_dims=sample_obs.shape,
        num_objects=args.num_objects,
        sigma=args.sigma,
        hinge=args.hinge,
        ignore_action=args.ignore_action,
        copy_action=args.copy_action,
        encoder=args.encoder,
    ).to(device)
    model.load_state_dict(
        torch.load(os.path.join(args_cli.save_folder, "model.pt"), map_location=device)
    )
    model.eval()

    deltas = []
    with torch.no_grad():
        for _ in range(args_cli.episodes):
            grid, obs = env.reset()
            action = env.action_space.sample()
            (grid2, next_obs), _, _, _ = env.step(action)
            obj = int(np.random.randint(0, grid.shape[0]))
            ys, xs = np.where(grid[obj] == 1)
            occ = obs.copy()
            if len(ys):
                r, c = int(ys[0]), int(xs[0])
                occ[:, r * 10 : (r + 1) * 10, c * 10 : (c + 1) * 10] = 0
            obs_t = torch.from_numpy(utils.to_float(obs)).unsqueeze(0)
            occ_t = torch.from_numpy(utils.to_float(occ)).unsqueeze(0)
            nxt_t = torch.from_numpy(utils.to_float(next_obs)).unsqueeze(0)
            act_t = torch.tensor([int(action)])
            z = model.obj_encoder(model.obj_extractor(obs_t))
            z_occ = model.obj_encoder(model.obj_extractor(occ_t))
            z_next = model.obj_encoder(model.obj_extractor(nxt_t))
            e0 = model.energy(z, act_t, z_next).item()
            e1 = model.energy(z_occ, act_t, z_next).item()
            slot_l1 = (z_occ - z).abs().mean(-1)[0].numpy()
            deltas.append((e0, e1, slot_l1, obj))
    e0 = np.mean([d[0] for d in deltas])
    e1 = np.mean([d[1] for d in deltas])
    peak = np.mean([np.max(d[2]) / (np.mean(d[2]) + 1e-8) for d in deltas])
    print("mean energy clean %.4f occluded %.4f" % (e0, e1))
    print("slot L1 peak / mean %.3f (higher means occlusion localized)" % peak)


if __name__ == "__main__":
    main()
```

```bash
python ablate_occlude.py --save-folder checkpoints/shapes --episodes 32
```

预期：挡住之后能量升高；`peak / mean` 明显大于 1，说明变化集中在少数槽。接近 1
则遮挡被摊到所有槽，和一条向量同一类失败。环境 `reset` 在 gym 0.21 返回元组，和
`data_gen/env.py` 一致。若你装到了 gym 0.26，这步会先炸，回到 Step 0 的版本约束。

### Step 9: 可选，像素解码器对照

论文「- contrastive loss」在仓库里就是 `--decoder`：

```bash
python train.py --dataset data/shapes_train.h5 --encoder small --name shapes_dec --decoder
```

```bash
python eval.py --dataset data/shapes_eval.h5 --save-folder checkpoints/shapes_dec --num-steps 1
```

```bash
python eval.py --dataset data/shapes_eval.h5 --save-folder checkpoints/shapes_dec --num-steps 10
```

论文 Table 1：这条对照在 2D Shapes 上 1 步 Hits@1 $49.9\%$，10 步 $1.4\%$。你的
数字不必贴近，方向应是：短程也许还行，长程远差于对比模型。预算不够就引用论文表，
在笔记里标明「本机未跑解码器对照」。

### Step 10: 留证据

在 `checkpoints/shapes/` 放 `NOTES.md`，最少写：

```text
日期与机器，CPU 或 GPU
仓库 commit 与 compat_patch.py 是否已跑
训练命令（含 epoch、seed 默认 42）
数据：shapes_train / shapes_eval 的局数
eval 1/5/10 步的 Hits@1 和 MRR
slot_report/masks.png 的一句判读（分开 / 塌了 / 说不清）
swap 与 shuffle 的能量或 Hits 对照
occlude 的 peak/mean
未跑的项（balls、decoder）如实写没跑
```

## 8. 配置与预算

默认超参全部抄 `train.py` 的 argparse，一个不用发明：

| 项 | 默认值 | 用在哪 |
|---|---|---|
| batch size | 1024 | `train.py` |
| epochs | 100 | Shapes / balls 的 README 都不改这个，Pong 才改成 200 |
| learning rate | $5\times 10^{-4}$ | Adam |
| encoder | `small` / `medium` / `large` | Shapes / balls / Cubes |
| embedding-dim | 2（Shapes）、4（balls） | 论文 4.5 节说加大 $D$ 结论不变 |
| num-objects | 5（Shapes）、3（balls） | 必须和环境里的物体数一致 |
| action-dim | 4 | Shapes 每个物体四个方向 |
| sigma | 0.5 | 能量缩放 |
| hinge | 1.0 | $\gamma$ |
| seed | 42 | `train.py` 默认 |

| 档位 | 数据与训练 | 设备与耗时（参考） | 用途 |
|---|---|---|---|
| 冒烟档 | Shapes 200/200 局，20 epoch；跳过 balls | CPU 一小时内 | 验补丁、看到掩码图 |
| 复现档（本课主验收） | Shapes 1000/10000 局，100 epoch；eval 1/5/10 步 | CPU 数小时，GPU 更快 | 论文复现 #6 |
| 互动加分 | balls 5000/1000 局，100 epoch，`--ignore-action` | 数据生成可能比训练慢 | 两个球的交换 / 删除 |
| 对照可选 | `--decoder` 再训一份 Shapes | 与复现档同预算 | 对照 Table 1 的「- contrastive loss」 |

论文还有 3D Cubes、Atari Pong、Space Invaders。Cubes 要用 matplotlib 的 3D 体素渲染，依赖更脆（`np.fromstring`、`Image.ANTIALIAS`、`np.bool`），本课不作为必做。Atari 需要 `atari-py==0.1.4` 和 `--copy-action`，Hits 方差大，论文自己也写了 Pong 上 $K=1$ 反而最好。这些只在第 11 节当改造选项。

磁盘：Shapes 两份 h5 通常几百 MB 以内；balls 会多一份 `.npz`。checkpoint 只有 `model.pt` 加一个很小的 `metadata.pkl`。

## 9. 验收

验收清单：

- [ ] 能在白纸上标出 `obj_extractor` 到 `obj_encoder` 到 `transition_model` 的张量形状，并说出 Shapes 默认 $K=5$、$D=2$、动作 20 维 one-hot 再切成 5×4；
- [ ] 复现档 1 步 Hits@1 明显高于随机乱猜（随机约 $1/N_{\text{eval}}$ 量级），10 步仍明显高于论文 AE 基线那一档的崩溃方向；冒烟档只要求损失下降且能出掩码图；
- [ ] `masks.png` 上多数槽各对应不同物体，而不是五张复制；
- [ ] 交换两个槽或打乱身份后，能量升高或 Hits@1 下降，笔记里有数字；
- [ ] 挡住一个物体后，能量升高，且槽 L1 变化不是均匀摊开；
- [ ] 能口头回答：向量 RSSM 做不到的是「对杯子动手时手机那一格保持」，槽模型用分槽加残差转移来做这件事；
- [ ] 能指出 C-SWM 与 Slot Attention 的差别：一次前馈掩码对迭代竞争绑定；
- [ ] `NOTES.md` 写明兼容补丁、数据局数、未跑项目。

眼见为实的附加检查：把某个槽的二维嵌入（Shapes 的 $D=2$）在评测集上散点画出来，颜色按环境格子坐标上那个物体的位置着色。论文图 3 显示槽坐标和真实格子只差一个线性变换。你的散点若是一团噪声，Hits 却很高，要怀疑评测实现；散点呈网格而掩码糊成一片，要怀疑你看错了掩码的通道维。

桌宠口头题：桌上同时有杯、手机、手、头，问模型最少几个槽、动作该写成什么。合格回答：至少四个槽（背景可以再占一格），动作必须能指向「对哪一个槽动手」，不能只有一个「伸手」标量却指望杯子和手机自己去猜。

## 10. 排错

| 症状 | 最可能的原因 | 怎么确认 | 怎么修 |
|---|---|---|---|
| `AttributeError: 'iterator' object has no attribute 'next'` | 没打 Step 1 补丁 | 打开 `train.py` 搜 `.next()` | 跑 `compat_patch.py` |
| `ImportError: cannot import name 'circle'` | scikit-image ≥ 0.19 删了 `circle` | `python -c "import skimage; print(skimage.__version__)"` | 补丁里的 `disk` 替换；或把 scikit-image 降到 0.16 |
| `reset() returns tuple` 或 h5 里 obs 形状离谱 | gym ≥ 0.26 | `python -c "import gym; print(gym.__version__)"` | 改回 `gym==0.21.0`，重建虚拟环境 |
| DataLoader 卡住不动 | macOS + `num_workers=4` | 把 workers 改 0 立刻能跑 | `train.py` / `eval.py` 的 `num_workers=0` |
| `RuntimeError: size mismatch` 加载模型 | `--name` 指错文件夹，或 `num-objects` 和训练时不同 | 读 `metadata.pkl` 里的 args | `--save-folder` 必须是训练时的那个 `checkpoints/<name>` |
| 训练损失不降 | 数据空、补丁改坏损失、`decoder` 误开 | 看 `peek_shapes.py` 的形状和 `log.txt` 是否在走 `contrastive_loss` | 先修数据；确认没传 `--decoder` |
| 五张掩码一模一样 | 槽坍缩或 $K$ 与物体数差太远 | Step 5 的 mean activation 是否几乎相等 | 加 epoch；确认 `--num-objects 5`；换几帧看是不是偶然 |
| Hits@1 接近 0 | eval 用了训练集、或 `num-steps` 超过 eval 局长度 | `ShapesEval-v0` 只有 10 步 | 评测集必须是 `shapes_eval.h5`；`--num-steps` 不超过 10 |
| `ablate_occlude.py` 在 `env.seed` 报错 | gym 新 API 去掉了 `seed` | 报错栈 | 回到 gym 0.21；不要在这步改写成 gymnasium |
| 3-body 生成很久且 `iter` 不像 env.py 那样刷 | 撞墙就重采整段 | `physics_sim.py` 里 `collision = True` 的循环 | 属正常；冒烟减 episode；不要改物理参数冒充论文设置 |
| CUDA 和 CPU 的 `edge_list` 缓存错设备 | 改过设备但 `TransitionGNN.edge_list` 还在 | 第一次 forward 后换 `.cuda()` | 训练脚本本身不会换设备；自己写探测脚本时先 `model.transition_model.edge_list = None` |

## 11. 前沿与改造

同一问题，2020 到 2023 年公开系统换了三件零件。绑槽：C-SWM 的前馈 $K$ 通道换成 Slot Attention 的迭代竞争，再换成 STEVE 在视频上的循环槽。动力学：一阶全连接 GNN 换成 SlotFormer 那种对多帧槽做自回归的 Transformer。目标：对比能量仍在，但 STEVE 证明在脏视频上重建解码器的质量会决定槽能不能出现；SlotFormer 常常冻住编码器只训转移。动作：C-SWM 假设动作能分到物体上，机器人里这通常不成立，后来的视觉预测控制更多是一个连续向量广播到所有槽，接近仓库的 `--copy-action`。

缩小版和前沿版的差距，规模占一块：Shapes 是 50×50 纯色几何，桌面摄像头是杂乱纹理、互遮挡、同类别多实例。机制占一块：没有迭代绑定就分不开两只一样的杯子；没有跨帧记忆，手挡住杯子两秒槽就会把杯子让给手；没有动作条件，SlotFormer 的视频预测还不是第 03 课意义上的世界模型。钱买得起更大的 STEVE，买不来「动作必须改变某个槽的未来」这件事，那是本课实验能教的。

动手改造清单（选做，各含预算和失败判据）：

1. 解码器对照。命令即 Step 9。预算：与主模型同数据同 epoch。预期：1 步 Hits 尚可，10 步明显低于对比模型。失败判据：两条曲线打平，先查 `--decoder` 是否真的走了像素分支（`train.py` 里 `if args.decoder`），再查评测是不是加载了错误文件夹。
2. 去掉图交互。在 `TransitionGNN.forward` 里把 `num_nodes > 1` 的边计算短路，使 `edge_attr is None`。预算：重训一份 Shapes。预期：1 步 Hits 仍高，10 步下降，方向同论文「- latent GNN」（10 步 Hits@1 约 0.90 对 0.999）。失败判据：10 步完全不掉，说明你的评测集动作很少撞上碰撞，加大评测局数。
3. 槽数扫一遍。`--num-objects` 取 1、3、5、8 各训一份冒烟档（20 epoch）。预算：四组，CPU 一个下午。预期：$K=1$ 退化成向量模型，长程 Hits 差；$K=5$ 对齐物体数最好；$K=8$ 多出的槽应接近空掩码。失败判据：四组打平，记为「冒烟数据分不出 $K$」，改复现档再下结论。
4. 桌面最小槽编码器。用笔记本摄像头拍 20 秒桌子（杯、手机、手入画），每帧过一个冻结的随机 `EncoderCNNSmall` 或自己写的 4 槽 CNN，画出掩码。预算：一小时，不训练动力学。预期：几乎肯定绑不稳。失败是默认结局，写进笔记：前馈抽取器在自然图像上不够，这正是 Slot Attention 要解决的问题。不要为了让掩码好看去用语言识别框冒充槽。

顺手复现映射：

| 论文结论 | 缩小版对应实验 | 预期 |
|---|---|---|
| 对比损失优于像素重建（Table 1 Shapes 10 步） | Step 9 `--decoder` | 能复现方向，数字不必贴近 |
| 去掉 GNN 伤害长程 | 改造 2 | 1 步不明显、10 步明显，同方向 |
| 去掉分槽伤害组合泛化 | 改造 3 的 $K=1$ | 能看到长程 Hits 变差 |
| 槽可解释、对应物体（图 3） | Step 5 掩码图 | 合成环境上通常看得到；桌面视频上通常看不到 |

## 12. 论文与延伸

1. C-SWM（Kipf, van der Pol, Welling, ICLR 2020, [arXiv:1911.12247](https://arxiv.org/abs/1911.12247)）。本课宪法。带着三个问题读：第 2.2 节把 TransE 改成 $z+T(z,a)$ 时，为什么负样本不跑 $T$？Table 1 里 Atari Pong 的 $K=5$ 反而差于 $K=1$，这对「槽越多越好」意味着什么？4.7 节列的两条限制（同外观实例、马尔可夫）哪一条会先在桌宠摄像头上爆？
2. Slot Attention（Locatello 等, NeurIPS 2020, [arXiv:2006.15055](https://arxiv.org/abs/2006.15055)）。带着问题读：Algorithm 1 的 softmax 为什么沿槽归一化而不是沿空间位置？置换等变性（Proposition 1）若被打破，测试时多加一个槽还会不会工作？它本身是世界模型吗，缺了循环里的哪一格？
3. STEVE（Singh, Wu, Ahn, [arXiv:2205.14065](https://arxiv.org/abs/2205.14065)）。标题是 *Simple Unsupervised Object-Centric Learning for Complex and Naturalistic Videos*。带着问题读：它把 SLATE 的槽条件 Transformer 解码器接到视频上，训练目标为什么退回重建？文中说 SLATE 单帧分割已经强过当时的视频 Slot Attention 基线，那 STEVE 多出来的时间模型到底买到了什么（提示：MOVi-Tex 和跨帧身份）？
4. SlotFormer（Wu, Dvornik, Greff, Kipf, Garg, ICLR 2023, [arXiv:2210.05861](https://arxiv.org/abs/2210.05861)）。带着问题读：编码器为什么先冻住再训 Transformer 动力学？它怎样声称自己能当规划用的世界模型，动作是在哪一层进模型的？和 C-SWM 的 GNN 残差比，多帧注意力解决了 4.7 节哪一条限制、没解决哪一条？
5. 锚定仓库 [tkipf/c-swm](https://github.com/tkipf/c-swm) 的 README。带着问题读：五组环境和五组训练命令里，哪几个开关（`--copy-action`、`--ignore-action`、`--encoder`、`--embedding-dim`）是论文正文里的设定，哪几个是实现细节？它没有 `requirements.txt`，这对你在 2026 年复现意味着什么？

到这课为止，第六幕的空间零件齐了：3D 让离开视野的杯子还在，4D 让挪过的杯子被更新，视频基础模型回答「大模型什么时候算世界模型」，物体槽让杯子和手不再共用一个向量。循环里「状态」这一格第一次有了物体边界。下一课把预测接上控制：Visual Foresight 和 DayDreamer 教你在想象里先推一把杯子，再决定真不真伸手。槽如果已经分开，那次想象就可以只动杯子那一格。
