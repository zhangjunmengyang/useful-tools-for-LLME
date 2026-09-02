---
id: 24_visual_foresight
title: "看见未来再动手"
summary: "会预测下一帧，怎样变成先在脑子里推一把杯子，再决定真不真推？"
unit: embodied
play_tools: []
checkpoints:
  - "规划超过动作盲基线的记录（论文复现 #7）。"
  - "DayDreamer 接口精读笔记。"
---

# 第 24 课：看见未来再动手

> 类型：复现（论文复现 #7：公开环境上的动作条件预测 + 零样本规划探针，方向性）+ 精读（DayDreamer 真机基础设施，四台机器人未复现）<br>
> 建议周期：3-5 天（主实验一天能出对照表；DINO-WM 选做再加 1-2 天；DayDreamer 精读半天）<br>
> 硬件：gym-pusht 主实验 Mac / 纯 CPU 可完成；单张 24GB 卡可跑像素模型和 DINO-WM 选做。本课不要求任何真机<br>
> 锚定仓库：[huggingface/gym-pusht](https://github.com/huggingface/gym-pusht)（主实验环境），[gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm)（选做：官方 PushT 规划），[danijar/daydreamer](https://github.com/danijar/daydreamer)（精读，不跑四台真机）；论文 Visual Foresight（arXiv:1812.00568）、DayDreamer（arXiv:2206.14176）、DINO-WM（arXiv:2411.04983）<br>
> 产物：PushT 上随机 / 动作盲 / 动作条件三组规划对照、三条想象轨迹的落桌标注、一份写明「未复现 DayDreamer 真机」的精读笔记

## 1. 这一课做什么

第六幕把状态从一张图变成了可查询的空间和可分开的物体。第 20、21 课让离开视野的杯子还在、挪过的杯子被更新；第 22 课在 DINO-WM 上做过动作对换和一次缩小规划；第 23 课把杯子和手分到不同槽。那还只是「看见」和「分得开」。主干循环的后半截，选动作，一直停在仿真游戏里：

```text
观察 先压成状态 再按动作预测下一状态 然后展开多条未来 给未来打分 最后选动作
```

第七幕从本课起把后三个动词接到操作上。[第 33 课](33_embodiment_degrees.md) 的尺子把这一幕放在 E2 到 E4：PushT 上的视觉 MPC 是可重置的 E2；接到真机并查询模型才是 E4。本课先把 E2 的「想完再动」跑通。桌宠真正危险的动作不是转头，是伸手。杯子离桌沿还有 4 厘米，手往哪边推 2 厘米，决定它留下来还是落地。人会在动手前先在脑子里走一遍；程序要做同一件事，就必须有一个听得懂动作的预测器，再加上一个在预测里挑动作的规划器。

本课是第七幕第一块肌肉，也是整门课第一次把「想完再动」用到桌面推物。零件来自两条已经发表的路。一条叫视觉预见（Visual Foresight：用动作条件视频预测当模型，在想象的画面上搜动作），2017 年 Finn 与 Levine 给出视觉模型预测控制（visual MPC），2018 年 Ebert、Finn、Dasari、Xie、Lee、Levine 写成完整系统（arXiv:1812.00568）。另一条叫 DayDreamer：把第 06 课的 Dreamer 直接放到真机上在线学，不再经过仿真器（Wu、Escontrela、Hafner、Goldberg、Abbeel，CoRL 2022，arXiv:2206.14176）。中间还要接上第 05 课已经拆过的 CEM，以及第 22 课已经跑过的 DINO-WM：本课不再重复它的安装，而是把同一套「动作条件预测 + 规划」接到控制循环上，并强制做动作盲对照。

动手环境三选一里，本课主线选 PushT。它是公开的桌面推物标准环境（Hugging Face 的 `gym-pusht`），DINO-WM 论文也用它报告规划成功率，安装是一条 pip，不需要真机械臂。Genesis 桌上推物留给第 27 课的仿真到真机；DINO-WM 官方权重作为 24GB 卡上的选做。DayDreamer 的四台真机（A1 四足、UR5、XArm、Sphero）本课只精读代码与论文数字，**没有在真机上复现，也不把论文的 1 小时走路、8 小时抓放写成你跑出来的结果**。

论文复现 #7 的合格线是方向性的：在同一套规划器下，动作条件模型的任务进度超过动作盲模型和随机策略。这已经足够证明「预测必须听动作」不是口号。它不够证明你复现了 DINO-WM 的 0.90 PushT 成功率，更不够证明 DayDreamer 的真机曲线。

术语速查：

| 术语 | 一句人话 |
|---|---|
| 视觉预见 / Visual Foresight | 用动作条件视频预测当世界模型，在想象的画面上选动作；本课的第一条历史线 |
| visual MPC | 视觉模型预测控制：每步用预测模型现场规划一小段，只执行第一步，下一步重来 |
| DNA / SNA | 动态神经平流及其带时间跳连的版本：用光流场去「搬」上一帧的像素，而不是凭空画下一帧 |
| 指定像素目标 | 在当前画面上点一个物体像素、再点它该去的位置，用预测的像素运动当规划代价 |
| CEM | 交叉熵方法：撒一批动作序列，模型里打分，留下尖子，收缩采样分布，第 05 课用过 |
| DayDreamer | 把 Dreamer 的世界模型 + 想象里的 actor-critic 直接接到真机，在线学，不用仿真器 |
| actor / learner 分线程 | 真机要低延迟出动作，学习可以在另一块卡上慢慢算；DayDreamer 把这两件事拆开并行 |
| 零样本规划 | 模型只在离线轨迹上学生动力学，测试时才给定目标，现场优化动作，不再为每个任务训策略 |
| 动作盲 | 预测器不吃动作，或训练时动作被丢掉；同一状态下左推右推的想象重合，规划器无从挑选 |
| PushT | 俯视桌面把 T 形滑块推进 T 形目标区；动作是二维目标坐标，奖励是滑块与目标的覆盖率 |
| 覆盖率 / coverage | PushT 的奖励：滑块与目标区重叠比例，完全重合为 1 |

## 2. 问题

会预测下一帧，离「先在脑子里推一把杯子，再决定真不真推」还差两步。本课要补上它们，并分清三条经常被混成一句的路线。

1. 预测怎样变成控制。像素预测可以很像真，却只是在播视频。控制需要一个标量代价：这条想象轨迹离目标还有多远、会不会把杯子送下桌。Visual Foresight 的答案是三种可替换的代价（指定像素、目标图配准、成功分类器），共用同一个视频预测器和同一套 CEM。你要能讲清每种代价解决什么任务、在什么情况下会失效。
2. 想象里选动作有两种用法，不要混。PlaNet 和 Visual Foresight 每走一步都现场用 CEM 搜；Dreamer 和 DayDreamer 用想象轨迹训练一个 actor，上场时直接吐动作。前者模型一更新、规划立刻变；后者把搜索费摊到训练期，真机上延迟更低。桌宠两样都用得上：安全层适合每步现搜，日常动作适合一个训好的手。
3. 动作必须进入预测。第 03 课的动作对换在这里变成规划器的命门：动作盲模型的想象不分岔，CEM 打出的分全一样，规划退化为随机。论文复现 #7 要你在 PushT 上把这件事量出来，对照随机策略和动作盲，而不是只看预测损失降了没有。

界限先划死。本课对 Visual Foresight 是机制精读加缩小规划，不复现 2018 年 Sawyer 真机操作上百种物体的数字。对 DayDreamer 是仓库精读加论文数字转述，四台机器人、1 小时走路、8 到 10 小时抓放，全部标成「论文报告，本课未跑」。对 DINO-WM 能跑通官方 PushT 规划的算实战，跑不通就停在 gym-pusht 自训的方向性对照，不要用论文表格里的 0.90 顶替你的日志。

## 3. 准备

- 上一课留下的系统：第 23 课把状态拆成槽，第 22 课已经克隆过 `dino_wm` 的人可以直接复用那个 conda 环境做选做 Step 8。本课真正直接用上的旧零件是第 05 课的 CEM-MPC（撒点、推演、打分、收缩）和第 06 课 Dreamer 的想象训练；第 03 课的动作对换和第 17 课「预测准不等于规划好」会反复出现。没有真机、没做完第 23 课的槽可视化，也可以做本课主实验。
- 环境：Python 3.10 左右，PyTorch，Gymnasium。主实验只多装一个 `gym-pusht`。有显示器更好，无显示器把渲染设成 `rgb_array`，必要时给 pygame 一个空显示驱动（第 10 节）。
- 选做 DINO-WM：第 22 课若已按 `environment.yaml` 建好环境，本课不要另装一份。新装则按官方 README。PushT 本身是 pymunk 物理；仓库里的 MuJoCo 说明主要给 PointMaze / Reach。数据集和部分 checkpoint 放在 OSF，链接以仓库 README 的 Datasets 节为准，打不开就不要卡在这一步。
- DayDreamer：TensorFlow 栈（README 写明 `tensorflow`、`tensorflow_probability`、`ruamel.yaml`、`cloudpickle`），和本课程主线的 PyTorch 不是同一套。精读足够；想摸 `a1_sim` 仿真任务可以另开环境，仍不是真机复现。
- 磁盘：gym-pusht 随机轨迹很小（几千局状态序列几十 MB）。若下载 DINO-WM 官方数据，按 OSF 页面的体积预留，常见是数 GB。
- 心理预算：主实验是「今天之内看出三条曲线谁高」；官方 DINO-WM 规划是「有卡再加一天」；DayDreamer 真机是「读懂为什么他们能在真机上跑，并且明确自己没跑」。

## 4. 学习目标

1. 白纸画出 visual MPC 的单步循环：当前观察进预测器、CEM 在想象里打分、只执行第一步、用新观察重规划；标出代价函数接在哪；
2. 用第 03 课的动作对换解释：动作盲的视频预测哪怕每帧都清晰，为什么仍然不能拿来推杯子；
3. 对照说出 Visual Foresight、PlaNet、Dreamer/DayDreamer、DINO-WM 各自预测什么、怎么选动作、训练时要不要奖励；
4. 默写 CEM 四拍，并指出 Visual Foresight 在像素（或指定像素）上打分、PlaNet 在奖励头上打分、DINO-WM 在目标特征距离上打分，搜索引擎可以是同一个；
5. 讲清 DayDreamer 相对第 06 课多了什么工程（actor/learner 分线程、多模态编码、同一套超参上四台真机），以及本课为什么没有复现那些数字；
6. 在 PushT 上交出一张对照表：随机、动作盲、动作条件，规划进度同方向于「动作条件更好」，并保存三条想象轨迹的落桌标注。

## 5. 原理

七个机制。每个仍然走直觉、运转、数学、代码落点、验证。第 05 课已经把 CEM 拆过一遍，这里不重讲采样公式，只讲它接到视觉预测和真机上时多出来的部分。

### 5.1 预测怎样变成控制

倒车时你会在脑子里试：方向盘再往左一点，车尾会不会蹭柱子。那个「试」有三步，缺一不可。第一步，根据现在的画面和准备做的动作，想象下一刻的画面。第二步，给想象打分：蹭到了就差，没蹭到且更接近库位就好。第三步，挑分数最好的那一小步真的打出去，落地后再想。少了第一步你是在盲试；少了第二步你只是在看电影；少了第三步你永远停在想象里。

世界模型课从第 01 课就把第一步写成

$$
P(s_{t+1} \mid s_t, a_t)
$$

控制把第二步和第三步接上。给定目标 $g$（一个像素位置、一张目标图、一个成功分类器、或 PushT 里滑块该去的姿态），规划器解的是

$$
a_{t:t+H}^{\star} = \arg\min_{a_{t:t+H}} \sum_{\tau=t}^{t+H} c\big(\hat{s}_{\tau}, g\big), \qquad \hat{s}_{t}=s_{t},\ \hat{s}_{\tau+1} \sim P(\cdot \mid \hat{s}_{\tau}, a_{\tau})
$$

MPC 的用法是：这条最优序列只执行 $a_t^{\star}$，真实世界走一步，得到新观察，再重新解一次。模型有偏时，重规划用新观察把状态拽回来，短视的错误不容易滚出桌子。

验证这件事的最小实验不是看预测图好不好看，是看「换一个目标 $g$，同一套模型选出的动作是不是跟着变」。第 7 节三条想象轨迹就是这个检验：同一张桌面，三个目标（推进目标区、推向画布边缘、原地小动），预测必须分岔，分岔必须能被标成「会不会落下桌」。

### 5.2 光流式视频预测：DNA 与 SNA

直接用卷积「画出」下一帧，模型会把物体的颜色和纹理重新发明一遍，一发明就会糊。Finn、Goodfellow、Levine 在 2016 年（NIPS，动作条件视频预测）和 Finn、Levine 在 2017 年（ICRA，arXiv:1610.00696）改用另一种假设：下一帧里的大多数像素是上一帧里某个像素搬过来的。网络不负责发明杯子长什么样，只负责报每个位置的像素从哪来。这个结构叫动态神经平流（DNA：dynamic neural advection）。

记当前帧为 $I_t$、动作为 $a_t$，网络输出一个二维流场 $\hat{F}_{t+1 \leftarrow t}$，用双线性采样把上一帧 warp 成下一帧：

$$
\hat{I}_{t+1} = \hat{F}_{t+1 \leftarrow t} \diamond I_t
$$

流场本身可以当成像素的随机转移算子。用户在第一帧点了一个物体像素，位置分布 $P_0$ 是该点处为 1、别处为 0 的图；之后每步用同一个流场去搬这个分布：

$$
\hat{P}_{t+1} = \hat{F}_{t+1 \leftarrow t} \diamond \hat{P}_t
$$

规划器要的「这个杯子会被推到哪」就从视频模型里免费掉出来，不必另训一个跟踪器。这是 Visual Foresight 把预测接进控制的第一块巧：代价定义在像素运动上，监督信号却全部来自重建下一帧，全程没有人标物体位置。

DNA 有一个硬伤。机械臂挡住杯子再移开，杯子的像素在中间那些帧里消失了，模型从「上一帧」里找不回来。Ebert 等人在 2018 年的完整 Visual Foresight 里给它加上时间跳连，变成 SNA（skip connection neural advection）：下一帧既可以从刚刚预测的 $\hat{I}_t$ warp，也可以直接从真实的第一帧 $I_0$ 取像素，再用一组 mask 加权合成。遮住的杯子还能从 $I_0$ 里被「召回」。论文用「推一个物体、另一个保持不动」这种必须看见被挡物体的任务，比较 DNA 和 SNA；SNA 把被推物体的改进从大约 0.83 像素拉到 10.6 像素（64×64 图像上的均值，论文 Table I）。

训练数据来自真机自己乱互动：随机动作、上百种物体、没有奖励、没有复位。模型学的是「这么推，画面会怎么变」，不是「这个任务该怎么完成」。测试时用户才给出目标。这和 DayDreamer、Dreamer 那种边做任务边学奖励的设定正好相反，也是后面 DINO-WM 坚持离线、任务无关的祖先。

代码落点：2018 年系统的开源在 [SudeepDasari/visual_foresight](https://github.com/SudeepDasari/visual_foresight)，TensorFlow 老栈，本课不把它当训练环境。机制以论文第 IV 节和项目页 [sites.google.com/view/visualforesight](https://sites.google.com/view/visualforesight) 为准。你要验证的是「流场能搬指定像素」，第 7 节用 PushT 的滑块坐标做同一件事的低维版本：预测器输出的是 5 维状态而不是光流，但「动作进模型、物体位置从预测里读出来」这条链是一样的。

### 5.3 三种代价：指定像素、目标图、成功分类器

有了未来帧，还缺 $c(\hat{s}, g)$。最笨的办法是拿预测帧和一张目标照片做逐像素 $\ell_2$。Visual Foresight 论文写得很直接：机械臂和影子占了画面的大部分，规划器会先把胳膊摆成目标图里的姿势，桌上的小物体根本排不上号。所以他们不用整图 $\ell_2$ 当主代价，改了三种更针对物体的写法。

指定像素。用户点物体上的一个点 $d_0$，再点它该去的位置 $d_g$。代价是预测像素分布到目标的期望距离：

$$
c = \sum_{t=1}^{T} \mathbb{E}_{\hat{d}_t \sim P_t}\big[\|\hat{d}_t - d_g\|_2\big]
$$

好处是界面快，一个点就定义「把这个杯子往左移 10 厘米」；对桌面上的干扰物也稳，因为其他像素根本不进代价。坏处是每步重规划时你得知道物体现在在哪。短距离推还行，长距离推、中间被人打掉，单靠模型自己把 $P_t$ 滚下去会漂。

目标图配准。再训一个配准网络 $R$，把当前帧配到起始帧和目标帧，读出指定点在当前帧里的位置，规划代价用配准成功的那一侧。配准和视频预测吃同一批无标签互动数据，仍然没有人工框。论文在长距离推的 20 个配置上，用配准的成功率 66%，用 OpenCV 跟踪 45%，只用预测器自己传播 20%（论文 Table II，成功定义为距目标小于约 7.5 cm）。桌宠用得上：人把杯子挪开再放回来，状态得重新锁上，不能死抱第一帧点过的那个像素。

成功分类器。有些目标不是一个点，是一类状态：「叉子放在盘子右边」，绝对坐标无所谓。他们用少量成功例子、配合元学习（CAML/MAML 一类）现场适配一个成功分类器，规划时把「预测帧被判为成功的概率」当负代价。代价更高的是标注成本：你得给每个新任务准备几张成功图。

三种代价共用同一个预测器和同一个 CEM。换的是 $c$，不是 $P(s'|s,a)$。这是本课对桌宠最有用的接口划分：世界模型负责「这么动会怎样」，任务层负责「怎样算好」。第 26 课的 VLA 会再来一次这种划分；现在先记住，预测器和任务代价不是同一个网络。

### 5.4 CEM：同一台搜索引擎，打分函数可以换

第 05 课 PlaNet 的 CEM 是四拍：从当前动作分布抽 $J$ 条长度为 $H$ 的序列；用世界模型的先验把每条在潜空间推演 $H$ 步；用奖励头打分；留前 $K$ 名，用它们的均值和方差更新分布。重复 $I$ 轮，执行均值的第一步。PlaNet 论文默认 $H=12, I=10, J=1000, K=100$。

Visual Foresight 用的是同一台引擎，打分从奖励头换成 5.3 的像素代价，推演从潜状态换成视频预测（或指定像素的分布）。DINO-WM 还是这台引擎，打分换成「预测终点特征和目标图特征的 MSE」。三个系统对外说法差得很远，搜索步骤可以逐行对齐。第 08 课的 MPPI 是它的软化版：不硬截断尖子，按分数做指数加权。本课主实验用硬截断的 CEM，方便和第 05 课对照。

和第 06 课 Dreamer 的差别在「搜索发生在何时」。CEM-MPC 每个环境步都现场搜，模型变好的当步就能用上，账单是每步成千上万次前向（PlaNet 大约 10 轮 × 1000 条 × 12 步）。Dreamer 把搜索费搬进训练：在想象里用反传或 REINFORCE 训一个 actor，真机上一次前向就能出动作。DayDreamer 选后一条，因为四足 20 Hz、机械臂也等不起每步跑完一轮 CEM。桌宠的安全层可以走前一条（「这步会不会把水推下桌」，宁可多等 100 毫秒），日常点头挥手走后一条。

类比到此失效。飞行员的模拟器是工程师按物理造的，误差有界；这里的模拟器是从数据里学的，策略会专挑它学错的地方走。Visual Foresight 用 MPC 重规划缓解；DayDreamer 用在线更新模型缓解；谁都没有从根上消掉第 04 课见过的「钻空子」。第 7 节动作盲对照会让你看到一种更早的失败：模型还没来得及学错，已经因为听不见动作而无法规划。

### 5.5 DayDreamer：Dreamer 上真机，不加新算法

DayDreamer 的主张写在摘要第一句之后：他们把 Dreamer 接到 4 台真机上在线学，不加新算法，不经仿真器。论文用的是 DreamerV2 官方实现（Hafner 等，arXiv:2010.02193）加一套真机基础设施。世界模型仍是 RSSM 四件套，第 05、06 课已经拆过：

$$
\begin{aligned}
z_t &\sim \mathrm{enc}_\theta(z_t \mid s_{t-1}, a_{t-1}, x_t) \\
\hat{x}_t &\approx \mathrm{dec}_\theta(s_t) \\
s_t &\sim \mathrm{dyn}_\theta(s_t \mid s_{t-1}, a_{t-1}) \\
\hat{r}_t &\approx \mathrm{rew}_\theta(s_{t+1})
\end{aligned}
$$

编码器把所有感官（关节角、力、RGB、深度）熔进离散随机码 $z_t$，循环状态 $h_t$ 负责把序列串起来。解码器重建感官，方便人检查想象，策略学习阶段不走像素。奖励头预测任务奖励。行为在潜空间里用 actor-critic 学，想象视野 $H=16$，critic 拟合 $\lambda$-return，连续动作用重参数梯度，离散动作用 REINFORCE，外加熵正则。这些和第 06 课是同一套；本课新的是真机上必须补的三件事。

第一，actor 线程和 learner 线程拆开。四足 20 Hz，等一轮世界模型训练再出动作会把控制周期打穿。论文 Figure 2：当前策略在真机上采数据，写入 replay；learner 不停从 replay 做监督训练世界模型和 actor-critic；actor 只负责低延迟算动作。没有第 06 课那种「每环境步固定 train_ratio」的同步闸。

第二，同一套超参打四台差异很大的机器。A1 四足：12 维连续电机角度、本体感受、稠密分阶段奖励（先直立、再站姿、再前向速度），从仰面朝天开始，1 小时内翻身、站起、用 pronking 步态走路；之后人为推倒，10 分钟内学会扛轻推或迅速翻回来。UR5：第三人称 RGB + 本体感受，离散动作，稀疏奖励（抓住 +1、同框放下 -1、放到对面框 +10），8 小时到约每分钟 2.5 个物体，接近摇杆上的人。XArm：更慢的消费级臂，RGB + 深度 + 本体感受，软物体，10 小时到约每分钟 3.1 个，Rainbow 停在「抓住又在同一框放下」的局部最优。Sphero Ollie：只有俯视 RGB、连续力矩、朝向从单帧看不出来，2 小时内导航到固定目标，平均距离约 0.15（以场地边长为单位）。这些数字全部来自论文第 3 节，本课未复现。

第三，他们公开对抗的是「真机必须先在仿真里练」这条当时的主流。相关工作里明确把 Visual Foresight 列为短视野、规划时还要生成图像、算得慢的前辈；DayDreamer 改在潜空间大规模并行想象（论文写单卡 batch 大约 16K），用策略网络消化视野以外的长期价值。附录 B 把想象解码成图给人看，也能看见模型会把一个球「想象」成另一个颜色：潜空间规划有用，不等于解码出来的视频物理正确。这和第 17 课「生成真 ≠ 规划好」是同一条缝。

局限论文自己写了：真机磨损耗、需要人在场处理卡住和维修、照明剧变（XArm 窗外日出）会先把成绩打穿再花大约 5 小时适应。桌宠质量更小，照样能扫落液体；第 27 课会把安全层接到这套规划器的出口。本课只要求你读懂：世界模型让真机少试错，但它仍然在真机上试错，不是零风险。

仓库落点见第 6 节。验证方法：对照 README 里 `a1_sim` / `a1_real`、`xarm_dummy` / `xarm_real`、`--run learning` / `--run acting` 能否讲出「仿真任务名、真机任务名、学习线程、执行线程」四件事。能讲清，精读过关。跑通 `a1_sim` 也不等于复现了 A1 真机走路。

### 5.6 DINO-WM：冻结补丁特征上的零样本规划

Visual Foresight 在像素里预测，规划时要生成图。DayDreamer 在自己学的 RSSM 里预测，但训练是在线的、带奖励的，换一个任务往往要重学。DINO-WM（Zhou、Pan、LeCun、Pinto，arXiv:2411.04983）要同时满足三件事：离线轨迹就能训、测试时再为具体目标优化动作、模型本身尽量任务无关。做法是把观察编码器换成冻结的 DINOv2 补丁特征，转移模型只在特征上做。

$$
z_t = \mathrm{enc}(o_t),\qquad z_t \in \mathbb{R}^{N \times E}
$$

$\mathrm{enc}$ 是预训练 DINOv2，训练和测试都不更新。转移模型是一个去掉 tokenization 的 ViT，吃长度为 $H$ 的历史特征和动作，按帧做因果注意（同一帧的补丁看成一组，不在帧内逐 token 自回归），输出下一帧补丁。动作经 MLP 升维后拼到每个补丁上。训练目标是教师强制下的特征一致：

$$
\mathcal{L}_{\mathrm{pred}} = \big\| p_\theta\big(\mathrm{enc}(o_{t-H:t}),\ \phi(a_{t-H:t})\big) - \mathrm{enc}(o_{t+1}) \big\|^2
$$

解码器可选，只用转置卷积从补丁画回图像，给人看；它的重建损失不回传到转移模型。规划时更不需要解码。给定当前图 $o_0$ 和目标图 $o_g$，CEM 优化动作序列，代价是终点特征对目标特征的 MSE：

$$
\mathcal{C} = \|\hat{z}_T - z_g\|^2, \qquad \hat{z}_0=\mathrm{enc}(o_0),\ z_g=\mathrm{enc}(o_g),\ \hat{z}_{t}=p(\hat{z}_{t-1}, a_{t-1})
$$

论文在六个环境上做零样本目标到达，不给专家示范、不训奖励、不训逆模型。PushT 上他们报告成功率 0.90，对照 DreamerV3 0.30、IRIS 0.32（论文 Table 1）。把编码器换成 R3M、ImageNet ResNet、DINO 的 CLS 向量，PushT 掉到 0.42 / 0.20 / 0.44（Table 2）：补丁保住了空间布局，全局向量把「T 在哪、角度多少」挤没了。这些是论文数字；你自己的缩小实验不必接近 0.90，但应能在动作条件对动作盲上看到同一方向。

DINO-WM 和第 15 课 V-JEPA 2 的亲缘值得点明：都是冻结或慢更新的视觉表征 + 少量交互数据上的动作条件预测 + 测试时在表征空间里规划。差别是 DINO-WM 用 DINOv2 静态补丁、CEM 做目标到达；V-JEPA 2-AC 用视频 JEPA 表征、能量最小。本课主实验用更小的状态预测器走同一条链，为的是 CPU 也能把「动作条件帮助规划」跑完。

### 5.7 动作盲：规划器还在搜，世界已经听不见

把 5.1 的公式里的 $a_t$ 抹掉，模型变成 $P(s_{t+1} \mid s_t)$。它仍可以给出清晰的未来，甚至在开环视频指标上好看，因为它学的是「世界自己会怎么滚」。CEM 抽 1000 条不同的动作序列，送进这个模型，得到的是 1000 份几乎相同的想象，分数也相同，尖子集合是噪声，收缩没有意义。执行时等于从初始分布里随机挑第一步。

这就是本课复现 #7 要量的差。动作条件模型不必很强，只要分岔：同一状态、两个动作，预测的滑块位置差必须大于测量噪声。动作盲模型即使一步 MSE 更低（它可以把平均运动学学得很精），规划成功率也应当掉到随机附近。第 17 课把「预测准」和「规划好」拆开，这里是拆开之后的第一张操作域对照表。

桌宠上的翻译：摄像头看得到杯子，若世界模型不吃「云台往左 5 度」或「末端往前 3 厘米」，它就无法回答「这步会不会把水送下桌」。第 7 节互动把这句话画成三条轨迹。

## 6. 源码导读

三个仓库，职责不同。gym-pusht 是你要跑的世界；dino_wm 是同环境上的前沿实现，能装再跑；daydreamer 是真机基础设施，本课读不跑真机。Visual Foresight 的 TensorFlow 仓库只作对照，不列入必读路径。

| 文件与位置 | 是哪个零件 | 带着什么问题读 |
|---|---|---|
| gym-pusht README 的 Quick start | 环境 | `gym.make("gym_pusht/PushT-v0")` 的观察有哪四种 `obs_type`？动作 `[x, y]` 的范围是什么、它是目标位置还是瞬时速度？ |
| gym-pusht 的 `PushTEnv` | 物理 | 奖励「覆盖率」在哪算？完全重合时是不是 1.0？一局何时 `terminated` / `truncated`？ |
| dino_wm `README.md` Installation / Datasets | 环境与数据 | conda 文件是不是 `environment.yaml`？OSF 数据怎么接到 `DATASET_DIR`？ |
| dino_wm `models/dino.py`、`models/visual_world_model.py`、`models/vit.py` | 冻结编码器与转移 | `feature_key` 取 patch 还是 cls？`encode` / `predict` / `rollout` 各改哪一维？第 22 课导读表更全，本课只盯规划和动作条件 |
| dino_wm `planning/cem.py`、`planning/objectives.py` | CEM 与代价 | 采样、rollout、topk、更新 mu/sigma 在哪几行？`mode: last` 比的是哪一帧？ |
| dino_wm `conf/plan_pusht.yaml`、`plan.py` | PushT 规划入口 | `n_evals`、`goal_H`、`opt_steps` 默认多大？`ckpt_base_path/outputs/{model_name}/checkpoints` 目录树缺哪一层会直接失败？ |
| daydreamer README Setup | 依赖 | 四条 pip 包是哪些？是不是 TensorFlow 而不是 PyTorch？ |
| daydreamer `embodied/agents/dreamerv2plus/train.py` | 训练与执行入口 | `--run learning` 和 `--run acting` 是否对应论文 Figure 2 的两条线程？ |
| daydreamer README 的 A1 / XArm 命令 | 真机任务名 | `a1_sim` 对 `a1_real`、`xarm_dummy` 对 `xarm_real` 差在哪？为什么 learner 可以用 GPU、actor 有时故意 `--tf.platform cpu`？ |
| [wuphilipp/robot_parts](https://github.com/wuphilipp/robot_parts) | 机械结构 | 项目页把它列为 Robot Parts。本课不组装，知道真机复现还缺硬件仓即可 |

读的顺序建议：先把 gym-pusht 的观察和动作空间打印出来（第 7 节 Step 2），再读 Visual Foresight 论文第 I、IV、V、VI 节（对应本课 5.1 到 5.4），然后读 DayDreamer 论文第 2 节和附录 D 超参，最后按 dino_wm 的 import 链从 `plan.py` 找到 CEM 实现。daydreamer 的学习线程里你会再次看见 RSSM 的先验/后验，那是第 05 课的零件，不要在本课重新推导。

Visual Foresight 仓库 [SudeepDasari/visual_foresight](https://github.com/SudeepDasari/visual_foresight) 仍公开，配套页是 [sites.google.com/view/visualforesight](https://sites.google.com/view/visualforesight)。它是 2018 年 TensorFlow 代码，依赖和机械臂接口都按当时的 Sawyer 写。本课把它标成只读：用来核对论文算法 1 的 CEM 伪代码是否还在仓库里，不用来训练。

## 7. 实验

主线在 PushT 上自训一个缩小的动作条件动力学，用 CEM 在想象里选动作，对照随机和动作盲。这是论文复现 #7 的方向性版本。DINO-WM 官方规划和 DayDreamer 仿真入口是选做，失败或跳过都写进笔记，不要用论文表格顶替。

互动实验嵌在 Step 7：同一张桌面展开三条想象轨迹，你来标哪条会把滑块送出画布。那就是桌宠「先做梦再推杯子」的低维版。

### Step 1: 独立环境并安装 PushT

```bash
python3 -m venv ~/.venv-pusht
```

```bash
source ~/.venv-pusht/bin/activate
```

先按你机器上的 PyTorch 官网指令装好 `torch`，再装环境。gym-pusht 的 README 要求 Python 3.10 档。

```bash
pip install gym-pusht
```

官方 Quick start 还需要 `gymnasium`（包依赖会带上）和渲染用的 pygame / pymunk。装完用下面这条确认入口存在。

```bash
python -c "import gymnasium as gym, gym_pusht; print('ok', gym.make('gym_pusht/PushT-v0').action_space)"
```

预期：打印 `ok` 和一个二维 `Box`，范围是 `[0, 512]`。对不上先看第 10 节「import gym_pusht 失败」。

### Step 2: 打印观察、动作、一局覆盖率

把下面存成 `probe_pusht.py`，在仓库外任意目录运行。目的是当场核对 README：动作是智能体（那个圆推杆）的目标位置，不是力矩；状态默认 5 维；像素观察是 96×96。

```python
import gymnasium as gym
import gym_pusht  # noqa: F401
import numpy as np

env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
obs, info = env.reset(seed=0)
print("obs", np.asarray(obs).shape, np.asarray(obs))
print("action_space", env.action_space)
print("info_keys", sorted(info.keys()) if isinstance(info, dict) else type(info))
total = 0.0
max_cov = 0.0
for t in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total += float(reward)
    max_cov = max(max_cov, float(reward))
    if terminated or truncated:
        print("end_at", t, "terminated", terminated, "truncated", truncated)
        break
print("sum_r", total, "max_r", max_cov)
env.close()

env_pix = gym.make("gym_pusht/PushT-v0", obs_type="pixels", render_mode="rgb_array")
pix, _ = env_pix.reset(seed=0)
print("pixels", np.asarray(pix).shape, np.asarray(pix).dtype)
env_pix.close()
```

```bash
python probe_pusht.py
```

预期：状态约 5 个数（智能体 x/y、滑块 x/y、滑块角）；像素 `(96, 96, 3)`、`uint8`；随机 50 步的 `max_r` 通常远小于 1，说明随机策略几乎推不进目标区。把 `info` 的键名抄下来，后面算覆盖率只用你实际看到的字段，没有的键不要编。

无显示器时若 pygame 报错，先执行下面这条再重跑探测脚本。

```bash
export SDL_VIDEODRIVER=dummy
```

### Step 3: 采一份随机轨迹

随机策略在 PushT 上成绩差，但覆盖足够用来学「推杆碰到滑块时滑块怎么动」。动作是目标坐标，白噪声会让推杆在画布上乱跳，接触事件偏少。下面的采集用「目标位置做布朗运动」：每步在上一个目标附近抖动，碰到边界再弹回。这和第 01 课不用白噪声采 CarRacing 是同一个理由。

把脚本存成 `collect_pusht.py`。

```python
import argparse, pathlib, numpy as np, gymnasium as gym, gym_pusht  # noqa: F401

def walk_action(prev, rng, sigma=40.0):
    nxt = prev + rng.normal(0.0, sigma, size=2)
    return np.clip(nxt, 0.0, 512.0)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--horizon", type=int, default=80)
    p.add_argument("--out", type=str, default="datasets/pusht_rand.npz")
    args = p.parse_args()
    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
    rng = np.random.default_rng(0)
    ss, aa, s2, rr = [], [], [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 10**6)))
        prev_a = np.array(obs[:2], dtype=np.float32)
        for _ in range(args.horizon):
            a = walk_action(prev_a, rng)
            nxt, r, term, trunc, _ = env.step(a)
            ss.append(np.asarray(obs, dtype=np.float32))
            aa.append(a.astype(np.float32))
            s2.append(np.asarray(nxt, dtype=np.float32))
            rr.append(np.float32(r))
            obs, prev_a = nxt, a
            if term or trunc:
                break
        if (ep + 1) % 50 == 0:
            print("ep", ep + 1, "n", len(ss))
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, s=np.stack(ss), a=np.stack(aa), s2=np.stack(s2), r=np.stack(rr))
    print("saved", args.out, "transitions", len(ss), "mean_r", float(np.mean(rr)))
    env.close()

if __name__ == "__main__":
    main()
```

```bash
python collect_pusht.py --episodes 400 --horizon 80 --out datasets/pusht_rand.npz
```

预期：打印大约三万条转移，`mean_r` 很小（随机几乎盖不住目标）。CPU 上数分钟。Mac 可完成。抽 20 条看 `s` 的滑块坐标是不是在动：若 `s2[:, 2:4] - s[:, 2:4]` 几乎全是 0，推杆没碰到滑块，把 `--episodes` 加到 800，或把布朗运动的 `sigma` 从 40 调到 25 让推杆少瞬移。

### Step 4: 训动作条件模型，再训一个动作盲对照

状态先除以尺度：位置 / 512，角 / $2\pi$，动作用同样方式归一。网络预测残差 $\Delta s$，加回 $s$ 得到 $\hat s_{t+1}$。动作盲版的前向不拼接 $a$，其余相同。把脚本存成 `train_dyn.py`。

```python
import argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SCALE = torch.tensor([512.0, 512.0, 512.0, 512.0, 2 * np.pi])

class Dyn(nn.Module):
    def __init__(self, action_cond=True, h=128):
        super().__init__()
        self.action_cond = action_cond
        inn = 7 if action_cond else 5
        self.net = nn.Sequential(nn.Linear(inn, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 5))

    def forward(self, s, a):
        sn = s / SCALE.to(s.device)
        if self.action_cond:
            an = a / 512.0
            x = torch.cat([sn, an], -1)
        else:
            x = sn
        return s + self.net(x)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="datasets/pusht_rand.npz")
    p.add_argument("--out", default="ckpts/dyn_cond.pt")
    p.add_argument("--blind", action="store_true")
    p.add_argument("--epochs", type=int, default=30)
    args = p.parse_args()
    pack = np.load(args.data)
    s = torch.tensor(pack["s"]); a = torch.tensor(pack["a"]); s2 = torch.tensor(pack["s2"])
    n = len(s); n_tr = int(0.9 * n)
    perm = torch.randperm(n)
    tr = TensorDataset(s[perm[:n_tr]], a[perm[:n_tr]], s2[perm[:n_tr]])
    va = TensorDataset(s[perm[n_tr:]], a[perm[n_tr:]], s2[perm[n_tr:]])
    model = Dyn(action_cond=not args.blind)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.SmoothL1Loss()
    loader = DataLoader(tr, batch_size=256, shuffle=True)
    for ep in range(args.epochs):
        model.train(); tot = 0.0
        for sb, ab, s2b in loader:
            pred = model(sb, ab)
            loss = loss_fn(pred / SCALE, s2b / SCALE)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(sb)
        model.eval()
        with torch.no_grad():
            vs, va_, v2 = va.tensors
            vloss = float(loss_fn(model(vs, va_) / SCALE, v2 / SCALE))
        if (ep + 1) % 5 == 0 or ep == 0:
            print("epoch", ep + 1, "train", tot / n_tr, "val", vloss)
    import pathlib; pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "action_cond": not args.blind}, args.out)
    print("saved", args.out)

if __name__ == "__main__":
    main()
```

```bash
python train_dyn.py --data datasets/pusht_rand.npz --out ckpts/dyn_cond.pt --epochs 30
```

```bash
python train_dyn.py --data datasets/pusht_rand.npz --out ckpts/dyn_blind.pt --blind --epochs 30
```

预期：动作条件的验证损失低于动作盲；两者都会降，因为滑块的「自己停着」这部分动作盲也能学。真正的判决在 Step 5 的动作对换，不在这一步的绝对值。CPU 上各几分钟。

### Step 5: 动作对换探针

同一批真实 $(s, a)$，把 $a$ 换成画布另一侧的目标位置，看 $\hat s_{t+1}$ 的滑块坐标动不动。把脚本存成 `swap_probe.py`。

```python
import numpy as np, torch
from train_dyn import Dyn, SCALE

def load(path):
    ck = torch.load(path, map_location="cpu")
    m = Dyn(action_cond=ck["action_cond"]); m.load_state_dict(ck["state_dict"]); m.eval()
    return m

pack = np.load("datasets/pusht_rand.npz")
s = torch.tensor(pack["s"][:512]); a = torch.tensor(pack["a"][:512])
a_swap = torch.stack([512.0 - a[:, 0], 512.0 - a[:, 1]], -1)
cond, blind = load("ckpts/dyn_cond.pt"), load("ckpts/dyn_blind.pt")
with torch.no_grad():
    dc = (cond(s, a) - cond(s, a_swap)).abs().mean(0)
    db = (blind(s, a) - blind(s, a_swap)).abs().mean(0)
print("cond_delta", dc.numpy())
print("blind_delta", db.numpy())
print("block_xy_cond", float(dc[2:4].mean()), "block_xy_blind", float(db[2:4].mean()))
```

```bash
python swap_probe.py
```

预期：动作条件模型在滑块 x/y（下标 2、3）上的对换差明显大于动作盲；动作盲应接近 0。若动作条件也接近 0，数据里接触太少或网络没学会，回到 Step 3 加数据，不要进入规划。这一步就是第 03 课动作对换在桌面推物上的复用。

### Step 6: CEM 规划，对照随机和动作盲

规划代价用「滑块位置贴近目标 T 的典型姿态」。gym-pusht 的目标区画在画面里，完全覆盖时奖励为 1。缩小实验不把成功定成 1.0（小模型很难），主指标用一局内最大覆盖率（也就是环境返回的 `reward` 的局内最大值）和终局覆盖率。CEM 在模型里最大化预测覆盖的代理：我们另训一个从状态读覆盖率的小头，数据里已经有 `r`。为少一个脚本，下面直接用「滑块中心靠近 (256, 256) 且角度靠近 $\pi/4$」当可复现的代理代价。这不是官方成功定义，只是让三组策略用同一把尺子。真实覆盖率仍然以环境 `reward` 为准，写进表的是环境数。

把脚本存成 `plan_cem.py`。

```python
import argparse, numpy as np, torch, gymnasium as gym, gym_pusht  # noqa: F401
from train_dyn import Dyn

def load(path):
    ck = torch.load(path, map_location="cpu")
    m = Dyn(action_cond=ck["action_cond"]); m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck["action_cond"]

def cost_fn(s, goal):
    # s: [B,H,5], goal: block_x, block_y, block_ang
    block = s[:, :, 2:5]
    w = torch.tensor([1.0, 1.0, 40.0])
    return ((block - goal) * w).pow(2).mean(-1).sum(-1)

def cem(model, s0, goal, H=8, J=128, K=16, iters=6, seed=0):
    rng = np.random.default_rng(seed)
    mu = np.array(s0[:2], dtype=np.float64)
    mus = np.repeat(mu[None, :], H, 0)
    std = np.full((H, 2), 60.0)
    s0t = torch.tensor(s0)[None, :].repeat(J, 1)
    goal_t = torch.tensor(goal)[None, :]
    best_seq, best_c = None, 1e9
    for _ in range(iters):
        seq = mus[None, :, :] + std[None, :, :] * rng.normal(size=(J, H, 2))
        seq = np.clip(seq, 0.0, 512.0)
        st = s0t.clone(); traj = []
        with torch.no_grad():
            for t in range(H):
                at = torch.tensor(seq[:, t], dtype=torch.float32)
                st = model(st, at)
                traj.append(st)
            cat = torch.stack(traj, 1)
            c = cost_fn(cat, goal_t).numpy()
        elite = seq[np.argsort(c)[:K]]
        mus, std = elite.mean(0), elite.std(0) + 1.0
        if c.min() < best_c:
            best_c, best_seq = float(c.min()), seq[int(np.argmin(c))]
    return best_seq.astype(np.float32), best_c

def rollout(env, actions=None, random=False, horizon=80, seed=0):
    obs, _ = env.reset(seed=seed)
    max_r, last_r = 0.0, 0.0
    prev = np.array(obs[:2], dtype=np.float32)
    rng = np.random.default_rng(seed + 999)
    for t in range(horizon):
        if random:
            a = np.clip(prev + rng.normal(0, 40, 2), 0, 512).astype(np.float32)
        else:
            a = actions[min(t, len(actions) - 1)]
        obs, r, term, trunc, _ = env.step(a)
        last_r, max_r, prev = float(r), max(max_r, float(r)), a
        if term or trunc:
            break
    return max_r, last_r

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cond", default="ckpts/dyn_cond.pt")
    p.add_argument("--blind", default="ckpts/dyn_blind.pt")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--goal", type=float, nargs=3, default=[256.0, 256.0, 0.785])
    args = p.parse_args()
    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
    cond, _ = load(args.cond); blind, _ = load(args.blind)
    rows = []
    for i in range(args.n):
        obs, _ = env.reset(seed=1000 + i)
        s0 = np.asarray(obs, dtype=np.float32)
        seq_c, _ = cem(cond, s0, args.goal, seed=i)
        seq_b, _ = cem(blind, s0, args.goal, seed=i)
        mc, lc = rollout(env, seq_c, seed=1000 + i)
        mb, lb = rollout(env, seq_b, seed=1000 + i)
        mr, lr = rollout(env, random=True, seed=1000 + i)
        rows.append((mc, lc, mb, lb, mr, lr))
        print(f"ep{i:02d} cond_max {mc:.3f} blind_max {mb:.3f} rand_max {mr:.3f}")
    arr = np.array(rows)
    names = ["cond_max", "cond_last", "blind_max", "blind_last", "rand_max", "rand_last"]
    for k, n in enumerate(names):
        print(n, "mean", arr[:, k].mean(), "std", arr[:, k].std())
    env.close()

if __name__ == "__main__":
    main()
```

```bash
python plan_cem.py --n 20
```

预期（方向，不是论文数字）：`cond_max` 的均值高于 `blind_max` 和 `rand_max`。动作盲应接近随机，因为 Step 5 已经显示它对动作不敏感，CEM 的尖子是噪声。20 局的标准差会不小，报均值 ± 标准差，不要用单局吹嘘。若三组打平，先看 Step 5：对换差不够大就回去加数据，而不是加 CEM 候选数。

MPC 的完整写法是每步只执行序列第一步再重规划。上面的脚本为了快，把一条 CEM 序列开环执行，属于缩小。选做：把 `rollout` 改成每 1 步或每 4 步重跑 `cem`，同一 20 个种子再比一次。开环已经能分出方向的话，重规划通常把 `cond_max` 再抬一点；动作盲仍然抬不起来。

### Step 7: 先做梦再推杯子（三条想象轨迹）

这一步是本课的互动实验。固定一个初始状态，手写三条动作序列，用动作条件模型开环想象 12 步，把滑块轨迹画出来（或打印坐标），你来标：哪条会把滑块送出画布（本课把画布边缘当成桌沿）。桌宠上的杯子就是这个滑块。

把脚本存成 `dream_three.py`。

```python
import numpy as np, torch, gymnasium as gym, gym_pusht  # noqa: F401
from train_dyn import Dyn

ck = torch.load("ckpts/dyn_cond.pt", map_location="cpu")
model = Dyn(action_cond=True); model.load_state_dict(ck["state_dict"]); model.eval()
env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="rgb_array")
obs, _ = env.reset(seed=7)
s0 = torch.tensor(np.asarray(obs, dtype=np.float32))
H = 12
agent = s0[:2].numpy()

def seq_towards(target, h=H):
    return np.linspace(agent, target, h + 1)[1:].astype(np.float32)

dreams = {
    "to_goal": seq_towards(np.array([256.0, 256.0])),
    "to_edge": seq_towards(np.array([500.0, 500.0])),
    "stay": np.repeat(agent[None, :], H, 0).astype(np.float32),
}

def imagine(seq):
    st = s0.clone()[None, :]
    traj = [st[0, 2:4].numpy()]
    with torch.no_grad():
        for t in range(len(seq)):
            at = torch.tensor(seq[t])[None, :]
            st = model(st, at)
            traj.append(st[0, 2:4].numpy())
    return np.stack(traj)

print("init_block", s0[2:4].numpy(), "init_agent", agent)
for name, seq in dreams.items():
    xy = imagine(seq)
    out = np.any((xy < 8) | (xy > 504), axis=1)
    print(name, "last_xy", xy[-1], "hits_edge", bool(out.any()), "path", np.round(xy, 1))
env.close()
```

```bash
python dream_three.py
```

你要交的不是脚本会不会跑，是三行标注：

| 轨迹 | 想象里滑块去哪 | 会不会推出画布（桌沿） | 若在真桌子上，这步做不做 |
|---|---|---|---|
| to_goal | （抄 last_xy） | 是 / 否 |  |
| to_edge |  |  |  |
| stay |  |  |  |

合格标准：三条路径在纸面上能分开；`to_edge` 比 `stay` 更靠近边缘或已经触发 `hits_edge`；你能用 5.1 的公式指出打分函数 $c$ 在桌宠里应该加一项「距离桌沿小于 5 cm 则代价为无穷」。第 27 课会把这一项做成真的安全过滤器。模型若把三条路径想成同一条，回到 Step 5，规划器没有分岔可标。

有显示器时可以把 `render_mode` 改成 `human`，把三条序列真的在环境里各执行一遍，对比「梦里的滑块」和「真 PushT 的滑块」。梦里出界、真环境没出界，是模型误差，不是标注错误；把这种不一致记一条，第 17 课的口径正好用上。

### Step 8: 官方 DINO-WM 在 PushT 上规划（选做）

有 24GB 卡、conda、且官方数据或 checkpoint 能下时再做。命令以仓库 README 现文为准，下面抄的是 2026-08 核对过的安装与规划入口。

```bash
git clone https://github.com/gaoyuezhou/dino_wm.git
```

```bash
cd dino_wm
```

```bash
conda env create -f environment.yaml
```

```bash
conda activate dino_wm
```

README 的训练示例是 PointMaze，不是 PushT。先读 `conf/`，确认 PushT 的 `env` 取值再训。若只做规划、且 README 的 Pre-trained Model Checkpoints 能下到 PushT 权重：

```bash
python plan.py --config-name plan_pusht.yaml model_name=pusht
```

README 里还有更通用的写法，例如 `python plan.py model_name=<model_name> n_evals=5 planner=cem goal_H=5 goal_source='random_state' planner.opt_steps=30`。`<model_name>` 要换成权重目录名，权重路径由 `conf/plan.yaml` 的 `ckpt_base_path` 指定。OSF 数据页打不开、MuJoCo 在 Mac 上装不上、DINOv2 权重下不动，都算选做失败，在笔记里写原因，主实验仍可验收。

不要把论文 Table 1 的 0.90 抄进你的「本课结果」。你跑出来的 `n_evals` 成功次数才是你的数。

### Step 9: 精读 DayDreamer，不跑四台真机

另开目录克隆，只读和最多跑仿真任务名。

```bash
git clone https://github.com/danijar/daydreamer.git
```

README Setup 写明的依赖是这一条：

```bash
pip install tensorflow tensorflow_probability ruamel.yaml cloudpickle
```

A1 的仿真学习线程，README 原文是（先清日志再开训，两条分开跑）：

```bash
rm -rf ~/logdir/run1
```

```bash
CUDA_VISIBLE_DEVICES=0 python embodied/agents/dreamerv2plus/train.py --configs a1 --task a1_sim --run learning --tf.platform gpu --logdir ~/logdir/run1
```

对应的真机执行线程（本课禁止把它接到真 A1 上冒充复现）：

```bash
CUDA_VISIBLE_DEVICES=1 python embodied/agents/dreamerv2plus/train.py --configs a1 --task a1_real --run acting --tf.platform gpu --env.kbreset True --imag_horizon 1 --replay_chunk 8 --replay_fixed.minlen 32 --imag_horizon 1 --logdir ~/logdir/run1
```

XArm 的 README 把 learner 指到 `--task xarm_dummy`、actor 指到 `--task xarm_real`，actor 还写了 `--tf.platform cpu --tf.jit False`。精读问题：为什么真机 actor 有时故意不用 GPU？提示在 5.5 节的延迟。把 `learning` / `acting`、`a1_sim` / `a1_real`、`xarm_dummy` / `xarm_real` 六对词写进笔记，并在笔记第一行写：「本课未复现 DayDreamer 的四台真机结果」。

`a1_sim` 若依赖内部仿真封装跑不起来，停。能跑出日志也不许改口成「我复现了 1 小时走路」。

## 8. 配置与预算

| 项目 | 缩小主实验（必做） | DINO-WM 选做 | DayDreamer 真机（本课不做） |
|---|---|---|---|
| 环境 | gym-pusht PushT-v0，状态 5 维 | 官方 PushT，224×224 图 + DINOv2 补丁 | A1 / UR5 / XArm / Sphero |
| 数据 | 随机布朗动作 400 局 × 80 步，约 3e4 转移 | README 指向的 OSF 离线集 | 真机在线 replay，论文里数小时到十小时 |
| 模型 | 两层 128 宽 MLP 残差动力学 ± 动作盲对照 | 冻结 DINOv2 + ViT 转移 | DreamerV2 RSSM + actor-critic |
| 规划 | CEM：$H=8, J=128, K=16, I=6$，20 局 | `plan_pusht.yaml`，`planner=cem` | 上场不跑 CEM，用 actor；想象 $H=16$ 在训练里 |
| 硬件 | Mac / CPU 数十分钟；有卡更快 | 单张 24GB，按官方配置 | 真机 + 工作站，本课无 |
| 卡时 | 0（CPU 即可） | 训一次按仓库默认，数小时到一天量级 | 论文：走路约 1 小时墙钟，抓放 8-10 小时 |
| 主指标 | 20 局最大覆盖率均值，动作条件 > 动作盲与随机 | 你自己的 `n_evals` 成功率 | 不申报 |

超参不要抄论文当自己的。PlaNet 的 1000 候选、DINO-WM 的 `opt_steps=30`、DayDreamer 的 $H=16$ 是他们的账单。你的缩小 CEM 用 128 候选是为了 CPU 上 20 局能在一小时内结束。加候选、改成逐步重规划，只作为第 11 节改造，并写清预算。

检查点：`datasets/pusht_rand.npz`、`ckpts/dyn_cond.pt`、`ckpts/dyn_blind.pt`、Step 6 的 20 行日志、Step 7 的三行标注表、DayDreamer 精读笔记首页那句未复现声明。缺任何一项，第 9 节不能打勾。

## 9. 验收

量化线（主实验，方向性复现 #7）：

- [ ] Step 2 打印的动作空间是二维、范围 `[0, 512]`，像素观察形状是 `(96, 96, 3)`
- [ ] Step 5 动作条件模型在滑块 x/y 上的对换差大于动作盲至少一个量级，或至少稳定地更大
- [ ] Step 6 的 20 局：动作条件的最大覆盖率均值高于动作盲和随机；报均值 ± 标准差，N=20
- [ ] Step 7 三条轨迹的坐标能分开，并完成「会不会推出画布」标注
- [ ] 书面一句：DayDreamer 四台真机的数字本课没有复现；DINO-WM 的 0.90 若未自己跑通，不得写入本课结果

可视检查：

- 打开一条 Step 3 轨迹的状态序列，滑块坐标必须有非零变化（推杆真的碰到过物体）
- Step 7 打印的 `to_goal` / `to_edge` / `stay` 三条 `path` 不是同一条直线
- 若做了选做 Step 8，保存至少一张官方规划的目标图与执行后画面，并记下 `n_evals`

口头能答：

1. Visual Foresight 的预测器训练时有没有任务奖励？测试时目标从哪来？
2. CEM 在 Visual Foresight、PlaNet、DINO-WM 里分别给什么打分？
3. DayDreamer 为什么把 actor 和 learner 拆成两条线程？
4. 动作盲的一步预测损失更低时，规划为什么仍可能更差？

桌宠用途写进笔记：这是「想完再动」的第一块肌肉。当前还没有真机，但代价函数里已经能加「出界则拒绝」。下一课模仿策略会很快做出像样的挥手；「别把水推倒」仍然走本课这条路。

## 10. 排错

| 症状 | 原因 | 怎么验证 | 怎么修 |
|---|---|---|---|
| `pip install gym-pusht` 后 `import gym_pusht` 失败 | 没进独立 venv，或 Python 版本低于仓库说明 | `python -V`；`pip show gym-pusht` | 按 README 用 3.10 档重建 venv，不要和本课程 Dreamer 环境混装 |
| pygame / SDL 报错，无窗口 | 无显示器或 Mac 权限 | 错误栈里是否出现 `video system` | `export SDL_VIDEODRIVER=dummy`，`render_mode="rgb_array"` |
| 采集的滑块坐标几乎不变 | 推杆目标瞬移，接触太少 | 看 `s2[:,2:4]-s[:,2:4]` 的均值 | 降低布朗 `sigma`，加长 `horizon`，加 `episodes` |
| 动作条件与动作盲的验证损失差不多 | 接触事件仍少，或网络过弱 | Step 5 对换差 | 先加数据，再加宽到 256；不要先加 CEM 候选 |
| Step 5 动作条件对换差也接近 0 | `action_cond` 存盘错了，或前向没拼动作 | 打印 `ck["action_cond"]`；临时把 `a` 乘 0 看预测变不变 | 重训；确认 `Dyn.forward` 在 `action_cond=True` 时 `cat` 了动作 |
| 三组规划分数打平 | 模型还不会分岔，或代理代价和覆盖率无关 | Step 5；把代理代价改成「只罚滑块到 (256,256) 的距离」再比 | 对换差不够就不要调规划器。代理代价只影响绝对数，不应抹平三组排序 |
| CEM 每局都给出几乎相同的动作 | `std` 初始化太小，或 `J` 太小抽不开 | 打印第 0 轮候选的 `seq.std()` | 把初始 `std` 从 60 加大；检查 `np.clip` 有没有把所有候选夹死 |
| 开环 CEM 比随机还差 | 模型在接触附近外推错，策略专走模型的幻觉 | 梦里滑块飞出、真环境原地 | 改成每 4 步重规划；或把规划视野从 8 降到 4。这是 5.4 节类比失效处 |
| `plan.py --config-name plan_pusht.yaml` 找不到权重 | `ckpt_base_path` 或 `model_name` 和磁盘对不上 | 读 `conf/plan.yaml` 和 README Checkpoints 节 | 按官方改路径；不要把论文成功率填进日志 |
| OSF 数据页打不开 | 链接带 `view_only`，权限或网络 | 浏览器能否下载 | 放弃选做，主实验仍可验收 |
| conda 装 dino_wm 要 MuJoCo 2.1、Mac 没有对应包 | README 的 wget 针对 Linux x86_64 | 安装脚本是否写 linux | 只跑 gym-pusht 主实验；Linux + 24GB 再回来 |
| DayDreamer `a1_sim` 立刻缺模块 | TensorFlow 栈和第 06 课 PyTorch 冲突 | `python -c "import tensorflow"` | 另建环境；缺仿真封装就停在精读 |
| 把论文 1 小时走路写进本课结果 | 档位混淆 | 笔记首页有没有未复现声明 | 删掉，改回「论文报告」 |

## 11. 前沿与改造

前沿怎么做。同一句「看见未来再动手」，2024-2026 年公开系统把它拆成三条仍然活着的工程路线。第一条是测试时规划：DINO-WM 把 Visual Foresight 的 CEM 从像素搬到冻结 DINOv2 补丁上，离线数据训动力学，目标图一给就能搜；第 15 课的 V-JEPA 2-AC 用视频表征加能量最小做同类事，规划一个动作在论文对照里大约 16 秒，而显式生成视频的对照要到分钟级。第二条是在线想象训练上真机：DayDreamer 2022 年把 DreamerV2 直接接到四台机器人，后续的 DreamerV3（第 06 课）把稳定性零件补齐，但公开的「四台真机、同一超参、不经仿真」仍然以 DayDreamer 那一篇为可核对的完整报告。第三条是不建动力学、直接模仿：Diffusion Policy、ACT 会在第 25 课上场，桌面上的推 T、抓放，先模仿往往比先训世界模型更快像样。三条路在桌宠上会并存：模仿给出手，世界模型给「这步会不会把杯子送下桌」的否决权，语言条件的 VLA（第 26 课）给「把笔递过来」这种指令接口。

我们差在哪。规模差：DINO-WM 的六套环境、DINOv2 编码、官方离线集；DayDreamer 的真机小时数、多模态融合、异步 actor/learner。这些是钱、机器和工时。机制差：本课主实验已经具备同一条链（动作条件预测、CEM、动作盲对照、目标代价可替换）。缺的是像素级或补丁级动力学、逐步重规划的完整 MPC、以及真正的桌沿安全项。机制缺的部分可以在本课改造清单里补一块；真机小时数本课不补，声明即可。

动手改造清单（均在 gym-pusht 缩小设置上，CPU 或单卡）：

1. 把开环序列改成每步重规划。改 `plan_cem.py` 的 `rollout`：每执行 1 步，用新状态再跑 CEM，只执行第一步。预算：20 局，CPU 上可能从数分钟变成数十分钟（每步 6 轮 × 128 条 × 8 步前向）。预期：动作条件的最大覆盖率再升一截；动作盲仍然贴随机。失败：三组仍打平，且 Step 5 对换差已经足够，这时把逐步重规划的候选轨迹存下来看是不是模型在接触点外推炸了。
2. 代价换成环境覆盖率头。用 `datasets/pusht_rand.npz` 里的 `r` 训一个 `r_hat(s)`，CEM 最大化 $\sum \hat r$。预算：多训一个小 MLP，一小时内。预期：排序仍是动作条件更好，绝对覆盖率可能高于「靠近 (256,256)」这种代理。失败：覆盖率头对所有状态输出接近 0（随机数据太稀），先做成功轨迹过采样再训头，不要立刻换规划器。
3. 像素版动力学。`obs_type="pixels"`，小卷积编码成 64 维，拼动作，预测下一潜向量，用预测潜向量与目标帧潜向量的 MSE 当 CEM 代价（DINO-WM 的缩小版）。预算：采集改存图像，显存 8GB 够，CPU 会慢。预期：动作对换在潜空间分岔；规划方向与状态版一致但更抖。失败：重建好看、对换不分岔，动作没进编码器。
4. 桌沿拒绝。在 Step 7 的打分里加一项：预测滑块坐标进入宽度 8 像素的边缘带，代价设为一个很大的数。预算：改 `cost_fn`，20 局。预期：`to_edge` 这类序列不再被选中，动作条件组把滑块推出画布的次数下降。失败：模型在边缘附近的预测本身不准，拒绝项拦不住，把这些失败帧留下给第 27 课。

顺手复现的映射：

| 论文结论 | 缩小版对应 | 预期 |
|---|---|---|
| 动作条件视频/特征预测才能做视觉 MPC（Visual Foresight；DINO-WM Table 2 全局向量变差） | Step 5 对换 + Step 6 动作盲对照 | 能复现方向：听不见动作，规划掉到随机附近 |
| 同一模型、换代价就能换任务（Visual Foresight 三种 $c$） | 改造 2 与改造 4 | 能复现方向：换 $c$ 不换动力学，选出的动作风格变 |
| 规划时不必解码像素（DINO-WM；DayDreamer 行为学习阶段） | 主实验在 5 维状态上规划 | 能复现方向：没有一张重建图也能选动作 |
| Dreamer 在真机上 1 小时走路、8 小时抓放（DayDreamer 第 3 节） | 无 | 不能复现，本课无真机 |

## 12. 论文与延伸

1. Visual Foresight（Ebert, Finn, Dasari, Xie, Lee, Levine，[arXiv:1812.00568](https://arxiv.org/abs/1812.00568)）。本课第一条主线。带着三个问题读：第 IV 节 DNA 和 SNA 差在哪一层跳连，对应他们 Table I 那组「一动一静」任务为什么必须看见被遮住的物体？第 V 节三种代价各解决哪种目标，整图 $\ell_2$ 失败的例子是不是机械臂占画面？算法 1 的 CEM 与第 05 课 PlaNet 的四拍能否逐项对齐，采样时如何保证动作仍在训练分布里？
2. Deep Visual Foresight for Planning Robot Motion（Finn, Levine，ICRA 2017，[arXiv:1610.00696](https://arxiv.org/abs/1610.00696)）。2018 年长文的原点。带着问题读：动作条件视频预测怎样直接接到 MPC、训练时为什么可以完全没有任务标签？它相对 2016 年 Finn、Goodfellow、Levine 的视频预测论文，控制这一步多了什么？
3. DayDreamer（Wu, Escontrela, Hafner, Goldberg, Abbeel，CoRL 2022，[arXiv:2206.14176](https://arxiv.org/abs/2206.14176)）。本课第二条主线，配套页 [danijar.com/daydreamer](https://danijar.com/daydreamer)，代码 [danijar/daydreamer](https://github.com/danijar/daydreamer)。带着问题读：四台机器的观察、动作、奖励结构各是什么，同一套超参具体固定了哪些量？actor/learner 分线程解决的是训练稳定性还是控制延迟？附录 B 解码出来的想象里物体会变色，这件事支持还是削弱「潜空间规划依赖像素保真」？读完在笔记第一行写未复现声明。
4. DINO-WM（Zhou, Pan, LeCun, Pinto，[arXiv:2411.04983](https://arxiv.org/abs/2411.04983)）。零样本规划的当代写法，项目页 [dino-wm.github.io](https://dino-wm.github.io/)，代码 [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm)。带着问题读：冻结 DINOv2 补丁、转移损失不走像素重建，规划代价为什么能直接用特征 MSE？Table 1 的 PushT 0.90 对照的是哪些离线世界模型、它们训的时候有没有奖励？Table 2 换编码器之后谁在操作任务上崩了，崩的原因是空间信息还是预训练领域？
5. PlaNet（Hafner 等，[arXiv:1811.04551](https://arxiv.org/abs/1811.04551)，第 05 课已读）。带着新问题回读：CEM 的 $H=12,I=10,J=1000,K=100$ 打的是奖励头，Visual Foresight 打的是像素运动，DINO-WM 打的是目标特征，三套代价能否接在同一个搜索循环上？为什么 DayDreamer 在真机上反而弃用每步 CEM？
6. DreamerV2（Hafner 等，[arXiv:2010.02193](https://arxiv.org/abs/2010.02193)）。DayDreamer 论文写明自己建在这份实现上。带着问题读：离散潜变量和 actor-critic 在想象里训练，哪些超参被 DayDreamer 原样拿到四台真机上？和第 06 课 DreamerV3 比，少了哪些稳定性零件、真机实验还能否跑起来？

读这六篇时记住档位。1、2、3、4 是本课正文引用过、用 WebFetch 核对过摘要与方法节的文献；5、6 是课程里已经出现、本课用来对照搜索引擎和真机基础设施的。站点 [sites.google.com/view/visualforesight](https://sites.google.com/view/visualforesight) 与 DayDreamer、DINO-WM 项目页可以当视频材料，不代替论文里的表。

到这一课，主干循环第一次在桌面推物上转完一圈：观察被压成状态，状态按动作走到下一步，多条未来被展开、打分、选出一步。桌宠因此有了拒绝的权利。下一步会碰到另一种诱惑：不建动力学，直接模仿专家的手。第 25 课用 ACT 与 Diffusion Policy 问这件事能走多远，以及「别把水推倒」为什么仍然需要你刚刚训出来的这个会做梦的模型。



