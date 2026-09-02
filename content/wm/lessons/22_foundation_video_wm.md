---
id: 22_foundation_video_wm
title: "视频基础模型什么时候才算世界模型"
summary: "Cosmos、Genie、Sora 类都会往后播。动作起不起作用？能规划吗？"
unit: spatial
play_tools: []
checkpoints:
  - "动作对换是否分岔的记录。"
  - "一张前沿系统评测表。"
---

# 第 22 课：视频基础模型什么时候才算世界模型

> 类型：实战（DINO-WM 的动作对换与缩小规划）+ 只讲（Cosmos-Predict2.5、UniSim、GAIA-1、Genie 3）<br>
> 建议周期：2-3 天<br>
> 硬件：单张 24GB 卡跑 DINO-WM 的 PushT 规划与动作对换；Mac / 纯 CPU 可完成阅读、三问打分、1xgpt 数据检查。Hugging Face 卡片写明 Cosmos-Predict2.5-2B 的 Video2World 约需 32.54GB 显存，主线 24GB 不够，本课对 Cosmos 只读文档与官方 demo<br>
> 锚定仓库：[gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm)（主实验），对照文档 [nvidia-cosmos/cosmos-predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5)，动作条件检查用 [1x-technologies/1xgpt](https://github.com/1x-technologies/1xgpt)<br>
> 产物：PushT 上的动作对换记录、一次缩小零样本规划（只看方向）、一张开源三档 × 三分评测表

## 1. 这一课做什么

第六幕要补两样东西：离开视野的物体还在不在，桌上的东西是不是分开的。
第 20、21 课处理的是持久的 3D/4D 状态：杯子被手挡住两秒，状态里杯子还在不在。
那种状态会更新，但还不是动作条件动力学 $P(s_{t+1}\mid s_t, a_t)$。
本课加的零件是资格审查：同样自称世界模型的大视频模型，按
[第 12 课](12_frontier_landscape.md) 的三问筛选器，谁过关。

主干循环还是这一条：

```text
观察 先压成状态 再按动作预测下一状态 然后展开多条未来 给未来打分 最后选动作
```

基础视频模型通常停在前半段。它们在互联网或机器人视频上预训练
$P(\text{future video}\mid\text{past}, \text{可选文本或动作})$，
画面可以极真，动作端口却经常缺席、被文本代替，或慢到没法在测试时搜动作。
桌宠这边桌面视频很多，真机动作标签很少。这一课要回答：能不能从视频里挤出动力学，
以及挤出来的东西能不能拿去选动作。

主实验锚定 DINO-WM（Zhou, Pan, LeCun, Pinto，arXiv:2411.04983）。
它冻住 DINOv2 的空间 patch 特征，只在特征上训一个动作条件的 ViT 预测器，
测试时用 CEM（交叉熵方法：一种不求导、靠采样再保留精英的轨迹优化）在想象里搜动作。
论文宣称在六个环境上零样本规划，不需要专家演示、奖励模型或逆动力学。
本课能跑的部分算实战：加载官方检查点，在 PushT 上做动作对换，再跑一次缩小规划，
只看方向（T 块是否往目标方向走），不声称复现论文表 1 的 0.90 成功率。
论文复现 #7 写在第 24 课，那里才把规划做成对照实验。

对照三家都标成只讲。NVIDIA Cosmos-Predict2.5 代码和权重大多公开，但 2B 的
Video2World 官方写明约 32.54GB 显存，主线 24GB 卡跑不起来，不假装训过基础模型。
UniSim（Yang et al.，arXiv:2310.06114）和 GAIA-1（Hu et al.，arXiv:2309.17080）
无本课可跑的官方权重。Genie 3 无权重，不能练。open-oasis 的崩坏帧数
[第 12 课](12_frontier_landscape.md) 已经量过，本课只把它当作对照行，不重复测量。

术语速查：

| 术语 | 一句人话 |
|---|---|
| 视频基础模型 | 在大规模视频上预训练、用来往后播画面的大模型；条件可以是文本、首帧或动作 |
| 三问筛选器 | 第 12 课的资格审查：动作起作用吗、离开视野的东西还在吗、有没有人用它选过动作 |
| 开源三档 | 可跑（代码权重齐）/ 仅权重（有推理脚本、无训练代码）/ 纯 demo（连权重都没有） |
| 三分评测 | 预测准、生成真、规划好，三根互相独立的尺子，第 12 课立观念，[第 17 课](17_evaluating_world_models.md) 写成协议 |
| DINOv2 patch | 把一张图切成格子，每个格子一个向量；DINO-WM 默认 `dinov2_vits14` 的 `x_norm_patchtokens` |
| 动作对换 | 钉死当前观察，只换动作，看预测分不分岔；[第 03 课](03_mdn_rnn_action_conditioned.md) 用它揪过动作盲 |
| CEM | 交叉熵方法：采样一群动作序列，留下损失最低的一批，用它们的均值方差再采样 |
| 零样本规划 | 训练时不看任务奖励，测试时给一张目标图，在模型里搜动作去够这张图 |
| PushT | 俯视桌面上用圆形推杆把 T 形滑块推到目标位姿，DINO-WM 的主操作环境之一 |
| 只讲 | 本课不跑权重、不训练；读论文和仓库文档，把宣称和可验证事实分开写 |

## 2. 问题

同样叫世界模型的系统，输入输出可以完全不同。Cosmos-Predict2.5 的基础检查点吃文本加图像或视频，
吐未来视频；DINO-WM 吃观察和低层动作，吐下一时刻的 patch 特征；Genie 3 是订阅网页上的
交互演示。名称本身不能告诉你动作起不起作用、状态能保持多久、能不能拿去选动作。

本课解决三件具体的事：

1. 用第 12 课已经立住的三问和三分评测，给 Cosmos、DINO-WM、open-oasis、Genie 3
   各填一格。open-oasis 那一行的预测准，引用你第 12 课量过的崩坏帧数，不要再跑一遍。
   Genie 3 官方博客里的“数分钟一致性”是宣称，写进表时必须标成宣称，不能写成你测过。
2. 在一个真有动作标签的操作环境里，证明“预训练视觉特征上的动力学”会不会听动作。
   工具还是动作对换：同一段 PushT 观察，滑杆改推的方向，预测必须分岔。
   不分岔，后面的规划就是在一个动作盲模型上搜随机数。
3. 尝试一次零样本规划，只看方向。论文表 1 写 PushT 成功率 0.90（50 个起终点，
   目标保证 25 步内可达）。你用官方检查点、把 `n_evals` 降到 5，看 T 块是否朝目标挪。
   成功方向成立即可，样本太少，数字不构成复现。

一条界限先划清：基础模型的生成真，和决策意义上的世界模型，经常不是同一笔买卖。
自动驾驶公司要可控的罕见场景视频，生成真就是产品；桌宠要先在脑子里推一把杯子再决定
真不真推，缺了动作条件和可规划性，视频再真也只是画师。本课不贬低画师，只把坐标标对。

## 3. 准备

- 一张 NVIDIA 卡的 Linux 机器。DINO-WM 的 `environment.yaml` 锁的是 Python 3.9、
  `torch==2.3.0` 和一组 CUDA 12.1 的 NVIDIA 轮子，按官方 conda 环境走最省事。
  24GB 跑 `dinov2_vits14` 加 6 层 ViT 预测器非常宽裕。Mac / 纯 CPU：官方环境没有
  MPS 路径，动手的规划与对换做不了；第 5、6、11、12 节和第 7 节的评测表全部可做。
- 磁盘留 20GB 以上。OSF 数据包含 `pusht_noise`、`point_maze`、`wall_single` 和
  可变形物体，本课主线只解压 `pusht_noise` 和 PushT 检查点即可。
- [DINO-WM 项目页](https://dino-wm.github.io/) 先看一遍 PushT 的想象 rollout。
  上排真环境、下排模型想象，右边是目标图。先建立手感再读代码。
- 一个 Weights & Biases 账号，或接受离线模式。`train.py` 和 `plan.py` 都会
  `wandb.init`，不登录又不上离线会在第一步卡住。
- 第 12 课的前沿地图笔记。本课的验收表要引用你当时给 open-oasis 填的崩坏帧数。
- 选做：Hugging Face 账号，用来看 Cosmos-Predict2.5 的模型卡和 1xgpt 的 token 数据。
  不要求下载 Cosmos 权重。

## 4. 学习目标

1. 用三问筛选器给 Cosmos-Predict2.5、DINO-WM、open-oasis、Genie 3 各答一遍，
   并指出每一问的证据是论文、仓库代码、你自己的实验，还是官方宣称；
2. 画出 DINO-WM 的数据流：冻住的 DINOv2 patch、动作/本体感觉编码器、因果 ViT
   预测器、可选解码器，并说出训练损失落在哪一段、规划损失落在哪一段；
3. 解释为什么解码器对规划是可选的，以及为什么论文消融里把重建损失反传到预测器
   会伤规划；
4. 在 PushT 上完成动作对换：同一观察、两个相反推的方向，预测特征距离必须明显
   大于同一动作的重复前向；
5. 跑一次缩小 CEM 规划（`n_evals=5`），对照目标图判断 T 块是否朝正确方向移动，
   并写明这不是论文数字的复现；
6. 交出一张开源三档 × 三分评测表，Genie 3 的“数分钟一致性”必须标成宣称。

## 5. 原理

四个机制。前两个回答“什么时候才算”，后两个回答“DINO-WM 怎么挤出动力学、怎么选动作”。
每个走完直觉、机制、数学、代码落点、验证。

### 5.1 基础视频模型的资格：三问还在，尺子不换

第 12 课已经证明：名称通胀要用三问拆开。本课把同一把尺子挪到“基础模型”这一档。
基础模型的宣传册喜欢写物理一致性、数分钟记忆、世界基础模型平台。这些词可以当线索，
不能当证据。

三问落到本课四个系统上，答案形态会变，问题本身不变：

1. 动作起作用吗？条件里必须有逐步动作，且同一观察换动作后预测分岔。
   Cosmos-Predict2.5 的基础检查点条件是文本加图像或视频，动作端口在后训练的
   `robot/action-cond` 分支里才出现。DINO-WM 的每个 patch 都拼了动作嵌入，
   第一问用对换实验当场验。Genie 3 的交互演示看起来过第一问，外界没有权重可复核。
   open-oasis 第 12 课已经确认 `generate.py` 每步吃动作切片。
2. 离开视野的东西还在吗？DINO-WM 的上下文是 `num_hist=3` 帧（再乘 `frameskip=5`），
   记忆半径按帧数计很短，它靠的是 DINOv2 空间特征里已经编码好的物体布局，不是长视频
   记忆。Cosmos 官方有滑窗自回归加长视频的文档，那是生成长度，不是你测过的物体恒常。
   Genie 3 宣称数分钟一致，本课标宣称。open-oasis 你已经量过：32 帧窗口外结构保不住。
3. 有没有人用它选过动作，并且回到真环境验收过？DINO-WM 论文表 1 给出了六个环境的
   规划成功率或 Chamfer 距离，代码在 `plan.py`。Cosmos 平台把裁判交给后训练和下游任务，
   基础模型本身不直接规划。Genie 3 官方定位是智能体训练场，公开材料里没有可复核的
   决策验收。open-oasis 没有公开决策验收。

狭义定义没变。世界模型学的是 $P(o_{t+1}\mid o_{\le t}, a_{\le t})$，
动作必须出现在条件里并且改变输出。基础模型多出来的东西是规模和条件种类：文本、多视角、
控制图。规模能把“生成真”买上去，买不来第一问。缺动作端口的系统，在决策地图上仍在场外。

验证就是第 9 节那张表。四行都填完，且每一格能指出证据来源，这一节才算过。

### 5.2 DINO-WM：冻住眼睛，只在格子上预测下一刻

像素世界模型有两笔贵账。一笔是重建：每一步都要把世界画清楚，容量花在纹理和光照上。
另一笔是从零学眼睛：每个新环境都重新训编码器，桌面数据少的时候眼睛先塌。
DINO-WM 的赌法是：眼睛已经在互联网图像上训好了，世界模型只负责“这些格子下一步往哪走”。

具体拆件。观察模型是冻住的 DINOv2。默认配置 `conf/encoder/dino.yaml` 写着
`name: "dinov2_vits14"`、`feature_key: "x_norm_patchtokens"`。
一张 224×224 的图，patch 大小 14，得到 $N=16\times 16=256$ 个格子，
每个格子 $E=384$ 维。对照配置 `dino_cls.yaml` 改用 `x_norm_clstoken`，
整张图收成一个向量，论文表 2 里这条叫 DINO CLS，PushT 成功率从 0.90 掉到 0.44。
空间格子不是装饰，操作任务靠它记住 T 块和推杆各在哪。

转移模型是 `models/vit.py::ViTPredictor`：去掉了 ViT 的分词层，直接吃 patch 序列，
6 层、16 头，带帧级因果掩码。`generate_mask_matrix` 按帧而不是按 token 遮挡未来，
同一帧里的 256 个格子可以互看，下一帧不能偷看。动作经 MLP（`models/proprio.py` 那套
编码器，配置里 `action_emb_dim: 10`）映射后，按 `concat_dim: 1` 拼到每个格子的特征维上。
`num_hist: 3`、`num_pred: 1`、`frameskip: 5`：模型一次看 3 个抽过帧的时间步，预测下一步。

训练是教师强制，损失只在特征里算：

$$
\mathcal{L}_{\mathrm{pred}}=\bigl\|p_\theta\bigl(\mathrm{enc}(o_{t-H:t}),\phi(a_{t-H:t})\bigr)-\mathrm{enc}(o_{t+1})\bigr\|^2
$$

编码器不更新（`model.train_encoder: False`）。解码器是一叠转置卷积，损失独立，
规划时可以关掉。论文附录写：把重建损失反传到预测器会伤规划。源码里对应
`models/visual_world_model.py::VWorldModel.forward`：对 `z_pred` 做 `decode` 之前
先 `.detach()`，重建梯度到不了预测器。

和前几课的位置关系要说清。它和 [第 15 课](15_vjepa2_in_practice.md) 的 V-JEPA 2 一样
不把像素当预测目标，但 V-JEPA 2 的公开权重主线仍是视频表征，动作条件在 AC 接口里；
DINO-WM 从第一天就把低层动作拼进每个 patch。它和 [第 09 课](09_iris_token_world_model.md)
的 IRIS 都用 Transformer 做动力学，IRIS 在离散 token 上自回归，DINO-WM 在连续 patch
上整帧预测。它和 [第 06 课](06_dreamerv3_imagination.md) 的 DreamerV3 都做想象，
Dreamer 在线采数据并预测奖励，DINO-WM 只用离线轨迹、训练阶段不看任务。

代码落点：`models/dino.py::DinoV2Encoder` 调 `torch.hub.load("facebookresearch/dinov2", name)`；
`VWorldModel.encode` 负责把视觉、本体感觉、动作拼成 $z$；
`VWorldModel.predict` 把时间维和格子维摊平后送进 ViT；
`VWorldModel.rollout` 是规划时的多步想象，每步用 `replace_actions_from_z` 把新动作写回。

验证分两层。训练对不对：`z_loss` 下降，解码器（若开着）能看出 T 块和推杆。
动力学对不对：第 7 节的动作对换。损失下降不能代替对换，第 03 课已经演示过惯性外推
也能把平均误差做低。

### 5.3 动作对换：滑杆改方向，预测必须分岔

PushT 的动作是推杆在平面上的位移。同一帧俯视图，往左推和往右推，T 块之后的位姿
应当不同。如果模型忽略动作，两个预测会几乎重合，规划器无论怎么搜都走同一条梦。

对换协议和 [第 03 课](03_mdn_rnn_action_conditioned.md) 相同，只是状态从 32 维 $z$
换成了 $N\times E$ 的 patch 图：

1. 钉死观察 $o_{t-H:t}$ 和历史动作；
2. 只改当前及以后的动作，做成一对方向相反的序列 $a$ 与 $a'$；
3. 用 `VWorldModel.rollout` 得到 $\hat{z}(a)$ 与 $\hat{z}(a')$；
4. 量末帧视觉特征的均方距离 $d(a,a')$。同一条动作跑两次的距离应当接近 0
   （`eval()` 下无随机性），相反方向的距离应当明显更大。

互动版是把方向做成滑杆。角度 $\theta\in[0,2\pi)$ 对应一个单位位移，
连续改 $\theta$，看解码出来的推杆/T 块是否跟着转，以及 $d(\theta, \theta+\pi)$
是否稳定地大于 $d(\theta, \theta)$。分岔当场打分：肉眼能分清两个方向记 1，
分不清记 0。分数是本课“第一问过关”的证据，不是论文指标。

类比：这像在沙盘上把手指往两个方向拨，看模型的沙有没有跟着走。
类比失效处：沙盘是真物理，误差有界；这里的沙是学出来的特征动力学，
分岔只说明动作进了计算图，不说明多步之后还准。多步准要另看规划。

### 5.4 零样本规划：目标是一张图，搜索在特征里

训练阶段 DINO-WM 没见过任务奖励。测试时给你一张目标图 $o_g$，问题变成：
找一条动作序列，让想象中的最后一帧特征靠近目标特征。论文第 3.2 节把代价写成

$$
\mathcal{C}=\|\hat{z}_T-z_g\|^2,\quad \hat{z}_0=\mathrm{enc}(o_0),\quad z_g=\mathrm{enc}(o_g)
$$

实现里还加了本体感觉项。`planning/objectives.py::create_objective_fn` 默认 `mode: last`，

```text
loss = loss_visual + alpha * loss_proprio
```

只比较预测序列的最后一帧和目标。`plan_pusht.yaml` 里 `alpha: 1`。

搜索用 CEM，包在 MPC 外环里。`planning/cem.py::CEMPlanner.plan` 的一步是：
从当前均值方差采样 `num_samples` 条动作（PushT 配置 300），对每条做 `wm.rollout`，
按代价取 `topk=30` 更新均值方差，重复 `opt_steps=30`。
`planning/mpc.py::MPCPlanner` 再决定一次执行多少步。仓库也实现了梯度下降规划
（`planning/gd.py`），论文附录 A.5.3 写 CEM 更稳。直觉上，特征空间里的代价对动作
并不是一个好脾气的凸函数：推杆绕过 T 块、擦边、正撞，都会让 $\|\hat{z}_T-z_g\|^2$
出现好几个坑。CEM 靠采样跳坑，梯度下降容易卡在“看起来近、实际上推死”的局部。
本课缩小规划沿用官方 `plan_pusht.yaml` 的 CEM，不要改成 gd 再和论文表 1 比。

“零样本”在这里有精确含义：不需要专家演示、不训奖励头、不另训逆动力学。
不是“没见过这个环境”。模型在该环境的离线轨迹上训过，换的是任务指定方式：
一张目标图，而不是一个奖励函数。缩小设置里你只看方向：规划出来的推杆是否把 T
往目标方位推。5 个样本的成功率方差极大，报数字没有资格跟论文 0.90 比。

验证：`plan.py` 会把最终评估写成 `logs.json`，并在 `plan_outputs/` 下存视频。
你要对照目标图逐段看，写下“朝目标 / 推反了 / 几乎没动”，这比抄一个成功率有用。

### 5.5 Cosmos 为什么多数情况下只讲

NVIDIA 把世界基础模型拆成三家，仓库 README 写得很清楚：
`cosmos-predict` 预测未来视频，`cosmos-transfer` 按空间控制信号改视频，
`cosmos-reason` 是物理 AI 的视觉语言模型，Predict2.5 拿它当文本编码器。
Predict2.5 本身是 flow matching（流匹配：学一条从噪声到数据的速度场，一步或多步积分出样本），
把 Text2World、Image2World、Video2World 收进同一个模型。
论文（arXiv:2511.00062）写训练用了约 2 亿段整理过的视频，放出 2B 和 14B。

三问上，基础检查点卡在第一问的“逐步动作”：默认推理脚本
`examples/inference.py` 的条件是提示文本加图像或视频。动作条件在
`docs/inference_robot_action_cond.md` 的后训练分支，入口文件是
`examples/action_conditioned.py`，文档写明暂不支持多卡，`context_parallel_size`
必须为 1。那一支的检查点名字在 README 的模型表里叫
`Cosmos-Predict2.5-2B/robot/action-cond`，输入列写的是 action，不再是
text + image or video。第三问上，平台把自己定位成合成数据和后训练的底座，
规划交给下游；2026 年 2 月的更新还加了 `robot/policy`（Libero、RoboCasa）和蒸馏说明，
那些是策略头，更不是你在本课要训的世界模型。

本课标只讲，硬件理由是硬的。Hugging Face 的 `nvidia/Cosmos-Predict2.5-2B` 卡片写
Video2World（720p，16FPS）约需 32.54GB 显存。主线假设 24GB。仓库 README 顶部还写着
Cosmos 3 已发布、Predict2.5 不再活跃开发，谱系仍在换名。第 12 课因此不锚定它；
本课读文档、填表，不假装跑过。有 8×A100 短租的人，加餐走官方 `docs/inference.md`，
结论单独标注“加餐，非本课必做”。

## 6. 源码导读

克隆 `gaoyuezhou/dino_wm` 之后按这个顺序读，每个文件带着问题进去：

| 文件 | 是哪个零件 | 带着什么问题读 |
|---|---|---|
| `models/dino.py` | 冻住的眼睛 | `feature_key` 取 patch 还是 cls？`emb_dim` 从哪来？ |
| `models/visual_world_model.py` | 世界模型本体 | `encode` / `predict` / `rollout` / `replace_actions_from_z` 各自改哪一维？ |
| `models/vit.py` | 转移模型 | `generate_mask_matrix` 为什么按帧遮挡？`NUM_FRAMES` 何时被改写？ |
| `models/proprio.py` | 动作与本体感觉编码器 | 为什么动作和本体感觉共用一个小 MLP？ |
| `conf/train.yaml` | 训练默认 | `train_encoder: False`、`num_hist: 3`、`concat_dim: 1` 分别意味着什么？ |
| `conf/encoder/dino.yaml` 与 `dino_cls.yaml` | 编码器消融 | 论文表 2 的 DINOPatch / DINO CLS 如何一键切换？ |
| `conf/env/pusht.yaml` | PushT 数据 | `data_path` 怎样拼 `DATASET_DIR`？`with_velocity` 打开了什么？ |
| `conf/plan_pusht.yaml` | PushT 规划 | `n_evals`、`goal_H`、`num_samples`、`opt_steps` 默认多大？ |
| `planning/cem.py` | CEM | 采样、`rollout`、取 topk、更新 mu/sigma 在哪几行？ |
| `planning/objectives.py` | 规划代价 | `mode: last` 和 `mode: all` 比的是哪些帧？ |
| `train.py` | 训练入口 | Hydra 配置路径？检查点写到哪？何时 `wandb.init`？ |
| `plan.py` | 规划入口 | 检查点路径怎样拼出来？`model_epoch` 默认 `latest` 对应哪个文件名？ |

三处最要紧的细节，读的时候记下来。

第一，检查点布局被写死。`plan.py` 里

```text
model_path = f"{ckpt_base_path}/outputs/{model_name}/"
model_ckpt = ... / "checkpoints" / f"model_{model_epoch}.pth"
```

官方 README 要求先改 `ckpt_base_path`，再跑 `model_name=pusht`。
OSF 压缩包若不是这个目录树，你要自己摆成 `outputs/pusht/hydra.yaml` 加
`outputs/pusht/checkpoints/model_latest.pth`。缺 `hydra.yaml` 会在读配置那一行直接失败。

第二，解码器不参与规划梯度。`VWorldModel.forward` 对 `z_pred` 先 `detach` 再 `decode`；
`CEMPlanner` 的 `rollout` 全程 `torch.no_grad()`。你会看的视频来自可选解码器或环境回放，
规划器自己只比较特征。

第三，官方训练配置假定 Slurm。`conf/train.yaml` 的 Hydra launcher 是 `submitit_slurm`，
还写了 `gres: "gpu:h100:1"`。单机直接 `python train.py ...` 走的是 Hydra 的普通 run，
不会真去要 H100；那几行是他们集群扫参用的。你的卡是 24GB 也没关系，默认 batch 32
对 ViT-S 预测器够用。

对照仓库只读、不跑：`nvidia-cosmos/cosmos-predict2.5` 的 `docs/setup.md`、
`docs/inference.md`、`docs/inference_robot_action_cond.md`。把三家分工和两条推理命令
抄进笔记即可。1xgpt 的 README 有一句需要圈出来：当前 GENIE 实现“only trains on video
sequences, not actions”。这是第 7 节动作条件检查的标准答案来源。

## 7. 实验

主线顺序：先按官方 README 把环境和数据立住，再做动作对换（含滑杆），然后用官方
`plan.py` 跑一次缩小规划，最后读 1xgpt / Cosmos 文档填表。对换在规划前面，是因为
规划失败有两种完全不同的原因：模型不听动作，或模型听动作但搜不好。先把第一种排除。

全程开独立 conda 环境。下面的路径以仓库根目录为准。

### Step 1: 克隆与环境

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

这四条与仓库 README 的 Installation 一致。第一次创建会装很久：环境里有
`torch==2.3.0`、`mujoco==3.2.7`、`dm-control`、`wandb`。若 conda 求解失败，先确认
你在 Linux x86_64 上，不要把这份 `environment.yaml` 拿去硬套 macOS。

PushT 主线不需要 README 里的 MuJoCo 210 和 PyFlex。那两段是 PointMaze 仿真加速和
绳子/颗粒环境用的。本课用离线 `pusht_noise` 做对换，规划才起仓库自己的 gym 环境。

### Step 2: 数据、检查点与环境变量

数据集和官方检查点都在 README 给出的 OSF 链接：
[osf.io/bmw48](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28)。
浏览器下载后解压。主线只要两样：`pusht_noise` 目录，以及 PushT 的世界模型检查点。
解压后的数据树必须能对上 README 写的结构：

```text
data
├── deformable
├── point_maze
├── pusht_noise
└── wall_single
```

`datasets/pusht_dset.py::load_pusht_slice_train_val` 会再拼上 `/train` 和 `/val`，
所以磁盘上实际是 `DATASET_DIR/pusht_noise/train` 与 `.../val`，里面要有
`states.pth`、`rel_actions.pth`、`velocities.pth`、`seq_lengths.pkl` 和 `obses/`。
`conf/env/pusht.yaml` 用 Hydra 的 `oc.env:DATASET_DIR` 拼 `pusht_noise` 当根路径，所以：

```bash
export DATASET_DIR=/path/to/data
```

把 `/path/to/data` 换成你解压后、能直接看到 `pusht_noise` 的那个目录。

`train.py` 和 `plan.py` 都会调用 `wandb.init`。本课不上传云端：

```bash
export WANDB_MODE=offline
```

检查点必须让 `plan.py` 第 442 行的拼接成立。默认 `model_epoch: latest`，对应文件名
`checkpoints/model_latest.pth`。在某个绝对路径 `CKPT_BASE` 下摆成：

```text
CKPT_BASE/outputs/pusht/hydra.yaml
CKPT_BASE/outputs/pusht/checkpoints/model_latest.pth
```

OSF 压缩包若已经是 `outputs/pusht/...`，把 `CKPT_BASE` 指到上一级即可。
若只有权重文件，缺少 `hydra.yaml`，规划脚本读配置会失败，这时用下面 Step 4 的
对换脚本仍可工作（脚本会自己用 `conf/` 里的默认配置重建模型，权重对得上才行）。

第一次跑会从 `torch.hub` 拉 `facebookresearch/dinov2`。需要能访问 GitHub。

### Step 3: 动作对换（先看分不分岔）

在仓库根目录新建 `action_swap_slider.py`。它不启动 gym，只吃一份轨迹和检查点，
用 `VWorldModel.rollout` 比较原动作、取反动作、以及滑杆给出的一组方向。

```python
"""第 22 课胶水：PushT 动作对换与方向滑杆。在 dino_wm 仓库根目录运行。"""
import argparse
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import call
from omegaconf import OmegaConf

from plan import load_model


def load_bundle(ckpt_base, model_name, epoch, device):
    model_dir = Path(ckpt_base) / "outputs" / model_name
    cfg_path = model_dir / "hydra.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"缺少 {cfg_path}，对照 plan.py 的路径拼接检查目录树")
    train_cfg = OmegaConf.load(cfg_path)
    ckpt = model_dir / "checkpoints" / f"model_{epoch}.pth"
    wm = load_model(ckpt, train_cfg, train_cfg.num_action_repeat, device)
    wm.eval()
    _, trajs = call(
        train_cfg.env.dataset,
        num_hist=train_cfg.num_hist,
        num_pred=train_cfg.num_pred,
        frameskip=train_cfg.frameskip,
    )
    return wm, trajs["valid"], train_cfg


def take_segment(dset, num_hist, horizon, seed=0):
    rng = np.random.RandomState(seed)
    need = num_hist + horizon
    for _ in range(200):
        i = int(rng.randint(0, len(dset)))
        obs, act, state, _ = dset[i]
        if obs["visual"].shape[0] >= need and act.shape[0] >= need:
            return obs, act
    raise RuntimeError("验证集里找不到足够长的轨迹，检查 DATASET_DIR/pusht_noise")


def visual_l2(z_a, z_b):
    a = z_a["visual"][:, -1]
    b = z_b["visual"][:, -1]
    return torch.mean((a - b) ** 2).item()


def rollout_from(wm, obs, act, horizon, device):
    # PushTDataset.get_frames 已经除以 255、换成 CHW 并做过 transform，动作也已标准化。
    # 不要再走 preprocessor.transform_obs，否则会二次归一化。
    obs0 = {
        "visual": obs["visual"][: wm.num_hist].unsqueeze(0).to(device),
        "proprio": obs["proprio"][: wm.num_hist].unsqueeze(0).to(device),
    }
    a = act[: wm.num_hist + horizon].unsqueeze(0).to(device)
    with torch.no_grad():
        z_obs, _ = wm.rollout(obs_0=obs0, act=a)
    return z_obs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-base", required=True)
    p.add_argument("--model-name", default="pusht")
    p.add_argument("--model-epoch", default="latest")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="action_swap_out")
    p.add_argument("--slider", action="store_true")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm, dset, cfg = load_bundle(
        args.ckpt_base, args.model_name, args.model_epoch, device
    )
    obs, act = take_segment(dset, cfg.num_hist, args.horizon, args.seed)
    act_orig = act.clone()
    act_flip = act.clone()
    act_flip[:, :2] = -act_flip[:, :2]
    z0 = rollout_from(wm, obs, act_orig, args.horizon, device)
    z1 = rollout_from(wm, obs, act_orig, args.horizon, device)
    zf = rollout_from(wm, obs, act_flip, args.horizon, device)
    d_rep = visual_l2(z0, z1)
    d_flip = visual_l2(z0, zf)
    print(f"repeat_l2={d_rep:.6f}")
    print(f"flip_l2={d_flip:.6f}")
    print(f"ratio={d_flip / max(d_rep, 1e-12):.3f}")

    radii = act_orig[:, :2].norm(dim=-1).clamp(min=1e-4)
    thetas = np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)
    dist_to_zero = []
    for th in thetas:
        a = act_orig.clone()
        a[:, 0] = radii * math.cos(th)
        a[:, 1] = radii * math.sin(th)
        zt = rollout_from(wm, obs, a, args.horizon, device)
        dist_to_zero.append(visual_l2(z0, zt))
    np.savetxt(
        os.path.join(args.out, "theta_l2.csv"),
        np.stack([thetas, np.array(dist_to_zero)], axis=1),
        delimiter=",",
        header="theta,l2_to_original",
        comments="",
    )
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(thetas, dist_to_zero, marker="o")
    ax.set_xlabel("push angle (rad)")
    ax.set_ylabel("feature L2 vs original action")
    ax.set_title("action swap: does the prediction branch?")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "theta_l2.png"), dpi=140)
    plt.close(fig)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(f"repeat_l2={d_rep:.6f}\n")
        f.write(f"flip_l2={d_flip:.6f}\n")
        f.write(f"branch={'yes' if d_flip > 5 * d_rep + 1e-6 else 'no'}\n")
    print(f"wrote {args.out}")

    if args.slider:
        from matplotlib.widgets import Slider

        fig2, ax2 = plt.subplots(figsize=(6, 3.6))
        plt.subplots_adjust(bottom=0.25)
        line = ax2.plot(thetas, dist_to_zero, marker="o")[0]
        ax2.set_xlabel("push angle (rad)")
        ax2.set_ylabel("feature L2 vs original")
        ax_sl = fig2.add_axes([0.15, 0.08, 0.7, 0.08])
        sl = Slider(ax_sl, "theta", 0.0, 2.0 * math.pi, valinit=0.0)

        def on_change(val):
            a = act_orig.clone()
            a[:, 0] = radii * math.cos(val)
            a[:, 1] = radii * math.sin(val)
            zt = rollout_from(wm, obs, a, args.horizon, device)
            d = visual_l2(z0, zt)
            ax2.set_title(f"theta={val:.2f}  L2={d:.5f}")
            fig2.canvas.draw_idle()
            _ = line

        sl.on_changed(on_change)
        plt.show()


if __name__ == "__main__":
    main()
```

`Preprocessor` 来自仓库根目录的 `preprocessor.py`，`load_model` 来自 `plan.py`，
都是现成函数，不要改名。有显示器再加 `--slider`；服务器上只看 `theta_l2.png` 和
`summary.txt` 即可。

```bash
python action_swap_slider.py --ckpt-base /abs/path/to/ckpt_base --model-name pusht --out action_swap_out
```

把 `/abs/path/to/ckpt_base` 换成 Step 2 的 `CKPT_BASE`。预期输出类似：

```text
repeat_l2=0.000000
flip_l2=0.0xxxx
ratio=很大的数
wrote action_swap_out
```

`repeat_l2` 应当极小（同一输入、`eval()`、无随机性）。`flip_l2` 明显更大，
`summary.txt` 里 `branch=yes`。`theta_l2.png` 应当随角度起伏，不是一条平线。
若 `flip_l2` 和 `repeat_l2` 同一量级，先别急着规划：检查动作前两维是不是位移、
检查检查点是否真是 PushT、检查 `DATASET_DIR` 是否指到 `pusht_noise`。

读图：横轴是推的方向，纵轴是“这个方向的想象特征”和“原动作想象特征”的距离。
一条平线意味着模型不在乎你往哪推。有起伏才说明动作进了动力学。
`pusht_dset.py` 里 `ACTION_MEAN` 只有两维，取反前两维就是把位移拨到对面；
`with_velocity: true` 影响的是本体感觉（状态里拼了速度），不是动作维数。

有图形界面时：

```bash
python action_swap_slider.py --ckpt-base /abs/path/to/ckpt_base --model-name pusht --slider
```

拖滑杆改 $\theta$，标题里的 L2 应当跟着变。分岔当场打分：能看出距离随角度变化记 1，
几乎不变记 0。把分数写进 `action_swap_out/summary.txt` 下面一行。

### Step 4: 缩小零样本规划（只看方向）

官方 README 的 PushT 规划命令是：

```bash
python plan.py --config-name plan_pusht.yaml model_name=pusht
```

默认 `n_evals: 50`、`goal_H: 5`、CEM 每步 300 条样本，一次要跑很久。本课覆盖成 5 个
起点，只看方向：

```bash
python plan.py --config-name plan_pusht.yaml model_name=pusht n_evals=5 goal_H=5 ckpt_base_path=/abs/path/to/ckpt_base
```

`plan_pusht.yaml` 里规划器已经是 `planning.mpc.MPCPlanner` 加 `planning.cem.CEMPlanner`，
不必再改 `planner=`。成功的话，Hydra 会在 `plan_outputs/` 下建目录，里面有
`logs.json`、`plan_targets.pkl` 和若干 `output_final` 视频。

看视频时只记三档，不要发明更细的分数：朝目标、推反了、几乎没动。5 局里朝目标的
多于推反的，方向就算立住。论文表 1 的 0.90 是 50 个样本、完整优化步数的结果，
你的 5 局没有资格和它比。把这句写进笔记。

若 `gym.make("pusht")` 报环境未注册，先读 `env/` 目录里 PushT 的注册代码，确认
`conda activate dino_wm` 后 `python -c "import gym; gym.make('pusht')"` 能过。
环境起不来时，Step 3 的对换仍然有效，规划记成“环境未起，只完成对换”。

### Step 5: （选做）冒烟训练，确认你能从零启动

官方训练例是 PointMaze。本课若还想摸一下训练入口，改环境名即可：

```bash
python train.py --config-name train.yaml env=pusht frameskip=5 num_hist=3
```

这条与 README 的 `python train.py --config-name train.yaml env=point_maze frameskip=5 num_hist=3`
同一入口，只把 `env` 换成已有的 `conf/env/pusht.yaml`。检查点会写到
`conf/train.yaml` 里 `ckpt_base_path` 下的 `outputs/<日期>/<时间>/checkpoints/`。
冒烟看 `z_loss` 是否在前几个 epoch 下降，不要训满 100 个 epoch。
`debug=True` 仍然会 `wandb.init`，只是项目名变成 `dino_wm_debug`，离线模式照旧够用。

### Step 6: 1xgpt 的动作条件检查（真实机器人 token）

DINO-WM 的动作是仿真器里的低层位移。1xgpt 提供 100 小时以上的 EVE 第一人称 token
和原始动作，用来回答另一个问题：公开的真机世界模型基线，动作进没进模型。

```bash
git clone https://github.com/1x-technologies/1xgpt.git
```

```bash
cd 1xgpt
```

```bash
./build.sh
```

`./build.sh` 是 README 写明的安装与下数据入口。然后打开 `README.md` 的 GENIE 一节：
作者写明这份实现只在视频序列上训练，不在动作上训练，并说“加一个加法嵌入就能接动作”。
再打开 `genie/generate.py` 的命令行参数，确认没有动作文件入口。官方 1X GENIE Baseline
一节给出的生成命令只接收检查点目录、输出目录、样本编号、MaskGIT 步数和温度，
没有动作张量这一项。你在真机 token 上能做的动作条件检查，结论就是：数据里有原始动作，
公开 GENIE_138M 基线没有把它接进 $P(z_{t+1}\mid z_t, a_t)$。想做对换，得自己加嵌入并
重训，那超出本课实战范围。把“数据有动作、基线没用动作”写进评测表的备注，不要把
1xgpt 基线填成动作条件世界模型。

有余力再生成一帧，确认环境没装坏。命令与 README 的 Baseline 一节相同：

```bash
python genie/generate.py --checkpoint_dir 1x-technologies/GENIE_138M --output_dir data/genie_check --example_ind 0 --maskgit_steps 2 --temperature 0
```

### Step 7: 填开源三档 × 三分评测表

新建 `foundation_wm_scorecard.md`，按下表抄一份，用你自己的证据改措辞。
open-oasis 那一行的“预测准”必须引用第 12 课的崩坏记录，禁止本课重跑 Oasis。
Genie 3 的长时程一致性必须出现“宣称”二字。

| 系统 | 开源档 | 本课档位 | 预测准 | 生成真 | 规划好 |
|---|---|---|---|---|---|
| DINO-WM | 可跑（代码、数据、检查点） | 实战 | 对换 L2 填这里；多步只看规划视频方向 | 解码器可选，不是卖点 | 论文表 1 有成功率；你的 5 局只看方向 |
| Cosmos-Predict2.5 | 可跑但 2B Video2World 约 32.54GB | 只讲 | 未在 24GB 上测 | 官方 demo / 论文，生成真是卖点 | 基础模型不直接规划，靠后训练 |
| open-oasis | 仅权重 | 第 12 课体验，本课不重测 | 填你第 12 课的崩坏起始帧 | 中（Minecraft 风格） | 无公开决策验收 |
| Genie 3 | 纯 demo，无权重，不能练 | 只讲 | 宣称数分钟一致，未测 | 官方博客 720p 24fps，未测 | 定位训练场，无公开验收 |

表下写四行三问速答（动作 / 状态 / 规划），每行标明证据是代码、是你的对换数字、
还是官方宣称。

## 8. 配置与预算

DINO-WM 官方训练默认（`conf/train.yaml`）和本课用到的规划默认如下。数字以仓库为准，
不要把冒烟当论文复现。

| 项 | 官方默认 | 本课用法 |
|---|---|---|
| 图像边长 | `img_size: 224` | 不改 |
| 历史帧 | `num_hist: 3` | 不改 |
| 预测步 | `num_pred: 1` | 不改 |
| 抽帧 | `frameskip: 5` | 不改 |
| 编码器 | `dinov2_vits14` patch，冻住 | 不改 |
| 预测器 | ViT 6 层 16 头，`mlp_dim: 2048` | 不改 |
| 训练 epoch | 100 | 选做冒烟看 `z_loss` 下降即可 |
| 训练 batch | 32 | 24GB 够用 |
| PushT 规划评估数 | `n_evals: 50` | 改成 5 |
| 规划地平线 | `goal_H: 5` | 不改 |
| CEM | 300 样本 / topk 30 / 30 步 | 不改，时间主要花在这里 |
| 解码器 | 训练时开，规划时不反传 | 对换脚本不依赖解码 |

时间量级（单张 24GB，依机器浮动）：

| 环节 | 大概耗时 | 说明 |
|---|---|---|
| 装 conda 环境 | 数十分钟到两小时 | `environment.yaml` 很重 |
| 下载 OSF 数据与检查点 | 视网速 | 主线只需 `pusht_noise` + PushT 权重 |
| 动作对换 + 8 个角度 | 数分钟 | 含第一次拉 DINOv2 |
| 规划 `n_evals=5` | 数十分钟量级 | CEM 每步 300 次 `rollout` |
| 选做冒烟训练 | 数小时量级 | 不要求训完 100 epoch |
| 1xgpt `./build.sh` | 视下载 | 本课只要求读到“基线不用动作” |

Mac / CPU：第 7 节 Step 3-5 需要 CUDA。`models/vit.py` 里因果掩码写了 `.to('cuda')`，
没有 MPS 分支。阅读、填表、1xgpt 的文档检查可以在无 GPU 机器上完成。

Cosmos：`docs/setup.md` 要求 Ampere 及以上、驱动需兼容 CUDA 12.8。2B Video2World 的
官方显存数字是约 32.54GB。主线不要下载那份权重。8×A100 加餐才走
`python examples/inference.py -i assets/base/robot_pouring.json -o outputs/base_video2world --inference-type=video2world`。

## 9. 验收

- [ ] 能不看笔记说出 DINO-WM 四个零件：冻住的 DINOv2 patch、动作/本体感觉编码器、
      因果 ViT 预测器、可选解码器，并指出规划用哪一段、不用哪一段；
- [ ] `action_swap_out/summary.txt` 里 `repeat_l2` 接近 0，`flip_l2` 明显更大，
      `branch=yes`；`theta_l2.png` 不是一条平线；
- [ ] 滑杆或 8 个角度扫描有当场打分（1 分岔 / 0 不分岔），分数写进笔记；
- [ ] 缩小规划要么交出 `plan_outputs/` 下至少一段视频和五局里“朝目标 / 推反 / 没动”
      的计数，要么写明 gym 环境未起、只完成对换；无论哪种，都写了“不是论文 0.90 的复现”；
- [ ] `foundation_wm_scorecard.md` 四行齐全：DINO-WM、Cosmos-Predict2.5、open-oasis、
      Genie 3；开源档、本课档位、三分评测三列都不空；
- [ ] open-oasis 的预测准引用第 12 课崩坏帧数，本课没有重跑 `generate.py`；
- [ ] Genie 3 的长时程一致性带“宣称”，没有写成你测过；
- [ ] 1xgpt 备注写清：数据有原始动作，GENIE_138M 基线训练不用动作；
- [ ] 能口头回答：Cosmos 基础检查点卡在三问的哪一问，动作条件要去哪个后训练文档里找。

## 10. 排错

| 症状 | 最可能的原因 | 怎么确认 | 怎么修 |
|---|---|---|---|
| `conda env create` 求解失败 | 不在 Linux x86_64，或网络拉不到 NVIDIA 轮子 | 看 conda 报错里的平台和 URL | 换 Linux 机器；不要改 `environment.yaml` 去套 Mac |
| `DATASET_DIR` 相关 FileNotFound | 指到了 `pusht_noise/train` 的上两级，或少了 `val` | 列出 `DATASET_DIR/pusht_noise/train/obses` | 根路径要能拼出 `/train` 与 `/val` |
| `plan.py` 报缺少 `hydra.yaml` | 检查点目录树和 `ckpt_base_path/outputs/pusht/` 对不上 | 对照 Step 2 的目录树 | 建目录或改 `ckpt_base_path` |
| `plan.py` 报 `model_latest.pth` 不存在 | OSF 文件名不是 `model_latest.pth` | `ls outputs/pusht/checkpoints` | 把实际文件名拷成 `model_latest.pth`，或覆盖 `model_epoch=` |
| wandb 要登录 | 没设离线 | 看报错是否含 wandb | `export WANDB_MODE=offline` 后重跑 |
| `torch.hub` 拉 dinov2 失败 | 访问不了 GitHub | 报错含 github.com | 配置代理，或按 facebookresearch/dinov2 文档手动缓存到 hub 目录 |
| `vit.py` 报 CUDA / `.to('cuda')` | 没 GPU，或脚本跑在 CPU | `python -c "import torch; print(torch.cuda.is_available())"` | 对换和规划必须 CUDA；Mac 走阅读路径 |
| `repeat_l2` 不是 0 | 模型仍在 `train()`，或两次前向用了 dropout | 检查 `wm.eval()` | 用本课胶水，不要在 `train()` 下对换 |
| `flip_l2` 几乎等于 `repeat_l2` | 检查点不是 PushT，或动作维不是位移 | 打印 `act.shape`，PushT 动作是 2 维 | 确认 `model_name=pusht`；看 `pusht_dset.py` 的 `ACTION_MEAN` 长度 |
| `gym.make("pusht")` 失败 | 没进 `dino_wm` 环境，或没走到仓库的 env 注册 | 在仓库根目录试 `import gym` | 先完成对换；规划标环境未起 |
| CEM 规划极慢 | `n_evals` 仍是 50 | 看 Hydra 工作目录里的配置 dump | 确认命令行覆盖了 `n_evals=5` |
| 1xgpt `./build.sh` 很慢或失败 | 在下 Hugging Face 数据 | 看脚本打印 | 本课验收不依赖生成成功，读 README 即可 |
| 想跑 Cosmos 在 24GB OOM | 2B Video2World 官方约 32.54GB | 对照 HF 模型卡 | 停。标只讲，不要关 guardrail 硬挤 |
| `hydra.yaml` 里数据路径仍是作者机器上的绝对路径 | OSF 检查点带走了训练时的配置 | 打开 `outputs/pusht/hydra.yaml` 搜 `data_path` | 保证已导出 `DATASET_DIR`；必要时把路径改回配置里的 `oc.env:DATASET_DIR` 写法 |

## 11. 前沿与改造

同一问题，2023-2026 年公开系统走了三条不同的路。本课只引用核对过的材料。

UniSim（Yang et al.，arXiv:2310.06114）把互联网图文、全景扫描、
导航、真机和仿真动作收进同一个“动作进、视频出”的扩散观察预测器，高阶指令和低层
位移都当成条件；论文展示在仿真里训的策略零样本放到真机。它把交互写成
$p(o_t\mid h_{t-1}, a_{t-1})$，历史 $h$ 只用最近一段，再自回归滚下去。
这和 [第 12 课](12_frontier_landscape.md) 的滑窗是同一类记忆：窗口外的东西只能靠先验
补。它过第一问的方式是把语言和连续控制都编码进动作嵌入，代价是扩散一步很贵，
测试时做 CEM 那种 300 次 rollout 不现实。GAIA-1（Hu et al.，arXiv:2309.17080）把
驾驶视频、文本、自车速度与曲率编成 token：每帧 576 个图像 token，32 个文本 token，
2 个动作 token，交织顺序是文本-图像-动作。世界模型是自回归 Transformer，扩散解码器
负责把 token 变成高分辨率视频。动作是 2 个标量（速度、曲率），第一问在驾驶域成立；
第三问的用途声明是造场景和学表示，不是在模型里搜方向盘。无公开权重，本课只讲。
Cosmos-Predict2.5（arXiv:2511.00062）把 Text/Image/Video2World 收成一个 flow 模型，
Reason1 当文本编码器，Transfer2.5 做空间控制；动作条件是后训练分支
`robot/action-cond`。Genie 3 无权重，官方博客宣称 720p、24 fps、数分钟世界一致，
本课不能把这句话写成测量结果。[第 11 课](11_genie_latent_actions.md) 的 Genie 1
论文（arXiv:2402.15391）才是能读机制的那一档：动作是自己挖的 latent action。

DINO-WM 的缩小实战和这些基础模型之间，规模是一截：数据从
OSF 上的一条 PushT 轨迹集，到 Cosmos 论文写的约 2 亿段视频。钱能买规模，买不来
第一问。机制上的差距有两处。一处是条件接口：基础模型爱用文本，规划爱用低层动作，
二者不是自动互通的，Cosmos 用后训练补动作端口，DINO-WM 从一开始就拼动作。
另一处是推理成本：特征空间里的 ViT 前向便宜，才能在测试时跑 CEM；像素扩散每步
去噪，300 次采样会把规划拖死。这正是 DINO-WM 论文相关工作里批评视频生成世界模型
的那一点。

**动手改造清单**（都在 `gaoyuezhou/dino_wm` 上，预算按 24GB 单卡写）：

1. 编码器消融：把 `conf/encoder/dino.yaml` 换成 `encoder=dino_cls`
   （`feature_key: "x_norm_clstoken"`），在同一份 `pusht_noise` 上冒烟训预测器，
   再跑 Step 3 对换。预算：冒烟数小时。预期：cls 也能分岔（动作仍拼在特征上），
   但 `theta_l2.png` 的起伏变钝，规划方向更差。论文表 2 的方向是 PushT 成功率
   0.90 对 0.44。失败判据：cls 对换完全不分岔，那说明动作嵌入没接上，先查
   `concat_dim`。
2. 对换距离对地平线：`horizon` 取 1、5、10，各跑 Step 3。预算：三次对换，一小时内。
   预期：`flip_l2 / repeat_l2` 在短地平线最大，拉长后因误差累积比值未必单调增加。
   失败判据：horizon=1 就不分岔，模型第一问没过，规划不必做。
3. CEM 样本数扫描：`plan.py` 覆盖 `planner.sub_planner.num_samples=50` 与默认 300，
   `n_evals=5` 不变。预算：两次缩小规划。预期：50 样本更快，朝目标的局数下降或
   不变。失败也是结果：若 50 与 300 看不出差别，写明 5 局分辨率不够，不要声称
   “样本数无影响”。
4. 1xgpt 加动作嵌入（加餐，不占主线）：按 README 说的 additive embedding 给
   `genie` 接上动作，用挑战数据里的 raw actions 做一次对换。预算：改代码半天加
   短训。预期：接上之后对换 L2 上升。失败判据：短训后仍不分岔，动作嵌入维数或
   时间对齐有问题。

论文结论“预训练 patch 比 cls / ResNet / R3M 更利于操作规划”
对应改造 1，缩小设置预期能看到对换起伏变钝这一同方向趋势，不能复现表 2 的绝对
成功率。论文结论“离线世界模型可以零样本规划”对应 Step 4，缩小设置只复现方向，
不复现 0.90。论文结论“视频扩散世界模型难用于测试时 MPC”对应你读 Cosmos 文档时
的体感：一条 Video2World 推理已经要 30GB 以上显存，CEM 乘 300 在本课硬件上不成立。

## 12. 论文与延伸

1. DINO-WM（Zhou, Pan, LeCun, Pinto，[arXiv:2411.04983](https://arxiv.org/abs/2411.04983)）。
   带着问题读：第 3.1 节为什么把解码器写成 optional？公式 (1) 的教师强制和
   `VWorldModel.forward` 的 `z_tgt = z[:, num_pred:]` 是否同一件事？表 1、表 2
   哪些数字你的 24GB 实验有资格对照，哪些没有？项目页
   [dino-wm.github.io](https://dino-wm.github.io/) 的 PushT 视频和下排想象差在哪一帧？
2. Cosmos World Foundation Model Platform（NVIDIA，[arXiv:2501.03575](https://arxiv.org/abs/2501.03575)）
   与 Cosmos-Predict2.5（[arXiv:2511.00062](https://arxiv.org/abs/2511.00062)）。
   带着问题读：Predict / Transfer / Reason 三家各自的输入输出是什么？基础模型的
   裁判交给谁？动作条件出现在论文的哪一节、仓库的哪篇文档？把“平台”和“可规划的
   世界模型”分开写。
3. UniSim（Yang et al.，[arXiv:2310.06114](https://arxiv.org/abs/2310.06114)）。
   带着问题读：它怎样把语言指令和 $\Delta x,\Delta y$ 收进同一个动作嵌入？有限历史
   的观察预测器和第 12 课滑窗自回归是不是同一个记忆问题？策略在仿真里训完再放真机，
   过的是三问里的哪一问？
4. GAIA-1（Hu et al.，[arXiv:2309.17080](https://arxiv.org/abs/2309.17080)）。
   带着问题读：速度与曲率这两个动作 token 怎样插入文本-图像序列？世界模型（离散
   token）和扩散解码器的分工，和 DINO-WM“特征里规划、像素只是可选可视化”有何同构？
   论文有没有把 GAIA-1 当成在线规划器？
5. DINOv2（Oquab et al.，[arXiv:2304.07193](https://arxiv.org/abs/2304.07193)）。
   带着问题读：`x_norm_patchtokens` 相对 cls token 多保留了什么空间信息？为什么
   DINO-WM 冻住它而不微调（对照 `train_encoder: False`）？
6. Genie（Bruce et al.，[arXiv:2402.15391](https://arxiv.org/abs/2402.15391)，
   [第 11 课](11_genie_latent_actions.md) 主论文）。带着问题重读：latent action
   的第一问该怎么验？Genie 3 无权重，本课为什么必须停在只讲？
7. 选读：1X World Model Challenge 的 README（[1x-technologies/1xgpt](https://github.com/1x-technologies/1xgpt)）。
   带着问题读：压缩挑战的 temporally teacher-forced 指标量的是哪一种“预测准”？
   Evaluation Challenge 若开放，和本课第三问是不是同一件事？

现在系统会：在冻住的空间特征上按动作预测下一刻，并用对换证明动作不是摆设。
下一课要处理另一件桌宠躲不掉的事：杯子倒了、手机没动，一条向量状态会把两样东西
揉在一起。第 23 课用物体中心的槽，让杯子和手不再共用同一个向量。

