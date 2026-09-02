---
id: 27_sim_to_real
title: "仿真里会了，真桌子上为什么还摔"
summary: "接触、延迟、标定和安全约束，哪一样会把梦里的高分变成真机碰撞？"
unit: embodied
play_tools: []
checkpoints:
  - "一张 sim-to-real 失败清单。"
  - "带安全过滤的规划记录。"
---

# 第 27 课：仿真里会了，真桌子上为什么还摔

> 类型：实战（Genesis 桌面推杯仿真）+ 只讲（真机接触、标定与安全层）<br>
> 建议周期：2-3 天<br>
> 硬件：Mac / CPU 可完成本课全部必做实验；NVIDIA GPU 只用来加快并行扫描；本课不上真机<br>
> 锚定仓库：[Genesis-Embodied-AI/genesis-world](https://github.com/Genesis-Embodied-AI/genesis-world)（PyPI 包名 `genesis-world`，文档 [genesis-world.readthedocs.io](https://genesis-world.readthedocs.io/en/latest/)）；对照 [google-deepmind/mujoco_playground](https://github.com/google-deepmind/mujoco_playground) 与 [simpler-env/SimplerEnv](https://github.com/simpler-env/SimplerEnv)<br>
> 产物：一张桌子加杯子加官方臂模型的 Genesis 场景、摩擦与质量随机化下的规划成功率表、桌沿 5 cm 安全过滤器的规划记录、一份 sim-to-real 失败清单

## 1. 这一课做什么

第七幕前三课把循环接到了身体上：第 24 课用视觉预测在想象里选动作，第 25 课用模仿策略快速做出像样的手势，第 26 课把语言进、动作出的 VLA 和世界模型的未来预测拆开分工。到这里，规划器在自己的模型里已经能把杯子推到目标附近。本课加的零件不是又一个预测头，而是承认一件难堪的事实：你用来规划的那个世界，和你桌子上的那个世界，不是同一个。

整门课的主干还是这一句：

```text
观察 先压成状态 再按动作预测下一状态 然后展开多条未来 给未来打分 最后选动作
```

前 26 课默认“预测下一状态”用的 $P$ 和真实环境的 $P$ 足够接近，分数才会从梦里搬回环境。桌面接触任务把这个默认打穿。摩擦差 0.3、相机外参偏 1 cm、指令晚到 80 ms，同一条推杯动作就能从“到位”变成“扫落”。本课要你看见成功率怎么掉，并在动作出口装上一层安全过滤器：想象轨迹一旦把杯子送到桌沿 5 cm 以内，截断后续动作。

档位必须写清楚。仿真实验是**实战**：跑的是 Genesis World 当前仓库和官方文档里的 API，桌子、杯子、臂都来自真实仿真器与官方模型，不另造物理引擎。真机接触、标定、延迟补偿是**只讲**：你可以在仿真里注入这些误差，但本课不把任何动作发到 Reachy Mini 或 SO-101 上。没有安全层，不准上真机。第八幕第 28 课才立项桌宠硬件。

术语速查：

| 术语 | 一句人话 |
|---|---|
| 现实间隙（reality gap） | 仿真里学到的行为搬到真机上失效，因为两边的物理、视觉和控制并不相同 |
| 域随机化（domain randomization） | 训练或规划时故意把摩擦、质量、光照等抽成一个分布，逼策略别死磕某一组参数 |
| 系统辨识（system identification, SysID） | 用真机轨迹反推仿真器里该用的增益、摩擦、惯量，让两边的开环响应尽量贴近 |
| 接触求解（contact solver） | 仿真器每一步用来算“谁碰到谁、力有多大、会不会穿过去”的数值程序 |
| 动作延迟（action delay） | 你发出的指令真正作用到电机上已经过了若干步；规划器若当零延迟用，就会超调 |
| 视觉域间隙 | 仿真渲染的桌子和真相机拍到的桌子，纹理、光照、背景都不一样，视觉策略会当没见过 |
| 安全过滤器 | 动作出口的硬闸门：世界模型或几何预测说“会出界/会撞”，就截断或改成停 |
| 规划世界 / 评估世界 | 本课把同一套 Genesis 场景拆成两组参数：一组给规划器当“脑子”，一组当“桌子” |

## 2. 问题

一个在 CarRacing 或 DMC 里几乎看不见的问题，到了桌面上会在第一小时集中爆发：规划器用的转移 $P_{\text{sim}}(s'|s,a)$ 不是真桌子的 $P_{\text{real}}(s'|s,a)$。差从五处进来。

标定。相机到桌面的外参、关节零位、桌面高度，仿真里是精确的一组数，真机上是你拿尺子和标定板凑出来的。差 1 cm，夹爪会从杯沿上方擦过去，或者直接顶进杯壁。

延迟。真机链路是：曝光、传输、推理、下发、电机跟踪。每段都耗时间。仿真默认“这一步发出的 $a_t$，下一步的 $s_{t+1}$ 就已经吃进去”。规划器按零延迟把力用满，真机上力会在杯子已经开始动之后才到，结果是扫飞。

接触。推杯是典型的干摩擦加碰撞。库仑摩擦系数、扭转摩擦、接触求解是罚方法还是约束法、时间步长，都会改滑动距离。仿真器为了稳定还会让物体轻微穿模或粘住。真桌面上还有桌垫纹理、杯底积水、臂的柔性，这些在刚体模型里经常被设成一个常数。

视觉域。第 24 课的视觉预测和第 26 课的 VLA 都吃像素。仿真渲染再好看，灯、阴影、桌布花纹和真相机也不一样。SIMPLER（arXiv:2405.05941）把这件事测成了评测问题：视觉对不上，真机策略在仿真里的排名都会乱。

安全。前面四处是“为什么会摔”。第五处是“摔了有没有人拦”。世界模型可以想象出杯子掉下去，VLA 可以听懂“把杯子推到桌沿”，模仿策略可以复现专家的一次险棋。如果动作出口没有过滤器，这些能力都会变成把水扫到键盘上的能力。桌宠质量小，仍能扫落液体。

本课用 Genesis 把前四处做成可调旋钮，把第五处做成必须通过的闸门。规划仍用第 24 课那种“在模型里展开再挑动作”的结构，只不过这里的模型先用 Genesis 自己的刚体求解器充当，用来暴露间隙；真机理论只讲到能写进失败清单，不接到硬件。

## 3. 准备

- [第 03 课](./03_mdn_rnn_action_conditioned.md) 的动作对换和 [第 17 课](./17_evaluating_world_models.md) 的评测纪律：同一套量法、一次改一个因素、报均值时带次数。本课的“成功率”按这个写。
- 第 24 课的规划结构：给定当前状态，在预测器里展开若干动作序列，打分，执行第一步。本课用 Genesis 当这台预测器，不新训网络。
- Python 3.10 到 3.13（官方要求 `>=3.10,<3.14`），已能安装与 CUDA 或 CPU 匹配的 PyTorch。先按 [pytorch.org/get-started](https://pytorch.org/get-started/locally/) 装好 PyTorch，再装 Genesis。
- 磁盘约 2 GB：`genesis-world` 的 wheel 在 PyPI 上约 80 MB，首次编译内核会再占缓存。
- 本课**不需要**真机。有 SO-101 或 Reachy Mini 的人，把官方 URDF 留到第 8 节的选做和第 28 课，不要在本课接串口。
- 建议单独建虚拟环境，避免和前几课的老 gym、LeRobot 依赖搅在一起。

## 4. 学习目标

1. 写出 $P_{\text{sim}}(s'|s,a)$ 和 $P_{\text{real}}(s'|s,a)$ 不相等的五个来源，并能各举一个桌面推杯的失败样子；
2. 用 Genesis World 官方 API 搭出桌子、杯子和自带的 Franka 臂，说明为什么本课不用自造积分器；
3. 调用 `set_friction_ratio` 和 `set_mass_shift`，画出规划成功率随摩擦、质量变化的表；
4. 给规划器加上动作延迟缓冲，指出延迟多大时同一条推杯轨迹开始超调或穿模；
5. 实现桌沿 5 cm 安全过滤器，保存一份被截断的规划记录，口头说清它拦的是动作出口而不是损失函数；
6. 对照 SIMPLER、LIBERO、DayDreamer 三篇核对过的论文，说出“仿真评真机策略”“仿真基准测 VLA”“跳过仿真直接在真机上学”各自解决哪一段，以及本课为什么仍不准无安全层上真机。

## 5. 原理

五个机制，顺序按你动手时会碰到的顺序走：仿真器是什么、间隙从哪来、随机化怎么补、延迟和接触怎么把轨迹弄坏、安全过滤器为什么必须在出口。

### 5.1 仿真是另一个世界，不是缩小的真机

飞行员上的飞行模拟器是工程师按气动方程造的，误差有界，当局还发合格证。机器人仿真器看起来很像那台模拟器：有重力、有接触、有关节电机。类比失效处在于，桌面接触的关键参数（摩擦系数、接触刚度、关节间隙、桌面微结构）你并没有一张合格证。你填进 XML 或 `gs.materials.Rigid` 的那个 `friction=1.0`，只是作者的默认值，不是你那只杯子底的测量值。

所以仿真器提供的是**另一个**条件分布。记状态 $s$、动作 $a$，规划器在仿真里优化的是

$$
a_{0:H-1}^{\star} = \arg\max_{a_{0:H-1}} \; J_{\text{sim}}\!\left(s_0, a_{0:H-1}\right)
$$

其中 $J_{\text{sim}}$ 沿 $P_{\text{sim}}$ 展开。真桌子上执行同一串动作，轨迹来自 $P_{\text{real}}$。两者的差就是现实间隙。本课后面所有实验，都是在 Genesis 里再造一个“评估世界”，人为让它和“规划世界”差一组摩擦、质量或延迟，从而把 $P_{\text{sim}} \neq P_{\text{real}}$ 变成你能画出来的成功率掉落。

Genesis World 本身是 2024 年 12 月以 Genesis 为名开源、现由 Genesis AI 继续维护的仿真平台，当前仓库是 [Genesis-Embodied-AI/genesis-world](https://github.com/Genesis-Embodied-AI/genesis-world)，PyPI 包名 `genesis-world`。文档把它分成四层：面向用户的仿真接口（URDF / MJCF / 网格）、统一的多物理求解（刚体、FEM、MPM、粒子）、三条渲染路径、以及把 Python 核编译到 CUDA / Metal / x86 的 Quadrants。本课只用刚体层。刚体已经够让杯子滑、撞、翻；流体和布料是加餐，不是本课及格线。

MuJoCo 是对照物，不是敌人。Genesis 的刚体求解在文档里写明参考过 MuJoCo；MuJoCo Playground（`pip install playground`）把一批机器人学习环境接到 MJX 上。你若已经在第 05、06 课用过 DeepMind Control，那是同一家族的刚体接触。本课选 Genesis，是因为它的桌面物体、官方臂模型和域随机化 API 写在同一套 Python 对象上，改摩擦不必先挖 XML。

### 5.2 间隙的五条通道

把 $s$ 拆开看，桌面推杯至少含：杯的位姿与速度、臂的关节角与速度、桌面几何、相机外参。$P$ 不相等，可以从这五条通道分别走进来。

**标定。** 仿真里桌子高度是你写进 `pos` 的那个标量。真机上它是相机外参、桌腿垫片和关节零位的混合物。外参偏了，视觉世界模型看到的“杯在桌心”其实已经靠沿。关节零位偏了，同一组位置指令会把末端送到另一个笛卡尔点。Peng 等人的动力学随机化工作（arXiv:1710.06537）专门测过：标定误差一大，只在单一仿真里训出的推物策略会在真机上反复推偏。

**延迟。** 把控制循环写成

$$
s_{t+1} \sim P\!\left(s_{t+1} \mid s_t, a_{t-d}\right)
$$

$d$ 是以控制周期计的延迟步数。$d=0$ 是仿真默认；$d=4$ 在 `dt=0.01` s 时就是 40 ms，对桌面推杯已经能让末端在接触发生后才加力。规划器若在 $P(\cdot \mid s_t, a_t)$ 里搜，等于拿错了条件。

**接触。** 刚体接触要同时满足互不穿透和摩擦锥。Genesis 默认用牛顿约束求解，摩擦锥可选金字塔或椭球；文档建议物体需要稳稳停住时改用椭球锥，并打开 Signorini 条件，让摩擦力被法向力限制，避免滑块被“提离”桌面。时间步太大、接触约束没解收敛，就会穿模：杯子的 $z$ 掉到桌面以下，或者臂的网格和杯体重叠。穿模不是渲染 bug，是求解器在这一步放弃了互不穿透。

**视觉域。** 像素观察下，$s$ 根本进不去规划器，进的是渲染图。纹理、光照、背景一变，同一套权重会给出另一串动作。SIMPLER 把这个问题从“看起来不像”推进到可量化：他们用绿幕把真背景贴进仿真、把真纹理烤到物体上，再拿配对的真机/仿真评估去算 Pearson 相关和 MMRV（平均最大排名破坏）。结论很具体：视觉对上了，仿真里的策略排名才跟真机排名走在一起；只用验证集动作 MSE 排名，相关很差。

**安全。** 这条不是物理通道，是系统通道。即便 $P_{\text{sim}}$ 很准，规划器仍可能选出一条 $J$ 很高、杯子擦着桌沿走的轨迹。专家演示里也有这种险棋。过滤器不改进 $P$，它改的是“哪条 $a$ 被允许离开进程”。

五条通道对应第 9 节那张失败清单。实验里我们能直接拧的是接触（摩擦、质量）和延迟；标定和视觉域用 SIMPLER 的结论讲清楚，本课不重建绿幕管线；安全层本课必须自己写。

### 5.3 域随机化：把未知变成分布

Tobin 等人（arXiv:1703.06907）的原始域随机化针对的是**视觉**：给仿真随机贴纹理，多到真图看起来只是其中一张。Peng 等人（arXiv:1710.06537）把它搬到**动力学**：随机化质量、摩擦、阻尼，让策略在一簇 $P_{\text{sim}}$ 上工作，而不是在某一个 $P_{\text{sim}}$ 上过拟合。

数学上，规划或训练的目标从“在这一组参数 $\theta_0$ 上最优”改成“在分布 $\theta \sim p(\theta)$ 上够好”：

$$
\max_{\pi} \; \mathbb{E}_{\theta \sim p(\theta)}\, \mathbb{E}\!\left[J_{\theta}(\pi)\right]
$$

$p(\theta)$ 应该盖住你对真机参数的不确定性，而不是盖住整个物理宇宙。Genesis 文档写得很直白：能测准的别乱随机化，范围以你的最佳估计为中心、以你的不确定度为宽度。

Genesis 把这件事做成了带 batch 维的 setter。官方示例 [examples/rigid/domain_randomization.py](https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/examples/rigid/domain_randomization.py) 在 `scene.build(n_envs=8)` 之后对 Go2 四足做三件事（文档 [Domain randomization](https://genesis-world.readthedocs.io/en/v1.3.0/user_guide/policy_training/best_practices/domain_randomization.html) 逐行讲解过）：

- `set_friction_ratio`：按 link 乘上摩擦系数，形状 `(n_envs, n_links)`，`1.0` 表示沿用模型原值；
- `set_mass_shift`：按 link 加上质量偏移（千克），加法不是替换；
- `set_COM_shift`：按 link 平移质心。

本课的杯子通常只有一个刚体 link，于是 `n_links=1`。规划世界固定 `friction_ratio=1.0`、`mass_shift=0`；评估世界把这两项打成网格。成功率从规划世界掉到评估世界的那一截，就是你这堂课能量到的间隙。

随机化**不是**把仿真变成真机。它只是让策略别把某一个错误参数当成真理。真机参数若落在 $p(\theta)$ 外面，该摔还是摔。所以随机化之后仍然要安全层。

### 5.4 规划器怎样在仿真里选推杯动作

第 24 课的 Visual Foresight（arXiv:1812.00568）和 PlaNet / DayDreamer 走的是同一张图：用模型展开未来，打分，执行。本课把视频预测器换成 Genesis 刚体步进，动作改成末端在桌面高度上的一段平移。这样做的理由有两条。第一，本课要隔离“间隙”，不能再把网络预测误差和物理间隙搅在一起。第二，Genesis 的 `scene.step()` 就是一个可重置、可改参数的 $P_{\text{sim}}$，正好当规划用的世界模型。

具体的打分。设杯子水平位置为 $p=(x,y)$，目标为 $p^{\star}$，桌面矩形中心 $c$、半边长 $h=(h_x,h_y)$，沿内缩 5 cm 的安全框为

$$
\mathcal{S}_{\text{safe}} = \bigl\{p : |p_x-c_x| \le h_x-0.05,\; |p_y-c_y| \le h_y-0.05\bigr\}
$$

一条候选动作序列的分数是

$$
J = -\|p_H - p^{\star}\|_2 \;-\; \lambda \cdot \mathbf{1}\![p_H \notin \mathcal{S}_{\text{safe}}] \;-\; \mu \sum_{k=1}^{H} \max(0, z_{\text{table}} - z_k)
$$

第一项要杯子靠近目标；第二项惩罚出安全框；第三项惩罚穿模（杯子的 $z$ 掉到桌面以下）。$\lambda$ 取大，安全框在规划阶段就已经是软约束。第 7 节的过滤器会再加一道硬约束：一旦预测的 $p$ 走出 $\mathcal{S}_{\text{safe}}$，后面的动作全部改成“保持当前关节目标”，相当于把剩余地平线截成停。

延迟加进来之后，规划世界若仍按 $d=0$ 搜，评估世界用 $d>0$ 执行，超调会出现在 $J$ 的第一项和第三项上：杯子走过 $p^{\star}$，或者末端还在推的时候杯子已经到沿。第 7 节的滑杆实验就是把 $d$ 和摩擦拧到穿模或落杯发生。

### 5.5 安全过滤器在动作出口，不在训练损失里

过滤器是一个从动作到动作的函数 $f(s, a) \mapsto a'$，满足两条性质。

1. **默认拒绝危险。** 预测不确定时，$a'$ 是停，不是“按原计划再试一步”。第 32 课的“失败承认”会把这句变成桌宠行为；本课先在推杯上落地。
2. **它读预测，不读愿望。** 输入是当前状态加候选动作在模型里展开的未来，输出是截断或限幅后的动作。它不看语言指令是否听起来合理，也不看模仿策略的对数似然有多高。第 26 课写过：VLA 成功不等于理解物理。过滤器就是那句的执行机构。

桌沿 5 cm 是本课写死的阈值，不是从数据里学来的。选 5 cm 是因为：杯子半径大约 3 cm，再留 2 cm 给标定误差和延迟滑移，桌宠工作空间大约 60 cm × 40 cm 时这个边距肉眼可见。真机上还要叠加速度上限和急停按钮，那些放在第 8 节只讲，本课仿真里用“截断为保持”代替急停。

把过滤器塞进训练损失里（比如给掉落加一个很大的负奖励）不够。策略可以在训练分布里学会绕开，到了分布外（更滑的杯、更晚的指令）仍然会走险棋。出口过滤器对所有策略一视同仁：模仿、VLA、MPC、人遥操作，都要过同一道闸。

DayDreamer（arXiv:2206.14176）选择了另一条路：不经过仿真器，直接在四台真机上用 Dreamer 在线学，四足从倒下到走路大约一小时。它回避了 $P_{\text{sim}} \neq P_{\text{real}}$，但把试错代价交给了真机。论文里的机器人有保护，课程里的桌宠没有实验室级的龙门架。所以即便你以后走 DayDreamer 这条路，动作出口仍然要过滤器。本课不复现那四台真机。

## 6. 源码导读

克隆或安装之后，按这个顺序读官方文件。路径以仓库当前 `main` 和文档为准；API 名称以 [Hello, Genesis World](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/hello_genesis.html) 和域随机化页为准。

| 文件或文档 | 是哪一层 | 带着什么问题读 |
|---|---|---|
| [examples/tutorials/hello_genesis.py](https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/examples/tutorials/hello_genesis.py) | 最小仿真 | `gs.init`、`Scene`、`add_entity`、`build`、`step` 这五步各自分配了什么？相对路径 `xml/franka_emika_panda/panda.xml` 从哪里解析？ |
| [examples/tutorials/control_your_robot.py](https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/examples/tutorials/control_your_robot.py) | 关节控制 | `set_dofs_position` 和 `control_dofs_position` 差在哪？为什么力控零指令会让臂倒下？ |
| [examples/tutorials/IK_motion_planning_grasp.py](https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/examples/tutorials/IK_motion_planning_grasp.py) | 末端规划 | `inverse_kinematics` 的 `pos`/`quat` 用哪套坐标？抓取时手指为什么改用力控？ |
| [examples/rigid/franka_cube.py](https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/examples/rigid/franka_cube.py) | 接触 | 官方怎么把臂和方块放进同一场景？ |
| [examples/rigid/domain_randomization.py](https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/examples/rigid/domain_randomization.py) | 随机化 | `set_friction_ratio` 的形状为什么是 `(n_envs, n_links)`？`batch_dofs_info` 什么时候才需要？ |
| [examples/tutorials/parallel_simulation.py](https://github.com/Genesis-Embodied-AI/genesis-world/blob/main/examples/tutorials/parallel_simulation.py) | 并行 | `n_envs>0` 之后 `get_dofs_position` 的 batch 维加在哪？ |
| [刚体实体页](https://genesis-world.readthedocs.io/en/latest/user_guide/physics/rigid_bodies.html) | 材料 | `gs.materials.Rigid` 的 `rho`、`friction` 各改哪件事？桌子为什么必须 `fixed=True`？ |

读的时候盯住三条分界。第一，`set_*` 写状态，绕过物理；`control_*` 发指令，走 PD 和力限制。规划器执行必须用 `control_*`，重置杯子才用 `set_pos`。第二，MJCF 的 Franka 基座怎么连到世界由 XML 决定；URDF（以后选做 SO-101）默认是自由基座，必须 `fixed=True` 才能当桌面臂。第三，域随机化的摩擦、质量 setter 改的是每个环境的状态缓冲，不需要 `batch_dofs_info`；PD 增益的 per-env 随机化才需要那个开关。本课只随机化杯子的摩擦和质量，用不到该开关。

仓库根目录的 README 还列了 FEM、MPM、SPH、布料和 Nyx 渲染。那些是 Genesis 作为通用物理平台的能力，本课故意不碰。用自造的欧拉积分去“模拟”推杯，属于课程禁止项。

## 7. 实验

全程在 Genesis 里做。规划世界用标称摩擦和质量；评估世界拧摩擦、质量、延迟。互动实验在 Step 5：你改两个数，看同一条推杯轨迹何时穿模、何时把杯送出安全框。安全过滤器在 Step 6 装上，没有它不算完成本课。

先建工作目录。后面的脚本都放这里，本节命令都在该目录下执行。

```bash
mkdir -p ~/learn-wm/l27
```

### Step 1: 安装并确认官方示例能转起来

独立环境，先装与你平台匹配的 PyTorch，再装 Genesis World。官方文档和 README 都是这一句（Python `>=3.10,<3.14`）。本课核对日（2026-08-19）PyPI 最新版是 `1.3.3`。

```bash
pip install genesis-world
```

想钉死核对过的版本，改用下一行，不要和上一行同时跑。

```bash
pip install "genesis-world==1.3.3"
```

验证导入。第一次初始化会编译内核，可能要几分钟，之后有缓存。

```bash
python -c "import genesis as gs; gs.init(backend=gs.cpu); print('ok', gs.__file__)"
```

预期：打印 Genesis 的安装路径，没有 `Genesis hasn't been initialized`。若你在 Genesis 源码目录里跑，官方排错写过会触发循环导入，换一个工作目录再试。

克隆仓库只为读示例，不是为了自造引擎。装包和克隆可以并行存在。

```bash
git clone https://github.com/Genesis-Embodied-AI/genesis-world.git
```

跑官方最小例子。无显示器的机器不要开 viewer，直接用文档里那段 headless 循环：加载平面和自带的 Franka MJCF。把下面存成 `hello_desk_check.py` 也可以，内容与官方 `hello_genesis.py` 同类。

```python
import genesis as gs

gs.init(backend=gs.cpu)
scene = gs.Scene(show_viewer=False)
scene.add_entity(gs.morphs.Plane())
scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
scene.build()
for _ in range(50):
    scene.step()
print("hello genesis: 50 steps ok")
```

```bash
python hello_desk_check.py
```

预期：进程正常结束，打印 `hello genesis: 50 steps ok`。第一次 `build()` 慢是在编译内核，中途用 `Ctrl-C` 退出（不要 `Ctrl-\`），缓存才会留下。有 GPU 的机器可以把 `gs.cpu` 换成 `gs.gpu`。Apple Silicon 用 `gs.metal` 或 `gs.gpu`。

本课主路径用 Genesis **自带**的 Franka Panda（`xml/franka_emika_panda/panda.xml`，7 个臂关节加 2 个夹爪）。课纲写过 6 自由度臂或 Reachy Mini 官方模型：Reachy Mini 并不打在 Genesis 的默认资源包里；SO-101 的仿真 URDF 以 LeRobot 文档为准，本课放在 Step 8 当选做。主实验必须能在只装 `genesis-world` 的机器上跑通。

### Step 2: 搭桌子、杯子和臂

把下面存成 `desk_scene.py`。桌子是固定盒子，杯子是可动盒子（几何简单，接触比带把手的网格稳），臂是官方 Franka。尺寸按桌宠工作空间收小：桌面约 70 cm × 50 cm，厚度 4 cm，桌面顶在 z = 0.42 m。

```python
import argparse
import genesis as gs
import numpy as np
import torch


TABLE_POS = (0.55, 0.00, 0.40)
TABLE_SIZE = (0.70, 0.50, 0.04)
CUP_SIZE = (0.06, 0.06, 0.08)
CUP_XY0 = (0.55, 0.10)
DT = 0.01


def table_top_z():
    return TABLE_POS[2] + TABLE_SIZE[2] / 2.0


def cup_spawn_pos():
    return (CUP_XY0[0], CUP_XY0[1], table_top_z() + CUP_SIZE[2] / 2.0 + 0.001)


def make_scene(show_viewer=False, backend="cpu"):
    backend_map = {"cpu": gs.cpu, "gpu": gs.gpu, "metal": gs.metal}
    gs.init(backend=backend_map[backend], logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, gravity=(0.0, 0.0, -9.8)),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.6, -1.4, 1.2),
            camera_lookat=(0.45, 0.0, 0.35),
            camera_fov=40,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    table = scene.add_entity(
        gs.morphs.Box(pos=TABLE_POS, size=TABLE_SIZE, fixed=True),
        material=gs.materials.Rigid(rho=800.0, friction=1.0),
    )
    cup = scene.add_entity(
        gs.morphs.Box(pos=cup_spawn_pos(), size=CUP_SIZE),
        material=gs.materials.Rigid(rho=800.0, friction=1.0),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    scene.build()
    return scene, table, cup, franka


def configure_franka(franka):
    motors = np.arange(7)
    fingers = np.arange(7, 9)
    franka.set_dofs_kp(
        np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100])
    )
    franka.set_dofs_kv(
        np.array([450, 450, 350, 350, 200, 200, 200, 10, 10])
    )
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )
    return motors, fingers


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cpu", choices=["cpu", "gpu", "metal"])
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    scene, table, cup, franka = make_scene(args.viewer, args.backend)
    configure_franka(franka)
    for _ in range(100):
        scene.step()
    p = to_numpy(cup.get_pos())
    print("cup pos after settle:", np.round(p, 4))
    print("table top z:", round(table_top_z(), 4))
```

```bash
python desk_scene.py
```

预期：杯子在桌面上安顿，`cup pos` 的 z 大约 0.46，没有掉到 z=0 的地面上。若杯子直接穿过桌面，先查 `fixed=True` 有没有写上，再把 `dt` 保持在 0.01。开窗口看一眼：

```bash
python desk_scene.py --viewer
```

无显示器就跳过 `--viewer`。这一步的验收是：三样东西都在，杯子停在桌上，臂没有倒进桌板。

### Step 3: 在规划世界里推出一条成功轨迹

规划器很短：末端降到杯侧，沿 $-y$ 推一段，目标是把杯子从 $y=0.10$ 推到 $y=0.00$（桌心附近），并且全程留在安全框内。动作序列在标称物理里展开，选出位移接近 10 cm、且不穿模、不越安全框的那条。把下面存成 `push_planner.py`。

```python
from collections import deque
from desk_scene import (
    CUP_XY0,
    TABLE_POS,
    TABLE_SIZE,
    configure_franka,
    cup_spawn_pos,
    make_scene,
    table_top_z,
    to_numpy,
)
import argparse
import csv
import numpy as np


SAFE_MARGIN = 0.05
TARGET_XY = (TABLE_POS[0], 0.00)
PUSH_HEIGHT = table_top_z() + 0.04
APPROACH = (CUP_XY0[0], CUP_XY0[1] + 0.11, PUSH_HEIGHT)
PUSH_END = (CUP_XY0[0], CUP_XY0[1] - 0.12, PUSH_HEIGHT)
HOME_Q = np.array([0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.8, 0.04, 0.04])
DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])


def edge_margin(xy):
    hx, hy = TABLE_SIZE[0] / 2.0, TABLE_SIZE[1] / 2.0
    mx = hx - abs(xy[0] - TABLE_POS[0])
    my = hy - abs(xy[1] - TABLE_POS[1])
    return min(mx, my)


def reset_episode(scene, cup, franka):
    cup.set_pos(np.array(cup_spawn_pos()))
    franka.set_dofs_position(HOME_Q)
    for _ in range(20):
        scene.step()


def ik_or_home(franka, pos):
    hand = franka.get_link("hand")
    qpos = franka.inverse_kinematics(link=hand, pos=np.array(pos), quat=DOWN_QUAT)
    q = to_numpy(qpos).copy()
    q[-2:] = 0.04
    return q


def apply_physics(cup, friction_ratio, mass_shift):
    n_links = cup.n_links
    cup.set_friction_ratio(
        friction_ratio=np.full((n_links,), friction_ratio, dtype=np.float32),
        links_idx_local=np.arange(n_links),
    )
    cup.set_mass_shift(
        mass_shift=np.full((n_links,), mass_shift, dtype=np.float32),
        links_idx_local=np.arange(n_links),
    )


def rollout(
    scene,
    cup,
    franka,
    motors,
    delay=0,
    friction_ratio=1.0,
    mass_shift=0.0,
    safety=True,
    hold_after_cut=True,
):
    apply_physics(cup, friction_ratio, mass_shift)
    reset_episode(scene, cup, franka)
    q_approach = ik_or_home(franka, APPROACH)
    q_push = ik_or_home(franka, PUSH_END)
    plan = [q_approach] * 80 + [
        q_approach * (1.0 - t) + q_push * t for t in np.linspace(0, 1, 120)
    ]
    delay_buf = deque([HOME_Q.copy()] * delay, maxlen=delay) if delay > 0 else None
    log = []
    cut = False
    for t, q_cmd in enumerate(plan):
        pred = to_numpy(cup.get_pos())[:2]
        if safety and edge_margin(pred) < SAFE_MARGIN:
            cut = True
            if hold_after_cut:
                q_cmd = to_numpy(franka.get_dofs_position())
        if delay_buf is None:
            q_exec = q_cmd
        else:
            delay_buf.append(np.array(q_cmd, dtype=float))
            q_exec = delay_buf[0]
        franka.control_dofs_position(q_exec)
        scene.step()
        p = to_numpy(cup.get_pos())
        log.append(
            {
                "t": t,
                "x": float(p[0]),
                "y": float(p[1]),
                "z": float(p[2]),
                "margin": float(edge_margin(p[:2])),
                "penetr": float(max(0.0, table_top_z() - (p[2] - 0.04))),
                "cut": int(cut),
                "q_exec": q_exec.tolist(),
            }
        )
    last = log[-1]
    moved = np.linalg.norm([last["x"] - CUP_XY0[0], last["y"] - CUP_XY0[1]])
    goal_err = np.linalg.norm([last["x"] - TARGET_XY[0], last["y"] - TARGET_XY[1]])
    on_table = last["z"] > table_top_z() - 0.02
    safe = last["margin"] >= SAFE_MARGIN
    success = bool(moved > 0.04 and on_table and safe and last["penetr"] < 0.01)
    return success, log, {
        "moved": moved,
        "goal_err": goal_err,
        "margin": last["margin"],
        "penetr": last["penetr"],
        "cut": int(any(row["cut"] for row in log)),
        "on_table": int(on_table),
    }


def write_log(path, log):
    keys = ["t", "x", "y", "z", "margin", "penetr", "cut"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in log:
            w.writerow({k: row[k] for k in keys})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cpu", choices=["cpu", "gpu", "metal"])
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--mass-shift", type=float, default=0.0)
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument("--no-safety", action="store_true")
    parser.add_argument("--log", default="plan_nominal.csv")
    args = parser.parse_args()
    scene, table, cup, franka = make_scene(False, args.backend)
    motors, _ = configure_franka(franka)
    ok, log, stats = rollout(
        scene, cup, franka, motors,
        delay=args.delay,
        friction_ratio=args.friction,
        mass_shift=args.mass_shift,
        safety=not args.no_safety,
    )
    write_log(args.log, log)
    print("success", int(ok), "stats", {k: round(float(v), 4) for k, v in stats.items()})
    print("wrote", args.log, "steps", len(log))
```

先在标称世界、零延迟、开着安全层跑一次。

```bash
python push_planner.py --log plan_nominal.csv
```

预期：`success 1`，`moved` 大于 0.04，`margin` 不小于 0.05，`penetr` 接近 0，`cut` 为 0 或 1 都可以，但成功时杯子必须还在桌上。把 `plan_nominal.csv` 留着，这是后面所有对照的规划世界基线。

若 IK 报奇异或臂去撞桌，先把 `HOME_Q` 改成官方 control 教程里能站住的那组，再确认 Franka 基座位姿没被你改过。官方 IK 教程用的末端四元数 `(0, 1, 0, 0)` 是绕世界 X 转 180 度、夹爪朝下，本课沿用。

### Step 4: 随机化摩擦和质量，看成功率怎么掉

规划器仍然按标称世界想好那一条推法（脚本里的 `APPROACH` 到 `PUSH_END` 是固定的，相当于“在脑子里搜完了再执行”）。评估时只改杯子的摩擦比和质量偏移。把下面存成 `sweep_physics.py`。

```python
import csv
from desk_scene import configure_franka, make_scene
from push_planner import rollout
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cpu", choices=["cpu", "gpu", "metal"])
    parser.add_argument("--out", default="sweep_physics.csv")
    args = parser.parse_args()
    frictions = [0.3, 0.6, 1.0, 1.5, 2.0]
    masses = [-0.2, 0.0, 0.4]
    scene, table, cup, franka = make_scene(False, args.backend)
    motors, _ = configure_franka(franka)
    rows = []
    for mu in frictions:
        for dm in masses:
            ok, log, stats = rollout(
                scene, cup, franka, motors,
                delay=0,
                friction_ratio=mu,
                mass_shift=dm,
                safety=False,
            )
            row = {"friction": mu, "mass_shift": dm, "success": int(ok), **stats}
            rows.append(row)
            print(row)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    n_ok = sum(r["success"] for r in rows)
    print("success rate", n_ok, "/", n, "=", round(n_ok / n, 3))


if __name__ == "__main__":
    main()
```

这一次故意关掉安全层，专门量物理间隙。

```bash
python sweep_physics.py --out sweep_physics.csv
```

预期：标称格 `(1.0, 0.0)` 接近成功；摩擦降到 0.3 时杯子滑得更远，`margin` 变小甚至变负；质量加上 0.4 kg 时同样推力推不动，`moved` 变小。把 15 个格子抄进笔记，成功率相对标称世界的掉落就是本课的主数字。格子不必和论文数字看齐，方向要对：更滑更容易出界，更重更推不动。

若所有格子都失败，回到 Step 3 把标称轨迹修到成功，再扫。一次改一组物理参数，不要同时开延迟。

### Step 5: 互动。拧摩擦和延迟，看轨迹何时穿模

这是本课的互动实验。同一条规划好的推法，你只改 `--friction` 和 `--delay`，脚本把每一步的 $x,y,z$ 和桌沿余量写进 CSV。延迟的单位是控制步，`dt=0.01` s，所以 `--delay 8` 是 80 ms。

先看标称：

```bash
python push_planner.py --friction 1.0 --delay 0 --no-safety --log traj_mu1_d0.csv
```

再把摩擦拧到很滑：

```bash
python push_planner.py --friction 0.2 --delay 0 --no-safety --log traj_mu02_d0.csv
```

再把延迟拧到 80 ms，摩擦回到 1.0：

```bash
python push_planner.py --friction 1.0 --delay 8 --no-safety --log traj_mu1_d8.csv
```

最后两个一起拧：

```bash
python push_planner.py --friction 0.2 --delay 8 --no-safety --log traj_mu02_d8.csv
```

画图需要 matplotlib。没有就先装。

```bash
pip install matplotlib
```

把下面存成 `plot_traj.py`，用来看穿模和出界发生在第几步。

```python
import argparse
import csv
import matplotlib.pyplot as plt


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    t = [int(r["t"]) for r in rows]
    y = [float(r["y"]) for r in rows]
    z = [float(r["z"]) for r in rows]
    m = [float(r["margin"]) for r in rows]
    p = [float(r["penetr"]) for r in rows]
    return t, y, z, m, p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    args = parser.parse_args()
    t, y, z, m, p = load(args.csv_path)
    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    ax[0].plot(t, y)
    ax[0].axhline(0.00, linestyle=":", linewidth=1)
    ax[0].set_ylabel("cup y (m)")
    ax[1].plot(t, z)
    ax[1].set_ylabel("cup z (m)")
    ax[2].plot(t, m, label="edge margin")
    ax[2].plot(t, p, label="penetration")
    ax[2].axhline(0.05, linestyle=":", linewidth=1)
    ax[2].set_ylabel("m / penetr (m)")
    ax[2].set_xlabel("step")
    ax[2].legend()
    fig.tight_layout()
    out = args.csv_path.replace(".csv", ".png")
    fig.savefig(out, dpi=120)
    print("wrote", out)
    print("min margin", round(min(m), 4), "max penetr", round(max(p), 4))


if __name__ == "__main__":
    main()
```

```bash
python plot_traj.py traj_mu1_d0.csv
```

```bash
python plot_traj.py traj_mu02_d0.csv
```

```bash
python plot_traj.py traj_mu1_d8.csv
```

```bash
python plot_traj.py traj_mu02_d8.csv
```

对照时盯三件事。`y` 是否冲过目标很多（超调，延迟典型症状）。`z` 是否掉到桌面以下（穿模，接触求解在这一组速度/摩擦下没守住）。`margin` 是否跌破 0.05 甚至变负（杯子中心到桌沿不足 5 cm，再滑就要掉下去）。把四张图里第一次跌破 5 cm 的步号记下来。摩擦和延迟不是对称的：低摩擦拉长滑动；延迟让力在错误的时刻加上去，接触速度更高，穿模更容易出现。

网站若提供本课的滑杆组件，用它做同一件事：摩擦、延迟两个滑杆，推杯轨迹三条（标称、当前、安全截断后）。组件没上线就用上面四条命令，效果相同。

### Step 6: 装上桌沿 5 cm 过滤器，留下规划记录

同一组会失败的参数，把安全层打开。过滤器读的是**当前**杯子位置的桌沿余量；余量小于 5 cm 就把后续指令改成保持。这是保守的硬闸门，不依赖再训一个网络。

```bash
python push_planner.py --friction 0.2 --delay 8 --log plan_safe_mu02_d8.csv
```

对照：

```bash
python push_planner.py --friction 0.2 --delay 8 --no-safety --log plan_unsafe_mu02_d8.csv
```

预期：无安全层的那条 `cut=0`，`margin` 更小，成功率常常是 0；有安全层的那条 `cut=1`，杯子停在安全框里或框沿，`success` 按 Step 3 的定义可能仍是 0（没推到目标），但这正是过滤器的工作：目标没到，杯子也没掉。把两份 CSV 放进笔记，写三行：何时第一次 `cut` 变成 1、截断后 `margin` 的最小值、无安全层时 `margin` 的最小值。这份对比就是验收要的“带安全过滤的规划记录”。

过滤器读的是评估世界里的杯子位置。真机上你没有这颗精确的 `get_pos()`，要用第 29 课的感知去估。本课在仿真里用真位置，是为了先把闸门的逻辑跑对；把估计误差再叠进去，是第 29 课的事。出口过滤器仍然要按估得更差的方向收紧边距，而不是按估得更乐观的方向放松。

### Step 7: 写失败清单，并给每一项一个本课证据

把下面复制到 `NOTES.md`，用你自己的数字填空。五项是课纲规定的最低集，后面两项是本课多看到的。

```text
日期 / 机器 / genesis-world 版本 / 后端（cpu/gpu/metal）
规划世界：friction_ratio=1.0, mass_shift=0, delay=0, success=?
评估网格：sweep_physics.csv（成功率 = ）
延迟 80 ms 无安全：min margin=  max penetr=
延迟 80 ms 有安全：第一次 cut 的步号=  截断后 min margin=
失败清单（每一项用本课的文件或论文页码顶上）：
1. 标定：本课未在真机上测；对应 Peng arXiv:1710.06537 的标定误差讨论，以及 SIMPLER 用离线轨迹做 PD SysID。
2. 延迟：traj_mu1_d8.csv 相对 traj_mu1_d0.csv 的超调。
3. 接触：sweep_physics.csv 的低摩擦格；穿模看 max penetr。
4. 视觉域：本课未改渲染；对应 SIMPLER 绿幕与纹理匹配，以及“验证 MSE 不能替真机排名”。
5. 安全：plan_safe_mu02_d8.csv 对 plan_unsafe_mu02_d8.csv。
6. 质量：sweep_physics.csv 的 mass_shift 列。
7. 控制增益：本课沿用官方 Franka kp/kv，未做 SysID；真机上必须重做。
结论一句：没有安全层，本课轨迹不得接到真机。
```

### Step 8: 选做。换 SO-101 的 URDF，不换实验协议

Genesis 加载 URDF 的官方方式是 `gs.morphs.URDF(file=..., fixed=True)`。SO-101 的仿真 URDF 以 LeRobot 的 [SO-101 文档](https://huggingface.co/docs/lerobot/en/so101) 和 TheRobotStudio 的 SO-ARM100 仓库为准，不要从课程里发明一份。加载后，IK 的末端 link 名、关节数、kp/kv 都要按那份 URDF 改，不能沿用 Franka 的 9 个 dof。选做的及格标准与主实验相同：标称能推、摩擦网格能掉、安全层能截。Reachy Mini 是第八幕的桌宠档身体，本课不把它接进 Genesis 场景，避免把没有随 Genesis 打包的网格写成“官方模型”。

## 8. 配置与预算

本课几乎不训练网络。预算花在首次内核编译和摩擦网格的反复步进上。

| 档位 | 做什么 | 后端 | 墙钟时间（参考） | 用途 |
|---|---|---|---|---|
| 冒烟 | Step 1-3，标称推杯一次 | CPU 或 Metal | 首次编译 5-20 分钟，之后单条轨迹一两分钟 | 确认场景和 IK |
| 本课必做 | Step 4-7，15 格物理扫描 + 4 条延迟对照 + 安全开关 | CPU 足够；GPU 更快 | 扫描半小时量级 | 成功率表和规划记录 |
| 选做 SO-101 | 换 URDF，重调 kp/kv 和末端 link | 同必做 | 额外半天，主要花在 URDF 和 IK | 对齐第八幕 L 档身体 |
| 只讲、不跑 | SIMPLER / LIBERO / DayDreamer 精读 | 无 | 论文半天 | 真机理论 |

超参数以脚本默认值为准，不要扫出第二套“更好看”的增益再拿来当结论。

| 量 | 本课取值 | 为什么这样取 |
|---|---|---|
| `dt` | 0.01 s | 官方 control / IK 教程同款 |
| 桌面 | 0.70 m × 0.50 m × 0.04 m，顶面 z = 0.42 m | 接近桌宠工作空间，边缘肉眼可见 |
| 安全边距 | 5 cm | 课纲写死；杯半宽约 3 cm，余量留给误差 |
| 摩擦比网格 | 0.3, 0.6, 1.0, 1.5, 2.0 | 盖住“更滑”和“更黏”，1.0 是规划世界 |
| 质量偏移 | -0.2, 0, +0.4 kg | 加法偏移，官方 `set_mass_shift` 的语义 |
| 延迟 | 0 / 8 步 | 0 对 80 ms，覆盖一帧曝光加一次推理的量级 |
| Franka kp/kv | 官方 control 教程那组 | 本课不做 SysID；真机上必须重测 |

真机相关的预算只写边界，避免装成可跑。相机标定、关节零位、末端力限制、急停按钮，每一项都是第八幕接硬件时的单独工作。DayDreamer 在真机上用大约一小时学会四足站立，那是实验室保护下的在线 RL，不是本课作业。LIBERO 的 130 个任务、OpenVLA 在 LIBERO 上的数字，都是仿真基准，不能当成你桌子上的成功率。

Genesis 文档里的并行环境（`scene.build(n_envs=B)`）可以把 Step 4 的 15 格变成一次 batch。有 GPU 时值得改；CPU 上收益小，先串行扫完更省事。`set_friction_ratio` 在 `n_envs>0` 时要 `(n_envs, n_links)`；单环境无 batch 维时改成长度为 `n_links` 的一维数组。形状报错先查这一条。

## 9. 验收

验收清单：

- [ ] 能在白纸上画出规划世界、评估世界、真桌子三层，并标出安全过滤器卡在动作出口；
- [ ] `hello_desk_check.py` 或官方 `hello_genesis.py` 跑通，用的是 `genesis-world` 包，不是自写积分器；
- [ ] `desk_scene.py` 里杯子安顿在桌面上，z 没有掉到地面；
- [ ] 标称世界推杯 `success=1`，`plan_nominal.csv` 在证据目录；
- [ ] `sweep_physics.csv` 至少 15 格，标称格和低摩擦格的成功率或 `margin` 有可见差异；
- [ ] 四条摩擦/延迟轨迹图都在，能指出哪一条开始穿模或跌破 5 cm；
- [ ] `plan_safe_mu02_d8.csv` 对 `plan_unsafe_mu02_d8.csv`：安全层触发后余量不再继续变差，或杯子未掉下桌；
- [ ] `NOTES.md` 的失败清单至少含标定、延迟、接触、视觉域、安全五项，每项有本课文件或核对过的论文支撑；
- [ ] 口头关：向没上过这门课的人解释“为什么仿真成功不能发到真机”，必须说到安全层，且明确本课没有真机实验。

成功的定义写死如下，改定义必须在笔记里声明，不能事后为了让表更好看再改。

| 字段 | 通过条件 |
|---|---|
| `moved` | 杯子水平位移 > 4 cm |
| `on_table` | 结束时杯心 z 高于桌面减 2 cm |
| `safe` | 结束时桌沿余量 ≥ 5 cm |
| `penetr` | 最大穿模深度 < 1 cm |
| `success` | 以上四条同时成立 |

安全层触发导致没推到目标、因而 `success=0`，算过滤器工作正常，不算实验失败。实验失败是：无安全层时杯子掉了你却没记下来，或有安全层时余量已经变负过滤器仍未截断。

## 10. 排错

| 症状 | 最可能的原因 | 怎么确认 | 怎么修 |
|---|---|---|---|
| `Genesis hasn't been initialized` | 在 `gs.init()` 前 import 了 engine 子模块 | 对照官方 Installation 排错段 | 先 `import genesis as gs` 再 `gs.init()`，子模块放到 init 之后 |
| 在仓库源码目录里 `import genesis` 循环导入 | 非 editable 安装且当前目录就是源码树 | 官方 Installation 写过这条 | 换到 `~/learn-wm/l27` 再跑，或按 README 改成 `pip install -e ".[dev]"` |
| 第一次 `scene.build()` 极慢 | 新场景在即时编译内核 | 终端有编译日志 | 正常；用 `Ctrl-C` 退出以留下缓存，不要 `Ctrl-\` |
| 杯子穿过桌面掉到地上 | 桌子没 `fixed=True`，或 `dt` 太大，或生成位置嵌进桌板 | 打印桌面顶 z 和杯心 z | 确认 `fixed=True`；杯底略高于桌面；`dt=0.01` |
| `set_friction_ratio` 报形状错 | 单环境传了 `(1, n_links)`，或并行环境传了 `(n_links,)` | 打印 `scene.n_envs`、`cup.n_links` | 无 batch 用一维；`n_envs>0` 用 `(n_envs, n_links)` |
| IK 给出扭曲姿态或报失败 | 目标在工作空间外，或四元数不是朝下 | 先把目标打到官方 IK 教程的 `(0.65, 0, 0.25)` 看能否复现 | 缩短推程；保持 `DOWN_QUAT=(0,1,0,0)`；先 `set_dofs_position` 到 `HOME_Q` |
| 臂一开跑就倒 | 用了 `set_dofs_position` 当控制，或力指令为零 | 对照 control 教程：力控零指令会让臂倒下 | 执行阶段只用 `control_dofs_position`，并设置官方 kp/kv |
| 所有摩擦格都 `moved≈0` | 末端高度不对，根本没碰到杯 | 看 viewer 或打印末端 z 与杯心 z | 把 `PUSH_HEIGHT` 降到杯高一半附近，不要插进桌面 |
| 所有摩擦格都把杯扫下桌 | 推程太长，目标已经贴沿 | 看 `TARGET_XY` 和桌半宽 | 缩短 `PUSH_END` 的 y；先在标称世界拿到 `success=1` |
| 延迟实验看不出差别 | `delay` 相对推程太短，或缓冲实现成了零延迟 | 打印 `q_cmd` 与 `q_exec` 是否错开 | 用 `--delay 8` 或更大；确认 `deque` 先 append 再取左端 |
| 安全层一开臂突然缩回原点 | 截断时误发了 `HOME_Q` | 看 `cut==1` 之后的 `q_exec` | 截断后保持当前关节目标，见下面补丁 |
| Mac 上 viewer 黑屏或极慢 | 渲染落到软件实现 | 官方 Installation 的 GPU / EGL 段 | 本课用 `show_viewer=False` 加 CSV，不依赖窗口 |
| 想用 SO-101 但臂从桌面掉下去 | URDF 自由基座，没 `fixed=True` | 官方 Hello 页写明 URDF 默认 free | `gs.morphs.URDF(file=..., fixed=True)` |

安全层保持当前指令的写法：`cut` 为真时不要发 `HOME_Q`，发当前关节位置。

```python
if safety and edge_margin(pred) < SAFE_MARGIN:
    cut = True
    q_cmd = to_numpy(franka.get_dofs_position())
```

## 11. 前沿与改造

同一问题，2024-2026 年公开系统大致走三条路，本课每条都对得上一段精读，但不把任何一条装成“已经在你桌子上复现”。

第一条，把仿真做成可靠的评测器，而不是可靠的训练场。SIMPLER（Li 等，arXiv:2405.05941，CoRL 2024）针对的是**已经在真机上训好的**操作策略，问：能不能在仿真里排出和真机一样的名次。他们把间隙拆成控制间隙和视觉间隙。控制侧用离线演示轨迹做 PD 参数的系统辨识，让开环末端轨迹贴上真机；视觉侧用绿幕把真背景贴进仿真、把真纹理烤到物体上。配对评估之后，Visual Matching 在 Google Robot 任务上的平均 Pearson $r$ 大约 0.92，MMRV 大约 0.056；用验证集动作 MSE 来排名则差很多。本课 Step 4 的摩擦网格是这条路的动力学缩版：不追求绝对成功率等于真机，追求相对排序还说得通。差距也清楚：SIMPLER 做了绿幕和纹理匹配，本课一步没做，所以视觉域只能写进失败清单，不能写进你的表。

第二条，把仿真当成终身学习的任务工厂。LIBERO（Liu 等，arXiv:2306.03310）提供 130 个语言条件操作任务、四套任务组和人工遥操作演示。OpenVLA 等模型后来把 LIBERO 当成公开分数板。它测的是任务之间的知识迁移，不是 $P_{\text{sim}}$ 对 $P_{\text{real}}$。第 26 课用它当 VLA 的尺子；本课要记住：LIBERO 高分仍然是仿真高分。把 LIBERO 成功率抄到桌宠立项文档里当“真机已经会了”，属于张冠李戴。

第三条，干脆不经过仿真器。DayDreamer（Wu, Escontrela, Hafner 等，arXiv:2206.14176）把 Dreamer 直接放到四台真机上在线学：四足约一小时从倒下到走路，被推了还能在约十分钟内适应；机械臂从图像和稀疏奖励学抓放。同一组超参打四台机器。这条路回避了本课的 $P_{\text{sim}} \neq P_{\text{real}}$，把代价换成真机试错。第 24 课已经交代过：课程不复现那四台机器人。本课补一句：真机在线学习仍然要动作出口的速度限制、碰撞预测和急停。没有这三样，桌宠的“一小时学习”就是一小时扫杯子。

规模差距：SIMPLER 的配对真机评估、LIBERO 的 130 任务、DayDreamer 的四台真机，都不是单卡周末作业。机制差距：本课的摩擦/质量随机化、延迟缓冲、桌沿过滤器，和 SIMPLER 的 SysID、Peng 的动力学随机化、第 24 课的想象规划是同一套零件，单机全部摸得到。

动手改造清单（选做，每个写预算和失败判据）：

1. 把规划器从固定推程改成第 05 课那种 CEM：在标称 Genesis 里采样若干末端位移，展开 $H$ 步，按 5.4 的 $J$ 打分，执行第一步。改 `push_planner.py` 的 `plan` 生成段。预算：CPU 半天。预期：标称世界成功率不低于固定推程，低摩擦格掉得更少。失败判据：标称世界反而不如固定推程，先查 $J$ 有没有把穿模项写进去。
2. 并行扫描。按官方 `parallel_simulation.py` 把 Step 4 改成 `n_envs=15`，一次 `set_friction_ratio` 喂完整张网格。预算：有 GPU 一两小时，含把 setter 形状改成 `(n_envs, n_links)`。预期：结论方向与串行扫描一致，墙钟明显下降。失败判据：并行与串行的 `margin` 系统性对不上，查 `env_spacing` 是否被误当成物理偏移（文档写明它只影响可视化）。
3. 视觉域最小探针。给场景加一台 `scene.add_camera`，对同一条轨迹渲染 RGB，用第 03 课的动作对换逻辑：把图像里的杯子位置当观察，故意改光照或桌面颜色，看一个线性回归器从图像估 $y$ 会偏多少。预算：半天，不训大模型。预期：改颜色后估计偏差明显变大。失败判据：偏差几乎不变，记录下来，说明这条渲染路径对你的探针不敏感，不能据此声称“视觉域已对齐”。
4. 安全框消融。把 `SAFE_MARGIN` 改成 1 cm、5 cm、10 cm，在 `--friction 0.2 --delay 8` 上各跑 10 次（每次把杯的初始 y 加一点噪声）。预算：一两小时。预期：1 cm 仍会掉杯，10 cm 几乎推不动，5 cm 是“目标偶尔达不到、杯子几乎不掉”的折中。失败判据：5 cm 仍然掉杯且穿模项很大，把边距加到 8 cm 并写进笔记，不要为了表好看把掉落改判成功。

Peng 等人“动力学随机化能让推物策略在真机上保持相近表现”对应改造 1+2 的方向：分布上规划，比单点规划抗参数误差。SIMPLER“视觉匹配优于只做外观随机化”对应改造 3：本课缩小版只能看到“外观一变估计就偏”，看不到绿幕带来的排名恢复，所以改造 3 的结论止步于“视觉域存在”，不要写成“已经对齐”。

## 12. 论文与延伸

1. SIMPLER（Li, Hsu, Gu 等，[arXiv:2405.05941](https://arxiv.org/abs/2405.05941)），仓库 [simpler-env/SimplerEnv](https://github.com/simpler-env/SimplerEnv)。本课真机理论的主文。带着三个问题读：控制间隙和视觉间隙各用什么办法补，为什么他们说不必造完整数字孪生？MMRV 相对 Pearson $r$ 多惩罚了哪类“排名错但分差很小”的情况？Visual Matching 和 Variant Aggregation 谁更贴近真机排名，原因写在哪一节？
2. LIBERO（Liu, Zhu, Gao 等，[arXiv:2306.03310](https://arxiv.org/abs/2306.03310)），仓库 [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)。带着问题读：130 个任务测的是陈述性知识还是程序性知识，还是两者的混合物？论文发现顺序微调在前向迁移上好过现成终身学习方法，这对“在仿真基准上刷 VLA 分数”意味着什么？为什么这份基准再高，也不能替代本课的失败清单？
3. DayDreamer（Wu, Escontrela, Hafner 等，[arXiv:2206.14176](https://arxiv.org/abs/2206.14176)），第 24 课主锚的真机段落。带着新问题读：四台机器人共用同一组超参，说明世界模型的哪一部分对身体细节不敏感？他们选择不经过仿真器，回避了本课哪一条通道，又把哪一种风险留给了真机？课程为什么仍然禁止把这条路写成“无安全层也可以上桌宠”？
4. Genesis World 技术说明。安装与 API 以当前文档为准：[Installation](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html)、[Hello, Genesis World](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/hello_genesis.html)、[Domain randomization](https://genesis-world.readthedocs.io/en/v1.3.0/user_guide/policy_training/best_practices/domain_randomization.html)。仓库引用用官方 bibtex（文档在技术报告发布前给出的条目）：Genesis Authors, *Genesis: A Generative and Universal Physics Engine for Robotics and Beyond*, 2024, https://github.com/Genesis-Embodied-AI/genesis-world 。带着问题读：`set_*` 和 `control_*` 的分界如何保护你，不至于在规划时把杯子瞬移到目标？URDF 为什么默认自由基座？
5. 动力学随机化（Peng, Andrychowicz, Zaremba, Abbeel，[arXiv:1710.06537](https://arxiv.org/abs/1710.06537)）。带着问题读：他们随机化了哪些量，哪些量保持固定？文中关于标定误差的实验，和本课失败清单第一项怎么互证？
6. 视觉域随机化（Tobin 等，[arXiv:1703.06907](https://arxiv.org/abs/1703.06907)）。带着问题读：这篇随机化的是纹理不是摩擦，为什么后来 SIMPLER 还要做绿幕，而不是把纹理随机化进行到底？
7. 选读：Visual Foresight（Ebert 等，[arXiv:1812.00568](https://arxiv.org/abs/1812.00568)），回访第 24 课。问：像素预测规划在真机上能工作，依赖了哪些本课清单里的项（标定、安全、接触）被实验者用工程手段按住了？

对照工具，不必当本课主实验。MuJoCo Playground 的安装命令是 `pip install playground`（[google-deepmind/mujoco_playground](https://github.com/google-deepmind/mujoco_playground)）。ManiSkill 是另一套操作仿真与任务库。本课主锚保持 Genesis World，避免三套仿真器各跑一遍却说不清间隙。

到这里，第七幕收官。循环仍然是观察、状态、按动作预测、展开、打分、选动作；本课在“选动作”和“真的发出去”之间加了一道闸，并证明闸门前面的预测器无论多准，用的都可能是另一个世界。下一课第 28 课把工作空间收成一张桌子，立项桌宠：有界、物体少、人是主要外生过程、安全约束写死、每天能采集。没有本课这张失败清单和过滤器，第八幕不准接真机；有摄像头的人，用屏幕当脸也能把过滤器接到“先想再动”上。
