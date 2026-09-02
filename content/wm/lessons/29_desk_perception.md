---
id: 29_desk_perception
title: "把桌子看成状态"
summary: "桌宠的状态至少要有物体、人的注意、自己的身体。像素和关节角怎样变成这三样？"
unit: deskpet
play_tools: []
checkpoints:
  - "至少 1 分钟的状态日志。"
  - "观察与推断的区分说明。"
---

# 第 29 课：把桌子压成可查询的状态

> 类型：实战（跑公开权重与官方视觉任务库，不从头训练）<br>
> 建议周期：2-3 天<br>
> 硬件：MediaPipe 日志与直播叠加 Mac / CPU 必做；VGGT 抽帧与 CUT3R 恒常检查走单张 24GB 卡。CUT3R 官方 demo 依赖 CUDA RoPE 核，无 NVIDIA 卡改走二维恒常规则<br>
> 锚定：第 28 课自己的 2 分钟桌面视频；[facebookresearch/vggt](https://github.com/facebookresearch/vggt)（权重 `facebook/VGGT-1B`）；[CUT3R/CUT3R](https://github.com/CUT3R/CUT3R)；[MediaPipe Face Landmarker Python 指南](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python)（页面标注 Last updated 2026-08-17）与同套 Tasks 的 [Object Detector](https://developers.google.com/edge/mediapipe/solutions/vision/object_detector/python)<br>
> 产物：至少 1 分钟 jsonl 状态日志、观察与推断对照表、物体恒常灯记录、叠加杯子框 / 人脸框 / 注意箭头的直播或回放

## 1. 这一课做什么

你现在在第八幕第二课。第一幕到第七幕已经把循环从 CarRacing 接到真机安全层：先压
状态，再按动作想下一步，再展开、打分、选动作。第 28 课把世界收成一张桌子，交了
POMDP 表、S/M/L 档位、一段 2 分钟桌面视频。那段视频还是观察：一串像素。桌宠不能
拿像素去想「伸手会不会碰杯」，因为它还没有一份可以查询的现在。本课只换主干循环
里「观察先压成状态」这一截。第六幕的 VGGT 与 CUT3R、[第 15 课](./15_vjepa2_in_practice.md) 的探针、
[第 23 课](./23_object_centric_wm.md) 的槽，在这里第一次同时接到同一张桌子上。

整门课的循环没变：

```text
观察(像素 / 关节 / 人脸朝向)
  先压成状态(物体 / 人的注意 / 自己的身体)
  再按「当前状态 + 动作」预测下一状态
  然后展开多条可能的未来
  给这些未来打分(会不会碰杯、人会不会看过来)
  最后选动作，或明确说「不确定，先别动」
```

第 28 课已经把最小状态钉成三块：桌上的物体、对面那个人的注意、自己的身体。本课
把这三块从符号变成可以逐帧写出的字段。物体走检测框，有 GPU 再用第 20 课的 VGGT
和第 21 课的 CUT3R 把框抬到三维，并用物体恒常灯检查「人伸手挡住杯子」。人的注意
走官方 Face Landmarker：478 个三维关键点、可选的 52 个 blendshape、一张 4×4 面部
变换矩阵。自己的身体，M 档读 Reachy Mini 关节；S 档用键盘假头部，键是真实控制
信号，身体是假的。

做完你能拿出三样东西。第一，一份至少 1 分钟的时间表，杯子在不在画面、人脸框、
人是否看镜头、头朝哪、自己的头朝哪，按时间排下来。第二，一张对照表，写清哪些
数字是传感器当时吐出的观察，哪些是你用规则或三维状态推出来的推断。第三，一段
直播或回放：画面上叠杯子框、人脸框和一根注意箭头，你转头或挡杯子，状态栏要跟上。

没有这一层，第 30 课会在像素上训 $P(s_{t+1}\mid s_t,a_t)$，模型背的是纹理，不是
杯子和注视。第 31 课若已有本课的 `look`、`face_on`、`cup_on` 字段，可以直接复用
日志再补互动段；没有本课，第 31 课会按自己的脚本现采，两课字段必须同向，阈值
不许各写一套。

档位先说死：本课是实战。VGGT、CUT3R、Face Landmarker、Object Detector 全部跑
公开权重，禁止从头训练这些模型。禁止把一帧丢给多模态语言模型，让它写「桌上很乱」
当作状态。那句话既不能对换动作，也不能在你挡住杯子之后给出可复现的字段。

术语速查：

| 术语 | 一句人话 |
|---|---|
| 观察 | 这一帧真正记录到的信号：像素、检测框、关键点坐标、关节角、按键 |
| 状态 | 根据历史估出的当前世界：杯还在桌左后、人正看镜头、自己的头朝左 |
| 推断 | 观察之上的规则或几何结论，例如「看镜头」「被挡住但仍在」 |
| Face Landmarker | MediaPipe Tasks 里的人脸网格任务：478 点、可选 blendshape 与 4×4 变换矩阵 |
| 面部变换矩阵 | 把标准脸变到当前检测脸上的 4×4 矩阵，旋转部分用来读头部朝向 |
| 看镜头 | 本课的操作定义：偏航、俯仰绝对值都小于 20°，且左右 `eyeLookDown` 平均小于 0.35 |
| 物体恒常 | 东西被挡住或暂时出画，状态里它仍在；真被拿走，状态才改 |
| 假头部 | S 档用键盘代替关节；日志里必须标明这是命令，不是编码器读数 |
| 探针 | 冻结大模型，只读一个小头；本课的朝向规则就是几何探针，第 15 课是学习探针 |
| 实战 | 跑公开权重做抽取与可视化，不从头训练 |

## 2. 问题

桌宠的摄像头每秒吐出几十万个像素，关节再加几个角。这些都是观察。第 01 课就立过
规矩：观察不等于世界的完整状态。放到桌子上，这句话变成三个具体坑。

第一，当前帧里没有杯子，不等于桌上没有杯子。人伸手挡住两秒，检测器会丢框；头
转开两秒，杯子离开画面。若状态等于「这一帧检到了什么」，桌宠一转头就以为桌是空
的，第 32 课的「别扫杯」没有查询对象。第 21 课的物体恒常灯就是为这一坑准备的。

第二，脸上有 478 个点，还不等于「人在看我」。点是观察。看镜头是推断：本课用头部
朝向加眼部下看系数做成一条可复现的规则，并写明它不是眼动仪。头正对相机但眼睛
大幅下看，多半在看手机；头转开但眼睛还瞄着镜头，仍可能在看你。第 31 课要预报
1 秒后同一事件，本课先把「现在」估稳。两边必须用同一套阈值，否则第 31 课在拟合
你这课的漂移。

第三，身体不在画面里。S 档没有关节编码器，键盘假头部是唯一的本体感觉通道。按
「看左」却不转镜头，日志会撒谎。M 档用 `get_current_joint_positions()` 读头 7 维、
天线 2 维（弧度），那才是观察。两档的状态字段同名，来源不同，必须在日志里写清。

本课要解决的题目因此收成一句：从第 28 课的视频里提出杯子位置、是否在画面、人脸
框、是否看镜头、相机或头朝向，做成时间表，再用恒常检查挡住杯子的那几秒。互动
是同一套管线对着摄像头或回放视频画框。成功标准不是「看起来懂桌子」，是字段能
逐帧对上你做的事，并且观察栏和推断栏分得开。

一条红线单独写：用大语言模型看图写场景描述，不算状态。描述没有时间戳对齐的
几何，不能在挡住杯子时给出 `cup_still=1`，也不能在第 30 课被当成 $s_t$ 做动作
对换。同一帧你可以让模型写出「桌上很乱、杯子大概在左边」，换一只随机种子它会
写出另一句，两句话都不能和 `events.csv` 里的挡住秒数对账。本课的字段必须让
另一台机器按同一脚本跑出同一份 jsonl（检测分数允许有少量抖动，规则字段必须
逐行可复现）。

## 3. 准备

- [第 28 课](./28_desk_pet_brief.md) 的 2 分钟桌面视频。剧本应覆盖静物、挪杯、
  手挡杯子约两秒、注视从镜头切到屏幕或手机、笔的小位移。没有这段视频就按同一
  剧本现在拍，真人真手，不要用电影对视镜头顶上。第 21 课若已拍过 `occlude.mp4`
  / `remove.mp4` / `turn.mp4`，本课恒常检查直接复用。第 15 课的探针和第 23 课
  的槽作为阅读背景即可，本课不重新训练它们。
- [第 21 课](./21_persistent_4d.md) 的 CUT3R 环境（`cut3r` conda 环境、已编译的
  CroCo RoPE 核、`src/cut3r_512_dpt_4_64.pth`）。没有 GPU 或不想重装，本课 3D
  恒常改为选做，二维恒常规则仍是必做。
- [第 20 课](./20_spatial_3d_state.md) 的 VGGT 环境若还在，抽 8 到 12 帧做一次
  三维抬升。仓库 `requirements.txt` 钉过 `torch==2.3.1`，不要和 Dreamer 或 CUT3R
  混装。没有这套环境，本课允许只交二维状态，笔记里写明未做三维抬升。
- Python 3.10 到 3.12 的独立虚拟环境，给 MediaPipe 专用。本课核验过的安装命令
  是官方 Python 指南里的 `python -m pip install mediapipe`，另装 `opencv-python`
  和 `numpy`。不要去克隆 MediaPipe 那个巨型 C++ 单仓。
- 磁盘：Face Landmarker 任务包大约 3 到 4 MB，EfficientDet-Lite0 数 MB；2 分钟
  1080p 视频通常几百 MB；VGGT 权重按 fp32 估大约 5GB 量级；CUT3R 512 档 checkpoint
  是 GB 量级。工作目录建议 `~/learn-wm-l29/`。
- M 档：第 28 课已经跑通的 `reachy-mini-daemon`，本课只读
  `get_current_joint_positions()` 和 `mini.media.get_frame()`，不写新的运动技能。
- 桌上固定一只检测器认得出的杯子或水瓶。纸杯、马克杯、塑料瓶都可以，开采前先
  在窗口里确认类别名是 `cup` 或 `bottle`。深色异形杯经常被漏检，换一只比调阈值
  便宜。

## 4. 学习目标

1. 白纸写出桌面状态的三块：物体、人的注意、自己的身体。对每一块指出输入是观察
   还是推断，以及缺了它第 30 课会背什么。
2. 用官方 Tasks API（VIDEO 模式、`detect_for_video`）从视频抽出人脸框、478 点、
   4×4 变换矩阵和杯子框，不调用已停用的 `mp.solutions.face_mesh`。
3. 按本课与第 31 课共用的规则标出「是否看镜头」，抽 20 帧肉眼核对，错一半以上
   必须改阈值并重跑，不许只改报告措辞。
4. 做成至少 1 分钟 jsonl 时间表，观察字段和推断字段分栏；S 档写入键盘假头部，
   M 档写入真实关节。
5. 完成一次物体恒常检查：挡住杯子约两秒，检测器可以丢框，状态里「杯子仍在」
   必须还能亮；真把杯子端走，同一盏灯必须灭。有 CUT3R 用第 21 课的三维盒子，
   没有就用二维滞回。
6. 对着摄像头或回放视频叠加杯子框、人脸框、注意箭头，转头和挡杯时状态栏跟上。

## 5. 原理

五个机制。每个仍按第 01 课的节奏：为什么需要、怎么运转、精确定义、代码落在哪、
怎么证明做对了。

### 5.1 状态是三块槽，不是一张网图

第 28 课把最小状态写成

$$
s_t = \bigl(\{x_t^{(i)}\}_{i \in \mathcal{O}},\; h_t,\; q_t\bigr)
$$

$\mathcal{O}$ 是桌上那几只刚体，本课主线只强制杯子；$h_t$ 是人的注意；$q_t$ 是
自己的身体。第 23 课在 2D Shapes 上证明过：杯子倒了、手机没动，一条向量会糊成
同一次纹理变化。本课还不到训练槽动力学，但日志必须按槽写。杯子、脸、自己的头
各有字段，禁止把整帧压成一句「桌面场景」。

观察 $z_t$ 是这一帧的传感器读数。状态 $s_t$ 可以含有当前帧看不见的东西，因为它
被允许使用历史。两者的关系，POMDP 里叫观察模型 $Z(z_t \mid s_t)$：杯子仍在桌上
但被手挡住时，$z_t$ 里没有杯框，$s_t$ 里杯子还在。本课不做信念更新的完整贝叶斯
滤波，只用一条滞回规则逼近这件事。规则是推断，必须单独存放。

类比：你转头看窗外，并不等于房间里的桌子被搬走了。类比失效处也要写清。人的
物体恒常靠的是一辈子的物理先验；本课的 2 秒窗口只是一条计时器。杯子被端走
1.8 秒，状态仍说它在，这是窗口的错，不是桌子的错。CUT3R 的循环 token 比计时器
强，仍然没有「谁是杯子」的身份，手和杯在点图里可以糊成一团。所以日志里必须
同时保留检测器的 `cup_on`（观察）和 `cup_still`（推断），第 30 课才能决定信哪
一个。

验证：打开任意一行 jsonl，应能指出至少两个观察字段和两个推断字段。若一行里只有
「caption」或「scene_text」，这行作废。

### 5.2 杯子：先框，再决定抬不抬到三维

二维检测是观察。MediaPipe Object Detector 的 EfficientDet-Lite 系列在 COCO 上
训练，类别含 `cup` 和 `bottle`。官方 Python 指南（2026-08-17）要求用
`ObjectDetector.create_from_options`，视频流走 `detect_for_video(mp_image, timestamp_ms)`，
时间戳必须单调递增的毫秒。结果里的框是像素坐标：`origin_x, origin_y, width, height`。
本课把「当前帧检到杯」写成 `cup_on`，框写成 `cup_bbox`，分数写成 `cup_score`。
这三样都是观察。

检不到不等于没有。挡住、出画、曝光一闪，检测器都会丢。推断层维护
`t_last_cup` 和上一份位置。记 $T=2$ 秒为本课默认的恒常窗口：

$$
\texttt{cup\_still}_t =
\begin{cases}
1 & \text{若 } \texttt{cup\_on}_t=1 \\
1 & \text{若 } \texttt{cup\_on}_t=0 \text{ 且 } t-t_{\mathrm{last}} < T \\
0 & \text{否则}
\end{cases}
$$

`cup_occluded` 在「当前丢框且 `cup_still=1`」时为真。这是规则，不是物理定律。
人把杯子端走超过 $T$，灯必须灭；一直亮，说明你的窗口太大，状态在撒谎。

有 GPU 时，把二维框抬到三维。VGGT 吃一组图，一次前向给出相机和点图，参考系是
第一张图的相机（第 20 课）。用法是抽 8 到 12 帧桌面图，在杯框内对高置信点取中位数，
得到 $x^{\mathrm{cup}}\in\mathbb{R}^3$。CUT3R 吃的是流：内部带着 $s_{t-1}$，新帧
只更新，点图画在同一世界系（第 21 课）。挡住那两秒，CUT3R 的状态 token 里杯子
可能还在，当前帧的 self 点图里杯子会被手盖住。恒常灯读的是世界系盒子里的点数，
不是当前帧检测器。

抬升时最容易写错的是坐标系。`load_and_preprocess_images` 会把原图缩放到仓库
默认边长（以该函数当前实现为准，写课时 README 示例按这个函数走），杯框若仍用
原图像素去取样，取到的是桌布或键盘。正确顺序：用原图跑 Object Detector 得到框，
把框的四角按宽高比映到预处理图，再在点图的对应像素里做中位数。CUT3R 的
`--size 512` 同理。映错了宁可不报米制坐标，只在 viser 里用眼睛看杯子是不是一簇
点。编一个「杯心在 0.31 米处」却对不上任何像素，比没有三维更糟，第 30 课会把
这个假坐标当成 $s_t$ 的一维。

VGGT 没有时间记忆，记忆就是你喂进去的那一组图。CUT3R 有循环状态，仍没有动作
端口：人伸手之前，它不能回答「推完杯子会在哪」。两者都是感知状态更新
$s_t=f(s_{t-1},I_t)$ 或 $s=f(I_{1:N})$，不是 $P(s_{t+1}\mid s_t,a_t)$。本课用它们
给杯子一个坐标，第 30 课才在这份状态上听动作。

三件感知器在桌上的分工可以写成一张对照，免得以后把点云演示当成世界模型：

| 零件 | 吃什么 | 吐什么 | 记住什么 | 本课拿它干什么 |
|---|---|---|---|---|
| Object Detector + Face Landmarker | 当前帧 | 框、点、矩阵、blendshape | 几乎不记，VIDEO 模式只有短跟踪 | 10 Hz 日志和 overlay |
| VGGT | 一组无序或少序的图 | 相机、点图、深度 | 输入集合本身 | 静物段给杯一个三维点 |
| CUT3R | 图像流 | 每帧点图和位姿，加一份循环状态 | $s_{t-1}$ | 挡住 / 转头时问杯子还在不在 |

三行都没有 $a_t$。谁若在 viser 里看到杯子还在，就宣布「桌宠已经会想后果」，把
感知一致性和动作条件预测混成一件事。第 21 课的书面区分本课继续有效。

验证：静物段 `cup_on` 应长时间为 1；挡住段 `cup_on` 允许掉 0，同时 `cup_still`
应保持 1；端走段超过 2 秒后 `cup_still` 必须掉 0。三维路径再加一条：挡住期间
盒子里的点数仍明显高于空桌基线，端走后掉到基线附近。这就是第 21 课灯的规则，
本课原样继承，不许三条视频各调一套盒子来让灯变绿。

### 5.3 人的注意：点是观察，看镜头是推断

Face Landmarker 是官方当前任务 API，不是旧的 `mp.solutions.face_mesh`。指南写明
三种运行模式：`IMAGE` 单图、`VIDEO` 已解码视频帧、`LIVE_STREAM` 摄像头异步流。
视频必须用 `VIDEO` 加 `detect_for_video`，并传入单调时间戳；用 `IMAGE` 逐帧调用
会关掉跟踪平滑，脸框会抖。直播叠加用 `LIVE_STREAM` 加 `detect_async` 和
`result_callback`，也可以退回 `VIDEO` 模式自己控时间戳，本课两种都承认，日志里
写你用的是哪一种。

打开两个默认关闭的开关，否则拿不到本课要的推断原料：

- `output_face_blendshapes=True`：52 个表情系数，含 `eyeLookDownLeft` /
  `eyeLookDownRight`。
- `output_facial_transformation_matrixes=True`：4×4 矩阵，把标准脸变到当前脸上。

478 个 `NormalizedLandmark` 的 $(x,y)$ 在 $[0,1]$，$z$ 是相对深度。人脸框取所有
点的轴对齐包围盒，这是观察。变换矩阵的左上 3×3 是旋转 $R$。本课与第 31 课共用
同一套欧拉提取，避免两课各解一套角：

$$
\texttt{pitch} = \operatorname{atan2}\bigl(-R_{21},\;\sqrt{R_{20}^2+R_{22}^2}\bigr),\quad
\texttt{yaw} = \operatorname{atan2}(R_{20}, R_{22})
$$

再转成角度。看镜头的操作定义写死，训练、直播、第 31 课复用日志时不许各改各的：

$$
\texttt{look}_t = \mathbf{1}\bigl[
\texttt{face\_on}_t=1,\;
|\texttt{yaw}_t|<20^\circ,\;
|\texttt{pitch}_t|<20^\circ,\;
\tfrac{1}{2}(\texttt{eyeLookDownL}+\texttt{eyeLookDownR})<0.35
\bigr]
$$

这是推断。它测量的是「头大致正对相机、眼睛没有明显下看」，不是虹膜落点，更不是
「他在想什么」。478 点里含虹膜，本课故意不用虹膜坐标做主标签：桌面摄像头分辨率
和侧脸角度下，虹膜估计比头部朝向更抖，第 31 课若拿本课日志当真值，抖会变成
「下一秒看镜头」的假转折。阈值是起点。第 7 节抽 20 帧肉眼对，错一半以上就改
阈值并重跑整份日志，同时重算第 31 课若已依赖这份文件的基线。

注意箭头是给互动用的可视化：从脸框中心沿偏航、俯仰所指的方向画一段。箭头方向
是推断的几何展示，不是新的观察。

验证：你看镜头时 `look` 应为 1，低头看手机应为 0，侧脸超过约 20° 应为 0。连续
一段里 `look` 从不翻转，多半是阈值太宽或矩阵没打开。`face_landmarks` 有点但
`facial_transformation_matrixes` 为空，就是开关没开。

### 5.4 自己的身体：真关节或键盘假头部

$q_t$ 是本体感觉。M 档的观察是电机读数。Reachy Mini 官方 API 文档给出
`get_current_joint_positions()`，返回头 7 维与天线 2 维，单位弧度；头部位姿另有
4×4 矩阵接口。相机帧走 `mini.media.get_frame()`，`uint8` 的 $(H,W,3)$。本课把
这九个数原样写入 `joints_head`、`joints_antennas`。它们是观察。SDK 的
`start_head_tracking()` 已经在做人脸追踪，那是控制层的融合结果。世界模型要消费
的是原始关节和本课自己算的 `look`，不要把官方追踪框再当一份「更真的状态」混进去。
第 28 课读过 `head_tracking.py` 时问的就是这个问题。

S 档没有编码器。第 28 课把假动作收成四个整数：0 停住、1 看左、2 看右、3 假伸手。
本课在回放或直播时用键盘写入同一套编码，字段名 `key_action`。这是命令，不是
测到的头角。若你按了 1 却不转镜头，第 30 课会把「看左」学成「画面不变」。本课
验收不看动作对换，但日志必须能让第 30 课接上。

没有 Reachy Mini、也不想对着视频按键，`key_action` 可以整列填 0，并在笔记写
「本段无身体通道」。缺身体的状态仍然合法，只是第 30 课不能用这段做动作对换。

验证：M 档转一下头，`joints_head` 必须变；S 档按一下键，`key_action` 必须变。
两者都不变，身体槽是空的。

把本课字段一次分完。验收时只准用这张表，不准临时发明「场景描述」列。

| 字段 | 栏 | 从哪来 | 丢了会怎样 |
|---|---|---|---|
| `face_bbox` / 478 点 | 观察 | Face Landmarker | 没有人脸几何，后面一切朝向都是编的 |
| `cup_bbox` / `cup_score` / `cup_name` | 观察 | Object Detector | 没有杯子几何，恒常只能瞎猜 |
| `yaw` / `pitch` / `eye_down` | 观察原料 | 4×4 矩阵与 blendshape | 规则没有输入 |
| `joints_head` / `joints_antennas` | 观察 | Reachy Mini 读数 | M 档身体槽为空 |
| `key_action` | 观察（命令） | 键盘 | S 档没有 $a_t$，第 30 课做不了动作对换 |
| `face_on` / `cup_on` | 观察 | 检测器是否给出至少一个实例 | 当前帧在不在画里 |
| `look` | 推断 | 20° 与 0.35 规则 | 把朝向误当成注视事实 |
| `cup_still` / `cup_occluded` | 推断 | 2 秒滞回 | 挡住会被写成「杯子消失」 |
| `cup_center_xy` 或三维杯心 | 推断 | 上一份检出位置或点图中位数 | 出画后没有可查询坐标 |
| 注意箭头 | 推断的可视化 | 由 yaw / pitch 画出 | 只影响 overlay，不进动力学 |

`yaw` 进观察栏，是因为它是矩阵解出的几何量，还没有经过「看镜头」这个决策。
`look` 进推断栏，是因为同一组角换一套阈值就会改标签。第 31 课复用顶层
`look` 时，等于复用本课的决策，不是复用一个独立真值。

### 5.5 探针：几何规则就是一种冻结读出

第 15 课的 V-JEPA 2 探针是：冻结 20 亿参数的视频骨干，只训一个 attentive probe
去读「这段跳水是哪一类」。本课没有在桌面视频上训那个骨干，但分工同构。Face
Landmarker 和 Object Detector 是冻结的感知器；看镜头规则、恒常滞回、三维中位数
是读出头。读出头可以是公式，也可以是以后要训的线性层。第 31 课会在这些低维
字段上再训一个 1 秒预报头，那才是「从现在读未来」。本课停在现在。

可选对照（第 11 节改造）是：把 478 个点压成一个向量，训一个逻辑回归去拟合你
肉眼标的「看镜头」，再和 20° / 0.35 规则比。那是第 15 课探针思路的桌面缩微版，
预算是 CPU 上几分钟，不是 64 卡。主线不要求做。主线要求你能说出：几何规则量的
是朝向，学习探针量的是你标注的那个概念，两者数字不能直接比，除非标注协议相同。

验证：口头讲清「冻结感知器 + 小读出」和「让语言模型看图说话」的差别。前者有固定
输入输出和可复现阈值，后者没有与 `a_t` 对账的计算图。

## 6. 源码导读

本课没有新的训练仓库。要读的是三份已经存在的官方资产，外加第 7 节的胶水脚本。
胶水允许短，主体必须是这些资产的推理接口。

### 6.1 MediaPipe Tasks：人脸与杯子

不要克隆 `google-ai-edge/mediapipe` 那个 C++ 单仓。Python 任务 API 随
`pip install mediapipe` 进来。以 [Face Landmarker Python 指南](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python)
2026-08-17 页为准。带问题读下面这张表。

| 位置 | 零件 | 带着什么问题读 |
|---|---|---|
| 指南「Create the task」IMAGE / VIDEO / LIVE_STREAM 三节 | 运行模式 | 视频为什么必须 `detect_for_video` 而不是 `detect`？LIVE_STREAM 为什么强制 `result_callback`？ |
| `FaceLandmarkerOptions` | 开关 | `output_face_blendshapes` 和 `output_facial_transformation_matrixes` 默认都是关的，不打开本课的 `look` 算不出来 |
| 指南「Handle and display results」 | 输出 | 478 个点、52 个 blendshape、4×4 矩阵分别在结果对象的哪个字段？ |
| [Object Detector Python 指南](https://developers.google.com/edge/mediapipe/solutions/vision/object_detector/python) | 杯子框 | `category_allowlist` 和 `score_threshold` 怎么滤到 `cup` / `bottle`？框的坐标系是像素还是归一化？ |
| 两份指南的「Prepare data」 | 颜色 | OpenCV 读到的是 BGR，`mp.Image` 要 SRGB，漏掉 `cvtColor` 会怎样？ |

VIDEO 模式的构造按官方示例写，本课只多开两个开关：

```python
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1,
)
```

`num_faces` 默认 1，平滑只在单脸时启用，桌宠对面通常就一张脸，保持 1。任务包直链
用第 31 课已经核对过的 float16 第 1 版，避免 `latest` 在课程周期里换文件：

```text
storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite
```

旧 API `mp.solutions.face_mesh.FaceMesh` 本课禁止出现在你的脚本里。官方 Tasks
页面已经用 `create_from_options` 替换它。画框用 OpenCV 即可，不必再去碰
`mp.solutions.drawing_utils`。

### 6.2 VGGT：一组图，一份点图

仓库：[facebookresearch/vggt](https://github.com/facebookresearch/vggt)。本课只做
推理。README 当前公开入口包括 `demo_viser.py`、`demo_gradio.py`、`demo_colmap.py`
和 Python 模块 `vggt.models.vggt.VGGT`。带问题读：

| 文件 | 带着什么问题读 |
|---|---|
| `vggt/models/vggt.py` 的 `VGGT` | 一次 `model(images)` 吐出哪些键？`pose_enc`、深度、点图是否都在？ |
| `vggt/utils/load_fn.py` 的 `load_and_preprocess_images` | 输入图被缩到多大？杯框要从原图映射到这张预处理图 |
| `vggt/utils/pose_enc.py` 的 `pose_encoding_to_extri_intri` | 外参约定是不是 OpenCV 的 camera-from-world？第一张图是不是世界原点？ |
| `vggt/utils/geometry.py` 的 `unproject_depth_map_to_point_map` | 点图和深度谁更适合取杯框内中位数？ |
| `demo_viser.py` | `--image_folder` 怎么喂你抽的桌面帧？ |

权重：`VGGT.from_pretrained("facebook/VGGT-1B")`。Hugging Face 卡片标 CC-BY-NC-4.0，
教学推理按非商业走。禁止编造训练命令；仓库 `training/` 是另档微调入口，本课不进。

### 6.3 CUT3R：流式状态，不是重推一组图

第 21 课已经把目录读过一遍。本课只回访和恒常相关的三处：

| 文件 | 带着什么问题读 |
|---|---|
| `demo.py` | `--seq_path` 能否直接接第 28 课的 mp4？viser 默认端口还是 8080 吗？ |
| `src/dust3r/inference.py` | 新帧是逐张更新状态，还是编码器先并行？显存为什么随帧数涨？ |
| `src/dust3r/model.py` 的 `ARCroco3DStereo` | 状态 token 在哪更新？挡住杯子时，被更新的是状态还是当前 self 点图？ |

CUT3R 官方 README 的推理入口是仓库根目录的 `demo.py`。必填参数是
`--model_path`、`--seq_path`（文件夹或视频）、`--size`、`--vis_threshold`、
`--output_dir`。第 21 课 Step 3 的官方示例把 `--size` 设为 512、`--vis_threshold`
设为 1.5，本课原样沿用，命令写在第 7 节 Step 5，这里不重复。

Webcam 在线 demo 在仓库 TODO 里，写课时仍未作为已发布入口，不要去找一个不存在的
`demo_webcam.py`。直播叠加走本课的 OpenCV 脚本。

Reachy Mini 的阅读清单见第 28 课第 6 节，本课不再重复安装。要读的只有观察侧：
`get_current_joint_positions`、`mini.media.get_frame()`。

## 7. 实验

工作目录默认 `~/learn-wm-l29`。每一步先写预期，再跑，再对照。视频必须是你或同伴
在镜头前的真实运动。

### Step 1: 建目录与 MediaPipe 环境

```bash
mkdir -p ~/learn-wm-l29/models ~/learn-wm-l29/data ~/learn-wm-l29/frames ~/learn-wm-l29/runs
```

```bash
python3 -m venv ~/learn-wm-l29/.venv
```

```bash
source ~/learn-wm-l29/.venv/bin/activate
```

官方 Python 指南的安装命令：

```bash
python -m pip install mediapipe
```

```bash
python -m pip install opencv-python numpy
```

预期：`python -c "import mediapipe as mp; print(mp.__version__)"` 能打印版本，不要
和 VGGT、CUT3R 的 conda 环境混用。

### Step 2: 下载任务包

与第 31 课同一组直链，字段才能复用：

```bash
curl -L -o ~/learn-wm-l29/models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

```bash
curl -L -o ~/learn-wm-l29/models/efficientdet_lite0.tflite https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite
```

预期：两个文件都非空，`face_landmarker.task` 大约 3 到 4 MB。curl 被拦就用指南页
上的模型按钮，手动放到 `models/`。

### Step 3: 放入第 28 课视频并抽查剧本

把 2 分钟视频拷到 `data/desk28.mp4`。若还没拍，现在拍，固定摄像头，人坐对面，
桌上有杯有笔，至少覆盖：

1. 静物 15 秒，人看镜头。
2. 人低头看手机或屏幕约 10 秒，再抬眼看镜头。
3. 手挡住杯子约 2 秒，手拿开，杯子没动。
4. 把杯子挪到另一侧，停住。
5. 把杯子端出画面超过 3 秒，再放回。
6. 可选：按键盘假头部的 1/2 同时真的转一下镜头，给第 30 课留身体通道。

用系统播放器看一遍，在 `data/events.csv` 写下事件秒数，三列 `name,start_s,end_s`，
至少有 `occlude`、`remove`、`look_away` 各一行。时间精确到 0.5 秒够用。这段视频
是观察，不是第 30 课的动作数据集：人伸手是外生事件，不要在 `key_action` 里写成
3，除非你同时按了假伸手键并且镜头或手臂真的在动。第 28 课验收过「你没动、人动
了」和「你做出可见动作、人随后有反应」各至少一段，本课沿用，缺了就补拍，不要
用静物循环充 2 分钟。

### Step 4: 把视频变成带观察 / 推断分栏的 jsonl

把下面存成 `~/learn-wm-l29/extract_state.py`。它按视频时间戳走 VIDEO 模式，10 Hz
采样，写出与第 31 课同名的 `face_on` / `look` / `yaw` / `pitch` / `eye_down` /
`cup_on`，并多写观察框、恒常推断和假头部。

```python
import argparse
import json
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions

HOLD_S = 2.0
YAW_LIM = 20.0
PITCH_LIM = 20.0
EYE_LIM = 0.35


def yaw_pitch(mat4):
    r = np.array(mat4).reshape(4, 4)[:3, :3]
    pitch = math.atan2(-r[2, 1], math.sqrt(r[2, 0] ** 2 + r[2, 2] ** 2))
    yaw = math.atan2(r[2, 0], r[2, 2])
    return math.degrees(yaw), math.degrees(pitch)


def blend_map(face_blendshapes):
    if not face_blendshapes:
        return {}
    return {c.category_name: float(c.score) for c in face_blendshapes[0]}


def face_bbox_px(landmarks, w, h):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return [round(x0), round(y0), round(x1 - x0), round(y1 - y0)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", default="data/desk29.jsonl")
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--hold", type=float, default=HOLD_S)
    args = p.parse_args()

    face_opt = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/face_landmarker.task"),
        running_mode=VisionRunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    det_opt = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path="models/efficientdet_lite0.tflite"),
        running_mode=VisionRunningMode.VIDEO,
        score_threshold=0.3,
        max_results=5,
        category_allowlist=["cup", "bottle"],
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit("cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(int(round(fps / args.hz)), 1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_cup_t = -1e9
    last_center = None
    frame_i = 0
    kept = 0

    with FaceLandmarker.create_from_options(face_opt) as face, \
            ObjectDetector.create_from_options(det_opt) as det, \
            out_path.open("w") as f:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if frame_i % stride != 0:
                frame_i += 1
                continue
            t_s = frame_i / fps
            ts_ms = int(t_s * 1000)
            h, w, _ = bgr.shape
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            fr = face.detect_for_video(mp_img, ts_ms)
            od = det.detect_for_video(mp_img, ts_ms)

            yaw = pitch = down = 0.0
            face_on = 1 if fr.face_landmarks else 0
            face_box = None
            look = 0
            if fr.face_landmarks:
                face_box = face_bbox_px(fr.face_landmarks[0], w, h)
            if fr.facial_transformation_matrixes:
                yaw, pitch = yaw_pitch(fr.facial_transformation_matrixes[0])
                bm = blend_map(fr.face_blendshapes)
                down = 0.5 * (
                    bm.get("eyeLookDownLeft", 0.0)
                    + bm.get("eyeLookDownRight", 0.0)
                )
                look = int(
                    face_on
                    and abs(yaw) < YAW_LIM
                    and abs(pitch) < PITCH_LIM
                    and down < EYE_LIM
                )

            cup_box = None
            cup_score = None
            cup_name = None
            for d in od.detections:
                cat = d.categories[0]
                cup_name = cat.category_name
                cup_score = float(cat.score)
                b = d.bounding_box
                cup_box = [int(b.origin_x), int(b.origin_y), int(b.width), int(b.height)]
                break
            cup_on = int(cup_box is not None)
            if cup_on:
                last_cup_t = t_s
                last_center = [
                    (cup_box[0] + cup_box[2] / 2.0) / w,
                    (cup_box[1] + cup_box[3] / 2.0) / h,
                ]
            cup_still = int(cup_on or (t_s - last_cup_t) < args.hold)
            cup_occluded = int((not cup_on) and cup_still)

            rec = {
                "t": round(t_s, 3),
                "frame": frame_i,
                "obs": {
                    "face_on": face_on,
                    "face_bbox": face_box,
                    "cup_on": cup_on,
                    "cup_bbox": cup_box,
                    "cup_score": None if cup_score is None else round(cup_score, 3),
                    "cup_name": cup_name,
                    "yaw": round(yaw, 2),
                    "pitch": round(pitch, 2),
                    "eye_down": round(down, 3),
                    "key_action": 0,
                },
                "infer": {
                    "look": look,
                    "cup_still": cup_still,
                    "cup_occluded": cup_occluded,
                    "cup_center_xy": last_center,
                    "hold_s": args.hold,
                },
                "face_on": face_on,
                "look": look,
                "yaw": round(yaw, 2),
                "pitch": round(pitch, 2),
                "eye_down": round(down, 3),
                "cup_on": cup_on,
            }
            f.write(json.dumps(rec) + "\n")
            kept += 1
            frame_i += 1

    cap.release()
    print(f"wrote {kept} rows to {out_path}")


if __name__ == "__main__":
    main()
```

在 `~/learn-wm-l29` 下跑：

```bash
python extract_state.py --video data/desk28.mp4 --out data/desk29.jsonl
```

预期：2 分钟视频大约 1200 行（10 Hz）。打开前 5 行，确认同时有 `obs` 和 `infer`，
并且顶层有 `look`、`face_on`、`cup_on`，方便第 31 课直接读。`facial_transformation_matrixes`
若整段为空，回去看 FaceLandmarker 的两个 output 开关。

抽 20 帧做肉眼表。从 jsonl 里均匀取 20 个 `t`，在播放器暂停到该秒，填四列：
秒、你看见的是否看镜头、日志 `look`、你看见的杯是否在桌（不论挡没挡）、日志
`cup_still`。看镜头对不上超过 10 帧，先改人的动作再改阈值；杯子恒常对不上，先
看 `events.csv` 的挡住时段是否真的短于 `--hold`。

### Step 5: 物体恒常检查（二维必做，三维有卡再做）

先做二维灯，不依赖 CUDA。把 `data/events.csv` 和 jsonl 对上：

```python
import csv, json
from pathlib import Path

rows = [json.loads(l) for l in Path("data/desk29.jsonl").read_text().splitlines() if l.strip()]
events = list(csv.DictReader(open("data/events.csv")))
print("n", len(rows), "cup_on mean", sum(r["cup_on"] for r in rows) / len(rows))
print("look mean", sum(r["look"] for r in rows) / len(rows))
for e in events:
    a, b = float(e["start_s"]), float(e["end_s"])
    seg = [r for r in rows if a <= r["t"] <= b]
    if not seg:
        print(e["name"], "empty")
        continue
    on = sum(r["cup_on"] for r in seg) / len(seg)
    still = sum(r["infer"]["cup_still"] for r in seg) / len(seg)
    occ = sum(r["infer"]["cup_occluded"] for r in seg) / len(seg)
    look = sum(r["look"] for r in seg) / len(seg)
    print(e["name"], "n", len(seg), "cup_on", round(on, 2), "still", round(still, 2), "occ", round(occ, 2), "look", round(look, 2))
```

预期写死，和第 21 课灯同方向：

| 事件 | `cup_on` | `cup_still` | 说明 |
|---|---|---|---|
| 静物 / 看镜头 | 高 | 1 | 检测器和状态一致 |
| `occlude`（挡住约 2 秒） | 允许掉 | 应接近 1 | 观察丢了，推断还在 |
| `remove` 开始 2 秒之后 | 低 | 应变 0 | 真拿走必须灭灯 |
| `look_away` | 无所谓 | 无所谓 | 看 `look` 应接近 0 |

挡住时段若 `cup_on` 仍为 1，说明手没把杯盖住，检测器太稳，换一只手或把杯放远
一点再拍。端走后 `cup_still` 仍为 1，把 `--hold` 从 2 降到 1.5 重跑，不要把
窗口拉到 10 秒来「看起来恒常」。

三维路径：复用第 21 课环境。先按 10 fps 抽挡住那段，避免 30 fps 把显存打满。
在 CUT3R 仓库根目录、已 `conda activate cut3r` 的终端里：

```bash
python demo.py --model_path src/cut3r_512_dpt_4_64.pth --seq_path /absolute/path/to/desk28.mp4 --size 512 --vis_threshold 1.5 --output_dir /absolute/path/to/learn-wm-l29/runs/occlude
```

然后用第 21 课的 `constancy_lamp.py` 协议：第一帧标杯的像素框，挡住期间盒子里
高置信点数应明显高于空桌基线，端走后掉到基线附近。本课不重写那份脚本。没有
CUDA，笔记写「三维恒常未做，二维灯结果如下」，仍然可以过第 9 节，但第 11 节
改造 1 对你关闭。

### Step 6: VGGT 给杯子一个三维坐标（24GB 卡）

从视频里按静物段抽 8 到 12 张图，保存到 `frames/`，编号 `00.jpg` 起，每张都要
看见杯子。VGGT 单独环境，命令以仓库 README 为准：

```bash
git clone https://github.com/facebookresearch/vggt.git
```

之后在 `vggt/` 里：

```bash
pip install -r requirements.txt
```

```bash
python demo_viser.py --image_folder /absolute/path/to/learn-wm-l29/frames
```

预期：viser 里能看见桌面点云和相机。同一只杯子在不同视角下应是一簇点，不是
两只漂开的杯。把杯框中心对应的三维点记进 `data/cup_xyz.txt`，格式三列数字即可。
OOM 就减到 8 张。Mac / CPU 可改走官方 Hugging Face Space 上传同样 8 张，把你在
浏览器里读到的定性结论写进笔记，并标明不是本地 `demo_viser.py` 的输出。

Python 模块入口（README 写法，在已安装的 vggt 环境里）是
`VGGT.from_pretrained("facebook/VGGT-1B")` 加 `load_and_preprocess_images`。杯框
必须映射到预处理后的分辨率再取样，对不上就只交 viser 里的定性观察，不要编一个
假的米制坐标。

### Step 7: 直播或回放叠加

把下面存成 `~/learn-wm-l29/overlay.py`。默认打开摄像头；给 `--video` 则回放
第 28 课素材。窗口里画杯子框、人脸框、从脸心指出的注意箭头，以及状态栏。

```python
import argparse
import math
import time

import cv2
import mediapipe as mp
import numpy as np

from extract_state import (
    YAW_LIM,
    PITCH_LIM,
    EYE_LIM,
    blend_map,
    face_bbox_px,
    yaw_pitch,
)

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--hold", type=float, default=2.0)
    args = p.parse_args()

    face_opt = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/face_landmarker.task"),
        running_mode=VisionRunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    det_opt = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path="models/efficientdet_lite0.tflite"),
        running_mode=VisionRunningMode.VIDEO,
        score_threshold=0.3,
        max_results=5,
        category_allowlist=["cup", "bottle"],
    )
    src = args.video if args.video else args.camera
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit("cannot open source")

    last_cup_t = -1e9
    t0 = time.time()
    key_action = 0
    with FaceLandmarker.create_from_options(face_opt) as face, \
            ObjectDetector.create_from_options(det_opt) as det:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            h, w, _ = bgr.shape
            now = time.time() - t0
            ts = int(now * 1000)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            fr = face.detect_for_video(mp_img, ts)
            od = det.detect_for_video(mp_img, ts)

            yaw = pitch = down = 0.0
            look = 0
            if fr.face_landmarks:
                x, y, bw, bh = face_bbox_px(fr.face_landmarks[0], w, h)
                cv2.rectangle(bgr, (x, y), (x + bw, y + bh), (255, 180, 0), 2)
                cx, cy = x + bw // 2, y + bh // 2
                if fr.facial_transformation_matrixes:
                    yaw, pitch = yaw_pitch(fr.facial_transformation_matrixes[0])
                    bm = blend_map(fr.face_blendshapes)
                    down = 0.5 * (
                        bm.get("eyeLookDownLeft", 0.0)
                        + bm.get("eyeLookDownRight", 0.0)
                    )
                    look = int(
                        abs(yaw) < YAW_LIM
                        and abs(pitch) < PITCH_LIM
                        and down < EYE_LIM
                    )
                    dx = int(80 * math.sin(math.radians(yaw)))
                    dy = int(80 * math.sin(math.radians(pitch)))
                    cv2.arrowedLine(bgr, (cx, cy), (cx + dx, cy + dy), (0, 255, 255), 2)
            cup_on = 0
            for d in od.detections:
                b = d.bounding_box
                x, y, bw, bh = int(b.origin_x), int(b.origin_y), int(b.width), int(b.height)
                cv2.rectangle(bgr, (x, y), (x + bw, y + bh), (0, 220, 0), 2)
                cup_on = 1
                last_cup_t = now
                break
            still = int(cup_on or (now - last_cup_t) < args.hold)
            lamp = "CUP YES" if still else "CUP NO"
            gaze = "LOOK" if look else "away"
            cv2.putText(
                bgr,
                f"{gaze} yaw={yaw:.0f} pitch={pitch:.0f} {lamp} key={key_action}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if look else (0, 128, 255),
                2,
            )
            cv2.imshow("desk29", bgr)
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or k == ord("q"):
                break
            if k == ord("s"):
                key_action = 0
            elif k == ord("a"):
                key_action = 1
            elif k == ord("d"):
                key_action = 2
            elif k == ord("w"):
                key_action = 3

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

摄像头直播：

```bash
python overlay.py
```

回放第 28 课视频：

```bash
python overlay.py --video data/desk28.mp4
```

预期：入画后出现人脸框；看镜头时状态栏为 `LOOK`，低头或侧脸变成 `away`；杯子
出现绿框；伸手挡住杯子，绿框可以消失，状态栏在约 2 秒内仍写 `CUP YES`，端走
超过 2 秒变成 `CUP NO`。箭头应随转头偏转。按 `a` / `d` / `w` / `s` 改 `key=`，
对应第 28 课的 1 / 2 / 3 / 0。Esc 或 `q` 退出。

LIVE_STREAM 模式是官方摄像头写法，要自己提供 `result_callback`，忙的时候会丢帧。
本课用 VIDEO 模式自己控时间戳，直播和回放共用一条路径，少一种失败模式。想对照
官方异步接口的人，按指南 LIVE_STREAM 小节改，笔记里写你改了哪几行。

### Step 8: M 档读真实关节（没有机器就跳过）

daemon 已在 `localhost:8000` 的前提下，把下面存成 `read_joints.py` 跑 10 秒，
确认身体槽是观察：

```python
import time
from reachy_mini import ReachyMini

with ReachyMini() as mini:
    t0 = time.time()
    while time.time() - t0 < 10:
        head, antennas = mini.get_current_joint_positions()
        print([round(x, 3) for x in head], [round(x, 3) for x in antennas])
        time.sleep(0.2)
```

预期：你扳一下头（重力补偿模式）或 `goto_target` 动一下，打印的数跟着变。把
同一瞬间的九个数写进 jsonl 的 `obs.joints_head` / `obs.joints_antennas` 即可，
本课不强制改 `extract_state.py` 去接 daemon。S 档用 Step 7 的按键通道。

### Step 9: 留证据

在 `~/learn-wm-l29/NOTES.md` 写下：

```text
日期与机器
视频文件名与时长、events.csv 三行事件
mediapipe 版本、两个任务包路径
look 规则：20 度 / 0.35，20 帧肉眼核对对了几帧
恒常：挡住时段 cup_on / cup_still；端走后 cup_still 何时掉零
CUT3R / VGGT：做了或未做，命令原文
观察字段列表
推断字段列表
下一课要补的动作通道：有 / 无
```

第 30 课会来取走 `desk29.jsonl` 和原视频。字段名不要在交卷前再改一版。

## 8. 配置与预算

| 档位 | 做什么 | 硬件 | 耗时（参考） | 产出 |
|---|---|---|---|---|
| 必做（S / Mac / CPU） | 装 MediaPipe、抽 jsonl、20 帧核对、二维恒常灯、overlay | 笔记本摄像头 | 半天到一天 | `desk29.jsonl`、肉眼表、灯记录、叠加窗口 |
| 加做 3D（24GB） | 第 21 课 CUT3R 对挡住 / 端走各跑一次；VGGT 8-12 帧 | 单张 24GB | 数小时含装环境 | 三维灯、杯心坐标 |
| M 档加做 | 读 `get_current_joint_positions` 10 秒 | Reachy Mini 或仿真 | 半小时 | 关节列进笔记 |
| 不要做 | 从头训 VGGT / CUT3R / Face Landmarker；用语言模型给帧写描述 | 任何卡 | 浪费本课 | 不算状态 |

超参本课几乎没有。写进笔记的只有四个数：采样 10 Hz、恒常窗口 2 秒、看镜头
偏航俯仰 20°、`eyeLookDown` 0.35。改了其中任何一个，必须重跑 jsonl 并重做
20 帧核对。Object Detector 的 `score_threshold=0.3` 是起点，漏检先换杯子，再降
到 0.2；降到 0.1 仍把键盘认成杯子，换回 0.3 并换杯子。

VGGT 权重约 5GB 量级，CUT3R 512 档 checkpoint 是 GB 量级，两者都只推理。论文
原训练（VGGT 64×A100、CUT3R 见 `docs/train.md`）不是本课预算。

## 9. 验收

- [ ] jsonl 不少于 1 分钟（按 10 Hz 至少约 600 行），同时含 `obs` 与 `infer`，
      顶层含 `look`、`face_on`、`cup_on`、`yaw`、`pitch`。
- [ ] 20 帧肉眼表：看镜头与 `look` 同向不少于 15 帧；说得出 5 帧里错的是侧脸、
      低头还是矩阵没开。
- [ ] 挡住约 2 秒：`cup_on` 允许掉，`cup_still` 保持 1；端走超过窗口后
      `cup_still` 掉 0。两种情况都写进 NOTES。
- [ ] overlay 窗口同时出现杯子框、人脸框、注意箭头；转头时箭头和 `LOOK`/`away`
      跟上；挡杯时灯按窗口延时，不是跟检测器同帧闪灭。
- [ ] 书面列出观察字段和推断字段。`yaw` 来自矩阵，算观察原料；`look` 来自阈值，
      算推断。两者不许写在同一栏。
- [ ] 口头关：对着没上过这门课的人讲「当前帧没有杯子，状态里为什么还可以有杯子」，
      必须出现滞回窗口或 CUT3R 状态，不许出现「模型懂了物体恒常」这种空话。
- [ ] 能指出本课的 $s_t$ 仍缺动作条件，第 30 课要在这份状态上训
      $P(s_{t+1}\mid s_t,a_t)$。
- [ ] 未用语言模型看图当作状态。NOTES 里没有场景散文充数。

## 10. 排错

| 症状 | 最可能的原因 | 怎么确认 | 怎么修 |
|---|---|---|---|
| `create_from_options` 报找不到模型 | 任务包路径错或下到了 HTML | `ls -l models/` 看体积 | 用 Step 2 的 curl 重下，或从指南页手动保存 |
| 整段 `facial_transformation_matrixes` 为空 | 两个 output 开关仍是默认 False | 打印 `fr.facial_transformation_matrixes` | 按第 6 节把 blendshape 和 matrix 打开 |
| `look` 从不翻转 | 人几乎正对镜头，或阈值太宽，或矩阵没开 | overlay 里 yaw/pitch 是否在动 | 先做明显的低头和侧脸；再查开关 |
| `look` 狂闪 | 用了 IMAGE 模式逐帧调 | 代码里是否 `detect` 而不是 `detect_for_video` | 改 VIDEO，时间戳单调递增 |
| 时间戳报错或不跟踪 | `timestamp_ms` 没增加或回绕 | 打印 ts | 用 `frame_index / fps * 1000` 或墙上时钟毫秒，只增不减 |
| 完全检不到杯子 | 杯子不在 COCO 的 `cup`/`bottle`，或太暗 | overlay 去掉 allowlist 看检出了什么 | 换纸杯或水瓶；必要时允许 `bottle` |
| 键盘被当成杯子 | 阈值过低 | 打印 `category_name` 和 score | 抬回 0.3，收紧画面 |
| OpenCV 窗口有框但颜色怪、点乱 | BGR 当 SRGB 喂进去 | 是否 `cvtColor` | 按指南把 numpy 转成 RGB 再建 `mp.Image` |
| CUT3R demo 在 Mac 失败 | 官方依赖 CUDA RoPE 核 | 第 21 课第 10 节 | 本课改走二维灯，笔记标明 |
| gdown 下到几 MB | Drive 返回了网页 | `file` 看类型 | 浏览器登录后手动下载到 `src/` |
| VGGT OOM | 帧太多或分辨率太大 | 显存曲线 | 减到 8 张；预处理尺寸以仓库函数为准，不要自己先放大 |
| 挡住时 `cup_still` 也掉 0 | 挡住超过 `--hold`，或挡住前就没检出过 | 看挡住前 1 秒的 `cup_on` | 先保证静物段能检出，再挡；挡的时间短于窗口 |
| overlay 从 `extract_state` 导入失败 | 不在同一目录 | 工作目录 | 在 `~/learn-wm-l29` 下启动 |

## 11. 前沿与改造

同一问题，2024 到 2026 年公开系统怎么拆。空间状态：VGGT 把一组图一次前向成
相机和点图（CVPR 2025 Best Paper），CUT3R 把这件事收成循环状态，新帧更新旧状态
（CVPR 2025 Oral）。两者都回答「现在世界长什么样」，都不吃动作。人的注意：
Gaze360、ETH-XGaze 估当前 3D 注视；Chong 等（CVPR 2020）估场景里的注视目标热图；
Gaze-LLE（Ryan 等，CVPR 2025，arXiv:2412.09586）冻结 DINOv2 只训注视解码器。
它们是当前帧估计器。桌宠要的下一步是第 31 课那种 1 秒预报。物体中心：C-SWM
把杯和手拆进不同槽，本课的 jsonl 按槽写字段，是同一思想的手工版。

缩小版和前沿版的差距，一半是规模。VGGT-1B、CUT3R 的多数据集训练、Gaze-LLE 的
DINOv2 骨干，钱能换。另一半是机制：本课的恒常是 2 秒滞回，不是可查询的未见
视角；看镜头是朝向阈值，不是热图落点；杯子身份靠 COCO 类别名，不是槽注意力。
机制差距里，滞回和分栏是本课能改的，换骨干是下一档预算。你若只有 Mac，三维路径
整段缺席，缺口也要写进 NOTES：缺的是点图，不是「状态」这个概念。二维字段仍然
是第 30 课合法的 $s_t$，只是没有米制坐标，规划时的「桌沿 5 厘米」得先自己标定
像素到厘米的粗比例，或诚实改成「框心出画面下三分之一」。

动手改造清单（选做，每个写预算和失败判据）：

1. 滑窗 VGGT 对循环 CUT3R。同一段 `occlude`，VGGT 每次只吃最近 4 帧，CUT3R 吃
   完整流。预算：两边环境已在的话半天。预期：挡住期间 4 帧窗口全是手，VGGT
   杯点应掉光，CUT3R 盒子里还应有点。失败：CUT3R 同样掉光，则「持久」在你的
   桌子上没发生，报告如实写，二维滞回仍算本课过关。
2. 朝向规则对学习探针。肉眼标 200 帧 `look`，用 478 点的中心和眼宽比训一个
   逻辑回归，和 20° / 0.35 规则比准确率。预算：CPU 一小时。预期：规则在正对
   镜头时够用，侧脸和学习探针可能打平或探针略好。失败：探针在训练集 99%、
   隔天录像掉到随机，说明你在背当天的光照，把这个写进笔记。
3. 把手加进物体槽。装官方 Hand Landmarker（第 31 课同一任务包），在 jsonl 增
   `hand_on` 和手头距离。预算：两小时。预期：伸手挡杯时 `hand_on=1` 且
   `cup_occluded=1` 同时成立，比单靠丢框更像「挡住」而不是「拿走」。失败：
   手检不稳，挡住和端走仍分不开，不要用它替换二维灯的主结论。
4. 把 `key_action` 和画面真的对上。录 1 分钟，按 `a` 时必须左转镜头，事后抽
   查 20 帧：键为 1 时，背景应向右移。预算：一小时。预期：对上的帧才能进
   第 30 课。失败：对不上，整列填 0，不要把假标签交给动力学。

VGGT 论文「一组图即可查询几何」对应改造 1 的 VGGT 侧；CUT3R 论文「挡住后仍
能从状态读结构」对应改造 1 的 CUT3R 侧。单卡可验方向，不能拿你的 2 分钟视频
去复现他们表格里的 KITTI / DTU 数字。

## 12. 论文与延伸

1. VGGT（Wang, Chen, Karaev, Vedaldi, Rupprecht, Novotny，CVPR 2025 Best Paper，
   [arXiv:2503.11651](https://arxiv.org/abs/2503.11651)）。带着问题读：一次前向
   吐出的相机、点图、深度、轨迹，哪几样本课真用上了？摘要里的
   「one, a few, or hundreds of views」和 CUT3R 的 stream 差在输入是集合还是流？
   为什么作者把 BA 写成可选，本课为什么连可选都不跑？
2. CUT3R（Wang, Zhang, Holynski, Efros, Kanazawa，CVPR 2025 Oral，
   [arXiv:2501.12387](https://arxiv.org/abs/2501.12387)）。带着问题读：状态更新
   和状态读出如何共用 decoder？raymap 查询未见视角会不会改状态？论文如何区分
   在线设定和 revisiting 设定，你的 `demo.py` 跑的是哪一列？
3. V-JEPA 2（Assran, Bardes, Fan, Garrido 等，
   [arXiv:2506.09985](https://arxiv.org/abs/2506.09985)），回访第 15 课。带着
   新问题：attentive probe 量的是当前片段类别，本课的朝向规则量的是当前朝向，
   两者都不是预报。论文里 Epic-Kitchens 动作预判那一栏，为什么要留给第 31 课
   而不是本课？
4. DUSt3R（Wang 等，[arXiv:2312.14132](https://arxiv.org/abs/2312.14132)）。
   CUT3R 的点图表示从哪来？两视图 pairwise 和 CUT3R 的循环状态，谁更适合桌宠
   这种「人一直坐着、手突然伸进来」的流？
5. Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs
   （Kartynnik, Ablavatski, Grishchenko, Grundmann，
   [arXiv:1907.06724](https://arxiv.org/abs/1907.06724)）。Face Mesh 468 点的
   出处。带着问题：它估的是网格，不是注视；本课为什么还要 blendshape 和 4×4
   矩阵才能谈「看镜头」？
6. C-SWM（Kipf, van der Pol, Welling，ICLR 2020，
   [arXiv:1911.12247](https://arxiv.org/abs/1911.12247)），第 23 课主论文。
   带着问题：本课手工分栏的杯 / 脸 / 身体，和论文里学出来的槽，差在谁负责
   绑定物体身份？桌面上若检测器把杯认成 bottle，槽会不会把身份换掉？

仓库与文档入口：VGGT README 与 `facebook/VGGT-1B`；CUT3R README、`docs/eval.md`、
`docs/train.md`（只读）；MediaPipe Face Landmarker / Object Detector 的 Python
指南（以 2026-08-17 页为准）。注视估计的 Gaze360、ETH-XGaze、Chong CVPR 2020
放到第 31 课精读，本课知道它们是「现在」不是「下一秒」即可。

现在整个系统多了一层可以查询的现在：杯子在不在、人看不看你、自己的头朝哪。
它仍不会听动作。第 30 课要在这份状态上训 $P(s_{t+1}\mid s_t,a_t)$，同一段历史，
头左转和手往杯方向伸，想象必须分岔。日志里的 `key_action` 或关节，就是那一课
的 $a_t$。没把观察和推断分开的字段，到时候会把检测器抖动学成动力学。第 31 课
会在同一份 `look` 上预报 1 秒后的转折，第 32 课会查询「碰不碰杯」再决定动不动。
把 `NOTES.md` 放在视频旁边，[第 30 课](./30_desk_world_model.md) 从这里接着采动作通道。
