---
id: 02_vae_visual_compression
title: "用 32 个数表示一帧赛道"
summary: "压缩一定会丢信息。怎么知道丢掉的是不是要命的那部分？"
unit: vmc
play_tools: []
checkpoints:
  - "一个训练好的 VAE，重建质量有量化记录。"
  - "latent 走查报告：哪些信息保住了、哪些丢了、为什么无所谓或有所谓。"
  - "不同 β 和 latent 维数的对比记录。"
---

# 第 02 课：用 32 个数表示一帧赛道

> 类型：复现（World Models 复现的第二步，01-04 课合力完成）<br>
> 建议周期：2-3 天<br>
> 硬件：单张 24GB 显卡数小时；Mac/纯 CPU 可完成小配置训练与全部走查实验<br>
> 锚定仓库：[ctallec/world-models](https://github.com/ctallec/world-models) 的 `models/vae.py` 与 `trainvae.py`<br>
> 产物：正式训练的 VAE（外加 β 和维数各不相同的三个对照 VAE）、latent 走查报告（插值图、逐维遍历图、重建损失解剖、弯道探针对照表）

## 1. 这一课做什么

上一课的 V-M-C 流水线已经能跑，但其中每个模型都只是冒烟配置。从这一课开始逐个把
零件训扎实，先处理负责视觉压缩的 VAE。我们会把数据补到 1000 条轨迹，训练至收敛，
然后检查那 32 维潜向量究竟保存了什么：逐维改变它、在两帧之间插值、分别统计路面与
草地的重建误差，再用线性探针读取道路弯度。

先把标题里的一个细节说诚实：CarRacing 吐出的画面是 96×96 像素，但论文和这份仓库
都会先把它缩到 64×64 再喂给 VAE（`trainvae.py` 的预处理里有一行 `Resize`，尺寸常量
`RED_SIZE = 64`）。所以"96×96 压成 32 个数"实际分两段：先是一次不学习的缩放
（96×96 到 64×64，丢掉的是分辨率），然后才是学习出来的压缩，64×64×3 = 12288 个
像素值进去，32 个数出来，压缩了将近四百倍。关键不在于少了多少数字，而在于留下了
什么。CarRacing 的草地纹理占据大量像素，对控制真正重要的却是道路位置、弯向和车身
姿态。下一课的 M 只接收 $z$，第 04 课的 controller 也依赖这份表示；如果弯道信息已在
压缩时丢失，后面的模型没有办法补回来。

因此，这里不把“重建图看起来像”当作验收标准。最后会并排比较重建损失和弯道探针：
两者的排名可能不同。这份结果也会成为第 13 课讨论 JEPA、第 16 课比较三条路线时的
第一组证据。

术语速查：

| 术语 | 一句人话 |
|---|---|
| encoder / decoder | VAE 的两半：encoder 把图压成 32 个数，decoder 从 32 个数把图解回来 |
| $\mu$ 和 $\sigma$ | encoder 的输出其实不是一个点，是一团高斯的中心和胖瘦；取样才得到 $z$ |
| ELBO | VAE 的训练目标，拆开是两项：重建项（解得回来）加 KL 项（编码要规整） |
| KL 散度 | 度量两个分布差多远；这里管"你的编码分布离标准正态有多远" |
| 重参数化 | 一个让"采样"这步也能传梯度的小手术，VAE 能训起来全靠它 |
| β | KL 项前面的权重系数，压缩带宽的税率；本仓库没写这个系数，等于默认 β=1 |
| 后验塌缩 | KL 项赢得太彻底的病：编码不再携带信息，decoder 对什么输入都吐差不多的图 |
| latent 插值 | 在两帧的 $z$ 之间连直线，逐点解码，看潜空间"路上"长什么 |
| 逐维遍历 | 固定 31 个数不动，只拧 1 个数，看画面哪里变，给每根旋钮找工作 |
| 闲置维 | 遍历时拧了没反应的维度：模型根本没用它编码任何东西 |
| 线性探针 | 冻住 encoder，只训一个线性回归去从 $z$ 里读某个量（本课读"路的弯度"），读得出说明信息在 |

## 2. 问题

压缩必然丢信息，这没得商量，12288 个数塞进 32 个数，数学上就装不下。真正的问题
是：**丢的是不是要命的那部分？** 这里把这个问题拆成三个能动手回答的小问题：

1. VAE 到底把什么装进了那 32 个数？光看重建图只能得到"大概像"的印象。插值
   和逐维遍历是两把解剖刀，能看到每根维度各自管什么、有多少维在摸鱼。
2. 训练目标在鼓励它保住什么？重建损失是按像素算的，草地占的像素多，路缘占的
   像素少，所以模型天然把预算花在草地上。这不是猜测，本课你会把重建误差按区域拆开，
   算出草地到底吃掉了多少损失预算。
3. "重建好"和"对下游有用"是同一回事吗？这是本课的问题。我们训四个
   配置不同的 VAE（latent 维数 8/32/64，β 取 1 和 4），分别量它们的重建损失和
   "从 $z$ 读出路面横向位置"的探针精度，看两个排名是否一致。如果不一致，而且
   多半会不一致，你就拿到了第 13 课 JEPA 路线的出发点：既然重建指标不忠于
   决策需求，为什么还要为重建像素付出全部代价？

顺带解决一个工程问题：这份仓库的 `trainvae.py` 既没有 latent 维数开关也没有 β 开关
（维数是 `utils/misc.py` 里的常量 `LSIZE`，β 压根不存在于代码里）。第 01 课改造
清单 3 留下的钩子，这课正式接上，你会改两处各一行的代码，这也是全课程第一次
对锚定仓库动刀。

## 3. 准备

- 第 01 课的产出：能跑的环境（老 gym 那套依赖）、`datasets/carracing` 数据目录、
  你的证据目录习惯。没跑过第 01 课的先回去，本课所有命令都建立在它之上。
- 数据要补货。第 01 课的 100 条轨迹这里不够用，有个硬性原因：`data/loaders.py`
  切分训练集和测试集的方式是"最后 600 个轨迹文件做测试集，其余做训练集"。文件数
  不到 600 时训练集直接是空的，训练会在除零或空迭代上崩掉。所以 Step 1 先把数据
  补到 1000 条（README 的正式配置），磁盘按 15-20GB 预留。
- Mac/CPU 用户：本课全部实验都能做。采数据吃 CPU 多核（比 GPU 用户慢些但可行），
  训练用小 epoch 数（Step 2 给了 Mac 档参数），走查实验全是轻量推理。
- Python 侧会写三小段胶水脚本（走查、损失解剖、探针），用到 numpy 和 torchvision，
  第 01 课的环境里都有；探针如果想用 sklearn 也可以，正文给的是纯 numpy 版。

## 4. 学习目标

1. 白纸写出 VAE 的训练目标，说清两项各管什么、谁和谁在打架；
2. 用三行伪代码解释重参数化在解决什么问题，删掉它梯度断在哪；
3. 说出本仓库 VAE 的结构（encoder 几层卷积、通道怎么翻倍、decoder 怎么解回去），
   并解释 latent 维数改动时哪些层的形状会跟着变；
4. 给任何一个训好的 VAE 做完整走查：重建、插值、逐维遍历，并把结果讲成人话；
5. 解释 β 拧大拧小分别买到什么、付出什么，以及 sum 和 mean 两种损失归约方式会
   怎样悄悄改变有效 β；
6. 用自己的实验数据回答：重建损失的排名和下游探针的排名一致吗？这对"该用什么
   目标训练表征"意味着什么？

## 5. 原理

四个机制，每个仍按第 01 课的节奏走：为什么需要（直觉）、怎么运转（机制）、精确
定义（数学）、在源码哪里（代码）、怎么证明做对了（验证）。

### 5.1 压缩-重建目标：ELBO 的两项各管一件事

最朴素的压缩器是普通自编码器：encoder 压、decoder 解，只要求"解得
回来"。它能训出很低的重建误差，但压出来的空间是一团乱麻：每张训练图各自占一个
犄角旮旯，两张图的编码之间是无人区，从无人区取个点解码出来是碎片。这对世界模型
是致命的，下一课的 M 要在这个空间里**预测**，预测值几乎必然落在训练编码点之间；
第 04 课做梦时还要在这个空间里**采样**。所以 V 需要的不只是"压得下、解得回"，
还要"空间本身规整连续"：任取一点解码都像一帧合法画面，相邻的点解码出相似的画面。
VAE 就是给自编码器加上这条纪律的版本。

两处改动。第一，encoder 不再输出一个点，而是输出一团高斯的中心 $\mu$ 和
胖瘦 $\sigma$，训练时从这团高斯里采一个 $z$ 交给 decoder，每张图占据的是潜空间里
一小片云而非一个针尖，云和云互相搭界，空隙被填掉。第二，损失里加一项 KL 散度，
惩罚每团云偏离标准正态 $\mathcal{N}(0, I)$ 的程度，所有云被拽向原点附近挤在一起，
不许散落到犄角旮旯。你可以把 KL 项理解成**编码规整税**：想把图编码到偏僻的位置
（$\mu$ 很大）或者把云收得极窄（$\sigma$ 趋零，退回针尖）都要交重税。重建项和
KL 项天然打架：前者希望每张图的编码越独特越好认，后者希望所有编码挤成一团标准
正态。训练就是在这场拉锯里找平衡点。

VAE 最大化的是对数似然的一个下界，叫 ELBO（evidence lower bound，
证据下界）。写成损失（取负号最小化）是两项：

$$
\mathcal{L} = \underbrace{\mathbb{E}_{z \sim q(z|x)} \big[ -\log p(x|z) \big]}_{\text{重建项}} + \underbrace{D_{\mathrm{KL}}\big( q(z|x) \,\|\, \mathcal{N}(0, I) \big)}_{\text{KL 项}}
$$

当 $q(z|x)$ 是对角高斯、先验是标准正态时，KL 项有闭式解，不用采样就能精确算出：

$$
D_{\mathrm{KL}} = -\frac{1}{2} \sum_{i} \big( 1 + \log \sigma_i^2 - \mu_i^2 - \sigma_i^2 \big)
$$

重建项在"像素是高斯噪声观测"的假设下就是逐像素平方误差（MSE）。

`trainvae.py` 的 `loss_function` 函数，一共三行有效代码。重建项是
`F.mse_loss(recon_x, x, size_average=False)`，注意两个坑：变量名叫 `BCE`，但它
实际是 MSE（历史遗留的命名，别被骗）；`size_average=False` 表示对整个 batch 的
全部像素**求和**而不是求平均，这个选择后面 5.3 节会再回来算它的账。KL 项写作
`-0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())`，和上面
闭式解逐项对得上，只是网络输出的是 $\log \sigma$，所以公式里的 $\log \sigma^2$
写成了 `2 * logsigma`。最后一行 `return BCE + KLD`，两项直接相加，没有任何系数。

训练时别只盯总损失：把两项分开打印（`loss_function` 里顺手加两个
`print` 或记到日志），健康的训练里重建项持续下降、KL 项先升后稳在一个正数，
KL 稳定为正说明编码确实携带着信息（每张图的云和先验有差别，差别就是信息）。KL
掉到接近零是病（后验塌缩，见第 10 节）；KL 一路疯涨不回头也是病（规整税收不动了）。

### 5.2 重参数化：让"采样"这一步也能传梯度

上一节说 encoder 输出一团云、训练时从云里采样。问题来了：反向传播遇到
"采样"就断了，采样是掷骰子，骰子的结果对 $\mu$ 和 $\sigma$ 求导没有定义，梯度
传不回 encoder，encoder 就永远学不会怎么调整这团云。VAE 论文最核心的一手就是解决
这个问题，而且解法漂亮得只有一行代数。

把"从 $\mathcal{N}(\mu, \sigma^2)$ 里采样"改写成"先从标准正态采一个
$\epsilon$，再做确定性变换 $z = \mu + \sigma \epsilon$"。两种写法采出来的分布一模
一样，但第二种把随机性外包给了 $\epsilon$，它不依赖任何网络参数，只是个外来的
噪声输入。于是从损失回望 $\mu$ 和 $\sigma$ 的路径上全是确定性运算，梯度畅通无阻。
打个比方：原来是"骰子决定 $z$"，现在是"骰子只决定偏移方向，$z$ 的位置仍由
$\mu$、$\sigma$ 这两个可调参数控制"，账能算到参数头上了。类比失效处：这招只对
连续分布好使，第 09 课 IRIS 用的离散 token 没法这么变换，得换直通估计等更糙的手段，
到时你会怀念高斯的好。

目标是对 $\mathbb{E}_{z \sim \mathcal{N}(\mu, \sigma^2)}[f(z)]$ 求
$\mu, \sigma$ 的梯度。重参数化把它改写成
$\mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)}[f(\mu + \sigma \epsilon)]$，期望的
分布不再含参数，梯度可以直接搬进期望号里，用单个 $\epsilon$ 样本做无偏估计。
Kingma 与 Welling 论文的主要贡献就是指出这个估计的方差远低于当时通行的打分函数
估计器，低到单样本就能稳定训练。

`models/vae.py` 的 `VAE.forward`，四行，建议背下来：

```python
mu, logsigma = self.encoder(x)
sigma = logsigma.exp()
eps = torch.randn_like(sigma)
z = eps.mul(sigma).add_(mu)
```

`torch.randn_like(sigma)` 就是那个外包的骰子 $\epsilon$，`eps.mul(sigma).add_(mu)`
就是 $z = \mu + \sigma \epsilon$。随后 `recon_x = self.decoder(z)`，`forward`
返回三元组 `(recon_x, mu, logsigma)`，损失函数正好需要这三样。

一个两分钟的思想实验加一个五分钟的代码实验。思想实验：如果直接写
`z = torch.normal(mu, sigma)`，PyTorch 不会报错，但梯度到 `z` 就断了（采样算子
不可导），encoder 的参数永远不更新，训练损失会降（decoder 还在学），但重建质量
上限极低。代码实验：训练脚本跑起来后，打印 `model.encoder.fc_mu.weight.grad` 的
范数，非零说明梯度确实穿过了采样这一步，重参数化在干活。

### 5.3 β：压缩带宽的税率旋钮

5.1 节说重建项和 KL 项在拉锯，那自然要问：拉锯的力量对比能不能调？
β-VAE 的回答就是在 KL 项前面乘个系数 β。把 KL 项理解成规整税，β 就是税率。税率
调高，encoder 交得起税的信息量变少，它被迫只保留最"值钱"（对重建帮助最大、
且能用最少维度表达）的信息，其余全扔；顺带的副产品是各维度倾向各管一个独立的变化
因素（β-VAE 论文的主打卖点：解耦）。税率调低甚至归零，encoder 想编多少编多少，
重建当然好，但空间退化回乱麻，插值和采样开始穿帮。所以 β 本质上是一根**信息带宽
旋钮**：拧它就是在"重建保真"和"空间规整、编码精炼"之间移动交换比。

β 大于 1 时：KL 项把每维的 $q(z_i|x)$ 往标准正态摁，信息量不够付税的
维度干脆缴械，$\mu_i$ 恒等于 0、$\sigma_i$ 恒等于 1，变成闲置维（你会在逐维遍历
实验里看到这些拧了没反应的旋钮）。留下来干活的维度个个身兼要职，往往对应
画面里独立的宏观因素。β 小于 1 时反过来：维度个个塞满细节，重建锐利，但两团云
之间的空隙变大，插值中段解码出叠影或碎片。

β-VAE 的目标就是把 5.1 的损失改写成：

$$
\mathcal{L}_{\beta} = \mathbb{E}_{z \sim q(z|x)} \big[ -\log p(x|z) \big] + \beta \, D_{\mathrm{KL}}\big( q(z|x) \,\|\, \mathcal{N}(0, I) \big)
$$

论文把它推导为一个带约束优化的拉格朗日形式：最大化重建似然，约束条件是编码携带的
信息量不超过某个额度，β 是约束的拉格朗日乘子。这正是"信息瓶颈"三个字的字面意思：
β 决定瓶颈的口径。这里有个极易踩的暗坑：**β 的有效值取决于两项各自的归约方式**。
本仓库重建项对 12288 个像素求和、KL 项对 32 个维度求和，两项的量纲天然差几百倍；
如果你手痒把 `mse_loss` 的求和改成求平均（除以 12288）而 KL 不动，等效 β 瞬间被
放大几百倍，模型多半直接后验塌缩，这不是假设，是初学者改 VAE 损失的第一大翻车
现场。比较不同 β 时，必须保证归约方式完全一致。

仓库里没有 β，`trainvae.py` 的 `loss_function` 最后一行
`return BCE + KLD` 写死了两项等权相加，在它自己的求和归约约定下就是 β=1。命令行
也没有开关（`trainvae.py` 的全部参数只有 `--batch-size`、`--epochs`、`--logdir`、
`--noreload`、`--nosamples`）。所以本课的 β 实验要动刀：给 argparse 加一个
`--beta` 参数（float，默认 1.0），把最后一行改成 `return BCE + args.beta * KLD`。
两处各一行，Step 8 给出完整操作。

β 的效果用三面镜子照：重建图（β 大则糊）、随机采样图（`--nosamples`
没开的话，训练时每个 epoch 都会从标准正态采 $z$ 解码存进 `samples/` 目录，β 大
的模型采样图更干净完整，β 小的更碎）、逐维遍历（β 大则闲置维多、活跃维职责更
单一）。三面镜子的方向都对上了，你的 β 改动就是生效的。

### 5.4 对重建重要，不等于对决策重要（本课灵魂）

重建损失是逐像素平方误差，等于**按像素数投票**。一帧 CarRacing 里草地
占了大半个画面，路面和路缘占的像素少得多，而车身姿态、路的弯向这些决策命脉，在
像素账本上只值几个百分点。于是 MSE 训出来的 VAE 把预算优先花在票仓大户上：草地的
色块位置、赛道的整体明暗，它拼命保；路缘再弯一点还是直一点、车头偏没偏，它觉得
无所谓，反正错了也没几个像素的损失。可下游要开车，全指着后面这几个百分点。
一句话：**损失函数的选票分布和决策的信息需求分布，压根不是一张图。** 这不是
CarRacing 的特例，是"用重建当目标学表征"这条路线的结构性毛病，第 13 课的 JEPA、
第 08 课的 TD-MPC2 都是冲着它来的。

这个论断听起来顺耳，但顺耳的论断最需要动手验证，本课设计了两个实验
夹住它。**实验一（损失解剖）**：把验证集重建误差按像素区域拆开，草地像素贡献
多少、路面像素贡献多少，再对照两类像素各占多少面积。如果草地以大面积吃掉大头
损失预算，"按像素数投票"就坐实了。**实验二（探针对决）**：训四个不同配置的
VAE，每个量两个指标，验证集重建损失，和"冻住 encoder、用线性回归从 $\mu$ 读出
路面横向偏移"的探针误差。路面横向偏移是"路往哪边弯"的直接代理，是决策最需要的
信息。然后看两个排名是否一致。只要出现一次倒挂（甲重建比乙好、探针却比乙差），
"重建好等于压得对"就被证伪了。

损失解剖就是把求和拆成两个子集：总重建误差等于草地像素上的误差和
非草地像素上的误差之和，各自除以总误差得到贡献占比，再和面积占比并排放。探针是
岭回归：特征取 encoder 输出的 $\mu \in \mathbb{R}^{L}$，标签 $y$ 是从原始像素算出
的路面横向偏移标量，解 $\min_w \|Xw - y\|^2 + \lambda \|w\|^2$，在留出集上报
$R^2$（决定系数：1 是完美预测，0 是不比报均值强）。用线性模型是刻意的：探针越笨，
越能证明信息是**明摆着**编码在 $z$ 里的，而不是靠探针自己算出来的。

全部是胶水脚本（第 7 节 Step 6、Step 7 给出完整代码）：草地掩码用
"绿色通道显著高于红蓝"的规则从像素直接算，标签用掩码质心，特征用
`model.encoder(x)` 返回的 `mu`。锚定仓库不用改任何代码。

两个实验各有自检。损失解剖：掩码叠加图肉眼过一遍，草地确实被标绿了才
算数。探针：先给标签本身做可视化抽查（把算出的偏移值标在原图上，弯道帧的值该明显
偏向弯的方向）；再跑一个"打乱对照"，把标签随机打乱后重训探针，$R^2$ 应跌到
0 附近，否则你的探针在作弊（比如数据泄漏）。预先说明结果的两种读法：如果四个模型
的探针排名和重建排名出现倒挂，论断直接成立；如果没倒挂（比如这个任务太简单，
连 8 维 VAE 都把路位置编码得很好），结论是"在本设置下重建目标顺带保住了路的
信息，但保住的代价和必要性存疑"，这个悬念恰好留给第 16 课：同一批数据、同一个
探针，换 JEPA 目标再比一次。两种结局都值得写进走查报告，不许只报好看的那种。

## 6. 源码导读

这里只涉及四个文件，都不长，建议全部通读，每个带着问题进去：

| 文件 | 是什么 | 带着什么问题读 |
|---|---|---|
| `models/vae.py` | V 的全部结构定义 | 三个类 `Encoder`、`Decoder`、`VAE` 各自的 forward 返回什么？重参数化那四行在哪？|
| `trainvae.py` | V 的训练脚本 | `loss_function` 两项怎么写的？checkpoint 存到哪、什么条件下存 `best.tar`？|
| `utils/misc.py` | 全仓库共享常量 | `LSIZE` 在哪一行定义？还有谁在用它（这决定了改维数实验的注意事项）？|
| `data/loaders.py` | 数据加载 | 训练集和测试集怎么切的？buffer 机制为什么存在、一个 epoch 实际见到多少数据？|

把结构对着代码数一遍（数完你就能默写）。`Encoder`：四层卷积，通道
3 到 32 到 64 到 128 到 256，全部 kernel 4、stride 2；64×64 输入走完四层剩 2×2，
展平成 $2 \times 2 \times 256 = 1024$ 维，接两个并排的全连接头 `fc_mu` 和
`fc_logsigma`，各输出 `latent_size` 维，这是 $\mu$ 和 $\log\sigma$ 的出处。
`Decoder` 反着来：全连接 `fc1` 把 `latent_size` 维升到 1024，reshape 成
1024 通道的 1×1 特征图，四层反卷积（通道 1024 到 128 到 64 到 32 到 3，kernel
前两层 5、后两层 6，stride 都是 2）解回 64×64×3，最后过 sigmoid 把像素压回
0 到 1。注意一个不对称的细节：encoder 的展平维度 1024 是由输入尺寸 64×64 算出来
的，这是为什么改 `RED_SIZE` 会直接崩（形状对不上），而改 `LSIZE` 不会（它只
出现在全连接层的一端）。

`utils/misc.py` 里那行关键常量，需要打交道的就是它：

```python
ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE =\
    3, 32, 256, 64, 64
```

`LSIZE` 是 latent 维数（32），`RED_SIZE` 是喂给 VAE 前的缩放尺寸（64）。
`trainvae.py` 实例化模型的那行 `model = VAE(3, LSIZE)` 从这里取值，所以维数
实验改的是 `misc.py` 这一行，不是 `vae.py`。警告一次，第 10 节还会再警告一次：
`LSIZE` 同时被 `trainmdrnn.py` 和 controller 相关代码引用，本课实验改完**必须
改回 32**，否则第 03 课加载模型时形状对不上。

`trainvae.py` 还有三个值得留意的行为：数据路径写死为 `datasets/carracing`（没有
命令行参数可改，数据必须放这）；启动时默认会加载 `logdir/vae/best.tar` 续训，
想从头训要么换 logdir 要么加 `--noreload`；训练带 `ReduceLROnPlateau`（验证损失
5 个 epoch 不降就减半学习率）和 patience 为 30 的 early stopping，所以默认
`--epochs 1000` 实际不会跑满，收敛了自己会停。

## 7. 实验

所有命令都在仓库根目录执行（胶水脚本也放根目录，因为要 `from models.vae import VAE`）。
每步先写预期，再跑，再对照。

### Step 1: 把数据补到正式规模

```bash
python data/generation_script.py --rollouts 1000 --threads 8 --rootdir datasets/carracing
```

这是 README 的正式配置（论文用了 10000 条，我们不需要）。无头服务器和第 01 课
一样套 `xvfb-run` 前缀。预期：`datasets/carracing` 的各线程子目录下共出现约 1000
个 `rollout_*.npz` 文件（每个文件四个键：`observations`、`actions`、`rewards`、
`terminals`），磁盘增长 15-20GB；线程数和第 01 课相同时，同名旧文件会被覆盖，
无所谓，都是随机策略采的。**为什么必须补**：`data/loaders.py` 把目录里**最后
600 个文件划成测试集**，其余做训练集；1000 个文件意味着训练集 400 条、测试集
600 条（比例怪，但这是仓库原样，这里不改它）；只有第 01 课那 100 条的话训练集
为空，`trainvae.py` 会在空数据上崩掉。Mac 用户这步吃 CPU 多核，慢但可完成，
不想等可以降到 700 条（保证明显多于 600 就行），后续步骤不变。

### Step 2: 正式训练基线 VAE（latent 32、β=1）

```bash
python trainvae.py --logdir runs/L32_b1
```

给每个配置独立的 logdir（`runs/L32_b1` 这种命名后面 Step 8 会感谢自己），别复用
第 01 课的 `exp_dir`，`trainvae.py` 启动时会自动加载 logdir 里已有的
`best.tar` 续训，复用旧目录等于在冒烟模型上接着训，实验就不干净了。GPU 用户直接
跑，early stopping（30 个 epoch 无改善）会自己收工，单卡数小时量级。Mac/CPU 用户
加 `--epochs 30` 封顶，几小时能出一个够走查用的模型。预期：训练日志里 train/test
损失稳定下降后走平；`runs/L32_b1/vae/` 下出现 `best.tar` 和 `checkpoint.tar`；
`runs/L32_b1/vae/samples/` 每个 epoch 落一张 `sample_N.png`，这是从标准正态
随机采 $z$ 解码的图，翻着看它从噪声块逐渐长成"像模像样的赛道"，这本身就是
KL 项在工作的证据（空间规整到了随机点也能解码出合法画面的程度）。

### Step 3: 跑 latent 走查脚本

把下面的胶水脚本存为仓库根目录的 `vae_walk.py`：

```python
"""vae_walk.py ， 重建 / 插值 / 逐维遍历三合一。用法: python vae_walk.py runs/L32_b1 32"""
import sys
from os import listdir
from os.path import join, isdir
import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import save_image
from models.vae import VAE

logdir, lsize = sys.argv[1], int(sys.argv[2])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VAE(3, lsize).to(device)
state = torch.load(join(logdir, 'vae', 'best.tar'), map_location=device)
model.load_state_dict(state['state_dict'])
model.eval()

root = 'datasets/carracing'
files = sorted(join(root, sd, f) for sd in listdir(root)
               if isdir(join(root, sd)) for f in listdir(join(root, sd)))
tf = transforms.Compose([transforms.ToPILImage(),
                         transforms.Resize((64, 64)), transforms.ToTensor()])

def frame(fpath, t):
    obs = np.load(fpath)['observations']
    return tf(obs[min(t, len(obs) - 1)])

x = torch.stack([frame(files[-1], 60),   # 两帧锚点取自不同轨迹
                 frame(files[-2], 150)]).to(device)  # 挑到没弯道的帧就换文件或帧号重跑

with torch.no_grad():
    recon, mu, logsigma = model(x)
    save_image(torch.cat([x, recon]), join(logdir, 'walk_recon.png'), nrow=2)

    alphas = torch.linspace(0, 1, 8, device=device).unsqueeze(1)
    z_line = (1 - alphas) * mu[0] + alphas * mu[1]
    save_image(model.decoder(z_line), join(logdir, 'walk_interp.png'), nrow=8)

    sweep = torch.linspace(-3, 3, 7, device=device)
    rows = []
    for d in range(lsize):
        z = mu[0].repeat(7, 1)
        z[:, d] = sweep
        rows.append(model.decoder(z))
    save_image(torch.cat(rows), join(logdir, 'walk_traverse.png'), nrow=7)

print('已写入 walk_recon / walk_interp / walk_traverse 到', logdir)
```

然后执行：

```bash
python vae_walk.py runs/L32_b1 32
```

预期：logdir 下多出三张图。`walk_recon.png` 上排原图下排重建，赛道走向必须
认得出，路缘的红白锯齿多半糊掉（记住这个现象，Step 6 解剖它），草地纹理细节
消失、只剩两色色块（这是"扔掉废话"的样子）。三张图分别在本步和接下来两步读。

### Step 4: 读插值图：检查潜空间的"路"通不通

打开 `walk_interp.png`：8 帧从左到右，最左是第一帧的重建，最右是第二帧的重建，
中间 6 帧是潜空间连线上的点解码出来的。逐帧问一个问题：**这一帧单独拿出来，像
不像一帧合法的 CarRacing 画面？** 健康的 VAE 给出的中间帧是"赛道在连续变形"，
弯度渐变、路的位置平滑移动，每帧都像真画面；坏的信号是中间帧出现"两张图叠在
一起"的鬼影（左边一条路、右边又一条半透明的路），那说明两团编码云之间是无人区，
decoder 没见过这种 $z$，只能硬混两边的记忆。基线 β=1 的模型应该基本通畅；等
Step 8 训完对照组，回来把 β=4 和（如果你做了第 11 节改造）β=0 的插值图并排放：
β 越大中间帧越干净、越小越容易鬼影，这是 5.3 节"税率买规整"的直接可视化。
一句话把标准记牢：**插值图检验的是空间质量，重建图检验的是单点质量，两者互不
替代。** 下一课 M 的预测值全部落在"两团云之间"这种位置，所以这张图对世界模型
不是美学检查，是生死检查。

### Step 5: 读逐维遍历图：给 32 根旋钮找工作

打开 `walk_traverse.png`：32 行（每行一个维度）乘 7 列（该维从 -3 拧到 +3，其余
31 维锚定不动）。拿支笔逐行标注，把维度分成三类：

1. 管大局的：拧动时路的弯向、路的横向位置、车身相对姿态在变。找到"拧它路
   就左右弯"的那一根（或几根，弯度经常由两三维联合编码），记下行号，这是
   你要找的"弯度旋钮"，走查报告的主角。
2. 管次要观感的：拧动时草地色块分布、画面整体明暗在变，赛道结构不动。
3. 闲置维：整行 7 帧几乎一样。这是 5.3 节说的"缴不起税就缴械"的维度。

预期：32 维的基线模型三类都有，闲置维通常不少，这是正常现象而且信息量很大：
它说明 CarRacing 单帧的有效变化因素远少于 32 个，也预示了 Step 8 里 8 维模型
不会溃败。目测之外还有个严格判据：一个维度是否闲置，看它在数据集上的平均 KL
贡献是否接近零（信息为零的维度对 KL 项零贡献），Step 7 的探针脚本会顺带把每维
KL 打印出来，回头和你的手工标注对账，两者应大体吻合，这也是对你"读图能力"
的校准。另外提醒：遍历图是以某一帧为锚点的局部照相，换一帧锚点重跑一次（改脚本
里的帧号即可），职责分工的结论应该大体稳定，不稳定的维度说明编码是纠缠的，
β=4 的模型在这一点上会明显更"一维一职"，这是 β-VAE 论文解耦主张的缩小版复现。

### Step 6: 解剖重建损失：草地吃掉多少预算

存为 `loss_anatomy.py`：

```python
"""loss_anatomy.py ， 重建误差按草地/非草地拆账。用法: python loss_anatomy.py runs/L32_b1 32"""
import sys
from os import listdir
from os.path import join, isdir
import numpy as np
import torch
from torchvision import transforms
from models.vae import VAE

logdir, lsize = sys.argv[1], int(sys.argv[2])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VAE(3, lsize).to(device)
model.load_state_dict(torch.load(join(logdir, 'vae', 'best.tar'),
                                 map_location=device)['state_dict'])
model.eval()

root = 'datasets/carracing'
files = sorted(join(root, sd, f) for sd in listdir(root)
               if isdir(join(root, sd)) for f in listdir(join(root, sd)))[-600:]
tf = transforms.Compose([transforms.ToPILImage(),
                         transforms.Resize((64, 64)), transforms.ToTensor()])

err_sum = np.zeros(2)   # [草地, 非草地] 误差总量
px_sum = np.zeros(2)    # [草地, 非草地] 像素总量
for fpath in files[:100]:
    obs = np.load(fpath)['observations']
    for t in range(0, len(obs), 100):
        x = tf(obs[t]).unsqueeze(0).to(device)
        with torch.no_grad():
            recon, _, _ = model(x)
        err = ((recon - x) ** 2).squeeze(0).sum(0).cpu().numpy()
        xi = x.squeeze(0).cpu().numpy()
        grass = (xi[1] > xi[0] + 0.08) & (xi[1] > xi[2] + 0.08)
        err_sum += [err[grass].sum(), err[~grass].sum()]
        px_sum += [grass.sum(), (~grass).sum()]

for name, i in [('草地', 0), ('路面及其他', 1)]:
    print('%s: 面积占比 %5.1f%% | 损失占比 %5.1f%% | 每像素平均误差 %.5f' % (
        name, 100 * px_sum[i] / px_sum.sum(),
        100 * err_sum[i] / err_sum.sum(), err_sum[i] / px_sum[i]))
```

执行：

```bash
python loss_anatomy.py runs/L32_b1 32
```

预期与读法：先确认掩码没瞎标，把 `grass` 掩码叠回原图存一张看看（两行代码的
事），绿的地方确实是草才继续。然后看打印的三列数字。面积一列会告诉你草地是画面
的绝对大票仓（具体百分比以你的实测为准，不要引用任何"标准答案"）；损失占比和
每像素误差两列合起来读：通常非草地区域（路缘、车身这些高频细节）**每像素**误差
更高，模型确实觉得它们难；但乘上面积之后，草地在**总量**上仍然握着大话语权。
这是"按像素数投票"的账本证据：训练梯度的大头花在大面积区域上，路缘每像素
再金贵，在总损失里也是零头。把三列数字原样抄进走查报告，这是 5.4 论断的第一根
柱子。这个实验只说明损失预算怎么分配，不直接说明"决策信息丢了"，后者要靠
Step 7 的探针，两步合起来才是完整论证。

### Step 7: 弯道探针：32 个数里到底有没有"路往哪弯"

存为 `probe_road.py`。它做三件事：从像素直接计算"路面横向偏移"标签（画面上半部
路面像素质心相对画面中线的偏移，归一到 $[-1, 1]$，车前方的路偏左就是负、偏右就是
正，这是"路往哪弯"的直接代理）；用冻结的 encoder 取 $\mu$ 做特征训岭回归探针；
顺带打印每维平均 KL（给 Step 5 的闲置维判断对账）。

```python
"""probe_road.py ， 弯道线性探针。用法: python probe_road.py runs/L32_b1 32"""
import sys
from os import listdir
from os.path import join, isdir
import numpy as np
import torch
from torchvision import transforms
from models.vae import VAE

logdir, lsize = sys.argv[1], int(sys.argv[2])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VAE(3, lsize).to(device)
model.load_state_dict(torch.load(join(logdir, 'vae', 'best.tar'),
                                 map_location=device)['state_dict'])
model.eval()

root = 'datasets/carracing'
files = sorted(join(root, sd, f) for sd in listdir(root)
               if isdir(join(root, sd)) for f in listdir(join(root, sd)))[-600:]
tf = transforms.Compose([transforms.ToPILImage(),
                         transforms.Resize((64, 64)), transforms.ToTensor()])

def road_offset(obs_t):
    h, w, _ = obs_t.shape
    top = obs_t[:h // 2].astype(int)
    r, g, b = top[..., 0], top[..., 1], top[..., 2]
    road = ~((g > r + 20) & (g > b + 20))
    if road.sum() < 50:          # 上半图几乎没路（冲进草地深处），丢弃该帧
        return None
    return np.nonzero(road)[1].mean() / ((w - 1) / 2) - 1.0

feats, labels, kls = [], [], []
for fpath in files[:400]:
    obs = np.load(fpath)['observations']
    for t in range(0, len(obs), 40):
        y = road_offset(obs[t])
        if y is None:
            continue
        x = tf(obs[t]).unsqueeze(0).to(device)
        with torch.no_grad():
            mu, logsigma = model.encoder(x)
        feats.append(mu.squeeze(0).cpu().numpy())
        labels.append(y)
        kl_d = -0.5 * (1 + 2 * logsigma - mu ** 2 - (2 * logsigma).exp())
        kls.append(kl_d.squeeze(0).cpu().numpy())

X, y = np.array(feats), np.array(labels)
print('样本数:', len(y), '| 每维平均KL(四舍五入):', np.round(np.mean(kls, 0), 2))
n_tr = int(0.8 * len(y))
Xtr, Xte, ytr, yte = X[:n_tr], X[n_tr:], y[:n_tr], y[n_tr:]
Xm, ym = Xtr.mean(0), ytr.mean()
w = np.linalg.solve((Xtr - Xm).T @ (Xtr - Xm) + 1e-2 * np.eye(X.shape[1]),
                    (Xtr - Xm).T @ (ytr - ym))
pred = (Xte - Xm) @ w + ym
r2 = 1 - ((yte - pred) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
print('弯道探针 R^2 = %.3f' % r2)
```

执行：

```bash
python probe_road.py runs/L32_b1 32
```

预期：基线模型的 $R^2$ 显著大于 0（路的位置是画面里最大的结构性变化，很难完全
压丢）。两个必做的自检：其一，标签抽查，挑十来帧把 `road_offset` 的值和原图
并排看，直道接近 0、左弯为负、右弯为正才算标签合格；其二，打乱对照，把
`labels` 随机打乱后重跑探针，$R^2$ 必须掉到 0 附近，否则流程有泄漏。每维 KL 的
打印顺手和 Step 5 的手工标注对账：你标为闲置的维度，KL 应接近 0。单个模型的
$R^2$ 本身没什么可说的，它的全部意义在 Step 8 的横向比较里。

### Step 8: β 与维数对照：重建排名对得上探针排名吗

现在接上第 01 课改造清单 3 留的钩子，动仓库的两处代码，训三个对照模型。

改动一（latent 维数）：`utils/misc.py` 里那行
`ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE = 3, 32, 256, 64, 64`，把其中的 `32` 改成
`8`，训一个模型；再改成 `64`，训一个。每次训练命令相同，只换 logdir：

```bash
python trainvae.py --logdir runs/L8_b1
```

```bash
python trainvae.py --logdir runs/L64_b1
```

改动二（β 开关）：`trainvae.py` 动两行，argparse 部分加
`parser.add_argument('--beta', type=float, default=1.0)`；`loss_function` 的
`return BCE + KLD` 改成 `return BCE + args.beta * KLD`（`loss_function` 定义在
`args` 之后的模块作用域里，直接引用即可；不放心就给函数加个 `beta` 参数从调用处
传入）。把 `LSIZE` 改回 32，训 β=4：

```bash
python trainvae.py --logdir runs/L32_b4 --beta 4
```

训完立刻把 `LSIZE` 改回 32，它是全仓库共享常量，留着 8 或 64 不改，第 03
课的 `trainmdrnn.py` 会拿错维数崩给你看。四个模型（含 Step 2 的基线）每个都跑一遍
Step 3 的走查和 Step 7 的探针（`vae_walk.py` 和 `probe_road.py` 的第二个参数换成
对应维数），另外从各自训练日志里抄下最终验证损失里的重建项（你在 5.1 的验证环节
已经把两项分开打印了；如果没有，用 `loss_anatomy.py` 的误差总量代替，口径注明
即可）。汇成一张表，两列排名并排放：

| 模型 | 验证重建损失（排名） | 弯道探针 $R^2$（排名） | 插值质量 | 闲置维数 |
|---|---|---|---|---|
| L8-β1 | 填实测 | 填实测 | 目测记录 | 填实测 |
| L32-β1 | 填实测 | 填实测 | 目测记录 | 填实测 |
| L64-β1 | 填实测 | 填实测 | 目测记录 | 填实测 |
| L32-β4 | 填实测 | 填实测 | 目测记录 | 填实测 |

读表姿势（预期，但以实测为准）：重建损失一列几乎必然是 L64 最好、L8 最差、β4
垫底附近，维数是容量、β 是税率，这列只是常识复算。真正的看点是第二列和第一列
对不对得上。最常见也最有教学价值的结局：L8 的重建损失比 L32 差一大截，探针
$R^2$ 却只差一点，路的横向位置是个低维信息，8 个数也装得下，L8 丢的主要是草地
和纹理这些"重建账本上很贵、决策账本上不值钱"的东西。只要这种"重建差很多、
探针差很少"或任何排名倒挂出现一次，你就有了自己的证据：**重建损失度量的不是
下游需要的那种好。** 如果你的实测里两列排名完全一致，也如实记录，结论变成
"本设置下探针任务太容易，分不开四个模型"，并把悬念记到第 16 课（同数据换 JEPA
目标再赛一场，那时探针工具直接复用今天这两个脚本）。

### Step 9: 写 latent 走查报告

在 `runs/` 下写 `REPORT.md`，这是本课交付物，五个部分：

```text
1. 四个模型的训练记录：命令、数据规模、early stopping 停在第几个 epoch
2. 保住了什么：弯度旋钮在哪几维（附遍历图行号）、插值是否平滑
3. 丢了什么：重建里消失的细节清单（路缘锯齿、草地纹理、车影……）
4. 账本证据：损失解剖三列数字 + 四模型对照表
5. 判词：丢掉的东西里哪些无所谓、哪些有所谓，判断依据是哪个实验
```

第 5 部分逼自己下判断：草地纹理丢了无所谓（探针和后续课程都不需要它）；路缘
糊了要警惕（它是弯度的视觉载体，但只要探针 $R^2$ 高就还够用）；如果哪个模型
把"路的位置"都编糊了（探针塌了），直接判死刑。这份报告第 03 课选 V 权重时
要用，第 16 课三路对决还要再引用。

## 8. 配置与预算

| 档位 | 数据 | 训练 | 走查全家桶 | 总耗时（参考） | 用途 |
|---|---|---|---|---|---|
| Mac/CPU 档 | 700-1000 rollouts | 基线 + L8 两个模型，各 `--epochs 30` | 全部可跑（纯推理） | 采数据数小时 + 每模型数小时 | 走查方法学全掌握，对照表两行 |
| 单卡正式档 | 1000 rollouts | 四个模型，early stopping 自动收工 | 全部 | 采数据 1-2 小时 + 训练一晚 | 本课完整交付 |

补充说明：四个模型可以串行跑（一晚），显存充裕的话开两个进程并行；`--batch-size`
默认 32，24GB 卡可以加大到 128 提速，记得在报告里注明（batch 变了，损失是求和
口径，单步数值会等比变大，比较时用同 batch 的口径）。数据只采一次，四个模型
共享。探针和走查脚本都是推理，单模型几分钟。第 03 课会用本课基线 L32-β1 的
`best.tar`，把它的 logdir 记进报告，别训完就忘了哪个是哪个。

## 9. 验收

验收清单：

- [ ] 白纸默写 VAE 损失的两项，并各用一句话说明拧掉哪一项会得什么病（拧掉 KL：
      空间乱麻、插值鬼影；拧掉重建：后验塌缩、全体输出均值图）；
- [ ] 基线模型的 `walk_recon.png` 里赛道走向可辨认，且你能列出至少三样被丢掉的
      细节；
- [ ] `walk_interp.png` 的中间帧全部像合法画面，无明显鬼影；
- [ ] 在遍历图上指认出弯度旋钮（至少一维），且你标注的闲置维与每维 KL 打印
      大体吻合；
- [ ] 损失解剖的三列数字（面积占比、损失占比、每像素误差）进了报告，掩码叠图
      抽查通过；
- [ ] 探针标签抽查通过、打乱对照的 $R^2$ 归零；
- [ ] 四模型对照表填满，且你能指着"重建排名"和"探针排名"两列说出它们哪里
      一致、哪里倒挂、各说明什么；
- [ ] `LSIZE` 已经改回 32（打开 `utils/misc.py` 确认）；
- [ ] 能口头回答：为什么"重建好"不等于"压得对"？你的哪两个实验分别提供了
      账本证据和下游证据？

## 10. 排错

| 症状 | 最可能的原因 | 怎么确认 | 怎么修 |
|---|---|---|---|
| `trainvae.py` 一启动就崩（除零或空迭代） | 数据文件不足 600 个，训练集被切成空集 | 数一下 `datasets/carracing` 各子目录的 npz 总数 | 回 Step 1 补数据到 700 条以上 |
| 启动时报 `size_average` 相关警告或错误 | 新版 PyTorch 对老式损失参数的处理 | 看警告文字是否指向 `mse_loss` | 只是警告则忽略；报错就把 `size_average=False` 改成 `reduction='sum'`，数值等价 |
| 损失第一个 epoch 就异常低 | logdir 里有旧 `best.tar` 被自动续训 | 看启动日志有没有 reloading 字样 | 换新 logdir，或加 `--noreload` |
| KL 项掉到接近 0，`samples/` 全是几乎相同的糊图 | 后验塌缩：β 设太大，或把重建项改成了 mean 归约导致有效 β 暴涨 | 分开打印两项；检查 `mse_loss` 归约方式 | β 往回调；两项保持同为求和归约 |
| 插值中间帧鬼影严重 | KL 太弱（β=0 一定如此）或训练数据太少 | 对比 β=1 与更高 β 的插值图 | 提高 β 或补数据；若是 β=0 对照组，这本身就是预期结果 |
| 遍历图 32 行全都纹丝不动 | 载错 checkpoint，或走查脚本的维数参数与模型不符 | 先看 `walk_recon.png` 重建是否正常 | 核对 logdir、维数参数、`misc.py` 当前 `LSIZE` 三者一致 |
| 探针 $R^2$ 接近 0 | 标签提取错了（掩码阈值不适配，把路标成草） | 标签叠图抽查十帧 | 调 `road_offset` 里的阈值，重抽查再重跑 |
| 第 03 课 `trainmdrnn.py` 报形状不匹配 | 本课改了 `LSIZE` 忘了改回 | 打开 `utils/misc.py` 看那行常量 | 改回 32；对照组模型的走查用命令行参数区分，不靠改常量 |

## 11. 前沿与改造

你今天解剖出的毛病，重建目标按像素投票、决策信息只占零头，
正是 2018 年之后"眼睛"这个零件每次换代的靶心。第 05-06 课的 RSSM/Dreamer 仍然
保留重建，但状态不再逐帧独立，时序信息帮着分摊了压缩压力；第 08 课 TD-MPC2 直接
扔掉 decoder，表征只对"预测价值和动作后果"负责；第 09 课 IRIS 把连续 latent 换成
离散 token，重建还在但换了载体；第 10 课 DIAMOND 反其道行之，用扩散把像素细节
保到底（它的论点恰好是你实验的镜像：有些任务里弹道准星这种小细节就是命，糊不得）；
第 13-15 课的 JEPA 系是最激进的回答，像素一概不重建，只在表征空间对齐预测。
你的四模型对照表就是理解这整条演化线的钥匙：每一代都在回答"损失该向谁效忠"。

规模差距：前沿系统的眼睛在百万级视频、更高分辨率上训练，
我们是 1000 条 64×64 轨迹，这部分钱能解决。机制差距：MSE 重建这个目标本身，
钱解决不了，换目标才行，这正是后面十几课要逐个试的。

动手改造清单（选做）：

1. 路面加权重建：把 Step 6 的草地掩码搬进 `trainvae.py` 的 `loss_function`，
   草地像素的平方误差乘 0.2 再求和（掩码用 5.4 的绿色规则从 batch 图像现算）。
   预算：一个模型，单卡数小时。预期：重建图草更糊、路缘更清，探针 $R^2$ 持平或
   上升，总重建损失（旧口径）变差。失败判据：探针不升说明 32 维容量下路的信息
   本来就没丢，损失加权买不到新东西，这个结论同样值得写进报告。
2. β=0 纯自编码器：用你的 `--beta` 开关训 `--beta 0`。预算：数小时。预期：
   重建损失全场最佳，但 `samples/` 随机采样图是碎片、插值鬼影明显，用最便宜的
   方式看到"没有 KL 税，空间就不成其为空间"。失败判据：如果插值依然平滑，
   说明这份数据的流形太简单，记录下来（这也是缩小版实验的诚实边界）。
3. 用采样 $z$ 代替 $\mu$ 做探针特征：改 `probe_road.py` 两行，特征换成
   $\mu + \sigma \epsilon$。预算：几分钟。预期：$R^2$ 略降，降幅反映"云的胖瘦"
   对读数的干扰；对比各模型的降幅和它们的平均 $\sigma$，你会对"KL 把没信息的维度
   摁成纯噪声"有肉感认识。

β-VAE 论文的两个核心结论在本课缩小版里都能看方向：β 大于 1 时
逐维遍历更接近"一维管一个因素"（解耦更好）、重建质量变差，对照 L32-β1 和
L32-β4 的遍历图与损失表即可。World Models 原文"32 维 latent 足以支撑开车"的
隐含主张，本课探针给出旁证（路的位置信息在），最终裁决在第 04 课的真实分数。

## 12. 论文与延伸

1. Auto-Encoding Variational Bayes（Kingma & Welling, 2013,
   [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)），VAE 原论文。带着三个
   问题读：重参数化到底在解决哪个"估计的方差太大"问题（这才是论文的主贡献，
   别只记住结构图）？ELBO 是从什么不等式推出来的、丢掉的那部分是什么？附录 B 的
   高斯 KL 闭式解和 `trainvae.py` 里那行 `KLD` 能不能逐项对上？
2. **β-VAE: Learning Basic Visual Concepts with a Constrained Variational
   Framework**（Higgins et al., ICLR 2017,
   [OpenReview](https://openreview.net/forum?id=Sy2fzU9gl)；注意这篇没有 arXiv 版）
   ，带着两个问题读：β 是从哪个带约束的优化问题里作为拉格朗日乘子冒出来的？
   他们自己承认的代价（重建变差）和你 L32-β4 的实测对得上吗？
3. World Models（Ha & Schmidhuber, 2018,
   [arXiv:1803.10122](https://arxiv.org/abs/1803.10122)），这次只重读 V 那一节
   和模型细节附录。带着问题读：论文为什么坚持 V 单独训练、不让 M 和 C 的梯度
   碰它（提示：想想数据效率和分工，第 16 课还会回到这个决定）？交互版网页里
   拖动 $z$ 滑条的演示，对应你本课哪张图？
4. 选读：**An Introduction to Variational Autoencoders**（Kingma & Welling, 2019,
   [arXiv:1906.02691](https://arxiv.org/abs/1906.02691)），原作者六年后的长篇
   教程，5.1 节没喂饱的人去这里补全套推导。
5. 选读：**Understanding disentangling in β-VAE**（Burgess et al., 2018,
   [arXiv:1804.03599](https://arxiv.org/abs/1804.03599)），带着问题读：他们用
   什么办法在 β 很大时把重建质量救回来一部分（容量退火）？这对你的 β 扫描是
   否有可搬的改进？

下一课把 M 装上：MDN-RNN 读着你今天压出的 $z$ 序列和动作，学着预测下一个 $z$
的分布。你会发现 V 的走查报告立刻派上用场，V 压丢的信息，M 一丁点也补不回来；
而"预测必须随动作分岔"的动作对换实验，会成为整个世界模型是真是假的试金石。
