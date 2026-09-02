---
id: 11_olora_treelora
title: "低秩更新为什么要正交"
summary: "每个任务一个 LoRA，互相对齐时会抢同一方向。正交在防什么？"
unit: llm
play_tools: []
checkpoints:
  - "LoRA 方向夹角。"
  - "naive LoRA vs O-LoRA 的 4 任务矩阵。"
  - "论文复现 #3。"
---

# 第 11 课：低秩更新之间为什么要正交

> 类型：复现（论文复现 #3，方向性）+ 实战（读官方仓库、跑缩小序列）<br>
> 建议周期：3-4 天<br>
> 硬件：浏览器与 CPU 机制实验任何机器可做；单张 24GB 可跑 1B 级四任务 LoRA；T5-large / 7B 与 TreeLoRA 官方 8 任务、24 任务标加分<br>
> 锚定仓库：[cmnfriend/O-LoRA](https://github.com/cmnfriend/O-LoRA)（EMNLP 2023 Findings）；[ZinYY/TreeLoRA](https://github.com/ZinYY/TreeLoRA)（ICML 2025 官方实现）<br>
> 产物：两个 LoRA 的 $\langle A_1,A_2\rangle$ 夹角记录、naive LoRA 与 O-LoRA 的 4×4 准确率矩阵、复现 #3 方向性结论

## 1. 这一课做什么

第三幕走到一半。第 09 课把通用小模型接到窄领域上续预训练，看见通用能力掉；第 10 课把同一套问题搬到指令微调：分类、摘要、代码、数学一个接一个训，填 4×4 矩阵，用第 03 课的平均准确率、遗忘和反向迁移（BWT：学完后面任务后，前面任务是变好还是变差）量遗忘。那一课的主线是 naive 顺序 LoRA（低秩适配：冻住原权重，只训两个很小的矩阵）和「所有任务混在一起训」的上限。

缺的零件是：每个任务一个 LoRA 时，它们会不会抢同一个方向。抢到了，后一个任务的更新会把前一个任务刚写进去的那一截削掉。本课加的写法是正交（两个向量点积接近 0，互相不怎么投影到对方身上）：O-LoRA 把新任务的 LoRA 矩阵 $A$ 约束到与旧任务的 $A$ 近似正交；TreeLoRA 再加一棵按梯度相似度长出来的树，让相似任务共用浅层适配器、冲突任务分开，避免每来一个新任务就对全部旧任务算一遍梯度。

没有这一课，你只会「接着训 LoRA」，第 10 课矩阵下三角发浅的原因说不清楚，也分不清「冻结旧 adapter」和「新 adapter 真正不踩旧方向」是两件事。做完之后你能验证三件事：两个 $A$ 的内积是否被拉近 0；四任务序列上 O-LoRA 的平均准确率是否高于 naive LoRA（复现 #3 的通过线）；TreeLoRA 的树在源码里到底按什么量分组。

整门课那条主干里，本课换的是「怎么写」：覆盖改成正交约束，写的位置仍是快速权重（LoRA），不是上下文，也不是外挂记忆。第 07 课用扩结构和 prompt 把新知识写到旧权重旁边；第 08 课用梯度投影禁止踩旧任务。大模型上两条路都贵：扩结构要在推理时选专家或选 prompt，投影要存旧梯度。LoRA 已经把可训练参数降到万分之一量级（Hu et al. 对 GPT-3 175B 的数量级描述），O-LoRA 是在这组小参数上加几何约束，TreeLoRA 是在这组小参数上加一棵任务树。第 12 课会把「训的时候约束」换成「训完再把几个成品加起来」，那是另一条大模型特有的路，本课先把在线这一头做完。

术语速查：

| 术语 | 一句解释 |
|---|---|
| LoRA | 冻住原矩阵 $W$，只训低秩增量 $\Delta W$，推理时可把增量折回 $W$，不增加延迟 |
| $A$ / $B$ | 组成 $\Delta W$ 的两个小矩阵；O-LoRA 把旧任务的梯度子空间近似成 $A$ 的列空间 |
| 正交约束 | 逼 $A_i^\top A_t\approx 0$，让新任务更新少投影到旧任务已经占用的方向 |
| naive LoRA / SeqLoRA | 同一组 LoRA 参数接着训下一个任务，旧方向没有保护 |
| IncLoRA | 每个新任务加一组新 LoRA，冻旧的，但新 $A$ 仍可与旧 $A$ 对齐 |
| O-LoRA | IncLoRA 加上 $A$ 之间的正交损失，论文编号 arXiv:2310.14152 |
| TreeLoRA | 按层、按梯度相似度把任务挂到一棵树上，用 bandit 选相似枝，稀疏更新 |
| 平均准确率 AA | 学完最后一个任务后，所有任务测试准确率的平均，见 [第 03 课](03_cl_evaluation.md) |
| 复现 #3 | 顺序指令上 O-LoRA 的 AA 或遗忘方向与论文一致，不要求对齐原文小数 |

## 2. 问题

第 10 课的 4×4 矩阵如果对角线高、下三角浅，直观解释是「后任务把前任务冲了」。落到 LoRA 上，冲的具体位置是权重空间里的一个低维平面。LoRA 假设微调时 $W$ 的变化落在很低的秩上：一个 $d\times k$ 的全矩阵更新，被两个小矩阵的乘积代替。若任务 2 的 $A_2$ 和任务 1 的 $A_1$ 几乎平行，任务 2 的梯度会沿着任务 1 刚走的那条走廊继续走，任务 1 的损失会再升回去。

这和 [第 08 课](08_gem_gdumb.md) 的正交梯度下降（OGD：新梯度先投影掉旧任务梯度所在的方向再走）是同一类想法，差别在存储。OGD 要存旧数据的梯度，大模型存不起；O-LoRA 的假设是：旧任务的梯度子空间已经被那一组 LoRA 的 $A$ 列向量概括了，只要新 $A$ 和旧 $A$ 正交，就不必存梯度、也不必回放旧样本。

和 [第 07 课](07_architecture_prompts.md) 也要分清。PackNet 用掩码把一部分权重量成「任务 1 的房子」并上锁；L2P / DualPrompt 把新知识写进 prompt 池。O-LoRA 不扩网络、不加 prompt，只在已经引入的低秩方向上加几何约束。IncLoRA 已经在「扩一块」（每任务一组新 $A,B$），O-LoRA 多出来的那一项才是正交。

论文设定比梁文峰转写里说的「两个月上岗」干净得多（转写未获 DeepSeek 确认，课内只当路线判断）：训练时知道任务边界，测试时不给任务编号。真实在岗学习没有这种边界。本课先把「低秩方向互抢」这件事在干净设定里量清楚，第 12 课再看不接着训、把几个成品模型加起来能回收多少。

四个任务的语义差也值得先想一遍。dbpedia 是百科主题分类，amazon / yahoo 偏评论与问答，ag news 是新闻主题。主题分类和情感分析共用一部分「把一段话映到标签」的方向，比「代码补全接数学证明」更容易抢 $A$。所以 order 1 上 naive LoRA 会忘，并不奇怪；若你改成四个几乎相同的情感数据集，正交项的收益会变小，那是任务设计问题，不是 O-LoRA 失效的充分证据。论文另外三组 15 任务顺序（order 4/5/6）才是「长序列」压力测试，主线不做。

本课要回答的具体问题：
1. 两个 LoRA 的 $A$ 对齐时，任务 1 的有效更新被削掉多少？夹角 0° 和 90° 差在哪？
2. 只冻旧 LoRA、不加正交损失（IncLoRA），够不够？
3. O-LoRA 在四任务顺序指令上，平均准确率是否高于 naive LoRA？
4. TreeLoRA 的树在省什么：不是再发明一种正交，而是避免对全部旧任务线性扫一遍梯度。

## 3. 准备

- 第 03 课的评测协议：AA、平均遗忘、BWT。第 10 课的 4×4 矩阵写法。不必已经跑完 7B。若第 10 课还没写完报告，至少要会解释「对角线高、下三角浅」对应哪一种遗忘。
- Python 3.10+、NumPy；机制实验不下载模型、不需要 GPU。
- 要跑锚定仓库：独立虚拟环境。O-LoRA 官方依赖写在仓库 `requirements.txt`，其中 `transformers==4.28.0`、`deepspeed==0.10.0`，和你日常环境混装会炸。
- 权重与数据：O-LoRA 官方主线是 Hugging Face 的 T5-large，放到 `initial_model/t5-large`；数据在仓库的 `CL_Benchmark`。TreeLoRA 把 TRACE 与 O-LoRA 用过的数据混成 24 任务，预训练模型放到 `./PTM/`，例如 `Llama-3.2-1B-Instruct`。
- 磁盘：T5-large 加上四个任务的 adapter 和日志预留 20GB 以上；1B 级更省。
- 本课网页自带「LoRA 正交」实验，先做浏览器、再做 CPU、最后才碰仓库。

官方 O-LoRA 脚本默认 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`，那是论文配方，不是本课主线。主线按档 C：单卡 24GB 跑 1B 级或把官方脚本改成单卡小 batch。档 A 只要求浏览器和 `python3 run.py run 11`。

记录 commit SHA 从现在开始。O-LoRA 和 TreeLoRA 都在改 PEFT 相关文件，默认分支移动后形状注释可能变。报告里的路径以你 `git rev-parse HEAD` 的那次为准。克隆后先看 `README.md` 的 Setup 与 Training 两节，命令以当前 README 为准，课文里的脚本名若与 README 冲突，以仓库为准。O-LoRA 的 README 还要求把 T5-large 重命名进 `initial_model/t5-large`，少这一步脚本第一行就会找不到模型。LLaMA 路径则改名为 `initial_model/llama`，与 `scripts_llama/` 里的 `--model_name_or_path` 对齐，避免脚本去错误目录找权重。

## 4. 学习目标

1. 写出 LoRA 的前向公式，并指出 O-LoRA 论文里的 $A$ 和 Hugging Face PEFT 里名为 `lora_A` 的矩阵差一个转置。
2. 解释为什么冻住旧 LoRA 仍可能遗忘：新 $A$ 可以和旧 $A$ 对齐，隐状态仍被拉动。
3. 写出正交损失 $L_{\mathrm{orth}}(A_i,A_t)$，说明它和 Frobenius 内积 $\langle A_i,A_t\rangle_F$ 的关系。
4. 在二维上画出夹角从 0° 到 90° 时，任务 1 的更新被任务 2 削掉的比例，并和浏览器实验对上。
5. 对照官方 `uie_trainer_lora.py`，指出正交项加在哪、旧参数如何被冻结、$\lambda_1=0.5$ 写在哪条命令里。
6. 读懂 TreeLoRA 的 `KDTreeNode.build_node`：按当前层梯度与均值的点积分左右子树，叶子是单个任务或不可再分的组。
7. 交一份四任务矩阵，并给出复现 #3 的方向性判断（O-LoRA 的 AA 高于 naive LoRA，或写明本设定下的反例与配置）。

## 5. 原理

六个机制，每个按同一节奏：为什么需要、怎么运转、精确定义、源码落点、怎么验证。

### 5.1 LoRA：把一次全量更新压成两个瘦矩阵

全量微调要动整个 $W\in\mathbb{R}^{d\times k}$。大模型里这一张表动不起，也容易把预训练里已经会的事先冲掉。LoRA（Hu et al., arXiv:2106.09685）冻住 $W$，只训一个低秩增量。类比：你不重写整面墙，只在墙上贴一张可以随时撕掉的便签。类比失效处：便签贴多了会互相遮挡；若两张便签写在同一位置，后一张会盖住前一张。这正是 5.2 要量化的事。

Hu et al. 的写法是 $\Delta W=BA$，其中 $B\in\mathbb{R}^{d\times r}$、$A\in\mathbb{R}^{r\times k}$，$r\ll\min(d,k)$。前向变成

$$
h=Wx+\Delta Wx=Wx+BAx
$$

O-LoRA 论文为了强调「$A$ 的列张成更新子空间」，把字母对调成 $\Delta W=AB$，其中 $A\in\mathbb{R}^{d\times r}$、$B\in\mathbb{R}^{r\times k}$。两种写法差一个转置，几何相同。本课公式跟 O-LoRA 论文走；读 PEFT 代码时再转回来。

Hugging Face PEFT 和 cmnfriend/O-LoRA 仓库里，`lora_A` 的形状是 $(r,d)$，`lora_B` 的形状是 $(k,r)$。论文里的 $A_{\mathrm{paper}}$ 等于代码里 `lora_A` 的转置。验证：打印 `param.shape`，看到 `[r, dim]` 就知道你在代码约定里。

### 5.2 两个 $A$ 对齐时，旧任务被削掉多少

设任务 1 学完得到增量方向 $u_1$（把 $A_1$ 的列张成的子空间压成一个代表向量），任务 2 再走 $u_2$。若 $\|u_1\|=\|u_2\|=1$，任务 2 的更新在任务 1 方向上的投影是 $\langle u_1,u_2\rangle=\cos\theta$。$\theta=0^\circ$ 时投影为 1，任务 1 刚写下的那一截被等量反向或同向覆盖，旧损失最容易升；$\theta=90^\circ$ 时投影为 0，任务 2 走自己的轴，任务 1 的分量不动。

浏览器实验把这件事缩到二维：两个单位向量，拖夹角，读「任务 1 被削掉的比例」$|\cos\theta|$。CPU 实验把同一几何放到 $d\times r$ 的 $A$ 上，用矩阵内积代替二维点积。

SeqLoRA（论文基线名）更狠：连新的 $A,B$ 都不加，同一组 LoRA 接着训，旧方向直接被改写。IncLoRA 给每个任务一组新 LoRA 并冻旧的，容量增加了，但新 $A_t$ 仍可长到和 $A_i$ 平行，隐状态 $h$ 里旧任务用到的那一段仍会被 $\Delta W_t x=A_t B_t x$ 拉动。O-LoRA 论文图 3 用「新任务训完后，旧样本损失变化的直方图」说明：加上正交项（$\lambda_1=0.5$）后，旧损失的漂移明显小于 $\lambda_1=0$ 的 IncLoRA。

验证：算完两个 $A$ 之后不要只看准确率，先看 5.3 的 $O_{i,t}$。夹角没拉开、准确率却高，多半是任务太像或评测太粗。

二维数值例子，方便和浏览器对答案。取 $u_1=(1,0)$，$u_2=(\cos\theta,\sin\theta)$，任务 2 把权重加上 $u_2$ 之后，任务 1 方向上多出来的分量是 $\cos\theta$。

| $\theta$ | $\cos\theta$ | 任务 1 被削掉的比例 |
|---|---|---|
| $0^\circ$ | $1.00$ | $1.00$ |
| $30^\circ$ | $0.87$ | $0.87$ |
| $45^\circ$ | $0.71$ | $0.71$ |
| $60^\circ$ | $0.50$ | $0.50$ |
| $90^\circ$ | $0.00$ | $0.00$ |

45° 不是「一半」：一半对应 60°。浏览器实验若把「一半」设成干扰选项，选它就该判错。真实 LoRA 里 $A$ 有 $r$ 列，夹角改成子空间主夹角（最大奇异值对应的余弦），定性不变：列空间叠得越紧，旧任务越容易被后任务的 $\Delta W$ 拉动。
### 5.3 O-LoRA：用 $A$ 的列空间代理旧梯度子空间

OGD 要把新梯度投影到与旧梯度正交的补空间。大模型存全部旧梯度做不到。O-LoRA 的关键假设：微调发生在低秩子空间里，这个子空间被 $A_t$ 的列概括：

$$
A_t=[a_t^1,\ldots,a_t^r],\qquad \mathcal{U}_t=\mathrm{span}\{a_t^1,\ldots,a_t^r\}
$$

两个子空间正交当且仅当任意一对列点积为 0，也就是矩阵

$$
O_{i,t}=A_i^\top A_t=0
$$

训练目标在任务 $t$ 的指令微调似然上加正交项：

$$
\sum_{(x,y)\in\mathcal{D}_t}\log p_\Theta(y\mid x)+\lambda_1\sum_{i=1}^{t-1}L_{\mathrm{orth}}(A_i,A_t)
$$

$$
L_{\mathrm{orth}}(A_i,A_t)=\sum_{j,k}\|O_{i,t}[j,k]\|^2=\|A_i^\top A_t\|_F^2
$$

$\|A_i^\top A_t\|_F^2=0$ 当且仅当每个列对的点积为 0，也当且仅当 Frobenius 内积在 $A_i^\top A_t$ 这个 Gram 矩阵上被打没。课程口令 $\langle A_1,A_2\rangle$ 指的就是这个量（或它的归一化版本，即列空间夹角的余弦）。官方代码没用平方 Frobenius，而用绝对值求和，零点相同、梯度尺度不同：

仓库实现把旧 adapter 叫做 `lora_A`，当前任务叫做 `loranew_A`，在 `src/uie_trainer_lora.py` 的训练步进里：

$$
L_{\mathrm{code}}=\sum | \mathrm{lora\_A}\,\mathrm{loranew\_A}^\top |
$$

形状注释写的是 `[r * dim] * [dim * r]`，所以这是 PEFT 约定下的 $A_{\mathrm{code}}A_{\mathrm{code,new}}^\top$。它和论文的 $A_i^\top A_t$ 是同一组数的转置，迹与 F 范数相同。

训练时冻 $\{A_i,B_i:i<t\}$，只训当前 `{loranew_A, loranew_B}`。LoRA 插在注意力的 $W_q$ 和 $W_v$ 上，与 Hu et al. 默认一致。任务数变多时，可以把已经学完的 $\sum_i A_i B_i$ 折回 $W$，避免 GPU 上 adapter 个数线性涨。测试不需要任务编号：所有已学 LoRA 一起参与前向，所以能走指令微调那种「看到题自己判断答法」的设定，这一点论文用来对照 ProgPrompt（推理必须带任务 ID）。

$\lambda_1$ 在标准 CL 四任务（order 1/2/3）上论文和脚本都取 0.5；15 任务的 order 4/5/6 里 $\lambda_1$ 会按任务改成 0.5 或 5，并额外用 $\lambda_2$ 给 `loranew_` 加 L2。主线四任务保持 $\lambda_1=0.5$、$\lambda_2=0$。

任务数变多时论文给出折回：

$$
W_{\mathrm{init}} \leftarrow W_{\mathrm{init}}+\sum_{i=1}^{t} A_i B_i
$$

折回之后 GPU 上不再为每个旧任务单挂一份 adapter，推理成本回到一张网。代价是：折回去的方向再也拆不出来，正交约束只能保护「还没折回去、仍以 $A$ 形式挂着」的那些任务。官方训练循环是「当前任务用 `loranew_`，旧任务用冻住的 `lora_`」，折回发生在任务边界上，读 `order_1.sh` 时不要假设四段都在同一组未合并的矩阵上做正交。

假设的失效处：若真实梯度并不落在 $A$ 的列空间里，正交 $A$ 保护不了旧损失；$B$ 仍可把同一组方向组合成不同的 $\Delta W$；折回 $W$ 之后，你再也拆不出「这是任务 1 的那一截」。O-LoRA 论文自己也写了限制：训练仍需要任务边界来决定哪一份是 `loranew_`；几百个任务有没有用，他们没做。

泛化是论文相对 ProgPrompt 的卖点。他们先把 LLaMA-7B 在 Alpaca 上 LoRA 过，再接到标准 CL 四任务上，用 MMLU 当未见任务。Table 3：Alpaca-LoRA 的 MMLU 37.5；顺序 naive LoRA 掉到 23.3（四分类随机约 25）；IncLoRA 28.6；O-LoRA 33.6，CL 四任务 AA 76.8。课内 1B 四任务不必跑 MMLU；读到这里只要记住：只冻旧 adapter 救不了未见任务，正交项在他们的设定里能把 MMLU 从随机附近拉回去一截。你自己的复现 #3 不把 MMLU 当通过线。
### 5.4 和 EWC、GEM、回放的位置关系

[第 05 课](05_ewc_regularization.md) 的 EWC 给每个标量权重一根弹簧，劲度来自 Fisher 对角线。O-LoRA 不按单个权重钉，按低秩子空间钉。[第 08 课](08_gem_gdumb.md) 的 GEM / A-GEM 用旧样本梯度当不等式约束。O-LoRA 用 $A$ 代替那些梯度，因此 rehearsal-free（不回放旧用户数据）。[第 06 课](06_replay_der.md) 的 DER 仍是最稳的一类，但要存样本；本课走的是「不存样本、只存几个小矩阵」的 PEFT-CL 路线。

论文 Table 1 把方法按四列对照：是否 rehearsal-free、是否参数高效、推理是否要任务 ID、能否做未见任务。O-LoRA 四格都打勾。ProgPrompt 推理要任务 ID，所以指令模型那种「不告诉你这是哪套题」的评测它接不住。这不是骂 prompt 方法，是边界不同。

### 5.5 TreeLoRA：相似任务共用枝，避免线性扫旧任务

O-LoRA 每来一个新任务，都要对所有旧 $A$ 算正交项，旧任务数 $N$ 变大后是 $O(N)$ 次矩阵乘。TreeLoRA（Qian et al., ICML 2025, arXiv:2506.10355）要省的是这段扫描，外加挖掘任务之间的共享。直觉：浅层更像「大家都会的句法」，深层更像「这一题的标签空间」。把 LoRA 按层挂到一棵树上，根是全体任务，叶子是单个任务，中间节点是梯度方向接近的任务组。

相似度用当前样本在各旧任务适配器上的梯度差，论文写 L1。直接对每个旧任务求梯度仍是 $O(N)$。他们把「选哪个旧枝来比」做成多臂赌博机：每一轮只拉一只臂（查一个旧任务或一个节点的梯度），用下置信界（LCB：估计相似度减去不确定度，鼓励去碰样本少的节点）在树上搜。定理 1 在光滑树假设下把对 $N$ 的依赖从普通 bandit 的 $\sqrt{N}$ 收到问题相关的 $\sqrt{|J_\eta|}$；良性时接近 $\sqrt{\log N}$。这是论文自己的遗憾界，课内不拿它当实验指标。

树的构造在 `utils/kd_lora_tree.py` 的 `KDTreeNode.build_node`：当前深度取该层梯度，算均值向量，用各任务梯度与均值的点积做中位数分裂，左右子树递归，深度到 `lora_depth` 或节点只剩一个任务就停。阈值 $\delta$ 不手调，分裂时取中位数，类似 K-D tree。训练一步里，`model/Regular/Tree_LoRA.py` 把当前 `loranew_A` 当成梯度代理塞进树，用 `tree_search` 得到最像的旧枝，再把

$$
\ell_{\mathrm{reg}}^t=\|\hat g_n^t-\hat g_k^t\|_1
$$

加进损失（实现里 `tree_lora_loss` 用的是负点积，鼓励当前梯度靠近选中旧枝）。`--reg` 默认 0.5，脚本 `scripts/lora_based_methods/Tree_LoRA.sh` 写明。ViT 上论文把树深设为 5，LLM 上设为 64，且深度不得超过 Transformer 层数。推理时他们选用最近学完的 adapter，测试同样不给任务 ID。

TreeLoRA 和 O-LoRA 不是互斥叙事：O-LoRA 强制「别踩旧方向」；TreeLoRA 强制「相似的靠在一起走，搜的时候别线性扫」。官方仓库把 O-LoRA 做成基线（`model/Regular/O_LoRA.py` 几乎是 `uie_trainer_lora.py` 那套正交项的移植）。本课主线仍是正交；树是效率零件，以读代码和官方小实验为主。

### 5.6 交付时要量的两张表

第一张：$\langle A_1,A_2\rangle$。对每个插入了 LoRA 的模块，取任务 1 的 `lora_A` 和任务 2 的 `loranew_A`（或两份保存下来的 $A$），算

$$
\cos(A_1,A_2)=\frac{\langle A_1,A_2\rangle_F}{\|A_1\|_F\|A_2\|_F},\qquad \langle A_1,A_2\rangle_F=\mathrm{Tr}(A_1^\top A_2)
$$

O-LoRA 训完应接近 0；naive / IncLoRA 通常明显大于 0。CPU 实验把阈值钉死：无约束时余弦 $>0.3$，有正交项时 $<0.08$。

第二张：4×4 准确率矩阵。行是「学完任务 $j$ 之后」，列是「在任务 $i$ 上测」。对角线是刚学完时的成绩，下三角是遗忘。四任务用 O-LoRA 论文 order 1：dbpedia、amazon、yahoo、ag news。学完任务 4 后的一行平均就是 AA。复现 #3 只要求：同一数据、同一模型规模下，O-LoRA 的 AA 高于 SeqLoRA / naive LoRA，方向与论文 Table 2 一致。论文在 T5-large、order 1 上 SeqLoRA 44.6、O-LoRA 75.4、多任务上限 80.0；你的 1B 或合成数据不会得到同一小数，不要抄进报告里冒充复现。

## 6. 源码导读

按一条样本的路径读，不要按文件名排序。先 O-LoRA，再 TreeLoRA。结论以你克隆时的 commit 为准。

### 6.1 cmnfriend/O-LoRA

| 路径 | 带着什么问题读 |
|---|---|
| `src/uie_trainer_lora.py` | 正交项在 `backward` 之前加到 `loss` 上；`lora_A` 对 `loranew_A` 做 `torch.mm(param, param_.T)` 后取绝对值再求和 |
| `src/run_uie_lora.py` | 命令行 `--lamda_1`、`--lamda_2` 从这里进 trainer |
| `src/peft/tuners/lora.py` | PEFT 的 $A,B$ 形状、哪些模块被注入；O-LoRA 只动 Q/V |
| `scripts/order_1.sh` | 四任务顺序：dbpedia、amazon、yahoo、agnews；每段 `--lamda_1 0.5 --lamda_2 0`；下一段的 `--model_name_or_path` 指向上一段的 `.../adapter` |
| `scripts/order_2.sh`、`scripts/order_3.sh` | 同一四个数据集，排列不同；论文 Table 2 的 Order-1/2/3 |
| `scripts_llama/order_1.sh` | LLaMA 路径，batch 改成 1、梯度累积 8，仍是 8 卡 DeepSpeed |
| `configs/order1_configs/` | 每个任务的 train/dev/test json，任务边界写在这里 |
| `configs/instruction_config.json` | 指令模板：任务定义、选项、正文、答案 |

`order_1.sh` 里每一段都是一次完整的 `deepspeed src/run_uie_lora.py`。第一段从 `initial_model/t5-large` 出发；之后每一段读上一段输出的 adapter，这样旧 `lora_A` 已经在模型里，新任务的 `loranew_A` 才能和它做矩阵乘。若你自己改成单卡，只改 `CUDA_VISIBLE_DEVICES` 和 batch，不要把四段训缩成一段，否则旧 $A$ 进不了正交项。

正交项的匹配逻辑是字符串：`name.split("lora_A")[0] == name_.split("loranew_A")[0]`，保证同一层、同一模块才比。匹配成功后 `break`。漏匹配的层等于没加约束，夹角量出来会偏大。

### 6.2 ZinYY/TreeLoRA

| 路径 | 带着什么问题读 |
|---|---|
| `model/Regular/Tree_LoRA.py` | `train_one_task`：`--reg>0` 时取所有 `loranew_A`，stack 成 `(lora_depth, dim)`，`insert_grad`，再 `tree_search` + `get_loss`；任务结束 `end_task` |
| `utils/kd_lora_tree.py` | `KDTreeNode` 中位数分裂；`tree_lora_loss` 对选中旧枝做负点积；`end_task` 用相邻任务梯度差更新树 |
| `model/Regular/O_LoRA.py` | 对照：同一套 `lora_A`/`loranew_A` 正交项，没有树 |
| `scripts/lora_based_methods/Tree_LoRA.sh` | TRACE 八任务、`--CL_method Tree_LoRA`、`--reg 0.5`、默认 4 卡 DeepSpeed |
| `scripts/lora_based_methods/O_LoRA.sh` | 同一数据与模型上的 O-LoRA 对照 |
| `scripts/lora_based_methods/lora.sh` | SeqLoRA 对照 |
| `training/main.py`、`training/params.py` | `--CL_method` 映射到 `Tree_LoRA` / `O_LoRA` / `lora` |
| `scripts/O_LoRA_Dataset/order1/lora_based_methods/Tree_LoRA.sh` | 24 任务混合基准里的 order 1，加分项 |

README 写明 24 任务是 TRACE 与 O-LoRA 数据的混合，表里能看到 C-STANCE、Py150、ScienceQA、yelp、amazon、SST-2 等。官方小实验用的是 TRACE 八任务：`C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten`，数据目录 `data/LLM-CL-Benchmark/LLM-CL-Benchmark_500`。全量 24 任务不要当主线。

保存 adapter 时 `Tree_LoRA.save_model` 会把 `adapter_config.json` 里的 `r_sum` 写成 0，注释写这是为了和 O-LoRA 的 PEFT 配置兼容。读权重时若加载失败，先查这个字段。

## 7. 实验

三层都要做。浏览器和 CPU 是机制；锚定仓库才碰真实指令数据。论文复现 #3 只在第四层的四任务矩阵上判定方向，冒烟分数不算复现。

### Step 0: 浏览器「LoRA 正交」先预测再运行

打开本课页面下方的交互实验（页面里 `id="interactive-lab"` 那一块）。两个二维单位向量表示任务 1、任务 2 的低秩方向，夹角从 0° 拖到 90°。运行前先预测：

- 0° 时，任务 1 被削掉的比例最接近 100%、50% 还是 0%？
- 90° 时呢？
- 45° 时是不是「一半」？

机制：削掉的比例是 $|\cos\theta|$，不是 $\theta/90$。45° 的余弦约 0.71，被削的比一半多。预测写对再按运行。改滑块会作废上次结果，需要重新预测。

### Step 1: CPU 机制实验

在课程仓库的 `experiments/` 目录：

```bash
python3 run.py run 11
```

实验用固定种子在合成 rank-1 LoRA 上钉几何，结果写入 `artifacts/lesson11/result.json`。现在应当全绿，五条 `checks` 全为真：

| check | 本机为真时在说什么 |
|---|---|
| `task1_lora_works` | 任务 1 的 LoRA 准确率高于 0.90 |
| `naive_cosine_not_near_zero` | 无约束时两个 $A$ 的 $|\cos|$ 大于 0.25 |
| `olora_cosine_near_zero` | 任务 2 的 $A$ 正交投影后 $|\cos|$ 小于 0.08 |
| `olora_retains_task1_at_least_as_well` | 两份 adapter 叠在一起后，任务 1 不低于 naive |
| `olora_learns_task2` | 正交后任务 2 准确率仍高于 0.75 |

本机一次运行（Python 3.13.13，seed 11）：naive $|\cos|=0.508$，正交后 0；叠后任务 1：0.931 vs 0.985。换机器数字会变，方向不应变。

这一层是八维上的两个 rank-1 方向，不证明 T5-large 官方 O-LoRA 表。复现 #3 仍走下面的 O-LoRA 仓库，只要求方向。不要手写一份假 JSON。

锚定仓库上量夹角时，按 PEFT 形状把两个 $A$ 都看成 $(r,d)$，余弦用

$$
\cos=\frac{\sum_{jk}A_1[j,k]A_2[j,k]}{\|A_1\|_F\|A_2\|_F}
$$

不要先把矩阵拉成向量再和另一层的 $A$ 胡乱点。只对同一层、同一模块（例如第 3 层的 `q_proj`）成对计算，再对所有模块取平均或取最大。报告里写清楚你报的是平均还是最大：平均会被已经正交的层稀释，最大更能暴露没加上约束的那一层。

通过标准：`checks` 全为真。它证明的是几何，不证明 T5-large 上的官方表。

### Step 2: 克隆 O-LoRA 并核对正交项

```bash
git clone https://github.com/cmnfriend/O-LoRA.git
```

```bash
git -C O-LoRA rev-parse HEAD
```

把 SHA 写进报告。打开 `src/uie_trainer_lora.py`，找到 `orthogonal_loss = 0.` 那一段，对照 5.3 的公式：代码是 `torch.abs(torch.mm(param, param_.T)).sum()`，论文是 $\sum_{j,k}\|O_{i,t}[j,k]\|^2$。零点相同。再打开 `scripts/order_1.sh`，确认四段任务名和 `--lamda_1 0.5`。

### Step 3: 四任务 naive LoRA vs O-LoRA（复现 #3 主线）

官方论文配方是 8 卡 T5-large、`bash scripts/order_1.sh`。档 C 用缩小版：同一四任务顺序，模型换成你机器装得下的（1B 级或仓库支持的 LLaMA 路径），`CUDA_VISIBLE_DEVICES=0`，`per_device_train_batch_size` 降到 1，必要时加梯度累积。$\lambda_1$ 保持 0.5。对照臂把正交项关掉：同一脚本把 `--lamda_1` 改成 0，这就是 IncLoRA；再跑一组真正的 SeqLoRA（不新增 `loranew_`，接着训同一组 $A,B$）。

每一任务结束，把已经见过的全部任务重测一遍，填 4×4。用第 03 课公式算 AA 和 BWT。

目录不要覆盖：naive 和 O-LoRA 分两个 `output_dir`。官方默认输出在 `logs_and_outputs/order_1/outputs/TASK_NAME/predict_results.json`。

矩阵怎么填。学完任务 $j$ 后，对 $i=1,\ldots,j$ 各测一次准确率 $a_{i,j}$。没学过的格子留空，不要填 0 冒充忘光。四任务全部结束后你手里是下三角加对角线共 10 个数；AA 只用最后一列四个数的平均。BWT 用第 03 课的定义：对 $i<T$，比较 $a_{i,T}$ 和刚学完 $i$ 时的 $a_{i,i}$。SeqLoRA 常见形状：对角线尚可、最后一列前面几个接近随机。O-LoRA 常见形状：最后一列四个数都离各自对角线不远。IncLoRA 往往对角线高、最后一列前几个掉一截但比 SeqLoRA 浅。

夹角和矩阵一起看。若 O-LoRA 的 $\cos(A_1,A_2)$ 已经小于 0.08，但 4×4 最后一列仍塌，优先怀疑评测解码（`generation_max_length` 太短、指令模板和训练不一致），而不是先加 $\lambda_1$。若余弦根本没降，矩阵再漂亮也不能写「正交起了作用」。

通过标准（方向性）：O-LoRA 的 AA 高于 SeqLoRA。IncLoRA 通常介于两者之间：冻旧 adapter 已经有用，正交项再补一截。若你的设定里 O-LoRA 不高于 naive，报告里写模型规模、$\lambda_1$、任务顺序和 4×4，标成反例，不算默默改通过线。
论文 Table 2 仅作对照，禁止把 75.8 抄进你的结果表。

### Step 4: TreeLoRA 读代码并跑官方小实验

```bash
git clone https://github.com/ZinYY/TreeLoRA.git
```

读 `utils/kd_lora_tree.py` 的 `build_node` 和 `model/Regular/Tree_LoRA.py` 的 `train_one_task`。用纸画出：三个任务到来时，树如何从根长出左右子树。核对 README 的 Quick Start：数据在 `data/LLM-CL-Benchmark`，模型在 `./PTM/`。

有 4 卡再跑官方 TRACE 八任务。先设模型名：

```bash
export model_name="Llama-3.2-1B-Instruct"
```

```bash
bash scripts/lora_based_methods/Tree_LoRA.sh
```

脚本里 `gpu_nodes="0,1,2,3"`、`epochs=2,1,3,2,1,2,2,3`、`reg=0.5`。单卡把 `gpu_nodes` 改成 `"0"`，并接受变慢或 OOM。这是实战档：跑通、记下 OP / BWT / 墙钟时间，不宣称对齐论文 Table 3 的 7B 分数。

没有多卡：停在读代码。把 `end_task` 里「用相邻任务梯度差更新树」那几行抄进笔记，说明树在省 $O(N)$ 梯度查询。

读代码时画一张三任务草图就够：任务 1 结束，树只有根加一个叶子；任务 2 结束，`end_task` 用两份累积 `loranew_A` 的差当「任务 2 的方向」，按当前层与均值的点积决定进左枝还是右枝；任务 3 训练时 `tree_search` 每步只选一个旧枝做正则，不会对任务 1 和任务 2 各算一遍完整反向。若日志里从未出现 `Prev_id_matrix`，搜索没工作，时间对 SeqLoRA 的优势写不出来。论文在 ViT 上报告相对先前方法最多约 3.2 倍训练加速、LLM 上约 2.4 倍，那是他们完整设定的墙钟比，你的八任务小跑只需记录自己的秒数，不要把 3.2 写进报告当测到的数。
### Step 5: 24 任务与 7B（加分）

`scripts/O_LoRA_Dataset/order1/lora_based_methods/Tree_LoRA.sh` 与 `O_LoRA.sh` 是 24 任务混合基准。7B 需要档 D。加分项单独写报告，不并进复现 #3 的通过线。

## 8. 配置与预算

| 档 | 做什么 | 大概时间和显存 | 算不算复现 #3 |
|---|---|---|---|
| A 浏览器 + CPU | Step 0–1 | 分钟级，无 GPU | 否，只证几何 |
| C 单卡 24GB | 1B 级四任务 naive vs O-LoRA | 数小时到一天，看 epoch 和序列长度 | 是，方向性 |
| D 多卡 | 官方 T5-large 8 卡 order_1；TreeLoRA 4 卡 TRACE 八任务 | 按仓库 | 是（T5-large）；TreeLoRA 算实战 |
| 加分 | 24 任务或 7B | 按仓库，明显更贵 | 单独标注 |

O-LoRA 四任务官方超参（论文 Appendix A.1 与 `scripts/order_1.sh`）：T5-large，1 epoch，学习率 $1\times 10^{-3}$，常数调度，batch 64（8 卡 × 8），$\lambda_1=0.5$，$\lambda_2=0$，LoRA 加在 $W_q,W_v$。rank $r$ 论文在 T5-base 上扫过 2/4/8/16，平均准确率差不到 2 个点（Table 5），主线不必盲追大 $r$。

TreeLoRA 官方脚本：学习率 $1\times 10^{-4}$，cosine 调度，`--zero_stage 2`，`--reg 0.5`，prompt 最长 1024、答案最长 512。八任务的 epoch 向量在 `Tree_LoRA.sh` 里写死，不要随手改成全 1 再和论文比时间。

缩小而不改题意：任务数保持 4、顺序保持 dbpedia 到 amazon 到 yahoo 到 ag、$\lambda_1$ 保持 0.5。可以缩小的是模型、batch、生成长度。把四任务改成两个、或把 $\lambda_1$ 调到能过线，都使复现 #3 作废。

## 9. 验收

- 浏览器：0° / 45° / 90° 三次预测，45° 不能答「一半」，运行结果与 $|\cos\theta|$ 一致。
- CPU：`python3 run.py run 11` 现在应当全绿，`checks` 全真：`task1_lora_works`、`naive_cosine_not_near_zero`、`olora_cosine_near_zero`、`olora_retains_task1_at_least_as_well`、`olora_learns_task2`。`result.json` 里看两个 $A$ 的余弦和叠后任务 1，没有四任务 AA。
- 夹角：O-LoRA 训练后至少一组模块的 $\cos(A_1,A_2)<0.08$；naive 同一模块 $>0.3$。把模块名和数值写进报告。
- 4×4 矩阵：naive LoRA 与 O-LoRA 各一张，附 AA 与 BWT。
- 复现 #3：O-LoRA 的 AA 高于 naive LoRA（SeqLoRA），或写明反例的模型、$\lambda_1$、顺序。方向与 Wang et al. 2023 Table 2 一致即可。
- 源码：能指出 `uie_trainer_lora.py` 正交项的两行，以及 TreeLoRA `build_node` 用点积中位数分裂。
- TreeLoRA：读代码笔记必交；官方八任务能跑则交 OP/BWT 和时间，跑不跑不挡复现 #3。
- 分档诚实：CPU 几何不是论文复现；T5-large 官方数字不是你的数字；24 任务是加分。

复现报告最小结构：配置（模型、卡数、$\lambda_1$、$r$、任务顺序、commit SHA）、两张 4×4、一行 AA/BWT 对照、一组 $\cos(A_1,A_2)$、一句方向性结论。不要把论文 Table 2 的 75.8 写进「我的结果」。若只完成了浏览器和 CPU，报告标题写成「机制验证」，不要写「复现 O-LoRA」。

现在系统长什么样：你会在顺序指令上给每个任务挂一份 LoRA，并用 $A$ 的正交减少互抢；也知道相似任务可以按梯度挂到树上省扫描。还缺的是：已经各自训完的几个模型，能不能在不回放、不再训练的情况下缝成一个多任务模型。那是第 12 课的任务向量算术。
## 10. 排错

| 症状 | 原因 | 验证 | 修法 |
|---|---|---|---|
| `naive_cosine_not_near_zero` 为假 | 无约束两方向已经接近正交 | `metrics.naive_abs_cosine` | 看 5.2：任务 2 应与任务 1 有重叠，不要把两任务设成 90° |
| `olora_cosine_near_zero` 为假 | 正交投影没把旧 $A$ 抠掉 | `metrics.olora_abs_cosine` | 看 5.3：新 $A$ 应减去在旧 $A$ 上的投影；对照仓库里的 `orthogonal_loss` |
| `olora_retains_task1_at_least_as_well` 为假 | 叠 adapter 后任务 1 比 naive 更差 | `olora_stacked_task1` 对 `naive_stacked_task1` | 正交后任务 2 仍应能学（`olora_learns_task2`）；否则是投影把可学方向削光了 |
| 正交损失恒为 0 | 旧 `lora_A` 没加载进当前模型 | 打印所有 `named_parameters()` 里含 `lora_A` 与 `loranew_A` 的名字 | 确认第二段 `--model_name_or_path` 指向上一段 `adapter`；名字前缀能对上 |
| 余弦降不下去 | $\lambda_1=0$，或只训一组 LoRA | 日志里的 `λ1` 和 `orthogonal_loss` | `--lamda_1 0.5`；不要用 SeqLoRA 脚本冒充 O-LoRA |
| O-LoRA 还不如 naive | 任务太像、或 $r$ 太小、或生成长度截断 | 看 4×4：若对角线也低，是没学会新任务 | 先保证新任务对角线接近 IncLoRA；再查正交项是否过强 |
| DeepSpeed / transformers 报版本错 | 官方钉死 4.28.0 与 deepspeed 0.10.0 | `pip show transformers deepspeed` | 独立虚拟环境按 `requirements.txt` 装 |
| 8 卡脚本在 1 卡 OOM | `per_device_train_batch_size 8` 按 8 卡写的 | `nvidia-smi` | 单卡 batch=1，加 `--gradient_accumulation_steps` |
| TreeLoRA `r_sum` 加载失败 | 保存时被写成 0 以兼容 O-LoRA | 读 `adapter_config.json` | 按 `save_model` 的注释处理，不要混用未改过的 PEFT 加载逻辑 |
| 24 任务磁盘或时间炸掉 | 把加分项当主线 | 任务列表长度是否为 24 | 退回 TRACE 八任务或四任务 |
| 夹角接近 0 但旧任务仍掉 | $A$ 正交不等于 $\Delta W$ 正交，隐状态仍可能被 $B$ 拉动 | 对比冻 $B$ 与不冻 $B$ | 记进报告：正交是 $A$ 列空间上的代理，不是旧损失的充分条件 |

## 11. 前沿与改造

PEFT-CL 在 2024–2025 年的一条线是：新任务的更新必须落在对旧任务干扰小的子空间里。O-LoRA 用 $A$ 的正交当代理；InfLoRA（Liang & Li, CVPR 2024, [arXiv:2404.00228](https://arxiv.org/abs/2404.00228)）把「无干扰」写进注入子空间的构造，微调注入参数等价于在一个预先消掉旧任务干扰的子空间里动预训练权重。TreeLoRA 把效率放到前台：梯度相似树 + LCB，避免对旧任务线性扫描。C-LoRA（Smith et al., arXiv:2304.06027）走另一条正则：让新 LoRA 不要和历史 LoRA 太像，用在扩散模型持续定制，塑性往往更差。2025 年起还有三条对照：OLieRA（Cao & Wu, [arXiv:2509.06100](https://arxiv.org/abs/2509.06100)）把加法更新换成李群上的逐元素乘法；Sculpting Subspaces（Nayak et al., [arXiv:2504.07097](https://arxiv.org/abs/2504.07097)）用自适应 SVD 把正交约束接到全量微调；KeepLoRA（Luo et al., [arXiv:2601.19659](https://arxiv.org/abs/2601.19659)）把新梯度投到预训练主空间和旧任务特征之外。

我们差在：主线只有四任务、小模型、没有把 InfLoRA 的子空间投影实现进课内实验；TreeLoRA 的遗憾界和 3.2× 加速是论文数字，课内不自动复现。

动手改造（01–12 课精简版，仍要可执行）：

1. 把 `uie_trainer_lora.py` 的 `abs().sum()` 改成论文的平方 Frobenius，比较 `orthogonal_loss` 曲线和最终 AA。预算：四任务 1B 再跑一遍。预期：零点相同，余弦下降速度可能变。失败：AA 差超过 2 个点却说「只改了等价实现」而不查学习率。
2. 在 CPU 实验里加第三臂：随机正交初始化 $A_2$ 但不加损失。预期：初始化正交会被 SGD 拉斜，最终余弦回升。失败：若不回升，检查学习率是否为 0。
3. 读 InfLoRA 的注入公式，用 5.3 的符号写清它和 $A_i^\top A_t=0$ 的差别（谁在约束 $A$，谁在约束允许的梯度子空间）。不要求训练。
4. TreeLoRA：把 `--reg` 设成 0，对照 `reg=0.5` 的 OP。预算：官方八任务缩小 epoch。预期：无正则时更接近 SeqLoRA。失败：两边分数难分且树从未打印 `Prev_id_matrix`，说明搜索没跑起来。

顺手复现映射：本课是复现 #3。改损失的那一项仍算同一编号，只要四任务协议不变。InfLoRA 不进本课编号。

## 12. 论文与延伸

1. Wang et al., 2023, *Orthogonal Subspace Learning for Language Model Continual Learning*, [arXiv:2310.14152](https://arxiv.org/abs/2310.14152)。
贡献：用 LoRA 的 $A$ 列空间代理旧梯度子空间，加 $A_i^\top A_t=0$，不回放。机制发明处，不是本课主阅读。
机制：每个新任务挂一组新 $A,B$ 并冻旧的；损失里加 $\|A_i^\top A_t\|_F^2$。官方仓库用绝对值求和，零点相同、梯度尺度不同。测试时所有已学 LoRA 一起前向，不给任务编号。
和本课：`olora_cosine_near_zero` 与 `olora_retains_task1_at_least_as_well` 对应「夹角拉开、叠后旧任务更稳」。课内 CPU 每步把 $A_2$ 投影掉 $A_1$，是硬约束；论文主文是软惩罚。T5-large 四任务 AA 走 Step 3 的 4×4，CPU 答不了那张官方表。
阅读问题：`olora_abs_cosine` 已经接近 0 之后，叠后任务 1 是否仍不低于 naive？若否，论文的 $A$ 代理假设在你的 rank-1 合成数据上哪里破了？

2. Hu et al., 2021, *LoRA: Low-Rank Adaptation of Large Language Models*, [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)。
贡献：冻住 $W$，只训低秩增量 $\Delta W=BA$。机制发明处，不是本课主阅读。
机制：改的是可训练参数的位置和数量，不改持续学习损失。摘要写相对 GPT-3 175B 的 Adam 全量微调，可训练参数降约 1 万倍、显存约 3 倍；$B$ 零初始化、$A$ 高斯初始化，推理可把增量折回 $W$。
和本课：5.1 的字母约定；CPU 用 `np.outer(adapter_a, adapter_b)` 当 rank-1 $\Delta W$。论文不讨论正交，也没有旧任务考试。
阅读问题：Hu et al. 的正交项作用在哪一个矩阵上？没有。O-LoRA 的正交项作用在论文约定的 $A$ 上。你打印 PEFT 的 `lora_A` 形状时，有没有把 $(r,d)$ 当成论文的 $A$？

3. Liang & Li, 2024, *InfLoRA: Interference-Free Low-Rank Adaptation for Continual Learning*, [arXiv:2404.00228](https://arxiv.org/abs/2404.00228)。
贡献：把无干扰写进注入子空间的构造，微调注入参数等价于在该子空间里动预训练权重。
机制：新任务先设计降维矩阵 $B_t$，使更新落在 $B_t$ 的行张成的子空间里；只训 $A_t$，冻 $W$、旧枝和 $B_t$。HTML 命题 1：训 $A_t$ 等价于在 $\mathrm{span}\{b_1^t,\ldots,b_r^t\}$ 里动 $W$。约束写在 $B_t$ 的构造里；O-LoRA 把同一几何写进损失。设定是无回放、类增量、推理不给任务 ID，主实验是 ViT。
和本课：改造清单第 3 项；CPU 的硬投影更接近「子空间里才允许动」，不像 O-LoRA 原文的软损失。InfLoRA 的 $B_t$ 设计与 ViT 表，本课 4×4 答不了，因为没有实现那条注入公式。
阅读问题：打开 `lesson_11.py` 里 `freeze_a` 那几行。它是在损失里加正交项，还是每步把 $A_2$ 投影掉 $A_1$？用这条 check 的实现回答 InfLoRA 和 O-LoRA 差在约束写在哪。

4. Qian et al., 2025, *TreeLoRA: Efficient Continual Learning via Layer-Wise LoRAs Guided by a Hierarchical Gradient-Similarity Tree*, [arXiv:2506.10355](https://arxiv.org/abs/2506.10355)。
贡献：按层把 LoRA 挂到梯度相似树上，用 LCB bandit 搜枝，稀疏更新。
机制：改的是「和哪个旧任务比」以及正则怎么加，不是再发明一种 $A_i^\top A_t=0$。树用梯度与均值的点积中位数分裂；每步只拉一只旧枝。仓库 `tree_lora_loss` 用负点积鼓励靠近选中枝，`--reg` 默认 0.5。
和本课：Step 4 读 `KDTreeNode.build_node` 与 `tree_search`。CPU 只有两个 rank-1 方向，没有树，答不了加速或遗憾界。
阅读问题：仓库 `tree_lora_loss` 用负点积，论文把相似度写成 L1。你的八任务小跑若从未打印 `Prev_id_matrix`，这两项差别你能区分吗？本课 CPU 答不了，因为没有树。

5. Cao & Wu, 2025, *Orthogonal Low-rank Adaptation in Lie Groups for Continual Learning of Large Language Models*, [arXiv:2509.06100](https://arxiv.org/abs/2509.06100)。
贡献：在任务子空间正交之外，用李群上的乘法更新保住参数几何。
机制：把参数看成 Hadamard 李群，更新写成 $W\odot\exp(\Delta W)$，再对任务子空间加正交。仍无回放、推理仍不给任务 ID。它改的是更新运算（加法换成乘法），不是评测协议。
和本课：CPU 与 O-LoRA 仓库都是 $W+AB$。乘法更新、Standard CL 上的分数，本课实验答不了，因为没有实现指数映射。
阅读问题：若只把本课正交项从平方 F 范数改成绝对值求和，这算不算论文里的李群更新？用 5.3 的前向公式回答。指数映射本课实验答不了，因为 CPU 和仓库都是加法 $W+AB$。

6. Nayak et al., 2025, *Sculpting Subspaces: Constrained Full Fine-Tuning in LLMs for Continual Learning*, [arXiv:2504.07097](https://arxiv.org/abs/2504.07097)。
贡献：用自适应 SVD 做全量微调，更新正交到旧任务关键方向，不加每任务新参数、不存旧梯度。
机制：对每个权重矩阵做 SVD，大奇异值方向当旧知识，小奇异值方向留给新任务；新梯度投到与旧表示正交的低秩子空间。摘要写在 T5-Large 与 LLaMA-2 7B 上平均准确率可比 O-LoRA 高到 7 个点。改的是全量梯度的允许子空间，不是 LoRA 损失。
和本课：主线是 LoRA，没有全量 SVD 臂。CPU 的 `olora_cosine_near_zero` 只量两个 $A$ 的夹角，答不了「权重矩阵的主奇异方向有没有被保护」。
阅读问题：若你在 1B 四任务上把 O-LoRA 换成全量微调再做 SVD 投影，`task1_lora_works` 这条 check 还适用吗？本课实验答不了，因为没有全量臂。

7. Feng et al., 2025, *Recurrent Knowledge Identification and Fusion for Language Model Continual Learning*, [arXiv:2502.17510](https://arxiv.org/abs/2502.17510)。
贡献：用内环估参数重要性、外环剪冗余再合并，让重要性随训练改写。
机制：内环在新任务上快适应并标出重要坐标；外环按当前重要性分布做剪枝与关键知识合并，多轮重复。改的是合并与掩码，不是 $A$ 的正交损失。摘要写模型从 770M 到 13B。
和本课：本课既不估 Fisher 式重要性，也不做多轮融合。`olora_learns_task2` 只说明正交后新任务还能学，答不了「重要性分布有没有随任务改」。
阅读问题：O-LoRA 冻旧 $A$ 是不是一种静态重要性？本课能指出「冻的是整份 adapter，不是按坐标估的重要性」；论文里的多轮融合本课实验答不了，因为没有外环。

8. Ling et al., 2025, *LoRA-Based Continual Learning with Constraints on Critical Parameter Changes*, [arXiv:2504.13407](https://arxiv.org/abs/2504.13407)。
贡献：正交 LoRA 之后，旧任务的关键参数仍会动；先冻 ViT 里那些关键矩阵，再用 QR 合成正交 LoRA。
机制：学后任务前，把对前任务最关键的参数矩阵冻住；在正交 LoRA 上用 QR 做 LoRAC。改的是哪些预训练矩阵还允许被 LoRA 碰到。摘要写 Split CIFAR-100 准确率高 6.35 个点、遗忘低 3.24 个点。
和本课：主线是语言模型四任务，CPU 没有 ViT，也没有「关键矩阵」选择。摘要里 Split CIFAR-100 的两个百分点本课实验答不了，因为没有那张视觉基准。
阅读问题：本课冻的是旧任务的 $A,B$，论文冻的是 ViT 里被标成关键的预训练矩阵。这两份冻结对象一样吗？用 5.3 的「只训 `loranew_`」回答前半句。关键矩阵探针本课实验答不了，因为没有 ViT 层重要性扫描。

9. Chen et al., 2024/2026, *Replay-Free Continual Low-Rank Adaptation with Dynamic Memory*, [arXiv:2411.00623](https://arxiv.org/abs/2411.00623)。
贡献：每个 ViT 层并行挂正交 LoRA 和残差 LoRA，推理时用动态记忆按输入压残差枝。
机制：正交枝 $O$ 只沿与旧特征子空间正交的方向更新；残差枝 $R$ 在旧任务残差基里更新。推理用动态记忆按样本相关性调制 $R$，并估计任务身份、校准输出。改的是更新投影和推理路由，无回放。
和本课：CPU 只有一份 $A$，没有残差枝，也没有按输入选枝。`olora_retains_task1_at_least_as_well` 能说明硬投影保住任务 1，答不了 DualLoRA 的推理期调制。
阅读问题：若去掉残差枝、只留正交枝，这还剩本课哪一条几何？用 `olora_cosine_near_zero` 回答；任务身份校准本课实验答不了，因为 CPU 不预测任务编号。

10. Luo et al., 2026, *KeepLoRA: Continual Learning with Residual Gradient Adaptation*, [arXiv:2601.19659](https://arxiv.org/abs/2601.19659)。
贡献：通用知识在权重主空间，任务知识在残差空间；新 LoRA 梯度投到两者之外。
机制：对预训练注意力权重做 SVD，大奇异值当主空间，小奇异值当残差。新任务梯度再正交到主空间和旧任务特征主导方向，LoRA 只在残差里写。改的是梯度投影，不是正交损失。主实验是 CLIP 与 LLaVA。
和本课：CPU 把 $A_2$ 投影掉 $A_1$，保护的是任务 1 的 LoRA 方向，不是预训练主空间。`naive_cosine_not_near_zero` 说明无约束会抢方向；预训练主空间有没有被动，本课实验答不了，因为没有对 $W$ 做 SVD。
阅读问题：KeepLoRA 要同时保住预训练能力和旧任务。本课 `task1_lora_works` 只保证任务 1 的 LoRA 能分类，它能当「预训练主空间还在」的证据吗？不能，写明差在哪一份矩阵。

11. Le & Venkatesh, 2026, *Continual Fine-Tuning of Large Language Models via Program Memory*, [arXiv:2605.13162](https://arxiv.org/abs/2605.13162)。
贡献：把 LoRA 做成可检索的程序记忆槽，按输入取槽，再写回底层 adapter。
机制：槽按输入条件注意力动态取用，相似输入复用同一区域，空槽留给以后的数据；取到的槽与底层 adapter 合并。仍在 LoRA 参数化里，摘要写推理不增加成本。改的是 LoRA 内部的分区与检索，不是 $A$ 两两正交。
和本课：TreeLoRA 的树是按梯度相似度分组；ProCL 的槽是按输入检索。CPU 和 O-LoRA 仓库都没有槽注意力。本课实验答不了「相似输入是否复用同一槽」，因为没有按样本选 adapter 的探针。
阅读问题：若把 TreeLoRA 的叶子当成槽，检索键是梯度还是输入？用 `kd_lora_tree.py` 的分裂量回答；ProCL 的输入条件注意力本课实验答不了，因为仓库没实现那套槽。

下一课把「接着训」这条路放到一边：几个已经训好的模型，权重直接加减。任务向量加法有时能得到多任务模型，TIES 要修符号冲突，负任务向量能做粗遗忘。那是事后缝合，不是本课这种在线约束。带着你的 4×4 去 [第 12 课](12_model_merging.md)：合并能不能在不看旧数据的情况下，回收 naive LoRA 丢掉的那一截。若正交已经把夹角拉开，合并时符号冲突会少一些，第 12 课的 TIES 修剪量也可以对照着看。
