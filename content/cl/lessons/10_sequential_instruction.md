---
id: 10_sequential_instruction
title: "指令任务一个接一个"
summary: "先教数学再教摘要，模型会不会只会最后那件事？"
unit: llm
play_tools: []
checkpoints:
  - "4×4 准确率矩阵。"
  - "和「混训上限」的差距。"
---

# 第 10 课：四个指令任务接着训，填一张遗忘矩阵

> 类型：实战（顺序指令 LoRA；TRACE 全量太大，课内跑 4 任务子集并写明）<br>
> 建议周期：2-4 天<br>
> 硬件：Mac / CPU 完成浏览器实验和 `python3 run.py run 10`；单张 24GB 卡跑 135M 或 1B 级 LoRA（主线）；7B LoRA 标加分<br>
> 锚定仓库：[BeyonderXX/TRACE](https://github.com/BeyonderXX/TRACE)（论文 [arXiv:2310.06762](https://arxiv.org/abs/2310.06762)）；任务序列与 4 任务对照可改用 [cmnfriend/O-LoRA](https://github.com/cmnfriend/O-LoRA) 仓库内 `CL_Benchmark`（论文 [arXiv:2310.14152](https://arxiv.org/abs/2310.14152)，课前核对后的编号，不是 2305.18870）<br>
> 产物：一张 4×4 准确率矩阵、用 [第 03 课](03_cl_evaluation.md) 协议算出的遗忘与 BWT、以及和「四任务混在一起训」上限的差距

## 1. 这一课做什么

[第 09 课](09_continual_pretraining.md) 把通用基座接到 Python 上续训，测的是领域对通用：下一词分布换了，WikiText 和填空会不会一起塌。那是 Wu et al. 2024 三阶段里的第一截（CPT）。本课进入第二截：持续指令微调（CIT）。数据不再是无标签混料，而是一条条「指令 + 输出」。写的位置仍是权重，但主线改成 LoRA（低秩适配：冻住基座，只训两块小矩阵），这样 24GB 卡跑得动，也为 [第 11 课](11_olora_treelora.md) 的正交约束把接口留好。

具体场景：先教分类，再教摘要，再教代码补全，再教简单数学。每教完一个，把前面全部任务重测一遍，填进 4×4 矩阵。行是「刚训完哪个任务」，列是「现在考哪一科」。对角线是刚学会时的分数；对角线以下（若行是时间、列是任务号）是后来还记不记得。只用最终平均分，会把「只会最后一科」的模型夸成好学生，这正是第 03 课要打假的事。

锚定有两条，必须分清，不要混报数字。

- TRACE 是给已经对齐过的 LLM 准备的基准：8 个任务，覆盖领域知识、多语言、代码、数学，每任务采样 5,000 条训练、2,000 条测试。仓库入口是 `train.py` 与 `scripts/train_lora.sh`。全量 8 任务 × 7B 超出课内主线，所以主线只跑 4 个任务的子集，每任务最多 500 条训练（TRACE 论文在 RCL 对照里也用过 500 条这一档）。
- O-LoRA 仓库自带 `CL_Benchmark` 数据和 `scripts/order_1.sh`。标准 CL 的 order 1 本身就是 4 个分类任务：DBpedia → Amazon → Yahoo → AG News。数据在仓库里，不依赖 Google Drive。官方实现是 T5-large；课内主线把同一套 4 任务改接到 SmolLM2-135M-Instruct 的 LoRA 上，方便单卡。T5-large 官方脚本标加分，7B 再标一档加分。

做完你手里有三样东西：naive 顺序 LoRA 的 4×4；四任务打乱混训的那一行上限；用第 03 课公式算出的平均准确率、平均遗忘、BWT。下一课才在这块 LoRA 上加正交。本课故意先看「只换 PEFT、不加持续学习约束」会忘成什么样。TRACE 在 LLaMA-2-7B-Chat 上给过一个清楚的对照：全参数顺序微调的 Overall Performance 是 48.7（BWT −8.3%），LoRA 顺序微调是 12.7（BWT −45.7%）。LoRA 省显存，它本身不是抗遗忘方法。

和第 09 课的分工再钉一次。第 09 课的旧能力是「通用下一词」，新能力是「Python 下一词」，两条曲线可以画在同一张时间图上。本课的旧能力是「任务 1 的指令」，新能力是「任务 2 的指令」，必须用矩阵，因为任务数大于 2，平均分会撒谎。第 09 课的 30% WikiText 回放，对应「旧分布的无标签样本」；本课若要回放，混的必须是旧任务的指令-输出对。把 WikiText 塞进 CIT 的 batch，救的是语言流畅，救不了 AG News 的标签词。

[第 08 课](08_gem_gdumb.md) 的 GDumb 在这里几乎搬不进来：你没法把四个指令任务的 500 条攒进一个小缓冲、再从头训一个 135M 还宣称这是便宜上限。混训上限扮演了那个「让人难堪」的角色：若顺序 LoRA 远差于混训，说明接龙本身在掉分；若两者接近，先怀疑 500 条太小、四科还没真正分开。

术语速查：

| 术语 | 一句解释 |
|---|---|
| 持续指令微调 CIT | 一条任务流上连续做指令-输出监督，目标是新任务会听，旧任务还能听 |
| LoRA | 冻住 $W$，只训低秩更新 $\Delta W=BA$，推理时可把 $BA$ 加回 $W$，不增加延迟 |
| SeqLoRA | 同一块 LoRA 上按任务顺序接着训，不加回放、不加正交。本课 naive |
| IncLoRA | 每个任务一块新 LoRA，旧块冻住。无正交时仍会在隐空间里互相干扰 |
| 4×4 矩阵 | 元素 $R_{t,i}$：训完第 $t$ 个任务之后，第 $i$ 个任务的测试分数 |
| 混训上限 MTL | 四个任务的样本搅在同一个 dataloader 里一次训完，当作本设定的乐观上限 |
| Overall Performance | TRACE 用的平均：训到第 $t$ 个任务时，对已见任务分数取平均 |
| GAD / IFD / SD | TRACE 额外三项：通用能力、指令跟随、安全在顺序训练后的变化量 |
| 输出格式崩 | 后一个任务把答案模板改写，前一个任务还知道内容，却开始用错格式交卷 |

## 2. 问题

指令微调过的模型，看起来什么都会一点。把它再拿去「先数学后摘要」，会不会只剩下最后那种答法？这个问题和 Split MNIST 是同一类，只是任务边界更脏：分类要一个标签词，摘要要一段话，代码要下一行，数学要一个数。后一个任务的梯度会改 LoRA，也会间接触到生成分布的「该怎么收尾」。

TRACE 的设计针对的就是「现有 CL 基准对对齐后的大模型不够难」：AG News 这类分类很多已经进过 Flan 一类指令集，模型可能见过；TRACE 选了 ScienceQA、FOMC、MeetingBank、C-STANCE、20Minuten、Py150、NumGLUE 两个子任务，并且额外要你测通用能力、指令跟随、安全，而不是只报目标任务平均分。论文里 LLaMA-2-13B-Chat 在 GSM8K（8-shot CoT，表 2）从 43.14 掉到顺序全参数训练后的 2.12。抽象段另一处写 28.8% 到 2%，评测协议不同，课内引用以表 2 带 shot 设置为准。方向一致：目标任务序列训完，数学推理这种「通用能力」可以塌掉。

O-LoRA 论文把 naive 写得更干净。T5-large、标准 5 任务 CL 的三个顺序上，SeqLoRA 平均准确率 43.7，O-LoRA 75.8，多任务上限 80.0；拉长到 15 任务，SeqLoRA 平均只剩 1.6，几乎在随机瞎猜。这些数字属于 T5-large 和他们的数据切分，你的 135M 矩阵不会、也不许抄成自己的结果。它们只说明：顺序 LoRA 可以忘到只剩最后一个任务。

本课要你自己回答的是更小、但可复现的四个问题：

1. 四个指令任务按固定顺序 LoRA 训完，4×4 的下三角（先学的任务，后来再考）是不是明显低于对角线？
2. 用第 03 课的 Average Accuracy、Average Forgetting、BWT 算完，会不会出现「平均分还行、BWT 很差」？
3. 把四个任务混在一起训的上限，和顺序 LoRA 的差距有多大？差距来自遗忘还是来自根本没学好新任务？
4. 忘的是标签空间、解题步骤，还是交卷格式？TRACE 观察到带推理路径的 ScienceQA 对保住推理更友好；NumGLUE 只有数字答案时，推理能力照样掉。课内 4 任务里至少要有一个「答案很短」和一个「答案是一段话」，才能看见格式崩。

全量 TRACE 的 40,000 条训练、16,000 条测试、再加 OpenCompass 和 GPT-4 打的指令/安全，主线做不起。子集必须在笔记第一行写明：哪些任务、每任务多少条、和论文 5,000 条档差多少。写「复现 TRACE」而实际只跑了 500 条 × 4，是把课写坏。

TRACE 八个任务按论文第 4.1 节归类如下，主线子集从每类里各抽一个（多语言那一类本课丢掉，避免 135M 中文/德语评测先垮掉、把遗忘和「根本不会这门语言」混在一起）：

| 类别 | 任务 | 课内是否采用 | 主指标 |
|---|---|---|---|
| 领域分类 | FOMC 鹰派/鸽派 | 采用，当分类 | 准确率 |
| 领域长文本 | MeetingBank 会议摘要、ScienceQA | 采用 MeetingBank 当摘要 | ROUGE-L 或 token F1 |
| 代码 | Py150 下一行补全 | 采用 | 行级精确匹配 |
| 数学 | NumGLUE-cm、NumGLUE-ds | 采用 cm | 数字精确匹配 |
| 多语言 | C-STANCE 中文立场、20Minuten 德语简化 | 主线不用 | 全量才碰 |

这个四件套对应课程蓝图写的「分类、摘要、代码补全、简单数学」。它们的输出形态差得够远：一个词、一段话、一行代码、一个数。O-LoRA order 1 四个都是分类，遗忘仍然会发生（标签词互抢），但格式崩会比较轻。两条路测到的病不同，报告里写你走的是哪条，不要把分类四任务的矩阵解释成「摘要也被冲掉了」。

## 3. 准备

- 第 03 课的矩阵公式要能默写；第 09 课知道基座和 Instruct 不能混用。本课加载 Instruct 或带指令模板的生成模型，因为任务是听指令，不是续预训练。
- 档 A：浏览器热力图 + CPU 机制实验。CPU 实验用缩小的 4 任务分数矩阵钉死「对角线高、下三角遗忘」，不下载模型。
- 档 C 主线：`HuggingFaceTB/SmolLM2-135M-Instruct` + PEFT LoRA，序列 512，单卡 24GB。1B 级（TinyLlama-1.1B-Chat 或 SmolLM2-1.7B-Instruct 的 LoRA）同脚本换模型名，标为加分但仍算「1B 级主线加宽」，不是 7B。
- 数据两条路，选一条当主结果，另一条当对照即可：
  - 路 B（推荐先走，数据在 git 里）：clone O-LoRA，用 `CL_Benchmark` 的 order 1 四个分类任务，指令模板用论文附录的 Topic / Sentiment 提示。
  - 路 A（更贴 TRACE）：从 README 的 Google Drive 链接下载处理后数据，只保留 FOMC、MeetingBank、Py150、NumGLUE-cm。每任务训练 500、测试 200（从 2,000 测试里固定种子抽）。Drive 下不下来就不要空等，改路 B，并在报告写原因。
- 软件：`transformers`、`peft`、`datasets`、`accelerate`。跑官方 TRACE 脚本还需要他们的 `requirements.txt`（README 写明实验环境 CUDA 12.2、torch 2.0.1）。官方脚本和课内 PEFT 胶水不要混在同一个虚拟环境里硬装，版本冲突时给 TRACE 单独建环境。
- 不要在本课启用 O-LoRA 的正交损失。那是第 11 课的开关。本课 naive 必须是「会忘」的那条，否则矩阵打假没有对照。

## 4. 学习目标

1. 指出本课落在 Wu et al. 的 CIT，并说明它和第 09 课 CPT 在损失、模板、评测矩阵上的三点不同。
2. 写出 LoRA 的前向公式，数清 135M 模型在 $r=8$、只挂 `q_proj`/`v_proj` 时大约新增多少可训练参数（数量级即可）。
3. 按第 03 课协议，从一张 4×4 矩阵算出 Average Accuracy、Average Forgetting、BWT；能指出哪一种假矩阵平均分高但 BWT 差。
4. 在单卡上跑完顺序 LoRA 与混训上限，提交矩阵和差距。
5. 用至少一例说明「格式崩」：旧任务内容还在，交卷模板被后任务改写。
6. 说出 TRACE 全量和本课子集的差别，以及 LoRA 在 TRACE 论文里并没有自动减轻遗忘这一事实。

## 5. 原理

### 5.1 CIT：听指令的权重，也会被下一条指令覆盖

CPT 吃的是无标签文本，模型继续做 $p(x_t\mid x_{<t})$。CIT 吃的是 $(指令, 输出)$，损失通常只打在输出 token 上（prompt 侧 label 为 `-100`）。写人话：你现在教的是「看见这句人话，该吐哪种答卷」，不是「网页上下一个词更像什么」。

Wu et al. 把 CIT 再分成任务增量、领域增量、工具增量。本课 4 任务是任务增量：标签空间和输出形态都在变。Continual-T0（Scialom et al. 2022）表明，带旧任务回放的指令模型可以连续学新生成任务；O-LoRA 则走无回放、靠正交子空间。本课 naive 两边都不走，先把病跑出来。

监督目标：第 $t$ 个任务的数据 $\mathcal{D}_t=\{(x,y)\}$，

$$
\mathcal{L}_t(\theta)=-\mathbb{E}_{(x,y)\sim\mathcal{D}_t}\sum_{k\in M_y}\log p_{\theta}(y_k\mid x,y_{<k})
$$

$M_y$ 只含答案位置。顺序 CIT 的残酷处在于 $t$ 时刻你只能碰到 $\mathcal{D}_t$。第 09 课至少还能混 WikiText；本课 naive 连旧指令都不混。混训上限把 $\bigcup_t\mathcal{D}_t$ 一次优化，相当于告诉模型四科要一起考试。

验证：同一条分类样本，分别用「只含任务 $t$ 的模板」和「混了任务 $t+1$ 模板的模型」解码。若后者开始用摘要口吻回答分类，就是 CIT 遗忘，不是 CPT 那种 Python 污染英语。

### 5.2 为什么先 LoRA，而不是一上来全参数

全参数顺序微调在 TRACE 上对目标任务更贴合，Overall Performance 高于 LoRA，但通用能力（MMLU、GSM、BBH）掉得也更狠。LoRA 把可训练参数砍到基座的百分之几，24GB 单卡才能把 1B 级甚至 7B 塞进去。Hu et al. 2021（arXiv:2106.09685）的观察是：适配所需的更新经常有低「本征秩」。冻住 $W_0$，令

$$
h=W_0x+\Delta Wx=W_0x+BAx
$$

其中 $B\in\mathbb{R}^{d\times r}$，$A\in\mathbb{R}^{r\times k}$，$r\ll\min(d,k)$。推理时可以把 $BA$ 合并进 $W_0$，深度不变。PEFT 里常再乘 $\alpha/r$。O-LoRA 论文采用 $A\in\mathbb{R}^{d\times r}$、$h=W_0x+ABx$，秩仍是 $r$，只是字母左右对调。课内跟 PEFT 默认：$A$ 是降维、$B$ 是升维。

SmolLM2-135M 隐层 576。只对每层 `q_proj`、`v_proj` 插 LoRA、$r=8$，粗算每层约 $2\times(576\times 8+8\times 576)=18{,}432$ 个参数（GQA 时 k/v 维更小，这个数是上限量级）。30 层就是约 0.55M，相对 135M 不到 0.5%。省的是显存和checkpoint 体积，不是遗忘。

SeqLoRA：四任务共用这一组 $A,B$，训完任务 2，$A,B$ 已经为任务 2 改过，任务 1 的低秩方向被覆盖。IncLoRA：任务 2 另开 $A_2,B_2$，旧的 $A_1,B_1$ 冻住；没有正交时，$A_2$ 仍可学到和 $A_1$ 平行的方向，前向相加时互相干涉。O-LoRA 要 $A_i^\top A_t=0$，那是第 11 课。本课主线跑 SeqLoRA，加分可加一列 IncLoRA（无正交），用来预告「光加模块不够」。

验证：打印 `model.print_trainable_parameters()`。可训练比例应在 1% 量级。若你看到 100%，说明 LoRA 没挂上，全参数在更新，矩阵会变成另一种实验。

### 5.3 4×4 矩阵和第 03 课协议

记 $R_{t,i}$ 为刚完成第 $t$ 个任务训练之后，在第 $i$ 个任务测试集上的分数（分类用准确率，数学用精确匹配，摘要用 ROUGE-L，代码用下一行精确匹配）。$t,i\in\{1,2,3,4\}$。矩阵必须在每个任务结束时做一次全量重测，不能只在全部结束时测一行。

第 03 课 / Lopez-Paz & Ranzato 2017（GEM, arXiv:1706.08840）的核心三个量，本课这样落地（$T=4$）：

平均准确率，全部结束后：

$$
\mathrm{AA}=\frac{1}{T}\sum_{i=1}^{T}R_{T,i}
$$

后向迁移，负值就是遗忘：

$$
\mathrm{BWT}=\frac{1}{T-1}\sum_{i=1}^{T-1}\bigl(R_{T,i}-R_{i,i}\bigr)
$$

平均遗忘，采用 Chaudhry 常用的「历史最好减最终」（课内若第 03 课给了另一精确式，以第 03 课课文为准，并在报告里写你用的是哪一条）：

$$
\mathrm{AF}=\frac{1}{T-1}\sum_{i=1}^{T-1}\Bigl(\max_{t\in\{i,\ldots,T\}}R_{t,i}-R_{T,i}\Bigr)
$$

学习准确率 $R_{i,i}$ 衡量「当时到底学会没有」。FWT 需要任务 $i$ 在训练前的零样本 $b_i$：

$$
\mathrm{FWT}=\frac{1}{T-1}\sum_{i=2}^{T}\bigl(R_{i-1,i}-b_i\bigr)
$$

135M 在摘要和代码上的零样本可能接近 0，$b_i$ 要实测，不能假设是随机分类的 $1/C$。

TRACE 的 Overall Performance 在训到 $t$ 时是 $\mathrm{OP}_t=\frac{1}{t}\sum_{i=1}^{t}R_{t,i}$，BWT 的分母他们写成 $t$ 而不是 $t-1$（论文公式 2-3）。同一张矩阵两种分母会差一个常数因子。课内主表用第 03 课（$T-1$ 分母），若你对照 TRACE 日志，先换算再比。

打假用的假矩阵（不要当实验结果）：最后一列全 1、其余全 0，则 AA 可以是 0.25 到很高（若最后任务权重被平均掩盖，需要更多任务才夸张）。第 03 课的 5×5「只会最后一件事」在 4×4 上同样成立：AA 可能看起来不像灾难，BWT 一定差。浏览器实验会让你先猜哪类任务最容易被后任务冲掉。

用一张**标明为演算**的矩阵练手。下面数字是编的，只用来对公式，禁止抄进结果文件。

| 训完 \\ 测试 | 任务1 | 任务2 | 任务3 | 任务4 |
|---|---|---|---|---|
| 任务1 后 | 0.85 |  |  |  |
| 任务2 后 | 0.50 | 0.82 |  |  |
| 任务3 后 | 0.42 | 0.55 | 0.78 |  |
| 任务4 后 | 0.40 | 0.35 | 0.50 | 0.80 |

空白格是「尚未训练的未来任务」，算 AA/BWT 用不到它们（算 FWT 才需要任务开始前的 $b_i$）。于是

$$
\mathrm{AA}=(0.40+0.35+0.50+0.80)/4=0.5125
$$

$$
\mathrm{BWT}=\bigl[(0.40-0.85)+(0.35-0.82)+(0.50-0.78)\bigr]/3=-0.40
$$

平均分 51% 看起来不像崩盘，BWT −0.40 说明旧任务从刚学会时掉了四十个百分点。若有人只在摘要里写 AA，就会把这个方法说成「还行」。你的 GPU 矩阵把这三行计算附在图下，数字换成你跑出来的。

和第 06 课回放的衔接：若在每个新任务的 batch 里混 10% 旧指令，这张演算矩阵的下三角通常会抬起来，对角线可能略降。那是 CIT 版的配方 3。本课 naive 先不混，是为了让下三角有机会掉下去；加回放是第 11 节的改造 2，不是本课通过线。

### 5.4 后任务如何把前任务冲掉

三条常见通道，探针要能分开。

1. 标签词覆盖。Amazon 情感是 positive/negative，AG News 是世界/体育/商业/科技。生成式分类都在同一套词表上抢「该输出哪个单词」。SeqLoRA 会把 $B$ 的列转向新标签。
2. 长度与模板。摘要要长输出，分类要一个词。解码若仍用同一套 `max_new_tokens` 和停止符，前任务会被截断或被后任务的冗长模板带着跑。这是评测协议问题叠加遗忘，报告里要写死每任务的解码配置。
3. 推理链被抹掉。TRACE 发现 ScienceQA 这种带 rationales 的答案有助于保住 BBH；NumGLUE 只给数字时，推理掉得很快。课内数学任务如果只训练最终数字，不要惊讶通用四则运算探针会塌。

Kotha et al. 2023（arXiv:2309.10105）从隐式推断角度讨论语言模型遗忘；课内不实现那篇，只借用一句：表面上还在用同一指令模板，模型内部已经把「这类指令该调哪套程序」换掉了。

验证：对任务 1 的测试集，同时记录准确率和「输出是否仍匹配任务 1 的合法集合」。合法集合外的输出记为格式错误。格式错误率上升、但把输出映射到最近标签后准确率仍高，说明主要是格式崩。映射后仍然差，才是知识/决策边界被冲。

评测协议还有三处容易把遗忘测「假消失」。第一，分类任务若用 teacher-forcing 的 token loss 当准确率，模型可以在训练格式上 loss 很低，但自由解码已经改口；主矩阵必须用自由解码。第二，不同任务共用同一个 `max_new_tokens=128`，分类会被后任务的长句带着走，摘要会被截断；这是解码超参，不是权重遗忘。第三，Instruct 模板的 `assistant` 头如果在任务 2 之后被训成另一种结束符，任务 1 的解析脚本会把整段判错。解析脚本要按任务写，并在附录贴 2 条 raw 输出。发现「修解析之后 BWT 少了一半」，把这件事写进报告：你测到的一部分是协议，不是模型。

### 5.5 混训上限不是持续学习，但是一把尺子

O-LoRA 表 2 的 MTL 在标准 CL 上是 80.0，PerTaskFT（每任务一个模型）是 70.0。顺序方法若超过 PerTaskFT，说明有前向迁移；若远低于 MTL，说明共享一套参数时任务在打架。课内混训：四个训练集 concatenate 后 shuffle，同一 LoRA、同一总步数或同一总 token。公平性有两种，必须写明你用哪一种。

- 等步数：顺序是 4×$N$ step，混训也是 $4N$ step。混训每任务见到的次数更均匀。
- 等每任务步数：混训 $N$ step 里四任务各约 $N/4$，对顺序不公平。

主线用等总步数，并在笔记写「混训每条样本被看到的期望次数」。混训会泄漏「未来任务」的信息，所以它是上限，不是可部署的持续学习。GDumb 在第 08 课能打脸，是因为图像任务边界干净、缓冲重训强；指令任务的混训上限同样强，而且往往需要同时看见四科的指令模板。若你的顺序方法接近混训，先检查是不是数据太少、模型根本在死记 500 条，四科都没真正分开。

### 5.6 TRACE 还多测的三件事，主线只读不跑

训完目标任务，对齐过的模型可能：通用知识掉、不再听系统指令、安全话术被冲。TRACE 定义为

$$
\Delta R_t^{G}=\frac{1}{M}\sum_{i=1}^{M}\bigl(R_{t,i}^{G}-R_{0,i}^{G}\bigr)
$$

指令跟随 $\Delta R_t^{I}$、安全 $\Delta R_t^{S}$ 形式相同。评测通用用 MMLU、BBH、TyDiQA、BoolQ、PIQA；指令和安全用 GPT-4 打分。135M 主线不算这三项：OpenCompass 和 GPT-4 预算都不在档 C 的「数小时」里。7B 加分若要碰 GAD，只允许用公开免费的小型通用集（课内 20 题或 HellaSwag 子集），禁止把未实际跑的 TRACE 表 2 抄进你的 GAD。

读这些公式是为了第 16、24 课：在岗学习不能只看目标任务矩阵。本课 4×4 已经够把「只会最后一科」钉死。

## 6. 源码导读

先读 TRACE，再读 O-LoRA。两份代码都是同一组作者风格，路径以当前 README 为准。

TRACE（[BeyonderXX/TRACE](https://github.com/BeyonderXX/TRACE)，写课时 README 与 `train.py`、`scripts/` 仍在默认分支）：

| 路径 | 带着什么问题读 |
|---|---|
| `README.md` | 环境是 CUDA 12.2 / torch 2.0.1；数据 Drive 链接；`CL_method` 可取 `lora` / `base` / `EWC` / `OGD` / `GEM` / `O-Lora` 等 |
| `scripts/train_lora.sh` | LoRA 顺序训练入口。改任务列表和 `data_path` 就在这里 |
| `scripts/infer_lora.sh` | 对 `output_dir` 里每个任务结束后的模型做推理 |
| `scripts/train_seq_naive.sh` | 全参数顺序 SFT，主线 135M 可对照，7B 不要贸然开 |
| `scripts/train_replay.sh` | `past_task_ratio` 控制旧任务回放。本课 naive 不用，但要知道上限之外还有这条 |
| `train.py` | 真正的训练循环、`CL_method` 分支 |
| `utils/data/` 下的 dataset 与 collator | README 写数据应是 `prompt`/`answer` 列表；decoder-only 左 padding |
| `metrics.py` | 目标任务指标。GAD 不在这里，通用能力走 OpenCompass |
| `evaluations/`、`inference/` | 推理与 3H（helpful/honest/harmless）脚本 `scripts/infer_3H.sh` |

README 里八个训练任务名必须抄对：C-STANCE、FOMC、MeetingBank、Py150、ScienceQA、NumGLUE-cm、NumGLUE-ds、20Minuten，外加回放用的 Lima。课内子集取 FOMC、MeetingBank、Py150、NumGLUE-cm，顺序固定为这个，避免每人一个顺序无法对矩阵。若某份数据缺文件，在笔记里换成 ScienceQA 顶上 MeetingBank，并写明。

官方命令（各占一个围栏，cwd 为仓库根目录）：

```bash
pip install -r requirements.txt
```

```bash
bash scripts/train_lora.sh
```

```bash
bash scripts/infer_lora.sh
```

未改脚本之前，这三条会按仓库默认跑全序列。主线必须先改 `train_lora.sh` 里的任务列表和每任务样本上限，再跑。不要把未改脚本的全量 7B 日志当成课内结果。

O-LoRA（[cmnfriend/O-LoRA](https://github.com/cmnfriend/O-LoRA)）：

| 路径 | 带着什么问题读 |
|---|---|
| `README.md` | T5-large 放到 `initial_model/t5-large`；LLaMA2 放到 `initial_model/llama` |
| `scripts/order_1.sh` | dbpedia → amazon → yahoo → ag |
| `scripts/order_2.sh`、`scripts/order_3.sh` | 另外两个 4 任务顺序，用来检查顺序敏感性 |
| `CL_Benchmark/` | 任务数据。确认每个任务有 train/test |
| `src/` | LoRA 与正交损失。本课读前向，不打开正交项 |
| `logs_and_outputs/order_1/outputs/TASK_NAME/predict_results.json` | 官方结果落点 |

T5-large 官方复现命令：

```bash
pip install -r requirements.txt
```

```bash
bash scripts/order_1.sh
```

README 把日志重定向写在同一行里。风格指南要求每个 bash 围栏一条命令，课内把重定向拆开：先跑 `order_1.sh`，需要后台再自己加。T5-large + 4 任务在 24GB 上 LoRA 应当能站住；他们论文附录是 8×3090 跑多种顺序，那是全量实验，不是你必须的硬件。

课内 PEFT 胶水（路 B 主线）最小前向：

```python
from peft import LoraConfig, get_peft_model, TaskType
cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(base, cfg)
```

SmolLM2 是 GQA，模块名以 `model.named_modules()` 打印为准。若 `q_proj` 不存在，不要瞎猜，把实际名字写进脚本。

## 7. 实验

### Step 0: 浏览器实验「任务热力图」

打开本课页面的交互实验（`lab-10-task-heatmap`）。你会看到一张 4×4 空热力图，行是训练进度，列是四个任务（分类、摘要、代码、数学）。先预测：哪一类最容易被后任务冲掉？典型选项包括「分类，因为标签词少、后任务一张嘴就把词抢走」「摘要，因为长输出模板霸道」「代码，因为 token 分布离自然语言最远」「数学，因为答案短、梯度把数值格式改掉」。选完才能运行。系统会播放一张按简化动力学生成的矩阵：颜色越浅越忘。对完预测才算过关。改任务顺序或「是否混训」开关必须重新预测。

这一步对应 5.3 和 5.4。它用的不是你的 GPU 数字，而是同一套「后任务覆盖前任务方向」的缩小模型。你稍后提交的真实 4×4 应当和这里的深浅模式同类：对角线深、下三角浅。若你的真实矩阵四格一样深，先查评测是不是每次都测成了最后一个任务。

### Step 1: CPU 机制实验

```bash
python3 run.py run 10
```

在 `experiments/` 目录执行。现在应当全绿：写出 `artifacts/lesson10/result.json`，五条 `checks` 全真。

| check | 本机为真时在说什么 |
|---|---|
| `matrix_is_4x4` | 准确率矩阵是 4×4 |
| `diagonal_all_above_0_88` | 对角线全高于 0.88（刚学完的任务都学会了） |
| `lower_triangle_forgotten` | 下三角均值比对线最低值低 0.12 以上 |
| `bwt_negative` | BWT 低于 −0.10 |
| `last_row_peaks_on_last_task` | 最后一行最大值在最后一列 |

本机一次运行（Python 3.13.13，seed 10）：对角线最低 0.991，下三角均值 0.428，BWT=−0.677，最终平均准确率 AA=0.488。换机器数字会变，方向不应变：对角线高、下三角浅、最后一行峰值在最后一列。

这一层是四个互相冲突的二维方向（0°/90°/180°/270°）顺序训练，不下载 TRACE 数据，不证明 TRACE 八任务。混训上限在 Step 5，课内 CPU 不跑混训。不要手写一份假 JSON。

### Step 2: 克隆仓库并准备环境

```bash
git clone https://github.com/BeyonderXX/TRACE.git
```

```bash
git clone https://github.com/cmnfriend/O-LoRA.git
```

路 B 主线用 O-LoRA 的数据 + 自己的 PEFT 脚本，TRACE 仓库用来对照 README 与任务定义。路 A 在 TRACE 环境中：

```bash
pip install -r requirements.txt
```

数据：README 写「All the data after processing can be downloaded from Trace Benchmark」并给出 Drive 文件 id `1S0SmU0WEw5okW_XvP2Ns0URflNzZq6sV`。下好后目录应能让 `data_path` 指到含八个任务子目录的位置。只保留四个任务的 `train.json` / `test.json`，每条是 `prompt` 与 `answer`。

路 B 确认：

```bash
ls O-LoRA/CL_Benchmark
```

应看到 dbpedia、amazon、yahoo、ag 一类任务名（以你 clone 到的目录树为准）。每个任务抽训练 500、测试 200，种子 42，写进 jsonl，后续所有配方共用这一抽法。

### Step 3: 指令模板与评测脚本

四个任务必须统一成「指令 + 输入 + 只在答案上算损失」。O-LoRA 附录的分类提示可直接用，例如话题分类：「What is the topic of the following paragraph? Choose one from the option.」情感分类换相应句子。TRACE 路把仓库里已经处理好的 `prompt` 当指令，不要再包一层不同的系统提示，否则和官方数字更没法对。

评测脚本 `eval_matrix.py`（胶水）做一件事：给定当前 adapter，对任务 1..k 的测试集解码，写一行 $R_{k,1},\ldots,R_{k,k}$。分类和数学：规范化后精确匹配。代码：下一行 strip 后精确匹配。摘要：ROUGE-L F1（`evaluate.load("rouge")`）；没装就改 token 级 F1，并在表头标明。解码：`temperature=0`，`max_new_tokens` 分类 8、数学 16、代码 64、摘要 128。停止符用 Instruct 模板的结束符。这组解码超参四条配方共用。

### Step 4: SeqLoRA 填 4×4

模型 `HuggingFaceTB/SmolLM2-135M-Instruct`，LoRA $r=8$，$\alpha=16$，学习率 $1\times 10^{-4}$ 量级（O-LoRA 在 T5 上用过 $1\times 10^{-3}$，解码器 Instruct 通常更小，可在 $5\times 10^{-5}$ 到 $2\times 10^{-4}$ 扫一档，四任务必须同一 LR）。每任务 3 个 epoch 或固定 step（500 条、batch 8，约 60 step/epoch）。种子 42。

伪流程（逻辑，不是可执行 shell）：

```text
初始化一块 LoRA
对任务 t 从 1 到 4
    只在 D_t 上训
    保存 adapter_t
    对 i 从 1 到 t
        测 R[t, i]
写出 4x4 下三角加对角线
```

命令示例（你的胶水脚本名可以是 `train_seq_lora.py`）：

```bash
python3 train_seq_lora.py --order dbpedia,amazon,yahoo,ag --n_train 500 --n_test 200 --seed 42
```

路 A 把 `--order` 换成 `FOMC,MeetingBank,Py150,NumGLUE-cm`。跑完应得到 10 个数字（$R_{1,1}$；$R_{2,1},R_{2,2}$；…；最后一行 4 个）。把矩阵画成热力图，颜色映射固定，三份实验共用。

预期方向：对角线 $R_{t,t}$ 明显高于随机；最后一行里 $R_{4,1}$ 低于 $R_{1,1}$。若 $R_{t,t}$ 本身接近 0，先修模板和 `max_new_tokens`，再谈遗忘。135M 摘要和代码的绝对分会很难看，允许用「规范化子串是否命中」作辅助列，但主矩阵只能有一种主指标。

### Step 5: 混训上限

同一模型、同一 LoRA 配置、同一总 step（四任务顺序 step 之和），数据为四训练集 shuffle。只在全部结束时测四个测试集，得到一行四维向量，当作 $R^{\mathrm{MTL}}_{i}$。

```bash
python3 train_seq_lora.py --mix --order dbpedia,amazon,yahoo,ag --n_train 500 --n_test 200 --seed 42
```

计算 $\mathrm{AA}_{\mathrm{seq}}$、$\mathrm{BWT}$、$\mathrm{AF}$，以及 $\mathrm{AA}_{\mathrm{MTL}}-\mathrm{AA}_{\mathrm{seq}}$。O-LoRA 论文里 T5-large 标准 CL 上这个差距大约是 80.0−43.7=36.3 个百分点（SeqLoRA vs MTL 的平均）。你的 135M 差距会不同，只许写自己的。若差距接近 0，检查混训是否其实没混（concat 忘了 shuffle），或 500 条太容易、模型四科都过拟合到同一批特征。

### Step 6: 可选 IncLoRA 列，以及官方脚本子集

给每个任务单独一份 LoRA，评测任务 $i$ 时加载 $A_i,B_i$ 或把 1..t 的更新都加上。无正交时平均分通常高于 SeqLoRA（O-LoRA 表 2：IncLoRA 平均 66.4 vs SeqLoRA 43.7，T5-large）。这列用来防止你误以为「多一块 adapter 就等于持续学习」。任务 id 在测试时若不可用，IncLoRA 还要额外的路由，写进限制里。

若走 TRACE 官方脚本，改 `scripts/train_lora.sh`：任务列表缩到四个，`CL_method=lora`，样本数 500。然后再：

```bash
bash scripts/train_lora.sh
```

```bash
bash scripts/infer_lora.sh
```

`inference_model_path` 应对准训练的 `output_dir`。日志里若仍出现 20Minuten 或 C-STANCE，说明任务列表没改成功。

读 `train.py` 时盯三个分支：`CL_method=base` 走全参数；`lora` 只插低秩；`O-Lora` 会加正交项，本课禁止。数据折叠在 `utils/data`：README 写明若用 Hugging Face 的 `datasets.load_dataset(ds_name)`，或本地三份 `train.json` / `eval.json` / `test.json`。collator 按 batch 内最长序列左 padding，而不是全局最大长度，这是为了加速 decoder-only。你自己的 PEFT 胶水若用右 padding，和 TRACE 官方数字更不能比，但课内矩阵只要前后一致就行。

顺序敏感性值得单独记一笔。O-LoRA 论文对同一组任务给了三个顺序，平均准确率在 order 1/2/3 上分别是 SeqLoRA 的 44.6、32.7、53.7。顺序能让 naive 差出二十个百分点。课内主线锁死一个顺序，原因很实际：你只有一张卡、两天时间，该顺序并不比另外两个更正确。若加分跑了 order 2，把两张 4×4 并排，不要平均成一个数再和论文比。

### Step 7: 7B 加分

在 24GB 上对 7B 做 QLoRA（4bit）顺序四任务，协议与主线完全相同。模型可用仓库当时仍公开的 Llama-2-7B-Chat 或课程指定的同等 7B Instruct。不要把 TRACE 表 1 的 48.7 / 12.7 填进你的格子。能跑完 4×4 就写实战；OOM 或 Drive 数据不齐则写「哪一步卡住、卡在显存还是数据」，不要补假矩阵。

TinyLlama-1.1B-Chat 作为 1B 加分：同一脚本换模型名，LoRA 模块名按该模型打印。它比 135M 更能写出像样的摘要，矩阵更好读。

## 8. 配置与预算

| 项 | 主线 135M LoRA | 1B 加分 | 7B 加分 | 档 A |
|---|---|---|---|---|
| 模型 | SmolLM2-135M-Instruct | TinyLlama-1.1B-Chat 或 SmolLM2-1.7B-Instruct | 7B Instruct + QLoRA | 不加载 |
| 数据 | O-LoRA order 1，500/200；或 TRACE 四任务 500/200 | 同左 | 同左或 TRACE 5000 档 | 合成矩阵 |
| LoRA | $r=8$，$\alpha=16$，q/v | $r=16$ 可试 | $r=16$，4bit 基座 | 低维玩具 |
| 每任务 | 3 epoch 或等量 step | 同左 | 1-3 epoch | 瞬时 |
| 单卡时间 | 24GB 约 1-4 小时（顺序+混训） | 数小时 | 过夜量级 | <30 秒 |
| 官方脚本 | 可选 TRACE `train_lora.sh` 子集 | 不必须 | TRACE 全量不要求 | 不用 |

缩小：每任务 100 条、1 epoch，只冒烟。冒烟矩阵不得当交付。顺序固定：路 B 为 dbpedia → amazon → yahoo → ag；路 A 为 FOMC → MeetingBank → Py150 → NumGLUE-cm。换顺序算另一张表，不能和第一张平均在一起除非你明确在做顺序敏感性（O-LoRA 的 order 1/2/3）。

## 9. 验收

- 浏览器：先猜「哪类任务最容易被后任务冲掉」，再播放热力图，预测对才过关。
- CPU：`python3 run.py run 10` 现在应当全绿，`checks` 全真：`matrix_is_4x4`、`diagonal_all_above_0_88`、`lower_triangle_forgotten`、`bwt_negative`、`last_row_peaks_on_last_task`。课内矩阵来自二维冲突方向，不是 TRACE 八任务。
- GPU 主线：一张 4×4（含训练过程中的中间行，不只最后一行）、一行混训上限、AA / AF / BWT 三个数、$\mathrm{AA}_{\mathrm{MTL}}-\mathrm{AA}_{\mathrm{seq}}$。
- 书面：子集与 TRACE 全量的差别（任务数、每任务条数）；LoRA 不是抗遗忘方法，引用 TRACE 表 1 的 7B 数字时标明那是论文的 LLaMA-2-7B-Chat，不是你的 135M。
- 禁止：把 O-LoRA 表 2 的 75.8 / 43.7 填进自己的格子；打开正交损失却当成本课 naive；只报最终平均准确率；用 Instruct 基座做第 09 课那种无模板 CPT 却提交到本课。

方向性量化线（有 GPU）：至少有一个先学任务满足 $R_{T,i}<R_{i,i}$（遗忘发生）；混训 AA 不低于顺序 AA。若四任务都忘光到随机，先查解码。若完全无遗忘且未混训，先查是不是每次都重新初始化了 LoRA、或者评测时加载错了最后一个 adapter 去测所有任务。

## 10. 排错

| 症状 | 原因 | 验证 | 修法 |
|---|---|---|---|
| 可训练参数 100% | PEFT 没挂上 | `print_trainable_parameters` | `get_peft_model` 后再 Trainer |
| 四个任务分数完全相同 | 评测加载了同一 adapter，或测试集切错 | 打印每个测试文件的路径和样本前缀 | 每任务结束保存独立目录，评测显式传入 |
| 分类输出一长段摘要 | 后任务模板污染，或 `max_new_tokens` 过大 | 看 raw generation | 分类限 8 token；单独统计格式错误率 |
| ROUGE 全 0 | 没装 `rouge` 或解码空 | 打印生成文本 | 改 token F1，标明指标 |
| TRACE 脚本找不到数据 | Drive 未下或 `data_path` 错 | `ls` 八个任务目录 | 改路 B；不要空跑 |
| `CL_method` 写成 `O-Lora` | 提前打开第 11 课方法 | 读 shell 变量 | 本课必须 `lora` 或 PEFT naive |
| T5-large OOM | 当全参数训了 | 是否 `lora` | 按 README 放 `initial_model/t5-large`，用官方 LoRA 脚本 |
| 135M 数学全错 | 模型不会算术，不是遗忘 | 看 $R_{4,4}$ | 把数学改成「从选项里选数字」或只用分类四任务，并在报告声明 |
| 混训反而更差 | shuffle 失败或 LR 过大过拟合 | 检查 dataloader 是否跨任务 | 降 LR，确认四任务标签都出现在同一 epoch |
| `diagonal_all_above_0_88` 为假 | 当前任务没学会 | `metrics.diagonal_min` | 看 5.3：刚学完的 $R_{t,t}$ 应对角线高；学习率过小会让整表接近随机 |
| `lower_triangle_forgotten` 为假 | 下三角没有遗忘 | `lower_triangle_mean` 相对 `diagonal_min` | 四个方向应是 0°/90°/180°/270°；对照第 03 课的矩阵读法 |
| `bwt_negative` 为假 | 后任务没有伤到前任务 | `metrics.bwt` 应低于 −0.10 | 用第 03 课公式：对 $i<T$ 取 $R_{T,i}-R_{i,i}$ 再平均 |
| `last_row_peaks_on_last_task` 为假 | 最后一行最深的不是刚学完的任务 | `accuracy_matrix` 最后一行 argmax | 顺序训练结束后当前任务应对应该行最深的格子 |

## 11. 前沿与改造

前沿：CIT 的抗遗忘现在主要走三条。回放，Continual-T0 用约 1% 旧指令就能保住很多生成任务。正则与子空间，O-LoRA 把新 LoRA 赶到与旧 $A$ 正交的方向，InfLoRA、DAPT 是同一家族。扩结构，Progressive Prompts、LLaMA Pro 扩块。TRACE 自己的 RCL 给每条样本补推理链，用 500 条达到接近 5,000 条 SeqFT 的目标任务分，并减轻 GSM/BBH 的塌陷。2025 年又多一刀：SEFE（[arXiv:2505.02486](https://arxiv.org/abs/2505.02486)）把分数掉了拆成答卷格式偏了和知识真丢了；Zheng et al.（[arXiv:2501.13453](https://arxiv.org/abs/2501.13453)）把早期优化步对任务对齐的破坏叫虚假遗忘，用冻底层缓解。这些都不是本课 naive 的一部分。

我们差在哪：4 任务不是 8；500 条不是 5,000；135M 不是 Llama-2-13B；没有 GPT-4 打的 IFD/SD；没有 RCL 的 GPT-4 标注推理。差的是规模，不是「顺序会忘」这件事本身。

改造清单：

1. 把每任务训练条数从 100 扫到 500 再到 2000（若有数据）。预期：条数太少时对角线也低，谈不上遗忘；条数够了下三角才分开。失败标准：条数增加后面条 $R_{t,t}$ 仍近 0，说明模板或模型容量不够，应换任务而不是加步数到过拟合单条。
2. 加 10% 旧任务回放（TRACE 的 `past_task_ratio` 或自己在 batch 里混）。预期：BWT 改善，接近 Scialom 的「一点点回放就很强」。失败标准：回放后新任务 $R_{t,t}$ 明显下降且旧任务也不升，说明混合比或采样坏了。
3. 交换任务顺序，跑 order 2。预期：被夹在中间的相似任务（两个话题分类）互相冲得更狠。失败标准：三张矩阵在噪声内无差别，需要加 seed 再下结论。
4. 对任务 1 做格式/内容双记分。预期：SeqLoRA 的格式错误率先于内容准确率崩。失败标准：只改了 `max_new_tokens` 就把「遗忘」修没了，那是评测协议在演戏，把解码配置写回固定值。

顺手复现映射：正式复现 #3 在第 11 课（O-LoRA vs naive LoRA 方向性）。本课只提供 naive 的 4×4，作为那次复现的对照底稿。TRACE 论文数字只许对照，不许当本课分数。

## 12. 论文与延伸

1. Wang, Zhang, Chen, Gao, Jin, Yang, Xi, Zheng, Zou, Gui, Zhang, Huang, 2023, *TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models*, [arXiv:2310.06762](https://arxiv.org/abs/2310.06762)。
贡献：给已经对齐的 LLM 做更难的持续学习基准。机制发明处，不是本课主阅读。
机制：8 个任务，覆盖领域、多语言、代码、数学，统一成可自动打分的格式。摘要写 llama2-chat 13B 在 gsm8k 上从 28.8% 掉到顺序训练后的 2%。他们还加了推理增强的 RCL：给样本补任务线索和元推理，减轻遗忘并加快新任务收敛。课内主线不跑 RCL。
和本课：4 任务子集、浏览器热力图、CPU 的 `matrix_is_4x4` / `lower_triangle_forgotten` / `bwt_negative` 看见「顺序会忘」。gsm8k 从 28.8% 掉到 2%、GAD/IFD/SD、8 任务全量，本课实验答不了。
阅读问题：你的 4×4 只覆盖目标任务。CPU 的 `last_row_peaks_on_last_task` 为真，只说明最后一科最强。若 135M 在课内填空上也掉了，你怎么把它和「输出格式崩」分开？本课 CPU 没有通用填空列，写「答不了，差在评测」即可。

2. Wang, Liu, Shi, Li, Chen, Lu, Yang, 2024, *InsCL: A Data-efficient Continual Learning Paradigm for Fine-tuning Large Language Models with Instructions*, [arXiv:2403.11435](https://arxiv.org/abs/2403.11435)。
贡献：指令微调的回放按任务相似度动态分配，再按指令信息量偏爱高质量样本。
机制：用指令嵌入的 Wasserstein 距离估任务差，差得远的旧任务多回放。InsInfo 用指令标签的数量和稀有度打分，高分指令多采样。16 个 SuperNI 任务、多种顺序。摘要：训完全部任务后，相对随机回放 Relative Gain 高 3.0，相对无回放高 27.96。
和本课：naive SeqLoRA 无回放，对应他们的 No Replay。CPU 的 `bwt_negative` 能看见无回放会忘。Wasserstein 调度和 InsInfo，本课实验答不了。
阅读问题：改造清单里的 10% 旧任务回放是均匀混还是按任务相似度混？若你只做了均匀混，InsCL 的 3.0 Relative Gain 你验证不了，应写「本课实验答不了」。

3. He, Zhang, Huang, Zhang, Meng, Zhou, Zeng, Cai, 2024, *Don't Half-listen: Capturing Key-part Information in Continual Instruction Tuning*, [arXiv:2403.10056](https://arxiv.org/abs/2403.10056)。
贡献：顺序指令微调时，模型容易只记住指令的表面套话；他们用关键片段的信息增益来选回放、改训练目标。
机制：KPIG 对指令里被遮住的关键部分算信息增益，动态回放，并收紧损失让模型去抓和正确答案相关的任务信息。另给 P-score、V-score 分别测泛化和听指令。评测含已见任务和 held-out 任务。
和本课：CPU 四个方向互相冲突，下三角掉分。它分不清「套话还在、关键槽位没听」。格式崩要靠 GPU 解码样例，CPU check 答不了 P-score / V-score。
阅读问题：路 B 四个分类任务的指令长得很像（都是选话题/情感）。你是否在旧任务上看到「还在输出标签词、但选错了类」？若有，更像半听；本课没有 KPIG 实现，不能声称复现了这篇。

4. Chen, Zhu, Luo, Shen, Gao, Song, 2024, *CoIN: A Benchmark of Continual Instruction tuNing for Multimodel Large Language Model*, [arXiv:2403.08350](https://arxiv.org/abs/2403.08350)。
贡献：给多模态 LLM 做 10 数据集、8 类任务的顺序指令基准，并把评测拆成指令跟随和通用知识。
机制：顺序指令微调后，现有 MLLM 仍大面积遗忘。他们把主要责任记在意图对齐失败，知识侧次之。方法上把 MoELoRA 接到 MLLM，用来保住先前的指令对齐。标题里的 Multimodel 是摘要原文用字。
和本课：本课 4×4 是文本任务增量。`lower_triangle_forgotten` 能看见忘，分不开「不会听」和「知识没了」。视觉输入和 MoELoRA 本课实验答不了。
阅读问题：改造清单第 4 项要对任务 1 做格式/内容双记分。若格式错、内容对，更靠近 CoIN 的哪一列评测？本课 CPU 没有双记分，GPU 没做就写「答不了」。

5. Zheng, Ma, Liu, Wu, Feng, 2024, *Beyond Anti-Forgetting: Multimodal Continual Instruction Tuning with Positive Forward Transfer*, [arXiv:2401.09181](https://arxiv.org/abs/2401.09181)。
贡献：多模态持续指令微调里，旧知识会忘，未来任务也会被提前带坏（负向前向迁移）。
机制：对输入嵌入做 SVD，发现不同任务的嵌入差很大，模型会学到对旧任务和预训练都无关的方向。Fwd-Prompt 把 prompt 梯度投影到残差空间减干扰，再投影到预训练子空间复用旧能力。无旧样本，更新的参数更少。
和本课：CPU 每训完一个任务就测全部 4 列，上三角是还没训的任务，可以手算 FWT 方向。Fwd-Prompt 的 SVD 投影和预训练子空间，本课实验答不了。
阅读问题：打开 `accuracy_matrix` 的上三角。第 0 行第 3 列和第 2 行第 3 列相比，未来任务是被带坏了还是几乎没变？课内没有单独报 FWT，格子够你手算。SVD 投影仍然答不了。

6. Wu, Hartman, Jayaraman, Varshney, 2024, *SwitchCIT: Switching for Continual Instruction Tuning*, [arXiv:2407.11780](https://arxiv.org/abs/2407.11780)。
贡献：顺序指令学习时，用开关把计算路由到已经 PEFT 调好的专家，避免新任务覆盖同一块权重。
机制：每个任务一份参数高效适配器，推理时按任务切换。实验覆盖自然语言生成和视觉-语言。摘要强调效率、可扩展、可搬运、以及适配器可分开存放带来的隐私。本课 naive 是同一块 LoRA 接着训。
和本课：IncLoRA（每任务新块、旧块冻住）是它的弱亲戚，本课加分列才碰。CPU 只有一块线性分类器，没有路由。SwitchCIT 的开关本课实验答不了。
阅读问题：若推理时不告诉模型现在是第几科，SwitchCIT 还能否选对专家？本课 CPU 没有任务编号输入，写「答不了」。

7. Cao et al., 2024, *Continual LLaVA: Continual Instruction Tuning in Large Vision-Language Models*, [arXiv:2411.02564](https://arxiv.org/abs/2411.02564)。
贡献：冻住 LVLM，用双增量嵌入做无回放的持续指令微调，并给出 COAST 基准（领域增量、能力增量、数据集增量）。
机制：每条指令构造两类增量嵌入。固有增量：低秩池里按与用户指令的相似度挑候选。情境增量：把先前任务挑中的低秩嵌入用可学习加权和起来，给跨任务提示。基座冻住，不回放。
和本课：SeqLoRA 动的是同一块 $A,B$。这篇冻骨干、只写嵌入。CPU 二维分类器没有嵌入池。COAST 三配置本课实验答不了。
阅读问题：本课若改成 IncLoRA（旧块冻住、每任务新块），遗忘应当接近 Continual LLaVA 的哪一句主张？你必须实际跑加分列才能答；没跑就写「本课没做」。

8. Jiang, Jiang, Li, Xue, Zhou, Song, Lian, Wei, 2025, *Unlocking the Power of Function Vectors for Characterizing and Mitigating Catastrophic Forgetting in Continual Instruction Tuning*, [arXiv:2502.11019](https://arxiv.org/abs/2502.11019)。
贡献：用函数向量（FV）当遗忘是否发生的模型相关探针，并加正则稳住 FV。
机制：他们在多种任务顺序上看到遗忘既依赖任务也依赖模型。分析结论：LLM 的遗忘主要来自函数激活出现偏差，任务处理函数本身未必被覆盖。训练时加一项让 FV 稳定。四个基准上验证。
和本课：CPU 只有一层线性权重，没有可抽取的 FV。`bwt_negative` 能看见忘，不能看见「激活偏了还是函数被覆盖」。FV 正则本课实验答不了。
阅读问题：本课四个方向是 0°/90°/180°/270°，权重被后任务直接推走。这更接近「函数被覆盖」还是「激活偏差」？用线性模型的几何回答；不要把论文的 LLM 结论抄到 CPU 上。

9. Chen, Cong, Zhao, Yang, Hu, Ip, Kwong, 2025, *SEFE: Superficial and Essential Forgetting Eliminator for Multimodal Continual Instruction Tuning*, [arXiv:2505.02486](https://arxiv.org/abs/2505.02486)。
贡献：把多模态持续指令里的遗忘拆成表层（答卷格式被后任务带偏，知识未必没了）和本质（格式对、事实错）。
机制：先做 Answer Style Diversification（ASD），把各任务训练集改成彼此接近的多样化答卷风格，避免风格漂移把分数打穿。再在此之上用 RegLoRA 正则化存放旧知识的关键参数。两者合称 SEFE。
和本课：学习目标第 5 条和改造第 4 项就是表层遗忘。CPU 准确率把两种遗忘糊成一个数；GPU 若只报精确匹配，也会把「答成了摘要口吻」判成全错。RegLoRA 本课实验答不了。
阅读问题：看你的任务 1 在第 4 行的样例。若标签词还在、句子套了后任务模板，这是表层还是本质？CPU check 不能单独回答，必须看解码文本。

10. Guo, Zeng, Xiang, Zhu, Wang, Zhang, Liu, 2025, *HiDe-LLaVA: Hierarchical Decoupling for Continual Instruction Tuning of Multimodal Large Language Model*, [arXiv:2503.12941](https://arxiv.org/abs/2503.12941)。
贡献：按层间 CKA 相似度做任务专用扩张和任务通用融合，并指出旧基准有信息泄漏，另给更难的新基准。
机制：不同数据集训完后，各层表征的 CKA 变化不一样。他们据此把「该扩一块专用」和「该融回通用」分开，避免只用堆参数换分数。代码与数据在 Ghy0501/HiDe-LLaVA。
和本课：CPU 没有多层，也没有 CKA。4×4 能看见忘，看不见层间解耦。新基准的泄漏问题，本课 TRACE/O-LoRA 子集答不了。
阅读问题：O-LoRA order 1 四个都是分类，会不会也有「训练集指令泄漏到测试」？你抽 500/200 时有没有按论文那样检查重叠？没检查就写「本课没验泄漏」。

11. Zheng, Cai, Qiu, Ma, 2025, *Spurious Forgetting in Continual Learning of Language Models*, [arXiv:2501.13453](https://arxiv.org/abs/2501.13453)。
贡献：许多「分数塌了」其实是任务对齐丢了，知识未必被擦掉。
机制：合成数据上看到，新任务刚开始的若干优化步就会把旧任务对齐冲歪。理论把这种偏移连到权重的正交更新。方法是冻住底层，在四个持续学习设定里分数明显回升。
和本课：CPU 整份权重都在动，没有分层冻结。`diagonal_all_above_0_88` 只说明刚学完时对齐还在；下三角掉了，分不开「对齐丢了」和「方向被覆盖」。冻底层本课实验答不了。
阅读问题：若你在 GPU SeqLoRA 上冻 embedding 和前几层，只训 LoRA，下三角会不会浅一些？没做这个消融就写「本课实验答不了」。

12. Wang, Chen, Ge, Xia, Bao, Zheng, Zhang, Gui, Huang, 2023, *Orthogonal Subspace Learning for Language Model Continual Learning*, [arXiv:2310.14152](https://arxiv.org/abs/2310.14152)。
贡献：O-LoRA，让各任务低秩子空间互相正交，减轻干扰。机制发明处，不是本课主阅读。
机制：新任务的 LoRA 学在与旧 $A$ 正交的方向，$A_i^\top A_t=0$，不存旧数据。摘要写在持续学习基准上超过当时方法，并对未见表任务的泛化更好。正交损失本课必须关掉。
和本课：路 B 的 order 1 四任务就是这篇仓库的数据。naive SeqLoRA 的 4×4 是第 11 课复现的对照底线。CPU 没有 LoRA 的 $A$，算不了夹角。
阅读问题：CURRICULUM 曾出现 arXiv:2305.18870。你打开的摘要编号是 2310.14152 吗？若用错号，引用会指到无关论文。本课实验不验证编号，靠你打开 abs。

做完 4×4，你已经能用第 03 课的语言描述大模型指令接龙的遗忘。下一课要把 LoRA 的 $A$ 矩阵拿出来算内积：naive 会抢同一方向，O-LoRA 要它们接近正交。[第 12 课](12_model_merging.md) 则问：若你已经分别训好了两块 adapter，能不能干脆加起来，不再接着训。那是事后缝合，不是本课这种在线接龙；但本课保存下来的 `adapter_t` 正好是合并实验的原料。不要把四块 LoRA 删掉。

