---
id: 20_seal_rl_razor
title: "自己出题，以及为什么 RL 比较不易忘"
summary: "on-policy RL 比 SFT 更不易遗忘，是因为分布离原模型更近吗？"
unit: nested
play_tools: []
checkpoints:
  - "rl-razor-mnist 复现曲线，论文复现 #5。"
  - "SEAL 能跑通的部分写实战记录。"
---

# 第 20 课：自己出题，以及为什么 RL 比较不易忘

> 类型：复现（RL's Razor MNIST，课程论文复现 #5）+ 实战（SEAL 最小自编辑）+ 加分（SEAL 全量 RL）<br>
> 建议周期：3-4 天（MNIST 主线 1 天，SEAL 自编辑 1 天，全量 RL 另排）<br>
> 硬件：ParityMNIST 用 CPU 或单卡；SEAL 数据生成可用 OpenAI API，全量内环需要 2 张 A100/H100 级显卡，官方 README 写明<br>
> 锚定：[SakanaAI/rl-razor-mnist](https://github.com/SakanaAI/rl-razor-mnist)；[Continual-Intelligence/SEAL](https://github.com/Continual-Intelligence/SEAL)（[arXiv:2506.10943](https://arxiv.org/abs/2506.10943)）<br>
> 产物：rl-razor-mnist 的 SFT vs GRPO 曲线（遗忘 vs 新任务，遗忘 vs KL）；SEAL 自编辑实战记录（跑通哪一步、卡在资源还是代码）

## 1. 这一课做什么

第五幕最后一课。[第 19 课](19_nested_learning.md)把「写到哪里」做成了频率轴：快权重记本句，慢权重记本篇，Hope 的完整语言模型标了不能练。本课换零件：不再加转速，而问模型能不能给自己出训练题，以及同样学会新任务时，为什么强化学习往往比监督微调更少把旧能力冲掉。

上一课留下的系统会在测试时改一块快矩阵，但出题的人还是你。SEAL（Self-Adapting LLMs，自适配语言模型）把出题权交给模型：看见一篇新文章或几条示范，它先生成「自编辑」（self-edit），内容可以是从文章推出的蕴涵句、改写、问答对，也可以是数据增强和优化超参数；然后用这些文本对自己做一次监督微调。外环再用强化学习挑那些「改完权重之后下游分数真的涨了」的自编辑。另一条线来自 Shenfeld、Pari 与 Agrawal 的 *RL's Razor*（[arXiv:2509.04259](https://arxiv.org/abs/2509.04259)）：在新任务上，on-policy 的 RL 会偏向于离原模型 KL 最近的那批解，所以旧任务掉得少。Sakana 把这条规律缩到 ParityMNIST，仓库是本课主线复现。

两件事必须分开标档。rl-razor-mnist 是复现 #5：同一新任务，SFT 对上 GRPO，旧任务保持必须对 SFT 更差，并且遗忘与到原模型的正向 KL 相关。SEAL 全量 RL 贵，主线只跑官方最小自编辑（生成蕴涵句 / 读仓库里已有的 synthetic 数据）；TTT 内环和 ReST-EM 外环标加分。加分跑不通时，验收要写清卡住的是资源（双卡、7B、OpenAI 评分）还是代码。

没有这一课，第五幕会停在「架构里有快慢权重」。你会以为持续学习只是把记忆写进模块，却说不清：部署之后由谁来生成下一批训练数据，以及为什么用离线标准答案硬拉策略，旧能力先死。有了这一课，主线里「怎么写」多出一条：能走 on-policy 就别用一张离原模型很远的答卷去 SFT。

梁文峰转写里「还差一步：学习如何学习」在公开技术上最接近的两块，就是第 17–19 课的测试时更新，以及本课的自编辑加 on-policy。这段话出自 2026-07 对 2026-05 交流的整理，DeepSeek 没有正式确认。本课不引用转写里的卡数、定价或内部方法。

术语速查：

| 术语 | 一句解释 |
|---|---|
| 自编辑（self-edit） | 模型自己生成的微调数据或更新指令，用来改自己的权重 |
| SEAL | 用 RL 训练「如何写自编辑」的框架；内环仍是 SFT |
| ReST-EM | 拒收采样再 SFT：只对奖励为正的样本做监督。SEAL 外环用它，因为 GRPO/PPO 在他们设定里不稳 |
| on-policy | 训练用的输出是当前策略自己采样的，不是外部标准答案 |
| SFT | 监督微调：对固定标签最大化对数似然 |
| GRPO | 组相对策略优化：同一输入采一组动作，用组内均值当基线算优势 |
| 正向 KL | 原模型到微调模型的正向 KL，在新任务输入上算。RL's Razor 用它预测遗忘 |
| RL's Razor | 能解新任务的策略很多时，on-policy RL 偏向离原模型最近的那些 |
| ParityMNIST | 把数字奇偶当任务：偶数输出 0/2/4/6/8 都算对，所以正确策略不唯一 |
| 论文复现 #5 | 本课正式复现：RL 比 SFT 更不易忘，遗忘与到原模型的 KL 相关 |

## 2. 问题

部署之后还要学，你会面对两张试卷。第一张：新知识以文章、工具说明、几条示范的形式到来，并没有现成的大规模微调语料。直接把原文拿去 SFT，SEAL 论文在 SQuAD 的无上下文问答上几乎不涨（通道-only 约 33.5%，基座 32.7%）。人会把讲义改写成对自己有用的笔记；模型会不会写对自己有用的微调文本？第二张：就算你有标签，SFT 也可能用一种离原模型很远的方式「做对」新题，把旧题的概率质量抽走。两张试卷对应本课两条线，不要合成一句空话。

SEAL 的设计选择要逐项钉死。自编辑是自然语言，不是一个外挂超网络吐出来的 LoRA。内环是 SFT 加 LoRA，不是再套一层 TTT 矩阵。奖励来自更新后的模型在 $\tau$ 上的表现，所以奖励依赖于当前参数 $\theta$，旧回合的数据会过期，他们因此坚持 on-policy，并在 GRPO/PPO 不稳之后改用 ReST-EM。知识写入设定里，自编辑默认是「列出若干蕴涵」；少样本设定里，自编辑是增强工具和超参数的说明书。论文也写了限制：连续多次自编辑仍会遗忘；每次算奖励要微调再评估，单次约 30–45 秒。

RL's Razor 要回答的不是「RL 分数更高」，而是「同样做对新任务时，谁更少忘旧的」。他们在 Qwen 2.5 3B-Instruct 的数学、科学问答、工具使用，以及 OpenVLA 的抓取上画了帕累托前沿：RL 涨新任务时旧基准几乎不动，SFT 涨新任务时旧基准掉。为了解释，他们提出经验遗忘律：在新任务分布 $\tau$ 上算 $\mathbb{E}_{x\sim\tau}[\mathrm{KL}(\pi_0\|\pi)]$，就能预测遗忘，不必访问旧任务数据。ParityMNIST 把这件事缩到三层 MLP：预训练同时做奇偶和 FashionMNIST，微调只做奇偶，Fashion 的掉分当遗忘。偶数有五个合法标签，SFT 可以强行规定「偶数一律输出 0」，离原模型很远；RL 只要求奇偶对，会留在原模型已经觉得像的那些偶数上。

Chen、Razin、Narasimhan 与 Danqi Chen 的 *Retaining by Doing*（[arXiv:2510.18874](https://arxiv.org/abs/2510.18874)，ICML 2026）给出互补机制：RL 对应反向 KL（mode-seeking），SFT 对应正向 KL（mode-covering）；在多峰策略上，mode-seeking 反而更能保住旧峰。他们用消融说明：真正起作用的是 on-policy 数据，不是 KL 正则，也不是 GRPO 的优势估计。大约 on-policy（每个 epoch 重新采样再 SFT）已经能减轻遗忘。和 Razor 不完全相同：Chen 文附录写他们设定里 KL 与遗忘的相关没有 Razor 那么稳。本课复现以 Razor 的 MNIST 仓库为准，Chen 文用来解释「为什么 on-policy」。

SEAL 自己的图 6 把两条线接起来：内环若继续用 SFT 做一连串自编辑，先前段落的问答会掉。作者在限制节写：既然 RL 比 SFT 更不易忘，未来内环也可以改成 RL。本课不实现这个改法，但验收时要能指出这个缺口。

## 3. 准备

- 第 01 课的遗忘定义、第 03 课的「新任务分数和旧任务保持要一起报」、第 10 课顺序指令微调的直觉。本课 MNIST 实验是同一套账，标签空间有意做成多解。
- 第 17 课的测试时训练：SEAL 的内环是一次小规模 TTT（对自编辑做 LoRA SFT），外环才是学「如何写自编辑」。
- Python 3.10+，能装 PyTorch 和 torchvision。MNIST 与 FashionMNIST 会在首次运行时下载。不强制 Weights & Biases：仓库脚本带 `--wandb`，本课命令默认去掉。
- 读三篇摘要再动手：SEAL [arXiv:2506.10943](https://arxiv.org/abs/2506.10943) v2；RL's Razor [arXiv:2509.04259](https://arxiv.org/abs/2509.04259)；Retaining by Doing [arXiv:2510.18874](https://arxiv.org/abs/2510.18874) v3。Razor 项目页：`http://jyopari.github.io/posts/rl_razor`。SEAL 项目页：`https://jyopari.github.io/posts/seal`。
- SEAL 官方 README 要求 Python 3.12 环境、`pip install -r requirements.txt`、根目录 `.env` 里的 `OPENAI_API_KEY`（评分用 GPT-4）。全量实验写「2 张 A100/H100」。没有这些条件就走本课的「最小自编辑」：读提示模板、看仓库自带的 `squad_*.json` 和 `synthetic_data/`，能调用 API 再生成几条蕴涵。不要假装跑过 ReST-EM。
- 延伸阅读准备：Thinking Machines Lab, Kevin Lu, 2025-10, *On-Policy Distillation*（`https://thinkingmachines.ai/blog/on-policy-distillation/`）。它把 on-policy 采样和稠密 token 监督拼在一起，用来对照本课「SFT 离线 / RL on-policy」的二分。
- 磁盘与账号：MNIST 首次下载约几十 MB。SEAL clone 含 `squad_train.json`（约 28MB）和 `squad_val.json`。全量还要 Hugging Face 上的 Qwen2.5-7B 或 Llama-3.2-1B-Instruct。OpenAI 密钥只在评分和 `make_squad_data_openai.py` 时需要；档 A 不需要。

## 4. 学习目标

1. 画出 SEAL 的内外两层：内环 $\theta'\leftarrow\mathrm{SFT}(\theta,\mathrm{SE})$，外环用更新后模型在 $\tau$ 上的表现当奖励；说出他们为什么改用 ReST-EM。
2. 在 ParityMNIST 上解释「多个正确答案」如何让 SFT 和 RL 走到不同的输出分布。
3. 写出正向 KL 遗忘律，并说明度量是在新任务输入上算的，不需要旧任务数据。
4. 跑通 rl-razor-mnist 的预训练和至少一组 SFT-1 与一组 GRPO，填新任务准确率、FashionMNIST 保持、正向 KL 三列。方向必须满足复现 #5。
5. 按 SEAL 官方路径做最小自编辑（生成或阅读蕴涵句）；全量 RL 若未跑，书面写明卡在资源还是代码。
6. 指出 SEAL 内环仍是 SFT，因此连续自编辑会忘；这和 Razor / Chen 的结论是同一缺口。

## 5. 原理

六个机制。SEAL 和 Razor 共用「on-policy」这个词，但一层用在外环选自编辑，一层用在策略梯度的采样分布。读的时候把层标清楚。

### 5.1 自编辑：模型先改写数据，再改自己的权重

直觉。学生复习时很少把教材逐字背完，而是改写成例题、推论、问答。SEAL 把这一步变成可训练的生成：给定上下文 $C$，模型采样一段自编辑 SE，再用 SE 做 SFT 得到 $\theta'$。类比失效处：人的笔记不会立刻改大脑突触的学习率；SEAL 的少样本设定里，自编辑却可以点名学习率和训练哪些 token。知识写入设定更单纯，自编辑就是一段文本。

机制。任务实例是 $(C,\tau)$。知识写入：$C$ 是段落，$\tau$ 是关于它的问答，评测时段落不在上下文里。少样本：$C$ 是 ARC 式示范，$\tau$ 是留出的测试格。生成 SE 之后：

$$
\theta' \leftarrow \mathrm{SFT}(\theta, \mathrm{SE})
$$

仓库知识线用 LoRA，因为单段数据量小、更新次数多。奖励 $r(\mathrm{SE},\tau,\theta)$ 看的是 $\mathrm{LM}_{\theta'}$ 在 $\tau$ 上好不好，所以同一个 SE 在 $\theta$ 改过之后分数会变。这就是他们不用离线 RL 缓存的原因。

数学。外环目标（论文 (1)）是

$$
\mathcal{L}_{\mathrm{RL}}(\theta_t) = -\mathbb{E}_{(C,\tau)\sim\mathcal{D}}\Big[\mathbb{E}_{\mathrm{SE}\sim\mathrm{LM}_{\theta_t}(\cdot\mid C)}\big[r(\mathrm{SE},\tau,\theta_t)\big]\Big]
$$

知识线默认提示是「列出由内容直接或间接得到的若干蕴涵」，实现见 `general-knowledge/src/data_generation/make_squad_data.py` 的 `MAKE_SQUAD_DATA_TEMPLATES_BASE["implications"]`。同文件还提供 `rewrite`、`self-qa`、`implications-long` 等键；论文附录说换提示后 RL 仍然涨，本课最小实验用默认 `implications`。

代码。生成：`python3 -m general-knowledge.src.data_generation.make_squad_data`（本地 vLLM）或 `make_squad_data_openai.py`（OpenAI）。应用：`general-knowledge/src/inner/TTT_server.py` 收一套 `train_sequences` 和 `eval_questions`，临时 LoRA，返回 `baseline_accuracy` 与 `adapter_accuracy`。训练序列模板在 `general-knowledge/src/utils.py`：`TRAINING_SEQUENCE_TEMPLATE = "{title}\n{completion_text}"`。

验证。同一段落，把自编辑换成原文再 SFT，论文表 2 单段 LoRA：只训原文 33.5%，原文加基座蕴涵 39.7%，加 GPT-4.1 蕴涵 46.3%，SEAL 两轮后 47.0%。续预训练 200 篇全参：SEAL 58.2%，GPT-4.1 合成 59.4%，只训原文 36.0%。你若只生成了蕴涵、没跑 TTT，不能填这张表，只能检查 JSON 里 `completions` 是否像推论而不是复读。少样本表 1 用的是筛选过的 11+8 题，基座 1B 没有 ARC 预训练；72.5% 相对的是「无 RL 自编辑 20%」和「人工最优 TTT 100%」，不是公开排行榜。

### 5.2 外环为何是 ReST-EM，不是 GRPO

直觉。策略梯度要采样、打分、更新生成 SE 的那个策略。SEAL 试过 GRPO 和 PPO，写明训练不稳，于是改成 ReST-EM：从当前模型采样若干 SE，只把奖励为正的留下，对它们做 SFT。类比：只把「改完真会做题」的笔记抄进错题本。失效处：二值奖励丢掉了「涨了多少」；好样本少时外环几乎没信号。

机制。二值奖励（论文 (2)）：在 $\tau$ 上，用该 SE 适应之后若比适应前好，则 $r=1$，否则 0。知识线还可以只给一组候选里提升最大的那条记功。对 $r$ 关于 $\theta_t$ 不可微，他们把 $r$ 当常数（stop-gradient），于是蒙特卡洛梯度退化成「对好 SE 的 token 做普通对数似然」。这就是「滤完再 SFT」。因为 $r$ 依赖 $\theta$，必须用当前模型重新采样、重新打分。

数学。小批量 $N$ 个上下文、每个 $M$ 条 SE 时，论文 (3)(4) 给出

$$
\nabla_{\theta_t}\mathcal{L}_{\mathrm{RL}} \approx -\frac{1}{NM}\sum_{i=1}^{N}\sum_{j=1}^{M} r_{ij}\,\nabla_{\theta_t}\log p_{\theta_t}(\mathrm{SE}_{ij}\mid C_i)
$$

$r=0$ 的项消失，剩余即 SFT。仓库里对应 `general-knowledge/src/EM/build_SFT_dataset.py`：从 `query_server` 的 JSON 按 `adapter_mean` 取 top-k，写出 JSONL；再 `general-knowledge/src/EM/train_SFT.py` 做 LoRA。少样本线是 `few-shot/BC-self-edit.py`，读 `final_results.json` 里做对了的自编辑。

代码。知识线一轮外环的顺序写在 `general-knowledge/README.md`：造数据 → 开 TTT_server → `query_server` → `build_SFT_dataset.py` → `train_SFT.sh`。`train_SFT.sh` 默认 `accelerate launch --num_processes 2` 加 DeepSpeed ZeRO-3，模型 `Qwen/Qwen2.5-7B`。这是加分项，不是主线。

验证。外环若用过期的 $(\mathrm{SE},r)$ 去更新新 $\theta$，奖励会对错人。课内用 MNIST 线体会 on-policy；SEAL 外环的 on-policy 你若没跑双卡，就在笔记里引用论文 3.1 节这段限制，不要编造你见过不稳的 loss。

### 5.3 RL's Razor：遗忘由新任务上的 KL 预测

直觉。能解新任务的策略是一片区域，不是一个点。SFT 把你拉向标注分布的中心，那个中心可以离原模型任意远。On-policy RL 每次只在自己已经会说的句子附近重新加权。类比：学会用左手写字，可以稍微改握笔，也可以把全身姿势改成另一个人；两种都能写，第二种把你原来的右手习惯毁得更干净。失效处：若新任务的正确集和旧任务完全互斥，KL 最小的解也可能很远，Razor 救不了。

机制。SFT 的损失在离散输出上是

$$
\mathcal{L}_{\mathrm{SFT}}(\pi)=-\mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\beta}[\log\pi(y\mid x)]
$$

$y$ 来自外部 $\pi_\beta$。策略梯度是

$$
\mathcal{L}_{\mathrm{RL}}(\pi)=-\mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi}[A(x,y)\log\pi(y\mid x)]
$$

$y$ 来自当前 $\pi$。Razor 第 5 节把「on-policy」和「负样本」拆开：1-0 Reinforce（只对正确样本 SFT，仍从 $\pi$ 采样）表现像 GRPO；离线的 SimPO（外部正负样本）表现像 SFT。结论：关键是采样分布，不是负梯度。主实验的 RL 还故意关掉显式 KL 正则，只用 0/1 对错当奖励，避免「RL 不易忘只是因为加了 KL 项」。

数学。经验遗忘律：微调 $\pi$ 于新任务 $\tau$ 后，遗忘由

$$
\mathbb{E}_{x\sim\tau}\big[\mathrm{KL}(\pi_0\|\pi)\big]
$$

预测。ParityMNIST 上二次拟合 $R^2=0.96$；他们的 LLM 实验 $R^2=0.71$。仓库 `src/rl_razor/metrics.py` 的 `forward_kl()` 实现的是 $\sum \pi_0(\log\pi_0-\log\pi)$。论文引理 5.1：对 0/1 奖励，从 $p$ 做拒绝采样得到的 $q_{\mathrm{RS}}$ 正是满足期望奖励为 1 且 $\mathrm{KL}(q\|p)$ 最小的分布。定理 5.2：在凸策略族上，策略梯度收敛到可表示最优集里离 $\pi_0$ 最近的那个。仓库 `src/rl_razor/training/oracle.py` 注释写明：论文附录把 oracle 写成 $\mathrm{KL}(\pi_0\|q)$，但 $q$ 被限制在正确奇偶集合 $S$ 上时这个方向可以是无穷；实现用的是 I 投影 $q^{\star}(y)\propto\pi_0(y)$（$y\in S$）。复现时以代码和这条注释为准。

代码。模型是 785→512→256→10 的 MLP（784 像素加任务指示 ±1），`src/rl_razor/model.py`。数据包装在 `src/rl_razor/data.py` 的 `TaskIndicatorDataset`：Parity 指示 +1，Fashion 指示 −1。SFT-1：偶数→0、奇数→1。SFT-2：偶数在 {0,4} 里随机，奇数在 {1,5} 里随机。Oracle：按 $\pi_0$ 在正确集合上归一化再采样。GRPO：`src/rl_razor/training/grpo.py`，默认 `group_size=8`，优势为组内减均值，可选除组内标准差；`kl_coef=0` 才是论文主 RL。

把一组 GRPO 更新在纸上算一遍。同一张偶数图采 8 个动作，假设四个 0、三个 2、一个 1。1 是奇数，奖励 0；其余奖励 1。组均值是 7/8。四个 0 与三个 2 的优势为正，那个 1 为负。参数只被往「在这张图上更像偶数」推，不会被强迫只输出 0。SFT-1 则每张偶数图都把全部质量推向 0，哪怕 $\pi_0$ 更喜欢 2 和 8。Fashion 头连在同一套隐藏层上，SFT-1 为了把偶数通道拧到 0，隐藏层会动得更狠。这就是浏览器里「拉走」的几何。

验证。浏览器实验把 $\pi_0$ 放在原点，SFT 拉向离线标签中心，RL 沿 on-policy 小步走，距离对应旧任务保持。CPU 实验是同一套二维几何：本机一次运行 SFT 的 L2=2.945、KL=0.159、遗忘约 1.00；五小步 RL 的 L2=0.668、KL=0.009、遗忘 0.36；遗忘与距离的相关 0.921。它钉的是方向，报不了 FashionMNIST 像素准确率。复现 #5 仍走 `SakanaAI/rl-razor-mnist`，必须同时报 Fashion 掉分和 `forward_kl_new`。

### 5.4 反向 KL、多峰策略，以及大约 on-policy

直觉。教科书说反向 KL 会丢覆盖、正向 KL 会保覆盖。Chen et al. 指出：若策略本身已经是多峰（旧能力一个峰、新任务一个峰），反向 KL 往往只把「新峰」挪到目标上，旧峰不动；正向 KL 为了盖住新目标，会把质量从旧峰抽走。类比：你已经会中文和一点英语，新任务是把英语口语改成某个口音。只改英语那一簇，中文还在；若按外部标准口音从头对齐整条声带，中文腔也会被拽走。失效处：他们的高斯混合是一维玩具，不能直接当 LLM 内部几何。

机制。SFT 最小化 $\mathrm{KL}(\pi^{\star}\|\pi_\theta)$（正向）。KL 正则 RL 的最优策略是 $\pi^{\star}(y|x)\propto\pi_{\theta_0}(y|x)\exp(r/\beta)$，最大化 RL 目标等价于最小化 $\mathrm{KL}(\pi_\theta\|\pi^{\star})$（反向）。消融：GRPO 去掉 KL 正则，遗忘仍然低；REINFORCE（无 GRPO 那套优势）遗忘同样低、新任务涨得少一些。Self-SFT（只用初始策略生成的正确样本，滤完再 SFT）仍比逐步 on-policy 忘得多。Iterative-SFT：每个 epoch 开始重新让当前模型生成数据再 SFT，遗忘明显下降。这是「大约 on-policy」：不必每一步采样，但也不能只用 $\pi_0$ 的一份静态集。

数学。目标任务增益与非目标掉分按 Chen 文：

$$
\Delta_g=\mathcal{A}(\pi_{\theta_T},\mathcal{T})-\mathcal{A}(\pi_{\theta_0},\mathcal{T})
$$

$$
\Delta_d=\frac{1}{M}\sum_{j=1}^{M}\mathcal{A}(\pi_{\theta_0},\mathcal{T}'_j)-\mathcal{A}(\pi_{\theta_T},\mathcal{T}'_j)
$$

本课 MNIST 的 $\mathcal{T}$ 是 Parity，$\{\mathcal{T}'\}$ 先只取 FashionMNIST。LLM 尺度的 IFEval / MMLU / Countdown 是 Chen 文的实验，课内不训 8B。

代码。rl-razor-mnist 不实现 Iterative-SFT。你若做 11 节改造，应在 `sft_finetune` 每个 epoch 用当前 `model.sample` 刷新标签，而不是沿用 `label_mode="sft1"` 的固定映射。`grpo.py` 要求 `group_size >= 2`，否则组内均值没有对照。组太小则全对或全错时优势全是零，更新停摆；这不是实现 bug，是 0/1 奖励方差不够。遇到奖励抖在 0.5、参数几乎不动，先看组内标准差再加 `group_size`。

验证。Chen 文图 2：同样增益下，SFT 的 $\Delta_d$ 更大。若你的 MNIST 出现「SFT-1 和 GRPO 的 Fashion 掉分看不出差别」，先检查预训练是否真的把两个任务都学会了（Fashion 基线必须明显高于随机 0.1），以及 SFT 是否训到新任务已经饱和。未饱和时两者都可以「几乎没动」，假阴性。

### 5.5 两条线如何互相限制

直觉。SEAL 解决「数据不够像笔记」。Razor 解决「有数据时别用离线答案把策略拽走」。SEAL 的内环偏偏是 SFT，所以它在「自己出题」上成功，在「连续出题还不忘」上失败。论文图 6 随编辑次数上升，早先段落的准确率下降。作者列出三条后路：奖励里惩罚旧任务回退、用持续学习约束、把内环换成 RL。第三条直接指向本课复现 #5。

机制。知识线评测是无段落 SQuAD：文章进权重，提问时上下文里没有原文。这和 [第 04 课](04_not_just_rag.md)「把名录撤掉」是同一张考卷。SEAL 的涨分证明改写后的蕴涵比原文更适合作 SFT 材料；它不证明可以无限次写入。少样本线用 Llama-3.2-1B-Instruct，11 个可解的训练任务、8 个评估任务，筛选条件是「在最优 TTT 配置下基座能做对」。SEAL 成功率 72.5%，无 RL 的自编辑 20%，纯 ICL 0%，人工最优 TTT 100%。样本极少，论文自己写了。不要把 72.5% 说成通吃 ARC。

少样本自编辑点名的工具来自 Akyürek et al. 2025 的 TTT 协议，写在论文 3.2 节：旋转、翻转、反射、转置、改网格分辨率、链式重复变换；以及学习率、epoch、损失算在全部 token 还是只算输出格。模型看到示范之后，生成一份「调用哪些工具、超参数填多少」的说明书，再 LoRA。没有 RL 时它会乱点工具，所以 20% 远低于人工配好的 100%。本课主线不跑 ARC，但你读 `few-shot/self-edit.py` 时应对上这些开关。知识线没有工具，只有蕴涵文本，所以最小自编辑从知识线起步更便宜。

验证。最小自编辑只检查：模型（或 GPT）是否根据段落写出可微调的文本。连续遗忘必须跑 `general-knowledge/scripts/continual_self_edits.sh`，官方标了高磁盘（`--keep_adapter_dir`）。没跑就引用论文图 6，并写「未复现」。

### 5.6 和第五幕其他课的关系

TTT（第 17 课）改的是测试时一块 $W$。Titans（第 18 课）决定写不写。Nested Learning（第 19 课）决定写多勤。SEAL 决定写什么文本。RL's Razor 决定用哪类文本更新更安全。on-policy 蒸馏（Thinking Machines 博客）则说：学生自己 rollout，教师逐 token 打分，可以把 RL 的 on-policy 和 SFT 的稠密监督合在同一次更新里。本课不实现蒸馏，只要求你能指出：若蒸馏的轨迹来自学生自己，它就站在 Razor 的「近」这一侧；若轨迹来自教师演示，它就站在 SFT 的「远」这一侧。

### 5.7 第 16 课的规则行，和 on-policy 小步

[第 16 课](16_when_weights_must_move.md) 课内 4×4 的规则行是 RAG=0、记忆=0、编辑=0、权重=1。新的计分规则要进参数，日记召回一次不算学会。本课补的是进法。

两条路都会改权重。离线 SFT 把二维策略拉到离原点 L2=2.945、KL=0.159，旧任务遗忘约 1.00：标签来自一张固定答卷，原点附近的旧规则质量被抽走。on-policy 五小步停在 L2=0.668、KL=0.009，遗忘 0.36：只在当前策略附近重新加权，旧规则留得多。部署里若把「严重级别改按客户损失金额」做成一晚上的离线 SFT，规则行的权重格会亮，旧的排法也更容易无声消失。OPEN_PROBLEMS §2 要的「改完之后旧的排法不会无声消失」，对应的就是这条更近的路径。

Razor 不能推出「所有持续学习都改用 RL」。OPEN_PROBLEMS §5 写过：叫小王、改座位、内部流程没有现成的可验证奖励。没有奖励函数，就没有 GRPO 可跑，也没有 ReST-EM 可筛。还缺的是：在无奖励的工作流上如何构造 on-policy 数据，并且旧能力的考题不是再做一遍数学竞赛。SEAL 用下游分数当奖励，全量吃多卡和评委模型，课内只要求最小自编辑能读懂。

纸面核算（和 `lesson_20.py` 同一组超参）：目标 $(2.6,1.4)$，学习率 $0.05$，从原点走五步 $\theta\leftarrow\theta+0.05((2.6,1.4)-\theta)$。五步后 $\theta=(1-0.95^5)(2.6,1.4)$，$0.95^5\approx 0.7738$，系数 $\approx 0.2262$，L2 $\approx 0.668$。旧任务保持写成 $\exp(-\|\theta\|^2)$，原点是 1，五小步约 $0.640$，遗忘约 $0.360$。SFT 的 L2 取决于 40 个噪声点的均值，以 `result.json` 的 2.945 为准，不必手算随机数。

## 6. 源码导读

先读 rl-razor-mnist，再读 SEAL。两条线的文件不要混在一个虚拟环境里：SEAL 的 `requirements.txt` 钉了 `torch==2.7.0`、`vllm==0.9.1`，会和 MNIST 实验的轻量环境打架。

| 路径 | 零件 | 带着什么问题读 |
|---|---|---|
| `src/rl_razor/data.py` | 任务指示、SFT 标签 | +1/−1 接在 784 维后面了吗？`sft1` / `sft2` 如何改标签？ |
| `src/rl_razor/model.py` | 三层 MLP | `forward` 是否只输出 10 维 logits？`copy()` 是否深拷贝？ |
| `src/rl_razor/training/pretrain.py` | 联合预训练 | 每个任务 500 条是在哪里切的？ |
| `src/rl_razor/training/sft.py` | SFT | `label_mode` 三个取值各自走哪条损失？ |
| `src/rl_razor/training/grpo.py` | GRPO | 奖励是奇偶还是精确数字？`kl_coef` 默认 0 吗？ |
| `src/rl_razor/training/oracle.py` | KL 最小标签 | 实现是 I 投影（q 到 π0）还是论文附录那个无穷的方向？ |
| `src/rl_razor/metrics.py` | 正向 KL、奇偶准确率 | `forward_kl` 是否在新任务 loader 上算？ |
| `scripts/pretrain.py` / `scripts/finetune.py` | CLI | `--method` 的合法值是否为 `sft1,sft2,oracle,grpo,grpo_kl`？ |
| `runs/paper_replication/*.sh` | 作者自己的冒烟配置 | 预训练 5 epoch 还是 README 正文的 50？ |
| `SEAL/README.md` | 安装 | Python 3.12、`.env`、双卡声明 |
| `general-knowledge/src/data_generation/make_squad_data.py` | 自编辑生成 | `prompt_key` 默认是 `implications` 吗？`--k` 默认 5 吗？ |
| `general-knowledge/src/inner/TTT_server.py` | 内环 | JSON 里 `train_sequences` 和 `eval_questions` 各是什么；评分是否走 GPT-4？ |
| `general-knowledge/src/query/query_server.py` | 外环驱动 | 默认 `--n_articles` 在 argparse 里是 3，shell 网格却是 50，以哪次调用为准？ |
| `general-knowledge/src/EM/build_SFT_dataset.py` | ReST-EM 的 M 步 | 按 `adapter_mean` 取 top-k |
| `general-knowledge/scripts/train_SFT.sh` | 外环 SFT | 2 进程 DeepSpeed，LoRA rank 64 |
| `few-shot/self-edit.py` | ARC 自编辑 | `--n_self_edits_per_task` 训练 15、评估 5 |
| `few-shot/BC-self-edit.py` | 少样本 ReST-EM | 只强化做对测试格的那些编辑 |

`grpo.py` 里奖励函数是 `((predictions % 2) == (labels % 2)).float()`，标签仍是 0–9 的真数字，不是奇偶两位。读到这里才能理解「多个正确动作」。`TTT_server.py` 文档写明跨请求无状态：每条消息是一次完整的微调加评估。`utils.py` 的 `grade_with_gpt4` 解释了为什么没有 API 密钥就连内环奖励都算不了。

## 7. 实验

浏览器和 CPU 钉机制。锚定主线是 rl-razor-mnist。SEAL 最小自编辑是实战；全量 RL 是加分，失败必须分类。

### Step 0: 浏览器实验「SFT 拉走、RL 不远走」

打开本课网页实验。二维策略空间，原点是 $\pi_0$。你先预测：用离线标签做 SFT，点会停在离原点更远还是更近？用 on-policy 小步，旧任务保持会更高还是更低？再运行。过关条件：预测「SFT 更远、旧任务掉得更多」并且运行结果与之一致。改学习率或标签分布后必须重新预测。不要在未预测时按运行。

### Step 1: CPU 机制实验

在 `experiments/` 目录：

```bash
python3 run.py run 20
```

本机一次运行（Python 3.13.13，`artifacts/lesson20/result.json`；换机器会变，方向不应变）应 `PASS`。先把数字抄进笔记，再对 `checks`：

| 路径 | L2（到原点） | KL | 遗忘（相对原点旧任务） | 新任务准确率 |
|---|---|---|---|---|
| 原点 $\pi_0$ | 0 | 0 | 0 | 约 0.000 |
| 离线 SFT | 2.945 | 0.159 | 约 1.00 | 约 0.999 |
| on-policy 五小步 | 0.668 | 0.009 | 0.36 | 约 0.005 |

遗忘与到原点距离的相关为 0.921。五小步的新任务准确率只有 0.005，仍然高于原点：学习率 0.05 走五步，还没走到目标 $(2.6,1.4)$，这是故意的。SFT 把新任务拉满，也把旧任务几乎抽干。

五条 `checks` 必须全真：

| check | 本机为真时在说什么 |
|---|---|
| `sft_farther_from_origin` | SFT 的 L2 大于五小步（2.945 > 0.668） |
| `sft_kl_exceeds_on_policy` | SFT 的 KL 大于五小步（0.159 > 0.009） |
| `sft_forgets_more` | SFT 遗忘比五小步至少多 0.2（约 1.00 vs 0.36） |
| `distance_tracks_forgetting` | 沿 SFT 方向的 11 个混合点上，距离与遗忘相关 > 0.9（本机 0.921） |
| `both_improve_new_task` | 两条相对原点都提高了新任务准确率 |

纸面核五小步：$\theta_5=(1-0.95^5)(2.6,1.4)$，L2 应落到 0.668 附近。失败阈值写在 `summary` 里：SFT 的 L2 / KL 不超过 RL。不要手写假 JSON。

这是二维几何。KL 在二维 softmax 上算，遗忘是 $\exp(-\|\theta\|^2)$ 相对原点的掉分。它不报 FashionMNIST 像素准确率，不能填进复现 #5 的表格。复现 #5 仍走 Step 2 起的 `SakanaAI/rl-razor-mnist`：同一新任务上 SFT vs GRPO，旧任务保持必须对 SFT 更差，并且遗忘与到原模型的正向 KL 相关。CPU 层钉机制，仓库层钉论文方向，两层不能互相替代。

把日记拿去微调时，同一几何还在：

```bash
python3 run.py extra run onpolicy
```

```bash
python3 run.py extra run seqedit
```

`onpolicy`：整袋噪声 SFT 离原点更远、旧任务伤得更重。`seqedit`：连续四批写入同一张 $W$，无回放则第一批掉下去。GPU 上对应 `gpu print razor-mnist` 和 `gpu print seal`。

### Step 2: 克隆并安装 rl-razor-mnist

单独建环境，不要和 SEAL 共用。

```bash
git clone https://github.com/SakanaAI/rl-razor-mnist.git
```

```bash
cd rl-razor-mnist
```

```bash
pip install -e .
```

若 `pip install -e .` 因构建失败，改用 README 的依赖列表（仍是一条命令）：

```bash
pip install torch torchvision numpy matplotlib scikit-learn pyyaml tqdm
```

不要在这条里加 `wandb`，除非你打算上传。

### Step 3: 联合预训练

README 正文默认 50 epoch、每任务 500 样本、余弦带 warmup。首次下载 MNIST / FashionMNIST 需要网络。

```bash
python scripts/pretrain.py --epochs 50 --lr 1e-3 --n-samples 500 --scheduler cosine_with_warmup --exp-dir experiments/pretrain
```

赶时间允许改用仓库自己的冒烟脚本配置（5 epoch）。那一档只证明流程，分数不计入复现 #5。命令是：

```bash
python scripts/pretrain.py --epochs 5 --lr 1e-3 --n-samples 500 --scheduler cosine_with_warmup --weight-decay 0.0 --exp-dir experiments/pretrain_smoke
```

预期：目录里出现 `pretrained_model.pt` 以及一份 `results.json`。打开预训练结果，确认 FashionMNIST 准确率明显高于 0.1、Parity 奇偶准确率明显高于 0.5。任一条接近随机，后面的「遗忘」没有意义。把 checkpoint 路径抄下来，下文写成 `PRETRAINED`。

### Step 4: SFT-1 与 GRPO（复现 #5 的最小对照）

两条都不要加 `--wandb`。学习率、epoch 先跟 README 示例。

```bash
python scripts/finetune.py --pretrained-model PRETRAINED --method sft1 --lr 1e-4 --epochs 2 --exp-dir experiments/finetune_sft1
```

```bash
python scripts/finetune.py --pretrained-model PRETRAINED --method grpo --lr 1e-4 --epochs 2 --exp-dir experiments/finetune_grpo
```

把 `PRETRAINED` 换成 Step 3 的真实路径。预期：每个 `exp-dir` 写出含 Fashion 准确率、Parity 准确率、正向 KL 的结果文件（`scripts/finetune.py` 会调用 `compute_all_alternative_metrics`）。方向性通过标准：

1. 两组都能把新任务（奇偶）做上去，或 GRPO 至少不比 SFT-1 差到「根本没学」。
2. FashionMNIST 相对预训练的掉分，SFT-1 更大。
3. 新任务上的正向 KL，SFT-1 更大。

若 2 或 3 反号，先加一组 SFT-1 更长或更大学习率（仍单独一条命令），再比帕累托而不是单点。论文图 3 左是整条前沿，不是某一次 `--epochs 2`。

记录时固定三列：Parity 奇偶准确率、Fashion 准确率、新任务正向 KL。预训练那一行也要抄进来当 $\pi_0$。掉分定义为预训练 Fashion 减微调后 Fashion。不要用「感觉忘得少」代替这两列差。若 GRPO 的 Parity 低于 SFT-1 很多，先不要判复现失败：Razor 比的是「同样新任务水平下的旧任务保持」。这时应把 SFT-1 的 lr 降一档，或把 GRPO 再训几个 epoch，让新任务靠近后再比 Fashion。单点超参数打不成论文图 3，但方向必须能看见。

### Step 5: Oracle 与 SFT-2（加强 Razor，仍属主线）

```bash
python scripts/finetune.py --pretrained-model PRETRAINED --method oracle --lr 1e-4 --epochs 2 --exp-dir experiments/finetune_oracle
```

```bash
python scripts/finetune.py --pretrained-model PRETRAINED --method sft2 --lr 1e-4 --epochs 2 --exp-dir experiments/finetune_sft2
```

预期方向：Oracle 的 Fashion 保持应不差于 GRPO，且 KL 更小或相近；SFT-2 介于 SFT-1 与 Oracle 之间。若 Oracle 反而忘得最多，回头读 `oracle.py` 的方向注释，检查预训练模型是否被传进 `compute_oracle_loss`。可选：`--method grpo_kl` 看显式 KL 项改了多少。论文主结论在无 KL 正则的 GRPO 上已经成立。

### Step 6: 画图（有多组结果时）

```bash
python scripts/plot.py --results-dir experiments --pretrained-results experiments/pretrain/results.json --output-dir experiments/plots
```

`--results-dir` 要指到真正收集了各 method 输出的目录，按你 Step 4–5 的落盘改。预期：`figure3.png` 三面板（帕累托、KL 对遗忘、KL 对学习）、`table1.png`、`summary.json`。CPU 上两组点也能看方向；仓库 README 的 wandb sweep 约每方法 500 个配置，标加分，不是复现 #5 的门槛。Table 1 比较的是各指标对遗忘的 $R^2$：正向 KL 应明显高于权重 L1、谱范数、激活 L2。你只有两组点时不要算 $R^2$，只看排序：SFT-1 的 KL 和掉分都更大。这已经够通过方向性复现。

### Step 7: SEAL 最小自编辑（实战，允许停在生成）

另开环境。官方安装：

```bash
git clone https://github.com/Continual-Intelligence/SEAL.git
```

```bash
cd SEAL
```

```bash
python3.12 -m venv seal_env
```

```bash
source seal_env/bin/activate
```

```bash
pip install -r requirements.txt
```

没有 Python 3.12 就停在「资源：解释器版本」，不要强行用 3.10 装 `vllm==0.9.1` 然后把依赖冲突写成代码 bug。

最小可交付分三档，从低到高只做你硬件允许的最高档，并在笔记标明停在哪一档。

档 A（无 GPU、无 API）。读 `general-knowledge/src/data_generation/make_squad_data.py` 的提示模板，打开仓库自带的 `general-knowledge/data/squad_val.json` 前两篇文章，手写三条蕴涵，对照模板要求的「直接或间接推论」。再打开 `general-knowledge/data/synthetic_data/` 下已有的 `eval/base_val.json` 或 `train/iter0_train.json`（若 clone 后存在），看 `completions` 字段。交付：一段对照，说明模型写的蕴涵是否比原文更「原子」。这一档不算跑通官方生成脚本，但算读懂自编辑。

档 B（有 OpenAI 密钥，无本地 7B）。根目录建立 `.env`，写入 `OPENAI_API_KEY=...`（不要把真实密钥贴进课程作业）。把官方 `make_squad_data_openai.sh` 缩到 2 篇，避免 `--n 200` 的账单：

```bash
python general-knowledge/src/data_generation/make_squad_data_openai.py --dataset_in general-knowledge/data/squad_val.json --dataset_out general-knowledge/data/synthetic_data/eval/my_two.json --n 2
```

预期：输出 JSON 含 `completions` 列表。读两条，检查是蕴涵而不是复述。这一档算「跑通官方数据生成」。

档 C（单卡或双卡，能加载 Qwen2.5-7B）。按 `make_squad_data.sh` 起 vLLM 再生成。该 shell 同时起服务和调用 Python，本课把它拆开理解：先 `vllm serve`，健康检查通过后再调用 `python3 -m general-knowledge.src.data_generation.make_squad_data`。单卡冒烟把 `--n` 改成 2、`--k` 改成 2。显存不够就停，记「资源：7B + vLLM」。

### Step 8: SEAL 内环 TTT（加分）

官方 `general-knowledge/scripts/TTT_server.sh` 申请 `--gres=gpu:2`：GPU0 跑 vLLM（开 LoRA 热加载），GPU1 跑 `TTT_server.py`。评分默认 GPT-4。缺任一条件即停，不要改成「用正则匹配答案」却仍声称官方奖励。

若双卡和 API 都有，先起 TTT 服务（按仓库脚本），再单独调用 query，把文章数降到 argparse 默认附近：

```bash
python3 -u -m general-knowledge.src.query.query_server --exp_name smoke --dataset general-knowledge/data/synthetic_data/eval/base_val.json --n_articles 2 --k_completions 2 --eval_times 1 --server_host 127.0.0.1 --zmq_port 5555
```

`--dataset` 必须指向已经含 `completions` 的文件。预期 JSON 含 `baseline_mean_accuracy`、`adapter_mean_accuracy`、`mean_gain`。`mean_gain` 可正可负；冒烟只证明内环通路，不证明 47.0%。

### Step 9: SEAL 全量 ReST-EM（加分，默认标未跑）

完整一轮：档 B/C 的训练集生成（官方训练用 50 篇、每篇 5 条）→ TTT + query 的 `rank_iter0` →

```bash
python3 general-knowledge/src/EM/build_SFT_dataset.py general-knowledge/results/query_server/train/rank_iter0.json
```

再按 `train_SFT.sh` 做外环 LoRA。`train_SFT.sh` 写明 2 进程、DeepSpeed stage 3、`Qwen/Qwen2.5-7B`。论文知识线是两轮 ReST-EM。少样本线另见 `few-shot/README.md` 的 `self-edit.py` 与 `BC-self-edit.py`，基座 `meta-llama/Llama-3.2-1B-Instruct`。

未跑时，验收段落必须填表：

| 步骤 | 状态 | 卡住的是资源还是代码 |
|---|---|---|
| 安装 `requirements.txt` |  |  |
| 自编辑生成 |  |  |
| TTT_server |  |  |
| query_server |  |  |
| `build_SFT_dataset` + `train_SFT` |  |  |
| 少样本 `self-edit.py` |  |  |

「资源」包括：没有 3.12、没有双卡、下不了 7B、没有 OpenAI 评分密钥、单次 30–45 秒乘上 50×5×3 次评测的墙钟。 「代码」包括：ZMQ 连不上、LoRA 热加载失败、DeepSpeed 挂起、JSON 字段对不上。不要把资源问题写成仓库不能用。官方 README 的流程是完整的。

## 8. 配置与预算

| 项目 | 主线 | 缩小 | 不要做 |
|---|---|---|---|
| rl-razor 预训练 | 50 epoch，每任务 500 条 | 5 epoch 冒烟 | 不要把 5 epoch 的 Fashion 掉分写成论文 $R^2=0.96$ |
| 微调对照 | SFT-1 vs GRPO，lr $10^{-4}$，2 epoch | 再加 Oracle、SFT-2 | 不要一上来 wandb sweep 500 组 |
| 硬件 MNIST | CPU 数十分钟到两小时；有 GPU 更快 | 同左 | 不需要 7B |
| SEAL 最小 | 档 A 读模板 / 档 B `--n 2` | 只读 `make_squad_data.py` | 不要用社区山寨分数 |
| SEAL TTT | 2×A100/H100 + GPT-4 评分 | `n_articles=2` | 不要改评分器还声称官方奖励 |
| SEAL 全量 RL | 两轮 ReST-EM，50 篇 × 5 条 | 标未跑 | 不要把单次 TTT 增益写成 47.0% |
| 少样本 ARC | Llama-3.2-1B，加分 | 读 `few-shot/README.md` | 不要把 72.5% 说成全 ARC |

账单。档 B 的 `--n 2` 是数页 API 调用。官方 `make_squad_data_openai.sh` 的 `--n 200` 以及每次 TTT 用 GPT-4 评分，费用和延迟都按「加分」处理。磁盘：MNIST 很小；SEAL 仓库含约 28MB 的 `squad_train.json`，7B 权重另计。

## 9. 验收

复现 #5（必须）：

- 浏览器预测通过：SFT 更远、旧任务掉得更多。
- `python3 run.py run 20` 的五条 `checks` 全真：`sft_farther_from_origin`、`sft_kl_exceeds_on_policy`、`sft_forgets_more`、`distance_tracks_forgetting`、`both_improve_new_task`。本机一次运行应对上 L2 2.945 vs 0.668、KL 0.159 vs 0.009、遗忘约 1.00 vs 0.36、相关 0.921（换机器允许小数位变动，方向和相关 > 0.9 不能翻）。
- 书面：CPU 数字是二维几何，不能当作 rl-razor-mnist 的像素分数。
- 同一预训练 checkpoint 上，SFT-1 的 Fashion 掉分 ≥ GRPO，且 SFT-1 的新任务正向 KL ≥ GRPO。允许「新任务准确率尚未打平」但必须报告。Oracle 若已跑，保持应不差于 GRPO。这一条才是复现 #5，走 `SakanaAI/rl-razor-mnist`。
- 书面：遗忘律用的是 $\mathrm{KL}(\pi_0\|\pi)$ 在新任务输入上的期望；主 RL 实验可以没有显式 KL 正则。
- 书面：第 16 课规则行必须进权重；指出离线 SFT 更容易冲旧规则。Razor 不能推出「所有持续学习都改用 RL」，并举一个无奖励的工作流（叫小王、改座位、内部流程任选）。

SEAL 实战（必须有记录，不要求全量）：

- 至少完成 Step 7 档 A 或更高。
- Step 9 的卡住表填完。全量未跑不算失败，空表算失败。

禁止项：把 SEAL 论文表 2 的 47.0% 抄进「我的结果」；把 Chen 文 8B 数字当成你跑出来的；把 5 epoch 冒烟当成 $R^2=0.96$ 的复现。

## 10. 排错

| 症状 | 原因 | 验证 | 修法 |
|---|---|---|---|
| `sft_farther_from_origin` 为假 | 离线均值没有拉到目标附近，或五小步走太远 | `sft_l2` 对 `rl_l2` | 离线点应绕 $(2.6,1.4)$；RL 只允许五步、lr=0.05 |
| `sft_kl_exceeds_on_policy` 为假 | softmax KL 方向反了 | `sft_kl`、`rl_kl` | KL 应对 softmax(SFT) 相对原点，SFT 应更大 |
| `sft_forgets_more` 为假 | 旧任务保持公式被改，或两步离原点差不多远 | `forget_sft` 应约 1.00，`forget_rl` 应约 0.36 | 保持函数是 $\exp(-\mathrm{L2}^2)$，差必须超过 0.2 |
| `distance_tracks_forgetting` 为假 | 混合网格坏了 | `distance_forget_corr` 应 > 0.9 | 11 个点必须沿 SFT 方向按比例混合 |
| `both_improve_new_task` 为假 | 五小步或 SFT 相对原点没有更近目标 | `origin_new_acc` 约 0，SFT 约 0.999，RL 约 0.005 | RL 不要求拉满新任务，只要求比原点好 |
| 把 CPU 的 0.668 写成 Fashion 准确率 | 二维几何被当成复现 #5 | 数字有没有像素任务 | 复现 #5 必须来自 `rl-razor-mnist` 的 Fashion / `forward_kl_new` |
| Fashion 预训练约 0.1 | 任务指示接错或没学第二个任务 | 打印输入最后一维是否为 −1 | 检查 `TaskIndicatorDataset` |
| SFT 与 GRPO 遗忘差不多 | 新任务没学够，参数几乎没动 | 看 Parity 是否仍接近预训练 | 提高 lr 或 epoch，比前沿不要比单点 |
| GRPO 奖励恒为 0.5 附近抖 | 组内全对或全错，优势为零 | 打印 `rewards_g.std` | 增大 `group_size` 或先确认预训练奇偶已高于随机 |
| Oracle 比 SFT-1 更忘 | 预训练模型没传入 oracle 损失 | 读 `sft.py` 的 `label_mode=="oracle"` 分支 | 用 `--method oracle`，不要自己改标签 |
| `forward_kl` 极小但 Fashion 狂掉 | KL 算在了旧任务 loader 上 | 指标名应带 `_new` | 以 `forward_kl_new` 为准 |
| SEAL `pip` 解不出 | 解释器不是 3.12，或和 MNIST 环境混装 | `python --version` | 独立 `seal_env` |
| TTT 请求无响应 | 服务没绑 ZMQ 或 host 填错 | `TTT_server` 日志是否 `listening` | `server_host` 与绑定地址一致 |
| 评分全否 | 没设 API 密钥，或 GPT 模板被拒 | `.env` 是否被 `source` | 未设密钥就停在档 A/B 生成，不进 TTT |
| `train_SFT.sh` NCCL 挂起 | 双卡通信 | 脚本注释里的 `NCCL_P2P_DISABLE` | 按仓库注释打开；仍挂则记代码/环境，停 |
| query 读到空 completions | 数据集还是原始 SQuAD，没有生成 | JSON 有无 `completions` | 先跑 Step 7 |
| 浏览器未预测就能跑 | 没先选「谁更远 / 谁更保旧」 | 运行按钮应无效 | 先选预测，再运行；改滑块必须重选 |

## 11. 前沿与改造

前沿怎么做。Razor 把「贴近 $\pi_0$」写成可在新任务上监测的量。Chen 文把大约 on-policy 做成更便宜的配方。Thinking Machines 的 on-policy 蒸馏用教师的稠密 token 分数替代稀疏的 0/1 奖励。SEAL 作者已经写了下一步：内环不要停在 SFT。Lai et al. 2025 把 RL 不易忘归因于负例，Razor 第 5 节用 1-0 Reinforce 反对这一点；读新文章时先看它有没有拆开 on-policy 与负梯度。2026 年同主题里，EAFT 用 token 熵挡住自信冲突，SDFT 用示范条件化的自身当教师，SDPO 文则报告更稠密的自蒸馏会把参数漂得更远；*RL Forgets* 在多模态续训上反对「RL 天生少忘」，STABLE 用预算门控决定一次 LoRA 写进去还是缩放掉。细节见 §12。

Razor 不能推出「所有持续学习都改用 RL」。课内二维和 ParityMNIST 都有现成的 0/1 奖励：奇偶对了就给 1。叫小王、改座位、内部流程没有这根尺子。OPEN_PROBLEMS §5 把缺口写成：无奖励的工作流上，如何构造 on-policy 数据，并且旧能力的考题不是再做一遍数学竞赛。第 16 课规则行要进权重；若你只有离线示范、没有可验证奖励，Razor 帮你的是「别用一张离原模型很远的答卷去 SFT」，不是「改成 GRPO」。

我们差在哪。MNIST 主线没有扫出论文那种 $R^2=0.96$ 的整条曲线，除非你做 sweep。SEAL 全量在课内默认跑不完，连续自编辑的遗忘曲线几乎一定是「只讲」。没有把 SEAL 的自编辑生成器和 Razor 的 on-policy 更新接到同一个小模型上。课内五小步的新任务准确率只有 0.005，说明「近」和「新任务已经饱和」可以分开；论文帕累托要比的是饱和之后的旧任务保持。

动手改造清单：

1. 在 rl-razor 里加 Iterative-SFT。位置：`src/rl_razor/training/sft.py`，每个 epoch 用当前模型采样正确奇偶标签再 SFT。预算：单卡一小时。预期：Fashion 掉分低于静态 SFT-1，接近 GRPO。失败标准：掉分仍与 SFT-1 重叠，或新任务学不会。
2. 关掉 GRPO 的负优势，做成 1-0 Reinforce。位置：`grpo.py` 里把负 `A` 置零。预算：与 Step 4 同。预期：遗忘仍接近 GRPO，支持「关键是 on-policy」。失败标准：行为退回 SFT-1。
3. SEAL 提示消融。位置：`make_squad_data.py` 的 `--prompt_key`，对同一篇用 `implications` 和 `self-qa` 各生成 2 条。预算：档 B 的 API。预期：两种都比原文更适合当笔记；RL 是否进一步涨，本课未训外环则标「未测」。失败标准：输出与原文逐句相同。
4. 最小「内环也 on-policy」。位置：课包或独立脚本，用 ParityMNIST 模拟 SEAL：模型先生成一份标签分布（自编辑），再 SFT；对照直接 SFT-1。预算：CPU。预期：若生成分布靠近 $\pi_0$，遗忘应小于 SFT-1。失败标准：两种遗忘无差别。不要声称这是官方 SEAL。
5. 无奖励工作流的负对照。位置：第 16 课规则 10 题，或第 16 课 Step 8 产品日志里的一条内部流程。预算：纸面一小时。预期：写不出 0/1 奖励，就标「不能直接上 GRPO」，并改成「从当前策略采样轨迹、用人工或规则校验过的子集再更新」。失败标准：硬编一个「像不像标准答案」的奖励，却声称已经把 Razor 用到岗位流程上。

顺手复现映射。本课占课程复现承诺第 5 号。第 23 课的自改进环会回来用 SEAL 的生成–筛选–训练，以及本课「关掉筛选会强化错误」的预习：ReST-EM 若把 $r$ 的阈值去掉，等于把坏笔记也抄进去。

## 12. 论文与延伸

1. Zweiger, Pari, Guo, Akyürek, Kim, Agrawal, 2025, *Self-Adapting Language Models*，[arXiv:2506.10943](https://arxiv.org/abs/2506.10943)。
贡献：模型生成自编辑（改写数据、点名超参数或调用增强工具），内环 SFT 把编辑写进权重，外环用更新后模型的下游表现当奖励。
机制：自编辑是自然语言，不是外挂超网络。内环是监督微调（知识线常用 LoRA），所以连续写入仍会忘。奖励依赖当前参数，旧回合数据过期，他们因此坚持 on-policy，并在 GRPO/PPO 不稳后改用 ReST-EM。摘要没有给出课文 5.1 里那张表的百分点；没跑 TTT 不许填那张表。
和本课：Step 7–9 的最小自编辑。CPU 实验不生成文本，只对照「离线答卷 vs on-policy 小步」。连续自编辑遗忘（论文图 6）本课默认答不了，除非你跑了 `continual_self_edits.sh`。
阅读问题：你跑到的最高档自编辑，有没有改变「无段落时能否答题」？若只生成了文本、没做 TTT，必须写「本课实验答不了，因为权重没被内环改过」。

2. Shenfeld, Pari, Agrawal, 2025, *RL's Razor: Why Online Reinforcement Learning Forgets Less*，[arXiv:2509.04259](https://arxiv.org/abs/2509.04259)。
贡献：同样学会新任务时，遗忘由新任务输入上的 $\mathrm{KL}(\pi_0\|\pi)$ 决定；on-policy RL 偏向这族解里离原模型最近的那些。
机制：SFT 的标签来自外部 $\pi_\beta$，可以离 $\pi_0$ 任意远。策略梯度的 $y$ 来自当前 $\pi$，更新留在自己已经会说的句子附近。主 RL 实验关掉显式 KL 正则，只用 0/1 对错当奖励。课内二维几何把这件事缩成距离：离线均值拉远，五小步走得近。
和本课：复现 #5 与 `lesson_20.py`。`sft_kl_exceeds_on_policy`、`sft_forgets_more`、`distance_tracks_forgetting` 三条必须同时为真。课文 5.3 已写本机一次运行的 KL 与遗忘；不要另造数字。Fashion 像素分数 CPU 答不了，必须走 `rl-razor-mnist`。
阅读问题：你的 SFT-1 与 GRPO，Fashion 掉分差和 `forward_kl_new` 差是否同向？若不同向，是预训练没学好，还是只比较了单点超参数？CPU 上用 `sft_kl_exceeds_on_policy` 与 `sft_forgets_more` 先核对方向。

3. Chen, Razin, Narasimhan, Chen, 2025/2026, *Retaining by Doing: The Role of On-Policy Data in Mitigating Forgetting*，[arXiv:2510.18874](https://arxiv.org/abs/2510.18874)，ICML 2026。
贡献：RL 比 SFT 少忘，主因是 on-policy 数据的寻峰，而不是 KL 正则或 GRPO 的优势估计；每个 epoch 重新采样再 SFT 已经有用。
机制：他们把 LM 看成旧知识峰和新任务峰的混合。SFT 最小化正向 KL，为了盖住新目标会抽走旧峰。KL 正则 RL 的最优解走反向 KL，往往只挪新峰。消融里去掉 KL 正则、换成 REINFORCE，遗忘仍然低。Self-SFT（只用 $\pi_0$ 的一份静态正确集）仍比逐步 on-policy 忘得多。
和本课：5.4 与改造 1、2。仓库 GRPO 默认 `kl_coef=0`。若遗忘已经低于 SFT，这一条支持「起作用的是采样分布」。Iterative-SFT 课内没实现，改造 1 才测。8B 上的 IFEval / MMLU 本课实验答不了。
阅读问题：你的 GRPO 若已经关掉 KL 系数、Fashion 掉分仍低于 SFT-1，Chen 文「on-policy 数据够用」你能不能用这一条当证据？若两组掉分重叠，先查预训练 Fashion 是否高于随机，再下结论。

4. Lai, Zhao, Feng, Ma, Liu, Zhao, Lin, Yi, Zhang, Liu, Meng, Zhu, 2025, *Reinforcement Fine-Tuning Naturally Mitigates Forgetting in Continual Post-Training*，[arXiv:2507.05386](https://arxiv.org/abs/2507.05386)。
贡献：多模态持续后训练里，RFT 保住旧任务和通用基准，SFT 则把两者都冲掉；他们把稳定性归因于选择性更新，并给出按 rollout 筛样本的 RIF-RFT。
机制：对照的是同一条持续后训练流水线，只换 SFT 还是带奖励的策略更新。摘要写稳定性主要不是 KL 惩罚，也不是思维链。RIF-RFT 改的是进训练的实例集合：只留当前策略还学得动的样本。底座是 Qwen2.5-VL-7B-Instruct，任务是多模态。
和本课：方向与 Razor 相同，设定不同。本课 GRPO 也没有思维链，不能因此把 MNIST 写成 RFT 复现。`both_improve_new_task` 只说明二维里两种更新都靠近新目标，没有多模态旧任务列。
阅读问题：若有人把本课 `sft_forgets_more` 直接抄进「RFT 自然抗遗忘」的复现表，缺的是哪两样（模型、任务）？本课实验答不了他们的通用基准列。

5. Diao, Yang, Gong, Zhang, Yan, Han, Liang, Xu, Ma, 2026, *Entropy-Adaptive Fine-Tuning: Resolving Confident Conflicts to Mitigate Forgetting*，[arXiv:2601.02151](https://arxiv.org/abs/2601.02151)。
贡献：SFT 里一类破坏性梯度来自「自信冲突」token（模型自己很确定、却被外部标签拉开）；用 token 熵当门，挡住冲突、留下不确定样本。
机制：他们把 RL 少忘和 SFT 多忘收成分布差距：RL 跟着模型内部信念走，SFT 逼模型拟合外部监督。门控看的是熵，不是单纯的预测概率。摘要写在 Qwen / GLM 的 4B–32B、数学 / 医疗 / 智能体设定上，下游接近标准 SFT、通用能力掉得少。
和本课：CPU 的离线点全部当监督，没有 token 熵。`sft_farther_from_origin` 看得见「远」，看不见「哪一个坐标是冲突」。课内没有 4B 模型，答不了他们的通用能力表。
阅读问题：本课 SFT 的 40 个离线点若有几个其实和原点策略冲突，EAFT 会怎么处理？`lesson_20.py` 没有熵，必须写「本课实验答不了，因为没有 token 级门控」。

6. Shenfeld, Damani, Hübotter, Agrawal, 2026, *Self-Distillation Enables Continual Learning*，[arXiv:2601.19897](https://arxiv.org/abs/2601.19897)。
贡献：没有显式奖励时，用示范条件化的自身当教师，做 on-policy 蒸馏（SDFT），从示范里学新技能并少忘旧的。
机制：SFT 学示范是 off-policy。SDFT 把「看见示范之后的模型」当成教师，对学生自己的采样打分，于是训练信号仍在当前策略上。摘要写在技能学习和知识写入上，新任务准确率高于 SFT，顺序学习多技能时没有他们设定里的回退。
和本课：OPEN_PROBLEMS §5 要的「无奖励工作流如何构造 on-policy 数据」，这篇给了一条路。本课 CPU 没有示范、没有教师、没有 ICL 条件化。Thinking Machines 博客是另一条稠密蒸馏，教师是固定大模型，不是示范条件化的自身。
阅读问题：把第 16 课规则 10 题当成示范，SDFT 的教师应该是「读了这 10 题之后的当前模型」，还是一张固定标准答案？本课五小步没有教师，答不了 SDFT 的准确率，只能判断采样来自谁。

7. Wang, Zhao, Liu, Yang, Liu, Guo, Xie, Meng, Liu, Zhu, 2026, *Denser ≠ Better: Limits of On-Policy Self-Distillation for Continual Post-Training*，[arXiv:2607.01763](https://arxiv.org/abs/2607.01763)。
贡献：on-policy 自蒸馏（SDPO）在教师稳定时能加快域内特化，但持续后训练里遗忘更重，甚至崩溃；更稠密并不自动更稳。
机制：他们拿 SDPO 对照 GRPO。摘要写 SDPO 在参数空间和回复空间里漂得更远，并会经教师-学生环放大高频格式伪影。结论：光有 on-policy 数据不够；稠密自蒸馏不能当成持续后训练的默认稳定器。
和本课：Razor / Chen 说 on-policy 少忘，这篇给上限。课内没有蒸馏、没有教师 KL。`sft_l2` 对 `rl_l2` 测的是离线均值 vs 五小步，不是稠密 token 监督。参数漂移范数本课实验答不了。
阅读问题：若把第 12 篇博客的逐 token 教师分数接到连续两个新任务上，按这篇你应该先报哪一个量：新任务准确率，还是参数/回复漂移？本课 CPU 只有二维 L2，写明答不了回复空间漂移。

8. Luo, Wang, Zhou, Ye, Zhao, Zhang, Wei, 2026, *RL Forgets! Towards Continual Policy Optimization*，[arXiv:2607.04364](https://arxiv.org/abs/2607.04364)。
贡献：在新的多模态推理持续学习基准 MRCL 上，标准 RL 仍大忘；他们把原因写成「KL 正则算在当前任务数据上，遗忘却来自旧任务行为漂移」，并给出无回放的 CPO。
机制：常用 PPO/GRPO 的 KL 项在新任务输入上拉住 $\pi$。旧任务的输入分布不同，这项约束对不上。CPO 把历史 KL 松成稀疏的参数移动正则，不存旧数据。摘要写在 Qwen3-VL-8B 上，CPO 相对对照减少遗忘 13.7%，预训练能力提高 7.0%。
和本课：Razor 的遗忘律也在新任务输入上算 $\mathrm{KL}(\pi_0\|\pi)$。这篇说这一项管不住旧任务行为。复现 #5 必须同时看 Fashion 掉分和 `forward_kl_new`；若 KL 小而 Fashion 掉，就碰到缺口。CPU 二维只有一个 KL，没有旧任务输入分布，答不了 MRCL。
阅读问题：你的 `forward_kl_new` 很小、Fashion 却掉了很多，按这篇缺的是哪一项约束？若你只跑了 CPU，写「本课实验答不了 Fashion 像素，二维 KL 没有任务分布」。

9. Hoy, Celik, 2025, *STABLE: Gated Continual Learning for Large Language Models*，[arXiv:2510.16089](https://arxiv.org/abs/2510.16089)。
贡献：每次 LoRA 编辑先过稳定性预算，超了就缩放或拒绝，用来在顺序写入时限制遗忘。
机制：预算三选一：Exact Match 掉分、bits 增加（置信下降）、基座与适配器的 KL。超阈值则 clip LoRA 或丢掉这次更新。改的是「写不写进去」，不是采样分布。摘要写在 Qwen-2.5-7B 上，短序列里 Exact Match 门控累计表现最好。
和本课：SEAL 内环是不加这扇门的 LoRA SFT，所以连续自编辑会忘。Razor 走的是 on-policy 小步，STABLE 走的是写后闸门。没跑 TTT / LoRA，本课实验答不了门控是否拦住连续编辑。
阅读问题：SEAL 连续自编辑若每次 LoRA 先过 KL 预算，你在 Step 9 的哪一格能看见「被拒绝的编辑」？没跑 TTT 就写「本课实验答不了，因为没有 adapter」。

10. Shao, Wang, Zhu, Xu, Song, Bi, Zhang, Zhang, Li, Wu, Guo, 2024, *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*，[arXiv:2402.03300](https://arxiv.org/abs/2402.03300)。
贡献：在数学续预训练之外，提出组相对策略优化（GRPO）：同一输入采一组输出，用组内分数当基线，省掉 PPO 的 critic。
机制：优势是组内减均值（可选再除组内标准差），没有价值网络。这是本课 MNIST 主 RL 的算法来源。论文主任务是 MATH 竞赛题，不是遗忘。摘要写 DeepSeekMath 7B 在 MATH 上 51.7%（无工具、无投票）。
和本课：`SakanaAI/rl-razor-mnist` 的 `grpo.py`，默认 `group_size=8`、`kl_coef=0`。CPU 五小步不是 GRPO，没有组、没有相对优势。`both_improve_new_task` 答不了「组相对」。
阅读问题：课内哪一条 check 对应「组内均值当基线」？没有的话，写明必须看复现 #5 的 `group_size`，CPU 几何答不了 GRPO。

11. DeepSeek-AI, Guo, Yang, Zhang, Song et al., 2025, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*，[arXiv:2501.12948](https://arxiv.org/abs/2501.12948)。
贡献：推理能力可以用纯 RL 激励出来，不必靠人类标注的推理轨迹；得到的反思、校验等模式再用来带小模型。
机制：奖励来自可验证对错（数学、代码、STEM），采样来自当前策略。这和 Razor「$y\sim\pi$」是同一侧，和离线 SFT 示范不是同一侧。摘要没有给遗忘数字，本课也不许把 R1 的竞赛分数写成抗遗忘证据。
和本课：ParityMNIST 的奖励也是 0/1 对错，没有示范轨迹，方向相同。课内没有长思维链，也没有把推理模式蒸馏到小模型。`rl_kl` 相对 `sft_kl` 更小，只说明二维几何近，说明不了 R1 的反思模式。
阅读问题：本课 on-policy 五小步的奖励来自目标点，还是来自人类写好的推理文本？若是前者，它和 R1 的哪一句同向？思维链本身本课实验答不了。

12. Lu / Thinking Machines Lab, 2025-10, *On-Policy Distillation*，[实验室博客](https://thinkingmachines.ai/blog/on-policy-distillation/)。这是博客，不是 arXiv。
贡献：学生自己采样轨迹，教师对每个 token 打 reverse KL，把 on-policy 和稠密监督拼在同一次更新里。
机制：SFT 是 off-policy 稠密，RL 是 on-policy 稀疏，这篇要 on-policy 稠密。实现上把 RL 脚本里的奖励换成 $-\mathrm{KL}(\pi_\theta\|\pi_{\mathrm{teacher}})$。博客用棋步评分作类比：教师打的是你自己的棋，不是大师棋谱。失效处：本课没有教师模型。
和本课：5.6 的对照。轨迹若来自学生，按 Razor 应更近 $\pi_0$；若来自教师演示，就站在 SFT 一侧。MNIST 线没有教师，不能用课内 KL 验证这篇博客。和第 6 篇 SDFT、第 7 篇 SDPO 读的时候要分清教师是谁、监督有多密。
阅读问题：本课 MNIST 实验有没有「教师」角色？没有的话，你不能用 CPU 的 `sft_kl` / `rl_kl` 验证这篇博客。若你只做了二维几何，写「本课实验答不了，因为没有教师 logprob」。

现在第五幕的零件齐了：测试时可以改权重，可以按惊讶写入，可以按频率分档，可以给自己出题，并且更新时应走 on-policy。第六幕会问：若暂时不改权重，把成功的程序放进抽屉里下次检索，算不算持续学习？差哪一口？那是 [第 21 课](21_voyager_skill_library.md)。本课的结论会在那里被再用一次：技能库是外存，对应第 16 课流程 / 记忆格；on-policy 更新才碰权重，对应规则行。两条通道解决的经验类别不同。无奖励的工作流既不能假装已经切到 RL，也不能退回一张离线答卷把旧规则冲掉。



