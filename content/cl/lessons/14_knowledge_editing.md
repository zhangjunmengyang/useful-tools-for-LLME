---
id: 14_knowledge_editing
title: "改一条事实，别把邻居改坏"
summary: "ROME 凭什么说事实存在某层 MLP 的某几个关键？"
unit: memory
play_tools: []
checkpoints:
  - "四指标表。"
  - "编辑 vs 微调 vs RAG 适用表。"
---

# 第 14 课：改一条事实，别把邻居改坏

> 类型：实战（用 EasyEdit 对 GPT-2 级模型做 ROME / MEMIT 编辑和一次遗忘对照，不从头预训练；不列入课程复现表）<br>
> 建议周期：2-4 天<br>
> 硬件：档 A 完成本课浏览器实验和 `python3 run.py run 14`。GPT-2 XL 上 ROME / MEMIT 按 EasyEdit README 约需 10GB / 11GB 显存；更大模型要单卡。无 GPU 时用官方 Colab 或只跑 CPU 机制实验<br>
> 锚定仓库：[zjunlp/EasyEdit](https://github.com/zjunlp/EasyEdit)<br>
> 产物：可靠性 / 泛化 / 局部性 / 流畅性四指标表，一次 unlearning 小实验记录，编辑 vs 微调 vs RAG 适用表

## 1. 这一课做什么

[第 13 课](13_external_memory.md) 把日记写在模型外面。隔一轮还能叫到小王，是因为名录在 Mem0 或 MemFS 里，权重可以一动不动。有一类事实不在你的日记里：模型预训练时已经写进参数的世界知识。问「法国的首都」，它不检索也能答巴黎。这条错了、过时了、或你想做反事实实验时，外挂记忆盖不住：生成时模型仍会从权重里把旧宾语抬出来，和检索到的新宾语抢。

本课换零件：定位少量参数，改一条事实。这叫**知识编辑**（knowledge editing：在部署后的模型上改特定行为，尽量别伤无关行为）。主方法是 Meng 等人的 ROME（Rank-One Model Editing，arXiv:2202.05262）和后续的 MEMIT（Mass-Editing Memory in a Transformer，arXiv:2210.07229）。工具用浙江大学 NLP 组的 EasyEdit，它把 ROME、MEMIT、AlphaEdit 等收成统一的 `BaseEditor`。

第四幕的位置：第 13 课解决「叫得动小王」的外挂段；本课解决「权重里那一条」；[第 15 课](15_loss_of_plasticity.md) 会讲就算没有旧任务考试也会学不动；[第 16 课](16_when_weights_must_move.md) 把外挂、编辑、微调放进一张分流表。贯穿主干在本课是「写到哪里」选了慢速权重里的一小块，以及「怎么写」选了秩一或低秩的约束更新，而不是整网接着训。

编辑成功的标准，课程要你同时看四项：可靠性（改的那条对了）、泛化（换一种问法还对）、局部性（邻居事实别坏）、流畅性（别变成结巴）。EasyEdit 把前三项分别记成 `rewrite_acc`、`rephrase_acc`、`locality`；流畅性在 CounterFact 协议里用 n-gram 熵。只报「那一条问答对了」会把过拟合到原句的方法夸成好方法。这和第 03 课只报平均准确率是同一类错误。

本课还有一条相反方向：机器遗忘（unlearning：让模型忘掉指定内容，其余尽量不动）。编辑要把 $o$ 换成 $o^*$，遗忘要把关于某条或某批数据的痕迹压下去。MUSE（arXiv:2407.06460）给遗忘定了六项性质。全量 7B、哈利·波特语料本课不跑；你只做一次小实验：指定忘掉某条，看相邻知识还在不在。

和 [第 12 课](12_model_merging.md) 的任务向量减法比一下：合并是事后把两个已训模型的差向量拿来加或减，没有定位到「埃菲尔铁塔」那个键。ROME 知道键在哪。减法可能让一批相关事实一起淡掉，这正是局部性要抓住的差异。适用表里会把「粗粒度忘」和「定位擦除」分开，避免把 mergekit 当成本课方法。

术语速查：

| 术语 | 一句解释 |
|---|---|
| 知识编辑 | 改模型对某一条（主语, 关系, 宾语）的预测，尽量少动别的 |
| 定位-改写 | 先找到因果上负责这条事实的层和词元，再改那里的权重 |
| 因果追踪 | 把主语噪声化，再单独恢复某个隐状态，看它能不能把答案救回来 |
| ROME | 把一层 MLP 的输出投影看成线性联想记忆，做一次秩一更新插入新键值 |
| MEMIT | 把同一套更新摊到连续多层，一次写入成千上万条 |
| 可靠性 | 编辑描述句上，新宾语是否压过旧宾语 |
| 泛化 | 同义问法、改写 prompt 上是否仍答新宾语 |
| 局部性 | 无关主语、邻居事实是否保持原样 |
| 流畅性 | 生成是否还像人话，有没有开始循环重复 |
| 机器遗忘 | 与编辑相反：指定内容要没，其余能力要在 |

## 2. 问题

预训练模型把大量 $(s, r, o)$ 三元组压进了权重。$s$ 是主语（法国），$r$ 是关系（首都），$o$ 是宾语（巴黎）。用户或产品需要的是 $o^*$（比如反事实「伦敦」，或真实世界里刚换的首都）。你有三条路：

- 在 prompt 或 RAG 里写新值。权没改。模型仍可能把旧值当常识说出来，尤其是你撤掉检索的时候。第 13 课刚测过这件事。
- 用那一条样本微调。第 10 课见过：局部有效，邻居和旧任务容易一起动。
- 只改「存这条事实」的那一小块计算。这是 ROME 的主张。

核心问题：ROME 凭什么说「事实存在某层 MLP 的某几个关键」？证据分两步，缺一步都不能信。第一步是因果追踪：在主语最后一个词元、中间层的 MLP，恢复干净激活就能把被噪声毁掉的事实预测救回来。第二步是干预权重：真的只改那一层的 $W_{\mathrm{proj}}$，新事实能泛化到同义问法，而邻居主语不太动。若你只做第二步、定位是随便选的层，那只是又一次局部微调，没有验证「存在那里」。

第二个问题：编辑成功除了那一条对，还要看什么？CounterFact 和 EasyEdit 把「听起来改成功了」拆成可分开失败的指标。微调可以可靠性 100%、局部性崩盘（邻居全改成新宾语）。超网络编辑器可以原句对、换种问法就回到旧宾语。有的方法会让模型重复目标词。四项必须同时看。

第三个问题：忘掉和改写是不是同一套旋钮反着拧？表面上，把巴黎改成「不知道」像遗忘。MUSE 指出：逐字背诵没了、知识问答没了、隐私探测没了、无关效用还在、请求变大时还扛得住、连续多次遗忘还不崩，这六项经常不能一起满足。本课用一条事实的小实验看见「忘了目标、邻居还在」或「邻居一起死」，不把这个数字当成 MUSE 的 7B 结论。

## 3. 准备

- 读过 [第 13 课](13_external_memory.md) 的「权重冻结、事实在外」；没跑通 Mem0 也可以上本课，本课的事实来自模型参数，不依赖青禾科技名录。
- 会装 PyTorch，能从 Hugging Face 拉 `gpt2-xl` 或按 EasyEdit 的 `hugging_cache` 目录放权重。
- 课程 `experiments/` 的第 14 课机制实验不下载模型。
- GPU：EasyEdit README 的「Editing GPU memory usage」表（2026-08 核对）写 ROME 在 GPT-2 XL 约 10GB，MEMIT 约 11GB；LLaMA-7B 上 ROME 约 31GB。没有 10GB 时走官方 Colab（README 的 ROME GPT-2 链接）或只完成浏览器 + CPU 两层。
- 磁盘：`gpt2-xl` 约 6GB；EasyEdit 还要 Wikipedia 统计缓存 `data/stats`（ROME 的 $C = KK^\top$ 协方差）。第一次跑会现算或下载，预留 10GB。
- 克隆 EasyEdit 后给它独立虚拟环境。仓库同时支持 EasyEdit 1.0（改参数）和 EasyEdit 2.0（推理时 steering）。本课只用 1.0 的 `BaseEditor` + `hparams/ROME` / `hparams/MEMIT`。

## 4. 学习目标

1. 用因果追踪的三轮（干净、破坏、破坏后恢复）说明：主语最后词元、中间层 MLP 为什么被当成事实存储的位置。
2. 写出 ROME 的秩一更新 $\hat{W} = W + \Lambda (C^{-1} k_*)^\top$，并指出 $k_*$、$v_*$、$C$ 各从哪来。
3. 在 EasyEdit 上完成一次「X 的首都是 Y」类编辑，填写可靠性、泛化、局部性、流畅性。
4. 说明 MEMIT 相对 ROME 多做了什么（多层分摊、批量正常方程），以及何时该用它。
5. 做一次指定遗忘的小实验：目标条下去之后，邻居事实的局部性是否还在。
6. 填一张「编辑 / 微调 / RAG」适用表，并能指出本课实验支持哪几格、哪几格要留到第 16 课。

## 5. 原理

六个机制，每个走完直觉、机制、数学、代码、验证。

### 5.1 编辑任务：一条三元组，四种失败

设基础模型为 $f_\theta$。一条编辑描述是 $(x_e, y_e)$：提示 $x_e$ 描述 $(s, r)$，目标 $y_e$ 是新宾语 $o^*$。EasyEdit README 把单次编辑写成找到 $\theta'$ 使 $f_{\theta'}(x_e)$ 靠近 $y_e$；连续编辑则对整个编辑集求和。课内主线是单次或一小批，`sequential_edit=False`。

失败可以分家，所以必须分指标，不能只看 $x_e$ 本身：

| 指标 | 问的是 | 失败长什么样 |
|---|---|---|
| 可靠性 | $x_e$ 上 $o^*$ 是否压过 $o$ | 原句仍答巴黎 |
| 泛化 | 同义句「法国首都在哪」 | 原句伦敦、改写句又巴黎 |
| 局部性 | 「德国的首都」「里昂是哪个国家的」 | 邻居也变成伦敦 |
| 流畅性 | 以 $s$ 开头继续写 | 「伦敦伦敦伦敦」或句子崩掉 |

ROME 论文的 CounterFact 用 Efficacy / Paraphrase / Neighborhood / Generation Entropy 量这四件事；EasyEdit 映射为 `rewrite_acc`、`rephrase_acc`、`locality`，并在部分设定报告 Fluency。验证协议：四项都记录。缺局部性的「100% 成功」不算验收通过。

类比：改词典里「法国」那一页的首都字段。类比失效处：神经网络没有独立词条，改一个键值会在向量空间里碰到邻居。局部性就是在量这种碰撞。

### 5.2 因果追踪：凭什么说事实在中间层 MLP

ROME 论文先不改权重，改激活。把事实提示看成一张因果图：每个词元、每一层的隐状态 $h_i^{(l)}$ 都是节点。要问的是：哪些节点对「下一个词是西雅图 / 巴黎」有间接效应。

三轮：

1. 干净运行，得到正确宾语的概率 $\mathbb{P}[o]$ 和全部激活。
2. 破坏运行：在嵌入层给主语词元加噪声，模型常会答错，概率 $\mathbb{P}_*[o]$。
3. 破坏后恢复：整体仍用噪声主语，但把某一个 $h_i^{(l)}$ 塞回干净值，看概率能不能回来。

间接效应 $\mathrm{IE} = \mathbb{P}_{*,\ \mathrm{clean}\ h_i^{(l)}}[o] - \mathbb{P}_*[o]$。对约 1000 条事实平均，GPT-2 XL 上出现两个高峰：一个在最后词元的最后几层（临门一脚，不稀奇），一个在**主语最后一个词元的中间层**。把 MLP 和注意力拆开：早期高峰主要由 MLP 贡献（论文报告该处 MLP 的平均间接效应约 6.6%，同位置注意力约 1.6%）。再把 MLP 从计算图里冻结在破坏状态，早期层的因果效应消失，说明这些效应是通过中间层 MLP 走的。

由此得到可检验的存储假说：中间层 MLP 在主语最后词元处读入主语表示，写出关于它的属性；高层注意力再把这些属性拷到预测位置。验证：若假说对，把 ROME 打在该层该词元应同时得到泛化和局部性；打在别的层或别的词元，论文图 5 显示泛化或局部性会掉。浏览器实验「定位-改写」就是把这张层 × 词元热力图变成可点的格子。

### 5.3 ROME：一层 MLP 上的秩一插入

Geva 等人把 Transformer 的 MLP 看成键值记忆。ROME 进一步把第二层线性映射 $W_{\mathrm{proj}}^{(l)}$ 当成线性联想记忆：一堆键 $K$ 映到值 $V$，使 $WK \approx V$。插入新对 $(k_*, v_*)$ 时，在尽量少打扰旧记忆的约束下解最小二乘，得到闭式秩一更新（论文公式 (2)）：

$$
\hat{W} = W + \Lambda (C^{-1} k_*)^\top
$$

其中 $C = KK^\top$ 用维基文本上该层键向量的未中心化协方差估计，$\Lambda$ 与残差 $v_* - W k_*$ 成正比。EasyEdit 把这一层的模块名写在 `hparams/ROME/gpt2-xl.yaml`：`rewrite_module_tmp: "transformer.h.{}.mlp.c_proj"`，`layers: [17]`，`fact_token: "subject_last"`。

$k_*$ 不是手写的。在目标层、主语最后词元，取 MLP 非线性之后的激活，并对若干随机前缀平均，避免键只在一种上下文里有效：

$$
k_* = \frac{1}{N}\sum_{j=1}^{N} k(x_j + s)
$$

$v_*$ 用梯度下降找：把该层 MLP 输出换成候选向量 $z$，使模型在带随机前缀的提示上最大化 $o^*$ 的对数概率，同时用一条「{主语} is a」的 KL 项限制「主语本质」漂走。优化的是向量 $v_*$，不是整网权重。找到后再代入上面的秩一公式，一次写入 $W$。

代码落点：`easyeditor/models/rome/rome_main.py` 的 `apply_rome_to_model`；`compute_u.py` 算键（含 `get_inv_cov` 取 $C^{-1}$）；`compute_v.py` 优化值。验证：编辑后 $\mathbb{P}[o^* \mid x_e] > \mathbb{P}[o \mid x_e]$，同义句同样成立，邻居句方向相反。ROME 论文表 4 在 GPT-2 XL 的 CounterFact 上 Score 89.2，微调 FT 可靠性满分但 Neighborhood Success 掉到 40.4。你的单条实验对不齐这张表，只要求方向：邻居的变化幅度小于目标。

秩一插入和「在同一层上对这一句做 Adam」看起来都像局部微调，差别在约束。微调最小化 $-\log \mathbb{P}[o^* \mid x_e]$，梯度会沿着整层、甚至沿注意力漏到别的主语。ROME 先把问题收成一个键 $k_*$（这个主语）和一个值 $v_*$（这个属性），再用 $C^{-1}$ 把更新方向限制在「少打扰旧键」的那一维。$C$ 估计错或 $k_*$ 取在错误词元上，秩一就会退化成伤邻居的补丁。所以 yaml 里的 `fact_token: "subject_last"` 和 `layers: [17]` 不是装饰：它们是 5.2 的假说被写进配置。改这两项等于在做消融，必须重新填四指标。

### 5.4 四指标怎么算，什么叫没过

EasyEdit 的 `editor.edit(...)` 返回 `metrics` 字典。`post.rewrite_acc` 对应可靠性，`post.rephrase_acc` 对应泛化，`post.locality` 下面按你提供的 `locality_inputs` 分项。流畅性在 CounterFact 类协议里常用生成文本的 n-gram 熵：重复越严重熵越低。ROME 论文把加权的 bi/tri-gram 熵记为 GE，GPT-2 XL 原模型约 626.6，成功编辑应接近这个量级而不是掉到 300。

本课验收用四项的操作定义（单条或十条小集，不要假装是 CounterFact 全量）：

1. **可靠性**：编辑句上新宾语的概率或精确匹配高于旧宾语。
2. **泛化**：至少一条事先写好的同义问法同样成立。没有 `rephrase_prompts` 就没有这项，不许用可靠性冒充。
3. **局部性**：至少一条无关事实（换主语或换关系）的预测与编辑前一致，或 EasyEdit `locality` 分数接近 1。
4. **流畅性**：用编辑后模型从主语生成 2-3 句，人工看有无目标词死循环；有 GPU 时可记 GE。

四项里局部性最容易被丢掉，因为它要你额外准备邻居句。本课实验步骤会强制你写一条邻居。浏览器实验则让你看见：点偏层或点偏词元时，目标和邻居会一起动。

### 5.5 MEMIT：一条不够就摊到多层、一次写一批

ROME 一次改一条、一层。把 10 条、100 条依次 ROME 会互相踩：后面的秩一更新破坏前面的。MEMIT（Meng 等，ICLR 2023，arXiv:2210.07229）做两处推广。

第一，因果追踪在 GPT-J 上显示负责事实的是一段中间层 MLP，而不是单独一层。论文取 $\mathcal{R} = \{3,4,5,6,7,8\}$。EasyEdit 的 `hparams/MEMIT/gpt2-xl.yaml` 写成 `layers: [13, 14, 15, 16, 17]`，同样是一段连续中间层，落在 `mlp.c_proj`。

第二，把一批新键值同时写入线性联想记忆。设旧记忆满足法方程 $W_0 K_0 K_0^\top = M_0 K_0^\top$，新键值块为 $K_1, M_1$，则增量

$$
\Delta = R K_1^\top (C_0 + K_1 K_1^\top)^{-1}
$$

其中 $R = M_1 - W_0 K_1$ 是旧权重在新键上的残差，$C_0$ 仍用经验协方差估计。对每个目标 $z_i$（希望在这段层的末端隐状态变成的向量），把残差 $z_i - h_i^L$ 平均摊到尚未更新的层上，从浅到深逐层写入，每层写完再取下一层激活。这就是 `easyeditor/models/memit/memit_main.py` 里 `execute_memit` 的循环。

论文在 GPT-J 上一次写 10,000 条 zsRE，MEMIT 的综合分 50.7，顺序 ROME 掉到 2.6（特异性被踩穿）。本课不跑 10k。你需要记住的机制是：批量编辑用 MEMIT，单条机制验证用 ROME。两者都是定位-改写，不是超网络。

验证：同一条首都编辑，ROME 打一层 vs MEMIT 打 `layers` 列表，四指标都记。若 MEMIT 局部性明显更好，和论文方向一致；若两条都伤了邻居，先检查你的邻居句是不是其实共享了主语。

### 5.6 遗忘：方向相反，验收更苛刻

编辑插入或替换 $(s,r,o^*)$。遗忘要求模型表现得像没见过某条或某批数据。EasyEdit 把知识擦除列为第三种场景：敏感信息从「是 XXXX」变成空白。表面做法仍可以是 ROME：把 $o^*$ 设成拒绝词或无关词。这只保证可靠性意义上「原宾语下去了」，不保证隐私探测、连续多次请求、效用不崩。

MUSE（Shi 等，arXiv:2407.06460）把遗忘评成六项：(1) 无逐字背诵；(2) 无知识记忆（问答）；(3) 无隐私泄漏；(4) 保留未删除数据上的效用；(5) 删除规模变大时仍可用；(6) 连续遗忘请求下可维持。作者在 7B 模型上对哈利·波特书和新闻做八种算法对照：多数能在不同程度上压低逐字和知识记忆，几乎都会伤效用，也扛不住连续、大规模删除。本课不复制这套基准。

小实验只测两件事，对应六项里的 (2) 和 (4)：

- 目标条：编辑或梯度下降之后，原宾语不再是第一名。
- 邻居条：未列入遗忘集的事实，概率变化明显小于目标。

若邻居一起塌，你做的是损伤，不是遗忘。第 12 课的负任务向量（权重减法）也是一种粗粒度遗忘，和本课定位擦除不是同一精度；适用表里会写这一笔。

## 6. 源码导读

按一次编辑的执行顺序读，不要从 `easyeditor/` 根目录晃到结尾。路径以 2026-08 的 `zjunlp/EasyEdit` 主分支为准。

| 文件 / 符号 | 带着什么问题读 |
|---|---|
| `hparams/ROME/gpt2-xl.yaml` | 改哪一层？`layers: [17]` 和 `fact_token: subject_last` 是否对上 5.2 的早期高峰？ |
| `hparams/MEMIT/gpt2-xl.yaml` | `layers: [13, 14, 15, 16, 17]` 如何对应 5.5 的 $\mathcal{R}$？`mom2_update_weight` 是什么？ |
| `easyeditor/editors/editor.py` 的 `BaseEditor` | `from_hparams`、`edit` 的参数列表；`sequential_edit` 打开后评估点在哪 |
| `easyeditor/models/rome/rome_main.py` | `apply_rome_to_model` 何时 `deepcopy`，改完哪些 key |
| `easyeditor/models/rome/compute_u.py` | $k_*$ 和 $C^{-1}$；第一次跑为何要碰 `data/stats` |
| `easyeditor/models/rome/compute_v.py` | $v_*$ 的目标函数里，KL 项对应论文「essence drift」的哪一段 |
| `easyeditor/models/memit/memit_main.py` | `execute_memit` 如何按层循环、残差如何分摊 |
| `easyeditor/evaluate/` | `rewrite_acc` / `rephrase_acc` / `locality` 从哪来；最大长度 512 的限制 |
| `examples/run_knowedit_llama2.py` | 完整评测脚本；本课缩小实验不要一上来跑 KnowEdit 全量 |
| `tutorial-notebooks/EasyEdit_Example_US_President.ipynb` | README 推荐的总统事实演示，适合看 API，不一定用 ROME |

安装段落以 README「Requirements」为准，不要 `pip install easyedit` 这种未核对的包名。2026-08 打开的命令是先克隆再装依赖：

```text
仓库根目录 README 给出两条等价路线：pip 读 requirements.txt，或 uv pip install --torch-backend cpu -r requirements.txt
```

正式命令写在第 7 节，每条单独一个 bash 围栏。EasyEdit 2.0 的 steering 在 `README_2.md`，本课不走。AlphaEdit 在 `easyeditor/models/alphaedit/`，前沿改造用，主线实验不用。

## 7. 实验

三层都做。浏览器先建立「点错格子会伤邻居」的直觉；CPU 实验用小矩阵把「目标动、邻居少动」钉死；锚定仓库用 EasyEdit 跑一条真实编辑和一次遗忘对照。

固定一条编辑，后面所有步骤共用，避免每人挑不同事实导致无法对照。目标用反事实，这样编辑前后反差大（ROME 论文也强调 CounterFact 比「模型本来就会的真事实」更敏感）：

| 角色 | 文本 |
|---|---|
| 主语 $s$ | The Eiffel Tower |
| 关系提示 $x_e$ | The Eiffel Tower is located in |
| 旧宾语 $o$ | Paris |
| 新宾语 $o^*$ | Rome |
| 泛化句 | The Eiffel Tower is in the city of |
| 邻居句（局部性） | The Louvre is located in |
| 邻居旧宾语 | Paris |

若你的 `gpt2-xl` 在编辑前并不稳定输出 Paris，先换一条模型确实会的事实（例如 ROME 演示常用的 Space Needle / Seattle），但四列（原句、改写、邻居、生成）必须齐全。

### Step 0: 浏览器实验「定位-改写」（先预测）

打开本课页面上的「定位-改写」实验。热力图的横轴是词元（主语内部、主语最后、关系词、最后提示），纵轴是层（浅、中、深）。你点一个格子表示「我把编辑打在这里」，系统用缩小的因果效应和一次秩一更新，显示目标事实和邻居事实各自动了多少。

先预测再运行。未预测则运行按钮无效。改点格子必须作废上次结果。用下面三个预测，对了才过关：

| 你点的位置 | 先猜目标事实 | 先猜邻居事实 |
|---|---|---|
| 主语最后 × 中间层 | 应改到 $o^*$ | 应基本不动 |
| 最后提示词 × 最深层 | 原句可能改对 | 容易变成死记原句，泛化差，或伤流畅性 |
| 主语最后 × 最浅层 | 目标可能弱 | 表示还没汇总，邻居和目标都可能乱 |

过关条件：中间层、主语最后这一格，目标变化大于邻居变化；点偏之后至少一项（泛化或局部性）明显变差。这对应 CURRICULUM 的「层 × 词元热力图」。

### Step 1: CPU 机制实验

在 `experiments/` 目录：

```bash
python3 run.py run 14
```

不下载 GPT。实验对一张线性联想记忆做闭式 rank-1 编辑：先写入若干键值，再改目标键，并对照「丢掉旧表、只按新事实重写」。结果写入 `artifacts/lesson14/result.json`。`python3 run.py run 14` 现在应当全绿。`checks` 五条：`target_moves_to_new_value`、`target_change_exceeds_neighbors`、`unlearn_shrinks_target`、`unlearn_keeps_neighbors`、`naive_rewrite_hurts_neighbors`。

本机一次运行：编辑后目标余弦 1.0，局部性比 4.41；naive 重写邻键位移 4.52。换机器会变，方向不应变：目标贴近新值，邻居动得更少，整表重写更伤邻居。这一层不是 GPT-2 XL + CounterFact；EasyEdit 的四指标在下面几步。

### Step 2: 克隆 EasyEdit 并安装

独立虚拟环境。命令与 2026-08 的 README 一致：

```bash
git clone https://github.com/zjunlp/EasyEdit.git
```

进入仓库根目录后再装依赖（每个围栏一条命令）：

```bash
pip install -r requirements.txt
```

CPU 尝试加分项时，README 另给：

```bash
uv pip install --torch-backend cpu -r requirements.txt
```

默认权重路径是 yaml 里的 `./hugging_cache/gpt2-xl`。把 `model_name` 改成 Hugging Face 的 `gpt2-xl`，或预先把模型下到该目录。第一次 ROME 还需要 `data/stats` 里的层统计；没有时 `compute_u.py` 会按 `mom2_dataset: wikipedia` 现算，耗时长、需下载。

无本地 GPU 时，用 README 的 ROME GPT-2 Colab（Current Implementation 表内链接），步骤与下面相同，只是跑在云端。

### Step 3: 一次 ROME 编辑并打四指标

在 EasyEdit 根目录用 Python（可写成脚本，不要一次 bash 里塞两条命令）：

```python
from easyeditor import ROMEHyperParams, BaseEditor

hparams = ROMEHyperParams.from_hparams("./hparams/ROME/gpt2-xl")
editor = BaseEditor.from_hparams(hparams)

prompts = ["The Eiffel Tower is located in"]
ground_truth = ["Paris"]
target_new = ["Rome"]
rephrase_prompts = ["The Eiffel Tower is in the city of"]
subject = ["The Eiffel Tower"]
locality_inputs = {
    "neighborhood": {
        "prompt": ["The Louvre is located in"],
        "ground_truth": ["Paris"],
    }
}

metrics, edited_model, _ = editor.edit(
    prompts=prompts,
    ground_truth=ground_truth,
    target_new=target_new,
    rephrase_prompts=rephrase_prompts,
    subject=subject,
    locality_inputs=locality_inputs,
    sequential_edit=False,
)
print(metrics)
```

若 `edit` 接口在你这份代码里要求 `subject` 或 `keep_original_weight` 等参数，以 `BaseEditor.edit` 的签名为准，不要死抄。跑完从 `metrics` 填表：

| 指标 | 你填的数或观察 | 本课通过线（方向） |
|---|---|---|
| 可靠性 `rewrite_acc` |  | 编辑后原句偏向 Rome |
| 泛化 `rephrase_acc` |  | 改写句同样偏向 Rome |
| 局部性 `locality` |  | 卢浮宫仍在 Paris，或分数接近编辑前 |
| 流畅性 | 用 `edited_model` 从 "The Eiffel Tower" 生成 40 个 token | 无 "Rome Rome Rome" 死循环 |

四项里只过可靠性：记为失败，并在笔记写「过拟合到原句」。这就是第 2 节那个问题的实验回答。

编辑前后建议各打印一次续写，方便对照流畅性。温度设 0 或极低，避免把采样噪声写成「编辑改变了文风」。若 `edit` 返回的 `edited_model` 与原模型共享存储，先确认 `keep_original_weight` 或 `copy` 参数：ROME 源码里 `apply_rome_to_model(..., copy=False)` 默认就地改。需要前后对比时，要么复制一份权重，要么编辑前先把原模型的续写存进文本文件。

yaml 里 `device: 0` 表示第一块 GPU。只有 CPU 时按 EasyEdit 当前文档把 device 改成 CPU 设备号或改走 Colab，不要假设 `device: 0` 在 Mac 上能跑。

### Step 4: 同一条事实用 MEMIT 再来一次

换超参，不要和 ROME 共用已经改过的模型对象。重新加载：

```python
from easyeditor import MEMITHyperParams, BaseEditor

hparams = MEMITHyperParams.from_hparams("./hparams/MEMIT/gpt2-xl")
editor = BaseEditor.from_hparams(hparams)
```

`edit` 调用与 Step 3 相同。把四指标并排进同一张表。单条编辑上 MEMIT 不必赢 ROME；这一步的目的是确认多层配置能跑通，并观察局部性有没有变。要看 MEMIT 的批量优势，加分项是把 5-10 条不同主语的首都/位置事实一次 `edit`，对比顺序 ROME 5 次后的邻居损伤。10k 条是论文实验，本课不做。

### Step 5: 一次 unlearning 小实验

方向反过来。用未编辑的原模型，目标改为「降低 Eiffel Tower 与 Paris 的绑定」，邻居仍是 Louvre。有两种做法，选一种写进笔记：

**做法 A（定位擦除，推荐）。** 仍用 ROME，`target_new` 设为一个模型较少用作城市名的词（例如 `Nowhere`），或设成与事实无关的短词。测：原句还把 Paris 排第一吗？卢浮宫还在巴黎吗？这对应 EasyEdit 的 knowledge erase 场景：特定行为被改掉，无关样本应少动。

**做法 B（梯度上升对照）。** 在同一层 MLP 对 $\log \mathbb{P}[\mathrm{Paris} \mid x_e]$ 做若干步上升的反号更新（学习率要小）。这更接近「对这条样本做反学习」，通常局部性更差。若你做了 B，把它当负面对照：遗忘算法如果伤成这样，就不算成功。

记录三行：遗忘前目标概率、遗忘后目标概率、遗忘后邻居概率变化。成功方向：目标明显下降，邻居变化更小。若两项一起塌，写「本次方法是损伤不是遗忘」，不要改指标定义来让它通过。

MUSE 的六项里你只碰了知识和效用的缩小版。逐字背诵、隐私攻击、规模、连续请求，明确标「本课实验答不了」。

遗忘和编辑共用同一套定位工具，失败模式却不对称。编辑失败常常是「新宾语没站住」（可靠性或泛化不够）；遗忘失败常常是「目标下去了，模型开始胡言」或「同城的别的地标一起下去」。所以 Step 5 必须同时生成邻居句，不能只看埃菲尔那一句的 top-1。若你用做法 A 把目标改成 `Nowhere`，还要看生成里有没有开始把所有法国地标都写成 Nowhere：那是局部性灾难，不是干净的擦除。

### Step 6: 填适用表（书面）

用本课和 [第 13 课](13_external_memory.md) 的结果，填第 9 节那张「编辑 vs 微调 vs RAG」表。填的时候每格写「本课能支持 / 第 13 课能支持 / 尚未实验」。尚未实验的格留到第 16 课，不许用空话填满。

## 8. 配置与预算

| 项目 | 主线 | 缩小 / 加分 |
|---|---|---|
| 模型 | `gpt2-xl`（1.5B） | Colab 上的 GPT-2；加分：GPT-J 6B 按 `hparams/ROME/gpt-j-6B.yaml` |
| 方法 | ROME 一条 + MEMIT 一条 | 加分：5-10 条批量 MEMIT |
| 层 | ROME `[17]`；MEMIT `[13,14,15,16,17]` | 不要改 yaml 里的 `rewrite_module_tmp`，除非你在做改造清单第 1 项 |
| 显存 | README：ROME 10GB / MEMIT 11GB（GPT-2 XL） | 无 GPU：浏览器 + CPU + Colab |
| 编辑条数 | 1 条反事实 + 1 条遗忘 | KnowEdit / CounterFact 全量标加分，不进本课验收 |
| `v_num_grad_steps` | yaml 默认 20 | 冒烟可改小，但四指标会变，须在笔记声明 |
| 统计缓存 $C$ | `mom2_n_samples: 100000` | 第一次最耗时；缓存留下一趟后可复用 |
| CPU 机制 | `python3 run.py run 14` | 秒级 |

不要和 Mem0、HippoRAG 装在同一个环境。EasyEdit 钉死自己的 `transformers` 版本；README 对 Mistral 写过需 `transformers==4.34.0`，本课 GPT-2 XL 用仓库 `requirements.txt` 即可。2026-07 的更新日志提到 Transformers 5.x 兼容，以你装上的 `requirements.txt` 为准，不要混抄旧博客。

时间：第一次含下载和协方差，单卡可能数十分钟；之后单条 ROME README 称约 5 秒量级（随机器变）。预算按「能做完四指标」估，不按论文 10k 条的小时数估。

## 9. 验收

本课有一句必须留下的判断，也是课程对「改成功了」的定义：编辑成功不是那一条问答对了，而是可靠性、泛化、局部性、流畅性四项同时过关。

交付三样：

1. **四指标表。** Step 3 的 ROME 必填；Step 4 的 MEMIT 能跑就填，跑不完写显存/时间卡在哪。表头固定为：方法、可靠性、泛化、局部性、流畅性、备注（模型、层、yaml）。
2. **遗忘小实验。** 目标条下降、邻居条变化、你用的是做法 A 还是 B。邻居一起塌必须标失败。
3. **适用表。** 把下面这张表填完，每格注明证据来自哪一步。

编辑 vs 微调 vs RAG：

| 需求 | RAG / 外挂记忆 | 知识编辑（ROME/MEMIT） | 微调 / LoRA |
|---|---|---|---|
| 公司内部新事实（小王座位） | 首选。第 13 课已做 | 不合适：预训练里本来没有这条，硬插入也难泛化到业务流程 | 能记，但会伤旧指令；见 [第 10 课](10_sequential_instruction.md) |
| 预训练里一条过时/错误世界知识 | 检索能盖，一撤就露馅 | 合适：定位那条 $(s,r)$ | 能改，局部性通常更差 |
| 一次改几百上千条同类事实 | 检索库直接追加 | MEMIT 的设计点；本课只点到机制 | 可以，贵，遗忘风险按第 10 课协议测 |
| 指定忘掉一段版权文本 | 从语料库删除即可，模型权重仍可能背 | 擦除单条事实可以试，MUSE 显示不够 | 近似 unlearning 的常见做法，效用常掉 |
| 新技能、新推理习惯 | 第 13 课 Step 8 已失败 | 本课实验覆盖不了 | 更接近，但是否必须改权重见第 16 课 |
| 在线、每天都在变的名录 | 外挂主键覆盖 | 每天 ROME 不现实 | 每天全量 SFT 更不现实 |

量化线（方向）：

- ROME 之后，原句新宾语应压过旧宾语。
- 至少一条改写句方向相同，否则可靠性不得单独当作成功。
- 邻居句旧宾语仍应压过被误伤的新宾语，或 `locality` 不明显低于编辑前。
- CPU 实验 `checks` 全真。本机一次运行目标余弦 1.0、局部性比 4.41、naive 重写邻键位移 4.52。换机器会变，方向不应变。不是 GPT-2 XL + CounterFact。
- 标题和报告里不许写「复现 ROME 论文表 4」。课程复现表没有第 14 课。

## 10. 排错

| 症状 | 原因 | 验证 | 修法 |
|---|---|---|---|
| `CUDA error: device-side assert triggered` | 输入超过 EasyEdit 默认 512 | 看 prompt 长度 | README 指向 `easyeditor/evaluate/evaluate_utils.py` 的最大长度；本课句子很短，更可能是 tokenizer 特殊符 |
| 显存炸在 10GB 以下卡 | yaml 按 GPT-2 XL 写 | `nvidia-smi` | 改走 Colab；或确认没有把 LLaMA-7B yaml 拿来用 |
| 第一次极慢 | 在算 `C^{-1}` 或下 Wikipedia | 日志是否出现 `Retrieving inverse covariance` | 等它写完 `data/stats`；下次应走缓存 |
| `rewrite_acc` 高、`rephrase_acc` 接近 0 | 打在错误层/词元，或 $v_*$ 过拟合原句 | 对照浏览器：你是否等价于点了「最后提示 × 深层」 | 确认 `fact_token: subject_last`、`layers: [17]` |
| 邻居卢浮宫也变成 Rome | 局部性失败；FT 常见，ROME 也可能 | 编辑前后都打印邻居句的 top token | 减小更新幅度（`clamp_norm_factor`）；换更远的邻居句做对照 |
| 生成 Rome 循环 | 流畅性失败 | 看 40 token 样例 | 不要把这种结果写成成功；对照论文 GE 下降的失败方法 |
| `from easyeditor import ROMEHyperParams` 失败 | 没在仓库根目录，或依赖没装全 | `pwd` 和 `pip show transformers` | 在克隆根目录、已 `pip install -r requirements.txt` 的环境里跑 |
| MEMIT 和 ROME 分数几乎一样 | 单条编辑差异本来就小 | 是否真的加载了 MEMIT yaml | 看 `hparams.layers` 打印是 1 层还是 5 层 |
| `target_change_exceeds_neighbors` 或 `naive_rewrite_hurts_neighbors` 为假 | rank-1 残差没加对，或 naive 没有丢掉旧表 | 看 `locality_ratio`、`naive_neighbor_delta` | 目标位移应数倍于邻键；整表重写的邻键位移应明显更大 |
| 模型编辑前就不会 Paris | 这条对 gpt2-xl 不是稳事实 | 编辑前打印 `The Eiffel Tower is located in` 的续写 | 换 Space Needle / Seattle 等模型会的事实，四列仍要齐 |

## 11. 前沿与改造

**前沿怎么做。** 定位-改写之后有三条可见的线。MEMIT 把单层秩一变成多层批量。AlphaEdit（Fang 等，arXiv:2410.02355）把扰动投影到需保留知识的零空间，再写参数，目标是连续多次编辑时少伤旧事实；EasyEdit 在 2024-10-24 的更新里加入了它。EasyEdit 2.0（arXiv:2504.15133）改走推理时 steering，不改权重。另一条线是终身编辑的正则：Gupta 等（arXiv:2502.01636、2502.19416）指出顺序定位编辑会让被改矩阵范数一直涨；WikiBigEdit（arXiv:2503.05683）把真实 Wikidata 更新堆到五十万问答对。SimIE、UltraEdit 等宣称把次数从百推到万，本课不跑。遗忘线以 MUSE 为评测，而不是再发明一个「忘了就算成功」。

**我们差在哪。** 课内是一条反事实、一个 1.5B 模型和四项手工指标。没有 CounterFact 的 21,919 条，没有 GPT-J 10k 批量，没有 MUSE 的隐私攻击。连续编辑（`sequential_edit=True`）只在改造清单里。编辑过的权重也没有接到第 13 课的公司助手上：产品里通常是 RAG 盖内部事实、编辑修世界知识，两套并存。

**动手改造清单。**

1. **打偏层。** 位置：复制 `hparams/ROME/gpt2-xl.yaml`，把 `layers: [17]` 改成 `[0]` 或最后一层。预算：再跑一次 Step 3。预期：泛化或局部性相对 17 层变差，和论文图 5 同方向。失败标准：你改了层却没记四指标。
2. **AlphaEdit 对照。** 位置：`easyeditor/models/alphaedit/`，README 的 US President notebook 也列了它。预算：一条首都编辑。预期：连续两次编辑（埃菲尔到罗马，再改另一条建筑）后，第一条的局部性优于第二次 ROME。失败标准：两次都伤成一样却宣称 AlphaEdit 赢了。
3. **把第 13 课的冲突当编辑。** 位置：不要用 ROME 写「王磊座位」，那是内部名录。用 ROME 改一条预训练里的城市事实，再用 Mem0 存青禾座位，问一个混合问题。预算：一次生成。预期：世界知识走编辑后的权重，工位走检索。失败标准：关掉 Mem0 之后工位仍对（说明泄漏进了 prompt）。
4. **负任务向量对照。** 位置：第 12 课若已有两个 LoRA，做一次向量减法「忘掉」风格。预算：CPU 合并 + 本课邻居句。预期：粗粒度减法伤的范围比 ROME 大。失败标准：把 mergekit 减法写成 ROME。

**顺手复现映射。** 无课程复现编号。要对齐 Meng et al. 2022 表 4，需要 CounterFact 子集和 GPT-2 XL，标加分，数字不进本课验收。

## 12. 论文与延伸

本课对照改权重持续学习：定位-改写只动少数事实，不是在岗接着训。谱系只留 ROME。

1. Meng, Bau, Andonian, Belinkov, 2022, *Locating and Editing Factual Associations in GPT*, [arXiv:2202.05262](https://arxiv.org/abs/2202.05262)。
贡献：因果追踪定位主语最后词元的中间层 MLP，并用 ROME 做秩一插入。机制发明处，不是本课主阅读。
机制：先把主语噪声化，再恢复某个隐状态，看能不能把事实预测救回来。确认位置后，对那一层 MLP 的 $W_{\mathrm{proj}}$ 做秩一更新，插入新键值。CounterFact 上同时看泛化和特异性。
和本课：只改少数权重，不是整网持续学习。Step 0 热力图、Step 1 的 `target_moves_to_new_value` 与 `target_change_exceeds_neighbors`、Step 3 四指标都在验证这句话。
阅读问题：点偏到「最后提示 × 最深层」之后，可靠性是否还在、泛化或局部性是否变差？没改层的话这题答不了。

2. Zhang, Yao, Tian, Wang, Deng et al., 2024, *A Comprehensive Study of Knowledge Editing for Large Language Models*, [arXiv:2401.01286](https://arxiv.org/abs/2401.01286)。
贡献：把知识编辑分成外挂知识、把知识并进模型、改模型内部知识三类，并给出 KnowEdit 基准。
机制：定义部署后高效改特定行为、尽量保住其余输入。工程指标写成可靠性、泛化、局部性、可迁移性。v5 注释写 EasyEdit 修过 ROME / MEMIT 计算错误后表 4 数字有更新，课内不要抄旧表。
和本课：本课走第三类，只动少数权重。`BaseEditor.edit` 的 `rewrite_acc` / `rephrase_acc` / `locality` 就是这套指标。KnowEdit 全量本课不跑。
阅读问题：你的 `locality_inputs` 若为空，还能不能报局部性？Step 3 要求不能空；空了就只能报可靠性，算没测邻居。

3. Wang, Li, Zhang, Xu, Yao, Jiang, Xie, Huang, Chen, 2024, *WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models*, [arXiv:2405.14768](https://arxiv.org/abs/2405.14768)。
贡献：终身编辑时，单改参数或单靠检索激活会在可靠性、泛化、局部性上撞墙，于是用主记忆加侧记忆和路由器。
机制：预训练知识留在主参数，编辑写进侧参数，查询时路由器选路。连续编辑用知识分片，不同编辑集落在不同子空间再合并。代码在 EasyEdit。
和本课：仍是局部参数编辑，只改少数事实，不是整网持续学习。本课 `sequential_edit=False`，侧记忆和路由器没实现，答不了终身三角。
阅读问题：本课只编一条埃菲尔，WISE 要解决的「新旧编辑互抢」出现了吗？没开顺序编辑就写本课实验答不了。

4. Fang, Jiang, Wang, Ma, Jie, Wang, He, Chua, 2024, *AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models*, [arXiv:2410.02355](https://arxiv.org/abs/2410.02355)。
贡献：定位后把扰动投影到需保留知识的零空间，再写参数，减轻顺序编辑伤旧事实。
机制：仍是 locate-then-edit，多了一行投影。理论保证编辑后对保留知识的查询输出不变。摘要写在 LLaMA3、GPT2-XL、GPT-J 上给多数定位编辑方法平均提升 36.7%。EasyEdit 已收入。
和本课：只改少数权重，而且约束了扰动方向。改造清单第 2 项。没跑 AlphaEdit 不许编分数。CPU 实验没有零空间投影。
阅读问题：零空间约束和 EWC 的 Fisher 弹簧（[第 05 课](05_ewc_regularization.md)）差在哪？没跑 AlphaEdit 时，只允许从公式对比：一个是编辑扰动的硬投影，一个是整网二次惩罚。

5. Liu, Li, Qi, Liu, Tang, Zheng, Yin, Cheng, Huan, Wang, Gao, 2025, *Unlocking Efficient, Scalable, and Continual Knowledge Editing with Basis-Level Representation Fine-Tuning*, [arXiv:2503.00306](https://arxiv.org/abs/2503.00306)。
贡献：指出改参数对所有输入一视同仁，于是在表示子空间里按输入给每个基加权，做成 BaFT。
机制：对少数表示做与输入相关的基级加权，用来缓解线性更新里的编辑-局部性张力。三个 LLM、五个编辑基准。摘要没写具体百分点，课内不编。
和本课：仍是少量适配，不是整网持续学习。CPU 的 rank-1 对所有键共享同一张 $W$，正是他们说的全局效应；`naive_rewrite_hurts_neighbors` 看见整表重写更伤邻居。BaFT 本身本课没装。
阅读问题：rank-1 编辑后邻键位移明显小于目标，这能证明「参数更新没有全局效应」吗？不能，只能证明这一次残差对邻键较小。BaFT 的输入相关加权本课实验答不了。

6. Gupta, Fang, Ozdemir, Lu, Alaa, Hartvigsen, Anumanchipalli, 2025, *Norm Growth and Stability Challenges in Localized Sequential Knowledge Editing*, [arXiv:2502.19416](https://arxiv.org/abs/2502.19416)。
贡献：发现续预训练、全量微调、LoRA 和各类定位编辑都会让被更新矩阵的 Frobenius 范数上升，局部连续编辑时尤其伤平衡。
机制：只改子集矩阵、其余冻结，范数越积越大。中间激活范数下降，激活子空间相对未编辑模型发生偏移。
和本课：本课一条编辑看不出范数爬坡。`sequential_edit=False`。只改少数事实，距离改权重持续学习仍远。`naive_rewrite_hurts_neighbors` 说明乱改矩阵会伤邻居；连续多次的范数曲线本课答不了。
阅读问题：你只编一条，有没有资格说「范数没涨所以很稳」？没有。要写测量了几次编辑、看了哪一层 $\|W\|_F$。

7. Gupta, Prateepamornkul, Lu, Alaa, Hartvigsen, Anumanchipalli, 2025, *Lifelong Knowledge Editing requires Better Regularization*, [arXiv:2502.01636](https://arxiv.org/abs/2502.01636)。
贡献：把 locate-then-edit 写成两步微调，指出退化来自内部激活过优化和被编辑矩阵范数持续增长。
机制：用 Most-Probable Early Stopping 和显式 Frobenius 范数约束。摘要写这样能把定位编辑扩到 10,000 次，编辑时间降 42-61%。
和本课：本课没有 MPES，也没有范数约束。一条 ROME 不是终身编辑，只改少数事实。
阅读问题：EasyEdit 默认在概率刚过阈值时停了吗？打开你用的 yaml 和日志；若没有早停，本课实验答不了「过优化」这一句。

8. Thede, Roth, Bethge, Akata, Hartvigsen, 2025, *WikiBigEdit: Understanding the Limits of Lifelong Knowledge Editing in LLMs*, [arXiv:2503.05683](https://arxiv.org/abs/2503.05683)。
贡献：用真实 Wikidata 更新自动扩展出终身编辑基准，首版超过 50 万问答对，并把定位编辑拿去和检索增强、持续微调对照。
机制：改的是评测规模和现实性，不是一条新的秩一公式。要看现有编辑方法在大量真实事实上还能否用。
和本课：本课 1 条反事实，答不了 50 万。只编辑少数世界知识，不是在岗学流程。Step 6 适用表要求把「公司内部新事实」留给 RAG，把「预训练里一条世界知识」留给编辑。
阅读问题：把青禾科技 20 条名录拿去 ROME，算不算 WikiBigEdit 要测的那种编辑？不算，预训练里本来没有王磊座位。用 Step 6 适用表把这一格标给外挂。

9. Shi, Lee, Huang, Malladi, Zhao, Holtzman, Liu, Zettlemoyer, Smith, Zhang, 2024, *MUSE: Machine Unlearning Six-Way Evaluation for Language Models*, [arXiv:2407.06460](https://arxiv.org/abs/2407.06460)。
贡献：给语言模型遗忘定六项性质，并在 7B 上对哈利·波特书和新闻评八种流行算法。
机制：六项是无逐字背诵、无知识记忆、无隐私泄漏、无关效用还在、请求变大时扛得住、连续多次遗忘还不崩。摘要写多数算法能不同程度压背诵和知识，只有一种不明显泄漏隐私；效用、规模和连续请求普遍达不到部署方预期。
和本课：Step 5 与 CPU 的 `unlearn_shrinks_target`、`unlearn_keeps_neighbors` 只覆盖知识和效用的缩小版。这是擦除少数绑定，不是持续学习。逐字、隐私、规模、连续请求本课答不了。
阅读问题：你的「邻居还在」对应六项哪一项？对应效用保留。对应不了「无隐私泄漏」就要写答不了。

10. Gao, Wang, Ding, Weng, Wang, Zhu, 2024, *On Large Language Model Continual Unlearning*, [arXiv:2407.10223](https://arxiv.org/abs/2407.10223)。
贡献：面对连续到来的遗忘请求、又不能回放保留数据时，用正交 LoRA 拆开各次遗忘，再用 OOD 检测器决定推理时加载哪块遗忘适配器。
机制：OOO 框架。正交 LoRA 做参数解耦；OOD 检测器用对比熵损失和 glocal 打分。不依赖保留数据。三个任务七个数据集。摘要没写具体百分点，课内不编。
和本课：Step 5 只忘一条，没有连续请求，也没有 OOD 门控。遗忘适配器是少量参数，不是整网持续学习。
阅读问题：连续两次遗忘（先埃菲尔再卢浮宫）之后，第一次的目标会不会回潮？本课没做两次，写本课实验答不了。

11. Fu, Wu, Li, Zhang, Zheng, Ming, Wang, Wang, Zhao, 2025, *Model Merging for Knowledge Editing*, [arXiv:2506.12384](https://arxiv.org/abs/2506.12384)。
贡献：先用稳健监督微调把新知识写进去，再把微调模型和原基础模型合并，顺序编辑时既留新事实又留通用能力。
机制：两段式，不改架构。和第 12 课的任务向量合并是亲戚，目标是知识编辑。摘要写顺序编辑上明显优于现有编辑方法，并更好保住原性能。没有具体百分点。
和本课：本课 ROME 不走合并。改造清单第 4 项的负任务向量是粗粒度减法，范围通常比 ROME 大。合并动的是整网差分，仍只服务少数事实更新，不是在岗持续学习。
阅读问题：用第 12 课的减法「忘掉巴黎」，卢浮宫那一句会不会一起掉？本课邻居句能看方向；若你没做减法，写本课实验答不了，不要把 ROME 的局部性冒充合并。

12. Liu, Pandit, Ye, Choi, Durrett, 2024, *CodeUpdateArena: Benchmarking Knowledge Editing on API Updates*, [arXiv:2407.06249](https://arxiv.org/abs/2407.06249)。
贡献：把知识编辑评测换成代码 API 更新：更新文档在推理时不提供，模型必须靠编辑后的权重解程序综合题。
机制：GPT-4 生成原子可执行的函数更新，再生成会用到该更新的综合题。54 个函数、七个 Python 包、670 道题。摘要写给开源代码模型前置文档不够，现有编辑方法也有明显空档。
和本课：编辑的是函数语义，仍是少量行为，不是学会整套工程习惯。本课埃菲尔事实题答不了 API 更新。第 13 课 Step 8 已说明外挂名录不会发版技能；本课也没有把发版命令写进 ROME。
阅读问题：把「座位换到 5 楼」当成 API 更新去编辑，四指标里哪一项会先坏？预训练里没有这条 API，可靠性都可能立不住。用 Step 6 适用表回答：内部名录不该走 ROME。

做完本课，系统里多了一种写入位置：慢速权重里的局部联想记忆。它仍不会写技能，也没有回答「学着学着学不动」。[第 15 课](15_loss_of_plasticity.md) 换到可塑性丢失：没有旧任务考试，后期任务的学习速度照样会掉。第 16 课再把第 13、14 课收进分流器。

日记改一行、删一行，权重不会自动跟着变：

```bash
python3 run.py extra run stale
```

```bash
python3 run.py extra run tombstone
```

连续改多条事实，GPU 上走 EasyEdit WISE（`sequential_edit=True`）：

```bash
python3 run.py gpu print easyedit-wise
```
