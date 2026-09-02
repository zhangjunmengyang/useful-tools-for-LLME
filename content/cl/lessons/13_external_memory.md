---
id: 13_external_memory
title: "把日记写在模型外面"
summary: "分层记忆、抽取-更新、海马索引，各自解决「叫得动小王」的哪一段？"
unit: memory
play_tools: []
checkpoints:
  - "冲突写入案例。"
  - "外挂记忆不会的三件事：新技能、新推理模式、新运动策略。"
---

# 第 13 课：给公司助手装上可更新的外挂记忆

> 类型：实战（跑公开记忆系统、写入事实与冲突，不从头训练；不列入课程复现表）<br>
> 建议周期：2-4 天<br>
> 硬件：档 A（Mac / CPU）完成本课浏览器实验和 `python3 run.py run 13`；Mem0 / HippoRAG / Letta 需要 API 或本地小模型，效果取决于后端。档 C（单张 24GB）可把检索和抽取换成本地模型<br>
> 锚定仓库：[letta-ai/letta](https://github.com/letta-ai/letta)（原 MemGPT；当前安装入口在 [letta-ai/letta-code](https://github.com/letta-ai/letta-code)）、[mem0ai/mem0](https://github.com/mem0ai/mem0)、[OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG)<br>
> 产物：20 条公司事实的隔轮召回记录、一条「小王换座位」冲突写入案例、「外挂记忆不会的三件事」清单

## 1. 这一课做什么

整门课走到第四幕。前三幕一直在改权重：第 01 课把遗忘跑出来，第 05-08 课给权重加弹簧、回放、扩结构和梯度约束，第 09-12 课把同一套问题搬到语言模型上，用续预训练、顺序指令、正交 LoRA 和模型合并来应付接龙。第 12 课的结论要带着走：[合并](12_model_merging.md)发生在训练结束之后，是事后缝合，不是在线持续学习。

工业界现在真正天天在用的，多半不是 EWC。产品里的「这个助手还记得上周说的事」，通常是把日记写在模型外面：会话摘要、员工名录、向量库、知识图。权重可以一动不动。梁文峰在 2026-05 那场交流里用「把小王叫过来」当例子：人上岗两个月会认识同事，现在的模型每次都要把名录重说一遍。这段话出自 2026-07 公开转写，DeepSeek 没有正式确认。本课只借用这个场景，不把转写里的算力或内部方法当事实。

第 04 课已经对照过三种做法：每次把名录塞进 prompt、检索再塞、微调权重。有名录时大家都会；撤掉名录后，只有改过权重的还记得。那一课把问题钉死了：上下文满了不等于学会了。本课换零件：把名录和日记做成**外挂记忆**（写在模型外面、用时再取的可更新存储），看它能不能在「隔一轮再问」时叫到小王，以及故意写入冲突事实时会覆盖、并存还是胡编。

本课要造的是一个缩小的公司助手。你喂 20 条「青禾科技」内部事实，隔一轮再问「小王在哪」；再写入「小王换座位」，看系统怎么处理新旧两条。对照三条路线：Letta / MemGPT 式的分层记忆（工作记忆、情节记忆、语义记忆分抽屉）、Mem0 式的抽取-更新（对话进、短事实出）、HippoRAG 式的海马索引（实体关系图加 Personalized PageRank，而不是平面向量检索）。做完必须能写出一份「外挂记忆不会的三件事」：新技能、新推理模式、新运动策略。这三件留给第 16 课的分流矩阵。

贯穿主干在本课换的是「写到哪里」这一格：新经验优先写进外挂，不写进慢速权重。

```text
新经验进来
  先决定写到哪里（上下文 / 外挂记忆 / 快速权重 / 慢速权重）
  再决定怎么写（覆盖、追加、压缩、约束、正交、合并）
  写完立刻测：新任务会了没、旧任务还在不在
  长期还要测：还能不能继续学
```

术语速查：

| 术语 | 一句解释 |
|---|---|
| 外挂记忆 | 把日记、名录、摘要写在模型权重外面，用时再取。权重可以不动 |
| 工作记忆 | 当前这一轮还在上下文窗口里的内容，窗口一满就被挤掉 |
| 情节记忆 | 按时间记下的事件：谁在哪天说了什么、小王哪天换了座位 |
| 语义记忆 | 相对稳定的名录和规则：工号、项目代号、周会时间 |
| 虚拟上下文管理 | MemGPT 的核心手法：像操作系统换页一样，把慢存储里的内容搬进有限窗口 |
| 记忆块 / MemFS | Letta 把长期记忆做成 git 管理的文件系统；`system/` 下的文件每轮进系统提示 |
| 抽取-更新 | Mem0 的做法：用 LLM 从对话里抽出短事实，再写入向量库 |
| OpenIE | 开放信息抽取：从句子里抽出（主语, 关系, 宾语）三元组，HippoRAG 用来建图 |
| Personalized PageRank | 从查询相关的种子节点出发，在图上做有偏随机游走，给邻居打分 |
| 冲突写入 | 新事实和旧事实不能同时为真。系统必须覆盖、并存并标时间，或承认不确定 |

## 2. 问题

第 04 课的失败模式很具体。把 20 条公司常识塞进 prompt，助手当时会答；把名录撤掉，它就不知道小王是谁。检索能缓解窗口不够，但检索库如果同时存着「3 楼 12 号」和「5 楼 03 号」，平面向量检索常常把两条都捞回来，模型再在上下文里即兴调和。微调能把事实写进权重，可是第 10 课已经见过：接着训下一批指令，旧事实会被冲掉。

本课要回答的核心问题是：MemGPT 式的分层记忆、Mem0 式的抽取-更新、HippoRAG 式的海马索引，各自解决「叫得动小王」的哪一段？

拆开看，这件事其实有四段，任何一段断了都会在产品里表现为「助手忘了」：

1. **写入**。用户说「小王坐 3 楼 12 号」，系统决定记成短事实、日记原文，还是两者都记。
2. **存放**。记在当前窗口、会话日志、员工名录，还是知识图节点。位置决定下一轮还能不能找到。
3. **召回**。隔一轮再问「把小王叫过来」，系统靠关键词、向量相似度，还是图上的多跳。
4. **冲突**。过了两天有人说「小王换到 5 楼 03」。旧值删不删、新值盖不盖、回答时用哪一条。

三条路线各管一段，不要指望一个仓库包办：

| 路线 | 主要解决 | 不自动解决 |
|---|---|---|
| Letta / MemGPT | 有限窗口里怎么换页；哪些内容必须常驻系统提示 | 多跳「小王的项目的负责人坐哪」这类联想 |
| Mem0 | 对话里抽出可检索的短事实，跨会话按 `user_id` 召回 | 图结构上的多跳；技能和推理习惯 |
| HippoRAG | 用实体关系图做联想检索，补平面 RAG 过不了的多跳 | 助手人格、会话换页、自动冲突覆盖策略 |

还有一个必须当场拆穿的误解：记忆系统等于持续学习。它解决事实的写入和召回。权重没动，技能、偏好、推理习惯通常不会变。第 16 课会用四类经验乘四种写入位置把这句话画成矩阵；本课先用实验把它变成可复查的案例，而不是口号。

## 3. 准备

- 会用 Python 3.10+ 和命令行，读过 [第 04 课](04_not_just_rag.md) 的「上下文 / 检索 / 权重」对照。没写完第 04 课也没关系，本课自带 20 条事实，不依赖那一课的产物文件。
- 课程仓库的 `experiments/` 可以在 CPU 上跑机制实验，不下载模型、不访问网络。
- 锚定仓库实验需要二选一或都做：OpenAI 兼容 API（设 `OPENAI_API_KEY`），或本地 OpenAI 兼容端点（Ollama / vLLM / LM Studio）。没有 API 时，浏览器实验和 CPU 实验仍必须做完。
- 磁盘：Mem0 默认把 Qdrant 放在 `/tmp/qdrant`，历史库在 `~/.mem0/history.db`。HippoRAG 的 `save_dir` 会按「LLM × 嵌入模型」分子目录，预留 2GB 足够跑官方 demo 规模。
- 不要把 [letta-ai/letta](https://github.com/letta-ai/letta) 主分支当可安装的 Python 服务。写课前打开的 README 写明：该仓库现在是项目落地页，V1 服务器源码停在 `archive` 分支，不受支持；当前安装入口是 `npm install -g @letta-ai/letta-code`，需要 Node.js 22.19+。
- 生物类比（海马、睡眠巩固）只在 [第 02 课](02_stability_plasticity.md) 展开。HippoRAG 名字来自海马索引理论，本课只把它当检索算法：OpenIE 建图、Personalized PageRank 扩散。你的系统没有分离的编码器和睡眠。

## 4. 学习目标

1. 画出工作记忆、情节记忆、语义记忆三层抽屉，并说明新信息该进哪一层、召回失败通常卡在哪一层。
2. 用 MemGPT 的虚拟上下文管理解释：有限窗口怎样通过「换页」假装拥有更大记忆；对照 Letta 当前产品里 `system/` 常驻、其余按需读取的做法。
3. 用 Mem0 的 `add` / `search` 走通「喂 20 条事实、隔一轮再问小王在哪」；对冲突写入记录覆盖、并存还是胡编，并核对自己用的算法版本。
4. 跑通 HippoRAG 官方 demo 规模的 `index` + `rag_qa`，说出它和平面向量检索差在哪一步（三元组、图、Personalized PageRank）。
5. 列出外挂记忆不会的三件事（新技能、新推理模式、新运动策略），并各举一个本课实验答不上的例子。
6. 完成浏览器实验「记忆分层」：先预测再运行，预测对了才算过关。

## 5. 原理

六个机制，每个按同一节奏：为什么需要、怎么运转、精确定义、代码落点、怎么验证。

### 5.1 先决定写到哪里：上下文、外挂、权重

公司助手面临的不是「模型够不够聪明」，是「这条新经验写在哪」。同一句「小王坐 3 楼 12 号」可以进三个地方：

- 当前 prompt。便宜，窗口一满或会话一结束就没了。这是第 04 课的 (a)。
- 模型外面的存储。下次检索再塞回窗口。这是本课。
- 模型权重。下次不检索也还在，但改起来贵，还可能冲掉邻居。这是第 10、14 课。

记一条事实为三元组 $(s, r, o)$，例如 $s$ 为王磊、$r$ 为工位、$o$ 为 3 楼 12 号。外挂记忆要维护一个可查询的集合 $\mathcal{M}$，在时刻 $t$ 收到新事实 $e_t$ 时执行：

$$
\mathcal{M}_{t} = U(\mathcal{M}_{t-1}, e_t)
$$

$U$ 是写入规则：覆盖、追加、合并，或拒绝。回答查询 $q$ 时，系统先取子集 $\mathcal{R}(q, \mathcal{M}_t)$，再把它和当前对话一起交给冻结的语言模型 $f_\theta$：

$$
\hat{a} = f_\theta\big(q, \mathcal{R}(q, \mathcal{M}_t)\big)
$$

权重 $\theta$ 在本课里保持冻结。验证方法很硬：把 $\mathcal{R}$ 关掉再问同一句。若答案立刻塌掉，说明「会」发生在外挂上，不在权重里。这和第 04 课撤名录是同一根探针。

类比：外挂记忆像员工手册和座位表，模型像那个会读手册的人。类比失效处：人读手册的同时也会改自己的习惯；冻结的 $f_\theta$ 不会因为手册变厚就学会新的排错手法。

### 5.2 MemGPT / Letta：有限窗口的分层换页

2023 年 Packer 等人的 MemGPT（arXiv:2310.08560）把语言模型当成操作系统。物理内存是有限的上下文窗口，磁盘是无限的外部存储。系统用工具调用决定：哪些内容留在主上下文，哪些写到召回存储（近期对话），哪些写到档案存储（长期文档）。论文把它叫虚拟上下文管理（virtual context management）：用数据在快慢存储之间搬家，让有限窗口看起来像更大的记忆。评测做了两件事：超出窗口的文档分析，以及跨会话聊天里的记忆、反思和演化。

当前产品不叫 MemGPT 了。Letta 的文档把学习写成：智能体主动管理自己的上下文，构造身份、记忆和连续性的 token 空间表示，而不是更新模型权重。2026-08 打开的 [letta-ai/letta](https://github.com/letta-ai/letta) README 写明主仓库已变成落地页，活跃源码在 `letta-ai/letta-code`。安装命令以该 README 为准：

```bash
npm install -g @letta-ai/letta-code
```

记忆的当前落点是 MemFS：一份属于该智能体的 git 仓库，投影到它正在使用的那台机器上。路径即地址。`system/` 下的 Markdown 每轮进系统提示，适合人格、用户偏好、必须常驻的公司规则；`system/` 之外的文件只在需要时读取，目录树本身仍出现在系统提示里当路标。文档给出的最小树是：

```text
MEMORY_DIR/
  system/
    persona.md
    human.md
  reference/
    project-notes.md
  skills/
    my-skill/
      SKILL.md
```

写入规则在产品里变成「智能体自己改文件再 commit」。`/remember` 让你显式教它；dreaming（文档里的 sleep-time 后台子智能体）在对话间隙整理记忆。这对应本课三层抽屉里的：

| 本课抽屉 | MemGPT 论文用语 | Letta 当前落点 |
|---|---|---|
| 工作记忆 | 主上下文 / 窗口内消息 | 当前会话窗口 |
| 情节记忆 | 召回存储（近期对话） | 会话记录；可用消息搜索，不在 MemFS 默认向量索引里 |
| 语义记忆 | 档案存储（长期文档） | MemFS 里的名录、规则、`system/` 常驻块 |

验证：把「小王坐 3 楼 12 号」写进 `system/` 下的名录，新开一轮对话再问「小王在哪」，应能答；只写在当前窗口、不 commit 到 MemFS，新会话应答不上。Letta 文档写明：MemFS 默认不做语义向量索引，找记忆靠文件搜索和读取；要关键词或混合检索需另装 MemFS Search mod。不要把「装了 Letta」理解成「自动有一层 HippoRAG」。

### 5.3 Mem0：对话进、短事实出

Mem0 把记忆当成一层服务：对话进来，短事实出去，下次按用户取回。官方 Python 快速入门（[docs.mem0.ai/open-source/python-quickstart](https://docs.mem0.ai/open-source/python-quickstart)）在 2026-08 核对过，默认组件是 OpenAI `gpt-5-mini` 做抽取、`text-embedding-3-small` 做嵌入、本地 Qdrant 在 `/tmp/qdrant`、SQLite 历史在 `~/.mem0/history.db`。安装：

```bash
pip install mem0ai
```

最小循环：

```python
from mem0 import Memory

m = Memory()
messages = [
    {"role": "user", "content": "王磊工号 1024，坐在 3 楼 12 号工位。"},
    {"role": "assistant", "content": "已记下王磊的工位。"},
]
m.add(messages, user_id="qinghe-assistant")
results = m.search("小王在哪", filters={"user_id": "qinghe-assistant"})
print(results)
```

`Memory.add` 和 `Memory.search` 在 `mem0/memory/main.py`。`add` 负责抽取并写入向量库，`search` 按查询和 `filters` 召回。同一 `user_id` 是跨轮记忆的钥匙：隔一轮再问，只要 `filters` 还指这个助手，短事实应能回来。

写入规则 $U$ 在 Mem0 里不是一句能写死的公式，因为它依赖抽取用的 LLM 和算法版本。仓库 README 在 2026-04 宣布过新记忆算法：单次 ADD-only 抽取，一次 LLM 调用，抽取路径不再做 UPDATE/DELETE，记忆累积、不覆盖；检索侧用时间信息给「当前状态 / 过去事件 / 即将发生」排序。API 上仍然暴露 `m.update` 和 `m.delete`，那是你显式改库，不是抽取器自动覆盖。所以「小王换座位」在新算法下很可能是两条共存的情节，而不是名录上的一格被改写。本课实验必须把实际返回的 `results` 记下来，禁止假设「Mem0 一定覆盖」。

验证分三步：写入后立刻 `search`，应含 3 楼 12 号；隔一轮再用同一 `user_id` 搜，应仍在；写入换座位后再搜「小王现在坐哪」，记录是只回新值、两条都回，还是编造了第三值。第三条叫胡编，算失败。

### 5.4 HippoRAG：用图做多跳，而不是只比向量

平面 RAG 把每段文档压成一条向量，查询也压成一条向量，取最近邻。这对「王磊的工号是多少」够用：查询和那句话几乎同义。对「芦苇项目的负责人坐哪」不够用：查询里没有「3 楼 12 号」，也没有「工号 1024」，中间必须经过「芦苇的负责人是王磊」。这就是多跳。

Gutiérrez 等人的 HippoRAG（arXiv:2405.14831，NeurIPS 2024）把这件事做成：LLM 做 OpenIE 抽出三元组，建成知识图，查询时用 Personalized PageRank 从与查询相关的种子节点往外扩散。论文报告在多跳问答上相对当时的 RAG 方法最高约 20% 的提升；单步 HippoRAG 检索相对 IRCoT 一类迭代检索便宜 10-30 倍、快 6-13 倍。这些数字以论文为准，本课不要求对齐。

当前主分支已经是 HippoRAG 2（arXiv:2502.14802，ICML 2025），副标题是 *From RAG to Memory: Non-Parametric Continual Learning for Large Language Models*。它在 PageRank 之外加强了段落整合和在线用 LLM。仓库 README 把评测拆成三类：事实记忆（NaturalQuestions、PopQA）、意义整合（NarrativeQA）、联想（MuSiQue、2Wiki、HotpotQA、LV-Eval）。本课只跑机制：建图、检索、对比平面 DPR，不宣称复现论文分数。

安装以 README 为准。Python 3.10 环境里：

```bash
pip install hipporag
```

`src/hipporag/HippoRAG.py` 里的 `HippoRAG` 类是主入口。`index` 对文档做 OpenIE 并写图；`add_fact_edges` / `add_passage_edges` / `add_synonymy_edges` 分别加事实边、段落边、同义边；`retrieve` 的注释写明四步：事实检索、再认记忆式筛选、稠密段落打分、Personalized PageRank 重排。图上没捞到事实时，它会退回 `dense_passage_retrieval`。`rag_qa` 把检索接到问答。

Personalized PageRank 可以写成：在图的转移矩阵 $P$ 上，带重启的稳态分布

$$
\pi = (1 - \alpha)\, P^\top \pi + \alpha\, s
$$

$s$ 是查询相关的种子分布（和查询近的实体或事实），$\alpha$ 是重启概率，$\pi$ 高的节点被当成检索结果。平面向量检索没有 $P$，也就没有「沿关系走两步」这条路。

验证：同一份公司文档，问「芦苇项目的负责人坐哪」。平面检索应经常只命中「芦苇由王磊负责」那句，答不出工位；HippoRAG 若建图成功，应能从项目走到人再走到工位。若 OpenIE 把「坐」抽错，图上没有边，它会退回 DPR。把抽到的三元组打印出来，比看最终答案更有用。

### 5.5 冲突写入：覆盖、并存、胡编

持续学习的写入规则必须可测。对同一主键 $(s, r)$ 出现两个宾语 $o_{\text{old}}$ 和 $o_{\text{new}}$ 时，只有三种诚实结果：

| 结果 | 含义 | 问答时该怎样 |
|---|---|---|
| 覆盖 | $U$ 删除或作废旧值 | 「小王在哪」只答新座位 |
| 并存 | 两条都在，通常带时间戳 | 问「现在」应用新值；问「以前」可答旧值 |
| 胡编 | 检索或生成捏造了第三值 | 失败。比忘了更糟 |

Mem0 新抽取路径倾向并存。Letta 若把名录写成 `system/` 里的一格，智能体改文件时可能覆盖，也可能在 `reference/` 另记一笔情节。HippoRAG 默认 `index` 是增量往图上加节点和边，旧三元组还在；`delete` 可以按文档删，但「同一实体改属性」不是它的主路径。纯 RAG 把新旧两段都当文档，排序模型说了算。

数学上，覆盖是 $U$ 对主键唯一：$\mathcal{M}_t(s,r) = o_{\text{new}}$。并存是 $\mathcal{M}_t$ 变成带时间的多重映射。胡编发生在 $\mathcal{R}$ 或 $f_\theta$ 引入了 $\mathcal{M}$ 里没有的 $o$。验证时禁止只看生成文本是否「听起来圆」：必须列出检索到的记忆条目，再看生成有没有超出这些条目。

### 5.6 外挂记忆不会的三件事

梁文峰说的「两个月上岗」，转写里包含认识同事、也包含成为这个岗位上更好的同事。外挂记忆能覆盖前者的事实部分：小王是谁、坐哪、哪个项目。覆盖不了后者里至少三件：

1. **新技能**。例如一组 shell 惯用法：先拉 `main`，再跑仓库里的检查脚本，失败则看第 4 个日志文件。技能是可再调用的程序，不是一条可检索事实。Voyager 式技能库是第 21 课；本课的向量库存下「要用 gitlab.qinghe.local」帮不上执行顺序。
2. **新推理模式**。例如内部计分改成「先看回滚窗口再看错误率」。规则可以写成文档被检索到，但模型按新规则逐步算，通常要改权重或至少改系统里的推理痕迹。检索到规则和会用规则不是同一件事。
3. **新运动策略**。本课没有机器人，但原则一样：控制策略在策略网络的权重里。外挂一张「左拐更稳」的便签，不会改关节力矩。

这三件不是贬低外挂记忆。事实查询上，外挂几乎总是比改权重更便宜、更好撤、更好审计。本课的交付就是承认边界：叫得动小王，不等于上岗两个月。第 16 课会把「事实 / 文档 / 流程技能 / 推理模式」乘「上下文 / 记忆 / 编辑 / 权重」做成通过矩阵。

## 6. 源码导读

读代码按「一条事实从进到出」的路径，不要按文件名字母序。路径以你克隆时的 commit 为准；下面是 2026-08 核对过的入口。

**Mem0（公司助手主路径）**

| 文件 / 符号 | 带着什么问题读 |
|---|---|
| `mem0/memory/main.py` 里的 `Memory.add` | 对话怎样变成短事实？抽取失败时库里有没有条目？ |
| 同文件的 `Memory.search` | `filters={"user_id": ...}` 丢了会怎样？跨用户会不会串库？ |
| `Memory.update` / `Memory.delete` | 和 README 里 ADD-only 抽取是什么关系？谁有权覆盖？ |
| `mem0/vector_stores/` | 默认 Qdrant 路径；换本地嵌入时改哪 |
| `mem0/llms/` | 抽取用的 LLM；没有 `OPENAI_API_KEY` 时要换的配置 |

**HippoRAG（多跳对照）**

| 文件 / 符号 | 带着什么问题读 |
|---|---|
| `src/hipporag/HippoRAG.py` 的 `index` | 文档怎样变成三元组和段落节点？ |
| `add_fact_edges` / `add_passage_edges` / `add_synonymy_edges` | 三种边各连接什么？缺一种边会断在哪？ |
| `retrieve` | 何时走 PPR，何时退回 DPR？ |
| `graph_search_with_fact_entities` / `run_ppr` | 种子节点怎么选？重启之后分落在谁头上？ |
| `examples/demo_openai.py` | README 的最小可跑例子，本课仓库实验从这里缩 |

**Letta（分层对照，当前产品）**

| 位置 | 带着什么问题读 |
|---|---|
| [letta-ai/letta-code](https://github.com/letta-ai/letta-code) README | 当前安装命令、`letta` / `letta server` |
| [MemFS 文档](https://docs.letta.com/concepts/memfs/index.md) | `system/` 为什么每轮都在上下文里？ |
| [Memory & dreaming](https://docs.letta.com/configuration/memory/index.md) | `/remember`、`/init`、dreaming 各自何时写盘 |
| [Agent SDK Memory](https://docs.letta.com/agent-sdk/memory/index.md) | 创建智能体时 `memory` 条目怎样变成 `system/` 文件 |
| `letta-ai/letta` 的 `archive` 分支 | 仅供对照 MemGPT 时代的服务端，本课不安装、不部署 |

读 Letta 时把论文图和产品目录分开：论文的「主上下文 / 召回 / 档案」是机制；MemFS 是 2026 年的实现。不要把 `archive` 分支的 Python 服务器写进实验步骤。

## 7. 实验

三层都做。浏览器先建立「写错抽屉就会召不回」的手感；CPU 实验把冲突规则钉死，不依赖 API；锚定仓库用真实的 Mem0 和 HippoRAG 跑公司助手，Letta 作为分层记忆的体验档。

本课公司设定叫青禾科技。20 条事实固定如下，后面所有实验共用，不要临时改人名。冲突写入只用第 4 条。

| 编号 | 事实 |
|---|---|
| 1 | 公司名是青禾科技 |
| 2 | 总部在杭州云栖园区 B3 |
| 3 | CEO 是林夏 |
| 4 | 王磊（小王）工号 1024，座位在 3 楼 12 号 |
| 5 | 李宁（小李）做设计，座位在 3 楼 08 号 |
| 6 | 项目「芦苇」是内部知识库改版，负责人王磊 |
| 7 | 项目「潮汐」是计费系统，负责人陈可 |
| 8 | 周会每周二 10:00，在 3 楼北会议室 |
| 9 | 请假系统地址是 hr.qinghe.local |
| 10 | 内网必须先连 office-vpn |
| 11 | 代码仓库在 gitlab.qinghe.local |
| 12 | 默认分支是 main |
| 13 | 发版窗口是周四 21:00 到 23:00 |
| 14 | 客服机器人叫青青 |
| 15 | 打印机在 3 楼茶水间 |
| 16 | 食堂在 B1，开放 11:30 到 13:00 |
| 17 | 安全接口人是赵衡 |
| 18 | 差旅标准是高铁二等座 |
| 19 | 新员工导师制为 30 天 |
| 20 | 内部代号「竹简」指员工手册第 3 版 |

冲突句：王磊已换到 5 楼 03 号工位。

### Step 0: 浏览器实验「记忆分层」（先预测）

打开本课页面上的「记忆分层」实验。界面是三层抽屉：工作记忆（当前对话）、情节记忆（日记）、语义记忆（员工名录）。系统会给出一条新信息，你选择写进哪一层，然后模拟「隔一轮再问」。

先预测，再运行。运行前未选预测，按钮应无效。改选抽屉必须作废上次运行。建议用下面三道题做预测，对了才算过关：

| 新信息 | 你该写进哪层 | 隔一轮问什么 | 写错会怎样 |
|---|---|---|---|
| 「把小王叫到北会议室」这句当前指令 | 工作记忆 | 这句话本身不必进名录 | 写进语义记忆会把一次性指令当成永久座位 |
| 「周二周会改到 14:00，只此一次」 | 情节记忆 | 「上一次周会几点」 | 写进语义记忆会覆盖每周二 10:00 这条规则 |
| 「王磊换到 5 楼 03」 | 语义记忆（名录）加一条情节（何时换的） | 「小王现在在哪」 | 只写情节不改名录，问「现在」仍可能答 3 楼 12 |

过关条件：三道题的层选择和「隔轮能否召回」都与实验揭晓一致。这对应 CURRICULUM 的 lab「记忆分层」，网页里的变量名按抽屉三层来，方便和课文对齐。

### Step 1: CPU 机制实验

在课程仓库的 `experiments/` 目录运行：

```bash
python3 run.py run 13
```

这层不下载模型、不访问网络。入口是 `experiments/src/learn_cl_experiments/lessons/lesson_13.py`，结果写入 `artifacts/lesson13/result.json`。`python3 run.py run 13` 现在应当全绿。`checks` 七条：`overwrite_returns_new_seat`、`overwrite_drops_old_seat`、`append_keeps_both_seats`、`graph_overwrites_same_relation`、`graph_multihop_reaches_floor`、`flat_bow_cannot_compose_hops`、`semantic_directory_survives_working_flush`。

本机一次运行（seed 写在 `result.json`）：覆盖规则把小王座位从 A3 写成 B7，旧值不再作为当前座位返回；追加日记同时留下两个值；知识图沿 project→floor 两跳读到 2F，平面词袋检索打平所以答不出；工作记忆清空后语义名录仍在。换机器会变，方向不应变。公司 20 条事实留给下面的锚定仓库；这一层不是 Mem0 线上效果。

### Step 2: 安装 Mem0 并写入 20 条事实

单独建虚拟环境，避免和 HippoRAG / EasyEdit 抢依赖。

```bash
pip install mem0ai
```

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

没有 OpenAI 时，按 [Mem0 配置文档](https://docs.mem0.ai/open-source/configuration) 把 LLM 和嵌入换成 Ollama 或本地兼容端点。默认 `Memory()` 会调用云端抽取，账单按你的用量走，20 条事实加几次搜索通常很小。

把 20 条写成 `user` 消息逐条 `add`，`user_id` 固定为 `qinghe-assistant`。不要把 20 条糊成一条超长消息：抽取器容易丢掉工位这种短槽位。每条 `add` 之后不必立刻问答，先写完。

### Step 3: 隔一轮再问「小王在哪」

新开一个 Python 进程（模拟「下一轮会话」），只做搜索和生成，不再把 20 条塞进 prompt：

```python
from mem0 import Memory

m = Memory()
hits = m.search("小王在哪？王磊的座位是什么？", filters={"user_id": "qinghe-assistant"})
print(hits)
```

预期：`hits["results"]` 里出现工号 1024 或 3 楼 12 号。若为空，先检查 `user_id` 是否写错、Qdrant 路径是否被清掉。对照基线：同一句问一个没有 `search`、也没有名录的裸模型，应答不上青禾科技的工位。把两份输出贴进笔记，这就是「外挂在、权重不在」的证据。

### Step 4: 冲突写入（小王换座位）

在同一 `user_id` 下再 `add`：

```python
conflict = [
    {"role": "user", "content": "通知：王磊已换到 5 楼 03 号工位，3 楼 12 号空出。"},
    {"role": "assistant", "content": "已记下王磊的新工位。"},
]
m.add(conflict, user_id="qinghe-assistant")
print(m.search("小王现在坐哪", filters={"user_id": "qinghe-assistant"}))
```

在笔记里三选一打勾，禁止用「感觉更新了」代替：

| 观察 | 判定 |
|---|---|
| 检索只含 5 楼 03，不含 3 楼 12 作为当前工位 | 覆盖 |
| 两条都在，带不同时间戳或原文 | 并存 |
| 出现从未写入的楼层、或把李宁的座位安到王磊头上 | 胡编（失败） |

若是并存，追加一问「王磊以前坐哪」。时间检索如果生效，旧值应仍可取。把 Mem0 的算法版本（README 或包版本）写进同一段笔记：2026-04 之后的 ADD-only 抽取更常得到并存。

### Step 5: 对照纯 RAG

把 20 条事实写成一个文本文件，每行一条，用你现有的向量检索（Mem0 背后的嵌入也可以，但检索时不要走 `Memory.add` 的抽取，直接把行当文档）。先只索引旧 20 条，问「小王在哪」，应能答 3 楼 12。再把冲突句追加进同一个文件重新索引，再问「小王现在坐哪」。

预期：纯 RAG 经常新旧两条一起进上下文，模型可能答新、答旧、或各说一句。这和 Mem0 的抽取层不同：RAG 没有主键，只有相似度。把「RAG 在冲突上的表现」和 Step 4 的判定并排，这就是本课相对第 04 课多出来的那一列。

### Step 6: HippoRAG 官方 demo 规模

```bash
pip install hipporag
```

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

按 README 的最小例子跑通（完整可跑脚本在仓库 `examples/demo_openai.py`）：

```python
from hipporag import HippoRAG

docs = [
    "王磊工号 1024，座位在 3 楼 12 号。",
    "项目芦苇是内部知识库改版，负责人是王磊。",
    "李宁做设计，座位在 3 楼 08 号。",
]
queries = [
    "王磊的座位在哪？",
    "芦苇项目的负责人坐哪？",
]
hipporag = HippoRAG(
    save_dir="outputs/qinghe",
    llm_model_name="gpt-4o-mini",
    embedding_model_name="text-embedding-3-small",
)
hipporag.index(docs=docs)
print(hipporag.rag_qa(queries=queries))
```

本地 vLLM 时，把 `llm_base_url` 指到 `http://localhost:8000/v1`，模型名改成你在服务的那一个。第一问测单跳，第二问测多跳。把 OpenIE 抽出的三元组（`save_dir` 里的中间结果，或日志）抄三条进笔记。若第二问答不出工位，先看图上有没有「芦苇-负责人-王磊」和「王磊-座位-3 楼 12」这两条边，再谈 PageRank。

HippoRAG 1 的代码在仓库 `legacy` 分支。本课跟主分支（HippoRAG 2）。不要混装两套 `hipporag` 包。

### Step 7: Letta 分层体验（可选，不挡验收）

需要 Node.js 22.19+。

```bash
npm install -g @letta-ai/letta-code
```

```bash
letta
```

用 `/connect` 配模型，`/remember` 写入「王磊工号 1024，座位 3 楼 12 号」，再 `/new` 开一轮问「小王在哪」。打开记忆查看器或环境变量 `MEMORY_DIR` 指向的目录，确认名录落在 `system/` 还是 `reference/`。本步证明「常驻块 vs 按需文件」和浏览器抽屉是同一件事。CLI 装不上或没有模型密钥，用文档里的 MemFS 目录树口头走完 Step 0 的三道题即可，不把 Letta 当本课硬门槛。

### Step 8: 外挂不会的三件事（书面，不当场训模型）

用同一个助手，分别问下面三句，只许用已经写入的记忆，不许微调：

1. 技能：「按青禾的习惯发版，第一步输入哪条命令？」记忆里只有仓库地址和发版窗口，没有命令序列。
2. 推理：「错误率 2%、回滚窗口 5 分钟，按新内部计分该不该发版？」记忆里没有这条计分规则。
3. 运动：「机械臂把打印机从茶水间挪到北会议室，关节怎么动？」记忆里只有打印机位置。

预期：助手要么检索到无关事实然后编，要么承认不知道。把三句原话和回答贴进交付。这就是清单，不要改成「以后加个 Agent 就行了」这种空话。

## 8. 配置与预算

| 项目 | 建议 | 缩小配置 |
|---|---|---|
| Mem0 抽取模型 | 文档默认 `gpt-5-mini` | 本地小模型；抽取质量会掉，冲突实验仍可做 |
| Mem0 嵌入 | `text-embedding-3-small` | 本地嵌入；混合检索需 `pip install mem0ai[nlp]` |
| HippoRAG LLM | README 例子用 `gpt-4o-mini` | `gpt-4o-mini` 或本地 7B；OpenIE 质量决定图好不好 |
| HippoRAG 嵌入 | `text-embedding-3-small` | 同左；NV-Embed / GritLM / Contriever 按仓库说明 |
| 文档规模 | 20 条事实 + 1 条冲突 | HippoRAG 用 3-8 条即可看到单跳 vs 多跳 |
| Letta | `letta-code` + 你已有的 API | 跳过 CLI，只用文档对照抽屉 |
| CPU 实验 | `python3 run.py run 13` | 秒级，固定种子 |
| 预估费用 | 20 次 `add` + 若干 `search` + 一次 HippoRAG `index` | 无 API 则只做 Step 0-1 |

主线按档 C 写命令，但本课没有必须用 7B 的步骤。档 A 至少交浏览器预测记录和 CPU `result.json`。HippoRAG 全量论文数据集（MuSiQue 等）标成加分项，放在 `reproduce/dataset`，本课不跑。

Mem0 自托管服务器是另一条线：`cd server && make bootstrap`。本课用库模式 `pip install mem0ai` 就够。不要一上来 docker 全家桶。

## 9. 验收

书面交付四样，缺一样不算完：

1. **隔轮召回表**。20 条里至少抽 5 条在「写入后立刻问 / 新进程再问 / 撤掉检索再问」三列打勾。新进程再问应仍对；撤掉检索应对不上（除非你偷偷微调了，那就要在笔记里声明）。
2. **冲突写入案例**。原文、检索列表、三选一判定（覆盖 / 并存 / 胡编）、Mem0 或 Letta 的版本号。胡编必须当失败记录，不能改口说「模型比较有创造性」。
3. **HippoRAG 对照**。同一条多跳问句，平面检索 vs `rag_qa` 的命中文档。抄至少两条抽出的三元组。
4. **外挂记忆不会的三件事**。技能、推理模式、运动策略各一句问法和一句失败回答。

量化线（方向，不抄论文分数）：

- 有记忆时，「小王在哪」命中座位或工号。
- 无记忆无上下文时，同一问不得命中 3 楼 12 或 5 楼 03。
- 冲突后，当前问句不得返回从未写入的楼层。
- CPU 实验 `checks` 全真。本机一次运行覆盖读回 B7、丢掉 A3；追加同时留两个值；图两跳到 2F，词袋答不出。换机器会变，方向不应变。不是 Mem0 线上效果。

诚实分档：Mem0 和 HippoRAG 的官方 demo 是**实战**。Letta CLI 是**体验**。MemGPT 论文里的文档分析分数、HippoRAG 论文的多跳百分比是**只讲**，本课实验对不齐，不要写进验收数字。课程复现表（CURRICULUM §4）没有第 13 课，标题里不许写「复现 MemGPT」。

## 10. 排错

| 症状 | 原因 | 验证 | 修法 |
|---|---|---|---|
| `pip install letta` 或克隆 `letta-ai/letta` 主分支装不动服务 | 主仓库已改成落地页，V1 在 `archive` | 打开当前 README 是否写 landing page | 改用 `npm install -g @letta-ai/letta-code`，或跳过 Step 7 |
| Mem0 `search` 空结果 | `user_id` 不一致；Qdrant 被清；没设 API 键导致 `add` 静默失败 | 打印 `add` 返回值；看 `~/.mem0/history.db` 是否增长 | 固定同一个 `user_id`；检查 `OPENAI_API_KEY`；换本地 LLM 配置 |
| `add` 之后短事实面目全非 | 抽取模型改写过度，或 20 条糊成一条 | 对比输入原文和 `results` 里的 `memory` 字段 | 一条事实一次 `add`；必要时自己 `m.update` |
| 冲突后胡编第三座位 | 生成时没用检索到的条目约束 | 先打印 `search` 再生成 | 把命中记忆原文贴进 prompt，禁止自由发挥 |
| HippoRAG `index` 极慢或 OOM | 文档太长或本地 LLM 上下文不够 | 先用 README 的一句 `George Rankin is a politician.` | 把 20 条缩到 3-8 条；`max_model_len` 按仓库 vLLM 说明调 |
| 多跳问句退回 DPR | OpenIE 没抽出关键三元组 | 看 `save_dir` 里的 OpenIE 结果 | 改写句子成「芦苇的负责人是王磊」这种显式关系 |
| `overwrite_returns_new_seat` 或 `graph_multihop_reaches_floor` 为假 | 覆盖主键没写成最新值，或两跳边没接上 | 看 `overwrite_seat`、`graph_floor` | 覆盖必须读回 B7；图沿 project→floor 应到 2F |
| Letta 新会话仍不知道小王 | 写在当前窗口，没有进 MemFS | 看 `MEMORY_DIR` 下有没有对应 md | `/remember` 后再 `/new`；确认 commit |

## 11. 前沿与改造

**前沿怎么做。** 外挂记忆 2025-2026 年的主线有三条。Letta 把换页做成 MemFS 和 dreaming，强调智能体改自己的上下文而不是改权重。Mem0 把抽取做成单次 ADD-only，用时间排序处理「现在 vs 以前」，平台数字和开源 SDK 并不相同，README 写明托管平台含未开源优化。HippoRAG 2 把非参数持续学习说成：新知识进图和段落，不进 $\theta$。A-MEM（Xu 等，arXiv:2502.12110）进一步让记忆条目互相链接、动态组织，本课只作延伸，不装。评测上 LongMemEval / LongMemEval-V2 把跨会话更新、拒答和工作流经验写成基准；Hu 等（arXiv:2604.27003）说明外挂记忆把稳定性-可塑性瓶颈搬到检索，慢速权重可以不动。

**我们差在哪。** 课内助手没有真正的多天在岗，只有「新进程再 search」这一跳。冲突策略没有产品级的主键约束。技能仍是自然语言，不是可执行程序（那是第 21 课）。推理模式和权重无关，第 14 课才碰参数编辑，第 16 课才系统分流。外挂记忆的物理边界是：卸掉存储就不会。最终目的不是把日记做大，而是夜间把稳定事实写入权重，日记只留还没巩固的东西。

```bash
python3 run.py extra run unplug
```

```bash
python3 run.py extra run distill
```

```bash
python3 run.py extra run conflict
```

```bash
python3 run.py extra run route
```

```bash
python3 run.py extra run capacity
```

```bash
python3 run.py extra run shadow
```

```bash
python3 run.py extra run stale
```

```bash
python3 run.py extra run eligible
```

```bash
python3 run.py extra run tombstone
```

```bash
python3 run.py extra run longtail
```

```bash
python3 run.py extra run disagree
```

`unplug` 把「卸库=0」钉死。`distill` 把同一批事实写入 $W$ 后再卸库。`conflict` 对照日记覆盖和权重改写：只练新座位会冲掉其他人，加上回放才保住花名册。`route`：闲聊卸掉上下文就空，座位要进权重才留。`capacity`：36 条全倒进 dim=12 的 $W$ 会撞，只巩固常问的 6 条则这 6 条全中。`shadow`：提示还在时，未训权重也像会。`stale` / `disagree`：日记改了，权重还说旧的，这一条先别卸。`eligible`：一次噪声不要进 $W$。`tombstone`：删日记不等于改权重。`longtail`：常问进权重，长尾先留库。产品侧 Letta / Mem0 / HippoRAG / LangMem 的安装见 `GPU.md`，那是脚手架，不是终点。接法见仓库根目录 `AGENT_MEMORY.md`。

**动手改造清单（2-4 个）。**

1. **给 Mem0 加主键覆盖。** 位置：在 `Memory.add` 之后对 `search("王磊 工位")` 的命中做后处理，同一 $(s,r)$ 只留最新时间戳。预算：CPU，一小时。预期：冲突后当前查询只返回 5 楼 03。失败标准：旧值仍作为无时间标记的当前值出现。
2. **HippoRAG 冲突边。** 位置：`HippoRAG.index` 之后检查同一主语+关系是否有两条宾语，检索时把时间或文档号打进 `doc_metadata`。预算：一次小 `index`。预期：多跳问句能说明「最新座位」。失败标准：两条边权相同，回答随机。
3. **Letta 名录常驻。** 位置：SDK `createAgent({ memory: [{ label: "org-directory", value: "..." }] })`，确认文件落在 `system/`。预算：一次 CLI 会话。预期：`/new` 之后不问检索也能答小王。失败标准：名录只出现在某一次对话的用户消息里。
4. **负对照：把发版技能只写入记忆。** 位置：Step 8 第 1 句。预算：无训练。预期：助手说不出正确命令序列。失败标准：你把命令写进了记忆却声称这证明外挂会了技能。

**顺手复现映射。** 本课没有课程复现编号。若要对照论文，MemGPT 的跨会话聊天、HippoRAG 的多跳问答都是原作者仓库的评测脚本，标成加分阅读，分数不进本课验收。

## 12. 论文与延伸

本课对照改权重持续学习：下列系统把新经验写在模型外面，慢速权重冻结。谱系只留 MemGPT。

1. Packer, Wooders, Lin, Fang, Patil, Stoica, Gonzalez, 2023, *MemGPT: Towards LLMs as Operating Systems*, [arXiv:2310.08560](https://arxiv.org/abs/2310.08560)。
贡献：用分层存储和中断给有限窗口做虚拟上下文管理。机制发明处，不是本课主阅读。
机制：物理内存是上下文窗口，磁盘是外部存储。系统用工具调用决定哪些内容留在主上下文、哪些写到召回存储或档案存储。评测做了超窗口文档分析和跨会话聊天。权重不更新。
和本课：外挂换页通道，不改慢速权重。浏览器三层抽屉和 Step 7 的 MemFS 在演搬家；CPU 的 `semantic_directory_survives_working_flush` 对应窗口清空后名录仍在。论文文档分析分数本课答不了。
阅读问题：隔轮召回成功时，关掉检索或常驻块，答案会不会立刻塌？若塌了，说明「会」在外挂上。Step 3 能做这件事。

2. Gutiérrez, Shu, Gu, Yasunaga, Su, 2024, *HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models*, [arXiv:2405.14831](https://arxiv.org/abs/2405.14831)。
贡献：OpenIE 建图加 Personalized PageRank，补平面 RAG 过不了的多跳。
机制：新知识写成三元组和段落节点，不写进 $\theta$。查询从种子节点做有偏随机游走。摘要写多跳问答相对当时方法最高约 20%；单步检索相对 IRCoT 便宜 10-30 倍、快 6-13 倍。本课不要求对齐这些数。
和本课：外挂图检索，不改慢速权重。Step 6 的「芦苇负责人坐哪」和 CPU 的 `graph_multihop_reaches_floor`、`flat_bow_cannot_compose_hops` 看见两跳对词袋打平。
阅读问题：若 OpenIE 没抽出「芦苇-负责人-王磊」，你的多跳失败验证的是抽边还是 PageRank？把抽出的三元组抄出来再判断。

3. Gutiérrez, Shu, Qi, Zhou, Su, 2025, *From RAG to Memory: Non-Parametric Continual Learning for Large Language Models*, [arXiv:2502.14802](https://arxiv.org/abs/2502.14802)。
贡献：HippoRAG 2 把非参数持续学习做成新知识进图和段落，并补上事实记忆和意义整合，避免加了图反而伤普通事实检索。
机制：仍用 Personalized PageRank，加深段落整合和在线 LLM。摘要写联想任务相对当时最强嵌入模型约 7%。评测拆成事实、意义整合、联想三类。
和本课：主分支就是这一版，慢速权重冻结。20 条单跳事实可以看有没有比纯向量更差；论文百分点本课样本太小，答不了。
阅读问题：Step 5 的平面 RAG 和 Step 6 的 HippoRAG 在「王磊座位」这类单跳上，有没有明显回退？有就记下来；没有也不能宣称复现了那 7%。

4. Chhikara, Khant, Aryan, Singh, Yadav, 2025, *Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory*, [arXiv:2504.19413](https://arxiv.org/abs/2504.19413)。
贡献：从对话里动态抽取、合并、检索显著信息，并给出图记忆变体。
机制：改的是外存和抽取，不是 $\theta$。在 LOCOMO 上对比记忆系统、不同块大小的 RAG、全上下文、开源记忆、专有系统和托管平台。摘要写 LLM-as-a-Judge 相对 OpenAI 约 26% 相对提升；图记忆整体约高 2%；相对全上下文 p95 延迟低 91%，token 费省 90% 以上。
和本课：Step 2-4 的 `add` / `search` 就是这条抽取通道，不改慢速权重。CPU 的 `overwrite_returns_new_seat` 与 `append_keeps_both_seats` 对照覆盖和并存。论文四类题的完整分数本课答不了。
阅读问题：冲突写入后，你的 `search` 是覆盖、并存还是胡编？用 Step 4 的判定表，不要用论文的 26%。

5. Kang, Ji, Zhao, Bai, 2025, *Memory OS of AI Agent*, [arXiv:2506.06326](https://arxiv.org/abs/2506.06326)。
贡献：把智能体外挂记忆做成分层操作系统：短、中、长期个人记忆，外加存储、更新、检索、生成四个模块。
机制：短到中按对话链 FIFO；中到长用分段页。评测在 LoCoMo 上，摘要写 GPT-4o-mini 相对基线 F1 平均 49.11%、BLEU-1 平均 46.18%。权重不动。
和本课：浏览器三层抽屉是它的缩小版，不改慢速权重。Letta 的 `system/` 常驻更接近长期页。本课没有 FIFO 分页实现，答不了那两个百分比。
阅读问题：把「王磊换座位」只写进工作记忆，隔一轮还能不能答「现在坐哪」？用 Step 0 的层选择和 `semantic_directory_survives_working_flush` 回答。

6. Xu, Liang, Mei, Gao, Tan, Zhang, 2025, *A-MEM: Agentic Memory for LLM Agents*, [arXiv:2502.12110](https://arxiv.org/abs/2502.12110)。
贡献：按 Zettelkasten 给每条记忆生成结构化笔记，并在写入时与历史记忆动态建链、触发旧条目更新。
机制：新记忆带上下文描述、关键词、标签；系统分析历史条目建立链接。记忆演化发生在外存属性上，不更新慢速权重。六个基础模型上对比当时基线。摘要没写具体百分点，课内不编。
和本课：本课不装 A-MEM。Mem0 的短事实抽取是更扁的版本，慢速权重同样冻结。主键冲突是否被链接自动解决，课内只能对照 Step 4。
阅读问题：A-MEM 的动态链接能不能自动解决小王换座位的主键冲突？用 Step 4 的覆盖 / 并存 / 胡编表对照论文方法；对不上就写本课实验答不了。

7. Maharana, Lee, Tulyakov, Bansal, Barbieri, Fang, 2024, *Evaluating Very Long-Term Conversational Memory of LLM Agents*, [arXiv:2402.17753](https://arxiv.org/abs/2402.17753)。
贡献：给出 LoCoMo：平均约 300 轮、9K token、最多 35 个会话的超长对话，以及问答、事件摘要、多模态对话生成评测。
机制：用智能体加人物设定和时间事件图生成对话，人再改一致性。评测的是记忆系统能否跨很多会话取回事实，不要求改权重。摘要写长上下文 LLM 或 RAG 有帮助，仍明显落后人类。
和本课：Mem0 论文和 MemoryOS 都在这个基准上打分。本课通道仍是外挂，不改慢速权重。20 条事实加一次隔轮，答不了 35 会话。
阅读问题：Step 3 的「新进程再 search」对应 LoCoMo 的哪一段？它测的是跨会话召回，还是 300 轮时间因果？用你实际问的那一句回答。

8. Wu, Wang, Yu, Zhang, Chang, Yu, 2024, *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory*, [arXiv:2410.10813](https://arxiv.org/abs/2410.10813)。
贡献：把聊天助手长期记忆拆成五项：信息抽取、跨会话推理、时间推理、知识更新、拒答。
机制：500 道题嵌进可伸缩的用户-助手历史。摘要写商业助手和长上下文模型在持续交互上大约掉 30 个准确率点。他们把记忆设计拆成索引、检索、阅读，并试了会话切分、事实增强键、时间感知查询扩展。
和本课：不改慢速权重。Step 4 的冲突写入对应「知识更新」；拒答和 500 题规模本课答不了。
阅读问题：小王换座位之后，系统有没有拒答不确定的楼层，还是胡编了第三值？这对应五项里的更新还是拒答？用 Step 4 判定。

9. Wu, Ji, Kawatkar, Kwan, Gu, Peng, Chang, 2026, *LongMemEval-V2: Evaluating Long-Term Agent Memory Toward Experienced Colleagues*, [arXiv:2605.12493](https://arxiv.org/abs/2605.12493)。
贡献：把长期记忆评测从用户聊天史换成定制环境里像熟同事：静态状态、动态状态、工作流、环境坑、前提意识。
机制：451 题，历史轨迹最多 500 条、115M token。记忆系统消费轨迹、返回紧凑证据再问答。他们给出 AgentRunbook-R（分池 RAG）和 AgentRunbook-C（轨迹当文件、编程智能体取证）。摘要写 C 平均 72.5%，最强 RAG 48.5%，现成编程智能体 69.3%。
和本课：仍是外挂取证，不改慢速权重。Step 8 的「发版技能」对应工作流知识；本课没有 115M token 轨迹，答不了那三个准确率。
阅读问题：Step 8 问发版第一步命令时失败，说明缺的是事实名录还是工作流卡片？用你贴的那句回答。

10. Wang, Mao, Fried, Neubig, 2024, *Agent Workflow Memory*, [arXiv:2409.07429](https://arxiv.org/abs/2409.07429)。
贡献：从经验里归纳可复用工作流，再按需提供给智能体指导下一步动作。
机制：离线可从训练集归纳，在线可在测试时归纳。评测在 Mind2Web 和 WebArena。摘要写相对基线成功率相对提升 24.6% 和 51.1%；在线 AWM 在任务分布差变大时绝对点高 8.9 到 14.0。存的是流程卡片，不更新慢速权重。
和本课：Step 8 已经证明自然语言名录不会发版技能。AWM 走外挂工作流，不改慢速权重。本课没跑网页导航，答不了那两个相对提升。
阅读问题：若把发版命令序列写成一条记忆，Step 8 会不会从失败变成成功？本课实验能回答「有没有这张卡片」，不能回答 WebArena 分数。

11. Yu, Chen, Feng, Chen, Dai, Yu, Zhang, Ma, Liu, Wang, Zhou, 2025, *MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent*, [arXiv:2507.02259](https://arxiv.org/abs/2507.02259)。
贡献：分段读长文，用覆盖策略更新记忆，并用多会话 RL（扩展 DAPO）端到端训练这个记忆智能体。
机制：训练的是「怎么覆盖记忆槽」这条策略。长文档写在记忆槽里，不是把 3.5M token 写进世界知识的慢速权重。摘要写从 8K 上下文、32K 文本训到 3.5M QA，性能损失小于 5%；512K RULER 上 95% 以上。
和本课：覆盖写入对照 CPU 的 `overwrite_returns_new_seat` 与 `overwrite_drops_old_seat`。本课助手的 $\theta$ 完全冻结，不训练 MemAgent；3.5M 外推答不了。
阅读问题：本课覆盖规则读回 B7、丢掉 A3。隔一轮问「以前坐哪」，覆盖通道还能不能答？用 `append_keeps_both_seats` 对照：并存才能答「以前」，覆盖答不了。

12. Hu, Long, Wang, 2026, *When Continual Learning Moves to Memory: A Study of Experience Reuse in LLM Agents*, [arXiv:2604.27003](https://arxiv.org/abs/2604.27003)。
贡献：说明外挂记忆并没有取消持续学习，只是把瓶颈从改参数搬到检索时新旧经验互抢。
机制：$(k,v)$ 框架拆开「经验怎么表示」和「怎么组织检索」。在 ALFWorld 和 BabyAI 顺序任务上，抽象程序记忆比细轨迹更好迁移；负迁移更伤难题；更细的组织不一定更好。
和本课：这是本课对照改权重持续学习的收束。慢速权重不动，瓶颈在 $U$ 和 $\mathcal{R}$。CPU 的覆盖 / 追加 / 图是三种表示；`flat_bow_cannot_compose_hops` 是检索组织失败。ALFWorld 数字本课答不了。
阅读问题：把小王座位同时用覆盖主键和追加日记两种表示，哪一种会在「现在坐哪」上只回新值？用 `overwrite_returns_new_seat` 和 `append_keeps_both_seats` 回答。

下一课要碰权重了。外挂能改日记，改不了模型已经背进参数里的「法国的首都是巴黎」这种事实；硬改这一条又可能把邻居事实带偏。[第 14 课](14_knowledge_editing.md) 用 ROME / MEMIT 做定位-改写，并用可靠性、泛化、局部性、流畅性四项一起验收。第 16 课再回答：什么时候必须改权重。
