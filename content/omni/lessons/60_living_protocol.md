---
id: 60_living_protocol
title: "新论文收编卡"
summary: "课会过时之后，一篇新论文怎样填收编卡、接到已有课桶，并且在缺 N、缺套件、把 LIBERO 写成真机时被拒收？"
unit: domain-research
play_tools: []
checkpoints:
  - "给一篇新论文的每一行数字填收编卡：课桶、规模或机制、第 47 课类标签、第 31 课套件、N、单位、是否 fine-tune、缩小版能否复现方向"
  - "拒绝缺 N、缺套件的 LIBERO 行，并拒绝把 LIBERO 宏平均写成真机能力"
  - "把规模声明接到已有课当对照，不新开模型课；只对机制声明判断缩小版能否复现同方向趋势"
  - "在虚构新 VLA 表上先预测哪几张卡会标红，再拖桶，并让 CPU 夹具与第 31、47 课标签保持兼容"
---

# 第 60 课：把新论文接到可执行的验收口径

> 内容：新论文收编卡、规模与机制分流、与第 01 / 31 / 47 课标签兼容，不引入新模型<br>
> 建议周期：阅读约 70 分钟；浏览器收编实验约 10 分钟；CPU 必填字段夹具数分钟<br>
> 硬件：无 GPU 可完成本课阅读、教学模拟与 CPU 机制实验。对照论文规模评测需要对应基准的推理资源<br>
> 产物：收编卡模板、虚构新 VLA 表分桶记录、必填字段夹具

## 1. 新论文怎样接到这门课的验收口径上

课程有截止日期，论文没有。OpenVLA 的 7B 权重是 2024 年 6 月公开的，OpenVLA-OFT 把同一张 LIBERO 表从 76.5% 改到 97.1% 是 2025 年 2 月的事。中间这几个月里，若每来一篇新骨干就新开一课，目录会变成模型名录，验收条件会跟着标题漂移。活的协议的意思是：表值可以换，键名不能换。2026 年再来一张 13B 的离散动作表，仍然走第 27 课的动作表示和第 31 课的套件拆桶，不走「第 61 课：更新的 VLA」。活，活在字段上，不活在模型名上。[第 01 课](01_baseline_reproduction.md) 已经规定：官方评测不能写成自训复现。[第 31 课](31_vla_evaluation.md) 已经规定：LIBERO 四套件、SIMPLER 的 Pearson $r$、CALVIN 长程链和真机小样本不能横着减。[第 47 课](47_eval_taxonomy.md) 已经规定：六类评测数字互斥，LIBERO 平均进不了真机能力。本课不增加第七类，不增加新的动作头，不增加新的损失函数。本课只增加一张卡，用来回答四个问题：这篇论文进哪一课的桶；它报的是规模还是机制；课程里的缩小版能不能复现方向；缺哪条协议字段。

一张新表通常长这样。标题写 NovaVLA-8B（本课 Lab 里的这个名字是教学虚构，禁止当文献引用）。第一行写「真机成功率 81%」，脚注却是 LIBERO 四套件平均。第二行写 Spatial 88%，并且给了 $N=500$。第三行写「Long 高」，没有试验次数。第四行写「LIBERO 90%」，没有套件名。第五行写 VisMatch $r=0.81$。第六行写「13B 比 8B 高 4 个点」，配方仍是离散动作 token。六行可以印在同一页上，它们对课程的合法操作完全不同：有的进[第 31 课](31_vla_evaluation.md) 的 C5 仿真操作桶，有的进 C6 排序相关桶，有的只能拒收，有的是规模声明、接到已有的自回归 VLA 课当对照。把六行兑成「新模型 85%」再新开一课，等于同时违反 01、31、47 三课已经写下的禁则。

类比到此结束。类比失效处：图书馆给新书贴索书号，不会因为封面更大就新盖一座楼。本课的索书号是收编卡。卡上的类标签必须能被第 47 课的六类函数认出来；卡上的套件键必须能被第 31 课的 Spatial / Object / Goal / Long 认出来；卡上的 $N$ 若小于等于 25 且单位是成功率，必须能接上第 31 课已经手算过的 Wilson 区间。认不出来的字段留空，空字段的卡不得入账。浏览器 Lab 把虚构表的六行拖进桶，缺 $N$、缺套件、把 LIBERO 写成真机必须标红。CPU 实验证明必填字段、拒收规则和第 31、47 课标签兼容。夹具数字禁止写入模型卡。请把这一点写在实验记录的第一页。

本课术语：

| 术语 | 简要解释 |
|---|---|
| 收编卡 | 一篇论文里一行数字对应的记录：课桶、规模或机制、类标签、套件、$N$、单位、能否复现方向 |
| 课桶 | 接到哪一课已有的验收口径，例如 01 / 27 / 31 / 47，不新开模型课 |
| 规模声明 | 参数量、数据量、卡时变了，机制没变；缩小版复现不了表上的绝对数字 |
| 机制声明 | 可命名的结构或损失改动；缩小版至少能复现同方向趋势 |
| 可复现方向 | 论文差分与夹具差分同号，不要求幅度相等 |
| 必填字段 | 空值则 `reject_incomplete` 的键；C5 / LIBERO 行还要套件与成功谓词 |
| 标红 | 缺 $N$、缺套件、LIBERO 声称真机时，卡与桶必须显示拒绝 |
| 准入 | `admit` / `reject_incomplete` / `reject_illegal` 三值，不是成功率 |

术语表只服务本课。前面课已经解释过的词直接链回去：trace、golden case、loss mask 见[第 01 课](01_baseline_reproduction.md)；套件、谓词、Wilson、Visual Matching 见[第 31 课](31_vla_evaluation.md)；六类标签、协议卡、真机陷阱见[第 47 课](47_eval_taxonomy.md)；离散动作 token 见[第 25 课](25_action_tokenization.md) 与[第 27 课](27_autoregressive_vla.md)；数据混合物见[第 26 课](26_robot_data_mixture.md)。本课若再把这些词展开一遍，会和前三课抢篇幅。需要时点链接。本课只把它们接到收编决策上。

## 2. 本课解决的问题

当前系统（以及你读到的任何「新 SOTA 总表」）默认验收是「换一个模型名，报一个更高的百分数」。它解决不了六类失败，而且这六类会在「再开一课」里被兑掉：

1. 把缺 $N$ 的「Long 高」写进和 500 trials 同一列。
2. 把缺套件的「LIBERO 90%」当成可以和 OpenVLA Table 12 对照的点。
3. 把 LIBERO 四套件平均写进真机节，再声称身体智能已经过关。
4. 把 7B 对 55B 的绝对差当成 MiniMind-O 26M 也该复现的数字。
5. 把并行解码、动作块、连续 L1 这种机制，和单纯加参数量写进同一张「架构创新」幻灯片。
6. 为每一篇新 VLA 新开一课，让第 27、28、31 课的验收条件作废。

本课的改造范围只包括收编卡字段、准入函数和标红规则。不更换视觉编码器，不重做[第 02 课](02_multimodal_connector.md) 的 connector，不比较离散动作 token 与 flow matching（那是第 25、27、28 课），不把屏幕点击当成机械臂（那是[第 32 课](32_gui_agent_grounding.md)），不重训 MiniMind-O。

以下结果不能支持「新论文已经被这门课消化」或「已经具备真机能力」：

- 收编卡缺 `n` 或 LIBERO 行缺 `suite`，却写了 `admit`；
- 把 LIBERO 宏平均的 `claimed_class` 写成 `real_robot`；
- 把 SIMPLER 的 $r$ 填进成功率格；
- 把规模声明标成 `reducible=true`，并声称缩小版复现了 97.1%；
- 为 13B 对 8B、同一套离散 token 的对照新开模型课；
- 把 Lab 虚构表的 81% 和 OpenVLA Table 12 的 76.5% 写进同一列当复现误差。

执行顺序固定：先在 CPU 夹具上证明必填字段、缺字段拒收、LIBERO 当真机拒收、六类与四套件键兼容，再在浏览器里亲眼看到三张卡标红，最后如果有 GPU，才把同一套收编卡接到真实论文表上。没有 GPU 时，前两步已经构成完整交付。

和前三课的边界再钉死一次。第 01 课管一次训练是否可追溯：commit、数据哈希、golden case、trace 不改 logits。本课接管「别人的表」能不能进你的笔记，不重做过拟合 128 条。第 31 课管执行侧内部怎么拆桶、怎么给区间。本课接管这些字段在新论文到来时是否还被填写。第 47 课管六类互斥。本课禁止新论文用第七个百分数绕过六类。三课的检查票要一起用：先问字段齐不齐，再问类对不对，再问套件和 $N$，再问规模还是机制，最后问缩小版复现的是方向还是数字。

本课固定四个可证伪命题：

1. 存在一张必填字段表，缺 $N$ 或 LIBERO 行缺套件时，准入结果必须是 `reject_incomplete`。
2. 把 LIBERO 宏平均的声称类别写成 `real_robot`，必须被拒绝。
3. 第 47 课六条公开记录经过本课标签函数后仍互斥；第 31 课四个套件键仍被接受；$N=25$、$k=20$ 的 Wilson 区间仍约为 $[0.609,0.911]$。
4. 规模声明不得把 `opens_new_model_lesson` 置真；机制声明才允许 `reducible=true`。

四条都通过，只能说明收编被写成了可审计分类。它们不能说明 NovaVLA 存在，也不能说明 97.1% 被你的 26M 模型复现。可审计分类的对立面是可宣传分类：后者只要求数字变大、名字变新。本课的 CPU 夹具对宣传分类是故意不友好的：它没有「综合分」键，没有「新课申请」键，有的是缺失列表和非法声称。谁给夹具加这两个键，谁就把本课改回第 47 课已经拒绝过的总平均，外加第 27 课已经覆盖过的模型名录。请把「不加这两个键」写进代码评审清单。评审清单和收编卡一样，是给下一个人用的接口，不是给自己看的备忘。接口稳定，表值才允许换。这一句写进实验记录第一页。

## 3. 开始前需要准备什么

本课没有 MiniMind-O 训练步骤。开始前把上游事实和本课约定分开写进实验记录。

**上游事实（打开过的页面，不是口口相传）：**

- OpenVLA：[arXiv:2406.09246](https://arxiv.org/abs/2406.09246)，HTML：[v3](https://arxiv.org/html/2406.09246v3)。7B，970k 真机示范，Llama 2 + SigLIP/DINOv2。Appendix E Table 12：Spatial 84.7%、Object 88.4%、Goal 79.2%、Long 53.7%，宏平均 76.5%，各套件独立 LoRA，500 trials × 3 seeds。摘要：相对 RT-2-X（55B）在 29 项任务上绝对成功率高 16.5%；WidowX 17×10、Google Robot 12×5。
- OpenVLA-OFT：[arXiv:2502.19645](https://arxiv.org/abs/2502.19645)。并行解码、动作块、连续动作、L1。LIBERO 宏平均 76.5% 到 97.1%，动作生成吞吐 26 倍。真机 ALOHA 相对默认配方的 $\pi_0$、RDT-1B 以及从头训的 Diffusion Policy / ACT，平均成功率最多高 15 个绝对点。
- SIMPLER：[arXiv:2405.05941](https://arxiv.org/abs/2405.05941)。Google Robot Visual Matching 平均 Pearson $r=0.924$。沿用第 31 课已核对的 Table I。
- LIBERO：[arXiv:2306.03310](https://arxiv.org/abs/2306.03310)。四套件构造沿用第 31 课 §5.2，本课不重跑仿真。
- CALVIN：[arXiv:2112.03227](https://arxiv.org/abs/2112.03227)。短程 53.9%、五步 0.08%，沿用第 31 课 Fig. 8。
- Qwen2.5-Omni：[arXiv:2503.20215](https://arxiv.org/abs/2503.20215)。评测按输入输出模态分节，第 01、47 课已引。本课用它当「分节做对了、读者仍可能兑在一起」的对照，不讲 Thinker–Talker。
- CLIP：[arXiv:2103.00020](https://arxiv.org/abs/2103.00020)；LLaVA：[arXiv:2304.08485](https://arxiv.org/abs/2304.08485)。本课只借它们区分规模与机制，不重做对比损失或两阶段解冻。

**本课约定：**

- CPU 实验文件：`experiments/src/learn_omni_experiments/lessons/lesson_60.py`。编排者登记进 `registry.py` 之前，可以直接导入该模块跑 `run()`；登记之后用仓库脚本 `python3 run.py run 60`。登记前不要改共享注册表，本课文件可以独立阅读、独立调用。
- 浏览器 Lab：`Lesson60ProtocolLab`。标有「教学模拟」。六行来自虚构的 NovaVLA-8B 表，数字 81%、90%、+4 点都不是公开文献。拖动与点选两种操作等价。揭晓前不显示对错。
- 不把 Lab 的拖桶结果、CPU 的 0.765、论文表里的 97.1% 写进同一列当「本课复现结果」。三列来源不同：夹具、教学分类、他人论文。
- 引用任何公开百分数时必须出现课桶、类标签、规模或机制。缺三项中任一项就删掉该数字。
- 产出目录约定为 `artifacts/lesson60/`。编排者注册后，结果 JSON 会写到实验目录；注册前把 `run()` 返回值贴进实验记录即可。

需要会的前置技能：第 01 课「分桶再平均」和「官方评测 ≠ 自训」；第 31 课 Wilson 公式和套件键；第 47 课六类互斥。不要求会训 7B，不要求有机械臂。统计直觉停留在第 31 课已经手算过的那一层：$N=25$、$k=20$ 时 Wilson 宽度约 0.30。本课不再推导区间，只规定它在收编卡上的触发条件：单位是 `success_rate`，类是 C5 或真机，且 $N$ 已填写。不会 Wilson 的人可以先回第 31 课第 5.7 节。会 Wilson 的人不要把它套到 Pearson $r$ 上，也不要套到缺 $N$ 的行上。

建议花在各节上的时间：第 1–4 节约 40 分钟建立口径；第 5 节机制约 3 小时，OpenVLA / OFT / SIMPLER 各读原文表号；第 6 节公开实现约 30 分钟；第 7–12 节边跑边写收编卡约半天；第 13–14 节按「带着什么问题读」打勾。没有 GPU 的人到第 12 节就可以停，第 13 节的改造实验标 `deferred`。时间不够时优先顺序是：术语表、必填字段、三条标红、规模与机制、Lab、CPU。论文精读可以后补，但补的时候必须带着「这行进哪一课」这个问题，不要改成摘抄排行榜。时间够的人把第 5.19 节的十分钟操作对 OpenVLA 与 OFT 各做一遍，写下四张草稿里哪些仍缺 $N$。缺的那些草稿留在案卷，不要为了交卷而编一个 trials 数。编出来的 $N$ 比空着更糟：空着会拒收，编出来会进主表。

硬件分层写进记录。读课文和 CPU：笔记本即可。跑 Lab：现代浏览器。复现 LIBERO 500 trials：能加载 OpenVLA 7B 的单卡。真机：对应机械臂。没有真机就不要填真机格子。本课主路径明确不要求这些复现。有资源的人把真实评测当成第 13 节改造实验，没有资源的人把收编卡空行留着。空行表示范围声明已经写下。把空行填上别人的表值却不写对照来源，才算交付失败。主路径交付只需要课文、Lab、CPU 三件套，三件套里没有任何一项依赖 GPU。有 GPU 也不许跳过三件套直接去跑 7B：先拒收，再推理。顺序反了，空 $N$ 会在推理之后被「补」出来，第 17 课禁止的事后改公式会在收编里重演。

## 4. 完成后应具备的能力

完成后，拿到任意一篇 2024 年以后的 VLA 或 Omni 论文，应能在不新开课的前提下完成以下检查。检查对象是卡，不是模型昵称。昵称只出现在 `paper_id` 旁的备注里。

1. 把总表拆成「一行一卡」，而不是「一篇一课」；
2. 写出收编卡十个必填字段，LIBERO / C5 行再补套件与成功谓词；
3. 指出缺 $N$、缺套件的行必须 `reject_incomplete`，并在 Lab 里看见标红；
4. 拒绝把 LIBERO 宏平均写成真机能力，并能指出 OpenVLA Table 12 的 76.5% 是独立 fine-tune；
5. 给每一行打上 `scale` 或 `mechanism`，规模行不得新开模型课；
6. 只对机制行判断缩小版能否复现方向，并且不把方向等同于复现 97.1%；
7. 沿用第 47 课：MMMU 与 LIBERO 不能比；沿用第 31 课：$N=25$、$k=20$ 写出 Wilson 约 $[0.609,0.911]$；
8. 沿用第 01 课：论文表值是他人评测，不能写入 `baseline-v1` 的复现误差；
9. 在 Lab 里先预测「缺 $N$、缺套件、LIBERO 写成真机三张都会标红」，再拖桶；
10. 引用任何公开百分数时，能在 30 秒内指出表号、课桶、类标签、规模或机制。指不出就从讲稿里删掉该数。

五分钟审稿可以按这个口令做。打开一篇新 VLA 的总表，从左到右给每一列贴便利贴：课桶、C1–C6 或真机、规模或机制、$N$、套件。出现空 $N$，便利贴改红。出现 LIBERO 却没有 Spatial / Object / Goal / Long，便利贴改红。出现 LIBERO 和真机相邻且正文有「因此」，便利贴改红。出现「我们提出 XYZ-7B」却没有任何可命名的结构改动，便利贴写「规模，接到 27」。全部便利贴都有类、红色为零，这篇表可以进笔记；否则只许引用分列后的单格，不许引用作者的总平均句。口令短，是为了在组会当场用，不是为了代替收编卡。收编卡在会后补。

完成能力第 1 到 4 条可以在纸上验收：CPU 实验的 `missing_n_rejected`、`missing_suite_rejected` 与 `libero_as_real_rejected` 必须为真。第 5 到 6 条口头能讲清规模与机制。第 7 到 8 条能指到前三课的原句。第 9 条在教学桶上验收。漏掉第 3 条（缺字段拒收）或第 4 条（LIBERO 不当真机）等于本课没上。

## 5. 原理：边造边讲

下面这些机制按同一节奏写：为什么需要、怎么运转、精确定义、在公开实现或夹具里落在哪、怎么证明做对了。本课没有 MiniMind-O 源码可改，代码落点改到收编卡字典、准入函数，以及各论文自己的评测脚本标题。

### 5.1 课会过时，验收口径是稳定接口

为什么需要。课程目录是有限集合。VLA 论文标题是无限集合。若接口是「模型名字」，每来一篇就要改目录、改 Lab、改 CPU。若接口是「字段 + 类标签 + 课桶」，新论文只增加行，不增加课。

怎么运转。把课程看成一台只接受特定 schema 的评测机。输入是一行数字加一组字段，输出是三值准入。模型结构可以变，schema 不能变。第 27 课已经收下离散动作 token 的自回归 VLA；第 28 课已经收下 flow matching；第 31 课已经收下套件与区间；第 47 课已经收下六类。后来的论文若仍是「7B 视觉语言骨干 + 256 bin 动作」，进 27 当规模对照。若改的是并行解码和动作块，进 25 / 27 / 31 当机制对照。若改的是评测分节，进 47。没有第四条路叫「因为名字新，所以新开课」。

数学。令课程课桶集合为 $B=\{01,21,\ldots,48\}$（本波已上线的编号以仓库为准）。令新论文行 $m$ 的课桶为 $b(m)\in B\cup\{\texttt{unspecified}\}$。本课要求 $b(m)\in B$。$b(m)=\texttt{unspecified}$ 时不得 `admit`。本课明确禁止 $b(m)=\texttt{new\_model\_lesson}$。

验证。CPU 夹具里所有教学卡的 `opens_new_model_lesson` 均为假。Lab 的规模行正确桶是「规模声明」，不是「真机能力」，也不是新课。

稳定接口还有一层时间含义。2024 年 6 月的 OpenVLA 和 2025 年 2 月的 OFT 隔了八个月。八个月足够让一组同学从第 21 课学到第 31 课。若第 27 课的验收条件写成「复现 OpenVLA 的 76.5%」，OFT 一出，这句验收就假了。若验收条件写成「LIBERO 四套件分列、写 fine-tune、拒绝当真机」，OFT 的 97.1% 仍然能进同一张卡，只是 `value` 和 `claim_kind` 更新。课过时的是表值，不过时的是字段。本课把这句话写成规则：`value` 可以换，`class_id` / `suite` / `n` 的键名不能换。

### 5.2 收编卡：一篇论文一行数字必须带齐的字段

为什么需要。第 01 课已经要求指标追溯到 case 和生成参数。第 31 课把这条放大成机器人协议卡。第 47 课把协议卡推广到六类。本课再加三列：课桶、规模或机制、缩小版能否复现方向。缺这三列，你只知道「这是一个百分数」，不知道它该进哪一课、该不该让 26M 模型去追。

怎么运转。一张卡至少包含下面十个键。C5 或基准名为 LIBERO 的行，再加 `suite` 与 `success_predicate`。

| 键 | 含义 | 空值后果 |
|---|---|---|
| `paper_id` | arXiv 或稳定短名 | 无法追溯出处 |
| `lesson_bucket` | 接到哪一课 | 无法决定读哪篇课文 |
| `claim_kind` | `scale` 或 `mechanism` | 无法决定缩小版要不要追数字 |
| `class_id` | C1–C6 或合法的 `real_robot` | 无法决定能不能比 |
| `benchmark` | 基准名 | 无法核表 |
| `split` | test / val / 套件 / 视觉协议 | val 和 test 会被兑 |
| `n` | 题量或 trials | 不能写区间，本课直接拒收 |
| `unit` | accuracy / success_rate / pearson_r | $r$ 会被乘 100 |
| `fine_tune` | true / false / 未公开 | 76.5% 会被当成出厂 |
| `reducible` | 缩小版能否复现方向 | 规模会被当成机制 |

数学。把一行公开数字写成五元组 $(p,\mathcal{P},u,b,\kappa)$。报告估计的是 $p(\pi,\mathcal{P})$ 在单位 $u$ 下的值，课桶是 $b$，种类是 $\kappa\in\{\mathrm{scale},\mathrm{mechanism}\}$。缺少 $\mathcal{P}$ 的核心坐标（对 LIBERO 是套件，对所有成功率是 $N$）时，该格只能标 `unspecified`，不能进对照表。

代码落点。CPU 实验把十个键写成 `REQUIRED_FIELDS`，把套件与谓词写成 `C5_EXTRA_FIELDS`。`missing_fields` 在 `n is None` 或 `n==0` 时把 `n` 记入缺失列表。`admission` 在缺失非空时返回 `reject_incomplete`。

验证。虚构行 `nova_missing_n` 的缺失列表含 `n`；`nova_missing_suite` 的缺失列表含 `suite`。两条的准入都不是 `admit`。OpenVLA Spatial 行十个键加套件都填了，准入是 `admit`。

十个键的填写顺序有规定，避免对着摘要第一句倒填。先写 `paper_id` 和表号，再写 `benchmark` 与 `split`，再写 `n` 与 `unit`，再写 `fine_tune`，再写 `class_id`，LIBERO 再写 `suite`，最后才写 `lesson_bucket`、`claim_kind`、`reducible`。后三列依赖前七列：没有单位就不能谈机制，没有类标签就不能选课桶。倒序填写是组会上最常见的失败：先认定「这是第 27 课的新模型」，再回去找数字，数字不够就留空。本课要求空数字的卡在后三列写之前就被拒收，倒序填不下去。

### 5.3 一行一卡：一篇论文拆成多张卡

为什么需要。OpenVLA 一篇论文至少做了四件不同的事：提出 7B 离散动作 VLA；在 Open X-Embodiment 970k 轨迹上预训练；在 LIBERO 四套件上独立 fine-tune；在 WidowX 和 Google Robot 上做出厂真机评测。一件事一张卡。一篇一课会把 970k 数据混合物和 76.5% 仿真平均兑成「OpenVLA 很强」。

怎么运转。按测量对象切，不按作者切。

| 卡 | 进哪一课 | 种类 | 类标签 | 备注 |
|---|---|---|---|---|
| 256 bin、覆盖 Llama 词表最后 256 个 token | 25 / 27 | 机制 | 不适用评测六类 | 动作表示 |
| 970k OpenX 混合物、滤 DROID | 26 | 规模加配比 | 不适用 | 数据 |
| LIBERO Spatial 84.7% | 31 / 47 | 机制（评测拆桶） | C5，suite=spatial | 仿真 |
| 相对 RT-2-X +16.5%，29 任务 | 31 | 规模 | `real_robot`，$N=230$ 量级 | 真机出厂 |
| LoRA 可在消费级 GPU fine-tune | 18 / 27 | 机制 | 不适用 | 适配 |

数学。一篇论文 $P$ 对应卡集合 $\{m_1,\ldots,m_k\}$，$k\ge 1$。准入是对 $m_i$ 逐张做的，不对 $P$ 做一次总录取。存在 $i$ 使 $\mathrm{Admit}(m_i)=\mathrm{admit}$，并不蕴含 $\mathrm{Admit}(m_j)=\mathrm{admit}$。

验证。CPU 夹具同时放了 OpenVLA 的 Spatial 行（admit）和虚构的「LIBERO 当真机」行（reject_illegal）。两张卡可以属于同一 `paper_id` 前缀，决策必须分开。Lab 的六行共享虚构报告名，六行的正确桶仍有三种：C5、C6、缺字段、规模。

切卡时用「若删掉这一段，百分数还在不在」当刀。删掉 Appendix E，76.5% 消失，所以 Table 12 是独立卡。删掉第 5.1 节真机任务表，+16.5% 消失，所以真机是独立卡。删掉词表覆盖段，LIBERO 数字还在，所以动作表示和仿真成功率不是一张卡。刀法不依赖作者是否用了小节标题。作者把所有数字放进 Table 1 的同一行，你仍然要拆。Lab 的虚构 Table 2 就是这种「一行混多件事」的反面教材：行 A 把仿真平均写成真机，必须拆开再拒收真机声称。

### 5.4 课桶：接到已有课，不新开模型课

为什么需要。第 24 课已经讲 PaLM-E / RT-1 / RT-2 / Gato / SayCan。第 27 课已经讲自回归 VLA。第 28 课已经讲流匹配。新来一篇「也是 7B、也是离散 bin、也是 OpenX」，若再开「第 61 课：某某VLA」，读者会以为动作表示变了。其实变的是数据配比或评测数字。

怎么运转。用一张路由表，按机制关键词而不是按模型名。

| 你在新论文里看见 | 先接到 | 不要接到 |
|---|---|---|
| InfoNCE / sigmoid 对比 | 21 | 新对比学习课 |
| 冻 ViT、训投影、再解冻 | 22 | 新 VLM 配方课 |
| 离散动作 token、分箱 | 25 / 27 | 新骨干课 |
| 流匹配 / 扩散动作 | 28 | 新「生成式 VLA」课 |
| 快慢双系统 | 29 | 新规划课 |
| LIBERO 四套件成功率 | 31 / 47 C5 | 真机节 |
| SIMPLER $r$ / MMRV | 31 / 47 C6 | 成功率格 |
| MMMU / Video-MME / OmniBench | 47 C1–C3 | 操作节 |
| OSWorld | 47 C4 | 机械臂节 |
| 官方评测冒充自训 | 01 | 任何「我们复现了 SOTA」 |

数学。路由是函数 $r:\text{机制关键词}\to B$。$r$ 必须是单值的：同一关键词不能同时指向「已有课」和「新课」。规模声明的 $r$ 落在已有课上，作为对照行，不作为新课申请。

验证。CPU 里 `nova_scale_13b` 的 `lesson_bucket` 是 `"27"`，`opens_new_model_lesson` 为假，准入为 `admit`。它的 `reducible` 为假：26M 追不上 13B 对 8B 的 4 个点。

已核实文献表里还有几篇，名字新、课桶旧，写在这里以免组会上当场发明第 61 课。[RT-1](https://arxiv.org/abs/2212.06817) 与 [RT-2](https://arxiv.org/abs/2307.15818) 进[第 24 课](24_vlm_to_vla.md)：一个是 token 化动作加高效 Transformer，一个是把动作塞进 PaLI 词表。不要因为后来的 OpenVLA 引用了它们，就为 RT 系列再开一课。[Octo](https://arxiv.org/abs/2405.12213) 进[第 26 课](26_robot_data_mixture.md) 的异构混合物对照，政策骨架若被拿来和 OpenVLA 比 LIBERO，数字仍走第 31 课 C5，两张卡。[$\pi_0$](https://arxiv.org/abs/2410.24164) 的流匹配动作进[第 28 课](28_flow_matching_vla.md)；OFT Table I 里 $\pi_0$ fine-tune 平均 94.2%、Long 85.2% 仍是第 31 课的 C5 行，观察含腕部相机必须写进 `split`。[RDT-1B](https://arxiv.org/abs/2410.07864) 若作为 OFT 真机对照出现，课桶是 31 的真机行，不是新扩散课；若你要讲它的动作头，先打开原文确认是不是流匹配或扩散，再决定 28 还是 25，本课不替你发明结构。[GR00T N1](https://arxiv.org/abs/2503.14734) 若自称快慢双系统，接到[第 29 课](29_dual_system_vla.md)，人形身体的动作维另走第 37 课规格，评测数字仍按 31 / 47 填卡。以上每一篇都可以在目录一个字不改的前提下被收编。目录要改的唯一理由是：出现了现有课桶表达不了的新机制，并且缩小版夹具也造不出同方向的自变量。那种情况属于以后的扩课提案，不是本课 Lab 的通过条件。本课的通过条件相反：你能把新表接到旧桶。

路由表要当着论文用，不要凭记忆。打开 OFT，看见 parallel decoding、action chunking、continuous L1，关键词落在第 25 课（连续表示）和第 30 课（动作分块）和第 31 课（同一张 LIBERO 表）。三张卡，三个课桶，可以同时 `admit`。打开一篇只把隐藏层从 4096 加到 5120、动作头一字未改的报告，关键词是空的，课桶只能填 27，`claim_kind=scale`。空关键词却填 28，属于把规模伪装成流匹配，CPU 夹具不查这句话，组会便利贴要查。本课允许课桶填已有编号，不允许填「新课：更大的 27」。更大的 27 仍是 27。

### 5.5 规模还是机制

为什么需要。公开数字上涨有两种完全不同的原因。一种是钱：更多参数、更多轨迹、更多卡时。一种是可命名的结构或损失：对比损失从 softmax 改成 sigmoid，动作从串行 token 改成动作块，评测从宏平均改成套件分列。缩小版的 CPU 实验只对第二种承诺方向。对第一种，本课只承诺「记下数字，不追」。

怎么运转。用能否在现有课的夹具里改一行代码来判断。能改到、且改完后符号可预期，标 `mechanism`。只能改模型大小或数据倍数，标 `scale`。

| 例子 | 种类 | 缩小版能做什么 | 缩小版不能做什么 |
|---|---|---|---|
| CLIP 的 InfoNCE 与温度 | 机制 | 第 21 课 4×4 矩阵手算 | 复现 WIT 4 亿对 |
| LLaVA 两阶段解冻 | 机制 | 第 22 课可训练参数计数 | 复现 158K 指令微调分数 |
| OpenVLA 256 bin 覆盖词表 | 机制 | 第 25 / 27 课词表偏移 | 复现 970k 预训练 |
| OpenVLA 7B 对 RT-2-X 55B，+16.5% | 规模 | 记下 $N$ 与机体 | 用 26M 追 16.5 个点 |
| OFT：并行解码 + 动作块 + L1，76.5% 到 97.1% | 机制 | Long 被动作块抬起来的方向 | 复现 97.1% 与 26× 吞吐 |
| 虚构 13B 对 8B，同一离散 token，+4 点 | 规模 | 接到 27 当对照 | 新开课、标可复现 |

数学。种类 $\kappa(m)$ 是离散标签，不是连续的「创新程度」。禁止写 $\kappa=0.7$。可复现方向只在 $\kappa=\mathrm{mechanism}$ 时有定义：

$$
\rho(m)
=
\mathbf{1}[\kappa(m)=\mathrm{mechanism}]
\cdot
\mathbf{1}\bigl[\mathrm{sign}(\Delta_{\mathrm{paper}})=\mathrm{sign}(\Delta_{\mathrm{toy}})\bigr]
$$

若 $\kappa=\mathrm{scale}$，强制 $\rho(m)=0$。CPU 检查 `reducible_implies_mechanism`：`reducible=true` 的卡必须 `claim_kind=mechanism`。

验证。OFT 卡 `claim_kind=mechanism` 且 `reducible=true`。OpenVLA 对 RT-2-X 的真机差 `claim_kind=scale` 且 `reducible=false`。虚构 13B 行同样。三条同时成立，规模和机制才没有被兑在 `reducible` 这一个布尔里。

判断时不要被摘要里的「we propose」带跑。OpenVLA 摘要同时写了新开源权重和 +16.5%。前半句是工程交付，课桶可以是 27 的对照实现；后半句是真机小样本规模比较，课桶是 31 的真机行，$\rho=0$。同一句里可以有两种 $\kappa$，所以必须一行一卡。OFT 摘要写 Optimized Fine-Tuning recipe，并把 76.5% 到 97.1% 和 26× 吞吐放在一起。吞吐是延迟账，第 01 课的 TTFA 纪律管它，本课不把 26× 写进成功率格；97.1% 是 C5 机制卡。两个数字两张卡，两张都可以 `mechanism`，单位不同，仍不能减。

### 5.6 缩小版能不能复现方向

为什么需要。本课程的主路径是 26M MiniMind-O 加一组 CPU 夹具。公开 VLA 是 7B 级、真机 970k 轨迹。若「复现」指绝对值，本课所有 VLA 数字都会失败，收编规则会变成「全部拒收」。若「复现」指方向，第 31 课已经能复现「宏平均掩盖 Long」：76.5% 对 Long 53.7%，差超过 30 个点。OFT 把 Long 从 53.7% 做到 94.5%，方向是「动作块救长程」。缩小版可以造一条短程成功、长程失败的轨迹，再给一个玩具动作块把长程抬起来。抬起的幅度不必是 40.8 个点。

怎么运转。方向复现写三条，缺一条就标 $\rho=0$：

1. 自变量叫得出名字（套件、动作块长度、温度、fine-tune 开关）。
2. 因变量的符号与论文一致。
3. 绝对值误差允许任意大，但单位必须相同。

数学。令论文观测 $\Delta_{\mathrm{paper}}=\hat p_{\mathrm{after}}-\hat p_{\mathrm{before}}$，夹具观测 $\Delta_{\mathrm{toy}}$。方向成立当且仅当 $\Delta_{\mathrm{paper}}\Delta_{\mathrm{toy}}>0$。$\Delta_{\mathrm{paper}}=0$ 时本课不讨论方向。单位不同时乘积无定义，例如不能用夹具成功率去跟 SIMPLER 的 $r$ 比符号。

验证。CPU 用第 31 课同一组四套件点：宏平均 0.765，Spatial 减 Long 大于 0.3。这是「宏平均掩盖 Long」的方向，已被第 31、47 课检查过，本课再跑一遍，证明收编没有改掉旧夹具。OFT 的 97.1% 本身不出现在「必须相等」的检查里，只出现在 `value` 字段，证明它是他人表值。

方向复现还有一条否定句，必须写进卡脚。缩小版没有腕部相机，就不能声称复现了 OFT Table I 里「第三视角加腕部」那一行的增量。缩小版没有 500 trials，就不能声称复现了区间宽度。缩小版没有 ALOHA 双臂，就不能声称复现了「最多高 15 个绝对点」。三条否定句都成立，`reducible=true` 仍然可以成立：它只绑定「Long 相对 Spatial 更难」或「动作块抬长程」这种符号。把否定句删掉，方向会在口头报告里膨胀成绝对值。第 01 课禁止用 mini 去减 full；本课禁止用玩具去减 97.1%。两句是同一条复现纪律在不同尺度上的实例。

符号检查要写进实验记录的算式，不要只写「方向对了」。OpenVLA Table 12：$\Delta_{\mathrm{paper}}=0.537-0.847=-0.310$，CPU 夹具同一组点给出同样的差。符号为负，含义是 Long 低于 Spatial。OFT 把 Long 从 0.537 拉到 0.945，$\Delta_{\mathrm{paper}}=+0.408$，含义是动作块与连续 L1 抬长程。缩小版若只能复现第一条负号、复现不了第二条正号，应把 OFT 卡的 `reducible` 改成 false，或把自变量改成「动作块长度」，直到玩具里长程成功率随块长度上升。上升多少不重要，正号必须出现。若玩具里块长度增加、长程反而下降，论文方向未被复现，不得把夹具绿勾抄进 OFT 那一行。绿勾只证明你的玩具在跑，不证明你复现了 OFT。

### 5.7 与第 01 课复现纪律对接

为什么需要。第 01 课的读者最容易把本课收编卡理解成「又一个可以跑的训练」。收编卡是笔记格式，不是 recipe。`paper_id=2406.09246` 不等于你训出了 OpenVLA。`value=0.765` 不等于 `baseline-v1` 的评测输出。

怎么运转。三条禁则从第 01 课原样搬来，只改主语：

1. 他人评测与自训复现分列。OpenVLA Table 12 进「文献」列；你的 MiniMind-O golden case 进「自训」列。
2. 官方 full 与课程 mini 分列。不得把 76.5% 减 mini 的某次抓取玩具分，写成复现误差。
3. 指标必须能追溯。收编卡的 `paper_id` + 表号 + `split` 就是第 01 课要求的 provenance。缺这三项，等于第 01 课缺 commit SHA。

数学。第 01 课的 loss mask 集合 $M_t$ 只含 assistant target。本课的入账集合 $M_{\mathrm{card}}$ 只含字段齐全且合法的行。两句话同构：不该算账的位置，label 是 `-100` 或 `reject_*`，不能靠平均值把它们洗白。

验证。CPU 不把 0.765 标成「本课测得」。`metrics["libero_macro"]` 的注释在 summary 里写明是 Table 12 夹具常量。Lab 的 81% 在描述里写明虚构。两条来源在报告里必须分列。

第 01 课还有 golden case：100 个冻结样例，改结构之后回归。收编卡的回归对象是字段，不是样例。本课建议冻结十张卡当 golden protocol：OpenVLA Spatial、Object、Goal、Long、宏平均、SIMPLER VisMatch $r$、OFT 宏平均、OpenVLA 真机 +16.5%、MMMU GPT-4V test 55.7%、OSWorld GPT-4 12.24%。十张卡的键名以后不许改。`value` 可以随你引用的版本更新，但更新时必须改 `paper_id` 的版本后缀，例如 `2406.09246v3`。这和第 01 课「不用会移动的 master 充当版本号」是同一句。谁把 golden protocol 的 `class_id` 从 C5 改成 `general`，谁就破坏了第 47 课的互斥，也破坏了本课的收编。

### 5.8 与第 31 课标签兼容：套件、$N$、谓词

为什么需要。第 31 课已经证明四套件宏平均会掩盖 Long，接触成功与放置成功可以在同一条轨迹上不一致，$N=25$、$k=20$ 的 Wilson 约为 $[0.609,0.911]$。本课若另造一套套件名，例如 `spatial_rel` / `long_horizon`，旧夹具全部失效，新论文会在两套名字之间漂移。

怎么运转。套件键锁死为 `spatial` / `object` / `goal` / `long`。C5 行缺这四个之一，视为缺套件。$N$ 必须是正整数。单位是 `success_rate` 时，小样本必须能接 Wilson；单位是 `pearson_r` 时，禁止接 Wilson。成功谓词至少写 `pddl_conjunction`、`contact`、`place`、`hold` 之一，或论文给出的脚本名。

数学。沿用第 31 课：

$$
\hat p=\frac{k}{N},\quad
z=1.96
$$

Wilson 中心与半宽：

$$
\tilde p=\frac{\hat p+z^{2}/(2N)}{1+z^{2}/N},\quad
m=\frac{z}{1+z^{2}/N}
\sqrt{\frac{\hat p(1-\hat p)}{N}+\frac{z^{2}}{4N^{2}}}
$$

$N=25$、$k=20$ 时 $\hat p=0.8$，区间约为 $[0.608687,0.911395]$。缺 $N$ 时公式无定义，本课在公式之前就 `reject_incomplete`，避免有人用「看起来很高」代替区间。

代码落点。`SUITES_31` 与第 31 课 `SUITE_OPENVLA_FT` 的键相同。`labels_compatible_31_47` 在 C5 行要求 `suite in SUITES_31` 且 `unit=="success_rate"`。`wilson_interval` 与第 31、47 课同一实现。

验证。`lesson31_suite_keys_accepted` 为真。`n25_wilson_matches_lesson31` 为真。缺 $N$ 的虚构行进不了这两条检查的 admit 分支。

兼容还包括「宏平均不得单独充当套件」。OpenVLA Table 12 的 76.5% 是四套件算术平均。收编时宏平均可以另建一张卡，`split=four_suite_macro_ft`，但 `suite` 键仍要填一个代表，或把宏平均卡的 `suite` 留空并因此拒收。本课夹具选择给 OFT 宏平均卡填 `suite=long`，因为作者自己强调 Long 从 53.7% 走到 94.5% 是主要贡献之一；同时 `split` 仍写 `four_suite_macro_ft`，避免读者以为 97.1% 是 Long 单点。这是教学选择，不是论文原文字段。你自己的卡如果坚持宏平均不填套件，按本课规则会被 `reject_incomplete`。那是正确的严厉：宁可比不上，不要用 97.1% 冒充 Long。第 31 课已经要求四套件分列；本课把「分列」写成必填键，而不是写成建议。

### 5.9 与第 47 课标签兼容：六类互斥

为什么需要。第 47 课已经把 MMMU、Video-MME、OmniBench、OSWorld、LIBERO、SIMPLER 打成 C1–C6。本课若允许新论文自报 `class_id=C7_general_vla`，六类互斥立刻被绕开。真机陷阱也会被绕开：作者只要把 LIBERO 平均标成 C7，就不会触发 `real_robot` 检查。

怎么运转。`CLASS_IDS` 与第 47 课逐字相同。新论文的评测行必须落入六类之一，或落入合法的 `real_robot`（基准名不是 LIBERO，单位是 `success_rate`，$N$ 已填）。LIBERO 行的 `class_id` 只能是 `C5_sim_manip`。SIMPLER 排序行只能是 `C6_sim2real_rank`，单位 `pearson_r`。

数学。沿用第 47 课：标签函数 $c$ 在六类上是单值的。差值有定义当且仅当

$$
c(m_a)=c(m_b)
\ \wedge\
\mathrm{unit}(m_a)=\mathrm{unit}(m_b)
\ \wedge\
\mathcal{P}(m_a)\simeq\mathcal{P}(m_b)
$$

本课追加：$\mathrm{Admit}(m_a)=\mathrm{Admit}(m_b)=\mathrm{admit}$。一张被拒收的卡不能参加做差。缺 $N$ 的「Long 高」即使类标签写对了，也没有 $\Delta$。

代码落点。`RECORDS_47` 六条记录从第 47 课原样迁入。`assignments_are_mutex` 要求六类双射。`may_compare_47("mmmu_gpt4v_test","libero_openvla_macro")` 为假。`illegal_libero_as_real` 在 `benchmark=="LIBERO"` 且 `claimed_class=="real_robot"` 时为真。

验证。`lesson47_labels_still_mutex` 为真。`libero_as_real_rejected` 为真。`simpler_unit_is_pearson_r` 为真。三条同时成立，本课才没有在 47 课旁边另起一套账本。

六类之外的合法真机行要单独说清，避免读者以为本课取消了真机。OpenVLA 在 WidowX 和 Google Robot 上的出厂评测是真机，基准名不是 LIBERO，$N$ 是 170 与 60 量级，单位是成功率。这样的行 `class_id=real_robot`，`claim_kind` 往往是规模（相对 55B 的差），`reducible=false`，课桶仍是 31。它与 C5 并排展示，不能相减。第 47 课把真机做成陷阱格，是为了挡住 LIBERO 冒充；本课把真机做成第六类之外的合法行，是为了让真正的真机进得来。两句话不矛盾：陷阱挡的是假真机，合法行收的是真真机。假真机的识别特征写死：`benchmark=="LIBERO"`。不要改成「数字大于 70% 就是真机」。

### 5.10 三条标红规则：缺 $N$、缺套件、LIBERO 当真机

为什么需要。字段检查若只在 CPU 里返回字符串，浏览器里的人仍会把空格子拖进成功桶。Lab 必须在错误放置的当时把卡片涂红，而不是等揭晓以后用绿勾安慰。第 47 课已经对「LIBERO 进真机格」做了当场标红。本课把同一视觉语言扩到缺字段。

怎么运转。三条规则在放置时立即求值，不依赖揭晓按钮：

1. 卡声明真机，且基准是 LIBERO，且桶是「真机能力」：标红。正确桶是 C5 仿真操作。
2. 卡缺 $N$，且桶不是「缺字段拒收」：标红。正确桶是缺字段拒收。
3. 卡缺套件，且桶不是「缺字段拒收」：标红。正确桶是缺字段拒收。

揭晓之后才显示「放对了」。标红不表示「你已经懂了」，只表示「系统拒绝入账」。验收要的是：三张曾经或必须能被标红的卡，最终停在正确桶；预测选项选中「三张都会标红」。

数学。令放置映射 $\pi:\text{CardId}\to\text{BucketId}$。非法指示

$$
I_{\mathrm{red}}(m,\pi)
=
\mathbf{1}[\mathrm{LIBERO\text{-}as\text{-}real}(m,\pi)]
\ \vee\
\mathbf{1}[\mathrm{Missing}(m)\neq\emptyset\ \wedge\ \pi(m)\neq\mathrm{incomplete}]
$$

$I_{\mathrm{red}}=1$ 时卡片 CSS 走非法态。验收通过还要求 $\pi(m)=b^\star(m)$ 对全部六张成立，且预测等于 `triple`。

验证。Lab 描述写明三条标红。CPU 对同一三张虚构卡返回 `reject_illegal` 或 `reject_incomplete`。两边的失败集合必须相同：`nova_libero_as_real`、`nova_missing_n`、`nova_missing_suite`。SIMPLER 行与 13B 规模行不在这个失败集合里，预测若选「六张都会标红」应判错。

标红文案要写原因，不写分数。LIBERO 进真机格时，文案写「C5 不是真机」，不写「81% 太高」或「81% 太低」。缺 $N$ 时写「没有试验次数就不能写 Wilson」，不写「Long 应该是 53.7」。缺套件时写「Spatial/Object/Goal/Long 键对不上」，不写「90% 不可信」。文案绑定规则，不绑定表值。表值在虚构表里可以任意改，规则不能改。若有人把 Lab 里的 81% 抄进笔记当 OpenVLA 真机分数，第 01 课和第 47 课同时失败：来源是夹具，类是错的。本课在 Lab 标题里写「虚构的 NovaVLA-8B，不得当文献引用」，就是为了把这条抄写路径切断。揭晓按钮在三条标红之后才打开「测什么 / 不测什么」，是为了防止用解释倒推放置。先预测再揭晓与第 31 课三种协议、第 47 课六张卡同一纪律。本课预测的对象从「哪张最常被填进真机」改成「哪几张必须标红」，因为缺字段和假真机会同时出现在同一张新表上。只盯真机陷阱，会把空 $N$ 放进 C5。

### 5.11 准入函数

为什么需要。字段列表若只是清单，组会上仍会靠投票决定「这一行能不能进笔记」。本课把投票换成函数。函数值只有三个，没有「先记上以后再补」。

怎么运转。按这个顺序短路求值，不要并行投票：

1. 若 LIBERO 声称真机，或把 `pearson_r` 声称成成功率 / C5 / 真机：`reject_illegal`。
2. 若 `missing_fields` 非空：`reject_incomplete`。
3. 若标签与第 31、47 课不兼容：`reject_illegal`。
4. 若 `opens_new_model_lesson`：`reject_illegal`。
5. 否则 `admit`。

数学。

$$
\mathrm{Admit}(m)
=
\begin{cases}
\mathrm{reject\_illegal} & \neg\mathrm{Legal}(m)\\
\mathrm{reject\_incomplete} & \mathrm{Missing}(m)\neq\emptyset\\
\mathrm{reject\_illegal} & \kappa(m)=\mathrm{scale}\ \wedge\ b(m)\notin B\\
\mathrm{admit} & \text{otherwise}
\end{cases}
$$

$\mathrm{Legal}(m)$ 包含：六类或合法真机；LIBERO 不得声称真机；C5 套件键属于四套件；C6 单位为 $r$。短路顺序把非法声称放在缺字段前面：一张既缺 $N$、又把 LIBERO 写成真机的卡，返回 `reject_illegal`。Lab 里它只要进真机格就会红；即使它同时缺字段，红的原因仍应优先显示「不是真机」。教学上允许一张卡有两个缺陷，报告里两个都写，函数返回值只留一个，避免检查项爆炸。

代码落点。`admission` 按上述顺序实现。`ADMISSIONS` 集合恰好三个字符串。

验证。Spatial 行 `admit`；缺 $N$、缺套件 `reject_incomplete`；LIBERO 当真机 `reject_illegal`；13B 规模行 `admit` 且未开新课。四类返回覆盖了函数的所有分支。没有第四个返回值叫 `admit_with_gaps`。缺字段的善意收录，本课视为非法。

### 5.12 一张填满的收编卡：OpenVLA Table 12 Spatial

把前十一节收成一张能直接贴进实验记录的卡。下面这张卡的数字来自打开过的 OpenVLA v3 HTML Appendix E Table 12，不是夹具编造。

**论文。** OpenVLA，arXiv:2406.09246v3。

**这一行在测什么。** LIBERO-Spatial，10 个任务，物体集合相同、空间关系不同。仿真，robosuite，PDDL 谓词合取。

**收编字段。**

| 键 | 值 |
|---|---|
| `paper_id` | `2406.09246` |
| `lesson_bucket` | `31`（评测拆桶；动作表示另开 27 的卡） |
| `claim_kind` | `mechanism`（套件分列本身是评测机制） |
| `class_id` | `C5_sim_manip` |
| `benchmark` | `LIBERO` |
| `split` | `spatial_ft` |
| `n` | 500 trials / 套件，3 seeds |
| `unit` | `success_rate` |
| `fine_tune` | true，该套件独立 LoRA，第三视角，过滤失败示范 |
| `reducible` | true（宏平均掩盖 Long 的方向已在 CPU） |
| `suite` | `spatial` |
| `success_predicate` | `pddl_conjunction` |
| `value` | 0.847 |
| `forbidden` | `real_robot`；不得与 Object/Goal/Long 兑成唯一数字后只报平均 |

**同表必须并排的三行。** Object 0.884，Goal 0.792，Long 0.537。宏平均 0.765 可以存在，但不能删除这三行。Diffusion Policy 同行平均 0.724，Object 0.925，用来提醒「平均第一」和「Object 第一」不是同一句话。

**禁止事项。** 不把 0.847 写进真机节；不把 0.765 写成出厂零样本；不把 500 trials 的区间借给 $N=25$ 的真机；不把这一行的 `reducible=true` 理解成 MiniMind-O 达到 84.7%。

同一 Appendix 还写明：预训练全是真机数据、零仿真数据，所以 fine-tune 到 LIBERO 的增益比 fine-tune 到真机 Franka 更窄。这句话属于规模加域差距，另开一张卡，不要写进 Spatial 这张机制卡的 `value` 解释里。混写会让 84.7% 看起来像真机下界。它不是。

### 5.13 机制卡：OpenVLA-OFT Table I

**论文。** Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success，arXiv:2502.19645。

**机制名字。** 并行解码、动作块、连续动作表示、L1 回归。四件都叫得出，才标 `mechanism`。只说「我们做了 Optimized Fine-Tuning」而不列这四件，降级为规模或未完成卡。

**LIBERO 宏平均。** 76.5% 到 97.1%。Long 在加腕部相机的 OFT 行是 94.5%。$\pi_0$ fine-tune 同行平均 94.2%，Long 85.2%。$\pi_0+$FAST Long 60.2%。每套件 500 trials。观察含腕部相机的行必须把观察写进 `split` 或单独字段；缺观察描述就把 97.1% 从你的表里删掉。第 31 课已经写过这条，本课收编时原样执行。

**吞吐。** 动作生成 26×。单位不是成功率。另开一张卡，课桶可以是 01（延迟定义）或 30（控制频率与分块），`unit` 写 `throughput_ratio`，不要乘进 97.1%。

**真机 ALOHA。** 相对默认配方的 $\pi_0$、RDT-1B 以及从头训的 Diffusion Policy / ACT，平均成功率最多高 15 个绝对点。这是真机行，`class_id=real_robot`，不是 C5。15 个点不能和 97.1% 减。评测次数按任务在 10–24 量级，Wilson 会很宽，第 31 课的 $N=25$ 教学口径仍然适用。

**缩小版方向。** 可以复现的符号：动作块抬长程、连续 L1 相对离散 bin 在套件内上升。不能复现的数字：97.1%、26×、ALOHA +15%。`reducible=true` 只绑第一句。

### 5.14 规模卡：7B 对 55B 的 +16.5%

**论文。** 仍是 OpenVLA。这一行常常被写成「小模型打过大模型」，听起来像机制。打开原文：7B 对 55B，数据 970k 对 RT-2-X 所用混合物，视觉编码器是融合的 SigLIP+DINOv2，评测是 WidowX 170 rollouts 加 Google Robot 60 rollouts。参数量差一个数量级，试验次数是小样本，域是真机。本课把它标成 `scale`，`class_id=real_robot`，`reducible=false`，课桶 `31`。融合视觉编码器可以另开一张 21 / 23 的机制卡，但 +16.5% 这个减法本身不是那张卡的因变量。

**为什么缩小版追不上。** MiniMind-O 可训练主体约 26M，没有 970k 真机轨迹，没有 WidowX 协议。方向「开源 7B 可以在部分真机任务上超过闭源 55B」依赖数据与评测协议，26M 夹具给不出同号的 $\Delta$。强制 $\rho=0$，避免有人在 CPU 里用玩具成功率差去「验证」16.5。

**和 OFT 的差别。** OFT 的 76.5% 到 97.1% 发生在同一套件、同一 7B 骨架、同一 500 trials 量级上，自变量是解码与动作表示。+16.5% 发生在不同骨架、不同参数量、不同真机任务集合上。前者是机制卡，后者是规模卡。两张卡可以属于同一 `paper_id`，必须分开写 `claim_kind`。

### 5.15 虚构 NovaVLA 表：专门用来触发标红

Lab 与 CPU 共用六行教学夹具。数字全部是虚构的，表名 NovaVLA-8B 不是文献。它存在的唯一理由是：公开论文通常不会同时在一张表上犯三种错，教学需要一张会犯三种错的表。

| 行 | 图上的句子 | 缺陷 | 正确桶 | 准入 |
|---|---|---|---|---|
| A | 真机成功率 81% | LIBERO 平均冒充真机 | C5 仿真操作 | `reject_illegal`（若声称真机） |
| B | Spatial 88%，$N=500$ | 无 | C5 | 若其它字段齐，`admit` |
| C | Long 高 | 缺 $N$ | 缺字段拒收 | `reject_incomplete` |
| D | LIBERO 90% | 缺套件 | 缺字段拒收 | `reject_incomplete` |
| E | VisMatch $r=0.81$ | 无（单位已写 $r$） | C6 | `admit` |
| F | 13B 比 8B +4 点，同一离散 token | 无（规模） | 规模声明 | `admit`，$\rho=0$ |

行 A 的正确课桶仍是 31 的 C5，不是「删除这一行」。删除会让读者以为 LIBERO 平均不该被记录。该记录，该打 C5，该写 fine-tune，不该写真机。行 C、D 在补齐 $N$ 与套件之前，连 C5 主列都进不去。行 E 的 0.81 是虚构的 $r$，不要和 SIMPLER Table I 的 0.924 比谁大：单位相同但论文不同、政策名单不同，且一行是夹具。行 F 证明：规模行可以 `admit`，只要你不新开课、不把 +4 点标成可复现方向。

预测题的标准答案是「缺 $N$、缺套件、把 LIBERO 写成真机，三张都会标红」。只选行 A 的人，会在补字段之前把 C、D 放进 C5，让 90% 和 88% 看起来像可以减的两个点。只选 E、F 的人，把单位写对的 $r$ 和诚实的规模声明当成错误。六张都选的人，没有区分合法入账和非法入账。三种错选都在 Lab 揭晓后给红字，不在拖桶过程中提前显示「测什么 / 不测什么」，避免对着已经写好的解释倒推。

### 5.16 错误收编与合格收编对照

**错误报告（禁止提交）。**

> NovaVLA-8B 达到 85% 综合成功率，超过 OpenVLA，接近真机部署。Long 也很高。13B 版本再加 4 个点。我们下节课改为讲 NovaVLA。

这一段同时触发：跨类平均、LIBERO 当真机、缺 $N$、缺套件、规模当机制、新开模型课。任何一句单独出现，本课验收失败。

**合格报告（可以进笔记）。**

> OpenVLA Table 12：C5，fine-tune，Spatial / Object / Goal / Long = 84.7 / 88.4 / 79.2 / 53.7，平均 76.5%，每套件 500 trials × 3 seeds。该类不是真机。OFT Table I 在同一拆法上把平均做到 97.1%（观察含腕部相机），机制是并行解码、动作块、连续 L1；缩小版只复核「Long 相对更难、动作块抬长程」的方向。SIMPLER VisMatch $r=0.924$，单位 pearson_r，C6。OpenVLA 相对 RT-2-X 的 +16.5% 是真机小样本规模声明，$N$ 约 230，$\rho=0$。虚构 NovaVLA 表用于教学标红，不引用其百分数。不新开模型课。

合格报告里每一个百分数旁边都有类、套件或 $N$、种类。没有「综合」。没有「下节课改讲新模型」。

两段对照可以当组会投影。左页错误，右页合格，中间不要放作者 logo。logo 会把讨论拽回「这是不是 SOTA」。本课的问题是「这格能不能进账」。SOTA 是排行榜的问题，排行榜不在 $B$ 里。

### 5.17 七问决策树：拿到一篇新论文先问什么

按这个顺序问，不要跳。跳问会把规模当成机制，或把缺字段当成「作者没写所以不重要」。

1. **这一行的百分数有没有 $N$？** 没有：拒收。有：写下 $N$。
2. **基准是不是 LIBERO？** 是：套件键必须是四套件之一，类必须是 C5。缺套件：拒收。声称真机：拒收。
3. **单位是什么？** `pearson_r` 走 C6，禁止当成功率。`success_rate` 才能谈 Wilson。
4. **类标签是 C1–C6 还是真机？** 真机必须不是 LIBERO。跨类禁止做差。
5. **课桶是哪一课已有口径？** 填 01 / 21–32 / 47 等已有编号。想填新模型课：拒收。
6. **规模还是机制？** 只能改参数量和数据量：规模，$\rho=0$。能改夹具里的一行结构：机制，再谈方向。
7. **缩小版复现的是符号还是绝对值？** 绝对值：几乎一定失败，不要写进标题。符号：写进 `reducible`，并列出不能复现的否定句。

七问都有答案之后才允许把数字抄进总表。七问里第 1–3 问失败，后面四问不必做：卡已经死了。Lab 的揭晓按钮对应第 1–3 问的视觉结果；勾选「不引入新模型」和「标签与第 31、47 课兼容」对应第 4–6 问。第 7 问写在实验记录里，浏览器不替你判断符号，因为符号依赖你选的玩具自变量。

决策树与第 47 课七问不抢题。47 课问的是「这个数字是哪一类」。本课问的是「这个数字能不能进课程」。类对了仍可能缺 $N$，仍可能是规模，仍可能不该新开课。两棵树要串起来：先 47 再 60，或先 60 的第 1–3 问再 47。不要只做一棵。

### 5.18 把 SIMPLER 填成一张不会乘 100 的卡

C6 是收编时第二容易写错的类，第一是把 LIBERO 当真机。错法固定：看见 0.924，乘 100，画成柱，标题写成 sim-to-real 成功率。第 31、47 课已经禁过。本课把它写成一张填满的卡，证明 C6 也能 `admit`，只要单位不被改掉。

**论文。** Evaluating Real-World Robot Manipulation Policies in Simulation，arXiv:2405.05941。

**这一行在测什么。** 仿真排序和真机排序是否一致，不是某个政策的绝对成功率。机体：Google Robot。视觉协议：Visual Matching。配对政策数量：Table I 写六个 checkpoint，所以 $N=6$ 指政策点数，不是 trials。

**收编字段。**

| 键 | 值 |
|---|---|
| `paper_id` | `2405.05941` |
| `lesson_bucket` | `31` |
| `claim_kind` | `mechanism`（评测代理，不是新政策） |
| `class_id` | `C6_sim2real_rank` |
| `benchmark` | `SIMPLER` |
| `split` | `google_robot_vismatch` |
| `n` | 6 |
| `unit` | `pearson_r` |
| `fine_tune` | false（这是评测套件，不是对政策 fine-tune 的声明） |
| `reducible` | true（单位纪律与「不得当成功率」可在夹具复现） |
| `value` | 0.924 |
| `forbidden` | `success_rate`；`real_robot`；与 LIBERO 76.5% 做差 |

Variant Aggregation 同行平均 $r=0.778$，validation MSE 0.308。三行必须分列。Drawer 任务上 VarAgg 的 $r$ 只有 0.486，VisMatch 到 0.942，说明换视觉协议代理质量会垮。收编时 `split` 写 VisMatch 或 VarAgg，空白则与缺套件同罪：看起来有数字，协议对不上。

Wilson 不得画在 $r$ 上。$N=6$ 个政策点若被误当成 6 次伯努利试验，公式会吐出一个毫无对象的区间。CPU 对 SIMPLER 行不调用 `wilson_interval`。谁在笔记里给 0.924 画上 $[0.61,0.91]$，属于把第 31 课的教学口径套到错误单位，第 47 课的 `simpler_r_rejected_as_success_rate` 与本课的 `illegal_simpler_as_success` 都应捕获。Lab 的虚构 $r=0.81$ 正确桶是 C6，不标红：单位已经写对。标红集合不含诚实的 C6。

### 5.19 从摘要到收编卡的十分钟操作

组会上有人把手机递过来，屏幕上是一段摘要。本课要求十分钟内交出至少两张卡草稿，而不是一句「看起来很强」。用 OpenVLA 摘要当操练文本，因为它是已打开的原文，不是虚构。

摘要里同时出现：7B、970k 真机示范、相对 RT-2-X（55B）在 29 项任务上绝对成功率高 16.5%、消费级 GPU 上 LoRA fine-tune、量化推理。按第 5.3 节的刀法切：

1. **7B + 离散动作。** 机制卡，课桶 25 / 27。摘要没写 256 bin，所以 `split` 先标 `unspecified`，打开 §3.2 再补。未补之前这张卡不能拿 16.5% 当 `value`。
2. **970k 轨迹。** 规模加配比，课桶 26。`n` 是轨迹数 970000，单位不是成功率。另写一张，禁止和 16.5% 减。
3. **+16.5%，29 任务。** 真机规模卡，课桶 31，`class_id=real_robot`，$N$ 需要翻 §5.1：WidowX 170 加 Google Robot 60。摘要本身没写 170 与 60，所以从摘要直接填 $N=29$ 是错的：29 是任务数，不是 trials。缺正确 $N$ 时本课允许你把卡标成草稿并拒收主表，不允许你用 29 冒充 trials。
4. **LoRA 与量化。** 机制卡，课桶 18 / 27。因变量是显存与是否掉成功率，OpenVLA Table 2 写 4-bit 与 bfloat16 在 Bridge 上接近，8-bit 因频率掉点。这些数字进延迟与适配账，不进 LIBERO 账。

十分钟结束时，手上应有四张草稿，其中至少一张因为 $N$ 还没从正文核对而被 `reject_incomplete`。这个「故意不完整」是正确输出。错误输出是：一张卡，`value=0.165`，标题「OpenVLA 全面超过 RT-2-X」。全面超过把仿真、真机、LoRA、量化兑在一起，第 01、31、47、60 四课同时失败。

把同一操作再用 OFT 摘要做一遍。76.5% 到 97.1% 进 C5 机制卡，26× 进吞吐卡，ALOHA 最多 +15 个绝对点进真机卡。三张卡的 `paper_id` 相同，`class_id` 分别为 `C5_sim_manip`、不适用六类的吞吐、`real_robot`。十分钟够抄这三个 `value`，不够核对 Table I 的腕部相机。腕部相机没核对之前，97.1% 不得与「仅第三视角」的 76.5% 画成同一条曲线的两个端点。观察是协议 $\mathcal{P}$ 的一部分，和第 31 课相机节相同。本课收编不放松那一条。

虚构 NovaVLA 摘要可以故意写成一句话：「我们在真机上达到 81%，Long 也很高，13B 再加 4 个点。」十分钟操作的正确答案是三张拒收或规模卡，零张 `admit` 进真机。Lab 把这句话拆成行 A、C、F，强迫你用桶而不是用语气完成同一判断。

### 5.20 四课字段对照：01、31、47、60

收编卡不是另起一套名词。它是前三课字段的并集，再加上课桶、种类、方向三个键。对照写在一张表里，避免有人问「协议卡、评测类、收编卡到底填哪张纸」。

| 键 | 01 | 31 | 47 | 60 |
|---|---|---|---|---|
| 出处 / SHA / `paper_id` | commit、数据哈希 | 表号 | 表号 | `paper_id` + 表号 |
| 分桶 | 模态、长度、speaker | 套件 | 六类 | 六类 + 课桶 |
| $N$ | golden case 数、评测条数 | trials | 题量或 trials | 必填，空则拒收 |
| 单位 | WER、准确率、TTFA | 成功率、$r$ | 六类各有单位 | 沿用 47，错单位拒收 |
| fine-tune | mini 对 full 不得混 | 必须写 | 必须写 | 必须写 |
| 成功定义 | mask、ASR 终点 | 接触 / 放置 / 保持 | 选项 / 脚本 / 谓词 / $r$ | C5 必填谓词 |
| 真机 | 不涉及机械臂 | 第四桶，小样本 | 陷阱格 | LIBERO 声称则标红；真真机可 `admit` |
| 规模或机制 | 不显式 | 不显式 | 不显式 | 必填；规模 $\rho=0$ |
| 缩小版方向 | trace 不改 logits 可复现 | Long 掉点可复现 | 互斥可复现 | 仅机制为 true |
| 新模型课 | 禁止把官方权重当自训 | 禁止把仿真当真机 | 禁止第七类 | 禁止 `new_model_lesson` |

读表的方法：往右只增加约束，不删除左边的约束。本课 `admit` 并不免除第 01 课的哈希，也不免除第 31 课的 Wilson，也不免除第 47 课的互斥。谁在 60 课通过之后把六类平均写回幻灯片，47 课的验收被回滚，60 课也视为未交付。活的协议如果不能回滚旧禁则，就不是活的，是一张覆盖层。覆盖层会过时。并集不会：新论文到来时，你只往表里加行，不把旧列删掉。

对照表也能解释 CPU 为什么要再跑一遍 Wilson 和六类互斥。单独跑第 60 课的缺字段检查，而不跑第 31、47 课的旧检查，会出现一种假通过：字段齐全、类是 C7、套件叫 `long_horizon`、$N=25$ 却用正态宽度为零的公式。本课把旧检查嵌进 `run()`，就是为了让假通过在本课文件里失败。编排者登记时不要把这些旧检查删掉「以保持 60 课纯粹」。纯粹的 60 课如果不能兼容 31 和 47，规格就写错了。

## 6. 在公开实现中定位这些机制

本课不改 MiniMind-O。机制落在论文表格标题、评测脚本的列名，以及本课夹具的键名。行号会变，认职责，不背行号。

**OpenVLA 仓库**（[openvla.github.io](https://openvla.github.io)）。训练入口把动作离散成 256 bin，覆盖 Llama tokenizer 最后 256 个 token。评测 LIBERO 时看结果 JSON 的键是不是按 Spatial / Object / Goal / Long 分列，有没有 `unnorm_key`、LoRA 配置、第三视角。标题写 `success_rate` 却把四套件平均写进 `real_robot`，定位失败。真机评测脚本若存在，应与 LIBERO 脚本分成两个 `run_id`。

**OpenVLA-OFT 仓库**（[openvla-oft.github.io](https://openvla-oft.github.io/)）。定位并行解码、动作块长度、连续动作头、L1。缺这四段代码却报告 97.1%，那是规模卡或未完成卡。吞吐测量必须和成功率分列。ALOHA 评测次数写在任务级，不要平均进 LIBERO。

**LIBERO**（[arXiv:2306.03310](https://arxiv.org/abs/2306.03310)）。PDDL 的初始分布与目标谓词。OpenVLA 改的是数据过滤和 LoRA，不是谓词。你若改谓词再报 76.5%，协议已经换了。

**SIMPLER**（[arXiv:2405.05941](https://arxiv.org/abs/2405.05941)）。列名是 Pearson $r$ 与 MMRV。Visual Matching 与 Variant Aggregation 必须分列。把 `r` 列改名 `success` 再画柱，定位失败。

**第 01 课 MiniMind-O**。`eval_omni.py`、golden case、manifest 哈希。这些键属于自训账。不要把 OpenVLA 的 76.5% 写进同一份 `baseline-v1` 报告的主表。

定位时用一份对照清单，避免把训练循环误认成评测循环。C5 脚本返回的是谓词合取还是动作 MSE。C6 脚本输出的列名是 `pearson` 还是 `success`。真机脚本的 $N$ 是按任务还是按方法合计。OFT 训练循环里动作块长度是训练超参还是只在推理时改。四条里认错一条，收编卡的 `claim_kind` 或 `unit` 就会错：把动作 MSE 当 C5，或把推理步数消融当成预训练规模。

Qwen2.5-Omni 报告仍然是分节样例。Table 5 / 7 / 8 分开写 MMMU val、Video-MME、OmniBench。定位收编时，三张表三张卡，课桶都是 47，种类是评测机制（分节），不是新 Omni 骨干课。TTS WER 进第 01 课语音账。谁在仓库里搜到 `omni_score`，应当改成按类落盘，而不是给它补一个本课收编平均。

定位收编卡时，建议在仓库里搜三类符号，而不是搜模型昵称。第一类是评测列名：`success_rate`、`pearson`、`mmrv`、`accuracy`、`wer`。列名决定 `unit`。第二类是协议开关：`use_wrist_camera`、`use_subtitle`、`lora`、`action_chunk`、`num_trials`。开关决定 `split` 与 `n`。第三类是结果写入路径：是写入 `libero_spatial.json` 还是写入 `overall.json`。写入 `overall.json` 且没有套件键，等于本课的缺套件行。搜昵称只会把你带回 README 的第一句广告。广告不是 schema。

MiniMind-O 侧能对上的位置很少，这是故意的。`eval_omni.py` 可以对 golden case 输出文本与音频指标，那些键进第 01 课。它没有 LIBERO 环境，没有 OSWorld 虚拟机，没有 MMMU 科目桶。看见仓库里没有这些脚本，收编卡上对应行应写 `out_of_scope`，不要把 OpenVLA 的 76.5% 填进去充数。本课允许空行。本课不允许用别人的脚本输出冒充自己的 `run_id`。第 01 课的 SHA 纪律在这里的对应物是：`paper_id` 指向别人，`run_id` 指向你。两个字段相等，只有一种合法情况：你在做文献抄录，分区名必须叫 `literature`，不得叫 `baseline-v1`。

## 7. 数据与协议 recipe

本课的「数据」是收编记录，不是训练语料。recipe 是收编卡模板。复制下面这张表，一行数字一行。

| 字段 | 填写说明 | 非法例 |
|---|---|---|
| `paper_id` | arXiv 或 `nova-vla-fiction` | `twitter_screenshot` |
| `lesson_bucket` | 已有课号 | `new_vla_lesson` |
| `claim_kind` | scale / mechanism | `innovation=0.8` |
| `class_id` | C1–C6 或合法 real_robot | `C7_general` |
| `benchmark` | 论文基准名 | `内部集` 且无说明 |
| `split` | test / val / 套件 / VisMatch | 空白 |
| `n` | 正整数 | 空白；只写「大量」 |
| `unit` | accuracy / success_rate / pearson_r | `%` 且把 $r$ 乘了 100 |
| `fine_tune` | true / false / 未公开 | 把 FT 写成零样本 |
| `reducible` | 仅机制可为 true | 规模行标 true |
| `suite` | C5 / LIBERO 必填四键之一 | 空白却报 90% |
| `success_predicate` | 谓词或脚本名 | 「看起来做完了」 |
| `forbidden` | 本行不得声称的能力 | 空白却在正文声称真机 |

六行公开对照建议先抄进表，再处理新论文：MMMU test 55.7%（C1）、Video-MME 无字幕 75.0%（C2）、OmniBench 56.13%（C3）、OSWorld 12.24%（C4）、LIBERO 76.5%（C5）、SIMPLER $r=0.924$（C6）。六行的填写示范见第 47 课第 7 节。本课在六行之外加三行教学卡：OpenVLA Spatial 0.847、OFT 宏平均 0.971、OpenVLA 真机差 0.165。第三行 `class_id=real_robot`，`reducible=false`。LIBERO 两行的 `forbidden` 必须包含 `real_robot`。SIMPLER 行的 `forbidden` 必须包含 `success_rate`。规模行的 `forbidden` 必须包含 `new_model_lesson` 与 `reducible_absolute`。

可直接运行的 CPU 入口（编排者登记前）：

```bash
python3 -c "from learn_omni_experiments.lessons.lesson_60 import run; r=run(); print(r['checks']); print(r['metrics']['missing_n_admission'])"
```

登记之后改用仓库统一入口，一条命令：

```bash
python3 run.py run 60
```

推荐配置：先填表，再画柱。没有表的柱状图，本课视为未完成交付。教学模拟的六行已经填好虚构数字，你的作业是把它们抄进自己的表并标来源 `fiction`，同时另抄 OpenVLA / OFT / SIMPLER 的公开行并标来源 `arxiv`。混来源且不写 `paper_id`，等于第 01 课用 `master` 当 SHA。

若你要给 MiniMind-O 基线做对照，C1–C6 多数格子应写 `unspecified` 或 `out_of_scope`。空表是诚实的基线。拿 GPT-4V 的 55.7% 或 OpenVLA 的 76.5% 填进 MiniMind-O 的主表，属于第 01 课禁止的官方评测冒充自训。收编卡可以把这些数字放在「文献对照」分区，分区键必须与自训分区不同。

拒收行也要有示例，避免有人以为拒收就是删掉 JSON。下面这条可以原样放进 `artifacts/lesson60/rejected.json`，作为教学夹具，不是论文：

| 键 | 值 |
|---|---|
| `paper_id` | `nova-vla-fiction` |
| `lesson_bucket` | `31` |
| `claim_kind` | `mechanism` |
| `class_id` | `C5_sim_manip` |
| `benchmark` | `LIBERO` |
| `split` | `long` |
| `n` | 空 |
| `unit` | `success_rate` |
| `fine_tune` | true |
| `reducible` | false |
| `suite` | `long` |
| `claimed_class` | `C5_sim_manip` |
| `value` | 0.62 |
| `admission` | `reject_incomplete` |
| `missing` | `n` |

旁边再放一条 `claimed_class=real_robot`、`n=500` 的行，`admission=reject_illegal`。两条都保留在案卷里。案卷的作用和第 01 课保留失败 golden case 相同：以后有人问「为什么 81% 不能进真机节」，你可以出示拒收记录，而不必重新辩论。Lab 通过条件要求你最终把行 A 放回 C5，但实验记录里应留下它曾经进过真机格并被标红。没有这次标红，通过只说明你会拖对，不说明系统会拒绝。验收要的是拒绝能力。

## 8. 按依赖顺序执行实验

实验不训练模型。顺序从便宜到贵。

### Step 1: 跑必填字段夹具

在 `experiments/` 目录，确认 `PYTHONPATH` 能找到 `learn_omni_experiments` 之后：

```bash
python3 -c "from learn_omni_experiments.lessons.lesson_60 import run; import json; print(json.dumps(run()['checks'], ensure_ascii=False, indent=2))"
```

`checks` 必须全为 True。重点看：缺 $N$、缺套件拒收；LIBERO 当真机拒收；Spatial 行准入；OFT 机制可复现方向；规模行不新开课；第 47 课六类仍互斥；第 31 课套件键接受；Wilson 与 $N=25$、$k=20$ 一致。失败则先改对夹具理解，不要改公开数字，更不要改虚构行去让检查变绿。

### Step 2: 浏览器拖虚构表

打开本课 Lab。开始时桶是空的。先在四个选项里预测哪几张卡会标红。再把六行拖进桶，或先点卡再点桶。建议故意先把行 A 放进真机格、把行 C 和行 D 放进 C5，确认三条红文案出现，再改放到正确桶。全部放对、预测选「三张都会标红」、勾选「不引入新模型」与「标签与第 31、47 课兼容」后，验收通过。此步是教学模拟。揭晓之后六行下方会给出测什么、不测什么。把这些句子抄进收编卡的 `forbidden` 列，不要只截一张全绿的桶。预测选错时，即使桶放对了也不通过：本课要你事先知道三种缺陷都会红，而不是对着已经标绿的格子倒推。

### Step 3: 手写三张公开卡加一张虚构拒收卡

用第 7 节模板抄：OpenVLA Spatial、OFT 宏平均、SIMPLER VisMatch。再手写一张缺 $N$ 或缺套件的卡，把 `admission` 写成 `reject_incomplete`。禁止把 Lab 的「通过」写成 81% 被你复现。四张写完后做一次自检：C5 行含 `real_robot` 禁令；C6 行含 `success_rate` 禁令；规模行含 `new_model_lesson` 禁令；缺字段行没有 `value` 进主表。自检失败就还没交付，哪怕 Lab 已经全绿。

自检还有一个计时动作：把四张卡扣过去，只留键名，默写 `value` 的出处。Spatial 必须说得出 Table 12，OFT 必须说得出 Table I 与腕部相机，SIMPLER 必须说得出 Table I Visual Matching，缺 $N$ 那张必须说得出它是虚构。默写失败的数字从主表删除，直到你能指到 HTML 页的表号。这个动作对应第 01 课「不用会移动的 master 当版本号」：记不住表号的百分数，和记不住 SHA 的 checkpoint 一样不可审计。

### Step 4: 拒绝一次新开课

在笔记里写一句假申请：「建议新增第 61 课：NovaVLA-13B」。然后划掉，改写成：「规模对照，接到第 27 课，`reducible=false`」。这一步不算数学，算肌肉记忆：模型名欲出现时先问机制关键词。关键词空，就没有新课。

### Step 5（可选，有 GPU）

只跑一个已有课桶里的子集，例如 LIBERO-Spatial 10 任务 × 10 trials，greedy，记录 LoRA 与观察。报告标题必须含 C5、`suite=spatial`、$N=10$、checkpoint 名。不得把 10 trials 的点估计标成 Table 12 的 84.7%。$N=10$ 时即使 10 次全成功，也只许写 Wilson 下界约 0.72，不许写「已解决」。没有这一步，本课仍可交付。有了它，也仍然不能把自训数字和文献数字兑在 `omni_score` 里。超时或 OOM 记失败模式，不要改成「改抽 3 次重报」。看过结果再改 $N$，属于第 17 课和第 31 课已经禁止的事后改公式。本课沿用那条禁则。

## 9. 评测与测量

本课要测的不是模型，是收编是否合法。

测量对象：

- 必填字段是否在缺 $N$、缺套件时非空；
- LIBERO 声称真机是否为非法；
- 六类与四套件键是否仍与第 47、31 课相同；
- Wilson 是否仍与第 31 课 $N=25$、$k=20$ 一致；
- Lab 中三条标红是否可被触发；
- 规模行是否未打开新模型课。

不测量：

- 任何真实 VLA 的 LIBERO 分数；
- 任何真实政策的真机分数；
- NovaVLA 是否存在。

测量记录本身也要分桶。CPU 的 `metrics` 里 0.847、0.765、0.924 是夹具常量，用来核对标签函数，不得拷进模型卡。Lab 的 `placements` 是分类结果，单位是桶名，不是百分数。`liberoInTrap`、`missingNHot`、`missingSuiteHot` 是布尔，表示有没有触发标红。三类对象写进同一份 `result.json` 时必须带类型字段，避免以后回看时把 0.81 当成你测出来的 SIMPLER。第 01 课要求每个指标可追溯到 case；第 47 课要求每个数字可追溯到类；本课要求每张卡可追溯到准入值。三句话合在一起，才是完整的账本索引。

若你额外跑了真实评测，测量纪律回到对应课：C5 拆套件并写 fine-tune；C6 写 VisMatch 或 VarAgg；真机写 $N$ 与谓词。真实评测的数字另起 `run_id`，不要覆盖夹具 `metrics`。收编卡在真实评测上的 `paper_id` 改成你的 checkpoint SHA，`claim_kind` 几乎一定是规模（你没有改公开机制，只是在别人的协议上跑），`reducible` 对别人的表值无定义。不要把「我跑了 Spatial 10 次」写成「我复现了 OpenVLA」。

## 10. 验收条件

只有下列项目全部满足，才能把本课标记为完成：

- [ ] 能默写收编卡十个必填字段，以及 C5 / LIBERO 额外的套件与谓词；
- [ ] CPU `checks` 全为真，含缺 $N$、缺套件拒收、LIBERO 拒收真机、六类互斥、四套件键、Wilson；
- [ ] Lab 预测选中三张标红，六行进入正确桶，真机陷阱没有 LIBERO；若中途把 LIBERO 拖进真机格或把缺字段放进成功桶，看见标红；
- [ ] 手写至少三张公开收编卡加一张拒收卡，没有「综合成功率」单元格；
- [ ] 引用 76.5% 时写了 LIBERO、四套件、fine-tune、OpenVLA Table 12、C5；
- [ ] 引用 97.1% 时写了 OFT、观察、机制四件套，没有写成 MiniMind-O 结果；
- [ ] 引用 0.924 时写了 SIMPLER、Visual Matching、Pearson $r$、C6；
- [ ] 引用 +16.5% 时写了真机、7B 对 55B、$N$ 量级、`scale`、$\rho=0$；
- [ ] 没有为 13B 对 8B 的对照新开模型课；
- [ ] 明确写出：本课夹具和虚构表不是任何基准的复现分数。
- [ ] 勾选「不引入新模型，只引入收编规则」与「标签与第 31、47 课兼容」。

## 11. 根据症状定位失败环节

调试时从最便宜、最可观测的一层开始查。总表很好看，先查字段和类标签，不要重训模型。

| 症状 | 可能原因 | 诊断 | 修复 |
|---|---|---|---|
| 新论文直接新开一课 | 用模型名当接口 | 查机制关键词是否已在 25/27/28 | 接到已有课，标规模或机制 |
| 「Long 高」和 84.7% 并排 | 缺 $N$ 仍入账 | `missing_fields` 是否含 `n` | 拒收，或补 trials |
| 「LIBERO 90%」和 Spatial 比 | 缺套件 | `suite` 是否四键之一 | 拒收，或分列四套件 |
| 81% 写在真机节 | LIBERO 平均冒充 | `benchmark` 是否 LIBERO | 移回 C5，真机留空或另填真机行 |
| 97.1% 写成 26M 已复现 | 把方向当绝对值 | `reducible` 是否被滥用 | $\rho$ 只绑符号，否定句写回卡脚 |
| +16.5% 写成机制突破 | 参数量差被忽略 | 7B 对 55B 是否写明 | 改 `scale`，$\rho=0$ |
| $r=0.81$ 画成 81% 柱 | 单位乘了 100 | 列名是否 pearson | 改回 $r$，C6 |
| Lab 不通过但自认放对 | 预测不是 triple，或缺勾选 | 看预测与两个复选框 | 重选三张标红，勾选两句声明 |
| CPU 互斥失败 | 迁入的 47 课记录被改类 | 打印 `gold_47` | 恢复六类双射 |
| Wilson 对不上 | 改了 $z$ 或先看了输出再抄 | 手算 $z^{2}=3.8416$ | 与第 31 课对到 $5\times 10^{-6}$ |
| 把 OFT 26× 乘进成功率 | 吞吐和 C5 兑在一起 | 单位是否 throughput | 分卡 |
| 虚构 81% 进文献表 | 夹具当论文 | `paper_id` 是否 fiction | 删除，或标 teaching-only |
| val 与 test 兑在一起 | `split` 空 | 对照 MMMU Table 2 | 分列 |
| 把动作 MSE 当任务成功 | C5 脚本认错 | 列名是否 success | 回到谓词脚本 |
| 10/10 写成已解决 | 忘了 Wilson | $N=10$ 全成功下界约 0.72 | 补区间，不下部署结论 |

## 12. 交付物

1. 收编卡模板（十个必填字段 + C5 两列），含 `forbidden`；
2. 至少三张公开论文卡（OpenVLA Spatial、OFT 宏平均、SIMPLER $r$）和一张拒收卡；
3. CPU 实验 `checks` 全真的运行记录；
4. Lab 截图或文字记录：预测选三张标红、三条标红曾被触发或明确写「理解标红规则」、最终六行就位、两句声明已勾选；
5. 一份禁止事项清单，贴在以后任何新论文总表脚注；
6. 若跑了可选真实评测：独立 `run_id` 与范围声明。

交付物第 5 条的禁止事项清单建议印这七句：不新开模型课；不把缺 $N$ 的行入账；不把缺套件的 LIBERO 行入账；不把 LIBERO 当真机；不把 SIMPLER 的 $r$ 当成功率；不把规模声明标成绝对值可复现；不把夹具或虚构表当文献。七句都可以用本课夹具或 Lab 触发失败。清单不是口号，是验收开关的人话版。张贴位置不限：笔记本封面、评测脚本仓库的 README 第一段、组会投影的最后一页都可以。关键是写下一张总表之前能看见。看不见的禁则等于没有禁则。第 47 课已经要求把六类禁则贴在 Omni 总表脚注；本课把收编禁则叠上去，两张纸条一起用。只贴 47 会漏掉缺 $N$ 和新开课；只贴 60 会漏掉 MMMU 冒充视频。重叠的那几句（LIBERO 不是真机，$r$ 不是成功率）重复张贴没有坏处，删掉其中一张才有坏处。

## 13. 前沿对照与改造方向

**公开方案。** 同一问题，已经公开的系统其实在报告结构上有对、有错，本课只引用课程里已有或本课已打开的文献。OpenVLA 把 LIBERO 放在 Appendix E，把 WidowX / Google Robot 放在正文真机节：分节是对的，读者仍可能把 76.5% 读成真机，所以第 31、47 课和本课继续挡。OFT 在 Table I 写明 fine-tune、套件、观察、500 trials，并在摘要把 76.5% 到 97.1% 与 26× 吞吐并提：并提可以，混单位不行。SIMPLER 把 $r$ 和 MMRV 当主指标，作者自己写目标不是数字孪生：单位做对了。Qwen2.5-Omni 技术报告按图像、视频、混合模态、语音分节：分节做对了，读者仍可能兑成 Omni 均分，第 47 课已经挡过。CLIP 与 LLaVA 没有机器人表，但它们示范了机制（InfoNCE、两阶段）和规模（4 亿对、158K 指令）必须分列；第 21、22 课已经收编，本课不重做。这些公开方案的共同点：好的报告靠分节和脚注，不好的引用靠一张柱加一个新模型课申请。

**差距。** 规模差：你没有 64×A100、14 天、970k 轨迹，也没有 ALOHA 双臂，也没有 500×4 的 LIBERO 推理预算。机制差：必填字段、三条标红、规模与机制分流、课桶路由、方向复现的符号检查，不依赖模型大小。本课夹具用固定字典就能复现「缺字段拒收」和「LIBERO 不是真机」。不能复现的是 84.7%、97.1%、0.924、+16.5% 这些原文表值本身——它们已经是别人的评测，不是你的训练结果。也不能复现 NovaVLA，因为它不存在。

**动手改造清单。**

1. **日志禁则。** 改动位置：任何把多个基准分数 `mean()` 后写入 `omni_score` 的脚本，改成先写收编卡再按 `class_id` 落盘，并拒绝 `n is None`。预算：0，只改日志。预期：缺 $N$ 的行不能进 parquet。失败判定：仍能 grep 到跨类平均或空 $N$。
2. **陷阱格回归。** 改动位置：结果校验器增加 `forbidden`。LIBERO 行若 `claimed_class == real_robot` 则非零退出。预算：CPU。预期：故意写错的夹具被拒绝，和 Lab 标红同方向。失败判定：错标签仍能生成 HTML 报告。
3. **规模不得新开课。** 改动位置：课程元数据检查，`claim_kind==scale` 的行 `lesson_bucket` 必须已存在。预算：无训练。预期：13B 对照接到 27。失败判定：生成了 `61_novavla.md` 草案。
4. **方向复现开关。** 改动位置：只对 `claim_kind==mechanism` 的行允许 `reducible=true`；CPU 断言 `reducible_implies_mechanism`。预算：无训练。预期：OFT 行可通过，+16.5% 行不能标 true。失败判定：规模行的 `reducible` 为真仍能合并进主表。

**顺手复现。** 论文结论「宏平均掩盖 Long」（OpenVLA Table 12：76.5% vs Long 53.7%）对应 CPU 的 `libero_macro_matches_four_suites`，预期同方向。论文结论「OFT 靠动作块和连续 L1 抬 Long」对应 `oft_mechanism_reducible`，预期只复核方向。论文结论「SIMPLER 主指标是 $r$」对应 `simpler_unit_is_pearson_r`。论文结论「7B 对 55B 的真机差」对应 `scale_claim_not_reducible_to_sota`。若有人在 Lab 里看到 81% 就写「接近真机 80%」，判失败——行 A 是虚构 C5，且真机 $N=25$ 的 Wilson 下界只有约 0.61。

改造实验的失败判定要能被本课夹具触发。实验 1 用缺 $N$ 就能失败。实验 2 用 `real_robot` 声称就能失败。实验 3 用规模行就能失败。实验 4 用 `reducible=true` 的规模行就能失败。四条都不必等 GPU。

规模上追不上的部分写清楚，避免用机制通过去暗示表值被复现。LIBERO 每套件 500 trials、OFT 的 ALOHA、OpenVLA 的 21,500 A100-hour 预训练，都是钱和时间。机制上可以追上的部分也写清楚：字段、forbidden、admission、陷阱格标红、Wilson 只用于二项成功率、课桶不得新建模型课。报告里把能验收的机制和不能验收的规模分成两段。有人在 Lab 里把六行放对，就在标题写「复现 NovaVLA SOTA」，判失败——NovaVLA 是虚构，本课没有 SOTA，只有收编。

## 14. 论文与必读材料

按「评测表怎么收编、机制怎么收编、规模怎么收编、报告结构」顺序读。每篇材料对应一个能在收编卡或 CPU 夹具里验证的问题。

### 14.1 评测表（先填字段，再看百分数）

- [LIBERO](https://arxiv.org/abs/2306.03310) 与 [OpenVLA](https://arxiv.org/abs/2406.09246) Appendix E Table 12：带着问题：76.5% 是不是零样本？四套件能不能只报平均？$N$ 是多少？读完写出：C5，fine-tune，suite 分列，500 trials × 3 seeds，禁止当真机。HTML：LIBERO [v2](https://arxiv.org/html/2306.03310v2)，OpenVLA [v3](https://arxiv.org/html/2406.09246v3)。
- [SIMPLER](https://arxiv.org/abs/2405.05941)：读 Table I。带着问题：$r=0.924$ 的分母是政策还是 trials？VisMatch 和 VarAgg 能否混成一个 sim-to-real 百分数？读完把该行单位写成 pearson_r，课桶 31 / 47 C6。HTML：[v1](https://arxiv.org/html/2405.05941v1)。
- [CALVIN](https://arxiv.org/abs/2112.03227)：复习第 31 课 Fig. 8。带着问题：短程 53.9% 和五步 0.08% 能不能和 LIBERO 平均兑？读完把它留在 C5 内部的长程子桶，不单开新课。
- [第 47 课](47_eval_taxonomy.md) 论文节的 MMMU / Video-MME / OmniBench / OSWorld：带着问题：新 Omni 论文若只报一个均分，缺了哪几张卡？读完用本课准入函数把均分判 `reject_incomplete` 或 `reject_illegal`。

### 14.2 机制（可命名，才能 $\rho=1$）

- [OpenVLA-OFT](https://arxiv.org/abs/2502.19645)：读摘要与 Table I。带着问题：97.1% 相对 76.5% 改了哪四件结构？26× 是不是成功率？ALOHA +15% 是不是 C5？读完拆成至少三张卡：C5 机制、吞吐、真机。项目页：[openvla-oft.github.io](https://openvla-oft.github.io/)。
- [第 25 课](25_action_tokenization.md) 与 [第 27 课](27_autoregressive_vla.md)：带着问题：256 bin 覆盖词表最后 256 个 token，是不是已经够收编 OpenVLA 的动作表示？答案：够。新论文若只改 bin 数，仍是 25 / 27，不新开课。
- [第 28 课](28_flow_matching_vla.md)：带着问题：$\pi_0$ 的流匹配若只作为 OFT Table I 的对照行，课桶填谁？答案：机制对照可以填 28，LIBERO 数字仍填 31。两张卡。
- [CLIP](https://arxiv.org/abs/2103.00020) 与 [LLaVA](https://arxiv.org/abs/2304.08485)：带着问题：4 亿图文对和两阶段解冻，哪一句是规模、哪一句是机制？读完写进第 21、22 课已有桶，作为「老论文也要一行一卡」的练习。

### 14.3 规模与复现纪律

- OpenVLA 摘要与 §5.1：带着问题：+16.5% 的 $N$ 是 29 任务上的 170+60 rollouts 量级，还是 LIBERO 的 500 trials？读完标 `scale`、`real_robot`、$\rho=0$。
- [第 01 课](01_baseline_reproduction.md) 第 13 节与第 19 节：带着问题：golden case 分桶和本课收编卡是什么关系？答案：01 是自训可追溯，60 是文献可追溯。两层都要。把 Table 12 写进 `baseline-v1` 主表，两边同时失败。
- [Qwen2.5-Omni 技术报告](https://arxiv.org/abs/2503.20215)：读 §5.1.3–5.1.5。带着问题：作者有没有把 MMMU、Video-MME、OmniBench 放进同一小节平均？没有。读者能不能仍然兑成一个数？能。本课的禁则写给做幻灯片的人，也写给以后想为 Qwen3 新开课的人：分节已经够，缺的是读者侧的收编卡。

读完材料回头看：模型一行权重都不必改，评测却已经从「再开一课追新 SOTA」变成「一行一卡、缺字段拒收、规模接到旧课」。同一张幻灯片上的 76.5% 和 97.1%，差的可以是动作块，也可以只是你把 Long 藏进了平均。你现在应该能拿着任意一篇新 VLA 的总表，在五分钟内给每个格子打上课桶、C1–C6 或真机、规模或机制，并决定缺 $N$ 的那一格能不能进主表——不能。

把这条检查写进实验记录第一页，比把新模型名字写进目录更有用。读者应能独立复述四句：缺 $N$、缺套件拒收；LIBERO 平均不是真机；规模不新开课；机制才谈缩小版方向。能复述这四句，CPU 检查全为真，Lab 预测选中三张标红且陷阱格最终没有 LIBERO，本课就收工。交互实验还要留下一次标红记录或书面说明「理解三种标红」。三项齐了，再去碰公开评测脚本也不迟。公开脚本只能检验同一套字段在真实输出 JSON 上是否还能对上，不能回头否定夹具里已经核对过的拒收规则。字段是协议口径，榜上的百分数是别人的推理口径，两套数字不要写进同一格。

本课停在收编：新论文接到可执行的验收口径上。目录不必为下一个 7B 名字涨一课。过时的是表值，不过时的是卡。若下一篇论文真的引入了现有课表达不了的机制，那是扩课提案的事：先写机制名字、缩小版自变量、失败判定，再申请新编号。提案在字段齐全之前不应进入目录。本课把这条申请门槛写进验收，就是为了让目录的增长变慢，让口径的复用变快。
