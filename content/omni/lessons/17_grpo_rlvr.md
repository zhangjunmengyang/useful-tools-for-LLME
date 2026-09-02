---
id: 17_grpo_rlvr
title: "多模态 GRPO / RLVR"
summary: "让程序当判分员来训练 GRPO，模型在没练过的任务上正确率真会涨吗？你怎么证明 reward 上涨不是靠凑格式、背模板或者钻验证器的空子？"
unit: alignment
play_tools: []
checkpoints:
  - "亲手推导 group-relative advantage（组内相对打分）、clipped objective 和 KL 约束。"
  - "给 OCR、表格、grounding、ASR 和时序任务各写一个带版本号的 verifier。"
  - "盯住 zero-variance group、pass 分布、entropy、答案长度漂移和 rollout 成本。"
  - "用 public/hidden 双套 verifier、没见过的题目模板和人工逐 case 审计，揪出 reward hacking。"
---

# 第 17 课：多模态 GRPO 与 RLVR

本课先在纯文本和数学题上搭一个可重放的 GRPO 训练回路，把每个环节跑通、留下证据，再把同一套方法原样迁移到多模态任务。验收时，模型的能力增量必须同时通过隐藏测试、替代验证器和逐例人工审计；训练曲线上的 reward 只是一项中间观测，单独拿它说话不算数。

## 1. 奖励上涨不等于能力上涨

第五幕已经走了两课。第 15 课联合 SFT，老师把标准答案摆在面前，模型照着学；第 16 课 DPO/MPO，老师不给标准答案，给"这份好、那份差"的排序，模型学会把好回答的概率抬高。两者有个共同点：教材在训练前就印好了。模型此刻自己会犯什么新错、会走什么歪路，印好的教材管不着。

拆掉预制教材，让模型自己下场做题。同一道题让当前模型连做 G 遍（比如 8 遍），交给一个可执行的判分程序打分，然后组内互相比较：比同组平均分高的回答，生成它的每一步都提高概率；比平均低的降低概率。模型更新后，下一轮做题的水平变了，"教材"（它自己的答案）跟着变。这种拿当前模型自己的输出来训练的方式叫 on-policy。两个零件名：GRPO（Group Relative Policy Optimization，组内相对策略优化：不训练一个专门估分的第二个网络，直接拿同组其他回答的平均分当参照物）和 RLVR（Reinforcement Learning with Verifiable Rewards，可验证奖励强化学习：判分员是确定性程序，而非学出来的打分模型）。

强化学习的老毛病是：你奖励什么，模型就钻研什么，包括钻研判分员本身。用神经网络当判分员，模型的梯度上升等于对它做对抗搜索，总能找到"打高分但不对"的输出。判分程序没有"看起来像对"的模糊地带，想骗它只能钻实现 bug，而 bug 可测试、可修；这是本课最重要的直觉，第 6.3 节讲透。另外，不上这课，第 18 课读 Nemotron 官方 recipe 时，后训练最后两个阶段 Text GRPO、Vision GRPO 就是黑盒。还有一笔经济账：reward hacking（奖励投机：模型钻判分漏洞，分涨了能力没涨）这种事，先在 26M 的便宜模型上被坑一次，比以后在几十亿参数的训练上被坑便宜得多。

打开自己的训练面板，你能亲眼看到训练 reward 一路上涨、隐藏测试纹丝不动的"钻空子现场"，并顺着审计流程抓到那批高分错答；也能随手抽一条 rollout 账本，手工复算那一步的 reward、advantage 和 ratio，跟训练日志逐位对上。

本课术语：

| 术语 | 简要解释 |
|---|---|
| GRPO | 同题多答、组内比较的更新方法：组平均分当基准线，省掉专门的估分网络 |
| RLVR | 可验证奖励强化学习：判分员是确定性程序（单元测试、数值容差、区间重叠），而非打分模型 |
| policy | 正在训练的模型本体；old policy 是采样这批回答时的权重快照；reference 是第 0 步冻结副本，算 KL 用 |
| rollout | 模型对一道题实际生成的一条回答，连同 seed、log-prob、reward 一起记账 |
| critic（价值模型） | PPO 里专门估"这个局面值多少分"的第二个网络；GRPO 用组平均分把它省了 |
| advantage（优势） | 这条回答比同组平均好多少，折算成标准差的倍数 |
| clip（裁剪） | 新旧模型的概率比超出 1±ε 后不再给更多梯度收益，防一步迈太大 |
| KL | 当前模型与 reference 输出分布的偏离程度；加进 loss 里防训练跑飞 |
| verifier（验证器） | 判分程序：读模型输出和题目规格，返回结构化分数与状态码 |
| reward hacking（奖励投机） | 模型找到判分漏洞，训练分上涨、真实能力原地踏步 |
| pass@k | 每题独立生成 k 个答案，至少一个通过验证就算过 |
| 熵塌缩 | 输出分布过早变尖，全组回答长得一个样，组内比较失去信号 |

## 2. 本课要解决的问题：GRPO/RLVR 与 DPO/MPO 的差别

先分清两条训练路径。第 16 课的 DPO/MPO 读取预先固定的较好回答和较差回答（忘了的话回[第 16 课](16_multimodal_preference_optimization.md)）；本课的 GRPO 是"边生成、边评分、边更新"：让当前模型对同一个问题现场生成一组新回答，再由可执行的验证程序评分。模型更新后，下一轮回答也会变化。代价随之而来：生成参数、题目难度和验证程序的缺陷，都会直接写进训练结果——出题人和判卷人的每一个疏漏，模型都会主动找出来利用。

实验分两步推进。第一步只使用答案可以自动核对的文本和数学题，完整运行一次生成、评分、计算相对得分、更新参数和恢复 checkpoint。第二步沿用已经通过单元测试的训练代码，依次接入 OCR、表格、目标定位、ASR、视频时序和音视频匹配任务。每次只新增一种模态，出了问题才能分开定位模型、媒体预处理和验证程序哪一层坏了。

开始训练前，先画出一轮迭代的时间线，共六个阶段：出题、生成 G 个新回答、自动评分、算组内相对得分、更新模型、同步权重。先做小规模预实验，再用日志确认六个阶段都留下了可重放的记录。任一阶段缺少输入 hash、版本或耗时，实验都不能复现。

## 3. 要验证的结论与失败条件

研究问题需要在训练前写成可检验的比较——先写判据再看数据，防的是人看过结果后不自觉地挑有利口径。比较是：在 B/C 两组的采样总数、生成 token 和 policy update token 均相同的条件下，组合验证器的 GRPO 能否比只使用结果奖励的 GRPO 获得更高的未见任务正确率，并把增量保留到未见模板。

主指标固定为 B/C 在最终测试集上的宏平均 `pass@1` 差值。测试题和答案在 checkpoint、processor、生成参数与 verifier 全部冻结前不可访问：

$$
\Delta_{\text{test}}
=
\operatorname{macro\ pass@1}(C)
-
\operatorname{macro\ pass@1}(B).
$$

宏平均的计算方法是：先分别计算每类任务的 `pass@1`，再对任务类别取平均，避免样本多的任务决定总分。开训前还要写下最小有意义效应 $\delta_{\min}$，单位为百分点。这个值表示"提升至少多大才值得采用更复杂的 C 组奖励"，应根据任务误差和使用成本确定，不能看过结果再修改。bootstrap 是通过重复抽样估计结果波动范围的方法。本课使用成对的分层 bootstrap 计算 95% 置信区间：每轮先从三个 seed 中重复抽样，再在抽到的每个 seed、每类任务内成对抽取 B/C 的相同题目，最后对任务取宏平均。抽样轮数、随机种子和缺失样本规则一并预注册。

训练前登记四项假设：

1. 同组回答之间的相对得分可以提供更新信号，不需要再训练一个独立价值模型；
2. 结果正确奖励配合严格解析和证据奖励，可以减少只改格式或猜答案的情况；
3. 多个独立验证程序和隐藏测试可以发现模型利用评分漏洞的问题；
4. 先跑通文本任务，再逐项加入多模态任务，比一开始混合所有任务更稳定。

先在 `hypotheses.yaml` 中写下 $\Delta_{\text{test}}$、$\delta_{\min}$、置信区间算法、预算、seed、停止条件和统计口径。判定规则也要提前固定：如果训练奖励上升，而隐藏测试验证器、人工审计和未见模板指标没有同步改善，就把结果归类为奖励投机，不能声称任务能力提升。验收时提交该文件的 hash，并确认报告中的每个主要结论都对应一条预先写下的假设。

## 4. 固定训练起点：准备 step-0 权重

RL 只能放大模型已经采得出来的行为：advantage 给的是"组内谁更好"的相对加分，如果起点模型一条对的都采不出来，组里就没有可加分的对象。所以起点 checkpoint 直接决定小规模预实验能否产生有差异的组内奖励。按以下优先级选择：

1. 第 16 课的 `mdpo-c0-image-v1`；
2. 第 15 课的 `joint-sft-v1`；
3. 任一固定的 MiniMind-O/Omni SFT checkpoint。

目录：

```text
checkpoints/exp17_start/
├── policy/
├── tokenizer/
├── modality_connectors/
├── frozen_reference/       # exact step-0 copy
└── baseline_metrics.json
```

选定后，将 policy、tokenizer、模态 connector 和 step-0 reference 放入上述目录。`frozen_reference` 必须是 policy 在第 0 步的逐文件副本，不能在训练中跟随 policy 更新——它是后面算 KL 时的"出发点坐标"，坐标自己会动的话，偏离度就没意义了。

`pass@k` 表示每道题独立生成 `k` 个答案，只要其中至少一个通过验证，这道题就计为通过；`pass@1` 只生成并检查一个答案。

启动前逐项验证：

- 能稳定生成结构化最终答案；
- 在预实验任务上 pass@k 既非全 0 也非全 1；
- 可对新生成回答的 token 重新计算训练模型与参考模型的 log-prob；
- generation 和 training tokenizer 完全一致；
- media preprocessing 可复现。

用 100–500 个 prompt 做固定 seed 的 `pass@8` 预检。若结果低于 5%，先增加 SFT/cold start；组内没有正样本时，所有回答通常会得到相同奖励，归一化后的 advantage 接近 0——采样算力全花了，更新信号是零。将 checkpoint 文件 hash、固定样例输出和 baseline 指标一起保存，重新加载后应逐项一致。

## 5. 学完后应能完成

本课不要求背诵框架 API。完成实验后，应能根据公式、数据记录和训练曲线解释一次 GRPO 更新，并独立完成以下工作：

- 推导组内相对优势和带裁剪的模型更新目标；
- 解释使用当前模型采样、使用旧模型采样、KL 和熵之间的关系；
- 为多模态任务编写确定性、带版本且可单测的验证程序；
- 设计等价答案归一化而不放宽到可作弊；
- 发现奖励投机、格式投机、长度漂移和熵塌缩；
- 区分训练奖励、公开验证程序和隐藏验证程序；
- 计算回答生成、评分和参数更新的系统成本。

验收采用口头推导与产物检查两部分。先不看代码写出一轮 GRPO 的输入、shape 和更新时间线，再从任意一条 rollout ledger（逐条采样账本）复算 reward、advantage 与 ratio。复算结果和训练日志一致，才说明这些目标已经掌握。

## 6. 原理:边造边讲

三个机制，每个按同一节奏走：为什么需要（直觉）、怎么运转（机制）、精确定义（数学）、在哪实现（代码落点）、怎么证明做对了（验证）。

### 6.1 组内相对优势

一条回答拿了 1 分，是好是坏？没法单独回答——如果这道题人人都能拿 1 分，它不该被表扬；如果这道题几乎没人做对，它就该被重重表扬。所以强化学习需要一条基准线。PPO 的做法是训练一个 critic（价值模型：专门估"这个局面正常水平多少分"的第二个网络），成本几乎和 policy 本身一样大。GRPO 的观察是：同一道题反正要采 G 条回答，组平均分就是现成的基准线，免费，而且天然匹配这道题的难度。类比：按名次给分的考试——你的得分由你比同考场平均高出多少个标准差决定，卷子难不难被平均分自动吸收掉了。类比失效处：考场只考一次，GRPO 每轮都重考，且"同考场考生"是同一个模型的 G 个平行采样，模型进步后整个考场水平一起涨。

对 prompt `x` 采样 `G` 个回答，奖励记为 `r_i`。先计算组内均值和标准差，再把每个回答的奖励转成相对 advantage：

$$
\begin{aligned}
\bar r &= \frac{1}{G}\sum_{j=1}^{G}r_j, \\
s_r &= \sqrt{\frac{1}{G}\sum_{j=1}^{G}(r_j-\bar r)^2}, \\
A_i &= \frac{r_i-\bar r}{s_r+\varepsilon}.
\end{aligned}
$$

对一个 batch，reward 的 shape 通常是 `[B, G]`，归一化后的 `A` 仍是 `[B, G]`。训练 token 展开后，每条回答的标量 `A_i` 会广播到该回答的有效 policy token；padding、prompt token 和被 mask 的 token 不参与 policy loss。

若组内奖励全相同，`s_r=0`，所有 advantage 都接近 0。这样的组消耗了采样算力，却几乎不提供更新信号——第 4 节的 pass@8 预检防的就是这种"全场同分"。先用 `r=[0, 0, 1, 1]` 手算 $\bar r$、$s_r$ 和四个 $A_i$（忽略 $\varepsilon$ 时应得均值 0.5、标准差 0.5，advantage 依次为 -1、-1、+1、+1），再与训练代码的输出比较。随后监控：

- zero-variance group ratio；
- 每组 pass 数分布；
- group size；
- reward mean/std；
- 有效 policy token 数。

本课不绑定具体训练框架；对应物是第 15 节伪代码里的 `group_normalize` 一行，以及第 14 节参考配置的 `advantage_normalization: group`。落地时在你所用框架的训练器里找到这两处，把上面三行公式逐行对上。

手算与实现的最大绝对误差低于预设浮点容差；把四个奖励全部改为 0 后，advantage 和该组 policy loss 均应接近 0。

### 6.2 Policy 更新

上一节得到的是回答级 advantage，但参数更新发生在 token 级：一条好回答里的每个生成 token 都平摊这份功劳（标量广播）。还有一个时间差问题——回答是用采样那一刻的旧权重生成的，等轮到梯度更新时权重可能已经变了。概率比 ratio 就是校正这个时间差的汇率；clip 再给这个汇率设上下限，防止对一批旧数据反复压榨、一步更新迈得太大。类比：拿上个月的顾客问卷调整这个月的菜单，得先掂量口味已经漂了多少；漂得太多的意见就不再加码采纳。类比失效处：clip 并没有把 ratio 硬截成常数，loss 里取 min 的写法只在"会让更新更激进"的方向封顶收益，保守方向不受限。

对第 `i` 条回答的第 `t` 个有效 token，当前 policy 与采样时 old policy 的概率比为：

$$
\rho_{i,t}(\theta)
=
\frac{
\pi_\theta(y_{i,t}\mid x,y_{i,<t})
}{
\pi_{\theta_{\mathrm{old}}}(y_{i,t}\mid x,y_{i,<t})
}
$$

$$
\mathcal L_{\text{policy}}
=
-\mathbb E_{i,t}\!\left[
\min\!\left(
\rho_{i,t}A_i,\,
\operatorname{clip}(\rho_{i,t},1-\varepsilon,1+\varepsilon)A_i
\right)
\right]
$$

`rho=1` 表示当前 policy 尚未偏离采样 policy。clipping 把可利用的 ratio 限制在 `1-\varepsilon` 到 `1+\varepsilon` 附近，减少单个 batch 引起的过大更新。目标函数外还要加入 KL 或 reference penalty；具体 estimator 以所用 GRPO 框架为准。

伪代码（第 15 节）里的 `ratio = exp(logp - old_logp)` 与 `policy_loss` 两行；clip 范围来自参考配置的 `clip_range: 0.2`，KL 系数来自 `kl_coef: 0.02`。

先导出一个最小 batch 的 `old_logp`、`new_logp`、ratio、token mask、advantage 和逐 token loss。每个张量的形状都要与有效生成序列对齐，prompt 与 padding 位置的 loss 必须为 0。报告还要写清：

- advantage 是 response-level 还是 token broadcast；
- KL 使用 estimator；
- old policy 更新频率；
- rollout 是否完全 on-policy；
- truncated response 如何计 reward。

验证分两步。先令 current policy 等于 old policy，检查 ratio 的均值接近 1；再人为增大一个正 advantage token 的 `new_logp`，确认未触发 clipping 时 loss 朝有利方向变化，超过 clip range 后增益停止扩大。

### 6.3 RLVR

这一节回答本课最核心的问题：为什么"可验证奖励"比"学出来的奖励模型"更不容易被钻空子。先摆清处境：强化学习里的模型是个全力搜索的对手。你给它任何打分函数，它的全部梯度算力都用来寻找"这个函数在哪里给高分"，而非"怎么把任务真正做好"——两者恰好重合时，训练才有效。

判分员有两种做法。第一种，学出来的 reward model：拿一个神经网络在有限的偏好数据上拟合"人觉得好"的分数。它的问题有三层。其一，它只是训练分布内的近似，分布外大片区域的打分是外推猜的，而 policy 的梯度上升恰恰会系统性地把输出推向"分数虚高"的区域——模型撞上漏洞并非小概率事件，它是在全功率朝着漏洞搜索。其二，它的错误面是连续的：分数是个光滑函数，哪里坡度虚高，顺着爬就行，漏洞是一整片斜坡，堵不完。其三，它是移动目标：policy 分布一变，reward model 就更加偏离自己的训练分布，要么不停重训它，要么看着它被越钻越深。

第二种，写出来的 verifier：一个确定性程序，按硬判据判定——数值在容差内相等、单元测试通过、bbox IoU 不低于 0.5（IoU：交并比，两个框或区间的重叠部分占并集的比例）。硬判据没有可以讨好的模糊地带：答案对就是对，写得再自信流畅也多不了一分。剩下的攻击面只有实现 bug——输出里塞多个 final、用 NaN 骗过数值比较、把正确答案写在推理过程里而 final 写错。关键差别在漏洞的形状：这些 bug 是离散的、可枚举的、可写成回归测试的，修一个少一个；reward model 的漏洞是连续曲面上的坡，修补永远追不上搜索。类比：reward model 像一位会被话术打动的阅卷老师，答案写得漂亮就容易松手给分，而且模型专职研究这位老师的偏好；verifier 像自动判题机，只看输出对不对。类比失效处：判题机也有 bug，RLVR 只是把"被钻空子"从必然事件降级成可发现、可修复的工程事故——所以 Step 2 的攻击测试和隐藏测试仍是本课硬性步骤，不能因为用了 verifier 就省掉。

这套做法的代价也要说清：verifier 只覆盖答案可程序判定的任务（数学、代码、OCR 数值、区间、WER）。开放式的文风、有用性没有硬判据，verifier 判不了——这正是第 16 课偏好优化不被本课取代的原因，两条路线分工不同。DeepSeek-R1 在大规模 RL 中选择规则奖励、明确不用神经奖励模型，给出的理由与本节相同（见第 24 节）。

RLVR 指 Reinforcement Learning with Verifiable Rewards。这里的 verifier 是确定性程序：它读取模型输出和题目规格，返回结构化 reward 与状态码。它不负责评价文风，也不读取训练中不可见的 hidden case 内容。

一个可用于训练的 verifier 必须满足：

- verifier 输入/输出 contract 固定；
- 同一答案重复运行得同一分；
- 正负单元测试齐全；
- 训练不可访问 hidden cases；
- 等价答案规则预注册；
- verifier 失败与答案错误分开记。

伪代码（第 15 节）里的 `verifier_pool` 与 `retry_once_then_drop_failed_groups` 两行；verifier 本体按第 8 节的任务表逐类实现，攻击测试在第 13 节 Step 2。

先为一个数值 JSON 任务实现正例、等价表达、错值、非法 JSON、NaN、超时和验证器异常测试。相同输入重复执行 100 次，输出必须逐字节一致；异常要记录成独立状态，不能被折叠为答案错误或满分。LLM 裁判可以辅助人工检查，但不作为本课的核心奖励。

## 7. 数据字段：完整保存题目与采样结果

环境 contract 要解决两个常见问题：训练样本无法追溯，以及 rollout 只保存文本、无法重算奖励——账本上只有结论没有过程，审计就等于没做。每个 prompt 都要保存媒体身份、答案规格、verifier 版本、任务类型、难度、来源和 split。示例 manifest 如下：

```json
{
  "problem_id": "ocr_total_00087",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "asset_id": "img0"},
        {"type": "text", "text": "给出收据总额，只输出 JSON：{\"total\": number}"}
      ]
    }
  ],
  "assets": [{"asset_id": "img0", "uri": "sha256://...", "type": "image"}],
  "answer_spec": {
    "type": "json_numeric",
    "field": "total",
    "gold": 37.5,
    "tolerance": 0.001
  },
  "verifier": {"id": "json_numeric_v2", "hidden_tests": true},
  "task": "ocr_total",
  "difficulty": 3,
  "source": "licensed-source",
  "license": "explicit",
  "split": "train"
}
```

一次采样产生一条 rollout 记录。`policy_step` 表示生成该回答时的权重版本，`sample_id` 区分同组回答，`rewards` 保留各分项，token 数与结束原因用于成本和截断分析：

```json
{
  "problem_id": "ocr_total_00087",
  "policy_step": 120,
  "sample_id": 3,
  "seed": 99103,
  "response": "<think>...</think><final>{\"answer\":{\"total\":37.5},\"evidence\":[{\"asset_id\":\"img0\",\"kind\":\"bbox\",\"value\":[0.61,0.78,0.84,0.86]}]}</final>",
  "rewards": {"outcome": 1.0, "format": 1.0, "evidence": 0.5},
  "verifier_status": "ok",
  "prompt_tokens": 1832,
  "response_tokens": 92,
  "finish_reason": "stop"
}
```

若任务使用 evidence reward（证据奖励：要求模型指出答案出自媒体的哪个位置），模型输出必须包含可解析的 evidence，而不能只在训练数据中保存一段说明。统一使用下面的结构：

```json
{
  "answer": {"total": 37.5},
  "evidence": [
    {"asset_id": "img0", "kind": "bbox", "value": [0.61, 0.78, 0.84, 0.86]}
  ]
}
```

`kind` 只允许 `text_span`、`bbox` 或 `timestamp_ms`。文本 span 使用预处理后文本的 `[start, end)` 字符区间；bbox 使用归一化的 `[x1, y1, x2, y2]`；时间证据使用相对媒体起点的 `[start_ms, end_ms]`。gold evidence 必须来自数据集原始标注或双人复核，并在 manifest 中保存来源，不能由当前 policy 自己生成——让被考的人自己出参考答案，证据奖励就形同虚设。匹配规则在训练前固定：`bbox IoU >= 0.5`、`timestamp interval IoU >= 0.5`，文本 span 使用 exact span 或预注册的字符重叠阈值。没有经过核验的 gold evidence 时，该任务的 evidence 权重必须设为 0。

写一个字段校验器，分别从题目清单和采样记录中抽取 100 条，并用 `problem_id + policy_step + sample_id` 检查唯一性。再从 `assets.uri` 重新加载媒体，调用固定版本的验证器。重新计算的各项奖励应与账本一致。媒体无法加载、版本号缺失或主键重复时，数据检查必须失败。

## 8. 奖励计算规则：为任务配置验证器

任务库的难点不在模态数量，而在 gold 能否被程序稳定判定。每个任务都要明确答案空间、归一化规则和失败状态。候选任务如下：

| 模态 | 任务 | verifier |
|---|---|---|
| text | 数学/代码/逻辑 | exact、unit tests |
| image | OCR、计数、bbox | normalized exact、IoU |
| document/chart | 字段/数值 | schema + tolerance |
| audio | ASR、关键词、时长 | WER/CER、interval IoU |
| video | 事件顺序、计数 | ordered labels、timestamps |
| audio-video | 同步/错配 | binary/offset tolerance |

最终至少选择四类任务，但 pilot 只接入其中一类。开放式描述质量缺少确定性判定规则，不适合作为主 reward；如果保留，只能进入人工审计——这正是 6.3 节划出的能力边界。

先为每类任务各写 20 个验证器单元测试，再用起点 policy 生成 100 个回答。检查答案分布、解析失败率和错误通过的样例。只有单元测试全部通过、人工复核未发现高分错答的任务，才能写入训练清单。

## 9. 数据来源、许可与复现范围

数据边界决定实验能否公开复核。可使用的来源包括：

- 可程序生成且能留出隐藏模板的任务；
- 有明确 license 的 OCR/文档/图表/媒体；
- 自己采集并可授权的数据；
- 公开 benchmark 的 train split，前提是 license 允许训练且 test 严格隔离。

每条数据还必须记录：

- problem generator 版本和随机种子；
- verifier 代码 commit；
- gold 的产生/人工核验；
- asset license；
- train/public-dev/hidden-dev/test 划分；
- contamination 筛查。

Nemotron 等论文公开了 GRPO 阶段与部分 recipe，但没有因此公开全部训练环境、reward、数据混合和内部 hidden tests。本课复现的是公开算法在自建环境中的训练动力学，报告不能使用 `完整复现官方 RL` 这一表述。

为数据清单生成许可和来源汇总表，并检查训练集、公开开发集、隐藏开发集和测试集之间的近重复与模板污染。每个集合随机抽取 50 条，确认媒体可读、标准答案可重新计算、许可字段非空。来源无法确认或泄漏到测试集的样本要先隔离，再启动训练。

## 10. 三档规模：从 pilot 到 full

| 档位 | prompts | rollout | 用途 |
|---|---:|---:|---|
| pilot | 2k–10k | G=4，1k updates 内 | 验证端到端流程 |
| standard | 50k–300k | G=8，三臂三 seed | 机制结论 |
| full | 0.5M+ | G=8/16 | 扩展到更大模型 |

三档规模对应三种问题。pilot 检查软件、数据和 reward 是否连通；standard 才用于比较训练机制；full 只讨论更大模型上的扩展。不能用 pilot 的偶然曲线支持能力结论，也不应在 standard 未通过时直接扩大数据。

任务难度应让起点 `pass@1≈10%–60%`。成功率接近 100% 时，大多数组内奖励没有方差；成功率接近 0 时，又很难采到正样本——两头都会撞上 6.1 节的零方差陷阱，只有中间地带有训练信号。先对 1k 个 prompt 各生成 16 个固定顺序的回答，再分别取前 4、8、16 个估算三种 group size 的信号与成本。验证表至少报告 pass@1、pass@k、每组通过数和零方差组比例。

## 11. 三个对照实验组

| 臂 | 训练 | reward |
|---|---|---|
| A：no-RL control | 等 token 的 chosen/self-SFT | 正确样本 NLL |
| B：outcome GRPO | GRPO | outcome only |
| C：robust GRPO | GRPO | outcome + strict format + evidence |

三个实验组回答不同问题。A 衡量 step-0 自生成正确样本继续 SFT 的收益——它是"不做 RL，只把自己做对的题再学一遍"的对照，B/C 的增益必须先赢过它才算 RL 的功劳；B 测量只用结果奖励进行在线更新的效果；C 测量更严格奖励规则的增量。公平控制包括：

- 同一起点；
- B/C 使用相同的 prompt 顺序、采样总数和生成 token 预算；
- B/C 使用相同的 group size、温度和最大生成长度；
- 相同 policy update token 和 trainable modules；
- A 只使用**训练开始前**由 step-0 policy 生成并冻结的正确样本池，报告其实际 token；
- A 的一次性建池成本单独报告，不要求与 B/C 的在线采样数相同；
- 相同三 seed；
- 同一 `public-dev`、`hidden-dev`，以及冻结后一次性运行的同一 final test。

`G=4/8/16` 不再展开为额外主实验组。先在预实验中选择 `selected_group_size`，并在查看测试集结果前冻结。训练开始后，检查三组的起点 hash、更新 token、seed 与评测集合；B/C 还要检查 prompt 顺序和生成 token。任一应当一致的项目不一致，都要停止直接比较。

### 11.1 A 组冻结候选池的规则

A 臂使用的共同样本只能来自 step-0 pool。B/C 在训练中产生的 on-policy 轨迹依赖各自不断变化的 policy，若回流 A，会把在线训练信息泄漏到 control——对照组一旦沾上实验组的数据，三臂比较就废了。每个 seed 按以下时间线构建冻结池：

1. 冻结 `policy_step0_sha`、prompt manifest 与 sampler；
2. 在任何 optimizer step 之前，对全部 train prompts 按 `selected_group_size` 采样；
3. 保存每条 response、logprob、media hash、verifier version 和 reward；
4. 只把严格 verifier 通过的 response 写入 `control_pool_seedXX.jsonl`；
5. 对该文件做 SHA256，A 的整个训练期间只读；
6. 若正样本不足，用同一 step-0 policy 增加预注册的第二轮采样，不能借用 B/C 后期 rollout；
7. A/B/C 的 policy update token 相同；rollout 生成成本另列，不能假装 A 是 on-policy RL。

因此，A 测量 step-0 自生成正确样本继续 SFT 的效果，B/C 测量在线 group-relative update 的效果。

每个 `control_pool_seedXX.jsonl` 必须有对应的 `control_pool_seedXX.manifest.json`：

```json
{
  "seed": 17,
  "policy_step0_sha256": "sha256:...",
  "prompt_manifest_sha256": "sha256:...",
  "sampler_config_sha256": "sha256:...",
  "tokenizer_sha256": "sha256:...",
  "media_registry_sha256": "sha256:...",
  "verifier_code_commit": "...",
  "verifier_config_sha256": "sha256:...",
  "verifier_attack_suite_sha256": "sha256:...",
  "generation_environment_digest": "sha256:...",
  "group_size": 8,
  "sampling_rounds": 1,
  "generated_responses": 400000,
  "accepted_responses": 73124,
  "pool_file_sha256": "sha256:..."
}
```

这里的 `8` 只是示例，表示预实验最终选中了 `selected_group_size=8`。如果选中 4 或 16，候选池、B/C 配置和所有清单必须使用同一个值，不能只改训练命令。

A 组启动前要生成三份候选池、三份数据清单和排序后的 `control_pool.files.sha256`。训练进程以只读方式打开这些文件，并在每个 epoch 校验 hash。缺少文件、hash 不匹配或验证器版本无法重建时，A 组不能重放，该 seed 不进入主结果表。

轨迹隔离还要做机器检查：A 的每个 response id 必须属于对应 step-0 pool，B/C 的 response id 必须带各自 arm 与 policy step。发现跨臂 id 后，该 seed 立即作废，不能只删除单条记录后继续使用。

## 12. Reward 的组成与单元测试

reward contract 规定 verifier 输出怎样转成一个标量。C 臂示例：

$$
r
=
1.0\,r_{\text{outcome}}
+0.1\,r_{\text{format}}
+0.2\,r_{\text{evidence}}
-0.2\,\mathbb 1_{\text{invalid-model-output}}
$$

各分项的含义和权重必须在训练前固定：

- outcome 是主奖励；
- format 权重不能大到让格式正确但答案错误的样本仍获正优势；
- evidence 只能由可验证 span/bbox/timestamp 给出；
- response length 不直接给正奖励；
- 模型输出无法按已公布 schema 解析时，才记 `invalid-model-output`；
- verifier exception 或 timeout 不产生 reward，也不参与反向传播；
- 每个 reward 分项分别记录。

权重设计的原则就一句：辅助奖励是给正确答案锦上添花的，绝不能大到让"答错但姿势标准"反超"答对"。

verifier 返回 `error` 或 `timeout` 时，先用同一版本重试一次。仍失败就让该 rollout 所在的整个 group 失效：记录错误、释放这组样本，不计算 advantage，也不做 optimizer update。不能只删除组内一条回答后继续，因为组大小和相对优势已经改变。日志分别报告 `invalid_model_output_rate` 与 `verifier_failure_rate`；前者是模型行为，后者是系统可靠性，混在一起就没法判断该修模型还是修系统。

先列出五类合成输入：答案与格式都对、答案对但格式错、答案错但格式对、模型输出不可解析、验证器异常。手算它们的总奖励，并确认实现输出完全一致。答案错但格式对的样本不能因为辅助奖励而获得高于正确答案的总分。最后一种输入应没有标量 reward，且不会调用 backward。训练日志必须同时保存总奖励和每个分项；只保存总分无法检查奖励投机。

## 13. 实验步骤：从最小训练回路到 final evaluation

### Step 1：建立 text/math 最小训练回路

先排除多模态预处理的干扰——文本任务出问题只可能怪训练代码，多模态任务出问题嫌疑人立刻翻倍。准备 2k–5k 道可验证 text/math 题，每个 prompt 生成 `G` 个回答，依次运行 verifier、重算 log-prob、计算 advantage 并完成一次 policy update。保存 checkpoint 后结束进程，用新进程恢复，再对同一批固定 prompt 生成回答。

这一阶段只训练一个 update 也可以，但六类中间量必须落盘：response、reward、old/new log-prob、advantage、token mask 和参数更新前后 hash。恢复后的 optimizer step、policy hash 和固定 prompt 输出应与不中断运行对齐；若不一致，先修复 checkpoint 或 tokenizer 路径。

### Step 2：对 verifier 做攻击测试

训练会主动寻找评分程序的薄弱位置（6.3 节讲过原因），因此 verifier 要在上岗前先挨一轮打。每个 verifier 至少包含：

- 正确标准格式；
- 正确等价格式；
- 错答案正确格式；
- 注入多个 final；
- NaN/Infinity/超大数；
- Unicode/空白；
- 超时/异常；
- 在 rationale 中写正确答案、final 写错。

把这些样例写成自动测试，并记录预期状态码与各项奖励。模型超时由生成器记录；verifier 超时或异常必须走 group retry/drop 路径，并断言参数 hash 不变。只要攻击样本得到错误高分，或系统故障仍触发一次更新，验证器的输入输出约定就还不可靠，必须先修复再训练。验收以测试报告和验证器版本为准，不能只展示几个通过的样例。

### Step 3：选择可学习难度与 group size

对起点执行 `1k prompt × G=16` 的离线采样。保留每个 prompt 的 16 个 reward，再从同一数据模拟 `G=4/8/16` 的 `pass_count_per_group`，避免为三个候选 G 分别采样而引入额外差异——一次采样、三种切法，比较才干净。

画出每种 $G$ 对应的组内通过数分布、零方差组比例和预计采样成本。选择零方差组比例低于 60% 且成本可接受的 $G$，再用第二组固定 seed 复核。如果分布变化很大，先调整任务难度，不要直接增大 $G$。

### Step 4：做多模态环境 pilot

每类任务准备 500–2k 个 prompt，但按模态逐个接入。新增环境时保持 policy、采样参数和训练代码不变，只替换 manifest、processor 与 verifier。这样出现回归时，可以把范围限制在新环境。

对 50 条样本保存原始媒体 hash、预处理后的张量形状、模型实际输入、标准答案来源和验证器输出。重点检查 OCR 标准答案与训练输入是否来自同一个裁剪版本。随机交换媒体后，用成对样本计算原媒体与错媒体的奖励差，并报告 95% 置信区间；如果区间包含 0，就不能断言验证器依赖媒体，应继续检查提示词或标准答案是否泄露了答案。

### Step 5：跑 A/B/C 三臂

三臂开始前，把下列参数写入同一份只读实验配置：

- 每轮 prompt 数；
- rollout temperature；
- max response tokens；
- update epochs；
- KL；
- clip range；
- trainable adapter。

每 50–100 updates 保存 policy 与完整 rollout ledger。A 从冻结 pool 按 deterministic epoch sampler 取样；B/C 只消费各自当前 policy 生成的 on-policy rollout。三条 ledger 的 `trajectory_source` 必须按下表写入，不能共用默认值：

| 实验臂 | `trajectory_source` |
|---|---|
| A | `step0_frozen_control` |
| B | `arm_b_on_policy` |
| C | `arm_c_on_policy` |

在每个保存点检查回答编号是否跨实验组，并汇总 A 组的建池 token、B/C 的累计采样 token、各组的 policy update token 和 GPU-hours。任何跨组回答编号都视为轨迹泄漏，对应 seed 作废。三组更新 token 的偏差超出预先登记的容差时，暂停训练并补齐预算后再比较。

### Step 6：监控训练动力学

训练 reward 上升本身无法说明模型学会了任务——它同样可能在说明模型学会了钻 verifier 的空子。每 50–100 updates 同步绘制：

- train/public-dev/hidden-dev reward；
- pass@1/pass@k；
- zero-variance group ratio；
- KL、entropy、clip fraction；
- response length；
- invalid/repetition/refusal；
- 每任务采样与 reward。

这些曲线还要按任务和 seed 拆开，并标出 checkpoint、权重同步和异常重启时间。公开开发集上升而隐藏开发集不升时，立即停止增加步数，抽取高奖励失败样本并运行攻击测试。恢复训练前必须确定异常来自数据难度、验证器漏洞还是 policy 崩溃。

### Step 7：做 reward ablation 与 counterfactual

这一步判断 C 臂的收益依赖哪一个 reward 分项。冻结同一个 C checkpoint 和同一批回答，离线重算：

- 去 format；
- 去 evidence；
- 更严格 outcome；
- wrong media；
- 删除必要 evidence；
- 换未见模板。

这些操作不增加主实验组，也不更新模型——同一批回答换个打分口径重算，便宜且无污染。比较原始奖励与各项消融后的样本排序、通过率和 `hidden-dev` 指标，并分别检查替换媒体、删除证据和未见模板是否改变结果。`hidden-dev` 只用于 checkpoint 选择，与 Step 8 的最终测试集完全隔离。若去掉格式奖励后能力指标不变，只能说明格式项没有提高任务正确率，不能据此删改主实验记录。

### Step 8：冻结 checkpoint 做 final evaluation

模型选择完成后冻结 checkpoint、processor、生成参数和 verifier 版本，再在训练进程完全不可访问的 test 上运行：

- greedy pass@1；
- 固定温度 pass@k；
- 规则 verifier；
- alternate verifier；
- 人工审计。

先完成全部生成，再一次性解封标准答案和隐藏验证器。greedy pass@1 与固定温度 pass@k 使用独立输出文件，备用验证器不能调用主验证器的中间结果。最后检查文件 hash、样例数、随机 seed 和缺失样本；任何补跑都要单独记录原因。

### Step 9：逐 case reward hacking 审计

聚合指标会隐藏少量高分错答。至少逐条查看：

- reward 最高但 alternate verifier 错的 30 个；
- 长度最长的 20 个；
- format-only 得分的 20 个；
- hidden 失败的 30 个；
- 错媒体仍通过的 20 个。

五组共 120 个高风险 case 需要填写统一审计表，标出 gold、媒体、primary/alternate reward、长度、模板来源和失败层。两名审计者对能力结论有分歧时保留两份判断，不强行合并。验证完成后统计 false-positive 类型；出现新的漏洞类别时，将对应 case 加入 attack suite，并重新评估已保存 checkpoint。

## 14. 参考配置：可执行的基准参数

这份配置把三类容易混淆的状态分开：`policy_init` 是可训练起点，`reference_init` 用于 KL，`control_arm.pool_manifests` 只服务 A 臂。`prompts_per_iteration=256` 与 `group_size=8` 表示每轮最多产生 2048 条回答；`global_policy_tokens=262144` 约束更新预算，不等同于 rollout token 数。

```yaml
experiment:
  name: exp17_robust_grpo
  seeds: [17, 23, 41]
model:
  policy_init: checkpoints/exp17_start/policy
  reference_init: checkpoints/exp17_start/frozen_reference
  trainable: [backbone_lora, modality_connector_lora]
rollout:
  prompts_per_iteration: 256
  group_size: 8
  temperature: 0.9
  top_p: 0.95
  max_new_tokens: 768
  engines: 4
control_arm:
  pool_source: step0_policy_only
  pool_manifests:
    17: data/exp17/control_pool_seed17.jsonl
    23: data/exp17/control_pool_seed23.jsonl
    41: data/exp17/control_pool_seed41.jsonl
  immutable: true
  allow_cross_arm_trajectories: false
objective:
  type: grpo
  clip_range: 0.2
  kl_coef: 0.02
  advantage_normalization: group
  rewards:
    outcome: 1.0
    strict_format: 0.1
    evidence: 0.2
    invalid_model_output: -0.2
  verifier_failure:
    retry: 1
    action_after_retry: drop_group_without_backward
train:
  updates: 2000
  update_epochs: 1
  lr: 5.0e-7
  global_policy_tokens: 262144
  bf16: true
eval:
  every_updates: 50
  suites: [public_dev, hidden_dev, regression, attack]
```

把示例复制成每个实验组的独立配置，解析后生成规范化 JSON 与 SHA-256。启动前打印最终生效值，重点核对 `group_size`、`temperature`、`max_new_tokens`、KL、裁剪范围和可训练模块。验收时比较配置 hash、运行日志首行和 checkpoint 元数据；三处必须一致，环境变量带来的覆盖也要写入最终配置。

## 15. 伪代码：一轮 GRPO 的执行顺序

下面的伪代码给出一轮更新的时间顺序。`policy_old` 先生成回答；verifier 对回答分组评分；训练器基于保存的 `old_logp` 更新当前 policy；完成全部 minibatch 后，再把新 policy 快照同步为下一轮的 `policy_old`。`reference` 在本实验中保持冻结。

```python
for prompts in stream:
    with rollout_engine(policy_old):
        groups = [sample(prompt, n=G) for prompt in prompts]

    verified = verifier_pool(groups)
    valid_groups = retry_once_then_drop_failed_groups(verified)
    rewards = scalar_rewards(valid_groups)
    advantages = group_normalize(rewards)

    for batch in policy_batches(valid_groups):
        logp = policy.logp(batch)
        old_logp = batch.old_logp
        ratio = exp(logp - old_logp)
        policy_loss = -minimum(
            ratio * batch.advantage,
            clip(ratio, 1-eps, 1+eps) * batch.advantage,
        )
        kl = kl_estimator(policy, reference, batch)
        loss = policy_loss.mean() + kl_coef * kl.mean()
        update(loss)

    policy_old = snapshot(policy)
```

把伪代码映射到实际框架时，在每一行旁标注函数名、输入形状、设备和是否跨进程通信。用一个 prompt、`G=2` 和一个 minibatch 做单步跟踪，确认验证器在 optimizer 之前运行，`old_logp` 来自采样权重，reference 参数没有梯度，权重同步发生在更新完成后。实际顺序不同时，要在报告中解释对应的算法含义。

## 16. 训练预算与 8 卡运行方案

### 单模型教学配置

| lane | GPU 分配 | 用途 |
|---|---|---|
| debug | 1 rollout + 1 train | 端到端调试 |
| pilot | 2 rollout + 2 train | text 与单模态 |
| standard | 4 rollout + 4 train | 8 卡异步 GRPO |
| memory-heavy | 2 rollout + 6 train | 1B–3B 或长视频 |

8 张卡无需平均划分。生成侧、训练侧和 verifier 的耗时不同，固定 4+4 之前应先分别 profile：

- rollout tokens/s；
- policy update tokens/s；
- verifier CPU/GPU time；
- queue wait；
- weight sync；
- 每个有效正 advantage 的 GPU-hours。

先在调试配置下运行 20 次迭代，记录各阶段 P50/P95、队列空闲和同步耗时，再选择预实验或标准配置。若验证器成为瓶颈，优先批量执行，或缓存与 policy 无关的确定性计算；不能缓存依赖当前输出的归一化结果。修改 GPU 分配后重新运行同一性能测试，只有端到端吞吐提高且训练数值一致，修改才有效。

### 预算

| 档位 | 预计预算 |
|---|---:|
| pilot | 50–150 GPUh |
| standard/每臂 | 300–1500 GPUh |
| 三臂三 seed | 需先按 pilot tokens/s 精算 |

预算表只给出规划区间。用预实验测得的采样 tokens/s、更新 tokens/s、验证器吞吐和同步占比，估算每个实验组、每个 seed 的 GPU-hours，并预留失败重跑的成本。

wall-clock step 相同不表示算力公平。验证预算时分别汇总 rollout 生成 token、policy update token 和 GPU-hours；预计值与实测值的差异也要写入报告。

## 17. 指标：能力、动力学与系统开销

### 能力

能力指标回答模型是否在未见数据上解决了任务，并检查结果是否依赖正确媒体：

- hidden pass@1、pass@k；
- exact/F1/IoU/WER/temporal accuracy；
- alternate-verifier pass；
- 未见模板与未见媒体来源；
- wrong/remove modality gap。

所有指标按任务、难度、模板和媒体来源分层。先用单元样例验证 exact、F1、IoU、WER 和时序准确率的实现，再对基础模型与三个实验组使用同一评测输出。备用验证器与主验证器结论不一致的样例必须进入人工检查。

### RL 动力学

动力学指标解释训练为何变化。每个 update 记录：

- mean/std reward；
- zero-variance group ratio；
- advantage 分布；
- KL、entropy、clip fraction；
- response length；
- invalid、repetition、refusal；
- policy/reference log-prob。

把奖励、KL、熵、裁剪比例和回答长度画在同一个训练步数轴上，并标注异常点。验收时还要检查分布：若奖励上升的同时熵急降、长度激增或零方差组增多，应先诊断训练退化，不能只报告最终奖励。

### 系统与回归

系统指标衡量同一能力增量付出的代价，回归指标防止窄任务训练破坏已有模态能力：

- rollout/update tokens/s；
- end-to-end iteration P50/P95；
- peak HBM；
- queue idle；
- verifier exceptions；
- GPU-hours per +1 point；
- SFT text/image/audio/video 回归。

每个数值都要附采样窗口、卡数、模型版本和计时边界。用性能分析器的阶段耗时复核日志中的 P50/P95，并在固定 SFT 检查集上比较 step-0 与最终 checkpoint。缺少基线或计时范围不一致的吞吐数字不能进入主表。

## 18. 验收条件

### 18.1 先判断实验是否有效

实验有效和 C 组胜过 B 组是两个问题。以下条件全部满足，结果才可用于比较：

- B/C 的起点、训练题目、采样预算、生成 token、更新 token、可训练参数和三个 seed 符合预注册配置；
- 隐藏测试在 checkpoint 与评测代码冻结后只运行一次，B/C 使用完全相同的题目和生成参数；
- 每个测试样本都保存 B/C 的成对结果，缺失样本按预注册规则处理；
- verifier attack suite 100% 通过，120 个高风险样例完成人工审计；
- 置信区间可由原始逐题结果和固定 bootstrap 配置重新计算；
- invalid/repetition、回答长度、SFT 回归和 zero-variance group ratio 均按预注册阈值完整报告，不能只留下有利指标。

只要这些条件满足，即使 C 没有胜过 B，实验仍然有效。课程应接受可信的负结果，因为它同样回答了研究问题。

### 18.2 再判断 C 是否胜过 B

只有同时满足以下条件，报告才能写"C 胜过 B"：

- $\Delta_{\text{test}}$ 的点估计不小于预注册的 $\delta_{\min}$；
- $\Delta_{\text{test}}$ 的 95% 置信区间下界大于 0；
- alternate verifier 与未见模板上的 C−B 差值方向一致；
- reward 上升没有伴随预注册范围之外的 invalid/repetition、回答长度或 SFT 能力退化；
- 三个 seed 的逐 seed 结果全部列出，没有因方向不利而删除 seed。

如果点估计小于 $\delta_{\min}$，或者 95% 置信区间包含 0，结论应写成"本实验没有证明 C 胜过 B"。报告仍要给出点估计、区间和逐 seed 结果，不能修改阈值、增加测试次数或只挑有利任务。只有 18.1 的实验有效性条件失败，才需要判定本次比较无效并重做。

## 19. 失败诊断：用症状定位层级

诊断要遵循固定顺序：先确认数据与 verifier，再检查训练数值，最后调整超参数——超参数是最后的嫌疑人，因为它最容易背黑锅。下表把可观察症状对应到最小检查和修复动作（MoE router 一行涉及第 11 课的路由器，忘了的话回[第 11 课](11_tiny_moe.md)）：

| 症状 | 可能原因 | 检查 | 修复 |
|---|---|---|---|
| train reward 涨、hidden 不涨 | verifier 过拟合 | alternate verifier | 隐藏模板/多 verifier |
| 输出越来越长 | 长度与得分相关 | partial correlation | cap length、去长度奖励 |
| 只输出 JSON 壳 | format 权重过大 | outcome 分项 | 降 format 权重 |
| 每组 reward 一样 | 任务太易/难 | pass count | 调难度或 G |
| entropy 快速塌缩 | LR/KL 不当 | entropy/KL | 降 LR、增 KL |
| 答案写在 rationale、final 错 | parser 漏洞 | attack tests | 只验 final |
| NaN 得高分 | 数值解析漏洞 | edge cases | finite check |
| OCR 升、通用能力降 | 任务过窄 | regression | 混合 anchor prompts |
| wrong image 仍通过 | gold/媒体捷径 | media swap | 重建任务 |
| rollout GPU 空闲 | verifier/同步瓶颈 | timeline | 异步队列、batch |
| MoE router 塌缩 | router 被 RL 放大 | expert load | 冻结 router/aux loss |
| reward 抖动大 | group/任务混合不稳 | 分任务 std | 分层 batch |

每次只实施一项修复，并保留修复前后的配置 hash、20–50 次迭代曲线和同一 `hidden-dev` case 集。症状消失且 `hidden-dev` 指标没有恶化后，才将修复合入主实验；多项改动同时发生时，无法判断原因。

## 20. 逐个样例检查：重点检查高分错答

逐例记录用于区分感知、推理、序列化和 verifier 四类失败。每个 case 至少保存输入 hash、gold、模型回答、reward 分项、替代验证结果、错媒体输出、基线输出和失败层：

```yaml
problem_id: ocr_total_00087
policy_step: 1200
input_hash: sha256:...
gold: 37.5
response: '{"answer":{"total":37.50},"evidence":[{"asset_id":"img0","kind":"bbox","value":[0.61,0.78,0.84,0.86]}]}'
reward:
  outcome: 1.0
  format: 1.0
  evidence: 0.5
alternate_verifier: pass
wrong_image_response: '{"total": null}'
baseline_response: '{"total": 57.5}'
producer_evidence: "OCR region contains 37.50"
failure_layer: null
```

审计者逐项记录：

1. gold 与媒体是否一致；
2. verifier 是否在所有等价/攻击形式上正确；
3. reward 是否依赖 outcome；
4. 模型是否使用媒体；
5. 提升来自 perception、reasoning 还是格式；
6. alternate verifier 是否同意；
7. 该模式是否出现在训练模板；
8. 有无新回归。

从账本重新加载该样例，复算主验证器与备用验证器的奖励，并重跑错媒体条件。复算结果与记录不一致时，先修复来源记录或验证器版本，不能继续归因模型能力。

## 21. 交付物目录

产物目录要支持另一台机器从配置、数据和 provenance 重放到报告。建议结构如下：

```text
artifacts/exp17/
├── configs/{self_sft,outcome_grpo,robust_grpo}.yaml
├── environments/
│   ├── registry.yaml
│   ├── verifier_versions.json
│   └── attack_tests/
├── data/
│   ├── {train,public_dev,hidden_dev,test}.jsonl
│   ├── control_pool_seed17.jsonl
│   ├── control_pool_seed23.jsonl
│   ├── control_pool_seed41.jsonl
│   ├── control_pool_seed17.manifest.json
│   ├── control_pool_seed23.manifest.json
│   ├── control_pool_seed41.manifest.json
│   └── control_pool.files.sha256
├── provenance/
│   ├── policy_step0.sha256
│   ├── prompt_manifest.sha256
│   ├── sampler_config.yaml
│   ├── tokenizer.sha256
│   ├── media_registry.sha256
│   ├── verifier_code_commit.txt
│   ├── verifier_config.sha256
│   ├── verifier_attack_suite.sha256
│   └── generation_environment_digest.txt
├── rollouts/ledger-*.jsonl
├── checkpoints/index.json
├── metrics/{rl,hidden,alternate,regression,system}.jsonl
├── cases/reward_hacking_audit.md
└── report.md
```

在干净目录运行一次产物校验器，检查路径、hash、JSON 字段、checkpoint 索引和账本引用。报告把 `训练奖励`、`公开验证器`、`隐藏验证器`、`人工检查` 分成四栏，避免用其中一项替代其他证据。

## 22. 复现清单：最终检查

清单按执行顺序核对。每一项需要链接到具体文件或日志位置，只有勾选符号没有证据不算完成。

- [ ] 起点 policy/reference/hash 固定；
- [ ] A 臂 step-0 control pool 在训练前生成、逐 seed hash 固定；
- [ ] 三份 control pool、manifest 与排序后的 SHA-256 manifest 已交付；
- [ ] step-0 policy、prompt、sampler、tokenizer、media 与 verifier provenance 可重建；
- [ ] B/C rollout id 未流入 A，三臂 trajectory source 可审计；
- [ ] pilot pass@k 在可学习区间；
- [ ] verifier 有版本和单元/攻击测试；
- [ ] hidden tests 对训练不可见；
- [ ] 三臂 rollout/update token 公平；
- [ ] group size 在 test 前预注册；
- [ ] $\delta_{\min}$、置信区间算法和 bootstrap 随机种子在 test 前预注册；
- [ ] sampling 参数全记录；
- [ ] old policy/reference 语义明确；
- [ ] zero-variance、KL、entropy、length 已记录；
- [ ] public/hidden/alternate verifier 都跑；
- [ ] 隐藏测试在 checkpoint 冻结后只运行一次；
- [ ] C−B 点估计、95% 置信区间与逐 seed 结果已报告；
- [ ] wrong/remove media 已跑；
- [ ] SFT 回归已跑；
- [ ] rollout ledger 可重放；
- [ ] 120 个高风险 cases 已审计；
- [ ] 数据/license/生成器可追。

由未参与主训练的人在新进程中抽查 checkpoint 恢复、10 条 reward 重算和 10 条媒体重载。三项均通过后，清单才可签字归档。

## 23. 前沿对照与改造方向

GRPO 的出处是 [DeepSeekMath](https://arxiv.org/abs/2402.03300)：在数学题这种答案可自动核对的任务上，用组内平均分替代 PPO 的 critic，省掉一个与 policy 同量级的网络，本课 6.1 节的公式就来自它。[DeepSeek-R1](https://arxiv.org/abs/2501.12948) 把 RLVR 推到了极限：R1-Zero 直接从基座模型开始纯 RL，奖励只有基于规则的正确性与格式两类，论文明确说明不用神经奖励模型，理由正是大规模 RL 中的奖励投机风险——6.3 节的直觉在工业规模上的同一结论；训练中模型的回答自发变长、出现重新检查的推理行为，正式版 R1 则在纯 RL 之前加了 cold-start SFT 来稳住起点。多模态方向，[Nemotron 3 Nano Omni](https://arxiv.org/abs/2604.24954) 的后训练按 SFT、MPO、Text GRPO、Vision GRPO 分阶段推进（官方 recipe 见第 24 节），先文本后视觉的 GRPO 顺序与本课 Step 1 到 Step 4 的接入顺序同构。离线偏好路线也没有被淘汰：[MPO/MMPR](https://arxiv.org/abs/2411.10442) 那类固定偏好数据的方法仍负责可验证奖励覆盖不到的开放式质量——前沿系统里两条路并行分工，与本课和第 16 课的关系一样。

规模问题（砸钱能缩小）：policy 参数量 26M 对几十亿；rollout 吞吐上前沿用成百上千卡异步采样，我们 8 卡分 4+4；任务库规模上前沿是海量数学、代码与多模态题库，我们 pilot 只有 2k–10k。其中最贵的一种规模差距是基座能力：R1 式的长链推理和自发反思是"基座里已有的行为被 RL 放大"的现象，26M 基座没有可放大的存货，这个差距加卡也追不上，只能换基座。机制问题（本课教的能解决）：verifier 工程、攻击测试、组内相对更新的全部数值机制、reward hacking 审计、三臂公平控制、hidden test 隔离与预注册——这套手艺在 26M 和 30B 上是同一套，而且小模型迭代快、试错便宜；本课做完，你在"判分员纪律"这件事上与前沿同水平。



1. **奖励设计的投机压力测试。** 把 C 臂配置里 `objective.rewards.strict_format` 从 0.1 提到 0.5，复制成新实验 ID，在 pilot 档跑 text 加 OCR 各约 2k prompt、300–500 updates、单 seed，预算约 50–80 GPUh。改动位置：第 14 节参考配置的 rewards 段与实验注册文件，训练代码不动。预期：format-only 得分样本占比上升、回答长度分布移动、`hidden-dev` pass@1 不升甚至下降——在最便宜的地方亲眼看一次"辅助奖励过重"如何制造投机。失败判定：各指标与 0.1 权重基线在预注册容差内无差异，说明该任务的格式维度已饱和，换更复杂的输出 schema 重试。
2. **组内基线变体。** 改第 15 节伪代码的 `group_normalize`，实现三个变体：默认的减均值除标准差；只减均值不除标准差；leave-one-out——每条回答的基线取组内其余 G-1 条的均值。改动位置：训练器的 advantage 计算函数与 `advantage_normalization` 配置项。在 Step 1 的 text/math 最小回路上，G=8，每变体 500 updates、单 seed，共约 60–100 GPUh。预期：除标准差的版本在"几乎全对"或"几乎全错"的组上，把很小的奖励差放大成很大的 advantage（分母 $s_r$ 趋近 0），clip fraction 与分任务 reward std 曲线出现可见差异；只减均值的版本对难度不均的任务混合更平稳。失败判定：三条曲线在预注册容差内重合，说明任务难度太均匀、实验没有分辨力，回 Step 3 重配难度再跑。
3. **把判分员换成 LLM 裁判的反面教材实验。** 在 2k 道 text 题上复制 B 臂，唯一改动：outcome reward 改由一个冻结的 LLM 裁判打 0/1 分；程序 verifier 仍离线运行但只记录、不参与训练。本课主实验明确禁止这种做法，这个实验专门做给自己看。改动位置：reward 计算入口把 verifier 调用换成裁判调用，ledger 里两种分数并存。pilot 档预算约 50 GPUh。预期：训练 reward（裁判分）上升快于 B 臂，但程序 verifier 复核的真实正确率增幅更小，并出现"裁判给分、程序判错"的样本簇——6.3 节的直觉变成你自己的曲线。失败判定：裁判与程序 verifier 全程一致，说明该任务太好判、裁判无空子可钻，换有等价表达歧义（单位、格式、多解）的任务重跑。

把论文结论映射到 26M 缩小版，哪些趋势可望复现、哪些不能，要分开写：

- DeepSeekMath"去掉 critic，组内基线足以驱动可验证任务提升"：对应 standard 档 B 臂对 A 臂的比较。方向可望复现——组内相对更新的数学不依赖模型规模；前提是 Step 3 把难度调进 `pass@1≈10%–60%` 区间，否则零方差组吃掉全部信号，复现失败的原因会是难度配置而非算法。
- DeepSeek-R1 的 R1-Zero 涌现现象（回答自发变长、出现反思式重新检查）：26M 上不可复现，原因要写进报告——RL 的 advantage 只能给"模型已经采得出来的回答"加权，26M 基座采不出长链推理，组内永远没有可加分的对象；这与第 4 节"pass@8 低于 5% 先补 SFT"是同一条逻辑，也是 R1 正式版需要 cold start 的原因。缩小版上若看到 response length 上涨，更可能是长度投机（第 19 节诊断表第二行），别把它当"涌现"写进报告。
- DeepSeek-R1"规则奖励比神经奖励模型抗投机"：改造实验 3 就是它的缩小版。方向可望复现，且小模型上更快看到分化，因为小模型更早依赖捷径而非真实能力。
- Nemotron"text GRPO 先于 vision GRPO"的阶段安排：本课 Step 1（纯文本回路）先于 Step 4（逐模态接入）就是同一结论的缩小版。可做对照复现：加一个"从第一步就混合全部模态"的 pilot 变体，预期分阶段版本更早定位故障、崩溃率更低——第 3 节登记的假设 4 就是为它准备的。
- 不可复现的还有：R1 与 Nemotron 报告的具体 benchmark 数字、蒸馏到小模型的能力迁移结论。它们依赖大规模基座与私有数据，26M 设置里没有对应物，报告不做数字对齐。

## 24. 必读论文与官方 recipe

### [DeepSeekMath](https://arxiv.org/abs/2402.03300)

精读：GRPO 目标、PPO/GRPO 比较、训练数据与数学实验。

带着三个问题读：论文的 GRPO 目标函数与本课 6.2 节的 loss 逐项对得上吗，哪一项对应参考配置里的 `kl_coef`；被省掉的 critic 有多大、这笔算力折到第 16 节的 8 卡预算表里等于什么；组归一化在全同分组上给出什么 advantage，与 6.1 节的零方差陷阱对照。阅读记录需说明：GRPO 省去的 critic 成本；group normalization 可能引入的偏差；数学 reward 适合 RLVR 的条件，以及这些条件在开放式多模态描述中为何难以满足。

### [DeepSeek-R1](https://arxiv.org/abs/2501.12948)

精读：R1-Zero、cold-start、reasoning-oriented RL、distillation。

带着两个问题读：R1-Zero 不做 SFT 直接 RL 也能起步，它的基座凭什么做得到，而本课第 4 节要求 pass@8 不低于 5% 才开训；论文选择规则奖励、放弃神经奖励模型的段落，与本课 6.3 节的论证哪些重合。阅读记录需区分纯 RL 与 cold-start SFT 的作用，并标出依赖大规模模型与算力的涌现结论。这些结论不能直接外推到 MiniMind 教学实验。

### [MPO / MMPR](https://arxiv.org/abs/2411.10442)

精读：多模态离线偏好目标与数据。

带着问题读：哪些任务用离线偏好就够，哪些必须 on-policy verifier，判断依据是什么；第 16 课与本课的可比条件（同起点、同预算）该怎么搭。阅读记录需列出离线 preference 能处理的问题、需要 on-policy verifier 的问题，以及第 16/17 课的可比条件。两种方法的结论按任务和预算陈述，不使用绝对先进性的排名。

### [Nemotron 3 Nano Omni 论文](https://arxiv.org/abs/2604.24954)

精读：post-training、text GRPO、vision GRPO、相关评测。

带着问题读：text GRPO 排在 vision GRPO 之前的理由是什么，与本课 Step 1 到 Step 4 的顺序论证是否一致。阅读记录需解释 text GRPO 先于 vision GRPO 的阶段安排，并把公开的 reward/environment 与仍属内部资产的部分分栏记录。

### [Nemotron Omni 官方 recipe](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/omni3/README.md)

精读：SFT→MPO→Text GRPO→Vision GRPO 的阶段顺序、公开配置与命令。

带着问题读：哪些步骤可以用公开数据做机制复现，哪些卡在 checkpoint、data 和 reward 的缺口上。阅读记录需标注可用公开数据做机制复现的步骤，以及 checkpoint、data 和 reward 的缺口。存在这些缺口时，报告不能声称完整复现官方训练——下一课你会亲手验证这条边界。

每篇材料提交一页结构化笔记，并把至少一个公式或配置项映射到本课代码。验证方式是引用论文页码、官方路径或 commit；只有摘要性复述不能满足精读要求。

## 25. 扩展题：后续研究

这些扩展题都从已经完成验收的主实验分叉，每次只增加一个变量：

1. 比较 router frozen/trainable 的 MoE GRPO；
2. 让难度调度器维持每组 1–G-1 个 pass；
3. 引入过程 verifier，但与 outcome reward 分开报告；
4. 使用多 verifier 交叉验证，研究 false-positive reward；
5. 把视频 timestamp evidence 纳入 reward；
6. 用 async rollout 减少 8 卡 idle；
7. 将最严重 reward hacks 自动加入 attack suite；
8. 对同一 checkpoint 比较 DPO/MPO 与 GRPO 的单位 GPU-hour 收益。

每个扩展实验都要另建 experiment ID，沿用主课的 hidden test 与审计规则。完成标准是给出变量、预算、主指标和反证结果；只展示训练曲线不算完成。

## 26. `rlvr-omni-v1` 的发布内容

通过全部验收条件后，将 checkpoint 命名为 `rlvr-omni-v1`。

必须随 checkpoint 一起发布：

- environment/verifier 版本；
- rollout 参数；
- hidden 与 alternate verifier 结果；
- reward hacking 审计；
- SFT 回归；
- 实际 GPU-hours。

发布前由 artifact validator 检查上述六类文件，并从 ledger 随机重放 10 条 reward。缺少 verifier 或 rollout ledger 的 checkpoint 不能作为 RL 能力提升成果交付。

到这里，第五幕收官。回头看训练方法三连：[第 15 课](15_joint_multimodal_sft.md)的联合 SFT 让模型照着标准答案练基本功；[第 16 课](16_multimodal_preference_optimization.md)的 DPO/MPO 用固定的好坏对教它在两份答案里挑对的；本课的 GRPO/RLVR 拆掉预制教材，让它自己做题、程序判分、组内比较，能力增量还得扛住隐藏测试、替代验证器和逐例审计。三步连起来，就是现代模型后训练的标准流水线。下一课进入第六幕毕业设计：[第 18 课](18_nemotron_finetuning.md)把舞台换成 NVIDIA 的 Nemotron 3 Nano Omni，复现官方 LoRA 微调 recipe。它的官方后训练顺序 SFT、MPO、Text GRPO、Vision GRPO，正是你刚学完的这三课在 30B 级模型上的工业版本——这一次，你读它的每个阶段都知道里面在发生什么。
