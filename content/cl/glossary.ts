export type GlossaryTerm = {
  term: string;
  alias?: string;
  definition: string;
  diagnostic: string;
  lessons: string[];
};

export function glossaryAnchor(term: string) {
  return `term-${term
    .toLocaleLowerCase("en-US")
    .replace(/[^a-z0-9\u3400-\u9fff]+/g, "-")
    .replace(/^-+|-+$/g, "")}`;
}

export const glossary: GlossaryTerm[] = [
  {
    term: "夜间巩固",
    alias: "memory distillation",
    definition:
      "白天把新事实、流程写在外挂记忆里，晚上用这批条目训练权重，再逐步少依赖日记。卸掉存储之后仍能答，才算写进了权重。",
    diagnostic:
      "跑 python3 run.py extra run distill：巩固前卸库接近 0，巩固后应能召回大多数条目。",
    lessons: ["13", "16", "24"],
  },
  {
    term: "卸库测验",
    alias: "unplug test",
    definition:
      "把外挂存储停掉或把日记字典清空，再用同一句问「小王在哪」。答不上，说明「会」写在库上；仍能答，才算写进了权重。",
    diagnostic: "跑 python3 run.py extra run unplug，卸库必须是 0。巩固后再跑 distill 或 graduate。",
    lessons: ["04", "13", "16"],
  },
  {
    term: "毕业卸库",
    definition:
      "日记、技能卡、提示名录一起卸掉，座位、流程、计分规则仍只从权重里取。课内玩具实验是 extra graduate，不是已经上线的自我进化员工。",
    diagnostic: "python3 run.py extra run graduate：三件外挂全空，权重仍能叫人、取 torch、算 2a+3b。",
    lessons: ["16", "21", "24"],
  },
  {
    term: "生成回放",
    alias: "generative replay",
    definition:
      "不把旧样本留在磁盘上，让旧模型或一个生成器自己出题，给正在学新东西的网络当老师。课内玩具是 extra gendream：随机探针上匹配旧 W。真把旧键留下来回放更稳，但那还是外挂。",
    diagnostic:
      "python3 run.py extra run gendream：naive 学 B 后 A 应接近 0；做梦后 A 应明显高于 naive。GPU 对照 gpu print vandeven-dgr。",
    lessons: ["06", "16", "19"],
  },
  {
    term: "回放缓冲",
    alias: "replay buffer",
    definition:
      "DER / iCaRL 留下来的那一小袋旧样本。训练新任务时混进去，旧的不容易忘。袋本身是外挂：卸掉再学下一个任务，旧的会没。",
    diagnostic: "python3 run.py extra run buffer。GPU 对照 mammoth-icarl 或 vandeven-er。",
    lessons: ["06", "08", "16"],
  },
  {
    term: "影子读取",
    definition:
      "日记还塞在 prompt 里时，未训权重和已训权重看起来一样会。必须卸掉提示再测，才能知道东西有没有进权重。",
    diagnostic: "python3 run.py extra run shadow：提示在时两边都是满分，卸提示后未训接近 0。",
    lessons: ["04", "13", "16"],
  },
  {
    term: "资格写入",
    definition:
      "只把反复问对的条目写进权重。第一次检索到的噪声不要当晚巩固。",
    diagnostic: "python3 run.py extra run eligible：全倒会记住噪声；只写出现多次的，噪声应接近 0。",
    lessons: ["13", "16", "18"],
  },
  {
    term: "长尾仍留库",
    definition:
      "常问的进权重，很少问的先留在日记。目标是缩小外挂，不是第一夜清空 Mem0。",
    diagnostic: "python3 run.py extra run longtail：头在 W，尾在日记；把尾也卸掉，尾应接近 0。",
    lessons: ["13", "16", "24"],
  },
  {
    term: "主动遗忘",
    definition:
      "外挂可以删一行。权重不会跟着删。要把该键写向墓碑或相反方向，并回放其他人，否则不是忘了这个人，就是把花名册冲掉。",
    diagnostic: "python3 run.py extra run tombstone：删日记后权重仍指向旧座位；写墓碑后最近邻不再是旧座位。",
    lessons: ["13", "14", "16"],
  },
  {
    term: "快速权重",
    alias: "fast weights",
    definition:
      "会话或白天这一档更新得勤的参数。夜里把它的映射写入慢权重，再清掉快的。Nested Learning 把这种频率差写成嵌套优化。",
    diagnostic: "extra sleep：清掉快权重且不做夜间巩固，召回应接近 0。",
    lessons: ["17", "19", "16"],
  },
  {
    term: "自编辑",
    alias: "self-edit",
    definition:
      "模型给自己出微调数据和更新指令，再写进权重。SEAL 用强化学习挑自编辑；课内缩小版先用对错过滤器。",
    diagnostic:
      "不筛选就把生成题灌进权重，规则误差应明显变差。见 extra selfedit 与第 23 课。",
    lessons: ["20", "23"],
  },
  {
    term: "灾难性遗忘",
    alias: "catastrophic forgetting",
    definition:
      "网络先学任务 A 再学任务 B 之后，A 的表现突然垮掉。原因通常是新梯度把旧决策边界推走，而不是记忆慢慢衰减。",
    diagnostic: "画任务-时间热力图。任务 1 的准确率若随任务 2 训练步数单调下降，就是这件事。",
    lessons: ["01", "02", "05"],
  },
  {
    term: "稳定性-可塑性困境",
    definition:
      "记得住旧知识叫稳定，学得进新知识叫可塑。把学习率调到接近 0 两头都弱；完全接着训则旧的先死。",
    diagnostic: "把方法画在旧任务保持 vs 新任务准确率平面上，不要只报一个平均分。",
    lessons: ["02", "05", "15"],
  },
  {
    term: "task / domain / class incremental",
    definition:
      "三种增量设定。task 测试时知道任务编号；domain 输入分布变、标签空间可同；class 新类别进来且测试时不告诉你现在是哪一类。",
    diagnostic: "class-incremental 通常最难，也最容易被平均准确率掩盖。",
    lessons: ["01", "03", "08"],
  },
  {
    term: "BWT",
    alias: "backward transfer",
    definition: "学了后面的任务之后，前面任务变好还是变差。负的 BWT 就是遗忘。",
    diagnostic: "用第 03 课的结果矩阵按 Lopez-Paz 公式算，不要口头估计。",
    lessons: ["03", "08", "10"],
  },
  {
    term: "FWT",
    alias: "forward transfer",
    definition: "旧任务学完后，还没训的新任务是否已经比随机更好。",
    diagnostic: "需要每个任务开始前的零样本或冻结评测，否则算不出来。",
    lessons: ["03"],
  },
  {
    term: "EWC",
    alias: "elastic weight consolidation",
    definition:
      "用旧任务的 Fisher 信息对角线当弹簧劲度，重要的权重少更新。",
    diagnostic: "λ=0 应接近 naive；λ 过大则新任务学不会。",
    lessons: ["05"],
  },
  {
    term: "回放",
    alias: "experience replay",
    definition: "训练新任务时从旧样本小缓冲里抽样，混进当前 batch。",
    diagnostic: "固定缓冲大小做消融。GDumb 用同一缓冲从头重训，常能打赢花哨方法。",
    lessons: ["06", "08"],
  },
  {
    term: "DER++",
    definition:
      "Dark Experience Replay 的加强版：回放旧输入时，既对齐旧 logits，也对齐旧标签。",
    diagnostic: "关掉蒸馏项，遗忘应上升。这是本课复现 #1。",
    lessons: ["06"],
  },
  {
    term: "GDumb",
    definition:
      "只维护一个类别平衡的缓冲，每个阶段用缓冲从头训练。不是聪明算法，是让人难堪的强基线。",
    diagnostic: "若你的新方法赢不了 GDumb，先检查任务边界是不是太干净、缓冲是不是太大。",
    lessons: ["08"],
  },
  {
    term: "LoRA",
    definition:
      "低秩适配：不改原权重，另训一对小矩阵，用它们的乘积当更新。",
    diagnostic: "两个任务的 LoRA 方向夹角接近 0° 时，后一个会覆盖前一个。",
    lessons: ["10", "11"],
  },
  {
    term: "O-LoRA",
    definition: "让不同任务的 LoRA 更新近似正交，减少子空间互踩。",
    diagnostic: "训练后测量 ⟨A1, A2⟩，应比 naive LoRA 更接近 0。",
    lessons: ["11"],
  },
  {
    term: "任务向量",
    definition: "微调后的权重减微调前的权重。可以相加、相减或按 TIES 规则修剪。",
    diagnostic: "合并发生在训练结束之后，不是在线持续学习。",
    lessons: ["12"],
  },
  {
    term: "RAG",
    definition: "检索一段外部文本再塞进上下文。当时会答，撤掉检索库通常就不会。",
    diagnostic: "同一套事实，分别在「有检索」和「无检索」下测。",
    lessons: ["04", "13", "16"],
  },
  {
    term: "外挂记忆",
    definition:
      "把日记、名录、对话摘要写在模型外面，用时再取。MemGPT / Letta、Mem0、HippoRAG 属于这一类。",
    diagnostic: "写入冲突事实后，系统是覆盖、并存还是胡编，必须单独测。",
    lessons: ["13", "16", "21"],
  },
  {
    term: "知识编辑",
    alias: "knowledge editing",
    definition:
      "定位少量参数，改一条事实。成功标准不只是那一条对，还要看泛化、局部性和流畅性。",
    diagnostic: "用 EasyEdit 的四指标，不要只看那一个问答。",
    lessons: ["14"],
  },
  {
    term: "可塑性丢失",
    alias: "loss of plasticity",
    definition:
      "学着学着学不动了。即使没有旧任务考试，后期任务的学习速度也会下降。",
    diagnostic: "画第 k 个任务的学习速度，并统计死神经元比例。",
    lessons: ["15"],
  },
  {
    term: "测试时学习",
    alias: "test-time training",
    definition:
      "读当前这段输入时，部分权重按这段输入做内环更新。TTT 层的隐状态是一套可学习的 W。",
    diagnostic: "内环之后 W 的更新范数应大于 0；普通 RNN 做不到这一点。",
    lessons: ["17"],
  },
  {
    term: "Titans",
    definition:
      "用惊讶（当前损失或梯度范数）当写入门控的神经长期记忆。常见事件少写，稀有事件多写。",
    diagnostic: "合成序列里稀有 token 的写入幅度应大于常见 token。",
    lessons: ["18"],
  },
  {
    term: "嵌套学习",
    alias: "Nested Learning",
    definition:
      "把架构和优化器看成不同更新频率的学习问题。Hope 在 Titans 上加了自修改和连续谱记忆。",
    diagnostic: "停掉某一时间尺度，看丢的是本句、本篇还是本任务的信息。",
    lessons: ["19"],
  },
  {
    term: "SEAL",
    definition:
      "Self-Adapting LLMs：模型生成给自己的微调数据和指令，再用强化学习优化这种自编辑。",
    diagnostic: "没有验证筛选时，错误数据会自我强化。第 23 课专门测这件事。",
    lessons: ["20", "23"],
  },
  {
    term: "RL's Razor",
    definition:
      "on-policy RL 比离线 SFT 更不易遗忘，主张原因是更新后的分布离原模型更近。",
    diagnostic: "同一新任务上对比 SFT 与 RL 的旧任务保持，并看遗忘是否随 KL 增大。",
    lessons: ["20"],
  },
  {
    term: "技能库",
    definition:
      "把环境验证过的程序存起来，下次检索复用。Voyager 这样做。权重可以不变。",
    diagnostic: "对照「每次从零写程序」。有库之后尝试次数应下降。",
    lessons: ["21", "24"],
  },
  {
    term: "经验时代",
    definition:
      "Silver 和 Sutton 的说法：智能体主要从与世界交互产生的数据里学，而不是只吃静态网页。",
    diagnostic: "关掉环境数据河，技能数应停止增长。",
    lessons: ["22"],
  },
  {
    term: "prequential",
    definition:
      "按时间顺序：先预测下一条，再拿它来训练。适合没有干净任务边界的流。",
    diagnostic: "和「每个任务结束打一次分」并排报，不要互相冒充。",
    lessons: ["03", "22"],
  },
];
