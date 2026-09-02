export type ExtraExperimentCard = {
  id: string;
  title: string;
  lesson: string;
  question: string;
};

export const extraExperiments: readonly ExtraExperimentCard[] = [
  {
    id: "unplug",
    title: "拔掉外挂记忆",
    lesson: "04, 13",
    question: "日记会了，把库卸掉之后还会不会？",
  },
  {
    id: "distill",
    title: "夜间巩固：记忆写入权重",
    lesson: "13, 16, 24",
    question: "把日记里的事实练进矩阵 W 之后，卸掉日记还能不能叫到人？",
  },
  {
    id: "skill",
    title: "技能卡编译进策略",
    lesson: "16, 21",
    question: "Voyager 式技能库卸掉之后，编译进权重的流程还能不能跑？",
  },
  {
    id: "selfedit",
    title: "自编辑写入权重",
    lesson: "20, 23",
    question: "模型给自己出训练题时，不筛选会不会把规则写坏？",
  },
  {
    id: "conflict",
    title: "改一条座位",
    lesson: "13, 14",
    question: "日记覆盖和新写权重，谁会把花名册其余人冲掉？",
  },
  {
    id: "evolve",
    title: "五日进化：日记减、权重增",
    lesson: "16, 20, 23, 24",
    question: "Agent 能不能先靠外挂顶几天，再把会的东西写进权重，最后少依赖日记？",
  },
  {
    id: "route",
    title: "分流：写上下文、日记还是权重",
    lesson: "04, 13, 16",
    question: "闲聊、座位、计分规则，卸掉外挂之后各还剩什么？",
  },
  {
    id: "capacity",
    title: "容量：日记不能整本倒进权重",
    lesson: "13, 16",
    question: "小矩阵装不下整个 Mem0 时，全倒进去会怎样？只巩固常问的呢？",
  },
  {
    id: "sleep",
    title: "两档转速：白天快权重，夜里慢权重",
    lesson: "16, 19",
    question: "会话级快权重清掉之后，夜间写入的慢权重还能不能叫人？",
  },
  {
    id: "surprise",
    title: "惊讶门：稀有事实才该大力写",
    lesson: "18, 16",
    question: "日记流里天天重复的句子，会不会把只出现一次的座位冲掉？",
  },
  {
    id: "seqedit",
    title: "连续自编辑会忘更早的一批",
    lesson: "20, 23",
    question: "SEAL 式一连串写入同一张权重，不带回放，第一批还在吗？",
  },
  {
    id: "onpolicy",
    title: "小步 on-policy 离原点更近",
    lesson: "20, 23",
    question: "把日记整袋拿去微调，会不会比筛过的小步更伤旧能力？",
  },
  {
    id: "ortho",
    title: "正交子空间才留得住旧技能",
    lesson: "11, 21",
    question: "第二块 LoRA 若和第一块抢同一方向，旧技能会不会没？",
  },
  {
    id: "ewcmem",
    title: "巩固后再学新座位：护住旧方向",
    lesson: "05, 13, 16",
    question: "夜间已经写入权重的座位，第二天继续微调时，不在旧键方向上写，能不能少冲一点？",
  },
  {
    id: "plastic",
    title: "连续写入之后学新的变慢",
    lesson: "15, 23",
    question: "Agent 连续多天改自己的权重，后面还学得动吗？",
  },
  {
    id: "graduate",
    title: "毕业卸库：日记、技能卡、提示词全拔",
    lesson: "16, 21, 24",
    question: "三件外挂都卸掉之后，权重里的座位、流程和计分规则还在不在？",
  },
  {
    id: "buffer",
    title: "回放缓冲仍是外挂",
    lesson: "06, 08, 16",
    question: "DER / iCaRL 留下的那袋旧样本，卸掉之后再学新的，旧事实还在吗？",
  },
  {
    id: "gendream",
    title: "生成回放：不存旧键也能护住旧映射",
    lesson: "06, 16, 19",
    question: "卸掉日记之后还要接着学，能不能让旧权重自己出题给自己练，而不是再藏一袋样本？",
  },
  {
    id: "stale",
    title: "日记改了，权重还在说旧座位",
    lesson: "13, 14, 16",
    question: "工位当晚就换了。Mem0 立刻覆盖。权重要到哪一步才会改口？",
  },
  {
    id: "shadow",
    title: "影子读取：提示还在时看不出权重会不会",
    lesson: "04, 13, 16",
    question: "产品把日记塞进 prompt 再问模型，这个分数能证明已经写进权重了吗？",
  },
  {
    id: "eligible",
    title: "资格写入：只巩固反复问对的",
    lesson: "13, 16, 18",
    question: "第一次检索到的错座位，要不要当晚就写进权重？",
  },
  {
    id: "budget",
    title: "夜间预算：一夜写不完整本日记",
    lesson: "09, 13, 16",
    question: "GPU 晚上只能跑几百步时，该按查询次数巩固，还是把当天日志末尾倒进去？",
  },
  {
    id: "tombstone",
    title: "主动遗忘：删日记不等于改权重",
    lesson: "13, 14, 16",
    question: "有人离职了。Mem0 可以删一行。权重里的座位怎么拿掉，才不会把其他人一起冲掉？",
  },
  {
    id: "longtail",
    title: "长尾仍留库：先卸常问的，不是一夜清空",
    lesson: "13, 16, 24",
    question: "真要做到不靠外挂，是不是第一夜就把 Mem0 整库删掉？",
  },
  {
    id: "compose",
    title: "组合技能：库里没有这张卡",
    lesson: "16, 21",
    question: "Voyager 式技能卡没有「火把」这一张。两个已编译技能拼起来，卸掉库还能跑吗？",
  },
  {
    id: "disagree",
    title: "日记和权重打架时先别卸",
    lesson: "13, 16, 24",
    question: "Mem0 说工位换了，权重还说旧的。这种时候能不能把这一条从日记里删掉？",
  },
  {
    id: "keepfail",
    title: "过关才出日记：卸库探针当毕业考",
    lesson: "16, 23, 24",
    question: "夜间写完就宣布「已经进权重了」？还是先拔掉日记考一遍，没过的留到明天？",
  },
  {
    id: "rollback",
    title: "坏的一夜：检查点也是外挂",
    lesson: "20, 23, 24",
    question: "自编辑写坏了规则。没有权重检查点的话，还能回到昨晚之前吗？",
  },
];
