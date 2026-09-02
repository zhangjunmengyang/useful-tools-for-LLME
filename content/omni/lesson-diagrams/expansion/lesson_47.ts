import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson47Diagram: LessonDiagram = {
  lessonId: "47",
  title: "六类评测数字分桶记账",
  summary:
    "原始百分数先填协议卡，再打上互斥的六类标签。理解类与执行类分开入账，真机陷阱直接拒收，最后只在同类同单位上比较。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l47-raw",
      label: ["原始数字"],
      meta: "六张公开卡",
      kind: "input",
      x: 96,
      y: 180,
    },
    {
      id: "l47-card",
      label: ["协议卡"],
      meta: "基准 / 划分 / 单位",
      kind: "transform",
      x: 268,
      y: 180,
      width: 150,
    },
    {
      id: "l47-understand",
      label: ["理解三类"],
      meta: "C1 / C2 / C3",
      kind: "state",
      x: 468,
      y: 78,
      width: 150,
    },
    {
      id: "l47-act",
      label: ["执行三类"],
      meta: "C4 / C5 / C6",
      kind: "state",
      x: 468,
      y: 272,
      width: 150,
    },
    {
      id: "l47-label",
      label: ["互斥标签"],
      meta: "恰好一类",
      kind: "decision",
      x: 668,
      y: 180,
    },
    {
      id: "l47-trap",
      label: ["真机陷阱"],
      meta: "LIBERO 标红",
      kind: "decision",
      x: 668,
      y: 78,
    },
    {
      id: "l47-book",
      label: ["分桶账本"],
      meta: "禁止总平均",
      kind: "output",
      x: 860,
      y: 180,
      width: 132,
    },
  ],
  edges: [
    {
      id: "l47-e-raw-card",
      from: "l47-raw",
      to: "l47-card",
      label: "补字段",
      labelAt: { x: 178, y: 152 },
    },
    {
      id: "l47-e-card-und",
      from: "l47-card",
      to: "l47-understand",
      label: "图 / 视频 / 三模态",
      via: [{ x: 268, y: 78 }],
      labelAt: { x: 348, y: 52 },
    },
    {
      id: "l47-e-card-act",
      from: "l47-card",
      to: "l47-act",
      label: "电脑 / 仿真 / 相关",
      via: [{ x: 268, y: 272 }],
      labelAt: { x: 348, y: 302 },
    },
    {
      id: "l47-e-und-label",
      from: "l47-understand",
      to: "l47-label",
      label: "C1-C3",
      labelAt: { x: 568, y: 118 },
    },
    {
      id: "l47-e-act-label",
      from: "l47-act",
      to: "l47-label",
      label: "C4-C6",
      labelAt: { x: 568, y: 248 },
    },
    {
      id: "l47-e-und-trap",
      from: "l47-understand",
      to: "l47-trap",
      label: "不得冒充真机",
      labelAt: { x: 568, y: 52 },
    },
    {
      id: "l47-e-label-book",
      from: "l47-label",
      to: "l47-book",
      label: "同类才做差",
      labelAt: { x: 764, y: 152 },
    },
    {
      id: "l47-e-trap-book",
      from: "l47-trap",
      to: "l47-book",
      label: "拒收",
      via: [{ x: 860, y: 78 }],
      labelAt: { x: 790, y: 52 },
    },
  ],
  steps: [
    {
      title: "先写协议卡再看百分数",
      description:
        "每条数字必须带基准名、划分、输入模态、成功定义、N、是否 fine-tune 和单位。缺字段的格子不得入账。",
      focus: ["l47-raw", "l47-card", "l47-e-raw-card"],
    },
    {
      title: "打上互斥的六类标签",
      description:
        "C1 MMMU、C2 Video-MME、C3 OmniBench 走理解账；C4 OSWorld、C5 LIBERO、C6 SIMPLER 走执行账。一条记录恰好一类。",
      focus: [
        "l47-understand",
        "l47-act",
        "l47-label",
        "l47-e-card-und",
        "l47-e-card-act",
      ],
    },
    {
      title: "把真机陷阱标红",
      description:
        "LIBERO 四套件平均是仿真操作成功率。拖进真机能力必须标红。SIMPLER 的 Pearson r 也不是真机成功率。",
      focus: ["l47-trap", "l47-e-und-trap"],
    },
    {
      title: "只在同类同单位上比较",
      description:
        "OSWorld 的人类 72.36% 可以对 GPT-4 的 12.24%。MMMU 55.7% 不能减 Video-MME 75%。区间沿用第 31 课 Wilson。",
      focus: ["l47-label", "l47-book", "l47-e-label-book"],
    },
    {
      title: "账本禁止总平均",
      description:
        "六列可以并排展示，不能兑成一个 Omni 均分。第 01 课的分桶和第 23 课的探针在每一类内部继续生效。",
      focus: ["l47-book", "l47-e-trap-book"],
    },
  ],
  facts: [
    "MMMU 共 11550 题，GPT-4V 测试集准确率 55.7%，验证集专家最好 88.6%；该数字测大学学科静态图文，不测视频、三模态或缺一不可的音频。",
    "Video-MME：900 视频、2700 题。Gemini 1.5 Pro 无字幕 75.0%、有字幕 81.3%；长视频无字幕 67.4%。字幕是可选增益，不是 OmniBench 那种三模态硬约束。",
    "OmniBench 1142 题，正确作答必须同时用图和声。Gemini-1.5-Pro 42.91%，Qwen2.5-Omni-7B 56.13%（Speech 55.25 / Sound 60.00 / Music 52.83），人类专家 74.03%。",
    "OSWorld 369 项 Ubuntu 任务，人类 72.36%，GPT-4 无障碍树 12.24%；这是计算机执行成功率，不能和 LIBERO 仿真平均横比。",
    "OpenVLA 在 LIBERO 四套件独立 fine-tune 后宏平均 76.5%（Table 12），不是真机能力。SIMPLER Google Robot Visual Matching 平均 Pearson r=0.924，单位是排序相关，不是成功率。",
  ],
};
