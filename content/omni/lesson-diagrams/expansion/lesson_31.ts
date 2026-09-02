import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson31Diagram: LessonDiagram = {
  lessonId: "31",
  title: "可拆层 VLA 评测：从协议到区间",
  summary:
    "同一政策先被套件、初始态、视觉域和语言指令拆开，再用成功谓词逐条打分，最后用样本量给出区间，而不是一张平均成功率。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l31-protocol",
      label: ["评测协议"],
      meta: "套件 / 种子 / 域",
      kind: "input",
      x: 96,
      y: 180,
    },
    {
      id: "l31-suite",
      label: ["任务套件桶"],
      meta: "Spatial/Object/Goal/Long",
      kind: "transform",
      x: 292,
      y: 78,
      width: 168,
    },
    {
      id: "l31-init",
      label: ["初始态种子"],
      meta: "固定或随机",
      kind: "state",
      x: 292,
      y: 272,
    },
    {
      id: "l31-visual",
      label: ["视觉域"],
      meta: "相机 / 纹理 / 背景",
      kind: "transform",
      x: 520,
      y: 78,
      width: 160,
    },
    {
      id: "l31-language",
      label: ["指令改写"],
      meta: "未见句 / 歧义",
      kind: "transform",
      x: 520,
      y: 272,
    },
    {
      id: "l31-predicate",
      label: ["成功谓词"],
      meta: "接触 / 放置 / 保持",
      kind: "decision",
      x: 700,
      y: 180,
    },
    {
      id: "l31-report",
      label: ["拆桶报告"],
      meta: "点估计 + Wilson",
      kind: "output",
      x: 862,
      y: 180,
      width: 132,
    },
  ],
  edges: [
    {
      id: "l31-e-protocol-suite",
      from: "l31-protocol",
      to: "l31-suite",
      label: "按知识类型分桶",
      via: [{ x: 96, y: 78 }],
      labelAt: { x: 168, y: 52 },
    },
    {
      id: "l31-e-protocol-init",
      from: "l31-protocol",
      to: "l31-init",
      label: "锁定 reset",
      via: [{ x: 96, y: 272 }],
      labelAt: { x: 168, y: 302 },
    },
    {
      id: "l31-e-suite-visual",
      from: "l31-suite",
      to: "l31-visual",
      label: "渲染观察",
      labelAt: { x: 408, y: 52 },
    },
    {
      id: "l31-e-init-language",
      from: "l31-init",
      to: "l31-language",
      label: "条件语言",
      labelAt: { x: 408, y: 302 },
    },
    {
      id: "l31-e-visual-pred",
      from: "l31-visual",
      to: "l31-predicate",
      label: "像素进政策",
      labelAt: { x: 620, y: 118 },
    },
    {
      id: "l31-e-language-pred",
      from: "l31-language",
      to: "l31-predicate",
      label: "指令进政策",
      labelAt: { x: 620, y: 248 },
    },
    {
      id: "l31-e-pred-report",
      from: "l31-predicate",
      to: "l31-report",
      label: "k/N 与区间",
      labelAt: { x: 804, y: 148 },
    },
  ],
  steps: [
    {
      title: "先写协议再跑政策",
      description:
        "协议卡锁定套件、reset 种子、视觉域和指令集合。缺任何一项，成功率没有可比对象。",
      focus: ["l31-protocol", "l31-suite", "l31-init", "l31-e-protocol-suite", "l31-e-protocol-init"],
    },
    {
      title: "按套件拆桶",
      description:
        "LIBERO-Spatial / Object / Goal / Long 测的是不同知识。四套件宏平均 76.5% 不能代替 Long 的 53.7%。",
      focus: ["l31-suite", "l31-e-suite-visual"],
    },
    {
      title: "声明视觉域",
      description:
        "SIMPLER 的 Visual Matching 用绿幕叠真实背景；Variant Aggregation 则在纹理与光照上聚合。sim-to-real 声明必须写明用了哪一种。",
      focus: ["l31-visual", "l31-e-visual-pred"],
    },
    {
      title: "核对成功谓词",
      description:
        "接触、放入目标区域、保持若干步是三条不同的判定。CALVIN 对 34 个任务逐条写了几何阈值，不能用“看起来做完了”代替。",
      focus: ["l31-language", "l31-predicate", "l31-e-language-pred"],
    },
    {
      title: "用区间而不是一个百分数",
      description:
        "N=25、成功率 0.8 的 Wilson 95% 区间约为 [0.609, 0.911]。真机小样本和仿真 500 次试验不能横着比。",
      focus: ["l31-predicate", "l31-report", "l31-e-pred-report"],
    },
  ],
  facts: [
    "OpenVLA 在 LIBERO 四套件上独立 fine-tune 后的成功率为 Spatial 84.7%、Object 88.4%、Goal 79.2%、Long 53.7%，宏平均 76.5%（Table 12，每套件 500 trials × 3 seeds）。",
    "OpenVLA-OFT 在四套件上 fine-tune、加腕部相机与本体感受后的宏平均为 97.1%（Table I，每套件 500 trials）；π0 fine-tune 同表为 94.2%。",
    "CALVIN 的 MCIL 基线在 D→D、静态 RGB 上短程 MTLC 为 53.9%，五步长程链成功率为 0.08%（Fig. 8）。",
    "SIMPLER Google Robot 上 Visual Matching 的平均 Pearson r 为 0.924、MMRV 为 0.056；目标是相对排序相关，不是 1:1 复制真机成功率（Table I）。",
    "N=25、k=20 时，Wilson 95% 区间为 [0.609, 0.911]，正态近似为 [0.643, 0.957]；两者都不覆盖 0.5。",
  ],
};
