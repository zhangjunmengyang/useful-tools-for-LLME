import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson39Diagram: LessonDiagram = {
  lessonId: "39",
  title: "长程失败：窗口回放还是子目标栈",
  summary:
    "四步指令压进栈后只执行栈顶；成功则写入已提交表。失败时把历史拼进窗口会重做已提交步，pop 只重试失败步。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l39-instr",
      label: ["四步指令"],
      meta: "开抽 / 抓块 / 放入 / 关闭",
      kind: "input",
      x: 108,
      y: 88,
      width: 148,
    },
    {
      id: "l39-stack",
      label: ["子目标栈"],
      meta: "栈深 k",
      kind: "state",
      x: 318,
      y: 88,
      width: 132,
    },
    {
      id: "l39-skill",
      label: ["当前技能"],
      meta: "只跑栈顶",
      kind: "transform",
      x: 528,
      y: 88,
      width: 132,
    },
    {
      id: "l39-pred",
      label: ["成败判定"],
      meta: "状态差 Φ",
      kind: "decision",
      x: 748,
      y: 88,
      width: 132,
    },
    {
      id: "l39-commit",
      label: ["已提交表 C"],
      meta: "禁止再执行",
      kind: "state",
      x: 318,
      y: 268,
      width: 132,
    },
    {
      id: "l39-window",
      label: ["窗口回放"],
      meta: "长度 T",
      kind: "transform",
      x: 528,
      y: 268,
      width: 132,
    },
    {
      id: "l39-log",
      label: ["执行日志"],
      meta: "第一步计数",
      kind: "output",
      x: 748,
      y: 268,
      width: 132,
    },
  ],
  edges: [
    {
      id: "l39-e-instr-stack",
      from: "l39-instr",
      to: "l39-stack",
      label: "push 分解",
      labelAt: { x: 214, y: 58 },
    },
    {
      id: "l39-e-stack-skill",
      from: "l39-stack",
      to: "l39-skill",
      label: "读栈顶",
      labelAt: { x: 424, y: 58 },
    },
    {
      id: "l39-e-skill-pred",
      from: "l39-skill",
      to: "l39-pred",
      label: "执行后判 Φ",
      labelAt: { x: 638, y: 58 },
    },
    {
      id: "l39-e-pred-commit",
      from: "l39-pred",
      to: "l39-commit",
      label: "成功则 commit",
      via: [
        { x: 748, y: 180 },
        { x: 318, y: 180 },
      ],
      labelAt: { x: 530, y: 158 },
    },
    {
      id: "l39-e-pred-window",
      from: "l39-pred",
      to: "l39-window",
      label: "失败且拼接历史",
      via: [{ x: 748, y: 200 }],
      labelAt: { x: 680, y: 198 },
    },
    {
      id: "l39-e-commit-stack",
      from: "l39-commit",
      to: "l39-stack",
      label: "C 不再压回",
      labelAt: { x: 250, y: 178 },
    },
    {
      id: "l39-e-window-log",
      from: "l39-window",
      to: "l39-log",
      label: "重做第一步",
      labelAt: { x: 638, y: 238 },
    },
    {
      id: "l39-e-commit-log",
      from: "l39-commit",
      to: "l39-log",
      label: "栈臂第一步=1",
      via: [{ x: 318, y: 320 }],
      labelAt: { x: 530, y: 338 },
    },
  ],
  steps: [
    {
      title: "把四步压进栈",
      description:
        "开抽屉、抓蓝块、放入、关闭按相反顺序 push，栈顶是当前句。k 计步骤，不是 token。",
      focus: ["l39-instr", "l39-stack", "l39-e-instr-stack"],
    },
    {
      title: "只执行栈顶",
      description:
        "政策条件在当前技能与世界状态上。CALVIN 官方 LH-MTLC 也是当前句成功才切下一句。",
      focus: ["l39-stack", "l39-skill", "l39-e-stack-skill"],
    },
    {
      title: "成功写入已提交表",
      description:
        "状态差谓词为真则 pop 并写入 C。已提交名称不得再压回栈，也不能出现在后续执行表。",
      focus: ["l39-pred", "l39-commit", "l39-e-pred-commit", "l39-e-commit-stack"],
    },
    {
      title: "窗口回放会重做已提交步",
      description:
        "第二步失败后把整段指令拼进长度为 T 的窗口，启发式从列表头再发射，open_drawer 出现第二次。",
      focus: ["l39-pred", "l39-window", "l39-log", "l39-e-pred-window", "l39-e-window-log"],
    },
    {
      title: "pop 只重试失败步",
      description:
        "栈臂保持 C，只对栈顶加 r。执行表里开抽屉一次、抓蓝块两次。加大 T 不改变这个计数。",
      focus: ["l39-commit", "l39-log", "l39-e-commit-log"],
    },
  ],
  facts: [
    "CALVIN 的 MCIL 基线在 D→D、静态 RGB 上短程 MTLC 为 53.9%，五步长程链成功率为 0.08%（Fig. 8）。",
    "栈深 k 与窗口长度 T 单位不同：提交第一步后 k=3，窗口仍可取 48 或 128 token，T 加大不会取消已提交步的重放。",
    "四步任务第二步失败时，窗口回放的执行表含第二次 open_drawer；栈协议的 open_drawer 只出现一次。",
    "HULC 在 ABCD→D 上五步成功率为 38.3%、平均链长 3.06（Fig. 4，三次种子）；层次潜在计划没有 pop。",
    "SayCan 用 p(cπ|s,ℓπ) p(ℓπ|i) 选下一步，历史是技能名而不是关节 token（Algorithm 1）。",
  ],
};
