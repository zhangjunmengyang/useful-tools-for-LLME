import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson53Diagram: LessonDiagram = {
  lessonId: "53",
  title: "跨会话先选定 payload，再按字节上限过期",
  summary:
    "隔天再来时，记忆记录要在像素、框和摘要之间做取舍。只留昨日摘要会把红杯子写进今日答案；三条原图一起留下会打穿字节上限。过期删像素、留摘要，并用最新观察改写实体。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l53-obs",
      label: ["跨日观察"],
      meta: "昨日红 / 今日蓝",
      kind: "input",
      x: 96,
      y: 180,
      width: 132,
    },
    {
      id: "l53-write",
      label: ["写入器"],
      meta: "会话关闭时落盘",
      kind: "transform",
      x: 268,
      y: 180,
      width: 132,
    },
    {
      id: "l53-payload",
      label: ["选 payload"],
      meta: "像素 / 框 / 摘要",
      kind: "decision",
      x: 448,
      y: 180,
      width: 140,
    },
    {
      id: "l53-pix",
      label: ["像素槽"],
      meta: "H x W x 3",
      kind: "state",
      x: 648,
      y: 68,
      width: 128,
    },
    {
      id: "l53-box",
      label: ["框槽"],
      meta: "4 x int32",
      kind: "state",
      x: 648,
      y: 180,
      width: 128,
    },
    {
      id: "l53-sum",
      label: ["摘要槽"],
      meta: "一句话 UTF-8",
      kind: "state",
      x: 648,
      y: 292,
      width: 128,
    },
    {
      id: "l53-cap",
      label: ["字节上限"],
      meta: "过期删像素",
      kind: "decision",
      x: 848,
      y: 180,
      width: 132,
    },
  ],
  edges: [
    {
      id: "l53-e-obs-write",
      from: "l53-obs",
      to: "l53-write",
      label: "关会话",
      labelAt: { x: 176, y: 152 },
    },
    {
      id: "l53-e-write-payload",
      from: "l53-write",
      to: "l53-payload",
      label: "写哪一层",
      labelAt: { x: 352, y: 152 },
    },
    {
      id: "l53-e-payload-pix",
      from: "l53-payload",
      to: "l53-pix",
      label: "原图贵",
      via: [
        { x: 530, y: 180 },
        { x: 530, y: 68 },
      ],
      labelAt: { x: 492, y: 96 },
    },
    {
      id: "l53-e-payload-box",
      from: "l53-payload",
      to: "l53-box",
      label: "只记位置",
      labelAt: { x: 548, y: 156 },
    },
    {
      id: "l53-e-payload-sum",
      from: "l53-payload",
      to: "l53-sum",
      label: "丢颜色",
      via: [
        { x: 530, y: 180 },
        { x: 530, y: 292 },
      ],
      labelAt: { x: 492, y: 256 },
    },
    {
      id: "l53-e-pix-cap",
      from: "l53-pix",
      to: "l53-cap",
      via: [
        { x: 760, y: 68 },
        { x: 760, y: 180 },
      ],
      label: "先删旧图",
      labelAt: { x: 792, y: 92 },
    },
    {
      id: "l53-e-box-cap",
      from: "l53-box",
      to: "l53-cap",
      label: "框留下",
      labelAt: { x: 748, y: 156 },
    },
    {
      id: "l53-e-sum-cap",
      from: "l53-sum",
      to: "l53-cap",
      via: [
        { x: 760, y: 292 },
        { x: 760, y: 180 },
      ],
      label: "摘要留下",
      labelAt: { x: 792, y: 268 },
    },
  ],
  steps: [
    {
      title: "会话关闭才落盘",
      description:
        "昨日红杯子和今日蓝杯子属于两次会话。窗口里的 token 在关会话时丢掉；要隔天再用，必须写成外部记录。",
      focus: ["l53-obs", "l53-write", "l53-e-obs-write"],
    },
    {
      title: "先选定 payload",
      description:
        "同一实体可以写成原图像素、空间框，或一句话摘要。层选错，次日阅读器拿到的证据就不同。",
      focus: [
        "l53-payload",
        "l53-pix",
        "l53-box",
        "l53-sum",
        "l53-e-write-payload",
        "l53-e-payload-pix",
        "l53-e-payload-box",
        "l53-e-payload-sum",
      ],
    },
    {
      title: "三条记录对字节上限",
      description:
        "教学夹具里每张图按 64x64x3 计。三条都留像素时，总和超过 16384 字节上限。",
      focus: ["l53-pix", "l53-cap", "l53-e-pix-cap"],
    },
    {
      title: "过期删像素、留摘要和框",
      description:
        "超限时按写入日从旧到新丢掉像素槽，摘要和框不得一起删。账单必须回到上限以内。",
      focus: ["l53-sum", "l53-box", "l53-cap", "l53-e-sum-cap", "l53-e-box-cap"],
    },
    {
      title: "次日颜色看最新观察",
      description:
        "只读首条“红色杯子”会答错。混合策略在今日写入时改写实体摘要，并只保留仍装得下的新图。",
      focus: ["l53-obs", "l53-sum", "l53-cap"],
    },
  ],
  facts: [
    "LoCoMo（arXiv:2402.17753）50 段对话，平均 304.9 轮、19.3 个会话、9209.2 个 token，最长 35 个会话。",
    "MemoryBank（arXiv:2305.10250）表 2：SiliconFriend ChatGPT 英文正确率 0.716、检索准确率 0.763。",
    "PMMC（arXiv:2608.00962）去掉原图访问后，Mem-Gallery 的 Harmonized Judge 下降 4.9 分。",
    "MIRIX（arXiv:2507.07957）在 ScreenshotVQA 上比 RAG 高 35%，存储少 99.9%。",
    "第 14 课扩的是窗口 T，第 39 课管的是子目标栈 k，第 45 课选的是检索层；本课管的是跨会话记录存哪一种 payload。",
  ],
};
