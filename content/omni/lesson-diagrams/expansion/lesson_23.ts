import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson23Diagram: LessonDiagram = {
  lessonId: "23",
  title: "VQA 答对不能代替 grounding 命中",
  summary:
    "同一张图并行走出文字答案、预测框与是否存在探针；只有 IoU 或注意力落在被问物体上，才算看见了位置。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l23-input",
      label: ["图像 + 问题"],
      meta: "指代 / OCR / 空间",
      kind: "input",
      x: 88,
      y: 180,
      width: 150,
    },
    {
      id: "l23-tokens",
      label: ["视觉 token"],
      meta: "patch / 编号标记",
      kind: "transform",
      x: 280,
      y: 88,
      width: 150,
    },
    {
      id: "l23-attn",
      label: ["注意力 / 位置 token"],
      meta: "mask · loc vocab",
      kind: "state",
      x: 280,
      y: 268,
      width: 168,
    },
    {
      id: "l23-iou",
      label: ["框 IoU 判定"],
      meta: "Hit = IoU ≥ 0.5",
      kind: "decision",
      x: 490,
      y: 88,
      width: 150,
    },
    {
      id: "l23-pope",
      label: ["POPE 是否存在"],
      meta: "负例仍答 Yes",
      kind: "decision",
      x: 490,
      y: 268,
      width: 160,
    },
    {
      id: "l23-vqa",
      label: ["VQA 文字答案"],
      meta: "可与命中脱钩",
      kind: "output",
      x: 700,
      y: 88,
      width: 150,
    },
    {
      id: "l23-report",
      label: ["分项报告"],
      meta: "命中 ≠ 准确率",
      kind: "output",
      x: 860,
      y: 180,
      width: 140,
    },
  ],
  edges: [
    {
      id: "l23-e-input-tokens",
      from: "l23-input",
      to: "l23-tokens",
      label: "编码像素",
      via: [
        { x: 170, y: 180 },
        { x: 170, y: 88 },
      ],
      labelAt: { x: 128, y: 128 },
    },
    {
      id: "l23-e-input-attn",
      from: "l23-input",
      to: "l23-attn",
      label: "问句对齐区域",
      via: [
        { x: 170, y: 180 },
        { x: 170, y: 268 },
      ],
      labelAt: { x: 118, y: 232 },
    },
    {
      id: "l23-e-tokens-iou",
      from: "l23-tokens",
      to: "l23-iou",
      label: "预测框",
      labelAt: { x: 386, y: 52 },
    },
    {
      id: "l23-e-attn-iou",
      from: "l23-attn",
      to: "l23-iou",
      label: "质量分数",
      via: [
        { x: 360, y: 268 },
        { x: 360, y: 88 },
      ],
      labelAt: { x: 312, y: 178 },
    },
    {
      id: "l23-e-tokens-vqa",
      from: "l23-tokens",
      to: "l23-vqa",
      label: "next-token",
      via: [
        { x: 430, y: 48 },
        { x: 620, y: 48 },
      ],
      labelAt: { x: 530, y: 28 },
    },
    {
      id: "l23-e-attn-pope",
      from: "l23-attn",
      to: "l23-pope",
      label: "物体探针",
      labelAt: { x: 386, y: 318 },
    },
    {
      id: "l23-e-iou-report",
      from: "l23-iou",
      to: "l23-report",
      label: "Hit_τ",
      via: [
        { x: 620, y: 88 },
        { x: 800, y: 88 },
        { x: 800, y: 180 },
      ],
      labelAt: { x: 760, y: 68 },
    },
    {
      id: "l23-e-vqa-report",
      from: "l23-vqa",
      to: "l23-report",
      label: "Acc_VQA",
      labelAt: { x: 790, y: 118 },
    },
    {
      id: "l23-e-pope-report",
      from: "l23-pope",
      to: "l23-report",
      label: "R_hall",
      via: [
        { x: 700, y: 268 },
        { x: 800, y: 268 },
        { x: 800, y: 180 },
      ],
      labelAt: { x: 748, y: 296 },
    },
  ],
  steps: [
    {
      title: "同一输入分出两条账",
      description:
        "图像和问题先编码成视觉 token。文字答案走 next-token；位置走框、点或注意力质量分数。两条账不得互相替代。",
      focus: ["l23-input", "l23-tokens", "l23-attn"],
    },
    {
      title: "用 IoU 判定有没有看对地方",
      description:
        "预测框与真值框的交并比达到 0.5 才记一次 grounding 命中。注意力落在物体 mask 上的质量分数是同一判定的像素版。",
      focus: ["l23-tokens", "l23-attn", "l23-iou"],
    },
    {
      title: "用 POPE 探针抓存在幻觉",
      description:
        "物体不在图中时仍答 Yes 记入幻觉率。只统计负例，不能把模型爱说 Yes 的比例直接当成幻觉率。",
      focus: ["l23-attn", "l23-pope"],
    },
    {
      title: "VQA 答对可以和命中同时失败",
      description:
        "同色干扰物或语言共现就能写出正确答案，框和注意力却在另一件物体上。报告必须同时列出 Acc_VQA 与 Hit_τ。",
      focus: ["l23-iou", "l23-vqa", "l23-report"],
    },
    {
      title: "OCR 按定位而不是按配文评分",
      description:
        "读出发票金额或招牌文字，要求预测区域盖住字形。只生成含数字的句子，不算 OCR 命中。",
      focus: ["l23-tokens", "l23-iou", "l23-vqa", "l23-report"],
    },
  ],
  facts: [
    "指代表达理解常用 IoU ≥ 0.5 判定预测框是否命中真值。",
    "VQA 准确率高不能推出 grounding 命中，同色干扰物即可拆开这两项。",
    "POPE 把物体幻觉写成是否存在的二分类，幻觉率只在负例上计算。",
    "Kosmos-2 把框离散成位置 token，按 Markdown 超链接接到短语后面。",
  ],
};
