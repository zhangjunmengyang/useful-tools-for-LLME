import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson44Diagram: LessonDiagram = {
  lessonId: "44",
  title: "单元格命中要把内容和框同时过线",
  summary:
    "文档图像先编码，再排出阅读顺序并生成字段；内容匹配和框 IoU 分列判定，只有两列都过才记版面命中。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l44-doc",
      label: ["文档图像 + 问题"],
      meta: "发票 / 表格 / 跨页",
      kind: "input",
      x: 92,
      y: 180,
      width: 158,
    },
    {
      id: "l44-enc",
      label: ["视觉编码"],
      meta: "OCR-free patch",
      kind: "transform",
      x: 286,
      y: 88,
      width: 150,
    },
    {
      id: "l44-order",
      label: ["阅读顺序"],
      meta: "栅格 ≠ 版面",
      kind: "state",
      x: 286,
      y: 272,
      width: 150,
    },
    {
      id: "l44-fields",
      label: ["字段序列 / JSON"],
      meta: "key · cell · 页码",
      kind: "transform",
      x: 490,
      y: 180,
      width: 160,
    },
    {
      id: "l44-content",
      label: ["内容匹配"],
      meta: "字符串精确命中",
      kind: "decision",
      x: 690,
      y: 88,
      width: 150,
    },
    {
      id: "l44-box",
      label: ["框 IoU"],
      meta: "Hit = IoU ≥ 0.5",
      kind: "decision",
      x: 690,
      y: 272,
      width: 150,
    },
    {
      id: "l44-layout",
      label: ["版面命中"],
      meta: "内容 ∧ 框",
      kind: "output",
      x: 860,
      y: 180,
      width: 140,
    },
  ],
  edges: [
    {
      id: "l44-e-doc-enc",
      from: "l44-doc",
      to: "l44-enc",
      label: "像素",
      via: [
        { x: 174, y: 180 },
        { x: 174, y: 88 },
      ],
      labelAt: { x: 132, y: 128 },
    },
    {
      id: "l44-e-doc-order",
      from: "l44-doc",
      to: "l44-order",
      label: "栏 / 行 / 页",
      via: [
        { x: 174, y: 180 },
        { x: 174, y: 272 },
      ],
      labelAt: { x: 118, y: 232 },
    },
    {
      id: "l44-e-enc-fields",
      from: "l44-enc",
      to: "l44-fields",
      label: "next-token",
      labelAt: { x: 386, y: 52 },
    },
    {
      id: "l44-e-order-fields",
      from: "l44-order",
      to: "l44-fields",
      label: "序列约束",
      labelAt: { x: 386, y: 318 },
    },
    {
      id: "l44-e-fields-content",
      from: "l44-fields",
      to: "l44-content",
      label: "字段值",
      via: [
        { x: 560, y: 180 },
        { x: 560, y: 88 },
      ],
      labelAt: { x: 512, y: 128 },
    },
    {
      id: "l44-e-fields-box",
      from: "l44-fields",
      to: "l44-box",
      label: "单元格框",
      via: [
        { x: 560, y: 180 },
        { x: 560, y: 272 },
      ],
      labelAt: { x: 508, y: 232 },
    },
    {
      id: "l44-e-content-layout",
      from: "l44-content",
      to: "l44-layout",
      label: "ŝ = s",
      via: [
        { x: 790, y: 88 },
        { x: 820, y: 88 },
        { x: 820, y: 180 },
      ],
      labelAt: { x: 792, y: 64 },
    },
    {
      id: "l44-e-box-layout",
      from: "l44-box",
      to: "l44-layout",
      label: "IoU_τ",
      via: [
        { x: 790, y: 272 },
        { x: 820, y: 272 },
        { x: 820, y: 180 },
      ],
      labelAt: { x: 792, y: 296 },
    },
  ],
  steps: [
    {
      title: "文档先被当成一整张图",
      description:
        "发票、表格和跨页合同都从像素进去。OCR-free 编码器不先跑检测+识别流水线，但评测仍要能还原字段和框。",
      focus: ["l44-doc", "l44-enc"],
    },
    {
      title: "阅读顺序不是栅格扫描",
      description:
        "双栏文档按从上到下、从左到右扫会把右栏金额插进左栏条款中间。字段 JSON 必须服从版面阅读顺序，不能服从像素 y 再 x。",
      focus: ["l44-doc", "l44-order", "l44-fields"],
    },
    {
      title: "内容匹配只覆盖字符串",
      description:
        "问合计金额时输出 32.00，只说明语言模型或识别器拿到了数字。它不证明数字来自合计单元格。",
      focus: ["l44-fields", "l44-content"],
    },
    {
      title: "框 IoU 单独过线才算看对栏",
      description:
        "预测框与合计栏真值的交并比达到 0.5 才记框命中。框落在表头「金额」上时，即便字符串对，框命中仍为 0。",
      focus: ["l44-fields", "l44-box"],
    },
    {
      title: "版面命中是合取，不是均分",
      description:
        "单元格命中 = 内容对且框 IoU 过阈值。两列拆开记账。读对数字但框在表头，记版面失败。",
      focus: ["l44-content", "l44-box", "l44-layout"],
    },
  ],
  facts: [
    "Donut 把文档理解写成图像到 JSON 的 next-token，预训练目标是按阅读顺序读出全部文字。",
    "单元格命中要求内容匹配与框 IoU 同时成立，只有内容对不算版面对。",
    "Pix2Struct 用可变分辨率补丁和截图解析预训练，把问题直接渲染到图像顶部。",
    "字段级 F1 在漏一个字符时整字段失败；树编辑距离还要惩罚丢掉的嵌套结构。",
  ],
};
