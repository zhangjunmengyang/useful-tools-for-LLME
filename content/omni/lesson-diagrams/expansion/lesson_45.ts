import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson45Diagram: LessonDiagram = {
  lessonId: "45",
  title: "先选检索层，再把命中段交给精读",
  summary:
    "查询先决定走字幕、中间特征还是像素索引；只有进入该层 Top-k 的片段才会被阅读器看见。错误层召回会让目标段永远进不了上下文。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l45-query",
      label: ["查询"],
      meta: "文本 / 工具参数",
      kind: "input",
      x: 100,
      y: 180,
    },
    {
      id: "l45-router",
      label: ["选层"],
      meta: "字幕 / 中间 / 像素",
      kind: "decision",
      x: 268,
      y: 180,
    },
    {
      id: "l45-sub",
      label: ["字幕索引"],
      meta: "ASR / OCR",
      kind: "state",
      x: 430,
      y: 68,
    },
    {
      id: "l45-mid",
      label: ["中间特征"],
      meta: "片段向量",
      kind: "state",
      x: 430,
      y: 180,
    },
    {
      id: "l45-pix",
      label: ["像素索引"],
      meta: "帧 / patch",
      kind: "state",
      x: 430,
      y: 292,
    },
    {
      id: "l45-topk",
      label: ["Top-k 片段"],
      meta: "Recall@k",
      kind: "transform",
      x: 640,
      y: 180,
    },
    {
      id: "l45-read",
      label: ["精读阅读器"],
      meta: "预算内看图",
      kind: "output",
      x: 850,
      y: 180,
      width: 150,
    },
  ],
  edges: [
    {
      id: "l45-e-query-router",
      from: "l45-query",
      to: "l45-router",
      label: "选哪一层",
      labelAt: { x: 184, y: 228 },
    },
    {
      id: "l45-e-router-sub",
      from: "l45-router",
      to: "l45-sub",
      label: "只听字",
      via: [
        { x: 330, y: 180 },
        { x: 330, y: 68 },
      ],
      labelAt: { x: 292, y: 112 },
    },
    {
      id: "l45-e-router-mid",
      from: "l45-router",
      to: "l45-mid",
      label: "图文空间",
      labelAt: { x: 348, y: 156 },
    },
    {
      id: "l45-e-router-pix",
      from: "l45-router",
      to: "l45-pix",
      label: "原像素",
      via: [
        { x: 330, y: 180 },
        { x: 330, y: 292 },
      ],
      labelAt: { x: 292, y: 248 },
    },
    {
      id: "l45-e-sub-topk",
      from: "l45-sub",
      to: "l45-topk",
      via: [
        { x: 540, y: 68 },
        { x: 540, y: 180 },
      ],
      label: "无字幕则 0",
      labelAt: { x: 572, y: 92 },
    },
    {
      id: "l45-e-mid-topk",
      from: "l45-mid",
      to: "l45-topk",
      label: "默认可召回",
      labelAt: { x: 536, y: 156 },
    },
    {
      id: "l45-e-pix-topk",
      from: "l45-pix",
      to: "l45-topk",
      via: [
        { x: 540, y: 292 },
        { x: 540, y: 180 },
      ],
      label: "账单按帧涨",
      labelAt: { x: 572, y: 268 },
    },
    {
      id: "l45-e-topk-read",
      from: "l45-topk",
      to: "l45-read",
      label: "未入 k 看不见",
      labelAt: { x: 748, y: 228 },
    },
  ],
  steps: [
    {
      title: "查询先选层",
      description:
        "同一句问话可以打字幕索引、中间特征向量，或像素 / patch 索引。层选错，后面的阅读器再强也看不到目标段。",
      focus: ["l45-query", "l45-router", "l45-e-query-router"],
    },
    {
      title: "三层索引各记不同证据",
      description:
        "字幕层记 ASR 和 OCR；中间层记片段在图文空间里的向量；像素层记帧或 patch。无对白的动作只存在后两层。",
      focus: [
        "l45-sub",
        "l45-mid",
        "l45-pix",
        "l45-e-router-sub",
        "l45-e-router-mid",
        "l45-e-router-pix",
      ],
    },
    {
      title: "用 Recall@k 验收召回",
      description:
        "只有进入该层 Top-k 的片段会进入阅读器。目标不在该层索引里时，任意 k 的召回都是 0。",
      focus: ["l45-topk", "l45-e-sub-topk", "l45-e-mid-topk", "l45-e-pix-topk"],
    },
    {
      title: "精读只看召回集合",
      description:
        "阅读器按片段消耗视觉 token。像素层 k 稍大就会超过预算；中间层召回后再对少数片段精读，才能同时保住命中和账单。",
      focus: ["l45-topk", "l45-read", "l45-e-topk-read"],
    },
    {
      title: "工具调用带着视觉参数",
      description:
        "选层之后仍可再调 OCR、检测或裁剪。工具参数必须指向已召回的时间戳和框，不能对整小时视频再扫一遍像素。",
      focus: ["l45-router", "l45-mid", "l45-read"],
    },
  ],
  facts: [
    "Video-RAG（arXiv:2411.13093）在 Video-MME 上给 7 个开源 LVLM 平均加 2.8 个百分点，每条样本大约多 2.0K 辅助文本 token。",
    "KAIST VideoRAG（arXiv:2501.05874）表 2：只看视觉特征 R@1=0.054，只看文本 0.088，二者集成 0.103。",
    "CLIP4Clip（arXiv:2104.08860）在 MSR-VTT Training-9K 上 mean-pool R@1=43.1，时序 Transformer 为 44.5。",
    "Goldfish（arXiv:2407.12679）在 TVQA-long 上 41.78%；关掉检索、只均匀抽帧时约 25.07%。",
    "第 14 课扩的是位置编码窗口，第 39 课管的是子目标栈；本课管的是证据先进入哪一层索引。",
  ],
};
