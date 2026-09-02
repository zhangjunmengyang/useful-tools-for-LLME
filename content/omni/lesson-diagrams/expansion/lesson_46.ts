import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson46Diagram: LessonDiagram = {
  lessonId: "46",
  title: "把异构前向拆成可独立调度的 stage graph",
  summary:
    "三条请求先经编排器按模态切开；视觉编码、语言 decode 与 flow / 动作专家各用自己的 batch 维和 KV 页，再经连接器交出输出。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l46-req",
      label: ["三条请求"],
      meta: "文本 / 带图 / 动作",
      kind: "input",
      x: 88,
      y: 180,
    },
    {
      id: "l46-orch",
      label: ["stage 编排"],
      meta: "拒绝单条 CUDA graph",
      kind: "decision",
      x: 250,
      y: 180,
    },
    {
      id: "l46-enc",
      label: ["理解编码"],
      meta: "变长视觉 batch",
      kind: "transform",
      x: 430,
      y: 68,
    },
    {
      id: "l46-ar",
      label: ["语言 decode"],
      meta: "AR · 独立 KV",
      kind: "transform",
      x: 430,
      y: 180,
    },
    {
      id: "l46-flow",
      label: ["flow / 动作"],
      meta: "时间步循环",
      kind: "transform",
      x: 430,
      y: 292,
    },
    {
      id: "l46-kv",
      label: ["KV 与连接器"],
      meta: "跨 stage 只传条件",
      kind: "state",
      x: 640,
      y: 180,
    },
    {
      id: "l46-out",
      label: ["分阶段输出"],
      meta: "文本 / 图 / 动作块",
      kind: "output",
      x: 830,
      y: 180,
      width: 150,
    },
  ],
  edges: [
    {
      id: "l46-e-req-orch",
      from: "l46-req",
      to: "l46-orch",
      label: "路由",
      labelAt: { x: 168, y: 152 },
    },
    {
      id: "l46-e-orch-enc",
      from: "l46-orch",
      to: "l46-enc",
      label: "只批视觉",
      via: [
        { x: 318, y: 180 },
        { x: 318, y: 68 },
      ],
      labelAt: { x: 286, y: 108 },
    },
    {
      id: "l46-e-orch-ar",
      from: "l46-orch",
      to: "l46-ar",
      label: "AR 请求",
      labelAt: { x: 338, y: 156 },
    },
    {
      id: "l46-e-orch-flow",
      from: "l46-orch",
      to: "l46-flow",
      label: "动作请求",
      via: [
        { x: 318, y: 180 },
        { x: 318, y: 292 },
      ],
      labelAt: { x: 268, y: 252 },
    },
    {
      id: "l46-e-enc-ar",
      from: "l46-enc",
      to: "l46-ar",
      label: "视觉 embedding",
      labelAt: { x: 478, y: 118 },
    },
    {
      id: "l46-e-ar-kv",
      from: "l46-ar",
      to: "l46-kv",
      label: "AR KV 页",
      labelAt: { x: 536, y: 156 },
    },
    {
      id: "l46-e-kv-flow",
      from: "l46-kv",
      to: "l46-flow",
      label: "条件，隔离页",
      via: [
        { x: 640, y: 292 },
      ],
      labelAt: { x: 572, y: 268 },
    },
    {
      id: "l46-e-kv-out",
      from: "l46-kv",
      to: "l46-out",
      label: "流式交出",
      labelAt: { x: 736, y: 152 },
    },
  ],
  steps: [
    {
      title: "三条异构请求到达",
      description:
        "纯文本、带图理解、带动作专家的请求不能假定同一套循环、同一套张量形状。",
      focus: ["l46-req", "l46-orch", "l46-e-req-orch"],
    },
    {
      title: "编排器按 stage 切开",
      description:
        "vLLM-Omni 把 any-to-any 写成节点加边。一条 CUDA graph 锁住形状与控制流，不能同时表达 AR 逐步解码和 flow 时间步。",
      focus: [
        "l46-orch",
        "l46-enc",
        "l46-ar",
        "l46-flow",
        "l46-e-orch-enc",
        "l46-e-orch-ar",
        "l46-e-orch-flow",
      ],
    },
    {
      title: "视觉编码只批有图的请求",
      description:
        "变长 patch 按本 stage 的 max 做 padding，mask 关掉无效位置。纯文本请求不进入这一段。",
      focus: ["l46-enc", "l46-ar", "l46-e-enc-ar"],
    },
    {
      title: "语言 decode 持有独立 KV 页",
      description:
        "PagedAttention 允许同模型内分页；跨 stage 只允许把 hidden state 当条件，不允许动作专家读写 AR 的物理页。",
      focus: ["l46-ar", "l46-kv", "l46-e-ar-kv"],
    },
    {
      title: "flow / 动作专家用自己的时间步维",
      description:
        "动作块形状是 H×d，循环次数是积分步。把步数垫进 token 序列，等于用错误的 batch 维去捕获图。",
      focus: ["l46-flow", "l46-kv", "l46-e-kv-flow"],
    },
    {
      title: "连接器流式交出输出",
      description:
        "Talker 不必等 Thinker 整段结束才能开始；Vocoder 也可以吃到部分 codec。JCT 来自分阶段重叠，不是来自合成一条图。",
      focus: ["l46-kv", "l46-out", "l46-e-kv-out"],
    },
  ],
  facts: [
    "vLLM-Omni 在 Qwen3-Omni 上相对 Transformers 基线把 JCT 最多降 91.4%、RTF 降 90.7%（论文 §4.2）。",
    "Qwen3-Omni 的 Thinker 约 30B，Talker 更小但更吃算力；论文把更多加速器内存分给 Thinker（§3.3）。",
    "视频输入任务上平均输入 841.6 token、文本输出 150.9、音频输出 545.4，Talker 迭代次数多于 Thinker（§4.2）。",
    "Qwen2.5-Omni 上 Thinker2Talker 共享内存 5.49 ms、Mooncake 8.28 ms，相对数十秒推理可忽略（Table 1）。",
    "MiMo-Audio 打开 execution-graph compilation 后 RTF 从 0.60 降到 0.12；图是按 stage 编译的，不是跨理解与生成合图（§4.2）。",
  ],
};
