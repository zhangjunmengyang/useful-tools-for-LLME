import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson28Diagram: LessonDiagram = {
  lessonId: "28",
  title: "从噪声动作块积出一段连续轨迹",
  summary:
    "VLM 条件进入动作专家，专家在动作块上回归速度场，Euler 积分后再执行前 k 步并重规划。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l28-obs",
      label: ["观察 o_t"],
      meta: "图像 / 语言 / 关节",
      kind: "input",
      x: 88,
      y: 86,
    },
    {
      id: "l28-noise",
      label: ["噪声块 A⁰"],
      meta: "N(0, I), 形状 H×d",
      kind: "input",
      x: 88,
      y: 274,
    },
    {
      id: "l28-vlm",
      label: ["VLM 条件"],
      meta: "x_vlm，可缓存 KV",
      kind: "transform",
      x: 278,
      y: 86,
    },
    {
      id: "l28-expert",
      label: ["动作专家"],
      meta: "π0 约 300M",
      kind: "transform",
      x: 278,
      y: 274,
    },
    {
      id: "l28-vel",
      label: ["速度场 v_θ"],
      meta: "v_θ(a_t, t, x_vlm)",
      kind: "state",
      x: 478,
      y: 180,
    },
    {
      id: "l28-euler",
      label: ["积分是否完成"],
      meta: "π0 用 10 步 Euler",
      kind: "decision",
      x: 668,
      y: 180,
    },
    {
      id: "l28-chunk",
      label: ["动作块 A"],
      meta: "A ∈ R^{H×d}",
      kind: "output",
      x: 858,
      y: 86,
    },
    {
      id: "l28-exec",
      label: ["执行前 k 步"],
      meta: "然后重规划",
      kind: "transform",
      x: 858,
      y: 274,
    },
  ],
  edges: [
    {
      id: "l28-e-obs-vlm",
      from: "l28-obs",
      to: "l28-vlm",
      label: "编码观察",
      labelAt: { x: 183, y: 58 },
    },
    {
      id: "l28-e-vlm-expert",
      from: "l28-vlm",
      to: "l28-expert",
      label: "条件 x_vlm",
      via: [{ x: 278, y: 180 }],
      labelAt: { x: 214, y: 176 },
    },
    {
      id: "l28-e-noise-expert",
      from: "l28-noise",
      to: "l28-expert",
      label: "噪声动作",
      labelAt: { x: 183, y: 312 },
    },
    {
      id: "l28-e-expert-vel",
      from: "l28-expert",
      to: "l28-vel",
      label: "回归速度",
      via: [{ x: 390, y: 274 }, { x: 390, y: 180 }],
      labelAt: { x: 348, y: 228 },
    },
    {
      id: "l28-e-vel-euler",
      from: "l28-vel",
      to: "l28-euler",
      label: "Euler 一步",
      labelAt: { x: 573, y: 152 },
    },
    {
      id: "l28-e-euler-chunk",
      from: "l28-euler",
      to: "l28-chunk",
      label: "积到数据端",
      via: [{ x: 760, y: 180 }, { x: 760, y: 86 }],
      labelAt: { x: 704, y: 118 },
    },
    {
      id: "l28-e-chunk-exec",
      from: "l28-chunk",
      to: "l28-exec",
      label: "只执行前 k",
    },
    {
      id: "l28-e-exec-vlm",
      from: "l28-exec",
      to: "l28-vlm",
      label: "新观察再规划",
      via: [{ x: 760, y: 320 }, { x: 200, y: 320 }, { x: 200, y: 86 }],
      labelAt: { x: 470, y: 338 },
    },
  ],
  steps: [
    {
      title: "用 VLM 压观察",
      description:
        "多路相机、语言指令和关节角进入预训练 VLM，得到后续每一步积分都复用的条件 x_vlm。",
      focus: ["l28-obs", "l28-vlm", "l28-e-obs-vlm"],
    },
    {
      title: "从噪声块起步",
      description:
        "动作专家读入形状为 H×d 的高斯噪声块，而不是逐维离散 token。",
      focus: ["l28-noise", "l28-expert", "l28-e-noise-expert", "l28-e-vlm-expert"],
    },
    {
      title: "回归条件速度",
      description:
        "网络输出 v_θ(a_t, t, x_vlm)。直线路径下，监督目标是噪声减干净动作，或干净动作减噪声，取决于时间箭头。",
      focus: ["l28-expert", "l28-vel", "l28-e-expert-vel"],
    },
    {
      title: "有限步 Euler 积分",
      description:
        "π0 从噪声端走 10 步 Euler 到数据端。步数过少时，教学夹具里的轨迹仍靠近穿障的噪声直线。",
      focus: ["l28-vel", "l28-euler", "l28-chunk", "l28-e-vel-euler", "l28-e-euler-chunk"],
    },
    {
      title: "执行前 k 步再规划",
      description:
        "只执行 chunk 的前 k 步，再用新观察重新积分。k 等于 H 且目标被挪走时，剩余步仍指向旧目标。",
      focus: ["l28-chunk", "l28-exec", "l28-vlm", "l28-e-chunk-exec", "l28-e-exec-vlm"],
    },
  ],
  facts: [
    "π0 用 PaliGemma 3B 作 VLM 骨干，另加从零初始化的 300M 动作专家，合计 3.3B。",
    "π0 的动作块长度 H=50，灵巧任务控制频率最高 50 Hz，推理默认 10 步 Euler。",
    "π0 直线路径写 A^τ = τA + (1-τ)ε，目标速度为 A−ε；第 20 课约定与它差一个时间反号。",
    "Diffusion Policy 在动作空间回归噪声 ε，flow matching 回归速度场；两者都一次生成一段动作而不是一个词。",
  ],
};
