import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson32Diagram: LessonDiagram = {
  lessonId: "32",
  title: "同一套二维接地头服务屏幕与桌面",
  summary:
    "截图或俯视图先被编码，再叠编号或保留连续坐标；共享二维头在分类编号与回归 [0,1]^2 之间切换，最后分别发出点击或末端 xy。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l32-obs",
      label: ["视觉观察"],
      meta: "截图 / 桌面俯视",
      kind: "input",
      x: 88,
      y: 180,
    },
    {
      id: "l32-encode",
      label: ["视觉编码"],
      meta: "分辨率决定格子",
      kind: "transform",
      x: 248,
      y: 180,
    },
    {
      id: "l32-som",
      label: ["SoM 编号"],
      meta: "可点区域 1..K",
      kind: "transform",
      x: 408,
      y: 88,
    },
    {
      id: "l32-xy",
      label: ["归一化坐标"],
      meta: "[0,1]^2 / 256 bins",
      kind: "state",
      x: 408,
      y: 272,
    },
    {
      id: "l32-head",
      label: ["共享二维头"],
      meta: "分类 CE / 回归 L2",
      kind: "decision",
      x: 590,
      y: 180,
    },
    {
      id: "l32-click",
      label: ["GUI 动作"],
      meta: "click / type / scroll",
      kind: "output",
      x: 790,
      y: 88,
      width: 150,
    },
    {
      id: "l32-arm",
      label: ["末端 xy"],
      meta: "再接 6D / 夹爪",
      kind: "output",
      x: 790,
      y: 272,
      width: 150,
    },
  ],
  edges: [
    {
      id: "l32-e-obs-encode",
      from: "l32-obs",
      to: "l32-encode",
      label: "像素网格",
      labelAt: { x: 168, y: 228 },
    },
    {
      id: "l32-e-encode-som",
      from: "l32-encode",
      to: "l32-som",
      label: "叠 mark",
      via: [
        { x: 328, y: 180 },
        { x: 328, y: 88 },
      ],
      labelAt: { x: 286, y: 124 },
    },
    {
      id: "l32-e-encode-xy",
      from: "l32-encode",
      to: "l32-xy",
      label: "x/W, y/H",
      via: [
        { x: 328, y: 180 },
        { x: 328, y: 272 },
      ],
      labelAt: { x: 286, y: 236 },
    },
    {
      id: "l32-e-som-head",
      from: "l32-som",
      to: "l32-head",
      label: "选编号",
      labelAt: { x: 500, y: 64 },
    },
    {
      id: "l32-e-xy-head",
      from: "l32-xy",
      to: "l32-head",
      label: "回归点",
      labelAt: { x: 500, y: 296 },
    },
    {
      id: "l32-e-head-click",
      from: "l32-head",
      to: "l32-click",
      label: "屏幕动作",
      via: [
        { x: 690, y: 180 },
        { x: 690, y: 88 },
      ],
      labelAt: { x: 710, y: 124 },
    },
    {
      id: "l32-e-head-arm",
      from: "l32-head",
      to: "l32-arm",
      label: "桌面 xy",
      via: [
        { x: 690, y: 180 },
        { x: 690, y: 272 },
      ],
      labelAt: { x: 710, y: 236 },
    },
  ],
  steps: [
    {
      title: "同一观察接口",
      description:
        "GUI 截图和桌面俯视都先变成一张图。后续头不关心像素来自浏览器还是相机。",
      focus: ["l32-obs", "l32-encode", "l32-e-obs-encode"],
    },
    {
      title: "两条接地接口",
      description:
        "SoM 把可操作区域编成 1..K；连续头把像素位置除以宽高，落到 [0,1]^2，Magma 再切成 256 档。",
      focus: [
        "l32-encode",
        "l32-som",
        "l32-xy",
        "l32-e-encode-som",
        "l32-e-encode-xy",
      ],
    },
    {
      title: "共享二维头",
      description:
        "分类损失盯编号，回归损失盯归一化点。低分辨率时格子中心误差上升，正确编号的分类损失可以保持为 0。",
      focus: ["l32-som", "l32-xy", "l32-head", "l32-e-som-head", "l32-e-xy-head"],
    },
    {
      title: "动作空间在头之后分叉",
      description:
        "屏幕侧接 click / type / scroll；桌面侧先复用 xy，再补 z、姿态和夹爪。二维接地可以共享，完整动作空间不能假装相同。",
      focus: ["l32-head", "l32-click", "l32-arm", "l32-e-head-click", "l32-e-head-arm"],
    },
    {
      title: "分辨率验收",
      description:
        "把视觉网格降到 4×4 时，连续坐标误差必须大于 SoM；同一比例头在 UI 样本和桌面样本上误差都要下降。",
      focus: ["l32-encode", "l32-head", "l32-click", "l32-arm"],
    },
  ],
  facts: [
    "SeeClick 把点击写成两位小数的 [0,1] 坐标；命中定义为预测点落在真值框内。",
    "Magma 把坐标按图像高宽归一化后量化到 256 档，并用 SoM 做动作接地、ToM 做轨迹规划。",
    "OS-Atlas 把跨平台动作先收成 click / type / scroll 三个基本动作，再允许自定义动作。",
    "CogAgent 用 224 低分辨率分支加 1120 高分辨率交叉注意力读小字和图标。",
    "UI-TARS 官方报告（arXiv:2501.12326）用统一动作空间直接从截图预测 (x,y)，并在动作前生成 thought。",
  ],
};
