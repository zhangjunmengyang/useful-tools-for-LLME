import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

export const lesson24Diagram: LessonDiagram = {
  lessonId: "24",
  title: "动作进入与文字共享的 next-token",
  summary:
    "图像和指令先进入同一套 VLM 骨干；输出可走纯文字、离散动作 token，或文字映射技能。",
  viewBox: "0 0 960 360",
  nodes: [
    {
      id: "l24-image",
      label: ["相机图像"],
      meta: "RGB 观测",
      kind: "input",
      x: 100,
      y: 88,
      width: 150,
    },
    {
      id: "l24-instruction",
      label: ["语言指令"],
      meta: "pick / place",
      kind: "input",
      x: 100,
      y: 260,
      width: 150,
    },
    {
      id: "l24-vlm",
      label: ["VLM 骨干"],
      meta: "共享自注意力",
      kind: "transform",
      x: 310,
      y: 174,
      width: 158,
    },
    {
      id: "l24-route",
      label: ["输出通路"],
      meta: "文字 / token / 技能",
      kind: "decision",
      x: 500,
      y: 174,
      width: 158,
    },
    {
      id: "l24-bins",
      label: ["7 维分箱"],
      meta: "V + dB + b_d",
      kind: "state",
      x: 690,
      y: 88,
      width: 150,
    },
    {
      id: "l24-skills",
      label: ["技能打分"],
      meta: "LLM × 可行性",
      kind: "state",
      x: 690,
      y: 260,
      width: 150,
    },
    {
      id: "l24-action",
      label: ["末端指令"],
      meta: "位移 / 夹爪",
      kind: "output",
      x: 860,
      y: 174,
      width: 150,
    },
  ],
  edges: [
    {
      id: "l24-e-image-vlm",
      from: "l24-image",
      to: "l24-vlm",
      label: "视觉 token",
      via: [{ x: 210, y: 88 }, { x: 210, y: 174 }],
      labelAt: { x: 168, y: 128 },
    },
    {
      id: "l24-e-inst-vlm",
      from: "l24-instruction",
      to: "l24-vlm",
      label: "文本 token",
      via: [{ x: 210, y: 260 }, { x: 210, y: 174 }],
      labelAt: { x: 168, y: 214 },
    },
    {
      id: "l24-e-vlm-route",
      from: "l24-vlm",
      to: "l24-route",
      label: "隐藏状态",
      labelAt: { x: 404, y: 150 },
    },
    {
      id: "l24-e-route-bins",
      from: "l24-route",
      to: "l24-bins",
      label: "离散动作",
      via: [{ x: 590, y: 174 }, { x: 590, y: 88 }],
      labelAt: { x: 548, y: 118 },
    },
    {
      id: "l24-e-route-skills",
      from: "l24-route",
      to: "l24-skills",
      label: "技能名",
      via: [{ x: 590, y: 174 }, { x: 590, y: 260 }],
      labelAt: { x: 548, y: 222 },
    },
    {
      id: "l24-e-bins-action",
      from: "l24-bins",
      to: "l24-action",
      label: "反分箱",
      via: [{ x: 790, y: 88 }, { x: 790, y: 174 }],
      labelAt: { x: 748, y: 118 },
    },
    {
      id: "l24-e-skills-action",
      from: "l24-skills",
      to: "l24-action",
      label: "低层政策",
      via: [{ x: 790, y: 260 }, { x: 790, y: 174 }],
      labelAt: { x: 748, y: 222 },
    },
  ],
  steps: [
    {
      title: "读入图像与指令",
      description:
        "相机帧和自然语言先被编码成可进入同一 Transformer 的向量；PaLM-E 还允许把连续状态插进多模态句子。",
      focus: ["l24-image", "l24-instruction", "l24-e-image-vlm", "l24-e-inst-vlm"],
    },
    {
      title: "共享骨干",
      description:
        "VLM 在交错序列上做因果预测。到这一步，系统仍可以只生成文字，机械臂不会动。",
      focus: ["l24-vlm", "l24-e-vlm-route"],
    },
    {
      title: "选择输出通路",
      description:
        "纯文字停在描述；离散动作 token 直接占词表；技能通路先生成或打分技能名，再交给低层控制器。",
      focus: ["l24-route", "l24-bins", "l24-skills"],
    },
    {
      title: "7 维均匀分箱",
      description:
        "末端位姿每维切成 B 个 bin，token id 为语言词表大小加上维度偏移。RT-1 / RT-2 原文用 B=256。",
      focus: ["l24-bins", "l24-e-route-bins", "l24-e-bins-action"],
    },
    {
      title: "技能可行性加权",
      description:
        "SayCan 用语言模型分数乘技能价值函数；没有物体时对应技能的可行性接近 0。",
      focus: ["l24-skills", "l24-e-route-skills", "l24-e-skills-action"],
    },
    {
      title: "发出末端指令",
      description:
        "分箱整数反量化成连续位移，或技能名调用已有政策。场景唯一时错指令可能不改动作。",
      focus: ["l24-action"],
    },
  ],
  facts: [
    "RT-1 把臂的 7 维、底盘 3 维和 terminate 各自均匀切成 256 个 bin，35M 参数模型以 3 Hz 出动作。",
    "RT-2 把同一套 256 bin 写成文本 token，与网络图文任务共微调，不新增动作专用层。",
    "SayCan 的技能选择是 p(技能|指令) 乘 p(成功|状态,技能)；PaLM-SayCan 在 mock kitchen 的规划成功率是 84%。",
    "2603.19233 在 X-VLA 上测到：libero_goal 错提示成功率从 94% 降到 10%；libero_object 可保持 60–100%。",
  ],
};
