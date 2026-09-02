import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

const TITLES: Record<string, { title: string; summary: string; nodes: string[] }> =
  {
    "17": {
      title: "当前序列上的内环梯度更新",
      summary: "TTT 层的隐状态是可学习的 W，不是一条 RNN 向量。",
      nodes: ["当前 token", "记忆矩阵 W", "内环损失", "内环梯度", "更新 W", "下一步预测"],
    },
    "18": {
      title: "惊讶高的事件才写入长期记忆",
      summary: "常见 token 几乎不写，稀有 token 写入幅度更大。",
      nodes: ["token 流", "惊讶分数", "写入门", "长期记忆", "注意力", "当前预测"],
    },
    "19": {
      title: "快针和慢针同时转",
      summary: "不同更新频率的记忆对应不同时间尺度的信息。",
      nodes: ["token 钟", "序列钟", "任务钟", "快权重", "慢权重", "联合输出"],
    },
    "20": {
      title: "SFT 拉得远，on-policy 走得近",
      summary: "到原模型的距离和旧任务遗忘一起看。",
      nodes: ["原模型", "离线数据", "SFT", "on-policy RL", "KL", "旧任务保持"],
    },
    "21": {
      title: "成功的程序放进抽屉再检索",
      summary: "技能库是经验外存。权重可以不变。",
      nodes: ["环境任务", "写程序", "验证", "入库", "检索", "复用"],
    },
    "22": {
      title: "三条河只有环境河会持续涨技能",
      summary: "网页和专家包用完就停，交互还在产生新数据。",
      nodes: ["网页", "专家包", "环境交互", "筛选", "技能库", "增长曲线"],
    },
    "23": {
      title: "生成、筛选、训练、评测绕一圈",
      summary: "关掉筛选，错误数据会自我强化。",
      nodes: ["生成数据", "验证筛选", "微调自己", "评测", "旧任务", "下一轮"],
    },
    "24": {
      title: "14 个工作日三条曲线一起看",
      summary: "记忆条数、技能数、旧任务保持，缺一条就不是上岗。",
      nodes: ["Day 1", "每日任务", "记忆写入", "技能入库", "可选权重更新", "Day 14 回放"],
    },
  };

function makeDiagram(
  lessonId: string,
  spec: { title: string; summary: string; nodes: string[] },
): LessonDiagram {
  const xs = [90, 250, 410, 570, 730, 880];
  const kinds = [
    "input",
    "transform",
    "state",
    "decision",
    "transform",
    "output",
  ] as const;
  const nodes = spec.nodes.slice(0, 6).map((label, index) => ({
    id: `l${lessonId}-n${index}`,
    label: [label] as const,
    kind: kinds[index],
    x: xs[index],
    y: 180,
    width: 118,
  }));
  const edges = nodes.slice(0, -1).map((node, index) => ({
    id: `l${lessonId}-e${index}`,
    from: node.id,
    to: nodes[index + 1].id,
  }));
  return {
    lessonId,
    title: spec.title,
    summary: spec.summary,
    viewBox: "0 0 960 360",
    nodes,
    edges,
    steps: [
      {
        title: "当前经验",
        description: spec.summary,
        focus: [nodes[0].id, edges[0].id],
      },
      {
        title: "内环或外存",
        description: "测试时更新、技能库和自编辑都在这一步分叉。",
        focus: [nodes[1].id, nodes[2].id],
      },
      {
        title: "筛选或门控",
        description: "惊讶、验证、on-policy，决定写什么。",
        focus: [nodes[3].id, nodes[4].id],
      },
      {
        title: "还在不在",
        description: "长期看旧任务和还能不能继续学。",
        focus: [nodes[5].id],
      },
    ],
    facts: [
      "内环更新不等于把基础模型存盘。",
      "技能库成功不代表权重学会了该技能。",
      "自改进环没有筛选就会放大错误。",
    ],
  };
}

export const trainingDiagrams: LessonDiagram[] = Object.entries(TITLES).map(
  ([lessonId, spec]) => makeDiagram(lessonId, spec),
);
