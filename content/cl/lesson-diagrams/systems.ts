import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

const TITLES: Record<string, { title: string; summary: string; nodes: string[] }> =
  {
    "13": {
      title: "工作记忆、日记、名录三层抽屉",
      summary: "新信息先决定写进哪一层，冲突时再决定覆盖还是并存。",
      nodes: ["新事实", "工作记忆", "情节日记", "语义名录", "召回", "回答"],
    },
    "14": {
      title: "定位一层 MLP 再改一条事实",
      summary: "编辑要同时过可靠性、泛化、局部性、流畅性。",
      nodes: ["目标事实", "定位层", "关键", "写入", "邻居事实", "四指标"],
    },
    "15": {
      title: "长序列上学习速度自己掉下来",
      summary: "死神经元变多。continual backprop 把低使用率单元重置。",
      nodes: ["长任务流", "反向传播", "饱和单元", "学习速度", "重置", "后期任务"],
    },
    "16": {
      title: "四类经验要写到不同位置",
      summary: "事实可外挂，技能和推理规则通常要进权重。",
      nodes: ["新经验", "上下文", "外挂记忆", "编辑", "改权重", "通过矩阵"],
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
        title: "经验进门",
        description: spec.summary,
        focus: [nodes[0].id, edges[0].id],
      },
      {
        title: "选择写入位置",
        description: "上下文、记忆、编辑、权重不是同一件事。",
        focus: [nodes[1].id, nodes[2].id],
      },
      {
        title: "冲突与约束",
        description: "覆盖、并存或投影，都要留下可测的结果。",
        focus: [nodes[3].id, nodes[4].id],
      },
      {
        title: "新旧一起测",
        description: "只看新任务会高估方法。",
        focus: [nodes[5].id],
      },
    ],
    facts: [
      "外挂记忆不改慢权重。",
      "知识编辑是局部改事实，不是持续学习主循环。",
      "可塑性丢失和灾难性遗忘要分开量。",
    ],
  };
}

export const systemDiagrams: LessonDiagram[] = Object.entries(TITLES).map(
  ([lessonId, spec]) => makeDiagram(lessonId, spec),
);
