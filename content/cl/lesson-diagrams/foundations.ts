import type { LessonDiagram } from "@/lib/lesson-diagrams/types";

const TITLES: Record<string, { title: string; summary: string; nodes: string[] }> =
  {
    "01": {
      title: "顺序训练如何抹掉旧决策边界",
      summary: "任务 A 训完后接着训任务 B，同一组权重被新梯度推走。",
      nodes: ["任务 A 数据", "权重 θ", "任务 B 数据", "新梯度", "旧准确率", "热力图"],
    },
    "02": {
      title: "稳定性与可塑性此消彼长",
      summary: "冻骨干、降学习率、混旧数据，四个点落在平面不同位置。",
      nodes: ["旧任务", "新任务", "冻骨干", "小学习率", "混旧数据", "平面图"],
    },
    "03": {
      title: "平均准确率会被最后任务抬起来",
      summary: "同一张结果矩阵，换指标会得到完全不同的排序。",
      nodes: ["结果矩阵", "平均准确率", "遗忘", "BWT", "FWT", "协议"],
    },
    "04": {
      title: "上下文、检索、权重三条路",
      summary: "名录在时都会；名录撤掉后，只有写进权重或长期记忆的还在。",
      nodes: ["员工名录", "塞进 prompt", "检索", "改权重", "撤掉名录", "还能否叫小王"],
    },
    "05": {
      title: "Fisher 决定哪些权重该被钉住",
      summary: "EWC 用旧任务曲率当弹簧劲度，λ 扫过稳定性-可塑性平面。",
      nodes: ["旧任务损失", "Fisher 对角", "弹簧", "新任务梯度", "λ", "更新后的 θ"],
    },
    "06": {
      title: "回放缓冲把旧样本混进新 batch",
      summary: "缓冲大小和蒸馏项共同决定遗忘曲线。",
      nodes: ["新样本", "缓冲", "采样", "分类损失", "logits 蒸馏", "联合更新"],
    },
    "07": {
      title: "新知识写进新格子，旧格子上锁",
      summary: "PackNet 掩码和 prompt 池把容量显式切开。",
      nodes: ["共享骨干", "任务 1 掩码", "任务 2 掩码", "prompt 池", "任务头", "容量上限"],
    },
    "08": {
      title: "梯度投影与从头重训",
      summary: "A-GEM 把更新推进可行半平面；GDumb 只用缓冲重训。",
      nodes: ["新梯度", "旧任务梯度", "投影", "可行半平面", "缓冲", "GDumb 重训"],
    },
    "09": {
      title: "领域数据与通用数据抢学习率",
      summary: "续预训练时混入通用数据，相当于第 06 课的回放。",
      nodes: ["领域语料", "通用语料", "配比", "学习率", "领域分数", "通用分数"],
    },
    "10": {
      title: "指令任务接龙填 4×4 矩阵",
      summary: "每个任务结束，把前面全部任务重测一遍。",
      nodes: ["任务 1", "任务 2", "任务 3", "任务 4", "重测", "遗忘矩阵"],
    },
    "11": {
      title: "两个 LoRA 方向抢同一子空间",
      summary: "正交约束让夹角接近 90°，减少互相覆盖。",
      nodes: ["任务 1 LoRA", "任务 2 LoRA", "内积", "正交损失", "夹角", "旧任务保持"],
    },
    "12": {
      title: "训练结束后把任务向量加起来",
      summary: "合并是事后缝合。TIES 处理符号冲突。",
      nodes: ["模型 A", "模型 B", "任务向量", "相加", "TIES 修剪", "合成模型"],
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
    "transform",
    "state",
    "decision",
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
        title: "输入",
        description: spec.summary,
        focus: [nodes[0].id, edges[0].id],
      },
      {
        title: "变换",
        description: "中间模块把新经验写成可存储的形式。",
        focus: [nodes[1].id, nodes[2].id, edges[1].id],
      },
      {
        title: "写入约束",
        description: "约束、缓冲或正交决定旧知识还能不能保住。",
        focus: [nodes[3].id, nodes[4].id],
      },
      {
        title: "验收",
        description: "同时看新任务会了没、旧任务还在不在。",
        focus: [nodes[5].id],
      },
    ],
    facts: [
      "本图是教学机制图，不是论文网络结构照抄。",
      "节点从左到右是数据或更新流，不是层数。",
      "验收必须同时包含新任务和旧任务。",
    ],
  };
}

export const foundationDiagrams: LessonDiagram[] = Object.entries(TITLES).map(
  ([lessonId, spec]) => makeDiagram(lessonId, spec),
);
