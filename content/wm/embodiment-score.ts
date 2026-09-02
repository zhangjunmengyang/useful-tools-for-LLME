export type Answer = "yes" | "no" | "part";

export type Rung = "E0" | "E1" | "E2" | "E3" | "E4" | "E5";

export const EMBODIMENT_QUESTIONS = [
  "动作对换是否分岔？",
  "预测是否用于选动作或过滤动作？",
  "传感器与执行器是否共享同一物理世界？",
  "失败之后是否可以无限重置？",
  "接触或碰撞是否进入代价或安全层？",
  "是否存在控制不了、却必须预测的他者？",
  "不确定时停下是否算成功？",
] as const;

export type EmbodimentScore = {
  system: string;
  answers: readonly Answer[];
  envIsModelOnly: boolean;
  kind: "design" | "log";
  evidence?: Partial<Record<`q${1 | 2 | 3 | 4 | 5 | 6 | 7}`, string>>;
};

export type RungResult = {
  rung: Rung;
  reasons: readonly string[];
};

export function inferRung(score: {
  answers: readonly Answer[];
  envIsModelOnly?: boolean;
}): RungResult {
  const q = score.answers;
  const reasons: string[] = [];
  if (q.length !== 7) {
    return { rung: "E0", reasons: ["是否题必须是七条。"] };
  }

  if (q[0] === "no") {
    const openLoopBody =
      q[1] !== "yes" && (q[2] === "yes" || q[2] === "part");
    if (openLoopBody) {
      reasons.push("Q1 否、Q2 否、有身体：开环策略，没有可对换的前向模型，E3。");
      return { rung: "E3", reasons };
    }
    reasons.push("Q1 否：没有动作条件，又没有开环真机策略，停在 E0。");
    return { rung: "E0", reasons };
  }

  const usesModel = q[1] === "yes";
  const physical = q[2] === "yes";
  const physicalPart = q[2] === "part";
  const inWorld = physical || physicalPart;

  if (!usesModel) {
    if (inWorld) {
      reasons.push("Q2 否且 Q3 是或部分：有身体或摄像头，选动作不查询模型，E3。");
      return { rung: "E3", reasons };
    }
    reasons.push("Q2 否且无物理身体：至多有离线动力学，E1。");
    return { rung: "E1", reasons };
  }

  if (!inWorld) {
    if (score.envIsModelOnly) {
      reasons.push("Q2 是，动作只在模型内部 rollout，E1。");
      return { rung: "E1", reasons };
    }
    reasons.push("Q2 是，动作送给模型之外的环境（通常是 gym），E2。");
    if (q[3] === "yes") {
      reasons.push("Q4 是：失败可重置，这是 E2 的典型形态，不是升到 E4 的条件。");
    }
    return { rung: "E2", reasons };
  }

  const e5ready =
    physical &&
    q[4] !== "no" &&
    q[5] === "yes" &&
    q[6] === "yes";
  if (e5ready) {
    reasons.push("Q2 是、Q3 是，且接触、他者、停下三条齐，E5。");
    return { rung: "E5", reasons };
  }

  reasons.push("Q2 是，且有物理身体或摄像头档，模型在动作回路里，E4。");
  if (physicalPart) {
    reasons.push("Q3 部分：接触代价仍是想象中的，不到 E5。");
  }
  return { rung: "E4", reasons };
}

export const EXAMPLE_SYSTEMS = [
  {
    id: "sora",
    title: "无动作视频生成",
    answers: ["no", "no", "no", "yes", "no", "no", "no"] as const satisfies readonly Answer[],
    envIsModelOnly: true,
    note: "没有动作端口。分辨率不升档。",
  },
  {
    id: "wm",
    title: "CarRacing World Models",
    answers: ["yes", "yes", "no", "yes", "no", "no", "part"] as const satisfies readonly Answer[],
    envIsModelOnly: false,
    note: "动作送给 gym。可重置。没有物理身体。",
  },
  {
    id: "iris-dream",
    title: "IRIS 梦里玩",
    answers: ["yes", "yes", "no", "yes", "no", "no", "no"] as const satisfies readonly Answer[],
    envIsModelOnly: true,
    note: "动作只在世界模型内部步进，不调用外部 env.step。",
  },
  {
    id: "act",
    title: "真机 ACT",
    answers: ["no", "no", "yes", "no", "part", "no", "no"] as const satisfies readonly Answer[],
    envIsModelOnly: false,
    note: "有身体，开环策略。成功不等于查询了世界模型。",
  },
  {
    id: "pet",
    title: "带安全层的桌宠（摄像头档）",
    answers: ["yes", "yes", "part", "no", "part", "yes", "yes"] as const satisfies readonly Answer[],
    envIsModelOnly: false,
    note: "模型在回路里。Q3 部分，接触仍是想象，停在 E4。",
  },
] as const;
