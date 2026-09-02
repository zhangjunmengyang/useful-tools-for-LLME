"use client";

import { useState } from "react";
import { LabShell } from "@/components/labs/LabShell";

const systems = [
  {
    id: "oasis",
    title: "open-oasis",
    io: "动作进，下一帧出。有 500M 开源权重。",
    guessHint: "可玩，但滑窗会忘。",
  },
  {
    id: "cosmos3",
    title: "Cosmos 3",
    io: "语言、图像、视频、声音、动作可组合。大权重。",
    guessHint: "配置决定它像 VLM 还是像模拟器。",
  },
  {
    id: "genie3",
    title: "Genie 3",
    io: "文本进，实时可玩世界出。无公开权重。",
    guessHint: "数分钟一致性是官方宣称。",
  },
  {
    id: "dinowm",
    title: "DINO-WM",
    io: "观察加低层动作进，patch 特征出。可规划。",
    guessHint: "画质不是它的产品。",
  },
  {
    id: "t2v",
    title: "文生视频（Sora 类）",
    io: "文本或首帧进，视频出。通常没有动作端口。",
    guessHint: "生成真可以很高。",
  },
] as const;

const benches = [
  {
    id: "worldscore",
    title: "WorldScore",
    measures: "下一场景生成：可控、画质、动态。3D/4D/视频共用这把尺子。",
    helpsPet: false,
  },
  {
    id: "wmb",
    title: "WorldModelBench",
    measures: "视频生成器当世界模型用时，常识、指令、物理哪条先坏。",
    helpsPet: false,
  },
  {
    id: "phys",
    title: "Physics-IQ",
    measures: "真实物理实验的续写对不对，和观感可以脱钩。",
    helpsPet: false,
  },
  {
    id: "swap",
    title: "动作对换",
    measures: "同一观察只换动作，预测必须分岔。第 03 课的试金石。",
    helpsPet: true,
  },
  {
    id: "plan",
    title: "规划成功率",
    measures: "在模型里搜动作，任务有没有往前走。桌宠克制靠它。",
    helpsPet: true,
  },
] as const;

export default function LeaderboardLab({
  onComplete,
}: {
  onComplete?: (state?: Record<string, unknown>) => void;
}) {
  const [systemId, setSystemId] = useState<(typeof systems)[number]["id"]>("t2v");
  const [benchId, setBenchId] = useState<(typeof benches)[number]["id"]>("worldscore");
  const system = systems.find((item) => item.id === systemId) ?? systems[4];
  const bench = benches.find((item) => item.id === benchId) ?? benches[0];

  return (
    <LabShell
      brief="先选一个系统，再选一份榜。看这把尺子能不能回答桌宠要的问题。"
      verdict={`${system.title} 在「${bench.title}」上测的是：${bench.measures} ${
        bench.helpsPet
          ? "这根尺子能接到桌宠。"
          : "这根尺子好看，也不等于能做 MPC。"
      }`}
      tone={bench.helpsPet ? "ok" : "warn"}
    >
      <div className="wm-lab-toolbar">
        {systems.map((item) => (
          <button
            key={item.id}
            type="button"
            aria-pressed={systemId === item.id}
            onClick={() => setSystemId(item.id)}
          >
            {item.title}
          </button>
        ))}
      </div>
      <div className="wm-lab-toolbar">
        {benches.map((item) => (
          <button
            key={item.id}
            type="button"
            aria-pressed={benchId === item.id}
            onClick={() => setBenchId(item.id)}
          >
            {item.title}
          </button>
        ))}
        <button
          type="button"
          onClick={() => onComplete?.({ systemId, benchId })}
        >
          记下这张表
        </button>
      </div>
      <ul className="wm-lab-list">
        <li>
          <strong>这个系统</strong>
          {system.io} {system.guessHint}
        </li>
        <li>
          <strong>这把尺子</strong>
          {bench.measures}
        </li>
      </ul>
    </LabShell>
  );
}
