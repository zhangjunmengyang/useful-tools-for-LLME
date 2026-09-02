"use client";

import { useState } from "react";
import { LabShell } from "@/components/labs/LabShell";

const heads = [
  {
    id: "reason",
    title: "理解头 / Reasoner",
    answers: "杯子是什么、指令是什么。",
    fall: "它不模拟下一秒。克制不能只靠它。",
  },
  {
    id: "gen",
    title: "生成头 / Generator",
    answers: "接下来画面（和可选声音）长什么样。",
    fall: "没有动作条件就是画师。有动作条件才可能回答会不会倒。",
  },
  {
    id: "policy",
    title: "动作头 / Policy",
    answers: "现在该伸手还是停下。",
    fall: "默认 E3。要到 E4，必须查询前向模型再改写动作。",
  },
] as const;

export default function ThreeHeadsLab({
  onComplete,
}: {
  onComplete?: (state?: Record<string, unknown>) => void;
}) {
  const [id, setId] = useState<(typeof heads)[number]["id"]>("policy");
  const current = heads.find((item) => item.id === id) ?? heads[2];

  return (
    <LabShell
      brief="任务：伸手拿杯。三个头各自在回答什么。杯子会不会倒，只有前向模型能直接说。"
      verdict={`${current.title}：${current.fall}`}
      tone={id === "gen" ? "ok" : "warn"}
    >
      <div className="wm-lab-toolbar">
        {heads.map((item) => (
          <button
            key={item.id}
            type="button"
            aria-pressed={id === item.id}
            onClick={() => setId(item.id)}
          >
            {item.title}
          </button>
        ))}
        <button type="button" onClick={() => onComplete?.({ head: id })}>
          记下分工
        </button>
      </div>
      <ul className="wm-lab-list">
        <li>
          <strong>它在回答</strong>
          {current.answers}
        </li>
      </ul>
    </LabShell>
  );
}
