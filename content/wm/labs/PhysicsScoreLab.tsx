"use client";

import { useState } from "react";
import { LabShell } from "@/components/labs/LabShell";

const clips = [
  {
    id: "pretty",
    title: "好看但违守恒",
    look: "水花细、光影对、镜头稳。",
    physics: "倒进杯子的水比倒出的多。质量不守恒。",
    physScore: "Physics-IQ 会扣。",
    plan: "不能拿去判断杯子会不会满出来。",
  },
  {
    id: "rough",
    title: "糙但落点对",
    look: "边缘锯齿，材质假。",
    physics: "球的落点和真实实验录像对得上。",
    physScore: "Physics-IQ 相对更高。",
    plan: "仍不是规划成功率。只说明续写更接近那次实验。",
  },
] as const;

export default function PhysicsScoreLab({
  onComplete,
}: {
  onComplete?: (state?: Record<string, unknown>) => void;
}) {
  const [id, setId] = useState<(typeof clips)[number]["id"]>("pretty");
  const current = clips.find((item) => item.id === id) ?? clips[0];

  return (
    <LabShell
      brief="同一物理实验两段续写。先用观感打分，再按 Physics-IQ 的规则看。"
      verdict={`${current.title}：${current.physScore} ${current.plan}`}
      tone={id === "pretty" ? "bad" : "ok"}
    >
      <div className="wm-lab-toolbar">
        {clips.map((item) => (
          <button
            key={item.id}
            type="button"
            aria-pressed={id === item.id}
            onClick={() => setId(item.id)}
          >
            {item.title}
          </button>
        ))}
        <button type="button" onClick={() => onComplete?.({ clip: id })}>
          记下三列
        </button>
      </div>
      <ul className="wm-lab-list">
        <li>
          <strong>画面真</strong>
          {current.look}
        </li>
        <li>
          <strong>物理对</strong>
          {current.physics}
        </li>
        <li>
          <strong>规划好</strong>
          {current.plan}
        </li>
      </ul>
    </LabShell>
  );
}
