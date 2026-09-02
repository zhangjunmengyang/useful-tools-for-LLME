"use client";

import { useState } from "react";
import { LabShell } from "@/components/labs/LabShell";

const factors = [
  {
    id: "steer",
    title: "自车转向 / 速度",
    kind: "动作 a_t",
    why: "规划器选得了。对换必须分岔。",
  },
  {
    id: "other",
    title: "对向来车 / 行人",
    kind: "外生事件",
    why: "选不了，只能预测。桌宠里的「对面那个人」同类。",
  },
  {
    id: "rain",
    title: "下雨 / 夜晚",
    kind: "风格或条件",
    why: "通常不是动作。当成动作会让模型以为方向盘能关掉雨。",
  },
  {
    id: "lane",
    title: "车道线 / 地图语义",
    kind: "状态或条件",
    why: "静态结构。应进状态，不进动作空间。",
  },
] as const;

export default function DrivingCondLab({
  onComplete,
}: {
  onComplete?: (state?: Record<string, unknown>) => void;
}) {
  const [id, setId] = useState<(typeof factors)[number]["id"]>("steer");
  const current = factors.find((item) => item.id === id) ?? factors[0];

  return (
    <LabShell
      brief="GAIA-2 一类模型条件很多。先问：这是动作、外生，还是风格？"
      verdict={`${current.title} 应标成「${current.kind}」。${current.why}`}
      tone={id === "steer" ? "ok" : "warn"}
    >
      <div className="wm-lab-toolbar">
        {factors.map((item) => (
          <button
            key={item.id}
            type="button"
            aria-pressed={id === item.id}
            onClick={() => setId(item.id)}
          >
            {item.title}
          </button>
        ))}
        <button type="button" onClick={() => onComplete?.({ factor: id })}>
          记下分类
        </button>
      </div>
    </LabShell>
  );
}
