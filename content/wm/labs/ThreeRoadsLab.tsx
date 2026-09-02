"use client";

import { useState } from "react";
import { LabShell } from "@/components/labs/LabShell";

const roads = [
  {
    id: "pixels",
    title: "重建像素",
    canAnswer: "能画出被撞后的水花，但杯子、手、桌子糊在一张图里。",
    fall: "看不清是谁倒了。",
  },
  {
    id: "repr",
    title: "预测表征",
    canAnswer: "表征里可能有“接触发生了”，但没有单独的杯子变量。",
    fall: "探针也许读得出倾倒，规划时很难只抓住杯子。",
  },
  {
    id: "slots",
    title: "物体中心槽",
    canAnswer: "槽 A 是杯子，槽 B 是手，桌子在槽 C。手的速度进杯子槽的转移。",
    fall: "可以直接问杯子槽的下一姿态会不会过桌沿。",
  },
] as const;

export default function ThreeRoadsLab({
  onComplete,
}: {
  onComplete?: (state?: Record<string, unknown>) => void;
}) {
  const [road, setRoad] = useState<(typeof roads)[number]["id"]>("slots");
  const current = roads.find((item) => item.id === road) ?? roads[2];

  return (
    <LabShell
      brief="同一件事：手碰到一杯水。三种状态分别能回答什么。第 23 课才真正跑 C-SWM。"
      verdict={`${current.title}：${current.fall}`}
    >
      <div className="wm-lab-toolbar">
        {roads.map((item) => (
          <button
            key={item.id}
            type="button"
            aria-pressed={road === item.id}
            onClick={() => setRoad(item.id)}
          >
            {item.title}
          </button>
        ))}
        <button type="button" onClick={() => onComplete?.({ road })}>
          记下选型
        </button>
      </div>
      <ul className="wm-lab-list">
        <li>
          <strong>状态里有什么</strong>
          {current.canAnswer}
        </li>
        <li>
          <strong>杯子会不会倒</strong>
          {current.fall}
        </li>
      </ul>
    </LabShell>
  );
}
