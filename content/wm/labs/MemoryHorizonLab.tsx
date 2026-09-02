"use client";

import { useState } from "react";
import { LabShell } from "@/components/labs/LabShell";

const modes = [
  {
    id: "window",
    title: "滑窗",
    store: "只看最近 K 帧。K 之外的杯子被丢掉。",
    return: "转一圈回来，房子可以是新编的。第 12 课量过。",
    pet: "头转回去，杯子可能从状态里消失。",
  },
  {
    id: "hidden",
    title: "压缩隐状态",
    store: "RSSM 一类把历史压进向量。理论上无限长。",
    return: "实际会漂。杯子融进噪声。",
    pet: "记得「有过杯子」，位置可能已经错了。",
  },
  {
    id: "bank",
    title: "记忆库",
    store: "WorldMem 把帧、位姿、时间戳存成可查询单元。",
    return: "按当前位姿取回旧帧，而不是靠窗口碰巧还罩着。",
    pet: "转头后仍可能指向杯子，前提是查询键对。",
  },
] as const;

export default function MemoryHorizonLab({
  onComplete,
}: {
  onComplete?: (state?: Record<string, unknown>) => void;
}) {
  const [id, setId] = useState<(typeof modes)[number]["id"]>("window");
  const current = modes.find((item) => item.id === id) ?? modes[0];

  return (
    <LabShell
      brief="三种忘法。问的是：走开再回头，杯子还在不在。"
      verdict={`${current.title}：${current.pet}`}
      tone={id === "bank" ? "ok" : "warn"}
    >
      <div className="wm-lab-toolbar">
        {modes.map((item) => (
          <button
            key={item.id}
            type="button"
            aria-pressed={id === item.id}
            onClick={() => setId(item.id)}
          >
            {item.title}
          </button>
        ))}
        <button type="button" onClick={() => onComplete?.({ memory: id })}>
          记下忘法
        </button>
      </div>
      <ul className="wm-lab-list">
        <li>
          <strong>存什么</strong>
          {current.store}
        </li>
        <li>
          <strong>回头时</strong>
          {current.return}
        </li>
      </ul>
    </LabShell>
  );
}
