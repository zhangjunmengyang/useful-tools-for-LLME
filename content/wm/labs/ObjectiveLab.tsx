"use client";

import { useState } from "react";
import { LabShell } from "@/components/labs/LabShell";

const objectives = [
  {
    id: "recon",
    title: "像素重建 / ELBO",
    worldscore: "可能变好",
    physics: "不一定",
    swap: "只有动作进了模型才有机会",
    pet: "画得像，杯子仍可能糊进背景",
  },
  {
    id: "flow",
    title: "扩散 / flow matching",
    worldscore: "生成真常在这根尺子上刷",
    physics: "不保证",
    swap: "动作通道要另接",
    pet: "24GB 桌宠通常搬不走整段 DiT",
  },
  {
    id: "jepa",
    title: "JEPA",
    worldscore: "不动，它不生成世界",
    physics: "表征探针可能动",
    swap: "要有动作条件预测器",
    pet: "适合当编码器，不适合当唯一动力学",
  },
  {
    id: "value",
    title: "价值等价",
    worldscore: "不动",
    physics: "不动",
    swap: "间接：对决策有用的才留下",
    pet: "桌宠的克制更靠近这根，但状态不可视",
  },
  {
    id: "sf",
    title: "self-forcing",
    worldscore: "长视频观感可能更好",
    physics: "不保证",
    swap: "不自动给动作",
    pet: "第 30 课小模型可以混入自己的预测，这是搬得动的一块",
  },
] as const;

export default function ObjectiveLab({
  onComplete,
}: {
  onComplete?: (state?: Record<string, unknown>) => void;
}) {
  const [id, setId] = useState<(typeof objectives)[number]["id"]>("sf");
  const current = objectives.find((item) => item.id === id) ?? objectives[4];

  return (
    <LabShell
      brief="选一个目标函数。看 WorldScore、Physics-IQ、动作对换、桌宠克制哪根可能动。"
      verdict={`${current.title} 对桌宠：${current.pet}`}
    >
      <div className="wm-lab-toolbar">
        {objectives.map((item) => (
          <button
            key={item.id}
            type="button"
            aria-pressed={id === item.id}
            onClick={() => setId(item.id)}
          >
            {item.title}
          </button>
        ))}
        <button type="button" onClick={() => onComplete?.({ objective: id })}>
          记下带走的一块
        </button>
      </div>
      <ul className="wm-lab-list">
        <li>
          <strong>WorldScore</strong>
          {current.worldscore}
        </li>
        <li>
          <strong>Physics-IQ</strong>
          {current.physics}
        </li>
        <li>
          <strong>动作对换</strong>
          {current.swap}
        </li>
      </ul>
    </LabShell>
  );
}
