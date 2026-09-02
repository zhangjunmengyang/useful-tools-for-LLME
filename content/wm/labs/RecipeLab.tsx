"use client";

import { useState } from "react";
import { LabShell } from "@/components/labs/LabShell";

const recipes = [
  {
    id: "tf",
    title: "teacher forcing",
    train: "每一步喂真历史。训练损失好看。",
    infer: "推理必须吃自己的输出。误差按步滚。",
    interactive: "不适合边看边播的长交互。",
  },
  {
    id: "bidiff",
    title: "双向扩散 / 整段 flow",
    train: "一次看整段视频去噪或沿流。画质高。",
    infer: "要等整段算完。不能在第 8 帧改动作。",
    interactive: "生成器，不是实时世界引擎。",
  },
  {
    id: "causvid",
    title: "CausVid 式蒸馏",
    train: "把双向教师蒸成少步因果学生。",
    infer: "可以按块往前走，但仍可能和推理分布对不齐。",
    interactive: "朝可玩走近了一步。",
  },
  {
    id: "sf",
    title: "self-forcing",
    train: "训练时用 KV cache 吃自己刚生成的帧。",
    infer: "训练分布更接近推理分布。长视频仍会漂，所以才有 ++ / rolling。",
    interactive: "这是目前开源流式世界模型常用的补丁。",
  },
] as const;

export default function RecipeLab({
  onComplete,
}: {
  onComplete?: (state?: Record<string, unknown>) => void;
}) {
  const [id, setId] = useState<(typeof recipes)[number]["id"]>("tf");
  const current = recipes.find((item) => item.id === id) ?? recipes[0];

  return (
    <LabShell
      brief="同一副骨架，换训练协议。桌宠在乎的是推理时自己接自己会不会崩。"
      verdict={`${current.title}：${current.interactive}`}
      tone={id === "sf" || id === "causvid" ? "ok" : "warn"}
    >
      <div className="wm-lab-toolbar">
        {recipes.map((item) => (
          <button
            key={item.id}
            type="button"
            aria-pressed={id === item.id}
            onClick={() => setId(item.id)}
          >
            {item.title}
          </button>
        ))}
        <button type="button" onClick={() => onComplete?.({ recipe: id })}>
          记下配方
        </button>
      </div>
      <ul className="wm-lab-list">
        <li>
          <strong>训练时看什么</strong>
          {current.train}
        </li>
        <li>
          <strong>推理时发生什么</strong>
          {current.infer}
        </li>
      </ul>
    </LabShell>
  );
}
