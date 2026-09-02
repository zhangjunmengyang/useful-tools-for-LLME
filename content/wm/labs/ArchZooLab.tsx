"use client";

import { useState } from "react";
import { LabShell } from "@/components/labs/LabShell";

const arches = [
  {
    id: "rssm",
    title: "RSSM",
    state: "确定通道记历史，随机通道保留多种可能。",
    action: "动作进转移。适合 MPC。",
    budget: "24GB 能训小任务。桌宠用得上。",
  },
  {
    id: "ar",
    title: "AR token / MAGI chunk",
    state: "离散符号或视频块。状态就是上下文窗口。",
    action: "动作要编进 token 流。",
    budget: "大模型贵。IRIS 级可以体验。",
  },
  {
    id: "dit",
    title: "双向 DiT",
    state: "整段潜空间。没有单独的循环状态。",
    action: "动作常被文本替代。",
    budget: "画质好，交互差。",
  },
  {
    id: "mot",
    title: "MoT（Cosmos 3）",
    state: "多模态共享 mRoPE，Reasoner 与 Generator 分通路。",
    action: "动作可以是一条模态。",
    budget: "24GB 通常只读文档。",
  },
  {
    id: "jepa",
    title: "JEPA",
    state: "表征，不生成像素。",
    action: "要另接动作条件预测器。",
    budget: "V-JEPA 2 探针可做。规划看 AC 版。",
  },
  {
    id: "lat",
    title: "latent action",
    state: "视频 tokenizer 加动力学。",
    action: "动作从无标签视频里挖。",
    budget: "tinyworlds 能练。Genie 3 不能练。",
  },
] as const;

export default function ArchZooLab({
  onComplete,
}: {
  onComplete?: (state?: Record<string, unknown>) => void;
}) {
  const [id, setId] = useState<(typeof arches)[number]["id"]>("rssm");
  const current = arches.find((item) => item.id === id) ?? arches[0];

  return (
    <LabShell
      brief="给桌宠挑骨架。先看状态在哪、动作从哪进，再看 24GB 能不能跑。"
      verdict={`${current.title}：${current.budget}`}
    >
      <div className="wm-lab-toolbar">
        {arches.map((item) => (
          <button
            key={item.id}
            type="button"
            aria-pressed={id === item.id}
            onClick={() => setId(item.id)}
          >
            {item.title}
          </button>
        ))}
        <button type="button" onClick={() => onComplete?.({ arch: id })}>
          记下选型
        </button>
      </div>
      <ul className="wm-lab-list">
        <li>
          <strong>状态</strong>
          {current.state}
        </li>
        <li>
          <strong>动作</strong>
          {current.action}
        </li>
      </ul>
    </LabShell>
  );
}
