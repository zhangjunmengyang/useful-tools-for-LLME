"use client";

import { useState } from "react";
import { LabShell } from "@/components/labs/LabShell";

const tracks = [
  {
    id: "cup",
    title: "杯子碰到桌沿",
    energy: "短促撞击，高频一下。",
    fork: "听得懂的模型应提高「杯子过沿」的概率。",
  },
  {
    id: "keys",
    title: "键盘敲击",
    energy: "一串短脉冲，位置在画面外。",
    fork: "杯子不该因此移动。若预测跟着倒，模型在用声音当配乐。",
  },
  {
    id: "score",
    title: "BGM",
    energy: "持续宽带，和画面事件对不齐。",
    fork: "动力学不应分岔。分岔了说明声音通道被当成风格。",
  },
] as const;

const models = [
  {
    id: "gen",
    title: "音画生成器",
    usesSound: "把声音和画面编成一条好看的视频。",
    swap: "通常没有动作端口，也没有「同一画面换声轨」的对换协议。",
  },
  {
    id: "avwm",
    title: "AVWM 式世界模型",
    usesSound: "声音是 o_t 的一部分，和画面一起进 POMDP。",
    swap: "同一画面换声轨，下一刻视听预测必须允许分岔。",
  },
  {
    id: "pet",
    title: "桌宠麦克风",
    usesSound: "麦克风是观察，喇叭是可选动作。",
    swap: "听杯碰桌可以决定不出手；配乐不能当状态。",
  },
] as const;

export default function AudioWorldLab({
  onComplete,
}: {
  onComplete?: (state?: Record<string, unknown>) => void;
}) {
  const [trackId, setTrackId] = useState<(typeof tracks)[number]["id"]>("cup");
  const [modelId, setModelId] = useState<(typeof models)[number]["id"]>("gen");
  const track = tracks.find((item) => item.id === trackId) ?? tracks[0];
  const model = models.find((item) => item.id === modelId) ?? models[0];

  return (
    <LabShell
      brief="画面钉死：一只杯子停在桌沿。只换声轨。看三类系统会不会改下一秒的预测。"
      verdict={`${model.title} 听到「${track.title}」：${track.fork}`}
      tone={modelId === "gen" ? "warn" : "ok"}
    >
      <div className="wm-lab-toolbar">
        {tracks.map((item) => (
          <button
            key={item.id}
            type="button"
            aria-pressed={trackId === item.id}
            onClick={() => setTrackId(item.id)}
          >
            {item.title}
          </button>
        ))}
      </div>
      <div className="wm-lab-toolbar">
        {models.map((item) => (
          <button
            key={item.id}
            type="button"
            aria-pressed={modelId === item.id}
            onClick={() => setModelId(item.id)}
          >
            {item.title}
          </button>
        ))}
        <button
          type="button"
          onClick={() => onComplete?.({ trackId, modelId })}
        >
          记下对换
        </button>
      </div>
      <ul className="wm-lab-list">
        <li>
          <strong>声轨</strong>
          {track.energy}
        </li>
        <li>
          <strong>系统怎么用声音</strong>
          {model.usesSound} {model.swap}
        </li>
      </ul>
    </LabShell>
  );
}
