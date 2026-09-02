"use client";

import { useState } from "react";
import { LabShell } from "@/components/labs/LabShell";

type Move = "cover" | "remove" | "turn";

const moveLabels: Record<Move, string> = {
  cover: "挡住 2 秒",
  remove: "端走杯子",
  turn: "转头再转回",
};

export default function PermanenceLab({
  onComplete,
}: {
  onComplete?: (state?: Record<string, unknown>) => void;
}) {
  const [move, setMove] = useState<Move>("cover");
  const [windowed, setWindowed] = useState(false);

  const persistentOn =
    move === "cover" || move === "turn" ? true : false;
  const windowOn = move === "remove" ? false : move === "cover" ? false : true;
  const lampOn = windowed ? windowOn : persistentOn;

  const verdict = windowed
    ? move === "cover"
      ? "滑窗模型在杯子被挡住后把杯子忘掉了。灯该红。持久状态不该跟着灭。"
      : move === "remove"
        ? "杯子真的被端走了，两种模型都该更新：灯红是对的。"
        : "转头再转回，滑窗还可能把杯子重新发明。持久状态应还记得原来的杯子。"
    : move === "remove"
      ? "端走之后灯应灭。恒常不是“永远在”，是“没证据离开就还在”。"
      : "挡住或转头，杯子应还在状态里。这是感知状态更新，还不是按动作预测下一步。";

  return (
    <LabShell
      brief="物体恒常灯：挡住、端走、转头三种操作。绿灯表示状态里还认为杯子在桌上。"
      verdict={verdict}
      tone={!lampOn && move === "cover" ? "warn" : "ok"}
    >
      <div className="wm-lab-toolbar">
        {(Object.keys(moveLabels) as Move[]).map((item) => (
          <button
            key={item}
            type="button"
            aria-pressed={move === item}
            onClick={() => setMove(item)}
          >
            {moveLabels[item]}
          </button>
        ))}
        <button
          type="button"
          className={windowed ? "is-on" : undefined}
          aria-pressed={windowed}
          onClick={() => setWindowed((value) => !value)}
        >
          {windowed ? "模型：只看最近几帧" : "模型：持久 3D 状态"}
        </button>
        <button
          type="button"
          onClick={() => onComplete?.({ move, windowed, lampOn })}
        >
          记下这次灯色
        </button>
      </div>
      <div className="wm-lab-panel">
        <h3>灯 {lampOn ? "绿：杯子还在" : "红：杯子不在"}</h3>
        <svg className="wm-lab-scene" viewBox="0 0 320 160" role="img">
          <rect x="24" y="112" width="272" height="16" rx="3" fill="var(--color-paper-3)" />
          {move !== "remove" ? (
            <circle
              cx={move === "turn" ? 86 : 160}
              cy="92"
              r="16"
              fill="var(--color-accent)"
              opacity={move === "cover" ? 0.25 : 1}
            />
          ) : null}
          {move === "cover" ? (
            <rect x="132" y="68" width="56" height="40" rx="4" fill="var(--color-ink)" opacity="0.55" />
          ) : null}
          <circle
            cx="286"
            cy="36"
            r="16"
            fill={lampOn ? "var(--color-success)" : "var(--color-danger)"}
          />
        </svg>
      </div>
    </LabShell>
  );
}
