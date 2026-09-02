"use client";

import { useMemo, useState } from "react";
import { LabShell } from "@/components/labs/LabShell";

type Mode = "normal" | "swap" | "drop";

export default function SlotLab({
  onComplete,
}: {
  onComplete?: (state?: Record<string, unknown>) => void;
}) {
  const [mode, setMode] = useState<Mode>("normal");

  const balls = useMemo(() => {
    if (mode === "swap") {
      return [
        { id: "A", x: 180, color: "var(--color-warning)" },
        { id: "B", x: 80, color: "var(--color-accent)" },
      ];
    }
    if (mode === "drop") {
      return [{ id: "A", x: 80, color: "var(--color-accent)" }];
    }
    return [
      { id: "A", x: 80, color: "var(--color-accent)" },
      { id: "B", x: 180, color: "var(--color-warning)" },
    ];
  }, [mode]);

  const verdict =
    mode === "swap"
      ? "对调槽之后，若碰撞预测跟着身份走，说明槽绑定了物体，不是绑死了像素位置。"
      : mode === "drop"
        ? "删掉槽 B，只剩一个球。向量模型这时仍可能“平均”出两个球的幽灵。"
        : "正常情况：槽 A 追左球，槽 B 追右球，下一步各自有速度。";

  return (
    <LabShell brief="两个球、两个槽。对调或删除一个槽，看预测还知不知道谁撞谁。" verdict={verdict}>
      <div className="wm-lab-toolbar">
        <button type="button" aria-pressed={mode === "normal"} onClick={() => setMode("normal")}>
          正常
        </button>
        <button type="button" aria-pressed={mode === "swap"} onClick={() => setMode("swap")}>
          对调槽
        </button>
        <button type="button" aria-pressed={mode === "drop"} onClick={() => setMode("drop")}>
          删除槽 B
        </button>
        <button type="button" onClick={() => onComplete?.({ mode })}>
          记下槽实验
        </button>
      </div>
      <div className="wm-lab-panel">
        <h3>槽可视化</h3>
        <svg className="wm-lab-scene" viewBox="0 0 280 160" role="img">
          {balls.map((ball) => (
            <g key={ball.id}>
              <circle cx={ball.x} cy="88" r="18" fill={ball.color} />
              <text x={ball.x} y="94" textAnchor="middle" fontSize="12" fill="white">
                {ball.id}
              </text>
            </g>
          ))}
          {mode !== "drop" ? (
            <line x1="98" y1="88" x2="162" y2="88" stroke="var(--color-ink)" strokeWidth="2" />
          ) : null}
        </svg>
      </div>
    </LabShell>
  );
}
