"use client";

import { useMemo, useState } from "react";
import { LabShell } from "@/components/labs/LabShell";

type Action = "left" | "right" | "reach" | "wait";

const labels: Record<Action, string> = {
  left: "看左",
  right: "看右",
  reach: "伸手",
  wait: "不动",
};

function cupX(action: Action, blind: boolean) {
  if (blind || action === "wait" || action === "left" || action === "right") {
    return 120;
  }
  return 168;
}

function headX(action: Action, blind: boolean) {
  if (blind) return 120;
  if (action === "left") return 72;
  if (action === "right") return 168;
  return 120;
}

function Scene({
  action,
  blind,
  title,
}: {
  action: Action;
  blind: boolean;
  title: string;
}) {
  const cup = cupX(action, blind);
  const head = headX(action, blind);
  const fallen = !blind && action === "reach";
  return (
    <div className="wm-lab-panel">
      <h3>{title}</h3>
      <svg className="wm-lab-scene" viewBox="0 0 240 160" role="img">
        <rect x="16" y="108" width="208" height="18" rx="3" fill="var(--color-paper-3)" />
        <circle cx={cup} cy={fallen ? 126 : 92} r="14" fill="var(--color-accent)" />
        <rect x={head - 10} y="28" width="20" height="20" rx="6" fill="var(--color-ink)" />
        <text x="20" y="22" fontSize="11" fill="var(--color-muted)">
          {labels[action]}
        </text>
      </svg>
    </div>
  );
}

export default function ActionSwapLab({
  onComplete,
}: {
  onComplete?: (state?: Record<string, unknown>) => void;
}) {
  const [left, setLeft] = useState<Action>("left");
  const [right, setRight] = useState<Action>("reach");
  const [blind, setBlind] = useState(false);
  const [guess, setGuess] = useState<"yes" | "no" | null>(null);

  const diverges = useMemo(() => {
    if (blind) return false;
    return cupX(left, false) !== cupX(right, false) || headX(left, false) !== headX(right, false);
  }, [blind, left, right]);

  const verdict =
    guess === null
      ? "先猜这两条未来会不会分岔，再按动作看画面。"
      : guess === "yes" && diverges
        ? "分岔了。同一历史换动作，杯子或头的下一刻不同，模型才算听动作。"
        : guess === "no" && !diverges
          ? "没分岔。动作盲模型不管你按什么键，都会画出同一条未来。"
          : diverges
            ? "实际分岔了。若你猜不会，多半是把“画面还像桌子”当成了“听懂了动作”。"
            : "实际没分岔。打开动作盲开关对照一次：盲模型永远画出同一张图。";

  return (
    <LabShell brief="同一历史只换动作。两条未来如果完全一样，模型就是动作盲，不能拿去规划。桌上的杯子和赛道上的车，试金石是同一把。" verdict={verdict} tone={guess === null ? "ok" : diverges ? "ok" : "warn"}>
      <div className="wm-lab-toolbar">
        <span>左支</span>
        {(Object.keys(labels) as Action[]).map((action) => (
          <button key={`l-${action}`} type="button" aria-pressed={left === action} onClick={() => setLeft(action)}>
            {labels[action]}
          </button>
        ))}
      </div>
      <div className="wm-lab-toolbar">
        <span>右支</span>
        {(Object.keys(labels) as Action[]).map((action) => (
          <button key={`r-${action}`} type="button" aria-pressed={right === action} onClick={() => setRight(action)}>
            {labels[action]}
          </button>
        ))}
      </div>
      <div className="wm-lab-toolbar">
        <button type="button" className={blind ? "is-on" : undefined} aria-pressed={blind} onClick={() => setBlind((value) => !value)}>
          {blind ? "动作盲：开" : "动作盲：关"}
        </button>
        <button type="button" onClick={() => setGuess("yes")}>
          我猜会分岔
        </button>
        <button type="button" onClick={() => setGuess("no")}>
          我猜不会
        </button>
        <button
          type="button"
          onClick={() => onComplete?.({ left, right, blind, diverges, guess })}
        >
          记下这次对照
        </button>
      </div>
      <div className="wm-lab-stage">
        <Scene action={left} blind={blind} title="未来 A" />
        <Scene action={right} blind={blind} title="未来 B" />
      </div>
    </LabShell>
  );
}
