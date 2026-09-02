"use client";

import { useMemo, useState } from "react";
import { LabShell } from "@/components/labs/LabShell";

export default function SimGapLab({
  onComplete,
}: {
  onComplete?: (state?: Record<string, unknown>) => void;
}) {
  const [friction, setFriction] = useState(0.4);
  const [delay, setDelay] = useState(0);
  const [filter, setFilter] = useState(true);

  const result = useMemo(() => {
    const travel = 40 + (0.8 - friction) * 140 + delay * 0.9;
    const edge = 190;
    const wouldFall = travel > edge;
    const blocked = filter && travel > edge - 18;
    return { travel: Math.min(travel, 230), wouldFall, blocked };
  }, [delay, filter, friction]);

  const verdict = result.blocked
    ? "安全层截断了动作。杯子停在桌沿 5 cm 内。梦里的高分不能直接下发。"
    : result.wouldFall
      ? "杯子过了桌沿。摩擦更低或延迟更大时，同一条“推过去”会变成扫落。"
      : "杯子还在桌上。把摩擦拧低、延迟拧高，看同一条动作何时失败。";

  return (
    <LabShell
      brief="同一条推杯动作。拧摩擦和指令延迟，看仿真里成功的动作在哪条参数上把杯子扫下去。"
      verdict={verdict}
      tone={result.wouldFall && !result.blocked ? "bad" : "ok"}
    >
      <div className="wm-lab-toolbar">
        <label className="wm-lab-slider">
          摩擦 {friction.toFixed(2)}
          <input
            type="range"
            min="0.1"
            max="0.8"
            step="0.05"
            value={friction}
            onChange={(event) => setFriction(Number(event.target.value))}
          />
        </label>
        <label className="wm-lab-slider">
          延迟 {delay} ms
          <input
            type="range"
            min="0"
            max="120"
            step="10"
            value={delay}
            onChange={(event) => setDelay(Number(event.target.value))}
          />
        </label>
        <button
          type="button"
          className={filter ? "is-on" : undefined}
          aria-pressed={filter}
          onClick={() => setFilter((value) => !value)}
        >
          {filter ? "安全过滤：开" : "安全过滤：关"}
        </button>
        <button
          type="button"
          onClick={() => onComplete?.({ friction, delay, filter, ...result })}
        >
          记下这次推杯
        </button>
      </div>
      <div className="wm-lab-panel">
        <h3>桌子</h3>
        <svg className="wm-lab-scene" viewBox="0 0 280 140" role="img">
          <rect x="20" y="88" width="200" height="14" fill="var(--color-paper-3)" />
          <line x1="208" x2="208" y1="70" y2="110" stroke="var(--color-danger)" strokeDasharray="4 3" />
          <circle cx={20 + result.travel} cy="80" r="12" fill="var(--color-accent)" />
        </svg>
      </div>
    </LabShell>
  );
}
