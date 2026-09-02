"use client";

import { useMemo, useState } from "react";
import { LabShell } from "@/components/labs/LabShell";

const checks = [
  { id: "fps", label: "能边看边播（流式）" },
  { id: "delay", label: "动作延迟可接受" },
  { id: "lookback", label: "转身再回头还认得" },
  { id: "swap", label: "动作对换分岔" },
] as const;

export default function PlayableLab({
  onComplete,
}: {
  onComplete?: (state?: Record<string, unknown>) => void;
}) {
  const [on, setOn] = useState<Record<string, boolean>>({
    fps: true,
    delay: true,
    lookback: false,
    swap: false,
  });
  const verdict = useMemo(() => {
    if (!on.fps) return { text: "还是离线视频。", tone: "bad" as const };
    if (!on.delay) return { text: "可播，但玩起来像幻灯片。", tone: "warn" as const };
    if (!on.lookback) return { text: "实时交互有了，世界还是现编的。", tone: "warn" as const };
    if (!on.swap) return { text: "看起来可玩，仍可能是动作盲。", tone: "warn" as const };
    return { text: "这四项齐了，才配叫实时世界模型。Genie 3 的 24 fps 若你没测过，只能写宣称。", tone: "ok" as const };
  }, [on]);

  return (
    <LabShell
      brief="可玩性不是观感。四项里缺哪项，就不要把网页演示写成实时世界模型。"
      verdict={verdict.text}
      tone={verdict.tone}
    >
      <div className="wm-lab-toolbar">
        {checks.map((item) => (
          <button
            key={item.id}
            type="button"
            aria-pressed={on[item.id]}
            onClick={() => setOn((prev) => ({ ...prev, [item.id]: !prev[item.id] }))}
          >
            {item.label}
          </button>
        ))}
        <button type="button" onClick={() => onComplete?.({ checks: on })}>
          记下缺哪项
        </button>
      </div>
    </LabShell>
  );
}
