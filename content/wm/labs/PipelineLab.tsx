"use client";

import { useMemo, useState } from "react";
import { LabShell } from "@/components/labs/LabShell";

const gates = [
  { id: "video", label: "视频预训练完成" },
  { id: "action", label: "加了动作通道" },
  { id: "swap", label: "动作对换分岔" },
  { id: "plan", label: "能在想象里搜动作" },
  { id: "safe", label: "安全层能截断扫杯" },
] as const;

export default function PipelineLab({
  onComplete,
}: {
  onComplete?: (state?: Record<string, unknown>) => void;
}) {
  const [on, setOn] = useState<Record<string, boolean>>({
    video: true,
    action: false,
    swap: false,
    plan: false,
    safe: false,
  });
  const verdict = useMemo(() => {
    if (!on.action) return { text: "还是生成器。文本条件不算动作。", tone: "bad" as const };
    if (!on.swap) return { text: "有动作端口，但还没证明它在用。不要进规划。", tone: "warn" as const };
    if (!on.plan) return { text: "动力学能听动作了。还不能选动作。", tone: "warn" as const };
    if (!on.safe) return { text: "能规划，但不能上真机。扫杯没有闸。", tone: "warn" as const };
    return { text: "这才是工业世界-动作模型接到身体上的最小流水线。", tone: "ok" as const };
  }, [on]);

  return (
    <LabShell
      brief="后训练检查表。缺哪一步，就停在哪一步，不要把画质分数当成动力学。"
      verdict={verdict.text}
      tone={verdict.tone}
    >
      <div className="wm-lab-toolbar">
        {gates.map((item) => (
          <button
            key={item.id}
            type="button"
            aria-pressed={on[item.id]}
            onClick={() => setOn((prev) => ({ ...prev, [item.id]: !prev[item.id] }))}
          >
            {item.label}
          </button>
        ))}
        <button type="button" onClick={() => onComplete?.({ gates: on })}>
          记下卡在哪一步
        </button>
      </div>
    </LabShell>
  );
}
