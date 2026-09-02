"use client";

import { useMemo, useState } from "react";
import { LabShell } from "@/components/labs/LabShell";

type Cue = "wave" | "phone" | "still";

export default function GazeLab({
  onComplete,
}: {
  onComplete?: (state?: Record<string, unknown>) => void;
}) {
  const [cue, setCue] = useState<Cue>("wave");
  const [memory, setMemory] = useState(true);

  const inertia = 0.42;
  const model = useMemo(() => {
    if (cue === "wave") return memory ? 0.78 : 0.61;
    if (cue === "phone") return memory ? 0.18 : 0.33;
    return memory ? 0.4 : 0.42;
  }, [cue, memory]);

  const verdict =
    cue === "wave"
      ? memory
        ? "挥手后，带短时记忆的头把“人会看过来”抬到 0.78。惯性基线仍停在刚才的 0.42。"
        : "关掉记忆后，模型几乎只看当前帧，挥手的增益变薄。"
      : cue === "phone"
        ? "人低头看手机，预测应下降。若仍对视，桌宠会变成扰人的摆件。"
        : "没有新证据时，好的预测应靠近惯性，而不是胡乱摆动。";

  return (
    <LabShell
      brief="人不是可控动作，但必须被预测。对照惯性基线：永远猜当前值。"
      verdict={verdict}
    >
      <div className="wm-lab-toolbar">
        <button type="button" aria-pressed={cue === "wave"} onClick={() => setCue("wave")}>
          对着镜头挥手
        </button>
        <button type="button" aria-pressed={cue === "phone"} onClick={() => setCue("phone")}>
          低头看手机
        </button>
        <button type="button" aria-pressed={cue === "still"} onClick={() => setCue("still")}>
          没有新动作
        </button>
        <button
          type="button"
          className={memory ? "is-on" : undefined}
          aria-pressed={memory}
          onClick={() => setMemory((value) => !value)}
        >
          {memory ? "短时记忆：开" : "短时记忆：关"}
        </button>
        <button type="button" onClick={() => onComplete?.({ cue, memory, model, inertia })}>
          记下预测
        </button>
      </div>
      <div className="wm-lab-meter">
        <span>模型：下一秒看镜头 {model.toFixed(2)}</span>
        <div className="wm-lab-bar" aria-hidden="true">
          <i style={{ width: `${model * 100}%` }} />
        </div>
        <span>惯性基线 {inertia.toFixed(2)}</span>
        <div className="wm-lab-bar" aria-hidden="true">
          <i style={{ width: `${inertia * 100}%`, background: "var(--color-muted)" }} />
        </div>
      </div>
    </LabShell>
  );
}
