"use client";

import { useMemo, useState } from "react";
import { LabShell } from "@/components/labs/LabShell";

const inputs = ["语言", "图像", "视频", "声音", "动作"] as const;
const outputs = ["语言", "图像", "视频", "声音", "动作"] as const;

function classify(ins: Set<string>, outs: Set<string>) {
  const hasActionIn = ins.has("动作");
  const hasActionOut = outs.has("动作");
  const hasVideoOut = outs.has("视频") || outs.has("图像");
  const langOnly = [...ins].every((item) => item === "语言") && [...outs].every((item) => item === "语言");

  if (langOnly) {
    return { label: "语言模型", ask: "三问全没回答。", tone: "bad" as const };
  }
  if (hasActionOut && !hasVideoOut) {
    return {
      label: "策略头 / VLA",
      ask: "它在学 π(a|o)。杯子会不会倒，要另接前向模型。",
      tone: "warn" as const,
    };
  }
  if (hasActionIn && hasVideoOut) {
    return {
      label: "世界模拟器 / 世界-动作模型",
      ask: "还要做动作对换。分不分岔，决定能不能规划。",
      tone: "ok" as const,
    };
  }
  if (hasVideoOut) {
    return {
      label: "视频或音画生成器",
      ask: "第 12 课第一问：动作起作用吗？这套配置里动作没进模型。",
      tone: "warn" as const,
    };
  }
  if (outs.has("语言") && (ins.has("图像") || ins.has("视频"))) {
    return {
      label: "VLM / Reasoner",
      ask: "理解不等于模拟。没有未来状态，就没有克制。",
      tone: "warn" as const,
    };
  }
  return { label: "未命名配置", ask: "先写清输入输出，再贴世界模型标签。", tone: "warn" as const };
}

export default function OmnimodalLab({
  onComplete,
}: {
  onComplete?: (state?: Record<string, unknown>) => void;
}) {
  const [ins, setIns] = useState<string[]>(["语言", "图像"]);
  const [outs, setOuts] = useState<string[]>(["视频"]);
  const verdict = useMemo(() => classify(new Set(ins), new Set(outs)), [ins, outs]);

  function toggle(list: string[], value: string, setter: (next: string[]) => void) {
    setter(list.includes(value) ? list.filter((item) => item !== value) : [...list, value]);
  }

  return (
    <LabShell
      brief="勾选 Cosmos 3 一类模型的输入和输出。配置决定它是哪种产品，不是名称。"
      verdict={`${verdict.label}：${verdict.ask}`}
      tone={verdict.tone}
    >
      <p className="wm-lab-brief">输入</p>
      <div className="wm-lab-toolbar">
        {inputs.map((item) => (
          <button
            key={`in-${item}`}
            type="button"
            aria-pressed={ins.includes(item)}
            onClick={() => toggle(ins, item, setIns)}
          >
            {item}
          </button>
        ))}
      </div>
      <p className="wm-lab-brief">输出</p>
      <div className="wm-lab-toolbar">
        {outputs.map((item) => (
          <button
            key={`out-${item}`}
            type="button"
            aria-pressed={outs.includes(item)}
            onClick={() => toggle(outs, item, setOuts)}
          >
            {item}
          </button>
        ))}
        <button
          type="button"
          onClick={() => onComplete?.({ ins, outs, label: verdict.label })}
        >
          记下配置
        </button>
      </div>
    </LabShell>
  );
}
