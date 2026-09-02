"use client";

import { useMemo, useState } from "react";
import { LabShell } from "@/components/labs/LabShell";
import {
  EMBODIMENT_QUESTIONS,
  EXAMPLE_SYSTEMS,
  inferRung,
  type Answer,
} from "@/lib/embodiment-score";

const labels: Record<Answer, string> = {
  yes: "是",
  no: "否",
  part: "部分",
};

export default function EmbodimentLab({
  onComplete,
}: {
  onComplete?: (state?: Record<string, unknown>) => void;
}) {
  const [systemId, setSystemId] = useState<(typeof EXAMPLE_SYSTEMS)[number]["id"]>(
    "wm",
  );
  const [answers, setAnswers] = useState<Answer[]>(
    Array(EMBODIMENT_QUESTIONS.length).fill("part"),
  );
  const [envIsModelOnly, setEnvIsModelOnly] = useState(false);
  const [revealed, setRevealed] = useState(false);
  const system =
    EXAMPLE_SYSTEMS.find((item) => item.id === systemId) ?? EXAMPLE_SYSTEMS[1];
  const guessed = useMemo(
    () => inferRung({ answers, envIsModelOnly }),
    [answers, envIsModelOnly],
  );
  const example = inferRung({
    answers: system.answers,
    envIsModelOnly: system.envIsModelOnly,
  });

  return (
    <LabShell
      brief="七条是否题加一条备注：动作是否只在模型内部步进。打分函数与第 33 课正文同一份代码，避免尺子两套。"
      verdict={
        revealed
          ? `你的打分 ${guessed.rung}，课内示例 ${example.rung}。${system.note} ${guessed.reasons.join(" ")}`
          : "答完再揭晓。Q3 为否时，请勾选动作是否只在模型里 rollout。"
      }
      tone={revealed && guessed.rung !== example.rung ? "warn" : "ok"}
    >
      <div className="wm-lab-toolbar">
        {EXAMPLE_SYSTEMS.map((item) => (
          <button
            key={item.id}
            type="button"
            aria-pressed={systemId === item.id}
            onClick={() => {
              setSystemId(item.id);
              setRevealed(false);
            }}
          >
            {item.title}
          </button>
        ))}
      </div>
      <ul className="wm-lab-list">
        {EMBODIMENT_QUESTIONS.map((question, index) => (
          <li key={question}>
            <strong>
              Q{index + 1} {question}
            </strong>
            <div className="wm-lab-toolbar">
              {(["yes", "no", "part"] as Answer[]).map((value) => (
                <button
                  key={value}
                  type="button"
                  aria-pressed={answers[index] === value}
                  onClick={() => {
                    const next = [...answers];
                    next[index] = value;
                    setAnswers(next);
                    setRevealed(false);
                  }}
                >
                  {labels[value]}
                </button>
              ))}
            </div>
          </li>
        ))}
      </ul>
      <div className="wm-lab-toolbar">
        <button
          type="button"
          className={envIsModelOnly ? "is-on" : undefined}
          aria-pressed={envIsModelOnly}
          onClick={() => {
            setEnvIsModelOnly((value) => !value);
            setRevealed(false);
          }}
        >
          {envIsModelOnly
            ? "动作只在模型内部步进：开"
            : "动作只在模型内部步进：关"}
        </button>
        <button type="button" onClick={() => setRevealed(true)}>
          揭晓课内示例档
        </button>
        <button
          type="button"
          onClick={() =>
            onComplete?.({
              systemId,
              answers,
              envIsModelOnly,
              guessed: guessed.rung,
              example: example.rung,
            })
          }
        >
          记下打分
        </button>
      </div>
    </LabShell>
  );
}
