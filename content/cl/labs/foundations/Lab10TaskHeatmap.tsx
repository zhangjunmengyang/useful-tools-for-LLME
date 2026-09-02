"use client";

import { useMemo, useState } from "react";
import type { CSSProperties } from "react";
import styles from "./Lab10TaskHeatmap.module.css";
import type { FoundationLabProps } from "./types";
import { formatPct, initialNumber } from "./types";

const TASKS = ["代码", "数学", "摘要", "问答"] as const;
type TaskKey = (typeof TASKS)[number];

const SIM = [
  [1, 0.72, 0.22, 0.3],
  [0.72, 1, 0.18, 0.36],
  [0.22, 0.18, 1, 0.58],
  [0.3, 0.36, 0.58, 1],
];
const PEAK = [0.91, 0.88, 0.86, 0.9];

function buildMatrix(interfere: number) {
  const matrix = Array.from({ length: 4 }, () => Array(4).fill(0));
  for (let t = 0; t < 4; t += 1) {
    matrix[t][t] = PEAK[t];
    for (let j = 0; j < t; j += 1) {
      matrix[t][j] = matrix[t - 1][j] * Math.exp(-interfere * SIM[t][j]);
    }
  }
  return matrix;
}

export function Lab10TaskHeatmap({
  onComplete,
  initialState,
}: FoundationLabProps) {
  const [interfere, setInterfere] = useState(
    initialNumber(initialState, "interfere", 1),
  );
  const [weakPred, setWeakPred] = useState<TaskKey | null>(null);
  const [whyPred, setWhyPred] = useState<"early" | "last" | null>(null);
  const [hasRun, setHasRun] = useState(false);

  const heat = useMemo(() => {
    const matrix = buildMatrix(interfere);
    const last = matrix[3];
    let worst = 0;
    for (let j = 1; j < 3; j += 1) {
      if (last[j] < last[worst]) worst = j;
    }
    if (last[3] < last[worst]) worst = 3;
    return { matrix, worst: TASKS[worst] };
  }, [interfere]);

  const gatePassed = hasRun && weakPred === heat.worst && whyPred === "early";

  function invalidate() {
    setHasRun(false);
  }

  function runLab() {
    setHasRun(true);
    if (weakPred === heat.worst && whyPred === "early") {
      onComplete?.({
        interfere,
        worst: heat.worst,
        lastRow: heat.matrix[3],
      });
    }
  }

  function reset() {
    setInterfere(1);
    setWeakPred(null);
    setWhyPred(null);
    setHasRun(false);
  }

  return (
    <section className={styles.lab} aria-labelledby="lab10-title">
      <header className={styles.header}>
        <div>
          <div className={styles.labels}>
            <span>顺序指令</span>
            <span>任务热力图</span>
          </div>
          <h3 id="lab10-title">任务热力图：颜色越浅越忘</h3>
          <p>
            顺序是代码 → 数学 → 摘要 → 问答。后任务按相似度冲前面的格子：R[t][j] = R[t−1][j] · exp(−λ sim[t,j])。
          </p>
        </div>
        <button className={styles.reset} type="button" onClick={reset}>
          重置实验
        </button>
      </header>

      <div className={styles.workbench}>
        <div className={styles.controls}>
          <label>
            <span>
              冲刷强度 λ <strong>{interfere.toFixed(2)}</strong>
            </span>
            <input
              type="range"
              min="0.35"
              max="1.6"
              step="0.05"
              value={interfere}
              onChange={(event) => {
                setInterfere(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <table className={styles.sim}>
            <thead>
              <tr>
                <th>sim</th>
                {TASKS.map((name) => (
                  <th key={name}>{name}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {TASKS.map((rowName, i) => (
                <tr key={rowName}>
                  <th>{rowName}</th>
                  {SIM[i].map((value, j) => (
                    <td key={j}>{value.toFixed(2)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className={styles.stage} aria-live="polite">
          <table className={styles.heat}>
            <thead>
              <tr>
                <th>学完 \\ 测</th>
                {TASKS.map((name) => (
                  <th key={name}>{name}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {heat.matrix.map((row, i) => (
                <tr key={i}>
                  <th>{TASKS[i]}</th>
                  {row.map((value, j) => (
                    <td
                      key={j}
                      style={
                        {
                          "--cell": hasRun ? `${Math.round(value * 90)}%` : "0%",
                        } as CSSProperties
                      }
                    >
                      {hasRun ? (j > i ? "—" : formatPct(value)) : "·"}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className={styles.prediction}>
        <fieldset>
          <legend>预测 1：最后一行最浅（忘得最狠）的是哪一类？</legend>
          <div className={styles.choiceRow}>
            {TASKS.map((name) => (
              <button
                type="button"
                key={name}
                aria-pressed={weakPred === name}
                onClick={() => {
                  setWeakPred(name);
                  invalidate();
                }}
              >
                {name}
              </button>
            ))}
          </div>
        </fieldset>
        <fieldset>
          <legend>预测 2：主要因为什么？</legend>
          <div className={styles.choiceRow}>
            <button
              type="button"
              aria-pressed={whyPred === "early"}
              onClick={() => {
                setWhyPred("early");
                invalidate();
              }}
            >
              更早、又和后任务更像
            </button>
            <button
              type="button"
              aria-pressed={whyPred === "last"}
              onClick={() => {
                setWhyPred("last");
                invalidate();
              }}
            >
              刚学完的最后一项
            </button>
          </div>
        </fieldset>
        <button
          type="button"
          className={styles.run}
          disabled={!weakPred || !whyPred}
          onClick={runLab}
        >
          播放热力图
        </button>
      </div>

      <div
        className={`${styles.gate} ${
          hasRun ? (gatePassed ? styles.pass : styles.retry) : ""
        }`}
        role="status"
      >
        <strong>{gatePassed ? "验收已通过" : "完成验收"}</strong>
        <span>
          {!hasRun
            ? "先看相似度表再猜最浅的一列。"
            : gatePassed
              ? `最后一行最浅的是${heat.worst}。它最早上场，又和数学高度重叠。`
              : "最后一行的最小值通常在最早、且与后续任务相似度高的那一列。"}
        </span>
      </div>
    </section>
  );
}
