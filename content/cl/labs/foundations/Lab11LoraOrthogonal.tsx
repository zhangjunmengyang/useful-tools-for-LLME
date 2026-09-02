"use client";

import { useMemo, useState } from "react";
import styles from "./Lab11LoraOrthogonal.module.css";
import type { FoundationLabProps } from "./types";
import { formatPct, initialBoolean, initialNumber } from "./types";

type CutKey = "all" | "half" | "none";

function cutBucket(remain: number): CutKey {
  const cut = 1 - remain;
  if (cut >= 0.7) return "all";
  if (cut <= 0.2) return "none";
  return "half";
}

export function Lab11LoraOrthogonal({
  onComplete,
  initialState,
}: FoundationLabProps) {
  const [angle, setAngle] = useState(initialNumber(initialState, "angle", 0));
  const [olora, setOlora] = useState(
    initialBoolean(initialState, "olora", false),
  );
  const [cutPred, setCutPred] = useState<CutKey | null>(null);
  const [orthoPred, setOrthoPred] = useState<"protect" | "hurt" | null>(null);
  const [hasRun, setHasRun] = useState(false);

  const lora = useMemo(() => {
    const rad = (angle * Math.PI) / 180;
    const u1: [number, number] = [1, 0];
    const u2raw: [number, number] = [Math.cos(rad), Math.sin(rad)];
    const dot = u1[0] * u2raw[0] + u1[1] * u2raw[1];
    let u2: [number, number] = u2raw;
    let residual = 1;
    if (olora) {
      const rx = u2raw[0] - dot * u1[0];
      const ry = u2raw[1] - dot * u1[1];
      residual = Math.hypot(rx, ry);
      if (residual < 1e-6) u2 = [0, 1];
      else u2 = [rx / residual, ry / residual];
    }
    const overlap = Math.abs(u1[0] * u2[0] + u1[1] * u2[1]);
    const t1Remain = 1 - overlap * 0.92;
    const t2 = olora ? 0.5 + 0.45 * Math.max(residual, Math.abs(Math.sin(rad))) : 0.92;
    return { u1, u2, overlap, t1Remain, t2, cut: cutBucket(t1Remain) };
  }, [angle, olora]);

  const gatePassed = hasRun && cutPred === lora.cut && orthoPred === "protect";

  function invalidate() {
    setHasRun(false);
  }

  function runLab() {
    setHasRun(true);
    if (cutPred === lora.cut && orthoPred === "protect") {
      onComplete?.({
        angle,
        olora,
        overlap: lora.overlap,
        t1Remain: lora.t1Remain,
        cut: lora.cut,
      });
    }
  }

  function reset() {
    setAngle(0);
    setOlora(false);
    setCutPred(null);
    setOrthoPred(null);
    setHasRun(false);
  }

  const origin = { x: 170, y: 160 };
  const s = 110;

  return (
    <section className={styles.lab} aria-labelledby="lab11-title">
      <header className={styles.header}>
        <div>
          <div className={styles.labels}>
            <span>LoRA</span>
            <span>正交约束</span>
          </div>
          <h3 id="lab11-title">LoRA 正交：夹角决定旧更新被削掉多少</h3>
          <p>
            LoRA 只在一个低维方向上改权重。两个任务的方向夹角为 0° 时抢同一条轴；O-LoRA 把第二个方向投影到正交补上。
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
              夹角 <strong>{angle}°</strong>
            </span>
            <input
              type="range"
              min="0"
              max="90"
              step="5"
              value={angle}
              onChange={(event) => {
                setAngle(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <label>
            <input
              type="checkbox"
              checked={olora}
              onChange={(event) => {
                setOlora(event.target.checked);
                invalidate();
              }}
            />{" "}
            打开 O-LoRA 正交约束
          </label>
          <div className={styles.formula}>
            <code>overlap = |u₁ · u₂|</code>
            <code>T1 剩余 = 1 − 0.92 overlap</code>
          </div>
        </div>

        <div className={styles.stage} aria-live="polite">
          <div className={styles.metrics}>
            <div>
              <span>重叠 |cos|</span>
              <strong>{hasRun ? lora.overlap.toFixed(2) : "?"}</strong>
            </div>
            <div>
              <span>任务 1 剩余</span>
              <strong>{hasRun ? formatPct(lora.t1Remain) : "?"}</strong>
            </div>
            <div>
              <span>任务 2 写入</span>
              <strong>{hasRun ? formatPct(lora.t2) : "?"}</strong>
            </div>
          </div>
          <svg
            className={styles.plot}
            viewBox="0 0 340 300"
            role="img"
            aria-label="两个 LoRA 方向"
          >
            <line x1="40" y1="160" x2="300" y2="160" stroke="currentColor" />
            <line x1="170" y1="30" x2="170" y2="280" stroke="currentColor" />
            {hasRun ? (
              <>
                <line
                  x1={origin.x}
                  y1={origin.y}
                  x2={origin.x + lora.u1[0] * s}
                  y2={origin.y - lora.u1[1] * s}
                  stroke="#2156c8"
                  strokeWidth="3"
                />
                <line
                  x1={origin.x}
                  y1={origin.y}
                  x2={origin.x + lora.u2[0] * s}
                  y2={origin.y - lora.u2[1] * s}
                  stroke="#c47b12"
                  strokeWidth="3"
                />
              </>
            ) : null}
            <text x="24" y="292" fontSize="11">
              蓝=任务 1 橙=任务 2
            </text>
          </svg>
        </div>
      </div>

      <div className={styles.prediction}>
        <fieldset>
          <legend>预测 1：当前设定下，任务 1 的更新被削掉多少？</legend>
          <div className={styles.choiceRow}>
            <button
              type="button"
              aria-pressed={cutPred === "all"}
              onClick={() => {
                setCutPred("all");
                invalidate();
              }}
            >
              几乎全部
            </button>
            <button
              type="button"
              aria-pressed={cutPred === "half"}
              onClick={() => {
                setCutPred("half");
                invalidate();
              }}
            >
              大约一半
            </button>
            <button
              type="button"
              aria-pressed={cutPred === "none"}
              onClick={() => {
                setCutPred("none");
                invalidate();
              }}
            >
              几乎没有
            </button>
          </div>
        </fieldset>
        <fieldset>
          <legend>预测 2：O-LoRA 正交约束主要在防什么？</legend>
          <div className={styles.choiceRow}>
            <button
              type="button"
              aria-pressed={orthoPred === "protect"}
              onClick={() => {
                setOrthoPred("protect");
                invalidate();
              }}
            >
              两个 LoRA 抢同一方向
            </button>
            <button
              type="button"
              aria-pressed={orthoPred === "hurt"}
              onClick={() => {
                setOrthoPred("hurt");
                invalidate();
              }}
            >
              学习率过大
            </button>
          </div>
        </fieldset>
        <button
          type="button"
          className={styles.run}
          disabled={!cutPred || !orthoPred}
          onClick={runLab}
        >
          计算夹角
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
            ? "0° 且关掉正交时削掉几乎全部；90° 或打开 O-LoRA 则几乎不削。"
            : gatePassed
              ? `重叠 ${lora.overlap.toFixed(2)}，任务 1 还剩 ${formatPct(lora.t1Remain)}。`
              : "被削掉的比例约等于 |cos θ|。正交约束把 cos 推到 0。"}
        </span>
      </div>
    </section>
  );
}
