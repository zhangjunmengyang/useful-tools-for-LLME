"use client";

import { useMemo, useState } from "react";
import styles from "./Lab09DataMix.module.css";
import type { FoundationLabProps } from "./types";
import { formatPct, initialNumber } from "./types";

type MixBucket = "20" | "40" | "60" | "80";
type AnalogKey = "replay" | "fisher" | "pack";

function domainScore(r: number) {
  return 0.18 + 0.7 * (1 - Math.exp(-2.2 * r)) - 0.02 * r;
}

function generalScore(r: number) {
  return 0.86 * Math.exp(-1.15 * r) + 0.06;
}

function bucketOf(ratio: number): MixBucket {
  const pct = ratio * 100;
  const marks: MixBucket[] = ["20", "40", "60", "80"];
  return marks.reduce((best, mark) =>
    Math.abs(Number(mark) - pct) < Math.abs(Number(best) - pct) ? mark : best,
  );
}

export function Lab09DataMix({
  onComplete,
  initialState,
}: FoundationLabProps) {
  const [domainRatio, setDomainRatio] = useState(
    initialNumber(initialState, "domainRatio", 0.7),
  );
  const [crossPred, setCrossPred] = useState<MixBucket | null>(null);
  const [analogPred, setAnalogPred] = useState<AnalogKey | null>(null);
  const [hasRun, setHasRun] = useState(false);

  const curves = useMemo(() => {
    const samples = Array.from({ length: 101 }, (_, i) => {
      const r = i / 100;
      return { r, d: domainScore(r), g: generalScore(r) };
    });
    const cross = samples.reduce((best, item) =>
      Math.abs(item.d - item.g) < Math.abs(best.d - best.g) ? item : best,
    );
    return {
      samples,
      cross,
      bucket: bucketOf(cross.r),
      nowD: domainScore(domainRatio),
      nowG: generalScore(domainRatio),
    };
  }, [domainRatio]);

  const gatePassed =
    hasRun && crossPred === curves.bucket && analogPred === "replay";

  function invalidate() {
    setHasRun(false);
  }

  function runLab() {
    setHasRun(true);
    if (crossPred === curves.bucket && analogPred === "replay") {
      onComplete?.({
        domainRatio,
        crossRatio: curves.cross.r,
        bucket: curves.bucket,
        domain: curves.nowD,
        general: curves.nowG,
      });
    }
  }

  function reset() {
    setDomainRatio(0.7);
    setCrossPred(null);
    setAnalogPred(null);
    setHasRun(false);
  }

  function xOf(r: number) {
    return 36 + r * 270;
  }
  function yOf(score: number) {
    return 250 - score * 210;
  }

  return (
    <section className={styles.lab} aria-labelledby="lab09-title">
      <header className={styles.header}>
        <div>
          <div className={styles.labels}>
            <span>续预训练</span>
            <span>数据配比</span>
          </div>
          <h3 id="lab09-title">数据配比：两条能力曲线在哪交叉</h3>
          <p>
            领域数据比例越高，领域分数上去，通用分数下来。回放通用数据相当于第 06 课把旧样本带回背包。
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
              领域数据比例 <strong>{Math.round(domainRatio * 100)}%</strong>
            </span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.02"
              value={domainRatio}
              onChange={(event) => {
                setDomainRatio(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <div className={styles.formula}>
            <code>{"S_d(r) = 0.18 + 0.70(1-exp(-2.2 r)) - 0.02 r"}</code>
            <code>{"S_g(r) = 0.86 exp(-1.15 r) + 0.06"}</code>
          </div>
        </div>

        <div className={styles.stage} aria-live="polite">
          <div className={styles.metrics}>
            <div>
              <span>领域能力</span>
              <strong>{hasRun ? formatPct(curves.nowD) : "?"}</strong>
            </div>
            <div>
              <span>通用能力</span>
              <strong>{hasRun ? formatPct(curves.nowG) : "?"}</strong>
            </div>
            <div>
              <span>交叉点</span>
              <strong>
                {hasRun ? `${Math.round(curves.cross.r * 100)}%` : "待运行"}
              </strong>
            </div>
          </div>
          <svg
            className={styles.plot}
            viewBox="0 0 340 280"
            role="img"
            aria-label="领域与通用能力曲线"
          >
            <line x1="36" y1="250" x2="310" y2="250" stroke="currentColor" />
            <line x1="36" y1="250" x2="36" y2="28" stroke="currentColor" />
            {hasRun ? (
              <>
                <polyline
                  fill="none"
                  stroke="#2156c8"
                  strokeWidth="2"
                  points={curves.samples
                    .map((item) => `${xOf(item.r)},${yOf(item.d)}`)
                    .join(" ")}
                />
                <polyline
                  fill="none"
                  stroke="#c47b12"
                  strokeWidth="2"
                  points={curves.samples
                    .map((item) => `${xOf(item.r)},${yOf(item.g)}`)
                    .join(" ")}
                />
                <circle
                  cx={xOf(curves.cross.r)}
                  cy={yOf(curves.cross.d)}
                  r="5"
                  fill="currentColor"
                />
                <line
                  x1={xOf(domainRatio)}
                  y1="28"
                  x2={xOf(domainRatio)}
                  y2="250"
                  stroke="currentColor"
                  strokeDasharray="4 4"
                />
              </>
            ) : null}
            <text x="44" y="20" fontSize="11">
              蓝=领域 橙=通用
            </text>
          </svg>
        </div>
      </div>

      <div className={styles.prediction}>
        <fieldset>
          <legend>预测 1：两条曲线交叉时，领域比例最接近？</legend>
          <div className={styles.choiceRow}>
            {(["20", "40", "60", "80"] as MixBucket[]).map((key) => (
              <button
                type="button"
                key={key}
                aria-pressed={crossPred === key}
                onClick={() => {
                  setCrossPred(key);
                  invalidate();
                }}
              >
                {key}%
              </button>
            ))}
          </div>
        </fieldset>
        <fieldset>
          <legend>预测 2：回放通用数据相当于第 06 课的哪件事？</legend>
          <div className={styles.choiceRow}>
            <button
              type="button"
              aria-pressed={analogPred === "replay"}
              onClick={() => {
                setAnalogPred("replay");
                invalidate();
              }}
            >
              把旧样本带回背包
            </button>
            <button
              type="button"
              aria-pressed={analogPred === "fisher"}
              onClick={() => {
                setAnalogPred("fisher");
                invalidate();
              }}
            >
              Fisher 弹簧
            </button>
            <button
              type="button"
              aria-pressed={analogPred === "pack"}
              onClick={() => {
                setAnalogPred("pack");
                invalidate();
              }}
            >
              给权重砌墙
            </button>
          </div>
        </fieldset>
        <button
          type="button"
          className={styles.run}
          disabled={!crossPred || !analogPred}
          onClick={runLab}
        >
          画出配比曲线
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
            ? "先猜交叉点，再揭示公式曲线。"
            : gatePassed
              ? `交叉在 ${Math.round(curves.cross.r * 100)}%，最接近 40%。通用数据就是旧能力的回放。`
              : "在 0 到 1 上扫 r，找 S_d(r) 与 S_g(r) 最接近的位置。"}
        </span>
      </div>
    </section>
  );
}
