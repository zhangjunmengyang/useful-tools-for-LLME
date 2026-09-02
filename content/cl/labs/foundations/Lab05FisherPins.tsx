"use client";

import { useMemo, useState } from "react";
import styles from "./Lab05FisherPins.module.css";
import type { FoundationLabProps } from "./types";
import { initialNumber } from "./types";

const F11 = 4;
const F22 = 0.4;
const THETA_NEW: [number, number] = [1.6, 1.2];

function mapPoint(x: number, y: number) {
  return { x: 170 + x * 78, y: 168 - y * 78 };
}

function ewcTheta(lambda: number): [number, number] {
  return [THETA_NEW[0] / (1 + F11 * lambda), THETA_NEW[1] / (1 + F22 * lambda)];
}

function oldLoss(theta: readonly [number, number]) {
  return 0.5 * (F11 * theta[0] * theta[0] + F22 * theta[1] * theta[1]);
}

function newLoss(theta: readonly [number, number]) {
  const dx = theta[0] - THETA_NEW[0];
  const dy = theta[1] - THETA_NEW[1];
  return 0.5 * (dx * dx + dy * dy);
}

export function Lab05FisherPins({
  onComplete,
  initialState,
}: FoundationLabProps) {
  const [lambda, setLambda] = useState(
    initialNumber(initialState, "lambda", 1),
  );
  const [oldPred, setOldPred] = useState<"down" | "up" | null>(null);
  const [axisPred, setAxisPred] = useState<"short" | "long" | null>(null);
  const [hasRun, setHasRun] = useState(false);

  const result = useMemo(() => {
    const lo = ewcTheta(0.1);
    const hi = ewcTheta(10);
    const now = ewcTheta(lambda);
    return {
      now,
      oldNow: oldLoss(now),
      newNow: newLoss(now),
      oldLo: oldLoss(lo),
      oldHi: oldLoss(hi),
      shrinkX: Math.abs(hi[0]) < Math.abs(hi[1]),
    };
  }, [lambda]);

  const gatePassed = hasRun && oldPred === "down" && axisPred === "short";

  function invalidate() {
    setHasRun(false);
  }

  function runLab() {
    setHasRun(true);
    if (oldPred === "down" && axisPred === "short") {
      onComplete?.({
        lambda,
        theta: result.now,
        oldLoss: result.oldNow,
        newLoss: result.newNow,
      });
    }
  }

  function reset() {
    setLambda(1);
    setOldPred(null);
    setAxisPred(null);
    setHasRun(false);
  }

  const star = mapPoint(0, 0);
  const target = mapPoint(THETA_NEW[0], THETA_NEW[1]);
  const now = mapPoint(result.now[0], result.now[1]);
  const path = [0.05, 0.2, 0.6, 1.5, 4, 12]
    .map((value) => {
      const theta = ewcTheta(value);
      const p = mapPoint(theta[0], theta[1]);
      return `${p.x},${p.y}`;
    })
    .join(" ");

  return (
    <section className={styles.lab} aria-labelledby="lab05-title">
      <header className={styles.header}>
        <div>
          <div className={styles.labels}>
            <span>EWC</span>
            <span>Fisher 椭圆</span>
          </div>
          <h3 id="lab05-title">Fisher 钉钉子：λ 把最优点推进椭圆</h3>
          <p>
            Fisher 信息衡量这个权重对旧任务有多敏感。EWC（弹性权重巩固：给重要权重加弹簧）的二次项把 θ 拉向 θ*，弹簧劲度是 λF。
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
              λ <strong>{lambda.toFixed(2)}</strong>
            </span>
            <input
              type="range"
              min="0.05"
              max="16"
              step="0.05"
              value={lambda}
              onChange={(event) => {
                setLambda(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <div className={styles.formula}>
            <code>θ(λ) = (I + λF)⁻¹ θ_new</code>
            <code>F = diag(4, 0.4)</code>
            <code>L_old = ½ θᵀ F θ</code>
          </div>
        </div>

        <div className={styles.stage} aria-live="polite">
          <div className={styles.metrics}>
            <div>
              <span>θ</span>
              <strong>
                {hasRun
                  ? `(${result.now[0].toFixed(2)}, ${result.now[1].toFixed(2)})`
                  : "?"}
              </strong>
            </div>
            <div>
              <span>旧任务二次损失</span>
              <strong>{hasRun ? result.oldNow.toFixed(3) : "?"}</strong>
            </div>
            <div>
              <span>新任务二次损失</span>
              <strong>{hasRun ? result.newNow.toFixed(3) : "?"}</strong>
            </div>
          </div>
          <svg
            className={styles.plot}
            viewBox="0 0 340 300"
            role="img"
            aria-label="旧任务 Fisher 椭圆与新任务最优点"
          >
            <ellipse
              cx={star.x}
              cy={star.y}
              rx={78 * Math.sqrt(1 / F11)}
              ry={78 * Math.sqrt(1 / F22)}
              fill="none"
              stroke="#1b7a53"
              opacity="0.35"
            />
            <ellipse
              cx={star.x}
              cy={star.y}
              rx={78 * Math.sqrt(2.4 / F11)}
              ry={78 * Math.sqrt(2.4 / F22)}
              fill="none"
              stroke="#1b7a53"
              opacity="0.55"
            />
            <ellipse
              cx={star.x}
              cy={star.y}
              rx={78 * Math.sqrt(4.8 / F11)}
              ry={78 * Math.sqrt(4.8 / F22)}
              fill="none"
              stroke="#1b7a53"
            />
            {hasRun ? (
              <polyline
                points={path}
                fill="none"
                stroke="#2156c8"
                strokeWidth="1.6"
              />
            ) : null}
            <circle cx={star.x} cy={star.y} r="4" fill="currentColor" />
            <circle cx={target.x} cy={target.y} r="4" fill="#c47b12" />
            {hasRun ? (
              <circle cx={now.x} cy={now.y} r="6" fill="#2156c8" />
            ) : null}
            <text x={star.x + 8} y={star.y - 8} fontSize="11">
              θ*
            </text>
            <text x={target.x + 8} y={target.y} fontSize="11">
              θ_new
            </text>
            <text x="24" y="24" fontSize="11">
              短轴 = Fisher 大的 θ₁
            </text>
          </svg>
        </div>
      </div>

      <div className={styles.prediction}>
        <fieldset>
          <legend>预测 1：λ 从 0.1 加到 10，旧任务损失怎么变？</legend>
          <div className={styles.choiceRow}>
            <button
              type="button"
              aria-pressed={oldPred === "down"}
              onClick={() => {
                setOldPred("down");
                invalidate();
              }}
            >
              下降
            </button>
            <button
              type="button"
              aria-pressed={oldPred === "up"}
              onClick={() => {
                setOldPred("up");
                invalidate();
              }}
            >
              上升
            </button>
          </div>
        </fieldset>
        <fieldset>
          <legend>预测 2：更先被钉死的是哪根轴？</legend>
          <div className={styles.choiceRow}>
            <button
              type="button"
              aria-pressed={axisPred === "short"}
              onClick={() => {
                setAxisPred("short");
                invalidate();
              }}
            >
              短轴（Fisher 大）
            </button>
            <button
              type="button"
              aria-pressed={axisPred === "long"}
              onClick={() => {
                setAxisPred("long");
                invalidate();
              }}
            >
              长轴（Fisher 小）
            </button>
          </div>
        </fieldset>
        <button
          type="button"
          className={styles.run}
          disabled={!oldPred || !axisPred}
          onClick={runLab}
        >
          钉住并计算
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
            ? "先判断损失方向和被钉的轴，再拖 λ。"
            : gatePassed
              ? `λ=10 时旧损失 ${result.oldHi.toFixed(3)}，低于 λ=0.1 的 ${result.oldLo.toFixed(3)}。θ₁ 按 1/(1+4λ) 收缩，比 θ₂ 更快。`
              : "闭式解是 θ₁=1.6/(1+4λ)，θ₂=1.2/(1+0.4λ)。Fisher 大的轴是椭圆短轴。"}
        </span>
      </div>
    </section>
  );
}
