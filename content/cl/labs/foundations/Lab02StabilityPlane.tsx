"use client";

import { useMemo, useState } from "react";
import styles from "./Lab02StabilityPlane.module.css";
import type { FoundationLabProps, Point2 } from "./types";
import {
  createRng,
  formatPct,
  initialNumber,
  logisticAcc,
  logisticStep,
  makeBlobs,
} from "./types";

type MethodKey = "naive" | "freeze" | "smallLr" | "replay";

const methodMeta: Record<
  MethodKey,
  { name: string; note: string; color: string }
> = {
  naive: { name: "顺序微调", note: "任务 2 训满 80 步", color: "#c4472f" },
  freeze: { name: "冻骨干", note: "任务 2 不再改 w", color: "#5b6678" },
  smallLr: { name: "小学习率", note: "只再走 20 步", color: "#c47b12" },
  replay: { name: "混旧数据", note: "任务 1+2 各半", color: "#1b7a53" },
};

const methodKeys: MethodKey[] = ["naive", "freeze", "smallLr", "replay"];

function trainFor(
  start: { weights: [number, number]; bias: number },
  batch: Point2[],
  steps: number,
  lr: number,
  wd: number,
) {
  let weights = start.weights;
  let bias = start.bias;
  for (let i = 0; i < steps; i += 1) {
    const next = logisticStep(weights, bias, batch, lr, wd);
    weights = next.weights;
    bias = next.bias;
  }
  return { weights, bias };
}

export function Lab02StabilityPlane({
  onComplete,
  initialState,
}: FoundationLabProps) {
  const defaults = {
    mixRatio: initialNumber(initialState, "mixRatio", 0.5),
  };
  const [mixRatio, setMixRatio] = useState(defaults.mixRatio);
  const [bestPrediction, setBestPrediction] = useState<MethodKey | null>(null);
  const [forgetPrediction, setForgetPrediction] = useState<MethodKey | null>(
    null,
  );
  const [hasRun, setHasRun] = useState(false);

  const plane = useMemo(() => {
    const rng = createRng(11);
    const task1 = makeBlobs(rng, 80, [-1.2, -1.2], [1.2, 1.2], 0.4);
    const task2 = makeBlobs(rng, 80, [-1.2, 1.2], [1.2, -1.2], 0.4);
    const afterT1 = trainFor({ weights: [0, 0], bias: 0 }, task1, 80, 0.5, 0.05);
    const mixCount = Math.max(8, Math.round(task1.length * mixRatio));
    const mixed = [...task1.slice(0, mixCount), ...task2];
    const points: Record<MethodKey, { acc1: number; acc2: number }> = {
      naive: (() => {
        const fit = trainFor(afterT1, task2, 80, 0.5, 0.12);
        return {
          acc1: logisticAcc(fit.weights, fit.bias, task1),
          acc2: logisticAcc(fit.weights, fit.bias, task2),
        };
      })(),
      freeze: {
        acc1: logisticAcc(afterT1.weights, afterT1.bias, task1),
        acc2: logisticAcc(afterT1.weights, afterT1.bias, task2),
      },
      smallLr: (() => {
        const fit = trainFor(afterT1, task2, 20, 0.5, 0.12);
        return {
          acc1: logisticAcc(fit.weights, fit.bias, task1),
          acc2: logisticAcc(fit.weights, fit.bias, task2),
        };
      })(),
      replay: (() => {
        const fit = trainFor(afterT1, mixed, 80, 0.4, 0.08);
        return {
          acc1: logisticAcc(fit.weights, fit.bias, task1),
          acc2: logisticAcc(fit.weights, fit.bias, task2),
        };
      })(),
    };
    const best = methodKeys.reduce((lead, key) => {
      const score = Math.min(points[key].acc1, points[key].acc2);
      const leadScore = Math.min(points[lead].acc1, points[lead].acc2);
      return score > leadScore ? key : lead;
    }, "replay" as MethodKey);
    const forgetter = methodKeys.reduce((lead, key) => {
      const score = points[key].acc2 - points[key].acc1;
      const leadScore = points[lead].acc2 - points[lead].acc1;
      return score > leadScore ? key : lead;
    }, "naive" as MethodKey);
    return { points, best, forgetter };
  }, [mixRatio]);

  const gatePassed =
    hasRun &&
    bestPrediction === plane.best &&
    forgetPrediction === plane.forgetter;

  function invalidate() {
    setHasRun(false);
  }

  function runLab() {
    setHasRun(true);
    if (bestPrediction === plane.best && forgetPrediction === plane.forgetter) {
      onComplete?.({
        mixRatio,
        best: plane.best,
        forgetter: plane.forgetter,
        points: plane.points,
      });
    }
  }

  function reset() {
    setMixRatio(0.5);
    setBestPrediction(null);
    setForgetPrediction(null);
    setHasRun(false);
  }

  function toX(acc1: number) {
    return 48 + acc1 * 260;
  }
  function toY(acc2: number) {
    return 268 - acc2 * 230;
  }

  return (
    <section className={styles.lab} aria-labelledby="lab02-title">
      <header className={styles.header}>
        <div>
          <div className={styles.labels}>
            <span>教学模拟</span>
            <span>稳定性-可塑性</span>
          </div>
          <h3 id="lab02-title">稳定性-可塑性平面：四个方法四个点</h3>
          <p>
            稳定性是旧任务还在，可塑性是新任务能学会。横轴是任务 1 准确率，纵轴是任务 2 准确率。四个点由同一组二维高斯、同一条线性模型算出来。
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
              混旧数据比例 <strong>{Math.round(mixRatio * 100)}%</strong>
            </span>
            <input
              type="range"
              min="0.3"
              max="0.85"
              step="0.05"
              value={mixRatio}
              onChange={(event) => {
                setMixRatio(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <p>
            改比例只影响「混旧数据」那个点。冻骨干、小学习率和顺序微调的配方写在注释里，方便和课文对齐。
          </p>
          <ul className={styles.methodList}>
            {methodKeys.map((key) => (
              <li key={key}>
                <span>
                  <i
                    className={styles.dot}
                    style={{ ["--m" as string]: methodMeta[key].color }}
                  />{" "}
                  {methodMeta[key].name}
                </span>
                <span>{methodMeta[key].note}</span>
              </li>
            ))}
          </ul>
        </div>

        <div className={styles.stage} aria-live="polite">
          <svg
            className={styles.plot}
            viewBox="0 0 340 300"
            role="img"
            aria-label="稳定性-可塑性平面"
          >
            <line x1="48" y1="268" x2="318" y2="268" stroke="currentColor" />
            <line x1="48" y1="268" x2="48" y2="28" stroke="currentColor" />
            <text x="168" y="292" fontSize="11" fill="currentColor">
              旧任务保持
            </text>
            <text
              x="16"
              y="160"
              fontSize="11"
              fill="currentColor"
              transform="rotate(-90 16 160)"
            >
              新任务准确率
            </text>
            <line
              x1={toX(0.7)}
              y1="28"
              x2={toX(0.7)}
              y2="268"
              stroke="currentColor"
              strokeDasharray="3 4"
              opacity="0.35"
            />
            <line
              x1="48"
              y1={toY(0.7)}
              x2="318"
              y2={toY(0.7)}
              stroke="currentColor"
              strokeDasharray="3 4"
              opacity="0.35"
            />
            {hasRun
              ? methodKeys.map((key) => {
                  const point = plane.points[key];
                  return (
                    <g key={key}>
                      <circle
                        cx={toX(point.acc1)}
                        cy={toY(point.acc2)}
                        r="7"
                        fill={methodMeta[key].color}
                      />
                      <text
                        x={toX(point.acc1) + 10}
                        y={toY(point.acc2) - 8}
                        fontSize="11"
                        fill="currentColor"
                      >
                        {methodMeta[key].name} {formatPct(point.acc1)} /{" "}
                        {formatPct(point.acc2)}
                      </text>
                    </g>
                  );
                })
              : null}
          </svg>
        </div>
      </div>

      <div className={styles.prediction}>
        <fieldset>
          <legend>预测 1：哪个点最靠右上（两边都高）？</legend>
          <div className={styles.choiceRow}>
            {methodKeys.map((key) => (
              <button
                type="button"
                key={key}
                aria-pressed={bestPrediction === key}
                onClick={() => {
                  setBestPrediction(key);
                  invalidate();
                }}
              >
                {methodMeta[key].name}
              </button>
            ))}
          </div>
        </fieldset>
        <fieldset>
          <legend>预测 2：哪个点在左上（旧低新高）？</legend>
          <div className={styles.choiceRow}>
            {methodKeys.map((key) => (
              <button
                type="button"
                key={key}
                aria-pressed={forgetPrediction === key}
                onClick={() => {
                  setForgetPrediction(key);
                  invalidate();
                }}
              >
                {methodMeta[key].name}
              </button>
            ))}
          </div>
        </fieldset>
        <button
          type="button"
          className={styles.run}
          disabled={!bestPrediction || !forgetPrediction}
          onClick={runLab}
        >
          揭晓平面
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
            ? "先标出右上和左上，再看四个真实点。"
            : gatePassed
              ? `双高点是${methodMeta[plane.best].name}，左上是${methodMeta[plane.forgetter].name}。生物里的海马-新皮层有睡眠和分离编码器，这条线性模型没有。`
              : "看 min(旧, 新) 最大的点，以及新减旧最大的点。"}
        </span>
      </div>
    </section>
  );
}
