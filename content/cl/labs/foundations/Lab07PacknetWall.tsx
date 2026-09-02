"use client";

import { useMemo, useState } from "react";
import styles from "./Lab07PacknetWall.module.css";
import type { FoundationLabProps } from "./types";
import {
  createRng,
  formatPct,
  initialNumber,
  makeBlobs,
  sigmoid,
} from "./types";

const CELL_COUNT = 16;
const FEATURES = 16;

function phi(x: number, y: number) {
  return [
    x,
    y,
    1,
    x * y,
    x * x,
    y * y,
    x * x * y,
    x * y * y,
    Math.sin(x),
    Math.sin(y),
    Math.cos(x),
    Math.cos(y),
    Math.tanh(x),
    Math.tanh(y),
    x * x * x,
    y * y * y,
  ];
}

function magAt(index: number) {
  return Math.abs(Math.sin(index * 1.7) * 0.62 + Math.cos(index * 0.41) * 0.38);
}

function trainMasked(
  data: { x: number; y: number; cls: number }[],
  mask: boolean[],
  start: number[],
  steps: number,
  lr: number,
) {
  const weights = [...start];
  for (let step = 0; step < steps; step += 1) {
    const grad = Array(FEATURES).fill(0);
    for (const point of data) {
      const features = phi(point.x, point.y);
      let score = 0;
      for (let i = 0; i < FEATURES; i += 1) score += weights[i] * features[i];
      const err = sigmoid(score) - point.cls;
      for (let i = 0; i < FEATURES; i += 1) {
        if (mask[i]) grad[i] += err * features[i];
      }
    }
    const n = data.length || 1;
    for (let i = 0; i < FEATURES; i += 1) {
      if (mask[i]) weights[i] -= lr * (grad[i] / n + 0.04 * weights[i]);
    }
  }
  return weights;
}

function accOf(
  weights: number[],
  data: { x: number; y: number; cls: number }[],
) {
  let correct = 0;
  for (const point of data) {
    const features = phi(point.x, point.y);
    let score = 0;
    for (let i = 0; i < FEATURES; i += 1) score += weights[i] * features[i];
    if ((score >= 0 ? 1 : 0) === point.cls) correct += 1;
  }
  return correct / data.length;
}

export function Lab07PacknetWall({
  onComplete,
  initialState,
}: FoundationLabProps) {
  const [keepRatio, setKeepRatio] = useState(
    initialNumber(initialState, "keepRatio", 0.5),
  );
  const [capPred, setCapPred] = useState<"stuck" | "fine" | "collapse" | null>(
    null,
  );
  const [lockPred, setLockPred] = useState<"safe" | "overwrite" | null>(null);
  const [hasRun, setHasRun] = useState(false);

  const pack = useMemo(() => {
    const rng = createRng(9);
    const task1 = makeBlobs(rng, 72, [-1.25, -1.25], [1.25, 1.25], 0.38);
    const task2 = makeBlobs(rng, 72, [-1.25, 1.25], [1.25, -1.25], 0.38);
    const task3 = makeBlobs(rng, 72, [-1.4, 0], [1.4, 0], 0.38);
    const free = Array(FEATURES).fill(true);
    const owners = Array(FEATURES).fill(0);
    const locked = Array(FEATURES).fill(false);
    const keepCount = Math.max(1, Math.round(keepRatio * FEATURES));

    let weights = Array(FEATURES).fill(0);
    weights = trainMasked(task1, free, weights, 90, 0.35);
    const ranked = weights
      .map((value, index) => ({ index, mag: Math.abs(value) || magAt(index) }))
      .sort((a, b) => b.mag - a.mag);
    ranked.slice(0, keepCount).forEach((item) => {
      owners[item.index] = 1;
      locked[item.index] = true;
      free[item.index] = false;
    });
    const acc1 = accOf(weights, task1);

    const freeAfter1 = free.filter(Boolean).length;
    weights = trainMasked(task2, free, weights, 90, 0.35);
    const ranked2 = weights
      .map((value, index) => ({ index, mag: Math.abs(value), free: free[index] }))
      .filter((item) => item.free)
      .sort((a, b) => b.mag - a.mag);
    const keep2 = Math.max(1, Math.min(ranked2.length, Math.round(keepRatio * ranked2.length)));
    ranked2.slice(0, keep2).forEach((item) => {
      owners[item.index] = 2;
      locked[item.index] = true;
      free[item.index] = false;
    });
    const acc2 = accOf(weights, task2);
    const acc1After2 = accOf(weights, task1);

    const freeAfter2 = free.filter(Boolean).length;
    weights = trainMasked(task3, free, weights, 90, 0.35);
    free.forEach((isFree, index) => {
      if (isFree) owners[index] = 3;
    });
    const acc3 = accOf(weights, task3);
    const acc1Final = accOf(weights, task1);

    return {
      owners,
      locked,
      acc1,
      acc1After2,
      acc1Final,
      acc2,
      acc3,
      freeAfter1,
      freeAfter2,
      leftover: freeAfter2,
    };
  }, [keepRatio]);

  const stuck = pack.acc2 < 0.7;
  const collapsed = pack.acc1Final < 0.7;
  const capAnswer = collapsed ? "collapse" : stuck ? "stuck" : "fine";
  const gatePassed =
    hasRun && capPred === capAnswer && lockPred === "safe";

  function invalidate() {
    setHasRun(false);
  }

  function runLab() {
    setHasRun(true);
    if (capPred === capAnswer && lockPred === "safe") {
      onComplete?.({
        keepRatio,
        leftover: pack.leftover,
        acc1: pack.acc1Final,
        acc2: pack.acc2,
        acc3: pack.acc3,
      });
    }
  }

  function reset() {
    setKeepRatio(0.5);
    setCapPred(null);
    setLockPred(null);
    setHasRun(false);
  }

  return (
    <section className={styles.lab} aria-labelledby="lab07-title">
      <header className={styles.header}>
        <div>
          <div className={styles.labels}>
            <span>PackNet</span>
            <span>权重砌墙</span>
          </div>
          <h3 id="lab07-title">PackNet 砌墙：旧格子上锁，新任务用剩下的</h3>
          <p>
            PackNet 把权重量级从大到小排队，任务 1 占用一部分并上锁。任务 2 只能改没锁的格子。模型是 16 维特征上的逻辑回归，掩码乘在梯度上。
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
              每任务保留比例 <strong>{Math.round(keepRatio * 100)}%</strong>
            </span>
            <input
              type="range"
              min="0.3"
              max="0.8"
              step="0.05"
              value={keepRatio}
              onChange={(event) => {
                setKeepRatio(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <code className={styles.formula}>
            自由格子 ← 总量 − 已锁；Acc₃ 随自由格子下降
          </code>
        </div>

        <div className={styles.stage} aria-live="polite">
          <div className={styles.metrics}>
            <div>
              <span>任务 1</span>
              <strong>{hasRun ? formatPct(pack.acc1Final) : "?"}</strong>
            </div>
            <div>
              <span>任务 2</span>
              <strong>{hasRun ? formatPct(pack.acc2) : "?"}</strong>
            </div>
            <div>
              <span>任务 3</span>
              <strong>{hasRun ? formatPct(pack.acc3) : "?"}</strong>
            </div>
            <div>
              <span>剩余空格</span>
              <strong>{hasRun ? pack.leftover : "?"}</strong>
            </div>
          </div>
          <div className={styles.wall} aria-label="16 个权重格子">
            {Array.from({ length: CELL_COUNT }, (_, index) => (
              <div
                key={index}
                className={styles.cell}
                data-owner={hasRun ? String(pack.owners[index] || 0) : "0"}
                data-lock={hasRun && pack.locked[index] ? "1" : "0"}
              />
            ))}
          </div>
        </div>
      </div>

      <div className={styles.prediction}>
        <fieldset>
          <legend>预测 1：当前保留比例下，任务 1 上锁后任务 2 会怎样？</legend>
          <div className={styles.choiceRow}>
            <button
              type="button"
              aria-pressed={capPred === "stuck"}
              onClick={() => {
                setCapPred("stuck");
                invalidate();
              }}
            >
              准确率上不去（&lt; 70%）
            </button>
            <button
              type="button"
              aria-pressed={capPred === "fine"}
              onClick={() => {
                setCapPred("fine");
                invalidate();
              }}
            >
              仍能到 70% 以上
            </button>
            <button
              type="button"
              aria-pressed={capPred === "collapse"}
              onClick={() => {
                setCapPred("collapse");
                invalidate();
              }}
            >
              旧任务一起崩
            </button>
          </div>
        </fieldset>
        <fieldset>
          <legend>预测 2：任务 1 的上锁格子会被任务 2 改写吗？</legend>
          <div className={styles.choiceRow}>
            <button
              type="button"
              aria-pressed={lockPred === "safe"}
              onClick={() => {
                setLockPred("safe");
                invalidate();
              }}
            >
              不会
            </button>
            <button
              type="button"
              aria-pressed={lockPred === "overwrite"}
              onClick={() => {
                setLockPred("overwrite");
                invalidate();
              }}
            >
              会
            </button>
          </div>
        </fieldset>
        <button
          type="button"
          className={styles.run}
          disabled={!capPred || !lockPred}
          onClick={runLab}
        >
          砌墙并训练
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
            ? "先判断容量和上锁，再运行掩码训练。"
            : gatePassed
              ? `任务 2 准确率 ${formatPct(pack.acc2)}，任务 1 仍是 ${formatPct(pack.acc1Final)}。上锁格子的梯度被乘了 0。`
              : "保留比例高时，任务 2 可用格子变少，新任务准确率上不去；旧格子上锁后不会被改写。"}
        </span>
      </div>
    </section>
  );
}
