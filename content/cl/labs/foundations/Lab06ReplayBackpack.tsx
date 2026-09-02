"use client";

import { useMemo, useState } from "react";
import styles from "./Lab06ReplayBackpack.module.css";
import type { FoundationLabProps, Point2 } from "./types";
import {
  createRng,
  formatPct,
  gaussian,
  initialBoolean,
  initialNumber,
  softmax,
} from "./types";

const CAPACITY_CHOICES = [4, 8, 16, 32] as const;
const SEED = 7;
const KEEP_TARGET = 0.8;

function snapCapacity(value: number) {
  return CAPACITY_CHOICES.reduce((best, item) =>
    Math.abs(item - value) < Math.abs(best - value) ? item : best,
  );
}

function makeClass(
  rng: () => number,
  count: number,
  mean: readonly [number, number],
  label: number,
): Point2[] {
  const points: Point2[] = [];
  for (let i = 0; i < count; i += 1) {
    points.push({
      x: mean[0] + 0.32 * gaussian(rng),
      y: mean[1] + 0.32 * gaussian(rng),
      cls: label,
    });
  }
  return points;
}

function reservoir(stream: Point2[], size: number, rng: () => number) {
  const buffer: Point2[] = [];
  stream.forEach((item, index) => {
    if (buffer.length < size) {
      buffer.push(item);
      return;
    }
    const j = Math.floor(rng() * (index + 1));
    if (j < size) buffer[j] = item;
  });
  return buffer;
}

function trainSoftmax(buffer: Point2[], steps: number) {
  const weights = [
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
  ];
  const bias = [0, 0, 0, 0];
  const lr = 0.5;
  function logits(point: Point2) {
    return weights.map(
      (row, k) => row[0] * point.x + row[1] * point.y + bias[k],
    );
  }
  for (let step = 0; step < steps; step += 1) {
    const gW = [
      [0, 0],
      [0, 0],
      [0, 0],
      [0, 0],
    ];
    const gB = [0, 0, 0, 0];
    for (const point of buffer) {
      const probs = softmax(logits(point));
      for (let k = 0; k < 4; k += 1) {
        const err = probs[k] - (k === point.cls ? 1 : 0);
        gW[k][0] += err * point.x;
        gW[k][1] += err * point.y;
        gB[k] += err;
      }
    }
    const n = buffer.length || 1;
    for (let k = 0; k < 4; k += 1) {
      weights[k][0] -= lr * gW[k][0] / n;
      weights[k][1] -= lr * gW[k][1] / n;
      bias[k] -= lr * gB[k] / n;
    }
  }
  return { weights, bias, logits };
}

function accuracy(
  model: ReturnType<typeof trainSoftmax>,
  data: Point2[],
) {
  if (data.length === 0) return 0;
  let correct = 0;
  for (const point of data) {
    const logits = model.logits(point);
    let arg = 0;
    for (let k = 1; k < 4; k += 1) {
      if (logits[k] > logits[arg]) arg = k;
    }
    if (arg === point.cls) correct += 1;
  }
  return correct / data.length;
}

function evalBuffer(
  task1: Point2[],
  task2: Point2[],
  size: number,
  distill: boolean,
  rng: () => number,
) {
  const stream = [...task1, ...task2];
  const buffer = reservoir(stream, size, rng);
  const means = distill
    ? [
        { x: -1.4, y: -1.4, cls: 0 },
        { x: 1.4, y: 1.4, cls: 1 },
      ]
    : [];
  const trainSet = buffer.length ? [...buffer, ...means] : task2;
  const model = trainSoftmax(trainSet, 150);
  return {
    buffer,
    n1: buffer.filter((point) => point.cls < 2).length,
    acc1: accuracy(model, task1),
    acc2: accuracy(model, task2),
  };
}

export function Lab06ReplayBackpack({
  onComplete,
  initialState,
}: FoundationLabProps) {
  const [bufferSize, setBufferSize] = useState(
    snapCapacity(initialNumber(initialState, "bufferSize", 8)),
  );
  const [distill, setDistill] = useState(
    initialBoolean(initialState, "distill", false),
  );
  const [nPred, setNPred] = useState<number | null>(null);
  const [rolePred, setRolePred] = useState<"old" | "new" | null>(null);
  const [hasRun, setHasRun] = useState(false);

  const result = useMemo(() => {
    const rngData = createRng(SEED);
    const task1 = [
      ...makeClass(rngData, 36, [-1.4, -1.4], 0),
      ...makeClass(rngData, 36, [1.4, 1.4], 1),
    ];
    const task2 = [
      ...makeClass(rngData, 36, [-1.4, 1.4], 2),
      ...makeClass(rngData, 36, [1.4, -1.4], 3),
    ];
    const byN = CAPACITY_CHOICES.map((size) =>
      evalBuffer(task1, task2, size, distill, createRng(SEED * 100 + size)),
    );
    const minKeep =
      CAPACITY_CHOICES.find((_, index) => byN[index].acc1 >= KEEP_TARGET) ??
      32;
    const snapped = snapCapacity(bufferSize);
    const currentIndex = CAPACITY_CHOICES.indexOf(snapped);
    const current = byN[currentIndex];
    return { byN, minKeep, current };
  }, [bufferSize, distill]);

  const gatePassed = hasRun && nPred === result.minKeep && rolePred === "old";

  function invalidate() {
    setHasRun(false);
  }

  function runLab() {
    setHasRun(true);
    if (nPred === result.minKeep && rolePred === "old") {
      onComplete?.({
        bufferSize,
        distill,
        minKeep: result.minKeep,
        acc1: result.current.acc1,
        acc2: result.current.acc2,
        n1: result.current.n1,
      });
    }
  }

  function reset() {
    setBufferSize(8);
    setDistill(false);
    setNPred(null);
    setRolePred(null);
    setHasRun(false);
  }

  return (
    <section className={styles.lab} aria-labelledby="lab06-title">
      <header className={styles.header}>
        <div>
          <div className={styles.labels}>
            <span>回放缓冲</span>
            <span>四类 Softmax</span>
          </div>
          <h3 id="lab06-title">背包容量：旧样本被挤出之后</h3>
          <p>
            回放缓冲是随身带的旧样本格子。新任务进来用蓄水池抽样往里挤，旧格子会被顶掉。分类器只在背包里做四类 softmax。
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
              背包格子 N <strong>{bufferSize}</strong>
            </span>
            <input
              type="range"
              min="4"
              max="32"
              step="4"
              value={bufferSize}
              onChange={(event) => {
                setBufferSize(snapCapacity(Number(event.target.value)));
                invalidate();
              }}
            />
          </label>
          <label>
            <input
              type="checkbox"
              checked={distill}
              onChange={(event) => {
                setDistill(event.target.checked);
                invalidate();
              }}
            />{" "}
            打开蒸馏原型（DER 的缩小版）
          </label>
          <div className={styles.formula}>
            <code>蓄水池：j ~ Uniform(0, i)，j &lt; N 则替换</code>
            <code>min N s.t. Acc(T1) ≥ 80%</code>
          </div>
        </div>

        <div className={styles.stage} aria-live="polite">
          <div className={styles.metrics}>
            <div>
              <span>背包里的任务 1</span>
              <strong>{hasRun ? result.current.n1 : "?"}</strong>
            </div>
            <div>
              <span>任务 1 准确率</span>
              <strong>{hasRun ? formatPct(result.current.acc1) : "?"}</strong>
            </div>
            <div>
              <span>任务 2 准确率</span>
              <strong>{hasRun ? formatPct(result.current.acc2) : "?"}</strong>
            </div>
            <div>
              <span>保住 80% 的最小 N</span>
              <strong>{hasRun ? result.minKeep : "待运行"}</strong>
            </div>
          </div>
          <div className={styles.slots} aria-label="背包格子">
            {Array.from({ length: bufferSize }, (_, index) => {
              const point = result.current.buffer[index];
              const task = point ? (point.cls < 2 ? "1" : "2") : "0";
              return (
                <div
                  key={index}
                  className={styles.slot}
                  data-task={hasRun ? task : "0"}
                />
              );
            })}
          </div>
        </div>
      </div>

      <div className={styles.prediction}>
        <fieldset>
          <legend>预测 1：N 至少多少，任务 1 才能保住 80%？</legend>
          <div className={styles.choiceRow}>
            {CAPACITY_CHOICES.map((size) => (
              <button
                type="button"
                key={size}
                aria-pressed={nPred === size}
                onClick={() => {
                  setNPred(size);
                  invalidate();
                }}
              >
                {size}
              </button>
            ))}
          </div>
        </fieldset>
        <fieldset>
          <legend>预测 2：被挤出背包的主要是谁？</legend>
          <div className={styles.choiceRow}>
            <button
              type="button"
              aria-pressed={rolePred === "old"}
              onClick={() => {
                setRolePred("old");
                invalidate();
              }}
            >
              更早的任务 1 样本
            </button>
            <button
              type="button"
              aria-pressed={rolePred === "new"}
              onClick={() => {
                setRolePred("new");
                invalidate();
              }}
            >
              刚进来的任务 2 样本
            </button>
          </div>
        </fieldset>
        <button
          type="button"
          className={styles.run}
          disabled={nPred === null || !rolePred}
          onClick={runLab}
        >
          装满背包
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
            ? "先猜最小 N，再看蓄水池抽样结果。"
            : gatePassed
              ? `当前设定下最小 N 是 ${result.minKeep}。蓄水池对每个新样本一视同仁，先来的任务 1 更容易被顶掉。`
              : "扫一遍 4/8/16/32，看任务 1 准确率第一次跨过 80% 的格子数。"}
        </span>
      </div>
    </section>
  );
}
