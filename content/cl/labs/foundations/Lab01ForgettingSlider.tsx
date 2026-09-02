"use client";

import { useMemo, useState } from "react";
import type { CSSProperties } from "react";
import styles from "./Lab01ForgettingSlider.module.css";
import type { FoundationLabProps, Point2 } from "./types";
import {
  createRng,
  formatPct,
  initialNumber,
  logisticAcc,
  logisticStep,
  makeBlobs,
} from "./types";

type DropBucket = "lt20" | "20to60" | "60to120" | "never";

const lrChoices = [0.08, 0.22, 0.5, 1] as const;
const bucketLabels: Record<DropBucket, string> = {
  lt20: "不到 20 步",
  "20to60": "20–60 步",
  "60to120": "60–120 步",
  never: "160 步后仍高于随机",
};

const RANDOM_ACC = 0.55;
const T1_STEPS = 60;
const T2_MAX = 160;
const WD = 0.12;
const SEED = 11;

function bucketOf(dropStep: number | null): DropBucket {
  if (dropStep === null || dropStep > 120) return "never";
  if (dropStep < 20) return "lt20";
  if (dropStep <= 60) return "20to60";
  return "60to120";
}

function toSvg(x: number, y: number) {
  const x0 = 28;
  const y0 = 18;
  const width = 260;
  const height = 200;
  return {
    x: x0 + ((x + 2.6) / 5.2) * width,
    y: y0 + ((2.6 - y) / 5.2) * height,
  };
}

function boundaryLine(weights: readonly [number, number], bias: number) {
  const [w0, w1] = weights;
  const xs = [-2.5, 2.5];
  if (Math.abs(w1) < 1e-6) {
    const x = Math.abs(w0) < 1e-6 ? 0 : -bias / w0;
    const a = toSvg(x, -2.5);
    const b = toSvg(x, 2.5);
    return `M ${a.x} ${a.y} L ${b.x} ${b.y}`;
  }
  const p0 = toSvg(xs[0], -(w0 * xs[0] + bias) / w1);
  const p1 = toSvg(xs[1], -(w0 * xs[1] + bias) / w1);
  return `M ${p0.x} ${p0.y} L ${p1.x} ${p1.y}`;
}

function trainSequence(task1: Point2[], task2: Point2[], lr: number) {
  let weights: [number, number] = [0, 0];
  let bias = 0;
  for (let step = 0; step < T1_STEPS; step += 1) {
    const next = logisticStep(weights, bias, task1, lr, WD);
    weights = next.weights;
    bias = next.bias;
  }
  const afterTask1 = {
    weights,
    bias,
    acc1: logisticAcc(weights, bias, task1),
    acc2: logisticAcc(weights, bias, task2),
  };
  const history = [
    {
      step: 0,
      weights,
      bias,
      acc1: afterTask1.acc1,
      acc2: afterTask1.acc2,
    },
  ];
  let dropStep: number | null = null;
  for (let step = 1; step <= T2_MAX; step += 1) {
    const next = logisticStep(weights, bias, task2, lr, WD);
    weights = next.weights;
    bias = next.bias;
    const acc1 = logisticAcc(weights, bias, task1);
    const acc2 = logisticAcc(weights, bias, task2);
    history.push({ step, weights, bias, acc1, acc2 });
    if (dropStep === null && acc1 <= RANDOM_ACC) dropStep = step;
  }
  return { afterTask1, history, dropStep, bucket: bucketOf(dropStep) };
}

export function Lab01ForgettingSlider({
  onComplete,
  initialState,
}: FoundationLabProps) {
  const defaults = {
    lr: initialNumber(initialState, "lr", 0.5),
    task2Steps: initialNumber(initialState, "task2Steps", 40),
  };
  const [lr, setLr] = useState(
    lrChoices.includes(defaults.lr as (typeof lrChoices)[number])
      ? defaults.lr
      : 0.5,
  );
  const [task2Steps, setTask2Steps] = useState(defaults.task2Steps);
  const [dropPrediction, setDropPrediction] = useState<DropBucket | null>(null);
  const [maskPrediction, setMaskPrediction] = useState<"t1" | "both" | null>(
    null,
  );
  const [hasRun, setHasRun] = useState(false);

  const model = useMemo(() => {
    const rng = createRng(SEED);
    const task1 = makeBlobs(rng, 80, [-1.2, -1.2], [1.2, 1.2], 0.4);
    const task2 = makeBlobs(rng, 80, [-1.2, 1.2], [1.2, -1.2], 0.4);
    const sequence = trainSequence(task1, task2, lr);
    return { task1, task2, ...sequence };
  }, [lr]);

  const current =
    model.history[Math.min(task2Steps, model.history.length - 1)] ??
    model.history[0];
  const gatePassed =
    hasRun &&
    dropPrediction === model.bucket &&
    maskPrediction === "t1";

  function invalidate() {
    setHasRun(false);
  }

  function runLab() {
    setHasRun(true);
    if (dropPrediction === model.bucket && maskPrediction === "t1") {
      onComplete?.({
        lr,
        task2Steps,
        dropStep: model.dropStep,
        bucket: model.bucket,
        acc1: current.acc1,
        acc2: current.acc2,
      });
    }
  }

  function reset() {
    setLr(0.5);
    setTask2Steps(40);
    setDropPrediction(null);
    setMaskPrediction(null);
    setHasRun(false);
  }

  const spark = [0, 20, 40, 60, 80, 120, 160];

  return (
    <section className={styles.lab} aria-labelledby="lab01-title">
      <header className={styles.header}>
        <div>
          <div className={styles.labels} aria-label="实验类型">
            <span>教学模拟</span>
            <span>二维高斯</span>
          </div>
          <h3 id="lab01-title">遗忘滑块：任务 2 把决策边界拧走</h3>
          <p>
            灾难性遗忘指新任务学完后，旧任务准确率掉到接近乱猜。这里用两个二维高斯分类任务和一条线性决策边界（把两类分开的那条线），数字来自批量梯度下降，不是 GPU 实测。
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
              任务 2 训练步数 <strong>{task2Steps}</strong>
            </span>
            <input
              type="range"
              min="0"
              max={T2_MAX}
              step="5"
              value={task2Steps}
              onChange={(event) => {
                setTask2Steps(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <label>
            <span>学习率</span>
            <select
              value={lr}
              onChange={(event) => {
                setLr(Number(event.target.value));
                invalidate();
              }}
            >
              <option value="0.08">0.08 很慢</option>
              <option value="0.22">0.22 慢</option>
              <option value="0.5">0.50 默认</option>
              <option value="1">1.00 快</option>
            </select>
          </label>
          <div className={styles.formula}>
            <code>p = σ(w·x + b)</code>
            <code>w ← w − η (∇L + 0.12 w)</code>
            <code>随机阈值：任务 1 准确率 ≤ 55%</code>
          </div>
        </div>

        <div className={styles.stage} aria-live="polite">
          <div className={styles.metrics}>
            <div>
              <span>任务 1 准确率</span>
              <strong>{hasRun ? formatPct(current.acc1) : "?"}</strong>
            </div>
            <div>
              <span>任务 2 准确率</span>
              <strong>{hasRun ? formatPct(current.acc2) : "?"}</strong>
            </div>
            <div>
              <span>掉到随机的步数</span>
              <strong>
                {hasRun
                  ? model.dropStep === null
                    ? "未掉到"
                    : `${model.dropStep}`
                  : "待运行"}
              </strong>
            </div>
          </div>

          <div className={styles.plotWrap}>
            <div>
              <svg
                className={styles.plot}
                viewBox="0 0 316 236"
                role="img"
                aria-label="两个任务的样本和当前决策边界"
              >
                <rect x="28" y="18" width="260" height="200" fill="transparent" />
                {model.task1.map((point, index) => {
                  const p = toSvg(point.x, point.y);
                  return (
                    <circle
                      key={`t1-${index}`}
                      cx={p.x}
                      cy={p.y}
                      r="3.2"
                      fill={point.cls === 1 ? "#2156c8" : "#8aa4dc"}
                      opacity="0.9"
                    />
                  );
                })}
                {model.task2.map((point, index) => {
                  const p = toSvg(point.x, point.y);
                  return (
                    <rect
                      key={`t2-${index}`}
                      x={p.x - 2.4}
                      y={p.y - 2.4}
                      width="4.8"
                      height="4.8"
                      fill="none"
                      stroke="#c47b12"
                      strokeWidth="1.1"
                    />
                  );
                })}
                {hasRun ? (
                  <path
                    d={boundaryLine(current.weights, current.bias)}
                    stroke="var(--ink)"
                    strokeWidth="2"
                    fill="none"
                  />
                ) : null}
              </svg>
              <div className={styles.legend}>
                <span>
                  <i className={styles.swatch} style={{ background: "#2156c8" }} />
                  任务 1 样本
                </span>
                <span>
                  <i className={styles.swatch} style={{ background: "#c47b12" }} />
                  任务 2 样本
                </span>
                <span>实线是当前决策边界</span>
              </div>
            </div>
            <ol className={styles.curveList} aria-label="任务 1 遗忘曲线采样">
              {spark.map((step) => {
                const snap = model.history[step] ?? model.history[0];
                return (
                  <li key={step}>
                    <span>{step} 步</span>
                    <div
                      className={`${styles.track} ${step === task2Steps ? styles.trackT2 : ""}`}
                      aria-hidden="true"
                    >
                      <span
                        style={
                          {
                            "--bar-width": hasRun
                              ? `${Math.max(4, snap.acc1 * 100)}%`
                              : "0%",
                          } as CSSProperties
                        }
                      />
                    </div>
                    <strong>{hasRun ? formatPct(snap.acc1) : "—"}</strong>
                  </li>
                );
              })}
            </ol>
          </div>
        </div>
      </div>

      <div className={styles.prediction}>
        <fieldset>
          <legend>预测 1：按当前学习率，任务 1 多少步后掉到随机？</legend>
          <div className={styles.choiceRow}>
            {(Object.keys(bucketLabels) as DropBucket[]).map((key) => (
              <button
                type="button"
                key={key}
                aria-pressed={dropPrediction === key}
                onClick={() => {
                  setDropPrediction(key);
                  invalidate();
                }}
              >
                {bucketLabels[key]}
              </button>
            ))}
          </div>
        </fieldset>
        <fieldset>
          <legend>预测 2：被拧走的是哪条边界？</legend>
          <div className={styles.choiceRow}>
            <button
              type="button"
              aria-pressed={maskPrediction === "t1"}
              onClick={() => {
                setMaskPrediction("t1");
                invalidate();
              }}
            >
              任务 1 的分界线
            </button>
            <button
              type="button"
              aria-pressed={maskPrediction === "both"}
              onClick={() => {
                setMaskPrediction("both");
                invalidate();
              }}
            >
              两条边界会同时钉死
            </button>
          </div>
        </fieldset>
        <button
          type="button"
          className={styles.run}
          disabled={!dropPrediction || !maskPrediction}
          onClick={runLab}
        >
          运行遗忘滑块
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
            ? "先提交两项预测，再揭示公式结果。改学习率或步数会作废上次运行。"
            : gatePassed
              ? `任务 1 在 ${model.dropStep ?? "160 步内未到达"} 步处穿过 55%。同一条线性边界不能同时保住两条对角分界。`
              : "有一项预测不符。掉到随机的步数随学习率变；顺序训练只会拧向任务 2，任务 1 的线会被抹掉。"}
        </span>
      </div>
    </section>
  );
}
