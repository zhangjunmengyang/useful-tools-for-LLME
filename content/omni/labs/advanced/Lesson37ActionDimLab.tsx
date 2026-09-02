"use client";

import { useEffect, useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import { numberFrom, round, stringFrom, useCompletionGate } from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson37ActionDimLab.module.css";

const STEPS = 64;
const CONTROL_HZ = 20;
const TOKEN_BUDGET = 168;
const BIT_BUDGET = 336;
const H_FIXED = 8;
const FINGER_HZ = 2.5;
const FINGER_AMP = 0.28;
const TOKEN_BINS = 8;

function clamp(value: number, low = -1, high = 1) {
  return Math.min(high, Math.max(low, value));
}

function chunkLength(dims: number) {
  return Math.max(1, Math.floor(TOKEN_BUDGET / dims));
}

function bitsPerScalar(dims: number) {
  return BIT_BUDGET / (H_FIXED * dims);
}

function binsFromBits(bits: number) {
  return 2 ** Math.max(1, Math.floor(bits));
}

function binWidth(bins: number) {
  return 2 / bins;
}

function uniformBin(value: number, bins: number) {
  const width = binWidth(bins);
  const clipped = clamp(value);
  const index = Math.min(bins - 1, Math.floor((clipped + 1) / width));
  return -1 + (index + 0.5) * width;
}

function makeDemo(dims: number) {
  return Array.from({ length: STEPS }, (_, index) => {
    const phase = index / (STEPS - 1);
    const time = index / CONTROL_HZ;
    const finger = FINGER_AMP * Math.sin(2 * Math.PI * FINGER_HZ * time);
    const torso = -0.75 + 1.5 * phase;
    const wrist = 0.2 * Math.sin(2 * Math.PI * 0.55 * time);
    const row = [clamp(finger), clamp(torso), clamp(wrist)];
    while (row.length < dims) {
      const extra = row.length;
      row.push(
        clamp(0.06 * Math.sin(2 * Math.PI * 0.15 * time + 0.17 * extra)),
      );
    }
    return row.slice(0, dims);
  });
}

function subsampleIndices(horizon: number) {
  if (horizon >= STEPS) return Array.from({ length: STEPS }, (_, index) => index);
  if (horizon === 1) return [0];
  return Array.from({ length: horizon }, (_, index) =>
    Math.floor((index * (STEPS - 1)) / (horizon - 1)),
  );
}

function reconstruct(trajectory: number[][], horizon: number, bins: number) {
  const dims = trajectory[0].length;
  const anchors = subsampleIndices(horizon);
  const quantized = anchors.map((time) =>
    trajectory[time].map((value) => uniformBin(value, bins)),
  );
  const restored = Array.from({ length: STEPS }, () => Array(dims).fill(0));
  for (let dim = 0; dim < dims; dim += 1) {
    for (let time = 0; time < STEPS; time += 1) {
      if (time <= anchors[0]) {
        restored[time][dim] = quantized[0][dim];
        continue;
      }
      if (time >= anchors[anchors.length - 1]) {
        restored[time][dim] = quantized[quantized.length - 1][dim];
        continue;
      }
      let right = 1;
      while (anchors[right] < time) right += 1;
      const left = right - 1;
      const span = anchors[right] - anchors[left];
      const weight = (time - anchors[left]) / span;
      restored[time][dim] =
        (1 - weight) * quantized[left][dim] + weight * quantized[right][dim];
    }
  }
  return restored;
}

function movingAverage(series: number[], window = 5) {
  const half = Math.floor(window / 2);
  return series.map((_, index) => {
    const start = Math.max(0, index - half);
    const end = Math.min(series.length, index + half + 1);
    let total = 0;
    for (let cursor = start; cursor < end; cursor += 1) total += series[cursor];
    return total / (end - start);
  });
}

function residualRms(series: number[]) {
  const baseline = movingAverage(series);
  const energy = series.reduce((sum, value, index) => {
    const delta = value - baseline[index];
    return sum + delta * delta;
  }, 0);
  return Math.sqrt(energy / series.length);
}

function highFreqError(original: number[], reconstructed: number[]) {
  const baseline = movingAverage(original);
  const energy = original.reduce((sum, value, index) => {
    const source = value - baseline[index];
    const recon = reconstructed[index] - baseline[index];
    const delta = source - recon;
    return sum + delta * delta;
  }, 0);
  return Math.sqrt(energy / original.length);
}

function dimMse(original: number[][], reconstructed: number[][], dim: number) {
  const energy = original.reduce((sum, row, index) => {
    const delta = row[dim] - reconstructed[index][dim];
    return sum + delta * delta;
  }, 0);
  return energy / original.length;
}

function toPolyline(series: number[], width: number, height: number) {
  return series
    .map((value, index) => {
      const x = (index / (series.length - 1)) * width;
      const y = height / 2 - value * (height * 0.42);
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

export function Lesson37ActionDimLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    dims: numberFrom(initialState, "dims", 12, 7, 24),
    pred: stringFrom(initialState, "pred", ""),
  };
  const [dims, setDims] = useState(Math.min(24, Math.max(7, Math.round(defaults.dims))));
  const [pred, setPred] = useState(defaults.pred);
  const [ran, setRan] = useState(false);
  const [visited7, setVisited7] = useState(false);
  const [visited24, setVisited24] = useState(false);
  const [errorAt7, setErrorAt7] = useState<number | null>(null);
  const [errorAt24, setErrorAt24] = useState<number | null>(null);

  const calculation = useMemo(() => {
    const horizon = chunkLength(dims);
    const demo = makeDemo(dims);
    const reconstructed = reconstruct(demo, horizon, TOKEN_BINS);
    const fingerOrig = demo.map((row) => row[0]);
    const fingerRecon = reconstructed.map((row) => row[0]);
    const torsoOrig = demo.map((row) => row[1]);
    const torsoRecon = reconstructed.map((row) => row[1]);
    const bits = bitsPerScalar(dims);
    const bins = binsFromBits(bits);
    const originalWiggle = residualRms(fingerOrig);
    const reconWiggle = residualRms(fingerRecon);
    return {
      horizon,
      openLoop: horizon / CONTROL_HZ,
      bits,
      bins,
      width: binWidth(bins),
      fingerError: highFreqError(fingerOrig, fingerRecon),
      fingerRemain: originalWiggle < 1e-12 ? 0 : reconWiggle / originalWiggle,
      fingerMse: dimMse(demo, reconstructed, 0),
      torsoMse: dimMse(demo, reconstructed, 1),
      fingerOrig,
      fingerRecon,
      torsoOrig,
      torsoRecon,
    };
  }, [dims]);

  useEffect(() => {
    if (!ran) return;
    if (dims === 7) {
      setVisited7(true);
      setErrorAt7(calculation.fingerError);
    }
    if (dims === 24) {
      setVisited24(true);
      setErrorAt24(calculation.fingerError);
    }
  }, [calculation.fingerError, dims, ran]);

  const canRun = pred !== "";
  const errorRose =
    errorAt7 !== null &&
    errorAt24 !== null &&
    errorAt24 > errorAt7 + 0.05;
  const passed = ran && pred === "rise" && visited7 && visited24 && errorRose;

  const completion = useMemo(
    () => ({
      lessonId: 37,
      dims,
      pred,
      horizon: calculation.horizon,
      openLoop: round(calculation.openLoop, 4),
      width: round(calculation.width, 5),
      fingerError: round(calculation.fingerError, 5),
      fingerRemain: round(calculation.fingerRemain, 5),
      visited7,
      visited24,
      errorAt7: errorAt7 === null ? null : round(errorAt7, 5),
      errorAt24: errorAt24 === null ? null : round(errorAt24, 5),
    }),
    [
      calculation.fingerError,
      calculation.fingerRemain,
      calculation.horizon,
      calculation.openLoop,
      calculation.width,
      dims,
      errorAt24,
      errorAt7,
      pred,
      visited24,
      visited7,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function reset() {
    setDims(12);
    setPred("");
    setRan(false);
    setVisited7(false);
    setVisited24(false);
    setErrorAt7(null);
    setErrorAt24(null);
  }

  const wrongPrediction = ran && pred !== "rise";

  return (
    <LabFrame
      lesson="37"
      title="固定 token 数时维数升高丢掉哪一截"
      description="教学模拟，不是模型输出。token 预算锁在 168。先预测手指高频误差的方向，再把维数滑到 7 和 24，对照开环窗口与每维箱宽。"
    >
      <div className={styles.workspace}>
        <form
          className={styles.controls}
          onSubmit={(event) => event.preventDefault()}
        >
          <h3>预算控制台</h3>
          <label>
            <span>
              动作维数 d <output>{dims}</output>
            </span>
            <input
              type="range"
              min={7}
              max={24}
              step={1}
              value={dims}
              onChange={(event) => setDims(Number(event.target.value))}
            />
          </label>
          <p className={styles.note}>
            C = H · d = {TOKEN_BUDGET}。当前 H = {calculation.horizon}，开环{" "}
            {calculation.openLoop.toFixed(2)} s。比特账按 H=8、C_bit={BIT_BUDGET}{" "}
            另计，用来对照每维箱宽。
          </p>
        </form>

        <div className={styles.stage}>
          <div className={styles.formula}>
            <span>H = floor(C / d)；Δ = 2 / 2^⌊C_bit / (8d)⌋</span>
            <strong>
              H={calculation.horizon}；Δ={calculation.width.toFixed(4)}（{calculation.bins} 箱）
            </strong>
          </div>
          <div className={styles.plots}>
            <div className={styles.plot}>
              <header>
                <b>手指 2.5 Hz</b>
                <span>{ran ? "原轨迹 / 重建" : "只显示原轨迹"}</span>
              </header>
              <svg viewBox="0 0 320 132" role="img" aria-label="手指轨迹">
                <line
                  x1="0"
                  y1="66"
                  x2="320"
                  y2="66"
                  stroke="currentColor"
                  strokeOpacity="0.18"
                />
                <polyline
                  fill="none"
                  stroke="#314338"
                  strokeWidth="2"
                  points={toPolyline(calculation.fingerOrig, 320, 132)}
                />
                {ran ? (
                  <polyline
                    fill="none"
                    stroke="#176f48"
                    strokeWidth="2"
                    strokeDasharray="5 4"
                    points={toPolyline(calculation.fingerRecon, 320, 132)}
                  />
                ) : null}
              </svg>
            </div>
            <div className={styles.plot}>
              <header>
                <b>躯干慢速到达</b>
                <span>{ran ? "原轨迹 / 重建" : "只显示原轨迹"}</span>
              </header>
              <svg viewBox="0 0 320 132" role="img" aria-label="躯干轨迹">
                <line
                  x1="0"
                  y1="66"
                  x2="320"
                  y2="66"
                  stroke="currentColor"
                  strokeOpacity="0.18"
                />
                <polyline
                  fill="none"
                  stroke="#314338"
                  strokeWidth="2"
                  points={toPolyline(calculation.torsoOrig, 320, 132)}
                />
                {ran ? (
                  <polyline
                    fill="none"
                    stroke="#176f48"
                    strokeWidth="2"
                    strokeDasharray="5 4"
                    points={toPolyline(calculation.torsoRecon, 320, 132)}
                  />
                ) : null}
              </svg>
            </div>
          </div>
          <div className={styles.legend}>
            <span className={styles.orig}>
              <i />
              原示教
            </span>
            <span className={styles.recon}>
              <i />
              {ran ? "H 个锚点量化后插值" : "未揭晓"}
            </span>
          </div>
          <dl className={styles.metrics}>
            <div>
              <dt>开环 H/f</dt>
              <dd>{ran ? `${calculation.openLoop.toFixed(2)} s` : "—"}</dd>
            </div>
            <div>
              <dt>每维箱宽 Δ</dt>
              <dd>{ran ? calculation.width.toFixed(4) : "—"}</dd>
            </div>
            <div>
              <dt>手指高频误差</dt>
              <dd>{ran ? calculation.fingerError.toFixed(3) : "—"}</dd>
            </div>
            <div>
              <dt>手指残留 / 躯干 MSE</dt>
              <dd>
                {ran
                  ? `${calculation.fingerRemain.toFixed(2)} / ${calculation.torsoMse.toFixed(3)}`
                  : "—"}
              </dd>
            </div>
          </dl>
        </div>
      </div>

      <div className={styles.predict}>
        <fieldset>
          <legend>
            先预测：token 数锁死，把 d 从 7 加到 24，手指 2.5 Hz 的重建误差会怎样？
          </legend>
          {(
            [
              ["rise", "升高，时间样本变稀，高频先糊"],
              ["flat", "几乎不变，只是多了几维"],
              ["drop", "下降，维数多更接近真手"],
            ] as const
          ).map(([value, label]) => (
            <label key={value}>
              <input
                type="radio"
                name="pred-hf"
                value={value}
                checked={pred === value}
                onChange={() => {
                  setPred(value);
                  setRan(false);
                  setVisited7(false);
                  setVisited24(false);
                  setErrorAt7(null);
                  setErrorAt24(null);
                }}
              />
              <span>{label}</span>
            </label>
          ))}
        </fieldset>
        <div className={styles.actions}>
          <button type="button" className={styles.reset} onClick={reset}>
            重置
          </button>
          <button
            type="button"
            className={styles.run}
            disabled={!canRun}
            onClick={() => setRan(true)}
          >
            揭晓重建
          </button>
        </div>
      </div>
      {wrongPrediction ? (
        <p className={styles.feedback}>
          C=168 时 d=24 只能留下 H=7 个时间锚点，有效采样率掉到约 2.2
          Hz，2.5 Hz 手指低于奈奎斯特。平均 L2 几乎不动，是因为躯干和填充维把误差摊薄了。
        </p>
      ) : null}
      {ran && pred === "rise" && !passed ? (
        <p className={styles.feedback}>
          预测正确。请把滑条分别停在 d=7 与 d=24，对照手指高频误差和开环时长。
        </p>
      ) : null}
      <Gate passed={passed}>
        先提交“高频误差升高”的预测，再在 d=7 与 d=24
        揭晓：手指重建误差上升，开环缩短，每维箱宽变粗。数字来自固定示教上的公式，不能当成真机成功率。
      </Gate>
    </LabFrame>
  );
}
