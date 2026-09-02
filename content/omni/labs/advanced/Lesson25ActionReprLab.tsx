"use client";

import { useEffect, useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import { numberFrom, round, stringFrom, useCompletionGate } from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson25ActionReprLab.module.css";

type ReprMode = "bin" | "l1" | "chunk" | "dct";

const STEPS = 64;
const DIMS = 7;
const CONTROL_HZ = 20;
const WIGGLE_HZ = 6;
const DCT_SCALE = 10;
const DIM_LABELS = ["末端 x", "末端 y", "末端 z", "滚转", "俯仰", "偏航", "夹爪"];

function clamp(value: number, low = -1, high = 1) {
  return Math.min(high, Math.max(low, value));
}

function makeDemo(): number[][] {
  return Array.from({ length: STEPS }, (_, index) => {
    const phase = index / (STEPS - 1);
    const time = index / CONTROL_HZ;
    const reach = -0.8 + 1.6 * phase;
    const wiggle = 0.35 + 0.08 * Math.sin(2 * Math.PI * WIGGLE_HZ * time);
    const gripper = -0.95 + 1.9 / (1 + Math.exp(-14 * (phase - 0.72)));
    return [
      clamp(wiggle),
      clamp(reach),
      clamp(0.18 * Math.sin(2 * Math.PI * 0.35 * time)),
      clamp(0.12 * Math.sin(2 * Math.PI * 0.28 * time)),
      clamp(-0.22 + 0.4 * phase),
      clamp(0.15 * Math.cos(2 * Math.PI * 0.22 * time)),
      clamp(gripper),
    ];
  });
}

function uniformBin(value: number, bins: number) {
  const width = 2 / bins;
  const clipped = clamp(value);
  const index = Math.min(bins - 1, Math.floor((clipped + 1) / width));
  return -1 + (index + 0.5) * width;
}

function dct(signal: number[]) {
  const length = signal.length;
  return signal.map((_, freq) => {
    const scale = freq === 0 ? Math.sqrt(1 / length) : Math.sqrt(2 / length);
    let total = 0;
    for (let time = 0; time < length; time += 1) {
      total += signal[time] * Math.cos((Math.PI * (time + 0.5) * freq) / length);
    }
    return scale * total;
  });
}

function idct(coeffs: number[]) {
  const length = coeffs.length;
  return coeffs.map((_, time) => {
    let total = 0;
    for (let freq = 0; freq < length; freq += 1) {
      const scale = freq === 0 ? Math.sqrt(1 / length) : Math.sqrt(2 / length);
      total +=
        scale *
        coeffs[freq] *
        Math.cos((Math.PI * (time + 0.5) * freq) / length);
    }
    return total;
  });
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

function reconstructionL2(original: number[][], reconstructed: number[][]) {
  let total = 0;
  let count = 0;
  for (let time = 0; time < original.length; time += 1) {
    for (let dim = 0; dim < DIMS; dim += 1) {
      const delta = original[time][dim] - reconstructed[time][dim];
      total += delta * delta;
      count += 1;
    }
  }
  return Math.sqrt(total / count);
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

export function Lesson25ActionReprLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    mode: stringFrom(initialState, "mode", "bin") as ReprMode,
    bins: numberFrom(initialState, "bins", 8, 2, 256),
    chunk: numberFrom(initialState, "chunk", 8, 1, 32),
    keep: numberFrom(initialState, "keep", 4, 1, 16),
    dim: numberFrom(initialState, "dim", 0, 0, 6),
    predBin: stringFrom(initialState, "predBin", ""),
    predChunk: stringFrom(initialState, "predChunk", ""),
  };
  const [mode, setMode] = useState<ReprMode>(
    ["bin", "l1", "chunk", "dct"].includes(defaults.mode)
      ? defaults.mode
      : "bin",
  );
  const [bins, setBins] = useState(
    [2, 4, 8, 16, 32, 64, 128, 256].includes(defaults.bins)
      ? defaults.bins
      : 8,
  );
  const [chunk, setChunk] = useState(
    [1, 2, 4, 8, 16, 32].includes(defaults.chunk) ? defaults.chunk : 8,
  );
  const [keep, setKeep] = useState(Math.min(16, Math.max(1, defaults.keep)));
  const [dim, setDim] = useState(Math.min(6, Math.max(0, Math.round(defaults.dim))));
  const [predBin, setPredBin] = useState(defaults.predBin);
  const [predChunk, setPredChunk] = useState(defaults.predChunk);
  const [ran, setRan] = useState(false);
  const [visitedBin2, setVisitedBin2] = useState(false);
  const [visitedChunk8, setVisitedChunk8] = useState(false);
  const [visitedChunk16, setVisitedChunk16] = useState(false);

  const demo = useMemo(() => makeDemo(), []);

  const calculation = useMemo(() => {
    let reconstructed = demo.map((row) => row.slice());
    let vocab = 0;
    let tokens = DIMS * STEPS;
    let serial = DIMS;
    let dctTokens = 0;

    if (mode === "bin") {
      reconstructed = demo.map((row) =>
        row.map((value) => uniformBin(value, bins)),
      );
      vocab = bins;
      tokens = DIMS * STEPS;
      serial = DIMS;
    } else if (mode === "l1") {
      vocab = 0;
      tokens = DIMS * chunk;
      serial = 1;
    } else if (mode === "chunk") {
      vocab = 0;
      tokens = DIMS * chunk;
      serial = DIMS * chunk;
    } else {
      reconstructed = demo.map(() => Array.from({ length: DIMS }, () => 0));
      for (let axis = 0; axis < DIMS; axis += 1) {
        const signal = demo.map((row) => row[axis]);
        const coeffs = dct(signal).map((coeff, freq) => {
          if (freq >= keep) return 0;
          const quantized = Math.round(DCT_SCALE * coeff) / DCT_SCALE;
          if (quantized !== 0) dctTokens += 1;
          return quantized;
        });
        const restored = idct(coeffs);
        for (let time = 0; time < STEPS; time += 1) {
          reconstructed[time][axis] = clamp(restored[time]);
        }
      }
      vocab = 1024;
      tokens = dctTokens;
      serial = Math.max(1, dctTokens);
    }

    const originalWiggle = residualRms(demo.map((row) => row[0]));
    const reconWiggle = residualRms(reconstructed.map((row) => row[0]));
    const highFreqRemain =
      originalWiggle < 1e-9 ? 0 : reconWiggle / originalWiggle;

    return {
      reconstructed,
      vocab,
      tokens,
      serial,
      l2: reconstructionL2(demo, reconstructed),
      openLoop: chunk / CONTROL_HZ,
      highFreqRemain,
    };
  }, [bins, chunk, demo, keep, mode]);

  useEffect(() => {
    if (!ran) return;
    if (mode === "bin" && bins === 2 && calculation.highFreqRemain < 0.12) {
      setVisitedBin2(true);
    }
    if (mode === "chunk" && chunk === 8) setVisitedChunk8(true);
    if (mode === "chunk" && chunk === 16) setVisitedChunk16(true);
  }, [bins, calculation.highFreqRemain, chunk, mode, ran]);

  const canRun = predBin !== "" && predChunk !== "";
  const passed =
    ran &&
    predBin === "vanish" &&
    predChunk === "linear" &&
    visitedBin2 &&
    visitedChunk8 &&
    visitedChunk16;

  const completion = useMemo(
    () => ({
      lessonId: 25,
      mode,
      bins,
      chunk,
      keep,
      dim,
      reconL2: round(calculation.l2, 5),
      vocab: calculation.vocab,
      tokens: calculation.tokens,
      serial: calculation.serial,
      openLoop: round(calculation.openLoop, 4),
      highFreqRemain: round(calculation.highFreqRemain, 5),
      visitedBin2,
      visitedChunk8,
      visitedChunk16,
    }),
    [
      bins,
      calculation.highFreqRemain,
      calculation.l2,
      calculation.openLoop,
      calculation.serial,
      calculation.tokens,
      calculation.vocab,
      chunk,
      dim,
      keep,
      mode,
      visitedBin2,
      visitedChunk8,
      visitedChunk16,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function invalidate() {
    setRan(false);
  }

  function reset() {
    setMode("bin");
    setBins(8);
    setChunk(8);
    setKeep(4);
    setDim(0);
    setPredBin("");
    setPredChunk("");
    setRan(false);
    setVisitedBin2(false);
    setVisitedChunk8(false);
    setVisitedChunk16(false);
  }

  const originalLine = toPolyline(
    demo.map((row) => row[dim]),
    640,
    148,
  );
  const reconLine = toPolyline(
    calculation.reconstructed.map((row) => row[dim]),
    640,
    148,
  );

  const wrongPrediction =
    ran && (predBin !== "vanish" || predChunk !== "linear");

  return (
    <LabFrame
      lesson="25"
      title="四种动作表示怎么重建一条 7 维示教"
      description="教学模拟，不是模型输出。一条固定示教含 6 Hz 来回。先预测分箱和分块的后果，再切换表示，核对照重建误差、词表、串行步数和开环时长。"
    >
      <div className={styles.workspace}>
        <form
          className={styles.controls}
          onSubmit={(event) => event.preventDefault()}
        >
          <h3>表示控制台</h3>
          <fieldset className={styles.modeField}>
            <legend>动作表示</legend>
            <div>
              {(
                [
                  ["bin", "均匀分箱"],
                  ["l1", "连续 L1"],
                  ["chunk", "动作分块"],
                  ["dct", "DCT 保留"],
                ] as const
              ).map(([value, label]) => (
                <label key={value}>
                  <input
                    type="radio"
                    name="repr-mode"
                    value={value}
                    checked={mode === value}
                    onChange={() => {
                      setMode(value);
                      invalidate();
                    }}
                  />
                  <span>{label}</span>
                </label>
              ))}
            </div>
          </fieldset>
          <label>
            <span>
              每维箱数 B <output>{bins}</output>
            </span>
            <input
              type="range"
              min="1"
              max="8"
              step="1"
              value={Math.log2(bins)}
              onChange={(event) => {
                setBins(2 ** Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <label>
            <span>
              分块长度 H <output>{chunk}</output>
            </span>
            <input
              type="range"
              min="0"
              max="5"
              step="1"
              value={[1, 2, 4, 8, 16, 32].indexOf(chunk)}
              onChange={(event) => {
                const options = [1, 2, 4, 8, 16, 32];
                setChunk(options[Number(event.target.value)] ?? 8);
                invalidate();
              }}
            />
          </label>
          <label>
            <span>
              DCT 保留系数 <output>{keep}</output>
            </span>
            <input
              type="range"
              min="1"
              max="16"
              step="1"
              value={keep}
              onChange={(event) => {
                setKeep(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <label>
            <span>
              查看维度 <output>{DIM_LABELS[dim]}</output>
            </span>
            <input
              type="range"
              min="0"
              max="6"
              step="1"
              value={dim}
              onChange={(event) => setDim(Number(event.target.value))}
            />
          </label>
          <p className={styles.note}>
            控制频率固定 20 Hz。开环时长 T = H / f。末端 x 含 6 Hz 来回，幅值落在
            B=2 的同一箱内。
          </p>
        </form>

        <div className={styles.stage}>
          <div className={styles.formula}>
            <span>量化误差上界 1/B；开环 T = H / f</span>
            <strong>
              1/{bins} = {(1 / bins).toFixed(4)}；{chunk}/{CONTROL_HZ} ={" "}
              {(chunk / CONTROL_HZ).toFixed(2)} s
            </strong>
          </div>
          <div className={styles.plot}>
            <header>
              <b>{DIM_LABELS[dim]} 轨迹</b>
              <span>{ran ? "原轨迹 / 重建" : "只显示原轨迹，运行后揭晓重建"}</span>
            </header>
            <svg viewBox="0 0 640 148" role="img" aria-label="示教与重建轨迹">
              <line
                x1="0"
                y1="74"
                x2="640"
                y2="74"
                stroke="currentColor"
                strokeOpacity="0.18"
              />
              <polyline
                fill="none"
                stroke="#314338"
                strokeWidth="2"
                points={originalLine}
              />
              {ran ? (
                <polyline
                  fill="none"
                  stroke="#176f48"
                  strokeWidth="2"
                  strokeDasharray="5 4"
                  points={reconLine}
                />
              ) : null}
            </svg>
            <div className={styles.legend}>
              <span className={styles.orig}>
                <i />
                原示教
              </span>
              <span className={styles.recon}>
                <i />
                {ran ? "当前表示重建" : "未揭晓"}
              </span>
            </div>
          </div>
          <dl className={styles.metrics}>
            <div>
              <dt>重建 L2</dt>
              <dd>{ran ? calculation.l2.toFixed(4) : "—"}</dd>
            </div>
            <div>
              <dt>词表 / token 数</dt>
              <dd>
                {ran ? `${calculation.vocab} / ${calculation.tokens}` : "—"}
              </dd>
            </div>
            <div>
              <dt>串行步数</dt>
              <dd>{ran ? calculation.serial : "—"}</dd>
            </div>
            <div>
              <dt>开环时长 / 高频残留</dt>
              <dd>
                {ran
                  ? `${calculation.openLoop.toFixed(2)} s / ${calculation.highFreqRemain.toFixed(2)}`
                  : "—"}
              </dd>
            </div>
          </dl>
        </div>
      </div>

      <div className={styles.predict}>
        <div className={styles.predictQuestions}>
          <fieldset>
            <legend>先预测：把均匀分箱调到 B=2，6 Hz 来回会怎样？</legend>
            {[
              ["vanish", "几乎消失，掉进同一箱"],
              ["keep", "幅值基本保留"],
              ["noise", "变成更密的抖动"],
            ].map(([value, label]) => (
              <label key={value}>
                <input
                  type="radio"
                  name="pred-bin"
                  value={value}
                  checked={predBin === value}
                  onChange={() => {
                    setPredBin(value);
                    invalidate();
                  }}
                />
                <span>{label}</span>
              </label>
            ))}
          </fieldset>
          <fieldset>
            <legend>先预测：分块模式下把 H 从 8 加到 16，开环时长会怎样？</legend>
            {[
              ["linear", "按 H/f 线性加倍"],
              ["flat", "时长不变，只变词表"],
              ["square", "按 H 的平方变长"],
            ].map(([value, label]) => (
              <label key={value}>
                <input
                  type="radio"
                  name="pred-chunk"
                  value={value}
                  checked={predChunk === value}
                  onChange={() => {
                    setPredChunk(value);
                    invalidate();
                  }}
                />
                <span>{label}</span>
              </label>
            ))}
          </fieldset>
        </div>
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
          B=2 的箱宽是 1，6 Hz 来回的幅值只有 0.08、中心在 0.35，整段都落在正半轴那一箱。开环时长是 H 除以 20 Hz，H 加倍则时长加倍。
        </p>
      ) : null}
      {ran && predBin === "vanish" && predChunk === "linear" && !passed ? (
        <p className={styles.feedback}>
          预测正确。请切到均匀分箱并设 B=2 看高频残留，再切到动作分块分别运行 H=8 与 H=16。
        </p>
      ) : null}
      <Gate passed={passed}>
        先提交两项预测，再触发 B=2 高频来回消失，以及 H=8 与 H=16 的开环时长按 H/f
        加倍。数字来自固定示教上的公式计算，不能当成真机成功率。
      </Gate>
    </LabFrame>
  );
}
