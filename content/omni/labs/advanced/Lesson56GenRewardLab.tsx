"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import { numberFrom, round, stringFrom, useCompletionGate } from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson56GenRewardLab.module.css";

const SIZE = 16;
const PLUS = new Set([7, 8]);

type PredictionId = "same" | "invert" | "tie" | "";

function clamp01(value: number) {
  return Math.min(1, Math.max(0, value));
}

function makeReference() {
  return Array.from({ length: SIZE * SIZE }, (_, index) => {
    const x = index % SIZE;
    const y = Math.floor(index / SIZE);
    const checker = (x + y) % 2 === 0 ? 1 : 0;
    return PLUS.has(x) || PLUS.has(y) ? 1 : checker * 0.35;
  });
}

function boxBlur(image: number[]) {
  return image.map((_, index) => {
    const x = index % SIZE;
    const y = Math.floor(index / SIZE);
    let total = 0;
    let count = 0;
    for (let dy = -1; dy <= 1; dy += 1) {
      for (let dx = -1; dx <= 1; dx += 1) {
        const xx = x + dx;
        const yy = y + dy;
        if (xx >= 0 && xx < SIZE && yy >= 0 && yy < SIZE) {
          total += image[yy * SIZE + xx];
          count += 1;
        }
      }
    }
    return total / count;
  });
}

function blurPasses(image: number[], passes: number) {
  let current = image;
  for (let step = 0; step < passes; step += 1) {
    current = boxBlur(current);
  }
  return current;
}

function shiftImage(image: number[], dx: number) {
  return image.map((_, index) => {
    const x = index % SIZE;
    const y = Math.floor(index / SIZE);
    const sx = x - dx;
    return sx >= 0 && sx < SIZE ? image[y * SIZE + sx] : 0;
  });
}

function meanSquareError(left: number[], right: number[]) {
  return (
    left.reduce((sum, value, index) => {
      const delta = value - right[index];
      return sum + delta * delta;
    }, 0) / left.length
  );
}

function edgeEnergy(image: number[]) {
  let total = 0;
  let count = 0;
  for (let y = 0; y < SIZE; y += 1) {
    for (let x = 0; x < SIZE; x += 1) {
      const index = y * SIZE + x;
      if (x + 1 < SIZE) {
        const gx = image[index + 1] - image[index];
        total += gx * gx;
        count += 1;
      }
      if (y + 1 < SIZE) {
        const gy = image[index + SIZE] - image[index];
        total += gy * gy;
        count += 1;
      }
    }
  }
  return total / count;
}

function preferenceScore(image: number[]) {
  const mean = image.reduce((sum, value) => sum + value, 0) / image.length;
  const variance =
    image.reduce((sum, value) => sum + (value - mean) ** 2, 0) / image.length;
  const grayPenalty = 1 - 4 * mean * (1 - mean);
  return 2.4 * edgeEnergy(image) + 0.9 * variance - 0.35 * grayPenalty;
}

function mixImages(left: number[], right: number[], weight: number) {
  return left.map((value, index) => (1 - weight) * value + weight * right[index]);
}

function kendallTau(leftRanks: number[], rightRanks: number[]) {
  let concordant = 0;
  let discordant = 0;
  const size = leftRanks.length;
  for (let i = 0; i < size; i += 1) {
    for (let j = i + 1; j < size; j += 1) {
      const product =
        (leftRanks[i] - leftRanks[j]) * (rightRanks[i] - rightRanks[j]);
      if (product > 0) concordant += 1;
      else if (product < 0) discordant += 1;
    }
  }
  const pairs = (size * (size - 1)) / 2;
  return { tau: (concordant - discordant) / pairs, concordant, discordant, pairs };
}

function ranksFromScores(scores: number[], higherIsBetter: boolean) {
  const order = scores
    .map((score, index) => ({ score, index }))
    .sort((left, right) =>
      higherIsBetter ? right.score - left.score : left.score - right.score,
    );
  const ranks = Array.from({ length: scores.length }, () => 0);
  order.forEach((item, rank) => {
    ranks[item.index] = rank + 1;
  });
  return ranks;
}

const PREDICTIONS: { value: Exclude<PredictionId, "">; label: string }[] = [
  {
    value: "same",
    label: "L2 更低的那张，偏好分也会更高",
  },
  {
    value: "invert",
    label: "L2 更低的那张过平滑，人更不喜欢",
  },
  {
    value: "tie",
    label: "两把尺子会打出同一名次",
  },
];

export function Lesson56GenRewardLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    blurA: numberFrom(initialState, "blurA", 1, 0, 3),
    shiftB: numberFrom(initialState, "shiftB", 1, 0, 3),
    prediction: stringFrom(initialState, "prediction", "") as PredictionId,
  };
  const [blurA, setBlurA] = useState(Math.round(defaults.blurA));
  const [shiftB, setShiftB] = useState(Math.round(defaults.shiftB));
  const [prediction, setPrediction] = useState<PredictionId>(defaults.prediction);
  const [revealed, setRevealed] = useState(false);

  const calculation = useMemo(() => {
    const reference = makeReference();
    const oversmooth = blurPasses(reference, 1);
    const shifted = shiftImage(reference, 1);
    const imageA = blurPasses(reference, blurA);
    const imageB = shiftImage(reference, shiftB);
    const l2A = meanSquareError(imageA, reference);
    const l2B = meanSquareError(imageB, reference);
    const prefA = preferenceScore(imageA);
    const prefB = preferenceScore(imageB);
    const inverted =
      (l2A < l2B && prefA < prefB) || (l2B < l2A && prefB < prefA);
    const four = [
      oversmooth,
      mixImages(oversmooth, shifted, 0.15),
      mixImages(oversmooth, shifted, 0.4),
      shifted,
    ];
    const fourL2 = four.map((image) => meanSquareError(image, reference));
    const fourPref = four.map((image) => preferenceScore(image));
    const kendall = kendallTau(
      ranksFromScores(fourL2, false),
      ranksFromScores(fourPref, true),
    );
    const lowerL2 = l2A <= l2B ? "A" : "B";
    const higherPref = prefA >= prefB ? "A" : "B";
    return {
      imageA,
      imageB,
      l2A,
      l2B,
      prefA,
      prefB,
      inverted,
      kendall,
      lowerL2,
      higherPref,
    };
  }, [blurA, shiftB]);

  const passed = revealed && prediction === "invert" && calculation.inverted;
  const completion = useMemo(
    () => ({
      lessonId: 56,
      blurA,
      shiftB,
      prediction,
      l2A: round(calculation.l2A, 6),
      l2B: round(calculation.l2B, 6),
      prefA: round(calculation.prefA, 6),
      prefB: round(calculation.prefB, 6),
      inverted: calculation.inverted,
      kendallTau: round(calculation.kendall.tau, 4),
    }),
    [blurA, calculation, prediction, shiftB],
  );
  useCompletionGate(passed, onComplete, completion);

  function applyPreset(kind: "invert" | "agree") {
    setRevealed(false);
    if (kind === "invert") {
      setBlurA(1);
      setShiftB(1);
      return;
    }
    setBlurA(2);
    setShiftB(0);
  }

  return (
    <LabFrame
      lesson="56"
      title="L2 与偏好排序对打"
      description="教学模拟，不是 ImageReward 或 HPSv2 的权重输出。先选预测，再揭晓两张图的像素 L2 与教学偏好分。验收：必须造出 L2 更低的那张过平滑、偏好分也更低。"
    >
      <div className={styles.workspace}>
        <form
          className={styles.controls}
          onSubmit={(event) => event.preventDefault()}
        >
          <h3>候选控制台</h3>
          <p>
            参考图是高对比十字加棋盘。A 做盒滤波，B 做平移。像素错位会把 L2
            抬高，边却还在。
          </p>
          <label>
            <span>
              A 盒滤波次数 <output>{blurA}</output>
            </span>
            <input
              type="range"
              min={0}
              max={3}
              step={1}
              value={blurA}
              onChange={(event) => {
                setBlurA(Number(event.target.value));
                setRevealed(false);
              }}
            />
          </label>
          <label>
            <span>
              B 平移像素 <output>{shiftB}</output>
            </span>
            <input
              type="range"
              min={0}
              max={3}
              step={1}
              value={shiftB}
              onChange={(event) => {
                setShiftB(Number(event.target.value));
                setRevealed(false);
              }}
            />
          </label>
          <div className={styles.presets}>
            <button type="button" onClick={() => applyPreset("invert")}>
              预置反向对
            </button>
            <button type="button" onClick={() => applyPreset("agree")}>
              预置同序对
            </button>
          </div>
        </form>

        <div className={styles.stage}>
          <div className={styles.pair} aria-label="两张候选图">
            <PixelPanel
              title="图 A"
              hint={blurA === 0 ? "未滤波" : `${blurA} 次盒滤波`}
              pixels={calculation.imageA}
              l2={calculation.l2A}
              pref={calculation.prefA}
              revealed={revealed}
              mark={
                revealed && calculation.lowerL2 === "A" ? "L2 更低" : undefined
              }
            />
            <PixelPanel
              title="图 B"
              hint={shiftB === 0 ? "未平移" : `右移 ${shiftB} px`}
              pixels={calculation.imageB}
              l2={calculation.l2B}
              pref={calculation.prefB}
              revealed={revealed}
              mark={
                revealed && calculation.higherPref === "B"
                  ? "偏好更高"
                  : undefined
              }
            />
          </div>

          <dl className={styles.metrics}>
            <div>
              <dt>更低 L2</dt>
              <dd>{revealed ? `图 ${calculation.lowerL2}` : "—"}</dd>
            </div>
            <div>
              <dt>更高偏好</dt>
              <dd>{revealed ? `图 ${calculation.higherPref}` : "—"}</dd>
            </div>
            <div>
              <dt>本对是否反向</dt>
              <dd>
                {revealed ? (calculation.inverted ? "反向" : "同序") : "—"}
              </dd>
            </div>
            <div>
              <dt>四档 Kendall τ</dt>
              <dd>
                {revealed ? calculation.kendall.tau.toFixed(1) : "—"}
              </dd>
            </div>
          </dl>

          <div className={styles.predict}>
            <fieldset>
              <legend>先预测，再揭晓数字</legend>
              {PREDICTIONS.map((item) => (
                <label key={item.value}>
                  <input
                    type="radio"
                    name="lesson56-prediction"
                    value={item.value}
                    checked={prediction === item.value}
                    onChange={() => {
                      setPrediction(item.value);
                      setRevealed(false);
                    }}
                  />
                  <span>{item.label}</span>
                </label>
              ))}
            </fieldset>
            <div className={styles.actions}>
              <button
                type="button"
                className={styles.reset}
                onClick={() => {
                  setBlurA(1);
                  setShiftB(1);
                  setPrediction("");
                  setRevealed(false);
                }}
              >
                重置
              </button>
              <button
                type="button"
                className={styles.run}
                disabled={!prediction}
                onClick={() => setRevealed(true)}
              >
                揭晓分数
              </button>
            </div>
          </div>
        </div>
      </div>
      <Gate passed={passed}>
        {passed
          ? "已造出 L2 与偏好反向：过平滑那张像素误差更低，教学偏好分也更低。四档夹具的 Kendall τ 为 -1。"
          : "先选择“L2 更低的那张过平滑，人更不喜欢”，把 A 设成滤波、B 设成平移，再揭晓。两张都锐利或都糊时，尺子会回到同序。"}
      </Gate>
    </LabFrame>
  );
}

function PixelPanel({
  title,
  hint,
  pixels,
  l2,
  pref,
  revealed,
  mark,
}: {
  title: string;
  hint: string;
  pixels: number[];
  l2: number;
  pref: number;
  revealed: boolean;
  mark?: string;
}) {
  return (
    <article className={styles.panel}>
      <header>
        <b>{title}</b>
        <small>{hint}</small>
        {mark ? <em>{mark}</em> : null}
      </header>
      <div
        className={styles.grid}
        role="img"
        aria-label={`${title} 的 ${SIZE} 乘 ${SIZE} 像素`}
      >
        {pixels.map((value, index) => (
          <span
            key={`${title}-${index}`}
            style={{
              background: `rgb(${Math.round(clamp01(value) * 255)}, ${Math.round(
                clamp01(value) * 220 + 20,
              )}, ${Math.round(clamp01(value) * 200 + 18)})`,
            }}
          />
        ))}
      </div>
      <p>
        L2 {revealed ? l2.toFixed(4) : "未揭晓"} · 偏好{" "}
        {revealed ? pref.toFixed(3) : "未揭晓"}
      </p>
    </article>
  );
}
