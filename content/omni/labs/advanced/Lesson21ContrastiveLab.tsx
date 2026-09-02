"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import {
  numberFrom,
  round,
  sigmoid,
  softmax,
  stringFrom,
  useCompletionGate,
} from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson21ContrastiveLab.module.css";

const IMAGE_LABELS = ["图:猫", "图:狗", "图:车", "图:树"];
const TEXT_LABELS = ["文:橘猫", "文:金毛", "文:轿车", "文:松树"];
const ALIGNED_SIMILARITY = [
  [0.92, 0.11, 0.08, 0.05],
  [0.1, 0.88, 0.14, 0.07],
  [0.06, 0.12, 0.9, 0.09],
  [0.04, 0.08, 0.1, 0.86],
];
const SHUFFLE_PERM = [1, 2, 3, 0];
const SIGLIP_T = 10;
const SIGLIP_BIAS = -10;

function permuteColumns(matrix: number[][], permutation: number[]) {
  return matrix.map((row) => permutation.map((index) => row[index]));
}

function infonce(similarity: number[][], temperature: number) {
  const size = similarity.length;
  const scaled = similarity.map((row) =>
    row.map((value) => value / temperature),
  );
  const rowProbabilities = scaled.map((row) => softmax(row));
  const colProbabilities = Array.from({ length: size }, (_, column) =>
    softmax(scaled.map((row) => row[column])),
  );
  const imageToText =
    rowProbabilities.reduce(
      (sum, row, index) => sum - Math.log(row[index]),
      0,
    ) / size;
  const textToImage =
    colProbabilities.reduce(
      (sum, column, index) => sum - Math.log(column[index]),
      0,
    ) / size;
  const positiveProbabilities = rowProbabilities.map((row, index) => row[index]);
  return {
    imageToText,
    textToImage,
    loss: 0.5 * (imageToText + textToImage),
    rowProbabilities,
    positiveProbabilities,
    peak: Math.max(...positiveProbabilities),
  };
}

function sigmoidLoss(similarity: number[][]) {
  const size = similarity.length;
  let total = 0;
  for (let row = 0; row < size; row += 1) {
    for (let column = 0; column < size; column += 1) {
      const label = row === column ? 1 : -1;
      const logit = SIGLIP_T * similarity[row][column] + SIGLIP_BIAS;
      total += Math.log(sigmoid(label * logit));
    }
  }
  return -total / size;
}

function near(value: number, target: number) {
  return Math.abs(value - target) < 1e-8;
}

export function Lesson21ContrastiveLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    temperature: numberFrom(initialState, "temperature", 0.07, 0.01, 0.2),
    shuffled: stringFrom(initialState, "pairing", "aligned") === "shuffled",
    prediction: stringFrom(initialState, "prediction", ""),
  };
  const [temperature, setTemperature] = useState(defaults.temperature);
  const [shuffled, setShuffled] = useState(defaults.shuffled);
  const [prediction, setPrediction] = useState(defaults.prediction);
  const [revealed, setRevealed] = useState(false);
  const [alignedLoss, setAlignedLoss] = useState<number | null>(null);
  const [shuffledLoss, setShuffledLoss] = useState<number | null>(null);
  const [peakLow, setPeakLow] = useState<number | null>(null);
  const [peakHigh, setPeakHigh] = useState<number | null>(null);

  const similarity = useMemo(
    () =>
      shuffled
        ? permuteColumns(ALIGNED_SIMILARITY, SHUFFLE_PERM)
        : ALIGNED_SIMILARITY,
    [shuffled],
  );
  const contrastive = useMemo(
    () => infonce(similarity, temperature),
    [similarity, temperature],
  );
  const pairLoss = useMemo(() => sigmoidLoss(similarity), [similarity]);

  const recorded = useMemo(() => {
    if (!revealed) {
      return { alignedLoss, shuffledLoss, peakLow, peakHigh };
    }
    return {
      alignedLoss: shuffled ? alignedLoss : contrastive.loss,
      shuffledLoss: shuffled ? contrastive.loss : shuffledLoss,
      peakLow:
        !shuffled && near(temperature, 0.01) ? contrastive.peak : peakLow,
      peakHigh:
        !shuffled && near(temperature, 0.2) ? contrastive.peak : peakHigh,
    };
  }, [
    alignedLoss,
    contrastive.loss,
    contrastive.peak,
    peakHigh,
    peakLow,
    revealed,
    shuffled,
    shuffledLoss,
    temperature,
  ]);

  const lossOrderOk =
    recorded.alignedLoss !== null &&
    recorded.shuffledLoss !== null &&
    recorded.shuffledLoss + 1e-12 >= recorded.alignedLoss;
  const peakOrderOk =
    recorded.peakLow !== null &&
    recorded.peakHigh !== null &&
    recorded.peakLow > recorded.peakHigh + 1e-8;
  const passed =
    revealed && prediction === "up" && lossOrderOk && peakOrderOk;

  const completion = useMemo(
    () => ({
      lessonId: 21,
      temperature,
      pairing: shuffled ? "shuffled" : "aligned",
      infonce: round(contrastive.loss, 6),
      sigmoid: round(pairLoss, 6),
      positivePeak: round(contrastive.peak, 6),
      alignedLoss: recorded.alignedLoss,
      shuffledLoss: recorded.shuffledLoss,
      peakAt001: recorded.peakLow,
      peakAt02: recorded.peakHigh,
    }),
    [
      contrastive.loss,
      contrastive.peak,
      pairLoss,
      recorded.alignedLoss,
      recorded.peakHigh,
      recorded.peakLow,
      recorded.shuffledLoss,
      shuffled,
      temperature,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function rememberCurrent(nextShuffled: boolean, nextTemperature: number) {
    const nextSimilarity = nextShuffled
      ? permuteColumns(ALIGNED_SIMILARITY, SHUFFLE_PERM)
      : ALIGNED_SIMILARITY;
    const nextContrastive = infonce(nextSimilarity, nextTemperature);
    if (nextShuffled) {
      setShuffledLoss(nextContrastive.loss);
    } else {
      setAlignedLoss(nextContrastive.loss);
      if (near(nextTemperature, 0.01)) setPeakLow(nextContrastive.peak);
      if (near(nextTemperature, 0.2)) setPeakHigh(nextContrastive.peak);
    }
  }

  function reveal() {
    setRevealed(true);
    rememberCurrent(shuffled, temperature);
  }

  function reset() {
    setTemperature(defaults.temperature);
    setShuffled(false);
    setPrediction("");
    setRevealed(false);
    setAlignedLoss(null);
    setShuffledLoss(null);
    setPeakLow(null);
    setPeakHigh(null);
  }

  return (
    <LabFrame
      lesson="21"
      title="图文相似度矩阵"
      description="教学模拟，不是模型输出。先预测打乱配对后 InfoNCE 升还是降，再调温度、切换配对，核对损失和正对概率。"
    >
      <div className={styles.workspace}>
        <form
          className={styles.controls}
          onSubmit={(event) => event.preventDefault()}
        >
          <h3>对比控制台</h3>
          <p>
            对角是 batch 里声称的正对。打乱只重排文字列，图像行不动。
          </p>
          <label>
            <span>
              温度 τ <output>{temperature.toFixed(2)}</output>
            </span>
            <input
              type="range"
              min="0.01"
              max="0.2"
              step="0.01"
              value={temperature}
              onChange={(event) => {
                const value = Number(event.target.value);
                setTemperature(value);
                if (revealed) rememberCurrent(shuffled, value);
              }}
            />
          </label>
          <label className={styles.shuffle}>
            <input
              type="checkbox"
              checked={shuffled}
              onChange={(event) => {
                const value = event.target.checked;
                setShuffled(value);
                if (revealed) rememberCurrent(value, temperature);
              }}
            />
            <span>打乱文字配对</span>
          </label>
        </form>

        <div className={styles.stage}>
          <div className={styles.formula}>
            <span>
              s_ij / τ ；sigmoid 用 t=10, b=-10
            </span>
            <strong>
              {shuffled ? "配对已打乱" : "配对对齐"} · τ ={" "}
              {temperature.toFixed(2)}
            </strong>
          </div>
          <div className={styles.board} aria-label="图文相似度矩阵">
            <div className={styles.corner} />
            {TEXT_LABELS.map((label) => (
              <div className={styles.head} key={label}>
                {shuffled
                  ? TEXT_LABELS[SHUFFLE_PERM[TEXT_LABELS.indexOf(label)]]
                  : label}
              </div>
            ))}
            {IMAGE_LABELS.map((image, row) => (
              <div key={image} style={{ display: "contents" }}>
                <div className={styles.rowHead}>{image}</div>
                {similarity[row].map((value, column) => {
                  const probability = contrastive.rowProbabilities[row][column];
                  const isPositive = row === column;
                  return (
                    <div
                      className={`${styles.cell} ${
                        isPositive ? styles.cellPositive : ""
                      }`}
                      key={`${row}-${column}`}
                      style={{ "--fill": value } as React.CSSProperties}
                    >
                      <b>{value.toFixed(2)}</b>
                      <small>
                        {revealed ? `p=${probability.toFixed(2)}` : "p=?"}
                      </small>
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
          <dl className={styles.metrics}>
            <div>
              <dt>InfoNCE</dt>
              <dd>{revealed ? contrastive.loss.toFixed(3) : "—"}</dd>
            </div>
            <div>
              <dt>sigmoid 损失</dt>
              <dd>{revealed ? pairLoss.toFixed(3) : "—"}</dd>
            </div>
            <div>
              <dt>正对概率峰值</dt>
              <dd>{revealed ? contrastive.peak.toFixed(3) : "—"}</dd>
            </div>
            <div>
              <dt>对角均值</dt>
              <dd>
                {(
                  similarity.reduce((sum, row, index) => sum + row[index], 0) /
                  similarity.length
                ).toFixed(2)}
              </dd>
            </div>
          </dl>
          <ul className={styles.evidence}>
            <li className={recorded.alignedLoss !== null ? styles.ready : ""}>
              对齐 InfoNCE：
              {recorded.alignedLoss === null
                ? "未记录"
                : recorded.alignedLoss.toFixed(3)}
            </li>
            <li className={recorded.shuffledLoss !== null ? styles.ready : ""}>
              打乱 InfoNCE：
              {recorded.shuffledLoss === null
                ? "未记录"
                : recorded.shuffledLoss.toFixed(3)}
            </li>
            <li className={recorded.peakLow !== null ? styles.ready : ""}>
              τ=0.01 峰值：
              {recorded.peakLow === null ? "未记录" : recorded.peakLow.toFixed(3)}
            </li>
            <li className={recorded.peakHigh !== null ? styles.ready : ""}>
              τ=0.20 峰值：
              {recorded.peakHigh === null
                ? "未记录"
                : recorded.peakHigh.toFixed(3)}
            </li>
          </ul>
        </div>
      </div>

      <div className={styles.predict}>
        <fieldset>
          <legend>先预测：打乱配对之后，InfoNCE 会怎样？</legend>
          {[
            ["up", "升高"],
            ["down", "降低"],
            ["same", "几乎不变"],
          ].map(([value, label]) => (
            <label key={value}>
              <input
                type="radio"
                name="clip-prediction"
                value={value}
                checked={prediction === value}
                onChange={() => {
                  setPrediction(value);
                  setRevealed(false);
                  setAlignedLoss(null);
                  setShuffledLoss(null);
                  setPeakLow(null);
                  setPeakHigh(null);
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
            disabled={!prediction}
            onClick={reveal}
          >
            揭晓损失
          </button>
        </div>
      </div>
      {revealed && prediction !== "up" && (
        <p className={styles.feedback}>
          对角不再是原配对时，softmax 分母里的正对变小，负对变大，交叉熵应升高。
        </p>
      )}
      {revealed && prediction === "up" && !passed && (
        <p className={styles.feedback}>
          预测方向对了。请分别揭晓对齐与打乱，并把温度滑到 0.01 和 0.20（不要打乱）记录正对峰值。
        </p>
      )}
      <Gate passed={passed}>
        打乱后 InfoNCE 不得低于对齐时；温度从 0.01 调到 0.2，正对概率峰值必须下降。数字来自固定 4×4
        教学矩阵，不是公开 CLIP 权重。
      </Gate>
    </LabFrame>
  );
}
