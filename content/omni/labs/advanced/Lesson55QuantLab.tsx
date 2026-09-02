"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import { numberFrom, round, stringFrom, useCompletionGate } from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson55QuantLab.module.css";

type Bits = 8 | 4;
type Prediction = "text" | "action" | "all" | "none" | "";

const HIDDEN = [1.0, 0.5, -0.25, 0.125];
const TEXT_LABEL = 0;
const ACTION_BINS = 8;
const ACTION_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"] as const;
const ACTION_RANGES: readonly [number, number][] = [
  [-1, 1],
  [-1, 1],
  [-1, 1],
  [-1, 1],
  [-1, 1],
  [-1, 1],
  [0, 1],
];
const W_TEXT = [
  [2.4, 1.2, -0.4, 0.3],
  [0.8, 0.4, 0.2, 0.1],
  [0.1, -0.2, 0.3, 0.05],
  [-0.5, 0.1, 0.4, -0.2],
];
const W_VIS = [
  [0.9, 0.2, -0.1, 0.15],
  [0.1, 0.8, 0.25, -0.05],
  [-0.2, 0.15, 0.7, 0.3],
  [0.25, -0.1, 0.05, 0.85],
];
const W_ACT = [
  [0.08, -0.1, 0.04, 0.16],
  [0.2, -0.8, 0.1, 0],
  [0, 0, 0, 0],
  [0.05, 0.1, 0, 0],
  [-0.4, -0.1, 0.2, 0],
  [0.7, 0.2, 0, 0],
  [0.2, 0.2, 0, 0],
];

const PREDICTION_OPTIONS: { value: Exclude<Prediction, "">; label: string }[] = [
  { value: "text", label: "文本 top-1 先跳类，动作 bin 还在原箱" },
  { value: "action", label: "动作 bin 先跳类，文本 top-1 仍不变" },
  { value: "all", label: "文本、视觉粗类和动作 bin 一起跳" },
  { value: "none", label: "8 bit 与 4 bit 都不会跳类" },
];

function halfUp(value: number) {
  return Math.floor(value + 0.5);
}

function qmax(bits: number) {
  return 2 ** (bits - 1) - 1;
}

function quantizeTensor(values: number[], bits: number) {
  const peak = Math.max(...values.map((value) => Math.abs(value)));
  const limit = qmax(bits);
  const scale = peak === 0 ? 1 : peak / limit;
  return values.map((value) => {
    const code = Math.max(-limit, Math.min(limit, halfUp(value / scale)));
    return code * scale;
  });
}

function quantizeRows(matrix: number[][], bits: number) {
  return matrix.map((row) => quantizeTensor(row, bits));
}

function dot(left: number[], right: number[]) {
  return left.reduce((sum, value, index) => sum + value * right[index], 0);
}

function matvec(matrix: number[][], vector: number[]) {
  return matrix.map((row) => dot(row, vector));
}

function softmax(logits: number[]) {
  const peak = Math.max(...logits);
  const weights = logits.map((logit) => Math.exp(logit - peak));
  const total = weights.reduce((sum, value) => sum + value, 0);
  return weights.map((value) => value / total);
}

function uniformBin(value: number, low: number, high: number, bins: number) {
  if (value <= low) return 0;
  if (value >= high) return bins - 1;
  const width = (high - low) / bins;
  return Math.min(bins - 1, Math.floor((value - low) / width));
}

function mse(left: number[], right: number[]) {
  return (
    left.reduce((sum, value, index) => {
      const delta = value - right[index];
      return sum + delta * delta;
    }, 0) / left.length
  );
}

function l2(left: number[], right: number[]) {
  return Math.sqrt(
    left.reduce((sum, value, index) => {
      const delta = value - right[index];
      return sum + delta * delta;
    }, 0),
  );
}

function flatten(matrix: number[][]) {
  return matrix.flat();
}

function evaluate(bits: Bits | "fp") {
  const weightText = bits === "fp" ? W_TEXT : quantizeRows(W_TEXT, bits);
  const weightVis = bits === "fp" ? W_VIS : quantizeRows(W_VIS, bits);
  const weightAct = bits === "fp" ? W_ACT : quantizeRows(W_ACT, bits);
  const textLogits = matvec(weightText, HIDDEN);
  const textProbs = softmax(textLogits);
  const visTrue = matvec(W_VIS, HIDDEN);
  const visHat = matvec(weightVis, HIDDEN);
  const actions = matvec(weightAct, HIDDEN);
  const bins = actions.map((value, index) =>
    uniformBin(value, ACTION_RANGES[index][0], ACTION_RANGES[index][1], ACTION_BINS),
  );
  const ordered = [...textLogits].sort((left, right) => right - left);
  return {
    textLogits,
    textProbs,
    textTop1: textLogits.indexOf(Math.max(...textLogits)),
    textCe: -Math.log(Math.max(textProbs[TEXT_LABEL], 1e-12)),
    textMargin: ordered[0] - ordered[1],
    visL2: l2(visHat, visTrue),
    visMse: mse(visHat, visTrue),
    weightMseText: mse(flatten(W_TEXT), flatten(weightText)),
    weightMseVis: mse(flatten(W_VIS), flatten(weightVis)),
    weightMseAct: mse(flatten(W_ACT), flatten(weightAct)),
    actions,
    bins,
    visClass: visHat.indexOf(Math.max(...visHat)),
    visClassTrue: visTrue.indexOf(Math.max(...visTrue)),
  };
}

export function Lesson55QuantLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    bits: numberFrom(initialState, "bits", 8, 4, 8) as Bits,
    prediction: stringFrom(initialState, "prediction", "") as Prediction,
  };
  const [bits, setBits] = useState<Bits>(defaults.bits === 4 ? 4 : 8);
  const [prediction, setPrediction] = useState<Prediction>(
    defaults.prediction === "text" ||
      defaults.prediction === "action" ||
      defaults.prediction === "all" ||
      defaults.prediction === "none"
      ? defaults.prediction
      : "",
  );
  const [ran, setRan] = useState(false);
  const [sawEightStable, setSawEightStable] = useState(false);
  const [sawFourJump, setSawFourJump] = useState(false);

  const fp = useMemo(() => evaluate("fp"), []);
  const current = useMemo(() => evaluate(bits), [bits]);
  const jumped = current.bins
    .map((bin, index) => (bin !== fp.bins[index] ? ACTION_NAMES[index] : null))
    .filter((name): name is (typeof ACTION_NAMES)[number] => name !== null);
  const actionJumped = jumped.length > 0;
  const textStable = current.textTop1 === fp.textTop1;
  const visClassStable = current.visClass === fp.visClassTrue;

  const passed =
    ran &&
    sawEightStable &&
    sawFourJump &&
    prediction === "action" &&
    actionJumped &&
    textStable &&
    bits === 4;

  const completion = useMemo(
    () => ({
      lessonId: 55,
      bits,
      prediction,
      textTop1: current.textTop1,
      textCe: round(current.textCe, 6),
      visL2: round(current.visL2, 6),
      actionJumped,
      jumpedDims: jumped,
      pitchBin: current.bins[4],
      sawEightStable,
      sawFourJump,
    }),
    [
      actionJumped,
      bits,
      current.bins,
      current.textCe,
      current.textTop1,
      current.visL2,
      jumped,
      prediction,
      sawEightStable,
      sawFourJump,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function reveal() {
    if (!prediction) return;
    setRan(true);
    if (bits === 8 && !actionJumped && textStable) {
      setSawEightStable(true);
    }
    if (bits === 4 && actionJumped && textStable) {
      setSawFourJump(true);
    }
  }

  function reset() {
    setBits(8);
    setPrediction("");
    setRan(false);
    setSawEightStable(false);
    setSawFourJump(false);
  }

  return (
    <LabFrame
      lesson="55"
      title="同一序列上的 8/4 bit 损伤"
      description="教学模拟，不是模型输出。先预测哪一类 token 会先跳类，再把三个头量化到 8 bit 或 4 bit。"
    >
      <div className={styles.workspace}>
        <form
          className={styles.controls}
          onSubmit={(event) => event.preventDefault()}
        >
          <h3>量化台</h3>
          <div className={styles.modeSwitch} role="group" aria-label="比特宽度">
            <button
              type="button"
              aria-pressed={bits === 8}
              onClick={() => {
                setBits(8);
                setRan(false);
              }}
            >
              8 bit
            </button>
            <button
              type="button"
              aria-pressed={bits === 4}
              onClick={() => {
                setBits(4);
                setRan(false);
              }}
            >
              4 bit
            </button>
          </div>
          <p className={styles.note}>
            同一隐藏向量 h=[1.0, 0.5, −0.25, 0.125] 同时进文本头、视觉头和 7
            维动作头。按行对称 absmax 量化权重，舍入与 CPU 实验相同。
          </p>
          <p className={styles.note}>
            动作区间前六维 [−1, 1]、夹爪 [0, 1]，每维 8 箱。pitch 全精度落在
            −0.5，正好压在箱 2 的左边界。
          </p>
        </form>

        <div className={styles.stage}>
          <div className={styles.formula}>
            <span>
              位宽 <strong>W{bits}A16</strong>
            </span>
            <span>
              文本 top-1{" "}
              <strong>{ran ? (textStable ? "未跳" : "已跳") : "先预测"}</strong>
            </span>
            <span>
              动作跳维{" "}
              <strong>
                {ran ? (jumped.length ? jumped.join(",") : "无") : "先预测"}
              </strong>
            </span>
          </div>

          <div className={styles.tracks} aria-label="三类 token">
            <article className={styles.track}>
              <header>
                <b>文本</b>
                <span>4 类 CE</span>
              </header>
              <p>
                标签 0。全精度 margin {round(fp.textMargin, 3)}。
                {ran
                  ? ` 当前 top-1=${current.textTop1}，CE=${round(current.textCe, 4)}。`
                  : " 揭晓后才给出 CE 与 top-1。"}
              </p>
            </article>
            <article className={styles.track}>
              <header>
                <b>视觉</b>
                <span>重建 L2</span>
              </header>
              <p>
                粗类取重建向量最大分量。
                {ran
                  ? ` L2=${round(current.visL2, 4)}，粗类${
                      visClassStable ? "未跳" : "已跳"
                    }。`
                  : " L2 会上升，不等于分类跳了。"}
              </p>
            </article>
            <article className={styles.track}>
              <header>
                <b>动作</b>
                <span>7 维 × 8 箱</span>
              </header>
              <p>
                {ran
                  ? actionJumped
                    ? `${jumped.join("、")} 越过箱边界。`
                    : "七维仍在原箱。"
                  : "看哪一维先换箱号。"}
              </p>
            </article>
          </div>

          <div className={styles.binBoard} aria-label="七维动作箱">
            {ACTION_NAMES.map((name, index) => {
              const changed = ran && current.bins[index] !== fp.bins[index];
              return (
                <div
                  key={name}
                  className={`${styles.binCell} ${
                    changed ? styles.binJump : ""
                  }`}
                >
                  <b>{name}</b>
                  <span>
                    {ran
                      ? `箱 ${fp.bins[index]} → ${current.bins[index]}`
                      : `全精度箱 ${fp.bins[index]}`}
                  </span>
                  <small>
                    {ran
                      ? `${round(fp.actions[index], 3)} → ${round(
                          current.actions[index],
                          3,
                        )}`
                      : "数值待揭晓"}
                  </small>
                </div>
              );
            })}
          </div>

          <dl className={styles.metrics}>
            <div>
              <dt>文本 CE</dt>
              <dd>{ran ? round(current.textCe, 4) : "—"}</dd>
            </div>
            <div>
              <dt>视觉 L2</dt>
              <dd>{ran ? round(current.visL2, 4) : "—"}</dd>
            </div>
            <div>
              <dt>动作跳维</dt>
              <dd>{ran ? jumped.length : "—"}</dd>
            </div>
            <div>
              <dt>权重 MSE（动作头）</dt>
              <dd>{ran ? current.weightMseAct.toExponential(2) : "—"}</dd>
            </div>
          </dl>
        </div>
      </div>

      <div className={styles.predict}>
        <fieldset>
          <legend>先预测：把位宽从 8 拉到 4 以后，哪一类先坏？</legend>
          {PREDICTION_OPTIONS.map((option) => (
            <label key={option.value}>
              <input
                type="radio"
                name="lesson55-pred"
                value={option.value}
                checked={prediction === option.value}
                onChange={() => {
                  setPrediction(option.value);
                  setRan(false);
                }}
              />
              <span>{option.label}</span>
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
            揭晓 {bits} bit
          </button>
        </div>
      </div>
      {!prediction ? (
        <p className={styles.feedback}>先选预测，再揭晓关键数字。</p>
      ) : null}
      {ran && bits === 8 && !actionJumped ? (
        <p className={styles.feedback}>
          8 bit 下七维仍在原箱，文本 top-1 仍是 0。再切到 4 bit 揭晓一次。
        </p>
      ) : null}
      {ran && bits === 4 && actionJumped && textStable ? (
        <p className={styles.feedback}>
          4 bit 时 pitch 从箱 2 跳到箱 1，文本 CE 甚至略降，top-1 不变。这就是“动作 bin
          边界比文本 CE 先跳类”。
        </p>
      ) : null}

      <Gate passed={passed}>
        {passed
          ? "已先预测再揭晓：8 bit 全稳，4 bit 只有动作 pitch 跳箱，文本 top-1 仍不变。"
          : "请先提交预测，再分别揭晓 8 bit（不跳）和 4 bit（pitch 跳箱）。预测必须是动作先坏。"}
      </Gate>
    </LabFrame>
  );
}
