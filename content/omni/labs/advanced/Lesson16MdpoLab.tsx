"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import {
  numberFrom,
  round,
  stringFrom,
  useCompletionGate,
} from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson16MdpoLab.module.css";

type Mode = "dpo" | "mdpo";

export function Lesson16MdpoLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    chosenPolicy: numberFrom(initialState, "chosenPolicy", -1.2, -5, 0),
    rejectedPolicy: numberFrom(initialState, "rejectedPolicy", -2.2, -5, 0),
    chosenRef: numberFrom(initialState, "chosenRef", -1.4, -5, 0),
    rejectedRef: numberFrom(initialState, "rejectedRef", -1.9, -5, 0),
    beta: numberFrom(initialState, "beta", 0.5, 0.1, 2),
    confidence: numberFrom(initialState, "confidence", 0.6, 0, 1),
    lambda: numberFrom(initialState, "lambda", 0.4, 0, 1),
    prediction: stringFrom(initialState, "prediction", ""),
  };
  const [chosenPolicy, setChosenPolicy] = useState(defaults.chosenPolicy);
  const [rejectedPolicy, setRejectedPolicy] = useState(defaults.rejectedPolicy);
  const [chosenRef, setChosenRef] = useState(defaults.chosenRef);
  const [rejectedRef, setRejectedRef] = useState(defaults.rejectedRef);
  const [beta, setBeta] = useState(defaults.beta);
  const [confidence, setConfidence] = useState(defaults.confidence);
  const [lambda, setLambda] = useState(defaults.lambda);
  const [mode, setMode] = useState<Mode>("dpo");
  const [prediction, setPrediction] = useState(defaults.prediction);
  const [ran, setRan] = useState(false);

  const calculation = useMemo(() => {
    const policyGap = chosenPolicy - rejectedPolicy;
    const referenceGap = chosenRef - rejectedRef;
    const advantage = policyGap - referenceGap;
    const conditionalMargin = lambda * (1 - confidence);
    const effectiveAdvantage =
      mode === "mdpo" ? advantage - conditionalMargin : advantage;
    const logit = beta * effectiveAdvantage;
    const loss = Math.log1p(Math.exp(-logit));
    const chosenProbability = 1 / (1 + Math.exp(-logit));
    return {
      policyGap,
      referenceGap,
      advantage,
      conditionalMargin,
      effectiveAdvantage,
      logit,
      loss,
      chosenProbability,
    };
  }, [
    beta,
    chosenPolicy,
    chosenRef,
    confidence,
    lambda,
    mode,
    rejectedPolicy,
    rejectedRef,
  ]);

  const passed =
    ran &&
    mode === "mdpo" &&
    prediction === "increase-loss" &&
    calculation.effectiveAdvantage > 0;
  const completion = useMemo(
    () => ({
      lessonId: 16,
      mode,
      beta,
      chosenPolicy,
      rejectedPolicy,
      chosenRef,
      rejectedRef,
      confidence,
      lambda,
      rawAdvantage: round(calculation.advantage, 3),
      conditionalMargin: round(calculation.conditionalMargin, 3),
      effectiveAdvantage: round(calculation.effectiveAdvantage, 3),
      loss: round(calculation.loss, 4),
    }),
    [
      beta,
      calculation,
      chosenPolicy,
      chosenRef,
      confidence,
      lambda,
      mode,
      rejectedPolicy,
      rejectedRef,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function reset() {
    setChosenPolicy(defaults.chosenPolicy);
    setRejectedPolicy(defaults.rejectedPolicy);
    setChosenRef(defaults.chosenRef);
    setRejectedRef(defaults.rejectedRef);
    setBeta(defaults.beta);
    setConfidence(defaults.confidence);
    setLambda(defaults.lambda);
    setMode("dpo");
    setPrediction("");
    setRan(false);
  }

  return (
    <LabFrame
      lesson="16"
      title="拆开 DPO，再加一条可审计的条件 margin"
      description="先计算 policy 相对 reference 的优势，再观察低置信度证据加入 margin 后如何改变 loss。"
    >
      <section className={styles.comparator}>
        <div className={styles.modeTabs} aria-label="损失模式">
          <button
            type="button"
            aria-pressed={mode === "dpo"}
            onClick={() => {
              setMode("dpo");
              setRan(false);
            }}
          >
            DPO
          </button>
          <button
            type="button"
            aria-pressed={mode === "mdpo"}
            onClick={() => {
              setMode("mdpo");
              setRan(false);
            }}
          >
            Conditional mDPO
          </button>
        </div>
        <div className={styles.logprobGrid}>
          <LogprobControl
            label="πθ chosen"
            value={chosenPolicy}
            onChange={(value) => {
              setChosenPolicy(value);
              setRan(false);
            }}
          />
          <LogprobControl
            label="πθ rejected"
            value={rejectedPolicy}
            onChange={(value) => {
              setRejectedPolicy(value);
              setRan(false);
            }}
          />
          <LogprobControl
            label="πref chosen"
            value={chosenRef}
            onChange={(value) => {
              setChosenRef(value);
              setRan(false);
            }}
          />
          <LogprobControl
            label="πref rejected"
            value={rejectedRef}
            onChange={(value) => {
              setRejectedRef(value);
              setRan(false);
            }}
          />
        </div>
        <div className={styles.hyperparams}>
          <label>
            <span>β <output>{beta.toFixed(2)}</output></span>
            <input
              type="range"
              min="0.1"
              max="2"
              step="0.05"
              value={beta}
              onChange={(event) => {
                setBeta(Number(event.target.value));
                setRan(false);
              }}
            />
          </label>
          <label>
            <span>多模态证据置信度 q <output>{confidence.toFixed(2)}</output></span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={confidence}
              onChange={(event) => {
                setConfidence(Number(event.target.value));
                setRan(false);
              }}
            />
          </label>
          <label>
            <span>margin 系数 λ <output>{lambda.toFixed(2)}</output></span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={lambda}
              onChange={(event) => {
                setLambda(Number(event.target.value));
                setRan(false);
              }}
            />
          </label>
        </div>
      </section>

      <section className={styles.derivation}>
        <div>
          <span>① policy gap</span>
          <strong>{chosenPolicy.toFixed(2)} − ({rejectedPolicy.toFixed(2)}) = {calculation.policyGap.toFixed(2)}</strong>
        </div>
        <div>
          <span>② subtract ref gap</span>
          <strong>Δ = {calculation.policyGap.toFixed(2)} − {calculation.referenceGap.toFixed(2)} = {calculation.advantage.toFixed(2)}</strong>
        </div>
        <div className={mode === "mdpo" ? styles.activeStep : ""}>
          <span>③ conditional margin</span>
          <strong>m(q) = λ(1−q) = {calculation.conditionalMargin.toFixed(2)}</strong>
        </div>
        <div>
          <span>④ preference loss</span>
          <strong>−log σ(β(Δ−m)) = {ran ? calculation.loss.toFixed(4) : "—"}</strong>
        </div>
      </section>

      <section className={styles.meter} aria-label="chosen preference probability">
        <header>
          <span>模型赋给 chosen 的相对偏好概率</span>
          <strong>{ran ? `${(calculation.chosenProbability * 100).toFixed(1)}%` : "—"}</strong>
        </header>
        <div>
          <i
            style={{
              width: ran ? `${calculation.chosenProbability * 100}%` : "0%",
            }}
          />
          <span style={{ left: "50%" }} aria-hidden="true" />
        </div>
        <p>
          本实验将 conditional margin 定义为 m(q)=λ(1−q)。这是本课用于演示的公式，不代表论文中的未公开实现。
        </p>
      </section>

      <div className={styles.challenge}>
        <fieldset>
          <legend>先预测：其余不变，证据置信度 q 降低会怎样？</legend>
          <label>
            <input
              type="radio"
              name="mdpo-prediction"
              checked={prediction === "increase-loss"}
              onChange={() => {
                setPrediction("increase-loss");
                setRan(false);
              }}
            />
            margin 变大，loss 变大
          </label>
          <label>
            <input
              type="radio"
              name="mdpo-prediction"
              checked={prediction === "decrease-loss"}
              onChange={() => {
                setPrediction("decrease-loss");
                setRan(false);
              }}
            />
            margin 变小，loss 变小
          </label>
        </fieldset>
        <div>
          <button type="button" onClick={reset}>重置</button>
          <button
            type="button"
            className={styles.run}
            disabled={!prediction}
            onClick={() => setRan(true)}
          >
            计算 loss
          </button>
        </div>
      </div>
      <Gate passed={passed}>
        切到 Conditional mDPO、预测正确，并让扣除 margin 后的有效优势仍大于 0。
      </Gate>
    </LabFrame>
  );
}

function LogprobControl({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className={styles.logprob}>
      <span>{label}</span>
      <input
        type="number"
        min="-5"
        max="0"
        step="0.1"
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}
