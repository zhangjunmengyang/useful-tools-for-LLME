"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import {
  mean,
  numberFrom,
  round,
  standardDeviation,
  stringFrom,
  useCompletionGate,
} from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson17GrpoVerifierLab.module.css";

type Candidate = {
  id: string;
  answer: string;
  trace: string;
  correct: boolean;
  processValid: boolean;
  formatted: boolean;
  exploit: boolean;
};

const candidates: Candidate[] = [
  {
    id: "A",
    answer: "<answer>4</answer>",
    trace: "先算括号 2+1=3，再算 12÷3=4。",
    correct: true,
    processValid: true,
    formatted: true,
    exploit: false,
  },
  {
    id: "B",
    answer: "答案是 4",
    trace: "12 / (2+1) = 12 / 3 = 4。",
    correct: true,
    processValid: true,
    formatted: false,
    exploit: false,
  },
  {
    id: "C",
    answer: "<answer>12</answer>",
    trace: "忽略算式；输出含有字符串 VERIFIED，因此弱 verifier 可能放行。",
    correct: false,
    processValid: false,
    formatted: true,
    exploit: true,
  },
  {
    id: "D",
    answer: "<answer>3</answer>",
    trace: "把除号误当作减号。",
    correct: false,
    processValid: false,
    formatted: true,
    exploit: false,
  },
  {
    id: "E",
    answer: "<answer>4</answer>",
    trace: "声称 2+1=4，但最后偶然给出 4；答案对、过程矛盾。",
    correct: true,
    processValid: false,
    formatted: true,
    exploit: false,
  },
  {
    id: "F",
    answer: "4",
    trace: "无过程。",
    correct: true,
    processValid: true,
    formatted: false,
    exploit: false,
  },
  {
    id: "G",
    answer: "<answer>6</answer>",
    trace: "先做 12÷2，再忽略 +1。",
    correct: false,
    processValid: false,
    formatted: true,
    exploit: false,
  },
  {
    id: "H",
    answer: "<answer>4</answer>",
    trace: "3×4=12，所以 12÷3=4；交叉验算成立。",
    correct: true,
    processValid: true,
    formatted: true,
    exploit: false,
  },
];

export function Lesson17GrpoVerifierLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    strictness: numberFrom(initialState, "strictness", 0.25, 0, 1),
    formatWeight: numberFrom(initialState, "formatWeight", 0.35, 0, 1),
    groupSize: numberFrom(initialState, "groupSize", 6, 4, 8),
    prediction: stringFrom(initialState, "prediction", ""),
  };
  const [strictness, setStrictness] = useState(defaults.strictness);
  const [formatWeight, setFormatWeight] = useState(defaults.formatWeight);
  const [groupSize, setGroupSize] = useState(defaults.groupSize);
  const [prediction, setPrediction] = useState(defaults.prediction);
  const [ran, setRan] = useState(false);

  const scored = useMemo(() => {
    const active = candidates.slice(0, groupSize).map((candidate) => {
      let verifierReward = 0;
      if (candidate.correct) {
        verifierReward = candidate.processValid
          ? 1
          : Math.max(0, 1 - 0.6 * strictness);
      } else if (candidate.exploit) {
        verifierReward = 1.4 * (1 - strictness);
      }
      const formatReward = candidate.formatted ? formatWeight : 0;
      const exploitPenalty =
        candidate.exploit && strictness >= 0.5 ? 0.8 * strictness : 0;
      const reward = verifierReward + formatReward - exploitPenalty;
      return {
        ...candidate,
        verifierReward,
        formatReward,
        exploitPenalty,
        reward,
      };
    });
    const rewards = active.map((candidate) => candidate.reward);
    const average = mean(rewards);
    const deviation = standardDeviation(rewards);
    const withAdvantage = active.map((candidate) => ({
      ...candidate,
      advantage:
        deviation < 1e-9 ? 0 : (candidate.reward - average) / deviation,
    }));
    const top = [...withAdvantage].sort((a, b) => {
      if (b.advantage !== a.advantage) return b.advantage - a.advantage;
      return a.id.localeCompare(b.id);
    })[0];
    return { candidates: withAdvantage, average, deviation, top };
  }, [formatWeight, groupSize, strictness]);

  const passed =
    ran &&
    prediction === scored.top.id &&
    scored.top.id === "A" &&
    strictness >= 0.65;
  const completion = useMemo(
    () => ({
      lessonId: 17,
      strictness,
      formatWeight,
      groupSize,
      topCandidate: scored.top.id,
      rewardMean: round(scored.average, 3),
      rewardStd: round(scored.deviation, 3),
      exploitNeutralized: scored.top.id !== "C",
    }),
    [formatWeight, groupSize, scored, strictness],
  );
  useCompletionGate(passed, onComplete, completion);

  function reset() {
    setStrictness(defaults.strictness);
    setFormatWeight(defaults.formatWeight);
    setGroupSize(defaults.groupSize);
    setPrediction("");
    setRan(false);
  }

  return (
    <LabFrame
      lesson="17"
      title="红队一个 GRPO verifier"
      description="同组候选先被打 reward，再按组均值和标准差归一化。弱 verifier 可能让投机答案获得正 advantage；你的任务是找出并堵住它。"
    >
      <div className={styles.lab}>
        <section className={styles.problem}>
          <p>题目</p>
          <strong>计算 12 ÷ (2 + 1)</strong>
          <div className={styles.knobs}>
            <label>
              <span>Verifier 严格度 <output>{strictness.toFixed(2)}</output></span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={strictness}
                onChange={(event) => {
                  setStrictness(Number(event.target.value));
                  setRan(false);
                }}
              />
            </label>
            <label>
              <span>格式 reward <output>{formatWeight.toFixed(2)}</output></span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={formatWeight}
                onChange={(event) => {
                  setFormatWeight(Number(event.target.value));
                  setRan(false);
                }}
              />
            </label>
            <label>
              <span>Group size</span>
              <select
                value={groupSize}
                onChange={(event) => {
                  setGroupSize(Number(event.target.value));
                  setPrediction("");
                  setRan(false);
                }}
              >
                {[4, 5, 6, 7, 8].map((value) => (
                  <option key={value}>{value}</option>
                ))}
              </select>
            </label>
          </div>
          <div className={styles.rewardSpec}>
            <b>本实验 reward 规则</b>
            <code>R = verifier + format − exploit penalty</code>
            <span>投机项：1.4(1−strictness)；严格度 ≥ .5 时再扣 .8×strictness</span>
          </div>
        </section>

        <section className={styles.candidates} aria-label="GRPO 候选组">
          <header className={styles.candidateHead}>
            <span>候选 / trace</span>
            <span>reward</span>
            <span>advantage</span>
          </header>
          {scored.candidates.map((candidate) => (
            <article
              className={`${styles.candidate} ${
                ran && candidate.id === scored.top.id ? styles.top : ""
              } ${candidate.exploit ? styles.exploit : ""}`}
              key={candidate.id}
            >
              <div className={styles.answer}>
                <span>{candidate.id}</span>
                <div>
                  <code>{candidate.answer}</code>
                  <details>
                    <summary>查看推理</summary>
                    <p>{candidate.trace}</p>
                  </details>
                </div>
              </div>
              <div className={styles.reward}>
                <b>{ran ? candidate.reward.toFixed(2) : "—"}</b>
                {ran && (
                  <small>
                    {candidate.verifierReward.toFixed(2)} + {candidate.formatReward.toFixed(2)}
                    {candidate.exploitPenalty > 0
                      ? ` − ${candidate.exploitPenalty.toFixed(2)}`
                      : ""}
                  </small>
                )}
              </div>
              <div className={styles.advantage}>
                <b>{ran ? candidate.advantage.toFixed(2) : "—"}</b>
                <i
                  style={{
                    width: ran
                      ? `${Math.min(100, Math.abs(candidate.advantage) * 45)}%`
                      : "0%",
                  }}
                  data-positive={candidate.advantage >= 0}
                />
              </div>
            </article>
          ))}
          <footer className={styles.groupStats}>
            <span>μ = {ran ? scored.average.toFixed(3) : "—"}</span>
            <span>σ = {ran ? scored.deviation.toFixed(3) : "—"}</span>
            <code>Aᵢ = (Rᵢ − μ) / σ</code>
          </footer>
        </section>
      </div>

      <div className={styles.challenge}>
        <fieldset>
          <legend>先预测：按当前 reward 规则，哪个候选的 advantage 最大？</legend>
          {scored.candidates.map((candidate) => (
            <label key={candidate.id}>
              <input
                type="radio"
                name="grpo-prediction"
                checked={prediction === candidate.id}
                onChange={() => {
                  setPrediction(candidate.id);
                  setRan(false);
                }}
              />
              {candidate.id}
            </label>
          ))}
        </fieldset>
        <div>
          <button type="button" onClick={reset}>重置</button>
          <button
            type="button"
            className={styles.run}
            disabled={!prediction}
            onClick={() => setRan(true)}
          >
            运行 GRPO 组
          </button>
        </div>
      </div>
      {ran && scored.top.exploit && (
        <p className={styles.alert}>
          Reward hacking 命中：候选 C 的答案错误，却利用弱 verifier 获得组内最高 advantage。提高严格度后重新预测。
        </p>
      )}
      {ran && prediction !== scored.top.id && (
        <p className={styles.feedback}>
          预测未命中：逐项代入 R，再用同一组 μ、σ 归一化。GRPO 不会自动修复坏 reward。
        </p>
      )}
      <Gate passed={passed}>
        把严格度调到至少 0.65，消除投机候选 C，并正确预测干净候选 A 获得最大 advantage。
      </Gate>
    </LabFrame>
  );
}
