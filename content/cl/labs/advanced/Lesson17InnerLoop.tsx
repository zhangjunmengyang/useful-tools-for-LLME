"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import chrome from "./LabFrame.module.css";
import styles from "./Lesson17InnerLoop.module.css";
import type { AdvancedLabProps } from "./types";
import { numberFrom, round } from "./labUtils";

const VOCAB = ["春", "江", "花", "月", "夜", "游"] as const;
type Tok = (typeof VOCAB)[number];
const SEQ: Tok[] = [
  "春",
  "江",
  "花",
  "月",
  "夜",
  "春",
  "江",
  "花",
  "月",
  "夜",
  "春",
  "江",
  "花",
  "月",
  "夜",
  "游",
];

type RowPred = "current" | "all" | "next";

function evolve(lr: number) {
  let weights = VOCAB.map((_, row) =>
    VOCAB.map((__, col) => (row === col ? 0.18 : 0.03)),
  );
  const steps: {
    changedRow: number;
    delta: number;
    weights: number[][];
    token: Tok;
    target: Tok;
  }[] = [];

  for (let t = 0; t < SEQ.length - 1; t += 1) {
    const key = VOCAB.indexOf(SEQ[t]);
    const target = VOCAB.indexOf(SEQ[t + 1]);
    const row = weights[key];
    const error = row.map((value, col) => value - (col === target ? 1 : 0));
    const next = weights.map((current, index) =>
      index === key ? current.map((value, col) => value - lr * error[col]) : current,
    );
    const delta = Math.sqrt(error.reduce((sum, item) => sum + (lr * item) ** 2, 0));
    weights = next;
    steps.push({
      changedRow: key,
      delta: round(delta, 4),
      weights: next.map((line) => line.map((value) => round(value, 3))),
      token: SEQ[t],
      target: SEQ[t + 1],
    });
  }

  return { steps };
}

export function Lesson17InnerLoop({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    lr: numberFrom(initialState, "lr", 0.35, 0.1, 0.8),
    step: numberFrom(initialState, "step", 6, 1, 15),
  };
  const [lr, setLr] = useState(defaults.lr);
  const [step, setStep] = useState(defaults.step);
  const [rowPred, setRowPred] = useState<RowPred | null>(null);
  const [rnnPred, setRnnPred] = useState<"yes" | "no" | null>(null);
  const [hasRun, setHasRun] = useState(false);

  const sim = useMemo(() => evolve(lr), [lr]);
  const current = sim.steps[step - 1];
  const gatePassed = hasRun && rowPred === "current" && rnnPred === "no";

  function invalidate() {
    setHasRun(false);
  }

  function run() {
    setHasRun(true);
    if (rowPred === "current" && rnnPred === "no") {
      onComplete?.({
        lr,
        step,
        token: current.token,
        changedRow: VOCAB[current.changedRow],
        delta: current.delta,
      });
    }
  }

  function reset() {
    setLr(defaults.lr);
    setStep(6);
    setRowPred(null);
    setRnnPred(null);
    setHasRun(false);
  }

  return (
    <LabFrame
      lesson="17"
      title="内环台阶：W 的哪一行刚动了"
      description="TTT 层的隐状态是一套测试时可以用梯度更新的权重 W。对当前 token 做一次内环更新后，只有该 token 对应的行会动。普通 RNN 的隐状态是向量，做不到这件事。"
      onReset={reset}
    >
      <div className={chrome.workbench}>
        <div className={chrome.controls}>
          <label>
            <span>
              内环学习率 <strong>{lr.toFixed(2)}</strong>
            </span>
            <input
              type="range"
              min="0.1"
              max="0.8"
              step="0.05"
              value={lr}
              onChange={(event) => {
                setLr(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <label>
            <span>
              查看第 <strong>{step}</strong> 步（token {SEQ[step - 1]}）
            </span>
            <input
              type="range"
              min="1"
              max="15"
              step="1"
              value={step}
              onChange={(event) => {
                setStep(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <div className={chrome.formula}>
            <code>{"ℓ = ½‖W x_t − x_{t+1}‖²"}</code>
            <code>{"∇W = (W x_t − x_{t+1}) x_tᵀ"}</code>
            <code>x_t one-hot ⇒ 只更新第 t 行</code>
          </div>
        </div>

        <div className={chrome.panel}>
          <div className={chrome.metrics}>
            <div className={chrome.metric}>
              <span>当前 token</span>
              <strong>{hasRun ? current.token : "?"}</strong>
            </div>
            <div className={chrome.metric}>
              <span>变动行</span>
              <strong>{hasRun ? VOCAB[current.changedRow] : "?"}</strong>
            </div>
            <div className={chrome.metric}>
              <span>‖Δrow‖</span>
              <strong>{hasRun ? current.delta.toFixed(3) : "?"}</strong>
            </div>
          </div>
          <ol className={styles.seq} aria-label="16 token 序列">
            {SEQ.map((token, index) => (
              <li
                key={`${token}-${index}`}
                data-active={hasRun && index === step - 1 ? "true" : "false"}
              >
                {token}
              </li>
            ))}
          </ol>
          <table className={styles.matrix} aria-label="记忆矩阵 W">
            <thead>
              <tr>
                <th />
                {VOCAB.map((token) => (
                  <th key={token}>{token}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {VOCAB.map((rowName, row) => (
                <tr
                  key={rowName}
                  data-changed={
                    hasRun && current.changedRow === row ? "true" : "false"
                  }
                >
                  <th>{rowName}</th>
                  {VOCAB.map((colName, col) => (
                    <td key={colName}>
                      {hasRun ? current.weights[row][col].toFixed(2) : "—"}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className={chrome.prediction}>
        <fieldset>
          <legend>预测 1：这一步 W 的哪一行会动？</legend>
          <div className={chrome.choiceRow}>
            <button
              type="button"
              aria-pressed={rowPred === "current"}
              onClick={() => {
                setRowPred("current");
                invalidate();
              }}
            >
              当前 token 那一行
            </button>
            <button
              type="button"
              aria-pressed={rowPred === "next"}
              onClick={() => {
                setRowPred("next");
                invalidate();
              }}
            >
              下一 token 那一行
            </button>
            <button
              type="button"
              aria-pressed={rowPred === "all"}
              onClick={() => {
                setRowPred("all");
                invalidate();
              }}
            >
              所有行一起动
            </button>
          </div>
        </fieldset>
        <fieldset>
          <legend>预测 2：普通 RNN 隐状态能对当前 token 做多步梯度吗？</legend>
          <div className={chrome.choiceRow}>
            <button
              type="button"
              aria-pressed={rnnPred === "yes"}
              onClick={() => {
                setRnnPred("yes");
                invalidate();
              }}
            >
              能
            </button>
            <button
              type="button"
              aria-pressed={rnnPred === "no"}
              onClick={() => {
                setRnnPred("no");
                invalidate();
              }}
            >
              不能
            </button>
          </div>
        </fieldset>
        <button
          type="button"
          className={chrome.run}
          disabled={!rowPred || !rnnPred}
          onClick={run}
        >
          运行内环
        </button>
      </div>

      <Gate ran={hasRun} passed={gatePassed}>
        {!hasRun
          ? "先判断哪一行更新、RNN 能不能做内环梯度，再揭示 W。"
          : gatePassed
            ? `第 ${step} 步 token「${current.token}」只改了第 ${VOCAB[current.changedRow]} 行，Δ=${current.delta.toFixed(3)}。`
            : "x_t 是 one-hot 时，梯度是 rank-1，只碰到当前 token 对应行。RNN 隐状态是向量，不能在测试时对这段序列做多步梯度。"}
      </Gate>
    </LabFrame>
  );
}
