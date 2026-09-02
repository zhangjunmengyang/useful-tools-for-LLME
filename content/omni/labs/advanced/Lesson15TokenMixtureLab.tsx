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
import styles from "./Lesson15TokenMixtureLab.module.css";

type Modality = "text" | "image" | "audio" | "video";
type Row = {
  id: Modality;
  label: string;
  samples: number;
  tokens: number;
  color: string;
};

const baseRows: Row[] = [
  { id: "text", label: "Text", samples: 8, tokens: 512, color: "#607e70" },
  { id: "image", label: "Image", samples: 8, tokens: 1024, color: "#9a6c47" },
  { id: "audio", label: "Audio", samples: 8, tokens: 1500, color: "#4f7894" },
  { id: "video", label: "Video", samples: 8, tokens: 4096, color: "#875e83" },
];

export function Lesson15TokenMixtureLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaultBudget = numberFrom(initialState, "budget", 65536, 16384, 262144);
  const [rows, setRows] = useState<Row[]>(() =>
    baseRows.map((row) => ({
      ...row,
      samples: numberFrom(initialState, `${row.id}Samples`, row.samples, 1, 32),
      tokens: numberFrom(initialState, `${row.id}Tokens`, row.tokens, 64, 8192),
    })),
  );
  const [budget, setBudget] = useState(defaultBudget);
  const [mode, setMode] = useState<"raw" | "balanced">("raw");
  const [prediction, setPrediction] = useState(
    stringFrom(initialState, "prediction", ""),
  );
  const [ran, setRan] = useState(false);

  const calculation = useMemo(() => {
    const rawTokens = rows.map((row) => row.samples * row.tokens);
    const rawTotal = rawTokens.reduce((sum, value) => sum + value, 0);
    const targetTokens = budget / rows.length;
    const balancedSamples = rows.map((row) => targetTokens / row.tokens);
    const dominantIndex = rawTokens.indexOf(Math.max(...rawTokens));
    const maxBalancedTokens = Math.max(
      ...rows.map((row, index) => balancedSamples[index] * row.tokens),
    );
    const minBalancedTokens = Math.min(
      ...rows.map((row, index) => balancedSamples[index] * row.tokens),
    );
    return {
      rawTokens,
      rawTotal,
      targetTokens,
      balancedSamples,
      dominant: rows[dominantIndex].id,
      imbalance:
        minBalancedTokens === 0 ? Infinity : maxBalancedTokens / minBalancedTokens,
    };
  }, [budget, rows]);

  const passed =
    ran &&
    mode === "balanced" &&
    prediction === calculation.dominant &&
    calculation.imbalance <= 1.001;
  const completion = useMemo(
    () => ({
      lessonId: 15,
      tokenBudget: budget,
      dominantRawModality: calculation.dominant,
      balancedSamples: Object.fromEntries(
        rows.map((row, index) => [
          row.id,
          round(calculation.balancedSamples[index], 2),
        ]),
      ),
      ...Object.fromEntries(
        rows.flatMap((row) => [
          [`${row.id}Samples`, row.samples],
          [`${row.id}Tokens`, row.tokens],
        ]),
      ),
    }),
    [budget, calculation, rows],
  );
  useCompletionGate(passed, onComplete, completion);

  function updateRow(
    id: Modality,
    field: "samples" | "tokens",
    value: number,
  ) {
    setRows((current) =>
      current.map((row) =>
        row.id === id ? { ...row, [field]: value } : row,
      ),
    );
    setRan(false);
  }

  function reset() {
    setRows(baseRows);
    setBudget(defaultBudget);
    setMode("raw");
    setPrediction("");
    setRan(false);
  }

  return (
    <LabFrame
      lesson="15"
      title="配一锅按 token 公平的多模态数据"
      description="每种模态一条样本的 token 成本差异很大。“每类 8 条”并不公平；打开账本，把样本配比换算成 token 配比。"
    >
      <section className={styles.sheet}>
        <header className={styles.sheetHeader}>
          <div>
            <h3>Mixture ledger</h3>
            <p>所有数字都来自 samples × tokens/sample</p>
          </div>
          <label>
            每 batch token 预算
            <select
              value={budget}
              onChange={(event) => {
                setBudget(Number(event.target.value));
                setRan(false);
              }}
            >
              {[16384, 32768, 65536, 131072, 262144].map((value) => (
                <option key={value}>{value.toLocaleString()}</option>
              ))}
            </select>
          </label>
        </header>
        <div className={styles.table} role="table" aria-label="多模态 token 配比">
          <div className={styles.tableHead} role="row">
            <span role="columnheader">模态</span>
            <span role="columnheader">样本数</span>
            <span role="columnheader">token / sample</span>
            <span role="columnheader">
              {mode === "raw" ? "实际 token" : "建议样本数"}
            </span>
          </div>
          {rows.map((row, index) => {
            const tokenCount = calculation.rawTokens[index];
            const share =
              calculation.rawTotal === 0 ? 0 : tokenCount / calculation.rawTotal;
            return (
              <div className={styles.tableRow} role="row" key={row.id}>
                <span className={styles.modality} role="cell">
                  <i style={{ background: row.color }} />
                  {row.label}
                </span>
                <label role="cell">
                  <span className={styles.srOnly}>{row.label} 样本数</span>
                  <input
                    type="number"
                    min="1"
                    max="32"
                    value={row.samples}
                    onChange={(event) =>
                      updateRow(row.id, "samples", Number(event.target.value))
                    }
                  />
                </label>
                <label role="cell">
                  <span className={styles.srOnly}>{row.label} 每样本 token</span>
                  <input
                    type="number"
                    min="64"
                    max="8192"
                    step="64"
                    value={row.tokens}
                    onChange={(event) =>
                      updateRow(row.id, "tokens", Number(event.target.value))
                    }
                  />
                </label>
                <span className={styles.result} role="cell">
                  <b>
                    {mode === "raw"
                      ? tokenCount.toLocaleString()
                      : calculation.balancedSamples[index].toFixed(2)}
                  </b>
                  <small>
                    {mode === "raw"
                      ? `${(share * 100).toFixed(1)}%`
                      : `${calculation.targetTokens.toLocaleString()} tokens`}
                  </small>
                </span>
              </div>
            );
          })}
        </div>
      </section>

      <section className={styles.balanceView}>
        <div className={styles.modeSwitch} aria-label="配比模式">
          <button
            type="button"
            aria-pressed={mode === "raw"}
            onClick={() => {
              setMode("raw");
              setRan(false);
            }}
          >
            当前样本配比
          </button>
          <button
            type="button"
            aria-pressed={mode === "balanced"}
            onClick={() => {
              setMode("balanced");
              setRan(false);
            }}
          >
            Token-balanced 配方
          </button>
        </div>
        <div className={styles.stack} aria-label="各模态 token 占比">
          {rows.map((row, index) => {
            const percent =
              mode === "balanced"
                ? 25
                : (calculation.rawTokens[index] / calculation.rawTotal) * 100;
            return (
              <i
                key={row.id}
                style={{ width: `${percent}%`, background: row.color }}
                title={`${row.label}: ${percent.toFixed(1)}% tokens`}
              />
            );
          })}
        </div>
        <p className={styles.formula}>
          Token-balanced：n<sub>m</sub> = budget ÷ M ÷ cost<sub>m</sub>
          {" "}→ 每个模态恰好 {calculation.targetTokens.toLocaleString()} tokens
        </p>
      </section>

      <div className={styles.challenge}>
        <fieldset>
          <legend>先预测：按当前表格直接组 batch，哪种模态贡献的 token 最多？</legend>
          {rows.map((row) => (
            <label key={row.id}>
              <input
                type="radio"
                name="mixture-prediction"
                checked={prediction === row.id}
                onChange={() => {
                  setPrediction(row.id);
                  setRan(false);
                }}
              />
              {row.label}
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
            校验配方
          </button>
        </div>
      </div>
      {ran && prediction !== calculation.dominant && (
        <p className={styles.feedback}>
          先逐行算 samples × tokens/sample；样本数相近不代表 token 数相近。
        </p>
      )}
      <Gate passed={passed}>
        正确预测原始 token 主导模态，并切到 Token-balanced 配方后运行校验。
      </Gate>
    </LabFrame>
  );
}
