"use client";

import { useMemo, useState } from "react";
import styles from "./Lab02ConnectorWorkbench.module.css";
import type { FoundationLabProps } from "./types";
import { initialNumber } from "./types";

type ConnectorKey = "mlp" | "perceiver" | "qformer";

const connectorMeta: Record<
  ConnectorKey,
  { name: string; short: string; character: string }
> = {
  mlp: {
    name: "MLP / 下采样器",
    short: "MLP",
    character: "长度随输入变化",
  },
  perceiver: {
    name: "Perceiver Resampler",
    short: "Perceiver",
    character: "固定 latent 数",
  },
  qformer: {
    name: "Q-Former",
    short: "Q-Former",
    character: "固定 learnable query 数",
  },
};

export function Lab02ConnectorWorkbench({
  onComplete,
  initialState,
}: FoundationLabProps) {
  const defaults = {
    inputFrames: initialNumber(initialState, "inputFrames", 320),
    targetBudget: initialNumber(initialState, "targetBudget", 48),
    poolStride: initialNumber(initialState, "poolStride", 8),
    latentCount: initialNumber(initialState, "latentCount", 48),
    queryCount: initialNumber(initialState, "queryCount", 32),
  };
  const [inputFrames, setInputFrames] = useState(defaults.inputFrames);
  const [targetBudget, setTargetBudget] = useState(defaults.targetBudget);
  const [poolStride, setPoolStride] = useState(defaults.poolStride);
  const [latentCount, setLatentCount] = useState(defaults.latentCount);
  const [queryCount, setQueryCount] = useState(defaults.queryCount);
  const [budgetPrediction, setBudgetPrediction] =
    useState<ConnectorKey | null>(null);
  const [fixedPredictions, setFixedPredictions] = useState<ConnectorKey[]>([]);
  const [hasRun, setHasRun] = useState(false);

  const comparison = useMemo(() => {
    const outputTokens: Record<ConnectorKey, number> = {
      mlp: Math.ceil(inputFrames / poolStride),
      perceiver: latentCount,
      qformer: queryCount,
    };
    const keys: ConnectorKey[] = ["mlp", "perceiver", "qformer"];
    const best = [...keys].sort((a, b) => {
      const errorA = Math.abs(outputTokens[a] - targetBudget);
      const errorB = Math.abs(outputTokens[b] - targetBudget);
      return errorA - errorB || outputTokens[a] - outputTokens[b];
    })[0];
    return { outputTokens, best };
  }, [inputFrames, latentCount, poolStride, queryCount, targetBudget]);

  const fixedCorrect =
    fixedPredictions.length === 2 &&
    fixedPredictions.includes("perceiver") &&
    fixedPredictions.includes("qformer");
  const gatePassed =
    hasRun && budgetPrediction === comparison.best && fixedCorrect;

  function invalidate() {
    setHasRun(false);
  }

  function toggleFixed(key: ConnectorKey) {
    setFixedPredictions((current) =>
      current.includes(key)
        ? current.filter((item) => item !== key)
        : [...current, key],
    );
    invalidate();
  }

  function runComparison() {
    setHasRun(true);
    const passed = budgetPrediction === comparison.best && fixedCorrect;
    if (passed) {
      onComplete?.({
        inputFrames,
        targetBudget,
        poolStride,
        latentCount,
        queryCount,
        outputTokens: comparison.outputTokens,
        closestConnector: comparison.best,
      });
    }
  }

  function reset() {
    setInputFrames(defaults.inputFrames);
    setTargetBudget(defaults.targetBudget);
    setPoolStride(defaults.poolStride);
    setLatentCount(defaults.latentCount);
    setQueryCount(defaults.queryCount);
    setBudgetPrediction(null);
    setFixedPredictions([]);
    setHasRun(false);
  }

  const keys: ConnectorKey[] = ["mlp", "perceiver", "qformer"];

  return (
    <section className={styles.lab} aria-labelledby="lab02-title">
      <header className={styles.header}>
        <div>
          <div className={styles.tags}>
            <span>公式计算</span>
            <span>架构对照</span>
          </div>
          <h3 id="lab02-title">连接器不是一根线：把视觉帧压进 Token 预算</h3>
          <p>
            改变同一份输入，观察“池化后长度”与“固定查询长度”如何产生不同的上下文账单。
          </p>
        </div>
        <button type="button" className={styles.reset} onClick={reset}>
          重置
        </button>
      </header>

      <div className={styles.inputDeck}>
        <label>
          <span>编码器输入帧</span>
          <strong>{inputFrames}</strong>
          <input
            type="range"
            min="128"
            max="640"
            step="32"
            value={inputFrames}
            onChange={(event) => {
              setInputFrames(Number(event.target.value));
              invalidate();
            }}
          />
        </label>
        <label>
          <span>目标 Token 预算</span>
          <strong>{targetBudget}</strong>
          <input
            type="range"
            min="16"
            max="96"
            step="8"
            value={targetBudget}
            onChange={(event) => {
              setTargetBudget(Number(event.target.value));
              invalidate();
            }}
          />
        </label>
        <label>
          <span>MLP 池化步长</span>
          <select
            value={poolStride}
            onChange={(event) => {
              setPoolStride(Number(event.target.value));
              invalidate();
            }}
          >
            <option value="4">×4</option>
            <option value="8">×8</option>
            <option value="16">×16</option>
          </select>
        </label>
        <label>
          <span>Perceiver latents</span>
          <select
            value={latentCount}
            onChange={(event) => {
              setLatentCount(Number(event.target.value));
              invalidate();
            }}
          >
            <option value="24">24</option>
            <option value="48">48</option>
            <option value="64">64</option>
          </select>
        </label>
        <label>
          <span>Q-Former queries</span>
          <select
            value={queryCount}
            onChange={(event) => {
              setQueryCount(Number(event.target.value));
              invalidate();
            }}
          >
            <option value="16">16</option>
            <option value="32">32</option>
            <option value="64">64</option>
          </select>
        </label>
      </div>

      <div className={styles.connectorBoard}>
        <div className={styles.source}>
          <span>Encoder</span>
          <strong>{inputFrames}</strong>
          <small>时空位置</small>
        </div>
        <div className={styles.routes} aria-label="连接器输出比较">
          {keys.map((key) => {
            const tokens = comparison.outputTokens[key];
            const visibleTokens = Math.min(24, tokens);
            return (
              <article key={key} className={styles.route}>
                <div className={styles.routeHead}>
                  <div>
                    <strong>{connectorMeta[key].name}</strong>
                    <span>{connectorMeta[key].character}</span>
                  </div>
                  <b>{hasRun ? tokens : "?"}</b>
                </div>
                <div
                  className={styles.tokenRail}
                  aria-label={
                    hasRun ? `${tokens} 个输出 token` : "运行后显示输出 token"
                  }
                >
                  {Array.from({ length: visibleTokens }, (_, index) => (
                    <i
                      key={index}
                      className={hasRun ? styles.revealed : undefined}
                    />
                  ))}
                </div>
                <code>
                  {key === "mlp"
                    ? `⌈${inputFrames} / ${poolStride}⌉`
                    : key === "perceiver"
                      ? `L = ${latentCount}`
                      : `Q = ${queryCount}`}
                </code>
              </article>
            );
          })}
        </div>
      </div>

      <div className={styles.challenge}>
        <fieldset>
          <legend>
            预测 A：谁最接近 {targetBudget} tokens？同误差时取更少 token。
          </legend>
          <div className={styles.options}>
            {keys.map((key) => (
              <button
                key={key}
                type="button"
                aria-pressed={budgetPrediction === key}
                onClick={() => {
                  setBudgetPrediction(key);
                  invalidate();
                }}
              >
                {connectorMeta[key].short}
              </button>
            ))}
          </div>
        </fieldset>
        <fieldset>
          <legend>预测 B：哪些输出长度不随输入帧数改变？可多选。</legend>
          <div className={styles.options}>
            {keys.map((key) => (
              <button
                key={key}
                type="button"
                aria-pressed={fixedPredictions.includes(key)}
                onClick={() => toggleFixed(key)}
              >
                {connectorMeta[key].short}
              </button>
            ))}
          </div>
        </fieldset>
        <button
          className={styles.run}
          type="button"
          disabled={!budgetPrediction || fixedPredictions.length === 0}
          onClick={runComparison}
        >
          展开 Token 账本
        </button>
      </div>

      {hasRun && (
        <div className={styles.ledger} aria-live="polite">
          <div className={styles.ledgerHead}>
            <span>连接器</span>
            <span>输出长度</span>
            <span>距预算</span>
            <span>长度规律</span>
          </div>
          {keys.map((key) => (
            <div key={key} className={styles.ledgerRow}>
              <strong>{connectorMeta[key].short}</strong>
              <span>{comparison.outputTokens[key]}</span>
              <span>
                |{comparison.outputTokens[key]} − {targetBudget}| ={" "}
                {Math.abs(comparison.outputTokens[key] - targetBudget)}
              </span>
              <span>
                {key === "mlp" ? "⌈F / stride⌉" : "配置常数"}
              </span>
            </div>
          ))}
        </div>
      )}

      <div
        className={`${styles.gate} ${
          hasRun ? (gatePassed ? styles.pass : styles.retry) : ""
        }`}
        role="status"
      >
        <b>{gatePassed ? "验收已通过" : "完成验收"}</b>
        <span>
          {!hasRun
            ? "先做预算预测与长度规律判断。"
            : gatePassed
              ? "你区分了输入相关压缩与固定查询压缩。"
              : `再看公式：本组配置中最接近预算的是 ${connectorMeta[comparison.best].name}；固定长度来自 L 或 Q。`}
        </span>
      </div>
    </section>
  );
}
