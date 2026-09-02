"use client";

import { useMemo, useState } from "react";
import type { CSSProperties } from "react";
import styles from "./Lab10TokenPareto.module.css";
import type { FoundationLabProps } from "./types";
import { initialNumber, initialString } from "./types";

type ReducerKey = "uniform" | "recency" | "query";
type QueryKey = "timeline" | "speaker" | "cause";

const evidence = [
  { id: 0, label: "08:00", text: "会议开始" },
  { id: 1, label: "Alice", text: "提出压缩" },
  { id: 2, label: "原因", text: "显存不足" },
  { id: 3, label: "08:20", text: "方案改为池化" },
  { id: 4, label: "Bob", text: "反对固定裁剪" },
  { id: 5, label: "证据", text: "长音频溢出" },
  { id: 6, label: "08:45", text: "采用查询压缩" },
  { id: 7, label: "Carol", text: "确认结论" },
] as const;

const queryMeta: Record<
  QueryKey,
  { name: string; question: string; required: number[]; relevance: number[] }
> = {
  timeline: {
    name: "时间演变",
    question: "方案从开始到最终经历了什么变化？",
    required: [0, 3, 6],
    relevance: [0, 3, 6, 2, 5, 1, 4, 7],
  },
  speaker: {
    name: "人物归因",
    question: "Alice、Bob、Carol 分别做了什么？",
    required: [1, 4, 7],
    relevance: [1, 4, 7, 0, 3, 6, 2, 5],
  },
  cause: {
    name: "因果链",
    question: "为什么最后选择查询压缩？",
    required: [2, 5, 6],
    relevance: [2, 5, 6, 3, 4, 0, 1, 7],
  },
};

const reducerMeta: Record<ReducerKey, { name: string; color: string }> = {
  uniform: { name: "均匀采样", color: "#d36a39" },
  recency: { name: "只留最近", color: "#3979bb" },
  query: { name: "查询感知", color: "#24835d" },
};

function reduceTokens(
  reducer: ReducerKey,
  budget: number,
  query: QueryKey,
) {
  if (reducer === "recency") {
    return evidence.slice(-budget).map((item) => item.id);
  }
  if (reducer === "query") {
    return queryMeta[query].relevance.slice(0, budget);
  }
  if (budget === 1) return [0];
  const selected = Array.from({ length: budget }, (_, index) =>
    Math.round((index * (evidence.length - 1)) / (budget - 1)),
  );
  return [...new Set(selected)].slice(0, budget);
}

export function Lab10TokenPareto({
  onComplete,
  initialState,
}: FoundationLabProps) {
  const defaults = {
    query: initialString(
      initialState,
      "query",
      ["timeline", "speaker", "cause"] as const,
      "timeline",
    ),
    reducer: initialString(
      initialState,
      "reducer",
      ["uniform", "recency", "query"] as const,
      "uniform",
    ),
    budget: initialNumber(initialState, "budget", 4),
    tokenCostMs: initialNumber(initialState, "tokenCostMs", 3),
  };
  const [query, setQuery] = useState<QueryKey>(defaults.query);
  const [reducer, setReducer] = useState<ReducerKey>(defaults.reducer);
  const [budget, setBudget] = useState(defaults.budget);
  const [tokenCostMs, setTokenCostMs] = useState(defaults.tokenCostMs);
  const [minimumPrediction, setMinimumPrediction] = useState("");
  const [coveragePrediction, setCoveragePrediction] = useState<boolean | null>(
    null,
  );
  const [hasRun, setHasRun] = useState(false);

  const result = useMemo(() => {
    const required = queryMeta[query].required;
    const points = (
      ["uniform", "recency", "query"] as ReducerKey[]
    ).flatMap((method) =>
      Array.from({ length: evidence.length }, (_, index) => {
        const pointBudget = index + 1;
        const kept = reduceTokens(method, pointBudget, query);
        const coverage = required.filter((id) => kept.includes(id)).length;
        return {
          method,
          budget: pointBudget,
          kept,
          coverage,
          latencyMs: 18 + pointBudget * tokenCostMs,
          pareto: false,
        };
      }),
    );
    points.forEach((point) => {
      point.pareto = !points.some(
        (other) =>
          other.coverage >= point.coverage &&
          other.latencyMs <= point.latencyMs &&
          (other.coverage > point.coverage ||
            other.latencyMs < point.latencyMs),
      );
    });
    const currentKept = reduceTokens(reducer, budget, query);
    const currentCoverage = required.filter((id) =>
      currentKept.includes(id),
    ).length;
    const methodPoints = points.filter((point) => point.method === reducer);
    const minimumFullBudget =
      methodPoints.find((point) => point.coverage === required.length)?.budget ??
      evidence.length;
    return {
      required,
      points,
      currentKept,
      currentCoverage,
      currentLatencyMs: 18 + budget * tokenCostMs,
      minimumFullBudget,
    };
  }, [budget, query, reducer, tokenCostMs]);

  const currentHasFullCoverage =
    result.currentCoverage === result.required.length;
  const gatePassed =
    hasRun &&
    Number(minimumPrediction) === result.minimumFullBudget &&
    coveragePrediction === currentHasFullCoverage;

  function invalidate() {
    setHasRun(false);
  }

  function runReduction() {
    setHasRun(true);
    const passed =
      Number(minimumPrediction) === result.minimumFullBudget &&
      coveragePrediction === currentHasFullCoverage;
    if (passed) {
      onComplete?.({
        query,
        reducer,
        budget,
        tokenCostMs,
        keptTokens: result.currentKept,
        evidenceCoverage: `${result.currentCoverage}/${result.required.length}`,
        formulaLatencyMs: result.currentLatencyMs,
        minimumFullBudget: result.minimumFullBudget,
      });
    }
  }

  function reset() {
    setQuery(defaults.query);
    setReducer(defaults.reducer);
    setBudget(defaults.budget);
    setTokenCostMs(defaults.tokenCostMs);
    setMinimumPrediction("");
    setCoveragePrediction(null);
    setHasRun(false);
  }

  const reducerKeys: ReducerKey[] = ["uniform", "recency", "query"];

  return (
    <section className={styles.lab} aria-labelledby="lab10-title">
      <header className={styles.header}>
        <div>
          <div className={styles.tags}>
            <span>公式计算</span>
            <span>Pareto 实验</span>
          </div>
          <h3 id="lab10-title">少 Token、低延迟、够证据：三者怎么同时看？</h3>
          <p>
            不虚构“模型质量分”。本实验只计算可核验的必要证据覆盖率，并用明示延迟公式寻找非支配点。
          </p>
        </div>
        <button type="button" className={styles.reset} onClick={reset}>
          重置实验
        </button>
      </header>

      <div className={styles.queryPicker}>
        <span>选择下游问题</span>
        <div>
          {(Object.keys(queryMeta) as QueryKey[]).map((key) => (
            <button
              key={key}
              type="button"
              aria-pressed={query === key}
              onClick={() => {
                setQuery(key);
                invalidate();
              }}
            >
              <strong>{queryMeta[key].name}</strong>
              <small>{queryMeta[key].question}</small>
            </button>
          ))}
        </div>
      </div>

      <div className={styles.evidenceStrip} aria-label="八个原始证据片段">
        {evidence.map((item) => {
          const required = result.required.includes(item.id);
          const kept = result.currentKept.includes(item.id);
          return (
            <div
              key={item.id}
              className={[
                kept ? styles.kept : styles.dropped,
                required ? styles.required : "",
              ].join(" ")}
            >
              <span>E{item.id}</span>
              <strong>{item.label}</strong>
              <small>{item.text}</small>
              {required && <b>必要证据</b>}
            </div>
          );
        })}
      </div>

      <div className={styles.workbench}>
        <div className={styles.controls}>
          <fieldset>
            <legend>Reducer</legend>
            <div className={styles.reducers}>
              {reducerKeys.map((key) => (
                <button
                  type="button"
                  key={key}
                  aria-pressed={reducer === key}
                  onClick={() => {
                    setReducer(key);
                    invalidate();
                  }}
                >
                  <i style={{ background: reducerMeta[key].color }} />
                  {reducerMeta[key].name}
                </button>
              ))}
            </div>
          </fieldset>
          <label>
            <span>
              Token 预算 <strong>{budget}/8</strong>
            </span>
            <input
              type="range"
              min="1"
              max="8"
              step="1"
              value={budget}
              onChange={(event) => {
                setBudget(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <label>
            <span>
              每 Token 公式成本 <strong>{tokenCostMs} ms</strong>
            </span>
            <input
              type="range"
              min="1"
              max="6"
              step="1"
              value={tokenCostMs}
              onChange={(event) => {
                setTokenCostMs(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <div className={styles.costFormula}>
            <span>延迟教学公式</span>
            <code>L = 18 ms + kept_tokens × cost</code>
            <strong>
              {hasRun ? `${result.currentLatencyMs} ms` : "待运行"}
            </strong>
          </div>
        </div>

        <div className={styles.plotPanel}>
          <div className={styles.plotHead}>
            <div>
              <span>证据覆盖 — Token — 延迟</span>
              <strong>Pareto map</strong>
            </div>
            <small>空心点 = 被另一配置支配</small>
          </div>
          <div className={styles.plotWrap}>
            <div className={styles.yLabels} aria-hidden="true">
              <span>3/3</span>
              <span>2/3</span>
              <span>1/3</span>
              <span>0/3</span>
            </div>
            <div
              className={styles.plot}
              role="img"
              aria-label="三种 reducer 在八个 token 预算下的证据覆盖 Pareto 图"
            >
              {hasRun &&
                result.points.map((point) => (
                  <div
                    key={`${point.method}-${point.budget}`}
                    className={[
                      styles.point,
                      styles[point.method],
                      point.pareto ? styles.pareto : "",
                      point.method === reducer && point.budget === budget
                        ? styles.selected
                        : "",
                    ].join(" ")}
                    style={
                      {
                        "--plot-column": point.budget,
                        "--plot-row": 4 - point.coverage,
                        "--point-color": reducerMeta[point.method].color,
                      } as CSSProperties
                    }
                    title={`${reducerMeta[point.method].name} · ${point.budget} tokens · 覆盖 ${point.coverage}/3 · ${point.latencyMs} ms`}
                  >
                    <span />
                  </div>
                ))}
            </div>
          </div>
          <div className={styles.xLabels} aria-hidden="true">
            {Array.from({ length: 8 }, (_, index) => (
              <span key={index}>
                {index + 1}
                <small>{18 + (index + 1) * tokenCostMs}ms</small>
              </span>
            ))}
          </div>
          <div className={styles.legend}>
            {reducerKeys.map((key) => (
              <span key={key}>
                <i style={{ background: reducerMeta[key].color }} />
                {reducerMeta[key].name}
              </span>
            ))}
          </div>
        </div>
      </div>

      <div className={styles.challenge}>
        <div>
          <span>先预测，再生成 Pareto map</span>
          <strong>
            当前 reducer：{reducerMeta[reducer].name} · 当前预算：{budget}
          </strong>
        </div>
        <label>
          <span>保留全部 3 条必要证据的最小预算</span>
          <input
            type="number"
            min="1"
            max="8"
            value={minimumPrediction}
            onChange={(event) => {
              setMinimumPrediction(event.target.value);
              invalidate();
            }}
          />
        </label>
        <fieldset>
          <legend>当前预算能覆盖全部必要证据吗？</legend>
          <div>
            <button
              type="button"
              aria-pressed={coveragePrediction === true}
              onClick={() => {
                setCoveragePrediction(true);
                invalidate();
              }}
            >
              能
            </button>
            <button
              type="button"
              aria-pressed={coveragePrediction === false}
              onClick={() => {
                setCoveragePrediction(false);
                invalidate();
              }}
            >
              不能
            </button>
          </div>
        </fieldset>
        <button
          type="button"
          className={styles.run}
          disabled={
            minimumPrediction.trim() === "" || coveragePrediction === null
          }
          onClick={runReduction}
        >
          运行全部配置
        </button>
      </div>

      {hasRun && (
        <div className={styles.ledger} aria-live="polite">
          <div className={styles.ledgerHead}>
            <span>同预算比较</span>
            <span>保留 evidence id</span>
            <span>必要证据覆盖</span>
            <span>公式延迟</span>
            <span>Pareto</span>
          </div>
          {reducerKeys.map((method) => {
            const point = result.points.find(
              (item) => item.method === method && item.budget === budget,
            );
            return (
              <div className={styles.ledgerRow} key={method}>
                <strong>{reducerMeta[method].name}</strong>
                <code>{point?.kept.map((id) => `E${id}`).join(", ")}</code>
                <span>{point?.coverage}/3</span>
                <span>{point?.latencyMs} ms</span>
                <b>{point?.pareto ? "非支配" : "被支配"}</b>
              </div>
            );
          })}
        </div>
      )}

      <div
        className={`${styles.gate} ${
          hasRun ? (gatePassed ? styles.pass : styles.retry) : ""
        }`}
        role="status"
      >
        <strong>{gatePassed ? "验收已通过" : "完成验收"}</strong>
        <span>
          {!hasRun
            ? "先手算 reducer 的保留集合；覆盖率只统计三条明确必要证据。"
            : gatePassed
              ? "你已经用可审计的 evidence coverage 取代了虚构质量分，并读懂非支配配置。"
              : `${reducerMeta[reducer].name} 达到 3/3 覆盖的最小预算是 ${result.minimumFullBudget}；当前预算覆盖 ${result.currentCoverage}/3。`}
        </span>
      </div>
    </section>
  );
}
