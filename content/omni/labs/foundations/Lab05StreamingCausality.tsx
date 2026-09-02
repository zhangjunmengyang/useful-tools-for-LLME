"use client";

import { useMemo, useState } from "react";
import styles from "./Lab05StreamingCausality.module.css";
import type { FoundationLabProps } from "./types";
import { initialNumber } from "./types";

const chunkIds = [0, 1, 2, 3, 4, 5] as const;

export function Lab05StreamingCausality({
  onComplete,
  initialState,
}: FoundationLabProps) {
  const defaults = {
    chunkMs: initialNumber(initialState, "chunkMs", 160),
    lookahead: initialNumber(initialState, "lookahead", 1),
    cacheChunks: initialNumber(initialState, "cacheChunks", 2),
    outputChunk: initialNumber(initialState, "outputChunk", 3),
    mutationChunk: initialNumber(initialState, "mutationChunk", 4),
  };
  const [chunkMs, setChunkMs] = useState(defaults.chunkMs);
  const [lookahead, setLookahead] = useState(defaults.lookahead);
  const [cacheChunks, setCacheChunks] = useState(defaults.cacheChunks);
  const [outputChunk, setOutputChunk] = useState(defaults.outputChunk);
  const [mutationChunk, setMutationChunk] = useState(defaults.mutationChunk);
  const [prediction, setPrediction] = useState<boolean | null>(null);
  const [hasRun, setHasRun] = useState(false);

  const result = useMemo(() => {
    const windowStart = Math.max(0, outputChunk - cacheChunks);
    const windowEnd = Math.min(chunkIds.length - 1, outputChunk + lookahead);
    const canAffect =
      mutationChunk >= windowStart && mutationChunk <= windowEnd;
    return {
      windowStart,
      windowEnd,
      canAffect,
      rightContextMs: lookahead * chunkMs,
      cacheContextMs: cacheChunks * chunkMs,
      firstOutputReadyMs: (1 + lookahead) * chunkMs,
    };
  }, [cacheChunks, chunkMs, lookahead, mutationChunk, outputChunk]);

  const gatePassed = hasRun && prediction === result.canAffect;

  function invalidate() {
    setHasRun(false);
  }

  function runProbe() {
    setHasRun(true);
    if (prediction === result.canAffect) {
      onComplete?.({
        chunkMs,
        lookahead,
        cacheChunks,
        outputChunk,
        mutationChunk,
        dependencyWindow: [result.windowStart, result.windowEnd],
        canAffect: result.canAffect,
        firstOutputReadyMs: result.firstOutputReadyMs,
      });
    }
  }

  function reset() {
    setChunkMs(defaults.chunkMs);
    setLookahead(defaults.lookahead);
    setCacheChunks(defaults.cacheChunks);
    setOutputChunk(defaults.outputChunk);
    setMutationChunk(defaults.mutationChunk);
    setPrediction(null);
    setHasRun(false);
  }

  return (
    <section className={styles.lab} aria-labelledby="lab05-title">
      <header className={styles.header}>
        <div>
          <div className={styles.tags}>
            <span>教学模拟</span>
            <span>因果窗口</span>
          </div>
          <h3 id="lab05-title">流式不是切块：未来到底偷看了多少？</h3>
          <p>
            用 cache、当前块和 lookahead 拼出一个精确依赖窗口，再用“反事实改块”检验因果边界。
          </p>
        </div>
        <button type="button" onClick={reset} className={styles.reset}>
          重置窗口
        </button>
      </header>

      <div className={styles.config}>
        <fieldset>
          <legend>Chunk 时长</legend>
          <div className={styles.segmented}>
            {[80, 160, 320].map((value) => (
              <button
                key={value}
                type="button"
                aria-pressed={chunkMs === value}
                onClick={() => {
                  setChunkMs(value);
                  invalidate();
                }}
              >
                {value} ms
              </button>
            ))}
          </div>
        </fieldset>
        <label>
          <span>左侧 KV cache</span>
          <select
            value={cacheChunks}
            onChange={(event) => {
              setCacheChunks(Number(event.target.value));
              invalidate();
            }}
          >
            {[1, 2, 3].map((value) => (
              <option value={value} key={value}>
                {value} chunks
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>右侧 lookahead</span>
          <select
            value={lookahead}
            onChange={(event) => {
              setLookahead(Number(event.target.value));
              invalidate();
            }}
          >
            {[0, 1, 2].map((value) => (
              <option value={value} key={value}>
                {value} chunks
              </option>
            ))}
          </select>
        </label>
        <div className={styles.latency}>
          <span>首块最早可发射</span>
          <strong>
            {hasRun ? `${result.firstOutputReadyMs} ms` : "待计算"}
          </strong>
          <code>(1 + lookahead) × chunk_ms</code>
        </div>
      </div>

      <div className={styles.timeline}>
        <div className={styles.axis} aria-hidden="true">
          <span>0 ms</span>
          <span>{chunkMs * chunkIds.length} ms</span>
        </div>
        <div className={styles.chunkRow} aria-label="选择要计算的输出块">
          <span className={styles.rowLabel}>输出</span>
          {chunkIds.map((chunk) => (
            <button
              type="button"
              key={chunk}
              aria-pressed={outputChunk === chunk}
              onClick={() => {
                setOutputChunk(chunk);
                invalidate();
              }}
            >
              y{chunk}
              <small>chunk {chunk}</small>
            </button>
          ))}
        </div>
        <div className={styles.chunkRow} aria-label="选择要反事实修改的输入块">
          <span className={styles.rowLabel}>改块</span>
          {chunkIds.map((chunk) => (
            <button
              type="button"
              key={chunk}
              aria-pressed={mutationChunk === chunk}
              onClick={() => {
                setMutationChunk(chunk);
                invalidate();
              }}
            >
              x{chunk}
              <small>{mutationChunk === chunk ? "反事实" : "input"}</small>
            </button>
          ))}
        </div>

        <div className={styles.windowRow} aria-label="解码器依赖窗口">
          <span className={styles.rowLabel}>依赖</span>
          {chunkIds.map((chunk) => {
            const inWindow =
              chunk >= result.windowStart && chunk <= result.windowEnd;
            const kind =
              chunk < outputChunk
                ? "cache"
                : chunk === outputChunk
                  ? "current"
                  : "lookahead";
            return (
              <div
                key={chunk}
                className={[
                  styles.windowCell,
                  inWindow ? styles[kind] : styles.outside,
                  mutationChunk === chunk ? styles.mutated : "",
                ].join(" ")}
              >
                {inWindow ? (
                  <>
                    <b>x{chunk}</b>
                    <span>
                      {kind === "cache"
                        ? "cache"
                        : kind === "current"
                          ? "current"
                          : "future"}
                    </span>
                  </>
                ) : (
                  <span>不可见</span>
                )}
              </div>
            );
          })}
        </div>
      </div>

      <div className={styles.formulaBand}>
        <div>
          <span>依赖下界</span>
          <code>max(0, k − cache) = {result.windowStart}</code>
        </div>
        <div>
          <span>依赖上界</span>
          <code>min(5, k + lookahead) = {result.windowEnd}</code>
        </div>
        <div>
          <span>上下文长度</span>
          <code>
            {result.cacheContextMs} ms ← y{outputChunk} →{" "}
            {result.rightContextMs} ms
          </code>
        </div>
      </div>

      <div className={styles.question}>
        <div>
          <span>先预测，再运行</span>
          <strong>
            只修改 x{mutationChunk}，y{outputChunk} 有可能改变吗？
          </strong>
        </div>
        <div className={styles.yesNo}>
          <button
            type="button"
            aria-pressed={prediction === true}
            onClick={() => {
              setPrediction(true);
              invalidate();
            }}
          >
            会
          </button>
          <button
            type="button"
            aria-pressed={prediction === false}
            onClick={() => {
              setPrediction(false);
              invalidate();
            }}
          >
            不会
          </button>
        </div>
        <button
          type="button"
          className={styles.run}
          disabled={prediction === null}
          onClick={runProbe}
        >
          运行反事实探针
        </button>
      </div>

      <div
        className={`${styles.gate} ${
          hasRun ? (gatePassed ? styles.pass : styles.retry) : ""
        }`}
        role="status"
      >
        <strong>{gatePassed ? "验收已通过" : "完成验收"}</strong>
        <span>
          {!hasRun
            ? "判断被修改块是否落在闭区间 [k−cache, k+lookahead]。"
            : gatePassed
              ? `正确：x${mutationChunk} ${
                  result.canAffect ? "位于" : "不在"
                } y${outputChunk} 的可见窗口 [${result.windowStart}, ${result.windowEnd}]。`
              : `答案由索引窗口决定，不由“看起来离得近”决定。当前窗口是 [${result.windowStart}, ${result.windowEnd}]。`}
        </span>
      </div>
    </section>
  );
}
