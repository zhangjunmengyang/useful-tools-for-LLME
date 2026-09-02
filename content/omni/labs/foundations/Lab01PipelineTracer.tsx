"use client";

import { useMemo, useState } from "react";
import type { CSSProperties } from "react";
import styles from "./Lab01PipelineTracer.module.css";
import type { FoundationLabProps } from "./types";
import { initialNumber } from "./types";

type StageKey = "encoder" | "connector" | "prefill" | "decode";

const stageLabels: Record<StageKey, string> = {
  encoder: "音频编码",
  connector: "连接器",
  prefill: "多模态 Prefill",
  decode: "首 Token 解码",
};

export function Lab01PipelineTracer({
  onComplete,
  initialState,
}: FoundationLabProps) {
  const defaults = {
    audioSeconds: initialNumber(initialState, "audioSeconds", 6),
    strideMs: initialNumber(initialState, "strideMs", 20),
    promptTokens: initialNumber(initialState, "promptTokens", 64),
  };
  const [audioSeconds, setAudioSeconds] = useState(defaults.audioSeconds);
  const [strideMs, setStrideMs] = useState(defaults.strideMs);
  const [promptTokens, setPromptTokens] = useState(defaults.promptTokens);
  const [bottleneckPrediction, setBottleneckPrediction] =
    useState<StageKey | null>(null);
  const [lossPrediction, setLossPrediction] = useState<
    "user" | "assistant" | null
  >(null);
  const [hasRun, setHasRun] = useState(false);

  const trace = useMemo(() => {
    const frames = Math.ceil((audioSeconds * 1000) / strideMs);
    const connectorTokens = Math.ceil(frames / 4);
    const durations: Record<StageKey, number> = {
      encoder: Math.round(frames * 0.18),
      connector: Math.round(connectorTokens * 0.12),
      prefill: Math.round((connectorTokens + promptTokens) * 0.42),
      decode: 28,
    };
    const bottleneck = (Object.keys(durations) as StageKey[]).reduce(
      (largest, key) =>
        durations[key] > durations[largest] ? key : largest,
      "encoder",
    );
    return {
      frames,
      connectorTokens,
      durations,
      bottleneck,
      ttft: Object.values(durations).reduce((sum, value) => sum + value, 0),
    };
  }, [audioSeconds, promptTokens, strideMs]);

  const gatePassed =
    hasRun &&
    bottleneckPrediction === trace.bottleneck &&
    lossPrediction === "assistant";
  const maxDuration = Math.max(...Object.values(trace.durations));

  function invalidate() {
    setHasRun(false);
  }

  function runTrace() {
    setHasRun(true);
    const passed =
      bottleneckPrediction === trace.bottleneck &&
      lossPrediction === "assistant";
    if (passed) {
      onComplete?.({
        audioSeconds,
        strideMs,
        promptTokens,
        frames: trace.frames,
        connectorTokens: trace.connectorTokens,
        ttftMs: trace.ttft,
        bottleneck: trace.bottleneck,
      });
    }
  }

  function reset() {
    setAudioSeconds(defaults.audioSeconds);
    setStrideMs(defaults.strideMs);
    setPromptTokens(defaults.promptTokens);
    setBottleneckPrediction(null);
    setLossPrediction(null);
    setHasRun(false);
  }

  return (
    <section className={styles.lab} aria-labelledby="lab01-title">
      <header className={styles.header}>
        <div>
          <div className={styles.labels} aria-label="实验类型">
            <span>教学模拟</span>
            <span>公式计算</span>
          </div>
          <h3 id="lab01-title">从波形到首 Token：追踪一次 TTFT</h3>
          <p>
            先判断瓶颈和 loss mask，再运行可审计的延迟账本。这里的毫秒数是明示公式，
            不是硬件实测。
          </p>
        </div>
        <button className={styles.reset} type="button" onClick={reset}>
          重置实验
        </button>
      </header>

      <div className={styles.workbench}>
        <div className={styles.controls}>
          <label>
            <span>
              音频长度 <strong>{audioSeconds} s</strong>
            </span>
            <input
              type="range"
              min="2"
              max="12"
              step="1"
              value={audioSeconds}
              onChange={(event) => {
                setAudioSeconds(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <label>
            <span>编码器帧移</span>
            <select
              value={strideMs}
              onChange={(event) => {
                setStrideMs(Number(event.target.value));
                invalidate();
              }}
            >
              <option value="10">10 ms</option>
              <option value="20">20 ms</option>
              <option value="40">40 ms</option>
            </select>
          </label>
          <label>
            <span>
              文本提示 <strong>{promptTokens} tokens</strong>
            </span>
            <input
              type="range"
              min="16"
              max="192"
              step="16"
              value={promptTokens}
              onChange={(event) => {
                setPromptTokens(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <div className={styles.formula}>
            <code>frames = ⌈audio_ms / stride⌉</code>
            <code>audio_tokens = ⌈frames / 4⌉</code>
            <code>TTFT = Σ stage_ms</code>
          </div>
        </div>

        <div className={styles.trace} aria-live="polite">
          <div className={styles.traceTop}>
            <div>
              <span>输入帧</span>
              <strong>{hasRun ? trace.frames : "?"}</strong>
            </div>
            <div>
              <span>连接器 Token</span>
              <strong>{hasRun ? trace.connectorTokens : "?"}</strong>
            </div>
            <div className={styles.ttft}>
              <span>公式 TTFT</span>
              <strong>{hasRun ? `${trace.ttft} ms` : "待运行"}</strong>
            </div>
          </div>

          <ol className={styles.stageList} aria-label="TTFT 流水线阶段">
            {(Object.keys(stageLabels) as StageKey[]).map((key, index) => {
              const duration = trace.durations[key];
              return (
                <li
                  key={key}
                  className={
                    hasRun && trace.bottleneck === key
                      ? styles.bottleneck
                      : undefined
                  }
                >
                  <span className={styles.stageIndex}>
                    {String(index + 1).padStart(2, "0")}
                  </span>
                  <div className={styles.stageBody}>
                    <div className={styles.stageMeta}>
                      <span>{stageLabels[key]}</span>
                      <strong>{hasRun ? `${duration} ms` : "—"}</strong>
                    </div>
                    <div className={styles.track} aria-hidden="true">
                      <span
                        style={
                          {
                            "--stage-width": hasRun
                              ? `${Math.max(4, (duration / maxDuration) * 100)}%`
                              : "0%",
                          } as CSSProperties
                        }
                      />
                    </div>
                  </div>
                </li>
              );
            })}
          </ol>

          <div className={styles.maskPanel}>
            <div className={styles.maskTitle}>
              <strong>自回归目标的 loss mask</strong>
              <span>1 才进入交叉熵</span>
            </div>
            <div className={styles.mask} aria-label="loss mask 示例">
              <div data-mask="0">
                <span>音频</span>
                <b>0</b>
              </div>
              <div data-mask="0">
                <span>用户</span>
                <b>0</b>
              </div>
              <div data-mask="1">
                <span>助手</span>
                <b>1</b>
              </div>
              <div data-mask="1">
                <span>EOS</span>
                <b>1</b>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className={styles.prediction}>
        <fieldset>
          <legend>预测 1：哪一阶段会成为公式瓶颈？</legend>
          <div className={styles.choiceRow}>
            {(Object.keys(stageLabels) as StageKey[]).map((key) => (
              <button
                type="button"
                key={key}
                aria-pressed={bottleneckPrediction === key}
                onClick={() => {
                  setBottleneckPrediction(key);
                  invalidate();
                }}
              >
                {stageLabels[key]}
              </button>
            ))}
          </div>
        </fieldset>
        <fieldset>
          <legend>预测 2：标准 SFT 中谁承担回答 loss？</legend>
          <div className={styles.choiceRow}>
            <button
              type="button"
              aria-pressed={lossPrediction === "user"}
              onClick={() => {
                setLossPrediction("user");
                invalidate();
              }}
            >
              用户输入
            </button>
            <button
              type="button"
              aria-pressed={lossPrediction === "assistant"}
              onClick={() => {
                setLossPrediction("assistant");
                invalidate();
              }}
            >
              助手目标
            </button>
          </div>
        </fieldset>
        <button
          type="button"
          className={styles.run}
          disabled={!bottleneckPrediction || !lossPrediction}
          onClick={runTrace}
        >
          运行追踪器
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
            ? "先提交两项预测，再揭示公式结果。"
            : gatePassed
              ? `你正确定位了 ${stageLabels[trace.bottleneck]}，也读对了 loss mask。`
              : `有一项预测不符。公式瓶颈由四个 stage_ms 的最大值决定；loss 只监督目标侧。`}
        </span>
      </div>
    </section>
  );
}
