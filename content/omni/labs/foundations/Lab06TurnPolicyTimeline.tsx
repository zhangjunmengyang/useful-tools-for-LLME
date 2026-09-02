"use client";

import { useMemo, useState } from "react";
import type { CSSProperties } from "react";
import styles from "./Lab06TurnPolicyTimeline.module.css";
import type { FoundationLabProps } from "./types";
import { initialNumber, initialString } from "./types";

type PolicyState = "LISTEN" | "THINK" | "SPEAK" | "INTERRUPTED";
type ScenarioKey = "clean" | "hesitation" | "barge";

const scenarios: Record<
  ScenarioKey,
  { name: string; description: string; energies: number[] }
> = {
  clean: {
    name: "干净轮次",
    description: "用户说完后持续静音",
    energies: [0.72, 0.81, 0.66, 0.08, 0.05, 0.03, 0.02, 0.02, 0.03],
  },
  hesitation: {
    name: "犹豫停顿",
    description: "短暂停顿后继续补充",
    energies: [0.7, 0.1, 0.68, 0.12, 0.62, 0.08, 0.06, 0.04, 0.03],
  },
  barge: {
    name: "抢话打断",
    description: "助手开口后用户再次发声",
    energies: [0.75, 0.8, 0.65, 0.05, 0.03, 0.02, 0.02, 0.7, 0.74],
  },
};

const stateLabels: Record<PolicyState, string> = {
  LISTEN: "监听",
  THINK: "思考",
  SPEAK: "发言",
  INTERRUPTED: "被打断",
};

function simulatePolicy(
  energies: number[],
  threshold: number,
  commitMs: number,
) {
  let state: PolicyState = "LISTEN";
  let speechSeen = false;
  let silenceMs = 0;
  let thinkMs = 0;
  const trace: Array<{
    energy: number;
    state: PolicyState;
    reason: string;
    silenceMs: number;
  }> = [];

  energies.forEach((energy) => {
    let reason = energy >= threshold ? "检测到语音" : "低于阈值";
    if (state === "LISTEN") {
      if (energy >= threshold) {
        speechSeen = true;
        silenceMs = 0;
      } else if (speechSeen) {
        silenceMs += 200;
        reason = `累计静音 ${silenceMs} ms`;
        if (silenceMs >= commitMs) {
          state = "THINK";
          thinkMs = 0;
          reason = "达到 turn commit";
        }
      }
    } else if (state === "THINK") {
      if (energy >= threshold) {
        state = "LISTEN";
        speechSeen = true;
        silenceMs = 0;
        reason = "用户续说，撤回 commit";
      } else {
        thinkMs += 200;
        reason = `思考 ${thinkMs} ms`;
        if (thinkMs >= 400) {
          state = "SPEAK";
          reason = "思考预算完成";
        }
      }
    } else if (state === "SPEAK" && energy >= threshold) {
      state = "INTERRUPTED";
      reason = "发言期检测到用户语音";
    } else if (state === "INTERRUPTED") {
      reason = "保持中断，等待策略重置";
    }
    trace.push({ energy, state, reason, silenceMs });
  });

  return { finalState: state, trace };
}

export function Lab06TurnPolicyTimeline({
  onComplete,
  initialState,
}: FoundationLabProps) {
  const defaults = {
    scenario: initialString(
      initialState,
      "scenario",
      ["clean", "hesitation", "barge"] as const,
      "clean",
    ),
    threshold: initialNumber(initialState, "threshold", 0.55),
    commitMs: initialNumber(initialState, "commitMs", 400),
  };
  const [scenario, setScenario] = useState<ScenarioKey>(defaults.scenario);
  const [threshold, setThreshold] = useState(defaults.threshold);
  const [commitMs, setCommitMs] = useState(defaults.commitMs);
  const [prediction, setPrediction] = useState<PolicyState | null>(null);
  const [hasRun, setHasRun] = useState(false);

  const result = useMemo(
    () => simulatePolicy(scenarios[scenario].energies, threshold, commitMs),
    [commitMs, scenario, threshold],
  );
  const gatePassed = hasRun && prediction === result.finalState;

  function invalidate() {
    setHasRun(false);
  }

  function runPolicy() {
    setHasRun(true);
    if (prediction === result.finalState) {
      onComplete?.({
        scenario,
        threshold,
        commitMs,
        finalState: result.finalState,
        states: result.trace.map((item) => item.state),
      });
    }
  }

  function reset() {
    setScenario(defaults.scenario);
    setThreshold(defaults.threshold);
    setCommitMs(defaults.commitMs);
    setPrediction(null);
    setHasRun(false);
  }

  const states: PolicyState[] = [
    "LISTEN",
    "THINK",
    "SPEAK",
    "INTERRUPTED",
  ];

  return (
    <section className={styles.lab} aria-labelledby="lab06-title">
      <header className={styles.header}>
        <div>
          <div className={styles.tags}>
            <span>教学模拟</span>
            <span>确定性状态机</span>
          </div>
          <h3 id="lab06-title">“轮到谁说”必须是一台看得见的机器</h3>
          <p>
            调整 VAD 与静音提交门槛，逐个 200 ms 时间片执行 turn policy，避免把轮次切换藏进 prompt。
          </p>
        </div>
        <button className={styles.reset} type="button" onClick={reset}>
          重置状态机
        </button>
      </header>

      <div className={styles.stateRail} aria-label="轮次策略状态">
        {states.map((state, index) => (
          <div
            key={state}
            className={[
              styles.state,
              hasRun && result.trace.some((item) => item.state === state)
                ? styles.visited
                : "",
              hasRun && result.finalState === state ? styles.final : "",
            ].join(" ")}
          >
            <span>{String(index + 1).padStart(2, "0")}</span>
            <strong>{stateLabels[state]}</strong>
            <small>{state}</small>
          </div>
        ))}
      </div>

      <div className={styles.scenarios}>
        {(Object.keys(scenarios) as ScenarioKey[]).map((key) => (
          <button
            key={key}
            type="button"
            aria-pressed={scenario === key}
            onClick={() => {
              setScenario(key);
              invalidate();
            }}
          >
            <strong>{scenarios[key].name}</strong>
            <span>{scenarios[key].description}</span>
          </button>
        ))}
      </div>

      <div className={styles.tuning}>
        <label>
          <span>
            VAD 阈值 <strong>{threshold.toFixed(2)}</strong>
          </span>
          <input
            type="range"
            min="0.35"
            max="0.75"
            step="0.05"
            value={threshold}
            onChange={(event) => {
              setThreshold(Number(event.target.value));
              invalidate();
            }}
          />
        </label>
        <label>
          <span>Turn commit 静音</span>
          <select
            value={commitMs}
            onChange={(event) => {
              setCommitMs(Number(event.target.value));
              invalidate();
            }}
          >
            <option value="200">200 ms</option>
            <option value="400">400 ms</option>
            <option value="600">600 ms</option>
          </select>
        </label>
        <div className={styles.rule}>
          <span>转移规则</span>
          <code>silence ≥ commit → THINK</code>
          <code>THINK 400 ms → SPEAK</code>
          <code>SPEAK ∧ voice → INTERRUPTED</code>
        </div>
      </div>

      <div className={styles.timeline} aria-label="九个 200 毫秒事件片段">
        <div className={styles.timeHead}>
          <span>t</span>
          {scenarios[scenario].energies.map((_, index) => (
            <span key={index}>{index * 200}</span>
          ))}
        </div>
        <div className={styles.energyRow}>
          <span>能量</span>
          {scenarios[scenario].energies.map((energy, index) => (
            <div className={styles.energyCell} key={index}>
              <i
                style={{ "--energy": `${energy * 100}%` } as CSSProperties}
                className={energy >= threshold ? styles.voice : undefined}
              />
              <b>{energy.toFixed(2)}</b>
            </div>
          ))}
        </div>
        <div className={styles.thresholdLine}>
          <span>threshold {threshold.toFixed(2)}</span>
        </div>
        <div className={styles.traceRow}>
          <span>状态</span>
          {scenarios[scenario].energies.map((_, index) => {
            const item = result.trace[index];
            return (
              <div
                key={index}
                className={
                  hasRun ? styles[`state${item.state}`] : styles.hiddenState
                }
                title={hasRun ? item.reason : "运行后揭示"}
              >
                {hasRun ? stateLabels[item.state] : "?"}
              </div>
            );
          })}
        </div>
      </div>

      <div className={styles.predict}>
        <fieldset>
          <legend>先预测：9 个时间片后，策略停在哪个状态？</legend>
          <div>
            {states.map((state) => (
              <button
                key={state}
                type="button"
                aria-pressed={prediction === state}
                onClick={() => {
                  setPrediction(state);
                  invalidate();
                }}
              >
                {stateLabels[state]}
              </button>
            ))}
          </div>
        </fieldset>
        <button
          type="button"
          className={styles.run}
          disabled={!prediction}
          onClick={runPolicy}
        >
          执行状态转移
        </button>
      </div>

      {hasRun && (
        <div className={styles.eventLog} aria-live="polite">
          {result.trace.map((item, index) => (
            <div key={index}>
              <b>{index * 200} ms</b>
              <span>{stateLabels[item.state]}</span>
              <small>{item.reason}</small>
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
        <strong>{gatePassed ? "验收已通过" : "完成验收"}</strong>
        <span>
          {!hasRun
            ? "从左到右累计静音，并在每次用户续说时清零。"
            : gatePassed
              ? `正确，终态是 ${stateLabels[result.finalState]}。你已走完可审计的 turn-policy 路径。`
              : `终态是 ${stateLabels[result.finalState]}。打开事件日志，找到预测与转移规则分叉的时间片。`}
        </span>
      </div>
    </section>
  );
}
