"use client";

import { useMemo, useState } from "react";
import styles from "./Lab07FullDuplexPolicy.module.css";
import type { FoundationLabProps } from "./types";
import { initialNumber, initialString } from "./types";

type DuplexAction = "CONTINUE" | "PAUSE" | "REPLAN";
type ScenarioKey = "meeting" | "correction" | "noisy";

type SignalEvent = {
  at: number;
  utterance: string;
  vad: number;
  conflict: number;
};

const scenarios: Record<
  ScenarioKey,
  { name: string; prompt: string; events: SignalEvent[] }
> = {
  meeting: {
    name: "会议讲解",
    prompt: "解释为什么需要流式 cache",
    events: [
      { at: 320, utterance: "键盘声", vad: 0.22, conflict: 0.05 },
      { at: 640, utterance: "嗯", vad: 0.68, conflict: 0.16 },
      { at: 960, utterance: "等等，先解释延迟", vad: 0.76, conflict: 0.82 },
      { at: 1280, utterance: "环境声", vad: 0.31, conflict: 0.08 },
      { at: 1600, utterance: "对，继续", vad: 0.71, conflict: 0.28 },
    ],
  },
  correction: {
    name: "事实纠正",
    prompt: "给出一次旅行建议",
    events: [
      { at: 300, utterance: "嗯", vad: 0.64, conflict: 0.12 },
      { at: 600, utterance: "不是东京，是大阪", vad: 0.84, conflict: 0.94 },
      { at: 900, utterance: "碰杯声", vad: 0.24, conflict: 0.04 },
      { at: 1200, utterance: "预算也要低一点", vad: 0.73, conflict: 0.78 },
    ],
  },
  noisy: {
    name: "嘈杂客厅",
    prompt: "口述配置一台训练机器",
    events: [
      { at: 260, utterance: "电视背景声", vad: 0.42, conflict: 0.08 },
      { at: 520, utterance: "嗯哼", vad: 0.61, conflict: 0.18 },
      { at: 780, utterance: "不要消费卡", vad: 0.8, conflict: 0.87 },
      { at: 1040, utterance: "风扇声", vad: 0.38, conflict: 0.09 },
      { at: 1300, utterance: "八张卡", vad: 0.76, conflict: 0.72 },
    ],
  },
};

const actionMeta: Record<
  DuplexAction,
  { label: string; description: string }
> = {
  CONTINUE: { label: "继续", description: "继续输出下一个 token" },
  PAUSE: { label: "暂停", description: "保留计划与解码游标" },
  REPLAN: { label: "重规划", description: "废弃旧回答分支" },
};

function classify(
  event: SignalEvent,
  speechThreshold: number,
  conflictThreshold: number,
): DuplexAction {
  if (event.vad < speechThreshold) return "CONTINUE";
  if (event.conflict < conflictThreshold) return "PAUSE";
  return "REPLAN";
}

export function Lab07FullDuplexPolicy({
  onComplete,
  initialState,
}: FoundationLabProps) {
  const defaults = {
    scenario: initialString(
      initialState,
      "scenario",
      ["meeting", "correction", "noisy"] as const,
      "meeting",
    ),
    speechThreshold: initialNumber(initialState, "speechThreshold", 0.55),
    conflictThreshold: initialNumber(initialState, "conflictThreshold", 0.45),
  };
  const [scenario, setScenario] = useState<ScenarioKey>(defaults.scenario);
  const [speechThreshold, setSpeechThreshold] = useState(
    defaults.speechThreshold,
  );
  const [conflictThreshold, setConflictThreshold] = useState(
    defaults.conflictThreshold,
  );
  const [answers, setAnswers] = useState<
    Array<{ prediction: DuplexAction; actual: DuplexAction }>
  >([]);
  const [prediction, setPrediction] = useState<DuplexAction | null>(null);
  const [planVersion, setPlanVersion] = useState(1);
  const [assistantCursor, setAssistantCursor] = useState(0);
  const [assistantMode, setAssistantMode] =
    useState<"speaking" | "paused" | "replanning">("speaking");

  const events = scenarios[scenario].events;
  const currentEvent = events[answers.length];
  const allDone = answers.length === events.length;
  const gatePassed =
    allDone && answers.every((answer) => answer.prediction === answer.actual);
  const correctCount = answers.filter(
    (answer) => answer.prediction === answer.actual,
  ).length;
  const responseTokens = useMemo(
    () =>
      planVersion === 1
        ? ["先", "固定", "chunk", "边界", "再", "缓存", "历史", "状态"]
        : ["收到", "新约束", "改为", "先回答", "用户", "刚才", "的问题"],
    [planVersion],
  );

  function resetRun(keepSettings = true) {
    if (!keepSettings) {
      setScenario(defaults.scenario);
      setSpeechThreshold(defaults.speechThreshold);
      setConflictThreshold(defaults.conflictThreshold);
    }
    setAnswers([]);
    setPrediction(null);
    setPlanVersion(1);
    setAssistantCursor(0);
    setAssistantMode("speaking");
  }

  function runNext() {
    if (!currentEvent || !prediction) return;
    const actual = classify(
      currentEvent,
      speechThreshold,
      conflictThreshold,
    );
    const nextAnswers = [...answers, { prediction, actual }];
    setAnswers(nextAnswers);
    setPrediction(null);

    if (actual === "CONTINUE") {
      setAssistantMode("speaking");
      setAssistantCursor((cursor) =>
        Math.min(responseTokens.length, cursor + 1),
      );
    } else if (actual === "PAUSE") {
      setAssistantMode("paused");
    } else {
      setAssistantMode("replanning");
      setPlanVersion((version) => version + 1);
      setAssistantCursor(0);
    }

    const passed =
      nextAnswers.length === events.length &&
      nextAnswers.every((answer) => answer.prediction === answer.actual);
    if (passed) {
      onComplete?.({
        scenario,
        speechThreshold,
        conflictThreshold,
        actions: nextAnswers.map((answer) => answer.actual),
        planVersion:
          1 +
          nextAnswers.filter((answer) => answer.actual === "REPLAN").length,
      });
    }
  }

  const actions: DuplexAction[] = ["CONTINUE", "PAUSE", "REPLAN"];

  return (
    <section className={styles.lab} aria-labelledby="lab07-title">
      <header className={styles.header}>
        <div>
          <div className={styles.tags}>
            <span>教学模拟</span>
            <span>旗舰交互</span>
          </div>
          <h3 id="lab07-title">全双工策略台：Continue、Pause，还是 Replan？</h3>
          <p>
            对每个重叠事件先选择处理动作，再运行明确的话轮策略，检查回答游标与计划版本如何变化。
          </p>
        </div>
        <div className={styles.score} aria-live="polite">
          <span>POLICY GATE</span>
          <strong>
            {correctCount}/{events.length}
          </strong>
          <small>逐事件正确</small>
        </div>
      </header>

      <div className={styles.setup}>
        <label>
          <span>场景</span>
          <select
            value={scenario}
            onChange={(event) => {
              setScenario(event.target.value as ScenarioKey);
              resetRun();
            }}
          >
            {(Object.keys(scenarios) as ScenarioKey[]).map((key) => (
              <option value={key} key={key}>
                {scenarios[key].name}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>
            Speech 阈值 <strong>{speechThreshold.toFixed(2)}</strong>
          </span>
          <input
            type="range"
            min="0.35"
            max="0.75"
            step="0.05"
            value={speechThreshold}
            onChange={(event) => {
              setSpeechThreshold(Number(event.target.value));
              resetRun();
            }}
          />
        </label>
        <label>
          <span>
            Conflict 阈值 <strong>{conflictThreshold.toFixed(2)}</strong>
          </span>
          <input
            type="range"
            min="0.25"
            max="0.85"
            step="0.05"
            value={conflictThreshold}
            onChange={(event) => {
              setConflictThreshold(Number(event.target.value));
              resetRun();
            }}
          />
        </label>
        <button type="button" onClick={() => resetRun(false)}>
          重置全部
        </button>
      </div>

      <div className={styles.policyMatrix}>
        <div className={styles.matrixTitle}>
          <span>确定性 policy</span>
          <code>
            vad &lt; S ? Continue : conflict &lt; C ? Pause : Replan
          </code>
        </div>
        <div className={styles.matrixGrid}>
          <div className={styles.matrixCorner}>signal</div>
          <div className={styles.matrixHead}>低 conflict</div>
          <div className={styles.matrixHead}>高 conflict</div>
          <div className={styles.matrixHead}>低 speech</div>
          <div className={styles.continue}>Continue</div>
          <div className={styles.continue}>Continue</div>
          <div className={styles.matrixHead}>高 speech</div>
          <div className={styles.pause}>Pause</div>
          <div className={styles.replan}>Replan</div>
        </div>
        <p>vad / conflict 是本场景给定的教学输入，不是模型实测准确率。</p>
      </div>

      <div className={styles.duplex}>
        <div className={styles.assistantLane}>
          <div className={styles.laneHead}>
            <div>
              <span>ASSISTANT OUT</span>
              <strong>
                plan v{planVersion} · {assistantMode}
              </strong>
            </div>
            <small>任务：{scenarios[scenario].prompt}</small>
          </div>
          <div className={styles.tokenStream} aria-label="助手回答游标">
            {responseTokens.map((token, index) => (
              <span
                key={`${planVersion}-${index}`}
                className={[
                  index < assistantCursor ? styles.spoken : "",
                  index === assistantCursor && !allDone ? styles.cursor : "",
                ].join(" ")}
              >
                {token}
              </span>
            ))}
          </div>
        </div>

        <div className={styles.eventLane}>
          <div className={styles.laneHead}>
            <div>
              <span>USER IN · OVERLAP EVENTS</span>
              <strong>
                {allDone ? "场景已结束" : `事件 ${answers.length + 1}`}
              </strong>
            </div>
            <small>每张卡在回答流进行时到达</small>
          </div>
          <div className={styles.events}>
            {events.map((event, index) => {
              const answer = answers[index];
              const isCurrent = index === answers.length;
              return (
                <article
                  key={`${event.at}-${event.utterance}`}
                  className={[
                    answer ? styles.resolved : "",
                    isCurrent ? styles.current : "",
                  ].join(" ")}
                  aria-current={isCurrent ? "step" : undefined}
                >
                  <div className={styles.eventTop}>
                    <time>{event.at} ms</time>
                    {answer && (
                      <b className={styles[`action${answer.actual}`]}>
                        {actionMeta[answer.actual].label}
                      </b>
                    )}
                  </div>
                  <strong>“{event.utterance}”</strong>
                  <div className={styles.signals}>
                    <span>
                      vad <b>{event.vad.toFixed(2)}</b>
                    </span>
                    <span>
                      conflict <b>{event.conflict.toFixed(2)}</b>
                    </span>
                  </div>
                  {answer && (
                    <small>
                      预测 {actionMeta[answer.prediction].label} ·{" "}
                      {answer.prediction === answer.actual ? "正确" : "不符"}
                    </small>
                  )}
                </article>
              );
            })}
          </div>
        </div>
      </div>

      <div className={styles.console}>
        {!allDone && currentEvent ? (
          <>
            <div className={styles.currentSignal}>
              <span>当前重叠输入</span>
              <strong>“{currentEvent.utterance}”</strong>
              <code>
                vad={currentEvent.vad.toFixed(2)} · conflict=
                {currentEvent.conflict.toFixed(2)}
              </code>
            </div>
            <fieldset>
              <legend>先预测动作</legend>
              <div>
                {actions.map((action) => (
                  <button
                    key={action}
                    type="button"
                    aria-pressed={prediction === action}
                    onClick={() => setPrediction(action)}
                  >
                    <b>{actionMeta[action].label}</b>
                    <span>{actionMeta[action].description}</span>
                  </button>
                ))}
              </div>
            </fieldset>
            <button
              type="button"
              className={styles.run}
              disabled={!prediction}
              onClick={runNext}
            >
              运行本事件
            </button>
          </>
        ) : (
          <div className={styles.finished}>
            <div>
              <strong>场景运行完毕</strong>
              <span>
                {correctCount === events.length
                  ? "每次策略判断都与公式一致。"
                  : `答对 ${correctCount}/${events.length}，保留阈值可再挑战。`}
              </span>
            </div>
            <button type="button" onClick={() => resetRun()}>
              用同一配置再挑战
            </button>
          </div>
        )}
      </div>

      <div
        className={`${styles.gate} ${
          allDone ? (gatePassed ? styles.passGate : styles.retryGate) : ""
        }`}
        role="status"
      >
        <strong>{gatePassed ? "验收已通过" : "完成验收"}</strong>
        <span>
          {!allDone
            ? `必须先预测、再运行全部 ${events.length} 个重叠事件。`
            : gatePassed
              ? "你已把全双工拆成可执行、可回放的策略，而不是一个抽象能力标签。"
              : "完整运行已完成，但至少一个事件分支错误。按阈值矩阵逐项重试。"}
        </span>
      </div>
    </section>
  );
}
