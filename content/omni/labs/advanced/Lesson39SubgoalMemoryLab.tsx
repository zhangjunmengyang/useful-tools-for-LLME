"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import {
  numberFrom,
  stringFrom,
  useCompletionGate,
} from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson39SubgoalMemoryLab.module.css";

const SKILLS = [
  "open_drawer",
  "pick_blue",
  "place_in_drawer",
  "close_drawer",
] as const;

type Skill = (typeof SKILLS)[number];

type Prediction =
  | "window-replays-first"
  | "pop-replays-first"
  | "both-skip-first"
  | "neither-retries-fail";

const SKILL_LABEL: Record<Skill, string> = {
  open_drawer: "开抽屉",
  pick_blue: "抓蓝块",
  place_in_drawer: "放入抽屉",
  close_drawer: "关抽屉",
};

const PREDICTIONS: { value: Prediction; label: string }[] = [
  {
    value: "window-replays-first",
    label: "窗口会重做已成功的第一步，pop 只重试失败步",
  },
  {
    value: "pop-replays-first",
    label: "pop 会重做第一步，窗口只重试失败步",
  },
  {
    value: "both-skip-first",
    label: "两种协议都跳过第一步，只动失败步",
  },
  {
    value: "neither-retries-fail",
    label: "两种协议都从任务头重跑整条链",
  },
];

type Trace = {
  executed: Skill[];
  committed: Skill[];
  firstCount: number;
  failedCount: number;
  replayedCommitted: boolean;
  retriedFailedOnly: boolean;
  stoppedOnDelta: boolean;
};

function simulateStack(failIndex: number, retryBudget: number): Trace {
  const executed: Skill[] = [];
  const committed: Skill[] = [];
  let retries = 0;
  let index = 0;
  while (index < SKILLS.length && executed.length < 12) {
    const skill = SKILLS[index];
    executed.push(skill);
    const isFail = index === failIndex && retries === 0;
    if (isFail) {
      retries += 1;
      if (retries >= retryBudget) break;
      continue;
    }
    committed.push(skill);
    index += 1;
    retries = 0;
  }
  const failedSkill = SKILLS[failIndex];
  const firstAfter = executed.slice(1);
  return {
    executed,
    committed,
    firstCount: executed.filter((item) => item === SKILLS[0]).length,
    failedCount: executed.filter((item) => item === failedSkill).length,
    replayedCommitted: firstAfter.includes(SKILLS[0]),
    retriedFailedOnly:
      executed.filter((item) => item === SKILLS[0]).length === 1 &&
      executed.filter((item) => item === failedSkill).length >= 2,
    stoppedOnDelta: false,
  };
}

function simulateWindow(failIndex: number): Trace {
  const executed: Skill[] = [];
  const committed: Skill[] = [];
  let cursor = 0;
  let failedOnce = false;
  let stoppedOnDelta = false;
  while (cursor < SKILLS.length && executed.length < 12) {
    const skill = SKILLS[cursor];
    executed.push(skill);
    const forcedFail = cursor === failIndex && !failedOnce;
    if (forcedFail) {
      failedOnce = true;
      cursor = 0;
      continue;
    }
    if (skill === SKILLS[0] && committed.includes(SKILLS[0])) {
      stoppedOnDelta = true;
      break;
    }
    committed.push(skill);
    cursor += 1;
  }
  const failedSkill = SKILLS[failIndex];
  return {
    executed,
    committed,
    firstCount: executed.filter((item) => item === SKILLS[0]).length,
    failedCount: executed.filter((item) => item === failedSkill).length,
    replayedCommitted: executed.filter((item) => item === SKILLS[0]).length >= 2,
    retriedFailedOnly: false,
    stoppedOnDelta,
  };
}

export function Lesson39SubgoalMemoryLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    failStep: numberFrom(initialState, "failStep", 2, 2, 3),
    retryBudget: numberFrom(initialState, "retryBudget", 2, 1, 3),
    windowTokens: numberFrom(initialState, "windowTokens", 48, 32, 128),
    prediction: stringFrom(initialState, "prediction", "") as Prediction | "",
    ack: stringFrom(initialState, "ack", ""),
  };
  const [failStep, setFailStep] = useState(Math.round(defaults.failStep));
  const [retryBudget, setRetryBudget] = useState(
    Math.round(defaults.retryBudget),
  );
  const [windowTokens, setWindowTokens] = useState(
    Math.round(defaults.windowTokens),
  );
  const [prediction, setPrediction] = useState<Prediction | "">(
    PREDICTIONS.some((item) => item.value === defaults.prediction)
      ? defaults.prediction
      : "",
  );
  const [ack, setAck] = useState(defaults.ack === "yes");
  const [ran, setRan] = useState(false);

  const failIndex = failStep - 1;
  const simulation = useMemo(() => {
    const stack = simulateStack(failIndex, retryBudget);
    const windowTrace = simulateWindow(failIndex);
    return { stack, window: windowTrace };
  }, [failIndex, retryBudget]);

  const revealed = ran && prediction !== "";
  const structureHeld =
    simulation.window.replayedCommitted &&
    simulation.stack.firstCount === 1 &&
    simulation.stack.retriedFailedOnly &&
    simulation.window.firstCount >= 2;

  const passed =
    revealed &&
    prediction === "window-replays-first" &&
    ack &&
    structureHeld &&
    failStep === 2 &&
    retryBudget >= 2;

  const completion = useMemo(
    () => ({
      lessonId: 39,
      failStep,
      retryBudget,
      windowTokens,
      prediction,
      ack,
      stackFirstCount: simulation.stack.firstCount,
      windowFirstCount: simulation.window.firstCount,
      stackExecuted: simulation.stack.executed,
      windowExecuted: simulation.window.executed,
      structureHeld,
    }),
    [
      ack,
      failStep,
      prediction,
      retryBudget,
      simulation,
      structureHeld,
      windowTokens,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function reset() {
    setFailStep(2);
    setRetryBudget(2);
    setWindowTokens(48);
    setPrediction("");
    setAck(false);
    setRan(false);
  }

  return (
    <LabFrame
      lesson="39"
      title="四步任务第二步失败：窗口回放还是 pop 栈"
      description="教学模拟，不是模型输出。先选预测，再揭晓两侧执行表。塞进窗口会重做已成功的第一步；pop 只重试失败步。"
    >
      <div className={styles.workspace}>
        <aside className={styles.controls}>
          <h3>夹具参数</h3>
          <label>
            <span>
              失败步
              <output>{failStep}</output>
            </span>
            <input
              type="range"
              min={2}
              max={3}
              step={1}
              value={failStep}
              onChange={(event) => {
                setFailStep(Number(event.target.value));
                setRan(false);
              }}
            />
          </label>
          <label>
            <span>
              重试上界 Rmax
              <output>{retryBudget}</output>
            </span>
            <input
              type="range"
              min={1}
              max={3}
              step={1}
              value={retryBudget}
              onChange={(event) => {
                setRetryBudget(Number(event.target.value));
                setRan(false);
              }}
            />
          </label>
          <label>
            <span>
              窗口 token T
              <output>{windowTokens}</output>
            </span>
            <input
              type="range"
              min={32}
              max={128}
              step={8}
              value={windowTokens}
              onChange={(event) => {
                setWindowTokens(Number(event.target.value));
                setRan(false);
              }}
            />
          </label>
          <p className={styles.note}>
            T 只改变窗口账本上的 token 计数，不改变会不会重做第一步。k
            在提交第一步后为 3，与 T 不是同一个量。
          </p>
        </aside>
        <div className={styles.stage}>
          <ol className={styles.pipeline}>
            {SKILLS.map((skill, index) => (
              <li
                key={skill}
                className={
                  index === failIndex ? styles.failStep : styles.okStep
                }
              >
                <b>{index + 1}</b>
                <span>{SKILL_LABEL[skill]}</span>
              </li>
            ))}
          </ol>
          <div className={styles.desks}>
            <article className={styles.desk}>
              <header>
                <b>窗口回放</b>
                <span>拼接整段指令</span>
              </header>
              <p className={styles.meta}>T = {windowTokens} token</p>
              <ol className={styles.trace}>
                {(revealed ? simulation.window.executed : []).map(
                  (skill, index) => (
                    <li
                      key={`w-${skill}-${index}`}
                      data-repeat={
                        skill === SKILLS[0] && index > 0 ? "yes" : "no"
                      }
                    >
                      {SKILL_LABEL[skill]}
                    </li>
                  ),
                )}
                {!revealed ? <li className={styles.pending}>待揭晓</li> : null}
              </ol>
              <dl>
                <div>
                  <dt>第一步次数</dt>
                  <dd>{revealed ? simulation.window.firstCount : "—"}</dd>
                </div>
                <div>
                  <dt>状态差</dt>
                  <dd>
                    {revealed
                      ? simulation.window.stoppedOnDelta
                        ? "二次开抽屉失败"
                        : "未触发"
                      : "—"}
                  </dd>
                </div>
              </dl>
            </article>
            <article className={styles.desk}>
              <header>
                <b>子目标栈 pop</b>
                <span>只动栈顶</span>
              </header>
              <p className={styles.meta}>k 提交后 = 3</p>
              <ol className={styles.trace}>
                {(revealed ? simulation.stack.executed : []).map(
                  (skill, index) => (
                    <li key={`s-${skill}-${index}`}>{SKILL_LABEL[skill]}</li>
                  ),
                )}
                {!revealed ? <li className={styles.pending}>待揭晓</li> : null}
              </ol>
              <dl>
                <div>
                  <dt>第一步次数</dt>
                  <dd>{revealed ? simulation.stack.firstCount : "—"}</dd>
                </div>
                <div>
                  <dt>失败步次数</dt>
                  <dd>{revealed ? simulation.stack.failedCount : "—"}</dd>
                </div>
              </dl>
            </article>
          </div>
        </div>
      </div>
      <div className={styles.predict}>
        <fieldset>
          <legend>先预测：第二步失败后哪边会重做已成功的第一步？</legend>
          {PREDICTIONS.map((item) => (
            <label key={item.value}>
              <input
                type="radio"
                name="lesson39-prediction"
                value={item.value}
                checked={prediction === item.value}
                onChange={() => {
                  setPrediction(item.value);
                  setRan(false);
                }}
              />
              <span>{item.label}</span>
            </label>
          ))}
        </fieldset>
        <label className={styles.ack}>
          <input
            type="checkbox"
            checked={ack}
            onChange={(event) => setAck(event.target.checked)}
          />
          <span>教学模拟，不会把这里的次数写成 CALVIN 成功率</span>
        </label>
        <div className={styles.actions}>
          <button className={styles.reset} type="button" onClick={reset}>
            重置
          </button>
          <button
            className={styles.run}
            type="button"
            disabled={prediction === ""}
            onClick={() => setRan(true)}
          >
            揭晓对照
          </button>
        </div>
        {ran && prediction !== "window-replays-first" ? (
          <p className={styles.feedback}>
            再看两侧第一步计数。窗口列应大于 1，栈列应等于 1。
          </p>
        ) : null}
        {revealed && failStep !== 2 ? (
          <p className={styles.feedback}>
            验收默认失败步为第 2 步。结构在第 3 步同样成立，但请把滑条拨回 2
            再交卷。
          </p>
        ) : null}
        {revealed && retryBudget < 2 ? (
          <p className={styles.feedback}>
            重试上界为 1 时栈臂会在失败步中止。把 Rmax 调到 2 才能看到只重试失败步。
          </p>
        ) : null}
      </div>
      <Gate passed={passed}>
        {passed
          ? "窗口回放重做了已提交的第一步，pop 只重试失败步。T 与 k 分开记账。"
          : "先选预测，失败步保持为 2，Rmax 至少为 2，揭晓后核对第一步计数，并确认这是教学模拟。"}
      </Gate>
    </LabFrame>
  );
}
