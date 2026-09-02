"use client";

import { useEffect, useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import {
  numberFrom,
  round,
  stringFrom,
  useCompletionGate,
} from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson29DualClockLab.module.css";

type HoldPolicy = "consume-subgoal" | "repeat-action";
type InjectMode = "none" | "floor" | "wrong-shelf";
type Prediction = "stale-hold" | "self-replan" | "hard-stop";

type Point = { x: number; y: number };

type SubgoalId =
  | "approach"
  | "grasp"
  | "lift"
  | "transport"
  | "place-shelf"
  | "place-floor"
  | "wrong-shelf";

const SUBGOAL_LABEL: Record<SubgoalId, string> = {
  approach: "靠近杯子",
  grasp: "抓住杯子",
  lift: "抬离桌面",
  transport: "送向货架",
  "place-shelf": "放到第二层",
  "place-floor": "放到地面箱",
  "wrong-shelf": "放到错误层",
};

const SEQUENCE: SubgoalId[] = [
  "approach",
  "grasp",
  "lift",
  "transport",
  "place-shelf",
];

const WAYPOINTS: Record<SubgoalId, Point> = {
  approach: { x: 78, y: 128 },
  grasp: { x: 92, y: 132 },
  lift: { x: 92, y: 78 },
  transport: { x: 268, y: 48 },
  "place-shelf": { x: 318, y: 36 },
  "place-floor": { x: 318, y: 168 },
  "wrong-shelf": { x: 318, y: 88 },
};

const START: Point = { x: 36, y: 158 };
const CUP: Point = { x: 92, y: 132 };
const HORIZON = 80;
const STEP = 7.2;
const ARRIVE = 8;

const PREDICTION_LABEL: Record<Prediction, string> = {
  "stale-hold":
    "System 1 继续消费最后一条子目标（或重复最后动作），不会自己进入下一阶段",
  "self-replan": "System 1 会自己发明“放到第二层”，任务仍成功",
  "hard-stop": "System 1 立刻停机，不再消费当前子目标",
};

function distance(a: Point, b: Point) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function moveToward(from: Point, to: Point, step: number): Point {
  const span = distance(from, to);
  if (span <= step) return { ...to };
  const ratio = step / span;
  return {
    x: from.x + (to.x - from.x) * ratio,
    y: from.y + (to.y - from.y) * ratio,
  };
}

function resolveTarget(
  stage: number,
  inject: InjectMode,
  injectAt: number,
): SubgoalId {
  const current = SEQUENCE[Math.min(stage, SEQUENCE.length - 1)];
  if (stage < injectAt || inject === "none") return current;
  if (current === "transport" || current === "place-shelf") {
    return inject === "floor" ? "place-floor" : "wrong-shelf";
  }
  return current;
}

function simulate(options: {
  k: number;
  pauseStart: number;
  pauseDuration: number;
  inject: InjectMode;
  hold: HoldPolicy;
}) {
  const expire = 2 * options.k;
  const pauseEnd = options.pauseStart + options.pauseDuration;
  const injectAt = 3;
  const path: Point[] = [{ ...START }];
  const cupPath: Point[] = [{ ...CUP }];
  const subgoals: SubgoalId[] = [];
  let ee = { ...START };
  let cup = { ...CUP };
  let lastAction = { x: 0, y: 0 };
  let stage = 0;
  let lastPlan = -options.k;
  let planSteps = 0;
  let holdingCup = false;
  let staleFrom: number | null = null;
  let lastActionRepeated = 0;

  for (let tick = 0; tick < HORIZON; tick += 1) {
    const paused = tick >= options.pauseStart && tick < pauseEnd;
    if (tick % options.k === 0 && !paused) {
      lastPlan = tick;
      planSteps += 1;
      const targetId = resolveTarget(stage, options.inject, injectAt);
      if (distance(ee, WAYPOINTS[targetId]) < ARRIVE) {
        if (targetId === "grasp") holdingCup = true;
        if (targetId === "place-shelf" || targetId === "place-floor" || targetId === "wrong-shelf") {
          holdingCup = false;
        }
        if (stage < SEQUENCE.length - 1) stage += 1;
      }
    }

    const age = tick - lastPlan;
    if (age > expire && staleFrom === null) staleFrom = tick;

    const targetId = resolveTarget(stage, options.inject, injectAt);
    const target = WAYPOINTS[targetId];
    const toward = moveToward(ee, target, STEP);
    let action = { x: toward.x - ee.x, y: toward.y - ee.y };

    if (paused && options.hold === "repeat-action") {
      action = { ...lastAction };
    }

    const repeated =
      Math.abs(action.x - lastAction.x) < 0.01 &&
      Math.abs(action.y - lastAction.y) < 0.01;
    if (tick > 0 && repeated) lastActionRepeated += 1;

    ee = { x: ee.x + action.x, y: ee.y + action.y };
    lastAction = action;
    if (holdingCup) cup = { ...ee };
    path.push({ ...ee });
    cupPath.push({ ...cup });
    subgoals.push(targetId);
  }

  const placedShelf = distance(ee, WAYPOINTS["place-shelf"]) < 14 && !holdingCup;
  const withinTime = staleFrom === null;
  const success = placedShelf && withinTime && options.inject === "none";
  return {
    path,
    cupPath,
    subgoals,
    planSteps,
    controlSteps: HORIZON,
    expire,
    staleFrom,
    lastActionRepeated,
    holdingCup,
    final: ee,
    success,
  };
}

function pathLengthDelta(left: Point[], right: Point[]) {
  const n = Math.min(left.length, right.length);
  let sum = 0;
  for (let index = 0; index < n; index += 1) {
    sum += distance(left[index], right[index]);
  }
  return sum / n;
}

function polyline(points: Point[]) {
  return points.map((point) => `${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" ");
}

export function Lesson29DualClockLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    k: numberFrom(initialState, "k", 8, 4, 16),
    pauseStart: numberFrom(initialState, "pauseStart", 24, 0, 72),
    pauseDuration: numberFrom(initialState, "pauseDuration", 8, 0, 80),
    inject: stringFrom(initialState, "inject", "none") as InjectMode,
    hold: stringFrom(initialState, "hold", "consume-subgoal") as HoldPolicy,
    prediction: stringFrom(initialState, "prediction", "") as Prediction | "",
  };
  const [k, setK] = useState(defaults.k);
  const [pauseStart, setPauseStart] = useState(defaults.pauseStart);
  const [pauseDuration, setPauseDuration] = useState(defaults.pauseDuration);
  const [inject, setInject] = useState<InjectMode>(
    ["none", "floor", "wrong-shelf"].includes(defaults.inject)
      ? defaults.inject
      : "none",
  );
  const [hold, setHold] = useState<HoldPolicy>(
    defaults.hold === "repeat-action" ? "repeat-action" : "consume-subgoal",
  );
  const [prediction, setPrediction] = useState<Prediction | "">(
    defaults.prediction === "stale-hold" ||
      defaults.prediction === "self-replan" ||
      defaults.prediction === "hard-stop"
      ? defaults.prediction
      : "",
  );
  const [tick, setTick] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [ran, setRan] = useState(false);

  const baseline = useMemo(
    () =>
      simulate({
        k,
        pauseStart: HORIZON,
        pauseDuration: 0,
        inject: "none",
        hold: "consume-subgoal",
      }),
    [k],
  );
  const current = useMemo(
    () => simulate({ k, pauseStart, pauseDuration, inject, hold }),
    [hold, inject, k, pauseDuration, pauseStart],
  );

  const displayTick = ran ? tick : 0;
  const ee = current.path[Math.min(displayTick, current.path.length - 1)];
  const subgoal =
    current.subgoals[Math.min(Math.max(displayTick - 1, 0), current.subgoals.length - 1)] ??
    "approach";
  const pausedNow =
    ran && displayTick >= pauseStart && displayTick < pauseStart + pauseDuration;
  const staleNow =
    current.staleFrom !== null && displayTick >= current.staleFrom;
  const trajectoryDelta = pathLengthDelta(current.path, baseline.path);
  const lastActionRepeatRatio = current.lastActionRepeated / Math.max(1, HORIZON - 1);
  const taskFailed = !current.success;
  const pauseTooLong = pauseDuration > current.expire;
  const pathChanged = trajectoryDelta > 12;
  const finished = ran && displayTick >= HORIZON;
  const revealed = finished;

  const passed =
    revealed &&
    prediction === "stale-hold" &&
    pauseTooLong &&
    taskFailed &&
    inject !== "none" &&
    pathChanged;

  const completion = useMemo(
    () => ({
      lessonId: 29,
      k,
      pauseStart,
      pauseDuration,
      inject,
      hold,
      prediction,
      planSteps: current.planSteps,
      controlSteps: current.controlSteps,
      expireSteps: current.expire,
      staleFrom: current.staleFrom,
      taskFailed,
      trajectoryDelta: round(trajectoryDelta, 3),
      lastActionRepeatRatio: round(lastActionRepeatRatio, 3),
    }),
    [
      current.controlSteps,
      current.expire,
      current.planSteps,
      current.staleFrom,
      hold,
      inject,
      k,
      lastActionRepeatRatio,
      pauseDuration,
      pauseStart,
      prediction,
      taskFailed,
      trajectoryDelta,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  useEffect(() => {
    if (!playing) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setTick(HORIZON);
      setPlaying(false);
      return;
    }
    const timer = window.setInterval(() => {
      setTick((value) => {
        if (value >= HORIZON) {
          window.clearInterval(timer);
          setPlaying(false);
          return HORIZON;
        }
        return value + 1;
      });
    }, 38);
    return () => window.clearInterval(timer);
  }, [playing]);

  function invalidate() {
    setTick(0);
    setPlaying(false);
    setRan(false);
  }

  function reset() {
    setK(8);
    setPauseStart(24);
    setPauseDuration(8);
    setInject("none");
    setHold("consume-subgoal");
    setPrediction("");
    invalidate();
  }

  function runLab() {
    setRan(true);
    setTick(0);
    setPlaying(true);
  }

  const shownCup =
    current.cupPath[Math.min(displayTick, current.cupPath.length - 1)];

  return (
    <LabFrame
      lesson="29"
      title="双时钟：暂停 System 2，再注入错误子目标"
      description="先预测规划暂停后 System 1 会怎样，再跑教学模拟。把 System 2 暂停超过过期阈值，并注入错误子目标，观察末端轨迹是否被带着跑。"
    >
      <div className={styles.workspace}>
        <section className={styles.cockpit}>
          <div className={styles.toolbar}>
            <div>
              <span>CONTROL TICK</span>
              <strong>{String(displayTick).padStart(2, "0")}</strong>
            </div>
            <div className={styles.status}>
              <i data-live={playing} />
              {playing
                ? "RUNNING"
                : finished
                  ? "FINISHED"
                  : ran
                    ? "PAUSED"
                    : "READY"}
            </div>
            <label>
              暂停时 System 1
              <select
                value={hold}
                onChange={(event) => {
                  setHold(event.target.value as HoldPolicy);
                  invalidate();
                }}
              >
                <option value="consume-subgoal">继续消费最后子目标</option>
                <option value="repeat-action">重复最后动作增量</option>
              </select>
            </label>
          </div>

          <div className={styles.stageWrap}>
            <div className={styles.scene}>
              <svg viewBox="0 0 360 200" role="img" aria-label="桌面、杯子、货架与末端轨迹">
                <rect x="18" y="148" width="150" height="18" fill="#d7c3a3" stroke="#b39b78" />
                <rect x="286" y="20" width="56" height="12" fill="#8d6e4c" />
                <rect x="286" y="72" width="56" height="12" fill="#b08960" />
                <rect x="286" y="160" width="56" height="16" fill="#6d7c70" />
                <text x="288" y="16" fontSize="7" fill="#5e685f">货架二层</text>
                <text x="288" y="156" fontSize="7" fill="#5e685f">地面箱</text>
                <polyline
                  points={polyline(baseline.path)}
                  fill="none"
                  stroke="#c5d0c6"
                  strokeWidth="1.5"
                  strokeDasharray="3 3"
                />
                {revealed ? (
                  <polyline
                    points={polyline(current.path.slice(0, displayTick + 1))}
                    fill="none"
                    stroke="#176f48"
                    strokeWidth="2"
                  />
                ) : null}
                <circle cx={shownCup.x} cy={shownCup.y} r="6" fill="#c45c38" />
                <circle cx={ee.x} cy={ee.y} r="7" fill="#1a4f35" />
                <circle
                  cx={WAYPOINTS[subgoal].x}
                  cy={WAYPOINTS[subgoal].y}
                  r="4"
                  fill="none"
                  stroke="#3d6ea8"
                  strokeDasharray="2 2"
                />
                {pausedNow ? (
                  <text x="20" y="24" fontSize="8" fill="#b15a2c">
                    SYSTEM 2 PAUSED
                  </text>
                ) : null}
              </svg>
              <p className={styles.note}>
                虚线是无暂停、正确子目标的基线。实线是当前协议。教学模拟，不是模型输出。
              </p>
            </div>
            <div className={styles.clocks}>
              <h3>双时钟</h3>
              <div className={styles.lane}>
                <span>
                  System 2 规划
                  <b>
                    {revealed ? `${current.planSteps} / ${Math.ceil(HORIZON / k)}` : "—"}
                  </b>
                </span>
                <div className={styles.bar}>
                  <i
                    data-tone="slow"
                    data-stale={staleNow}
                    style={{
                      ["--fill" as string]: `${Math.min(100, (displayTick / HORIZON) * 100)}%`,
                    }}
                  />
                </div>
              </div>
              <div className={styles.lane}>
                <span>
                  System 1 控制
                  <b>{displayTick} / {HORIZON}</b>
                </span>
                <div className={styles.bar}>
                  <i
                    style={{
                      ["--fill" as string]: `${Math.min(100, (displayTick / HORIZON) * 100)}%`,
                    }}
                  />
                </div>
              </div>
              <p className={styles.note}>
                当前子目标：{SUBGOAL_LABEL[subgoal]}
                {pausedNow ? "（冻结）" : ""}
                {staleNow ? "（已过期）" : ""}
              </p>
              <p className={styles.note}>
                k={k}，ΔT2={k} ΔT1，过期阈值 {current.expire} 步。
              </p>
            </div>
          </div>

          <div className={styles.knobs}>
            <label className={styles.knob}>
              <span>规划间隔 k</span>
              <div>
                <input
                  type="range"
                  min={4}
                  max={16}
                  step={2}
                  value={k}
                  onChange={(event) => {
                    setK(Number(event.target.value));
                    invalidate();
                  }}
                />
                <output>{k}</output>
              </div>
            </label>
            <label className={styles.knob}>
              <span>暂停 System 2 起点</span>
              <div>
                <input
                  type="range"
                  min={0}
                  max={72}
                  step={4}
                  value={pauseStart}
                  onChange={(event) => {
                    setPauseStart(Number(event.target.value));
                    invalidate();
                  }}
                />
                <output>{pauseStart}</output>
              </div>
            </label>
            <label className={styles.knob}>
              <span>暂停时长</span>
              <div>
                <input
                  type="range"
                  min={0}
                  max={80}
                  step={4}
                  value={pauseDuration}
                  onChange={(event) => {
                    setPauseDuration(Number(event.target.value));
                    invalidate();
                  }}
                />
                <output>{pauseDuration}</output>
              </div>
            </label>
            <label className={styles.knob}>
              <span>注入子目标</span>
              <div>
                <select
                  value={inject}
                  onChange={(event) => {
                    setInject(event.target.value as InjectMode);
                    invalidate();
                  }}
                >
                  <option value="none">正确：架子第二层</option>
                  <option value="wrong-shelf">错误层</option>
                  <option value="floor">地面箱</option>
                </select>
              </div>
            </label>
          </div>

          <div className={styles.metrics} data-hidden={!revealed}>
            <div>
              <span>规划 / 控制步数</span>
              <b>
                {revealed
                  ? `${current.planSteps} / ${current.controlSteps}`
                  : "先预测再揭晓"}
              </b>
              <small>规划环不是每步都跑</small>
            </div>
            <div>
              <span>过期与任务</span>
              <b>
                {revealed
                  ? `${staleNow ? "STALE" : "FRESH"} · ${taskFailed ? "失败" : "成功"}`
                  : "先预测再揭晓"}
              </b>
              <small>暂停超过 {current.expire} 步应失败</small>
            </div>
            <div>
              <span>末端相对基线</span>
              <b>{revealed ? round(trajectoryDelta, 2) : "先预测再揭晓"}</b>
              <small>错误子目标必须抬高该距离</small>
            </div>
            <div>
              <span>最后动作重复率</span>
              <b>
                {revealed
                  ? `${Math.round(lastActionRepeatRatio * 100)}%`
                  : "先预测再揭晓"}
              </b>
              <small>
                {hold === "repeat-action"
                  ? "暂停后强制重复最后增量"
                  : "到达陈旧路点后动作为零"}
              </small>
            </div>
          </div>
        </section>

        <div className={styles.challenge}>
          <fieldset>
            <legend>先预测：暂停 System 2 之后，System 1 会怎样？</legend>
            {(Object.keys(PREDICTION_LABEL) as Prediction[]).map((key) => (
              <label key={key}>
                <input
                  type="radio"
                  name="dual-clock-prediction"
                  checked={prediction === key}
                  onChange={() => {
                    setPrediction(key);
                    invalidate();
                  }}
                />
                {PREDICTION_LABEL[key]}
              </label>
            ))}
          </fieldset>
          <div className={styles.actions}>
            <button type="button" className={styles.reset} onClick={reset}>
              重置
            </button>
            <button
              type="button"
              className={styles.run}
              disabled={!prediction || playing}
              onClick={runLab}
            >
              {playing ? "模拟中…" : "运行双时钟"}
            </button>
          </div>
        </div>
        {finished && prediction !== "stale-hold" ? (
          <p className={styles.feedback}>
            揭晓：System 1 不会发明下一阶段。它只消费冻结的子目标，或在“重复最后动作”开关下重放最后增量。
          </p>
        ) : null}
        {finished && (!pauseTooLong || inject === "none") ? (
          <p className={styles.feedback}>
            验收还差条件：把暂停时长调到大于过期阈值 {current.expire}，并把注入子目标改成错误层或地面箱。
          </p>
        ) : null}
        <Gate passed={passed}>
          预测“钉在最后子目标 / 重复最后动作”，让 System 2 暂停超过过期阈值导致任务失败，并注入错误子目标使末端轨迹偏离基线。
        </Gate>
      </div>
    </LabFrame>
  );
}
