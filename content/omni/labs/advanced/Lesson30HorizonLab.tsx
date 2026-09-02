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
import styles from "./Lesson30HorizonLab.module.css";

const START = 8;
const SPEED = 42;
const PERTURB_T = 0.4;
const PERTURB_DX = 20;
const ZONE_LO = 70;
const ZONE_HI = 84;
const FALL = 96;
const CATCH_RAD = 8;
const GRIPPER = 77;
const T_MAX_MS = 2500;
const TICK_MS = 1;

type Frame = {
  tMs: number;
  objectX: number;
  closed: boolean;
  status: string;
};

type SimResult = {
  openLoopS: number;
  commitS: number;
  delayS: number;
  stale: boolean;
  caught: boolean;
  staleMiss: boolean;
  closedAtS: number | null;
  frames: Frame[];
  reason: string;
};

function objectXAt(tMs: number) {
  const t = tMs / 1000;
  if (t < PERTURB_T) return START + SPEED * t;
  return START + SPEED * PERTURB_T + PERTURB_DX + SPEED * (t - PERTURB_T);
}

function planCloses(tObsMs: number, horizon: number, dtMs: number) {
  const x0 = objectXAt(tObsMs);
  return Array.from({ length: horizon }, (_, index) => {
    const tI = (tObsMs + (index + 1) * dtMs) / 1000;
    const predicted = x0 + SPEED * (tI - tObsMs / 1000);
    return predicted >= ZONE_LO && predicted <= ZONE_HI;
  });
}

function simulate(
  freqHz: number,
  horizon: number,
  kExec: number,
  delayMs: number,
): SimResult {
  const dtMs = Math.round(1000 / freqHz);
  const openLoopMs = Math.floor((horizon * 1000) / freqHz);
  const commitMs = Math.floor((kExec * 1000) / freqHz);
  const stale = delayMs >= openLoopMs;
  const frames: Frame[] = [];

  type Inflight = {
    tObsMs: number;
    availableAtMs: number;
    actions: boolean[];
  };
  const box: { flight: Inflight | null } = { flight: null };
  let current: { actions: boolean[]; executed: number } | null = null;
  let closed = false;
  let caught = false;
  let closedAtS: number | null = null;
  let staleDiscards = 0;

  const startInfer = (nowMs: number) => {
    if (box.flight) return;
    box.flight = {
      tObsMs: nowMs,
      availableAtMs: nowMs + delayMs,
      actions: planCloses(nowMs, horizon, dtMs),
    };
  };

  startInfer(0);

  for (let nowMs = 0; nowMs <= T_MAX_MS; nowMs += TICK_MS) {
    const objectX = objectXAt(nowMs);
    if (objectX >= FALL) {
      frames.push({
        tMs: nowMs,
        objectX,
        closed,
        status: caught ? "抓住" : "物体掉出",
      });
      break;
    }

    const pending = box.flight;
    if (pending && nowMs >= pending.availableAtMs) {
      const expired = nowMs > pending.tObsMs + openLoopMs || stale;
      if (expired) {
        staleDiscards += 1;
        box.flight = null;
      } else {
        current = { actions: pending.actions.slice(), executed: 0 };
        box.flight = null;
      }
    }

    if (nowMs % dtMs === 0) {
      if (current && current.executed < kExec && current.actions.length > 0) {
        const close = current.actions.shift() === true;
        current.executed += 1;
        if (close) {
          closed = true;
          if (
            objectX >= ZONE_LO &&
            objectX <= ZONE_HI &&
            Math.abs(objectX - GRIPPER) <= CATCH_RAD
          ) {
            caught = true;
            closedAtS = nowMs / 1000;
          }
        } else {
          closed = false;
        }
        if (current.executed >= kExec || current.actions.length === 0) {
          current = null;
          startInfer(nowMs);
        }
      } else if (!current && !box.flight) {
        startInfer(nowMs);
      }
    }

    if (nowMs % 20 === 0 || caught) {
      frames.push({
        tMs: nowMs,
        objectX,
        closed,
        status: caught
          ? "抓住"
          : nowMs / 1000 >= PERTURB_T
            ? "物体已被挪走"
            : "开环跟踪中",
      });
    }

    if (caught) break;
  }

  const last = frames[frames.length - 1];
  const staleMiss = !caught && stale;
  let reason = "开环计划错过闭合窗口";
  if (caught) reason = "fresh chunk 在截止前闭合";
  else if (staleMiss) reason = "延迟大于开环窗口，过期 chunk 被丢弃";
  else if (last && last.objectX >= FALL) reason = "物体离开传送带";

  return {
    openLoopS: openLoopMs / 1000,
    commitS: commitMs / 1000,
    delayS: delayMs / 1000,
    stale,
    caught,
    staleMiss,
    closedAtS,
    frames: frames.length > 0 ? frames : [{ tMs: 0, objectX: START, closed: false, status: "待运行" }],
    reason,
  };
}

export function Lesson30HorizonLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    freqHz: numberFrom(initialState, "freqHz", 10, 5, 50),
    horizon: numberFrom(initialState, "horizon", 12, 4, 40),
    kExec: numberFrom(initialState, "kExec", 12, 1, 40),
    delayMs: numberFrom(initialState, "delayMs", 200, 40, 1200),
    prediction: stringFrom(initialState, "prediction", ""),
  };
  const [freqHz, setFreqHz] = useState(
    Math.round(defaults.freqHz / 5) * 5,
  );
  const [horizon, setHorizon] = useState(defaults.horizon);
  const [kExec, setKExec] = useState(Math.min(defaults.kExec, defaults.horizon));
  const [delayMs, setDelayMs] = useState(defaults.delayMs);
  const [prediction, setPrediction] = useState(defaults.prediction);
  const [ran, setRan] = useState(false);
  const [hasCaught, setHasCaught] = useState(false);
  const [hasStaleMiss, setHasStaleMiss] = useState(false);
  const [frameIndex, setFrameIndex] = useState(0);

  const simulation = useMemo(
    () => simulate(freqHz, horizon, Math.min(kExec, horizon), delayMs),
    [delayMs, freqHz, horizon, kExec],
  );

  useEffect(() => {
    if (!ran) {
      setFrameIndex(0);
      return;
    }
    setFrameIndex(0);
    const timer = window.setInterval(() => {
      setFrameIndex((current) => {
        const next = current + 1;
        if (next >= simulation.frames.length - 1) {
          window.clearInterval(timer);
          return simulation.frames.length - 1;
        }
        return next;
      });
    }, 24);
    return () => window.clearInterval(timer);
  }, [ran, simulation]);

  const frame = simulation.frames[Math.min(frameIndex, simulation.frames.length - 1)];
  const passed =
    ran &&
    prediction === "stale-window" &&
    hasCaught &&
    hasStaleMiss;
  const completion = useMemo(
    () => ({
      lessonId: 30,
      freqHz,
      horizon,
      kExec: Math.min(kExec, horizon),
      delayMs,
      openLoopS: round(simulation.openLoopS, 3),
      commitS: round(simulation.commitS, 3),
      caught: simulation.caught,
      staleMiss: simulation.staleMiss,
      hasCaught,
      hasStaleMiss,
    }),
    [
      delayMs,
      freqHz,
      hasCaught,
      hasStaleMiss,
      horizon,
      kExec,
      simulation.caught,
      simulation.commitS,
      simulation.openLoopS,
      simulation.staleMiss,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function invalidate() {
    setRan(false);
    setFrameIndex(0);
  }

  function reset() {
    setFreqHz(10);
    setHorizon(12);
    setKExec(12);
    setDelayMs(200);
    setPrediction("");
    setRan(false);
    setHasCaught(false);
    setHasStaleMiss(false);
    setFrameIndex(0);
  }

  function runSim() {
    const result = simulate(freqHz, horizon, Math.min(kExec, horizon), delayMs);
    setHasCaught((current) => current || result.caught);
    setHasStaleMiss((current) => current || result.staleMiss);
    setRan(true);
  }

  const objectPct = Math.min(96, Math.max(4, frame.objectX));
  const gripperPct = GRIPPER;

  return (
    <LabFrame
      lesson="30"
      title="传送带上的开环窗口"
      description="教学模拟，不是模型输出。物体在 0.40 s 被挪走。先选预测，再调 f、H、执行 k 和推理延迟，分别找到能抓住的一组，以及延迟大于 H/f 而抓空的一组。"
    >
      <div className={styles.workspace}>
        <form className={styles.controls} onSubmit={(event) => event.preventDefault()}>
          <h3>控制回路旋钮</h3>
          <label>
            <span>控制频率 f <output>{freqHz} Hz</output></span>
            <input
              type="range"
              min="5"
              max="50"
              step="5"
              value={freqHz}
              onChange={(event) => {
                setFreqHz(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <label>
            <span>动作块长度 H <output>{horizon}</output></span>
            <input
              type="range"
              min="4"
              max="40"
              step="2"
              value={horizon}
              onChange={(event) => {
                const value = Number(event.target.value);
                setHorizon(value);
                setKExec((current) => Math.min(current, value));
                invalidate();
              }}
            />
          </label>
          <label>
            <span>执行前缀 k <output>{Math.min(kExec, horizon)}</output></span>
            <input
              type="range"
              min="1"
              max={horizon}
              step="1"
              value={Math.min(kExec, horizon)}
              onChange={(event) => {
                setKExec(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <label>
            <span>推理延迟 d <output>{delayMs} ms</output></span>
            <input
              type="range"
              min="40"
              max="1200"
              step="20"
              value={delayMs}
              onChange={(event) => {
                setDelayMs(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <p className={styles.note}>
            夹爪固定在抓取带上方，动作块只决定何时闭合。观察使用挪走后的真实位置，但推理完成前不能用。
          </p>
        </form>

        <div className={styles.stage}>
          <div className={styles.formula}>
            <span>T_open = H / f ， T_commit = k / f</span>
            <strong>
              {ran
                ? `T_open=${simulation.openLoopS.toFixed(2)}s · T_commit=${simulation.commitS.toFixed(2)}s · d=${simulation.delayS.toFixed(2)}s`
                : "运行后揭晓窗口与是否过期"}
            </strong>
          </div>
          <div className={styles.scene} aria-label="传送带抓取">
            <div
              className={styles.zone}
              style={{ left: `${ZONE_LO}%`, width: `${ZONE_HI - ZONE_LO}%` }}
            >
              <span>抓取带</span>
            </div>
            <div className={styles.belt} />
            <div
              className={`${styles.object} ${frame.tMs / 1000 >= PERTURB_T ? styles.objectMoved : ""}`}
              style={{ left: `${objectPct}%` }}
            />
            <div
              className={`${styles.gripper} ${frame.closed ? styles.gripperClosed : ""}`}
              style={{ left: `${gripperPct}%` }}
            >
              <div className={styles.palm} />
              <div style={{ display: "flex", gap: "0.9rem" }}>
                <div className={styles.jaw} />
                <div className={styles.jaw} />
              </div>
            </div>
            <div className={styles.caption}>
              <span>t = {ran ? (frame.tMs / 1000).toFixed(2) : "—"} s</span>
              <span>{ran ? frame.status : "等待预测后运行"}</span>
            </div>
          </div>
          <dl className={styles.metrics}>
            <div>
              <dt>开环窗口</dt>
              <dd>{ran ? `${simulation.openLoopS.toFixed(2)} s` : "—"}</dd>
            </div>
            <div>
              <dt>d ≥ H/f</dt>
              <dd>{ran ? (simulation.stale ? "过期" : "fresh") : "—"}</dd>
            </div>
            <div>
              <dt>本次结果</dt>
              <dd>{ran ? (simulation.caught ? "抓住" : "抓空") : "—"}</dd>
            </div>
            <div>
              <dt>闭合时刻</dt>
              <dd>
                {ran
                  ? simulation.closedAtS == null
                    ? "无有效闭合"
                    : `${simulation.closedAtS.toFixed(2)} s`
                  : "—"}
              </dd>
            </div>
          </dl>
          <div className={styles.findings}>
            <span className={hasCaught ? styles.ok : undefined}>
              {hasCaught ? "已找到能抓住的参数" : "还没找到抓住"}
            </span>
            <span className={hasStaleMiss ? styles.ok : undefined}>
              {hasStaleMiss ? "已找到延迟大于窗口的抓空" : "还没找到过期抓空"}
            </span>
          </div>
        </div>
      </div>

      <div className={styles.predict}>
        <fieldset>
          <legend>先预测：物体中途被挪走后，哪句话能同时解释抓住和抓空？</legend>
          {[
            ["max-h", "把 H 调到最大就能抓住，延迟再大也没关系"],
            [
              "stale-window",
              "延迟大于开环窗口 H/f 会抓空；要抓住需要 d < H/f，并且挪走后还能用新观察重规划",
            ],
            ["freq-only", "只要提高 f、不重规划，也能抓住被挪走的物体"],
          ].map(([value, label]) => (
            <label key={value}>
              <input
                type="radio"
                name="horizon-prediction"
                value={value}
                checked={prediction === value}
                onChange={() => {
                  setPrediction(value);
                  invalidate();
                }}
              />
              <span>{label}</span>
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
            disabled={!prediction}
            onClick={runSim}
          >
            运行抓取
          </button>
        </div>
      </div>
      {ran && prediction !== "stale-window" && (
        <p className={styles.feedback}>
          再核对两个窗口：加长 H 会加长盲走时间；只加 f、k=H
          时第一次计划仍按挪走前的到达时刻闭合。过期判定看 d 和 H/f。
        </p>
      )}
      {ran && prediction === "stale-window" && (
        <p className={styles.feedback}>
          {simulation.reason}。试一组短 k、小 d 去抓住；再把 d 拉到不小于 H/f 看抓空。
        </p>
      )}
      <Gate passed={passed}>
        先选对预测，再提交一组抓住和一组因 d ≥ H/f 抓空的参数。数字来自运动学夹具，不能写成真机成功率。
      </Gate>
    </LabFrame>
  );
}
