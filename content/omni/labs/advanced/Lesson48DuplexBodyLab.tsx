"use client";

import { useEffect, useMemo, useRef, useState, type PointerEvent } from "react";
import { Gate, LabFrame } from "./LabFrame";
import {
  numberFrom,
  round,
  stringFrom,
  useCompletionGate,
} from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson48DuplexBodyLab.module.css";

type Channel = "audio" | "action";
type Verb = "PAUSE" | "REPLAN" | "CONTINUE" | "UNDO";
type Prediction = "two-events" | "one-click-undo" | "pause-is-force";

type ControlEvent = {
  atMs: number;
  channel: Channel;
  action: Verb;
  sequence: number;
};

type Snapshot = {
  nowMs: number;
  audioAvailMs: number;
  actionAvailMs: number;
  audioMode: string;
  actionMode: string;
  branchId: number;
  pendingPcm: number;
  playedPcm: number;
  canceledPcm: number;
  remaining: number;
  executed: number;
  contact: boolean;
  cupCaught: boolean;
  armX: number;
  cupX: number;
  oldPcmAfterReplan: number;
  oldStepsAfterReplan: number;
  playedAtFirstEvent: number;
  contactAtFirstEvent: boolean;
  replanAt: number | null;
  pauseAt: number | null;
};

const ARM_HOME = 10;
const CUP_HOME = 32;
const CUP_AWAY = 78;
const GRASP = 36;
const CONTACT_STEP = 3;
const TICK = 10;
const T_MAX = 1400;
const PCM_LAG = 30;

const PREDICTION_LABEL: Record<Prediction, string> = {
  "two-events":
    "语音 PAUSE 和手臂 REPLAN 必须分成两次事件；已播 PCM 和已发生接触都不能撤回",
  "one-click-undo": "一次点击可以同时撤回已播音频和已发生的接触",
  "pause-is-force": "语音 PAUSE 等于力切断：手臂停住，接触会被抹掉",
};

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function openLoopMs(horizon: number, freqHz: number) {
  return Math.floor((horizon * 1000) / freqHz);
}

function audioStale(delayMs: number, blockMs: number) {
  return delayMs >= blockMs;
}

function actionStale(delayMs: number, horizon: number, freqHz: number) {
  return delayMs >= openLoopMs(horizon, freqHz);
}

function simulate(options: {
  nowMs: number;
  frameMs: number;
  freqHz: number;
  horizon: number;
  actionDelayMs: number;
  audioDelayMs: number;
  audioBlockMs: number;
  cupMoveMs: number | null;
  cupFrom: number;
  cupTo: number;
  events: ControlEvent[];
}): Snapshot {
  const dtMs = Math.round(1000 / options.freqHz);
  const windowMs = openLoopMs(options.horizon, options.freqHz);
  const events = [...options.events].sort(
    (left, right) => left.atMs - right.atMs || left.sequence - right.sequence,
  );

  type Branch = {
    parent: number | null;
    status: "active" | "superseded";
    audioMode: "SPEAKING" | "PAUSED";
    actionMode: "GENERATING" | "PAUSED" | "SAFE_HOLD";
  };
  const branches: Record<number, Branch> = {
    1: {
      parent: null,
      status: "active",
      audioMode: "SPEAKING",
      actionMode: "GENERATING",
    },
  };
  let active = 1;
  let pending: { branch: number; playAt: number }[] = [];
  let played = 0;
  let canceled = 0;
  let remaining: number[] = [];
  let executed = 0;
  let contact = false;
  let cupCaught = false;
  const flight: { current: { branch: number; tObs: number; avail: number } | null } = {
    current: null,
  };
  let armX = ARM_HOME;
  let pauseAt: number | null = null;
  let replanAt: number | null = null;
  let playedAtFirstEvent = 0;
  let contactAtFirstEvent = false;
  let sawFirstEvent = false;
  let oldPcmAfterReplan = 0;
  let oldStepsAfterReplan = 0;

  const cupAt = (timeMs: number) => {
    if (options.cupMoveMs != null && timeMs >= options.cupMoveMs) {
      return options.cupTo;
    }
    return options.cupFrom;
  };

  const startInfer = (now: number) => {
    const branch = branches[active];
    if (
      flight.current
      || remaining.length > 0
      || branch.actionMode !== "GENERATING"
      || branch.status !== "active"
    ) {
      return;
    }
    flight.current = {
      branch: active,
      tObs: now,
      avail: now + options.actionDelayMs,
    };
  };

  startInfer(0);

  for (let now = 0; now <= options.nowMs; now += TICK) {
    for (const event of events) {
      if (event.atMs !== now) continue;
      if (!sawFirstEvent) {
        playedAtFirstEvent = played;
        contactAtFirstEvent = contact;
        sawFirstEvent = true;
      }
      const branch = branches[active];
      if (event.action === "UNDO") {
        played = 0;
        contact = false;
        cupCaught = false;
        pending = [];
        remaining = [];
      } else if (event.action === "PAUSE" && event.channel === "audio") {
        branch.audioMode = "PAUSED";
        pauseAt = now;
      } else if (event.action === "PAUSE" && event.channel === "action") {
        branch.actionMode = "PAUSED";
      } else if (event.action === "CONTINUE" && event.channel === "audio") {
        branch.audioMode = "SPEAKING";
      } else if (event.action === "REPLAN") {
        canceled += pending.filter((item) => item.branch === active).length;
        pending = pending.filter((item) => item.branch !== active);
        remaining = [];
        flight.current = null;
        branch.status = "superseded";
        const parent = active;
        active += 1;
        branches[active] = {
          parent,
          status: "active",
          audioMode: "SPEAKING",
          actionMode: "GENERATING",
        };
        replanAt = now;
        startInfer(now);
      }
    }

    const branch = branches[active];
    const queued = flight.current;
    if (queued && now >= queued.avail && remaining.length === 0) {
      const delay = queued.avail - queued.tObs;
      const expired =
        now > queued.tObs + windowMs
        || actionStale(delay, options.horizon, options.freqHz);
      if (expired || queued.branch !== active) {
        flight.current = null;
      } else {
        remaining = Array.from({ length: options.horizon }, (_, index) => index);
        flight.current = null;
      }
    }

    if (
      now % options.frameMs === 0
      && branch.status === "active"
      && branch.audioMode === "SPEAKING"
    ) {
      const avail = now + options.audioDelayMs;
      if (audioStale(options.audioDelayMs, options.audioBlockMs)) {
        canceled += 1;
      } else {
        pending.push({ branch: active, playAt: avail + PCM_LAG });
      }
    }

    const still: typeof pending = [];
    for (const item of pending) {
      const owner = branches[item.branch];
      const due = item.playAt <= now;
      if (due && owner.status === "active" && owner.audioMode === "SPEAKING") {
        played += 1;
        if (replanAt != null && now >= replanAt && item.branch < active) {
          oldPcmAfterReplan += 1;
        }
      } else if (due && owner.status === "superseded") {
        canceled += 1;
      } else {
        still.push(item);
      }
    }
    pending = still;

    if (
      now % dtMs === 0
      && branch.status === "active"
      && branch.actionMode === "GENERATING"
      && remaining.length > 0
    ) {
      const step = remaining.shift() ?? 0;
      executed += 1;
      const planned = replanAt != null && now >= replanAt ? cupAt(now) : GRASP;
      const span = planned - armX;
      armX += span * 0.45;
      if (step === CONTACT_STEP) {
        contact = true;
        if (Math.abs(armX - cupAt(now)) <= 10) cupCaught = true;
      }
      if (replanAt != null && now >= replanAt && active === 1) {
        oldStepsAfterReplan += 1;
      }
      if (remaining.length === 0) startInfer(now);
    } else if (
      now % dtMs === 0
      && branch.status === "active"
      && branch.actionMode === "GENERATING"
      && remaining.length === 0
      && !flight.current
    ) {
      startInfer(now);
    }

    if (branch.actionMode === "GENERATING" && remaining.length === 0) {
      armX += (ARM_HOME + 6 - armX) * 0.02;
    }
  }

  const lastEvent = events[events.length - 1];
  return {
    nowMs: options.nowMs,
    audioAvailMs: lastEvent ? lastEvent.atMs : options.audioDelayMs,
    actionAvailMs: lastEvent ? lastEvent.atMs : options.actionDelayMs,
    audioMode: branches[active].audioMode,
    actionMode: branches[active].actionMode,
    branchId: active,
    pendingPcm: pending.length,
    playedPcm: played,
    canceledPcm: canceled,
    remaining: remaining.length,
    executed,
    contact,
    cupCaught,
    armX: clamp(armX, 6, 94),
    cupX: cupAt(options.nowMs),
    oldPcmAfterReplan,
    oldStepsAfterReplan,
    playedAtFirstEvent,
    contactAtFirstEvent,
    replanAt,
    pauseAt,
  };
}

export function Lesson48DuplexBodyLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    frameMs: numberFrom(initialState, "frameMs", 80, 40, 160),
    freqHz: numberFrom(initialState, "freqHz", 20, 10, 50),
    horizon: numberFrom(initialState, "horizon", 8, 4, 16),
    actionDelayMs: numberFrom(initialState, "actionDelayMs", 100, 40, 400),
    prediction: stringFrom(initialState, "prediction", "") as Prediction | "",
  };
  const [frameMs, setFrameMs] = useState(
    Math.round(defaults.frameMs / 20) * 20,
  );
  const [freqHz, setFreqHz] = useState(Math.round(defaults.freqHz / 5) * 5);
  const [horizon, setHorizon] = useState(defaults.horizon);
  const [actionDelayMs, setActionDelayMs] = useState(defaults.actionDelayMs);
  const [prediction, setPrediction] = useState<Prediction | "">(
    defaults.prediction === "two-events"
      || defaults.prediction === "one-click-undo"
      || defaults.prediction === "pause-is-force"
      ? defaults.prediction
      : "",
  );
  const [ran, setRan] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [clockMs, setClockMs] = useState(0);
  const [cupX, setCupX] = useState(CUP_HOME);
  const [cupMoved, setCupMoved] = useState(false);
  const [cupMoveMs, setCupMoveMs] = useState<number | null>(null);
  const [events, setEvents] = useState<ControlEvent[]>([]);
  const [illegalUndo, setIllegalUndo] = useState(false);
  const sceneRef = useRef<HTMLDivElement | null>(null);
  const dragging = useRef(false);

  const audioBlockMs = 320;
  const audioDelayMs = 40;
  const windowMs = openLoopMs(horizon, freqHz);

  const snapshot = useMemo(
    () =>
      simulate({
        nowMs: ran ? clockMs : 0,
        frameMs,
        freqHz,
        horizon,
        actionDelayMs,
        audioDelayMs,
        audioBlockMs,
        cupMoveMs,
        cupFrom: CUP_HOME,
        cupTo: cupX,
        events,
      }),
    [
      actionDelayMs,
      audioDelayMs,
      clockMs,
      cupMoveMs,
      cupX,
      events,
      frameMs,
      freqHz,
      horizon,
      ran,
    ],
  );

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      setClockMs((current) => {
        const next = Math.min(T_MAX, current + 40);
        if (next >= T_MAX) {
          window.clearInterval(timer);
          setPlaying(false);
        }
        return next;
      });
    }, 40);
    return () => window.clearInterval(timer);
  }, [playing]);

  const pauseCount = events.filter(
    (event) => event.action === "PAUSE" && event.channel === "audio",
  ).length;
  const replanCount = events.filter((event) => event.action === "REPLAN").length;
  const playedNotUndone =
    snapshot.playedPcm >= snapshot.playedAtFirstEvent
    && snapshot.playedPcm > 0;
  const contactNotUndone =
    !snapshot.contactAtFirstEvent || snapshot.contact;
  const passed =
    ran
    && prediction === "two-events"
    && cupMoved
    && pauseCount >= 1
    && replanCount >= 1
    && !illegalUndo
    && playedNotUndone
    && contactNotUndone
    && snapshot.oldPcmAfterReplan === 0
    && snapshot.oldStepsAfterReplan === 0
    && snapshot.replanAt != null
    && snapshot.pauseAt != null
    && snapshot.pauseAt !== snapshot.replanAt;

  const completion = useMemo(
    () => ({
      lessonId: 48,
      frameMs,
      freqHz,
      horizon,
      actionDelayMs,
      prediction,
      cupMoved,
      pauseCount,
      replanCount,
      illegalUndo,
      playedPcm: snapshot.playedPcm,
      remaining: snapshot.remaining,
      contact: snapshot.contact,
      oldPcmAfterReplan: snapshot.oldPcmAfterReplan,
      oldStepsAfterReplan: snapshot.oldStepsAfterReplan,
      audioAvailMs: snapshot.audioAvailMs,
      actionAvailMs: snapshot.actionAvailMs,
    }),
    [
      actionDelayMs,
      cupMoved,
      frameMs,
      freqHz,
      horizon,
      illegalUndo,
      pauseCount,
      prediction,
      replanCount,
      snapshot.actionAvailMs,
      snapshot.audioAvailMs,
      snapshot.contact,
      snapshot.oldPcmAfterReplan,
      snapshot.oldStepsAfterReplan,
      snapshot.playedPcm,
      snapshot.remaining,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function invalidateClocks() {
    setRan(false);
    setPlaying(false);
    setClockMs(0);
    setEvents([]);
    setIllegalUndo(false);
    setCupMoveMs(null);
  }

  function reset() {
    setFrameMs(80);
    setFreqHz(20);
    setHorizon(8);
    setActionDelayMs(100);
    setPrediction("");
    setCupX(CUP_HOME);
    setCupMoved(false);
    invalidateClocks();
  }

  function startTalking() {
    if (!prediction) return;
    setRan(true);
    setPlaying(true);
    setClockMs(0);
    setEvents([]);
    setIllegalUndo(false);
    setCupMoveMs(cupMoved ? 0 : null);
  }

  function pushEvent(channel: Channel, action: Verb) {
    if (!ran) return;
    setEvents((current) => [
      ...current,
      {
        atMs: Math.round(clockMs / TICK) * TICK,
        channel,
        action,
        sequence: current.length + 1,
      },
    ]);
    if (action === "UNDO") setIllegalUndo(true);
  }

  function moveCupTo(nextX: number, clientSource: "button" | "drag") {
    const value = clamp(nextX, 10, 90);
    setCupX(value);
    if (Math.abs(value - CUP_HOME) > 6) {
      setCupMoved(true);
      setCupMoveMs((current) => (current == null ? (ran ? clockMs : 0) : current));
    }
    void clientSource;
  }

  function onPointerDown(event: PointerEvent<HTMLButtonElement>) {
    dragging.current = true;
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function onPointerMove(event: PointerEvent<HTMLButtonElement>) {
    if (!dragging.current || !sceneRef.current) return;
    const rect = sceneRef.current.getBoundingClientRect();
    const pct = ((event.clientX - rect.left) / rect.width) * 100;
    moveCupTo(pct, "drag");
  }

  function onPointerUp() {
    dragging.current = false;
  }

  const audioTick = Math.floor(snapshot.nowMs / frameMs);
  const actionTick = Math.floor(snapshot.nowMs / Math.max(1, Math.round(1000 / freqHz)));

  return (
    <LabFrame
      lesson="48"
      title="说话时挪走杯子"
      description="教学模拟，不是模型输出。先选预测，再开始说话并伸手。把杯子拖走之后，语音 PAUSE 和手臂 REPLAN 必须分成两次点击。一次撤销全部会伪造历史，验收失败。"
    >
      <div className={styles.workspace}>
        <form className={styles.controls} onSubmit={(event) => event.preventDefault()}>
          <h3>双时钟旋钮</h3>
          <label>
            <span>音频帧 <output>{frameMs} ms</output></span>
            <input
              type="range"
              min="40"
              max="160"
              step="20"
              value={frameMs}
              onChange={(event) => {
                setFrameMs(Number(event.target.value));
                invalidateClocks();
              }}
            />
          </label>
          <label>
            <span>控制频率 f <output>{freqHz} Hz</output></span>
            <input
              type="range"
              min="10"
              max="50"
              step="5"
              value={freqHz}
              onChange={(event) => {
                setFreqHz(Number(event.target.value));
                invalidateClocks();
              }}
            />
          </label>
          <label>
            <span>动作块 H <output>{horizon}</output></span>
            <input
              type="range"
              min="4"
              max="16"
              step="1"
              value={horizon}
              onChange={(event) => {
                setHorizon(Number(event.target.value));
                invalidateClocks();
              }}
            />
          </label>
          <label>
            <span>动作延迟 d <output>{actionDelayMs} ms</output></span>
            <input
              type="range"
              min="40"
              max="400"
              step="20"
              value={actionDelayMs}
              onChange={(event) => {
                setActionDelayMs(Number(event.target.value));
                invalidateClocks();
              }}
            />
          </label>
          <p className={styles.note}>
            音频过期看 {audioBlockMs} ms 块；手臂过期看 H/f = {round(windowMs / 1000, 2)} s。
            两列 available_at 不能横着比。
          </p>
        </form>

        <div className={styles.stage}>
          <div className={styles.formula}>
            <span>t_audio = block_end + d_enc · t_action = t_obs + d</span>
            <strong>
              {ran
                ? `音频滴答 ${audioTick} · 控制滴答 ${actionTick} · t=${snapshot.nowMs}ms`
                : "运行后揭晓两列时间戳"}
            </strong>
          </div>

          <div className={styles.clocks} aria-label="双时钟">
            <div>
              <span>音频时钟</span>
              <i style={{ width: `${Math.min(100, (clockMs / T_MAX) * 100)}%` }} />
            </div>
            <div>
              <span>控制时钟</span>
              <i style={{ width: `${Math.min(100, (clockMs / T_MAX) * 100)}%` }} />
            </div>
          </div>

          <div className={styles.dual}>
            <div className={styles.pcmPanel}>
              <strong>PCM 队列</strong>
              <div className={styles.pcmTrack} aria-hidden="true">
                {Array.from({ length: 12 }, (_, index) => {
                  const filled = ran && index < snapshot.playedPcm;
                  const pending = ran && index >= snapshot.playedPcm && index < snapshot.playedPcm + snapshot.pendingPcm;
                  const canceledBar =
                    ran && index >= snapshot.playedPcm + snapshot.pendingPcm
                    && index < snapshot.playedPcm + snapshot.pendingPcm + Math.min(3, snapshot.canceledPcm);
                  return (
                    <b
                      key={index}
                      data-state={
                        filled ? "played" : pending ? "pending" : canceledBar ? "canceled" : "idle"
                      }
                    />
                  );
                })}
              </div>
              <p>
                {ran
                  ? `已播 ${snapshot.playedPcm} · 未播 ${snapshot.pendingPcm} · 取消 ${snapshot.canceledPcm}`
                  : "已播 / 未播 / 取消 运行后揭晓"}
              </p>
            </div>

            <div
              className={styles.scene}
              ref={sceneRef}
              aria-label="桌面与杯子"
            >
              <div className={styles.table} />
              <div
                className={`${styles.arm} ${snapshot.contact ? styles.armContact : ""}`}
                style={{ left: `${snapshot.armX}%` }}
              >
                <span />
                <span />
              </div>
              <button
                type="button"
                className={`${styles.cup} ${cupMoved ? styles.cupMoved : ""}`}
                style={{ left: `${ran ? snapshot.cupX : cupX}%` }}
                onPointerDown={onPointerDown}
                onPointerMove={onPointerMove}
                onPointerUp={onPointerUp}
                onPointerCancel={onPointerUp}
              >
                杯
              </button>
              <div className={styles.caption}>
                <span>{ran ? snapshot.audioMode : "未开始"}</span>
                <span>
                  {ran
                    ? `${snapshot.actionMode}${snapshot.cupCaught ? " · 套住" : ""}`
                    : "未开始"}
                </span>
              </div>
            </div>
          </div>

          <dl className={styles.metrics}>
            <div>
              <dt>audio_available_at</dt>
              <dd>{ran ? `${snapshot.audioAvailMs} ms` : "揭晓后"}</dd>
            </div>
            <div>
              <dt>action_available_at</dt>
              <dd>{ran ? `${snapshot.actionAvailMs} ms` : "揭晓后"}</dd>
            </div>
            <div>
              <dt>剩余步 / 接触</dt>
              <dd>
                {ran
                  ? `${snapshot.remaining} / ${snapshot.contact ? "已发生" : "未发生"}`
                  : "揭晓后"}
              </dd>
            </div>
            <div>
              <dt>旧未来泄漏</dt>
              <dd>
                {ran
                  ? `PCM ${snapshot.oldPcmAfterReplan} · 步 ${snapshot.oldStepsAfterReplan}`
                  : "揭晓后"}
              </dd>
            </div>
          </dl>

          <div className={styles.eventBar}>
            <button
              type="button"
              className={styles.secondary}
              onClick={() => moveCupTo(CUP_AWAY, "button")}
            >
              挪走杯子
            </button>
            <button
              type="button"
              className={styles.secondary}
              disabled={!ran}
              onClick={() => pushEvent("audio", "PAUSE")}
            >
              语音 PAUSE
            </button>
            <button
              type="button"
              className={styles.run}
              disabled={!ran}
              onClick={() => pushEvent("action", "REPLAN")}
            >
              手臂 REPLAN
            </button>
            <button
              type="button"
              className={styles.danger}
              disabled={!ran}
              onClick={() => pushEvent("audio", "UNDO")}
            >
              一次撤销全部
            </button>
          </div>

          <ul className={styles.log}>
            {events.length === 0 ? (
              <li>事件日志在运行后出现。PAUSE 与 REPLAN 必须是两条。</li>
            ) : (
              events.map((event) => (
                <li key={`${event.sequence}-${event.atMs}`}>
                  t={event.atMs}ms · {event.channel} · {event.action}
                </li>
              ))
            )}
          </ul>
        </div>
      </div>

      <div className={styles.predict}>
        <fieldset>
          <legend>先预测：杯子被挪走时，正确协议是什么？</legend>
          {(Object.keys(PREDICTION_LABEL) as Prediction[]).map((key) => (
            <label key={key}>
              <input
                type="radio"
                name="lesson48-prediction"
                value={key}
                checked={prediction === key}
                onChange={() => {
                  setPrediction(key);
                  invalidateClocks();
                }}
              />
              <span>{PREDICTION_LABEL[key]}</span>
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
            onClick={startTalking}
          >
            开始说话并伸手
          </button>
        </div>
      </div>

      {ran && illegalUndo ? (
        <p className={styles.feedback}>
          一次撤销全部同时清掉了已播 PCM 和接触标志。这是伪造历史，验收不能通过。
        </p>
      ) : null}
      {ran && prediction && prediction !== "two-events" ? (
        <p className={styles.feedback}>
          预测与协议不一致。正确做法是两次独立事件，并且不撤回已经发生的播出和接触。
        </p>
      ) : null}
      {ran && prediction === "two-events" && (!cupMoved || pauseCount < 1 || replanCount < 1) ? (
        <p className={styles.feedback}>
          还差步骤：先拖走或点挪走杯子，再分别点语音 PAUSE 和手臂 REPLAN。
        </p>
      ) : null}

      <Gate passed={passed}>
        {passed
          ? "两次事件已分开，旧 PCM 与旧剩余步不再执行，已播音频和接触没有被撤回。"
          : "先选预测，开始说话，挪走杯子，再分两次发出 PAUSE 与 REPLAN。不要点一次撤销全部。"}
      </Gate>
    </LabFrame>
  );
}
