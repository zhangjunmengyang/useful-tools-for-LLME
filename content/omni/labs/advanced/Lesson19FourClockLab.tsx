"use client";

import { useEffect, useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import {
  numberFrom,
  stringFrom,
  useCompletionGate,
} from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson19FourClockLab.module.css";

type InterruptPolicy = "playback" | "talker" | "cascade";

const policyLabels: Record<InterruptPolicy, string> = {
  playback: "只停播放器",
  talker: "停 Talker，保留队列",
  cascade: "Thinker + Talker + queue 全链路取消",
};

export function Lesson19FourClockLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    frameMs: numberFrom(initialState, "asrFrameMs", 40, 20, 100),
    endpointMs: numberFrom(initialState, "endpointMs", 700, 300, 1400),
    thinkLatency: numberFrom(initialState, "thinkLatency", 450, 150, 1000),
    lookahead: numberFrom(initialState, "talkerLookahead", 160, 40, 400),
    codecFps: numberFrom(initialState, "codecFps", 50, 25, 100),
    interruptMs: numberFrom(initialState, "interruptMs", 2200, 1000, 4000),
    policy: stringFrom(initialState, "policy", "playback") as InterruptPolicy,
    prediction: stringFrom(initialState, "prediction", ""),
  };
  const [frameMs, setFrameMs] = useState(defaults.frameMs);
  const [endpointMs, setEndpointMs] = useState(defaults.endpointMs);
  const [thinkLatency, setThinkLatency] = useState(defaults.thinkLatency);
  const [lookahead, setLookahead] = useState(defaults.lookahead);
  const [codecFps, setCodecFps] = useState(defaults.codecFps);
  const [interruptMs, setInterruptMs] = useState(defaults.interruptMs);
  const [policy, setPolicy] = useState<InterruptPolicy>(
    ["playback", "talker", "cascade"].includes(defaults.policy)
      ? defaults.policy
      : "playback",
  );
  const [prediction, setPrediction] = useState(defaults.prediction);
  const [clock, setClock] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [ran, setRan] = useState(false);

  const timeline = useMemo(() => {
    const inputEnd = endpointMs;
    const thinkerStart = inputEnd;
    const talkerStart = thinkerStart + thinkLatency;
    const playbackStart = talkerStart + lookahead;
    const responseDuration = 2400;
    const responseEnd = talkerStart + responseDuration;
    const validInterrupt =
      interruptMs > playbackStart && interruptMs < responseEnd;
    const bufferMs = Math.min(240, Math.max(80, lookahead));
    const staleMs =
      !validInterrupt
        ? 0
        : policy === "cascade"
          ? 0
          : policy === "talker"
            ? Math.min(bufferMs, responseEnd - interruptMs)
            : responseEnd - interruptMs;
    const thinkerEnd =
      validInterrupt && policy === "cascade" ? interruptMs : responseEnd;
    const talkerEnd =
      validInterrupt && policy !== "playback" ? interruptMs : responseEnd;
    const playbackEnd =
      validInterrupt
        ? policy === "talker"
          ? interruptMs + staleMs
          : interruptMs
        : responseEnd;
    const horizon = responseEnd + 300;
    return {
      inputEnd,
      thinkerStart,
      talkerStart,
      playbackStart,
      responseEnd,
      validInterrupt,
      bufferMs,
      staleMs,
      thinkerEnd,
      talkerEnd,
      playbackEnd,
      horizon,
    };
  }, [endpointMs, interruptMs, lookahead, policy, thinkLatency]);

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      setClock((current) => {
        const next = Math.min(timeline.horizon, current + 100);
        if (next >= timeline.horizon) {
          window.clearInterval(timer);
          setPlaying(false);
        }
        return next;
      });
    }, 55);
    return () => window.clearInterval(timer);
  }, [playing, timeline.horizon]);

  const reachedInterrupt = ran && clock >= interruptMs;
  const passed =
    reachedInterrupt &&
    prediction === "cascade" &&
    policy === "cascade" &&
    timeline.validInterrupt &&
    timeline.staleMs === 0;
  const completion = useMemo(
    () => ({
      lessonId: 19,
      policy,
      asrFrameMs: frameMs,
      endpointMs,
      thinkLatency,
      talkerLookahead: lookahead,
      codecFps,
      thinkerFirstTokenMs: timeline.talkerStart,
      playbackStartMs: timeline.playbackStart,
      interruptMs,
      staleAudioMs: timeline.staleMs,
    }),
    [
      codecFps,
      endpointMs,
      frameMs,
      interruptMs,
      lookahead,
      policy,
      thinkLatency,
      timeline,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function invalidate() {
    setClock(0);
    setPlaying(false);
    setRan(false);
  }

  function reset() {
    setFrameMs(defaults.frameMs);
    setEndpointMs(defaults.endpointMs);
    setThinkLatency(defaults.thinkLatency);
    setLookahead(defaults.lookahead);
    setCodecFps(defaults.codecFps);
    setInterruptMs(defaults.interruptMs);
    setPolicy("playback");
    setPrediction("");
    setClock(0);
    setPlaying(false);
    setRan(false);
  }

  function run() {
    setRan(true);
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setClock(timeline.horizon);
      setPlaying(false);
      return;
    }
    setClock(0);
    setPlaying(true);
  }

  const clockReadouts = {
    asr: `${Math.floor(Math.min(clock, timeline.inputEnd) / frameMs)} frames`,
    thinker:
      clock < timeline.thinkerStart
        ? "idle"
        : `${Math.floor((Math.min(clock, timeline.thinkerEnd) - timeline.thinkerStart) / 50)} tok`,
    talker:
      clock < timeline.talkerStart
        ? "idle"
        : `${Math.floor(
            ((Math.min(clock, timeline.talkerEnd) - timeline.talkerStart) /
              1000) *
              codecFps,
          )} codec`,
    playback:
      clock < timeline.playbackStart
        ? "idle"
        : `${Math.max(
            0,
            Math.floor(
              Math.min(clock, timeline.playbackEnd) - timeline.playbackStart,
            ),
          )} ms`,
  };

  return (
    <LabFrame
      lesson="19"
      title="四时钟全双工中断驾驶舱"
      description="调整输入帧、Thinker token、Talker codec 和扬声器播放的延迟，再检查用户打断后是否仍播放旧语音。"
    >
      <section className={styles.cockpit}>
        <div className={styles.toolbar}>
          <div>
            <span>SIM CLOCK</span>
            <strong>{clock.toString().padStart(4, "0")} ms</strong>
          </div>
          <div className={styles.status}>
            <i data-live={playing} />
            {playing
              ? "RUNNING"
              : reachedInterrupt
                ? "INTERRUPTED"
                : ran
                  ? "FINISHED"
                  : "READY"}
          </div>
          <label>
            取消策略
            <select
              value={policy}
              onChange={(event) => {
                setPolicy(event.target.value as InterruptPolicy);
                invalidate();
              }}
            >
              {(Object.keys(policyLabels) as InterruptPolicy[]).map((key) => (
                <option value={key} key={key}>
                  {policyLabels[key]}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className={styles.timeline}>
          <div className={styles.timeAxis}>
            {[0, 25, 50, 75, 100].map((percent) => (
              <span key={percent} style={{ left: `${percent}%` }}>
                {Math.round((timeline.horizon * percent) / 100)}ms
              </span>
            ))}
          </div>
          <div className={styles.trackOverlay} aria-hidden="true">
            <div
              className={styles.cursor}
              style={{ left: `${(clock / timeline.horizon) * 100}%` }}
            />
            <div
              className={styles.interruptLine}
              style={{ left: `${(interruptMs / timeline.horizon) * 100}%` }}
            >
              <span>BARGE-IN</span>
            </div>
          </div>
          <ClockLane
            label="01 · ASR input"
            readout={clockReadouts.asr}
            start={0}
            end={timeline.inputEnd}
            horizon={timeline.horizon}
            tone="asr"
          />
          <ClockLane
            label="02 · Thinker"
            readout={clockReadouts.thinker}
            start={timeline.thinkerStart}
            end={timeline.thinkerEnd}
            horizon={timeline.horizon}
            tone="thinker"
          />
          <ClockLane
            label="03 · Talker"
            readout={clockReadouts.talker}
            start={timeline.talkerStart}
            end={timeline.talkerEnd}
            horizon={timeline.horizon}
            tone="talker"
          />
          <ClockLane
            label="04 · Playback"
            readout={clockReadouts.playback}
            start={timeline.playbackStart}
            end={timeline.playbackEnd}
            horizon={timeline.horizon}
            tone="playback"
          />
        </div>

        <div className={styles.knobs}>
          <Knob
            label="ASR frame"
            value={frameMs}
            unit="ms"
            min={20}
            max={100}
            step={10}
            onChange={(value) => {
              setFrameMs(value);
              invalidate();
            }}
          />
          <Knob
            label="Endpoint"
            value={endpointMs}
            unit="ms"
            min={300}
            max={1400}
            step={50}
            onChange={(value) => {
              setEndpointMs(value);
              invalidate();
            }}
          />
          <Knob
            label="Think latency"
            value={thinkLatency}
            unit="ms"
            min={150}
            max={1000}
            step={50}
            onChange={(value) => {
              setThinkLatency(value);
              invalidate();
            }}
          />
          <Knob
            label="Talker lookahead"
            value={lookahead}
            unit="ms"
            min={40}
            max={400}
            step={20}
            onChange={(value) => {
              setLookahead(value);
              invalidate();
            }}
          />
          <Knob
            label="Codec rate"
            value={codecFps}
            unit="fps"
            min={25}
            max={100}
            step={25}
            onChange={(value) => {
              setCodecFps(value);
              invalidate();
            }}
          />
          <Knob
            label="Interrupt at"
            value={interruptMs}
            unit="ms"
            min={1000}
            max={4000}
            step={100}
            onChange={(value) => {
              setInterruptMs(value);
              invalidate();
            }}
          />
        </div>

        <div className={styles.diagnostics}>
          <div>
            <span>首包播放</span>
            <b>{timeline.playbackStart} ms</b>
            <small>endpoint + think + lookahead</small>
          </div>
          <div>
            <span>中断后陈旧音频</span>
            <b>{reachedInterrupt ? `${timeline.staleMs} ms` : "—"}</b>
            <small>
              {policy === "playback"
                ? "Talker 继续生成"
                : policy === "talker"
                  ? "保留已排队 buffer"
                  : "全链路取消并 flush"}
            </small>
          </div>
          <div>
            <span>中断窗口</span>
            <b>{timeline.validInterrupt ? "有效" : "无效"}</b>
            <small>必须落在 playback start 与 response end 之间</small>
          </div>
        </div>
      </section>

      <div className={styles.challenge}>
        <fieldset>
          <legend>先预测：哪种策略能保证 barge-in 后没有旧回复再次冒出来？</legend>
          {(Object.keys(policyLabels) as InterruptPolicy[]).map((key) => (
            <label key={key}>
              <input
                type="radio"
                name="clock-prediction"
                checked={prediction === key}
                onChange={() => {
                  setPrediction(key);
                  invalidate();
                }}
              />
              {policyLabels[key]}
            </label>
          ))}
        </fieldset>
        <div>
          <button type="button" onClick={reset}>重置</button>
          <button
            type="button"
            className={styles.run}
            disabled={!prediction || playing}
            onClick={run}
          >
            {playing ? "模拟中…" : "运行四时钟"}
          </button>
        </div>
      </div>
      <Gate passed={passed}>
        让中断落在有效播放窗口，预测并启用全链路 cascade cancel，使陈旧音频严格等于 0ms。
      </Gate>
    </LabFrame>
  );
}

function ClockLane({
  label,
  readout,
  start,
  end,
  horizon,
  tone,
}: {
  label: string;
  readout: string;
  start: number;
  end: number;
  horizon: number;
  tone: string;
}) {
  return (
    <div className={styles.lane}>
      <span>{label}</span>
      <div>
        <i
          className={styles[tone]}
          style={{
            left: `${(start / horizon) * 100}%`,
            width: `${Math.max(0, ((end - start) / horizon) * 100)}%`,
          }}
        />
      </div>
      <b>{readout}</b>
    </div>
  );
}

function Knob({
  label,
  value,
  unit,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  unit: string;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className={styles.knob}>
      <span>{label}</span>
      <div>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(event) => onChange(Number(event.target.value))}
        />
        <output>{value}{unit}</output>
      </div>
    </label>
  );
}
