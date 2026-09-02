"use client";

import { useEffect, useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import { numberFrom, stringFrom, useCompletionGate } from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson40ForceCutoffLab.module.css";

const FORCE_N = [4, 6, 9, 14, 22, 35, 52, 74, 102, 135, 172, 210];
const H = FORCE_N.length;
const CONTACT = 20;
const EE_STEP = 5;
const HOLD_TICKS = 3;
const SPEECH = ["把", "热", "水", "倒", "进", "杯", "子", "里", "到", "七", "分", "满"];
const PEAK_FORCE = FORCE_N[FORCE_N.length - 1];

type Frame = {
  step: number;
  forceN: number;
  eeMm: number;
  cupMm: number;
  remaining: number;
  mode: "GENERATING" | "SAFE_HOLD";
  executed: boolean;
  cutoff: boolean;
};

type SimResult = {
  cutoffStep: number | null;
  remaining: number;
  eeMm: number;
  eeAtCutoffMm: number;
  cupBeforeMm: number;
  cupAfterMm: number;
  naiveCupMm: number;
  naiveRemaining: number;
  objectRewound: boolean;
  poseHeld: boolean;
  speechRemaining: number;
  speechResettable: boolean;
  frames: Frame[];
};

function contactDisp(forceN: number) {
  return forceN > CONTACT ? forceN - CONTACT : 0;
}

function simulate(fMax: number): SimResult {
  const frames: Frame[] = [];
  let eeMm = 0;
  let cupMm = 0;
  let mode: Frame["mode"] = "GENERATING";
  let cutoffStep: number | null = null;
  let cupBeforeMm = 0;
  let eeAtCutoffMm = 0;
  let remaining = H;

  for (let step = 0; step < H; step += 1) {
    if (mode === "SAFE_HOLD") {
      remaining = 0;
      frames.push({
        step,
        forceN: FORCE_N[cutoffStep ?? step],
        eeMm,
        cupMm,
        remaining,
        mode,
        executed: false,
        cutoff: false,
      });
      continue;
    }

    const forceN = FORCE_N[step];
    const cupBeforeStep = cupMm;
    eeMm += EE_STEP;
    cupMm += contactDisp(forceN);
    remaining = H - step - 1;
    let cutoff = false;
    if (forceN > fMax) {
      cutoff = true;
      cutoffStep = step;
      cupBeforeMm = cupBeforeStep;
      eeAtCutoffMm = eeMm;
      remaining = 0;
      mode = "SAFE_HOLD";
    }
    frames.push({
      step,
      forceN,
      eeMm,
      cupMm,
      remaining,
      mode,
      executed: true,
      cutoff,
    });
  }

  for (let tick = 0; tick < HOLD_TICKS; tick += 1) {
    frames.push({
      step: H + tick,
      forceN: cutoffStep == null ? 0 : FORCE_N[cutoffStep],
      eeMm,
      cupMm,
      remaining: 0,
      mode: cutoffStep == null ? "GENERATING" : "SAFE_HOLD",
      executed: false,
      cutoff: false,
    });
  }

  const naiveCupMm = FORCE_N.reduce(
    (sum, forceN) => sum + contactDisp(forceN),
    0,
  );
  const naiveRemaining =
    cutoffStep == null ? 0 : H - cutoffStep - 1;

  return {
    cutoffStep,
    remaining: cutoffStep == null ? 0 : 0,
    eeMm,
    eeAtCutoffMm,
    cupBeforeMm,
    cupAfterMm: cupMm,
    naiveCupMm,
    naiveRemaining,
    objectRewound: cutoffStep != null && cupMm === cupBeforeMm,
    poseHeld: cutoffStep == null || eeMm === eeAtCutoffMm,
    speechRemaining: cutoffStep == null ? H : 0,
    speechResettable: cutoffStep != null,
    frames,
  };
}

export function Lesson40ForceCutoffLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    fMax: numberFrom(initialState, "fMax", 90, 20, 220),
    prediction: stringFrom(initialState, "prediction", ""),
  };
  const [fMax, setFMax] = useState(Math.round(defaults.fMax));
  const [prediction, setPrediction] = useState(defaults.prediction);
  const [ran, setRan] = useState(false);
  const [hasCutoff, setHasCutoff] = useState(false);
  const [hasIrreversible, setHasIrreversible] = useState(false);
  const [frameIndex, setFrameIndex] = useState(0);

  const simulation = useMemo(() => simulate(fMax), [fMax]);

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
    }, 160);
    return () => window.clearInterval(timer);
  }, [ran, simulation]);

  const frame =
    simulation.frames[Math.min(frameIndex, simulation.frames.length - 1)];
  const passed =
    ran &&
    prediction === "hold-irreversible" &&
    hasCutoff &&
    hasIrreversible &&
    simulation.poseHeld &&
    !simulation.objectRewound;
  const completion = useMemo(
    () => ({
      lessonId: 40,
      fMax,
      prediction,
      cutoffStep: simulation.cutoffStep,
      remaining: simulation.remaining,
      eeMm: simulation.eeMm,
      cupBeforeMm: simulation.cupBeforeMm,
      cupAfterMm: simulation.cupAfterMm,
      poseHeld: simulation.poseHeld,
      objectRewound: simulation.objectRewound,
      hasCutoff,
      hasIrreversible,
    }),
    [
      fMax,
      hasCutoff,
      hasIrreversible,
      prediction,
      simulation.cupAfterMm,
      simulation.cupBeforeMm,
      simulation.cutoffStep,
      simulation.eeMm,
      simulation.objectRewound,
      simulation.poseHeld,
      simulation.remaining,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function invalidate() {
    setRan(false);
    setFrameIndex(0);
  }

  function reset() {
    setFMax(90);
    setPrediction("");
    setRan(false);
    setHasCutoff(false);
    setHasIrreversible(false);
    setFrameIndex(0);
  }

  function runSim() {
    const result = simulate(fMax);
    const cutoff = result.cutoffStep != null && result.poseHeld;
    const irreversible =
      cutoff && result.cupAfterMm !== result.cupBeforeMm && !result.objectRewound;
    setHasCutoff((current) => current || cutoff);
    setHasIrreversible((current) => current || irreversible);
    setRan(true);
  }

  const tilt = 8 * Math.min(frame.step + 1, H);
  const cupPct = Math.min(86, 38 + frame.cupMm * 0.18);
  const forcePct = Math.min(100, (frame.forceN / PEAK_FORCE) * 100);
  const markPct = Math.min(100, (fMax / PEAK_FORCE) * 100);
  const pouring = frame.mode === "GENERATING" && frame.forceN > CONTACT;
  const spokenUntil =
    simulation.cutoffStep == null ? (ran ? Math.min(frame.step, H - 1) : -1) : simulation.cutoffStep;

  return (
    <LabFrame
      lesson="40"
      title="倒水时的力门限"
      description="教学模拟，不是模型输出。先预测超限后 chunk 和杯子会怎样，再调力阈值。超限必须停在当前姿态，剩余步清零；杯子不能退回超限前。右侧对照语音 PAUSE：未播 PCM 可丢，句子可重说。"
    >
      <div className={styles.workspace}>
        <form
          className={styles.controls}
          onSubmit={(event) => event.preventDefault()}
        >
          <h3>力门限旋钮</h3>
          <label>
            <span>
              F_max <output>{fMax} N</output>
            </span>
            <input
              type="range"
              min="20"
              max="220"
              step="5"
              value={fMax}
              onChange={(event) => {
                setFMax(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <p className={styles.note}>
            峰值接触力 {PEAK_FORCE} N。把阈值降到峰值以下才会切断；提到峰值以上则整块跑完，杯子被推得更远。
          </p>
        </form>

        <div className={styles.stage}>
          <div className={styles.formula}>
            <span>||F_i|| &gt; F_max 则丢掉 i+1..H-1，进入 SAFE_HOLD</span>
            <strong>
              {ran
                ? simulation.cutoffStep == null
                  ? `未切断 · 杯子 ${simulation.cupAfterMm} mm · 若切断前应在 ${simulation.cupBeforeMm || 0} mm`
                  : `i=${simulation.cutoffStep} · 剩余 ${simulation.remaining} · 末端 ${simulation.eeMm} mm`
                : "运行后揭晓切断步与剩余步"}
            </strong>
          </div>
          <div className={styles.desks}>
            <div className={styles.scene} aria-label="倒水切断面板">
              <div
                className={styles.pitcher}
                style={{ transform: `rotate(${ran ? tilt : 12}deg)` }}
              >
                <div className={styles.pitcherBody} />
                <div className={styles.pitcherSpout} />
                <div
                  className={`${styles.stream} ${pouring ? styles.streamOn : ""}`}
                  style={{ transform: `rotate(${ran ? Math.max(0, 28 - tilt) : 12}deg)` }}
                />
              </div>
              <div className={styles.table} />
              <div
                className={`${styles.cup} ${frame.cupMm > 80 ? styles.cupSpilled : ""}`}
                style={{ left: `${cupPct}%` }}
              />
              <div className={styles.caption}>
                <span>步 {ran ? frame.step : "—"} · {ran ? frame.mode : "等待预测后运行"}</span>
                <span>杯位移 {ran ? `${frame.cupMm} mm` : "—"}</span>
              </div>
            </div>
            <aside className={styles.speech}>
              <h4>语音 PAUSE 对照</h4>
              <div className={styles.tokens} aria-label="未播放 PCM">
                {SPEECH.map((token, index) => {
                  const dropped =
                    ran &&
                    simulation.cutoffStep != null &&
                    index > simulation.cutoffStep;
                  const done = ran && index <= spokenUntil && !dropped;
                  return (
                    <span
                      key={token}
                      className={`${styles.token} ${done ? styles.tokenDone : ""} ${dropped ? styles.tokenDropped : ""}`}
                    >
                      {token}
                    </span>
                  );
                })}
              </div>
              <p className={styles.speechNote}>
                {ran
                  ? simulation.cutoffStep == null
                    ? "未触发切断，句子整段说完。把 F_max 降到峰值以下。"
                    : `PAUSE 丢掉剩余 PCM（剩余 ${simulation.speechRemaining} 帧），句子可从空缓冲重说。杯子不能同样重来。`
                  : "超限时刻对应停嘴：未播的字可以丢，已经洒出的水不能收回。"}
              </p>
            </aside>
          </div>
          <div className={styles.meter} aria-label="接触力">
            <div
              className={`${styles.meterFill} ${frame.forceN > fMax ? styles.meterFillHot : ""}`}
              style={{ width: ran ? `${forcePct}%` : "0%" }}
            />
            <div className={styles.meterMark} style={{ left: `${markPct}%` }} />
          </div>
          <dl className={styles.metrics}>
            <div>
              <dt>切断步 i</dt>
              <dd>
                {ran
                  ? simulation.cutoffStep == null
                    ? "无"
                    : String(simulation.cutoffStep)
                  : "—"}
              </dd>
            </div>
            <div>
              <dt>剩余步</dt>
              <dd>{ran ? String(frame.remaining) : "—"}</dd>
            </div>
            <div>
              <dt>末端</dt>
              <dd>{ran ? `${frame.eeMm} mm` : "—"}</dd>
            </div>
            <div>
              <dt>杯子 vs 超限前</dt>
              <dd>
                {ran
                  ? simulation.cutoffStep == null
                    ? `${simulation.cupAfterMm} / 未超限`
                    : `${simulation.cupAfterMm} / ${simulation.cupBeforeMm} mm`
                  : "—"}
              </dd>
            </div>
          </dl>
          <div className={styles.findings}>
            <span className={hasCutoff ? styles.ok : undefined}>
              {hasCutoff ? "已看到超限后停在当前姿态" : "还没触发 SAFE_HOLD"}
            </span>
            <span className={hasIrreversible ? styles.ok : undefined}>
              {hasIrreversible
                ? "杯子停在超限后位置，不能回退"
                : "还没看到物体不可回放"}
            </span>
          </div>
        </div>
      </div>

      <div className={styles.predict}>
        <fieldset>
          <legend>先预测：力超限的那一步之后，动作块和杯子会怎样？</legend>
          {[
            [
              "continue-rewind",
              "剩余 chunk 继续执行，杯子能回到超限前位置",
            ],
            [
              "hold-irreversible",
              "进入 SAFE_HOLD，末端停在当前姿态，剩余步为 0，杯子停在超限时的位置且不能回退",
            ],
            [
              "pause-undo",
              "和语音 PAUSE 一样：未执行动作与洒出的水都可以 undo",
            ],
            [
              "replan-origin",
              "自动 REPLAN，末端和杯子都回到起点",
            ],
          ].map(([value, label]) => (
            <label key={value}>
              <input
                type="radio"
                name="force-cutoff-prediction"
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
            运行倒水
          </button>
        </div>
      </div>
      {ran && prediction !== "hold-irreversible" && (
        <p className={styles.feedback}>
          语音那边未播的字可以划掉重说。倒水这边，超限步已经把杯子推离原位。把 F_max
          降到峰值以下，看剩余步是否变成 0、末端是否还往前走。
        </p>
      )}
      {ran && prediction === "hold-irreversible" && (
        <p className={styles.feedback}>
          {simulation.cutoffStep == null
            ? `当前阈值 ${fMax} N 高于本段峰值 ${PEAK_FORCE} N，整块跑完。把阈值降到 100 N 附近才会切断。`
            : `第 ${simulation.cutoffStep} 步 ${FORCE_N[simulation.cutoffStep]} N 超过 ${fMax} N。剩余 ${simulation.remaining} 步；若不切断，杯子会到 ${simulation.naiveCupMm} mm。`}
        </p>
      )}
      <Gate passed={passed}>
        先选对预测，再调出力超限：末端停在当前姿态、剩余步为 0、杯子不能回到超限前。对照语音
        PAUSE 只丢掉 PCM。数字来自运动学夹具，不能写成真机或 ISO 认证。
      </Gate>
    </LabFrame>
  );
}
