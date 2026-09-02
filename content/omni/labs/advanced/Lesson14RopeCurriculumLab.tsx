"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import {
  numberFrom,
  round,
  stringFrom,
  useCompletionGate,
} from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson14RopeCurriculumLab.module.css";

const thetaChoices = [10_000, 50_000, 100_000, 500_000, 1_000_000];

export function Lesson14RopeCurriculumLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    theta: numberFrom(initialState, "theta", 100_000, 10_000, 1_000_000),
    source: numberFrom(initialState, "sourceContext", 4096, 2048, 8192),
    target: numberFrom(initialState, "targetContext", 32768, 8192, 131072),
    stages: numberFrom(initialState, "stages", 4, 2, 7),
    pair: numberFrom(initialState, "pairIndex", 28, 0, 31),
    prediction: stringFrom(initialState, "prediction", ""),
  };
  const [theta, setTheta] = useState(defaults.theta);
  const [source, setSource] = useState(defaults.source);
  const [target, setTarget] = useState(defaults.target);
  const [stages, setStages] = useState(defaults.stages);
  const [pair, setPair] = useState(defaults.pair);
  const [prediction, setPrediction] = useState(defaults.prediction);
  const [ran, setRan] = useState(false);

  const calculation = useMemo(() => {
    const exponent = (2 * pair) / 64;
    const inverseFrequency = 1 / theta ** exponent;
    const wavelength = (2 * Math.PI) / inverseFrequency;
    const cycles = target / wavelength;
    const ratio = (target / source) ** (1 / Math.max(1, stages - 1));
    const curriculum = Array.from({ length: stages }, (_, index) =>
      Math.round((source * ratio ** index) / 256) * 256,
    );
    curriculum[curriculum.length - 1] = target;
    const phases = Array.from({ length: 32 }, (_, index) => {
      const position = (index / 31) * target;
      return (position * inverseFrequency) % (2 * Math.PI);
    });
    return { exponent, inverseFrequency, wavelength, cycles, ratio, curriculum, phases };
  }, [pair, source, stages, target, theta]);

  const passed =
    ran && prediction === "increase-theta" && calculation.ratio <= 2.05;
  const completion = useMemo(
    () => ({
      lessonId: 14,
      theta,
      pairIndex: pair,
      wavelength: round(calculation.wavelength, 1),
      sourceContext: source,
      targetContext: target,
      stages,
      curriculum: calculation.curriculum,
      adjacentStageRatio: round(calculation.ratio, 3),
    }),
    [calculation, pair, source, stages, target, theta],
  );
  useCompletionGate(passed, onComplete, completion);

  function reset() {
    setTheta(defaults.theta);
    setSource(defaults.source);
    setTarget(defaults.target);
    setStages(defaults.stages);
    setPair(defaults.pair);
    setPrediction("");
    setRan(false);
  }

  return (
    <LabFrame
      lesson="14"
      title="看见 RoPE 的旋转，再设计长度课程"
      description="先计算各频带的旋转周期，再按几何级数增加上下文长度，观察位置外推怎样变化。"
    >
      <div className={styles.layout}>
        <section className={styles.controls}>
          <h3>频率实验台</h3>
          <label>
            <span>RoPE θ</span>
            <select
              value={theta}
              onChange={(event) => {
                setTheta(Number(event.target.value));
                setRan(false);
              }}
            >
              {thetaChoices.map((value) => (
                <option key={value} value={value}>
                  {value.toLocaleString()}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>通道对 i <output>{pair}/31</output></span>
            <input
              type="range"
              min="0"
              max="31"
              value={pair}
              onChange={(event) => {
                setPair(Number(event.target.value));
                setRan(false);
              }}
            />
          </label>
          <label>
            <span>起始长度</span>
            <select
              value={source}
              onChange={(event) => {
                const next = Number(event.target.value);
                setSource(next);
                setTarget((current) => Math.max(current, next * 2));
                setRan(false);
              }}
            >
              {[2048, 4096, 8192].map((value) => (
                <option key={value}>{value}</option>
              ))}
            </select>
          </label>
          <label>
            <span>目标长度</span>
            <select
              value={target}
              onChange={(event) => {
                setTarget(Number(event.target.value));
                setRan(false);
              }}
            >
              {[8192, 16384, 32768, 65536, 131072]
                .filter((value) => value > source)
                .map((value) => (
                  <option key={value}>{value}</option>
                ))}
            </select>
          </label>
          <label>
            <span>课程阶段 <output>{stages}</output></span>
            <input
              type="range"
              min="2"
              max="7"
              value={stages}
              onChange={(event) => {
                setStages(Number(event.target.value));
                setRan(false);
              }}
            />
          </label>
        </section>

        <section className={styles.phaseLab}>
          <div className={styles.formula}>
            <p>ωᵢ = θ<sup>−2i/d</sup>　·　λᵢ = 2π / ωᵢ</p>
            <strong>
              λ<sub>{pair}</sub> = {calculation.wavelength.toLocaleString(undefined, { maximumFractionDigits: 1 })} tokens
            </strong>
          </div>
          <div className={styles.phaseStrip} aria-label="目标上下文中的 RoPE 相位采样">
            {calculation.phases.map((phase, index) => (
              <i
                key={index}
                title={`位置 ${Math.round((index / 31) * target).toLocaleString()}，相位 ${phase.toFixed(2)} rad`}
                style={{
                  "--phase": `${(phase / (2 * Math.PI)) * 360}`,
                } as React.CSSProperties}
              />
            ))}
          </div>
          <div className={styles.axis}>
            <span>p = 0</span>
            <span>p = {target.toLocaleString()}</span>
          </div>
          <dl className={styles.readout}>
            <div>
              <dt>目标区间内旋转</dt>
              <dd>{ran ? `${calculation.cycles.toFixed(2)} 圈` : "—"}</dd>
            </div>
            <div>
              <dt>角速度</dt>
              <dd>{calculation.inverseFrequency.toExponential(2)} rad/token</dd>
            </div>
          </dl>
        </section>
      </div>

      <section className={styles.curriculum}>
        <header>
          <div>
            <h3>Context curriculum</h3>
            <p>Lₙ = L₀ × rⁿ，r = (target / source)<sup>1/(N−1)</sup></p>
          </div>
          <strong>相邻倍率 {calculation.ratio.toFixed(2)}×</strong>
        </header>
        <div className={styles.stages}>
          {calculation.curriculum.map((length, index) => (
            <div key={index}>
              <span>S{index + 1}</span>
              <i
                style={{ width: `${Math.max(8, (length / target) * 100)}%` }}
                aria-hidden="true"
              />
              <b>{length.toLocaleString()}</b>
            </div>
          ))}
        </div>
      </section>

      <div className={styles.challenge}>
        <fieldset>
          <legend>先预测：保持 i、d 与位置不变，怎样让旋转更慢、波长更长？</legend>
          <label>
            <input
              type="radio"
              name="rope-prediction"
              checked={prediction === "increase-theta"}
              onChange={() => {
                setPrediction("increase-theta");
                setRan(false);
              }}
            />
            增大 θ
          </label>
          <label>
            <input
              type="radio"
              name="rope-prediction"
              checked={prediction === "decrease-theta"}
              onChange={() => {
                setPrediction("decrease-theta");
                setRan(false);
              }}
            />
            减小 θ
          </label>
        </fieldset>
        <div>
          <button type="button" onClick={reset}>重置</button>
          <button
            type="button"
            className={styles.primary}
            disabled={!prediction}
            onClick={() => setRan(true)}
          >
            运行长度实验
          </button>
        </div>
      </div>
      <Gate passed={passed}>
        预测 RoPE 方向正确，并让课程的相邻长度倍率不超过约 2×（避免一步跨得过大）。
      </Gate>
    </LabFrame>
  );
}
