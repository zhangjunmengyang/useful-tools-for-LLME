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
import styles from "./Lesson54SynthDataLab.module.css";

type DomainId = "kitchen" | "drawer" | "mug" | "bimanual";
type SynthTarget = DomainId;
type SynthMode = "unique" | "duplicate";
type Prediction = "to-largest-worse" | "any-helps" | "dup-raises-neff";

type Domain = {
  id: DomainId;
  name: string;
  n: number;
  skill: string;
};

const DOMAINS: Domain[] = [
  { id: "kitchen", name: "厨房抓放", n: 8000, skill: "常见 pick-place" },
  { id: "drawer", name: "抽屉开合", n: 800, skill: "关节家具" },
  { id: "mug", name: "杯插入位", n: 200, skill: "精密插入" },
  { id: "bimanual", name: "双臂摆盘", n: 50, skill: "长尾双臂" },
];

const REAL_COUNTS = Object.fromEntries(
  DOMAINS.map((domain) => [domain.id, domain.n]),
) as Record<DomainId, number>;
const REAL_TOTAL = DOMAINS.reduce((sum, domain) => sum + domain.n, 0);
const LARGEST: DomainId = "kitchen";
const SMALLEST: DomainId = "bimanual";
const BATCH = 256;
const DEFF_DROP = 0.04;
const SMALL_GAIN = 0.15;

const PREDICTIONS: Array<[Prediction, string]> = [
  ["to-largest-worse", "合成只加到最大域时，α=1 的有效域数会下降"],
  ["any-helps", "不管加到哪一域，有效域数都会上升"],
  ["dup-raises-neff", "把同一条轨迹复制多份会提高有效样本量"],
];

const TARGETS: Array<[SynthTarget, string]> = [
  ["kitchen", "只加到厨房（最大域）"],
  ["drawer", "加到抽屉"],
  ["mug", "加到杯插入位"],
  ["bimanual", "补双臂摆盘（最小域）"],
];

const MODES: Array<[SynthMode, string]> = [
  ["unique", "新复位位姿（唯一哈希）"],
  ["duplicate", "重复已有轨迹"],
];

function mixture(
  counts: Record<DomainId, number>,
  alpha: number,
) {
  const weights = DOMAINS.map((domain) => counts[domain.id] ** alpha);
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  const probs = weights.map((weight) => weight / total);
  const shares = Object.fromEntries(
    DOMAINS.map((domain, index) => [domain.id, probs[index]]),
  ) as Record<DomainId, number>;
  const effective = 1 / probs.reduce((sum, probability) => sum + probability * probability, 0);
  return { shares, effective, maxShare: Math.max(...probs), minShare: Math.min(...probs) };
}

function ledger(target: SynthTarget, mode: SynthMode, synthCount: number, alpha: number) {
  const counts = { ...REAL_COUNTS };
  counts[target] += synthCount;
  const mix = mixture(counts, alpha);
  const realMix = mixture(REAL_COUNTS, alpha);
  const nRaw = REAL_TOTAL + synthCount;
  let nUnique = REAL_TOTAL;
  let nEff = REAL_TOTAL;
  if (synthCount > 0 && mode === "unique") {
    nUnique = nRaw;
    nEff = nRaw;
  } else if (synthCount > 0 && mode === "duplicate") {
    nUnique = REAL_TOTAL;
    const one = 1 + synthCount;
    const rest = REAL_TOTAL - 1;
    nEff = (nRaw * nRaw) / (one * one + rest);
  }
  return {
    counts,
    mix,
    realMix,
    nRaw,
    nUnique,
    nEff,
    nEffReal: REAL_TOTAL,
    smallShare: mix.shares[SMALLEST],
    largeShare: mix.shares[LARGEST],
    realSmallShare: realMix.shares[SMALLEST],
    realLargeShare: realMix.shares[LARGEST],
    realEffective: realMix.effective,
  };
}

export function Lesson54SynthDataLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    alpha: numberFrom(initialState, "alpha", 1, 0, 1),
    synthCount: numberFrom(initialState, "synthCount", 2000, 0, 4000),
    target: stringFrom(initialState, "target", "kitchen") as SynthTarget,
    mode: stringFrom(initialState, "mode", "unique") as SynthMode,
    prediction: stringFrom(initialState, "prediction", "") as Prediction | "",
  };
  const [alpha, setAlpha] = useState(defaults.alpha);
  const [synthCount, setSynthCount] = useState(defaults.synthCount);
  const [target, setTarget] = useState<SynthTarget>(
    DOMAINS.some((domain) => domain.id === defaults.target)
      ? defaults.target
      : "kitchen",
  );
  const [mode, setMode] = useState<SynthMode>(
    defaults.mode === "duplicate" ? "duplicate" : "unique",
  );
  const [prediction, setPrediction] = useState<Prediction | "">(
    defaults.prediction === "to-largest-worse" ||
      defaults.prediction === "any-helps" ||
      defaults.prediction === "dup-raises-neff"
      ? defaults.prediction
      : "",
  );
  const [ran, setRan] = useState(false);
  const [seenLargestWorse, setSeenLargestWorse] = useState(false);
  const [seenSmallestRecover, setSeenSmallestRecover] = useState(false);
  const [seenDuplicate, setSeenDuplicate] = useState(false);

  const calculation = useMemo(
    () => ledger(target, mode, synthCount, alpha),
    [alpha, mode, synthCount, target],
  );

  const passed =
    ran &&
    prediction === "to-largest-worse" &&
    seenLargestWorse &&
    seenSmallestRecover &&
    seenDuplicate;

  const completion = useMemo(
    () => ({
      lessonId: 54,
      alpha,
      synthCount,
      target,
      mode,
      prediction,
      maxShare: round(calculation.mix.maxShare, 4),
      smallShare: round(calculation.smallShare, 4),
      effectiveDomains: round(calculation.mix.effective, 3),
      nUnique: calculation.nUnique,
      nEff: round(calculation.nEff, 2),
      seenLargestWorse,
      seenSmallestRecover,
      seenDuplicate,
    }),
    [
      alpha,
      calculation.mix.effective,
      calculation.mix.maxShare,
      calculation.nEff,
      calculation.nUnique,
      calculation.smallShare,
      mode,
      prediction,
      seenDuplicate,
      seenLargestWorse,
      seenSmallestRecover,
      synthCount,
      target,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function reveal() {
    const next = ledger(target, mode, synthCount, alpha);
    if (
      Math.abs(alpha - 1) < 1e-9 &&
      target === LARGEST &&
      synthCount >= 1500 &&
      next.mix.effective < next.realEffective - DEFF_DROP &&
      next.largeShare > next.realLargeShare
    ) {
      setSeenLargestWorse(true);
    }
    if (
      Math.abs(alpha - 1) < 1e-9 &&
      target === SMALLEST &&
      synthCount >= 1500 &&
      next.smallShare > next.realSmallShare + SMALL_GAIN
    ) {
      setSeenSmallestRecover(true);
    }
    if (
      mode === "duplicate" &&
      synthCount >= 1500 &&
      next.nUnique === REAL_TOTAL &&
      next.nEff < next.nEffReal
    ) {
      setSeenDuplicate(true);
    }
    setRan(true);
  }

  function reset() {
    setAlpha(1);
    setSynthCount(2000);
    setTarget("kitchen");
    setMode("unique");
    setPrediction("");
    setRan(false);
    setSeenLargestWorse(false);
    setSeenSmallestRecover(false);
    setSeenDuplicate(false);
  }

  const hidden = !ran;
  const dash = "—";

  return (
    <LabFrame
      lesson="54"
      title="把合成轨迹加到哪一域"
      description="四个真实操作域，条数差两个数量级。先预测再揭晓：合成只灌进最大域时 α=1 更糟；补最小域时小域频率回升；重复轨迹不得增加有效样本量。教学模拟，不是 MimicGen 前向输出。"
    >
      <div className={styles.workspace}>
        <form className={styles.controls} onSubmit={(event) => event.preventDefault()}>
          <h3>合成控制台</h3>
          <label>
            <span>
              温度 α <output>{alpha.toFixed(2)}</output>
            </span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={alpha}
              onChange={(event) => {
                setAlpha(Number(event.target.value));
                setRan(false);
              }}
            />
          </label>
          <label>
            <span>
              合成条数 <output>{synthCount}</output>
            </span>
            <input
              type="range"
              min="0"
              max="4000"
              step="100"
              value={synthCount}
              onChange={(event) => {
                setSynthCount(Number(event.target.value));
                setRan(false);
              }}
            />
          </label>
          <fieldset className={styles.fieldset}>
            <legend>加到哪一域</legend>
            <div className={styles.toggleRow}>
              {TARGETS.map(([value, label]) => (
                <label key={value}>
                  <input
                    type="radio"
                    name="synth-target"
                    checked={target === value}
                    onChange={() => {
                      setTarget(value);
                      setRan(false);
                    }}
                  />
                  <span>{label}</span>
                </label>
              ))}
            </div>
          </fieldset>
          <fieldset className={styles.fieldset}>
            <legend>合成怎么写</legend>
            <div className={styles.toggleRow}>
              {MODES.map(([value, label]) => (
                <label key={value}>
                  <input
                    type="radio"
                    name="synth-mode"
                    checked={mode === value}
                    onChange={() => {
                      setMode(value);
                      setRan(false);
                    }}
                  />
                  <span>{label}</span>
                </label>
              ))}
            </div>
          </fieldset>
        </form>

        <div className={styles.stage}>
          <p className={styles.formula}>
            <span>
              p_d ∝ n_d<sup>α</sup>
            </span>
            <span>
              D_eff = 1 / Σ p_d²{" "}
              <strong>
                {hidden ? dash : round(calculation.mix.effective, 3)}
              </strong>
            </span>
            <span>
              n_eff = (Σ n_i)² / Σ n_i²{" "}
              <strong>{hidden ? dash : round(calculation.nEff, 1)}</strong>
            </span>
            <span>
              B={BATCH}
            </span>
          </p>

          <div className={styles.domains}>
            {DOMAINS.map((domain) => {
              const real = REAL_COUNTS[domain.id];
              const after = calculation.counts[domain.id];
              const share = calculation.mix.shares[domain.id];
              return (
                <div className={styles.domain} key={domain.id}>
                  <div className={styles.domainHeader}>
                    <b>{domain.name}</b>
                    <small>
                      真实 {real}
                      {after !== real ? ` +合成 ${after - real}` : ""}
                      {" · "}
                      {domain.skill}
                    </small>
                  </div>
                  <div className={styles.barTrack}>
                    <div
                      className={styles.barFill}
                      style={{
                        width: hidden ? "0%" : `${Math.max(2, share * 100)}%`,
                        opacity: hidden ? 0.25 : 1,
                      }}
                    />
                  </div>
                  <p className={styles.share}>
                    {hidden ? dash : `${(share * 100).toFixed(1)}%`}
                  </p>
                </div>
              );
            })}
          </div>

          <dl className={styles.metrics}>
            <div>
              <dt>最大域份额</dt>
              <dd>
                {hidden
                  ? dash
                  : `${(calculation.largeShare * 100).toFixed(1)}%`}
              </dd>
            </div>
            <div>
              <dt>最小域份额</dt>
              <dd>
                {hidden
                  ? dash
                  : `${(calculation.smallShare * 100).toFixed(2)}%`}
              </dd>
            </div>
            <div>
              <dt>原始条数</dt>
              <dd>{hidden ? dash : calculation.nRaw}</dd>
            </div>
            <div>
              <dt>唯一哈希</dt>
              <dd>{hidden ? dash : calculation.nUnique}</dd>
            </div>
          </dl>

          <div className={styles.probe} data-warn={mode === "duplicate" && ran}>
            <strong>
              {hidden
                ? "揭晓后对照合成前后的 D_eff 与 n_eff"
                : mode === "duplicate"
                  ? "重复轨迹：唯一哈希不变，有效样本量下降"
                  : target === LARGEST
                    ? "灌进最大域：α=1 时厨房更满，有效域数更低"
                    : target === SMALLEST
                      ? "补最小域：双臂份额回升，有效域数上升"
                      : "合成进了中间域，验收清单还不会全亮"}
            </strong>
            <p>
              {hidden
                ? "先提交预测。数字被遮住，避免对着已经画好的条形图补理由。"
                : `真实 D_eff ${round(calculation.realEffective, 3)}；当前 D_eff ${round(calculation.mix.effective, 3)}。真实 n_eff ${calculation.nEffReal}；当前 n_eff ${round(calculation.nEff, 1)}。`}
            </p>
          </div>

          <ul className={styles.checklist}>
            <li data-done={seenLargestWorse}>
              合成只加到最大域，α=1 更糟
            </li>
            <li data-done={seenSmallestRecover}>
              合成补最小域，小域频率回升
            </li>
            <li data-done={seenDuplicate}>
              重复轨迹不增加有效样本量
            </li>
          </ul>

          <div className={styles.predict}>
            <fieldset>
              <legend>先预测再揭晓</legend>
              {PREDICTIONS.map(([value, label]) => (
                <label key={value}>
                  <input
                    type="radio"
                    name="synth-prediction"
                    checked={prediction === value}
                    onChange={() => {
                      setPrediction(value);
                      setRan(false);
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
                onClick={reveal}
              >
                揭晓账本
              </button>
            </div>
          </div>
          {ran && prediction !== "to-largest-worse" ? (
            <p className={styles.feedback}>
              预测选错时仍可看条形图，但验收不通过。正确项是：合成只加到最大域时 α=1 更糟。
            </p>
          ) : null}
          {ran && prediction === "to-largest-worse" && !passed ? (
            <p className={styles.feedback}>
              还要切换三次配置再揭晓：α=1 且合成≥1500，分别对准厨房、双臂，以及打开“重复已有轨迹”。
            </p>
          ) : null}
        </div>
      </div>
      <Gate passed={passed}>
        {passed
          ? "三项清单已亮：灌进最大域会压低有效域数，补最小域会抬高小域频率，重复轨迹不增加 n_eff。"
          : "先选预测，再分别揭晓“加到厨房 / 补双臂 / 重复轨迹”三组。默认把合成灌进厨房，就是工业常见错误。"}
      </Gate>
    </LabFrame>
  );
}
