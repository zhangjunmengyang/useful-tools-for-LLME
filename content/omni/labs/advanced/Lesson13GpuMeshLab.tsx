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
import styles from "./Lesson13GpuMeshLab.module.css";

type Dimension = "dp" | "ep" | "cp";

const factors = [1, 2, 4, 8];
const labels: Record<Dimension, string> = {
  dp: "Data Parallel",
  ep: "Expert Parallel",
  cp: "Context Parallel",
};

export function Lesson13GpuMeshLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    dp: numberFrom(initialState, "dp", 2, 1, 8),
    ep: numberFrom(initialState, "ep", 2, 1, 8),
    cp: numberFrom(initialState, "cp", 2, 1, 8),
    gradientGb: numberFrom(initialState, "gradientShardGb", 4, 1, 24),
    sequence: numberFrom(initialState, "sequence", 16384, 4096, 65536),
    hidden: numberFrom(initialState, "hidden", 4096, 2048, 8192),
    microbatch: numberFrom(initialState, "microbatch", 2, 1, 8),
    prediction: stringFrom(initialState, "prediction", ""),
  };
  const [dp, setDp] = useState(defaults.dp);
  const [ep, setEp] = useState(defaults.ep);
  const [cp, setCp] = useState(defaults.cp);
  const [gradientGb, setGradientGb] = useState(defaults.gradientGb);
  const [sequence, setSequence] = useState(defaults.sequence);
  const [hidden, setHidden] = useState(defaults.hidden);
  const [microbatch, setMicrobatch] = useState(defaults.microbatch);
  const [prediction, setPrediction] = useState(defaults.prediction);
  const [focus, setFocus] = useState<Dimension>("dp");
  const [ran, setRan] = useState(false);

  const valid = dp * ep * cp === 8;
  const calculation = useMemo(() => {
    const activationGb = (microbatch * sequence * hidden * 2) / 1e9;
    const volumes: Record<Dimension, number> = {
      dp: dp > 1 ? (2 * (dp - 1) * gradientGb) / dp : 0,
      ep: ep > 1 ? (2 * (ep - 1) * activationGb) / ep : 0,
      cp: cp > 1 ? (2 * (cp - 1) * 2 * activationGb) / cp : 0,
    };
    const dominant = (Object.entries(volumes) as [Dimension, number][]).sort(
      (a, b) => b[1] - a[1],
    )[0][0];
    return { activationGb, volumes, dominant };
  }, [cp, dp, ep, gradientGb, hidden, microbatch, sequence]);

  const gpuCoordinates = useMemo(() => {
    if (!valid) return [];
    return Array.from({ length: 8 }, (_, rank) => {
      const c = rank % cp;
      const e = Math.floor(rank / cp) % ep;
      const d = Math.floor(rank / (cp * ep)) % dp;
      const group =
        focus === "dp" ? e * cp + c : focus === "ep" ? d * cp + c : d * ep + e;
      return { rank, d, e, c, group };
    });
  }, [cp, dp, ep, focus, valid]);

  const passed =
    ran && valid && prediction === calculation.dominant;
  const completion = useMemo(
    () => ({
      lessonId: 13,
      mesh: { dp, ep, cp },
      sequence,
      hidden,
      microbatch,
      gradientShardGb: gradientGb,
      communicationGbPerRank: {
        dp: round(calculation.volumes.dp, 3),
        ep: round(calculation.volumes.ep, 3),
        cp: round(calculation.volumes.cp, 3),
      },
      dominant: calculation.dominant,
    }),
    [calculation, cp, dp, ep, gradientGb, hidden, microbatch, sequence],
  );
  useCompletionGate(passed, onComplete, completion);

  function reset() {
    setDp(defaults.dp);
    setEp(defaults.ep);
    setCp(defaults.cp);
    setGradientGb(defaults.gradientGb);
    setSequence(defaults.sequence);
    setHidden(defaults.hidden);
    setMicrobatch(defaults.microbatch);
    setPrediction("");
    setFocus("dp");
    setRan(false);
  }

  function applyPreset(nextDp: number, nextEp: number, nextCp: number) {
    setDp(nextDp);
    setEp(nextEp);
    setCp(nextCp);
    setRan(false);
  }

  return (
    <LabFrame
      lesson="13"
      title="把 8 张卡织成三维并行网格"
      description="DP、EP、CP 不是三个独立开关：它们的乘积必须刚好覆盖 8 个 rank。搭网格、检查通信组，再算每个 rank 的理论通信字节。"
    >
      <div className={styles.cockpit}>
        <section className={styles.meshPanel}>
          <div className={styles.meshHeader}>
            <div>
              <h3>8-GPU mesh</h3>
              <p className={valid ? styles.valid : styles.invalid}>
                {dp} × {ep} × {cp} = {dp * ep * cp} {valid ? "✓" : "≠ 8"}
              </p>
            </div>
            <div className={styles.presets} aria-label="网格预设">
              <button type="button" onClick={() => applyPreset(8, 1, 1)}>8·1·1</button>
              <button type="button" onClick={() => applyPreset(2, 2, 2)}>2·2·2</button>
              <button type="button" onClick={() => applyPreset(1, 4, 2)}>1·4·2</button>
            </div>
          </div>
          <div className={styles.factorControls}>
            {(["dp", "ep", "cp"] as Dimension[]).map((dimension) => {
              const value = { dp, ep, cp }[dimension];
              const setter = { dp: setDp, ep: setEp, cp: setCp }[dimension];
              return (
                <label key={dimension}>
                  <span>{dimension.toUpperCase()}</span>
                  <select
                    value={value}
                    onChange={(event) => {
                      setter(Number(event.target.value));
                      setRan(false);
                    }}
                  >
                    {factors.map((factor) => (
                      <option key={factor}>{factor}</option>
                    ))}
                  </select>
                </label>
              );
            })}
          </div>
          <div className={styles.dimensionTabs} role="tablist" aria-label="查看通信组">
            {(["dp", "ep", "cp"] as Dimension[]).map((dimension) => (
              <button
                type="button"
                role="tab"
                aria-selected={focus === dimension}
                key={dimension}
                onClick={() => setFocus(dimension)}
              >
                {dimension.toUpperCase()} groups
              </button>
            ))}
          </div>
          {valid ? (
            <div className={styles.gpuGrid} aria-label={`${labels[focus]} 通信组`}>
              {gpuCoordinates.map(({ rank, d, e, c, group }) => (
                <div
                  className={styles.gpu}
                  key={rank}
                  style={{ "--group-hue": String(145 + group * 37) } as React.CSSProperties}
                >
                  <span>GPU {rank}</span>
                  <b>d{d} · e{e} · c{c}</b>
                  <small>{focus.toUpperCase()} group {group}</small>
                </div>
              ))}
            </div>
          ) : (
            <div className={styles.emptyMesh}>
              当前需要 {dp * ep * cp} 张卡。请让 DP × EP × CP = 8。
            </div>
          )}
        </section>

        <section className={styles.workload}>
          <h3>通信载荷</h3>
          <label>
            <span>梯度 shard <output>{gradientGb.toFixed(1)} GB</output></span>
            <input
              type="range"
              min="1"
              max="24"
              step="0.5"
              value={gradientGb}
              onChange={(event) => {
                setGradientGb(Number(event.target.value));
                setRan(false);
              }}
            />
          </label>
          <label>
            <span>序列长度 <output>{sequence.toLocaleString()}</output></span>
            <input
              type="range"
              min="4096"
              max="65536"
              step="4096"
              value={sequence}
              onChange={(event) => {
                setSequence(Number(event.target.value));
                setRan(false);
              }}
            />
          </label>
          <label>
            <span>Hidden size</span>
            <select
              value={hidden}
              onChange={(event) => {
                setHidden(Number(event.target.value));
                setRan(false);
              }}
            >
              {[2048, 4096, 6144, 8192].map((value) => (
                <option key={value}>{value}</option>
              ))}
            </select>
          </label>
          <label>
            <span>Micro batch</span>
            <select
              value={microbatch}
              onChange={(event) => {
                setMicrobatch(Number(event.target.value));
                setRan(false);
              }}
            >
              {[1, 2, 4, 8].map((value) => (
                <option key={value}>{value}</option>
              ))}
            </select>
          </label>
          <p className={styles.assumption}>
            BF16 activation A = B × S × H × 2 bytes ={" "}
            {calculation.activationGb.toFixed(3)} GB
          </p>
        </section>
      </div>

      <section className={styles.traffic} aria-label="理论通信量">
        {(["dp", "ep", "cp"] as Dimension[]).map((dimension) => {
          const volume = calculation.volumes[dimension];
          const peak = Math.max(...Object.values(calculation.volumes), 0.0001);
          const formula =
            dimension === "dp"
              ? "2(D−1)/D × gradient"
              : dimension === "ep"
                ? "2(E−1)/E × A"
                : "2(C−1)/C × 2A (K+V)";
          return (
            <div key={dimension}>
              <header>
                <b>{dimension.toUpperCase()}</b>
                <span>{formula}</span>
              </header>
              <div className={styles.trafficBar}>
                <i style={{ width: ran && valid ? `${(volume / peak) * 100}%` : "0%" }} />
              </div>
              <strong>{ran && valid ? `${volume.toFixed(3)} GB / rank` : "—"}</strong>
            </div>
          );
        })}
      </section>

      <div className={styles.challenge}>
        <fieldset>
          <legend>先预测：当前参数下，哪一维的理论每-rank通信量最大？</legend>
          {(["dp", "ep", "cp"] as Dimension[]).map((dimension) => (
            <label key={dimension}>
              <input
                type="radio"
                name="mesh-prediction"
                checked={prediction === dimension}
                onChange={() => {
                  setPrediction(dimension);
                  setRan(false);
                }}
              />
              {dimension.toUpperCase()}
            </label>
          ))}
        </fieldset>
        <div>
          <button type="button" onClick={reset}>重置</button>
          <button
            type="button"
            className={styles.run}
            disabled={!prediction}
            onClick={() => setRan(true)}
          >
            运行通信账本
          </button>
        </div>
      </div>
      {ran && valid && prediction !== calculation.dominant && (
        <p className={styles.feedback}>
          预测未命中。比较上面的公式：DP 用梯度 shard；EP/CP 用 activation A，CP 还要交换 K 与 V。
        </p>
      )}
      <Gate passed={passed}>
        先构造合法的 8-rank 网格，再正确预测当前工作负载的最大通信维度。
      </Gate>
    </LabFrame>
  );
}
