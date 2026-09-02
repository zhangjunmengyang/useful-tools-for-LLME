"use client";

import { useEffect, useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import { numberFrom, round, stringFrom, useCompletionGate } from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson50TrellisLab.module.css";

type DecoderId = "mesh" | "gaussian" | "nerf";

type Voxel = {
  x: number;
  y: number;
  z: number;
  latent: readonly [number, number, number, number];
};

const ACTIVE: readonly Voxel[] = [
  { x: 1, y: 1, z: 1, latent: [-0.15, 0.35, 0.48, 0.09] },
  { x: 2, y: 1, z: 1, latent: [0.05, 0.35, 0.48, 0.12] },
  { x: 1, y: 1, z: 2, latent: [-0.15, 0.35, 0.56, 0.12] },
  { x: 2, y: 1, z: 2, latent: [0.05, 0.35, 0.56, 0.15] },
  { x: 0, y: 1, z: 1, latent: [-0.35, 0.35, 0.48, 0.06] },
  { x: 0, y: 2, z: 1, latent: [-0.35, 0.45, 0.48, 0.09] },
];

function softplus(value: number) {
  return value > 20 ? value : Math.log1p(Math.exp(value));
}

function meshSdf(latent: readonly [number, number, number, number]) {
  const z0 = latent[0];
  const z3 = latent[3];
  const values: number[] = [];
  for (let corner = 0; corner < 8; corner += 1) {
    const ox = corner & 1;
    const oy = (corner >> 1) & 1;
    const oz = (corner >> 2) & 1;
    values.push(z0 + 0.15 * (ox + oy + oz - 1.5) + 0.05 * z3);
  }
  return values;
}

function signByte(sdf: number[]) {
  return sdf.reduce((bits, value, index) => (value > 0 ? bits | (1 << index) : bits), 0);
}

function project(x: number, y: number, z: number) {
  return {
    px: 28 + x * 34 + z * 16,
    py: 132 - y * 28 - z * 14,
  };
}

function topologyOf(latents: readonly (readonly [number, number, number, number])[]) {
  return ACTIVE.map((voxel, index) => {
    const sdf = meshSdf(latents[index]);
    return `${voxel.x}${voxel.y}${voxel.z}:${signByte(sdf).toString(16)}`;
  }).join("|");
}

function meanGaussianScale(
  latents: readonly (readonly [number, number, number, number])[],
  radius: number,
) {
  let total = 0;
  let count = 0;
  for (const latent of latents) {
    for (let k = 0; k < 2; k += 1) {
      total += softplus(latent[1] + 0.2 * k) * radius;
      count += 1;
    }
  }
  return total / count;
}

function rfFingerprint(latents: readonly (readonly [number, number, number, number])[]) {
  return latents
    .map((latent) =>
      [latent[0], latent[1], latent[2], latent[3]]
        .map((value) => value.toFixed(3))
        .join(","),
    )
    .join(";");
}

function countFaces() {
  const occupied = new Set(ACTIVE.map((voxel) => `${voxel.x},${voxel.y},${voxel.z}`));
  let faces = 0;
  for (const voxel of ACTIVE) {
    if (occupied.has(`${voxel.x + 1},${voxel.y},${voxel.z}`)) faces += 1;
    if (occupied.has(`${voxel.x},${voxel.y + 1},${voxel.z}`)) faces += 1;
    if (occupied.has(`${voxel.x},${voxel.y},${voxel.z + 1}`)) faces += 1;
  }
  return faces;
}

function applyCorrupt(
  base: readonly (readonly [number, number, number, number])[],
  radius: number,
  corrupt: boolean,
) {
  if (!corrupt) return base.map((latent) => [...latent] as [number, number, number, number]);
  const delta = (radius - 1) * 0.4;
  return base.map(
    (latent) => [latent[0] + delta, latent[1], latent[2], latent[3]] as [number, number, number, number],
  );
}

export function Lesson50TrellisLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    radius: numberFrom(initialState, "radius", 1, 0.4, 2.4),
    decoder: stringFrom(initialState, "decoder", "gaussian") as DecoderId,
    prediction: stringFrom(initialState, "prediction", ""),
    corrupt: stringFrom(initialState, "corrupt", "off") === "on",
  };
  const [radius, setRadius] = useState(defaults.radius);
  const [decoder, setDecoder] = useState<DecoderId>(
    defaults.decoder === "mesh" || defaults.decoder === "nerf"
      ? defaults.decoder
      : "gaussian",
  );
  const [prediction, setPrediction] = useState(defaults.prediction);
  const [corrupt, setCorrupt] = useState(defaults.corrupt);
  const [revealed, setRevealed] = useState(false);
  const [seenMesh, setSeenMesh] = useState(false);
  const [seenGs, setSeenGs] = useState(false);
  const [seenNerf, setSeenNerf] = useState(false);
  const [radiusMoved, setRadiusMoved] = useState(Math.abs(defaults.radius - 1) >= 0.35);

  const baseLatents = useMemo(
    () => ACTIVE.map((voxel) => voxel.latent),
    [],
  );

  const calculation = useMemo(() => {
    const clean = applyCorrupt(baseLatents, radius, false);
    const live = applyCorrupt(baseLatents, radius, corrupt);
    const meshTopo = topologyOf(live);
    const meshTopoClean = topologyOf(clean);
    const gsScale = meanGaussianScale(live, radius);
    const gsScaleDefault = meanGaussianScale(clean, 1);
    const rfLive = rfFingerprint(live);
    const rfClean = rfFingerprint(clean);
    return {
      meshTopo,
      meshTopoClean,
      meshUnchanged: meshTopo === meshTopoClean,
      gsScale,
      gsScaleDefault,
      gsChanged: Math.abs(gsScale - gsScaleDefault) > 0.08,
      rfLive,
      rfClean,
      rfUnchanged: rfLive === rfClean,
      faces: countFaces(),
      voxels: ACTIVE.length,
      gsCount: ACTIVE.length * 2,
    };
  }, [baseLatents, corrupt, radius]);

  useEffect(() => {
    if (!revealed) return;
    if (decoder === "mesh") setSeenMesh(true);
    if (decoder === "gaussian") setSeenGs(true);
    if (decoder === "nerf") setSeenNerf(true);
  }, [decoder, revealed]);

  const foundIsolation =
    !corrupt &&
    calculation.meshUnchanged &&
    calculation.gsChanged &&
    calculation.rfUnchanged &&
    radiusMoved;

  const passed =
    revealed &&
    prediction === "radius_only_gs" &&
    seenMesh &&
    seenGs &&
    seenNerf &&
    foundIsolation;

  const completion = useMemo(
    () => ({
      lessonId: 50,
      radius,
      decoder,
      prediction,
      corrupt,
      meshTopo: calculation.meshTopo,
      gsScale: round(calculation.gsScale, 4),
      meshUnchanged: calculation.meshUnchanged,
      gsChanged: calculation.gsChanged,
      rfUnchanged: calculation.rfUnchanged,
      seenMesh,
      seenGs,
      seenNerf,
    }),
    [
      calculation.gsChanged,
      calculation.gsScale,
      calculation.meshTopo,
      calculation.meshUnchanged,
      calculation.rfUnchanged,
      corrupt,
      decoder,
      prediction,
      radius,
      seenGs,
      seenMesh,
      seenNerf,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function resetProgress() {
    setRevealed(false);
    setSeenMesh(false);
    setSeenGs(false);
    setSeenNerf(false);
    setRadiusMoved(false);
  }

  function reset() {
    setRadius(1);
    setDecoder("gaussian");
    setPrediction("");
    setCorrupt(false);
    resetProgress();
  }

  const decoderLabel =
    decoder === "mesh" ? "Mesh · FlexiCubes / SDF" : decoder === "gaussian" ? "3D Gaussian" : "辐射场 · CP";

  return (
    <LabFrame
      lesson="50"
      title="同一份 SLAT：切三个解码器"
      description="6 个活跃体素表示一只杯子。先预测改高斯半径会不会改 mesh 拓扑，再切换 mesh / 高斯 / 辐射场。教学模拟，不是 TRELLIS 输出。"
    >
      <div className={styles.workspace}>
        <form className={styles.controls} onSubmit={(event) => event.preventDefault()}>
          <h3>SLAT 控制台</h3>
          <p className={styles.note}>
            N=4，L=6，教学通道 C=4。论文默认是 64³ / 约 2 万体素 / C=8。数字只用于 shape 契约。
          </p>
          <fieldset>
            <legend>当前解码器</legend>
            {(
              [
                ["mesh", "Mesh"],
                ["gaussian", "3D Gaussian"],
                ["nerf", "辐射场"],
              ] as const
            ).map(([value, label]) => (
              <label key={value}>
                <input
                  type="radio"
                  name="decoder"
                  checked={decoder === value}
                  onChange={() => setDecoder(value)}
                />
                <span>{label}</span>
              </label>
            ))}
          </fieldset>
          <label>
            <span>
              高斯半径倍率 <output>{radius.toFixed(2)}</output>
            </span>
            <input
              type="range"
              min="0.40"
              max="2.40"
              step="0.05"
              value={radius}
              onChange={(event) => {
                const next = Number(event.target.value);
                setRadius(next);
                if (Math.abs(next - 1) >= 0.35) setRadiusMoved(true);
              }}
            />
          </label>
          <fieldset>
            <legend>错误实现（对照）</legend>
            <label>
              <input
                type="checkbox"
                checked={corrupt}
                onChange={(event) => {
                  setCorrupt(event.target.checked);
                  setRevealed(false);
                }}
              />
              <span>高斯解码器把半径写回 SLAT</span>
            </label>
          </fieldset>
          <p className={styles.note}>
            验收要求关闭写回。打开写回是为了看 mesh 拓扑和辐射场因子一起被污染。
          </p>
        </form>

        <div className={styles.stage}>
          <div className={styles.screens}>
            <figure className={styles.screenActive}>
              <figcaption>共享 SLAT · 活跃体素</figcaption>
              <div className={styles.slatStage}>
                <svg className={styles.stageSvg} viewBox="0 0 240 180" role="img" aria-label="稀疏占用">
                  {ACTIVE.map((voxel) => {
                    const { px, py } = project(voxel.x, voxel.y, voxel.z);
                    return (
                      <g key={`${voxel.x}-${voxel.y}-${voxel.z}`}>
                        <rect
                          x={px}
                          y={py}
                          width="22"
                          height="18"
                          fill="color-mix(in srgb, var(--xlab-accent, #176f48) 28%, #fff)"
                          stroke="var(--xlab-accent-ink, #1a4f35)"
                        />
                      </g>
                    );
                  })}
                  <text x="12" y="18">
                    L=6 / 64
                  </text>
                </svg>
              </div>
            </figure>
            <figure className={styles.screen}>
              <figcaption>解码输出 · {decoderLabel}</figcaption>
              <div className={styles.decodeStage}>
                {revealed ? (
                  <svg
                    className={styles.stageSvg}
                    viewBox="0 0 240 180"
                    role="img"
                    aria-label="当前解码器输出"
                  >
                    {decoder === "gaussian"
                      ? ACTIVE.map((voxel, index) => {
                          const latent = applyCorrupt(baseLatents, radius, corrupt)[index];
                          const { px, py } = project(voxel.x, voxel.y, voxel.z);
                          const scale = softplus(latent[1]) * radius * 7;
                          return (
                            <circle
                              key={`${voxel.x}-${voxel.y}-${voxel.z}`}
                              cx={px + 11}
                              cy={py + 9}
                              r={Math.max(3, Math.min(16, scale))}
                              fill="color-mix(in srgb, var(--xlab-accent, #176f48) 35%, transparent)"
                              stroke="var(--xlab-accent, #176f48)"
                            />
                          );
                        })
                      : null}
                    {decoder === "mesh"
                      ? ACTIVE.flatMap((voxel, index) => {
                          const { px, py } = project(voxel.x, voxel.y, voxel.z);
                          const neighbors = ACTIVE.filter(
                            (other) =>
                              other.x > voxel.x &&
                              other.y === voxel.y &&
                              other.z === voxel.z,
                          );
                          return [
                            <rect
                              key={`m-${voxel.x}-${voxel.y}-${voxel.z}`}
                              x={px}
                              y={py}
                              width="20"
                              height="16"
                              fill="none"
                              stroke={
                                calculation.meshUnchanged
                                  ? "var(--xlab-accent-ink, #1a4f35)"
                                  : "#8a2b2b"
                              }
                              strokeWidth="1.6"
                            />,
                            ...neighbors.map((other) => {
                              const q = project(other.x, other.y, other.z);
                              return (
                                <line
                                  key={`e-${voxel.x}-${other.x}-${voxel.z}`}
                                  x1={px + 20}
                                  y1={py + 8}
                                  x2={q.px}
                                  y2={q.py + 8}
                                  stroke="var(--xlab-rule, #8ea392)"
                                />
                              );
                            }),
                          ];
                        })
                      : null}
                    {decoder === "nerf"
                      ? ACTIVE.map((voxel, index) => {
                          const latent = applyCorrupt(baseLatents, radius, corrupt)[index];
                          const { px, py } = project(voxel.x, voxel.y, voxel.z);
                          const density = Math.min(0.85, Math.max(0.15, latent[2]));
                          return (
                            <rect
                              key={`rf-${voxel.x}-${voxel.y}-${voxel.z}`}
                              x={px}
                              y={py}
                              width="20"
                              height="16"
                              fill={`rgba(23, 111, 72, ${density})`}
                              stroke="var(--xlab-accent, #176f48)"
                            />
                          );
                        })
                      : null}
                    <text x="12" y="170">
                      {decoder === "mesh"
                        ? `拓扑 ${calculation.meshTopo.slice(0, 18)}`
                        : decoder === "gaussian"
                          ? `均值半径 ${calculation.gsScale.toFixed(3)}`
                          : "CP 因子跟 z 走"}
                    </text>
                  </svg>
                ) : (
                  <div className={styles.hiddenMark}>
                    三种解码的拓扑哈希、高斯半径和辐射场因子在揭晓后显示
                  </div>
                )}
              </div>
            </figure>
          </div>
          <dl className={styles.metrics}>
            <div>
              <dt>Mesh 拓扑</dt>
              <dd
                className={
                  revealed
                    ? calculation.meshUnchanged
                      ? styles.pass
                      : styles.fail
                    : undefined
                }
              >
                {revealed ? (calculation.meshUnchanged ? "未变" : "已变") : "—"}
              </dd>
            </div>
            <div>
              <dt>高斯半径</dt>
              <dd
                className={
                  revealed ? (calculation.gsChanged ? styles.pass : styles.fail) : undefined
                }
              >
                {revealed ? calculation.gsScale.toFixed(3) : "—"}
              </dd>
            </div>
            <div>
              <dt>辐射场因子</dt>
              <dd
                className={
                  revealed
                    ? calculation.rfUnchanged
                      ? styles.pass
                      : styles.fail
                    : undefined
                }
              >
                {revealed ? (calculation.rfUnchanged ? "未变" : "已变") : "—"}
              </dd>
            </div>
            <div>
              <dt>写回 SLAT</dt>
              <dd className={corrupt ? styles.fail : styles.pass}>
                {corrupt ? "开" : "关"}
              </dd>
            </div>
          </dl>
          <p className={styles.formula}>
            D_GS: x=p+tanh(o), s=softplus(z_1)·r。D_M: 8 个 SDF 符号组成拓扑。D_RF: CP(z)。
            体素 {calculation.voxels}，高斯 {calculation.gsCount}，邻接面 {calculation.faces}。
          </p>
        </div>
      </div>

      <div className={styles.predict}>
        <fieldset>
          <legend>先预测：同一份 SLAT 上把高斯半径从 1 拖到约 1.8，哪句话成立？</legend>
          {(
            [
              ["radius_changes_mesh", "高斯球变大，mesh 面会裂开或合并，拓扑哈希改变"],
              ["radius_only_gs", "只有高斯球变大；mesh 拓扑哈希不变，辐射场因子也不变"],
              ["all_three_change", "三种解码共享 SLAT，半径一改三路输出一起变"],
              ["nerf_owns_topology", "辐射场体积会改 mesh 的边，高斯半径只是外观"],
            ] as const
          ).map(([value, label]) => (
            <label key={value}>
              <input
                type="radio"
                name="slat-prediction"
                value={value}
                checked={prediction === value}
                onChange={() => {
                  setPrediction(value);
                  setRevealed(false);
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
            onClick={() => setRevealed(true)}
          >
            揭晓三路输出
          </button>
        </div>
      </div>
      {revealed && prediction !== "radius_only_gs" ? (
        <p className={styles.feedback}>
          半径是 D_GS 的局部尺度。合法解码器只读 z_i。把写回关掉，再切到 mesh 看拓扑哈希是否仍等于基线。
        </p>
      ) : null}
      {revealed && prediction === "radius_only_gs" && corrupt ? (
        <p className={styles.feedback}>
          预测选对了，但写回开关把半径灌进了 z_0，mesh 与辐射场已被污染。关掉写回才能通过验收。
        </p>
      ) : null}
      {revealed && prediction === "radius_only_gs" && !corrupt && !radiusMoved ? (
        <p className={styles.feedback}>
          把半径滑到 0.65 以下或 1.35 以上，才能证明高斯尺度变了、拓扑没变。
        </p>
      ) : null}
      {revealed && prediction === "radius_only_gs" && !corrupt && radiusMoved && (!seenMesh || !seenGs || !seenNerf) ? (
        <p className={styles.feedback}>
          三个解码器都要切一遍：mesh 看拓扑，高斯看半径，辐射场看因子。
        </p>
      ) : null}
      <Gate passed={passed}>
        先选对“只有高斯球变大”，关闭写回，把半径拖离 1.0，并切过三种解码器。mesh
        拓扑与辐射场因子保持基线，高斯尺度改变。数字来自教学模拟，不是模型输出。
      </Gate>
    </LabFrame>
  );
}
