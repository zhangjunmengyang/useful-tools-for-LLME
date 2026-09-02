"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import { numberFrom, round, stringFrom, useCompletionGate } from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson35DepthGraspLab.module.css";

type DepthSource = "mean" | "pixel";

const FX = 240;
const FY = 240;
const CX = 160;
const CY = 120;
const IMG_W = 320;
const IMG_H = 240;
const U_OBJ = 184;
const V_OBJ = 96;
const RGB_PIXEL_TAU = 6;
const TAU_XY = 0.03;

function backproject(u: number, v: number, z: number) {
  return {
    x: ((u - CX) / FX) * z,
    y: ((v - CY) / FY) * z,
    z,
  };
}

function project(x: number, y: number, z: number) {
  return {
    u: (FX * x) / z + CX,
    v: (FY * y) / z + CY,
  };
}

function hypot2(dx: number, dy: number) {
  return Math.hypot(dx, dy);
}

export function Lesson35DepthGraspLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    zObj: numberFrom(initialState, "zObj", 0.52, 0.36, 0.92),
    zMean: numberFrom(initialState, "zMean", 0.8, 0.5, 0.95),
    tauZ: numberFrom(initialState, "tauZ", 0.04, 0.02, 0.1),
    depthSource: stringFrom(initialState, "depthSource", "mean") as DepthSource,
    prediction: stringFrom(initialState, "prediction", ""),
  };
  const [zObj, setZObj] = useState(defaults.zObj);
  const [zMean, setZMean] = useState(defaults.zMean);
  const [tauZ, setTauZ] = useState(defaults.tauZ);
  const [depthSource, setDepthSource] = useState<DepthSource>(
    defaults.depthSource === "pixel" ? "pixel" : "mean",
  );
  const [prediction, setPrediction] = useState(defaults.prediction);
  const [revealed, setRevealed] = useState(false);

  const calculation = useMemo(() => {
    const obj = backproject(U_OBJ, V_OBJ, zObj);
    const zGrip = depthSource === "pixel" ? zObj : zMean;
    const grip = backproject(U_OBJ, V_OBJ, zGrip);
    const objPix = project(obj.x, obj.y, obj.z);
    const gripPix = project(grip.x, grip.y, grip.z);
    const rgbError = hypot2(gripPix.u - objPix.u, gripPix.v - objPix.v);
    const rgbSuccess = rgbError <= RGB_PIXEL_TAU;
    const dxy = hypot2(grip.x - obj.x, grip.y - obj.y);
    const dz = Math.abs(grip.z - obj.z);
    const contact = dz <= tauZ && dxy <= TAU_XY;
    const gap = Math.hypot(grip.x - obj.x, grip.y - obj.y, grip.z - obj.z);
    return {
      obj,
      grip,
      objPix,
      gripPix,
      rgbError,
      rgbSuccess,
      dxy,
      dz,
      contact,
      gap,
      zGrip,
    };
  }, [depthSource, tauZ, zMean, zObj]);

  const foundRgbTrueContactFalse =
    calculation.rgbSuccess &&
    !calculation.contact &&
    depthSource === "mean" &&
    calculation.dz > tauZ + 0.02;

  const passed =
    revealed &&
    prediction === "rgb_true_contact_false" &&
    foundRgbTrueContactFalse;

  const completion = useMemo(
    () => ({
      lessonId: 35,
      zObj,
      zMean,
      tauZ,
      depthSource,
      prediction,
      rgbSuccess: calculation.rgbSuccess,
      contact: calculation.contact,
      rgbError: round(calculation.rgbError, 4),
      dz: round(calculation.dz, 4),
      gap: round(calculation.gap, 4),
    }),
    [
      calculation.contact,
      calculation.dz,
      calculation.gap,
      calculation.rgbError,
      calculation.rgbSuccess,
      depthSource,
      prediction,
      tauZ,
      zMean,
      zObj,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function reset() {
    setZObj(0.52);
    setZMean(0.8);
    setTauZ(0.04);
    setDepthSource("mean");
    setPrediction("");
    setRevealed(false);
  }

  const cupLeft = (U_OBJ / IMG_W) * 100;
  const cupTop = (V_OBJ / IMG_H) * 100;
  const zMin = 0.28;
  const zMax = 1.02;
  const sideX = (z: number) => 18 + ((z - zMin) / (zMax - zMin)) * 210;
  const sideY = (y: number, z: number) => 86 - (y / 0.12) * 28 - ((z - 0.6) * 4);

  return (
    <LabFrame
      lesson="35"
      title="同一抓取：RGB 命中还是三维接触"
      description="同一只杯子、同一条射线。先预测无深度时失败发生在哪一步，再切换均值深度与像素深度，核对 RGB 成功能否与接触带脱钩。教学模拟，不是模型输出。"
    >
      <div className={styles.workspace}>
        <form className={styles.controls} onSubmit={(event) => event.preventDefault()}>
          <h3>深度控制台</h3>
          <label>
            <span>
              杯子深度 Z* <output>{zObj.toFixed(2)} m</output>
            </span>
            <input
              type="range"
              min="0.36"
              max="0.92"
              step="0.01"
              value={zObj}
              onChange={(event) => {
                setZObj(Number(event.target.value));
                setRevealed(false);
              }}
            />
          </label>
          <label>
            <span>
              场景均值 Z̄ <output>{zMean.toFixed(2)} m</output>
            </span>
            <input
              type="range"
              min="0.50"
              max="0.95"
              step="0.01"
              value={zMean}
              onChange={(event) => {
                setZMean(Number(event.target.value));
                setRevealed(false);
              }}
            />
          </label>
          <label>
            <span>
              接触带 τ_z <output>{tauZ.toFixed(2)} m</output>
            </span>
            <input
              type="range"
              min="0.02"
              max="0.10"
              step="0.01"
              value={tauZ}
              onChange={(event) => {
                setTauZ(Number(event.target.value));
                setRevealed(false);
              }}
            />
          </label>
          <fieldset>
            <legend>深度来源</legend>
            <label>
              <input
                type="radio"
                name="depth-source"
                checked={depthSource === "mean"}
                onChange={() => {
                  setDepthSource("mean");
                  setRevealed(false);
                }}
              />
              <span>无深度：场景均值</span>
            </label>
            <label>
              <input
                type="radio"
                name="depth-source"
                checked={depthSource === "pixel"}
                onChange={() => {
                  setDepthSource("pixel");
                  setRevealed(false);
                }}
              />
              <span>有深度：像素 z</span>
            </label>
          </fieldset>
          <p className={styles.note}>
            夹爪始终瞄准杯子的 (u, v)。均值深度只改射线上的尺度，不改图像坐标。
          </p>
        </form>

        <div className={styles.stage}>
          <div className={styles.screens}>
            <figure className={styles.screenActive}>
              <figcaption>相机成像 · (u, v)</figcaption>
              <div className={styles.cameraStage}>
                <div
                  className={styles.cup}
                  style={{ left: `${cupLeft}%`, top: `${cupTop}%` }}
                >
                  杯
                </div>
                {revealed ? (
                  <i
                    className={styles.gripper}
                    style={{ left: `${cupLeft}%`, top: `${cupTop}%` }}
                    title="夹爪投影"
                  />
                ) : null}
              </div>
            </figure>
            <figure className={styles.screen}>
              <figcaption>侧视 · 沿射线的 Z</figcaption>
              <div className={styles.sideStage}>
                {revealed ? (
                  <svg
                    className={styles.sideSvg}
                    viewBox="0 0 240 180"
                    role="img"
                    aria-label="相机光心、射线、杯子与夹爪"
                  >
                    <line
                      x1="16"
                      y1="92"
                      x2="228"
                      y2="92"
                      stroke="var(--xlab-rule, #c5cec6)"
                      strokeDasharray="3 3"
                    />
                    <polygon points="14,86 26,92 14,98" fill="var(--xlab-ink, #213a2c)" />
                    <text x="10" y="78">
                      相机
                    </text>
                    <line
                      x1="20"
                      y1="92"
                      x2={sideX(1.0)}
                      y2={sideY(((V_OBJ - CY) / FY) * 1.0, 1.0)}
                      stroke="var(--xlab-rule, #8ea392)"
                    />
                    <rect
                      x={sideX(calculation.obj.z) - 8}
                      y={sideY(calculation.obj.y, calculation.obj.z) - 10 - (tauZ / 0.1) * 8}
                      width="16"
                      height={20 + (tauZ / 0.1) * 16}
                      fill="color-mix(in srgb, var(--xlab-accent, #176f48) 16%, transparent)"
                      stroke="var(--xlab-accent, #176f48)"
                      strokeDasharray="2 2"
                    />
                    <circle
                      cx={sideX(calculation.obj.z)}
                      cy={sideY(calculation.obj.y, calculation.obj.z)}
                      r="5"
                      fill="var(--xlab-accent, #176f48)"
                    />
                    <circle
                      cx={sideX(calculation.grip.z)}
                      cy={sideY(calculation.grip.y, calculation.grip.z)}
                      r="5"
                      fill="none"
                      stroke="var(--xlab-accent-ink, #1a4f35)"
                      strokeWidth="2"
                    />
                    <text
                      x={sideX(calculation.obj.z) - 10}
                      y={sideY(calculation.obj.y, calculation.obj.z) + 18}
                    >
                      杯
                    </text>
                    <text
                      x={sideX(calculation.grip.z) - 10}
                      y={sideY(calculation.grip.y, calculation.grip.z) - 12}
                    >
                      爪
                    </text>
                    <text x="168" y="168">
                      Z
                    </text>
                  </svg>
                ) : (
                  <div className={styles.hiddenMark}>侧视与接触数字在揭晓后显示</div>
                )}
              </div>
            </figure>
          </div>
          <dl className={styles.metrics}>
            <div>
              <dt>RGB 命中</dt>
              <dd className={revealed && calculation.rgbSuccess ? styles.pass : undefined}>
                {revealed ? (calculation.rgbSuccess ? "真" : "假") : "—"}
              </dd>
            </div>
            <div>
              <dt>三维接触</dt>
              <dd
                className={
                  revealed
                    ? calculation.contact
                      ? styles.pass
                      : styles.fail
                    : undefined
                }
              >
                {revealed ? (calculation.contact ? "真" : "假") : "—"}
              </dd>
            </div>
            <div>
              <dt>|Z_grip − Z*|</dt>
              <dd>{revealed ? `${calculation.dz.toFixed(3)} m` : "—"}</dd>
            </div>
            <div>
              <dt>射线间距</dt>
              <dd>{revealed ? `${calculation.gap.toFixed(3)} m` : "—"}</dd>
            </div>
          </dl>
          <p className={styles.formula}>
            X=(u−c_x)Z/f_x, Y=(v−c_y)Z/f_y。当前 Z_grip={revealed ? calculation.zGrip.toFixed(2) : "—"} m，
            (u,v)=({U_OBJ},{V_OBJ})。
          </p>
        </div>
      </div>

      <div className={styles.predict}>
        <fieldset>
          <legend>先预测：只用 RGB、深度改成场景均值时，哪句话成立？</legend>
          {(
            [
              ["rgb_implies_contact", "图像上重合，三维接触一定成立"],
              [
                "rgb_true_contact_false",
                "夹爪沿同一射线停在错误深度：RGB 成功，三维接触失败",
              ],
              ["rgb_fails_only", "无深度只会让图像命中失败，接触不受影响"],
              ["tau_makes_equal", "加宽接触带之后，均值深度等于杯子深度"],
            ] as const
          ).map(([value, label]) => (
            <label key={value}>
              <input
                type="radio"
                name="depth-prediction"
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
            揭晓接触
          </button>
        </div>
      </div>
      {revealed && prediction !== "rgb_true_contact_false" ? (
        <p className={styles.feedback}>
          同一条射线上的点都投到 (u, v)。RGB 命中只说明共线。把深度切到像素 z，夹爪才会进接触带。
        </p>
      ) : null}
      {revealed && prediction === "rgb_true_contact_false" && !foundRgbTrueContactFalse ? (
        <p className={styles.feedback}>
          预测选对了，但当前参数还没造出“RGB 真、接触假”。把深度来源保持为场景均值，并让 |Z* − Z̄| 明显大于 τ_z。
        </p>
      ) : null}
      <Gate passed={passed}>
        先选对“RGB 成功、三维接触失败”，再用均值深度找到一例图像命中为真、接触带为假的抓取。加上像素深度后应进入接触带。数字来自教学模拟，不是模型输出。
      </Gate>
    </LabFrame>
  );
}
