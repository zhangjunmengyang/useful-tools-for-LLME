"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import {
  mean,
  numberFrom,
  round,
  stringFrom,
  useCompletionGate,
} from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson33WorldModelLab.module.css";

const STEPS = 8;
const GRID = 8;
const CONTACT_CUP = 5;
const SEPARATED_CUP = 7;

type Scene = "contact" | "separated";

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function moverCol(step: number) {
  return Math.min(step, 5);
}

function texture(step: number, row: number, col: number, intensity: number) {
  const raw = ((step * 17 + row * 13 + col * 7) % 10) / 9;
  return 0.5 + (raw - 0.5) * intensity;
}

function renderTrue(step: number, cup: number, intensity: number) {
  return Array.from({ length: GRID }, (_, row) =>
    Array.from({ length: GRID }, (_, col) => {
      const inMover = row >= 3 && row <= 4 && col === moverCol(step);
      const inCup = row >= 3 && row <= 4 && col === cup;
      if (inMover && inCup) return 0.5;
      if (inMover) return 0.92;
      if (inCup) return 0.12;
      return texture(step, row, col, intensity);
    }),
  );
}

function renderPixelPred(
  step: number,
  cup: number,
  lastVisible: number,
  intensity: number,
) {
  const depth = Math.max(1, step - lastVisible);
  const smear = Math.max(1, Math.round(depth * (0.55 + 0.7 * intensity)));
  const predictedMover = clamp(
    moverCol(lastVisible) + (step - lastVisible),
    0,
    GRID - 1,
  );
  return Array.from({ length: GRID }, (_, row) =>
    Array.from({ length: GRID }, (_, col) => {
      let value = 0.5;
      if (row >= 3 && row <= 4) {
        if (Math.abs(col - predictedMover) <= smear) value = 0.72;
        if (Math.abs(col - cup) <= smear) {
          value = 0.5 * value + 0.5 * (Math.abs(col - cup) === 0 ? 0.18 : 0.32);
        }
      } else {
        value = 0.5 + 0.04 * intensity;
      }
      return value;
    }),
  );
}

function latent(step: number, cup: number, lastVisible: number | null) {
  const mover =
    lastVisible === null
      ? moverCol(step)
      : clamp(Math.min(moverCol(lastVisible) + (step - lastVisible), 5), 0, GRID - 1);
  const overlap = Math.max(0, 1 - Math.abs(mover - cup));
  const velocity = mover >= 5 ? 0 : 1;
  return {
    mover,
    cup,
    overlap,
    velocity,
    vector: [mover / GRID, cup / GRID, overlap, velocity],
  };
}

function occupancyMass(grid: number[][], col: number) {
  return (grid[3][col] + grid[4][col]) / 2;
}

function pixelOverlap(grid: number[][], mover: number, cup: number) {
  return Math.min(occupancyMass(grid, mover), occupancyMass(grid, cup));
}

function mse(left: number[][], right: number[][]) {
  const values: number[] = [];
  for (let row = 0; row < GRID; row += 1) {
    for (let col = 0; col < GRID; col += 1) {
      const delta = left[row][col] - right[row][col];
      values.push(delta * delta);
    }
  }
  return mean(values);
}

function shade(value: number) {
  const clipped = clamp(value, 0, 1);
  const channel = Math.round(24 + clipped * 214);
  return `rgb(${channel}, ${Math.round(channel * 0.96)}, ${Math.round(channel * 0.9)})`;
}

export function Lesson33WorldModelLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    occludeFrom: numberFrom(initialState, "occludeFrom", 5, 1, 7),
    texture: numberFrom(initialState, "texture", 0.85, 0, 1),
    scene: stringFrom(initialState, "scene", "contact") as Scene,
    prediction: stringFrom(initialState, "prediction", ""),
  };
  const [occludeFrom, setOccludeFrom] = useState(defaults.occludeFrom);
  const [textureIntensity, setTextureIntensity] = useState(defaults.texture);
  const [scene, setScene] = useState<Scene>(
    defaults.scene === "separated" ? "separated" : "contact",
  );
  const [prediction, setPrediction] = useState(defaults.prediction);
  const [revealed, setRevealed] = useState(false);

  const calculation = useMemo(() => {
    const lastVisible = occludeFrom - 1;
    const cup = scene === "contact" ? CONTACT_CUP : SEPARATED_CUP;
    const frames = Array.from({ length: STEPS }, (_, step) => {
      const hidden = step >= occludeFrom;
      const truth = renderTrue(step, cup, textureIntensity);
      const pixel = hidden
        ? renderPixelPred(step, cup, lastVisible, textureIntensity)
        : truth;
      const shown = hidden && !revealed ? null : pixel;
      return {
        step,
        hidden,
        truth,
        pixel,
        shown,
        latentTrue: latent(step, cup, null),
        latentPred: hidden
          ? latent(step, cup, lastVisible)
          : latent(step, cup, null),
      };
    });

    const occluded = frames.filter((frame) => frame.hidden);
    const pixelL2 = mean(
      occluded.map((frame) => mse(frame.pixel, frame.truth)),
    );
    const latentL2 = mean(
      occluded.map((frame) => {
        const pred = frame.latentPred.vector;
        const truth = frame.latentTrue.vector;
        return pred.reduce(
          (sum, value, index) => sum + (value - truth[index]) ** 2,
          0,
        );
      }),
    );

    const contactPixel = occluded.map((frame) => {
      const pred = renderPixelPred(
        frame.step,
        CONTACT_CUP,
        lastVisible,
        textureIntensity,
      );
      const mover = clamp(
        moverCol(lastVisible) + (frame.step - lastVisible),
        0,
        GRID - 1,
      );
      return pixelOverlap(pred, mover, CONTACT_CUP);
    });
    const separatedPixel = occluded.map((frame) => {
      const pred = renderPixelPred(
        frame.step,
        SEPARATED_CUP,
        lastVisible,
        textureIntensity,
      );
      const mover = clamp(
        moverCol(lastVisible) + (frame.step - lastVisible),
        0,
        GRID - 1,
      );
      return pixelOverlap(pred, mover, SEPARATED_CUP);
    });
    const contactLatent = occluded.map(
      (frame) => latent(frame.step, CONTACT_CUP, lastVisible).overlap,
    );
    const separatedLatent = occluded.map(
      (frame) => latent(frame.step, SEPARATED_CUP, lastVisible).overlap,
    );

    const pixelMargin = Math.abs(mean(contactPixel) - mean(separatedPixel));
    const latentMargin = mean(contactLatent) - mean(separatedLatent);
    const last = frames[STEPS - 1];

    return {
      frames,
      pixelL2,
      latentL2,
      pixelMargin,
      latentMargin,
      contactPixel: mean(contactPixel),
      separatedPixel: mean(separatedPixel),
      contactLatent: mean(contactLatent),
      separatedLatent: mean(separatedLatent),
      last,
    };
  }, [occludeFrom, revealed, scene, textureIntensity]);

  const passed =
    revealed &&
    prediction === "pixel_blurs_latent_separates" &&
    occludeFrom <= 5 &&
    textureIntensity >= 0.55 &&
    calculation.pixelMargin < 0.12 &&
    calculation.latentMargin > 0.5;

  const completion = useMemo(
    () => ({
      lessonId: 33,
      occludeFrom,
      texture: textureIntensity,
      scene,
      prediction,
      pixelL2: round(calculation.pixelL2, 4),
      latentL2: round(calculation.latentL2, 4),
      pixelMargin: round(calculation.pixelMargin, 4),
      latentMargin: round(calculation.latentMargin, 4),
    }),
    [
      calculation.latentL2,
      calculation.latentMargin,
      calculation.pixelL2,
      calculation.pixelMargin,
      occludeFrom,
      prediction,
      scene,
      textureIntensity,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function reset() {
    setOccludeFrom(5);
    setTextureIntensity(0.85);
    setScene("contact");
    setPrediction("");
    setRevealed(false);
  }

  const lastLatent = calculation.last.latentPred;

  return (
    <LabFrame
      lesson="33"
      title="遮挡未来：像素重建还是表征预测"
      description="同一段 8 步接触序列。先预测遮挡后哪条路还会留下接触，再揭晓像素重建和表征重叠。"
    >
      <div className={styles.workspace}>
        <form
          className={styles.controls}
          onSubmit={(event) => event.preventDefault()}
        >
          <h3>世界模型控制台</h3>
          <label>
            <span>
              从第几帧开始遮挡未来 <output>{occludeFrom}</output>
            </span>
            <input
              type="range"
              min="1"
              max="7"
              step="1"
              value={occludeFrom}
              onChange={(event) => {
                setOccludeFrom(Number(event.target.value));
                setRevealed(false);
              }}
            />
          </label>
          <label>
            <span>
              不可预测纹理强度 <output>{textureIntensity.toFixed(2)}</output>
            </span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={textureIntensity}
              onChange={(event) => {
                setTextureIntensity(Number(event.target.value));
                setRevealed(false);
              }}
            />
          </label>
          <fieldset>
            <legend>当前显示的序列</legend>
            <label>
              <input
                type="radio"
                name="scene"
                checked={scene === "contact"}
                onChange={() => {
                  setScene("contact");
                  setRevealed(false);
                }}
              />
              <span>接触（杯子在列 5）</span>
            </label>
            <label>
              <input
                type="radio"
                name="scene"
                checked={scene === "separated"}
                onChange={() => {
                  setScene("separated");
                  setRevealed(false);
                }}
              />
              <span>分离（杯子在列 7）</span>
            </label>
          </fieldset>
          <p className={styles.note}>
            接触探针始终同时算接触序列和分离序列。显示开关只改你看哪一条，不改验收用的两条路。
          </p>
        </form>

        <div className={styles.stage}>
          <div className={styles.timeline} aria-label="八步序列">
            {calculation.frames.map((frame) => (
              <div key={frame.step} className={styles.frame}>
                <b>t={frame.step}</b>
                <div
                  className={styles.frameGrid}
                  data-occluded={frame.hidden ? "true" : "false"}
                >
                  {(frame.shown ?? frame.truth).flatMap((row, rowIndex) =>
                    row.map((value, colIndex) => (
                      <i
                        key={`${frame.step}-${rowIndex}-${colIndex}`}
                        style={{
                          background:
                            frame.hidden && !revealed
                              ? "repeating-linear-gradient(135deg,#d7ddd8 0 4px,#eef2ee 4px 8px)"
                              : shade(value),
                        }}
                      />
                    )),
                  )}
                </div>
              </div>
            ))}
          </div>
          <div className={styles.panels}>
            <figure className={styles.panelActive}>
              <figcaption>像素路 · 遮挡后重建</figcaption>
              <div className={styles.recon}>
                {calculation.last.pixel.flatMap((row, rowIndex) =>
                  row.map((value, colIndex) => (
                    <i
                      key={`pix-${rowIndex}-${colIndex}`}
                      style={{
                        background: revealed
                          ? shade(value)
                          : "repeating-linear-gradient(135deg,#d7ddd8 0 6px,#eef2ee 6px 12px)",
                      }}
                    />
                  )),
                )}
              </div>
            </figure>
            <figure className={styles.panelActive}>
              <figcaption>表征路 · 位置与重叠</figcaption>
              <div className={styles.latentBars}>
                <div>
                  <span>
                    运动物体 x <output>{revealed ? lastLatent.mover : "—"}</output>
                  </span>
                  <div className={styles.bar}>
                    <i
                      style={{
                        width: revealed ? `${(lastLatent.mover / 7) * 100}%` : "0%",
                      }}
                    />
                  </div>
                </div>
                <div>
                  <span>
                    杯子 x <output>{revealed ? lastLatent.cup : "—"}</output>
                  </span>
                  <div className={styles.bar}>
                    <i
                      style={{
                        width: revealed ? `${(lastLatent.cup / 7) * 100}%` : "0%",
                      }}
                    />
                  </div>
                </div>
                <div>
                  <span>
                    重叠 overlap{" "}
                    <output>
                      {revealed ? lastLatent.overlap.toFixed(2) : "—"}
                    </output>
                  </span>
                  <div className={styles.bar}>
                    <i
                      style={{
                        width: revealed ? `${lastLatent.overlap * 100}%` : "0%",
                      }}
                    />
                  </div>
                </div>
              </div>
            </figure>
          </div>
          <dl className={styles.metrics}>
            <div>
              <dt>像素 L2</dt>
              <dd>{revealed ? calculation.pixelL2.toFixed(3) : "—"}</dd>
            </div>
            <div>
              <dt>表征 L2</dt>
              <dd>{revealed ? calculation.latentL2.toFixed(3) : "—"}</dd>
            </div>
            <div>
              <dt>像素接触差</dt>
              <dd>{revealed ? calculation.pixelMargin.toFixed(3) : "—"}</dd>
            </div>
            <div>
              <dt>表征接触差</dt>
              <dd>{revealed ? calculation.latentMargin.toFixed(3) : "—"}</dd>
            </div>
          </dl>
          <p className={styles.formula}>
            L_pix = mean ||xhat - x||^2 ； z = (x_m/8, x_c/8, overlap, v)；overlap =
            relu(1 - |x_m - x_c|)。数字来自教学夹具。
          </p>
        </div>
      </div>

      <div className={styles.predict}>
        <fieldset>
          <legend>先预测：遮挡接触发生的未来帧之后，哪句话成立？</legend>
          {[
            [
              "pixel_blurs_latent_separates",
              "像素路糊掉接触，表征路仍分得出接触/分离",
            ],
            [
              "both_keep_contact",
              "两条路都能分清接触和分离",
            ],
            [
              "latent_collapses",
              "表征路会塌成同一个向量，像素路更清楚",
            ],
            [
              "occlusion_irrelevant",
              "遮挡几帧几乎不改变两条损失",
            ],
          ].map(([value, label]) => (
            <label key={value}>
              <input
                type="radio"
                name="world-model-prediction"
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
            揭晓损失
          </button>
        </div>
      </div>
      {revealed && prediction !== "pixel_blurs_latent_separates" && (
        <p className={styles.feedback}>
          纹理在未来帧不可抄。像素 L2 会把格子噪声算进去；接触探针看的是糊掉的重叠。表征只保留位置和 overlap。
        </p>
      )}
      {revealed && occludeFrom > 5 && (
        <p className={styles.feedback}>
          接触发生在 t=5 起。把遮挡起点拖到 5 或更早，像素路才会看不到接触。
        </p>
      )}
      <Gate passed={passed}>
        先选对“像素路糊掉接触、表征路仍能分开”，再把未来遮到接触帧、纹理强度至少
        0.55。像素接触差必须小于 0.12，表征接触差必须大于 0.5。数字来自教学模拟，不是模型输出。
      </Gate>
    </LabFrame>
  );
}
