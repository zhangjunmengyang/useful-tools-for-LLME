"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import { numberFrom, round, stringFrom, useCompletionGate } from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson49VideoGenLab.module.css";

const HEIGHT = 8;
const WIDTH = 8;
const T_OBS = 5;
const CUP_COL = 5;
const CUP_ROWS: readonly number[] = [3, 4];
const CUP_VALUE = 0.08;
const BG_FILL = 0.5;

type PredictionId =
  | "understand_ok_gen_vanish"
  | "both_keep_cup"
  | "understand_wrong"
  | "l2_protects_cup"
  | "";

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function texture(step: number, row: number, col: number) {
  return ((step * 17 + row * 13 + col * 7) % 10) / 9;
}

function render(step: number, cupPresent: boolean) {
  return Array.from({ length: HEIGHT }, (_, row) =>
    Array.from({ length: WIDTH }, (_, col) => {
      if (cupPresent && CUP_ROWS.includes(row) && col === CUP_COL) {
        return CUP_VALUE;
      }
      return texture(step, row, col);
    }),
  );
}

function cupOccupancy(grid: number[][]) {
  const cells = CUP_ROWS.map((row) => grid[row][CUP_COL]);
  const meanValue = cells.reduce((sum, value) => sum + value, 0) / cells.length;
  return clamp((BG_FILL - meanValue) / (BG_FILL - CUP_VALUE), 0, 1);
}

function mixFrame(last: number[][], forget: number) {
  const mix = clamp(forget, 0, 1);
  return last.map((row) =>
    row.map((value) => (1 - mix) * value + mix * BG_FILL),
  );
}

function frameL2(left: number[][], right: number[][]) {
  let total = 0;
  let count = 0;
  for (let row = 0; row < HEIGHT; row += 1) {
    for (let col = 0; col < WIDTH; col += 1) {
      const delta = left[row][col] - right[row][col];
      total += delta * delta;
      count += 1;
    }
  }
  return total / count;
}

function shade(value: number) {
  const clipped = clamp(value, 0, 1);
  const channel = Math.round(24 + clipped * 214);
  return `rgb(${channel}, ${Math.round(channel * 0.96)}, ${Math.round(channel * 0.9)})`;
}

const PREDICTIONS: { value: Exclude<PredictionId, "">; label: string }[] = [
  {
    value: "understand_ok_gen_vanish",
    label: "理解答“还在”，生成下一帧杯子消失",
  },
  {
    value: "both_keep_cup",
    label: "两边都会保住杯子",
  },
  {
    value: "understand_wrong",
    label: "理解会先答错，生成帧反而更稳",
  },
  {
    value: "l2_protects_cup",
    label: "帧差 L2 更低就等于杯子还在",
  },
];

export function Lesson49VideoGenLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    forget: numberFrom(initialState, "forget", 0.3, 0, 1),
    historyHasCup: stringFrom(initialState, "historyHasCup", "yes") !== "no",
    prediction: stringFrom(initialState, "prediction", "") as PredictionId,
  };
  const [forget, setForget] = useState(defaults.forget);
  const [historyHasCup, setHistoryHasCup] = useState(defaults.historyHasCup);
  const [prediction, setPrediction] = useState<PredictionId>(defaults.prediction);
  const [revealed, setRevealed] = useState(false);

  const calculation = useMemo(() => {
    const observed = Array.from({ length: T_OBS }, (_, step) =>
      render(step, historyHasCup),
    );
    const truthNext = render(T_OBS, historyHasCup);
    const generated = mixFrame(observed[T_OBS - 1], forget);
    const copied = mixFrame(observed[T_OBS - 1], 0);
    const histOccupancy = cupOccupancy(observed[T_OBS - 1]);
    const genOccupancy = cupOccupancy(generated);
    const copyOccupancy = cupOccupancy(copied);
    const understandYes = histOccupancy >= 0.5;
    const pYes = understandYes ? 0.916827 : 0.083173;
    const captionCe = understandYes ? 0.086836 : 2.4869;
    const genL2 = frameL2(generated, truthNext);
    const copyL2 = frameL2(copied, truthNext);
    return {
      observed,
      truthNext,
      generated,
      histOccupancy,
      genOccupancy,
      copyOccupancy,
      understandYes,
      pYes,
      captionCe,
      genL2,
      copyL2,
      cePositions: 1,
      genPositions: HEIGHT * WIDTH,
      intersection: 0,
    };
  }, [forget, historyHasCup]);

  const passed =
    revealed &&
    prediction === "understand_ok_gen_vanish" &&
    historyHasCup &&
    forget >= 0.75 &&
    calculation.understandYes &&
    calculation.genOccupancy < 0.25;

  const completion = useMemo(
    () => ({
      lessonId: 49,
      prediction,
      forget: round(forget, 2),
      historyHasCup,
      pYes: round(calculation.pYes, 4),
      genOccupancy: round(calculation.genOccupancy, 4),
      genL2: round(calculation.genL2, 4),
      copyL2: round(calculation.copyL2, 4),
      intersection: calculation.intersection,
    }),
    [
      calculation.copyL2,
      calculation.genL2,
      calculation.genOccupancy,
      calculation.intersection,
      calculation.pYes,
      forget,
      historyHasCup,
      prediction,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function reset() {
    setForget(0.3);
    setHistoryHasCup(true);
    setPrediction("");
    setRevealed(false);
  }

  return (
    <LabFrame
      lesson="49"
      title="同一段：答杯子还在，还是画下一帧"
      description="教学模拟，不是模型输出。先选预测，再揭晓理解 CE 与生成帧。必须造出：理解答对、生成帧里杯子消失。"
    >
      <div className={styles.workspace}>
        <form
          className={styles.controls}
          onSubmit={(event) => event.preventDefault()}
        >
          <h3>分账控制台</h3>
          <label>
            <span>
              生成遗忘系数 forget <output>{forget.toFixed(2)}</output>
            </span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={forget}
              onChange={(event) => {
                setForget(Number(event.target.value));
                setRevealed(false);
              }}
            />
          </label>
          <fieldset>
            <legend>可见历史里有没有杯子</legend>
            <label>
              <input
                type="radio"
                name="history-cup"
                checked={historyHasCup}
                onChange={() => {
                  setHistoryHasCup(true);
                  setRevealed(false);
                }}
              />
              <span>有杯子（理解应答还在）</span>
            </label>
            <label>
              <input
                type="radio"
                name="history-cup"
                checked={!historyHasCup}
                onChange={() => {
                  setHistoryHasCup(false);
                  setRevealed(false);
                }}
              />
              <span>没有杯子（理解会答不在）</span>
            </label>
          </fieldset>
          <p className={styles.note}>
            forget=0 是抄最后一帧；forget=1 是把格子填成 0.5。纹理随时间变，均值填充的帧差可以低于抄帧，杯子占用却掉到 0。
          </p>
        </form>

        <div className={styles.stage}>
          <div className={styles.timeline} aria-label="五帧可见历史">
            {calculation.observed.map((grid, step) => (
              <div key={step} className={styles.frame}>
                <b>t={step}</b>
                <div className={styles.frameGrid}>
                  {grid.flatMap((row, rowIndex) =>
                    row.map((value, colIndex) => (
                      <i
                        key={`obs-${step}-${rowIndex}-${colIndex}`}
                        style={{ background: shade(value) }}
                      />
                    )),
                  )}
                </div>
              </div>
            ))}
          </div>
          <div className={styles.panels}>
            <figure className={styles.panelActive}>
              <figcaption>理解账 · 杯子还在不在</figcaption>
              <div className={styles.answerCard}>
                <p>问：杯子还在桌上吗？</p>
                <strong>
                  {revealed
                    ? calculation.understandYes
                      ? "还在"
                      : "不在"
                    : "揭晓后"}
                </strong>
                <span>
                  p(还在){" "}
                  {revealed ? calculation.pYes.toFixed(3) : "—"}
                </span>
                <span>
                  caption CE{" "}
                  {revealed ? calculation.captionCe.toFixed(3) : "—"}
                </span>
              </div>
            </figure>
            <figure className={styles.panelActive}>
              <figcaption>生成账 · 预测 t=5</figcaption>
              <div className={styles.recon}>
                {(revealed
                  ? calculation.generated
                  : calculation.observed[T_OBS - 1]
                ).flatMap((row, rowIndex) =>
                  row.map((value, colIndex) => (
                    <i
                      key={`gen-${rowIndex}-${colIndex}`}
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
          </div>
          <div className={styles.maskRow} aria-label="两张 loss mask">
            <span>CE</span>
            <div className={styles.maskTrack}>
              <i className={styles.hist} style={{ width: "62%" }} />
              <i className={styles.prompt} style={{ width: "8%" }} />
              <i
                className={styles.ce}
                data-active={revealed ? "true" : "false"}
                style={{ width: "6%" }}
              />
              <i className={styles.futureOff} style={{ width: "24%" }} />
            </div>
            <span>帧差</span>
            <div className={styles.maskTrack}>
              <i className={styles.hist} style={{ width: "62%" }} />
              <i className={styles.prompt} style={{ width: "8%" }} />
              <i className={styles.futureOff} style={{ width: "6%" }} />
              <i
                className={styles.gen}
                data-active={revealed ? "true" : "false"}
                style={{ width: "24%" }}
              />
            </div>
          </div>
          <dl className={styles.metrics}>
            <div>
              <dt>历史占用</dt>
              <dd>
                {revealed ? calculation.histOccupancy.toFixed(2) : "—"}
              </dd>
            </div>
            <div>
              <dt>生成占用</dt>
              <dd>
                {revealed ? calculation.genOccupancy.toFixed(2) : "—"}
              </dd>
            </div>
            <div>
              <dt>生成 L2</dt>
              <dd>{revealed ? calculation.genL2.toFixed(3) : "—"}</dd>
            </div>
            <div>
              <dt>mask 交</dt>
              <dd>{revealed ? calculation.intersection : "—"}</dd>
            </div>
          </dl>
          <p className={styles.formula}>
            L_und = CE(answer) ； L_gen = mean ||Ihat_5 - I_5||^2 ；
            occupancy = (0.5 - cup_mean) / 0.42 ； M_und ∩ M_gen = empty。
            数字来自教学夹具。
          </p>
        </div>
      </div>

      <div className={styles.predict}>
        <fieldset>
          <legend>先预测：把 forget 拉高之后，哪句话会出现？</legend>
          {PREDICTIONS.map((item) => (
            <label key={item.value}>
              <input
                type="radio"
                name="video-gen-prediction"
                value={item.value}
                checked={prediction === item.value}
                onChange={() => {
                  setPrediction(item.value);
                  setRevealed(false);
                }}
              />
              <span>{item.label}</span>
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
            揭晓两本账
          </button>
        </div>
      </div>
      {revealed && prediction !== "understand_ok_gen_vanish" && (
        <p className={styles.feedback}>
          理解 CE 只看答案 token。forget 把下一帧涂成均值时，杯子占用掉到接近 0，帧差却可能比抄上一帧更低。
        </p>
      )}
      {revealed && prediction === "understand_ok_gen_vanish" && !historyHasCup && (
        <p className={styles.feedback}>
          历史里没有杯子时，理解也会答不在。把“可见历史里有没有杯子”拨回有杯子。
        </p>
      )}
      {revealed &&
        prediction === "understand_ok_gen_vanish" &&
        historyHasCup &&
        forget < 0.75 && (
          <p className={styles.feedback}>
            forget 还不够大，生成帧仍残留杯子。拖到 0.75 或以上，占用会低于 0.25。
          </p>
        )}
      <Gate passed={passed}>
        {passed
          ? "理解答还在，生成帧杯子消失，两张 mask 不相交。教学模拟通过。"
          : "先选“理解对、生成帧消失”，保留历史杯子，把 forget 拉到 0.75 以上再揭晓。"}
      </Gate>
    </LabFrame>
  );
}
