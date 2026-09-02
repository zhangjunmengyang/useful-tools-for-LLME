"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import chrome from "./LabFrame.module.css";
import styles from "./Lesson20RlRazor.module.css";
import type { AdvancedLabProps } from "./types";
import { numberFrom, round } from "./labUtils";

type DistPred = "sft" | "rl" | "tie";

function sftPoint(dataX: number, dataY: number, steps: number) {
  const mix = 1 - (1 - 0.22) ** steps;
  return { x: mix * dataX, y: mix * dataY };
}

function rlPoint(dataX: number, dataY: number, steps: number) {
  let x = 0;
  let y = 0;
  for (let step = 0; step < steps; step += 1) {
    const gate = Math.exp(-Math.hypot(x, y) / 0.42);
    x += 0.07 * (dataX - x) * gate;
    y += 0.07 * (dataY - y) * gate;
  }
  return { x, y };
}

function retain(distance: number) {
  return Math.exp(-1.8 * distance * distance);
}

function newTask(x: number, y: number, dataX: number, dataY: number) {
  const d = Math.hypot(x - dataX, y - dataY);
  return Math.exp(-2.4 * d * d);
}

export function Lesson20RlRazor({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    radius: numberFrom(initialState, "radius", 0.95, 0.25, 1.2),
    angle: numberFrom(initialState, "angle", 32, 0, 180),
    sftSteps: numberFrom(initialState, "sftSteps", 8, 0, 12),
    rlSteps: numberFrom(initialState, "rlSteps", 8, 0, 12),
  };
  const [radius, setRadius] = useState(defaults.radius);
  const [angle, setAngle] = useState(defaults.angle);
  const [sftSteps, setSftSteps] = useState(defaults.sftSteps);
  const [rlSteps, setRlSteps] = useState(defaults.rlSteps);
  const [distPred, setDistPred] = useState<DistPred | null>(null);
  const [keepPred, setKeepPred] = useState<DistPred | null>(null);
  const [hasRun, setHasRun] = useState(false);

  const geo = useMemo(() => {
    const dataX = radius * Math.cos((angle * Math.PI) / 180);
    const dataY = radius * Math.sin((angle * Math.PI) / 180);
    const sft = sftPoint(dataX, dataY, sftSteps);
    const rl = rlPoint(dataX, dataY, rlSteps);
    const dSft = Math.hypot(sft.x, sft.y);
    const dRl = Math.hypot(rl.x, rl.y);
    const rSft = retain(dSft);
    const rRl = retain(dRl);
    const distWinner: DistPred =
      dSft - dRl > 0.08 ? "sft" : dRl - dSft > 0.08 ? "rl" : "tie";
    const keepWinner: DistPred =
      rRl - rSft > 0.05 ? "rl" : rSft - rRl > 0.05 ? "sft" : "tie";
    return {
      dataX,
      dataY,
      sft,
      rl,
      dSft: round(dSft, 3),
      dRl: round(dRl, 3),
      rSft: round(rSft, 3),
      rRl: round(rRl, 3),
      nSft: round(newTask(sft.x, sft.y, dataX, dataY), 3),
      nRl: round(newTask(rl.x, rl.y, dataX, dataY), 3),
      distWinner,
      keepWinner,
    };
  }, [angle, radius, rlSteps, sftSteps]);

  const gatePassed =
    hasRun && distPred === geo.distWinner && keepPred === geo.keepWinner;

  function invalidate() {
    setHasRun(false);
  }

  function run() {
    setHasRun(true);
    if (distPred === geo.distWinner && keepPred === geo.keepWinner) {
      onComplete?.({
        radius,
        angle,
        sftSteps,
        rlSteps,
        distanceSft: geo.dSft,
        distanceRl: geo.dRl,
        retainSft: geo.rSft,
        retainRl: geo.rRl,
        klStandIn: "euclidean ||θ-θ_ref|| in 2D; not a 7B KL",
      });
    }
  }

  function reset() {
    setRadius(defaults.radius);
    setAngle(defaults.angle);
    setSftSteps(defaults.sftSteps);
    setRlSteps(defaults.rlSteps);
    setDistPred(null);
    setKeepPred(null);
    setHasRun(false);
  }

  function project(x: number, y: number) {
    return { cx: 50 + x * 38, cy: 50 - y * 38 };
  }

  const origin = project(0, 0);
  const data = project(geo.dataX, geo.dataY);
  const sft = project(geo.sft.x, geo.sft.y);
  const rl = project(geo.rl.x, geo.rl.y);

  return (
    <LabFrame
      lesson="20"
      title="SFT 拉走，RL 不远走"
      description="二维策略空间里，原模型在原点。SFT 把点拉向离线数据中心；on-policy RL 沿着当前策略小步走，并被到原点的距离门控。图上的欧氏距离是 KL 的教学替代，不是真实 7B 的 KL 读数。"
      onReset={reset}
    >
      <div className={chrome.workbench}>
        <div className={chrome.controls}>
          <label>
            <span>
              离线数据半径 <strong>{radius.toFixed(2)}</strong>
            </span>
            <input
              type="range"
              min="0.25"
              max="1.2"
              step="0.05"
              value={radius}
              onChange={(event) => {
                setRadius(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <label>
            <span>
              数据方向 <strong>{angle}°</strong>
            </span>
            <input
              type="range"
              min="0"
              max="180"
              step="2"
              value={angle}
              onChange={(event) => {
                setAngle(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <label>
            <span>
              SFT 步数 <strong>{sftSteps}</strong>
            </span>
            <input
              type="range"
              min="0"
              max="12"
              step="1"
              value={sftSteps}
              onChange={(event) => {
                setSftSteps(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <label>
            <span>
              RL 步数 <strong>{rlSteps}</strong>
            </span>
            <input
              type="range"
              min="0"
              max="12"
              step="1"
              value={rlSteps}
              onChange={(event) => {
                setRlSteps(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <div className={chrome.formula}>
            <code>d = ‖θ − θ_ref‖　（教学距离，代替 KL）</code>
            <code>retain = exp(−1.8 d²)</code>
            <code>θ_SFT → μ_off；θ_RL 受 exp(−‖θ‖/0.42) 门控</code>
          </div>
        </div>

        <div className={chrome.panel}>
          <div className={chrome.metrics}>
            <div className={chrome.metric}>
              <span>SFT 距离 / 保持</span>
              <strong>
                {hasRun ? `${geo.dSft.toFixed(2)} / ${geo.rSft.toFixed(2)}` : "?"}
              </strong>
            </div>
            <div className={chrome.metric}>
              <span>RL 距离 / 保持</span>
              <strong>
                {hasRun ? `${geo.dRl.toFixed(2)} / ${geo.rRl.toFixed(2)}` : "?"}
              </strong>
            </div>
            <div className={chrome.metric}>
              <span>新任务（SFT / RL）</span>
              <strong>
                {hasRun ? `${geo.nSft.toFixed(2)} / ${geo.nRl.toFixed(2)}` : "?"}
              </strong>
            </div>
          </div>
          <svg className={styles.plane} viewBox="0 0 100 100" aria-label="二维策略空间">
            <circle cx="50" cy="50" r="18" className={styles.old} />
            <line x1="8" y1="50" x2="92" y2="50" />
            <line x1="50" y1="8" x2="50" y2="92" />
            <circle cx={origin.cx} cy={origin.cy} r="2.2" className={styles.ref} />
            {hasRun ? (
              <>
                <circle cx={data.cx} cy={data.cy} r="2.4" className={styles.data} />
                <circle cx={sft.cx} cy={sft.cy} r="2.8" className={styles.sft} />
                <circle cx={rl.cx} cy={rl.cy} r="2.8" className={styles.rl} />
              </>
            ) : null}
          </svg>
          <p className={chrome.note}>
            圆心浅圈是旧任务高密度区。黑点原模型，空心点离线数据，深色 SFT，绿色 RL。数字全部由上面的二维公式算出。
          </p>
        </div>
      </div>

      <div className={chrome.prediction}>
        <fieldset>
          <legend>预测 1：谁离原点（原模型）更远？</legend>
          <div className={chrome.choiceRow}>
            <button
              type="button"
              aria-pressed={distPred === "sft"}
              onClick={() => {
                setDistPred("sft");
                invalidate();
              }}
            >
              SFT
            </button>
            <button
              type="button"
              aria-pressed={distPred === "rl"}
              onClick={() => {
                setDistPred("rl");
                invalidate();
              }}
            >
              RL
            </button>
            <button
              type="button"
              aria-pressed={distPred === "tie"}
              onClick={() => {
                setDistPred("tie");
                invalidate();
              }}
            >
              接近
            </button>
          </div>
        </fieldset>
        <fieldset>
          <legend>预测 2：谁的旧任务保持更高？</legend>
          <div className={chrome.choiceRow}>
            <button
              type="button"
              aria-pressed={keepPred === "rl"}
              onClick={() => {
                setKeepPred("rl");
                invalidate();
              }}
            >
              RL
            </button>
            <button
              type="button"
              aria-pressed={keepPred === "sft"}
              onClick={() => {
                setKeepPred("sft");
                invalidate();
              }}
            >
              SFT
            </button>
            <button
              type="button"
              aria-pressed={keepPred === "tie"}
              onClick={() => {
                setKeepPred("tie");
                invalidate();
              }}
            >
              接近
            </button>
          </div>
        </fieldset>
        <button
          type="button"
          className={chrome.run}
          disabled={!distPred || !keepPred}
          onClick={run}
        >
          运行策略点
        </button>
      </div>

      <Gate ran={hasRun} passed={gatePassed}>
        {!hasRun
          ? "先判断距离和旧任务保持，再揭示两个策略点。"
          : gatePassed
            ? `SFT d=${geo.dSft.toFixed(2)} 保持 ${geo.rSft.toFixed(2)}；RL d=${geo.dRl.toFixed(2)} 保持 ${geo.rRl.toFixed(2)}。`
            : "默认设定下离线数据远离原点，SFT 几乎走到数据中心，RL 被距离门控留在近处，保持率更高。把数据半径拖到很小，两者会接近。"}
      </Gate>
    </LabFrame>
  );
}
