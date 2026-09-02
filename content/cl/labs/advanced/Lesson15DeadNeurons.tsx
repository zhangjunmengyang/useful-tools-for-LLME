"use client";

import { useMemo, useState, type CSSProperties } from "react";
import { Gate, LabFrame } from "./LabFrame";
import chrome from "./LabFrame.module.css";
import styles from "./Lesson15DeadNeurons.module.css";
import type { AdvancedLabProps } from "./types";
import { mean, numberFrom, polylinePoints, round } from "./labUtils";

const LAYERS = ["L1 嵌入", "L2", "L3", "L4 输出"] as const;
const P_DIE = [0.07, 0.1, 0.13, 0.17];

type SpeedPred = "sgd" | "cbp" | "tie";

function evolve(tasks: number, rho: number) {
  const sgdDead: number[][] = [];
  const cbpDead: number[][] = [];
  const speedSgd: number[] = [];
  const speedCbp: number[] = [];
  let dSgd = [0, 0, 0, 0];
  let dCbp = [0, 0, 0, 0];

  for (let task = 0; task < tasks; task += 1) {
    dSgd = dSgd.map((dead, layer) => dead + P_DIE[layer] * (1 - dead));
    dCbp = dCbp.map((dead, layer) => {
      const grown = dead + P_DIE[layer] * (1 - dead);
      return grown * (1 - rho);
    });
    sgdDead.push(dSgd.map((dead) => round(dead, 3)));
    cbpDead.push(dCbp.map((dead) => round(dead, 3)));
    speedSgd.push(round(mean(dSgd.map((dead) => 1 - dead)), 3));
    speedCbp.push(round(mean(dCbp.map((dead) => 1 - dead)), 3));
  }

  return { sgdDead, cbpDead, speedSgd, speedCbp };
}

export function Lesson15DeadNeurons({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    tasks: numberFrom(initialState, "tasks", 8, 1, 12),
    rho: numberFrom(initialState, "rho", 0.32, 0, 0.6),
  };
  const [tasks, setTasks] = useState(defaults.tasks);
  const [rho, setRho] = useState(defaults.rho);
  const [speedPred, setSpeedPred] = useState<SpeedPred | null>(null);
  const [resetPred, setResetPred] = useState<"drop" | "same" | null>(null);
  const [hasRun, setHasRun] = useState(false);

  const sim = useMemo(() => evolve(tasks, rho), [rho, tasks]);
  const last = tasks - 1;
  const gap = sim.speedCbp[last] - sim.speedSgd[last];
  const winner: SpeedPred = gap > 0.04 ? "cbp" : gap < -0.04 ? "sgd" : "tie";
  const barsFall = mean(sim.cbpDead[last]) < mean(sim.sgdDead[last]) - 0.03;
  const resetAnswer: "drop" | "same" = barsFall ? "drop" : "same";
  const gatePassed =
    hasRun && speedPred === winner && resetPred === resetAnswer;

  function invalidate() {
    setHasRun(false);
  }

  function run() {
    setHasRun(true);
    const passed = speedPred === winner && resetPred === resetAnswer;
    if (passed) {
      onComplete?.({
        tasks,
        rho,
        speedSgd: sim.speedSgd[last],
        speedCbp: sim.speedCbp[last],
        deadSgd: round(mean(sim.sgdDead[last]), 3),
        deadCbp: round(mean(sim.cbpDead[last]), 3),
      });
    }
  }

  function reset() {
    setTasks(defaults.tasks);
    setRho(defaults.rho);
    setSpeedPred(null);
    setResetPred(null);
    setHasRun(false);
  }

  const sgdPoints = polylinePoints(sim.speedSgd, 280, 88, 1);
  const cbpPoints = polylinePoints(sim.speedCbp, 280, 88, 1);

  return (
    <LabFrame
      lesson="15"
      title="死神经元：学着学着学不动"
      description="死神经元是激活长期接近 0、几乎不再更新的单元。和旧任务考砸不是一件事：就算不考旧卷，后期任务的学习速度也会掉。continual backprop 按低使用率重置一部分单元。"
      onReset={reset}
    >
      <div className={chrome.workbench}>
        <div className={chrome.controls}>
          <label>
            <span>
              任务序号 <strong>第 {tasks} 个</strong>
            </span>
            <input
              type="range"
              min="1"
              max="12"
              step="1"
              value={tasks}
              onChange={(event) => {
                setTasks(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <label>
            <span>
              重置比例 ρ <strong>{rho.toFixed(2)}</strong>
            </span>
            <input
              type="range"
              min="0"
              max="0.6"
              step="0.02"
              value={rho}
              onChange={(event) => {
                setRho(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <div className={chrome.formula}>
            <code>dead ← dead + p·(1-dead)</code>
            <code>CBP: dead ← dead·(1-ρ)</code>
            <code>speed = mean(1 - dead)</code>
          </div>
        </div>

        <div className={chrome.panel}>
          <div className={chrome.metrics}>
            <div className={chrome.metric}>
              <span>SGD 学习速度</span>
              <strong>{hasRun ? sim.speedSgd[last].toFixed(2) : "?"}</strong>
            </div>
            <div className={chrome.metric}>
              <span>CBP 学习速度</span>
              <strong>{hasRun ? sim.speedCbp[last].toFixed(2) : "?"}</strong>
            </div>
            <div className={chrome.metric}>
              <span>SGD 平均死比例</span>
              <strong>
                {hasRun ? `${Math.round(mean(sim.sgdDead[last]) * 100)}%` : "?"}
              </strong>
            </div>
          </div>
          <div className={styles.bars} aria-label="各层死神经元比例">
            {LAYERS.map((name, layer) => (
              <div key={name} className={styles.barCol}>
                <div className={styles.pair}>
                  <span
                    className={styles.sgd}
                    style={
                      {
                        "--h": hasRun
                          ? `${Math.max(4, sim.sgdDead[last][layer] * 100)}%`
                          : "4%",
                      } as CSSProperties
                    }
                  />
                  <span
                    className={styles.cbp}
                    style={
                      {
                        "--h": hasRun
                          ? `${Math.max(4, sim.cbpDead[last][layer] * 100)}%`
                          : "4%",
                      } as CSSProperties
                    }
                  />
                </div>
                <small>{name}</small>
              </div>
            ))}
          </div>
          <svg
            className={chrome.chart}
            viewBox="0 0 280 88"
            aria-label="后期学习速度"
          >
            <polyline points={hasRun ? sgdPoints : ""} stroke="#8b9690" strokeWidth="2" />
            <polyline points={hasRun ? cbpPoints : ""} stroke="#1b7a53" strokeWidth="2" />
          </svg>
          <p className={chrome.note}>
            灰柱 / 灰线是普通反向传播，绿柱 / 绿线是 continual backprop。ρ = 0 时两条重合。
          </p>
        </div>
      </div>

      <div className={chrome.prediction}>
        <fieldset>
          <legend>预测 1：到这个任务，谁的学习速度更高？</legend>
          <div className={chrome.choiceRow}>
            <button
              type="button"
              aria-pressed={speedPred === "sgd"}
              onClick={() => {
                setSpeedPred("sgd");
                invalidate();
              }}
            >
              普通反向传播
            </button>
            <button
              type="button"
              aria-pressed={speedPred === "cbp"}
              onClick={() => {
                setSpeedPred("cbp");
                invalidate();
              }}
            >
              continual backprop
            </button>
            <button
              type="button"
              aria-pressed={speedPred === "tie"}
              onClick={() => {
                setSpeedPred("tie");
                invalidate();
              }}
            >
              接近
            </button>
          </div>
        </fieldset>
        <fieldset>
          <legend>预测 2：打开重置后，死神经元柱子会怎样？</legend>
          <div className={chrome.choiceRow}>
            <button
              type="button"
              aria-pressed={resetPred === "drop"}
              onClick={() => {
                setResetPred("drop");
                invalidate();
              }}
            >
              比 SGD 更矮
            </button>
            <button
              type="button"
              aria-pressed={resetPred === "same"}
              onClick={() => {
                setResetPred("same");
                invalidate();
              }}
            >
              几乎一样
            </button>
          </div>
        </fieldset>
        <button
          type="button"
          className={chrome.run}
          disabled={!speedPred || !resetPred}
          onClick={run}
        >
          运行可塑性
        </button>
      </div>

      <Gate ran={hasRun} passed={gatePassed}>
        {!hasRun
          ? "先判断后期学习速度和柱子高低，再揭示各层死比例。"
          : gatePassed
            ? `第 ${tasks} 个任务：CBP 速度 ${sim.speedCbp[last].toFixed(2)}，SGD ${sim.speedSgd[last].toFixed(2)}。`
            : "死比例随任务累加；ρ>0 时每层有一部分被重置，绿柱变矮，后期速度掉得更慢。ρ=0 时两者接近。"}
      </Gate>
    </LabFrame>
  );
}
