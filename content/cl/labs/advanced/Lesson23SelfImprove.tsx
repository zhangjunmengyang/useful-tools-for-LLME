"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import chrome from "./LabFrame.module.css";
import styles from "./Lesson23SelfImprove.module.css";
import type { AdvancedLabProps } from "./types";
import { booleanFrom, numberFrom, polylinePoints, round } from "./labUtils";

type Dir = "up" | "down";

function loop(rounds: number, filterOn: boolean) {
  let acc = 0.62;
  let old = 0.91;
  const series = [
    {
      acc: round(acc, 3),
      old: round(old, 3),
      errorShare: round(1 - acc, 3),
    },
  ];
  for (let round = 1; round <= rounds; round += 1) {
    if (filterOn) {
      acc = acc + 0.28 * (0.91 - acc);
      old = Math.max(0.7, old - 0.012);
    } else {
      acc = acc - 0.35 * acc * (1 - acc);
      old = Math.max(0.45, old - 0.03);
    }
    acc = Math.min(0.99, Math.max(0.05, acc));
    series.push({
      acc: round(acc, 3),
      old: round(old, 3),
      errorShare: round(filterOn ? 0.08 : 1 - acc, 3),
    });
  }
  return series;
}

export function Lesson23SelfImprove({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    rounds: numberFrom(initialState, "rounds", 3, 2, 6),
    filterOn: booleanFrom(initialState, "filterOn", true),
  };
  const [rounds, setRounds] = useState(defaults.rounds);
  const [filterOn, setFilterOn] = useState(defaults.filterOn);
  const [offPred, setOffPred] = useState<Dir | null>(null);
  const [onPred, setOnPred] = useState<Dir | null>(null);
  const [hasRun, setHasRun] = useState(false);

  const filtered = useMemo(() => loop(rounds, true), [rounds]);
  const unfiltered = useMemo(() => loop(rounds, false), [rounds]);
  const active = filterOn ? filtered : unfiltered;
  const offDown = unfiltered[unfiltered.length - 1].acc < unfiltered[0].acc;
  const onUp = filtered[filtered.length - 1].acc > filtered[0].acc;
  const gatePassed =
    hasRun &&
    offPred === (offDown ? "down" : "up") &&
    onPred === (onUp ? "up" : "down") &&
    offDown;

  function invalidate() {
    setHasRun(false);
  }

  function run() {
    setHasRun(true);
    const passed =
      offPred === (offDown ? "down" : "up") &&
      onPred === (onUp ? "up" : "down") &&
      offDown;
    if (passed) {
      onComplete?.({
        rounds,
        filterOn,
        accOff: unfiltered[unfiltered.length - 1].acc,
        accOn: filtered[filtered.length - 1].acc,
        collapse: offDown,
      });
    }
  }

  function reset() {
    setRounds(defaults.rounds);
    setFilterOn(true);
    setOffPred(null);
    setOnPred(null);
    setHasRun(false);
  }

  const yMax = 1;
  const onPts = polylinePoints(
    filtered.map((row) => row.acc),
    280,
    88,
    yMax,
  );
  const offPts = polylinePoints(
    unfiltered.map((row) => row.acc),
    280,
    88,
    yMax,
  );

  return (
    <LabFrame
      lesson="23"
      title="自改进环：关掉筛选会越训越错"
      description="四块：生成训练数据、用验证集筛选、训练自己、再评测。关掉筛选后，错误样本被当成真理，验证分一轮比一轮低。这是课内缩小模拟，用来测「自我生成数据」会不会崩，不是 AGI 自迭代。"
      onReset={reset}
    >
      <div className={chrome.workbench}>
        <div className={chrome.controls}>
          <label>
            <span>
              轮数 <strong>{rounds}</strong>
            </span>
            <input
              type="range"
              min="2"
              max="6"
              step="1"
              value={rounds}
              onChange={(event) => {
                setRounds(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <label>
            <span>筛选</span>
            <select
              value={filterOn ? "on" : "off"}
              onChange={(event) => {
                setFilterOn(event.target.value === "on");
                invalidate();
              }}
            >
              <option value="on">打开</option>
              <option value="off">关掉</option>
            </select>
          </label>
          <div className={chrome.formula}>
            <code>开筛选：acc ← acc + 0.28(0.91 − acc)</code>
            <code>关筛选：acc ← acc − 0.35 acc(1−acc)</code>
            <code>errorShare = 1 − acc　（无筛选时进入训练）</code>
          </div>
        </div>

        <div className={chrome.panel}>
          <ol className={styles.blocks}>
            {["生成", "筛选", "训练", "评测"].map((name) => (
              <li
                key={name}
                data-muted={name === "筛选" && !filterOn ? "true" : "false"}
              >
                {name}
              </li>
            ))}
          </ol>
          <div className={chrome.metrics}>
            <div className={chrome.metric}>
              <span>当前路径验证分</span>
              <strong>
                {hasRun ? active[active.length - 1].acc.toFixed(2) : "?"}
              </strong>
            </div>
            <div className={chrome.metric}>
              <span>无筛选终点</span>
              <strong>
                {hasRun ? unfiltered[unfiltered.length - 1].acc.toFixed(2) : "?"}
              </strong>
            </div>
            <div className={chrome.metric}>
              <span>有筛选终点</span>
              <strong>
                {hasRun ? filtered[filtered.length - 1].acc.toFixed(2) : "?"}
              </strong>
            </div>
          </div>
          <svg className={chrome.chart} viewBox="0 0 280 88" aria-label="验证分轨迹">
            <polyline points={hasRun ? onPts : ""} stroke="#1b7a53" strokeWidth="2" />
            <polyline points={hasRun ? offPts : ""} stroke="#875e16" strokeWidth="2" />
          </svg>
          <table className={chrome.table}>
            <thead>
              <tr>
                <th>轮</th>
                <th>无筛选 acc</th>
                <th>错误占比</th>
                <th>有筛选 acc</th>
              </tr>
            </thead>
            <tbody>
              {unfiltered.map((row, index) => (
                <tr key={index}>
                  <td>{index}</td>
                  <td>{hasRun ? row.acc.toFixed(2) : "—"}</td>
                  <td>{hasRun ? `${Math.round(row.errorShare * 100)}%` : "—"}</td>
                  <td>{hasRun ? filtered[index].acc.toFixed(2) : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className={chrome.prediction}>
        <fieldset>
          <legend>预测 1：关掉筛选，验证分会升还是降？</legend>
          <div className={chrome.choiceRow}>
            <button
              type="button"
              aria-pressed={offPred === "up"}
              onClick={() => {
                setOffPred("up");
                invalidate();
              }}
            >
              升
            </button>
            <button
              type="button"
              aria-pressed={offPred === "down"}
              onClick={() => {
                setOffPred("down");
                invalidate();
              }}
            >
              降（越训越错）
            </button>
          </div>
        </fieldset>
        <fieldset>
          <legend>预测 2：打开筛选，验证分会升还是降？</legend>
          <div className={chrome.choiceRow}>
            <button
              type="button"
              aria-pressed={onPred === "up"}
              onClick={() => {
                setOnPred("up");
                invalidate();
              }}
            >
              升
            </button>
            <button
              type="button"
              aria-pressed={onPred === "down"}
              onClick={() => {
                setOnPred("down");
                invalidate();
              }}
            >
              降
            </button>
          </div>
        </fieldset>
        <button
          type="button"
          className={chrome.run}
          disabled={!offPred || !onPred}
          onClick={run}
        >
          运行自改进
        </button>
      </div>

      <Gate ran={hasRun} passed={gatePassed}>
        {!hasRun
          ? "先判断两条路径的方向，再揭示生成-筛选-训练-评测。"
          : gatePassed
            ? `无筛选从 0.62 降到 ${unfiltered[unfiltered.length - 1].acc.toFixed(2)}；有筛选升到 ${filtered[filtered.length - 1].acc.toFixed(2)}。`
            : "关筛选时错误样本按 acc(1−acc) 自我强化，验证分单调下降。开筛选则朝 0.91 的验证质量靠拢。"}
      </Gate>
    </LabFrame>
  );
}
