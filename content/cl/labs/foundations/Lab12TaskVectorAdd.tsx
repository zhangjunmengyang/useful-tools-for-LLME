"use client";

import { useMemo, useState } from "react";
import styles from "./Lab12TaskVectorAdd.module.css";
import type { FoundationLabProps } from "./types";
import { initialNumber } from "./types";

type Region = "q1" | "q2" | "q3" | "q4" | "origin";

const REGION_LABEL: Record<Region, string> = {
  q1: "第一象限",
  q2: "第二象限",
  q3: "第三象限",
  q4: "第四象限",
  origin: "原点附近",
};

const TAU_A: [number, number] = [1.1, 0.35];

function regionOf(x: number, y: number): Region {
  const eps = 0.08;
  if (Math.abs(x) < eps && Math.abs(y) < eps) return "origin";
  if (x >= 0 && y >= 0) return "q1";
  if (x < 0 && y >= 0) return "q2";
  if (x < 0 && y < 0) return "q3";
  return "q4";
}

function tiesMerge(a: [number, number], b: [number, number], trim: number) {
  function trimVec(v: [number, number]): [number, number] {
    const max = Math.max(Math.abs(v[0]), Math.abs(v[1]));
    const thr = trim * max;
    return [Math.abs(v[0]) >= thr ? v[0] : 0, Math.abs(v[1]) >= thr ? v[1] : 0];
  }
  const ta = trimVec(a);
  const tb = trimVec(b);
  const merged: [number, number] = [0, 0];
  for (let d = 0; d < 2; d += 1) {
    const pos = (ta[d] > 0 ? ta[d] : 0) + (tb[d] > 0 ? tb[d] : 0);
    const neg = (ta[d] < 0 ? -ta[d] : 0) + (tb[d] < 0 ? -tb[d] : 0);
    const sign = pos >= neg ? 1 : -1;
    const kept: number[] = [];
    if (Math.sign(ta[d]) === sign && ta[d] !== 0) kept.push(ta[d]);
    if (Math.sign(tb[d]) === sign && tb[d] !== 0) kept.push(tb[d]);
    merged[d] = kept.length ? kept.reduce((s, v) => s + v, 0) / kept.length : 0;
  }
  return merged;
}

export function Lab12TaskVectorAdd({
  onComplete,
  initialState,
}: FoundationLabProps) {
  const [angle, setAngle] = useState(initialNumber(initialState, "angle", 70));
  const [mag, setMag] = useState(initialNumber(initialState, "mag", 1.3));
  const [trim, setTrim] = useState(initialNumber(initialState, "trim", 0.5));
  const [sumPred, setSumPred] = useState<Region | null>(null);
  const [onlinePred, setOnlinePred] = useState<"no" | "yes" | null>(null);
  const [hasRun, setHasRun] = useState(false);

  const merge = useMemo(() => {
    const rad = (angle * Math.PI) / 180;
    const tauB: [number, number] = [mag * Math.cos(rad), mag * Math.sin(rad)];
    const sum: [number, number] = [TAU_A[0] + tauB[0], TAU_A[1] + tauB[1]];
    const ties = tiesMerge(TAU_A, tauB, trim);
    return {
      tauB,
      sum,
      ties,
      sumRegion: regionOf(sum[0], sum[1]),
      tiesRegion: regionOf(ties[0], ties[1]),
    };
  }, [angle, mag, trim]);

  const gatePassed =
    hasRun && sumPred === merge.sumRegion && onlinePred === "no";

  function invalidate() {
    setHasRun(false);
  }

  function runLab() {
    setHasRun(true);
    if (sumPred === merge.sumRegion && onlinePred === "no") {
      onComplete?.({
        angle,
        mag,
        trim,
        sum: merge.sum,
        ties: merge.ties,
        sumRegion: merge.sumRegion,
        tiesRegion: merge.tiesRegion,
      });
    }
  }

  function reset() {
    setAngle(70);
    setMag(1.3);
    setTrim(0.5);
    setSumPred(null);
    setOnlinePred(null);
    setHasRun(false);
  }

  function map(v: [number, number]) {
    return { x: 170 + v[0] * 52, y: 160 - v[1] * 52 };
  }

  const a = map(TAU_A);
  const b = map(merge.tauB);
  const s = map(merge.sum);
  const t = map(merge.ties);
  const o = { x: 170, y: 160 };

  const regionChoices: Region[] = ["q1", "q2", "q3", "q4", "origin"];

  return (
    <section className={styles.lab} aria-labelledby="lab12-title">
      <header className={styles.header}>
        <div>
          <div className={styles.labels}>
            <span>任务向量</span>
            <span>TIES</span>
          </div>
          <h3 id="lab12-title">任务向量加法：加、减、修剪落在哪一象限</h3>
          <p>
            任务向量是微调后的权重减去微调前。直接相加会在符号冲突的维度上对消。TIES 先修剪小分量，再按多数派符号合并。合并是事后缝合。
          </p>
        </div>
        <button className={styles.reset} type="button" onClick={reset}>
          重置实验
        </button>
      </header>

      <div className={styles.workbench}>
        <div className={styles.controls}>
          <label>
            <span>
              τ_B 方向 <strong>{angle}°</strong>
            </span>
            <input
              type="range"
              min="0"
              max="360"
              step="10"
              value={angle}
              onChange={(event) => {
                setAngle(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <label>
            <span>
              τ_B 长度 <strong>{mag.toFixed(2)}</strong>
            </span>
            <input
              type="range"
              min="0.4"
              max="1.8"
              step="0.05"
              value={mag}
              onChange={(event) => {
                setMag(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <label>
            <span>
              TIES 修剪比例 <strong>{trim.toFixed(2)}</strong>
            </span>
            <input
              type="range"
              min="0.1"
              max="0.8"
              step="0.05"
              value={trim}
              onChange={(event) => {
                setTrim(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <div className={styles.formula}>
            <code>τ = θ_ft − θ_0</code>
            <code>τ_sum = τ_A + τ_B</code>
            <code>τ_A = (1.10, 0.35)</code>
          </div>
        </div>

        <div className={styles.stage} aria-live="polite">
          <div className={styles.metrics}>
            <div>
              <span>相加</span>
              <strong>
                {hasRun
                  ? `(${merge.sum[0].toFixed(2)}, ${merge.sum[1].toFixed(2)})`
                  : "?"}
              </strong>
            </div>
            <div>
              <span>TIES</span>
              <strong>
                {hasRun
                  ? `(${merge.ties[0].toFixed(2)}, ${merge.ties[1].toFixed(2)})`
                  : "?"}
              </strong>
            </div>
            <div>
              <span>相加象限</span>
              <strong>{hasRun ? REGION_LABEL[merge.sumRegion] : "?"}</strong>
            </div>
          </div>
          <svg
            className={styles.plot}
            viewBox="0 0 340 300"
            role="img"
            aria-label="任务向量与合成向量"
          >
            <line x1="20" y1="160" x2="320" y2="160" stroke="currentColor" />
            <line x1="170" y1="20" x2="170" y2="290" stroke="currentColor" />
            {hasRun ? (
              <>
                <line
                  x1={o.x}
                  y1={o.y}
                  x2={a.x}
                  y2={a.y}
                  stroke="#2156c8"
                  strokeWidth="2.4"
                />
                <line
                  x1={o.x}
                  y1={o.y}
                  x2={b.x}
                  y2={b.y}
                  stroke="#c47b12"
                  strokeWidth="2.4"
                />
                <line
                  x1={o.x}
                  y1={o.y}
                  x2={s.x}
                  y2={s.y}
                  stroke="currentColor"
                  strokeWidth="3"
                />
                <line
                  x1={o.x}
                  y1={o.y}
                  x2={t.x}
                  y2={t.y}
                  stroke="#1b7a53"
                  strokeWidth="2"
                  strokeDasharray="5 4"
                />
              </>
            ) : null}
            <text x="24" y="20" fontSize="11">
              蓝 τ_A 橙 τ_B 黑相加 绿虚线 TIES
            </text>
          </svg>
        </div>
      </div>

      <div className={styles.prediction}>
        <fieldset>
          <legend>预测 1：直接相加的合成向量落在哪？</legend>
          <div className={styles.choiceRow}>
            {regionChoices.map((key) => (
              <button
                type="button"
                key={key}
                aria-pressed={sumPred === key}
                onClick={() => {
                  setSumPred(key);
                  invalidate();
                }}
              >
                {REGION_LABEL[key]}
              </button>
            ))}
          </div>
        </fieldset>
        <fieldset>
          <legend>预测 2：任务向量加法算不算在线持续学习？</legend>
          <div className={styles.choiceRow}>
            <button
              type="button"
              aria-pressed={onlinePred === "no"}
              onClick={() => {
                setOnlinePred("no");
                invalidate();
              }}
            >
              不算，这是事后缝合
            </button>
            <button
              type="button"
              aria-pressed={onlinePred === "yes"}
              onClick={() => {
                setOnlinePred("yes");
                invalidate();
              }}
            >
              算，因为它能找回多任务能力
            </button>
          </div>
        </fieldset>
        <button
          type="button"
          className={styles.run}
          disabled={!sumPred || !onlinePred}
          onClick={runLab}
        >
          合并向量
        </button>
      </div>

      <div
        className={`${styles.gate} ${
          hasRun ? (gatePassed ? styles.pass : styles.retry) : ""
        }`}
        role="status"
      >
        <strong>{gatePassed ? "验收已通过" : "完成验收"}</strong>
        <span>
          {!hasRun
            ? "先判断相加落点，再记住合并不是在线学习。"
            : gatePassed
              ? `相加在${REGION_LABEL[merge.sumRegion]}，TIES 在${REGION_LABEL[merge.tiesRegion]}。它能在不碰旧数据时回收能力，但仍是训练结束后的缝合。`
              : "τ_A + τ_B 的坐标符号决定象限。合并发生在训练之后，不是边做边学。"}
        </span>
      </div>
    </section>
  );
}
