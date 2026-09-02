"use client";

import { useMemo, useState } from "react";
import styles from "./Lab08GradientProjection.module.css";
import type { FoundationLabProps } from "./types";
import { initialNumber } from "./types";

type DotSign = "pos" | "zero" | "neg";

const G_OLD: [number, number] = [1, 0];

function projectAgem(gNew: [number, number]): [number, number] {
  const dot = gNew[0] * G_OLD[0] + gNew[1] * G_OLD[1];
  const denom = G_OLD[0] * G_OLD[0] + G_OLD[1] * G_OLD[1];
  if (dot >= 0) return gNew;
  const alpha = dot / denom;
  return [gNew[0] - alpha * G_OLD[0], gNew[1] - alpha * G_OLD[1]];
}

function signOf(value: number): DotSign {
  if (Math.abs(value) < 1e-6) return "zero";
  return value > 0 ? "pos" : "neg";
}

function arrow(from: [number, number], vec: [number, number], scale: number) {
  const x2 = from[0] + vec[0] * scale;
  const y2 = from[1] - vec[1] * scale;
  return { x1: from[0], y1: from[1], x2, y2 };
}

export function Lab08GradientProjection({
  onComplete,
  initialState,
}: FoundationLabProps) {
  const [angle, setAngle] = useState(initialNumber(initialState, "angle", 120));
  const [dotPred, setDotPred] = useState<DotSign | null>(null);
  const [gdumbPred, setGdumbPred] = useState<"free" | "constrained" | null>(
    null,
  );
  const [hasRun, setHasRun] = useState(false);

  const geom = useMemo(() => {
    const rad = (angle * Math.PI) / 180;
    const gNew: [number, number] = [Math.cos(rad), Math.sin(rad)];
    const projected = projectAgem(gNew);
    const projDotNew = projected[0] * gNew[0] + projected[1] * gNew[1];
    const projDotOld = projected[0] * G_OLD[0] + projected[1] * G_OLD[1];
    const gdumb: [number, number] = [
      (G_OLD[0] + gNew[0]) / 2,
      (G_OLD[1] + gNew[1]) / 2,
    ];
    return {
      gNew,
      projected,
      projDotNew,
      projDotOld,
      gdumb,
      sign: signOf(projDotNew),
    };
  }, [angle]);

  const gatePassed = hasRun && dotPred === geom.sign && gdumbPred === "free";

  function invalidate() {
    setHasRun(false);
  }

  function runLab() {
    setHasRun(true);
    if (dotPred === geom.sign && gdumbPred === "free") {
      onComplete?.({
        angle,
        projDotNew: geom.projDotNew,
        sign: geom.sign,
      });
    }
  }

  function reset() {
    setAngle(120);
    setDotPred(null);
    setGdumbPred(null);
    setHasRun(false);
  }

  const origin: [number, number] = [170, 160];
  const scale = 90;
  const aNew = arrow(origin, geom.gNew, scale);
  const aOld = arrow(origin, G_OLD, scale);
  const aProj = arrow(origin, geom.projected, scale);
  const aDumb = arrow(origin, geom.gdumb, scale);

  return (
    <section className={styles.lab} aria-labelledby="lab08-title">
      <header className={styles.header}>
        <div>
          <div className={styles.labels}>
            <span>A-GEM</span>
            <span>GDumb</span>
          </div>
          <h3 id="lab08-title">梯度投影：半平面里还指不指向新任务</h3>
          <p>
            A-GEM 要求新梯度不要增加旧任务损失：g·g_old ≥ 0。若违规，就投影到旧梯度的正交补。旁边的 GDumb 不投影，只用背包重训。
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
              新、旧梯度夹角 <strong>{angle}°</strong>
            </span>
            <input
              type="range"
              min="0"
              max="180"
              step="15"
              value={angle}
              onChange={(event) => {
                setAngle(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <div className={styles.formula}>
            <code>若 g_new·g_old &lt; 0</code>
            <code>g ← g_new − ((g_new·g_old)/‖g_old‖²) g_old</code>
          </div>
        </div>

        <div className={styles.stage} aria-live="polite">
          <div className={styles.metrics}>
            <div>
              <span>投影 · 新梯度</span>
              <strong>{hasRun ? geom.projDotNew.toFixed(2) : "?"}</strong>
            </div>
            <div>
              <span>投影 · 旧梯度</span>
              <strong>{hasRun ? geom.projDotOld.toFixed(2) : "?"}</strong>
            </div>
            <div>
              <span>符号</span>
              <strong>
                {hasRun
                  ? geom.sign === "zero"
                    ? "零"
                    : geom.sign === "pos"
                      ? "正"
                      : "负"
                  : "?"}
              </strong>
            </div>
          </div>
          <svg
            className={styles.plot}
            viewBox="0 0 340 300"
            role="img"
            aria-label="二维梯度与可行半平面"
          >
            <rect
              x="170"
              y="20"
              width="150"
              height="250"
              fill="#1f6f8a"
              opacity="0.08"
            />
            <line x1="20" y1="160" x2="320" y2="160" stroke="currentColor" />
            <line x1="170" y1="20" x2="170" y2="280" stroke="currentColor" />
            <text x="250" y="36" fontSize="11">
              可行半平面
            </text>
            <line
              x1={aOld.x1}
              y1={aOld.y1}
              x2={aOld.x2}
              y2={aOld.y2}
              stroke="#5b6678"
              strokeWidth="3"
            />
            {hasRun ? (
              <>
                <line
                  x1={aNew.x1}
                  y1={aNew.y1}
                  x2={aNew.x2}
                  y2={aNew.y2}
                  stroke="#c47b12"
                  strokeWidth="2.4"
                />
                <line
                  x1={aProj.x1}
                  y1={aProj.y1}
                  x2={aProj.x2}
                  y2={aProj.y2}
                  stroke="#2156c8"
                  strokeWidth="3"
                />
                <line
                  x1={aDumb.x1}
                  y1={aDumb.y1}
                  x2={aDumb.x2}
                  y2={aDumb.y2}
                  stroke="#1b7a53"
                  strokeWidth="2"
                  strokeDasharray="5 4"
                />
              </>
            ) : null}
            <text x="24" y="290" fontSize="11">
              灰=旧 橙=新 蓝=投影 绿虚线=GDumb
            </text>
          </svg>
        </div>
      </div>

      <div className={styles.prediction}>
        <fieldset>
          <legend>预测 1：当前夹角下，投影后与新梯度的点积是？</legend>
          <div className={styles.choiceRow}>
            <button
              type="button"
              aria-pressed={dotPred === "pos"}
              onClick={() => {
                setDotPred("pos");
                invalidate();
              }}
            >
              正（还指向新任务）
            </button>
            <button
              type="button"
              aria-pressed={dotPred === "zero"}
              onClick={() => {
                setDotPred("zero");
                invalidate();
              }}
            >
              零
            </button>
            <button
              type="button"
              aria-pressed={dotPred === "neg"}
              onClick={() => {
                setDotPred("neg");
                invalidate();
              }}
            >
              负
            </button>
          </div>
        </fieldset>
        <fieldset>
          <legend>预测 2：GDumb 要不要满足旧任务半平面约束？</legend>
          <div className={styles.choiceRow}>
            <button
              type="button"
              aria-pressed={gdumbPred === "free"}
              onClick={() => {
                setGdumbPred("free");
                invalidate();
              }}
            >
              不要，它用背包重训
            </button>
            <button
              type="button"
              aria-pressed={gdumbPred === "constrained"}
              onClick={() => {
                setGdumbPred("constrained");
                invalidate();
              }}
            >
              要，同样投影
            </button>
          </div>
        </fieldset>
        <button
          type="button"
          className={styles.run}
          disabled={!dotPred || !gdumbPred}
          onClick={runLab}
        >
          投影并对照
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
            ? "夹角会改变点积符号。180° 时投影是零向量。"
            : gatePassed
              ? `点积为 ${geom.projDotNew.toFixed(2)}。GDumb 的虚线可以进左半平面，因为它根本不走这条约束。`
              : "投影是到半平面的最近点，与 g_new 的点积不会为负；180° 时才为零。"}
        </span>
      </div>
    </section>
  );
}
