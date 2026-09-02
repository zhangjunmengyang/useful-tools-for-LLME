"use client";

import { useMemo, useState } from "react";
import type { CSSProperties } from "react";
import styles from "./Lab03MetricLiar.module.css";
import type { FoundationLabProps } from "./types";
import { formatPct, initialString } from "./types";

type MethodKey = "liar" | "replay";

const methodMeta: Record<MethodKey, { name: string; blurb: string }> = {
  liar: {
    name: "后任务专家",
    blurb: "每个阶段只把分类器拧向当前任务，前面的格子逐渐被冲淡。",
  },
  replay: {
    name: "均匀回放",
    blurb: "缓冲里旧新各半，下三角比较满。",
  },
};

function metricsOf(matrix: number[][]) {
  const tasks = matrix.length;
  const last = matrix[tasks - 1];
  const acc = last.reduce((sum, value) => sum + value, 0) / tasks;
  const forgets: number[] = [];
  const bwts: number[] = [];
  for (let j = 0; j < tasks - 1; j += 1) {
    let peak = matrix[j][j];
    for (let i = j; i < tasks - 1; i += 1) {
      peak = Math.max(peak, matrix[i][j]);
    }
    forgets.push(peak - last[j]);
    bwts.push(last[j] - matrix[j][j]);
  }
  const forget = forgets.reduce((sum, value) => sum + value, 0) / forgets.length;
  const bwt = bwts.reduce((sum, value) => sum + value, 0) / bwts.length;
  const diagonal =
    matrix.reduce((sum, row, index) => sum + row[index], 0) / tasks;
  const isContinual = forget < 0.18 && bwt > -0.18;
  const accFlatters = acc >= 0.6 && forget > 0.2;
  return { acc, forget, bwt, diagonal, isContinual, accFlatters };
}

function liarMatrix(): number[][] {
  return [
    [0.92, 0, 0, 0, 0],
    [0.4, 0.9, 0, 0, 0],
    [0.28, 0.55, 0.88, 0, 0],
    [0.22, 0.4, 0.7, 0.94, 0],
    [0.18, 0.32, 0.78, 0.91, 0.96],
  ];
}

function replayMatrix(): number[][] {
  return [
    [0.91, 0, 0, 0, 0],
    [0.84, 0.89, 0, 0, 0],
    [0.81, 0.83, 0.88, 0, 0],
    [0.79, 0.81, 0.84, 0.9, 0],
    [0.77, 0.8, 0.83, 0.86, 0.91],
  ];
}

export function Lab03MetricLiar({
  onComplete,
  initialState,
}: FoundationLabProps) {
  const [method, setMethod] = useState<MethodKey>(
    initialString(initialState, "method", ["liar", "replay"] as const, "liar"),
  );
  const [clPrediction, setClPrediction] = useState<"yes" | "no" | null>(null);
  const [liarPrediction, setLiarPrediction] = useState<"flatters" | "honest" | null>(
    null,
  );
  const [hasRun, setHasRun] = useState(false);

  const result = useMemo(() => {
    const matrix = method === "liar" ? liarMatrix() : replayMatrix();
    return { matrix, ...metricsOf(matrix) };
  }, [method]);

  const gatePassed =
    hasRun &&
    clPrediction === (result.isContinual ? "yes" : "no") &&
    liarPrediction === (result.accFlatters ? "flatters" : "honest");

  function invalidate() {
    setHasRun(false);
  }

  function runLab() {
    setHasRun(true);
    const passed =
      clPrediction === (result.isContinual ? "yes" : "no") &&
      liarPrediction === (result.accFlatters ? "flatters" : "honest");
    if (passed) {
      onComplete?.({
        method,
        acc: result.acc,
        forget: result.forget,
        bwt: result.bwt,
        isContinual: result.isContinual,
      });
    }
  }

  function reset() {
    setMethod("liar");
    setClPrediction(null);
    setLiarPrediction(null);
    setHasRun(false);
  }

  return (
    <section className={styles.lab} aria-labelledby="lab03-title">
      <header className={styles.header}>
        <div>
          <div className={styles.labels}>
            <span>矩阵指标</span>
            <span>评测协议</span>
          </div>
          <h3 id="lab03-title">指标打假器：平均准确率会帮谁说话</h3>
          <p>
            准确率矩阵 R[i][j] 是学完任务 i 之后测任务 j 的分数。遗忘是峰值减去最后一行；BWT（后向转移：学完后面的，回头看前面掉了多少）是最后一行减对角线。
          </p>
        </div>
        <button className={styles.reset} type="button" onClick={reset}>
          重置实验
        </button>
      </header>

      <div className={styles.workbench}>
        <div className={styles.controls}>
          <label>
            <span>方法档案</span>
            <select
              value={method}
              onChange={(event) => {
                setMethod(event.target.value as MethodKey);
                invalidate();
              }}
            >
              <option value="liar">{methodMeta.liar.name}</option>
              <option value="replay">{methodMeta.replay.name}</option>
            </select>
          </label>
          <p>{methodMeta[method].blurb}</p>
          <div className={styles.formula}>
            <code>ACC = mean(R[T])</code>
            <code>Forget_j = max_i R[i,j] − R[T,j]</code>
            <code>BWT = mean_j (R[T,j] − R[j,j])</code>
          </div>
        </div>

        <div className={styles.stage} aria-live="polite">
          <div className={styles.metrics}>
            <div>
              <span>ACC</span>
              <strong>{hasRun ? formatPct(result.acc) : "?"}</strong>
            </div>
            <div>
              <span>平均遗忘</span>
              <strong>{hasRun ? formatPct(result.forget) : "?"}</strong>
            </div>
            <div>
              <span>BWT</span>
              <strong>
                {hasRun ? `${(result.bwt * 100).toFixed(0)} pt` : "?"}
              </strong>
            </div>
            <div>
              <span>对角平均</span>
              <strong>{hasRun ? formatPct(result.diagonal) : "?"}</strong>
            </div>
          </div>
          <div className={styles.tableWrap}>
            <table className={styles.matrix}>
              <thead>
                <tr>
                  <th>学完 \\ 测</th>
                  {result.matrix.map((_, index) => (
                    <th key={index}>T{index + 1}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.matrix.map((row, i) => (
                  <tr key={i}>
                    <th>T{i + 1}</th>
                    {row.map((value, j) => (
                      <td
                        key={j}
                        style={
                          {
                            "--cell": hasRun ? `${Math.round(value * 85)}%` : "0%",
                          } as CSSProperties
                        }
                      >
                        {hasRun ? (value === 0 && j > i ? "—" : formatPct(value)) : "·"}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className={styles.prediction}>
        <fieldset>
          <legend>预测 1：按遗忘和 BWT，这算持续学习吗？</legend>
          <div className={styles.choiceRow}>
            <button
              type="button"
              aria-pressed={clPrediction === "yes"}
              onClick={() => {
                setClPrediction("yes");
                invalidate();
              }}
            >
              算
            </button>
            <button
              type="button"
              aria-pressed={clPrediction === "no"}
              onClick={() => {
                setClPrediction("no");
                invalidate();
              }}
            >
              不算
            </button>
          </div>
        </fieldset>
        <fieldset>
          <legend>预测 2：只看 ACC，它有没有把差方法夸过关？</legend>
          <div className={styles.choiceRow}>
            <button
              type="button"
              aria-pressed={liarPrediction === "flatters"}
              onClick={() => {
                setLiarPrediction("flatters");
                invalidate();
              }}
            >
              ACC 在夸它
            </button>
            <button
              type="button"
              aria-pressed={liarPrediction === "honest"}
              onClick={() => {
                setLiarPrediction("honest");
                invalidate();
              }}
            >
              ACC 没有单独撒谎
            </button>
          </div>
        </fieldset>
        <button
          type="button"
          className={styles.run}
          disabled={!clPrediction || !liarPrediction}
          onClick={runLab}
        >
          计算指标
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
            ? "先判断算不算、谁在撒谎，再揭开矩阵。"
            : gatePassed
              ? result.isContinual
                ? "遗忘和 BWT 都还压得住。ACC 这时和它们同向。"
                : `ACC 是 ${formatPct(result.acc)}，看起来还行；平均遗忘 ${formatPct(result.forget)}，BWT ${(result.bwt * 100).toFixed(0)} pt，这是后任务专家。`
              : "持续学习要同时看遗忘和 BWT。只会最后一件事时，ACC 仍可能被后三个容易任务抬起来。"}
        </span>
      </div>
    </section>
  );
}
