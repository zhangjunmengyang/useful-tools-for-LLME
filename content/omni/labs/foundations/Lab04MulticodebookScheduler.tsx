"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import styles from "./Lab04MulticodebookScheduler.module.css";
import type { FoundationLabProps } from "./types";
import { initialNumber, initialString } from "./types";

const CODEBOOK_OPTIONS = [3, 4, 5, 6, 8] as const;
const FRAME_OPTIONS = [2, 3, 4, 5, 6, 7] as const;
const DELAY_OPTIONS = [1, 2] as const;
const TOPOLOGIES = ["diagonal", "parallel", "grouped"] as const;

type Topology = (typeof TOPOLOGIES)[number];
type CellKind = "bos" | "pad" | "idle" | "token";

type ScheduleCell = {
  kind: CellKind;
  label: string;
  frame?: number;
  codebook?: number;
};

type ScheduleColumn = {
  index: number;
  primary: string;
  secondary?: string;
  outerStep: number;
  innerStep: number | null;
  ariaLabel: string;
};

type Schedule = {
  columns: ScheduleColumn[];
  rows: ScheduleCell[][];
  rowMeta: string[];
  targetFrame: number;
  targetPosition: number;
  targetOuterStep: number;
  targetInnerStep: number | null;
  formula: string;
  gridLabel: string;
  cornerLabel: string;
};

const TOPOLOGY_COPY: Record<
  Topology,
  { label: string; description: string }
> = {
  diagonal: {
    label: "对角延迟",
    description: "diagonal · 每层向右错开 Δ 个自回归 step",
  },
  parallel: {
    label: "同帧并行",
    description: "same-frame parallel · 同帧各层一步发射",
  },
  grouped: {
    label: "两组深度",
    description: "grouped depth · 每帧两个 inner step",
  },
};

function allowedNumber(
  value: number,
  options: readonly number[],
  fallback: number,
) {
  return options.includes(value) ? value : fallback;
}

function codebookGroup(codebook: number, codebooks: number) {
  return codebook < Math.ceil(codebooks / 2) ? 0 : 1;
}

function buildSchedule(
  topology: Topology,
  codebooks: number,
  frames: number,
  delay: number,
): Schedule {
  const targetFrame = Math.min(2, frames - 1);

  if (topology === "parallel") {
    const columns = Array.from({ length: frames }, (_, frame) => ({
      index: frame,
      primary: String(frame),
      outerStep: frame,
      innerStep: null,
      ariaLabel: `autoregressive step ${frame}`,
    }));
    const rows = Array.from({ length: codebooks }, (_, codebook) =>
      columns.map(({ index: frame }) => ({
        kind: "token" as const,
        label: `F${frame}`,
        frame,
        codebook,
      })),
    );

    return {
      columns,
      rows,
      rowMeta: Array.from({ length: codebooks }, () => "同帧"),
      targetFrame,
      targetPosition: targetFrame,
      targetOuterStep: targetFrame,
      targetInnerStep: null,
      formula: "s(f, q) = f",
      gridLabel: "同帧并行 codebook 调度网格",
      cornerLabel: "q \\ step",
    };
  }

  if (topology === "grouped") {
    const columns = Array.from({ length: frames * 2 }, (_, index) => {
      const outerStep = Math.floor(index / 2);
      const innerStep = index % 2;
      return {
        index,
        primary: `F${outerStep}`,
        secondary: `inner ${innerStep}`,
        outerStep,
        innerStep,
        ariaLabel: `frame ${outerStep}, inner step ${innerStep}, schedule slot ${index}`,
      };
    });
    const rows = Array.from({ length: codebooks }, (_, codebook) => {
      const group = codebookGroup(codebook, codebooks);
      return columns.map(({ outerStep, innerStep }) =>
        innerStep === group
          ? {
              kind: "token" as const,
              label: `F${outerStep}`,
              frame: outerStep,
              codebook,
            }
          : { kind: "idle" as const, label: "空" },
      );
    });
    const targetInnerStep = codebookGroup(codebooks - 1, codebooks);

    return {
      columns,
      rows,
      rowMeta: Array.from(
        { length: codebooks },
        (_, codebook) => `G${codebookGroup(codebook, codebooks) + 1}`,
      ),
      targetFrame,
      targetPosition: targetFrame * 2 + targetInnerStep,
      targetOuterStep: targetFrame,
      targetInnerStep,
      formula: "slot(f, q) = 2f + g(q), g(q) = ⌊2q / Q⌋",
      gridLabel: "两组 inner depth codebook 调度网格",
      cornerLabel: "q \\ slot",
    };
  }

  const maxStep = frames - 1 + (codebooks - 1) * delay;
  const columns = Array.from({ length: maxStep + 1 }, (_, step) => ({
    index: step,
    primary: String(step),
    outerStep: step,
    innerStep: null,
    ariaLabel: `autoregressive step ${step}`,
  }));
  const rows = Array.from({ length: codebooks }, (_, codebook) =>
    columns.map(({ index: step }) => {
      const frame = step - codebook * delay;
      if (frame < 0) return { kind: "bos" as const, label: "BOS" };
      if (frame >= frames) return { kind: "pad" as const, label: "·" };
      return {
        kind: "token" as const,
        label: `F${frame}`,
        frame,
        codebook,
      };
    }),
  );

  return {
    columns,
    rows,
    rowMeta: Array.from(
      { length: codebooks },
      (_, codebook) => `+${codebook * delay}`,
    ),
    targetFrame,
    targetPosition: targetFrame + (codebooks - 1) * delay,
    targetOuterStep: targetFrame + (codebooks - 1) * delay,
    targetInnerStep: null,
    formula: "s(f, q) = f + q × Δ",
    gridLabel: "对角延迟多 codebook 调度网格",
    cornerLabel: "q \\ step",
  };
}

export function Lab04MulticodebookScheduler({
  onComplete,
  initialState,
}: FoundationLabProps) {
  const defaults = {
    codebooks: allowedNumber(
      initialNumber(initialState, "codebooks", 4),
      CODEBOOK_OPTIONS,
      4,
    ),
    frames: allowedNumber(
      initialNumber(initialState, "frames", 5),
      FRAME_OPTIONS,
      5,
    ),
    delay: allowedNumber(
      initialNumber(initialState, "delay", 1),
      DELAY_OPTIONS,
      1,
    ),
    topology: initialString(
      initialState,
      "topology",
      TOPOLOGIES,
      "diagonal",
    ),
  };
  const [codebooks, setCodebooks] = useState(defaults.codebooks);
  const [frames, setFrames] = useState(defaults.frames);
  const [delay, setDelay] = useState(defaults.delay);
  const [topology, setTopology] = useState<Topology>(defaults.topology);
  const [prediction, setPrediction] = useState("");
  const [started, setStarted] = useState(false);
  const [cursor, setCursor] = useState(-1);
  const [playing, setPlaying] = useState(false);
  const completedRef = useRef(false);

  const schedule = useMemo(
    () => buildSchedule(topology, codebooks, frames, delay),
    [codebooks, delay, frames, topology],
  );
  const lastPosition = schedule.columns.length - 1;
  const numericPrediction = Number(prediction);
  const predictionCorrect =
    prediction.trim() !== "" &&
    Number.isFinite(numericPrediction) &&
    numericPrediction === schedule.targetPosition;
  const gatePassed = started && cursor >= lastPosition && predictionCorrect;

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      setCursor((current) => {
        if (current >= lastPosition) {
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, 560);
    return () => window.clearInterval(timer);
  }, [lastPosition, playing]);

  useEffect(() => {
    if (gatePassed && !completedRef.current) {
      completedRef.current = true;
      onComplete?.({
        topology,
        codebooks,
        frames,
        delay: topology === "diagonal" ? delay : 0,
        targetFrame: schedule.targetFrame,
        targetPosition: schedule.targetPosition,
        targetStep: schedule.targetOuterStep,
        targetInnerStep: schedule.targetInnerStep,
        totalScheduleSlots: schedule.columns.length,
      });
    }
  }, [
    codebooks,
    delay,
    frames,
    gatePassed,
    onComplete,
    schedule.columns.length,
    schedule.targetFrame,
    schedule.targetInnerStep,
    schedule.targetOuterStep,
    schedule.targetPosition,
    topology,
  ]);

  function invalidate() {
    setStarted(false);
    setCursor(-1);
    setPlaying(false);
    completedRef.current = false;
  }

  function changeTopology(nextTopology: Topology) {
    setTopology(nextTopology);
    invalidate();
  }

  function start() {
    setStarted(true);
    setCursor(-1);
    setPlaying(false);
    completedRef.current = false;
  }

  function step() {
    setPlaying(false);
    setCursor((current) => Math.min(lastPosition, current + 1));
  }

  function reset() {
    setCodebooks(defaults.codebooks);
    setFrames(defaults.frames);
    setDelay(defaults.delay);
    setTopology(defaults.topology);
    setPrediction("");
    setStarted(false);
    setCursor(-1);
    setPlaying(false);
    completedRef.current = false;
  }

  const currentColumn = cursor < 0 ? null : schedule.columns[cursor];
  const currentTokens =
    cursor < 0
      ? []
      : schedule.rows.flatMap((row, codebook) => {
          const cell = row[cursor];
          return cell.kind === "token" ? [{ ...cell, codebook }] : [];
        });
  const isGrouped = topology === "grouped";
  const predictionUnit = isGrouped ? "slot" : "step";

  const legend =
    topology === "diagonal"
      ? [
          ["BOS", "较深层尚未轮到真实帧，用起始占位。"],
          ["F#", "同一步可携带不同 codebook 的不同原始帧。"],
          ["·", "该层已发完，用 padding 保持矩形 batch。"],
        ]
      : topology === "parallel"
        ? [
            ["F#", "同一列的所有层属于同一个原始帧。"],
            ["并行", "同帧 codebook 不增加层间自回归深度。"],
            ["step", "列号就是 autoregressive step。"],
          ]
        : [
            ["G1", "前一半 codebook 在 inner step 0 并行生成。"],
            ["G2", "后一半 codebook 在 inner step 1 并行生成。"],
            ["注意", "inner step 是同帧计算深度，不是 temporal latency。"],
          ];

  return (
    <section className={styles.lab} aria-labelledby="lab04-title">
      <header className={styles.header}>
        <div>
          <div className={styles.tags}>
            <span>调度沙盘</span>
            <span>预测后验证</span>
          </div>
          <h3 id="lab04-title">同一组音频码，三种生成拓扑有什么不同？</h3>
          <p>
            固定 RVQ 帧与层数，切换生成拓扑。先预测目标 token
            的位置，再逐格运行调度器，观察“时间步”和“层内深度”是否被混为一谈。
          </p>
        </div>
        <div className={styles.clock} aria-live="polite" aria-atomic="true">
          <span>{isGrouped ? "SCHEDULE SLOT" : "AUTOREGRESSIVE STEP"}</span>
          <strong>
            {cursor < 0 ? "未开始" : String(cursor).padStart(2, "0")}
            <i>/ {String(lastPosition).padStart(2, "0")}</i>
          </strong>
          {isGrouped && currentColumn ? (
            <small>
              temporal F{currentColumn.outerStep} · inner{" "}
              {currentColumn.innerStep}
            </small>
          ) : null}
        </div>
      </header>

      <fieldset className={styles.topologyPicker}>
        <legend>生成拓扑</legend>
        <div>
          {TOPOLOGIES.map((value) => (
            <label
              className={topology === value ? styles.selectedTopology : ""}
              key={value}
            >
              <input
                type="radio"
                name="lab04-topology"
                value={value}
                checked={topology === value}
                onChange={() => changeTopology(value)}
              />
              <span>
                <strong>{TOPOLOGY_COPY[value].label}</strong>
                <small>{TOPOLOGY_COPY[value].description}</small>
              </span>
            </label>
          ))}
        </div>
      </fieldset>

      <div className={styles.toolbar}>
        <label>
          <span>Codebooks</span>
          <select
            value={codebooks}
            onChange={(event) => {
              setCodebooks(Number(event.target.value));
              invalidate();
            }}
          >
            {CODEBOOK_OPTIONS.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>Frames</span>
          <select
            value={frames}
            onChange={(event) => {
              setFrames(Number(event.target.value));
              invalidate();
            }}
          >
            {FRAME_OPTIONS.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
        {topology === "diagonal" ? (
          <label>
            <span>Δ / layer</span>
            <select
              value={delay}
              onChange={(event) => {
                setDelay(Number(event.target.value));
                invalidate();
              }}
            >
              {DELAY_OPTIONS.map((value) => (
                <option key={value} value={value}>
                  {value} step
                </option>
              ))}
            </select>
          </label>
        ) : (
          <div className={styles.fixedControl}>
            <span>层间 temporal delay</span>
            <strong>0 step</strong>
          </div>
        )}
        <div className={styles.formula} aria-live="polite">
          <span>
            {isGrouped ? "调度坐标（q 从 0 开始）" : "确定性调度公式（q 从 0 开始）"}
          </span>
          <code>{schedule.formula}</code>
          {isGrouped ? (
            <small>
              g(q)=0 属于 G1，g(q)=1 属于 G2；slot 只用于逐格演示。
            </small>
          ) : null}
        </div>
        <button type="button" className={styles.reset} onClick={reset}>
          全部重置
        </button>
      </div>

      <div className={styles.challenge}>
        <div>
          <span>坐标预测</span>
          <strong>
            最后一层 Q{codebooks} 的 F{schedule.targetFrame} 会在哪个{" "}
            {predictionUnit} 出现？
          </strong>
          {isGrouped ? (
            <small>
              回答线性 schedule slot；它对应 (temporal frame, inner step)，不等于音频延迟。
            </small>
          ) : null}
        </div>
        <label htmlFor="lab04-prediction">
          {predictionUnit} =
          <input
            id="lab04-prediction"
            type="number"
            min="0"
            max={lastPosition}
            value={prediction}
            onChange={(event) => {
              setPrediction(event.target.value);
              invalidate();
            }}
            inputMode="numeric"
            aria-describedby="lab04-formula-hint"
          />
        </label>
        <button
          type="button"
          onClick={start}
          disabled={
            prediction.trim() === "" || !Number.isFinite(numericPrediction)
          }
        >
          锁定预测，生成网格
        </button>
      </div>
      <p id="lab04-formula-hint" className={styles.srOnly}>
        使用上方当前拓扑的调度公式计算答案。
      </p>

      <div className={styles.scheduler}>
        <div className={styles.gridViewport}>
          <div
            className={styles.grid}
            role="grid"
            aria-label={schedule.gridLabel}
            aria-rowcount={codebooks + 1}
            aria-colcount={schedule.columns.length + 1}
            style={
              {
                "--columns": schedule.columns.length,
              } as CSSProperties
            }
          >
            <div className={styles.corner} role="columnheader">
              {schedule.cornerLabel}
            </div>
            {schedule.columns.map((column) => (
              <div
                className={`${styles.stepHead} ${
                  cursor === column.index ? styles.currentHead : ""
                }`}
                role="columnheader"
                aria-label={column.ariaLabel}
                aria-current={cursor === column.index ? "step" : undefined}
                key={`head-${column.index}`}
              >
                <strong>{column.primary}</strong>
                {column.secondary ? <small>{column.secondary}</small> : null}
              </div>
            ))}
            {schedule.rows.map((row, codebook) => (
              <div className={styles.gridRow} role="row" key={codebook}>
                <div className={styles.rowHead} role="rowheader">
                  <strong>Q{codebook + 1}</strong>
                  <span>{schedule.rowMeta[codebook]}</span>
                </div>
                {row.map((cell, position) => {
                  const isTarget =
                    codebook === codebooks - 1 &&
                    cell.kind === "token" &&
                    cell.frame === schedule.targetFrame;
                  const emitted =
                    started && position <= cursor && cell.kind === "token";
                  const active = started && position === cursor;
                  const activeToken = active && cell.kind === "token";
                  const stateLabel = active
                    ? cell.kind === "token"
                      ? "当前正在执行"
                      : "当前 slot"
                    : emitted
                      ? "已执行"
                      : "未执行";
                  return (
                    <div
                      role="gridcell"
                      key={position}
                      aria-current={active ? "step" : undefined}
                      aria-label={`Q${codebook + 1}, ${schedule.columns[position].ariaLabel}: ${
                        cell.kind === "token"
                          ? `frame ${cell.frame}`
                          : cell.kind === "idle"
                            ? "本组空闲"
                            : cell.label
                      }，${stateLabel}${isTarget ? "，预测目标" : ""}`}
                      className={[
                        styles.cell,
                        styles[cell.kind],
                        emitted ? styles.emitted : "",
                        activeToken ? styles.active : "",
                        isTarget ? styles.target : "",
                      ].join(" ")}
                    >
                      <span>{started ? cell.label : "?"}</span>
                      {emitted && !activeToken ? (
                        <b className={styles.executionMark} aria-hidden="true">
                          ✓
                        </b>
                      ) : null}
                      {activeToken ? (
                        <b className={styles.currentMark} aria-hidden="true">
                          当前
                        </b>
                      ) : null}
                      {isTarget && started ? <i>目标</i> : null}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>

        <div className={styles.transport}>
          <div className={styles.now}>
            <span>{isGrouped ? "本 inner step 同时发射" : "本步同时发射"}</span>
            <div aria-live="polite" aria-atomic="true">
              {cursor < 0 ? (
                <em>等待执行</em>
              ) : currentTokens.length ? (
                currentTokens.map((token) => (
                  <b key={token.codebook}>
                    Q{token.codebook + 1}:F{token.frame}
                  </b>
                ))
              ) : (
                <em>{topology === "diagonal" ? "只有 BOS / PAD" : "本组空闲"}</em>
              )}
            </div>
          </div>
          <div className={styles.transportButtons}>
            <button
              type="button"
              disabled={!started || cursor >= lastPosition}
              onClick={step}
            >
              单步执行
            </button>
            <button
              type="button"
              className={styles.play}
              disabled={!started || cursor >= lastPosition}
              aria-pressed={playing}
              onClick={() => setPlaying((value) => !value)}
            >
              {playing ? "暂停" : "连续播放"}
            </button>
            <button
              type="button"
              disabled={!started}
              onClick={() => {
                setCursor(-1);
                setPlaying(false);
                completedRef.current = false;
              }}
            >
              回到起点
            </button>
          </div>
        </div>
      </div>

      <div className={styles.explain}>
        {legend.map(([term, explanation]) => (
          <div key={term}>
            <b>{term}</b>
            <span>{explanation}</span>
          </div>
        ))}
      </div>

      <div
        className={`${styles.gate} ${
          cursor >= lastPosition
            ? gatePassed
              ? styles.pass
              : styles.retry
            : ""
        }`}
        role="status"
        aria-live="polite"
      >
        <strong>
          {gatePassed
            ? "✓ 验收已通过"
            : cursor >= lastPosition
              ? "↺ 需要重算"
              : "完成验收"}
        </strong>
        <span>
          {!started
            ? "锁定预测后，执行完整调度表。"
            : cursor < lastPosition
              ? `还剩 ${lastPosition - cursor} 个 ${
                  isGrouped ? "schedule slot" : "autoregressive step"
                }。`
              : gatePassed
                ? isGrouped
                  ? `预测正确：Q${codebooks} 的 F${schedule.targetFrame} 位于 temporal F${schedule.targetOuterStep}、inner ${schedule.targetInnerStep}，线性 slot 为 ${schedule.targetPosition}。inner depth 不计作 temporal latency。`
                  : `预测正确：目标位于 autoregressive step ${schedule.targetPosition}。`
                : `调度已执行完，但预测不对。请用 ${schedule.formula} 重算。`}
        </span>
      </div>
    </section>
  );
}
