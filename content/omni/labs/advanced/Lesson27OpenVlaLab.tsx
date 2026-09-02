"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import {
  numberFrom,
  round,
  stringFrom,
  useCompletionGate,
} from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson27OpenVlaLab.module.css";

type DecodeMode = "serial_ce" | "parallel_l1";
type ReportMode = "average" | "suites";

const SUITES = ["spatial", "object", "goal", "long"] as const;
const SUITE_LABEL: Record<(typeof SUITES)[number], string> = {
  spatial: "Spatial",
  object: "Object",
  goal: "Goal",
  long: "Long",
};

const BASE_SUCCESS: Record<DecodeMode, Record<(typeof SUITES)[number], number>> =
  {
    serial_ce: { spatial: 0.72, object: 0.78, goal: 0.7, long: 0.41 },
    parallel_l1: { spatial: 0.86, object: 0.89, goal: 0.84, long: 0.62 },
  };

const OCCLUSION_WEIGHT: Record<(typeof SUITES)[number], number> = {
  spatial: 0.52,
  object: 0.48,
  goal: 0.22,
  long: 0.34,
};

function clamp01(value: number) {
  return Math.max(0.02, Math.min(0.98, value));
}

function latencyMs(mode: DecodeMode, chunk: number) {
  const prefix = 50;
  const decode = 27;
  if (mode === "serial_ce") {
    return prefix + 7 * chunk * decode;
  }
  const stretch = 1 + 0.17 * ((chunk - 1) / 7);
  return prefix + decode * stretch;
}

function suiteSuccess(
  mode: DecodeMode,
  chunk: number,
  occlusion: number,
): Record<(typeof SUITES)[number], number> {
  const boost = mode === "parallel_l1" ? 0.035 : 0.015;
  return Object.fromEntries(
    SUITES.map((suite) => {
      const chunkGain = suite === "long" ? boost * (chunk - 1) : boost * 0.25 * (chunk - 1);
      const raw =
        BASE_SUCCESS[mode][suite] +
        chunkGain -
        occlusion * OCCLUSION_WEIGHT[suite];
      return [suite, clamp01(raw)];
    }),
  ) as Record<(typeof SUITES)[number], number>;
}

export function Lesson27OpenVlaLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    decode: (stringFrom(initialState, "decode", "serial_ce") ===
    "parallel_l1"
      ? "parallel_l1"
      : "serial_ce") as DecodeMode,
    chunk: numberFrom(initialState, "chunk", 1, 1, 8),
    occlusion: numberFrom(initialState, "occlusion", 0, 0, 0.7),
    report: (stringFrom(initialState, "report", "average") === "suites"
      ? "suites"
      : "average") as ReportMode,
    prediction: stringFrom(initialState, "prediction", ""),
  };
  const chunkDefault = [1, 4, 8].includes(defaults.chunk)
    ? defaults.chunk
    : 1;
  const occlusionDefault = [0, 0.35, 0.7].includes(defaults.occlusion)
    ? defaults.occlusion
    : 0;

  const [decode, setDecode] = useState<DecodeMode>(defaults.decode);
  const [chunk, setChunk] = useState(chunkDefault);
  const [occlusion, setOcclusion] = useState(occlusionDefault);
  const [report, setReport] = useState<ReportMode>(defaults.report);
  const [prediction, setPrediction] = useState(defaults.prediction);
  const [ran, setRan] = useState(false);
  const [sawAverage, setSawAverage] = useState(false);

  const simulation = useMemo(() => {
    const serialH1 = latencyMs("serial_ce", 1);
    const parallelH1 = latencyMs("parallel_l1", 1);
    const current = latencyMs(decode, chunk);
    const compare = latencyMs(
      decode === "serial_ce" ? "parallel_l1" : "serial_ce",
      chunk,
    );
    const suites = suiteSuccess(decode, chunk, occlusion);
    const average =
      SUITES.reduce((sum, name) => sum + suites[name], 0) / SUITES.length;
    const serialSteps = 7 * chunk;
    const parallelSteps = 1;
    return {
      serialH1,
      parallelH1,
      current,
      compare,
      suites,
      average,
      serialSteps,
      parallelSteps,
      currentHz: round((1000 * chunk) / current, 1),
    };
  }, [chunk, decode, occlusion]);

  const passed =
    ran &&
    prediction === "serial_slower" &&
    simulation.serialH1 > simulation.parallelH1 &&
    sawAverage &&
    report === "suites";

  const completion = useMemo(
    () => ({
      lessonId: 27,
      decode,
      chunk,
      occlusion,
      report,
      serialLatencyH1: round(simulation.serialH1, 1),
      parallelLatencyH1: round(simulation.parallelH1, 1),
      averageSuccess: round(simulation.average, 3),
      longSuccess: round(simulation.suites.long, 3),
    }),
    [chunk, decode, occlusion, report, simulation],
  );
  useCompletionGate(passed, onComplete, completion);

  function invalidate() {
    setRan(false);
  }

  function reset() {
    setDecode(defaults.decode);
    setChunk(chunkDefault);
    setOcclusion(occlusionDefault);
    setReport(defaults.report);
    setPrediction("");
    setRan(false);
    setSawAverage(false);
  }

  const maxBar = Math.max(
    simulation.current,
    simulation.compare,
    simulation.serialH1,
  );
  const compareLabel =
    decode === "serial_ce" ? "并行 L1 对照" : "串行 CE 对照";

  return (
    <LabFrame
      lesson="27"
      title="串行 CE、并行 L1 与套件拆桶"
      description="切换解码方式和 chunk，先预测延迟再看条形图。注入视觉遮挡后比较玩具成功率。只报四套件平均时界面标红。教学模拟，不是模型输出。"
    >
      <div className={styles.workspace}>
        <form
          className={styles.controls}
          onSubmit={(event) => event.preventDefault()}
        >
          <h3>解码控制台</h3>
          <fieldset>
            <legend>解码与损失</legend>
            <div className={styles.choiceRow}>
              {(
                [
                  ["serial_ce", "串行 CE"],
                  ["parallel_l1", "并行 L1"],
                ] as const
              ).map(([value, label]) => (
                <label key={value}>
                  <input
                    type="radio"
                    name="decode-mode"
                    value={value}
                    checked={decode === value}
                    onChange={() => {
                      setDecode(value);
                      invalidate();
                    }}
                  />
                  <span>{label}</span>
                </label>
              ))}
            </div>
          </fieldset>
          <fieldset>
            <legend>动作 chunk H</legend>
            <div className={styles.choiceRow}>
              {([1, 4, 8] as const).map((value) => (
                <label key={value}>
                  <input
                    type="radio"
                    name="chunk-h"
                    value={value}
                    checked={chunk === value}
                    onChange={() => {
                      setChunk(value);
                      invalidate();
                    }}
                  />
                  <span>H={value}</span>
                </label>
              ))}
            </div>
          </fieldset>
          <label>
            <span>
              视觉遮挡
              <output>{Math.round(occlusion * 100)}%</output>
            </span>
            <input
              type="range"
              min="0"
              max="0.7"
              step="0.35"
              value={occlusion}
              onChange={(event) => {
                setOcclusion(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <fieldset>
            <legend>成功率怎么报</legend>
            <div className={styles.choiceRow}>
              {(
                [
                  ["average", "只报平均"],
                  ["suites", "拆开四套件"],
                ] as const
              ).map(([value, label]) => (
                <label key={value}>
                  <input
                    type="radio"
                    name="report-mode"
                    value={value}
                    checked={report === value}
                    onChange={() => {
                      const next = value as ReportMode;
                      setReport(next);
                      if (next === "average") setSawAverage(true);
                    }}
                  />
                  <span>{label}</span>
                </label>
              ))}
            </div>
          </fieldset>
        </form>

        <div className={styles.stage}>
          <div className={styles.formula}>
            <span>
              N_serial = 7H, N_parallel = 1, t = t_prefix + N · t_decode
            </span>
            <strong>
              {decode === "serial_ce"
                ? `N = 7 × ${chunk} = ${simulation.serialSteps}`
                : `N = ${simulation.parallelSteps}（一次吐 ${chunk}×7）`}
            </strong>
          </div>
          <div className={styles.scene} aria-label="被遮挡的桌面场景">
            <div
              className={styles.occlusion}
              style={{ opacity: occlusion }}
            />
            <p>第三人称桌面：篮子在右侧，字母汤罐头在近处。</p>
            <small>
              指令 pick up the alphabet soup / 遮挡只改玩具成功率，不改延迟公式
            </small>
          </div>
          <div className={styles.delayList} aria-label="延迟条">
            <div className={styles.delayRow}>
              <span>当前配置延迟</span>
              <div className={styles.delayTrack}>
                <div
                  className={styles.delayFill}
                  style={{
                    width: ran
                      ? `${(simulation.current / maxBar) * 100}%`
                      : "0%",
                  }}
                />
              </div>
              <b>{ran ? `${Math.round(simulation.current)} ms` : "—"}</b>
            </div>
            <div className={styles.delayRow}>
              <span>{compareLabel}</span>
              <div className={styles.delayTrack}>
                <div
                  className={`${styles.delayFill} ${styles.delayFillCompare}`}
                  style={{
                    width: ran
                      ? `${(simulation.compare / maxBar) * 100}%`
                      : "0%",
                  }}
                />
              </div>
              <b>{ran ? `${Math.round(simulation.compare)} ms` : "—"}</b>
            </div>
            <div className={styles.delayRow}>
              <span>串行 7 步 (H=1)</span>
              <div className={styles.delayTrack}>
                <div
                  className={styles.delayFill}
                  style={{
                    width: ran
                      ? `${(simulation.serialH1 / maxBar) * 100}%`
                      : "0%",
                  }}
                />
              </div>
              <b>{ran ? `${Math.round(simulation.serialH1)} ms` : "—"}</b>
            </div>
            <div className={styles.delayRow}>
              <span>并行 1 步 (H=1)</span>
              <div className={styles.delayTrack}>
                <div
                  className={`${styles.delayFill} ${styles.delayFillCompare}`}
                  style={{
                    width: ran
                      ? `${(simulation.parallelH1 / maxBar) * 100}%`
                      : "0%",
                  }}
                />
              </div>
              <b>{ran ? `${Math.round(simulation.parallelH1)} ms` : "—"}</b>
            </div>
          </div>
          {report === "average" ? (
            <p className={styles.flag} role="alert">
              只报平均成功率会把套件差距藏进去。当前玩具平均{" "}
              {ran ? `${Math.round(simulation.average * 100)}%` : "（先运行）"}
              ，LIBERO-Long 夹具明显更低。必须拆套件再读数。
            </p>
          ) : (
            <dl className={styles.suites}>
              {SUITES.map((name) => (
                <div className={styles.suite} key={name}>
                  <dt>{SUITE_LABEL[name]}</dt>
                  <dd>
                    {ran
                      ? `${Math.round(simulation.suites[name] * 100)}%`
                      : "—"}
                  </dd>
                </div>
              ))}
            </dl>
          )}
          <dl className={styles.metrics}>
            <div>
              <dt>串行步数</dt>
              <dd>{decode === "serial_ce" ? simulation.serialSteps : "1"}</dd>
            </div>
            <div>
              <dt>玩具吞吐</dt>
              <dd>{ran ? `${simulation.currentHz} Hz` : "—"}</dd>
            </div>
            <div>
              <dt>H=1 延迟差</dt>
              <dd>
                {ran
                  ? `${Math.round(simulation.serialH1 - simulation.parallelH1)} ms`
                  : "—"}
              </dd>
            </div>
          </dl>
        </div>
      </div>

      <div className={styles.predict}>
        <fieldset>
          <legend>
            先预测：H=1 时，串行 7 步 CE 的延迟和并行 1 步 L1 比，谁更高？
          </legend>
          {(
            [
              ["serial_slower", "串行 7 步更高"],
              ["parallel_slower", "并行 1 步更高"],
              ["same", "两者相同"],
            ] as const
          ).map(([value, label]) => (
            <label key={value}>
              <input
                type="radio"
                name="openvla-prediction"
                value={value}
                checked={prediction === value}
                onChange={() => {
                  setPrediction(value);
                  setRan(false);
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
            onClick={() => {
              setRan(true);
              if (report === "average") setSawAverage(true);
            }}
          >
            运行对照
          </button>
        </div>
      </div>
      {ran && prediction !== "serial_slower" && (
        <p className={styles.feedback}>
          串行要连跑 7 次解码，并行把 7 维放进一次前向。先改预测，再看 H=1
          的两条延迟条。
        </p>
      )}
      {ran && prediction === "serial_slower" && report === "average" && (
        <p className={styles.feedback}>
          延迟对照已揭晓。把成功率改成“拆开四套件”，不要只留一个平均数。
        </p>
      )}
      <Gate passed={passed}>
        先选对“串行 7 步延迟更高”，确认 H=1 串行条长于并行条；再触发只报平均的红色警告，并改回按套件报告。
      </Gate>
    </LabFrame>
  );
}
