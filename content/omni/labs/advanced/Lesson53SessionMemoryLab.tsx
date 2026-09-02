"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import {
  numberFrom,
  stringFrom,
  useCompletionGate,
} from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson53SessionMemoryLab.module.css";

type PolicyId = "summary" | "pixels" | "hybrid";
type Prediction =
  | "both-ok"
  | "summary-wrong-pixel-blows"
  | "summary-ok-pixel-blows"
  | "summary-wrong-pixel-ok";

type PolicyRow = {
  id: PolicyId;
  label: string;
  answer: "red" | "blue" | "over";
  bytes: number;
  over: boolean;
  correct: boolean;
};

const QUERY = "杯子现在是什么颜色？";
const TRUE_COLOR = "blue";
const HEADER_BYTES = 64;
const BOX_BYTES = 16;
const SUMMARY_BYTES = 36;
const CHANNELS = 3;

const PREDICTION_OPTIONS: { value: Prediction; label: string }[] = [
  {
    value: "summary-wrong-pixel-blows",
    label: "只存摘要会答错颜色；只存原图像素会超预算",
  },
  {
    value: "both-ok",
    label: "摘要和原图像素都能答对，也都不超预算",
  },
  {
    value: "summary-ok-pixel-blows",
    label: "只存摘要能答对颜色，只存原图像素会超预算",
  },
  {
    value: "summary-wrong-pixel-ok",
    label: "只存摘要会答错颜色，只存原图像素不超预算",
  },
];

const POLICY_LABEL: Record<PolicyId, string> = {
  summary: "只存摘要",
  pixels: "只存原图像素",
  hybrid: "摘要加像素，过期删图",
};

function pixelBytes(side: number) {
  return side * side * CHANNELS;
}

function recordBytes(side: number, withPixels: boolean) {
  return (
    HEADER_BYTES +
    SUMMARY_BYTES +
    BOX_BYTES +
    (withPixels ? pixelBytes(side) : 0)
  );
}

function evaluate(side: number, extras: number, budget: number) {
  const recordCount = 2 + extras;
  const oneFull = recordBytes(side, true);
  const oneText = recordBytes(side, false);
  const allPixels = recordCount * oneFull;
  const keptToday = oneFull + (recordCount - 1) * oneText;

  const summary: PolicyRow = {
    id: "summary",
    label: POLICY_LABEL.summary,
    answer: "red",
    bytes: recordCount * oneText,
    over: recordCount * oneText > budget,
    correct: false,
  };
  const pixels: PolicyRow = {
    id: "pixels",
    label: POLICY_LABEL.pixels,
    answer: allPixels > budget ? "over" : "blue",
    bytes: allPixels,
    over: allPixels > budget,
    correct: allPixels <= budget,
  };
  const hybrid: PolicyRow = {
    id: "hybrid",
    label: POLICY_LABEL.hybrid,
    answer: keptToday > budget ? "over" : "blue",
    bytes: keptToday,
    over: keptToday > budget,
    correct: keptToday <= budget,
  };
  return { summary, pixels, hybrid, recordCount };
}

function answerLabel(answer: PolicyRow["answer"]) {
  if (answer === "red") return "红";
  if (answer === "blue") return "蓝";
  return "超预算拒答";
}

export function Lesson53SessionMemoryLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    side: numberFrom(initialState, "side", 64, 32, 96),
    extras: numberFrom(initialState, "extras", 1, 1, 4),
    budget: numberFrom(initialState, "budget", 16384, 8192, 49152),
    policy: stringFrom(initialState, "policy", "summary") as PolicyId,
    prediction: stringFrom(initialState, "prediction", "") as Prediction | "",
  };
  const [side, setSide] = useState(Math.round(defaults.side));
  const [extras, setExtras] = useState(Math.round(defaults.extras));
  const [budget, setBudget] = useState(Math.round(defaults.budget));
  const [policy, setPolicy] = useState<PolicyId>(
    ["summary", "pixels", "hybrid"].includes(defaults.policy)
      ? defaults.policy
      : "summary",
  );
  const [prediction, setPrediction] = useState<Prediction | "">(
    defaults.prediction,
  );
  const [ran, setRan] = useState(false);

  const rows = useMemo(
    () => evaluate(side, extras, budget),
    [budget, extras, side],
  );
  const inspected = rows[policy];
  const summaryWrong = !rows.summary.correct && rows.summary.answer === "red";
  const pixelBlows = rows.pixels.over;
  const hybridWorks = rows.hybrid.correct;

  const passed =
    ran &&
    prediction === "summary-wrong-pixel-blows" &&
    summaryWrong &&
    pixelBlows;

  const completion = useMemo(
    () => ({
      lessonId: 53,
      side,
      extras,
      budget,
      policy,
      prediction,
      summaryAnswer: rows.summary.answer,
      pixelAnswer: rows.pixels.answer,
      hybridAnswer: rows.hybrid.answer,
      summaryBytes: rows.summary.bytes,
      pixelBytes: rows.pixels.bytes,
      hybridBytes: rows.hybrid.bytes,
      pixelOver: rows.pixels.over,
      hybridOver: rows.hybrid.over,
      recordCount: rows.recordCount,
      trueColor: TRUE_COLOR,
    }),
    [budget, extras, policy, prediction, rows, side],
  );
  useCompletionGate(passed, onComplete, completion);

  function reset() {
    setSide(Math.round(defaults.side));
    setExtras(Math.round(defaults.extras));
    setBudget(Math.round(defaults.budget));
    setPolicy("summary");
    setPrediction("");
    setRan(false);
  }

  return (
    <LabFrame
      lesson="53"
      title="隔天杯子：先选 payload，再看颜色和账单"
      description="教学模拟，不是模型输出。昨日红杯子，今日换成蓝的。先预测只存摘要和只存原图像素会怎样，再揭晓三种策略的答案和字节。"
    >
      <div className={styles.workspace}>
        <aside className={styles.controls}>
          <h3>记忆设置</h3>
          <fieldset>
            <legend>当前查看的策略</legend>
            {(["summary", "pixels", "hybrid"] as PolicyId[]).map((id) => (
              <label key={id}>
                <input
                  type="radio"
                  name="lesson53-policy"
                  checked={policy === id}
                  onChange={() => {
                    setPolicy(id);
                    setRan(false);
                  }}
                />
                {POLICY_LABEL[id]}
              </label>
            ))}
          </fieldset>
          <label>
            <span>
              图像边长
              <output>{side}</output>
            </span>
            <input
              type="range"
              min={32}
              max={96}
              step={16}
              value={side}
              onChange={(event) => {
                setSide(Number(event.target.value));
                setRan(false);
              }}
            />
          </label>
          <label>
            <span>
              额外旧记录
              <output>{extras}</output>
            </span>
            <input
              type="range"
              min={1}
              max={4}
              step={1}
              value={extras}
              onChange={(event) => {
                setExtras(Number(event.target.value));
                setRan(false);
              }}
            />
          </label>
          <label>
            <span>
              字节上限
              <output>{budget}</output>
            </span>
            <input
              type="range"
              min={8192}
              max={49152}
              step={2048}
              value={budget}
              onChange={(event) => {
                setBudget(Number(event.target.value));
                setRan(false);
              }}
            />
          </label>
          <p className={styles.note}>
            每条记录固定表头 {HEADER_BYTES}、摘要 {SUMMARY_BYTES}、框{" "}
            {BOX_BYTES}。像素按边长平方乘 {CHANNELS}。默认 64 边长、1
            条额外记录、上限 16384，对应 CPU 夹具。
          </p>
        </aside>
        <div className={styles.stage}>
          <p className={styles.query}>
            查询：<strong>{QUERY}</strong>{" "}
            真值已写入夹具，揭晓前不显示策略答了什么、花了多少字节。
          </p>
          <div className={styles.days}>
            <article className={styles.day}>
              <h4>昨日会话</h4>
              <div className={styles.scene} aria-hidden="true">
                <span className={`${styles.cup} ${styles.cupRed}`}>
                  <span className={styles.handle} />
                </span>
                <span className={styles.tableTop} />
              </div>
              <p>写入“桌上有一只红色杯子”，并可选留下原图。</p>
            </article>
            <article className={styles.day}>
              <h4>今日会话</h4>
              <div className={styles.scene} aria-hidden="true">
                <span className={`${styles.cup} ${styles.cupBlue}`}>
                  <span className={styles.handle} />
                </span>
                <span className={styles.tableTop} />
              </div>
              <p>世界换成蓝杯子。摘要若不可改写，仍会读到昨日的红。</p>
            </article>
          </div>
          <div className={styles.legend}>
            <span>
              <i className={styles.legendRed} />
              昨日红杯子
            </span>
            <span>
              <i className={styles.legendBlue} />
              今日蓝杯子
            </span>
          </div>
          <dl className={styles.metrics}>
            <div>
              <dt>当前策略答案</dt>
              <dd>{ran ? answerLabel(inspected.answer) : "—"}</dd>
            </div>
            <div>
              <dt>当前策略字节</dt>
              <dd>{ran ? inspected.bytes : "—"}</dd>
            </div>
            <div>
              <dt>是否超预算</dt>
              <dd>{ran ? (inspected.over ? "超" : "否") : "—"}</dd>
            </div>
          </dl>
          {ran ? (
            <div className={styles.table}>
              <header>
                <span>策略</span>
                <span>答案</span>
                <span>字节</span>
                <span>判定</span>
              </header>
              {([rows.summary, rows.pixels, rows.hybrid] as const).map(
                (row) => (
                  <article key={row.id}>
                    <span>{row.label}</span>
                    <span className={row.correct ? styles.hit : styles.miss}>
                      {answerLabel(row.answer)}
                    </span>
                    <span className={row.over ? styles.miss : styles.hit}>
                      {row.bytes}
                    </span>
                    <span>
                      {row.correct
                        ? "颜色对且未超"
                        : row.over
                          ? "超预算"
                          : "颜色错"}
                    </span>
                  </article>
                ),
              )}
            </div>
          ) : null}
          <div className={styles.predict}>
            <fieldset>
              <legend>揭晓前预测：哪一种说法成立？</legend>
              {PREDICTION_OPTIONS.map((option) => (
                <label key={option.value}>
                  <input
                    type="radio"
                    name="lesson53-prediction"
                    checked={prediction === option.value}
                    onChange={() => {
                      setPrediction(option.value);
                      setRan(false);
                    }}
                  />
                  <span>{option.label}</span>
                </label>
              ))}
            </fieldset>
            <div className={styles.actions}>
              <button className={styles.reset} type="button" onClick={reset}>
                重置
              </button>
              <button
                className={styles.run}
                type="button"
                disabled={!prediction}
                onClick={() => setRan(true)}
              >
                揭晓
              </button>
            </div>
          </div>
          {!prediction ? (
            <p className={styles.feedback}>先选一个预测，才能揭晓数字。</p>
          ) : null}
          {ran ? (
            <p className={styles.reveal}>
              真值是蓝色。只存摘要读到昨日“红色杯子”，答红。
              {rows.recordCount} 条 {side}×{side} 原图共 {rows.pixels.bytes}{" "}
              字节
              {pixelBlows
                ? `，超过上限 ${budget}。`
                : `，当前上限 ${budget} 仍装得下，把边长调大或把上限调低才能打出爆预算。`}
              混合策略改写今日摘要
              {hybridWorks
                ? "，并删掉旧像素，颜色对且未超。"
                : "，但今日这一张图仍然装不进上限。"}
            </p>
          ) : null}
        </div>
      </div>
      <Gate passed={passed}>
        {passed
          ? "已先预测再揭晓：只存摘要答错颜色，只存原图像素超过字节上限。"
          : "先提交预测，再调参数揭晓。必须同时看到摘要答红、原图像素超预算。"}
      </Gate>
    </LabFrame>
  );
}
