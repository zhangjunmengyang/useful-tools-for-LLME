"use client";

import { useMemo, useState, type CSSProperties } from "react";
import { Gate, LabFrame } from "./LabFrame";
import chrome from "./LabFrame.module.css";
import styles from "./Lesson19NestedClocks.module.css";
import type { AdvancedLabProps } from "./types";
import { booleanFrom, cosine, round } from "./labUtils";

const TOKEN_VECS: Record<string, [number, number]> = {
  春: [1, 0],
  江: [0.72, 0.69],
  花: [0.1, 1],
  月: [-0.7, 0.71],
  夜: [-1, 0.05],
  游: [0.2, -0.98],
};

const SEQ = [
  "春",
  "江",
  "花",
  "月",
  "夜",
  "春",
  "江",
  "花",
  "月",
  "夜",
  "游",
  "春",
  "江",
  "花",
  "月",
  "夜",
  "春",
  "江",
  "花",
  "月",
  "夜",
  "春",
  "江",
  "花",
  "月",
  "夜",
  "游",
  "春",
  "江",
  "花",
  "月",
  "夜",
];

type LostPred = "token" | "style" | "task";

function add(a: [number, number], b: [number, number]): [number, number] {
  return [a[0] + b[0], a[1] + b[1]];
}

function scale(a: [number, number], k: number): [number, number] {
  return [a[0] * k, a[1] * k];
}

function simulate(fastOn: boolean, slowOn: boolean, taskOn: boolean) {
  const zero: [number, number] = [0, 0];
  let fast = zero;
  let slow = zero;
  let task = zero;
  const seqLen = 8;

  SEQ.forEach((token, index) => {
    const vec = TOKEN_VECS[token];
    if (fastOn) fast = vec;
    if (slowOn && (index + 1) % seqLen === 0) {
      const start = index + 1 - seqLen;
      const chunk = SEQ.slice(start, index + 1).map((item) => TOKEN_VECS[item]);
      const sum = chunk.reduce((acc, item) => add(acc, item), zero);
      slow = scale(sum, 1 / seqLen);
    }
    if (taskOn && (index + 1) % 16 === 0) {
      task = index < 16 ? [1, 0] : [0, 1];
    }
  });

  const last = TOKEN_VECS[SEQ[SEQ.length - 1]];
  const lastChunk = SEQ.slice(SEQ.length - 8);
  const styleTarget = scale(
    lastChunk.reduce((acc, token) => add(acc, TOKEN_VECS[token]), zero),
    1 / 8,
  );
  const taskTarget: [number, number] = [0, 1];

  return {
    tokenScore: round(Math.max(0, cosine(fast, last)), 3),
    styleScore: round(Math.max(0, cosine(slow, styleTarget)), 3),
    taskScore: round(Math.max(0, cosine(task, taskTarget)), 3),
    fast,
    slow,
    task,
  };
}

export function Lesson19NestedClocks({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const [fastOn, setFastOn] = useState(booleanFrom(initialState, "fastOn", true));
  const [slowOn, setSlowOn] = useState(booleanFrom(initialState, "slowOn", true));
  const [taskOn, setTaskOn] = useState(booleanFrom(initialState, "taskOn", true));
  const [slowLoss, setSlowLoss] = useState<LostPred | null>(null);
  const [taskKeepsToken, setTaskKeepsToken] = useState<"yes" | "no" | null>(null);
  const [hasRun, setHasRun] = useState(false);

  const current = useMemo(
    () => simulate(fastOn, slowOn, taskOn),
    [fastOn, slowOn, taskOn],
  );
  const ifSlowOff = useMemo(() => simulate(true, false, true), []);
  const ifTaskOff = useMemo(() => simulate(true, true, false), []);
  const gatePassed =
    hasRun && slowLoss === "style" && taskKeepsToken === "yes";

  function invalidate() {
    setHasRun(false);
  }

  function run() {
    setHasRun(true);
    if (slowLoss === "style" && taskKeepsToken === "yes") {
      onComplete?.({
        fastOn,
        slowOn,
        taskOn,
        tokenScore: current.tokenScore,
        styleScore: current.styleScore,
        taskScore: current.taskScore,
      });
    }
  }

  function reset() {
    setFastOn(true);
    setSlowOn(true);
    setTaskOn(true);
    setSlowLoss(null);
    setTaskKeepsToken(null);
    setHasRun(false);
  }

  const meters: { label: string; value: number }[] = [
    { label: "当前 token", value: current.tokenScore },
    { label: "本篇风格", value: current.styleScore },
    { label: "任务身份", value: current.taskScore },
  ];

  return (
    <LabFrame
      lesson="19"
      title="嵌套钟表：停掉一层会丢什么"
      description="嵌套学习把快权重、慢权重和优化器看成不同时间尺度的记忆。快针每 token 走一格，慢针每段序列走一格，更慢针每个任务走一格。停掉某一层，对应信息的探针分数会掉到 0。"
      onReset={reset}
    >
      <div className={chrome.workbench}>
        <div className={chrome.controls}>
          {(
            [
              ["fastOn", "快针（token）", fastOn, setFastOn],
              ["slowOn", "慢针（序列）", slowOn, setSlowOn],
              ["taskOn", "更慢针（任务）", taskOn, setTaskOn],
            ] as const
          ).map(([key, label, value, setter]) => (
            <label key={key}>
              <span>{label}</span>
              <select
                value={value ? "on" : "off"}
                onChange={(event) => {
                  setter(event.target.value === "on");
                  invalidate();
                }}
              >
                <option value="on">走动</option>
                <option value="off">停住</option>
              </select>
            </label>
          ))}
          <div className={chrome.formula}>
            <code>W_fast ← 每 token</code>
            <code>W_slow ← 每 8 token</code>
            <code>W_task ← 每 16 token</code>
          </div>
        </div>

        <div className={chrome.panel}>
          <div className={styles.clocks} aria-hidden="true">
            {[
              { name: "快", on: fastOn, angle: fastOn ? 200 : 8 },
              { name: "慢", on: slowOn, angle: slowOn ? 110 : 8 },
              { name: "任务", on: taskOn, angle: taskOn ? 300 : 8 },
            ].map((clock) => (
              <div key={clock.name} className={styles.clock} data-on={clock.on}>
                <span className={styles.face}>
                  <i
                    className={styles.hand}
                    style={{ "--angle": `${clock.angle}deg` } as CSSProperties}
                  />
                </span>
                <small>{clock.name}</small>
              </div>
            ))}
          </div>
          <div className={styles.meters}>
            {meters.map((meter) => (
              <div key={meter.label}>
                <span>{meter.label}</span>
                <strong>{hasRun ? meter.value.toFixed(2) : "?"}</strong>
                <span className={chrome.track}>
                  <i
                    className={chrome.fill}
                    style={
                      {
                        "--fill": hasRun ? `${meter.value * 100}%` : "0%",
                      } as CSSProperties
                    }
                  />
                </span>
              </div>
            ))}
          </div>
          <p className={chrome.note}>
            {hasRun
              ? `对照：停慢针时风格探针 ${ifSlowOff.styleScore.toFixed(2)}（token 仍 ${ifSlowOff.tokenScore.toFixed(2)}）；停任务针时任务探针 ${ifTaskOff.taskScore.toFixed(2)}，token 仍 ${ifTaskOff.tokenScore.toFixed(2)}。完整 Hope 语言模型无官方训练配方，这里只跑两时间尺度的缩小记忆。`
              : "完整 Hope 语言模型无官方训练配方，这里只跑两时间尺度的缩小记忆。"}
          </p>
        </div>
      </div>

      <div className={chrome.prediction}>
        <fieldset>
          <legend>预测 1：停掉慢针，哪类信息丢了？</legend>
          <div className={chrome.choiceRow}>
            <button
              type="button"
              aria-pressed={slowLoss === "token"}
              onClick={() => {
                setSlowLoss("token");
                invalidate();
              }}
            >
              当前 token
            </button>
            <button
              type="button"
              aria-pressed={slowLoss === "style"}
              onClick={() => {
                setSlowLoss("style");
                invalidate();
              }}
            >
              本篇风格
            </button>
            <button
              type="button"
              aria-pressed={slowLoss === "task"}
              onClick={() => {
                setSlowLoss("task");
                invalidate();
              }}
            >
              任务身份
            </button>
          </div>
        </fieldset>
        <fieldset>
          <legend>预测 2：停掉任务针，当前 token 记忆还在吗？</legend>
          <div className={chrome.choiceRow}>
            <button
              type="button"
              aria-pressed={taskKeepsToken === "yes"}
              onClick={() => {
                setTaskKeepsToken("yes");
                invalidate();
              }}
            >
              还在（快针仍走）
            </button>
            <button
              type="button"
              aria-pressed={taskKeepsToken === "no"}
              onClick={() => {
                setTaskKeepsToken("no");
                invalidate();
              }}
            >
              一起丢了
            </button>
          </div>
        </fieldset>
        <button
          type="button"
          className={chrome.run}
          disabled={!slowLoss || !taskKeepsToken}
          onClick={run}
        >
          运行钟表
        </button>
      </div>

      <Gate ran={hasRun} passed={gatePassed}>
        {!hasRun
          ? "先判断停针损失，再看三支探针。"
          : gatePassed
            ? `当前设置：token ${current.tokenScore.toFixed(2)} / 风格 ${current.styleScore.toFixed(2)} / 任务 ${current.taskScore.toFixed(2)}。`
            : "慢针存的是序列均值（风格）；任务针存任务身份。快针独立更新，停任务针不会拆掉当前 token。"}
      </Gate>
    </LabFrame>
  );
}
