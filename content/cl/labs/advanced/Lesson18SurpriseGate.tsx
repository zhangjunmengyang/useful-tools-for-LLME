"use client";

import { useMemo, useState, type CSSProperties } from "react";
import { Gate, LabFrame } from "./LabFrame";
import chrome from "./LabFrame.module.css";
import styles from "./Lesson18SurpriseGate.module.css";
import type { AdvancedLabProps } from "./types";
import { numberFrom, round } from "./labUtils";

const STREAM = [
  "的",
  "的",
  "猫",
  "的",
  "的",
  "的",
  "量子纠缠",
  "的",
  "的",
  "猫",
  "的",
  "的",
  "稀有告警",
  "的",
  "的",
  "的",
] as const;

const PRIOR: Record<string, number> = {
  的: 0.62,
  猫: 0.09,
  量子纠缠: 0.015,
  稀有告警: 0.01,
};

function surpriseOf(token: string) {
  return -Math.log(PRIOR[token] ?? 0.05);
}

export function Lesson18SurpriseGate({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    threshold: numberFrom(initialState, "threshold", 2.8, 0.4, 5),
  };
  const [threshold, setThreshold] = useState(defaults.threshold);
  const [marked, setMarked] = useState<boolean[]>(() => STREAM.map(() => false));
  const [commonPred, setCommonPred] = useState<"write" | "skip" | null>(null);
  const [hasRun, setHasRun] = useState(false);

  const scored = useMemo(
    () =>
      STREAM.map((token, index) => {
        const surprise = surpriseOf(token);
        return {
          index,
          token,
          surprise: round(surprise, 3),
          write: surprise > threshold,
        };
      }),
    [threshold],
  );
  const written = scored.filter((item) => item.write).map((item) => item.index);
  const markedSet = marked
    .map((flag, index) => (flag ? index : -1))
    .filter((index) => index >= 0);
  const sameSet =
    written.length === markedSet.length &&
    written.every((index) => markedSet.includes(index));
  const deWritten = scored.some((item) => item.token === "的" && item.write);
  const gatePassed =
    hasRun && sameSet && commonPred === (deWritten ? "write" : "skip");

  function invalidate() {
    setHasRun(false);
  }

  function toggle(index: number) {
    setMarked((current) => current.map((flag, i) => (i === index ? !flag : flag)));
    invalidate();
  }

  function run() {
    setHasRun(true);
    const passed = sameSet && commonPred === (deWritten ? "write" : "skip");
    if (passed) {
      onComplete?.({
        threshold,
        written: written.map((index) => STREAM[index]),
        blackboard: scored
          .filter((item) => item.write)
          .map((item) => ({ token: item.token, surprise: item.surprise })),
      });
    }
  }

  function reset() {
    setThreshold(defaults.threshold);
    setMarked(STREAM.map(() => false));
    setCommonPred(null);
    setHasRun(false);
  }

  const maxSurprise = Math.max(...scored.map((item) => item.surprise));

  return (
    <LabFrame
      lesson="18"
      title="惊讶门：只有意外才进长期黑板"
      description="Titans 用惊讶（这里用 −log 先验概率，对应损失/梯度范数的缩小版）当写入门控。高频「的」几乎不写；偶发的「量子纠缠」「稀有告警」会留下。先在 token 上标记你认为该留下的，再运行。"
      onReset={reset}
    >
      <div className={chrome.workbench}>
        <div className={chrome.controls}>
          <label>
            <span>
              写入阈值 τ <strong>{threshold.toFixed(1)}</strong>
            </span>
            <input
              type="range"
              min="0.4"
              max="5"
              step="0.1"
              value={threshold}
              onChange={(event) => {
                setThreshold(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <p className={chrome.note}>
            点选序列里你认为该写入黑板的 token。对照无门控的滑动平均：每个 token 都会留下一点，黑板会被「的」填满。
          </p>
          <div className={chrome.formula}>
            <code>s_t = −log p_prior(x_t)</code>
            <code>write if s_t &gt; τ</code>
            <code>p(的)=0.62，p(稀有告警)=0.01</code>
          </div>
        </div>

        <div className={chrome.panel}>
          <div className={chrome.metrics}>
            <div className={chrome.metric}>
              <span>写入条数</span>
              <strong>{hasRun ? written.length : "?"}</strong>
            </div>
            <div className={chrome.metric}>
              <span>你的标记</span>
              <strong>{markedSet.length}</strong>
            </div>
            <div className={chrome.metric}>
              <span>集合一致</span>
              <strong>{hasRun ? (sameSet ? "是" : "否") : "?"}</strong>
            </div>
          </div>
          <div className={styles.stream} aria-label="token 流">
            {scored.map((item) => (
              <button
                type="button"
                key={item.index}
                aria-pressed={marked[item.index]}
                data-write={hasRun && item.write ? "true" : "false"}
                onClick={() => toggle(item.index)}
              >
                <b>{item.token}</b>
                <span className={styles.bar}>
                  <i
                    style={
                      {
                        "--fill": hasRun
                          ? `${(item.surprise / maxSurprise) * 100}%`
                          : "0%",
                      } as CSSProperties
                    }
                  />
                </span>
              </button>
            ))}
          </div>
          <div className={styles.board}>
            <strong>长期黑板</strong>
            <p>
              {hasRun
                ? scored
                    .filter((item) => item.write)
                    .map((item) => `${item.token} (${item.surprise.toFixed(2)})`)
                    .join("、") || "（空）"
                : "待运行"}
            </p>
          </div>
        </div>
      </div>

      <div className={chrome.prediction}>
        <fieldset>
          <legend>预测 1：用标记表示哪些 token 会进黑板（再运行对照）</legend>
          <p className={chrome.note}>在上方序列里点选。当前阈值下，集合必须和模型写入一致。</p>
        </fieldset>
        <fieldset>
          <legend>预测 2：当前阈值下，高频「的」会写入吗？</legend>
          <div className={chrome.choiceRow}>
            <button
              type="button"
              aria-pressed={commonPred === "write"}
              onClick={() => {
                setCommonPred("write");
                invalidate();
              }}
            >
              会留下「的」
            </button>
            <button
              type="button"
              aria-pressed={commonPred === "skip"}
              onClick={() => {
                setCommonPred("skip");
                invalidate();
              }}
            >
              不会留下「的」
            </button>
          </div>
        </fieldset>
        <button
          type="button"
          className={chrome.run}
          disabled={!commonPred}
          onClick={run}
        >
          运行惊讶门
        </button>
      </div>

      <Gate ran={hasRun} passed={gatePassed}>
        {!hasRun
          ? "先标记该留下的 token，并判断「的」会不会进黑板。"
          : gatePassed
            ? `写入 ${written.length} 条，与你的标记一致。这是课内缩小门控，不是 Titans 语言模型分数。`
            : `模型写入了：${scored
                .filter((item) => item.write)
                .map((item) => item.token)
                .join("、") || "（空）"}。s=−log p，阈值越高，只剩更稀有的 token。`}
      </Gate>
    </LabFrame>
  );
}
