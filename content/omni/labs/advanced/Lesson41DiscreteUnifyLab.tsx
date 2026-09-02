"use client";

import { useEffect, useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import { numberFrom, stringFrom, useCompletionGate } from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson41DiscreteUnifyLab.module.css";

type MaskMode = "understand" | "generate";

const SIDE = 4;
const IMAGE = SIDE * SIDE;
const PROMPT = ["画", "一个", "L"];
const L_CELLS = new Set([0, 4, 8, 12, 13, 14, 15]);

function cellFill(index: number) {
  return L_CELLS.has(index) ? "#c2573a" : "#d9e2dc";
}

function understandAllows(query: number, key: number) {
  const queryVisual = query < IMAGE;
  const keyVisual = key < IMAGE;
  if (queryVisual) return keyVisual;
  if (keyVisual) return true;
  return key <= query;
}

function generateAllows(query: number, key: number) {
  const queryVisual = query >= PROMPT.length;
  const keyVisual = key >= PROMPT.length;
  if (!queryVisual) return !keyVisual && key <= query;
  if (!keyVisual) return true;
  return key <= query;
}

function imageQueryIndex(mode: MaskMode, cell: number) {
  return mode === "understand" ? cell : PROMPT.length + cell;
}

function imageKeyIndex(mode: MaskMode, cell: number) {
  return mode === "understand" ? cell : PROMPT.length + cell;
}

export function Lesson41DiscreteUnifyLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    query: numberFrom(initialState, "query", 5, 0, IMAGE - 2),
    mode: stringFrom(initialState, "mode", "understand") as MaskMode,
    prediction: stringFrom(initialState, "prediction", ""),
  };
  const [query, setQuery] = useState(
    defaults.query >= 0 && defaults.query <= IMAGE - 2 ? defaults.query : 5,
  );
  const [mode, setMode] = useState<MaskMode>(
    defaults.mode === "generate" ? "generate" : "understand",
  );
  const [prediction, setPrediction] = useState(defaults.prediction);
  const [revealed, setRevealed] = useState(false);
  const [seenUnderstand, setSeenUnderstand] = useState(false);
  const [seenGenerate, setSeenGenerate] = useState(false);

  const calculation = useMemo(() => {
    const understandQuery = imageQueryIndex("understand", query);
    const generateQuery = imageQueryIndex("generate", query);
    const understandVisible: number[] = [];
    const generateVisible: number[] = [];
    for (let cell = 0; cell < IMAGE; cell += 1) {
      if (understandAllows(understandQuery, imageKeyIndex("understand", cell))) {
        understandVisible.push(cell);
      }
      if (generateAllows(generateQuery, imageKeyIndex("generate", cell))) {
        generateVisible.push(cell);
      }
    }
    const future = Array.from({ length: IMAGE }, (_, cell) => cell).filter(
      (cell) => cell > query,
    );
    const understandFuture = future.filter((cell) =>
      understandVisible.includes(cell),
    );
    const generateFuture = future.filter((cell) => generateVisible.includes(cell));
    const visible = mode === "understand" ? understandVisible : generateVisible;
    const futureVisible =
      mode === "understand" ? understandFuture : generateFuture;
    return {
      understandVisible,
      generateVisible,
      understandFuture,
      generateFuture,
      visible,
      futureVisible,
      futureCount: future.length,
    };
  }, [mode, query]);

  useEffect(() => {
    if (!revealed) return;
    if (mode === "understand") setSeenUnderstand(true);
    if (mode === "generate") setSeenGenerate(true);
  }, [mode, revealed]);

  const passed =
    revealed &&
    prediction === "understand_full_generate_causal" &&
    seenUnderstand &&
    seenGenerate &&
    calculation.understandVisible.length === IMAGE &&
    calculation.understandFuture.length === calculation.futureCount &&
    calculation.generateFuture.length === 0 &&
    query < IMAGE - 1;

  const completion = useMemo(
    () => ({
      lessonId: 41,
      query,
      mode,
      prediction,
      understandVisible: calculation.understandVisible.length,
      understandFuture: calculation.understandFuture.length,
      generateVisible: calculation.generateVisible.length,
      generateFuture: calculation.generateFuture.length,
    }),
    [
      calculation.generateFuture.length,
      calculation.generateVisible.length,
      calculation.understandFuture.length,
      calculation.understandVisible.length,
      mode,
      prediction,
      query,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function resetProgress() {
    setRevealed(false);
    setSeenUnderstand(false);
    setSeenGenerate(false);
  }

  function reset() {
    setQuery(5);
    setMode("understand");
    setPrediction("");
    resetProgress();
  }

  const cells = Array.from({ length: IMAGE }, (_, index) => index);

  return (
    <LabFrame
      lesson="41"
      title="同一张图：理解 mask 和生成 mask"
      description="同一张 4×4 离散图。先预测理解路径会不会看见未来格，再揭晓注意力。教学模拟，不是模型输出。"
    >
      <div className={styles.workspace}>
        <form className={styles.controls} onSubmit={(event) => event.preventDefault()}>
          <h3>Mask 控制台</h3>
          <p className={styles.note}>
            点格子选 query。理解：图在前，图像 token 全双向。生成：字在前，图像按光栅因果。
          </p>
          <fieldset>
            <legend>当前路径</legend>
            <label>
              <input
                type="radio"
                name="mask-mode"
                checked={mode === "understand"}
                onChange={() => setMode("understand")}
              />
              <span>理解 mask</span>
            </label>
            <label>
              <input
                type="radio"
                name="mask-mode"
                checked={mode === "generate"}
                onChange={() => setMode("generate")}
              />
              <span>生成 mask</span>
            </label>
          </fieldset>
          <label>
            <span>
              Query 格 <output>{query}</output>
            </span>
            <span className={styles.note}>
              光栅 {Math.floor(query / SIDE)},{query % SIDE}；未来格{" "}
              {IMAGE - query - 1}
            </span>
          </label>
        </form>

        <div className={styles.stage}>
          <div className={styles.boards}>
            <figure className={styles.board}>
              <figcaption>同一张离散图 · 码本着色</figcaption>
              <div className={styles.prompt} aria-hidden="true">
                {PROMPT.map((token) => (
                  <span key={token}>{token}</span>
                ))}
              </div>
              <div className={styles.grid}>
                {cells.map((cell) => (
                  <button
                    key={`src-${cell}`}
                    type="button"
                    className={`${styles.cell} ${
                      cell === query ? styles.cellQuery : ""
                    }`}
                    style={{ "--cell-fill": cellFill(cell) } as React.CSSProperties}
                    onClick={() => {
                      setQuery(Math.min(IMAGE - 2, cell));
                      resetProgress();
                    }}
                  >
                    <b>{cell}</b>
                  </button>
                ))}
              </div>
            </figure>
            <figure className={styles.board}>
              <figcaption>
                {mode === "understand" ? "理解：query 能看谁" : "生成：query 能看谁"}
              </figcaption>
              <div className={styles.prompt} aria-hidden="true">
                {PROMPT.map((token, index) => (
                  <span key={`p-${token}`}>
                    {mode === "generate" ? `t${index}` : "·"}
                  </span>
                ))}
              </div>
              <div className={styles.grid}>
                {cells.map((cell) => {
                  const visible = revealed && calculation.visible.includes(cell);
                  const blocked = revealed && !calculation.visible.includes(cell);
                  const future = revealed && cell > query;
                  return (
                    <button
                      key={`mask-${cell}`}
                      type="button"
                      className={[
                        styles.cell,
                        cell === query ? styles.cellQuery : "",
                        visible ? styles.cellVisible : "",
                        blocked ? styles.cellBlocked : "",
                        future && visible ? styles.cellFuture : "",
                      ]
                        .filter(Boolean)
                        .join(" ")}
                      style={{ "--cell-fill": cellFill(cell) } as React.CSSProperties}
                      onClick={() => {
                        setQuery(Math.min(IMAGE - 2, cell));
                        resetProgress();
                      }}
                    >
                      <b>{revealed ? (visible ? "看" : "挡") : "?"}</b>
                    </button>
                  );
                })}
              </div>
            </figure>
          </div>
          <p className={styles.legend}>
            <span>
              <i className={styles.swatchQuery} />
              query
            </span>
            <span>
              <i className={styles.swatchVisible} />
              允许看见
            </span>
            <span>
              <i className={styles.swatchBlocked} />
              挡住
            </span>
          </p>
          <dl className={styles.metrics}>
            <div>
              <dt>可见图像格</dt>
              <dd>
                {revealed ? `${calculation.visible.length} / ${IMAGE}` : "—"}
              </dd>
            </div>
            <div>
              <dt>其中未来格</dt>
              <dd>
                {revealed
                  ? `${calculation.futureVisible.length} / ${calculation.futureCount}`
                  : "—"}
              </dd>
            </div>
            <div>
              <dt>码本编号范围</dt>
              <dd>[0, 7]</dd>
            </div>
          </dl>
          <p className={styles.formula}>
            理解：图像 query 看全部 16 格。生成：只看光栅编号 ≤ query 的格。L 形用编号 1，背景用 0。
          </p>
        </div>
      </div>

      <div className={styles.predict}>
        <fieldset>
          <legend>先预测：理解路径和生成路径对未来像素 token 怎么处理？</legend>
          {[
            [
              "understand_full_generate_causal",
              "理解看全图，生成不能看未来格",
            ],
            ["both_full", "两条路都能看未来格"],
            ["both_causal", "两条路都不能看未来格"],
            ["generate_sees_future", "只有生成能看未来格"],
          ].map(([value, label]) => (
            <label key={value}>
              <input
                type="radio"
                name="mask-prediction"
                value={value}
                checked={prediction === value}
                onChange={() => {
                  setPrediction(value);
                  resetProgress();
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
            onClick={() => setRevealed(true)}
          >
            揭晓 mask
          </button>
        </div>
      </div>
      {revealed && prediction !== "understand_full_generate_causal" && (
        <p className={styles.feedback}>
          切到理解路径，未来格应显示“看”；切到生成路径，未来格应显示“挡”。数字揭晓前不能当答案。
        </p>
      )}
      {revealed &&
        prediction === "understand_full_generate_causal" &&
        (!seenUnderstand || !seenGenerate) && (
          <p className={styles.feedback}>
            预测对了。请再切换一次路径，确认理解看见 16/16、生成的未来格为 0。
          </p>
        )}
      <Gate passed={passed}>
        先选对“理解看全图、生成不能看未来格”，再分别揭晓两条路径。数字来自教学模拟，不是模型输出。
      </Gate>
    </LabFrame>
  );
}
