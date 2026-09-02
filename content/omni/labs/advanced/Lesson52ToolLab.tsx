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
import styles from "./Lesson52ToolLab.module.css";

type PathId = "direct" | "calculator";
type PredictionId =
  | "mental-ok"
  | "saw-mental-wrong-tool-right"
  | "ocr-fail-both-wrong"
  | "lucky-tool";

type LineItem = {
  name: string;
  printed: number;
  dirty: number;
};

const ITEMS: LineItem[] = [
  { name: "美式浓缩咖啡", printed: 18.9, dirty: 18.9 },
  { name: "火腿三明治", printed: 26.5, dirty: 26.5 },
  { name: "凯撒沙拉", printed: 15.8, dirty: 15.08 },
];

const TRUE_SUBTOTAL = 61.2;
const CARRY_ERROR = 10;
const OCR_THRESHOLD = 0.6;

const PREDICTION_OPTIONS: { value: PredictionId; label: string }[] = [
  {
    value: "mental-ok",
    label: "看见了数字、心算对、工具多余",
  },
  {
    value: "saw-mental-wrong-tool-right",
    label: "看见了数字、心算错、工具对",
  },
  {
    value: "ocr-fail-both-wrong",
    label: "没看见数字、心算和工具都错",
  },
  {
    value: "lucky-tool",
    label: "没看见数字、工具碰巧对",
  },
];

function yuan(value: number) {
  return value.toFixed(2);
}

function sum(values: number[]) {
  return values.reduce((total, value) => total + value, 0);
}

export function Lesson52ToolLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    ocr: numberFrom(initialState, "ocr", 1, 0, 1),
    path: stringFrom(initialState, "path", "calculator") as PathId,
    prediction: stringFrom(initialState, "prediction", "") as PredictionId | "",
  };
  const [ocr, setOcr] = useState(defaults.ocr);
  const [path, setPath] = useState<PathId>(
    defaults.path === "direct" ? "direct" : "calculator",
  );
  const [prediction, setPrediction] = useState<PredictionId | "">(
    defaults.prediction,
  );
  const [ran, setRan] = useState(false);

  const result = useMemo(() => {
    const digits = ITEMS.map((item) =>
      ocr >= OCR_THRESHOLD ? item.printed : item.dirty,
    );
    const ocrSum = round(sum(digits), 2);
    const mental = round(ocrSum - CARRY_ERROR, 2);
    const tool = ocrSum;
    const answer = path === "calculator" ? tool : mental;
    const ocrMatch = digits.every(
      (value, index) => Math.abs(value - ITEMS[index].printed) < 1e-9,
    );
    const mentalWrong = Math.abs(mental - TRUE_SUBTOTAL) > 1e-9;
    const toolRight = Math.abs(tool - TRUE_SUBTOTAL) < 1e-9;
    const sawMentalWrongToolRight = ocrMatch && mentalWrong && toolRight;
    return {
      digits,
      ocrSum,
      mental,
      tool,
      answer,
      ocrMatch,
      mentalWrong,
      toolRight,
      sawMentalWrongToolRight,
    };
  }, [ocr, path]);

  const passed =
    ran &&
    prediction === "saw-mental-wrong-tool-right" &&
    path === "calculator" &&
    result.sawMentalWrongToolRight;

  const completion = useMemo(
    () => ({
      lessonId: 52,
      ocr: round(ocr, 2),
      path,
      prediction,
      ocrDigits: result.digits.map(yuan),
      mental: result.mental,
      tool: result.tool,
      answer: result.answer,
      trueSubtotal: TRUE_SUBTOTAL,
      ocrMatch: result.ocrMatch,
      mentalWrong: result.mentalWrong,
      toolRight: result.toolRight,
    }),
    [ocr, path, prediction, result],
  );
  useCompletionGate(passed, onComplete, completion);

  function reset() {
    setOcr(defaults.ocr);
    setPath("calculator");
    setPrediction("");
    setRan(false);
  }

  return (
    <LabFrame
      lesson="52"
      title="发票小计：看见数字以后还要不要调计算器"
      description="教学模拟，不是模型输出。印刷数字固定为 18.90、26.50、15.80，真值 61.20。玩具模型读数后永远漏掉十位进位。先预测四列关系，再揭晓 OCR、心算和工具。"
    >
      <div className={styles.workspace}>
        <aside className={styles.controls}>
          <h3>调用设置</h3>
          <label>
            <span>
              OCR 清晰度
              <output>{ocr.toFixed(2)}</output>
            </span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={ocr}
              onChange={(event) => {
                setOcr(Number(event.target.value));
                setRan(false);
              }}
            />
          </label>
          <fieldset>
            <legend>回答路径</legend>
            <label>
              <input
                type="radio"
                name="lesson52-path"
                checked={path === "direct"}
                onChange={() => {
                  setPath("direct");
                  setRan(false);
                }}
              />
              直接答（心算）
            </label>
            <label>
              <input
                type="radio"
                name="lesson52-path"
                checked={path === "calculator"}
                onChange={() => {
                  setPath("calculator");
                  setRan(false);
                }}
              />
              调用计算器
            </label>
          </fieldset>
          <p className={styles.note}>
            清晰度低于 {OCR_THRESHOLD.toFixed(1)} 时，沙拉被读成 15.08。心算固定把 OCR
            合计减去 10.00，模拟漏掉十位进位。计算器只对 OCR 读到的数字求和。
          </p>
        </aside>
        <div className={styles.stage}>
          <p className={styles.query}>
            查询：<strong>这张发票的税前小计是多少？</strong>
            印刷行金额可见，OCR 读数、心算和工具结果在揭晓前不显示。
          </p>
          <div className={styles.invoice} aria-label="教学发票">
            <header>
              <span>Learn Omni 咖啡店</span>
              <span>INV-52</span>
            </header>
            <div className={styles.lines}>
              <span>项目</span>
              <span>印刷金额</span>
              {ITEMS.map((item) => (
                <article key={item.name}>
                  <b>{item.name}</b>
                  <code>{yuan(item.printed)}</code>
                </article>
              ))}
            </div>
            <footer>
              <span>税前小计（真值）</span>
              <strong>{ran ? yuan(TRUE_SUBTOTAL) : "揭晓后显示"}</strong>
            </footer>
          </div>
          <dl className={styles.metrics}>
            <div>
              <dt>OCR 读数</dt>
              <dd>{ran ? result.digits.map(yuan).join(" + ") : "—"}</dd>
            </div>
            <div>
              <dt>心算</dt>
              <dd>{ran ? yuan(result.mental) : "—"}</dd>
            </div>
            <div>
              <dt>计算器</dt>
              <dd>{ran ? yuan(result.tool) : "—"}</dd>
            </div>
            <div>
              <dt>最终回答</dt>
              <dd>{ran ? yuan(result.answer) : "—"}</dd>
            </div>
          </dl>
          <div className={styles.predict}>
            <fieldset>
              <legend>揭晓前预测：哪一种说法成立？</legend>
              {PREDICTION_OPTIONS.map((option) => (
                <label key={option.value}>
                  <input
                    type="radio"
                    name="lesson52-prediction"
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
                揭晓小计
              </button>
            </div>
          </div>
          {!prediction ? (
            <p className={styles.feedback}>先选预测，再揭晓 OCR、心算和工具数字。</p>
          ) : null}
          {ran && prediction !== "saw-mental-wrong-tool-right" ? (
            <p className={styles.feedback}>
              验收要的是「看见了数字、心算错、工具对」。当前 OCR{" "}
              {result.ocrMatch ? "命中印刷数字" : "读错沙拉"}，心算 {yuan(result.mental)}
              ，工具 {yuan(result.tool)}。
            </p>
          ) : null}
          {ran &&
          prediction === "saw-mental-wrong-tool-right" &&
          !result.ocrMatch ? (
            <p className={styles.feedback}>
              沙拉被读成 15.08，工具也会错。把 OCR 清晰度拉到 {OCR_THRESHOLD.toFixed(1)}{" "}
              以上再揭晓。
            </p>
          ) : null}
          {ran &&
          prediction === "saw-mental-wrong-tool-right" &&
          result.ocrMatch &&
          path !== "calculator" ? (
            <p className={styles.feedback}>
              OCR 已经读对，但路径仍是直接答，最终回答是心算 {yuan(result.mental)}
              。切到「调用计算器」再揭晓。
            </p>
          ) : null}
          {ran && passed ? (
            <p className={styles.reveal}>
              OCR 读到 18.90 + 26.50 + 15.80。心算漏掉十位进位得到 {yuan(result.mental)}
              。计算器得到 {yuan(result.tool)}，等于真值 {yuan(TRUE_SUBTOTAL)}
              。看见了数字，仍必须调用工具。
            </p>
          ) : null}
        </div>
      </div>
      <Gate passed={passed}>
        先提交「看见了数字、心算错、工具对」，再把 OCR 拉清晰并调用计算器：印刷数字被读对，心算
        51.20，工具 61.20。
      </Gate>
    </LabFrame>
  );
}
