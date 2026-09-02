"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import { round, stringFrom, useCompletionGate } from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson51CotLab.module.css";

type QuestionId = "red" | "star";
type Occupant = {
  row: number;
  col: number;
  name: string;
  fill: string;
};

const GRID = 4;

const OCCUPANTS: Occupant[] = [
  { row: 0, col: 1, name: "红杯", fill: "#b42318" },
  { row: 0, col: 3, name: "绿叶", fill: "#2f6b3a" },
  { row: 1, col: 2, name: "蓝球", fill: "#1d4e89" },
  { row: 2, col: 0, name: "黄星", fill: "#c47b12" },
  { row: 2, col: 3, name: "红杯", fill: "#b42318" },
  { row: 3, col: 1, name: "白盘", fill: "#d9d2c5" },
];

const QUESTIONS: Record<
  QuestionId,
  { prompt: string; goldAnswer: string; goldCells: string[] }
> = {
  red: {
    prompt: "红色杯子有几个？",
    goldAnswer: "2",
    goldCells: ["0,1", "2,3"],
  },
  star: {
    prompt: "黄色星星有几个？",
    goldAnswer: "1",
    goldCells: ["2,0"],
  },
};

const PREDICTIONS = [
  {
    value: "off-correct-empty",
    label: "关掉必须引用后：答案对、引用格为空",
  },
  {
    value: "off-also-wrong",
    label: "关掉后答案也会错",
  },
  {
    value: "off-still-cites",
    label: "关掉后仍会写出引用格",
  },
  {
    value: "need-cite-for-answer",
    label: "只有开着引用时答案才对",
  },
] as const;

function cellKey(row: number, col: number) {
  return `${row},${col}`;
}

function occupantAt(row: number, col: number) {
  return OCCUPANTS.find((item) => item.row === row && item.col === col);
}

function simulate(question: QuestionId, mustCite: boolean) {
  const spec = QUESTIONS[question];
  const cited = mustCite ? [...spec.goldCells] : [];
  const reason = mustCite
    ? `格子 ${spec.goldCells.map((key) => `(${key})`).join(" 与 ")} 是被问物体`
    : "常见场景里这类物体成对或单独出现，不必点格";
  const answer = spec.goldAnswer;
  const answerCorrect = answer === spec.goldAnswer;
  const rAnswer = answerCorrect ? 1 : 0;
  const goldSet = new Set(spec.goldCells);
  const rProcess = cited.some((key) => goldSet.has(key)) ? 1 : 0;
  const rCombined = mustCite ? rAnswer * rProcess : rAnswer;
  return {
    prompt: spec.prompt,
    goldAnswer: spec.goldAnswer,
    goldCells: spec.goldCells,
    cited,
    reason,
    answer,
    answerCorrect,
    rAnswer,
    rProcess,
    rCombined,
    emptyCite: cited.length === 0,
  };
}

export function Lesson51CotLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    question: stringFrom(initialState, "question", "red") as QuestionId,
    mustCite: stringFrom(initialState, "mustCite", "on"),
    prediction: stringFrom(initialState, "prediction", ""),
  };
  const [question, setQuestion] = useState<QuestionId>(
    defaults.question === "star" ? "star" : "red",
  );
  const [mustCite, setMustCite] = useState(defaults.mustCite !== "off");
  const [prediction, setPrediction] = useState(defaults.prediction);
  const [revealed, setRevealed] = useState(false);

  const result = useMemo(
    () => simulate(question, mustCite),
    [mustCite, question],
  );

  const passed =
    revealed &&
    prediction === "off-correct-empty" &&
    !mustCite &&
    result.answerCorrect &&
    result.emptyCite;

  const completion = useMemo(
    () => ({
      lessonId: 51,
      question,
      mustCite,
      prediction,
      answer: result.answer,
      cited: result.cited,
      rAnswer: result.rAnswer,
      rProcess: result.rProcess,
      rCombined: round(result.rCombined, 3),
    }),
    [
      mustCite,
      prediction,
      question,
      result.answer,
      result.cited,
      result.rAnswer,
      result.rCombined,
      result.rProcess,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function reset() {
    setQuestion("red");
    setMustCite(true);
    setPrediction("");
    setRevealed(false);
  }

  return (
    <LabFrame
      lesson="51"
      title="计数题：关掉必须引用格子"
      description="教学模拟，不是模型输出。先预测关掉必须引用之后账本会怎样，再揭晓答案、引用格和过程奖励。"
    >
      <div className={styles.workspace}>
        <form
          className={styles.controls}
          onSubmit={(event) => event.preventDefault()}
        >
          <h3>推理控制台</h3>
          <fieldset className={styles.questionSet}>
            <legend>计数题</legend>
            {(
              [
                ["red", "红杯个数"],
                ["star", "黄星个数"],
              ] as const
            ).map(([value, label]) => (
              <label key={value}>
                <input
                  type="radio"
                  name="cot-question"
                  value={value}
                  checked={question === value}
                  onChange={() => {
                    setQuestion(value);
                    setRevealed(false);
                  }}
                />
                <span>{label}</span>
              </label>
            ))}
          </fieldset>
          <p className={styles.prompt}>{QUESTIONS[question].prompt}</p>
          <fieldset className={styles.toggleSet}>
            <legend>必须引用格子</legend>
            <label>
              <input
                type="radio"
                name="must-cite"
                checked={mustCite}
                onChange={() => {
                  setMustCite(true);
                  setRevealed(false);
                }}
              />
              <span>开：推理必须点格</span>
            </label>
            <label>
              <input
                type="radio"
                name="must-cite"
                checked={!mustCite}
                onChange={() => {
                  setMustCite(false);
                  setRevealed(false);
                }}
              />
              <span>关：允许空引用</span>
            </label>
          </fieldset>
          <p className={styles.note}>
            验收盯关掉开关的那一侧。改开关或题目会清掉揭晓，避免先看数字再选预测。
          </p>
        </form>

        <div className={styles.stage}>
          <div className={styles.gridWrap}>
            <div className={styles.grid} role="grid" aria-label="4 乘 4 计数网格">
              {Array.from({ length: GRID * GRID }, (_, index) => {
                const row = Math.floor(index / GRID);
                const col = index % GRID;
                const key = cellKey(row, col);
                const occupant = occupantAt(row, col);
                const cited = revealed && result.cited.includes(key);
                const gold = revealed && result.goldCells.includes(key);
                return (
                  <div
                    key={key}
                    role="gridcell"
                    className={styles.cell}
                    data-cited={cited ? "true" : "false"}
                    data-gold={gold ? "true" : "false"}
                    style={
                      occupant
                        ? { background: occupant.fill, color: "#fff" }
                        : undefined
                    }
                    aria-label={
                      occupant
                        ? `第 ${row} 行第 ${col} 列 ${occupant.name}`
                        : `第 ${row} 行第 ${col} 列空`
                    }
                  >
                    {occupant ? occupant.name : ""}
                  </div>
                );
              })}
            </div>
            <ul className={styles.legend}>
              <li>
                <i data-swatch="cited" />
                推理引用的格子
              </li>
              <li>
                <i data-swatch="gold" />
                真值格子
              </li>
            </ul>
          </div>

          <dl className={styles.trace}>
            <div>
              <dt>推理 span</dt>
              <dd>{revealed ? result.reason : "揭晓前不显示"}</dd>
            </div>
            <div>
              <dt>答案 span</dt>
              <dd>{revealed ? result.answer : "—"}</dd>
            </div>
          </dl>

          <dl className={styles.metrics}>
            <div>
              <dt>答案</dt>
              <dd>{revealed ? (result.answerCorrect ? "对" : "错") : "—"}</dd>
            </div>
            <div>
              <dt>引用格</dt>
              <dd>
                {revealed ? result.cited.join(" / ") || "空" : "—"}
              </dd>
            </div>
            <div>
              <dt>r_ans</dt>
              <dd>{revealed ? result.rAnswer.toFixed(0) : "—"}</dd>
            </div>
            <div>
              <dt>r_proc</dt>
              <dd>{revealed ? result.rProcess.toFixed(0) : "—"}</dd>
            </div>
          </dl>
          <p className={styles.formula}>
            r_ans = 1[answer = gold]；r_proc = 1[C(y) ∩ G ≠ ∅]；必须引用时 r = r_ans ·
            r_proc
          </p>
        </div>
      </div>

      <div className={styles.predict}>
        <fieldset>
          <legend>先预测：关掉必须引用格子之后，哪句话成立？</legend>
          {PREDICTIONS.map((option) => (
            <label key={option.value}>
              <input
                type="radio"
                name="cot-prediction"
                value={option.value}
                checked={prediction === option.value}
                onChange={() => {
                  setPrediction(option.value);
                  setRevealed(false);
                }}
              />
              <span>{option.label}</span>
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
            揭晓引用
          </button>
        </div>
      </div>
      {revealed && prediction !== "off-correct-empty" && (
        <p className={styles.feedback}>
          关掉必须引用后，玩具模型仍输出正确个数，但推理 span 里没有格子。过程奖励是
          0。
        </p>
      )}
      {revealed && prediction === "off-correct-empty" && mustCite && (
        <p className={styles.feedback}>
          预测句针对关掉开关的那一侧。把必须引用拨到关，再揭晓一次。
        </p>
      )}
      <Gate passed={passed}>
        先选对“关掉后答案对、引用格为空”，再关掉必须引用并揭晓。教学模拟，不是模型输出。
      </Gate>
    </LabFrame>
  );
}
