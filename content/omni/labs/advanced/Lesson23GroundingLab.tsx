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
import styles from "./Lesson23GroundingLab.module.css";

type QuestionId = "color" | "count" | "left" | "exist";
type PredictionId = "both-ok" | "wrong-cell" | "phantom" | "swap-lr";

type Occupant = {
  row: number;
  col: number;
  name: string;
  color: string;
  fill: string;
};

const GRID = 8;

const OCCUPANTS: Occupant[] = [
  { row: 0, col: 1, name: "杯A", color: "红", fill: "#b42318" },
  { row: 0, col: 5, name: "叶", color: "绿", fill: "#2f6b3a" },
  { row: 1, col: 1, name: "球", color: "蓝", fill: "#1d4e89" },
  { row: 3, col: 0, name: "星", color: "黄", fill: "#c47b12" },
  { row: 4, col: 2, name: "杯B", color: "红", fill: "#b42318" },
  { row: 4, col: 4, name: "盘", color: "白", fill: "#d9d2c5" },
  { row: 5, col: 3, name: "叉", color: "灰", fill: "#6d6a63" },
  { row: 6, col: 1, name: "苹", color: "绿", fill: "#3f7d3a" },
  { row: 6, col: 2, name: "蕉", color: "黄", fill: "#c9a227" },
];

const QUESTIONS: Record<
  QuestionId,
  { prompt: string; truthAnswer: string; truthCells: string[] }
> = {
  color: {
    prompt: "格子 (0,1) 的杯子是什么颜色？",
    truthAnswer: "红",
    truthCells: ["0,1"],
  },
  count: {
    prompt: "红色杯子有几个？",
    truthAnswer: "2",
    truthCells: ["0,1", "4,2"],
  },
  left: {
    prompt: "香蕉左边是什么？",
    truthAnswer: "苹",
    truthCells: ["6,1"],
  },
  exist: {
    prompt: "图里有酒杯吗？",
    truthAnswer: "无",
    truthCells: [],
  },
};

function cellKey(row: number, col: number) {
  return `${row},${col}`;
}

function occupantAt(row: number, col: number) {
  return OCCUPANTS.find((item) => item.row === row && item.col === col);
}

function simulate(question: QuestionId, prior: number) {
  const spec = QUESTIONS[question];
  const biased = prior >= 0.55;
  let answer = spec.truthAnswer;
  let used: string[] = [...spec.truthCells];
  let confidence = 0.82 - 0.18 * prior;

  if (question === "color") {
    if (biased) {
      used = ["4,2"];
      confidence = 0.71 + 0.2 * prior;
    } else {
      used = ["0,1"];
      confidence = 0.88;
    }
  } else if (question === "count") {
    if (biased) {
      used = ["4,2", "4,4"];
      answer = "2";
      confidence = 0.64 + 0.28 * prior;
    } else {
      used = ["0,1", "4,2"];
      confidence = 0.9;
    }
  } else if (question === "left") {
    if (biased) {
      used = ["6,2"];
      answer = "苹";
      confidence = 0.66 + 0.24 * prior;
    } else {
      used = ["6,1"];
      confidence = 0.87;
    }
  } else if (biased) {
    answer = "有";
    used = ["4,4"];
    confidence = 0.78 + 0.18 * prior;
  } else {
    answer = "无";
    used = [];
    confidence = 0.91;
  }

  const truthSet = new Set(spec.truthCells);
  const usedSet = new Set(used);
  const intersection = used.filter((key) => truthSet.has(key)).length;
  const union = new Set([...spec.truthCells, ...used]).size;
  const cellIoU = union === 0 ? (question === "exist" && answer === "无" ? 1 : 0) : intersection / union;
  const answerCorrect = answer === spec.truthAnswer;
  const cellsWrong =
    answerCorrect &&
    (used.length !== spec.truthCells.length ||
      used.some((key) => !truthSet.has(key)));
  const phantom =
    question === "exist" && spec.truthAnswer === "无" && answer === "有";
  const swapped =
    question === "left" && answerCorrect && usedSet.has("6,2") && !usedSet.has("6,1");

  return {
    prompt: spec.prompt,
    truthAnswer: spec.truthAnswer,
    truthCells: spec.truthCells,
    answer,
    used,
    confidence,
    cellIoU,
    answerCorrect,
    cellsWrong,
    phantom,
    swapped,
    hit: cellIoU >= 0.5 && !phantom,
  };
}

export function Lesson23GroundingLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    prior: numberFrom(initialState, "prior", 0.7, 0, 1),
    question: stringFrom(initialState, "question", "color") as QuestionId,
    prediction: stringFrom(initialState, "prediction", "") as PredictionId | "",
  };
  const [prior, setPrior] = useState(defaults.prior);
  const [question, setQuestion] = useState<QuestionId>(
    ["color", "count", "left", "exist"].includes(defaults.question)
      ? defaults.question
      : "color",
  );
  const [prediction, setPrediction] = useState<PredictionId | "">(
    defaults.prediction,
  );
  const [guessCell, setGuessCell] = useState<string | null>(null);
  const [ran, setRan] = useState(false);
  const [foundMismatch, setFoundMismatch] = useState(false);
  const [foundPhantom, setFoundPhantom] = useState(false);

  const result = useMemo(() => simulate(question, prior), [prior, question]);

  const passed = foundMismatch && foundPhantom;
  const completion = useMemo(
    () => ({
      lessonId: 23,
      question,
      prior: round(prior, 2),
      answer: result.answer,
      usedCells: result.used,
      cellIoU: round(result.cellIoU, 3),
      answerCorrect: result.answerCorrect,
      cellsWrong: result.cellsWrong,
      phantom: result.phantom,
      foundMismatch,
      foundPhantom,
    }),
    [
      foundMismatch,
      foundPhantom,
      prior,
      question,
      result.answer,
      result.answerCorrect,
      result.cellIoU,
      result.cellsWrong,
      result.phantom,
      result.used,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function invalidate() {
    setRan(false);
  }

  function reset() {
    setPrior(defaults.prior);
    setQuestion("color");
    setPrediction("");
    setGuessCell(null);
    setRan(false);
    setFoundMismatch(false);
    setFoundPhantom(false);
  }

  function run() {
    const next = simulate(question, prior);
    if (next.cellsWrong) setFoundMismatch(true);
    if (next.phantom && next.confidence >= 0.8) setFoundPhantom(true);
    setRan(true);
  }

  const guessHit = guessCell !== null && result.used.includes(guessCell);

  return (
    <LabFrame
      lesson="23"
      title="8×8 网格上拆开答案和格子"
      description="教学模拟，不是模型输出。先预测这一问会「都对」「答案对格子错」还是「物体不存在仍高置信答有」，再揭晓玩具模型的答案和它实际使用的格子。"
    >
      <div className={styles.workspace}>
        <form className={styles.controls} onSubmit={(event) => event.preventDefault()}>
          <h3>探针控制台</h3>
          <fieldset className={styles.questionSet}>
            <legend>提问</legend>
            {(
              [
                ["color", "颜色"],
                ["count", "计数"],
                ["left", "左边是什么"],
                ["exist", "是否存在"],
              ] as const
            ).map(([value, label]) => (
              <label key={value}>
                <input
                  type="radio"
                  name="grounding-question"
                  value={value}
                  checked={question === value}
                  onChange={() => {
                    setQuestion(value);
                    invalidate();
                  }}
                />
                <span>{label}</span>
              </label>
            ))}
          </fieldset>
          <p className={styles.prompt}>{QUESTIONS[question].prompt}</p>
          <label>
            <span>
              语言先验强度 <output>{prior.toFixed(2)}</output>
            </span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={prior}
              onChange={(event) => {
                setPrior(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <p className={styles.hint}>
            先验 ≥ 0.55 时，玩具模型会走共现和同色捷径。点格子可先猜它会看哪里，揭晓前不显示答案和命中。
          </p>
        </form>

        <div className={styles.stage}>
          <div className={styles.gridWrap}>
            <div className={styles.grid} role="grid" aria-label="8 乘 8 物体网格">
              {Array.from({ length: GRID * GRID }, (_, index) => {
                const row = Math.floor(index / GRID);
                const col = index % GRID;
                const key = cellKey(row, col);
                const occupant = occupantAt(row, col);
                const used = ran && result.used.includes(key);
                const truth = ran && result.truthCells.includes(key);
                const guessed = guessCell === key;
                return (
                  <button
                    type="button"
                    key={key}
                    role="gridcell"
                    className={styles.cell}
                    data-used={used ? "true" : "false"}
                    data-truth={truth ? "true" : "false"}
                    data-guess={guessed ? "true" : "false"}
                    style={
                      occupant
                        ? { background: occupant.fill, color: "#fff" }
                        : undefined
                    }
                    aria-label={
                      occupant
                        ? `第 ${row} 行第 ${col} 列 ${occupant.color}色${occupant.name}`
                        : `第 ${row} 行第 ${col} 列空`
                    }
                    onClick={() => {
                      setGuessCell(key);
                      invalidate();
                    }}
                  >
                    {occupant ? occupant.name : ""}
                  </button>
                );
              })}
            </div>
            <ul className={styles.legend}>
              <li>
                <i data-swatch="used" />
                模型使用的格子
              </li>
              <li>
                <i data-swatch="truth" />
                真值格子
              </li>
              <li>
                <i data-swatch="guess" />
                你点选的猜测
              </li>
            </ul>
          </div>

          <dl className={styles.metrics}>
            <div>
              <dt>模型答案</dt>
              <dd>{ran ? result.answer : "—"}</dd>
            </div>
            <div>
              <dt>置信度</dt>
              <dd>{ran ? result.confidence.toFixed(2) : "—"}</dd>
            </div>
            <div>
              <dt>使用格子</dt>
              <dd>{ran ? (result.used.join(" / ") || "无") : "—"}</dd>
            </div>
            <div>
              <dt>格子 IoU</dt>
              <dd>{ran ? result.cellIoU.toFixed(2) : "—"}</dd>
            </div>
            <div>
              <dt>VQA</dt>
              <dd>{ran ? (result.answerCorrect ? "对" : "错") : "—"}</dd>
            </div>
            <div>
              <dt>命中</dt>
              <dd>{ran ? (result.hit ? "是" : "否") : "—"}</dd>
            </div>
          </dl>
        </div>
      </div>

      <div className={styles.predict}>
        <fieldset>
          <legend>先预测：揭晓后这一问会出现哪种账本？</legend>
          {(
            [
              ["both-ok", "答案和格子都对"],
              ["wrong-cell", "答案对、格子错"],
              ["phantom", "物体不存在仍高置信答有"],
              ["swap-lr", "答案对，但看的是右边的物体"],
            ] as const
          ).map(([value, label]) => (
            <label key={value}>
              <input
                type="radio"
                name="grounding-prediction"
                value={value}
                checked={prediction === value}
                onChange={() => {
                  setPrediction(value);
                  invalidate();
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
            onClick={run}
          >
            揭晓答案和格子
          </button>
        </div>
      </div>

      {ran && (
        <p className={styles.feedback}>
          {result.cellsWrong
            ? "VQA 对、格子错：同色杯 B 或共现的盘子接住了答案，被问的像素没有被用上。"
            : result.phantom
              ? "酒杯不在网格里。先验高时玩具模型仍对盘子给出高置信「有」，这是 POPE 对抗负例的缩小版。"
              : result.swapped
                ? "「香蕉左边」的答案碰巧是苹果，使用的格子却是香蕉自己。"
                : "当前设置下答案和格子一致。把先验调到 0.55 以上，再分别问颜色和是否存在。"}
          {guessCell
            ? guessHit
              ? " 你点的格子在揭晓后的使用集合里。"
              : " 你点的格子不在揭晓后的使用集合里。"
            : ""}
        </p>
      )}

      <ul className={styles.checklist}>
        <li data-done={foundMismatch ? "true" : "false"}>
          找到一例：答案对、格子错
        </li>
        <li data-done={foundPhantom ? "true" : "false"}>
          找到一例：物体不存在仍高置信答有
        </li>
      </ul>

      <Gate passed={passed}>
        必须先后触发「颜色或计数在同色/共现格子上答对」和「酒杯不存在仍答有」。先验旋钮和提问都要动，不能只看静态图。
      </Gate>
    </LabFrame>
  );
}
