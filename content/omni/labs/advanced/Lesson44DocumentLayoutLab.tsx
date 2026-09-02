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
import styles from "./Lesson44DocumentLayoutLab.module.css";

type QuestionId = "total" | "line1" | "contract";
type OrderId = "layout" | "raster";
type PredictionId = "both-ok" | "header-box" | "wrong-number" | "cross-page";
type RegionId =
  | "invoiceNo"
  | "amountHeader"
  | "line1"
  | "line2"
  | "total"
  | "promo"
  | "contract";

type Box = readonly [number, number, number, number];

const BOXES: Record<RegionId, Box> = {
  invoiceNo: [78, 6, 96, 18],
  amountHeader: [70, 30, 95, 42],
  line1: [70, 44, 95, 56],
  line2: [70, 58, 95, 70],
  total: [70, 80, 95, 96],
  promo: [8, 108, 48, 120],
  contract: [12, 20, 70, 36],
};

const QUESTIONS: Record<
  QuestionId,
  { prompt: string; truthText: string; truthRegion: RegionId }
> = {
  total: {
    prompt: "合计金额是多少？",
    truthText: "32.00",
    truthRegion: "total",
  },
  line1: {
    prompt: "打印纸这一行的金额是多少？",
    truthText: "24.00",
    truthRegion: "line1",
  },
  contract: {
    prompt: "合同编号是什么？",
    truthText: "HT-2024-09",
    truthRegion: "contract",
  },
};

function boxArea(box: Box) {
  return Math.max(0, box[2] - box[0]) * Math.max(0, box[3] - box[1]);
}

function iou(pred: Box, gt: Box) {
  const ix1 = Math.max(pred[0], gt[0]);
  const iy1 = Math.max(pred[1], gt[1]);
  const ix2 = Math.min(pred[2], gt[2]);
  const iy2 = Math.min(pred[3], gt[3]);
  const intersection = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  const union = boxArea(pred) + boxArea(gt) - intersection;
  return union <= 0 ? 0 : intersection / union;
}

function simulate(question: QuestionId, prior: number, order: OrderId) {
  const spec = QUESTIONS[question];
  const biased = prior >= 0.55;
  let answer = spec.truthText;
  let used: RegionId = spec.truthRegion;

  if (question === "total") {
    if (biased) {
      answer = "32.00";
      used = "amountHeader";
    } else {
      answer = "32.00";
      used = "total";
    }
  } else if (question === "line1") {
    if (order === "raster" && biased) {
      answer = "32.00";
      used = "amountHeader";
    } else if (biased) {
      answer = "24.00";
      used = "amountHeader";
    } else {
      answer = "24.00";
      used = "line1";
    }
  } else if (biased) {
    answer = "128";
    used = "invoiceNo";
  } else {
    answer = "HT-2024-09";
    used = "contract";
  }

  const contentOk = answer === spec.truthText;
  const boxIoU = iou(BOXES[used], BOXES[spec.truthRegion]);
  const boxOk = boxIoU >= 0.5;
  const layoutOk = contentOk && boxOk;
  const headerFail = contentOk && used === "amountHeader" && !boxOk;
  const wrongNumber = !contentOk && boxOk;
  const crossFail = question === "contract" && !contentOk && used === "invoiceNo";

  return {
    prompt: spec.prompt,
    truthText: spec.truthText,
    truthRegion: spec.truthRegion,
    answer,
    used,
    contentOk,
    boxIoU,
    boxOk,
    layoutOk,
    headerFail,
    wrongNumber,
    crossFail,
  };
}

export function Lesson44DocumentLayoutLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    prior: numberFrom(initialState, "prior", 0.7, 0, 1),
    question: stringFrom(initialState, "question", "total") as QuestionId,
    order: stringFrom(initialState, "order", "layout") as OrderId,
    prediction: stringFrom(initialState, "prediction", "") as PredictionId | "",
  };
  const [prior, setPrior] = useState(defaults.prior);
  const [question, setQuestion] = useState<QuestionId>(
    ["total", "line1", "contract"].includes(defaults.question)
      ? defaults.question
      : "total",
  );
  const [order, setOrder] = useState<OrderId>(
    defaults.order === "raster" ? "raster" : "layout",
  );
  const [prediction, setPrediction] = useState<PredictionId | "">(
    defaults.prediction,
  );
  const [guess, setGuess] = useState<RegionId | null>(null);
  const [page, setPage] = useState<1 | 2>(1);
  const [ran, setRan] = useState(false);
  const [foundHeaderFail, setFoundHeaderFail] = useState(false);
  const [foundLayoutOk, setFoundLayoutOk] = useState(false);

  const result = useMemo(
    () => simulate(question, prior, order),
    [order, prior, question],
  );

  const passed = foundHeaderFail && foundLayoutOk;
  const completion = useMemo(
    () => ({
      lessonId: 44,
      question,
      order,
      prior: round(prior, 2),
      answer: result.answer,
      used: result.used,
      boxIoU: round(result.boxIoU, 3),
      contentOk: result.contentOk,
      layoutOk: result.layoutOk,
      headerFail: result.headerFail,
      foundHeaderFail,
      foundLayoutOk,
    }),
    [
      foundHeaderFail,
      foundLayoutOk,
      order,
      prior,
      question,
      result.answer,
      result.boxIoU,
      result.contentOk,
      result.headerFail,
      result.layoutOk,
      result.used,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function invalidate() {
    setRan(false);
  }

  function reset() {
    setPrior(defaults.prior);
    setQuestion("total");
    setOrder("layout");
    setPrediction("");
    setGuess(null);
    setPage(1);
    setRan(false);
    setFoundHeaderFail(false);
    setFoundLayoutOk(false);
  }

  function run() {
    const next = simulate(question, prior, order);
    if (next.headerFail) setFoundHeaderFail(true);
    if (next.layoutOk) setFoundLayoutOk(true);
    setRan(true);
    if (question === "contract" && next.used === "contract") {
      setPage(2);
    }
  }

  const guessHit = guess !== null && result.used === guess;

  function mark(region: RegionId) {
    return {
      "data-used": ran && result.used === region ? "true" : "false",
      "data-truth": ran && result.truthRegion === region ? "true" : "false",
      "data-guess": guess === region ? "true" : "false",
    };
  }

  return (
    <LabFrame
      lesson="44"
      title="发票上拆开数字和单元格"
      description="教学模拟，不是模型输出。先预测这一问会「内容对且框对」「读对数字但框在表头」「框对但数字错」还是「跨页键值失败」，再揭晓玩具模型的字符串和它实际框住的格子。"
    >
      <div className={styles.workspace}>
        <form className={styles.controls} onSubmit={(event) => event.preventDefault()}>
          <h3>版面探针</h3>
          <fieldset className={styles.questionSet}>
            <legend>提问</legend>
            {(
              [
                ["total", "合计金额"],
                ["line1", "行内金额"],
                ["contract", "跨页合同号"],
              ] as const
            ).map(([value, label]) => (
              <label key={value}>
                <input
                  type="radio"
                  name="layout-question"
                  value={value}
                  checked={question === value}
                  onChange={() => {
                    setQuestion(value);
                    setPage(value === "contract" ? 2 : 1);
                    invalidate();
                  }}
                />
                <span>{label}</span>
              </label>
            ))}
          </fieldset>
          <p className={styles.prompt}>{QUESTIONS[question].prompt}</p>
          <fieldset className={styles.orderSet}>
            <legend>阅读顺序</legend>
            {(
              [
                ["layout", "版面顺序"],
                ["raster", "栅格扫描"],
              ] as const
            ).map(([value, label]) => (
              <label key={value}>
                <input
                  type="radio"
                  name="layout-order"
                  value={value}
                  checked={order === value}
                  onChange={() => {
                    setOrder(value);
                    invalidate();
                  }}
                />
                <span>{label}</span>
              </label>
            ))}
          </fieldset>
          <label>
            <span>
              表头捷径强度 <output>{prior.toFixed(2)}</output>
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
            捷径 ≥ 0.55 时，玩具模型会把「金额」读成表头栏。揭晓前不显示字符串、框和 IoU。点格子可先猜它会框哪里。
          </p>
        </form>

        <div className={styles.stage}>
          <div className={styles.pageTabs}>
            <button
              type="button"
              data-active={page === 1 ? "true" : "false"}
              onClick={() => setPage(1)}
            >
              第 1 页 发票
            </button>
            <button
              type="button"
              data-active={page === 2 ? "true" : "false"}
              onClick={() => setPage(2)}
            >
              第 2 页 合同号
            </button>
          </div>

          <div className={styles.invoiceWrap}>
            {page === 1 ? (
              <div className={styles.invoice} aria-label="教学发票第一页">
                <div className={styles.invoiceHead}>
                  <div>
                    <h4>教学发票</h4>
                    <small>日期 2024-03-08</small>
                  </div>
                  <button
                    type="button"
                    className={styles.cell}
                    data-kind="label"
                    aria-label="发票号 128"
                    onClick={() => {
                      setGuess("invoiceNo");
                      invalidate();
                    }}
                    {...mark("invoiceNo")}
                  >
                    No.128
                  </button>
                </div>
                <div className={styles.table}>
                  <span className={styles.cell} data-kind="label">
                    项目
                  </span>
                  <span className={styles.cell} data-kind="label">
                    数量
                  </span>
                  <span className={styles.cell} data-kind="label">
                    单价
                  </span>
                  <button
                    type="button"
                    className={styles.cell}
                    data-kind="label"
                    aria-label="表头金额"
                    onClick={() => {
                      setGuess("amountHeader");
                      invalidate();
                    }}
                    {...mark("amountHeader")}
                  >
                    金额
                  </button>
                  <span className={styles.cell}>打印纸</span>
                  <span className={styles.cell}>2</span>
                  <span className={styles.cell}>12.00</span>
                  <button
                    type="button"
                    className={styles.cell}
                    aria-label="打印纸金额 24.00"
                    onClick={() => {
                      setGuess("line1");
                      invalidate();
                    }}
                    {...mark("line1")}
                  >
                    24.00
                  </button>
                  <span className={styles.cell}>装订</span>
                  <span className={styles.cell}>1</span>
                  <span className={styles.cell}>8.00</span>
                  <button
                    type="button"
                    className={styles.cell}
                    aria-label="装订金额 8.00"
                    onClick={() => {
                      setGuess("line2");
                      invalidate();
                    }}
                    {...mark("line2")}
                  >
                    8.00
                  </button>
                </div>
                <div className={styles.totalRow}>
                  <span>合计</span>
                  <button
                    type="button"
                    className={styles.cell}
                    aria-label="合计 32.00"
                    onClick={() => {
                      setGuess("total");
                      invalidate();
                    }}
                    {...mark("total")}
                  >
                    32.00
                  </button>
                </div>
                <p className={styles.note}>
                  备注：
                  <button
                    type="button"
                    className={styles.cell}
                    aria-label="促销 满128减20"
                    onClick={() => {
                      setGuess("promo");
                      invalidate();
                    }}
                    {...mark("promo")}
                  >
                    满128减20
                  </button>
                </p>
              </div>
            ) : (
              <div className={styles.page2} aria-label="教学发票第二页">
                <h4>续页 · 合同信息</h4>
                <p>
                  第 1 页只有发票号 128。合同编号印在这一页。只看第 1 页的模型会把 128 当成编号。
                </p>
                <button
                  type="button"
                  className={styles.cell}
                  aria-label="合同编号 HT-2024-09"
                  onClick={() => {
                    setGuess("contract");
                    invalidate();
                  }}
                  {...mark("contract")}
                >
                  合同编号 HT-2024-09
                </button>
              </div>
            )}
            <ul className={styles.legend}>
              <li>
                <i data-swatch="used" />
                模型框住的格子
              </li>
              <li>
                <i data-swatch="truth" />
                真值单元格
              </li>
              <li>
                <i data-swatch="guess" />
                你点选的猜测
              </li>
            </ul>
          </div>

          <dl className={styles.metrics}>
            <div>
              <dt>模型字符串</dt>
              <dd>{ran ? result.answer : "—"}</dd>
            </div>
            <div>
              <dt>内容命中</dt>
              <dd>{ran ? (result.contentOk ? "是" : "否") : "—"}</dd>
            </div>
            <div>
              <dt>框 IoU</dt>
              <dd>{ran ? result.boxIoU.toFixed(2) : "—"}</dd>
            </div>
            <div>
              <dt>框命中</dt>
              <dd>{ran ? (result.boxOk ? "是" : "否") : "—"}</dd>
            </div>
            <div>
              <dt>版面命中</dt>
              <dd>{ran ? (result.layoutOk ? "是" : "否") : "—"}</dd>
            </div>
            <div>
              <dt>使用格子</dt>
              <dd>{ran ? result.used : "—"}</dd>
            </div>
          </dl>
        </div>
      </div>

      <div className={styles.predict}>
        <fieldset>
          <legend>先预测：揭晓后这一问会出现哪种账本？</legend>
          {(
            [
              ["both-ok", "内容对且框对"],
              ["header-box", "读对数字但框在表头"],
              ["wrong-number", "框对但数字错"],
              ["cross-page", "跨页键值失败"],
            ] as const
          ).map(([value, label]) => (
            <label key={value}>
              <input
                type="radio"
                name="layout-prediction"
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
            揭晓字符串和框
          </button>
        </div>
      </div>

      {ran && (
        <p className={styles.feedback}>
          {result.headerFail
            ? "字符串 32.00 或 24.00 对了，框却盖在表头「金额」上。内容命中成立，版面命中失败。"
            : result.crossFail
              ? "合同编号在第 2 页。捷径高时玩具模型只看见第 1 页的 128，跨页键值失败。"
              : result.wrongNumber
                ? "框落在真值单元格上，但字符串来自发票号或邻行，内容失败，版面仍失败。"
                : result.layoutOk
                  ? "当前设置下内容和框都过线。把捷径调到 0.55 以上再问合计，才能看到表头假成功。"
                  : "当前设置下内容和框都未同时过线。"}
          {guess
            ? guessHit
              ? " 你点的格子就是揭晓后的使用格子。"
              : " 你点的格子不是揭晓后的使用格子。"
            : ""}
        </p>
      )}

      <ul className={styles.checklist}>
        <li data-done={foundHeaderFail ? "true" : "false"}>
          找到一例：读对数字但框在表头，版面失败
        </li>
        <li data-done={foundLayoutOk ? "true" : "false"}>
          找到一例：内容与框同时过线
        </li>
      </ul>

      <Gate passed={passed}>
        必须先触发「合计或行内金额读对、框在表头」，再找到一组内容和框都过线的对照。揭晓前要提交预测。
      </Gate>
    </LabFrame>
  );
}
