"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import { round, stringFrom, useCompletionGate } from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson43AudioUnderstandLab.module.css";

type UtteranceId = "math" | "meeting" | "keyboard";
type RecipeId =
  | "asr-pretrain"
  | "contaminated"
  | "instruction-sft"
  | "lora-discount";
type PredictionId = "both-ok" | "asr-ok-if-bad" | "asr-bad-if-ok" | "both-bad";

type Utterance = {
  heard: string;
  transcript: string;
  action: string;
};

const UTTERANCES: Record<UtteranceId, Utterance> = {
  math: {
    heard: "把三加五的结果乘以二，只回答数字",
    transcript: "把三加五的结果乘以二，只回答数字",
    action: "16",
  },
  meeting: {
    heard: "把会议室改到下午三点，不要回读我说的话，只回确认码 7K",
    transcript: "把会议室改到下午三点，不要回读我说的话，只回确认码 7K",
    action: "7K",
  },
  keyboard: {
    heard: "键盘声之后问：这是什么声音",
    transcript: "这是什么声音",
    action: "键盘敲击声",
  },
};

const ASR_CE = -Math.log(0.8);
const IF_CE = -Math.log(0.15);
const MIX_CE = (5 * ASR_CE + 3 * IF_CE) / 8;
const BOTH_CE = ASR_CE;
const DISCOUNT_ASR_CE = -Math.log(0.55);
const DISCOUNT_IF_CE = -Math.log(0.8);

function simulate(recipe: RecipeId, utteranceId: UtteranceId) {
  const sample = UTTERANCES[utteranceId];
  if (recipe === "asr-pretrain") {
    return {
      asrOutput: sample.transcript,
      ifOutput: sample.transcript,
      asrOk: true,
      ifOk: false,
      asrSet: "11-15",
      ifSet: "16-18",
      asrCount: 5,
      ifCount: 3,
      asrCe: ASR_CE,
      ifCe: IF_CE,
      mixCe: MIX_CE,
      maskNote: "两张 mask 不相等。执行头仍在复读听写。",
    };
  }
  if (recipe === "contaminated") {
    return {
      asrOutput: sample.transcript,
      ifOutput: `${sample.transcript} / ${sample.action.slice(0, 1)}…`,
      asrOk: true,
      ifOk: false,
      asrSet: "11-18",
      ifSet: "11-18",
      asrCount: 8,
      ifCount: 8,
      asrCe: ASR_CE,
      ifCe: IF_CE,
      mixCe: MIX_CE,
      maskNote: "污染 mask 把听写和执行并进同一张有效集合，计数变成 8。",
    };
  }
  if (recipe === "instruction-sft") {
    return {
      asrOutput: sample.transcript,
      ifOutput: sample.action,
      asrOk: true,
      ifOk: true,
      asrSet: "11-15",
      ifSet: "16-18",
      asrCount: 5,
      ifCount: 3,
      asrCe: BOTH_CE,
      ifCe: BOTH_CE,
      mixCe: BOTH_CE,
      maskNote: "指令 SFT：执行头输出动作，听写头仍对。两集合仍然不相等。",
    };
  }
  const asrOutput =
    utteranceId === "keyboard"
      ? "这是什么响声"
      : sample.transcript.replace("三", "四").replace("三点", "四点");
  return {
    asrOutput,
    ifOutput: sample.action,
    asrOk: false,
    ifOk: true,
    asrSet: "11-15",
    ifSet: "16-18",
    asrCount: 5,
    ifCount: 3,
    asrCe: DISCOUNT_ASR_CE,
    ifCe: DISCOUNT_IF_CE,
    mixCe: (5 * DISCOUNT_ASR_CE + 3 * DISCOUNT_IF_CE) / 8,
    maskNote: "LoRA 缩放减半：听写出现替换错误，执行头能跟指令。",
  };
}

function outcomeOf(result: { asrOk: boolean; ifOk: boolean }): PredictionId {
  if (result.asrOk && result.ifOk) return "both-ok";
  if (result.asrOk && !result.ifOk) return "asr-ok-if-bad";
  if (!result.asrOk && result.ifOk) return "asr-bad-if-ok";
  return "both-bad";
}

const TOKEN_LANES = [
  { id: "audio", label: "音频条件", count: 6, kind: "cond" },
  { id: "prompt", label: "提示条件", count: 5, kind: "cond" },
  { id: "asr", label: "听写目标", count: 5, kind: "asr" },
  { id: "action", label: "执行目标", count: 3, kind: "if" },
  { id: "eos", label: "eos", count: 1, kind: "cond" },
] as const;

export function Lesson43AudioUnderstandLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    utterance: stringFrom(initialState, "utterance", "math") as UtteranceId,
    recipe: stringFrom(initialState, "recipe", "asr-pretrain") as RecipeId,
    prediction: stringFrom(initialState, "prediction", "") as PredictionId | "",
  };
  const [utterance, setUtterance] = useState<UtteranceId>(
    ["math", "meeting", "keyboard"].includes(defaults.utterance)
      ? defaults.utterance
      : "math",
  );
  const [recipe, setRecipe] = useState<RecipeId>(
    [
      "asr-pretrain",
      "contaminated",
      "instruction-sft",
      "lora-discount",
    ].includes(defaults.recipe)
      ? defaults.recipe
      : "asr-pretrain",
  );
  const [prediction, setPrediction] = useState<PredictionId | "">(
    defaults.prediction,
  );
  const [ran, setRan] = useState(false);
  const [foundMismatch, setFoundMismatch] = useState(false);

  const result = useMemo(
    () => simulate(recipe, utterance),
    [recipe, utterance],
  );
  const actual = outcomeOf(result);
  const mismatch = result.asrOk && !result.ifOk;

  const passed = foundMismatch;
  const completion = useMemo(
    () => ({
      lessonId: 43,
      utterance,
      recipe,
      prediction,
      asrOk: result.asrOk,
      ifOk: result.ifOk,
      asrCount: result.asrCount,
      ifCount: result.ifCount,
      asrCe: round(result.asrCe, 5),
      ifCe: round(result.ifCe, 5),
      foundMismatch,
    }),
    [
      foundMismatch,
      prediction,
      recipe,
      result.asrCe,
      result.asrCount,
      result.asrOk,
      result.ifCe,
      result.ifCount,
      result.ifOk,
      utterance,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function invalidate() {
    setRan(false);
  }

  function reset() {
    setUtterance("math");
    setRecipe("asr-pretrain");
    setPrediction("");
    setRan(false);
    setFoundMismatch(false);
  }

  function run() {
    if (!prediction) return;
    const next = simulate(recipe, utterance);
    if (next.asrOk && !next.ifOk) setFoundMismatch(true);
    setRan(true);
  }

  const sample = UTTERANCES[utterance];
  const setsEqual = ran && result.asrSet === result.ifSet;

  return (
    <LabFrame
      lesson="43"
      title="同一句语音：转写对还是指令对"
      description="教学模拟，不是模型输出。先预测这一句在当前配方下会「两边都对」「转写对、指令错」「转写错、指令对」还是「两边都错」，再揭晓两头输出和两张有效集合。"
    >
      <div className={styles.workspace}>
        <form
          className={styles.controls}
          onSubmit={(event) => event.preventDefault()}
        >
          <h3>语音与配方</h3>
          <fieldset className={styles.choiceSet}>
            <legend>同一句语音</legend>
            {(
              [
                ["math", "算术约束"],
                ["meeting", "改会议"],
                ["keyboard", "键盘问句"],
              ] as const
            ).map(([value, label]) => (
              <label key={value}>
                <input
                  type="radio"
                  name="lesson43-utterance"
                  value={value}
                  checked={utterance === value}
                  onChange={() => {
                    setUtterance(value);
                    invalidate();
                  }}
                />
                <span>{label}</span>
              </label>
            ))}
          </fieldset>
          <p className={styles.heard}>{sample.heard}</p>
          <fieldset className={styles.choiceSet}>
            <legend>训练配方</legend>
            {(
              [
                ["asr-pretrain", "听写预训"],
                ["contaminated", "污染 mask"],
                ["instruction-sft", "指令 SFT"],
                ["lora-discount", "LoRA 打折"],
              ] as const
            ).map(([value, label]) => (
              <label key={value}>
                <input
                  type="radio"
                  name="lesson43-recipe"
                  value={value}
                  checked={recipe === value}
                  onChange={() => {
                    setRecipe(value);
                    invalidate();
                  }}
                />
                <span>{label}</span>
              </label>
            ))}
          </fieldset>
          <p className={styles.hint}>
            听写预训和污染 mask 会在执行列复读。揭晓前不显示输出、交叉熵和有效集合。
          </p>
        </form>

        <div className={styles.stage}>
          <div className={styles.lanes} aria-hidden="true">
            {TOKEN_LANES.map((lane) => (
              <div
                key={lane.id}
                className={styles.lane}
                data-kind={
                  ran && recipe === "contaminated" && lane.kind !== "cond"
                    ? "mix"
                    : lane.kind
                }
                style={{ flex: lane.count }}
              >
                <b>{lane.label}</b>
                <span>{lane.count}</span>
              </div>
            ))}
          </div>
          <dl className={styles.metrics}>
            <div>
              <dt>转写输出</dt>
              <dd>{ran ? result.asrOutput : "—"}</dd>
            </div>
            <div>
              <dt>执行输出</dt>
              <dd>{ran ? result.ifOutput : "—"}</dd>
            </div>
            <div>
              <dt>转写对？</dt>
              <dd>{ran ? (result.asrOk ? "对" : "错") : "—"}</dd>
            </div>
            <div>
              <dt>指令对？</dt>
              <dd>{ran ? (result.ifOk ? "对" : "错") : "—"}</dd>
            </div>
            <div>
              <dt>ASR 有效集合</dt>
              <dd>{ran ? `${result.asrSet} (${result.asrCount})` : "—"}</dd>
            </div>
            <div>
              <dt>指令有效集合</dt>
              <dd>{ran ? `${result.ifSet} (${result.ifCount})` : "—"}</dd>
            </div>
            <div>
              <dt>ASR CE</dt>
              <dd>{ran ? result.asrCe.toFixed(3) : "—"}</dd>
            </div>
            <div>
              <dt>指令 CE</dt>
              <dd>{ran ? result.ifCe.toFixed(3) : "—"}</dd>
            </div>
            <div>
              <dt>集合相等？</dt>
              <dd>
                {ran
                  ? result.asrCount === result.ifCount &&
                    result.asrSet === result.ifSet
                    ? "是（污染）"
                    : "否"
                  : "—"}
              </dd>
            </div>
          </dl>
        </div>
      </div>

      <div className={styles.predict}>
        <fieldset>
          <legend>先预测：揭晓后两列会怎样？</legend>
          {(
            [
              ["both-ok", "两边都对"],
              ["asr-ok-if-bad", "转写对、指令错"],
              ["asr-bad-if-ok", "转写错、指令对"],
              ["both-bad", "两边都错"],
            ] as const
          ).map(([value, label]) => (
            <label key={value}>
              <input
                type="radio"
                name="lesson43-prediction"
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
            揭晓两头输出
          </button>
        </div>
      </div>

      {ran && (
        <p className={styles.feedback}>
          {mismatch
            ? "转写对、指令错：听写参考命中，执行回复仍是复读或污染碎片。这是本课必须构造的一例。"
            : actual === "both-ok"
              ? "当前配方下两列都等于参考。切到听写预训或污染 mask，再选执行句，才能看到指令失败。"
              : actual === "asr-bad-if-ok"
                ? "LoRA 打折后听写出现替换，执行头反而能跟。对照 SALMONN 缩小缩放因子的方向。"
                : "两列都偏离参考。换一句语音或换配方。"}
          {prediction
            ? prediction === actual
              ? " 你的预测与揭晓一致。"
              : " 你的预测与揭晓不一致。"
            : ""}{" "}
          {result.maskNote}
          {setsEqual ? " 污染条件下两张有效集合变成同一段 11-18。" : ""}
        </p>
      )}

      <ul className={styles.checklist}>
        <li data-done={foundMismatch ? "true" : "false"}>
          找到一例：转写对、指令错
        </li>
        <li data-done={ran && prediction ? "true" : "false"}>
          先选预测再揭晓
        </li>
      </ul>

      <Gate passed={passed}>
        必须构造转写对、指令错的一例。听写预训或污染
        mask 加任意一句语音即可触发。数字由规则算出，不是 Qwen
        权重的前向输出。
      </Gate>
    </LabFrame>
  );
}
