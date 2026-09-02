"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import { round, stringFrom, useCompletionGate } from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson59SoundLab.module.css";

type CodecId = "speech8" | "music50" | "q0only";
type PredictionId =
  | "both-ok"
  | "speech-ok-drum-collapse"
  | "speech-bad-drum-ok"
  | "both-bad";

type CodecResult = {
  speechWer: number;
  drumF1: number;
  drumPrecision: number;
  drumRecall: number;
  nPredOnsets: number;
  nRefOnsets: number;
  speechText: string;
  drumNote: string;
  codebookNote: string;
  speechOk: boolean;
  drumCollapsed: boolean;
};

const SPEECH_WORDS = ["ming", "tian", "ba", "dian", "kai", "hui"] as const;
const DRUM_REF = 16;
const WER_PASS = 0.2;
const F1_COLLAPSE = 0.7;

const CODECS: Record<
  CodecId,
  { label: string; lanes: number; frameMs: number }
> = {
  speech8: { label: "语音 8 路 / 80 ms", lanes: 8, frameMs: 80 },
  music50: { label: "音乐 4 路 / 20 ms", lanes: 4, frameMs: 20 },
  q0only: { label: "只留第 0 路", lanes: 8, frameMs: 80 },
};

function simulate(codec: CodecId): CodecResult {
  if (codec === "speech8") {
    return {
      speechWer: 0,
      drumF1: 2 / 3,
      drumPrecision: 1,
      drumRecall: 0.5,
      nPredOnsets: 8,
      nRefOnsets: DRUM_REF,
      speechText: "明天八点开会",
      drumNote: "8 组 flam 并进 8 个 80 ms 帧心，16 个真 onset 只保住一半。",
      codebookNote: "8 路码本、12.5 Hz。语音音节跨 2 帧仍可懂，鼓点双击共用一格。",
      speechOk: true,
      drumCollapsed: true,
    };
  }
  if (codec === "music50") {
    return {
      speechWer: 0,
      drumF1: 1,
      drumPrecision: 1,
      drumRecall: 1,
      nPredOnsets: 16,
      nRefOnsets: DRUM_REF,
      speechText: "明天八点开会",
      drumNote: "20 ms 帧把 20 ms 间距的双击拆开，16 个 onset 全部命中。",
      codebookNote: "对照用的音乐 tokenizer：50 Hz、4 路。本课验收不靠这一臂。",
      speechOk: true,
      drumCollapsed: false,
    };
  }
  return {
    speechWer: 0,
    drumF1: 0.4,
    drumPrecision: 1,
    drumRecall: 0.25,
    nPredOnsets: 4,
    nRefOnsets: DRUM_REF,
    speechText: "明天八点开会",
    drumNote: "瞬态残差丢掉后只剩 4 个可检出峰值，网格更稀。",
    codebookNote: "仍是那 8 本字典，解码只用第 0 路。语音轮廓还在，鼓点更糊。",
    speechOk: true,
    drumCollapsed: true,
  };
}

function outcomeOf(result: CodecResult): PredictionId {
  if (result.speechOk && !result.drumCollapsed) return "both-ok";
  if (result.speechOk && result.drumCollapsed) return "speech-ok-drum-collapse";
  if (!result.speechOk && !result.drumCollapsed) return "speech-bad-drum-ok";
  return "both-bad";
}

function sharedScore(wer: number, f1: number) {
  return (1 - wer + f1) / 2;
}

const SPEECH_OCCUPANCY = [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0];
const DRUM_SPEECH8 = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
const DRUM_MUSIC = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
const DRUM_Q0 = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0];

function drumOccupancy(codec: CodecId) {
  if (codec === "music50") return DRUM_MUSIC;
  if (codec === "q0only") return DRUM_Q0;
  return DRUM_SPEECH8;
}

export function Lesson59SoundLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    codec: stringFrom(initialState, "codec", "speech8") as CodecId,
    prediction: stringFrom(initialState, "prediction", "") as PredictionId | "",
  };
  const [codec, setCodec] = useState<CodecId>(
    ["speech8", "music50", "q0only"].includes(defaults.codec)
      ? defaults.codec
      : "speech8",
  );
  const [prediction, setPrediction] = useState<PredictionId | "">(
    defaults.prediction,
  );
  const [ran, setRan] = useState(false);
  const [foundCollapse, setFoundCollapse] = useState(false);

  const result = useMemo(() => simulate(codec), [codec]);
  const actual = outcomeOf(result);
  const shared = sharedScore(result.speechWer, result.drumF1);
  const drums = drumOccupancy(codec);
  const spec = CODECS[codec];

  const passed = foundCollapse;
  const completion = useMemo(
    () => ({
      lessonId: 59,
      codec,
      prediction,
      speechWer: round(result.speechWer, 5),
      drumF1: round(result.drumF1, 5),
      sharedScore: round(shared, 5),
      foundCollapse,
    }),
    [codec, foundCollapse, prediction, result.drumF1, result.speechWer, shared],
  );
  useCompletionGate(passed, onComplete, completion);

  function invalidate() {
    setRan(false);
  }

  function reset() {
    setCodec("speech8");
    setPrediction("");
    setRan(false);
    setFoundCollapse(false);
  }

  function run() {
    if (!prediction) return;
    const next = simulate(codec);
    if (next.speechOk && next.drumCollapsed) setFoundCollapse(true);
    setRan(true);
  }

  return (
    <LabFrame
      lesson="59"
      title="同一 8 路码本：语音句对鼓点网格"
      description="教学模拟，不是模型输出。固定同一套残差量化网格，先预测语音是否可懂、鼓点网格是否塌掉，再揭晓 WER 与事件 F1。两列标签不得合成一个分数。"
    >
      <div className={styles.workspace}>
        <form
          className={styles.controls}
          onSubmit={(event) => event.preventDefault()}
        >
          <h3>Codec 网格</h3>
          <fieldset className={styles.choiceSet}>
            <legend>同一套码本怎么切时间</legend>
            {(
              [
                ["speech8", "语音 8 路 / 80 ms"],
                ["q0only", "只留第 0 路"],
                ["music50", "音乐 4 路 / 20 ms"],
              ] as const
            ).map(([value, label]) => (
              <label key={value}>
                <input
                  type="radio"
                  name="lesson59-codec"
                  value={value}
                  checked={codec === value}
                  onChange={() => {
                    setCodec(value);
                    invalidate();
                  }}
                />
                <span>{label}</span>
              </label>
            ))}
          </fieldset>
          <p className={styles.heard}>语音句：明天八点开会</p>
          <p className={styles.hint}>
            鼓点：8 组相距 20 ms 的 flam，真值 16 个 onset。容差固定 40
            ms，不能靠放宽容差把网格救回来。揭晓前不显示 WER、F1
            和重建字。
          </p>
        </form>

        <div className={styles.stage}>
          <p className={styles.panelTitle}>
            {spec.lanes} 路 × 16 帧（{spec.frameMs} ms）
          </p>
          <div className={styles.dual}>
            <div>
              <p className={styles.stripLabel}>语音句</p>
              <div className={styles.grid} aria-hidden="true">
                {Array.from({ length: spec.lanes }, (_, lane) => (
                  <div key={`s-${lane}`} className={styles.row}>
                    {SPEECH_OCCUPANCY.map((on, frame) => (
                      <span
                        key={`s-${lane}-${frame}`}
                        className={styles.cell}
                        data-on={on ? "true" : "false"}
                        data-kind="speech"
                        data-dim={
                          codec === "q0only" && lane > 0 ? "true" : "false"
                        }
                      />
                    ))}
                  </div>
                ))}
              </div>
            </div>
            <div>
              <p className={styles.stripLabel}>鼓点网格</p>
              <div className={styles.grid} aria-hidden="true">
                {Array.from({ length: spec.lanes }, (_, lane) => (
                  <div key={`d-${lane}`} className={styles.row}>
                    {drums.map((on, frame) => (
                      <span
                        key={`d-${lane}-${frame}`}
                        className={styles.cell}
                        data-on={on ? "true" : "false"}
                        data-kind="drum"
                        data-dim={
                          codec === "q0only" && lane > 0 ? "true" : "false"
                        }
                      />
                    ))}
                  </div>
                ))}
              </div>
            </div>
          </div>
          <dl className={styles.metrics}>
            <div>
              <dt>语音重建</dt>
              <dd>{ran ? result.speechText : "—"}</dd>
            </div>
            <div>
              <dt>语音 WER</dt>
              <dd>
                {ran ? result.speechWer.toFixed(3) : "—"}
              </dd>
            </div>
            <div>
              <dt>鼓点事件 F1</dt>
              <dd>{ran ? result.drumF1.toFixed(3) : "—"}</dd>
            </div>
            <div>
              <dt>检出 / 真值 onset</dt>
              <dd>
                {ran ? `${result.nPredOnsets} / ${result.nRefOnsets}` : "—"}
              </dd>
            </div>
            <div>
              <dt>Precision / Recall</dt>
              <dd>
                {ran
                  ? `${result.drumPrecision.toFixed(2)} / ${result.drumRecall.toFixed(2)}`
                  : "—"}
              </dd>
            </div>
            <div>
              <dt>非法共用分 (1-WER+F1)/2</dt>
              <dd>{ran ? shared.toFixed(3) : "—"}</dd>
            </div>
          </dl>
        </div>
      </div>

      <div className={styles.predict}>
        <fieldset>
          <legend>先预测：揭晓后两列会怎样？</legend>
          {(
            [
              ["both-ok", "两边都过"],
              ["speech-ok-drum-collapse", "语音可懂、鼓点网格塌掉"],
              ["speech-bad-drum-ok", "语音不可懂、鼓点还在"],
              ["both-bad", "两边都坏"],
            ] as const
          ).map(([value, label]) => (
            <label key={value}>
              <input
                type="radio"
                name="lesson59-prediction"
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
            揭晓两列指标
          </button>
        </div>
      </div>

      {ran && (
        <p className={styles.feedback}>
          {result.speechOk && result.drumCollapsed
            ? "语音 WER 为 0，鼓点 F1 低于 0.70。可懂不等于网格还在。这是本课必须构造的一例。"
            : actual === "both-ok"
              ? "当前网格把 flam 拆开了。切回语音 8 路 / 80 ms，才能看到鼓点并帧。"
              : "当前两列都没有同时满足「语音可懂、鼓点塌掉」。"}
          {prediction
            ? prediction === actual
              ? " 你的预测与揭晓一致。"
              : " 你的预测与揭晓不一致。"
            : ""}{" "}
          {result.codebookNote} {result.drumNote}
          {ran && shared >= 0.8 && result.drumF1 < F1_COLLAPSE
            ? ` 共用分 ${shared.toFixed(3)} 会把塌掉的网格藏过去。`
            : ""}
        </p>
      )}

      <ul className={styles.checklist}>
        <li data-done={foundCollapse ? "true" : "false"}>
          找到一例：语音可懂（WER ≤ {WER_PASS}）、鼓点网格塌掉（F1 &lt;{" "}
          {F1_COLLAPSE}）
        </li>
        <li data-done={ran && prediction ? "true" : "false"}>
          先选预测再揭晓
        </li>
        <li data-done={ran ? "true" : "false"}>
          词序列 {SPEECH_WORDS.length} 个、onset {DRUM_REF} 个，标签家族不同
        </li>
      </ul>

      <Gate passed={passed}>
        必须构造语音重建可懂、鼓点网格塌掉的一例。语音 8 路 / 80
        ms，或只留第 0 路，都能触发。数字由帧对齐规则算出，不是
        MusicGen 或 Mimi 权重的前向输出。
      </Gate>
    </LabFrame>
  );
}
