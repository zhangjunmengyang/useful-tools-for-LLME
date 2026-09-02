"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import {
  numberFrom,
  stringFrom,
  useCompletionGate,
} from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson45MmRetrieveLab.module.css";

type LayerId = "subtitle" | "mid" | "pixel";
type Prediction = "subtitle" | "mid" | "pixel" | "all";

type Clip = {
  minute: number;
  kind: "speech" | "silent" | "target";
  asr: string;
  visual: string;
};

const MINUTES = 60;
const TARGET_MINUTE = 47;
const READER_BUDGET = 4096;
const TOKEN_SUBTITLE = 40;
const TOKEN_MID = 192;
const TOKEN_PER_FRAME = 64;
const DENSE_FRAMES = 60;

const QUERY = "第几分钟有人把红色阀门转了半圈？";

const PREDICTION_OPTIONS: { value: Prediction; label: string }[] = [
  { value: "subtitle", label: "只检索字幕就能命中，预算也够" },
  { value: "mid", label: "中间特征能命中，且精读预算够用" },
  { value: "pixel", label: "只检索像素就能命中，预算也够" },
  { value: "all", label: "三层都能命中，且都不超预算" },
];

const LAYER_LABEL: Record<LayerId, string> = {
  subtitle: "字幕层",
  mid: "中间特征",
  pixel: "像素层",
};

function buildClips(): Clip[] {
  return Array.from({ length: MINUTES }, (_, index) => {
    const minute = index + 1;
    if (minute === TARGET_MINUTE) {
      return {
        minute,
        kind: "target",
        asr: "",
        visual: "red_valve_turn",
      };
    }
    if (minute === 12) {
      return {
        minute,
        kind: "speech",
        asr: "把阀门再拧紧一点",
        visual: "talking_head",
      };
    }
    if (minute === 9 || minute === 23 || minute === 40) {
      return {
        minute,
        kind: "silent",
        asr: "",
        visual: "red_cup",
      };
    }
    if (minute % 5 === 0) {
      return {
        minute,
        kind: "speech",
        asr: "继续按清单检查管路",
        visual: "talking_head",
      };
    }
    return {
      minute,
      kind: minute % 7 === 3 ? "silent" : "speech",
      asr: minute % 7 === 3 ? "" : "当前压力正常",
      visual: "panel",
    };
  });
}

const CLIPS = buildClips();

function scoreClip(clip: Clip, layer: LayerId): number {
  if (layer === "subtitle") {
    if (clip.asr.includes("阀门")) return 0.92;
    if (clip.asr.includes("检查")) return 0.18;
    if (clip.asr) return 0.08;
    return 0.01;
  }
  if (layer === "mid") {
    if (clip.visual === "red_valve_turn") return 0.97;
    if (clip.visual === "red_cup") return 0.41;
    if (clip.visual === "talking_head") return 0.12;
    return 0.08;
  }
  if (clip.visual === "red_cup") return 0.88;
  if (clip.visual === "red_valve_turn") return 0.86;
  return 0.2 + (clip.minute % 9) * 0.01;
}

function retrieve(layer: LayerId, k: number, frames: number) {
  const ranked = [...CLIPS].sort((left, right) => {
    const delta = scoreClip(right, layer) - scoreClip(left, layer);
    return delta !== 0 ? delta : left.minute - right.minute;
  });
  const top = ranked.slice(0, k);
  const minutes = top.map((clip) => clip.minute);
  const hit = minutes.includes(TARGET_MINUTE);
  const cost =
    layer === "subtitle"
      ? k * TOKEN_SUBTITLE
      : layer === "mid"
        ? k * TOKEN_MID
        : k * frames * TOKEN_PER_FRAME;
  return { minutes, hit, cost };
}

export function Lesson45MmRetrieveLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    k: numberFrom(initialState, "k", 5, 1, 8),
    frames: numberFrom(initialState, "frames", 24, 4, 60),
    layer: stringFrom(initialState, "layer", "mid") as LayerId,
    prediction: stringFrom(initialState, "prediction", "") as Prediction | "",
  };
  const [k, setK] = useState(Math.round(defaults.k));
  const [frames, setFrames] = useState(Math.round(defaults.frames));
  const [layer, setLayer] = useState<LayerId>(
    ["subtitle", "mid", "pixel"].includes(defaults.layer)
      ? defaults.layer
      : "mid",
  );
  const [prediction, setPrediction] = useState<Prediction | "">(
    defaults.prediction,
  );
  const [ran, setRan] = useState(false);

  const rows = useMemo(() => {
    const subtitle = retrieve("subtitle", k, frames);
    const mid = retrieve("mid", k, frames);
    const pixel = retrieve("pixel", k, frames);
    const pixelDense = retrieve("pixel", k, DENSE_FRAMES);
    return { subtitle, mid, pixel, pixelDense };
  }, [frames, k]);

  const inspected = rows[layer];
  const subtitleMiss = !rows.subtitle.hit;
  const pixelBlows = rows.pixelDense.cost > READER_BUDGET;
  const midWorks = rows.mid.hit && rows.mid.cost <= READER_BUDGET;

  const passed =
    ran &&
    prediction === "mid" &&
    subtitleMiss &&
    pixelBlows &&
    midWorks;

  const completion = useMemo(
    () => ({
      lessonId: 45,
      k,
      frames,
      layer,
      prediction,
      subtitleHit: rows.subtitle.hit,
      midHit: rows.mid.hit,
      pixelHit: rows.pixel.hit,
      subtitleCost: rows.subtitle.cost,
      midCost: rows.mid.cost,
      pixelCost: rows.pixel.cost,
      pixelDenseCost: rows.pixelDense.cost,
      readerBudget: READER_BUDGET,
    }),
    [frames, k, layer, prediction, rows],
  );
  useCompletionGate(passed, onComplete, completion);

  function reset() {
    setK(Math.round(defaults.k));
    setFrames(Math.round(defaults.frames));
    setLayer("mid");
    setPrediction("");
    setRan(false);
  }

  return (
    <LabFrame
      lesson="45"
      title="一小时时间线：先选层，再看召回"
      description="教学模拟，不是模型输出。查询固定为无对白的红色阀门动作，目标在第 47 分钟。先预测哪一层能在预算内召回，再揭晓三层的 Recall 和 token 账单。"
    >
      <div className={styles.workspace}>
        <aside className={styles.controls}>
          <h3>检索设置</h3>
          <fieldset>
            <legend>当前查看的层</legend>
            {(["subtitle", "mid", "pixel"] as LayerId[]).map((id) => (
              <label key={id}>
                <input
                  type="radio"
                  name="lesson45-layer"
                  checked={layer === id}
                  onChange={() => {
                    setLayer(id);
                    setRan(false);
                  }}
                />
                {LAYER_LABEL[id]}
              </label>
            ))}
          </fieldset>
          <label>
            <span>
              Top-k
              <output>{k}</output>
            </span>
            <input
              type="range"
              min={1}
              max={8}
              step={1}
              value={k}
              onChange={(event) => {
                setK(Number(event.target.value));
                setRan(false);
              }}
            />
          </label>
          <label>
            <span>
              像素层每分钟帧数
              <output>{frames}</output>
            </span>
            <input
              type="range"
              min={4}
              max={60}
              step={2}
              value={frames}
              onChange={(event) => {
                setFrames(Number(event.target.value));
                setRan(false);
              }}
            />
          </label>
          <p className={styles.note}>
            阅读器预算固定 {READER_BUDGET} token。字幕每段 {TOKEN_SUBTITLE}，中间特征每段 {TOKEN_MID}，像素层每帧 {TOKEN_PER_FRAME}。密采样对照固定每分钟 {DENSE_FRAMES} 帧。
          </p>
        </aside>
        <div className={styles.stage}>
          <p className={styles.query}>
            查询：<strong>{QUERY}</strong>
            {" "}目标时刻已写入夹具，揭晓前不显示是否命中。
          </p>
          <div className={styles.timeline}>
            <div className={styles.axis} aria-label="六十分钟时间线">
              {CLIPS.map((clip) => {
                const retrieved = ran && inspected.minutes.includes(clip.minute);
                const missedTarget =
                  ran && clip.kind === "target" && !inspected.hit;
                const className = [
                  styles.tick,
                  clip.kind === "speech" ? styles.tickSpeech : "",
                  clip.kind === "silent" ? styles.tickSilent : "",
                  clip.kind === "target" ? styles.tickTarget : "",
                  retrieved ? styles.tickRetrieved : "",
                  missedTarget ? styles.tickMissed : "",
                ]
                  .filter(Boolean)
                  .join(" ");
                return (
                  <span
                    key={clip.minute}
                    className={className}
                    title={`第 ${clip.minute} 分钟`}
                  />
                );
              })}
            </div>
            <div className={styles.legend}>
              <span>
                <i className={styles.legendSpeech} />
                有对白
              </span>
              <span>
                <i className={styles.legendSilent} />
                无对白画面
              </span>
              <span>
                <i className={styles.legendTarget} />
                目标段位置
              </span>
            </div>
          </div>
          <dl className={styles.metrics}>
            <div>
              <dt>当前层 Recall@k</dt>
              <dd>
                {ran ? (inspected.hit ? "1.00" : "0.00") : "—"}
              </dd>
            </div>
            <div>
              <dt>当前层账单</dt>
              <dd>{ran ? inspected.cost : "—"}</dd>
            </div>
            <div>
              <dt>是否超预算</dt>
              <dd>
                {ran ? (inspected.cost > READER_BUDGET ? "超" : "否") : "—"}
              </dd>
            </div>
          </dl>
          {ran ? (
            <div className={styles.table}>
              <header>
                <span>层</span>
                <span>命中</span>
                <span>token</span>
                <span>召回的分钟</span>
              </header>
              {(
                [
                  ["字幕层", rows.subtitle],
                  ["中间特征", rows.mid],
                  ["像素层(当前帧率)", rows.pixel],
                  ["像素层(密采样)", rows.pixelDense],
                ] as const
              ).map(([name, row]) => (
                <article key={name}>
                  <span>{name}</span>
                  <span className={row.hit ? styles.hit : styles.miss}>
                    {row.hit ? "命中 47" : "未中"}
                  </span>
                  <span className={row.cost > READER_BUDGET ? styles.miss : styles.hit}>
                    {row.cost}
                  </span>
                  <span>{row.minutes.join(", ")}</span>
                </article>
              ))}
            </div>
          ) : null}
          <div className={styles.predict}>
            <fieldset>
              <legend>揭晓前预测：哪一种说法成立？</legend>
              {PREDICTION_OPTIONS.map((option) => (
                <label key={option.value}>
                  <input
                    type="radio"
                    name="lesson45-prediction"
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
                揭晓召回
              </button>
            </div>
          </div>
          {!prediction ? (
            <p className={styles.feedback}>先选预测，再揭晓三层数字。</p>
          ) : null}
          {ran && prediction !== "mid" ? (
            <p className={styles.feedback}>
              预测未对准夹具。字幕层被第 12 分钟的口头“阀门”带走；密采样像素层在 k={k} 时账单 {rows.pixelDense.cost}，超过 {READER_BUDGET}。
            </p>
          ) : null}
          {ran && prediction === "mid" && !pixelBlows ? (
            <p className={styles.feedback}>
              中间层已命中，但密采样像素层在 k={k} 时账单 {rows.pixelDense.cost}，尚未超过 {READER_BUDGET}。把 k 调到 2 或以上再揭晓，才能看到只检索像素会爆预算。
            </p>
          ) : null}
          {ran && prediction === "mid" && pixelBlows ? (
            <p className={styles.reveal}>
              字幕 Recall@k = {rows.subtitle.hit ? 1 : 0}，命中分钟 {rows.subtitle.minutes.join("/")}，没有第 47 分钟。中间特征命中第 47 分钟，账单 {rows.mid.cost}。密采样像素层账单 {rows.pixelDense.cost}，超过阅读预算。
            </p>
          ) : null}
        </div>
      </div>
      <Gate passed={passed}>
        先提交“中间特征能命中且预算够用”，再揭晓：只检索字幕错过无对白动作，只检索像素（密采样）爆预算。
      </Gate>
    </LabFrame>
  );
}
