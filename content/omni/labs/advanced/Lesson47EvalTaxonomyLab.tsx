"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import { stringFrom, useCompletionGate } from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson47EvalTaxonomyLab.module.css";

type CardId =
  | "mmmu"
  | "videomme"
  | "omnibench"
  | "osworld"
  | "libero"
  | "simpler";

type BucketId =
  | "expert_static"
  | "video_temporal"
  | "tri_modal"
  | "computer_exec"
  | "sim_manip"
  | "sim2real_rank"
  | "real_robot";

type PredictionId = "mmmu" | "libero" | "osworld" | "simpler" | "";

const CARDS: {
  id: CardId;
  title: string;
  figure: string;
  correct: Exclude<BucketId, "real_robot">;
  measures: string;
  notMeasures: string;
}[] = [
  {
    id: "mmmu",
    title: "MMMU test",
    figure: "GPT-4V 55.7%",
    correct: "expert_static",
    measures: "大学学科静态图文准确率",
    notMeasures: "不测视频、音频同时推理、接地框或操作",
  },
  {
    id: "videomme",
    title: "Video-MME",
    figure: "Gemini 1.5 Pro 75.0% 无字幕",
    correct: "video_temporal",
    measures: "短中长视频多项选择",
    notMeasures: "字幕是可选增益，不是三模态硬约束",
  },
  {
    id: "omnibench",
    title: "OmniBench",
    figure: "Qwen2.5-Omni-7B 56.13%",
    correct: "tri_modal",
    measures: "图+声+文缺一不可的准确率",
    notMeasures: "不测长视频执行，也不测电脑或机械臂",
  },
  {
    id: "osworld",
    title: "OSWorld",
    figure: "GPT-4 a11y 12.24%",
    correct: "computer_exec",
    measures: "真实电脑任务的脚本成功率",
    notMeasures: "不是试卷准确率，也不是仿真抓取",
  },
  {
    id: "libero",
    title: "LIBERO 四套件平均",
    figure: "OpenVLA FT 76.5%",
    correct: "sim_manip",
    measures: "仿真桌面四套件成功率（fine-tune）",
    notMeasures: "不是真机能力，Long 只有 53.7%",
  },
  {
    id: "simpler",
    title: "SIMPLER VisMatch",
    figure: "Pearson r = 0.924",
    correct: "sim2real_rank",
    measures: "仿真排序与真机排序相关",
    notMeasures: "单位是 r，不是成功率，更不是真机百分数",
  },
];

const BUCKETS: { id: BucketId; title: string; hint: string }[] = [
  { id: "expert_static", title: "专家静态图文", hint: "C1 试卷 + 异构图" },
  { id: "video_temporal", title: "视频时序理解", hint: "C2 短 / 中 / 长" },
  { id: "tri_modal", title: "三模态同时推理", hint: "C3 图声文缺一不可" },
  { id: "computer_exec", title: "计算机执行", hint: "C4 脚本判定 0/1" },
  { id: "sim_manip", title: "仿真操作", hint: "C5 套件成功率" },
  { id: "sim2real_rank", title: "仿真-真机排序", hint: "C6 Pearson r / MMRV" },
  { id: "real_robot", title: "真机能力", hint: "陷阱格：LIBERO 拖进来必须标红" },
];

const PREDICTIONS: { value: Exclude<PredictionId, "">; label: string }[] = [
  { value: "mmmu", label: "MMMU 55.7% 最常被填进真机能力" },
  { value: "libero", label: "LIBERO 平均 76.5% 最常被填进真机能力" },
  { value: "osworld", label: "OSWorld 12.24% 最常被填进真机能力" },
  { value: "simpler", label: "SIMPLER r=0.924 最常被填进真机能力" },
];

function parsePlacements(raw: string): Record<CardId, BucketId | ""> {
  const empty: Record<CardId, BucketId | ""> = {
    mmmu: "",
    videomme: "",
    omnibench: "",
    osworld: "",
    libero: "",
    simpler: "",
  };
  if (!raw) {
    return empty;
  }
  try {
    const parsed = JSON.parse(raw) as Record<string, string>;
    for (const card of CARDS) {
      const value = parsed[card.id];
      if (BUCKETS.some((bucket) => bucket.id === value) || value === "") {
        empty[card.id] = value as BucketId | "";
      }
    }
  } catch {
    return empty;
  }
  return empty;
}

export function Lesson47EvalTaxonomyLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    prediction: stringFrom(initialState, "prediction", "") as PredictionId,
    placements: parsePlacements(stringFrom(initialState, "placements", "")),
    notCross: stringFrom(initialState, "notCross", "") === "yes",
  };
  const [prediction, setPrediction] = useState<PredictionId>(defaults.prediction);
  const [placements, setPlacements] = useState<Record<CardId, BucketId | "">>(
    defaults.placements,
  );
  const [selected, setSelected] = useState<CardId | "">("");
  const [ran, setRan] = useState(false);
  const [notCross, setNotCross] = useState(defaults.notCross);

  const liberoInTrap = placements.libero === "real_robot";
  const allPlaced = CARDS.every((card) => placements[card.id] !== "");
  const allCorrect = CARDS.every(
    (card) => placements[card.id] === card.correct,
  );

  const passed =
    ran &&
    prediction === "libero" &&
    allCorrect &&
    !liberoInTrap &&
    notCross;

  const completion = useMemo(
    () => ({
      lessonId: 47,
      prediction,
      placements,
      liberoInTrap,
      allCorrect,
      notCross,
    }),
    [allCorrect, liberoInTrap, notCross, placements, prediction],
  );
  useCompletionGate(passed, onComplete, completion);

  function placeCard(cardId: CardId, bucketId: BucketId) {
    setPlacements((current) => ({ ...current, [cardId]: bucketId }));
    setSelected("");
    setRan(false);
  }

  function reset() {
    setPrediction("");
    setPlacements({
      mmmu: "",
      videomme: "",
      omnibench: "",
      osworld: "",
      libero: "",
      simpler: "",
    });
    setSelected("");
    setRan(false);
    setNotCross(false);
  }

  const unplaced = CARDS.filter((card) => placements[card.id] === "");

  return (
    <LabFrame
      lesson="47"
      title="六张评测卡拖进桶"
      description="教学模拟，不是模型输出。先预测哪张卡最常被误填成真机能力，再把六张卡拖进桶。把 LIBERO 平均拖进真机能力必须标红。揭晓前不显示对错。"
    >
      <div className={styles.workspace}>
        <form
          className={styles.controls}
          onSubmit={(event) => event.preventDefault()}
        >
          <h3>分桶规则</h3>
          <p className={styles.note}>
            六类互斥：专家静态图文、视频时序、三模态同时、计算机执行、仿真操作、仿真-真机排序。真机能力是陷阱格。
          </p>
          <p className={styles.note}>
            先在下方选出预测，再拖卡或点选卡片后点桶。数字来自已打开的原文表，不能兑成一个 Omni 均分。
          </p>
        </form>
        <div className={styles.stage}>
          <div
            className={styles.pool}
            onDragOver={(event) => event.preventDefault()}
            onDrop={() => {
              if (selected) {
                setPlacements((current) => ({ ...current, [selected]: "" }));
                setSelected("");
                setRan(false);
              }
            }}
          >
            {unplaced.length === 0 ? (
              <span className={styles.poolEmpty}>卡已全部入桶。可拖回这里改放。</span>
            ) : (
              unplaced.map((card) => (
                <button
                  key={card.id}
                  type="button"
                  draggable
                  className={`${styles.card} ${
                    selected === card.id ? styles.selected : ""
                  }`}
                  onClick={() =>
                    setSelected((current) => (current === card.id ? "" : card.id))
                  }
                  onDragStart={() => setSelected(card.id)}
                >
                  <b>{card.title}</b>
                  <span>{card.figure}</span>
                </button>
              ))
            )}
          </div>
          <div className={styles.buckets} aria-label="七个评测桶">
            {BUCKETS.map((bucket) => {
              const occupants = CARDS.filter(
                (card) => placements[card.id] === bucket.id,
              );
              const trapHot = bucket.id === "real_robot" && liberoInTrap;
              const revealedCorrect =
                ran &&
                bucket.id !== "real_robot" &&
                occupants.length > 0 &&
                occupants.every((card) => card.correct === bucket.id);
              return (
                <section
                  key={bucket.id}
                  className={`${styles.bucket} ${
                    trapHot ? styles.trap : ""
                  } ${revealedCorrect ? styles.correctBucket : ""}`}
                  onDragOver={(event) => event.preventDefault()}
                  onDrop={() => {
                    if (selected) {
                      placeCard(selected, bucket.id);
                    }
                  }}
                  onClick={() => {
                    if (selected) {
                      placeCard(selected, bucket.id);
                    }
                  }}
                >
                  <header>
                    <b>{bucket.title}</b>
                    <span>{bucket.hint}</span>
                  </header>
                  {occupants.map((card) => {
                    const illegal = card.id === "libero" && bucket.id === "real_robot";
                    const ok = ran && card.correct === bucket.id;
                    return (
                      <button
                        key={card.id}
                        type="button"
                        draggable
                        className={`${styles.card} ${
                          selected === card.id ? styles.selected : ""
                        } ${illegal ? styles.illegal : ""} ${
                          ok ? styles.ok : ""
                        }`}
                        onClick={(event) => {
                          event.stopPropagation();
                          setSelected((current) =>
                            current === card.id ? "" : card.id,
                          );
                        }}
                        onDragStart={(event) => {
                          event.stopPropagation();
                          setSelected(card.id);
                        }}
                      >
                        <b>{card.title}</b>
                        <span>
                          {card.figure}
                          {illegal ? " 标红：不是真机" : ""}
                        </span>
                      </button>
                    );
                  })}
                </section>
              );
            })}
          </div>
        </div>
      </div>

      <div className={styles.predict}>
        <fieldset>
          <legend>先预测：哪张卡最常被误填进“真机能力”</legend>
          {PREDICTIONS.map((option) => (
            <label key={option.value}>
              <input
                type="radio"
                name="eval-taxonomy-prediction"
                value={option.value}
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
        <label className={styles.ack}>
          <input
            type="checkbox"
            checked={notCross}
            onChange={(event) => {
              setNotCross(event.target.checked);
            }}
          />
          <span>六类数字不能横着比，也不能兑成一个总平均</span>
        </label>
        <div className={styles.actions}>
          <button type="button" className={styles.reset} onClick={reset}>
            重置
          </button>
          <button
            type="button"
            className={styles.run}
            disabled={!prediction || !allPlaced}
            onClick={() => setRan(true)}
          >
            揭晓分桶
          </button>
        </div>
      </div>

      {liberoInTrap && (
        <p className={styles.feedback}>
          LIBERO 四套件平均拖进了“真机能力”，这一格必须标红。76.5% 是
          robosuite 仿真、独立 fine-tune 之后的宏平均，Long 只有 53.7%。把它拖回仿真操作。
        </p>
      )}
      {ran && prediction !== "libero" && (
        <p className={styles.feedback}>
          预测应选 LIBERO 平均。试卷准确率、电脑执行成功率和 Pearson r
          也会被误用，但把仿真宏平均写成真机，是报告里最常见的跨类记账。
        </p>
      )}
      {ran && prediction === "libero" && allCorrect && !notCross && (
        <p className={styles.feedback}>
          六张卡都进了正确的桶。还需要勾选：六类数字不能横着比。
        </p>
      )}
      {ran && (
        <dl className={styles.verdict}>
          {CARDS.map((card) => (
            <div key={card.id}>
              <dt>{card.title}</dt>
              <dd>
                {card.figure}。测：{card.measures}。不测：{card.notMeasures}。
              </dd>
            </div>
          ))}
        </dl>
      )}
      <Gate passed={passed}>
        预测选中 LIBERO、六张卡进入互斥桶、真机陷阱为空，并声明不能横着比。LIBERO
        若被拖进真机能力，卡片与桶保持标红，验收不通过。
      </Gate>
    </LabFrame>
  );
}
