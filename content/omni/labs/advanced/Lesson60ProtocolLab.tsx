"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import { stringFrom, useCompletionGate } from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson60ProtocolLab.module.css";

type CardId =
  | "libero_as_real"
  | "spatial_ok"
  | "missing_n"
  | "missing_suite"
  | "simpler_r"
  | "scale_13b";

type BucketId =
  | "sim_manip"
  | "sim2real_rank"
  | "real_robot"
  | "incomplete"
  | "scale";

type PredictionId = "only_real" | "triple" | "simpler_scale" | "all_six" | "";

const CARDS: {
  id: CardId;
  title: string;
  figure: string;
  flags: string[];
  missing: Array<"n" | "suite">;
  claimsReal: boolean;
  correct: BucketId;
  measures: string;
  notMeasures: string;
}[] = [
  {
    id: "libero_as_real",
    title: "Table 2 行 A",
    figure: "真机成功率 81%",
    flags: ["LIBERO 四套件平均", "声称真机"],
    missing: [],
    claimsReal: true,
    correct: "sim_manip",
    measures: "仿真桌面宏平均，类标签 C5",
    notMeasures: "不是真机能力；拖进真机格必须标红",
  },
  {
    id: "spatial_ok",
    title: "Table 2 行 B",
    figure: "Spatial 88%",
    flags: ["N=500", "suite=spatial", "FT"],
    missing: [],
    claimsReal: false,
    correct: "sim_manip",
    measures: "LIBERO-Spatial 谓词成功率",
    notMeasures: "不能代替 Long，也不能代替真机",
  },
  {
    id: "missing_n",
    title: "Table 2 行 C",
    figure: "Long 高",
    flags: ["缺 N", "suite=long"],
    missing: ["n"],
    claimsReal: false,
    correct: "incomplete",
    measures: "字段不全，只能拒收",
    notMeasures: "没有 N 就不能写 Wilson，也不能入 C5 主列",
  },
  {
    id: "missing_suite",
    title: "Table 2 行 D",
    figure: "LIBERO 90%",
    flags: ["缺套件", "N=500"],
    missing: ["suite"],
    claimsReal: false,
    correct: "incomplete",
    measures: "字段不全，只能拒收",
    notMeasures: "缺 Spatial/Object/Goal/Long 就不能和 Table 12 对照",
  },
  {
    id: "simpler_r",
    title: "Table 2 行 E",
    figure: "VisMatch r=0.81",
    flags: ["pearson_r", "n=6 政策"],
    missing: [],
    claimsReal: false,
    correct: "sim2real_rank",
    measures: "仿真排序与真机排序相关，类标签 C6",
    notMeasures: "单位是 r，不是成功率",
  },
  {
    id: "scale_13b",
    title: "Table 2 行 F",
    figure: "13B 比 8B +4 点",
    flags: ["同一离散 token", "规模"],
    missing: [],
    claimsReal: false,
    correct: "scale",
    measures: "参数变大，接到第 27 课当对照",
    notMeasures: "不新开模型课，缩小版复现不了这 4 个点",
  },
];

const BUCKETS: { id: BucketId; title: string; hint: string }[] = [
  { id: "sim_manip", title: "仿真操作 C5", hint: "第 31 / 47 课套件桶" },
  { id: "sim2real_rank", title: "仿真-真机排序 C6", hint: "Pearson r / MMRV" },
  { id: "real_robot", title: "真机能力", hint: "陷阱格：LIBERO 拖进来必须标红" },
  { id: "incomplete", title: "缺字段拒收", hint: "缺 N 或缺套件" },
  { id: "scale", title: "规模声明", hint: "不新开模型课" },
];

const PREDICTIONS: { value: Exclude<PredictionId, "">; label: string }[] = [
  { value: "only_real", label: "只有写成真机的那一行会标红" },
  {
    value: "triple",
    label: "缺 N、缺套件、把 LIBERO 写成真机，三张都会标红",
  },
  { value: "simpler_scale", label: "只有 SIMPLER 的 r 和 13B 规模行会标红" },
  { value: "all_six", label: "六张都会标红" },
];

function parsePlacements(raw: string): Record<CardId, BucketId | ""> {
  const empty: Record<CardId, BucketId | ""> = {
    libero_as_real: "",
    spatial_ok: "",
    missing_n: "",
    missing_suite: "",
    simpler_r: "",
    scale_13b: "",
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

function getCard(id: CardId) {
  const found = CARDS.find((card) => card.id === id);
  if (!found) {
    throw new Error(`unknown protocol card: ${id}`);
  }
  return found;
}

function isIllegal(card: (typeof CARDS)[number], bucket: BucketId | "") {
  if (!bucket) {
    return false;
  }
  if (card.claimsReal && bucket === "real_robot") {
    return true;
  }
  if (card.missing.length > 0 && bucket !== "incomplete") {
    return true;
  }
  return false;
}

export function Lesson60ProtocolLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    prediction: stringFrom(initialState, "prediction", "") as PredictionId,
    placements: parsePlacements(stringFrom(initialState, "placements", "")),
    noNewModel: stringFrom(initialState, "noNewModel", "") === "yes",
    labelsOk: stringFrom(initialState, "labelsOk", "") === "yes",
  };
  const [prediction, setPrediction] = useState<PredictionId>(defaults.prediction);
  const [placements, setPlacements] = useState<Record<CardId, BucketId | "">>(
    defaults.placements,
  );
  const [selected, setSelected] = useState<CardId | "">("");
  const [ran, setRan] = useState(false);
  const [noNewModel, setNoNewModel] = useState(defaults.noNewModel);
  const [labelsOk, setLabelsOk] = useState(defaults.labelsOk);

  const liberoInTrap = placements.libero_as_real === "real_robot";
  const missingNHot = isIllegal(getCard("missing_n"), placements.missing_n);
  const missingSuiteHot = isIllegal(
    getCard("missing_suite"),
    placements.missing_suite,
  );
  const allPlaced = CARDS.every((card) => placements[card.id] !== "");
  const allCorrect = CARDS.every(
    (card) => placements[card.id] === card.correct,
  );

  const passed =
    ran &&
    prediction === "triple" &&
    allCorrect &&
    !liberoInTrap &&
    noNewModel &&
    labelsOk;

  const completion = useMemo(
    () => ({
      lessonId: 60,
      prediction,
      placements,
      liberoInTrap,
      missingNHot,
      missingSuiteHot,
      allCorrect,
      noNewModel,
      labelsOk,
    }),
    [
      allCorrect,
      labelsOk,
      liberoInTrap,
      missingNHot,
      missingSuiteHot,
      noNewModel,
      placements,
      prediction,
    ],
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
      libero_as_real: "",
      spatial_ok: "",
      missing_n: "",
      missing_suite: "",
      simpler_r: "",
      scale_13b: "",
    });
    setSelected("");
    setRan(false);
    setNoNewModel(false);
    setLabelsOk(false);
  }

  const unplaced = CARDS.filter((card) => placements[card.id] === "");

  return (
    <LabFrame
      lesson="60"
      title="把虚构的新 VLA 表拖进桶"
      description="教学模拟，不是模型输出。表来自虚构的 NovaVLA-8B，不得当文献引用。先预测哪几张卡会标红，再把六行拖进桶。缺 N、缺套件、把 LIBERO 写成真机必须标红。揭晓前不显示对错。"
    >
      <div className={styles.workspace}>
        <form
          className={styles.controls}
          onSubmit={(event) => event.preventDefault()}
        >
          <h3>收编规则</h3>
          <p className={styles.note}>
            本课不引入新模型。一行数字一张卡：课桶、规模或机制、第 47 课类标签、第
            31 课套件、N、单位。
          </p>
          <p className={styles.note}>
            先选预测，再拖卡或点选后点桶。缺字段进「缺字段拒收」。LIBERO
            平均进「真机能力」必须变红。
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
              <span className={styles.poolEmpty}>
                行已全部入桶。可拖回这里改放。
              </span>
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
                  <span className={styles.flags}>
                    {card.flags.map((flag) => (
                      <span
                        key={flag}
                        className={`${styles.flag} ${
                          flag.includes("缺") || flag.includes("真机")
                            ? styles.flagHot
                            : ""
                        }`}
                      >
                        {flag}
                      </span>
                    ))}
                  </span>
                </button>
              ))
            )}
          </div>
          <div className={styles.buckets} aria-label="五个收编桶">
            {BUCKETS.map((bucket) => {
              const occupants = CARDS.filter(
                (card) => placements[card.id] === bucket.id,
              );
              const trapHot = bucket.id === "real_robot" && liberoInTrap;
              const revealedCorrect =
                ran &&
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
                    const illegal = isIllegal(card, bucket.id);
                    const ok = ran && card.correct === bucket.id && !illegal;
                    const redLabel = illegal
                      ? card.claimsReal
                        ? " 标红：不是真机"
                        : card.missing.includes("n")
                          ? " 标红：缺 N"
                          : " 标红：缺套件"
                      : "";
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
                          {redLabel}
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
          <legend>先预测：揭晓后哪几张卡必须标红</legend>
          {PREDICTIONS.map((option) => (
            <label key={option.value}>
              <input
                type="radio"
                name="living-protocol-prediction"
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
            checked={noNewModel}
            onChange={(event) => {
              setNoNewModel(event.target.checked);
            }}
          />
          <span>不引入新模型，只引入收编规则</span>
        </label>
        <label className={styles.ack}>
          <input
            type="checkbox"
            checked={labelsOk}
            onChange={(event) => {
              setLabelsOk(event.target.checked);
            }}
          />
          <span>类标签与第 31、47 课兼容：C5 要套件，C6 单位是 r，LIBERO 不是真机</span>
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
            揭晓收编
          </button>
        </div>
      </div>

      {liberoInTrap && (
        <p className={styles.feedback}>
          LIBERO 四套件平均拖进了「真机能力」，这一格必须标红。81%
          是教学虚构的仿真宏平均，和 OpenVLA Table 12 的 76.5%
          同一类错误：C5 进不了真机。把它拖回仿真操作。
        </p>
      )}
      {missingNHot && (
        <p className={styles.feedback}>
          「Long 高」缺 N。没有试验次数就不能写 Wilson 区间，也不能和 500 trials
          的格子比。这张卡必须进「缺字段拒收」，放进成功桶保持标红。
        </p>
      )}
      {missingSuiteHot && (
        <p className={styles.feedback}>
          「LIBERO 90%」缺套件。第 31 课的 Spatial / Object / Goal / Long
          键对不上，宏平均会把 Long 兑掉。这张卡必须进「缺字段拒收」。
        </p>
      )}
      {ran && prediction !== "triple" && (
        <p className={styles.feedback}>
          预测应选三张都会标红。只盯真机陷阱会漏掉空 N 和空套件；SIMPLER 的 r
          只要单位写对就不标红；13B 规模行进规模桶，不标红。
        </p>
      )}
      {ran && prediction === "triple" && allCorrect && (!noNewModel || !labelsOk) && (
        <p className={styles.feedback}>
          六行都进了正确的桶。还需要勾选：不新开模型课，以及标签与第 31、47 课兼容。
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
        预测选中三张标红、六行进入正确桶、真机陷阱没有 LIBERO，并声明不新开模型课、标签与第
        31、47 课兼容。缺 N、缺套件若放进成功桶，卡片保持标红，验收不通过。
      </Gate>
    </LabFrame>
  );
}
