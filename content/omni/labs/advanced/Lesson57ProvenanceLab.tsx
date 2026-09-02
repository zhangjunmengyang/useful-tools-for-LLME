"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import { stringFrom, useCompletionGate } from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson57ProvenanceLab.module.css";

type SampleId = "img-001" | "img-002" | "img-003";
type BucketId = "train" | "reject";
type PredictionId = "all" | "onlyA" | "ab" | "ac" | "cRejects";

type Sample = {
  id: SampleId;
  title: string;
  caption: string;
  license: string;
  sha256: string;
  sourceUrl: string;
  synthetic: boolean;
  retractable: boolean;
  correct: BucketId;
  rejectReason: string;
};

const SAMPLES: Sample[] = [
  {
    id: "img-001",
    title: "杯子静物",
    caption: "相机拍摄",
    license: "CC-BY-4.0",
    sha256: "complete",
    sourceUrl: "https://example.org/cup.jpg",
    synthetic: false,
    retractable: true,
    correct: "train",
    rejectReason: "",
  },
  {
    id: "img-002",
    title: "街景抓拍",
    caption: "缺许可",
    license: "",
    sha256: "present",
    sourceUrl: "https://cdn.example.net/street.png",
    synthetic: false,
    retractable: false,
    correct: "reject",
    rejectReason: "缺许可，不得进训练集",
  },
  {
    id: "img-003",
    title: "合成桌面",
    caption: "缺哈希",
    license: "CC-BY-4.0",
    sha256: "",
    sourceUrl: "https://gen.example.ai/table.webp",
    synthetic: true,
    retractable: true,
    correct: "reject",
    rejectReason: "缺哈希，字段不齐",
  },
];

const PREDICTIONS: { value: Exclude<PredictionId, "">; label: string }[] = [
  { value: "all", label: "三行字段看起来都有图，都会进训练集" },
  { value: "onlyA", label: "只有完整行进训练集，缺许可和缺哈希都被拒" },
  { value: "ab", label: "哈希能对上就行，缺许可的街景也能进" },
  { value: "ac", label: "合成图有许可就能进，缺哈希可以事后补" },
  { value: "cRejects", label: "合成图一律拒，缺许可的真实图可以进" },
];

function parsePlacements(raw: string): Record<SampleId, BucketId | ""> {
  const empty: Record<SampleId, BucketId | ""> = {
    "img-001": "",
    "img-002": "",
    "img-003": "",
  };
  if (!raw) {
    return empty;
  }
  try {
    const parsed = JSON.parse(raw) as Record<string, string>;
    for (const sample of SAMPLES) {
      const value = parsed[sample.id];
      if (value === "train" || value === "reject" || value === "") {
        empty[sample.id] = value;
      }
    }
  } catch {
    return empty;
  }
  return empty;
}

export function Lesson57ProvenanceLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    prediction: stringFrom(initialState, "prediction", "") as PredictionId | "",
    placements: parsePlacements(stringFrom(initialState, "placements", "")),
    ack: stringFrom(initialState, "ack", "") === "yes",
  };
  const [prediction, setPrediction] = useState<PredictionId | "">(
    defaults.prediction,
  );
  const [placements, setPlacements] = useState<Record<SampleId, BucketId | "">>(
    defaults.placements,
  );
  const [selected, setSelected] = useState<SampleId | "">("");
  const [ran, setRan] = useState(false);
  const [ack, setAck] = useState(defaults.ack);

  const missingLicenseInTrain = placements["img-002"] === "train";
  const missingHashInTrain = placements["img-003"] === "train";
  const allPlaced = SAMPLES.every((sample) => placements[sample.id] !== "");
  const allCorrect = SAMPLES.every(
    (sample) => placements[sample.id] === sample.correct,
  );
  const admittedCount = SAMPLES.filter(
    (sample) => placements[sample.id] === "train",
  ).length;
  const rejectedCount = SAMPLES.filter(
    (sample) => placements[sample.id] === "reject",
  ).length;

  const passed =
    ran &&
    prediction === "onlyA" &&
    allCorrect &&
    !missingLicenseInTrain &&
    !missingHashInTrain &&
    ack;

  const completion = useMemo(
    () => ({
      lessonId: 57,
      prediction,
      placements,
      admittedCount,
      rejectedCount,
      missingLicenseInTrain,
      missingHashInTrain,
      ack,
    }),
    [
      ack,
      admittedCount,
      missingHashInTrain,
      missingLicenseInTrain,
      placements,
      prediction,
      rejectedCount,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function placeSample(sampleId: SampleId, bucketId: BucketId) {
    setPlacements((current) => ({ ...current, [sampleId]: bucketId }));
    setSelected("");
    setRan(false);
  }

  function reset() {
    setPrediction("");
    setPlacements({
      "img-001": "",
      "img-002": "",
      "img-003": "",
    });
    setSelected("");
    setRan(false);
    setAck(false);
  }

  const unplaced = SAMPLES.filter((sample) => placements[sample.id] === "");

  return (
    <LabFrame
      lesson="57"
      title="三条样本的出处准入门"
      description="教学模拟，不是模型输出。先预测哪几行能进训练集，再把三条样本放进训练集或拒收。缺许可的行不得进训练集；缺字段的行被拒。揭晓前不显示准入数字。"
    >
      <div className={styles.workspace}>
        <form
          className={styles.controls}
          onSubmit={(event) => event.preventDefault()}
        >
          <h3>必填字段</h3>
          <p className={styles.note}>
            sidecar 一行对应一张图：sample_id、source_url、license、sha256、is_synthetic、retractable。空字符串算缺字段。
          </p>
          <p className={styles.note}>
            本课允许集合：CC-BY-4.0、CC0-1.0、Apache-2.0。unspecified 与空许可都非法。合成标记不能代替许可或哈希。
          </p>
          <dl className={styles.legend}>
            <div>
              <dt>硬绑定</dt>
              <dd>SHA-256 与字节相等才算哈希合法</dd>
            </div>
            <div>
              <dt>软绑定</dt>
              <dd>水印或指纹不进本课准入谓词</dd>
            </div>
          </dl>
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
                三条都已入桶。可拖回这里改放。揭晓前桶上不显示对错。
              </span>
            ) : (
              unplaced.map((sample) => (
                <button
                  key={sample.id}
                  type="button"
                  draggable
                  className={`${styles.card} ${
                    selected === sample.id ? styles.selected : ""
                  }`}
                  onClick={() =>
                    setSelected((current) =>
                      current === sample.id ? "" : sample.id,
                    )
                  }
                  onDragStart={() => setSelected(sample.id)}
                >
                  <b>{sample.title}</b>
                  <span>{sample.id}</span>
                </button>
              ))
            )}
          </div>
          <div className={styles.buckets} aria-label="训练集与拒收">
            {(
              [
                { id: "train" as const, title: "训练集", hint: "许可与哈希都必须合法" },
                { id: "reject" as const, title: "拒收", hint: "缺许可或缺字段" },
              ] as const
            ).map((bucket) => {
              const occupants = SAMPLES.filter(
                (sample) => placements[sample.id] === bucket.id,
              );
              const trapHot =
                bucket.id === "train" &&
                (missingLicenseInTrain || missingHashInTrain);
              const revealedCorrect =
                ran &&
                occupants.length > 0 &&
                occupants.every((sample) => sample.correct === bucket.id);
              return (
                <section
                  key={bucket.id}
                  className={`${styles.bucket} ${
                    trapHot ? styles.trap : ""
                  } ${revealedCorrect ? styles.correctBucket : ""}`}
                  onDragOver={(event) => event.preventDefault()}
                  onDrop={() => {
                    if (selected) {
                      placeSample(selected, bucket.id);
                    }
                  }}
                  onClick={() => {
                    if (selected) {
                      placeSample(selected, bucket.id);
                    }
                  }}
                >
                  <header>
                    <b>{bucket.title}</b>
                    <span>{bucket.hint}</span>
                  </header>
                  {occupants.map((sample) => {
                    const illegal =
                      bucket.id === "train" && sample.correct === "reject";
                    const ok = ran && sample.correct === bucket.id;
                    return (
                      <button
                        key={sample.id}
                        type="button"
                        draggable
                        className={`${styles.card} ${
                          selected === sample.id ? styles.selected : ""
                        } ${illegal ? styles.illegal : ""} ${
                          ok ? styles.ok : ""
                        }`}
                        onClick={(event) => {
                          event.stopPropagation();
                          setSelected((current) =>
                            current === sample.id ? "" : sample.id,
                          );
                        }}
                        onDragStart={(event) => {
                          event.stopPropagation();
                          setSelected(sample.id);
                        }}
                      >
                        <b>{sample.title}</b>
                        <span>
                          {sample.caption}
                          {illegal && sample.id === "img-002"
                            ? " 标红：缺许可"
                            : ""}
                          {illegal && sample.id === "img-003"
                            ? " 标红：缺哈希"
                            : ""}
                        </span>
                      </button>
                    );
                  })}
                </section>
              );
            })}
          </div>
          <table className={styles.sheet}>
            <caption>三条 sidecar（教学夹具）</caption>
            <thead>
              <tr>
                <th>样本</th>
                <th>许可</th>
                <th>哈希</th>
                <th>来源 URL</th>
                <th>合成</th>
                <th>可撤回</th>
              </tr>
            </thead>
            <tbody>
              {SAMPLES.map((sample) => (
                <tr key={sample.id}>
                  <td>{sample.title}</td>
                  <td>{sample.license === "" ? "空" : sample.license}</td>
                  <td>{sample.sha256 === "" ? "空" : "已填 SHA-256"}</td>
                  <td>{sample.sourceUrl}</td>
                  <td>{sample.synthetic ? "是" : "否"}</td>
                  <td>{sample.retractable ? "是" : "否"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className={styles.predict}>
        <fieldset>
          <legend>先预测：揭晓后哪几行会进入训练集</legend>
          {PREDICTIONS.map((option) => (
            <label key={option.value}>
              <input
                type="radio"
                name="provenance-prediction"
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
            checked={ack}
            onChange={(event) => {
              setAck(event.target.checked);
            }}
          />
          <span>缺许可或缺哈希为非法，C2PA 水印不能替代这两项</span>
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
            揭晓准入
          </button>
        </div>
      </div>

      {missingLicenseInTrain && (
        <p className={styles.feedback}>
          街景缺许可，被放进了训练集。这一格必须标红。许可字段为空或 unspecified 都不得进训练集。
        </p>
      )}
      {missingHashInTrain && (
        <p className={styles.feedback}>
          合成桌面缺哈希，被放进了训练集。缺字段的行被拒。合成标记不能补上哈希。
        </p>
      )}
      {ran && prediction !== "onlyA" && (
        <p className={styles.feedback}>
          预测应选“只有完整行进训练集”。三条夹具里准入计数是 1，拒收计数是 2。
        </p>
      )}
      {ran && prediction === "onlyA" && allCorrect && !ack && (
        <p className={styles.feedback}>
          分桶正确。还需要勾选：缺许可或缺哈希为非法。
        </p>
      )}
      {ran && (
        <dl className={styles.verdict}>
          <div>
            <dt>准入计数</dt>
            <dd>
              {SAMPLES.filter((sample) => sample.correct === "train").length}{" "}
              行进入训练集，
              {SAMPLES.filter((sample) => sample.correct === "reject").length}{" "}
              行拒收。
            </dd>
          </div>
          {SAMPLES.map((sample) => (
            <div key={sample.id}>
              <dt>{sample.title}</dt>
              <dd>
                {sample.correct === "train" ? "准入。" : "拒收。"}
                {sample.rejectReason ||
                  "许可在允许集合内，SHA-256 字段非空，来源 URL 存在。"}
              </dd>
            </div>
          ))}
        </dl>
      )}
      <Gate passed={passed}>
        先提交“只有完整行进训练集”，把缺许可和缺哈希放进拒收，并声明缺许可或缺哈希为非法。把缺许可的行拖进训练集必须标红，验收不通过。
      </Gate>
    </LabFrame>
  );
}
