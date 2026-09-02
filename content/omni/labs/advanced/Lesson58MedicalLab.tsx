"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import { numberFrom, stringFrom, useCompletionGate } from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson58MedicalLab.module.css";

type ImageId = "empty" | "opacity";
type PredictionId = "empty-unboxed" | "empty-refuse" | "boxed-positive";

type SimResult = {
  finding: string;
  positive: boolean;
  boxed: boolean;
  unboxedPositive: boolean;
  unboxedCount: number;
  report: string;
};

function simulate(image: ImageId, forbidUnboxed: boolean): SimResult {
  const hasBox = image === "opacity";

  if (forbidUnboxed && !hasBox) {
    return {
      finding: "未见局灶实变",
      positive: false,
      boxed: false,
      unboxedPositive: false,
      unboxedCount: 0,
      report: "FINDINGS: 双肺野示意均匀，未见局灶实变。IMPRESSION: 未见需要框出的阳性发现。",
    };
  }
  if (forbidUnboxed && hasBox) {
    return {
      finding: "右下肺斑片",
      positive: true,
      boxed: true,
      unboxedPositive: false,
      unboxedCount: 0,
      report: "FINDINGS: 右下肺斑片 [BOX 0.58,0.52,0.82,0.78]。IMPRESSION: 阳性发现与框一致。",
    };
  }
  if (!forbidUnboxed && !hasBox) {
    return {
      finding: "肺炎",
      positive: true,
      boxed: false,
      unboxedPositive: true,
      unboxedCount: 1,
      report: "FINDINGS: 肺炎。IMPRESSION: 符合肺炎。无框。",
    };
  }
  return {
    finding: "肺炎",
    positive: true,
    boxed: false,
    unboxedPositive: true,
    unboxedCount: 1,
    report: "FINDINGS: 肺炎。IMPRESSION: 符合肺炎。图中有斑片，但解码器未绑定框。",
  };
}

export function Lesson58MedicalLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    image: stringFrom(initialState, "image", "empty") as ImageId,
    forbidUnboxed: numberFrom(initialState, "forbidUnboxed", 1, 0, 1) === 1,
    prediction: stringFrom(initialState, "prediction", "") as PredictionId | "",
  };
  const [image, setImage] = useState<ImageId>(
    defaults.image === "opacity" ? "opacity" : "empty",
  );
  const [forbidUnboxed, setForbidUnboxed] = useState(defaults.forbidUnboxed);
  const [prediction, setPrediction] = useState<PredictionId | "">(
    defaults.prediction === "empty-unboxed" ||
      defaults.prediction === "empty-refuse" ||
      defaults.prediction === "boxed-positive"
      ? defaults.prediction
      : "",
  );
  const [ran, setRan] = useState(false);

  const result = useMemo(
    () => simulate(image, forbidUnboxed),
    [forbidUnboxed, image],
  );
  const caughtUnboxed =
    image === "empty" && !forbidUnboxed && result.unboxedPositive;
  const passed =
    ran && prediction === "empty-unboxed" && caughtUnboxed;
  const completion = useMemo(
    () => ({
      lessonId: 58,
      image,
      forbidUnboxed,
      prediction,
      unboxedCount: result.unboxedCount,
      unboxedPositive: result.unboxedPositive,
      finding: result.finding,
    }),
    [
      forbidUnboxed,
      image,
      prediction,
      result.finding,
      result.unboxedCount,
      result.unboxedPositive,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function invalidate() {
    setRan(false);
  }

  function reset() {
    setImage("empty");
    setForbidUnboxed(true);
    setPrediction("");
    setRan(false);
  }

  return (
    <LabFrame
      lesson="58"
      title="胸片无框断言门控"
      description="这是教学模拟，不是模型输出，也不是影像诊断。先预测关掉“禁止无框断言”之后空图会怎样，再拨开关运行。验收：必须抓到空图上的无框肯定。"
    >
      <div className={styles.workspace}>
        <form className={styles.controls} onSubmit={(event) => event.preventDefault()}>
          <h3>检查台</h3>
          <fieldset className={styles.imagePick}>
            <legend>示意胸片</legend>
            <label>
              <input
                type="radio"
                name="cxr-image"
                checked={image === "empty"}
                onChange={() => {
                  setImage("empty");
                  invalidate();
                }}
              />
              <span>空图：无斑片</span>
            </label>
            <label>
              <input
                type="radio"
                name="cxr-image"
                checked={image === "opacity"}
                onChange={() => {
                  setImage("opacity");
                  invalidate();
                }}
              />
              <span>右下肺斑片</span>
            </label>
          </fieldset>
          <label className={styles.toggle}>
            <span>
              禁止无框断言
              <output>{forbidUnboxed ? "开" : "关"}</output>
            </span>
            <input
              type="checkbox"
              checked={forbidUnboxed}
              onChange={(event) => {
                setForbidUnboxed(event.target.checked);
                invalidate();
              }}
            />
            <small>
              {forbidUnboxed
                ? "阳性发现必须绑定框，否则不得写病灶名"
                : "解码器可按语言先验写病灶名，不要求框"}
            </small>
          </label>
          <p className={styles.recipe}>
            {image === "empty" ? "当前输入是空图。" : "当前输入含示意斑片。"}
            {forbidUnboxed ? " 门控开。" : " 门控关。"}
            关键计数 U = 无框且阳性的句子数。
          </p>
        </form>

        <div className={styles.stage}>
          <div className={styles.formula}>
            <span>U = 1[pos(y)=1 且 box=空]；空图 + 门控关 必须得到 U=1</span>
            <strong>
              {ran
                ? `U ${result.unboxedCount} · ${result.unboxedPositive ? "无框肯定" : "无无框肯定"}`
                : "先预测再运行，才会揭晓报告和 U"}
            </strong>
          </div>
          <div className={styles.film} aria-label="示意胸片">
            <ChestSchematic
              empty={image === "empty"}
              showBox={ran && result.boxed}
              warnUnboxed={ran && result.unboxedPositive && image === "empty"}
            />
            <dl className={styles.metrics}>
              <div className={ran && result.unboxedPositive ? styles.warn : undefined}>
                <dt>U 无框肯定</dt>
                <dd>{ran ? result.unboxedCount : "—"}</dd>
                <small>目标空图门控关 = 1</small>
              </div>
              <div>
                <dt>阳性发现</dt>
                <dd>{ran ? (result.positive ? "有" : "无") : "—"}</dd>
                <small>门控开时必须带框</small>
              </div>
              <div>
                <dt>是否绑框</dt>
                <dd>{ran ? (result.boxed ? "有框" : "无框") : "—"}</dd>
                <small>空图不应有框</small>
              </div>
            </dl>
            {ran ? (
              <p className={styles.report} data-unboxed={result.unboxedPositive ? "true" : "false"}>
                {result.report}
              </p>
            ) : (
              <p className={styles.reportHidden}>报告已生成但未揭晓。选出预测后再运行。</p>
            )}
          </div>
          {ran && (
            <p className={styles.feedback}>
              {caughtUnboxed
                ? "抓到了：空图、门控关闭、解码器仍写出肺炎且没有框。这是语言先验，不是图像证据。"
                : forbidUnboxed && image === "empty"
                  ? "门控开着，空图被写成未见局灶实变，U=0。把“禁止无框断言”关掉再跑空图。"
                  : forbidUnboxed && image === "opacity"
                    ? "门控开且图中有斑片时，允许带框的阳性。验收要的是空图上的无框肯定。"
                    : "门控关且图中有斑片时也会漏绑框。先换成空图，才能单独归因语言先验。"}
            </p>
          )}
        </div>
      </div>

      <div className={styles.predict}>
        <fieldset>
          <legend>先预测：关掉“禁止无框断言”之后，空图会怎样？</legend>
          {(
            [
              ["empty-unboxed", "空图仍会无框报肺炎"],
              ["empty-refuse", "空图会拒报或写未见"],
              ["boxed-positive", "空图也会长出框再报肺炎"],
            ] as const
          ).map(([value, label]) => (
            <label key={value}>
              <input
                type="radio"
                name="med-prediction"
                value={value}
                checked={prediction === value}
                onChange={() => {
                  setPrediction(value);
                  setRan(false);
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
            onClick={() => setRan(true)}
          >
            运行解码
          </button>
        </div>
      </div>
      {ran && prediction !== "empty-unboxed" && (
        <p className={styles.feedback}>
          空图没有斑片可框。门控关掉以后，模型并不会凭空长框，它会直接写出“肺炎”。要抓的是无框肯定，不是拒报，也不是带框阳性。
        </p>
      )}
      <Gate passed={passed}>
        先选“空图仍会无框报肺炎”，再关掉禁止无框断言、选空图并运行，使 U=1。这是教学模拟，不是临床读片。
      </Gate>
    </LabFrame>
  );
}

function ChestSchematic({
  empty,
  showBox,
  warnUnboxed,
}: {
  empty: boolean;
  showBox: boolean;
  warnUnboxed: boolean;
}) {
  return (
    <svg
      className={styles.cxr}
      viewBox="0 0 200 240"
      role="img"
      aria-label={empty ? "空的示意胸片" : "带右下肺斑片的示意胸片"}
    >
      <rect x="8" y="8" width="184" height="224" rx="8" className={styles.filmBg} />
      <ellipse cx="72" cy="118" rx="42" ry="78" className={styles.lung} />
      <ellipse cx="128" cy="118" rx="42" ry="78" className={styles.lung} />
      <path d="M100 42 L100 188" className={styles.spine} />
      {Array.from({ length: 6 }, (_, index) => {
        const y = 58 + index * 18;
        return (
          <g key={y}>
            <path d={`M100 ${y} C 70 ${y + 4}, 48 ${y + 10}, 36 ${y + 8}`} className={styles.rib} />
            <path d={`M100 ${y} C 130 ${y + 4}, 152 ${y + 10}, 164 ${y + 8}`} className={styles.rib} />
          </g>
        );
      })}
      {!empty && (
        <ellipse cx="140" cy="158" rx="22" ry="16" className={styles.opacity} />
      )}
      {showBox && (
        <rect x="116" y="140" width="48" height="36" className={styles.box} />
      )}
      {warnUnboxed && (
        <text x="100" y="28" textAnchor="middle" className={styles.warnLabel}>
          无框肯定
        </text>
      )}
      <text x="100" y="226" textAnchor="middle" className={styles.caption}>
        教学示意图，不是放射影像
      </text>
    </svg>
  );
}
