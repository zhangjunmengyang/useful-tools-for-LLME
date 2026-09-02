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
import styles from "./Lesson22RecipeLab.module.css";

type Prediction = "" | "alignment" | "instruction" | "old-text";

const BASELINE = {
  alignment: 24,
  instruction: 30,
  oldText: 90,
};

const ALIGN_GAIN_THRESHOLD = 30;
const TEXT_DROP_THRESHOLD = 18;

function simulate(freezeVit: boolean, trainProjector: boolean, trainLlm: boolean) {
  if (freezeVit && trainProjector && !trainLlm) {
    return { alignment: 64, instruction: 38, oldText: 89 };
  }
  if (freezeVit && trainProjector && trainLlm) {
    return { alignment: 72, instruction: 70, oldText: 80 };
  }
  if (!freezeVit && trainProjector && trainLlm) {
    return { alignment: 80, instruction: 68, oldText: 51 };
  }
  if (!freezeVit && trainProjector && !trainLlm) {
    return { alignment: 70, instruction: 42, oldText: 84 };
  }
  if (freezeVit && !trainProjector && trainLlm) {
    return { alignment: 26, instruction: 50, oldText: 76 };
  }
  if (!freezeVit && !trainProjector && trainLlm) {
    return { alignment: 28, instruction: 46, oldText: 62 };
  }
  if (!freezeVit && !trainProjector && !trainLlm) {
    return { alignment: 38, instruction: 31, oldText: 87 };
  }
  return { ...BASELINE };
}

function recipeName(freezeVit: boolean, trainProjector: boolean, trainLlm: boolean) {
  if (freezeVit && trainProjector && !trainLlm) return "阶段 1：冻 ViT / 只训投影 / 冻 LLM";
  if (freezeVit && trainProjector && trainLlm) return "阶段 2：冻 ViT / 训投影 / 训 LLM";
  if (!freezeVit && trainProjector && trainLlm) return "过早解冻：ViT + 投影 + LLM 同时更新";
  if (!freezeVit && trainProjector && !trainLlm) return "冻 LLM，解冻 ViT 与投影";
  if (freezeVit && !trainProjector && trainLlm) return "冻视觉通路，只改 LLM";
  if (!freezeVit && !trainProjector && trainLlm) return "投影冻结，ViT 与 LLM 一起动";
  if (!freezeVit && !trainProjector && !trainLlm) return "只让 ViT 走冻结投影回传";
  return "无更新：三处都不训";
}

export function Lesson22RecipeLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    freezeVit: numberFrom(initialState, "freezeVit", 1, 0, 1) === 1,
    trainProjector: numberFrom(initialState, "trainProjector", 1, 0, 1) === 1,
    trainLlm: numberFrom(initialState, "trainLlm", 0, 0, 1) === 1,
    prediction: stringFrom(initialState, "prediction", "") as Prediction,
  };
  const [freezeVit, setFreezeVit] = useState(defaults.freezeVit);
  const [trainProjector, setTrainProjector] = useState(defaults.trainProjector);
  const [trainLlm, setTrainLlm] = useState(defaults.trainLlm);
  const [prediction, setPrediction] = useState<Prediction>(
    defaults.prediction === "alignment" ||
      defaults.prediction === "instruction" ||
      defaults.prediction === "old-text"
      ? defaults.prediction
      : "",
  );
  const [ran, setRan] = useState(false);

  const scores = useMemo(
    () => simulate(freezeVit, trainProjector, trainLlm),
    [freezeVit, trainLlm, trainProjector],
  );
  const alignmentGain = scores.alignment - BASELINE.alignment;
  const textDrop = BASELINE.oldText - scores.oldText;
  const foundFailure = alignmentGain >= ALIGN_GAIN_THRESHOLD && textDrop >= TEXT_DROP_THRESHOLD;
  const passed = ran && prediction === "old-text" && foundFailure;
  const completion = useMemo(
    () => ({
      lessonId: 22,
      freezeVit,
      trainProjector,
      trainLlm,
      prediction,
      alignment: scores.alignment,
      instruction: scores.instruction,
      oldText: scores.oldText,
      alignmentGain: round(alignmentGain, 1),
      textDrop: round(textDrop, 1),
    }),
    [
      alignmentGain,
      freezeVit,
      prediction,
      scores.alignment,
      scores.instruction,
      scores.oldText,
      textDrop,
      trainLlm,
      trainProjector,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function invalidate() {
    setRan(false);
  }

  function reset() {
    setFreezeVit(true);
    setTrainProjector(true);
    setTrainLlm(false);
    setPrediction("");
    setRan(false);
  }

  return (
    <LabFrame
      lesson="22"
      title="三阶段解冻开关"
      description="这是教学模拟，不是真实模型输出。先预测过早解冻 ViT 会砸哪一项，再拨开关跑一轮玩具训练：图文对齐上升且旧文本能力跌过阈值，才算找到失败配方。"
    >
      <div className={styles.workspace}>
        <form className={styles.controls} onSubmit={(event) => event.preventDefault()}>
          <h3>冻结控制台</h3>
          <Toggle
            label="冻 ViT"
            checked={freezeVit}
            onChange={(value) => {
              setFreezeVit(value);
              invalidate();
            }}
            detail={freezeVit ? "视觉塔 requires_grad = false" : "视觉塔接收梯度"}
          />
          <Toggle
            label="训投影"
            checked={trainProjector}
            onChange={(value) => {
              setTrainProjector(value);
              invalidate();
            }}
            detail={trainProjector ? "更新 W : d_v → d_llm" : "W 冻结"}
          />
          <Toggle
            label="训 LLM"
            checked={trainLlm}
            onChange={(value) => {
              setTrainLlm(value);
              invalidate();
            }}
            detail={trainLlm ? "语言模型权重参与更新" : "语言模型冻结"}
          />
          <p className={styles.recipe}>{recipeName(freezeVit, trainProjector, trainLlm)}</p>
        </form>

        <div className={styles.stage}>
          <div className={styles.formula}>
            <span>H_v = W Z_v；对齐增益阈值 {ALIGN_GAIN_THRESHOLD}，旧文本跌幅阈值 {TEXT_DROP_THRESHOLD}</span>
            <strong>
              {ran
                ? `Δ对齐 ${alignmentGain >= 0 ? "+" : ""}${alignmentGain} · Δ旧文本 ${textDrop >= 0 ? "−" : "+"}${Math.abs(textDrop)}`
                : "先预测再运行，才会揭晓三项分数"}
            </strong>
          </div>
          <div className={styles.pipeline} aria-label="模块冻结状态">
            <ModuleCard
              name="ViT"
              status={freezeVit ? "冻结" : "可训练"}
              active={!freezeVit}
            />
            <span className={styles.arrow} aria-hidden="true">
              W
            </span>
            <ModuleCard
              name="投影"
              status={trainProjector ? "可训练" : "冻结"}
              active={trainProjector}
            />
            <span className={styles.arrow} aria-hidden="true">
              tok
            </span>
            <ModuleCard
              name="LLM"
              status={trainLlm ? "可训练" : "冻结"}
              active={trainLlm}
            />
          </div>
          <dl className={styles.metrics}>
            <Metric
              label="图文对齐"
              value={ran ? scores.alignment : null}
              baseline={BASELINE.alignment}
            />
            <Metric
              label="指令跟随"
              value={ran ? scores.instruction : null}
              baseline={BASELINE.instruction}
            />
            <Metric
              label="旧文本能力"
              value={ran ? scores.oldText : null}
              baseline={BASELINE.oldText}
              warn={ran && textDrop >= TEXT_DROP_THRESHOLD}
            />
          </dl>
          {ran && (
            <p className={styles.feedback}>
              {foundFailure
                ? "这组开关让对齐上升，同时旧文本能力跌过阈值。过早把 ViT 和 LLM 放进同一组更新，会改写已经学好的文本方向。"
                : freezeVit && trainProjector && trainLlm
                  ? "标准第二阶段：对齐和指令都升，旧文本只轻降，没有越过阈值。"
                  : freezeVit && trainProjector && !trainLlm
                    ? "标准第一阶段：投影把视觉特征送进词嵌入维，旧文本几乎不动。"
                    : "还没同时满足“对齐上升超过阈值”和“旧文本跌过阈值”。试把 ViT 解冻，并同时打开投影和 LLM。"}
            </p>
          )}
        </div>
      </div>

      <div className={styles.predict}>
        <fieldset>
          <legend>
            先预测：第一步就解冻 ViT，并且同时训投影和 LLM，哪一项会跌过阈值？
          </legend>
          {(
            [
              ["alignment", "图文对齐会跌过阈值"],
              ["instruction", "指令跟随会跌过阈值"],
              ["old-text", "旧文本能力会跌过阈值"],
            ] as const
          ).map(([value, label]) => (
            <label key={value}>
              <input
                type="radio"
                name="recipe-prediction"
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
            运行玩具训练
          </button>
        </div>
      </div>
      {ran && prediction !== "old-text" && (
        <p className={styles.feedback}>
          对齐在解冻 ViT 后通常继续升，指令跟随也不归零。被改写的是 LLM 里已经排好的文本方向，所以旧文本探针会先跌穿。
        </p>
      )}
      <Gate passed={passed}>
        先选对“旧文本能力会跌过阈值”，再找到一组开关：对齐相对基线至少升 {ALIGN_GAIN_THRESHOLD}，旧文本至少跌 {TEXT_DROP_THRESHOLD}。
      </Gate>
    </LabFrame>
  );
}

function Toggle({
  label,
  checked,
  onChange,
  detail,
}: {
  label: string;
  checked: boolean;
  onChange: (value: boolean) => void;
  detail: string;
}) {
  return (
    <label className={styles.toggle}>
      <span>
        {label}
        <output>{checked ? "开" : "关"}</output>
      </span>
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
      />
      <small>{detail}</small>
    </label>
  );
}

function ModuleCard({
  name,
  status,
  active,
}: {
  name: string;
  status: string;
  active: boolean;
}) {
  return (
    <article className={`${styles.module} ${active ? styles.moduleOn : ""}`}>
      <b>{name}</b>
      <span>{status}</span>
    </article>
  );
}

function Metric({
  label,
  value,
  baseline,
  warn = false,
}: {
  label: string;
  value: number | null;
  baseline: number;
  warn?: boolean;
}) {
  return (
    <div className={warn ? styles.warn : undefined}>
      <dt>{label}</dt>
      <dd>{value === null ? "—" : value}</dd>
      <small>基线 {baseline}</small>
    </div>
  );
}
