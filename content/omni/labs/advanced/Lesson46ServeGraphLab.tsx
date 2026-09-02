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
import styles from "./Lesson46ServeGraphLab.module.css";

type ServeMode = "fused" | "stage";
type Prediction = "ok" | "pad" | "kv" | "both" | "";

type RequestSpec = {
  id: "text" | "image" | "action";
  title: string;
  vision: number;
  text: number;
  action: number;
};

type SlotKind = "vision" | "text" | "action" | "pad";

const PREDICTION_OPTIONS: { value: Exclude<Prediction, "">; label: string }[] =
  [
    {
      value: "ok",
      label: "只是慢一点，三条请求的数值仍然正确",
    },
    {
      value: "pad",
      label: "纯文本被垫到最长视觉长度，有效 token 比下降",
    },
    {
      value: "kv",
      label: "动作专家和语言 decode 共用同一张 KV 页表",
    },
    {
      value: "both",
      label: "同时出现 padding 浪费和错误共享 KV",
    },
  ];

function slotsForRequest(
  request: RequestSpec,
  maxVision: number,
  maxText: number,
  maxAction: number,
  mode: ServeMode,
): SlotKind[] {
  if (mode === "stage") {
    const slots: SlotKind[] = [
      ...Array.from({ length: request.vision }, () => "vision" as const),
      ...Array.from({ length: request.text }, () => "text" as const),
      ...Array.from({ length: request.action }, () => "action" as const),
    ];
    return slots.length > 0 ? slots : ["pad"];
  }
  return [
    ...Array.from({ length: request.vision }, () => "vision" as const),
    ...Array.from({ length: maxVision - request.vision }, () => "pad" as const),
    ...Array.from({ length: request.text }, () => "text" as const),
    ...Array.from({ length: maxText - request.text }, () => "pad" as const),
    ...Array.from({ length: request.action }, () => "action" as const),
    ...Array.from({ length: maxAction - request.action }, () => "pad" as const),
  ];
}

export function Lesson46ServeGraphLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    imagePatches: numberFrom(initialState, "imagePatches", 48, 16, 80),
    actionSteps: numberFrom(initialState, "actionSteps", 8, 4, 16),
    mode: stringFrom(initialState, "mode", "fused") as ServeMode,
    prediction: stringFrom(initialState, "prediction", "") as Prediction,
  };
  const [imagePatches, setImagePatches] = useState(
    Math.round(defaults.imagePatches),
  );
  const [actionSteps, setActionSteps] = useState(
    Math.round(defaults.actionSteps),
  );
  const [mode, setMode] = useState<ServeMode>(
    defaults.mode === "stage" ? "stage" : "fused",
  );
  const [prediction, setPrediction] = useState<Prediction>(
    defaults.prediction,
  );
  const [ran, setRan] = useState(false);
  const [sawFusedFailure, setSawFusedFailure] = useState(false);

  const simulation = useMemo(() => {
    const requests: RequestSpec[] = [
      { id: "text", title: "纯文本", vision: 0, text: 16, action: 0 },
      {
        id: "image",
        title: "带图理解",
        vision: imagePatches,
        text: 12,
        action: 0,
      },
      {
        id: "action",
        title: "带动作专家",
        vision: 24,
        text: 8,
        action: actionSteps,
      },
    ];
    const maxVision = Math.max(...requests.map((item) => item.vision));
    const maxText = Math.max(...requests.map((item) => item.text));
    const maxAction = Math.max(...requests.map((item) => item.action));
    const valid = requests.reduce(
      (sum, item) => sum + item.vision + item.text + item.action,
      0,
    );
    const fusedPadded = requests.length * (maxVision + maxText + maxAction);
    const stagePadded =
      requests
        .filter((item) => item.vision > 0)
        .reduce((sum) => sum + maxVision, 0) +
      requests.reduce((sum, item) => sum + item.text, 0) +
      maxAction;
    const padded = mode === "fused" ? fusedPadded : stagePadded;
    const ratio = valid / padded;
    const waste = 1 - ratio;
    const textWaste = maxVision + maxAction;
    const kvAlias = mode === "fused";
    const rows = requests.map((request) => ({
      ...request,
      slots: slotsForRequest(request, maxVision, maxText, maxAction, mode),
      valid: request.vision + request.text + request.action,
      padded:
        mode === "fused"
          ? maxVision + maxText + maxAction
          : request.vision + request.text + request.action,
    }));
    const kvCells = Array.from({ length: 12 }, (_, index) => {
      if (mode === "fused") {
        if (index < 8) {
          return {
            id: index,
            label: index < actionSteps ? "AR+动作" : "AR",
            kind: index < actionSteps ? "alias" : "ar",
          };
        }
        return { id: index, label: "空闲", kind: "idle" };
      }
      if (index < 8) {
        return { id: index, label: "AR", kind: "ar" };
      }
      return { id: index, label: "动作", kind: "action" };
    });
    return {
      requests: rows,
      maxVision,
      maxText,
      maxAction,
      valid,
      padded,
      ratio,
      waste,
      textWaste,
      kvAlias,
      kvCells,
      encodeBatch: requests.filter((item) => item.vision > 0).length,
      decodeBatch: requests.length,
      actionBatch: mode === "fused" ? requests.length : 1,
    };
  }, [actionSteps, imagePatches, mode]);

  const diagnosisOk =
    prediction === "pad" || prediction === "kv" || prediction === "both";
  const failureVisible =
    simulation.waste > 0.15 || simulation.kvAlias;
  const passed =
    ran &&
    sawFusedFailure &&
    diagnosisOk &&
    failureVisible &&
    mode === "fused";

  const completion = useMemo(
    () => ({
      lessonId: 46,
      imagePatches,
      actionSteps,
      mode,
      prediction,
      validTokenRatio: round(simulation.ratio, 4),
      paddingWaste: round(simulation.waste, 4),
      textRequestWastedSlots: simulation.textWaste,
      kvAliasing: simulation.kvAlias,
      sawFusedFailure,
    }),
    [
      actionSteps,
      imagePatches,
      mode,
      prediction,
      sawFusedFailure,
      simulation.kvAlias,
      simulation.ratio,
      simulation.textWaste,
      simulation.waste,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function invalidate() {
    setRan(false);
  }

  function reset() {
    setImagePatches(48);
    setActionSteps(8);
    setMode("fused");
    setPrediction("");
    setRan(false);
    setSawFusedFailure(false);
  }

  return (
    <LabFrame
      lesson="46"
      title="把三条请求强行编进一条 CUDA graph"
      description="教学模拟，不是模型输出。先预测合图之后会坏在哪，再捕获静态图，对照 stage graph 的分阶段组批。"
    >
      <div className={styles.workspace}>
        <form
          className={styles.controls}
          onSubmit={(event) => event.preventDefault()}
        >
          <h3>调度台</h3>
          <div className={styles.modeSwitch} role="group" aria-label="捕获方式">
            <button
              type="button"
              aria-pressed={mode === "fused"}
              onClick={() => {
                setMode("fused");
                invalidate();
              }}
            >
              一条 CUDA graph
            </button>
            <button
              type="button"
              aria-pressed={mode === "stage"}
              onClick={() => {
                setMode("stage");
                invalidate();
              }}
            >
              stage graph
            </button>
          </div>
          <label>
            <span>
              带图请求的 patch 数 <output>{imagePatches}</output>
            </span>
            <input
              type="range"
              min="16"
              max="80"
              step="8"
              value={imagePatches}
              onChange={(event) => {
                setImagePatches(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <label>
            <span>
              动作专家积分步 <output>{actionSteps}</output>
            </span>
            <input
              type="range"
              min="4"
              max="16"
              step="2"
              value={actionSteps}
              onChange={(event) => {
                setActionSteps(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <p className={styles.note}>
            纯文本 16 token、0 patch；带图请求 12 个文本 token；带动作专家的请求固定
            24 个视觉 token。合图时形状取三者最大值。
          </p>
        </form>

        <div className={styles.stage}>
          <div className={styles.formula}>
            <span>
              形状{" "}
              <strong>
                {mode === "fused"
                  ? `3 × (${simulation.maxVision}+${simulation.maxText}+${simulation.maxAction})`
                  : "按 stage 分别组批"}
              </strong>
            </span>
            <span>
              有效 / padded{" "}
              <strong>
                {simulation.valid}/{simulation.padded}
              </strong>
            </span>
            <span>
              ρ <strong>{round(simulation.ratio, 3)}</strong>
            </span>
          </div>

          {mode === "fused" ? (
            <div className={styles.fusedBanner}>
              静态图按最长视觉、最长文本、最多动作步捕获。纯文本请求仍要走完{" "}
              {simulation.maxVision} 个视觉槽和 {simulation.maxAction} 个空动作步。
            </div>
          ) : null}

          <div className={styles.requests} aria-label="三条请求的占用">
            {simulation.requests.map((request) => (
              <article className={styles.request} key={request.id}>
                <header>
                  <b>{request.title}</b>
                  <span>
                    有效 {request.valid} / 槽 {request.padded}
                  </span>
                </header>
                <div className={styles.bar} aria-hidden="true">
                  {request.slots.map((kind, index) => (
                    <span
                      key={`${request.id}-${index}`}
                      className={`${styles.cell} ${
                        kind === "vision"
                          ? styles.cellVision
                          : kind === "text"
                            ? styles.cellText
                            : kind === "action"
                              ? styles.cellAction
                              : styles.cellPad
                      }`}
                    />
                  ))}
                </div>
              </article>
            ))}
          </div>
          <p className={styles.legend}>
            <span>
              <i className={styles.cellVision} />
              视觉
            </span>
            <span>
              <i className={styles.cellText} />
              文本
            </span>
            <span>
              <i className={styles.cellAction} />
              动作步
            </span>
            <span>
              <i className={styles.cellPad} />
              padding
            </span>
          </p>

          <div className={styles.kvWrap}>
            <h3>KV 页表（教学 12 页）</h3>
            <div className={styles.kvGrid} aria-label="KV 页占用">
              {simulation.kvCells.map((cell) => (
                <div
                  key={cell.id}
                  className={`${styles.kvCell} ${
                    cell.kind === "alias"
                      ? styles.kvAlias
                      : cell.kind === "ar"
                        ? styles.kvAr
                        : cell.kind === "action"
                          ? styles.kvAction
                          : ""
                  }`}
                >
                  {cell.label}
                </div>
              ))}
            </div>
          </div>

          <div className={styles.stageCols}>
            <article className={styles.stageCard}>
              <b>理解编码</b>
              <p>
                {mode === "fused"
                  ? `三条请求都进 encoder，batch=3，视觉长度锁死为 ${simulation.maxVision}。`
                  : `只有 ${simulation.encodeBatch} 条带图请求进 encoder。`}
              </p>
            </article>
            <article className={styles.stageCard}>
              <b>语言 decode</b>
              <p>
                batch={simulation.decodeBatch}。
                {mode === "fused"
                  ? " KV 页从 0 号开始排，和动作专家重叠。"
                  : " 独立页表，不读动作页。"}
              </p>
            </article>
            <article className={styles.stageCard}>
              <b>flow / 动作</b>
              <p>
                {mode === "fused"
                  ? `空请求也要跑 ${simulation.maxAction} 步，batch=${simulation.actionBatch}。`
                  : `只有动作请求跑 ${actionSteps} 步，batch=1。`}
              </p>
            </article>
          </div>

          <dl className={styles.metrics}>
            <div>
              <dt>有效 token 比 ρ</dt>
              <dd>{ran ? round(simulation.ratio, 3) : "—"}</dd>
            </div>
            <div>
              <dt>padding 浪费</dt>
              <dd>{ran ? `${round(simulation.waste * 100, 1)}%` : "—"}</dd>
            </div>
            <div>
              <dt>纯文本空槽</dt>
              <dd>{ran ? simulation.textWaste : "—"}</dd>
            </div>
            <div>
              <dt>KV 别名</dt>
              <dd>{ran ? (simulation.kvAlias ? "是" : "否") : "—"}</dd>
            </div>
          </dl>

          <div className={styles.predict}>
            <fieldset>
              <legend>先预测：合成一条 CUDA graph 之后会发生什么？</legend>
              {PREDICTION_OPTIONS.map((option) => (
                <label key={option.value}>
                  <input
                    type="radio"
                    name="lesson46-prediction"
                    value={option.value}
                    checked={prediction === option.value}
                    onChange={() => {
                      setPrediction(option.value);
                      invalidate();
                    }}
                  />
                  <span>{option.label}</span>
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
                onClick={() => {
                  setRan(true);
                  if (mode === "fused") {
                    setSawFusedFailure(true);
                  }
                }}
              >
                捕获并揭晓
              </button>
            </div>
          </div>
          {ran && prediction === "ok" ? (
            <p className={styles.feedback}>
              合图并没有保持正确性：纯文本请求付出了 {simulation.textWaste}{" "}
              个空槽，KV 页表上动作专家与 AR 重叠。
            </p>
          ) : null}
          {ran && mode === "stage" ? (
            <p className={styles.feedback}>
              验收要求先在“一条 CUDA graph”模式下看到浪费或 KV 别名。对照完
              stage graph 后，请切回去再捕获一次。
            </p>
          ) : null}
          {ran && mode === "fused" && diagnosisOk ? (
            <p className={styles.feedback}>
              揭晓：ρ={round(simulation.ratio, 3)}，浪费{" "}
              {round(simulation.waste * 100, 1)}%，KV 别名
              {simulation.kvAlias ? "成立" : "不成立"}。切到 stage graph
              可对照分阶段组批。
            </p>
          ) : null}
        </div>
      </div>
      <Gate passed={passed}>
        先选择预测，再把三条请求捕获进一条 CUDA graph。必须看到 padding
        浪费（有效比下降）或动作专家与语言 decode 的 KV 别名。
      </Gate>
    </LabFrame>
  );
}
