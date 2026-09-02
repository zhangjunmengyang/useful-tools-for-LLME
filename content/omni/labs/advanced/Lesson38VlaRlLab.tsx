"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import {
  mean,
  numberFrom,
  round,
  stringFrom,
  useCompletionGate,
} from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson38VlaRlLab.module.css";

type BatchId = "fail" | "mixed";
type RewardId = "sparse" | "dense";

type Rollout = {
  id: string;
  name: string;
  x: number;
  y: number;
  clearance: number;
  gripperClosed: boolean;
  lifted: boolean;
  inBin: boolean;
  force: number;
};

const FAIL_GROUP: Rollout[] = [
  {
    id: "A",
    name: "空抓",
    x: 0.22,
    y: 0.3,
    clearance: 0.18,
    gripperClosed: false,
    lifted: false,
    inBin: false,
    force: 0,
  },
  {
    id: "B",
    name: "悬停",
    x: 0.46,
    y: 0.42,
    clearance: 0.055,
    gripperClosed: false,
    lifted: false,
    inBin: false,
    force: 0.02,
  },
  {
    id: "C",
    name: "擦边",
    x: 0.52,
    y: 0.5,
    clearance: 0.018,
    gripperClosed: true,
    lifted: false,
    inBin: false,
    force: 0.22,
  },
  {
    id: "D",
    name: "压溃",
    x: 0.54,
    y: 0.53,
    clearance: 0.006,
    gripperClosed: true,
    lifted: false,
    inBin: false,
    force: 1.35,
  },
];

const SUCCESS_ROLLOUT: Rollout = {
  id: "E",
  name: "放入",
  x: 0.78,
  y: 0.7,
  clearance: 0.02,
  gripperClosed: true,
  lifted: true,
  inBin: true,
  force: 0.18,
};

const PREDICTIONS = [
  {
    value: "sparse_zero_dense_rank",
    label: "失败批次上稀疏优势全零，dense 仍有非零更新",
  },
  {
    value: "both_zero",
    label: "两种奖励在失败批次上都没有更新",
  },
  {
    value: "sparse_better",
    label: "失败批次上稀疏成功的优势更大",
  },
  {
    value: "only_mixed",
    label: "只有混进成功轨迹时 dense 才有信号",
  },
] as const;

function clamp01(value: number) {
  return Math.max(0, Math.min(1, value));
}

function sparseReward(rollout: Rollout) {
  return rollout.lifted && rollout.inBin ? 1 : 0;
}

function denseReward(
  rollout: Rollout,
  tau: number,
  contactBand: number,
  forceSafe: number,
) {
  const approach = Math.exp(-rollout.clearance / tau);
  const contact =
    rollout.gripperClosed && rollout.clearance <= contactBand ? 1 : 0;
  const lift = rollout.lifted ? 1 : 0;
  const forcePenalty = Math.max(0, rollout.force - forceSafe);
  return clamp01(
    0.35 * approach + 0.25 * contact + 0.4 * lift - 0.3 * forcePenalty,
  );
}

function scoreGroup(
  group: Rollout[],
  rewardId: RewardId,
  tau: number,
  contactBand: number,
  forceSafe: number,
) {
  const scored = group.map((rollout) => {
    const reward =
      rewardId === "sparse"
        ? sparseReward(rollout)
        : denseReward(rollout, tau, contactBand, forceSafe);
    return { ...rollout, reward };
  });
  const rewards = scored.map((item) => item.reward);
  const average = mean(rewards);
  const variance = mean(rewards.map((reward) => (reward - average) ** 2));
  const withAdvantage = scored.map((item) => ({
    ...item,
    advantage: item.reward - average,
  }));
  const updateNorm = withAdvantage.reduce(
    (sum, item) => sum + Math.abs(item.advantage),
    0,
  );
  const ranked = [...withAdvantage].sort((left, right) => {
    if (right.reward !== left.reward) return right.reward - left.reward;
    return left.id.localeCompare(right.id);
  });
  return {
    items: withAdvantage,
    average,
    variance,
    updateNorm,
    ranked,
    allAdvantagesZero: withAdvantage.every(
      (item) => Math.abs(item.advantage) < 1e-9,
    ),
    hasNonzeroUpdate: updateNorm > 1e-6,
  };
}

export function Lesson38VlaRlLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    tau: numberFrom(initialState, "tau", 0.04, 0.02, 0.12),
    contactBand: numberFrom(initialState, "contactBand", 0.03, 0.01, 0.06),
    forceSafe: numberFrom(initialState, "forceSafe", 0.45, 0.15, 1.2),
    batch: stringFrom(initialState, "batch", "fail") as BatchId,
    reward: stringFrom(initialState, "reward", "sparse") as RewardId,
    prediction: stringFrom(initialState, "prediction", ""),
  };
  const [tau, setTau] = useState(defaults.tau);
  const [contactBand, setContactBand] = useState(defaults.contactBand);
  const [forceSafe, setForceSafe] = useState(defaults.forceSafe);
  const [batch, setBatch] = useState<BatchId>(
    defaults.batch === "mixed" ? "mixed" : "fail",
  );
  const [reward, setReward] = useState<RewardId>(
    defaults.reward === "dense" ? "dense" : "sparse",
  );
  const [prediction, setPrediction] = useState(defaults.prediction);
  const [revealed, setRevealed] = useState(false);

  const group = useMemo(
    () => (batch === "mixed" ? [...FAIL_GROUP.slice(0, 3), SUCCESS_ROLLOUT] : FAIL_GROUP),
    [batch],
  );

  const sparse = useMemo(
    () => scoreGroup(group, "sparse", tau, contactBand, forceSafe),
    [contactBand, forceSafe, group, tau],
  );
  const dense = useMemo(
    () => scoreGroup(group, "dense", tau, contactBand, forceSafe),
    [contactBand, forceSafe, group, tau],
  );
  const active = reward === "sparse" ? sparse : dense;

  const passed =
    revealed &&
    prediction === "sparse_zero_dense_rank" &&
    batch === "fail" &&
    sparse.allAdvantagesZero &&
    dense.hasNonzeroUpdate;

  const completion = useMemo(
    () => ({
      lessonId: 38,
      tau,
      contactBand,
      forceSafe,
      batch,
      reward,
      prediction,
      sparseVariance: round(sparse.variance, 6),
      denseVariance: round(dense.variance, 6),
      sparseUpdate: round(sparse.updateNorm, 6),
      denseUpdate: round(dense.updateNorm, 6),
    }),
    [
      batch,
      contactBand,
      dense.updateNorm,
      dense.variance,
      forceSafe,
      prediction,
      reward,
      sparse.updateNorm,
      sparse.variance,
      tau,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function reset() {
    setTau(0.04);
    setContactBand(0.03);
    setForceSafe(0.45);
    setBatch("fail");
    setReward("sparse");
    setPrediction("");
    setRevealed(false);
  }

  return (
    <LabFrame
      lesson="38"
      title="同一抓取：稀疏成功还是接触 dense"
      description="四条轨迹来自同一只杯子。先预测失败批次上谁还有更新，再揭晓组均值优势。数字由教学公式计算，不是模型输出。"
    >
      <div className={styles.workspace}>
        <form className={styles.controls} onSubmit={(event) => event.preventDefault()}>
          <h3>奖励控制台</h3>
          <fieldset>
            <legend>批次</legend>
            <label>
              <input
                type="radio"
                name="batch"
                checked={batch === "fail"}
                onChange={() => {
                  setBatch("fail");
                  setRevealed(false);
                }}
              />
              <span>全失败（空抓 / 悬停 / 擦边 / 压溃）</span>
            </label>
            <label>
              <input
                type="radio"
                name="batch"
                checked={batch === "mixed"}
                onChange={() => {
                  setBatch("mixed");
                  setRevealed(false);
                }}
              />
              <span>混入一条放入成功</span>
            </label>
          </fieldset>
          <fieldset>
            <legend>当前查看的奖励</legend>
            <label>
              <input
                type="radio"
                name="reward"
                checked={reward === "sparse"}
                onChange={() => {
                  setReward("sparse");
                  setRevealed(false);
                }}
              />
              <span>稀疏成功 I_success</span>
            </label>
            <label>
              <input
                type="radio"
                name="reward"
                checked={reward === "dense"}
                onChange={() => {
                  setReward("dense");
                  setRevealed(false);
                }}
              />
              <span>接触 dense</span>
            </label>
          </fieldset>
          <label>
            <span>接近尺度 τ <output>{tau.toFixed(3)}</output></span>
            <input
              type="range"
              min="0.02"
              max="0.12"
              step="0.005"
              value={tau}
              onChange={(event) => {
                setTau(Number(event.target.value));
                setRevealed(false);
              }}
            />
          </label>
          <label>
            <span>接触带 <output>{contactBand.toFixed(3)}</output></span>
            <input
              type="range"
              min="0.01"
              max="0.06"
              step="0.002"
              value={contactBand}
              onChange={(event) => {
                setContactBand(Number(event.target.value));
                setRevealed(false);
              }}
            />
          </label>
          <label>
            <span>力安全阈 <output>{forceSafe.toFixed(2)}</output></span>
            <input
              type="range"
              min="0.15"
              max="1.2"
              step="0.05"
              value={forceSafe}
              onChange={(event) => {
                setForceSafe(Number(event.target.value));
                setRevealed(false);
              }}
            />
          </label>
          <p className={styles.note}>
            验收盯失败批次。稀疏看物体是否离桌进盒；dense 看接近、夹爪闭合和力超限。改滑条会清掉揭晓。
          </p>
        </form>

        <div className={styles.stage}>
          <div className={styles.table} aria-label="俯视抓取教学场景">
            <div className={styles.cup} style={{ left: "52%", top: "50%" }}>
              <span className={styles.label}>杯</span>
            </div>
            <div className={styles.bin} style={{ left: "78%", top: "70%" }}>
              <span className={styles.label}>盒</span>
            </div>
            {group.map((rollout) => {
              const item = active.items.find((row) => row.id === rollout.id);
              const best = revealed && item?.id === active.ranked[0]?.id;
              return (
                <div
                  key={rollout.id}
                  className={styles.gripper}
                  data-fail={rollout.lifted ? "false" : "true"}
                  data-best={best ? "true" : "false"}
                  style={{ left: `${rollout.x * 100}%`, top: `${rollout.y * 100}%` }}
                >
                  {rollout.id}
                  <span className={styles.label}>{rollout.name}</span>
                </div>
              );
            })}
            <p className={styles.caption}>
              夹爪终点。揭晓前不显示分数。
            </p>
          </div>

          <div className={styles.rows}>
            <div className={styles.rowHead}>
              <span>编号</span>
              <span>轨迹</span>
              <span>优势条</span>
              <span>奖励</span>
              <span>Â = r−r̄</span>
            </div>
            {active.items.map((item) => (
              <div key={item.id} className={styles.row}>
                <b>{item.id}</b>
                <span>{item.name}</span>
                <i
                  className={styles.bar}
                  data-positive={item.advantage >= 0 ? "true" : "false"}
                  style={{
                    width: revealed
                      ? `${Math.min(100, Math.abs(item.advantage) * 180)}%`
                      : "0%",
                  }}
                />
                <span>{revealed ? item.reward.toFixed(3) : "—"}</span>
                <span>
                  {revealed
                    ? `${item.advantage >= 0 ? "+" : ""}${item.advantage.toFixed(3)}`
                    : "—"}
                </span>
              </div>
            ))}
          </div>

          <dl className={styles.metrics}>
            <div>
              <dt>稀疏方差</dt>
              <dd>{revealed ? sparse.variance.toFixed(4) : "—"}</dd>
            </div>
            <div>
              <dt>稀疏 Σ|Â|</dt>
              <dd>{revealed ? sparse.updateNorm.toFixed(3) : "—"}</dd>
            </div>
            <div>
              <dt>dense 方差</dt>
              <dd>{revealed ? dense.variance.toFixed(4) : "—"}</dd>
            </div>
            <div>
              <dt>dense Σ|Â|</dt>
              <dd>{revealed ? dense.updateNorm.toFixed(3) : "—"}</dd>
            </div>
          </dl>
          <p className={styles.formula}>
            r_sparse = I[lifted ∧ in_bin]；r_dense = clip(0.35 e^(−c/τ) + 0.25 I_contact + 0.40 I_lift − 0.30 [F−F_safe]_+, 0, 1)；Â_i = r_i − r̄
          </p>
        </div>
      </div>

      <div className={styles.predict}>
        <fieldset>
          <legend>先预测：四条都没放入时，哪句话成立？</legend>
          {PREDICTIONS.map((option) => (
            <label key={option.value}>
              <input
                type="radio"
                name="vla-rl-prediction"
                value={option.value}
                checked={prediction === option.value}
                onChange={() => {
                  setPrediction(option.value);
                  setRevealed(false);
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
            onClick={() => setRevealed(true)}
          >
            揭晓优势
          </button>
        </div>
      </div>
      {revealed && prediction !== "sparse_zero_dense_rank" && (
        <p className={styles.feedback}>
          稀疏成功在全失败组里全是 0，减均值还是 0。接触 dense 仍能把擦边排到空抓前面。
        </p>
      )}
      {revealed && prediction === "sparse_zero_dense_rank" && batch !== "fail" && (
        <p className={styles.feedback}>
          预测句针对失败批次。切回“全失败”，再揭晓一次。
        </p>
      )}
      <Gate passed={passed}>
        先选对失败批次上稀疏优势全零、dense 仍有更新，再在全失败批次上揭晓。教学模拟，不是模型输出。
      </Gate>
    </LabFrame>
  );
}
