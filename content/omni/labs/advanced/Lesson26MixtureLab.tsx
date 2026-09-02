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
import styles from "./Lesson26MixtureLab.module.css";

type Probe = "intact" | "drop_language" | "shuffle_instruction" | "shuffle_embodiment";
type Prediction = "alpha1" | "alpha0-kills" | "cap-kills";

type Domain = {
  id: string;
  name: string;
  n: number;
  action: string;
  camera: string;
  cloth: string;
  robot: string;
};

const DOMAINS: Domain[] = [
  {
    id: "google",
    name: "Google 厨房",
    n: 10000,
    action: "7D Δee",
    camera: "第三人称",
    cloth: "白桌",
    robot: "G",
  },
  {
    id: "franka",
    name: "Franka 桌面",
    n: 1000,
    action: "7D Δee",
    camera: "第三人称+腕",
    cloth: "木纹",
    robot: "F",
  },
  {
    id: "widowx",
    name: "WidowX 水槽",
    n: 200,
    action: "7D Δee",
    camera: "第三人称",
    cloth: "蓝布",
    robot: "W",
  },
  {
    id: "bimanual",
    name: "双臂装配",
    n: 100,
    action: "关节",
    camera: "双目",
    cloth: "灰台",
    robot: "B",
  },
];

const BATCH = 256;
const ALPHA1_SHARE_GATE = 0.75;
const SMALL_SHARE_GATE = 0.08;
const LEAK_ACCURACY_GATE = 0.8;

const PREDICTIONS: Array<[Prediction, string]> = [
  ["alpha1", "α=1 且不加上限时，最大域会超过 batch 的 75%"],
  ["alpha0-kills", "α=0 会让最小域比按条数采样更少出现"],
  ["cap-kills", "加上每域上限会进一步压掉最小域"],
];

const PROBES: Array<[Probe, string]> = [
  ["intact", "完整输入"],
  ["drop_language", "去掉语言"],
  ["shuffle_instruction", "打乱指令"],
  ["shuffle_embodiment", "打乱机体 ID"],
];

function mixture(alpha: number, cap: number) {
  const capped = DOMAINS.map((domain) => Math.min(domain.n, cap));
  const weights = capped.map((count) => count ** alpha);
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  const probs = weights.map((weight) => weight / total);
  const shares = Object.fromEntries(
    DOMAINS.map((domain, index) => [domain.id, probs[index]]),
  );
  const maxShare = Math.max(...probs);
  const minShare = Math.min(...probs);
  const effective = 1 / probs.reduce((sum, probability) => sum + probability * probability, 0);
  return { probs, shares, maxShare, minShare, effective, capped };
}

function probeAccuracy(leak: boolean, probe: Probe) {
  if (probe === "intact") return leak ? 0.93 : 0.84;
  if (probe === "drop_language" || probe === "shuffle_instruction") {
    return leak ? 0.89 : 0.31;
  }
  return leak ? 0.28 : 0.83;
}

export function Lesson26MixtureLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    alpha: numberFrom(initialState, "alpha", 1, 0, 1),
    cap: numberFrom(initialState, "cap", 10000, 100, 10000),
    leak: stringFrom(initialState, "leak", "off") === "on",
    probe: stringFrom(initialState, "probe", "intact") as Probe,
    prediction: stringFrom(initialState, "prediction", "") as Prediction | "",
  };
  const [alpha, setAlpha] = useState(defaults.alpha);
  const [cap, setCap] = useState(defaults.cap);
  const [leak, setLeak] = useState(defaults.leak);
  const [probe, setProbe] = useState<Probe>(
    ["intact", "drop_language", "shuffle_instruction", "shuffle_embodiment"].includes(
      defaults.probe,
    )
      ? defaults.probe
      : "intact",
  );
  const [prediction, setPrediction] = useState<Prediction | "">(
    defaults.prediction === "alpha1" ||
      defaults.prediction === "alpha0-kills" ||
      defaults.prediction === "cap-kills"
      ? defaults.prediction
      : "",
  );
  const [ran, setRan] = useState(false);
  const [seenAlpha1, setSeenAlpha1] = useState(false);
  const [seenCapRecover, setSeenCapRecover] = useState(false);
  const [seenLeak, setSeenLeak] = useState(false);

  const calculation = useMemo(() => mixture(alpha, cap), [alpha, cap]);
  const accuracy = probeAccuracy(leak, probe);

  const passed =
    ran &&
    prediction === "alpha1" &&
    seenAlpha1 &&
    seenCapRecover &&
    seenLeak;

  const completion = useMemo(
    () => ({
      lessonId: 26,
      alpha,
      cap,
      leak,
      probe,
      prediction,
      maxShare: round(calculation.maxShare, 4),
      minShare: round(calculation.minShare, 4),
      effectiveDomains: round(calculation.effective, 3),
      probeAccuracy: accuracy,
      seenAlpha1,
      seenCapRecover,
      seenLeak,
    }),
    [
      accuracy,
      alpha,
      calculation.effective,
      calculation.maxShare,
      calculation.minShare,
      cap,
      leak,
      prediction,
      probe,
      seenAlpha1,
      seenCapRecover,
      seenLeak,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function reveal() {
    const next = mixture(alpha, cap);
    const nextAccuracy = probeAccuracy(leak, probe);
    if (Math.abs(alpha - 1) < 1e-9 && cap >= 10000 && next.maxShare >= ALPHA1_SHARE_GATE) {
      setSeenAlpha1(true);
    }
    if (cap <= 400 && next.minShare >= SMALL_SHARE_GATE) {
      setSeenCapRecover(true);
    }
    if (
      leak &&
      (probe === "drop_language" || probe === "shuffle_instruction") &&
      nextAccuracy >= LEAK_ACCURACY_GATE
    ) {
      setSeenLeak(true);
    }
    setRan(true);
  }

  function reset() {
    setAlpha(1);
    setCap(10000);
    setLeak(false);
    setProbe("intact");
    setPrediction("");
    setRan(false);
    setSeenAlpha1(false);
    setSeenCapRecover(false);
    setSeenLeak(false);
  }

  const hidden = !ran;

  return (
    <LabFrame
      lesson="26"
      title="配一锅异构机体数据并抓捷径"
      description="四个域的条数差两个数量级。先预测再揭晓：调 α 和每域上限，看 batch 组成；打开机体 ID 泄漏，看去掉语言后准确率是否几乎不变。教学模拟，不是模型输出。"
    >
      <div className={styles.workspace}>
        <form className={styles.controls} onSubmit={(event) => event.preventDefault()}>
          <h3>混合控制台</h3>
          <label>
            <span>
              温度 α <output>{alpha.toFixed(2)}</output>
            </span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={alpha}
              onChange={(event) => {
                setAlpha(Number(event.target.value));
                setRan(false);
              }}
            />
          </label>
          <label>
            <span>
              每域上限 C <output>{cap}</output>
            </span>
            <input
              type="range"
              min="100"
              max="10000"
              step="100"
              value={cap}
              onChange={(event) => {
                setCap(Number(event.target.value));
                setRan(false);
              }}
            />
          </label>
          <fieldset className={styles.fieldset}>
            <legend>机体 ID 泄漏</legend>
            <div className={styles.toggleRow}>
              {[
                [false, "关闭"],
                [true, "写入 batch"],
              ].map(([value, label]) => (
                <label key={String(value)}>
                  <input
                    type="radio"
                    name="leak"
                    checked={leak === value}
                    onChange={() => {
                      setLeak(Boolean(value));
                      setRan(false);
                    }}
                  />
                  <span>{label}</span>
                </label>
              ))}
            </div>
          </fieldset>
          <fieldset className={styles.fieldset}>
            <legend>负对照</legend>
            <div className={styles.toggleRow}>
              {PROBES.map(([value, label]) => (
                <label key={value}>
                  <input
                    type="radio"
                    name="probe"
                    checked={probe === value}
                    onChange={() => {
                      setProbe(value);
                      setRan(false);
                    }}
                  />
                  <span>{label}</span>
                </label>
              ))}
            </div>
          </fieldset>
        </form>

        <div className={styles.stage}>
          <div className={styles.formula}>
            <span>p_d ∝ min(n_d, C)^α · batch={BATCH}</span>
            <strong>
              α={alpha.toFixed(2)}, C={cap}, D_eff=
              {hidden ? "—" : calculation.effective.toFixed(2)}
            </strong>
          </div>
          <div className={styles.domains} aria-label="四域 batch 组成">
            {DOMAINS.map((domain, index) => {
              const share = calculation.probs[index];
              return (
                <div className={styles.domain} key={domain.id}>
                  <div className={styles.domainHeader}>
                    <b>{domain.name}</b>
                    <small>
                      n={domain.n} · {domain.action} · {domain.cloth}
                    </small>
                  </div>
                  <div className={styles.barTrack}>
                    <div
                      className={styles.barFill}
                      style={{ width: hidden ? "0%" : `${Math.max(2, share * 100)}%` }}
                    />
                  </div>
                  <p className={styles.share}>
                    {hidden ? "—" : `${(share * 100).toFixed(1)}%`}
                  </p>
                </div>
              );
            })}
          </div>
          <dl className={styles.metrics}>
            <div>
              <dt>最大域占比</dt>
              <dd>{hidden ? "—" : `${(calculation.maxShare * 100).toFixed(1)}%`}</dd>
            </div>
            <div>
              <dt>最小域占比</dt>
              <dd>{hidden ? "—" : `${(calculation.minShare * 100).toFixed(1)}%`}</dd>
            </div>
            <div>
              <dt>小域期望条数</dt>
              <dd>
                {hidden
                  ? "—"
                  : round(calculation.minShare * BATCH, 2).toFixed(2)}
              </dd>
            </div>
            <div>
              <dt>去掉语言后准确率</dt>
              <dd>
                {hidden
                  ? "—"
                  : `${(probeAccuracy(leak, "drop_language") * 100).toFixed(0)}%`}
              </dd>
            </div>
          </dl>
          <div className={styles.probe} data-leak={leak}>
            <strong>
              当前对照：{PROBES.find(([value]) => value === probe)?.[1]}
            </strong>
            <p>
              {hidden
                ? "先选预测，再揭晓 batch 与探针准确率。"
                : `玩具政策准确率 ${(accuracy * 100).toFixed(0)}%。泄漏${
                    leak ? "已打开" : "关闭"
                  }。机体 ID 来自机器人外观占位符 ${DOMAINS.map((domain) => domain.robot).join("/")}。`}
            </p>
          </div>
          <ul className={styles.checklist}>
            <li data-done={seenAlpha1}>α=1 最大域 &gt; 75%</li>
            <li data-done={seenCapRecover}>cap 后小域回升</li>
            <li data-done={seenLeak}>机体 ID 泄漏可触发</li>
          </ul>
        </div>
      </div>

      <div className={styles.predict}>
        <fieldset>
          <legend>先预测：默认四域计数下，哪句话一定成立？</legend>
          {PREDICTIONS.map(([value, label]) => (
            <label key={value}>
              <input
                type="radio"
                name="mixture-prediction"
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
            onClick={reveal}
          >
            揭晓 batch
          </button>
        </div>
      </div>
      {ran && prediction !== "alpha1" && (
        <p className={styles.feedback}>
          先看条数：10000 / 1000 / 200 / 100 在 α=1 时最大域约占 88%。α=0
          是对域均匀；上限截断大域后，小域份额只会升、不会降。
        </p>
      )}
      {ran && prediction === "alpha1" && !passed && (
        <p className={styles.feedback}>
          预测对了。请分别揭晓三组：α=1 且 C=10000；把 C 降到 400 附近；打开机体
          ID 泄漏并选“去掉语言”或“打乱指令”。
        </p>
      )}
      <Gate passed={passed}>
        先提交正确预测，再触发三件事：α=1 时最大域过高、加 cap
        后小域回升、机体 ID 泄漏在去掉语言后准确率几乎不变。数字来自教学夹具，不是真机成功率。
      </Gate>
    </LabFrame>
  );
}
