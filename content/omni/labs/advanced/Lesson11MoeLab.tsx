"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import {
  entropy,
  numberFrom,
  round,
  softmax,
  stringFrom,
  useCompletionGate,
} from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson11MoeLab.module.css";

type Dispatch = {
  token: number;
  expert: number;
  probability: number;
  accepted: boolean;
};

const tokenLabels = [
  "red",
  "谱",
  "dog",
  "笑",
  "42",
  "雨",
  "go",
  "音",
  "blue",
  "猫",
  "why",
  "帧",
  "7",
  "风",
  "run",
  "像",
  "green",
  "光",
  "who",
  "声",
  "9",
  "云",
  "stop",
  "图",
];

export function Lesson11MoeLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    tokens: numberFrom(initialState, "tokens", 16, 8, 24),
    experts: numberFrom(initialState, "experts", 4, 2, 8),
    topK: numberFrom(initialState, "topK", 2, 1, 2),
    capacityFactor: numberFrom(initialState, "capacityFactor", 1.25, 0.5, 2),
    temperature: numberFrom(initialState, "temperature", 0.85, 0.4, 1.6),
    prediction: stringFrom(initialState, "prediction", ""),
  };
  const [tokens, setTokens] = useState(defaults.tokens);
  const [experts, setExperts] = useState(defaults.experts);
  const [topK, setTopK] = useState(defaults.topK);
  const [capacityFactor, setCapacityFactor] = useState(
    defaults.capacityFactor,
  );
  const [temperature, setTemperature] = useState(defaults.temperature);
  const [prediction, setPrediction] = useState(defaults.prediction);
  const [ran, setRan] = useState(false);

  const simulation = useMemo(() => {
    const capacity = Math.ceil((capacityFactor * tokens * topK) / experts);
    const loads = Array.from({ length: experts }, () => 0);
    const dispatches: Dispatch[] = [];

    for (let token = 0; token < tokens; token += 1) {
      const logits = Array.from({ length: experts }, (_, expert) => {
        const periodic = Math.sin((token + 1) * (expert + 2) * 1.37);
        const languageBias = ((token * 7 + expert * 3) % 11) / 8;
        const expertBias = expert === token % Math.max(2, experts - 1) ? 0.7 : 0;
        return (periodic + languageBias + expertBias) / temperature;
      });
      const probabilities = softmax(logits);
      const selected = probabilities
        .map((probability, expert) => ({ probability, expert }))
        .sort((a, b) => b.probability - a.probability)
        .slice(0, topK);

      selected.forEach(({ expert, probability }) => {
        const accepted = loads[expert] < capacity;
        if (accepted) loads[expert] += 1;
        dispatches.push({ token, expert, probability, accepted });
      });
    }

    const accepted = loads.reduce((sum, load) => sum + load, 0);
    const distribution =
      accepted === 0 ? loads.map(() => 0) : loads.map((load) => load / accepted);
    const normalizedEntropy =
      experts > 1 ? entropy(distribution) / Math.log(experts) : 1;

    return {
      capacity,
      loads,
      dispatches,
      dropped: tokens * topK - accepted,
      entropy: normalizedEntropy,
    };
  }, [capacityFactor, experts, temperature, tokens, topK]);

  const passed =
    ran && prediction === "capacity" && simulation.dropped === 0;
  const completion = useMemo(
    () => ({
      lessonId: 11,
      tokens,
      experts,
      topK,
      capacityFactor,
      temperature,
      dropped: simulation.dropped,
      normalizedEntropy: round(simulation.entropy, 3),
    }),
    [
      capacityFactor,
      experts,
      simulation.dropped,
      simulation.entropy,
      temperature,
      tokens,
      topK,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function reset() {
    setTokens(defaults.tokens);
    setExperts(defaults.experts);
    setTopK(defaults.topK);
    setCapacityFactor(defaults.capacityFactor);
    setTemperature(defaults.temperature);
    setPrediction("");
    setRan(false);
  }

  return (
    <LabFrame
      lesson="11"
      title="亲手调度一个稀疏 MoE"
      description="路由器先为每个 token 计算专家概率，再应用 top-k 和专家容量限制。调整参数，检查 token 在哪一步被丢弃。"
    >
      <div className={styles.workspace}>
        <form className={styles.controls} onSubmit={(event) => event.preventDefault()}>
          <h3>路由控制台</h3>
          <label>
            <span>Token 数 <output>{tokens}</output></span>
            <input
              type="range"
              min="8"
              max="24"
              step="4"
              value={tokens}
              onChange={(event) => {
                setTokens(Number(event.target.value));
                setRan(false);
              }}
            />
          </label>
          <label>
            <span>专家数 <output>{experts}</output></span>
            <input
              type="range"
              min="2"
              max="8"
              step="1"
              value={experts}
              onChange={(event) => {
                const value = Number(event.target.value);
                setExperts(value);
                setTopK((current) => Math.min(current, value));
                setRan(false);
              }}
            />
          </label>
          <label>
            <span>Top-k <output>{topK}</output></span>
            <input
              type="range"
              min="1"
              max={Math.min(2, experts)}
              step="1"
              value={topK}
              onChange={(event) => {
                setTopK(Number(event.target.value));
                setRan(false);
              }}
            />
          </label>
          <label>
            <span>容量因子 <output>{capacityFactor.toFixed(2)}</output></span>
            <input
              type="range"
              min="0.5"
              max="2"
              step="0.05"
              value={capacityFactor}
              onChange={(event) => {
                setCapacityFactor(Number(event.target.value));
                setRan(false);
              }}
            />
          </label>
          <label>
            <span>路由温度 <output>{temperature.toFixed(2)}</output></span>
            <input
              type="range"
              min="0.4"
              max="1.6"
              step="0.05"
              value={temperature}
              onChange={(event) => {
                setTemperature(Number(event.target.value));
                setRan(false);
              }}
            />
          </label>
        </form>

        <div className={styles.stage}>
          <div className={styles.formula}>
            <span>C = ceil(capacity factor × T × k ÷ E)</span>
            <strong>
              C = ceil({capacityFactor.toFixed(2)} × {tokens} × {topK} ÷{" "}
              {experts}) = {simulation.capacity}
            </strong>
          </div>
          <div className={styles.expertGrid} aria-label="专家负载">
            {simulation.loads.map((load, expert) => (
              <div
                className={styles.expert}
                key={expert}
                style={{ "--fill": `${Math.min(100, (load / simulation.capacity) * 100)}%` } as React.CSSProperties}
              >
                <span>E{expert}</span>
                <b>{ran ? load : "–"}</b>
                <small>/{simulation.capacity}</small>
              </div>
            ))}
          </div>
          <div className={styles.tokenField} aria-label="token 路由结果">
            {Array.from({ length: tokens }, (_, token) => {
              const routes = simulation.dispatches.filter(
                (dispatch) => dispatch.token === token,
              );
              return (
                <div className={styles.token} key={token}>
                  <b>{tokenLabels[token]}</b>
                  <span>
                    {ran
                      ? routes
                          .map((route) =>
                            route.accepted ? `E${route.expert}` : `E${route.expert}×`,
                          )
                          .join(" · ")
                      : "待路由"}
                  </span>
                </div>
              );
            })}
          </div>
          <dl className={styles.metrics}>
            <div>
              <dt>丢弃 assignment</dt>
              <dd>{ran ? simulation.dropped : "—"}</dd>
            </div>
            <div>
              <dt>归一化负载熵</dt>
              <dd>{ran ? simulation.entropy.toFixed(3) : "—"}</dd>
            </div>
            <div>
              <dt>总容量</dt>
              <dd>{experts * simulation.capacity}</dd>
            </div>
          </dl>
        </div>
      </div>

      <div className={styles.predict}>
        <fieldset>
          <legend>先预测：保持当前 logits 不变，哪个旋钮必然不会增加丢弃数？</legend>
          {[
            ["temperature", "升高温度"],
            ["capacity", "升高容量因子"],
            ["topk", "升高 top-k"],
          ].map(([value, label]) => (
            <label key={value}>
              <input
                type="radio"
                name="moe-prediction"
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
            运行 dispatch
          </button>
        </div>
      </div>
      {ran && prediction !== "capacity" && (
        <p className={styles.feedback}>
          再看容量上限：温度会改路由排序，top-k 会增加 assignment；只有提高 C
          保证每个专家可接收的数量不下降。
        </p>
      )}
      <Gate passed={passed}>
        先选择正确预测，再把丢弃数调到 0。熵只描述负载均衡程度，不是“模型质量分数”。
      </Gate>
    </LabFrame>
  );
}
