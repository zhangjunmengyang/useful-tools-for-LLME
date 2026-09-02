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
import styles from "./Lesson12HybridLayerLab.module.css";

type LayerKind = "local" | "global";

function buildPattern(total: number, every: number): LayerKind[] {
  return Array.from({ length: total }, (_, index) =>
    (index + 1) % every === 0 ? "global" : "local",
  );
}

export function Lesson12HybridLayerLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    total: numberFrom(initialState, "total", 24, 8, 40),
    every: numberFrom(initialState, "globalEvery", 6, 2, 10),
    context: numberFrom(initialState, "context", 8192, 2048, 32768),
    window: numberFrom(initialState, "localWindow", 512, 128, 2048),
    prediction: stringFrom(initialState, "prediction", ""),
  };
  const [total, setTotal] = useState(defaults.total);
  const [every, setEvery] = useState(defaults.every);
  const [context, setContext] = useState(defaults.context);
  const [window, setWindow] = useState(defaults.window);
  const [layers, setLayers] = useState<LayerKind[]>(() =>
    buildPattern(defaults.total, defaults.every),
  );
  const [prediction, setPrediction] = useState(defaults.prediction);
  const [ran, setRan] = useState(false);

  const calculation = useMemo(() => {
    const globalCount = layers.filter((layer) => layer === "global").length;
    const localCount = layers.length - globalCount;
    const standardSlots = layers.length * context;
    const hybridSlots =
      globalCount * context + localCount * Math.min(context, window);
    const saving =
      standardSlots === 0 ? 0 : 1 - hybridSlots / standardSlots;
    return {
      globalCount,
      localCount,
      standardSlots,
      hybridSlots,
      saving,
    };
  }, [context, layers, window]);

  const passed =
    ran &&
    prediction === "global" &&
    calculation.saving >= 0.5 &&
    calculation.globalCount > 0 &&
    calculation.localCount > 0;
  const completion = useMemo(
    () => ({
      lessonId: 12,
      total: layers.length,
      globalEvery: every,
      globalLayers: calculation.globalCount,
      localLayers: calculation.localCount,
      context,
      localWindow: window,
      layerKinds: layers,
      cacheSavingPercent: round(calculation.saving * 100, 1),
    }),
    [calculation, context, every, layers, window],
  );
  useCompletionGate(passed, onComplete, completion);

  function applyPattern(nextTotal = total, nextEvery = every) {
    setLayers(buildPattern(nextTotal, nextEvery));
    setRan(false);
  }

  function reset() {
    setTotal(defaults.total);
    setEvery(defaults.every);
    setContext(defaults.context);
    setWindow(defaults.window);
    setLayers(buildPattern(defaults.total, defaults.every));
    setPrediction("");
    setRan(false);
  }

  return (
    <LabFrame
      lesson="12"
      title="搭一座 Hybrid Attention 塔"
      description="全局层看完整上下文，局部层只保留滑窗。点击任意楼层改变类型，再用真实 token-slot 账本比较 KV cache。"
    >
      <div className={styles.builder}>
        <section className={styles.inspector} aria-label="结构参数">
          <h3>结构参数</h3>
          <label>
            <span>层数</span>
            <select
              value={total}
              onChange={(event) => {
                const value = Number(event.target.value);
                setTotal(value);
                setLayers(buildPattern(value, every));
                setRan(false);
              }}
            >
              {[8, 12, 16, 24, 32, 40].map((value) => (
                <option key={value}>{value}</option>
              ))}
            </select>
          </label>
          <label>
            <span>每 N 层放一个全局层</span>
            <select
              value={every}
              onChange={(event) => {
                const value = Number(event.target.value);
                setEvery(value);
                applyPattern(total, value);
              }}
            >
              {[2, 3, 4, 5, 6, 8, 10].map((value) => (
                <option key={value}>{value}</option>
              ))}
            </select>
          </label>
          <button type="button" onClick={() => applyPattern()}>
            重新铺设周期
          </button>
          <label className={styles.range}>
            <span>
              上下文 <output>{context.toLocaleString()}</output>
            </span>
            <input
              type="range"
              min="2048"
              max="32768"
              step="2048"
              value={context}
              onChange={(event) => {
                setContext(Number(event.target.value));
                setRan(false);
              }}
            />
          </label>
          <label className={styles.range}>
            <span>
              局部窗口 <output>{window.toLocaleString()}</output>
            </span>
            <input
              type="range"
              min="128"
              max="2048"
              step="128"
              value={window}
              onChange={(event) => {
                setWindow(Number(event.target.value));
                setRan(false);
              }}
            />
          </label>
        </section>

        <section className={styles.tower}>
          <div className={styles.towerHead}>
            <div>
              <h3>Layer builder</h3>
              <p>点一层可在 Local / Global 间切换</p>
            </div>
            <div className={styles.legend}>
              <span><i className={styles.localDot} /> Local</span>
              <span><i className={styles.globalDot} /> Global</span>
            </div>
          </div>
          <div className={styles.layers} aria-label="可编辑注意力层">
            {layers.map((kind, index) => (
              <button
                type="button"
                className={kind === "global" ? styles.global : styles.local}
                aria-label={`第 ${index + 1} 层，当前为${kind === "global" ? "全局" : "局部"}注意力，点击切换`}
                aria-pressed={kind === "global"}
                key={index}
                onClick={() => {
                  setLayers((current) =>
                    current.map((layer, layerIndex) =>
                      layerIndex === index
                        ? layer === "global"
                          ? "local"
                          : "global"
                        : layer,
                    ),
                  );
                  setRan(false);
                }}
              >
                <span>L{String(index + 1).padStart(2, "0")}</span>
                <b>{kind === "global" ? "G" : "L"}</b>
              </button>
            ))}
          </div>
        </section>
      </div>

      <section className={styles.ledger} aria-label="KV cache 账本">
        <div className={styles.equation}>
          <p>标准：L × S</p>
          <strong>{total} × {context.toLocaleString()} = {calculation.standardSlots.toLocaleString()}</strong>
        </div>
        <div className={styles.equation}>
          <p>混合：G × S + Local × W</p>
          <strong>
            {calculation.globalCount} × {context.toLocaleString()} +{" "}
            {calculation.localCount} × {Math.min(context, window).toLocaleString()}
          </strong>
        </div>
        <div className={styles.saving}>
          <span>token-slot 节省</span>
          <strong>{ran ? `${(calculation.saving * 100).toFixed(1)}%` : "—"}</strong>
          <div aria-hidden="true">
            <i style={{ width: ran ? `${calculation.saving * 100}%` : "0%" }} />
          </div>
        </div>
      </section>

      <div className={styles.challenge}>
        <fieldset>
          <legend>先预测：当 S 翻倍且 S 仍大于 W，哪些层的 cache 会跟着翻倍？</legend>
          {[
            ["all", "所有层"],
            ["local", "仅 Local 层"],
            ["global", "仅 Global 层"],
          ].map(([value, label]) => (
            <label key={value}>
              <input
                type="radio"
                name="hybrid-prediction"
                checked={prediction === value}
                onChange={() => {
                  setPrediction(value);
                  setRan(false);
                }}
              />
              {label}
            </label>
          ))}
        </fieldset>
        <div>
          <button type="button" onClick={reset}>重置</button>
          <button
            type="button"
            className={styles.primary}
            disabled={!prediction}
            onClick={() => setRan(true)}
          >
            结算 cache
          </button>
        </div>
      </div>
      <Gate passed={passed}>
        保留至少一个 Global 和一个 Local 层，预测正确，并搭出节省不低于 50% 的结构。
      </Gate>
    </LabFrame>
  );
}
