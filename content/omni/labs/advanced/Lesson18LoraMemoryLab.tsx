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
import styles from "./Lesson18LoraMemoryLab.module.css";

export function Lesson18LoraMemoryLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    rank: numberFrom(initialState, "rank", 16, 4, 128),
    sequence: numberFrom(initialState, "sequence", 8192, 2048, 32768),
    microbatch: numberFrom(initialState, "microbatch", 1, 1, 4),
    precision: numberFrom(initialState, "baseBits", 4, 4, 16),
    gpuCount: numberFrom(initialState, "gpuCount", 8, 1, 8),
    gpuMemory: numberFrom(initialState, "gpuMemory", 24, 16, 80),
    checkpointing:
      typeof initialState?.checkpointing === "boolean"
        ? initialState.checkpointing
        : true,
    zero:
      typeof initialState?.zeroOptimizer === "boolean"
        ? initialState.zeroOptimizer
        : true,
    prediction: stringFrom(initialState, "prediction", ""),
  };
  const [rank, setRank] = useState(defaults.rank);
  const [sequence, setSequence] = useState(defaults.sequence);
  const [microbatch, setMicrobatch] = useState(defaults.microbatch);
  const [precision, setPrecision] = useState(defaults.precision);
  const [gpuCount, setGpuCount] = useState(defaults.gpuCount);
  const [gpuMemory, setGpuMemory] = useState(defaults.gpuMemory);
  const [checkpointing, setCheckpointing] = useState(defaults.checkpointing);
  const [zero, setZero] = useState(defaults.zero);
  const [prediction, setPrediction] = useState(defaults.prediction);
  const [ran, setRan] = useState(false);

  const calculation = useMemo(() => {
    const totalParametersB = 31;
    const hidden = 4096;
    const layers = 32;
    const targetMatrices = 4;
    const loraParameters =
      2 * rank * hidden * targetMatrices * layers;
    const basePerGpu =
      (totalParametersB * precision) / 8 / gpuCount;
    const adapterStateTotal = (loraParameters * 12) / 1e9;
    const adapterPerGpu = zero
      ? adapterStateTotal / gpuCount
      : adapterStateTotal;
    const retainedActivationTensors = checkpointing ? 2 : 6;
    const activationPerGpu =
      (layers *
        sequence *
        microbatch *
        hidden *
        2 *
        retainedActivationTensors) /
      1e9;
    const subtotal = basePerGpu + adapterPerGpu + activationPerGpu;
    const reserve = subtotal * 0.15;
    const total = subtotal + reserve;
    return {
      totalParametersB,
      loraParameters,
      basePerGpu,
      adapterStateTotal,
      adapterPerGpu,
      activationPerGpu,
      retainedActivationTensors,
      reserve,
      total,
      fits: total <= gpuMemory,
    };
  }, [
    checkpointing,
    gpuCount,
    microbatch,
    precision,
    rank,
    sequence,
    zero,
    gpuMemory,
  ]);

  const passed =
    ran &&
    prediction === "quantize" &&
    calculation.fits &&
    precision === 4;
  const completion = useMemo(
    () => ({
      lessonId: 18,
      checkpoint: "Nemotron-3-Nano-Omni-30B-A3B",
      gpuCount,
      gpuMemoryGb: gpuMemory,
      baseBits: precision,
      rank,
      sequence,
      microbatch,
      checkpointing,
      zeroOptimizer: zero,
      estimatedGbPerGpu: round(calculation.total, 2),
      fits: calculation.fits,
    }),
    [
      calculation.fits,
      calculation.total,
      checkpointing,
      gpuCount,
      gpuMemory,
      microbatch,
      precision,
      rank,
      sequence,
      zero,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function reset() {
    setRank(defaults.rank);
    setSequence(defaults.sequence);
    setMicrobatch(defaults.microbatch);
    setPrecision(defaults.precision);
    setGpuCount(defaults.gpuCount);
    setGpuMemory(defaults.gpuMemory);
    setCheckpointing(defaults.checkpointing);
    setZero(defaults.zero);
    setPrediction("");
    setRan(false);
  }

  return (
    <LabFrame
      lesson="18"
      title="Nemotron LoRA 显存预检"
      description="先别启动训练。用一个公开、可改的上界模型拆开冻结权重、LoRA 状态、activation 和预留空间，判断你的 8 卡配置能否进场。"
    >
      <div className={styles.preflight}>
        <section className={styles.config}>
          <header>
            <h3>30B-A3B checkpoint</h3>
            <span>31B total · ~3B active / token</span>
          </header>
          <div className={styles.fieldGrid}>
            <label>
              <span>冻结权重精度</span>
              <select
                value={precision}
                onChange={(event) => {
                  setPrecision(Number(event.target.value));
                  setRan(false);
                }}
              >
                <option value="16">BF16 · 16 bit</option>
                <option value="8">INT8 · 8 bit</option>
                <option value="4">4-bit · 理论值</option>
              </select>
            </label>
            <label>
              <span>GPU 数</span>
              <select
                value={gpuCount}
                onChange={(event) => {
                  setGpuCount(Number(event.target.value));
                  setRan(false);
                }}
              >
                {[1, 2, 4, 8].map((value) => (
                  <option key={value}>{value}</option>
                ))}
              </select>
            </label>
            <label>
              <span>单卡显存</span>
              <select
                value={gpuMemory}
                onChange={(event) => {
                  setGpuMemory(Number(event.target.value));
                  setRan(false);
                }}
              >
                {[16, 24, 32, 48, 80].map((value) => (
                  <option key={value}>{value} GB</option>
                ))}
              </select>
            </label>
            <label>
              <span>Micro batch</span>
              <select
                value={microbatch}
                onChange={(event) => {
                  setMicrobatch(Number(event.target.value));
                  setRan(false);
                }}
              >
                {[1, 2, 3, 4].map((value) => (
                  <option key={value}>{value}</option>
                ))}
              </select>
            </label>
          </div>
          <label className={styles.range}>
            <span>LoRA rank <output>{rank}</output></span>
            <input
              type="range"
              min="4"
              max="128"
              step="4"
              value={rank}
              onChange={(event) => {
                setRank(Number(event.target.value));
                setRan(false);
              }}
            />
          </label>
          <label className={styles.range}>
            <span>序列长度 <output>{sequence.toLocaleString()}</output></span>
            <input
              type="range"
              min="2048"
              max="32768"
              step="2048"
              value={sequence}
              onChange={(event) => {
                setSequence(Number(event.target.value));
                setRan(false);
              }}
            />
          </label>
          <div className={styles.switches}>
            <label>
              <input
                type="checkbox"
                checked={checkpointing}
                onChange={(event) => {
                  setCheckpointing(event.target.checked);
                  setRan(false);
                }}
              />
              Activation checkpointing
            </label>
            <label>
              <input
                type="checkbox"
                checked={zero}
                onChange={(event) => {
                  setZero(event.target.checked);
                  setRan(false);
                }}
              />
              Shard LoRA optimizer states
            </label>
          </div>
        </section>

        <section className={styles.memory}>
          <header>
            <div>
              <h3>Per-GPU memory envelope</h3>
              <p>{ran ? `${calculation.total.toFixed(2)} / ${gpuMemory} GB` : "等待预检"}</p>
            </div>
            <strong
              className={
                !ran ? styles.pending : calculation.fits ? styles.pass : styles.fail
              }
            >
              {!ran ? "PENDING" : calculation.fits ? "PASS" : "OOM"}
            </strong>
          </header>
          <div className={styles.memoryBar} aria-label="显存组成">
            {[
              ["base", calculation.basePerGpu, "#47755d"],
              ["lora", calculation.adapterPerGpu, "#8a6949"],
              ["activation", calculation.activationPerGpu, "#52758d"],
              ["reserve", calculation.reserve, "#9c9279"],
            ].map(([key, value, color]) => (
              <i
                key={String(key)}
                style={{
                  width: ran
                    ? `${Math.min(100, (Number(value) / gpuMemory) * 100)}%`
                    : "0%",
                  background: String(color),
                }}
              />
            ))}
          </div>
          <dl className={styles.breakdown}>
            <div>
              <dt>冻结 base / GPU</dt>
              <dd>{ran ? `${calculation.basePerGpu.toFixed(2)} GB` : "—"}</dd>
              <small>31B × {precision}/8 ÷ {gpuCount}</small>
            </div>
            <div>
              <dt>LoRA param + grad + Adam</dt>
              <dd>{ran ? `${calculation.adapterPerGpu.toFixed(3)} GB` : "—"}</dd>
              <small>2rH × 4 matrices × 32 layers × 12B</small>
            </div>
            <div>
              <dt>Activation / GPU</dt>
              <dd>{ran ? `${calculation.activationPerGpu.toFixed(2)} GB` : "—"}</dd>
              <small>32 × S × B × 4096 × 2B × {calculation.retainedActivationTensors}</small>
            </div>
            <div>
              <dt>15% runtime reserve</dt>
              <dd>{ran ? `${calculation.reserve.toFixed(2)} GB` : "—"}</dd>
              <small>kernel workspace / fragmentation 预算</small>
            </div>
          </dl>
          <p className={styles.disclaimer}>
            这是显式教学估算，不是框架实测峰值；真实开跑仍需用同一 batch 做一次 dry-run。4-bit
            一栏是纯 tensor 理论值，不等同于具体 checkpoint 文件大小。
          </p>
        </section>
      </div>

      <div className={styles.challenge}>
        <fieldset>
          <legend>先预测：哪个动作会把 31B 冻结权重的理论字节数直接降为 BF16 的 1/4？</legend>
          <label>
            <input
              type="radio"
              name="memory-prediction"
              checked={prediction === "rank"}
              onChange={() => {
                setPrediction("rank");
                setRan(false);
              }}
            />
            LoRA rank 减半
          </label>
          <label>
            <input
              type="radio"
              name="memory-prediction"
              checked={prediction === "quantize"}
              onChange={() => {
                setPrediction("quantize");
                setRan(false);
              }}
            />
            Base 从 16-bit 改为 4-bit
          </label>
          <label>
            <input
              type="radio"
              name="memory-prediction"
              checked={prediction === "batch"}
              onChange={() => {
                setPrediction("batch");
                setRan(false);
              }}
            />
            Micro batch 减半
          </label>
        </fieldset>
        <div>
          <button type="button" onClick={reset}>重置</button>
          <button
            type="button"
            className={styles.run}
            disabled={!prediction}
            onClick={() => setRan(true)}
          >
            执行 preflight
          </button>
        </div>
      </div>
      <Gate passed={passed}>
        选择 4-bit base、预测正确，并让教学估算小于单卡显存上限。
      </Gate>
    </LabFrame>
  );
}
