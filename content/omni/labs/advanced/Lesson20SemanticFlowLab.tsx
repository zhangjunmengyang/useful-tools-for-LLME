"use client";

import { useEffect, useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import {
  mean,
  numberFrom,
  round,
  stringFrom,
  useCompletionGate,
} from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson20SemanticFlowLab.module.css";

type PathMode = "understand" | "generate";

function clamp(value: number) {
  return Math.max(0, Math.min(1, value));
}

function makeSource() {
  return Array.from({ length: 64 }, (_, index) => {
    const x = index % 8;
    const y = Math.floor(index / 8);
    const distance = Math.sqrt((x - 3.5) ** 2 + (y - 3.5) ** 2);
    const shape = Math.max(0, 1 - distance / 5);
    const texture = (x + y) % 2 === 0 ? 0.16 : -0.08;
    return clamp(0.12 + shape * 0.78 + texture);
  });
}

function makeNoise(scale: number) {
  return Array.from({ length: 64 }, (_, index) => {
    const raw = Math.sin((index + 1) * 12.9898) * 43758.5453;
    const unit = raw - Math.floor(raw);
    return clamp(0.5 + (unit - 0.5) * 2 * scale);
  });
}

function pool(source: number[], blockSize: number) {
  const side = 8 / blockSize;
  return Array.from({ length: side * side }, (_, index) => {
    const cellX = index % side;
    const cellY = Math.floor(index / side);
    const values: number[] = [];
    for (let y = 0; y < blockSize; y += 1) {
      for (let x = 0; x < blockSize; x += 1) {
        values.push(
          source[(cellY * blockSize + y) * 8 + cellX * blockSize + x],
        );
      }
    }
    return mean(values);
  });
}

function upsample(semantic: number[], blockSize: number) {
  const side = 8 / blockSize;
  return Array.from({ length: 64 }, (_, index) => {
    const x = index % 8;
    const y = Math.floor(index / 8);
    return semantic[Math.floor(y / blockSize) * side + Math.floor(x / blockSize)];
  });
}

export function Lesson20SemanticFlowLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    blockSize: numberFrom(initialState, "semanticBlock", 2, 2, 4),
    detailMix: numberFrom(initialState, "detailMix", 0.65, 0, 1),
    noiseScale: numberFrom(initialState, "vaeNoise", 0.9, 0.1, 1),
    steps: numberFrom(initialState, "flowSteps", 8, 2, 16),
    mode: stringFrom(initialState, "mode", "understand") as PathMode,
    prediction: stringFrom(initialState, "prediction", ""),
  };
  const [blockSize, setBlockSize] = useState(
    defaults.blockSize === 4 ? 4 : 2,
  );
  const [detailMix, setDetailMix] = useState(defaults.detailMix);
  const [noiseScale, setNoiseScale] = useState(defaults.noiseScale);
  const [steps, setSteps] = useState(defaults.steps);
  const [mode, setMode] = useState<PathMode>(
    defaults.mode === "generate" ? "generate" : "understand",
  );
  const [prediction, setPrediction] = useState(defaults.prediction);
  const [currentStep, setCurrentStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [ran, setRan] = useState(false);

  const calculation = useMemo(() => {
    const source = makeSource();
    const semantic = pool(source, blockSize);
    const semanticBase = upsample(semantic, blockSize);
    const detailResidual = source.map(
      (value, index) => value - semanticBase[index],
    );
    const target = semanticBase.map((value, index) =>
      clamp(value + detailMix * detailResidual[index]),
    );
    const noise = makeNoise(noiseScale);
    const t = currentStep / steps;
    const flow = noise.map(
      (value, index) => (1 - t) * value + t * target[index],
    );
    const initialL1 = mean(
      noise.map((value, index) => Math.abs(value - target[index])),
    );
    const currentL1 = mean(
      flow.map((value, index) => Math.abs(value - target[index])),
    );
    const sourceReconstructionL1 = mean(
      target.map((value, index) => Math.abs(value - source[index])),
    );
    return {
      source,
      semantic,
      semanticBase,
      detailResidual,
      target,
      noise,
      flow,
      t,
      initialL1,
      currentL1,
      sourceReconstructionL1,
    };
  }, [blockSize, currentStep, detailMix, noiseScale, steps]);

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      setCurrentStep((current) => {
        const next = Math.min(steps, current + 1);
        if (next >= steps) {
          window.clearInterval(timer);
          setPlaying(false);
        }
        return next;
      });
    }, 240);
    return () => window.clearInterval(timer);
  }, [playing, steps]);

  const passed =
    ran &&
    !playing &&
    currentStep === steps &&
    mode === "generate" &&
    prediction === "linear" &&
    calculation.currentL1 < 1e-9;
  const completion = useMemo(
    () => ({
      lessonId: 20,
      semanticBlock: blockSize,
      semanticGrid: `${8 / blockSize}x${8 / blockSize}`,
      detailMix,
      vaeNoise: noiseScale,
      flowSteps: steps,
      finalTargetL1: round(calculation.currentL1, 6),
      reconstructionL1: round(calculation.sourceReconstructionL1, 4),
    }),
    [
      blockSize,
      calculation.currentL1,
      calculation.sourceReconstructionL1,
      detailMix,
      noiseScale,
      steps,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function invalidate() {
    setPlaying(false);
    setCurrentStep(0);
    setRan(false);
  }

  function reset() {
    setBlockSize(defaults.blockSize === 4 ? 4 : 2);
    setDetailMix(defaults.detailMix);
    setNoiseScale(defaults.noiseScale);
    setSteps(defaults.steps);
    setMode("understand");
    setPrediction("");
    setCurrentStep(0);
    setPlaying(false);
    setRan(false);
  }

  function run() {
    setMode("generate");
    setRan(true);
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setCurrentStep(steps);
      setPlaying(false);
      return;
    }
    setCurrentStep(0);
    setPlaying(true);
  }

  return (
    <LabFrame
      lesson="20"
      title="走通 Understand / Generate 的双路径"
      description="理解路径把连续信号压成语义格；生成路径把语义结构与 VAE 细节合成目标 latent，再用一个可手算的 straight-flow 从噪声搬运过去。"
    >
      <section className={styles.studio}>
        <div className={styles.topbar}>
          <div className={styles.pathTabs} aria-label="路径模式">
            <button
              type="button"
              aria-pressed={mode === "understand"}
              onClick={() => {
                setMode("understand");
                invalidate();
              }}
            >
              Understand path
            </button>
            <button
              type="button"
              aria-pressed={mode === "generate"}
              onClick={() => {
                setMode("generate");
                invalidate();
              }}
            >
              Generate path
            </button>
          </div>
          <p>
            {mode === "understand"
              ? "x → semantic pool → discrete structure"
              : "semantic + VAE detail → target z₁ ← flow(z₀)"}
          </p>
        </div>

        <div className={styles.pipeline}>
          <GridPanel
            title="Input x"
            subtitle="8×8 连续信号"
            values={calculation.source}
            side={8}
            active={mode === "understand"}
          />
          <div className={styles.operator} aria-label="平均池化">
            <span>POOL</span>
            <code>sᵢⱼ = mean(block)</code>
          </div>
          <GridPanel
            title="Semantic s"
            subtitle={`${8 / blockSize}×${8 / blockSize} 结构格`}
            values={calculation.semantic}
            side={8 / blockSize}
            active
          />
          <div className={styles.operator} aria-label="语义和 VAE 细节合成">
            <span>FUSE</span>
            <code>y = up(s) + α·residual</code>
          </div>
          <GridPanel
            title="VAE / flow zₜ"
            subtitle={`t = ${currentStep}/${steps}`}
            values={mode === "generate" ? calculation.flow : calculation.semanticBase}
            side={8}
            active={mode === "generate"}
          />
        </div>

        <div className={styles.controls}>
          <label>
            <span>Semantic block</span>
            <select
              value={blockSize}
              onChange={(event) => {
                setBlockSize(Number(event.target.value));
                invalidate();
              }}
            >
              <option value="2">2×2 pool → 4×4 tokens</option>
              <option value="4">4×4 pool → 2×2 tokens</option>
            </select>
          </label>
          <label>
            <span>VAE detail mix α <output>{detailMix.toFixed(2)}</output></span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={detailMix}
              onChange={(event) => {
                setDetailMix(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <label>
            <span>Initial noise scale <output>{noiseScale.toFixed(2)}</output></span>
            <input
              type="range"
              min="0.1"
              max="1"
              step="0.05"
              value={noiseScale}
              onChange={(event) => {
                setNoiseScale(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <label>
            <span>Flow steps <output>{steps}</output></span>
            <input
              type="range"
              min="2"
              max="16"
              step="1"
              value={steps}
              onChange={(event) => {
                setSteps(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
        </div>

        <section className={styles.flowMath}>
          <header>
            <div>
              <h3>Straight-flow toy</h3>
              <p>zₜ = (1−t)z₀ + t·y，t = step / N</p>
            </div>
            <strong>L1(zₜ, y) = {calculation.currentL1.toFixed(4)}</strong>
          </header>
          <div className={styles.stepRail}>
            {Array.from({ length: steps + 1 }, (_, step) => (
              <button
                type="button"
                key={step}
                aria-label={`跳到 flow 第 ${step} 步`}
                aria-current={currentStep === step ? "step" : undefined}
                onClick={() => {
                  setMode("generate");
                  setCurrentStep(step);
                  setPlaying(false);
                  setRan(true);
                }}
              >
                <i />
                <span>{step}</span>
              </button>
            ))}
          </div>
          <div className={styles.errorIdentity}>
            <span>初始距离 {calculation.initialL1.toFixed(4)}</span>
            <span>× (1 − {calculation.t.toFixed(3)})</span>
            <strong>= {calculation.currentL1.toFixed(4)}</strong>
          </div>
          <p className={styles.truthNote}>
            在这个解析示例中，Steps 只改变离散观察点的数量。它不表示生成质量；按公式计算时，t=1 会精确到达 target。
          </p>
        </section>
      </section>

      <div className={styles.challenge}>
        <fieldset>
          <legend>先预测：在 straight-flow 公式中，L1(zₜ, y) 随 t 如何变化？</legend>
          <label>
            <input
              type="radio"
              name="flow-prediction"
              checked={prediction === "linear"}
              onChange={() => {
                setPrediction("linear");
                invalidate();
              }}
            />
            按 (1−t) 线性下降
          </label>
          <label>
            <input
              type="radio"
              name="flow-prediction"
              checked={prediction === "random"}
              onChange={() => {
                setPrediction("random");
                invalidate();
              }}
            />
            随机波动
          </label>
          <label>
            <input
              type="radio"
              name="flow-prediction"
              checked={prediction === "constant"}
              onChange={() => {
                setPrediction("constant");
                invalidate();
              }}
            />
            保持常数
          </label>
        </fieldset>
        <div>
          <button type="button" onClick={reset}>重置</button>
          <button
            type="button"
            className={styles.run}
            disabled={!prediction || playing}
            onClick={run}
          >
            {playing ? "Flow 运行中…" : "运行生成路径"}
          </button>
        </div>
      </div>
      <Gate passed={passed}>
        正确预测 straight-flow 距离，运行 Generate path 到 t=1，并验证 target L1 精确为 0。
      </Gate>
    </LabFrame>
  );
}

function GridPanel({
  title,
  subtitle,
  values,
  side,
  active,
}: {
  title: string;
  subtitle: string;
  values: number[];
  side: number;
  active: boolean;
}) {
  return (
    <article className={`${styles.gridPanel} ${active ? styles.active : ""}`}>
      <header>
        <b>{title}</b>
        <span>{subtitle}</span>
      </header>
      <div
        className={styles.signalGrid}
        style={{ gridTemplateColumns: `repeat(${side}, 1fr)` }}
        aria-label={`${title} 数值格`}
      >
        {values.map((value, index) => (
          <i
            key={index}
            title={value.toFixed(3)}
            style={{
              background: `hsl(198 48% ${88 - value * 58}%)`,
            }}
          />
        ))}
      </div>
    </article>
  );
}
