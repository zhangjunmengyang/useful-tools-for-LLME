"use client";

import { useMemo, useState } from "react";
import styles from "./Lab03CodecWorkbench.module.css";
import type { FoundationLabProps } from "./types";
import { initialNumber } from "./types";

export function Lab03CodecWorkbench({
  onComplete,
  initialState,
}: FoundationLabProps) {
  const defaults = {
    frameRate: initialNumber(initialState, "frameRate", 50),
    layers: initialNumber(initialState, "layers", 4),
    codebookSize: initialNumber(initialState, "codebookSize", 1024),
  };
  const [frameRate, setFrameRate] = useState(defaults.frameRate);
  const [layers, setLayers] = useState(defaults.layers);
  const [codebookSize, setCodebookSize] = useState(defaults.codebookSize);
  const [prediction, setPrediction] = useState("");
  const [hasRun, setHasRun] = useState(false);

  const result = useMemo(() => {
    const bitsPerIndex = Math.log2(codebookSize);
    const bitsPerSecond = frameRate * layers * bitsPerIndex;
    const kbps = bitsPerSecond / 1000;
    const kilobytesPerMinute = (bitsPerSecond * 60) / 8 / 1000;
    const indices = Array.from({ length: layers }, (_, layer) =>
      Array.from(
        { length: 8 },
        (_, frame) => (frame * 31 + layer * 17 + 7) % codebookSize,
      ),
    );
    return {
      bitsPerIndex,
      bitsPerSecond,
      kbps,
      kilobytesPerMinute,
      indices,
    };
  }, [codebookSize, frameRate, layers]);

  const numericPrediction = Number(prediction);
  const gatePassed =
    hasRun &&
    Number.isFinite(numericPrediction) &&
    Math.abs(numericPrediction - result.kbps) < 0.0005;

  function invalidate() {
    setHasRun(false);
  }

  function runCodec() {
    setHasRun(true);
    if (
      Number.isFinite(numericPrediction) &&
      Math.abs(numericPrediction - result.kbps) < 0.0005
    ) {
      onComplete?.({
        frameRate,
        layers,
        codebookSize,
        bitsPerIndex: result.bitsPerIndex,
        bitrateKbps: result.kbps,
        kilobytesPerMinute: result.kilobytesPerMinute,
      });
    }
  }

  function reset() {
    setFrameRate(defaults.frameRate);
    setLayers(defaults.layers);
    setCodebookSize(defaults.codebookSize);
    setPrediction("");
    setHasRun(false);
  }

  return (
    <section className={styles.lab} aria-labelledby="lab03-title">
      <header className={styles.header}>
        <div>
          <div className={styles.tags} aria-label="实验类型">
            <span>公式计算</span>
            <span>RVQ 拆解</span>
          </div>
          <h3 id="lab03-title">一秒音频到底花掉多少离散 Token？</h3>
          <p>
            逐层观察残差向量量化，并计算 codec 的裸索引码率。码本层数只用于码率计算，不直接表示音质。
          </p>
        </div>
        <button className={styles.reset} type="button" onClick={reset}>
          恢复初始配置
        </button>
      </header>

      <div className={styles.equation} aria-label="码率公式">
        <span>码率</span>
        <strong>{frameRate}</strong>
        <i>frames/s</i>
        <b>×</b>
        <strong>{layers}</strong>
        <i>layers</i>
        <b>×</b>
        <strong>
          log<sub>2</sub>({codebookSize})
        </strong>
        <i>bits/index</i>
        <b>=</b>
        <output>{hasRun ? `${result.kbps.toFixed(3)} kbps` : "?"}</output>
      </div>

      <div className={styles.main}>
        <div className={styles.controls}>
          <label>
            <span>Codec 帧率</span>
            <select
              value={frameRate}
              onChange={(event) => {
                setFrameRate(Number(event.target.value));
                invalidate();
              }}
            >
              <option value="25">25 frames/s</option>
              <option value="50">50 frames/s</option>
              <option value="75">75 frames/s</option>
            </select>
          </label>
          <label>
            <span>
              RVQ 层数 <strong>{layers}</strong>
            </span>
            <input
              type="range"
              min="2"
              max="8"
              step="1"
              value={layers}
              onChange={(event) => {
                setLayers(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <fieldset>
            <legend>每层 codebook 大小</legend>
            <div className={styles.segmented}>
              {[256, 512, 1024].map((size) => (
                <button
                  key={size}
                  type="button"
                  aria-pressed={codebookSize === size}
                  onClick={() => {
                    setCodebookSize(size);
                    invalidate();
                  }}
                >
                  {size}
                </button>
              ))}
            </div>
          </fieldset>
          <div className={styles.note}>
            <b>严格边界</b>
            <span>
              结果只计算离散索引位数，不包含容器、熵编码、模型权重或网络协议开销。
            </span>
          </div>
        </div>

        <div className={styles.rvq}>
          <div className={styles.signal}>
            <span>encoder latent</span>
            <div />
          </div>
          <div className={styles.layers} aria-label={`${layers} 层 RVQ`}>
            {Array.from({ length: layers }, (_, layer) => (
              <div className={styles.layer} key={layer}>
                <span>残差 {layer}</span>
                <strong>Q{layer + 1}</strong>
                <small>
                  {hasRun
                    ? `${result.bitsPerIndex} bit`
                    : `log₂(${codebookSize})`}
                </small>
              </div>
            ))}
          </div>
          <div className={styles.reconstruct}>
            <span>Σ quantized vectors</span>
            <strong>重构 latent</strong>
          </div>
        </div>
      </div>

      <div className={styles.predict}>
        <label htmlFor="lab03-prediction">
          先预测：当前配置的裸索引码率是多少？
          <span>
            <input
              id="lab03-prediction"
              type="number"
              min="0"
              step="0.001"
              inputMode="decimal"
              value={prediction}
              onChange={(event) => {
                setPrediction(event.target.value);
                invalidate();
              }}
              placeholder="例如 2.000"
            />
            <b>kbps</b>
          </span>
        </label>
        <button
          type="button"
          onClick={runCodec}
          disabled={
            prediction.trim() === "" || !Number.isFinite(numericPrediction)
          }
        >
          编码 8 帧并核算
        </button>
      </div>

      {hasRun && (
        <div className={styles.readout} aria-live="polite">
          <div className={styles.stats}>
            <div>
              <span>每层每帧</span>
              <strong>{result.bitsPerIndex} bits</strong>
            </div>
            <div>
              <span>每秒索引</span>
              <strong>{frameRate * layers}</strong>
            </div>
            <div>
              <span>十进制存储 / 分钟</span>
              <strong>{result.kilobytesPerMinute.toFixed(2)} kB</strong>
            </div>
          </div>
          <div
            className={styles.indexTable}
            role="table"
            aria-label="确定性生成的八帧 RVQ 索引教学样例"
          >
            <div className={styles.tableHead} role="row">
              <span role="columnheader">层</span>
              {Array.from({ length: 8 }, (_, frame) => (
                <span role="columnheader" key={frame}>
                  F{frame}
                </span>
              ))}
            </div>
            {result.indices.map((row, layer) => (
              <div className={styles.tableRow} role="row" key={layer}>
                <b role="rowheader">Q{layer + 1}</b>
                {row.map((index, frame) => (
                  <span role="cell" key={frame}>
                    {index}
                  </span>
                ))}
              </div>
            ))}
          </div>
          <p>
            索引由教学函数{" "}
            <code>(31 × frame + 17 × layer + 7) mod K</code>{" "}
            生成，只用于看懂多层数据形状，不代表真实 codec 输出。
          </p>
        </div>
      )}

      <div
        className={`${styles.gate} ${
          hasRun ? (gatePassed ? styles.pass : styles.retry) : ""
        }`}
        role="status"
      >
        <strong>{gatePassed ? "验收已通过" : "完成验收"}</strong>
        <span>
          {!hasRun
            ? "用公式算完再运行；答案精确到 0.001 kbps。"
            : gatePassed
              ? "你已经能从帧率、RVQ 层数和 codebook 位宽推导码率。"
              : `再算一次：${frameRate} × ${layers} × ${result.bitsPerIndex} = ${result.bitsPerSecond} bit/s。`}
        </span>
      </div>
    </section>
  );
}
