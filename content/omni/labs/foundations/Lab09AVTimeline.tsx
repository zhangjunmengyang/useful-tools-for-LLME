"use client";

import { useMemo, useState } from "react";
import styles from "./Lab09AVTimeline.module.css";
import type { FoundationLabProps } from "./types";
import { initialNumber } from "./types";

type Packet = {
  kind: "video" | "audio";
  index: number;
  mediaPts: number;
  rawPts: number;
  delay: number;
  arrival: number;
  correctedPts: number;
  usable: boolean;
};

const videoDelays = [70, 10, 55, 25, 90];
const audioDelays = [35, 15, 50, 10, 65, 25, 45, 20, 75, 30];

export function Lab09AVTimeline({
  onComplete,
  initialState,
}: FoundationLabProps) {
  const defaults = {
    sourceOffset: initialNumber(initialState, "sourceOffset", 40),
    correction: initialNumber(initialState, "correction", 40),
    bufferMs: initialNumber(initialState, "bufferMs", 60),
  };
  const [sourceOffset, setSourceOffset] = useState(defaults.sourceOffset);
  const [correction, setCorrection] = useState(defaults.correction);
  const [bufferMs, setBufferMs] = useState(defaults.bufferMs);
  const [audioPrediction, setAudioPrediction] = useState("");
  const [survivorPrediction, setSurvivorPrediction] = useState("");
  const [hasRun, setHasRun] = useState(false);

  const result = useMemo(() => {
    const video: Packet[] = videoDelays.map((delay, index) => {
      const mediaPts = index * 40;
      return {
        kind: "video",
        index,
        mediaPts,
        rawPts: mediaPts,
        delay,
        arrival: mediaPts + delay,
        correctedPts: mediaPts,
        usable: delay <= bufferMs,
      };
    });
    const audio: Packet[] = audioDelays.map((delay, index) => {
      const mediaPts = index * 20;
      const rawPts = mediaPts + sourceOffset;
      return {
        kind: "audio",
        index,
        mediaPts,
        rawPts,
        delay,
        arrival: mediaPts + delay,
        correctedPts: rawPts - correction,
        usable: delay <= bufferMs,
      };
    });
    const usableVideo = video.filter((packet) => packet.usable);
    const usableAudio = audio.filter((packet) => packet.usable);
    const targetVideo = video[2];
    const alignedAudio = [...usableAudio].sort(
      (a, b) =>
        Math.abs(a.correctedPts - targetVideo.correctedPts) -
          Math.abs(b.correctedPts - targetVideo.correctedPts) ||
        a.index - b.index,
    )[0];
    const arrivalOrder = [...video, ...audio].sort(
      (a, b) =>
        a.arrival - b.arrival ||
        (a.kind === "video" ? 0 : 1) - (b.kind === "video" ? 0 : 1),
    );
    const presentationOrder = [...usableVideo, ...usableAudio].sort(
      (a, b) =>
        a.correctedPts - b.correctedPts ||
        (a.kind === "video" ? 0 : 1) - (b.kind === "video" ? 0 : 1),
    );
    return {
      video,
      audio,
      usableVideo,
      usableAudio,
      targetVideo,
      alignedAudio,
      arrivalOrder,
      presentationOrder,
    };
  }, [bufferMs, correction, sourceOffset]);

  const predictionComplete =
    audioPrediction.trim() !== "" && survivorPrediction.trim() !== "";
  const gatePassed =
    hasRun &&
    Number(audioPrediction) === result.alignedAudio?.index &&
    Number(survivorPrediction) === result.usableVideo.length;

  function invalidate() {
    setHasRun(false);
  }

  function runReorder() {
    setHasRun(true);
    const passed =
      Number(audioPrediction) === result.alignedAudio?.index &&
      Number(survivorPrediction) === result.usableVideo.length;
    if (passed) {
      onComplete?.({
        sourceOffset,
        correction,
        bufferMs,
        videoSurvivors: result.usableVideo.length,
        alignedAudioIndex: result.alignedAudio?.index,
        alignedDeltaMs: result.alignedAudio
          ? Math.abs(
              result.alignedAudio.correctedPts -
                result.targetVideo.correctedPts,
            )
          : null,
      });
    }
  }

  function reset() {
    setSourceOffset(defaults.sourceOffset);
    setCorrection(defaults.correction);
    setBufferMs(defaults.bufferMs);
    setAudioPrediction("");
    setSurvivorPrediction("");
    setHasRun(false);
  }

  function packetLabel(packet: Packet) {
    return `${packet.kind === "video" ? "V" : "A"}${packet.index}`;
  }

  return (
    <section className={styles.lab} aria-labelledby="lab09-title">
      <header className={styles.header}>
        <div>
          <div className={styles.tags}>
            <span>公式计算</span>
            <span>时间戳实验</span>
          </div>
          <h3 id="lab09-title">包可以乱序到达，但嘴型不能乱序播放</h3>
          <p>
            把 arrival time、媒体 PTS 与音频时钟 offset 分开处理，再用有限
            reorder buffer 决定哪些包还有资格进入呈现时间轴。
          </p>
        </div>
        <button type="button" className={styles.reset} onClick={reset}>
          重置时间轴
        </button>
      </header>

      <div className={styles.controls}>
        <label>
          <span>音频源时钟 offset</span>
          <select
            value={sourceOffset}
            onChange={(event) => {
              setSourceOffset(Number(event.target.value));
              invalidate();
            }}
          >
            {[20, 40, 60].map((value) => (
              <option key={value} value={value}>
                +{value} ms
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>
            应用音频校正 <strong>{correction >= 0 ? "+" : ""}{correction} ms</strong>
          </span>
          <input
            type="range"
            min="-20"
            max="80"
            step="20"
            value={correction}
            onChange={(event) => {
              setCorrection(Number(event.target.value));
              invalidate();
            }}
          />
        </label>
        <label>
          <span>Reorder buffer</span>
          <select
            value={bufferMs}
            onChange={(event) => {
              setBufferMs(Number(event.target.value));
              invalidate();
            }}
          >
            {[20, 40, 60, 80, 100].map((value) => (
              <option key={value} value={value}>
                {value} ms
              </option>
            ))}
          </select>
        </label>
        <div className={styles.formulas}>
          <code>arrival = media_pts + network_delay</code>
          <code>audio_corrected_pts = raw_pts − correction</code>
          <code>usable ⇔ network_delay ≤ buffer</code>
        </div>
      </div>

      <div className={styles.packetBook}>
        <div className={styles.packetTable}>
          <div className={styles.tableHead}>
            <span>视频包</span>
            <span>media PTS</span>
            <span>delay</span>
            <span>buffer 判定</span>
          </div>
          {result.video.map((packet) => (
            <div className={styles.tableRow} key={packet.index}>
              <b>V{packet.index}</b>
              <span>{packet.mediaPts} ms</span>
              <span>{packet.delay} ms</span>
              <em>{hasRun ? (packet.usable ? "保留" : "过期") : "?"}</em>
            </div>
          ))}
        </div>
        <div className={styles.packetTable}>
          <div className={styles.tableHead}>
            <span>音频包</span>
            <span>raw PTS</span>
            <span>delay</span>
            <span>校正后 PTS</span>
          </div>
          {result.audio.map((packet) => (
            <div className={styles.tableRow} key={packet.index}>
              <b>A{packet.index}</b>
              <span>{packet.rawPts} ms</span>
              <span>{packet.delay} ms</span>
              <em>{hasRun ? `${packet.correctedPts} ms` : "?"}</em>
            </div>
          ))}
        </div>
      </div>

      <div className={styles.challenge}>
        <div>
          <span>先预测，再开 buffer</span>
          <strong>目标：视频 V2，PTS = 80 ms</strong>
        </div>
        <label>
          <span>最近且可用的音频</span>
          <span className={styles.inlineInput}>
            A
            <input
              type="number"
              min="0"
              max="9"
              value={audioPrediction}
              onChange={(event) => {
                setAudioPrediction(event.target.value);
                invalidate();
              }}
              aria-label="预测对齐的音频包索引"
            />
          </span>
        </label>
        <label>
          <span>可用视频帧数</span>
          <input
            type="number"
            min="0"
            max="5"
            value={survivorPrediction}
            onChange={(event) => {
              setSurvivorPrediction(event.target.value);
              invalidate();
            }}
            aria-label="预测未过期的视频帧数"
          />
        </label>
        <button
          type="button"
          disabled={!predictionComplete}
          onClick={runReorder}
        >
          运行重排与对齐
        </button>
      </div>

      <div className={styles.timelineBoard}>
        <div className={styles.orderLane}>
          <div className={styles.laneLabel}>
            <span>网络到达顺序</span>
            <small>按 arrival 排列，未做校正</small>
          </div>
          <div className={styles.packetRail}>
            {result.arrivalOrder.map((packet) => (
              <div
                key={`${packet.kind}-${packet.index}`}
                className={
                  packet.kind === "video" ? styles.video : styles.audio
                }
              >
                <b>{packetLabel(packet)}</b>
                <span>{packet.arrival}ms</span>
              </div>
            ))}
          </div>
        </div>
        <div className={styles.reorderArrow}>
          <span>REORDER BUFFER</span>
          <b>{bufferMs} ms</b>
        </div>
        <div className={styles.orderLane}>
          <div className={styles.laneLabel}>
            <span>呈现顺序</span>
            <small>丢弃过期包，按 corrected PTS 排列</small>
          </div>
          <div className={styles.packetRail} aria-live="polite">
            {hasRun ? (
              result.presentationOrder.map((packet) => (
                <div
                  key={`${packet.kind}-${packet.index}`}
                  className={[
                    packet.kind === "video" ? styles.video : styles.audio,
                    packet.kind === "video" && packet.index === 2
                      ? styles.target
                      : "",
                    packet.kind === "audio" &&
                    packet.index === result.alignedAudio?.index
                      ? styles.aligned
                      : "",
                  ].join(" ")}
                >
                  <b>{packetLabel(packet)}</b>
                  <span>{packet.correctedPts}ms</span>
                </div>
              ))
            ) : (
              <div className={styles.locked}>运行后解锁呈现时间轴</div>
            )}
          </div>
        </div>
      </div>

      {hasRun && (
        <div className={styles.alignment} aria-live="polite">
          <div>
            <span>目标视频</span>
            <strong>V2 · 80 ms</strong>
          </div>
          <b>↔</b>
          <div>
            <span>最近可用音频</span>
            <strong>
              A{result.alignedAudio?.index} ·{" "}
              {result.alignedAudio?.correctedPts} ms
            </strong>
          </div>
          <code>
            |80 − {result.alignedAudio?.correctedPts}| ={" "}
            {result.alignedAudio
              ? Math.abs(80 - result.alignedAudio.correctedPts)
              : "—"}{" "}
            ms
          </code>
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
            ? "先按 delay ≤ buffer 筛包，再按 corrected PTS 找最近邻。"
            : gatePassed
              ? "你正确分离了网络乱序、buffer 生存判定与 A/V 时钟校正。"
              : `结果：${result.usableVideo.length} 个视频帧可用，V2 最近的可用音频是 A${result.alignedAudio?.index}。`}
        </span>
      </div>
    </section>
  );
}
