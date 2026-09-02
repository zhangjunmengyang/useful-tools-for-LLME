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
import styles from "./Lesson32SomGroundingLab.module.css";

type HeadMode = "som" | "continuous";
type Domain = "ui" | "table";

type MarkedItem = {
  id: number;
  name: string;
  x: number;
  y: number;
  width: number;
  height: number;
};

const UI_ITEMS: MarkedItem[] = [
  { id: 1, name: "搜索", x: 0.18, y: 0.16, width: 0.2, height: 0.1 },
  { id: 2, name: "提交", x: 0.82, y: 0.16, width: 0.18, height: 0.1 },
  { id: 3, name: "菜单", x: 0.12, y: 0.5, width: 0.16, height: 0.1 },
  { id: 4, name: "购物车", x: 0.88, y: 0.5, width: 0.16, height: 0.1 },
  { id: 5, name: "取消", x: 0.22, y: 0.84, width: 0.18, height: 0.1 },
  { id: 6, name: "设置", x: 0.78, y: 0.84, width: 0.18, height: 0.1 },
];

const TABLE_ITEMS: MarkedItem[] = [
  { id: 1, name: "杯子", x: 0.26, y: 0.32, width: 0.12, height: 0.14 },
  { id: 2, name: "盘子", x: 0.62, y: 0.28, width: 0.16, height: 0.12 },
  { id: 3, name: "瓶子", x: 0.8, y: 0.68, width: 0.1, height: 0.18 },
  { id: 4, name: "海绵", x: 0.22, y: 0.74, width: 0.14, height: 0.12 },
  { id: 5, name: "抹布", x: 0.5, y: 0.54, width: 0.16, height: 0.1 },
  { id: 6, name: "碗", x: 0.4, y: 0.2, width: 0.12, height: 0.12 },
];

const PROBE_RES = 4;

function clamp01(value: number) {
  return Math.max(0, Math.min(1, value));
}

function cellCenter(unit: number, resolution: number) {
  const bin = Math.min(resolution - 1, Math.floor(clamp01(unit) * resolution));
  return (bin + 0.5) / resolution;
}

function cellIndex(unit: number, resolution: number) {
  return Math.min(resolution - 1, Math.floor(clamp01(unit) * resolution));
}

function distance(ax: number, ay: number, bx: number, by: number) {
  return Math.hypot(ax - bx, ay - by);
}

function itemsFor(domain: Domain) {
  return domain === "ui" ? UI_ITEMS : TABLE_ITEMS;
}

function collidingMarks(items: MarkedItem[], resolution: number) {
  const buckets = new Map<string, number[]>();
  items.forEach((item) => {
    const key = `${cellIndex(item.x, resolution)}:${cellIndex(item.y, resolution)}`;
    const current = buckets.get(key) ?? [];
    current.push(item.id);
    buckets.set(key, current);
  });
  return [...buckets.values()].filter((ids) => ids.length > 1);
}

function predictItem(
  item: MarkedItem,
  items: MarkedItem[],
  resolution: number,
  steps: number,
  mode: HeadMode,
) {
  const progress = 1 - 0.68 ** steps;
  const observedX = cellCenter(item.x, resolution);
  const observedY = cellCenter(item.y, resolution);
  const collisions = collidingMarks(items, resolution);
  const confused = collisions.some((ids) => ids.includes(item.id));

  if (mode === "som") {
    const somReady = progress > 0.18 && !confused;
    if (somReady) {
      return { x: item.x, y: item.y, hit: true, confused: false };
    }
    if (confused) {
      const group = collisions.find((ids) => ids.includes(item.id)) ?? [item.id];
      const other = items.find((candidate) => candidate.id === group[0]) ?? item;
      const mix = 0.5 * (1 - progress);
      return {
        x: item.x * (1 - mix) + other.x * mix,
        y: item.y * (1 - mix) + other.y * mix,
        hit: group.length === 1,
        confused: true,
      };
    }
    return {
      x: observedX,
      y: observedY,
      hit: false,
      confused: false,
    };
  }

  return {
    x: observedX + (item.x - observedX) * progress,
    y: observedY + (item.y - observedY) * progress,
    hit: false,
    confused: false,
  };
}

function domainMetrics(
  domain: Domain,
  resolution: number,
  steps: number,
  mode: HeadMode,
) {
  const items = itemsFor(domain);
  const predictions = items.map((item) =>
    predictItem(item, items, resolution, steps, mode),
  );
  const errors = items.map((item, index) =>
    distance(predictions[index].x, predictions[index].y, item.x, item.y),
  );
  return {
    items,
    predictions,
    meanError: mean(errors),
    collisions: collidingMarks(items, resolution).length,
  };
}

export function Lesson32SomGroundingLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    resolution: numberFrom(initialState, "resolution", 8, 4, 32),
    steps: numberFrom(initialState, "steps", 0, 0, 12),
    mode: stringFrom(initialState, "mode", "continuous") as HeadMode,
    domain: stringFrom(initialState, "domain", "ui") as Domain,
    prediction: stringFrom(initialState, "prediction", ""),
  };
  const [resolution, setResolution] = useState(
    [4, 8, 16, 32].includes(defaults.resolution) ? defaults.resolution : 8,
  );
  const [steps, setSteps] = useState(defaults.steps);
  const [mode, setMode] = useState<HeadMode>(
    defaults.mode === "som" ? "som" : "continuous",
  );
  const [domain, setDomain] = useState<Domain>(
    defaults.domain === "table" ? "table" : "ui",
  );
  const [prediction, setPrediction] = useState(defaults.prediction);
  const [revealed, setRevealed] = useState(false);

  const calculation = useMemo(() => {
    const uiContinuous = domainMetrics("ui", resolution, steps, "continuous");
    const uiSom = domainMetrics("ui", resolution, steps, "som");
    const tableContinuous = domainMetrics("table", resolution, steps, "continuous");
    const tableSom = domainMetrics("table", resolution, steps, "som");
    const probeContinuous = mean([
      domainMetrics("ui", PROBE_RES, steps, "continuous").meanError,
      domainMetrics("table", PROBE_RES, steps, "continuous").meanError,
    ]);
    const probeSom = mean([
      domainMetrics("ui", PROBE_RES, steps, "som").meanError,
      domainMetrics("table", PROBE_RES, steps, "som").meanError,
    ]);
    const uiInit = domainMetrics("ui", resolution, 0, mode).meanError;
    const tableInit = domainMetrics("table", resolution, 0, mode).meanError;
    const active =
      domain === "ui"
        ? mode === "som"
          ? uiSom
          : uiContinuous
        : mode === "som"
          ? tableSom
          : tableContinuous;
    return {
      uiContinuous,
      uiSom,
      tableContinuous,
      tableSom,
      probeContinuous,
      probeSom,
      uiInit,
      tableInit,
      uiNow: mode === "som" ? uiSom.meanError : uiContinuous.meanError,
      tableNow: mode === "som" ? tableSom.meanError : tableContinuous.meanError,
      active,
    };
  }, [domain, mode, resolution, steps]);

  const passed =
    revealed &&
    prediction === "lowres_continuous_worse" &&
    steps >= 4 &&
    calculation.probeContinuous > calculation.probeSom + 0.01 &&
    calculation.uiNow < calculation.uiInit - 0.004 &&
    calculation.tableNow < calculation.tableInit - 0.004;

  const completion = useMemo(
    () => ({
      lessonId: 32,
      resolution,
      steps,
      mode,
      domain,
      prediction,
      probeContinuous: round(calculation.probeContinuous, 4),
      probeSom: round(calculation.probeSom, 4),
      uiError: round(calculation.uiNow, 4),
      tableError: round(calculation.tableNow, 4),
    }),
    [
      calculation.probeContinuous,
      calculation.probeSom,
      calculation.tableNow,
      calculation.uiNow,
      domain,
      mode,
      prediction,
      resolution,
      steps,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function reset() {
    setResolution(8);
    setSteps(0);
    setMode("continuous");
    setDomain("ui");
    setPrediction("");
    setRevealed(false);
  }

  const activeItems = itemsFor(domain);
  const grid = Array.from({ length: resolution * resolution }, (_, index) => index);

  return (
    <LabFrame
      lesson="32"
      title="同一套二维头：编号分类还是连续坐标"
      description="左屏是带编号的玩具 UI，右屏是桌面俯视图。先预测低分辨率下谁更脆，再切换 SoM / 连续头并训练共享比例头。"
    >
      <div className={styles.workspace}>
        <form className={styles.controls} onSubmit={(event) => event.preventDefault()}>
          <h3>接地控制台</h3>
          <label>
            <span>视觉网格 <output>{resolution}×{resolution}</output></span>
            <input
              type="range"
              min="4"
              max="32"
              step="4"
              value={resolution}
              onChange={(event) => {
                setResolution(Number(event.target.value));
                setRevealed(false);
              }}
            />
          </label>
          <label>
            <span>共享头训练步 <output>{steps}</output></span>
            <input
              type="range"
              min="0"
              max="12"
              step="1"
              value={steps}
              onChange={(event) => {
                setSteps(Number(event.target.value));
                setRevealed(false);
              }}
            />
          </label>
          <fieldset>
            <legend>预测头</legend>
            <label>
              <input
                type="radio"
                name="head-mode"
                checked={mode === "continuous"}
                onChange={() => {
                  setMode("continuous");
                  setRevealed(false);
                }}
              />
              <span>连续坐标回归</span>
            </label>
            <label>
              <input
                type="radio"
                name="head-mode"
                checked={mode === "som"}
                onChange={() => {
                  setMode("som");
                  setRevealed(false);
                }}
              />
              <span>SoM 编号分类</span>
            </label>
          </fieldset>
          <fieldset>
            <legend>观察域</legend>
            <label>
              <input
                type="radio"
                name="domain"
                checked={domain === "ui"}
                onChange={() => {
                  setDomain("ui");
                  setRevealed(false);
                }}
              />
              <span>左屏 UI</span>
            </label>
            <label>
              <input
                type="radio"
                name="domain"
                checked={domain === "table"}
                onChange={() => {
                  setDomain("table");
                  setRevealed(false);
                }}
              />
              <span>右屏桌面</span>
            </label>
          </fieldset>
          <p className={styles.note}>
            训练始终混用 UI 与桌面样本。域开关只改你看哪一侧，不拆开共享头。
          </p>
        </form>

        <div className={styles.stage}>
          <div className={styles.screens}>
            <figure className={domain === "ui" ? styles.screenActive : styles.screen}>
              <figcaption>左屏 · 玩具 UI</figcaption>
              <div
                className={styles.uiStage}
                style={{ "--res": resolution } as React.CSSProperties}
              >
                <div className={styles.grid} aria-hidden="true">
                  {grid.map((cell) => (
                    <span key={`ui-cell-${cell}`} />
                  ))}
                </div>
                {UI_ITEMS.map((item) => (
                  <button
                    key={`ui-${item.id}`}
                    type="button"
                    className={styles.widget}
                    style={{
                      left: `${item.x * 100}%`,
                      top: `${item.y * 100}%`,
                      width: `${item.width * 100}%`,
                      height: `${item.height * 100}%`,
                    }}
                  >
                    <b>{item.id}</b>
                    <span>{item.name}</span>
                  </button>
                ))}
                {revealed && domain === "ui"
                  ? calculation.active.predictions.map((pred, index) => (
                      <i
                        key={`ui-pred-${index}`}
                        className={styles.pred}
                        style={{ left: `${pred.x * 100}%`, top: `${pred.y * 100}%` }}
                      />
                    ))
                  : null}
              </div>
            </figure>
            <figure className={domain === "table" ? styles.screenActive : styles.screen}>
              <figcaption>右屏 · 桌面俯视</figcaption>
              <div
                className={styles.tableStage}
                style={{ "--res": resolution } as React.CSSProperties}
              >
                <div className={styles.grid} aria-hidden="true">
                  {grid.map((cell) => (
                    <span key={`table-cell-${cell}`} />
                  ))}
                </div>
                {TABLE_ITEMS.map((item) => (
                  <div
                    key={`table-${item.id}`}
                    className={styles.object}
                    style={{
                      left: `${item.x * 100}%`,
                      top: `${item.y * 100}%`,
                      width: `${item.width * 100}%`,
                      height: `${item.height * 100}%`,
                    }}
                  >
                    <b>{item.id}</b>
                    <span>{item.name}</span>
                  </div>
                ))}
                {revealed && domain === "table"
                  ? calculation.active.predictions.map((pred, index) => (
                      <i
                        key={`table-pred-${index}`}
                        className={styles.pred}
                        style={{ left: `${pred.x * 100}%`, top: `${pred.y * 100}%` }}
                      />
                    ))
                  : null}
              </div>
            </figure>
          </div>
          <dl className={styles.metrics}>
            <div>
              <dt>当前头 · UI 误差</dt>
              <dd>{revealed ? calculation.uiNow.toFixed(3) : "—"}</dd>
            </div>
            <div>
              <dt>当前头 · 桌面误差</dt>
              <dd>{revealed ? calculation.tableNow.toFixed(3) : "—"}</dd>
            </div>
            <div>
              <dt>4×4 连续 / SoM</dt>
              <dd>
                {revealed
                  ? `${calculation.probeContinuous.toFixed(3)} / ${calculation.probeSom.toFixed(3)}`
                  : "—"}
              </dd>
            </div>
          </dl>
          <p className={styles.formula}>
            u = (x / W, y / H)；连续头停在格子中心，SoM 选中编号后回到物体中心。
            指令：{activeItems[mode === "som" ? 1 : 0].name}
          </p>
        </div>
      </div>

      <div className={styles.predict}>
        <fieldset>
          <legend>先预测：把视觉网格降到 4×4 时，哪句话成立？</legend>
          {[
            ["lowres_continuous_worse", "连续坐标误差大于 SoM"],
            ["lowres_som_worse", "SoM 误差大于连续坐标"],
            ["resolution_irrelevant", "分辨率几乎不改变两者差距"],
            ["ui_only", "共享头只会降低 UI 误差"],
          ].map(([value, label]) => (
            <label key={value}>
              <input
                type="radio"
                name="som-prediction"
                value={value}
                checked={prediction === value}
                onChange={() => {
                  setPrediction(value);
                  setRevealed(false);
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
            onClick={() => setRevealed(true)}
          >
            揭晓误差
          </button>
        </div>
      </div>
      {revealed && prediction !== "lowres_continuous_worse" && (
        <p className={styles.feedback}>
          格子变大时，连续头只能报格子中心；编号只要还能分开，点的就是物体中心。
        </p>
      )}
      <Gate passed={passed}>
        先选对“低分辨率下连续误差更大”，再把共享头训练到至少 4 步，使 UI 和桌面误差同时下降。数字来自教学模拟，不是模型输出。
      </Gate>
    </LabFrame>
  );
}
