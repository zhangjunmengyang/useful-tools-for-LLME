"use client";

import { useEffect, useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import {
  numberFrom,
  round,
  stringFrom,
  useCompletionGate,
} from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson28FlowActionLab.module.css";

type Point = { x: number; y: number };
type Rect = { x: number; y: number; w: number; h: number };
type Prediction = "" | "coarse-and-stale" | "always-smooth" | "latency-only";

const START: Point = { x: 42, y: 208 };
const WAYPOINT: Point = { x: 191, y: 40 };
const TARGET_A: Point = { x: 358, y: 208 };
const TARGET_B: Point = { x: 358, y: 46 };
const OBSTACLE: Rect = { x: 152, y: 88, w: 78, h: 148 };
const N_REF = 8;
const DISTURB_AT = 5;
const BUDGET = 32;
const REACH = 12;

function lerp(left: Point, right: Point, amount: number): Point {
  return {
    x: left.x + (right.x - left.x) * amount,
    y: left.y + (right.y - left.y) * amount,
  };
}

function distance(left: Point, right: Point) {
  return Math.hypot(left.x - right.x, left.y - right.y);
}

function resample(points: Point[], count: number): Point[] {
  if (count <= 1) return [{ ...points[0] }];
  const lengths: number[] = [];
  let total = 0;
  for (let index = 0; index < points.length - 1; index += 1) {
    const length = distance(points[index], points[index + 1]);
    lengths.push(length);
    total += length;
  }
  return Array.from({ length: count }, (_, index) => {
    const target = (index / (count - 1)) * total;
    let acc = 0;
    for (let segment = 0; segment < lengths.length; segment += 1) {
      const length = lengths[segment];
      if (acc + length >= target || segment === lengths.length - 1) {
        const local = length === 0 ? 0 : (target - acc) / length;
        return lerp(
          points[segment],
          points[segment + 1],
          Math.min(1, Math.max(0, local)),
        );
      }
      acc += length;
    }
    return { ...points[points.length - 1] };
  });
}

function blend(noise: Point[], clean: Point[], tau: number) {
  return noise.map((point, index) => lerp(point, clean[index], tau));
}

function makeChunk(from: Point, to: Point, horizon: number, tau: number) {
  const noise = resample([from, to], horizon);
  const clean = resample([from, WAYPOINT, to], horizon);
  return blend(noise, clean, tau);
}

function pointInRect(point: Point, rect: Rect) {
  return (
    point.x >= rect.x &&
    point.x <= rect.x + rect.w &&
    point.y >= rect.y &&
    point.y <= rect.y + rect.h
  );
}

function cross(origin: Point, left: Point, right: Point) {
  return (left.x - origin.x) * (right.y - origin.y) - (left.y - origin.y) * (right.x - origin.x);
}

function segmentsIntersect(a: Point, b: Point, c: Point, d: Point) {
  const d1 = cross(a, b, c);
  const d2 = cross(a, b, d);
  const d3 = cross(c, d, a);
  const d4 = cross(c, d, b);
  return (
    ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
    ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))
  );
}

function segmentHitsRect(start: Point, end: Point, rect: Rect) {
  if (pointInRect(start, rect) || pointInRect(end, rect)) return true;
  const corners = [
    { x: rect.x, y: rect.y },
    { x: rect.x + rect.w, y: rect.y },
    { x: rect.x + rect.w, y: rect.y + rect.h },
    { x: rect.x, y: rect.y + rect.h },
  ];
  const edges: Array<[Point, Point]> = [
    [corners[0], corners[1]],
    [corners[1], corners[2]],
    [corners[2], corners[3]],
    [corners[3], corners[0]],
  ];
  return edges.some(([left, right]) => segmentsIntersect(start, end, left, right));
}

function pathHitsObstacle(path: Point[]) {
  return path.some((point, index) => {
    if (index === 0) return false;
    return segmentHitsRect(path[index - 1], point, OBSTACLE);
  });
}

function skipCurrent(chunk: Point[], position: Point) {
  if (chunk.length > 1 && distance(chunk[0], position) < 1) return 1;
  return 0;
}

function simulate(
  steps: number,
  horizon: number,
  executeK: number,
  moveTarget: boolean,
) {
  const tau = Math.min(1, steps / N_REF);
  let position = { ...START };
  let target = { ...TARGET_A };
  const executed: Point[] = [{ ...position }];
  let chunk = makeChunk(position, target, horizon, tau);
  const planned = [chunk.map((point) => ({ ...point }))];
  let cursor = 0;
  let replans = 0;
  let moved = false;
  let stepsSincePlan = 0;

  for (let tick = 0; tick < BUDGET; tick += 1) {
    if (cursor >= chunk.length) {
      if (executeK < horizon) {
        chunk = makeChunk(position, target, horizon, tau);
        planned.push(chunk.map((point) => ({ ...point })));
        cursor = skipCurrent(chunk, position);
        replans += 1;
        stepsSincePlan = 0;
      } else {
        break;
      }
    }

    position = chunk[cursor];
    executed.push(position);
    cursor += 1;
    stepsSincePlan += 1;

    if (moveTarget && !moved && executed.length - 1 === DISTURB_AT) {
      target = { ...TARGET_B };
      moved = true;
    }

    if (distance(position, target) < REACH && (!moveTarget || moved)) {
      break;
    }

    if (stepsSincePlan >= executeK && executeK < horizon) {
      chunk = makeChunk(position, target, horizon, tau);
      planned.push(chunk.map((point) => ({ ...point })));
      cursor = skipCurrent(chunk, position);
      replans += 1;
      stepsSincePlan = 0;
    }
  }

  const end = executed[executed.length - 1];
  const endToOld = distance(end, TARGET_A);
  const endToNew = distance(end, TARGET_B);
  const collision = pathHitsObstacle(executed);
  const stale =
    moveTarget &&
    executeK >= horizon &&
    endToOld <= 22 &&
    endToNew > endToOld + 40;

  return {
    tau,
    executed,
    remainder: chunk.slice(cursor),
    planned,
    replans,
    moved,
    collision,
    stale,
    endToOld,
    endToNew,
  };
}

function toPolyline(points: Point[]) {
  return points.map((point) => `${point.x},${point.y}`).join(" ");
}

export function Lesson28FlowActionLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    steps: numberFrom(initialState, "steps", 8, 1, 16),
    horizon: numberFrom(initialState, "horizon", 18, 6, 24),
    executeK: numberFrom(initialState, "executeK", 18, 1, 24),
    moveTarget: stringFrom(initialState, "moveTarget", "1") === "1",
    prediction: stringFrom(initialState, "prediction", "") as Prediction,
  };
  const [steps, setSteps] = useState(defaults.steps);
  const [horizon, setHorizon] = useState(defaults.horizon);
  const [executeK, setExecuteK] = useState(
    Math.min(defaults.executeK, defaults.horizon),
  );
  const [moveTarget, setMoveTarget] = useState(defaults.moveTarget);
  const [prediction, setPrediction] = useState<Prediction>(
    defaults.prediction === "coarse-and-stale" ||
      defaults.prediction === "always-smooth" ||
      defaults.prediction === "latency-only"
      ? defaults.prediction
      : "",
  );
  const [ran, setRan] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [visibleCount, setVisibleCount] = useState(0);
  const [seenCollision, setSeenCollision] = useState(false);
  const [seenStale, setSeenStale] = useState(false);

  const result = useMemo(
    () => simulate(steps, horizon, executeK, moveTarget),
    [executeK, horizon, moveTarget, steps],
  );

  useEffect(() => {
    if (!playing) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduced) {
      setVisibleCount(result.executed.length);
      setPlaying(false);
      return;
    }
    const timer = window.setInterval(() => {
      setVisibleCount((current) => {
        const next = Math.min(result.executed.length, current + 1);
        if (next >= result.executed.length) {
          window.clearInterval(timer);
          setPlaying(false);
        }
        return next;
      });
    }, 70);
    return () => window.clearInterval(timer);
  }, [playing, result.executed.length]);

  const revealed = ran && Boolean(prediction);
  const trail = result.executed.slice(0, Math.max(1, visibleCount));
  const gripper = trail[trail.length - 1];
  const passed =
    revealed &&
    !playing &&
    prediction === "coarse-and-stale" &&
    seenCollision &&
    seenStale;

  const completion = useMemo(
    () => ({
      lessonId: 28,
      steps,
      horizon,
      executeK,
      moveTarget,
      tau: round(result.tau, 3),
      collision: result.collision,
      stale: result.stale,
      replans: result.replans,
      endToOld: round(result.endToOld, 2),
      endToNew: round(result.endToNew, 2),
    }),
    [
      executeK,
      horizon,
      moveTarget,
      result.collision,
      result.endToNew,
      result.endToOld,
      result.replans,
      result.stale,
      result.tau,
      steps,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function invalidate() {
    setRan(false);
    setPlaying(false);
    setVisibleCount(0);
  }

  function reset() {
    setSteps(defaults.steps);
    setHorizon(defaults.horizon);
    setExecuteK(Math.min(defaults.executeK, defaults.horizon));
    setMoveTarget(defaults.moveTarget);
    setPrediction("");
    setSeenCollision(false);
    setSeenStale(false);
    invalidate();
  }

  function run() {
    const snapshot = simulate(steps, horizon, executeK, moveTarget);
    if (snapshot.collision) setSeenCollision(true);
    if (snapshot.stale) setSeenStale(true);
    setRan(true);
    setVisibleCount(1);
    setPlaying(true);
  }

  return (
    <LabFrame
      lesson="28"
      title="二维抓取：去噪步数、动作块和重规划"
      description="教学模拟，不是模型输出。直线噪声路径穿过障碍，干净路径绕行。去噪步数映射为积分完成度；只执行前 k 步后才按新观察重规划。"
    >
      <div className={styles.workspace}>
        <form
          className={styles.controls}
          onSubmit={(event) => event.preventDefault()}
        >
          <h3>控制台</h3>
          <p>
            τ = min(1, N / {N_REF})。N 小于 {N_REF}{" "}
            时轨迹更靠近穿障直线。k = H 时整段开环放出。
          </p>
          <label>
            <span>
              去噪步数 N <output>{steps}</output>
            </span>
            <input
              type="range"
              min="1"
              max="16"
              step="1"
              value={steps}
              onChange={(event) => {
                setSteps(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <label>
            <span>
              chunk 长度 H <output>{horizon}</output>
            </span>
            <input
              type="range"
              min="6"
              max="24"
              step="1"
              value={horizon}
              onChange={(event) => {
                const next = Number(event.target.value);
                setHorizon(next);
                setExecuteK((current) => Math.min(current, next));
                invalidate();
              }}
            />
          </label>
          <label>
            <span>
              执行前 k 步 <output>{executeK}</output>
            </span>
            <input
              type="range"
              min="1"
              max={horizon}
              step="1"
              value={executeK}
              onChange={(event) => {
                setExecuteK(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <label className={styles.check}>
            <input
              type="checkbox"
              checked={moveTarget}
              onChange={(event) => {
                setMoveTarget(event.target.checked);
                invalidate();
              }}
            />
            执行第 {DISTURB_AT} 步后把目标挪走
          </label>
        </form>

        <div className={styles.stage}>
          <svg
            className={styles.scene}
            viewBox="0 0 400 280"
            role="img"
            aria-label="二维抓取桌面"
          >
            <rect x="0" y="0" width="400" height="280" fill="#142029" />
            <rect
              x={OBSTACLE.x}
              y={OBSTACLE.y}
              width={OBSTACLE.w}
              height={OBSTACLE.h}
              fill="#354853"
              stroke="#718794"
            />
            <text x={OBSTACLE.x + 10} y={OBSTACLE.y + 20} fill="#9cb0ba" fontSize="10">
              障碍
            </text>
            <circle cx={START.x} cy={START.y} r="6" fill="#70c2e2" />
            <circle
              cx={TARGET_A.x}
              cy={TARGET_A.y}
              r="8"
              fill={moveTarget ? "transparent" : "#e8b86d"}
              stroke="#e8b86d"
            />
            {moveTarget ? (
              <circle cx={TARGET_B.x} cy={TARGET_B.y} r="8" fill="#e8b86d" />
            ) : null}
            {ran && result.remainder.length > 1 ? (
              <polyline
                points={toPolyline([gripper, ...result.remainder])}
                fill="none"
                stroke="#4a616e"
                strokeDasharray="4 4"
                strokeWidth="2"
              />
            ) : null}
            {trail.length > 1 ? (
              <polyline
                points={toPolyline(trail)}
                fill="none"
                stroke="#70c2e2"
                strokeWidth="2.4"
              />
            ) : null}
            <circle cx={gripper.x} cy={gripper.y} r="7" fill="#89d3ef" />
            <text x="12" y="18" fill="#91a5b0" fontSize="10">
              夹爪从左下出发，抓右侧目标
            </text>
          </svg>
          <p className={styles.legend}>
            <span>
              <i style={{ color: "#70c2e2" }} />
              已执行
            </span>
            <span>
              <i style={{ color: "#4a616e" }} />
              未执行剩余 chunk
            </span>
            <span>
              <i style={{ color: "#e8b86d" }} />
              目标
            </span>
          </p>
          <p className={styles.formula}>
            A(τ) = (1−τ)·直线穿障 + τ·绕行；τ = {revealed ? result.tau.toFixed(3) : "运行后揭晓"}
          </p>
          <dl className={styles.metrics}>
            <div>
              <dt>是否穿障碍</dt>
              <dd className={revealed ? undefined : styles.hidden}>
                {revealed ? (result.collision ? "是" : "否") : "先预测再运行"}
              </dd>
            </div>
            <div>
              <dt>重规划次数</dt>
              <dd className={revealed ? undefined : styles.hidden}>
                {revealed ? result.replans : "—"}
              </dd>
            </div>
            <div>
              <dt>沿旧目标？</dt>
              <dd className={revealed ? undefined : styles.hidden}>
                {revealed ? (result.stale ? "是" : "否") : "—"}
              </dd>
            </div>
            <div>
              <dt>终点到旧目标</dt>
              <dd className={revealed ? undefined : styles.hidden}>
                {revealed ? result.endToOld.toFixed(1) : "—"}
              </dd>
            </div>
            <div>
              <dt>终点到新目标</dt>
              <dd className={revealed ? undefined : styles.hidden}>
                {revealed ? result.endToNew.toFixed(1) : "—"}
              </dd>
            </div>
            <div>
              <dt>已观察到的失败</dt>
              <dd>
                {seenCollision ? "穿障" : "穿障未触发"} /{" "}
                {seenStale ? "旧轨迹" : "旧轨迹未触发"}
              </dd>
            </div>
          </dl>
        </div>
      </div>

      <div className={styles.predict}>
        <fieldset>
          <legend>先预测：步数过少、以及长 chunk 且不重规划时，分别会发生什么？</legend>
          <label>
            <input
              type="radio"
              name="flow-action-prediction"
              checked={prediction === "coarse-and-stale"}
              onChange={() => {
                setPrediction("coarse-and-stale");
                invalidate();
              }}
            />
            <span>步数过少穿障碍；长 chunk 不重规划时目标被挪走仍沿旧轨迹</span>
          </label>
          <label>
            <input
              type="radio"
              name="flow-action-prediction"
              checked={prediction === "always-smooth"}
              onChange={() => {
                setPrediction("always-smooth");
                invalidate();
              }}
            />
            <span>步数越少越平滑；长 chunk 会自动改去新目标</span>
          </label>
          <label>
            <input
              type="radio"
              name="flow-action-prediction"
              checked={prediction === "latency-only"}
              onChange={() => {
                setPrediction("latency-only");
                invalidate();
              }}
            />
            <span>两者只改变延迟，不改变几何轨迹</span>
          </label>
        </fieldset>
        <div className={styles.actions}>
          <button type="button" className={styles.reset} onClick={reset}>
            重置
          </button>
          <button
            type="button"
            className={styles.run}
            disabled={!prediction || playing}
            onClick={run}
          >
            {playing ? "轨迹执行中…" : "运行当前参数"}
          </button>
        </div>
      </div>
      {revealed && prediction !== "coarse-and-stale" ? (
        <p className={styles.feedback}>
          积分未完成时混合结果靠近穿障直线；k 等于 H 时不会在目标被挪走后改计划。改预测后再跑两组参数。
        </p>
      ) : null}
      {revealed && prediction === "coarse-and-stale" && (!seenCollision || !seenStale) ? (
        <p className={styles.note}>
          预测正确。还需要触发两种验收：N 调到 3 以下观察穿障；H 与 k 都调高并勾选挪走目标，观察仍沿旧轨迹。
        </p>
      ) : null}
      <Gate passed={passed}>
        先提交正确预测，再分别触发“步数过少穿障碍”和“长 chunk 不重规划仍追旧目标”。
      </Gate>
    </LabFrame>
  );
}
