"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import { numberFrom, stringFrom, useCompletionGate } from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson36NavActionLab.module.css";

type PhaseId = "nav" | "grasp";
type PolicyId = "waypoint" | "velocity";
type PredictionId = "split" | "same" | "swap" | "both-ok";

type Cell = { c: number; r: number };

const COLS = 7;
const ROWS = 6;
const BINS = 8;
const TEXT_VOCAB = 32;
const N_MODES = 3;
const N_WAYPOINTS = 6;
const COUNTER_NODE = 3;
const ARM_START = TEXT_VOCAB;
const BASE_START = ARM_START + 7 * BINS;
const MODE_START = BASE_START + 2 * BINS;
const WAYPOINT_START = MODE_START + N_MODES;

const ARM_DIMS: Array<{ name: string; low: number; high: number }> = [
  { name: "x", low: -1, high: 1 },
  { name: "y", low: -1, high: 1 },
  { name: "z", low: -1, high: 1 },
  { name: "roll", low: -1, high: 1 },
  { name: "pitch", low: -1, high: 1 },
  { name: "yaw", low: -1, high: 1 },
  { name: "grip", low: 0, high: 1 },
];

const BASE_DIMS: Array<{ name: string; low: number; high: number }> = [
  { name: "v", low: 0, high: 1.5 },
  { name: "omega", low: -1.5, high: 1.5 },
];

const WAYPOINTS: Array<{ id: number; name: string; cell: Cell }> = [
  { id: 0, name: "起点", cell: { c: 1, r: 1 } },
  { id: 1, name: "走廊", cell: { c: 2, r: 1 } },
  { id: 2, name: "厨房门", cell: { c: 3, r: 3 } },
  { id: 3, name: "台面", cell: { c: 5, r: 4 } },
  { id: 4, name: "冰箱", cell: { c: 5, r: 1 } },
  { id: 5, name: "餐桌", cell: { c: 1, r: 4 } },
];

const CUP: Cell = { c: 5, r: 4 };
const START: Cell = { c: 1, r: 1 };

const NAV_PATH: Cell[] = [
  { c: 1, r: 1 },
  { c: 2, r: 1 },
  { c: 3, r: 1 },
  { c: 3, r: 2 },
  { c: 3, r: 3 },
  { c: 4, r: 3 },
  { c: 5, r: 3 },
  { c: 5, r: 4 },
];

const SPIN_PATH: Cell[] = [
  { c: 1, r: 1 },
  { c: 2, r: 1 },
  { c: 2, r: 2 },
  { c: 1, r: 2 },
  { c: 1, r: 1 },
  { c: 2, r: 1 },
  { c: 3, r: 1 },
  { c: 4, r: 1 },
];

const PHASE_LABEL: Record<PhaseId, string> = {
  nav: "走到台面",
  grasp: "抓杯子",
};

const POLICY_LABEL: Record<PolicyId, string> = {
  waypoint: "路点政策",
  velocity: "速度政策",
};

const PREDICTION_LABEL: Record<PredictionId, string> = {
  split: "路点政策输出非法节点索引；速度政策仍输出合法 (v,ω)，但会撞墙或转圈",
  same: "两种政策同样停住，并且同样报非法索引",
  swap: "速度政策报非法索引，路点政策撞墙",
  "both-ok": "丢掉地图后两者仍能走到台面",
};

function isWall(c: number, r: number) {
  if (c <= 0 || r <= 0 || c >= COLS - 1 || r >= ROWS - 1) return true;
  return (c === 4 && r === 1) || (c === 4 && r === 2);
}

function clipBin(index: number) {
  return Math.max(0, Math.min(BINS - 1, index));
}

function uniformBin(value: number, low: number, high: number) {
  if (value <= low) return 0;
  if (value >= high) return BINS - 1;
  const width = (high - low) / BINS;
  return clipBin(Math.floor((value - low) / width));
}

function encodeArm(action: number[]) {
  return action.map((value, dimension) => {
    const spec = ARM_DIMS[dimension];
    return ARM_START + dimension * BINS + uniformBin(value, spec.low, spec.high);
  });
}

function encodeBase(action: number[]) {
  return action.map((value, dimension) => {
    const spec = BASE_DIMS[dimension];
    return BASE_START + dimension * BINS + uniformBin(value, spec.low, spec.high);
  });
}

function encodeMode(mode: number) {
  return MODE_START + mode;
}

function encodeWaypoint(nodeId: number, nNodes: number) {
  return {
    token: WAYPOINT_START + nodeId,
    legal: nNodes > 0 && nodeId >= 0 && nodeId < nNodes,
  };
}

function cellToPercent(cell: Cell) {
  return {
    left: ((cell.c + 0.5) / COLS) * 100,
    top: ((cell.r + 0.5) / ROWS) * 100,
  };
}

function graspAction() {
  return [0.62, 0.48, -0.55, 0, 0.18, 0.05, 0.12];
}

function idleArm() {
  return [0, 0, 0, 0, 0, 0, 1];
}

type SimResult = {
  body: "arm" | "base" | "none";
  tokens: number[];
  modeToken: number;
  waypointIndex: number | null;
  nNodes: number;
  illegal: boolean;
  failMode: "none" | "illegal-index" | "collision-spin";
  velocity: [number, number] | null;
  pose: Cell;
  path: Cell[];
  spinning: boolean;
  hitWall: boolean;
  summary: string;
};

function simulate(options: {
  phase: PhaseId;
  policy: PolicyId;
  mapOn: boolean;
}): SimResult {
  const { phase, policy, mapOn } = options;
  if (phase === "grasp") {
    const action = graspAction();
    return {
      body: "arm",
      tokens: encodeArm(action),
      modeToken: encodeMode(0),
      waypointIndex: null,
      nNodes: mapOn ? N_WAYPOINTS : 0,
      illegal: false,
      failMode: "none",
      velocity: null,
      pose: CUP,
      path: [CUP],
      spinning: false,
      hitWall: false,
      summary: mapOn
        ? "抓杯子走手臂 7 维词表，不查路点表。夹爪闭合，模式=控臂。"
        : "地图关掉也不查路点。抓取仍用手臂词表；本课失败探针放在走到台面这一段。",
    };
  }

  if (policy === "waypoint") {
    if (mapOn) {
      const coded = encodeWaypoint(COUNTER_NODE, N_WAYPOINTS);
      return {
        body: "none",
        tokens: [coded.token],
        modeToken: encodeMode(1),
        waypointIndex: COUNTER_NODE,
        nNodes: N_WAYPOINTS,
        illegal: false,
        failMode: "none",
        velocity: null,
        pose: CUP,
        path: NAV_PATH,
        spinning: false,
        hitWall: false,
        summary: `路点政策沿图走到台面节点 ${COUNTER_NODE}，token ${coded.token} 落在 [0, N) 内。`,
      };
    }
    const coded = encodeWaypoint(COUNTER_NODE, 0);
    return {
      body: "none",
      tokens: [coded.token],
      modeToken: encodeMode(1),
      waypointIndex: COUNTER_NODE,
      nNodes: 0,
      illegal: true,
      failMode: "illegal-index",
      velocity: null,
      pose: START,
      path: [START],
      spinning: false,
      hitWall: false,
      summary: `地图节点数 N=0。政策仍吐出旧节点 ${COUNTER_NODE}，token ${coded.token} 无法解码，底盘不移动。`,
    };
  }

  if (mapOn) {
    const velocity: [number, number] = [0.45, 0.08];
    return {
      body: "base",
      tokens: encodeBase(velocity),
      modeToken: encodeMode(1),
      waypointIndex: null,
      nNodes: N_WAYPOINTS,
      illegal: false,
      failMode: "none",
      velocity,
      pose: CUP,
      path: NAV_PATH,
      spinning: false,
      hitWall: false,
      summary: "速度政策输出合法 (v,ω)，沿可通行格子走到台面。token 落在底盘切片内。",
    };
  }

  const velocity: [number, number] = [1.1, 1.45];
  const wallCell = SPIN_PATH[SPIN_PATH.length - 1];
  return {
    body: "base",
    tokens: encodeBase(velocity),
    modeToken: encodeMode(1),
    waypointIndex: null,
    nNodes: 0,
    illegal: false,
    failMode: "collision-spin",
    velocity,
    pose: wallCell,
    path: SPIN_PATH,
    spinning: true,
    hitWall: isWall(wallCell.c, wallCell.r) || wallCell.c === 4,
    summary:
      "丢掉地图后速度头仍输出合法 (v=1.1, ω=1.45)。机器人先转圈，再撞上隔墙。失败发生在物理层，不发生在词表越界。",
  };
}

export function Lesson36NavActionLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    phase: stringFrom(initialState, "phase", "nav") as PhaseId,
    policy: stringFrom(initialState, "policy", "waypoint") as PolicyId,
    mapOn: numberFrom(initialState, "mapOn", 1, 0, 1),
    prediction: stringFrom(initialState, "prediction", "") as PredictionId | "",
  };
  const [phase, setPhase] = useState<PhaseId>(
    defaults.phase === "grasp" ? "grasp" : "nav",
  );
  const [policy, setPolicy] = useState<PolicyId>(
    defaults.policy === "velocity" ? "velocity" : "waypoint",
  );
  const [mapOn, setMapOn] = useState(defaults.mapOn === 1);
  const [prediction, setPrediction] = useState<PredictionId | "">(
    defaults.prediction === "split" ||
      defaults.prediction === "same" ||
      defaults.prediction === "swap" ||
      defaults.prediction === "both-ok"
      ? defaults.prediction
      : "",
  );
  const [ran, setRan] = useState(false);
  const [sawWaypointIllegal, setSawWaypointIllegal] = useState(false);
  const [sawVelocityPhysical, setSawVelocityPhysical] = useState(false);

  const waypointLost = useMemo(
    () => simulate({ phase: "nav", policy: "waypoint", mapOn: false }),
    [],
  );
  const velocityLost = useMemo(
    () => simulate({ phase: "nav", policy: "velocity", mapOn: false }),
    [],
  );

  const current = useMemo(
    () => simulate({ phase, policy, mapOn }),
    [mapOn, phase, policy],
  );

  const failuresDiffer =
    waypointLost.failMode === "illegal-index" &&
    velocityLost.failMode === "collision-spin" &&
    waypointLost.illegal &&
    !velocityLost.illegal &&
    (velocityLost.hitWall || velocityLost.spinning);

  const passed =
    ran &&
    prediction === "split" &&
    sawWaypointIllegal &&
    sawVelocityPhysical &&
    failuresDiffer;

  const completion = useMemo(
    () => ({
      lessonId: 36,
      phase,
      policy,
      mapOn,
      prediction,
      waypointIllegal: waypointLost.illegal,
      velocityIllegal: velocityLost.illegal,
      velocityFail: velocityLost.failMode,
      waypointIndex: waypointLost.waypointIndex,
      nNodesLost: waypointLost.nNodes,
      failuresDiffer,
    }),
    [
      failuresDiffer,
      mapOn,
      phase,
      policy,
      prediction,
      velocityLost.failMode,
      velocityLost.illegal,
      waypointLost.illegal,
      waypointLost.nNodes,
      waypointLost.waypointIndex,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function runCurrent() {
    setRan(true);
    if (phase === "nav" && !mapOn && policy === "waypoint") {
      setSawWaypointIllegal(true);
    }
    if (phase === "nav" && !mapOn && policy === "velocity") {
      setSawVelocityPhysical(true);
    }
  }

  function runProbe(kind: "waypoint" | "velocity") {
    setPhase("nav");
    setMapOn(false);
    setPolicy(kind);
    setRan(true);
    if (kind === "waypoint") setSawWaypointIllegal(true);
    else setSawVelocityPhysical(true);
  }

  function reset() {
    setPhase("nav");
    setPolicy("waypoint");
    setMapOn(true);
    setPrediction("");
    setRan(false);
    setSawWaypointIllegal(false);
    setSawVelocityPhysical(false);
  }

  const cells = useMemo(() => {
    const list: Array<{ c: number; r: number; kind: "wall" | "floor" | "counter" }> =
      [];
    for (let r = 0; r < ROWS; r += 1) {
      for (let c = 0; c < COLS; c += 1) {
        const kind =
          CUP.c === c && CUP.r === r
            ? "counter"
            : isWall(c, r)
              ? "wall"
              : "floor";
        list.push({ c, r, kind });
      }
    }
    return list;
  }, []);

  const pose = ran ? current.pose : START;
  const robotPercent = cellToPercent(pose);
  const cupPercent = cellToPercent(CUP);
  const revealClass =
    ran && current.failMode === "illegal-index"
      ? styles.failWaypoint
      : ran && current.failMode === "collision-spin"
        ? styles.failVelocity
        : styles.reveal;

  return (
    <LabFrame
      lesson="36"
      title="厨房任务的两套动作词表"
      description="教学模拟，不是模型输出。把「去厨房拿杯子」切成走到台面和抓杯子。先预测丢掉地图后两种政策如何失败，再揭晓非法索引与撞墙或转圈。"
    >
      <div className={styles.workspace}>
        <form
          className={styles.controls}
          onSubmit={(event) => event.preventDefault()}
        >
          <h3>控制台</h3>
          <label>
            <span>任务切分</span>
            <select
              value={phase}
              onChange={(event) => {
                setPhase(event.target.value as PhaseId);
                setRan(false);
              }}
            >
              <option value="nav">走到台面</option>
              <option value="grasp">抓杯子</option>
            </select>
          </label>
          <label>
            <span>底盘政策</span>
            <select
              value={policy}
              onChange={(event) => {
                setPolicy(event.target.value as PolicyId);
                setRan(false);
              }}
            >
              <option value="waypoint">路点政策</option>
              <option value="velocity">速度政策</option>
            </select>
          </label>
          <fieldset>
            <legend>地图</legend>
            <label className={styles.toggle}>
              <input
                type="checkbox"
                checked={mapOn}
                onChange={(event) => {
                  setMapOn(event.target.checked);
                  setRan(false);
                }}
              />
              <span>{mapOn ? "地图接通" : "地图已关掉"}</span>
            </label>
          </fieldset>
          <p className={styles.note}>
            词表切片：文本 [0, 32)、手臂 [32, 88)、底盘 [88, 104)、模式 [104, 107)、路点
            [107, 113)。
          </p>
        </form>

        <div className={styles.stage}>
          <p className={styles.formula}>
            <span>
              <strong>i</strong> {mapOn ? COUNTER_NODE : COUNTER_NODE} / N{" "}
              {mapOn ? N_WAYPOINTS : 0}
            </span>
            <span>
              <strong>(v,ω)</strong>{" "}
              {current.velocity
                ? `${current.velocity[0].toFixed(2)}, ${current.velocity[1].toFixed(2)}`
                : phase === "grasp"
                  ? "手臂段不使用"
                  : "路点段不使用"}
            </span>
            <span>
              <strong>模式</strong> {phase === "grasp" ? "控臂" : "控底盘"}
            </span>
          </p>
          <div className={styles.tableWrap}>
            <div
              className={`${styles.kitchen} ${mapOn ? "" : styles.kitchenBlind}`}
              aria-label="厨房俯视图"
            >
              {cells.map((cell) => (
                <span
                  key={`${cell.c}-${cell.r}`}
                  className={`${styles.cell} ${
                    cell.kind === "wall"
                      ? styles.wall
                      : cell.kind === "counter"
                        ? styles.counter
                        : styles.floor
                  }`}
                />
              ))}
              <span className={styles.kitchenHint}>
                {mapOn ? "厨房俯视 · 拓扑图可见" : "地图已关掉 · 节点表为空"}
              </span>
              {WAYPOINTS.map((node) => {
                const percent = cellToPercent(node.cell);
                return (
                  <b
                    key={node.id}
                    className={`${styles.marker} ${styles.waypoint} ${
                      mapOn ? "" : styles.waypointHidden
                    }`}
                    style={{ left: `${percent.left}%`, top: `${percent.top}%` }}
                  >
                    {node.id}
                  </b>
                );
              })}
              {ran &&
                current.path.map((cell, index) => {
                  const percent = cellToPercent(cell);
                  return (
                    <i
                      key={`p-${index}`}
                      className={styles.pathDot}
                      style={{ left: `${percent.left}%`, top: `${percent.top}%` }}
                    />
                  );
                })}
              <b
                className={`${styles.marker} ${styles.cup}`}
                style={{ left: `${cupPercent.left}%`, top: `${cupPercent.top}%` }}
              >
                杯
              </b>
              <b
                className={`${styles.marker} ${styles.robot}`}
                style={{
                  left: `${robotPercent.left}%`,
                  top: `${robotPercent.top}%`,
                }}
              >
                机
              </b>
            </div>
            <p className={styles.caption}>
              {PHASE_LABEL[phase]} · {POLICY_LABEL[policy]} ·{" "}
              {mapOn ? "地图接通" : "地图已关掉"}
            </p>
          </div>

          <dl className={styles.metrics}>
            <div>
              <dt>输出 token</dt>
              <dd>{ran ? current.tokens.join(" ") : "—"}</dd>
            </div>
            <div>
              <dt>路点索引</dt>
              <dd>
                {ran
                  ? current.waypointIndex === null
                    ? "未使用"
                    : `${current.waypointIndex} / N=${current.nNodes}${
                        current.illegal ? " · 非法" : " · 合法"
                      }`
                  : "—"}
              </dd>
            </div>
            <div>
              <dt>失败模式</dt>
              <dd>
                {ran
                  ? current.failMode === "illegal-index"
                    ? "非法索引"
                    : current.failMode === "collision-spin"
                      ? "撞墙或转圈"
                      : "无"
                  : "—"}
              </dd>
            </div>
          </dl>

          {ran && (
            <div className={revealClass}>
              <p>
                <strong>揭晓：</strong>
                {current.summary}
              </p>
              <p>
                <strong>模式 token：</strong>
                {current.modeToken}
                {phase === "grasp"
                  ? "（控臂）"
                  : "（控底盘）"}
                。手臂 idle 对照 {encodeArm(idleArm()).join(" ")}。
              </p>
              <p>
                <strong>对照：</strong>
                路点丢地图 {waypointLost.illegal ? "非法" : "合法"}，速度丢地图 token{" "}
                {velocityLost.tokens.join(" ")}
                {velocityLost.illegal ? " 越界" : " 仍在底盘切片"}。
              </p>
            </div>
          )}
        </div>
      </div>

      <div className={styles.predict}>
        <fieldset>
          <legend>
            先预测：走到台面时关掉地图，路点政策和速度政策会怎样失败？
          </legend>
          {(Object.keys(PREDICTION_LABEL) as PredictionId[]).map((value) => (
            <label key={value}>
              <input
                type="radio"
                name="nav-action-prediction"
                value={value}
                checked={prediction === value}
                onChange={() => {
                  setPrediction(value);
                  setRan(false);
                  setSawWaypointIllegal(false);
                  setSawVelocityPhysical(false);
                }}
              />
              <span>{PREDICTION_LABEL[value]}</span>
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
            onClick={runCurrent}
          >
            揭晓当前设置
          </button>
          <button
            type="button"
            className={styles.run}
            disabled={!prediction}
            onClick={() => runProbe("waypoint")}
          >
            揭晓路点丢地图
          </button>
          <button
            type="button"
            className={styles.run}
            disabled={!prediction}
            onClick={() => runProbe("velocity")}
          >
            揭晓速度丢地图
          </button>
        </div>
      </div>

      {ran && prediction && prediction !== "split" && (
        <p className={styles.feedback}>
          两种政策的失败不在同一层。路点政策没有节点表可读，输出的是非法索引；速度政策的
          (v,ω) 仍在底盘词表里，电机会转圈或撞墙。
        </p>
      )}
      {ran && sawWaypointIllegal && sawVelocityPhysical && (
        <p className={styles.feedback}>
          路点丢地图：索引 {waypointLost.waypointIndex}，N={waypointLost.nNodes}，
          {waypointLost.illegal ? "非法" : "合法"}。速度丢地图：token{" "}
          {velocityLost.tokens.join(" ")}，
          {velocityLost.hitWall ? "撞墙" : ""}
          {velocityLost.spinning ? "转圈" : ""}，索引
          {velocityLost.illegal ? "非法" : "仍合法"}。
        </p>
      )}
      <Gate passed={passed}>
        先提交「失败模式不同」的预测，再分别揭晓路点丢地图与速度丢地图。抓杯子阶段应走手臂词表，不查路点。
      </Gate>
    </LabFrame>
  );
}
