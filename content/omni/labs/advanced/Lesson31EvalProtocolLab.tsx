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
import styles from "./Lesson31EvalProtocolLab.module.css";

type ProtocolId = "fixed" | "random" | "distract";

type Ranking =
  | "fixed>random>distract"
  | "fixed>distract>random"
  | "random>fixed>distract"
  | "random>distract>fixed"
  | "distract>fixed>random"
  | "distract>random>fixed";

type ObjectSpec = {
  id: string;
  x: number;
  y: number;
  kind: "target" | "distract";
  label: string;
};

const RANKING_OPTIONS: { value: Ranking; label: string }[] = [
  {
    value: "fixed>random>distract",
    label: "固定初始态 > 初始态随机 > 加 distractor",
  },
  {
    value: "fixed>distract>random",
    label: "固定初始态 > 加 distractor > 初始态随机",
  },
  {
    value: "random>fixed>distract",
    label: "初始态随机 > 固定初始态 > 加 distractor",
  },
  {
    value: "random>distract>fixed",
    label: "初始态随机 > 加 distractor > 固定初始态",
  },
  {
    value: "distract>fixed>random",
    label: "加 distractor > 固定初始态 > 初始态随机",
  },
  {
    value: "distract>random>fixed",
    label: "加 distractor > 初始态随机 > 固定初始态",
  },
];

const PROTOCOL_LABEL: Record<ProtocolId, string> = {
  fixed: "固定初始态",
  random: "初始态随机",
  distract: "加 distractor",
};

function wilson(k: number, n: number, z = 1.96) {
  const p = k / n;
  const z2 = z * z;
  const denom = 1 + z2 / n;
  const center = (p + z2 / (2 * n)) / denom;
  const margin =
    (z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / denom;
  return {
    p,
    low: Math.max(0, center - margin),
    high: Math.min(1, center + margin),
  };
}

function protocolProbabilities(radius: number, distractors: number) {
  const fixed = 0.86;
  const random = fixed - 0.04 - 0.22 * radius;
  const distract = random - 0.06 * distractors;
  return { fixed, random, distract };
}

function guaranteedCounts(
  probabilities: { fixed: number; random: number; distract: number },
  trials: number,
) {
  const rawFixed = Math.round(probabilities.fixed * trials);
  const rawRandom = Math.round(probabilities.random * trials);
  const rawDistract = Math.round(probabilities.distract * trials);
  const fixed = Math.min(trials, Math.max(2, rawFixed));
  const random = Math.min(fixed - 1, Math.max(1, rawRandom));
  const distract = Math.min(random - 1, Math.max(0, rawDistract));
  return { fixed, random, distract };
}

function sceneObjects(
  protocol: ProtocolId,
  radius: number,
  distractors: number,
): ObjectSpec[] {
  const objects: ObjectSpec[] = [
    {
      id: `${protocol}-target`,
      x: protocol === "fixed" ? 42 : 42 + radius * 28,
      y: protocol === "fixed" ? 38 : 38 + radius * 18,
      kind: "target",
      label: "杯",
    },
  ];
  if (protocol === "distract") {
    for (let index = 0; index < distractors; index += 1) {
      objects.push({
        id: `${protocol}-d${index}`,
        x: 18 + (index % 3) * 22,
        y: 62 + Math.floor(index / 3) * 16,
        kind: "distract",
        label: "扰",
      });
    }
  }
  return objects;
}

export function Lesson31EvalProtocolLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    radius: numberFrom(initialState, "radius", 0.45, 0.2, 1),
    distractors: numberFrom(initialState, "distractors", 2, 1, 4),
    trials: numberFrom(initialState, "trials", 25, 8, 50),
    prediction: stringFrom(initialState, "prediction", "") as Ranking | "",
    notSota: stringFrom(initialState, "notSota", ""),
  };
  const [radius, setRadius] = useState(defaults.radius);
  const [distractors, setDistractors] = useState(
    Math.round(defaults.distractors),
  );
  const [trials, setTrials] = useState(Math.round(defaults.trials));
  const [prediction, setPrediction] = useState<Ranking | "">(
    defaults.prediction,
  );
  const [notSota, setNotSota] = useState(defaults.notSota === "yes");
  const [ran, setRan] = useState(false);

  const simulation = useMemo(() => {
    const probabilities = protocolProbabilities(radius, distractors);
    const counts = guaranteedCounts(probabilities, trials);
    const order: ProtocolId[] = ["fixed", "random", "distract"];
    const rows = order.map((id) => {
      const k = counts[id];
      const interval = wilson(k, trials);
      return {
        id,
        k,
        n: trials,
        p: interval.p,
        low: interval.low,
        high: interval.high,
        objects: sceneObjects(id, radius, distractors),
      };
    });
    return {
      rows,
      rankingHeld:
        counts.fixed > counts.random && counts.random > counts.distract,
    };
  }, [distractors, radius, trials]);

  const passed =
    ran &&
    prediction === "fixed>random>distract" &&
    notSota &&
    simulation.rankingHeld;

  const completion = useMemo(
    () => ({
      lessonId: 31,
      radius,
      distractors,
      trials,
      prediction,
      notSota,
      rates: Object.fromEntries(
        simulation.rows.map((row) => [row.id, round(row.p, 4)]),
      ),
      rankingHeld: simulation.rankingHeld,
    }),
    [distractors, notSota, prediction, radius, simulation, trials],
  );
  useCompletionGate(passed, onComplete, completion);

  function reset() {
    setRadius(defaults.radius);
    setDistractors(Math.round(defaults.distractors));
    setTrials(Math.round(defaults.trials));
    setPrediction("");
    setNotSota(false);
    setRan(false);
  }

  return (
    <LabFrame
      lesson="31"
      title="三种协议下的同一政策"
      description="教学模拟，不是模型输出。同一套玩具抓取政策，只改初始态和 distractor。先预测三种成功率的排序，再揭晓点估计与 Wilson 区间。禁止把单一数字标成 SOTA。"
    >
      <div className={styles.workspace}>
        <form
          className={styles.controls}
          onSubmit={(event) => event.preventDefault()}
        >
          <h3>协议旋钮</h3>
          <label>
            <span>
              初始态随机半径 <output>{radius.toFixed(2)}</output>
            </span>
            <input
              type="range"
              min="0.2"
              max="1"
              step="0.05"
              value={radius}
              onChange={(event) => {
                setRadius(Number(event.target.value));
                setRan(false);
              }}
            />
          </label>
          <label>
            <span>
              distractor 个数 <output>{distractors}</output>
            </span>
            <input
              type="range"
              min="1"
              max="4"
              step="1"
              value={distractors}
              onChange={(event) => {
                setDistractors(Number(event.target.value));
                setRan(false);
              }}
            />
          </label>
          <label>
            <span>
              每协议试验次数 N <output>{trials}</output>
            </span>
            <input
              type="range"
              min="8"
              max="50"
              step="1"
              value={trials}
              onChange={(event) => {
                setTrials(Number(event.target.value));
                setRan(false);
              }}
            />
          </label>
          <p className={styles.note}>
            政策权重冻结。三种协议共用同一套成功判定：杯子进入放置圈且夹爪张开。夹具保证固定初始态最高、加 distractor 最低。
          </p>
        </form>

        <div className={styles.stage}>
          <div className={styles.desks} aria-label="三种评测协议的桌面">
            {simulation.rows.map((row) => (
              <article key={row.id} className={styles.desk}>
                <header>
                  <b>{PROTOCOL_LABEL[row.id]}</b>
                  <span>
                    {row.id === "fixed"
                      ? "种子锁定"
                      : row.id === "random"
                        ? `半径 ${radius.toFixed(2)}`
                        : `${distractors} 个干扰物`}
                  </span>
                </header>
                <div className={styles.table}>
                  <i className={styles.arm} />
                  {row.objects.map((object) => (
                    <span
                      key={object.id}
                      className={
                        object.kind === "target"
                          ? styles.target
                          : styles.distract
                      }
                      style={{ left: `${object.x}%`, top: `${object.y}%` }}
                    >
                      {object.label}
                    </span>
                  ))}
                  <span className={styles.goal}>放置圈</span>
                </div>
                <dl>
                  <div>
                    <dt>成功 / N</dt>
                    <dd>{ran ? `${row.k} / ${row.n}` : "待揭晓"}</dd>
                  </div>
                  <div>
                    <dt>点估计</dt>
                    <dd>{ran ? `${(row.p * 100).toFixed(1)}%` : "—"}</dd>
                  </div>
                  <div>
                    <dt>Wilson 95%</dt>
                    <dd>
                      {ran
                        ? `${(row.low * 100).toFixed(1)}–${(row.high * 100).toFixed(1)}%`
                        : "—"}
                    </dd>
                  </div>
                </dl>
                <div className={styles.barTrack} aria-hidden="true">
                  <span
                    className={styles.barFill}
                    style={{
                      width: ran ? `${row.p * 100}%` : "0%",
                    }}
                  />
                  {ran && (
                    <i
                      className={styles.ci}
                      style={{
                        left: `${row.low * 100}%`,
                        width: `${(row.high - row.low) * 100}%`,
                      }}
                    />
                  )}
                </div>
              </article>
            ))}
          </div>
          <p className={styles.banner}>
            三个数字来自同一教学政策的三种协议，禁止把其中任何一个标成 SOTA，也禁止写进模型卡的“仿真即真机”栏。
          </p>
        </div>
      </div>

      <div className={styles.predict}>
        <fieldset>
          <legend>先预测三种协议的成功率排序，再运行</legend>
          {RANKING_OPTIONS.map((option) => (
            <label key={option.value}>
              <input
                type="radio"
                name="eval-ranking"
                value={option.value}
                checked={prediction === option.value}
                onChange={() => {
                  setPrediction(option.value);
                  setRan(false);
                }}
              />
              <span>{option.label}</span>
            </label>
          ))}
        </fieldset>
        <label className={styles.ack}>
          <input
            type="checkbox"
            checked={notSota}
            onChange={(event) => {
              setNotSota(event.target.checked);
            }}
          />
          <span>我不会把这里的单一成功率标成 SOTA 或真机能力</span>
        </label>
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
            揭晓三种协议
          </button>
        </div>
      </div>
      {ran && prediction !== "fixed>random>distract" && (
        <p className={styles.feedback}>
          排序夹具是固定初始态最高、加 distractor 最低。随机初始态关掉了“背住这一摆”的捷径；distractor 再关掉“看见桌上有物体就抓”的捷径。政策没变，协议在拆层。
        </p>
      )}
      {ran && prediction === "fixed>random>distract" && !notSota && (
        <p className={styles.feedback}>
          排序对了。还需要勾选：不会把单一数字标成 SOTA。
        </p>
      )}
      <Gate passed={passed}>
        提交正确排序、勾选非 SOTA 声明，并确认固定初始态最高、加 distractor
        最低。Wilson 区间随 N 变宽变窄，它测的是试验次数，不是模型榜首。
      </Gate>
    </LabFrame>
  );
}
