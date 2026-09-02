"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import { numberFrom, round, stringFrom, useCompletionGate } from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson34CosmosLab.module.css";

type Mode = "engine" | "controller";
type Block = {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  visible: boolean;
};

const N_FRAMES = 8;
const DT = 1;
const G_TRUE = 0.085;
const TARGET_X = 0.5;
const ACTION_FRAME = 1;
const VANISH_ID = 2;
const VANISH_FRAME = 4;
const START: Block[] = [
  { id: 1, x: 0.22, y: 0.16, vx: 0, vy: 0, visible: true },
  { id: 2, x: 0.5, y: 0.16, vx: 0, vy: 0, visible: true },
  { id: 3, x: 0.78, y: 0.16, vx: 0, vy: 0, visible: true },
];

function clamp01(value: number) {
  return Math.max(0.06, Math.min(0.92, value));
}

function stepBlock(block: Block, gravity: number, ax: number): Block {
  const vx = block.vx + ax;
  const vy = block.vy + gravity * DT;
  return {
    ...block,
    vx,
    vy,
    x: clamp01(block.x + vx * DT),
    y: Math.min(0.84, block.y + vy * DT),
  };
}

function rollout(gravity: number, ax: number, vanish: boolean) {
  let state = START.map((block) => ({ ...block }));
  const frames: Block[][] = [state.map((block) => ({ ...block }))];
  for (let t = 0; t < N_FRAMES - 1; t += 1) {
    const impulse = t === ACTION_FRAME ? ax : 0;
    state = state.map((block) =>
      block.visible ? stepBlock(block, gravity, impulse) : block,
    );
    if (vanish && t + 1 >= VANISH_FRAME) {
      state = state.map((block) =>
        block.id === VANISH_ID ? { ...block, visible: false } : block,
      );
    }
    frames.push(state.map((block) => ({ ...block })));
  }
  return frames;
}

function missingCount(frames: Block[][]) {
  let total = 0;
  for (let t = 0; t < frames.length - 1; t += 1) {
    const prev = new Set(frames[t].filter((block) => block.visible).map((block) => block.id));
    const next = new Set(
      frames[t + 1].filter((block) => block.visible).map((block) => block.id),
    );
    prev.forEach((id) => {
      if (!next.has(id)) total += 1;
    });
  }
  return total;
}

function gravityAlarm(frames: Block[][], expectedSign: number) {
  const track = frames.map((frame) => frame.find((block) => block.id === 1 && block.visible));
  for (let t = 0; t < track.length - 1; t += 1) {
    const cur = track[t];
    const nxt = track[t + 1];
    if (!cur || !nxt) continue;
    const delta = nxt.vy - cur.vy;
    if (Math.abs(delta) < 1e-9) return true;
    const observed = delta > 0 ? 1 : -1;
    if (observed !== expectedSign) return true;
  }
  return false;
}

function catchError(frames: Block[][]) {
  const last = frames[frames.length - 1].find((block) => block.id === 1 && block.visible);
  if (!last) return 1;
  return Math.abs(last.x - TARGET_X);
}

export function Lesson34CosmosLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    action: numberFrom(initialState, "action", 0.08, -0.16, 0.16),
    frame: numberFrom(initialState, "frame", 0, 0, N_FRAMES - 1),
    mode: stringFrom(initialState, "mode", "engine") as Mode,
    prediction: stringFrom(initialState, "prediction", ""),
  };
  const [action, setAction] = useState(defaults.action);
  const [frame, setFrame] = useState(
    Math.min(N_FRAMES - 1, Math.max(0, Math.round(defaults.frame))),
  );
  const [mode, setMode] = useState<Mode>(
    defaults.mode === "controller" ? "controller" : "engine",
  );
  const [prediction, setPrediction] = useState(defaults.prediction);
  const [revealed, setRevealed] = useState(false);

  const calculation = useMemo(() => {
    const vanish = mode === "engine";
    const modelGravity = mode === "controller" ? -G_TRUE : G_TRUE;
    const cleanFrames = rollout(G_TRUE, action, false);
    const trueFrames = rollout(G_TRUE, action, vanish);
    const modelFrames = rollout(modelGravity, action, false);
    const missing = missingCount(trueFrames);
    const alarm = gravityAlarm(modelFrames, 1);
    const reported = catchError(modelFrames);
    const actual = catchError(cleanFrames);
    return {
      cleanFrames,
      trueFrames,
      modelFrames,
      missing,
      alarm,
      reported,
      actual,
      vanishFired: missing > 0,
      gravityFired: alarm,
    };
  }, [action, mode]);

  const passed =
    revealed &&
    prediction === "engine_vanish_controller_gravity" &&
    Math.abs(action) >= 0.04 &&
    (calculation.vanishFired || calculation.gravityFired);

  const completion = useMemo(
    () => ({
      lessonId: 34,
      mode,
      action: round(action, 3),
      frame,
      prediction,
      missing: calculation.missing,
      gravityAlarm: calculation.alarm,
      reportedCatch: round(calculation.reported, 4),
      actualCatch: round(calculation.actual, 4),
    }),
    [
      action,
      calculation.actual,
      calculation.alarm,
      calculation.missing,
      calculation.reported,
      frame,
      mode,
      prediction,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function reset() {
    setAction(0.08);
    setFrame(0);
    setMode("engine");
    setPrediction("");
    setRevealed(false);
  }

  const trueNow = (revealed ? calculation.trueFrames : calculation.cleanFrames)[frame];
  const predNow = (revealed ? calculation.modelFrames : calculation.cleanFrames)[frame];

  return (
    <LabFrame
      lesson="34"
      title="落下的方块：生成视频还是拿来控"
      description="给三个下落方块加水平动作。生成视频路会丢物体 ID，控制器路会把重力符号弄反。先预测哪条探针会响，再揭晓。数字来自教学模拟，不是模型输出。"
    >
      <div className={styles.workspace}>
        <form className={styles.controls} onSubmit={(event) => event.preventDefault()}>
          <h3>世界模型控制台</h3>
          <fieldset>
            <legend>用途</legend>
            <label>
              <input
                type="radio"
                name="wm-mode"
                checked={mode === "engine"}
                onChange={() => {
                  setMode("engine");
                  setRevealed(false);
                }}
              />
              <span>生成视频（数据引擎）</span>
            </label>
            <label>
              <input
                type="radio"
                name="wm-mode"
                checked={mode === "controller"}
                onChange={() => {
                  setMode("controller");
                  setRevealed(false);
                }}
              />
              <span>用作控制器</span>
            </label>
          </fieldset>
          <label>
            <span>
              水平动作 a_x
              <output>{action.toFixed(2)}</output>
            </span>
            <input
              type="range"
              min="-0.16"
              max="0.16"
              step="0.02"
              value={action}
              onChange={(event) => {
                setAction(Number(event.target.value));
                setRevealed(false);
              }}
            />
          </label>
          <label>
            <span>
              查看帧
              <output>
                {frame + 1}/{N_FRAMES}
              </output>
            </span>
            <input
              type="range"
              min="0"
              max={N_FRAMES - 1}
              step="1"
              value={frame}
              onChange={(event) => setFrame(Number(event.target.value))}
            />
          </label>
          <p className={styles.note}>
            夹具保证：生成路在动作之后丢掉 2 号物体；控制路的预测重力取反。把 |a_x| 拉到至少
            0.04。这些数字不能写成真机成功率。
          </p>
        </form>

        <div className={styles.stage}>
          <div className={styles.screens}>
            <figure className={styles.screenActive}>
              <figcaption>真实世界滚动</figcaption>
              <div className={styles.world}>
                <div className={styles.floor} />
                <i className={styles.target} style={{ left: `${TARGET_X * 100}%` }} />
                {trueNow.map((block) =>
                  block.visible ? (
                    <div
                      key={`true-${block.id}`}
                      className={styles.block}
                      style={{ left: `${block.x * 100}%`, top: `${block.y * 100}%` }}
                    >
                      {block.id}
                    </div>
                  ) : (
                    <span
                      key={`missing-${block.id}`}
                      className={styles.missing}
                      style={{ left: `${block.x * 100}%`, top: `${block.y * 100}%` }}
                    >
                      ID {block.id} 消失
                    </span>
                  ),
                )}
              </div>
            </figure>
            <figure className={styles.screen}>
              <figcaption>世界模型预测</figcaption>
              <div className={styles.world}>
                <div className={styles.floor} />
                <i className={styles.target} style={{ left: `${TARGET_X * 100}%` }} />
                {predNow.map((block) => (
                  <div
                    key={`pred-${block.id}`}
                    className={`${styles.block} ${styles.blockGhost}`}
                    style={{ left: `${block.x * 100}%`, top: `${block.y * 100}%` }}
                  >
                    {block.id}
                  </div>
                ))}
              </div>
            </figure>
          </div>
          <dl className={styles.metrics}>
            <div className={calculation.vanishFired && revealed ? styles.alarm : undefined}>
              <dt>丢失 ID 计数</dt>
              <dd>{revealed ? calculation.missing : "—"}</dd>
            </div>
            <div className={calculation.gravityFired && revealed ? styles.alarm : undefined}>
              <dt>重力符号报警</dt>
              <dd>{revealed ? (calculation.alarm ? "触发" : "未触发") : "—"}</dd>
            </div>
            <div>
              <dt>模型自评捕捉误差</dt>
              <dd>{revealed ? calculation.reported.toFixed(3) : "—"}</dd>
            </div>
            <div>
              <dt>真实世界捕捉误差</dt>
              <dd>{revealed ? calculation.actual.toFixed(3) : "—"}</dd>
            </div>
          </dl>
          <p className={styles.formula}>
            D = sum |I_t \ I_t+1|；自由下落要求 sign(Δv_y) = sign(g)。y 向下为正。
          </p>
        </div>
      </div>

      <div className={styles.predict}>
        <fieldset>
          <legend>先预测：给落下的方块加动作之后，哪句话成立？</legend>
          {[
            [
              "engine_vanish_controller_gravity",
              "生成视频路会丢物体 ID，控制器路会把重力符号弄反",
            ],
            ["both_clean", "两条路都服从重力，探针都不会响"],
            ["bigger_action_fixes", "加大水平动作就能同时修好永久性和重力"],
            ["controller_is_real", "控制器自评误差小，就可以写成真机抓取能力"],
          ].map(([value, label]) => (
            <label key={value}>
              <input
                type="radio"
                name="wm-prediction"
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
            揭晓探针
          </button>
        </div>
      </div>
      {revealed && prediction !== "engine_vanish_controller_gravity" && (
        <p className={styles.feedback}>
          夹具在生成路丢掉 2 号物体，在控制路把预测重力取反。加大 a_x
          只改变水平帧差，修不好永久性和重力符号。自评误差不是真机成功率。
        </p>
      )}
      <Gate passed={passed}>
        先选对“生成路丢 ID、控制路重力反号”，把 |a_x| 拉到至少 0.04，再揭晓并看到至少一条探针触发。教学模拟不能写成真机成功率。
      </Gate>
    </LabFrame>
  );
}
