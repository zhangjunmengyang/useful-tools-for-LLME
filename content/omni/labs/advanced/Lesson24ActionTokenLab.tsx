"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import { numberFrom, stringFrom, useCompletionGate } from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson24ActionTokenLab.module.css";

type SceneId = "unique" | "multi";
type InstructionId = "pick_cup" | "pick_apple" | "pick_bowl" | "empty";
type PathwayId = "text" | "discrete" | "skill";
type ObjectId = "cup" | "apple" | "bowl";
type PredictionId = "suite" | "always-language" | "always-vision";

type LabObject = {
  id: ObjectId;
  label: string;
  x: number;
  y: number;
};

const OBJECTS: Record<ObjectId, LabObject> = {
  cup: { id: "cup", label: "杯", x: 0.28, y: 0.42 },
  apple: { id: "apple", label: "苹", x: 0.72, y: 0.38 },
  bowl: { id: "bowl", label: "碗", x: 0.5, y: 0.64 },
};

const DIMS: Array<{ name: string; low: number; high: number }> = [
  { name: "x", low: -1, high: 1 },
  { name: "y", low: -1, high: 1 },
  { name: "z", low: -1, high: 1 },
  { name: "roll", low: -1, high: 1 },
  { name: "pitch", low: -1, high: 1 },
  { name: "yaw", low: -1, high: 1 },
  { name: "grip", low: 0, high: 1 },
];

const BINS = 8;
const TEXT_VOCAB = 32;
const IDLE = [0, 0, 0, 0, 0, 0, 1];

const INSTRUCTION_LABEL: Record<InstructionId, string> = {
  pick_cup: "抓住杯子",
  pick_apple: "抓住苹果",
  pick_bowl: "抓住碗",
  empty: "（空指令）",
};

const PATHWAY_LABEL: Record<PathwayId, string> = {
  text: "只生成文字",
  discrete: "离散动作 token",
  skill: "文字映射技能",
};

function clipBin(index: number) {
  return Math.max(0, Math.min(BINS - 1, index));
}

function uniformBin(value: number, low: number, high: number) {
  if (value <= low) return 0;
  if (value >= high) return BINS - 1;
  const width = (high - low) / BINS;
  return clipBin(Math.floor((value - low) / width));
}

function encodeAction(action: number[]) {
  return action.map((value, dimension) => {
    const spec = DIMS[dimension];
    return TEXT_VOCAB + dimension * BINS + uniformBin(value, spec.low, spec.high);
  });
}

function actionToward(objectId: ObjectId | null) {
  if (!objectId) return IDLE.slice();
  const object = OBJECTS[objectId];
  return [
    (object.x - 0.5) * 2,
    (object.y - 0.5) * 2,
    -0.62,
    0,
    0.12,
    0,
    0.12,
  ];
}

function sceneObjects(scene: SceneId): ObjectId[] {
  return scene === "unique" ? ["cup"] : ["cup", "apple", "bowl"];
}

function wantedObject(instruction: InstructionId): ObjectId | null {
  if (instruction === "pick_cup") return "cup";
  if (instruction === "pick_apple") return "apple";
  if (instruction === "pick_bowl") return "bowl";
  return null;
}

function tokensEqual(left: number[], right: number[]) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function skillScores(
  instruction: InstructionId,
  visible: ObjectId[],
  languageOn: boolean,
) {
  const llm: Record<string, number> = languageOn
    ? {
        pick_cup:
          instruction === "pick_cup" ? 0.82 : instruction === "empty" ? 0.25 : 0.07,
        pick_apple:
          instruction === "pick_apple" ? 0.82 : instruction === "empty" ? 0.25 : 0.07,
        pick_bowl:
          instruction === "pick_bowl" ? 0.82 : instruction === "empty" ? 0.25 : 0.07,
        idle: instruction === "empty" ? 0.25 : 0.07,
      }
    : {
        pick_cup: 0.25,
        pick_apple: 0.25,
        pick_bowl: 0.25,
        idle: 0.25,
      };
  const affordance = {
    pick_cup: visible.includes("cup") ? 0.92 : 0.04,
    pick_apple: visible.includes("apple") ? 0.92 : 0.04,
    pick_bowl: visible.includes("bowl") ? 0.92 : 0.04,
    idle: 0.55,
  };
  const combined = {
    pick_cup: llm.pick_cup * affordance.pick_cup,
    pick_apple: llm.pick_apple * affordance.pick_apple,
    pick_bowl: llm.pick_bowl * affordance.pick_bowl,
    idle: llm.idle * affordance.idle,
  };
  const ranked = Object.entries(combined).sort((a, b) => b[1] - a[1]);
  return { llm, affordance, combined, winner: ranked[0][0] };
}

function resolveDiscreteTarget(
  scene: SceneId,
  instruction: InstructionId,
  visionOn: boolean,
  languageOn: boolean,
): ObjectId | null {
  if (!visionOn) return null;
  const visible = sceneObjects(scene);
  if (visible.length === 1) return visible[0];
  if (!languageOn) return null;
  const wanted = wantedObject(instruction);
  if (wanted && visible.includes(wanted)) return wanted;
  return null;
}

function resolveSkillTarget(
  scene: SceneId,
  instruction: InstructionId,
  visionOn: boolean,
  languageOn: boolean,
): ObjectId | null {
  const visible = visionOn ? sceneObjects(scene) : [];
  if (!languageOn && visible.length > 1) return null;
  const scores = skillScores(instruction, visible, languageOn);
  if (scores.winner === "pick_cup") return "cup";
  if (scores.winner === "pick_apple") return "apple";
  if (scores.winner === "pick_bowl") return "bowl";
  return null;
}

function textOutput(
  scene: SceneId,
  instruction: InstructionId,
  visionOn: boolean,
  languageOn: boolean,
) {
  const visible = visionOn ? sceneObjects(scene) : [];
  const seen =
    visible.length === 0
      ? "图像被切断，看不见物体"
      : `看见 ${visible.map((id) => OBJECTS[id].label).join("、")}`;
  if (!languageOn) {
    return `${seen}。没有指令，只生成空描述。`;
  }
  const wanted = wantedObject(instruction);
  if (!wanted) return `${seen}。指令为空，输出「等待」。`;
  if (!visionOn) {
    return `按字面写「${INSTRUCTION_LABEL[instruction]}」，但没有视觉坐标。`;
  }
  if (visible.includes(wanted)) {
    return `${seen}。文字通路输出「${INSTRUCTION_LABEL[instruction]}」。`;
  }
  return `${seen}。文字仍写「${INSTRUCTION_LABEL[instruction]}」，目标不在视野里。`;
}

function simulate(options: {
  scene: SceneId;
  instruction: InstructionId;
  pathway: PathwayId;
  visionOn: boolean;
  languageOn: boolean;
}) {
  const { scene, instruction, pathway, visionOn, languageOn } = options;
  const visible = visionOn ? sceneObjects(scene) : [];
  const discreteTarget = resolveDiscreteTarget(
    scene,
    instruction,
    visionOn,
    languageOn,
  );
  const skillTarget = resolveSkillTarget(scene, instruction, visionOn, languageOn);
  const motorTarget =
    pathway === "text"
      ? null
      : pathway === "discrete"
        ? discreteTarget
        : skillTarget;
  const action = pathway === "text" ? IDLE.slice() : actionToward(motorTarget);
  const tokens = encodeAction(action);
  const scores = skillScores(instruction, visible, languageOn);
  const cupTokens = encodeAction(actionToward("cup"));
  const appleTokens = encodeAction(actionToward("apple"));
  return {
    visible,
    discreteTarget,
    skillTarget,
    motorTarget,
    action,
    tokens,
    scores,
    text: textOutput(scene, instruction, visionOn, languageOn),
    matchesCup: tokensEqual(tokens, cupTokens),
    matchesApple: tokensEqual(tokens, appleTokens),
    isIdle: tokensEqual(tokens, encodeAction(IDLE)),
  };
}

export function Lesson24ActionTokenLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    scene: stringFrom(initialState, "scene", "unique") as SceneId,
    instruction: stringFrom(
      initialState,
      "instruction",
      "pick_cup",
    ) as InstructionId,
    pathway: stringFrom(initialState, "pathway", "discrete") as PathwayId,
    vision: numberFrom(initialState, "vision", 1, 0, 1),
    language: numberFrom(initialState, "language", 1, 0, 1),
    prediction: stringFrom(initialState, "prediction", "") as PredictionId | "",
  };
  const [scene, setScene] = useState<SceneId>(
    defaults.scene === "multi" ? "multi" : "unique",
  );
  const [instruction, setInstruction] = useState<InstructionId>(
    defaults.instruction in INSTRUCTION_LABEL
      ? defaults.instruction
      : "pick_cup",
  );
  const [pathway, setPathway] = useState<PathwayId>(
    defaults.pathway in PATHWAY_LABEL ? defaults.pathway : "discrete",
  );
  const [visionOn, setVisionOn] = useState(defaults.vision === 1);
  const [languageOn, setLanguageOn] = useState(defaults.language === 1);
  const [prediction, setPrediction] = useState<PredictionId | "">(
    defaults.prediction === "suite" ||
      defaults.prediction === "always-language" ||
      defaults.prediction === "always-vision"
      ? defaults.prediction
      : "",
  );
  const [ran, setRan] = useState(false);
  const [sawUniqueWrong, setSawUniqueWrong] = useState(false);
  const [sawMultiWrong, setSawMultiWrong] = useState(false);

  const uniqueWrong = useMemo(
    () =>
      simulate({
        scene: "unique",
        instruction: "pick_apple",
        pathway: "discrete",
        visionOn: true,
        languageOn: true,
      }),
    [],
  );
  const uniqueCorrect = useMemo(
    () =>
      simulate({
        scene: "unique",
        instruction: "pick_cup",
        pathway: "discrete",
        visionOn: true,
        languageOn: true,
      }),
    [],
  );
  const multiWrong = useMemo(
    () =>
      simulate({
        scene: "multi",
        instruction: "pick_apple",
        pathway: "discrete",
        visionOn: true,
        languageOn: true,
      }),
    [],
  );
  const multiCorrect = useMemo(
    () =>
      simulate({
        scene: "multi",
        instruction: "pick_cup",
        pathway: "discrete",
        visionOn: true,
        languageOn: true,
      }),
    [],
  );

  const uniqueUnchanged = tokensEqual(uniqueWrong.tokens, uniqueCorrect.tokens);
  const multiChanged = !tokensEqual(multiWrong.tokens, multiCorrect.tokens);

  const current = useMemo(
    () =>
      simulate({
        scene,
        instruction,
        pathway,
        visionOn,
        languageOn,
      }),
    [instruction, languageOn, pathway, scene, visionOn],
  );

  const passed =
    ran &&
    prediction === "suite" &&
    sawUniqueWrong &&
    sawMultiWrong &&
    uniqueUnchanged &&
    multiChanged;

  const completion = useMemo(
    () => ({
      lessonId: 24,
      scene,
      instruction,
      pathway,
      visionOn,
      languageOn,
      prediction,
      uniqueUnchanged,
      multiChanged,
      uniqueWrongTokens: uniqueWrong.tokens,
      multiWrongTokens: multiWrong.tokens,
    }),
    [
      instruction,
      languageOn,
      multiChanged,
      multiWrong.tokens,
      pathway,
      prediction,
      scene,
      uniqueUnchanged,
      uniqueWrong.tokens,
      visionOn,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function markObservation() {
    if (
      scene === "unique" &&
      instruction === "pick_apple" &&
      pathway === "discrete" &&
      visionOn &&
      languageOn
    ) {
      setSawUniqueWrong(true);
    }
    if (
      scene === "multi" &&
      instruction === "pick_apple" &&
      pathway === "discrete" &&
      visionOn &&
      languageOn
    ) {
      setSawMultiWrong(true);
    }
  }

  function runCurrent() {
    setRan(true);
    markObservation();
  }

  function runProbe(kind: "unique" | "multi") {
    setPathway("discrete");
    setVisionOn(true);
    setLanguageOn(true);
    setInstruction("pick_apple");
    if (kind === "unique") {
      setScene("unique");
      setSawUniqueWrong(true);
    } else {
      setScene("multi");
      setSawMultiWrong(true);
    }
    setRan(true);
  }

  function reset() {
    setScene("unique");
    setInstruction("pick_cup");
    setPathway("discrete");
    setVisionOn(true);
    setLanguageOn(true);
    setPrediction("");
    setRan(false);
    setSawUniqueWrong(false);
    setSawMultiWrong(false);
  }

  const objects = sceneObjects(scene);

  return (
    <LabFrame
      lesson="24"
      title="三条通路上的动作 token"
      description="教学模拟，不是模型输出。同一视觉输入可走只文字、离散动作 token、或文字映射技能。先预测视觉捷径，再揭晓 token。"
    >
      <div className={styles.workspace}>
        <form
          className={styles.controls}
          onSubmit={(event) => event.preventDefault()}
        >
          <h3>控制台</h3>
          <label>
            <span>场景</span>
            <select
              value={scene}
              onChange={(event) => {
                setScene(event.target.value as SceneId);
                setRan(false);
              }}
            >
              <option value="unique">唯一物体（只有杯）</option>
              <option value="multi">多目标（杯 / 苹 / 碗）</option>
            </select>
          </label>
          <label>
            <span>指令</span>
            <select
              value={instruction}
              onChange={(event) => {
                setInstruction(event.target.value as InstructionId);
                setRan(false);
              }}
            >
              <option value="pick_cup">抓住杯子</option>
              <option value="pick_apple">抓住苹果（错或另一目标）</option>
              <option value="pick_bowl">抓住碗</option>
              <option value="empty">空指令</option>
            </select>
          </label>
          <label>
            <span>输出通路</span>
            <select
              value={pathway}
              onChange={(event) => {
                setPathway(event.target.value as PathwayId);
                setRan(false);
              }}
            >
              <option value="text">只生成文字</option>
              <option value="discrete">离散动作 token</option>
              <option value="skill">文字映射技能</option>
            </select>
          </label>
          <fieldset>
            <legend>切断模态</legend>
            <label className={styles.toggle}>
              <input
                type="checkbox"
                checked={visionOn}
                onChange={(event) => {
                  setVisionOn(event.target.checked);
                  setRan(false);
                }}
              />
              <span>视觉接通</span>
            </label>
            <label className={styles.toggle}>
              <input
                type="checkbox"
                checked={languageOn}
                onChange={(event) => {
                  setLanguageOn(event.target.checked);
                  setRan(false);
                }}
              />
              <span>语言接通</span>
            </label>
          </fieldset>
        </form>

        <div className={styles.stage}>
          <div className={styles.tableWrap}>
            <div
              className={`${styles.table} ${visionOn ? "" : styles.tableBlind}`}
              aria-label="桌面俯视图"
            >
              <span className={styles.tableHint}>
                {visionOn ? "俯视桌面" : "视觉已切断"}
              </span>
              {objects.map((id) => {
                const object = OBJECTS[id];
                const targeted =
                  ran && current.motorTarget === id && pathway !== "text";
                return (
                  <b
                    key={id}
                    className={`${styles.object} ${targeted ? styles.objectOn : ""}`}
                    style={{ left: `${object.x * 100}%`, top: `${object.y * 100}%` }}
                  >
                    {object.label}
                  </b>
                );
              })}
            </div>
            <p className={styles.caption}>
              通路 {PATHWAY_LABEL[pathway]} · 指令{" "}
              {languageOn ? INSTRUCTION_LABEL[instruction] : "已切断"}
            </p>
          </div>

          <dl className={styles.metrics}>
            <div>
              <dt>动作 token</dt>
              <dd>{ran ? current.tokens.join(" ") : "—"}</dd>
            </div>
            <div>
              <dt>电机目标</dt>
              <dd>
                {ran
                  ? pathway === "text"
                    ? "无（文字通路）"
                    : current.motorTarget
                      ? OBJECTS[current.motorTarget].label
                      : "idle"
                  : "—"}
              </dd>
            </div>
            <div>
              <dt>与杯程序相同</dt>
              <dd>
                {ran ? (current.matchesCup ? "是" : "否") : "—"}
              </dd>
            </div>
          </dl>

          {ran && (
            <div className={styles.reveal}>
              <p>
                <strong>文字头：</strong>
                {current.text}
              </p>
              <p>
                <strong>技能乘积：</strong>
                cup {current.scores.combined.pick_cup.toFixed(3)} · apple{" "}
                {current.scores.combined.pick_apple.toFixed(3)} · bowl{" "}
                {current.scores.combined.pick_bowl.toFixed(3)} · idle{" "}
                {current.scores.combined.idle.toFixed(3)}
              </p>
              <p>
                <strong>7 维连续动作：</strong>
                {current.action.map((value) => value.toFixed(2)).join(" , ")}
              </p>
            </div>
          )}
        </div>
      </div>

      <div className={styles.predict}>
        <fieldset>
          <legend>
            先预测：离散动作通路上，把指令从「抓住杯子」改成「抓住苹果」会发生什么？
          </legend>
          {[
            [
              "suite",
              "唯一物体场景动作不变，多目标同场景必须改动作",
            ],
            ["always-language", "两种场景都会跟着错指令改动作"],
            ["always-vision", "两种场景都忽略指令，动作始终抓杯"],
          ].map(([value, label]) => (
            <label key={value}>
              <input
                type="radio"
                name="vla-prediction"
                value={value}
                checked={prediction === value}
                onChange={() => {
                  setPrediction(value as PredictionId);
                  setRan(false);
                  setSawUniqueWrong(false);
                  setSawMultiWrong(false);
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
            onClick={runCurrent}
          >
            揭晓当前设置
          </button>
          <button
            type="button"
            className={styles.run}
            disabled={!prediction}
            onClick={() => runProbe("unique")}
          >
            揭晓唯一场景错指令
          </button>
          <button
            type="button"
            className={styles.run}
            disabled={!prediction}
            onClick={() => runProbe("multi")}
          >
            揭晓多目标错指令
          </button>
        </div>
      </div>

      {ran && prediction && prediction !== "suite" && (
        <p className={styles.feedback}>
          对应 2603.19233 的定性结论：语言是否起作用取决于任务结构。场景已经唯一决定任务时，视觉通路可以不听指令；同场景多个目标时，错指令必须改动作。
        </p>
      )}
      {ran && sawUniqueWrong && sawMultiWrong && (
        <p className={styles.feedback}>
          唯一场景错指令 token {uniqueWrong.tokens.join(" ")}
          {uniqueUnchanged ? " 与正确指令相同" : " 与正确指令不同"}
          ；多目标错指令 token {multiWrong.tokens.join(" ")}
          {multiChanged ? " 已离开杯子程序" : " 仍停在杯子程序"}。
        </p>
      )}
      <Gate passed={passed}>
        先提交正确预测，再分别揭晓唯一场景与多目标场景的错指令。文字通路始终无电机目标；切断视觉后离散通路应变为 idle。
      </Gate>
    </LabFrame>
  );
}
