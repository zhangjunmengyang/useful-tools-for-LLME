"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import {
  numberFrom,
  stringFrom,
  useCompletionGate,
} from "./labUtils";
import type { AdvancedLabProps } from "./types";
import styles from "./Lesson42ScheduleLab.module.css";

type StageId = "T1" | "I" | "T2";
type Policy = "workspace" | "pollute";
type Role = "text" | "image_commit" | "image_inner";

type StageSpec = {
  id: StageId;
  kind: "text" | "image";
  label: string;
  nCommit: number;
};

type Position = {
  stage: StageId;
  rank: number;
  role: Role;
  inner: number | null;
  slot: number;
  label: string;
};

const LIBRARY: Record<StageId, StageSpec> = {
  T1: { id: "T1", kind: "text", label: "字A", nCommit: 3 },
  I: { id: "I", kind: "image", label: "图", nCommit: 4 },
  T2: { id: "T2", kind: "text", label: "字B", nCommit: 3 },
};

const DEFAULT_ORDER: StageId[] = ["T1", "I", "T2"];

function parseOrder(raw: string): StageId[] {
  const parts = raw.split("-") as StageId[];
  const allowed = new Set<StageId>(["T1", "I", "T2"]);
  if (parts.length !== 3 || parts.some((part) => !allowed.has(part))) {
    return DEFAULT_ORDER;
  }
  if (new Set(parts).size !== 3) return DEFAULT_ORDER;
  return parts;
}

function moveStage(order: StageId[], index: number, direction: -1 | 1) {
  const next = index + direction;
  if (next < 0 || next >= order.length) return order;
  const copy = order.slice();
  const current = copy[index];
  copy[index] = copy[next];
  copy[next] = current;
  return copy;
}

function expandPositions(
  order: StageId[],
  innerSteps: number,
  policy: Policy,
): Position[] {
  const positions: Position[] = [];
  order.forEach((id, rank) => {
    const stage = LIBRARY[id];
    if (stage.kind === "text") {
      for (let slot = 0; slot < stage.nCommit; slot += 1) {
        positions.push({
          stage: id,
          rank,
          role: "text",
          inner: null,
          slot,
          label: `${stage.label}${slot + 1}`,
        });
      }
      return;
    }
    if (policy === "pollute") {
      for (let step = 0; step < innerSteps; step += 1) {
        for (let slot = 0; slot < stage.nCommit; slot += 1) {
          positions.push({
            stage: id,
            rank,
            role: "image_inner",
            inner: step,
            slot,
            label: `噪${step + 1}.${slot + 1}`,
          });
        }
      }
    }
    for (let slot = 0; slot < stage.nCommit; slot += 1) {
      positions.push({
        stage: id,
        rank,
        role: "image_commit",
        inner: null,
        slot,
        label: `${stage.label}${slot + 1}`,
      });
    }
  });
  return positions;
}

function attentionMask(positions: Position[]) {
  const size = positions.length;
  const mask = Array.from({ length: size }, () =>
    Array.from({ length: size }, () => false),
  );
  for (let query = 0; query < size; query += 1) {
    for (let key = 0; key < size; key += 1) {
      const q = positions[query];
      const k = positions[key];
      const futureText = k.role === "text" && k.rank > q.rank;
      if (futureText) {
        mask[query][key] = false;
        continue;
      }
      const sameImage =
        q.role === "image_commit" &&
        k.role === "image_commit" &&
        q.stage === k.stage;
      mask[query][key] = sameImage || key <= query;
    }
  }
  return mask;
}

function leakCount(positions: Position[], mask: boolean[][]) {
  let leaks = 0;
  positions.forEach((query, queryIndex) => {
    if (query.role !== "text") return;
    positions.forEach((key, keyIndex) => {
      if (key.role === "image_inner" && mask[queryIndex][keyIndex]) {
        leaks += 1;
      }
    });
  });
  return leaks;
}

function chipClass(role: Role) {
  if (role === "text") return styles.chipText;
  if (role === "image_commit") return styles.chipImage;
  return styles.chipInner;
}

export function Lesson42ScheduleLab({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    order: parseOrder(stringFrom(initialState, "order", "T1-I-T2")),
    innerSteps: numberFrom(initialState, "innerSteps", 8, 4, 12),
    policy: stringFrom(initialState, "policy", "workspace") as Policy,
    prediction: stringFrom(initialState, "prediction", ""),
  };
  const [order, setOrder] = useState<StageId[]>(defaults.order);
  const [innerSteps, setInnerSteps] = useState(
    [4, 6, 8, 10, 12].includes(defaults.innerSteps) ? defaults.innerSteps : 8,
  );
  const [policy, setPolicy] = useState<Policy>(
    defaults.policy === "pollute" ? "pollute" : "workspace",
  );
  const [prediction, setPrediction] = useState(defaults.prediction);
  const [revealed, setRevealed] = useState(false);
  const [seenOrders, setSeenOrders] = useState<string[]>([]);

  const calculation = useMemo(() => {
    const positions = expandPositions(order, innerSteps, policy);
    const mask = attentionMask(positions);
    const leaks = leakCount(positions, mask);
    const outputOrder = order.map((id) => LIBRARY[id].label).join("|");
    const committed = positions.filter(
      (position) => position.role !== "image_inner",
    );
    const inners = positions.filter((position) => position.role === "image_inner");
    const imageIndex = committed.findIndex(
      (position) => position.role === "image_commit",
    );
    return {
      positions,
      mask,
      leaks,
      outputOrder,
      committed,
      inners,
      imageIndex,
      kvLen: positions.length,
      committedLen: committed.length,
    };
  }, [innerSteps, order, policy]);

  const isTextImageText =
    order[0] === "T1" && order[1] === "I" && order[2] === "T2";
  const passed =
    revealed &&
    prediction === "workspace" &&
    policy === "workspace" &&
    calculation.leaks === 0 &&
    isTextImageText &&
    seenOrders.length >= 2;

  const completion = useMemo(
    () => ({
      lessonId: 42,
      order: order.join("-"),
      innerSteps,
      policy,
      leaks: calculation.leaks,
      kvLen: calculation.kvLen,
      outputOrder: calculation.outputOrder,
      seenOrders,
    }),
    [
      calculation.kvLen,
      calculation.leaks,
      calculation.outputOrder,
      innerSteps,
      order,
      policy,
      seenOrders,
    ],
  );
  useCompletionGate(passed, onComplete, completion);

  function invalidateReveal() {
    setRevealed(false);
  }

  function reset() {
    setOrder(DEFAULT_ORDER);
    setInnerSteps(8);
    setPolicy("workspace");
    setPrediction("");
    setRevealed(false);
    setSeenOrders([]);
  }

  function reveal() {
    if (!prediction) return;
    setRevealed(true);
    setSeenOrders((current) =>
      current.includes(calculation.outputOrder)
        ? current
        : [...current, calculation.outputOrder],
    );
  }

  const maskSize = Math.min(calculation.positions.length, 16);
  const showNumbers = revealed;

  return (
    <LabFrame
      lesson="42"
      title="排出一段字-图-字日程"
      description="教学模拟，不是模型输出。先预测图像内步会不会写入文本 KV，再揭晓泄漏计数；调换三个阶段后，输出顺序必须改变。"
    >
      <div className={styles.workspace}>
        <form
          className={styles.controls}
          onSubmit={(event) => event.preventDefault()}
        >
          <h3>日程台</h3>
          <div className={styles.stageList} aria-label="阶段顺序">
            {order.map((id, index) => (
              <div className={styles.stageRow} key={`${id}-${index}`}>
                <b>{index + 1}</b>
                <span>
                  {LIBRARY[id].label} · 提交 {LIBRARY[id].nCommit}
                </span>
                <div className={styles.mover}>
                  <button
                    type="button"
                    aria-label={`上移 ${LIBRARY[id].label}`}
                    disabled={index === 0}
                    onClick={() => {
                      setOrder(moveStage(order, index, -1));
                      invalidateReveal();
                    }}
                  >
                    上
                  </button>
                  <button
                    type="button"
                    aria-label={`下移 ${LIBRARY[id].label}`}
                    disabled={index === order.length - 1}
                    onClick={() => {
                      setOrder(moveStage(order, index, 1));
                      invalidateReveal();
                    }}
                  >
                    下
                  </button>
                </div>
              </div>
            ))}
          </div>
          <label>
            <span>
              图像内步 T <output>{innerSteps}</output>
            </span>
            <input
              type="range"
              min="4"
              max="12"
              step="2"
              value={innerSteps}
              onChange={(event) => {
                setInnerSteps(Number(event.target.value));
                invalidateReveal();
              }}
            />
          </label>
          <fieldset>
            <legend>写入策略</legend>
            <label>
              <input
                type="radio"
                name="policy"
                checked={policy === "workspace"}
                onChange={() => {
                  setPolicy("workspace");
                  invalidateReveal();
                }}
              />
              工作区：内步不进文本 KV
            </label>
            <label>
              <input
                type="radio"
                name="policy"
                checked={policy === "pollute"}
                onChange={() => {
                  setPolicy("pollute");
                  invalidateReveal();
                }}
              />
              错误：扩散步写入因果位置
            </label>
          </fieldset>
          <p className={styles.note}>
            必须先排一次字A-图-字B，再至少调换一次顺序并重新揭晓。泄漏计数在揭晓前不显示。
          </p>
        </form>

        <div className={styles.stage}>
          <div className={styles.formula}>
            <span>
              S = {order.map((id) => LIBRARY[id].label).join(" / ")}
            </span>
            <span>
              L_KV = Σ n
              {policy === "pollute" ? " + n_img·T" : ""}
            </span>
            <strong>
              {showNumbers
                ? `leak = ${calculation.leaks}`
                : "leak = 先预测再揭晓"}
            </strong>
          </div>

          <div className={styles.lanes}>
            <section className={styles.lane}>
              <header>
                <span>已提交前缀</span>
                <span>
                  {showNumbers
                    ? `${calculation.committedLen} 槽`
                    : "揭晓后显示"}
                </span>
              </header>
              {showNumbers ? (
                <div className={styles.chips}>
                  {calculation.committed.map((position, index) => (
                    <span
                      className={`${styles.chip} ${chipClass(position.role)}`}
                      key={`c-${index}`}
                    >
                      {position.label}
                    </span>
                  ))}
                </div>
              ) : (
                <p className={styles.hiddenValue}>提交序列已隐藏</p>
              )}
            </section>
            <section className={styles.lane}>
              <header>
                <span>图像工作区 / 误写入的内步</span>
                <span>
                  {showNumbers
                    ? policy === "workspace"
                      ? `${innerSteps} 步仅在工作区`
                      : `${calculation.inners.length} 个噪声槽进了 KV`
                    : "揭晓后显示"}
                </span>
              </header>
              {showNumbers ? (
                <div className={styles.chips}>
                  {policy === "workspace" ? (
                    Array.from({ length: innerSteps }, (_, step) => (
                      <span
                        className={`${styles.chip} ${styles.chipInner}`}
                        key={`w-${step}`}
                      >
                        t={step + 1}
                      </span>
                    ))
                  ) : calculation.inners.length > 0 ? (
                    calculation.inners.slice(0, 16).map((position, index) => (
                      <span
                        className={`${styles.chip} ${chipClass(position.role)}`}
                        key={`i-${index}`}
                      >
                        {position.label}
                      </span>
                    ))
                  ) : (
                    <p className={styles.hiddenValue}>无内步</p>
                  )}
                </div>
              ) : (
                <p className={styles.hiddenValue}>内步可见性已隐藏</p>
              )}
            </section>
            <section className={styles.lane}>
              <header>
                <span>注意力 mask（前 {maskSize} 槽）</span>
                <span>绿可见 / 红为文本读到噪声步</span>
              </header>
              {showNumbers ? (
                <div
                  className={styles.mask}
                  role="img"
                  aria-label="注意力可见性矩阵"
                >
                  {calculation.mask.slice(0, maskSize).map((row, query) => (
                    <div
                      className={styles.maskRow}
                      key={`r-${query}`}
                      style={{
                        gridTemplateColumns: `repeat(${maskSize}, 0.72rem)`,
                      }}
                    >
                      {row.slice(0, maskSize).map((on, key) => {
                        const leak =
                          on &&
                          calculation.positions[query]?.role === "text" &&
                          calculation.positions[key]?.role === "image_inner";
                        return (
                          <span
                            key={`c-${query}-${key}`}
                            className={`${styles.cell} ${
                              leak
                                ? styles.cellLeak
                                : on
                                  ? styles.cellOn
                                  : ""
                            }`}
                          />
                        );
                      })}
                    </div>
                  ))}
                </div>
              ) : (
                <p className={styles.hiddenValue}>mask 在揭晓前不绘制</p>
              )}
            </section>
          </div>

          <dl className={styles.metrics}>
            <div>
              <dt>KV 槽</dt>
              <dd>{showNumbers ? calculation.kvLen : "—"}</dd>
            </div>
            <div>
              <dt>泄漏</dt>
              <dd>{showNumbers ? calculation.leaks : "—"}</dd>
            </div>
            <div>
              <dt>图提交下标</dt>
              <dd>{showNumbers ? calculation.imageIndex : "—"}</dd>
            </div>
            <div>
              <dt>已见顺序</dt>
              <dd>{seenOrders.length}</dd>
            </div>
          </dl>

          <div className={styles.predict}>
            <fieldset>
              <legend>先预测：图像采样步写进文本 KV 会怎样？</legend>
              <label>
                <input
                  type="radio"
                  name="prediction"
                  checked={prediction === "harmless"}
                  onChange={() => {
                    setPrediction("harmless");
                    invalidateReveal();
                  }}
                />
                <span>写进去也没关系，后面的字仍只读干净图</span>
              </label>
              <label>
                <input
                  type="radio"
                  name="prediction"
                  checked={prediction === "workspace"}
                  onChange={() => {
                    setPrediction("workspace");
                    invalidateReveal();
                  }}
                />
                <span>内步必须停在工作区；调换日程会改变输出顺序</span>
              </label>
              <label>
                <input
                  type="radio"
                  name="prediction"
                  checked={prediction === "invariant"}
                  onChange={() => {
                    setPrediction("invariant");
                    invalidateReveal();
                  }}
                />
                <span>三种排法最后都会写成同一条 KV，顺序不变</span>
              </label>
            </fieldset>
            <div className={styles.actions}>
              <button type="button" className={styles.reset} onClick={reset}>
                重置
              </button>
              <button
                type="button"
                className={styles.run}
                disabled={!prediction}
                onClick={reveal}
              >
                揭晓 mask
              </button>
            </div>
          </div>
          {!prediction ? (
            <p className={styles.feedback}>先选预测，才能看到泄漏计数和输出顺序。</p>
          ) : null}
          {revealed && prediction !== "workspace" ? (
            <p className={styles.feedback}>
              预测未打中。内步若进入因果位置，后续文本会读到噪声槽；调换日程也会改变提交顺序。
            </p>
          ) : null}
          {revealed && prediction === "workspace" && policy === "pollute" ? (
            <p className={styles.feedback}>
              当前是错误写入。切回工作区，泄漏必须降到 0。
            </p>
          ) : null}
          {revealed &&
          prediction === "workspace" &&
          policy === "workspace" &&
          seenOrders.length < 2 ? (
            <p className={styles.feedback}>
              泄漏已是 0。请调换阶段顺序再揭晓一次，输出顺序必须与现在不同。
            </p>
          ) : null}
          {revealed &&
          prediction === "workspace" &&
          policy === "workspace" &&
          seenOrders.length >= 2 &&
          !isTextImageText ? (
            <p className={styles.feedback}>
              顺序已经变过。请排回字A-图-字B 再揭晓，完成本课指定日程。
            </p>
          ) : null}
        </div>
      </div>
      <Gate passed={passed}>
        {passed
          ? "字-图-字日程下图像内步未写入文本 KV，且至少两种日程的输出顺序不同。"
          : "完成：先预测，工作区泄漏为 0，排过字-图-字，并调换日程使输出顺序改变。"}
      </Gate>
    </LabFrame>
  );
}
