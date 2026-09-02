"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import chrome from "./LabFrame.module.css";
import styles from "./Lesson13MemoryDrawers.module.css";
import type { AdvancedLabProps } from "./types";
import { numberFrom, pickFrom } from "./labUtils";

type Drawer = "working" | "episodic" | "semantic";
type ThemePred = "keep" | "drop";
type SeatPred = "fresh" | "stale";

const DRAWER_LABEL: Record<Drawer, string> = {
  working: "工作记忆",
  episodic: "情节记忆",
  semantic: "语义记忆",
};

const SEAT_EVENT = "小王工位改到5楼";
const THEME_TURN = "用户在问请假流程";

export function Lesson13MemoryDrawers({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    slots: numberFrom(initialState, "slots", 4, 2, 8),
    delay: numberFrom(initialState, "delay", 6, 0, 10),
    writeTarget: pickFrom(
      initialState,
      "writeTarget",
      ["working", "episodic", "semantic"] as const,
      "working",
    ),
  };
  const [slots, setSlots] = useState(defaults.slots);
  const [delay, setDelay] = useState(defaults.delay);
  const [writeTarget, setWriteTarget] = useState<Drawer>(defaults.writeTarget);
  const [themePred, setThemePred] = useState<ThemePred | null>(null);
  const [seatPred, setSeatPred] = useState<SeatPred | null>(null);
  const [hasRun, setHasRun] = useState(false);

  // workingMemory FIFO; episodicMemory append; semanticMemory key-overwrite
  const trace = useMemo(() => {
    let working = [THEME_TURN];
    const episodic = ["周二站会：讨论排期"];
    const semantic: Record<string, string> = {
      小王部门: "法务",
      小王工位: "3楼",
    };

    if (writeTarget === "working") working.push(SEAT_EVENT);
    if (writeTarget === "episodic") episodic.push(SEAT_EVENT);
    if (writeTarget === "semantic") semantic["小王工位"] = "5楼";

    for (let turn = 0; turn < delay; turn += 1) {
      working.push(`闲聊 ${turn + 1}`);
    }
    if (working.length > slots) {
      working = working.slice(working.length - slots);
    }

    const themeRecall = working.includes(THEME_TURN);
    const seatInWorking = working.includes(SEAT_EVENT);
    const seatInEpisodic = episodic.includes(SEAT_EVENT);
    const seatAnswer =
      seatInWorking || seatInEpisodic ? "5楼" : semantic["小王工位"];
    const seatFresh = seatAnswer === "5楼";

    return {
      working,
      episodic,
      semantic,
      themeRecall,
      seatAnswer,
      seatFresh,
    };
  }, [delay, slots, writeTarget]);

  const gatePassed =
    hasRun &&
    themePred === (trace.themeRecall ? "keep" : "drop") &&
    seatPred === (trace.seatFresh ? "fresh" : "stale");

  function invalidate() {
    setHasRun(false);
  }

  function run() {
    setHasRun(true);
    const passed =
      themePred === (trace.themeRecall ? "keep" : "drop") &&
      seatPred === (trace.seatFresh ? "fresh" : "stale");
    if (passed) {
      onComplete?.({
        slots,
        delay,
        writeTarget,
        themeRecall: trace.themeRecall,
        seatAnswer: trace.seatAnswer,
      });
    }
  }

  function reset() {
    setSlots(defaults.slots);
    setDelay(defaults.delay);
    setWriteTarget("working");
    setThemePred(null);
    setSeatPred(null);
    setHasRun(false);
  }

  return (
    <LabFrame
      lesson="13"
      title="三层抽屉：新信息写到哪里"
      description="工作记忆是当前对话窗口，满了会挤掉旧轮。情节记忆是带时间的日记。语义记忆是员工名录那种稳定键值。把「小王换座位」写入一层，隔若干轮再问，看召回成败。"
      onReset={reset}
    >
      <div className={chrome.workbench}>
        <div className={chrome.controls}>
          <label>
            <span>
              工作记忆容量 <strong>{slots} 轮</strong>
            </span>
            <input
              type="range"
              min="2"
              max="8"
              step="1"
              value={slots}
              onChange={(event) => {
                setSlots(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <label>
            <span>
              询问前再聊 <strong>{delay} 轮</strong>
            </span>
            <input
              type="range"
              min="0"
              max="10"
              step="1"
              value={delay}
              onChange={(event) => {
                setDelay(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <div className={chrome.field}>
            <span>把「小王工位改到 5 楼」写入</span>
            <div className={chrome.choiceRow}>
              {(Object.keys(DRAWER_LABEL) as Drawer[]).map((key) => (
                <button
                  type="button"
                  key={key}
                  aria-pressed={writeTarget === key}
                  onClick={() => {
                    setWriteTarget(key);
                    invalidate();
                  }}
                >
                  {DRAWER_LABEL[key]}
                </button>
              ))}
            </div>
          </div>
          <div className={chrome.formula}>
            <code>working = FIFO(last N turns)</code>
            <code>episodic.append(event)</code>
            <code>semantic[key] = value</code>
          </div>
        </div>

        <div className={chrome.panel}>
          <div className={chrome.metrics}>
            <div className={chrome.metric}>
              <span>请假主题</span>
              <strong>
                {hasRun ? (trace.themeRecall ? "召回" : "丢失") : "待运行"}
              </strong>
            </div>
            <div className={chrome.metric}>
              <span>小王坐哪</span>
              <strong>{hasRun ? trace.seatAnswer : "?"}</strong>
            </div>
            <div className={chrome.metric}>
              <span>写入层</span>
              <strong>{DRAWER_LABEL[writeTarget]}</strong>
            </div>
          </div>
          <div className={styles.drawers}>
            <article className={styles.drawer}>
              <h4>工作记忆</h4>
              {(hasRun ? trace.working : ["?"]).map((item) => (
                <p key={item}>{item}</p>
              ))}
            </article>
            <article className={styles.drawer}>
              <h4>情节记忆</h4>
              {(hasRun ? trace.episodic : ["?"]).map((item) => (
                <p key={item}>{item}</p>
              ))}
            </article>
            <article className={styles.drawer}>
              <h4>语义记忆</h4>
              {(hasRun
                ? Object.entries(trace.semantic).map(
                    ([key, value]) => `${key} = ${value}`,
                  )
                : ["?"]
              ).map((item) => (
                <p key={item}>{item}</p>
              ))}
            </article>
          </div>
          <p className={chrome.note}>
            语义层里原先记着「工位 = 3 楼」。只把新座位写进工作记忆，隔轮之后问「坐哪」，会读到过期名录。
          </p>
        </div>
      </div>

      <div className={chrome.prediction}>
        <fieldset>
          <legend>预测 1：隔这些轮后，「请假主题」还在工作记忆吗？</legend>
          <div className={chrome.choiceRow}>
            <button
              type="button"
              aria-pressed={themePred === "keep"}
              onClick={() => {
                setThemePred("keep");
                invalidate();
              }}
            >
              还在
            </button>
            <button
              type="button"
              aria-pressed={themePred === "drop"}
              onClick={() => {
                setThemePred("drop");
                invalidate();
              }}
            >
              被挤出
            </button>
          </div>
        </fieldset>
        <fieldset>
          <legend>预测 2：现在问「小王坐哪」，会得到哪一层的答案？</legend>
          <div className={chrome.choiceRow}>
            <button
              type="button"
              aria-pressed={seatPred === "fresh"}
              onClick={() => {
                setSeatPred("fresh");
                invalidate();
              }}
            >
              5 楼（新）
            </button>
            <button
              type="button"
              aria-pressed={seatPred === "stale"}
              onClick={() => {
                setSeatPred("stale");
                invalidate();
              }}
            >
              3 楼（过期）
            </button>
          </div>
        </fieldset>
        <button
          type="button"
          className={chrome.run}
          disabled={!themePred || !seatPred}
          onClick={run}
        >
          运行召回
        </button>
      </div>

      <Gate ran={hasRun} passed={gatePassed}>
        {!hasRun
          ? "先提交两项预测，再揭示三层抽屉里还剩什么。"
          : gatePassed
            ? `请假主题${trace.themeRecall ? "还在窗口里" : "已被挤出"}；座位答案是 ${trace.seatAnswer}。`
            : "有一项预测不符。工作记忆按容量 FIFO 丢轮；座位只有写进情节或语义（或还留在窗口里）才会变成 5 楼。"}
      </Gate>
    </LabFrame>
  );
}
