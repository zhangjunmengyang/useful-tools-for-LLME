"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import chrome from "./LabFrame.module.css";
import styles from "./Lesson24FourteenDays.module.css";
import type { AdvancedLabProps } from "./types";
import { numberFrom, pickFrom, polylinePoints } from "./labUtils";

type Mode = "freeze" | "rag" | "skills" | "weights";

const MODE_LABEL: Record<Mode, string> = {
  freeze: "冻结",
  rag: "只塞当天手册",
  skills: "记忆+技能",
  weights: "再加权重",
};

type DayRow = {
  day: number;
  title: string;
  mem: number;
  skills: number;
  seat: number;
  rule: number;
};

const DAY_TITLE = [
  "叫小王",
  "读发布手册",
  "跑发布脚本",
  "站会笔记",
  "回放叫人",
  "项目在哪",
  "座位换到 B7",
  "夜间同步出现",
  "计分改成客户损失",
  "再用发布脚本",
  "叫小李",
  "回放叫小王",
  "一张新工单怎么计分",
  "周五复盘",
];

function simulate(mode: Mode): DayRow[] {
  let seat: string | null = null;
  let skills = 0;
  let tickets = 0;
  let ruleFitted = false;
  const rows: DayRow[] = [];
  for (let day = 1; day <= 14; day += 1) {
    const worldSeat = day >= 7 ? "B7" : "A3";
    const writesMemory = mode === "skills" || mode === "weights";
    const writesSkills = writesMemory;
    const writesWeights = mode === "weights";
    const seatHit = writesMemory && seat === worldSeat ? 1 : 0;
    if (writesMemory) seat = worldSeat;
    if (writesSkills) skills = day >= 8 ? 2 : Math.max(skills, 1);
    if (writesWeights && day >= 9) {
      tickets += 1;
      ruleFitted = tickets >= 3;
    }
    rows.push({
      day,
      title: DAY_TITLE[day - 1],
      mem: writesMemory ? (seat ? 1 : 0) : 0,
      skills: writesSkills ? skills : 0,
      seat: seatHit,
      rule: writesWeights && ruleFitted ? 1 : 0,
    });
  }
  return rows;
}

export function Lesson24FourteenDays({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const [mode, setMode] = useState<Mode>(
    pickFrom(
      initialState,
      "mode",
      ["freeze", "rag", "skills", "weights"] as const,
      "freeze",
    ),
  );
  const [visible, setVisible] = useState(
    numberFrom(initialState, "visible", 14, 1, 14),
  );
  const [seatPred, setSeatPred] = useState<"freeze" | "semantic" | null>(null);
  const [rulePred, setRulePred] = useState<"memory" | "weights" | null>(null);
  const [hasRun, setHasRun] = useState(false);

  const all = useMemo(
    () => ({
      freeze: simulate("freeze"),
      rag: simulate("rag"),
      skills: simulate("skills"),
      weights: simulate("weights"),
    }),
    [],
  );
  const series = all[mode];
  const last = series[series.length - 1];
  const finals = {
    freeze: all.freeze[13],
    rag: all.rag[13],
    skills: all.skills[13],
    weights: all.weights[13],
  };
  const gatePassed =
    hasRun && seatPred === "semantic" && rulePred === "weights";

  function invalidate() {
    setHasRun(false);
  }

  function run() {
    setHasRun(true);
    if (seatPred === "semantic" && rulePred === "weights") {
      onComplete?.({
        mode,
        seat: last.seat,
        rule: last.rule,
        freezeSeat: finals.freeze.seat,
        skillsSeat: finals.skills.seat,
        weightsRule: finals.weights.rule,
      });
    }
  }

  function reset() {
    setMode("freeze");
    setVisible(14);
    setSeatPred(null);
    setRulePred(null);
    setHasRun(false);
  }

  const seatPts = polylinePoints(
    series.map((row) => row.seat),
    280,
    88,
    1,
  );
  const skillPts = polylinePoints(
    series.map((row) => row.skills),
    280,
    88,
    2,
  );
  const rulePts = polylinePoints(
    series.map((row) => row.rule),
    280,
    88,
    1,
  );
  const shown = hasRun ? series.slice(0, visible) : [];

  return (
    <LabFrame
      lesson="24"
      title="14 日看板：用当前世界回放"
      description="保持率按今天的世界来量：第 7 日小王换到 B7 之后，冻结通道叫不到人。计分规则第 9 日改成客户损失，只有再加权重能在没有当日手册时算对。"
      onReset={reset}
    >
      <div className={chrome.workbench}>
        <div className={chrome.controls}>
          <div className={chrome.field}>
            <span>学习通道</span>
            <div className={chrome.choiceRow}>
              {(Object.keys(MODE_LABEL) as Mode[]).map((key) => (
                <button
                  type="button"
                  key={key}
                  aria-pressed={mode === key}
                  onClick={() => {
                    setMode(key);
                    invalidate();
                  }}
                >
                  {MODE_LABEL[key]}
                </button>
              ))}
            </div>
          </div>
          <label>
            <span>
              揭晓到第 <strong>{visible}</strong> 天
            </span>
            <input
              type="range"
              min="1"
              max="14"
              step="1"
              value={visible}
              onChange={(event) => {
                setVisible(Number(event.target.value));
              }}
            />
          </label>
          <div className={chrome.formula}>
            <code>seat_probe = 语义名录 == 今日座位</code>
            <code>rule_probe = 拟合 2h+3loss，不读当日手册</code>
            <code>冻结/RAG 的 seat_probe 为 0，不是 1</code>
          </div>
        </div>

        <div className={chrome.panel}>
          <div className={chrome.metrics}>
            <div className={chrome.metric}>
              <span>叫到今日座位</span>
              <strong>{hasRun ? last.seat : "?"}</strong>
            </div>
            <div className={chrome.metric}>
              <span>技能数</span>
              <strong>{hasRun ? last.skills : "?"}</strong>
            </div>
            <div className={chrome.metric}>
              <span>新规则探针</span>
              <strong>{hasRun ? last.rule : "?"}</strong>
            </div>
          </div>
          <div className={styles.board} aria-label="14 日任务">
            {DAY_TITLE.map((title, index) => {
              const revealed = hasRun && index < visible;
              return (
                <article key={title} data-on={revealed ? "true" : "false"}>
                  <span>D{index + 1}</span>
                  <strong>{revealed ? title : "未揭晓"}</strong>
                </article>
              );
            })}
          </div>
          <svg
            className={chrome.chart}
            viewBox="0 0 280 88"
            aria-label="座位探针"
          >
            <polyline
              points={hasRun ? seatPts : ""}
              stroke="#225ecb"
              strokeWidth="2"
            />
          </svg>
          <svg
            className={chrome.chart}
            viewBox="0 0 280 88"
            aria-label="技能数"
          >
            <polyline
              points={hasRun ? skillPts : ""}
              stroke="#1b7a53"
              strokeWidth="2"
            />
          </svg>
          <svg
            className={chrome.chart}
            viewBox="0 0 280 88"
            aria-label="新规则探针"
          >
            <polyline
              points={hasRun ? rulePts : ""}
              stroke="#875e16"
              strokeWidth="2"
            />
          </svg>
          {hasRun ? (
            <table className={chrome.table}>
              <thead>
                <tr>
                  <th>通道</th>
                  <th>B7</th>
                  <th>技能</th>
                  <th>新规则</th>
                </tr>
              </thead>
              <tbody>
                {(Object.keys(MODE_LABEL) as Mode[]).map((key) => (
                  <tr key={key}>
                    <td>{MODE_LABEL[key]}</td>
                    <td>{finals[key].seat}</td>
                    <td>{finals[key].skills}</td>
                    <td>{finals[key].rule}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className={chrome.note}>
              运行后对照四条通道。冻结的座位探针是 0：它从没学会，不能算「保持 1」。
            </p>
          )}
          {shown.length ? (
            <p className={chrome.note}>
              第 {visible} 天「{shown[shown.length - 1].title}」。蓝线能否叫到今日座位，绿线技能，黄线新规则。
            </p>
          ) : null}
        </div>
      </div>

      <div className={chrome.prediction}>
        <fieldset>
          <legend>预测 1：第 14 日用当前世界回放「叫小王」，谁能叫到 B7？</legend>
          <div className={chrome.choiceRow}>
            <button
              type="button"
              aria-pressed={seatPred === "freeze"}
              onClick={() => {
                setSeatPred("freeze");
                invalidate();
              }}
            >
              冻结 / 只塞当天手册
            </button>
            <button
              type="button"
              aria-pressed={seatPred === "semantic"}
              onClick={() => {
                setSeatPred("semantic");
                invalidate();
              }}
            >
              记忆+技能 与 再加权重
            </button>
          </div>
        </fieldset>
        <fieldset>
          <legend>预测 2：计分改成客户损失后，谁能在没有当日手册时算对？</legend>
          <div className={chrome.choiceRow}>
            <button
              type="button"
              aria-pressed={rulePred === "memory"}
              onClick={() => {
                setRulePred("memory");
                invalidate();
              }}
            >
              记忆+技能就够
            </button>
            <button
              type="button"
              aria-pressed={rulePred === "weights"}
              onClick={() => {
                setRulePred("weights");
                invalidate();
              }}
            >
              必须再加权重
            </button>
          </div>
        </fieldset>
        <button
          type="button"
          className={chrome.run}
          disabled={!seatPred || !rulePred}
          onClick={run}
        >
          运行 14 日
        </button>
      </div>

      <Gate ran={hasRun} passed={gatePassed}>
        {!hasRun
          ? "先判断座位探针和计分规则分别靠哪条通道，再揭晓看板。"
          : gatePassed
            ? `冻结/RAG 的 B7 探针是 0；记忆+技能是 ${finals.skills.seat}，新规则仍是 0；再加权重新规则是 ${finals.weights.rule}。当前通道是${MODE_LABEL[mode]}。`
            : "用今天的世界回放。冻结从来没写下座位，探针是 0 不是 1。记忆能叫到 B7，但新计分规则要拟合进权重。"}
      </Gate>
    </LabFrame>
  );
}
