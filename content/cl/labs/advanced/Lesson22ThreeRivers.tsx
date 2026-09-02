"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import chrome from "./LabFrame.module.css";
import styles from "./Lesson22ThreeRivers.module.css";
import type { AdvancedLabProps } from "./types";
import { booleanFrom, numberFrom, polylinePoints } from "./labUtils";

function flow(days: number, web: number, expertFlow: number, pack: number, envOn: boolean, envFlow: number) {
  let facts = 0;
  let skills = 0;
  let expertLeft = pack;
  const series: { facts: number; skills: number }[] = [];
  for (let day = 1; day <= days; day += 1) {
    facts += web;
    const take = Math.min(expertFlow, expertLeft);
    skills += take;
    expertLeft -= take;
    if (envOn) skills += envFlow;
    series.push({ facts, skills });
  }
  return series;
}

export function Lesson22ThreeRivers({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    days: numberFrom(initialState, "days", 10, 3, 16),
    web: numberFrom(initialState, "web", 3, 0, 6),
    expertFlow: numberFrom(initialState, "expertFlow", 2, 0, 4),
    pack: numberFrom(initialState, "pack", 6, 2, 12),
    envFlow: numberFrom(initialState, "envFlow", 1, 1, 3),
    envOn: booleanFrom(initialState, "envOn", true),
  };
  const [days, setDays] = useState(defaults.days);
  const [web, setWeb] = useState(defaults.web);
  const [expertFlow, setExpertFlow] = useState(defaults.expertFlow);
  const [pack, setPack] = useState(defaults.pack);
  const [envFlow, setEnvFlow] = useState(defaults.envFlow);
  const [envOn, setEnvOn] = useState(defaults.envOn);
  const [growthPred, setGrowthPred] = useState<"yes" | "no" | null>(null);
  const [webPred, setWebPred] = useState<"facts" | "skills" | null>(null);
  const [hasRun, setHasRun] = useState(false);

  const withEnv = useMemo(
    () => flow(days, web, expertFlow, pack, true, envFlow),
    [days, envFlow, expertFlow, pack, web],
  );
  const noEnv = useMemo(
    () => flow(days, web, expertFlow, pack, false, envFlow),
    [days, envFlow, expertFlow, pack, web],
  );
  const active = envOn ? withEnv : noEnv;
  const last = active[active.length - 1];
  const lastNo = noEnv[noEnv.length - 1];
  const gatePassed = hasRun && growthPred === "no" && webPred === "facts";

  function invalidate() {
    setHasRun(false);
  }

  function run() {
    setHasRun(true);
    if (growthPred === "no" && webPred === "facts") {
      onComplete?.({
        days,
        web,
        expertFlow,
        pack,
        envOn,
        envFlow,
        facts: last.facts,
        skills: last.skills,
        skillsIfEnvOff: lastNo.skills,
      });
    }
  }

  function reset() {
    setDays(defaults.days);
    setWeb(defaults.web);
    setExpertFlow(defaults.expertFlow);
    setPack(defaults.pack);
    setEnvFlow(defaults.envFlow);
    setEnvOn(true);
    setGrowthPred(null);
    setWebPred(null);
    setHasRun(false);
  }

  const skillOn = polylinePoints(
    withEnv.map((row) => row.skills),
    280,
    88,
    Math.max(last.skills, lastNo.skills, 1),
  );
  const skillOff = polylinePoints(
    noEnv.map((row) => row.skills),
    280,
    88,
    Math.max(last.skills, lastNo.skills, 1),
  );
  const factLine = polylinePoints(
    active.map((row) => row.facts),
    280,
    88,
  );

  return (
    <LabFrame
      lesson="22"
      title="三条河：数据从哪来"
      description="网页河增加事实，专家包是有限演示，实时环境会不断冒出新配方。关掉环境河之后，技能数在专家包用尽处停止增长。经验时代和普通 RL 的差别在时间跨度、非平稳、没有任务边界。"
      onReset={reset}
    >
      <div className={chrome.workbench}>
        <div className={chrome.controls}>
          <label>
            <span>
              天数 <strong>{days}</strong>
            </span>
            <input
              type="range"
              min="3"
              max="16"
              step="1"
              value={days}
              onChange={(event) => {
                setDays(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <label>
            <span>
              网页河（事实/日） <strong>{web}</strong>
            </span>
            <input
              type="range"
              min="0"
              max="6"
              step="1"
              value={web}
              onChange={(event) => {
                setWeb(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <label>
            <span>
              专家包大小 <strong>{pack}</strong>
            </span>
            <input
              type="range"
              min="2"
              max="12"
              step="1"
              value={pack}
              onChange={(event) => {
                setPack(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <label>
            <span>
              专家流入 / 日 <strong>{expertFlow}</strong>
            </span>
            <input
              type="range"
              min="0"
              max="4"
              step="1"
              value={expertFlow}
              onChange={(event) => {
                setExpertFlow(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <label>
            <span>环境河</span>
            <select
              value={envOn ? "on" : "off"}
              onChange={(event) => {
                setEnvOn(event.target.value === "on");
                invalidate();
              }}
            >
              <option value="on">流动</option>
              <option value="off">关掉</option>
            </select>
          </label>
          <div className={chrome.formula}>
            <code>facts += web</code>
            <code>skills += min(expert_flow, remaining)</code>
            <code>if env: skills += env_flow</code>
          </div>
        </div>

        <div className={chrome.panel}>
          <div className={chrome.metrics}>
            <div className={chrome.metric}>
              <span>事实</span>
              <strong>{hasRun ? last.facts : "?"}</strong>
            </div>
            <div className={chrome.metric}>
              <span>技能（当前河）</span>
              <strong>{hasRun ? last.skills : "?"}</strong>
            </div>
            <div className={chrome.metric}>
              <span>关掉环境后技能</span>
              <strong>{hasRun ? lastNo.skills : "?"}</strong>
            </div>
          </div>
          <div className={styles.rivers} aria-hidden="true">
            <span data-on="true">网页</span>
            <span data-on="true">专家包</span>
            <span data-on={envOn ? "true" : "false"}>环境</span>
          </div>
          <svg className={chrome.chart} viewBox="0 0 280 88" aria-label="技能增长">
            <polyline points={hasRun ? skillOn : ""} stroke="#1b7a53" strokeWidth="2" />
            <polyline
              points={hasRun ? skillOff : ""}
              stroke="#8b9690"
              strokeWidth="2"
              strokeDasharray="4 3"
            />
          </svg>
          <svg className={chrome.chart} viewBox="0 0 280 88" aria-label="事实增长">
            <polyline points={hasRun ? factLine : ""} stroke="#225ecb" strokeWidth="2" />
          </svg>
          <p className={chrome.note}>
            绿线：环境河开着的技能数。灰虚线：关掉环境河。蓝线：事实（网页河）。专家包用尽后，虚线变平。
          </p>
        </div>
      </div>

      <div className={chrome.prediction}>
        <fieldset>
          <legend>预测 1：关掉环境河、专家包用尽后，技能数还会随天数涨吗？</legend>
          <div className={chrome.choiceRow}>
            <button
              type="button"
              aria-pressed={growthPred === "yes"}
              onClick={() => {
                setGrowthPred("yes");
                invalidate();
              }}
            >
              还会涨
            </button>
            <button
              type="button"
              aria-pressed={growthPred === "no"}
              onClick={() => {
                setGrowthPred("no");
                invalidate();
              }}
            >
              不再涨
            </button>
          </div>
        </fieldset>
        <fieldset>
          <legend>预测 2：网页河主要增加的是什么？</legend>
          <div className={chrome.choiceRow}>
            <button
              type="button"
              aria-pressed={webPred === "facts"}
              onClick={() => {
                setWebPred("facts");
                invalidate();
              }}
            >
              事实
            </button>
            <button
              type="button"
              aria-pressed={webPred === "skills"}
              onClick={() => {
                setWebPred("skills");
                invalidate();
              }}
            >
              可执行技能
            </button>
          </div>
        </fieldset>
        <button
          type="button"
          className={chrome.run}
          disabled={!growthPred || !webPred}
          onClick={run}
        >
          运行三条河
        </button>
      </div>

      <Gate ran={hasRun} passed={gatePassed}>
        {!hasRun
          ? "先判断环境河关掉后技能还涨不涨，以及网页河增加的是什么。"
          : gatePassed
            ? `第 ${days} 天事实 ${last.facts}、技能 ${last.skills}；关掉环境后技能停在 ${lastNo.skills}。`
            : "网页只加事实。技能来自有限专家包和持续的环境新配方；关掉环境河且专家包用尽后，技能曲线变平。"}
      </Gate>
    </LabFrame>
  );
}
