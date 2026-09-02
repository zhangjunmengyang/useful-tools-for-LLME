"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import chrome from "./LabFrame.module.css";
import styles from "./Lesson16Router.module.css";
import type { AdvancedLabProps } from "./types";

type Exp = "fact" | "doc" | "skill" | "reason";
type Dest = "context" | "memory" | "edit" | "weights";

const ITEMS: { id: Exp; label: string }[] = [
  { id: "fact", label: "事实：小王座位改到 5 楼" },
  { id: "doc", label: "文档：刚检索到的项目周报" },
  { id: "skill", label: "流程：发布脚本的惯用命令" },
  { id: "reason", label: "推理模式：超时扣 2 分的新规则" },
];

const DEST_LABEL: Record<Dest, string> = {
  context: "上下文",
  memory: "记忆",
  edit: "编辑",
  weights: "权重",
};

const DEST_KEYS = Object.keys(DEST_LABEL) as Dest[];

function routeOutcome(exp: Exp, dest: Dest) {
  if (exp === "fact") {
    if (dest === "context") {
      return { ok: false, reason: "下一次会话窗口被清空，座位又变回旧的。" };
    }
    return {
      ok: true,
      reason:
        dest === "weights"
          ? "能过，但一条座位事实不必动全部权重。"
          : "下次还能叫到小王。",
    };
  }
  if (exp === "doc") {
    if (dest === "context" || dest === "memory") {
      return {
        ok: true,
        reason:
          dest === "context"
            ? "当前提问能引用这篇周报。"
            : "周报进了档案，以后还能检索。",
      };
    }
    return { ok: false, reason: "整篇文档不是一条可定位的事实，编辑和微调都塞不进去。" };
  }
  if (exp === "skill") {
    if (dest === "memory") {
      return { ok: true, reason: "惯用命令写成可再调用的卡片。" };
    }
    if (dest === "weights") {
      return { ok: true, reason: "可以过，但流程技能优先放进外存。" };
    }
    if (dest === "context") {
      return { ok: false, reason: "这轮对话结束，命令习惯就丢了。" };
    }
    return { ok: false, reason: "知识编辑改的是事实三元组，改不了一串命令。" };
  }
  if (dest === "weights") {
    return { ok: true, reason: "新计分规则写进权重，换题还能用。" };
  }
  return {
    ok: false,
    reason: "记忆和提示只能撑这一次；同类新题仍按旧规则打分。",
  };
}

export function Lesson16Router({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const restored =
    initialState?.routes &&
    typeof initialState.routes === "object" &&
    !Array.isArray(initialState.routes)
      ? (initialState.routes as Partial<Record<Exp, Dest>>)
      : {};
  const [routes, setRoutes] = useState<Partial<Record<Exp, Dest>>>({
    fact: restored.fact,
    doc: restored.doc,
    skill: restored.skill,
    reason: restored.reason,
  });
  const [reasonPred, setReasonPred] = useState<"pass" | "fail" | null>(null);
  const [distillPred, setDistillPred] = useState<"hold" | "drop" | null>(null);
  const [hasRun, setHasRun] = useState(false);

  const outcomes = useMemo(
    () =>
      ITEMS.map((item) => {
        const dest = routes[item.id];
        return {
          ...item,
          dest,
          result: dest ? routeOutcome(item.id, dest) : null,
        };
      }),
    [routes],
  );
  const allRouted = outcomes.every((row) => row.dest);
  const allOk = outcomes.every((row) => row.result?.ok);
  const gatePassed =
    hasRun && allOk && reasonPred === "fail" && distillPred === "hold";

  function invalidate() {
    setHasRun(false);
  }

  function setRoute(id: Exp, dest: Dest) {
    setRoutes((current) => ({ ...current, [id]: dest }));
    invalidate();
  }

  function run() {
    setHasRun(true);
    const passed =
      allOk && reasonPred === "fail" && distillPred === "hold";
    if (passed) {
      onComplete?.({
        routes,
        reasonMemoryFails: true,
        distillKeepsFacts: true,
      });
    }
  }

  function reset() {
    setRoutes({});
    setReasonPred(null);
    setDistillPred(null);
    setHasRun(false);
  }

  return (
    <LabFrame
      lesson="16"
      title="分流器：新经验写到哪一层"
      description="来了一条新经验，先决定写进上下文、外挂记忆、知识编辑，还是改权重。系统按本课的四类经验打分：错分会给出失败案例。"
      onReset={reset}
    >
      <div className={chrome.workbench}>
        <div className={chrome.controls}>
          {ITEMS.map((item) => (
            <div className={chrome.field} key={item.id}>
              <span>{item.label}</span>
              <div className={chrome.choiceRow}>
                {DEST_KEYS.map((dest) => (
                  <button
                    type="button"
                    key={dest}
                    aria-pressed={routes[item.id] === dest}
                    onClick={() => setRoute(item.id, dest)}
                  >
                    {DEST_LABEL[dest]}
                  </button>
                ))}
              </div>
            </div>
          ))}
          <div className={chrome.formula}>
            <code>事实 → 记忆或编辑</code>
            <code>文档 → 上下文或记忆</code>
            <code>流程 → 记忆（技能卡）</code>
            <code>推理模式 → 权重</code>
          </div>
        </div>

        <div className={chrome.panel}>
          <table className={chrome.table}>
            <thead>
              <tr>
                <th>经验</th>
                <th>去向</th>
                <th>结果</th>
              </tr>
            </thead>
            <tbody>
              {outcomes.map((row) => (
                <tr key={row.id}>
                  <td>{row.label.split("：")[0]}</td>
                  <td>{row.dest ? DEST_LABEL[row.dest] : "—"}</td>
                  <td
                    className={
                      hasRun && row.result
                        ? row.result.ok
                          ? styles.ok
                          : styles.fail
                        : undefined
                    }
                  >
                    {hasRun && row.result
                      ? row.result.ok
                        ? `过关。${row.result.reason}`
                        : `失败。${row.result.reason}`
                      : "待运行"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className={chrome.note}>
            这张通过矩阵是第六幕技能库和 14 日看板的设计依据。外挂记忆不会的三件事：新技能的执行策略、新推理模式、新运动策略——后两件必须动权重。事实可以先写日记，夜间巩固进权重后再卸库；CPU 实验是 extra distill / evolve。
          </p>
        </div>
      </div>

      <div className={chrome.prediction}>
        <fieldset>
          <legend>预测：新计分规则只写入记忆，下周同类题会过吗？</legend>
          <div className={chrome.choiceRow}>
            <button
              type="button"
              aria-pressed={reasonPred === "pass"}
              onClick={() => {
                setReasonPred("pass");
                invalidate();
              }}
            >
              会过
            </button>
            <button
              type="button"
              aria-pressed={reasonPred === "fail"}
              onClick={() => {
                setReasonPred("fail");
                invalidate();
              }}
            >
              不会过
            </button>
          </div>
        </fieldset>
        <fieldset>
          <legend>
            预测：座位先写入记忆，夜间练进权重并卸掉日记，还能叫到小王吗？
          </legend>
          <div className={chrome.choiceRow}>
            <button
              type="button"
              aria-pressed={distillPred === "hold"}
              onClick={() => {
                setDistillPred("hold");
                invalidate();
              }}
            >
              还能
            </button>
            <button
              type="button"
              aria-pressed={distillPred === "drop"}
              onClick={() => {
                setDistillPred("drop");
                invalidate();
              }}
            >
              叫不到
            </button>
          </div>
        </fieldset>
        <button
          type="button"
          className={chrome.run}
          disabled={!allRouted || !reasonPred || !distillPred}
          onClick={run}
        >
          运行分流
        </button>
      </div>

      <Gate ran={hasRun} passed={gatePassed}>
        {!hasRun
          ? "先给四类经验各选一个去向，并判断：规则放进记忆会不会过、事实巩固进权重后卸库还在不在。"
          : gatePassed
            ? "四类都分对了。规则必须改权重；事实可以先写日记，巩固进权重后再卸库。"
            : outcomes.some((row) => row.result && !row.result.ok)
              ? outcomes
                  .filter((row) => row.result && !row.result.ok)
                  .map((row) => row.result?.reason)
                  .join(" ")
              : reasonPred !== "fail"
                ? "推理模式写入记忆，下周换一道同规则的题仍会按旧规则打分。"
                : "卸掉日记之后，只有写进权重的座位还在。巩固是训练，不是再抄一遍卡片。"}
      </Gate>
    </LabFrame>
  );
}
