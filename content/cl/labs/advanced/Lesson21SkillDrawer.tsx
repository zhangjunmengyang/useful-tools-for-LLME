"use client";

import { useMemo, useState } from "react";
import { Gate, LabFrame } from "./LabFrame";
import chrome from "./LabFrame.module.css";
import styles from "./Lesson21SkillDrawer.module.css";
import type { AdvancedLabProps } from "./types";
import { pickFrom } from "./labUtils";

type Card = "none" | "goto" | "pickup" | "craft";

const CARDS: { id: Exclude<Card, "none">; name: string; program: string }[] = [
  { id: "goto", name: "走到箱子", program: "goto(chest)" },
  { id: "pickup", name: "捡起木头", program: "pickup(wood)" },
  { id: "craft", name: "合成木板", program: "craft(planks)" },
];

const ATTEMPTS: Record<Card, number> = {
  none: 16,
  goto: 12,
  pickup: 9,
  craft: 5,
};

export function Lesson21SkillDrawer({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const [retrieve, setRetrieve] = useState<Card>(
    pickFrom(initialState, "retrieve", ["none", "goto", "pickup", "craft"] as const, "none"),
  );
  const [fewerPred, setFewerPred] = useState<"yes" | "no" | null>(null);
  const [hasRun, setHasRun] = useState(false);

  const used = ATTEMPTS[retrieve];
  const scratch = ATTEMPTS.none;
  const gatePassed = hasRun && retrieve === "craft" && fewerPred === "yes";

  function invalidate() {
    setHasRun(false);
  }

  function run() {
    setHasRun(true);
    if (retrieve === "craft" && fewerPred === "yes") {
      onComplete?.({
        retrieve,
        used,
        scratch,
        saved: scratch - used,
      });
    }
  }

  function reset() {
    setRetrieve("none");
    setFewerPred(null);
    setHasRun(false);
  }

  const path = useMemo(() => {
    if (retrieve === "craft") return ["取木头", "沿合成图改编", "craft(sticks)"];
    if (retrieve === "pickup") return ["取木头", "从零想配方", "多次失败", "craft(sticks)"];
    if (retrieve === "goto") return ["走到箱子", "找不到合成台逻辑", "重写"];
    return ["提出任务", "写程序", "环境失败", "再写", "再失败"];
  }, [retrieve]);

  return (
    <LabFrame
      lesson="21"
      title="技能抽屉：第四个任务检索哪张卡"
      description="前三个任务成功的程序会变成抽屉里的卡片。第四个任务「合成木棍」来了，你可以检索一张卡片，或从零写。网格世界只用来数尝试次数，不是官方 Voyager 分数。"
      onReset={reset}
    >
      <div className={chrome.workbench}>
        <div className={chrome.controls}>
          <div className={chrome.field}>
            <span>第四个任务检索</span>
            <div className={chrome.choiceRow}>
              <button
                type="button"
                aria-pressed={retrieve === "none"}
                onClick={() => {
                  setRetrieve("none");
                  invalidate();
                }}
              >
                从零写
              </button>
              {CARDS.map((card) => (
                <button
                  type="button"
                  key={card.id}
                  aria-pressed={retrieve === card.id}
                  onClick={() => {
                    setRetrieve(card.id);
                    invalidate();
                  }}
                >
                  {card.name}
                </button>
              ))}
            </div>
          </div>
          <div className={chrome.formula}>
            <code>scratch = 16</code>
            <code>retrieve(craft_planks) = 5</code>
            <code>{"retrieve(wrong card) ∈ {12, 9}"}</code>
          </div>
        </div>

        <div className={chrome.panel}>
          <div className={chrome.metrics}>
            <div className={chrome.metric}>
              <span>尝试次数</span>
              <strong>{hasRun ? used : "?"}</strong>
            </div>
            <div className={chrome.metric}>
              <span>从零基线</span>
              <strong>{scratch}</strong>
            </div>
            <div className={chrome.metric}>
              <span>少走</span>
              <strong>{hasRun ? scratch - used : "?"}</strong>
            </div>
          </div>
          <div className={styles.cards}>
            {CARDS.map((card, index) => (
              <article
                key={card.id}
                data-active={hasRun && retrieve === card.id ? "true" : "false"}
              >
                <span>任务 {index + 1}</span>
                <strong>{card.name}</strong>
                <code>{card.program}</code>
              </article>
            ))}
          </div>
          <ol className={styles.path}>
            {(hasRun ? path : ["待运行"]).map((step) => (
              <li key={step}>{step}</li>
            ))}
          </ol>
          <p className={chrome.note}>
            权重没有变。这是经验的外存，对应第 16 课矩阵里的「流程技能 / 记忆」格。
          </p>
        </div>
      </div>

      <div className={chrome.prediction}>
        <fieldset>
          <legend>预测：检索对口的「合成木板」后，尝试次数会比从零少吗？</legend>
          <div className={chrome.choiceRow}>
            <button
              type="button"
              aria-pressed={fewerPred === "yes"}
              onClick={() => {
                setFewerPred("yes");
                invalidate();
              }}
            >
              会更少
            </button>
            <button
              type="button"
              aria-pressed={fewerPred === "no"}
              onClick={() => {
                setFewerPred("no");
                invalidate();
              }}
            >
              不会更少
            </button>
          </div>
        </fieldset>
        <button
          type="button"
          className={chrome.run}
          disabled={!fewerPred}
          onClick={run}
        >
          运行任务 4
        </button>
      </div>

      <Gate ran={hasRun} passed={gatePassed}>
        {!hasRun
          ? "先判断检索能否减少步数，并选择一张卡片（或从零）。"
          : gatePassed
            ? `检索合成木板后只需 ${used} 步，从零要 ${scratch} 步。`
            : "过关需要检索「合成木板」：配方结构最近。走箱子或从零都会多试许多次。"}
      </Gate>
    </LabFrame>
  );
}
