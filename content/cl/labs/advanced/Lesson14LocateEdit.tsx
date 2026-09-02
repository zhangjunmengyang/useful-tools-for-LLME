"use client";

import { useMemo, useState, type CSSProperties } from "react";
import { Gate, LabFrame } from "./LabFrame";
import chrome from "./LabFrame.module.css";
import styles from "./Lesson14LocateEdit.module.css";
import type { AdvancedLabProps } from "./types";
import { numberFrom, round, sigmoid } from "./labUtils";

const TOKENS = ["法国", "的", "首都", "是", "巴黎"] as const;
const LAYERS = [0, 1, 2, 3, 4, 5] as const;
const SUBJECT = 0;
const OBJECT = 4;
const PEAK_LAYER = 3;

type TokenPred = (typeof TOKENS)[number];
type NeighborPred = "stable" | "moved";

function causal(layer: number, token: number) {
  return Math.exp(
    -((layer - PEAK_LAYER) ** 2) / 1.55 - (token - SUBJECT) ** 2 / 0.62,
  );
}

function editOutcome(layer: number, token: number, strength: number) {
  const score = causal(layer, token);
  const peak = causal(PEAK_LAYER, SUBJECT);
  const reliability = sigmoid(8 * ((score / peak) * strength - 0.42));
  const early = (5 - layer) / 5;
  const offSubject =
    token === SUBJECT ? 0 : 0.5 + (token === OBJECT ? 0.35 : 0.12);
  const neighborMove = Math.min(
    1,
    (0.1 + 0.62 * early + offSubject) *
      (0.35 + 0.65 * (1 - score / peak)) *
      (0.55 + 0.45 * strength),
  );
  return {
    score,
    reliability: round(reliability, 3),
    neighborMove: round(neighborMove, 3),
    locality: round(1 - neighborMove, 3),
  };
}

export function Lesson14LocateEdit({
  onComplete,
  initialState,
}: AdvancedLabProps) {
  const defaults = {
    strength: numberFrom(initialState, "strength", 1, 0.4, 1.6),
  };
  const [strength, setStrength] = useState(defaults.strength);
  const [picked, setPicked] = useState<{ layer: number; token: number } | null>(
    null,
  );
  const [tokenPred, setTokenPred] = useState<TokenPred | null>(null);
  const [neighborPred, setNeighborPred] = useState<NeighborPred | null>(null);
  const [hasRun, setHasRun] = useState(false);

  const grid = useMemo(
    () =>
      LAYERS.map((layer) =>
        TOKENS.map((_, token) => ({
          layer,
          token,
          ...editOutcome(layer, token, strength),
        })),
      ),
    [strength],
  );
  const peak = causal(PEAK_LAYER, SUBJECT);
  const selected = picked
    ? editOutcome(picked.layer, picked.token, strength)
    : null;
  const atPeak =
    picked !== null &&
    picked.token === SUBJECT &&
    picked.layer >= 2 &&
    picked.layer <= 4;
  const success =
    selected !== null &&
    selected.reliability > 0.55 &&
    selected.neighborMove < 0.4;
  const gatePassed =
    hasRun &&
    tokenPred === "法国" &&
    neighborPred === "stable" &&
    atPeak &&
    success;

  function invalidate() {
    setHasRun(false);
  }

  function run() {
    setHasRun(true);
    const passed =
      tokenPred === "法国" &&
      neighborPred === "stable" &&
      atPeak &&
      selected !== null &&
      selected.reliability > 0.55 &&
      selected.neighborMove < 0.4;
    if (passed && picked && selected) {
      onComplete?.({
        strength,
        layer: picked.layer,
        token: TOKENS[picked.token],
        reliability: selected.reliability,
        locality: selected.locality,
      });
    }
  }

  function reset() {
    setStrength(defaults.strength);
    setPicked(null);
    setTokenPred(null);
    setNeighborPred(null);
    setHasRun(false);
  }

  return (
    <LabFrame
      lesson="14"
      title="定位-改写：改首都，别改邻居"
      description="ROME 把一条事实当作中层 MLP、主语末词元上的键值。点选层 × 词元做一次秩一改写，看目标「法国首都 → 里昂」和邻居「德国首都 / 法语」谁动。"
      onReset={reset}
    >
      <div className={chrome.workbench}>
        <div className={chrome.controls}>
          <label>
            <span>
              编辑强度 <strong>{strength.toFixed(1)}</strong>
            </span>
            <input
              type="range"
              min="0.4"
              max="1.6"
              step="0.1"
              value={strength}
              onChange={(event) => {
                setStrength(Number(event.target.value));
                invalidate();
              }}
            />
          </label>
          <p className={chrome.note}>
            点热力图选编辑位置。因果分数在运行后才染色；预测要先判断峰值该在哪。
          </p>
          <div className={chrome.formula}>
            <code>s(l,t) = exp(-(l-3)²/1.55 - (t-法国)²/0.62)</code>
            <code>reliability = σ(8(s/s*·α - 0.42))</code>
            <code>neighbor ∝ 浅层 + 偏离主语 + (1 - s/s*)</code>
          </div>
        </div>

        <div className={chrome.panel}>
          <div className={chrome.metrics}>
            <div className={chrome.metric}>
              <span>目标改写</span>
              <strong>
                {hasRun && selected
                  ? `${Math.round(selected.reliability * 100)}%`
                  : "待运行"}
              </strong>
            </div>
            <div className={chrome.metric}>
              <span>邻居被带偏</span>
              <strong>
                {hasRun && selected
                  ? `${Math.round(selected.neighborMove * 100)}%`
                  : "?"}
              </strong>
            </div>
            <div className={chrome.metric}>
              <span>编辑点</span>
              <strong>
                {picked
                  ? `L${picked.layer} · ${TOKENS[picked.token]}`
                  : "未点选"}
              </strong>
            </div>
          </div>
          <div className={styles.heatmap} aria-label="层乘词元因果热力图">
            <div className={styles.head}>
              <span />
              {TOKENS.map((token) => (
                <span key={token}>{token}</span>
              ))}
            </div>
            {grid.map((row) => (
              <div key={row[0].layer} className={styles.row}>
                <span>L{row[0].layer}</span>
                {row.map((cell) => {
                  const heat = hasRun ? cell.score / peak : 0;
                  const isPicked =
                    picked?.layer === cell.layer && picked.token === cell.token;
                  return (
                    <button
                      key={`${cell.layer}-${cell.token}`}
                      type="button"
                      aria-pressed={isPicked}
                      className={isPicked ? styles.picked : undefined}
                      style={{ "--heat": `${Math.round(heat * 100)}%` } as CSSProperties}
                      onClick={() => {
                        setPicked({ layer: cell.layer, token: cell.token });
                        invalidate();
                      }}
                    >
                      {hasRun ? round(cell.score / peak, 2) : "·"}
                    </button>
                  );
                })}
              </div>
            ))}
          </div>
          <p className={chrome.note}>
            目标事实：法国的首都是里昂。邻居：德国的首都是柏林；法国的语言是法语。
          </p>
        </div>
      </div>

      <div className={chrome.prediction}>
        <fieldset>
          <legend>预测 1：因果峰值在哪个词元？</legend>
          <div className={chrome.choiceRow}>
            {TOKENS.map((token) => (
              <button
                type="button"
                key={token}
                aria-pressed={tokenPred === token}
                onClick={() => {
                  setTokenPred(token);
                  invalidate();
                }}
              >
                {token}
              </button>
            ))}
          </div>
        </fieldset>
        <fieldset>
          <legend>预测 2：在峰值处编辑，邻居会大幅改写吗？</legend>
          <div className={chrome.choiceRow}>
            <button
              type="button"
              aria-pressed={neighborPred === "stable"}
              onClick={() => {
                setNeighborPred("stable");
                invalidate();
              }}
            >
              基本不动
            </button>
            <button
              type="button"
              aria-pressed={neighborPred === "moved"}
              onClick={() => {
                setNeighborPred("moved");
                invalidate();
              }}
            >
              一起被改
            </button>
          </div>
        </fieldset>
        <button
          type="button"
          className={chrome.run}
          disabled={!tokenPred || !neighborPred || !picked}
          onClick={run}
        >
          运行编辑
        </button>
      </div>

      <Gate ran={hasRun} passed={gatePassed}>
        {!hasRun
          ? "先判断峰值词元和邻居命运，再点选编辑位置并运行。"
          : gatePassed
            ? "你点在主语末词元的中层附近，目标改写成功，邻居局部性保住了。"
            : "峰值在「法国」（主语末词元）× 中层 MLP。点浅层或「巴黎」会带偏邻居；强度过低则目标改不稳。"}
      </Gate>
    </LabFrame>
  );
}
