"use client";

import { useMemo, useState } from "react";
import styles from "./Lab04CallXiaowang.module.css";
import type { FoundationLabProps } from "./types";
import { initialBoolean, initialString } from "./types";

type WriteKey = "restate" | "retrieve" | "weights";

const roster = [
  { name: "小王", desk: "3-12", ext: "2047", team: "仓储" },
  { name: "小李", desk: "2-04", ext: "1188", team: "财务" },
  { name: "小陈", desk: "5-19", ext: "3310", team: "法务" },
] as const;

const writeMeta: Record<WriteKey, { name: string; hint: string }> = {
  restate: { name: "每次重说", hint: "名录只活在当前 prompt 里" },
  retrieve: { name: "检索", hint: "名录是外挂文档，藏起来就 404" },
  weights: { name: "改权重", hint: "名字到工位写进参数" },
};

function canCall(
  write: WriteKey,
  hidden: boolean,
  keepIndex: boolean,
): boolean {
  if (!hidden) return true;
  if (write === "restate") return false;
  if (write === "retrieve") return keepIndex;
  return true;
}

export function Lab04CallXiaowang({
  onComplete,
  initialState,
}: FoundationLabProps) {
  const [write, setWrite] = useState<WriteKey>(
    initialString(
      initialState,
      "write",
      ["restate", "retrieve", "weights"] as const,
      "restate",
    ),
  );
  const [keepIndex, setKeepIndex] = useState(
    initialBoolean(initialState, "keepIndex", false),
  );
  const [survivePrediction, setSurvivePrediction] = useState<WriteKey | null>(
    null,
  );
  const [minePrediction, setMinePrediction] = useState<"yes" | "no" | null>(
    null,
  );
  const [hasRun, setHasRun] = useState(false);

  const hidden = hasRun;
  const outcome = useMemo(() => {
    const rows = (Object.keys(writeMeta) as WriteKey[]).map((key) => ({
      key,
      ok: canCall(key, true, keepIndex),
    }));
    return { rows, mine: canCall(write, true, keepIndex) };
  }, [keepIndex, write]);

  const gatePassed =
    hasRun &&
    survivePrediction === "weights" &&
    minePrediction === (outcome.mine ? "yes" : "no");

  function invalidate() {
    setHasRun(false);
  }

  function runLab() {
    setHasRun(true);
    const passed =
      survivePrediction === "weights" &&
      minePrediction === (canCall(write, true, keepIndex) ? "yes" : "no");
    if (passed) {
      onComplete?.({
        write,
        keepIndex,
        hidden: true,
        survivors: outcome.rows.filter((row) => row.ok).map((row) => row.key),
      });
    }
  }

  function reset() {
    setWrite("restate");
    setKeepIndex(false);
    setSurvivePrediction(null);
    setMinePrediction(null);
    setHasRun(false);
  }

  return (
    <section className={styles.lab} aria-labelledby="lab04-title">
      <header className={styles.header}>
        <div>
          <div className={styles.labels}>
            <span>教学模拟</span>
            <span>上下文 / 检索 / 权重</span>
          </div>
          <h3 id="lab04-title">叫小王：名录撤掉之后谁还在</h3>
          <p>
            一间虚拟公司要执行「把小王叫过来」。写入位置不同，名录藏起来之后结果不同。检索是去外挂文档里找，不是把流程写进模型。
          </p>
        </div>
        <button className={styles.reset} type="button" onClick={reset}>
          重置实验
        </button>
      </header>

      <div className={styles.workbench}>
        <div className={styles.controls}>
          <p>先选一种写法，再决定藏名录时检索引擎还在不在。</p>
          <div className={styles.choiceRow}>
            {(Object.keys(writeMeta) as WriteKey[]).map((key) => (
              <button
                type="button"
                key={key}
                aria-pressed={write === key}
                onClick={() => {
                  setWrite(key);
                  invalidate();
                }}
              >
                {writeMeta[key].name}
              </button>
            ))}
          </div>
          <label>
            <input
              type="checkbox"
              checked={keepIndex}
              onChange={(event) => {
                setKeepIndex(event.target.checked);
                invalidate();
              }}
            />{" "}
            藏名录时保留检索副本
          </label>
          <p>{writeMeta[write].hint}</p>
        </div>

        <div className={styles.stage} aria-live="polite">
          <table className={styles.roster}>
            <thead>
              <tr>
                <th>员工</th>
                <th>工位</th>
                <th>分机</th>
                <th>状态</th>
              </tr>
            </thead>
            <tbody>
              {roster.map((person) => (
                <tr key={person.name}>
                  <td>{hidden ? "—" : person.name}</td>
                  <td>{hidden ? "—" : person.desk}</td>
                  <td>{hidden ? "—" : person.ext}</td>
                  <td>{hidden ? "名录已撤" : person.team}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <table className={styles.channels}>
            <thead>
              <tr>
                <th>通道</th>
                <th>叫小王</th>
              </tr>
            </thead>
            <tbody>
              {outcome.rows.map((row) => (
                <tr key={row.key}>
                  <td>{writeMeta[row.key].name}</td>
                  <td className={hasRun ? (row.ok ? styles.ok : styles.fail) : undefined}>
                    {hasRun ? (row.ok ? "能叫到" : "叫不到") : "待运行"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className={styles.trace}>
            {hidden
              ? "名录从工位上撤走。prompt 变空；检索只在你勾选「保留副本」时还能命中；权重里的名字映射还在。"
              : "名录还在。三种写法现在都能执行。"}
          </p>
        </div>
      </div>

      <div className={styles.prediction}>
        <fieldset>
          <legend>预测 1：名录和索引都撤掉后，谁还能叫到小王？</legend>
          <div className={styles.choiceRow}>
            {(Object.keys(writeMeta) as WriteKey[]).map((key) => (
              <button
                type="button"
                key={key}
                aria-pressed={survivePrediction === key}
                onClick={() => {
                  setSurvivePrediction(key);
                  invalidate();
                }}
              >
                {writeMeta[key].name}
              </button>
            ))}
          </div>
        </fieldset>
        <fieldset>
          <legend>预测 2：你选的写法，撤掉后还能叫来小王吗？</legend>
          <div className={styles.choiceRow}>
            <button
              type="button"
              aria-pressed={minePrediction === "yes"}
              onClick={() => {
                setMinePrediction("yes");
                invalidate();
              }}
            >
              能
            </button>
            <button
              type="button"
              aria-pressed={minePrediction === "no"}
              onClick={() => {
                setMinePrediction("no");
                invalidate();
              }}
            >
              不能
            </button>
          </div>
        </fieldset>
        <button
          type="button"
          className={styles.run}
          disabled={!survivePrediction || !minePrediction}
          onClick={runLab}
        >
          撤掉名录并验收
        </button>
      </div>

      <div
        className={`${styles.gate} ${
          hasRun ? (gatePassed ? styles.pass : styles.retry) : ""
        }`}
        role="status"
      >
        <strong>{gatePassed ? "验收已通过" : "完成验收"}</strong>
        <span>
          {!hasRun
            ? "先判断谁能活下来，再撤名录。"
            : gatePassed
              ? keepIndex
                ? "你留了检索副本，检索和权重都能叫到小王。这仍是外挂记忆，不是上岗那种改权重的学习。"
                : "名录和索引一起撤掉后，只有写进权重的映射还在。"
              : "默认（不保留副本）只有改权重还在。勾选保留副本时，检索也能命中。"}
        </span>
      </div>
    </section>
  );
}
