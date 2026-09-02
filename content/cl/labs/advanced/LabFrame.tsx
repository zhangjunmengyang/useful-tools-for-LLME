"use client";

import type { ReactNode } from "react";
import styles from "./LabFrame.module.css";

type LabFrameProps = {
  lesson: string;
  title: string;
  description: string;
  onReset?: () => void;
  children: ReactNode;
};

export function LabFrame({
  lesson,
  title,
  description,
  onReset,
  children,
}: LabFrameProps) {
  return (
    <section className={styles.frame} aria-labelledby={`advanced-lab-${lesson}`}>
      <header className={styles.header}>
        <div>
          <div className={styles.labels} aria-label="实验类型">
            <span>教学模拟</span>
            <span>公式计算</span>
          </div>
          <h3 id={`advanced-lab-${lesson}`}>{title}</h3>
          <p className={styles.description}>{description}</p>
        </div>
        {onReset ? (
          <button className={styles.reset} type="button" onClick={onReset}>
            重置实验
          </button>
        ) : null}
      </header>
      {children}
    </section>
  );
}

export function Gate({
  passed,
  ran = false,
  children,
}: {
  passed: boolean;
  ran?: boolean;
  children: ReactNode;
}) {
  const stateClass = ran
    ? passed
      ? styles.gatePassed
      : styles.gateRetry
    : "";

  return (
    <aside className={`${styles.gate} ${stateClass}`} role="status">
      <strong>{passed ? "验收已通过" : "完成验收"}</strong>
      <span>{children}</span>
    </aside>
  );
}
