"use client";

import type { ReactNode } from "react";
import styles from "./LabFrame.module.css";

type LabFrameProps = {
  lesson: string;
  title: string;
  description: string;
  children: ReactNode;
};

export function LabFrame({
  lesson,
  title,
  description,
  children,
}: LabFrameProps) {
  return (
    <section className={styles.frame} aria-labelledby={`advanced-lab-${lesson}`}>
      <header className={styles.header}>
        <div>
          <p className={styles.labels}>
            <span>教学模拟</span>
            <span>公式计算</span>
          </p>
          <h3 id={`advanced-lab-${lesson}`}>{title}</h3>
          <p className={styles.description}>{description}</p>
        </div>
      </header>
      {children}
    </section>
  );
}

export function Gate({
  passed,
  children,
}: {
  passed: boolean;
  children: ReactNode;
}) {
  return (
    <aside
      className={`${styles.gate} ${passed ? styles.gatePassed : ""}`}
      aria-live="polite"
    >
      <span className={styles.gateMark} aria-hidden="true">
        {passed ? "✓" : "G"}
      </span>
      <div>
        <strong>{passed ? "验收已通过" : "完成验收"}</strong>
        <p>{children}</p>
      </div>
    </aside>
  );
}
