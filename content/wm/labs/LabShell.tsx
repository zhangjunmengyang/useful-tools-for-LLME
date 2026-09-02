"use client";

import type { ReactNode } from "react";

export function LabShell({
  brief,
  children,
  verdict,
  tone = "ok",
}: {
  brief: string;
  children: ReactNode;
  verdict?: string;
  tone?: "ok" | "warn" | "bad";
}) {
  return (
    <div className="wm-lab">
      <p className="wm-lab-brief">{brief}</p>
      {children}
      {verdict ? (
        <p className={`wm-lab-verdict${tone === "ok" ? "" : ` is-${tone}`}`}>
          {verdict}
        </p>
      ) : null}
    </div>
  );
}
