import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { fetchLabs, fetchTools } from "../api";
import { useI18n } from "./I18nProvider";
import { labGroup, labName, labPageLabel, toolDescription, toolLabel } from "../lib/catalog-i18n";
import type { LabPage, ToolSpec } from "../types";

export function ToolkitIndex({ highlightToolIds = [] }: { highlightToolIds?: string[] }) {
  const { language, tr } = useI18n();
  const [tools, setTools] = useState<ToolSpec[]>([]);
  const [labs, setLabs] = useState<LabPage[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    Promise.all([fetchTools(), fetchLabs()])
      .then(([toolPayload, labPayload]) => {
        if (cancelled) return;
        setTools(toolPayload.tools);
        setLabs(labPayload.pages);
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const highlighted = new Set(highlightToolIds);

  return (
    <section className="mt-8 border-t border-border pt-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold tracking-tight">{tr("完整工具箱", "Full toolkit")}</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            {tr("课里的控件只是入口。下面这些都能打开，一个没删。", "The widgets in the lesson are just doors. Everything below still opens.")}
          </p>
        </div>
        <div className="flex gap-3 text-xs font-medium">
          <Link to="/tools" className="text-primary-ink underline-offset-4 hover:underline">
            {tr("工具台", "Tools")}
          </Link>
          <Link to="/explore" className="text-primary-ink underline-offset-4 hover:underline">
            {tr("实验室", "Labs")}
          </Link>
        </div>
      </div>
      {error ? <p className="mt-3 text-sm text-destructive">{error}</p> : null}
      <div className="mt-4 grid gap-6 lg:grid-cols-2">
        <div>
          <h3 className="text-xs font-medium text-muted-foreground">{tr("API 工具 /api/tools", "API tools /api/tools")}</h3>
          <ul className="mt-2 divide-y divide-border rounded-md border border-border">
            {tools.map((tool) => (
              <li key={tool.id}>
                <Link
                  to={`/tools?tool=${tool.id}`}
                  className="flex items-start justify-between gap-3 px-3 py-2 hover:bg-muted/40"
                >
                  <span>
                    <span className="block text-sm font-medium">
                      {toolLabel(language, tool)}
                      {highlighted.has(tool.id) ? (
                        <span className="ml-2 text-2xs font-normal text-primary-ink">{tr("本课", "This lesson")}</span>
                      ) : null}
                    </span>
                    <span className="mt-0.5 block text-2xs text-muted-foreground">{toolDescription(language, tool)}</span>
                  </span>
                  <span className="shrink-0 font-mono text-2xs text-muted-foreground">{tool.id}</span>
                </Link>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h3 className="text-xs font-medium text-muted-foreground">{tr("可视化实验室", "Chart labs")}</h3>
          <ul className="mt-2 divide-y divide-border rounded-md border border-border">
            {labs.map((page) => (
              <li key={page.id}>
                <Link to={`/explore/${page.id}`} className="flex items-start justify-between gap-3 px-3 py-2 hover:bg-muted/40">
                  <span>
                    <span className="block text-sm font-medium">{labPageLabel(language, page)}</span>
                    <span className="mt-0.5 block text-2xs text-muted-foreground">
                      {labName(language, page)} · {labGroup(language, page.group)}
                    </span>
                  </span>
                  <span className="shrink-0 font-mono text-2xs text-muted-foreground">{page.id}</span>
                </Link>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
