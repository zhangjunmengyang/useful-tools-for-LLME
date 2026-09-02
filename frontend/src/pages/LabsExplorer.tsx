import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { fetchLabs } from "../api";
import { useI18n } from "../components/I18nProvider";
import { labGroup, labGroupDescription, labName, labPageLabel } from "../lib/catalog-i18n";
import { cn } from "../lib/utils";
import type { LabPage, LabsPayload } from "../types";

export function LabsExplorer() {
  const { pageId = "" } = useParams();
  const { language, tr } = useI18n();
  const [payload, setPayload] = useState<LabsPayload | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    fetchLabs()
      .then((next) => {
        if (!cancelled) setPayload(next);
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const pages = payload?.pages ?? [];
  const groups = useMemo(() => {
    const seen: string[] = [];
    for (const page of pages) {
      if (!seen.includes(page.group)) seen.push(page.group);
    }
    return seen;
  }, [pages]);

  const selected = pages.find((page) => page.id === pageId) ?? pages[0];
  const embedSrc = selected?.embed_url ?? "/labs/";

  return (
    <div className="flex h-full min-h-0 flex-col bg-background text-foreground">
      <div className="flex h-12 shrink-0 items-center justify-between gap-3 border-b border-border px-4 md:px-6">
        <div className="min-w-0">
          <p className="text-2xs font-medium uppercase tracking-[0.14em] text-muted-foreground">
            {tr("Gradio 实验室", "Gradio Labs")}
          </p>
          <h2 className="truncate text-sm font-semibold">
            {selected
              ? `${labName(language, selected)} / ${labPageLabel(language, selected)}`
              : tr("可视化实验室", "Chart labs")}
          </h2>
        </div>
        <Link to="/tools" className="shrink-0 text-xs font-medium text-primary-ink underline-offset-4 hover:underline">
          {tr("打开 API 工具台", "Open API tools")}
        </Link>
      </div>

      {error ? <p className="px-6 py-4 text-sm text-destructive">{error}</p> : null}
      {!payload && !error ? (
        <p className="px-6 py-4 text-sm text-muted-foreground">{tr("正在读取实验室目录…", "Loading the lab catalog…")}</p>
      ) : null}

      <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[260px_minmax(0,1fr)]">
        <nav
          className="min-h-0 overflow-y-auto border-b border-border px-3 py-3 lg:border-b-0 lg:border-r"
          aria-label={tr("实验室目录", "Lab catalog")}
        >
          {groups.map((group) => (
            <section key={group} className="mb-4">
              <p className="px-2 pb-1.5 text-2xs font-medium text-muted-foreground">{labGroup(language, group)}</p>
              <ul className="space-y-0.5">
                {pages
                  .filter((page) => page.group === group)
                  .map((page) => (
                    <LabNavItem key={page.id} page={page} active={page.id === selected?.id} />
                  ))}
              </ul>
            </section>
          ))}
        </nav>
        <section className="flex min-h-0 min-w-0 flex-col">
          {selected ? (
            <>
              <div className="shrink-0 border-b border-border px-4 py-3 md:px-6">
                <p className="text-xs text-muted-foreground">{labGroupDescription(language, selected)}</p>
                <p className="mt-1 font-mono text-2xs text-muted-foreground">{selected.module}</p>
              </div>
              <iframe
                key={selected.id}
                title={`${labName(language, selected)} ${labPageLabel(language, selected)}`}
                src={embedSrc}
                className="min-h-0 w-full flex-1 border-0 bg-background"
              />
            </>
          ) : null}
        </section>
      </div>
    </div>
  );
}

function LabNavItem({ page, active }: { page: LabPage; active: boolean }) {
  const { language } = useI18n();
  return (
    <li>
      <Link
        to={`/explore/${page.id}`}
        aria-current={active ? "page" : undefined}
        className={cn(
          "block rounded-md px-2 py-1.5 text-xs transition-colors",
          active
            ? "bg-primary-muted/50 font-medium text-primary-ink"
            : "text-muted-foreground hover:bg-muted/50 hover:text-foreground",
        )}
      >
        <span className="block">{labPageLabel(language, page)}</span>
        <span className="mt-0.5 block text-2xs text-muted-foreground">{labName(language, page)}</span>
      </Link>
    </li>
  );
}
