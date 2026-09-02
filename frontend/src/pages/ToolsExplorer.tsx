import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";

import { exportTool, fetchTools, runTool } from "../api";
import { useI18n } from "../components/I18nProvider";
import { categoryCopy, toolDescription, toolLabel } from "../lib/catalog-i18n";
import {
  buildExampleInput,
  copyCurlCommand,
  firstCategoryWithTools,
  formatJson,
  toolsForCategory,
} from "../lib/mechanics";
import { cn } from "../lib/utils";
import type { ToolRun, ToolsPayload } from "../types";

export function ToolsExplorer({ initialToolId }: { initialToolId?: string }) {
  const { language, tr } = useI18n();
  const [payload, setPayload] = useState<ToolsPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedCategoryId, setSelectedCategoryId] = useState("");
  const [selectedToolId, setSelectedToolId] = useState(initialToolId ?? "");
  const [jsonInput, setJsonInput] = useState("{}");
  const [runResult, setRunResult] = useState<ToolRun | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [copied, setCopied] = useState(false);
  const runSequenceRef = useRef(0);
  const selectedToolIdRef = useRef(selectedToolId);

  useEffect(() => {
    let active = true;
    setLoading(true);
    fetchTools()
      .then((nextPayload) => {
        if (!active) return;
        setPayload(nextPayload);
        const preferred = initialToolId
          ? nextPayload.tools.find((tool) => tool.id === initialToolId)
          : undefined;
        const nextCategoryId =
          preferred?.mechanics_category ||
          firstCategoryWithTools(nextPayload.categories, nextPayload.tools);
        setSelectedCategoryId(nextCategoryId);
        setSelectedToolId(preferred?.id ?? toolsForCategory(nextPayload.tools, nextCategoryId)[0]?.id ?? "");
        setError(null);
      })
      .catch((fetchError: unknown) => {
        if (!active) return;
        setError(fetchError instanceof Error ? fetchError.message : tr("无法读取工具", "Could not load tools"));
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [initialToolId]);

  const categories = payload?.categories ?? [];
  const tools = payload?.tools ?? [];
  const selectedCategory = categories.find((category) => category.id === selectedCategoryId) ?? categories[0];
  const categoryTools = useMemo(
    () => toolsForCategory(tools, selectedCategory?.id ?? ""),
    [selectedCategory?.id, tools],
  );

  useEffect(() => {
    if (categoryTools.length === 0) {
      setSelectedToolId("");
      invalidateActiveRun("");
      return;
    }
    if (!categoryTools.some((tool) => tool.id === selectedToolId)) {
      setSelectedToolId(categoryTools[0].id);
      invalidateActiveRun(categoryTools[0].id);
    }
  }, [categoryTools, selectedToolId]);

  const selectedTool = categoryTools.find((tool) => tool.id === selectedToolId) ?? categoryTools[0];

  useEffect(() => {
    invalidateActiveRun(selectedTool?.id ?? "");
    setJsonInput(buildExampleInput(selectedTool));
  }, [selectedTool?.id]);

  function invalidateActiveRun(toolId: string) {
    runSequenceRef.current += 1;
    selectedToolIdRef.current = toolId;
    setRunResult(null);
    setRunError(null);
    setRunning(false);
    setExporting(false);
  }

  function selectCategory(categoryId: string) {
    setSelectedCategoryId(categoryId);
    const nextTool = toolsForCategory(tools, categoryId)[0];
    const nextToolId = nextTool?.id ?? "";
    setSelectedToolId(nextToolId);
    if (nextToolId !== selectedToolIdRef.current) invalidateActiveRun(nextToolId);
  }

  function selectTool(toolId: string) {
    if (toolId === selectedToolIdRef.current) return;
    setSelectedToolId(toolId);
    invalidateActiveRun(toolId);
  }

  async function handleRun(kind: "run" | "export") {
    if (!selectedTool) return;
    const toolId = selectedTool.id;
    const runSequence = runSequenceRef.current + 1;
    runSequenceRef.current = runSequence;
    selectedToolIdRef.current = toolId;
    const isActiveRun = () => runSequenceRef.current === runSequence && selectedToolIdRef.current === toolId;
    if (kind === "export") setExporting(true);
    else setRunning(true);
    setRunError(null);
    setRunResult(null);
    try {
      const inputs = JSON.parse(jsonInput) as Record<string, unknown>;
      const nextResult = kind === "export" ? await exportTool(toolId, inputs) : await runTool(toolId, inputs);
      if (isActiveRun()) {
        if (nextResult.status === "error") {
          setRunError(nextResult.error ?? tr("工具返回了错误", "The tool returned an error"));
          setRunResult(null);
        } else {
          setRunResult(nextResult);
        }
      }
    } catch (toolError: unknown) {
      if (isActiveRun()) {
        setRunError(toolError instanceof Error ? toolError.message : tr("工具没有跑起来", "The tool did not run"));
      }
    } finally {
      if (isActiveRun()) {
        setRunning(false);
        setExporting(false);
      }
    }
  }

  async function handleCopyCurl() {
    if (!selectedTool) return;
    await copyCurlCommand(selectedTool.id, jsonInput);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  }

  return (
    <div className="flex h-full min-h-0 flex-col bg-background text-foreground">
      <div className="flex h-12 shrink-0 items-center justify-between gap-3 border-b border-border px-4 md:px-6">
        <div className="min-w-0">
          <p className="text-2xs font-medium uppercase tracking-[0.14em] text-muted-foreground">
            {tr("机制浏览器", "Mechanics Explorer")}
          </p>
          <h2 className="truncate text-sm font-semibold">{tr("全部 API 工具", "All API tools")}</h2>
        </div>
        <Link
          to="/explore"
          className="shrink-0 text-xs font-medium text-primary-ink underline-offset-4 hover:underline"
        >
          {tr("打开可视化实验室", "Open chart labs")}
        </Link>
      </div>

      {loading ? <p className="px-6 py-4 text-sm text-muted-foreground">{tr("正在读取 /api/tools", "Loading /api/tools")}</p> : null}
      {error ? <p className="px-6 py-4 text-sm text-destructive">{error}</p> : null}

      <section className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[220px_minmax(0,1fr)_320px]" aria-label={tr("工具台", "Tools")}>
        <nav className="min-h-0 overflow-y-auto border-b border-border lg:border-b-0 lg:border-r" aria-label={tr("机制分类", "Mechanics categories")}>
          {categories.map((category) => {
            const copy = categoryCopy(language, category);
            return (
            <button
              key={category.id}
              type="button"
              onClick={() => selectCategory(category.id)}
              className={cn(
                "flex w-full items-start gap-2 border-l-2 px-3 py-2.5 text-left transition-colors",
                category.id === selectedCategory?.id
                  ? "border-primary bg-primary-muted/50 text-primary-ink"
                  : "border-transparent text-muted-foreground hover:bg-muted/50 hover:text-foreground",
              )}
            >
              <span className="mt-0.5 font-mono text-2xs tabular-nums">{category.stage.toString().padStart(2, "0")}</span>
              <span className="min-w-0">
                <span className="block text-xs font-medium">{copy.label}</span>
                <span className="mt-0.5 block text-2xs text-muted-foreground">{copy.subtitle}</span>
              </span>
            </button>
            );
          })}
        </nav>

        <section className="min-h-0 overflow-y-auto border-b border-border px-4 py-4 lg:border-b-0 lg:border-r md:px-6">
          <p className="text-2xs uppercase tracking-[0.14em] text-muted-foreground">
            {tr("机制画布", "Mechanics Canvas")}
          </p>
          <h3 className="mt-1 text-lg font-semibold tracking-tight">
            {selectedCategory ? categoryCopy(language, selectedCategory).label : tr("还没有分类", "No category yet")}
          </h3>
          <p className="mt-1 max-w-[65ch] text-sm text-muted-foreground">
            {selectedCategory
              ? categoryCopy(language, selectedCategory).description
              : tr("没有机制分类。", "No mechanics categories.")}
          </p>
          <div className="mt-4 grid gap-2 sm:grid-cols-2">
            {categoryTools.map((tool) => (
              <button
                key={tool.id}
                type="button"
                onClick={() => selectTool(tool.id)}
                className={cn(
                  "rounded-md border px-3 py-2.5 text-left transition-colors",
                  tool.id === selectedTool?.id
                    ? "border-primary bg-primary-muted/40"
                    : "border-border hover:bg-muted/40",
                )}
              >
                <span className="font-mono text-2xs text-muted-foreground">
                  {tr("阶段", "Stage")} {tool.mechanics_stage.toString().padStart(2, "0")}
                </span>
                <span className="mt-1 block text-sm font-medium">{toolLabel(language, tool)}</span>
                <span className="mt-1 block text-2xs text-muted-foreground">{toolDescription(language, tool)}</span>
              </button>
            ))}
          </div>
          <div className="mt-4 rounded-md border border-border bg-muted/30 p-3">
            {runResult ? (
              <pre className="max-h-[28rem] overflow-auto font-mono text-[12px] leading-relaxed">
                {formatJson(runResult.result)}
              </pre>
            ) : (
              <p className="text-sm text-muted-foreground">
                <strong className="block text-foreground">
                  {selectedTool ? toolLabel(language, selectedTool) : tr("先选一个工具", "Pick a tool first")}
                </strong>
                {tr(
                  "跑一次后，结果只留在这一页。需要图、热力图或交互探索时，打开对应实验室。",
                  "Results stay on this page after a run. Open the matching lab when you need charts or heatmaps.",
                )}
              </p>
            )}
          </div>
        </section>

        <aside className="min-h-0 overflow-y-auto px-4 py-4 md:px-5">
          <p className="text-2xs uppercase tracking-[0.14em] text-muted-foreground">
            {tr("检查器", "Inspector")}
          </p>
          {selectedTool ? (
            <>
              <h3 className="mt-1 text-sm font-semibold">{toolLabel(language, selectedTool)}</h3>
              <p className="mt-1 text-xs text-muted-foreground">{toolDescription(language, selectedTool)}</p>
              <label className="mt-3 block text-2xs text-muted-foreground" htmlFor="json-input">
                {tr("JSON 输入", "JSON Input")}
              </label>
              <textarea
                id="json-input"
                aria-label={tr("JSON 输入", "JSON Input")}
                value={jsonInput}
                onChange={(event) => setJsonInput(event.target.value)}
                rows={8}
                className="mt-1 w-full rounded-md border border-input bg-background px-2.5 py-2 font-mono text-[12px]"
              />
              <div className="mt-2 flex flex-wrap gap-2">
                <button
                  type="button"
                  disabled={running || exporting}
                  onClick={() => void handleRun("run")}
                  className="rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground disabled:opacity-60"
                >
                  {running ? tr("在跑…", "Running") : tr("运行", "Run")}
                </button>
                <button
                  type="button"
                  disabled={running || exporting}
                  onClick={() => void handleRun("export")}
                  className="rounded-md border border-border px-3 py-1.5 text-xs font-medium hover:bg-muted disabled:opacity-60"
                >
                  {exporting ? tr("正在导出…", "Exporting") : tr("导出", "Export")}
                </button>
                <button
                  type="button"
                  onClick={() => void handleCopyCurl()}
                  className="rounded-md border border-border px-3 py-1.5 text-xs font-medium hover:bg-muted"
                >
                  {copied ? tr("已复制", "Copied") : tr("复制 cURL", "Copy cURL")}
                </button>
              </div>
              {runError ? <p className="mt-2 text-xs text-destructive">{runError}</p> : null}
              {runResult?.artifact ? (
                <p className="mt-2 text-2xs text-muted-foreground">
                  {tr("已写出", "Wrote")} {runResult.artifact.markdown_path}
                </p>
              ) : null}
              <p className="mt-4 font-mono text-2xs text-muted-foreground">POST /api/tools/{selectedTool.id}/run</p>
              {selectedTool.page_id ? (
                <Link
                  to={`/explore/${selectedTool.page_id}`}
                  className="mt-2 inline-block text-xs font-medium text-primary-ink underline-offset-4 hover:underline"
                >
                  {tr("打开对应图表页", "Open the matching chart page")}
                </Link>
              ) : null}
              <SchemaBlock label={tr("当前载荷", "Current Payload")} value={jsonInput} raw />
              <SchemaBlock label={tr("请求 schema", "Request Schema")} value={selectedTool.input_schema} />
              <SchemaBlock label={tr("响应 schema", "Response Schema")} value={selectedTool.output_schema} />
            </>
          ) : (
            <p className="mt-2 text-sm text-muted-foreground">
              {tr("选一个工具再检查输入和 schema。", "Pick a tool to inspect its input and schema.")}
            </p>
          )}
        </aside>
      </section>
    </div>
  );
}

function SchemaBlock({
  label,
  value,
  raw,
}: {
  label: string;
  value: unknown;
  raw?: boolean;
}) {
  return (
    <section className="mt-3">
      <h4 className="text-2xs font-medium uppercase tracking-[0.12em] text-muted-foreground">{label}</h4>
      <pre className="mt-1 max-h-40 overflow-auto rounded-md bg-muted/50 px-2 py-1.5 font-mono text-[11px] leading-relaxed">
        {raw && typeof value === "string" ? value : formatJson(value)}
      </pre>
    </section>
  );
}
