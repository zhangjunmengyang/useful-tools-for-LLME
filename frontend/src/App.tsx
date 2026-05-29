import { useEffect, useMemo, useRef, useState } from "react";

import { fetchTools, runTool } from "./api";
import { firstCategoryWithTools, formatJson, toolsForCategory } from "./mechanics";
import type { ToolRun, ToolSpec, ToolsPayload } from "./types";

type AppProps = {
  initialPayload?: ToolsPayload;
};

export default function App({ initialPayload }: AppProps) {
  const [payload, setPayload] = useState<ToolsPayload | null>(
    initialPayload ?? null
  );
  const [loading, setLoading] = useState(!initialPayload);
  const [error, setError] = useState<string | null>(null);
  const [selectedCategoryId, setSelectedCategoryId] = useState(() =>
    initialPayload
      ? firstCategoryWithTools(initialPayload.categories, initialPayload.tools)
      : ""
  );
  const [selectedToolId, setSelectedToolId] = useState<string>("");
  const [jsonInput, setJsonInput] = useState("{\n  \"text\": \"Ａ café\"\n}");
  const [runResult, setRunResult] = useState<ToolRun | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const runSequenceRef = useRef(0);
  const selectedToolIdRef = useRef(selectedToolId);

  useEffect(() => {
    if (initialPayload) {
      const nextCategoryId = firstCategoryWithTools(
        initialPayload.categories,
        initialPayload.tools
      );
      const nextTool = toolsForCategory(initialPayload.tools, nextCategoryId)[0];
      setPayload(initialPayload);
      setSelectedCategoryId(nextCategoryId);
      setSelectedToolId(nextTool?.id ?? "");
      setLoading(false);
      setError(null);
      return;
    }

    let active = true;
    setLoading(true);
    fetchTools()
      .then((nextPayload) => {
        if (!active) {
          return;
        }
        setPayload(nextPayload);
        setSelectedCategoryId(
          firstCategoryWithTools(nextPayload.categories, nextPayload.tools)
        );
        setError(null);
      })
      .catch((fetchError: unknown) => {
        if (!active) {
          return;
        }
        setError(
          fetchError instanceof Error
            ? fetchError.message
            : "Failed to load tools"
        );
      })
      .finally(() => {
        if (active) {
          setLoading(false);
        }
      });

    return () => {
      active = false;
    };
  }, [initialPayload]);

  const categories = payload?.categories ?? [];
  const tools = payload?.tools ?? [];
  const selectedCategory =
    categories.find((category) => category.id === selectedCategoryId) ??
    categories[0];
  const categoryTools = useMemo(
    () => toolsForCategory(tools, selectedCategory?.id ?? ""),
    [selectedCategory?.id, tools]
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

  const selectedTool =
    categoryTools.find((tool) => tool.id === selectedToolId) ??
    categoryTools[0];

  useEffect(() => {
    invalidateActiveRun(selectedTool?.id ?? "");
  }, [selectedTool?.id]);

  function selectCategory(categoryId: string) {
    setSelectedCategoryId(categoryId);
    const nextTool = toolsForCategory(tools, categoryId)[0];
    const nextToolId = nextTool?.id ?? "";
    setSelectedToolId(nextToolId);
    if (nextToolId !== selectedToolIdRef.current) {
      invalidateActiveRun(nextToolId);
    }
  }

  function selectTool(toolId: string) {
    if (toolId === selectedToolIdRef.current) {
      return;
    }
    setSelectedToolId(toolId);
    invalidateActiveRun(toolId);
  }

  function invalidateActiveRun(toolId: string) {
    runSequenceRef.current += 1;
    selectedToolIdRef.current = toolId;
    setRunResult(null);
    setRunError(null);
    setRunning(false);
  }

  async function handleRunTool() {
    if (!selectedTool) {
      return;
    }

    const toolId = selectedTool.id;
    const runSequence = runSequenceRef.current + 1;
    runSequenceRef.current = runSequence;
    selectedToolIdRef.current = toolId;
    const isActiveRun = () =>
      runSequenceRef.current === runSequence &&
      selectedToolIdRef.current === toolId;

    setRunning(true);
    setRunError(null);
    setRunResult(null);
    try {
      const inputs = JSON.parse(jsonInput) as Record<string, unknown>;
      const nextResult = await runTool(toolId, inputs);
      if (isActiveRun()) {
        setRunResult(nextResult);
      }
    } catch (toolError: unknown) {
      if (isActiveRun()) {
        setRunError(
          toolError instanceof Error ? toolError.message : "Failed to run tool"
        );
      }
    } finally {
      if (isActiveRun()) {
        setRunning(false);
      }
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Stateless API Workbench</p>
          <h1>LLM Mechanics Explorer</h1>
        </div>
        <span className="stack-pill">React + FastAPI</span>
      </header>

      {loading ? (
        <p className="notice">Loading tools from /api/tools</p>
      ) : null}
      {error ? <p className="notice error">{error}</p> : null}

      <section className="workspace-grid" aria-label="Mechanics workspace">
        <nav className="pipeline-rail" aria-label="Pipeline Rail">
          {categories.map((category) => (
            <button
              className={`rail-item ${
                category.id === selectedCategory?.id ? "active" : ""
              }`}
              key={category.id}
              type="button"
              onClick={() => selectCategory(category.id)}
            >
              <span className="rail-stage">
                {category.stage.toString().padStart(2, "0")}
              </span>
              <span className="rail-copy">
                <span className="rail-label">{category.label}</span>
                <span className="rail-subtitle">{category.subtitle}</span>
              </span>
            </button>
          ))}
        </nav>

        <section className="canvas-panel">
          <p className="eyebrow">Mechanics Canvas</p>
          <h2>{selectedCategory?.label ?? "No category selected"}</h2>
          <p className="panel-description">
            {selectedCategory?.description ?? "No mechanics categories loaded."}
          </p>

          <div className="tool-grid">
            {categoryTools.map((tool) => (
              <ToolCard
                key={tool.id}
                selected={tool.id === selectedTool?.id}
                tool={tool}
                onSelect={() => selectTool(tool.id)}
              />
            ))}
          </div>

          <div className="canvas-empty">
            {runResult ? (
              <pre>{formatJson(runResult.result)}</pre>
            ) : (
              <>
                <strong>{selectedTool?.label ?? "Select a tool"}</strong>
                <span>
                  Run the selected tool to render its mechanism output. Results
                  are kept in client state only.
                </span>
              </>
            )}
          </div>
        </section>

        <aside className="inspector-panel">
          <p className="eyebrow">Inspector</p>
          {selectedTool ? (
            <>
              <h2>{selectedTool.label}</h2>
              <p className="panel-description">{selectedTool.description}</p>
              <label className="input-label" htmlFor="json-input">
                JSON Input
              </label>
              <textarea
                id="json-input"
                aria-label="JSON Input"
                value={jsonInput}
                onChange={(event) => setJsonInput(event.target.value)}
                rows={8}
              />
              <button
                className="run-button"
                disabled={!selectedTool || running}
                onClick={handleRunTool}
                type="button"
              >
                {running ? "Running" : "Run Tool"}
              </button>
              {runError ? <div className="notice error">{runError}</div> : null}
              <div className="api-drawer">
                <p className="endpoint">
                  POST /api/tools/{selectedTool.id}/run
                </p>
                <SchemaBlock
                  label="Request Schema"
                  value={selectedTool.input_schema}
                />
                <SchemaBlock
                  label="Response Schema"
                  value={selectedTool.output_schema}
                />
              </div>
            </>
          ) : (
            <p className="panel-description">Select a tool to inspect.</p>
          )}
        </aside>
      </section>
    </main>
  );
}

function ToolCard({
  onSelect,
  selected,
  tool
}: {
  onSelect: () => void;
  selected: boolean;
  tool: ToolSpec;
}) {
  return (
    <button
      className={`tool-card ${selected ? "active" : ""}`}
      type="button"
      onClick={onSelect}
    >
      <span className="tool-stage">
        Stage {tool.mechanics_stage.toString().padStart(2, "0")}
      </span>
      <span className="tool-title">{tool.label}</span>
      <span className="tool-description">{tool.description}</span>
    </button>
  );
}

function SchemaBlock({
  label,
  value
}: {
  label: string;
  value: Record<string, unknown>;
}) {
  return (
    <section className="schema-block">
      <h3>{label}</h3>
      <pre>{formatJson(value)}</pre>
    </section>
  );
}
