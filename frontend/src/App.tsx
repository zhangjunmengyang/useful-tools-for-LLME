import { useEffect, useMemo, useRef, useState } from "react";

import { fetchTools, runTool } from "./api";
import { firstCategoryWithTools, formatJson, toolsForCategory } from "./mechanics";
import type { ToolRun, ToolSpec, ToolsPayload } from "./types";

type AppProps = {
  initialPayload?: ToolsPayload;
};

const TOOL_INPUT_EXAMPLES: Record<string, Record<string, unknown>> = {
  data_clean: {
    text: "<p>LLM tools   normalize text.</p> https://example.com",
    rules: ["html", "url", "whitespace"],
    unicode_form: "NFC"
  },
  dataset_quality_check: {
    samples: [
      { instruction: "Summarize the passage", output: "A concise summary." },
      { instruction: "Summarize the passage", output: "A concise summary." },
      { instruction: "", output: "Missing instruction example." }
    ],
    text_fields: ["instruction", "output"]
  },
  eval_metrics: {
    predictions: ["The model explains tokenization."],
    references: ["The answer explains tokenization."]
  },
  ffn_activation_compare: {
    x_values: [-2, -1, 0, 1, 2]
  },
  instruct_format: {
    data: {
      instruction: "Explain KV cache in one paragraph.",
      input: "",
      output: "KV cache stores attention keys and values for reuse."
    },
    target_format: "chatml",
    system_prompt: "You are a precise LLM systems tutor."
  },
  kv_cache_estimate: {
    num_layers: 32,
    hidden_size: 4096,
    num_heads: 32,
    seq_length: 2048,
    batch_size: 1,
    dtype_bytes: 2
  },
  kv_cache_growth: {
    prompt_length: 1024,
    generation_length: 128,
    num_layers: 32,
    hidden_size: 4096,
    num_heads: 32,
    batch_size: 1,
    dtype_bytes: 2
  },
  lora_params_estimate: {
    hidden_size: 4096,
    num_layers: 32,
    num_heads: 32,
    intermediate_size: 11008,
    rank: 8,
    target_modules: ["q_proj", "v_proj"],
    base_params: 7000000000,
    use_quantization: true,
    quantization_bits: 4
  },
  rag_chunk: {
    text: "Retrieval augmented generation works best when source text is chunked into coherent passages.",
    method: "recursive",
    chunk_size: 80,
    overlap: 16
  },
  rag_lexical_retrieval: {
    query: "tokenization unicode normalization",
    documents: [
      "Tokenizer diagnostics reveal byte fallback and Unicode behavior.",
      "KV cache memory grows during decode.",
      "Dataset cleaning removes markup and URLs."
    ],
    top_k: 2
  },
  rope_frequencies: {
    dim: 64,
    max_position: 128,
    max_distance: 32,
    base: 10000
  },
  sampling_distribution: {
    logits: [4.2, 2.1, 1.3, 0.4],
    tokens: [" token", " word", " byte", " id"],
    temperature: 0.8,
    top_k: 3,
    top_p: 0.95
  },
  tokenizer_encode: {
    model_name: "gpt2",
    text: "Ａ café"
  },
  trace_analyze: {
    trace_json: JSON.stringify(
      [
        {
          event_type: "tool_call",
          agent_name: "researcher",
          action: "retrieve",
          start_time: "2026-05-29T00:00:00Z",
          end_time: "2026-05-29T00:00:01Z"
        }
      ],
      null,
      2
    )
  },
  training_cost_estimate: {
    model_params: 7000000000,
    tokens: 100000000,
    gpu_tflops: 312,
    cost_per_hour: 2.5,
    num_gpus: 8,
    mfu: 0.45,
    is_full_finetune: false
  },
  unicode_analyze: {
    text: "Ａ café"
  },
  vector_similarity: {
    vectors: [
      [1, 0, 0],
      [0.9, 0.1, 0],
      [0, 1, 0]
    ],
    labels: ["query", "near", "far"]
  }
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
  const [jsonInput, setJsonInput] = useState("{}");
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
    setJsonInput(buildExampleInput(selectedTool));
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
        if (nextResult.status === "error") {
          setRunError(nextResult.error ?? "Tool returned an error");
          setRunResult(null);
        } else {
          setRunResult(nextResult);
        }
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
                <button
                  className="copy-button"
                  onClick={() => copyCurlCommand(selectedTool.id, jsonInput)}
                  type="button"
                >
                  Copy cURL
                </button>
                <CodeBlock label="Current Payload" value={jsonInput} />
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

function buildExampleInput(tool?: ToolSpec): string {
  if (!tool) {
    return "{}";
  }
  return formatJson(TOOL_INPUT_EXAMPLES[tool.id] ?? buildSchemaExample(tool));
}

function copyCurlCommand(toolId: string, jsonInput: string) {
  const curl = [
    "curl -X POST",
    shellQuote(`/api/tools/${toolId}/run`),
    "-H 'Content-Type: application/json'",
    `-d ${shellQuote(jsonInput)}`
  ].join(" ");
  void navigator.clipboard?.writeText(curl);
}

function shellQuote(value: string): string {
  return `'${value.replace(/'/g, "'\\''")}'`;
}

function buildSchemaExample(tool: ToolSpec): Record<string, unknown> {
  const schema = tool.input_schema;
  const properties = asRecord(schema.properties);
  const required = Array.isArray(schema.required)
    ? schema.required.filter((key): key is string => typeof key === "string")
    : Object.keys(properties).slice(0, 4);
  const example: Record<string, unknown> = {};

  for (const key of required) {
    example[key] = exampleValueForProperty(key, asRecord(properties[key]));
  }

  return example;
}

function exampleValueForProperty(
  key: string,
  property: Record<string, unknown>
): unknown {
  const enumValues = Array.isArray(property.enum) ? property.enum : [];
  if (enumValues.length > 0) {
    return enumValues[0];
  }

  switch (property.type) {
    case "array":
      return [exampleValueForProperty(key, asRecord(property.items))];
    case "boolean":
      return false;
    case "integer":
      return 1;
    case "number":
      return 1;
    case "object":
      return {};
    case "string":
    default:
      if (key.includes("model")) {
        return "gpt2";
      }
      if (key.includes("text")) {
        return "Ａ café";
      }
      return "example";
  }
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
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

function CodeBlock({ label, value }: { label: string; value: string }) {
  return (
    <section className="schema-block">
      <h3>{label}</h3>
      <pre>{value}</pre>
    </section>
  );
}
