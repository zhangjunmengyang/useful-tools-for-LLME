import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { exportTool, fetchTools, runTool } from "../api";
import { useI18n } from "./I18nProvider";
import { fieldLabel, toolDescription, toolLabel } from "../lib/catalog-i18n";
import { buildExampleInput, copyCurlCommand, formatJson } from "../lib/mechanics";
import { cn } from "../lib/utils";
import type { ToolRun, ToolSpec } from "../types";

type Field = { key: string; label: string; kind: "text" | "number" | "json" };

const FORMS: Record<string, Field[]> = {
  unicode_analyze: [{ key: "text", label: "文本", kind: "text" }],
  tokenizer_encode: [
    { key: "model_name", label: "词表 / 模型名", kind: "text" },
    { key: "text", label: "文本", kind: "text" },
  ],
  vector_similarity: [
    { key: "vectors", label: "向量 (JSON)", kind: "json" },
    { key: "labels", label: "标签 (JSON)", kind: "json" },
  ],
  rope_frequencies: [
    { key: "dim", label: "维数 (偶数)", kind: "number" },
    { key: "max_position", label: "最大位置", kind: "number" },
    { key: "max_distance", label: "最大距离", kind: "number" },
    { key: "base", label: "base", kind: "number" },
  ],
  ffn_activation_compare: [{ key: "x_values", label: "x 取值 (JSON)", kind: "json" }],
  sampling_distribution: [
    { key: "logits", label: "logits (JSON)", kind: "json" },
    { key: "tokens", label: "token (JSON)", kind: "json" },
    { key: "temperature", label: "温度", kind: "number" },
    { key: "top_k", label: "top-k", kind: "number" },
    { key: "top_p", label: "top-p", kind: "number" },
  ],
  kv_cache_growth: [
    { key: "prompt_length", label: "提示长度", kind: "number" },
    { key: "generation_length", label: "生成长度", kind: "number" },
    { key: "num_layers", label: "层数", kind: "number" },
    { key: "hidden_size", label: "隐层维", kind: "number" },
  ],
  kv_cache_estimate: [
    { key: "num_layers", label: "层数", kind: "number" },
    { key: "hidden_size", label: "隐层维", kind: "number" },
    { key: "seq_length", label: "序列长度", kind: "number" },
  ],
  lora_params_estimate: [
    { key: "hidden_size", label: "隐层维", kind: "number" },
    { key: "num_layers", label: "层数", kind: "number" },
    { key: "rank", label: "秩", kind: "number" },
    { key: "target_modules", label: "目标模块 (JSON)", kind: "json" },
    { key: "intermediate_size", label: "中间层维", kind: "number" },
    { key: "num_heads", label: "头数", kind: "number" },
  ],
  training_cost_estimate: [
    { key: "model_params", label: "参数量", kind: "number" },
    { key: "tokens", label: "token 数", kind: "number" },
    { key: "gpu_tflops", label: "GPU TFLOPS", kind: "number" },
    { key: "cost_per_hour", label: "每小时费用", kind: "number" },
    { key: "mfu", label: "MFU", kind: "number" },
  ],
  data_clean: [
    { key: "text", label: "文本", kind: "text" },
    { key: "rules", label: "规则 (JSON)", kind: "json" },
  ],
  dataset_quality_check: [
    { key: "samples", label: "样本 (JSON)", kind: "json" },
    { key: "text_fields", label: "文本字段 (JSON)", kind: "json" },
  ],
  instruct_format: [
    { key: "data", label: "数据 (JSON)", kind: "json" },
    { key: "target_format", label: "目标格式", kind: "text" },
  ],
  rag_chunk: [
    { key: "text", label: "文本", kind: "text" },
    { key: "method", label: "切块方法", kind: "text" },
    { key: "chunk_size", label: "块大小", kind: "number" },
    { key: "overlap", label: "重叠", kind: "number" },
  ],
  rag_lexical_retrieval: [
    { key: "query", label: "查询", kind: "text" },
    { key: "documents", label: "文档 (JSON)", kind: "json" },
    { key: "top_k", label: "top-k", kind: "number" },
  ],
  eval_metrics: [
    { key: "predictions", label: "预测 (JSON)", kind: "json" },
    { key: "references", label: "参考 (JSON)", kind: "json" },
  ],
  trace_analyze: [{ key: "trace_json", label: "轨迹 JSON", kind: "text" }],
};

function stringifyValue(value: unknown): string {
  if (typeof value === "string") return value;
  if (value == null) return "";
  return JSON.stringify(value);
}

function parseField(field: Field, raw: string): unknown {
  if (field.kind === "number") {
    const n = Number(raw);
    return Number.isFinite(n) ? n : raw;
  }
  if (field.kind === "json") return JSON.parse(raw);
  return raw;
}

export function ToolWidget({ toolId }: { toolId: string }) {
  const { language, tr } = useI18n();
  const [spec, setSpec] = useState<ToolSpec | null>(null);
  const [values, setValues] = useState<Record<string, string>>({});
  const [run, setRun] = useState<ToolRun | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetchTools()
      .then((payload) => {
        if (cancelled) return;
        const found = payload.tools.find((item) => item.id === toolId) ?? null;
        setSpec(found);
        const sample = found?.sample_input ?? {};
        const next: Record<string, string> = {};
        for (const field of FORMS[toolId] ?? []) {
          next[field.key] = stringifyValue(sample[field.key]);
        }
        if (!(toolId in FORMS)) {
          next.__json = found ? buildExampleInput(found) : "{}";
        }
        setValues(next);
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [toolId]);

  const fields = FORMS[toolId];

  function currentInputs(): Record<string, unknown> {
    if (fields) {
      const inputs: Record<string, unknown> = {};
      for (const field of fields) {
        inputs[field.key] = parseField(field, values[field.key] ?? "");
      }
      return inputs;
    }
    return JSON.parse(values.__json || "{}") as Record<string, unknown>;
  }

  async function onAction(kind: "run" | "export") {
    setBusy(true);
    setError("");
    try {
      const inputs = currentInputs();
      setRun(kind === "export" ? await exportTool(toolId, inputs) : await runTool(toolId, inputs));
    } catch (err) {
      setError(err instanceof Error ? err.message : tr("输入不是合法 JSON", "Input is not valid JSON"));
    } finally {
      setBusy(false);
    }
  }

  async function onCopyCurl() {
    try {
      await copyCurlCommand(toolId, formatJson(currentInputs()));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch (err) {
      setError(err instanceof Error ? err.message : tr("复制失败", "Copy failed"));
    }
  }

  return (
    <section className="rounded-md border border-border bg-card p-4">
      <header className="mb-3 flex items-baseline justify-between gap-3">
        <div>
          <h3 className="text-[1.02rem] font-semibold tracking-tight">
            {spec ? toolLabel(language, spec) : toolId}
          </h3>
          <p className="mt-1 text-sm text-muted-foreground">
            {spec
              ? toolDescription(language, spec)
              : tr("现成工具，跑的是 /api/tools。", "A live tool. It hits /api/tools.")}
          </p>
        </div>
        {spec?.requires_model_download ? (
          <p className="shrink-0 text-xs text-muted-foreground">
            {tr("首次可能下载词表", "First run may download a vocab")}
          </p>
        ) : null}
      </header>
      <div className="grid gap-3">
        {fields ? (
          fields.map((field) => {
            const long = field.kind === "text" || (field.kind === "json" && (values[field.key] ?? "").length > 40);
            return (
              <label key={field.key} className="grid gap-1 text-sm">
                <span className="text-muted-foreground">{fieldLabel(language, field.key, field.label)}</span>
                {long ? (
                  <textarea
                    className="min-h-20 w-full rounded-md border border-input bg-background px-2.5 py-2 font-mono text-[13px]"
                    value={values[field.key] ?? ""}
                    onChange={(event) => setValues((prev) => ({ ...prev, [field.key]: event.target.value }))}
                  />
                ) : (
                  <input
                    className="w-full rounded-md border border-input bg-background px-2.5 py-2 font-mono text-[13px]"
                    value={values[field.key] ?? ""}
                    onChange={(event) => setValues((prev) => ({ ...prev, [field.key]: event.target.value }))}
                  />
                )}
              </label>
            );
          })
        ) : (
          <label className="grid gap-1 text-sm">
            <span className="text-muted-foreground">{tr("输入 JSON", "JSON input")}</span>
            <textarea
              className="min-h-32 w-full rounded-md border border-input bg-background px-2.5 py-2 font-mono text-[13px]"
              value={values.__json ?? "{}"}
              onChange={(event) => setValues((prev) => ({ ...prev, __json: event.target.value }))}
            />
          </label>
        )}
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() => void onAction("run")}
          disabled={busy}
          className={cn("rounded-md bg-primary px-3 py-1.5 text-sm text-primary-foreground", busy && "opacity-60")}
        >
          {busy ? tr("在跑…", "Running…") : tr("运行", "Run")}
        </button>
        <button
          type="button"
          onClick={() => void onAction("export")}
          disabled={busy}
          className="rounded-md border border-border px-3 py-1.5 text-sm hover:bg-muted disabled:opacity-60"
        >
          {tr("导出", "Export")}
        </button>
        <button
          type="button"
          onClick={() => void onCopyCurl()}
          className="rounded-md border border-border px-3 py-1.5 text-sm hover:bg-muted"
        >
          {copied ? tr("已复制", "Copied") : tr("复制 cURL", "Copy cURL")}
        </button>
        {spec?.page_id ? (
          <Link to={`/explore/${spec.page_id}`} className="text-xs font-medium text-primary-ink underline-offset-4 hover:underline">
            {tr("打开图表页", "Open chart page")}
          </Link>
        ) : null}
        {run ? <span className="text-xs text-muted-foreground">{Math.round(run.duration_ms)} ms</span> : null}
      </div>
      {error ? <p className="mt-3 text-sm text-destructive">{error}</p> : null}
      {run?.status === "error" ? <p className="mt-3 text-sm text-destructive">{run.error}</p> : null}
      {run?.artifact ? (
        <p className="mt-3 text-xs text-muted-foreground">
          {tr("已写出", "Wrote")} {run.artifact.markdown_path}
        </p>
      ) : null}
      {run?.status === "success" ? (
        <pre className="mt-3 max-h-80 overflow-auto rounded-md bg-muted px-3 py-2 font-mono text-[12px] leading-relaxed">
          {JSON.stringify(run.result, null, 2)}
        </pre>
      ) : null}
    </section>
  );
}
