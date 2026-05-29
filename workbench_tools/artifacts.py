"""Research artifact export helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import re

from .schemas import ToolArtifact, ToolRun, ToolSpec, make_json_safe


def safe_slug(value: str, fallback: str = "tool-run") -> str:
    """生成文件系统安全的短标识。"""
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower())
    slug = slug.strip("-._")
    return slug or fallback


def ensure_inside_output_dir(path: Path, output_dir: Path) -> Path:
    """确保写入路径不会逃出输出目录。"""
    resolved_output = output_dir.resolve()
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(resolved_output):
        raise ValueError(f"Artifact path escapes output directory: {path}")
    return resolved_path


def render_markdown_artifact(run: ToolRun, spec: ToolSpec) -> str:
    """把工具运行结果渲染成 AI 友好的 Markdown。"""
    enriched = build_artifact_payload(run, spec, None)
    lines = [
        f"# {spec.label}",
        "",
        f"- Tool ID: `{spec.id}`",
        f"- Lab: {spec.lab}",
        f"- Source Page: `{enriched['source_page']}`",
        f"- Status: {run.status}",
        f"- Started At: {run.started_at}",
        f"- Duration: {run.duration_ms:.2f} ms",
        "",
        "## Summary",
        "",
        enriched["summary"],
        "",
        "## Key Findings",
        "",
        *[f"- {finding}" for finding in enriched["key_findings"]],
        "",
        "## Description",
        "",
        spec.description,
        "",
        "## Limitations",
        "",
        *[f"- {limitation}" for limitation in enriched["limitations"]],
        "",
        "## Inputs",
        "",
        "```json",
        json.dumps(make_json_safe(run.inputs), ensure_ascii=False, indent=2, allow_nan=False),
        "```",
        "",
        "## Result",
        "",
        "```json",
        json.dumps(make_json_safe(run.result), ensure_ascii=False, indent=2, allow_nan=False),
        "```",
        "",
        "## Reproduce",
        "",
        "```bash",
        enriched["reproduce_command"],
        "```",
    ]
    if run.error:
        lines.extend(["", "## Error", "", f"```text\n{run.error}\n```"])
    return "\n".join(lines) + "\n"


def build_artifact_payload(
    run: ToolRun,
    spec: ToolSpec,
    output_dir: str | Path | None,
) -> dict:
    """生成 Markdown 和 JSON 共享的研究证据字段。"""
    key_findings = []
    for key, value in run.result.items():
        if isinstance(value, (str, int, float, bool)):
            key_findings.append(f"{key}: {value}")
        if len(key_findings) >= 6:
            break
    if not key_findings:
        key_findings = ["No scalar findings were produced; inspect the structured result payload."]

    limitations = [
        "This artifact records tool output for the provided inputs; research interpretation remains explicit user work.",
    ]
    if spec.requires_model_download:
        limitations.append("This tool may depend on locally cached or downloadable model artifacts.")

    output_hint = str(output_dir or "research")
    return {
        "summary": f"{spec.label} finished with status `{run.status}` for tool `{spec.id}`.",
        "key_findings": key_findings,
        "limitations": limitations,
        "reproduce_command": (
            f"python -m workbench_tools run {spec.id} "
            f"--config <config.json> --output-dir {output_hint}"
        ),
        "source_page": spec.page_id or "",
        "model_download_required": spec.requires_model_download,
    }


def write_tool_artifact(
    run: ToolRun,
    spec: ToolSpec,
    output_dir: str | Path = "research",
) -> ToolArtifact:
    """写入 Markdown + JSON 研究产物。"""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_slug = f"{timestamp}-{safe_slug(spec.id)}"
    markdown_path = ensure_inside_output_dir(root / f"{run_slug}.md", root)
    json_path = ensure_inside_output_dir(root / f"{run_slug}.json", root)

    enriched = build_artifact_payload(run, spec, root)
    payload = {
        "spec": spec.to_dict(),
        "tool_id": run.tool_id,
        "status": run.status,
        "inputs": make_json_safe(run.inputs),
        "result": make_json_safe(run.result),
        "duration_ms": run.duration_ms,
        "error": run.error,
        "started_at": run.started_at,
        **enriched,
    }

    markdown_path.write_text(render_markdown_artifact(run, spec), encoding="utf-8")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return ToolArtifact(
        markdown_path=str(markdown_path),
        json_path=str(json_path),
    )
