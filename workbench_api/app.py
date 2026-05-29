"""Stateless HTTP API for Mechanics Explorer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from workbench_tools.mechanics import MECHANICS_CATEGORIES, enrich_tool_spec
from workbench_tools.registry import get_registry


class ExportRequest(BaseModel):
    """显式 artifact 导出请求。"""

    inputs: dict[str, Any] = Field(default_factory=dict)
    output_dir: str = "research"


app = FastAPI(
    title="LLM Tools Workbench API",
    version="0.1.0",
)


def _registry():
    """返回工具注册表。"""
    return get_registry()


@app.get("/api/health")
def health() -> dict[str, str]:
    """健康检查。"""
    return {"status": "ok"}


@app.get("/api/tools")
def list_tools() -> dict[str, Any]:
    """返回 Mechanics Explorer 分类和工具元数据。"""
    specs = sorted(
        _registry().list_specs(),
        key=lambda spec: (spec.mechanics_stage, spec.label.lower()),
    )
    return {
        "categories": MECHANICS_CATEGORIES,
        "tools": [enrich_tool_spec(spec) for spec in specs],
    }


@app.get("/api/tools/{tool_id}")
def inspect_tool(tool_id: str) -> dict[str, Any]:
    """返回单个工具元数据。"""
    try:
        spec = _registry().get_spec(tool_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown tool: {tool_id}") from exc
    return enrich_tool_spec(spec)


@app.post("/api/tools/{tool_id}/run")
def run_tool(tool_id: str, inputs: dict[str, Any]) -> dict[str, Any]:
    """运行一个 stateless 工具。"""
    try:
        _registry().get_spec(tool_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown tool: {tool_id}") from exc
    run = _registry().run(tool_id, inputs, export=False)
    return run.to_dict()


@app.post("/api/tools/{tool_id}/export")
def export_tool(tool_id: str, request: ExportRequest) -> dict[str, Any]:
    """运行工具并显式导出 artifact。"""
    try:
        _registry().get_spec(tool_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown tool: {tool_id}") from exc
    output_dir = Path(request.output_dir)
    run = _registry().run(tool_id, request.inputs, export=True, output_dir=output_dir)
    return run.to_dict()
