"""Stateless HTTP API for Mechanics Explorer."""

from __future__ import annotations

from json import JSONDecodeError
import os
from pathlib import Path
import sys
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from learn_platform.labs import list_lab_pages
from learn_platform.routes import router as learn_router
from workbench_tools.mechanics import MECHANICS_CATEGORIES, enrich_tool_spec
from workbench_tools.registry import get_registry
from workbench_tools.schemas import ToolRun


app = FastAPI(
    title="Learn Platform API",
    version="0.2.0",
)
app.include_router(learn_router)
_LABS_MOUNTED = False

DEFAULT_ARTIFACT_ROOT = Path("research")
DEFAULT_FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"


def _registry():
    """返回工具注册表。"""
    return get_registry()


def _run_payload(run: ToolRun, *, include_artifact: bool = False) -> dict[str, Any]:
    """返回 API 用 ToolRun 载荷。"""
    payload = run.to_dict()
    if not include_artifact or payload.get("artifact") is None:
        payload.pop("artifact", None)
    return payload


def _request_body_error(tool_id: str) -> dict[str, Any]:
    """返回请求体格式错误。"""
    run = ToolRun(
        tool_id=tool_id,
        status="error",
        inputs={},
        result={},
        error="Request body must be a JSON object",
    )
    return _run_payload(run)


def _artifact_root() -> Path:
    """返回 HTTP API 允许写入的 artifact 根目录。"""
    configured_root = getattr(
        app.state,
        "artifact_root",
        os.environ.get("WORKBENCH_ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT),
    )
    return Path(configured_root).expanduser().resolve()


def _frontend_dist() -> Path:
    """返回可托管的 React 构建目录。"""
    configured_dist = getattr(
        app.state,
        "frontend_dist",
        os.environ.get("WORKBENCH_FRONTEND_DIST", DEFAULT_FRONTEND_DIST),
    )
    return Path(configured_dist).expanduser().resolve()


async def _read_json_object(request: Request) -> dict[str, Any] | None:
    """读取 JSON object 请求体。"""
    try:
        body = await request.json()
    except (JSONDecodeError, ValueError):
        return None
    if not isinstance(body, dict):
        return None
    return body


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
async def run_tool(tool_id: str, request: Request) -> dict[str, Any]:
    """运行一个 stateless 工具。"""
    try:
        _registry().get_spec(tool_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown tool: {tool_id}") from exc
    inputs = await _read_json_object(request)
    if inputs is None:
        return _request_body_error(tool_id)
    run = _registry().run(tool_id, inputs, export=False)
    return _run_payload(run)


@app.post("/api/tools/{tool_id}/export")
async def export_tool(tool_id: str, request: Request) -> dict[str, Any]:
    """运行工具并显式导出 artifact。"""
    try:
        _registry().get_spec(tool_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown tool: {tool_id}") from exc
    body = await _read_json_object(request)
    if body is None:
        return _request_body_error(tool_id)
    inputs = body.get("inputs", {})
    if not isinstance(inputs, dict):
        return _request_body_error(tool_id)
    run = _registry().run(tool_id, inputs, export=True, output_dir=_artifact_root())
    return _run_payload(run, include_artifact=True)


@app.get("/api/labs")
def list_labs() -> dict[str, Any]:
    """返回全部 Gradio 实验室页面，供新壳挂入口。"""
    return list_lab_pages(labs_mounted=_LABS_MOUNTED)


def _should_mount_labs() -> bool:
    """测试进程不挂 Gradio，避免拖慢 /api 用例。"""
    if os.environ.get("WORKBENCH_MOUNT_LABS") == "1":
        return True
    if os.environ.get("WORKBENCH_SKIP_LABS") == "1":
        return False
    if "pytest" in sys.modules:
        return False
    return not any(name.startswith("tests.") or name.startswith("test_") for name in sys.modules)


def _mount_labs() -> None:
    """把旧 Gradio 工作台挂到 /labs，图表页继续可达。"""
    global _LABS_MOUNTED
    if not _should_mount_labs():
        return
    import gradio as gr
    from app_gradio import create_app as create_gradio_app

    demo = create_gradio_app()
    gr.mount_gradio_app(app, demo, path="/labs")
    _LABS_MOUNTED = True


def _register_frontend() -> None:
    """前端托管放在 Gradio mount 之后，并避开 /api 与 /labs。"""

    @app.get("/")
    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str = ""):
        """托管构建后的 React 工作台，并支持前端深链。"""
        if full_path.startswith("api/") or full_path.startswith("labs"):
            raise HTTPException(status_code=404, detail="Not Found")

        dist_root = _frontend_dist()
        index_path = dist_root / "index.html"
        if not index_path.is_file():
            raise HTTPException(
                status_code=404,
                detail="Frontend build not found. Run `cd frontend && npm run build`.",
            )

        if full_path:
            candidate = (dist_root / full_path).resolve()
            if candidate.is_file() and candidate.is_relative_to(dist_root):
                return FileResponse(candidate)

        return FileResponse(index_path)


_mount_labs()
_register_frontend()
