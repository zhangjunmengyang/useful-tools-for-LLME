"""Tool registry shared by Gradio UI, CLI, and future MCP adapters."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

from .artifacts import write_tool_artifact
from .builtin_tools import BUILTIN_TOOLS
from .schemas import ToolRun, ToolSpec, make_json_safe
from .validation import validate_against_schema


ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


class ToolRegistry:
    """保存研究工具定义并统一执行。"""

    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolSpec, ToolHandler]] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        """注册一个工具。"""
        if spec.id in self._tools:
            raise ValueError(f"Tool already registered: {spec.id}")
        self._tools[spec.id] = (spec, handler)

    def list_specs(self) -> list[ToolSpec]:
        """按工具 id 返回所有工具定义。"""
        return [self._tools[key][0] for key in sorted(self._tools)]

    def get_spec(self, tool_id: str) -> ToolSpec:
        """返回工具定义。"""
        if tool_id not in self._tools:
            raise KeyError(f"Unknown tool: {tool_id}")
        return self._tools[tool_id][0]

    def run(
        self,
        tool_id: str,
        inputs: dict[str, Any],
        *,
        export: bool = False,
        output_dir: str | Path = "research",
    ) -> ToolRun:
        """执行工具并可选导出研究产物。"""
        if tool_id not in self._tools:
            raise KeyError(f"Unknown tool: {tool_id}")

        spec, handler = self._tools[tool_id]
        safe_inputs = make_json_safe(inputs)
        started = perf_counter()
        validation_errors = validate_against_schema(safe_inputs, spec.input_schema)
        if validation_errors:
            result = {}
            status = "error"
            error = "; ".join(validation_errors)
        else:
            try:
                result = make_json_safe(handler(dict(inputs)))
                status = "success"
                error = None
            except Exception as exc:
                result = {}
                status = "error"
                error = str(exc)

        run = ToolRun(
            tool_id=tool_id,
            status=status,
            inputs=safe_inputs,
            result=result,
            duration_ms=(perf_counter() - started) * 1000,
            error=error,
        )

        if export:
            run.artifact = write_tool_artifact(run, spec, output_dir)
        return run


_REGISTRY: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """返回包含内置工具的单例 registry。"""
    global _REGISTRY
    if _REGISTRY is None:
        registry = ToolRegistry()
        for spec, handler in BUILTIN_TOOLS:
            registry.register(spec, handler)
        _REGISTRY = registry
    return _REGISTRY
