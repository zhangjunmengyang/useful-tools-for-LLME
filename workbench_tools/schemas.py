"""研究工具箱的稳定数据结构。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import math


def utc_now_iso() -> str:
    """生成稳定的 UTC 时间戳。"""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_json_safe(value: Any) -> Any:
    """把常见 Python / numpy 对象转换为 JSON 兼容结构。"""
    if is_dataclass(value):
        return make_json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return make_json_safe(value.tolist())
        except (TypeError, ValueError):
            pass
    return value


@dataclass(slots=True)
class ToolSpec:
    """描述一个可由 UI、CLI 或外部 Agent 调用的研究工具。"""

    id: str
    label: str
    description: str
    lab: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    concepts: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    requires_model_download: bool = False
    page_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """返回 JSON 兼容字典。"""
        return make_json_safe(self)


@dataclass(slots=True)
class ToolArtifact:
    """一次工具运行导出的研究产物。"""

    markdown_path: str
    json_path: str

    def to_dict(self) -> dict[str, Any]:
        """返回 JSON 兼容字典。"""
        return make_json_safe(self)


@dataclass(slots=True)
class ToolRun:
    """一次工具运行的结构化结果。"""

    tool_id: str
    status: str
    inputs: dict[str, Any]
    result: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    error: str | None = None
    artifact: ToolArtifact | None = None
    started_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        """返回 JSON 兼容字典。"""
        return make_json_safe(self)


@dataclass(slots=True)
class LabCapability:
    """把 Gradio 页面映射到可复用工具能力。"""

    page_id: str
    page_label: str
    lab: str
    tool_ids: list[str]
    concepts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """返回 JSON 兼容字典。"""
        return make_json_safe(self)
