"""Lab capability mapping for UI navigation and future MCP exposure."""

from __future__ import annotations

from .registry import get_registry
from .schemas import LabCapability


PAGE_LABELS = {
    "toolbox_tool_runner": ("Tool Runner", "Toolbox"),
    "token_playground": ("Playground", "TokenLab"),
    "generation_logits": ("Logits Inspector", "GenerationLab"),
    "generation_kv_cache": ("KV Cache", "GenerationLab"),
    "interpretability_rope": ("RoPE Explorer", "InterpretabilityLab"),
    "interpretability_ffn": ("FFN Activation", "InterpretabilityLab"),
    "data_dataset_viewer": ("Dataset Viewer", "DataLab"),
    "data_cleaner": ("Data Cleaner", "DataLab"),
    "data_formatter": ("Format Converter", "DataLab"),
    "rag_chunking": ("Chunking", "RAGLab"),
    "rag_retrieval": ("Retrieval", "RAGLab"),
    "model_peft": ("PEFT Calculator", "ModelLab"),
    "finetune_training_cost": ("Training Cost", "FineTuneLab"),
    "eval_pipeline": ("Eval Pipeline", "Eval Lab"),
    "agent_trace_analyzer": ("Trace Analyzer", "Agent Trace Lab"),
}


def get_lab_capabilities() -> list[LabCapability]:
    """把工具注册表聚合成页面能力声明。"""
    grouped: dict[str, list[str]] = {"toolbox_tool_runner": []}
    concepts: dict[str, set[str]] = {"toolbox_tool_runner": set()}
    for spec in get_registry().list_specs():
        grouped["toolbox_tool_runner"].append(spec.id)
        concepts["toolbox_tool_runner"].update(spec.concepts)
        if spec.page_id:
            grouped.setdefault(spec.page_id, []).append(spec.id)
            concepts.setdefault(spec.page_id, set()).update(spec.concepts)

    capabilities = []
    for page_id, tool_ids in sorted(grouped.items()):
        label, lab = PAGE_LABELS.get(page_id, (page_id, "Workbench"))
        capabilities.append(
            LabCapability(
                page_id=page_id,
                page_label=label,
                lab=lab,
                tool_ids=sorted(tool_ids),
                concepts=sorted(concepts.get(page_id, set())),
            )
        )
    return capabilities
