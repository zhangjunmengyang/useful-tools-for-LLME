"""Mechanics Explorer taxonomy helpers."""

from __future__ import annotations

from typing import Any

from .default_configs import DEFAULT_CONFIGS
from .schemas import ToolSpec, make_json_safe


MECHANICS_CATEGORIES: list[dict[str, Any]] = [
    {
        "id": "input_tokens",
        "label": "Input & Tokens",
        "subtitle": "Text to model-ready token IDs.",
        "description": "Inspect tokenization, Unicode normalization, compression, and chat template rendering.",
        "stage": 1,
    },
    {
        "id": "representation_space",
        "label": "Representation Space",
        "subtitle": "Vectors, similarity, and latent geometry.",
        "description": "Explore embedding spaces, vector arithmetic, semantic similarity, and sparse-versus-dense behavior.",
        "stage": 2,
    },
    {
        "id": "probability_decoding",
        "label": "Probability & Decoding",
        "subtitle": "Logits to next-token decisions.",
        "description": "Inspect logits, sampling controls, top-k, top-p, temperature, and beam search behavior.",
        "stage": 3,
    },
    {
        "id": "transformer_anatomy",
        "label": "Transformer Anatomy",
        "subtitle": "Attention, RoPE, FFN, and KV cache.",
        "description": "Visualize transformer internals and inference-time memory mechanics.",
        "stage": 4,
    },
    {
        "id": "data_context",
        "label": "Data & Context",
        "subtitle": "Datasets and context before the model.",
        "description": "Inspect datasets, cleaning, formatting, chunking, and retrieval diagnostics.",
        "stage": 5,
    },
    {
        "id": "adaptation_cost",
        "label": "Adaptation & Cost",
        "subtitle": "Fine-tuning, memory, and budget.",
        "description": "Estimate LoRA parameters, training cost, model memory, and configuration differences.",
        "stage": 6,
    },
    {
        "id": "evaluation_traces",
        "label": "Evaluation & Traces",
        "subtitle": "Metrics, judges, and run behavior.",
        "description": "Evaluate predictions and inspect model or agent traces.",
        "stage": 7,
    },
]

CATEGORY_BY_ID = {category["id"]: category for category in MECHANICS_CATEGORIES}
VALID_MECHANICS_CATEGORY_IDS = set(CATEGORY_BY_ID)


def enrich_tool_spec(spec: ToolSpec) -> dict[str, Any]:
    """返回带 Mechanics Explorer 分类信息的工具定义。"""
    payload = spec.to_dict()
    payload["sample_input"] = make_json_safe(DEFAULT_CONFIGS.get(spec.id, {}))
    category = CATEGORY_BY_ID.get(spec.mechanics_category or "")
    if category:
        payload["mechanics_category_label"] = category["label"]
        payload["mechanics_category_subtitle"] = category["subtitle"]
    else:
        payload["mechanics_category_label"] = "Uncategorized"
        payload["mechanics_category_subtitle"] = "No mechanics category assigned."
    return payload
