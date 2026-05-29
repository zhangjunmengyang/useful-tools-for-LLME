"""Built-in research tools backed by existing Lab utility functions."""

from __future__ import annotations

from collections import Counter
import json
import re
from typing import Any, Callable

from .schemas import ToolSpec, make_json_safe


ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


def _string_list(inputs: dict[str, Any], key: str) -> list[str]:
    """读取字符串列表参数。"""
    value = inputs.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"`{key}` must be a list of strings")
    return value


def _number_list(inputs: dict[str, Any], key: str) -> list[float]:
    """读取数值列表参数。"""
    value = inputs.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"`{key}` must be a non-empty list of numbers")
    numbers = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"`{key}` must be a non-empty list of numbers")
        numbers.append(float(item))
    return numbers


def _lexical_terms(text: str) -> list[str]:
    """提取轻量词法检索用的 term。"""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def run_eval_metrics(inputs: dict[str, Any]) -> dict[str, Any]:
    """运行文本生成评测指标。"""
    from eval_lab.eval_utils import calculate_all_metrics

    predictions = _string_list(inputs, "predictions")
    references = _string_list(inputs, "references")
    if len(predictions) != len(references):
        raise ValueError("`predictions` and `references` must have the same length")
    return make_json_safe(calculate_all_metrics(predictions, references))


def run_tokenizer_encode(inputs: dict[str, Any]) -> dict[str, Any]:
    """运行 tokenizer 编码分析。"""
    from token_lab.tokenizer_utils import (
        calculate_compression_stats,
        get_token_info,
        get_tokenizer_info,
        load_tokenizer,
    )

    model_name = str(inputs["model_name"])
    text = str(inputs["text"])
    tokenizer = load_tokenizer(model_name)
    if tokenizer is None:
        raise ValueError(f"Failed to load tokenizer: {model_name}")
    token_info = get_token_info(tokenizer, text)
    return make_json_safe(
        {
            "model_name": model_name,
            "token_count": len(token_info),
            "tokens": token_info,
            "compression": calculate_compression_stats(text, len(token_info)),
            "tokenizer": get_tokenizer_info(tokenizer),
        }
    )


def run_unicode_analyze(inputs: dict[str, Any]) -> dict[str, Any]:
    """分析 Unicode 字符、字节与规范化差异。"""
    from token_lab.tokenizer_utils import get_normalization_info, get_unicode_info

    text = str(inputs["text"])
    normalization = get_normalization_info(text)
    normalization["nfkc_equal"] = text == normalization.get("NFKC")
    normalization["nfkd_equal"] = text == normalization.get("NFKD")
    characters = [get_unicode_info(char) for char in text]
    category_counts = Counter(item.get("category", "Unknown") for item in characters)
    return make_json_safe(
        {
            "text": text,
            "char_count": len(text),
            "byte_count": len(text.encode("utf-8")),
            "normalization": normalization,
            "characters": characters,
            "category_counts": dict(category_counts),
        }
    )


def run_sampling_distribution(inputs: dict[str, Any]) -> dict[str, Any]:
    """对一组 logits 应用 temperature/top-k/top-p。"""
    import torch
    from generation_lab.generation_utils import get_sampling_distribution

    logits = inputs["logits"]
    if not isinstance(logits, list) or not logits:
        raise ValueError("`logits` must be a non-empty list of numbers")
    tokens = inputs.get("tokens") or [str(index) for index in range(len(logits))]
    if not isinstance(tokens, list) or len(tokens) != len(logits):
        raise ValueError("`tokens` must be a list with the same length as `logits`")

    processed, probabilities = get_sampling_distribution(
        torch.tensor(logits, dtype=torch.float32),
        temperature=float(inputs.get("temperature", 1.0)),
        top_k=int(inputs.get("top_k", 0)),
        top_p=float(inputs.get("top_p", 1.0)),
    )
    distribution = []
    for index, token in enumerate(tokens):
        distribution.append(
            {
                "index": index,
                "token": token,
                "logit": float(logits[index]),
                "processed_logit": float(processed[index].item()),
                "probability": float(probabilities[index].item()),
            }
        )
    distribution.sort(key=lambda item: item["probability"], reverse=True)
    return make_json_safe({"distribution": distribution})


def run_kv_cache_growth(inputs: dict[str, Any]) -> dict[str, Any]:
    """模拟 prefill/decode 过程中的 KV Cache 增长曲线。"""
    from generation_lab.generation_utils import simulate_kv_cache_growth

    config = {
        "num_hidden_layers": int(inputs.get("num_layers", 32)),
        "hidden_size": int(inputs.get("hidden_size", 4096)),
        "num_attention_heads": int(inputs.get("num_heads", 32)),
    }
    steps = simulate_kv_cache_growth(
        config,
        prompt_length=int(inputs["prompt_length"]),
        generation_length=int(inputs["generation_length"]),
        batch_size=int(inputs.get("batch_size", 1)),
        dtype_bytes=int(inputs.get("dtype_bytes", 2)),
    )
    final_cache_gb = steps[-1]["cache_gb"] if steps else 0
    peak_delta_gb = max((step["delta_gb"] for step in steps), default=0)
    return make_json_safe(
        {
            "config": config,
            "steps": steps,
            "final_cache_gb": final_cache_gb,
            "peak_delta_gb": peak_delta_gb,
        }
    )


def run_rope_frequencies(inputs: dict[str, Any]) -> dict[str, Any]:
    """计算 RoPE 频率矩阵与相对距离衰减样本。"""
    from interpretability_lab.interpretability_utils import (
        compute_rope_decay,
        compute_rope_frequencies,
    )

    dim = int(inputs.get("dim", 64))
    if dim <= 0 or dim % 2 != 0:
        raise ValueError("`dim` must be a positive even integer")
    max_position = int(inputs.get("max_position", 128))
    max_distance = int(inputs.get("max_distance", min(32, max_position)))
    base = float(inputs.get("base", 10000.0))
    freqs, positions = compute_rope_frequencies(dim, max_position=max_position, base=base)
    decay = compute_rope_decay(dim, max_distance=max_distance, base=base)
    return make_json_safe(
        {
            "dim": dim,
            "base": base,
            "freq_shape": list(freqs.shape),
            "positions": positions[: min(16, len(positions))],
            "frequency_sample": freqs[: min(8, freqs.shape[0]), : min(8, freqs.shape[1])],
            "decay": [
                {"distance": index, "dot_product": value}
                for index, value in enumerate(decay[: min(32, len(decay))])
            ],
        }
    )


def run_ffn_activation_compare(inputs: dict[str, Any]) -> dict[str, Any]:
    """比较常见 FFN 激活函数的数值曲线。"""
    import numpy as np
    from interpretability_lab.interpretability_utils import compare_activation_functions

    if "x_values" in inputs:
        x_values = _number_list(inputs, "x_values")
    else:
        x_values = np.linspace(-4.0, 4.0, 81).tolist()
    x_range = np.array(x_values, dtype=np.float32)
    activations = compare_activation_functions(x_range)
    summary = {
        name: {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
        }
        for name, values in activations.items()
    }
    return make_json_safe(
        {
            "x_values": x_values,
            "activations": activations,
            "summary": summary,
        }
    )


def run_data_clean(inputs: dict[str, Any]) -> dict[str, Any]:
    """运行数据清洗规则。"""
    from data_lab.data_utils import CLEANING_RULES, clean_text, normalize_unicode

    aliases = {
        "html": "remove_html",
        "url": "remove_urls",
        "urls": "remove_urls",
        "email": "remove_emails",
        "emails": "remove_emails",
        "whitespace": "normalize_whitespace",
        "special": "remove_special_chars",
    }
    rules = [aliases.get(str(rule), str(rule)) for rule in inputs.get("rules", [])]
    invalid = [rule for rule in rules if rule not in CLEANING_RULES]
    if invalid:
        raise ValueError(f"Unknown cleaning rules: {invalid}")
    normalized = normalize_unicode(str(inputs["text"]), str(inputs.get("unicode_form", "NFC")))
    cleaned = clean_text(normalized, rules)
    return make_json_safe(
        {
            "cleaned_text": cleaned,
            "original_length": len(str(inputs["text"])),
            "cleaned_length": len(cleaned),
            "rules": rules,
        }
    )


def run_dataset_quality_check(inputs: dict[str, Any]) -> dict[str, Any]:
    """检查数据样本重复、空字段与文本长度分布。"""
    samples = inputs["samples"]
    if not isinstance(samples, list) or not all(isinstance(item, dict) for item in samples):
        raise ValueError("`samples` must be a list of objects")

    raw_fields = inputs.get("text_fields")
    if raw_fields is None:
        text_fields = sorted(
            {
                key
                for sample in samples
                for key, value in sample.items()
                if isinstance(value, str)
            }
        )
    elif isinstance(raw_fields, list) and all(isinstance(item, str) for item in raw_fields):
        text_fields = raw_fields
    else:
        raise ValueError("`text_fields` must be a list of strings")

    serialized_rows = [json.dumps(sample, ensure_ascii=False, sort_keys=True) for sample in samples]
    row_counts = Counter(serialized_rows)
    duplicate_count = sum(count - 1 for count in row_counts.values() if count > 1)
    seen: set[str] = set()
    problem_samples = []
    empty_field_count = 0
    field_lengths: dict[str, list[int]] = {field: [] for field in text_fields}

    for index, sample in enumerate(samples):
        issues = []
        row_key = serialized_rows[index]
        if row_key in seen:
            issues.append("duplicate_row")
        seen.add(row_key)

        for field in text_fields:
            value = sample.get(field)
            text_value = "" if value is None else str(value)
            field_lengths[field].append(len(text_value))
            if field not in sample:
                empty_field_count += 1
                issues.append(f"missing:{field}")
            elif not text_value.strip():
                empty_field_count += 1
                issues.append(f"empty:{field}")

        if issues:
            problem_samples.append({"index": index, "issues": issues, "sample": sample})

    length_stats = {}
    for field, lengths in field_lengths.items():
        if not lengths:
            length_stats[field] = {"min": 0, "max": 0, "mean": 0}
            continue
        length_stats[field] = {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
        }

    return make_json_safe(
        {
            "total_samples": len(samples),
            "duplicate_count": duplicate_count,
            "empty_field_count": empty_field_count,
            "text_fields": text_fields,
            "field_lengths": length_stats,
            "problem_samples": problem_samples[:50],
        }
    )


def run_instruct_format(inputs: dict[str, Any]) -> dict[str, Any]:
    """转换 SFT 指令数据格式。"""
    from data_lab.data_utils import convert_to_format

    data = inputs["data"]
    if not isinstance(data, dict):
        raise ValueError("`data` must be an object")
    target_format = str(inputs["target_format"])
    formatted = convert_to_format(data, target_format, str(inputs.get("system_prompt", "")))
    return make_json_safe({"formatted": formatted, "target_format": target_format})


def run_kv_cache_estimate(inputs: dict[str, Any]) -> dict[str, Any]:
    """估算 KV Cache 显存占用。"""
    from generation_lab.generation_utils import calculate_kv_cache_size

    params = {
        "num_layers": int(inputs.get("num_layers", 32)),
        "hidden_size": int(inputs.get("hidden_size", 4096)),
        "num_heads": int(inputs.get("num_heads", 32)),
        "seq_length": int(inputs.get("seq_length", 2048)),
        "batch_size": int(inputs.get("batch_size", 1)),
        "dtype_bytes": int(inputs.get("dtype_bytes", 2)),
    }
    return make_json_safe(calculate_kv_cache_size(**params))


def run_lora_params_estimate(inputs: dict[str, Any]) -> dict[str, Any]:
    """估算 LoRA 参数量。"""
    from finetune_lab.finetune_utils import calculate_lora_params, estimate_lora_memory

    params = calculate_lora_params(
        hidden_size=int(inputs["hidden_size"]),
        num_layers=int(inputs["num_layers"]),
        num_heads=int(inputs["num_heads"]),
        num_kv_heads=int(inputs.get("num_kv_heads", inputs["num_heads"])),
        intermediate_size=int(inputs["intermediate_size"]),
        rank=int(inputs["rank"]),
        target_modules=list(inputs["target_modules"]),
    )
    base_params = int(inputs.get("base_params", 0))
    if base_params > 0:
        params["memory"] = estimate_lora_memory(
            params["total_params"],
            base_params,
            bool(inputs.get("use_quantization", False)),
            int(inputs.get("quantization_bits", 4)),
        )
    return make_json_safe(params)


def run_training_cost_estimate(inputs: dict[str, Any]) -> dict[str, Any]:
    """估算训练 FLOPs、时间和成本。"""
    from finetune_lab.finetune_utils import (
        estimate_training_cost,
        estimate_training_time,
        calculate_training_flops,
    )

    total_flops = calculate_training_flops(
        int(inputs["model_params"]),
        int(inputs["tokens"]),
        bool(inputs.get("is_full_finetune", False)),
    )
    training_hours = estimate_training_time(
        total_flops,
        float(inputs["gpu_tflops"]),
        int(inputs.get("num_gpus", 1)),
        float(inputs.get("mfu", 0.5)),
    )
    total_cost = estimate_training_cost(
        training_hours,
        float(inputs["cost_per_hour"]),
        int(inputs.get("num_gpus", 1)),
    )
    return make_json_safe(
        {
            "total_flops": total_flops,
            "training_hours": training_hours,
            "total_cost_usd": total_cost,
        }
    )


def run_rag_chunk(inputs: dict[str, Any]) -> dict[str, Any]:
    """执行 RAG 文本分块分析。"""
    from rag_lab.rag_utils import (
        chunk_by_sentence,
        chunk_fixed_size,
        chunk_recursive,
        compute_chunk_stats,
    )

    text = str(inputs.get("text", ""))
    method = str(inputs.get("method", "recursive")).lower()
    chunk_size = int(inputs.get("chunk_size", 500))
    overlap = int(inputs.get("overlap", 50))
    if method == "fixed":
        chunks = chunk_fixed_size(text, chunk_size, overlap)
    elif method == "sentence":
        chunks = chunk_by_sentence(text, chunk_size, overlap)
    elif method == "recursive":
        chunks = chunk_recursive(text, chunk_size, overlap)
    else:
        raise ValueError("`method` must be one of: fixed, sentence, recursive")
    return make_json_safe({"chunks": chunks, "stats": compute_chunk_stats(chunks)})


def run_rag_lexical_retrieval(inputs: dict[str, Any]) -> dict[str, Any]:
    """执行无需 embedding 模型的词法检索诊断。"""
    query = str(inputs["query"])
    documents = _string_list(inputs, "documents")
    top_k = int(inputs.get("top_k", min(5, len(documents))))
    query_terms = Counter(_lexical_terms(query))
    if not query_terms:
        raise ValueError("`query` must contain at least one lexical term")

    results = []
    for index, document in enumerate(documents):
        document_terms = Counter(_lexical_terms(document))
        overlap = sorted(set(query_terms) & set(document_terms))
        overlap_weight = sum(min(query_terms[term], document_terms[term]) for term in overlap)
        coverage = overlap_weight / max(sum(query_terms.values()), 1)
        density = overlap_weight / max(sum(document_terms.values()), 1)
        score = coverage + density
        results.append(
            {
                "document_index": index,
                "score": score,
                "coverage": coverage,
                "density": density,
                "overlap_terms": overlap,
                "document": document,
            }
        )

    results.sort(key=lambda item: (item["score"], item["coverage"]), reverse=True)
    return make_json_safe(
        {
            "query": query,
            "query_terms": dict(query_terms),
            "results": results[: max(top_k, 0)],
        }
    )


def run_trace_analyze(inputs: dict[str, Any]) -> dict[str, Any]:
    """分析 Agent/Tool trace，但不承担 Agent runtime 职责。"""
    from agent_trace_lab.trace_utils import (
        calculate_trace_stats,
        detect_bottlenecks,
        find_critical_path,
        parse_trace_from_json,
    )

    trace_json = inputs.get("trace_json")
    if not isinstance(trace_json, str):
        raise ValueError("`trace_json` must be a JSON string")
    events = parse_trace_from_json(trace_json)
    critical_path = [
        {
            "event_type": event.event_type,
            "agent_name": event.agent_name,
            "action": event.action,
            "duration_ms": duration,
        }
        for event, duration in find_critical_path(events)
    ]
    bottlenecks = [
        {
            "event_type": item["event"].event_type,
            "agent_name": item["event"].agent_name,
            "action": item["event"].action,
            "duration_ms": item["duration_ms"],
            "slowdown_factor": item["slowdown_factor"],
            "suggestion": item["suggestion"],
        }
        for item in detect_bottlenecks(events)
    ]
    return make_json_safe(
        {
            "stats": calculate_trace_stats(events),
            "critical_path": critical_path,
            "bottlenecks": bottlenecks,
        }
    )


BUILTIN_TOOLS: list[tuple[ToolSpec, ToolHandler]] = [
    (
        ToolSpec(
            id="eval_metrics",
            label="Evaluation Metrics",
            description="Compute common text-generation evaluation metrics.",
            lab="Eval Lab",
            page_id="eval_pipeline",
            concepts=["evaluation", "metrics", "benchmarking"],
            requires_model_download=False,
            input_schema={
                "type": "object",
                "required": ["predictions", "references"],
                "properties": {
                    "predictions": {"type": "array", "items": {"type": "string"}},
                    "references": {"type": "array", "items": {"type": "string"}},
                },
            },
            output_schema={"type": "object"},
        ),
        run_eval_metrics,
    ),
    (
        ToolSpec(
            id="tokenizer_encode",
            label="Tokenizer Encode",
            description="Encode text with a HuggingFace tokenizer and report token-level diagnostics.",
            lab="TokenLab",
            page_id="token_playground",
            concepts=["tokenization", "unicode", "byte-fallback"],
            requires_model_download=True,
            dependencies=["transformers"],
            input_schema={
                "type": "object",
                "required": ["model_name", "text"],
                "properties": {
                    "model_name": {"type": "string"},
                    "text": {"type": "string"},
                },
            },
            output_schema={"type": "object"},
        ),
        run_tokenizer_encode,
    ),
    (
        ToolSpec(
            id="unicode_analyze",
            label="Unicode Analysis",
            description="Inspect Unicode characters, UTF-8 bytes, and normalization differences.",
            lab="TokenLab",
            page_id="token_playground",
            concepts=["tokenization", "unicode", "normalization"],
            requires_model_download=False,
            input_schema={
                "type": "object",
                "required": ["text"],
                "properties": {
                    "text": {"type": "string"},
                },
            },
            output_schema={"type": "object"},
        ),
        run_unicode_analyze,
    ),
    (
        ToolSpec(
            id="sampling_distribution",
            label="Sampling Distribution",
            description="Apply temperature, top-k, and top-p sampling to a logits vector.",
            lab="GenerationLab",
            page_id="generation_logits",
            concepts=["generation", "sampling", "logits"],
            requires_model_download=False,
            dependencies=["torch"],
            input_schema={
                "type": "object",
                "required": ["logits"],
                "properties": {
                    "logits": {"type": "array", "items": {"type": "number"}},
                    "tokens": {"type": "array", "items": {"type": "string"}},
                    "temperature": {"type": "number"},
                    "top_k": {"type": "integer"},
                    "top_p": {"type": "number"},
                },
            },
            output_schema={"type": "object"},
        ),
        run_sampling_distribution,
    ),
    (
        ToolSpec(
            id="kv_cache_growth",
            label="KV Cache Growth",
            description="Simulate prefill and decode KV Cache growth for a transformer workload.",
            lab="GenerationLab",
            page_id="generation_kv_cache",
            concepts=["generation", "kv-cache", "inference-memory"],
            requires_model_download=False,
            input_schema={
                "type": "object",
                "required": ["prompt_length", "generation_length"],
                "properties": {
                    "prompt_length": {"type": "integer"},
                    "generation_length": {"type": "integer"},
                    "num_layers": {"type": "integer"},
                    "hidden_size": {"type": "integer"},
                    "num_heads": {"type": "integer"},
                    "batch_size": {"type": "integer"},
                    "dtype_bytes": {"type": "integer"},
                },
            },
            output_schema={"type": "object"},
        ),
        run_kv_cache_growth,
    ),
    (
        ToolSpec(
            id="rope_frequencies",
            label="RoPE Frequencies",
            description="Compute RoPE frequency matrices and relative-distance decay samples.",
            lab="InterpretabilityLab",
            page_id="interpretability_rope",
            concepts=["interpretability", "rope", "position-encoding"],
            requires_model_download=False,
            dependencies=["numpy"],
            input_schema={
                "type": "object",
                "properties": {
                    "dim": {"type": "integer"},
                    "max_position": {"type": "integer"},
                    "max_distance": {"type": "integer"},
                    "base": {"type": "number"},
                },
            },
            output_schema={"type": "object"},
        ),
        run_rope_frequencies,
    ),
    (
        ToolSpec(
            id="ffn_activation_compare",
            label="FFN Activation Compare",
            description="Compare GELU, ReLU, SiLU, and simplified SwiGLU activation curves.",
            lab="InterpretabilityLab",
            page_id="interpretability_ffn",
            concepts=["interpretability", "ffn", "activation"],
            requires_model_download=False,
            dependencies=["numpy"],
            input_schema={
                "type": "object",
                "properties": {
                    "x_values": {"type": "array", "items": {"type": "number"}},
                },
            },
            output_schema={"type": "object"},
        ),
        run_ffn_activation_compare,
    ),
    (
        ToolSpec(
            id="data_clean",
            label="Data Cleaning",
            description="Apply deterministic text-cleaning rules for dataset preparation.",
            lab="DataLab",
            page_id="data_cleaner",
            concepts=["data", "cleaning", "sft"],
            requires_model_download=False,
            input_schema={
                "type": "object",
                "required": ["text"],
                "properties": {
                    "text": {"type": "string"},
                    "rules": {"type": "array", "items": {"type": "string"}},
                    "unicode_form": {"type": "string"},
                },
            },
            output_schema={"type": "object"},
        ),
        run_data_clean,
    ),
    (
        ToolSpec(
            id="dataset_quality_check",
            label="Dataset Quality Check",
            description="Check samples for duplicates, empty text fields, and length distribution.",
            lab="DataLab",
            page_id="data_dataset_viewer",
            concepts=["data", "quality", "diagnostics"],
            requires_model_download=False,
            input_schema={
                "type": "object",
                "required": ["samples"],
                "properties": {
                    "samples": {"type": "array", "items": {"type": "object"}},
                    "text_fields": {"type": "array", "items": {"type": "string"}},
                },
            },
            output_schema={"type": "object"},
        ),
        run_dataset_quality_check,
    ),
    (
        ToolSpec(
            id="instruct_format",
            label="Instruction Format",
            description="Convert one instruction sample to Alpaca, ShareGPT, ChatML, or Llama-2 format.",
            lab="DataLab",
            page_id="data_formatter",
            concepts=["data", "sft", "format-conversion"],
            requires_model_download=False,
            input_schema={
                "type": "object",
                "required": ["data", "target_format"],
                "properties": {
                    "data": {"type": "object"},
                    "target_format": {
                        "type": "string",
                        "enum": ["alpaca", "sharegpt", "chatml", "llama2"],
                    },
                    "system_prompt": {"type": "string"},
                },
            },
            output_schema={"type": "object"},
        ),
        run_instruct_format,
    ),
    (
        ToolSpec(
            id="kv_cache_estimate",
            label="KV Cache Estimate",
            description="Estimate KV Cache memory for a transformer decode workload.",
            lab="GenerationLab",
            page_id="generation_kv_cache",
            concepts=["generation", "kv-cache", "inference-memory"],
            requires_model_download=False,
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        ),
        run_kv_cache_estimate,
    ),
    (
        ToolSpec(
            id="lora_params_estimate",
            label="LoRA Parameter Estimate",
            description="Estimate trainable LoRA parameters and optional memory overhead.",
            lab="ModelLab",
            page_id="model_peft",
            concepts=["lora", "peft", "fine-tuning"],
            requires_model_download=False,
            input_schema={
                "type": "object",
                "required": [
                    "hidden_size",
                    "num_layers",
                    "num_heads",
                    "intermediate_size",
                    "rank",
                    "target_modules",
                ],
                "properties": {
                    "hidden_size": {"type": "integer"},
                    "num_layers": {"type": "integer"},
                    "num_heads": {"type": "integer"},
                    "num_kv_heads": {"type": "integer"},
                    "intermediate_size": {"type": "integer"},
                    "rank": {"type": "integer"},
                    "target_modules": {"type": "array", "items": {"type": "string"}},
                    "base_params": {"type": "integer"},
                    "use_quantization": {"type": "boolean"},
                    "quantization_bits": {"type": "integer"},
                },
            },
            output_schema={"type": "object"},
        ),
        run_lora_params_estimate,
    ),
    (
        ToolSpec(
            id="training_cost_estimate",
            label="Training Cost Estimate",
            description="Estimate fine-tuning FLOPs, wall-clock hours, and GPU cost.",
            lab="FineTuneLab",
            page_id="finetune_training_cost",
            concepts=["fine-tuning", "cost", "flops"],
            requires_model_download=False,
            input_schema={
                "type": "object",
                "required": ["model_params", "tokens", "gpu_tflops", "cost_per_hour"],
                "properties": {
                    "model_params": {"type": "integer"},
                    "tokens": {"type": "integer"},
                    "gpu_tflops": {"type": "number"},
                    "cost_per_hour": {"type": "number"},
                    "num_gpus": {"type": "integer"},
                    "mfu": {"type": "number"},
                    "is_full_finetune": {"type": "boolean"},
                },
            },
            output_schema={"type": "object"},
        ),
        run_training_cost_estimate,
    ),
    (
        ToolSpec(
            id="rag_chunk",
            label="RAG Chunking",
            description="Split text into chunks and report chunk statistics.",
            lab="RAGLab",
            page_id="rag_chunking",
            concepts=["rag", "chunking", "retrieval"],
            requires_model_download=False,
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        ),
        run_rag_chunk,
    ),
    (
        ToolSpec(
            id="rag_lexical_retrieval",
            label="RAG Lexical Retrieval",
            description="Rank documents with a transparent lexical overlap diagnostic.",
            lab="RAGLab",
            page_id="rag_retrieval",
            concepts=["rag", "retrieval", "diagnostics"],
            requires_model_download=False,
            input_schema={
                "type": "object",
                "required": ["query", "documents"],
                "properties": {
                    "query": {"type": "string"},
                    "documents": {"type": "array", "items": {"type": "string"}},
                    "top_k": {"type": "integer"},
                },
            },
            output_schema={"type": "object"},
        ),
        run_rag_lexical_retrieval,
    ),
    (
        ToolSpec(
            id="trace_analyze",
            label="Trace Analysis",
            description="Analyze trace timing, critical path, and bottlenecks.",
            lab="Agent Trace Lab",
            page_id="agent_trace_analyzer",
            concepts=["trace", "tool-calling", "latency"],
            requires_model_download=False,
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        ),
        run_trace_analyze,
    ),
]
