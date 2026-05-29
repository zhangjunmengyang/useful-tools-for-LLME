"""研究工具的可运行默认配置。"""

from __future__ import annotations

from typing import Any


DEFAULT_CONFIGS: dict[str, dict[str, Any]] = {
    "data_clean": {
        "text": "<p>Hello world</p> https://example.com",
        "rules": ["html", "url", "whitespace"],
    },
    "dataset_quality_check": {
        "samples": [
            {"instruction": "Say hi", "output": "Hi"},
            {"instruction": "", "output": "Missing instruction"},
            {"instruction": "Say hi", "output": "Hi"},
        ],
        "text_fields": ["instruction", "output"],
    },
    "eval_metrics": {
        "predictions": ["Paris"],
        "references": ["Paris"],
    },
    "ffn_activation_compare": {
        "x_values": [-1.0, 0.0, 1.0],
    },
    "instruct_format": {
        "data": {
            "instruction": "Explain KV Cache in one sentence.",
            "input": "",
            "output": "KV Cache stores prior keys and values so decoding can reuse attention state.",
        },
        "target_format": "chatml",
    },
    "kv_cache_estimate": {
        "num_layers": 32,
        "hidden_size": 4096,
        "num_heads": 32,
        "seq_length": 2048,
        "batch_size": 1,
        "dtype_bytes": 2,
    },
    "kv_cache_growth": {
        "prompt_length": 1024,
        "generation_length": 16,
        "num_layers": 32,
        "hidden_size": 4096,
        "num_heads": 32,
        "batch_size": 1,
        "dtype_bytes": 2,
    },
    "lora_params_estimate": {
        "hidden_size": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 32,
        "intermediate_size": 11008,
        "rank": 16,
        "target_modules": ["q_proj", "v_proj"],
    },
    "rag_chunk": {
        "text": "Retrieval augmented generation connects a user query to external documents before generation.",
        "method": "recursive",
        "chunk_size": 80,
        "overlap": 10,
    },
    "rag_lexical_retrieval": {
        "query": "python language",
        "documents": [
            "Python is a programming language.",
            "Coffee is a drink.",
        ],
        "top_k": 1,
    },
    "rope_frequencies": {
        "dim": 8,
        "max_position": 4,
        "max_distance": 4,
        "base": 10000.0,
    },
    "sampling_distribution": {
        "logits": [2.0, 1.0, 0.0],
        "tokens": ["A", "B", "C"],
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
    },
    "tokenizer_encode": {
        "model_name": "openai-community/gpt2",
        "text": "LLM research toolbox",
    },
    "trace_analyze": {
        "trace_json": "[]",
    },
    "training_cost_estimate": {
        "model_params": 7000000000,
        "tokens": 100000000,
        "gpu_tflops": 312,
        "cost_per_hour": 3.0,
        "num_gpus": 1,
        "mfu": 0.5,
    },
    "unicode_analyze": {
        "text": "Ａ café",
    },
}
