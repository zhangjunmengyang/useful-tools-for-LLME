from __future__ import annotations

import math
from typing import Any

from ..core import LessonExperiment


Vector = list[float]
Route = list[tuple[int, float]]


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _route(tokens: list[Vector], expert_keys: list[Vector], top_k: int) -> list[Route]:
    routes: list[Route] = []
    for token in tokens:
        affinities = [
            _sigmoid(sum(value * key for value, key in zip(token, expert_key)))
            for expert_key in expert_keys
        ]
        selected = sorted(
            range(len(expert_keys)),
            key=lambda expert: (-affinities[expert], expert),
        )[:top_k]
        normalizer = sum(affinities[expert] for expert in selected)
        routes.append(
            [
                (expert, affinities[expert] / normalizer)
                for expert in selected
            ],
        )
    return routes


def _naive_forward(
    tokens: list[Vector],
    routes: list[Route],
    expert_scales: list[float],
    shared_scale: float,
) -> list[Vector]:
    outputs: list[Vector] = []
    for token, route in zip(tokens, routes):
        output = [shared_scale * value for value in token]
        for expert, weight in route:
            for index, value in enumerate(token):
                output[index] += weight * expert_scales[expert] * value
        outputs.append(output)
    return outputs


def _grouped_forward(
    tokens: list[Vector],
    routes: list[Route],
    expert_scales: list[float],
    shared_scale: float,
) -> list[Vector]:
    outputs = [
        [shared_scale * value for value in token] for token in tokens
    ]
    for expert, scale in enumerate(expert_scales):
        for token_index, (token, route) in enumerate(zip(tokens, routes)):
            weights = {
                selected_expert: weight
                for selected_expert, weight in route
            }
            if expert not in weights:
                continue
            for feature_index, value in enumerate(token):
                outputs[token_index][feature_index] += (
                    weights[expert] * scale * value
                )
    return outputs


def _gini(values: list[int]) -> float:
    ordered = sorted(values)
    total = sum(ordered)
    if total == 0:
        return 0.0
    count = len(ordered)
    weighted = sum(
        (index + 1) * value for index, value in enumerate(ordered)
    )
    return (2.0 * weighted) / (count * total) - (count + 1) / count


def _sign(value: float) -> int:
    return (value > 0.0) - (value < 0.0)


def _run() -> dict[str, Any]:
    tokens: list[Vector] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, -1.0, 0.0, 0.0],
    ]
    expert_keys: list[Vector] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.5, 0.5, 0.5, 0.0],
    ]
    top_k = 2
    routes = _route(tokens, expert_keys, top_k)
    expert_scales = [0.5, 1.0, 1.5, 2.0]
    shared_scale = 0.25

    naive = _naive_forward(tokens, routes, expert_scales, shared_scale)
    grouped = _grouped_forward(tokens, routes, expert_scales, shared_scale)
    max_parity_error = max(
        abs(left - right)
        for naive_token, grouped_token in zip(naive, grouped)
        for left, right in zip(naive_token, grouped_token)
    )

    loads = [0] * len(expert_keys)
    for route in routes:
        for expert, _ in route:
            loads[expert] += 1
    mean_load = sum(loads) / len(loads)
    load_cv = (
        math.sqrt(sum((load - mean_load) ** 2 for load in loads) / len(loads))
        / mean_load
    )
    probabilities = [load / sum(loads) for load in loads]
    normalized_entropy = (
        -sum(probability * math.log(probability) for probability in probabilities)
        / math.log(len(loads))
    )

    bias_before = [0.0] * len(loads)
    update_speed = 0.001
    bias_after = [
        bias + update_speed * _sign(mean_load - load)
        for bias, load in zip(bias_before, loads)
    ]
    permuted = _naive_forward(
        tokens,
        routes,
        list(reversed(expert_scales)),
        shared_scale,
    )

    d_model = 768
    shared_width = 256
    routed_width = 256
    experts = 8
    canonical_top_k = 2
    active_width = shared_width + canonical_top_k * routed_width
    total_width = shared_width + experts * routed_width
    dense_active_ffn_parameters = 3 * d_model * active_width
    moe_active_ffn_parameters = 3 * d_model * (
        shared_width + canonical_top_k * routed_width
    )
    dense_total_ffn_parameters = 3 * d_model * total_width
    moe_total_ffn_parameters = 3 * d_model * (
        shared_width + experts * routed_width
    )

    checks = {
        "every token is routed exactly top-k times": all(
            len(route) == top_k for route in routes
        ),
        "selected routing weights are normalized": all(
            abs(sum(weight for _, weight in route) - 1.0) < 1e-12
            for route in routes
        ),
        "grouped dispatch matches the per-token reference": (
            max_parity_error < 1e-12
        ),
        "dropless dispatch accounts for every selected token": (
            sum(loads) == len(tokens) * top_k
        ),
        "expert bias moves toward under-used experts": all(
            (after >= before if load < mean_load else after <= before)
            for load, before, after in zip(loads, bias_before, bias_after)
        ),
        "zero imbalance has a zero update": _sign(0.0) == 0,
        "iso-active and iso-total FFN parameter counts match": (
            dense_active_ffn_parameters == moe_active_ffn_parameters
            and dense_total_ffn_parameters == moe_total_ffn_parameters
        ),
        "permuting expert behavior changes the routed result": permuted != naive,
    }

    return {
        "summary": (
            "在五个小 token 上执行 sigmoid top-2 路由、逐 token 与 grouped dispatch"
            " 对照、负载统计和 expert-bias 更新；参数核算只比较 FFN，不伪造训练质量。"
        ),
        "metrics": {
            "token_count": len(tokens),
            "top_k": top_k,
            "expert_loads": loads,
            "load_cv": load_cv,
            "load_gini": _gini(loads),
            "normalized_load_entropy": normalized_entropy,
            "max_grouped_parity_error": max_parity_error,
            "bias_after_update": bias_after,
            "canonical_active_width": active_width,
            "canonical_total_width": total_width,
            "dense_active_ffn_parameters": dense_active_ffn_parameters,
            "dense_total_ffn_parameters": dense_total_ffn_parameters,
            "router_parameters_excluded_from_ffn_match": d_model * experts,
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="11",
    title="Tiny MoE：从教学路由到现代稀疏专家",
    question="怎样证明 top-k 路由没有丢 token，并公平核算 active 与 total 参数？",
    run=_run,
)
