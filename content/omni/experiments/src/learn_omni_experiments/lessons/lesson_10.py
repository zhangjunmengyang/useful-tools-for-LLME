from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from ..core import LessonExperiment


Token = dict[str, Any]
Window = tuple[int, int, int]


def _make_tokens() -> list[Token]:
    tokens: list[Token] = []
    for time in range(4):
        for row in range(2):
            for column in range(2):
                position = time * 4 + row * 2 + column
                tokens.append(
                    {
                        "position": position,
                        "coordinate": (time, row, column),
                        "feature": [
                            float(time),
                            float(row),
                            float(column),
                            float((time % 2) * (row + column + 1)),
                        ],
                        "mass": 1.0,
                        "members": [position],
                    },
                )
    return tokens


def _window(token: Token) -> Window:
    time, row, column = token["coordinate"]
    return time // 2, row // 2, column // 2


def _allocate_budget(tokens: list[Token], budget: int) -> dict[Window, int]:
    groups: dict[Window, list[Token]] = defaultdict(list)
    for token in tokens:
        groups[_window(token)].append(token)
    if budget < len(groups) or budget > len(tokens):
        raise ValueError("budget must reserve one token per non-empty window")

    allocation = {window: 1 for window in groups}
    remaining = budget - len(groups)
    capacities = {
        window: len(group) - 1 for window, group in groups.items()
    }
    total_capacity = sum(capacities.values())
    if remaining:
        shares = {
            window: remaining * capacity / total_capacity
            for window, capacity in capacities.items()
        }
        for window, share in shares.items():
            allocation[window] += math.floor(share)
        leftovers = budget - sum(allocation.values())
        order = sorted(
            groups,
            key=lambda window: (-(shares[window] % 1), window),
        )
        for window in order[:leftovers]:
            allocation[window] += 1
    return allocation


def _uniform(tokens: list[Token], budget: int) -> list[Token]:
    allocation = _allocate_budget(tokens, budget)
    groups: dict[Window, list[Token]] = defaultdict(list)
    for token in tokens:
        groups[_window(token)].append(token)

    selected: list[Token] = []
    for window in sorted(groups):
        group = sorted(groups[window], key=lambda token: token["position"])
        target = allocation[window]
        if target == 1:
            indices = [0]
        else:
            indices = [
                round(index * (len(group) - 1) / (target - 1))
                for index in range(target)
            ]
        selected.extend(group[index].copy() for index in indices)
    return sorted(selected, key=lambda token: token["position"])


def _cosine(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return -1.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _merge_pair(left: Token, right: Token) -> Token:
    mass = left["mass"] + right["mass"]
    feature = [
        (left["mass"] * a + right["mass"] * b) / mass
        for a, b in zip(left["feature"], right["feature"])
    ]
    if left["mass"] > right["mass"]:
        position = left["position"]
    elif right["mass"] > left["mass"]:
        position = right["position"]
    else:
        position = min(left["position"], right["position"])
    return {
        "position": position,
        "coordinate": (
            left["coordinate"]
            if position == left["position"]
            else right["coordinate"]
        ),
        "feature": feature,
        "mass": mass,
        "members": sorted(left["members"] + right["members"]),
    }


def _similarity_merge(
    tokens: list[Token],
    budget: int,
    stats: dict[str, int] | None = None,
) -> list[Token]:
    allocation = _allocate_budget(tokens, budget)
    groups: dict[Window, list[Token]] = defaultdict(list)
    for token in tokens:
        groups[_window(token)].append(
            {
                **token,
                "feature": list(token["feature"]),
                "members": list(token["members"]),
            },
        )

    merged_output: list[Token] = []
    for window in sorted(groups):
        current = groups[window]
        target = allocation[window]
        while len(current) > target:
            candidates: list[tuple[int, int, int, int, int]] = []
            for left_index in range(len(current)):
                for right_index in range(left_index + 1, len(current)):
                    if stats is not None:
                        stats["pair_comparisons"] += 1
                    left = current[left_index]
                    right = current[right_index]
                    quantized = round(
                        _cosine(left["feature"], right["feature"]) * 1_000_000,
                    )
                    candidates.append(
                        (
                            -quantized,
                            min(left["position"], right["position"]),
                            max(left["position"], right["position"]),
                            left_index,
                            right_index,
                        ),
                    )
            candidates.sort()
            used: set[int] = set()
            chosen: list[tuple[int, int]] = []
            pair_limit = min(len(current) - target, len(current) // 2)
            for _, _, _, left_index, right_index in candidates:
                if left_index in used or right_index in used:
                    continue
                chosen.append((left_index, right_index))
                used.update((left_index, right_index))
                if len(chosen) == pair_limit:
                    break
            next_round = [
                token for index, token in enumerate(current) if index not in used
            ]
            next_round.extend(
                _merge_pair(current[left], current[right])
                for left, right in chosen
            )
            if stats is not None:
                stats["merges"] += len(chosen)
            current = sorted(
                next_round,
                key=lambda token: token["position"],
            )
        merged_output.extend(current)
    return sorted(merged_output, key=lambda token: token["position"])


def _evs(tokens: list[Token], budget: int) -> list[Token]:
    allocation = _allocate_budget(tokens, budget)
    groups: dict[Window, list[Token]] = defaultdict(list)
    by_coordinate = {
        token["coordinate"]: token for token in tokens
    }
    for token in tokens:
        groups[_window(token)].append(token)

    selected: list[Token] = []
    for window in sorted(groups):
        group = groups[window]
        target = allocation[window]
        anchor = min(group, key=lambda token: token["position"])
        choices = [anchor]
        scored: list[tuple[float, int, Token]] = []
        for token in group:
            if token is anchor:
                continue
            time, row, column = token["coordinate"]
            previous = by_coordinate.get((time - 1, row, column))
            change = 0.0 if previous is None else (
                1.0 - _cosine(token["feature"], previous["feature"])
            )
            scored.append((-change, token["position"], token))
        scored.sort()
        choices.extend(item[2] for item in scored[: target - 1])
        selected.extend(token.copy() for token in choices)
    return sorted(selected, key=lambda token: token["position"])


def _reconstruction_mse(original: list[Token], reduced: list[Token]) -> float:
    member_to_feature: dict[int, list[float]] = {}
    for token in reduced:
        for member in token["members"]:
            member_to_feature[member] = token["feature"]

    squared_error = 0.0
    value_count = 0
    for token in original:
        reconstructed = member_to_feature.get(token["position"])
        if reconstructed is None:
            reconstructed = min(
                reduced,
                key=lambda candidate: sum(
                    (a - b) ** 2
                    for a, b in zip(token["feature"], candidate["feature"])
                ),
            )["feature"]
        squared_error += sum(
            (a - b) ** 2
            for a, b in zip(token["feature"], reconstructed)
        )
        value_count += len(token["feature"])
    return squared_error / value_count


def _reduce(
    method: str,
    tokens: list[Token],
    budget: int,
) -> dict[str, Any]:
    feature_size = len(tokens[0]["feature"])
    if method == "uniform":
        reduced = _uniform(tokens, budget)
        reducer_work_units = len(tokens) + budget
        method_trace: dict[str, int] = {"scanned_tokens": len(tokens)}
    elif method == "similarity_merge":
        method_trace = {"pair_comparisons": 0, "merges": 0}
        reduced = _similarity_merge(tokens, budget, method_trace)
        reducer_work_units = (
            len(tokens)
            + method_trace["pair_comparisons"]
            + method_trace["merges"] * feature_size
            + budget
        )
    elif method == "evs":
        by_coordinate = {token["coordinate"] for token in tokens}
        temporal_comparisons = sum(
            (time - 1, row, column) in by_coordinate
            for time, row, column in (
                token["coordinate"] for token in tokens
            )
        )
        reduced = _evs(tokens, budget)
        reducer_work_units = (
            len(tokens) + temporal_comparisons * feature_size + budget
        )
        method_trace = {"temporal_comparisons": temporal_comparisons}
    else:
        raise ValueError(f"unknown reducer: {method}")

    positions = [int(token["position"]) for token in reduced]
    kept_positions = set(positions)
    return {
        "tokens": reduced,
        "reduced_z": [list(token["feature"]) for token in reduced],
        "original_position_ids": positions,
        "token_mass": [float(token["mass"]) for token in reduced],
        "keep_mask": [
            int(token["position"]) in kept_positions for token in tokens
        ],
        "trace": {
            "method": method,
            "input_tokens": len(tokens),
            "budget": budget,
            "output_tokens": len(reduced),
            "reducer_work_units": reducer_work_units,
            **method_trace,
        },
    }


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    dimensions = (
        "feature_reconstruction_mse",
        "output_tokens",
        "compute_cost_units",
    )
    no_worse = all(left[name] <= right[name] for name in dimensions)
    strictly_better = any(left[name] < right[name] for name in dimensions)
    return no_worse and strictly_better


def _run() -> dict[str, Any]:
    tokens = _make_tokens()
    methods = ("uniform", "similarity_merge", "evs")
    budgets = (len(tokens), 12, 8, 4)
    results = {
        (method, budget): _reduce(method, tokens, budget)
        for method in methods
        for budget in budgets
    }

    points: list[dict[str, Any]] = []
    for method in methods:
        for budget in budgets:
            result = results[(method, budget)]
            reducer_work = int(result["trace"]["reducer_work_units"])
            prefill_attention_work = budget * budget
            points.append(
                {
                    "id": f"{method}@{budget}",
                    "method": method,
                    "budget": budget,
                    "keep_ratio": budget / len(tokens),
                    "output_tokens": len(result["tokens"]),
                    "feature_reconstruction_mse": _reconstruction_mse(
                        tokens,
                        result["tokens"],
                    ),
                    "reducer_work_units": reducer_work,
                    "prefill_attention_work_units": prefill_attention_work,
                    "compute_cost_units": reducer_work + prefill_attention_work,
                },
            )

    dominated_by = {
        point["id"]: sorted(
            candidate["id"]
            for candidate in points
            if candidate is not point and _dominates(candidate, point)
        )
        for point in points
    }
    frontier = sorted(
        point["id"] for point in points if not dominated_by[point["id"]]
    )
    full_budget_results = {
        method: results[(method, len(tokens))] for method in methods
    }
    merged_at_half = results[("similarity_merge", 8)]

    invalid_budget_rejected = False
    try:
        _similarity_merge(tokens, budget=1)
    except ValueError:
        invalid_budget_rejected = True

    checks = {
        "all_three_full_budget_reducers_satisfy_the_complete_contract": all(
            set(result)
            == {
                "tokens",
                "reduced_z",
                "original_position_ids",
                "token_mass",
                "keep_mask",
                "trace",
            }
            and result["tokens"] == tokens
            and result["reduced_z"]
            == [token["feature"] for token in tokens]
            and result["original_position_ids"] == list(range(len(tokens)))
            and result["token_mass"] == [1.0] * len(tokens)
            and result["keep_mask"] == [True] * len(tokens)
            and result["trace"]["output_tokens"] == len(tokens)
            for result in full_budget_results.values()
        ),
        "every_reducer_meets_every_exact_integer_budget": all(
            len(results[(method, budget)]["tokens"]) == budget
            and sum(results[(method, budget)]["keep_mask"]) == budget
            for method in methods
            for budget in budgets
        ),
        "similarity_merging_conserves_mass_at_every_budget": all(
            sum(results[("similarity_merge", budget)]["token_mass"])
            == len(tokens)
            for budget in budgets
        ),
        "all outputs preserve monotonically ordered original positions": all(
            result["original_position_ids"]
            == sorted(result["original_position_ids"])
            for result in results.values()
        ),
        "similarity merge is deterministic under a fixed tie break": (
            merged_at_half["tokens"] == _similarity_merge(tokens, 8)
        ),
        "an impossible per-window budget fails closed": invalid_budget_rejected,
        "pareto_scans_three_methods_at_four_budgets": (
            len(points) == 12
            and {point["keep_ratio"] for point in points}
            == {1.0, 0.75, 0.5, 0.25}
        ),
        "pareto_frontier_is_computed_by_explicit_dominance": (
            0 < len(frontier) < len(points)
            and all(
                not any(
                    _dominates(candidate, point)
                    for candidate in points
                    if candidate is not point
                )
                for point in points
                if point["id"] in frontier
            )
            and all(dominated_by[point["id"]] for point in points
                    if point["id"] not in frontier)
        ),
    }

    return {
        "summary": (
            "对三种 reducer 实际扫描 100/75/50/25% 四档预算，并按特征重建"
            "误差、输出 token 和确定性计算成本求非支配前沿。计算成本是教学用"
            "工作量代理，不是实测毫秒，也不是模型任务分数。"
        ),
        "metrics": {
            "input_tokens": len(tokens),
            "budgets": list(budgets),
            "pareto_points": points,
            "pareto_frontier": frontier,
            "dominated_by": dominated_by,
            "half_budget_merged_clusters": [
                token["members"] for token in merged_at_half["tokens"]
            ],
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="10",
    title="视觉/视频 Token Reduction 与质量—Token—延迟 Pareto",
    question="压缩视觉 token 时，怎样同时守住整数预算、原始位置和合并质量？",
    run=_run,
)
