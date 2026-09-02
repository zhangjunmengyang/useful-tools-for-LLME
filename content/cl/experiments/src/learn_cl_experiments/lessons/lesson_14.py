from __future__ import annotations

import math
import random
from typing import Any

from ..core import LessonExperiment


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _norm(vector: list[float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _sub(left: list[float], right: list[float]) -> list[float]:
    return [a - b for a, b in zip(left, right)]


def _scale(vector: list[float], scalar: float) -> list[float]:
    return [scalar * value for value in vector]


def _cosine(left: list[float], right: list[float]) -> float:
    denom = _norm(left) * _norm(right)
    if denom == 0.0:
        return 0.0
    return _dot(left, right) / denom


def _matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [_dot(row, vector) for row in matrix]


def _outer_add(
    matrix: list[list[float]],
    left: list[float],
    right: list[float],
) -> list[list[float]]:
    return [
        [matrix[row][col] + left[row] * right[col] for col in range(len(right))]
        for row in range(len(left))
    ]


def _rank1_edit(
    weights: list[list[float]],
    key: list[float],
    new_value: list[float],
) -> list[list[float]]:
    residual = _sub(new_value, _matvec(weights, key))
    key_norm_sq = _dot(key, key)
    return _outer_add(weights, residual, _scale(key, 1.0 / key_norm_sq))


def _mean_change(
    before: list[list[float]],
    after: list[list[float]],
    keys: list[list[float]],
) -> float:
    deltas = [
        _norm(_sub(_matvec(after, key), _matvec(before, key))) for key in keys
    ]
    return sum(deltas) / len(deltas)


def run() -> dict[str, Any]:
    rng = random.Random(0)
    n_facts = 6
    dim = 20
    keys = [[rng.gauss(0.0, 1.0) for _ in range(dim)] for _ in range(n_facts)]
    values = [[rng.gauss(0.0, 1.0) for _ in range(dim)] for _ in range(n_facts)]
    weights = [[0.0] * dim for _ in range(dim)]
    for key, value in zip(keys, values):
        weights = _rank1_edit(weights, key, value)

    target_new = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    edited = _rank1_edit(weights, keys[0], target_new)
    unlearned = _rank1_edit(weights, keys[0], [0.0] * dim)
    naive = [[0.0] * dim for _ in range(dim)]
    naive = _rank1_edit(naive, keys[0], target_new)

    target_before = _matvec(weights, keys[0])
    target_after = _matvec(edited, keys[0])
    target_delta = _norm(_sub(target_after, target_before))
    neighbor_keys = keys[1:]
    neighbor_delta = _mean_change(weights, edited, neighbor_keys)
    naive_neighbor_delta = _mean_change(weights, naive, neighbor_keys)
    unlearn_target = _norm(_matvec(unlearned, keys[0]))
    original_target = _norm(target_before)
    neighbor_cosine = sum(
        _cosine(_matvec(weights, key), _matvec(unlearned, key))
        for key in neighbor_keys
    ) / len(neighbor_keys)

    checks = {
        "target_moves_to_new_value": _cosine(target_after, target_new) > 0.95,
        "target_change_exceeds_neighbors": target_delta > 4.0 * neighbor_delta,
        "unlearn_shrinks_target": unlearn_target < 0.15 * original_target,
        "unlearn_keeps_neighbors": neighbor_cosine > 0.9,
        "naive_rewrite_hurts_neighbors": naive_neighbor_delta > 3.0 * neighbor_delta,
    }
    return {
        "summary": (
            "对线性联想记忆做一次闭式 rank-1 编辑：目标键余弦 > 0.95，"
            "邻键位移小于目标的 1/4。把目标改成零向量后邻键余弦仍 > 0.9；"
            "丢掉旧记忆、只按新事实重写整张表，邻键位移会大得多。"
        ),
        "metrics": {
            "target_cosine_after_edit": _cosine(target_after, target_new),
            "target_delta": target_delta,
            "neighbor_delta": neighbor_delta,
            "locality_ratio": target_delta / max(neighbor_delta, 1e-12),
            "unlearn_target_norm": unlearn_target,
            "original_target_norm": original_target,
            "unlearn_neighbor_cosine": neighbor_cosine,
            "naive_neighbor_delta": naive_neighbor_delta,
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="14",
    title="改一条事实，别把邻居改坏",
    question="改一条事实之后，无关事实的位移是不是明显更小？",
    run=run,
)
