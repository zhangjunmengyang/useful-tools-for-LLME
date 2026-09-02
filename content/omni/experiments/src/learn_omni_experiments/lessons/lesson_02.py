from __future__ import annotations

import math
from typing import Any

from ..core import LessonExperiment


Vector = list[float]
Matrix = list[Vector]


def _softmax(values: Vector) -> Vector:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def _project(vector: Vector) -> Vector:
    weights = (
        (0.5, -0.25, 0.75),
        (-0.4, 0.8, 0.2),
    )
    return [sum(weight * value for weight, value in zip(row, vector)) for row in weights]


def _attend(query: Vector, features: Matrix, valid: list[bool]) -> tuple[Vector, Vector]:
    valid_features = [feature for feature, keep in zip(features, valid) if keep]
    scale = math.sqrt(len(query))
    weights = _softmax(
        [sum(q * value for q, value in zip(query, feature)) / scale for feature in valid_features],
    )
    output = [
        sum(weight * feature[dimension] for weight, feature in zip(weights, valid_features))
        for dimension in range(len(query))
    ]
    return output, weights


def _resample(
    features: Matrix,
    valid: list[bool],
    queries: Matrix,
) -> tuple[Matrix, Matrix]:
    outputs: Matrix = []
    attention: Matrix = []
    for query in queries:
        output, weights = _attend(query, features, valid)
        outputs.append(output)
        attention.append(weights)
    return outputs, attention


def _two_stage_queries(features: Matrix, valid: list[bool], queries: Matrix) -> Matrix:
    first_stage, _ = _resample(features, valid, queries)
    shared_context = [
        sum(vector[dimension] for vector in first_stage) / len(first_stage)
        for dimension in range(len(first_stage[0]))
    ]
    refined_queries = [
        [query_value + 0.25 * context for query_value, context in zip(query, shared_context)]
        for query in queries
    ]
    second_stage, _ = _resample(features, valid, refined_queries)
    return second_stage


def _distance(left: Matrix, right: Matrix) -> float:
    return math.sqrt(
        sum(
            (left_value - right_value) ** 2
            for left_row, right_row in zip(left, right)
            for left_value, right_value in zip(left_row, right_row)
        ),
    )


def _close(left: Matrix, right: Matrix, tolerance: float = 1e-12) -> bool:
    return _distance(left, right) <= tolerance


def run() -> dict[str, Any]:
    features = [
        [1.0, 0.0, 0.5],
        [0.5, 1.0, -0.5],
        [-0.25, 0.75, 1.0],
        [0.8, -0.4, 0.2],
        [999.0, -999.0, 500.0],
    ]
    changed_padding = features[:-1] + [[-500.0, 700.0, -900.0]]
    valid = [True, True, True, True, False]
    queries = [[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]]

    mlp_output = [_project(feature) for feature, keep in zip(features, valid) if keep]
    resampled, attention = _resample(features, valid, queries)
    resampled_with_changed_padding, _ = _resample(changed_padding, valid, queries)
    refined = _two_stage_queries(features, valid, queries)

    wrong_modality = [
        [-1.0, 0.0, -0.5],
        [-0.5, -1.0, 0.5],
        [0.25, -0.75, -1.0],
        [-0.8, 0.4, -0.2],
        [0.0, 0.0, 0.0],
    ]
    wrong_resampled, _ = _resample(wrong_modality, valid, queries)
    modality_distance = _distance(resampled, wrong_resampled)

    checks = {
        "mlp_preserves_valid_token_count": len(mlp_output) == sum(valid),
        "learned_queries_set_output_length": len(resampled) == len(queries),
        "padding_values_are_masked": _close(resampled, resampled_with_changed_padding),
        "attention_is_normalized": all(
            math.isclose(sum(row), 1.0, rel_tol=0.0, abs_tol=1e-12)
            for row in attention
        ),
        "two_stage_queries_keep_contract": len(refined) == 2
        and all(len(vector) == 3 for vector in refined),
        "wrong_modality_changes_representation": modality_distance > 1.0,
    }
    return {
        "summary": (
            "在同一组 encoder features 上计算逐 token 投影、learned-query 压缩和"
            "两阶段 query 更新，并用 padding 与错模态反事实检查接口。"
        ),
        "metrics": {
            "valid_input_tokens": sum(valid),
            "mlp_output_tokens": len(mlp_output),
            "query_output_tokens": len(resampled),
            "token_reduction_ratio": sum(valid) / len(resampled),
            "wrong_modality_l2": round(modality_distance, 6),
            "attention_row_sums": [round(sum(row), 12) for row in attention],
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="02",
    title="比较多模态 Connector",
    question="固定输入后，connector 如何改变 token 数并保留有效信息？",
    run=run,
)
