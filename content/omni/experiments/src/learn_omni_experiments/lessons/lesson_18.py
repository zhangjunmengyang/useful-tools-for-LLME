from __future__ import annotations

import copy
import json
import math
import xml.etree.ElementTree as ET

from ..core import LessonExperiment


Matrix = list[list[float]]


def _matvec(matrix: Matrix, vector: list[float]) -> list[float]:
    return [
        sum(value * vector[index] for index, value in enumerate(row))
        for row in matrix
    ]


def _matmul(left: Matrix, right: Matrix) -> Matrix:
    right_columns = list(zip(*right))
    return [
        [
            sum(a * b for a, b in zip(row, column))
            for column in right_columns
        ]
        for row in left
    ]


def _add(left: Matrix, right: Matrix, scale: float = 1.0) -> Matrix:
    return [
        [
            left[row][column] + scale * right[row][column]
            for column in range(len(left[row]))
        ]
        for row in range(len(left))
    ]


def _lora_output(
    base: Matrix,
    adapter_a: Matrix,
    adapter_b: Matrix,
    vector: list[float],
    scale: float,
) -> list[float]:
    base_output = _matvec(base, vector)
    adapter_output = _matvec(adapter_b, _matvec(adapter_a, vector))
    return [
        base_value + scale * adapter_value
        for base_value, adapter_value in zip(base_output, adapter_output)
    ]


def _mse(values: list[float], target: list[float]) -> float:
    return sum(
        (value - expected) ** 2
        for value, expected in zip(values, target)
    ) / len(values)


def _adapter_loss(
    base: Matrix,
    adapter_a: Matrix,
    adapter_b: Matrix,
    vector: list[float],
    target: list[float],
    scale: float,
) -> float:
    return _mse(
        _lora_output(base, adapter_a, adapter_b, vector, scale),
        target,
    )


def _finite_difference(
    base: Matrix,
    adapter_a: Matrix,
    adapter_b: Matrix,
    vector: list[float],
    target: list[float],
    scale: float,
) -> tuple[Matrix, Matrix]:
    epsilon = 1e-6

    def gradient_for(which: str, source: Matrix) -> Matrix:
        gradient = [[0.0 for _ in row] for row in source]
        for row in range(len(source)):
            for column in range(len(source[row])):
                plus_a = copy.deepcopy(adapter_a)
                minus_a = copy.deepcopy(adapter_a)
                plus_b = copy.deepcopy(adapter_b)
                minus_b = copy.deepcopy(adapter_b)
                plus = plus_a if which == "a" else plus_b
                minus = minus_a if which == "a" else minus_b
                plus[row][column] += epsilon
                minus[row][column] -= epsilon
                plus_loss = _adapter_loss(
                    base,
                    plus_a,
                    plus_b,
                    vector,
                    target,
                    scale,
                )
                minus_loss = _adapter_loss(
                    base,
                    minus_a,
                    minus_b,
                    vector,
                    target,
                    scale,
                )
                gradient[row][column] = (
                    plus_loss - minus_loss
                ) / (2.0 * epsilon)
        return gradient

    return gradient_for("a", adapter_a), gradient_for("b", adapter_b)


def _parse_total(document: str) -> float | None:
    try:
        root = ET.fromstring(document)
        value = root.findtext("total")
        return None if value is None else float(value)
    except (ET.ParseError, ValueError):
        return None


def run() -> dict[str, object]:
    base = [[0.2, -0.1, 0.3], [-0.4, 0.5, 0.1]]
    adapter_a = [[0.1, 0.0, -0.2], [-0.1, 0.2, 0.1]]
    adapter_b = [[0.3, -0.2], [0.1, 0.4]]
    vector = [1.0, -2.0, 0.5]
    target = [1.0, -1.0]
    rank = 2
    alpha = 4.0
    scale = alpha / rank

    unmerged = _lora_output(
        base,
        adapter_a,
        adapter_b,
        vector,
        scale,
    )
    merged_weight = _add(base, _matmul(adapter_b, adapter_a), scale)
    merged = _matvec(merged_weight, vector)
    merge_max_error = max(
        abs(left - right) for left, right in zip(unmerged, merged)
    )

    initial_loss = _adapter_loss(
        base,
        adapter_a,
        adapter_b,
        vector,
        target,
        scale,
    )
    gradient_a, gradient_b = _finite_difference(
        base,
        adapter_a,
        adapter_b,
        vector,
        target,
        scale,
    )
    learning_rate = 0.05
    updated_a = _add(adapter_a, gradient_a, -learning_rate)
    updated_b = _add(adapter_b, gradient_b, -learning_rate)
    updated_loss = _adapter_loss(
        base,
        updated_a,
        updated_b,
        vector,
        target,
        scale,
    )

    serialized_adapter = json.dumps(
        {"a": adapter_a, "b": adapter_b, "alpha": alpha, "rank": rank},
        sort_keys=True,
    )
    restored = json.loads(serialized_adapter)
    restored_output = _lora_output(
        base,
        restored["a"],
        restored["b"],
        vector,
        restored["alpha"] / restored["rank"],
    )

    rendered_tokens = [
        "<bos>",
        "user",
        "<image>",
        "extract",
        "assistant",
        "<receipt>",
        "<total>",
        "37.50",
        "</total>",
        "</receipt>",
    ]
    labels = [
        -100 if index <= 4 else index
        for index in range(len(rendered_tokens))
    ]
    image_placeholder_count = rendered_tokens.count("<image>")
    visual_positions_per_placeholder = 256
    accounted_sequence_length = (
        len(rendered_tokens)
        - image_placeholder_count
        + image_placeholder_count * visual_positions_per_placeholder
    )
    valid_document = "<receipt><total>37.50</total></receipt>"
    invalid_document = "<receipt><total>37.50</receipt>"
    parsed_total = _parse_total(valid_document)

    theoretical_parameters = rank * (
        len(adapter_a[0]) + len(adapter_b)
    )
    actual_parameters = sum(len(row) for row in adapter_a) + sum(
        len(row) for row in adapter_b
    )

    return {
        "summary": (
            "在手写小矩阵上计算 LoRA 有限差分更新与 merge parity，并对 "
            "toy adapter 矩阵做 JSON round-trip。CORD 部分只对手写 token "
            "列表按 256 个视觉位置记账，不代表真实 processor 或 collator。"
        ),
        "metrics": {
            "adapter_restore_scope": (
                "toy_adapter_matrices_only_not_optimizer_state"
            ),
            "sequence_accounting_scope": (
                "declared_visual_positions_not_processor_output"
            ),
            "rank": rank,
            "alpha": alpha,
            "scale": scale,
            "theoretical_adapter_parameters": theoretical_parameters,
            "actual_adapter_parameters": actual_parameters,
            "initial_loss": round(initial_loss, 8),
            "updated_loss": round(updated_loss, 8),
            "merge_max_abs_error": round(merge_max_error, 12),
            "loss_bearing_tokens": sum(label != -100 for label in labels),
            "pre_replacement_length": len(rendered_tokens),
            "image_placeholder_count": image_placeholder_count,
            "declared_visual_positions_per_placeholder": (
                visual_positions_per_placeholder
            ),
            "accounted_sequence_length": accounted_sequence_length,
            "parsed_total": parsed_total,
        },
        "checks": {
            "LoRA参数量公式与矩阵元素数一致": (
                theoretical_parameters == actual_parameters
            ),
            "这组toy矩阵只更新adapter可降低目标": (
                updated_loss < initial_loss
            ),
            "toy base矩阵数值未被改写": base
            == [[0.2, -0.1, 0.3], [-0.4, 0.5, 0.1]],
            "toy矩阵merge前后输出在容差内一致": all(
                math.isclose(
                    left,
                    right,
                    rel_tol=0.0,
                    abs_tol=1e-10,
                )
                for left, right in zip(unmerged, merged)
            ),
            "仅恢复toy adapter状态后输出在容差内一致": all(
                math.isclose(
                    left,
                    right,
                    rel_tol=0.0,
                    abs_tol=1e-10,
                )
                for left, right in zip(unmerged, restored_output)
            ),
            "prompt与image位置不承担监督": all(
                label == -100 for label in labels[:5]
            ),
            "assistant内容承担监督": all(
                label != -100 for label in labels[5:]
            ),
            "按每个占位符256个视觉位置记账后长度正确": (
                image_placeholder_count == 1
                and accounted_sequence_length == 265
            ),
            "合法结构化输出可解析": math.isclose(
                parsed_total or 0.0,
                37.5,
                abs_tol=1e-12,
            ),
            "损坏的XML不会被当成正确结果": (
                _parse_total(invalid_document) is None
            ),
        },
    }


LESSON = LessonExperiment(
    lesson_id="18",
    title="Nemotron Omni LoRA 微调机制",
    question="LoRA 为什么能冻结大模型基座，同时保存并合并一个低秩更新？",
    run=run,
)
