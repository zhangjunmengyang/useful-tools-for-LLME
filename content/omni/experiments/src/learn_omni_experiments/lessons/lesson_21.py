from __future__ import annotations

import math
from typing import Any

from ..core import LessonExperiment


# 固定 4×4 余弦相似度。对角是正对，其余是同 batch 负对。
ALIGNED_SIMILARITY = (
    (0.92, 0.11, 0.08, 0.05),
    (0.10, 0.88, 0.14, 0.07),
    (0.06, 0.12, 0.90, 0.09),
    (0.04, 0.08, 0.10, 0.86),
)

# 把文字塔按 batch 下标循环移位：图像 i 的对角正对变成原矩阵的 (i, i+1)。
SHUFFLE_PERM = (1, 2, 3, 0)

# SigLIP 论文 Algorithm 1 的默认初始化：t = exp(log 10) = 10，b = -10。
SIGLIP_T = 10.0
SIGLIP_BIAS = -10.0

TAU_LOW = 0.01
TAU_CLIP_INIT = 0.07
TAU_HIGH = 0.2


def _permute_columns(
    matrix: tuple[tuple[float, ...], ...],
    permutation: tuple[int, ...],
) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(row[index] for index in permutation) for row in matrix)


def _softmax(values: list[float]) -> list[float]:
    peak = max(values)
    exponentials = [math.exp(value - peak) for value in values]
    total = sum(exponentials)
    return [item / total for item in exponentials]


def _log_sigmoid(value: float) -> float:
    if value >= 0.0:
        return -math.log1p(math.exp(-value))
    return value - math.log1p(math.exp(value))


def infonce_parts(
    similarity: tuple[tuple[float, ...], ...],
    temperature: float,
) -> dict[str, Any]:
    size = len(similarity)
    scaled = [
        [similarity[row][col] / temperature for col in range(size)]
        for row in range(size)
    ]
    image_to_text = []
    text_to_image = []
    positive_probabilities = []
    for index in range(size):
        row_prob = _softmax(scaled[index])
        column_prob = _softmax([scaled[row][index] for row in range(size)])
        image_to_text.append(-math.log(row_prob[index]))
        text_to_image.append(-math.log(column_prob[index]))
        positive_probabilities.append(row_prob[index])
    image_loss = sum(image_to_text) / size
    text_loss = sum(text_to_image) / size
    return {
        "image_to_text": image_loss,
        "text_to_image": text_loss,
        "loss": 0.5 * (image_loss + text_loss),
        "positive_probabilities": positive_probabilities,
        "row_softmax_sums": [
            sum(_softmax(scaled[row])) for row in range(size)
        ],
    }


def sigmoid_loss(
    similarity: tuple[tuple[float, ...], ...],
    temperature: float = SIGLIP_T,
    bias: float = SIGLIP_BIAS,
) -> float:
    """Pairwise logistic loss. Algorithm 1 divides the N×N sum by N, not N²."""
    size = len(similarity)
    total = 0.0
    for row in range(size):
        for col in range(size):
            label = 1.0 if row == col else -1.0
            logit = temperature * similarity[row][col] + bias
            total += _log_sigmoid(label * logit)
    return -total / size


def _add_constant(
    matrix: tuple[tuple[float, ...], ...],
    shift: float,
) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(value + shift for value in row) for row in matrix)


def run() -> dict[str, Any]:
    shuffled = _permute_columns(ALIGNED_SIMILARITY, SHUFFLE_PERM)
    shifted = _add_constant(ALIGNED_SIMILARITY, 0.30)

    aligned_low = infonce_parts(ALIGNED_SIMILARITY, TAU_LOW)
    aligned_init = infonce_parts(ALIGNED_SIMILARITY, TAU_CLIP_INIT)
    aligned_high = infonce_parts(ALIGNED_SIMILARITY, TAU_HIGH)
    shuffled_init = infonce_parts(shuffled, TAU_CLIP_INIT)
    shifted_init = infonce_parts(shifted, TAU_CLIP_INIT)

    aligned_sigmoid = sigmoid_loss(ALIGNED_SIMILARITY)
    shuffled_sigmoid = sigmoid_loss(shuffled)
    shifted_sigmoid = sigmoid_loss(shifted)

    peak_low = max(aligned_low["positive_probabilities"])
    peak_high = max(aligned_high["positive_probabilities"])
    softmax_error = max(
        abs(total - 1.0) for total in aligned_init["row_softmax_sums"]
    )

    diagonal_is_row_max = all(
        row[index] == max(row)
        for index, row in enumerate(ALIGNED_SIMILARITY)
    )
    shuffled_diagonal_is_not_row_max = all(
        row[index] != max(row) for index, row in enumerate(shuffled)
    )

    checks = {
        "shuffled_infonce_not_below_aligned": (
            shuffled_init["loss"] >= aligned_init["loss"]
        ),
        "temperature_raise_drops_positive_peak": peak_low > peak_high,
        "shuffled_sigmoid_not_below_aligned": (
            shuffled_sigmoid >= aligned_sigmoid
        ),
        "infonce_is_mean_of_both_directions": math.isclose(
            aligned_init["loss"],
            0.5
            * (
                aligned_init["image_to_text"]
                + aligned_init["text_to_image"]
            ),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "softmax_rows_sum_to_one": softmax_error < 1e-12,
        "infonce_invariant_to_global_shift": math.isclose(
            aligned_init["loss"],
            shifted_init["loss"],
            rel_tol=0.0,
            abs_tol=1e-10,
        ),
        "sigmoid_changes_under_global_shift": not math.isclose(
            aligned_sigmoid,
            shifted_sigmoid,
            rel_tol=0.0,
            abs_tol=1e-8,
        ),
        "aligned_diagonal_is_largest": diagonal_is_row_max,
        "shuffle_breaks_diagonal_maximum": shuffled_diagonal_is_not_row_max,
    }

    return {
        "summary": (
            "在固定 4×4 相似度矩阵上手算对称 InfoNCE 与 SigLIP 的 pairwise "
            "sigmoid 损失：打乱文字配对后两种损失都不低于对齐矩阵；"
            "温度从 0.01 升到 0.2 时正对 softmax 峰值下降；"
            "InfoNCE 对全局平移不变，sigmoid 损失会变。"
        ),
        "metrics": {
            "batch_size": len(ALIGNED_SIMILARITY),
            "tau_low": TAU_LOW,
            "tau_clip_init": TAU_CLIP_INIT,
            "tau_high": TAU_HIGH,
            "aligned_infonce_tau_0_01": aligned_low["loss"],
            "aligned_infonce_tau_0_07": aligned_init["loss"],
            "aligned_infonce_tau_0_2": aligned_high["loss"],
            "shuffled_infonce_tau_0_07": shuffled_init["loss"],
            "positive_peak_tau_0_01": peak_low,
            "positive_peak_tau_0_2": peak_high,
            "aligned_sigmoid": aligned_sigmoid,
            "shuffled_sigmoid": shuffled_sigmoid,
            "shifted_sigmoid": shifted_sigmoid,
            "softmax_row_sum_error": softmax_error,
            "siglip_logit_temperature": SIGLIP_T,
            "siglip_bias": SIGLIP_BIAS,
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="21",
    title="用对比学习建立图文共享空间",
    question="温度和错误配对怎样改变 InfoNCE 与 sigmoid 损失？",
    run=run,
)
