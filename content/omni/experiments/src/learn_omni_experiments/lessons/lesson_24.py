from __future__ import annotations

import math
from typing import Any

from ..core import LessonExperiment


BINS = 8
TEXT_VOCAB = 32
ACTION_DIMS = (
    ("x", -1.0, 1.0),
    ("y", -1.0, 1.0),
    ("z", -1.0, 1.0),
    ("roll", -1.0, 1.0),
    ("pitch", -1.0, 1.0),
    ("yaw", -1.0, 1.0),
    ("gripper", 0.0, 1.0),
)
ACTION_VOCAB = len(ACTION_DIMS) * BINS
HIDDEN = 4
EXAMPLE = [0.4, -0.2, 0.0, 0.1, -0.5, 0.8, 0.3]


def _clip_bin(index: int, bins: int) -> int:
    if index < 0:
        return 0
    if index >= bins:
        return bins - 1
    return index


def uniform_bin(value: float, low: float, high: float, bins: int) -> int:
    if high <= low:
        raise ValueError("action range must be non-empty")
    if value <= low:
        return 0
    if value >= high:
        return bins - 1
    width = (high - low) / bins
    return _clip_bin(math.floor((value - low) / width), bins)


def bin_width(low: float, high: float, bins: int) -> float:
    return (high - low) / bins


def bin_center(index: int, low: float, high: float, bins: int) -> float:
    return low + (index + 0.5) * bin_width(low, high, bins)


def encode_action(action: list[float]) -> list[int]:
    tokens: list[int] = []
    for dimension, value in enumerate(action):
        _name, low, high = ACTION_DIMS[dimension]
        index = uniform_bin(value, low, high, BINS)
        tokens.append(TEXT_VOCAB + dimension * BINS + index)
    return tokens


def decode_action(tokens: list[int]) -> list[float]:
    decoded: list[float] = []
    for dimension, token in enumerate(tokens):
        _name, low, high = ACTION_DIMS[dimension]
        start = TEXT_VOCAB + dimension * BINS
        index = token - start
        if index < 0 or index >= BINS:
            raise ValueError("token is outside the slice of this action dimension")
        decoded.append(bin_center(index, low, high, BINS))
    return decoded


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _softmax(logits: list[float]) -> list[float]:
    peak = max(logits)
    exponentials = [math.exp(logit - peak) for logit in logits]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _ce_row_grads(
    hidden: list[float],
    weight_rows: list[list[float]],
    label: int,
) -> list[list[float]]:
    logits = [_dot(row, hidden) for row in weight_rows]
    probabilities = _softmax(logits)
    grads: list[list[float]] = []
    for index, probability in enumerate(probabilities):
        coefficient = probability - (1.0 if index == label else 0.0)
        grads.append([coefficient * feature for feature in hidden])
    return grads


def _max_abs(matrix: list[list[float]]) -> float:
    return max(abs(value) for row in matrix for value in row)


def _language_only_action_grads() -> tuple[float, float]:
    hidden = [0.5, -0.25, 0.75, 0.1]
    language_rows = [
        [0.2 * (row + 1) + 0.05 * column for column in range(HIDDEN)]
        for row in range(TEXT_VOCAB)
    ]
    action_rows = [
        [0.3 * (row + 1) - 0.04 * column for column in range(HIDDEN)]
        for row in range(ACTION_VOCAB)
    ]
    language_grads = _ce_row_grads(hidden, language_rows, label=7)
    # Separate language head: action rows are not in the loss graph.
    action_head_grads = [[0.0] * HIDDEN for _ in range(ACTION_VOCAB)]
    joint_rows = language_rows + action_rows
    joint_grads = _ce_row_grads(hidden, joint_rows, label=7)
    leaked = joint_grads[TEXT_VOCAB:]
    _ = language_grads
    return _max_abs(action_head_grads), _max_abs(leaked)


def run() -> dict[str, Any]:
    example_tokens = encode_action(EXAMPLE)
    reconstructed = decode_action(example_tokens)
    reconstruction_error = [
        abs(original - recovered)
        for original, recovered in zip(EXAMPLE, reconstructed)
    ]
    half_widths = [
        bin_width(low, high, BINS) / 2 for _name, low, high in ACTION_DIMS
    ]

    low_edges = [
        uniform_bin(low, low, high, BINS) for _name, low, high in ACTION_DIMS
    ]
    high_edges = [
        uniform_bin(high, low, high, BINS) for _name, low, high in ACTION_DIMS
    ]
    just_below_second = uniform_bin(-1.0 + bin_width(-1.0, 1.0, BINS) - 1e-12, -1.0, 1.0, BINS)
    at_second = uniform_bin(-1.0 + bin_width(-1.0, 1.0, BINS), -1.0, 1.0, BINS)

    dim_slices = [
        list(range(TEXT_VOCAB + dimension * BINS, TEXT_VOCAB + (dimension + 1) * BINS))
        for dimension in range(len(ACTION_DIMS))
    ]
    flat_ids = [token for slice_ids in dim_slices for token in slice_ids]

    language_head_action_grad, joint_leak_grad = _language_only_action_grads()

    checks = {
        "bin_low_edge_is_zero": low_edges == [0] * len(ACTION_DIMS),
        "bin_high_edge_is_last": high_edges == [BINS - 1] * len(ACTION_DIMS),
        "bin_interval_is_left_closed": just_below_second == 0 and at_second == 1,
        "vocab_offset_has_no_overlap_with_text": min(flat_ids) == TEXT_VOCAB
        and max(flat_ids) == TEXT_VOCAB + ACTION_VOCAB - 1,
        "dim_slices_are_contiguous_and_disjoint": flat_ids
        == list(range(TEXT_VOCAB, TEXT_VOCAB + ACTION_VOCAB)),
        "language_head_only_zeros_action_token_grad": language_head_action_grad == 0.0,
        "joint_softmax_language_loss_leaks_into_action_rows": joint_leak_grad > 0.0,
        "reconstruction_error_bounded_by_half_bin": all(
            error <= width + 1e-12
            for error, width in zip(reconstruction_error, half_widths)
        ),
    }
    return {
        "summary": (
            "对 7 维末端动作做均匀分箱并加上语言词表偏移；"
            "只训独立语言头时动作 token 行梯度为 0，"
            "联合 softmax 的语言 CE 仍会经配分函数漏到动作行。"
        ),
        "metrics": {
            "bins": BINS,
            "text_vocab": TEXT_VOCAB,
            "action_dims": len(ACTION_DIMS),
            "action_vocab": ACTION_VOCAB,
            "example_tokens": example_tokens,
            "reconstruction_l_inf": max(reconstruction_error),
            "language_head_action_grad_abs_max": language_head_action_grad,
            "joint_softmax_action_row_grad_abs_max": joint_leak_grad,
            "first_action_token_id": TEXT_VOCAB,
            "last_action_token_id": TEXT_VOCAB + ACTION_VOCAB - 1,
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="24",
    title="把动作接进视觉语言模型",
    question="7 维动作均匀分箱后如何进入词表，以及只训语言头时动作 token 是否拿到梯度？",
    run=run,
)
