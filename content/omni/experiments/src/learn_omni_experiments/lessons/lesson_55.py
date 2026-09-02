from __future__ import annotations

import math
from typing import Any

from ..core import LessonExperiment


HIDDEN = [1.0, 0.50, -0.25, 0.125]
TEXT_LABEL = 0
ACTION_BINS = 8
ACTION_RANGES = (
    ("x", -1.0, 1.0),
    ("y", -1.0, 1.0),
    ("z", -1.0, 1.0),
    ("roll", -1.0, 1.0),
    ("pitch", -1.0, 1.0),
    ("yaw", -1.0, 1.0),
    ("gripper", 0.0, 1.0),
)
PITCH_DIM = 4
W_TEXT = [
    [2.40, 1.20, -0.40, 0.30],
    [0.80, 0.40, 0.20, 0.10],
    [0.10, -0.20, 0.30, 0.05],
    [-0.50, 0.10, 0.40, -0.20],
]
W_VIS = [
    [0.90, 0.20, -0.10, 0.15],
    [0.10, 0.80, 0.25, -0.05],
    [-0.20, 0.15, 0.70, 0.30],
    [0.25, -0.10, 0.05, 0.85],
]
W_ACT = [
    [0.08, -0.10, 0.04, 0.16],
    [0.20, -0.80, 0.10, 0.00],
    [0.00, 0.00, 0.00, 0.00],
    [0.05, 0.10, 0.00, 0.00],
    [-0.40, -0.10, 0.20, 0.00],
    [0.70, 0.20, 0.00, 0.00],
    [0.20, 0.20, 0.00, 0.00],
]


def half_up(value: float) -> int:
    """Round half toward +inf, matching JavaScript Math.round for lab parity."""
    return math.floor(value + 0.5)


def qmax(bits: int) -> int:
    if bits < 2:
        raise ValueError("bits must be at least 2")
    return (1 << (bits - 1)) - 1


def absmax_scale(values: list[float], bits: int) -> float:
    peak = max(abs(value) for value in values)
    if peak == 0.0:
        return 1.0
    return peak / qmax(bits)


def quantize_tensor(values: list[float], bits: int) -> tuple[list[float], float, list[int]]:
    scale = absmax_scale(values, bits)
    codes: list[int] = []
    dequantized: list[float] = []
    limit = qmax(bits)
    for value in values:
        code = int(half_up(value / scale))
        code = max(-limit, min(limit, code))
        codes.append(code)
        dequantized.append(code * scale)
    return dequantized, scale, codes


def quantize_rows(matrix: list[list[float]], bits: int) -> list[list[float]]:
    return [quantize_tensor(row, bits)[0] for row in matrix]


def flatten(matrix: list[list[float]]) -> list[float]:
    return [value for row in matrix for value in row]


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [dot(row, vector) for row in matrix]


def softmax(logits: list[float]) -> list[float]:
    peak = max(logits)
    weights = [math.exp(logit - peak) for logit in logits]
    total = sum(weights)
    return [weight / total for weight in weights]


def cross_entropy(probabilities: list[float], label: int) -> float:
    return -math.log(max(probabilities[label], 1e-12))


def mse(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("mse requires equal non-empty vectors")
    return sum((a - b) * (a - b) for a, b in zip(left, right)) / len(left)


def l2(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) * (a - b) for a, b in zip(left, right)))


def uniform_bin(value: float, low: float, high: float, bins: int) -> int:
    if high <= low:
        raise ValueError("action range must be non-empty")
    if value <= low:
        return 0
    if value >= high:
        return bins - 1
    width = (high - low) / bins
    return min(bins - 1, int(math.floor((value - low) / width)))


def bin_width(low: float, high: float, bins: int) -> float:
    return (high - low) / bins


def action_bins(values: list[float]) -> list[int]:
    return [
        uniform_bin(value, low, high, ACTION_BINS)
        for value, (_name, low, high) in zip(values, ACTION_RANGES)
    ]


def top1(logits: list[float]) -> int:
    winner = 0
    for index, logit in enumerate(logits):
        if logit > logits[winner]:
            winner = index
    return winner


def margin(logits: list[float]) -> float:
    ordered = sorted(logits, reverse=True)
    return ordered[0] - ordered[1]


def evaluate(weight_text: list[list[float]], weight_vis: list[list[float]], weight_act: list[list[float]]) -> dict[str, Any]:
    text_logits = matvec(weight_text, HIDDEN)
    text_probs = softmax(text_logits)
    vis_hat = matvec(weight_vis, HIDDEN)
    vis_true = matvec(W_VIS, HIDDEN)
    actions = matvec(weight_act, HIDDEN)
    bins = action_bins(actions)
    return {
        "text_logits": text_logits,
        "text_probs": text_probs,
        "text_top1": top1(text_logits),
        "text_ce": cross_entropy(text_probs, TEXT_LABEL),
        "text_margin": margin(text_logits),
        "vis_hat": vis_hat,
        "vis_true": vis_true,
        "vis_l2": l2(vis_hat, vis_true),
        "vis_mse": mse(vis_hat, vis_true),
        "actions": actions,
        "bins": bins,
        "weight_mse_text": mse(flatten(W_TEXT), flatten(weight_text)),
        "weight_mse_vis": mse(flatten(W_VIS), flatten(weight_vis)),
        "weight_mse_act": mse(flatten(W_ACT), flatten(weight_act)),
    }


def run() -> dict[str, Any]:
    fp = evaluate(W_TEXT, W_VIS, W_ACT)
    w8_text = quantize_rows(W_TEXT, 8)
    w8_vis = quantize_rows(W_VIS, 8)
    w8_act = quantize_rows(W_ACT, 8)
    w4_text = quantize_rows(W_TEXT, 4)
    w4_vis = quantize_rows(W_VIS, 4)
    w4_act = quantize_rows(W_ACT, 4)
    q8 = evaluate(w8_text, w8_vis, w8_act)
    q4 = evaluate(w4_text, w4_vis, w4_act)

    pitch_fp = fp["actions"][PITCH_DIM]
    pitch_8 = q8["actions"][PITCH_DIM]
    pitch_4 = q4["actions"][PITCH_DIM]
    pitch_row_4, pitch_scale_4, pitch_codes_4 = quantize_tensor(W_ACT[PITCH_DIM], 4)
    pitch_width = bin_width(*ACTION_RANGES[PITCH_DIM][1:], ACTION_BINS)
    jumped_8 = [
        ACTION_RANGES[index][0]
        for index, (left, right) in enumerate(zip(fp["bins"], q8["bins"]))
        if left != right
    ]
    jumped_4 = [
        ACTION_RANGES[index][0]
        for index, (left, right) in enumerate(zip(fp["bins"], q4["bins"]))
        if left != right
    ]

    checks = {
        "eight_bit_keeps_every_action_bin": q8["bins"] == fp["bins"] and jumped_8 == [],
        "four_bit_pitch_crosses_bin_boundary": (
            fp["bins"][PITCH_DIM] == 2
            and q8["bins"][PITCH_DIM] == 2
            and q4["bins"][PITCH_DIM] == 1
            and pitch_fp == -0.5
            and abs(pitch_8 + 0.5) < 1e-12
            and pitch_4 < -0.5
            and jumped_4 == ["pitch"]
        ),
        "text_top1_unchanged_at_four_bit": (
            fp["text_top1"] == TEXT_LABEL
            and q8["text_top1"] == TEXT_LABEL
            and q4["text_top1"] == TEXT_LABEL
            and q4["text_margin"] > 2.0
        ),
        "per_modality_error_rises_at_four_bit": (
            q4["weight_mse_text"] > q8["weight_mse_text"]
            and q4["weight_mse_vis"] > q8["weight_mse_vis"]
            and q4["weight_mse_act"] > q8["weight_mse_act"]
            and q4["vis_l2"] > q8["vis_l2"]
            and q8["vis_l2"] > 0.0
        ),
        "pitch_sits_on_bin_boundary": (
            abs(pitch_fp - ACTION_RANGES[PITCH_DIM][1] - 2 * pitch_width) < 1e-12
            and pitch_width == 0.25
        ),
        "four_bit_pitch_row_uses_absmax_grid": (
            pitch_codes_4 == [-7, -2, 4, 0]
            and abs(pitch_scale_4 - 0.4 / 7) < 1e-12
            and abs(pitch_row_4[0] + 0.4) < 1e-12
        ),
        "half_up_matches_lab_rounding": half_up(-3.5) == -3 and half_up(3.5) == 4,
        "text_and_action_heads_are_separate_rows": (
            len(W_TEXT) == 4
            and len(W_ACT) == 7
            and len(W_VIS) == 4
            and fp["text_ce"] > 0.0
            and q4["text_ce"] > 0.0
        ),
    }

    return {
        "summary": (
            "同一隐藏向量上，按行对称 absmax 量化三个头："
            "8 bit 时七维动作 bin 与文本 top-1 都与全精度一致；"
            "4 bit 时 pitch 从箱 2 跳到箱 1，文本 top-1 仍是标签 0。"
        ),
        "metrics": {
            "bits_8": 8,
            "bits_4": 4,
            "hidden": HIDDEN,
            "text_label": TEXT_LABEL,
            "text_top1_fp": fp["text_top1"],
            "text_top1_8": q8["text_top1"],
            "text_top1_4": q4["text_top1"],
            "text_ce_fp": round(fp["text_ce"], 6),
            "text_ce_8": round(q8["text_ce"], 6),
            "text_ce_4": round(q4["text_ce"], 6),
            "text_margin_fp": round(fp["text_margin"], 6),
            "text_margin_8": round(q8["text_margin"], 6),
            "text_margin_4": round(q4["text_margin"], 6),
            "vis_l2_8": round(q8["vis_l2"], 6),
            "vis_l2_4": round(q4["vis_l2"], 6),
            "weight_mse_text_8": round(q8["weight_mse_text"], 8),
            "weight_mse_text_4": round(q4["weight_mse_text"], 8),
            "weight_mse_vis_8": round(q8["weight_mse_vis"], 8),
            "weight_mse_vis_4": round(q4["weight_mse_vis"], 8),
            "weight_mse_act_8": round(q8["weight_mse_act"], 8),
            "weight_mse_act_4": round(q4["weight_mse_act"], 8),
            "action_fp": [round(value, 6) for value in fp["actions"]],
            "action_8": [round(value, 6) for value in q8["actions"]],
            "action_4": [round(value, 6) for value in q4["actions"]],
            "bins_fp": fp["bins"],
            "bins_8": q8["bins"],
            "bins_4": q4["bins"],
            "jumped_dims_8": jumped_8,
            "jumped_dims_4": jumped_4,
            "pitch_fp": pitch_fp,
            "pitch_8": pitch_8,
            "pitch_4": round(pitch_4, 6),
            "pitch_bin_fp": fp["bins"][PITCH_DIM],
            "pitch_bin_8": q8["bins"][PITCH_DIM],
            "pitch_bin_4": q4["bins"][PITCH_DIM],
            "pitch_bin_width": pitch_width,
            "pitch_scale_4": pitch_scale_4,
            "pitch_codes_4": pitch_codes_4,
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="55",
    title="测量端侧量化对各模态 token 的损伤",
    question="同一序列上把三个头量化到 4 bit 时，为什么动作 bin 会先跳类，而文本 top-1 仍不变？",
    run=run,
)
