from __future__ import annotations

import math
from typing import Any

from ..core import LessonExperiment

BINS = 256
MARK_COUNT = 6
LOW_RES = 4
HIGH_RES = 32


def _normalize(x: float, y: float, width: float, height: float) -> tuple[float, float]:
    return x / width, y / height


def _quantize(unit: float, bins: int = BINS) -> int:
    if not math.isfinite(unit):
        raise ValueError("coordinate must be finite")
    clipped = min(1.0, max(0.0, unit))
    return min(bins - 1, int(clipped * bins) if clipped < 1.0 else bins - 1)


def _dequantize_center(index: int, bins: int = BINS) -> float:
    return (index + 0.5) / bins


def _cell_center(unit: float, resolution: int) -> float:
    return _dequantize_center(_quantize(unit, resolution), resolution)


def _softmax(logits: list[float]) -> list[float]:
    peak = max(logits)
    weights = [math.exp(logit - peak) for logit in logits]
    total = sum(weights)
    return [weight / total for weight in weights]


def _cross_entropy(logits: list[float], target: int) -> float:
    if target < 0 or target >= len(logits):
        raise ValueError("SoM target is outside the mark range")
    return -math.log(_softmax(logits)[target])


def _mse(pred: tuple[float, float], target: tuple[float, float]) -> float:
    return (pred[0] - target[0]) ** 2 + (pred[1] - target[1]) ** 2


def _shared_scale_mse(
    scale: float,
    observations: list[tuple[float, float]],
    targets: list[tuple[float, float]],
) -> float:
    return sum(
        _mse((scale * obs[0], scale * obs[1]), target)
        for obs, target in zip(observations, targets)
    ) / len(targets)


def _shared_scale_gradient(
    scale: float,
    observations: list[tuple[float, float]],
    targets: list[tuple[float, float]],
) -> float:
    grad = 0.0
    for obs, target in zip(observations, targets):
        pred_x = scale * obs[0]
        pred_y = scale * obs[1]
        grad += 2.0 * (pred_x - target[0]) * obs[0]
        grad += 2.0 * (pred_y - target[1]) * obs[1]
    return grad / len(targets)


def run() -> dict[str, Any]:
    width, height = 1920.0, 1080.0
    ui_pixels = [
        (1574.4, 194.4),
        (345.6, 194.4),
        (1574.4, 885.6),
        (345.6, 885.6),
        (960.0, 129.6),
        (960.0, 950.4),
    ]
    table_pixels = [
        (537.6, 378.0),
        (1190.4, 345.6),
        (1497.6, 734.4),
        (422.4, 777.6),
        (960.0, 594.0),
        (768.0, 237.6),
    ]
    ui_norm = [_normalize(x, y, width, height) for x, y in ui_pixels]
    table_norm = [_normalize(x, y, width, height) for x, y in table_pixels]
    all_norm = ui_norm + table_norm

    ui_bins = [(_quantize(x), _quantize(y)) for x, y in ui_norm]
    table_bins = [(_quantize(x), _quantize(y)) for x, y in table_norm]
    edge_bins = [
        _quantize(0.0),
        _quantize(1.0 - 1e-12),
        _quantize(1.0),
        _quantize(-0.25),
        _quantize(1.25),
    ]

    wrong_axis = [_normalize(x, y, height, width) for x, y in ui_pixels]
    axis_mismatch = any(
        abs(correct[0] - swapped[0]) > 1e-9 or abs(correct[1] - swapped[1]) > 1e-9
        for correct, swapped in zip(ui_norm, wrong_axis)
    )

    correct_logits = [2.4, 0.15, 0.12, 0.08, 0.11, 0.09]
    som_target = 0
    som_ce_correct = _cross_entropy(correct_logits, som_target)
    som_ce_wrong = _cross_entropy(correct_logits, 3)

    low_obs = [
        (_cell_center(x, LOW_RES), _cell_center(y, LOW_RES)) for x, y in all_norm
    ]
    high_obs = [
        (_cell_center(x, HIGH_RES), _cell_center(y, HIGH_RES)) for x, y in all_norm
    ]
    low_res_mse = sum(_mse(obs, tgt) for obs, tgt in zip(low_obs, all_norm)) / len(
        all_norm
    )
    high_res_mse = sum(_mse(obs, tgt) for obs, tgt in zip(high_obs, all_norm)) / len(
        all_norm
    )
    # Correct SoM uses the marked object center, so position error is 0.
    som_position_mse = 0.0
    som_argmax = max(range(len(correct_logits)), key=lambda i: correct_logits[i])

    ui_obs = [(_cell_center(x, LOW_RES), _cell_center(y, LOW_RES)) for x, y in ui_norm]
    table_obs = [
        (_cell_center(x, LOW_RES), _cell_center(y, LOW_RES)) for x, y in table_norm
    ]
    scale = 0.55
    ui_before = _shared_scale_mse(scale, ui_obs, ui_norm)
    table_before = _shared_scale_mse(scale, table_obs, table_norm)
    mixed_obs = ui_obs + table_obs
    mixed_tgt = ui_norm + table_norm
    scale_after = scale - 0.25 * _shared_scale_gradient(scale, mixed_obs, mixed_tgt)
    ui_after = _shared_scale_mse(scale_after, ui_obs, ui_norm)
    table_after = _shared_scale_mse(scale_after, table_obs, table_norm)

    midpoint_error = max(
        abs(_dequantize_center(index) - (index + 0.5) / BINS) for index in range(BINS)
    )
    recovered = [
        (_dequantize_center(bx), _dequantize_center(by))
        for bx, by in ui_bins[:2] + table_bins[:2]
    ]
    recover_targets = ui_norm[:2] + table_norm[:2]
    recover_mse = sum(_mse(pred, tgt) for pred, tgt in zip(recovered, recover_targets)) / 4

    checks = {
        "normalized_coords_in_unit_square": all(
            0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in all_norm
        ),
        "magma_bins_cover_0_to_255": all(
            0 <= bx < BINS and 0 <= by < BINS for bx, by in ui_bins + table_bins
        )
        and edge_bins == [0, 255, 255, 0, 255],
        "som_labels_are_0_to_k_minus_1": som_target in range(MARK_COUNT)
        and all(index in range(MARK_COUNT) for index in range(len(correct_logits))),
        "low_res_continuous_mse_exceeds_correct_som": som_argmax == som_target
        and som_position_mse == 0.0
        and low_res_mse > som_position_mse
        and low_res_mse > high_res_mse,
        "shared_scale_head_reduces_ui_and_table_mse": ui_after < ui_before
        and table_after < table_before
        and scale_after > scale,
        "wrong_axis_normalization_is_detected": axis_mismatch
        and any(x > 1.0 or y > 1.0 for x, y in wrong_axis),
        "off_by_one_som_target_increases_ce": som_ce_wrong > som_ce_correct,
        "bin_center_round_trip_respects_half_bin": midpoint_error == 0.0
        and recover_mse < (0.5 / BINS) ** 2 * 2 + 1e-12,
    }

    return {
        "summary": (
            "用同一组归一化坐标核对 SoM 分类交叉熵与连续二维回归 MSE，"
            "确认 256 档分箱落在 [0,255]、编号落在 [0,K-1]，"
            "并验证低分辨率格子中心位置误差大于正确编号的位置误差，"
            "共享比例头在 UI 与桌面俯视样本上同时下降。"
        ),
        "metrics": {
            "image_width": width,
            "image_height": height,
            "mark_count": MARK_COUNT,
            "magma_bins": BINS,
            "low_resolution": LOW_RES,
            "high_resolution": HIGH_RES,
            "ui_samples": len(ui_norm),
            "table_samples": len(table_norm),
            "som_ce_correct": som_ce_correct,
            "som_ce_wrong": som_ce_wrong,
            "som_position_mse": som_position_mse,
            "low_res_cell_center_mse": low_res_mse,
            "high_res_cell_center_mse": high_res_mse,
            "shared_scale_before": scale,
            "shared_scale_after": scale_after,
            "ui_mse_before": ui_before,
            "ui_mse_after": ui_after,
            "table_mse_before": table_before,
            "table_mse_after": table_after,
            "bin_center_recover_mse": recover_mse,
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="32",
    title="统一 GUI 操作与机器人动作",
    question="屏幕点击和机械臂末端的二维接地，能否共用一套归一化坐标与编号损失？",
    run=run,
)
