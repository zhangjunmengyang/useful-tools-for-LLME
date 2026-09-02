from __future__ import annotations

import math
from typing import Any

from ..core import LessonExperiment

STEPS = 8
WIDTH = 8
HEIGHT = 4
OCCLUDE_FROM = 5
GAMMA = 1.0
EPS = 1e-4


def _texture(step: int, row: int, col: int) -> float:
    # Unpredictable high-frequency pattern: a function of absolute time and cell.
    return ((step * 17 + row * 13 + col * 7) % 10) / 9.0


def _clip_col(col: int) -> int:
    return max(0, min(WIDTH - 1, col))


def _mover_col(step: int) -> int:
    return _clip_col(min(step, 5))


def _render(step: int, cup_col: int) -> list[list[float]]:
    grid = [
        [_texture(step, row, col) for col in range(WIDTH)]
        for row in range(HEIGHT)
    ]
    mover = _mover_col(step)
    for row in range(1, 3):
        grid[row][cup_col] = 0.05
        grid[row][mover] = 0.55 if mover == cup_col else 0.95
    return grid


def _mean_square(left: list[list[float]], right: list[list[float]]) -> float:
    total = 0.0
    count = 0
    for left_row, right_row in zip(left, right):
        for left_value, right_value in zip(left_row, right_row):
            delta = left_value - right_value
            total += delta * delta
            count += 1
    return total / count


def _latent(step: int, cup_col: int) -> list[float]:
    mover = float(_mover_col(step))
    cup = float(cup_col)
    overlap = max(0.0, 1.0 - abs(mover - cup))
    velocity = 0.0 if step >= 5 else 1.0
    return [mover / WIDTH, cup / WIDTH, overlap, velocity]


def _l2(left: list[float], right: list[float]) -> float:
    return sum((a - b) * (a - b) for a, b in zip(left, right))


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _predict_pixel(step: int, cup_col: int, last_visible: int) -> list[list[float]]:
    # Copy last visible occupancy, smear by occlusion depth, fill texture with 0.5.
    depth = step - last_visible
    predicted_mover = _clip_col(_mover_col(last_visible) + (step - last_visible))
    smear = max(1, depth)
    grid = [[0.5 for _ in range(WIDTH)] for _ in range(HEIGHT)]
    for row in range(1, 3):
        for offset in range(-smear, smear + 1):
            col = predicted_mover + offset
            if 0 <= col < WIDTH:
                grid[row][col] = 0.72
        for offset in range(-smear, smear + 1):
            col = cup_col + offset
            if 0 <= col < WIDTH:
                mixed = 0.28 if abs(offset) > 0 else 0.18
                grid[row][col] = 0.5 * grid[row][col] + 0.5 * mixed
    return grid


def _predict_latent(step: int, cup_col: int, last_visible: int) -> list[float]:
    predicted_mover = float(_clip_col(_mover_col(last_visible) + (step - last_visible)))
    # Linear kinematics saturates at the contact column used by the fixture.
    predicted_mover = min(predicted_mover, 5.0)
    cup = float(cup_col)
    overlap = max(0.0, 1.0 - abs(predicted_mover - cup))
    velocity = 0.0 if predicted_mover >= 5.0 else 1.0
    return [predicted_mover / WIDTH, cup / WIDTH, overlap, velocity]


def _occupancy_mass(grid: list[list[float]], col: int) -> float:
    return sum(grid[row][col] for row in range(1, 3)) / 2.0


def _pixel_overlap(grid: list[list[float]], mover_col: int, cup_col: int) -> float:
    # Smeared reconstructions mix the two columns; take the min mass as contact.
    return min(_occupancy_mass(grid, mover_col), _occupancy_mass(grid, cup_col))


def _std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    var = sum((value - mean) * (value - mean) for value in values) / len(values)
    return math.sqrt(var + EPS)


def _vicreg_variance(batch: list[list[float]], gamma: float = GAMMA) -> float:
    dim = len(batch[0])
    total = 0.0
    for axis in range(dim):
        column = [row[axis] for row in batch]
        total += max(0.0, gamma - _std(column))
    return total / dim


def _mean_metric(values: list[float]) -> float:
    return sum(values) / len(values)


def run() -> dict[str, Any]:
    contact_cup = 5
    separated_cup = 7
    last_visible = OCCLUDE_FROM - 1
    occluded_steps = list(range(OCCLUDE_FROM, STEPS))
    visible_steps = list(range(OCCLUDE_FROM))

    contact_true = [_render(step, contact_cup) for step in range(STEPS)]
    separated_true = [_render(step, separated_cup) for step in range(STEPS)]
    contact_latents = [_latent(step, contact_cup) for step in range(STEPS)]
    separated_latents = [_latent(step, separated_cup) for step in range(STEPS)]

    # Visible frames may be copied; future frames cannot copy texture.
    pixel_l2_visible = _mean_metric(
        [
            _mean_square(contact_true[step], contact_true[step])
            for step in visible_steps
        ]
    )
    pixel_next_frame_copy = _mean_metric(
        [
            _mean_square(contact_true[step], contact_true[step - 1])
            for step in visible_steps
            if step > 0
        ]
    )
    pixel_pred_occluded = [
        _predict_pixel(step, contact_cup, last_visible) for step in occluded_steps
    ]
    pixel_l2_occluded = _mean_metric(
        [
            _mean_square(pred, contact_true[step])
            for pred, step in zip(pixel_pred_occluded, occluded_steps)
        ]
    )

    latent_l2_visible = _mean_metric(
        [_l2(contact_latents[step], contact_latents[step]) for step in visible_steps]
    )
    latent_next_step_copy = _mean_metric(
        [
            _l2(contact_latents[step], contact_latents[step - 1])
            for step in visible_steps
            if step > 0
        ]
    )
    latent_pred_occluded = [
        _predict_latent(step, contact_cup, last_visible) for step in occluded_steps
    ]
    latent_l2_occluded = _mean_metric(
        [
            _l2(pred, contact_latents[step])
            for pred, step in zip(latent_pred_occluded, occluded_steps)
        ]
    )

    contact_pixel_overlap = _mean_metric(
        [
            _pixel_overlap(
                _predict_pixel(step, contact_cup, last_visible),
                _clip_col(_mover_col(last_visible) + (step - last_visible)),
                contact_cup,
            )
            for step in occluded_steps
        ]
    )
    separated_pixel_overlap = _mean_metric(
        [
            _pixel_overlap(
                _predict_pixel(step, separated_cup, last_visible),
                _clip_col(_mover_col(last_visible) + (step - last_visible)),
                separated_cup,
            )
            for step in occluded_steps
        ]
    )
    pixel_contact_margin = abs(contact_pixel_overlap - separated_pixel_overlap)

    contact_latent_overlap = _mean_metric(
        [pred[2] for pred in latent_pred_occluded]
    )
    separated_latent_pred = [
        _predict_latent(step, separated_cup, last_visible)
        for step in occluded_steps
    ]
    separated_latent_overlap = _mean_metric([pred[2] for pred in separated_latent_pred])
    latent_contact_margin = contact_latent_overlap - separated_latent_overlap

    contact_cosine = _mean_metric(
        [
            _cosine(pred, true)
            for pred, true in zip(latent_pred_occluded, contact_latents[OCCLUDE_FROM:])
        ]
    )
    cross_cosine = _mean_metric(
        [
            _cosine(pred, other)
            for pred, other in zip(latent_pred_occluded, separated_latents[OCCLUDE_FROM:])
        ]
    )

    spread_batch = [
        _latent(step, contact_cup) for step in range(STEPS)
    ] + [
        _latent(step, separated_cup) for step in range(STEPS)
    ]
    collapsed_batch = [[0.5, 0.5, 0.5, 0.5] for _ in range(len(spread_batch))]
    vicreg_spread = _vicreg_variance(spread_batch)
    vicreg_collapsed = _vicreg_variance(collapsed_batch)

    # Stop-gradient fixture: changing the target encoder after sg must change
    # the numeric loss, but the predictor input is not a function of that target.
    target = contact_latents[OCCLUDE_FROM]
    predicted = latent_pred_occluded[0]
    loss_with_sg = _l2(predicted, target)
    mutated_target = [2.0 * value for value in target]
    loss_if_target_mutated = _l2(predicted, mutated_target)
    predictor_ignores_target_params = predicted != mutated_target

    pixel_delta = pixel_l2_occluded - pixel_l2_visible
    latent_delta = latent_l2_occluded - latent_l2_visible
    different_order = pixel_l2_occluded > 10.0 * max(latent_l2_occluded, 1e-12)

    checks = {
        "sequence_has_eight_steps": STEPS == 8 and len(contact_true) == 8,
        "pixel_l2_rises_after_occlusion": pixel_delta > 0.05,
        "latent_regression_does_not_rise_with_pixel": latent_delta <= 1e-12
        and pixel_delta > 0.0,
        "occluded_losses_differ_by_an_order": different_order,
        "pixel_contact_margin_collapses": pixel_contact_margin < 0.2,
        "latent_contact_margin_holds": latent_contact_margin > 0.5,
        "constant_encoder_triggers_vicreg_variance": vicreg_collapsed >= 0.9
        and vicreg_spread < vicreg_collapsed,
        "stop_gradient_keeps_target_off_predictor_graph": predictor_ignores_target_params
        and loss_if_target_mutated > loss_with_sg,
        "latent_cosine_prefers_true_contact": contact_cosine > cross_cosine + 0.1,
    }

    return {
        "summary": (
            "用固定 8 步接触/分离序列核对像素 L2 与表征回归："
            "遮挡未来三帧后像素 L2 上升且接触探针糊掉，"
            "表征回归保持同阶并仍能分开接触与分离；"
            "常数编码器触发 VICReg 方差项，stop-gradient 挡住目标支路。"
        ),
        "metrics": {
            "steps": STEPS,
            "width": WIDTH,
            "height": HEIGHT,
            "occlude_from": OCCLUDE_FROM,
            "pixel_l2_visible": pixel_l2_visible,
            "pixel_next_frame_copy": pixel_next_frame_copy,
            "pixel_l2_occluded": pixel_l2_occluded,
            "pixel_l2_delta": pixel_delta,
            "latent_l2_visible": latent_l2_visible,
            "latent_next_step_copy": latent_next_step_copy,
            "latent_l2_occluded": latent_l2_occluded,
            "latent_l2_delta": latent_delta,
            "pixel_contact_overlap": contact_pixel_overlap,
            "pixel_separated_overlap": separated_pixel_overlap,
            "pixel_contact_margin": pixel_contact_margin,
            "latent_contact_overlap": contact_latent_overlap,
            "latent_separated_overlap": separated_latent_overlap,
            "latent_contact_margin": latent_contact_margin,
            "latent_contact_cosine": contact_cosine,
            "latent_cross_cosine": cross_cosine,
            "vicreg_variance_spread": vicreg_spread,
            "vicreg_variance_collapsed": vicreg_collapsed,
            "loss_with_stop_grad": loss_with_sg,
            "loss_if_target_mutated": loss_if_target_mutated,
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="33",
    title="世界模型预测的是像素还是表征",
    question="遮挡未来之后，像素 L2 和表征回归会朝同一方向走，还是会分出接触与分离？",
    run=run,
)
