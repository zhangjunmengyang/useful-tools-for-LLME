from __future__ import annotations

import math
from typing import Any

from ..core import LessonExperiment


T_STEPS = 64
CONTROL_HZ = 20.0
TOKEN_BUDGET = 168
BIT_BUDGET = 336
H_FIXED = 8
RANGE_LOW = -1.0
RANGE_HIGH = 1.0
RANGE_SPAN = RANGE_HIGH - RANGE_LOW
FINGER_HZ = 2.5
FINGER_AMP = 0.28
FINGER_DIM = 0
TORSO_DIM = 1
DIM_7 = 7
DIM_24 = 24


def _clamp(value: float, low: float = RANGE_LOW, high: float = RANGE_HIGH) -> float:
    return min(high, max(low, value))


def chunk_length(dims: int, token_budget: int = TOKEN_BUDGET) -> int:
    if dims <= 0:
        raise ValueError("dims must be positive")
    return max(1, token_budget // dims)


def bits_per_scalar(dims: int, horizon: int = H_FIXED, bit_budget: int = BIT_BUDGET) -> float:
    return bit_budget / (horizon * dims)


def bins_from_bits(bits: float) -> int:
    return 2 ** max(1, int(math.floor(bits)))


def bin_width(bins: int, span: float = RANGE_SPAN) -> float:
    return span / bins


def half_width(bins: int, span: float = RANGE_SPAN) -> float:
    return bin_width(bins, span) / 2.0


def uniform_bin_value(value: float, bins: int) -> float:
    width = bin_width(bins)
    clipped = _clamp(value)
    index = min(bins - 1, int((clipped - RANGE_LOW) / width))
    return RANGE_LOW + (index + 0.5) * width


def make_demo(dims: int) -> list[list[float]]:
    """Slow torso, 6 Hz finger, mid-band wrist, remaining dims as low-amp drift."""
    trajectory: list[list[float]] = []
    for index in range(T_STEPS):
        phase = index / (T_STEPS - 1)
        time_s = index / CONTROL_HZ
        finger = FINGER_AMP * math.sin(2.0 * math.pi * FINGER_HZ * time_s)
        torso = -0.75 + 1.50 * phase
        wrist = 0.20 * math.sin(2.0 * math.pi * 0.55 * time_s)
        row = [_clamp(finger), _clamp(torso), _clamp(wrist)]
        while len(row) < dims:
            extra_index = len(row)
            drift = 0.06 * math.sin(2.0 * math.pi * 0.15 * time_s + 0.17 * extra_index)
            row.append(_clamp(drift))
        trajectory.append(row[:dims])
    return trajectory


def column(trajectory: list[list[float]], dim: int) -> list[float]:
    return [row[dim] for row in trajectory]


def subsample_indices(horizon: int, steps: int = T_STEPS) -> list[int]:
    if horizon >= steps:
        return list(range(steps))
    if horizon == 1:
        return [0]
    return [index * (steps - 1) // (horizon - 1) for index in range(horizon)]


def reconstruct(
    trajectory: list[list[float]],
    horizon: int,
    bins: int,
) -> list[list[float]]:
    steps = len(trajectory)
    dims = len(trajectory[0])
    anchors = subsample_indices(horizon, steps)
    quantized = [
        [uniform_bin_value(trajectory[time][dim], bins) for dim in range(dims)]
        for time in anchors
    ]
    reconstructed = [[0.0] * dims for _ in range(steps)]
    for dim in range(dims):
        for time in range(steps):
            if time <= anchors[0]:
                reconstructed[time][dim] = quantized[0][dim]
                continue
            if time >= anchors[-1]:
                reconstructed[time][dim] = quantized[-1][dim]
                continue
            right = 1
            while anchors[right] < time:
                right += 1
            left = right - 1
            span = anchors[right] - anchors[left]
            weight = (time - anchors[left]) / span
            reconstructed[time][dim] = (1.0 - weight) * quantized[left][dim] + weight * quantized[right][dim]
    return reconstructed


def reconstruction_l2(original: list[list[float]], reconstructed: list[list[float]]) -> float:
    total = 0.0
    count = 0
    for source_row, recon_row in zip(original, reconstructed):
        for source, recon in zip(source_row, recon_row):
            delta = source - recon
            total += delta * delta
            count += 1
    return math.sqrt(total / count)


def max_abs_error(original: list[list[float]], reconstructed: list[list[float]]) -> float:
    peak = 0.0
    for source_row, recon_row in zip(original, reconstructed):
        for source, recon in zip(source_row, recon_row):
            peak = max(peak, abs(source - recon))
    return peak


def moving_average(series: list[float], window: int = 5) -> list[float]:
    half = window // 2
    smoothed: list[float] = []
    for index in range(len(series)):
        start = max(0, index - half)
        end = min(len(series), index + half + 1)
        smoothed.append(sum(series[start:end]) / (end - start))
    return smoothed


def residual_rms(series: list[float]) -> float:
    baseline = moving_average(series)
    energy = 0.0
    for value, mean_value in zip(series, baseline):
        delta = value - mean_value
        energy += delta * delta
    return math.sqrt(energy / len(series))


def high_freq_remain(original: list[float], reconstructed: list[float]) -> float:
    source = residual_rms(original)
    if source < 1e-12:
        return 0.0
    return residual_rms(reconstructed) / source


def high_freq_error(original: list[float], reconstructed: list[float]) -> float:
    """RMS error of the high-pass residual, using the original slow baseline."""
    baseline = moving_average(original)
    energy = 0.0
    for source, recon, mean_value in zip(original, reconstructed, baseline):
        delta = (source - mean_value) - (recon - mean_value)
        energy += delta * delta
    return math.sqrt(energy / len(original))


def max_anchor_abs_error(trajectory: list[list[float]], horizon: int, bins: int) -> float:
    peak = 0.0
    for time in subsample_indices(horizon, len(trajectory)):
        for value in trajectory[time]:
            peak = max(peak, abs(value - uniform_bin_value(value, bins)))
    return peak


def dim_mse(original: list[list[float]], reconstructed: list[list[float]], dim: int) -> float:
    energy = 0.0
    for source_row, recon_row in zip(original, reconstructed):
        delta = source_row[dim] - recon_row[dim]
        energy += delta * delta
    return energy / len(original)


def open_loop_seconds(horizon: int, frequency: float = CONTROL_HZ) -> float:
    return horizon / frequency


def run() -> dict[str, Any]:
    h_token_7 = chunk_length(DIM_7)
    h_token_24 = chunk_length(DIM_24)
    bits_7 = bits_per_scalar(DIM_7)
    bits_24 = bits_per_scalar(DIM_24)
    bins_7 = bins_from_bits(bits_7)
    bins_24 = bins_from_bits(bits_24)
    width_7 = bin_width(bins_7)
    width_24 = bin_width(bins_24)

    demo_7 = make_demo(DIM_7)
    demo_24 = make_demo(DIM_24)
    recon_token_7 = reconstruct(demo_7, h_token_7, bins=8)
    recon_token_24 = reconstruct(demo_24, h_token_24, bins=8)
    recon_bit_7 = reconstruct(demo_7, H_FIXED, bins_7)
    recon_bit_24 = reconstruct(demo_24, H_FIXED, bins_24)

    finger_remain_token_7 = high_freq_remain(column(demo_7, FINGER_DIM), column(recon_token_7, FINGER_DIM))
    finger_remain_token_24 = high_freq_remain(column(demo_24, FINGER_DIM), column(recon_token_24, FINGER_DIM))
    finger_error_token_7 = high_freq_error(column(demo_7, FINGER_DIM), column(recon_token_7, FINGER_DIM))
    finger_error_token_24 = high_freq_error(column(demo_24, FINGER_DIM), column(recon_token_24, FINGER_DIM))
    finger_mse_token_7 = dim_mse(demo_7, recon_token_7, FINGER_DIM)
    finger_mse_token_24 = dim_mse(demo_24, recon_token_24, FINGER_DIM)
    torso_mse_token_7 = dim_mse(demo_7, recon_token_7, TORSO_DIM)
    torso_mse_token_24 = dim_mse(demo_24, recon_token_24, TORSO_DIM)
    finger_remain_bit_7 = high_freq_remain(column(demo_7, FINGER_DIM), column(recon_bit_7, FINGER_DIM))
    finger_remain_bit_24 = high_freq_remain(column(demo_24, FINGER_DIM), column(recon_bit_24, FINGER_DIM))
    finger_error_bit_7 = high_freq_error(column(demo_7, FINGER_DIM), column(recon_bit_7, FINGER_DIM))
    finger_error_bit_24 = high_freq_error(column(demo_24, FINGER_DIM), column(recon_bit_24, FINGER_DIM))
    finger_mse_bit_7 = dim_mse(demo_7, recon_bit_7, FINGER_DIM)
    finger_mse_bit_24 = dim_mse(demo_24, recon_bit_24, FINGER_DIM)
    torso_mse_bit_7 = dim_mse(demo_7, recon_bit_7, TORSO_DIM)
    torso_mse_bit_24 = dim_mse(demo_24, recon_bit_24, TORSO_DIM)
    max_anchor_bit_7 = max_anchor_abs_error(demo_7, H_FIXED, bins_7)
    max_anchor_bit_24 = max_anchor_abs_error(demo_24, H_FIXED, bins_24)

    open_7 = open_loop_seconds(h_token_7)
    open_24 = open_loop_seconds(h_token_24)
    finger_mse_gain_token = (finger_mse_token_24 + 1e-12) / (finger_mse_token_7 + 1e-12)
    torso_mse_gain_token = (torso_mse_token_24 + 1e-12) / (torso_mse_token_7 + 1e-12)

    checks = {
        "token_budget_conserved_at_d7": h_token_7 * DIM_7 == TOKEN_BUDGET,
        "token_budget_conserved_at_d24": h_token_24 * DIM_24 == TOKEN_BUDGET,
        "bit_budget_splits_to_6_and_1_75": abs(bits_7 - 6.0) < 1e-12 and abs(bits_24 - 1.75) < 1e-12,
        "width_24_coarser_than_width_7": width_24 > width_7 + 1e-12,
        "width_matches_range_over_bins": abs(width_7 - RANGE_SPAN / 64) < 1e-12 and abs(width_24 - RANGE_SPAN / 2) < 1e-12,
        "anchor_quantization_respects_half_width": (
            max_anchor_bit_7 <= half_width(bins_7) + 1e-12
            and max_anchor_bit_24 <= half_width(bins_24) + 1e-12
        ),
        "open_loop_shortens_when_d_rises": open_24 < open_7 - 1e-12,
        "finger_high_freq_error_rises_with_d": finger_error_token_24 > finger_error_token_7 + 0.05,
        "finger_mse_rises_more_than_torso_on_token_budget": finger_mse_gain_token > torso_mse_gain_token + 1e-6,
        "h_fixed_times_d_matches_bit_slots": H_FIXED * DIM_7 * bits_7 == BIT_BUDGET and H_FIXED * DIM_24 * bits_24 == BIT_BUDGET,
    }

    return {
        "summary": (
            "固定 token 预算 168 时 d=7 分得 H=24、d=24 分得 H=7，开环从 1.2 s 缩到 0.35 s；"
            "固定比特预算 336、H=8 时每维从 6 bit（箱宽 0.03125）降到 1.75 bit（箱宽 1.0）；"
            "手指 2.5 Hz 高频重建误差随维数升高而上升，且升幅大于低频躯干。"
        ),
        "metrics": {
            "steps": T_STEPS,
            "control_hz": CONTROL_HZ,
            "finger_hz": FINGER_HZ,
            "token_budget": TOKEN_BUDGET,
            "bit_budget": BIT_BUDGET,
            "h_fixed": H_FIXED,
            "d7": DIM_7,
            "d24": DIM_24,
            "h_token_d7": h_token_7,
            "h_token_d24": h_token_24,
            "open_loop_d7_s": open_7,
            "open_loop_d24_s": open_24,
            "bits_per_scalar_d7": bits_7,
            "bits_per_scalar_d24": bits_24,
            "bins_d7": bins_7,
            "bins_d24": bins_24,
            "width_d7": width_7,
            "width_d24": width_24,
            "half_width_d7": half_width(bins_7),
            "half_width_d24": half_width(bins_24),
            "max_anchor_bit_d7": round(max_anchor_bit_7, 6),
            "max_anchor_bit_d24": round(max_anchor_bit_24, 6),
            "l2_token_d7": round(reconstruction_l2(demo_7, recon_token_7), 6),
            "l2_token_d24": round(reconstruction_l2(demo_24, recon_token_24), 6),
            "l2_bit_d7": round(reconstruction_l2(demo_7, recon_bit_7), 6),
            "l2_bit_d24": round(reconstruction_l2(demo_24, recon_bit_24), 6),
            "finger_remain_token_d7": round(finger_remain_token_7, 6),
            "finger_remain_token_d24": round(finger_remain_token_24, 6),
            "finger_error_token_d7": round(finger_error_token_7, 6),
            "finger_error_token_d24": round(finger_error_token_24, 6),
            "finger_mse_token_d7": round(finger_mse_token_7, 6),
            "finger_mse_token_d24": round(finger_mse_token_24, 6),
            "torso_mse_token_d7": round(torso_mse_token_7, 6),
            "torso_mse_token_d24": round(torso_mse_token_24, 6),
            "finger_mse_gain_token": round(finger_mse_gain_token, 6),
            "torso_mse_gain_token": round(torso_mse_gain_token, 6),
            "finger_remain_bit_d7": round(finger_remain_bit_7, 6),
            "finger_remain_bit_d24": round(finger_remain_bit_24, 6),
            "finger_error_bit_d7": round(finger_error_bit_7, 6),
            "finger_error_bit_d24": round(finger_error_bit_24, 6),
            "finger_mse_bit_d7": round(finger_mse_bit_7, 6),
            "finger_mse_bit_d24": round(finger_mse_bit_24, 6),
            "torso_mse_bit_d7": round(torso_mse_bit_7, 6),
            "torso_mse_bit_d24": round(torso_mse_bit_24, 6),
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="37",
    title="比较高维动作在固定预算下的损失",
    question="动作维数从 7 升到 24、token 预算不变时，丢掉的是开环窗口还是每维量化宽度？高频手指分量先坏在哪？",
    run=run,
)
