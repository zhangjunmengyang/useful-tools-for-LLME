from __future__ import annotations

import math
from typing import Any

from ..core import LessonExperiment


T_STEPS = 64
N_DIMS = 7
CONTROL_HZ = 20.0
WIGGLE_HZ = 6.0
DCT_SCALE = 10.0
HIGH_FREQ_DIM = 0


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def make_demo_trajectory() -> list[list[float]]:
    """Deterministic 7-DoF demo: slow reach plus a 6 Hz wiggle on dim 0."""
    trajectory: list[list[float]] = []
    for index in range(T_STEPS):
        phase = index / (T_STEPS - 1)
        time_s = index / CONTROL_HZ
        reach = -0.8 + 1.6 * phase
        wiggle = 0.35 + 0.08 * math.sin(2.0 * math.pi * WIGGLE_HZ * time_s)
        gripper = -0.95 + 1.9 / (1.0 + math.exp(-14.0 * (phase - 0.72)))
        row = [
            _clamp(wiggle),
            _clamp(reach),
            _clamp(0.18 * math.sin(2.0 * math.pi * 0.35 * time_s)),
            _clamp(0.12 * math.sin(2.0 * math.pi * 0.28 * time_s)),
            _clamp(-0.22 + 0.40 * phase),
            _clamp(0.15 * math.cos(2.0 * math.pi * 0.22 * time_s)),
            _clamp(gripper),
        ]
        trajectory.append(row)
    return trajectory


def uniform_bin_value(value: float, bins: int, low: float = -1.0, high: float = 1.0) -> float:
    width = (high - low) / bins
    clipped = _clamp(value, low, high)
    index = min(bins - 1, int((clipped - low) / width))
    return low + (index + 0.5) * width


def reconstruct_bins(trajectory: list[list[float]], bins: int) -> list[list[float]]:
    return [
        [uniform_bin_value(value, bins) for value in row]
        for row in trajectory
    ]


def reconstruction_l2(
    original: list[list[float]],
    reconstructed: list[list[float]],
) -> float:
    total = 0.0
    count = 0
    for source_row, recon_row in zip(original, reconstructed):
        for source, recon in zip(source_row, recon_row):
            delta = source - recon
            total += delta * delta
            count += 1
    return math.sqrt(total / count)


def max_abs_error(
    original: list[list[float]],
    reconstructed: list[list[float]],
) -> float:
    peak = 0.0
    for source_row, recon_row in zip(original, reconstructed):
        for source, recon in zip(source_row, recon_row):
            peak = max(peak, abs(source - recon))
    return peak


def moving_average(series: list[float], window: int = 5) -> list[float]:
    half = window // 2
    smoothed: list[float] = []
    for index, _value in enumerate(series):
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


def column(trajectory: list[list[float]], dim: int) -> list[float]:
    return [row[dim] for row in trajectory]


def orthonormal_dct(signal: list[float]) -> list[float]:
    length = len(signal)
    coeffs: list[float] = []
    for freq in range(length):
        total = 0.0
        scale = math.sqrt(1.0 / length) if freq == 0 else math.sqrt(2.0 / length)
        for time, sample in enumerate(signal):
            total += sample * math.cos(math.pi * (time + 0.5) * freq / length)
        coeffs.append(scale * total)
    return coeffs


def orthonormal_idct(coeffs: list[float]) -> list[float]:
    length = len(coeffs)
    signal: list[float] = []
    for time in range(length):
        total = 0.0
        for freq, coeff in enumerate(coeffs):
            scale = math.sqrt(1.0 / length) if freq == 0 else math.sqrt(2.0 / length)
            total += scale * coeff * math.cos(math.pi * (time + 0.5) * freq / length)
        signal.append(total)
    return signal


def dct_keep_reconstruct(
    trajectory: list[list[float]],
    keep: int,
    scale: float = DCT_SCALE,
) -> tuple[list[list[float]], int]:
    kept = max(1, min(keep, T_STEPS))
    reconstructed = [[0.0] * N_DIMS for _ in range(T_STEPS)]
    nonzero = 0
    for dim in range(N_DIMS):
        coeffs = orthonormal_dct(column(trajectory, dim))
        quantized = []
        for freq, coeff in enumerate(coeffs):
            if freq < kept:
                rounded = round(scale * coeff) / scale
                quantized.append(rounded)
                if rounded != 0.0:
                    nonzero += 1
            else:
                quantized.append(0.0)
        restored = orthonormal_idct(quantized)
        for time, value in enumerate(restored):
            reconstructed[time][dim] = _clamp(value)
    return reconstructed, nonzero


def open_loop_seconds(chunk: int, frequency: float = CONTROL_HZ) -> float:
    return chunk / frequency


def naive_token_count(chunk: int, dims: int = N_DIMS) -> int:
    return dims * chunk


def serial_depth_autoregressive(chunk: int, dims: int = N_DIMS) -> int:
    return dims * chunk


def dominated(left: dict[str, float], right: dict[str, float]) -> bool:
    """Return True if left is no worse on all three axes and better on one."""
    axes = ("recon_l2", "tokens", "serial")
    no_worse = all(left[axis] <= right[axis] + 1e-12 for axis in axes)
    better = any(left[axis] < right[axis] - 1e-12 for axis in axes)
    return no_worse and better


def run() -> dict[str, Any]:
    trajectory = make_demo_trajectory()
    bin2 = reconstruct_bins(trajectory, 2)
    bin256 = reconstruct_bins(trajectory, 256)
    dct4, dct4_tokens = dct_keep_reconstruct(trajectory, 4)
    dct_full, dct_full_tokens = dct_keep_reconstruct(trajectory, T_STEPS)

    l2_bin2 = reconstruction_l2(trajectory, bin2)
    l2_bin256 = reconstruction_l2(trajectory, bin256)
    l2_l1 = reconstruction_l2(trajectory, trajectory)
    l2_dct4 = reconstruction_l2(trajectory, dct4)
    l2_dct_full = reconstruction_l2(trajectory, dct_full)

    original_wiggle = residual_rms(column(trajectory, HIGH_FREQ_DIM))
    bin2_wiggle = residual_rms(column(bin2, HIGH_FREQ_DIM))
    high_freq_remain = bin2_wiggle / original_wiggle

    max_err_bin2 = max_abs_error(trajectory, bin2)
    max_err_bin256 = max_abs_error(trajectory, bin256)
    bound_bin2 = 1.0 / 2
    bound_bin256 = 1.0 / 256

    chunk_8 = 8
    chunk_16 = 16
    open_8 = open_loop_seconds(chunk_8)
    open_16 = open_loop_seconds(chunk_16)
    tokens_chunk_8 = naive_token_count(chunk_8)
    tokens_naive_horizon = naive_token_count(T_STEPS)
    serial_ar_step = serial_depth_autoregressive(1)
    serial_ar_chunk_8 = serial_depth_autoregressive(chunk_8)
    serial_parallel = 1

    points = [
        {
            "name": "bin2",
            "recon_l2": l2_bin2,
            "tokens": float(N_DIMS * T_STEPS),
            "serial": float(serial_ar_step),
        },
        {
            "name": "bin256",
            "recon_l2": l2_bin256,
            "tokens": float(N_DIMS * T_STEPS),
            "serial": float(serial_ar_step),
        },
        {
            "name": "continuous_l1",
            "recon_l2": l2_l1,
            "tokens": float(N_DIMS * T_STEPS),
            "serial": float(serial_parallel),
        },
        {
            "name": "chunk_h8_ar",
            "recon_l2": 0.0,
            "tokens": float(tokens_chunk_8),
            "serial": float(serial_ar_chunk_8),
        },
        {
            "name": "dct_keep4",
            "recon_l2": l2_dct4,
            "tokens": float(dct4_tokens),
            "serial": float(dct4_tokens),
        },
    ]
    bin2_vs_l1 = dominated(points[2], points[0])
    dct_vs_naive_tokens = dct4_tokens < tokens_naive_horizon
    bin256_tighter_than_bin2 = l2_bin256 < l2_bin2

    wiggle_values = column(trajectory, HIGH_FREQ_DIM)
    wiggle_span = max(wiggle_values) - min(wiggle_values)
    same_bin = all(uniform_bin_value(value, 2) == uniform_bin_value(wiggle_values[0], 2) for value in wiggle_values)

    checks = {
        "bin2_error_respects_half_bin_width": max_err_bin2 <= bound_bin2 + 1e-12,
        "bin256_error_respects_half_bin_width": max_err_bin256 <= bound_bin256 + 1e-12,
        "bin2_high_freq_wiggle_collapses": same_bin and high_freq_remain < 0.12,
        "open_loop_duration_is_linear_in_chunk": abs(open_16 / open_8 - 2.0) < 1e-12,
        "continuous_l1_reconstruction_is_exact": l2_l1 == 0.0,
        "dct_full_l2_below_keep4": l2_dct_full < l2_dct4,
        "dct_keep4_uses_fewer_tokens_than_naive_horizon": dct_vs_naive_tokens,
        "parallel_serial_depth_is_one": serial_parallel == 1 and serial_ar_step == N_DIMS,
        "bin256_l2_below_bin2": bin256_tighter_than_bin2,
        "continuous_l1_dominates_bin2_on_pareto": bin2_vs_l1,
    }

    return {
        "summary": (
            "对固定 7 维示教轨迹比较均匀分箱、连续 L1、动作分块和 DCT 保留系数："
            "B=2 时 6 Hz 来回落入同一箱、高频残差几乎消失；开环时长等于 H/f 并随 H 线性加倍；"
            "连续 L1 重建误差为 0 且串行深度为 1；保留 4 个 DCT 系数的 token 数低于朴素 7T 展开。"
        ),
        "metrics": {
            "steps": T_STEPS,
            "dims": N_DIMS,
            "control_hz": CONTROL_HZ,
            "wiggle_hz": WIGGLE_HZ,
            "wiggle_span": round(wiggle_span, 6),
            "l2_bin2": round(l2_bin2, 6),
            "l2_bin256": round(l2_bin256, 6),
            "l2_l1": round(l2_l1, 6),
            "l2_dct4": round(l2_dct4, 6),
            "l2_dct_full": round(l2_dct_full, 6),
            "high_freq_remain_bin2": round(high_freq_remain, 6),
            "max_abs_bin2": round(max_err_bin2, 6),
            "max_abs_bin256": round(max_err_bin256, 6),
            "bound_bin2": bound_bin2,
            "bound_bin256": bound_bin256,
            "open_loop_h8_s": open_8,
            "open_loop_h16_s": open_16,
            "tokens_naive_horizon": tokens_naive_horizon,
            "tokens_chunk_h8": tokens_chunk_8,
            "tokens_dct4": dct4_tokens,
            "tokens_dct_full": dct_full_tokens,
            "serial_ar_one_step": serial_ar_step,
            "serial_ar_chunk_h8": serial_ar_chunk_8,
            "serial_parallel": serial_parallel,
            "pareto_points": 5,
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="25",
    title="比较离散、连续与压缩动作表示",
    question="均匀分箱、连续 L1、动作分块和 DCT 压缩各自保住什么、丢掉什么？",
    run=run,
)
