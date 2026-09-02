from __future__ import annotations

import math
from typing import Any

from ..core import LessonExperiment


Z_95 = 1.96
SUITE_OPENVLA_FT = {
    "spatial": 0.847,
    "object": 0.884,
    "goal": 0.792,
    "long": 0.537,
}


def point_estimate(outcomes: list[int]) -> float:
    if not outcomes:
        raise ValueError("outcomes must be non-empty")
    return sum(outcomes) / len(outcomes)


def normal_interval(
    outcomes: list[int],
    z: float = Z_95,
) -> tuple[float, float, float]:
    n = len(outcomes)
    p_hat = point_estimate(outcomes)
    se = math.sqrt(p_hat * (1.0 - p_hat) / n)
    return p_hat, p_hat - z * se, p_hat + z * se


def wilson_interval(
    outcomes: list[int],
    z: float = Z_95,
) -> tuple[float, float, float]:
    n = len(outcomes)
    p_hat = point_estimate(outcomes)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    margin = (
        z
        * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n))
        / denom
    )
    return p_hat, max(0.0, center - margin), min(1.0, center + margin)


def covers(low: float, high: float, value: float) -> bool:
    return low <= value <= high


def suite_macro_average(rates: dict[str, float]) -> float:
    return sum(rates.values()) / len(rates)


def contact_success(min_clearance: float, contact_threshold: float) -> bool:
    return min_clearance <= contact_threshold


def place_success(
    final_xy: tuple[float, float],
    target_xy: tuple[float, float],
    radius: float,
    gripper_open: bool,
) -> bool:
    dx = final_xy[0] - target_xy[0]
    dy = final_xy[1] - target_xy[1]
    return gripper_open and dx * dx + dy * dy <= radius * radius


def hold_success(in_place_steps: int, required_steps: int) -> bool:
    return in_place_steps >= required_steps


def run() -> dict[str, Any]:
    n25 = [1] * 20 + [0] * 5
    n5 = [1, 1, 1, 1, 0]
    n10_all = [1] * 10
    n3 = [1, 1, 0]

    p25, n25_wilson_lo, n25_wilson_hi = wilson_interval(n25)
    _, n25_normal_lo, n25_normal_hi = normal_interval(n25)
    p5, n5_wilson_lo, n5_wilson_hi = wilson_interval(n5)
    _, n5_normal_lo, n5_normal_hi = normal_interval(n5)
    p10, n10_wilson_lo, n10_wilson_hi = wilson_interval(n10_all)
    _, n10_normal_lo, n10_normal_hi = normal_interval(n10_all)
    _, n3_wilson_lo, n3_wilson_hi = wilson_interval(n3)

    n25_wilson_width = n25_wilson_hi - n25_wilson_lo
    n25_normal_width = n25_normal_hi - n25_normal_lo
    n5_covers_half = covers(n5_wilson_lo, n5_wilson_hi, 0.5)
    n25_covers_half = covers(n25_wilson_lo, n25_wilson_hi, 0.5)
    n3_covers_half = covers(n3_wilson_lo, n3_wilson_hi, 0.5)

    macro = suite_macro_average(SUITE_OPENVLA_FT)
    long_gap = SUITE_OPENVLA_FT["spatial"] - SUITE_OPENVLA_FT["long"]

    touched_not_placed = {
        "min_clearance": 0.004,
        "final_xy": (0.18, 0.31),
        "target_xy": (0.12, 0.12),
        "gripper_open": False,
        "in_place_steps": 0,
    }
    contact = contact_success(touched_not_placed["min_clearance"], 0.01)
    placed = place_success(
        touched_not_placed["final_xy"],
        touched_not_placed["target_xy"],
        0.03,
        touched_not_placed["gripper_open"],
    )
    held = hold_success(touched_not_placed["in_place_steps"], 20)

    checks = {
        "n25_point_estimate_is_0_8": math.isclose(p25, 0.8, abs_tol=1e-12),
        "n25_wilson_matches_hand_interval": (
            math.isclose(n25_wilson_lo, 0.608687, abs_tol=5e-7)
            and math.isclose(n25_wilson_hi, 0.911395, abs_tol=5e-7)
        ),
        "n25_normal_matches_hand_interval": (
            math.isclose(n25_normal_lo, 0.6432, abs_tol=5e-7)
            and math.isclose(n25_normal_hi, 0.9568, abs_tol=5e-7)
        ),
        "n5_wilson_covers_one_half": n5_covers_half,
        "n25_wilson_does_not_cover_one_half": not n25_covers_half,
        "n3_wilson_covers_one_half": n3_covers_half,
        "perfect_ten_normal_width_is_zero": math.isclose(
            n10_normal_hi - n10_normal_lo,
            0.0,
            abs_tol=1e-12,
        ),
        "perfect_ten_wilson_width_is_positive": n10_wilson_hi - n10_wilson_lo
        > 0.2,
        "four_suite_average_hides_long_horizon_drop": (
            math.isclose(macro, 0.765, abs_tol=5e-4)
            and long_gap > 0.3
        ),
        "contact_place_hold_disagree_on_touch_only_trial": (
            contact and (not placed) and (not held)
        ),
    }
    return {
        "summary": (
            "对固定成功/失败序列计算点估计、正态近似区间和 Wilson 区间；"
            "N=5 时区间覆盖 0.5，N=25、成功率 0.8 时 Wilson 区间约为 "
            "[0.609, 0.911] 且不覆盖 0.5；10/10 的正态区间宽度为 0。"
            "四套件宏平均会掩盖 Long 的掉点。接触成功与放置/保持成功在同一条"
            "轨迹上可以不一致。本实验不评测真实政策，也不产生 SOTA 数字。"
        ),
        "metrics": {
            "z": Z_95,
            "n25": 25,
            "k25": 20,
            "p25": p25,
            "n25_wilson_low": round(n25_wilson_lo, 6),
            "n25_wilson_high": round(n25_wilson_hi, 6),
            "n25_wilson_width": round(n25_wilson_width, 6),
            "n25_normal_low": round(n25_normal_lo, 6),
            "n25_normal_high": round(n25_normal_hi, 6),
            "n25_normal_width": round(n25_normal_width, 6),
            "n5": 5,
            "p5": p5,
            "n5_wilson_low": round(n5_wilson_lo, 6),
            "n5_wilson_high": round(n5_wilson_hi, 6),
            "n10_all_success_wilson_low": round(n10_wilson_lo, 6),
            "n10_all_success_normal_width": 0.0,
            "suite_spatial": SUITE_OPENVLA_FT["spatial"],
            "suite_object": SUITE_OPENVLA_FT["object"],
            "suite_goal": SUITE_OPENVLA_FT["goal"],
            "suite_long": SUITE_OPENVLA_FT["long"],
            "suite_macro": round(macro, 4),
            "spatial_minus_long": round(long_gap, 4),
            "touch_only_contact": contact,
            "touch_only_place": placed,
            "touch_only_hold": held,
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="31",
    title="设计可拆层的 VLA 评测",
    question="同一政策在不同协议下的成功率，怎样拆桶、给区间，并且拒绝横着比？",
    run=run,
)
