from __future__ import annotations

import math
from typing import Any

from ..core import LessonExperiment


Z_95 = 1.96

CLASS_IDS = (
    "C1_expert_static",
    "C2_video_temporal",
    "C3_tri_modal",
    "C4_computer_exec",
    "C5_sim_manip",
    "C6_sim2real_rank",
)

# Public numbers. Each record has exactly one class.
# Values are copied from opened papers or from lesson 31's already-cited tables.
RECORDS: dict[str, dict[str, Any]] = {
    "mmmu_gpt4v_test": {
        "class_id": "C1_expert_static",
        "benchmark": "MMMU",
        "split": "test",
        "value": 0.557,
        "unit": "accuracy",
        "n": 10500,
        "fine_tune": False,
    },
    "videomme_gemini_frames": {
        "class_id": "C2_video_temporal",
        "benchmark": "Video-MME",
        "split": "overall_no_subtitle",
        "value": 0.75,
        "unit": "accuracy",
        "n": 2700,
        "fine_tune": False,
    },
    "omnibench_qwen25_omni": {
        "class_id": "C3_tri_modal",
        "benchmark": "OmniBench",
        "split": "overall",
        "value": 0.5613,
        "unit": "accuracy",
        "n": 1142,
        "fine_tune": False,
    },
    "osworld_gpt4_a11y": {
        "class_id": "C4_computer_exec",
        "benchmark": "OSWorld",
        "split": "ubuntu_overall",
        "value": 0.1224,
        "unit": "success_rate",
        "n": 369,
        "fine_tune": False,
    },
    "libero_openvla_macro": {
        "class_id": "C5_sim_manip",
        "benchmark": "LIBERO",
        "split": "four_suite_macro_ft",
        "value": 0.765,
        "unit": "success_rate",
        "n": 6000,
        "fine_tune": True,
    },
    "simpler_vismatch_r": {
        "class_id": "C6_sim2real_rank",
        "benchmark": "SIMPLER",
        "split": "google_robot_vismatch",
        "value": 0.924,
        "unit": "pearson_r",
        "n": 6,
        "fine_tune": False,
    },
}

SUITE_OPENVLA_FT = {
    "spatial": 0.847,
    "object": 0.884,
    "goal": 0.792,
    "long": 0.537,
}

ILLEGAL_CLAIMS = (
    ("libero_openvla_macro", "real_robot"),
    ("simpler_vismatch_r", "success_rate"),
    ("mmmu_gpt4v_test", "C2_video_temporal"),
    ("videomme_gemini_frames", "C3_tri_modal"),
    ("osworld_gpt4_a11y", "C5_sim_manip"),
    ("mmmu_gpt4v_test", "grounding_hit"),
)


def true_class(record_id: str) -> str:
    return str(RECORDS[record_id]["class_id"])


def assignments_are_mutex(mapping: dict[str, str]) -> bool:
    """Each record gets exactly one class, and the six classes form a partition."""
    if set(mapping) != set(RECORDS):
        return False
    labels = list(mapping.values())
    return len(labels) == 6 and set(labels) == set(CLASS_IDS)


def may_compare(record_a: str, record_b: str) -> bool:
    if record_a == record_b:
        return False
    left = RECORDS[record_a]
    right = RECORDS[record_b]
    return (
        left["class_id"] == right["class_id"]
        and left["unit"] == right["unit"]
    )


def claim_is_legal(record_id: str, claimed: str) -> bool:
    record = RECORDS[record_id]
    if claimed == "real_robot":
        return False
    if claimed == "grounding_hit":
        return False
    if claimed == "success_rate" and record["unit"] != "success_rate":
        return False
    if claimed in CLASS_IDS:
        return claimed == record["class_id"]
    return claimed == record["unit"]


def illegal_macro_average(values: list[float]) -> float:
    return sum(values) / len(values)


def wilson_interval(outcomes: list[int], z: float = Z_95) -> tuple[float, float, float]:
    n = len(outcomes)
    p_hat = sum(outcomes) / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    margin = (
        z
        * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n))
        / denom
    )
    return p_hat, max(0.0, center - margin), min(1.0, center + margin)


def suite_macro_average(rates: dict[str, float]) -> float:
    return sum(rates.values()) / len(rates)


def pope_hallucination_rate(exists: list[int], predicted_yes: list[int]) -> float:
    negatives = 0
    false_yes = 0
    for present, yes in zip(exists, predicted_yes):
        if present == 0:
            negatives += 1
            if yes == 1:
                false_yes += 1
    if negatives == 0:
        return 0.0
    return false_yes / negatives


def run() -> dict[str, Any]:
    gold_map = {record_id: true_class(record_id) for record_id in RECORDS}
    collided = dict(gold_map)
    collided["libero_openvla_macro"] = "C4_computer_exec"
    collided["osworld_gpt4_a11y"] = "C4_computer_exec"

    n25 = [1] * 20 + [0] * 5
    p25, wilson_lo, wilson_hi = wilson_interval(n25)
    macro = suite_macro_average(SUITE_OPENVLA_FT)
    long_gap = SUITE_OPENVLA_FT["spatial"] - SUITE_OPENVLA_FT["long"]

    exists = [1, 1, 0, 0, 0, 0]
    predicted_yes = [1, 1, 0, 1, 1, 1]
    hallu = pope_hallucination_rate(exists, predicted_yes)
    yes_ratio = sum(predicted_yes) / len(predicted_yes)
    mmmu_acc = float(RECORDS["mmmu_gpt4v_test"]["value"])

    illegal_values = [
        float(RECORDS["mmmu_gpt4v_test"]["value"]),
        float(RECORDS["videomme_gemini_frames"]["value"]),
        float(RECORDS["omnibench_qwen25_omni"]["value"]),
        float(RECORDS["osworld_gpt4_a11y"]["value"]),
        float(RECORDS["libero_openvla_macro"]["value"]),
        float(RECORDS["simpler_vismatch_r"]["value"]),
    ]
    bogus_mean = illegal_macro_average(illegal_values)

    legal_flags = [claim_is_legal(record_id, claimed) for record_id, claimed in ILLEGAL_CLAIMS]
    class_list = [true_class(record_id) for record_id in sorted(RECORDS)]

    checks = {
        "six_class_labels_are_mutex": assignments_are_mutex(gold_map),
        "colliding_labels_fail_mutex": not assignments_are_mutex(collided),
        "each_record_has_exactly_one_true_class": (
            len(class_list) == 6 and len(set(class_list)) == 6
        ),
        "libero_macro_rejected_as_real_robot": not claim_is_legal(
            "libero_openvla_macro",
            "real_robot",
        ),
        "simpler_r_rejected_as_success_rate": not claim_is_legal(
            "simpler_vismatch_r",
            "success_rate",
        ),
        "mmmu_not_comparable_to_videomme": not may_compare(
            "mmmu_gpt4v_test",
            "videomme_gemini_frames",
        ),
        "osworld_not_comparable_to_libero": not may_compare(
            "osworld_gpt4_a11y",
            "libero_openvla_macro",
        ),
        "videomme_not_comparable_to_omnibench": not may_compare(
            "videomme_gemini_frames",
            "omnibench_qwen25_omni",
        ),
        "all_listed_illegal_claims_rejected": not any(legal_flags),
        "libero_macro_matches_four_suites": math.isclose(macro, 0.765, abs_tol=5e-4)
        and long_gap > 0.3,
        "n25_wilson_matches_lesson31": (
            math.isclose(p25, 0.8, abs_tol=1e-12)
            and math.isclose(wilson_lo, 0.608687, abs_tol=5e-7)
            and math.isclose(wilson_hi, 0.911395, abs_tol=5e-7)
        ),
        "pope_hallucination_is_not_mmmu_accuracy": (
            math.isclose(hallu, 0.75, abs_tol=1e-12)
            and abs(hallu - mmmu_acc) > 0.1
            and abs(yes_ratio - hallu) > 1e-9
        ),
        "cross_class_macro_is_not_a_capability": bogus_mean > 0.5
        and abs(bogus_mean - macro) > 0.05,
    }

    return {
        "summary": (
            "六条公开评测数字打上互斥标签：C1 MMMU、C2 Video-MME、C3 OmniBench、"
            "C4 OSWorld、C5 LIBERO、C6 SIMPLER。LIBERO 宏平均不得标成真机能力，"
            "SIMPLER 的 r 不得当成功率，跨类做差与总平均被拒绝。"
            "Wilson 区间沿用第 31 课 N=25、k=20 的手算。"
            "本实验不评测真实模型。"
        ),
        "metrics": {
            "n_classes": 6,
            "n_records": 6,
            "mmmu_gpt4v_test": RECORDS["mmmu_gpt4v_test"]["value"],
            "videomme_gemini_frames": RECORDS["videomme_gemini_frames"]["value"],
            "omnibench_qwen25_omni": RECORDS["omnibench_qwen25_omni"]["value"],
            "osworld_gpt4_a11y": RECORDS["osworld_gpt4_a11y"]["value"],
            "libero_openvla_macro": RECORDS["libero_openvla_macro"]["value"],
            "simpler_vismatch_r": RECORDS["simpler_vismatch_r"]["value"],
            "libero_spatial": SUITE_OPENVLA_FT["spatial"],
            "libero_long": SUITE_OPENVLA_FT["long"],
            "libero_spatial_minus_long": round(long_gap, 4),
            "illegal_six_number_mean": round(bogus_mean, 6),
            "n25": 25,
            "k25": 20,
            "n25_wilson_low": round(wilson_lo, 6),
            "n25_wilson_high": round(wilson_hi, 6),
            "pope_hallucination_rate": hallu,
            "pope_yes_ratio": yes_ratio,
            "illegal_claim_count": len(ILLEGAL_CLAIMS),
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="47",
    title="把六类评测数字分桶记账",
    question="六类评测数字为什么不能横着比，LIBERO 平均为什么进不了真机能力？",
    run=run,
)
