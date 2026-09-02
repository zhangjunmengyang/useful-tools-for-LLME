from __future__ import annotations

import math
from typing import Any

from ..core import LessonExperiment


Z_95 = 1.96

# Identical to lesson 47. A living-protocol card may not invent a seventh class.
CLASS_IDS = (
    "C1_expert_static",
    "C2_video_temporal",
    "C3_tri_modal",
    "C4_computer_exec",
    "C5_sim_manip",
    "C6_sim2real_rank",
)

# Identical to lesson 31 suite keys.
SUITES_31 = ("spatial", "object", "goal", "long")

SUITE_OPENVLA_FT = {
    "spatial": 0.847,
    "object": 0.884,
    "goal": 0.792,
    "long": 0.537,
}

CLAIM_KINDS = ("scale", "mechanism")
ADMISSIONS = ("admit", "reject_incomplete", "reject_illegal")

# Shared required fields. VLA simulation rows additionally need suite.
REQUIRED_FIELDS = (
    "paper_id",
    "lesson_bucket",
    "claim_kind",
    "class_id",
    "benchmark",
    "split",
    "n",
    "unit",
    "fine_tune",
    "reducible",
)
C5_EXTRA_FIELDS = ("suite", "success_predicate")

# Lesson 47 records, reused so living-protocol labels stay compatible.
RECORDS_47: dict[str, dict[str, Any]] = {
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

# Living-protocol cards. Fictional NovaVLA rows are fixtures, not citations.
CARDS: dict[str, dict[str, Any]] = {
    "openvla_libero_spatial": {
        "paper_id": "2406.09246",
        "lesson_bucket": "31",
        "claim_kind": "mechanism",
        "class_id": "C5_sim_manip",
        "benchmark": "LIBERO",
        "split": "spatial_ft",
        "n": 500,
        "unit": "success_rate",
        "fine_tune": True,
        "reducible": True,
        "suite": "spatial",
        "success_predicate": "pddl_conjunction",
        "claimed_class": "C5_sim_manip",
        "value": 0.847,
    },
    "oft_libero_macro": {
        "paper_id": "2502.19645",
        "lesson_bucket": "31",
        "claim_kind": "mechanism",
        "class_id": "C5_sim_manip",
        "benchmark": "LIBERO",
        "split": "four_suite_macro_ft",
        "n": 2000,
        "unit": "success_rate",
        "fine_tune": True,
        "reducible": True,
        "suite": "long",
        "success_predicate": "pddl_conjunction",
        "claimed_class": "C5_sim_manip",
        "value": 0.971,
        "note": "macro 97.1%; Long 94.5%; parallel decode + chunk + L1",
    },
    "openvla_scale_vs_rt2x": {
        "paper_id": "2406.09246",
        "lesson_bucket": "31",
        "claim_kind": "scale",
        "class_id": "real_robot",
        "benchmark": "real_widowx_google",
        "split": "29_tasks_out_of_box",
        "n": 230,
        "unit": "success_rate",
        "fine_tune": False,
        "reducible": False,
        "claimed_class": "real_robot",
        "value": 0.165,
        "note": "7B vs 55B absolute gap on 29 tasks; not a MiniMind-O number",
    },
    "simpler_vismatch": {
        "paper_id": "2405.05941",
        "lesson_bucket": "31",
        "claim_kind": "mechanism",
        "class_id": "C6_sim2real_rank",
        "benchmark": "SIMPLER",
        "split": "google_robot_vismatch",
        "n": 6,
        "unit": "pearson_r",
        "fine_tune": False,
        "reducible": True,
        "claimed_class": "C6_sim2real_rank",
        "value": 0.924,
    },
    "nova_libero_as_real": {
        "paper_id": "nova-vla-fiction",
        "lesson_bucket": "31",
        "claim_kind": "mechanism",
        "class_id": "C5_sim_manip",
        "benchmark": "LIBERO",
        "split": "four_suite_macro",
        "n": 500,
        "unit": "success_rate",
        "fine_tune": True,
        "reducible": False,
        "suite": "spatial",
        "success_predicate": "pddl_conjunction",
        "claimed_class": "real_robot",
        "value": 0.81,
    },
    "nova_missing_n": {
        "paper_id": "nova-vla-fiction",
        "lesson_bucket": "31",
        "claim_kind": "mechanism",
        "class_id": "C5_sim_manip",
        "benchmark": "LIBERO",
        "split": "long",
        "n": None,
        "unit": "success_rate",
        "fine_tune": True,
        "reducible": False,
        "suite": "long",
        "success_predicate": "pddl_conjunction",
        "claimed_class": "C5_sim_manip",
        "value": 0.62,
    },
    "nova_missing_suite": {
        "paper_id": "nova-vla-fiction",
        "lesson_bucket": "31",
        "claim_kind": "mechanism",
        "class_id": "C5_sim_manip",
        "benchmark": "LIBERO",
        "split": "unspecified",
        "n": 500,
        "unit": "success_rate",
        "fine_tune": True,
        "reducible": False,
        "suite": None,
        "success_predicate": "pddl_conjunction",
        "claimed_class": "C5_sim_manip",
        "value": 0.90,
    },
    "nova_scale_13b": {
        "paper_id": "nova-vla-fiction",
        "lesson_bucket": "27",
        "claim_kind": "scale",
        "class_id": "C5_sim_manip",
        "benchmark": "LIBERO",
        "split": "spatial_ft",
        "n": 500,
        "unit": "success_rate",
        "fine_tune": True,
        "reducible": False,
        "suite": "spatial",
        "success_predicate": "pddl_conjunction",
        "claimed_class": "C5_sim_manip",
        "value": 0.91,
        "opens_new_model_lesson": False,
    },
}


def required_keys(card: dict[str, Any]) -> tuple[str, ...]:
    keys = list(REQUIRED_FIELDS)
    needs_suite = (
        card.get("class_id") == "C5_sim_manip"
        or card.get("benchmark") == "LIBERO"
    )
    if needs_suite:
        keys.extend(C5_EXTRA_FIELDS)
    return tuple(keys)


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def missing_fields(card: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for key in required_keys(card):
        value = card.get(key)
        if key == "n" and (value is None or value == 0):
            missing.append(key)
            continue
        if is_blank(value):
            missing.append(key)
    return missing


def card_is_complete(card: dict[str, Any]) -> bool:
    return len(missing_fields(card)) == 0


def illegal_libero_as_real(card: dict[str, Any]) -> bool:
    return (
        card.get("benchmark") == "LIBERO"
        and card.get("claimed_class") == "real_robot"
    )


def illegal_simpler_as_success(card: dict[str, Any]) -> bool:
    return card.get("unit") == "pearson_r" and card.get("claimed_class") in {
        "C5_sim_manip",
        "real_robot",
        "success_rate",
    }


def labels_compatible_31_47(card: dict[str, Any]) -> bool:
    class_id = card.get("class_id")
    if card.get("claim_kind") not in CLAIM_KINDS:
        return False
    if illegal_libero_as_real(card):
        return False
    if illegal_simpler_as_success(card):
        return False
    if class_id == "real_robot":
        return (
            card.get("benchmark") != "LIBERO"
            and card.get("unit") == "success_rate"
        )
    if class_id not in CLASS_IDS:
        return False
    if class_id == "C5_sim_manip":
        suite = card.get("suite")
        if suite not in SUITES_31:
            return False
        if card.get("unit") != "success_rate":
            return False
    if class_id == "C6_sim2real_rank" and card.get("unit") != "pearson_r":
        return False
    return True


def opens_new_model_lesson(card: dict[str, Any]) -> bool:
    if card.get("opens_new_model_lesson") is True:
        return True
    if card.get("claim_kind") == "scale" and card.get("lesson_bucket") not in {
        "01",
        "27",
        "28",
        "31",
        "47",
    }:
        return True
    return False


def admission(card: dict[str, Any]) -> str:
    if illegal_libero_as_real(card) or illegal_simpler_as_success(card):
        return "reject_illegal"
    if not card_is_complete(card):
        return "reject_incomplete"
    if not labels_compatible_31_47(card):
        return "reject_illegal"
    if opens_new_model_lesson(card):
        return "reject_illegal"
    return "admit"


def may_compare_47(record_a: str, record_b: str) -> bool:
    left = RECORDS_47[record_a]
    right = RECORDS_47[record_b]
    return left["class_id"] == right["class_id"] and left["unit"] == right["unit"]


def assignments_are_mutex(mapping: dict[str, str]) -> bool:
    if set(mapping) != set(RECORDS_47):
        return False
    labels = list(mapping.values())
    return len(labels) == 6 and set(labels) == set(CLASS_IDS)


def suite_macro_average(rates: dict[str, float]) -> float:
    return sum(rates.values()) / len(rates)


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


def reducible_implies_mechanism(card: dict[str, Any]) -> bool:
    if not card.get("reducible"):
        return True
    return card.get("claim_kind") == "mechanism"


def run() -> dict[str, Any]:
    spatial = CARDS["openvla_libero_spatial"]
    oft = CARDS["oft_libero_macro"]
    scale_gap = CARDS["openvla_scale_vs_rt2x"]
    simpler = CARDS["simpler_vismatch"]
    as_real = CARDS["nova_libero_as_real"]
    missing_n = CARDS["nova_missing_n"]
    missing_suite = CARDS["nova_missing_suite"]
    scale_13b = CARDS["nova_scale_13b"]

    gold_47 = {record_id: rec["class_id"] for record_id, rec in RECORDS_47.items()}
    collided_47 = dict(gold_47)
    collided_47["libero_openvla_macro"] = "C4_computer_exec"

    n25 = [1] * 20 + [0] * 5
    p25, wilson_lo, wilson_hi = wilson_interval(n25)
    macro = suite_macro_average(SUITE_OPENVLA_FT)
    long_gap = SUITE_OPENVLA_FT["spatial"] - SUITE_OPENVLA_FT["long"]

    checks = {
        "required_field_count_is_ten": len(REQUIRED_FIELDS) == 10,
        "c5_requires_suite_and_predicate": C5_EXTRA_FIELDS == ("suite", "success_predicate"),
        "missing_n_rejected": (
            "n" in missing_fields(missing_n)
            and admission(missing_n) == "reject_incomplete"
        ),
        "missing_suite_rejected": (
            "suite" in missing_fields(missing_suite)
            and admission(missing_suite) == "reject_incomplete"
        ),
        "libero_as_real_rejected": (
            illegal_libero_as_real(as_real)
            and admission(as_real) == "reject_illegal"
            and not labels_compatible_31_47(as_real)
        ),
        "complete_spatial_admitted": (
            card_is_complete(spatial)
            and admission(spatial) == "admit"
            and labels_compatible_31_47(spatial)
            and spatial["suite"] == "spatial"
        ),
        "oft_mechanism_reducible": (
            oft["claim_kind"] == "mechanism"
            and oft["reducible"] is True
            and admission(oft) == "admit"
            and reducible_implies_mechanism(oft)
        ),
        "scale_claim_not_reducible_to_sota": (
            scale_gap["claim_kind"] == "scale"
            and scale_gap["reducible"] is False
            and scale_gap["class_id"] == "real_robot"
            and admission(scale_gap) == "admit"
            and scale_13b["claim_kind"] == "scale"
            and scale_13b["reducible"] is False
            and not opens_new_model_lesson(scale_13b)
            and admission(scale_13b) == "admit"
        ),
        "simpler_unit_is_pearson_r": (
            simpler["unit"] == "pearson_r"
            and simpler["class_id"] == "C6_sim2real_rank"
            and labels_compatible_31_47(simpler)
            and admission(simpler) == "admit"
            and illegal_simpler_as_success(
                {**simpler, "claimed_class": "success_rate"},
            )
        ),
        "lesson47_labels_still_mutex": assignments_are_mutex(gold_47)
        and not assignments_are_mutex(collided_47),
        "lesson31_suite_keys_accepted": set(SUITE_OPENVLA_FT) == set(SUITES_31),
        "libero_macro_matches_four_suites": math.isclose(macro, 0.765, abs_tol=5e-4)
        and long_gap > 0.3,
        "n25_wilson_matches_lesson31": (
            math.isclose(p25, 0.8, abs_tol=1e-12)
            and math.isclose(wilson_lo, 0.608687, abs_tol=5e-7)
            and math.isclose(wilson_hi, 0.911395, abs_tol=5e-7)
        ),
        "mmmu_not_comparable_to_libero": not may_compare_47(
            "mmmu_gpt4v_test",
            "libero_openvla_macro",
        ),
        "no_new_model_lesson_opened": not any(
            opens_new_model_lesson(card) for card in CARDS.values()
        ),
        "admissions_only_use_known_labels": set(ADMISSIONS)
        == {"admit", "reject_incomplete", "reject_illegal"},
    }

    return {
        "summary": (
            "收编卡必填十个字段；C5 / LIBERO 行额外要套件与成功谓词。"
            "缺 N、缺套件拒收；把 LIBERO 写成真机拒收。"
            "类标签与第 47 课六类互斥，套件键与第 31 课 Spatial/Object/Goal/Long 一致。"
            "Wilson 区间沿用第 31 课 N=25、k=20。"
            "规模声明不得新开模型课；机制声明才允许标可复现方向。"
            "NovaVLA 行是教学夹具，不是文献。"
        ),
        "metrics": {
            "n_required_fields": len(REQUIRED_FIELDS),
            "n_c5_extra_fields": len(C5_EXTRA_FIELDS),
            "n_class_ids": len(CLASS_IDS),
            "n_suites": len(SUITES_31),
            "n_cards": len(CARDS),
            "missing_n_fields": missing_fields(missing_n),
            "missing_suite_fields": missing_fields(missing_suite),
            "spatial_admission": admission(spatial),
            "as_real_admission": admission(as_real),
            "missing_n_admission": admission(missing_n),
            "missing_suite_admission": admission(missing_suite),
            "oft_value": oft["value"],
            "openvla_spatial": spatial["value"],
            "libero_macro": round(macro, 4),
            "libero_long": SUITE_OPENVLA_FT["long"],
            "simpler_r": simpler["value"],
            "n25": 25,
            "k25": 20,
            "n25_wilson_low": round(wilson_lo, 6),
            "n25_wilson_high": round(wilson_hi, 6),
            "opens_new_model_lesson_count": sum(
                1 for card in CARDS.values() if opens_new_model_lesson(card)
            ),
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="60",
    title="把新论文接到可执行的验收口径",
    question="新论文怎样填收编卡、接到已有课桶，并且在缺 N、缺套件、把 LIBERO 写成真机时被拒收？",
    run=run,
)
