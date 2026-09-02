from __future__ import annotations

from typing import Any

from ..core import LessonExperiment


COUNTS: dict[str, int] = {
    "google_kitchen": 10_000,
    "franka_table": 1_000,
    "widowx_sink": 200,
    "bimanual_rare": 100,
}
DOMAIN_ORDER = tuple(COUNTS.keys())
COUNT_CAP = 400
BATCH_SIZE = 256
LARGEST = "google_kitchen"
SMALLEST = "bimanual_rare"

# Two instructions per embodiment. Leak labels ignore the instruction.
INSTRUCTION_ACTIONS = {
    "pick_can": 0,
    "pick_bottle": 1,
    "pick_block": 2,
    "pick_tool": 3,
    "pick_cup": 4,
    "pick_plate": 5,
    "insert_peg": 6,
    "tighten_screw": 7,
}
EMBODIMENT_DEFAULT = {
    0: 0,  # google kitchen defaults to pick_can
    1: 2,
    2: 4,
    3: 6,
}
def _row(instruction: str, embodiment: int) -> dict[str, int | str]:
    return {
        "instruction": instruction,
        "embodiment": embodiment,
        "label_lang": INSTRUCTION_ACTIONS[instruction],
        "label_leak": EMBODIMENT_DEFAULT[embodiment],
    }


FIXTURE = (
    _row("pick_can", 0),
    _row("pick_bottle", 0),
    _row("pick_block", 1),
    _row("pick_tool", 1),
    _row("pick_cup", 2),
    _row("pick_plate", 2),
    _row("insert_peg", 3),
    _row("tighten_screw", 3),
)


def mixture_probs(counts: dict[str, int], alpha: float) -> dict[str, float]:
    weights = {name: count**alpha for name, count in counts.items()}
    total = sum(weights.values())
    return {name: weight / total for name, weight in weights.items()}


def cap_counts(counts: dict[str, int], cap: int) -> dict[str, int]:
    return {name: min(count, cap) for name, count in counts.items()}


def effective_domains(probs: dict[str, float]) -> float:
    return 1.0 / sum(probability * probability for probability in probs.values())


def expected_batch(probs: dict[str, float], batch_size: int = BATCH_SIZE) -> dict[str, float]:
    return {name: batch_size * probability for name, probability in probs.items()}


def oversampling_ratio(
    counts: dict[str, int],
    probs: dict[str, float],
    name: str,
) -> float:
    natural = counts[name] / sum(counts.values())
    return probs[name] / natural


def language_predict(instruction: str) -> int:
    return INSTRUCTION_ACTIONS[instruction]


def leak_predict(embodiment: int) -> int:
    return EMBODIMENT_DEFAULT[embodiment]


def shuffle_instructions(
    rows: tuple[dict[str, int | str], ...],
) -> list[dict[str, int | str]]:
    """Keep image, embodiment and labels; rotate only the instruction string."""
    shuffled: list[dict[str, int | str]] = []
    length = len(rows)
    for index, row in enumerate(rows):
        source = rows[(index - 1) % length]
        shuffled.append({**row, "instruction": source["instruction"]})
    return shuffled


def shuffle_embodiment_ids(
    rows: tuple[dict[str, int | str], ...],
) -> list[dict[str, int | str]]:
    """Keep demonstrated labels; shift the ID the policy sees."""
    return [
        {**row, "embodiment": (int(row["embodiment"]) + 1) % 4}
        for row in rows
    ]


def accuracy(
    rows: list[dict[str, int | str]] | tuple[dict[str, int | str], ...],
    leak: bool,
    predictor: str,
) -> float:
    hits = 0
    for row in rows:
        label = int(row["label_leak"] if leak else row["label_lang"])
        if predictor == "language":
            guess = language_predict(str(row["instruction"]))
        elif predictor == "leak":
            guess = leak_predict(int(row["embodiment"]))
        else:
            raise ValueError(f"unknown predictor {predictor}")
        hits += int(guess == label)
    return hits / len(rows)


def run() -> dict[str, Any]:
    alpha1 = mixture_probs(COUNTS, 1.0)
    alpha0 = mixture_probs(COUNTS, 0.0)
    alpha_half = mixture_probs(COUNTS, 0.5)
    capped = cap_counts(COUNTS, COUNT_CAP)
    alpha1_capped = mixture_probs(capped, 1.0)
    natural_small = COUNTS[SMALLEST] / sum(COUNTS.values())

    shuffled = shuffle_instructions(FIXTURE)
    intact_language = accuracy(FIXTURE, leak=False, predictor="language")
    shuffled_language = accuracy(shuffled, leak=False, predictor="language")
    intact_language_on_leak = accuracy(FIXTURE, leak=True, predictor="language")
    shuffled_language_on_leak = accuracy(shuffled, leak=True, predictor="language")
    intact_leak_policy = accuracy(FIXTURE, leak=True, predictor="leak")
    shuffled_leak_policy = accuracy(shuffled, leak=True, predictor="leak")
    embodiment_shuffled = shuffle_embodiment_ids(FIXTURE)
    leak_policy_after_id_shuffle = accuracy(
        embodiment_shuffled,
        leak=True,
        predictor="leak",
    )
    language_after_id_shuffle = accuracy(
        embodiment_shuffled,
        leak=False,
        predictor="language",
    )

    deff_alpha1 = effective_domains(alpha1)
    deff_alpha0 = effective_domains(alpha0)
    deff_capped = effective_domains(alpha1_capped)
    small_ratio_alpha_half = oversampling_ratio(COUNTS, alpha_half, SMALLEST)

    checks = {
        "alpha1_matches_count_share": abs(
            alpha1[LARGEST] - COUNTS[LARGEST] / sum(COUNTS.values()),
        )
        < 1e-12,
        "alpha0_is_uniform": all(abs(alpha0[name] - 0.25) < 1e-12 for name in COUNTS),
        "alpha1_largest_exceeds_three_quarters": alpha1[LARGEST] > 0.75,
        "count_cap_raises_smallest_domain": (
            alpha1_capped[SMALLEST] > alpha1[SMALLEST] + 0.05
        ),
        "count_cap_cuts_largest_share": alpha1_capped[LARGEST] < alpha1[LARGEST] - 0.3,
        "intact_language_policy_is_perfect": intact_language == 1.0,
        "instruction_shuffle_drops_language_policy": (
            shuffled_language < intact_language - 0.4
        ),
        "leak_policy_is_perfect_on_leak_labels": intact_leak_policy == 1.0,
        "leak_policy_survives_instruction_shuffle": shuffled_leak_policy == 1.0,
        "leak_policy_breaks_when_embodiment_ids_shuffle": (
            leak_policy_after_id_shuffle <= 0.25
        ),
        "language_policy_ignores_embodiment_id_shuffle": (
            language_after_id_shuffle == intact_language
        ),
        "effective_domain_count_rises_after_cap": deff_capped > deff_alpha1 + 0.5,
        "uniform_mixture_has_four_effective_domains": abs(deff_alpha0 - 4.0) < 1e-12,
        "small_domain_is_oversampled_at_alpha_half": small_ratio_alpha_half > 1.0,
    }

    return {
        "summary": (
            "在四域计数 (10000, 1000, 200, 100) 上核对 p_d ∝ n_d^α："
            "α=1 时最大域占比超过 75%；每域条数上限 400 后最小域回升；"
            "指令打乱会打掉语言政策，机体 ID 泄漏政策在打乱指令后仍满分，"
            "但打乱机体标签后下降。"
        ),
        "metrics": {
            "counts": COUNTS,
            "count_cap": COUNT_CAP,
            "batch_size": BATCH_SIZE,
            "alpha1_probs": {name: round(alpha1[name], 6) for name in DOMAIN_ORDER},
            "alpha0_probs": {name: round(alpha0[name], 6) for name in DOMAIN_ORDER},
            "alpha_half_probs": {
                name: round(alpha_half[name], 6) for name in DOMAIN_ORDER
            },
            "alpha1_capped_probs": {
                name: round(alpha1_capped[name], 6) for name in DOMAIN_ORDER
            },
            "alpha1_batch": {
                name: round(expected_batch(alpha1)[name], 4) for name in DOMAIN_ORDER
            },
            "alpha1_capped_batch": {
                name: round(expected_batch(alpha1_capped)[name], 4)
                for name in DOMAIN_ORDER
            },
            "natural_small_share": round(natural_small, 6),
            "smallest_oversample_alpha_half": round(small_ratio_alpha_half, 6),
            "effective_domains_alpha1": round(deff_alpha1, 6),
            "effective_domains_alpha0": round(deff_alpha0, 6),
            "effective_domains_capped": round(deff_capped, 6),
            "fixture_size": len(FIXTURE),
            "intact_language_accuracy": intact_language,
            "shuffled_language_accuracy": shuffled_language,
            "intact_language_on_leak_labels": intact_language_on_leak,
            "shuffled_language_on_leak_labels": shuffled_language_on_leak,
            "intact_leak_policy_accuracy": intact_leak_policy,
            "shuffled_leak_policy_accuracy": shuffled_leak_policy,
            "id_shuffled_leak_policy_accuracy": leak_policy_after_id_shuffle,
            "id_shuffled_language_accuracy": language_after_id_shuffle,
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="26",
    title="混合异构机器人数据并控制机体捷径",
    question="按条数采样时大域如何淹没小域，指令打乱和机体 ID 泄漏分别改变什么准确率？",
    run=run,
)
