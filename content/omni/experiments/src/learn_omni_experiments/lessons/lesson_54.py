from __future__ import annotations

from typing import Any

from ..core import LessonExperiment


REAL_COUNTS: dict[str, int] = {
    "kitchen_pick": 8_000,
    "drawer": 800,
    "mug_insert": 200,
    "bimanual": 50,
}
DOMAIN_ORDER = tuple(REAL_COUNTS.keys())
LARGEST = "kitchen_pick"
SMALLEST = "bimanual"
SYNTH_COUNT = 2_000
BATCH_SIZE = 256

# 2-D pose fixture for the object-centric warp.
# Source object at the origin; controller targets stay 0.2 and 0.4 along +x.
# New object is translated by (1.0, 2.0). Relative pose must be preserved.
SOURCE_OBJECT = (0.0, 0.0)
NEW_OBJECT = (1.0, 2.0)
SOURCE_TARGETS = ((0.2, 0.0), (0.4, 0.0))
CORRECT_TARGETS = ((1.2, 2.0), (1.4, 2.0))
SWAPPED_TARGETS = ((-0.8, -2.0), (-0.6, -2.0))
EE_START = (0.0, 0.0)
INTERP_STEPS = 5

# Generation-rate fixture copied from the real-robot Stack numbers in MimicGen
# Sec. 6.4: 200 successes kept out of 243 attempts (82.3%).
GEN_ATTEMPTS = 243
GEN_SUCCESSES = 200


def mixture_probs(counts: dict[str, int], alpha: float) -> dict[str, float]:
    weights = {name: count**alpha for name, count in counts.items()}
    total = sum(weights.values())
    return {name: weight / total for name, weight in weights.items()}


def add_synth(counts: dict[str, int], domain: str, extra: int) -> dict[str, int]:
    updated = dict(counts)
    updated[domain] += extra
    return updated


def effective_domains(probs: dict[str, float]) -> float:
    return 1.0 / sum(probability * probability for probability in probs.values())


def herfindahl_sample_size(multiplicities: list[int]) -> float:
    """n_eff = (sum n_i)^2 / sum n_i^2 over unique hashes."""
    raw = sum(multiplicities)
    sum_sq = sum(count * count for count in multiplicities)
    return (raw * raw) / sum_sq


def unique_count(multiplicities: list[int]) -> int:
    return sum(1 for count in multiplicities if count > 0)


def real_multiplicities() -> list[int]:
    return [1] * sum(REAL_COUNTS.values())


def duplicate_multiplicities(copies: int) -> list[int]:
    """Keep every real hash, then add `copies` extra copies of one kitchen hash."""
    # One kitchen hash becomes 1 + copies; the remaining kitchen hashes stay 1.
    remaining_kitchen = REAL_COUNTS[LARGEST] - 1
    others = sum(REAL_COUNTS[name] for name in DOMAIN_ORDER if name != LARGEST)
    return [1 + copies] + [1] * remaining_kitchen + [1] * others


def unique_synth_multiplicities(extra: int) -> list[int]:
    return [1] * (sum(REAL_COUNTS.values()) + extra)


def translate(point: tuple[float, float], offset: tuple[float, float]) -> tuple[float, float]:
    return (point[0] + offset[0], point[1] + offset[1])


def invert_translate(offset: tuple[float, float]) -> tuple[float, float]:
    return (-offset[0], -offset[1])


def warp_targets(
    targets: tuple[tuple[float, float], ...],
    source_object: tuple[float, float],
    new_object: tuple[float, float],
    swapped: bool = False,
) -> tuple[tuple[float, float], ...]:
    """Apply T_C' = T_O' T_O^{-1} T_C, or the swapped last-line formula if asked."""
    warped: list[tuple[float, float]] = []
    for target in targets:
        relative = translate(target, invert_translate(source_object))
        if swapped:
            # T_C' = T_O T_O'^{-1} T_C  (MimicGen appendix M last displayed line)
            to_source = translate(target, invert_translate(new_object))
            warped.append(translate(to_source, source_object))
        else:
            warped.append(translate(relative, new_object))
    return tuple(warped)


def interpolate(
    start: tuple[float, float],
    end: tuple[float, float],
    steps: int,
) -> list[tuple[float, float]]:
    if steps < 2:
        raise ValueError("interpolation needs at least two poses")
    path: list[tuple[float, float]] = []
    for index in range(steps):
        ratio = index / (steps - 1)
        path.append(
            (
                start[0] + ratio * (end[0] - start[0]),
                start[1] + ratio * (end[1] - start[1]),
            ),
        )
    return path


def close(left: tuple[float, float], right: tuple[float, float], tol: float = 1e-12) -> bool:
    return abs(left[0] - right[0]) < tol and abs(left[1] - right[1]) < tol


def generation_rate(successes: int, attempts: int) -> float:
    return successes / attempts


def run() -> dict[str, Any]:
    alpha1 = mixture_probs(REAL_COUNTS, 1.0)
    alpha0 = mixture_probs(REAL_COUNTS, 0.0)
    largest = add_synth(REAL_COUNTS, LARGEST, SYNTH_COUNT)
    smallest = add_synth(REAL_COUNTS, SMALLEST, SYNTH_COUNT)
    alpha1_largest = mixture_probs(largest, 1.0)
    alpha1_smallest = mixture_probs(smallest, 1.0)

    deff_base = effective_domains(alpha1)
    deff_largest = effective_domains(alpha1_largest)
    deff_smallest = effective_domains(alpha1_smallest)
    deff_uniform = effective_domains(alpha0)

    real_mult = real_multiplicities()
    dup_mult = duplicate_multiplicities(SYNTH_COUNT)
    unique_mult = unique_synth_multiplicities(SYNTH_COUNT)
    n_unique_real = unique_count(real_mult)
    n_unique_dup = unique_count(dup_mult)
    n_unique_unique = unique_count(unique_mult)
    n_eff_real = herfindahl_sample_size(real_mult)
    n_eff_dup = herfindahl_sample_size(dup_mult)
    n_eff_unique = herfindahl_sample_size(unique_mult)
    n_raw_real = sum(REAL_COUNTS.values())
    n_raw_after = n_raw_real + SYNTH_COUNT

    correct = warp_targets(SOURCE_TARGETS, SOURCE_OBJECT, NEW_OBJECT, swapped=False)
    swapped = warp_targets(SOURCE_TARGETS, SOURCE_OBJECT, NEW_OBJECT, swapped=True)
    path = interpolate(EE_START, correct[0], INTERP_STEPS)
    relative_ok = all(
        close(
            translate(new, invert_translate(NEW_OBJECT)),
            translate(old, invert_translate(SOURCE_OBJECT)),
        )
        for old, new in zip(SOURCE_TARGETS, correct)
    )

    gen_rate = generation_rate(GEN_SUCCESSES, GEN_ATTEMPTS)
    # Policy success on the kept set is a different number (MimicGen Stack real: 36%).
    # The fixture only asserts the two rates are not interchangeable.
    policy_success_on_kept = 0.36

    checks = {
        "alpha1_matches_count_share": abs(
            alpha1[LARGEST] - REAL_COUNTS[LARGEST] / n_raw_real,
        )
        < 1e-12,
        "alpha0_is_uniform": all(abs(alpha0[name] - 0.25) < 1e-12 for name in REAL_COUNTS),
        "synth_to_largest_drops_effective_domains": deff_largest < deff_base - 0.04,
        "synth_to_largest_raises_max_share": (
            alpha1_largest[LARGEST] > alpha1[LARGEST] + 0.01
        ),
        "synth_to_smallest_raises_small_share": (
            alpha1_smallest[SMALLEST] > alpha1[SMALLEST] + 0.15
        ),
        "synth_to_smallest_raises_effective_domains": deff_smallest > deff_base + 0.4,
        "duplicate_unique_count_unchanged": n_unique_dup == n_unique_real,
        "duplicate_does_not_increase_neff": n_eff_dup <= n_eff_real,
        "duplicate_raw_count_still_grows": n_raw_after == n_raw_real + SYNTH_COUNT,
        "unique_synth_increases_nunique": n_unique_unique == n_unique_real + SYNTH_COUNT,
        "unique_synth_neff_equals_nunique": abs(n_eff_unique - n_unique_unique) < 1e-12,
        "transform_preserves_object_relative_pose": relative_ok
        and all(close(got, want) for got, want in zip(correct, CORRECT_TARGETS)),
        "swapped_object_poses_miss_the_object": all(
            close(got, want) for got, want in zip(swapped, SWAPPED_TARGETS)
        ),
        "interpolation_starts_at_ee_and_ends_at_first_target": (
            close(path[0], EE_START) and close(path[-1], correct[0])
        ),
        "generation_rate_is_not_policy_success": abs(gen_rate - policy_success_on_kept)
        > 0.3,
        "uniform_mixture_has_four_effective_domains": abs(deff_uniform - 4.0) < 1e-12,
    }

    return {
        "summary": (
            "四域真实计数 (8000, 800, 200, 50) 上核对合成账："
            "2000 条只加到最大域时 α=1 的有效域数从 1.27 掉到 1.21；"
            "同样 2000 条补最小域时小域份额从 0.55% 升到 18.6%；"
            "把 2000 条写成同一条厨房轨迹的重复，有效样本量从 9050 掉到约 30，"
            "唯一哈希数不变。"
        ),
        "metrics": {
            "real_counts": REAL_COUNTS,
            "synth_count": SYNTH_COUNT,
            "batch_size": BATCH_SIZE,
            "alpha1_probs": {name: round(alpha1[name], 6) for name in DOMAIN_ORDER},
            "alpha1_largest_probs": {
                name: round(alpha1_largest[name], 6) for name in DOMAIN_ORDER
            },
            "alpha1_smallest_probs": {
                name: round(alpha1_smallest[name], 6) for name in DOMAIN_ORDER
            },
            "effective_domains_real": round(deff_base, 6),
            "effective_domains_synth_largest": round(deff_largest, 6),
            "effective_domains_synth_smallest": round(deff_smallest, 6),
            "effective_domains_uniform": round(deff_uniform, 6),
            "n_raw_real": n_raw_real,
            "n_raw_after_synth": n_raw_after,
            "n_unique_real": n_unique_real,
            "n_unique_duplicate": n_unique_dup,
            "n_unique_unique_synth": n_unique_unique,
            "n_eff_real": round(n_eff_real, 6),
            "n_eff_duplicate": round(n_eff_dup, 6),
            "n_eff_unique_synth": round(n_eff_unique, 6),
            "smallest_share_real": round(alpha1[SMALLEST], 6),
            "smallest_share_after_fill": round(alpha1_smallest[SMALLEST], 6),
            "largest_share_real": round(alpha1[LARGEST], 6),
            "largest_share_after_pad": round(alpha1_largest[LARGEST], 6),
            "correct_targets": [list(point) for point in correct],
            "swapped_targets": [list(point) for point in swapped],
            "interpolation_path": [list(point) for point in path],
            "generation_rate": round(gen_rate, 6),
            "policy_success_on_kept": policy_success_on_kept,
            "generation_attempts": GEN_ATTEMPTS,
            "generation_successes": GEN_SUCCESSES,
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="54",
    title="用合成数据补分布而不是只加条数",
    question="合成轨迹补的是数量还是分布？重复轨迹会不会被算成新的有效样本？",
    run=run,
)
