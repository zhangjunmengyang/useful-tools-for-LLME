"""Dump SFT walks far from the pretrained point; small on-policy steps stay closer."""

from __future__ import annotations

import math
import random
from typing import Any

from ..extra_core import ExtraExperiment
from ..lin import linear_fit, linear_mae


def _old(a: float, b: float) -> float:
    return a


def _new(a: float, b: float) -> float:
    return 2.0 * a + 3.0 * b


def _l2(weights: tuple[float, float], origin: tuple[float, float]) -> float:
    return math.sqrt((weights[0] - origin[0]) ** 2 + (weights[1] - origin[1]) ** 2)


def run() -> dict[str, Any]:
    rng = random.Random(31)
    origin = (1.0, 0.0)
    old_probes = [(float(i) * 0.3, 0.0) for i in range(1, 9)]
    new_probes = [(float(i) * 0.25, float(i) * -0.2) for i in range(1, 11)]

    noisy = []
    for _ in range(48):
        a, b = rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)
        y = _new(a, b)
        if rng.random() < 0.4:
            y += rng.choice((-7.0, 7.0))
        noisy.append(((a, b), y))
    filtered = [item for item in noisy if abs(item[1] - _new(*item[0])) < 1e-9]

    dump = linear_fit(noisy, steps=280, lr=0.12, rng=random.Random(32), start=origin)
    small = linear_fit(filtered, steps=12, lr=0.04, rng=random.Random(33), start=origin)

    dump_old = linear_mae(dump, old_probes, _old)
    small_old = linear_mae(small, old_probes, _old)
    dump_new = linear_mae(dump, new_probes, _new)
    small_new = linear_mae(small, new_probes, _new)
    origin_new = linear_mae(origin, new_probes, _new)
    dump_dist = _l2(dump, origin)
    small_dist = _l2(small, origin)

    checks = {
        "dump_farther_from_origin": dump_dist > small_dist + 0.4,
        "dump_hurts_old_more": dump_old > small_old + 0.3,
        "small_improves_new": small_new < origin_new - 0.2,
        "filtered_not_empty": len(filtered) >= 10,
        "small_stays_near_origin": small_dist < dump_dist,
    }
    return {
        "summary": (
            f"原点是旧任务 (1,0)。大步灌噪声 SFT 走到 L2={dump_dist:.3f}，"
            f"旧任务误差 {dump_old:.3f}；筛选后小步 L2={small_dist:.3f}，"
            f"旧任务 {small_old:.3f}，新任务从 {origin_new:.3f} 降到 {small_new:.3f}。"
            "失败阈值：大步反而离原点更近，或小步完全没学新规则。"
        ),
        "metrics": {
            "dump": [dump[0], dump[1]],
            "small": [small[0], small[1]],
            "dump_dist": dump_dist,
            "small_dist": small_dist,
            "dump_old": dump_old,
            "small_old": small_old,
            "dump_new": dump_new,
            "small_new": small_new,
            "origin_new": origin_new,
            "filtered": len(filtered),
        },
        "checks": checks,
    }


EXPERIMENT = ExtraExperiment(
    extra_id="onpolicy",
    title="小步 on-policy 离原点更近",
    question="把日记整袋拿去微调，会不会比筛过的小步更伤旧能力？",
    lesson_hint="20,23",
    run=run,
)
