"""Nightly loop: probe without the diary. Hits leave the store. Misses stay for another night."""

from __future__ import annotations

import random
from typing import Any

from ..extra_core import ExtraExperiment
from ..lin import hebbian_fit, make_roster, matvec, nearest_name

DIM = 20
N = 8
NIGHT1_TAKE = 3
NIGHT1 = 280
NIGHT2 = 640
LR = 0.24


def _hits(
    weights: list[list[float]],
    keys: dict[str, list[float]],
    values: dict[str, list[float]],
    items: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    passed = []
    for person, desk in items:
        predicted = nearest_name(matvec(weights, keys[person]), values)
        if predicted == desk:
            passed.append((person, desk))
    return passed


def run() -> dict[str, Any]:
    rng = random.Random(181)
    items, keys, values, pairs = make_roster(rng, N, DIM, "k")
    diary = {person: desk for person, desk in items}

    # Night 1 only has time for a prefix of the diary, like a GPU job that
    # stops at the budget. The rest stay for the next night.
    tonight = items[:NIGHT1_TAKE]
    w1 = hebbian_fit(
        [(keys[p], values[d]) for p, d in tonight],
        DIM,
        NIGHT1,
        LR,
        random.Random(182),
    )
    night1_pass = _hits(w1, keys, values, items)
    night1_fail = [(p, d) for p, d in items if (p, d) not in night1_pass]
    for person, _desk in night1_pass:
        diary.pop(person, None)

    leftover_pairs = [(keys[p], values[d]) for p, d in night1_fail]
    # Dream the graduated facts from W itself so night 2 does not overwrite them.
    dream_pairs = [(keys[p], matvec(w1, keys[p])) for p, _d in night1_pass]
    w2 = hebbian_fit(
        leftover_pairs + dream_pairs,
        DIM,
        NIGHT2,
        LR,
        random.Random(183),
        start=w1,
    )
    for person, _desk in _hits(w2, keys, values, night1_fail):
        diary.pop(person, None)

    unplug_hits = len(_hits(w2, keys, values, items))
    diary_left = len(diary)
    night1_n = len(night1_pass)

    checks = {
        "night1_underfits": night1_n <= N - 2,
        "night1_does_not_score_zero": night1_n >= 1,
        "graduated_leave_diary": all(p not in diary for p, _d in night1_pass),
        "failures_stay_after_night1": len(night1_fail) >= 2,
        "after_two_nights_weights_cover": unplug_hits >= 6,
        "diary_shrinks": diary_left <= max(0, N - 6),
    }
    return {
        "summary": (
            f"第一夜只来得及练 {NIGHT1_TAKE} 条，卸库探针过关 {night1_n}/{N}，这些从日记删掉；"
            f"没过的 {len(night1_fail)} 条留到第二夜。"
            f"第二夜用没过的日记 + 已过关事实的做梦回放，卸库后 {unplug_hits}/{N}，"
            f"日记还剩 {diary_left}。"
            "失败阈值：第一夜就全过（看不出筛选），或两夜之后权重仍不会。"
        ),
        "metrics": {
            "night1_pass": night1_n,
            "night1_fail": len(night1_fail),
            "unplug_hits": unplug_hits,
            "diary_left": diary_left,
            "night1_take": NIGHT1_TAKE,
            "night1_steps": NIGHT1,
            "night2_steps": NIGHT2,
        },
        "checks": checks,
    }


EXPERIMENT = ExtraExperiment(
    extra_id="keepfail",
    title="过关才出日记：卸库探针当毕业考",
    question="夜间写完就宣布「已经进权重了」？还是先拔掉日记考一遍，没过的留到明天？",
    lesson_hint="16,23,24",
    run=run,
)
