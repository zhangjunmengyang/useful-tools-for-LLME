"""Diary can update a seat tonight. Weights keep serving the old seat until you rewrite."""

from __future__ import annotations

import random
from typing import Any

from ..extra_core import ExtraExperiment
from ..lin import hebbian_fit, make_roster, matvec, nearest_name, residual_norm, unit

DIM = 20
N = 8
STEPS = 640
LR = 0.22


def run() -> dict[str, Any]:
    rng = random.Random(101)
    items, keys, values, pairs = make_roster(rng, N, DIM, "s")
    values["moved"] = unit(rng, DIM)
    w_old = hebbian_fit(pairs, DIM, STEPS, LR, random.Random(102))

    person0, desk0 = items[0]
    diary = {person: desk for person, desk in items}
    diary[person0] = "moved"

    def lookup(store: dict[str, str], person: str) -> str | None:
        return store.get(person)

    def weight_name(person: str) -> str:
        return nearest_name(matvec(w_old, keys[person]), values)

    diary_now = lookup(diary, person0)
    weight_now = weight_name(person0)
    others_old = sum(
        nearest_name(matvec(w_old, keys[p]), values) == d
        for p, d in items
        if p != person0
    )
    stale_residual = residual_norm(w_old, keys[person0], values["moved"])
    old_residual = residual_norm(w_old, keys[person0], values[desk0])

    # Rewrite with replay of the other people. Conflict already showed naive-only.
    rewrite_pairs = [(keys[person0], values["moved"])] + [
        (keys[p], values[d]) for p, d in items if p != person0
    ]
    w_new = hebbian_fit(rewrite_pairs, DIM, STEPS, LR, random.Random(103), start=w_old)
    weight_after = nearest_name(matvec(w_new, keys[person0]), values)
    others_after = sum(
        nearest_name(matvec(w_new, keys[p]), values) == d
        for p, d in items
        if p != person0
    )
    new_residual = residual_norm(w_new, keys[person0], values["moved"])

    checks = {
        "diary_updates_immediately": diary_now == "moved",
        "weights_still_serve_old_seat": weight_now == desk0,
        "old_residual_smaller_than_new_before_rewrite": old_residual < stale_residual,
        "others_untouched_while_stale": others_old >= 6,
        "rewrite_moves_the_seat": weight_after == "moved",
        "rewrite_keeps_others": others_after >= 5,
        "residual_falls_after_rewrite": new_residual < stale_residual - 0.05,
    }
    return {
        "summary": (
            f"日记把 {person0} 改到 moved，权重仍指向 {weight_now}。"
            f"改写前进新座位的残差 {stale_residual:.3f}，旧座位 {old_residual:.3f}。"
            f"带回放改写后权重指向 {weight_after}，其他人 {others_after}/{N - 1}，"
            f"新残差 {new_residual:.3f}。"
            "失败阈值：日记改了权重立刻跟着变，或改写时把其他人冲掉。"
        ),
        "metrics": {
            "diary_now": diary_now,
            "weight_now": weight_now,
            "weight_after": weight_after,
            "others_old": others_old,
            "others_after": others_after,
            "stale_residual": stale_residual,
            "old_residual": old_residual,
            "new_residual": new_residual,
        },
        "checks": checks,
    }


EXPERIMENT = ExtraExperiment(
    extra_id="stale",
    title="日记改了，权重还在说旧座位",
    question="工位当晚就换了。Mem0 立刻覆盖。权重要到哪一步才会改口？",
    lesson_hint="13,14,16",
    run=run,
)
