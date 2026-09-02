"""Deleting a row in the diary does not delete it from W. Unlearning has to be a write."""

from __future__ import annotations

import random
from typing import Any

from ..extra_core import ExtraExperiment
from ..lin import hebbian_fit, make_roster, matvec, nearest_name, recall_hits, unit

DIM = 20
N = 8
STEPS = 640
LR = 0.22


def run() -> dict[str, Any]:
    rng = random.Random(141)
    items, keys, values, pairs = make_roster(rng, N, DIM, "t")
    values["gone"] = unit(rng, DIM)
    w_old = hebbian_fit(pairs, DIM, STEPS, LR, random.Random(142))
    before = recall_hits(w_old, keys, values, items)

    person0, desk0 = items[0]
    others = [(p, d) for p, d in items if p != person0]
    diary = {person: desk for person, desk in items}
    del diary[person0]
    diary_gone = person0 not in diary
    still_in_w = nearest_name(matvec(w_old, keys[person0]), values) == desk0

    # Active unlearn: write this key toward a tombstone, replay everyone else.
    unlearn_pairs = [(keys[person0], values["gone"])] + [
        (keys[p], values[d]) for p, d in others
    ]
    w_unlearn = hebbian_fit(
        unlearn_pairs,
        DIM,
        STEPS,
        LR,
        random.Random(143),
        start=w_old,
    )
    after_name = nearest_name(matvec(w_unlearn, keys[person0]), values)
    others_after = recall_hits(w_unlearn, keys, values, others)

    checks = {
        "roster_was_in_weights": before >= 7,
        "diary_delete_is_instant": diary_gone,
        "weights_keep_deleted_person": still_in_w,
        "unlearn_leaves_old_desk": after_name != desk0,
        "unlearn_points_at_tombstone": after_name == "gone",
        "others_survive_unlearn": others_after >= 5,
    }
    return {
        "summary": (
            f"花名册写入后 {before}/{N}。日记删掉 {person0}，权重仍指向 {desk0}。"
            f"主动把该键写向墓碑后，最近邻是 {after_name}，其他人 {others_after}/{N - 1}。"
            "失败阈值：删日记就等于从权重里消失，或遗忘一个人时把花名册冲掉。"
        ),
        "metrics": {
            "before": before,
            "still_in_w": still_in_w,
            "after_name": after_name,
            "others_after": others_after,
            "old_desk": desk0,
        },
        "checks": checks,
    }


EXPERIMENT = ExtraExperiment(
    extra_id="tombstone",
    title="主动遗忘：删日记不等于改权重",
    question="有人离职了。Mem0 可以删一行。权重里的座位怎么拿掉，才不会把其他人一起冲掉？",
    lesson_hint="13,14,16",
    run=run,
)
