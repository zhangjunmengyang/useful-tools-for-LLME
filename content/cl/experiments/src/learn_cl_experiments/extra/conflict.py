"""Updating one fact in memory vs rewriting it in weights with and without replay."""

from __future__ import annotations

import random
from typing import Any

from ..extra_core import ExtraExperiment
from ..lin import hebbian_fit, recall_hits, unit

DIM = 24
STEPS = 500
LR = 0.25


def run() -> dict[str, Any]:
    rng = random.Random(4)
    people = [f"person_{index}" for index in range(10)]
    old_desks = [f"desk_{index}" for index in range(10)]
    items_old = list(zip(people, old_desks))
    keys = {name: unit(rng, DIM) for name in people}
    values = {name: unit(rng, DIM) for name in old_desks}
    values["desk_moved"] = unit(rng, DIM)

    pairs_old = [(keys[p], values[d]) for p, d in items_old]
    w_old = hebbian_fit(pairs_old, DIM, STEPS, LR, rng)
    old_hits = recall_hits(w_old, keys, values, items_old)

    memory = {p: d for p, d in items_old}
    memory["person_0"] = "desk_moved"
    memory_new = memory["person_0"] == "desk_moved"
    weights_still_old = (
        recall_hits(w_old, keys, values, [("person_0", "desk_0")]) == 1
    )

    pairs_overwrite = [(keys["person_0"], values["desk_moved"])]
    w_naive = hebbian_fit(pairs_overwrite, DIM, STEPS, LR, random.Random(5))
    naive_new = recall_hits(w_naive, keys, values, [("person_0", "desk_moved")])
    naive_others = recall_hits(
        w_naive,
        keys,
        values,
        [(p, d) for p, d in items_old if p != "person_0"],
    )

    pairs_replay = pairs_overwrite + [
        (keys[p], values[d]) for p, d in items_old if p != "person_0"
    ]
    w_replay = hebbian_fit(pairs_replay, DIM, STEPS, LR, random.Random(5))
    replay_new = recall_hits(w_replay, keys, values, [("person_0", "desk_moved")])
    replay_others = recall_hits(
        w_replay,
        keys,
        values,
        [(p, d) for p, d in items_old if p != "person_0"],
    )

    checks = {
        "old_weights_know_original_roster": old_hits >= 8,
        "memory_can_overwrite_one_seat": memory_new,
        "weights_keep_old_seat_until_retrained": weights_still_old,
        "naive_rewrite_forgets_other_people": naive_others <= 3,
        "replay_keeps_others_and_new_seat": replay_new == 1 and replay_others >= 6,
    }
    return {
        "summary": (
            f"旧花名册写入 W 后召回 {old_hits}/10。记忆把 person_0 改成 desk_moved，"
            f"权重仍指向旧座位。只练这一条时其他人只剩 {naive_others} 条；"
            f"加上回放后新座位 {replay_new}、其他人 {replay_others}。"
            "失败阈值：只改一条却不忘其他人，或回放后新座位没写进去。"
        ),
        "metrics": {
            "old_hits": old_hits,
            "naive_new": naive_new,
            "naive_others": naive_others,
            "replay_new": replay_new,
            "replay_others": replay_others,
        },
        "checks": checks,
    }


EXPERIMENT = ExtraExperiment(
    extra_id="conflict",
    title="改一条座位",
    question="日记覆盖和新写权重，谁会把花名册其余人冲掉？",
    lesson_hint="13,14,16",
    run=run,
)
