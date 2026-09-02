"""When diary and weights disagree, keep the key in the diary. Graduate only after rewrite."""

from __future__ import annotations

import random
from typing import Any

from ..extra_core import ExtraExperiment
from ..lin import (
    hebbian_fit,
    make_roster,
    matvec,
    nearest_name,
    residual_norm,
    unit,
)

DIM = 20
N = 8
STEPS = 640
LR = 0.22
THRESHOLD = 0.45


def run() -> dict[str, Any]:
    rng = random.Random(171)
    items, keys, values, pairs = make_roster(rng, N, DIM, "d")
    values["moved"] = unit(rng, DIM)
    w = hebbian_fit(pairs, DIM, STEPS, LR, random.Random(172))

    person0, desk0 = items[0]
    diary = {person: desk for person, desk in items}
    diary[person0] = "moved"

    res_new = residual_norm(w, keys[person0], values["moved"])
    res_old = residual_norm(w, keys[person0], values[desk0])
    weight_name = nearest_name(matvec(w, keys[person0]), values)
    keep_in_diary = res_new > THRESHOLD
    others_ok = all(
        residual_norm(w, keys[p], values[d]) < THRESHOLD for p, d in items[1:]
    )

    rewrite = [(keys[person0], values["moved"])] + [
        (keys[p], values[d]) for p, d in items if p != person0
    ]
    w2 = hebbian_fit(rewrite, DIM, STEPS, LR, random.Random(173), start=w)
    res_after = residual_norm(w2, keys[person0], values["moved"])
    can_graduate = res_after < THRESHOLD
    after_name = nearest_name(matvec(w2, keys[person0]), values)
    if can_graduate:
        del diary[person0]
    still_kept = person0 in diary

    checks = {
        "weights_still_old_before_rewrite": weight_name == desk0,
        "new_residual_is_high": res_new > res_old + 0.1,
        "policy_keeps_conflict_in_diary": keep_in_diary,
        "others_already_below_threshold": others_ok,
        "after_rewrite_can_graduate": can_graduate and after_name == "moved",
        "graduated_key_leaves_diary": still_kept is False,
    }
    return {
        "summary": (
            f"日记已改成 moved，权重仍指向 {weight_name}。"
            f"对新座位残差 {res_new:.3f}（阈值 {THRESHOLD}），这条还不能卸。"
            f"改写后残差 {res_after:.3f}，最近邻 {after_name}，日记里{'还在' if still_kept else '已删'}。"
            "失败阈值：冲突时就卸库，或改写后仍达不到毕业线。"
        ),
        "metrics": {
            "res_new": res_new,
            "res_old": res_old,
            "res_after": res_after,
            "threshold": THRESHOLD,
            "weight_name": weight_name,
            "after_name": after_name,
            "keep_in_diary": keep_in_diary,
            "still_kept": still_kept,
        },
        "checks": checks,
    }


EXPERIMENT = ExtraExperiment(
    extra_id="disagree",
    title="日记和权重打架时先别卸",
    question="Mem0 说工位换了，权重还说旧的。这种时候能不能把这一条从日记里删掉？",
    lesson_hint="13,16,24",
    run=run,
)
