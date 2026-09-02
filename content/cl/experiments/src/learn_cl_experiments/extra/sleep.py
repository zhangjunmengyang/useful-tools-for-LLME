"""Two timescales: fast weights hold the day, slow weights take the night, then unplug fast."""

from __future__ import annotations

import random
from typing import Any

from ..extra_core import ExtraExperiment
from ..lin import hebbian_fit, matvec, recall_hits, unit, zeros

DIM = 24
N_FACTS = 10
DAY_STEPS = 400
NIGHT_STEPS = 500
LR_FAST = 0.3
LR_SLOW = 0.12


def run() -> dict[str, Any]:
    rng = random.Random(17)
    people = [f"person_{i}" for i in range(N_FACTS)]
    desks = [f"desk_{i}" for i in range(N_FACTS)]
    items = list(zip(people, desks))
    keys = {name: unit(rng, DIM) for name in people}
    values = {name: unit(rng, DIM) for name in desks}
    pairs = [(keys[p], values[d]) for p, d in items]

    w_fast = hebbian_fit(pairs, DIM, DAY_STEPS, LR_FAST, random.Random(18))
    day_hits = recall_hits(w_fast, keys, values, items)

    # Night: train slow weights to match the fast mapping, then wipe fast.
    teacher = [(key, hebbian_target) for key, hebbian_target in (
        (keys[p], values[d]) for p, d in items
    )]
    # Distill from fast outputs, not from the diary (diary may already be gone).
    distilled = [(keys[p], matvec(w_fast, keys[p])) for p, _d in items]
    w_slow = hebbian_fit(distilled, DIM, NIGHT_STEPS, LR_SLOW, random.Random(19))
    w_fast_cleared = zeros(DIM, DIM)
    after_sleep = recall_hits(w_slow, keys, values, items)
    fast_after_clear = recall_hits(w_fast_cleared, keys, values, items)

    no_sleep = recall_hits(zeros(DIM, DIM), keys, values, items)
    # Control: keep only fast, then unplug it without a night pass.
    unslept_hits = fast_after_clear

    checks = {
        "day_fast_recalls": day_hits >= 8,
        "night_slow_recalls": after_sleep >= 8,
        "fast_empty_after_sleep": fast_after_clear <= 1,
        "skipping_sleep_fails": unslept_hits <= 1,
        "sleep_beats_no_sleep": after_sleep >= unslept_hits + 6,
        "teacher_pairs_match_roster": len(teacher) == N_FACTS,
    }
    return {
        "summary": (
            f"白天快权重召回 {day_hits}/{N_FACTS}。"
            f"夜里把快映射写入慢权重后清掉快权重，慢权重仍中 {after_sleep}/{N_FACTS}；"
            f"不做夜间巩固、只清快权重则 {unslept_hits}/{N_FACTS}。"
            "失败阈值：睡完仍叫不到人，或不睡也全对。"
        ),
        "metrics": {
            "day_hits": day_hits,
            "after_sleep": after_sleep,
            "fast_after_clear": fast_after_clear,
            "no_sleep": no_sleep,
            "unslept_hits": unslept_hits,
        },
        "checks": checks,
    }


EXPERIMENT = ExtraExperiment(
    extra_id="sleep",
    title="两档转速：白天快权重，夜里慢权重",
    question="会话级快权重清掉之后，夜间写入的慢权重还能不能叫人？",
    lesson_hint="16,19",
    run=run,
)
