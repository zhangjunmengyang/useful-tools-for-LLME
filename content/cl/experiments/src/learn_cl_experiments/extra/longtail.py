"""Unplug the frequent facts; keep the long tail in the diary. The store shrinks, it does not vanish."""

from __future__ import annotations

import random
from typing import Any

from ..extra_core import ExtraExperiment
from ..lin import hebbian_fit, make_roster, recall_hits

DIM = 18
N_HEAD = 8
N_TAIL = 16
STEPS = 700
LR = 0.22


def run() -> dict[str, Any]:
    rng = random.Random(151)
    head_items, keys, values, head_pairs = make_roster(rng, N_HEAD, DIM, "h")
    tail_items, tail_keys, tail_values, _tail_pairs = make_roster(rng, N_TAIL, DIM, "t")
    keys.update(tail_keys)
    values.update(tail_values)

    diary = {p: d for p, d in head_items + tail_items}
    w_head = hebbian_fit(head_pairs, DIM, STEPS, LR, random.Random(152))

    def hybrid_hits(
        plugged_tail: bool,
        items: list[tuple[str, str]],
    ) -> int:
        hits = 0
        for person, desk in items:
            if person.startswith("hp"):
                predicted = recall_hits(w_head, keys, values, [(person, desk)])
                hits += predicted
            elif plugged_tail and diary.get(person) == desk:
                hits += 1
        return hits

    head_in_w = recall_hits(w_head, keys, values, head_items)
    tail_in_w = recall_hits(w_head, keys, values, tail_items)
    unplug_all_head = head_in_w
    unplug_all_tail = tail_in_w
    hybrid_head = hybrid_hits(True, head_items)
    hybrid_tail = hybrid_hits(True, tail_items)
    unplug_hybrid_tail = hybrid_hits(False, tail_items)
    diary_tail = sum(diary.get(p) == d for p, d in tail_items)

    checks = {
        "head_lives_in_weights": head_in_w >= 6,
        "tail_not_in_weights": tail_in_w <= 4,
        "unplug_all_loses_tail": unplug_all_tail <= 4,
        "hybrid_keeps_head": hybrid_head >= 6,
        "hybrid_keeps_tail_from_diary": hybrid_tail == N_TAIL,
        "unplug_tail_diary_loses_tail": unplug_hybrid_tail <= 4,
        "diary_still_has_tail": diary_tail == N_TAIL,
    }
    return {
        "summary": (
            f"常问 {N_HEAD} 条写入 W 后 {head_in_w}/{N_HEAD}；长尾 {N_TAIL} 条在权重里只有 {tail_in_w}。"
            f"日记还留着长尾时，头 {hybrid_head}、尾 {hybrid_tail}；"
            f"把长尾日记也卸掉，尾只剩 {unplug_hybrid_tail}。"
            "失败阈值：长尾不写进权重却还能在卸库后召回，或常问的反而没写进去。"
        ),
        "metrics": {
            "head_in_w": head_in_w,
            "tail_in_w": tail_in_w,
            "unplug_all_head": unplug_all_head,
            "unplug_all_tail": unplug_all_tail,
            "hybrid_head": hybrid_head,
            "hybrid_tail": hybrid_tail,
            "unplug_hybrid_tail": unplug_hybrid_tail,
            "n_head": N_HEAD,
            "n_tail": N_TAIL,
        },
        "checks": checks,
    }


EXPERIMENT = ExtraExperiment(
    extra_id="longtail",
    title="长尾仍留库：先卸常问的，不是一夜清空",
    question="真要做到不靠外挂，是不是第一夜就把 Mem0 整库删掉？",
    lesson_hint="13,16,24",
    run=run,
)
