"""Nightly GPU time is finite. Frequency beats FIFO when you cannot distill the whole diary."""

from __future__ import annotations

import random
from typing import Any

from ..extra_core import ExtraExperiment
from ..lin import hebbian_fit, make_roster, recall_hits

DIM = 12
N = 12
KEEP = 4
STEPS = 70
LR = 0.22
# Query counts for the 12 people, most to least.
FREQ = (28, 22, 18, 14, 8, 6, 3, 2, 1, 1, 1, 1)


def run() -> dict[str, Any]:
    rng = random.Random(131)
    items, keys, values, pairs = make_roster(rng, N, DIM, "q")
    counts = {items[i][0]: FREQ[i] for i in range(N)}
    stream: list[tuple[str, str]] = []
    for person, desk in items:
        stream.extend([(person, desk)] * counts[person])
    rng.shuffle(stream)
    # End of day: a burst of the four rarest names. FIFO of "last distinct" then
    # is the tail, not the people who actually got asked all day.
    rare = items[-KEEP:]
    stream.extend(rare * 3)

    seen: list[tuple[str, str]] = []
    seen_names: set[str] = set()
    for person, desk in reversed(stream):
        if person in seen_names:
            continue
        seen.append((person, desk))
        seen_names.add(person)
        if len(seen) >= KEEP:
            break
    fifo_items = list(reversed(seen))

    ranked = sorted(items, key=lambda item: counts[item[0]], reverse=True)
    freq_items = ranked[:KEEP]
    tail_head = ranked[:KEEP]

    fifo_pairs = [(keys[p], values[d]) for p, d in fifo_items]
    freq_pairs = [(keys[p], values[d]) for p, d in freq_items]
    dump_all = hebbian_fit(pairs, DIM, STEPS, LR, random.Random(132))
    w_fifo = hebbian_fit(fifo_pairs, DIM, STEPS, LR, random.Random(133))
    w_freq = hebbian_fit(freq_pairs, DIM, STEPS, LR, random.Random(134))

    dump_head = recall_hits(dump_all, keys, values, tail_head)
    fifo_head = recall_hits(w_fifo, keys, values, tail_head)
    freq_head = recall_hits(w_freq, keys, values, tail_head)
    fifo_on_fifo = recall_hits(w_fifo, keys, values, fifo_items)
    freq_on_freq = recall_hits(w_freq, keys, values, freq_items)
    dump_all_hits = recall_hits(dump_all, keys, values, items)

    checks = {
        "only_four_slots": len(freq_items) == KEEP and len(fifo_items) == KEEP,
        "freq_recalls_its_four": freq_on_freq >= 3,
        "fifo_recalls_its_four": fifo_on_fifo >= 3,
        "fifo_misses_the_head": fifo_head <= 1,
        "freq_beats_fifo_on_head_queries": freq_head >= fifo_head + 1,
        "head_is_the_four_most_asked": [p for p, _d in freq_items]
        == [p for p, _d in tail_head],
        "fifo_is_not_the_same_set": {p for p, _d in fifo_items}
        != {p for p, _d in freq_items},
    }
    return {
        "summary": (
            f"一夜只巩固 {KEEP} 条。全倒进 W 时常问仍可能中（本机 {dump_head}/4，全表 {dump_all_hits}/{N}）；"
            f"把名额给当天日志末尾，常问只中 {fifo_head}/4；按查询次数取 4 条，常问中 {freq_head}/4。"
            "失败阈值：次数选择并不比 FIFO 更能保住常问的人。"
        ),
        "metrics": {
            "dump_all_hits": dump_all_hits,
            "dump_head": dump_head,
            "fifo_head": fifo_head,
            "freq_head": freq_head,
            "fifo_on_fifo": fifo_on_fifo,
            "freq_on_freq": freq_on_freq,
            "fifo_names": [p for p, _d in fifo_items],
            "freq_names": [p for p, _d in freq_items],
            "steps": STEPS,
        },
        "checks": checks,
    }


EXPERIMENT = ExtraExperiment(
    extra_id="budget",
    title="夜间预算：一夜写不完整本日记",
    question="GPU 晚上只能跑几百步时，该按查询次数巩固，还是把当天日志末尾倒进去？",
    lesson_hint="09,13,16",
    run=run,
)
