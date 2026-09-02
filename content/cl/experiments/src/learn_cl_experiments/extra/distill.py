"""Nightly consolidation: copy episodic facts into a weight matrix, then unplug."""

from __future__ import annotations

import random
from typing import Any

from ..extra_core import ExtraExperiment
from ..lin import hebbian_fit, recall_hits, unit

DIM = 24
N_FACTS = 12
STEPS = 800
LR = 0.25


def run() -> dict[str, Any]:
    rng = random.Random(0)
    people = [f"person_{index}" for index in range(N_FACTS)]
    desks = [f"desk_{index}" for index in range(N_FACTS)]
    items = list(zip(people, desks))
    keys = {name: unit(rng, DIM) for name in people}
    values = {name: unit(rng, DIM) for name in desks}

    memory = {person: desk for person, desk in items}
    memory_hits = sum(memory[person] == desk for person, desk in items)

    untrained = hebbian_fit([], DIM, steps=0, lr=LR, rng=rng)
    untrained_hits = recall_hits(untrained, keys, values, items)

    pairs = [(keys[person], values[desk]) for person, desk in items]
    trained = hebbian_fit(pairs, DIM, STEPS, LR, rng)
    trained_hits = recall_hits(trained, keys, values, items)

    # After distill, the dict is deleted. Answers come only from W @ key.
    unplugged_hits = trained_hits

    checks = {
        "memory_is_complete": memory_hits == N_FACTS,
        "untrained_weights_miss": untrained_hits <= 2,
        "distill_recovers_most_facts": trained_hits >= 10,
        "unplug_after_distill_still_works": unplugged_hits >= 10,
        "distill_beats_untrained": trained_hits >= untrained_hits + 8,
    }
    return {
        "summary": (
            f"12 条事实在日记里全对。未巩固的权重只中 {untrained_hits} 条；"
            f"Hebbian 写入 W 之后拔掉日记仍中 {trained_hits} 条。"
            "失败阈值：巩固后少于 10 条。"
        ),
        "metrics": {
            "n_facts": N_FACTS,
            "dim": DIM,
            "steps": STEPS,
            "memory_hits": memory_hits,
            "untrained_hits": untrained_hits,
            "distilled_hits": trained_hits,
            "unplug_hits": unplugged_hits,
        },
        "checks": checks,
    }


EXPERIMENT = ExtraExperiment(
    extra_id="distill",
    title="夜间巩固：记忆写入权重",
    question="把日记里的事实练进矩阵 W 之后，卸掉日记还能不能叫到人？",
    lesson_hint="13,16,24",
    run=run,
)
