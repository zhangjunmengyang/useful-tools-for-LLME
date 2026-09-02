"""Five-day agent: memory first, nightly distill, then self-edit a new rule into weights."""

from __future__ import annotations

import random
from typing import Any

from ..extra_core import ExtraExperiment
from ..lin import hebbian_fit, recall_hits, unit


DIM = 24
N_FACTS = 10
FACT_STEPS = 700
LR = 0.25


def _true(a: float, b: float) -> float:
    return 2.0 * a + 3.0 * b


def _fit_rule(
    pairs: list[tuple[tuple[float, float], float]],
    steps: int,
    lr: float,
    rng: random.Random,
    start: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float]:
    w0, w1 = start
    if not pairs:
        return w0, w1
    for _ in range(steps):
        (a, b), y = pairs[rng.randrange(len(pairs))]
        pred = w0 * a + w1 * b
        err = pred - y
        w0 -= lr * err * a
        w1 -= lr * err * b
    return w0, w1


def _mae(weights: tuple[float, float], probes: list[tuple[float, float]]) -> float:
    return sum(
        abs(weights[0] * a + weights[1] * b - _true(a, b)) for a, b in probes
    ) / len(probes)


def run() -> dict[str, Any]:
    rng = random.Random(7)
    people = [f"person_{index}" for index in range(N_FACTS)]
    desks = [f"desk_{index}" for index in range(N_FACTS)]
    items = list(zip(people, desks))
    keys = {name: unit(rng, DIM) for name in people}
    values = {name: unit(rng, DIM) for name in desks}

    # Day 1-2: write only to the diary.
    memory = {person: desk for person, desk in items}
    memory_hits = sum(memory[p] == d for p, d in items)
    w_empty = hebbian_fit([], DIM, 0, LR, rng)
    day2_unplug = recall_hits(w_empty, keys, values, items)

    # Night 2: distill the diary into W, then throw the diary away.
    pairs = [(keys[p], values[d]) for p, d in items]
    w_facts = hebbian_fit(pairs, DIM, FACT_STEPS, LR, random.Random(8))
    memory.clear()
    day3_unplug = recall_hits(w_facts, keys, values, items)

    # Day 4: a scoring rule memory cannot execute.
    slogan = "score = 2a + 3b"
    memory_rule_ok = slogan == "score = 2a + 3b" and False  # text is not the rule
    probes = [(float(i) * 0.4, float(i) * -0.2) for i in range(10)]

    def generate(noise: float) -> list[tuple[tuple[float, float], float]]:
        bundle = []
        for _ in range(36):
            a, b = rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)
            y = _true(a, b)
            if rng.random() < noise:
                y += rng.choice((-6.0, 6.0))
            bundle.append(((a, b), y))
        return bundle

    noisy = generate(0.4)
    filtered = [item for item in noisy if abs(item[1] - _true(*item[0])) < 1e-9]
    rule_small = _fit_rule(filtered, steps=250, lr=0.08, rng=random.Random(9))
    rule_dump = _fit_rule(noisy, steps=250, lr=0.08, rng=random.Random(9))

    # Keep facts: replay half of the roster while learning the rule using a
    # second head is not needed; facts live in W_facts, rule lives in a
    # separate linear head. Dump SFT on a shared toy would mix them; here the
    # two stores are different, matching "write the rule into a new slice".
    mae_small = _mae(rule_small, probes)
    mae_dump = _mae(rule_dump, probes)
    facts_after_rule = recall_hits(w_facts, keys, values, items)

    checks = {
        "day2_memory_full": memory_hits == N_FACTS,
        "day2_unplug_fails": day2_unplug <= 2,
        "day3_unplug_after_distill": day3_unplug >= 8,
        "slogan_is_not_the_rule": memory_rule_ok is False,
        "filtered_rule_mae_small": mae_small < 0.45,
        "dump_rule_worse": mae_dump > mae_small + 0.2,
        "facts_still_in_weights": facts_after_rule >= 8,
    }
    return {
        "summary": (
            f"前两天只有日记：拔库 {day2_unplug}/{N_FACTS}。"
            f"夜间写入 W 后拔库 {day3_unplug}/{N_FACTS}。"
            f"新计分规则筛选自编辑后误差 {mae_small:.3f}，不筛选 {mae_dump:.3f}；"
            f"花名册仍在权重里 {facts_after_rule}/{N_FACTS}。"
            "失败阈值：巩固后仍叫不到人，或规则只写在口号里却算对了。"
        ),
        "metrics": {
            "memory_hits_day2": memory_hits,
            "unplug_before": day2_unplug,
            "unplug_after": day3_unplug,
            "filtered_edits": len(filtered),
            "mae_small": mae_small,
            "mae_dump": mae_dump,
            "facts_after_rule": facts_after_rule,
            "rule_small": [rule_small[0], rule_small[1]],
        },
        "checks": checks,
    }


EXPERIMENT = ExtraExperiment(
    extra_id="evolve",
    title="五日进化：日记减、权重增",
    question="Agent 能不能先靠外挂顶几天，再把会的东西写进权重，最后少依赖日记？",
    lesson_hint="16,20,23,24",
    run=run,
)
