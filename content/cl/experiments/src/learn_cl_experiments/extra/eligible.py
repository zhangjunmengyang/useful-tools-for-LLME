"""Only write facts that have been retrieved often. One-shot noise should not enter W."""

from __future__ import annotations

import random
from typing import Any

from ..extra_core import ExtraExperiment
from ..lin import hebbian_fit, make_roster, recall_hits, unit

DIM = 20
N_REAL = 6
N_NOISE = 6
STEPS = 620
LR = 0.22
ELIGIBLE_AT = 4
REAL_REPEATS = 8


def run() -> dict[str, Any]:
    rng = random.Random(121)
    real_items, keys, values, real_pairs = make_roster(rng, N_REAL, DIM, "r")
    noise_people = [f"n{i}" for i in range(N_NOISE)]
    noise_desks = [f"nd{i}" for i in range(N_NOISE)]
    for person, desk in zip(noise_people, noise_desks):
        keys[person] = unit(rng, DIM)
        values[desk] = unit(rng, DIM)
    noise_items = list(zip(noise_people, noise_desks))
    noise_pairs = [(keys[p], values[d]) for p, d in noise_items]

    stream: list[tuple[list[float], list[float]]] = []
    counts: dict[str, int] = {p: 0 for p, _d in real_items + noise_items}
    for person, desk in real_items:
        for _ in range(REAL_REPEATS):
            stream.append((keys[person], values[desk]))
            counts[person] += 1
    for person, desk in noise_items:
        stream.append((keys[person], values[desk]))
        counts[person] += 1
    rng.shuffle(stream)

    dump_pairs = real_pairs + noise_pairs
    w_dump = hebbian_fit(dump_pairs, DIM, STEPS, LR, random.Random(122))
    dump_real = recall_hits(w_dump, keys, values, real_items)
    dump_noise = recall_hits(w_dump, keys, values, noise_items)

    eligible_pairs = [
        (keys[p], values[d])
        for p, d in real_items + noise_items
        if counts[p] >= ELIGIBLE_AT
    ]
    w_elig = hebbian_fit(eligible_pairs, DIM, STEPS, LR, random.Random(123))
    elig_real = recall_hits(w_elig, keys, values, real_items)
    elig_noise = recall_hits(w_elig, keys, values, noise_items)
    n_eligible = len(eligible_pairs)

    checks = {
        "real_facts_are_frequent": all(counts[p] >= ELIGIBLE_AT for p, _d in real_items),
        "noise_is_rare": all(counts[p] < ELIGIBLE_AT for p, _d in noise_items),
        "dump_memorizes_noise": dump_noise >= 4,
        "eligible_keeps_real": elig_real >= 5,
        "eligible_rejects_noise": elig_noise <= 1,
        "eligible_set_is_only_real": n_eligible == N_REAL,
        "dump_real_also_works": dump_real >= 4,
    }
    return {
        "summary": (
            f"6 条真座位各出现 {REAL_REPEATS} 次，6 条噪声只出现 1 次。"
            f"全倒进 W：真 {dump_real}/{N_REAL}，噪声 {dump_noise}/{N_NOISE}。"
            f"只写出现 ≥{ELIGIBLE_AT} 次的：真 {elig_real}，噪声 {elig_noise}。"
            "失败阈值：筛选后噪声仍进得去，或真座位反而丢了。"
        ),
        "metrics": {
            "dump_real": dump_real,
            "dump_noise": dump_noise,
            "elig_real": elig_real,
            "elig_noise": elig_noise,
            "n_eligible": n_eligible,
            "eligible_at": ELIGIBLE_AT,
        },
        "checks": checks,
    }


EXPERIMENT = ExtraExperiment(
    extra_id="eligible",
    title="资格写入：只巩固反复问对的",
    question="第一次检索到的错座位，要不要当晚就写进权重？",
    lesson_hint="13,16,18",
    run=run,
)
