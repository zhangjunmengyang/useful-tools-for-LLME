"""Sequential self-edits into the same W: later batches wipe earlier ones unless replayed."""

from __future__ import annotations

import random
from typing import Any

from ..extra_core import ExtraExperiment
from ..lin import hebbian_fit, recall_hits, unit, zeros

DIM = 18
BATCH = 5
N_BATCHES = 4
STEPS_NAIVE = 800
STEPS_REPLAY = 1400
LR = 0.24


def _batch(
    rng: random.Random,
    tag: str,
) -> tuple[list[tuple[str, str]], dict[str, list[float]], dict[str, list[float]]]:
    people = [f"{tag}_p{i}" for i in range(BATCH)]
    desks = [f"{tag}_d{i}" for i in range(BATCH)]
    items = list(zip(people, desks))
    keys = {name: unit(rng, DIM) for name in people}
    values = {name: unit(rng, DIM) for name in desks}
    return items, keys, values


def run() -> dict[str, Any]:
    rng = random.Random(23)
    batches = [_batch(rng, f"b{i}") for i in range(N_BATCHES)]
    all_keys: dict[str, list[float]] = {}
    all_values: dict[str, list[float]] = {}
    for items, keys, values in batches:
        all_keys.update(keys)
        all_values.update(values)

    naive = zeros(DIM, DIM)
    naive_hits: list[int] = []
    for items, keys, values in batches:
        pairs = [(keys[p], values[d]) for p, d in items]
        naive = hebbian_fit(pairs, DIM, STEPS_NAIVE, LR, random.Random(24 + len(naive_hits)), start=naive)
        naive_hits.append(recall_hits(naive, all_keys, all_values, items))

    replay = zeros(DIM, DIM)
    seen: list[tuple[list[float], list[float]]] = []
    replay_hits: list[int] = []
    for items, keys, values in batches:
        seen.extend((keys[p], values[d]) for p, d in items)
        replay = hebbian_fit(seen, DIM, STEPS_REPLAY, LR, random.Random(30 + len(replay_hits)))
        replay_hits.append(recall_hits(replay, all_keys, all_values, items))

    first_after_naive = recall_hits(naive, all_keys, all_values, batches[0][0])
    last_after_naive = recall_hits(naive, all_keys, all_values, batches[-1][0])
    first_after_replay = recall_hits(replay, all_keys, all_values, batches[0][0])
    last_after_replay = recall_hits(replay, all_keys, all_values, batches[-1][0])

    checks = {
        "naive_learns_latest": last_after_naive >= 4,
        "naive_forgets_first": first_after_naive <= 2,
        "replay_keeps_first": first_after_replay >= 3,
        "replay_keeps_latest": last_after_replay >= 4,
        "replay_beats_naive_on_first": first_after_replay >= first_after_naive + 2,
    }
    return {
        "summary": (
            f"{N_BATCHES} 次自编辑写入同一张 W。无回放：第 1 批 {first_after_naive}/{BATCH}，"
            f"最后一批 {last_after_naive}/{BATCH}。"
            f"带回放：第 1 批 {first_after_replay}/{BATCH}，最后一批 {last_after_replay}/{BATCH}。"
            "失败阈值：连续写完第 1 批还在，或回放仍救不回第 1 批。"
        ),
        "metrics": {
            "naive_hits": naive_hits,
            "replay_hits": replay_hits,
            "first_after_naive": first_after_naive,
            "last_after_naive": last_after_naive,
            "first_after_replay": first_after_replay,
            "last_after_replay": last_after_replay,
        },
        "checks": checks,
    }


EXPERIMENT = ExtraExperiment(
    extra_id="seqedit",
    title="连续自编辑会忘更早的一批",
    question="SEAL 式一连串写入同一张权重，不带回放，第一批还在吗？",
    lesson_hint="20,23",
    run=run,
)
