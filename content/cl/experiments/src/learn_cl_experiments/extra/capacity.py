"""Associative memory has a capacity. Dumping the whole diary into a small W collides."""

from __future__ import annotations

import random
from typing import Any

from ..extra_core import ExtraExperiment
from ..lin import hebbian_fit, recall_hits, unit

DIM = 12
N_SMALL = 6
N_LARGE = 36
STEPS_SMALL = 500
STEPS_LARGE = 900
LR = 0.2


def _roster(rng: random.Random, count: int, prefix: str) -> tuple[
    list[tuple[str, str]],
    dict[str, list[float]],
    dict[str, list[float]],
]:
    people = [f"{prefix}_p{i}" for i in range(count)]
    desks = [f"{prefix}_d{i}" for i in range(count)]
    items = list(zip(people, desks))
    keys = {name: unit(rng, DIM) for name in people}
    values = {name: unit(rng, DIM) for name in desks}
    return items, keys, values


def run() -> dict[str, Any]:
    rng = random.Random(13)
    small_items, small_keys, small_values = _roster(rng, N_SMALL, "s")
    large_items, large_keys, large_values = _roster(rng, N_LARGE, "l")

    small_pairs = [(small_keys[p], small_values[d]) for p, d in small_items]
    large_pairs = [(large_keys[p], large_values[d]) for p, d in large_items]
    w_small = hebbian_fit(small_pairs, DIM, STEPS_SMALL, LR, random.Random(14))
    w_large = hebbian_fit(large_pairs, DIM, STEPS_LARGE, LR, random.Random(15))
    small_hits = recall_hits(w_small, small_keys, small_values, small_items)
    large_hits = recall_hits(w_large, large_keys, large_values, large_items)

    # Compress: keep the six most "queried" people (first six by construction).
    selected = large_items[:N_SMALL]
    selected_pairs = [(large_keys[p], large_values[d]) for p, d in selected]
    w_pick = hebbian_fit(selected_pairs, DIM, STEPS_SMALL, LR, random.Random(16))
    selected_hits = recall_hits(w_pick, large_keys, large_values, selected)
    leftover = large_items[N_SMALL:]
    leftover_hits = recall_hits(w_pick, large_keys, large_values, leftover)

    small_rate = small_hits / N_SMALL
    large_rate = large_hits / N_LARGE
    checks = {
        "small_set_recalls": small_hits >= 5,
        "dumping_all_collides": large_rate <= 0.65 and large_rate < small_rate - 0.2,
        "selected_still_recalls": selected_hits >= 5,
        "unselected_mostly_gone": leftover_hits <= 8,
        "selection_beats_dump_on_kept": selected_hits >= 5,
    }
    return {
        "summary": (
            f"dim={DIM}。6 条巩固后召回 {small_hits}/6；"
            f"把 36 条全倒进同一张 W 只中 {large_hits}/36。"
            f"只巩固查询最多的 6 条，这 6 条中 {selected_hits}，其余 {leftover_hits}/30。"
            "失败阈值：小集合也召不回，或全倒进去并不比小集合差。"
        ),
        "metrics": {
            "dim": DIM,
            "n_small": N_SMALL,
            "n_large": N_LARGE,
            "small_hits": small_hits,
            "large_hits": large_hits,
            "small_rate": small_hits / N_SMALL,
            "large_rate": large_hits / N_LARGE,
            "selected_hits": selected_hits,
            "leftover_hits": leftover_hits,
        },
        "checks": checks,
    }


EXPERIMENT = ExtraExperiment(
    extra_id="capacity",
    title="容量：日记不能整本倒进权重",
    question="小矩阵装不下整个 Mem0 时，全倒进去会怎样？只巩固常问的呢？",
    lesson_hint="13,16",
    run=run,
)
