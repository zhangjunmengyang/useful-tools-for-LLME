"""As columns of W get occupied, later batches have less room and learn worse."""

from __future__ import annotations

import random
from typing import Any

from ..extra_core import ExtraExperiment
from ..lin import add_outer, copy_mat, hebbian_fit, l2, matvec, recall_hits, sub, unit, zeros

DIM = 10
N_TASKS = 8
N_PER = 4
STEPS = 360
LR = 0.25


def _batch(rng: random.Random, tag: str) -> tuple[
    list[tuple[str, str]],
    dict[str, list[float]],
    dict[str, list[float]],
]:
    people = [f"{tag}_p{i}" for i in range(N_PER)]
    desks = [f"{tag}_d{i}" for i in range(N_PER)]
    items = list(zip(people, desks))
    keys = {name: unit(rng, DIM) for name in people}
    values = {name: unit(rng, DIM) for name in desks}
    return items, keys, values


def _column_norms(weights: list[list[float]]) -> list[float]:
    return [
        l2([weights[row][col] for row in range(len(weights))])
        for col in range(len(weights[0]))
    ]


def _fit_unfrozen(
    start: list[list[float]],
    pairs: list[tuple[list[float], list[float]]],
    frozen: set[int],
    steps: int,
    lr: float,
    rng: random.Random,
) -> list[list[float]]:
    weights = copy_mat(start)
    if not pairs:
        return weights
    for _ in range(steps):
        key, value = pairs[rng.randrange(len(pairs))]
        masked = [0.0 if index in frozen else coord for index, coord in enumerate(key)]
        residual = sub(value, matvec(weights, key))
        add_outer(weights, residual, masked, lr)
    return weights


def run() -> dict[str, Any]:
    rng = random.Random(51)
    tasks = [_batch(rng, f"t{index}") for index in range(N_TASKS)]
    sequential = zeros(DIM, DIM)
    frozen: set[int] = set()
    seq_latest: list[int] = []
    fresh_latest: list[int] = []
    frozen_counts: list[int] = []

    for index, (items, keys, values) in enumerate(tasks):
        pairs = [(keys[p], values[d]) for p, d in items]
        sequential = _fit_unfrozen(
            sequential,
            pairs,
            frozen,
            STEPS,
            LR,
            random.Random(80 + index),
        )
        seq_latest.append(recall_hits(sequential, keys, values, items))
        fresh = hebbian_fit(pairs, DIM, STEPS, LR, random.Random(90 + index))
        fresh_latest.append(recall_hits(fresh, keys, values, items))
        # Occupy the strongest still-free column, like a unit that has been used up.
        norms = _column_norms(sequential)
        free = [col for col in range(DIM) if col not in frozen]
        if free:
            frozen.add(max(free, key=lambda col: norms[col]))
        frozen_counts.append(len(frozen))

    early_seq = sum(seq_latest[:2]) / 2.0
    late_seq = sum(seq_latest[-2:]) / 2.0
    late_fresh = sum(fresh_latest[-2:]) / 2.0

    checks = {
        "early_tasks_learn": early_seq >= 3.0,
        "later_tasks_learn_less": late_seq <= early_seq - 1.0,
        "late_worse_than_fresh": late_seq <= late_fresh - 1.0,
        "frozen_columns_rise": frozen_counts[-1] >= frozen_counts[0] + 4,
        "fresh_still_learns_late": late_fresh >= 3.0,
    }
    return {
        "summary": (
            f"每学完一批就占用一列。前两批召回 {early_seq:.1f}，后两批 {late_seq:.1f}；"
            f"空白矩阵后两批仍是 {late_fresh:.1f}。占用列从 {frozen_counts[0]} 到 {frozen_counts[-1]}。"
            "失败阈值：列占满之后新批次仍和一开始一样好学。"
        ),
        "metrics": {
            "seq_latest": seq_latest,
            "fresh_latest": fresh_latest,
            "frozen_counts": frozen_counts,
            "early_seq": early_seq,
            "late_seq": late_seq,
            "late_fresh": late_fresh,
        },
        "checks": checks,
    }


EXPERIMENT = ExtraExperiment(
    extra_id="plastic",
    title="连续写入之后学新的变慢",
    question="Agent 连续多天改自己的权重，后面还学得动吗？",
    lesson_hint="15,23",
    run=run,
)
