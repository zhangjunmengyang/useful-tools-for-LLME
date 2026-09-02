"""After a roster is in W, new seats with an EWC-style penalty keep more of the old roster."""

from __future__ import annotations

import random
from typing import Any

from ..extra_core import ExtraExperiment
from ..lin import (
    add_outer,
    copy_mat,
    hebbian_fit,
    matvec,
    orthonormalize,
    project_out,
    recall_hits,
    sub,
    unit,
    zeros,
)

DIM = 16
N_OLD = 6
N_NEW = 6
STEPS = 480
LR = 0.22


def _protect_fit(
    start: list[list[float]],
    pairs: list[tuple[list[float], list[float]]],
    old_keys: list[list[float]],
    steps: int,
    lr: float,
    rng: random.Random,
) -> list[list[float]]:
    """Update only in the complement of old keys — GEM / O-LoRA geometry."""
    weights = copy_mat(start)
    basis = orthonormalize(old_keys)
    if not pairs:
        return weights
    for _ in range(steps):
        key, value = pairs[rng.randrange(len(pairs))]
        residual = sub(value, matvec(weights, key))
        free = project_out(key, basis)
        add_outer(weights, residual, free, lr)
    return weights


def run() -> dict[str, Any]:
    rng = random.Random(41)
    old_people = [f"old_{i}" for i in range(N_OLD)]
    new_people = [f"new_{i}" for i in range(N_NEW)]
    old_desks = [f"odesk_{i}" for i in range(N_OLD)]
    new_desks = [f"ndesk_{i}" for i in range(N_NEW)]
    old_items = list(zip(old_people, old_desks))
    new_items = list(zip(new_people, new_desks))
    keys = {name: unit(rng, DIM) for name in old_people + new_people}
    values = {name: unit(rng, DIM) for name in old_desks + new_desks}
    # New seats sit in the same subspace as old ones, otherwise dim=14 still
    # has room and naive will not forget.
    for index in range(N_NEW):
        old_key = keys[old_people[index]]
        keys[new_people[index]] = [
            0.55 * old_key[j] + 0.45 * keys[new_people[index]][j]
            for j in range(DIM)
        ]

    old_pairs = [(keys[p], values[d]) for p, d in old_items]
    new_pairs = [(keys[p], values[d]) for p, d in new_items]
    w_old = hebbian_fit(old_pairs, DIM, STEPS, LR, random.Random(42))
    old_hits = recall_hits(w_old, keys, values, old_items)

    w_naive = hebbian_fit(new_pairs, DIM, STEPS, LR, random.Random(43), start=w_old)
    naive_old = recall_hits(w_naive, keys, values, old_items)
    naive_new = recall_hits(w_naive, keys, values, new_items)

    w_ewc = _protect_fit(
        w_old,
        new_pairs,
        [keys[p] for p, _d in old_items],
        STEPS,
        LR,
        random.Random(44),
    )
    ewc_old = recall_hits(w_ewc, keys, values, old_items)
    ewc_new = recall_hits(w_ewc, keys, values, new_items)

    checks = {
        "old_roster_was_in_weights": old_hits >= 5,
        "naive_forgets_old": naive_old <= 3,
        "naive_learns_new": naive_new >= 2,
        "ewc_keeps_more_old": ewc_old >= naive_old + 1,
        "ewc_still_learns_some_new": ewc_new >= 1,
        "unused_zero_init_empty": recall_hits(zeros(DIM, DIM), keys, values, old_items) <= 1,
    }
    return {
        "summary": (
            f"旧花名册写入后 {old_hits}/{N_OLD}。"
            f"接着只练新座位：旧的剩 {naive_old}，新的 {naive_new}/{N_NEW}。"
            f"新座位只写在旧键的正交补上：旧的 {ewc_old}，新的 {ewc_new}。"
            "失败阈值：护住旧方向并不比 naive 更能保住花名册。"
        ),
        "metrics": {
            "old_hits": old_hits,
            "naive_old": naive_old,
            "naive_new": naive_new,
            "ewc_old": ewc_old,
            "ewc_new": ewc_new,
        },
        "checks": checks,
    }


EXPERIMENT = ExtraExperiment(
    extra_id="ewcmem",
    title="巩固后再学新座位：护住旧方向",
    question="夜间已经写入权重的座位，第二天继续微调时，不在旧键方向上写，能不能少冲一点？",
    lesson_hint="05,13,16",
    run=run,
)
