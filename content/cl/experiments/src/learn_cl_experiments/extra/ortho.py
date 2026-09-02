"""Orthogonal keys keep old facts when a new batch is written; overlapping keys do not."""

from __future__ import annotations

import random
from typing import Any

from ..extra_core import ExtraExperiment
from ..lin import dot, hebbian_fit, l2, orthonormalize, recall_hits, scale, sub, unit

DIM = 24
N = 5
STEPS = 500
LR = 0.2


def run() -> dict[str, Any]:
    rng = random.Random(34)
    a_people = [f"a_p{i}" for i in range(N)]
    b_people = [f"b_p{i}" for i in range(N)]
    a_desks = [f"a_d{i}" for i in range(N)]
    b_desks = [f"b_d{i}" for i in range(N)]
    a_items = list(zip(a_people, a_desks))
    b_items = list(zip(b_people, b_desks))
    values = {name: unit(rng, DIM) for name in a_desks + b_desks}

    a_keys = {name: unit(rng, DIM) for name in a_people}
    overlap_keys = {name: unit(rng, DIM) for name in b_people}
    # Force overlap: B keys sit close to A keys, the product-side analogue of
    # writing a new skill into the same subspace.
    overlap_keys = {
        b_people[i]: [
            0.82 * a_keys[a_people[i]][j] + 0.18 * overlap_keys[b_people[i]][j]
            for j in range(DIM)
        ]
        for i in range(N)
    }

    raw_b = [unit(rng, DIM) for _ in range(N)]
    a_basis = orthonormalize(list(a_keys.values()))
    ortho_raw = []
    for vector in raw_b:
        leftover = vector[:]
        for axis in a_basis:
            leftover = sub(leftover, scale(axis, dot(leftover, axis)))
        leftover = scale(leftover, 1.0 / (l2(leftover) or 1.0))
        ortho_raw.append(leftover)
    ortho_keys = {b_people[i]: ortho_raw[i] for i in range(N)}

    a_pairs = [(a_keys[p], values[d]) for p, d in a_items]
    w_a = hebbian_fit(a_pairs, DIM, STEPS, LR, random.Random(35))

    overlap_pairs = [(overlap_keys[p], values[d]) for p, d in b_items]
    w_overlap = hebbian_fit(overlap_pairs, DIM, STEPS, LR, random.Random(36), start=w_a)
    ortho_pairs = [(ortho_keys[p], values[d]) for p, d in b_items]
    w_ortho = hebbian_fit(ortho_pairs, DIM, STEPS, LR, random.Random(37), start=w_a)

    a_after_overlap = recall_hits(w_overlap, a_keys, values, a_items)
    b_after_overlap = recall_hits(w_overlap, overlap_keys, values, b_items)
    a_after_ortho = recall_hits(w_ortho, a_keys, values, a_items)
    b_after_ortho = recall_hits(w_ortho, ortho_keys, values, b_items)

    checks = {
        "overlap_learns_b": b_after_overlap >= 3,
        "overlap_hurts_a": a_after_overlap <= 2,
        "ortho_keeps_a": a_after_ortho >= 4,
        "ortho_learns_b": b_after_ortho >= 3,
        "ortho_saves_more_of_a": a_after_ortho >= a_after_overlap + 2,
    }
    return {
        "summary": (
            f"新技能写进旧技能的子空间：A 只剩 {a_after_overlap}/{N}，B 中 {b_after_overlap}/{N}。"
            f"B 的键与 A 正交：A 仍 {a_after_ortho}/{N}，B {b_after_ortho}/{N}。"
            "失败阈值：正交写入仍冲掉 A，或重叠写入并不伤 A。"
        ),
        "metrics": {
            "a_after_overlap": a_after_overlap,
            "b_after_overlap": b_after_overlap,
            "a_after_ortho": a_after_ortho,
            "b_after_ortho": b_after_ortho,
        },
        "checks": checks,
    }


EXPERIMENT = ExtraExperiment(
    extra_id="ortho",
    title="正交子空间才留得住旧技能",
    question="第二块 LoRA 若和第一块抢同一方向，旧技能会不会没？",
    lesson_hint="11,21",
    run=run,
)
