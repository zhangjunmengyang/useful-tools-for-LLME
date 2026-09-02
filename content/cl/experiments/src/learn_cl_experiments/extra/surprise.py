"""Surprise-gated writes: rare facts keep a foothold; constant-lr streams bury them."""

from __future__ import annotations

import random
from typing import Any

from ..extra_core import ExtraExperiment
from ..lin import hebbian_stream, recall_hits, unit

DIM = 14
N_COMMON = 6
N_RARE = 6
COMMON_REPS = 18
LR = 0.18


def run() -> dict[str, Any]:
    rng = random.Random(21)
    common_people = [f"common_{i}" for i in range(N_COMMON)]
    rare_people = [f"rare_{i}" for i in range(N_RARE)]
    common_desks = [f"cdesk_{i}" for i in range(N_COMMON)]
    rare_desks = [f"rdesk_{i}" for i in range(N_RARE)]
    common_items = list(zip(common_people, common_desks))
    rare_items = list(zip(rare_people, rare_desks))
    keys = {name: unit(rng, DIM) for name in common_people + rare_people}
    values = {name: unit(rng, DIM) for name in common_desks + rare_desks}

    stream: list[tuple[list[float], list[float]]] = []
    for _ in range(COMMON_REPS):
        for person, desk in common_items:
            stream.append((keys[person], values[desk]))
    # Rare facts arrive after the common stream has already occupied W,
    # matching "a new seat shows up once in a long diary".
    for person, desk in rare_items:
        stream.append((keys[person], values[desk]))

    w_const = hebbian_stream(stream, DIM, LR, gate="constant")
    w_gate = hebbian_stream(stream, DIM, LR, gate="surprise")

    const_common = recall_hits(w_const, keys, values, common_items)
    const_rare = recall_hits(w_const, keys, values, rare_items)
    gate_common = recall_hits(w_gate, keys, values, common_items)
    gate_rare = recall_hits(w_gate, keys, values, rare_items)

    checks = {
        "constant_learns_common": const_common >= 4,
        "constant_buries_rare": const_rare <= 3,
        "surprise_keeps_rare": gate_rare >= const_rare + 1,
        "surprise_keeps_common": gate_common >= 3,
        "rare_gap_is_the_point": gate_rare > const_rare,
    }
    return {
        "summary": (
            f"常见座位出现 {COMMON_REPS} 次、稀有 1 次。"
            f"固定步长：常见 {const_common}/{N_COMMON}，稀有 {const_rare}/{N_RARE}。"
            f"按残差门控：常见 {gate_common}/{N_COMMON}，稀有 {gate_rare}/{N_RARE}。"
            "失败阈值：门控后稀有并不比固定步长好。"
        ),
        "metrics": {
            "const_common": const_common,
            "const_rare": const_rare,
            "gate_common": gate_common,
            "gate_rare": gate_rare,
            "stream_len": len(stream),
        },
        "checks": checks,
    }


EXPERIMENT = ExtraExperiment(
    extra_id="surprise",
    title="惊讶门：稀有事实才该大力写",
    question="日记流里天天重复的句子，会不会把只出现一次的座位冲掉？",
    lesson_hint="18,16",
    run=run,
)
