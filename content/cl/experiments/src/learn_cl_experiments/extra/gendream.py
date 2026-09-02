"""Generative replay: match the old map on random probes. No stored keys, no diary."""

from __future__ import annotations

import random
from typing import Any

from ..extra_core import ExtraExperiment
from ..lin import hebbian_fit, hebbian_lwf, make_roster, overlap_keys, recall_hits

DIM = 18
N = 6
STEPS = 560
LR = 0.2
LWF_LR = 0.14


def run() -> dict[str, Any]:
    rng = random.Random(91)
    a_items, keys, values, a_pairs = make_roster(rng, N, DIM, "a")
    b_items, b_keys, b_values, _b_pairs = make_roster(rng, N, DIM, "b")
    keys.update(b_keys)
    values.update(b_values)
    overlap_keys(keys, [p for p, _ in a_items], [p for p, _ in b_items], mix=0.58)
    b_pairs = [(keys[p], values[d]) for p, d in b_items]

    w_a = hebbian_fit(a_pairs, DIM, STEPS, LR, random.Random(92))
    a_before = recall_hits(w_a, keys, values, a_items)

    w_naive = hebbian_fit(b_pairs, DIM, STEPS, LR, random.Random(93), start=w_a)
    naive_a = recall_hits(w_naive, keys, values, a_items)
    naive_b = recall_hits(w_naive, keys, values, b_items)

    # Dream: no A keys stored. Random probes ask the frozen teacher W_A.
    w_dream = hebbian_lwf(
        w_a,
        b_pairs,
        w_a,
        DIM,
        STEPS,
        lr=0.16,
        lwf_lr=LWF_LR,
        rng=random.Random(94),
        n_probes=3,
    )
    dream_a = recall_hits(w_dream, keys, values, a_items)
    dream_b = recall_hits(w_dream, keys, values, b_items)

    # Control: a real buffer of A keys. Upper bound, but it is still a plugin.
    w_buf = hebbian_fit(
        b_pairs + a_pairs,
        DIM,
        STEPS,
        LR,
        random.Random(95),
        start=w_a,
    )
    buf_a = recall_hits(w_buf, keys, values, a_items)
    buf_b = recall_hits(w_buf, keys, values, b_items)

    checks = {
        "a_was_in_weights": a_before >= 5,
        "naive_forgets_a": naive_a <= 2,
        "naive_learns_b": naive_b >= 3,
        "dream_keeps_more_a_than_naive": dream_a >= naive_a + 2,
        "dream_still_learns_b": dream_b >= 3,
        "buffer_is_stronger_but_is_a_plugin": buf_a >= dream_a,
        "dream_uses_no_stored_a_keys": True,
    }
    return {
        "summary": (
            f"A 在权重里 {a_before}/{N}。只练 B：A 剩 {naive_a}。"
            f"学 B 时用旧 W 在随机探针上当老师（不存 A 的键）：A {dream_a}、B {dream_b}。"
            f"真把 A 的键留下来回放：A {buf_a}、B {buf_b}。缓冲更稳，但那还是外挂。"
            "失败阈值：做梦并不比 naive 更能保住 A。"
        ),
        "metrics": {
            "a_before": a_before,
            "naive_a": naive_a,
            "naive_b": naive_b,
            "dream_a": dream_a,
            "dream_b": dream_b,
            "buf_a": buf_a,
            "buf_b": buf_b,
        },
        "checks": checks,
    }


EXPERIMENT = ExtraExperiment(
    extra_id="gendream",
    title="生成回放：不存旧键也能护住旧映射",
    question="卸掉日记之后还要接着学，能不能让旧权重自己出题给自己练，而不是再藏一袋样本？",
    lesson_hint="06,16,19",
    run=run,
)
