"""A replay buffer still is a plugin. Unplug it, then learn the next batch, old facts go."""

from __future__ import annotations

import random
from typing import Any

from ..extra_core import ExtraExperiment
from ..lin import hebbian_fit, make_roster, overlap_keys, recall_hits

DIM = 18
N = 6
STEPS = 520
LR = 0.22


def run() -> dict[str, Any]:
    rng = random.Random(81)
    a_items, keys, values, a_pairs = make_roster(rng, N, DIM, "a")
    b_items, b_keys, b_values, b_pairs = make_roster(rng, N, DIM, "b")
    c_items, c_keys, c_values, c_pairs = make_roster(rng, N, DIM, "c")
    keys.update(b_keys)
    keys.update(c_keys)
    values.update(b_values)
    values.update(c_values)
    overlap_keys(keys, [p for p, _ in a_items], [p for p, _ in b_items], mix=0.6)
    overlap_keys(keys, [p for p, _ in a_items], [p for p, _ in c_items], mix=0.55)
    b_pairs = [(keys[p], values[d]) for p, d in b_items]
    c_pairs = [(keys[p], values[d]) for p, d in c_items]

    w_a = hebbian_fit(a_pairs, DIM, STEPS, LR, random.Random(82))
    a_after_a = recall_hits(w_a, keys, values, a_items)

    w_naive_b = hebbian_fit(b_pairs, DIM, STEPS, LR, random.Random(83), start=w_a)
    naive_a = recall_hits(w_naive_b, keys, values, a_items)
    naive_b = recall_hits(w_naive_b, keys, values, b_items)

    buffer_a = list(a_pairs)
    w_buf_b = hebbian_fit(
        b_pairs + buffer_a,
        DIM,
        STEPS,
        LR,
        random.Random(84),
        start=w_a,
    )
    buf_a = recall_hits(w_buf_b, keys, values, a_items)
    buf_b = recall_hits(w_buf_b, keys, values, b_items)

    # Product trap: after A+B are in W, throw the buffer away, then learn C.
    w_unplug_c = hebbian_fit(c_pairs, DIM, STEPS, LR, random.Random(85), start=w_buf_b)
    unplug_a = recall_hits(w_unplug_c, keys, values, a_items)
    unplug_b = recall_hits(w_unplug_c, keys, values, b_items)
    unplug_c = recall_hits(w_unplug_c, keys, values, c_items)

    buffer_ab = a_pairs + b_pairs
    w_keep_c = hebbian_fit(
        c_pairs + buffer_ab,
        DIM,
        STEPS,
        LR,
        random.Random(86),
        start=w_buf_b,
    )
    keep_a = recall_hits(w_keep_c, keys, values, a_items)
    keep_b = recall_hits(w_keep_c, keys, values, b_items)
    keep_c = recall_hits(w_keep_c, keys, values, c_items)

    checks = {
        "a_was_in_weights": a_after_a >= 5,
        "naive_b_forgets_a": naive_a <= 2,
        "naive_b_learns_b": naive_b >= 3,
        "buffer_keeps_a_while_learning_b": buf_a >= 5 and buf_b >= 4,
        "unplug_buffer_then_c_forgets_old": unplug_a + unplug_b <= 4,
        "kept_buffer_survives_c": keep_a >= 4 and keep_b >= 4 and keep_c >= 3,
    }
    return {
        "summary": (
            f"A 写入后 {a_after_a}/{N}。接着只练 B：A 剩 {naive_a}。"
            f"带着 A 的回放缓冲练 B：A {buf_a}、B {buf_b}。"
            f"卸掉缓冲再练 C：A {unplug_a}、B {unplug_b}、C {unplug_c}。"
            f"缓冲里还留着 A+B 再练 C：A {keep_a}、B {keep_b}、C {keep_c}。"
            "失败阈值：带着缓冲也不如 naive，或卸掉缓冲接着学仍不忘旧的。"
        ),
        "metrics": {
            "a_after_a": a_after_a,
            "naive_a": naive_a,
            "naive_b": naive_b,
            "buf_a": buf_a,
            "buf_b": buf_b,
            "unplug_a": unplug_a,
            "unplug_b": unplug_b,
            "unplug_c": unplug_c,
            "keep_a": keep_a,
            "keep_b": keep_b,
            "keep_c": keep_c,
            "buffer_size": len(buffer_ab),
        },
        "checks": checks,
    }


EXPERIMENT = ExtraExperiment(
    extra_id="buffer",
    title="回放缓冲仍是外挂",
    question="DER / iCaRL 留下的那袋旧样本，卸掉之后再学新的，旧事实还在吗？",
    lesson_hint="06,08,16",
    run=run,
)
