"""Prompt + weights can look perfect while the weights still know nothing."""

from __future__ import annotations

import random
from typing import Any

from ..extra_core import ExtraExperiment
from ..lin import hebbian_fit, make_roster, recall_hits

DIM = 22
N = 10
STEPS = 700
LR = 0.24


def _combined(
    prompt: dict[str, str],
    weight_hits: int,
    items: list[tuple[str, str]],
) -> int:
    """Use the prompt when it still has the name, else fall back to weights."""
    hits = 0
    for person, desk in items:
        if prompt.get(person) == desk:
            hits += 1
        elif prompt.get(person) is None:
            # weights-only path is counted outside; here prompt is empty.
            hits += 0
        else:
            hits += 0
    if not prompt:
        return weight_hits
    return hits


def run() -> dict[str, Any]:
    rng = random.Random(111)
    items, keys, values, pairs = make_roster(rng, N, DIM, "h")
    prompt = {person: desk for person, desk in items}

    w_empty = hebbian_fit([], DIM, 0, LR, rng)
    empty_hits = recall_hits(w_empty, keys, values, items)
    combined_untrained = _combined(prompt, empty_hits, items)

    w_trained = hebbian_fit(pairs, DIM, STEPS, LR, random.Random(112))
    trained_hits = recall_hits(w_trained, keys, values, items)
    combined_trained = _combined(prompt, trained_hits, items)

    prompt_unplugged: dict[str, str] = {}
    combined_untrained_unplug = _combined(prompt_unplugged, empty_hits, items)
    combined_trained_unplug = _combined(prompt_unplugged, trained_hits, items)
    prompt_only_unplug = sum(prompt_unplugged.get(p) == d for p, d in items)

    checks = {
        "untrained_weights_miss": empty_hits <= 2,
        "prompt_plus_empty_weights_looks_perfect": combined_untrained == N,
        "trained_weights_work_alone": trained_hits >= 8,
        "combined_score_cannot_tell_them_apart": combined_trained == combined_untrained,
        "unplug_prompt_exposes_untrained": combined_untrained_unplug <= 2,
        "unplug_prompt_keeps_trained": combined_trained_unplug >= 8,
        "prompt_alone_dies_when_unplugged": prompt_only_unplug == 0,
    }
    return {
        "summary": (
            f"提示名录还在时：未训权重 {empty_hits}/{N}，合起来却 {combined_untrained}/{N}；"
            f"训过的权重单独 {trained_hits}/{N}，合起来也 {combined_trained}/{N}。"
            f"卸掉提示：未训 {combined_untrained_unplug}/{N}，训过 {combined_trained_unplug}/{N}。"
            "失败阈值：合起来的分数能把未训和已训分开，或卸掉提示后训过的也不会。"
        ),
        "metrics": {
            "empty_hits": empty_hits,
            "trained_hits": trained_hits,
            "combined_untrained": combined_untrained,
            "combined_trained": combined_trained,
            "combined_untrained_unplug": combined_untrained_unplug,
            "combined_trained_unplug": combined_trained_unplug,
        },
        "checks": checks,
    }


EXPERIMENT = ExtraExperiment(
    extra_id="shadow",
    title="影子读取：提示还在时看不出权重会不会",
    question="产品把日记塞进 prompt 再问模型，这个分数能证明已经写进权重了吗？",
    lesson_hint="04,13,16",
    run=run,
)
