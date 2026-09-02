"""External memory answers until you unplug the store."""

from __future__ import annotations

from typing import Any

from ..extra_core import ExtraExperiment

FACTS = {
    f"person_{index}": f"desk_{index}"
    for index in range(12)
}


def _lookup(store: dict[str, str], items: dict[str, str]) -> int:
    return sum(store.get(name) == seat for name, seat in items.items())


def run() -> dict[str, Any]:
    memory = dict(FACTS)
    prompt = dict(FACTS)
    weights: dict[str, str] = {}
    empty: dict[str, str] = {}

    memory_hits = _lookup(memory, FACTS)
    prompt_hits = _lookup(prompt, FACTS)
    unplug_hits = _lookup(empty, FACTS)
    weight_hits = _lookup(weights, FACTS)

    checks = {
        "memory_answers_all": memory_hits == len(FACTS),
        "prompt_answers_while_present": prompt_hits == len(FACTS),
        "unplug_memory_scores_zero": unplug_hits == 0,
        "untrained_weights_score_zero": weight_hits == 0,
        "unplug_is_worse_than_memory": unplug_hits < memory_hits,
    }
    return {
        "summary": (
            f"12 条座位在外挂记忆和上下文里都能全对；拔掉库之后 {unplug_hits}/12；"
            f"从未写入的权重 {weight_hits}/12。失败阈值：拔库后仍能答对。"
        ),
        "metrics": {
            "n_facts": len(FACTS),
            "memory_hits": memory_hits,
            "prompt_hits": prompt_hits,
            "unplug_hits": unplug_hits,
            "weight_hits": weight_hits,
        },
        "checks": checks,
    }


EXPERIMENT = ExtraExperiment(
    extra_id="unplug",
    title="拔掉外挂记忆",
    question="日记会了，把库卸掉之后还会不会？",
    lesson_hint="04,13,16",
    run=run,
)
