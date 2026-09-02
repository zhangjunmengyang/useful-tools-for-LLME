"""Route each experience: context, diary, or weights. Unplug shows what survives."""

from __future__ import annotations

import random
from typing import Any

from ..extra_core import ExtraExperiment
from ..lin import hebbian_fit, linear_fit, linear_mae, recall_hits, unit

DIM = 20
N_FACTS = 8
STEPS = 600
LR = 0.25


def _rule(a: float, b: float) -> float:
    return 2.0 * a + 3.0 * b


def run() -> dict[str, Any]:
    rng = random.Random(11)
    chatter = {f"chat_{i}": f"smalltalk_{i}" for i in range(4)}
    one_off = {f"session_{i}": f"tmp_{i}" for i in range(3)}
    people = [f"person_{i}" for i in range(N_FACTS)]
    desks = [f"desk_{i}" for i in range(N_FACTS)]
    items = list(zip(people, desks))
    keys = {name: unit(rng, DIM) for name in people}
    values = {name: unit(rng, DIM) for name in desks}

    context = dict(chatter)
    context.update(one_off)
    diary = {person: desk for person, desk in items}

    def _lookup(store: dict[str, str], catalog: dict[str, str]) -> int:
        return sum(store.get(name) == value for name, value in catalog.items())

    plugged_chatter = _lookup(context, chatter)
    plugged_facts = _lookup(diary, dict(items))

    pairs = [(keys[person], values[desk]) for person, desk in items]
    weights = hebbian_fit(pairs, DIM, STEPS, LR, rng)
    clean_rule = []
    fit_rng = random.Random(12)
    for _ in range(24):
        a, b = fit_rng.uniform(-1.5, 1.5), fit_rng.uniform(-1.5, 1.5)
        clean_rule.append(((a, b), _rule(a, b)))
    rule = linear_fit(clean_rule, steps=250, lr=0.08, rng=random.Random(12))
    probes = [(float(i) * 0.4, float(i) * -0.15) for i in range(10)]

    context.clear()
    diary.clear()

    unplug_chatter = _lookup(context, chatter)
    unplug_one_off = _lookup(context, one_off)
    unplug_diary_facts = _lookup(diary, dict(items))
    unplug_weight_facts = recall_hits(weights, keys, values, items)
    rule_mae = linear_mae(rule, probes, _rule)

    checks = {
        "chatter_answers_while_plugged": plugged_chatter == len(chatter),
        "facts_in_diary_while_plugged": plugged_facts == N_FACTS,
        "chatter_dies_after_unplug": unplug_chatter == 0,
        "one_off_never_survives_unplug": unplug_one_off == 0,
        "diary_empty_after_unplug": unplug_diary_facts == 0,
        "distilled_facts_survive": unplug_weight_facts >= 6,
        "procedure_lives_in_weights": rule_mae < 0.35,
    }
    return {
        "summary": (
            f"闲聊和一次性会话卸掉上下文后 {unplug_chatter}/{len(chatter)}、"
            f"{unplug_one_off}/{len(one_off)}。"
            f"日记清空后查库 {unplug_diary_facts}/{N_FACTS}，"
            f"巩固进 W 仍中 {unplug_weight_facts}/{N_FACTS}。"
            f"计分规则在权重里误差 {rule_mae:.3f}。"
            "失败阈值：闲聊卸库后仍能答，或座位没进权重。"
        ),
        "metrics": {
            "plugged_chatter": plugged_chatter,
            "plugged_facts": plugged_facts,
            "unplug_chatter": unplug_chatter,
            "unplug_one_off": unplug_one_off,
            "unplug_diary_facts": unplug_diary_facts,
            "unplug_weight_facts": unplug_weight_facts,
            "rule_mae": rule_mae,
        },
        "checks": checks,
    }


EXPERIMENT = ExtraExperiment(
    extra_id="route",
    title="分流：写上下文、日记还是权重",
    question="闲聊、座位、计分规则，卸掉外挂之后各还剩什么？",
    lesson_hint="04,13,16",
    run=run,
)
