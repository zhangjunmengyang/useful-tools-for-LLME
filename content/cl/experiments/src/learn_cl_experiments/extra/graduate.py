"""Graduation: unplug diary, skill cards, and the prompt roster. Weights must still work."""

from __future__ import annotations

import random
from typing import Any

from ..extra_core import ExtraExperiment
from ..lin import hebbian_fit, linear_fit, linear_mae, matvec, nearest_name, recall_hits, unit

DIM = 22
N_FACTS = 8
FACT_STEPS = 700
SKILL_STEPS = 600
LR = 0.22

SKILLS = {
    "board": ("goto_bench", "use_saw", "emit_plank"),
    "torch": ("take_plank", "bind"),
}


def _rule(a: float, b: float) -> float:
    return 2.0 * a + 3.0 * b


def run() -> dict[str, Any]:
    rng = random.Random(71)
    people = [f"person_{i}" for i in range(N_FACTS)]
    desks = [f"desk_{i}" for i in range(N_FACTS)]
    items = list(zip(people, desks))
    keys = {name: unit(rng, DIM) for name in people}
    values = {name: unit(rng, DIM) for name in desks}

    diary = {person: desk for person, desk in items}
    prompt = dict(diary)
    library = dict(SKILLS)

    fact_pairs = [(keys[p], values[d]) for p, d in items]
    w_facts = hebbian_fit(fact_pairs, DIM, FACT_STEPS, LR, random.Random(72))

    cues = {name: unit(rng, DIM) for name in SKILLS}
    actions = sorted({step for steps in SKILLS.values() for step in steps})
    action_vec = {name: unit(rng, DIM) for name in actions}

    def embed(name: str) -> list[float]:
        total = [0.0] * DIM
        for step in SKILLS[name]:
            vec = action_vec[step]
            total = [a + b for a, b in zip(total, vec)]
        norm = sum(v * v for v in total) ** 0.5 or 1.0
        return [v / norm for v in total]

    skill_embeds = {name: embed(name) for name in SKILLS}
    w_skill = hebbian_fit(
        [(cues[name], skill_embeds[name]) for name in SKILLS],
        DIM,
        SKILL_STEPS,
        LR,
        random.Random(73),
    )
    clean = []
    fit_rng = random.Random(74)
    for _ in range(24):
        a, b = fit_rng.uniform(-1.5, 1.5), fit_rng.uniform(-1.5, 1.5)
        clean.append(((a, b), _rule(a, b)))
    w_rule = linear_fit(clean, steps=250, lr=0.08, rng=random.Random(74))
    probes = [(float(i) * 0.35, float(i) * -0.2) for i in range(10)]

    before_facts = sum(diary[p] == d for p, d in items)
    before_skill = library.get("torch") == SKILLS["torch"]
    before_prompt = all(prompt[p] == d for p, d in items)

    diary.clear()
    library.clear()
    prompt.clear()

    unplug_diary = sum(diary.get(p) == d for p, d in items)
    unplug_prompt = sum(prompt.get(p) == d for p, d in items)
    unplug_library = library.get("torch")
    weight_facts = recall_hits(w_facts, keys, values, items)
    compiled = SKILLS[nearest_name(matvec(w_skill, cues["torch"]), skill_embeds)]
    rule_mae = linear_mae(w_rule, probes, _rule)

    checks = {
        "plugged_stack_works": before_facts == N_FACTS and before_skill and before_prompt,
        "diary_gone": unplug_diary == 0,
        "prompt_gone": unplug_prompt == 0,
        "library_gone": unplug_library is None,
        "weights_still_call_people": weight_facts >= 6,
        "weights_still_run_torch": compiled == SKILLS["torch"],
        "weights_still_score": rule_mae < 0.35,
    }
    return {
        "summary": (
            f"卸掉日记、提示名录和技能卡之后：查库 {unplug_diary}/{N_FACTS}，"
            f"卡片 {unplug_library!r}。"
            f"权重仍能叫人 {weight_facts}/{N_FACTS}，取出 torch={compiled!r}，"
            f"计分误差 {rule_mae:.3f}。"
            "失败阈值：外挂空了之后权重也不会。"
        ),
        "metrics": {
            "before_facts": before_facts,
            "unplug_diary": unplug_diary,
            "unplug_prompt": unplug_prompt,
            "weight_facts": weight_facts,
            "compiled": list(compiled),
            "rule_mae": rule_mae,
        },
        "checks": checks,
    }


EXPERIMENT = ExtraExperiment(
    extra_id="graduate",
    title="毕业卸库：日记、技能卡、提示词全拔",
    question="三件外挂都卸掉之后，权重里的座位、流程和计分规则还在不在？",
    lesson_hint="16,21,24",
    run=run,
)
