"""Skill cards vs compiling a procedure into a policy matrix."""

from __future__ import annotations

import random
from typing import Any

from ..extra_core import ExtraExperiment
from ..lin import hebbian_fit, matvec, nearest_name, unit

DIM = 20
STEPS = 600
LR = 0.2

SKILLS = {
    "board": ("goto_bench", "use_saw", "emit_plank"),
    "stick": ("goto_bench", "use_knife", "emit_stick"),
    "torch": ("take_plank", "take_stick", "bind"),
}


def run() -> dict[str, Any]:
    rng = random.Random(1)
    library = dict(SKILLS)
    cues = {name: unit(rng, DIM) for name in SKILLS}
    actions = sorted({step for steps in SKILLS.values() for step in steps})
    action_vec = {name: unit(rng, DIM) for name in actions}

    def _embed_skill(name: str) -> list[float]:
        total = [0.0] * DIM
        for step in SKILLS[name]:
            vec = action_vec[step]
            total = [a + b for a, b in zip(total, vec)]
        norm = sum(v * v for v in total) ** 0.5 or 1.0
        return [v / norm for v in total]

    skill_embeds = {name: _embed_skill(name) for name in SKILLS}
    pairs = [(cues[name], skill_embeds[name]) for name in SKILLS]
    policy = hebbian_fit(pairs, DIM, STEPS, LR, rng)

    def run_from_library(name: str, plugged: bool) -> tuple[str, ...] | None:
        if not plugged:
            return None
        return library.get(name)

    lib_plugged = all(run_from_library(name, True) == steps for name, steps in SKILLS.items())
    lib_unplugged = all(run_from_library(name, False) is None for name in SKILLS)
    compiled = {
        name: SKILLS[nearest_name(matvec(policy, cues[name]), skill_embeds)]
        for name in SKILLS
    }
    compiled_ok = all(compiled[name] == steps for name, steps in SKILLS.items())

    # Composition torch needs board+stick products. Library has a card. Weights
    # that only stored the three cards can still retrieve "torch" by cue.
    torch_from_lib = run_from_library("torch", True)
    torch_from_w = compiled["torch"]
    torch_lib_unplug = run_from_library("torch", False)

    checks = {
        "library_runs_when_plugged": lib_plugged,
        "library_empty_when_unplugged": lib_unplugged,
        "compiled_policy_matches_cards": compiled_ok,
        "torch_card_exists": torch_from_lib == SKILLS["torch"],
        "torch_survives_unplug_in_weights": (
            torch_from_w == SKILLS["torch"] and torch_lib_unplug is None
        ),
    }
    return {
        "summary": (
            "三张技能卡在库里时能逐步执行；卸掉库之后卡片路径全空。"
            f"把技能编码进 W 之后，不查库仍能取出 torch={compiled['torch']!r}。"
            "失败阈值：巩固后的策略对不上卡片，或卸库后权重也取不出。"
        ),
        "metrics": {
            "library_size": len(library),
            "compiled": {name: list(steps) for name, steps in compiled.items()},
            "lib_plugged": lib_plugged,
            "compiled_ok": compiled_ok,
        },
        "checks": checks,
    }


EXPERIMENT = ExtraExperiment(
    extra_id="skill",
    title="技能卡编译进策略",
    question="Voyager 式技能库卸掉之后，编译进权重的流程还能不能跑？",
    lesson_hint="16,21",
    run=run,
)
