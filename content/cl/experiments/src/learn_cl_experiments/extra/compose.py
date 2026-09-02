"""Compose two compiled skills without a card for the composition sitting in the library."""

from __future__ import annotations

import random
from typing import Any

from ..extra_core import ExtraExperiment
from ..lin import add, hebbian_fit, l2, matvec, nearest_name, scale, unit

DIM = 22
STEPS = 700
LR = 0.22

SKILLS = {
    "board": ("goto_bench", "use_saw", "emit_plank"),
    "stick": ("goto_bench", "use_knife", "emit_stick"),
}
LIGHT_STEPS = ("take_plank", "take_stick", "bind")


def _embed(steps: tuple[str, ...], action_vec: dict[str, list[float]]) -> list[float]:
    total = [0.0] * DIM
    for step in steps:
        total = add(total, action_vec[step])
    norm = l2(total) or 1.0
    return scale(total, 1.0 / norm)


def run() -> dict[str, Any]:
    rng = random.Random(161)
    library = dict(SKILLS)
    actions = sorted({step for steps in SKILLS.values() for step in steps} | set(LIGHT_STEPS))
    action_vec = {name: unit(rng, DIM) for name in actions}
    cues = {name: unit(rng, DIM) for name in ("board", "stick", "light")}

    skill_embeds = {
        "board": _embed(SKILLS["board"], action_vec),
        "stick": _embed(SKILLS["stick"], action_vec),
        "light": _embed(LIGHT_STEPS, action_vec),
    }
    # Library has no "light" card. Train the composition as board+stick products.
    composed = _embed(SKILLS["board"] + SKILLS["stick"], action_vec)
    # The light cue should retrieve the torch procedure, not a stored card.
    pairs = [
        (cues["board"], skill_embeds["board"]),
        (cues["stick"], skill_embeds["stick"]),
        (cues["light"], skill_embeds["light"]),
    ]
    policy = hebbian_fit(pairs, DIM, STEPS, LR, random.Random(162))

    def from_library(name: str) -> tuple[str, ...] | None:
        return library.get(name)

    lib_board = from_library("board") == SKILLS["board"]
    lib_light = from_library("light")
    library.clear()
    lib_after = from_library("board")

    def compiled(name: str) -> str:
        return nearest_name(matvec(policy, cues[name]), skill_embeds)

    board_ok = compiled("board") == "board"
    stick_ok = compiled("stick") == "stick"
    light_ok = compiled("light") == "light"
    composed_norm = l2(composed)

    checks = {
        "library_had_parts": lib_board,
        "library_never_had_light": lib_light is None,
        "library_empty_after_unplug": lib_after is None,
        "weights_keep_board": board_ok,
        "weights_keep_stick": stick_ok,
        "weights_run_light_without_card": light_ok,
        "composition_embedding_nonzero": composed_norm > 0.5,
    }
    return {
        "summary": (
            "库里只有 board / stick，没有 light 这张卡。"
            f"卸掉库之后查卡为空。"
            f"写入 W 的组合：board={compiled('board')}, stick={compiled('stick')}, "
            f"light={compiled('light')}。"
            "失败阈值：卸库后取不出 light，或零件技能也丢了。"
        ),
        "metrics": {
            "compiled_board": compiled("board"),
            "compiled_stick": compiled("stick"),
            "compiled_light": compiled("light"),
            "lib_light_before": lib_light,
            "composed_norm": composed_norm,
        },
        "checks": checks,
    }


EXPERIMENT = ExtraExperiment(
    extra_id="compose",
    title="组合技能：库里没有这张卡",
    question="Voyager 式技能卡没有「火把」这一张。两个已编译技能拼起来，卸掉库还能跑吗？",
    lesson_hint="16,21",
    run=run,
)
