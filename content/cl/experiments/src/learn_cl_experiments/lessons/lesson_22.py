from __future__ import annotations

from typing import Any

from ..core import LessonExperiment


WEB = {"search", "summarize"}
EXPERT = {"deploy_v1", "oncall_script"}
ENV_SCHEDULE = {
    1: {"smelt_ore"},
    4: {"craft_gear"},
    8: {"repair_bot"},
    11: {"night_batch"},
}
DAYS = 12


def _curve(use_web: bool, use_expert: bool, use_env: bool) -> list[int]:
    skills: set[str] = set()
    counts: list[int] = []
    for day in range(1, DAYS + 1):
        if use_web:
            skills |= WEB
        if use_expert:
            skills |= EXPERT
        if use_env:
            for unlock_day, recipes in ENV_SCHEDULE.items():
                if unlock_day <= day:
                    skills |= recipes
        counts.append(len(skills))
    return counts


def _skills(use_web: bool, use_expert: bool, use_env: bool) -> set[str]:
    skills: set[str] = set()
    if use_web:
        skills |= WEB
    if use_expert:
        skills |= EXPERT
    if use_env:
        for recipes in ENV_SCHEDULE.values():
            skills |= recipes
    return skills


def run() -> dict[str, Any]:
    with_env = _curve(True, True, True)
    no_env = _curve(True, True, False)
    only_env = _curve(False, False, True)
    web_expert = _skills(True, True, False)
    env_all = _skills(False, False, True)
    late_recipes = ENV_SCHEDULE[8] | ENV_SCHEDULE[11]
    plateau_start = 1
    no_env_flat = all(count == no_env[plateau_start] for count in no_env[plateau_start:])

    checks = {
        "env_grows_past_web_expert": with_env[-1] > no_env[-1],
        "closing_env_stops_growth": no_env_flat,
        "env_curve_rises_after_day1": with_env[-1] > with_env[0],
        "late_recipes_not_in_web_or_expert": late_recipes.isdisjoint(web_expert),
        "closed_env_misses_repair_bot": "repair_bot" not in web_expert,
        "env_alone_finds_new_recipes": env_all >= late_recipes and only_env[-1] == len(env_all),
    }
    return {
        "summary": (
            f"三条数据河跑 12 天：打开环境河技能数到 {with_env[-1]}，"
            f"关掉环境河后停在 {no_env[-1]} 且第 2 天起不再增长。"
            "repair_bot / night_batch 只出现在环境日程里。失败阈值：关环境河后曲线仍上升。"
        ),
        "metrics": {
            "with_env_curve": with_env,
            "no_env_curve": no_env,
            "only_env_curve": only_env,
            "with_env_final": with_env[-1],
            "no_env_final": no_env[-1],
            "web_expert_size": len(web_expert),
            "late_recipe_count": len(late_recipes),
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="22",
    title="数据从世界来",
    question="关掉环境这条河之后，技能数还会不会增长？",
    run=run,
)
