from __future__ import annotations

import random
from typing import Any

from ..core import LessonExperiment


Program = tuple[str, ...]


TASKS = (
    {
        "name": "collect_wood",
        "tags": ("wood", "pickup"),
        "solution": ("goto_wood", "pickup"),
    },
    {
        "name": "collect_stone",
        "tags": ("stone", "pickup"),
        "solution": ("goto_stone", "pickup"),
    },
    {
        "name": "craft_plank",
        "tags": ("wood", "craft"),
        "solution": ("goto_wood", "pickup", "goto_bench", "craft_plank"),
    },
    {
        "name": "craft_axe",
        "tags": ("stone", "wood", "craft"),
        "solution": (
            "goto_wood",
            "pickup",
            "goto_stone",
            "pickup",
            "goto_bench",
            "craft_axe",
        ),
    },
)

CANDIDATES: dict[str, tuple[Program, ...]] = {
    "collect_wood": (
        ("goto_wood", "pickup"),
        ("goto_stone", "pickup"),
        ("wait",),
        ("goto_bench",),
        ("goto_wood", "wait"),
        ("pickup",),
    ),
    "collect_stone": (
        ("goto_stone", "pickup"),
        ("goto_wood", "pickup"),
        ("wait",),
        ("goto_bench", "craft_plank"),
        ("goto_stone", "wait"),
        ("pickup",),
    ),
    "craft_plank": (
        ("goto_wood", "pickup", "goto_bench", "craft_plank"),
        ("goto_stone", "pickup"),
        ("goto_bench", "craft_plank"),
        ("wait", "wait"),
        ("goto_wood", "goto_bench"),
        ("pickup", "craft_plank"),
    ),
    "craft_axe": (
        ("goto_wood", "pickup", "goto_stone", "pickup", "goto_bench", "craft_axe"),
        ("goto_wood", "pickup", "goto_bench", "craft_plank"),
        ("goto_stone", "pickup"),
        ("wait",),
        ("goto_bench", "craft_axe"),
        ("goto_wood", "pickup"),
    ),
}


class SkillLibrary:
    def __init__(self) -> None:
        self.skills: list[dict[str, Any]] = []

    def add(self, name: str, tags: tuple[str, ...], program: Program) -> None:
        self.skills.append({"name": name, "tags": tags, "program": program})

    def retrieve(self, tags: tuple[str, ...]) -> Program | None:
        if not self.skills:
            return None
        query = set(tags)
        scored = []
        for skill in self.skills:
            skill_tags = set(skill["tags"])
            overlap = len(query & skill_tags)
            subset = int(skill_tags <= query)
            scored.append((subset, overlap, skill["program"], skill["name"]))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if scored[0][1] <= 0:
            return None
        return scored[0][2]

    def find(self, name: str) -> Program | None:
        for skill in self.skills:
            if skill["name"] == name:
                return skill["program"]
        return None


def _execute(program: Program, solution: Program) -> bool:
    return program == solution


def _compose(task_name: str, library: SkillLibrary) -> Program | None:
    if task_name == "craft_plank":
        wood = library.find("collect_wood")
        if wood is None:
            return None
        return wood + ("goto_bench", "craft_plank")
    if task_name == "craft_axe":
        wood = library.find("collect_wood")
        stone = library.find("collect_stone")
        if wood is None or stone is None:
            return None
        return wood + stone + ("goto_bench", "craft_axe")
    return None


def _solve(
    task: dict[str, Any],
    library: SkillLibrary | None,
    rng: random.Random,
    cap: int = 24,
) -> dict[str, Any]:
    attempts = 0
    retrieved = False
    if library is not None:
        program = library.retrieve(task["tags"])
        composed = _compose(task["name"], library)
        if composed is not None:
            retrieved = True
            attempts += 1
            if _execute(composed, task["solution"]):
                library.add(task["name"], task["tags"], composed)
                return {
                    "attempts": attempts,
                    "retrieved": True,
                    "success": True,
                }
        if program is not None:
            retrieved = True
            attempts += 1
            if _execute(program, task["solution"]):
                return {
                    "attempts": attempts,
                    "retrieved": True,
                    "success": True,
                }
    options = list(CANDIDATES[task["name"]])
    while attempts < cap:
        attempts += 1
        program = options[rng.randrange(len(options))]
        if _execute(program, task["solution"]):
            if library is not None:
                library.add(task["name"], task["tags"], program)
            return {
                "attempts": attempts,
                "retrieved": retrieved,
                "success": True,
            }
    return {"attempts": attempts, "retrieved": retrieved, "success": False}


def _run(use_library: bool, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    library = SkillLibrary() if use_library else None
    logs = [_solve(task, library, rng) for task in TASKS]
    later = logs[2:]
    return {
        "attempts": [log["attempts"] for log in logs],
        "later_mean": sum(log["attempts"] for log in later) / len(later),
        "library_size": 0 if library is None else len(library.skills),
        "retrieved_later": any(log["retrieved"] for log in later),
        "all_success": all(log["success"] for log in logs),
    }


def run() -> dict[str, Any]:
    with_lib = _run(use_library=True, seed=0)
    without = _run(use_library=False, seed=0)

    checks = {
        "library_cuts_later_attempts": with_lib["later_mean"] < without["later_mean"],
        "later_with_library_is_short": with_lib["later_mean"] <= 2.0,
        "library_grows_after_success": with_lib["library_size"] >= 2,
        "later_task_reuses_skill": with_lib["retrieved_later"],
        "from_scratch_has_empty_library": without["library_size"] == 0,
        "all_tasks_eventually_succeed": with_lib["all_success"] and without["all_success"],
    }
    return {
        "summary": (
            f"网格世界四个任务：有技能库时后两个任务平均尝试 {with_lib['later_mean']:.2f} 次，"
            f"从零写程序是 {without['later_mean']:.2f} 次。库在成功后增长到 "
            f"{with_lib['library_size']} 条并被后续任务检索。失败阈值：后期尝试次数没有下降。"
        ),
        "metrics": {
            "with_library_attempts": with_lib["attempts"],
            "without_library_attempts": without["attempts"],
            "later_mean_with_library": with_lib["later_mean"],
            "later_mean_without_library": without["later_mean"],
            "library_size": with_lib["library_size"],
            "retrieved_later": int(with_lib["retrieved_later"]),
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="21",
    title="技能写成能再调用的代码",
    question="有技能库之后，后续任务的尝试次数会不会下降？",
    run=run,
)
