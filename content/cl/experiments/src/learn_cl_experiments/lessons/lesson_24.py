from __future__ import annotations

from typing import Any

from ..core import LessonExperiment


DAYS = 14
TASKS_PER_DAY = 10
SEAT_CHANGE_DAY = 7
NEW_TOOL_DAY = 8
OLD_PROBE = (
    ("call", "xiaowang"),
    ("call", "xiaoli"),
    ("project", "beiji"),
    ("script", "deploy"),
)


def _world(day: int) -> dict[str, Any]:
    seat = "B7" if day >= SEAT_CHANGE_DAY else "A3"
    tools = ["deploy"]
    if day >= NEW_TOOL_DAY:
        tools.append("nightly_sync")
    return {
        "seats": {"xiaowang": seat, "xiaoli": "C2", "xiaozhao": "D4"},
        "projects": {"beiji": "north-wing", "lighthouse": "lab-2"},
        "tools": tools,
    }


def _tasks(day: int) -> list[tuple[str, str]]:
    roster = [
        ("call", "xiaowang"),
        ("call", "xiaoli"),
        ("call", "xiaozhao"),
        ("project", "beiji"),
        ("project", "lighthouse"),
        ("script", "deploy"),
        ("call", "xiaowang"),
        ("project", "beiji"),
        ("script", "deploy"),
        ("call", "xiaoli"),
    ]
    if day >= NEW_TOOL_DAY:
        roster[5] = ("script", "nightly_sync")
        roster[8] = ("script", "nightly_sync")
    return roster[:TASKS_PER_DAY]


class Agent:
    def __init__(self, learn: bool) -> None:
        self.learn = learn
        self.memory: dict[str, str] = {}
        self.skills: dict[str, int] = {}
        self.tool_attempts: list[int] = []

    def act(self, task: tuple[str, str], world: dict[str, Any]) -> tuple[bool, int]:
        kind, name = task
        if kind == "call":
            predicted = self.memory.get(f"seat:{name}")
            actual = world["seats"][name]
            success = predicted == actual
            attempts = 1 if success else 2
            if self.learn:
                self.memory[f"seat:{name}"] = actual
            return success, attempts
        if kind == "project":
            predicted = self.memory.get(f"project:{name}")
            actual = world["projects"][name]
            success = predicted == actual
            attempts = 1 if success else 2
            if self.learn:
                self.memory[f"project:{name}"] = actual
            return success, attempts
        known_tool = name in world["tools"]
        success = name in self.skills and known_tool
        attempts = 1 if success else 5
        if self.learn and known_tool:
            self.skills[name] = 1
        if name == "nightly_sync":
            self.tool_attempts.append(attempts)
        return success, attempts

    def replay(self, world: dict[str, Any]) -> float:
        hits = 0
        for task in OLD_PROBE:
            kind, name = task
            if kind == "call":
                hits += int(self.memory.get(f"seat:{name}") == world["seats"][name])
            elif kind == "project":
                hits += int(self.memory.get(f"project:{name}") == world["projects"][name])
            else:
                hits += int(name in self.skills and name in world["tools"])
        return hits / len(OLD_PROBE)


def _simulate(learn: bool) -> dict[str, Any]:
    agent = Agent(learn=learn)
    daily_success: list[float] = []
    memory_curve: list[int] = []
    skill_curve: list[int] = []
    first_day_tasks = _tasks(1)
    for day in range(1, DAYS + 1):
        world = _world(day)
        outcomes = []
        for task in _tasks(day):
            success, _ = agent.act(task, world)
            outcomes.append(success)
        daily_success.append(sum(outcomes) / len(outcomes))
        memory_curve.append(len(agent.memory))
        skill_curve.append(len(agent.skills))
    retention = agent.replay(_world(DAYS))
    day1_world = _world(1)
    stale_seat = agent.memory.get("seat:xiaowang") == day1_world["seats"]["xiaowang"]
    return {
        "daily_success": daily_success,
        "memory_curve": memory_curve,
        "skill_curve": skill_curve,
        "retention": retention,
        "xiaowang_seat": agent.memory.get("seat:xiaowang", ""),
        "stale_seat": stale_seat,
        "tool_attempts": agent.tool_attempts,
        "final_memory": len(agent.memory),
        "final_skills": len(agent.skills),
        "day1_task_count": len(first_day_tasks),
    }


def run() -> dict[str, Any]:
    learning = _simulate(learn=True)
    frozen = _simulate(learn=False)
    learner_tool = learning["tool_attempts"]
    tool_improved = bool(learner_tool) and learner_tool[-1] < learner_tool[0]

    checks = {
        "frozen_retention_worse": frozen["retention"] < learning["retention"],
        "frozen_old_tasks_fail": frozen["retention"] < 0.3,
        "learner_keeps_old_tasks": learning["retention"] > 0.7,
        "seat_conflict_overwritten": learning["xiaowang_seat"] == "B7",
        "learner_memory_grows": learning["final_memory"] > 0 and frozen["final_memory"] == 0,
        "new_tool_attempts_drop": tool_improved,
    }
    return {
        "summary": (
            f"14 个工作日对照：会写记忆/技能的 Agent 第 14 日旧任务保持 "
            f"{learning['retention']:.2f}，冻结对照 {frozen['retention']:.2f}。"
            "第 7 日座位冲突覆盖成 B7；夜间同步工具的尝试次数随入库下降。"
            "失败阈值：冻结保持率不低于学习者，或保持率 ≥ 0.3。"
        ),
        "metrics": {
            "learning_retention": learning["retention"],
            "frozen_retention": frozen["retention"],
            "learning_memory_curve": learning["memory_curve"],
            "frozen_memory_curve": frozen["memory_curve"],
            "learning_skill_curve": learning["skill_curve"],
            "xiaowang_seat": learning["xiaowang_seat"],
            "learning_tool_attempts": learning["tool_attempts"],
            "learning_daily_success": learning["daily_success"],
            "frozen_daily_success": frozen["daily_success"],
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="24",
    title="两个月上岗（缩小版）",
    question="冻结对照在第 14 日的旧任务保持是不是更差？",
    run=run,
)
