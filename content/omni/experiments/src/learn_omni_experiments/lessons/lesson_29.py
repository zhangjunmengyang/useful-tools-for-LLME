from __future__ import annotations

from typing import Any

from ..core import LessonExperiment


CONTROL_DT = 1
PLAN_RATIO = 4
PLAN_DT = PLAN_RATIO * CONTROL_DT
HORIZON = 24
EXPIRE_STEPS = 8
GAIN = 0.5
GOAL = 5.0
SUCCESS_EPS = 0.35
WRONG_VALUE = 0.0
PAUSE_START = 8
PAUSE_END = HORIZON
WRONG_FROM = 8


def _correct_subgoal(tick: int) -> float:
    stage = tick // PLAN_RATIO
    return min(GOAL, float(stage + 1))


def _simulate(
    *,
    pause_start: int | None = None,
    pause_end: int | None = None,
    wrong_from: int | None = None,
    wrong_value: float = WRONG_VALUE,
    expire_steps: int = EXPIRE_STEPS,
) -> dict[str, Any]:
    position = 0.0
    subgoal = 0.0
    last_plan_tick = -PLAN_RATIO
    plan_ticks: list[int] = []
    consumed: list[float] = []
    trajectory: list[float] = []
    stale_from: int | None = None

    for tick in range(HORIZON):
        paused = (
            pause_start is not None
            and pause_end is not None
            and pause_start <= tick < pause_end
        )
        if tick % PLAN_RATIO == 0 and not paused:
            if wrong_from is not None and tick >= wrong_from:
                subgoal = wrong_value
            else:
                subgoal = _correct_subgoal(tick)
            last_plan_tick = tick
            plan_ticks.append(tick)

        age = tick - last_plan_tick
        if age > expire_steps and stale_from is None:
            stale_from = tick

        action = GAIN * (subgoal - position)
        position = position + action
        consumed.append(subgoal)
        trajectory.append(position)

    success = abs(position - GOAL) <= SUCCESS_EPS and stale_from is None
    return {
        "trajectory": trajectory,
        "consumed": consumed,
        "plan_ticks": plan_ticks,
        "plan_count": len(plan_ticks),
        "final_x": position,
        "stale_from": stale_from,
        "success": success,
        "max_age": HORIZON - 1 - last_plan_tick,
    }


def _trajectory_l2(left: list[float], right: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right)) ** 0.5


def run() -> dict[str, Any]:
    baseline = _simulate()
    paused = _simulate(pause_start=PAUSE_START, pause_end=PAUSE_END)
    injected = _simulate(wrong_from=WRONG_FROM, wrong_value=WRONG_VALUE)
    first_block = baseline["consumed"][:PLAN_RATIO]
    second_block = baseline["consumed"][PLAN_RATIO : 2 * PLAN_RATIO]
    last_plan_before_pause = max(
        tick for tick in range(0, PAUSE_START, PLAN_RATIO)
    )

    checks = {
        "control_loop_ran_every_step": len(baseline["consumed"]) == HORIZON,
        "planning_loop_is_not_every_step": (
            baseline["plan_count"] == HORIZON // PLAN_RATIO
            and baseline["plan_count"] < HORIZON
        ),
        "plan_period_is_k_control_periods": PLAN_DT == PLAN_RATIO * CONTROL_DT,
        "every_control_step_consumed_current_subgoal": (
            len(baseline["consumed"]) == HORIZON
            and all(item == first_block[0] for item in first_block)
            and all(item == second_block[0] for item in second_block)
            and first_block[0] != second_block[0]
        ),
        "plan_ticks_are_multiples_of_k": all(
            tick % PLAN_RATIO == 0 for tick in baseline["plan_ticks"]
        ),
        "long_system2_pause_fails_after_expiry": (
            paused["success"] is False
            and paused["stale_from"] == last_plan_before_pause + EXPIRE_STEPS + 1
            and paused["plan_count"] < baseline["plan_count"]
        ),
        "wrong_subgoal_changes_end_effector_trajectory": (
            injected["trajectory"] != baseline["trajectory"]
            and _trajectory_l2(injected["trajectory"], baseline["trajectory"]) > 1.0
        ),
        "paused_planner_skips_control_ticks": paused["plan_count"]
        == PAUSE_START // PLAN_RATIO,
    }

    return {
        "summary": (
            "两个离散时间循环：控制环每步消费当前子目标，规划环只在 "
            f"t mod {PLAN_RATIO} = 0 且未暂停时更新；"
            f"ΔT2={PLAN_DT}、ΔT1={CONTROL_DT}。"
            "System 2 暂停超过过期步数后任务失败；错误子目标改变末端轨迹。"
        ),
        "metrics": {
            "control_dt": CONTROL_DT,
            "plan_dt": PLAN_DT,
            "plan_ratio_k": PLAN_RATIO,
            "horizon": HORIZON,
            "expire_steps": EXPIRE_STEPS,
            "baseline_plan_steps": baseline["plan_count"],
            "baseline_control_steps": len(baseline["consumed"]),
            "baseline_final_x": round(baseline["final_x"], 6),
            "paused_plan_steps": paused["plan_count"],
            "paused_stale_from": paused["stale_from"],
            "paused_final_x": round(paused["final_x"], 6),
            "injected_traj_l2": round(
                _trajectory_l2(injected["trajectory"], baseline["trajectory"]),
                6,
            ),
            "last_plan_before_pause": last_plan_before_pause,
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="29",
    title="拆开快慢双系统 VLA",
    question="控制环是否每步消费当前子目标，而规划环不是每步都跑？",
    run=run,
)
