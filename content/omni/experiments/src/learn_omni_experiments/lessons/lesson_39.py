from __future__ import annotations

from typing import Any

from ..core import LessonExperiment


GOALS = ("open_drawer", "pick_blue", "place_in_drawer", "close_drawer")
FAIL_INDEX = 1
R_MAX = 2
TOKENS_PER_GOAL = 8
TOKENS_PER_STEP = 20


def empty_world() -> dict[str, bool]:
    return {
        "drawer_open": False,
        "holding_blue": False,
        "blue_in_drawer": False,
        "drawer_closed": False,
    }


def delta_success(skill: str, world: dict[str, bool]) -> bool:
    if skill == "open_drawer":
        return not world["drawer_open"]
    if skill == "pick_blue":
        return world["drawer_open"] and not world["holding_blue"]
    if skill == "place_in_drawer":
        return world["holding_blue"] and world["drawer_open"]
    if skill == "close_drawer":
        return world["drawer_open"] and world["blue_in_drawer"] and not world["drawer_closed"]
    raise ValueError(f"unknown skill: {skill}")


def abs_true(skill: str, world: dict[str, bool]) -> bool:
    if skill == "open_drawer":
        return world["drawer_open"]
    if skill == "pick_blue":
        return world["holding_blue"]
    if skill == "place_in_drawer":
        return world["blue_in_drawer"]
    if skill == "close_drawer":
        return world["drawer_closed"]
    raise ValueError(f"unknown skill: {skill}")


def apply_skill(skill: str, world: dict[str, bool]) -> dict[str, bool]:
    next_world = dict(world)
    if skill == "open_drawer":
        next_world["drawer_open"] = True
        next_world["drawer_closed"] = False
    elif skill == "pick_blue":
        next_world["holding_blue"] = True
    elif skill == "place_in_drawer":
        next_world["holding_blue"] = False
        next_world["blue_in_drawer"] = True
    elif skill == "close_drawer":
        next_world["drawer_open"] = False
        next_world["drawer_closed"] = True
    return next_world


def apply_ops(ops: list[tuple[str, str]]) -> dict[str, Any]:
    """Replay a recorded (op, skill) list. ops: push / emit / commit / retry / drop."""
    stack: list[str] = []
    committed: list[str] = []
    executed: list[str] = []
    retries: dict[str, int] = {goal: 0 for goal in GOALS}
    for op, skill in ops:
        if op == "push":
            stack.append(skill)
        elif op == "emit":
            if not stack or stack[-1] != skill:
                raise ValueError("emit must target stack top")
            executed.append(skill)
        elif op == "commit":
            if not stack or stack[-1] != skill:
                raise ValueError("commit must target stack top")
            stack.pop()
            committed.append(skill)
        elif op == "retry":
            if not stack or stack[-1] != skill:
                raise ValueError("retry must target stack top")
            retries[skill] += 1
        elif op == "drop":
            if not stack or stack[-1] != skill:
                raise ValueError("drop must target stack top")
            stack.pop()
        else:
            raise ValueError(f"unknown op: {op}")
    return {
        "stack": tuple(stack),
        "committed": tuple(committed),
        "executed": tuple(executed),
        "retries": dict(retries),
    }


def stack_protocol(
    fail_index: int = FAIL_INDEX,
    r_max: int = R_MAX,
    succeed_on_retry: bool = True,
) -> dict[str, Any]:
    ops: list[tuple[str, str]] = [("push", goal) for goal in reversed(GOALS)]
    world = empty_world()
    executed: list[str] = []
    committed: list[str] = []
    retries = {goal: 0 for goal in GOALS}
    delta_fails: list[str] = []
    index = 0
    while index < len(GOALS):
        skill = GOALS[index]
        forced_fail = index == fail_index and retries[skill] == 0
        can_delta = delta_success(skill, world) and not forced_fail
        ops.append(("emit", skill))
        executed.append(skill)
        if can_delta and (index != fail_index or succeed_on_retry or retries[skill] == 0):
            if index == fail_index and retries[skill] == 0:
                ops.append(("retry", skill))
                retries[skill] += 1
                if retries[skill] >= r_max:
                    break
                continue
            world = apply_skill(skill, world)
            ops.append(("commit", skill))
            committed.append(skill)
            index += 1
            continue
        if not can_delta and not forced_fail:
            delta_fails.append(skill)
        ops.append(("retry", skill))
        retries[skill] += 1
        if retries[skill] >= r_max:
            break
        if forced_fail and succeed_on_retry:
            continue
        break
    return {
        "ops": ops,
        "executed": tuple(executed),
        "committed": tuple(committed),
        "retries": retries,
        "world": world,
        "delta_fails": tuple(delta_fails),
        "stack_depth_after_first_commit": (
            len(GOALS) - 1 if list(committed[:1]) == [GOALS[0]] else None
        ),
    }


def window_protocol(
    fail_index: int = FAIL_INDEX,
    window_tokens: int = 48,
) -> dict[str, Any]:
    world = empty_world()
    executed: list[str] = []
    committed_truth: list[str] = []
    delta_fails: list[str] = []
    abs_on_delta_fail: list[bool] = []
    cursor = 0
    failed_once = False
    steps = 0
    while cursor < len(GOALS) and steps < 12:
        skill = GOALS[cursor]
        forced_fail = cursor == fail_index and not failed_once
        can_delta = delta_success(skill, world) and not forced_fail
        executed.append(skill)
        steps += 1
        if forced_fail:
            failed_once = True
            cursor = 0
            continue
        if not can_delta:
            delta_fails.append(skill)
            abs_on_delta_fail.append(abs_true(skill, world))
            break
        world = apply_skill(skill, world)
        committed_truth.append(skill)
        cursor += 1
    history_tokens = max(window_tokens, steps * TOKENS_PER_STEP + len(GOALS) * TOKENS_PER_GOAL)
    return {
        "executed": tuple(executed),
        "committed_truth": tuple(committed_truth),
        "delta_fails": tuple(delta_fails),
        "abs_on_delta_fail": tuple(abs_on_delta_fail),
        "window_tokens": history_tokens,
        "open_drawer_count": executed.count("open_drawer"),
    }


def legal_candidates(committed: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(goal for goal in GOALS if goal not in committed)


def run() -> dict[str, Any]:
    stack_once = stack_protocol()
    stack_again = apply_ops(stack_once["ops"])
    stack_replay = apply_ops(stack_once["ops"])
    window_short = window_protocol(window_tokens=48)
    window_long = window_protocol(window_tokens=128)
    world_after_open = apply_skill("open_drawer", empty_world())
    redo_delta = delta_success("open_drawer", world_after_open)
    redo_abs = abs_true("open_drawer", world_after_open)
    aborted = stack_protocol(succeed_on_retry=False, r_max=1)
    committed_after_open = (GOALS[0],)
    candidates = legal_candidates(committed_after_open)
    stack_open_count = stack_once["executed"].count("open_drawer")
    later = stack_once["executed"][1:]
    committed_reexecuted = any(item in later for item in stack_once["committed"][:1])
    k_after = stack_once["stack_depth_after_first_commit"]
    t_short = window_short["window_tokens"]
    t_long = window_long["window_tokens"]

    checks = {
        "stack_ops_are_replayable": (
            stack_again["executed"] == stack_once["executed"]
            and stack_again["committed"] == stack_once["committed"]
            and stack_again["stack"] == ()
            and stack_replay == stack_again
        ),
        "failure_does_not_reexecute_committed": (
            stack_open_count == 1 and not committed_reexecuted
        ),
        "window_replays_committed_first_step": window_short["open_drawer_count"] >= 2,
        "stack_depth_differs_from_window_tokens": (
            k_after == 3 and t_short != k_after and t_long != k_after and t_long >= t_short
        ),
        "retry_respects_r_max": (
            stack_once["retries"]["pick_blue"] <= R_MAX
            and aborted["executed"].count("pick_blue") <= 1
            and "place_in_drawer" not in aborted["executed"]
        ),
        "delta_and_abs_diverge_on_replayed_open": (
            (not redo_delta)
            and redo_abs
            and window_short["delta_fails"][:1] == ("open_drawer",)
            and window_short["abs_on_delta_fail"][:1] == (True,)
        ),
        "legal_candidates_exclude_committed": (
            GOALS[0] not in candidates and GOALS[1] in candidates
        ),
        "window_token_growth_does_not_remove_replay": (
            window_long["open_drawer_count"] == window_short["open_drawer_count"]
        ),
    }
    return {
        "summary": (
            "四步链在第二步首次失败时，栈协议只重试 pick_blue、open_drawer 只执行一次；"
            "窗口回放从列表头再发射，open_drawer 出现第二次且状态差谓词失败、绝对值仍为开。"
            "同一操作序列回放后栈与已提交表不变。k=3 与 T 不是同一个量。"
            "本实验不评测真实 CALVIN 政策。"
        ),
        "metrics": {
            "goals": list(GOALS),
            "fail_index": FAIL_INDEX,
            "r_max": R_MAX,
            "stack_executed": list(stack_once["executed"]),
            "stack_committed": list(stack_once["committed"]),
            "stack_open_drawer_count": stack_open_count,
            "stack_pick_blue_count": stack_once["executed"].count("pick_blue"),
            "stack_depth_after_first_commit": k_after,
            "window_executed": list(window_short["executed"]),
            "window_open_drawer_count": window_short["open_drawer_count"],
            "window_tokens_short": t_short,
            "window_tokens_long": t_long,
            "redo_open_delta_success": redo_delta,
            "redo_open_abs_true": redo_abs,
            "aborted_executed": list(aborted["executed"]),
            "candidates_after_open": list(candidates),
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="39",
    title="用子目标栈处理长程失败",
    question="长程第二步失败后，该把历史塞进窗口回放，还是 pop 栈只重试失败步？",
    run=run,
)
