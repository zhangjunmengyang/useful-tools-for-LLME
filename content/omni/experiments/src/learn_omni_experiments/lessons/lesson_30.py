from __future__ import annotations

import json
from typing import Any

from ..core import LessonExperiment


CONTINUE = "CONTINUE"
PAUSE = "PAUSE"
REPLAN = "REPLAN"


def open_loop_window_ms(horizon: int, freq_hz: int) -> int:
    if freq_hz <= 0 or horizon < 0:
        raise ValueError("freq_hz must be positive and horizon must be >= 0")
    return (horizon * 1000) // freq_hz


def commit_window_ms(k_exec: int, freq_hz: int) -> int:
    if freq_hz <= 0 or k_exec < 0:
        raise ValueError("freq_hz must be positive and k_exec must be >= 0")
    return (k_exec * 1000) // freq_hz


def chunk_is_stale(delay_ms: int, horizon: int, freq_hz: int) -> bool:
    """延迟大于或等于开环窗口则丢弃。恰好相等也过期。"""
    return delay_ms >= open_loop_window_ms(horizon, freq_hz)


def _pop_due(queue: list[tuple[int, int, str]], now_ms: int) -> list[tuple[int, int, str]]:
    due: list[tuple[int, int, str]] = []
    still: list[tuple[int, int, str]] = []
    for item in queue:
        if item[0] <= now_ms:
            due.append(item)
        else:
            still.append(item)
    queue[:] = still
    due.sort(key=lambda item: (item[0], item[1]))
    return due


def replay(
    freq_hz: int,
    horizon: int,
    k_exec: int,
    delay_ms: int,
    control_events: list[tuple[int, str]],
    end_ms: int = 1600,
) -> dict[str, Any]:
    if not (1 <= k_exec <= horizon):
        raise ValueError("k_exec must satisfy 1 <= k_exec <= horizon")

    dt_ms = 1000 // freq_hz
    tick_ms = 10
    if dt_ms % tick_ms != 0:
        raise ValueError("dt_ms must be a multiple of 10 ms")
    window_ms = open_loop_window_ms(horizon, freq_hz)
    commit_ms = commit_window_ms(k_exec, freq_hz)
    event_queue = [
        (time_ms, sequence_no, action)
        for sequence_no, (time_ms, action) in enumerate(control_events, start=1)
    ]
    branches: dict[int, dict[str, Any]] = {
        1: {"parent": None, "status": "active"},
    }
    active_branch = 1
    mode = "GENERATING"
    next_chunk_id = 1
    in_flight: dict[str, int] | None = None
    current: dict[str, Any] | None = None
    executed: list[dict[str, int]] = []
    discarded: list[dict[str, int | str]] = []
    consumed: list[dict[str, int | str]] = []
    trace: list[dict[str, int | str]] = []
    paused_branch: int | None = None
    resumed_branch: int | None = None
    replan_at: int | None = None
    stale_discards = 0

    def start_inference(now_ms: int) -> None:
        nonlocal next_chunk_id, in_flight
        if in_flight is not None:
            return
        chunk_id = next_chunk_id
        next_chunk_id += 1
        in_flight = {
            "chunk_id": chunk_id,
            "branch_id": active_branch,
            "t_obs_ms": now_ms,
            "available_at_ms": now_ms + delay_ms,
        }
        trace.append(
            {
                "at_ms": now_ms,
                "kind": "infer_start",
                "chunk_id": chunk_id,
                "branch_id": active_branch,
            },
        )

    def drop_remaining(reason: str, now_ms: int) -> int:
        nonlocal current
        dropped = 0
        if current is None:
            return 0
        for step_index in current["remaining"]:
            discarded.append(
                {
                    "at_ms": now_ms,
                    "chunk_id": int(current["chunk_id"]),
                    "branch_id": int(current["branch_id"]),
                    "step_index": int(step_index),
                    "reason": reason,
                },
            )
            dropped += 1
        current = None
        return dropped

    def cancel_in_flight(reason: str, now_ms: int) -> None:
        nonlocal in_flight, stale_discards
        if in_flight is None:
            return
        discarded.append(
            {
                "at_ms": now_ms,
                "chunk_id": int(in_flight["chunk_id"]),
                "branch_id": int(in_flight["branch_id"]),
                "step_index": -1,
                "reason": reason,
            },
        )
        if reason == "stale":
            stale_discards += 1
        trace.append(
            {
                "at_ms": now_ms,
                "kind": "discard",
                "chunk_id": int(in_flight["chunk_id"]),
                "branch_id": int(in_flight["branch_id"]),
                "reason": reason,
            },
        )
        in_flight = None

    start_inference(0)

    for now_ms in range(0, end_ms + 1, tick_ms):
        for available_at, sequence_no, action in _pop_due(event_queue, now_ms):
            consumed.append(
                {
                    "available_at_ms": available_at,
                    "consumed_at_ms": now_ms,
                    "sequence_no": sequence_no,
                    "action": action,
                },
            )
            if action == PAUSE and mode == "GENERATING":
                mode = "PAUSED"
                paused_branch = active_branch
            elif action == CONTINUE and mode == "PAUSED":
                mode = "GENERATING"
                resumed_branch = active_branch
            elif action == REPLAN:
                branches[active_branch]["status"] = "superseded"
                drop_remaining("replan", now_ms)
                cancel_in_flight("replan", now_ms)
                parent = active_branch
                active_branch = max(branches) + 1
                branches[active_branch] = {"parent": parent, "status": "active"}
                mode = "GENERATING"
                replan_at = now_ms
                start_inference(now_ms)
            trace.append(
                {
                    "at_ms": now_ms,
                    "kind": "control",
                    "action": action,
                    "branch_id": active_branch,
                },
            )

        if in_flight is not None and now_ms >= int(in_flight["available_at_ms"]):
            expiry_ms = int(in_flight["t_obs_ms"]) + window_ms
            chunk_id = int(in_flight["chunk_id"])
            branch_id = int(in_flight["branch_id"])
            if now_ms > expiry_ms or chunk_is_stale(delay_ms, horizon, freq_hz):
                stale_discards += 1
                discarded.append(
                    {
                        "at_ms": now_ms,
                        "chunk_id": chunk_id,
                        "branch_id": branch_id,
                        "step_index": -1,
                        "reason": "stale",
                    },
                )
                trace.append(
                    {
                        "at_ms": now_ms,
                        "kind": "stale_drop",
                        "chunk_id": chunk_id,
                        "branch_id": branch_id,
                    },
                )
                in_flight = None
            elif branch_id != active_branch or branches[branch_id]["status"] != "active":
                discarded.append(
                    {
                        "at_ms": now_ms,
                        "chunk_id": chunk_id,
                        "branch_id": branch_id,
                        "step_index": -1,
                        "reason": "superseded",
                    },
                )
                in_flight = None
            else:
                current = {
                    "chunk_id": chunk_id,
                    "branch_id": branch_id,
                    "t_obs_ms": int(in_flight["t_obs_ms"]),
                    "available_at_ms": int(in_flight["available_at_ms"]),
                    "remaining": list(range(horizon)),
                    "executed": 0,
                }
                trace.append(
                    {
                        "at_ms": now_ms,
                        "kind": "chunk_ready",
                        "chunk_id": chunk_id,
                        "branch_id": branch_id,
                    },
                )
                in_flight = None

        on_control_tick = now_ms % dt_ms == 0
        if (
            on_control_tick
            and mode == "GENERATING"
            and current is not None
            and current["remaining"]
        ):
            if int(current["executed"]) < k_exec:
                step_index = int(current["remaining"].pop(0))
                current["executed"] = int(current["executed"]) + 1
                executed.append(
                    {
                        "at_ms": now_ms,
                        "chunk_id": int(current["chunk_id"]),
                        "branch_id": int(current["branch_id"]),
                        "step_index": step_index,
                    },
                )
                trace.append(
                    {
                        "at_ms": now_ms,
                        "kind": "execute",
                        "chunk_id": int(current["chunk_id"]),
                        "branch_id": int(current["branch_id"]),
                        "step_index": step_index,
                    },
                )
            if int(current["executed"]) >= k_exec or not current["remaining"]:
                leftover = list(current["remaining"])
                for step_index in leftover:
                    discarded.append(
                        {
                            "at_ms": now_ms,
                            "chunk_id": int(current["chunk_id"]),
                            "branch_id": int(current["branch_id"]),
                            "step_index": int(step_index),
                            "reason": "receding_uncommitted",
                        },
                    )
                current = None
                start_inference(now_ms)
        elif (
            on_control_tick
            and mode == "GENERATING"
            and current is None
            and in_flight is None
        ):
            start_inference(now_ms)

    return {
        "freq_hz": freq_hz,
        "horizon": horizon,
        "k_exec": k_exec,
        "delay_ms": delay_ms,
        "dt_ms": dt_ms,
        "window_ms": window_ms,
        "commit_ms": commit_ms,
        "stale": chunk_is_stale(delay_ms, horizon, freq_hz),
        "branches": branches,
        "executed": executed,
        "discarded": discarded,
        "consumed": consumed,
        "trace": trace,
        "stale_discards": stale_discards,
        "paused_branch": paused_branch,
        "resumed_branch": resumed_branch,
        "replan_at": replan_at,
        "active_branch": active_branch,
    }


def _trace_digest(trace: list[dict[str, int | str]]) -> str:
    return json.dumps(trace, sort_keys=True, separators=(",", ":"))


def run() -> dict[str, Any]:
    freq_hz = 10
    horizon = 8
    k_exec = 4
    fresh_delay_ms = 200
    stale_delay_ms = 900

    replan = replay(
        freq_hz,
        horizon,
        k_exec,
        fresh_delay_ms,
        control_events=[(350, REPLAN)],
    )
    replan_again = replay(
        freq_hz,
        horizon,
        k_exec,
        fresh_delay_ms,
        control_events=[(350, REPLAN)],
    )
    pause_resume = replay(
        freq_hz,
        horizon,
        k_exec,
        fresh_delay_ms,
        control_events=[(250, PAUSE), (550, CONTINUE)],
    )
    stale = replay(
        freq_hz,
        horizon,
        k_exec,
        stale_delay_ms,
        control_events=[],
        end_ms=1200,
    )

    replan_at = int(replan["replan_at"])
    old_steps_after_replan = [
        item
        for item in replan["executed"]
        if int(item["branch_id"]) == 1 and int(item["at_ms"]) >= replan_at
    ]
    new_steps_after_replan = [
        item
        for item in replan["executed"]
        if int(item["branch_id"]) == 2 and int(item["at_ms"]) >= replan_at
    ]
    replan_drops = [
        item
        for item in replan["discarded"]
        if item["reason"] == "replan" and int(item["branch_id"]) == 1
    ]
    pause_exec = [
        item
        for item in pause_resume["executed"]
        if 250 <= int(item["at_ms"]) < 550
    ]
    resumed_exec = [
        item
        for item in pause_resume["executed"]
        if int(item["at_ms"]) >= 550 and int(item["branch_id"]) == 1
    ]
    stale_executed = stale["executed"]
    consumed = replan["consumed"]

    checks = {
        "boundary_equal_window_is_stale": chunk_is_stale(800, 8, 10)
        and not chunk_is_stale(799, 8, 10),
        "open_loop_window_matches_h_over_f": replan["window_ms"]
        == open_loop_window_ms(horizon, freq_hz)
        == 800,
        "commit_window_matches_k_over_f": replan["commit_ms"]
        == commit_window_ms(k_exec, freq_hz)
        == 400,
        "fresh_delay_is_below_open_loop_window": fresh_delay_ms
        < replan["window_ms"]
        and not replan["stale"],
        "stale_delay_is_dropped_and_not_executed": stale["stale"]
        and stale["stale_discards"] > 0
        and stale_executed == [],
        "events_never_consumed_before_available": all(
            int(event["consumed_at_ms"]) >= int(event["available_at_ms"])
            for event in consumed
        ),
        "replan_marks_old_branch_superseded": replan["branches"][1]["status"]
        == "superseded"
        and replan["branches"][2]["parent"] == 1,
        "replan_does_not_execute_old_remaining_steps": old_steps_after_replan
        == []
        and len(replan_drops) > 0,
        "replan_executes_new_branch_only": len(new_steps_after_replan) > 0
        and all(int(item["branch_id"]) == 2 for item in new_steps_after_replan),
        "pause_resume_keeps_same_branch": pause_resume["paused_branch"]
        == pause_resume["resumed_branch"]
        == 1
        and pause_exec == []
        and len(resumed_exec) > 0,
        "event_replay_is_deterministic": replan["trace"] == replan_again["trace"]
        and _trace_digest(replan["trace"]) == _trace_digest(replan_again["trace"]),
    }

    return {
        "summary": (
            "用控制时钟回放动作分块：fresh chunk 满足 d < H/f，"
            "过期 chunk 被丢弃，REPLAN 后不再执行旧剩余步，PAUSE/CONTINUE 保留同一分支。"
        ),
        "metrics": {
            "freq_hz": freq_hz,
            "horizon": horizon,
            "k_exec": k_exec,
            "open_loop_window_ms": replan["window_ms"],
            "commit_window_ms": replan["commit_ms"],
            "fresh_delay_ms": fresh_delay_ms,
            "stale_delay_ms": stale_delay_ms,
            "replan_executed_steps": len(replan["executed"]),
            "replan_discarded": len(replan["discarded"]),
            "old_steps_after_replan": len(old_steps_after_replan),
            "new_steps_after_replan": len(new_steps_after_replan),
            "stale_discards": stale["stale_discards"],
            "stale_executed_steps": len(stale_executed),
            "pause_steps_during_pause": len(pause_exec),
            "pause_steps_after_resume": len(resumed_exec),
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="30",
    title="在控制回路里权衡频率、分块和延迟",
    question="推理延迟超过开环窗口时，过期 chunk 和 REPLAN 后的旧剩余步分别该怎么处理？",
    run=run,
)
