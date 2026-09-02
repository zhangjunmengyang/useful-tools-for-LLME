from __future__ import annotations

import json
from typing import Any

from ..core import LessonExperiment

CONTINUE = "CONTINUE"
PAUSE = "PAUSE"
REPLAN = "REPLAN"
FORCE_CUTOFF = "FORCE_CUTOFF"

AUDIO_BLOCK_MS = 320
AUDIO_ENCODE_MS = 40
AUDIO_FRAME_MS = 80
PCM_PLAY_LAG_MS = 30
FREQ_HZ = 20
HORIZON = 8
ACTION_DELAY_MS = 100
CONTACT_STEP = 3
TICK_MS = 10


def open_loop_window_ms(horizon: int, freq_hz: int) -> int:
    if freq_hz <= 0 or horizon < 0:
        raise ValueError("freq_hz must be positive and horizon must be >= 0")
    return (horizon * 1000) // freq_hz


def audio_is_stale(delay_ms: int, block_ms: int = AUDIO_BLOCK_MS) -> bool:
    """音频块延迟大于或等于一块时长则过期。恰好相等也过期。"""
    return delay_ms >= block_ms


def action_is_stale(delay_ms: int, horizon: int, freq_hz: int) -> bool:
    """动作推理延迟大于或等于开环窗口 H/f 则过期。恰好相等也过期。"""
    return delay_ms >= open_loop_window_ms(horizon, freq_hz)


def _pop_due(
    queue: list[tuple[int, int, str, str]],
    now_ms: int,
) -> list[tuple[int, int, str, str]]:
    due: list[tuple[int, int, str, str]] = []
    still: list[tuple[int, int, str, str]] = []
    for item in queue:
        if item[0] <= now_ms:
            due.append(item)
        else:
            still.append(item)
    queue[:] = still
    due.sort(key=lambda item: (item[0], item[1]))
    return due


def replay(
    events: list[tuple[int, str, str]],
    *,
    freq_hz: int = FREQ_HZ,
    horizon: int = HORIZON,
    action_delay_ms: int = ACTION_DELAY_MS,
    audio_delay_ms: int = AUDIO_ENCODE_MS,
    end_ms: int = 900,
    illegal_undo: bool = False,
) -> dict[str, Any]:
    """一张状态表、两列时间戳的确定性回放。

    events 每项为 (available_at_ms, channel, action)。
    channel 只能是 audio 或 action。一次事件只改对应列。
    illegal_undo=True 会在 REPLAN 时错误撤回已播 PCM 和已发生接触，仅作反例。
    """
    dt_ms = 1000 // freq_hz
    window_ms = open_loop_window_ms(horizon, freq_hz)
    event_queue = [
        (time_ms, sequence_no, channel, action)
        for sequence_no, (time_ms, channel, action) in enumerate(events, start=1)
    ]
    branches: dict[int, dict[str, Any]] = {
        1: {
            "parent": None,
            "status": "active",
            "audio_mode": "SPEAKING",
            "action_mode": "GENERATING",
        },
    }
    active_branch = 1
    pending_pcm: list[dict[str, int]] = []
    played_pcm: list[dict[str, int]] = []
    canceled_pcm: list[dict[str, int | str]] = []
    remaining: list[int] = []
    executed: list[dict[str, int]] = []
    discarded_steps: list[dict[str, int | str]] = []
    consumed: list[dict[str, int | str]] = []
    table_rows: list[dict[str, int | str | bool]] = []
    trace: list[dict[str, int | str]] = []
    next_frame_id = 0
    in_flight: dict[str, int] | None = None
    contact_occurred = False
    contact_undone = False
    played_undone = False
    paused_audio_branch: int | None = None
    replan_at: int | None = None
    force_at: int | None = None
    played_at_replan: int | None = None
    contact_at_replan: bool | None = None
    stale_audio_drops = 0
    stale_action_drops = 0

    def snapshot(now_ms: int, kind: str) -> None:
        branch = branches[active_branch]
        table_rows.append(
            {
                "at_ms": now_ms,
                "kind": kind,
                "branch_id": active_branch,
                "status": str(branch["status"]),
                "audio_available_at_ms": now_ms if kind != "boot" else audio_delay_ms,
                "action_available_at_ms": (
                    int(in_flight["available_at_ms"])
                    if in_flight is not None
                    else now_ms
                ),
                "audio_mode": str(branch["audio_mode"]),
                "action_mode": str(branch["action_mode"]),
                "pending_pcm": len(pending_pcm),
                "played_pcm": len(played_pcm),
                "remaining_steps": len(remaining),
                "executed_steps": len(executed),
                "contact_occurred": contact_occurred,
            },
        )

    def start_action_inference(now_ms: int) -> None:
        nonlocal in_flight
        branch = branches[active_branch]
        if (
            in_flight is not None
            or remaining
            or branch["action_mode"] != "GENERATING"
            or branch["status"] != "active"
        ):
            return
        in_flight = {
            "chunk_id": active_branch,
            "branch_id": active_branch,
            "t_obs_ms": now_ms,
            "available_at_ms": now_ms + action_delay_ms,
        }
        trace.append(
            {
                "at_ms": now_ms,
                "kind": "infer_start",
                "branch_id": active_branch,
            },
        )

    def drop_remaining(reason: str, now_ms: int) -> int:
        nonlocal remaining
        dropped = 0
        for step_index in remaining:
            discarded_steps.append(
                {
                    "at_ms": now_ms,
                    "branch_id": active_branch,
                    "step_index": int(step_index),
                    "reason": reason,
                },
            )
            dropped += 1
        remaining = []
        return dropped

    def cancel_pending_pcm(reason: str, now_ms: int) -> int:
        nonlocal pending_pcm
        canceled = 0
        kept: list[dict[str, int]] = []
        for item in pending_pcm:
            if int(item["branch_id"]) == active_branch:
                canceled_pcm.append({**item, "reason": reason, "at_ms": now_ms})
                canceled += 1
            else:
                kept.append(item)
        pending_pcm = kept
        return canceled

    start_action_inference(0)
    snapshot(0, "boot")

    for now_ms in range(0, end_ms + 1, TICK_MS):
        for available_at, sequence_no, channel, action in _pop_due(event_queue, now_ms):
            if channel not in {"audio", "action"}:
                raise ValueError("channel must be audio or action")
            consumed.append(
                {
                    "available_at_ms": available_at,
                    "consumed_at_ms": now_ms,
                    "sequence_no": sequence_no,
                    "channel": channel,
                    "action": action,
                },
            )
            branch = branches[active_branch]
            if action == PAUSE and channel == "audio" and branch["audio_mode"] == "SPEAKING":
                branch["audio_mode"] = "PAUSED"
                paused_audio_branch = active_branch
            elif action == CONTINUE and channel == "audio" and branch["audio_mode"] == "PAUSED":
                branch["audio_mode"] = "SPEAKING"
            elif action == PAUSE and channel == "action" and branch["action_mode"] == "GENERATING":
                branch["action_mode"] = "PAUSED"
            elif action == CONTINUE and channel == "action" and branch["action_mode"] == "PAUSED":
                branch["action_mode"] = "GENERATING"
            elif action == FORCE_CUTOFF and channel == "action":
                drop_remaining("force_cutoff", now_ms)
                if in_flight is not None and int(in_flight["branch_id"]) == active_branch:
                    in_flight = None
                branch["action_mode"] = "SAFE_HOLD"
                force_at = now_ms
            elif action == REPLAN:
                played_at_replan = len(played_pcm)
                contact_at_replan = contact_occurred
                cancel_pending_pcm("replan", now_ms)
                drop_remaining("replan", now_ms)
                if in_flight is not None:
                    in_flight = None
                branch["status"] = "superseded"
                parent = active_branch
                active_branch = max(branches) + 1
                branches[active_branch] = {
                    "parent": parent,
                    "status": "active",
                    "audio_mode": "SPEAKING",
                    "action_mode": "GENERATING",
                }
                replan_at = now_ms
                if illegal_undo:
                    played_pcm.clear()
                    contact_occurred = False
                    contact_undone = True
                    played_undone = True
                start_action_inference(now_ms)
            trace.append(
                {
                    "at_ms": now_ms,
                    "kind": "control",
                    "channel": channel,
                    "action": action,
                    "branch_id": active_branch,
                },
            )
            snapshot(now_ms, "control")

        branch = branches[active_branch]
        if (
            in_flight is not None
            and now_ms >= int(in_flight["available_at_ms"])
            and not remaining
        ):
            delay_ms = int(in_flight["available_at_ms"]) - int(in_flight["t_obs_ms"])
            expiry_ms = int(in_flight["t_obs_ms"]) + window_ms
            if now_ms > expiry_ms or action_is_stale(delay_ms, horizon, freq_hz):
                stale_action_drops += 1
                discarded_steps.append(
                    {
                        "at_ms": now_ms,
                        "branch_id": int(in_flight["branch_id"]),
                        "step_index": -1,
                        "reason": "stale",
                    },
                )
                in_flight = None
            elif int(in_flight["branch_id"]) != active_branch:
                in_flight = None
            else:
                remaining = list(range(horizon))
                in_flight = None
                trace.append(
                    {
                        "at_ms": now_ms,
                        "kind": "chunk_ready",
                        "branch_id": active_branch,
                    },
                )

        if (
            now_ms % AUDIO_FRAME_MS == 0
            and branch["status"] == "active"
            and branch["audio_mode"] == "SPEAKING"
        ):
            frame_available = now_ms + audio_delay_ms
            if audio_is_stale(audio_delay_ms):
                stale_audio_drops += 1
                canceled_pcm.append(
                    {
                        "branch_id": active_branch,
                        "frame_id": next_frame_id,
                        "play_at_ms": frame_available + PCM_PLAY_LAG_MS,
                        "reason": "stale",
                        "at_ms": now_ms,
                    },
                )
            else:
                pending_pcm.append(
                    {
                        "branch_id": active_branch,
                        "frame_id": next_frame_id,
                        "play_at_ms": frame_available + PCM_PLAY_LAG_MS,
                    },
                )
            next_frame_id += 1

        ready_pcm: list[dict[str, int]] = []
        still_pcm: list[dict[str, int]] = []
        for item in pending_pcm:
            owner = branches[int(item["branch_id"])]
            due = int(item["play_at_ms"]) <= now_ms
            if due and owner["status"] == "active" and owner["audio_mode"] == "SPEAKING":
                ready_pcm.append(item)
            elif due and owner["status"] == "superseded":
                canceled_pcm.append({**item, "reason": "superseded", "at_ms": now_ms})
            elif due and owner["audio_mode"] == "PAUSED":
                still_pcm.append(item)
            else:
                still_pcm.append(item)
        pending_pcm = still_pcm
        for item in ready_pcm:
            played_pcm.append({**item, "played_at_ms": now_ms})
            trace.append(
                {
                    "at_ms": now_ms,
                    "kind": "played",
                    "branch_id": int(item["branch_id"]),
                    "frame_id": int(item["frame_id"]),
                },
            )

        if (
            now_ms % dt_ms == 0
            and branch["status"] == "active"
            and branch["action_mode"] == "GENERATING"
            and remaining
        ):
            step_index = remaining.pop(0)
            executed.append(
                {
                    "at_ms": now_ms,
                    "branch_id": active_branch,
                    "step_index": step_index,
                },
            )
            if step_index == CONTACT_STEP:
                contact_occurred = True
            trace.append(
                {
                    "at_ms": now_ms,
                    "kind": "execute",
                    "branch_id": active_branch,
                    "step_index": step_index,
                },
            )
            if not remaining:
                start_action_inference(now_ms)
        elif (
            now_ms % dt_ms == 0
            and branch["status"] == "active"
            and branch["action_mode"] == "GENERATING"
            and not remaining
            and in_flight is None
        ):
            start_action_inference(now_ms)

    snapshot(end_ms, "final")
    return {
        "freq_hz": freq_hz,
        "horizon": horizon,
        "dt_ms": dt_ms,
        "window_ms": window_ms,
        "audio_block_ms": AUDIO_BLOCK_MS,
        "audio_frame_ms": AUDIO_FRAME_MS,
        "action_delay_ms": action_delay_ms,
        "audio_delay_ms": audio_delay_ms,
        "branches": branches,
        "table_rows": table_rows,
        "consumed": consumed,
        "played_pcm": played_pcm,
        "pending_pcm": pending_pcm,
        "canceled_pcm": canceled_pcm,
        "executed": executed,
        "remaining": remaining,
        "discarded_steps": discarded_steps,
        "contact_occurred": contact_occurred,
        "contact_undone": contact_undone,
        "played_undone": played_undone,
        "paused_audio_branch": paused_audio_branch,
        "replan_at": replan_at,
        "force_at": force_at,
        "played_at_replan": played_at_replan,
        "contact_at_replan": contact_at_replan,
        "stale_audio_drops": stale_audio_drops,
        "stale_action_drops": stale_action_drops,
        "active_branch": active_branch,
        "trace": trace,
    }


def _digest(trace: list[dict[str, int | str]]) -> str:
    return json.dumps(trace, sort_keys=True, separators=(",", ":"))


def run() -> dict[str, Any]:
    pause_then_replan = replay(
        [
            (180, "audio", PAUSE),
            (360, "action", REPLAN),
        ],
    )
    pause_then_replan_again = replay(
        [
            (180, "audio", PAUSE),
            (360, "action", REPLAN),
        ],
    )
    force_cutoff = replay([(220, "action", FORCE_CUTOFF)])
    illegal = replay(
        [
            (180, "audio", PAUSE),
            (360, "action", REPLAN),
        ],
        illegal_undo=True,
    )
    stale_action = replay(
        [],
        action_delay_ms=open_loop_window_ms(HORIZON, FREQ_HZ),
        end_ms=700,
    )
    stale_audio_only = audio_is_stale(AUDIO_BLOCK_MS) and not action_is_stale(
        AUDIO_BLOCK_MS,
        HORIZON,
        FREQ_HZ,
    )

    replan_at = int(pause_then_replan["replan_at"])
    old_pcm_after = [
        item
        for item in pause_then_replan["played_pcm"]
        if int(item["branch_id"]) == 1 and int(item["played_at_ms"]) >= replan_at
    ]
    old_steps_after = [
        item
        for item in pause_then_replan["executed"]
        if int(item["branch_id"]) == 1 and int(item["at_ms"]) >= replan_at
    ]
    new_steps_after = [
        item
        for item in pause_then_replan["executed"]
        if int(item["branch_id"]) == 2 and int(item["at_ms"]) >= replan_at
    ]
    pause_canceled = [
        item
        for item in pause_then_replan["canceled_pcm"]
        if item["reason"] == "replan"
    ]
    steps_during_audio_pause = [
        item
        for item in pause_then_replan["executed"]
        if 180 <= int(item["at_ms"]) < 360
    ]
    pcm_played_during_audio_pause = [
        item
        for item in pause_then_replan["played_pcm"]
        if 180 <= int(item["played_at_ms"]) < 360
        and int(item["branch_id"]) == 1
    ]
    force_audio_played_after = [
        item
        for item in force_cutoff["played_pcm"]
        if int(item["played_at_ms"]) >= 220
    ]
    consumed = pause_then_replan["consumed"]
    first_row = pause_then_replan["table_rows"][0]
    control_rows = [
        row for row in pause_then_replan["table_rows"] if row["kind"] == "control"
    ]
    columns_present = all(
        "audio_available_at_ms" in row and "action_available_at_ms" in row
        for row in pause_then_replan["table_rows"]
    )
    two_event_channels = [str(event["channel"]) for event in consumed]

    checks = {
        "state_table_has_two_timestamp_columns": columns_present
        and "audio_available_at_ms" in first_row
        and "action_available_at_ms" in first_row,
        "audio_and_action_expiry_use_different_windows": stale_audio_only
        and pause_then_replan["window_ms"] == 400
        and pause_then_replan["audio_block_ms"] == 320,
        "events_never_consumed_before_own_available_at": all(
            int(event["consumed_at_ms"]) >= int(event["available_at_ms"])
            for event in consumed
        ),
        "pause_and_replan_are_two_channel_events": two_event_channels
        == ["audio", "action"]
        and consumed[0]["action"] == PAUSE
        and consumed[1]["action"] == REPLAN,
        "audio_pause_does_not_drop_remaining_action_steps": len(steps_during_audio_pause)
        > 0
        and pause_then_replan["paused_audio_branch"] == 1,
        "audio_pause_does_not_play_pending_pcm": pcm_played_during_audio_pause == [],
        "replan_cancels_unplayed_old_pcm": len(pause_canceled) > 0
        and old_pcm_after == [],
        "replan_does_not_execute_old_remaining_steps": old_steps_after == []
        and len(new_steps_after) > 0,
        "replan_does_not_undo_played_pcm_or_contact": (
            int(pause_then_replan["played_at_replan"] or 0) > 0
            and len(pause_then_replan["played_pcm"])
            >= int(pause_then_replan["played_at_replan"] or 0)
            and pause_then_replan["contact_occurred"] is True
            and pause_then_replan["contact_at_replan"] is True
            and pause_then_replan["contact_undone"] is False
            and pause_then_replan["played_undone"] is False
        ),
        "force_cutoff_is_not_audio_pause": (
            force_cutoff["force_at"] == 220
            and force_cutoff["branches"][1]["action_mode"] == "SAFE_HOLD"
            and force_cutoff["branches"][1]["audio_mode"] == "SPEAKING"
            and len(force_audio_played_after) > 0
            and force_cutoff["remaining"] == []
        ),
        "stale_action_chunk_is_not_executed": stale_action["stale_action_drops"] > 0
        and stale_action["executed"] == [],
        "illegal_undo_rewinds_history_and_must_be_rejected": (
            illegal["played_undone"] is True
            and illegal["contact_undone"] is True
            and int(illegal["played_at_replan"] or 0) > 0
            and all(int(item["branch_id"]) != 1 for item in illegal["played_pcm"])
        ),
        "event_replay_is_deterministic": pause_then_replan["trace"]
        == pause_then_replan_again["trace"]
        and _digest(pause_then_replan["trace"])
        == _digest(pause_then_replan_again["trace"]),
        "control_rows_keep_separate_clocks": len(control_rows) == 2
        and control_rows[0]["audio_mode"] == "PAUSED"
        and control_rows[0]["action_mode"] == "GENERATING",
    }

    return {
        "summary": (
            "一张状态表两列时间戳：音频过期用 320 ms 块，动作过期用 H/f。"
            "PAUSE 只停未播 PCM 出队，不丢手臂剩余步；"
            "REPLAN 后旧 PCM 与旧剩余步都不执行，且不撤回已播音频和已发生接触。"
        ),
        "metrics": {
            "audio_block_ms": AUDIO_BLOCK_MS,
            "audio_frame_ms": AUDIO_FRAME_MS,
            "open_loop_window_ms": pause_then_replan["window_ms"],
            "freq_hz": FREQ_HZ,
            "horizon": HORIZON,
            "played_pcm": len(pause_then_replan["played_pcm"]),
            "canceled_pcm": len(pause_then_replan["canceled_pcm"]),
            "executed_steps": len(pause_then_replan["executed"]),
            "discarded_steps": len(pause_then_replan["discarded_steps"]),
            "old_pcm_played_after_replan": len(old_pcm_after),
            "old_steps_after_replan": len(old_steps_after),
            "new_steps_after_replan": len(new_steps_after),
            "contact_occurred": int(pause_then_replan["contact_occurred"]),
            "contact_undone": int(pause_then_replan["contact_undone"]),
            "force_remaining_steps": len(force_cutoff["remaining"]),
            "force_audio_played_after": len(force_audio_played_after),
            "stale_action_executed": len(stale_action["executed"]),
            "table_row_count": len(pause_then_replan["table_rows"]),
            "event_count": len(consumed),
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="48",
    title="把语音打断和手臂重规划接到一张状态表",
    question="CONTINUE / PAUSE / REPLAN 在音频时钟和控制时钟上能否共用一行状态，却必须分成两列时间戳？",
    run=run,
)
