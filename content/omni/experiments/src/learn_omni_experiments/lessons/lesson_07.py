from __future__ import annotations

import hashlib
import heapq
import json
from typing import Any

from ..core import LessonExperiment


def _simulate() -> dict[str, Any]:
    event_heap = [
        (20, 2, "CONTINUE"),
        (20, 1, "CONTINUE"),
        (40, 3, "PAUSE"),
        (60, 4, "CONTINUE"),
        (80, 5, "REPLAN"),
    ]
    heapq.heapify(event_heap)
    branches: dict[int, dict[str, Any]] = {
        1: {"parent": None, "status": "active", "tokens": 0},
    }
    active_branch = 1
    pending_audio: list[dict[str, int | str]] = []
    played_audio: list[dict[str, int | str]] = []
    consumed_events: list[dict[str, int | str]] = []
    emitted_tokens: list[dict[str, int | str]] = []
    trace: list[dict[str, int | str]] = []
    canceled_audio = 0
    canceled_on_pause = 0
    canceled_on_replan = 0
    pause_pending_count = 0
    paused_branch: int | None = None
    resumed_branch: int | None = None
    replan_at: int | None = None

    for now_ms in range(0, 131, 10):
        while event_heap and event_heap[0][0] <= now_ms:
            available_at, sequence_no, action = heapq.heappop(event_heap)
            consumed_events.append(
                {
                    "available_at_ms": available_at,
                    "consumed_at_ms": now_ms,
                    "sequence_no": sequence_no,
                    "action": action,
                },
            )
            branch = branches[active_branch]
            if action == "PAUSE" and branch["status"] == "active":
                branch["status"] = "paused"
                paused_branch = active_branch
                pause_pending_count = sum(
                    item["branch_id"] == active_branch
                    for item in pending_audio
                )
            elif action == "CONTINUE" and branch["status"] == "paused":
                branch["status"] = "active"
                resumed_branch = active_branch
            elif action == "REPLAN":
                branch["status"] = "superseded"
                before = len(pending_audio)
                pending_audio = [
                    item for item in pending_audio if item["branch_id"] != active_branch
                ]
                canceled_on_replan += before - len(pending_audio)
                canceled_audio += before - len(pending_audio)
                old_branch = active_branch
                active_branch = max(branches) + 1
                branches[active_branch] = {
                    "parent": old_branch,
                    "status": "active",
                    "tokens": 0,
                }
                replan_at = now_ms
            trace.append(
                {
                    "at_ms": now_ms,
                    "kind": "control",
                    "action": action,
                    "branch_id": active_branch,
                },
            )

        ready: list[dict[str, int | str]] = []
        still_pending: list[dict[str, int | str]] = []
        for item in pending_audio:
            branch_status = str(branches[int(item["branch_id"])]["status"])
            due = int(item["play_at_ms"]) <= now_ms
            if due and branch_status == "active":
                ready.append(item)
            elif due and branch_status == "superseded":
                canceled_audio += 1
            else:
                still_pending.append(item)
        pending_audio = still_pending
        for item in ready:
            branch_id = int(item["branch_id"])
            played_audio.append({**item, "played_at_ms": now_ms})
            trace.append(
                {
                    "at_ms": now_ms,
                    "kind": "played",
                    "branch_id": branch_id,
                },
            )

        branch = branches[active_branch]
        if branch["status"] == "active":
            token_index = int(branch["tokens"])
            token = f"b{active_branch}-t{token_index}"
            branch["tokens"] = token_index + 1
            emitted_tokens.append(
                {
                    "at_ms": now_ms,
                    "branch_id": active_branch,
                    "token": token,
                },
            )
            pending_audio.append(
                {
                    "branch_id": active_branch,
                    "token": token,
                    "play_at_ms": now_ms + 30,
                },
            )
            trace.append(
                {
                    "at_ms": now_ms,
                    "kind": "decode",
                    "branch_id": active_branch,
                    "token": token,
                },
            )

    trace_digest = hashlib.sha256(
        json.dumps(trace, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    ).hexdigest()
    return {
        "branches": branches,
        "consumed_events": consumed_events,
        "emitted_tokens": emitted_tokens,
        "played_audio": played_audio,
        "pending_audio": pending_audio,
        "canceled_audio": canceled_audio,
        "canceled_on_pause": canceled_on_pause,
        "canceled_on_replan": canceled_on_replan,
        "pause_pending_count": pause_pending_count,
        "paused_branch": paused_branch,
        "resumed_branch": resumed_branch,
        "replan_at": replan_at,
        "trace": trace,
        "trace_sha256": trace_digest,
    }


def run() -> dict[str, Any]:
    result = _simulate()
    repeated = _simulate()
    consumed = result["consumed_events"]
    replan_at = int(result["replan_at"])
    old_branch_played_after_replan = [
        item
        for item in result["played_audio"]
        if int(item["branch_id"]) == 1
        and int(item["played_at_ms"]) >= replan_at
    ]
    resumed_pending = [
        item
        for item in result["played_audio"]
        if int(item["branch_id"]) == 1
        and int(item["played_at_ms"]) == 60
        and int(item["play_at_ms"]) <= 60
    ]
    events_at_twenty = [
        int(event["sequence_no"])
        for event in consumed
        if int(event["available_at_ms"]) == 20
    ]
    input_during_output = any(
        int(event["consumed_at_ms"]) > int(result["emitted_tokens"][0]["at_ms"])
        for event in consumed
    )

    checks = {
        "ready_queue_orders_by_time_then_sequence": events_at_twenty == [1, 2],
        "events_are_never_consumed_before_available": all(
            int(event["consumed_at_ms"]) >= int(event["available_at_ms"])
            for event in consumed
        ),
        "input_advances_while_output_exists": input_during_output,
        "pause_resume_keeps_same_branch": result["paused_branch"]
        == result["resumed_branch"]
        == 1,
        "pause_preserves_pending_pcm_without_advancing_cursor": (
            int(result["pause_pending_count"]) > 0
            and int(result["canceled_on_pause"]) == 0
            and len(resumed_pending) == int(result["pause_pending_count"])
        ),
        "replan_creates_child_branch": result["branches"][2]["parent"] == 1
        and result["branches"][1]["status"] == "superseded",
        "replan_cancels_unplayed_old_branch_pcm": (
            int(result["canceled_on_replan"]) > 0
            and int(result["canceled_audio"]) >= int(result["canceled_on_replan"])
        ),
        "superseded_branch_cannot_play_after_replan": old_branch_played_after_replan
        == [],
        "event_replay_is_deterministic": result["trace_sha256"]
        == repeated["trace_sha256"]
        and result["trace"] == repeated["trace"],
    }
    return {
        "summary": (
            "用离散事件运行时同时推进 ingress、decode branch 和可撤销播放队列，"
            "验证 Pause/Resume 保持原分支，Replan 建立新分支并阻止旧输出泄漏。"
        ),
        "metrics": {
            "consumed_events": len(consumed),
            "emitted_tokens": len(result["emitted_tokens"]),
            "played_chunks": len(result["played_audio"]),
            "canceled_chunks": result["canceled_audio"],
            "pause_preserved_chunks": result["pause_pending_count"],
            "replan_canceled_chunks": result["canceled_on_replan"],
            "branch_count": len(result["branches"]),
            "trace_sha256": result["trace_sha256"],
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="07",
    title="实现真双工 Routing 的状态语义",
    question="输入与输出并行推进时，Pause、Resume 和 Replan 如何保持状态一致？",
    run=run,
)
