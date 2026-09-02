"""North Harbor 14-day hire: prequential stream, four channels, three curves.

This is the post-lesson-24 protocol. It is not a 25th course experiment.
Lesson 24 still answers a smaller question (frozen vs learner retention).
This module asks the lesson-16 question on a calendar: which writer
survives once the world changes and today's context is withdrawn.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

Channel = Literal["frozen", "rag", "memory_skill", "full"]
Kind = Literal["fact", "document", "procedure", "reasoning", "ephemeral"]

DAYS = 14
SEAT_CHANGE_DAY = 7
TOOL_DAY = 8
RULE_CHANGE_DAY = 9
CHANNELS: tuple[Channel, ...] = ("frozen", "rag", "memory_skill", "full")

HANDBOOK = {
    "deploy": "run freeze then deploy then rollback_on_fail",
    "nightly_sync": "drain queue then nightly_sync then ack",
    "hours_rule": "score equals hours",
    "loss_rule": "score equals 2*hours + 3*loss",
}


def true_score(hours: float, loss: float, rule: str) -> float:
    if rule == "hours":
        return hours
    return 2.0 * hours + 3.0 * loss


def world_on(day: int) -> dict[str, Any]:
    rule = "loss" if day >= RULE_CHANGE_DAY else "hours"
    tools = ["deploy"]
    if day >= TOOL_DAY:
        tools.append("nightly_sync")
    return {
        "seats": {
            "xiaowang": "B7" if day >= SEAT_CHANGE_DAY else "A3",
            "xiaoli": "C2",
        },
        "projects": {"beiji": "north-wing"},
        "tools": tools,
        "rule": rule,
    }


def _ticket(day: int) -> tuple[float, float]:
    hours = float((day % 5) + 1)
    loss = float((day % 3) + 1)
    return hours, loss


def build_stream() -> list[dict[str, Any]]:
    """One mixed stream. No task bell. Predict before write."""
    events: list[dict[str, Any]] = []
    index = 0
    for day in range(1, DAYS + 1):
        world = world_on(day)
        hours, loss = _ticket(day)
        day_events: list[dict[str, Any]] = [
            {
                "id": index,
                "day": day,
                "kind": "fact",
                "name": "call_xiaowang",
                "key": "seat:xiaowang",
                "value": world["seats"]["xiaowang"],
            },
            {
                "id": index + 1,
                "day": day,
                "kind": "document",
                "name": "handbook_deploy",
                "key": "doc:deploy",
                "value": HANDBOOK["deploy"],
            },
            {
                "id": index + 2,
                "day": day,
                "kind": "procedure",
                "name": "run_deploy",
                "key": "skill:deploy",
                "value": "deploy",
            },
            {
                "id": index + 3,
                "day": day,
                "kind": "reasoning",
                "name": "score_ticket",
                "hours": hours,
                "loss": loss,
                "value": true_score(hours, loss, world["rule"]),
                "rule": world["rule"],
            },
            {
                "id": index + 4,
                "day": day,
                "kind": "ephemeral",
                "name": "standup_note",
                "key": "note:standup",
                "value": f"stand-up day {day}",
            },
        ]
        if day >= TOOL_DAY:
            day_events.append(
                {
                    "id": index + 5,
                    "day": day,
                    "kind": "procedure",
                    "name": "run_nightly",
                    "key": "skill:nightly_sync",
                    "value": "nightly_sync",
                },
            )
        if day == SEAT_CHANGE_DAY:
            day_events.insert(
                1,
                {
                    "id": index + 9,
                    "day": day,
                    "kind": "fact",
                    "name": "seat_moved",
                    "key": "seat:xiaowang",
                    "value": "B7",
                },
            )
        events.extend(day_events)
        index += 10
    return events


class Store:
    def __init__(self, channel: Channel) -> None:
        self.channel = channel
        self.working: dict[str, str] = {}
        self.semantic: dict[str, str] = {}
        self.skills: set[str] = set()
        self.samples: list[tuple[float, float, float]] = []
        self.weights: tuple[float, float] | None = None
        self.context: dict[str, str] = {}

    def flush_day(self) -> None:
        self.working.clear()
        self.context.clear()

    def _fit(self) -> None:
        if len(self.samples) < 2:
            return
        gram00 = sum(h * h for h, _, _ in self.samples)
        gram01 = sum(h * loss for h, loss, _ in self.samples)
        gram11 = sum(loss * loss for _, loss, _ in self.samples)
        rhs0 = sum(h * score for h, _, score in self.samples)
        rhs1 = sum(loss * score for _, loss, score in self.samples)
        det = gram00 * gram11 - gram01 * gram01
        if abs(det) < 1e-9:
            return
        w0 = (gram11 * rhs0 - gram01 * rhs1) / det
        w1 = (gram00 * rhs1 - gram01 * rhs0) / det
        self.weights = (w0, w1)

    def predict(self, event: dict[str, Any]) -> Any:
        kind = event["kind"]
        if kind == "fact":
            return (
                self.semantic.get(event["key"])
                or self.working.get(event["key"])
                or self.context.get(event["key"])
            )
        if kind == "document":
            return self.context.get(event["key"]) or self.semantic.get(event["key"])
        if kind == "procedure":
            return event["value"] in self.skills
        if kind == "reasoning":
            if self.weights is not None:
                w0, w1 = self.weights
                return w0 * event["hours"] + w1 * event["loss"]
            if "doc:rule" in self.context:
                return true_score(event["hours"], event["loss"], event["rule"])
            return event["hours"]
        return self.working.get(event["key"])

    def write(self, event: dict[str, Any]) -> None:
        channel = self.channel
        kind = event["kind"]
        if channel == "frozen":
            return
        if channel == "rag":
            if kind in {"document", "fact", "ephemeral"}:
                self.context[event["key"]] = str(event["value"])
            if kind == "reasoning":
                self.context["doc:rule"] = event["rule"]
            return
        if kind == "ephemeral":
            self.working[event["key"]] = str(event["value"])
            return
        if kind == "fact":
            self.semantic[event["key"]] = str(event["value"])
            return
        if kind == "document":
            self.semantic[event["key"]] = str(event["value"])
            return
        if kind == "procedure" and channel in {"memory_skill", "full"}:
            self.skills.add(str(event["value"]))
            return
        if kind == "reasoning" and channel == "full":
            self.samples.append(
                (float(event["hours"]), float(event["loss"]), float(event["value"])),
            )
            self.samples = self.samples[-6:]
            self._fit()


def _correct(event: dict[str, Any], prediction: Any) -> bool:
    kind = event["kind"]
    if kind == "fact":
        return prediction == event["value"]
    if kind == "document":
        return prediction == event["value"]
    if kind == "procedure":
        return bool(prediction)
    if kind == "reasoning":
        if prediction is None:
            return False
        return abs(float(prediction) - float(event["value"])) < 1e-6
    return prediction == event["value"]


def run_channel(channel: Channel) -> dict[str, Any]:
    store = Store(channel)
    stream = build_stream()
    day_cursor = 1
    first_tool_attempts = 0
    later_tool_attempts = 0
    tool_seen = 0
    fact_hits_after_change: list[int] = []
    rule_hits_after_change: list[int] = []
    preq_hits: list[int] = []
    for event in stream:
        if event["day"] != day_cursor:
            store.flush_day()
            day_cursor = int(event["day"])
        prediction = store.predict(event)
        hit = _correct(event, prediction)
        preq_hits.append(int(hit))
        if event["name"] == "call_xiaowang" and event["day"] >= SEAT_CHANGE_DAY:
            fact_hits_after_change.append(int(hit))
        if event["kind"] == "reasoning" and event["day"] >= RULE_CHANGE_DAY:
            rule_hits_after_change.append(int(hit))
        if event["name"] == "run_nightly":
            tool_seen += 1
            if tool_seen == 1:
                first_tool_attempts = 1 if hit else 4
            elif tool_seen == 2:
                later_tool_attempts = 1 if hit else 4
        store.write(event)

    store.flush_day()
    end_world = world_on(DAYS)
    seat_probe = store.semantic.get("seat:xiaowang") == end_world["seats"]["xiaowang"]
    note_probe = store.working.get("note:standup") == f"stand-up day {DAYS}"
    deploy_skill = "deploy" in store.skills
    nightly_skill = "nightly_sync" in store.skills
    rule_ok = False
    if store.weights is not None:
        w0, w1 = store.weights
        pred = w0 * 4.0 + w1 * 2.0
        rule_ok = abs(pred - true_score(4.0, 2.0, "loss")) < 1e-6

    return {
        "channel": channel,
        "preq_accuracy": sum(preq_hits) / len(preq_hits),
        "seat_probe": seat_probe,
        "note_leaked": note_probe,
        "deploy_skill": deploy_skill,
        "nightly_skill": nightly_skill,
        "rule_ok": rule_ok,
        "fact_after_change": (
            sum(fact_hits_after_change) / len(fact_hits_after_change)
            if fact_hits_after_change
            else 0.0
        ),
        "rule_after_change": (
            sum(rule_hits_after_change) / len(rule_hits_after_change)
            if rule_hits_after_change
            else 0.0
        ),
        "first_tool_attempts": first_tool_attempts,
        "later_tool_attempts": later_tool_attempts,
        "semantic_size": len(store.semantic),
        "skill_count": len(store.skills),
        "has_weights": store.weights is not None,
        "preq_hits": preq_hits,
    }


def run_hire() -> dict[str, Any]:
    results = {channel: run_channel(channel) for channel in CHANNELS}
    frozen = results["frozen"]
    rag = results["rag"]
    memory = results["memory_skill"]
    full = results["full"]

    checks = {
        "prequential_frozen_near_chance": frozen["preq_accuracy"] < 0.35,
        "rag_cannot_keep_seat": rag["seat_probe"] is False,
        "memory_keeps_new_seat": memory["seat_probe"] is True,
        "full_keeps_new_seat": full["seat_probe"] is True,
        "memory_fails_new_rule": memory["rule_ok"] is False,
        "full_learns_new_rule": full["rule_ok"] is True,
        "frozen_has_no_skills": frozen["skill_count"] == 0,
        "skill_channels_store_deploy": memory["deploy_skill"] and full["deploy_skill"],
        "later_tool_faster_than_first": (
            full["later_tool_attempts"] < full["first_tool_attempts"]
        ),
        "standup_does_not_survive": (
            (not memory["note_leaked"]) and (not full["note_leaked"])
        ),
        "full_beats_frozen_preq": full["preq_accuracy"] > frozen["preq_accuracy"] + 0.15,
        "memory_beats_rag_on_seat": memory["fact_after_change"]
        > rag["fact_after_change"] + 0.3,
    }
    summary = (
        "14 日预衡上岗：冻结预衡准确率 "
        f"{frozen['preq_accuracy']:.3f}；RAG 撤掉当日手册后座位探针为假；"
        f"记忆通道叫到 B7；只有 full 拟合出 2h+3loss。"
        f"技能通道后期夜间同步尝试 {full['later_tool_attempts']}，"
        f"首次 {full['first_tool_attempts']}。"
    )
    return {
        "schema": {"name": "learn-cl-hire-result", "version": 1},
        "title": "北港文具 14 日上岗",
        "question": "四条写入通道在世界会变、当日上下文会撤掉时，各保住哪一类经验？",
        "summary": summary,
        "metrics": {
            "frozen": {key: value for key, value in frozen.items() if key != "preq_hits"},
            "rag": {key: value for key, value in rag.items() if key != "preq_hits"},
            "memory_skill": {
                key: value for key, value in memory.items() if key != "preq_hits"
            },
            "full": {key: value for key, value in full.items() if key != "preq_hits"},
            "seat_change_day": SEAT_CHANGE_DAY,
            "tool_day": TOOL_DAY,
            "rule_change_day": RULE_CHANGE_DAY,
        },
        "checks": checks,
    }


def write_hire(output_root: Path) -> Path:
    payload = run_hire()
    destination = output_root / "capstone" / "result.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination
