from __future__ import annotations

from typing import Any

from ..core import LessonExperiment


Prediction = dict[str, int | str]
Reference = dict[str, int | str | list[str]]


def _compatible(prediction: Prediction, reference: Reference) -> bool:
    action = str(prediction["action"])
    acceptable = [str(item) for item in reference.get("acceptable", [])]
    action_ok = action == reference["preferred"] or action in acceptable
    earliest = int(reference["anchor_ms"]) - int(reference["before_ms"])
    latest = int(reference["anchor_ms"]) + int(reference["after_ms"])
    return action_ok and earliest <= int(prediction["at_ms"]) <= latest


def _match(
    predictions: list[Prediction],
    references: list[Reference],
) -> dict[str, Any]:
    best_pairs: list[tuple[int, int]] = []
    best_cost = 10**18

    def visit(
        prediction_index: int,
        used_references: set[int],
        pairs: list[tuple[int, int]],
        cost: int,
    ) -> None:
        nonlocal best_pairs, best_cost
        if prediction_index == len(predictions):
            if len(pairs) > len(best_pairs) or (
                len(pairs) == len(best_pairs) and cost < best_cost
            ):
                best_pairs = list(pairs)
                best_cost = cost
            return
        visit(prediction_index + 1, used_references, pairs, cost)
        for reference_index, reference in enumerate(references):
            if reference_index in used_references:
                continue
            if not _compatible(predictions[prediction_index], reference):
                continue
            delta = abs(
                int(predictions[prediction_index]["at_ms"])
                - int(reference["anchor_ms"]),
            )
            used_references.add(reference_index)
            pairs.append((prediction_index, reference_index))
            visit(prediction_index + 1, used_references, pairs, cost + delta)
            pairs.pop()
            used_references.remove(reference_index)

    visit(0, set(), [], 0)
    used_predictions = {prediction for prediction, _ in best_pairs}
    used_references = {reference for _, reference in best_pairs}
    return {
        "pairs": best_pairs,
        "cost_ms": 0 if not best_pairs else best_cost,
        "spurious": [
            index for index in range(len(predictions)) if index not in used_predictions
        ],
        "missed": [
            index for index in range(len(references)) if index not in used_references
        ],
    }


def _hysteresis(probabilities: list[dict[str, float]]) -> dict[str, Any]:
    enter = {"BARGE_IN": 0.8, "TAKE_TURN": 0.7, "BACKCHANNEL": 0.65}
    exit_ = {"BARGE_IN": 0.55, "TAKE_TURN": 0.45, "BACKCHANNEL": 0.40}
    required = {"BARGE_IN": 2, "TAKE_TURN": 3, "BACKCHANNEL": 1}
    cooldown = {"BARGE_IN": 2, "TAKE_TURN": 1, "BACKCHANNEL": 3}
    priority = ("BARGE_IN", "TAKE_TURN", "BACKCHANNEL")
    streaks = {action: 0 for action in priority}
    active = {action: False for action in priority}
    last_emitted = {action: -10**9 for action in priority}
    events: list[tuple[int, str]] = []
    timeline: list[tuple[int, str]] = []
    transitions: list[dict[str, int | str]] = []
    for frame, scores in enumerate(probabilities):
        for action in priority:
            score = scores.get(action, 0.0)
            if active[action] and score < exit_[action]:
                active[action] = False
                transitions.append(
                    {"frame": frame, "action": action, "state": "exit"},
                )
            if not active[action] and score > enter[action]:
                streaks[action] += 1
            else:
                streaks[action] = 0
        emitted_action = "HOLD"
        for action in priority:
            cooldown_ready = frame - last_emitted[action] >= cooldown[action]
            if (
                not active[action]
                and streaks[action] >= required[action]
                and cooldown_ready
            ):
                events.append((frame, action))
                emitted_action = action
                active[action] = True
                last_emitted[action] = frame
                transitions.append(
                    {"frame": frame, "action": action, "state": "enter"},
                )
                streaks = {name: 0 for name in priority}
                break
        timeline.append((frame, emitted_action))
    return {
        "events": events,
        "timeline": timeline,
        "transitions": transitions,
        "enter_thresholds": enter,
        "exit_thresholds": exit_,
    }


def run() -> dict[str, Any]:
    reference = {
        "preferred": "TAKE_TURN",
        "acceptable": [],
        "anchor_ms": 1000,
        "before_ms": 100,
        "after_ms": 300,
    }
    predictions = [
        {"action": "TAKE_TURN", "at_ms": 980},
        {"action": "TAKE_TURN", "at_ms": 1040},
    ]
    single_match = _match(predictions, [reference])

    overlapping = _match(
        [
            {"action": "TAKE_TURN", "at_ms": 1010},
            {"action": "TAKE_TURN", "at_ms": 1070},
        ],
        [
            {
                "preferred": "TAKE_TURN",
                "acceptable": [],
                "anchor_ms": 1000,
                "before_ms": 80,
                "after_ms": 100,
            },
            {
                "preferred": "TAKE_TURN",
                "acceptable": [],
                "anchor_ms": 1080,
                "before_ms": 100,
                "after_ms": 80,
            },
        ],
    )
    incompatible = _match(
        [{"action": "BARGE_IN", "at_ms": 1000}],
        [reference],
    )
    empty = _match([], [])

    probabilities = [
        {"BACKCHANNEL": 0.70},
        {"BACKCHANNEL": 0.72},
        {},
        {"BACKCHANNEL": 0.71},
        {"TAKE_TURN": 0.75},
        {"TAKE_TURN": 0.76},
        {"TAKE_TURN": 0.77},
        {"BARGE_IN": 0.85},
        {"BARGE_IN": 0.86},
    ]
    policy = _hysteresis(probabilities)
    emitted = policy["events"]
    timeline = policy["timeline"]

    checks = {
        "one_reference_matches_only_once": len(single_match["pairs"]) == 1
        and len(single_match["spurious"]) == 1,
        "closest_prediction_wins": single_match["pairs"] == [(0, 0)]
        and single_match["cost_ms"] == 20,
        "overlapping_windows_use_global_one_to_one_assignment": len(
            overlapping["pairs"],
        )
        == 2,
        "incompatible_action_is_rejected": incompatible["pairs"] == []
        and incompatible["missed"] == [0],
        "empty_session_is_well_defined": empty
        == {"pairs": [], "cost_ms": 0, "spurious": [], "missed": []},
        "take_turn_requires_three_frames": (6, "TAKE_TURN") in emitted,
        "barge_in_requires_two_frames": (8, "BARGE_IN") in emitted,
        "backchannel_cooldown_prevents_repeat": [
            frame for frame, action in emitted if action == "BACKCHANNEL"
        ]
        == [0, 3],
        "enter_thresholds_are_stricter_than_exit_thresholds": all(
            policy["enter_thresholds"][action]
            > policy["exit_thresholds"][action]
            for action in policy["enter_thresholds"]
        ),
        "hold_means_no_new_floor_action": (
            timeline[1] == (1, "HOLD")
            and timeline[2] == (2, "HOLD")
            and len(timeline) == len(probabilities)
            and [
                (frame, action)
                for frame, action in timeline
                if action != "HOLD"
            ]
            == emitted
        ),
    }
    return {
        "summary": (
            "用全局一对一事件匹配避免重复计分，再对固定概率序列执行连续帧门槛"
            "、高进入/低退出阈值和 cooldown；没有新动作的帧明确输出 HOLD。"
        ),
        "metrics": {
            "predictions": len(predictions),
            "references": 1,
            "matched": len(single_match["pairs"]),
            "spurious": len(single_match["spurious"]),
            "matching_cost_ms": single_match["cost_ms"],
            "emitted_events": [
                {"frame": frame, "action": action}
                for frame, action in emitted
            ],
            "frame_actions": [
                {"frame": frame, "action": action}
                for frame, action in timeline
            ],
            "hysteresis_transitions": policy["transitions"],
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="06",
    title="实现学习式 Turn Policy 的事件规则",
    question="如何避免重复命中，并把概率稳定地转换成话轮动作？",
    run=run,
)
