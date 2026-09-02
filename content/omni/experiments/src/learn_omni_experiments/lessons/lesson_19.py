from __future__ import annotations

import math
from typing import Iterable

from ..core import LessonExperiment


def _byte_spans(text: str, pieces: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    character_cursor = 0
    for piece in pieces:
        start = text.index(piece, character_cursor)
        end = start + len(piece)
        byte_start = len(text[:start].encode("utf-8"))
        byte_end = len(text[:end].encode("utf-8"))
        spans.append((byte_start, byte_end))
        character_cursor = end
    return spans


def _cumulative_byte_ends(pieces: Iterable[str]) -> list[int]:
    prefix = ""
    ends: list[int] = []
    for piece in pieces:
        prefix += piece
        ends.append(len(prefix.encode("utf-8")))
    return ends


def _map_tokens_to_thinker(
    minimind_spans: list[tuple[int, int]],
    thinker_byte_ends: list[int],
) -> list[int]:
    mapping: list[int] = []
    for _, token_end in minimind_spans:
        mapping.append(
            next(
                index
                for index, thinker_end in enumerate(thinker_byte_ends)
                if token_end <= thinker_end
            ),
        )
    return mapping


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _attention(
    query: list[float],
    keys: list[list[float]],
    values: list[list[float]],
    final_visible_index: int,
) -> list[float]:
    visible_keys = keys[: final_visible_index + 1]
    visible_values = values[: final_visible_index + 1]
    scores = [_dot(query, key) for key in visible_keys]
    largest = max(scores)
    weights = [math.exp(score - largest) for score in scores]
    normalizer = sum(weights)
    normalized = [weight / normalizer for weight in weights]
    return [
        sum(
            normalized[index] * visible_values[index][dimension]
            for index in range(len(normalized))
        )
        for dimension in range(len(visible_values[0]))
    ]


def _prefix_monotone(prefixes: list[str]) -> bool:
    return all(
        current.startswith(previous)
        for previous, current in zip(prefixes, prefixes[1:])
    )


def _apply_fixture_events(
    events: list[dict[str, object]],
) -> tuple[str, list[str]]:
    state = "playing"
    states: list[str] = []
    for event in events:
        kind = event["kind"]
        if kind == "INTERRUPT_CANDIDATE" and state == "playing":
            state = "holding"
        elif kind == "FAST_PAUSE" and state == "holding":
            state = "paused"
        elif kind == "KEEP_PLAYING" and state == "holding":
            state = "playing"
        elif kind == "REPLAN" and state == "paused":
            state = "replanning"
        elif kind == "RESUME" and state in {"paused", "replanning"}:
            state = "playing"
        states.append(state)
    return state, states


def run() -> dict[str, object]:
    text = "你好，Omni🙂"
    thinker_pieces = ["你", "好，", "Om", "ni", "🙂"]
    minimind_pieces = ["你好", "，", "O", "mni", "🙂"]
    thinker_byte_ends = _cumulative_byte_ends(thinker_pieces)
    minimind_spans = _byte_spans(text, minimind_pieces)
    token_index = _map_tokens_to_thinker(
        minimind_spans,
        thinker_byte_ends,
    )

    keys = [
        [0.2, 0.1],
        [0.0, 0.3],
        [0.4, -0.2],
        [0.1, 0.5],
        [-0.3, 0.2],
    ]
    values = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, -1.0],
        [-1.0, 2.0],
    ]
    queries = [
        [0.1, 0.2],
        [0.2, 0.1],
        [0.3, -0.1],
        [0.0, 0.4],
        [-0.2, 0.2],
    ]
    offline = [
        _attention(query, keys, values, token_index[index])
        for index, query in enumerate(queries)
    ]
    incremental = [
        _attention(
            query,
            keys[: visible + 1],
            values[: visible + 1],
            visible,
        )
        for query, visible in zip(queries, token_index)
    ]
    parity_error = max(
        abs(left - right)
        for offline_row, incremental_row in zip(offline, incremental)
        for left, right in zip(offline_row, incremental_row)
    )

    future_safe = True
    for step, (query, visible) in enumerate(zip(queries, token_index)):
        mutated_values = [
            value.copy() if index <= visible else [999.0, -999.0]
            for index, value in enumerate(values)
        ]
        original = _attention(query, keys, values, visible)
        mutated = _attention(query, keys, mutated_values, visible)
        future_safe = future_safe and all(
            math.isclose(a, b, abs_tol=1e-12)
            for a, b in zip(original, mutated)
        )

    rewrite_detected = not _prefix_monotone(["测", "测试", "测验"])
    normal_prefixes_monotone = _prefix_monotone(
        ["你", "你好", "你好，", "你好，Omni"],
    )

    trainable_flags = {
        "nemotron.layer.0": False,
        "minimind.talker": False,
        "minimind.speaker_projection": False,
        "bridge.cross_attention": True,
        "bridge.gate": True,
    }
    trainable_names = sorted(
        name for name, trainable in trainable_flags.items() if trainable
    )

    interrupt_events = [
        {"id": 1, "time": 0.00, "kind": "AUDIO_FRAME"},
        {"id": 2, "time": 0.10, "kind": "INTERRUPT_CANDIDATE"},
        {"id": 3, "time": 0.21, "kind": "FAST_PAUSE"},
        {"id": 4, "time": 0.28, "kind": "DAC_STOP"},
        {"id": 5, "time": 0.50, "kind": "REPLAN"},
        {"id": 6, "time": 0.65, "kind": "RESUME"},
    ]
    final_state, interrupt_states = _apply_fixture_events(interrupt_events)
    backchannel_events = [
        {"id": 1, "time": 0.00, "kind": "AUDIO_FRAME"},
        {"id": 2, "time": 0.10, "kind": "INTERRUPT_CANDIDATE"},
        {"id": 3, "time": 0.18, "kind": "KEEP_PLAYING"},
    ]
    backchannel_state, _ = _apply_fixture_events(backchannel_events)
    fixture_fast_pause_to_dac_event_delta_ms = round(
        (0.28 - 0.21) * 1000,
    )

    monotonic_events = all(
        previous["time"] <= current["time"]
        and int(current["id"]) == int(previous["id"]) + 1
        for previous, current in zip(
            interrupt_events,
            interrupt_events[1:],
        )
    )

    return {
        "summary": (
            "把两套 toy tokenizer 切分映射到同一 UTF-8 字节轴，并验证 "
            "causal attention 计算的离线/增量一致性。双工部分只是把手工"
            "事件 fixture 送入 toy 状态机；它没有运行播放器、四时钟链路，"
            "也没有证明真实停播、分支恢复或端到端时延。"
        ),
        "metrics": {
            "duplex_scope": (
                "hand_authored_event_fixture_not_player_or_clock_trace"
            ),
            "trainable_flag_scope": "declared_toy_names_not_model_parameters",
            "text_utf8_bytes": len(text.encode("utf-8")),
            "thinker_byte_ends": thinker_byte_ends,
            "minimind_byte_spans": [list(span) for span in minimind_spans],
            "text_token_index": token_index,
            "bridge_parity_max_abs_error": parity_error,
            "trainable_names": trainable_names,
            "fixture_interrupt_states": interrupt_states,
            "fixture_fast_pause_to_dac_event_delta_ms": (
                fixture_fast_pause_to_dac_event_delta_ms
            ),
        },
        "checks": {
            "跨tokenizer映射使用UTF8字节边界": (
                minimind_spans[-1][1] == len(text.encode("utf-8"))
            ),
            "MiniMind位置映射单调不回退": all(
                left <= right
                for left, right in zip(token_index, token_index[1:])
            ),
            "每个MiniMind token终点映射到首个覆盖它的Thinker边界": all(
                span[1] <= thinker_byte_ends[index]
                and (
                    index == 0
                    or thinker_byte_ends[index - 1] < span[1]
                )
                for span, index in zip(minimind_spans, token_index)
            ),
            "正常增量解码保持prefix_monotone": normal_prefixes_monotone,
            "toy prefix检查器能检测重写": rewrite_detected,
            "离线和增量bridge数值一致": parity_error < 1e-12,
            "修改未来value不会影响当前输出": future_safe,
            "给定toy参数flag只打开bridge命名空间": (
                bool(trainable_names)
                and all(name.startswith("bridge.") for name in trainable_names)
            ),
            "手工interrupt fixture经过pause_replan_resume后回到playing": (
                final_state == "playing"
            ),
            "手工backchannel fixture结束时仍为playing": (
                backchannel_state == "playing"
            ),
            "fixture事件id和局部时间单调": monotonic_events,
            "fixture中两个手填事件的局部时差不超过80ms": (
                fixture_fast_pause_to_dac_event_delta_ms <= 80
            ),
        },
    }


LESSON = LessonExperiment(
    lesson_id="19",
    title="Thinker × Talker 因果桥与双工调度",
    question="不同 tokenizer 的时间轴怎样安全对齐，并在打断时保持可重放状态？",
    run=run,
)
