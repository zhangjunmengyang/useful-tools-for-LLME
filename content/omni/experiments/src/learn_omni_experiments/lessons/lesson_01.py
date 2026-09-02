from __future__ import annotations

import random
from typing import Any

from ..core import LessonExperiment


BOS = -2
PAD = -1
IGNORE = -100


def _build_diagonal_targets(codes: list[list[int]]) -> list[list[int]]:
    codebooks = len(codes)
    frames = len(codes[0])
    if any(len(row) != frames for row in codes):
        raise ValueError("all codebooks must contain the same number of frames")
    return [
        [BOS] * codebook + row + [PAD] * (codebooks - codebook - 1)
        for codebook, row in enumerate(codes)
    ]


def _reconstruct_frames(delayed: list[list[int]], frames: int) -> list[list[int]]:
    return [
        [delayed[codebook][frame + codebook] for codebook in range(len(delayed))]
        for frame in range(frames)
    ]


def _toy_forward(
    tokens: list[int],
    generator: random.Random,
    trace: list[dict[str, int]] | None = None,
) -> list[int]:
    """Run one stochastic forward with an RNG owned by the caller."""
    weighted_sum = sum((position + 1) * token for position, token in enumerate(tokens))
    stochastic_draw = generator.randrange(1_000_003)
    logits = [
        weighted_sum + sum(tokens) + stochastic_draw,
        weighted_sum - sum(tokens) - stochastic_draw,
        (weighted_sum + stochastic_draw) % 97,
    ]
    if trace is not None:
        trace.append(
            {
                "token_count": len(tokens),
                "weighted_sum": weighted_sum,
                "stochastic_draw": stochastic_draw,
                "logit_checksum": sum(logits),
            },
        )
    return logits


def run() -> dict[str, Any]:
    canonical = [
        [100 * codebook + frame for frame in range(4)]
        for codebook in range(8)
    ]
    delayed = _build_diagonal_targets(canonical)
    reconstructed = _reconstruct_frames(delayed, frames=4)

    token_roles = [
        "system",
        "user",
        "user",
        "assistant",
        "assistant",
        "assistant",
        "padding",
    ]
    token_ids = [11, 21, 22, 31, 32, 33, 0]
    labels = [
        token if role == "assistant" else IGNORE
        for role, token in zip(token_roles, token_ids)
    ]

    trace: list[dict[str, int]] = []
    reference_generator = random.Random(42)
    traced_generator = random.Random(42)
    logits_without_trace = _toy_forward(token_ids, reference_generator)
    logits_with_trace = _toy_forward(token_ids, traced_generator, trace)
    states_match_after_forward = (
        reference_generator.getstate() == traced_generator.getstate()
    )

    consuming_generator = random.Random(42)
    consuming_generator.random()
    negative_control_logits = _toy_forward(token_ids, consuming_generator, [])
    negative_control_detected = (
        negative_control_logits != logits_without_trace
        or consuming_generator.getstate() != reference_generator.getstate()
    )

    checks = {
        "nine_stream_layout": 1 + len(canonical) == 9,
        "diagonal_schedule_has_expected_width": all(
            len(row) == 4 + 8 - 1 for row in delayed
        ),
        "delay_round_trip_is_exact": reconstructed
        == [
            [canonical[codebook][frame] for codebook in range(8)]
            for frame in range(4)
        ],
        "loss_mask_is_assistant_only": [
            index for index, label in enumerate(labels) if label != IGNORE
        ]
        == [3, 4, 5],
        "trace_does_not_change_logits": logits_without_trace == logits_with_trace,
        "controlled_rng_state_matches_after_traced_forward": (
            states_match_after_forward
        ),
        "rng_consuming_negative_control_is_detected": negative_control_detected,
    }
    return {
        "summary": (
            "用确定性的八码本张量验证 diagonal delay、assistant-only loss mask，"
            "再从相同 RNG 状态比较一次带 trace 和不带 trace 的随机 forward；"
            "该受控测试通过，且会消费随机数的负例能够被检出。"
        ),
        "metrics": {
            "text_streams": 1,
            "audio_codebooks": len(canonical),
            "audio_frames": len(canonical[0]),
            "delayed_steps": len(delayed[0]),
            "assistant_target_tokens": sum(label != IGNORE for label in labels),
            "trace_records": len(trace),
            "logit_checksum": sum(logits_with_trace),
            "negative_control_logit_checksum": sum(negative_control_logits),
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="01",
    title="建立可追溯的 MiniMind-O 基线",
    question="观测与记录是否会改变原本要复现的计算？",
    run=run,
)
