from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Hashable

from ..core import LessonExperiment


BOS = -2
PAD = -1
IGNORE = -100


def _diagonal_schedule(codes: list[list[int]]) -> list[list[int]]:
    codebooks = len(codes)
    return [
        [BOS] * codebook + row + [PAD] * (codebooks - codebook - 1)
        for codebook, row in enumerate(codes)
    ]


def _reconstruct(schedule: list[list[int]], frames: int) -> list[list[int]]:
    return [
        [schedule[codebook][frame + codebook] for codebook in range(len(schedule))]
        for frame in range(frames)
    ]


def _parallel_frames(codes: list[list[int]]) -> list[list[int]]:
    """Emit all codebooks for each frame in one outer step."""
    return [
        [codes[codebook][frame] for codebook in range(len(codes))]
        for frame in range(len(codes[0]))
    ]


def _grouped_frames(
    codes: list[list[int]],
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """Emit q0, then predict later codebooks from the same-frame prefix."""
    frames: list[list[int]] = []
    trace: list[dict[str, Any]] = []
    for frame_index in range(len(codes[0])):
        prefix: list[int] = []
        for codebook_index, codebook in enumerate(codes):
            target = codebook[frame_index]
            trace.append(
                {
                    "frame": frame_index,
                    "codebook": codebook_index,
                    "same_frame_prefix": list(prefix),
                    "target": target,
                },
            )
            prefix.append(target)
        frames.append(prefix)
    return frames, trace


def _entropy(values: list[Hashable]) -> float:
    counts = Counter(values)
    total = len(values)
    return -sum(
        (count / total) * math.log(count / total)
        for count in counts.values()
    )


def _conditional_entropy(
    targets: list[Hashable],
    conditions: list[Hashable],
) -> float:
    groups: dict[Hashable, list[Hashable]] = defaultdict(list)
    for condition, target in zip(conditions, targets):
        groups[condition].append(target)
    total = len(targets)
    return sum(
        (len(group) / total) * _entropy(group)
        for group in groups.values()
    )


def run() -> dict[str, Any]:
    canonical = [
        [10, 11],
        [20, 21],
        [30, 31],
    ]
    expected_schedule = [
        [10, 11, PAD, PAD],
        [BOS, 20, 21, PAD],
        [BOS, BOS, 30, 31],
    ]
    schedule = _diagonal_schedule(canonical)
    reconstructed = _reconstruct(schedule, frames=2)
    model_inputs = [[BOS] + row[:-1] for row in schedule]
    next_token_targets = [list(row) for row in schedule]
    next_token_labels = [
        [
            token if token not in {BOS, PAD} else IGNORE
            for token in row
        ]
        for row in next_token_targets
    ]
    supervised_targets = [
        token
        for row in next_token_labels
        for token in row
        if token != IGNORE
    ]
    parallel_frames = _parallel_frames(canonical)
    grouped_frames, grouped_trace = _grouped_frames(canonical)

    frames = [
        (0, 4, 0),
        (1, 7, 1),
        (0, 2, 0),
        (1, 5, 1),
        (0, 9, 0),
        (1, 3, 1),
    ]
    q0 = [frame[0] for frame in frames]
    q2 = [frame[2] for frame in frames]
    independent_nll = _entropy(q2)
    conditioned_nll = _conditional_entropy(q2, q0)

    checks = {
        "three_by_two_schedule_matches_derivation": schedule == expected_schedule,
        "schedule_round_trip_is_exact": reconstructed
        == [[10, 20, 30], [11, 21, 31]],
        "teacher_forcing_shift_preserves_each_canonical_target": (
            all(
                inputs[1:] == labels[:-1]
                for inputs, labels in zip(model_inputs, next_token_targets)
            )
            and Counter(supervised_targets)
            == Counter(token for row in canonical for token in row)
            and len(supervised_targets) == 6
        ),
        "same_frame_condition_reduces_uncertainty": conditioned_nll < independent_nll,
        "three_topologies_reconstruct_the_same_frames": (
            parallel_frames == grouped_frames == reconstructed
        ),
        "grouped_path_records_same_frame_prefixes": (
            grouped_trace[0]["same_frame_prefix"] == []
            and grouped_trace[2]["same_frame_prefix"] == [10, 20]
            and len(grouped_trace) == 6
        ),
        "diagonal_uses_more_outer_steps": len(schedule[0]) > len(canonical[0]),
    }
    return {
        "summary": (
            "先精确复现 3×2 diagonal delay，再用经验熵计算展示：当后层码本"
            "确实依赖同帧前层码本时，帧内条件会降低预测不确定性。"
        ),
        "metrics": {
            "codebooks": len(canonical),
            "frames": len(canonical[0]),
            "diagonal_outer_steps": len(schedule[0]),
            "same_frame_outer_steps": len(canonical[0]),
            "grouped_inner_depth": 2,
            "supervised_canonical_tokens": supervised_targets,
            "independent_q2_nll_nats": round(independent_nll, 8),
            "conditioned_q2_nll_nats": round(conditioned_nll, 8),
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="04",
    title="比较多码本 Talker 拓扑",
    question="同帧前层码本何时能降低后层码本的预测不确定性？",
    run=run,
)
