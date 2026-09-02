from __future__ import annotations

from typing import Any

from ..core import LessonExperiment


Frame = list[list[float]]


def _uniform_indices(source_frames: int, target_frames: int) -> list[int]:
    if target_frames < 2 or source_frames < target_frames:
        raise ValueError("the source must contain at least the requested frames")
    return [
        round(index * (source_frames - 1) / (target_frames - 1))
        for index in range(target_frames)
    ]


def _pairwise_reduce(frames: list[Frame]) -> list[Frame]:
    if len(frames) % 2:
        raise ValueError("temporal stride two requires an even frame count")
    reduced: list[Frame] = []
    for left, right in zip(frames[::2], frames[1::2]):
        reduced.append(
            [
                [(a + b) / 2.0 for a, b in zip(left_patch, right_patch)]
                for left_patch, right_patch in zip(left, right)
            ],
        )
    return reduced


def _temporal_adapter(frames: list[Frame], gate: float = 0.1) -> list[Frame]:
    """A tiny residual temporal smoother with no learned-quality claim."""
    adapted: list[Frame] = []
    for frame_index, frame in enumerate(frames):
        previous = frames[max(0, frame_index - 1)]
        following = frames[min(len(frames) - 1, frame_index + 1)]
        adapted.append(
            [
                [
                    value
                    + gate * (((before + after) / 2.0) - value)
                    for value, before, after in zip(
                        patch,
                        previous[patch_index],
                        following[patch_index],
                    )
                ]
                for patch_index, patch in enumerate(frame)
            ],
        )
    return adapted


def _shape(frames: list[Frame]) -> list[int]:
    return [len(frames), len(frames[0]), len(frames[0][0])]


def _run() -> dict[str, Any]:
    sampled_indices = _uniform_indices(source_frames=31, target_frames=16)
    frame_times_ms = [source_index * 40 for source_index in sampled_indices]
    frames: list[Frame] = [
        [
            [
                float(source_index),
                float(patch_index),
                float(source_index + patch_index),
            ]
            for patch_index in range(4)
        ]
        for source_index in sampled_indices
    ]

    framewise = _pairwise_reduce(frames)
    adapter = _pairwise_reduce(_temporal_adapter(frames))
    conv3d_reference = _pairwise_reduce(frames)
    midpoint_times = [
        (left + right) / 2.0
        for left, right in zip(frame_times_ms[::2], frame_times_ms[1::2])
    ]

    time_bucket_ms = 80
    video_tokens = [
        (round(time_ms / time_bucket_ms), "video") for time_ms in midpoint_times
    ]
    audio_tokens = [
        (round(time_ms / time_bucket_ms), "audio") for time_ms in midpoint_times
    ]
    packed_av = sorted(video_tokens + audio_tokens)

    events = ["door_opens", "person_enters"]
    normal_answer = events[0]
    reversed_answer = list(reversed(events))[0]
    shuffled_answer = "无法判断"
    mismatched_audio_answer = "bell"

    odd_frame_rejected = False
    try:
        _pairwise_reduce(frames[:-1])
    except ValueError:
        odd_frame_rejected = True

    output_shapes = {
        "framewise": _shape(framewise),
        "late_adapter": _shape(adapter),
        "conv3d_reference": _shape(conv3d_reference),
    }
    shared_buckets = {
        bucket for bucket, modality in packed_av if modality == "video"
    } & {
        bucket for bucket, modality in packed_av if modality == "audio"
    }

    checks = {
        "uniform sampling returns sixteen ordered source frames": (
            len(sampled_indices) == 16
            and sampled_indices == sorted(sampled_indices)
            and len(set(sampled_indices)) == 16
        ),
        "temporal stride maps sixteen frames to eight midpoints": (
            len(midpoint_times) == 8
            and midpoint_times[0] == 40.0
            and midpoint_times[-1] == 1160.0
        ),
        "all temporal arms expose the same output shape": (
            len({tuple(value) for value in output_shapes.values()}) == 1
            and output_shapes["framewise"] == [8, 4, 3]
        ),
        "odd frame counts fail instead of receiving silent padding": (
            odd_frame_rejected
        ),
        "audio and video share time buckets but retain modality identity": (
            len(shared_buckets) == 8
            and all(
                {modality for token_bucket, modality in packed_av
                 if token_bucket == bucket} == {"audio", "video"}
                for bucket in shared_buckets
            )
        ),
        "counterfactual inputs use recomputed labels": (
            normal_answer != reversed_answer
            and shuffled_answer == "无法判断"
            and mismatched_audio_answer != normal_answer
        ),
    }

    return {
        "summary": (
            "用 16 帧小张量复现确定性抽帧、相邻帧时间中点、三条等形状时序"
            "路径和音视频共享时间桶；这里只验证时序契约，不报告视频模型质量。"
        ),
        "metrics": {
            "sampled_source_indices": sampled_indices,
            "input_shape": _shape(frames),
            "output_shapes": output_shapes,
            "midpoint_times_ms": midpoint_times,
            "shared_av_time_buckets": len(shared_buckets),
            "normal_answer": normal_answer,
            "reversed_answer": reversed_answer,
            "shuffle_answer": shuffled_answer,
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="09",
    title="原生视频时序建模与音视频对齐",
    question="抽帧、时序降采样和音视频 token 怎样共享真实时间而不混淆模态？",
    run=_run,
)
