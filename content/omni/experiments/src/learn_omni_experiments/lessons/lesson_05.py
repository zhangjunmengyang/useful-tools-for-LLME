from __future__ import annotations

import math
from typing import Any

from ..core import LessonExperiment


Feature = dict[str, float | int]


class StreamingFrontend:
    def __init__(self, frame_samples: int, sample_rate: int, lookahead_ms: int = 0) -> None:
        self.frame_samples = frame_samples
        self.sample_rate = sample_rate
        self.lookahead_ms = lookahead_ms
        self.pending: list[float] = []
        self.pending_start = 0

    def _feature(self, frame: list[float], start_sample: int) -> Feature:
        end_sample = start_sample + len(frame)
        start_ms = start_sample * 1000 // self.sample_rate
        end_ms = end_sample * 1000 // self.sample_rate
        return {
            "mean": sum(frame) / len(frame),
            "rms": math.sqrt(sum(value * value for value in frame) / len(frame)),
            "source_start_ms": start_ms,
            "source_end_ms": end_ms,
            "available_at_ms": end_ms + self.lookahead_ms,
        }

    def push(self, samples: list[float]) -> list[Feature]:
        merged = self.pending + samples
        complete = len(merged) - len(merged) % self.frame_samples
        output = []
        for offset in range(0, complete, self.frame_samples):
            frame = merged[offset : offset + self.frame_samples]
            output.append(self._feature(frame, self.pending_start + offset))
        self.pending = merged[complete:]
        self.pending_start += complete
        return output

    def finalize(self) -> list[Feature]:
        if not self.pending:
            return []
        output = [self._feature(self.pending, self.pending_start)]
        self.pending_start += len(self.pending)
        self.pending = []
        return output


def _encode(
    samples: list[float],
    chunks: list[int],
    frame_samples: int,
    sample_rate: int,
) -> list[Feature]:
    frontend = StreamingFrontend(frame_samples, sample_rate)
    output: list[Feature] = []
    cursor = 0
    for chunk in chunks:
        output.extend(frontend.push(samples[cursor : cursor + chunk]))
        cursor += chunk
    output.extend(frontend.finalize())
    return output


def run() -> dict[str, Any]:
    sample_rate = 1000
    frame_samples = 80
    samples = [
        float(((index * 17) % 31) - 15) / 15.0
        for index in range(1600)
    ]
    offline = _encode(samples, [len(samples)], frame_samples, sample_rate)
    chunked = _encode(
        samples,
        [320, 160, 240, 80, 480, 320],
        frame_samples,
        sample_rate,
    )

    shared_prefix = samples[:960]
    suffix_a = samples[960:]
    suffix_b = [-value for value in suffix_a]
    prefix_a = _encode(shared_prefix + suffix_a, [960, 640], frame_samples, sample_rate)
    prefix_b = _encode(shared_prefix + suffix_b, [960, 640], frame_samples, sample_rate)
    shared_frames = len(shared_prefix) // frame_samples

    first = StreamingFrontend(frame_samples, sample_rate)
    second = StreamingFrontend(frame_samples, sample_rate)
    first_output = first.push(samples[:137])
    second_output = second.push(list(reversed(samples))[:203])
    first_output += first.push(samples[137:])
    second_output += second.push(list(reversed(samples))[203:])
    first_output += first.finalize()
    second_output += second.finalize()

    partial = StreamingFrontend(frame_samples, sample_rate)
    before_flush = partial.push(samples[:35])
    after_flush = partial.finalize()
    timestamps_valid = all(
        feature["available_at_ms"] >= feature["source_end_ms"]
        for feature in chunked
    )
    boundaries_contiguous = all(
        chunked[index]["source_end_ms"] == chunked[index + 1]["source_start_ms"]
        for index in range(len(chunked) - 1)
    )

    checks = {
        "chunked_frontend_matches_offline": chunked == offline,
        "feature_timestamps_are_causal": timestamps_valid,
        "frame_boundaries_are_contiguous": boundaries_contiguous,
        "future_suffix_cannot_change_published_prefix": (
            prefix_a[:shared_frames] == prefix_b[:shared_frames]
        ),
        "interleaved_sessions_keep_separate_state": (
            first_output == offline
            and second_output
            == _encode(
                list(reversed(samples)),
                [len(samples)],
                frame_samples,
                sample_rate,
            )
        ),
        "finalize_flushes_partial_frame_once": before_flush == []
        and len(after_flush) == 1
        and partial.finalize() == [],
    }
    return {
        "summary": (
            "把同一波形按多种 packet 边界送入有状态 frontend，逐帧核对特征、"
            "source span、available_at、finalize 和未来后缀不变性。"
        ),
        "metrics": {
            "sample_rate_hz": sample_rate,
            "frame_ms": frame_samples * 1000 // sample_rate,
            "input_samples": len(samples),
            "emitted_frames": len(chunked),
            "shared_prefix_frames": shared_frames,
            "lookahead_ms": 0,
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="05",
    title="实现因果流式 Listener",
    question="分块输入能否与离线结果一致，并证明已发布前缀没有读取未来？",
    run=run,
)
