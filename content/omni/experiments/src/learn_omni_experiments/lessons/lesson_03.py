from __future__ import annotations

import math
from typing import Any

from ..core import LessonExperiment


COARSE_BOOK = (-0.75, -0.25, 0.25, 0.75)
RESIDUAL_BOOK = (-0.25, -1.0 / 12.0, 1.0 / 12.0, 0.25)


def _nearest(value: float, codebook: tuple[float, ...]) -> int:
    return min(range(len(codebook)), key=lambda index: abs(value - codebook[index]))


def _quantize(value: float) -> tuple[tuple[int, int], float, float]:
    coarse_index = _nearest(value, COARSE_BOOK)
    coarse = COARSE_BOOK[coarse_index]
    residual_index = _nearest(value - coarse, RESIDUAL_BOOK)
    reconstructed = coarse + RESIDUAL_BOOK[residual_index]
    return (coarse_index, residual_index), coarse, reconstructed


class StreamingToyCodec:
    def __init__(self, frame_size: int) -> None:
        self.frame_size = frame_size
        self.pending: list[float] = []
        self.valid_lengths: list[int] = []

    def push(self, samples: list[float]) -> list[tuple[int, int]]:
        merged = self.pending + samples
        complete_samples = len(merged) - len(merged) % self.frame_size
        codes = []
        for start in range(0, complete_samples, self.frame_size):
            frame = merged[start : start + self.frame_size]
            codes.append(_quantize(sum(frame) / len(frame))[0])
            self.valid_lengths.append(self.frame_size)
        self.pending = merged[complete_samples:]
        return codes

    def flush(self) -> list[tuple[int, int]]:
        if not self.pending:
            return []
        frame = self.pending
        self.pending = []
        self.valid_lengths.append(len(frame))
        return [_quantize(sum(frame) / len(frame))[0]]


def _encode_with_lengths(
    samples: list[float],
    frame_size: int,
) -> tuple[list[tuple[int, int]], list[int]]:
    codec = StreamingToyCodec(frame_size)
    codes = codec.push(samples) + codec.flush()
    return codes, list(codec.valid_lengths)


def _encode(samples: list[float], frame_size: int) -> list[tuple[int, int]]:
    return _encode_with_lengths(samples, frame_size)[0]


def _decode(
    codes: list[tuple[int, int]],
    frame_size: int,
    valid_lengths: list[int] | None = None,
) -> list[float]:
    lengths = valid_lengths or [frame_size] * len(codes)
    if len(lengths) != len(codes):
        raise ValueError("each code frame must have one valid length")
    output = []
    for (coarse_index, residual_index), valid_length in zip(codes, lengths):
        if not 1 <= valid_length <= frame_size:
            raise ValueError("valid length must be within one codec frame")
        value = COARSE_BOOK[coarse_index] + RESIDUAL_BOOK[residual_index]
        output.extend([value] * valid_length)
    return output


def _rmse(targets: list[float], predictions: list[float]) -> float:
    return math.sqrt(
        sum((target - prediction) ** 2 for target, prediction in zip(targets, predictions))
        / len(targets),
    )


def run() -> dict[str, Any]:
    frame_size = 4
    samples = [
        0.9 * math.sin(2.0 * math.pi * index / 32.0)
        + 0.1 * math.sin(2.0 * math.pi * index / 7.0)
        for index in range(240)
    ]
    frame_means = [
        sum(samples[start : start + frame_size]) / frame_size
        for start in range(0, len(samples), frame_size)
    ]
    quantized = [_quantize(value) for value in frame_means]
    coarse_reconstruction = [item[1] for item in quantized]
    residual_reconstruction = [item[2] for item in quantized]
    offline_codes, offline_valid_lengths = _encode_with_lengths(samples, frame_size)

    streaming_codec = StreamingToyCodec(frame_size)
    chunk_sizes = [7, 13, 29, 5, 41, 17, 53, 75]
    streaming_codes: list[tuple[int, int]] = []
    cursor = 0
    for chunk_size in chunk_sizes:
        streaming_codes.extend(streaming_codec.push(samples[cursor : cursor + chunk_size]))
        cursor += chunk_size
    streaming_codes.extend(streaming_codec.flush())
    streaming_valid_lengths = list(streaming_codec.valid_lengths)

    first_session = StreamingToyCodec(frame_size)
    second_session = StreamingToyCodec(frame_size)
    first_codes = first_session.push(samples[:37]) + first_session.push(samples[37:])
    first_codes += first_session.flush()
    reversed_samples = list(reversed(samples))
    second_codes = second_session.push(reversed_samples[:19])
    second_codes += second_session.push(reversed_samples[19:])
    second_codes += second_session.flush()

    short_codec = StreamingToyCodec(frame_size)
    short_before_flush = short_codec.push([0.2, 0.4])
    short_after_flush = short_codec.flush()
    short_decoded = _decode(
        short_after_flush,
        frame_size,
        short_codec.valid_lengths,
    )
    decoded = _decode(offline_codes, frame_size, offline_valid_lengths)
    coarse_decoded = [
        value
        for coarse_value, valid_length in zip(
            coarse_reconstruction,
            offline_valid_lengths,
        )
        for value in [coarse_value] * valid_length
    ]
    bitrate = (8000 / frame_size) * (
        math.log2(len(COARSE_BOOK)) + math.log2(len(RESIDUAL_BOOK))
    )

    frame_mean_coarse_rmse = _rmse(frame_means, coarse_reconstruction)
    frame_mean_residual_rmse = _rmse(frame_means, residual_reconstruction)
    coarse_sample_rmse = _rmse(samples, coarse_decoded)
    decoded_sample_rmse = _rmse(samples, decoded)
    checks = {
        "codes_follow_q_by_t_contract": all(
            len(code) == 2 for code in offline_codes
        ),
        "code_indices_stay_in_range": all(
            0 <= coarse < len(COARSE_BOOK)
            and 0 <= residual < len(RESIDUAL_BOOK)
            for coarse, residual in offline_codes
        ),
        "second_rvq_stage_reduces_sample_reconstruction_error": (
            decoded_sample_rmse < coarse_sample_rmse
            and frame_mean_residual_rmse < frame_mean_coarse_rmse
        ),
        "streaming_matches_offline": streaming_codes == offline_codes
        and streaming_valid_lengths == offline_valid_lengths,
        "sessions_do_not_share_state": first_codes == offline_codes
        and second_codes == _encode(reversed_samples, frame_size),
        "short_input_waits_for_flush": short_before_flush == []
        and len(short_after_flush) == 1
        and short_codec.valid_lengths == [2],
        "short_flush_crops_decode_to_valid_samples": len(short_decoded) == 2,
        "decoded_duration_matches_source_samples": len(decoded) == len(samples),
        "encoding_is_deterministic": offline_codes == _encode(samples, frame_size),
    }
    return {
        "summary": (
            "用两级标量 RVQ 编码合成信号，实际计算码率、重建误差和流式状态，"
            "说明 codec 质量、token 契约与 chunk 行为需要分别验证。"
        ),
        "metrics": {
            "samples": len(samples),
            "frames": len(offline_codes),
            "codebooks": 2,
            "nominal_bitrate_bps": bitrate,
            "coarse_sample_rmse": round(coarse_sample_rmse, 8),
            "two_stage_sample_rmse": round(decoded_sample_rmse, 8),
            "frame_mean_two_stage_rmse": round(frame_mean_residual_rmse, 8),
            "streaming_chunks": len(chunk_sizes),
            "short_flush_valid_samples": short_codec.valid_lengths[0],
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="03",
    title="比较 Audio Codec",
    question="增加 RVQ 层能否降低误差，同时保持可验证的流式 token 契约？",
    run=run,
)
