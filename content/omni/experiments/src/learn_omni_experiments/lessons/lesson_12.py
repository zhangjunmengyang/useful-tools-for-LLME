from __future__ import annotations

import math
from typing import Any

from ..core import LessonExperiment


def _recurrent(
    values: list[float],
    decay: float,
    initial_state: float = 0.0,
) -> list[float]:
    state = initial_state
    outputs: list[float] = []
    for value in values:
        state = decay * state + value
        outputs.append(state)
    return outputs


def _closed_form(values: list[float], decay: float) -> list[float]:
    return [
        sum(
            (decay ** (output_index - input_index)) * values[input_index]
            for input_index in range(output_index + 1)
        )
        for output_index in range(len(values))
    ]


def _packed(
    samples: list[list[float]],
    decay: float,
    reset_at_boundaries: bool,
) -> list[float]:
    if reset_at_boundaries:
        return [
            output
            for sample in samples
            for output in _recurrent(sample, decay)
        ]
    return _recurrent(
        [value for sample in samples for value in sample],
        decay,
    )


def _position_embedding(contract: dict[str, Any]) -> list[float]:
    position = float(contract["position_in_sample"])
    modality = float(contract["modality_id"])
    image_valid = bool(contract["image_xy_valid"])
    time_valid = bool(contract["time_valid"])
    image_y, image_x = contract["image_xy"]
    return [
        math.sin(position / 10.0),
        math.cos(position / 10.0),
        modality,
        float(image_y) if image_valid else 0.0,
        float(image_x) if image_valid else 0.0,
        float(contract["time_bucket"]) if time_valid else 0.0,
        float(contract["segment_id"]),
    ]


def _max_error(left: list[float], right: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(left, right))


def _run() -> dict[str, Any]:
    decay = 0.75
    sample_a = [1.0, -0.5, 2.0]
    sample_b = [4.0, 1.5]
    separate = _recurrent(sample_a, decay) + _recurrent(sample_b, decay)
    packed_reset = _packed([sample_a, sample_b], decay, True)
    packed_leaky = _packed([sample_a, sample_b], decay, False)
    recurrent = _recurrent(sample_a, decay)
    full_sequence = _closed_form(sample_a, decay)

    interleaved = [
        ("request-a", 1.0),
        ("request-b", 4.0),
        ("request-a", -0.5),
        ("request-b", 1.5),
        ("request-a", 2.0),
    ]
    states: dict[str, float] = {}
    cache_outputs: dict[str, list[float]] = {
        "request-a": [],
        "request-b": [],
    }
    for request_id, value in interleaved:
        state = decay * states.get(request_id, 0.0) + value
        states[request_id] = state
        cache_outputs[request_id].append(state)

    image_contract = {
        "position_in_sample": 7,
        "modality_id": 1,
        "segment_id": 2,
        "image_xy": (0.25, 0.75),
        "image_xy_valid": True,
        "time_bucket": 0,
        "time_valid": False,
    }
    moved_image = {**image_contract, "image_xy": (0.75, 0.25)}
    text_contract = {
        **image_contract,
        "modality_id": 0,
        "image_xy_valid": False,
    }
    text_with_irrelevant_xy = {
        **text_contract,
        "image_xy": (0.99, 0.01),
    }
    audio_contract = {
        **text_contract,
        "modality_id": 2,
        "time_bucket": 12,
        "time_valid": True,
    }
    moved_audio = {**audio_contract, "time_bucket": 18}

    packed_error = _max_error(separate, packed_reset)
    full_recurrent_error = _max_error(recurrent, full_sequence)
    cache_a_error = _max_error(
        cache_outputs["request-a"],
        _recurrent(sample_a, decay),
    )
    cache_b_error = _max_error(
        cache_outputs["request-b"],
        _recurrent(sample_b, decay),
    )

    checks = {
        "packed samples match separate runs when state resets": (
            packed_error < 1e-12
        ),
        "the no-reset negative control exposes cross-sample leakage": (
            packed_leaky != separate
        ),
        "full-sequence algebra matches recurrent decoding": (
            full_recurrent_error < 1e-12
        ),
        "interleaved request caches remain isolated": (
            cache_a_error < 1e-12 and cache_b_error < 1e-12
        ),
        "image coordinates affect position embeddings only when valid": (
            _position_embedding(image_contract)
            != _position_embedding(moved_image)
            and _position_embedding(text_contract)
            == _position_embedding(text_with_irrelevant_xy)
        ),
        "audio time changes enter the shared position contract": (
            _position_embedding(audio_contract)
            != _position_embedding(moved_audio)
        ),
    }

    return {
        "summary": (
            "用一个可手算的线性状态递推验证 full/recurrent 一致性、packed 样本"
            "边界清零和 request cache 隔离，并检查图像坐标与音频时间确实进入"
            "统一位置契约；它不是 Mamba-2 CUDA kernel 或质量基准。"
        ),
        "metrics": {
            "decay": decay,
            "sample_lengths": [len(sample_a), len(sample_b)],
            "packed_cu_seqlens": [0, len(sample_a), len(sample_a) + len(sample_b)],
            "max_packed_parity_error": packed_error,
            "max_full_recurrent_error": full_recurrent_error,
            "leaky_first_output_of_second_sample": packed_leaky[len(sample_a)],
            "reset_first_output_of_second_sample": packed_reset[len(sample_a)],
            "final_request_cache": states,
            "position_embedding_width": len(_position_embedding(image_contract)),
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="12",
    title="Mamba-2 / Attention Hybrid：公平的长序列骨干实验",
    question="packed sequence 的边界和异构 cache 如何避免状态从一个样本泄漏到另一个？",
    run=_run,
)
