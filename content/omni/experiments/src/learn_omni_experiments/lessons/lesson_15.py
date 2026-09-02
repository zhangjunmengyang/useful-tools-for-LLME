from __future__ import annotations

import math
from typing import Iterable

from ..core import LessonExperiment


def _shares(values: dict[str, int]) -> dict[str, float]:
    total = sum(values.values())
    return {name: value / total for name, value in values.items()}


def _l1_error(
    actual: dict[str, float],
    expected: dict[str, float],
) -> float:
    return sum(abs(actual[name] - expected[name]) for name in expected)


def _cosine(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = tuple(left)
    right_values = tuple(right)
    dot = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    return dot / (left_norm * right_norm)


def _build_audio_layout() -> tuple[list[list[int]], list[list[int]], int, int]:
    sequence_length = 18
    assistant_start = 3
    frame_count = 3
    audio_pad_token = 2049
    audio_stop_token = 2050

    y_audio = [
        [audio_pad_token for _ in range(sequence_length)]
        for _ in range(8)
    ]
    audio_target = [[-100 for _ in range(sequence_length)] for _ in range(8)]

    for codebook in range(8):
        target = [
            100 + 10 * codebook + frame
            for frame in range(frame_count)
        ] + [audio_stop_token]
        write_start = assistant_start + codebook + 1
        for offset, token in enumerate(target):
            source_position = write_start + offset
            y_audio[codebook][source_position] = token
            audio_target[codebook][source_position] = token

    audio_inputs = [row[:-1] for row in y_audio]
    audio_labels = [row[1:] for row in audio_target]
    return audio_inputs, audio_labels, assistant_start, frame_count


def run() -> dict[str, object]:
    # These are hand-authored example counts. A real run must obtain equivalent
    # counts from its tokenizer/collator output rather than copying them.
    example_tokens_per_sample = {"text": 16, "image": 32, "audio": 64}
    example_sample_counts = {"text": 4, "image": 4, "audio": 4}
    sample_balanced_tokens = {
        name: example_sample_counts[name] * example_tokens_per_sample[name]
        for name in example_tokens_per_sample
    }
    example_token_balanced_counts = {"text": 16, "image": 8, "audio": 4}
    token_balanced_tokens = {
        name: (
            example_token_balanced_counts[name]
            * example_tokens_per_sample[name]
        )
        for name in example_tokens_per_sample
    }
    target_shares = {
        name: 1.0 / 3.0 for name in example_tokens_per_sample
    }
    sample_token_shares = _shares(sample_balanced_tokens)
    balanced_token_shares = _shares(token_balanced_tokens)

    audio_inputs, audio_labels, assistant_start, frame_count = (
        _build_audio_layout()
    )
    valid_positions = [
        [index for index, token in enumerate(row) if token != -100]
        for row in audio_labels
    ]
    expected_positions = [
        list(
            range(
                assistant_start + codebook,
                assistant_start + codebook + frame_count + 1,
            ),
        )
        for codebook in range(8)
    ]
    stop_positions = [
        assistant_start + codebook + frame_count
        for codebook in range(8)
    ]

    task_gradients = {
        "text": (2.0, -1.0, 0.0),
        "image": (1.0, 1.0, 0.0),
        "audio": (-1.0, 0.0, 2.0),
    }
    gradient_cosines = {
        f"{left}:{right}": _cosine(
            task_gradients[left],
            task_gradients[right],
        )
        for left in task_gradients
        for right in task_gradients
    }

    sample_error = _l1_error(sample_token_shares, target_shares)
    token_error = _l1_error(balanced_token_shares, target_shares)
    codec_mask_exact = valid_positions == expected_positions
    stop_tokens_supervised = all(
        audio_labels[codebook][position] == 2050
        for codebook, position in enumerate(stop_positions)
    )
    next_token_shift_is_exact = all(
        audio_inputs[codebook][position + 1]
        == audio_labels[codebook][position]
        for codebook, positions in enumerate(valid_positions)
        for position in positions
    )

    return {
        "summary": (
            "用一组手工示例 token 计数演示 sample-balanced 与 "
            "token-balanced 的账本差异，并构造 toy 8 路 codec target 的 "
            "diagonal delay、stop token 与 loss mask。它不读取真实数据集、"
            "tokenizer 或 collator 输出。"
        ),
        "metrics": {
            "token_count_source": "hand_authored_example_not_dataset_replay",
            "example_tokens_per_sample": example_tokens_per_sample,
            "example_sample_counts": example_sample_counts,
            "example_token_balanced_counts": (
                example_token_balanced_counts
            ),
            "sample_balanced_token_shares": {
                name: round(value, 6)
                for name, value in sample_token_shares.items()
            },
            "token_balanced_token_shares": {
                name: round(value, 6)
                for name, value in balanced_token_shares.items()
            },
            "sample_balance_l1_error": round(sample_error, 6),
            "token_balance_l1_error": round(token_error, 6),
            "audio_input_shape": [8, len(audio_inputs[0])],
            "valid_codec_targets_per_lane": [
                len(positions) for positions in valid_positions
            ],
            "gradient_cosines": {
                name: round(value, 6)
                for name, value in gradient_cosines.items()
            },
        },
        "checks": {
            "八路toy音频输入长度一致": (
                len(audio_inputs) == 8
                and all(len(row) == 17 for row in audio_inputs)
            ),
            "codec目标位置符合逐码本延迟": codec_mask_exact,
            "每路stop_token都参与监督": stop_tokens_supervised,
            "label相对teacher_forcing输入前移一位": next_token_shift_is_exact,
            "这组示例中按样本平衡产生token暴露偏差": sample_error > 0.4,
            "这组示例中按token平衡更接近目标比例": (
                token_error < sample_error
            ),
            "给定toy梯度包含负余弦": (
                gradient_cosines["text:audio"] < 0.0
            ),
        },
    }


LESSON = LessonExperiment(
    lesson_id="15",
    title="现代多模态 Joint SFT",
    question="采样单位和多码本 loss mask 如何改变联合训练的真实优化份额？",
    run=run,
)
