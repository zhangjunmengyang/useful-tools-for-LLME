from __future__ import annotations

import math
from typing import Any

from ..core import LessonExperiment


IGNORE = -100
BINS = 8
VOCAB = 32
ACTION_DIM = 7
CHUNK = 4
BATCH = 2
PROMPT_LEN = 5
PAD_LEN = 2
ACTION_TOKEN_START = VOCAB - BINS
Q01 = -1.0
Q99 = 1.0


def _clip(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _discretize(value: float, bins: int = BINS) -> int:
    width = (Q99 - Q01) / bins
    index = math.floor((value - Q01) / width)
    return int(_clip(index, 0, bins - 1))


def _bin_to_token(bin_id: int) -> int:
    return ACTION_TOKEN_START + bin_id


def _logsumexp(values: list[float]) -> float:
    peak = max(values)
    return peak + math.log(sum(math.exp(value - peak) for value in values))


def _log_softmax(logits: list[float]) -> list[float]:
    normalizer = _logsumexp(logits)
    return [logit - normalizer for logit in logits]


def _cross_entropy(logits: list[float], label: int) -> float:
    return -_log_softmax(logits)[label]


def _mean_l1(pred: list[float], target: list[float], mask: list[float]) -> float:
    weight = sum(mask)
    return sum(
        weight_i * abs(pred_i - target_i)
        for pred_i, target_i, weight_i in zip(pred, target, mask)
    ) / weight


def _teacher_forced_ce(
    logits: list[list[list[float]]],
    labels: list[list[int]],
) -> tuple[float, int]:
    total = 0.0
    count = 0
    for row_logits, row_labels in zip(logits, labels):
        for token_logits, label in zip(row_logits, row_labels):
            if label == IGNORE:
                continue
            total += _cross_entropy(token_logits, label)
            count += 1
    return total / count, count


def run() -> dict[str, Any]:
    actions = [
        [0.12, -0.40, 0.88, 0.05, -0.91, 0.33, 0.70],
        [0.44, 0.20, -0.15, 0.62, -0.05, -0.70, -0.22],
    ]
    prompt_ids = [1, 2, 3, 4, 5]
    pad_id = 0

    ce_labels: list[list[int]] = []
    ce_tokens: list[list[int]] = []
    for action in actions:
        action_tokens = [_bin_to_token(_discretize(value)) for value in action]
        tokens = prompt_ids + action_tokens + [pad_id] * PAD_LEN
        labels = (
            [IGNORE] * PROMPT_LEN
            + action_tokens
            + [IGNORE] * PAD_LEN
        )
        ce_tokens.append(tokens)
        ce_labels.append(labels)

    seq_len = PROMPT_LEN + ACTION_DIM + PAD_LEN
    ce_logits = [
        [
            [
                0.15 * (token - ACTION_TOKEN_START) - 0.04 * vocab + 0.01 * step
                for vocab in range(VOCAB)
            ]
            for step, token in enumerate(row)
        ]
        for row in ce_tokens
    ]
    ce_loss, ce_counted = _teacher_forced_ce(ce_logits, ce_labels)
    ce_action_positions = [
        [index for index, label in enumerate(row) if label != IGNORE]
        for row in ce_labels
    ]

    chunk_actions = [
        [list(actions[0]) for _ in range(CHUNK)],
        [
            list(actions[1]) if step < CHUNK - 1 else [0.0] * ACTION_DIM
            for step in range(CHUNK)
        ],
    ]
    l1_mask = [
        [[1.0] * ACTION_DIM for _ in range(CHUNK)],
        [
            [1.0] * ACTION_DIM if step < CHUNK - 1 else [0.0] * ACTION_DIM
            for step in range(CHUNK)
        ],
    ]
    l1_pred = [
        [
            [
                value + 0.05 * (step + 1) * (dim + 1) / 40
                for dim, value in enumerate(action)
            ]
            for step, action in enumerate(sample)
        ]
        for sample in chunk_actions
    ]
    flat_pred: list[float] = []
    flat_target: list[float] = []
    for sample_index, sample in enumerate(chunk_actions):
        for step, action in enumerate(sample):
            for dim, value in enumerate(action):
                if l1_mask[sample_index][step][dim] > 0:
                    flat_pred.append(l1_pred[sample_index][step][dim])
                    flat_target.append(value)
    flat_mask = [1.0] * len(flat_pred)
    l1_loss = _mean_l1(flat_pred, flat_target, flat_mask)

    edge = Q01 + (Q99 - Q01) / BINS
    left = edge - 1e-9
    right = edge + 1e-9
    left_bin = _discretize(left)
    right_bin = _discretize(right)
    ce_jump = abs(
        _cross_entropy(
            [0.0 if index != left_bin else 4.0 for index in range(BINS)],
            right_bin,
        )
        - _cross_entropy(
            [0.0 if index != left_bin else 4.0 for index in range(BINS)],
            left_bin,
        )
    )
    l1_across_edge = abs(left - right)

    serial_steps_h1 = ACTION_DIM
    serial_steps_chunk = ACTION_DIM * CHUNK
    parallel_steps = 1

    ce_logit_shape = [BATCH, seq_len, VOCAB]
    l1_pred_shape = [BATCH, CHUNK, ACTION_DIM]
    l1_mask_shape = [len(sample) for sample in l1_mask], [
        len(step) for step in l1_mask[0]
    ]

    prompt_ignored = all(
        label == IGNORE
        for row in ce_labels
        for label in row[:PROMPT_LEN]
    )
    padding_ignored = all(
        label == IGNORE
        for row in ce_labels
        for label in row[PROMPT_LEN + ACTION_DIM :]
    )
    action_label_count = [
        sum(label != IGNORE for label in row) for row in ce_labels
    ]
    l1_valid = sum(
        1
        for sample in l1_mask
        for step in sample
        for flag in step
        if flag > 0
    )
    l1_invalid_tail = all(flag == 0.0 for flag in l1_mask[1][-1])

    checks = {
        "ce_logits_shape_is_batch_seq_vocab": ce_logit_shape
        == [len(ce_logits), len(ce_logits[0]), len(ce_logits[0][0])],
        "ce_loss_mask_skips_prompt_and_padding": prompt_ignored and padding_ignored,
        "ce_counts_seven_action_tokens": action_label_count == [ACTION_DIM, ACTION_DIM]
        and ce_counted == BATCH * ACTION_DIM,
        "serial_depth_is_seven_or_seven_h": serial_steps_h1 == 7
        and serial_steps_chunk == 7 * CHUNK
        and parallel_steps == 1,
        "l1_pred_shape_is_batch_chunk_dim": l1_pred_shape
        == [len(l1_pred), len(l1_pred[0]), len(l1_pred[0][0])]
        and l1_mask_shape == ([CHUNK, CHUNK], [ACTION_DIM] * CHUNK),
        "l1_mask_drops_padded_timestep": l1_invalid_tail and l1_valid == (
            BATCH * CHUNK * ACTION_DIM - ACTION_DIM
        ),
        "bin_boundary_splits_ce_classes": left_bin != right_bin and ce_jump > 1.0,
        "l1_penalty_stays_linear_across_bin_edge": l1_across_edge < 1e-8,
    }
    return {
        "summary": (
            "用 7 维 bin 化动作核对 teacher-forced CE 的序列形状与 loss mask，"
            "再用并行 L1 核对 [B, H, 7] 预测和 padded timestep 的位置 mask；"
            "同一 bin 边界上 CE 类别跳变，L1 仍按连续差计罚。"
        ),
        "metrics": {
            "vocab_size": VOCAB,
            "bins": BINS,
            "action_dim": ACTION_DIM,
            "chunk": CHUNK,
            "ce_seq_len": seq_len,
            "ce_counted_tokens": ce_counted,
            "ce_loss": round(ce_loss, 6),
            "l1_loss": round(l1_loss, 6),
            "l1_valid_entries": l1_valid,
            "serial_steps_h1": serial_steps_h1,
            "serial_steps_chunk": serial_steps_chunk,
            "left_bin": left_bin,
            "right_bin": right_bin,
            "action_token_start": ACTION_TOKEN_START,
            "ce_action_positions": ce_action_positions[0],
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="27",
    title="实现并诊断自回归 VLA",
    question="bin 化动作的 teacher-forced CE 与并行 L1 在形状和 loss mask 上是否一致可核？",
    run=run,
)
