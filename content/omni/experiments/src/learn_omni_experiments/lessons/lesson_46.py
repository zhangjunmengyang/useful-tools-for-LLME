from __future__ import annotations

import math
from typing import Any

from ..core import LessonExperiment

VISION_A = 3
TEXT_A = 6
VISION_B = 9
TEXT_B = 2
VISION_C = 0
TEXT_C = 4
ACTION_STEPS_A = 0
ACTION_STEPS_B = 0
ACTION_STEPS_C = 8


def _layout(
    vision_len: int,
    text_len: int,
    vision_max: int,
    text_max: int,
) -> tuple[list[int], list[str]]:
    if vision_len > vision_max or text_len > text_max:
        raise ValueError("request length exceeds padded max")
    mask: list[int] = []
    kinds: list[str] = []
    for index in range(vision_max):
        if index < vision_len:
            mask.append(1)
            kinds.append("vision")
        else:
            mask.append(0)
            kinds.append("pad_vision")
    for index in range(text_max):
        if index < text_len:
            mask.append(1)
            kinds.append("text")
        else:
            mask.append(0)
            kinds.append("pad_text")
    return mask, kinds


def _valid_indices(mask: list[int]) -> list[int]:
    return [index for index, bit in enumerate(mask) if bit == 1]


def _invalid_indices(mask: list[int]) -> list[int]:
    return [index for index, bit in enumerate(mask) if bit == 0]


def _softmax(scores: list[float]) -> list[float]:
    peak = max(scores)
    weights = [math.exp(score - peak) for score in scores]
    total = sum(weights)
    return [weight / total for weight in weights]


def _masked_attention(
    scores: list[float],
    mask: list[int],
) -> tuple[list[float], float]:
    if len(scores) != len(mask):
        raise ValueError("scores and mask must have the same length")
    masked_scores = [
        score if bit == 1 else float("-inf") for score, bit in zip(scores, mask)
    ]
    if all(bit == 0 for bit in mask):
        raise ValueError("mask must keep at least one valid position")
    weights = _softmax(masked_scores)
    invalid_mass = sum(
        weight for weight, bit in zip(weights, mask) if bit == 0
    )
    return weights, invalid_mass


def _valid_token_ratio(valid: int, padded: int) -> float:
    if padded <= 0:
        raise ValueError("padded token count must be positive")
    return valid / padded


def run() -> dict[str, Any]:
    vision_max = max(VISION_A, VISION_B)
    text_max = max(TEXT_A, TEXT_B)
    padded_len = vision_max + text_max

    mask_a, kinds_a = _layout(VISION_A, TEXT_A, vision_max, text_max)
    mask_b, kinds_b = _layout(VISION_B, TEXT_B, vision_max, text_max)

    valid_a = sum(mask_a)
    valid_b = sum(mask_b)
    valid_total = valid_a + valid_b
    padded_total = 2 * padded_len
    invalid_total = padded_total - valid_total
    ratio = _valid_token_ratio(valid_total, padded_total)

    valid_idx_a = _valid_indices(mask_a)
    invalid_idx_a = _invalid_indices(mask_a)
    valid_idx_b = _valid_indices(mask_b)
    invalid_idx_b = _invalid_indices(mask_b)

    naive_valid_total = padded_total
    naive_ratio = _valid_token_ratio(naive_valid_total, padded_total)

    scores_a = [0.0] * padded_len
    _, invalid_mass_correct = _masked_attention(scores_a, mask_a)
    weights_wrong = _softmax(scores_a)
    invalid_mass_wrong = sum(
        weight for weight, bit in zip(weights_wrong, mask_a) if bit == 0
    )
    expected_wrong_mass = (padded_len - valid_a) / padded_len

    packed_len = (VISION_A + TEXT_A) + (VISION_B + TEXT_B)
    packed_ratio = _valid_token_ratio(valid_total, packed_len)

    vision_max_three = max(VISION_A, VISION_B, VISION_C)
    text_max_three = max(TEXT_A, TEXT_B, TEXT_C)
    mask_c, _ = _layout(VISION_C, TEXT_C, vision_max_three, text_max_three)
    mask_a3, _ = _layout(VISION_A, TEXT_A, vision_max_three, text_max_three)
    mask_b3, _ = _layout(VISION_B, TEXT_B, vision_max_three, text_max_three)
    valid_three = sum(mask_a3) + sum(mask_b3) + sum(mask_c)
    padded_three = 3 * (vision_max_three + text_max_three)
    ratio_three = _valid_token_ratio(valid_three, padded_three)

    action_max = max(ACTION_STEPS_A, ACTION_STEPS_B, ACTION_STEPS_C)
    dummy_action_steps = (
        (action_max - ACTION_STEPS_A)
        + (action_max - ACTION_STEPS_B)
        + (action_max - ACTION_STEPS_C)
    )
    fused_compute_slots = padded_three + 3 * action_max
    fused_valid_slots = valid_three + (
        ACTION_STEPS_A + ACTION_STEPS_B + ACTION_STEPS_C
    )
    fused_ratio = _valid_token_ratio(fused_valid_slots, fused_compute_slots)

    ar_pages = list(range(padded_three))
    action_pages = list(range(action_max))
    fused_kv_collision = sorted(set(ar_pages) & set(action_pages))
    stage_action_pages = list(range(padded_three, padded_three + action_max))
    stage_kv_collision = sorted(set(ar_pages) & set(stage_action_pages))

    pad_vision_in_valid_a = any(
        kinds_a[index] == "pad_vision" for index in valid_idx_a
    )
    pad_text_in_valid_b = any(
        kinds_b[index] == "pad_text" for index in valid_idx_b
    )

    checks = {
        "two_vision_requests_have_unequal_lengths": (
            VISION_A != VISION_B and TEXT_A != TEXT_B
        ),
        "invalid_positions_excluded_from_valid_count": (
            valid_total == VISION_A + TEXT_A + VISION_B + TEXT_B
            and valid_total != padded_total
            and invalid_total == (vision_max - VISION_A) + (text_max - TEXT_B)
            and not pad_vision_in_valid_a
            and not pad_text_in_valid_b
            and set(valid_idx_a).isdisjoint(invalid_idx_a)
            and set(valid_idx_b).isdisjoint(invalid_idx_b)
            and sorted(valid_idx_a + invalid_idx_a) == list(range(padded_len))
            and sorted(valid_idx_b + invalid_idx_b) == list(range(padded_len))
        ),
        "valid_token_ratio_uses_mask_not_padded_shape": (
            abs(ratio - 20 / 30) < 1e-12
            and abs(naive_ratio - 1.0) < 1e-12
            and ratio < naive_ratio
        ),
        "wrong_padding_mask_puts_mass_on_invalid_keys": (
            invalid_mass_correct == 0.0
            and abs(invalid_mass_wrong - expected_wrong_mass) < 1e-12
            and invalid_mass_wrong > 0.0
        ),
        "packed_layout_keeps_every_real_token": (
            packed_len == valid_total
            and abs(packed_ratio - 1.0) < 1e-12
        ),
        "text_only_request_in_fused_batch_drops_ratio": (
            VISION_C == 0
            and ratio_three < ratio
            and abs(ratio_three - 24 / 45) < 1e-12
        ),
        "fused_action_steps_are_a_separate_padded_dim": (
            action_max == ACTION_STEPS_C
            and dummy_action_steps == 2 * action_max
            and fused_ratio < ratio_three
            and fused_valid_slots == valid_three + ACTION_STEPS_C
        ),
        "fused_kv_aliases_action_and_ar_pages": (
            fused_kv_collision == list(range(action_max))
            and stage_kv_collision == []
        ),
    }

    return {
        "summary": (
            "两条变长视觉请求按 max(vision)=9、max(text)=6 组 batch；"
            "有效 token 只统计 mask=1 的位置，得到 20/30，"
            "padding 位置不得进入有效计数。"
            "错误 mask 会把无效 key 算进注意力；"
            "把纯文本和动作专家硬塞进同一条静态图会继续拉低有效比，并造成 KV 页别名。"
        ),
        "metrics": {
            "vision_a": VISION_A,
            "text_a": TEXT_A,
            "vision_b": VISION_B,
            "text_b": TEXT_B,
            "vision_max": vision_max,
            "text_max": text_max,
            "padded_len": padded_len,
            "valid_a": valid_a,
            "valid_b": valid_b,
            "valid_total": valid_total,
            "padded_total": padded_total,
            "invalid_total": invalid_total,
            "valid_token_ratio": ratio,
            "naive_valid_token_ratio": naive_ratio,
            "packed_len": packed_len,
            "invalid_attention_mass_correct_mask": invalid_mass_correct,
            "invalid_attention_mass_wrong_mask": invalid_mass_wrong,
            "valid_indices_a": valid_idx_a,
            "invalid_indices_a": invalid_idx_a,
            "valid_indices_b": valid_idx_b,
            "invalid_indices_b": invalid_idx_b,
            "fused_three_valid": valid_three,
            "fused_three_padded": padded_three,
            "fused_three_ratio": ratio_three,
            "dummy_action_steps": dummy_action_steps,
            "fused_compute_ratio": fused_ratio,
            "fused_kv_collision_pages": fused_kv_collision,
            "stage_kv_collision_pages": stage_kv_collision,
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="46",
    title="用 stage graph 调度多模态推理",
    question="两条变长视觉请求组 batch 时，padding 位置为什么不能进入有效 token 计数？",
    run=run,
)
