from __future__ import annotations

import math
from typing import Any

from ..core import LessonExperiment

IGNORE = -100
SEQUENCE_LENGTH = 20
AUDIO_POSITIONS = tuple(range(0, 6))
PROMPT_POSITIONS = tuple(range(6, 11))
ASR_POSITIONS = tuple(range(11, 16))
IF_POSITIONS = tuple(range(16, 19))
EOS_POSITION = 19

# 教学词表 id：听写 31..35，执行 41..43。
TRANSCRIPT_IDS = (31, 32, 33, 34, 35)
ACTION_IDS = (41, 42, 43)

ASR_CORRECT_PROB = 0.80
IF_COPY_PROB = 0.15


def _labels() -> tuple[int, ...]:
    values = [IGNORE] * SEQUENCE_LENGTH
    for index, token in zip(ASR_POSITIONS, TRANSCRIPT_IDS):
        values[index] = token
    for index, token in zip(IF_POSITIONS, ACTION_IDS):
        values[index] = token
    return tuple(values)


def _mask(positions: tuple[int, ...]) -> tuple[int, ...]:
    flags = [0] * SEQUENCE_LENGTH
    for index in positions:
        flags[index] = 1
    return tuple(flags)


def _effective(mask: tuple[int, ...], labels: tuple[int, ...]) -> frozenset[int]:
    return frozenset(
        index
        for index, (flag, label) in enumerate(zip(mask, labels))
        if flag == 1 and label != IGNORE
    )


def _cross_entropy(mask_set: frozenset[int], probability: dict[int, float]) -> float:
    if not mask_set:
        raise ValueError("effective token set must be non-empty")
    total = 0.0
    for index in sorted(mask_set):
        prob = probability[index]
        if not 0.0 < prob <= 1.0:
            raise ValueError("probability must be in (0, 1]")
        total += -math.log(prob)
    return total / len(mask_set)


def run() -> dict[str, Any]:
    labels = _labels()
    asr_mask = _mask(ASR_POSITIONS)
    if_mask = _mask(IF_POSITIONS)
    contaminated_mask = tuple(
        1 if asr_bit or if_bit else 0 for asr_bit, if_bit in zip(asr_mask, if_mask)
    )

    asr_set = _effective(asr_mask, labels)
    if_set = _effective(if_mask, labels)
    mix_set = _effective(contaminated_mask, labels)
    audio_set = frozenset(AUDIO_POSITIONS)
    prompt_set = frozenset(PROMPT_POSITIONS)

    copy_probability = {
        index: ASR_CORRECT_PROB if index in asr_set else IF_COPY_PROB
        for index in asr_set | if_set
    }
    asr_ce = _cross_entropy(asr_set, copy_probability)
    if_ce = _cross_entropy(if_set, copy_probability)
    mix_ce = _cross_entropy(mix_set, copy_probability)
    expected_asr = -math.log(ASR_CORRECT_PROB)
    expected_if = -math.log(IF_COPY_PROB)
    expected_mix = (
        len(asr_set) * expected_asr + len(if_set) * expected_if
    ) / (len(asr_set) + len(if_set))

    checks = {
        "asr_and_if_effective_sets_differ": asr_set != if_set,
        "effective_counts_unequal": len(asr_set) != len(if_set),
        "asr_and_if_sets_are_disjoint": asr_set.isdisjoint(if_set),
        "audio_tokens_excluded_from_both_masks": asr_set.isdisjoint(audio_set)
        and if_set.isdisjoint(audio_set),
        "prompt_tokens_excluded_from_both_masks": asr_set.isdisjoint(prompt_set)
        and if_set.isdisjoint(prompt_set),
        "eos_excluded_from_both_masks": EOS_POSITION not in asr_set
        and EOS_POSITION not in if_set
        and labels[EOS_POSITION] == IGNORE,
        "contaminated_mask_is_strict_superset": mix_set == asr_set | if_set
        and asr_set < mix_set
        and if_set < mix_set
        and len(mix_set) == 8,
        "copy_model_asr_ce_below_if_ce": abs(asr_ce - expected_asr) < 1e-12
        and abs(if_ce - expected_if) < 1e-12
        and abs(mix_ce - expected_mix) < 1e-12
        and asr_ce < mix_ce < if_ce,
        "labels_outside_targets_are_ignore": all(
            labels[index] == IGNORE
            for index in range(SEQUENCE_LENGTH)
            if index not in ASR_POSITIONS + IF_POSITIONS
        ),
    }

    return {
        "summary": (
            "在固定 20 位置序列上构造 ASR mask 与指令跟随 mask，"
            "核对两类有效 token 集合不相等、不相交，"
            "音频与提示不进损失，污染集合是真超集，"
            "复读模型的 ASR 交叉熵低于污染交叉熵、再低于指令交叉熵。"
        ),
        "metrics": {
            "sequence_length": SEQUENCE_LENGTH,
            "n_audio_cond": len(AUDIO_POSITIONS),
            "n_text_cond": len(PROMPT_POSITIONS),
            "n_asr_valid": len(asr_set),
            "n_if_valid": len(if_set),
            "n_contaminated_valid": len(mix_set),
            "asr_positions": sorted(asr_set),
            "if_positions": sorted(if_set),
            "contaminated_positions": sorted(mix_set),
            "asr_ce_copy_model": asr_ce,
            "if_ce_copy_model": if_ce,
            "contaminated_ce_copy_model": mix_ce,
            "asr_correct_prob": ASR_CORRECT_PROB,
            "if_copy_prob": IF_COPY_PROB,
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="43",
    title="分开语音转写和语音指令跟随",
    question="同一句语音上，ASR 交叉熵与指令跟随交叉熵的有效 token 集合是否不相等？",
    run=run,
)
