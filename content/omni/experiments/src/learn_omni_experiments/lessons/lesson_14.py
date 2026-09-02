from __future__ import annotations

import json
import math
from typing import Any

from ..core import LessonExperiment


def _runtime_position_config(config: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "max_position_embeddings",
        "original_max_position_embeddings",
        "factor",
        "attention_factor",
    }
    unknown = set(config) - expected
    missing = expected - set(config)
    if unknown or missing:
        raise ValueError(
            f"position config mismatch; unknown={sorted(unknown)}, "
            f"missing={sorted(missing)}",
        )
    return {
        "max_position_embeddings": int(config["max_position_embeddings"]),
        "rope_scaling": {
            "type": "yarn",
            "original_max_position_embeddings": int(
                config["original_max_position_embeddings"],
            ),
            "factor": float(config["factor"]),
            "attention_factor": float(config["attention_factor"]),
        },
    }


def _rotate_with_uniform_position_interpolation(
    vector: list[float],
    position: int,
    base: float,
    factor: float,
) -> list[float]:
    """A uniform position-interpolation toy, not a complete YaRN transform."""
    if len(vector) % 2:
        raise ValueError("RoPE requires an even vector width")
    scaled_position = position / factor
    rotated: list[float] = []
    pair_count = len(vector) // 2
    for pair_index in range(pair_count):
        left = vector[2 * pair_index]
        right = vector[2 * pair_index + 1]
        inverse_frequency = base ** (-pair_index / pair_count)
        angle = scaled_position * inverse_frequency
        cosine = math.cos(angle)
        sine = math.sin(angle)
        rotated.extend(
            [
                left * cosine - right * sine,
                left * sine + right * cosine,
            ],
        )
    return rotated


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _learning_rate_multiplier(
    consumed_tokens: int,
    total_tokens: int,
    warmup_ratio: float,
) -> float:
    warmup_tokens = total_tokens * warmup_ratio
    if consumed_tokens <= warmup_tokens:
        return consumed_tokens / warmup_tokens
    progress = (consumed_tokens - warmup_tokens) / (
        total_tokens - warmup_tokens
    )
    progress = min(1.0, max(0.0, progress))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _validate_manifest(records: list[dict[str, Any]]) -> None:
    source_splits: dict[str, str] = {}
    for record in records:
        source_id = record["source_item_id"]
        split = record["split"]
        previous_split = source_splits.setdefault(source_id, split)
        if previous_split != split:
            raise ValueError("one source item appears in multiple splits")

        segment_ids = {segment["id"] for segment in record["segments"]}
        for evidence_id in record["evidence_ids"]:
            if evidence_id not in segment_ids:
                raise ValueError("evidence id does not resolve to a segment")
        expected_distance = (
            record["answer_token"] - record["evidence_token"]
        )
        if expected_distance != record["distance_tokens"]:
            raise ValueError("recorded evidence distance is incorrect")


def _run() -> dict[str, Any]:
    position_config = {
        "max_position_embeddings": 131_072,
        "original_max_position_embeddings": 32_768,
        "factor": 4.0,
        "attention_factor": 1.0,
    }
    runtime_config = _runtime_position_config(position_config)

    vector = [0.2, -0.5, 1.0, 0.25, -0.75, 0.4, 0.1, 0.9]
    positions = [0, 2_047, 32_767, 131_071]
    rotated = {
        position: _rotate_with_uniform_position_interpolation(
            vector,
            position,
            base=1_000_000.0,
            factor=4.0,
        )
        for position in positions
    }
    norm_errors = {
        position: abs(_norm(value) - _norm(vector))
        for position, value in rotated.items()
    }
    unscaled_long = _rotate_with_uniform_position_interpolation(
        vector,
        positions[-1],
        base=1_000_000.0,
        factor=1.0,
    )

    stages = [
        {"max_seq_len": 8_192, "stage_tokens": 280_000_000},
        {"max_seq_len": 32_768, "stage_tokens": 280_000_000},
        {"max_seq_len": 131_072, "stage_tokens": 240_000_000},
    ]
    cumulative = 0
    for stage in stages:
        cumulative += stage["stage_tokens"]
        stage["target_nonpad_tokens"] = cumulative

    token_budget = 10_000
    mixture = {
        "long_text": 0.50,
        "multi_image_text": 0.20,
        "segmented_audio_text": 0.20,
        "short_context_anchor": 0.10,
    }
    mixture_tokens = {
        name: round(token_budget * ratio) for name, ratio in mixture.items()
    }

    saved_state = {
        "consumed_nonpad_tokens": 280_000_000,
        "sampler_cursor": 173,
    }
    restored_state = json.loads(json.dumps(saved_state))
    lr_before_save = _learning_rate_multiplier(
        saved_state["consumed_nonpad_tokens"],
        800_000_000,
        0.03,
    )
    lr_after_restore = _learning_rate_multiplier(
        restored_state["consumed_nonpad_tokens"],
        800_000_000,
        0.03,
    )
    restored_state["consumed_nonpad_tokens"] += 280_000_000

    valid_manifest = [
        {
            "id": "train-1",
            "source_item_id": "source-a",
            "split": "train",
            "segments": [{"id": "evidence-alpha"}, {"id": "context"}],
            "evidence_ids": ["evidence-alpha"],
            "evidence_token": 100,
            "answer_token": 8_100,
            "distance_tokens": 8_000,
        },
        {
            "id": "test-1",
            "source_item_id": "source-b",
            "split": "test",
            "segments": [{"id": "evidence-beta"}, {"id": "context"}],
            "evidence_ids": ["evidence-beta"],
            "evidence_token": 200,
            "answer_token": 31_900,
            "distance_tokens": 31_700,
        },
    ]
    _validate_manifest(valid_manifest)

    corrupt_manifest = [
        {**valid_manifest[0], "evidence_ids": ["missing-evidence"]},
    ]
    corrupt_manifest_rejected = False
    try:
        _validate_manifest(corrupt_manifest)
    except ValueError:
        corrupt_manifest_rejected = True

    split_leak_manifest = [
        valid_manifest[0],
        {
            **valid_manifest[1],
            "source_item_id": valid_manifest[0]["source_item_id"],
        },
    ]
    split_leak_rejected = False
    try:
        _validate_manifest(split_leak_manifest)
    except ValueError:
        split_leak_rejected = True

    required_evidence = {"alpha", "beta"}
    complete_evidence = {"alpha", "beta", "irrelevant"}
    deleted_evidence = {"alpha", "irrelevant"}
    complete_answer = (
        "alpha 在 beta 之前"
        if required_evidence <= complete_evidence
        else "无法判断"
    )
    deleted_answer = (
        "alpha 在 beta 之前"
        if required_evidence <= deleted_evidence
        else "无法判断"
    )

    unknown_config_rejected = False
    try:
        _runtime_position_config({**position_config, "unused_field": 1})
    except ValueError:
        unknown_config_rejected = True

    checks = {
        "every position field reaches the runtime config": (
            runtime_config["max_position_embeddings"] == 131_072
            and runtime_config["rope_scaling"]["factor"] == 4.0
            and runtime_config["rope_scaling"][
                "original_max_position_embeddings"
            ] == 32_768
        ),
        "unknown position fields fail instead of being ignored": (
            unknown_config_rejected
        ),
        "the interpolation-only rotary toy stays finite and preserves norm": (
            all(
                math.isfinite(value)
                for output in rotated.values()
                for value in output
            )
            and max(norm_errors.values()) < 1e-10
        ),
        "uniform position interpolation changes the long-position transform": (
            max(
                abs(left - right)
                for left, right in zip(
                    rotated[positions[-1]],
                    unscaled_long,
                )
            )
            > 1e-9
        ),
        "given stage budgets produce the declared cumulative token targets": (
            [stage["target_nonpad_tokens"] for stage in stages]
            == [280_000_000, 560_000_000, 800_000_000]
        ),
        "the declared mixture arithmetic exactly spends its toy budget": (
            sum(mixture_tokens.values()) == token_budget
        ),
        "toy JSON state round-trip preserves scheduler formula inputs": (
            saved_state == json.loads(json.dumps(saved_state))
            and math.isclose(
                lr_before_save,
                lr_after_restore,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and restored_state["consumed_nonpad_tokens"] == 560_000_000
        ),
        "manifest validation rejects unresolved evidence": (
            corrupt_manifest_rejected
        ),
        "manifest validation rejects source leakage across splits": (
            split_leak_rejected
        ),
        "deleting required evidence changes the legal answer": (
            complete_answer != "无法判断"
            and deleted_answer == "无法判断"
        ),
    }

    return {
        "summary": (
            "把长上下文 recipe 拆成可核对的运行时位置配置、uniform "
            "position-interpolation-only toy、"
            "累计 token 阶段、token-balanced mixture、恢复计数和证据删除检查。"
            "这里没有实现 YaRN 的 beta_fast、beta_slow 或完整频率分区，"
            "阶段预算与 mixture 也只是声明值的算术检查，因此不声称复现"
            "完整 YaRN、数据流水线或长上下文训练。"
        ),
        "metrics": {
            "position_math_scope": "uniform_interpolation_only_not_full_yarn",
            "curriculum_scope": (
                "declared_budget_arithmetic_not_training_execution"
            ),
            "runtime_position_config": runtime_config,
            "tested_positions": positions,
            "max_interpolation_rotation_norm_error": round(
                max(norm_errors.values()),
                12,
            ),
            "stages": stages,
            "mixture_tokens": mixture_tokens,
            "lr_multiplier_at_280m_tokens": round(lr_before_save, 12),
            "resumed_consumed_nonpad_tokens": restored_state[
                "consumed_nonpad_tokens"
            ],
            "valid_manifest_records": len(valid_manifest),
            "complete_evidence_answer": complete_answer,
            "deleted_evidence_answer": deleted_answer,
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="14",
    title="渐进式长上下文与模态混合课程",
    question="位置扩展、累计 token 课程和证据使用怎样分别验证，避免只改配置就宣称成功？",
    run=_run,
)
