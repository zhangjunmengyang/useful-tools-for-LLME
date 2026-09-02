from __future__ import annotations

import hashlib
import json
import math
import statistics
from typing import Any

from ..core import LessonExperiment


def _advantages(rewards: list[float], epsilon: float = 1e-8) -> list[float]:
    mean = statistics.fmean(rewards)
    variance = statistics.fmean((reward - mean) ** 2 for reward in rewards)
    standard_deviation = math.sqrt(variance)
    return [
        (reward - mean) / (standard_deviation + epsilon)
        for reward in rewards
    ]


def _clipped_surrogate(ratio: float, advantage: float, clip: float) -> float:
    clipped_ratio = min(max(ratio, 1.0 - clip), 1.0 + clip)
    return min(ratio * advantage, clipped_ratio * advantage)


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _verify_json_numeric(response: str, gold: float) -> dict[str, object]:
    if response == "__VERIFIER_ERROR__":
        return {"status": "system_error", "reward": None}
    try:
        payload = json.loads(
            response,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError):
        return {"status": "invalid_model_output", "reward": -0.2}
    if not isinstance(payload, dict) or set(payload) != {"answer"}:
        return {"status": "invalid_model_output", "reward": -0.2}
    answer = payload["answer"]
    if (
        isinstance(answer, bool)
        or not isinstance(answer, (int, float))
        or not math.isfinite(float(answer))
    ):
        return {"status": "invalid_model_output", "reward": -0.2}
    correct = math.isclose(float(answer), gold, abs_tol=1e-6)
    return {
        "status": "pass" if correct else "wrong_answer",
        "reward": 1.1 if correct else 0.1,
    }


def _update(weight: float, gradient: float, learning_rate: float = 0.1) -> float:
    return weight - learning_rate * gradient


def run() -> dict[str, object]:
    rewards = [0.0, 0.0, 1.0, 1.0]
    advantages = _advantages(rewards)
    zero_variance_advantages = _advantages([0.0, 0.0, 0.0, 0.0])

    old_logp = -1.0
    ratios = {
        "unchanged": math.exp(old_logp - old_logp),
        "inside_clip": 1.1,
        "outside_clip": 1.5,
        "far_outside_clip": 2.0,
    }
    clipped_gain = _clipped_surrogate(
        ratios["outside_clip"],
        advantage=1.0,
        clip=0.2,
    )
    far_clipped_gain = _clipped_surrogate(
        ratios["far_outside_clip"],
        advantage=1.0,
        clip=0.2,
    )

    verifier_cases = {
        "correct": _verify_json_numeric('{"answer":37.5}', 37.5),
        "equivalent": _verify_json_numeric('{"answer":37.500000}', 37.5),
        "wrong": _verify_json_numeric('{"answer":57.5}', 37.5),
        "invalid_json": _verify_json_numeric('{"answer":', 37.5),
        "nan": _verify_json_numeric('{"answer":NaN}', 37.5),
        "duplicate": _verify_json_numeric(
            '{"answer":37.5,"answer":57.5}',
            37.5,
        ),
        "system_error": _verify_json_numeric("__VERIFIER_ERROR__", 37.5),
    }
    repeated = [
        json.dumps(
            _verify_json_numeric('{"answer":37.5}', 37.5),
            sort_keys=True,
        )
        for _ in range(100)
    ]

    invalid_output_group_statuses = [
        verifier_cases["correct"]["status"],
        verifier_cases["wrong"]["status"],
        verifier_cases["invalid_json"]["status"],
    ]
    invalid_output_group_is_dropped = (
        "system_error" in invalid_output_group_statuses
    )
    system_error_group_statuses = [
        verifier_cases["correct"]["status"],
        verifier_cases["system_error"]["status"],
    ]
    system_error_group_is_dropped = (
        "system_error" in system_error_group_statuses
    )

    toy_scalar_continuous = _update(_update(1.0, 0.2), -0.1)
    toy_scalar_checkpoint = json.dumps(
        {"weight": _update(1.0, 0.2)},
        sort_keys=True,
    )
    restored_toy_scalar = float(
        json.loads(toy_scalar_checkpoint)["weight"],
    )
    toy_scalar_resumed = _update(restored_toy_scalar, -0.1)

    given_fixture_ledger = {
        "problem_id": "math_0001",
        "responses": ["37.5", "57.5"],
        "rewards": [1.0, 0.0],
        "old_logp": [-1.2, -1.4],
        "verifier": "json_numeric_v2",
    }
    ledger_bytes = json.dumps(
        given_fixture_ledger,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    first_hash = hashlib.sha256(ledger_bytes).hexdigest()
    replay_hash = hashlib.sha256(ledger_bytes).hexdigest()

    return {
        "summary": (
            "手算组内相对优势与 PPO clip，并对确定性 JSON verifier "
            "执行等价答案、非法输出、NaN、重复键和系统错误测试。非法模型"
            "输出保留低分，只有 verifier system error 才使 toy group 丢弃；"
            "保存恢复只覆盖一个 JSON 标量，不代表 optimizer state 恢复。"
        ),
        "metrics": {
            "checkpoint_scope": "single_toy_scalar_not_optimizer_state",
            "rewards": rewards,
            "advantages": [round(value, 6) for value in advantages],
            "zero_variance_advantages": [
                round(value, 6) for value in zero_variance_advantages
            ],
            "ratios": ratios,
            "clipped_positive_gain": clipped_gain,
            "verifier_statuses": {
                name: result["status"]
                for name, result in verifier_cases.items()
            },
            "toy_scalar_continuous_weight": toy_scalar_continuous,
            "toy_scalar_resumed_weight": toy_scalar_resumed,
            "given_fixture_ledger_sha256": first_hash,
        },
        "checks": {
            "组内优势均值为零": abs(statistics.fmean(advantages)) < 1e-8,
            "高低奖励产生相反优势": (
                advantages[0] < 0.0 < advantages[-1]
            ),
            "零方差组不产生更新信号": all(
                abs(value) < 1e-12 for value in zero_variance_advantages
            ),
            "新旧policy相同时ratio为一": math.isclose(
                ratios["unchanged"],
                1.0,
                abs_tol=1e-12,
            ),
            "超过clip后正优势增益不再扩大": math.isclose(
                clipped_gain,
                far_clipped_gain,
                abs_tol=1e-12,
            ),
            "等价数值答案得到相同通过状态": (
                verifier_cases["correct"] == verifier_cases["equivalent"]
            ),
            "NaN和重复键不能通过验证": (
                verifier_cases["nan"]["status"] == "invalid_model_output"
                and verifier_cases["duplicate"]["status"]
                == "invalid_model_output"
            ),
            "错误或非法模型输出保留在组内并得到低分": (
                not invalid_output_group_is_dropped
                and verifier_cases["wrong"]["reward"] == 0.1
                and verifier_cases["invalid_json"]["reward"] == -0.2
            ),
            "只有system_error fixture触发整组丢弃": (
                system_error_group_is_dropped
                and not invalid_output_group_is_dropped
            ),
            "验证器重复一百次逐字节一致": len(set(repeated)) == 1,
            "toy标量JSON保存恢复与连续计算一致": math.isclose(
                toy_scalar_continuous,
                toy_scalar_resumed,
                abs_tol=1e-12,
            ),
            "同一给定账本序列化得到相同hash": (
                first_hash == replay_hash
            ),
        },
    }


LESSON = LessonExperiment(
    lesson_id="17",
    title="多模态 GRPO / RLVR",
    question="组内奖励、裁剪更新和 verifier 故障如何共同决定一次 RL 更新？",
    run=run,
)
