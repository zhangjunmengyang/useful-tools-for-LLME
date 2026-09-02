from __future__ import annotations

import math
import statistics
from typing import Any

from ..core import LessonExperiment


EPSILON = 1e-8
CONTACT_BAND = 0.03
APPROACH_TAU = 0.04
FORCE_SAFE = 0.45
LEARNING_RATE = 0.1


def sparse_success(lifted: bool, in_bin: bool) -> float:
    return 1.0 if lifted and in_bin else 0.0


def dense_contact_reward(
    clearance: float,
    gripper_closed: bool,
    lifted: bool,
    force: float,
    *,
    tau: float = APPROACH_TAU,
    contact_band: float = CONTACT_BAND,
    force_safe: float = FORCE_SAFE,
) -> float:
    approach = math.exp(-clearance / tau)
    contact = 1.0 if gripper_closed and clearance <= contact_band else 0.0
    lift = 1.0 if lifted else 0.0
    force_penalty = max(0.0, force - force_safe)
    reward = 0.35 * approach + 0.25 * contact + 0.40 * lift - 0.30 * force_penalty
    return max(0.0, min(1.0, reward))


def mean_advantages(rewards: list[float]) -> list[float]:
    baseline = statistics.fmean(rewards)
    return [reward - baseline for reward in rewards]


def standardized_advantages(
    rewards: list[float],
    epsilon: float = EPSILON,
) -> list[float]:
    baseline = statistics.fmean(rewards)
    variance = statistics.fmean((reward - baseline) ** 2 for reward in rewards)
    deviation = math.sqrt(variance)
    return [(reward - baseline) / (deviation + epsilon) for reward in rewards]


def reward_variance(rewards: list[float]) -> float:
    baseline = statistics.fmean(rewards)
    return statistics.fmean((reward - baseline) ** 2 for reward in rewards)


def apply_advantage_update(
    weight: float,
    advantage: float,
    feature: float = 1.0,
    learning_rate: float = LEARNING_RATE,
) -> float:
    return weight + learning_rate * advantage * feature


def ranking_ids(items: list[tuple[str, float]]) -> list[str]:
    return [
        item_id
        for item_id, _ in sorted(items, key=lambda item: (-item[1], item[0]))
    ]


def run() -> dict[str, Any]:
    fail_group = [
        {
            "id": "A",
            "name": "空抓",
            "clearance": 0.18,
            "gripper_closed": False,
            "lifted": False,
            "in_bin": False,
            "force": 0.0,
        },
        {
            "id": "B",
            "name": "悬停",
            "clearance": 0.055,
            "gripper_closed": False,
            "lifted": False,
            "in_bin": False,
            "force": 0.02,
        },
        {
            "id": "C",
            "name": "擦边",
            "clearance": 0.018,
            "gripper_closed": True,
            "lifted": False,
            "in_bin": False,
            "force": 0.22,
        },
        {
            "id": "D",
            "name": "压溃",
            "clearance": 0.006,
            "gripper_closed": True,
            "lifted": False,
            "in_bin": False,
            "force": 1.35,
        },
    ]
    mixed_group = fail_group[:3] + [
        {
            "id": "E",
            "name": "放入",
            "clearance": 0.02,
            "gripper_closed": True,
            "lifted": True,
            "in_bin": True,
            "force": 0.18,
        },
    ]

    sparse_fail = [
        sparse_success(item["lifted"], item["in_bin"]) for item in fail_group
    ]
    dense_fail = [
        dense_contact_reward(
            item["clearance"],
            item["gripper_closed"],
            item["lifted"],
            item["force"],
        )
        for item in fail_group
    ]
    sparse_mixed = [
        sparse_success(item["lifted"], item["in_bin"]) for item in mixed_group
    ]
    dense_mixed = [
        dense_contact_reward(
            item["clearance"],
            item["gripper_closed"],
            item["lifted"],
            item["force"],
        )
        for item in mixed_group
    ]

    sparse_fail_adv = mean_advantages(sparse_fail)
    dense_fail_adv = mean_advantages(dense_fail)
    sparse_mixed_adv = mean_advantages(sparse_mixed)
    dense_mixed_adv = mean_advantages(dense_mixed)
    sparse_fail_std = standardized_advantages(sparse_fail)
    dense_fail_std = standardized_advantages(dense_fail)

    start_weight = 1.0
    sparse_fail_weights = [
        apply_advantage_update(start_weight, advantage)
        for advantage in sparse_fail_adv
    ]
    dense_fail_weights = [
        apply_advantage_update(start_weight, advantage)
        for advantage in dense_fail_adv
    ]

    dense_fail_ranked = ranking_ids(list(zip(
        [item["id"] for item in fail_group],
        dense_fail,
    )))
    sparse_mixed_ranked = ranking_ids(list(zip(
        [item["id"] for item in mixed_group],
        sparse_mixed,
    )))

    all_success_sparse = [1.0, 1.0, 1.0, 1.0]
    all_success_adv = mean_advantages(all_success_sparse)

    return {
        "summary": (
            "同一抓取组对照稀疏成功与接触 dense。"
            "全失败批次上稀疏奖励方差为零、组内优势与标量更新均为零；"
            "dense 仍能按接近、接触和力超限排序，并给出非零更新。"
        ),
        "metrics": {
            "sparse_fail_rewards": [round(value, 6) for value in sparse_fail],
            "dense_fail_rewards": [round(value, 6) for value in dense_fail],
            "sparse_fail_advantages": [
                round(value, 6) for value in sparse_fail_adv
            ],
            "dense_fail_advantages": [
                round(value, 6) for value in dense_fail_adv
            ],
            "sparse_fail_variance": round(reward_variance(sparse_fail), 8),
            "dense_fail_variance": round(reward_variance(dense_fail), 8),
            "sparse_mixed_rewards": [round(value, 6) for value in sparse_mixed],
            "dense_mixed_rewards": [round(value, 6) for value in dense_mixed],
            "sparse_mixed_advantages": [
                round(value, 6) for value in sparse_mixed_adv
            ],
            "dense_mixed_advantages": [
                round(value, 6) for value in dense_mixed_adv
            ],
            "dense_fail_ranked": dense_fail_ranked,
            "sparse_mixed_ranked": sparse_mixed_ranked,
            "sparse_fail_updated_weights": [
                round(value, 6) for value in sparse_fail_weights
            ],
            "dense_fail_updated_weights": [
                round(value, 6) for value in dense_fail_weights
            ],
            "standardized_sparse_fail": [
                round(value, 6) for value in sparse_fail_std
            ],
        },
        "checks": {
            "稀疏失败组奖励全为零": all(value == 0.0 for value in sparse_fail),
            "稀疏失败组方差为零": reward_variance(sparse_fail) == 0.0,
            "零方差组不产生更新": all(
                abs(value) < 1e-12 for value in sparse_fail_adv
            )
            and all(
                math.isclose(weight, start_weight, abs_tol=1e-12)
                for weight in sparse_fail_weights
            ),
            "标准化零方差组优势也接近零": all(
                abs(value) < 1e-8 for value in sparse_fail_std
            ),
            "dense失败轨迹仍可排序": (
                len(set(round(value, 6) for value in dense_fail)) == 4
                and dense_fail_ranked == ["C", "D", "B", "A"]
            ),
            "dense失败组仍有非零更新": (
                reward_variance(dense_fail) > 1e-6
                and any(abs(value) > 1e-6 for value in dense_fail_adv)
                and any(
                    not math.isclose(weight, start_weight, abs_tol=1e-9)
                    for weight in dense_fail_weights
                )
            ),
            "组内均值优势之和为零": (
                abs(sum(dense_fail_adv)) < 1e-12
                and abs(sum(sparse_mixed_adv)) < 1e-12
                and abs(sum(dense_mixed_adv)) < 1e-12
            ),
            "全成功稀疏组同样没有相对优势": all(
                abs(value) < 1e-12 for value in all_success_adv
            ),
            "混入成功后稀疏才能分出正负": (
                sparse_mixed_ranked[0] == "E"
                and sparse_mixed_adv[3] > 0.0
                and all(value < 0.0 for value in sparse_mixed_adv[:3])
            ),
            "dense标准化优势在失败组仍可分正负": (
                max(dense_fail_std) > 0.0 > min(dense_fail_std)
            ),
        },
    }


LESSON = LessonExperiment(
    lesson_id="38",
    title="给 VLA 接上可验证强化学习",
    question="稀疏成功与接触 dense 如何改变同一组失败轨迹上的组内优势与更新？",
    run=run,
)
