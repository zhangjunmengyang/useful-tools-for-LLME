from __future__ import annotations

import hashlib
import json
import math

from ..core import LessonExperiment


def _preference_loss(margin: float) -> float:
    return math.log1p(math.exp(-margin))


def _crop_record(pair_id: str) -> dict[str, object]:
    digest = hashlib.sha256(pair_id.encode("utf-8")).digest()
    width = 100
    height = 100
    crop_width = 40
    crop_height = 40
    x = digest[0] % (width - crop_width + 1)
    y = digest[1] % (height - crop_height + 1)
    record = {
        "pair_id": pair_id,
        "box_xywh": [x, y, crop_width, crop_height],
        "retained_area_ratio": crop_width * crop_height / (width * height),
        "implementation": "deterministic_crop_v1",
    }
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
    record["record_sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return record


def _valid_c0_record(record: dict[str, object]) -> bool:
    allowed_keys = {
        "pair_id",
        "modalities",
        "chosen",
        "rejected",
        "original_image",
        "crop_image",
        "counterfactual",
    }
    return (
        not (set(record) - allowed_keys)
        and record.get("modalities") == ["image", "text"]
        and record.get("counterfactual") == "crop_0_20"
    )


def _objective(
    policy_logps: list[float],
    reference_logps: tuple[float, float, float],
    beta: float,
) -> tuple[float, dict[str, float]]:
    chosen, rejected, crop = policy_logps
    ref_chosen, ref_rejected, ref_crop = reference_logps
    chosen_reward = beta * (chosen - ref_chosen)
    rejected_reward = beta * (rejected - ref_rejected)
    crop_reward = beta * (crop - ref_crop)
    margins = {
        "response": chosen_reward - rejected_reward,
        "copo": chosen_reward - crop_reward,
        "ancpo": chosen_reward,
    }
    total = sum(_preference_loss(value) for value in margins.values())
    return total, margins


def _finite_difference_gradient(
    values: list[float],
    reference: tuple[float, float, float],
    beta: float,
) -> list[float]:
    epsilon = 1e-6
    gradient: list[float] = []
    for index in range(len(values)):
        plus = values.copy()
        minus = values.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_loss, _ = _objective(plus, reference, beta)
        minus_loss, _ = _objective(minus, reference, beta)
        gradient.append((plus_loss - minus_loss) / (2.0 * epsilon))
    return gradient


def run() -> dict[str, object]:
    beta = 0.1
    given_toy_policy_sequence_logps = [-2.0, -3.0, -2.7]
    given_toy_reference_sequence_logps = (-2.4, -2.8, -2.5)
    initial_loss, margins = _objective(
        given_toy_policy_sequence_logps,
        given_toy_reference_sequence_logps,
        beta,
    )

    _, response_swapped_margins = _objective(
        [
            given_toy_policy_sequence_logps[1],
            given_toy_policy_sequence_logps[0],
            given_toy_policy_sequence_logps[2],
        ],
        (
            given_toy_reference_sequence_logps[1],
            given_toy_reference_sequence_logps[0],
            given_toy_reference_sequence_logps[2],
        ),
        beta,
    )
    _, crop_swapped_margins = _objective(
        [
            given_toy_policy_sequence_logps[2],
            given_toy_policy_sequence_logps[1],
            given_toy_policy_sequence_logps[0],
        ],
        (
            given_toy_reference_sequence_logps[2],
            given_toy_reference_sequence_logps[1],
            given_toy_reference_sequence_logps[0],
        ),
        beta,
    )

    gradient = _finite_difference_gradient(
        given_toy_policy_sequence_logps,
        given_toy_reference_sequence_logps,
        beta,
    )
    learning_rate = 0.5
    updated_policy = [
        value - learning_rate * derivative
        for value, derivative in zip(
            given_toy_policy_sequence_logps,
            gradient,
        )
    ]
    updated_loss, updated_margins = _objective(
        updated_policy,
        given_toy_reference_sequence_logps,
        beta,
    )

    first_crop = _crop_record("pref_img_000103")
    replayed_crop = _crop_record("pref_img_000103")
    valid_c0 = {
        "pair_id": "pref_img_000103",
        "modalities": ["image", "text"],
        "chosen": "37.50",
        "rejected": "57.50",
        "original_image": "original.png",
        "crop_image": "crop.png",
        "counterfactual": "crop_0_20",
    }
    invalid_c1 = {
        **valid_c0,
        "wrong_image": "other.png",
        "counterfactual": "wrong_media",
    }

    return {
        "summary": (
            "从手工给定的 toy sequence log-prob 计算 response DPO、CoPO "
            "和 AncPO，并对三个 policy log-prob 标量做一步有限差分更新。"
            "这里没有运行 policy/reference 模型，也没有从真实序列计算 log-prob。"
        ),
        "metrics": {
            "logprob_source": "given_toy_sequence_values_not_model_forward",
            "crop_scope": "deterministic_box_record_not_image_transform",
            "beta": beta,
            "initial_loss": round(initial_loss, 8),
            "updated_loss": round(updated_loss, 8),
            "initial_margins": {
                name: round(value, 8) for name, value in margins.items()
            },
            "updated_margins": {
                name: round(value, 8)
                for name, value in updated_margins.items()
            },
            "finite_difference_gradient": [
                round(value, 8) for value in gradient
            ],
            "crop_box_xywh": first_crop["box_xywh"],
            "crop_area_ratio": first_crop["retained_area_ratio"],
        },
        "checks": {
            "交换chosen和rejected会翻转DPO方向": math.isclose(
                response_swapped_margins["response"],
                -margins["response"],
                abs_tol=1e-12,
            ),
            "交换原图和crop会翻转CoPO方向": math.isclose(
                crop_swapped_margins["copo"],
                -margins["copo"],
                abs_tol=1e-12,
            ),
            "AncPO只读取原图chosen_reward": math.isclose(
                margins["ancpo"],
                beta
                * (
                    given_toy_policy_sequence_logps[0]
                    - given_toy_reference_sequence_logps[0]
                ),
                abs_tol=1e-12,
            ),
            "toy标量的有限差分更新降低三项目标": (
                updated_loss < initial_loss
            ),
            "给定reference标量在更新时保持不变": (
                given_toy_reference_sequence_logps
                == (-2.4, -2.8, -2.5)
            ),
            "同一pair生成相同crop记录和hash": (
                first_crop == replayed_crop
            ),
            "toy crop box面积比例在0到20百分比内": (
                0.0 < float(first_crop["retained_area_ratio"]) <= 0.20
            ),
            "C0接受论文crop字段": _valid_c0_record(valid_c0),
            "C0拒绝wrong_media扩展字段": not _valid_c0_record(invalid_c1),
        },
    }


LESSON = LessonExperiment(
    lesson_id="16",
    title="image-only mDPO",
    question="CoPO 如何只改变图像条件，而不混入回答文本差异？",
    run=run,
)
