from __future__ import annotations

import hashlib
import json
import math
import random
import statistics

from ..core import LessonExperiment


def _flow_velocity(x: float, t: float, mean: float, sigma: float) -> float:
    mean_xt = (1.0 - t) * mean
    variance_xt = (1.0 - t) ** 2 * sigma**2 + t**2
    covariance = t - (1.0 - t) * sigma**2
    mean_velocity = -mean
    return mean_velocity + covariance / variance_xt * (x - mean_xt)


def _path_value(x0: float, noise: float, t: float) -> float:
    return (1.0 - t) * x0 + t * noise


def _reverse_sample(
    noises: list[float],
    mean: float,
    sigma: float,
    steps: int,
) -> list[float]:
    values = noises.copy()
    for step in range(steps):
        t_now = 1.0 - step / steps
        t_next = 1.0 - (step + 1) / steps
        delta = t_next - t_now
        values = [
            value + delta * _flow_velocity(value, t_now, mean, sigma)
            for value in values
        ]
    return values


def _block_mask(route: str, token_types: list[str]) -> list[list[bool]]:
    mask: list[list[bool]] = []
    for query_index, query_type in enumerate(token_types):
        row: list[bool] = []
        for key_index, key_type in enumerate(token_types):
            if route == "understand":
                if query_type == "visual":
                    allowed = key_type == "visual"
                elif key_type == "visual":
                    allowed = True
                else:
                    allowed = key_index <= query_index
            else:
                if query_type == "prompt":
                    allowed = (
                        key_type == "prompt" and key_index <= query_index
                    )
                elif key_type == "prompt":
                    allowed = True
                else:
                    allowed = True
            row.append(allowed)
        mask.append(row)
    return mask


def _patchify(latent: list[list[list[int]]]) -> list[list[int]]:
    channels = len(latent)
    height = len(latent[0])
    width = len(latent[0][0])
    tokens: list[list[int]] = []
    for top in range(0, height, 2):
        for left in range(0, width, 2):
            token: list[int] = []
            for channel in range(channels):
                for row in range(top, top + 2):
                    for column in range(left, left + 2):
                        token.append(latent[channel][row][column])
            tokens.append(token)
    return tokens


def _unpatchify(
    tokens: list[list[int]],
    channels: int,
    height: int,
    width: int,
) -> list[list[list[int]]]:
    latent = [
        [[0 for _ in range(width)] for _ in range(height)]
        for _ in range(channels)
    ]
    token_index = 0
    for top in range(0, height, 2):
        for left in range(0, width, 2):
            offset = 0
            for channel in range(channels):
                for row in range(top, top + 2):
                    for column in range(left, left + 2):
                        latent[channel][row][column] = tokens[token_index][offset]
                        offset += 1
            token_index += 1
    return latent


def _scene_manifest(seed: int, count: int) -> list[dict[str, object]]:
    generator = random.Random(seed)
    shapes = ["circle", "square", "triangle"]
    colors = ["red", "green", "blue", "yellow"]
    scenes = []
    for scene_id in range(count):
        scenes.append(
            {
                "scene_id": scene_id,
                "shape": shapes[generator.randrange(len(shapes))],
                "color": colors[generator.randrange(len(colors))],
                "x": generator.randrange(16, 241),
                "y": generator.randrange(16, 241),
                "size": generator.randrange(8, 33),
            },
        )
    return scenes


def run() -> dict[str, object]:
    mean = 2.0
    sigma = 0.5
    generator = random.Random(20260723)
    sampled_flow_pairs = []
    for _ in range(2000):
        x0 = generator.gauss(mean, sigma)
        noise = generator.gauss(0.0, 1.0)
        t = generator.random()
        xt = _path_value(x0, noise, t)
        target_velocity = noise - x0
        predicted_velocity = _flow_velocity(xt, t, mean, sigma)
        sampled_flow_pairs.append((target_velocity, predicted_velocity))
    zero_predictor_mse = statistics.fmean(
        target**2 for target, _ in sampled_flow_pairs
    )
    conditional_field_mse = statistics.fmean(
        (predicted - target) ** 2
        for target, predicted in sampled_flow_pairs
    )

    sample_generator = random.Random(41)
    initial_noise = [sample_generator.gauss(0.0, 1.0) for _ in range(2000)]
    generated = _reverse_sample(initial_noise, mean, sigma, steps=64)
    initial_distribution_error = (
        abs(statistics.fmean(initial_noise) - mean)
        + abs(statistics.pstdev(initial_noise) - sigma)
    )
    generated_distribution_error = (
        abs(statistics.fmean(generated) - mean)
        + abs(statistics.pstdev(generated) - sigma)
    )

    x0 = 1.2
    noise = -0.4
    t = 0.3
    epsilon = 1e-6
    numeric_derivative = (
        _path_value(x0, noise, t + epsilon)
        - _path_value(x0, noise, t - epsilon)
    ) / (2.0 * epsilon)
    analytic_velocity = noise - x0

    understand_types = ["visual", "visual", "text", "text"]
    understand_mask = _block_mask("understand", understand_types)
    generate_types = ["prompt", "prompt", "latent", "latent"]
    generate_mask = _block_mask("generate", generate_types)

    latent = [
        [[100 * channel + 10 * row + column for column in range(4)]
         for row in range(4)]
        for channel in range(2)
    ]
    patches = _patchify(latent)
    reconstructed = _unpatchify(patches, channels=2, height=4, width=4)

    first_manifest = _scene_manifest(20260723, 32)
    replayed_manifest = _scene_manifest(20260723, 32)
    manifest_bytes = json.dumps(
        first_manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()

    understand_rules_hold = (
        understand_mask[0][0]
        and understand_mask[0][1]
        and not understand_mask[0][2]
        and understand_mask[2][0]
        and understand_mask[2][2]
        and not understand_mask[2][3]
        and understand_mask[3][2]
    )
    generation_rules_hold = (
        generate_mask[0][0]
        and not generate_mask[0][1]
        and not generate_mask[0][2]
        and generate_mask[2][0]
        and generate_mask[2][1]
        and generate_mask[2][2]
        and generate_mask[2][3]
    )

    return {
        "summary": (
            "在一维高斯 toy 分布上执行解析向量场的反向 Euler 采样，"
            "并检查两种 attention mask 与整数 latent 的 patch round-trip。"
            "本实验没有实例化 A/B/C 三臂、没有做联合训练、没有读取真实"
            "参数账本，也不证明理解与生成使用同一个 semantic token。"
        ),
        "metrics": {
            "implementation_scope": (
                "flow_mask_patch_prerequisites_only_no_three_arm_model"
            ),
            "not_implemented": [
                "joint_training",
                "three_arm_routing",
                "semantic_token_equivalence",
                "model_parameter_ledger",
            ],
            "zero_predictor_mse": round(zero_predictor_mse, 6),
            "conditional_vector_field_mse": round(
                conditional_field_mse,
                6,
            ),
            "initial_mean": round(statistics.fmean(initial_noise), 6),
            "generated_mean": round(statistics.fmean(generated), 6),
            "target_mean": mean,
            "initial_std": round(statistics.pstdev(initial_noise), 6),
            "generated_std": round(statistics.pstdev(generated), 6),
            "target_std": sigma,
            "initial_distribution_error": round(
                initial_distribution_error,
                6,
            ),
            "generated_distribution_error": round(
                generated_distribution_error,
                6,
            ),
            "patch_token_shape": [len(patches), len(patches[0])],
            "scene_manifest_sha256": manifest_hash,
        },
        "checks": {
            "直线路径数值导数等于noise减data": math.isclose(
                numeric_derivative,
                analytic_velocity,
                rel_tol=0.0,
                abs_tol=1e-9,
            ),
            "条件向量场优于零速度基线": (
                conditional_field_mse < zero_predictor_mse
            ),
            "反向Euler把噪声分布移向数据分布": (
                generated_distribution_error < initial_distribution_error
            ),
            "UNDERSTAND遮罩符合视觉双向文本因果规则": (
                understand_rules_hold
            ),
            "GENERATE遮罩符合prompt因果latent双向规则": (
                generation_rules_hold
            ),
            "toy整数latent的patchify与unpatchify逐元素可逆": (
                reconstructed == latent
            ),
            "同一seed生成相同场景manifest": (
                first_manifest == replayed_manifest
            ),
        },
    }


LESSON = LessonExperiment(
    lesson_id="20",
    title="统一视觉理解与图像生成",
    question="理解与生成怎样共享核心参数，同时保持不同的表示和注意力规则？",
    run=run,
)
