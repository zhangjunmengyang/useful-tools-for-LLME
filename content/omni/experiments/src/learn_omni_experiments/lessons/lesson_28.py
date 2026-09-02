from __future__ import annotations

import math
from typing import Any

from ..core import LessonExperiment


Action = list[list[float]]


def _add(left: Action, right: Action) -> Action:
    return [
        [left_value + right_value for left_value, right_value in zip(left_row, right_row)]
        for left_row, right_row in zip(left, right)
    ]


def _scale(action: Action, coefficient: float) -> Action:
    return [[coefficient * value for value in row] for row in action]


def _sub(left: Action, right: Action) -> Action:
    return _add(left, _scale(right, -1.0))


def _path_lesson20(clean: Action, noise: Action, time: float) -> Action:
    return _add(_scale(clean, 1.0 - time), _scale(noise, time))


def _path_pi0(clean: Action, noise: Action, tau: float) -> Action:
    return _add(_scale(clean, tau), _scale(noise, 1.0 - tau))


def _velocity_lesson20(clean: Action, noise: Action) -> Action:
    return _sub(noise, clean)


def _velocity_pi0(clean: Action, noise: Action) -> Action:
    return _sub(clean, noise)


def _frobenius(action: Action) -> float:
    return math.sqrt(sum(value * value for row in action for value in row))


def _almost_equal(left: Action, right: Action, abs_tol: float = 1e-12) -> bool:
    return all(
        math.isclose(left_value, right_value, rel_tol=0.0, abs_tol=abs_tol)
        for left_row, right_row in zip(left, right)
        for left_value, right_value in zip(left_row, right_row)
    )


def _euler(current: Action, velocity: Action, delta: float) -> Action:
    return _add(current, _scale(velocity, delta))


def _integrate_lesson20(
    start: Action,
    clean: Action,
    noise: Action,
    steps: int,
    delta: float = -0.10,
) -> Action:
    current = [row[:] for row in start]
    for _ in range(steps):
        current = _euler(current, _velocity_lesson20(clean, noise), delta)
    return current


def run() -> dict[str, Any]:
    clean: Action = [
        [0.20, 0.10],
        [0.80, 0.30],
    ]
    noise: Action = [
        [1.00, -0.50],
        [0.00, 1.00],
    ]
    time = 0.40
    epsilon = 1e-6

    path = _path_lesson20(clean, noise, time)
    expected_path = [
        [0.52, -0.14],
        [0.48, 0.58],
    ]
    target_velocity = _velocity_lesson20(clean, noise)
    expected_velocity = [
        [0.80, -0.60],
        [-0.80, 0.70],
    ]
    numeric_velocity = _scale(
        _sub(
            _path_lesson20(clean, noise, time + epsilon),
            _path_lesson20(clean, noise, time - epsilon),
        ),
        1.0 / (2.0 * epsilon),
    )

    start_from_noise = _path_lesson20(clean, noise, 1.0)
    correct_next = _euler(start_from_noise, target_velocity, -0.10)
    wrong_next = _euler(start_from_noise, _scale(target_velocity, -1.0), -0.10)
    distance_before = _frobenius(_sub(start_from_noise, clean))
    distance_correct = _frobenius(_sub(correct_next, clean))
    distance_wrong = _frobenius(_sub(wrong_next, clean))

    pi0_next = _euler(noise, _velocity_pi0(clean, noise), 0.10)
    two_steps = _integrate_lesson20(noise, clean, noise, steps=2)
    ten_steps = _integrate_lesson20(noise, clean, noise, steps=10)
    distance_two = _frobenius(_sub(two_steps, clean))
    distance_ten = _frobenius(_sub(ten_steps, clean))

    return {
        "summary": (
            "在 H=2、d=2 的动作块上核对直线路径速度、单步 Euler 方向，"
            "以及第 20 课时间箭头与 π0 时间箭头的等价。样本是动作向量，不是图像 latent。"
        ),
        "metrics": {
            "chunk_shape": [len(clean), len(clean[0])],
            "lesson20_time": time,
            "path_00": path[0][0],
            "path_01": path[0][1],
            "path_10": path[1][0],
            "path_11": path[1][1],
            "velocity_00": target_velocity[0][0],
            "velocity_01": target_velocity[0][1],
            "velocity_10": target_velocity[1][0],
            "velocity_11": target_velocity[1][1],
            "distance_before": round(distance_before, 6),
            "distance_after_correct_step": round(distance_correct, 6),
            "distance_after_flipped_step": round(distance_wrong, 6),
            "distance_after_2_steps": round(distance_two, 6),
            "distance_after_10_steps": round(distance_ten, 6),
        },
        "checks": {
            "手算路径点与公式一致": _almost_equal(path, expected_path),
            "直线路径数值导数等于噪声减干净动作": _almost_equal(
                numeric_velocity,
                expected_velocity,
                abs_tol=1e-8,
            ),
            "正确速度的一步Euler缩小到干净动作的距离": (
                distance_correct < distance_before - 1e-9
            ),
            "反号速度的一步Euler加大到干净动作的距离": (
                distance_wrong > distance_before + 1e-9
            ),
            "第二十课约定与pi0约定一步走到同一状态": _almost_equal(
                correct_next,
                pi0_next,
            ),
            "更多积分步更接近干净动作块": distance_ten < distance_two,
            "动作块形状保持为H乘d": (
                len(path) == 2
                and all(len(row) == 2 for row in path)
                and len(ten_steps) == 2
                and all(len(row) == 2 for row in ten_steps)
            ),
        },
    }


LESSON = LessonExperiment(
    lesson_id="28",
    title="用流匹配生成连续动作块",
    question="动作块上的直线路径速度场，怎样保证积分把噪声推向干净动作而不是反向？",
    run=run,
)
