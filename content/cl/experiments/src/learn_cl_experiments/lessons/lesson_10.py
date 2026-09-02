from __future__ import annotations

from typing import Any

import numpy as np

from ..core import LessonExperiment

SEED = 10
EPS = 1e-12


def _num(value: float) -> float:
    return float(round(float(value), 6))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(np.clip(shifted, -40.0, 40.0))
    return exp / exp.sum(axis=1, keepdims=True)


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    encoded = np.zeros((len(y), n_classes), dtype=np.float64)
    encoded[np.arange(len(y)), y] = 1.0
    return encoded


def _accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    return float((logits.argmax(axis=1) == y).mean())


def _make_task(
    rng: np.random.Generator,
    n_samples: int,
    direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    axis = direction / (np.linalg.norm(direction) + EPS)
    labels = rng.integers(0, 2, size=n_samples)
    features = rng.normal(0.0, 0.55, (n_samples, axis.size))
    features = features + 1.5 * (2 * labels - 1)[:, None] * axis
    return features, labels


def _sgd(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    lr: float,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_classes = weights.shape[1]
    for _ in range(steps):
        probs = _softmax(features @ weights + bias)
        grad = (probs - _one_hot(labels, n_classes)) / len(features)
        weights = weights - lr * (features.T @ grad)
        bias = bias - lr * grad.sum(axis=0)
    return weights, bias


def _bwt(matrix: np.ndarray) -> float:
    n_tasks = matrix.shape[0]
    return float(
        np.mean([matrix[n_tasks - 1, task] - matrix[task, task] for task in range(n_tasks - 1)]),
    )


def run() -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    n_samples = 220
    angles = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
    directions = [np.array([np.cos(angle), np.sin(angle)]) for angle in angles]
    trains = [_make_task(rng, n_samples, direction) for direction in directions]
    tests = [_make_task(rng, n_samples, direction) for direction in directions]

    weights = rng.normal(0.0, 0.1, (2, 2))
    bias = np.zeros(2)
    matrix = np.zeros((4, 4), dtype=np.float64)
    for task_i in range(4):
        weights, bias = _sgd(
            trains[task_i][0], trains[task_i][1], weights, bias, lr=0.75, steps=70,
        )
        for task_j in range(4):
            matrix[task_i, task_j] = _accuracy(
                tests[task_j][0] @ weights + bias, tests[task_j][1],
            )

    diagonal = np.diag(matrix)
    lower = matrix[np.tril_indices(4, k=-1)]
    diagonal_min = float(diagonal.min())
    lower_mean = float(lower.mean())
    avg_acc = float(matrix[-1].mean())
    backward = _bwt(matrix)

    return {
        "summary": (
            "四个互相冲突的二维方向（0°/90°/180°/270°）顺序训练，填 4×4 准确率矩阵。"
            f"对角线最低 {diagonal_min:.3f}，下三角均值 {lower_mean:.3f}，"
            f"BWT={backward:.3f}，最终平均准确率 {avg_acc:.3f}。"
            "阈值：对角线全 >0.88，下三角均值比对角线最低值低 0.12 以上，"
            "最后一行的最大值在最后一列。"
        ),
        "metrics": {
            "seed": SEED,
            "accuracy_matrix": [[_num(value) for value in row] for row in matrix],
            "diagonal_min": _num(diagonal_min),
            "diagonal_mean": _num(float(diagonal.mean())),
            "lower_triangle_mean": _num(lower_mean),
            "bwt": _num(backward),
            "average_accuracy": _num(avg_acc),
        },
        "checks": {
            "matrix_is_4x4": bool(matrix.shape == (4, 4)),
            "diagonal_all_above_0_88": bool(diagonal_min > 0.88),
            "lower_triangle_forgotten": bool(lower_mean < diagonal_min - 0.12),
            "bwt_negative": bool(backward < -0.10),
            "last_row_peaks_on_last_task": bool(int(np.argmax(matrix[-1])) == 3),
        },
    }


LESSON = LessonExperiment(
    lesson_id="10",
    title="指令任务一个接一个",
    question="4×4 准确率矩阵是否对角线高、下三角遗忘？",
    run=run,
)
