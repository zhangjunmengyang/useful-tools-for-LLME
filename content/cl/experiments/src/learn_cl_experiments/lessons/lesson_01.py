from __future__ import annotations

from typing import Any

import numpy as np

from ..core import LessonExperiment

SEED = 1
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
    direction: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    axis = np.asarray(direction, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + EPS)
    labels = rng.integers(0, 2, size=n_samples)
    features = rng.normal(0.0, 0.85, (n_samples, axis.size))
    features = features + 1.4 * (2 * labels - 1)[:, None] * axis
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


def run() -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    n_samples = 240
    train_a, label_a = _make_task(rng, n_samples, [1.0, 0.0])
    train_b, label_b = _make_task(rng, n_samples, [0.0, 1.0])
    test_a, test_label_a = _make_task(rng, n_samples, [1.0, 0.0])
    test_b, test_label_b = _make_task(rng, n_samples, [0.0, 1.0])

    weights = rng.normal(0.0, 0.1, (2, 2))
    bias = np.zeros(2)
    weights, bias = _sgd(train_a, label_a, weights, bias, lr=0.8, steps=80)
    acc_a_after_a = _accuracy(test_a @ weights + bias, test_label_a)

    curve: list[float] = []
    weights_b, bias_b = weights.copy(), bias.copy()
    for chunk in range(9):
        if chunk:
            weights_b, bias_b = _sgd(
                train_b, label_b, weights_b, bias_b, lr=0.8, steps=10,
            )
        curve.append(_accuracy(test_a @ weights_b + bias_b, test_label_a))

    acc_a_after_b = _accuracy(test_a @ weights_b + bias_b, test_label_a)
    acc_b_after_b = _accuracy(test_b @ weights_b + bias_b, test_label_b)
    drop = acc_a_after_a - acc_a_after_b

    return {
        "summary": (
            "二维线性分类器先学沿 x 轴可分的任务 A，再学沿 y 轴可分的任务 B。"
            f"任务 A 测试准确率从 {acc_a_after_a:.3f} 降到 {acc_a_after_b:.3f}"
            f"（下降 {drop:.3f}）。阈值：A 先学到 >0.90，B 也能 >0.90，"
            "且接着训 B 后 A 下降超过 0.25 并落到 0.70 以下。"
        ),
        "metrics": {
            "seed": SEED,
            "acc_task1_after_task1": _num(acc_a_after_a),
            "acc_task1_after_task2": _num(acc_a_after_b),
            "acc_task2_after_task2": _num(acc_b_after_b),
            "task1_drop": _num(drop),
            "task1_acc_every_10_task2_steps": [_num(value) for value in curve],
        },
        "checks": {
            "task1_learned_above_0_90": bool(acc_a_after_a > 0.90),
            "task2_learned_above_0_90": bool(acc_b_after_b > 0.90),
            "task1_drop_exceeds_0_25": bool(drop > 0.25),
            "task1_after_task2_below_0_70": bool(acc_a_after_b < 0.70),
        },
    }


LESSON = LessonExperiment(
    lesson_id="01",
    title="把遗忘跑出来",
    question="同一个网络先学任务 A 再学任务 B，A 的准确率为什么会塌？",
    run=run,
)
