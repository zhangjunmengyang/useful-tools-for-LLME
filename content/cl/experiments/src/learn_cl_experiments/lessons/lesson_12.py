from __future__ import annotations

from typing import Any

import numpy as np

from ..core import LessonExperiment

SEED = 12
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
    features = rng.normal(0.0, 0.5, (n_samples, axis.size))
    features = features + 1.6 * (2 * labels - 1)[:, None] * axis
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


def _flatten(weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.concatenate([weights.ravel(), bias.ravel()])


def _unflatten(
    flat: np.ndarray,
    base_w: np.ndarray,
    base_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_w = base_w.size
    return base_w + flat[:n_w].reshape(base_w.shape), base_b + flat[n_w:]


def _project(vector: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return (np.dot(vector, axis) / (np.dot(axis, axis) + EPS)) * axis


def run() -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    n_samples = 240
    train_a, label_a = _make_task(rng, n_samples, [1.0, 0.0])
    train_b, label_b = _make_task(rng, n_samples, [0.0, 1.0])
    test_a, test_label_a = _make_task(rng, n_samples, [1.0, 0.0])
    test_b, test_label_b = _make_task(rng, n_samples, [0.0, 1.0])

    base_w = rng.normal(0.0, 0.02, (2, 2))
    base_b = np.zeros(2)
    w1, b1 = _sgd(train_a, label_a, base_w.copy(), base_b.copy(), lr=0.7, steps=80)
    w2, b2 = _sgd(train_b, label_b, base_w.copy(), base_b.copy(), lr=0.7, steps=80)

    tau1 = _flatten(w1 - base_w, b1 - base_b)
    tau2 = _flatten(w2 - base_w, b2 - base_b)
    summed = tau1 + tau2
    add_w, add_b = _unflatten(summed, base_w, base_b)
    proj1_w, proj1_b = _unflatten(_project(summed, tau1), base_w, base_b)
    proj2_w, proj2_b = _unflatten(_project(summed, tau2), base_w, base_b)
    neg_w, neg_b = _unflatten(-tau1, base_w, base_b)

    acc_add_a = _accuracy(test_a @ add_w + add_b, test_label_a)
    acc_add_b = _accuracy(test_b @ add_w + add_b, test_label_b)
    acc_only1_a = _accuracy(test_a @ w1 + b1, test_label_a)
    acc_only1_b = _accuracy(test_b @ w1 + b1, test_label_b)
    acc_only2_a = _accuracy(test_a @ w2 + b2, test_label_a)
    acc_only2_b = _accuracy(test_b @ w2 + b2, test_label_b)
    acc_proj1_b = _accuracy(test_b @ proj1_w + proj1_b, test_label_b)
    acc_proj2_a = _accuracy(test_a @ proj2_w + proj2_b, test_label_a)
    acc_neg_a = _accuracy(test_a @ neg_w + neg_b, test_label_a)
    tau_cos = float(
        np.dot(tau1, tau2)
        / ((np.linalg.norm(tau1) + EPS) * (np.linalg.norm(tau2) + EPS)),
    )

    return {
        "summary": (
            "两个正交二维任务各自从同一原点微调，得到任务向量 τ1、τ2。"
            f"相加后两任务准确率 {acc_add_a:.3f} / {acc_add_b:.3f}，"
            f"好于单任务模型在另一任务上的 {acc_only2_a:.3f} / {acc_only1_b:.3f}，"
            f"也好于把和投影到单一任务向量（{acc_proj2_a:.3f} / {acc_proj1_b:.3f}）。"
            "阈值：相加在两任务上都 >0.85，并比单任务/单任务投影至少高 0.10。"
        ),
        "metrics": {
            "seed": SEED,
            "acc_add_task1": _num(acc_add_a),
            "acc_add_task2": _num(acc_add_b),
            "acc_task1_only_on_task1": _num(acc_only1_a),
            "acc_task1_only_on_task2": _num(acc_only1_b),
            "acc_task2_only_on_task1": _num(acc_only2_a),
            "acc_task2_only_on_task2": _num(acc_only2_b),
            "acc_proj_tau1_on_task2": _num(acc_proj1_b),
            "acc_proj_tau2_on_task1": _num(acc_proj2_a),
            "acc_neg_tau1_on_task1": _num(acc_neg_a),
            "task_vector_cosine": _num(tau_cos),
        },
        "checks": {
            "add_beats_other_single_on_task1": bool(acc_add_a > acc_only2_a + 0.15),
            "add_beats_other_single_on_task2": bool(acc_add_b > acc_only1_b + 0.15),
            "add_beats_projection_on_task1": bool(acc_add_a > acc_proj2_a + 0.10),
            "add_beats_projection_on_task2": bool(acc_add_b > acc_proj1_b + 0.10),
            "add_both_tasks_above_0_85": bool(acc_add_a > 0.85 and acc_add_b > 0.85),
            "negative_task_vector_hurts_task1": bool(acc_neg_a < acc_only1_a - 0.20),
        },
    }


LESSON = LessonExperiment(
    lesson_id="12",
    title="不接着训，把几个模型加起来",
    question="任务向量相加是否能让两个合成任务都好于单任务投影？",
    run=run,
)
