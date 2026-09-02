from __future__ import annotations

from typing import Any

import numpy as np

from ..core import LessonExperiment

SEED = 2
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


def _make_task(
    rng: np.random.Generator,
    n_samples: int,
    direction: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    axis = np.asarray(direction, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + EPS)
    labels = rng.integers(0, 2, size=n_samples)
    features = rng.normal(0.0, 0.55, (n_samples, axis.size))
    features = features + 1.6 * (2 * labels - 1)[:, None] * axis
    return features, labels


class _MultiHead:
    """Scalar linear backbone plus a head per task (task-incremental)."""

    def __init__(self, rng: np.random.Generator, n_dim: int, n_tasks: int) -> None:
        self.backbone = rng.normal(0.0, 0.4, (n_dim, 1))
        self.backbone_bias = np.zeros(1)
        self.heads = [rng.normal(0.0, 0.4, (1, 2)) for _ in range(n_tasks)]
        self.head_biases = [np.zeros(2) for _ in range(n_tasks)]

    def clone(self) -> _MultiHead:
        other = self.__class__.__new__(self.__class__)
        other.backbone = self.backbone.copy()
        other.backbone_bias = self.backbone_bias.copy()
        other.heads = [head.copy() for head in self.heads]
        other.head_biases = [bias.copy() for bias in self.head_biases]
        return other

    def hidden(self, features: np.ndarray) -> np.ndarray:
        return features @ self.backbone + self.backbone_bias

    def accuracy(self, features: np.ndarray, labels: np.ndarray, task: int) -> float:
        logits = self.hidden(features) @ self.heads[task] + self.head_biases[task]
        return float((logits.argmax(axis=1) == labels).mean())

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        task: int,
        lr: float,
        steps: int,
        freeze_backbone: bool,
    ) -> None:
        n_samples = len(features)
        for _ in range(steps):
            hidden = self.hidden(features)
            logits = hidden @ self.heads[task] + self.head_biases[task]
            grad_logits = (_softmax(logits) - _one_hot(labels, 2)) / n_samples
            self.heads[task] = self.heads[task] - lr * (hidden.T @ grad_logits)
            self.head_biases[task] = self.head_biases[task] - lr * grad_logits.sum(axis=0)
            if freeze_backbone:
                continue
            grad_hidden = grad_logits @ self.heads[task].T
            self.backbone = self.backbone - lr * (features.T @ grad_hidden)
            self.backbone_bias = self.backbone_bias - lr * grad_hidden.sum(axis=0)


def run() -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    n_samples = 280
    train_a, label_a = _make_task(rng, n_samples, [1.0, 0.0])
    train_b, label_b = _make_task(rng, n_samples, [0.0, 1.0])
    test_a, test_label_a = _make_task(rng, n_samples, [1.0, 0.0])
    test_b, test_label_b = _make_task(rng, n_samples, [0.0, 1.0])

    model = _MultiHead(rng, n_dim=2, n_tasks=2)
    model.train(train_a, label_a, task=0, lr=0.5, steps=120, freeze_backbone=False)
    acc_a_after_a = model.accuracy(test_a, test_label_a, 0)
    backbone_after_a = model.backbone.copy()

    naive = model.clone()
    frozen = model.clone()
    naive.train(train_b, label_b, task=1, lr=0.8, steps=200, freeze_backbone=False)
    frozen.train(train_b, label_b, task=1, lr=0.8, steps=200, freeze_backbone=True)

    naive_a = naive.accuracy(test_a, test_label_a, 0)
    naive_b = naive.accuracy(test_b, test_label_b, 1)
    freeze_a = frozen.accuracy(test_a, test_label_a, 0)
    freeze_b = frozen.accuracy(test_b, test_label_b, 1)
    naive_delta = float(np.linalg.norm(naive.backbone - backbone_after_a))
    freeze_delta = float(np.linalg.norm(frozen.backbone - backbone_after_a))

    return {
        "summary": (
            "标量线性骨干 + 每任务一个头。冻骨干后任务 A 保持 "
            f"{freeze_a:.3f}、任务 B 只有 {freeze_b:.3f}；全网接着训则 A 掉到 "
            f"{naive_a:.3f}、B 达到 {naive_b:.3f}。"
            "阈值：冻骨干比 naive 的旧任务准确率高 0.05 以上，新任务低 0.10 以上，"
            "且冻骨干的骨干位移为 0。"
        ),
        "metrics": {
            "seed": SEED,
            "acc_task1_after_task1": _num(acc_a_after_a),
            "naive_acc_task1": _num(naive_a),
            "naive_acc_task2": _num(naive_b),
            "freeze_acc_task1": _num(freeze_a),
            "freeze_acc_task2": _num(freeze_b),
            "backbone_l2_naive": _num(naive_delta),
            "backbone_l2_freeze": _num(freeze_delta),
        },
        "checks": {
            "task1_learned_above_0_90": bool(acc_a_after_a > 0.90),
            "freeze_more_stable_than_naive": bool(freeze_a > naive_a + 0.05),
            "freeze_less_plastic_than_naive": bool(freeze_b < naive_b - 0.10),
            "frozen_backbone_does_not_move": bool(freeze_delta < 1e-12),
            "naive_backbone_moves": bool(naive_delta > 0.05),
        },
    }


LESSON = LessonExperiment(
    lesson_id="02",
    title="既要记得住又要学得进",
    question="把旧的钉死，新的就学不会；放开学新的，旧的又没了。这个矛盾能不能在冻骨干上被量出来？",
    run=run,
)
