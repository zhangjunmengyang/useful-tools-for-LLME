from __future__ import annotations

from typing import Any

import numpy as np

from ..core import LessonExperiment

SEED = 7


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


def _make_slice_task(
    rng: np.random.Generator,
    n_samples: int,
    n_dim: int,
    start: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = rng.integers(0, 2, size=n_samples)
    features = rng.normal(0.0, 0.4, (n_samples, n_dim))
    direction = np.zeros(n_dim)
    direction[start : start + width] = 1.0 / np.sqrt(width)
    features = features + 1.7 * (2 * labels - 1)[:, None] * direction
    return features, labels


class _MLP:
    def __init__(self, rng: np.random.Generator, n_dim: int, hidden: int) -> None:
        self.w1 = rng.normal(0.0, 0.25, (n_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0.0, 0.25, (hidden, 2))
        self.b2 = np.zeros(2)

    def tensors(self) -> dict[str, np.ndarray]:
        return {"w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2}

    def clone_from(self, snapshot: dict[str, np.ndarray]) -> None:
        self.w1 = snapshot["w1"].copy()
        self.b1 = snapshot["b1"].copy()
        self.w2 = snapshot["w2"].copy()
        self.b2 = snapshot["b2"].copy()

    def accuracy(self, features: np.ndarray, labels: np.ndarray) -> float:
        hidden = np.maximum(features @ self.w1 + self.b1, 0.0)
        logits = hidden @ self.w2 + self.b2
        return float((logits.argmax(axis=1) == labels).mean())

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        lr: float,
        steps: int,
        mask: dict[str, np.ndarray] | None = None,
    ) -> None:
        n_samples = len(features)
        for _ in range(steps):
            pre = features @ self.w1 + self.b1
            hidden = np.maximum(pre, 0.0)
            logits = hidden @ self.w2 + self.b2
            grad_logits = (_softmax(logits) - _one_hot(labels, 2)) / n_samples
            grad_w2 = hidden.T @ grad_logits
            grad_b2 = grad_logits.sum(axis=0)
            grad_hidden = (grad_logits @ self.w2.T) * (pre > 0)
            grad_w1 = features.T @ grad_hidden
            grad_b1 = grad_hidden.sum(axis=0)
            if mask is not None:
                grad_w1 = grad_w1 * mask["w1"]
                grad_b1 = grad_b1 * mask["b1"]
                grad_w2 = grad_w2 * mask["w2"]
                grad_b2 = grad_b2 * mask["b2"]
            self.w1 = self.w1 - lr * grad_w1
            self.b1 = self.b1 - lr * grad_b1
            self.w2 = self.w2 - lr * grad_w2
            self.b2 = self.b2 - lr * grad_b2


def run() -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    n_samples, n_dim, hidden = 280, 6, 10
    train_a, label_a = _make_slice_task(rng, n_samples, n_dim, 0, 3)
    train_b, label_b = _make_slice_task(rng, n_samples, n_dim, 3, 3)
    test_a, test_label_a = _make_slice_task(rng, n_samples, n_dim, 0, 3)
    test_b, test_label_b = _make_slice_task(rng, n_samples, n_dim, 3, 3)

    net = _MLP(rng, n_dim, hidden)
    net.train(train_a, label_a, lr=0.35, steps=100)
    acc_a = net.accuracy(test_a, test_label_a)

    tensors = net.tensors()
    magnitudes = np.concatenate([np.abs(value).ravel() for value in tensors.values()])
    threshold = float(np.quantile(magnitudes, 0.5))
    occupied = {name: np.abs(value) >= threshold for name, value in tensors.items()}
    free = {name: np.logical_not(mask) for name, mask in occupied.items()}
    occupied_frac = float(
        np.mean(np.concatenate([mask.ravel() for mask in occupied.values()])),
    )
    # Magnitude pruning: occupied weights are the large ones at freeze time.
    occupied_abs = np.concatenate(
        [np.abs(tensors[name])[occupied[name]].ravel() for name in tensors if occupied[name].any()],
    )
    free_abs = np.concatenate(
        [np.abs(tensors[name])[free[name]].ravel() for name in tensors if free[name].any()],
    )
    occupied_min = float(occupied_abs.min())
    free_max = float(free_abs.max())

    for name, value in tensors.items():
        value *= occupied[name]
    net.train(train_a, label_a, lr=0.35, steps=40, mask=occupied)
    acc_a_pruned = net.accuracy(test_a, test_label_a)
    snapshot = {name: value.copy() for name, value in net.tensors().items()}

    net.train(train_b, label_b, lr=0.35, steps=80, mask=free)
    moved_occupied = max(
        float(np.max(np.abs(net.tensors()[name] - snapshot[name]) * occupied[name]))
        for name in snapshot
    )
    moved_free = max(
        float(np.max(np.abs(net.tensors()[name] - snapshot[name]) * free[name]))
        for name in snapshot
    )

    killed_occupied = _MLP(rng, n_dim, hidden)
    killed_free = _MLP(rng, n_dim, hidden)
    killed_occupied.clone_from(snapshot)
    killed_free.clone_from(snapshot)
    for name, value in killed_occupied.tensors().items():
        value *= free[name]
    for name, value in killed_free.tensors().items():
        value *= occupied[name]
    acc_if_zero_occupied = killed_occupied.accuracy(test_a, test_label_a)
    acc_if_zero_free = killed_free.accuracy(test_a, test_label_a)
    acc_a_after_b = net.accuracy(test_a, test_label_a)
    acc_b_after_b = net.accuracy(test_b, test_label_b)

    return {
        "summary": (
            "MLP 上做 PackNet：按幅度剪掉 50% 最小权重，剩下的标成任务 1 占用并上锁。"
            f"占用比例 {occupied_frac:.3f}，任务 2 时占用权重位移 {moved_occupied:.1e}、"
            f"空闲权重位移 {moved_free:.3f}。清掉占用后任务 1 准确率 {acc_if_zero_occupied:.3f}，"
            f"清掉空闲仍有 {acc_if_zero_free:.3f}。"
            "阈值：占用约一半；占用权重冻结；任务 1 知识在占用集合里。"
        ),
        "metrics": {
            "seed": SEED,
            "acc_task1_before_prune": _num(acc_a),
            "acc_task1_after_prune": _num(acc_a_pruned),
            "acc_task1_after_task2": _num(acc_a_after_b),
            "acc_task2_after_task2": _num(acc_b_after_b),
            "occupied_fraction": _num(occupied_frac),
            "prune_threshold": _num(threshold),
            "occupied_min_abs": _num(occupied_min),
            "free_max_abs_at_prune": _num(free_max),
            "occupied_max_update": _num(moved_occupied),
            "free_max_update": _num(moved_free),
            "task1_if_zero_occupied": _num(acc_if_zero_occupied),
            "task1_if_zero_free": _num(acc_if_zero_free),
        },
        "checks": {
            "occupied_fraction_near_half": bool(0.35 < occupied_frac < 0.65),
            "occupied_are_larger_than_free": bool(occupied_min >= free_max - 1e-12),
            "occupied_weights_frozen": bool(moved_occupied < 1e-12),
            "free_weights_moved": bool(moved_free > 1e-4),
            "task1_lives_in_occupied_mask": bool(
                acc_if_zero_occupied < 0.70 and acc_if_zero_free > 0.85,
            ),
            "task1_kept_after_task2": bool(acc_a_after_b > 0.80),
        },
    }


LESSON = LessonExperiment(
    lesson_id="07",
    title="不改旧权重就再长一块",
    question="PackNet 如何把任务 1 占用的权重量成占用？",
    run=run,
)
