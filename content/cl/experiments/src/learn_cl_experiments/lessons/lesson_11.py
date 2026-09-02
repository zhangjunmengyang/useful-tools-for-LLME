from __future__ import annotations

from typing import Any

import numpy as np

from ..core import LessonExperiment

SEED = 11
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


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = (np.linalg.norm(left) + EPS) * (np.linalg.norm(right) + EPS)
    return float(np.dot(left, right) / denom)


def _make_task(
    rng: np.random.Generator,
    n_samples: int,
    direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    axis = direction / (np.linalg.norm(direction) + EPS)
    labels = rng.integers(0, 2, size=n_samples)
    features = rng.normal(0.0, 0.5, (n_samples, axis.size))
    features = features + 1.6 * (2 * labels - 1)[:, None] * axis
    return features, labels


def _train_lora(
    features: np.ndarray,
    labels: np.ndarray,
    base: np.ndarray,
    adapter_a: np.ndarray,
    adapter_b: np.ndarray,
    lr: float,
    steps: int,
    freeze_a: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n_classes = base.shape[1]
    for _ in range(steps):
        weights = base + np.outer(adapter_a, adapter_b)
        probs = _softmax(features @ weights)
        grad_w = features.T @ ((probs - _one_hot(labels, n_classes)) / len(features))
        adapter_a = adapter_a - lr * (grad_w @ adapter_b)
        adapter_b = adapter_b - lr * (grad_w.T @ adapter_a)
        if freeze_a is not None:
            adapter_a = (
                adapter_a
                - (np.dot(adapter_a, freeze_a) / (np.dot(freeze_a, freeze_a) + EPS))
                * freeze_a
            )
    return adapter_a, adapter_b


def run() -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    n_samples, n_dim = 260, 8
    base = rng.normal(0.0, 0.05, (n_dim, 2))
    dir_a = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dir_b = np.array([-1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    train_a, label_a = _make_task(rng, n_samples, dir_a)
    train_b, label_b = _make_task(rng, n_samples, dir_b)
    test_a, test_label_a = _make_task(rng, n_samples, dir_a)
    test_b, test_label_b = _make_task(rng, n_samples, dir_b)

    a1, b1 = rng.normal(0.0, 0.1, n_dim), rng.normal(0.0, 0.1, 2)
    a1, b1 = _train_lora(train_a, label_a, base, a1, b1, lr=0.4, steps=90)

    a2_naive, b2_naive = rng.normal(0.0, 0.1, n_dim), rng.normal(0.0, 0.1, 2)
    a2_naive, b2_naive = _train_lora(
        train_b, label_b, base, a2_naive, b2_naive, lr=0.4, steps=90,
    )
    a2_ortho, b2_ortho = rng.normal(0.0, 0.1, n_dim), rng.normal(0.0, 0.1, 2)
    a2_ortho, b2_ortho = _train_lora(
        train_b, label_b, base, a2_ortho, b2_ortho, lr=0.4, steps=90, freeze_a=a1,
    )

    naive_cos = abs(_cosine(a1, a2_naive))
    ortho_cos = abs(_cosine(a1, a2_ortho))

    acc_a1 = _accuracy(test_a @ (base + np.outer(a1, b1)), test_label_a)
    naive_stack_a = _accuracy(
        test_a @ (base + np.outer(a1, b1) + np.outer(a2_naive, b2_naive)),
        test_label_a,
    )
    ortho_stack_a = _accuracy(
        test_a @ (base + np.outer(a1, b1) + np.outer(a2_ortho, b2_ortho)),
        test_label_a,
    )
    naive_b = _accuracy(test_b @ (base + np.outer(a2_naive, b2_naive)), test_label_b)
    ortho_b = _accuracy(test_b @ (base + np.outer(a2_ortho, b2_ortho)), test_label_b)

    return {
        "summary": (
            "两个 rank-1 LoRA 方向：任务 2 与任务 1 在输入上有反向重叠。"
            f"naive 余弦 {naive_cos:.3f}，正交投影后 {ortho_cos:.3f}。"
            f"叠在一起后任务 1：naive {naive_stack_a:.3f} vs O-LoRA {ortho_stack_a:.3f}。"
            "阈值：naive |cos|>0.25，O-LoRA |cos|<0.08，O-LoRA 叠后任务 1 不低于 naive。"
        ),
        "metrics": {
            "seed": SEED,
            "task1_lora_acc": _num(acc_a1),
            "naive_abs_cosine": _num(naive_cos),
            "olora_abs_cosine": _num(ortho_cos),
            "naive_stacked_task1": _num(naive_stack_a),
            "olora_stacked_task1": _num(ortho_stack_a),
            "naive_task2": _num(naive_b),
            "olora_task2": _num(ortho_b),
        },
        "checks": {
            "task1_lora_works": bool(acc_a1 > 0.90),
            "naive_cosine_not_near_zero": bool(naive_cos > 0.25),
            "olora_cosine_near_zero": bool(ortho_cos < 0.08),
            "olora_retains_task1_at_least_as_well": bool(ortho_stack_a >= naive_stack_a),
            "olora_learns_task2": bool(ortho_b > 0.75),
        },
    }


LESSON = LessonExperiment(
    lesson_id="11",
    title="低秩更新为什么要正交",
    question="正交约束后两个 LoRA 方向的余弦是否接近 0？",
    run=run,
)
