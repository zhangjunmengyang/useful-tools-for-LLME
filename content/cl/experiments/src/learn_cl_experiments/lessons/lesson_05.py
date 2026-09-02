from __future__ import annotations

from typing import Any

import numpy as np

from ..core import LessonExperiment

SEED = 5
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
    features = rng.normal(0.0, 0.7, (n_samples, axis.size))
    features = features + 1.4 * (2 * labels - 1)[:, None] * axis
    return features, labels


def _sgd(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    lr: float,
    steps: int,
    fisher_w: np.ndarray | None = None,
    fisher_b: np.ndarray | None = None,
    star_w: np.ndarray | None = None,
    star_b: np.ndarray | None = None,
    lam: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    n_classes = weights.shape[1]
    for _ in range(steps):
        probs = _softmax(features @ weights + bias)
        grad = (probs - _one_hot(labels, n_classes)) / len(features)
        weights = weights - lr * (features.T @ grad)
        bias = bias - lr * grad.sum(axis=0)
        # Proximal step on (λ/2) F (θ-θ*)^2 so large λ is stable and pins θ*.
        if fisher_w is not None and lam > 0:
            step_w = lr * lam * fisher_w
            step_b = lr * lam * fisher_b
            weights = (weights + step_w * star_w) / (1.0 + step_w)
            bias = (bias + step_b * star_b) / (1.0 + step_b)
    return weights, bias


def _fisher(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probs = _softmax(features @ weights + bias)
    residual = probs - _one_hot(labels, weights.shape[1])
    fisher_w = (features ** 2).T @ (residual ** 2) / len(features)
    fisher_b = (residual ** 2).mean(axis=0)
    return fisher_w + 1e-3, fisher_b + 1e-3


def run() -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    n_samples = 240
    train_a, label_a = _make_task(rng, n_samples, [1.0, 0.2])
    train_b, label_b = train_a.copy(), 1 - label_a
    test_a, test_label_a = _make_task(rng, n_samples, [1.0, 0.2])
    test_b, test_label_b = test_a.copy(), 1 - test_label_a

    weights = rng.normal(0.0, 0.1, (2, 2))
    bias = np.zeros(2)
    weights, bias = _sgd(train_a, label_a, weights, bias, lr=0.8, steps=80)
    acc_a = _accuracy(test_a @ weights + bias, test_label_a)
    fisher_w, fisher_b = _fisher(train_a, label_a, weights, bias)

    naive_w, naive_b = _sgd(train_b, label_b, weights.copy(), bias.copy(), lr=0.8, steps=80)
    lam0_w, lam0_b = _sgd(
        train_b, label_b, weights.copy(), bias.copy(), lr=0.8, steps=80,
        fisher_w=fisher_w, fisher_b=fisher_b, star_w=weights, star_b=bias, lam=0.0,
    )
    large_w, large_b = _sgd(
        train_b, label_b, weights.copy(), bias.copy(), lr=0.8, steps=80,
        fisher_w=fisher_w, fisher_b=fisher_b, star_w=weights, star_b=bias, lam=2.0e5,
    )

    naive_a = _accuracy(test_a @ naive_w + naive_b, test_label_a)
    naive_b_acc = _accuracy(test_b @ naive_w + naive_b, test_label_b)
    lam0_a = _accuracy(test_a @ lam0_w + lam0_b, test_label_a)
    lam0_b_acc = _accuracy(test_b @ lam0_w + lam0_b, test_label_b)
    large_a = _accuracy(test_a @ large_w + large_b, test_label_a)
    large_b_acc = _accuracy(test_b @ large_w + large_b, test_label_b)
    lam0_l2 = float(np.linalg.norm(lam0_w - naive_w) + np.linalg.norm(lam0_b - naive_b))

    return {
        "summary": (
            "任务 B 是任务 A 的标签翻转，重要权重必须动才能学会 B。"
            f"λ=0 与 naive 的权重量合（L2={lam0_l2:.1e}），A 掉到 {lam0_a:.3f}、B 到 {lam0_b_acc:.3f}；"
            f"λ=2e5 时 A 保持 {large_a:.3f}，B 只有 {large_b_acc:.3f}。"
            "阈值：λ=0 与 naive 权重 L2<1e-9；大 λ 时 A>0.85 且 B<0.30。"
        ),
        "metrics": {
            "seed": SEED,
            "acc_task1_after_task1": _num(acc_a),
            "naive_acc_task1": _num(naive_a),
            "naive_acc_task2": _num(naive_b_acc),
            "lambda0_acc_task1": _num(lam0_a),
            "lambda0_acc_task2": _num(lam0_b_acc),
            "lambda_2e5_acc_task1": _num(large_a),
            "lambda_2e5_acc_task2": _num(large_b_acc),
            "lambda0_weight_l2_vs_naive": _num(lam0_l2),
            "fisher_mean": _num(float(fisher_w.mean())),
            "lambda_large": 200000.0,
        },
        "checks": {
            "lambda0_matches_naive": bool(lam0_l2 < 1e-9),
            "lambda0_forgets_task1": bool(lam0_a < 0.30),
            "lambda0_learns_task2": bool(lam0_b_acc > 0.90),
            "large_lambda_keeps_task1": bool(large_a > 0.85),
            "large_lambda_blocks_task2": bool(large_b_acc < 0.30),
        },
    }


LESSON = LessonExperiment(
    lesson_id="05",
    title="重要的权重不许动太多",
    question="EWC 的 λ=0 是否就是 naive？λ 过大时新任务还学不学得会？",
    run=run,
)
