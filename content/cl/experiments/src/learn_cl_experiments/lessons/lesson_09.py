from __future__ import annotations

from typing import Any

import numpy as np

from ..core import LessonExperiment

SEED = 9
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
    features = rng.normal(0.0, 0.6, (n_samples, axis.size))
    features = features + 1.5 * (2 * labels - 1)[:, None] * axis
    return features, labels


def _sgd(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    lr: float,
    steps: int,
    replay_x: np.ndarray | None = None,
    replay_y: np.ndarray | None = None,
    replay_frac: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n_classes = weights.shape[1]
    for _ in range(steps):
        if replay_x is not None and replay_frac > 0:
            n_new = max(1, int(round(len(features) * (1.0 - replay_frac))))
            n_old = max(1, len(features) - n_new)
            new_idx = rng.integers(0, len(features), size=n_new)
            old_idx = rng.integers(0, len(replay_x), size=n_old)
            batch_x = np.concatenate([features[new_idx], replay_x[old_idx]], axis=0)
            batch_y = np.concatenate([labels[new_idx], replay_y[old_idx]], axis=0)
        else:
            batch_x, batch_y = features, labels
        probs = _softmax(batch_x @ weights + bias)
        grad = (probs - _one_hot(batch_y, n_classes)) / len(batch_x)
        weights = weights - lr * (batch_x.T @ grad)
        bias = bias - lr * grad.sum(axis=0)
    return weights, bias


def run() -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    n_samples, n_dim = 280, 6
    general_dir = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    domain_dir = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    train_g, label_g = _make_task(rng, n_samples, general_dir)
    train_d, label_d = _make_task(rng, n_samples, domain_dir)
    test_g, test_label_g = _make_task(rng, n_samples, general_dir)
    test_d, test_label_d = _make_task(rng, n_samples, domain_dir)

    weights = rng.normal(0.0, 0.1, (n_dim, 2))
    bias = np.zeros(2)
    weights, bias = _sgd(train_g, label_g, weights, bias, lr=0.6, steps=60)
    general_before = _accuracy(test_g @ weights + bias, test_label_g)

    domain_w, domain_b = _sgd(train_d, label_d, weights.copy(), bias.copy(), lr=0.6, steps=80)
    mix_rng = np.random.default_rng(91)
    mix_w, mix_b = _sgd(
        train_d, label_d, weights.copy(), bias.copy(), lr=0.6, steps=80,
        replay_x=train_g, replay_y=label_g, replay_frac=0.3, rng=mix_rng,
    )

    domain_only_general = _accuracy(test_g @ domain_w + domain_b, test_label_g)
    domain_only_domain = _accuracy(test_d @ domain_w + domain_b, test_label_d)
    mix_general = _accuracy(test_g @ mix_w + mix_b, test_label_g)
    mix_domain = _accuracy(test_d @ mix_w + mix_b, test_label_d)
    drop_domain_only = general_before - domain_only_general
    drop_mix = general_before - mix_general

    return {
        "summary": (
            "通用方向预训练后接到窄领域。"
            f"纯领域续训通用指标从 {general_before:.3f} 掉 {drop_domain_only:.3f}；"
            f"每 batch 混 30% 通用数据只掉 {drop_mix:.3f}，领域仍有 {mix_domain:.3f}。"
            "阈值：纯领域通用下降 >0.15；混数据下降再少 0.08 以上；混数据领域 >0.85。"
        ),
        "metrics": {
            "seed": SEED,
            "mix_general_frac": 0.3,
            "general_before": _num(general_before),
            "domain_only_general": _num(domain_only_general),
            "domain_only_domain": _num(domain_only_domain),
            "mix_general": _num(mix_general),
            "mix_domain": _num(mix_domain),
            "drop_domain_only": _num(drop_domain_only),
            "drop_mix": _num(drop_mix),
        },
        "checks": {
            "pretrained_general_above_0_90": bool(general_before > 0.90),
            "domain_only_general_drops": bool(drop_domain_only > 0.15),
            "mix_drops_less_than_domain_only": bool(drop_mix < drop_domain_only - 0.08),
            "mix_still_learns_domain": bool(mix_domain > 0.85),
            "domain_only_learns_domain": bool(domain_only_domain > 0.90),
        },
    }


LESSON = LessonExperiment(
    lesson_id="09",
    title="换领域时旧能力怎么掉",
    question="混入通用数据之后，通用指标是否掉得更少？",
    run=run,
)
