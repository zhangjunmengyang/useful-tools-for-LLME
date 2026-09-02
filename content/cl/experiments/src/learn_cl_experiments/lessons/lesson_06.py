from __future__ import annotations

from typing import Any

import numpy as np

from ..core import LessonExperiment

SEED = 6
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
    n_samples = 240
    train_a, label_a = _make_task(rng, n_samples, [1.0, 0.0])
    train_b, label_b = _make_task(rng, n_samples, [0.0, 1.0])
    test_a, test_label_a = _make_task(rng, n_samples, [1.0, 0.0])
    test_b, test_label_b = _make_task(rng, n_samples, [0.0, 1.0])

    weights = rng.normal(0.0, 0.1, (2, 2))
    bias = np.zeros(2)
    weights, bias = _sgd(train_a, label_a, weights, bias, lr=0.7, steps=80)
    acc_a = _accuracy(test_a @ weights + bias, test_label_a)

    none_w, none_b = _sgd(train_b, label_b, weights.copy(), bias.copy(), lr=0.7, steps=80)
    buffer_idx = rng.choice(n_samples, size=80, replace=False)
    replay_w, replay_b = _sgd(
        train_b, label_b, weights.copy(), bias.copy(), lr=0.7, steps=80,
        replay_x=train_a[buffer_idx], replay_y=label_a[buffer_idx],
        replay_frac=0.5, rng=rng,
    )

    none_a = _accuracy(test_a @ none_w + none_b, test_label_a)
    none_b_acc = _accuracy(test_b @ none_w + none_b, test_label_b)
    replay_a = _accuracy(test_a @ replay_w + replay_b, test_label_a)
    replay_b_acc = _accuracy(test_b @ replay_w + replay_b, test_label_b)
    drop_none = acc_a - none_a
    drop_replay = acc_a - replay_a

    return {
        "summary": (
            "同一对二维任务，缓冲 80 条旧样本、每步混 50% 回放。"
            f"无回放任务 A 从 {acc_a:.3f} 掉到 {none_a:.3f}（下降 {drop_none:.3f}），"
            f"有回放掉到 {replay_a:.3f}（下降 {drop_replay:.3f}），且任务 B 仍有 {replay_b_acc:.3f}。"
            "阈值：无回放下降比有回放至少多 0.08；有回放 A、B 都 >0.80；无回放 A<0.75。"
        ),
        "metrics": {
            "seed": SEED,
            "buffer_size": 80,
            "replay_frac": 0.5,
            "acc_task1_after_task1": _num(acc_a),
            "no_replay_acc_task1": _num(none_a),
            "no_replay_acc_task2": _num(none_b_acc),
            "replay_acc_task1": _num(replay_a),
            "replay_acc_task2": _num(replay_b_acc),
            "drop_no_replay": _num(drop_none),
            "drop_replay": _num(drop_replay),
        },
        "checks": {
            "no_replay_forgets_more": bool(drop_none > drop_replay + 0.08),
            "no_replay_task1_below_0_75": bool(none_a < 0.75),
            "replay_keeps_task1_above_0_80": bool(replay_a > 0.80),
            "replay_learns_task2_above_0_80": bool(replay_b_acc > 0.80),
        },
    }


LESSON = LessonExperiment(
    lesson_id="06",
    title="把旧样本带在身上",
    question="无回放的遗忘是否一定大于有回放？",
    run=run,
)
