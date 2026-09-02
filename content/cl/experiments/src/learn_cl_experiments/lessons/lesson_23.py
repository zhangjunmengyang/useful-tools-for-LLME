from __future__ import annotations

import random
from typing import Any

from ..core import LessonExperiment


ROUNDS = 3
GEN_SIZE = 60
VAL_SIZE = 40
STEPS = 80
LR = 0.2
NOISE = 0.22


def _true_label(point: tuple[float, float]) -> int:
    return 1 if point[0] + point[1] > 0.0 else 0


def _predict(weights: tuple[float, float], point: tuple[float, float]) -> int:
    return 1 if weights[0] * point[0] + weights[1] * point[1] > 0.0 else 0


def _error_fraction(samples: list[tuple[tuple[float, float], int]]) -> float:
    if not samples:
        return 1.0
    return sum(_true_label(point) != label for point, label in samples) / len(samples)


def _accuracy(weights: tuple[float, float], points: list[tuple[float, float]]) -> float:
    return sum(_predict(weights, point) == _true_label(point) for point in points) / len(points)


def _sample_point(rng: random.Random) -> tuple[float, float]:
    return rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0)


def _knn_label(
    val: list[tuple[tuple[float, float], int]],
    point: tuple[float, float],
    k: int = 5,
) -> int:
    ranked = sorted(
        val,
        key=lambda item: (item[0][0] - point[0]) ** 2 + (item[0][1] - point[1]) ** 2,
    )
    votes = sum(label for _, label in ranked[:k])
    return 1 if votes >= (k / 2.0) else 0


def _train(
    weights: tuple[float, float],
    samples: list[tuple[tuple[float, float], int]],
    rng: random.Random,
) -> tuple[float, float]:
    w0, w1 = weights
    if not samples:
        return weights
    for _ in range(STEPS):
        point, label = samples[rng.randrange(len(samples))]
        target = 1.0 if label == 1 else -1.0
        score = w0 * point[0] + w1 * point[1]
        if target * score < 1.0:
            w0 += LR * target * point[0]
            w1 += LR * target * point[1]
    return w0, w1


def _generate(
    weights: tuple[float, float],
    rng: random.Random,
    extra_noise: float,
) -> list[tuple[tuple[float, float], int]]:
    samples = []
    for _ in range(GEN_SIZE):
        point = _sample_point(rng)
        label = _predict(weights, point)
        if rng.random() < NOISE + extra_noise:
            label = 1 - label
        samples.append((point, label))
    return samples


def _loop(use_filter: bool, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    weights = (1.0, 1.0)
    val = []
    for _ in range(VAL_SIZE):
        point = _sample_point(rng)
        val.append((point, _true_label(point)))
    holdout = [_sample_point(rng) for _ in range(80)]
    train_errors: list[float] = []
    gen_errors: list[float] = []
    val_accs: list[float] = []
    extra = 0.0
    for _ in range(ROUNDS):
        generated = _generate(weights, rng, extra)
        gen_errors.append(_error_fraction(generated))
        if use_filter:
            accepted = [
                sample
                for sample in generated
                if sample[1] == _knn_label(val, sample[0])
            ]
            extra = 0.0
        else:
            accepted = generated
            extra += 0.12
        train_errors.append(_error_fraction(accepted))
        weights = _train(weights, accepted, rng)
        val_accs.append(_accuracy(weights, holdout))
    return {
        "train_errors": train_errors,
        "gen_errors": gen_errors,
        "val_accs": val_accs,
        "final_train_error": train_errors[-1],
        "first_train_error": train_errors[0],
    }


def run() -> dict[str, Any]:
    filtered = _loop(use_filter=True, seed=0)
    unfiltered = _loop(use_filter=False, seed=0)

    checks = {
        "filter_lowers_train_error": (
            filtered["final_train_error"] < unfiltered["final_train_error"]
        ),
        "unfiltered_error_rises": (
            unfiltered["final_train_error"] > unfiltered["first_train_error"] + 0.05
        ),
        "filtered_error_stays_low": filtered["final_train_error"] < 0.2,
        "generated_contains_mistakes": unfiltered["gen_errors"][0] > 0.05,
        "filter_val_not_worse": filtered["val_accs"][-1] >= unfiltered["val_accs"][-1] - 1e-12,
    }
    return {
        "summary": (
            f"三轮自生成数据：筛选后第 3 轮训练错误率 {filtered['final_train_error']:.3f}，"
            f"关掉筛选后从 {unfiltered['first_train_error']:.3f} 升到 "
            f"{unfiltered['final_train_error']:.3f}。"
            "失败阈值：无筛选时错误率没有上升超过 0.05，或筛选后仍不低于无筛选。"
        ),
        "metrics": {
            "filtered_train_errors": filtered["train_errors"],
            "unfiltered_train_errors": unfiltered["train_errors"],
            "filtered_gen_errors": filtered["gen_errors"],
            "unfiltered_gen_errors": unfiltered["gen_errors"],
            "filtered_val_accs": filtered["val_accs"],
            "unfiltered_val_accs": unfiltered["val_accs"],
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="23",
    title="自己做研究、自己出下一版",
    question="关掉验证筛选之后，训练数据里的错误比例会不会上升？",
    run=run,
)
