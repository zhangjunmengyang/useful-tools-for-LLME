"""Self-edits written into weights: filter keeps the rule, dump SFT poisons it."""

from __future__ import annotations

import random
from typing import Any

from ..extra_core import ExtraExperiment


def _true(a: float, b: float) -> float:
    return 2.0 * a + 3.0 * b


def _fit(
    pairs: list[tuple[tuple[float, float], float]],
    steps: int,
    lr: float,
    rng: random.Random,
) -> tuple[float, float]:
    w0, w1 = 0.0, 0.0
    if not pairs:
        return w0, w1
    for _ in range(steps):
        (a, b), y = pairs[rng.randrange(len(pairs))]
        pred = w0 * a + w1 * b
        err = pred - y
        w0 -= lr * err * a
        w1 -= lr * err * b
    return w0, w1


def _mae(weights: tuple[float, float], probes: list[tuple[float, float]]) -> float:
    if not probes:
        return 1e9
    total = 0.0
    for a, b in probes:
        total += abs(weights[0] * a + weights[1] * b - _true(a, b))
    return total / len(probes)


def run() -> dict[str, Any]:
    rng = random.Random(2)
    clean = [((float(i), float(i + 1)), _true(float(i), float(i + 1))) for i in range(8)]
    probes = [(float(i) * 0.5, float(i) * 0.3) for i in range(12)]

    def generate(noise: float) -> list[tuple[tuple[float, float], float]]:
        items = []
        for _ in range(40):
            a, b = rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0)
            y = _true(a, b)
            if rng.random() < noise:
                y += rng.choice((-8.0, 8.0))
            items.append(((a, b), y))
        return items

    noisy = generate(0.45)
    filtered = [item for item in noisy if abs(item[1] - _true(*item[0])) < 1e-6]
    dump = noisy

    w_filter = _fit(filtered, steps=400, lr=0.05, rng=random.Random(3))
    w_dump = _fit(dump, steps=400, lr=0.05, rng=random.Random(3))
    w_none = (0.0, 0.0)

    mae_filter = _mae(w_filter, probes)
    mae_dump = _mae(w_dump, probes)
    mae_none = _mae(w_none, probes)

    checks = {
        "filter_keeps_majority_clean": len(filtered) >= 12,
        "filtered_mae_small": mae_filter < 0.4,
        "dump_mae_worse": mae_dump > mae_filter + 0.8,
        "untrained_worse_than_filtered": mae_none > mae_filter + 1.0,
        "filtered_weights_near_true": (
            abs(w_filter[0] - 2.0) < 0.35 and abs(w_filter[1] - 3.0) < 0.35
        ),
    }
    return {
        "summary": (
            f"自编辑 40 条，筛选后留 {len(filtered)} 条干净样本。"
            f"写入权重后规则误差 {mae_filter:.3f}；不筛选直接灌进去 {mae_dump:.3f}。"
            "失败阈值：筛选后仍学不像 2a+3b，或不筛选反而更好。"
        ),
        "metrics": {
            "generated": len(noisy),
            "filtered": len(filtered),
            "mae_filter": mae_filter,
            "mae_dump": mae_dump,
            "mae_none": mae_none,
            "w_filter": [w_filter[0], w_filter[1]],
            "w_dump": [w_dump[0], w_dump[1]],
        },
        "checks": checks,
    }


EXPERIMENT = ExtraExperiment(
    extra_id="selfedit",
    title="自编辑写入权重",
    question="模型给自己出训练题时，不筛选会不会把规则写坏？",
    lesson_hint="20,23",
    run=run,
)
