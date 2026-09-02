"""A bad self-edit is permanent unless you kept a checkpoint. The checkpoint is another plugin."""

from __future__ import annotations

import random
from typing import Any

from ..extra_core import ExtraExperiment
from ..lin import linear_fit, linear_mae


def _true(a: float, b: float) -> float:
    return 2.0 * a + 3.0 * b


def run() -> dict[str, Any]:
    rng = random.Random(191)
    clean_pairs = []
    for _ in range(28):
        a, b = rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)
        clean_pairs.append(((a, b), _true(a, b)))
    probes = [(float(i) * 0.31, float(i) * -0.19) for i in range(12)]

    good = linear_fit(clean_pairs, steps=280, lr=0.08, rng=random.Random(192))
    mae_good = linear_mae(good, probes, _true)
    checkpoint = (good[0], good[1])

    noisy = []
    for _ in range(36):
        a, b = rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)
        y = _true(a, b)
        if rng.random() < 0.45:
            y += rng.choice((-7.0, 7.0))
        noisy.append(((a, b), y))
    wrecked = linear_fit(
        noisy,
        steps=280,
        lr=0.08,
        rng=random.Random(193),
        start=good,
    )
    mae_wrecked = linear_mae(wrecked, probes, _true)

    restored = checkpoint
    mae_restored = linear_mae(restored, probes, _true)
    # If you unplugged the checkpoint, you only have wrecked.
    mae_unplug_ckpt = mae_wrecked

    checks = {
        "clean_rule_fits": mae_good < 0.25,
        "bad_night_breaks_rule": mae_wrecked > mae_good + 0.8,
        "checkpoint_restores": mae_restored < 0.25,
        "unplug_checkpoint_cannot_restore": mae_unplug_ckpt > mae_good + 0.8,
        "restored_matches_checkpoint": restored == checkpoint,
        "wrecked_is_not_the_checkpoint": wrecked != checkpoint,
    }
    return {
        "summary": (
            f"筛选后的规则误差 {mae_good:.3f}。一晚把脏自编辑灌进去，误差 {mae_wrecked:.3f}。"
            f"从检查点恢复后 {mae_restored:.3f}；如果检查点也卸掉了，只剩 {mae_unplug_ckpt:.3f}。"
            "失败阈值：坏写入并不更差，或不留检查点也能恢复。"
        ),
        "metrics": {
            "mae_good": mae_good,
            "mae_wrecked": mae_wrecked,
            "mae_restored": mae_restored,
            "mae_unplug_ckpt": mae_unplug_ckpt,
            "good": [good[0], good[1]],
            "wrecked": [wrecked[0], wrecked[1]],
        },
        "checks": checks,
    }


EXPERIMENT = ExtraExperiment(
    extra_id="rollback",
    title="坏的一夜：检查点也是外挂",
    question="自编辑写坏了规则。没有权重检查点的话，还能回到昨晚之前吗？",
    lesson_hint="20,23,24",
    run=run,
)
