from __future__ import annotations

import random
from typing import Any

from ..core import LessonExperiment


def _mae(pairs: list[tuple[float, float]]) -> float:
    return sum(abs(left - right) for left, right in pairs) / len(pairs)


def _run(use_fast: bool, use_slow: bool, seed: int = 0) -> dict[str, Any]:
    rng = random.Random(seed)
    documents = 6
    sequences = 4
    length = 6
    token_pairs: list[tuple[float, float]] = []
    style_pairs: list[tuple[float, float]] = []
    fast_updates = 0
    slow_updates = 0
    for _ in range(documents):
        style = 2.0 if rng.random() < 0.5 else -2.0
        slow = 0.0
        for _ in range(sequences):
            residual_sum = 0.0
            for _ in range(length):
                token = rng.gauss(0.0, 1.2)
                target = token + style
                fast = token if use_fast else 0.0
                pred = fast + (slow if use_slow else 0.0)
                token_pairs.append((pred, target))
                residual_sum += target - token
                fast_updates += 1
            estimate = residual_sum / length
            if use_slow:
                slow = (1.0 - 0.5) * slow + 0.5 * estimate
                slow_updates += 1
            style_pairs.append((slow if use_slow else 0.0, style))
    return {
        "token_mae": _mae(token_pairs),
        "style_mae": _mae(style_pairs),
        "fast_updates": fast_updates,
        "slow_updates": slow_updates,
    }


def run() -> dict[str, Any]:
    both = _run(use_fast=True, use_slow=True)
    no_slow = _run(use_fast=True, use_slow=False)
    no_fast = _run(use_fast=False, use_slow=True)

    checks = {
        "two_timescales_fit_token_and_style": (
            both["token_mae"] < 1.0 and both["style_mae"] < 1.2
        ),
        "no_slow_loses_style": no_slow["style_mae"] > both["style_mae"] + 0.8,
        "no_fast_loses_token": no_fast["token_mae"] > both["token_mae"] + 0.4,
        "slow_updates_once_per_sequence": both["slow_updates"] == 6 * 4,
        "fast_updates_once_per_token": both["fast_updates"] == 6 * 4 * 6,
        "slow_is_rarer_than_fast": both["slow_updates"] < both["fast_updates"],
    }
    return {
        "summary": (
            "快权重每 token 记住当前 x，慢权重每段序列估计文档风格 μ。"
            f"两时间尺度 token MAE={both['token_mae']:.3f}、style MAE={both['style_mae']:.3f}；"
            f"关掉慢权重后 style MAE 升到 {no_slow['style_mae']:.3f}。"
            "失败阈值：无慢权重时风格误差没有比完整模型高出 0.8。"
        ),
        "metrics": {
            "both_token_mae": both["token_mae"],
            "both_style_mae": both["style_mae"],
            "no_slow_style_mae": no_slow["style_mae"],
            "no_fast_token_mae": no_fast["token_mae"],
            "fast_updates": both["fast_updates"],
            "slow_updates": both["slow_updates"],
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="19",
    title="优化器也是一层记忆",
    question="关掉慢权重之后，跨序列的风格信息会不会丢？",
    run=run,
)
