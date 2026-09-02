from __future__ import annotations

import math
import random
from typing import Any

from ..core import LessonExperiment


def _l2(theta: tuple[float, float]) -> float:
    return math.hypot(theta[0], theta[1])


def _softmax(logits: tuple[float, float]) -> tuple[float, float]:
    peak = max(logits)
    exps = (math.exp(logits[0] - peak), math.exp(logits[1] - peak))
    total = exps[0] + exps[1]
    return exps[0] / total, exps[1] / total


def _kl(p: tuple[float, float], q: tuple[float, float]) -> float:
    return p[0] * math.log(p[0] / q[0]) + p[1] * math.log(p[1] / q[1])


def _old_acc(theta: tuple[float, float]) -> float:
    return math.exp(-_l2(theta) ** 2)


def _new_acc(theta: tuple[float, float], target: tuple[float, float]) -> float:
    return math.exp(-math.hypot(theta[0] - target[0], theta[1] - target[1]) ** 2)


def run() -> dict[str, Any]:
    rng = random.Random(0)
    origin = (0.0, 0.0)
    new_opt = (2.6, 1.4)
    offline = [
        (new_opt[0] + rng.gauss(0.0, 0.15), new_opt[1] + rng.gauss(0.0, 0.15))
        for _ in range(40)
    ]
    sft = (
        sum(point[0] for point in offline) / len(offline),
        sum(point[1] for point in offline) / len(offline),
    )
    rl = origin
    rl_lr = 0.05
    for _ in range(5):
        rl = (
            rl[0] + rl_lr * (new_opt[0] - rl[0]),
            rl[1] + rl_lr * (new_opt[1] - rl[1]),
        )

    origin_policy = _softmax(origin)
    sft_kl = _kl(_softmax(sft), origin_policy)
    rl_kl = _kl(_softmax(rl), origin_policy)
    forget_sft = _old_acc(origin) - _old_acc(sft)
    forget_rl = _old_acc(origin) - _old_acc(rl)

    mix_grid = [index / 10.0 for index in range(11)]
    distances = []
    forgets = []
    for mix in mix_grid:
        theta = (mix * sft[0], mix * sft[1])
        distances.append(_l2(theta))
        forgets.append(_old_acc(origin) - _old_acc(theta))
    mean_d = sum(distances) / len(distances)
    mean_f = sum(forgets) / len(forgets)
    cov = sum(
        (dist - mean_d) * (forg - mean_f)
        for dist, forg in zip(distances, forgets)
    ) / len(distances)
    var_d = sum((dist - mean_d) ** 2 for dist in distances) / len(distances)
    var_f = sum((forg - mean_f) ** 2 for forg in forgets) / len(forgets)
    corr = cov / math.sqrt(var_d * var_f)

    checks = {
        "sft_farther_from_origin": _l2(sft) > _l2(rl),
        "sft_kl_exceeds_on_policy": sft_kl > rl_kl,
        "sft_forgets_more": forget_sft > forget_rl + 0.2,
        "distance_tracks_forgetting": corr > 0.9,
        "both_improve_new_task": (
            _new_acc(sft, new_opt) > _new_acc(origin, new_opt)
            and _new_acc(rl, new_opt) > _new_acc(origin, new_opt)
        ),
    }
    return {
        "summary": (
            f"离线 SFT 把二维策略拉到离原点 L2={_l2(sft):.3f}、KL={sft_kl:.3f}；"
            f"on-policy 五小步后 L2={_l2(rl):.3f}、KL={rl_kl:.3f}。"
            f"遗忘与到原点距离的相关为 {corr:.3f}。失败阈值：SFT 的 L2/KL 不超过 RL。"
        ),
        "metrics": {
            "sft_l2": _l2(sft),
            "rl_l2": _l2(rl),
            "sft_kl": sft_kl,
            "rl_kl": rl_kl,
            "forget_sft": forget_sft,
            "forget_rl": forget_rl,
            "distance_forget_corr": corr,
            "sft_new_acc": _new_acc(sft, new_opt),
            "rl_new_acc": _new_acc(rl, new_opt),
            "origin_new_acc": _new_acc(origin, new_opt),
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="20",
    title="自己出题，以及为什么 RL 比较不易忘",
    question="离线 SFT 到原点的距离是不是大于 on-policy 小步？",
    run=run,
)
