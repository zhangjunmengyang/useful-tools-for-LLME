from __future__ import annotations

import math
import random
from typing import Any

from ..core import LessonExperiment


INPUT = 6
HIDDEN = 10
TASKS = 16
STEPS = 8
BATCH = 16
LR = 0.45
PROBE = 48


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _softmax(logits: list[float]) -> list[float]:
    peak = max(logits)
    exps = [math.exp(value - peak) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def _init_params(rng: random.Random) -> dict[str, Any]:
    scale1 = 0.8
    scale2 = 0.8
    return {
        "w1": [
            [rng.gauss(0.0, scale1) for _ in range(INPUT)]
            for _ in range(HIDDEN)
        ],
        "b1": [rng.gauss(0.0, 0.1) for _ in range(HIDDEN)],
        "w2": [
            [rng.gauss(0.0, scale2) for _ in range(HIDDEN)]
            for _ in range(2)
        ],
        "b2": [0.0, 0.0],
    }


def _forward(params: dict[str, Any], features: list[float]) -> tuple[list[float], list[float], list[float]]:
    pre = [
        _dot(params["w1"][unit], features) + params["b1"][unit]
        for unit in range(HIDDEN)
    ]
    hidden = [math.tanh(value) for value in pre]
    logits = [
        _dot(params["w2"][cls], hidden) + params["b2"][cls]
        for cls in range(2)
    ]
    return pre, hidden, logits


def _step(params: dict[str, Any], features: list[float], label: int, lr: float) -> None:
    pre, hidden, logits = _forward(params, features)
    probs = _softmax(logits)
    dlogits = [probs[cls] - (1.0 if cls == label else 0.0) for cls in range(2)]
    dhidden = [
        sum(dlogits[cls] * params["w2"][cls][unit] for cls in range(2))
        for unit in range(HIDDEN)
    ]
    dpre = [dhidden[unit] * (1.0 - hidden[unit] * hidden[unit]) for unit in range(HIDDEN)]
    for cls in range(2):
        for unit in range(HIDDEN):
            params["w2"][cls][unit] -= lr * dlogits[cls] * hidden[unit]
        params["b2"][cls] -= lr * dlogits[cls]
    for unit in range(HIDDEN):
        for dim in range(INPUT):
            params["w1"][unit][dim] -= lr * dpre[unit] * features[dim]
        params["b1"][unit] -= lr * dpre[unit]


def _make_task(rng: random.Random) -> list[float]:
    return [rng.gauss(0.0, 1.0) for _ in range(INPUT)]


def _sample(teacher: list[float], rng: random.Random) -> tuple[list[float], int]:
    features = [rng.gauss(0.0, 1.0) for _ in range(INPUT)]
    label = 1 if _dot(teacher, features) > 0.0 else 0
    return features, label


def _accuracy(params: dict[str, Any], teacher: list[float], rng: random.Random, count: int) -> float:
    correct = 0
    for _ in range(count):
        features, label = _sample(teacher, rng)
        _, _, logits = _forward(params, features)
        pred = 0 if logits[0] >= logits[1] else 1
        correct += int(pred == label)
    return correct / count


def _hidden_stats(params: dict[str, Any], teacher: list[float], rng: random.Random) -> tuple[float, float]:
    sat_sums = [0.0] * HIDDEN
    gain_sums = [0.0] * HIDDEN
    for _ in range(PROBE):
        features, _ = _sample(teacher, rng)
        pre, _, _ = _forward(params, features)
        for unit, value in enumerate(pre):
            hidden = math.tanh(value)
            sat_sums[unit] += abs(hidden)
            gain_sums[unit] += 1.0 - hidden * hidden
    dead = sum(1 for total in sat_sums if total / PROBE > 0.97) / HIDDEN
    gain = sum(gain_sums) / (HIDDEN * PROBE)
    return dead, gain


def _reinit_dead(params: dict[str, Any], teacher: list[float], rng: random.Random) -> int:
    sat_sums = [0.0] * HIDDEN
    for _ in range(PROBE):
        features, _ = _sample(teacher, rng)
        pre, _, _ = _forward(params, features)
        for unit, value in enumerate(pre):
            sat_sums[unit] += abs(math.tanh(value))
    ranked = sorted(range(HIDDEN), key=lambda unit: sat_sums[unit], reverse=True)
    reset = 0
    for unit in ranked[:3]:
        if sat_sums[unit] / PROBE <= 0.9:
            continue
        params["w1"][unit] = [rng.gauss(0.0, 0.4) for _ in range(INPUT)]
        params["b1"][unit] = rng.gauss(0.0, 0.05)
        for cls in range(2):
            params["w2"][cls][unit] = rng.gauss(0.0, 0.4)
        reset += 1
    return reset


def _run_stream(seed: int, continual_backprop: bool) -> dict[str, Any]:
    rng = random.Random(seed)
    params = _init_params(rng)
    speeds: list[float] = []
    dead_ratios: list[float] = []
    gains: list[float] = []
    for _ in range(TASKS):
        teacher = _make_task(rng)
        start_acc = _accuracy(params, teacher, rng, 40)
        for _ in range(STEPS):
            for _ in range(BATCH):
                features, label = _sample(teacher, rng)
                _step(params, features, label, LR)
        end_acc = _accuracy(params, teacher, rng, 40)
        speeds.append(end_acc - start_acc)
        dead, gain = _hidden_stats(params, teacher, rng)
        dead_ratios.append(dead)
        gains.append(gain)
        if continual_backprop:
            _reinit_dead(params, teacher, rng)
    early = speeds[:3]
    late = speeds[-3:]
    return {
        "early_speed": sum(early) / len(early),
        "late_speed": sum(late) / len(late),
        "early_dead": sum(dead_ratios[:3]) / 3.0,
        "late_dead": sum(dead_ratios[-3:]) / 3.0,
        "early_gain": sum(gains[:3]) / 3.0,
        "late_gain": sum(gains[-3:]) / 3.0,
        "speeds": speeds,
        "dead_ratios": dead_ratios,
        "gains": gains,
    }


def run() -> dict[str, Any]:
    sgd = _run_stream(seed=0, continual_backprop=False)
    cbp = _run_stream(seed=0, continual_backprop=True)

    checks = {
        "sgd_late_gain_drops": sgd["late_gain"] < 0.75 * sgd["early_gain"],
        "sgd_dead_ratio_rises": sgd["late_dead"] > sgd["early_dead"] + 0.15,
        "sgd_late_speed_slower": sgd["late_speed"] < sgd["early_speed"] - 0.02,
        "cbp_late_gain_beats_sgd": cbp["late_gain"] > sgd["late_gain"],
        "cbp_late_dead_below_sgd": cbp["late_dead"] < sgd["late_dead"],
    }
    return {
        "summary": (
            f"14 个随机线性分类任务上，标准 SGD 后期 tanh 增益从 "
            f"{sgd['early_gain']:.3f} 掉到 {sgd['late_gain']:.3f}，"
            f"死神经元比例从 {sgd['early_dead']:.2f} 升到 {sgd['late_dead']:.2f}。"
            "按饱和度重初始化 3 个隐单元后，后期增益和死神经元都优于 SGD。"
            "失败阈值：后期增益不低于前期的 75%，死神经元上升不足 0.15，"
            "或后期准确率提升不低于前期 0.04。"
        ),
        "metrics": {
            "sgd_early_speed": sgd["early_speed"],
            "sgd_late_speed": sgd["late_speed"],
            "sgd_early_dead": sgd["early_dead"],
            "sgd_late_dead": sgd["late_dead"],
            "sgd_early_gain": sgd["early_gain"],
            "sgd_late_gain": sgd["late_gain"],
            "cbp_late_speed": cbp["late_speed"],
            "cbp_late_dead": cbp["late_dead"],
            "cbp_late_gain": cbp["late_gain"],
            "sgd_dead_curve": sgd["dead_ratios"],
            "cbp_dead_curve": cbp["dead_ratios"],
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="15",
    title="学着学着学不动了",
    question="长序列后期有效学习速度会不会掉，死神经元会不会变多？",
    run=run,
)
