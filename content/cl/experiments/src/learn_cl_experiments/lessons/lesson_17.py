from __future__ import annotations

import math
import random
from typing import Any

from ..core import LessonExperiment


DIM = 4


def _zeros(rows: int, cols: int) -> list[list[float]]:
    return [[0.0] * cols for _ in range(rows)]


def _identity(dim: int, scale: float) -> list[list[float]]:
    matrix = _zeros(dim, dim)
    for index in range(dim):
        matrix[index][index] = scale
    return matrix


def _copy(matrix: list[list[float]]) -> list[list[float]]:
    return [row[:] for row in matrix]


def _matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(row[col] * vector[col] for col in range(len(vector))) for row in matrix]


def _frobenius(left: list[list[float]], right: list[list[float]]) -> float:
    total = 0.0
    for row in range(len(left)):
        for col in range(len(left[0])):
            delta = left[row][col] - right[row][col]
            total += delta * delta
    return math.sqrt(total)


def _mse(pred: list[float], target: list[float]) -> float:
    return sum((a - b) * (a - b) for a, b in zip(pred, target)) / len(pred)


def _ttt_step(
    weights: list[list[float]],
    token: list[float],
    target: list[float],
    lr: float,
) -> float:
    pred = _matvec(weights, token)
    residual = [pred[index] - target[index] for index in range(DIM)]
    for row in range(DIM):
        for col in range(DIM):
            weights[row][col] -= lr * 2.0 * residual[row] * token[col] / DIM
    return _mse(pred, target)


def _ttt_run(
    tokens: list[list[float]],
    targets: list[list[float]],
    steps: int,
    lr: float,
) -> tuple[list[list[float]], list[float], float]:
    weights = _identity(DIM, 0.15)
    start = _copy(weights)
    losses: list[float] = []
    limit = min(steps, len(tokens))
    for index in range(limit):
        losses.append(_ttt_step(weights, tokens[index], targets[index], lr))
    return weights, losses, _frobenius(weights, start)


def run() -> dict[str, Any]:
    rng = random.Random(0)
    length = 16
    tokens = [[rng.gauss(0.0, 1.0) for _ in range(DIM)] for _ in range(length)]
    targets = tokens
    weights_full, losses_full, delta_full = _ttt_run(tokens, targets, length, 0.08)
    _, _, delta_one = _ttt_run(tokens, targets, 1, 0.08)
    _, _, delta_frozen = _ttt_run(tokens, targets, length, 0.0)

    hidden = [0.0] * DIM
    recurrent = _identity(DIM, 0.4)
    incoming = _identity(DIM, 0.3)
    for token in tokens:
        pre = [
            sum(recurrent[row][col] * hidden[col] for col in range(DIM))
            + sum(incoming[row][col] * token[col] for col in range(DIM))
            for row in range(DIM)
        ]
        hidden = [math.tanh(value) for value in pre]

    checks = {
        "inner_loop_moves_W": delta_full > 1e-4,
        "more_steps_larger_delta": delta_full > delta_one,
        "zero_lr_freezes_W": delta_frozen == 0.0,
        "reconstruction_loss_drops": losses_full[-1] < losses_full[0],
        "rnn_state_is_vector": len(hidden) == DIM and not isinstance(hidden[0], list),
        "ttt_state_is_matrix": len(weights_full) == DIM and len(weights_full[0]) == DIM,
    }
    return {
        "summary": (
            f"TTT-Linear 在 16 个 token 上对当前向量做内环回归，"
            f"||ΔW||_F={delta_full:.4f} > 0，且大于只走 1 步的 {delta_one:.4f}；"
            "学习率为 0 时范数为 0。同规模 RNN 的隐状态是向量，不能对当前序列做多步梯度。"
        ),
        "metrics": {
            "delta_w_full": delta_full,
            "delta_w_one_step": delta_one,
            "delta_w_zero_lr": delta_frozen,
            "loss_first": losses_full[0],
            "loss_last": losses_full[-1],
            "sequence_length": length,
            "rnn_hidden_dim": len(hidden),
            "ttt_rows": len(weights_full),
            "ttt_cols": len(weights_full[0]),
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="17",
    title="读这段话的时候权重正在动",
    question="TTT 内环之后记忆矩阵 W 的更新范数是不是大于 0？",
    run=run,
)
