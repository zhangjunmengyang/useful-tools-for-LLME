from __future__ import annotations

from typing import Any

import numpy as np

from ..core import LessonExperiment

SEED = 3


def _num(value: float) -> float:
    return float(round(float(value), 6))


def _average_accuracy(matrix: np.ndarray) -> float:
    return float(matrix[-1].mean())


def _learning_accuracy(matrix: np.ndarray) -> float:
    return float(np.diag(matrix).mean())


def _bwt(matrix: np.ndarray) -> float:
    n_tasks = matrix.shape[0]
    return float(
        np.mean([matrix[n_tasks - 1, task] - matrix[task, task] for task in range(n_tasks - 1)]),
    )


def _fwt(matrix: np.ndarray) -> float:
    n_tasks = matrix.shape[0]
    return float(np.mean([matrix[task, task + 1] for task in range(n_tasks - 1)]))


def _average_forgetting(matrix: np.ndarray) -> float:
    n_tasks = matrix.shape[0]
    return float(
        np.mean([matrix[task, task] - matrix[n_tasks - 1, task] for task in range(n_tasks - 1)]),
    )


def run() -> dict[str, Any]:
    specialist_two = np.array(
        [
            [0.98, 0.50],
            [0.50, 0.98],
        ],
        dtype=np.float64,
    )
    specialist_five = np.array(
        [
            [1.00, 0.50, 0.50, 0.50, 0.50],
            [0.50, 1.00, 0.50, 0.50, 0.50],
            [0.50, 0.50, 1.00, 0.50, 0.50],
            [0.50, 0.50, 0.50, 1.00, 0.50],
            [0.50, 0.50, 0.50, 0.50, 1.00],
        ],
        dtype=np.float64,
    )
    # Same final-row average as the 2-task specialist, but almost no forgetting.
    honest_two = np.array(
        [
            [0.76, 0.50],
            [0.73, 0.75],
        ],
        dtype=np.float64,
    )

    spec_acc = _average_accuracy(specialist_two)
    spec_bwt = _bwt(specialist_two)
    spec_la = _learning_accuracy(specialist_two)
    spec_forget = _average_forgetting(specialist_two)
    honest_acc = _average_accuracy(honest_two)
    honest_bwt = _bwt(honest_two)
    five_acc = _average_accuracy(specialist_five)
    five_bwt = _bwt(specialist_five)
    five_fwt = _fwt(specialist_five)

    return {
        "summary": (
            "构造「只会最后任务」的 2×2 准确率矩阵：最终平均准确率 "
            f"{spec_acc:.2f} 看起来能过关，但 BWT={spec_bwt:.2f}、"
            f"Learning Accuracy={spec_la:.2f}。对照矩阵把平均准确率也做成 "
            f"{honest_acc:.2f}，BWT 却只有 {honest_bwt:.2f}。"
            "阈值：最后任务专家的 ACC>0.70、BWT<-0.40、LA>0.95；"
            "两张 2×2 的 ACC 差 <0.02，但 BWT 差 >0.35。"
        ),
        "metrics": {
            "seed": SEED,
            "specialist2_average_accuracy": _num(spec_acc),
            "specialist2_bwt": _num(spec_bwt),
            "specialist2_learning_accuracy": _num(spec_la),
            "specialist2_average_forgetting": _num(spec_forget),
            "specialist5_average_accuracy": _num(five_acc),
            "specialist5_bwt": _num(five_bwt),
            "specialist5_fwt": _num(five_fwt),
            "honest2_average_accuracy": _num(honest_acc),
            "honest2_bwt": _num(honest_bwt),
            "specialist2_matrix": [
                [_num(value) for value in row] for row in specialist_two
            ],
        },
        "checks": {
            "last_task_specialist_final_row": bool(
                specialist_two[1, 1] > 0.95 and specialist_two[1, 0] < 0.55,
            ),
            "average_accuracy_looks_high": bool(spec_acc > 0.70),
            "bwt_strongly_negative": bool(spec_bwt < -0.40),
            "learning_accuracy_hides_forgetting": bool(
                spec_la > 0.95 and spec_forget > 0.40,
            ),
            "same_acc_different_bwt": bool(
                abs(spec_acc - honest_acc) < 0.02 and (honest_bwt - spec_bwt) > 0.35,
            ),
            "fwt_stays_at_chance_before_learning": bool(abs(five_fwt - 0.5) < 1e-12),
        },
    }


LESSON = LessonExperiment(
    lesson_id="03",
    title="怎么量才算学会了",
    question="只报最终平均准确率，为什么会把「根本没学、只会最后一件事」的方法夸成好方法？",
    run=run,
)
