from __future__ import annotations

from typing import Any

import numpy as np

from ..core import LessonExperiment

SEED = 4


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


def _hit_rate(predicted: list[int | None], seats: np.ndarray) -> float:
    hits = 0
    for person, seat in enumerate(predicted):
        if seat is not None and seat == int(seats[person]):
            hits += 1
    return hits / len(seats)


def run() -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    n_people = 20
    seats = np.array([(person * 7 + 3) % n_people for person in range(n_people)])
    directory = {person: int(seats[person]) for person in range(n_people)}

    weights = rng.normal(0.0, 0.01, (n_people, n_people))
    bias = np.zeros(n_people)
    identity = np.eye(n_people)
    for _ in range(120):
        probs = _softmax(identity @ weights + bias)
        grad = (probs - _one_hot(seats, n_people)) / n_people
        weights = weights - 1.5 * (identity.T @ grad)
        bias = bias - 1.5 * grad.sum(axis=0)

    prompt_with = [directory.get(person) for person in range(n_people)]
    prompt_without = [None for _ in range(n_people)]
    rag_with = [directory[person] for person in range(n_people)]
    rag_without = [None for _ in range(n_people)]
    weight_answers = [int(np.argmax(weights[person] + bias)) for person in range(n_people)]

    acc_prompt_with = _hit_rate(prompt_with, seats)
    acc_prompt_without = _hit_rate(prompt_without, seats)
    acc_rag_with = _hit_rate(rag_with, seats)
    acc_rag_without = _hit_rate(rag_without, seats)
    acc_weights = _hit_rate(weight_answers, seats)

    return {
        "summary": (
            "20 条「谁坐哪」事实。名录在上下文里时 prompt/RAG 都满分；"
            "撤掉名录后两者准确率为 0，改过权重的线性联想记忆仍是 "
            f"{acc_weights:.3f}。阈值：有上下文/有知识库 = 1.0，"
            "撤上下文后 prompt 与 RAG <0.15，权重 >0.95。"
        ),
        "metrics": {
            "seed": SEED,
            "n_facts": n_people,
            "prompt_with_context": _num(acc_prompt_with),
            "prompt_without_context": _num(acc_prompt_without),
            "rag_with_kb": _num(acc_rag_with),
            "rag_without_kb": _num(acc_rag_without),
            "weights_without_context": _num(acc_weights),
        },
        "checks": {
            "prompt_perfect_with_context": bool(acc_prompt_with == 1.0),
            "prompt_fails_without_context": bool(acc_prompt_without < 0.15),
            "rag_fails_without_kb": bool(acc_rag_without < 0.15),
            "rag_works_with_kb": bool(acc_rag_with == 1.0),
            "weights_survive_without_context": bool(acc_weights > 0.95),
        },
    }


LESSON = LessonExperiment(
    lesson_id="04",
    title="把上下文塞满不等于学会了",
    question="把名录撤掉之后，只靠 prompt 的方法还能不能答对事实？",
    run=run,
)
