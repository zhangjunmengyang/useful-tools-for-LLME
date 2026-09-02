from __future__ import annotations

import re
from typing import Any

from ..core import LessonExperiment


REASON_OPEN = "<REASONING>"
REASON_CLOSE = "</REASONING>"
ANSWER_OPEN = "<CONCLUSION>"
ANSWER_CLOSE = "</CONCLUSION>"
CELL_PATTERN = re.compile(r"\((\d+),(\d+)\)")


def extract_span(text: str, open_tag: str, close_tag: str) -> str:
    start = text.find(open_tag)
    end = text.find(close_tag)
    if start < 0 or end < 0 or end <= start:
        return ""
    return text[start + len(open_tag) : end].strip()


def tokenize_span(span: str) -> list[str]:
    return [piece for piece in span.split() if piece]


def reason_tokens(text: str) -> list[str]:
    return tokenize_span(extract_span(text, REASON_OPEN, REASON_CLOSE))


def answer_tokens(text: str) -> list[str]:
    return tokenize_span(extract_span(text, ANSWER_OPEN, ANSWER_CLOSE))


def token_index_sets(text: str) -> tuple[set[int], set[int]]:
    """Positional ids of reason vs answer tokens in the flattened sequence."""
    reason = reason_tokens(text)
    answer = answer_tokens(text)
    reason_ids = set(range(len(reason)))
    answer_ids = set(range(len(reason), len(reason) + len(answer)))
    return reason_ids, answer_ids


def cited_cells(text: str) -> set[tuple[int, int]]:
    span = extract_span(text, REASON_OPEN, REASON_CLOSE)
    return {(int(row), int(col)) for row, col in CELL_PATTERN.findall(span)}


def parse_answer(text: str) -> str:
    tokens = answer_tokens(text)
    return tokens[-1] if tokens else ""


def answer_reward(predicted: str, gold: str) -> float:
    return 1.0 if predicted == gold else 0.0


def process_reward(
    cited: set[tuple[int, int]],
    gold_cells: set[tuple[int, int]],
) -> float:
    if not cited:
        return 0.0
    return 1.0 if cited & gold_cells else 0.0


def combined_reward(
    r_answer: float,
    r_process: float,
    require_cite: bool,
) -> float:
    if require_cite:
        return r_answer * r_process
    return r_answer


def mean_advantages(rewards: list[float]) -> list[float]:
    baseline = sum(rewards) / len(rewards)
    return [reward - baseline for reward in rewards]


def reward_variance(rewards: list[float]) -> float:
    baseline = sum(rewards) / len(rewards)
    return sum((reward - baseline) ** 2 for reward in rewards) / len(rewards)


GOLD_ANSWER = "2"
GOLD_CELLS = {(0, 1), (2, 3)}

CITED_CORRECT = (
    f"{REASON_OPEN} 格子 (0,1) 与 (2,3) 是红杯 {REASON_CLOSE} "
    f"{ANSWER_OPEN} 共 {GOLD_ANSWER} {ANSWER_CLOSE}"
)
EMPTY_CITE_CORRECT = (
    f"{REASON_OPEN} 常见场景杯子成对出现 {REASON_CLOSE} "
    f"{ANSWER_OPEN} 共 {GOLD_ANSWER} {ANSWER_CLOSE}"
)
WRONG_CELLS_CORRECT = (
    f"{REASON_OPEN} 格子 (1,2) 与 (3,1) 看起来像杯子 {REASON_CLOSE} "
    f"{ANSWER_OPEN} 共 {GOLD_ANSWER} {ANSWER_CLOSE}"
)
CITED_WRONG_ANSWER = (
    f"{REASON_OPEN} 格子 (0,1) 与 (2,3) 是红杯 {REASON_CLOSE} "
    f"{ANSWER_OPEN} 共 3 {ANSWER_CLOSE}"
)
CITE_IN_ANSWER_ONLY = (
    f"{REASON_OPEN} 我数过了 {REASON_CLOSE} "
    f"{ANSWER_OPEN} 格子 (0,1) (2,3) 共 {GOLD_ANSWER} {ANSWER_CLOSE}"
)


def _score(text: str) -> dict[str, Any]:
    cited = cited_cells(text)
    predicted = parse_answer(text)
    r_answer = answer_reward(predicted, GOLD_ANSWER)
    r_process = process_reward(cited, GOLD_CELLS)
    reason_ids, answer_ids = token_index_sets(text)
    return {
        "cited": sorted(cited),
        "predicted": predicted,
        "r_answer": r_answer,
        "r_process": r_process,
        "r_require": combined_reward(r_answer, r_process, True),
        "r_answer_only": combined_reward(r_answer, r_process, False),
        "reason_ids": sorted(reason_ids),
        "answer_ids": sorted(answer_ids),
        "disjoint": reason_ids.isdisjoint(answer_ids),
        "reason_tokens": reason_tokens(text),
        "answer_tokens": answer_tokens(text),
    }


def run() -> dict[str, Any]:
    cited_ok = _score(CITED_CORRECT)
    empty_ok = _score(EMPTY_CITE_CORRECT)
    wrong_cells = _score(WRONG_CELLS_CORRECT)
    cited_wrong = _score(CITED_WRONG_ANSWER)
    cite_in_answer = _score(CITE_IN_ANSWER_ONLY)

    answer_only_group = [
        empty_ok["r_answer_only"],
        empty_ok["r_answer_only"],
        empty_ok["r_answer_only"],
        empty_ok["r_answer_only"],
    ]
    process_group = [
        cited_ok["r_require"],
        empty_ok["r_require"],
        wrong_cells["r_require"],
        cited_wrong["r_require"],
    ]
    answer_only_adv = mean_advantages(answer_only_group)
    process_adv = mean_advantages(process_group)

    return {
        "summary": (
            "同一道计数题拆成推理 token 与答案 token。"
            "引用格为空时过程奖励为 0；关掉必须引用后答案仍可对。"
        ),
        "metrics": {
            "gold_answer": GOLD_ANSWER,
            "gold_cells": [list(cell) for cell in sorted(GOLD_CELLS)],
            "cited_correct_cells": [list(cell) for cell in cited_ok["cited"]],
            "empty_cite_cells": [list(cell) for cell in empty_ok["cited"]],
            "wrong_cite_cells": [list(cell) for cell in wrong_cells["cited"]],
            "cite_in_answer_cells": [
                list(cell) for cell in cite_in_answer["cited"]
            ],
            "cited_correct_r_answer": cited_ok["r_answer"],
            "cited_correct_r_process": cited_ok["r_process"],
            "empty_cite_r_answer": empty_ok["r_answer"],
            "empty_cite_r_process": empty_ok["r_process"],
            "empty_cite_r_require": empty_ok["r_require"],
            "empty_cite_r_answer_only": empty_ok["r_answer_only"],
            "wrong_cells_r_process": wrong_cells["r_process"],
            "cited_wrong_r_answer": cited_wrong["r_answer"],
            "cited_wrong_r_process": cited_wrong["r_process"],
            "answer_only_variance": round(reward_variance(answer_only_group), 8),
            "process_group_rewards": process_group,
            "process_group_advantages": [
                round(value, 6) for value in process_adv
            ],
            "reason_token_count": len(cited_ok["reason_tokens"]),
            "answer_token_count": len(cited_ok["answer_tokens"]),
        },
        "checks": {
            "推理与答案位置集合不相交": (
                cited_ok["disjoint"]
                and empty_ok["disjoint"]
                and cite_in_answer["disjoint"]
            ),
            "无视觉引用则过程奖励为0": (
                empty_ok["r_process"] == 0.0
                and empty_ok["cited"] == []
                and cite_in_answer["r_process"] == 0.0
                and cite_in_answer["cited"] == []
            ),
            "关掉必须引用后答案对引用空": (
                empty_ok["r_answer"] == 1.0
                and empty_ok["r_answer_only"] == 1.0
                and empty_ok["r_require"] == 0.0
                and empty_ok["predicted"] == GOLD_ANSWER
            ),
            "引用真值格才有过程分": (
                cited_ok["r_process"] == 1.0
                and cited_ok["r_require"] == 1.0
                and wrong_cells["r_answer"] == 1.0
                and wrong_cells["r_process"] == 0.0
            ),
            "只奖答案的全对组没有相对优势": (
                reward_variance(answer_only_group) == 0.0
                and all(abs(value) < 1e-12 for value in answer_only_adv)
            ),
            "过程奖励能把空引用排到零": (
                process_group == [1.0, 0.0, 0.0, 0.0]
                and process_adv[0] > 0.0
                and all(value < 0.0 for value in process_adv[1:])
            ),
            "答案错即使引用对也没有合取分": (
                cited_wrong["r_answer"] == 0.0
                and cited_wrong["r_process"] == 1.0
                and cited_wrong["r_require"] == 0.0
            ),
            "答案span里的格子不算推理引用": (
                "(0,1)" in cite_in_answer["answer_tokens"]
                or any("0,1" in token for token in cite_in_answer["answer_tokens"])
            )
            and cite_in_answer["r_process"] == 0.0,
        },
    }


LESSON = LessonExperiment(
    lesson_id="51",
    title="给视觉语言模型加上分步推理",
    question="推理 token 与答案 token 分开后，无视觉引用时过程奖励为什么必须是 0？",
    run=run,
)
