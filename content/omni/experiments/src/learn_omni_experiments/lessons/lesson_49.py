from __future__ import annotations

import math
from typing import Any

from ..core import LessonExperiment

HEIGHT = 4
WIDTH = 8
T_OBS = 5
NEXT_T = 5
CUP_COL = 5
CUP_ROWS = (1, 2)
CUP_VALUE = 0.08
BG_FILL = 0.5
FORGET = 1.0
YES = 1
NO = 0
ANSWER_LOGITS = (0.1, 2.5)

N_SPATIAL = HEIGHT * WIDTH
N_HIST = T_OBS * N_SPATIAL
N_PROMPT = 4
N_ANSWER = 1
N_FUTURE = N_SPATIAL
SEQ_LEN = N_HIST + N_PROMPT + N_ANSWER + N_FUTURE
ANSWER_INDEX = N_HIST + N_PROMPT
FUTURE_START = ANSWER_INDEX + N_ANSWER

LEDGERS = {
    "videomme_accuracy": "understand_c2",
    "caption_ce": "understand_text",
    "vbench_dynamic_degree": "generation_vbench",
    "camera_type_match": "generation_camera",
    "object_permanence_miss": "generation_physics",
    "next_frame_l2": "generation_pixel",
}


def texture(step: int, row: int, col: int) -> float:
    return ((step * 17 + row * 13 + col * 7) % 10) / 9.0


def render(step: int, cup_present: bool) -> list[list[float]]:
    grid = [
        [texture(step, row, col) for col in range(WIDTH)]
        for row in range(HEIGHT)
    ]
    if cup_present:
        for row in CUP_ROWS:
            grid[row][CUP_COL] = CUP_VALUE
    return grid


def cup_occupancy(grid: list[list[float]]) -> float:
    cells = [grid[row][CUP_COL] for row in CUP_ROWS]
    mean_value = sum(cells) / len(cells)
    return max(0.0, min(1.0, (BG_FILL - mean_value) / (BG_FILL - CUP_VALUE)))


def predict_next_frame(
    last: list[list[float]],
    forget: float,
) -> list[list[float]]:
    mix = max(0.0, min(1.0, forget))
    return [
        [(1.0 - mix) * value + mix * BG_FILL for value in row]
        for row in last
    ]


def frame_l2(left: list[list[float]], right: list[list[float]]) -> float:
    total = 0.0
    count = 0
    for left_row, right_row in zip(left, right):
        for left_value, right_value in zip(left_row, right_row):
            delta = left_value - right_value
            total += delta * delta
            count += 1
    return total / count


def understand_ce_mask(seq_len: int = SEQ_LEN) -> list[int]:
    mask = [0] * seq_len
    mask[ANSWER_INDEX] = 1
    return mask


def generate_frame_mask(seq_len: int = SEQ_LEN) -> list[int]:
    mask = [0] * seq_len
    for index in range(FUTURE_START, seq_len):
        mask[index] = 1
    return mask


def illegal_shared_mask(seq_len: int = SEQ_LEN) -> list[int]:
    combined = understand_ce_mask(seq_len)
    generate = generate_frame_mask(seq_len)
    return [int(left or right) for left, right in zip(combined, generate)]


def mask_positions(mask: list[int]) -> set[int]:
    return {index for index, flag in enumerate(mask) if flag == 1}


def softmax(logits: tuple[float, float] | list[float]) -> list[float]:
    peak = max(logits)
    exponentials = [math.exp(value - peak) for value in logits]
    denom = sum(exponentials)
    return [value / denom for value in exponentials]


def cross_entropy(logits: tuple[float, float] | list[float], target: int) -> float:
    return -math.log(softmax(logits)[target])


def understand_answer(observed: list[list[list[float]]]) -> int:
    last_occupancy = cup_occupancy(observed[-1])
    return YES if last_occupancy >= 0.5 else NO


def same_ledger(metric_a: str, metric_b: str) -> bool:
    return LEDGERS[metric_a] == LEDGERS[metric_b]


def may_post_as_generation_score(metric_name: str) -> bool:
    return LEDGERS[metric_name].startswith("generation_")


def run() -> dict[str, Any]:
    observed = [render(step, cup_present=True) for step in range(T_OBS)]
    truth_next = render(NEXT_T, cup_present=True)
    predicted_next = predict_next_frame(observed[-1], FORGET)
    copy_next = predict_next_frame(observed[-1], 0.0)

    hist_occupancy = [cup_occupancy(frame) for frame in observed]
    gen_occupancy = cup_occupancy(predicted_next)
    copy_occupancy = cup_occupancy(copy_next)
    truth_occupancy = cup_occupancy(truth_next)

    answer = understand_answer(observed)
    probs = softmax(ANSWER_LOGITS)
    ce = cross_entropy(ANSWER_LOGITS, YES)
    gen_l2 = frame_l2(predicted_next, truth_next)
    copy_l2 = frame_l2(copy_next, truth_next)

    ce_mask = understand_ce_mask()
    gen_mask = generate_frame_mask()
    shared = illegal_shared_mask()
    ce_pos = mask_positions(ce_mask)
    gen_pos = mask_positions(gen_mask)
    shared_pos = mask_positions(shared)
    intersection = ce_pos & gen_pos

    prompt_in_ce = any(ce_mask[index] == 1 for index in range(N_HIST, ANSWER_INDEX))
    hist_in_ce = any(ce_mask[index] == 1 for index in range(N_HIST))
    hist_in_gen = any(gen_mask[index] == 1 for index in range(N_HIST))
    future_in_ce = any(ce_mask[index] == 1 for index in range(FUTURE_START, SEQ_LEN))
    answer_in_gen = gen_mask[ANSWER_INDEX] == 1

    videomme_as_gen = may_post_as_generation_score("videomme_accuracy")
    permanence_as_gen = may_post_as_generation_score("object_permanence_miss")
    camera_vs_videomme = same_ledger("camera_type_match", "videomme_accuracy")
    vbench_vs_videomme = same_ledger(
        "vbench_dynamic_degree",
        "videomme_accuracy",
    )

    cup_vanished = gen_occupancy < 0.15
    understand_ok = answer == YES and hist_occupancy[-1] > 0.9

    checks = {
        "understand_ce_and_generate_masks_disjoint": intersection == set(),
        "prompt_tokens_excluded_from_ce": not prompt_in_ce,
        "history_pixels_excluded_from_ce": not hist_in_ce,
        "history_pixels_excluded_from_generate": not hist_in_gen,
        "future_pixels_excluded_from_ce": not future_in_ce,
        "answer_token_excluded_from_generate": not answer_in_gen,
        "illegal_or_mask_covers_both_ledgers": shared_pos == ce_pos | gen_pos
        and len(shared_pos) == len(ce_pos) + len(gen_pos),
        "understand_answers_cup_still_there": understand_ok,
        "generated_frame_cup_vanished": cup_vanished,
        "copy_last_frame_keeps_cup": copy_occupancy > 0.9,
        "truth_next_frame_keeps_cup": truth_occupancy > 0.9,
        "videomme_accuracy_rejected_as_generation_score": not videomme_as_gen,
        "object_permanence_stays_on_generation_ledger": permanence_as_gen,
        "camera_match_not_same_ledger_as_videomme": not camera_vs_videomme,
        "vbench_not_same_ledger_as_videomme": not vbench_vs_videomme,
        "mean_fill_l2_beats_copy_while_cup_vanishes": gen_l2 < copy_l2
        and cup_vanished
        and copy_occupancy > 0.9,
    }

    return {
        "summary": (
            "Observed 5 frames keep the cup; caption CE is correct; "
            "forgetful next-frame mix drops cup occupancy to 0. "
            "CE mask and frame-diff mask occupy disjoint sequence indices."
        ),
        "metrics": {
            "seq_len": SEQ_LEN,
            "n_hist": N_HIST,
            "n_prompt": N_PROMPT,
            "answer_index": ANSWER_INDEX,
            "future_start": FUTURE_START,
            "ce_positions": len(ce_pos),
            "gen_positions": len(gen_pos),
            "intersection_size": len(intersection),
            "shared_or_size": len(shared_pos),
            "hist_occupancy_last": round(hist_occupancy[-1], 6),
            "truth_occupancy": round(truth_occupancy, 6),
            "copy_occupancy": round(copy_occupancy, 6),
            "gen_occupancy": round(gen_occupancy, 6),
            "understand_answer": answer,
            "p_yes": round(probs[YES], 6),
            "caption_ce": round(ce, 6),
            "copy_l2": round(copy_l2, 6),
            "gen_l2": round(gen_l2, 6),
            "forget": FORGET,
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="49",
    title="把视频生成和视频理解拆开记账",
    question="生成下一帧的帧差损失和看懂这一段的 caption CE，有效位置是否不相交？",
    run=run,
)
