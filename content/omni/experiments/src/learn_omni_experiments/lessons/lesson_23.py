from __future__ import annotations

from typing import Any

from ..core import LessonExperiment


IOU_THRESHOLD = 0.5
ATTENTION_HIT_THRESHOLD = 0.5


def _box_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(pred: tuple[float, float, float, float], gt: tuple[float, float, float, float]) -> float:
    """Intersection-over-union of axis-aligned boxes `(x1, y1, x2, y2)`."""
    ix1 = max(pred[0], gt[0])
    iy1 = max(pred[1], gt[1])
    ix2 = min(pred[2], gt[2])
    iy2 = min(pred[3], gt[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = _box_area(pred) + _box_area(gt) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def grounding_hit(
    pred: tuple[float, float, float, float],
    gt: tuple[float, float, float, float],
    threshold: float = IOU_THRESHOLD,
) -> bool:
    return iou(pred, gt) >= threshold


def attention_mass(weights: list[list[float]], mask: list[list[int]]) -> float:
    """Fraction of attention sitting on the object mask."""
    total = 0.0
    on_object = 0.0
    for row_w, row_m in zip(weights, mask):
        for weight, flag in zip(row_w, row_m):
            total += weight
            if flag:
                on_object += weight
    if total <= 0.0:
        return 0.0
    return on_object / total


def attention_hit(
    weights: list[list[float]],
    mask: list[list[int]],
    threshold: float = ATTENTION_HIT_THRESHOLD,
) -> bool:
    return attention_mass(weights, mask) >= threshold


def hallucination_rate(exists: list[int], predicted_yes: list[int]) -> float:
    """Share of negative probes (object absent) that the model still answers yes."""
    negatives = 0
    false_yes = 0
    for present, yes in zip(exists, predicted_yes):
        if present == 0:
            negatives += 1
            if yes == 1:
                false_yes += 1
    if negatives == 0:
        return 0.0
    return false_yes / negatives


def chair_i(mentioned: list[str], present: set[str]) -> float:
    hallucinated = [name for name in mentioned if name not in present]
    if not mentioned:
        return 0.0
    return len(hallucinated) / len(mentioned)


def chair_s(captions: list[list[str]], present_sets: list[set[str]]) -> float:
    if not captions:
        return 0.0
    bad = 0
    for mentioned, present in zip(captions, present_sets):
        if any(name not in present for name in mentioned):
            bad += 1
    return bad / len(captions)


def _zeros(side: int) -> list[list[float]]:
    return [[0.0 for _ in range(side)] for _ in range(side)]


def _mask(side: int, cells: list[tuple[int, int]]) -> list[list[int]]:
    grid = [[0 for _ in range(side)] for _ in range(side)]
    for row, col in cells:
        grid[row][col] = 1
    return grid


def run() -> dict[str, Any]:
    known_pred = (0.0, 0.0, 2.0, 2.0)
    known_gt = (1.0, 1.0, 3.0, 3.0)
    known_iou = iou(known_pred, known_gt)

    identical_iou = iou((2.0, 2.0, 6.0, 5.0), (2.0, 2.0, 6.0, 5.0))
    disjoint_iou = iou((0.0, 0.0, 1.0, 1.0), (2.0, 2.0, 3.0, 3.0))
    touching_iou = iou((0.0, 0.0, 1.0, 1.0), (1.0, 0.0, 2.0, 1.0))

    # Two red cups. VQA "what color is the left cup?" is "red".
    # The predicted box sits on the right cup, so IoU with the left cup is 0.
    left_cup = (8.0, 8.0, 24.0, 28.0)
    right_cup = (72.0, 40.0, 90.0, 62.0)
    vqa_answer = "red"
    vqa_label = "red"
    predicted_box = right_cup
    answer_correct = vqa_answer == vqa_label
    hit_on_queried = grounding_hit(predicted_box, left_cup)
    hit_on_distractor = grounding_hit(predicted_box, right_cup)

    side = 8
    weights = _zeros(side)
    # Language prior dumps mass on the co-occurring plate at (4, 4) and the
    # distractor cup at (4, 2), not on the queried cup at (0, 1).
    weights[0][1] = 0.04
    weights[4][2] = 0.55
    weights[4][4] = 0.31
    weights[6][1] = 0.10
    queried_mask = _mask(side, [(0, 1)])
    distractor_mask = _mask(side, [(4, 2)])
    plate_mask = _mask(side, [(4, 4)])
    queried_mass = attention_mass(weights, queried_mask)
    distractor_mass = attention_mass(weights, distractor_mask)
    plate_mass = attention_mass(weights, plate_mask)
    queried_attention_hit = attention_hit(weights, queried_mask)

    # POPE-style probes. y=1 means the object is in the image.
    exists = [1, 1, 0, 0, 0, 0]
    predicted_yes = [1, 1, 0, 1, 1, 1]
    # negatives: indices 2..5 -> 4 negatives, 3 false yes.
    rate = hallucination_rate(exists, predicted_yes)
    yes_ratio = sum(predicted_yes) / len(predicted_yes)

    popular_exists = [1, 0, 0]
    popular_yes = [1, 1, 1]
    adversarial_exists = [1, 0, 0]
    adversarial_yes = [1, 1, 1]
    random_rate = hallucination_rate(exists, predicted_yes)
    popular_rate = hallucination_rate(popular_exists, popular_yes)
    adversarial_rate = hallucination_rate(adversarial_exists, adversarial_yes)

    mentioned = ["cup", "plate", "wine glass", "fork"]
    present = {"cup", "plate", "fork"}
    caption_chair_i = chair_i(mentioned, present)
    caption_chair_s = chair_s(
        [
            ["cup", "plate"],
            ["cup", "wine glass"],
            ["fork"],
        ],
        [
            {"cup", "plate", "fork"},
            {"cup", "plate", "fork"},
            {"cup", "plate", "fork"},
        ],
    )

    # Boundary: two area-2 boxes overlapping 4/3 have IoU exactly 0.5, which is a hit.
    half_pred = (0.0, 0.0, 2.0, 1.0)
    half_gt = (2.0 / 3.0, 0.0, 8.0 / 3.0, 1.0)
    boundary_iou = iou(half_pred, half_gt)
    boundary_hit = grounding_hit(half_pred, half_gt)
    below_half = iou((0.0, 0.0, 2.0, 1.0), (1.2, 0.0, 3.2, 1.0))
    below_half_miss = below_half < IOU_THRESHOLD

    vqa_accuracy = 1.0 if answer_correct else 0.0
    grounding_accuracy = 1.0 if hit_on_queried else 0.0

    checks = {
        "iou_known_overlap_is_one_seventh": abs(known_iou - (1.0 / 7.0)) < 1e-12,
        "identical_boxes_have_iou_one": identical_iou == 1.0,
        "disjoint_boxes_have_iou_zero": disjoint_iou == 0.0,
        "touching_edges_have_iou_zero": touching_iou == 0.0,
        "vqa_correct_on_twin_cup_fixture": answer_correct,
        "twin_cup_fixture_fails_queried_hit": (not hit_on_queried) and hit_on_distractor,
        "vqa_accuracy_is_not_grounding_accuracy": vqa_accuracy == 1.0
        and grounding_accuracy == 0.0,
        "attention_mass_not_on_queried_cup": queried_mass < ATTENTION_HIT_THRESHOLD,
        "attention_mass_peaks_on_distractor": distractor_mass > queried_mass
        and distractor_mass > plate_mass,
        "queried_attention_hit_fails": not queried_attention_hit,
        "hallucination_rate_is_three_quarters": abs(rate - 0.75) < 1e-12,
        "pope_counts_only_negative_probes": abs(rate - (3 / 4)) < 1e-12,
        "yes_ratio_is_not_hallucination_rate": abs(yes_ratio - rate) > 1e-9,
        "popular_and_adversarial_are_harder_than_mixed_random": popular_rate
        >= random_rate
        and adversarial_rate >= random_rate,
        "chair_i_is_one_quarter": abs(caption_chair_i - 0.25) < 1e-12,
        "chair_s_is_one_third": abs(caption_chair_s - (1.0 / 3.0)) < 1e-12,
        "iou_half_is_inclusive_hit": boundary_hit and abs(boundary_iou - 0.5) < 1e-12,
        "iou_below_half_is_miss": below_half_miss,
    }

    return {
        "summary": (
            "用固定框与 8×8 注意力夹具核验 IoU 命中、注意力质量分数和 POPE 幻觉率；"
            "夹具里 VQA 答案为 red 且正确，但对被问杯子的 IoU 与注意力命中都失败。"
        ),
        "metrics": {
            "known_iou": known_iou,
            "identical_iou": identical_iou,
            "disjoint_iou": disjoint_iou,
            "queried_iou": iou(predicted_box, left_cup),
            "distractor_iou": iou(predicted_box, right_cup),
            "vqa_accuracy": vqa_accuracy,
            "grounding_accuracy": grounding_accuracy,
            "queried_attention_mass": queried_mass,
            "distractor_attention_mass": distractor_mass,
            "plate_attention_mass": plate_mass,
            "hallucination_rate": rate,
            "yes_ratio": yes_ratio,
            "popular_hallucination_rate": popular_rate,
            "adversarial_hallucination_rate": adversarial_rate,
            "chair_i": caption_chair_i,
            "chair_s": caption_chair_s,
            "iou_threshold": IOU_THRESHOLD,
            "attention_hit_threshold": ATTENTION_HIT_THRESHOLD,
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="23",
    title="检验指代、OCR 与空间幻觉",
    question="选择题答对，是否等于模型用对了像素？",
    run=run,
)
