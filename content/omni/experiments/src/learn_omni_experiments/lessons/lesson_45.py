from __future__ import annotations

import math
from typing import Any

from ..core import LessonExperiment

CLIP_COUNT = 60
TARGET_INDEX = 46  # minute 47 on a 0-based 60-minute tape
TOP_K = 5
READER_BUDGET = 4096
COST_SUBTITLE = 40
COST_MID = 192
COST_PIXEL = 3840
COST_FINE_READ = 512
QUERY = (4.0, 3.0, 2.0, 0.1)


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _norm(vector: tuple[float, ...]) -> float:
    return math.sqrt(_dot(vector, vector))


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    denom = _norm(left) * _norm(right)
    if denom == 0.0:
        return 0.0
    return _dot(left, right) / denom


def _rank(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda index: (-scores[index], index))


def _recall_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hit = len(relevant.intersection(ranked[:k]))
    return hit / len(relevant)


def _clip_vectors() -> tuple[
    list[tuple[float, ...]],
    list[tuple[float, ...]],
    list[tuple[float, ...]],
]:
    subtitles: list[tuple[float, ...]] = []
    mids: list[tuple[float, ...]] = []
    pixels: list[tuple[float, ...]] = []
    for index in range(CLIP_COUNT):
        if index == TARGET_INDEX:
            # Silent action: empty ASR, unique mid-layer valve signature.
            subtitles.append((0.0, 0.0, 0.0, 0.2))
            mids.append((3.8, 2.9, 2.1, 0.0))
            pixels.append((0.82, 0.12, 0.10, 0.0))
            continue
        if index == 11:
            # Spoken "turn the valve" over a talking head: subtitle trap.
            subtitles.append((0.2, 3.4, 3.1, 1.0))
            mids.append((0.4, 0.3, 0.2, 2.8))
            pixels.append((0.31, 0.33, 0.29, 0.0))
            continue
        if index in {8, 22, 39}:
            # Other red objects: pixel neighbors, not the valve action.
            subtitles.append((0.1, 0.0, 0.0, 0.8 + 0.01 * index))
            mids.append((1.1, 0.2, 0.1, 0.4))
            pixels.append((0.80, 0.14, 0.11, 0.0))
            continue
        phase = (index % 7) * 0.15
        subtitles.append((0.05, 0.08 + phase, 0.04, 0.6 + 0.01 * (index % 5)))
        mids.append((0.2 + phase, 0.15, 0.12, 0.7))
        pixels.append((0.28 + phase * 0.1, 0.30, 0.27, 0.0))
    return subtitles, mids, pixels


def run() -> dict[str, Any]:
    subtitles, mids, pixels = _clip_vectors()
    relevant = {TARGET_INDEX}

    subtitle_scores = [_cosine(QUERY, vector) for vector in subtitles]
    mid_scores = [_cosine(QUERY, vector) for vector in mids]
    pixel_query = (0.55, 0.52, 0.48, 0.0)
    pixel_scores = [_cosine(pixel_query, vector) for vector in pixels]

    subtitle_rank = _rank(subtitle_scores)
    mid_rank = _rank(mid_scores)
    pixel_rank = _rank(pixel_scores)

    subtitle_recall = _recall_at_k(subtitle_rank, relevant, TOP_K)
    mid_recall = _recall_at_k(mid_rank, relevant, TOP_K)
    pixel_recall = _recall_at_k(pixel_rank, relevant, TOP_K)

    subtitle_budget = TOP_K * COST_SUBTITLE
    mid_budget = TOP_K * COST_MID
    pixel_budget = TOP_K * COST_PIXEL
    pixel_full_scan = CLIP_COUNT * COST_PIXEL

    reader_from_subtitle = set(subtitle_rank[:TOP_K])
    reader_from_mid = set(mid_rank[:TOP_K])
    reader_from_pixel = set(pixel_rank[:TOP_K])

    hierarchical_candidates = mid_rank[:TOP_K]
    hierarchical_budget = mid_budget + 2 * COST_FINE_READ
    hierarchical_has_target = TARGET_INDEX in hierarchical_candidates

    target_subtitle_score = subtitle_scores[TARGET_INDEX]
    trap_subtitle_score = subtitle_scores[11]
    target_mid_score = mid_scores[TARGET_INDEX]
    next_mid_score = mid_scores[mid_rank[1]]

    checks = {
        "target_missing_from_subtitle_topk": subtitle_recall == 0.0
        and TARGET_INDEX not in reader_from_subtitle,
        "target_recalled_only_in_mid_layer": mid_recall == 1.0
        and pixel_recall == 0.0
        and subtitle_recall == 0.0
        and TARGET_INDEX == mid_rank[0],
        "wrong_layer_reader_never_sees_target": TARGET_INDEX
        not in reader_from_subtitle
        and TARGET_INDEX not in reader_from_pixel,
        "pixel_or_fullscan_exceeds_reader_budget": pixel_budget > READER_BUDGET
        and pixel_full_scan > READER_BUDGET
        and mid_budget <= READER_BUDGET
        and subtitle_budget <= READER_BUDGET,
        "spoken_valve_is_harder_than_silent_target_on_asr": trap_subtitle_score
        > target_subtitle_score,
        "mid_score_separates_target_from_runner_up": target_mid_score
        > next_mid_score + 1e-9,
        "recall_at_k_matches_set_formula": abs(
            mid_recall - len(relevant.intersection(mid_rank[:TOP_K])) / len(relevant)
        )
        < 1e-12,
        "hierarchical_keeps_target_under_budget": hierarchical_has_target
        and hierarchical_budget <= READER_BUDGET
        and hierarchical_budget < pixel_full_scan,
    }

    return {
        "summary": (
            "在 60 段一分钟索引上核对三层召回：目标在第 47 分钟，"
            "只在中间层进入 Recall@5；字幕层被有声阀门描述带走，"
            "像素层把查询投到错误空间且整段扫描超过阅读预算。"
        ),
        "metrics": {
            "clip_count": CLIP_COUNT,
            "target_index": TARGET_INDEX,
            "target_minute": TARGET_INDEX + 1,
            "top_k": TOP_K,
            "reader_budget": READER_BUDGET,
            "subtitle_recall_at_k": subtitle_recall,
            "mid_recall_at_k": mid_recall,
            "pixel_recall_at_k": pixel_recall,
            "subtitle_budget": subtitle_budget,
            "mid_budget": mid_budget,
            "pixel_topk_budget": pixel_budget,
            "pixel_full_scan_budget": pixel_full_scan,
            "hierarchical_budget": hierarchical_budget,
            "target_subtitle_score": target_subtitle_score,
            "trap_subtitle_score": trap_subtitle_score,
            "target_mid_score": target_mid_score,
            "subtitle_rank_of_target": subtitle_rank.index(TARGET_INDEX),
            "mid_rank_of_target": mid_rank.index(TARGET_INDEX),
            "pixel_rank_of_target": pixel_rank.index(TARGET_INDEX),
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="45",
    title="为长视频选择检索层",
    question="长视频和长操作记录该检索字幕、中间特征还是像素，错误层召回为何让精读永远看不见目标段？",
    run=run,
)
