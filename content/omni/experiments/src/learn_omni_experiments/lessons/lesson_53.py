from __future__ import annotations

from typing import Any

from ..core import LessonExperiment

HEADER_BYTES = 64
BOX_BYTES = 16
HEIGHT = 64
WIDTH = 64
CHANNELS = 3
PIXEL_BYTES = HEIGHT * WIDTH * CHANNELS
BUDGET_BYTES = 16_384
TRUE_COLOR = "blue"

SUMMARIES = {
    "cup_day0": "昨天桌上有一只红色杯子。",
    "cup_day1": "今天桌上有一只蓝色杯子。",
    "note_day0": "便签写着下午三点开会。",
}


def utf8_bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def record_bytes(summary: str, has_box: bool, has_pixels: bool) -> int:
    return (
        HEADER_BYTES
        + utf8_bytes(summary)
        + (BOX_BYTES if has_box else 0)
        + (PIXEL_BYTES if has_pixels else 0)
    )


def color_from_summary(summary: str) -> str | None:
    if "红" in summary:
        return "red"
    if "蓝" in summary:
        return "blue"
    return None


def build_store() -> list[dict[str, Any]]:
    return [
        {
            "id": "cup_day0",
            "entity": "desk_cup",
            "day": 0,
            "summary": SUMMARIES["cup_day0"],
            "has_box": True,
            "has_pixels": True,
            "color": "red",
        },
        {
            "id": "note_day0",
            "entity": "desk_note",
            "day": 0,
            "summary": SUMMARIES["note_day0"],
            "has_box": True,
            "has_pixels": True,
            "color": None,
        },
        {
            "id": "cup_day1",
            "entity": "desk_cup",
            "day": 1,
            "summary": SUMMARIES["cup_day1"],
            "has_box": True,
            "has_pixels": True,
            "color": "blue",
        },
    ]


def store_bytes(records: list[dict[str, Any]]) -> int:
    return sum(
        record_bytes(record["summary"], record["has_box"], record["has_pixels"])
        for record in records
    )


def expire_pixels_keep_summaries(
    records: list[dict[str, Any]],
    budget: int = BUDGET_BYTES,
) -> list[dict[str, Any]]:
    """Drop oldest pixels first until the store fits. Summaries and boxes stay."""
    working = [dict(record) for record in records]
    ordered = sorted(range(len(working)), key=lambda index: (working[index]["day"], index))
    cursor = 0
    while store_bytes(working) > budget and cursor < len(ordered):
        target = working[ordered[cursor]]
        target["has_pixels"] = False
        cursor += 1
    return working


def answer_summary_only(records: list[dict[str, Any]]) -> str:
    """Immutable first-write: never rewrite the entity color sentence."""
    for record in records:
        if record["entity"] == "desk_cup":
            color = color_from_summary(record["summary"])
            if color is not None:
                return color
    return "unknown"


def answer_pixel_only(records: list[dict[str, Any]], budget: int = BUDGET_BYTES) -> str:
    """Keep every raw image. If the three-record bill exceeds the cap, refuse today."""
    if store_bytes(records) > budget:
        return "budget_exceeded"
    latest = max(
        (record for record in records if record["entity"] == "desk_cup" and record["has_pixels"]),
        key=lambda record: record["day"],
        default=None,
    )
    if latest is None:
        return "unknown"
    return str(latest["color"])


def answer_hybrid(records: list[dict[str, Any]], budget: int = BUDGET_BYTES) -> str:
    """Rewrite the cup summary on day 1, then expire old pixels under the cap."""
    expired = expire_pixels_keep_summaries(records, budget)
    latest = max(
        (record for record in expired if record["entity"] == "desk_cup"),
        key=lambda record: record["day"],
    )
    color = color_from_summary(latest["summary"])
    if store_bytes(expired) > budget:
        return "budget_exceeded"
    return color or "unknown"


def run() -> dict[str, Any]:
    records = build_store()
    per_record = {
        record["id"]: record_bytes(record["summary"], True, True) for record in records
    }
    full_bytes = store_bytes(records)
    expired = expire_pixels_keep_summaries(records)
    expired_bytes = store_bytes(expired)
    expired_pixel_flags = [record["has_pixels"] for record in expired]
    expired_summaries = [record["summary"] for record in expired]

    summary_answer = answer_summary_only(records)
    pixel_answer = answer_pixel_only(records)
    hybrid_answer = answer_hybrid(records)

    cup_day0_utf8 = utf8_bytes(SUMMARIES["cup_day0"])
    cup_day0_formula = HEADER_BYTES + cup_day0_utf8 + BOX_BYTES + PIXEL_BYTES

    checks = {
        "three_records_with_pixels_exceed_budget": full_bytes > BUDGET_BYTES
        and len(records) == 3,
        "expire_drops_pixels_keeps_summaries": expired_pixel_flags.count(True) < 3
        and expired_summaries == [
            SUMMARIES["cup_day0"],
            SUMMARIES["note_day0"],
            SUMMARIES["cup_day1"],
        ],
        "after_expire_under_budget": expired_bytes <= BUDGET_BYTES < full_bytes,
        "summary_only_answers_stale_red": summary_answer == "red"
        and TRUE_COLOR == "blue",
        "pixel_only_blows_budget": pixel_answer == "budget_exceeded",
        "hybrid_answers_blue_under_budget": hybrid_answer == TRUE_COLOR
        and expired_bytes <= BUDGET_BYTES,
        "oldest_pixels_deleted_first": expired[0]["has_pixels"] is False
        and expired[1]["has_pixels"] is False
        and expired[2]["has_pixels"] is True,
        "byte_formula_matches_header_utf8_box_pixels": per_record["cup_day0"]
        == cup_day0_formula
        == HEADER_BYTES + cup_day0_utf8 + BOX_BYTES + PIXEL_BYTES,
    }

    return {
        "summary": (
            "三条跨会话记录在 64x64x3 像素下超过 16384 字节上限；"
            "过期删掉旧像素、留下摘要和框之后账单回到上限内。"
            "只读首条摘要会答红色，混合策略在预算内答蓝色。"
        ),
        "metrics": {
            "record_count": len(records),
            "header_bytes": HEADER_BYTES,
            "box_bytes": BOX_BYTES,
            "pixel_bytes": PIXEL_BYTES,
            "budget_bytes": BUDGET_BYTES,
            "cup_day0_utf8": cup_day0_utf8,
            "cup_day0_full_bytes": per_record["cup_day0"],
            "full_store_bytes": full_bytes,
            "expired_store_bytes": expired_bytes,
            "expired_pixel_count": expired_pixel_flags.count(True),
            "summary_only_answer": summary_answer,
            "pixel_only_answer": pixel_answer,
            "hybrid_answer": hybrid_answer,
            "true_color": TRUE_COLOR,
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="53",
    title="跨会话记忆：像素、框与摘要的字节上限",
    question="隔天再问杯子颜色时，只存摘要会答错、只存原图像素会超过字节上限吗？",
    run=run,
)
