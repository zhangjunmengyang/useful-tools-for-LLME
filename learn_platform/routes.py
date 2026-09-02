"""HTTP routes for the multi-topic learn platform."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from learn_platform.catalog import list_topics, topic_lesson, topic_outline

router = APIRouter(prefix="/api/learn", tags=["learn"])


@router.get("/topics")
def get_topics() -> dict[str, Any]:
    """Return the four switcher topics."""
    return {"topics": list_topics()}


@router.get("/topics/{topic_id}")
def get_topic(topic_id: str) -> dict[str, Any]:
    """Return one topic outline with units and lessons."""
    try:
        return topic_outline(topic_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown topic: {topic_id}") from exc


@router.get("/topics/{topic_id}/lessons/{lesson_id}")
def get_lesson(topic_id: str, lesson_id: str) -> dict[str, Any]:
    """Return read / learn / play payloads for one lesson."""
    try:
        topic_outline(topic_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown topic: {topic_id}") from exc
    try:
        return topic_lesson(topic_id, lesson_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown lesson: {lesson_id}") from exc
