from __future__ import annotations

from .core import LessonExperiment
from .lessons import (
    lesson_01,
    lesson_02,
    lesson_03,
    lesson_04,
    lesson_05,
    lesson_06,
    lesson_07,
    lesson_08,
    lesson_09,
    lesson_10,
    lesson_11,
    lesson_12,
    lesson_13,
    lesson_14,
    lesson_15,
    lesson_16,
    lesson_17,
    lesson_18,
    lesson_19,
    lesson_20,
    lesson_21,
    lesson_22,
    lesson_23,
    lesson_24,
)

_LESSONS = (
    lesson_01.LESSON,
    lesson_02.LESSON,
    lesson_03.LESSON,
    lesson_04.LESSON,
    lesson_05.LESSON,
    lesson_06.LESSON,
    lesson_07.LESSON,
    lesson_08.LESSON,
    lesson_09.LESSON,
    lesson_10.LESSON,
    lesson_11.LESSON,
    lesson_12.LESSON,
    lesson_13.LESSON,
    lesson_14.LESSON,
    lesson_15.LESSON,
    lesson_16.LESSON,
    lesson_17.LESSON,
    lesson_18.LESSON,
    lesson_19.LESSON,
    lesson_20.LESSON,
    lesson_21.LESSON,
    lesson_22.LESSON,
    lesson_23.LESSON,
    lesson_24.LESSON,
)

LESSONS: dict[str, LessonExperiment] = {
    lesson.lesson_id: lesson for lesson in _LESSONS
}

if len(LESSONS) != 24:
    raise RuntimeError("Learn CL must register exactly 24 unique lessons")


def get_lesson(lesson_id: str) -> LessonExperiment:
    normalized = lesson_id.zfill(2)
    try:
        return LESSONS[normalized]
    except KeyError as error:
        raise KeyError(f"Unknown lesson: {lesson_id}") from error
