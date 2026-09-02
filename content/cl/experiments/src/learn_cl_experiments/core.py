from __future__ import annotations

import hashlib
import inspect
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ExperimentPayload = dict[str, Any]
RESULT_SCHEMA = {
    "name": "learn-cl-experiment-result",
    "version": 1,
}
RESULT_FIELDS = {
    "schema",
    "lesson_id",
    "title",
    "question",
    "summary",
    "metrics",
    "checks",
    "runtime",
    "source_digest",
}
LESSON_RESULT_FIELDS = {"summary", "metrics", "checks"}


class ResultValidationError(ValueError):
    """Raised when an experiment result does not match the release contract."""


def runtime_metadata() -> dict[str, str]:
    """Return stable facts about the interpreter that produced an artifact."""
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_cache_tag": sys.implementation.cache_tag or "unknown",
        "platform_system": platform.system() or "unknown",
        "platform_machine": platform.machine() or "unknown",
    }


def _source_digest(run: Callable[[], ExperimentPayload]) -> dict[str, str]:
    lesson_source_name = inspect.getsourcefile(run)
    if lesson_source_name is None:
        raise ResultValidationError(
            f"Cannot locate source file for {run.__module__}.{run.__name__}",
        )

    lesson_source = Path(lesson_source_name)
    core_source = Path(__file__)
    try:
        lesson_bytes = lesson_source.read_bytes()
        core_bytes = core_source.read_bytes()
    except OSError as error:
        raise ResultValidationError(
            f"Cannot read experiment source: {error}",
        ) from error

    digest = hashlib.sha256()
    digest.update(b"learn-cl:lesson-source:v1\0")
    digest.update(lesson_bytes)
    digest.update(b"\0learn-cl:experiment-core:v1\0")
    digest.update(core_bytes)
    return {
        "algorithm": "sha256",
        "scope": "lesson-module+experiment-core",
        "module": run.__module__,
        "value": digest.hexdigest(),
    }


def _validate_json_value(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ResultValidationError(f"{path} must contain only finite numbers")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ResultValidationError(
                    f"{path} object keys must be non-empty strings",
                )
            _validate_json_value(item, f"{path}.{key}")
        return
    raise ResultValidationError(
        f"{path} contains unsupported JSON type {type(value).__name__}",
    )


def validate_result_payload(
    payload: Any,
    lesson: LessonExperiment,
) -> dict[str, Any]:
    """Validate both freshly produced and persisted lesson artifacts."""
    if not isinstance(payload, dict):
        raise ResultValidationError("result payload must be a JSON object")

    fields = set(payload)
    missing = RESULT_FIELDS - fields
    unexpected = fields - RESULT_FIELDS
    if missing or unexpected:
        raise ResultValidationError(
            "result fields mismatch; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}",
        )

    if payload["schema"] != RESULT_SCHEMA:
        raise ResultValidationError(
            f"schema must equal {RESULT_SCHEMA!r}",
        )
    if payload["lesson_id"] != lesson.lesson_id:
        raise ResultValidationError(
            f"lesson_id must equal {lesson.lesson_id!r}",
        )
    if payload["title"] != lesson.title:
        raise ResultValidationError("title does not match the registered lesson")
    if payload["question"] != lesson.question:
        raise ResultValidationError(
            "question does not match the registered lesson",
        )

    summary = payload["summary"]
    if not isinstance(summary, str) or not summary.strip():
        raise ResultValidationError("summary must be a non-empty string")

    metrics = payload["metrics"]
    if not isinstance(metrics, dict) or not metrics:
        raise ResultValidationError("metrics must be a non-empty object")

    checks = payload["checks"]
    if not isinstance(checks, dict) or not checks:
        raise ResultValidationError("checks must be a non-empty object")
    if any(not isinstance(name, str) or not name for name in checks):
        raise ResultValidationError("check names must be non-empty strings")
    if any(not isinstance(value, bool) for value in checks.values()):
        raise ResultValidationError("checks must contain only booleans")

    expected_runtime = runtime_metadata()
    if payload["runtime"] != expected_runtime:
        raise ResultValidationError(
            "runtime metadata does not match the current interpreter",
        )

    expected_source_digest = lesson.source_digest()
    if payload["source_digest"] != expected_source_digest:
        raise ResultValidationError(
            "source digest is stale or does not match the registered lesson",
        )

    _validate_json_value(payload, "result")
    return payload


@dataclass(frozen=True)
class LessonExperiment:
    lesson_id: str
    title: str
    question: str
    run: Callable[[], ExperimentPayload]

    def source_digest(self) -> dict[str, str]:
        return _source_digest(self.run)

    def execute(self) -> ExperimentPayload:
        payload = self.run()
        if not isinstance(payload, dict):
            raise ResultValidationError(
                f"Lesson {self.lesson_id} must return a result object",
            )
        fields = set(payload)
        missing = LESSON_RESULT_FIELDS - fields
        unexpected = fields - LESSON_RESULT_FIELDS
        if missing or unexpected:
            raise ResultValidationError(
                f"Lesson {self.lesson_id} result fields mismatch; "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}",
            )

        result = {
            "schema": dict(RESULT_SCHEMA),
            "lesson_id": self.lesson_id,
            "title": self.title,
            "question": self.question,
            **payload,
            "runtime": runtime_metadata(),
            "source_digest": self.source_digest(),
        }
        return validate_result_payload(result, self)
