from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .core import (
    ExperimentPayload,
    RESULT_FIELDS,
    ResultValidationError,
    _source_digest,
    _validate_json_value,
    runtime_metadata,
)

EXTRA_SCHEMA = {
    "name": "learn-cl-extra-result",
    "version": 1,
}


def extra_source_digest(run: Callable[[], ExperimentPayload]) -> dict[str, str]:
    digest = _source_digest(run)
    helper = Path(__file__).with_name("lin.py")
    extra_core = Path(__file__)
    blob = hashlib.sha256()
    blob.update(digest["value"].encode("ascii"))
    blob.update(b"\0learn-cl:lin\0")
    blob.update(helper.read_bytes())
    blob.update(b"\0learn-cl:extra-core\0")
    blob.update(extra_core.read_bytes())
    return {
        "algorithm": "sha256",
        "scope": "extra-module+lin+extra-core",
        "module": run.__module__,
        "value": blob.hexdigest(),
    }


@dataclass(frozen=True)
class ExtraExperiment:
    extra_id: str
    title: str
    question: str
    lesson_hint: str
    run: Callable[[], ExperimentPayload]

    @property
    def lesson_id(self) -> str:
        return self.extra_id

    def source_digest(self) -> dict[str, str]:
        return extra_source_digest(self.run)

    def execute(self) -> dict[str, Any]:
        payload = self.run()
        if not isinstance(payload, dict):
            raise ResultValidationError(
                f"Extra {self.extra_id} must return a result object",
            )
        required = {"summary", "metrics", "checks"}
        missing = required - set(payload)
        unexpected = set(payload) - required
        if missing or unexpected:
            raise ResultValidationError(
                f"Extra {self.extra_id} result fields mismatch; "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}",
            )
        result = {
            "schema": dict(EXTRA_SCHEMA),
            "lesson_id": self.extra_id,
            "title": self.title,
            "question": self.question,
            **payload,
            "runtime": runtime_metadata(),
            "source_digest": self.source_digest(),
        }
        return _validate_extra(result, self)


def _validate_extra(payload: dict[str, Any], extra: ExtraExperiment) -> dict[str, Any]:
    # Reuse field-shape checks, then swap schema expectation.
    if payload.get("schema") != EXTRA_SCHEMA:
        raise ResultValidationError(f"schema must equal {EXTRA_SCHEMA!r}")
    if payload.get("lesson_id") != extra.extra_id:
        raise ResultValidationError("lesson_id must equal extra_id")
    if payload.get("title") != extra.title:
        raise ResultValidationError("title does not match the registered extra")
    if payload.get("question") != extra.question:
        raise ResultValidationError("question does not match the registered extra")
    summary = payload.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ResultValidationError("summary must be a non-empty string")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ResultValidationError("metrics must be a non-empty object")
    checks = payload.get("checks")
    if not isinstance(checks, dict) or not checks:
        raise ResultValidationError("checks must be a non-empty object")
    if any(not isinstance(name, str) or not name for name in checks):
        raise ResultValidationError("check names must be non-empty strings")
    if any(not isinstance(value, bool) for value in checks.values()):
        raise ResultValidationError("checks must contain only booleans")
    if payload.get("runtime") != runtime_metadata():
        raise ResultValidationError("runtime metadata does not match the current interpreter")
    if payload.get("source_digest") != extra.source_digest():
        raise ResultValidationError("source digest is stale or does not match the extra")
    fields = set(payload)
    missing = RESULT_FIELDS - fields
    unexpected = fields - RESULT_FIELDS
    if missing or unexpected:
        raise ResultValidationError(
            "result fields mismatch; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}",
        )
    _validate_json_value(payload, "result")
    return payload


def validate_extra_payload(payload: Any, extra: ExtraExperiment) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ResultValidationError("result payload must be a JSON object")
    return _validate_extra(payload, extra)
