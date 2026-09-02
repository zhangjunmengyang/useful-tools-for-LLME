from __future__ import annotations

import hashlib
import re
from typing import Any

from ..core import LessonExperiment


REQUIRED_FIELDS: tuple[str, ...] = (
    "sample_id",
    "source_url",
    "license",
    "sha256",
    "is_synthetic",
    "retractable",
)

ALLOWED_LICENSES: frozenset[str] = frozenset(
    {
        "CC-BY-4.0",
        "CC0-1.0",
        "Apache-2.0",
    },
)

ILLEGAL_LICENSE_TOKENS: frozenset[str] = frozenset(
    {
        "",
        "unspecified",
        "unknown",
        "n/a",
        "none",
    },
)

SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")

PAYLOADS: dict[str, bytes] = {
    "img-001": b"learn-omni:cup-photo:v1",
    "img-002": b"learn-omni:street-photo:v1",
    "img-003": b"learn-omni:synth-table:v1",
    "img-004": b"learn-omni:hash-mismatch:v1",
}


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def missing_fields(row: dict[str, Any]) -> list[str]:
    absent: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in row:
            absent.append(field)
            continue
        value = row[field]
        if value is None:
            absent.append(field)
            continue
        if isinstance(value, str) and value.strip() == "":
            absent.append(field)
    return absent


def license_is_illegal(license_value: object) -> bool:
    if not isinstance(license_value, str):
        return True
    token = license_value.strip().lower()
    if token in ILLEGAL_LICENSE_TOKENS:
        return True
    return license_value.strip() not in ALLOWED_LICENSES


def hash_is_illegal(digest: object) -> bool:
    if not isinstance(digest, str):
        return True
    return SHA256_HEX.fullmatch(digest.strip().lower()) is None


def validate_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic gate record for one sidecar line."""
    absent = missing_fields(row)
    license_value = row.get("license", "")
    digest = row.get("sha256", "")
    sample_id = str(row.get("sample_id", ""))
    payload = PAYLOADS.get(sample_id)
    expected = sha256_hex(payload) if payload is not None else ""
    hash_mismatch = (
        payload is not None
        and isinstance(digest, str)
        and SHA256_HEX.fullmatch(digest.strip().lower()) is not None
        and digest.strip().lower() != expected
    )
    reasons: list[str] = []
    if "license" in absent or license_is_illegal(license_value):
        reasons.append("missing_or_illegal_license")
    if "sha256" in absent or hash_is_illegal(digest):
        reasons.append("missing_or_illegal_hash")
    other_absent = [field for field in absent if field not in {"license", "sha256"}]
    if other_absent:
        reasons.append("missing_required_fields")
    if hash_mismatch:
        reasons.append("hash_mismatch")
    admitted = len(reasons) == 0
    return {
        "sample_id": sample_id,
        "admitted": admitted,
        "reasons": reasons,
        "missing_fields": absent,
        "expected_sha256": expected,
    }


def _row(
    sample_id: str,
    *,
    license_value: str,
    digest: str | None = None,
    source_url: str = "https://example.org/asset.jpg",
    is_synthetic: bool = False,
    retractable: bool = True,
    drop: tuple[str, ...] = (),
) -> dict[str, Any]:
    payload = PAYLOADS[sample_id]
    record: dict[str, Any] = {
        "sample_id": sample_id,
        "source_url": source_url,
        "license": license_value,
        "sha256": sha256_hex(payload) if digest is None else digest,
        "is_synthetic": is_synthetic,
        "retractable": retractable,
    }
    for field in drop:
        record.pop(field, None)
    return record


def run() -> dict[str, Any]:
    complete = _row(
        "img-001",
        license_value="CC-BY-4.0",
        source_url="https://example.org/cup.jpg",
    )
    missing_license = _row(
        "img-002",
        license_value="",
        source_url="https://cdn.example.net/street.png",
        retractable=False,
    )
    missing_hash = _row(
        "img-003",
        license_value="CC-BY-4.0",
        digest="",
        source_url="https://gen.example.ai/table.webp",
        is_synthetic=True,
    )
    unspecified = _row("img-001", license_value="unspecified")
    unknown_license = _row("img-001", license_value="Unknown")
    nc_license = _row("img-001", license_value="CC-BY-NC-4.0")
    mismatch = _row(
        "img-004",
        license_value="CC-BY-4.0",
        digest=sha256_hex(PAYLOADS["img-004"])[:-1] + (
            "0" if sha256_hex(PAYLOADS["img-004"])[-1] != "0" else "1"
        ),
    )
    missing_url = _row("img-001", license_value="CC-BY-4.0", drop=("source_url",))
    synthetic_ok = _row(
        "img-003",
        license_value="CC-BY-4.0",
        source_url="https://gen.example.ai/table.webp",
        is_synthetic=True,
    )
    synthetic_no_license = _row(
        "img-003",
        license_value="",
        source_url="https://gen.example.ai/table.webp",
        is_synthetic=True,
    )
    not_retractable = _row(
        "img-001",
        license_value="CC-BY-4.0",
        retractable=False,
    )

    complete_gate = validate_row(complete)
    license_gate = validate_row(missing_license)
    hash_gate = validate_row(missing_hash)
    unspecified_gate = validate_row(unspecified)
    unknown_gate = validate_row(unknown_license)
    nc_gate = validate_row(nc_license)
    mismatch_gate = validate_row(mismatch)
    url_gate = validate_row(missing_url)
    synthetic_ok_gate = validate_row(synthetic_ok)
    synthetic_no_license_gate = validate_row(synthetic_no_license)
    retract_gate = validate_row(not_retractable)

    fixture_rows = (complete, missing_license, missing_hash)
    admitted_ids = [
        str(row["sample_id"])
        for row in fixture_rows
        if validate_row(row)["admitted"]
    ]

    checks = {
        "required_field_count_is_six": len(REQUIRED_FIELDS) == 6,
        "complete_row_is_admitted": complete_gate["admitted"] is True,
        "missing_license_is_illegal": (
            license_gate["admitted"] is False
            and "missing_or_illegal_license" in license_gate["reasons"]
        ),
        "missing_hash_is_illegal": (
            hash_gate["admitted"] is False
            and "missing_or_illegal_hash" in hash_gate["reasons"]
        ),
        "unspecified_license_is_illegal": unspecified_gate["admitted"] is False,
        "unknown_license_token_is_illegal": unknown_gate["admitted"] is False,
        "noncommercial_license_not_in_allowed_set": nc_gate["admitted"] is False,
        "hash_mismatch_is_illegal": (
            mismatch_gate["admitted"] is False
            and "hash_mismatch" in mismatch_gate["reasons"]
        ),
        "missing_source_url_is_illegal": (
            url_gate["admitted"] is False
            and "missing_required_fields" in url_gate["reasons"]
        ),
        "synthetic_flag_does_not_bypass_license": (
            synthetic_ok_gate["admitted"] is True
            and synthetic_no_license_gate["admitted"] is False
        ),
        "retractable_false_still_requires_license_and_hash": (
            retract_gate["admitted"] is True
        ),
        "three_sample_fixture_admits_only_complete_row": admitted_ids == ["img-001"],
        "complete_hash_matches_payload": (
            complete["sha256"] == sha256_hex(PAYLOADS["img-001"])
        ),
    }

    return {
        "summary": (
            "用六项必填 sidecar 字段做训练集准入门："
            "缺许可或缺哈希为非法；哈希与载荷不一致为非法；"
            "合成标记不能绕过许可；三条夹具里只有完整行进入训练集。"
        ),
        "metrics": {
            "required_fields": list(REQUIRED_FIELDS),
            "allowed_licenses": sorted(ALLOWED_LICENSES),
            "sha256_hex_length": 64,
            "complete_sha256": complete["sha256"],
            "admitted_sample_ids": admitted_ids,
            "admitted_count": len(admitted_ids),
            "rejected_count": len(fixture_rows) - len(admitted_ids),
            "missing_license_reasons": license_gate["reasons"],
            "missing_hash_reasons": hash_gate["reasons"],
            "hash_mismatch_reasons": mismatch_gate["reasons"],
            "synthetic_ok_admitted": synthetic_ok_gate["admitted"],
            "synthetic_no_license_admitted": synthetic_no_license_gate["admitted"],
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="57",
    title="给训练图像留下可核查出处",
    question="一张图进训练集前，缺许可或缺哈希为什么必须被判为非法？",
    run=run,
)
