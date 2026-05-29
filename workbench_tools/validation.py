"""Small JSON-schema subset validator for tool inputs."""

from __future__ import annotations

from typing import Any


def _type_matches(value: Any, expected_type: str) -> bool:
    """检查 JSON schema 基础类型。"""
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    return True


def validate_against_schema(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """验证本项目需要的 JSON schema 子集。"""
    errors: list[str] = []
    expected_type = schema.get("type")
    if expected_type and not _type_matches(value, expected_type):
        errors.append(f"{path} must be {expected_type}")
        return errors

    if expected_type == "object":
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                errors.append(f"{path}.{key} is required")

        properties = schema.get("properties", {})
        for key, child_schema in properties.items():
            if key in value:
                errors.extend(validate_against_schema(value[key], child_schema, f"{path}.{key}"))

    if expected_type == "array":
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                errors.extend(validate_against_schema(item, item_schema, f"{path}[{index}]"))

    enum = schema.get("enum")
    if enum and value not in enum:
        errors.append(f"{path} must be one of {enum}")

    return errors
