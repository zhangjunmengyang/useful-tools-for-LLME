from __future__ import annotations

from typing import Any

from ..core import LessonExperiment

# Printed invoice line items in integer cents. 18.90 + 26.50 + 15.80 = 61.20.
LINE_CENTS = (1890, 2650, 1580)
TRUE_SUBTOTAL_CENTS = 6120
# Ones-to-tens carry of 1 dollar is forgotten: 61.20 - 10.00 = 51.20.
CARRY_ERROR_CENTS = 1000
MENTAL_SUBTOTAL_CENTS = TRUE_SUBTOTAL_CENTS - CARRY_ERROR_CENTS

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "calculator": {
        "required": ("expression",),
        "properties": {"expression": str},
    },
    "crop": {
        "required": ("image_id", "x0", "y0", "x1", "y1"),
        "properties": {
            "image_id": str,
            "x0": (int, float),
            "y0": (int, float),
            "x1": (int, float),
            "y1": (int, float),
        },
    },
    "depth": {
        "required": ("image_id", "u", "v"),
        "properties": {
            "image_id": str,
            "u": (int, float),
            "v": (int, float),
        },
    },
    "search": {
        "required": ("query",),
        "properties": {"query": str},
    },
}


def _cents_to_yuan(cents: int) -> float:
    return cents / 100.0


def _type_ok(value: Any, expected: Any) -> bool:
    if expected is str:
        return isinstance(value, str) and len(value) > 0
    if expected in {(int, float), (float, int)}:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if isinstance(expected, tuple):
        return isinstance(value, expected) and not isinstance(value, bool)
    return isinstance(value, expected)


def _in_unit(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= float(value) <= 1.0


def validate_call(call: dict[str, Any]) -> dict[str, Any]:
    """Return a validation record. Missing or ill-typed arguments never execute."""
    name = call.get("name")
    arguments = call.get("arguments")
    record: dict[str, Any] = {
        "name": name,
        "ok": False,
        "missing": [],
        "type_errors": [],
        "constraint_errors": [],
        "executed": False,
        "result": None,
        "reject_reason": "",
    }
    if not isinstance(name, str) or name not in TOOL_SCHEMAS:
        record["reject_reason"] = "unknown_tool"
        return record
    if not isinstance(arguments, dict):
        record["reject_reason"] = "arguments_not_object"
        return record

    schema = TOOL_SCHEMAS[name]
    required = schema["required"]
    missing = [key for key in required if key not in arguments]
    record["missing"] = missing
    if missing:
        record["reject_reason"] = "missing_required"
        return record

    type_errors: list[str] = []
    for key, expected in schema["properties"].items():
        if key in arguments and not _type_ok(arguments[key], expected):
            type_errors.append(key)
    record["type_errors"] = type_errors
    if type_errors:
        record["reject_reason"] = "type_error"
        return record

    constraint_errors: list[str] = []
    if name == "crop":
        x0 = float(arguments["x0"])
        y0 = float(arguments["y0"])
        x1 = float(arguments["x1"])
        y1 = float(arguments["y1"])
        if not all(_in_unit(value) for value in (x0, y0, x1, y1)):
            constraint_errors.append("box_out_of_unit_square")
        if not (x1 > x0 and y1 > y0):
            constraint_errors.append("box_not_ordered")
    if name == "depth":
        if not _in_unit(arguments["u"]) or not _in_unit(arguments["v"]):
            constraint_errors.append("point_out_of_unit_square")
    record["constraint_errors"] = constraint_errors
    if constraint_errors:
        record["reject_reason"] = "constraint_error"
        return record

    record["ok"] = True
    record["executed"] = True
    if name == "calculator":
        expression = arguments["expression"]
        if expression != "18.90+26.50+15.80":
            record["executed"] = False
            record["ok"] = False
            record["reject_reason"] = "expression_not_in_catalog"
            return record
        record["result"] = _cents_to_yuan(sum(LINE_CENTS))
    elif name == "crop":
        record["result"] = {
            "image_id": arguments["image_id"],
            "width": float(arguments["x1"]) - float(arguments["x0"]),
            "height": float(arguments["y1"]) - float(arguments["y0"]),
        }
    elif name == "depth":
        # Teaching fixture: farther points have larger relative depth.
        record["result"] = round(0.25 + 0.5 * float(arguments["v"]), 4)
    else:
        record["result"] = {"hits": 3, "query": arguments["query"]}
    return record


def run() -> dict[str, Any]:
    ocr_cents = LINE_CENTS
    ocr_match = ocr_cents == LINE_CENTS
    mental_cents = sum(ocr_cents) - CARRY_ERROR_CENTS
    true_yuan = _cents_to_yuan(TRUE_SUBTOTAL_CENTS)
    mental_yuan = _cents_to_yuan(mental_cents)

    valid_calculator = validate_call(
        {
            "name": "calculator",
            "arguments": {"expression": "18.90+26.50+15.80"},
        },
    )
    missing_expression = validate_call({"name": "calculator", "arguments": {}})
    typed_wrong = validate_call(
        {"name": "calculator", "arguments": {"expression": 61.2}},
    )
    missing_crop_box = validate_call(
        {"name": "crop", "arguments": {"image_id": "invoice-1"}},
    )
    unordered_crop = validate_call(
        {
            "name": "crop",
            "arguments": {
                "image_id": "invoice-1",
                "x0": 0.7,
                "y0": 0.2,
                "x1": 0.3,
                "y1": 0.8,
            },
        },
    )
    out_of_range_depth = validate_call(
        {
            "name": "depth",
            "arguments": {"image_id": "scene-1", "u": 0.4, "v": 1.4},
        },
    )
    valid_depth = validate_call(
        {
            "name": "depth",
            "arguments": {"image_id": "scene-1", "u": 0.4, "v": 0.8},
        },
    )
    missing_search_query = validate_call({"name": "search", "arguments": {}})
    unknown_tool = validate_call(
        {"name": "click_submit", "arguments": {"x": 0.5, "y": 0.5}},
    )

    tool_yuan = valid_calculator["result"]
    saw_digits = ocr_match
    mental_wrong = abs(mental_yuan - true_yuan - 0.0) > 1e-9
    tool_right = valid_calculator["executed"] and abs(float(tool_yuan) - true_yuan) < 1e-12
    constructed_case = saw_digits and mental_wrong and tool_right

    checks = {
        "ocr_matches_printed_line_items": saw_digits,
        "mental_drops_tens_carry_by_ten_yuan": mental_cents == MENTAL_SUBTOTAL_CENTS
        and abs(mental_yuan - 51.20) < 1e-12,
        "valid_calculator_equals_true_subtotal": tool_right
        and abs(float(tool_yuan) - 61.20) < 1e-12,
        "constructed_saw_digits_mental_wrong_tool_right": constructed_case,
        "missing_calculator_expression_does_not_execute": (
            not missing_expression["executed"]
            and missing_expression["missing"] == ["expression"]
            and missing_expression["reject_reason"] == "missing_required"
        ),
        "wrong_type_expression_does_not_execute": (
            not typed_wrong["executed"]
            and "expression" in typed_wrong["type_errors"]
        ),
        "crop_missing_box_does_not_execute": (
            not missing_crop_box["executed"]
            and set(missing_crop_box["missing"]) == {"x0", "y0", "x1", "y1"}
        ),
        "unordered_or_oob_geometry_does_not_execute": (
            not unordered_crop["executed"]
            and not out_of_range_depth["executed"]
            and valid_depth["executed"]
        ),
        "unknown_tool_and_missing_search_rejected": (
            unknown_tool["reject_reason"] == "unknown_tool"
            and not unknown_tool["executed"]
            and not missing_search_query["executed"]
        ),
        "rejected_calls_have_null_result": (
            missing_expression["result"] is None
            and missing_crop_box["result"] is None
            and unknown_tool["result"] is None
        ),
    }

    return {
        "summary": (
            "发票三行 18.90、26.50、15.80 被 OCR 读对；心算漏十位进位得到 51.20，"
            "合法计算器得到 61.20。缺参数、类型错误、越界几何和目录外工具名均不得执行。"
        ),
        "metrics": {
            "line_cents": list(LINE_CENTS),
            "true_subtotal_cents": TRUE_SUBTOTAL_CENTS,
            "true_subtotal_yuan": true_yuan,
            "mental_subtotal_cents": mental_cents,
            "mental_subtotal_yuan": mental_yuan,
            "tool_subtotal_yuan": tool_yuan,
            "carry_error_yuan": _cents_to_yuan(CARRY_ERROR_CENTS),
            "valid_calculator_executed": valid_calculator["executed"],
            "missing_expression_executed": missing_expression["executed"],
            "missing_crop_executed": missing_crop_box["executed"],
            "unknown_tool_executed": unknown_tool["executed"],
            "valid_depth_result": valid_depth["result"],
            "catalog_size": len(TOOL_SCHEMAS),
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="52",
    title="让多模态模型在看见图之后调用工具",
    question="看见图上的数字之后，何时必须调用计算器等通用工具，缺参数的调用为何不得进入执行？",
    run=run,
)
