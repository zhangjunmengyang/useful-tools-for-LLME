from __future__ import annotations

from typing import Any

from ..core import LessonExperiment


IOU_THRESHOLD = 0.5

# Teaching invoice in a 100 x 140 canvas. Boxes are (x1, y1, x2, y2).
AMOUNT_HEADER = (70.0, 30.0, 95.0, 42.0)
LINE1_AMOUNT = (70.0, 44.0, 95.0, 56.0)
LINE2_AMOUNT = (70.0, 58.0, 95.0, 70.0)
TOTAL_CELL = (70.0, 80.0, 95.0, 96.0)
INVOICE_NO = (78.0, 6.0, 96.0, 18.0)
PROMO = (8.0, 108.0, 48.0, 120.0)
CONTRACT_PAGE2 = (12.0, 20.0, 70.0, 36.0)

TOTAL_TEXT = "32.00"
HEADER_TEXT = "金额"
INVOICE_NO_TEXT = "128"
CONTRACT_TEXT = "HT-2024-09"


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


def content_hit(pred: str, gt: str) -> bool:
    return pred == gt


def box_hit(
    pred: tuple[float, float, float, float],
    gt: tuple[float, float, float, float],
    threshold: float = IOU_THRESHOLD,
) -> bool:
    return iou(pred, gt) >= threshold


def layout_hit(
    pred_text: str,
    gt_text: str,
    pred_box: tuple[float, float, float, float],
    gt_box: tuple[float, float, float, float],
    threshold: float = IOU_THRESHOLD,
) -> bool:
    """A cell is a layout hit only when both content and box pass."""
    return content_hit(pred_text, gt_text) and box_hit(pred_box, gt_box, threshold)


def field_f1(pred: dict[str, str], gt: dict[str, str]) -> float:
    """Field-level F1: a field counts only if the string is an exact match."""
    if not pred and not gt:
        return 1.0
    matched = sum(1 for key, value in pred.items() if gt.get(key) == value)
    if not pred or not gt:
        return 0.0
    precision = matched / len(pred)
    recall = matched / len(gt)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def node_count(tree: Any) -> int:
    if isinstance(tree, dict):
        return 1 + sum(node_count(value) for value in tree.values())
    if isinstance(tree, list):
        return 1 + sum(node_count(item) for item in tree)
    return 1


def tree_edit_distance(pred: Any, gt: Any) -> int:
    """Unordered labeled TED on dict / list / scalar trees."""
    if type(pred) is not type(gt):
        return node_count(pred) + node_count(gt)
    if isinstance(pred, dict):
        keys = set(pred) | set(gt)
        cost = 0
        if set(pred) != set(gt):
            cost += 1
        for key in keys:
            if key not in pred:
                cost += node_count(gt[key])
            elif key not in gt:
                cost += node_count(pred[key])
            else:
                cost += tree_edit_distance(pred[key], gt[key])
        return cost
    if isinstance(pred, list):
        length = max(len(pred), len(gt))
        cost = abs(len(pred) - len(gt))
        for index in range(length):
            if index >= len(pred):
                cost += node_count(gt[index])
            elif index >= len(gt):
                cost += node_count(pred[index])
            else:
                cost += tree_edit_distance(pred[index], gt[index])
        return cost
    return 0 if pred == gt else 1


def ted_accuracy(pred: Any, gt: Any) -> float:
    empty_cost = node_count(gt)
    if empty_cost <= 0:
        return 1.0
    return max(0.0, 1.0 - tree_edit_distance(pred, gt) / empty_cost)


def raster_order(items: list[tuple[str, tuple[float, float, float, float]]]) -> list[str]:
    """Top-left to bottom-right by (y1, x1)."""
    ranked = sorted(items, key=lambda item: (item[1][1], item[1][0], item[0]))
    return [text for text, _box in ranked]


def inversion_count(actual: list[str], expected: list[str]) -> int:
    rank = {token: index for index, token in enumerate(expected)}
    indices = [rank[token] for token in actual if token in rank]
    inversions = 0
    for left in range(len(indices)):
        for right in range(left + 1, len(indices)):
            if indices[left] > indices[right]:
                inversions += 1
    return inversions


def cross_page_lookup(
    pages: dict[int, dict[str, str]],
    key: str,
    visible_pages: set[int],
) -> str | None:
    for page_id, fields in pages.items():
        if page_id in visible_pages and key in fields:
            return fields[key]
    return None


def run() -> dict[str, Any]:
    header_vs_total = iou(AMOUNT_HEADER, TOTAL_CELL)
    identical_iou = iou(TOTAL_CELL, TOTAL_CELL)
    disjoint_line_vs_total = iou(LINE1_AMOUNT, TOTAL_CELL)

    ocr_pred_text = TOTAL_TEXT
    ocr_pred_box = AMOUNT_HEADER
    ocr_content = content_hit(ocr_pred_text, TOTAL_TEXT)
    ocr_box = box_hit(ocr_pred_box, TOTAL_CELL)
    ocr_layout = layout_hit(ocr_pred_text, TOTAL_TEXT, ocr_pred_box, TOTAL_CELL)

    wrong_text_right_box = layout_hit(INVOICE_NO_TEXT, TOTAL_TEXT, TOTAL_CELL, TOTAL_CELL)
    both_right = layout_hit(TOTAL_TEXT, TOTAL_TEXT, TOTAL_CELL, TOTAL_CELL)
    line_as_total = layout_hit("24.00", TOTAL_TEXT, LINE1_AMOUNT, TOTAL_CELL)

    half_pred = (0.0, 0.0, 2.0, 1.0)
    half_gt = (2.0 / 3.0, 0.0, 8.0 / 3.0, 1.0)
    boundary_iou = iou(half_pred, half_gt)
    boundary_hit = box_hit(half_pred, half_gt)
    below_half = iou((0.0, 0.0, 2.0, 1.0), (1.2, 0.0, 3.2, 1.0))

    gt_fields = {
        "total": TOTAL_TEXT,
        "date": "2024-01-02",
        "vendor": "Acme",
    }
    pred_fields_char_miss = {
        "total": TOTAL_TEXT,
        "date": "2024-01-0",
        "vendor": "Acme",
    }
    pred_fields_ok = dict(gt_fields)
    f1_char_miss = field_f1(pred_fields_char_miss, gt_fields)
    f1_ok = field_f1(pred_fields_ok, gt_fields)
    f1_empty_pred = field_f1({}, gt_fields)

    gt_tree = {
        "menu": [
            {"name": "纸", "price": "24"},
            {"name": "订", "price": "8"},
        ],
        "total": "32",
    }
    pred_tree_ok = {
        "menu": [
            {"name": "纸", "price": "24"},
            {"name": "订", "price": "8"},
        ],
        "total": "32",
    }
    pred_tree_flat = {"total": "32", "name": "纸", "price": "24"}
    ted_ok = tree_edit_distance(pred_tree_ok, gt_tree)
    ted_flat = tree_edit_distance(pred_tree_flat, gt_tree)
    ted_empty = tree_edit_distance({}, gt_tree)
    acc_ok = ted_accuracy(pred_tree_ok, gt_tree)
    acc_flat = ted_accuracy(pred_tree_flat, gt_tree)
    acc_empty = ted_accuracy({}, gt_tree)

    two_column = [
        ("条款一", (8.0, 10.0, 40.0, 22.0)),
        ("32.00", (68.0, 10.0, 92.0, 22.0)),
        ("条款二", (8.0, 40.0, 40.0, 52.0)),
    ]
    reading = ["条款一", "条款二", "32.00"]
    raster = raster_order(two_column)
    raster_inversions = inversion_count(raster, reading)

    pages = {
        1: {"发票号": INVOICE_NO_TEXT},
        2: {"合同编号": CONTRACT_TEXT},
    }
    page1_only = cross_page_lookup(pages, "合同编号", {1})
    both_pages = cross_page_lookup(pages, "合同编号", {1, 2})
    page1_invoice = cross_page_lookup(pages, "发票号", {1})

    gt_count = node_count(gt_tree)

    checks = {
        "header_and_total_are_disjoint": header_vs_total == 0.0,
        "identical_total_boxes_have_iou_one": identical_iou == 1.0,
        "line_item_and_total_are_disjoint": disjoint_line_vs_total == 0.0,
        "content_can_hit_while_box_misses": ocr_content and (not ocr_box),
        "layout_fails_when_box_is_on_header": (not ocr_layout) and ocr_content,
        "layout_fails_when_content_is_wrong": not wrong_text_right_box,
        "layout_passes_only_when_both_hit": both_right,
        "line_item_is_not_the_total_cell": not line_as_total,
        "iou_half_is_inclusive_hit": boundary_hit and abs(boundary_iou - 0.5) < 1e-12,
        "iou_below_half_is_miss": below_half < IOU_THRESHOLD,
        "field_f1_is_one_on_exact_fields": abs(f1_ok - 1.0) < 1e-12,
        "single_char_miss_drops_field_f1": abs(f1_char_miss - (2.0 / 3.0)) < 1e-12,
        "empty_prediction_has_zero_field_f1": f1_empty_pred == 0.0,
        "identical_trees_have_ted_zero": ted_ok == 0,
        "ted_accuracy_is_one_on_identical_trees": abs(acc_ok - 1.0) < 1e-12,
        "flattening_structure_lowers_ted_accuracy": acc_flat < acc_ok and ted_flat > 0,
        "empty_tree_ted_equals_node_count": ted_empty == gt_count,
        "empty_tree_ted_accuracy_is_zero": abs(acc_empty - 0.0) < 1e-12,
        "raster_order_is_not_reading_order": raster != reading,
        "raster_inserts_amount_between_clauses": raster == ["条款一", "32.00", "条款二"],
        "raster_has_one_inversion": raster_inversions == 1,
        "page1_only_misses_contract_id": page1_only is None,
        "both_pages_recover_contract_id": both_pages == CONTRACT_TEXT,
        "page1_still_reads_invoice_number": page1_invoice == INVOICE_NO_TEXT,
        "layout_is_conjunction_not_content_or_box": (
            ocr_content
            and box_hit(TOTAL_CELL, TOTAL_CELL)
            and (not layout_hit(ocr_pred_text, TOTAL_TEXT, ocr_pred_box, TOTAL_CELL))
        ),
    }

    return {
        "summary": (
            "用固定发票夹具核验单元格命中 = 内容对且框 IoU 过阈值；"
            "读对 32.00 但框落在表头「金额」时版面失败，内容命中与框命中可以脱钩。"
        ),
        "metrics": {
            "header_vs_total_iou": header_vs_total,
            "ocr_content_hit": float(ocr_content),
            "ocr_box_hit": float(ocr_box),
            "ocr_layout_hit": float(ocr_layout),
            "wrong_text_layout_hit": float(wrong_text_right_box),
            "both_right_layout_hit": float(both_right),
            "boundary_iou": boundary_iou,
            "field_f1_char_miss": f1_char_miss,
            "field_f1_ok": f1_ok,
            "ted_ok": ted_ok,
            "ted_flat": ted_flat,
            "ted_accuracy_ok": acc_ok,
            "ted_accuracy_flat": acc_flat,
            "gt_node_count": gt_count,
            "raster_inversions": raster_inversions,
            "iou_threshold": IOU_THRESHOLD,
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="44",
    title="把文档版面从单图 OCR 里拆出来",
    question="版面和表格为什么不是单图 OCR？",
    run=run,
)
