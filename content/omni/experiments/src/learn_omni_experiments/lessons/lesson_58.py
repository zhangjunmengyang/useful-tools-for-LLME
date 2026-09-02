from __future__ import annotations

from typing import Any

from ..core import LessonExperiment


POSITIVE_FINDINGS = ("pneumonia", "effusion", "consolidation")
NEGATION_CUES = ("no", "without", "absent", "clear")

NATURAL_TRAIN_ROLES = frozenset({"caption"})
MEDICAL_TRAIN_ROLES = frozenset({"findings", "impression"})
NAIVE_TRAIN_ROLES = frozenset(
    {
        "user",
        "caption",
        "header",
        "indication",
        "comparison",
        "findings",
        "impression",
    },
)

NATURAL_SEQUENCE: list[tuple[str, str]] = [
    ("<image>", "condition"),
    ("Describe", "user"),
    ("the", "user"),
    ("image", "user"),
    (".", "user"),
    ("A", "caption"),
    ("dog", "caption"),
    ("sits", "caption"),
    ("on", "caption"),
    ("a", "caption"),
    ("sofa", "caption"),
    (".", "caption"),
]

MEDICAL_SEQUENCE: list[tuple[str, str]] = [
    ("<image>", "condition"),
    ("INDICATION:", "header"),
    ("cough", "indication"),
    ("COMPARISON:", "header"),
    ("none", "comparison"),
    ("FINDINGS:", "header"),
    ("lungs", "findings"),
    ("are", "findings"),
    ("clear", "findings"),
    (".", "findings"),
    ("IMPRESSION:", "header"),
    ("no", "impression"),
    ("acute", "impression"),
    ("process", "impression"),
    (".", "impression"),
]


def train_mask(roles: list[str], allowed: frozenset[str]) -> list[int]:
    """1 if the token participates in next-token loss, else 0."""
    return [1 if role in allowed else 0 for role in roles]


def masked_tokens(sequence: list[tuple[str, str]], allowed: frozenset[str]) -> list[str]:
    return [token for token, role in sequence if role in allowed]


def is_positive_sentence(text: str) -> bool:
    tokens = text.lower().replace(".", " ").split()
    has_finding = any(finding in tokens for finding in POSITIVE_FINDINGS)
    has_negation = any(cue in tokens for cue in NEGATION_CUES)
    return has_finding and not has_negation


def unboxed_assertion_count(sentences: list[dict[str, Any]]) -> int:
    """Count positive findings that have no bounding box."""
    total = 0
    for sentence in sentences:
        if is_positive_sentence(str(sentence["text"])) and sentence.get("box") is None:
            total += 1
    return total


def positive_count(sentences: list[dict[str, Any]]) -> int:
    return sum(1 for sentence in sentences if is_positive_sentence(str(sentence["text"])))


def unboxed_rate(sentences: list[dict[str, Any]]) -> float:
    positives = positive_count(sentences)
    if positives == 0:
        return 0.0
    return unboxed_assertion_count(sentences) / positives


def open_recall(predicted: list[str], gold: list[str]) -> float:
    """Share of gold tokens that appear in the generated sequence (LLaVA-Med open-set)."""
    if not gold:
        return 1.0
    predicted_set = set(predicted)
    return sum(1 for token in gold if token in predicted_set) / len(gold)


def closed_accuracy(predicted: list[str], gold: list[str]) -> float:
    if not gold:
        return 1.0
    return sum(int(left == right) for left, right in zip(predicted, gold)) / len(gold)


def run() -> dict[str, Any]:
    natural_roles = [role for _, role in NATURAL_SEQUENCE]
    medical_roles = [role for _, role in MEDICAL_SEQUENCE]

    natural_mask = train_mask(natural_roles, NATURAL_TRAIN_ROLES)
    medical_mask = train_mask(medical_roles, MEDICAL_TRAIN_ROLES)
    naive_on_medical = train_mask(medical_roles, NAIVE_TRAIN_ROLES)

    natural_supervised = masked_tokens(NATURAL_SEQUENCE, NATURAL_TRAIN_ROLES)
    medical_supervised = masked_tokens(MEDICAL_SEQUENCE, MEDICAL_TRAIN_ROLES)
    naive_supervised = masked_tokens(MEDICAL_SEQUENCE, NAIVE_TRAIN_ROLES)

    empty_gate_off = [{"text": "there is pneumonia", "box": None}]
    empty_gate_on = [{"text": "no pneumonia", "box": None}]
    boxed_positive = [{"text": "pneumonia in the right base", "box": (0.58, 0.52, 0.82, 0.78)}]
    mixed_batch = [
        {"text": "there is pneumonia", "box": None},
        {"text": "pneumonia in the right base", "box": (0.58, 0.52, 0.82, 0.78)},
        {"text": "no effusion", "box": None},
        {"text": "lungs are clear", "box": None},
    ]

    unboxed_empty_off = unboxed_assertion_count(empty_gate_off)
    unboxed_empty_on = unboxed_assertion_count(empty_gate_on)
    unboxed_boxed = unboxed_assertion_count(boxed_positive)
    unboxed_mixed = unboxed_assertion_count(mixed_batch)
    positives_mixed = positive_count(mixed_batch)
    rate_mixed = unboxed_rate(mixed_batch)

    # High open recall can still hide an unboxed positive.
    open_pred = ["patchy", "infiltrates", "pneumonia", "wires"]
    open_gold = ["pneumonia"]
    recall_value = open_recall(open_pred, open_gold)
    closed_pred = ["yes", "no", "yes"]
    closed_gold = ["yes", "no", "no"]
    closed_value = closed_accuracy(closed_pred, closed_gold)

    indication_in_medical = "cough" in medical_supervised
    indication_in_naive = "cough" in naive_supervised
    comparison_in_medical = "none" in medical_supervised
    findings_in_medical = "lungs" in medical_supervised
    impression_in_medical = "process" in medical_supervised
    caption_in_natural = "dog" in natural_supervised
    condition_in_natural = "<image>" in natural_supervised
    condition_in_medical = "<image>" in medical_supervised

    checks = {
        "natural_and_medical_masks_differ": natural_supervised != medical_supervised,
        "indication_excluded_from_medical_mask": (
            not indication_in_medical and "cough" in [token for token, _ in MEDICAL_SEQUENCE]
        ),
        "naive_caption_recipe_would_train_indication": indication_in_naive and not indication_in_medical,
        "comparison_excluded_from_medical_mask": not comparison_in_medical,
        "findings_and_impression_are_supervised": findings_in_medical and impression_in_medical,
        "image_tokens_never_enter_loss": (not condition_in_natural) and (not condition_in_medical),
        "empty_image_unboxed_positive_when_gate_off": unboxed_empty_off == 1,
        "empty_image_unboxed_zero_when_gate_on": unboxed_empty_on == 0,
        "boxed_positive_is_not_unboxed": unboxed_boxed == 0,
        "mixed_unboxed_rate_is_half_of_positives": (
            unboxed_mixed == 1 and positives_mixed == 2 and abs(rate_mixed - 0.5) < 1e-12
        ),
        "open_recall_can_be_one_with_unboxed_positive": (
            abs(recall_value - 1.0) < 1e-12 and unboxed_empty_off == 1
        ),
        "closed_accuracy_is_exact_match_not_recall": abs(closed_value - (2 / 3)) < 1e-12,
        "caption_tokens_are_the_natural_supervision": caption_in_natural and "Describe" not in natural_supervised,
        "naive_mask_has_more_tokens_than_medical": len(naive_supervised) > len(medical_supervised),
    }

    return {
        "summary": (
            "医学报告字段的 loss mask 与自然图像 caption 不同：indication / comparison "
            "不得当作由当前图像生成的监督；空图在关闭无框门控时产生 1 条无框肯定，"
            "打开门控后计数归零。开放集 recall 为 1 不能掩盖无框断言。"
        ),
        "metrics": {
            "natural_supervised_tokens": len(natural_supervised),
            "medical_supervised_tokens": len(medical_supervised),
            "naive_supervised_tokens": len(naive_supervised),
            "natural_mask_sum": sum(natural_mask),
            "medical_mask_sum": sum(medical_mask),
            "naive_mask_sum": sum(naive_on_medical),
            "unboxed_empty_gate_off": unboxed_empty_off,
            "unboxed_empty_gate_on": unboxed_empty_on,
            "unboxed_boxed_positive": unboxed_boxed,
            "unboxed_mixed_count": unboxed_mixed,
            "positive_mixed_count": positives_mixed,
            "unboxed_mixed_rate": rate_mixed,
            "open_recall": recall_value,
            "closed_accuracy": closed_value,
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="58",
    title="把医学图文从自然图像配方里拆出来",
    question="空图无框肯定如何计数？医学报告字段的 mask 与自然 caption 差在哪？",
    run=run,
)
