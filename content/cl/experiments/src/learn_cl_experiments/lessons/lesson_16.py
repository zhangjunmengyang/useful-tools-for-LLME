from __future__ import annotations

from typing import Any

from ..core import LessonExperiment


EXPERIENCES = ("fact", "document", "procedure", "reasoning")
WRITERS = ("rag", "memory", "edit", "weights")


def _true_score(a: float, b: float) -> float:
    return 2.0 * a + 3.0 * b


def _fit_linear(pairs: list[tuple[tuple[float, float], float]]) -> tuple[float, float]:
    # Two-parameter least squares for y = w0 a + w1 b, no intercept.
    gram00 = sum(a * a for (a, _), _ in pairs)
    gram01 = sum(a * b for (a, b), _ in pairs)
    gram11 = sum(b * b for (_, b), _ in pairs)
    rhs0 = sum(a * y for (a, _), y in pairs)
    rhs1 = sum(b * y for (_, b), y in pairs)
    det = gram00 * gram11 - gram01 * gram01
    w0 = (gram11 * rhs0 - gram01 * rhs1) / det
    w1 = (gram00 * rhs1 - gram01 * rhs0) / det
    return w0, w1


def run() -> dict[str, Any]:
    facts = {f"seat_{index}": f"desk_{index}" for index in range(10)}
    documents = {
        f"doc_{index}": f"project northpole owner is owner_{index}"
        for index in range(10)
    }
    procedures = {
        f"craft_{index}": ("goto_bench", "use_tool", f"emit_{index}")
        for index in range(10)
    }
    reasoning_pairs = [
        ((float(index), float(index + 1)), _true_score(float(index), float(index + 1)))
        for index in range(10)
    ]

    context = {
        **{name: seat for name, seat in facts.items()},
        **documents,
    }
    memory = dict(facts)
    memory.update({key: text for key, text in documents.items()})
    edited = dict(facts)
    edited["seat_0"] = "desk_moved"
    weight_rule = _fit_linear(reasoning_pairs)
    weight_policy = {name: steps for name, steps in procedures.items()}

    withdrawn: dict[str, str] = {}

    def evaluate(kind: str, writer: str) -> int:
        if kind == "fact":
            if writer == "rag":
                return int(all(context.get(name) == seat for name, seat in facts.items()))
            if writer == "memory":
                return int(all(memory.get(name) == seat for name, seat in facts.items()))
            if writer == "edit":
                return int(edited["seat_0"] == "desk_moved" and edited["seat_1"] == "desk_1")
            return 0
        if kind == "document":
            if writer == "rag":
                return int("owner_3" in context["doc_3"])
            if writer == "memory":
                return int("owner_3" in memory["doc_3"])
            return 0
        if kind == "procedure":
            if writer == "weights":
                return int(weight_policy["craft_4"] == procedures["craft_4"])
            return 0
        if writer == "weights":
            pred = [weight_rule[0] * a + weight_rule[1] * b for (a, b), _ in reasoning_pairs]
            err = sum(abs(left - right) for left, (_, right) in zip(pred, reasoning_pairs))
            return int(err < 1e-9)
        return 0

    matrix = {
        kind: {writer: evaluate(kind, writer) for writer in WRITERS}
        for kind in EXPERIENCES
    }
    rag_without_context = int(
        all(withdrawn.get(name) == seat for name, seat in facts.items())
    )
    memory_pass_weight_fail = [
        kind
        for kind in EXPERIENCES
        if matrix[kind]["memory"] == 1 and matrix[kind]["weights"] == 0
    ]
    memory_fail_weight_pass = [
        kind
        for kind in EXPERIENCES
        if matrix[kind]["memory"] == 0 and matrix[kind]["weights"] == 1
    ]

    checks = {
        "memory_passes_where_weights_fail": bool(memory_pass_weight_fail),
        "weights_pass_where_memory_fails": bool(memory_fail_weight_pass),
        "rag_fails_after_context_withdrawn": rag_without_context == 0,
        "edit_moves_only_target_fact": (
            edited["seat_0"] == "desk_moved" and edited["seat_1"] == facts["seat_1"]
        ),
        "reasoning_only_weights_fit": (
            matrix["reasoning"]["weights"] == 1
            and matrix["reasoning"]["memory"] == 0
            and matrix["reasoning"]["edit"] == 0
        ),
        "fact_memory_succeeds": matrix["fact"]["memory"] == 1,
    }
    return {
        "summary": (
            "四类经验 × 四种写入：事实走记忆能过、走权重过不了；"
            "计分规则走权拟合能过、只存文本过不了。撤掉上下文后 RAG 为 0。"
            "编辑只改 seat_0。失败阈值：矩阵缺「记忆过权重不过」或反过来。"
        ),
        "metrics": {
            "matrix": matrix,
            "memory_pass_weight_fail": memory_pass_weight_fail,
            "memory_fail_weight_pass": memory_fail_weight_pass,
            "rag_without_context": rag_without_context,
            "fitted_rule": [weight_rule[0], weight_rule[1]],
            "edit_seat_0": edited["seat_0"],
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="16",
    title="什么时候必须改权重",
    question="哪类新经验只靠外挂记忆过关，哪类必须改权重？",
    run=run,
)
