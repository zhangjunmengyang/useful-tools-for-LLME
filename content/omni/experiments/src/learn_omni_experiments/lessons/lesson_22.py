from __future__ import annotations

from typing import Any

from ..core import LessonExperiment


Vector = list[float]
Matrix = list[Vector]

VISION_DIM = 8
LLM_DIM = 12
LORA_RANK = 2
LORA_ALPHA = 4.0
LORA_SCALE = LORA_ALPHA / LORA_RANK


def _zeros(rows: int, cols: int) -> Matrix:
    return [[0.0] * cols for _ in range(rows)]


def _shape(matrix: Matrix) -> tuple[int, int]:
    return len(matrix), len(matrix[0])


def _numel(matrix: Matrix) -> int:
    rows, cols = _shape(matrix)
    return rows * cols


def _matvec(matrix: Matrix, vector: Vector) -> Vector:
    return [
        sum(weight * value for weight, value in zip(row, vector))
        for row in matrix
    ]


def _transpose(matrix: Matrix) -> Matrix:
    return [list(column) for column in zip(*matrix)]


def _outer(left: Vector, right: Vector) -> Matrix:
    return [[left_value * right_value for right_value in right] for left_value in left]


def _add(left: Matrix, right: Matrix) -> Matrix:
    return [
        [left_value + right_value for left_value, right_value in zip(left_row, right_row)]
        for left_row, right_row in zip(left, right)
    ]


def _scale(matrix: Matrix, factor: float) -> Matrix:
    return [[factor * value for value in row] for row in matrix]


def _abs_max(matrix: Matrix) -> float:
    return max(abs(value) for row in matrix for value in row)


def _lora_params(rank: int, in_dim: int, out_dim: int) -> int:
    return rank * (in_dim + out_dim)


def _forward(
    vision: Vector,
    vit_w1: Matrix,
    vit_w2: Matrix,
    projector: Matrix,
    llm_w1: Matrix,
    llm_w2: Matrix,
    lora_a: Matrix,
    lora_b: Matrix,
) -> dict[str, Vector]:
    hidden1 = _matvec(vit_w1, vision)
    hidden2 = _matvec(vit_w2, hidden1)
    projected = _matvec(projector, hidden2)
    adapter = _matvec(lora_b, _matvec(lora_a, projected))
    llm_hidden = [
        base + LORA_SCALE * delta
        for base, delta in zip(_matvec(llm_w1, projected), adapter)
    ]
    logits = _matvec(llm_w2, llm_hidden)
    return {
        "hidden1": hidden1,
        "hidden2": hidden2,
        "projected": projected,
        "adapter_mid": _matvec(lora_a, projected),
        "llm_hidden": llm_hidden,
        "logits": logits,
    }


def _loss(logits: Vector, target: Vector) -> float:
    return 0.5 * sum((value - expected) ** 2 for value, expected in zip(logits, target))


def _backward(
    vision: Vector,
    activations: dict[str, Vector],
    target: Vector,
    vit_w1: Matrix,
    vit_w2: Matrix,
    projector: Matrix,
    llm_w1: Matrix,
    llm_w2: Matrix,
    lora_a: Matrix,
    lora_b: Matrix,
) -> dict[str, Matrix]:
    logits = activations["logits"]
    llm_hidden = activations["llm_hidden"]
    projected = activations["projected"]
    hidden2 = activations["hidden2"]
    hidden1 = activations["hidden1"]
    adapter_mid = activations["adapter_mid"]

    grad_logits = [value - expected for value, expected in zip(logits, target)]
    grad_llm_w2 = _outer(grad_logits, llm_hidden)
    grad_llm_hidden = _matvec(_transpose(llm_w2), grad_logits)

    grad_llm_w1 = _outer(grad_llm_hidden, projected)
    grad_from_base = _matvec(_transpose(llm_w1), grad_llm_hidden)

    grad_adapter = [LORA_SCALE * value for value in grad_llm_hidden]
    grad_lora_b = _outer(grad_adapter, adapter_mid)
    grad_adapter_mid = _matvec(_transpose(lora_b), grad_adapter)
    grad_lora_a = _outer(grad_adapter_mid, projected)
    grad_from_adapter = _matvec(_transpose(lora_a), grad_adapter_mid)

    grad_projected = [
        left + right for left, right in zip(grad_from_base, grad_from_adapter)
    ]
    grad_projector = _outer(grad_projected, hidden2)
    grad_hidden2 = _matvec(_transpose(projector), grad_projected)
    grad_vit_w2 = _outer(grad_hidden2, hidden1)
    grad_hidden1 = _matvec(_transpose(vit_w2), grad_hidden2)
    grad_vit_w1 = _outer(grad_hidden1, vision)

    return {
        "vit_w1": grad_vit_w1,
        "vit_w2": grad_vit_w2,
        "projector": grad_projector,
        "llm_w1": grad_llm_w1,
        "llm_w2": grad_llm_w2,
        "lora_a": grad_lora_a,
        "lora_b": grad_lora_b,
    }


def _mask(
    grads: dict[str, Matrix],
    train_vit: bool,
    train_projector: bool,
    train_llm: bool,
    train_lora: bool,
) -> dict[str, Matrix]:
    masked = dict(grads)
    if not train_vit:
        masked["vit_w1"] = _zeros(*_shape(grads["vit_w1"]))
        masked["vit_w2"] = _zeros(*_shape(grads["vit_w2"]))
    if not train_projector:
        masked["projector"] = _zeros(*_shape(grads["projector"]))
    if not train_llm:
        masked["llm_w1"] = _zeros(*_shape(grads["llm_w1"]))
        masked["llm_w2"] = _zeros(*_shape(grads["llm_w2"]))
    if not train_lora:
        masked["lora_a"] = _zeros(*_shape(grads["lora_a"]))
        masked["lora_b"] = _zeros(*_shape(grads["lora_b"]))
    return masked


def _trainable_count(
    modules: dict[str, Matrix],
    train_vit: bool,
    train_projector: bool,
    train_llm: bool,
    train_lora: bool,
) -> int:
    total = 0
    if train_vit:
        total += _numel(modules["vit_w1"]) + _numel(modules["vit_w2"])
    if train_projector:
        total += _numel(modules["projector"])
    if train_llm:
        total += _numel(modules["llm_w1"]) + _numel(modules["llm_w2"])
    if train_lora:
        total += _numel(modules["lora_a"]) + _numel(modules["lora_b"])
    return total


def run() -> dict[str, Any]:
    vision = [1.0, -0.5, 0.25, 0.0, 0.75, -0.25, 0.5, 0.125]
    target = [0.5, -0.25, 0.75, 0.0, -0.5, 0.25, 0.125, -0.75, 0.4, -0.1, 0.2, 0.3]

    vit_w1 = [
        [0.2, -0.1, 0.0, 0.1, 0.05, 0.0, -0.05, 0.15],
        [0.0, 0.25, -0.2, 0.05, 0.1, -0.05, 0.0, 0.1],
        [0.1, 0.0, 0.3, -0.1, 0.0, 0.2, -0.15, 0.05],
        [-0.05, 0.1, 0.0, 0.2, -0.1, 0.05, 0.15, 0.0],
        [0.15, -0.05, 0.1, 0.0, 0.25, -0.1, 0.0, 0.05],
        [0.0, 0.1, -0.05, 0.15, 0.0, 0.2, -0.1, 0.05],
        [0.05, 0.0, 0.1, -0.15, 0.05, 0.0, 0.3, -0.1],
        [-0.1, 0.05, 0.0, 0.1, -0.05, 0.15, 0.0, 0.2],
    ]
    vit_w2 = [
        [0.3, 0.0, -0.1, 0.05, 0.1, 0.0, -0.05, 0.1],
        [0.0, 0.2, 0.1, -0.05, 0.0, 0.15, 0.05, -0.1],
        [0.1, -0.1, 0.25, 0.0, 0.05, -0.05, 0.1, 0.0],
        [0.05, 0.15, 0.0, 0.2, -0.1, 0.0, 0.05, 0.1],
        [-0.05, 0.0, 0.1, 0.05, 0.3, -0.1, 0.0, 0.05],
        [0.1, 0.05, -0.05, 0.0, 0.1, 0.2, -0.15, 0.0],
        [0.0, -0.1, 0.15, 0.05, 0.0, 0.1, 0.25, -0.05],
        [0.05, 0.1, 0.0, -0.1, 0.05, 0.0, 0.1, 0.2],
    ]
    projector = [
        [0.4, -0.1, 0.05, 0.0, 0.1, -0.05, 0.2, 0.0],
        [0.0, 0.3, -0.1, 0.15, 0.0, 0.1, -0.05, 0.05],
        [0.1, 0.0, 0.25, -0.05, 0.1, 0.0, 0.05, -0.1],
        [-0.05, 0.1, 0.0, 0.2, -0.1, 0.05, 0.0, 0.15],
        [0.2, -0.05, 0.1, 0.0, 0.3, -0.1, 0.05, 0.0],
        [0.0, 0.15, -0.05, 0.1, 0.0, 0.25, -0.1, 0.05],
        [0.05, 0.0, 0.1, -0.15, 0.05, 0.0, 0.2, 0.1],
        [-0.1, 0.05, 0.0, 0.1, -0.05, 0.15, 0.0, 0.3],
        [0.15, 0.0, 0.05, -0.1, 0.1, 0.0, 0.05, -0.05],
        [0.0, 0.2, -0.15, 0.05, 0.0, 0.1, -0.05, 0.1],
        [0.1, -0.1, 0.0, 0.15, 0.05, -0.05, 0.2, 0.0],
        [0.05, 0.05, 0.1, 0.0, -0.1, 0.15, 0.0, 0.2],
    ]
    llm_w1 = [
        [0.2, 0.0, -0.1, 0.05, 0.1, 0.0, 0.05, -0.05, 0.1, 0.0, 0.05, 0.0],
        [0.0, 0.25, 0.05, -0.1, 0.0, 0.1, -0.05, 0.05, 0.0, 0.1, -0.05, 0.05],
        [0.1, -0.05, 0.3, 0.0, 0.05, -0.1, 0.0, 0.1, 0.05, 0.0, 0.1, -0.05],
        [-0.05, 0.1, 0.0, 0.2, -0.1, 0.05, 0.1, 0.0, -0.05, 0.05, 0.0, 0.1],
        [0.05, 0.0, 0.1, -0.05, 0.25, 0.0, -0.1, 0.05, 0.1, 0.0, 0.05, -0.05],
        [0.0, 0.15, -0.05, 0.1, 0.0, 0.2, 0.05, -0.1, 0.0, 0.05, -0.05, 0.1],
        [0.1, -0.1, 0.05, 0.0, 0.1, -0.05, 0.3, 0.0, 0.05, 0.1, 0.0, -0.05],
        [-0.1, 0.05, 0.0, 0.1, -0.05, 0.15, 0.0, 0.2, -0.1, 0.0, 0.05, 0.05],
        [0.05, 0.0, 0.15, -0.1, 0.0, 0.05, 0.1, -0.05, 0.25, 0.0, -0.1, 0.05],
        [0.0, 0.1, -0.1, 0.05, 0.15, 0.0, -0.05, 0.1, 0.0, 0.2, 0.05, -0.05],
        [0.1, -0.05, 0.0, 0.1, 0.0, -0.1, 0.05, 0.0, 0.1, -0.05, 0.3, 0.0],
        [0.05, 0.05, 0.1, 0.0, -0.1, 0.1, 0.0, 0.05, -0.05, 0.15, 0.0, 0.2],
    ]
    llm_w2 = [list(row) for row in llm_w1]
    lora_a = [
        [0.15, -0.05, 0.1, 0.0, 0.05, -0.1, 0.0, 0.05, 0.1, -0.05, 0.0, 0.05],
        [0.0, 0.1, -0.05, 0.15, 0.0, 0.05, -0.1, 0.0, 0.05, 0.1, -0.05, 0.1],
    ]
    lora_b = [
        [0.2, -0.1],
        [0.0, 0.15],
        [0.1, 0.05],
        [-0.05, 0.2],
        [0.15, 0.0],
        [0.05, -0.15],
        [0.0, 0.1],
        [0.1, -0.05],
        [-0.1, 0.2],
        [0.05, 0.05],
        [0.2, -0.1],
        [0.0, 0.15],
    ]

    modules = {
        "vit_w1": vit_w1,
        "vit_w2": vit_w2,
        "projector": projector,
        "llm_w1": llm_w1,
        "llm_w2": llm_w2,
        "lora_a": lora_a,
        "lora_b": lora_b,
    }
    activations = _forward(
        vision,
        vit_w1,
        vit_w2,
        projector,
        llm_w1,
        llm_w2,
        lora_a,
        lora_b,
    )
    raw_grads = _backward(
        vision,
        activations,
        target,
        vit_w1,
        vit_w2,
        projector,
        llm_w1,
        llm_w2,
        lora_a,
        lora_b,
    )
    stage1_grads = _mask(
        raw_grads,
        train_vit=False,
        train_projector=True,
        train_llm=False,
        train_lora=False,
    )
    stage2_grads = _mask(
        raw_grads,
        train_vit=False,
        train_projector=True,
        train_llm=False,
        train_lora=True,
    )
    stage3_grads = _mask(
        raw_grads,
        train_vit=True,
        train_projector=True,
        train_llm=False,
        train_lora=True,
    )

    projector_only = _trainable_count(
        modules,
        train_vit=False,
        train_projector=True,
        train_llm=False,
        train_lora=False,
    )
    projector_and_lora = _trainable_count(
        modules,
        train_vit=False,
        train_projector=True,
        train_llm=False,
        train_lora=True,
    )
    projector_lora_vit = _trainable_count(
        modules,
        train_vit=True,
        train_projector=True,
        train_llm=False,
        train_lora=True,
    )
    full_unfreeze = _trainable_count(
        modules,
        train_vit=True,
        train_projector=True,
        train_llm=True,
        train_lora=True,
    )

    vit_params = _numel(vit_w1) + _numel(vit_w2)
    llm_params = _numel(llm_w1) + _numel(llm_w2)
    projector_params = _numel(projector)
    lora_params = _numel(lora_a) + _numel(lora_b)
    formula_lora = _lora_params(LORA_RANK, LLM_DIM, LLM_DIM)
    formula_stage1 = VISION_DIM * LLM_DIM
    formula_stage2 = formula_stage1 + formula_lora
    formula_stage3 = formula_stage2 + vit_params

    checks = {
        "stage1_trainable_equals_projector": projector_only == projector_params == formula_stage1,
        "stage1_vit_and_llm_grads_are_masked": (
            _abs_max(stage1_grads["vit_w1"]) == 0.0
            and _abs_max(stage1_grads["vit_w2"]) == 0.0
            and _abs_max(stage1_grads["llm_w1"]) == 0.0
            and _abs_max(stage1_grads["llm_w2"]) == 0.0
            and _abs_max(stage1_grads["lora_a"]) == 0.0
            and _abs_max(stage1_grads["lora_b"]) == 0.0
        ),
        "stage1_projector_grad_is_nonzero": _abs_max(stage1_grads["projector"]) > 1e-8,
        "unmasked_vit_and_llm_grads_are_nonzero": (
            _abs_max(raw_grads["vit_w1"]) > 1e-8
            and _abs_max(raw_grads["llm_w1"]) > 1e-8
        ),
        "lora_parameter_formula_matches_matrices": (
            lora_params == formula_lora == 2 * LORA_RANK * LLM_DIM
        ),
        "stage2_counts_projector_plus_lora": (
            projector_and_lora == projector_params + lora_params == formula_stage2
        ),
        "stage2_base_llm_grads_remain_masked": (
            _abs_max(stage2_grads["llm_w1"]) == 0.0
            and _abs_max(stage2_grads["llm_w2"]) == 0.0
            and _abs_max(stage2_grads["lora_a"]) > 1e-8
        ),
        "stage3_unfreezes_vit_only_on_top_of_stage2": (
            projector_lora_vit == formula_stage3
            and _abs_max(stage3_grads["vit_w1"]) > 1e-8
            and _abs_max(stage3_grads["llm_w1"]) == 0.0
        ),
    }

    return {
        "summary": (
            "在手写小矩阵上实现 H=WZ 的三阶段冻结协议：只训投影时 ViT 与 LLM "
            "梯度被 mask 为零；LoRA 阶段只打开低秩增量；解冻 ViT 后视觉塔梯度恢复。"
            "参数量按模块计数，并与 r(d_in+d_out) 公式核对。"
        ),
        "metrics": {
            "vision_dim": VISION_DIM,
            "llm_dim": LLM_DIM,
            "lora_rank": LORA_RANK,
            "lora_scale": LORA_SCALE,
            "vit_parameters": vit_params,
            "projector_parameters": projector_params,
            "llm_parameters": llm_params,
            "lora_parameters": lora_params,
            "stage1_trainable": projector_only,
            "stage2_trainable": projector_and_lora,
            "stage3_trainable": projector_lora_vit,
            "full_unfreeze_trainable": full_unfreeze,
            "stage1_projector_grad_abs_max": round(_abs_max(stage1_grads["projector"]), 8),
            "stage1_vit_grad_abs_max": round(
                max(_abs_max(stage1_grads["vit_w1"]), _abs_max(stage1_grads["vit_w2"])),
                8,
            ),
            "stage1_llm_grad_abs_max": round(
                max(_abs_max(stage1_grads["llm_w1"]), _abs_max(stage1_grads["llm_w2"])),
                8,
            ),
            "unmasked_vit_grad_abs_max": round(_abs_max(raw_grads["vit_w1"]), 8),
            "unmasked_llm_grad_abs_max": round(_abs_max(raw_grads["llm_w1"]), 8),
            "loss": round(_loss(activations["logits"], target), 8),
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="22",
    title="复现视觉语言模型的标准训练配方",
    question="只训投影时，LLM 与 ViT 的梯度是否被 mask 掉？三阶段各有多少可训练参数？",
    run=run,
)
