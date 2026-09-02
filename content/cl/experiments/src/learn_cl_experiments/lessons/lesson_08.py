from __future__ import annotations

from typing import Any

import numpy as np

from ..core import LessonExperiment

SEED = 8
EPS = 1e-12


def _num(value: float) -> float:
    return float(round(float(value), 6))


def _project_update(update: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, bool, float]:
    """Project u onto {z | z · n ≤ 0}. n is ∇L_old, the constraint normal."""
    raw_dot = float(np.dot(update, normal))
    if raw_dot <= 0:
        return update, False, raw_dot
    projected = update - (raw_dot / (float(np.dot(normal, normal)) + EPS)) * normal
    return projected, True, raw_dot


def run() -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    normal = np.array([2.0, 0.0], dtype=np.float64)
    violating = np.array([1.0, 1.0], dtype=np.float64)
    allowed = np.array([-1.0, 2.0], dtype=np.float64)

    projected, did_violate, raw_dot = _project_update(violating, normal)
    projected_dot = float(np.dot(projected, normal))
    allowed_proj, allowed_violated, allowed_dot = _project_update(allowed, normal)

    random_normal = rng.normal(size=16)
    random_update = rng.normal(size=16)
    if float(np.dot(random_update, random_normal)) <= 0:
        random_update = -random_update
    random_proj, random_violated, random_raw = _project_update(random_update, random_normal)
    random_proj_dot = float(np.dot(random_proj, random_normal))

    # A-GEM on gradients g = -u, g_ref = n, onto {g | g · g_ref ≥ 0}.
    gradient = -violating
    g_ref = normal
    alignment = float(np.dot(gradient, g_ref))
    if alignment < 0:
        agem = gradient - (alignment / (float(np.dot(g_ref, g_ref)) + EPS)) * g_ref
    else:
        agem = gradient
    agem_dot = float(np.dot(agem, g_ref))
    equivalent = float(np.linalg.norm(projected + agem))

    return {
        "summary": (
            "旧任务约束法向 n=∇L_old。更新 u 若 u·n>0 就会抬高旧损失；"
            "投影到半平面 u·n≤0 之后，二维例子点积 "
            f"{projected_dot:.1e}，16 维随机例子点积 {random_proj_dot:.1e}。"
            "A-GEM 对梯度的投影与 -u 投影重合。阈值：投影后点积 ≤1e-10，"
            "不违规更新保持不变。"
        ),
        "metrics": {
            "seed": SEED,
            "violating_raw_dot": _num(raw_dot),
            "projected_dot": _num(projected_dot),
            "allowed_raw_dot": _num(allowed_dot),
            "random_raw_dot": _num(random_raw),
            "random_projected_dot": _num(random_proj_dot),
            "agem_gradient_dot": _num(agem_dot),
            "update_proj_neg_agem_l2": _num(equivalent),
        },
        "checks": {
            "violating_update_detected": bool(did_violate and raw_dot > 0),
            "projected_dot_non_positive": bool(projected_dot <= 1e-10),
            "non_violating_update_unchanged": bool(
                (not allowed_violated) and bool(np.allclose(allowed_proj, allowed)),
            ),
            "random_projected_dot_non_positive": bool(
                random_violated and random_proj_dot <= 1e-10,
            ),
            "agem_matches_update_projection": bool(equivalent < 1e-10),
        },
    }


LESSON = LessonExperiment(
    lesson_id="08",
    title="梯度不许踩旧任务，以及那个尴尬的基线",
    question="违反旧任务约束的梯度，投影之后与约束法向的点积是否 ≤ 0？",
    run=run,
)
