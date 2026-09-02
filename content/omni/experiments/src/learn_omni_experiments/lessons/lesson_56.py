from __future__ import annotations

import math
from typing import Any

from ..core import LessonExperiment

SIZE = 8
# 相对锐利参考图，一次 3×3 盒滤波比 1 像素平移更近，却更不被偏好。
BLUR_PASSES = 1
SHIFT_DX = 1
MIX_LOW = 0.15
MIX_MID = 0.40


def make_reference() -> tuple[tuple[float, ...], ...]:
    """高对比十字加棋盘：像素均值接近 0.5，高频能量集中在边。"""
    rows: list[tuple[float, ...]] = []
    for y in range(SIZE):
        row: list[float] = []
        for x in range(SIZE):
            checker = 1.0 if (x + y) % 2 == 0 else 0.0
            on_plus = x in {3, 4} or y in {3, 4}
            row.append(1.0 if on_plus else checker * 0.35)
        rows.append(tuple(row))
    return tuple(rows)


def flatten(image: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
    return tuple(value for row in image for value in row)


def mean_square_error(
    left: tuple[tuple[float, ...], ...],
    right: tuple[tuple[float, ...], ...],
) -> float:
    total = 0.0
    count = 0
    for y in range(SIZE):
        for x in range(SIZE):
            delta = left[y][x] - right[y][x]
            total += delta * delta
            count += 1
    return total / count


def box_blur_once(image: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    rows: list[tuple[float, ...]] = []
    for y in range(SIZE):
        row: list[float] = []
        for x in range(SIZE):
            acc = 0.0
            count = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy = y + dy
                    xx = x + dx
                    if 0 <= yy < SIZE and 0 <= xx < SIZE:
                        acc += image[yy][xx]
                        count += 1
            row.append(acc / count)
        rows.append(tuple(row))
    return tuple(rows)


def box_blur(
    image: tuple[tuple[float, ...], ...],
    passes: int,
) -> tuple[tuple[float, ...], ...]:
    current = image
    for _ in range(passes):
        current = box_blur_once(current)
    return current


def shift_image(
    image: tuple[tuple[float, ...], ...],
    dx: int,
    dy: int = 0,
) -> tuple[tuple[float, ...], ...]:
    """零填充平移。边错位会让 L2 暴涨，人眼仍觉得图是锐利的十字。"""
    rows: list[tuple[float, ...]] = []
    for y in range(SIZE):
        row: list[float] = []
        for x in range(SIZE):
            sx = x - dx
            sy = y - dy
            if 0 <= sx < SIZE and 0 <= sy < SIZE:
                row.append(image[sy][sx])
            else:
                row.append(0.0)
        rows.append(tuple(row))
    return tuple(rows)


def mix_images(
    left: tuple[tuple[float, ...], ...],
    right: tuple[tuple[float, ...], ...],
    weight: float,
) -> tuple[tuple[float, ...], ...]:
    """pixel = (1-w)*left + w*right。w 越大越接近锐利但错位的图。"""
    rows: list[tuple[float, ...]] = []
    for y in range(SIZE):
        row: list[float] = []
        for x in range(SIZE):
            row.append((1.0 - weight) * left[y][x] + weight * right[y][x])
        rows.append(tuple(row))
    return tuple(rows)


def edge_energy(image: tuple[tuple[float, ...], ...]) -> float:
    """离散梯度能量：过平滑会把它压下去，平移几乎不伤。"""
    total = 0.0
    count = 0
    for y in range(SIZE):
        for x in range(SIZE):
            if x + 1 < SIZE:
                gx = image[y][x + 1] - image[y][x]
                total += gx * gx
                count += 1
            if y + 1 < SIZE:
                gy = image[y + 1][x] - image[y][x]
                total += gy * gy
                count += 1
    return total / count


def preference_score(image: tuple[tuple[float, ...], ...]) -> float:
    """教学用标量奖励：奖励边和对比度，惩罚发灰。不是公开 ImageReward 权重。"""
    values = flatten(image)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    gray_penalty = 1.0 - 4.0 * mean * (1.0 - mean)
    return 2.4 * edge_energy(image) + 0.9 * variance - 0.35 * gray_penalty


def ranks_from_scores(scores: list[float], *, higher_is_better: bool) -> list[int]:
    order = sorted(
        range(len(scores)),
        key=lambda index: (-scores[index] if higher_is_better else scores[index], index),
    )
    ranks = [0] * len(scores)
    for rank, index in enumerate(order, start=1):
        ranks[index] = rank
    return ranks


def kendall_tau(left_ranks: list[int], right_ranks: list[int]) -> dict[str, float | int]:
    """Kendall τ-a：无并列时 (C-D)/C(n,2)。两序完全相反则为 -1。"""
    size = len(left_ranks)
    if size != len(right_ranks) or size < 2:
        raise ValueError("kendall_tau needs two rank lists of length >= 2")
    concordant = 0
    discordant = 0
    for i in range(size):
        for j in range(i + 1, size):
            left_sign = left_ranks[i] - left_ranks[j]
            right_sign = right_ranks[i] - right_ranks[j]
            product = left_sign * right_sign
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    pair_count = size * (size - 1) // 2
    tau = (concordant - discordant) / pair_count
    return {
        "concordant": concordant,
        "discordant": discordant,
        "pair_count": pair_count,
        "tau": tau,
    }


def bradley_terry(winner: float, loser: float) -> float:
    return 1.0 / (1.0 + math.exp(-(winner - loser)))


def reward_loss(winner: float, loser: float) -> float:
    """与 ImageReward 式 (1) 同形：-log σ(r_w - r_l)。"""
    return -math.log(bradley_terry(winner, loser))


def build_candidates() -> dict[str, Any]:
    reference = make_reference()
    oversmooth = box_blur(reference, BLUR_PASSES)
    shifted = shift_image(reference, SHIFT_DX, 0)
    candidates = {
        "oversmooth": oversmooth,
        "mostly_smooth": mix_images(oversmooth, shifted, MIX_LOW),
        "half_sharp": mix_images(oversmooth, shifted, MIX_MID),
        "shifted_sharp": shifted,
    }
    names = ("oversmooth", "mostly_smooth", "half_sharp", "shifted_sharp")
    l2_scores = [mean_square_error(candidates[name], reference) for name in names]
    pref_scores = [preference_score(candidates[name]) for name in names]
    l2_ranks = ranks_from_scores(l2_scores, higher_is_better=False)
    pref_ranks = ranks_from_scores(pref_scores, higher_is_better=True)
    return {
        "reference": reference,
        "candidates": candidates,
        "names": names,
        "l2_scores": l2_scores,
        "pref_scores": pref_scores,
        "l2_ranks": l2_ranks,
        "pref_ranks": pref_ranks,
        "kendall": kendall_tau(l2_ranks, pref_ranks),
    }


def run() -> dict[str, Any]:
    fixture = build_candidates()
    names = fixture["names"]
    l2_scores = fixture["l2_scores"]
    pref_scores = fixture["pref_scores"]
    l2_ranks = fixture["l2_ranks"]
    pref_ranks = fixture["pref_ranks"]
    kendall = fixture["kendall"]
    replay = build_candidates()

    over_i = names.index("oversmooth")
    mostly_i = names.index("mostly_smooth")
    half_i = names.index("half_sharp")
    shift_i = names.index("shifted_sharp")

    bt_shift_over_blur = bradley_terry(pref_scores[shift_i], pref_scores[over_i])
    bt_blur_over_shift = bradley_terry(pref_scores[over_i], pref_scores[shift_i])
    same_rank_tau = kendall_tau(l2_ranks, l2_ranks)["tau"]
    reversed_pref_ranks = [len(names) + 1 - rank for rank in pref_ranks]
    restored_tau = float(kendall_tau(l2_ranks, reversed_pref_ranks)["tau"])

    identity_l2 = mean_square_error(fixture["reference"], fixture["reference"])
    identity_pref = preference_score(fixture["reference"])
    mean_image = tuple(tuple(0.5 for _ in range(SIZE)) for _ in range(SIZE))
    mmse_l2 = mean_square_error(mean_image, fixture["reference"])
    mmse_pref = preference_score(mean_image)

    l2_order = [name for _, name in sorted(zip(l2_scores, names))]
    pref_order = [name for _, name in sorted(zip(pref_scores, names), reverse=True)]

    checks = {
        "l2_vs_preference_kendall_is_negative": float(kendall["tau"]) < 0.0,
        "l2_vs_preference_kendall_is_minus_one": float(kendall["tau"]) == -1.0
        and int(kendall["concordant"]) == 0
        and int(kendall["discordant"]) == int(kendall["pair_count"]),
        "oversmooth_has_lowest_l2_and_lowest_preference": l2_ranks[over_i] == 1
        and pref_ranks[over_i] == 4
        and l2_scores[over_i] < l2_scores[shift_i]
        and pref_scores[over_i] < pref_scores[shift_i],
        "preference_ranking_is_reverse_of_l2": pref_ranks
        == [len(names) + 1 - rank for rank in l2_ranks],
        "shift_beats_blur_under_bradley_terry": bt_shift_over_blur > 0.5
        and bt_blur_over_shift < 0.5
        and reward_loss(pref_scores[shift_i], pref_scores[over_i])
        < reward_loss(pref_scores[over_i], pref_scores[shift_i]),
        "mmse_gray_beats_shift_on_l2_but_loses_preference": mmse_l2
        < l2_scores[shift_i]
        and mmse_pref < pref_scores[shift_i],
        "identity_l2_is_zero_and_self_tau_is_one": identity_l2 == 0.0
        and same_rank_tau == 1.0
        and restored_tau == 1.0,
        "fixture_is_deterministic": fixture["l2_scores"] == replay["l2_scores"]
        and fixture["pref_scores"] == replay["pref_scores"]
        and fixture["kendall"] == replay["kendall"],
        "mix_weights_are_strictly_ordered": l2_scores[over_i]
        < l2_scores[mostly_i]
        < l2_scores[half_i]
        < l2_scores[shift_i]
        and pref_scores[over_i]
        < pref_scores[mostly_i]
        < pref_scores[half_i]
        < pref_scores[shift_i],
        "shifted_sharp_is_worst_l2_best_preference": l2_ranks[shift_i] == 4
        and pref_ranks[shift_i] == 1,
    }

    return {
        "summary": (
            "四张候选相对同一张锐利参考图：过平滑 L2 最低但偏好最低，"
            "1 像素平移 L2 最高但偏好最高。L2 序与偏好序完全相反，"
            f"Kendall τ-a = {float(kendall['tau']):.1f}。"
        ),
        "metrics": {
            "size": SIZE,
            "identity_l2": identity_l2,
            "identity_pref": identity_pref,
            "mmse_gray_l2": mmse_l2,
            "mmse_gray_pref": mmse_pref,
            "oversmooth_l2": l2_scores[over_i],
            "mostly_smooth_l2": l2_scores[mostly_i],
            "half_sharp_l2": l2_scores[half_i],
            "shifted_sharp_l2": l2_scores[shift_i],
            "oversmooth_pref": pref_scores[over_i],
            "mostly_smooth_pref": pref_scores[mostly_i],
            "half_sharp_pref": pref_scores[half_i],
            "shifted_sharp_pref": pref_scores[shift_i],
            "l2_rank_oversmooth": l2_ranks[over_i],
            "pref_rank_oversmooth": pref_ranks[over_i],
            "kendall_tau": float(kendall["tau"]),
            "kendall_concordant": int(kendall["concordant"]),
            "kendall_discordant": int(kendall["discordant"]),
            "kendall_pairs": int(kendall["pair_count"]),
            "bt_shift_over_oversmooth": bt_shift_over_blur,
            "reward_loss_correct_pair": reward_loss(
                pref_scores[shift_i],
                pref_scores[over_i],
            ),
            "reward_loss_reversed_pair": reward_loss(
                pref_scores[over_i],
                pref_scores[shift_i],
            ),
            "l2_best_to_worst": ",".join(l2_order),
            "preference_best_to_worst": ",".join(pref_order),
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="56",
    title="用偏好模型给生成打分",
    question="图像或视频生成的偏好奖励与像素 L2 何时会给出相反的排序？",
    run=run,
)
