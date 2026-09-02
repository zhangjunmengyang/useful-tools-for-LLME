"""Tiny linear algebra used by extra experiments. Stdlib only."""

from __future__ import annotations

import math
import random
from typing import Callable


def zeros(rows: int, cols: int) -> list[list[float]]:
    return [[0.0] * cols for _ in range(rows)]


def unit(rng: random.Random, dim: int) -> list[float]:
    raw = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(value * value for value in raw)) or 1.0
    return [value / norm for value in raw]


def add(left: list[float], right: list[float]) -> list[float]:
    return [a + b for a, b in zip(left, right)]


def scale(vector: list[float], factor: float) -> list[float]:
    return [factor * value for value in vector]


def sub(left: list[float], right: list[float]) -> list[float]:
    return [a - b for a, b in zip(left, right)]


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def l2(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [dot(row, vector) for row in matrix]


def add_outer(
    matrix: list[list[float]],
    left: list[float],
    right: list[float],
    factor: float,
) -> None:
    for i, left_i in enumerate(left):
        row = matrix[i]
        scaled = factor * left_i
        for j, right_j in enumerate(right):
            row[j] += scaled * right_j


def copy_mat(matrix: list[list[float]]) -> list[list[float]]:
    return [row[:] for row in matrix]


def frobenius(matrix: list[list[float]]) -> float:
    return math.sqrt(sum(value * value for row in matrix for value in row))


def project_out(vector: list[float], basis: list[list[float]]) -> list[float]:
    leftover = vector[:]
    for axis in basis:
        leftover = sub(leftover, scale(axis, dot(leftover, axis)))
    return leftover


def orthonormalize(vectors: list[list[float]]) -> list[list[float]]:
    basis: list[list[float]] = []
    for vector in vectors:
        leftover = project_out(vector, basis)
        norm = l2(leftover) or 1.0
        basis.append(scale(leftover, 1.0 / norm))
    return basis


def hebbian_fit(
    pairs: list[tuple[list[float], list[float]]],
    dim: int,
    steps: int,
    lr: float,
    rng: random.Random,
    start: list[list[float]] | None = None,
) -> list[list[float]]:
    weights = copy_mat(start) if start is not None else zeros(dim, dim)
    if not pairs:
        return weights
    for _ in range(steps):
        key, value = pairs[rng.randrange(len(pairs))]
        pred = matvec(weights, key)
        add_outer(weights, sub(value, pred), key, lr)
    return weights


def hebbian_stream(
    stream: list[tuple[list[float], list[float]]],
    dim: int,
    lr: float,
    gate: str = "constant",
) -> list[list[float]]:
    """One pass over an ordered stream. gate='surprise' scales the step by residual norm."""
    weights = zeros(dim, dim)
    for key, value in stream:
        residual = sub(value, matvec(weights, key))
        factor = 1.0 + 3.0 * l2(residual) if gate == "surprise" else 1.0
        add_outer(weights, residual, key, lr * factor)
    return weights


def linear_fit(
    pairs: list[tuple[tuple[float, float], float]],
    steps: int,
    lr: float,
    rng: random.Random,
    start: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float]:
    w0, w1 = start
    if not pairs:
        return w0, w1
    for _ in range(steps):
        (a, b), y = pairs[rng.randrange(len(pairs))]
        pred = w0 * a + w1 * b
        err = pred - y
        w0 -= lr * err * a
        w1 -= lr * err * b
    return w0, w1


def make_roster(
    rng: random.Random,
    count: int,
    dim: int,
    prefix: str,
) -> tuple[
    list[tuple[str, str]],
    dict[str, list[float]],
    dict[str, list[float]],
    list[tuple[list[float], list[float]]],
]:
    people = [f"{prefix}p{i}" for i in range(count)]
    desks = [f"{prefix}d{i}" for i in range(count)]
    items = list(zip(people, desks))
    keys = {name: unit(rng, dim) for name in people}
    values = {name: unit(rng, dim) for name in desks}
    pairs = [(keys[person], values[desk]) for person, desk in items]
    return items, keys, values, pairs


def overlap_keys(
    keys: dict[str, list[float]],
    old_names: list[str],
    new_names: list[str],
    mix: float = 0.55,
) -> None:
    """Push new keys into the span of old ones so naive overwrite actually forgets."""
    if not old_names:
        return
    for index, new_name in enumerate(new_names):
        old = keys[old_names[index % len(old_names)]]
        fresh = keys[new_name]
        keys[new_name] = [
            mix * old_j + (1.0 - mix) * fresh_j
            for old_j, fresh_j in zip(old, fresh)
        ]


def residual_norm(
    weights: list[list[float]],
    key: list[float],
    value: list[float],
) -> float:
    return l2(sub(value, matvec(weights, key)))


def hebbian_lwf(
    start: list[list[float]],
    new_pairs: list[tuple[list[float], list[float]]],
    old_weights: list[list[float]],
    dim: int,
    steps: int,
    lr: float,
    lwf_lr: float,
    rng: random.Random,
    n_probes: int = 2,
) -> list[list[float]]:
    """Learn new pairs while matching the old map on random probes. No stored keys."""
    weights = copy_mat(start)
    if not new_pairs:
        return weights
    for _ in range(steps):
        key, value = new_pairs[rng.randrange(len(new_pairs))]
        add_outer(weights, sub(value, matvec(weights, key)), key, lr)
        for _probe in range(n_probes):
            probe = unit(rng, dim)
            teacher = matvec(old_weights, probe)
            add_outer(
                weights,
                sub(teacher, matvec(weights, probe)),
                probe,
                lwf_lr,
            )
    return weights


def linear_mae(
    weights: tuple[float, float],
    probes: list[tuple[float, float]],
    target: Callable[[float, float], float],
) -> float:
    if not probes:
        return 1e9
    total = 0.0
    for a, b in probes:
        total += abs(weights[0] * a + weights[1] * b - target(a, b))
    return total / len(probes)


def nearest_name(
    query: list[float],
    catalog: dict[str, list[float]],
) -> str:
    best_name = next(iter(catalog))
    best_score = -1e18
    for name, vector in catalog.items():
        score = dot(query, vector)
        if score > best_score:
            best_score = score
            best_name = name
    return best_name


def recall_hits(
    weights: list[list[float]],
    keys: dict[str, list[float]],
    values: dict[str, list[float]],
    items: list[tuple[str, str]],
) -> int:
    hits = 0
    for key_name, value_name in items:
        predicted = nearest_name(matvec(weights, keys[key_name]), values)
        if predicted == value_name:
            hits += 1
    return hits
