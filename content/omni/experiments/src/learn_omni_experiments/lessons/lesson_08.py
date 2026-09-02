from __future__ import annotations

import math
from typing import Any

from ..core import LessonExperiment


Grid = tuple[int, int]
Position = tuple[int, float, float]


def _choose_grid(width: int, height: int, max_tiles: int) -> Grid:
    """Choose the closest row/column grid without exceeding the tile budget."""
    if width <= 0 or height <= 0 or max_tiles <= 0:
        raise ValueError("image dimensions and max_tiles must be positive")

    image_ratio = width / height
    candidates: list[tuple[float, int, int, int]] = []
    for rows in range(1, max_tiles + 1):
        for columns in range(1, max_tiles + 1):
            tile_count = rows * columns
            if tile_count > max_tiles:
                continue
            ratio_error = abs(math.log(image_ratio / (columns / rows)))
            candidates.append((ratio_error, -tile_count, rows, columns))
    _, _, rows, columns = min(candidates)
    return rows, columns


def _query_positions(image_id: int, side: int = 8) -> list[Position]:
    """Place one query at the centre of each cell in a normalized 2-D grid."""
    return [
        (image_id, (row + 0.5) / side, (column + 0.5) / side)
        for row in range(side)
        for column in range(side)
    ]


def _run() -> dict[str, Any]:
    max_tiles = 4
    wide_grid = _choose_grid(width=1600, height=600, max_tiles=max_tiles)
    tall_grid = _choose_grid(width=600, height=1600, max_tiles=max_tiles)
    square_grid = _choose_grid(width=1024, height=1024, max_tiles=max_tiles)

    first_image = _query_positions(image_id=0)
    second_image = _query_positions(image_id=1)
    packed_positions = first_image + second_image

    first_xy = [(y, x) for _, y, x in first_image]
    second_xy = [(y, x) for _, y, x in second_image]
    shuffled_xy = first_xy[1:] + first_xy[:1]
    aligned_before = list(zip(first_xy, range(len(first_xy))))
    aligned_after = list(zip(shuffled_xy, range(len(first_xy))))

    padding_position = (None, None, None)
    grids = (wide_grid, tall_grid, square_grid)
    query_count_per_image = len(first_image)

    checks = {
        "tile grids stay within the pixel budget": all(
            rows * columns <= max_tiles for rows, columns in grids
        ),
        "wide and tall inputs choose orientation-aware grids": (
            wide_grid[1] > wide_grid[0] and tall_grid[0] > tall_grid[1]
        ),
        "each image has exactly 64 query positions": (
            query_count_per_image == 64 and len(second_image) == 64
        ),
        "query coordinates remain inside the normalized image": all(
            0.0 < y < 1.0 and 0.0 < x < 1.0
            for _, y, x in packed_positions
        ),
        "image identity disambiguates equal local coordinates": (
            first_xy == second_xy
            and len(set(packed_positions)) == len(packed_positions)
        ),
        "padding carries no valid two-dimensional position": all(
            value is None for value in padding_position
        ),
        "shuffling query coordinates is detected as misalignment": (
            aligned_before != aligned_after
        ),
    }

    return {
        "summary": (
            "用确定性网格规划和两张图的 8×8 query 网格，验证 tile 预算、"
            "二维坐标与 image_id 必须同时进入视觉 token 的位置契约。"
        ),
        "metrics": {
            "max_tiles": max_tiles,
            "wide_grid": list(wide_grid),
            "tall_grid": list(tall_grid),
            "square_grid": list(square_grid),
            "query_count_per_image": query_count_per_image,
            "packed_query_count": len(packed_positions),
            "unique_position_triplets": len(set(packed_positions)),
            "first_query": list(first_image[0]),
            "last_query": list(first_image[-1]),
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="08",
    title="动态图像分辨率、多图输入与 M-RoPE",
    question="tile 预算、二维位置和多图身份怎样组成一个不会串图的位置契约？",
    run=_run,
)
