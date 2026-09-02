from __future__ import annotations

import math
from typing import Any

from ..core import LessonExperiment


BINS = 8
TEXT_VOCAB = 32
N_MODES = 3
N_WAYPOINTS = 6
COUNTER_NODE = 3
ARM_EXAMPLE = [0.4, -0.2, 0.0, 0.1, -0.5, 0.8, 0.3]
BASE_EXAMPLE = [0.4, 0.3]
LOST_MAP_VELOCITY = [1.1, 1.45]

ARM_DIMS = (
    ("x", -1.0, 1.0),
    ("y", -1.0, 1.0),
    ("z", -1.0, 1.0),
    ("roll", -1.0, 1.0),
    ("pitch", -1.0, 1.0),
    ("yaw", -1.0, 1.0),
    ("gripper", 0.0, 1.0),
)
BASE_DIMS = (
    ("v", 0.0, 1.5),
    ("omega", -1.5, 1.5),
)

ARM_VOCAB = len(ARM_DIMS) * BINS
BASE_VOCAB = len(BASE_DIMS) * BINS
ARM_START = TEXT_VOCAB
BASE_START = ARM_START + ARM_VOCAB
MODE_START = BASE_START + BASE_VOCAB
WAYPOINT_START = MODE_START + N_MODES


def _clip_bin(index: int, bins: int) -> int:
    if index < 0:
        return 0
    if index >= bins:
        return bins - 1
    return index


def uniform_bin(value: float, low: float, high: float, bins: int) -> int:
    if high <= low:
        raise ValueError("action range must be non-empty")
    if value <= low:
        return 0
    if value >= high:
        return bins - 1
    width = (high - low) / bins
    return _clip_bin(math.floor((value - low) / width), bins)


def bin_width(low: float, high: float, bins: int) -> float:
    return (high - low) / bins


def bin_center(index: int, low: float, high: float, bins: int) -> float:
    return low + (index + 0.5) * bin_width(low, high, bins)


def encode_dims(
    values: list[float],
    specs: tuple[tuple[str, float, float], ...],
    offset: int,
) -> list[int]:
    tokens: list[int] = []
    for dimension, value in enumerate(values):
        _name, low, high = specs[dimension]
        index = uniform_bin(value, low, high, BINS)
        tokens.append(offset + dimension * BINS + index)
    return tokens


def decode_dims(
    tokens: list[int],
    specs: tuple[tuple[str, float, float], ...],
    offset: int,
) -> list[float]:
    decoded: list[float] = []
    for dimension, token in enumerate(tokens):
        _name, low, high = specs[dimension]
        start = offset + dimension * BINS
        index = token - start
        if index < 0 or index >= BINS:
            raise ValueError("token is outside the slice of this action dimension")
        decoded.append(bin_center(index, low, high, BINS))
    return decoded


def encode_arm(action: list[float]) -> list[int]:
    return encode_dims(action, ARM_DIMS, ARM_START)


def encode_base(action: list[float]) -> list[int]:
    return encode_dims(action, BASE_DIMS, BASE_START)


def encode_mode(mode: int) -> int:
    if mode < 0 or mode >= N_MODES:
        raise ValueError("mode is outside {arm, base, terminate}")
    return MODE_START + mode


def encode_waypoint(node_id: int, n_nodes: int) -> tuple[int, bool]:
    token = WAYPOINT_START + node_id
    legal = n_nodes > 0 and 0 <= node_id < n_nodes
    return token, legal


def ranges_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def misbin_velocity_on_arm_x(value: float) -> tuple[int, float]:
    """Wrong protocol: encode chassis speed with the arm x bin edges."""
    _name, low, high = ARM_DIMS[0]
    index = uniform_bin(value, low, high, BINS)
    return index, bin_center(index, low, high, BINS)


def unicycle_step(
    x: float,
    y: float,
    theta: float,
    velocity: float,
    omega: float,
    dt: float,
) -> tuple[float, float, float]:
    next_theta = theta + omega * dt
    next_x = x + velocity * math.cos(theta) * dt
    next_y = y + velocity * math.sin(theta) * dt
    return next_x, next_y, next_theta


def hits_wall(x: float, y: float) -> bool:
    return x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0


def run() -> dict[str, Any]:
    arm_tokens = encode_arm(ARM_EXAMPLE)
    base_tokens = encode_base(BASE_EXAMPLE)
    arm_recovered = decode_dims(arm_tokens, ARM_DIMS, ARM_START)
    base_recovered = decode_dims(base_tokens, BASE_DIMS, BASE_START)

    arm_ids = list(range(ARM_START, BASE_START))
    base_ids = list(range(BASE_START, MODE_START))
    mode_ids = list(range(MODE_START, WAYPOINT_START))
    waypoint_ids = list(range(WAYPOINT_START, WAYPOINT_START + N_WAYPOINTS))
    text_ids = list(range(0, TEXT_VOCAB))

    arm_x_width = bin_width(*ARM_DIMS[0][1:], BINS)
    base_v_width = bin_width(*BASE_DIMS[0][1:], BINS)
    arm_yaw_width = bin_width(*ARM_DIMS[5][1:], BINS)
    base_omega_width = bin_width(*BASE_DIMS[1][1:], BINS)

    v_wrong_index, v_wrong_center = misbin_velocity_on_arm_x(BASE_EXAMPLE[0])
    v_right_index = uniform_bin(BASE_EXAMPLE[0], *BASE_DIMS[0][1:], BINS)
    v_high_wrong_index, v_high_wrong_center = misbin_velocity_on_arm_x(1.2)
    v_high_right_index = uniform_bin(1.2, *BASE_DIMS[0][1:], BINS)

    map_on_token, map_on_legal = encode_waypoint(COUNTER_NODE, N_WAYPOINTS)
    map_off_token, map_off_legal = encode_waypoint(COUNTER_NODE, 0)
    zero_off_token, zero_off_legal = encode_waypoint(0, 0)

    lost_velocity_tokens = encode_base(LOST_MAP_VELOCITY)
    lost_velocity_in_base_slice = all(
        BASE_START <= token < MODE_START for token in lost_velocity_tokens
    )

    x, y, theta = 0.25, 0.25, 0.0
    collided = False
    heading_travel = 0.0
    for _ in range(8):
        x, y, theta = unicycle_step(
            x,
            y,
            theta,
            LOST_MAP_VELOCITY[0],
            LOST_MAP_VELOCITY[1],
            dt=0.6,
        )
        heading_travel += abs(LOST_MAP_VELOCITY[1] * 0.6)
        if hits_wall(x, y):
            collided = True
            break

    arm_recon_error = [
        abs(original - recovered)
        for original, recovered in zip(ARM_EXAMPLE, arm_recovered)
    ]
    arm_half_widths = [bin_width(low, high, BINS) / 2 for _name, low, high in ARM_DIMS]

    disjoint_pairs = [
        not ranges_overlap((0, TEXT_VOCAB), (ARM_START, BASE_START)),
        not ranges_overlap((ARM_START, BASE_START), (BASE_START, MODE_START)),
        not ranges_overlap((BASE_START, MODE_START), (MODE_START, WAYPOINT_START)),
        not ranges_overlap(
            (MODE_START, WAYPOINT_START),
            (WAYPOINT_START, WAYPOINT_START + N_WAYPOINTS),
        ),
        not ranges_overlap((ARM_START, BASE_START), (BASE_START, MODE_START)),
    ]

    checks = {
        "arm_and_base_offsets_disjoint": min(base_ids) == BASE_START
        and max(arm_ids) == BASE_START - 1
        and set(arm_ids).isdisjoint(base_ids),
        "concatenated_slices_cover_without_gap": arm_ids + base_ids + mode_ids
        == list(range(TEXT_VOCAB, WAYPOINT_START)),
        "bin_boundaries_not_shared": arm_x_width != base_v_width
        and arm_yaw_width != base_omega_width
        and v_wrong_index != v_right_index,
        "shared_arm_bins_clip_fast_base": v_high_wrong_index == BINS - 1
        and v_high_right_index < BINS - 1
        and v_high_wrong_center < 1.2,
        "map_on_waypoint_is_legal": map_on_legal
        and 0 <= COUNTER_NODE < N_WAYPOINTS
        and map_on_token == WAYPOINT_START + COUNTER_NODE,
        "map_off_waypoint_is_illegal": (not map_off_legal)
        and map_off_token == WAYPOINT_START + COUNTER_NODE
        and (not zero_off_legal),
        "map_off_velocity_stays_in_base_vocab": lost_velocity_in_base_slice
        and min(lost_velocity_tokens) >= BASE_START
        and max(lost_velocity_tokens) < MODE_START,
        "map_off_failures_differ": (not map_off_legal)
        and collided
        and heading_travel > math.pi / 2
        and lost_velocity_in_base_slice,
        "text_arm_base_mode_waypoint_disjoint": all(disjoint_pairs)
        and set(text_ids).isdisjoint(arm_ids + base_ids + mode_ids + waypoint_ids),
        "arm_reconstruction_bounded_by_half_bin": all(
            error <= width + 1e-12
            for error, width in zip(arm_recon_error, arm_half_widths)
        ),
    }

    return {
        "summary": (
            "手臂 7 维与底盘 (v,ω) 使用不同 bin 边界并拼接成不重叠词表；"
            "丢地图后路点策略输出非法节点索引，速度策略仍落在底盘词表内并撞墙或转圈。"
        ),
        "metrics": {
            "bins": BINS,
            "text_vocab": TEXT_VOCAB,
            "arm_vocab": ARM_VOCAB,
            "base_vocab": BASE_VOCAB,
            "n_modes": N_MODES,
            "n_waypoints": N_WAYPOINTS,
            "arm_start": ARM_START,
            "base_start": BASE_START,
            "mode_start": MODE_START,
            "waypoint_start": WAYPOINT_START,
            "arm_tokens": arm_tokens,
            "base_tokens": base_tokens,
            "arm_x_bin_width": arm_x_width,
            "base_v_bin_width": base_v_width,
            "shared_bin_v_index": v_wrong_index,
            "proper_v_index": v_right_index,
            "map_on_waypoint_token": map_on_token,
            "map_off_waypoint_token": map_off_token,
            "map_off_waypoint_legal": map_off_legal,
            "map_off_zero_index_token": zero_off_token,
            "map_off_zero_index_legal": zero_off_legal,
            "lost_map_velocity_tokens": lost_velocity_tokens,
            "lost_map_collided": collided,
            "lost_map_heading_travel": heading_travel,
            "mode_arm": encode_mode(0),
            "mode_base": encode_mode(1),
            "mode_terminate": encode_mode(2),
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="36",
    title="把导航底盘和手臂动作分成两套词表",
    question="手臂词表和底盘词表的偏移为何不能重叠，丢地图后路点策略为何会吐出非法索引？",
    run=run,
)
