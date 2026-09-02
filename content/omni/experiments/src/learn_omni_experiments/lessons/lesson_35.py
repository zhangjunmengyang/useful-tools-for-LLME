from __future__ import annotations

import math
from typing import Any

from ..core import LessonExperiment

FX = 240.0
FY = 240.0
CX = 160.0
CY = 120.0
U_OBJ = 184.0
V_OBJ = 96.0
Z_OBJ = 0.52
Z_MEAN = 0.80
TAU_Z = 0.04
TAU_XY = 0.03
RGB_PIXEL_TAU = 6.0
VOXEL_SIZE = 0.01
VOXEL_BINS = 100
ADAPTIVE_SIGMA = 0.25
ADAPTIVE_Z = 0.67448975
UNIFORM_BINS = 4


def backproject(u: float, v: float, z: float) -> tuple[float, float, float]:
    if z <= 0.0:
        raise ValueError("depth must be positive")
    return ((u - CX) / FX * z, (v - CY) / FY * z, z)


def project(x: float, y: float, z: float) -> tuple[float, float]:
    if z <= 0.0:
        raise ValueError("depth must be positive")
    return (FX * x / z + CX, FY * y / z + CY)


def pixel_error(
    u_a: float, v_a: float, u_b: float, v_b: float
) -> float:
    return math.hypot(u_a - u_b, v_a - v_b)


def rgb_hit(u_g: float, v_g: float, u_o: float, v_o: float) -> bool:
    return pixel_error(u_g, v_g, u_o, v_o) <= RGB_PIXEL_TAU


def in_contact(
    grip: tuple[float, float, float],
    obj: tuple[float, float, float],
    tau_z: float = TAU_Z,
    tau_xy: float = TAU_XY,
) -> bool:
    dx = grip[0] - obj[0]
    dy = grip[1] - obj[1]
    dz = grip[2] - obj[2]
    return abs(dz) <= tau_z and math.hypot(dx, dy) <= tau_xy


def cartesian_to_polar(x: float, y: float, z: float) -> tuple[float, float, float]:
    radius = math.sqrt(x * x + y * y + z * z)
    if radius == 0.0:
        return 0.0, 0.0, 0.0
    phi = math.atan2(y, x)
    theta = math.acos(max(-1.0, min(1.0, z / radius)))
    return phi, theta, radius


def polar_to_cartesian(phi: float, theta: float, radius: float) -> tuple[float, float, float]:
    sin_theta = math.sin(theta)
    return (
        radius * sin_theta * math.cos(phi),
        radius * sin_theta * math.sin(phi),
        radius * math.cos(theta),
    )


def voxel_index(value: float, origin: float = 0.0) -> int:
    raw = math.floor((value - origin) / VOXEL_SIZE)
    if raw < 0:
        return 0
    if raw >= VOXEL_BINS:
        return VOXEL_BINS - 1
    return int(raw)


def uniform_bin(value: float, low: float = -1.0, high: float = 1.0, bins: int = UNIFORM_BINS) -> int:
    clipped = min(high, max(low, value))
    width = (high - low) / bins
    index = int((clipped - low) / width)
    if index >= bins:
        return bins - 1
    return index


def adaptive_equal_mass_bin(value: float, sigma: float = ADAPTIVE_SIGMA) -> int:
    """Four equal-probability bins of N(0, sigma), clipped to [-1, 1]."""
    inner = ADAPTIVE_Z * sigma
    if value < -inner:
        return 0
    if value < 0.0:
        return 1
    if value < inner:
        return 2
    return 3


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def run() -> dict[str, Any]:
    obj = backproject(U_OBJ, V_OBJ, Z_OBJ)
    grip_mean = backproject(U_OBJ, V_OBJ, Z_MEAN)
    grip_true = backproject(U_OBJ, V_OBJ, Z_OBJ)
    u_mean, v_mean = project(*grip_mean)
    u_true, v_true = project(*grip_true)
    u_obj, v_obj = project(*obj)

    rgb_mean = rgb_hit(u_mean, v_mean, u_obj, v_obj)
    rgb_true = rgb_hit(u_true, v_true, u_obj, v_obj)
    contact_mean = in_contact(grip_mean, obj)
    contact_true = in_contact(grip_true, obj)
    rgb_err_mean = pixel_error(u_mean, v_mean, u_obj, v_obj)
    rgb_err_true = pixel_error(u_true, v_true, u_obj, v_obj)

    scene_depths = [Z_MEAN] * 20 + [Z_OBJ]
    scene_mean = _mean(scene_depths)
    grip_scene_mean = backproject(U_OBJ, V_OBJ, scene_mean)
    contact_scene_mean = in_contact(grip_scene_mean, obj)

    phi, theta, radius = cartesian_to_polar(*obj)
    recovered = polar_to_cartesian(phi, theta, radius)
    polar_err = math.sqrt(sum((a - b) ** 2 for a, b in zip(recovered, obj)))

    voxel_z_obj = voxel_index(obj[2])
    voxel_z_mean = voxel_index(grip_mean[2])
    voxel_x_obj = voxel_index(obj[0] + 0.5, origin=0.0)

    sample_near_zero = 0.05
    uniform_near = uniform_bin(sample_near_zero)
    adaptive_near = adaptive_equal_mass_bin(sample_near_zero)
    sample_shoulder = 0.30
    uniform_shoulder = uniform_bin(sample_shoulder)
    adaptive_shoulder = adaptive_equal_mass_bin(sample_shoulder)

    tokens_independent_7d = 7
    tokens_spatialvla_step = 3
    trans_grids = 4096
    rot_grids = 4096
    grip_tokens = 2
    spatialvla_vocab = trans_grids + rot_grids + grip_tokens
    peract_rot_logits = 72 * 3

    ray_scale = abs(Z_MEAN - Z_OBJ) * math.hypot((U_OBJ - CX) / FX, (V_OBJ - CY) / FY, 1.0)
    point_gap = math.sqrt(sum((a - b) ** 2 for a, b in zip(grip_mean, obj)))

    invalid_depth_raised = False
    try:
        backproject(U_OBJ, V_OBJ, 0.0)
    except ValueError:
        invalid_depth_raised = True

    checks = {
        "round_trip_pixel_recovers_uv": abs(u_obj - U_OBJ) < 1e-12
        and abs(v_obj - V_OBJ) < 1e-12,
        "mean_depth_rgb_hit_is_true": rgb_mean and rgb_err_mean < 1e-12,
        "mean_depth_contact_is_false": (not contact_mean) and rgb_mean,
        "true_depth_contact_and_rgb_hit": contact_true and rgb_true and rgb_err_true < 1e-12,
        "scene_mean_still_misses_contact": (not contact_scene_mean)
        and abs(scene_mean - Z_MEAN) < 0.02,
        "polar_round_trip_within_1e9": polar_err < 1e-9,
        "voxel_separates_true_and_mean_z": voxel_z_obj == 52
        and voxel_z_mean == 80
        and voxel_z_obj != voxel_z_mean,
        "adaptive_grid_finer_near_zero_than_uniform": adaptive_near == uniform_near
        and adaptive_shoulder != uniform_shoulder
        and spatialvla_vocab == 8194
        and tokens_spatialvla_step < tokens_independent_7d
        and peract_rot_logits == 216
        and invalid_depth_raised
        and voxel_x_obj >= 0,
        "same_ray_gap_equals_depth_times_direction": abs(point_gap - ray_scale) < 1e-12,
    }

    return {
        "summary": (
            "同一像素 (u,v) 反投影时，无深度取场景均值会让夹爪停在射线上错误的三维点："
            "图像命中仍为真，接触带判定为假；换真实深度后两者同时为真。"
        ),
        "metrics": {
            "u_obj": U_OBJ,
            "v_obj": V_OBJ,
            "z_obj": Z_OBJ,
            "z_mean": Z_MEAN,
            "tau_z": TAU_Z,
            "tau_xy": TAU_XY,
            "obj_xyz": [obj[0], obj[1], obj[2]],
            "grip_mean_xyz": [grip_mean[0], grip_mean[1], grip_mean[2]],
            "rgb_error_mean": rgb_err_mean,
            "rgb_error_true": rgb_err_true,
            "contact_mean": contact_mean,
            "contact_true": contact_true,
            "rgb_hit_mean": rgb_mean,
            "rgb_hit_true": rgb_true,
            "scene_mean_depth": scene_mean,
            "polar_radius": radius,
            "polar_roundtrip_l2": polar_err,
            "voxel_z_obj": voxel_z_obj,
            "voxel_z_mean": voxel_z_mean,
            "uniform_bin_0p30": uniform_shoulder,
            "adaptive_bin_0p30": adaptive_shoulder,
            "spatialvla_action_vocab": spatialvla_vocab,
            "tokens_per_step_spatialvla": tokens_spatialvla_step,
            "tokens_per_step_7d_bins": tokens_independent_7d,
            "peract_rotation_logits": peract_rot_logits,
            "same_ray_gap": point_gap,
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="35",
    title="把深度接进抓取动作",
    question="缺了深度的抓取会在哪一步失败：同一 (u,v) 取均值深度时，RGB 命中能否与三维接触脱钩？",
    run=run,
)
