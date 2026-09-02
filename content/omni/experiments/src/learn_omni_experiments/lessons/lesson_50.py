from __future__ import annotations

import math
from typing import Any

from ..core import LessonExperiment

# Teaching grid. TRELLIS default is N=64, L≈20K; this fixture is hand-checkable.
N = 4
C = 4
K_GS = 2
R_RF = 2
MESH_CORNERS = 8
RF_AXIS = 8
RF_COLOR = 4

# Paper contracts, quoted from Xiang et al., arXiv:2412.01506v3.
PAPER_N = 64
PAPER_K = 32
PAPER_R = 16
PAPER_MESH_SUB = 64  # 4^3 FlexiCubes cells per latent after 64^3 -> 256^3
PAPER_FLEX_W = 45
PAPER_C = 8
PAPER_RF_AXIS = 8
PAPER_RF_COLOR = 4

# Six active voxels of a mug: 2x2 body plus a two-voxel handle.
ACTIVE = (
    (1, 1, 1),
    (2, 1, 1),
    (1, 1, 2),
    (2, 1, 2),
    (0, 1, 1),
    (0, 2, 1),
)

RADIUS_DEFAULT = 1.0
RADIUS_WIDE = 1.8
CORRUPT_GAIN = 0.4


def latent_at(position: tuple[int, int, int]) -> tuple[float, float, float, float]:
    x, y, z = position
    return (
        0.2 * x - 0.35,
        0.1 * y + 0.25,
        0.08 * z + 0.4,
        0.03 * (x + y + z),
    )


def make_slat() -> list[dict[str, Any]]:
    return [
        {"p": position, "z": latent_at(position)}
        for position in ACTIVE
    ]


def clone_slat(slat: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"p": item["p"], "z": item["z"]} for item in slat]


def _softplus(value: float) -> float:
    if value > 20.0:
        return value
    return math.log1p(math.exp(value))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exp_neg = math.exp(-value)
        return 1.0 / (1.0 + exp_neg)
    exp_pos = math.exp(value)
    return exp_pos / (1.0 + exp_pos)


def mesh_sdf(z: tuple[float, float, float, float]) -> tuple[float, ...]:
    z0, _, _, z3 = z
    values = []
    for corner in range(MESH_CORNERS):
        ox = corner & 1
        oy = (corner >> 1) & 1
        oz = (corner >> 2) & 1
        values.append(z0 + 0.15 * (ox + oy + oz - 1.5) + 0.05 * z3)
    return tuple(values)


def mesh_sign_byte(sdf: tuple[float, ...]) -> int:
    bits = 0
    for index, value in enumerate(sdf):
        if value > 0.0:
            bits |= 1 << index
    return bits


def decode_mesh(slat: list[dict[str, Any]]) -> dict[str, Any]:
    cells = []
    for item in slat:
        sdf = mesh_sdf(item["z"])
        cells.append(
            {
                "p": item["p"],
                "sdf": sdf,
                "sign": mesh_sign_byte(sdf),
            }
        )
    topology = tuple((cell["p"], cell["sign"]) for cell in cells)
    occupied = {item["p"] for item in slat}
    faces = 0
    for x, y, z in occupied:
        for neighbor in ((x + 1, y, z), (x, y + 1, z), (x, y, z + 1)):
            if neighbor in occupied:
                faces += 1
    return {
        "shape": (len(slat), MESH_CORNERS),
        "cells": cells,
        "topology": topology,
        "n_faces": faces,
        "n_voxels": len(slat),
    }


def decode_gaussians(
    slat: list[dict[str, Any]],
    radius_mul: float = RADIUS_DEFAULT,
) -> dict[str, Any]:
    gaussians = []
    for item in slat:
        px, py, pz = item["p"]
        z0, z1, z2, z3 = item["z"]
        for k in range(K_GS):
            offset = (
                math.tanh(z0 + 0.1 * k),
                math.tanh(z1),
                math.tanh(z2 - 0.1 * k),
            )
            scale = _softplus(z1 + 0.2 * k) * radius_mul
            color = (_sigmoid(z2), _sigmoid(z3), _sigmoid(z0))
            opacity = _sigmoid(z2 + 0.3)
            rotation = (1.0, 0.0, 0.0, 0.0)
            center = (px + offset[0], py + offset[1], pz + offset[2])
            gaussians.append(
                {
                    "p": item["p"],
                    "k": k,
                    "offset": offset,
                    "center": center,
                    "scale": (scale, scale, scale),
                    "color": color,
                    "opacity": opacity,
                    "rotation": rotation,
                }
            )
    mean_scale = sum(item["scale"][0] for item in gaussians) / len(gaussians)
    attrs = 3 + 3 + 3 + 1 + 4
    return {
        "shape": (len(slat), K_GS, attrs),
        "gaussians": gaussians,
        "mean_scale": mean_scale,
        "count": len(gaussians),
    }


def decode_radiance(slat: list[dict[str, Any]]) -> dict[str, Any]:
    vx: list[list[list[float]]] = []
    vy: list[list[list[float]]] = []
    vz: list[list[list[float]]] = []
    vc: list[list[list[float]]] = []
    for item in slat:
        z = item["z"]
        axis_x = []
        axis_y = []
        axis_z = []
        color = []
        for rank in range(R_RF):
            axis_x.append(
                [z[rank % C] + 0.01 * axis + 0.02 * rank for axis in range(RF_AXIS)]
            )
            axis_y.append(
                [z[(rank + 1) % C] + 0.015 * axis - 0.01 * rank for axis in range(RF_AXIS)]
            )
            axis_z.append(
                [z[(rank + 2) % C] + 0.012 * axis + 0.005 * rank for axis in range(RF_AXIS)]
            )
            color.append(
                [z[(rank + axis) % C] * 0.5 + 0.1 * axis for axis in range(RF_COLOR)]
            )
        vx.append(axis_x)
        vy.append(axis_y)
        vz.append(axis_z)
        vc.append(color)
    return {
        "shape_xyz": (len(slat), R_RF, RF_AXIS),
        "shape_c": (len(slat), R_RF, RF_COLOR),
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "vc": vc,
    }


def writeback_gaussian_radius(
    slat: list[dict[str, Any]],
    radius_mul: float,
) -> list[dict[str, Any]]:
    """Illegal decoder: scale leaks into the shared latent."""
    mutated = []
    delta = (radius_mul - 1.0) * CORRUPT_GAIN
    for item in slat:
        z0, z1, z2, z3 = item["z"]
        mutated.append({"p": item["p"], "z": (z0 + delta, z1, z2, z3)})
    return mutated


def gaussians_local(gaussians: dict[str, Any]) -> bool:
    for item in gaussians["gaussians"]:
        ox, oy, oz = item["offset"]
        if abs(ox) >= 1.0 or abs(oy) >= 1.0 or abs(oz) >= 1.0:
            return False
        px, py, pz = item["p"]
        cx, cy, cz = item["center"]
        if abs(cx - px) >= 1.0 or abs(cy - py) >= 1.0 or abs(cz - pz) >= 1.0:
            return False
    return True


def rf_equal(left: dict[str, Any], right: dict[str, Any], tol: float = 1e-12) -> bool:
    if left["shape_xyz"] != right["shape_xyz"] or left["shape_c"] != right["shape_c"]:
        return False
    for field in ("vx", "vy", "vz", "vc"):
        a = left[field]
        b = right[field]
        for i, row in enumerate(a):
            for r, vec in enumerate(row):
                for c, value in enumerate(vec):
                    if abs(value - b[i][r][c]) > tol:
                        return False
    return True


def slat_equal(left: list[dict[str, Any]], right: list[dict[str, Any]], tol: float = 1e-12) -> bool:
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if a["p"] != b["p"]:
            return False
        for u, v in zip(a["z"], b["z"]):
            if abs(u - v) > tol:
                return False
    return True


def positions_in_grid(slat: list[dict[str, Any]]) -> bool:
    seen: set[tuple[int, int, int]] = set()
    for item in slat:
        x, y, z = item["p"]
        if not (0 <= x < N and 0 <= y < N and 0 <= z < N):
            return False
        if item["p"] in seen:
            return False
        seen.add(item["p"])
        if len(item["z"]) != C:
            return False
    return True


def run() -> dict[str, Any]:
    slat = make_slat()
    snapshot = clone_slat(slat)

    mesh_default = decode_mesh(slat)
    gs_default = decode_gaussians(slat, RADIUS_DEFAULT)
    rf_default = decode_radiance(slat)

    gs_wide = decode_gaussians(slat, RADIUS_WIDE)
    mesh_after_radius = decode_mesh(slat)
    rf_after_radius = decode_radiance(slat)

    corrupted = writeback_gaussian_radius(slat, RADIUS_WIDE)
    mesh_corrupt = decode_mesh(corrupted)
    rf_corrupt = decode_radiance(corrupted)
    gs_corrupt = decode_gaussians(corrupted, RADIUS_WIDE)

    paper_gs_shape = (PAPER_N, PAPER_K)  # documented per-voxel K, not a dense tensor
    paper_mesh_cell_shape = (PAPER_MESH_SUB, PAPER_FLEX_W)
    paper_mesh_sdf_shape = (PAPER_MESH_SUB, MESH_CORNERS)
    paper_rf_xyz_shape = (PAPER_R, PAPER_RF_AXIS)
    paper_rf_c_shape = (PAPER_R, PAPER_RF_COLOR)

    occupancy = len(slat) / (N ** 3)
    paper_occupancy_note = 20000 / (PAPER_N ** 3)

    radius_preserves_mesh = mesh_default["topology"] == mesh_after_radius["topology"]
    radius_changes_gs = gs_wide["mean_scale"] > gs_default["mean_scale"] * 1.5
    radius_preserves_rf = rf_equal(rf_default, rf_after_radius)
    latent_untouched = slat_equal(slat, snapshot)

    corrupt_breaks_mesh = mesh_corrupt["topology"] != mesh_default["topology"]
    corrupt_breaks_rf = not rf_equal(rf_default, rf_corrupt)
    corrupt_still_has_gs = gs_corrupt["count"] == gs_wide["count"]

    mesh_shape_ok = mesh_default["shape"] == (len(ACTIVE), MESH_CORNERS)
    gs_shape_ok = gs_default["shape"] == (len(ACTIVE), K_GS, 14)
    rf_shape_ok = (
        rf_default["shape_xyz"] == (len(ACTIVE), R_RF, RF_AXIS)
        and rf_default["shape_c"] == (len(ACTIVE), R_RF, RF_COLOR)
    )

    paper_contract_ok = (
        PAPER_K == 32
        and PAPER_R == 16
        and PAPER_MESH_SUB == 64
        and PAPER_FLEX_W == 45
        and PAPER_C == 8
        and PAPER_RF_AXIS == 8
        and PAPER_RF_COLOR == 4
        and paper_gs_shape == (64, 32)
        and paper_mesh_cell_shape == (64, 45)
        and paper_mesh_sdf_shape == (64, 8)
        and paper_rf_xyz_shape == (16, 8)
        and paper_rf_c_shape == (16, 4)
    )

    checks = {
        "mesh_shape_contract": mesh_shape_ok and mesh_default["n_voxels"] == len(ACTIVE),
        "gaussian_shape_contract": gs_shape_ok and gs_default["count"] == len(ACTIVE) * K_GS,
        "radiance_shape_contract": rf_shape_ok,
        "radius_preserves_mesh_topology": radius_preserves_mesh
        and mesh_default["n_faces"] == mesh_after_radius["n_faces"],
        "radius_changes_gaussian_scale": radius_changes_gs,
        "radius_preserves_radiance_factors": radius_preserves_rf,
        "pure_decoders_do_not_write_slat": latent_untouched,
        "gaussian_writeback_breaks_mesh_and_radiance": corrupt_breaks_mesh
        and corrupt_breaks_rf
        and corrupt_still_has_gs,
        "active_voxels_in_grid_and_sparse": positions_in_grid(slat)
        and len(slat) == len(ACTIVE)
        and occupancy < 0.25,
        "gaussian_offsets_stay_in_voxel": gaussians_local(gs_default)
        and gaussians_local(gs_wide),
        "paper_decoder_contracts_quoted": paper_contract_ok,
    }

    return {
        "summary": (
            "在 4^3 稀疏 SLAT（6 个活跃体素）上分别解码 mesh / 3D Gaussian / 辐射场，"
            "核对三种输出的 shape 契约；加大高斯半径时 mesh 拓扑与辐射场因子不变；"
            "若高斯解码器把半径写回共享 latent，mesh 与辐射场必须同时失败。"
            "本实验不训练 TRELLIS，也不报告 Toys4k FID。"
        ),
        "metrics": {
            "n": N,
            "c": C,
            "l_active": len(ACTIVE),
            "occupancy": occupancy,
            "k_gs": K_GS,
            "r_rf": R_RF,
            "mesh_shape": list(mesh_default["shape"]),
            "gaussian_shape": list(gs_default["shape"]),
            "radiance_shape_xyz": list(rf_default["shape_xyz"]),
            "radiance_shape_c": list(rf_default["shape_c"]),
            "mesh_faces": mesh_default["n_faces"],
            "mesh_topology_default": [list(item[0]) + [item[1]] for item in mesh_default["topology"]],
            "mesh_topology_wide": [
                list(item[0]) + [item[1]] for item in mesh_after_radius["topology"]
            ],
            "mesh_topology_corrupt": [
                list(item[0]) + [item[1]] for item in mesh_corrupt["topology"]
            ],
            "gs_mean_scale_default": gs_default["mean_scale"],
            "gs_mean_scale_wide": gs_wide["mean_scale"],
            "paper_n": PAPER_N,
            "paper_k": PAPER_K,
            "paper_r": PAPER_R,
            "paper_mesh_sub": PAPER_MESH_SUB,
            "paper_flex_w": PAPER_FLEX_W,
            "paper_c": PAPER_C,
            "paper_occupancy_note": paper_occupancy_note,
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="50",
    title="把三维资产生成接到统一 latent",
    question="网格、高斯和辐射场为何不能共用一个解码器？同一份 SLAT 的 shape 契约如何拆开，写坏共享 latent 时另两路为何必须失败？",
    run=run,
)
