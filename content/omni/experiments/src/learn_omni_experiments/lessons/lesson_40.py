from __future__ import annotations

import json
from typing import Any

from ..core import LessonExperiment

GENERATING = "GENERATING"
SAFE_HOLD = "SAFE_HOLD"
DONE = "DONE"
HUMAN = "HUMAN"

# 教学夹具：8 步倒水 chunk。力是执行该步之后测到的接触合力，单位牛顿。
FORCE_N = (4, 6, 9, 14, 22, 35, 52, 74)
H = len(FORCE_N)
EE_STEP_MM = 4
CONTACT_ONSET_N = 20
DEFAULT_F_MAX = 50
HOLD_TICKS = 4


def force_norm(fx: int, fy: int, fz: int) -> int:
    """整数牛顿上的欧氏范数，四舍五入到最近整数。"""
    return int(round((fx * fx + fy * fy + fz * fz) ** 0.5))


def contact_disp_mm(force_n: int) -> int:
    if force_n <= CONTACT_ONSET_N:
        return 0
    return int(force_n - CONTACT_ONSET_N)


def cutoff_chunk(
    forces: tuple[int, ...] | list[int],
    f_max: int,
    extra_hold_ticks: int = HOLD_TICKS,
) -> dict[str, Any]:
    """在 chunk 步 i 若 ||F_i|| > F_max，丢掉 i+1 以后的剩余步，保持当前姿态。"""
    horizon = len(forces)
    ee_mm = 0
    cup_mm = 0
    mode = GENERATING
    remaining = horizon
    cutoff_step: int | None = None
    cup_before_mm = 0
    ee_at_cutoff_mm = 0
    executed: list[int] = []
    discarded: list[int] = []
    trace: list[dict[str, int | str | None]] = []

    for step in range(horizon):
        if mode != GENERATING:
            discarded.append(step)
            remaining = 0
            trace.append(
                {
                    "step": step,
                    "kind": "discard_hold",
                    "force_n": int(forces[step]),
                    "ee_mm": ee_mm,
                    "cup_mm": cup_mm,
                    "remaining": remaining,
                    "mode": mode,
                },
            )
            continue

        force_n = int(forces[step])
        cup_before_step = cup_mm
        ee_mm += EE_STEP_MM
        cup_mm += contact_disp_mm(force_n)
        executed.append(step)
        remaining = horizon - step - 1
        tripped = force_n > f_max
        if tripped:
            cutoff_step = step
            cup_before_mm = cup_before_step
            ee_at_cutoff_mm = ee_mm
            remaining = 0
            mode = SAFE_HOLD
        trace.append(
            {
                "step": step,
                "kind": "execute",
                "force_n": force_n,
                "ee_mm": ee_mm,
                "cup_mm": cup_mm,
                "remaining": remaining,
                "mode": mode,
            },
        )

    if mode == GENERATING:
        mode = DONE
        remaining = 0

    hold_ee = []
    for tick in range(extra_hold_ticks):
        hold_ee.append(ee_mm)
        trace.append(
            {
                "step": horizon + tick,
                "kind": "hold",
                "force_n": int(forces[cutoff_step]) if cutoff_step is not None else 0,
                "ee_mm": ee_mm,
                "cup_mm": cup_mm,
                "remaining": 0,
                "mode": mode,
            },
        )

    naive_ee = horizon * EE_STEP_MM
    naive_cup = 0
    for force_n in forces:
        naive_cup += contact_disp_mm(int(force_n))

    return {
        "horizon": horizon,
        "f_max": f_max,
        "mode": mode,
        "cutoff_step": cutoff_step,
        "remaining": remaining,
        "executed": executed,
        "discarded": discarded,
        "ee_mm": ee_mm,
        "ee_at_cutoff_mm": ee_at_cutoff_mm,
        "cup_mm": cup_mm,
        "cup_before_mm": cup_before_mm,
        "hold_ee_mm": hold_ee,
        "naive_ee_mm": naive_ee,
        "naive_cup_mm": naive_cup,
        "object_rewound": cup_mm == cup_before_mm and cutoff_step is not None,
        "trace": trace,
    }


def enter_human_takeover(result: dict[str, Any], ticks: int = 3) -> dict[str, Any]:
    if result["mode"] != SAFE_HOLD:
        raise ValueError("human takeover only starts from SAFE_HOLD")
    ee_mm = int(result["ee_mm"])
    cup_mm = int(result["cup_mm"])
    takeover_ee = []
    for _ in range(ticks):
        takeover_ee.append(ee_mm)
    return {
        "mode": HUMAN,
        "ee_mm": ee_mm,
        "cup_mm": cup_mm,
        "takeover_ee_mm": takeover_ee,
        "remaining": 0,
    }


def pause_audio(horizon: int, pause_step: int) -> dict[str, Any]:
    """语音 PAUSE：丢掉未播放 PCM，前缀可重说。物理世界没有被改写。"""
    if not 0 <= pause_step < horizon:
        raise ValueError("pause_step out of range")
    remaining_pcm = 0
    kept_prefix = pause_step + 1
    return {
        "horizon": horizon,
        "pause_step": pause_step,
        "remaining_pcm": remaining_pcm,
        "kept_prefix": kept_prefix,
        "transcript_resettable": True,
        "world_unchanged": True,
    }


def filter_dangerous_demos(
    peak_forces: list[int],
    f_max: int,
) -> dict[str, Any]:
    kept = [force for force in peak_forces if force <= f_max]
    dropped = [force for force in peak_forces if force > f_max]
    return {
        "n": len(peak_forces),
        "kept": kept,
        "dropped": dropped,
        "kept_count": len(kept),
        "dropped_count": len(dropped),
    }


def _trace_digest(trace: list[dict[str, int | str | None]]) -> str:
    return json.dumps(trace, sort_keys=True, separators=(",", ":"))


def run() -> dict[str, Any]:
    gated = cutoff_chunk(FORCE_N, DEFAULT_F_MAX)
    gated_again = cutoff_chunk(FORCE_N, DEFAULT_F_MAX)
    equal_boundary = cutoff_chunk(FORCE_N, 52)
    no_trip = cutoff_chunk(FORCE_N, 74)
    takeover = enter_human_takeover(gated)
    audio = pause_audio(H, int(gated["cutoff_step"]))
    demos = filter_dangerous_demos([40, 55, 80, 12], DEFAULT_F_MAX)

    vector_50 = force_norm(30, 40, 0)
    vector_52 = force_norm(0, 0, 52)

    cutoff_step = int(gated["cutoff_step"])
    remaining_after = int(gated["remaining"])
    hold_ee = gated["hold_ee_mm"]
    executed_after_cutoff = [
        step for step in gated["executed"] if int(step) > cutoff_step
    ]

    checks = {
        "force_norm_matches_3_4_5": vector_50 == 50 and vector_52 == 52,
        "strict_greater_trips_at_52_over_50": cutoff_step == 6
        and FORCE_N[6] == 52
        and FORCE_N[6] > DEFAULT_F_MAX,
        "equal_to_threshold_does_not_trip_at_that_step": equal_boundary["cutoff_step"]
        == 7
        and FORCE_N[6] == 52,
        "remaining_steps_after_cutoff_are_zero": remaining_after == 0
        and gated["discarded"] == [7],
        "safe_hold_does_not_advance_ee": gated["mode"] == SAFE_HOLD
        and gated["ee_mm"] == gated["ee_at_cutoff_mm"] == 28
        and hold_ee == [28, 28, 28, 28]
        and executed_after_cutoff == [],
        "object_cannot_rewind_past_contact": gated["cup_before_mm"] == 17
        and gated["cup_mm"] == 49
        and gated["cup_mm"] != gated["cup_before_mm"]
        and not gated["object_rewound"]
        and gated["naive_cup_mm"] == 103,
        "audio_pause_drops_pcm_and_can_reset": audio["remaining_pcm"] == 0
        and audio["transcript_resettable"]
        and audio["world_unchanged"]
        and audio["pause_step"] == cutoff_step,
        "human_takeover_keeps_hold_pose": takeover["mode"] == HUMAN
        and takeover["remaining"] == 0
        and takeover["ee_mm"] == 28
        and takeover["cup_mm"] == 49
        and takeover["takeover_ee_mm"] == [28, 28, 28],
        "no_trip_runs_full_chunk": no_trip["mode"] == DONE
        and no_trip["cutoff_step"] is None
        and no_trip["ee_mm"] == 32
        and no_trip["remaining"] == 0,
        "dangerous_demos_filtered_by_peak_force": demos["kept_count"] == 2
        and demos["dropped_count"] == 2
        and demos["kept"] == [40, 12],
        "event_replay_is_deterministic": gated["trace"] == gated_again["trace"]
        and _trace_digest(gated["trace"]) == _trace_digest(gated_again["trace"]),
    }

    return {
        "summary": (
            "力门限在 chunk 步 i 用 ||F_i|| > F_max 切断："
            "剩余步为 0，SAFE_HOLD 不推进末端，杯子停在超限后位置且不能回退；"
            "语音 PAUSE 只丢掉未播放 PCM，物理世界未被改写。"
        ),
        "metrics": {
            "horizon": H,
            "f_max_n": DEFAULT_F_MAX,
            "cutoff_step": cutoff_step,
            "force_at_cutoff_n": FORCE_N[cutoff_step],
            "remaining_after_cutoff": remaining_after,
            "ee_mm_after_hold": gated["ee_mm"],
            "cup_mm_before": gated["cup_before_mm"],
            "cup_mm_after": gated["cup_mm"],
            "naive_cup_mm": gated["naive_cup_mm"],
            "audio_remaining_pcm": audio["remaining_pcm"],
            "demos_kept": demos["kept_count"],
            "vector_norm_30_40_0": vector_50,
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="40",
    title="在力超限时切断动作块",
    question="力超限后剩余动作步如何清零，SAFE_HOLD 为什么不能推进末端，也不能把物体退回超限前？",
    run=run,
)
