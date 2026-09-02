from __future__ import annotations

import math
from typing import Any

from ..core import LessonExperiment

N_FRAMES = 8
DT = 0.1
G_TRUE = 9.8
N_OBJECTS = 3
ACTION_AX = 1.5
VANISH_FRAME = 4
VANISH_ID = 2


def _init_state() -> list[dict[str, float | int]]:
    # y increases downward. Free-fall objects start at rest.
    return [
        {"id": 1, "x": 0.0, "y": 0.4, "vx": 0.0, "vy": 0.0},
        {"id": 2, "x": 1.0, "y": 0.4, "vx": 0.0, "vy": 0.0},
        {"id": 3, "x": 2.0, "y": 0.4, "vx": 0.0, "vy": 0.0},
    ]


def _step_object(
    obj: dict[str, float | int],
    gravity: float,
    ax: float,
) -> dict[str, float | int]:
    vx = float(obj["vx"]) + ax
    vy = float(obj["vy"]) + gravity * DT
    return {
        "id": int(obj["id"]),
        "x": float(obj["x"]) + vx * DT,
        "y": float(obj["y"]) + vy * DT,
        "vx": vx,
        "vy": vy,
    }


def _rollout(
    gravity: float,
    action_frame: int,
    action_ax: float,
    vanish_id: int | None,
    vanish_frame: int | None,
) -> list[list[dict[str, float | int]]]:
    state = _init_state()
    frames = [list(state)]
    for t in range(N_FRAMES - 1):
        ax = action_ax if t == action_frame else 0.0
        nxt = [_step_object(obj, gravity, ax) for obj in state]
        if vanish_id is not None and vanish_frame is not None and t + 1 >= vanish_frame:
            nxt = [obj for obj in nxt if int(obj["id"]) != vanish_id]
        state = nxt
        frames.append(list(state))
    return frames


def _ids(frame: list[dict[str, float | int]]) -> list[int]:
    return sorted(int(obj["id"]) for obj in frame)


def _missing_id_counts(frames: list[list[dict[str, float | int]]]) -> list[int]:
    counts = []
    for t in range(len(frames) - 1):
        prev = set(_ids(frames[t]))
        nxt = set(_ids(frames[t + 1]))
        counts.append(len(prev - nxt))
    return counts


def _object_track(
    frames: list[list[dict[str, float | int]]],
    object_id: int,
) -> list[dict[str, float | int] | None]:
    track: list[dict[str, float | int] | None] = []
    for frame in frames:
        found = next((obj for obj in frame if int(obj["id"]) == object_id), None)
        track.append(found)
    return track


def _gravity_alarms(
    track: list[dict[str, float | int] | None],
    expected_sign: int,
) -> list[bool]:
    alarms = []
    for t in range(len(track) - 1):
        cur = track[t]
        nxt = track[t + 1]
        if cur is None or nxt is None:
            alarms.append(False)
            continue
        delta_vy = float(nxt["vy"]) - float(cur["vy"])
        if abs(delta_vy) < 1e-12:
            alarms.append(True)
            continue
        observed = 1 if delta_vy > 0.0 else -1
        alarms.append(observed != expected_sign)
    return alarms


def _mean_abs_dx(frames: list[list[dict[str, float | int]]], object_id: int) -> float:
    track = _object_track(frames, object_id)
    deltas = []
    for t in range(len(track) - 1):
        cur = track[t]
        nxt = track[t + 1]
        if cur is None or nxt is None:
            continue
        deltas.append(abs(float(nxt["x"]) - float(cur["x"])))
    if not deltas:
        raise ValueError("object track is empty")
    return sum(deltas) / len(deltas)


def _catch_error(
    frames: list[list[dict[str, float | int]]],
    object_id: int,
    target_x: float,
) -> float:
    last = _object_track(frames, object_id)[-1]
    if last is None:
        return math.inf
    return abs(float(last["x"]) - target_x)


def _choose_action_sign(predicted_g: float) -> float:
    # Catch by pushing toward x=1. If the model thinks gravity pulls the other way,
    # it also reverses the horizontal correction it would apply in the true world.
    return ACTION_AX if predicted_g > 0.0 else -ACTION_AX


def run() -> dict[str, Any]:
    clean = _rollout(G_TRUE, action_frame=1, action_ax=ACTION_AX, vanish_id=None, vanish_frame=None)
    vanished = _rollout(
        G_TRUE,
        action_frame=1,
        action_ax=ACTION_AX,
        vanish_id=VANISH_ID,
        vanish_frame=VANISH_FRAME,
    )
    flipped = _rollout(-G_TRUE, action_frame=1, action_ax=ACTION_AX, vanish_id=None, vanish_frame=None)
    no_action = _rollout(G_TRUE, action_frame=1, action_ax=0.0, vanish_id=None, vanish_frame=None)

    clean_missing = _missing_id_counts(clean)
    vanished_missing = _missing_id_counts(vanished)
    flipped_missing = _missing_id_counts(flipped)

    clean_track = _object_track(clean, 1)
    flipped_track = _object_track(flipped, 1)
    vanished_track_2 = _object_track(vanished, VANISH_ID)
    expected_sign = 1 if G_TRUE > 0.0 else -1
    clean_alarms = _gravity_alarms(clean_track, expected_sign)
    flipped_alarms = _gravity_alarms(flipped_track, expected_sign)

    action_dx = _mean_abs_dx(clean, 1)
    idle_dx = _mean_abs_dx(no_action, 1)

    target_x = 1.0
    true_action = _choose_action_sign(G_TRUE)
    model_action = _choose_action_sign(-G_TRUE)
    executed_true = _rollout(
        G_TRUE,
        action_frame=1,
        action_ax=true_action,
        vanish_id=None,
        vanish_frame=None,
    )
    executed_model = _rollout(
        G_TRUE,
        action_frame=1,
        action_ax=model_action,
        vanish_id=None,
        vanish_frame=None,
    )
    catch_true = _catch_error(executed_true, 1, target_x)
    catch_model = _catch_error(executed_model, 1, target_x)

    vy_increases = all(
        clean_track[t] is not None
        and clean_track[t + 1] is not None
        and float(clean_track[t + 1]["vy"]) > float(clean_track[t]["vy"])  # type: ignore[index]
        for t in range(N_FRAMES - 1)
    )
    ids_persist_clean = all(len(_ids(frame)) == N_OBJECTS for frame in clean)
    vanished_after = vanished_track_2[VANISH_FRAME:]
    id_gone = all(item is None for item in vanished_after)

    checks = {
        "clean_ids_persist": ids_persist_clean and sum(clean_missing) == 0,
        "vanished_id_counter_fires": sum(vanished_missing) >= 1 and id_gone,
        "gravity_flip_alarm_fires": any(flipped_alarms) and not any(clean_alarms),
        "clean_free_fall_keeps_positive_delta_vy": vy_increases and not any(clean_alarms),
        "action_increases_horizontal_frame_delta": action_dx > idle_dx + 1e-9,
        "controller_with_flipped_gravity_misses_catch": catch_model > catch_true + 1e-9
        and true_action == ACTION_AX
        and model_action == -ACTION_AX,
        "flip_does_not_drop_object_ids": sum(flipped_missing) == 0,
        "vanish_does_not_flip_surviving_gravity": not any(
            _gravity_alarms(_object_track(vanished, 1), expected_sign)
        ),
    }

    return {
        "summary": (
            "用落下的方块核对跨帧物体 ID 计数器和重力符号探针："
            "干净滚动不丢 ID、加速度符号与 g 一致；"
            "人为丢掉 ID 后计数器报警；把 g 取反后符号探针报警；"
            "水平动作增大帧差；用翻转重力选出的动作在真实世界里更偏离捕捉目标。"
        ),
        "metrics": {
            "n_frames": N_FRAMES,
            "n_objects": N_OBJECTS,
            "dt": DT,
            "g_true": G_TRUE,
            "action_ax": ACTION_AX,
            "vanish_frame": VANISH_FRAME,
            "vanish_id": VANISH_ID,
            "clean_missing_total": sum(clean_missing),
            "vanished_missing_total": sum(vanished_missing),
            "flipped_missing_total": sum(flipped_missing),
            "clean_alarm_count": sum(clean_alarms),
            "flipped_alarm_count": sum(flipped_alarms),
            "mean_abs_dx_with_action": action_dx,
            "mean_abs_dx_idle": idle_dx,
            "catch_error_true_gravity_action": catch_true,
            "catch_error_flipped_gravity_action": catch_model,
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="34",
    title="世界模型当数据引擎还是当控制器",
    question="生成物理视频给别人训，和自己拿预测去控，评价标准为什么不能共用一张表？",
    run=run,
)
