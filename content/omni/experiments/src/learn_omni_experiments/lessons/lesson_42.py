from __future__ import annotations

from typing import Any, Literal

from ..core import LessonExperiment

Modality = Literal["text", "image", "audio"]
Policy = Literal["workspace", "pollute"]
Role = Literal[
    "text",
    "image_commit",
    "image_inner",
    "audio_commit",
    "audio_inner",
]

TRANSFUSION_LAMBDA = 5.0
SHOWO2_ALPHA_STAGE1 = 0.2
SHOWO2_ALPHA_STAGE2 = 1.0


def _stages_tit() -> list[dict[str, Any]]:
    return [
        {"name": "T1", "modality": "text", "n_commit": 3, "n_inner": 1},
        {"name": "I", "modality": "image", "n_commit": 4, "n_inner": 8},
        {"name": "T2", "modality": "text", "n_commit": 3, "n_inner": 1},
    ]


def _stages_tti() -> list[dict[str, Any]]:
    return [
        {"name": "T1", "modality": "text", "n_commit": 3, "n_inner": 1},
        {"name": "T2", "modality": "text", "n_commit": 3, "n_inner": 1},
        {"name": "I", "modality": "image", "n_commit": 4, "n_inner": 8},
    ]


def _stages_with_audio() -> list[dict[str, Any]]:
    return [
        {"name": "T1", "modality": "text", "n_commit": 2, "n_inner": 1},
        {"name": "I", "modality": "image", "n_commit": 3, "n_inner": 6},
        {"name": "A", "modality": "audio", "n_commit": 4, "n_inner": 5},
        {"name": "T2", "modality": "text", "n_commit": 2, "n_inner": 1},
    ]


def expand_positions(
    stages: list[dict[str, Any]],
    policy: Policy,
) -> list[dict[str, Any]]:
    positions: list[dict[str, Any]] = []
    rank = {stage["name"]: index for index, stage in enumerate(stages)}
    for stage in stages:
        name = stage["name"]
        modality: Modality = stage["modality"]
        n_commit = int(stage["n_commit"])
        n_inner = int(stage["n_inner"])
        if modality == "text":
            for index in range(n_commit):
                positions.append(
                    {
                        "stage": name,
                        "rank": rank[name],
                        "role": "text",
                        "inner": None,
                        "slot": index,
                    },
                )
            continue
        if policy == "pollute":
            inner_role: Role = (
                "image_inner" if modality == "image" else "audio_inner"
            )
            width = n_commit if modality == "image" else 1
            for step in range(n_inner):
                for slot in range(width):
                    positions.append(
                        {
                            "stage": name,
                            "rank": rank[name],
                            "role": inner_role,
                            "inner": step,
                            "slot": slot,
                        },
                    )
        commit_role: Role = (
            "image_commit" if modality == "image" else "audio_commit"
        )
        for slot in range(n_commit):
            positions.append(
                {
                    "stage": name,
                    "rank": rank[name],
                    "role": commit_role,
                    "inner": None,
                    "slot": slot,
                },
            )
    return positions


def emission_order(stages: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(stage["name"] for stage in stages)


def committed_length(stages: list[dict[str, Any]]) -> int:
    return sum(int(stage["n_commit"]) for stage in stages)


def inner_kv_length(stages: list[dict[str, Any]]) -> int:
    total = 0
    for stage in stages:
        if stage["modality"] == "text":
            continue
        width = int(stage["n_commit"]) if stage["modality"] == "image" else 1
        total += int(stage["n_inner"]) * width
    return total


def attention_mask(
    positions: list[dict[str, Any]],
    style: Literal["transfusion", "janus"],
) -> list[list[bool]]:
    size = len(positions)
    mask = [[False] * size for _ in range(size)]
    for query_index, query in enumerate(positions):
        for key_index, key in enumerate(positions):
            causal = key_index <= query_index
            same_image_block = (
                query["role"] == "image_commit"
                and key["role"] == "image_commit"
                and query["stage"] == key["stage"]
            )
            same_audio_block = (
                query["role"] == "audio_commit"
                and key["role"] == "audio_commit"
                and query["stage"] == key["stage"]
            )
            future_text = key["role"] == "text" and key["rank"] > query["rank"]
            if future_text:
                mask[query_index][key_index] = False
                continue
            if style == "transfusion" and (same_image_block or same_audio_block):
                mask[query_index][key_index] = True
                continue
            mask[query_index][key_index] = causal
    return mask


def leak_count(
    positions: list[dict[str, Any]],
    mask: list[list[bool]],
) -> int:
    leaks = 0
    for query_index, query in enumerate(positions):
        if query["role"] != "text":
            continue
        for key_index, key in enumerate(positions):
            if key["role"] in {"image_inner", "audio_inner"} and mask[query_index][
                key_index
            ]:
                leaks += 1
    return leaks


def inner_in_kv(positions: list[dict[str, Any]]) -> int:
    return sum(
        1
        for position in positions
        if position["role"] in {"image_inner", "audio_inner"}
    )


def image_block_fully_visible(
    positions: list[dict[str, Any]],
    mask: list[list[bool]],
    stage_name: str,
) -> bool:
    indices = [
        index
        for index, position in enumerate(positions)
        if position["stage"] == stage_name and position["role"] == "image_commit"
    ]
    if not indices:
        return False
    return all(mask[query][key] for query in indices for key in indices)


def text_is_causal(
    positions: list[dict[str, Any]],
    mask: list[list[bool]],
) -> bool:
    for query_index, query in enumerate(positions):
        if query["role"] != "text":
            continue
        for key_index, key in enumerate(positions):
            if key["role"] != "text":
                continue
            allowed = mask[query_index][key_index]
            should = key_index <= query_index
            if allowed != should:
                return False
    return True


def image_cannot_see_future_text(
    positions: list[dict[str, Any]],
    mask: list[list[bool]],
) -> bool:
    for query_index, query in enumerate(positions):
        if query["role"] not in {"image_commit", "image_inner"}:
            continue
        for key_index, key in enumerate(positions):
            if key["role"] == "text" and key["rank"] > query["rank"]:
                if mask[query_index][key_index]:
                    return False
    return True


def first_commit_index(
    positions: list[dict[str, Any]],
    role: Role,
) -> int:
    for index, position in enumerate(positions):
        if position["role"] == role:
            return index
    return -1


def combined_loss(lm: float, image: float, weight: float) -> float:
    return lm + weight * image


def run() -> dict[str, Any]:
    tit = _stages_tit()
    tti = _stages_tti()
    with_audio = _stages_with_audio()

    workspace_tit = expand_positions(tit, "workspace")
    pollute_tit = expand_positions(tit, "pollute")
    workspace_tti = expand_positions(tti, "workspace")
    workspace_audio = expand_positions(with_audio, "workspace")
    pollute_audio = expand_positions(with_audio, "pollute")

    mask_workspace = attention_mask(workspace_tit, "transfusion")
    mask_pollute = attention_mask(pollute_tit, "transfusion")
    mask_janus = attention_mask(workspace_tit, "janus")
    mask_audio = attention_mask(workspace_audio, "transfusion")

    workspace_leaks = leak_count(workspace_tit, mask_workspace)
    pollute_leaks = leak_count(pollute_tit, mask_pollute)
    audio_leaks = leak_count(workspace_audio, mask_audio)
    pollute_audio_leaks = leak_count(
        pollute_audio,
        attention_mask(pollute_audio, "transfusion"),
    )

    lm_term = 1.25
    ddpm_term = 0.40
    transfusion_loss = combined_loss(lm_term, ddpm_term, TRANSFUSION_LAMBDA)
    showo2_stage1 = SHOWO2_ALPHA_STAGE1 * lm_term + ddpm_term
    showo2_stage2 = SHOWO2_ALPHA_STAGE2 * lm_term + ddpm_term

    image_start_tit = first_commit_index(workspace_tit, "image_commit")
    image_start_tti = first_commit_index(workspace_tti, "image_commit")

    janus_differs = any(
        mask_workspace[row][column] != mask_janus[row][column]
        for row in range(len(workspace_tit))
        for column in range(len(workspace_tit))
    )

    checks = {
        "workspace_kv_excludes_inner": inner_in_kv(workspace_tit) == 0,
        "pollute_kv_includes_inner": inner_in_kv(pollute_tit)
        == inner_kv_length(tit),
        "workspace_committed_length": len(workspace_tit) == committed_length(tit),
        "workspace_text_leak_is_zero": workspace_leaks == 0,
        "pollute_text_leak_is_positive": pollute_leaks > 0,
        "swap_changes_emission_order": emission_order(tit) != emission_order(tti),
        "swap_moves_image_commit": image_start_tit != image_start_tti
        and image_start_tit >= 0
        and image_start_tti >= 0,
        "image_block_bidirectional": image_block_fully_visible(
            workspace_tit,
            mask_workspace,
            "I",
        ),
        "janus_image_is_not_bidirectional": not image_block_fully_visible(
            workspace_tit,
            mask_janus,
            "I",
        ),
        "text_causal": text_is_causal(workspace_tit, mask_workspace),
        "image_cannot_see_future_text": image_cannot_see_future_text(
            workspace_tit,
            mask_workspace,
        ),
        "transfusion_janus_masks_differ": janus_differs,
        "audio_workspace_excludes_inner": inner_in_kv(workspace_audio) == 0
        and audio_leaks == 0,
        "audio_pollute_leaks": pollute_audio_leaks > 0,
        "transfusion_lambda_weights_image": transfusion_loss
        == lm_term + TRANSFUSION_LAMBDA * ddpm_term
        and transfusion_loss > lm_term + ddpm_term,
        "showo2_stage_alpha_changes_text_weight": abs(
            (showo2_stage2 - showo2_stage1)
            - (SHOWO2_ALPHA_STAGE2 - SHOWO2_ALPHA_STAGE1) * lm_term,
        )
        < 1e-12,
    }

    return {
        "summary": (
            "字-图-字日程在工作区策略下不把图像内步写入文本 KV；"
            "污染策略会产生泄漏；调换日程会改变提交顺序。"
        ),
        "metrics": {
            "workspace_kv_len": len(workspace_tit),
            "pollute_kv_len": len(pollute_tit),
            "committed_len": committed_length(tit),
            "inner_kv_len": inner_kv_length(tit),
            "workspace_leaks": workspace_leaks,
            "pollute_leaks": pollute_leaks,
            "image_commit_index_tit": image_start_tit,
            "image_commit_index_tti": image_start_tti,
            "transfusion_loss": transfusion_loss,
            "showo2_stage1_loss": showo2_stage1,
            "showo2_stage2_loss": showo2_stage2,
            "audio_workspace_kv_len": len(workspace_audio),
            "pollute_audio_leaks": pollute_audio_leaks,
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="42",
    title="交错生成日程与 KV 可见性",
    question="图像采样步会不会写入文本 KV？调换字-图-字日程后输出顺序是否改变？",
    run=run,
)
