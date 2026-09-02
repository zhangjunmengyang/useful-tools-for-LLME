from __future__ import annotations

import math
from typing import Any

from ..core import LessonExperiment

TEXT_VOCAB = 32
CODEBOOK = 8
SIDE = 2
IMAGE_TOKENS = SIDE * SIDE
PROMPT_TOKENS = 2
UNIFIED_VOCAB = TEXT_VOCAB + CODEBOOK
CODEBOOK_OFFSET = TEXT_VOCAB

# 2x2 raster image ids in the codebook, plus two prompt tokens.
IMAGE_CODES = [0, 7, 3, 4]
PROMPT_IDS = [4, 11]


def _clip_code(index: int, codebook: int = CODEBOOK) -> int:
    if index < 0:
        return 0
    if index >= codebook:
        return codebook - 1
    return index


def encode_image_token(code: int, codebook: int = CODEBOOK) -> int:
    if code < 0 or code >= codebook:
        raise ValueError("codebook index is outside [0, K-1]")
    return CODEBOOK_OFFSET + code


def decode_image_token(token: int, codebook: int = CODEBOOK) -> int:
    code = token - CODEBOOK_OFFSET
    if code < 0 or code >= codebook:
        raise ValueError("unified id is outside the image slice")
    return code


def spatial_budget(codebook: int, height: int, width: int) -> int:
    return codebook * height * width


def understand_mask(n_image: int, n_text: int) -> list[list[bool]]:
    """Image tokens first (full visual), then text (causal, may see all image)."""
    length = n_image + n_text
    mask: list[list[bool]] = []
    for query in range(length):
        row: list[bool] = []
        query_visual = query < n_image
        for key in range(length):
            key_visual = key < n_image
            if query_visual:
                allowed = key_visual
            elif key_visual:
                allowed = True
            else:
                allowed = key <= query
            row.append(allowed)
        mask.append(row)
    return mask


def generate_mask(n_text: int, n_image: int) -> list[list[bool]]:
    """Prompt first (causal text), then image tokens (causal raster, no future pixels)."""
    length = n_text + n_image
    mask: list[list[bool]] = []
    for query in range(length):
        row: list[bool] = []
        query_visual = query >= n_text
        for key in range(length):
            key_visual = key >= n_text
            if not query_visual:
                allowed = (not key_visual) and key <= query
            elif not key_visual:
                allowed = True
            else:
                allowed = key <= query
            row.append(allowed)
        mask.append(row)
    return mask


def visible_image_keys(
    mask: list[list[bool]],
    query: int,
    image_indices: list[int],
) -> list[int]:
    return [index for index in image_indices if mask[query][index]]


def future_image_keys(query_image_order: int, n_image: int) -> list[int]:
    return list(range(query_image_order + 1, n_image))


def mtp_loss_mask(n_text: int, n_image: int, masked_image: set[int]) -> list[bool]:
    """Show-o style: CE only on masked image tokens in a prompt-then-image layout."""
    flags = [False] * (n_text + n_image)
    for image_order in masked_image:
        flags[n_text + image_order] = True
    return flags


def _softmax(logits: list[float]) -> list[float]:
    peak = max(logits)
    weights = [math.exp(logit - peak) for logit in logits]
    total = sum(weights)
    return [weight / total for weight in weights]


def _cross_entropy(logits: list[float], target: int) -> float:
    if target < 0 or target >= len(logits):
        raise ValueError("target is outside the shared softmax")
    return -math.log(_softmax(logits)[target])


def run() -> dict[str, Any]:
    understand = understand_mask(IMAGE_TOKENS, PROMPT_TOKENS)
    generate = generate_mask(PROMPT_TOKENS, IMAGE_TOKENS)

    understand_image_ids = list(range(IMAGE_TOKENS))
    generate_image_ids = list(range(PROMPT_TOKENS, PROMPT_TOKENS + IMAGE_TOKENS))

    # Query the second image token (raster index 1). Future pixels are 2 and 3.
    query_image_order = 1
    understand_query = query_image_order
    generate_query = PROMPT_TOKENS + query_image_order

    understand_visible = visible_image_keys(
        understand,
        understand_query,
        understand_image_ids,
    )
    generate_visible = visible_image_keys(
        generate,
        generate_query,
        generate_image_ids,
    )
    understand_future = [
        understand_image_ids[order]
        for order in future_image_keys(query_image_order, IMAGE_TOKENS)
        if understand[understand_query][understand_image_ids[order]]
    ]
    generate_future = [
        generate_image_ids[order]
        for order in future_image_keys(query_image_order, IMAGE_TOKENS)
        if generate[generate_query][generate_image_ids[order]]
    ]

    understand_text_query = IMAGE_TOKENS  # first text token after the image
    understand_text_sees_image = all(
        understand[understand_text_query][index] for index in understand_image_ids
    )
    generate_prompt_query = 0
    generate_prompt_sees_image = any(
        generate[generate_prompt_query][index] for index in generate_image_ids
    )

    encoded = [encode_image_token(code) for code in IMAGE_CODES]
    decoded = [decode_image_token(token) for token in encoded]
    budget = spatial_budget(CODEBOOK, SIDE, SIDE)

    range_ok = all(0 <= code < CODEBOOK for code in IMAGE_CODES)
    offset_ok = encoded == [CODEBOOK_OFFSET + code for code in IMAGE_CODES]
    round_trip_ok = decoded == IMAGE_CODES

    out_of_range_rejected = False
    try:
        encode_image_token(CODEBOOK)
    except ValueError:
        out_of_range_rejected = True
    try:
        encode_image_token(-1)
        out_of_range_rejected = False
    except ValueError:
        pass
    try:
        decode_image_token(CODEBOOK_OFFSET - 1)
        out_of_range_rejected = False
    except ValueError:
        pass
    try:
        decode_image_token(CODEBOOK_OFFSET + CODEBOOK)
        out_of_range_rejected = False
    except ValueError:
        pass

    clipped = [_clip_code(-1), _clip_code(0), _clip_code(CODEBOOK - 1), _clip_code(CODEBOOK)]
    clipped_ok = clipped == [0, 0, CODEBOOK - 1, CODEBOOK - 1]

    masked_image = {1, 3}
    mtp_flags = mtp_loss_mask(PROMPT_TOKENS, IMAGE_TOKENS, masked_image)
    mtp_only_masked = mtp_flags == [
        False,
        False,
        False,
        True,
        False,
        True,
    ]

    target_code = IMAGE_CODES[query_image_order]
    logits = [0.05] * UNIFIED_VOCAB
    logits[encode_image_token(target_code)] = 2.4
    shared_ce = _cross_entropy(logits, encode_image_token(target_code))
    wrong_ce = _cross_entropy(logits, encode_image_token((target_code + 1) % CODEBOOK))
    softmax_dim_ok = len(logits) == UNIFIED_VOCAB

    understand_no_text_for_visual = not any(
        understand[understand_query][IMAGE_TOKENS + offset]
        for offset in range(PROMPT_TOKENS)
    )

    checks = {
        "understand_image_query_sees_all_image_tokens": understand_visible
        == understand_image_ids,
        "understand_image_query_sees_future_pixels": understand_future
        == [understand_image_ids[2], understand_image_ids[3]]
        and understand_no_text_for_visual,
        "generate_image_query_cannot_see_future_pixels": generate_visible
        == generate_image_ids[: query_image_order + 1]
        and generate_future == [],
        "understand_text_sees_full_image": understand_text_sees_image,
        "generate_prompt_cannot_see_image": not generate_prompt_sees_image,
        "codebook_indices_in_0_to_k_minus_1": range_ok
        and offset_ok
        and round_trip_ok
        and out_of_range_rejected
        and clipped_ok,
        "shared_softmax_covers_text_plus_codebook": softmax_dim_ok
        and UNIFIED_VOCAB == TEXT_VOCAB + CODEBOOK
        and budget == CODEBOOK * SIDE * SIDE
        and wrong_ce > shared_ce,
        "mtp_loss_only_on_masked_image_tokens": mtp_only_masked
        and not any(mtp_flags[:PROMPT_TOKENS]),
    }

    return {
        "summary": (
            "在 2x2 图像 token 加两枚提示 token 上构造理解与生成两张注意力表，"
            "核对理解路径看全图、生成路径不能看未来像素 token，"
            "并检查码本索引落在 [0, K-1]、统一词表偏移和 MTP 的 loss 位置。"
            "本实验不训练 Chameleon / Emu3 / Show-o，也不报告真实 FID 或 VQA。"
        ),
        "metrics": {
            "text_vocab": TEXT_VOCAB,
            "codebook_size": CODEBOOK,
            "unified_vocab": UNIFIED_VOCAB,
            "spatial_tokens": IMAGE_TOKENS,
            "codebook_times_spatial": budget,
            "image_codes": IMAGE_CODES,
            "encoded_ids": encoded,
            "understand_visible_image": understand_visible,
            "understand_future_visible": understand_future,
            "generate_visible_image": generate_visible,
            "generate_future_visible": generate_future,
            "shared_ce_correct": shared_ce,
            "shared_ce_wrong": wrong_ce,
            "mtp_loss_positions": [index for index, flag in enumerate(mtp_flags) if flag],
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="41",
    title="用离散 token 统一图像与文本",
    question="图像离散 token 和文本怎样共用一套 next-token，理解与生成的 mask 差在哪？",
    run=run,
)
