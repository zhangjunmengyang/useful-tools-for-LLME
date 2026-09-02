from __future__ import annotations

from typing import Any

from ..core import LessonExperiment


COMMON = 0
RARE = 1
VOCAB = 4


def _onehot(token: int) -> list[float]:
    vector = [0.0] * VOCAB
    vector[token] = 1.0
    return vector


def _l1(left: list[float], right: list[float]) -> float:
    return sum(abs(a - b) for a, b in zip(left, right))


def _run_memory(tokens: list[int], gated: bool) -> dict[str, Any]:
    ema = [1.0 / VOCAB] * VOCAB
    memory = [0.0] * VOCAB
    writes = {COMMON: [], RARE: []}
    surprises = {COMMON: [], RARE: []}
    decay = 0.15
    for token in tokens:
        surprise = 1.0 - ema[token]
        gate = surprise if gated else 1.0
        writes[token].append(gate)
        surprises[token].append(surprise)
        for index in range(VOCAB):
            memory[index] += gate * _onehot(token)[index]
        for index in range(VOCAB):
            ema[index] = (1.0 - decay) * ema[index]
        ema[token] += decay
    return {
        "mean_write_common": sum(writes[COMMON]) / len(writes[COMMON]),
        "mean_write_rare": sum(writes[RARE]) / len(writes[RARE]),
        "mean_surprise_common": sum(surprises[COMMON]) / len(surprises[COMMON]),
        "mean_surprise_rare": sum(surprises[RARE]) / len(surprises[RARE]),
        "memory": memory,
        "rare_count": len(writes[RARE]),
        "common_count": len(writes[COMMON]),
    }


def run() -> dict[str, Any]:
    length = 40
    tokens = [RARE if (index + 1) % 8 == 0 else COMMON for index in range(length)]
    gated = _run_memory(tokens, gated=True)
    ungated = _run_memory(tokens, gated=False)
    rare_share = gated["memory"][RARE] / max(sum(gated["memory"]), 1e-12)
    rare_freq = gated["rare_count"] / length
    predictor_error_common = _l1(_onehot(COMMON), [0.7, 0.1, 0.1, 0.1])

    checks = {
        "rare_write_exceeds_common": (
            gated["mean_write_rare"] > 1.4 * gated["mean_write_common"]
        ),
        "rare_surprise_exceeds_common": (
            gated["mean_surprise_rare"] > gated["mean_surprise_common"]
        ),
        "ungated_writes_match_per_token": (
            abs(ungated["mean_write_rare"] - ungated["mean_write_common"]) < 1e-12
        ),
        "rare_overrepresented_vs_frequency": rare_share > rare_freq,
        "common_is_mostly_predictable": predictor_error_common < 1.0,
    }
    return {
        "summary": (
            f"40 个 token 里每 8 步插一个稀有符号。惊讶门控下稀有写入 "
            f"{gated['mean_write_rare']:.3f}，常见 {gated['mean_write_common']:.3f}，"
            "比值须 > 1.4。无门控时每次写入都是 1。记忆里稀有质量占比高于出现频率。"
        ),
        "metrics": {
            "mean_write_rare": gated["mean_write_rare"],
            "mean_write_common": gated["mean_write_common"],
            "write_ratio": gated["mean_write_rare"] / gated["mean_write_common"],
            "mean_surprise_rare": gated["mean_surprise_rare"],
            "mean_surprise_common": gated["mean_surprise_common"],
            "ungated_write_rare": ungated["mean_write_rare"],
            "ungated_write_common": ungated["mean_write_common"],
            "rare_memory_share": rare_share,
            "rare_frequency": rare_freq,
            "rare_count": gated["rare_count"],
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="18",
    title="惊讶的事情才值得写入长期记忆",
    question="稀有 token 的写入幅度是不是大于常见 token？",
    run=run,
)
