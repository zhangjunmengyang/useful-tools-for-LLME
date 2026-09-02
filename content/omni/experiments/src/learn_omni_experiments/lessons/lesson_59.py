from __future__ import annotations

from typing import Any

from ..core import LessonExperiment

# Teaching clip: 16 speech-codec frames at 80 ms = 1280 ms.
SPEECH_FRAME_MS = 80
MUSIC_FRAME_MS = 20
ONSET_TAU_MS = 40
N_SPEECH_FRAMES = 16

# Word labels for WER. These are not timestamps.
SPEECH_REF: tuple[str, ...] = ("ming", "tian", "ba", "dian", "kai", "hui")
SPEECH_HYP_SPEECH_CODEC: tuple[str, ...] = (
    "ming",
    "tian",
    "ba",
    "dian",
    "kai",
    "hui",
)

# Flam pairs: 8 double-strokes, 20 ms apart, every 160 ms.
# Event labels are onset times in milliseconds, not words.
DRUM_REF_MS: tuple[int, ...] = (
    0,
    20,
    160,
    180,
    320,
    340,
    480,
    500,
    640,
    660,
    800,
    820,
    960,
    980,
    1120,
    1140,
)

WER_PASS = 0.20
F1_COLLAPSE = 0.70
SHARED_PASS = 0.80


def levenshtein(reference: tuple[str, ...], hypothesis: tuple[str, ...]) -> int:
    n_ref = len(reference)
    n_hyp = len(hypothesis)
    previous = list(range(n_hyp + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i] + [0] * n_hyp
        for j, hyp_token in enumerate(hypothesis, start=1):
            substitution = previous[j - 1] + (0 if ref_token == hyp_token else 1)
            current[j] = min(previous[j] + 1, current[j - 1] + 1, substitution)
        previous = current
    return previous[n_hyp]


def word_error_rate(reference: tuple[str, ...], hypothesis: tuple[str, ...]) -> float:
    if not reference:
        raise ValueError("speech WER requires a non-empty word reference")
    if any(not isinstance(token, str) for token in reference + hypothesis):
        raise TypeError("speech WER labels must be word strings")
    return levenshtein(reference, hypothesis) / len(reference)


def snap_onsets(times_ms: tuple[int, ...], frame_ms: int) -> tuple[int, ...]:
    """Map each onset to its frame-center; merge hits that share a frame."""
    if frame_ms <= 0:
        raise ValueError("frame_ms must be positive")
    centers: list[int] = []
    seen: set[int] = set()
    for time_ms in times_ms:
        if time_ms < 0:
            raise ValueError("onset times must be non-negative")
        bin_index = time_ms // frame_ms
        if bin_index in seen:
            continue
        seen.add(bin_index)
        centers.append(bin_index * frame_ms + frame_ms // 2)
    return tuple(centers)


def event_f1(
    reference_ms: tuple[int, ...],
    hypothesis_ms: tuple[int, ...],
    tau_ms: int,
) -> dict[str, float | int]:
    if any(not isinstance(time_ms, int) for time_ms in reference_ms + hypothesis_ms):
        raise TypeError("event F1 labels must be integer milliseconds")
    if tau_ms < 0:
        raise ValueError("onset tolerance must be non-negative")
    n_ref = len(reference_ms)
    n_hyp = len(hypothesis_ms)
    if n_ref == 0 or n_hyp == 0:
        return {
            "tp": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "n_ref": n_ref,
            "n_hyp": n_hyp,
        }

    candidates: list[tuple[int, int, int]] = []
    for ref_index, ref_time in enumerate(reference_ms):
        for hyp_index, hyp_time in enumerate(hypothesis_ms):
            distance = abs(ref_time - hyp_time)
            if distance <= tau_ms:
                candidates.append((distance, ref_index, hyp_index))
    candidates.sort()

    used_ref = [False] * n_ref
    used_hyp = [False] * n_hyp
    true_positives = 0
    for _distance, ref_index, hyp_index in candidates:
        if used_ref[ref_index] or used_hyp[hyp_index]:
            continue
        used_ref[ref_index] = True
        used_hyp[hyp_index] = True
        true_positives += 1

    precision = true_positives / n_hyp
    recall = true_positives / n_ref
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "tp": true_positives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_ref": n_ref,
        "n_hyp": n_hyp,
    }


def illegal_shared_score(wer: float, f1: float) -> float:
    """Arithmetic mean of (1 - WER) and F1. Not a legal audio quality score."""
    return (1.0 - wer + f1) / 2.0


def run() -> dict[str, Any]:
    speech_wer = word_error_rate(SPEECH_REF, SPEECH_HYP_SPEECH_CODEC)
    drum_speech_codec = snap_onsets(DRUM_REF_MS, SPEECH_FRAME_MS)
    drum_music_codec = snap_onsets(DRUM_REF_MS, MUSIC_FRAME_MS)
    speech_codec_events = event_f1(DRUM_REF_MS, drum_speech_codec, ONSET_TAU_MS)
    music_codec_events = event_f1(DRUM_REF_MS, drum_music_codec, ONSET_TAU_MS)
    shared = illegal_shared_score(speech_wer, float(speech_codec_events["f1"]))

    speech_kinds = {type(token) for token in SPEECH_REF}
    event_kinds = {type(time_ms) for time_ms in DRUM_REF_MS}
    wer_on_raw_onsets_raised = False
    try:
        word_error_rate(DRUM_REF_MS, drum_speech_codec)  # type: ignore[arg-type]
    except TypeError:
        wer_on_raw_onsets_raised = True
    # Casting onsets to strings still uses the wrong reference family.
    stringified_event_wer = word_error_rate(
        tuple(str(time_ms) for time_ms in DRUM_REF_MS),
        tuple(str(time_ms) for time_ms in drum_speech_codec),
    )

    checks = {
        "speech_labels_are_words": speech_kinds == {str}
        and all(token.isalpha() for token in SPEECH_REF),
        "event_labels_are_onset_ms": event_kinds == {int}
        and all(time_ms >= 0 for time_ms in DRUM_REF_MS),
        "wer_and_f1_use_different_label_families": speech_kinds != event_kinds
        and len(SPEECH_REF) != len(DRUM_REF_MS),
        "speech_codec_wer_passes": speech_wer <= WER_PASS and speech_wer == 0.0,
        "speech_codec_drum_f1_collapses": float(speech_codec_events["f1"]) < F1_COLLAPSE
        and int(speech_codec_events["n_hyp"]) == 8
        and int(speech_codec_events["n_ref"]) == 16
        and int(speech_codec_events["tp"]) == 8,
        "shared_score_hides_the_collapse": shared >= SHARED_PASS
        and float(speech_codec_events["f1"]) < F1_COLLAPSE
        and speech_wer <= WER_PASS,
        "music_frame_rate_recovers_onsets": float(music_codec_events["f1"])
        > float(speech_codec_events["f1"])
        and int(music_codec_events["n_hyp"]) == 16
        and float(music_codec_events["f1"]) == 1.0,
        "onset_tolerance_is_not_a_word_unit": ONSET_TAU_MS == 40
        and ONSET_TAU_MS != len(SPEECH_REF)
        and stringified_event_wer > speech_wer,
        "wer_rejects_raw_onset_integers": wer_on_raw_onsets_raised,
        "clip_covers_sixteen_speech_frames": N_SPEECH_FRAMES * SPEECH_FRAME_MS == 1280
        and max(DRUM_REF_MS) < N_SPEECH_FRAMES * SPEECH_FRAME_MS,
    }

    return {
        "summary": (
            "同一 8 路、80 ms 语音 codec 网格上，语音句用词序列算 WER、"
            "鼓点用毫秒 onset 算事件 F1。两类标签不可互换。"
            "语音 WER 为 0 时，20 ms 间距的 flam 被并进同一帧，F1 只有 2/3；"
            "把 (1-WER) 与 F1 平均会得到看似合格的共用分数。"
        ),
        "metrics": {
            "speech_frame_ms": SPEECH_FRAME_MS,
            "music_frame_ms": MUSIC_FRAME_MS,
            "onset_tau_ms": ONSET_TAU_MS,
            "n_speech_frames": N_SPEECH_FRAMES,
            "n_speech_words": len(SPEECH_REF),
            "n_drum_onsets": len(DRUM_REF_MS),
            "speech_ref": list(SPEECH_REF),
            "speech_hyp_speech_codec": list(SPEECH_HYP_SPEECH_CODEC),
            "speech_wer": speech_wer,
            "drum_ref_ms": list(DRUM_REF_MS),
            "drum_hyp_speech_codec_ms": list(drum_speech_codec),
            "drum_hyp_music_codec_ms": list(drum_music_codec),
            "speech_codec_event_tp": speech_codec_events["tp"],
            "speech_codec_event_precision": speech_codec_events["precision"],
            "speech_codec_event_recall": speech_codec_events["recall"],
            "speech_codec_event_f1": speech_codec_events["f1"],
            "music_codec_event_f1": music_codec_events["f1"],
            "music_codec_event_tp": music_codec_events["tp"],
            "illegal_shared_score": shared,
            "stringified_onset_wer": stringified_event_wer,
            "wer_pass_threshold": WER_PASS,
            "f1_collapse_threshold": F1_COLLAPSE,
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="59",
    title="把音乐和环境声从语音 codec 里拆出来",
    question="同一 8 路语音码本上，语音 WER 和鼓点事件 F1 能否共用一个分数？",
    run=run,
)
