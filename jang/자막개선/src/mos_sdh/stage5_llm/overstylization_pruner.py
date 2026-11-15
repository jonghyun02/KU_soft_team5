"""Scene-level over-stylization & CPS pruner preventing SDH cognitive fatigue (Sun et al., 2024)."""

from __future__ import annotations

import re

from mos_sdh.config import AnalysisConfig
from mos_sdh.schemas.document import SegmentMoS

_MILD_DIRECTION_RE = re.compile(r"\((힘주어|밝은 목소리로|슬픈 목소리로|차분하게)\)\s*")


def prune_overstylized_cues(
    segments: list[SegmentMoS],
    config: AnalysisConfig,
) -> tuple[list[SegmentMoS], float]:
    """Enforce Sun et al. (2024) anti-over-stylization cap on mild stage directions while preserving critical cues.

    Critical cues NEVER pruned:
      - Irony (`비꼬는 어조로` / `냉소`)
      - Dramatic honorific shift (`격앙되어 반말로`)
      - Non-speech acoustic events (`한숨`, `웃음`, `배경 음향`)
    """
    if not segments:
        return segments, 0.0

    def _has_direction(s: SegmentMoS) -> bool:
        return "(" in s.sdh_srt_text

    cue_Indices = [idx for idx, s in enumerate(segments) if _has_direction(s)]
    ratio = len(cue_Indices) / len(segments)

    if ratio <= config.max_cue_density_ratio:
        # Still prune mild directions on individual segments that severely exceed CPS
        for seg in segments:
            dur_s = max(0.5, (seg.end_ms - seg.start_ms) / 1000.0)
            curr_cps = len(seg.sdh_srt_text.replace(" ", "")) / dur_s
            if curr_cps > config.recommended_max_cps and _is_mild_prunable(seg):
                _strip_mild_direction(seg)
        final_ratio = sum(1 for s in segments if _has_direction(s)) / len(segments)
        return segments, round(final_ratio, 4)

    # Sort prunable mild segments by lowest arousal first until density <= max_cue_density_ratio
    prunable = [s for s in segments if _is_mild_prunable(s)]
    prunable.sort(key=lambda s: abs(s.arousal_zscore))

    target_max_cues = max(1, int(len(segments) * config.max_cue_density_ratio))
    current_cues = sum(1 for s in segments if _has_direction(s))

    for seg in prunable:
        if current_cues <= target_max_cues:
            break
        if _strip_mild_direction(seg):
            current_cues -= 1

    final_ratio = sum(1 for s in segments if _has_direction(s)) / len(segments)
    return segments, round(final_ratio, 4)


def _is_mild_prunable(seg: SegmentMoS) -> bool:
    if seg.emotion_irony.irony_detected:
        return False
    if seg.honorific.consistency_status == "dramatic_shift":
        return False
    if seg.cues:
        return False
    return bool(_MILD_DIRECTION_RE.search(seg.sdh_srt_text))


def _strip_mild_direction(seg: SegmentMoS) -> bool:
    new_srt = _MILD_DIRECTION_RE.sub("", seg.sdh_srt_text)
    new_srt = re.sub(r"\s{2,}", " ", new_srt).strip()
    if new_srt != seg.sdh_srt_text:
        seg.sdh_srt_text = new_srt
        seg.stage_direction_pruned = True
        return True
    return False
