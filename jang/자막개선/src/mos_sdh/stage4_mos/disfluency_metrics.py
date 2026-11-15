"""Spontaneous disfluency density (Kim et al., 2024) and multi-window distribution (Sun et al., 2024)."""

from __future__ import annotations

import re
import numpy as np

from mos_sdh.config import AnalysisConfig
from mos_sdh.schemas.document import DisfluencyStats, SegmentMoS

_KOREAN_FILLERS = {"음", "어", "그", "저", "저기", "막", "아", "에", "응"}
_PUNCT_STRIP_RE = re.compile(r"[^\w가-힣]+")
_STUTTER_HYPHEN_RE = re.compile(r"\b([가-힣]{1,2})[,\s…\-]+\1")


def analyze_disfluency_and_cps(
    segments: list[SegmentMoS],
    config: AnalysisConfig,
) -> list[SegmentMoS]:
    """Compute FW/SR densities (Kim et al., 2024) and CPS readability per subtitle segment."""
    for seg in segments:
        base_text = seg.existing_subtitle if seg.existing_subtitle is not None else seg.raw_transcript
        raw_tokens = base_text.split()
        norm_tokens = [_PUNCT_STRIP_RE.sub("", t) for t in raw_tokens]
        norm_tokens = [t for t in norm_tokens if t]

        n_tok = len(norm_tokens)
        if n_tok == 0:
            seg.disfluency = DisfluencyStats()
            continue

        # (1) Filler Word count C_FW
        filler_indices: set[int] = set()
        for idx, tok in enumerate(norm_tokens):
            if tok in _KOREAN_FILLERS:
                filler_indices.add(idx)

        # (2) Self-Repair / consecutive repetition count C_SR (n < 5 n-gram repetitions + syllable stutters)
        sr_count = len(_STUTTER_HYPHEN_RE.findall(base_text))
        non_filler = [tok for idx, tok in enumerate(norm_tokens) if idx not in filler_indices]
        for n in (1, 2, 3):
            for i in range(0, len(non_filler) - 2 * n + 1):
                if non_filler[i : i + n] == non_filler[i + n : i + 2 * n]:
                    sr_count += 1

        fw_density = round(len(filler_indices) / n_tok, 4)
        sr_density = round(sr_count / n_tok, 4)

        dur_sec = max(0.5, (seg.end_ms - seg.start_ms) / 1000.0)
        visible_chars = len(re.sub(r"\s+", "", base_text))
        seg.cps = round(visible_chars / dur_sec, 2)
        seg.cps_exceeded = seg.cps > config.recommended_max_cps

        # Preserve acting disfluencies when emotional arousal is elevated or self-repair conveys hesitation;
        # suggest cleanup only when flat-arousal filler words push CPS above comfort limit.
        preserve = True
        cleaned: str | None = None
        if filler_indices and seg.cps_exceeded and abs(seg.arousal_zscore) < 0.8 and sr_count == 0:
            preserve = False
            kept = [tok for idx, tok in enumerate(raw_tokens) if idx not in filler_indices]
            cleaned = " ".join(kept).strip() or base_text

        seg.disfluency = DisfluencyStats(
            token_count=n_tok,
            filler_count=len(filler_indices),
            self_repair_count=sr_count,
            fw_density=fw_density,
            sr_density=sr_density,
            preserve_disfluency=preserve,
            cleaned_text=cleaned,
        )

    return segments


def compute_multi_window_mos_distribution(
    segments: list[SegmentMoS],
    num_bins: int = 5,
) -> tuple[float, list[float]]:
    """Implement Sun et al. (2024, Eq. 2-3): Score_text across n in {1..4} windows & 5-bin Score_dist."""
    if not segments:
        return 0.0, [0.0] * num_bins

    # Per-utterance MoS stylistic intensity on 0..100 scale (emphasis + cues + dramatic shift)
    unit_scores: list[float] = []
    for seg in segments:
        max_emph = max((w.emphasis_level for w in seg.words), default=0)
        cue_bonus = min(2, len(seg.cues)) * 20.0
        shift_bonus = 25.0 if seg.honorific.consistency_status == "dramatic_shift" else 0.0
        irony_bonus = 25.0 if seg.emotion_irony.irony_detected else 0.0
        raw_score = min(100.0, (max_emph / 4.0) * 55.0 + cue_bonus + shift_bonus + irony_bonus)
        unit_scores.append(raw_score)

    arr = np.asarray(unit_scores, dtype=np.float64)
    m_len = len(arr)

    # Eq. (2): Score_text = 1/4 * sum_{n in {1,2,3,4}} mean(Score_{seq-n})
    win_means: list[float] = []
    for n in (1, 2, 3, 4):
        if m_len < n:
            win_means.append(float(np.mean(arr)))
        else:
            kernel = np.ones(n, dtype=np.float64) / n
            seq_n = np.convolve(arr, kernel, mode="valid")
            win_means.append(float(np.mean(seq_n)))

    score_text = round(float(np.mean(win_means)), 2)

    # Eq. (3): Score_dist = [s_1, ..., s_K] across K=5 equal positional bins
    bins = np.array_split(arr, num_bins)
    score_dist = [round(float(np.mean(b)), 2) if len(b) > 0 else 0.0 for b in bins]
    return score_text, score_dist
