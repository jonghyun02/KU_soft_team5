"""Semantic embedding preservation C_i & Harmonic Mean O_i candidate ranker (de Rivero et al., 2021 Eq. 3 & 5)."""

from __future__ import annotations

import numpy as np

from mos_sdh.config import AnalysisConfig
from mos_sdh.core.vram import StageVRAMReport, gpu_stage
from mos_sdh.schemas.document import SegmentMoS, SubtitleCandidate
from mos_sdh.stage4_mos.korean_honorifics import (
    analyze_utterance_honorifics,
    strip_sdh_brackets,
)


def score_and_rank_candidates(
    segments: list[SegmentMoS],
    config: AnalysisConfig,
    telemetry_sink: list[StageVRAMReport] | None = None,
) -> list[SegmentMoS]:
    """Compute semantic inner-product C_i, MoS fidelity S_MoS, and harmonic mean O_i, then select top candidate."""
    all_pairs: list[tuple[SegmentMoS, SubtitleCandidate, str, str]] = []
    for seg in segments:
        src_plain = strip_sdh_brackets(seg.existing_subtitle or seg.raw_transcript)
        for cand in seg.candidates:
            cand_plain = strip_sdh_brackets(cand.plain_dialogue)
            all_pairs.append((seg, cand, src_plain, cand_plain))

    if not all_pairs:
        return segments

    unique_texts = sorted({txt for _, _, s_txt, c_txt in all_pairs for txt in (s_txt, c_txt) if txt})
    emb_lookup: dict[str, np.ndarray] = {}

    if unique_texts:
        with gpu_stage("Stage 5-B: Semantic Embedding C_i Ranker", telemetry_sink=telemetry_sink) as device:
            try:
                from sentence_transformers import SentenceTransformer

                encoder = SentenceTransformer(config.embedding_model, device=device)
                vecs = encoder.encode(
                    unique_texts,
                    batch_size=32,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                for txt, vec in zip(unique_texts, vecs, strict=True):
                    emb_lookup[txt] = np.asarray(vec, dtype=np.float32)
                del encoder
            except Exception:
                # Character 2-gram cosine fallback if offline without cached embedding weights
                for txt in unique_texts:
                    emb_lookup[txt] = _char_ngram_unit_vector(txt)

    for seg, cand, src_plain, cand_plain in all_pairs:
        if not src_plain or not cand_plain:
            ci = 1.0 if src_plain == cand_plain else 0.0
        elif src_plain == cand_plain:
            ci = 1.0
        else:
            v_in = emb_lookup.get(src_plain)
            v_out = emb_lookup.get(cand_plain)
            if v_in is not None and v_out is not None and v_in.shape == v_out.shape:
                # de Rivero et al. (2021) Eq. (3): C_i = R_input . R_output_i on normalized vectors
                ci = float(max(0.0, min(1.0, float(np.dot(v_in, v_out)))))
            else:
                ci = 0.85

        s_mos = _evaluate_candidate_mos_fidelity(seg, cand, config)
        passed = ci >= config.semantic_min_ci

        # de Rivero et al. (2021) Eq. (5) generalized to [0,1]: Harmonic mean O_i = 2 * F * M / (F + M)
        oi = (2.0 * s_mos * ci) / (s_mos + ci + 1e-8) if (s_mos + ci) > 0 else 0.0

        cand.semantic_ci = round(ci, 4)
        cand.mos_fidelity = round(s_mos, 4)
        cand.harmonic_oi = round(oi if passed else oi * 0.5, 4)
        cand.passed_guard = passed

    # Sort each segment's candidates descending by harmonic_oi and adopt the winner
    for seg in segments:
        if not seg.candidates:
            continue
        seg.candidates.sort(key=lambda c: (c.passed_guard, c.harmonic_oi, c.semantic_ci), reverse=True)
        for idx, cand in enumerate(seg.candidates):
            cand.selected = idx == 0
        best = seg.candidates[0]
        seg.sdh_srt_text = best.srt_text
        seg.sdh_ass_text = best.ass_text

    return segments


def _evaluate_candidate_mos_fidelity(
    seg: SegmentMoS,
    cand: SubtitleCandidate,
    config: AnalysisConfig,
) -> float:
    """Score how faithfully a candidate realizes required honorifics, irony/emotion cues, emphasis, and CPS."""
    score = 0.75

    # 1. Check Korean 6-speech-level target satisfaction
    cand_hon = analyze_utterance_honorifics(cand.plain_dialogue)
    if seg.honorific.consistency_status == "mismatch_error" and seg.honorific.expected_speech_level:
        exp_polite = seg.honorific.expected_speech_level in {"합쇼체", "해요체"}
        if cand_hon.speech_level == seg.honorific.expected_speech_level:
            score += 0.18
        elif cand_hon.is_polite_group == exp_polite:
            score += 0.12
        else:
            score -= 0.25
    elif seg.honorific.speech_level != "미분류":
        if cand_hon.speech_level == seg.honorific.speech_level:
            score += 0.10

    # 2. Check Irony / Dramatic Shift stage descriptor presence in SRT
    if seg.emotion_irony.irony_detected:
        if "비꼬" in cand.srt_text or "냉소" in cand.srt_text:
            score += 0.12
        else:
            score -= 0.15

    if seg.honorific.consistency_status == "dramatic_shift":
        if "(" in cand.srt_text:
            score += 0.08

    # 3. Check CPS readability comfort
    dur_sec = max(0.5, (seg.end_ms - seg.start_ms) / 1000.0)
    cand_cps = len(cand.srt_text.replace(" ", "")) / dur_sec
    if cand_cps <= config.recommended_max_cps:
        score += 0.05
    elif cand_cps > config.recommended_max_cps * 1.25:
        score -= 0.12

    return float(max(0.05, min(1.0, score)))


def _char_ngram_unit_vector(text: str, dim: int = 256) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    clean = text.replace(" ", "")
    if not clean:
        vec[0] = 1.0
        return vec
    for i in range(len(clean)):
        h1 = hash(clean[i]) % dim
        vec[h1] += 1.0
        if i + 1 < len(clean):
            h2 = hash(clean[i : i + 2]) % dim
            vec[h2] += 1.5
    norm = float(np.linalg.norm(vec)) + 1e-8
    return vec / norm
