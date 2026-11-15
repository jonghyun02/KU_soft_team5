"""Discourse speech-level consistency & dramatic shift vs ASR error separator (Hwang et al., 2021)."""

from __future__ import annotations

from collections import Counter, defaultdict

from mos_sdh.config import AnalysisConfig
from mos_sdh.schemas.document import SceneContext, SegmentMoS, SpeechLevelType
from mos_sdh.stage4_mos.korean_honorifics import (
    analyze_utterance_honorifics,
    convert_speech_level_cape,
)


def evaluate_discourse_honorific_consistency(
    segments: list[SegmentMoS],
    config: AnalysisConfig,
    scenes: list[SceneContext] | None = None,
) -> list[SegmentMoS]:
    """Analyze 6-speech-levels across K=3000ms context windows & speaker-turn history.

    Separates unintended speech-level slips (`mismatch_error` -> CAPE fix) from
    high-arousal dramatic shifts (`dramatic_shift` -> SDH dramatic cue).
    """
    if not segments:
        return segments

    # First pass: morphological honorific parsing per utterance (use existing_subtitle in Mode B if present)
    for seg in segments:
        target_text = seg.existing_subtitle if seg.existing_subtitle is not None else seg.raw_transcript
        hon = analyze_utterance_honorifics(target_text)
        hon.arousal_zscore = seg.arousal_zscore
        seg.honorific = hon

    # Track directional speaker-addressee pair history (Sa -> inferred Sb) and per-speaker baseline
    pair_history: dict[tuple[str, str], Counter[SpeechLevelType]] = defaultdict(Counter)
    speaker_history: dict[str, Counter[SpeechLevelType]] = defaultdict(Counter)

    # Seed relationship priors from Stage 3 VLM / scene context when available
    if scenes:
        for sc in scenes:
            for rel in sc.inferred_relations:
                if rel.prior_speech_level != "미분류":
                    pair_history[(rel.from_speaker, rel.to_speaker)][rel.prior_speech_level] += 1

    for idx, seg in enumerate(segments):
        hon = seg.honorific
        if hon.speech_level == "미분류":
            hon.consistency_status = "insufficient_context"
            continue

        # Hwang et al. (2021) Section 6.1: 0 < t_start(source) - t_start(context) <= K (K=3000ms), up to 2 utterances
        ctx_candidates: list[SegmentMoS] = []
        for prev_idx in range(idx - 1, -1, -1):
            prev_seg = segments[prev_idx]
            # Measure gap using both start-to-start (Hwang et al.) and end-to-start proximity for long drama turns
            delta_start = seg.start_ms - prev_seg.start_ms
            gap_from_end = seg.start_ms - prev_seg.end_ms
            if 0 < delta_start <= config.context_window_ms or 0 <= gap_from_end <= config.context_window_ms:
                ctx_candidates.append(prev_seg)
                if len(ctx_candidates) >= config.max_context_utterances:
                    break
            elif gap_from_end > config.context_window_ms * 2:
                break

        hon.context_segment_ids = [c.segment_id for c in reversed(ctx_candidates)]

        # Infer addressee from immediate preceding turn of a different speaker
        addressee = "상대방"
        for prev_idx in range(idx - 1, max(-1, idx - 4), -1):
            if segments[prev_idx].speaker_id != seg.speaker_id:
                addressee = segments[prev_idx].speaker_id
                break

        pair_key = (seg.speaker_id, addressee)
        prior_counts = pair_history[pair_key] or speaker_history[seg.speaker_id]

        # Also check immediate same-speaker utterance inside K=3000ms window (excluding prior mismatch_error slips)
        immediate_same_spk = [
            c
            for c in ctx_candidates
            if c.speaker_id == seg.speaker_id
            and c.honorific.speech_level != "미분류"
            and c.honorific.consistency_status != "mismatch_error"
        ]

        expected_level: SpeechLevelType | None = None
        if immediate_same_spk:
            expected_level = immediate_same_spk[0].honorific.speech_level
        elif sum(prior_counts.values()) >= 1:
            expected_level = prior_counts.most_common(1)[0][0]

        hon.expected_speech_level = expected_level

        if expected_level is None:
            hon.consistency_status = "insufficient_context"
        else:
            expected_polite = expected_level in {"합쇼체", "해요체"}
            curr_polite = hon.speech_level in {"합쇼체", "해요체"}

            # Compatible polite mix (합쇼체 <-> 해요체 in natural Korean conversation) vs polarity clash (존댓말 <-> 반말)
            if hon.speech_level == expected_level or (expected_polite and curr_polite):
                hon.consistency_status = "consistent"
            else:
                # Polarity clash between polite and non-polite!
                # High acoustic arousal (or explicit exclamation) => intentional dramatic shift!
                target_text = seg.existing_subtitle if seg.existing_subtitle is not None else seg.raw_transcript
                has_strong_emph = any(w.emphasis_level >= 3 for w in seg.words)
                is_high_arousal = (
                    seg.arousal_zscore >= config.dramatic_arousal_z_threshold
                    or has_strong_emph
                    or "!" in target_text
                )
                ending_low_conf = (
                    bool(seg.words)
                    and seg.words[-1].confidence < config.low_asr_confidence_threshold
                    and seg.arousal_zscore < 1.80
                )

                if is_high_arousal and not ending_low_conf:
                    hon.consistency_status = "dramatic_shift"
                else:
                    hon.consistency_status = "mismatch_error"
                    hon.suggested_correction = convert_speech_level_cape(target_text, expected_level)

        # Only update baseline history when not flagged as an unintended error
        if hon.consistency_status != "mismatch_error":
            pair_history[pair_key][hon.speech_level] += 1
            speaker_history[seg.speaker_id][hon.speech_level] += 1

    return segments
