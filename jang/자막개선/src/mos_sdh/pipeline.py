"""End-to-end 5-Stage MoS-SDH orchestrator supporting Mode A (Video-Only) and Mode B (Subtitle QA)."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
import numpy as np
import pysubs2

from mos_sdh.config import AnalysisConfig
from mos_sdh.core.vram import StageVRAMReport
from mos_sdh.exporters.subtitle_exporter import (
    build_ass_line,
    build_srt_line,
    populate_sdh_render_strings,
)
from mos_sdh.schemas.document import (
    BenchmarkMetrics,
    MoSDocument,
    ParalinguisticCue,
    SceneContext,
    SegmentMoS,
    StageVRAMTelemetry,
    SWERComparison,
    WordUnit,
)
from mos_sdh.stage1_audio.diarization_vad import (
    DiarizedRegion,
    assign_speaker_by_overlap,
    run_vad_and_diarization,
)
from mos_sdh.stage1_audio.media_io import extract_standardized_audio
from mos_sdh.stage1_audio.prosody_emphasis import annotate_prosody_and_emphasis
from mos_sdh.stage1_audio.separator import separate_vocals_and_bgm
from mos_sdh.stage1_audio.stt_whisper import transcribe_with_word_alignment
from mos_sdh.stage2_visual.face_lip_fer import analyze_faces_lips_and_expressions
from mos_sdh.stage2_visual.scene_detector import (
    detect_scenes_and_keyframes,
    ensure_ascii_safe_video_path,
)
from mos_sdh.stage2_visual.scene_ocr import run_scene_ocr_and_collision_detection
from mos_sdh.stage3_vlm.scene_vlm import infer_scene_contexts_and_relationships
from mos_sdh.stage4_mos.context_consistency import evaluate_discourse_honorific_consistency
from mos_sdh.stage4_mos.disfluency_metrics import (
    analyze_disfluency_and_cps,
    compute_multi_window_mos_distribution,
)
from mos_sdh.stage4_mos.swer_evaluator import evaluate_cer_wer_swer
from mos_sdh.stage4_mos.trimodal_cmgi import analyze_trimodal_emotion_and_irony
from mos_sdh.stage5_llm.candidate_generator import generate_sdh_candidates
from mos_sdh.stage5_llm.candidate_ranker import score_and_rank_candidates
from mos_sdh.stage5_llm.overstylization_pruner import prune_overstylized_cues


def run_full_pipeline(
    input_media: Path,
    config: AnalysisConfig,
    existing_subtitle: Path | None = None,
) -> MoSDocument:
    """Execute Stages 1 through 5 sequentially with automatic GPU VRAM offloading between stages."""
    telemetry: list[StageVRAMReport] = []

    # =========================================================================
    # STAGE 1: Audio Demux, Source Separation, Diarization/VAD, STT & Prosody
    # =========================================================================
    extracted = extract_standardized_audio(input_media, config.cache_dir)

    sep = separate_vocals_and_bgm(
        stereo_44k_wav=extracted.stereo_44k_wav,
        fallback_mono_16k_wav=extracted.mono_16k_wav,
        enable_separation=config.enable_source_separation,
        telemetry_sink=telemetry,
    )

    diarized_regions, vocal_cues = run_vad_and_diarization(sep.vocals_16k_wav, config)

    stt_segments = transcribe_with_word_alignment(
        vocals_16k_wav=sep.vocals_16k_wav,
        diarized_regions=diarized_regions,
        config=config,
        telemetry_sink=telemetry,
    )

    mode = "mode_b_subtitle_qa" if existing_subtitle is not None else "mode_a_video_only"
    swer_comp: SWERComparison | None = None

    if existing_subtitle is not None:
        segments, swer_comp = _align_existing_subtitles_with_stt(
            existing_subtitle_path=existing_subtitle,
            stt_segments=stt_segments,
            diarized_regions=diarized_regions,
        )
    else:
        segments = stt_segments

    segments = annotate_prosody_and_emphasis(sep.vocals_16k_wav, segments, config)

    all_cues = vocal_cues + sep.bgm_sfx_cues
    _attach_paralinguistic_cues(segments, all_cues)

    # =========================================================================
    # STAGE 2: High-Density Visual Scene Cuts, Lip ASD + ViT FER & Scene OCR
    # =========================================================================
    scenes: list[SceneContext] = []
    keyframe_paths: dict[int, Path] = {}
    temporal_fer_sequences: dict[int, np.ndarray] = {}

    if config.enable_visual_stage:
        safe_video = ensure_ascii_safe_video_path(input_media, config.cache_dir)
        if safe_video is not None:
            scenes, keyframe_paths = detect_scenes_and_keyframes(
                safe_video_path=safe_video,
                duration_ms=extracted.duration_ms,
                segments=segments,
                cache_dir=config.cache_dir,
            )
            temporal_fer_sequences = analyze_faces_lips_and_expressions(
                safe_video_path=safe_video,
                segments=segments,
                config=config,
                telemetry_sink=telemetry,
            )
            run_scene_ocr_and_collision_detection(
                safe_video_path=safe_video,
                segments=segments,
                scenes=scenes,
                config=config,
                telemetry_sink=telemetry,
            )

    # =========================================================================
    # STAGE 3: Local 4-bit VLM Scene Context & Speaker Social Relationship Graph
    # =========================================================================
    if scenes:
        scenes = infer_scene_contexts_and_relationships(
            scenes=scenes,
            keyframe_paths=keyframe_paths,
            segments=segments,
            config=config,
            telemetry_sink=telemetry,
        )

    # =========================================================================
    # STAGE 4: 6-Speech-Level Consistency + Trimodal CmGI Emotion/Irony + FW/SR
    # =========================================================================
    segments = evaluate_discourse_honorific_consistency(segments, config, scenes=scenes)
    segments = analyze_trimodal_emotion_and_irony(segments, temporal_fer_sequences, config)
    segments = analyze_disfluency_and_cps(segments, config)
    score_text, score_dist = compute_multi_window_mos_distribution(segments, num_bins=5)

    # =========================================================================
    # STAGE 5: Positional Tag + Local 4-bit LLM Candidates, C_i/O_i Rank & Prune
    # =========================================================================
    segments = generate_sdh_candidates(
        segments=segments,
        config=config,
        render_srt_fn=build_srt_line,
        render_ass_fn=build_ass_line,
        telemetry_sink=telemetry,
    )
    segments = score_and_rank_candidates(
        segments=segments,
        config=config,
        telemetry_sink=telemetry,
    )
    segments, cue_ratio = prune_overstylized_cues(segments, config)

    metrics = _compile_benchmark_metrics(
        segments=segments,
        scenes=scenes,
        score_text=score_text,
        score_dist=score_dist,
        cue_ratio=cue_ratio,
        swer_comp=swer_comp,
    )

    doc = MoSDocument(
        mode=mode,
        input_media_path=str(input_media.resolve()),
        input_subtitle_path=str(existing_subtitle.resolve()) if existing_subtitle else None,
        duration_ms=extracted.duration_ms,
        scenes=scenes,
        segments=segments,
        metrics=metrics,
        vram_telemetry=[
            StageVRAMTelemetry(
                stage_name=t.stage_name,
                device=t.device,
                gpu_name=t.gpu_name,
                allocated_before_mb=t.allocated_before_mb,
                peak_allocated_mb=t.peak_allocated_mb,
                allocated_after_mb=t.allocated_after_mb,
            )
            for t in telemetry
        ],
    )
    return populate_sdh_render_strings(doc, overwrite=False)


def run_phase1a_pipeline(
    input_media: Path,
    config: AnalysisConfig,
    existing_subtitle: Path | None = None,
) -> MoSDocument:
    """Backward-compatible fast audio+linguistic MoS pipeline entrypoint."""
    fast_cfg = replace(
        config,
        enable_visual_stage=False,
        enable_vlm_stage=False,
        enable_llm_stage=False,
    )
    return run_full_pipeline(
        input_media=input_media,
        config=fast_cfg,
        existing_subtitle=existing_subtitle,
    )


def _align_existing_subtitles_with_stt(
    existing_subtitle_path: Path,
    stt_segments: list[SegmentMoS],
    diarized_regions: list[DiarizedRegion],
) -> tuple[list[SegmentMoS], SWERComparison | None]:
    """Load existing .srt/.vtt, align words & STT transcripts by timestamp overlap, and compute sWER."""
    subs = pysubs2.load(str(existing_subtitle_path), encoding="utf-8")
    all_stt_words: list[WordUnit] = [w for s in stt_segments for w in s.words]

    aligned_segments: list[SegmentMoS] = []
    ref_lines: list[str] = []
    hyp_lines: list[str] = []

    for idx, ev in enumerate(subs.events, start=1):
        if ev.is_comment:
            continue
        sub_text = ev.plaintext.replace("\n", " ").strip()
        if not sub_text:
            continue

        s_ms = int(ev.start)
        e_ms = max(s_ms + 100, int(ev.end))
        spk = assign_speaker_by_overlap(s_ms, e_ms, diarized_regions)

        matched_stt_words = [
            sw
            for sw in all_stt_words
            if (s_ms - 200) <= 0.5 * (sw.start_ms + sw.end_ms) <= (e_ms + 200)
        ]
        if matched_stt_words:
            stt_joined = " ".join(sw.text for sw in matched_stt_words).strip()
        else:
            overlapping_stt = [
                st
                for st in stt_segments
                if max(0, min(e_ms, st.end_ms) - max(s_ms, st.start_ms)) > 0
            ]
            stt_joined = " ".join(st.raw_transcript for st in overlapping_stt).strip()

        if stt_joined:
            ref_lines.append(stt_joined)
            hyp_lines.append(sub_text)

        sub_tokens = sub_text.split()
        span = max(100, e_ms - s_ms)
        step = span / max(1, len(sub_tokens))
        mapped_words: list[WordUnit] = []
        for w_i, tok in enumerate(sub_tokens):
            ws = s_ms + int(round(w_i * step))
            we = s_ms + int(round((w_i + 1) * step))
            near_probs = [
                sw.confidence
                for sw in all_stt_words
                if max(0, min(we, sw.end_ms) - max(ws, sw.start_ms)) > 0
            ]
            conf = round(sum(near_probs) / len(near_probs), 4) if near_probs else 0.9
            mapped_words.append(WordUnit(text=tok, start_ms=ws, end_ms=we, confidence=conf))

        aligned_segments.append(
            SegmentMoS(
                segment_id=idx,
                start_ms=s_ms,
                end_ms=e_ms,
                speaker_id=spk,
                raw_transcript=stt_joined or sub_text,
                existing_subtitle=sub_text,
                asr_avg_confidence=round(
                    sum(w.confidence for w in mapped_words) / max(1, len(mapped_words)), 4
                ),
                words=mapped_words,
            )
        )

    swer_comp = evaluate_cer_wer_swer(ref_lines, hyp_lines) if ref_lines else None
    return aligned_segments, swer_comp


def _attach_paralinguistic_cues(
    segments: list[SegmentMoS],
    cues: list[ParalinguisticCue],
) -> None:
    """Bind detected non-speech cues (b/, l/, sfx) to overlapping or immediately preceding/following segments."""
    if not segments or not cues:
        return

    for cue in cues:
        cue_mid = 0.5 * (cue.start_ms + cue.end_ms)
        best_seg = min(
            segments,
            key=lambda s: (
                0.0
                if s.start_ms <= cue_mid <= s.end_ms
                else min(abs(cue_mid - s.start_ms), abs(cue_mid - s.end_ms))
            ),
        )
        dist_ms = (
            0.0
            if best_seg.start_ms <= cue_mid <= best_seg.end_ms
            else min(abs(cue_mid - best_seg.start_ms), abs(cue_mid - best_seg.end_ms))
        )
        if dist_ms <= 2500:
            if not any(existing.cue_code == cue.cue_code for existing in best_seg.cues):
                best_seg.cues.append(cue)


def _compile_benchmark_metrics(
    segments: list[SegmentMoS],
    scenes: list[SceneContext],
    score_text: float,
    score_dist: list[float],
    cue_ratio: float,
    swer_comp: SWERComparison | None,
) -> BenchmarkMetrics:
    if not segments:
        return BenchmarkMetrics(
            scene_count=len(scenes),
            score_dist_5bin=score_dist,
            swer_comparison=swer_comp,
        )

    level_counts = Counter(seg.honorific.speech_level for seg in segments)
    emo_counts = Counter(seg.emotion_irony.dominant_emotion for seg in segments)
    evaluated = [
        seg
        for seg in segments
        if seg.honorific.consistency_status in {"consistent", "mismatch_error", "dramatic_shift"}
    ]
    mismatch_cnt = sum(1 for seg in segments if seg.honorific.consistency_status == "mismatch_error")
    dramatic_cnt = sum(1 for seg in segments if seg.honorific.consistency_status == "dramatic_shift")
    irony_cnt = sum(1 for seg in segments if seg.emotion_irony.irony_detected)
    off_screen_cnt = sum(1 for seg in segments if seg.visual.on_screen_speaker is False)
    ocr_avoid_cnt = sum(1 for seg in segments if seg.visual.avoid_bottom_region)

    consistency_rate = (
        round(((len(evaluated) - mismatch_cnt) / len(evaluated)) * 100.0, 2) if evaluated else 100.0
    )

    emph_cnt = sum(1 for seg in segments for w in seg.words if w.emphasis_level >= 1)
    strong_emph_cnt = sum(1 for seg in segments for w in seg.words if w.emphasis_level >= 3)
    cue_cnt = sum(len(seg.cues) for seg in segments)

    selected_cands = [next((c for c in seg.candidates if c.selected), None) for seg in segments]
    valid_cands = [c for c in selected_cands if c is not None]
    mean_ci = (
        round(sum(c.semantic_ci for c in valid_cands) / len(valid_cands), 4)
        if valid_cands
        else 1.0
    )
    mean_oi = (
        round(sum(c.harmonic_oi for c in valid_cands) / len(valid_cands), 4)
        if valid_cands
        else 1.0
    )

    mean_cps = round(sum(seg.cps for seg in segments) / len(segments), 2)
    cps_ok = sum(1 for seg in segments if not seg.cps_exceeded)
    cps_rate = round((cps_ok / len(segments)) * 100.0, 2)

    return BenchmarkMetrics(
        total_segments=len(segments),
        scene_count=len(scenes),
        speaker_count=len({seg.speaker_id for seg in segments}),
        speech_level_counts=dict(level_counts),
        emotion_counts=dict(emo_counts),
        honorific_consistency_rate=consistency_rate,
        mismatch_error_count=mismatch_cnt,
        dramatic_shift_count=dramatic_cnt,
        irony_segment_count=irony_cnt,
        off_screen_segment_count=off_screen_cnt,
        ocr_collision_avoided_count=ocr_avoid_cnt,
        emphasized_word_count=emph_cnt,
        strong_emphasis_word_count=strong_emph_cnt,
        paralinguistic_cue_count=cue_cnt,
        mean_semantic_ci=mean_ci,
        mean_harmonic_oi=mean_oi,
        mean_cps=mean_cps,
        cps_compliance_rate=cps_rate,
        cue_attachment_ratio=cue_ratio,
        score_text_intensity=score_text,
        score_dist_5bin=score_dist,
        swer_comparison=swer_comp,
    )
