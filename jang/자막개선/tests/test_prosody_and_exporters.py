"""Unit tests for STFT/Log-Mel, Parselmouth 5-level emphasis, multi-window scores, and SDH exporters."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import soundfile as sf

from mos_sdh.config import AnalysisConfig
from mos_sdh.exporters.report_exporter import export_reports
from mos_sdh.exporters.subtitle_exporter import export_subtitles
from mos_sdh.schemas.document import (
    BenchmarkMetrics,
    HonorificAnalysis,
    MoSDocument,
    ParalinguisticCue,
    SegmentMoS,
    VisualContextSlot,
    WordUnit,
)
from mos_sdh.stage1_audio.prosody_emphasis import (
    annotate_prosody_and_emphasis,
    compute_log_mel_spectrogram,
)
from mos_sdh.stage4_mos.disfluency_metrics import compute_multi_window_mos_distribution


def _synth_voiced_tone(sr: int, duration_s: float, f0_hz: float, amplitude: float) -> np.ndarray:
    t = np.linspace(0.0, duration_s, int(sr * duration_s), endpoint=False)
    # Rich harmonic glottal-like wave so Praat autocorrelation pitch tracker locks onto F0 cleanly
    sig = (
        0.60 * np.sin(2 * np.pi * f0_hz * t)
        + 0.28 * np.sin(2 * np.pi * 2 * f0_hz * t)
        + 0.12 * np.sin(2 * np.pi * 3 * f0_hz * t)
    )
    return (amplitude * sig).astype(np.float32)


def test_log_mel_spectrogram_shape_and_finiteness() -> None:
    sr = 16000
    wav = _synth_voiced_tone(sr, 1.0, 150.0, 0.2)
    log_mel = compute_log_mel_spectrogram(wav, sr=sr, n_mels=80)
    assert log_mel.shape[0] == 80
    assert log_mel.shape[1] > 90
    assert np.all(np.isfinite(log_mel))


def test_parselmouth_emphasis_assigns_toe_token_to_prominent_word(tmp_path: Path) -> None:
    sr = 16000
    # 3 calm words (F0=135Hz, amp=0.12, 0.4s) + 1 strongly emphasized word (F0=290Hz, amp=0.85, 0.8s)
    calm = _synth_voiced_tone(sr, 0.40, 135.0, 0.12)
    gap = np.zeros(int(0.05 * sr), dtype=np.float32)
    emph = _synth_voiced_tone(sr, 0.80, 290.0, 0.85)

    full_audio = np.concatenate([calm, gap, calm, gap, emph, gap, calm])
    wav_path = tmp_path / "emph_test.wav"
    sf.write(str(wav_path), full_audio, sr)

    seg = SegmentMoS(
        segment_id=1,
        start_ms=0,
        end_ms=int(len(full_audio) * 1000 / sr),
        speaker_id="화자_1",
        raw_transcript="나는 오늘 정말 화났다",
        words=[
            WordUnit(text="나는", start_ms=0, end_ms=400),
            WordUnit(text="오늘", start_ms=450, end_ms=850),
            WordUnit(text="정말", start_ms=900, end_ms=1700),
            WordUnit(text="화났다", start_ms=1750, end_ms=2150),
        ],
    )

    cfg = AnalysisConfig()
    out = annotate_prosody_and_emphasis(wav_path, [seg], cfg)
    words = out[0].words

    assert words[2].pitch_hz > words[0].pitch_hz + 100.0
    assert words[2].energy_db > words[0].energy_db + 10.0
    assert words[2].emphasis_level >= 3
    assert words[2].emphasis_token in {"<toe_3>", "<toe_4>"}
    assert words[0].emphasis_level == 0
    assert "<toe_" in out[0].tagged_transcript


def test_dual_sdh_exporters_and_multi_window_distribution(tmp_path: Path) -> None:
    seg = SegmentMoS(
        segment_id=1,
        start_ms=500,
        end_ms=3000,
        speaker_id="화자_1",
        raw_transcript="참 잘 하십니다",
        tagged_transcript="참 <toe_4> 잘 하십니다",
        arousal_zscore=1.7,
        words=[
            WordUnit(text="참", start_ms=500, end_ms=900, emphasis_level=0),
            WordUnit(text="잘", start_ms=950, end_ms=1800, emphasis_level=4, emphasis_token="<toe_4>"),
            WordUnit(text="하십니다", start_ms=1850, end_ms=3000, emphasis_level=0),
        ],
        honorific=HonorificAnalysis(
            subject_honorific=True,
            speech_level="합쇼체",
            is_polite_group=True,
            consistency_status="dramatic_shift",
        ),
        cues=[ParalinguisticCue(cue_code="b/", label_ko="한숨", start_ms=2800, end_ms=3100)],
        visual=VisualContextSlot(avoid_bottom_region=True),
    )

    score_text, score_dist = compute_multi_window_mos_distribution([seg], num_bins=5)
    assert score_text > 0.0
    assert len(score_dist) == 5

    doc = MoSDocument(
        mode="mode_a_video_only",
        input_media_path="sample.mp4",
        duration_ms=3500,
        segments=[seg],
        metrics=BenchmarkMetrics(
            total_segments=1,
            speaker_count=1,
            speech_level_counts={"합쇼체": 1},
            score_text_intensity=score_text,
            score_dist_5bin=score_dist,
        ),
    )

    subs = export_subtitles(doc, tmp_path)
    reps = export_reports(doc, tmp_path)

    srt_content = subs["srt"].read_text(encoding="utf-8")
    ass_content = subs["ass"].read_text(encoding="utf-8")
    md_content = reps["markdown"].read_text(encoding="utf-8")

    assert "[화자_1]" in srt_content
    assert "'잘'" in srt_content
    assert "(한숨)" in srt_content
    assert r"{\an8}" in ass_content
    assert r"\fscx125\fscy125" in ass_content
    assert "Score_dist" in md_content
