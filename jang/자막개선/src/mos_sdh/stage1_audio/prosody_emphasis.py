"""STFT/Log-Mel frontend (Park & Hong, 2025) & 5-level Emph-1-Token prosody extractor (Tokuyama et al., 2021)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
import librosa
import numpy as np
import parselmouth
import soundfile as sf

from mos_sdh.config import AnalysisConfig
from mos_sdh.schemas.document import SegmentMoS

_HANGUL_SYLLABLE_RE = re.compile(r"[가-힣A-Za-z0-9]")


def compute_log_mel_spectrogram(
    waveform: np.ndarray,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 160,
    n_mels: int = 80,
    eps: float = 1e-10,
) -> np.ndarray:
    """Implement Park & Hong (2025, Eq. 1-3): STFT -> Power Spectrum -> Mel Filterbank -> Log-Mel."""
    # Eq. (1): Short-Time Fourier Transform X(tau, omega)
    stft_complex = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length, win_length=400, window="hann")
    power_spec = np.abs(stft_complex) ** 2

    # Eq. (2): S_Mel = H * |X(tau, omega)|^2
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    s_mel = np.dot(mel_basis, power_spec)

    # Eq. (3): S_LogMel = log(S_Mel + eps)
    s_log_mel = np.log(s_mel + eps)
    return s_log_mel


def quantize_emphasis_level(
    z_score: float,
    thresholds: tuple[float, float, float, float] = (0.8, 1.3, 1.8, 2.4),
) -> int:
    """Map composite prosodic Z-score into Tokuyama et al. (2021) 5-level emphasis e_i in {0..4}."""
    t1, t2, t3, t4 = thresholds
    if z_score < t1:
        return 0
    if z_score < t2:
        return 1
    if z_score < t3:
        return 2
    if z_score < t4:
        return 3
    return 4


def annotate_prosody_and_emphasis(
    vocals_16k_wav: Path,
    segments: list[SegmentMoS],
    config: AnalysisConfig,
) -> list[SegmentMoS]:
    """Extract F0, intensity, and syllable duration via Praat/Parselmouth and assign <toe_k> emphasis tokens."""
    if not segments:
        return segments

    audio_np, sr = sf.read(str(vocals_16k_wav), dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)

    snd = parselmouth.Sound(audio_np, sampling_frequency=float(sr))
    pitch_obj = snd.to_pitch_ac(time_step=0.01, pitch_floor=70.0, pitch_ceiling=500.0)
    intensity_obj = snd.to_intensity(minimum_pitch=70.0, time_step=0.01)

    pitch_xs = pitch_obj.xs()
    pitch_vals = pitch_obj.selected_array["frequency"]
    int_xs = intensity_obj.xs()
    int_vals = intensity_obj.values[0]

    # Collect per-speaker raw distributions for speaker-normalized Z-scores (Graave et al., 2024)
    spk_pitches: dict[str, list[float]] = defaultdict(list)
    spk_energies: dict[str, list[float]] = defaultdict(list)
    spk_durations: dict[str, list[float]] = defaultdict(list)

    for seg in segments:
        # Synthesize uniform word slices if segments came from external SRT without word alignment
        if not seg.words and seg.raw_transcript.strip():
            tokens = seg.raw_transcript.strip().split()
            span = max(100, seg.end_ms - seg.start_ms)
            step = span / max(1, len(tokens))
            from mos_sdh.schemas.document import WordUnit

            seg.words = [
                WordUnit(
                    text=tok,
                    start_ms=seg.start_ms + int(round(i * step)),
                    end_ms=seg.start_ms + int(round((i + 1) * step)),
                )
                for i, tok in enumerate(tokens)
            ]

        s_sec = seg.start_ms / 1000.0
        e_sec = seg.end_ms / 1000.0
        seg_p_mask = (pitch_xs >= s_sec) & (pitch_xs <= e_sec) & (pitch_vals > 0.0)
        seg_i_mask = (int_xs >= s_sec) & (int_xs <= e_sec) & np.isfinite(int_vals)

        voiced_f0 = pitch_vals[seg_p_mask]
        valid_int = int_vals[seg_i_mask]

        seg.f0_mean_hz = round(float(np.mean(voiced_f0)), 2) if len(voiced_f0) > 0 else 0.0
        seg.f0_std_hz = round(float(np.std(voiced_f0)), 2) if len(voiced_f0) > 1 else 0.0
        seg.energy_mean_db = round(float(np.mean(valid_int)), 2) if len(valid_int) > 0 else 0.0

        for w in seg.words:
            ws_sec = w.start_ms / 1000.0
            we_sec = max(ws_sec + 0.02, w.end_ms / 1000.0)

            wp_mask = (pitch_xs >= ws_sec) & (pitch_xs <= we_sec) & (pitch_vals > 0.0)
            wi_mask = (int_xs >= ws_sec) & (int_xs <= we_sec) & np.isfinite(int_vals)

            w_f0 = pitch_vals[wp_mask]
            w_int = int_vals[wi_mask]

            # Peak-weighted mean captures acute emphatic pitch/intensity accents within a word
            if len(w_f0) > 0:
                w.pitch_hz = round(float(0.6 * np.max(w_f0) + 0.4 * np.mean(w_f0)), 2)
                spk_pitches[seg.speaker_id].append(w.pitch_hz)
            else:
                w.pitch_hz = seg.f0_mean_hz

            if len(w_int) > 0:
                w.energy_db = round(float(0.6 * np.max(w_int) + 0.4 * np.mean(w_int)), 2)
                spk_energies[seg.speaker_id].append(w.energy_db)
            else:
                w.energy_db = seg.energy_mean_db

            syl_count = max(1, len(_HANGUL_SYLLABLE_RE.findall(w.text)))
            w.duration_per_syllable_ms = round((w.end_ms - w.start_ms) / syl_count, 2)
            spk_durations[seg.speaker_id].append(w.duration_per_syllable_ms)

    # Compute robust speaker baseline stats (Median + scaled MAD prevents single emphatic screams from masking Z-scores)
    spk_stats: dict[str, tuple[float, float, float, float, float, float]] = {}
    for spk in {s.speaker_id for s in segments}:
        p_arr = np.asarray(spk_pitches[spk] or [150.0], dtype=np.float32)
        e_arr = np.asarray(spk_energies[spk] or [60.0], dtype=np.float32)
        d_arr = np.asarray(spk_durations[spk] or [180.0], dtype=np.float32)

        med_p = float(np.median(p_arr))
        mad_p = float(1.4826 * np.median(np.abs(p_arr - med_p)))
        med_e = float(np.median(e_arr))
        mad_e = float(1.4826 * np.median(np.abs(e_arr - med_e)))
        med_d = float(np.median(d_arr))
        mad_d = float(1.4826 * np.median(np.abs(d_arr - med_d)))

        spk_stats[spk] = (
            med_p,
            max(mad_p, 18.0),
            med_e,
            max(mad_e, 4.0),
            med_d,
            max(mad_d, 55.0),
        )

    # Assign composite Z-score, discrete level e_i in {0..4}, <toe_k> tagged transcript, and segment arousal
    for seg in segments:
        mu_p, sd_p, mu_e, sd_e, mu_d, sd_d = spk_stats[seg.speaker_id]

        seg_z_p = (seg.f0_mean_hz - mu_p) / sd_p if seg.f0_mean_hz > 0 else 0.0
        seg_z_e = (seg.energy_mean_db - mu_e) / sd_e if seg.energy_mean_db > 0 else 0.0
        seg.arousal_zscore = round(float(max(seg_z_p, seg_z_e, 0.5 * (seg_z_p + seg_z_e))), 3)

        tagged_parts: list[str] = []
        for w in seg.words:
            zp = (w.pitch_hz - mu_p) / sd_p if w.pitch_hz > 0 else 0.0
            ze = (w.energy_db - mu_e) / sd_e if w.energy_db > 0 else 0.0
            zd = (w.duration_per_syllable_ms - mu_d) / sd_d

            z_emph = config.weight_pitch * zp + config.weight_energy * ze + config.weight_duration * zd
            w.z_emphasis = round(float(z_emph), 3)
            w.emphasis_level = quantize_emphasis_level(w.z_emphasis, config.emphasis_z_thresholds)
            if w.emphasis_level > 0:
                w.emphasis_token = f"<toe_{w.emphasis_level}>"
                tagged_parts.append(f"{w.emphasis_token} {w.text}")
            else:
                w.emphasis_token = None
                tagged_parts.append(w.text)

        seg.tagged_transcript = " ".join(tagged_parts) if tagged_parts else seg.raw_transcript

    return segments
