"""Zero-auth VAD, acoustic speaker diarization, and non-speech vocal cue (b/, l/) detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

from mos_sdh.config import AnalysisConfig
from mos_sdh.schemas.document import ParalinguisticCue


@dataclass(slots=True)
class DiarizedRegion:
    """VAD speech interval tagged with an estimated speaker label."""

    start_ms: int
    end_ms: int
    speaker_id: str


def run_vad_and_diarization(
    vocals_16k_wav: Path,
    config: AnalysisConfig,
) -> tuple[list[DiarizedRegion], list[ParalinguisticCue]]:
    """Detect speech turns (enforcing <=30s Graave et al. rule), cluster speakers, and spot sighs/laughs."""
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    audio_np, sr = sf.read(str(vocals_16k_wav), dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)

    vad_model = load_silero_vad()
    wav_tensor = torch.from_numpy(audio_np)
    raw_stamps = get_speech_timestamps(
        wav_tensor,
        vad_model,
        sampling_rate=sr,
        min_speech_duration_ms=int(config.min_utterance_sec * 1000),
        max_speech_duration_s=config.max_utterance_sec,
        min_silence_duration_ms=250,
    )

    speech_intervals: list[tuple[int, int]] = []
    for item in raw_stamps:
        s_ms = int(round(item["start"] * 1000.0 / sr))
        e_ms = int(round(item["end"] * 1000.0 / sr))
        if e_ms - s_ms >= int(config.min_utterance_sec * 1000):
            speech_intervals.append((s_ms, e_ms))

    speaker_ids = _cluster_speakers_acoustic(audio_np, sr, speech_intervals, config)
    regions = [
        DiarizedRegion(start_ms=s_ms, end_ms=e_ms, speaker_id=spk)
        for (s_ms, e_ms), spk in zip(speech_intervals, speaker_ids, strict=True)
    ]

    non_speech_cues = _detect_vocal_non_speech_cues(audio_np, sr, speech_intervals)
    return regions, non_speech_cues


def assign_speaker_by_overlap(
    seg_start_ms: int,
    seg_end_ms: int,
    regions: list[DiarizedRegion],
) -> str:
    """Assign speaker_id to a transcription segment via maximum temporal overlap."""
    if not regions:
        return "화자_1"

    best_spk = regions[0].speaker_id
    best_overlap = -1
    for reg in regions:
        overlap = max(0, min(seg_end_ms, reg.end_ms) - max(seg_start_ms, reg.start_ms))
        if overlap > best_overlap:
            best_overlap = overlap
            best_spk = reg.speaker_id

    if best_overlap == 0:
        # Pick nearest region in time
        mid = 0.5 * (seg_start_ms + seg_end_ms)
        nearest = min(regions, key=lambda r: abs(0.5 * (r.start_ms + r.end_ms) - mid))
        return nearest.speaker_id
    return best_spk


def _cluster_speakers_acoustic(
    audio: np.ndarray,
    sr: int,
    intervals_ms: list[tuple[int, int]],
    config: AnalysisConfig,
) -> list[str]:
    """Cluster speech intervals using normalized MFCC + spectral + F0 summary embeddings."""
    if len(intervals_ms) <= 1:
        return ["화자_1" for _ in intervals_ms]

    embeddings: list[np.ndarray] = []
    for s_ms, e_ms in intervals_ms:
        s_idx = max(0, int(s_ms * sr / 1000))
        e_idx = min(len(audio), int(e_ms * sr / 1000))
        clip = audio[s_idx:e_idx]
        if len(clip) < int(0.15 * sr):
            clip = np.pad(clip, (0, int(0.15 * sr) - len(clip)))

        mfcc = librosa.feature.mfcc(y=clip, sr=sr, n_mfcc=20)
        centroid = librosa.feature.spectral_centroid(y=clip, sr=sr)
        f0 = librosa.yin(clip, fmin=65, fmax=450, sr=sr)
        voiced_f0 = f0[(f0 > 70.0) & (f0 < 430.0)]
        f0_med = float(np.median(voiced_f0)) / 300.0 if len(voiced_f0) > 0 else 0.5
        f0_iqr = float(np.percentile(voiced_f0, 75) - np.percentile(voiced_f0, 25)) / 150.0 if len(voiced_f0) > 2 else 0.1

        feat = np.concatenate(
            [
                np.mean(mfcc, axis=1) / 50.0,
                np.std(mfcc, axis=1) / 25.0,
                [float(np.mean(centroid)) / 4000.0, f0_med, f0_iqr],
            ]
        )
        norm = np.linalg.norm(feat) + 1e-8
        embeddings.append(feat / norm)

    emb_matrix = np.vstack(embeddings)
    dist_matrix = cosine_distances(emb_matrix)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=config.diarization_distance_threshold,
    )
    raw_labels = clustering.fit_predict(dist_matrix)

    # Remap cluster IDs in chronological order of first appearance (화자_1, 화자_2, ...)
    ordered_map: dict[int, str] = {}
    speaker_names: list[str] = []
    for lbl in raw_labels:
        lbl_int = int(lbl)
        if lbl_int not in ordered_map:
            idx = min(len(ordered_map) + 1, config.max_speakers)
            ordered_map[lbl_int] = f"화자_{idx}"
        speaker_names.append(ordered_map[lbl_int])
    return speaker_names


def _detect_vocal_non_speech_cues(
    audio: np.ndarray,
    sr: int,
    speech_intervals_ms: list[tuple[int, int]],
) -> list[ParalinguisticCue]:
    """Scan inter-speech vocal gaps for audible breath/sigh (b/) and rhythmic laughter (l/)."""
    total_ms = int(round(len(audio) * 1000.0 / sr))
    gaps: list[tuple[int, int]] = []
    cursor = 0
    for s_ms, e_ms in speech_intervals_ms:
        if s_ms - cursor >= 350:
            gaps.append((cursor, s_ms))
        cursor = max(cursor, e_ms)
    if total_ms - cursor >= 350:
        gaps.append((cursor, total_ms))

    cues: list[ParalinguisticCue] = []
    for g_start, g_end in gaps:
        s_idx = int(g_start * sr / 1000)
        e_idx = int(g_end * sr / 1000)
        gap_wav = audio[s_idx:e_idx]
        if len(gap_wav) < int(0.35 * sr):
            continue

        win = int(0.35 * sr)
        hop = int(0.20 * sr)
        for offset in range(0, len(gap_wav) - win + 1, hop):
            sub = gap_wav[offset : offset + win]
            rms_db = 20.0 * float(np.log10(np.sqrt(np.mean(sub * sub)) + 1e-7))
            if rms_db < -38.0:
                continue

            flatness = float(np.mean(librosa.feature.spectral_flatness(y=sub)))
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=sub)))

            c_start = g_start + int(round(offset * 1000.0 / sr))
            c_end = c_start + 350

            # Avoid duplicate overlapping cues
            if cues and c_start - cues[-1].end_ms < 250:
                continue

            if flatness > 0.08 and zcr > 0.09 and -38.0 <= rms_db <= -16.0:
                cues.append(
                    ParalinguisticCue(
                        cue_code="b/",
                        label_ko="한숨·숨소리",
                        start_ms=c_start,
                        end_ms=c_end,
                        confidence=0.72,
                    )
                )
            elif rms_db > -28.0 and flatness < 0.06:
                env = np.abs(sub)
                env_frames = librosa.feature.rms(y=env, frame_length=512, hop_length=160)[0]
                if len(env_frames) > 4 and float(np.std(env_frames) / (np.mean(env_frames) + 1e-6)) > 0.55:
                    cues.append(
                        ParalinguisticCue(
                            cue_code="l/",
                            label_ko="웃음",
                            start_ms=c_start,
                            end_ms=c_end,
                            confidence=0.70,
                        )
                    )
    return cues
