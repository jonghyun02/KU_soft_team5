"""Vocal vs accompaniment/SFX source separation (Demucs) with immediate VRAM release."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import soundfile as sf

from mos_sdh.core.vram import StageVRAMReport, gpu_stage
from mos_sdh.schemas.document import ParalinguisticCue
from mos_sdh.stage1_audio.media_io import resample_wav_to_mono_16k


@dataclass(slots=True)
class SeparationResult:
    """Separated 16kHz mono vocal stem and detected background audio cues."""

    vocals_16k_wav: Path
    accompaniment_16k_wav: Path | None
    bgm_sfx_cues: list[ParalinguisticCue]


def separate_vocals_and_bgm(
    stereo_44k_wav: Path,
    fallback_mono_16k_wav: Path,
    enable_separation: bool = True,
    telemetry_sink: list[StageVRAMReport] | None = None,
) -> SeparationResult:
    """Isolate speech vocals from drama BGM/SFX using Hybrid Demucs and detect BGM/SFX surges."""
    if not enable_separation:
        return SeparationResult(
            vocals_16k_wav=fallback_mono_16k_wav,
            accompaniment_16k_wav=None,
            bgm_sfx_cues=[],
        )

    work_dir = stereo_44k_wav.parent
    vocals_16k = work_dir / "vocals_16k_mono.wav"
    accomp_16k = work_dir / "accompaniment_16k_mono.wav"

    if not (vocals_16k.exists() and accomp_16k.exists()):
        vocals_44k = work_dir / "vocals_44k.wav"
        accomp_44k = work_dir / "accompaniment_44k.wav"

        with gpu_stage("Stage 1-A: Demucs Source Separation", telemetry_sink=telemetry_sink) as device:
            import torch
            from demucs.apply import apply_model
            from demucs.pretrained import get_model

            wav_np, sr = sf.read(str(stereo_44k_wav), dtype="float32", always_2d=True)
            # Shape: (channels, samples)
            wav_tensor = torch.from_numpy(wav_np.T)
            if wav_tensor.shape[0] == 1:
                wav_tensor = wav_tensor.repeat(2, 1)

            model = get_model("htdemucs")
            model.to(device)
            model.eval()

            ref = wav_tensor.mean(0)
            ref_mean = ref.mean()
            ref_std = ref.std().clamp(min=1e-6)
            norm_wav = (wav_tensor - ref_mean) / ref_std

            with torch.no_grad():
                sources = apply_model(
                    model,
                    norm_wav.unsqueeze(0).to(device),
                    device=device,
                    shifts=0,
                    split=True,
                    overlap=0.25,
                )[0]
                sources = sources * ref_std + ref_mean

            source_names = list(model.sources)
            vocal_idx = source_names.index("vocals")
            vocals_audio = sources[vocal_idx].cpu().numpy().T
            accomp_indices = [idx for idx in range(len(source_names)) if idx != vocal_idx]
            accomp_audio = sources[accomp_indices].sum(dim=0).cpu().numpy().T

            sf.write(str(vocals_44k), vocals_audio, sr, subtype="PCM_16")
            sf.write(str(accomp_44k), accomp_audio, sr, subtype="PCM_16")

            del sources, norm_wav, wav_tensor, model

        resample_wav_to_mono_16k(vocals_44k, vocals_16k)
        resample_wav_to_mono_16k(accomp_44k, accomp_16k)

    bgm_cues = detect_background_sfx_surges(accomp_16k)
    return SeparationResult(
        vocals_16k_wav=vocals_16k,
        accompaniment_16k_wav=accomp_16k,
        bgm_sfx_cues=bgm_cues,
    )


def detect_background_sfx_surges(accompaniment_16k_wav: Path) -> list[ParalinguisticCue]:
    """Detect prominent background sound/BGM surges for SDH non-speech awareness."""
    audio, sr = sf.read(str(accompaniment_16k_wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if len(audio) < sr // 2:
        return []

    win = int(0.25 * sr)
    hop = int(0.125 * sr)
    rms_list: list[float] = []
    times_ms: list[int] = []

    for start in range(0, max(1, len(audio) - win + 1), hop):
        frame = audio[start : start + win]
        rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
        rms_list.append(rms)
        times_ms.append(int(round(start * 1000.0 / sr)))

    rms_arr = np.asarray(rms_list, dtype=np.float32)
    db_arr = 20.0 * np.log10(np.maximum(rms_arr, 1e-6))
    median_db = float(np.median(db_arr))
    std_db = float(max(np.std(db_arr), 1.5))

    # Require both absolute audibility (> -32 dBFS) and salient surge above median background
    active = (db_arr > -32.0) & (db_arr > median_db + max(6.0, 1.8 * std_db))

    cues: list[ParalinguisticCue] = []
    in_run = False
    run_start = 0
    for idx, flag in enumerate(active):
        if flag and not in_run:
            in_run = True
            run_start = idx
        elif not flag and in_run:
            in_run = False
            start_ms = times_ms[run_start]
            end_ms = times_ms[idx - 1] + 250
            if end_ms - start_ms >= 350:
                cues.append(
                    ParalinguisticCue(
                        cue_code="sfx",
                        label_ko="배경 음향·음악 고조",
                        start_ms=start_ms,
                        end_ms=end_ms,
                        confidence=0.75,
                    )
                )
    if in_run:
        start_ms = times_ms[run_start]
        end_ms = times_ms[-1] + 250
        if end_ms - start_ms >= 350:
            cues.append(
                ParalinguisticCue(
                    cue_code="sfx",
                    label_ko="배경 음향·음악 고조",
                    start_ms=start_ms,
                    end_ms=end_ms,
                    confidence=0.75,
                )
            )
    return cues
