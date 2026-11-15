"""Word-aligned Faster-Whisper STT engine with disfluency preservation and VRAM offloading."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import soundfile as sf

from mos_sdh.config import AnalysisConfig
from mos_sdh.core.vram import StageVRAMReport, bind_windows_cuda_dlls, gpu_stage
from mos_sdh.schemas.document import SegmentMoS, WordUnit
from mos_sdh.stage1_audio.diarization_vad import DiarizedRegion, assign_speaker_by_overlap


def transcribe_with_word_alignment(
    vocals_16k_wav: Path,
    diarized_regions: list[DiarizedRegion],
    config: AnalysisConfig,
    telemetry_sink: list[StageVRAMReport] | None = None,
) -> list[SegmentMoS]:
    """Run Faster-Whisper with word-level timestamps and map turns to diarized speakers."""
    bind_windows_cuda_dlls()

    with gpu_stage("Stage 1-B: Faster-Whisper Word-Aligned STT", telemetry_sink=telemetry_sink) as device:
        from faster_whisper import WhisperModel

        compute_type = "float16" if device == "cuda" else "int8"
        try:
            model = WhisperModel(config.whisper_model, device=device, compute_type=compute_type)
        except Exception:
            # Graceful fallback to CPU int8 if CUDA runtime fails on an unsupported driver
            device = "cpu"
            model = WhisperModel(config.whisper_model, device="cpu", compute_type="int8")

        # Load 16kHz mono float32 waveform directly via soundfile (bypasses PyAV container API changes & re-decoding)
        audio_np, sr = sf.read(str(vocals_16k_wav), dtype="float32")
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        if sr != 16000:
            import librosa

            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
        audio_np = np.ascontiguousarray(audio_np, dtype=np.float32)

        segments_iter, _info = model.transcribe(
            audio_np,
            language=config.language,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 300,
                "max_speech_duration_s": config.max_utterance_sec,
            },
            condition_on_previous_text=False,
        )

        results: list[SegmentMoS] = []
        for idx, seg in enumerate(segments_iter, start=1):
            raw_text = seg.text.strip()
            if not raw_text:
                continue

            seg_start_ms = max(0, int(round(seg.start * 1000.0)))
            seg_end_ms = max(seg_start_ms + 50, int(round(seg.end * 1000.0)))

            word_units: list[WordUnit] = []
            probs: list[float] = []
            if seg.words:
                for w in seg.words:
                    w_text = w.word.strip()
                    if not w_text:
                        continue
                    w_start = max(seg_start_ms, int(round(w.start * 1000.0)))
                    w_end = max(w_start + 30, int(round(w.end * 1000.0)))
                    prob = float(getattr(w, "probability", 1.0))
                    probs.append(prob)
                    word_units.append(
                        WordUnit(
                            text=w_text,
                            start_ms=w_start,
                            end_ms=w_end,
                            confidence=round(prob, 4),
                        )
                    )

            if not word_units:
                avg_conf = 1.0
                spk = assign_speaker_by_overlap(seg_start_ms, seg_end_ms, diarized_regions)
                results.append(
                    SegmentMoS(
                        segment_id=len(results) + 1,
                        start_ms=seg_start_ms,
                        end_ms=seg_end_ms,
                        speaker_id=spk,
                        raw_transcript=raw_text,
                        asr_avg_confidence=avg_conf,
                        words=[],
                    )
                )
                continue

            # Split multi-sentence Whisper segments at sentence-ending punctuation or >=450ms pauses for SDH readability
            sub_word_groups: list[list[WordUnit]] = []
            current_group: list[WordUnit] = []
            for w_idx, wu in enumerate(word_units):
                current_group.append(wu)
                is_last = w_idx == len(word_units) - 1
                if is_last:
                    sub_word_groups.append(current_group)
                    break
                next_wu = word_units[w_idx + 1]
                gap_ms = next_wu.start_ms - wu.end_ms
                ends_sentence = wu.text.endswith((".", "?", "!")) and len(current_group) >= 2
                if ends_sentence or gap_ms >= 450:
                    sub_word_groups.append(current_group)
                    current_group = []

            for grp in sub_word_groups:
                g_start = grp[0].start_ms
                g_end = max(g_start + 100, grp[-1].end_ms)
                g_text = " ".join(w.text for w in grp).strip()
                g_conf = round(sum(w.confidence for w in grp) / len(grp), 4)
                spk = assign_speaker_by_overlap(g_start, g_end, diarized_regions)
                results.append(
                    SegmentMoS(
                        segment_id=len(results) + 1,
                        start_ms=g_start,
                        end_ms=g_end,
                        speaker_id=spk,
                        raw_transcript=g_text,
                        asr_avg_confidence=g_conf,
                        words=grp,
                    )
                )

        del model

    return results
