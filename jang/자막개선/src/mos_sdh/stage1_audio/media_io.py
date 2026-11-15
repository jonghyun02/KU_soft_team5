"""FFmpeg media audio extraction and resampling utilities."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
import soundfile as sf


@dataclass(slots=True)
class ExtractedAudioPaths:
    """Paths to standardized WAV files stored in the local non-OneDrive cache."""

    stereo_44k_wav: Path
    mono_16k_wav: Path
    duration_ms: int


def extract_standardized_audio(input_media: Path, cache_dir: Path) -> ExtractedAudioPaths:
    """Extract 44.1kHz stereo WAV and 16kHz mono PCM WAV from video or audio input via FFmpeg."""
    if not input_media.exists():
        raise FileNotFoundError(f"입력 미디어 파일을 찾을 수 없습니다: {input_media}")

    stat = input_media.stat()
    fingerprint = hashlib.sha256(
        f"{input_media.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    ).hexdigest()[:16]

    work_dir = cache_dir / f"media_{fingerprint}"
    work_dir.mkdir(parents=True, exist_ok=True)

    stereo_44k = work_dir / "source_44k_stereo.wav"
    mono_16k = work_dir / "source_16k_mono.wav"

    if not stereo_44k.exists():
        _run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(input_media),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "44100",
                "-ac",
                "2",
                str(stereo_44k),
            ]
        )

    if not mono_16k.exists():
        _run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(stereo_44k),
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(mono_16k),
            ]
        )

    info = sf.info(str(mono_16k))
    duration_ms = int(round(info.duration * 1000.0))
    return ExtractedAudioPaths(
        stereo_44k_wav=stereo_44k,
        mono_16k_wav=mono_16k,
        duration_ms=duration_ms,
    )


def resample_wav_to_mono_16k(src_wav: Path, dst_wav: Path) -> Path:
    """Convert any separated WAV stem into 16kHz mono PCM WAV for STT and prosody analysis."""
    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src_wav),
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(dst_wav),
        ]
    )
    return dst_wav


def _run_ffmpeg(cmd: list[str]) -> None:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg 실행 실패 (exit={proc.returncode}):\n{proc.stderr[-800:]}")
