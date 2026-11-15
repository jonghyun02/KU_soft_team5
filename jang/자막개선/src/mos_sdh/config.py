"""Central configuration and hyperparameters for the 5-Stage MoS-SDH pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def get_default_cache_dir() -> Path:
    """Return local AppData cache directory outside OneDrive to avoid sync locks."""
    env_dir = os.environ.get("MOS_SDH_CACHE_DIR")
    if env_dir:
        return Path(env_dir)
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "mos-sdh" / "cache"
    return Path.home() / ".cache" / "mos-sdh"


@dataclass(slots=True)
class AnalysisConfig:
    """Configuration parameters across Stages 1 through 5."""

    cache_dir: Path = field(default_factory=get_default_cache_dir)
    whisper_model: str = "large-v3-turbo"
    language: str = "ko"

    # Stage toggles for flexible local GPU budgeting
    enable_source_separation: bool = True
    enable_visual_stage: bool = True
    enable_vlm_stage: bool = True
    enable_llm_stage: bool = True

    # Stage 2: High-density visual tracking, FER & OCR
    visual_sample_fps: float = 5.0
    fer_model: str = "trpakov/vit-face-expression"
    ocr_bottom_zone_ymin: float = 0.72
    lip_active_threshold: float = 0.018

    # Stage 3: Local VLM scene & relationship inference (4-bit quantized)
    vlm_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    # Stage 4: Trimodal CmGI emotion/irony & Korean 6-speech-level consistency
    context_window_ms: int = 3000
    max_context_utterances: int = 2
    dramatic_arousal_z_threshold: float = 1.15
    low_asr_confidence_threshold: float = 0.72
    irony_threshold: float = 0.50

    # Stage 5: Local LLM multi-candidate generator & de Rivero / Sun ranker/pruner
    llm_model: str = "Qwen/Qwen2.5-3B-Instruct"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    candidate_count: int = 4
    semantic_min_ci: float = 0.80
    recommended_max_cps: float = 14.0
    max_cue_density_ratio: float = 0.45

    # Graave et al. (2024): Utterance segmentation & noise filtering constraints
    max_utterance_sec: float = 30.0
    min_utterance_sec: float = 0.25

    # Tokuyama et al. (2021) & Park & Hong (2025): 5-level acoustic emphasis weights & Z-thresholds
    weight_pitch: float = 0.40
    weight_energy: float = 0.40
    weight_duration: float = 0.20
    emphasis_z_thresholds: tuple[float, float, float, float] = (0.8, 1.3, 1.8, 2.4)

    # Speaker diarization
    max_speakers: int = 6
    diarization_distance_threshold: float = 0.38
