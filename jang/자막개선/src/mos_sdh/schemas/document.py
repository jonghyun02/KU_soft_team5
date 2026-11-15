"""Pydantic v2 data contracts shared across Stages 1 through 5 of MoS-SDH Studio."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

SpeechLevelType = Literal[
    "합쇼체",
    "해요체",
    "하오체",
    "하게체",
    "반말체",
    "해라체",
    "미분류",
]

ConsistencyStatus = Literal[
    "consistent",
    "mismatch_error",
    "dramatic_shift",
    "insufficient_context",
]

CueCode = Literal["b/", "l/", "c/", "o/", "sfx"]

EmotionLabel = Literal["분노", "슬픔", "기쁨", "공포", "냉소", "중립"]

SpeakerZone = Literal["left", "center", "right"]


class WordUnit(BaseModel):
    """Word-level alignment and acoustic emphasis features (Tokuyama et al., 2021; Park & Hong, 2025)."""

    text: str
    start_ms: int
    end_ms: int
    confidence: float = 1.0
    pitch_hz: float = 0.0
    energy_db: float = 0.0
    duration_per_syllable_ms: float = 0.0
    z_emphasis: float = 0.0
    emphasis_level: int = Field(default=0, ge=0, le=4)
    emphasis_token: str | None = None


class HonorificAnalysis(BaseModel):
    """Korean 3-axis honorifics and discourse speech-level consistency (Hwang et al., 2021; Jeong et al., 2021)."""

    subject_honorific: bool = False
    object_honorific: bool = False
    speech_level: SpeechLevelType = "미분류"
    is_polite_group: bool | None = None
    ending_morphemes: list[str] = Field(default_factory=list)
    expected_speech_level: SpeechLevelType | None = None
    consistency_status: ConsistencyStatus = "insufficient_context"
    context_segment_ids: list[int] = Field(default_factory=list)
    arousal_zscore: float = 0.0
    suggested_correction: str | None = None


class ParalinguisticCue(BaseModel):
    """Non-speech and paralinguistic cues (Bang et al., 2020 KsponSpeech taxonomy)."""

    cue_code: CueCode
    label_ko: str
    start_ms: int
    end_ms: int
    confidence: float = 0.8


class DisfluencyStats(BaseModel):
    """Spontaneous speech disfluency metrics (Kim et al., 2024 LearnerVoice L2S adaptation)."""

    token_count: int = 0
    filler_count: int = 0
    self_repair_count: int = 0
    fw_density: float = 0.0
    sr_density: float = 0.0
    preserve_disfluency: bool = True
    cleaned_text: str | None = None


class OCRBox(BaseModel):
    """Normalized [0..1] bounding box and text recognized by Stage 2 Scene OCR."""

    text: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float = 0.9


class VisualContextSlot(BaseModel):
    """Stage 2 high-density visual tracking, lip ASD, facial expression, and OCR metadata."""

    scene_id: int = 1
    face_detected: bool = False
    on_screen_speaker: bool | None = None
    speaker_zone: SpeakerZone = "center"
    lip_activity_score: float = 0.0
    facial_expression: str | None = None
    expression_probs: dict[str, float] = Field(default_factory=dict)
    ocr_boxes: list[OCRBox] = Field(default_factory=list)
    ocr_texts: list[str] = Field(default_factory=list)
    avoid_bottom_region: bool = False


class SpeakerRelation(BaseModel):
    """Directional social relationship & speech-level prior inferred by Stage 3 VLM."""

    from_speaker: str
    to_speaker: str
    relation_ko: str
    prior_speech_level: SpeechLevelType = "해요체"
    confidence: float = 0.7


class SceneContext(BaseModel):
    """Stage 3 scene boundary and local VLM situational/social relationship summary."""

    scene_id: int
    start_ms: int
    end_ms: int
    keyframe_ms: int
    description_ko: str = ""
    ocr_summary: list[str] = Field(default_factory=list)
    inferred_relations: list[SpeakerRelation] = Field(default_factory=list)


class EmotionIronyAnalysis(BaseModel):
    """Stage 4 Trimodal Temporal-Aware Gated Fusion (CmGI) emotion & irony metrics (Gao et al., 2024)."""

    dominant_emotion: EmotionLabel = "중립"
    emotion_probs: dict[str, float] = Field(default_factory=dict)
    lexical_polarity: float = 0.0
    audiovisual_polarity: float = 0.0
    irony_score: float = 0.0
    irony_detected: bool = False
    gate_weights_mean: dict[str, float] = Field(default_factory=dict)


class SubtitleCandidate(BaseModel):
    """Stage 5 multi-candidate SDH option scored by semantic preservation C_i & harmonic O_i (de Rivero et al., 2021)."""

    candidate_id: int
    strategy: str
    srt_text: str
    ass_text: str
    plain_dialogue: str
    semantic_ci: float = 1.0
    mos_fidelity: float = 1.0
    harmonic_oi: float = 1.0
    passed_guard: bool = True
    selected: bool = False


class SegmentMoS(BaseModel):
    """Single subtitle segment annotated across all 5 multimodal MoS stages."""

    segment_id: int
    start_ms: int
    end_ms: int
    speaker_id: str = "화자_1"
    raw_transcript: str
    existing_subtitle: str | None = None
    tagged_transcript: str = ""
    sdh_srt_text: str = ""
    sdh_ass_text: str = ""
    stage_direction_pruned: bool = False
    f0_mean_hz: float = 0.0
    f0_std_hz: float = 0.0
    energy_mean_db: float = 0.0
    arousal_zscore: float = 0.0
    asr_avg_confidence: float = 1.0
    cps: float = 0.0
    cps_exceeded: bool = False
    words: list[WordUnit] = Field(default_factory=list)
    honorific: HonorificAnalysis = Field(default_factory=HonorificAnalysis)
    emotion_irony: EmotionIronyAnalysis = Field(default_factory=EmotionIronyAnalysis)
    cues: list[ParalinguisticCue] = Field(default_factory=list)
    disfluency: DisfluencyStats = Field(default_factory=DisfluencyStats)
    visual: VisualContextSlot = Field(default_factory=VisualContextSlot)
    candidates: list[SubtitleCandidate] = Field(default_factory=list)


class SWERComparison(BaseModel):
    """Comparison metrics against an existing or reference subtitle (Bang et al., 2020)."""

    cer_percent: float
    wer_percent: float
    swer_percent: float
    compared_utterances: int


class BenchmarkMetrics(BaseModel):
    """Global episode/clip-level MoS summary metrics (Sun et al., 2024; Hwang et al., 2021; Gao et al., 2024)."""

    total_segments: int = 0
    scene_count: int = 0
    speaker_count: int = 0
    speech_level_counts: dict[str, int] = Field(default_factory=dict)
    emotion_counts: dict[str, int] = Field(default_factory=dict)
    honorific_consistency_rate: float = 100.0
    mismatch_error_count: int = 0
    dramatic_shift_count: int = 0
    irony_segment_count: int = 0
    off_screen_segment_count: int = 0
    ocr_collision_avoided_count: int = 0
    emphasized_word_count: int = 0
    strong_emphasis_word_count: int = 0
    paralinguistic_cue_count: int = 0
    mean_semantic_ci: float = 1.0
    mean_harmonic_oi: float = 1.0
    mean_cps: float = 0.0
    cps_compliance_rate: float = 100.0
    cue_attachment_ratio: float = 0.0
    score_text_intensity: float = 0.0
    score_dist_5bin: list[float] = Field(default_factory=list)
    swer_comparison: SWERComparison | None = None


class StageVRAMTelemetry(BaseModel):
    """VRAM usage snapshot per pipeline stage."""

    stage_name: str
    device: str
    gpu_name: str
    allocated_before_mb: float
    peak_allocated_mb: float
    allocated_after_mb: float


class MoSDocument(BaseModel):
    """Root JSON output schema of MoS-SDH Studio."""

    version: str = "1.0.0"
    mode: Literal["mode_a_video_only", "mode_b_subtitle_qa"]
    input_media_path: str
    input_subtitle_path: str | None = None
    duration_ms: int = 0
    scenes: list[SceneContext] = Field(default_factory=list)
    segments: list[SegmentMoS] = Field(default_factory=list)
    metrics: BenchmarkMetrics = Field(default_factory=BenchmarkMetrics)
    vram_telemetry: list[StageVRAMTelemetry] = Field(default_factory=list)
