"""arXiv:2309.11093 기반 노래(가사·보컬) 싱어블 번역 모델 및 오디오 코덱의 하이퍼파라미터 설정 모듈."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass
class LyricTransConfig:
    """섹션·음절 조건부 멀티스트림 트랜스포머, Depformer, 코덱 및 싱어빌리티 제약 규격 설정 클래스입니다."""

    source_lang: str = "ko"
    target_lang: str = "en"

    sample_rate: int = 24000
    frame_rate: float = 12.5
    audio_card: int = 2048
    num_codebooks_per_stream: int = 8
    codec_dim: int = 512
    codec_vq_dim: int = 256
    codec_strides: tuple[int, ...] = (8, 6, 5, 4, 2)

    text_card: int = 32000
    text_eos_id: int = 2
    text_pad_id: int = 3
    zero_token_id: int = -1
    ungenerated_token_id: int = -2

    max_syllables_per_line: int = 64
    num_section_types: int = 8
    max_repeat_groups: int = 16
    num_speaker_sim_bins: int = 5

    dim: int = 2048
    num_heads: int = 16
    num_layers: int = 16
    dim_feedforward: int = 5632
    context_frames: int = 1000
    rope_max_period: float = 10000.0
    norm_eps: float = 1e-5

    depformer_dim: int = 1024
    depformer_num_heads: int = 16
    depformer_num_layers: int = 4
    depformer_dim_feedforward: int = 2048
    depformer_low_rank_emb: Optional[int] = 128

    depformer_weights_schedule: list[int] = field(
        default_factory=lambda: list(range(8))
    )

    train_on_source_audio: bool = True

    def __post_init__(self) -> None:
        """지원 언어 쌍(한-영 또는 영-한) 및 음절·섹션 하이퍼파라미터 유효성을 검증합니다."""
        valid_pairs = {("ko", "en"), ("en", "ko"), ("auto", "auto")}
        if (self.source_lang, self.target_lang) not in valid_pairs:
            raise ValueError(
                f"지원하지 않는 번역 방향입니다: {self.source_lang} -> {self.target_lang} "
                "(지원: ko->en, en->ko)"
            )
        if self.max_syllables_per_line < 4:
            raise ValueError(
                f"max_syllables_per_line({self.max_syllables_per_line})은 4 이상이어야 합니다."
            )

    @property
    def n_q(self) -> int:
        """타깃 보컬과 소스 보컬 오디오 스트림을 합친 전체 오디오 코드북 개수를 반환합니다."""
        return 2 * self.num_codebooks_per_stream

    @property
    def dep_q(self) -> int:
        """추론 시 생성할 타깃 보컬 오디오 스트림의 코드북 개수를 반환합니다."""
        return self.num_codebooks_per_stream

    @property
    def total_streams(self) -> int:
        """가사 텍스트 스트림 1개와 전체 보컬 오디오 스트림을 합친 총 스트림 개수를 반환합니다."""
        return 1 + self.n_q

    @property
    def audio_bos_id(self) -> int:
        """보컬 오디오 스트림의 시작(BOS) 토큰 ID를 반환합니다."""
        return self.audio_card

    @property
    def text_bos_id(self) -> int:
        """가사 텍스트 스트림의 시작(BOS) 토큰 ID를 반환합니다."""
        return self.text_card

    @property
    def delays(self) -> list[int]:
        """가사 텍스트 및 타깃·소스 보컬 각 코드북에 적용할 프레임 지연(Delay) 리스트를 반환합니다."""
        q = self.num_codebooks_per_stream
        audio_stream_delay = [0] + [1] * (q - 1)
        return [0] + audio_stream_delay + audio_stream_delay

    @classmethod
    def preset_tiny(cls) -> "LyricTransConfig":
        """단위 테스트 및 CPU/경량 GPU 고속 검증을 위한 초경량 프리셋 설정을 생성합니다."""
        return cls(
            num_codebooks_per_stream=4,
            codec_dim=128,
            codec_vq_dim=64,
            dim=256,
            num_heads=8,
            num_layers=4,
            dim_feedforward=768,
            depformer_dim=128,
            depformer_num_heads=4,
            depformer_num_layers=2,
            depformer_dim_feedforward=256,
            depformer_low_rank_emb=32,
            depformer_weights_schedule=list(range(4)),
        )

    @classmethod
    def preset_1b(cls) -> "LyricTransConfig":
        """1B급 표준 노래 번역 프리셋 설정을 생성합니다."""
        return cls(
            num_codebooks_per_stream=8,
            dim=2048,
            num_heads=16,
            num_layers=16,
            dim_feedforward=5632,
            depformer_weights_schedule=list(range(8)),
        )

    @classmethod
    def preset_2b(cls) -> "LyricTransConfig":
        """풀 스케일 2B급 노래 번역 프리셋 설정을 생성합니다."""
        return cls(
            num_codebooks_per_stream=16,
            dim=2560,
            num_heads=20,
            num_layers=24,
            dim_feedforward=7040,
            depformer_weights_schedule=[0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8],
        )

    @classmethod
    def from_preset(
        cls,
        preset: str,
        source_lang: str = "ko",
        target_lang: str = "en",
    ) -> "LyricTransConfig":
        """프리셋 이름('tiny', '1b', '2b')과 언어 방향으로부터 LyricTransConfig 인스턴스를 생성합니다."""
        presets = {
            "tiny": cls.preset_tiny,
            "1b": cls.preset_1b,
            "2b": cls.preset_2b,
        }
        if preset not in presets:
            raise ValueError(
                f"알 수 없는 프리셋입니다: {preset} (지원: {tuple(presets.keys())})"
            )
        cfg = presets[preset]()
        cfg.source_lang = source_lang
        cfg.target_lang = target_lang
        cfg.__post_init__()
        return cfg

    @classmethod
    def from_dict(cls, raw_cfg: Mapping[str, Any]) -> "LyricTransConfig":
        """직렬화된 딕셔너리로부터 타입을 복원하여 LyricTransConfig 인스턴스를 생성합니다."""
        cfg_dict = dict(raw_cfg)
        if "codec_strides" in cfg_dict and isinstance(cfg_dict["codec_strides"], list):
            cfg_dict["codec_strides"] = tuple(cfg_dict["codec_strides"])
        if "depformer_weights_schedule" in cfg_dict and isinstance(
            cfg_dict["depformer_weights_schedule"], tuple
        ):
            cfg_dict["depformer_weights_schedule"] = list(
                cfg_dict["depformer_weights_schedule"]
            )
        return cls(**cfg_dict)
