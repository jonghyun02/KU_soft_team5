"""EDarch 동시통역 모델 및 오디오 코덱의 하이퍼파라미터 설정 데이터클래스 모듈."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass
class EDarchConfig:
    """EDarch 멀티스트림 트랜스포머, Depformer, 오디오 코덱 및 어휘 규격을 정의하는 설정 클래스입니다."""

    source_lang: str = "en"
    target_lang: str = "ko"

    sample_rate: int = 24000
    frame_rate: float = 12.5
    audio_card: int = 2048
    num_codebooks_per_stream: int = 16
    codec_dim: int = 512
    codec_vq_dim: int = 256
    codec_strides: tuple[int, ...] = (8, 6, 5, 4, 2)

    text_card: int = 32000
    text_eos_id: int = 2
    text_pad_id: int = 3
    zero_token_id: int = -1
    ungenerated_token_id: int = -2

    dim: int = 2560
    num_heads: int = 20
    num_layers: int = 24
    dim_feedforward: int = 7040
    context_frames: int = 1000
    rope_max_period: float = 10000.0
    norm_eps: float = 1e-5

    depformer_dim: int = 1024
    depformer_num_heads: int = 16
    depformer_num_layers: int = 4
    depformer_dim_feedforward: int = 2048
    depformer_low_rank_emb: Optional[int] = 128

    depformer_weights_schedule: list[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8]
    )

    num_speaker_sim_bins: int = 5
    train_on_source_audio: bool = True

    def __post_init__(self) -> None:
        """지원 언어 방향(영어 -> 한국어) 설정의 유효성을 검증합니다."""
        if self.source_lang != "en" or self.target_lang != "ko":
            raise ValueError(
                f"EDarch는 영어(en) -> 한국어(ko) 번역만 지원합니다: "
                f"{self.source_lang} -> {self.target_lang}"
            )

    @property
    def n_q(self) -> int:
        """타깃과 소스 오디오 스트림을 합친 전체 오디오 코드북 개수를 반환합니다."""
        return 2 * self.num_codebooks_per_stream

    @property
    def dep_q(self) -> int:
        """추론 시 생성할 타깃 오디오 스트림의 코드북 개수를 반환합니다."""
        return self.num_codebooks_per_stream

    @property
    def total_streams(self) -> int:
        """텍스트 스트림 1개와 전체 오디오 스트림을 합친 총 스트림 개수를 반환합니다."""
        return 1 + self.n_q

    @property
    def audio_bos_id(self) -> int:
        """오디오 스트림의 시작(BOS) 토큰 ID를 반환합니다."""
        return self.audio_card

    @property
    def text_bos_id(self) -> int:
        """텍스트 스트림의 시작(BOS) 토큰 ID를 반환합니다."""
        return self.text_card

    @property
    def delays(self) -> list[int]:
        """텍스트 및 타깃·소스 오디오 각 코드북에 적용할 프레임 지연(Delay) 리스트를 반환합니다."""
        q = self.num_codebooks_per_stream
        audio_stream_delay = [0] + [1] * (q - 1)
        return [0] + audio_stream_delay + audio_stream_delay

    @classmethod
    def EDarch_2b(cls) -> "EDarchConfig":
        """풀 스케일 2B급 EDarch 기본 프리셋 설정을 생성합니다."""
        return cls()

    @classmethod
    def EDarch_m_1b(cls) -> "EDarchConfig":
        """1B급 표준 EDarch 프리셋 설정을 생성합니다."""
        return cls(
            num_codebooks_per_stream=8,
            dim=2048,
            num_heads=16,
            num_layers=16,
            dim_feedforward=5632,
            depformer_weights_schedule=list(range(8)),
        )

    @classmethod
    def from_preset(cls, preset: str) -> "EDarchConfig":
        """프리셋 이름('1b' 또는 '2b')으로부터 EDarchConfig 인스턴스를 생성합니다."""
        presets = {
            "1b": cls.EDarch_m_1b,
            "2b": cls.EDarch_2b,
        }
        if preset not in presets:
            raise ValueError(f"알 수 없는 프리셋입니다: {preset} (지원: {tuple(presets.keys())})")
        return presets[preset]()

    @classmethod
    def from_dict(cls, raw_cfg: Mapping[str, Any]) -> "EDarchConfig":
        """직렬화된 딕셔너리로부터 타입을 복원하여 EDarchConfig 인스턴스를 생성합니다."""
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
