"""논문 그대로 판 설정.

값마다 논문(arXiv 2604.09111) 식 번호나 절을 적었다. '해석' 은 논문에 값이나 방법이 없어 여기서 정한 것이다.
TTS 근사 값과 모델 2, 모델 1 값은 기존 판 Config 기본값을 그대로 가져와 두 판이 같은 숫자를 쓰게 한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields

from ..config import ROOT, Config
from .g2p import EN_VOICE, EN_WITH_STRESS

_BASE = Config()


@dataclass
class PaperConfig:
    # ---------------------------------------------------------------- ISO (3.1절)
    # 원문 음성(SA) 프레임: 22.05 kHz, 창 1024, hop 256 (3.1절 쉼 검출). 식 (2), (6) 의 단위
    frame_sec_sa: float = 256 / 22050
    # 식 (6): K' - 26 <= K(T_b) <= K' + 26. 본문은 'within 30 frames (~350ms)' 라고 적어 식과 어긋나서 식을 따른다
    iso_tol_frames: int = 26
    labse_model: str = _BASE.labse_model
    labse_min: float = 0.75                 # 식 (7)
    iso_batch: int = 10                     # 해석: 한 번 부를 때 줄마다 받는 바꿔 쓰기 후보 수(호출 수 줄이기)
    iso_max: int = 60                       # 3.1절 'the selection process repeats up to 60 iterations'

    # ---------------------------------------------------------------- PS (3.2절), PS-Comet (3.3절)
    ps_n: int = 60                          # 3.2절 '60 candidate sentences'
    ps_chunk: int = 2                       # 해석: PS 생성 때 한 번에 넣는 줄 수(줄마다 60개라 답이 김)
    ps_extra_rounds: int = 2                # 해석: 정리 뒤 60개가 안 되면 모자란 만큼 더 묻는 횟수
    alpha: float = 1.6                      # 식 (20)
    beta: float = 0.4                       # 식 (20)
    # 반경은 논문에 없다. tslearn [44] 기본값(None = 반경 1)을 그대로 쓴다(해석)
    sakoe_chiba_radius: int | None = None
    comet_model: str = _BASE.comet_model    # 해석: 식 (20) 이 Comet(S, T) 두 인자 꼴이라 참조 없는 모델
    comet_batch: int = _BASE.comet_batch

    # ---------------------------------------------------------------- G2P (3.1절, 영어 phonemizer 설정)
    # 해석: 논문에 설정이 없다. Fig. 5(c) 의 영어 모음 14개 = espeak en-gb 모음, Fig. 4 에 강세표 없음(g2p.py 머리말)
    g2p_en_voice: str = EN_VOICE
    g2p_en_stress: bool = EN_WITH_STRESS

    # ---------------------------------------------------------------- TTS 프런트엔드
    tts: str = "approx"                     # 'approx'(TTS 근사, 지금 기본) | 'yourtts'(학습한 체크포인트)
    # approx: 기존 판과 같은 음절 속도 근사
    en_rate: float = _BASE.en_rate
    coef: float = _BASE.coef
    vowel_weight: float = _BASE.vowel_weight
    cons_weight: float = _BASE.cons_weight
    glide_weight: float = _BASE.glide_weight
    approx_null_cost: float = 1.0
    # yourtts: 학습한 Coqui VITS/YourTTS 체크포인트
    tts_dir: str | None = None
    tts_checkpoint: str | None = None       # 없으면 tts_dir 에서 best_model*.pth 또는 model_file.pth 를 찾음
    tts_speaker: str | None = None          # d-vector 키나 화자 이름. 없으면 첫 화자
    tts_lang_en: str = "en"
    tts_lang_ko: str = "ko"
    vowel_table: str | None = None          # 식 (8), (9) 모음표 JSON. 없으면 tts_dir/vowel_table.json
    noise_scale_dp: float = 0.0             # 해석: 길이를 결정적으로(확률 길이 예측기의 잡음 0)
    length_scale: float = 1.0

    # ---------------------------------------------------------------- 모델 2 (논문 GPT-4o 자리)
    llm: str = _BASE.llm
    llm_model: str = _BASE.llm_model
    llm_effort: str = _BASE.llm_effort
    llm_base_url: str = _BASE.llm_base_url
    llm_api_key_env: str | None = _BASE.llm_api_key_env
    llm_file: str | None = _BASE.llm_file
    llm_chunk: int = _BASE.llm_chunk
    llm_workers: int = _BASE.llm_workers
    llm_timeout: int = _BASE.llm_timeout
    llm_cache: bool = _BASE.llm_cache

    # ---------------------------------------------------------------- 모델 1 (논문의 NMT 번역문)
    nmt_dir: str = _BASE.nmt_dir
    nmt_ckpt: str = _BASE.nmt_ckpt
    nmt_beam: int = _BASE.nmt_beam

    # ---------------------------------------------------------------- 실행
    device: str = "cuda"                    # cuda 를 못 쓰면 멈춘다. 조용히 cpu 로 바꾸지 않는다
    cpu_threads: int = _BASE.cpu_threads
    out_dir: str = str(ROOT / "out" / "paper")
    extra: dict = field(default_factory=dict)

    @property
    def ko_rate(self) -> float:
        """approx 의 한국어 음절 속도(음절/초) = 영어 음절 속도 x coef. 기존 판과 같다."""
        return self.en_rate * self.coef

    # base_config 가 옮기는 칸. 모델 2, 모델 1, 점수기, 장치, approx 속도만 옮긴다.
    # DTW 반경처럼 두 판의 뜻이 다른 칸은 옮기지 않는다.
    _SHARED = ("llm", "llm_model", "llm_effort", "llm_base_url", "llm_api_key_env", "llm_file", "llm_chunk",
               "llm_workers", "llm_timeout", "llm_cache", "nmt_dir", "nmt_ckpt", "nmt_beam", "device",
               "cpu_threads", "comet_model", "comet_batch", "labse_model", "en_rate", "coef")

    def base_config(self) -> Config:
        """기존 판 함수(make_backend, LabseScorer, CometScorer, Model1Drafter)에 넘길 Config."""
        names = {f.name for f in fields(Config)}
        base = Config()
        for name in self._SHARED:
            if name in names:
                setattr(base, name, getattr(self, name))
        return base
