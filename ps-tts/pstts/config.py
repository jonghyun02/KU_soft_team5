"""기본 설정값과 그 근거.

논문 값은 PS-TTS(arXiv 2604.09111) 식 번호를 적었다. '여기서 정함' 은 논문에 없거나 텍스트 입력이라 바꾼 값이다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # D:\PS-TTS
DATA_DIR = Path(__file__).resolve().parent / "data"  # 모음표 캐시
CACHE_DIR = ROOT / "cache"                            # LLM 응답 캐시, LLM 작업 폴더


@dataclass
class Config:
    # ---------------------------------------------------------------- 길이(ISO)
    # 논문 프레임: 22.05 kHz, hop 256 (3.1절 쉼 검출과 같은 단위)
    frame_sec: float = 256 / 22050
    # 논문 식 (6): K' - 26 <= K(T) <= K' + 26 프레임(약 0.30초). 논문 본문은 같은 조건을 'within 30 frames (~350ms)' 로도
    # 적어서 두 값이 어긋난다. 여기서는 식 (6) 의 26 을 쓴다(30 이면 허용 오차가 ±2 가 아니라 ±3음절이 된다).
    iso_tol_frames: int = 26
    # 영어 낭독 음절 속도(음절/초). Pellegrino, Coupe, Marsico (2011) Language 87(3) 의 영어 값.
    en_rate: float = 6.19
    # 영어 1음절당 한국어 음절 수(= 한국어 음절 속도 / 영어 음절 속도). 같은 길이 칸에 들어가는 한국어 음절 수
    # = 영어 음절 수 x coef. 임시값 1.28 은 2026-11-21 에 잰 값이다:
    #   .venv\Scripts\python.exe scripts\measure_rate.py --from-dub-rhythm C:\projects\dub-rhythm-demo\work\lines.json
    #   usable 112줄, 한국어 목표 음절 합 1539 / 영어 음절 합(eSpeak) 1202 = 1.280(줄별 중앙값 1.273).
    # 한국어 쪽은 OmniVoice 글자 가중치로 정한 n_target 이라 실제 낭독 속도가 아니다. AI Hub 71524 로 다시 잰다.
    coef: float = 1.28
    # 허용 오차(음절). None 이면 iso_tol_frames x frame_sec x 한국어 음절 속도를 반올림(약 ±2)
    tol_syllables: int | None = None
    # 'relative' = 영어 음절 수 x coef. 논문 식 (3), (4), (6) 을 풀면 K' 가 지워져 '후보의 TTS 길이 = 원문의 TTS 길이' 가
    #              남는다. 그 TTS 길이 맞추기를 음절 속도 모델로 옮긴 것이다(논문과 같은 목표).
    # 'seconds'  = 칸 길이(초) x 한국어 평균 음절 속도. 대사 칸을 보통 빠르기의 한국어로 채우는 이 구현의 선택이다.
    #              논문에서는 칸 길이가 말 빠르기 R 과 허용 오차(26/R 프레임)만 바꾸고 목표는 바꾸지 않는다.
    length_basis: str = "relative"

    # ---------------------------------------------------------------- 뜻(ISO)
    labse_model: str = "sentence-transformers/LaBSE"
    labse_min: float = 0.75                 # 논문 식 (7)

    # ---------------------------------------------------------------- 후보 생성(모델 2)
    # 논문: ISO 는 GPT-4o 로 최대 60번 바꿔 쓰기, PS 는 후보 60개. 여기서는 한 차수에 n_per_round 개씩,
    # 최대 max_rounds 차수(합계 최대 max_candidates 개)까지 만들고, 통과 후보가 min_pass 개 모이면 멈춘다.
    n_per_round: int = 20
    max_rounds: int = 3
    max_candidates: int = 60
    min_pass: int = 10
    llm: str = "claude"                     # 'claude' | 'openai' | 'file'
    llm_model: str = "claude-sonnet-5-5"    # 별칭 대신 정확한 모델 ID
    llm_effort: str = "low"
    llm_base_url: str = "http://127.0.0.1:8080"   # llm='openai' 일 때(llama-server·vLLM 등)
    llm_api_key_env: str | None = None      # llm='openai' 에서 키가 필요하면 환경 변수 이름
    llm_file: str | None = None             # llm='file' 일 때 후보 JSONL
    llm_chunk: int = 6                      # 한 번 부를 때 넣는 대사 수
    llm_workers: int = 2                    # 동시에 띄우는 claude 프로세스 수
    llm_timeout: int = 900                  # 호출 하나 최대 초
    llm_cache: bool = True                  # 같은 프롬프트면 저장된 응답을 다시 씀

    # ---------------------------------------------------------------- 말투(논문에 없음, 여기서 추가)
    check_register: bool = True
    # 숫자 조건: 영어 대사의 수(3, ten …)가 후보에 없으면 떨어뜨림(select.missing_numbers). 논문에도, 합의한 거르기
    # (길이 + 뜻 + 말투)에도 없어서 기본은 끈다(--number-check 로 켬). LaBSE 0.75 가 '3번 승강장' 이 빠진 후보를
    # 통과시킨 일이 있어 만들었지만, 반(7:30), 과장(a thousand times), 굳은 말(24/7), 현지화(ten bucks -> 만 원)처럼
    # 맞는 번역도 떨어뜨릴 수 있다(README 한계).
    check_numbers: bool = False

    # ---------------------------------------------------------------- PS(모음 DTW)
    # 모음 거리: 논문은 Your-TTS 텍스트 인코더 모음 임베딩(K-means 5개 중심 평균)의 유클리드 거리(식 8, 9).
    # 여기서는 eSpeak IPA -> PanPhon 조음 자질 벡터의 유클리드 거리를 [0,1] 로 정규화해 쓴다.
    null_cost: float = 1.0          # 모음이 아닌 칸(자음·글라이드)과 모음 사이 비용. 논문에 값이 없어 여기서 정함
    # Sakoe-Chiba 반경(프레임). 논문은 tslearn [44] 과 Sakoe-Chiba 제약을 썼다고만 적고 반경은 밝히지 않았다.
    # tslearn 은 두 열의 길이가 다르면 띠 너비를 |n - m| + 반경으로 넓힌다. 반경이 작으면 원문과 프레임 수가 같은
    # 후보만 대각선 근처에 묶이고, 길이가 어긋난 후보는 어긋난 만큼 더 휘어 DTW 가 작아진다. 그래서 반경 1(tslearn
    # 기본값)은 길이가 맞는 후보를 벌주었다. 여기서는 식 (6) 의 길이 허용과 같은 26프레임(약 0.30초)을 쓴다:
    # 줄 전체에 허용한 시간 어긋남만큼 모음 하나하나에도 어긋남을 허용한다.
    # 잰 값(scripts\measure_dtw_band.py out\demo\results.jsonl, 2026-11-22 데모의 입 모양 줄 통과 후보 120개):
    # 같은 글을 원문 프레임 수 -26~+26 으로 늘이고 줄였을 때, 원문과 길이가 같을 때의 DTW 가 10프레임 이상 어긋난 쪽
    # 평균보다 큰 글이 반경 1 에서 107개, 8 에서 86개, 16 에서 57개, 20 에서 39개, 26 에서 36개, 40 에서 34개다
    # (20 부터 거의 줄지 않음. 남는 몫은 식 10 이 비용의 합이라 프레임이 적은 쪽이 조금 유리한 데서 온다).
    sakoe_chiba_radius: int = 26
    # 음절 안 시간 나누기 가중치(텍스트만 있어서 TTS 길이 예측기 대신 씀). 여기서 정함
    vowel_weight: float = 2.0
    cons_weight: float = 1.0
    glide_weight: float = 0.5

    # ---------------------------------------------------------------- PS-Comet
    alpha: float = 1.6              # 논문 식 (20)
    beta: float = 0.4               # 논문 식 (20)
    # 참조 없는 COMET(원문, 후보). 논문 식 (20) 의 Comet(S, T) 꼴. 공개(게이트 없음)·Apache-2.0
    comet_model: str = "Unbabel/wmt20-comet-qe-da"
    # 참조형 모델을 쓸 때 'draft' = 모델 1 초벌을 참조로. 이때 초벌 자신은 참조와 같아 점수가 늘 높으므로
    # COMET 을 재지 않고, 다른 후보가 있으면 고르기에서 뺀다(select.selection_pool).
    comet_ref: str = "none"
    comet_batch: int = 16

    # ---------------------------------------------------------------- 모델 1(초벌 번역)
    nmt_dir: str = r"D:\transformer-base\enko-nmt"
    nmt_ckpt: str = r"runs\base_v2\final\avg_last5.pt"   # nmt_dir 기준 상대 경로도 됨
    nmt_beam: int = 4
    use_draft: bool = True

    # ---------------------------------------------------------------- 실행
    device: str = "cuda"            # 'cuda' | 'cpu'. cuda 가 안 되면 조용히 cpu 로 바꾸지 않고 멈춘다
    cpu_threads: int = 6
    out_dir: str = str(ROOT / "out")
    extra: dict = field(default_factory=dict)

    @property
    def ko_rate(self) -> float:
        """한국어 음절 속도(음절/초) = 영어 음절 속도 x coef."""
        return self.en_rate * self.coef

    def tolerance(self) -> int:
        if self.tol_syllables is not None:
            return int(self.tol_syllables)
        return max(1, round(self.iso_tol_frames * self.frame_sec * self.ko_rate))
