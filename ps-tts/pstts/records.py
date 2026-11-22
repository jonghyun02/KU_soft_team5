"""모듈끼리 주고받는 자료형."""
from __future__ import annotations

from dataclasses import dataclass, field

# 말투 이름은 '더빙 실습 코드/prepare.py' 의 REGISTER 와 같다.
REGISTERS = ("미지정", "반말", "해요체", "하십시오체")


@dataclass
class LineItem:
    """입력 대사 한 줄."""
    id: str
    source_en: str
    prev_en: str = ""                 # 바로 앞 대사(문맥)
    relationship: str = ""            # 화자 관계·장면 설명(말투 판단용)
    register: str = "미지정"          # REGISTERS 중 하나
    target_syllables: int | None = None   # 사람이 정한 목표 음절 수(있으면 그대로 씀)
    seconds: float | None = None      # 대사 칸 길이(초, 쉼 제외). length_basis='seconds' 일 때 씀
    lip: bool = True                  # 입이 보이는 장면이면 True(PS-Comet), 아니면 COMET 만으로 고름
    draft_ko: str | None = None       # 초벌이 이미 있으면 모델 1 을 건너뜀
    meaning_notes: str = ""           # 꼭 살릴 뜻(사실·부정·숫자·강조)
    extra: dict = field(default_factory=dict)   # 입력 파일의 나머지 칸(출력에 그대로 붙임)


@dataclass
class Target:
    """길이 목표(한글 음절 수)."""
    n: int                    # 목표 음절 수
    tol: int                  # 허용 오차(±음절)
    en_syllables: int         # 영어 원문 음절 수(eSpeak)
    basis: str                # 'explicit' | 'relative' | 'seconds'
    seconds_est: float        # 목표 길이를 초로 환산한 값(참고용)

    @property
    def low(self) -> int:
        return max(1, self.n - self.tol)

    @property
    def high(self) -> int:
        return self.n + self.tol


@dataclass
class Candidate:
    """한국어 후보 대사 하나와 그 점수."""
    text: str
    origin: str               # 'model1' | 'llm' | 'file' | 'given'
    round: int = 0            # 0 = 초벌, 1부터 = 모델 2 생성 차수
    syllables: int | None = None
    labse: float | None = None
    register_found: list[str] | None = None   # 문장별 말투 판정
    ok_length: bool | None = None     # 길이(논문 식 6)
    ok_meaning: bool | None = None    # 뜻: LaBSE(논문 식 7)와 한국어 여부
    ok_register: bool | None = None   # 말투(논문에 없음)
    ok_numbers: bool | None = None    # 숫자(논문에 없음, 켰을 때만 True/False, 끄면 None)
    reasons: list[str] = field(default_factory=list)   # 탈락 까닭(한국어)
    dtw: float | None = None          # 모음 DTW 누적 비용(작을수록 입 모양이 비슷)
    dtw_inv: float | None = None      # 후보끼리 [0,1] 정규화 후 뒤집은 값(클수록 좋음)
    comet: float | None = None
    score: float | None = None        # 최종 선택 점수

    @property
    def passed(self) -> bool:
        return bool(self.ok_length and self.ok_meaning and self.ok_register) and self.ok_numbers is not False

    @property
    def iso_ok(self) -> bool:
        """논문 ISO 의 두 조건(식 6 길이, 식 7 LaBSE)과 한국어 여부만 채웠는지. 통과 후보가 없을 때 쓴다."""
        return bool(self.ok_length and self.ok_meaning)


@dataclass
class LineResult:
    item: LineItem
    target: Target
    draft_ko: str
    candidates: list[Candidate]
    selected: Candidate | None
    status: str               # 'ok' | 'fallback' | 'error'
    rounds_used: int = 0
    message: str = ""
