"""길이 목표 정하기(한글 음절 수).

논문 ISO 는 R = K' / dur_TTS(S_b)(식 3), K(T_b) = R x dur_TTS(T_b)(식 4), |K(T_b) - K'| <= 26(식 6) 이다.
식을 풀면 K' 가 지워져 |dur_TTS(T_b) - dur_TTS(S_b)| <= 26 / R 이 남는다. 곧 목표는 '후보의 TTS 길이 = 원문의
TTS 길이' 이고, 원문 음성 길이 K' 는 R 을 거쳐 허용 오차만 바꾼다. 텍스트판은 TTS 길이를 음절 속도 모델로 바꾼다.
- relative: 영어 음절 수(eSpeak) x coef. 위 TTS 길이 맞추기를 옮긴 것이다(R = 1 로 본 허용 오차).
- explicit: 사람이 정한 목표 음절 수를 그대로 쓴다.
- seconds : 칸 길이(초) x 한국어 평균 음절 속도. 대사 칸을 보통 빠르기의 한국어로 채우는 이 구현의 선택이다.
            논문이라면 같은 칸 길이로 R 만 재고 목표는 relative 와 같게 둔다(README 논문과 다른 점).
허용 오차는 Config.tolerance()(식 6 의 26프레임을 한국어 음절 속도로 환산, 약 ±2)다.
"""
from __future__ import annotations

import math

from .config import Config
from .phonetics import english_syllable_count
from .records import LineItem, Target

LENGTH_BASES = ("relative", "seconds")


def round_half_up(x: float) -> int:
    """반올림(0.5 는 올림). 파이썬 round 는 0.5 를 짝수 쪽으로 보내서 쓰지 않는다."""
    return int(math.floor(x + 0.5))


def target_for(item: LineItem, cfg: Config) -> Target:
    """대사 한 줄의 길이 목표."""
    if cfg.length_basis not in LENGTH_BASES:
        raise ValueError(f"length_basis 는 {', '.join(LENGTH_BASES)} 중 하나여야 합니다: {cfg.length_basis}")
    ko_rate = cfg.ko_rate
    if ko_rate <= 0:
        raise ValueError(f"한국어 음절 속도가 0 이하입니다(en_rate {cfg.en_rate}, coef {cfg.coef}).")
    tol = cfg.tolerance()
    if tol < 0:
        raise ValueError(f"허용 오차는 0 이상이어야 합니다: {tol}")
    en_syllables = english_syllable_count(item.source_en)
    if item.target_syllables is not None:
        n = int(item.target_syllables)
        if n < 1:
            raise ValueError(f"{item.id}: 목표 음절 수는 1 이상이어야 합니다: {item.target_syllables}")
        basis = "explicit"
    elif item.seconds is not None and cfg.length_basis == "seconds":
        if item.seconds <= 0:
            raise ValueError(f"{item.id}: 대사 칸 길이(초)는 0 보다 커야 합니다: {item.seconds}")
        n = max(1, round_half_up(item.seconds * ko_rate))
        basis = "seconds"
    else:
        n = max(1, round_half_up(en_syllables * cfg.coef))
        basis = "relative"
    return Target(n=n, tol=tol, en_syllables=en_syllables, basis=basis, seconds_est=n / ko_rate)
