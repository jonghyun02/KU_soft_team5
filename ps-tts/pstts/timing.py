"""음절 열을 프레임 열로 늘린다(논문의 '길이 예측으로 음소를 반복해 늘린 열' S_b', T_b,i' 대신).

텍스트만 있어서 TTS 길이 예측기 대신 음절 속도 모델을 쓴다. 음절 하나 = 1/rate 초.
음절 안에서는 자음·글라이드·모음 가중치 비율로 시간을 나누고, 모음 구간에만 모음 기호를 붙인다.

논문과 다른 텍스트판의 선택: 논문의 S_b', T_b,i' 에는 빈 토큰 b 와 낱말 사이 공백(음소 하나)도 들어 있고
길이 예측기가 준 길이만큼 null 프레임이 된다. 여기서는 공백·문장부호·빈 토큰이 음절 열에 없어서 길이 0 이고,
그 시간은 음절 안 자음·글라이드 몫(null 프레임)에 녹아 있다. 음절 속도(en_rate, ko_rate)는 쉼을 뺀 낭독에서 잰 값이다.
"""
from __future__ import annotations

from typing import Sequence

from .config import Config
from .phonetics import Syl


def _segments(syl: Syl, cfg: Config) -> list[tuple[str | None, float]]:
    """음절 하나를 (라벨, 가중치) 조각으로 나눈다. 라벨은 모음 기호 또는 None."""
    if not syl.nucleus:
        return [(None, 1.0)]
    parts: list[tuple[str | None, float]] = [(None, cfg.cons_weight) for _ in syl.onset]
    if syl.glide:
        parts.append((None, cfg.glide_weight))
    share = cfg.vowel_weight / len(syl.nucleus)        # 이중모음은 모음 몫을 반씩
    parts.extend((v, share) for v in syl.nucleus)
    parts.extend((None, cfg.cons_weight) for _ in syl.coda)
    return parts


def frame_labels(syls: Sequence[Syl], rate_sps: float, cfg: Config) -> list[str | None]:
    """음절 열 -> 프레임마다 모음 기호 또는 None.

    연속 시간 경계를 누적한 뒤 반올림해 프레임(cfg.frame_sec) 경계로 바꾸므로 반올림 어긋남이 쌓이지 않는다.
    전체 프레임 수는 round(음절 수 / rate / frame_sec) 와 1 프레임 안에서 같다.
    """
    if rate_sps <= 0:
        raise ValueError(f"말 빠르기는 0보다 커야 합니다: {rate_sps}")
    if cfg.frame_sec <= 0:
        raise ValueError(f"프레임 길이는 0보다 커야 합니다: {cfg.frame_sec}")
    if cfg.vowel_weight <= 0 or cfg.cons_weight < 0 or cfg.glide_weight < 0:
        raise ValueError("모음 가중치는 0보다 크고, 자음·글라이드 가중치는 0 이상이어야 합니다.")
    syl_sec = 1.0 / rate_sps
    labels: list[str | None] = []
    t = 0.0
    edge = 0
    for syl in syls:
        parts = _segments(syl, cfg)
        total = sum(w for _, w in parts)
        for label, w in parts:
            t += syl_sec * w / total
            nxt = round(t / cfg.frame_sec)
            if nxt > edge:
                labels.extend([label] * (nxt - edge))
                edge = nxt
    return labels
