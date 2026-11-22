"""음소 distinct-2 (식 3), Phodeg, Phovar (3.2절).

음소로 나누는 방법은 논문이 Kim et al. (2023, ISMIR, arXiv 2308.13715) 을 따른다고만 했다.
그 논문 4.2절을 따라 아래처럼 했다.
- 영어: ARPAbet, 강세 숫자 없음 (3.2절 예시 "AA", "N", ..., "CH", "EY", "N", "JH").
  [해석] 발음 사전은 g2p_en (CMUdict, 사전에 없는 단어는 g2p_en 신경망).
  비슷한 모음은 같은 모음: IY→IH, UW→UH, AE→EH (Kim 2023 에 적힌 세 쌍만).
- 한국어: 자모 (Kim 2023 의 "같이" 표기가 같이, 즉 소리 나는 대로가 아니라 글자 그대로의 자모).
  비슷한 모음은 같은 모음: ㅐ=ㅔ, ㅚ=ㅙ=ㅞ (Kim 2023 "AE-E", "OE-OI-OAE").
  [해석] 겹받침은 두 자음으로 나눈다 (Kim 2023 의 자음 19개 체계). 초성 ㅇ 도 넣는다.
  [해석] 한국어 가사 속 영어 단어는 영어 방식으로 나눈다.
- 세기 전에 text.normalize_text (둥근 따옴표, 키릴 글자) 를 거친다.
- 줄마다 끝에 <eos> 를 붙이고, 마디 안 모든 음소를 이어 bigram 을 만든다 (3.2절 예시).
"""
from __future__ import annotations

import re
from functools import lru_cache
from statistics import mean, pstdev

from .text import normalize_text

EN_MERGE = {"IY": "IH", "UW": "UH", "AE": "EH"}  # 3.2절 예시가 IH 로 적혀 있어 IH 쪽으로 모음
KO_VOWEL_MERGE = {"ㅔ": "ㅐ", "ㅙ": "ㅚ", "ㅞ": "ㅚ"}

_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JONG = ["", "ㄱ", "ㄲ", "ㄱㅅ", "ㄴ", "ㄴㅈ", "ㄴㅎ", "ㄷ", "ㄹ", "ㄹㄱ", "ㄹㅁ", "ㄹㅂ", "ㄹㅅ", "ㄹㅌ",
         "ㄹㅍ", "ㄹㅎ", "ㅁ", "ㅂ", "ㅂㅅ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]

_TOKEN = re.compile(r"[가-힣]+|[A-Za-z]+(?:'[A-Za-z]+)*")
_g2p = None


def _get_g2p():
    global _g2p
    if _g2p is None:
        from g2p_en import G2p
        _g2p = G2p()
    return _g2p


@lru_cache(maxsize=200_000)
def en_word_phonemes(word: str) -> tuple[str, ...]:
    out = []
    for p in _get_g2p()(word):
        p = re.sub(r"\d", "", p.strip())
        if not p or not p.isalpha() or not p.isupper():
            continue
        out.append(EN_MERGE.get(p, p))
    return tuple(out)


def ko_syllable_jamo(ch: str) -> list[str]:
    code = ord(ch) - 0xAC00
    cho, rest = divmod(code, 588)
    jung, jong = divmod(rest, 28)
    v = _JUNG[jung]
    out = [_CHO[cho], KO_VOWEL_MERGE.get(v, v)]
    out.extend(_JONG[jong])
    return out


def line_phonemes(line: str) -> list[str]:
    out: list[str] = []
    for tok in _TOKEN.findall(normalize_text(line)):
        if "가" <= tok[0] <= "힣":
            for ch in tok:
                out.extend(ko_syllable_jamo(ch))
        else:
            out.extend(en_word_phonemes(tok.lower()))
    return out


def section_phonemes(lines: list[str]) -> list[str]:
    seq: list[str] = []
    for line in lines:
        seq.extend(line_phonemes(line))
        seq.append("<eos>")
    return seq


def pho(lines: list[str]) -> float | None:
    """식 3: 서로 다른 bigram 수 / 전체 bigram 수. bigram 이 없으면 None."""
    seq = section_phonemes(lines)
    bigrams = list(zip(seq, seq[1:]))
    if not bigrams:
        return None
    return len(set(bigrams)) / len(bigrams)


def song_pho(sections: list[list[str]]) -> tuple[float, float] | None:
    """한 곡의 (Phodeg, Phovar): 마디별 pho 의 평균과 표준편차 (3.2절).

    [해석] 표준편차는 모표준편차(pstdev). 마디가 하나면 0.
    """
    vals = [v for v in (pho(s) for s in sections) if v is not None]
    if not vals:
        return None
    return mean(vals), pstdev(vals)
