"""arXiv:2309.11093 기반 한-영 음절·음소 분석, 각운(Rhyme), 비어휘 보컬 감지 및 자기유사도 행렬(SSM) 모듈."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import math
import re
from statistics import mean, pstdev
from typing import Dict, List, Optional, Sequence, Set, Tuple
import unicodedata

import torch

_HANGUL_RE = re.compile(r"[가-힣]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")
_DIGIT_RE = re.compile(r"[0-9]")
_TOKEN_RE = re.compile(r"[가-힣]+|[A-Za-z]+(?:'[A-Za-z]+)*|[0-9]+")
_SECTION_HEADER_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$")

_APOS_MAP = {"’": "'", "‘": "'", "ʼ": "'", "`": "'", "´": "'"}
_HOMOGLYPH_MAP = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ѕ": "s", "ј": "j", "ԁ": "d", "һ": "h", "ӏ": "l", "ԛ": "q",
    "ԝ": "w", "ɡ": "g", "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M",
    "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X", "У": "Y",
    "І": "I", "Ј": "J", "Ѕ": "S", "ο": "o", "α": "a", "ε": "e", "ι": "i",
    "ν": "v", "Ο": "O", "Α": "A", "Ε": "E", "Ι": "I", "Τ": "T",
}
_NORM_TRANS = str.maketrans({**_APOS_MAP, **_HOMOGLYPH_MAP})

_KO_CHO = (
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ",
    "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
)
_KO_JUNG = (
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ",
    "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
)
_KO_JONG: Tuple[Tuple[str, ...], ...] = (
    (), ("ㄱ",), ("ㄲ",), ("ㄱ", "ㅅ"), ("ㄴ",), ("ㄴ", "ㅈ"), ("ㄴ", "ㅎ"),
    ("ㄷ",), ("ㄹ",), ("ㄹ", "ㄱ"), ("ㄹ", "ㅁ"), ("ㄹ", "ㅂ"), ("ㄹ", "ㅅ"),
    ("ㄹ", "ㅌ"), ("ㄹ", "ㅍ"), ("ㄹ", "ㅎ"), ("ㅁ",), ("ㅂ",), ("ㅂ", "ㅅ"),
    ("ㅅ",), ("ㅆ",), ("ㅇ",), ("ㅈ",), ("ㅊ",), ("ㅋ",), ("ㅌ",), ("ㅍ",), ("ㅎ",),
)
_KO_VOWEL_MERGE = {"ㅔ": "ㅐ", "ㅙ": "ㅚ", "ㅞ": "ㅚ", "ㅖ": "ㅒ"}

_KO_LIAISON_SPLIT: Dict[int, Tuple[Tuple[str, ...], str]] = {
    1: ((), "ㄱ"), 2: ((), "ㄲ"), 3: (("ㄱ",), "ㅆ"), 4: ((), "ㄴ"), 5: (("ㄴ",), "ㅈ"),
    6: (("ㄴ",), "ㅎ"), 7: ((), "ㄷ"), 8: ((), "ㄹ"), 9: (("ㄹ",), "ㄱ"), 10: (("ㄹ",), "ㅁ"),
    11: (("ㄹ",), "ㅂ"), 12: (("ㄹ",), "ㅆ"), 13: (("ㄹ",), "ㅌ"), 14: (("ㄹ",), "ㅍ"), 15: (("ㄹ",), "ㅎ"),
    16: ((), "ㅁ"), 17: ((), "ㅂ"), 18: (("ㅂ",), "ㅆ"), 19: ((), "ㅅ"), 20: ((), "ㅆ"),
    22: ((), "ㅈ"), 23: ((), "ㅊ"), 24: ((), "ㅋ"), 25: ((), "ㅌ"), 26: ((), "ㅍ"),
}

_EN_VOWEL_MERGE = {"IY": "IH", "UW": "UH", "AE": "EH"}
_EN_ARPABET_VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW",
}

_BUILTIN_EN_LEXICON: Dict[str, Tuple[int, Tuple[str, ...]]] = {
    "a": (1, ("AH",)), "again": (2, ("AH", "G", "EH", "N")), "all": (1, ("AO", "L")),
    "alone": (2, ("AH", "L", "OW", "N")), "always": (2, ("AO", "L", "W", "EY", "Z")),
    "am": (1, ("EH", "M")), "and": (1, ("EH", "N", "D")), "are": (1, ("AA", "R")),
    "around": (2, ("AH", "R", "AW", "N", "D")), "away": (2, ("AH", "W", "EY")),
    "babe": (1, ("B", "EY", "B")), "baby": (2, ("B", "EY", "B", "IH")), "back": (1, ("B", "EH", "K")),
    "be": (1, ("B", "IH")), "beautiful": (3, ("B", "Y", "UH", "T", "IH", "F", "AH", "L")),
    "because": (2, ("B", "IH", "K", "AO", "Z")), "before": (2, ("B", "IH", "F", "AO", "R")),
    "blue": (1, ("B", "L", "UH")), "boom": (1, ("B", "UH", "M")), "boy": (1, ("B", "OY")),
    "break": (1, ("B", "R", "EY", "K")), "bright": (1, ("B", "R", "AY", "T")), "burn": (1, ("B", "ER", "N")),
    "burning": (2, ("B", "ER", "N", "IH", "NG")), "can": (1, ("K", "EH", "N")), "can't": (1, ("K", "EH", "N", "T")),
    "chance": (1, ("CH", "EH", "N", "S")), "change": (1, ("CH", "EY", "N", "JH")), "close": (1, ("K", "L", "OW", "S")),
    "come": (1, ("K", "AH", "M")), "crazy": (2, ("K", "R", "EY", "Z", "IH")), "cry": (1, ("K", "R", "AY")),
    "dance": (1, ("D", "EH", "N", "S")), "dancing": (2, ("D", "EH", "N", "S", "IH", "NG")),
    "dark": (1, ("D", "AA", "R", "K")), "day": (1, ("D", "EY")), "days": (1, ("D", "EY", "Z")),
    "deep": (1, ("D", "IH", "P")), "do": (1, ("D", "UH")), "don't": (1, ("D", "OW", "N", "T")),
    "down": (1, ("D", "AW", "N")), "dream": (1, ("D", "R", "IH", "M")), "dreams": (1, ("D", "R", "IH", "M", "Z")),
    "every": (3, ("EH", "V", "ER", "IH")), "everyday": (4, ("EH", "V", "ER", "IH", "D", "EY")),
    "everything": (3, ("EH", "V", "R", "IH", "TH", "IH", "NG")), "eyes": (1, ("AY", "Z")),
    "fall": (1, ("F", "AO", "L")), "falling": (2, ("F", "AO", "L", "IH", "NG")), "far": (1, ("F", "AA", "R")),
    "feel": (1, ("F", "IH", "L")), "feeling": (2, ("F", "IH", "L", "IH", "NG")), "fire": (2, ("F", "AY", "ER")),
    "fly": (1, ("F", "L", "AY")), "for": (1, ("F", "AO", "R")), "forever": (3, ("F", "ER", "EH", "V", "ER")),
    "free": (1, ("F", "R", "IH")), "from": (1, ("F", "R", "AH", "M")), "get": (1, ("G", "EH", "T")),
    "girl": (1, ("G", "ER", "L")), "give": (1, ("G", "IH", "V")), "glow": (1, ("G", "L", "OW")),
    "go": (1, ("G", "OW")), "gone": (1, ("G", "AO", "N")), "good": (1, ("G", "UH", "D")),
    "goodbye": (2, ("G", "UH", "D", "B", "AY")), "hand": (1, ("HH", "EH", "N", "D")),
    "hands": (1, ("HH", "EH", "N", "D", "Z")), "heart": (1, ("HH", "AA", "R", "T")),
    "heaven": (2, ("HH", "EH", "V", "AH", "N")), "here": (1, ("HH", "IH", "R")), "high": (1, ("HH", "AY")),
    "higher": (2, ("HH", "AY", "ER")), "hold": (1, ("HH", "OW", "L", "D")), "home": (1, ("HH", "OW", "M")),
    "hope": (1, ("HH", "OW", "P")), "i": (1, ("AY",)), "i'll": (1, ("AY", "L")), "i'm": (1, ("AY", "M")),
    "in": (1, ("IH", "N")), "inside": (2, ("IH", "N", "S", "AY", "D")), "into": (2, ("IH", "N", "T", "UH")),
    "is": (1, ("IH", "Z")), "it": (1, ("IH", "T")), "it's": (1, ("IH", "T", "S")),
    "just": (1, ("JH", "AH", "S", "T")), "keep": (1, ("K", "IH", "P")), "kiss": (1, ("K", "IH", "S")),
    "know": (1, ("N", "OW")), "la": (1, ("L", "AA")), "let": (1, ("L", "EH", "T")),
    "life": (1, ("L", "AY", "F")), "light": (1, ("L", "AY", "T")), "lights": (1, ("L", "AY", "T", "S")),
    "like": (1, ("L", "AY", "K")), "little": (2, ("L", "IH", "T", "AH", "L")), "live": (1, ("L", "IH", "V")),
    "lonely": (2, ("L", "OW", "N", "L", "IH")), "long": (1, ("L", "AO", "NG")), "look": (1, ("L", "UH", "K")),
    "lose": (1, ("L", "UH", "Z")), "lost": (1, ("L", "AO", "S", "T")), "love": (1, ("L", "AH", "V")),
    "make": (1, ("M", "EY", "K")), "me": (1, ("M", "IH")), "melody": (3, ("M", "EH", "L", "AH", "D", "IH")),
    "memory": (3, ("M", "EH", "M", "ER", "IH")), "mind": (1, ("M", "AY", "N", "D")), "mine": (1, ("M", "AY", "N")),
    "miss": (1, ("M", "IH", "S")), "moment": (2, ("M", "OW", "M", "AH", "N", "T")), "moon": (1, ("M", "UH", "N")),
    "moonlight": (2, ("M", "UH", "N", "L", "AY", "T")), "more": (1, ("M", "AO", "R")),
    "morning": (2, ("M", "AO", "R", "N", "IH", "NG")), "move": (1, ("M", "UH", "V")),
    "music": (2, ("M", "Y", "UH", "Z", "IH", "K")), "my": (1, ("M", "AY")), "na": (1, ("N", "AA")),
    "name": (1, ("N", "EY", "M")), "need": (1, ("N", "IH", "D")), "never": (2, ("N", "EH", "V", "ER")),
    "new": (1, ("N", "UH")), "night": (1, ("N", "AY", "T")), "no": (1, ("N", "OW")),
    "not": (1, ("N", "AA", "T")), "nothing": (2, ("N", "AH", "TH", "IH", "NG")), "now": (1, ("N", "AW")),
    "of": (1, ("AH", "V")), "oh": (1, ("OW",)), "on": (1, ("AA", "N")), "once": (1, ("W", "AH", "N", "S")),
    "one": (1, ("W", "AH", "N")), "only": (2, ("OW", "N", "L", "IH")), "ooh": (1, ("UH",)),
    "open": (2, ("OW", "P", "AH", "N")), "our": (1, ("AW", "ER")), "out": (1, ("AW", "T")),
    "over": (2, ("OW", "V", "ER")), "pain": (1, ("P", "EY", "N")), "place": (1, ("P", "L", "EY", "S")),
    "play": (1, ("P", "L", "EY")), "promise": (2, ("P", "R", "AA", "M", "IH", "S")), "rain": (1, ("R", "EY", "N")),
    "ready": (2, ("R", "EH", "D", "IH")), "real": (1, ("R", "IH", "L")),
    "remember": (3, ("R", "IH", "M", "EH", "M", "B", "ER")), "rhythm": (2, ("R", "IH", "DH", "AH", "M")),
    "right": (1, ("R", "AY", "T")), "rise": (1, ("R", "AY", "Z")), "run": (1, ("R", "AH", "N")),
    "running": (2, ("R", "AH", "N", "IH", "NG")), "say": (1, ("S", "EY")), "see": (1, ("S", "IH")),
    "shadow": (2, ("SH", "EH", "D", "OW")), "shadows": (2, ("SH", "EH", "D", "OW", "Z")),
    "shine": (1, ("SH", "AY", "N")), "shining": (2, ("SH", "AY", "N", "IH", "NG")),
    "sing": (1, ("S", "IH", "NG")), "singing": (2, ("S", "IH", "NG", "IH", "NG")), "sky": (1, ("S", "K", "AY")),
    "smile": (1, ("S", "M", "AY", "L")), "so": (1, ("S", "OW")),
    "somebody": (3, ("S", "AH", "M", "B", "AA", "D", "IH")), "someone": (2, ("S", "AH", "M", "W", "AH", "N")),
    "something": (2, ("S", "AH", "M", "TH", "IH", "NG")), "song": (1, ("S", "AO", "NG")),
    "soul": (1, ("S", "OW", "L")), "sound": (1, ("S", "AW", "N", "D")), "star": (1, ("S", "T", "AA", "R")),
    "starlight": (2, ("S", "T", "AA", "R", "L", "AY", "T")), "stars": (1, ("S", "T", "AA", "R", "Z")),
    "stay": (1, ("S", "T", "EY")), "still": (1, ("S", "T", "IH", "L")), "stop": (1, ("S", "T", "AA", "P")),
    "story": (2, ("S", "T", "AO", "R", "IH")), "sun": (1, ("S", "AH", "N")), "sweet": (1, ("S", "W", "IH", "T")),
    "take": (1, ("T", "EY", "K")), "tears": (1, ("T", "IH", "R", "Z")), "tell": (1, ("T", "EH", "L")),
    "that": (1, ("DH", "EH", "T")), "the": (1, ("DH", "AH")), "there": (1, ("DH", "EH", "R")),
    "they": (1, ("DH", "EY")), "think": (1, ("TH", "IH", "NG", "K")), "this": (1, ("DH", "IH", "S")),
    "through": (1, ("TH", "R", "UH")), "time": (1, ("T", "AY", "M")), "to": (1, ("T", "UH")),
    "today": (2, ("T", "AH", "D", "EY")), "together": (3, ("T", "AH", "G", "EH", "DH", "ER")),
    "tomorrow": (3, ("T", "AH", "M", "AA", "R", "OW")), "tonight": (2, ("T", "AH", "N", "AY", "T")),
    "touch": (1, ("T", "AH", "CH")), "true": (1, ("T", "R", "UH")), "turn": (1, ("T", "ER", "N")),
    "uh": (1, ("AH",)), "under": (2, ("AH", "N", "D", "ER")), "up": (1, ("AH", "P")),
    "us": (1, ("AH", "S")), "voice": (1, ("V", "OY", "S")), "wait": (1, ("W", "EY", "T")),
    "waiting": (2, ("W", "EY", "T", "IH", "NG")), "walk": (1, ("W", "AO", "K")), "want": (1, ("W", "AA", "N", "T")),
    "way": (1, ("W", "EY")), "we": (1, ("W", "IH")), "what": (1, ("W", "AH", "T")),
    "when": (1, ("W", "EH", "N")), "where": (1, ("W", "EH", "R")), "whisper": (2, ("W", "IH", "S", "P", "ER")),
    "who": (1, ("HH", "UH")), "whoa": (1, ("W", "OW")), "why": (1, ("W", "AY")),
    "will": (1, ("W", "IH", "L")), "wind": (1, ("W", "IH", "N", "D")), "wings": (1, ("W", "IH", "NG", "Z")),
    "wish": (1, ("W", "IH", "SH")), "with": (1, ("W", "IH", "DH")), "without": (2, ("W", "IH", "DH", "AW", "T")),
    "woah": (1, ("W", "OW")), "world": (1, ("W", "ER", "L", "D")), "yeah": (1, ("Y", "EH")),
    "yes": (1, ("Y", "EH", "S")), "you": (1, ("Y", "UH")), "you're": (1, ("Y", "UH", "R")),
    "your": (1, ("Y", "AO", "R")),
}

_NON_LEXICAL_VOCABLES: Set[str] = {
    "oh", "ooh", "ah", "aah", "eh", "uh", "umm", "mmm", "mm", "hmm",
    "yeah", "yea", "yep", "nah", "na", "la", "lala", "lalala", "nana",
    "nanana", "da", "dada", "woah", "whoa", "hey", "yo", "baby", "babe",
    "boom", "pow", "uhhuh", "oohla", "우", "아", "오", "예", "워", "라", "라라", "나나",
}

SECTION_TYPE_MAP: Dict[str, int] = {
    "verse": 0, "prechorus": 1, "pre-chorus": 1, "chorus": 2,
    "hook": 3, "bridge": 4, "intro": 5, "outro": 6, "other": 7,
}


@dataclass
class VocableCodeSwitchAnalysis:
    """가사 한 행에서 감지된 비어휘적 보컬(Vocables) 및 한-영 코드스위칭 구간 분석 결과입니다."""

    vocables: List[str]
    english_spans: List[str]
    korean_spans: List[str]
    is_english_only: bool
    is_code_switched: bool


def normalize_lyric_text(text: str) -> str:
    """NFC 정규화, 둥근 따옴표 및 키릴/그리스 동형 문자 치환, 공백 정리를 수행합니다."""
    if not text:
        return ""
    norm = unicodedata.normalize("NFC", str(text)).translate(_NORM_TRANS)
    return re.sub(r"[^\S\n]+", " ", norm).strip()


def resolve_section_type_id(header_or_tag: str) -> int:
    """섹션 헤더 문자열(예: '[Chorus 1]', 'Verse', 'Pre-Chorus')을 표준 섹션 타입 ID(0..7)로 변환합니다."""
    cleaned = re.sub(r"[^a-zA-Z\-]", "", str(header_or_tag).lower())
    for key, sec_id in SECTION_TYPE_MAP.items():
        if key in cleaned:
            return sec_id
    return SECTION_TYPE_MAP["other"]


@lru_cache(maxsize=65536)
def count_english_word_syllables(word: str) -> int:
    """내장 발음 사전과 모음 핵(Vowel Nucleus)·무음 e·접미사 규칙을 결합해 영어 단어의 음절 수를 계산합니다."""
    w = re.sub(r"[^a-z']", "", word.lower()).strip("'")
    if not w:
        return 0
    if w in _BUILTIN_EN_LEXICON:
        return _BUILTIN_EN_LEXICON[w][0]

    if "'" in w:
        base_parts = [p for p in w.split("'") if p]
        if len(base_parts) == 2 and base_parts[1] in ("m", "ll", "ve", "re", "d", "t", "s"):
            return count_english_word_syllables(base_parts[0])

    if len(w) <= 3:
        return 1

    vowel_groups = re.findall(r"[aeiouy]+", w)
    count = len(vowel_groups)

    if w.endswith("e") and not w.endswith(("le", "ee", "oe", "ye")):
        count -= 1
    elif (
        w.endswith("le")
        and len(w) > 2
        and w[-3] not in "aeiouy"
    ):
        pass
    elif w.endswith("es") and not w.endswith(("ses", "zes", "xes", "ches", "shes", "ges", "ces")):
        count -= 1
    elif w.endswith("ed") and not w.endswith(("ted", "ded")):
        count -= 1

    for pat in (r"ia", r"io", r"iu", r"ua", r"uo", r""):
        if pat in ("ia", "io", "iu", "ua", "uo") and re.search(pat, w):
            if not re.search(r"[cst]ia|[cst]io|qua|gue|gui", w):
                count += 1

    if w.startswith("mc"):
        count += 1
    if w.endswith("ism"):
        count += 1

    return max(1, count)


def count_korean_syllables(text: str) -> int:
    """문자열 내 완성형 한글 음절(가-힣)의 글자 수를 정확히 카운팅합니다."""
    return len(_HANGUL_RE.findall(normalize_lyric_text(text)))


def count_english_syllables(text: str) -> int:
    """문자열 내 모든 영어 단어의 음절 수 합계를 계산합니다."""
    norm = normalize_lyric_text(text)
    return sum(count_english_word_syllables(w) for w in _LATIN_WORD_RE.findall(norm))


@lru_cache(maxsize=65536)
def count_line_syllables(line: str, lang: str = "auto") -> int:
    """한-영 코드스위칭 및 숫자가 포함된 가사 한 행의 총 음절 수를 계산합니다."""
    del lang
    norm = normalize_lyric_text(line)
    if not norm:
        return 0
    ko_cnt = len(_HANGUL_RE.findall(norm))
    en_cnt = sum(count_english_word_syllables(w) for w in _LATIN_WORD_RE.findall(norm))
    digit_cnt = len(_DIGIT_RE.findall(norm))
    return ko_cnt + en_cnt + digit_cnt


def decompose_hangul_syllable(ch: str, merge_similar_vowels: bool = True) -> List[str]:
    """단일 한글 음절을 초성·중성·종성(겹받침 분리 포함) 자모 리스트로 분해합니다."""
    code = ord(ch)
    if not (0xAC00 <= code <= 0xD7A3):
        return []
    offset = code - 0xAC00
    cho_idx, rem = divmod(offset, 588)
    jung_idx, jong_idx = divmod(rem, 28)
    vowel = _KO_JUNG[jung_idx]
    if merge_similar_vowels:
        vowel = _KO_VOWEL_MERGE.get(vowel, vowel)
    out: List[str] = [_KO_CHO[cho_idx], vowel]
    out.extend(_KO_JONG[jong_idx])
    return out


def extract_korean_phonemes(
    text: str,
    apply_liaison: bool = True,
    merge_similar_vowels: bool = True,
) -> List[str]:
    """한국어 가사에서 연음 법칙(종성 + 뒤 음절 초성 'ㅇ')과 유사 모음 병합을 반영한 음소 시퀀스를 추출합니다."""
    norm = normalize_lyric_text(text)
    syllables: List[Optional[Tuple[int, int, int]]] = []
    for ch in norm:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            offset = code - 0xAC00
            cho_i, rem = divmod(offset, 588)
            jung_i, jong_i = divmod(rem, 28)
            syllables.append((cho_i, jung_i, jong_i))
        else:
            syllables.append(None)

    n = len(syllables)
    cho_overrides: Dict[int, str] = {}
    phonemes: List[str] = []

    for i, syl in enumerate(syllables):
        if syl is None:
            continue
        cho_i, jung_i, jong_i = syl
        cho_ph = cho_overrides.get(i, _KO_CHO[cho_i])
        vowel_ph = _KO_JUNG[jung_i]
        if merge_similar_vowels:
            vowel_ph = _KO_VOWEL_MERGE.get(vowel_ph, vowel_ph)
        jong_tuple = _KO_JONG[jong_i]

        if (
            apply_liaison
            and jong_i in _KO_LIAISON_SPLIT
            and i + 1 < n
            and syllables[i + 1] is not None
        ):
            next_cho, _, _ = syllables[i + 1]
            if next_cho == 11:
                rem_jong, moved_cho = _KO_LIAISON_SPLIT[jong_i]
                jong_tuple = rem_jong
                cho_overrides[i + 1] = moved_cho

        phonemes.append(cho_ph)
        phonemes.append(vowel_ph)
        phonemes.extend(jong_tuple)

    return phonemes


@lru_cache(maxsize=65536)
def extract_english_word_phonemes(word: str) -> Tuple[str, ...]:
    """영어 단어를 내장 사전 및 자소-음소(G2P) 규칙으로 ARPAbet 음소 시퀀스로 변환합니다."""
    w = re.sub(r"[^a-z']", "", word.lower()).strip("'")
    if not w:
        return ()
    if w in _BUILTIN_EN_LEXICON:
        return tuple(_EN_VOWEL_MERGE.get(p, p) for p in _BUILTIN_EN_LEXICON[w][1])

    rules: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
        ("tion", ("SH", "AH", "N")),
        ("sion", ("ZH", "AH", "N")),
        ("ight", ("AY", "T")),
        ("ough", ("OW",)),
        ("eigh", ("EY",)),
        ("tch", ("CH",)),
        ("dge", ("JH",)),
        ("ing", ("IH", "NG")),
        ("ang", ("EH", "NG")),
        ("ong", ("AO", "NG")),
        ("ung", ("AH", "NG")),
        ("sh", ("SH",)),
        ("ch", ("CH",)),
        ("th", ("TH",)),
        ("ph", ("F",)),
        ("wh", ("W",)),
        ("ck", ("K",)),
        ("qu", ("K", "W")),
        ("NG", ("NG",)),
        ("ng", ("NG",)),
        ("ee", ("IH",)),
        ("ea", ("IH",)),
        ("oo", ("UH",)),
        ("ou", ("AW",)),
        ("ow", ("OW",)),
        ("ai", ("EY",)),
        ("ay", ("EY",)),
        ("oi", ("OY",)),
        ("oy", ("OY",)),
        ("au", ("AO",)),
        ("aw", ("AO",)),
        ("er", ("ER",)),
        ("ir", ("ER",)),
        ("ur", ("ER",)),
        ("ar", ("AA", "R")),
        ("or", ("AO", "R")),
    )
    single_map: Dict[str, str] = {
        "a": "EH", "b": "B", "c": "K", "d": "D", "e": "EH", "f": "F",
        "g": "G", "h": "HH", "i": "IH", "j": "JH", "k": "K", "l": "L",
        "m": "M", "n": "N", "o": "AA", "p": "P", "q": "K", "r": "R",
        "s": "S", "t": "T", "u": "AH", "v": "V", "w": "W", "x": "K",
        "y": "IH", "z": "Z",
    }

    has_magic_e = (
        len(w) >= 4
        and w.endswith("e")
        and not w.endswith(("ee", "le", "re", "ve"))
        and w[-2] not in "aeiouy"
        and w[-3] in "aeiou"
    )
    long_vowel_map = {"a": "EY", "e": "IH", "i": "AY", "o": "OW", "u": "UH"}

    work = w[:-1] if has_magic_e else w
    out: List[str] = []
    idx = 0
    n = len(work)
    while idx < n:
        if work[idx] == "'":
            idx += 1
            continue
        if has_magic_e and idx == n - 2 and work[idx] in long_vowel_map:
            out.append(long_vowel_map[work[idx]])
            idx += 1
            continue
        matched = False
        for pat, phs in rules:
            if work.startswith(pat, idx):
                out.extend(_EN_VOWEL_MERGE.get(p, p) for p in phs)
                idx += len(pat)
                matched = True
                break
        if matched:
            continue
        ch = work[idx]
        if idx + 1 < n and work[idx + 1] == ch and ch not in "aeiou":
            idx += 1
        ph = single_map.get(ch)
        if ph is not None:
            out.append(_EN_VOWEL_MERGE.get(ph, ph))
        idx += 1

    return tuple(out) if out else ("AH",)


def extract_english_phonemes(text: str) -> List[str]:
    """문자열 내 영어 단어들을 ARPAbet 음소 시퀀스로 추출합니다."""
    norm = normalize_lyric_text(text)
    out: List[str] = []
    for w in _LATIN_WORD_RE.findall(norm):
        out.extend(extract_english_word_phonemes(w))
    return out


def extract_line_phonemes(line: str, apply_ko_liaison: bool = True) -> List[str]:
    """한-영 혼용 가사 한 행을 언어별 음소 분석기에 통과시켜 통합 음소 시퀀스로 변환합니다."""
    norm = normalize_lyric_text(line)
    out: List[str] = []
    for tok in _TOKEN_RE.findall(norm):
        first = tok[0]
        if "가" <= first <= "힣":
            out.extend(extract_korean_phonemes(tok, apply_liaison=apply_ko_liaison))
        elif first.isalpha():
            out.extend(extract_english_word_phonemes(tok))
        elif first.isdigit():
            out.extend(list(tok))
    return out


def extract_end_rhyme_phonemes(line: str) -> Tuple[str, ...]:
    """가사 행의 마지막 발화 단어에서 각운(Rhyme Nucleus + Coda) 음소 시퀀스를 추출합니다."""
    norm = normalize_lyric_text(line)
    tokens = _TOKEN_RE.findall(norm)
    if not tokens:
        return ()
    last_tok = tokens[-1]
    if "가" <= last_tok[0] <= "힣":
        phs = extract_korean_phonemes(last_tok, apply_liaison=True)
        ko_vowels = set(_KO_JUNG)
        for idx in range(len(phs) - 1, -1, -1):
            if phs[idx] in ko_vowels:
                return tuple(phs[idx:])
        return tuple(phs[-2:])
    if last_tok[0].isalpha():
        phs = list(extract_english_word_phonemes(last_tok))
        for idx in range(len(phs) - 1, -1, -1):
            if phs[idx] in _EN_ARPABET_VOWELS:
                return tuple(phs[idx:])
        return tuple(phs[-2:])
    return (last_tok[-1],)


def compute_rhyme_similarity(line_a: str, line_b: str) -> float:
    """두 가사 행 끝의 라임(Nucleus + Coda) 음소 일치도를 `0.0 ~ 1.0` 점수로 계산합니다."""
    rhyme_a = extract_end_rhyme_phonemes(line_a)
    rhyme_b = extract_end_rhyme_phonemes(line_b)
    if not rhyme_a or not rhyme_b:
        return 0.0
    if rhyme_a == rhyme_b:
        return 1.0
    if rhyme_a[0] == rhyme_b[0]:
        coda_a = rhyme_a[1:]
        coda_b = rhyme_b[1:]
        if not coda_a and not coda_b:
            return 1.0
        overlap = len(set(coda_a) & set(coda_b))
        denom = max(1, max(len(coda_a), len(coda_b)))
        return 0.65 + 0.35 * (float(overlap) / float(denom))
    if rhyme_a[-1] == rhyme_b[-1]:
        return 0.35
    return 0.0


def detect_vocables_and_codeswitch(line: str) -> VocableCodeSwitchAnalysis:
    """가사 한 행에서 비어휘적 보컬('Oh', 'Yeah', 'La-la') 및 한-영 코드스위칭 구간을 감지합니다."""
    norm = normalize_lyric_text(line)
    tokens = _TOKEN_RE.findall(norm)
    vocables: List[str] = []
    en_spans: List[str] = []
    ko_spans: List[str] = []

    for tok in tokens:
        clean_low = re.sub(r"[^a-z가-힣]", "", tok.lower())
        if clean_low in _NON_LEXICAL_VOCABLES or (
            len(clean_low) >= 2 and len(set(clean_low)) == 1 and clean_low[0] in "oauemln"
        ):
            vocables.append(tok)
        if "가" <= tok[0] <= "힣":
            ko_spans.append(tok)
        elif tok[0].isalpha():
            en_spans.append(tok)

    has_ko = len(ko_spans) > 0
    has_en = len(en_spans) > 0
    return VocableCodeSwitchAnalysis(
        vocables=vocables,
        english_spans=en_spans,
        korean_spans=ko_spans,
        is_english_only=(has_en and not has_ko),
        is_code_switched=(has_ko and has_en),
    )


def compute_codeswitch_preserve_score(src_line: str, pred_line: str) -> float:
    """원문 가사에 포함된 비어휘적 보컬 또는 코드스위칭 영어 구절이 번역문에 보존된 비율을 반환합니다."""
    src_info = detect_vocables_and_codeswitch(src_line)
    target_keep = {w.lower() for w in src_info.vocables if w[0].isascii()}
    if src_info.is_code_switched:
        for w in src_info.english_spans:
            if len(w) >= 2:
                target_keep.add(w.lower())
    if not target_keep:
        return 1.0
    pred_words = {w.lower() for w in _LATIN_WORD_RE.findall(normalize_lyric_text(pred_line))}
    kept = len(target_keep & pred_words)
    return float(kept) / float(len(target_keep))


def compute_phoneme_distinct_2(lines: Sequence[str]) -> Optional[float]:
    """arXiv:2309.11093 식 (3)에 따라 섹션 내 음소 바이그램 다양성(`Pho`)을 계산합니다."""
    seq: List[str] = []
    for line in lines:
        phs = extract_line_phonemes(line)
        if phs:
            seq.extend(phs)
            seq.append("<eos>")
    if len(seq) < 2:
        return None
    bigrams = list(zip(seq, seq[1:]))
    return float(len(set(bigrams))) / float(len(bigrams))


def compute_song_phoneme_metrics(sections: Sequence[Sequence[str]]) -> Tuple[float, float]:
    """arXiv:2309.11093 3.2절의 곡 단위 음소 다양성 평균(`Phodeg`)과 표준편차(`Phovar`)를 반환합니다."""
    vals = [v for v in (compute_phoneme_distinct_2(sec) for sec in sections) if v is not None]
    if not vals:
        return 0.0, 0.0
    return float(mean(vals)), float(pstdev(vals)) if len(vals) > 1 else 0.0


def compute_line_scd(src_syllables: int, tgt_syllables: int) -> float:
    """arXiv:2309.11093 식 (4)에 따른 행 단위 음절 수 거리(Syllable Count Distance)를 계산합니다."""
    s = max(0, int(src_syllables))
    t = max(0, int(tgt_syllables))
    diff = abs(s - t)
    if s > 0 and t > 0:
        return 0.5 * ((float(diff) / float(s)) + (float(diff) / float(t)))
    if s == 0 and t == 0:
        return 0.0
    return 1.0


def compute_section_scd(
    target_syllables: Sequence[int],
    pred_syllables: Sequence[int],
) -> Tuple[float, float]:
    """섹션 전체 행들에 대한 평균 음절 수 거리(SCD)와 음절 불일치 행 비율(Error Rate)을 반환합니다."""
    if not target_syllables:
        return 0.0, 0.0
    n = len(target_syllables)
    padded_pred = list(pred_syllables[:n]) + [0] * max(0, n - len(pred_syllables))
    scd_vals = [compute_line_scd(s, t) for s, t in zip(target_syllables, padded_pred)]
    err_cnt = sum(1 for s, t in zip(target_syllables, padded_pred) if int(s) != int(t))
    return float(sum(scd_vals) / float(n)), float(err_cnt) / float(n)


def _sequence_ngram_jaccard(seq_a: Sequence[str], seq_b: Sequence[str]) -> float:
    """두 시퀀스의 유니그램·바이그램 멀티셋 코사인/자카드 혼합 유사도(`0.0 ~ 1.0`)를 계산합니다."""
    if not seq_a and not seq_b:
        return 1.0
    if not seq_a or not seq_b:
        return 0.0
    if list(seq_a) == list(seq_b):
        return 1.0

    c1_a = Counter(seq_a)
    c1_b = Counter(seq_b)
    inter1 = sum(min(cnt, c1_b[k]) for k, cnt in c1_a.items())
    union1 = sum(max(cnt, c1_b[k]) for k, cnt in (c1_a | c1_b).items())
    sim1 = float(inter1) / float(max(1, union1))

    if len(seq_a) >= 2 and len(seq_b) >= 2:
        bg_a = Counter(zip(seq_a, seq_a[1:]))
        bg_b = Counter(zip(seq_b, seq_b[1:]))
        inter2 = sum(min(cnt, bg_b[k]) for k, cnt in bg_a.items())
        union2 = sum(max(cnt, bg_b[k]) for k, cnt in (bg_a | bg_b).items())
        sim2 = float(inter2) / float(max(1, union2))
        return 0.5 * sim1 + 0.5 * sim2
    return sim1


def compute_phoneme_ssm(lines: Sequence[str]) -> List[List[float]]:
    """섹션 내 L개 행 간의 음소 시퀀스 및 행 끝 라임 유사도를 결합한 자기유사도 행렬(SSM)을 계산합니다."""
    n = len(lines)
    if n == 0:
        return []
    ph_seqs = [extract_line_phonemes(line) for line in lines]
    matrix: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            ph_sim = _sequence_ngram_jaccard(ph_seqs[i], ph_seqs[j])
            rhyme_sim = compute_rhyme_similarity(lines[i], lines[j])
            sim = min(1.0, 0.75 * ph_sim + 0.25 * rhyme_sim)
            matrix[i][j] = sim
            matrix[j][i] = sim
    return matrix


def compute_semantic_ssm(lines: Sequence[str]) -> List[List[float]]:
    """섹션 내 L개 행 간의 어휘 및 문자 바이그램 기반 의미·구조 반복 자기유사도 행렬(SSM)을 계산합니다."""
    n = len(lines)
    if n == 0:
        return []
    char_seqs = [
        [ch.lower() for ch in normalize_lyric_text(line) if not ch.isspace()]
        for line in lines
    ]
    tok_seqs = [
        [t.lower() for t in _TOKEN_RE.findall(normalize_lyric_text(line))]
        for line in lines
    ]
    matrix: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            c_sim = _sequence_ngram_jaccard(char_seqs[i], char_seqs[j])
            t_sim = _sequence_ngram_jaccard(tok_seqs[i], tok_seqs[j])
            sim = 0.5 * c_sim + 0.5 * t_sim
            matrix[i][j] = sim
            matrix[j][i] = sim
    return matrix


def compute_repetition_group_ids(
    lines: Sequence[str],
    similarity_threshold: float = 0.85,
    max_groups: int = 16,
) -> List[int]:
    """섹션 내 동일하거나 고도로 유사한 반복 행들에 동일한 반복 그룹 ID(0..max_groups-1)를 부여합니다."""
    n = len(lines)
    if n == 0:
        return []
    sem_ssm = compute_semantic_ssm(lines)
    group_ids: List[int] = []
    next_gid = 0
    for i in range(n):
        assigned: Optional[int] = None
        for j in range(i):
            if sem_ssm[i][j] >= similarity_threshold:
                assigned = group_ids[j]
                break
        if assigned is None:
            assigned = min(next_gid, max_groups - 1)
            next_gid += 1
        group_ids.append(assigned)
    return group_ids


def compute_ssm_alignment_score(
    src_ssm: Sequence[Sequence[float]],
    pred_ssm: Sequence[Sequence[float]],
) -> float:
    """원문 섹션의 반복 행렬 M_src와 번역 섹션의 음소 반복 행렬 M_pred 간 정합성 점수(1 - L1 오차)를 계산합니다."""
    n_src = len(src_ssm)
    n_pred = len(pred_ssm)
    if n_src <= 1:
        return 1.0 if n_pred == n_src else 0.5
    if n_pred == 0:
        return 0.0

    n = max(n_src, n_pred)
    diff_sum = 0.0
    pair_cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            v_src = float(src_ssm[i][j]) if (i < n_src and j < len(src_ssm[i])) else 0.0
            v_pred = float(pred_ssm[i][j]) if (i < n_pred and j < len(pred_ssm[i])) else 0.0
            diff_sum += abs(v_src - v_pred)
            pair_cnt += 1

    if pair_cnt == 0:
        return 1.0
    return float(max(0.0, min(1.0, 1.0 - (diff_sum / float(pair_cnt)))))


def compute_section_rhyme_and_consistency_score(
    src_lines: Sequence[str],
    pred_lines: Sequence[str],
    repeat_group_ids: Optional[Sequence[int]] = None,
) -> float:
    """섹션 내 행 끝 각운 보존도, 반복 행 동일 번역 유지율, 코드스위칭 보존율을 통합 평가합니다."""
    if not src_lines or not pred_lines:
        return 0.0
    n = len(src_lines)
    padded_pred = list(pred_lines[:n]) + [""] * max(0, n - len(pred_lines))
    gids = (
        list(repeat_group_ids[:n])
        if (repeat_group_ids is not None and len(repeat_group_ids) >= n)
        else compute_repetition_group_ids(src_lines)
    )

    repeat_pairs = 0
    repeat_sim_sum = 0.0
    rhyme_pairs = 0
    rhyme_score_sum = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            if gids[i] == gids[j]:
                repeat_pairs += 1
                tok_a = [t.lower() for t in _TOKEN_RE.findall(padded_pred[i])]
                tok_b = [t.lower() for t in _TOKEN_RE.findall(padded_pred[j])]
                repeat_sim_sum += _sequence_ngram_jaccard(tok_a, tok_b)
            src_rhyme = compute_rhyme_similarity(src_lines[i], src_lines[j])
            if src_rhyme >= 0.5 or j == i + 1:
                rhyme_pairs += 1
                pred_rhyme = compute_rhyme_similarity(padded_pred[i], padded_pred[j])
                rhyme_score_sum += min(1.0, pred_rhyme / max(0.35, src_rhyme))

    repeat_score = (repeat_sim_sum / float(repeat_pairs)) if repeat_pairs > 0 else 1.0
    rhyme_score = (rhyme_score_sum / float(rhyme_pairs)) if rhyme_pairs > 0 else 0.75
    cs_scores = [
        compute_codeswitch_preserve_score(s_ln, p_ln)
        for s_ln, p_ln in zip(src_lines, padded_pred)
    ]
    cs_score = sum(cs_scores) / float(len(cs_scores))

    return float(max(0.0, min(1.0, 0.40 * rhyme_score + 0.35 * repeat_score + 0.25 * cs_score)))


def synthesize_rhythm_guide_waveform(
    syllable_counts: Sequence[int],
    sample_rate: int = 24000,
    sec_per_syllable: float = 0.24,
    line_pause_sec: float = 0.24,
    base_freq_hz: float = 220.0,
) -> Tuple[torch.Tensor, List[int], List[int]]:
    """원곡 보컬 WAV가 없을 때 행별 음절 수에 맞춘 리듬 가이드 보컬 파형과 행별 시작·종료 프레임을 생성합니다."""
    samples_per_frame = int(round(sample_rate / 12.5))
    syl_samples = max(samples_per_frame, int(round(sec_per_syllable * sample_rate)))
    pause_samples = max(samples_per_frame, int(round(line_pause_sec * sample_rate)))

    pieces: List[torch.Tensor] = []
    start_frames: List[int] = []
    end_frames: List[int] = []
    cursor_samples = 0
    pentatonic_ratios = (1.0, 1.125, 1.25, 1.5, 1.667, 2.0)

    for line_idx, raw_syl in enumerate(syllable_counts):
        n_syl = max(1, int(raw_syl))
        start_f = cursor_samples // samples_per_frame
        line_chunks: List[torch.Tensor] = []
        for s_i in range(n_syl):
            t = torch.linspace(0.0, float(syl_samples) / float(sample_rate), syl_samples)
            freq = base_freq_hz * pentatonic_ratios[(line_idx + s_i) % len(pentatonic_ratios)]
            env = torch.sin(math.pi * torch.linspace(0.0, 1.0, syl_samples)).pow(0.7)
            tone = (
                0.25 * torch.sin(2.0 * math.pi * freq * t)
                + 0.08 * torch.sin(4.0 * math.pi * freq * t)
            ) * env
            line_chunks.append(tone.float())
        line_wav = torch.cat(line_chunks, dim=0)
        pieces.append(line_wav)
        cursor_samples += int(line_wav.numel())
        end_f = max(start_f + 1, (cursor_samples + samples_per_frame - 1) // samples_per_frame)
        start_frames.append(start_f)
        end_frames.append(end_f)

        pause_wav = torch.zeros(pause_samples, dtype=torch.float32)
        pieces.append(pause_wav)
        cursor_samples += pause_samples

    full_wav = (
        torch.cat(pieces, dim=0)
        if pieces
        else torch.zeros(samples_per_frame * 4, dtype=torch.float32)
    )
    return full_wav, start_frames, end_frames


def align_lyric_tokens_to_frames(
    token_ids: Sequence[int],
    start_frame: int,
    end_frame: int,
    last_token_frame: int,
    lead_frames: int = 1,
) -> Tuple[List[Tuple[int, int]], int]:
    """단일 가사 행의 서브워드 토큰들을 지정된 보컬 시작·종료 프레임 구간에 단조 증가하도록 정렬합니다."""
    num_tokens = len(token_ids)
    if num_tokens == 0:
        return [], last_token_frame

    eff_start = max(0, start_frame - max(0, lead_frames), last_token_frame + 1)
    span_len = max(1, end_frame - eff_start)
    placements: List[Tuple[int, int]] = []
    cur_last = last_token_frame

    for idx, tid in enumerate(token_ids):
        ideal = eff_start + (idx * span_len) // max(1, num_tokens)
        f_pos = max(ideal, cur_last + 1)
        placements.append((f_pos, int(tid)))
        cur_last = f_pos
    return placements, cur_last


def run_phonetics_self_test() -> None:
    """한-영 음절 카운팅, 연음 음소 분해, 각운 추출, 코드스위칭 감지, SSM 및 SCD 계산을 자체 검증합니다."""
    ko_line = "같이 걸어가요 오늘 밤"
    en_line = "Walking together into the night"
    cs_line = "Oh baby 오늘 밤에 dance with me"

    ko_syl = count_line_syllables(ko_line)
    en_syl = count_line_syllables(en_line)
    cs_syl = count_line_syllables(cs_line)
    assert ko_syl == 9, f"한국어 음절 수 오류: 기대 9, 실제 {ko_syl}"
    assert en_syl == 9, f"영어 음절 수 오류: 기대 9, 실제 {en_syl}"
    assert cs_syl == 10, f"코드스위칭 음절 수 오류: 기대 10, 실제 {cs_syl}"

    liaison_ph = extract_korean_phonemes("같이")
    assert "ㅌ" in liaison_ph, f"연음 추출 오류: {liaison_ph}"

    rhyme_1 = compute_rhyme_similarity("Shining in the night", "Holding on so tight")
    rhyme_2 = compute_rhyme_similarity("Shining in the night", "Walking in the rain")
    assert rhyme_1 > rhyme_2 and rhyme_1 >= 0.9, f"각운 유사도 오류: {rhyme_1} vs {rhyme_2}"

    cs_analysis = detect_vocables_and_codeswitch(cs_line)
    assert cs_analysis.is_code_switched and len(cs_analysis.vocables) >= 2

    sec_ko = [
        "빛나는 별빛 아래 너와 나",
        "영원히 함께할 이 순간",
        "빛나는 별빛 아래 너와 나",
        "멈추지 않는 우리의 시간",
    ]
    sec_en = [
        "Under the shining starlight you and me",
        "Forever together in this time",
        "Under the shining starlight you and me",
        "Never stopping rhythm in our mind",
    ]
    ssm_src = compute_semantic_ssm(sec_ko)
    ssm_pred = compute_phoneme_ssm(sec_en)
    ssm_score = compute_ssm_alignment_score(ssm_src, ssm_pred)
    scd_val, err_rate = compute_section_scd(
        [count_line_syllables(x) for x in sec_ko],
        [count_line_syllables(x) for x in sec_en],
    )
    assert ssm_src[0][2] == 1.0 and ssm_pred[0][2] == 1.0
    assert 0.0 <= ssm_score <= 1.0

    print(
        f"[Self-Test] Phonetics 검증 완료 | "
        f"KO음절={ko_syl}, EN음절={en_syl}, CS음절={cs_syl} | "
        f"Rhyme(night/tight)={rhyme_1:.2f} | SSM정합={ssm_score:.3f} | SCD={scd_val:.3f} (err={err_rate:.2f})"
    )
    print("[OK] arXiv:2309.11093 음절·음소·라임·SSM 분석기 무결성 검증 통과!")
