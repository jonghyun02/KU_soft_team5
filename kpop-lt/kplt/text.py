"""글 정리, 음절 세기, 가사 마디 나누기, meta.csv 읽기.

음절
- 영어: syllables 라이브러리 (그림 2 설명 "we obtained the syllable counts by employing the syllables library").
- 한국어: 글자 수 (그림 2 설명 "we simply counted the number of characters").
- [해석] 한 줄에 두 언어가 섞이는 일이 흔해(K-pop 원문 속 영어, 영어판 속 한글) 양쪽 모두 같은 규칙으로 센다:
  한글 글자 1개 = 1음절, 영어 단어는 syllables, 숫자는 한 자 1음절.

정리 (normalize_text)
- [해석] Genius 글에는 둥근 따옴표(’)와 라틴 글자처럼 보이는 키릴 글자(е, а, о ...)가 섞여 있어
  단어가 쪼개지고 음절·음소가 틀린다. 음절·음소를 세기 전과 데이터를 만들 때 곧은 따옴표와 라틴 글자로 바꾼다.
"""
from __future__ import annotations

import csv
import re
import unicodedata
from functools import lru_cache

import syllables as _syl

HANGUL = re.compile(r"[가-힣]")
LATIN_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")
DIGIT = re.compile(r"[0-9]")
SECTION_HEADER = re.compile(r"^\s*\[[^\]]*\]\s*$")

_APOS = {"’": "'", "‘": "'", "ʼ": "'", "`": "'", "´": "'"}
# 키릴·그리스 글자 중 라틴 글자와 모양이 같은 것
_HOMOGLYPH = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ѕ": "s", "ј": "j",
    "ԁ": "d", "һ": "h", "ӏ": "l", "ԛ": "q", "ԝ": "w", "ɡ": "g",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
    "Х": "X", "У": "Y", "І": "I", "Ј": "J", "Ѕ": "S",
    "ο": "o", "α": "a", "ε": "e", "ι": "i", "ν": "v", "Ο": "O", "Α": "A", "Ε": "E", "Ι": "I", "Τ": "T",
}
_TRANS = str.maketrans({**_APOS, **_HOMOGLYPH})


def norm_apostrophe(s: str) -> str:
    return s.translate(str.maketrans(_APOS))


def normalize_text(s: str) -> str:
    """NFC, 곧은 따옴표, 라틴 글자. 특수 공백은 보통 공백으로."""
    s = unicodedata.normalize("NFC", s).translate(_TRANS)
    return re.sub(r"[^\S\n]", " ", s)


@lru_cache(maxsize=300_000)
def _en_word(word: str) -> int:
    return max(1, _syl.estimate(word))


@lru_cache(maxsize=300_000)
def count_syllables(line: str) -> int:
    line = normalize_text(line)
    return (len(HANGUL.findall(line)) + sum(_en_word(w.lower()) for w in LATIN_WORD.findall(line))
            + len(DIGIT.findall(line)))


def en_syllables(line: str) -> int:
    return count_syllables(line)


def ko_syllables(line: str) -> int:
    return count_syllables(line)


def syllables(line: str, lang: str = "") -> int:
    return count_syllables(line)


def has_hangul(s: str) -> bool:
    return bool(HANGUL.search(s))


def split_sections(lyrics: str) -> list[list[str]]:
    """Genius 형식 가사를 마디로 나눈다. [Verse] 같은 머리줄과 빈 줄을 모두 마디 경계로 본다.

    머리줄은 버리고, 빈 마디는 버리고, 줄은 normalize_text 로 정리한다.
    """
    sections: list[list[str]] = []
    cur: list[str] = []
    for raw in lyrics.splitlines():
        line = normalize_text(raw).strip()
        if SECTION_HEADER.match(line) or not line:
            if cur:
                sections.append(cur)
                cur = []
            continue
        cur.append(line)
    if cur:
        sections.append(cur)
    return sections


def load_meta(path) -> dict[str, dict]:
    """meta.csv 를 읽는다. 944번 줄처럼 빈 칸이 하나 더 끼어 장르가 밀린 줄을 고친다. 장르는 소문자."""
    rows = list(csv.reader(open(path, encoding="utf-8")))
    head = rows[0]
    out = {}
    for r in rows[1:]:
        if len(r) == len(head) + 1 and r[4] == "":
            r = r[:4] + r[5:]
        d = dict(zip(head, r))
        d["GENRE"] = d["GENRE"].lower()
        out[d["LID"]] = d
    return out
