"""발음과 음절.

영어는 eSpeak NG(en-us) IPA 음소를 낱말 단위로 받아 음절로 묶는다. 한국어는 한글 음절을 유니코드로
풀어 초성·글라이드·모음·종성으로 나눈다. 음절마다 모음 기호(IPA)를 붙여 두면 timing.py 가 시간 열로 늘리고
vowels.py 의 모음 거리표로 비교한다.

eSpeak 부분은 C:\\Users\\jongh\\Desktop\\Study\\메모\\더빙 실습 코드\\phones.py 의 engine()/pronunciation() 을
옮겼다. eSpeak 는 스레드에 안전하지 않아서 초기화·목소리 고르기·변환을 한 잠금 안에서 한다.
"""
from __future__ import annotations

import ctypes as c
import re
import threading
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from .textnorm import latin_syllables, read_digits, speak_english_numbers

EN_VOICE = "en-us"

_LOCK = threading.RLock()
_ENGINE: c.CDLL | None = None
_VOICE: str | None = None

# espeak_TextToPhonemes 의 phonememode: 2 = IPA 출력, 8~23비트 = 음소 사이 구분자('|')
_PHONEME_MODE = 2 | (ord("|") << 8)
_TEXT_MODE_UTF8 = 1
_LANG_SWITCH = re.compile(r"\([A-Za-z\-]+\)")   # (fr) 같은 언어 바뀜 표시


def engine() -> c.CDLL:
    """eSpeak NG 라이브러리를 한 번만 열고 초기화한다."""
    global _ENGINE
    with _LOCK:
        if _ENGINE is None:
            from espeakng_loader import get_data_path, get_library_path

            library = c.CDLL(get_library_path())
            library.espeak_Initialize.argtypes = [c.c_int, c.c_int, c.c_char_p, c.c_int]
            library.espeak_SetVoiceByName.argtypes = [c.c_char_p]
            library.espeak_TextToPhonemes.argtypes = [c.POINTER(c.c_void_p), c.c_int, c.c_int]
            library.espeak_TextToPhonemes.restype = c.c_char_p
            # 오디오 출력을 쓰지 않는 동기 모드. 데이터 경로가 잘못되면 오류를 돌려준다.
            parent = str(Path(get_data_path()).parent).encode()
            if library.espeak_Initialize(2, 0, parent, 0x8000) <= 0:
                raise RuntimeError("eSpeak NG 발음 자료를 열지 못했습니다.")
            _ENGINE = library
        return _ENGINE


@lru_cache(maxsize=8192)
def pronunciation(text: str, language: str = EN_VOICE) -> tuple[tuple[str, ...], ...]:
    """글을 eSpeak IPA 음소로 바꾼다.

    낱말마다 음소 튜플 하나를 돌려준다. 강세 기호 ˈ ˌ 는 빼고 길이 기호 ː 는 남긴다.
    """
    global _VOICE
    clauses: list[str] = []
    with _LOCK:
        library = engine()
        if _VOICE != language:
            if library.espeak_SetVoiceByName(language.encode()) != 0:
                _VOICE = None                      # 다음 호출에서 목소리를 다시 고르게 한다
                raise ValueError(f"발음 언어를 찾을 수 없습니다: {language}")
            _VOICE = language
        buffer = c.create_string_buffer(text.encode("utf-8"))
        cursor = c.cast(buffer, c.c_void_p)
        while cursor.value:
            clause = library.espeak_TextToPhonemes(c.byref(cursor), _TEXT_MODE_UTF8, _PHONEME_MODE)
            if clause is None:
                raise RuntimeError("문장 발음 변환에 실패했습니다.")
            clauses.append(clause.decode("utf-8", errors="replace"))
    words: list[tuple[str, ...]] = []
    for clause in clauses:
        for raw in _LANG_SWITCH.sub("", clause).split():
            phones = tuple(t for t in (p.strip().replace("ˈ", "").replace("ˌ", "") for p in raw.split("|")) if t)
            if phones:
                words.append(phones)
    return tuple(words)


@dataclass
class Syl:
    """음절 하나. nucleus 는 모음 기호(IPA) 1개, 이중모음은 2개. 모음이 없는 음절은 빈 목록."""
    onset: list[str] = field(default_factory=list)
    glide: str | None = None
    nucleus: list[str] = field(default_factory=list)
    coda: list[str] = field(default_factory=list)
    src: str = ""


# ---------------------------------------------------------------------------- 영어
class EnVowel(NamedTuple):
    """eSpeak 모음 토큰 하나가 만드는 음절 봉우리와 꼬리 자음."""
    nuclei: tuple[tuple[str, ...], ...]   # 음절마다 모음 기호(보통 음절 1개)
    coda: tuple[str, ...] = ()            # 토큰에 붙은 자음(r 색 모음의 ɹ, əl 의 l)


def _v(*nuclei: str, coda: str = "") -> EnVowel:
    return EnVowel(tuple(tuple(n.split()) for n in nuclei), tuple(coda.split()))


# eSpeak NG 1.52.0(espeakng-loader 0.2.4) en-us 모음 토큰 -> 모음 기호(강세 ˈˌ·길이 ː 를 뺀 꼴로 찾음).
# 2026-11-21 에 D:\transformer-base\enko-nmt\data\clean_v2 의 valid.tsv·test.tsv 영어 11,997줄,
# C:\projects\dub-rhythm-demo\work\lines.json 160줄, 감탄사·약어·외래 이름 26줄을 바꿔 모은 모음 토큰 32종이
# 모두 여기서 풀린다(ɑ̃ 는 결합 부호를 떼고 ɑ). e, a, o, ɝ, oɹ, aʊɚ 는 표본에 없었지만 명세 예시·다른 판 대비로 둔다.
# 음절 수는 CMU 발음 사전과 맞춰 정했다(idea 3, theater 3, science 2, fire 2, period 3, viral 2).
EN_NUCLEUS: dict[str, EnVowel] = {
    "i": _v("i"), "ɪ": _v("ɪ"), "ᵻ": _v("ɨ"),
    "e": _v("e"), "ɛ": _v("ɛ"), "æ": _v("æ"),
    "a": _v("a"), "ɑ": _v("ɑ"), "ʌ": _v("ʌ"),
    "ə": _v("ə"), "ɚ": _v("ə"), "ɜ": _v("ɜ"), "ɝ": _v("ɜ"),
    # ɐ(about·a 의 약한 모음)는 PanPhon 이 e 와 똑같은 자질로 적어서(차이 0) ə 로 돌린다.
    "ɐ": _v("ə"),
    "u": _v("u"), "ʊ": _v("ʊ"), "o": _v("o"), "ɔ": _v("ɔ"),
    # 이중모음은 두 기호로 쪼갠다
    "eɪ": _v("e ɪ"), "aɪ": _v("a ɪ"), "aʊ": _v("a ʊ"), "oʊ": _v("o ʊ"), "ɔɪ": _v("ɔ ɪ"),
    # r 이 붙은 모음은 모음 + 꼬리 ɹ
    "ɪɹ": _v("ɪ", coda="ɹ"), "ɛɹ": _v("ɛ", coda="ɹ"), "ʊɹ": _v("ʊ", coda="ɹ"),
    "ɑɹ": _v("ɑ", coda="ɹ"), "ɔɹ": _v("ɔ", coda="ɹ"), "oɹ": _v("o", coda="ɹ"),
    # people, little 의 əl 은 ə + 꼬리 l
    "əl": _v("ə", coda="l"),
    # 두 음절 토큰: idea·previous·museum 의 iə, science·quiet 의 aɪə, fire·tired 의 aɪɚ
    "iə": _v("i", "ə"), "aɪə": _v("a ɪ", "ə"), "aɪɚ": _v("a ɪ", "ə"), "aʊɚ": _v("a ʊ", "ə"),
    # 글자를 늘여 쓴 감탄사(Aaaah!)나 철자 읽기(TTGAAAGG…)에서 같은 모음 둘이 한 토큰으로 붙어 나온다.
    # 길게 늘인 모음 하나로 본다. ɐɐ 는 표본에 없었고 Aaaaaaaaah! 를 넣어 봐서 찾았다.
    "ææ": _v("æ"), "ɐɐ": _v("ə"),
}
# 바로 뒤가 ɹ 이면 한 음절로 읽는 토큰: period·zero·theory 의 iə, viral·requiring 의 aɪɚ
EN_NUCLEUS_BEFORE_R: dict[str, EnVowel] = {
    "iə": _v("i ə"),
    "aɪɚ": _v("a ɪ"),
}

EN_VOWELS: frozenset[str] = frozenset(
    sym for table in (EN_NUCLEUS, EN_NUCLEUS_BEFORE_R) for ent in table.values() for nuc in ent.nuclei for sym in nuc
)

_SYLLABIC_MARK = "\u0329"       # n̩ 처럼 성절 자음에 붙는 표시
_VOWEL_CHARS = frozenset("aeiouyɑɐɒæɛəɘɚɜɝɞɪɨɯɔɵœøʊʌʉɶɤʏᵻᵿ")
_GLIDES = ("j", "w")

# 실행 중 사상표에 없어서 '모음 없는 음절'로 센 토큰. 보고용으로 모은다.
UNKNOWN_TOKENS: set[str] = set()


def _strip_marks(token: str) -> str:
    """길이 기호와 결합 부호(비음화 등)를 뺀다."""
    base = unicodedata.normalize("NFD", token.replace("ː", "").replace("ˑ", ""))
    return unicodedata.normalize("NFC", "".join(ch for ch in base if not unicodedata.combining(ch)))


def _classify(token: str, nxt: str | None) -> tuple[str, EnVowel | str]:
    """토큰 하나를 ('V', EnVowel) | ('S', 성절 자음) | ('C', 자음) 로 가른다."""
    tok = token.replace("ː", "")
    if nxt is not None:
        nxt = nxt.replace("ː", "")
    if nxt == "ɹ" and tok in EN_NUCLEUS_BEFORE_R:
        return "V", EN_NUCLEUS_BEFORE_R[tok]
    if tok in EN_NUCLEUS:
        return "V", EN_NUCLEUS[tok]
    if _SYLLABIC_MARK in tok:
        return "S", token
    if any(ch in _VOWEL_CHARS for ch in tok):
        bare = _strip_marks(tok)
        if nxt == "ɹ" and bare in EN_NUCLEUS_BEFORE_R:
            return "V", EN_NUCLEUS_BEFORE_R[bare]
        if bare in EN_NUCLEUS:
            return "V", EN_NUCLEUS[bare]
        UNKNOWN_TOKENS.add(token)
        return "S", token
    return "C", token


def _en_word_syllables(phones: tuple[str, ...]) -> list[Syl]:
    """낱말 하나의 음소 열을 음절로 나눈다.

    모음 사이 자음이 1개면 뒤 음절 첫소리, 2개 이상이면 마지막 1개만 뒤 첫소리이고 나머지는 앞 음절 꼬리.
    낱말 앞 자음은 첫 음절 첫소리, 끝 자음은 마지막 음절 꼬리. 첫소리 끝의 j·w 는 글라이드로 뗀다.
    """
    # 펼친 열: ('C', (자음,)) | ('N', 모음 기호들) | ('S', (성절 자음,))
    flat: list[tuple[str, tuple[str, ...]]] = []
    for k, ph in enumerate(phones):
        kind, val = _classify(ph, phones[k + 1] if k + 1 < len(phones) else None)
        if isinstance(val, EnVowel):
            flat.extend(("N", nuc) for nuc in val.nuclei)
            flat.extend(("C", (cc,)) for cc in val.coda)
        else:
            flat.append((kind, (val,)))

    peaks = [i for i, (kind, _) in enumerate(flat) if kind != "C"]
    if not peaks:
        # psst 처럼 모음이 하나도 없는 낱말은 모음 없는 음절 하나로 센다
        cons = [v[0] for _, v in flat]
        return [Syl(onset=cons, src="".join(cons))] if cons else []

    sylls = [Syl() for _ in peaks]
    for n, p in enumerate(peaks):
        kind, val = flat[p]
        if kind == "N":
            sylls[n].nucleus = list(val)
        else:
            sylls[n].coda.append(val[0])            # 성절 자음은 꼬리 앞자리에 둔다
    sylls[0].onset = [v[0] for _, v in flat[:peaks[0]]]
    for n in range(len(peaks) - 1):
        between = [v[0] for _, v in flat[peaks[n] + 1:peaks[n + 1]]]
        if between:
            sylls[n].coda.extend(between[:-1])
            sylls[n + 1].onset = between[-1:]
    sylls[-1].coda.extend(v[0] for _, v in flat[peaks[-1] + 1:])

    for s in sylls:
        if s.nucleus and s.onset and s.onset[-1] in _GLIDES:
            s.glide = s.onset.pop()
        s.src = "".join(s.onset) + (s.glide or "") + "".join(s.nucleus) + "".join(s.coda)
    return sylls


def english_syllables(text: str) -> list[Syl]:
    """영어 글을 eSpeak en-us 음소로 바꿔 음절 목록으로 만든다.

    숫자는 먼저 대사에서 말하는 꼴의 낱말로 바꾼다(textnorm.speak_english_numbers). eSpeak 에 숫자를 그대로 주면
    911 을 nine hundred eleven, 10:00 을 ten zero zero 로 읽어 음절 수와 원문 프레임 열이 부푼다.
    """
    out: list[Syl] = []
    for phones in pronunciation(speak_english_numbers(text), EN_VOICE):
        out.extend(_en_word_syllables(phones))
    return out


def english_syllable_count(text: str) -> int:
    return len(english_syllables(text))


# ---------------------------------------------------------------------------- 한국어
# 초성 19자. ㅇ 은 첫소리 없음.
_CHO = ("k", "k͈", "n", "t", "t͈", "ɾ", "m", "p", "p͈", "s", "s͈", "", "tɕ", "t͈ɕ", "tɕʰ", "kʰ", "tʰ", "pʰ", "h")
_CHO_IEUNG = 11
_CHO_AFFRICATES = {12, 13, 14}           # ㅈ ㅉ ㅊ
# 중성 21자 -> (글라이드, 모음)
_JUNG: tuple[tuple[str | None, str], ...] = (
    (None, "a"), (None, "ɛ"), ("j", "a"), ("j", "ɛ"), (None, "ʌ"), (None, "e"), ("j", "ʌ"), ("j", "e"),
    (None, "o"), ("w", "a"), ("w", "ɛ"), ("w", "e"), ("j", "o"), (None, "u"), ("w", "ʌ"), ("w", "e"),
    ("w", "i"), ("j", "u"), (None, "ɯ"), ("ɰ", "i"), (None, "i"),
)
_JUNG_UI = 19
# 종성 27자(0 = 없음)는 받침 대표음 하나로 적는다.
_JONG = ("", "k", "k", "k", "n", "n", "n", "t", "l", "k", "m", "l", "l", "l", "p", "l",
         "m", "p", "p", "t", "t", "ŋ", "t", "t", "k", "t", "p", "t")

KO_VOWELS: frozenset[str] = frozenset(v for _, v in _JUNG)
# 한국어 모음 기호 -> 홑모음 자모(표·보고용)
KO_VOWEL_JAMO: dict[str, str] = {"a": "ㅏ", "ɛ": "ㅐ", "ʌ": "ㅓ", "e": "ㅔ", "o": "ㅗ", "u": "ㅜ", "ɯ": "ㅡ", "i": "ㅣ"}

_KO_WORD = re.compile(r"[가-힣A-Za-z]+")
_KO_PIECE = re.compile(r"[가-힣]|[A-Za-z]+")


def _hangul_syllable(ch: str, word_initial: bool) -> Syl:
    code = ord(ch) - 0xAC00
    cho, jung, jong = code // 588, (code % 588) // 28, code % 28
    glide, vowel = _JUNG[jung]
    if jung == _JUNG_UI and (cho != _CHO_IEUNG or not word_initial):
        glide = None                       # 희망 -> 히망, 회의 -> 회이. 낱말 첫 '의' 만 ɰ+i
    if cho in _CHO_AFFRICATES and glide == "j":
        glide = None                       # 져 -> 저, 쳐 -> 처
    onset = [_CHO[cho]] if cho != _CHO_IEUNG else []
    coda = [_JONG[jong]] if jong else []
    return Syl(onset=onset, glide=glide, nucleus=[vowel], coda=coda, src=ch)


def korean_syllables(text: str) -> list[Syl]:
    """한국어 글을 음절 목록으로 만든다.

    숫자는 먼저 한자어로 읽는다(20 -> 이십). 한국어 문장 속 영문 낱말은 latin_syllables 개수만큼
    모음 없는 음절로 둔다. 띄어쓰기·문장부호·그 밖의 기호는 버린다.
    """
    out: list[Syl] = []
    for word in _KO_WORD.findall(read_digits(text)):
        first = True
        for piece in _KO_PIECE.findall(word):
            if len(piece) == 1 and "가" <= piece <= "힣":
                out.append(_hangul_syllable(piece, first))
            else:
                out.extend(Syl(src=piece) for _ in range(latin_syllables(piece)))
            first = False
    return out


def korean_syllable_count(text: str) -> int:
    """textnorm.ko_syllable_count 와 같은 값이다."""
    return len(korean_syllables(text))
