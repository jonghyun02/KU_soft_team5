"""비교용 글: 대본 글과 받아쓰기 글을 견주는 규칙을 한 곳에 둔다(align·final·validate 가 같이 쓴다).

- ``key(text)``: same 판정 열쇠. NFC, 영문 소문자, 공백·문장부호·기호를 모두 뺀 글자열.
  들린 글(heard)의 열쇠가 대본 요소 text 의 열쇠와 같으면 same, 다르면 changed 다(명세 D4).
  글자가 아닌 것으로 치는 것: 한글 호환 낱자모(ㅋㅋ, ㅠㅠ, 줄표로 친 ㅡ, 채움 문자), 변형 선택자.
  전각 영문·숫자는 반각으로 접는다. 숫자에 붙은 부호 가운데 말로 읽거나 수를 가르는 것은 남긴다:
  소수점은 '점'(1.5 와 15 는 다른 글), 숫자 바로 뒤 % 는 그대로(100% 와 100 은 다른 글), 숫자 줄과 숫자 줄
  사이의 경계는 ``SEP``(2, 3일 과 23일 은 다른 글. 사이에 낀 것이 공백이든 쉼표든 붙임표든 경계 하나다).
  자릿수 쉼표(3,000)만 없는 것으로 친다.
- ``loc(text, other)``: 자리 찾기용 글자열. ``key`` 에서 아라비아 숫자 줄을 읽는 꼴로, 짧은 영문 약어를 한글
  읽기로 바꾼다. 읽는 꼴이 여럿이면(자릿수 읽기 '천오백', 한 자리씩 '일오공공', 고유어 '열두') 맞은편 글
  (``other``)에 있는 것을 고르고, 없으면 앞뒤 말을 보지 않는 기본 읽기다. 판정에는 쓰지 않는다.
- ``diffs(script_text, heard)``: 다른 구간을 어절 단위로 묶고 종류(filler, repeat, word, number, spelling)를 붙인다.
  종류는 틀림없을 때만 붙인다: 같은 수를 달리 적은 것만 number, 소리가 같거나 굳은 표기 흔들림만 spelling,
  그 자리에서 군말인 것만 filler, 남은 말을 되풀이하거나 뒤 낱말의 앞머리인 것만 repeat, 나머지는 word.
- 군말·받아쓰기 헛소리 꼴 판별.

순수 함수뿐이고 표준 라이브러리와 rapidfuzz 만 쓴다.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from rapidfuzz.distance import LCSseq, Levenshtein

# ------------------------------------------------------------------ 열쇠

_FULLWIDTH = {c: c - 0xFEE0 for r in (range(0xFF10, 0xFF1A), range(0xFF21, 0xFF3B), range(0xFF41, 0xFF5B)) for c in r}
POINT = "점"               # 숫자 사이 소수점을 열쇠에 남기는 글자
SEP = chr(0xB7)            # 숫자 줄과 숫자 줄 사이의 경계를 열쇠에 남기는 글자(가운뎃점. 문장부호라 다른 뜻으로는 열쇠에 못 들어온다)
PERCENT = "%"              # 숫자 바로 뒤의 % 를 열쇠에 남기는 글자
_MARKS = POINT + SEP + PERCENT


def _keep(c: str) -> bool:
    """글자(문자·숫자)면 참. 공백·문장부호·기호는 거짓. 결합 부호(M)는 글자에 딸린 것이라 남긴다.
    한글 호환 낱자모(U+3131~U+318E: ㅋ, ㅠ, 줄표로 친 ㅡ, 채움 문자 U+3164, 아래아 가운뎃점 U+318D)와
    변형 선택자(U+FE00~U+FE0F)는 말이 아니라서 뺀다."""
    o = ord(c)
    if c.isalnum():
        return not 0x3131 <= o <= 0x318E
    return o > 0x2FF and unicodedata.category(c)[0] == "M" and not 0xFE00 <= o <= 0xFE0F


def _norm(text: str) -> str:
    """NFC, 전각 영문·숫자는 반각, 영문은 소문자."""
    return unicodedata.normalize("NFC", text).translate(_FULLWIDTH).lower()


def _thousands(t: str, gap: int, i: int) -> bool:
    """t[gap] 이 자릿수 쉼표인가: 숫자와 숫자 사이에 쉼표 하나뿐이고 그 뒤로 숫자가 꼭 세 개다(3,000 / 1,234,567)."""
    group = t[i:i + 3]
    return i == gap + 1 and t[gap] == "," and len(group) == 3 and group.isdecimal() and not t[i + 3:i + 4].isdecimal()


def _decimal_point(t: str, i: int) -> bool:
    """t[i] 가 소수점인가: 숫자와 숫자 사이의 마침표이고, 그 수의 앞뒤에 마침표로 이어진 숫자가 더 없다.
    (1.5 는 소수. 2008.2.14 나 016.9265.4885 의 마침표는 숫자 줄을 나누는 부호다)"""
    a = i
    while a > 0 and t[a - 1].isdecimal():
        a -= 1
    b = i + 1
    while b < len(t) and t[b].isdecimal():
        b += 1
    if a == i or b == i + 1:
        return False
    return not (a >= 2 and t[a - 1] == "." and t[a - 2].isdecimal()) and not (t[b:b + 1] == "." and t[b + 1:b + 2].isdecimal())


def _kept(t: str) -> list[str | None]:
    """정규화한 글(``_norm``)의 글자마다 열쇠에 들어갈 글, 없으면 None.

    숫자에 붙은 부호만 따로 다룬다. 숫자 사이의 소수점은 ``POINT``, 숫자 뒤의 % 는 ``PERCENT``.
    앞 숫자 줄과 떨어진(사이에 공백·문장부호가 낀) 숫자는 ``SEP`` 을 앞에 단 두 글자다. 자릿수 쉼표는 잇는다."""
    out: list[str | None] = []
    num = False                 # 바로 앞에 남긴 글자가 숫자인가
    gap = -1                    # 그 숫자 뒤로 처음 버린 글자의 자리(아직 없으면 -1)
    for i, c in enumerate(t):
        if _keep(c):
            if c.isdecimal():
                if num and gap >= 0 and not _thousands(t, gap, i):
                    c = SEP + c
                num = True
            else:
                num = False
            gap = -1
            out.append(c)
        elif c == "." and num and gap < 0 and _decimal_point(t, i):
            out.append(POINT)
            num = False
        elif c == "%" and num and (gap < 0 or t[gap:i].isspace()):
            out.append(PERCENT)
            num = False
            gap = -1
        else:
            out.append(None)
            if num and gap < 0:
                gap = i
    return out


def key(text: str | None) -> str:
    """same 판정 열쇠: NFC, 소문자, 글자만. ``key(a) == key(b)`` 이면 띄어쓰기·문장부호만 다른 글이다."""
    if not text:
        return ""
    return "".join(c for c in _kept(_norm(text)) if c)


def sayable(text: str | None) -> bool:
    """말할 글자가 하나라도 있으면 참(``.......`` 나 빈 글, ``ㅠ.ㅠ`` 는 거짓)."""
    return bool(key(text))


# ------------------------------------------------------------------ 숫자·영문 읽기

_DIGIT = "공일이삼사오육칠팔구"
_NATIVE_ADN = ("", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉")      # 세 명, 열두 시
_NATIVE_BASE = ("", "하나", "둘", "셋", "넷", "다섯", "여섯", "일곱", "여덟", "아홉")
_NATIVE_TENS = ("", "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔")
# 고유어 수사로 세는 말(뒤따르면 1~99 를 고유어로 먼저 읽는다). '개월' 은 한자어로 읽으니 따로 뺀다.
_NATIVE_COUNTERS = ("시", "살", "명", "개", "잔", "병", "마리", "사람", "군데", "가지", "켤레", "그루", "송이", "달")
# 양을 세는 말(뒤따르면 네 자리 수도 자릿수로 읽는다: 2004년, 5000원). 번호는 한 자리씩 읽는다(4885).
_QUANTITY_UNITS = ("년", "원", "만", "억", "천", "명", "개", "평", "호", "불", "달러", "미터", "킬로", "그램", "리터", "톤")
_PERCENT_READ = ("퍼센트", "프로", "퍼")
_LETTER = {"a": ("에이",), "b": ("비",), "c": ("씨", "시"), "d": ("디",), "e": ("이",), "f": ("에프",), "g": ("지",),
           "h": ("에이치",), "i": ("아이",), "j": ("제이",), "k": ("케이",), "l": ("엘",), "m": ("엠",), "n": ("엔",),
           "o": ("오",), "p": ("피",), "q": ("큐",), "r": ("알",), "s": ("에스",), "t": ("티",), "u": ("유",),
           "v": ("브이", "비"), "w": ("더블유",), "x": ("엑스",), "y": ("와이",), "z": ("제트", "지")}
LATIN_MAX = 5              # 이 글자 수 이하인 영문 줄만 약어로 보고 한글 읽기를 찾는다
READ_TAIL = 2              # 읽는 꼴을 고를 때 숫자 뒤에 붙여 보는 글자 수('12' + '시간')
READ_ALONE = 3             # 뒤에 붙여 볼 글자가 없을 때 맞은편 글에서 찾아 믿는 읽는 꼴의 최소 글자 수
READ_CAP = 256             # 한 구간의 읽는 꼴 후보 수 상한


def read_number(digits: str) -> str:
    """아라비아 숫자 줄을 읽는 꼴로(뒤따르는 말을 모를 때의 기본 규칙): 4자리 이상이거나 0 으로 시작하면 한 자리씩
    (4885 -> 사팔팔오, 016 -> 공일육), 아니면 자릿수 읽기(306 -> 삼백육, 15 -> 십오)."""
    if len(digits) >= 4 or digits[0] == "0":
        return "".join(_DIGIT[int(c)] for c in digits)
    return _below_10000(int(digits))


def _below_10000(n: int) -> str:
    out = ""
    for unit, name in ((1000, "천"), (100, "백"), (10, "십")):
        d = n // unit % 10
        if d:
            out += ("" if d == 1 else _DIGIT[d]) + name
    if n % 10:
        out += _DIGIT[n % 10]
    return out


def _sino(n: int) -> str:
    """자릿수 읽기(만·억 포함): 1500 -> 천오백, 2004 -> 이천사, 15000 -> 만오천."""
    if n == 0:
        return "영"
    out = ""
    for unit, name in ((10 ** 8, "억"), (10 ** 4, "만"), (1, "")):
        g = n // unit % 10 ** 4
        if g:
            out += ("" if g == 1 and name == "만" else _below_10000(g)) + name
    return out


def number_readings(digits: str, after: str = "") -> list[str]:
    """아라비아 숫자 줄(ASCII 숫자)을 읽는 꼴 후보. 가장 그럴듯한 것이 맨 앞이다.

    후보: 자릿수 읽기(천오백), 고유어 수사(1~99: 열두 / 열둘), 한 자리씩(일오공공, 0 은 공·영).
    after(숫자 뒤에 오는 글자들)로 순서를 정한다. 0 으로 시작하면 번호라 한 자리씩만. 고유어로 세는 말
    (시, 명, 살, 개 …)이 뒤따르면 고유어 먼저(12시간 -> 열두). 네 자리 수는 양을 세는 말(년, 원, 만 …)이
    뒤따르거나 00 으로 끝나면 자릿수 읽기 먼저(2004년 -> 이천사, 1500 -> 천오백), 아니면 번호로 보고 한 자리씩
    먼저(4885 -> 사팔팔오). 그 밖은 자릿수 읽기 먼저(306 -> 삼백육, 112 는 후보에 일일이 도 있다)."""
    spoken = "".join(_DIGIT[int(c)] for c in digits)
    spoken0 = spoken.replace("공", "영")
    numeric = digits[0] != "0" or digits == "0"
    n = int(digits) if numeric else -1
    sino = [_sino(n)] if 0 <= n < 10 ** 12 else []
    native: list[str] = []
    if 1 <= n <= 99:
        t, u = divmod(n, 10)
        native = [("스무" if t == 2 and not u else _NATIVE_TENS[t]) + _NATIVE_ADN[u], _NATIVE_TENS[t] + _NATIVE_BASE[u]]
    if not numeric:
        order = [spoken, spoken0]
    elif native and after.startswith(_NATIVE_COUNTERS) and not after.startswith("개월"):
        order = native + sino + [spoken, spoken0]
    elif len(digits) >= 4 and not (after.startswith(_QUANTITY_UNITS) or digits.endswith("00")):
        order = [spoken] + sino + [spoken0]
    else:
        order = sino + native + [spoken, spoken0]
    seen: list[str] = []
    for r in order:
        if r and r not in seen:
            seen.append(r)
    return seen


def _latin_readings(run: str) -> list[str]:
    out = [""]
    for ch in run:
        out = [x + y for x in out for y in _LETTER[ch]][:READ_CAP]
    return out


def _is_digit(x: str | None) -> bool:
    """``_kept`` 의 한 칸이 앞 숫자에 이어지는 숫자인가(SEP 이 붙은 숫자는 새 숫자 줄이라 아니다)."""
    return x is not None and len(x) == 1 and x.isdecimal()


def loc(text: str | None, other: str | None = None) -> str:
    """자리 찾기용 글자열: ``key`` 에서 숫자 줄을 읽는 꼴로 바꾼 것. 판정(same/changed)에는 쓰지 않는다.

    숫자 줄은 문장부호·공백에서 끊는다('2, 3일' 은 이·삼일, '016-9265-4885' 는 세 줄. 자릿수 쉼표 '3,000' 만 잇는다).
    읽는 꼴 후보(``number_readings``) 가운데 other(맞은편 글의 열쇠 글자열)에 뒤따르는 글자와 함께 있는 것을 고른다
    (12시간 -> 열두시간, 1500 -> 천오백, 112 -> 일일이, 21-15 -> 이십일다시십오). 뒤따르는 글자가 없으면 READ_ALONE 글자
    이상인 읽기만 믿는다. 맞은편에 없거나 other 를 주지 않으면 기본 읽기(``read_number``)다. 기본 읽기는 앞뒤 말을
    보지 않는다: 양쪽이 다 숫자로 적었으면 낱말이 어디서 끊겼든 같은 글자열이 된다.
    소수점은 '점', 소수점 뒤는 한 자리씩(1.25 -> 일점이오), % 는 퍼센트(맞은편에 '백프로' 가 있으면 프로).
    영문 약어(LATIN_MAX 글자 이하)는 other 에 그 영문이 없고 한글 읽기(TV -> 티비)가 있을 때만 바꾼다."""
    if not text:
        return ""
    t = _norm(text)
    kept = _kept(t)
    n = len(t)
    out: list[str] = []
    i = 0
    point = False               # 바로 앞에 남긴 것이 소수점인가
    while i < n:
        c = kept[i]
        if c is None:
            i += 1
        elif c[-1].isdecimal():
            if other and out and c[0] == SEP:
                g = i
                while g > 0 and kept[g - 1] is None:
                    g -= 1
                if "-" in t[g:i]:                            # 21-15 를 '이십일다시십오' 로 읽은 글
                    out.append(next((x for x in ("다시", "에") if out[-1] + x in other), ""))
            digits = str(unicodedata.decimal(c[-1]))
            j = i + 1
            while j < n:
                if _is_digit(kept[j]):
                    digits += str(unicodedata.decimal(kept[j]))
                    j += 1
                elif kept[j] is None and j + 1 < n and _is_digit(kept[j + 1]):
                    j += 1                                   # 자릿수 쉼표(그 밖의 틈은 뒤 숫자에 SEP 이 붙는다)
                else:
                    break
            after = ""
            k = j
            while k < n and len(after) < READ_TAIL:
                x = kept[k]
                if x is not None:
                    if x[-1].isdecimal() or "a" <= x <= "z" or x == PERCENT:
                        break
                    after += x
                k += 1
            cands = number_readings(digits, after)
            # 기본 읽기는 앞뒤 말을 보지 않는다: 대본 줄과 받아쓰기 낱말은 끊긴 자리가 달라도 같은 숫자를 같게 읽어야 한다.
            # 소수점 앞뒤만 낱말 안에서 정해지니 따로 읽는다(0.5 -> 영점오, 1.25 -> 일점이오).
            if point:
                pick = "".join(_DIGIT[int(x)] for x in digits)
            elif digits == "0" and j < n and kept[j] == POINT and t[j] == ".":
                pick = "영"
            else:
                pick = read_number(digits)
            if other:
                hit = next((r for r in cands if after and r + after in other), None)
                if hit is None:                              # 뒤따르는 글자 없이는 세 글자 이상인 읽기만 믿는다(영화 한 편 글에는 두 글자쯤은 우연히 있다)
                    hit = next((r for r in cands if len(r) >= READ_ALONE and r in other), None)
                pick = hit or pick
            out.append(pick)
            i = j
            point = False
        elif c == PERCENT:
            out.append(next((r for r in _PERCENT_READ if other and out and out[-1] + r in other), _PERCENT_READ[0]))
            i += 1
        elif "a" <= c <= "z":
            j = i
            while j < n and kept[j] is not None and "a" <= kept[j] <= "z":
                j += 1
            run = t[i:j]
            if other and 2 <= len(run) <= LATIN_MAX and run not in other:
                run = next((r for r in _latin_readings(run) if r in other), run)
            out.append(run)
            i = j
            point = False
        else:
            out.append(c)
            point = c == POINT and t[i] == "."
            i += 1
    return "".join(out)


def lcs(a: str, b: str) -> int:
    """순서를 지키며 정확히 맞는 글자 수(최장 공통 부분열 길이)."""
    return int(LCSseq.similarity(a, b)) if a and b else 0


def similarity(a: str, b: str) -> float:
    """글자 닮음 0~1: 2 × 맞은 글자 수 ÷ (두 글 길이 합). 둘 다 비면 1."""
    if not a and not b:
        return 1.0
    return 2.0 * lcs(a, b) / (len(a) + len(b))


# ------------------------------------------------------------------ 자모·소리

_LEADS = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_VOWELS = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_TAILS = ("", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
          "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ")


def jamo(c: str) -> tuple[int, int, int] | None:
    """한글 음절을 (초성, 중성, 종성) 번호로. 한글 음절이 아니면 None."""
    k = ord(c) - 0xAC00
    return (k // 588, (k % 588) // 28, k % 28) if 0 <= k < 11172 else None


def jamo_string(text: str) -> str:
    """글자열을 자모 번호 열(문자열)로. 한글 음절이 아닌 글자는 그대로 둔다. 종성 없음은 적지 않는다."""
    out = []
    for c in text:
        j = jamo(c)
        if j is None:
            out.append(c)
        else:
            out.append(chr(0x1100 + j[0]) + chr(0x1161 + j[1]) + (chr(0x11A7 + j[2]) if j[2] else ""))
    return "".join(out)


def _syllable(lead: str, vowel: str, tail: str) -> str:
    return chr(0xAC00 + _LEADS.index(lead) * 588 + _VOWELS.index(vowel) * 28 + _TAILS.index(tail))


# 받침의 대표음
_CODA_REP = {c: rep for rep, group in (("ㄱ", "ㄱㄲㅋㄳㄺ"), ("ㄴ", "ㄴㄵㄶ"), ("ㄷ", "ㄷㅅㅆㅈㅊㅌㅎ"), ("ㄹ", "ㄹㄼㄽㄾㅀ"),
                                      ("ㅁ", "ㅁㄻ"), ("ㅂ", "ㅂㅍㅄㄿ"), ("ㅇ", "ㅇ")) for c in group}
# 겹받침: (남는 받침, 뒤 음절로 넘어가는 소리)
_CODA_SPLIT = {"ㄳ": ("ㄱ", "ㅅ"), "ㄵ": ("ㄴ", "ㅈ"), "ㄶ": ("ㄴ", "ㅎ"), "ㄺ": ("ㄹ", "ㄱ"), "ㄻ": ("ㄹ", "ㅁ"), "ㄼ": ("ㄹ", "ㅂ"),
               "ㄽ": ("ㄹ", "ㅅ"), "ㄾ": ("ㄹ", "ㅌ"), "ㄿ": ("ㄹ", "ㅍ"), "ㅀ": ("ㄹ", "ㅎ"), "ㅄ": ("ㅂ", "ㅅ")}
_TENSE = {"ㄱ": "ㄲ", "ㄷ": "ㄸ", "ㅂ": "ㅃ", "ㅅ": "ㅆ", "ㅈ": "ㅉ"}
_LAX = {v: k for k, v in _TENSE.items()}
_ASPIRATE = {"ㄱ": "ㅋ", "ㄷ": "ㅌ", "ㅂ": "ㅍ", "ㅈ": "ㅊ"}
_NASAL = {"ㄱ": "ㅇ", "ㄷ": "ㄴ", "ㅂ": "ㅁ"}
_VOWEL_SAME = {"ㅔ": "ㅐ", "ㅒ": "ㅐ", "ㅖ": "ㅐ", "ㅚ": "ㅙ", "ㅞ": "ㅙ"}        # 요즘 말에서 가르지 않는 모음(네/내, 얘/애, 되/돼, 웬/왠)
_H_CODA = {"ㅎ": "", "ㄶ": "ㄴ", "ㅀ": "ㄹ"}

# 같은 말의 굳은 다른 표기(낱말 머리에서만. 글자 수가 같다): (접는 꼴, 다른 꼴들)
_WORD_VARIANTS = (("씨발", ("시발", "씨팔", "시팔")), ("인마", ("임마",)), ("조금", ("쪼금", "쪼끔", "조끔")), ("좀", ("쫌",)))
_GEO_TENSE = {"꺼": "거", "께": "게", "껄": "걸", "껀": "건"}       # ㄹ 받침 뒤 '거·게·걸·건' 의 된소리 표기(할 꺼야, 갈께, 할껄)
_GEO_ENDINGS = frozenset(("", "야", "예요", "에요", "다", "냐", "니", "지", "죠", "요", "라고", "라구", "잖아", "거든", "니까",
                          "는", "를", "도", "만", "랑", "면"))          # '꺼' 뒤에 붙어 '거' 의 꼴이 되는 말(내 꺼야, 누구 꺼예요)
# 한자어에 없는 받침(용언 줄기·고유어에만 있다): 이 받침 뒤의 '구' 는 연결 어미 '고' 의 입말 꼴이다(했구, 갖구, 싶구)
_STEM_CODA = frozenset("ㄷㅅㅆㅈㅊㅌㅍㅎㄲㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅄ")
_GO_BEFORE = frozenset("라다냐자려하리")           # ~라구, ~다구, ~냐구, ~자구, ~려구, 하구, 그리구
_YO_BEFORE = frozenset("어아에세해")               # ~어여, ~아여, ~에여, ~세여, 해여 ('뭐여', '거여' 는 사투리 '뭐야' 라 아니다)
_NI_AFTER = frozenset("가들")                      # 니가 / 네가, 니들 / 네들


def _fold_variants(chars: str, head: list[bool]) -> str:
    """같은 말의 흔한 다른 표기를 한 꼴로 접은 글(글자 수는 그대로). head[k]: k 번째 글자가 낱말의 첫 글자인가.

    뜻이나 소리가 갈릴 수 있는 자리는 접지 않는다(한때/한데, 인싸/인사, 꺼내/거네, 입구/입고, 뭐여/뭐요). 접는 것:
      낱말 머리의 굳은 꼴(``_WORD_VARIANTS``: 시발/씨발, 임마/인마, 쫌/좀), 것/거,
      ㄹ 받침 뒤 거·게·걸·건의 된소리 표기(할께/할게, 올 꺼야/올 거야),
      ㅏ 줄기 뒤의 어/아(알어/알아, 맞었어/맞았어), 같애/같아, 낱말 끝의 ~어여/~어요,
      어미 고의 입말 꼴 구(뭐라구, 하자구요, 갖구), ㄹ 받침 뒤 낱말 끝의 루/로(일루, 별루)."""
    n = len(chars)
    out = list(chars)
    for k in range(n):
        if not head[k]:
            continue
        end = k + 1
        while end < n and not head[end]:
            end += 1
        word = chars[k:end]
        for canon, variants in _WORD_VARIANTS:
            hit = next((v for v in variants if word.startswith(v)), None)
            if hit:
                out[k:k + len(hit)] = canon
                break
    for k in range(n):
        c = chars[k]
        j = jamo(c)
        if j is None or out[k] != c:
            continue
        lead, vowel, tail = _LEADS[j[0]], _VOWELS[j[1]], _TAILS[j[2]]
        hd = head[k]
        bj = jamo(chars[k - 1]) if k else None                             # 바로 앞 글자(앞 낱말의 끝 글자일 수도 있다)
        pj = None if hd else bj                                            # 같은 낱말의 앞 음절
        p_vowel, p_tail = (_VOWELS[pj[1]], _TAILS[pj[2]]) if pj else ("", "")
        ends = k + 1 == n or head[k + 1]                                   # 낱말의 마지막 음절인가
        ends_yo = not ends and chars[k + 1] == "요" and (k + 2 == n or head[k + 2])
        if c == "것":
            out[k] = "거"
        elif c in _GEO_TENSE and bj is not None and _TAILS[bj[2]] == "ㄹ":
            out[k] = _GEO_TENSE[c]
        elif hd:
            if c == "꺼":                                                  # 낱말이 의존명사 '거' 의 꼴일 때(꺼야, 꺼예요). '꺼내' 는 아니다
                end = k + 1
                while end < n and not head[end]:
                    end += 1
                if chars[k + 1:end] in _GEO_ENDINGS:
                    out[k] = "거"
        elif lead == "ㅇ" and vowel == "ㅓ" and tail in ("", "ㅆ") and p_vowel in ("ㅏ", "ㅑ") and p_tail not in ("", "ㅇ", "ㅆ"):
            out[k] = _syllable("ㅇ", "ㅏ", tail)                           # 알어, 맞었어. '갔어' 의 어는 바른 꼴이라 두고, ㅇ 받침 뒤는 용언이 아니다(상어)
        elif lead == "ㅇ" and vowel == "ㅐ" and tail in ("", "ㅆ") and chars[k - 1] == "같":
            out[k] = _syllable("ㅇ", "ㅏ", tail)
        elif c == "여" and ends and chars[k - 1] in _YO_BEFORE:
            out[k] = "요"
        elif c == "구" and (ends or ends_yo) and (p_tail in _STEM_CODA or chars[k - 1] in _GO_BEFORE or
                                                  (k >= 2 and not head[k - 1] and chars[k - 2:k] == "가지")):
            out[k] = "고"
        elif c == "루" and ends and p_tail == "ㄹ":
            out[k] = "로"
    return "".join(out)


def _fold_ni(chars: str, head: list[bool]) -> str:
    """'너의·네가' 를 '니' 로 적은 것을 '네' 로 바꾼 글: 낱말이 통째로 '니' 이거나 '니가', '니들' 로 시작할 때.
    소리 규칙과 따로 쓴다. 네/내 는 소리가 같아 표기 차이지만 니/내 는 다른 말이라, 니·네·내 를 한 꼴로 접으면 안 된다."""
    out = list(chars)
    for k, c in enumerate(chars):
        if c == "니" and head[k] and (k + 1 == len(chars) or head[k + 1] or chars[k + 1] in _NI_AFTER):
            out[k] = "네"
    return "".join(out)


def _sound(chars: str, head: list[bool]) -> list[tuple[str, str, str]]:
    """글자마다 (첫소리, 가운뎃소리, 끝소리): 표기 흔들림을 접고(``_fold_variants``) 소리 나는 대로 고친 것.
    한글 음절이 아닌 글자는 (글자, "", "")."""
    n = len(chars)
    syl: list[list[str] | None] = []
    for c in _fold_variants(chars, head):
        j = jamo(c)
        if j is None:
            syl.append(None)
            continue
        lead, vowel = _LEADS[j[0]], _VOWELS[j[1]]
        vowel = _VOWEL_SAME.get(vowel, vowel)
        if vowel == "ㅢ" and lead != "ㅇ":                                # 희 -> 히, 늬 -> 니
            vowel = "ㅣ"
        syl.append([lead, vowel, _TAILS[j[2]]])
    for k in range(n):
        cur = syl[k]
        if cur is None or not cur[2]:
            continue
        tail = cur[2]
        nxt = syl[k + 1] if k + 1 < n else None
        inside = nxt is not None and not head[k + 1]                       # 뒤 음절이 같은 낱말 안에 있다
        if inside and nxt[0] == "ㅇ":                                      # 받침이 뒤 음절 첫소리로: 몇일 = 며칠, 좋아 = 조아
            if tail == "ㅇ":
                pass
            elif tail in _H_CODA:
                cur[2] = ""
                nxt[0] = _H_CODA[tail] or "ㅇ"
            else:
                cur[2], move = _CODA_SPLIT.get(tail, ("", tail))
                if move in "ㄷㅌ" and nxt[1] in ("ㅣ", "ㅕ"):               # 같이 = 가치
                    move = "ㅈ" if move == "ㄷ" else "ㅊ"
                nxt[0] = move
        elif inside and nxt[0] == "ㅎ" and _CODA_REP[_CODA_SPLIT.get(tail, ("", tail))[1]] in _ASPIRATE:   # 어떡해 = 어떠캐
            rest, move = _CODA_SPLIT.get(tail, ("", tail))
            cur[2] = rest if rest in ("ㄴ", "ㄹ") else ""
            nxt[0] = "ㅊ" if move in ("ㅈ", "ㅊ") else _ASPIRATE[_CODA_REP[move]]
        elif inside and tail in _H_CODA and nxt[0] in "ㄱㄷㅈㅅㄴ":          # 어떻게 = 어떠케, 좋니 = 존니
            cur[2] = _H_CODA[tail]
            if nxt[0] == "ㅅ":
                nxt[0] = "ㅆ"
            elif nxt[0] == "ㄴ":
                cur[2] = cur[2] or "ㄴ"
            else:
                nxt[0] = _ASPIRATE[nxt[0]]
        else:
            cur[2] = _CODA_REP[tail]
        if inside and cur[2] in _NASAL:                                    # 터지는 받침 뒤: 된소리(택시 = 택씨), 콧소리(좆나 = 존나)
            if nxt[0] in _TENSE:
                nxt[0] = _TENSE[nxt[0]]
            elif nxt[0] in ("ㄴ", "ㅁ"):
                cur[2] = _NASAL[cur[2]]
            if cur[2] == "ㄷ" and nxt[0] in _LAX:                          # 된소리 앞의 ㄷ 받침은 들리지 않는다(있다가 = 이따가)
                cur[2] = ""
    return [(c, "", "") if x is None else ("" if x[0] == "ㅇ" else x[0], x[1], x[2]) for c, x in zip(chars, syl)]


def _flat(units: list[tuple[str, str, str]]) -> str:
    return "".join(a + b + c for a, b, c in units)


def spelling_like(a: str, b: str, first: list[bool] | None = None) -> bool:
    """같은 말을 달리 적은 것인가: 소리가 같거나(되요/돼요, 안/않, 몇일/며칠, 괜찮아/괜찬아, 어떡해/어떻게, 네가/내가)
    굳은 표기 흔들림(할게/할께, 거야/꺼야, 알어/알아, 뭐라고/뭐라구, 니가/네가, 임마/인마)만 다르다.

    뜻이 갈리는 차이(너/나, 물/불, 간다/한다, 경찰을/경찰은)와 받아쓰기의 비슷한 소리 오류(괜찮니/괜창니,
    초소/초수)는 아니다. first[k]: k 번째 글자가 낱말의 첫 글자인가(두 글의 길이가 같을 때만 쓴다. 주지 않으면
    글 전체가 한 낱말)."""
    if not a or not b or a == b:
        return False

    def heads(x: str) -> list[bool]:
        return list(first) if first is not None and len(first) == len(x) else [k == 0 for k in range(len(x))]

    ha, hb = heads(a), heads(b)
    return _flat(_sound(a, ha)) == _flat(_sound(b, hb)) or _fold_ni(a, ha) == _fold_ni(b, hb)


# ------------------------------------------------------------------ 군말·더듬기

# 군말 낱말(늘인 꼴 포함): 낱말 하나가 통째로 이것일 때만 군말이다('뭐야', '저자' 는 아니다).
# 뜻 있는 낱말과 꼴이 같은 것은 자리와 부호를 본다(차이 조각의 종류를 정할 때만).
FILLERS_SURE = ("아이고", "아이구", "아이씨", "에이씨", "아씨", "에이", "아유", "어유", "에휴", "아휴", "어휴",
                "어", "음", "아", "에", "응", "으", "흠")
FILLERS_LEAD = ("자",)                        # 줄 맨 앞이거나 뒤에 쉼표·말줄임표가 붙었을 때만('빨리 자' 는 아니다)
FILLERS_MARKED = ("저", "그", "뭐", "하", "허", "아이", "오")   # 뒤에 쉼표·말줄임표가 붙었을 때만('그 사람', '뭐 먹을래', '아이 어디 갔어', '오 형사')
FILLERS_CALL = ("야", "어이")                 # 꼴은 군말이지만 사람을 부르는 말이라 차이 조각에서는 군말로 치지 않는다(word)
FILLERS = FILLERS_SURE + FILLERS_LEAD + FILLERS_MARKED + FILLERS_CALL


def _filler_re(names: tuple[str, ...], extra: str = "") -> re.Pattern:
    return re.compile("(?:" + "|".join(f + "+" for f in names) + extra + ")")


_FILLER = _filler_re(FILLERS, "|으+음+")
_FILLER_SURE = _filler_re(FILLERS_SURE, "|으+음+")
_FILLER_LEAD = _filler_re(FILLERS_LEAD)
_FILLER_CALL = _filler_re(FILLERS_CALL)
FILLER_SHORT = 2          # '한두 음절 군말'의 음절 수
# 뒤 낱말과 자모 앞머리만 겹칠 때('이 인간아', '다 달라고') 더듬기로 보지 않는 한 음절 낱말
_ONSET_WORDS = frozenset("이그저내네제나너다더또안못좀잘새차소수")


def is_filler(word_key: str) -> bool:
    """낱말 하나(열쇠)가 군말 꼴이면 참. 늘인 꼴(어어, 음음음, 아이이, 으음)도 군말이다. 꼴만 본다:
    '그', '저', '뭐', '야' 처럼 뜻 있는 낱말과 꼴이 같은 것도 참이니, 자리와 부호는 쓰는 쪽이 본다."""
    return bool(word_key) and _FILLER.fullmatch(word_key) is not None


def _unlengthen(text: str) -> str:
    """끝소리를 늘인 꼴을 줄인다(야아 -> 야, 네에 -> 네, 엄마아 -> 엄마): 받침 없는 음절 뒤에 그 모음만 되풀이된 음절."""
    base = {2: 0, 3: 1, 6: 4, 7: 5, 12: 8, 17: 13, 9: 0, 14: 4, 10: 1, 15: 5, 11: 5}     # ㅑ->ㅏ, ㅘ->ㅏ 처럼 끝나는 소리
    while len(text) >= 2:
        last, prev = jamo(text[-1]), jamo(text[-2])
        if last is None or prev is None or last[0] != 11 or last[2] or prev[2]:
            break
        if last[1] != prev[1] and last[1] != base.get(prev[1]):
            break
        text = text[:-1]
    return text


# ------------------------------------------------------------------ 받아쓰기 헛소리 꼴

NOISE_GREETINGS = ("감사합니다", "고맙습니다", "감사해요", "고마워요", "감사드립니다", "안녕하세요", "안녕히계세요",
                   "안녕히가세요", "수고하셨습니다", "수고하세요", "시청해주셔서감사합니다", "다음영상에서만나요")
NOISE_CREDIT = re.compile(r"(?:한글|한국어|우리말)\s*자막|자막\s*(?:제공|제작|번역|협찬|by|[:：])|"
                          r"구독\s*(?:과|하고|및)?\s*좋아요|좋아요\s*(?:와|및)?\s*구독|구독\s*(?:부탁|눌러)|"
                          r"시청\s*(?:해\s*)?주셔서|알림\s*설정", re.I)
NOISE_REPEAT = 4          # 같은 낱말이 이만큼 이어지면 모델의 되풀이로 본다('엄마! 엄마!' 두세 번은 부르는 말)


def noise_shape(tokens: list[str]) -> str | None:
    """받아쓰기 모델이 말 없는 곳에 적는 꼴이면 그 이름: 'credit'(자막 제작·구독 문구), 'greeting'(인사말만,
    또는 같은 인사말 되풀이), 'repeat'(한 낱말이 NOISE_REPEAT 번 이상). 아니면 None. 꼴만 보며 지우지 않는다.
    인사말과 되풀이는 진짜 대사일 수도 있으니 쓰는 쪽이 낱말 확률과 함께 봐야 한다."""
    keys = [k for k in (key(t) for t in tokens) if k]
    if not keys:
        return None
    if NOISE_CREDIT.search(" ".join(tokens)):
        return "credit"
    joined = "".join(keys)
    if any(len(joined) % len(g) == 0 and joined == g * (len(joined) // len(g)) for g in NOISE_GREETINGS):
        return "greeting"
    if len(keys) >= NOISE_REPEAT and len(set(keys)) == 1:
        return "repeat"
    return None


# ------------------------------------------------------------------ 차이 조각

_CLOSERS = "\"'”’」』)]"
_END_MARKS = ".?!。？！"
_SOFT_PAUSE = re.compile(r"[,，、…~\-—–]|\.\.")       # 말이 이어지는 쉼: 쉼표, 말줄임표, 줄표(문장 끝 . ? ! 는 아니다)
_CALL = re.compile(r"[!?！？]")
_KIND_RANK = {"word": 0, "number": 1, "spelling": 2, "repeat": 3, "filler": 4}


def ends_sentence(word: str) -> bool:
    """낱말이 문장 끝 부호(. ? !)로 끝나면 참(뒤따르는 닫는 따옴표·괄호는 넘긴다).
    말줄임표(…, ..)로만 끝나면 말이 이어지는 꼴이라 거짓이다('그게...' 뒤에는 같은 사람의 말이 온다)."""
    w = word.strip().rstrip(_CLOSERS)
    run = w[len(w.rstrip(_END_MARKS + "…")):]
    if not run:
        return False
    return "?" in run or "!" in run or "？" in run or "！" in run or run in (".", "。")


@dataclass
class _Side:
    """한쪽 글(대본 또는 들린 글)을 견주기 좋게 푼 것."""
    words: list[str] = field(default_factory=list)       # 어절(공백으로 나눈 낱말, 보여 줄 글 그대로)
    chars: str = ""                                      # 열쇠 글자열
    w_of: list[int] = field(default_factory=list)        # 글자 -> 어절 번호
    t_of: list[int] = field(default_factory=list)        # 글자 -> 잔 조각 번호
    sep: list[str] = field(default_factory=list)         # 글자 -> 바로 앞 글자와의 사이에 낀 글(붙어 있으면 "")
    toks: list[str] = field(default_factory=list)        # 잔 조각 열쇠(공백이나 문장부호로 끊긴 글자 덩어리)
    tails: list[str] = field(default_factory=list)       # 잔 조각 바로 뒤에 붙은 문장부호
    starts: list[int] = field(default_factory=list)      # 잔 조각 -> 첫 글자 자리(끝에 글자 수 하나 더)
    head: list[bool] = field(default_factory=list)       # 글자 -> 잔 조각의 첫 글자인가


def _side(text: str) -> _Side:
    """잔 조각은 공백이나 문장부호로 끊긴 글자 덩어리다(띄어쓰기 없는 대본의 '아,상무님' 도 '아' / '상무님').
    글자열은 ``key(text)`` 와 같다(숫자 줄 경계 SEP 은 뒤 숫자가 든 조각의 첫 글자다)."""
    raw = unicodedata.normalize("NFC", text or "")
    s = _Side(words=raw.split())
    t = _norm(raw)
    chars: list[str] = []
    between = ""                            # 앞 글자 뒤로 지나온 공백·문장부호
    wi = -1
    spaced = True                           # 지금 공백 위에 있다
    cur = None                              # 지금 잇고 있는 잔 조각
    last = None                             # 뒤 부호를 받을 잔 조각(이 어절 안에서 바로 앞의 것)
    for c, k in zip(t, _kept(t)):
        if c.isspace():
            if not spaced and chars:
                between += " "
            spaced, cur, last = True, None, None
            continue
        if spaced:
            wi += 1
            spaced = False
        if k is None:
            if last is not None:
                s.tails[last] += c
            between += c
            cur = None
            continue
        if cur is None:
            cur = len(s.toks)
            s.toks.append("")
            s.tails.append("")
            s.starts.append(len(chars))
        last = cur
        s.toks[cur] += k
        for n, ch in enumerate(k):
            chars.append(ch)
            s.w_of.append(wi)
            s.t_of.append(cur)
            s.sep.append("" if n else between)
        between = ""
    s.chars = "".join(chars)
    s.starts.append(len(chars))
    s.head = [k == 0 or s.t_of[k] != s.t_of[k - 1] for k in range(len(chars))]
    return s


def _whole_tokens(side: _Side, a: int, b: int) -> tuple[int, int] | None:
    """글자 구간 [a, b) 가 잔 조각 몇 개를 통째로 덮으면 그 조각 번호 범위 (처음, 끝 다음), 아니면 None."""
    if a >= b:
        return None
    t_of = side.t_of
    if (a > 0 and t_of[a - 1] == t_of[a]) or (b < len(t_of) and t_of[b - 1] == t_of[b]):
        return None
    return t_of[a], t_of[b - 1] + 1


def _slide(side: _Side, a: int, b: int, lo: int, hi: int) -> tuple[int, int]:
    """한쪽에만 있는 글자 구간 [a, b) 를 같은 글자가 이어지는 만큼 밀어(왼쪽으로 lo 까지, 오른쪽으로 hi 까지)
    잔 조각 경계에 맞는 자리를 찾는다('어 어디' 에서 더해진 '어' 가 낱말 첫 글자가 아니라 앞 낱말이 되게).
    맞는 자리가 없으면 그대로."""
    if _whole_tokens(side, a, b) is not None:
        return a, b
    chars = side.chars
    best = None
    d = 1
    while a - d >= lo and chars[a - d] == chars[b - d]:            # 왼쪽으로
        if _whole_tokens(side, a - d, b - d) is not None:
            best = (a - d, b - d)
            break
        d += 1
    d = 1
    while b + d <= hi and chars[a + d - 1] == chars[b + d - 1]:    # 오른쪽으로
        if _whole_tokens(side, a + d, b + d) is not None:
            if best is None or d < a - best[0]:
                best = (a + d, b + d)
            break
        d += 1
    return best if best is not None else (a, b)


def _filler_at(side: _Side, k: int) -> bool:
    """잔 조각 k 가 그 자리에서 군말인가. 뜻 있는 낱말과 꼴이 같은 것은 뒤에 쉼표·말줄임표가 붙었을 때만이고,
    '자' 는 줄 맨 앞(뒤에 말이 이어지고 ! ? 가 붙지 않음)도 받는다. 부르는 말(야, 어이)은 군말이 아니다."""
    tok = side.toks[k]
    if _FILLER_SURE.fullmatch(tok):
        return True
    if not is_filler(tok) or _FILLER_CALL.fullmatch(tok):
        return False
    tail = side.tails[k]
    if _SOFT_PAUSE.search(tail):
        return True
    return k == 0 and len(side.toks) > 1 and _FILLER_LEAD.fullmatch(tok) is not None and not _CALL.search(tail)


def _piece_runs(side: _Side, a: int, b: int, heard_side: bool) -> list[tuple[int, int, str]]:
    """한쪽에만 있는 글자 구간 [a, b) 를 낱말마다 종류를 매겨, 같은 종류가 이어진 구간 (처음, 끝, 종류)들로 나눈다.
    낱말 가운데서 끊긴 구간은 통째로 word 다.

    repeat: 여러 낱말이 통째로 바로 뒤나 앞의 같은 수의 낱말과 같거나(문 열어 문 열어), 옆 낱말과 통째로 같거나
            (진짜 진짜), 뒤 낱말의 앞머리다(자, 잠깐 / 살려 살려주세요). 한 음절 앞머리는 뒤에 쉼표·말줄임표가 붙었거나
            들린 글에 더해진 것일 때만 더듬기로 본다(대본의 '나 나갈게' 에서 '나' 가 빠진 것은 더듬기가 아니다).
            들린 글에서도 자모만 겹치는 한 음절 낱말(``_ONSET_WORDS``: 이 인간아)은 아니다.
    filler: 그 자리에서 군말이다(``_filler_at``). 틀림없는 군말(어, 아, 음)은 뒤 낱말의 앞머리와 겹쳐도 군말이다
            (아, 안양이요).
    word:   그 밖."""
    rng = _whole_tokens(side, a, b)
    if rng is None:
        return [(a, b, "word")]
    t0, t1 = rng
    toks = side.toks
    # 되풀이: 낱말 m 개가 통째로 바로 뒤나 앞의 낱말 m 개와 같다(진짜 진짜 / 일 년... 일 년). 견준 쪽은 조각 밖에 남은 말이거나
    # 이미 되풀이로 본 말이어야 한다(조각 안에서만 되풀이된 '야! 야!' 가 통째로 빠졌으면 되풀이가 아니라 빠진 말이다).
    again: set[int] = set()
    grew = True
    while grew:
        grew = False
        for m in range(t1 - t0, 0, -1):
            for p in range(t0, t1 - m + 1):
                span = range(p, p + m)
                if again.issuperset(span):
                    continue
                after, before = range(p + m, p + 2 * m), range(p - m, p)
                if ((toks[p + m:p + 2 * m] == toks[p:p + m] and all(q >= t1 or q in again for q in after)) or
                        (p >= m and toks[p - m:p] == toks[p:p + m] and all(q < t0 or q in again for q in before))):
                    again.update(span)
                    grew = True
    runs: list[tuple[int, int, str]] = []
    for k in range(t0, t1):
        tok = toks[k]
        nxt = toks[k + 1] if k + 1 < len(toks) else ""
        exact = bool(nxt) and nxt != tok and nxt.startswith(tok)
        onset = exact or (bool(nxt) and nxt != tok and jamo_string(nxt).startswith(jamo_string(tok)))
        if k in again:
            kind = "repeat"
        elif _FILLER_SURE.fullmatch(tok):
            kind = "filler"
        elif onset and _SOFT_PAUSE.search(side.tails[k]):
            kind = "repeat"
        elif _filler_at(side, k):
            kind = "filler"
        elif onset and (len(tok) >= 2 or (heard_side and (exact or tok not in _ONSET_WORDS))):
            kind = "repeat"
        else:
            kind = "word"
        if runs and runs[-1][2] == kind:
            runs[-1] = (runs[-1][0], side.starts[k + 1], kind)
        else:
            runs.append((side.starts[k], side.starts[k + 1], kind))
    return runs


def _numeric(text: str) -> bool:
    return any(c.isdecimal() or c in (SEP, PERCENT) for c in text)


def _readings(side: _Side, a: int, b: int, likely: bool = False) -> set[str]:
    """글자 구간 [a, b) 를 읽는 꼴 후보들. 숫자 줄마다 후보를 곱한다(``READ_CAP`` 개까지).
    likely: 숫자 줄마다 가장 그럴듯한 읽기 하나만 쓴다(양쪽이 다 숫자라 말한 꼴을 모를 때)."""
    outs = [""]
    chars, sep = side.chars, side.sep
    k = a
    while k < b:
        c = chars[k]
        if c == SEP:
            if "-" in sep[k]:                                       # 21-15 는 '이십일다시십오' 로도 읽는다
                outs = [o + x for o in outs for x in ("", "다시", "에")][:READ_CAP]
            k += 1
        elif c == PERCENT:
            outs = [o + x for o in outs for x in _PERCENT_READ][:READ_CAP]
            k += 1
        elif not c.isdecimal():
            outs = [o + c for o in outs]
            k += 1
        else:
            j = k + 1
            while j < b and chars[j].isdecimal():
                j += 1
            cands = number_readings("".join(str(unicodedata.decimal(x)) for x in chars[k:j]), chars[j:j + READ_TAIL])
            outs = [o + x for o in outs for x in (cands[:1] if likely else cands)][:READ_CAP]
            k = j
    if a < b < len(chars) and chars[b] == SEP and "-" in sep[b]:    # 붙임표 앞에서 끝난 구간: 잇는 말까지 읽은 꼴도(893-1 의 '팔구삼에')
        outs += [o + x for o in outs for x in ("다시", "에")]
    return set(outs)


def _is_mark(chars: str, p: int) -> bool:
    """chars[p] 가 숫자이거나 숫자에 붙은 부호(숫자 줄 경계, %, 숫자 사이 소수점)인가."""
    c = chars[p]
    if c == POINT:
        return 0 < p < len(chars) - 1 and chars[p - 1].isdecimal() and chars[p + 1].isdecimal()
    return c.isdecimal() or c in (SEP, PERCENT)


def _anchor_tokens(side: _Side) -> tuple[list[str], list[int]]:
    """닻으로 쓸 낱말 목록과 첫 글자 자리(끝에 글자 수 하나 더). 잔 조각을 숫자 줄과 그 밖으로 더 나눈 것이다
    ('12시간' 은 '12' / '시간': 맞은편의 '열두 시간' 과 '시간' 이 맞고 숫자만 따로 남는다)."""
    toks: list[str] = []
    starts: list[int] = []
    chars = side.chars
    for k in range(len(side.toks)):
        p, end = side.starts[k], side.starts[k + 1]
        while p < end:
            q = p + 1
            mark = _is_mark(chars, p)
            while q < end and _is_mark(chars, q) == mark:
                q += 1
            toks.append(chars[p:q])
            starts.append(p)
            p = q
    starts.append(len(chars))
    return toks, starts


def _number_run(side: _Side, a: int, b: int) -> tuple[int, int]:
    """글자 구간 [a, b) 를 양옆으로 이어진 숫자·숫자 부호(소수점, %, 숫자 줄 경계)까지 넓힌 구간."""
    chars = side.chars
    while a > 0 and (chars[a - 1].isdecimal() or (chars[a - 1] in _MARKS and a >= 2 and chars[a - 2].isdecimal())):
        a -= 1
    while b < len(chars) and (chars[b].isdecimal() or chars[b] in (SEP, PERCENT) or
                              (chars[b] == POINT and chars[b + 1:b + 2].isdecimal())):
        b += 1
    return a, b


def diffs(script_text: str, heard: str) -> list[dict]:
    """대본 글과 들린 글의 다른 구간 목록 ``[{"kind", "script", "heard"}]`` (final.json diffs 의 꼴).

    먼저 양쪽에서 통째로 같은 낱말(잔 조각)을 차례대로 짝짓고(낱말 단위 최장 공통 부분열. 숫자가 든 낱말은
    맞은편에 그 읽는 꼴인 낱말이 있으면 같은 낱말로 친다: 4885 / 사팔팔오), 그 사이 구간만 글자 단위로 견줘
    다른 구간을 묶는다. 보여 주는 글은 양쪽 어절(공백으로 나눈 낱말)로 넓히고, 같은 어절에 닿는 구간은 하나로
    합친다. 띄어쓰기·문장부호·대소문자만 다르면 빈 목록이다.

    kind:
      number   같은 수를 달리 적었다: 한쪽에 아라비아 숫자가 있고 읽는 꼴(자릿수 읽기, 한 자리씩, 고유어 수사)이
               맞은편 글과 같다(삼백육호 / 306호, 열두 시 / 12시, 백 프로 / 100%). 숫자 줄을 나눈 자리만 다른 것도
               (016-9265-4885 / 01692654885). 수가 실제로 다르거나 숫자가 낀 딴 말이면 word 다.
      filler   한쪽에만 있는 말이 통째 낱말이고 모두 그 자리에서 군말이다(어, 음, 쉼표 붙은 그·저·뭐).
               군말이 다른 군말로 바뀐 것도 filler.
      repeat   한쪽에만 있는 말이 옆 낱말의 되풀이거나 뒤 낱말의 앞머리다(더듬기). 끝소리를 늘인 것(야 / 야아)도.
      spelling 같은 말의 다른 표기: 소리가 같거나 굳은 표기 흔들림만 다르다(``spelling_like``).
      word     그 밖(낱말이 바뀜·빠짐·더해짐. 받아쓰기의 비슷한 소리 오류도 여기다).
    합쳐진 구간의 종류는 word > number > spelling > repeat > filler 순으로 가장 무거운 것이다.
    """
    s, h = _side(script_text), _side(heard)
    s_c, h_c = s.chars, h.chars
    if s_c == h_c:
        return []
    # 1. 통째로 같은 낱말을 닻으로 삼고, 그 사이 구간만 글자 단위 편집 연산. 이어진 다른 구간은 하나로.
    raw: list[list] = []                    # [i1, i2, j1, j2, 미리 정한 종류 또는 None]

    s_toks, s_at = _anchor_tokens(s)
    h_toks, h_at = _anchor_tokens(h)
    s_edge, h_edge = set(s_at), set(h_at)

    def between(ta0: int, ta1: int, tb0: int, tb1: int) -> None:
        i0, i1, j0, j1 = s_at[ta0], s_at[ta1], h_at[tb0], h_at[tb1]
        a, b = s_c[i0:i1], h_c[j0:j1]
        if a == b:
            return
        if ta1 - ta0 == 1 and tb1 - tb0 == 1 and _unlengthen(a) == _unlengthen(b):
            raw.append([i0, i1, j0, j1, "repeat"])
            return
        for op in Levenshtein.opcodes(a, b):
            if op.tag == "equal":
                continue
            i, i2, j, j2 = i0 + op.src_start, i0 + op.src_end, j0 + op.dest_start, j0 + op.dest_end
            last = raw[-1] if raw else None
            # 이어진 다른 구간은 하나로 묶되, 양쪽 다 낱말 경계인 자리에서는 끊는다(빠진 낱말과 바뀐 낱말을 따로 본다).
            # 숫자끼리는 끊지 않는다(둘 셋 / 2, 3 은 통째로 읽어야 같은 수인지 안다).
            if (last and last[4] is None and last[1] == i and last[3] == j and
                    (not (i in s_edge and j in h_edge) or
                     (_numeric(s_c[last[0]:i] + h_c[last[2]:j]) and _numeric(s_c[i:i2] + h_c[j:j2])))):
                last[1], last[3] = i2, j2
            else:
                raw.append([i, i2, j, j2, None])

    def read_as(side: _Side, toks: list[str], at: list[int], other: list[str]) -> list[str]:
        """닻 낱말 목록. 숫자 줄은 맞은편에 그 읽는 꼴인 낱말이 있으면 그 낱말로 바꾼다.
        한 글자 읽기(이, 두, 세, 오)는 딴 낱말과 겹치기 쉬워, 맞은편에서도 바로 뒤 낱말이 같을 때만 쓴다(3 명 / 세 명)."""
        have = set(other)
        pairs = set(zip(other, other[1:]))
        out = []
        for k, tok in enumerate(toks):
            if tok not in have and _numeric(tok):
                nxt = toks[k + 1] if k + 1 < len(toks) else None
                tok = min((r for r in _readings(side, at[k], at[k + 1]) & have if len(r) >= 2 or (r, nxt) in pairs),
                          default=tok)
            out.append(tok)
        return out

    pa = pb = 0
    for op in LCSseq.opcodes(read_as(s, s_toks, s_at, h_toks), read_as(h, h_toks, h_at, s_toks)):
        if op.tag != "equal":
            continue
        between(pa, op.src_start, pb, op.dest_start)
        for d in range(op.src_end - op.src_start):
            ta, tb = op.src_start + d, op.dest_start + d
            if s_toks[ta] != h_toks[tb]:                            # 읽는 꼴로 맞은 낱말
                raw.append([s_at[ta], s_at[ta + 1], h_at[tb], h_at[tb + 1], "number"])
        pa, pb = op.src_end, op.dest_end
    between(pa, len(s_toks), pb, len(h_toks))
    # 2. 한쪽에만 있는 구간은 낱말 경계에 맞게 민다(양옆 같은 구간 안에서만).
    for k, r in enumerate(raw):
        i1, i2, j1, j2, forced = r
        if forced:
            continue
        left = raw[k - 1] if k else None
        right = raw[k + 1] if k + 1 < len(raw) else None
        if i1 == i2 and j2 > j1:            # 들린 쪽에만
            lo = left[3] if left else 0
            hi = right[2] if right else len(h_c)
            a, b = _slide(h, j1, j2, lo, hi)
            r[0] = r[1] = i1 + (a - j1)
            r[2], r[3] = a, b
        elif j1 == j2 and i2 > i1:          # 대본 쪽에만
            lo = left[1] if left else 0
            hi = right[0] if right else len(s_c)
            a, b = _slide(s, i1, i2, lo, hi)
            r[2] = r[3] = j1 + (a - i1)
            r[0], r[1] = a, b
    # 3. 같은 글자의 짝(넓히기에 쓴다)
    s2h: list[int | None] = [None] * len(s_c)
    h2s: list[int | None] = [None] * len(h_c)
    pi = pj = 0
    for i1, i2, j1, j2, _ in raw + [[len(s_c), len(s_c), len(h_c), len(h_c), None]]:
        for d in range(min(i1 - pi, j1 - pj)):
            s2h[pi + d], h2s[pj + d] = pj + d, pi + d
        pi, pj = i2, j2
    # 4. 구간마다 종류와 닿는 어절
    sound_s = sound_h = ni_s = ni_h = None  # 소리 나는 꼴과 니 -> 네 로 바꾼 글(표기 차이를 볼 때만 만든다)

    def same_sound(i1: int, i2: int, j1: int, j2: int) -> bool:
        """두 구간의 소리가 같고 그 구간 때문에 앞 글자의 끝소리나 뒤 글자의 첫소리가 달라지지도 않는가.
        또는 니가 / 네가 처럼 '니' 로 적은 것만 다른가."""
        nonlocal sound_s, sound_h, ni_s, ni_h
        if sound_s is None:
            sound_s, sound_h = _sound(s_c, s.head), _sound(h_c, h.head)
            ni_s, ni_h = _fold_ni(s_c, s.head), _fold_ni(h_c, h.head)
        if ni_s[i1:i2] == ni_h[j1:j2]:
            return True
        if _flat(sound_s[i1:i2]) != _flat(sound_h[j1:j2]):
            return False
        before = (sound_s[i1 - 1][2] if i1 else "") == (sound_h[j1 - 1][2] if j1 else "")
        after = (sound_s[i2][0] if i2 < len(s_c) else "") == (sound_h[j2][0] if j2 < len(h_c) else "")
        return before and after

    rows: list[dict] = []

    def row(kind: str, i1: int, i2: int, j1: int, j2: int) -> None:
        ss = set(s.w_of[i1:i2])
        hs = set(h.w_of[j1:j2])
        if not ss and 0 < i1 < len(s_c) and s.w_of[i1 - 1] == s.w_of[i1]:
            ss = {s.w_of[i1]}               # 대본 어절 안에 글자가 더해짐
        if not hs and 0 < j1 < len(h_c) and h.w_of[j1 - 1] == h.w_of[j1]:
            hs = {h.w_of[j1]}
        rows.append({"kind": kind, "s": ss, "h": hs})

    for i1, i2, j1, j2, forced in raw:
        a, b = s_c[i1:i2], h_c[j1:j2]
        if forced:
            row(forced, i1, i2, j1, j2)
        elif a and b:
            if _numeric(a + b):
                kind = "number" if _readings(s, i1, i2) & _readings(h, j1, j2) else "word"
            elif (all(_filler_at(s, k) for k in range(s.t_of[i1], s.t_of[i2 - 1] + 1)) and
                  all(_filler_at(h, k) for k in range(h.t_of[j1], h.t_of[j2 - 1] + 1))):
                kind = "filler"             # 군말이 다른 군말로(닿는 낱말이 양쪽 다 군말)
            elif same_sound(i1, i2, j1, j2):
                kind = "spelling"
            else:
                kind = "word"
            row(kind, i1, i2, j1, j2)
        elif all(c in _MARKS for c in a + b) and any(c.isdecimal() for c in (_span(s, i1, i2) if a else _span(h, j1, j2))):
            # 숫자 부호만 다르다(소수점, %, 숫자 줄을 나눈 자리): 양쪽 숫자 줄을 통째로, 가장 그럴듯하게 읽어 같으면 같은 수다
            # (016-9265-4885 / 01692654885 는 둘 다 한 자리씩. 2, 3일 / 23일 은 이삼 / 이십삼 이라 다른 수)
            row("number" if _readings(s, *_number_run(s, i1, i2), likely=True) &
                _readings(h, *_number_run(h, j1, j2), likely=True) else "word", i1, i2, j1, j2)
        elif not a:
            if _hyphen_read(b, s, i1):
                row("number", i1, i2, j1, j2)
            else:                           # 낱말마다 종류를 매겨 같은 종류끼리 따로 낸다(아 진짜 -> 군말 '아' / 낱말 '진짜')
                for lo, hi, kind in _piece_runs(h, j1, j2, True):
                    row(kind, i1, i2, lo, hi)
        elif _hyphen_read(a, h, j1):
            row("number", i1, i2, j1, j2)
        else:
            for lo, hi, kind in _piece_runs(s, i1, i2, False):
                row(kind, lo, hi, j1, j2)
    # 5. 양쪽 온 어절로 넓히고, 같은 어절에 닿으면 합친다.
    s_chars_of: dict[int, list[int]] = {}
    for k, w in enumerate(s.w_of):
        s_chars_of.setdefault(w, []).append(k)
    h_chars_of: dict[int, list[int]] = {}
    for k, w in enumerate(h.w_of):
        h_chars_of.setdefault(w, []).append(k)

    def widen(ss: set, hs: set) -> tuple[set, set]:
        while True:
            hs2 = hs | {h.w_of[s2h[k]] for w in ss for k in s_chars_of.get(w, ()) if s2h[k] is not None}
            ss2 = ss | {s.w_of[h2s[k]] for w in hs2 for k in h_chars_of.get(w, ()) if h2s[k] is not None}
            if ss2 == ss and hs2 == hs:
                return ss, hs
            ss, hs = ss2, hs2

    out: list[dict] = []
    for r in rows:
        ss, hs = widen(r["s"], r["h"])
        if out and (ss & out[-1]["s"] or hs & out[-1]["h"]):
            last = out[-1]
            last["kind"] = min(last["kind"], r["kind"], key=_KIND_RANK.get)
            last["s"] |= ss
            last["h"] |= hs
        else:
            out.append({"kind": r["kind"], "s": ss, "h": hs})
    return [{"kind": r["kind"],
             "script": " ".join(s.words[i] for i in sorted(r["s"])),
             "heard": " ".join(h.words[j] for j in sorted(r["h"]))} for r in out]


def _span(side: _Side, a: int, b: int) -> str:
    """글자 구간 [a, b) 를 이어진 숫자·숫자 부호까지 넓힌 글."""
    a, b = _number_run(side, a, b)
    return side.chars[a:b]


def _hyphen_read(piece: str, other: _Side, at: int) -> bool:
    """한쪽에만 있는 말이 숫자 사이 붙임표를 읽은 말인가: 맞은편 그 자리가 붙임표로 나뉜 숫자 줄 경계다
    (이십일 다시 십오 / 21-15 의 '다시')."""
    return piece in ("다시", "에") and other.chars[at:at + 1] == SEP and "-" in other.sep[at]


def verdict(script_text: str, heard: str) -> tuple[str, list[dict]]:
    """(status, diffs): 열쇠가 같으면 ("same", []), 다르면 ("changed", 차이 조각 1개 이상)."""
    if key(script_text) == key(heard):
        return "same", []
    return "changed", diffs(script_text, heard)
