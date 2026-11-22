"""글 정리: 숫자 읽기, 한국어 음절 세기, 문장 나누기, 후보 다듬기, 영어 대사의 수를 말하는 꼴로 풀기.

sino_number, read_digits, latin_syllables 는 C:\\projects\\dub-rhythm-demo\\scripts\\common.py 의
같은 이름 함수를 옮겼다. 옮기면서 천 단위 쉼표(1,500)와 소수점(3.5)을 한 수로 읽게 더했고,
10000 을 '일만' 대신 말할 때처럼 '만' 으로 읽게 고쳤다. latin_syllables 는 Lizzie 처럼 끝이 ie 인 낱말의
음절을 하나 덜 세던 것을 고쳤다.

speak_english_numbers 는 영어 대사의 숫자를 대사에서 말하는 꼴의 영어 낱말로 바꾼다. eSpeak 는 911 을
nine hundred eleven, 10:00 을 ten zero zero, 555-1234 를 dash 를 넣어 읽고 1999 를 nineteen hundred ninety nine
으로 읽어서 영어 음절 수(길이 목표)가 부풀었다(2026-11-22 확인).
"""
from __future__ import annotations

import re

HANGUL = re.compile(r"[가-힣]")
LATIN_WORD = re.compile(r"[A-Za-z]+")

_DIG = "영일이삼사오육칠팔구"
# 천 단위 쉼표가 들어간 수(1,500 / 12,000,000). 쉼표를 빼고 한 수로 읽는다.
_GROUPED = re.compile(r"(?<![\d,])\d{1,3}(?:,\d{3})+(?![\d,])")
# 소수(3.5 -> 삼 점 오). 소수 부분은 한 자리씩 읽는다.
_DECIMAL = re.compile(r"(\d+)\.(\d+)")
_DIGITS = re.compile(r"\d+")

# 문장 끝 부호. 뒤에 붙은 닫는 따옴표·괄호도 그 문장에 넣는다.
_END = ".?!…~"
_CLOSERS = "\"'”’)]」』»"
_SENT = re.compile(
    rf"[^{re.escape(_END)}\n]*[{re.escape(_END)}]+[{re.escape(_CLOSERS)}]*|[^{re.escape(_END)}\n]+"
)
_DOT_IN_NUMBER = "\uE000"   # 3.5 같은 소수점은 문장 끝이 아니므로 잠시 바꿔 둔다

# 후보 정리용
_LEAD_NUMBER = re.compile(r"^\s*(?:\(\d{1,3}\)|\d{1,3}\s*[.)](?!\d))\s*")
_LEAD_BULLET = re.compile(r"^\s*[-–—•·*]+\s*")
_PAREN = re.compile(r"\s*[(\[（【][^()\[\]（）【】]*[)\]）】]")
_QUOTE_PAIRS = (('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"), ("「", "」"), ("『", "』"), ("«", "»"))
_SPACES = re.compile(r"\s+")
_NON_WORD = re.compile(r"[\W_]+")


def sino_number(n: int) -> str:
    """정수를 한자어 수사로 읽은 소리(20 -> 이십, 1940 -> 천구백사십)."""
    if n == 0:
        return "영"
    out = []
    for unit_val, unit in ((10**8, "억"), (10**4, "만")):
        if n >= unit_val:
            head = n // unit_val
            # 만 앞의 '일'은 읽지 않는다(10000 -> 만). 억은 일억으로 읽는다.
            out.append(("" if unit == "만" and head == 1 else sino_number(head)) + unit)
            n %= unit_val
    for unit_val, unit in ((1000, "천"), (100, "백"), (10, "십")):
        d = n // unit_val
        if d:
            out.append(("" if d == 1 else _DIG[d]) + unit)
        n %= unit_val
    if n:
        out.append(_DIG[n])
    return "".join(out)


def _read_decimal(m: re.Match) -> str:
    whole = sino_number(int(m.group(1)))
    frac = "".join(_DIG[int(d)] for d in m.group(2))
    return f"{whole}점{frac}"


def read_digits(text: str) -> str:
    """숫자를 읽는 소리 한글로 바꾼다(음절 세기·합성 입력용).

    20 -> 이십, 1,500 -> 천오백, 3.5 -> 삼점오.
    """
    t = _GROUPED.sub(lambda m: m.group().replace(",", ""), text)
    t = _DECIMAL.sub(_read_decimal, t)
    return _DIGITS.sub(lambda m: sino_number(int(m.group())), t)


def latin_syllables(word: str) -> int:
    """번역에 남은 영문 이름(Ruth, Hildy 등)의 음절 수를 모음 묶음으로 어림한다."""
    w = word.lower()
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    # 끝의 묵음 e(Bruce, Jane)만 뺀다. -le(table)와 끝 모음 묶음이 e 하나가 아닌 경우(Lizzie)는 센다.
    if n > 1 and w.endswith("e") and not w.endswith("le") and groups[-1] == "e":
        n -= 1
    return max(1, n)


def ko_syllable_count(text: str) -> int:
    """말할 때의 음절 수.

    한글은 글자 수(공식 Sandglass v3 와 같은 규칙), 숫자는 한자어 읽기(20 -> 이십 = 2음절),
    남은 영문 낱말은 모음 묶음으로 센다. 띄어쓰기·문장부호·그 밖의 기호는 세지 않는다.
    """
    t = read_digits(text)
    return len(HANGUL.findall(t)) + sum(latin_syllables(w) for w in LATIN_WORD.findall(t))


def split_sentences(text: str) -> list[str]:
    """. ? ! … ~ 와 줄바꿈으로 문장을 나눈다.

    끝 문장부호와 뒤따르는 닫는 따옴표는 그 문장에 남긴다. 글자·숫자가 없는 조각은 버린다.
    숫자 속 소수점(3.5)에서는 나누지 않는다.
    """
    protected = re.sub(r"(?<=\d)\.(?=\d)", _DOT_IN_NUMBER, text)
    out = []
    for m in _SENT.finditer(protected):
        s = m.group().strip().replace(_DOT_IN_NUMBER, ".")
        if s and re.search(r"\w", s):
            out.append(s)
    return out


def _strip_wrapping_quotes(text: str) -> str:
    changed = True
    while changed and len(text) >= 2:
        changed = False
        for left, right in _QUOTE_PAIRS:
            if text.startswith(left) and text.endswith(right):
                inner = text[len(left):len(text) - len(right)]
                # 안쪽에 같은 따옴표가 또 있으면 감싼 것이 아니다("a" 와 "b").
                if left not in inner and right not in inner:
                    text = inner.strip()
                    changed = True
                break
    return text


def clean_candidate(text: str) -> str:
    """LLM 후보 한 줄을 다듬는다.

    앞쪽 번호("1.", "1)", "(1)")·글머리("-", "•")·감싼 따옴표·괄호 설명을 지우고 공백을 하나로 줄인다.
    """
    t = _SPACES.sub(" ", text).strip()
    prev = None
    while prev != t:                      # "- 1. ..." 처럼 겹친 머리표도 다 지운다
        prev = t
        t = _LEAD_BULLET.sub("", _LEAD_NUMBER.sub("", t, count=1), count=1)
    prev = None
    while prev != t:                      # 괄호 안 괄호는 안쪽부터 지운다
        prev = t
        t = _PAREN.sub("", t)
    t = _strip_wrapping_quotes(t.strip())
    return _SPACES.sub(" ", t).strip()


def dedupe_key(text: str) -> str:
    """중복 후보를 가리는 열쇠. 공백·문장부호를 빼고 영문은 소문자로 맞춘다."""
    return _NON_WORD.sub("", text).casefold()


# ---------------------------------------------------------------- 영어 수를 말하는 꼴로

_EN_ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
            "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
_EN_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_EN_SCALES = ((10**9, "billion"), (10**6, "million"), (1000, "thousand"))
_EN_DECADES = {10: "tens", 20: "twenties", 30: "thirties", 40: "forties", 50: "fifties", 60: "sixties",
               70: "seventies", 80: "eighties", 90: "nineties"}
_EN_ORDINAL = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
               "nine": "ninth", "twelve": "twelfth"}
# 이 낱말 바로 뒤의 1000~2099 는 해(연도)로 읽는다(in 1999 -> in nineteen ninety-nine)
_YEAR_CUES = ("in", "since", "by", "from", "until", "till", "of", "year", "circa", "before", "after", "during",
              "january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
              "november", "december", "summer", "winter", "spring", "fall", "autumn")
# 이 낱말 뒤의 수는 전화번호라 한 자리씩 읽는다(call 911 -> call nine one one)
_DIAL_CUES = ("call", "dial", "phone", "extension", "ext")
# 이 낱말 뒤의 세·네 자리 수는 둘씩 끊어 읽는다(room 101 -> room one oh one, flight 714 -> flight seven fourteen)
_CODE_CUES = ("room", "flight", "gate", "platform", "track", "bus", "route", "channel", "apartment", "apt", "suite",
              "unit", "highway", "interstate", "exit", "agent", "number", "train", "car", "line")

_N = r"(?<![\w.,:/$#])"            # 수 앞: 낱말·소수점·쉼표·쌍점 등에 붙어 있지 않음
_E = r"(?![\w]|[.,:]\d)"           # 수 뒤: 낱말에 붙지 않고 소수점·천 단위 쉼표가 이어지지 않음(끝 마침표·쉼표는 됨)
_PHONE = re.compile(_N + r"(\d{3}-\d{4}|\d+(?:-\d+){2,})(?![\w-]|[.,]\d)")
_TIME = re.compile(_N + r"([01]?\d|2[0-3]):([0-5]\d)(?![\d:])")
_TWENTY_FOUR_SEVEN = re.compile(r"(?<![\d/])24\s*/\s*7(?![\d/])")
_DECADE = re.compile(r"(?<![\w'])'?((?:1\d|20)\d0|[1-9]0)'?s\b")
_ORDINAL_DIGITS = re.compile(_N + r"(\d+)(?:st|nd|rd|th)\b", re.I)
_DOLLARS = re.compile(r"\$\s?((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(\s+(?:hundred|thousand|million|billion))?" + _E)
_PERCENT = re.compile(r"(\d)\s?%")
_NUMBER_SIGN = re.compile(r"(?:#|\bNo\.)\s*(?=\d)")
_CLOCK_AMPM = re.compile(_N + r"(\d{1,2})\s*([ap])\.?\s?m\.?(?![A-Za-z])", re.I)
_DIAL = re.compile(r"\b(" + "|".join(_DIAL_CUES) + r")(\.?\s+)(\d{3,})" + _E, re.I)
_LEADING_ZERO = re.compile(_N + r"(0\d+)" + _E)
_CODE = re.compile(r"\b(" + "|".join(_CODE_CUES) + r")(\.?\s+)(\d{3,4})(?![\w-]|[.,:]\d)", re.I)
_YEAR = re.compile(r"\b(" + "|".join(_YEAR_CUES) + r")(\s+)(1\d{3}|20\d{2})" + _E, re.I)
_RANGE = re.compile(_N + r"(\d+)\s*[-–]\s*(\d+)(?![\w-]|[.,:]\d)")
_PAIR_YEAR = re.compile(_N + r"(1[1-9]\d\d)" + _E)
_CARDINAL = re.compile(_N + r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?!\w)")


def english_number_words(n: int) -> str:
    """정수를 영어 기수 낱말로(35 -> thirty-five, 2024 -> two thousand twenty-four)."""
    if n < 0:
        return "minus " + english_number_words(-n)
    if n < 20:
        return _EN_ONES[n]
    if n < 100:
        tens, unit = divmod(n, 10)
        return _EN_TENS[tens] + (f"-{_EN_ONES[unit]}" if unit else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return f"{_EN_ONES[hundreds]} hundred" + (f" {english_number_words(rest)}" if rest else "")
    value, name = next((v, s) for v, s in _EN_SCALES if n >= v)
    head, rest = divmod(n, value)
    return f"{english_number_words(head)} {name}" + (f" {english_number_words(rest)}" if rest else "")


def _pair_words(n: int) -> str:
    """세·네 자리 수를 앞뒤 둘로 끊어 읽는다(1999 -> nineteen ninety-nine, 101 -> one oh one, 1500 -> fifteen hundred)."""
    head, tail = divmod(n, 100)
    if tail == 0:
        return f"{english_number_words(head)} hundred"
    if tail < 10:
        return f"{english_number_words(head)} oh {_EN_ONES[tail]}"
    return f"{english_number_words(head)} {english_number_words(tail)}"


def _year_words(n: int) -> str:
    """해(연도) 읽기: 2000~2009 는 two thousand five, 나머지는 둘씩 끊어(nineteen ninety-nine, twenty twenty-four)."""
    return english_number_words(n) if 2000 <= n <= 2009 else _pair_words(n)


def _digit_words(digits: str) -> str:
    """한 자리씩 읽는다. 0 은 oh(전화번호·코드를 말할 때의 꼴)."""
    return " ".join("oh" if d == "0" else _EN_ONES[int(d)] for d in digits)


def _ordinal_words(n: int) -> str:
    head, last = re.match(r"(.*?)([a-z]+)$", english_number_words(n)).groups()
    if last in _EN_ORDINAL:
        return head + _EN_ORDINAL[last]
    return head + (last[:-1] + "ieth" if last.endswith("y") else last + "th")


def _decade_words(n: int) -> str:
    """1990 -> nineteen nineties, 2000 -> two thousands, 90 -> nineties."""
    if n < 100:
        return _EN_DECADES[n]
    head, tail = divmod(n, 100)
    if tail:
        return f"{english_number_words(head)} {_EN_DECADES[tail]}"
    if n % 1000 == 0:
        return f"{english_number_words(n // 1000)} thousands"
    return f"{english_number_words(head)} hundreds"


def _time_words(m: re.Match) -> str:
    hour, minute = int(m.group(1)), int(m.group(2))
    words = english_number_words(hour)
    if minute == 0:
        return f"{words} o'clock" if hour <= 12 else f"{words} hundred"
    if minute < 10:
        return f"{words} oh {_EN_ONES[minute]}"
    return f"{words} {english_number_words(minute)}"


def _cardinal_words(m: re.Match) -> str:
    words = english_number_words(int(m.group(1).replace(",", "")))
    if m.group(2):
        words += " point " + " ".join(_EN_ONES[int(d)] for d in m.group(2))
    return words


def _dollar_words(m: re.Match) -> str:
    """$20 -> 20 dollars, $3.50 -> 3 dollars 50, $1.5 million -> 1.5 million dollars(숫자는 뒤에서 낱말로 바꾼다)."""
    amount, scale = m.group(1), m.group(2) or ""
    whole, _, frac = amount.partition(".")
    if scale or len(frac) != 2:
        return f"{amount}{scale} " + ("dollar" if amount == "1" and not scale else "dollars")
    return f"{whole} " + ("dollar" if whole == "1" else "dollars") + (f" {int(frac)}" if int(frac) else "")


def _range_words(m: re.Match) -> str:
    def side(digits: str) -> str:
        n = int(digits)
        return _year_words(n) if len(digits) == 4 and 1000 <= n <= 2099 else english_number_words(n)
    return f"{side(m.group(1))} to {side(m.group(2))}"


def speak_english_numbers(text: str) -> str:
    """영어 대사의 숫자를 대사에서 말하는 꼴의 영어 낱말로 바꾼다(숫자가 없으면 그대로).

    - 시각: 10:00 -> ten o'clock, 7:30 -> seven thirty, 8:05 -> eight oh five(13시 이후 :00 은 hundred).
    - 전화번호(555-1234, 1-800-555-0199), call·dial 뒤의 수, 0 으로 시작하는 수(007): 한 자리씩, 0 은 oh, dash 없음.
    - room·flight·gate·platform 같은 낱말 뒤의 세·네 자리 수: 둘씩 끊어(one oh one, seven fourteen).
    - 해: in·since·by·of·달 이름 뒤의 1000~2099 와 쉼표 없는 1100~1999 는 둘씩 끊어(nineteen ninety-nine).
      2000~2099 는 그런 낱말 뒤에서만 해로 읽고(twenty twenty-four), 아니면 기수(two thousand twenty-four).
    - 1990s -> nineteen nineties, 24/7 -> twenty-four seven, 3rd -> third, $20 -> twenty dollars, 50% -> fifty percent,
      #1·No. 1 -> number one, 10-20 -> ten to twenty, 그 밖의 수(1,500·3.5) -> 기수(one thousand five hundred, three point five).
    """
    if not any(ch.isdigit() for ch in text):
        return text
    t = _NUMBER_SIGN.sub("number ", text)
    t = _PHONE.sub(lambda m: ", ".join(_digit_words(g) for g in m.group(1).split("-")), t)
    t = _TIME.sub(_time_words, t)
    t = _TWENTY_FOUR_SEVEN.sub("twenty-four seven", t)
    t = _DECADE.sub(lambda m: _decade_words(int(m.group(1))), t)
    t = _ORDINAL_DIGITS.sub(lambda m: _ordinal_words(int(m.group(1))), t)
    t = _DOLLARS.sub(_dollar_words, t)
    t = _PERCENT.sub(r"\1 percent", t)
    t = _CLOCK_AMPM.sub(lambda m: f"{english_number_words(int(m.group(1)))} {m.group(2).lower()}.m.", t)
    t = _DIAL.sub(lambda m: m.group(1) + m.group(2) + _digit_words(m.group(3)), t)
    t = _LEADING_ZERO.sub(lambda m: _digit_words(m.group(1)), t)
    t = _CODE.sub(lambda m: m.group(1) + m.group(2) + _pair_words(int(m.group(3))), t)
    t = _YEAR.sub(lambda m: m.group(1) + m.group(2) + _year_words(int(m.group(3))), t)
    t = _RANGE.sub(_range_words, t)
    t = _PAIR_YEAR.sub(lambda m: _pair_words(int(m.group(1))), t)
    return _CARDINAL.sub(_cardinal_words, t)
