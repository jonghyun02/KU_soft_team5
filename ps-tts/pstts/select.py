"""후보 거르기와 고르기.

- assess: 음절 수(길이, 논문 식 6), LaBSE(뜻, 식 7), 말투(논문에 없음)를 재서 통과 여부와 한국어 탈락 까닭을 적는다.
  한글이 없거나 영문이 더 많은 후보도 뜻 조건에서 떨어뜨린다. 번역하지 않은 영어 원문은 LaBSE 1.0 을 받아서다.
  숫자 조건(missing_numbers, 논문에 없음)은 켰을 때만 본다(기본 끔).
- apply_line_register: 말투 미지정 줄은 모델 2 후보 다수의 말투(반말 또는 높임말)를 그 줄의 말투로 보고 다시 거른다.
- add_dtw: 원문과 후보의 모음 프레임 열 사이 DTW(식 10). 작을수록 입 모양이 비슷하다.
- selection_pool: 고를 후보 묶음. 통과 후보, 없으면 논문 ISO 의 두 조건(식 6, 7)을 채운 후보, 그것도 없으면
  LaBSE 가 가장 높은 후보(fallback, 3.1절 끝의 규칙).
- ps_comet: 묶음 가운데 alpha x 뒤집은 DTW + beta x COMET 이 가장 큰 후보(식 20).
  입이 안 보이는 줄(lip 끔)은 COMET 만으로 고른다.
"""
from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import NamedTuple

from . import register as register_mod
from .config import Config
from .dtw import ps_distance
from .phonetics import english_syllables, korean_syllables
from .records import Candidate, LineItem, Target
from .textnorm import HANGUL, LATIN_WORD, ko_syllable_count, latin_syllables, read_digits, sino_number
from .timing import frame_labels
from .vowels import VowelTable, load_or_build

_SAME = 1e-12        # DTW 최댓값과 최솟값이 이보다 가까우면 모두 같다고 본다

# 영어 수: 시각(7:30), 아라비아 숫자(1,500 / 3.5 포함), 기수 낱말
_EN_TOKEN = re.compile(r"(?P<time>(?<![\d:])\d{1,2}:\d{2}(?![\d:]))|(?P<num>\d+(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
                       r"|(?P<word>[A-Za-z]+)")
# 수가 들어 있어도 수로 옮기지 않는 굳은 말(24/7 -> 언제나, nine-to-five -> 평범한 직장)
_EN_IDIOMS = re.compile(r"(?<![\d/])24\s*/\s*7(?![\d/])|\b(?:nine|9)[\s-]to[\s-](?:five|5)\b", re.I)
# 이 낱말 뒤의 'nineteen eighty-four' 같은 두 낱말 수는 해(1984)로 본다
_YEAR_CUES = frozenset("in since by from until till of year circa before after during".split())
_EN_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
             "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
             "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_EN_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
            "ninety": 90}
_EN_BIG = {"hundred": 100, "thousand": 1000, "million": 10**6, "billion": 10**9}
_EN_WORDS = {**_EN_UNITS, **_EN_TENS, **_EN_BIG}
# 고유어 수 읽기(관형형·서수 꼴 포함). 11~99 는 십의 자리 낱말로 시작하는지만 본다(스물다섯, 열두).
_KO_NATIVE = {1: ("하나", "한", "첫"), 2: ("둘", "두"), 3: ("셋", "세", "석"), 4: ("넷", "네", "넉"),
              5: ("다섯", "닷"), 6: ("여섯", "엿"), 7: ("일곱",), 8: ("여덟",), 9: ("아홉",), 10: ("열",),
              20: ("스물", "스무"), 30: ("서른",), 40: ("마흔",), 50: ("쉰",), 60: ("예순",), 70: ("일흔",),
              80: ("여든",), 90: ("아흔",)}
_HANGUL_WORD = re.compile(r"[가-힣]+")


def korean_problem(text: str) -> str:
    """한국어 대사로 쓸 수 없으면 그 까닭, 쓸 수 있으면 빈 문자열."""
    t = read_digits(text)
    n_ko = len(HANGUL.findall(t))
    if n_ko == 0:
        return "한국어 아님: 한글이 없음"
    n_latin = sum(latin_syllables(w) for w in LATIN_WORD.findall(t))
    if n_latin > n_ko:
        return f"한국어 아님: 영문 {n_latin}음절이 한글 {n_ko}음절보다 많음"
    return ""


def _joins(phrase: list[str], word: str) -> bool:
    """word 가 앞 낱말들과 한 수를 이루는지(twenty five, two hundred five, three thousand)."""
    if word in _EN_BIG or phrase[-1] in _EN_BIG:
        return True
    return phrase[-1] in _EN_TENS and 1 <= _EN_UNITS.get(word, 0) <= 9


def _phrase_value(words: Sequence[str]) -> int:
    total = current = 0
    for w in words:
        if w in _EN_UNITS:
            current += _EN_UNITS[w]
        elif w in _EN_TENS:
            current += _EN_TENS[w]
        elif w == "hundred":
            current = max(current, 1) * 100
        else:
            total += max(current, 1) * _EN_BIG[w]
            current = 0
    return total + current


class EnNumber(NamedTuple):
    """영어 대사 속 수 하나."""
    shown: str                                # 원문 표기
    value: int                                # 값(소수는 정수 부분, '1.5 million' 은 곱한 값)
    extra: frozenset[str] = frozenset()       # 더 받아 주는 한국어 읽기(시각의 반, 한 자리씩 읽기 등)


def _digit_by_digit(digits: str, zero: str) -> str:
    return "".join(zero if d == "0" else sino_number(int(d)) for d in digits)


def english_numbers(text: str) -> list[EnNumber]:
    """영어 대사 속 수 목록.

    - 아라비아 숫자와 기수 낱말(two, twenty-five, two hundred and five)을 센다. 'one' 혼자는 'no one', 'the one' 처럼
      수가 아닐 때가 많아서, 'a hundred'·'a thousand'·'a million' 혼자는 '천 번은 말했어' 같은 과장일 때가 많아서
      세지 않는다. 서수(first), dozen, hundreds, 24/7, nine-to-five 도 세지 않는다.
    - 시각 7:30 은 7 과 30, 10:00 은 10 만 센다. 시 바로 뒤의 30(7:30, seven-thirty, ten thirty)은 '반' 도 받는다.
    - 1.5 million 처럼 숫자 뒤에 단위 낱말이 오면 한 수(1500000)로 보고 '일점오' 도 받는다.
    - 세 자리 넘는 숫자와 0 으로 시작하는 숫자는 한 자리씩 읽은 소리(911 구일일, 007 공공칠)도 받는다.
      1990s 의 1990 은 '구십'(90년대)도 받는다.
    - in·since·of 같은 낱말 뒤의 'nineteen eighty-four' 는 해 하나(1984)로 본다.
    """
    text = _EN_IDIOMS.sub(lambda m: " " * len(m.group()), text)     # 글자 위치는 그대로 둔다
    found: list[tuple[EnNumber, int, int]] = []                       # (수, 시작, 끝)
    tokens = list(_EN_TOKEN.finditer(text))
    phrase: list[str] = []
    span: list[int] = []
    before: list[str] = [""]                                          # 구절 바로 앞 낱말(소문자)

    def flush() -> None:
        hyperbole = len(phrase) == 1 and phrase[0] in _EN_BIG and before[0] == "a"
        if phrase and phrase != ["one"] and not hyperbole:
            found.append((EnNumber(text[span[0]:span[1]], _phrase_value(phrase)), span[0], span[1]))
        phrase.clear()
        span.clear()

    k = 0
    while k < len(tokens):
        m = tokens[k]
        if m.group("time"):
            flush()
            hour, minute = m.group("time").split(":")
            found.append((EnNumber(hour, int(hour)), m.start(), m.start() + len(hour)))
            if int(minute):
                found.append((EnNumber(minute, int(minute)), m.end() - 2, m.end()))
            k += 1
            continue
        if m.group("num"):
            flush()
            digits = m.group("num").replace(",", "")
            whole = digits.split(".")[0]
            nxt = tokens[k + 1] if k + 1 < len(tokens) else None
            if (nxt is not None and nxt.group().lower() in _EN_BIG and not text[m.end():nxt.start()].strip()):
                value = round(float(digits) * _EN_BIG[nxt.group().lower()])
                extra = {read_digits(digits)} if "." in digits else set()
                found.append((EnNumber(text[m.start():nxt.end()], value, frozenset(extra)), m.start(), nxt.end()))
                k += 2
                continue
            extra = {read_digits(digits)} if "." in digits else set()
            if len(whole) >= 3 or (len(whole) > 1 and whole.startswith("0")):
                extra |= {_digit_by_digit(whole, "공"), _digit_by_digit(whole, "영")}
            if len(whole) == 4 and (text[m.end():m.end() + 1] == "s" or text[m.end():m.end() + 2] in ("'s", "’s")):
                extra.add(sino_number(int(whole) % 100))
            found.append((EnNumber(m.group("num"), int(whole), frozenset(extra)), m.start(), m.end()))
            k += 1
            continue
        word = m.group().lower()
        if word in _EN_WORDS:
            if phrase and not _joins(phrase, word):
                flush()
            if not phrase:
                prev = tokens[k - 1].group().lower() if k > 0 else ""
                before[0] = prev if prev and not text[tokens[k - 1].end():m.start()].strip() else ""
            phrase.append(word)
            span[:] = [span[0] if span else m.start(), m.end()]
        elif (word == "and" and phrase and phrase[-1] in _EN_BIG and k + 1 < len(tokens)
              and tokens[k + 1].group().lower() in _EN_WORDS):
            pass                                       # two hundred and five
        else:
            flush()
        k += 1
    flush()
    return _join_years_and_halves(text, found)


def _join_years_and_halves(text: str, found: list[tuple[EnNumber, int, int]]) -> list[EnNumber]:
    """해 낱말(nineteen eighty-four)을 하나로 묶고, 시 바로 뒤의 30 에 '반' 을 더한다."""
    out: list[tuple[EnNumber, int, int]] = []
    for num, start, end in found:
        if out:
            prev, p_start, p_end = out[-1]
            gap = text[p_end:start]
            cue = re.findall(r"[A-Za-z]+", text[:p_start])
            if (not gap.strip(" -") and 11 <= prev.value <= 20 and " " not in prev.shown and "-" not in prev.shown
                    and 10 <= num.value <= 99 and not prev.shown[0].isdigit() and not num.shown[0].isdigit()
                    and cue and cue[-1].lower() in _YEAR_CUES):
                year = prev.value * 100 + num.value
                out[-1] = (EnNumber(text[p_start:end], year), p_start, end)
                continue
            if num.value == 30 and 1 <= prev.value <= 12 and not gap.strip(" -:"):
                num = num._replace(extra=num.extra | {"반"})
        out.append((num, start, end))
    return [num for num, _, _ in out]


def number_readings(value: int) -> set[str]:
    """수를 한국어로 읽는 소리들(한자어 전체, 고유어 낱말)."""
    readings = {sino_number(value)}
    if value == 0:
        readings.add("공")
    if value in _KO_NATIVE:
        readings.update(_KO_NATIVE[value])
    elif 10 < value < 100:
        readings.update(_KO_NATIVE[value // 10 * 10])
    return readings


def _has_reading(word: str, reading: str) -> bool:
    """낱말이 그 읽기로 시작하거나 '제' 다음에 그 읽기가 오는지(제3 -> 제삼 승강장)."""
    return word.startswith(reading) or word.startswith("제" + reading)


def missing_numbers(source_en: str, text: str) -> list[str]:
    """영어 대사의 수 가운데 후보에 없는 것(원문 표기 목록). 숫자 조건을 켰을 때만 쓴다(논문에 없음).

    후보에 같은 값의 아라비아 숫자가 있거나(코로나19, 제3), 후보의 한글 낱말 가운데 그 수의 읽는 소리
    (3 이면 삼, 세, 셋, 석, english_numbers 가 더한 읽기)로 시작하는 것이 있으면 있다고 본다.
    '열차' 의 '열' 처럼 우연히 맞는 낱말도 있는 느슨한 확인이다. 굳은 말·현지화(ten bucks -> 만 원, 911 -> 119),
    1990s -> 90년대 밖의 어림말은 '빠짐' 으로 볼 수 있어서 README 에 적고 기본으로 끈다.
    """
    numbers = english_numbers(source_en)
    if not numbers:
        return []
    words = _HANGUL_WORD.findall(read_digits(text))
    digit_values = {int(t.replace(",", "")) for t in re.findall(r"\d+(?:,\d{3})*", text)}
    return [num.shown for num in numbers
            if num.value not in digit_values
            and not any(_has_reading(w, r) for w in words for r in number_readings(num.value) | num.extra)]


def _floor2(x: float) -> str:
    """소수 둘째 자리까지 내림해 적는다. 0.7496 이 '0.75 < 0.75' 로 보이지 않게 한다."""
    return f"{math.floor(x * 100 + 1e-9) / 100:.2f}"


def assess(cands: Sequence[Candidate], item: LineItem, target: Target, labse_scores: Sequence[float],
           cfg: Config) -> list[Candidate]:
    """후보마다 음절 수·LaBSE·말투(숫자 조건을 켰으면 숫자도)를 재고 통과 여부와 탈락 까닭을 채운다.

    후보를 그대로 고쳐 돌려준다. labse_scores 는 후보마다 하나(pipeline 이 원문과 숫자를 푼 원문 가운데 큰 값을 준다).
    """
    if len(cands) != len(labse_scores):
        raise ValueError(f"후보 {len(cands)}개와 LaBSE 점수 {len(labse_scores)}개의 수가 다릅니다.")
    for cand, sim in zip(cands, labse_scores):
        reasons: list[str] = []
        cand.syllables = ko_syllable_count(cand.text)
        cand.ok_length = target.low <= cand.syllables <= target.high
        if not cand.ok_length:
            reasons.append(f"길이 {cand.syllables}음절(목표 {target.n}±{target.tol})")
        cand.labse = float(sim)
        not_korean = korean_problem(cand.text)
        cand.ok_meaning = cand.labse >= cfg.labse_min and not not_korean
        if cand.labse < cfg.labse_min:
            reasons.append(f"뜻 유사도 {_floor2(cand.labse)} < {cfg.labse_min:.2f}")
        if not_korean:
            reasons.append(not_korean)
        if cfg.check_numbers:
            missing = missing_numbers(item.source_en, cand.text)
            cand.ok_numbers = not missing
            if missing:
                reasons.append(f"숫자: {', '.join(missing)} 빠짐")
        else:
            cand.ok_numbers = None
        if cfg.check_register:
            ok, found, why = register_mod.check(cand.text, item.register)
        else:
            ok, found, why = True, register_mod.detect(cand.text), ""
        cand.ok_register, cand.register_found = ok, found
        if why:
            reasons.append(why)
        cand.reasons = reasons
    return cands


def line_register(cands: Sequence[Candidate]) -> str | None:
    """말투 미지정 줄에서 모델 2 후보(차수 1 이상) 다수가 쓴 말투. '반말' | '해요체'(높임말) | None(정하지 못함).

    반말 문장만 있는 후보는 반말, 해요체·하십시오체 문장만 있는 후보는 높임말로 센다. 불명 문장은 세지 않는다.
    두 쪽 수가 같으면 정하지 않는다.
    """
    casual = polite = 0
    for c in cands:
        if c.round < 1:
            continue
        found = set(c.register_found if c.register_found is not None else register_mod.detect(c.text)) - {"불명"}
        if found == {"반말"}:
            casual += 1
        elif found and "반말" not in found:
            polite += 1
    if casual == polite:
        return None
    return "반말" if casual > polite else "해요체"


def apply_line_register(item: LineItem, cands: Sequence[Candidate], cfg: Config) -> str | None:
    """말투 미지정 줄은 모델 2 후보 다수의 말투를 그 줄의 말투로 보고 후보마다 다시 본다. 정한 말투를 돌려준다.

    모델 2 는 미지정 줄에서 관계와 앞 대사로 말투를 하나 정해 모든 후보에 쓰라고 받는다. 앞뒤 문맥 없이 번역한
    모델 1 초벌이 다른 말투로 같은 조건에서 겨루지 않게 한다. 높임말 쪽이면 해요체·하십시오체 섞임은 받는다.
    말투 조건을 껐거나 말투를 정한 줄, 다수를 정하지 못한 줄은 그대로 둔다.
    """
    if not cfg.check_register or (item.register or "미지정") != "미지정":
        return None
    reg = line_register(cands)
    if reg is None:
        return None
    label = "반말" if reg == "반말" else "높임말"
    for c in cands:
        ok, found, _ = register_mod.check(c.text, reg)
        c.register_found = found
        if not ok:
            c.ok_register = False
            c.reasons.append(f"말투: 미지정 줄인데 모델 2 후보 다수({label})와 다름")
    return reg


def source_seconds(item: LineItem, target: Target, cfg: Config) -> float:
    """원문 대사가 차지하는 시간(초) 어림. 원문 프레임 열을 이 길이로 늘린다.

    seconds 기준이면 실제 칸 길이, explicit 이면 사람이 정한 음절 수가 차지하는 칸(target.seconds_est),
    relative 면 영어 음절 수 / 영어 음절 속도(en_rate)다.
    """
    if target.basis == "seconds" and item.seconds:
        return float(item.seconds)
    if target.basis == "explicit":
        return target.seconds_est
    return target.en_syllables / cfg.en_rate


def add_dtw(item: LineItem, target: Target, cands: Sequence[Candidate], cfg: Config,
            table: VowelTable | None = None) -> bool:
    """후보마다 원문과의 모음 DTW 를 cand.dtw 에 적는다. 원문에 음절이 없어 잴 수 없으면 False.

    원문은 source_seconds 길이로, 후보는 한국어 음절 속도(cfg.ko_rate)로 프레임 열을 만든다.
    프레임 열이 빈 후보(한글·영문 음절이 하나도 없음)는 dtw 가 None 으로 남는다.
    """
    syls = english_syllables(item.source_en)
    if not syls:
        return False
    src = frame_labels(syls, len(syls) / source_seconds(item, target, cfg), cfg)
    if not src:
        return False
    table = table if table is not None else load_or_build()
    for cand in cands:
        tgt = frame_labels(korean_syllables(cand.text), cfg.ko_rate, cfg)
        cand.dtw = ps_distance(src, tgt, table, cfg) if tgt else None
    return True


def _round9(x: float) -> float:
    return round(float(x), 9)


def _rank_key(cand: Candidate, index: int, target: Target | None) -> tuple:
    """점수 높은 순, 같으면 COMET 높은 순, 그다음 목표 음절 수에 가까운 순, 그다음 먼저 나온 순."""
    off = abs((cand.syllables or 0) - target.n) if target is not None else 0
    return (-_round9(cand.score if cand.score is not None else -math.inf),
            -_round9(cand.comet if cand.comet is not None else -math.inf), off, index)


def ps_comet(passing: Sequence[Candidate], cfg: Config, lip: bool, target: Target | None = None) -> Candidate:
    """고를 묶음(보통 통과 후보, selection_pool 참고) 가운데 하나를 고른다(논문 식 20). dtw_inv 와 score 를 채운다.

    lip 이면 DTW 를 묶음 안에서 최소-최대 정규화해 뒤집고(dtw_inv = 1 - (d - 최소) / (최대 - 최소),
    모두 같으면 1.0) score = alpha x dtw_inv + beta x COMET. lip 이 아니면 score = COMET.
    """
    if not passing:
        raise ValueError("고를 후보가 없습니다.")
    if any(c.comet is None for c in passing):
        raise ValueError("COMET 점수가 없는 후보가 있습니다.")
    if lip:
        if any(c.dtw is None for c in passing):
            raise ValueError("DTW 를 재지 못한 후보가 있습니다.")
        lo = min(c.dtw for c in passing)
        hi = max(c.dtw for c in passing)
        for c in passing:
            c.dtw_inv = 1.0 if hi - lo <= _SAME else 1.0 - (c.dtw - lo) / (hi - lo)
            c.score = cfg.alpha * c.dtw_inv + cfg.beta * c.comet
    else:
        for c in passing:
            c.dtw_inv = None
            c.score = c.comet
    order = sorted(range(len(passing)), key=lambda i: _rank_key(passing[i], i, target))
    return passing[order[0]]


def fallback(all_cands: Sequence[Candidate]) -> Candidate:
    """논문 3.1절 끝의 규칙: LaBSE 가 가장 높은 후보(같으면 먼저 나온 것).

    한글이 있는 후보가 하나라도 있으면 그 가운데서 고른다(번역 안 된 영어가 LaBSE 1.0 으로 뽑히지 않게).
    selection_pool 이 통과 후보도, 식 6·7 을 채운 후보도 없을 때만 쓴다.
    """
    scored = [c for c in all_cands if c.labse is not None]
    if not scored:
        raise ValueError("LaBSE 점수가 있는 후보가 없어서 고를 수 없습니다.")
    pool = [c for c in scored if not korean_problem(c.text)] or scored
    best = pool[0]
    for c in pool[1:]:
        if _round9(c.labse) > _round9(best.labse):
            best = c
    return best


# selection_pool 이 돌려주는 묶음 종류
POOL_PASS = "pass"      # 모든 조건(길이, 뜻, 말투, 켰으면 숫자)을 채운 후보
POOL_ISO = "iso"        # 통과 후보가 없어 논문 ISO 의 두 조건(식 6 길이, 식 7 LaBSE)과 한국어 여부만 채운 후보
POOL_LABSE = "labse"    # 그것도 없어 LaBSE 가 가장 높은 후보 하나(논문 3.1절 끝)
POOL_NONE = "none"      # 후보가 하나도 없음


def selection_pool(cands: Sequence[Candidate], draft_is_reference: bool = False) -> tuple[str, list[Candidate]]:
    """고를 후보 묶음과 그 종류(POOL_*).

    통과 후보가 있으면 그것, 없으면 논문 ISO 의 두 조건(식 6, 7)을 채운 후보(말투·숫자 조건만 못 채운 후보)를
    PS-Comet 으로 고르게 넘긴다. 둘 다 없을 때만 논문의 규칙대로 LaBSE 가 가장 높은 후보 하나를 넘긴다.
    draft_is_reference 면(초벌을 COMET 참조로 씀) 초벌(차수 0)은 자기 자신을 참조로 점수를 받아 늘 유리하므로,
    묶음에 다른 후보가 있으면 뺀다.
    """
    def without_draft(group: list[Candidate]) -> list[Candidate]:
        rest = [c for c in group if c.round >= 1] if draft_is_reference else group
        return rest or group

    passing = [c for c in cands if c.passed]
    if passing:
        return POOL_PASS, without_draft(passing)
    iso = [c for c in cands if c.iso_ok]
    if iso:
        return POOL_ISO, without_draft(iso)
    if cands:
        return POOL_LABSE, [fallback(cands)]
    return POOL_NONE, []


def ranked(cands: Sequence[Candidate], target: Target | None = None) -> list[Candidate]:
    """보여 줄 순서: 점수가 매겨진 후보(ps_comet 이 잰 묶음, 그 순서) 다음에 나머지(LaBSE 높은 순)."""
    index = {id(c): i for i, c in enumerate(cands)}
    head = sorted((c for c in cands if c.score is not None),
                  key=lambda c: _rank_key(c, index[id(c)], target))
    used = {id(c) for c in head}
    rest = sorted((c for c in cands if id(c) not in used),
                  key=lambda c: (-_round9(c.labse if c.labse is not None else -math.inf), index[id(c)]))
    return head + rest
