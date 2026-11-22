"""논문에 적힌 예시 숫자로 핵심 함수를 검사한다."""
import math
import random

import pytest

from kplt.metrics import scd
from kplt.model.common import fit_lines, render, render_unit, split_output, syl_token
from kplt.phoneme import line_phonemes, pho, section_phonemes, song_pho
from kplt.text import en_syllables, ko_syllables, split_sections
from kplt.data.match import Matcher, recover, clean


# ---------- 음절 (그림 2, 4.2절, 4.3절 예시) ----------
def test_syllables_paper_examples():
    assert en_syllables("Hello") == 2                                  # 4.2절 "Hello" 2음절
    assert en_syllables("I'll stray off the path I'm walking") == 8     # 4.3절 s2 = 8
    assert ko_syllables("하늘을 피해 숨지") == 7                        # 4.3절 s~2 = 7
    assert ko_syllables("고요한 밤 거룩한 밤") == 8                      # Kim 2023 예시


def test_ko_syllables_mixed():
    assert ko_syllables("마지막엔 break your heart") == 4 + 3


# ---------- SCD 식 4 ----------
def test_scd_formula():
    assert scd([8], [7]) == pytest.approx((1 / 8 + 1 / 7) / 2)
    assert scd([5, 5], [5, 5]) == 0
    # 한쪽이 0 음절이면 정의되는 항 하나만 (= 1)
    assert scd([3], [0]) == pytest.approx(1.0)
    assert scd([0], [0]) == 0


# ---------- pho 식 3 ----------
def test_phonemes_paper_example():
    seq = section_phonemes(["On the winds can change", "I'll stray off the path I'm walking"])
    assert seq[:2] == ["AA", "N"]                       # 3.2절 "AA", "N", ...
    i = seq.index("<eos>")
    assert seq[i - 4:i] == ["CH", "EY", "N", "JH"]      # ..., "CH", "EY", "N", "JH", "<eos>"
    assert seq[i + 1] == "AY"                           # "AY", ...
    assert seq[-4:] == ["K", "IH", "NG", "<eos>"]       # ..., "K", "IH", "NG", "<eos>"
    assert not any(ch.isdigit() for p in seq for ch in p)


def test_pho_definition():
    # 같은 줄을 반복하면 서로 다른 bigram 비율이 줄어든다
    one = pho(["la la la"])
    many = pho(["la la la"] * 4)
    assert many < one
    seq = section_phonemes(["la la la"] * 4)
    bigrams = list(zip(seq, seq[1:]))
    assert many == pytest.approx(len(set(bigrams)) / len(bigrams))


def test_korean_jamo():
    assert line_phonemes("같이") == ["ㄱ", "ㅏ", "ㅌ", "ㅇ", "ㅣ"]        # 글자 그대로의 자모 (Kim 2023)
    assert line_phonemes("네") == line_phonemes("내")                      # ㅐ=ㅔ
    assert line_phonemes("왜") == line_phonemes("외") == line_phonemes("웨")
    assert line_phonemes("닭") == ["ㄷ", "ㅏ", "ㄹ", "ㄱ"]                  # 겹받침 나눔


def test_song_pho_mean_std():
    deg, var = song_pho([["la la"], ["la la la la la la"]])
    a, b = pho(["la la"]), pho(["la la la la la la"])
    assert deg == pytest.approx((a + b) / 2)
    assert var == pytest.approx(abs(a - b) / 2)


# ---------- 렌더링 (4.2절) ----------
def test_render_syl_and_sep():
    s, t = render(["하늘을", "피해 숨지"], ["i love", "you so much"], syl=True)
    assert s == "<SYL2> 하늘을 <SYL3> 피해 숨지"
    assert t == "<SYL2> i love <SYL3> you so much"
    s, t = render(["a", "b"], ["c", "d"], syl=False)
    assert s == "a <SEP> b" and t == "c <SEP> d"
    s, _ = render(["하늘을 피해 숨지"], None, syl=True, syl_counts=[7])
    assert s == "<SYL7> 하늘을 피해 숨지"
    assert syl_token(0) == "<SYL1>" and syl_token(99) == "<SYL49>"


def test_render_unit_shuffle_keeps_pairs():
    unit = {"src": [f"k{i}" for i in range(6)], "tgt": [f"e{i}" for i in range(6)], "shuffle": True}
    s, t = render_unit(unit, syl=False, rng=random.Random(0))
    ks = [x.strip() for x in s.split("<SEP>")]
    es = [x.strip() for x in t.split("<SEP>")]
    assert [k[1:] for k in ks] == [e[1:] for e in es]       # 같은 순서로 섞임
    assert sorted(ks) == [f"k{i}" for i in range(6)]


def test_split_and_fit():
    assert split_output("<pad> <SYL3> a b <SYL2> c</s>") == ["a b", "c"]
    assert split_output("a <SEP> b <SEP> c") == ["a", "b", "c"]
    assert fit_lines(["a", "b", "c"], 2) == ["a", "b c"]
    assert fit_lines(["a"], 3) == ["a", "", ""]


# ---------- Genius 마디 나누기 ----------
def test_split_sections():
    text = "[Verse 1]\nline one\nline two\n\n[Chorus]\nhook\n"
    assert split_sections(text) == [["line one", "line two"], ["hook"]]


# ---------- 지문 정렬 (저자 README 의 반짝반짝 예시) ----------
TWINKLE = """Twinkle, twinkle, little star,
How I wonder what you are!
Up above the world so high,
Like a diamond in the sky.

'Tis your bright and tiny spark,
Lights the trav'ller in the dark:"""


def test_matcher_readme_example():
    sym = [[["ttls", "twinkle", "star", 24], ["hiwwya", "how", "are", 20], ["uatwsh", "up", "high", 21],
            ["ladits", "like", "sky", 20]], [["'ybats", "'tis", "spark", 26], ["lttitd", "lights", "dark", 27]]]
    lines, st = recover(TWINKLE, sym, "en")
    assert st[""] == 0
    assert lines[0][0] == "twinkle twinkle little star"
    assert lines[1][0] == "'tis your bright and tiny spark"
    assert lines[1][1] == "lights the trav'ller in the dark"


def test_matcher_curly_apostrophe_both_ways():
    text = "Oh on this night I’m suddenly awake"
    # 저자 규칙: ’ 를 지움 -> im, 길이 28
    lines, st = recover(text, [[["ootnisa", "oh", "awake", 28]]], "en")
    assert st["exact"] == 1
    # 곧은 따옴표 판: i'm, 길이 29
    lines, st = recover(text, [[["ootnisa", "oh", "awake", 29]]], "en")
    assert st["exact"] == 1 and lines[0][0] == "oh on this night i'm suddenly awake"


def test_matcher_hangul_in_english_and_short_lines():
    text = "빨리빨리 피해 right\nyeah"
    lines, st = recover(text, [[["빨피r", "빨리빨리", "right", 11], ["y", "yeah", "yeah", 4]]], "en")
    assert lines[0] == ["빨리빨리 피해 right", "yeah"]


def test_matcher_fallback_spacing():
    # 지문은 "심장소리에" 를 한 단어로 본 띄어쓰기
    text = "네 심장 소리에 맞게 뛰기 시작해"
    lines, st = recover(text, [[["네심맞뛰시", "네", "시작해", 13]]], "ko")
    assert st["fallback"] == 1 and lines[0][0] == "네 심장 소리에 맞게 뛰기 시작해"


# ---------- 검토 뒤 고친 것 ----------
def test_curly_and_cyrillic_normalized():
    assert en_syllables("I’m walking") == en_syllables("I'm walking") == 3
    assert en_syllables("don’t") == 1
    assert en_syllables("exaggеration") == en_syllables("exaggeration")      # 키릴 е
    assert line_phonemes("don’t") == line_phonemes("don't")
    assert ko_syllables("1, 2, 3") == en_syllables("1, 2, 3") == 3          # 숫자 같은 규칙
    assert en_syllables("빨리빨리 right") == 5                              # 영어판 속 한글


def test_processed_file_headers_do_not_shift_lines():
    text = ("[Verse 1: Krystal & Sulli]\nYou don't know me\nHit you with a smile, not goodbye\n\n"
            "[Chorus]\nBaby no more real love")
    sym = [[["ydkm", "you", "me", 14], ["hywasng", "hit", "goodbye", 26]], [["bnmrl", "baby", "love", 18]]]
    lines, st = recover(text, sym, "en")
    assert lines == [["you don't know me", "hit you with a smile not goodbye"], ["baby no more real love"]]


def test_matcher_cyrillic_line_comes_back_latin():
    lines, _ = recover("Thе way I cried for you", [[["twicfy", "th", "you", 17]]], "en")
    assert lines[0][0] == "the way i cried for you"


def test_load_meta_fixes_row_944():
    from kplt.config import VENDOR
    from kplt.text import load_meta
    m = load_meta(VENDOR / "meta.csv")
    assert m["944"]["GENRE"] == "k-pop" and m["944"]["IS_OFFICIAL"].startswith("1")
    assert len(m) == 1000


def test_dataset_song_overlap():
    from kplt.data.dump import DatasetSongs
    same = DatasetSongs()
    assert same("BTS ()", "FIRE 불타오르네")
    assert same("BIGBANG", "뱅뱅뱅 BANG BANG BANG")
    assert same("Donny Osmond", "I'll Make a Man Out of You")
    assert not same("IU", "Good Day")
