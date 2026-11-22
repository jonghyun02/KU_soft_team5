"""phonetics: eSpeak 영어 음절, 한글 음절 풀기, 음절 수."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from pstts import phonetics
from pstts.phonetics import (EN_NUCLEUS, EN_VOWELS, KO_VOWELS, UNKNOWN_TOKENS, Syl, english_syllable_count,
                             english_syllables, korean_syllable_count, korean_syllables, pronunciation)
from pstts.textnorm import ko_syllable_count

# 2026-11-21 에 영어 11,997+160+26줄을 eSpeak NG 1.52.0 en-us 로 바꿔 모은 토큰(강세 기호를 뺀 꼴).
OBSERVED_VOWEL_TOKENS = [
    "ɪ", "ə", "ɛ", "æ", "ʌ", "iː", "ɚ", "eɪ", "aɪ", "uː", "ᵻ", "i", "oʊ", "ɐ", "ɑː", "əl", "ɔːɹ", "ɜː",
    "ɔː", "ɑːɹ", "aʊ", "ʊ", "ɔ", "iə", "ɛɹ", "ɔɪ", "ʊɹ", "ɪɹ", "aɪɚ", "aɪə", "ææ", "ɑ̃",
]
OBSERVED_SYLLABIC = ["n̩"]
OBSERVED_CONSONANTS = [
    "n", "t", "s", "d", "ɹ", "k", "l", "z", "m", "ð", "p", "v", "f", "b", "w", "ŋ", "ʃ", "ɾ", "h", "ɡ", "j",
    "dʒ", "tʃ", "θ", "ʒ", "ʔ", "r", "ɬ",
]


def shape(syls: list[Syl]) -> list[tuple]:
    return [(s.onset, s.glide, s.nucleus, s.coda) for s in syls]


# ---------------------------------------------------------------------------- 한국어
def test_ui_word_initial_is_glide() -> None:
    s = korean_syllables("의사")
    assert shape(s) == [([], "ɰ", ["i"], []), (["s"], None, ["a"], [])]


def test_ui_non_initial_is_i() -> None:
    s = korean_syllables("회의")
    assert shape(s) == [(["h"], "w", ["e"], []), ([], None, ["i"], [])]


def test_ui_after_consonant_is_i() -> None:
    s = korean_syllables("희망")
    assert shape(s) == [(["h"], None, ["i"], []), (["m"], None, ["a"], ["ŋ"])]


def test_ui_particle_after_word_and_latin() -> None:
    # 낱말 첫 음절이 아닌 '의'는 i. 영문 낱말 바로 뒤에 붙은 조사도 첫 음절이 아니다.
    s = korean_syllables("나의 의의")
    assert [x.glide for x in s] == [None, None, "ɰ", None]
    s = korean_syllables("iPhone의")
    assert s[-1].glide is None and s[-1].nucleus == ["i"]


@pytest.mark.parametrize("text, vowel", [("져", "ʌ"), ("쳐", "ʌ"), ("쪄", "ʌ"), ("죠", "o"), ("쥬", "u"), ("졔", "e")])
def test_glide_drop_after_affricates(text: str, vowel: str) -> None:
    (s,) = korean_syllables(text)
    assert s.glide is None and s.nucleus == [vowel]


def test_glide_kept_elsewhere() -> None:
    assert [(s.glide, s.nucleus) for s in korean_syllables("겨야왜")] == [("j", ["ʌ"]), ("j", ["a"]), ("w", ["ɛ"])]


def test_numbers_read_as_sino() -> None:
    s = korean_syllables("20")
    assert [x.src for x in s] == ["이", "십"]
    assert korean_syllable_count("20명") == 3


def test_latin_word_inside_korean() -> None:
    s = korean_syllables("iPhone을 샀어")
    assert len(s) == 5
    assert [x.nucleus for x in s[:2]] == [[], []] and s[0].src == "iPhone"
    assert [x.src for x in s[2:]] == ["을", "샀", "어"]


def test_onset_coda() -> None:
    s = korean_syllables("앉아 닭값")
    assert shape(s) == [([], None, ["a"], ["n"]), ([], None, ["a"], []),
                        (["t"], None, ["a"], ["k"]), (["k"], None, ["a"], ["p"])]


@pytest.mark.parametrize("text", ["", "   ", "?!...", "~!@#$%^&*()", "ㅋㅋㅋ"])
def test_korean_empty_and_punctuation(text: str) -> None:
    assert korean_syllables(text) == []
    assert korean_syllable_count(text) == 0


def test_korean_nuclei_are_known_vowels() -> None:
    text = "".join(chr(0xAC00 + 28 * j) for j in range(21))     # 가 개 갸 ... 기: 중성 21자 전부
    syls = korean_syllables(text)
    assert len(syls) == 21
    assert {v for s in syls for v in s.nucleus} == KO_VOWELS
    assert all(len(s.nucleus) == 1 for s in syls)


@pytest.mark.parametrize("text", [
    "안녕하세요", "회의 중에 의사가 희망을 말했어요.", "20명이 1,500원씩 3.5배를 냈다", "iPhone을 샀어, OK?",
    "Lizzie 랑 Bruce 가 왔다", "", "?!", "ㅋㅋ 좋아~", "BBC의 뉴스", "졌어 쳐 줘",
])
def test_korean_count_matches_textnorm(text: str) -> None:
    assert korean_syllable_count(text) == ko_syllable_count(text)


# ---------------------------------------------------------------------------- 영어
def test_pronunciation_words_without_stress() -> None:
    words = pronunciation("Hello world.")
    assert words == (("h", "ə", "l", "oʊ"), ("w", "ɜː", "l", "d"))
    assert all("ˈ" not in p and "ˌ" not in p for w in words for p in w)


@pytest.mark.parametrize("text, n", [
    ("Hello world.", 3), ("I didn't order two tickets.", 7), ("idea", 3), ("theater", 3), ("period", 3),
    ("zero", 2), ("fire", 2), ("viral", 2), ("science", 2), ("button", 2), ("people", 2), ("music", 2),
    ("Psst!", 1), ("", 0), ("?!", 0), ("...", 0),
])
def test_english_syllable_count(text: str, n: int) -> None:
    assert english_syllable_count(text) == n


@pytest.mark.parametrize("text, n, before", [
    # (글, 지금 음절 수, 숫자를 eSpeak 에 그대로 줬을 때 음절 수). 2026-11-22 검토에서 길이 목표가 부풀던 줄들
    ("Call 911!", 4, 7),                     # nine one one(전에는 nine hundred eleven)
    ("Be there at 10:00.", 6, 8),            # ten o'clock(전에는 ten zero zero)
    ("My number is 555-1234.", 11, 20),      # 한 자리씩, dash 없음
    ("I was born in 1999.", 9, 11),          # nineteen ninety-nine
    ("Agent 007 is here.", 8, 10),           # oh oh seven
    ("platform 3", 3, 3),
])
def test_english_syllables_read_numbers_as_spoken(text: str, n: int, before: int) -> None:
    assert english_syllable_count(text) == n
    assert sum(len(phonetics._en_word_syllables(w)) for w in pronunciation(text)) == before


def test_english_structure() -> None:
    assert shape(english_syllables("music")) == [(["m"], "j", ["u"], []), (["z"], None, ["ɪ"], ["k"])]
    assert shape(english_syllables("I")) == [([], None, ["a", "ɪ"], [])]
    assert shape(english_syllables("order")) == [([], None, ["ɔ"], ["ɹ"]), (["d"], None, ["ə"], [])]
    assert shape(english_syllables("car")) == [(["k"], None, ["ɑ"], ["ɹ"])]
    assert shape(english_syllables("queen")) == [(["k"], "w", ["i"], ["n"])]


def test_syllabify_intervocalic_rule() -> None:
    w = phonetics._en_word_syllables
    # 모음 사이 자음 1개 -> 뒤 음절 첫소리
    assert shape(w(("ə", "b", "aʊ", "t"))) == [([], None, ["ə"], []), (["b"], None, ["a", "ʊ"], ["t"])]
    # 2개 이상 -> 마지막 1개만 뒤 첫소리
    assert shape(w(("ɛ", "k", "s", "t", "ɹ", "ə"))) == [([], None, ["ɛ"], ["k", "s", "t"]), (["ɹ"], None, ["ə"], [])]
    # 낱말 앞 자음은 첫 음절, 끝 자음은 마지막 음절
    assert shape(w(("s", "t", "ɹ", "ɛ", "ŋ", "k", "θ", "s"))) == [(["s", "t", "ɹ"], None, ["ɛ"], ["ŋ", "k", "θ", "s"])]
    # r 색 토큰의 꼬리 ɹ 도 자음으로 센다
    assert shape(w(("s", "ɑːɹ", "i"))) == [(["s"], None, ["ɑ"], []), (["ɹ"], None, ["i"], [])]


def test_syllabic_consonant_and_vowelless_word() -> None:
    w = phonetics._en_word_syllables
    assert shape(w(("b", "ʌ", "ʔ", "n̩"))) == [(["b"], None, ["ʌ"], []), (["ʔ"], None, [], ["n̩"])]
    assert shape(w(("p", "s", "s", "t"))) == [(["p", "s", "s", "t"], None, [], [])]
    assert w(()) == []


def test_two_syllable_tokens_and_before_r() -> None:
    w = phonetics._en_word_syllables
    assert [s.nucleus for s in w(("aɪ", "d", "iə"))] == [["a", "ɪ"], ["i"], ["ə"]]          # idea
    assert [s.nucleus for s in w(("z", "iə", "ɹ", "oʊ"))] == [["i", "ə"], ["o", "ʊ"]]       # zero
    assert [s.nucleus for s in w(("f", "aɪɚ"))] == [["a", "ɪ"], ["ə"]]                     # fire
    assert [s.nucleus for s in w(("v", "aɪɚ", "ɹ", "əl"))] == [["a", "ɪ"], ["ə"]]          # viral


def test_observed_tokens_all_map() -> None:
    UNKNOWN_TOKENS.clear()
    for tok in OBSERVED_VOWEL_TOKENS:
        kind, val = phonetics._classify(tok, None)
        assert kind == "V", tok
        assert all(1 <= len(nuc) <= 2 for nuc in val.nuclei), tok
        assert all(sym in EN_VOWELS for nuc in val.nuclei for sym in nuc), tok
        assert set(val.coda) <= {"ɹ", "l"}, tok
    for tok in OBSERVED_SYLLABIC:
        assert phonetics._classify(tok, None) == ("S", tok)
    for tok in OBSERVED_CONSONANTS:
        assert phonetics._classify(tok, None) == ("C", tok)
    assert UNKNOWN_TOKENS == set()


@pytest.mark.parametrize("text", ["Aaaah!", "Aaaaaaaaah!", "Whoaaaa", "Nooooo!", "Shhhh", "Grrrr", "中文", "emoji 😀"])
def test_odd_english_input_has_no_unknown_tokens(text: str) -> None:
    UNKNOWN_TOKENS.clear()
    syls = english_syllables(text)
    assert syls and all(v in EN_VOWELS for s in syls for v in s.nucleus)
    assert UNKNOWN_TOKENS == set()


def test_nasalized_vowel_strips_diacritic() -> None:
    UNKNOWN_TOKENS.clear()
    assert [s.nucleus for s in phonetics._en_word_syllables(("k", "w", "ɑ", "s", "ɑ̃"))] == [["ɑ"], ["ɑ"]]
    assert UNKNOWN_TOKENS == set()


def test_unknown_vowel_token_is_counted_and_remembered() -> None:
    UNKNOWN_TOKENS.clear()
    syls = phonetics._en_word_syllables(("k", "ɤ", "t"))
    assert shape(syls) == [(["k"], None, [], ["ɤ", "t"])]
    assert UNKNOWN_TOKENS == {"ɤ"}
    UNKNOWN_TOKENS.clear()


def test_en_vowel_sets() -> None:
    nuclei = {sym for ent in EN_NUCLEUS.values() for nuc in ent.nuclei for sym in nuc}
    assert nuclei <= EN_VOWELS
    assert "ɐ" not in EN_VOWELS and "ᵻ" not in EN_VOWELS and "ɚ" not in EN_VOWELS
    assert {"a", "e", "i", "o", "u"} <= EN_VOWELS


def test_english_nuclei_in_vowel_set() -> None:
    text = ("The quick brown fox jumps over the lazy dog. Fire, hour, here, there, car, more, tour, bird, water. "
            "Button, little, rhythm. Cooperate, idea, theatre, science, beautiful music, I'll, you'll, they're.")
    for s in english_syllables(text):
        assert all(v in EN_VOWELS for v in s.nucleus)
        assert len(s.nucleus) <= 2


def test_pronunciation_thread_safe() -> None:
    texts = [f"Sentence number {i} has {i % 7} quick words about {['fire', 'water', 'music'][i % 3]}."
             for i in range(60)]
    raw = pronunciation.__wrapped__                # 캐시를 건너뛰고 eSpeak 를 정말 부른다
    expected = [raw(t, "en-us") for t in texts]
    with ThreadPoolExecutor(max_workers=8) as pool:
        got = list(pool.map(lambda t: raw(t, "en-us"), texts * 3))
    assert got == expected * 3


def test_pronunciation_bad_voice() -> None:
    with pytest.raises(ValueError):
        pronunciation.__wrapped__("hello", "no-such-voice")
