"""textnorm: 숫자 읽기, 음절 세기, 문장 나누기, 후보 다듬기."""
from __future__ import annotations

import pytest

from pstts.textnorm import (clean_candidate, dedupe_key, english_number_words, ko_syllable_count, latin_syllables,
                            read_digits, sino_number, speak_english_numbers, split_sentences)


@pytest.mark.parametrize("n, spoken", [
    (0, "영"), (1, "일"), (10, "십"), (11, "십일"), (20, "이십"), (100, "백"), (1000, "천"),
    (1940, "천구백사십"), (10000, "만"), (10001, "만일"), (20000, "이만"), (110000, "십일만"),
    (100000000, "일억"), (123456789, "일억이천삼백사십오만육천칠백팔십구"),
])
def test_sino_number(n: int, spoken: str) -> None:
    assert sino_number(n) == spoken


@pytest.mark.parametrize("text, spoken", [
    ("20명", "이십명"),
    ("1,500원", "천오백원"),
    ("12,000,000달러", "천이백만달러"),
    ("3.5배", "삼점오배"),
    ("0.25", "영점이오"),
    ("숫자 없음", "숫자 없음"),
    ("1, 2, 3", "일, 이, 삼"),
])
def test_read_digits(text: str, spoken: str) -> None:
    assert read_digits(text) == spoken


@pytest.mark.parametrize("word, n", [
    ("Ruth", 1), ("Hildy", 2), ("Bruce", 1), ("Jane", 1), ("table", 2), ("BBC", 1), ("Lizzie", 2),
    ("iPhone", 2), ("Lee", 1),
])
def test_latin_syllables(word: str, n: int) -> None:
    assert latin_syllables(word) == n


@pytest.mark.parametrize("text, n", [
    ("안녕하세요", 5),
    ("안녕, 하세요!", 5),
    ("20명", 3),                 # 이십명
    ("1,500원", 4),              # 천오백원
    ("iPhone을 샀어", 5),        # iPhone 2 + 을샀어 3
    ("OK, 좋아.", 3),
    ("ㅋㅋㅋ 좋아", 2),          # 낱자모는 세지 않는다(Sandglass v3 와 같음)
    ("", 0),
    ("?!...", 0),
    ("   ", 0),
])
def test_ko_syllable_count(text: str, n: int) -> None:
    assert ko_syllable_count(text) == n


def test_split_sentences_basic() -> None:
    text = "안녕? 잘 지냈어! 나는... 좋아~ 그래."
    assert split_sentences(text) == ["안녕?", "잘 지냈어!", "나는...", "좋아~", "그래."]


def test_split_sentences_newline_and_trailing() -> None:
    assert split_sentences("첫 줄\n둘째 줄") == ["첫 줄", "둘째 줄"]
    assert split_sentences("끝 부호 없음") == ["끝 부호 없음"]
    assert split_sentences("정말?! 진짜…") == ["정말?!", "진짜…"]


def test_split_sentences_keeps_quote_and_decimal() -> None:
    assert split_sentences('"정말?" 그가 물었다.') == ['"정말?"', "그가 물었다."]
    assert split_sentences("3.5배야. 좋아") == ["3.5배야.", "좋아"]


def test_split_sentences_drops_empty() -> None:
    assert split_sentences("") == []
    assert split_sentences("...") == []
    assert split_sentences("...그래.") == ["그래."]
    assert split_sentences("\n\n?!\n") == []


@pytest.mark.parametrize("raw, clean", [
    ("1. 안녕하세요", "안녕하세요"),
    ("2) 좋아", "좋아"),
    ("(3) 좋아", "좋아"),
    ("- 그래", "그래"),
    ("• 그래", "그래"),
    ('"안녕"', "안녕"),
    ("“안녕”", "안녕"),
    ("「좋아」", "좋아"),
    ("안녕하세요 (12음절)", "안녕하세요"),
    ("안녕 [해요체] 하세요", "안녕 하세요"),
    ("  여러   칸  ", "여러 칸"),
    ("- 1. \"좋아\" (해요체)", "좋아"),
    ("1.5배 빨라", "1.5배 빨라"),             # 소수는 번호가 아니다
    ('"정말?" 그가 물었다.', '"정말?" 그가 물었다.'),   # 앞뒤가 같은 따옴표 한 쌍이 아니다
    ("'그래', '좋아'", "'그래', '좋아'"),
    ("", ""),
])
def test_clean_candidate(raw: str, clean: str) -> None:
    assert clean_candidate(raw) == clean


def test_dedupe_key() -> None:
    assert dedupe_key("안녕, 하세요!") == dedupe_key("안녕하세요")
    assert dedupe_key("OK 좋아") == dedupe_key("ok, 좋아...")
    assert dedupe_key("안녕하세요") != dedupe_key("안녕하세요요")
    assert dedupe_key("?! ...") == ""


# ---------------------------------------------------------------- 영어 수를 말하는 꼴로

@pytest.mark.parametrize("n, words", [
    (0, "zero"), (7, "seven"), (19, "nineteen"), (20, "twenty"), (35, "thirty-five"), (100, "one hundred"),
    (105, "one hundred five"), (1000, "one thousand"), (2024, "two thousand twenty-four"),
    (1_500_000, "one million five hundred thousand"), (10**9 + 7, "one billion seven"),
])
def test_english_number_words(n: int, words: str) -> None:
    assert english_number_words(n) == words


@pytest.mark.parametrize("text, spoken", [
    # 2026-11-22 검토: eSpeak 가 숫자를 그대로 읽으면 음절 수가 부풀던 꼴들
    ("Call 911!", "Call nine one one!"),
    ("Be there at 10:00.", "Be there at ten o'clock."),
    ("8:00 a.m.", "eight o'clock a.m."),
    ("My number is 555-1234.", "My number is five five five, one two three four."),
    ("Agent 007 is here.", "Agent oh oh seven is here."),
    ("Room 101, please.", "Room one oh one, please."),
    ("I was born in 1999.", "I was born in nineteen ninety-nine."),
    ("Meet me at 7:30.", "Meet me at seven thirty."),
    ("It's 8:05 now.", "It's eight oh five now."),
    ("15:00", "fifteen hundred"),
    ("In 2024 we moved.", "In twenty twenty-four we moved."),
    ("It was 2005.", "It was two thousand five."),
    ("1500 people", "fifteen hundred people"),
    ("2000 dollars", "two thousand dollars"),
    ("He's 35 years old.", "He's thirty-five years old."),
    ("It cost 1.5 million dollars.", "It cost one point five million dollars."),
    ("Flight 714 to Seoul.", "Flight seven fourteen to Seoul."),
    ("It's 24/7.", "It's twenty-four seven."),
    ("The 1990s were wild.", "The nineteen nineties were wild."),
    ("In his 30s", "In his thirties"),
    ("the 3rd floor, 21st, 100th", "the third floor, twenty-first, one hundredth"),
    ("$20, $1, $3.50", "twenty dollars, one dollar, three dollars fifty"),
    ("50% off", "fifty percent off"),
    ("#1 and No. 5", "number one and number five"),
    ("at 5pm.", "at five p.m."),
    ("10-20 minutes", "ten to twenty minutes"),
    ("1939-1945", "nineteen thirty-nine to nineteen forty-five"),
    ("1-800-555-0199", "one, eight oh oh, five five five, oh one nine nine"),
    ("COVID-19", "COVID-nineteen"),
    ("1,500 people, 3.5 seconds", "one thousand five hundred people, three point five seconds"),
    ("No digits here.", "No digits here."),
])
def test_speak_english_numbers(text: str, spoken: str) -> None:
    assert speak_english_numbers(text) == spoken
