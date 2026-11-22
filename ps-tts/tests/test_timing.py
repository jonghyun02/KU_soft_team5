"""timing: 음절 열 -> 프레임 라벨 열."""
from __future__ import annotations

import random
from collections import Counter

import pytest

from pstts.config import Config
from pstts.phonetics import EN_VOWELS, KO_VOWELS, Syl, english_syllables, korean_syllables
from pstts.timing import frame_labels

CFG = Config()
VOWELS = sorted(EN_VOWELS | KO_VOWELS)


def expected_frames(n_syl: int, rate: float, cfg: Config = CFG) -> int:
    return round(n_syl / rate / cfg.frame_sec)


def random_syl(rng: random.Random) -> Syl:
    nucleus = [] if rng.random() < 0.1 else rng.sample(VOWELS, rng.choice([1, 1, 1, 2]))
    return Syl(onset=["k"] * rng.randint(0, 3), glide=rng.choice([None, None, "j", "w"]),
               nucleus=nucleus, coda=["n"] * rng.randint(0, 3), src="x")


def test_total_frames_random() -> None:
    rng = random.Random(7)
    for _ in range(300):
        syls = [random_syl(rng) for _ in range(rng.randint(1, 60))]
        rate = rng.uniform(2.5, 10.0)
        labels = frame_labels(syls, rate, CFG)
        assert abs(len(labels) - expected_frames(len(syls), rate)) <= 1


@pytest.mark.parametrize("text, lang", [
    ("I didn't order two tickets. What can I do for you?", "en"),
    ("It's the first time I ever double-crossed a governor.", "en"),
    ("내가 주지사를 속인 건 이번이 처음이야. 무슨 일로 왔어?", "ko"),
    ("리노에서 육 주, 그다음엔 버뮤다", "ko"),
])
def test_total_frames_real_text(text: str, lang: str) -> None:
    syls = english_syllables(text) if lang == "en" else korean_syllables(text)
    rate = CFG.en_rate if lang == "en" else CFG.ko_rate
    labels = frame_labels(syls, rate, CFG)
    assert abs(len(labels) - expected_frames(len(syls), rate)) <= 1
    assert {lab for lab in labels if lab is not None} <= {v for s in syls for v in s.nucleus}


def test_weights_split_inside_syllable() -> None:
    cfg = Config(frame_sec=0.01, cons_weight=1.0, glide_weight=0.5, vowel_weight=2.0)
    # 자음 1 + 모음 2 + 자음 1 = 4 -> 1초 음절에서 모음 0.5초 = 50프레임
    labels = frame_labels([Syl(onset=["k"], nucleus=["a"], coda=["n"])], 1.0, cfg)
    assert len(labels) == 100
    assert labels[:25] == [None] * 25 and labels[25:75] == ["a"] * 50 and labels[75:] == [None] * 25
    # 글라이드 0.5: 1 + 0.5 + 2 = 3.5 -> 모음 2/3.5 = 57프레임
    labels = frame_labels([Syl(onset=["k"], glide="w", nucleus=["a"])], 1.0, cfg)
    assert Counter(labels)["a"] == round(100 * 2 / 3.5)


def test_diphthong_halves() -> None:
    cfg = Config(frame_sec=0.01)
    labels = frame_labels([Syl(nucleus=["a", "ɪ"])], 1.0, cfg)
    c = Counter(labels)
    assert c["a"] == 50 and c["ɪ"] == 50
    assert labels.index("ɪ") > labels.index("a")


def test_vowelless_syllable_all_none() -> None:
    labels = frame_labels([Syl(onset=["p", "s", "s", "t"])], 5.0, CFG)
    assert len(labels) == expected_frames(1, 5.0) and set(labels) == {None}


def test_empty_and_order() -> None:
    assert frame_labels([], 6.0, CFG) == []
    labels = frame_labels(korean_syllables("아이우"), 5.0, CFG)
    seq = [lab for i, lab in enumerate(labels) if lab is not None and (i == 0 or labels[i - 1] != lab)]
    assert seq == ["a", "i", "u"]


def test_no_drift_over_long_sequence() -> None:
    # 반올림이 쌓이면 긴 열에서 프레임 수가 크게 벗어난다. 누적 반올림이면 1 프레임 안이다.
    syls = korean_syllables("가나다라마바사아자차카타파하" * 40)
    for rate in (4.7, 7.4, 9.3):
        assert abs(len(frame_labels(syls, rate, CFG)) - expected_frames(len(syls), rate)) <= 1


@pytest.mark.parametrize("rate", [0.0, -1.0])
def test_bad_rate(rate: float) -> None:
    with pytest.raises(ValueError):
        frame_labels([Syl(nucleus=["a"])], rate, CFG)


def test_bad_weights() -> None:
    with pytest.raises(ValueError):
        frame_labels([Syl(nucleus=["a"])], 5.0, Config(vowel_weight=0.0))
