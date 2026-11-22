"""paper.frontend: Frontend 인터페이스, make_frontend, ApproxFrontend(길이 합 = 음절/속도, 무게, 0길이 토큰, 소리 없는
초성 ㅇ, 모음표 라벨, 국소 비용), 기존 판 영어 음절 수와의 일치, 영어 모음 대응 상수와 PanPhon, UTF-8 모드 없이 돌기."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pstts import phonetics
from pstts.config import ROOT
from pstts.paper import g2p
from pstts.paper.config import PaperConfig
from pstts.paper.frontend import (APPROX_FRAME_SEC, EN_LABELS, ApproxFrontend, Frontend, en_label_map, ko_label_map,
                                  ko_vowel_info, make_frontend)
from pstts.paper.g2p import BLANK
from pstts.vowels import cost_matrix, load_or_build

VALID_TSV = Path("D:/transformer-base/enko-nmt/data/clean_v2/valid.tsv")


@pytest.fixture
def front() -> ApproxFrontend:
    return ApproxFrontend(PaperConfig())


# ---------------------------------------------------------------------------- 만들기
def test_make_frontend_approx() -> None:
    f = make_frontend(PaperConfig(), "cpu")
    assert isinstance(f, ApproxFrontend) and isinstance(f, Frontend)
    assert f.name == "approx" and f.frame_sec == APPROX_FRAME_SEC == 256 / 22050
    assert f.en_voice == "en-gb" and f.en_stress is False


def test_make_frontend_yourtts_is_imported_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    """tts='yourtts' 이면 그때 pstts.paper.yourtts.YourTTSFrontend(cfg, device) 를 부른다."""
    calls: list[tuple[PaperConfig, str]] = []

    class FakeYourTTS:
        def __init__(self, cfg: PaperConfig, device: str) -> None:
            calls.append((cfg, device))

    fake = types.ModuleType("pstts.paper.yourtts")
    fake.YourTTSFrontend = FakeYourTTS
    monkeypatch.setitem(sys.modules, "pstts.paper.yourtts", fake)
    cfg = PaperConfig(tts="yourtts")
    out = make_frontend(cfg, "cpu")
    assert isinstance(out, FakeYourTTS) and calls == [(cfg, "cpu")]


def test_frontend_module_does_not_import_yourtts_or_torch() -> None:
    code = ("import sys, pstts.paper.frontend; "
            "print('pstts.paper.yourtts' in sys.modules, 'torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                         encoding="utf-8", check=True)
    assert out.stdout.split() == ["False", "False"]


def test_make_frontend_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="approx, yourtts"):
        make_frontend(PaperConfig(tts="vits"), "cpu")


@pytest.mark.parametrize("change", [{"en_rate": 0.0}, {"coef": -1.0}, {"vowel_weight": 0.0},
                                    {"cons_weight": -0.1}, {"approx_null_cost": -1.0}, {"g2p_en_voice": "en-xx"}])
def test_approx_rejects_bad_values(change: dict) -> None:
    with pytest.raises(ValueError):
        ApproxFrontend(replace(PaperConfig(), **change))


def test_approx_reads_table_when_made(monkeypatch: pytest.MonkeyPatch) -> None:
    """모음표는 만들 때 읽는다. 문제가 있으면 ISO, PS 에서 모델 2 를 부르기 전(프런트엔드를 만들 때) 멈춘다."""
    import pstts.paper.frontend as frontend

    def broken() -> None:
        raise RuntimeError("모음표를 읽지 못함")

    monkeypatch.setattr(frontend, "load_or_build", broken)
    with pytest.raises(RuntimeError, match="모음표"):
        ApproxFrontend(PaperConfig())


# ---------------------------------------------------------------------------- 한국어 길이
def test_ko_weights_and_sum(front: ApproxFrontend) -> None:
    rate = PaperConfig().ko_rate
    toks = front.tokens("가", "ko")
    assert toks == [BLANK, "ㄱ", BLANK, "ㅏ", BLANK]
    (d,) = front.durations([toks], "ko")
    assert d == pytest.approx(np.array([0, 1, 0, 2, 0]) / 3 / rate)        # 자음 1 : 홑모음 2
    (d,) = front.durations([front.tokens("각", "ko")], "ko")
    assert d[1::2] == pytest.approx(np.array([1, 2, 1]) / 4 / rate)       # 받침 1
    (d,) = front.durations([front.tokens("아", "ko")], "ko")
    assert d[1] == 0 and d[3] == pytest.approx(1 / rate)                  # 소리 없는 초성 ㅇ 은 0
    toks = front.tokens("가야", "ko")
    (d,) = front.durations([toks], "ko")
    assert toks[1::2] == ["ㄱ", "ㅏ", "ㅇ", "ㅑ"]
    assert d[1::2] == pytest.approx(np.array([1, 2, 0, 2.5]) / 5.5 * 2 / rate)   # 글라이드가 든 모음 2.5


def test_ko_ieung_weight_depends_on_position(front: ApproxFrontend) -> None:
    """Fig. 4 처럼 초성 ㅇ 과 받침 ㅇ 은 같은 토큰이다. 근사는 다음 토큰이 모음이면 소리 없는 초성(0), 아니면 받침(자음 1)."""
    toks = front.tokens("안녕", "ko")
    assert toks[1::2] == ["ㅇ", "ㅏ", "ㄴ", "ㄴ", "ㅕ", "ㅇ"]
    assert front.weights(toks, "ko")[1::2].tolist() == [0.0, 2.0, 1.0, 1.0, 2.5, 1.0]
    toks = front.tokens("강아지", "ko")                                    # ㄱ ㅏ ㅇ ㅇ ㅏ ㅈ ㅣ
    assert front.weights(toks, "ko")[1::2].tolist() == [1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0]
    assert not front.weights(toks, "ko")[0::2].any()


def test_ko_glide_vowels_follow_phonetics() -> None:
    glide = {j for j, (_, has_glide) in ko_vowel_info().items() if has_glide}
    assert glide == set("ㅑㅒㅕㅖㅘㅙㅚㅛㅝㅞㅟㅠㅢ")
    assert len(ko_vowel_info()) == 21 and set(ko_vowel_info()) == g2p.KO_VOWEL_JAMO


def test_ko_syllables_are_vowel_jamo(front: ApproxFrontend) -> None:
    toks = front.tokens("같이 읽고 신라에 가자.", "ko")                 # 가치 일꼬 실라에 가자.
    assert front.syllables(toks, "ko") == 9
    (d,) = front.durations([toks], "ko")
    assert d.sum() == pytest.approx(9 / PaperConfig().ko_rate)


def test_ko_zero_length_tokens(front: ApproxFrontend) -> None:
    toks = front.tokens("PVC 바닥재, 좋아요!", "ko")                    # 바닥째, 조아요: ㅇ 은 둘 다 초성
    (d,) = front.durations([toks], "ko")
    silent = g2p.silent_ieung(toks)
    assert sum(silent) == 2
    for tok, x, s in zip(toks, d, silent):
        if tok in (BLANK, g2p.SPACE, ",", "!", "P", "V", "C") or s:
            assert x == 0.0
        else:
            assert x > 0.0


# ---------------------------------------------------------------------------- 영어 길이
def test_en_sum_weights_and_zero_tokens(front: ApproxFrontend) -> None:
    cfg = PaperConfig()
    text = "We have to leave before the storm hits."
    toks = front.tokens(text, "en")
    assert toks == g2p.tokens(text, "en")
    (d,) = front.durations([toks], "en")
    n = front.syllables(toks, "en")
    assert n == phonetics.english_syllable_count(text) == 9
    assert d.sum() == pytest.approx(n / cfg.en_rate)
    unit = d[toks.index("h")]                                            # 자음 1.0
    for tok, x in zip(toks, d):
        cls = g2p.token_class(tok, "en")
        want = {"vowel": 2.0, "consonant": 1.0, "glide": 0.5, "zero": 0.0}[cls]
        assert x == pytest.approx(want * unit)
    assert {t for t, x in zip(toks, d) if x == 0} == {BLANK, g2p.SPACE, "ː", "."}


@pytest.mark.parametrize("word", ["idea", "zero", "fire", "science", "people", "psst", "period", "viral",
                                  "beautiful", "theater", "naïve café", "here", "scared", "hour", "cure", "shop"])
def test_en_syllables_use_main_version_rules(front: ApproxFrontend, word: str) -> None:
    """음절은 en_rate 를 잰 셈(en-us 음소 + 기존 판 규칙)이라 en-gb 의 here(h iə)도 1음절로 센다."""
    assert front.syllables(front.tokens(word, "en"), "en") == phonetics.english_syllable_count(word)


@pytest.mark.parametrize("voice, stress", [("en-us", False), ("en-us", True), ("en-gb", True)])
def test_en_sentence_length_does_not_depend_on_token_voice(front: ApproxFrontend, voice: str, stress: bool) -> None:
    other = ApproxFrontend(replace(PaperConfig(), g2p_en_voice=voice, g2p_en_stress=stress))
    for text in ["I know you're scared. But if we stay here, we'll never find out what happened to her.",
                 "Three years later, the little bakery had become the busiest shop on the street."]:
        a, b = front.tokens(text, "en"), other.tokens(text, "en")
        assert a != b
        assert front.durations([a], "en")[0].sum() == pytest.approx(other.durations([b], "en")[0].sum(), abs=1e-12)


def test_en_sequence_must_come_from_this_frontend(front: ApproxFrontend) -> None:
    other = ApproxFrontend(PaperConfig())
    toks = other.tokens("Hello there.", "en")
    assert other.durations([toks], "en")[0].sum() > 0
    with pytest.raises(ValueError, match="tokens"):
        front.durations([toks], "en")
    assert front.durations([front.tokens("Hello there.", "en")], "en")[0] == pytest.approx(other.durations([toks], "en")[0])
    other.close()
    with pytest.raises(ValueError):
        other.durations([toks], "en")


def test_durations_batch_and_empty(front: ApproxFrontend) -> None:
    seqs = [front.tokens(t, "ko") for t in ["같이 가자.", "", "..."]]
    out = front.durations(seqs, "ko")
    assert [len(x) for x in out] == [len(s) for s in seqs]
    assert out[1].tolist() == [0.0] and not out[2].any()
    assert front.durations([front.tokens("", "en")], "en")[0].tolist() == [0.0]
    assert front.durations([], "en") == []


# ---------------------------------------------------------------------------- 라벨과 비용
def test_ko_labels(front: ApproxFrontend) -> None:
    want = {**dict.fromkeys("ㅏㅑㅘ", "a"), **dict.fromkeys("ㅐㅒㅙ", "ɛ"), **dict.fromkeys("ㅓㅕㅝ", "ʌ"),
            **dict.fromkeys("ㅔㅖㅚㅞ", "e"), **dict.fromkeys("ㅗㅛ", "o"), **dict.fromkeys("ㅜㅠ", "u"),
            "ㅡ": "ɯ", **dict.fromkeys("ㅣㅟㅢ", "i")}
    assert ko_label_map() == want
    toks = front.tokens("의사 회의", "ko")
    assert front.labels(toks, "ko") == [None, None, None, "i", None, None, None, "a", None, None,
                                        None, None, None, "e", None, None, None, "i", None]


def test_en_labels(front: ApproxFrontend) -> None:
    table = load_or_build()
    m = en_label_map()
    assert set(m) == g2p.EN_VOWEL_CHARS and set(m.values()) <= set(table.symbols) and m == EN_LABELS
    assert {k: m[k] for k in "ɚɐᵻɝɒ"} == {"ɚ": "ə", "ɐ": "ə", "ᵻ": "ɨ", "ɝ": "ɜ", "ɒ": "ɑ"}
    toks = front.tokens("Hello, world!", "en")                           # en-gb: həlˈəʊ 에서 강세표 뺀 hələʊ
    labs = front.labels(toks, "en")
    assert [x for x in labs if x] == ["ə", "ə", "ʊ", "ɜ"]
    assert all(lab is None for tok, lab in zip(toks, labs) if not g2p.is_vowel(tok, "en"))


def test_en_label_constant_matches_panphon() -> None:
    """EN_LABELS 는 PanPhon 으로 정한 대응을 적은 상수다. PanPhon 은 UTF-8 모드에서만 읽히므로 따로 띄워 확인한다."""
    code = ("import json, sys; from pstts.paper.frontend import derive_en_label_map; "
            "sys.stdout.write(json.dumps(derive_en_label_map(), ensure_ascii=True))")
    out = subprocess.run([sys.executable, "-X", "utf8", "-c", code], cwd=ROOT, capture_output=True, text=True,
                         encoding="utf-8", check=True)
    assert json.loads(out.stdout) == EN_LABELS


def test_approx_runs_without_utf8_mode() -> None:
    """UTF-8 모드가 아니어도(한국어 Windows 의 기본 cp949) 근사 프런트엔드가 PanPhon 없이 끝까지 돈다."""
    code = ("import sys, json; from pstts.paper.config import PaperConfig; "
            "from pstts.paper.frontend import ApproxFrontend; f = ApproxFrontend(PaperConfig()); "
            "e = f.tokens('Hello, world!', 'en'); k = f.tokens('안녕, 세상아!', 'ko'); "
            "d = f.durations([e], 'en')[0].sum() + f.durations([k], 'ko')[0].sum(); "
            "c = f.cost(f.labels(e, 'en'), f.labels(k, 'ko')); json.dumps(f.info()); "
            "sys.stdout.write(f'{sys.flags.utf8_mode} {c.shape == (len(e), len(k))} {d > 0}')")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONUTF8"}
    out = subprocess.run([sys.executable, "-X", "utf8=0", "-c", code], cwd=ROOT, capture_output=True, text=True,
                         encoding="utf-8", errors="replace", env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split() == ["0", "True", "True"]


@pytest.mark.parametrize("text", ["We have to leave before the storm hits.", "I didn't order two tickets.",
                                  "Please find out the earliest shipping date possible."])
def test_en_us_label_sequence_matches_main_version_nuclei(text: str) -> None:
    """토큰 목소리를 en-us 로 두면 모음 라벨 열이 기존 판(eSpeak en-us) 음절 핵 열과 같다."""
    f = ApproxFrontend(replace(PaperConfig(), g2p_en_voice="en-us"))
    labs = [x for x in f.labels(f.tokens(text, "en"), "en") if x]
    assert labs == [v for s in phonetics.english_syllables(text) for v in s.nucleus]


def test_table_sizes() -> None:
    table = load_or_build()
    assert len(phonetics.EN_VOWELS) == 15 and len(phonetics.KO_VOWELS) == 8
    assert set(table.symbols) == phonetics.EN_VOWELS | phonetics.KO_VOWELS


def test_cost_matches_main_version(front: ApproxFrontend) -> None:
    src = front.labels(front.tokens("Hello, world!", "en"), "en")
    tgt = front.labels(front.tokens("안녕, 세상아!", "ko"), "ko")
    c = front.cost(src, tgt)
    assert c.shape == (len(src), len(tgt))
    assert np.array_equal(c, cost_matrix(src, tgt, load_or_build(), null_cost=1.0))
    table = load_or_build()
    for i, a in enumerate(src):
        for j, b in enumerate(tgt):
            if a is None and b is None:
                assert c[i, j] == 0.0
            elif a is None or b is None:
                assert c[i, j] == 1.0
            else:
                assert c[i, j] == table.distance(a, b)


def test_null_cost_from_config() -> None:
    f = ApproxFrontend(replace(PaperConfig(), approx_null_cost=0.5))
    assert f.cost([None, "a"], ["i", None]).tolist() == [[0.5, 0.0], [load_or_build().distance("a", "i"), 0.5]]


def test_info_is_json(front: ApproxFrontend) -> None:
    info = front.info()
    json.dumps(info, ensure_ascii=False)
    assert info["name"] == "approx" and info["frame_sec"] == APPROX_FRAME_SEC
    assert info["ko_rate"] == pytest.approx(info["en_rate"] * info["coef"])
    assert info["weights"]["ko_glide_vowel"] == 2.5 and info["null_cost"] == 1.0
    assert len(info["vowel_table"]["sha1"]) == 40 and len(info["vowel_table"]["symbols"]) == 16
    assert info["g2p"]["versions"]["phonemizer"] == "3.6.0" and info["g2p"]["versions"]["g2pK"] == "0.9.4"
    assert info["g2p"]["en"]["voice"] == "en-gb" and info["g2p"]["en"]["with_stress"] is False
    assert info["g2p"]["scheme"] == g2p.SCHEME and "en-us" in info["en_syllable_count"]


# ---------------------------------------------------------------------------- 기존 판과 영어 음절 수
def test_en_syllables_agree_with_main_version_on_sample(front: ApproxFrontend) -> None:
    """표본 2,000줄 가운데 숫자 없는 줄 500개에서 기존 판(eSpeak 문장 통째) 음절 수와 같은 줄 수.

    2026-11-22 측정 499/500. 다른 1줄은 'T.gondii' 를 eSpeak 이 문장 통째로 읽으면 'T dot gondii' 로,
    phonemizer 는 마침표에서 나눠 'T gondii' 로 읽어서 생긴다(숫자 읽기 보정은 두 판 모두 숫자 없는 줄이라 무관).
    """
    assert VALID_TSV.is_file(), f"표본 파일이 없습니다: {VALID_TSV}"
    rows = [line.split("\t") for line in VALID_TSV.read_text(encoding="utf-8").splitlines()]
    en = [rows[k * len(rows) // 2000][0] for k in range(2000)]
    plain = [s for s in en if not re.search(r"\d", s)]
    pick = [plain[k * len(plain) // 500] for k in range(500)]
    same = sum(front.syllables(front.tokens(s, "en"), "en") == phonetics.english_syllable_count(s) for s in pick)
    assert same >= 499
