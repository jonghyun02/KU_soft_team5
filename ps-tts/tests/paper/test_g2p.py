"""paper.g2p: 빈 토큰 꼴, 공백 토큰, 영어 목소리와 강세표(논문 Fig. 4, 5c), g2pK 발음 규칙, 호환 자모(Fig. 4),
소리 없는 초성 ㅇ, 모음 판정, 표본 말뭉치 분류(Fig. 5c 모음), G2P 설정 파일, 기존 pstts.phonetics eSpeak 와 한 프로세스에서
번갈아 쓰기."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from pstts import phonetics
from pstts.paper import g2p
from pstts.paper.g2p import BLANK, SPACE

# 표본 말뭉치(읽기만). 영역별로 500줄씩 모여 있어 고르게 2,000줄을 뽑는다.
VALID_TSV = Path("D:/transformer-base/enko-nmt/data/clean_v2/valid.tsv")
N_SAMPLE = 2000
# 논문 Fig. 5(c) 의 영어 축(학습한 모델의 영어 모음 14개)과 한국어 축(21개)
FIG5C_EN = set("aeiuɐɑɒɔəɛɜɪʊʌ")
FIG5C_KO = set("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")


@pytest.fixture
def unknown_guard() -> Iterator[None]:
    """시험이 일부러 넣은 모르는 토큰을 시험 뒤에 지운다."""
    before = set(g2p.UNKNOWN_TOKENS)
    yield
    g2p.UNKNOWN_TOKENS.clear()
    g2p.UNKNOWN_TOKENS.update(before)


def coqui_intersperse(seq: list[str], blank: str) -> list[str]:
    """Coqui TTS TTSTokenizer.intersperse_blank_char 의 본문(TTS/tts/utils/text/tokenizer.py)."""
    result = [blank] * (len(seq) * 2 + 1)
    result[1::2] = seq
    return result


def sample_rows() -> list[list[str]]:
    assert VALID_TSV.is_file(), f"표본 파일이 없습니다: {VALID_TSV}"
    rows = [line.split("\t") for line in VALID_TSV.read_text(encoding="utf-8").splitlines()]
    return [rows[k * len(rows) // N_SAMPLE] for k in range(N_SAMPLE)]


# ---------------------------------------------------------------------------- 빈 토큰
def test_with_blanks_shape() -> None:
    assert g2p.with_blanks([]) == [BLANK]
    assert g2p.with_blanks(["a"]) == [BLANK, "a", BLANK]
    assert g2p.with_blanks(("a", " ", "b")) == [BLANK, "a", BLANK, " ", BLANK, "b", BLANK]


@pytest.mark.parametrize("n", [0, 1, 2, 7, 30])
def test_with_blanks_matches_coqui(n: int) -> None:
    seq = [chr(ord("a") + i % 26) for i in range(n)]
    out = g2p.with_blanks(seq)
    assert out == coqui_intersperse(seq, BLANK)
    assert len(out) == 2 * n + 1 and set(out[0::2]) == {BLANK} and out[1::2] == seq


def test_with_blanks_does_not_change_input() -> None:
    seq = ["x", "y"]
    g2p.with_blanks(seq)
    assert seq == ["x", "y"]


def test_blank_is_not_a_real_character() -> None:
    assert len(BLANK) > 1
    assert BLANK not in g2p.phonemes("<b> blank", "en")
    assert BLANK not in g2p.phonemes("<b> 빈칸", "ko")


@pytest.mark.parametrize("text, lang", [("We have to leave.", "en"), ("같이 읽고 가자.", "ko"), ("", "en"), ("", "ko")])
def test_tokens_are_phonemes_with_blanks(text: str, lang: str) -> None:
    assert g2p.tokens(text, lang) == g2p.with_blanks(g2p.phonemes(text, lang))
    assert (g2p.tokens(text, lang, en_voice="en-us", en_stress=True)
            == g2p.with_blanks(g2p.phonemes(text, lang, en_voice="en-us", en_stress=True)))


# ---------------------------------------------------------------------------- 한국어 g2pK, 호환 자모
@pytest.mark.parametrize("text, pron", [
    ("같이", "가치"),                       # 구개음화
    ("읽고", "일꼬"),                       # 겹받침, 된소리
    ("신라", "실라"),                       # 유음화
    ("3시 10분에 만나", "세시 십뿌네 만나"),  # 숫자 읽기(고유어 세, 한자어 십)
    ("PVC 바닥재", "PVC 바닥째"),           # 사전에 없는 약어는 로마자로 남음
    ("의사 회의 희망", "의사 회의 히망"),     # 희 -> 히
])
def test_g2pk_rules(text: str, pron: str) -> None:
    assert g2p.ko_pronounce(text) == pron


def test_ko_tokens_are_compatibility_jamo_like_fig4() -> None:
    """Fig. 4: 안녕 -> ㅇ ㅏ ㄴ ㄴ ㅕ ㅇ. 초성과 받침이 같은 토큰이라 두 ㅇ(24)과 두 ㄴ(1)이 같은 번호를 받는다."""
    toks = g2p.phonemes("안녕", "ko")
    assert toks == ["ㅇ", "ㅏ", "ㄴ", "ㄴ", "ㅕ", "ㅇ"]
    assert toks[0] == toks[5] and toks[2] == toks[3]
    assert g2p.phonemes("같이", "ko") == ["ㄱ", "ㅏ", "ㅊ", "ㅣ"]                  # 가치
    assert g2p.phonemes("닭", "ko") == ["ㄷ", "ㅏ", "ㄱ"]                          # 닥: 받침 ㄱ 도 초성 ㄱ 과 같은 토큰
    for tok in g2p.phonemes("폭풍이 오기 전에 떠나야 해", "ko"):
        assert tok == SPACE or 0x3131 <= ord(tok) <= 0x3163


def test_silent_ieung_is_onset_before_vowel() -> None:
    toks = g2p.phonemes("안녕 강아지", "ko")          # ㅇㅏㄴㄴㅕㅇ ㄱㅏㅇㅇㅏㅈㅣ
    silent = g2p.silent_ieung(toks)
    assert [i for i, s in enumerate(silent) if s] == [0, 10]     # 안 의 ㅇ, 아 의 ㅇ. 녕, 강 의 받침 ㅇ 은 아님
    assert toks[5] == toks[9] == "ㅇ" and not silent[5] and not silent[9]
    with_b = g2p.silent_ieung(g2p.with_blanks(toks))              # 빈 토큰은 건너뛰고 다음 토큰을 본다
    assert with_b[1::2] == silent and not any(with_b[0::2])
    assert g2p.silent_ieung([]) == [] and g2p.silent_ieung(["ㅇ"]) == [False]


def test_ko_space_tokens() -> None:
    toks = g2p.phonemes("같이 가", "ko")
    assert toks == ["ㄱ", "ㅏ", "ㅊ", "ㅣ", SPACE, "ㄱ", "ㅏ"]
    assert g2p.phonemes("  같이   가  ", "ko") == toks          # 공백 줄은 칸 하나, 앞뒤 공백 없음


def test_ko_punctuation_and_leftovers_stay_as_tokens() -> None:
    assert g2p.phonemes("...", "ko") == [".", ".", "."]
    toks = g2p.phonemes("PVC 바닥재.", "ko")
    assert toks[:4] == ["P", "V", "C", SPACE] and toks[-1] == "."


# ---------------------------------------------------------------------------- 영어 phonemizer
def test_en_default_is_en_gb_without_stress() -> None:
    """기본은 en-gb, 강세표 없음(Fig. 5c 의 영어 모음, Fig. 4 의 강세표 없는 토큰). espeak NG 1.52.0 출력."""
    assert g2p.EN_VOICE == "en-gb" and g2p.EN_WITH_STRESS is False
    assert g2p.en_phonemize("Hello, world!").ipa == "hələʊ, wɜːld!"
    toks = g2p.phonemes("Hello, world!", "en")
    assert toks == list("hələʊ, wɜːld!")
    assert "ˈ" not in toks and "ˌ" not in toks and "ː" in toks and toks.count(SPACE) == 1 and toks[-1] == "!"


def test_en_fig4_hello_is_the_en_us_form() -> None:
    """Fig. 4 의 hello 토큰 h ə l o ʊ 는 en-us(강세표 없음) 꼴이고 en-gb 는 h ə l ə ʊ 다(두 그림이 어긋남)."""
    assert g2p.phonemes("hello", "en", en_voice="en-us") == ["h", "ə", "l", "o", "ʊ"]
    assert g2p.phonemes("hello", "en") == ["h", "ə", "l", "ə", "ʊ"]


def test_en_voice_and_stress_options() -> None:
    assert g2p.en_phonemize("Hello, world!", "en-us", True).ipa == "həlˈoʊ, wˈɜːld!"
    assert g2p.phonemes("Hello, world!", "en", en_voice="en-gb", en_stress=True) == list("həlˈəʊ, wˈɜːld!")
    assert g2p.phonemes("Hello", "en", en_voice="EN-US") == g2p.phonemes("Hello", "en", en_voice="en-us")


def test_check_en_voice() -> None:
    assert g2p.check_en_voice(" EN-GB ") == "en-gb" and g2p.check_en_voice("en-us") == "en-us"
    with pytest.raises(ValueError, match="en 으로 시작"):
        g2p.check_en_voice("fr")
    with pytest.raises(ValueError, match="en-gb"):            # 없는 목소리면 있는 영어 목소리를 적는다
        g2p.check_en_voice("en-xx")
    with pytest.raises(ValueError):
        g2p.phonemes("hello", "en", en_voice="en-xx")


def test_en_space_tokens() -> None:
    toks = g2p.phonemes("We have to leave before the storm hits.", "en")
    assert toks[0] != SPACE and toks[-1] != SPACE
    assert "".join(toks).split(SPACE) == ["wiː", "hav", "tə", "liːv", "bɪfɔː", "ðə", "stɔːm", "hɪts."]
    assert g2p.phonemes("hello    world\n", "en") == g2p.phonemes("hello world", "en")


def test_en_empty_and_punctuation_only() -> None:
    assert g2p.phonemes("", "en") == [] and g2p.tokens("   ", "en") == [BLANK]
    assert g2p.en_phonemize("...") == g2p.EnPhonemes("...", ())


@pytest.mark.parametrize("voice, stress", [("en-gb", False), ("en-us", True)])
def test_en_phone_separator_does_not_change_tokens(voice: str, stress: bool) -> None:
    """낱말별 음소를 얻으려고 넣은 '|' 를 지운 글은 구분자 없이 부른 phonemizer 결과와 같다."""
    backend = g2p._en_backend(voice, stress)
    for text in ["Hello, world!", "The U.S. army (ret.) \"said\" no; fine?", "I didn't order two tickets."]:
        with g2p._EN_LOCK:
            plain = backend.phonemize([text], strip=True)[0]
        assert g2p.en_phonemize(text, voice, stress).ipa == " ".join(plain.split())


def test_en_phones_match_pstts_pronunciation() -> None:
    """en-us 낱말 음소는 기존 판 eSpeak(en-us)과 같다(근사의 영어 음절 셈이 기존 판과 같은 까닭)."""
    for text in ["we have to leave before the storm hits", "an idea about the zero period of the viral fire",
                 "people psst beautiful science"]:
        assert g2p.en_phones(text, "en-us") == [list(w) for w in phonetics.pronunciation(text)]


def test_en_words_do_not_depend_on_stress() -> None:
    for text in ["We have to leave before the storm hits.", "Three years later, the little bakery had become busy."]:
        assert g2p.en_phonemize(text, "en-us", True).words == g2p.en_phonemize(text, "en-us", False).words


def test_en_phones_split_at_inner_punctuation() -> None:
    # 'U.S.' 는 마침표에서 따로 음소화돼 붙는다. 마침표 자리를 음소 경계로 본다.
    assert ["j", "uː", "ɛ", "s"] in g2p.en_phones("The U.S. army", "en-us")
    assert ["j", "uː", "ɛ", "s"] in g2p.en_phones("The U.S. army")


def test_engines_coexist_in_one_process() -> None:
    """phonemizer(임시 폴더에 복사한 espeak)와 pstts.phonetics(원본 espeak DLL)를 번갈아 불러도 결과가 같다."""
    texts = ["we have to leave before the storm hits", "i did not order two tickets", "people like science"]
    first = None
    for _ in range(3):
        round_out = []
        for text in texts:
            old = phonetics.pronunciation.__wrapped__(text)       # 캐시를 거치지 않고 엔진을 부른다
            new = g2p.en_phonemize.__wrapped__(text, "en-us", False)
            gb = g2p.en_phonemize.__wrapped__(text)               # 기본(en-gb) 목소리도 번갈아 부른다
            round_out.append((old, new, gb))
            assert [list(w) for w in old] == [list(w) for w in new.words]
        first = first or round_out
        assert round_out == first


# ---------------------------------------------------------------------------- 모음 판정
def test_vowel_sets() -> None:
    assert len(g2p.KO_VOWEL_JAMO) == 21 and g2p.KO_VOWEL_JAMO == FIG5C_KO
    assert {ord(c) for c in g2p.KO_VOWEL_JAMO} == set(range(0x314F, 0x3164))
    assert {ord(c) for c in g2p.KO_CONSONANT_JAMO} == set(range(0x3131, 0x314F)) and len(g2p.KO_CONSONANT_JAMO) == 30
    assert not g2p.KO_VOWEL_JAMO & g2p.KO_CONSONANT_JAMO and g2p.KO_IEUNG in g2p.KO_CONSONANT_JAMO
    assert set("iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻɚɝ") <= g2p.EN_VOWEL_CHARS
    assert FIG5C_EN <= g2p.EN_VOWEL_CHARS
    assert not g2p.EN_VOWEL_CHARS & (g2p.EN_CONSONANT_CHARS | g2p.EN_GLIDE_CHARS)
    assert not g2p.EN_CONSONANT_CHARS & g2p.EN_GLIDE_CHARS


@pytest.mark.parametrize("token, lang, kind", [
    ("ɚ", "en", "vowel"), ("ᵻ", "en", "vowel"), ("ɐ", "en", "vowel"), ("ɪ", "en", "vowel"), ("ɒ", "en", "vowel"),
    ("j", "en", "glide"), ("w", "en", "glide"),
    ("ɹ", "en", "consonant"), ("ɡ", "en", "consonant"), ("ʔ", "en", "consonant"),
    ("ˈ", "en", "mark"), ("ˌ", "en", "mark"), ("ː", "en", "mark"), ("\u0329", "en", "mark"),
    (",", "en", "punct"), ("—", "en", "punct"), ("“", "en", "punct"),
    (BLANK, "en", "blank"), (SPACE, "en", "space"), (BLANK, "ko", "blank"), (SPACE, "ko", "space"),
    ("ㅏ", "ko", "vowel"), ("ㅢ", "ko", "vowel"), ("ㅘ", "ko", "vowel"),
    ("ㄱ", "ko", "consonant"), ("ㅎ", "ko", "consonant"), ("ㄳ", "ko", "consonant"), ("ㅇ", "ko", "consonant"),
    (".", "ko", "punct"), ("·", "ko", "punct"), ("ᆞ", "ko", "punct"), ("ㆍ", "ko", "punct"), ("「", "ko", "punct"),
    ("°", "ko", "punct"),
    ("P", "ko", "foreign"), ("k", "ko", "foreign"), ("α", "ko", "foreign"), ("世", "ko", "foreign"),
])
def test_token_kind(token: str, lang: str, kind: str) -> None:
    assert g2p.token_kind(token, lang) == kind
    assert g2p.token_class(token, lang) == g2p.KIND_CLASS[kind]
    assert g2p.is_vowel(token, lang) == (kind == "vowel")


def test_four_way_classes() -> None:
    assert set(g2p.KIND_CLASS.values()) == {"vowel", "consonant", "glide", "zero"}
    assert g2p.token_class("ㅇ", "ko") == "consonant" and g2p.token_class("ˈ", "en") == "zero"


def test_unknown_tokens_do_not_crash(unknown_guard: None) -> None:
    g2p.UNKNOWN_TOKENS.clear()
    assert g2p.token_kind("ᄏ", "ko") == "unknown"        # 조합형 자모는 G2P 가 내지 않는다(호환 자모로 바꿈)
    assert g2p.token_class("A", "en") == "zero"           # espeak 는 대문자를 내지 않는다
    assert not g2p.is_vowel("ab", "en")                    # 여러 글자 토큰
    assert g2p.UNKNOWN_TOKENS == {("ko", "ᄏ"), ("en", "A"), ("en", "ab")}


def test_check_lang() -> None:
    assert g2p.check_lang(" EN ") == "en"
    with pytest.raises(ValueError):
        g2p.check_lang("ja")
    with pytest.raises(ValueError):
        g2p.tokens("bonjour", "fr")


def test_missing_cmudict_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """cmudict 가 없으면 g2pK 가 몰래 내려받기 전에 멈추고 받는 법을 알린다."""
    import nltk

    def not_found(name: str, *args: object, **kwargs: object) -> str:
        raise LookupError(name)

    monkeypatch.setattr(g2p, "_KO_G2P", None)
    monkeypatch.setattr(nltk.data, "find", not_found)
    with pytest.raises(RuntimeError, match="cmudict"):
        g2p._ko_g2p()


def test_versions() -> None:
    v = g2p.versions()
    assert v["phonemizer"] == "3.6.0" and v["g2pK"] == "0.9.4" and v["espeak"] == "1.52.0"
    assert v["python-mecab-ko"] and v["jamo"] and v["nltk"]


# ---------------------------------------------------------------------------- G2P 설정 파일(g2p.json)
def test_settings_describe_token_form() -> None:
    s = g2p.settings()
    assert s["scheme"] == g2p.SCHEME == "pstts-paper-g2p/2" and s["blank"] == BLANK
    assert s["en"]["voice"] == "en-gb" and s["en"]["with_stress"] is False and s["en"]["preserve_punctuation"] is True
    assert "호환 자모" in s["ko"]["jamo"]
    assert g2p.settings("EN-US", True)["en"]["voice"] == "en-us"


def test_settings_file_round_trip_and_check(tmp_path: Path) -> None:
    path = tmp_path / g2p.SETTINGS_NAME
    doc = g2p.write_settings(path)
    assert g2p.read_settings(path) == doc and doc["versions"]["g2pK"] == "0.9.4"
    g2p.check_settings(doc)                                              # 같으면 조용히 지나간다
    with pytest.raises(ValueError, match="--g2p-en-voice en-gb") as e:
        g2p.check_settings(doc, "en-us", False)
    assert "en" in str(e.value)
    us = g2p.write_settings(tmp_path / "us.json", "en-us", True)
    with pytest.raises(ValueError, match="--g2p-en-voice en-us --g2p-en-stress"):
        g2p.check_settings(us)
    old = {**doc, "scheme": "pstts-paper-g2p/1"}                         # 조합형 자모로 학습한 옛 토큰 꼴
    with pytest.raises(ValueError, match="scheme"):
        g2p.check_settings(old)
    (tmp_path / "bad.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError):
        g2p.read_settings(tmp_path / "bad.json")


# ---------------------------------------------------------------------------- 표본 말뭉치
@pytest.fixture(scope="module")
def corpus_tokens() -> dict[str, list[list[str]]]:
    rows = sample_rows()
    return {"en": [g2p.tokens(r[0], "en") for r in rows], "ko": [g2p.tokens(r[1], "ko") for r in rows]}


def test_corpus_every_token_classified(corpus_tokens: dict[str, list[list[str]]], unknown_guard: None) -> None:
    """표본 영어 2,000줄, 한국어 2,000줄에서 나온 토큰이 모두 모음, 자음, 글라이드, 0길이 가운데 하나로 갈린다."""
    g2p.UNKNOWN_TOKENS.clear()
    kinds: dict[str, dict[str, str]] = {}
    for lang, seqs in corpus_tokens.items():
        assert len(seqs) == N_SAMPLE
        kinds[lang] = {tok: g2p.token_kind(tok, lang) for seq in seqs for tok in seq}
    assert g2p.UNKNOWN_TOKENS == set()
    for lang, inv in kinds.items():
        assert "unknown" not in inv.values()
        assert {tok for tok, k in inv.items() if k == "vowel"} <= (g2p.EN_VOWEL_CHARS if lang == "en"
                                                                   else g2p.KO_VOWEL_JAMO)
    # 2026-11-22 표본에서 나온 모음: 영어 14종 = 논문 Fig. 5(c) 의 영어 축, 한국어 21종 = Fig. 5(c) 의 한국어 축
    assert {t for t, k in kinds["en"].items() if k == "vowel"} == FIG5C_EN
    assert {t for t, k in kinds["ko"].items() if k == "vowel"} == FIG5C_KO
    assert {t for t, k in kinds["en"].items() if k == "glide"} == {"j", "w"}
    assert {t for t, k in kinds["ko"].items() if k == "consonant"} <= g2p.KO_CONSONANT_JAMO
    assert {k for k in kinds["ko"].values()} == {"vowel", "consonant", "blank", "space", "punct", "foreign"}
    assert "ˈ" not in kinds["en"] and "ˌ" not in kinds["en"]


def test_corpus_en_us_vowels_differ_from_fig5c() -> None:
    """en-us 로 바꾸면 Fig. 5(c) 에 없는 모음(o, æ, ɚ, ᵻ)이 나오고 ɒ 는 안 나온다(목소리를 en-gb 로 둔 근거)."""
    rows = sample_rows()[:400]
    vowels = {t for r in rows for t in g2p.phonemes(r[0], "en", en_voice="en-us") if t in g2p.EN_VOWEL_CHARS}
    assert {"o", "æ", "ɚ", "ᵻ"} <= vowels - FIG5C_EN and "ɒ" not in vowels


def test_corpus_blank_structure(corpus_tokens: dict[str, list[list[str]]]) -> None:
    for seqs in corpus_tokens.values():
        for seq in seqs:
            assert seq[0::2] == [BLANK] * (len(seq) // 2 + 1)
            assert BLANK not in seq[1::2]
            inner = seq[1::2]
            assert not inner or (inner[0] != SPACE and inner[-1] != SPACE)
