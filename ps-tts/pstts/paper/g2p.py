"""논문 3.1절 음소화와 토큰화(G2P).

- 영어: phonemizer [33] 의 espeak 백엔드. 기본은 목소리 en-gb, 강세표 없음(EN_VOICE, EN_WITH_STRESS).
  - 논문에는 설정이 없다. 두 그림에서 읽은 것이다(해석, README_paper 의 (나) 표).
  - Fig. 5(c) 의 영어 모음 14개(a e i u ɐ ɑ ɒ ɔ ə ɛ ɜ ɪ ʊ ʌ)는 espeak en-gb 가 내는 모음과 똑같다.
    en-us 는 o, æ, ɚ, ᵻ 가 더 나오고 ɒ 가 없다(enko-nmt valid 영어 2,000줄로 잼).
  - Fig. 4 의 hello 토큰(h ə l o ʊ)에는 강세표가 없다. phonemizer 기본값도 강세표 없음이다.
  - Fig. 4 의 o 는 en-us 꼴(en-gb 는 h ə l ə ʊ)이라 두 그림이 어긋난다. 학습한 모델의 글자 목록인 Fig. 5(c) 를 따른다.
  - PaperConfig.g2p_en_voice, g2p_en_stress(명령줄 --g2p-en-voice, --g2p-en-stress)로 바꿀 수 있다.
  - 문장부호는 남긴다. 논문에 없고 그림으로도 알 수 없어서 coqui-tts 의 espeak 음소화기 기본값(keep_puncs)을 따른다(해석).
  - espeak 라이브러리와 발음 자료는 espeakng_loader 가 준 것을 쓴다.
- 한국어: g2pK [32] 로 발음 한글을 얻고, jamo.h2j 로 자모로 푼 뒤 jamo.j2hcj 로 호환 자모(ㄱ U+3131 ~ ㅣ U+3163)로 바꾼다.
  - Fig. 4 처럼 초성과 받침이 같은 토큰이다. 안녕은 ㅇ ㅏ ㄴ ㄴ ㅕ ㅇ 이고 두 ㅇ, 두 ㄴ 이 같은 번호(24, 1)를 받는다.
  - 소리 없는 초성 ㅇ 과 받침 ㅇ(ŋ)도 같은 토큰이다. 근사(frontend.ApproxFrontend)의 길이 무게만 자리로 가른다(silent_ieung).
- 토큰은 글자 하나 단위다. Coqui TTS 가 IPA 문자열을 글자로 잘라 id 로 바꾸는 것과 같다. 낱말 사이 공백도 토큰 하나다.
- 빈 토큰 b 는 Coqui TTSTokenizer.intersperse_blank_char 와 같은 꼴로 처음, 사이, 끝에 끼운다(S_b, T_b).

토큰은 모음, 자음, 글라이드, 0길이 가운데 하나로 나뉜다. 0길이는 빈 토큰, 공백, 문장부호와 기호, 강세표·길이표·결합
부호, g2pK 가 한글로 바꾸지 못하고 남긴 글자(사전에 없는 로마자 약어, 그리스 문자, 한자)다.
그 밖의 토큰은 멈추지 않고 0길이로 두며 UNKNOWN_TOKENS 에 (언어, 토큰) 으로 남긴다.

학습(scripts/yourtts/phonemize_manifest.py)과 추론(yourtts.YourTTSFrontend)이 같은 G2P 를 쓰도록 settings() 를
g2p.json(SETTINGS_NAME)에 적고 체크포인트 폴더에서 확인한다(check_settings).

espeak 는 스레드에 안전하지 않고 g2pK(mecab)도 같이 부르면 안 되어서 언어마다 잠금 하나를 둔다.
phonemizer 는 espeak 라이브러리를 임시 폴더에 복사해 따로 올리므로 pstts.phonetics 의 eSpeak(원본 DLL)와 상태를 나누지 않는다.
"""
from __future__ import annotations

import json
import os
import threading
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple, Sequence

BLANK = "<b>"       # 우리 토큰 목록 안의 빈 토큰 b. 실제 글자와 겹치지 않게 여러 글자로 둔다
SPACE = " "         # 낱말 사이 공백 토큰
LANGS = ("en", "ko")

# 토큰 꼴의 판. 토큰 꼴을 바꾸면 올린다(g2p.json 에 적혀 학습한 모델과 추론의 G2P 가 같은지 가린다).
# 2: 영어 en-gb·강세표 없음, 한국어 호환 자모(Fig. 4, 5c). 1 은 en-us·강세표·조합형 자모였다.
SCHEME = "pstts-paper-g2p/2"
SETTINGS_NAME = "g2p.json"

EN_VOICE = "en-gb"          # Fig. 5(c) 의 영어 모음 14개 = espeak en-gb 모음
EN_WITH_STRESS = False      # Fig. 4 에 강세표 없음(phonemizer 기본값)
# 목소리와 강세 말고 EspeakBackend 에 늘 주는 인자
EN_BACKEND_OPTIONS: dict[str, Any] = {
    "preserve_punctuation": True,
    "tie": False,
    "language_switch": "remove-flags",
    "words_mismatch": "ignore",
}
# 낱말별 음소(phone) 목록을 같은 호출에서 얻으려고 phone 사이에 넣는 구분자. espeak IPA 출력에는 없는 글자다.
# phonemizer 는 espeak 가 준 '_' 구분자를 이 글자로 바꿀 뿐이라 이 글자를 지우면 구분자 없이 부른 결과와 같다.
_PHONE_SEP = "|"

# ---------------------------------------------------------------------------- 영어 토큰 분류
# IPA 모음 글자. 명세의 목록에 ᵿ(가운데 원순 약모음. pstts.phonetics 의 모음 글자 목록에도 있다)를 더했다.
EN_VOWEL_CHARS: frozenset[str] = frozenset("iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻɚɝᵿ")
# 글라이드(반모음, IPA 접근음). 영어의 j, w 와 다른 말 낱말(언어 바뀜)에서 들어오는 ɥ, ɰ.
EN_GLIDE_CHARS: frozenset[str] = frozenset("jwɥɰ")
# IPA 자음 글자: 폐 자음, 비폐 자음, 그 밖의 기호, 붙여 쓴 파찰음 글자. g 는 ɡ 와 같은 소리로 IPA 가 둘 다 허용한다.
EN_CONSONANT_CHARS: frozenset[str] = frozenset(
    "pbtdʈɖcɟkgɡqɢʔʡmɱnɳɲŋɴʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻlɭʎʟ"
    "ʘǀǃǂǁɓɗʄɠʛ"
    "ʍʜʢɕʑɺɧɫ"
    "ʦʣʧʤʨʥ"
)
# 강세표, 길이표, 이음표. 유니코드 수식 글자(Lm, Sk)와 결합 부호(M*)도 같은 0길이 표시로 본다.
EN_MARK_CHARS: frozenset[str] = frozenset("ˈˌːˑ͜͡")

# ---------------------------------------------------------------------------- 한국어 토큰 분류
KO_CONSONANT_JAMO: frozenset[str] = frozenset(chr(c) for c in range(0x3131, 0x314F))  # 호환 자음 30자(겹받침 포함)
KO_VOWEL_JAMO: frozenset[str] = frozenset(chr(c) for c in range(0x314F, 0x3164))      # 호환 모음 21자(합친 모음도 하나)
KO_IEUNG = "ㅇ"       # 초성이면 소리 없음, 받침이면 ŋ. 토큰은 하나다(Fig. 4)
# 아래아(ᆞ U+119E, ㆍ U+318D)는 한국어 글에서 가운뎃점으로 쓰인다(표본의 '유산ᆞ사산', '질병ᆞ사고'). 문장부호로 본다.
# j2hcj 가 ᆞ 를 ㆍ 로 바꾸므로 토큰에는 ㆍ 만 나온다.
KO_INTERPUNCT: frozenset[str] = frozenset("ᆞㆍ")
# 한글 글자 영역(조합형 자모, 호환 자모, 확장 자모, 음절, 반각 자모). 여기 있는 글자는 '남은 외국 글자'가 아니다.
_HANGUL_BLOCKS = ((0x1100, 0x11FF), (0x3130, 0x318F), (0xA960, 0xA97F), (0xAC00, 0xD7A3),
                  (0xD7B0, 0xD7FF), (0xFFA0, 0xFFDC))

# ---------------------------------------------------------------------------- 분류 이름
VOWEL, CONSONANT, GLIDE, ZERO = "vowel", "consonant", "glide", "zero"
# 세부 종류 -> 네 갈래 분류. 모음·자음·글라이드 말고는 모두 0길이다.
KIND_CLASS: dict[str, str] = {
    "vowel": VOWEL, "consonant": CONSONANT, "glide": GLIDE,
    "blank": ZERO,      # 빈 토큰 b
    "space": ZERO,      # 낱말 사이 공백
    "punct": ZERO,      # 문장부호와 기호
    "mark": ZERO,       # 강세표, 길이표, 결합 부호
    "foreign": ZERO,    # g2pK 가 한글로 바꾸지 못한 글자(로마자, 그리스 문자, 한자 등)
    "unknown": ZERO,    # 위 어디에도 들지 않는 토큰. UNKNOWN_TOKENS 에 남는다
}

# 실행 중 분류하지 못한 토큰 (언어, 토큰). 보고용으로 모은다.
UNKNOWN_TOKENS: set[tuple[str, str]] = set()

_EN_LOCK = threading.RLock()
_KO_LOCK = threading.RLock()
_EN_BACKENDS: dict[tuple[str, bool], Any] = {}
_ESPEAK_READY = False
_KO_G2P: Any = None


# ---------------------------------------------------------------------------- 공통
def check_lang(lang: str) -> str:
    """언어 코드를 확인해 'en' 또는 'ko' 로 돌려준다."""
    code = str(lang).strip().lower()
    if code not in LANGS:
        raise ValueError(f"언어는 en 또는 ko 여야 합니다: {lang}")
    return code


def with_blanks(tokens: Sequence[str]) -> list[str]:
    """[b, t1, b, t2, ..., b, tN, b]. Coqui TTSTokenizer.intersperse_blank_char 와 같은 꼴이다(빈 목록이면 [b])."""
    out = [BLANK] * (len(tokens) * 2 + 1)
    out[1::2] = list(tokens)
    return out


def _normalize(text: str) -> str:
    """NFC 로 맞추고 공백 줄을 칸 하나로 줄인다(앞뒤 공백은 없앰)."""
    return " ".join(unicodedata.normalize("NFC", str(text)).split())


def phonemes(text: str, lang: str, *, en_voice: str = EN_VOICE, en_stress: bool = EN_WITH_STRESS) -> list[str]:
    """빈 토큰 없는 음소 토큰(글자 하나 단위, 낱말 사이 공백 포함). en_voice, en_stress 는 영어에만 쓴다."""
    if check_lang(lang) == "en":
        return list(en_phonemize(text, en_voice, en_stress).ipa)
    from jamo import h2j, j2hcj

    return list(j2hcj(h2j(ko_pronounce(text))))


def tokens(text: str, lang: str, *, en_voice: str = EN_VOICE, en_stress: bool = EN_WITH_STRESS) -> list[str]:
    """빈 토큰을 끼운 토큰 열(S_b, T_b) = with_blanks(phonemes(text, lang))."""
    return with_blanks(phonemes(text, lang, en_voice=en_voice, en_stress=en_stress))


# ---------------------------------------------------------------------------- 영어
class EnPhonemes(NamedTuple):
    """영어 한 문장의 음소화 결과(phonemizer 한 번 부른 것)."""
    ipa: str                                # 토큰이 되는 IPA 글. 낱말 사이 ' ', 문장부호(강세를 켜면 강세표도) 포함
    words: tuple[tuple[str, ...], ...]      # 낱말별 espeak 음소. 강세표·문장부호를 빼고 길이표는 둔다


def _espeak_ready() -> None:
    """phonemizer 가 espeakng_loader 의 espeak 라이브러리와 발음 자료를 쓰게 한 번만 알려 준다."""
    global _ESPEAK_READY
    with _EN_LOCK:
        if not _ESPEAK_READY:
            import espeakng_loader
            from phonemizer.backend.espeak.wrapper import EspeakWrapper

            EspeakWrapper.set_library(espeakng_loader.get_library_path())
            EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
            _ESPEAK_READY = True


@lru_cache(maxsize=64)
def check_en_voice(voice: str) -> str:
    """espeak 영어 목소리 이름(en-gb, en-us 등)을 확인해 소문자로 돌려준다. 없는 이름이면 한국어 ValueError."""
    code = str(voice).strip().lower()
    if not code.startswith("en"):
        raise ValueError(f"영어 목소리는 en 으로 시작해야 합니다(예: en-gb, en-us): {voice}")
    _espeak_ready()
    from phonemizer.backend import EspeakBackend

    with _EN_LOCK:
        known = sorted(k for k in EspeakBackend.supported_languages() if k.startswith("en"))
    if code not in known:
        raise ValueError(f"espeak 에 없는 영어 목소리입니다: {voice} (있는 것: {', '.join(known)})")
    return code


def _en_backend(voice: str = EN_VOICE, with_stress: bool = EN_WITH_STRESS) -> Any:
    """(목소리, 강세표) 마다 phonemizer EspeakBackend 를 한 번만 만든다."""
    key = (check_en_voice(voice), bool(with_stress))
    with _EN_LOCK:
        backend = _EN_BACKENDS.get(key)
        if backend is None:
            from phonemizer.backend import EspeakBackend
            from phonemizer.logger import get_logger

            # 언어 바뀜 표시를 지운 안내 같은 phonemizer 기록은 stderr 에 내지 않는다
            backend = EspeakBackend(key[0], logger=get_logger("quiet"), with_stress=key[1], **EN_BACKEND_OPTIONS)
            _EN_BACKENDS[key] = backend
        return backend


_STRESS_MARKS = str.maketrans("", "", "ˈˌ")


def _word_phones(word: str) -> tuple[str, ...]:
    """'|' 로 나뉜 낱말 하나를 음소 목록으로. pstts.phonetics.pronunciation 처럼 강세표는 빼고 길이표는 둔다.

    phonemizer 가 되돌려 붙인 문장부호(쉼표, 따옴표, 괄호 등)는 빼고 음소 경계로 본다. 낱말 안의 문장부호
    (U.S. 의 마침표)는 앞뒤 조각을 따로 음소화해 붙인 자리라 그 사이에 '|' 가 없다. 남는 것이 없는 조각은 버린다.
    """
    out: list[str] = []
    buf: list[str] = []
    for ch in word.translate(_STRESS_MARKS) + _PHONE_SEP:
        if ch == _PHONE_SEP or unicodedata.category(ch)[0] in "PSZ":
            if buf:
                out.append("".join(buf))
                buf = []
        else:
            buf.append(ch)
    return tuple(out)


@lru_cache(maxsize=16384)
def en_phonemize(text: str, voice: str = EN_VOICE, with_stress: bool = EN_WITH_STRESS) -> EnPhonemes:
    """영어 글을 phonemizer(espeak, 목소리 voice)로 한 번 바꿔 IPA 글과 낱말별 음소 목록을 함께 돌려준다.

    숫자 읽기 보정 같은 앞처리는 하지 않는다(espeak 가 읽은 그대로). 공백만 칸 하나로 줄인다.
    낱말별 음소는 강세표를 빼므로 with_stress 와 상관없이 같다(enko-nmt valid 영어 2,000줄 en-us 로 확인).
    """
    clean = _normalize(text)
    if not clean:
        return EnPhonemes("", ())
    from phonemizer.separator import Separator

    separator = Separator(phone=_PHONE_SEP, word=" ", syllable=None)
    backend = _en_backend(voice, with_stress)
    with _EN_LOCK:
        raw = backend.phonemize([clean], separator=separator, strip=True)[0]
    raw = " ".join(raw.split())
    words = tuple(w for w in (_word_phones(word) for word in raw.split(" ")) if w)
    return EnPhonemes(raw.replace(_PHONE_SEP, ""), words)


def en_phones(text: str, voice: str = EN_VOICE) -> list[list[str]]:
    """낱말별 phone 목록(음절 수 세기용). 모양은 pstts.phonetics.pronunciation 과 같다."""
    return [list(w) for w in en_phonemize(text, voice, False).words]


# ---------------------------------------------------------------------------- 한국어
def _ko_g2p() -> Any:
    """g2pK G2p 를 한 번만 만든다. g2pK 는 처음 불러올 때 nltk cmudict 가 없으면 몰래 내려받으므로 먼저 확인한다."""
    global _KO_G2P
    with _KO_LOCK:
        if _KO_G2P is None:
            import nltk

            try:
                nltk.data.find("corpora/cmudict.zip")
            except LookupError:
                raise RuntimeError(
                    "g2pK 가 쓰는 nltk cmudict 가 없습니다. setup_paper.cmd 를 실행하거나 "
                    "python -c \"import nltk, sys, os; nltk.download('cmudict', "
                    "download_dir=os.path.join(sys.prefix, 'nltk_data'))\" 로 받으세요.") from None
            from g2pk import G2p

            _KO_G2P = G2p()
        return _KO_G2P


@lru_cache(maxsize=16384)
def ko_pronounce(text: str) -> str:
    """g2pK 발음(한글 음절 글). 기본 인자로 부르고 공백만 칸 하나로 줄인다."""
    clean = _normalize(text)
    if not clean:
        return ""
    with _KO_LOCK:
        out = _ko_g2p()(clean)
    return " ".join(out.split())


def silent_ieung(seq: Sequence[str]) -> list[bool]:
    """한국어 토큰 열에서 소리 없는 초성 ㅇ 자리. 다음 토큰(빈 토큰은 건너뜀)이 모음이면 초성이라 참, 아니면 받침 ㅇ.

    g2pK 발음을 자모로 풀면 음절마다 초성이 있다(모음으로 시작하는 음절은 ㅇ). 그래서 모음 바로 앞 자음은 늘 초성이고,
    받침 뒤에는 다음 음절의 초성이나 공백, 문장부호, 끝이 온다.
    """
    out = [False] * len(seq)
    nxt: str | None = None          # 오른쪽에서 처음 만난, 빈 토큰이 아닌 토큰
    for i in range(len(seq) - 1, -1, -1):
        tok = seq[i]
        if tok == BLANK:
            continue
        out[i] = tok == KO_IEUNG and nxt is not None and nxt in KO_VOWEL_JAMO
        nxt = tok
    return out


# ---------------------------------------------------------------------------- 토큰 분류
def _is_hangul(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _HANGUL_BLOCKS)


@lru_cache(maxsize=8192)
def _kind(token: str, lang: str) -> str:
    if token == BLANK:
        return "blank"
    if token == SPACE:
        return "space"
    if len(token) != 1:
        return "unknown"
    cat = unicodedata.category(token)
    if lang == "en":
        if token in EN_VOWEL_CHARS:
            return "vowel"
        if token in EN_GLIDE_CHARS:
            return "glide"
        if token in EN_CONSONANT_CHARS:
            return "consonant"
        if token in EN_MARK_CHARS or cat in ("Lm", "Sk") or cat[0] == "M":
            return "mark"
        if cat[0] in "PSZ":
            return "punct"
        return "unknown"
    if token in KO_VOWEL_JAMO:
        return "vowel"
    if token in KO_CONSONANT_JAMO:
        return "consonant"
    if token in KO_INTERPUNCT or cat[0] in "PSZ":
        return "punct"
    if cat[0] == "M":
        return "mark"
    if cat[0] in "LN" and not _is_hangul(token):
        return "foreign"
    return "unknown"


def token_kind(token: str, lang: str) -> str:
    """세부 종류: vowel, consonant, glide, blank, space, punct, mark, foreign, unknown.

    unknown 은 멈추지 않고 UNKNOWN_TOKENS 에 (언어, 토큰) 으로 남긴다. 한국어 ㅇ 은 자리와 상관없이 consonant 다.
    """
    code = check_lang(lang)
    kind = _kind(token, code)
    if kind == "unknown":
        UNKNOWN_TOKENS.add((code, token))
    return kind


def token_class(token: str, lang: str) -> str:
    """네 갈래 분류: 'vowel', 'consonant', 'glide', 'zero'(0길이). 모르는 토큰은 'zero' 다."""
    return KIND_CLASS[token_kind(token, lang)]


def is_vowel(token: str, lang: str) -> bool:
    """영어는 IPA 모음 글자, 한국어는 호환 모음 자모 21자이면 참."""
    return token_kind(token, lang) == "vowel"


# ---------------------------------------------------------------------------- 설정과 판
@lru_cache(maxsize=1)
def _versions() -> tuple[tuple[str, str | None], ...]:
    from importlib.metadata import PackageNotFoundError, version

    out: list[tuple[str, str | None]] = []
    for name in ("phonemizer", "espeakng-loader", "g2pK", "python-mecab-ko", "python-mecab-ko-dic", "jamo", "nltk"):
        try:
            out.append((name, version(name)))
        except PackageNotFoundError:
            out.append((name, None))
    out.append(("espeak", ".".join(str(v) for v in _en_backend().version())))
    return tuple(out)


def versions() -> dict[str, str | None]:
    """G2P 에 쓰는 패키지와 espeak 의 판."""
    return dict(_versions())


def settings(en_voice: str = EN_VOICE, en_stress: bool = EN_WITH_STRESS) -> dict[str, Any]:
    """이 G2P 의 토큰 꼴(JSON 으로 쓸 수 있는 값). 학습과 추론에서 같아야 하는 것만 담는다."""
    return {
        "scheme": SCHEME,
        "blank": BLANK,
        "en": {"backend": "phonemizer espeak", "voice": check_en_voice(en_voice), "with_stress": bool(en_stress),
               **EN_BACKEND_OPTIONS},
        "ko": {"g2p": "g2pK 기본 인자", "jamo": "호환 자모(jamo.j2hcj(jamo.h2j(발음)))"},
    }


def write_settings(path: str | os.PathLike, en_voice: str = EN_VOICE, en_stress: bool = EN_WITH_STRESS) -> dict[str, Any]:
    """settings() 와 패키지 판을 g2p.json 으로 쓴다(phonemize_manifest.py). 쓴 문서를 돌려준다."""
    doc = {**settings(en_voice, en_stress), "versions": versions()}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="")
    os.replace(tmp, p)
    return doc


def read_settings(path: str | os.PathLike) -> dict[str, Any]:
    """g2p.json 을 읽는다."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or "scheme" not in doc:
        raise ValueError(f"{path} 는 G2P 설정 파일(g2p.json)이 아닙니다.")
    return doc


def check_settings(doc: dict[str, Any], en_voice: str = EN_VOICE, en_stress: bool = EN_WITH_STRESS,
                   where: str = SETTINGS_NAME) -> None:
    """g2p.json(학습 때 G2P)이 지금 설정과 같은지 본다. 다르면 무엇이 다른지 적어 한국어 ValueError."""
    want = settings(en_voice, en_stress)
    diffs = [f"{k}: 학습 {doc.get(k)!r}, 지금 {want[k]!r}" for k in ("scheme", "blank", "en", "ko") if doc.get(k) != want[k]]
    if diffs:
        hint = ""
        en = doc.get("en") if isinstance(doc.get("en"), dict) else {}
        if doc.get("scheme") == SCHEME and en.get("voice") and "with_stress" in en:
            stress = " --g2p-en-stress" if en.get("with_stress") else ""
            hint = f" 학습 때와 같게 돌리려면 --g2p-en-voice {en['voice']}{stress} 를 주세요."
        raise ValueError(f"{where} 의 G2P 가 지금 설정과 다릅니다: " + "; ".join(diffs) + "." + hint)
