"""TTS 프런트엔드: 음소 토큰(S_b, T_b), 토큰별 길이, 모음표 라벨, DTW 국소 비용.

논문은 이 값들을 학습한 Your-TTS 에서 얻는다(길이 = 길이 예측기 [39], 모음 거리 = 식 8, 9).
그 모델이 아직 없어서 지금은 근사(ApproxFrontend)를 쓰고, 학습하면 YourTTSFrontend(yourtts.py)로 바꾼다.
두 프런트엔드는 같은 Frontend 인터페이스를 따른다. 토큰은 둘 다 g2p.tokens 꼴이다(영어 목소리와 강세표는 PaperConfig).

근사가 논문과 다른 점(README_paper 의 TTS 근사 묶음):
- 문장 길이(초) = 음절 수 / 음절 속도. 한국어 속도 = en_rate x coef. 값은 기존 판 Config 기본값이다.
- 영어 음절 수는 en_rate 를 잰 셈과 같게 센다. phonemizer en-us(APPROX_COUNT_VOICE) 낱말 음소에 기존 판 모음 핵 규칙을 쓴다.
  토큰(무게, 모음 라벨)은 논문 판 G2P(기본 en-gb) 토큰이다. en-gb 음소에는 기존 판 규칙에 없는 이중모음(eə, əʊ, ʊə)과 ɒ 가
  있고, here 의 iə 가 idea 의 iə 와 같은 꼴이라 en-gb 음소로 세면 음절 수가 어긋난다.
- 토큰마다 무게에 비례해 나눈다. 빈 토큰과 공백은 0 이다(논문은 빈 토큰도 길이 예측기에서 길이를 받음).
  한국어 ㅇ 은 다음 토큰이 모음이면 소리 없는 초성이라 0, 아니면 받침(ŋ)이라 자음 무게다(g2p.silent_ieung).
- 모음 거리는 PanPhon 조음 자질 거리표(기존 pstts.vowels)다. 모음 아닌 칸과 모음 사이 비용은 approx_null_cost.
- 영어 모음 글자의 표 기호 대응(EN_LABELS)은 PanPhon 으로 정해 적어 둔 상수다. 실행 중에는 PanPhon 을 부르지 않는다.
  PanPhon 0.22.2 는 자료 파일을 인코딩 없이 열어서 UTF-8 모드가 아닌 한국어 Windows(cp949)에서 깨지기 때문이다.
  시험(tests/paper/test_frontend_approx.py)이 상수가 derive_en_label_map() 과 같은지 본다.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

# _en_word_syllables 는 기존 판이 낱말 하나의 espeak 음소를 음절로 묶는 함수다(EN_NUCLEUS, EN_NUCLEUS_BEFORE_R 규칙).
# 규칙을 베끼지 않고 그대로 불러 써서 두 판의 영어 음절 셈이 같게 한다.
from ..phonetics import EN_NUCLEUS, _en_word_syllables, korean_syllables
from ..phonetics import EN_VOICE as APPROX_COUNT_VOICE
from ..vowels import VowelTable, cost_matrix, load_or_build, panphon_vector
from . import g2p
from .config import PaperConfig

# 근사의 프레임(PS 늘리기 단위): 22.05 kHz, hop 256. 논문 3.1절 프레임과 같고, 학습 레시피(명세 3.12)의 Your-TTS 도 같다.
APPROX_FRAME_SEC = 256 / 22050

# 영어 모음 글자(g2p.EN_VOWEL_CHARS 32자) -> 근사 모음표 기호. derive_en_label_map() 결과를 적은 상수다
# (2026-11-22, panphon 0.22.2, pstts/data/vowel_table.json 16기호).
EN_LABELS: dict[str, str] = {
    "a": "a", "e": "e", "i": "i", "o": "o", "u": "u", "y": "u", "æ": "æ", "ø": "e", "œ": "ɛ", "ɐ": "ə", "ɑ": "ɑ",
    "ɒ": "ɑ", "ɔ": "ɔ", "ɘ": "e", "ə": "ə", "ɚ": "ə", "ɛ": "ɛ", "ɜ": "ɜ", "ɝ": "ɜ", "ɞ": "ɛ", "ɤ": "ʌ", "ɨ": "ɨ",
    "ɪ": "ɪ", "ɯ": "ɯ", "ɵ": "e", "ɶ": "æ", "ʉ": "u", "ʊ": "ʊ", "ʌ": "ʌ", "ʏ": "ɪ", "ᵻ": "ɨ", "ᵿ": "u",
}

# 표에 없는 영어 모음 글자 가운데 PanPhon 최근접을 그대로 쓰지 않는 것(derive_en_label_map).
# ɒ: 명세의 대응(ɒ→ɑ). IPA 모음표에서 같은 자리(낮은 뒤 모음)이고 원순만 다르다.
#    PanPhon 거리로는 a, o(2.0)가 ɑ(2.236)보다 가깝게 나와서 따로 적는다.
_EN_LABEL_FIXED = {"ɒ": "ɑ"}
# ᵿ: PanPhon 에 없는 글자. pstts.phonetics 가 ᵻ 를 ɨ 로 읽듯 가로줄 글자를 가운데 모음으로 읽어 ʉ 로 보고,
#    ʉ 의 PanPhon 최근접 표 기호를 쓴다.
_EN_LABEL_VIA = {"ᵿ": "ʉ"}


@runtime_checkable
class Frontend(Protocol):
    """ISO, PS 가 쓰는 TTS 값. src = 영어 원문, tgt = 한국어 후보."""

    name: str           # 'approx' | 'yourtts'
    frame_sec: float    # PS 늘리기의 프레임 길이(초)

    def tokens(self, text: str, lang: str) -> list[str]:
        """빈 토큰을 끼운 토큰 열(S_b, T_b)."""

    def durations(self, seqs: list[list[str]], lang: str) -> list[np.ndarray]:
        """열마다 토큰별 길이(초, 실수). 배열 길이 = 토큰 수."""

    def labels(self, toks: list[str], lang: str) -> list[str | None]:
        """토큰마다 모음표 기호 또는 None(null: 자음, 빈 토큰, 공백, 문장부호, 강세표 등)."""

    def cost(self, src_labels: Sequence[str | None], tgt_labels: Sequence[str | None]) -> np.ndarray:
        """(len(src), len(tgt)) 국소 비용 행렬."""

    def info(self) -> dict:
        """run.json 에 남길 설정과 판(JSON 으로 쓸 수 있는 값만)."""

    def close(self) -> None:
        """올린 것을 내린다."""


# ---------------------------------------------------------------------------- 모음 대응표
@lru_cache(maxsize=1)
def ko_vowel_info() -> dict[str, tuple[str, bool]]:
    """호환 모음 자모마다 (핵 모음 기호, 글라이드가 든 모음인지).

    pstts.phonetics 의 한글 음절 풀이를 그대로 쓴다. '아' 에 중성을 바꿔 끼운 받침 없는 낱말 첫 음절로 묻는다
    (ㅢ 는 낱말 첫 '의' 라 ɰ+i). 호환 모음(U+314F ~ U+3163)과 중성(U+1161 ~ U+1175)은 순서가 같다.
    결과: ㅏㅑㅘ a, ㅐㅒㅙ ɛ, ㅓㅕㅝ ʌ, ㅔㅖㅚㅞ e, ㅗㅛ o, ㅜㅠ u, ㅡ ɯ, ㅣㅟㅢ i.
    """
    base = ord("아")
    out: dict[str, tuple[str, bool]] = {}
    for j in range(21):
        (syl,) = korean_syllables(chr(base + j * 28))
        out[chr(0x314F + j)] = (syl.nucleus[0], syl.glide is not None)
    return out


def _nearest(symbol: str, table: VowelTable) -> str:
    """PanPhon 자질 벡터의 유클리드 거리로 가장 가까운 표 기호. 거리가 같으면 표 순서(혀 앞, 높음 먼저)."""
    v = np.asarray(panphon_vector(symbol), dtype=np.float64)
    dist = [float(np.sqrt(((v - np.asarray(panphon_vector(s), dtype=np.float64)) ** 2).sum())) for s in table.symbols]
    return table.symbols[int(np.argmin(dist))]


def derive_en_label_map(table: VowelTable | None = None) -> dict[str, str]:
    """EN_LABELS 를 만든 규칙(PanPhon 을 부르므로 UTF-8 모드에서만 돈다. 시험과 상수를 고칠 때만 쓴다).

    1) pstts.phonetics.EN_NUCLEUS 에 한 글자 토큰으로 있으면 그 기호(ɚ→ə, ɐ→ə, ᵻ→ɨ, ɝ→ɜ 등 기존 판과 같음).
    2) 표에 그대로 있으면 자기 자신(ɨ, ɯ).
    3) ɒ→ɑ(명세), ᵿ 는 ʉ 를 거쳐.
    4) 나머지는 PanPhon 최근접 표 기호.
    """
    table = table or load_or_build()
    out: dict[str, str] = {}
    for ch in sorted(g2p.EN_VOWEL_CHARS):
        ent = EN_NUCLEUS.get(ch)
        if ent is not None and len(ent.nuclei) == 1 and len(ent.nuclei[0]) == 1:
            out[ch] = ent.nuclei[0][0]
        elif ch in table.index:
            out[ch] = ch
        elif ch in _EN_LABEL_FIXED:
            out[ch] = _EN_LABEL_FIXED[ch]
        else:
            out[ch] = _nearest(_EN_LABEL_VIA.get(ch, ch), table)
    return out


@lru_cache(maxsize=1)
def en_label_map() -> dict[str, str]:
    """영어 모음 글자 -> 모음표 기호(EN_LABELS). 모음 글자가 모두 있고 기호가 모두 표에 있는지 확인한다."""
    table = load_or_build()
    lacking = sorted(g2p.EN_VOWEL_CHARS - set(EN_LABELS))
    if lacking:
        raise ValueError(f"EN_LABELS 에 없는 영어 모음 글자가 있습니다: {lacking}")
    missing = sorted(s for s in EN_LABELS.values() if s not in table.index)
    if missing:
        raise ValueError(f"모음표에 없는 기호로 대응됐습니다: {missing}")
    return dict(EN_LABELS)


@lru_cache(maxsize=1)
def ko_label_map() -> dict[str, str]:
    """한국어 호환 모음 자모 -> 모음표 기호(핵 모음)."""
    return {jamo: nucleus for jamo, (nucleus, _) in ko_vowel_info().items()}


def table_sha1(table: VowelTable) -> str:
    """모음표 기호와 거리 행렬(소수 여섯째 자리)의 sha1. run.json 에서 어떤 표를 썼는지 가리는 데 쓴다."""
    doc = {"symbols": list(table.symbols), "matrix": [[round(float(v), 6) for v in row] for row in table.matrix]}
    return hashlib.sha1(json.dumps(doc, ensure_ascii=False).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------- 근사
class ApproxFrontend:
    """TTS 근사(지금 기본). 모델을 올리지 않아서 장치를 쓰지 않는다.

    영어 음절 수는 phonemizer en-us 의 낱말별 음소에 기존 판의 모음 핵 규칙(pstts.phonetics)을 써서 센다. 그 음소 묶음은
    tokens() 에서 기억해 두므로 영어 열의 길이는 이 프런트엔드의 tokens() 로 만든 열에만 잴 수 있다.
    모음표와 모음 대응은 만들 때 읽어 둔다. 문제가 있으면 ISO, PS 에서 모델 2 를 부르기 전에 멈춘다.
    """

    name = "approx"
    frame_sec = APPROX_FRAME_SEC

    def __init__(self, cfg: PaperConfig) -> None:
        if cfg.en_rate <= 0 or cfg.coef <= 0:
            raise ValueError(f"음절 속도는 0보다 커야 합니다: en_rate={cfg.en_rate}, coef={cfg.coef}")
        if cfg.vowel_weight <= 0 or cfg.cons_weight < 0 or cfg.glide_weight < 0:
            raise ValueError("모음 무게는 0보다 크고, 자음·글라이드 무게는 0 이상이어야 합니다.")
        if cfg.approx_null_cost < 0:
            raise ValueError(f"null 비용은 0 이상이어야 합니다: {cfg.approx_null_cost}")
        self.cfg = cfg
        self.en_voice = g2p.check_en_voice(cfg.g2p_en_voice)
        self.en_stress = bool(cfg.g2p_en_stress)
        self._table = load_or_build()
        self._en_labels = en_label_map()
        self._ko_labels = ko_label_map()
        # 토큰 IPA 글 -> 음절을 셀 낱말별 음소(en-us). tokens() 가 채우고 durations() 가 읽는다.
        self._en_words: dict[str, tuple[tuple[str, ...], ...]] = {}

    # ------------------------------------------------------------------ 토큰
    def tokens(self, text: str, lang: str) -> list[str]:
        """g2p.tokens(text, lang) 와 같은 열(영어는 cfg 의 목소리와 강세표). 영어는 음절 세기용 en-us 음소도 기억한다."""
        if g2p.check_lang(lang) == "en":
            ph = g2p.en_phonemize(text, self.en_voice, self.en_stress)
            self._en_words[ph.ipa] = g2p.en_phonemize(text, APPROX_COUNT_VOICE, False).words
            return g2p.with_blanks(list(ph.ipa))
        return g2p.tokens(text, "ko")

    # ------------------------------------------------------------------ 길이
    def rate(self, lang: str) -> float:
        """음절 속도(음절/초). 한국어 = en_rate x coef."""
        return self.cfg.en_rate if g2p.check_lang(lang) == "en" else self.cfg.ko_rate

    def syllables(self, seq: Sequence[str], lang: str) -> int:
        """음절 수. 영어는 en-us 음소에 pstts.phonetics 의 모음 핵 규칙, 한국어는 모음 자모 토큰 수(= g2pK 발음의 음절 수)."""
        if g2p.check_lang(lang) == "ko":
            return sum(1 for t in seq if t in g2p.KO_VOWEL_JAMO)
        ipa = "".join(t for t in seq if t != g2p.BLANK)
        words = self._en_words.get(ipa)
        if words is None:
            raise ValueError("이 영어 토큰 열의 낱말별 음소를 모릅니다. 같은 ApproxFrontend 의 tokens() 로 만든 열을 주세요: "
                             f"{ipa[:60]!r}")
        return sum(len(_en_word_syllables(w)) for w in words)

    def weights(self, seq: Sequence[str], lang: str) -> np.ndarray:
        """토큰별 무게. 영어: 모음 글자 vowel_weight, 자음 cons_weight, 글라이드 glide_weight.
        한국어: 홑모음 vowel_weight, 글라이드가 든 모음 vowel_weight + glide_weight, 자음 cons_weight.
        빈 토큰, 공백, 문장부호, 강세표·길이표, 소리 없는 초성 ㅇ(다음 토큰이 모음), 남은 외국 글자, 모르는 토큰은 0."""
        code = g2p.check_lang(lang)
        cfg = self.cfg
        info = ko_vowel_info()
        silent = g2p.silent_ieung(seq) if code == "ko" else [False] * len(seq)
        out = np.zeros(len(seq), dtype=np.float64)
        for i, tok in enumerate(seq):
            cls = g2p.token_class(tok, code)
            if cls == g2p.VOWEL:
                glide_vowel = code == "ko" and info[tok][1]
                out[i] = cfg.vowel_weight + (cfg.glide_weight if glide_vowel else 0.0)
            elif cls == g2p.CONSONANT and not silent[i]:
                out[i] = cfg.cons_weight
            elif cls == g2p.GLIDE:
                out[i] = cfg.glide_weight
        return out

    def durations(self, seqs: list[list[str]], lang: str) -> list[np.ndarray]:
        """열마다 토큰별 길이(초). 합 = 음절 수 / 음절 속도, 토큰 몫은 무게에 비례."""
        rate = self.rate(lang)
        out: list[np.ndarray] = []
        for seq in seqs:
            w = self.weights(seq, lang)
            n = self.syllables(seq, lang)
            total = float(w.sum())
            if n == 0 or total <= 0:
                out.append(np.zeros(len(seq), dtype=np.float64))
            else:
                out.append(w * (n / rate / total))
        return out

    # ------------------------------------------------------------------ 모음표
    @property
    def table(self) -> VowelTable:
        """기존 판의 PanPhon 모음 거리표(EN 15, KO 8, 합쳐 16 기호)."""
        return self._table

    def labels(self, toks: list[str], lang: str) -> list[str | None]:
        """모음 토큰은 모음표 기호로, 나머지는 None(null)."""
        code = g2p.check_lang(lang)
        mapping = self._en_labels if code == "en" else self._ko_labels
        return [mapping[t] if g2p.is_vowel(t, code) else None for t in toks]

    def cost(self, src_labels: Sequence[str | None], tgt_labels: Sequence[str | None]) -> np.ndarray:
        """pstts.vowels.cost_matrix. null 끼리 0, null 과 모음 사이 approx_null_cost, 모음끼리는 표 거리."""
        return cost_matrix(src_labels, tgt_labels, self._table, null_cost=self.cfg.approx_null_cost)

    # ------------------------------------------------------------------ 기록
    def info(self) -> dict:
        cfg = self.cfg
        table = self._table
        return {
            "name": self.name,
            "frame_sec": self.frame_sec,
            "en_rate": cfg.en_rate,
            "coef": cfg.coef,
            "ko_rate": cfg.ko_rate,
            "weights": {
                "vowel": cfg.vowel_weight,
                "ko_glide_vowel": cfg.vowel_weight + cfg.glide_weight,
                "consonant": cfg.cons_weight,
                "glide": cfg.glide_weight,
                "blank_space_punct_marks_silent_ieung": 0.0,
            },
            "en_syllable_count": f"phonemizer {APPROX_COUNT_VOICE} 낱말 음소 + pstts.phonetics 모음 핵 규칙(en_rate 를 잰 셈)",
            "null_cost": cfg.approx_null_cost,
            "vowel_table": {"method": table.method, "symbols": list(table.symbols), "sha1": table_sha1(table)},
            "en_labels": dict(self._en_labels),
            "ko_labels": dict(self._ko_labels),
            "g2p": {**g2p.settings(self.en_voice, self.en_stress), "versions": g2p.versions()},
        }

    def close(self) -> None:
        """기억한 영어 음소 묶음을 비운다. G2P 엔진은 모듈에서 함께 쓰므로 그대로 둔다."""
        self._en_words.clear()


def make_frontend(cfg: PaperConfig, device: str) -> Frontend:
    """cfg.tts 에 맞는 프런트엔드. yourtts 모듈(torch, coqui-tts)은 그것을 고를 때만 불러온다."""
    kind = str(cfg.tts).strip().lower()
    if kind == "approx":
        return ApproxFrontend(cfg)
    if kind == "yourtts":
        from .yourtts import YourTTSFrontend

        return YourTTSFrontend(cfg, device)
    raise ValueError(f"TTS 프런트엔드 이름이 잘못됐습니다: {cfg.tts} (approx, yourtts 중 하나)")
