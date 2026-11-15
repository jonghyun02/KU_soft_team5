"""Korean 3-axis honorifics and 6-speech-level parser & rule-based CAPE converter (Hwang et al., 2021; Jeong et al., 2021)."""

from __future__ import annotations

from functools import lru_cache
import re
from kiwipiepy import Kiwi

from mos_sdh.schemas.document import HonorificAnalysis, SpeechLevelType

_SUBJECT_HONORIFIC_VERBS = {
    "계시",
    "주무시",
    "잡수시",
    "드시",
    "돌아가시",
    "편찮으시",
    "말씀하시",
}

_OBJECT_HONORIFIC_LEMMAS = {
    "드리",
    "올리",
    "여쭈",
    "여쭙",
    "뵙",
    "뵈",
    "모시",
    "말씀드리",
}

_HAPSYO_EF = {
    "ㅂ니다",
    "ᆸ니다",
    "습니다",
    "옵니다",
    "사옵니다",
    "나이다",
    "ㅂ니까",
    "ᆸ니까",
    "습니까",
    "옵니까",
    "십시오",
    "으십시오",
    "소서",
    "ㅂ시다",
    "ᆸ시다",
    "읍시다",
}

_HAO_EF = {"오", "소", "구려", "리다"}

_HAGE_UNAMBIGUOUS_EF = {"게", "세", "는가", "은가", "던가", "렷다", "음세"}

_HAERA_EF = {
    "다",
    "는다",
    "ㄴ다",
    "ᆫ다",
    "단다",
    "란다",
    "더구나",
    "구나",
    "군",
    "구먼",
    "느냐",
    "으냐",
    "니",
    "거라",
    "너라",
    "어라",
    "아라",
    "라",
    "으라",
    "자",
    "마",
    "랴",
    "리라",
}

_POLITE_LEVELS: set[SpeechLevelType] = {"합쇼체", "해요체"}
_NON_POLITE_LEVELS: set[SpeechLevelType] = {"하오체", "하게체", "반말체", "해라체"}

_PAREN_CUE_RE = re.compile(r"(\([^)]*\)|\[[^\]]*\])")


@lru_cache(maxsize=1)
def get_kiwi() -> Kiwi:
    """Singleton Kiwi morphological analyzer instance."""
    return Kiwi()


def strip_sdh_brackets(text: str) -> str:
    """Remove bracketed SDH speaker/sound descriptors before morphological speech-level parsing."""
    return _PAREN_CUE_RE.sub(" ", text).strip()


def analyze_utterance_honorifics(text: str) -> HonorificAnalysis:
    """Parse Korean subject, object, and 6-level addressee honorifics using Kiwi POS tags."""
    clean = strip_sdh_brackets(text)
    if not clean:
        return HonorificAnalysis()

    kiwi = get_kiwi()
    tokens = kiwi.tokenize(clean)
    if not tokens:
        return HonorificAnalysis()

    subj_hon = False
    obj_hon = False

    for idx, tok in enumerate(tokens):
        if tok.tag == "JKS" and tok.form == "께서":
            subj_hon = True
        if tok.tag == "EP" and tok.form in {"시", "으시", "샤"}:
            subj_hon = True
        if tok.tag.startswith("V") and tok.form in _SUBJECT_HONORIFIC_VERBS:
            subj_hon = True

        if tok.tag == "JKB" and tok.form == "께":
            obj_hon = True
        if tok.tag.startswith(("VV", "VX", "NNG")) and tok.form in _OBJECT_HONORIFIC_LEMMAS:
            # Exclude causative/auxiliary '드리다' false positives only when clearly unrelated; standard lemma matches
            obj_hon = True
        if tok.form == "말씀" and idx + 1 < len(tokens) and tokens[idx + 1].form == "드리":
            obj_hon = True

    # Inspect trailing clause tokens (ignoring final punctuation SF/SP/SS/SE/SW)
    content_tokens = [t for t in tokens if not t.tag.startswith("S")]
    if not content_tokens:
        return HonorificAnalysis(subject_honorific=subj_hon, object_honorific=obj_hon)

    ending_morphemes: list[str] = []
    ef_token = None
    has_final_yo_jx = False

    if content_tokens[-1].tag == "JX" and content_tokens[-1].form == "요":
        has_final_yo_jx = True
        ending_morphemes.append("요/JX")

    for tok in reversed(content_tokens):
        if tok.tag == "EF":
            ef_token = tok
            ending_morphemes.insert(0, f"{tok.form}/EF")
            break

    # Subtitle lines often omit final punctuation, causing Kiwi to tag sentence-final endings as EC
    if ef_token is None and content_tokens[-1].tag == "EC":
        ef_token = content_tokens[-1]
        ending_morphemes.insert(0, f"{ef_token.form}/EC")

    speech_level: SpeechLevelType = "미분류"
    if has_final_yo_jx:
        speech_level = "해요체"
    elif ef_token is not None:
        ef_form = ef_token.form
        if ef_form.endswith("요") or ef_form in {"세요", "으세요"}:
            speech_level = "해요체"
        elif ef_form in _HAPSYO_EF or ef_form.endswith(("ㅂ니다", "ᆸ니다", "습니다", "ㅂ니까", "ᆸ니까", "습니까", "십시오")):
            speech_level = "합쇼체"
        elif ef_form in _HAO_EF:
            speech_level = "하오체"
        elif ef_form in _HAGE_UNAMBIGUOUS_EF:
            speech_level = "하게체"
        elif ef_form == "네" and any(t.form in {"자네", "이보게"} for t in content_tokens):
            speech_level = "하게체"
        elif ef_form in _HAERA_EF or ef_form.endswith(("는다", "ㄴ다", "ᆫ다", "구나", "느냐")):
            speech_level = "해라체"
        else:
            speech_level = "반말체"

    is_polite: bool | None = None
    if speech_level in _POLITE_LEVELS:
        is_polite = True
    elif speech_level in _NON_POLITE_LEVELS:
        is_polite = False

    return HonorificAnalysis(
        subject_honorific=subj_hon,
        object_honorific=obj_hon,
        speech_level=speech_level,
        is_polite_group=is_polite,
        ending_morphemes=ending_morphemes,
    )


def convert_speech_level_cape(text: str, target_level: SpeechLevelType) -> str:
    """Rule-based morphological CAPE post-editor converting between polite and non-polite endings."""
    clean = text.strip()
    if not clean:
        return clean

    kiwi = get_kiwi()
    tokens = kiwi.tokenize(clean)
    if not tokens:
        return clean

    # Extract trailing punctuation if present
    trail_punct = ""
    core_end = len(tokens)
    while core_end > 0 and tokens[core_end - 1].tag.startswith("S"):
        core_end -= 1
    if core_end < len(tokens):
        trail_punct = clean[tokens[core_end].start :]

    core_tokens = tokens[:core_end]
    if not core_tokens:
        return clean

    morphs: list[tuple[str, str]] = [(t.form, t.tag) for t in core_tokens]

    if target_level == "해요체":
        # If ends in 반말체 EF/EC (e.g. 아/어/여/지/네/거든/잖아), append JX '요' or convert 해라체 '다' -> '어요'
        last_form, last_tag = morphs[-1]
        if last_tag == "JX" and last_form == "요":
            return clean
        if last_tag in {"EF", "EC"}:
            if last_form in {"아", "어", "여", "지", "네", "거든", "잖아", "군", "걸", "래", "대", "까", "는데", "은데", "ㄴ데", "ᆫ데"}:
                morphs[-1] = (last_form, "EF")
                morphs.append(("요", "JX"))
                return kiwi.join(morphs) + trail_punct
            if last_form in {"다", "는다", "ㄴ다", "ᆫ다"}:
                morphs[-1] = ("어요", "EF")
                return kiwi.join(morphs) + trail_punct
            if last_form in {"ㅂ니다", "ᆸ니다", "습니다"}:
                morphs[-1] = ("어요", "EF")
                return kiwi.join(morphs) + trail_punct

    elif target_level == "합쇼체":
        if morphs[-1] == ("요", "JX") and len(morphs) >= 2 and morphs[-2][1] in {"EF", "EC"}:
            morphs.pop()
        last_form, last_tag = morphs[-1]
        is_question = "?" in trail_punct or last_form in {"니", "느냐", "까", "나요", "가요"}
        target_ef = "습니까" if is_question else "습니다"
        if last_tag in {"EF", "EC"}:
            morphs[-1] = (target_ef, "EF")
            return kiwi.join(morphs) + trail_punct

    elif target_level == "반말체":
        if morphs[-1] == ("요", "JX"):
            morphs.pop()
            return kiwi.join(morphs) + trail_punct
        last_form, last_tag = morphs[-1]
        if last_tag in {"EF", "EC"} and last_form.endswith("요") and len(last_form) > 1:
            morphs[-1] = (last_form[:-1], "EF")
            return kiwi.join(morphs) + trail_punct
        if last_tag in {"EF", "EC"} and last_form in {"ㅂ니다", "ᆸ니다", "습니다", "다", "는다", "ㄴ다", "ᆫ다"}:
            morphs[-1] = ("어", "EF")
            return kiwi.join(morphs) + trail_punct

    return clean
