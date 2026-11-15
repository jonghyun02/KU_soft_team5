"""Dual-format Barrier-Free SDH subtitle renderer (.srt/.vtt text cues + .ass visual typography)."""

from __future__ import annotations

from pathlib import Path
import pysubs2
from pysubs2 import Alignment, Color, SSAEvent, SSAFile, SSAStyle

from mos_sdh.schemas.document import MoSDocument, SegmentMoS

# ASS BGR-ordered pastel speaker palette (high legibility with dark outline for SDH viewers)
_SPEAKER_COLORS: list[Color] = [
    Color(255, 255, 255, 0),  # 화자_1: Pure White
    Color(153, 238, 255, 0),  # 화자_2: Warm Soft Yellow
    Color(178, 255, 190, 0),  # 화자_3: Mint Green
    Color(255, 214, 165, 0),  # 화자_4: Soft Sky Blue
    Color(255, 198, 255, 0),  # 화자_5: Soft Lavender
    Color(180, 210, 255, 0),  # 화자_6: Peach
]

_EMPH_SCALE_PCT = {
    2: 108,
    3: 116,
    4: 125,
}

_EMOTION_TONE_KO = {
    "분노": "분노하며",
    "슬픔": "슬픈 목소리로",
    "기쁨": "밝은 목소리로",
    "공포": "떨리는 목소리로",
    "냉소": "비꼬는 어조로",
}

_ASS_BOTTOM_ALIGN = {
    "left": r"{\an1}",
    "center": "",
    "right": r"{\an3}",
}

_ASS_TOP_ALIGN = {
    "left": r"{\an7}",
    "center": r"{\an8}",
    "right": r"{\an9}",
}


def populate_sdh_render_strings(doc: MoSDocument, overwrite: bool = False) -> MoSDocument:
    """Populate `sdh_srt_text` and `sdh_ass_text` fields on every SegmentMoS according to PRD Section 6."""
    for seg in doc.segments:
        if overwrite or not seg.sdh_srt_text:
            seg.sdh_srt_text = build_srt_line(seg)
        if overwrite or not seg.sdh_ass_text:
            seg.sdh_ass_text = build_ass_line(seg)
    return doc


def export_subtitles(
    doc: MoSDocument,
    out_dir: Path,
    stem: str = "prototype_sdh",
) -> dict[str, Path]:
    """Write .srt, .vtt, and rich-styled .ass Barrier-Free subtitle prototypes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    populate_sdh_render_strings(doc, overwrite=False)

    # 1. Standard Text Cue Format (.srt and .vtt)
    srt_subs = SSAFile()
    for seg in doc.segments:
        ev = SSAEvent(
            start=seg.start_ms,
            end=max(seg.start_ms + 300, seg.end_ms),
            text=seg.sdh_srt_text,
        )
        srt_subs.events.append(ev)

    srt_path = out_dir / f"{stem}.srt"
    vtt_path = out_dir / f"{stem}.vtt"
    srt_subs.save(str(srt_path), format_="srt", encoding="utf-8")
    srt_subs.save(str(vtt_path), format_="vtt", encoding="utf-8")

    # 2. Visual Typography Format (.ass)
    ass_subs = SSAFile()
    ass_subs.info["PlayResX"] = "1920"
    ass_subs.info["PlayResY"] = "1080"
    ass_subs.info["Title"] = "MoS-SDH Barrier-Free Visual Prototype"

    speakers = sorted({seg.speaker_id for seg in doc.segments})
    for idx, spk in enumerate(speakers):
        style = SSAStyle(
            fontname="Malgun Gothic",
            fontsize=54.0,
            primarycolor=_SPEAKER_COLORS[idx % len(_SPEAKER_COLORS)],
            outlinecolor=Color(20, 20, 20, 0),
            backcolor=Color(0, 0, 0, 120),
            bold=False,
            outline=3.0,
            shadow=1.5,
            alignment=Alignment.BOTTOM_CENTER,
            marginv=55,
        )
        ass_subs.styles[spk] = style

    ass_subs.styles["Default"] = ass_subs.styles.get(
        "화자_1", SSAStyle(fontname="Malgun Gothic", fontsize=54.0)
    )

    for seg in doc.segments:
        style_name = seg.speaker_id if seg.speaker_id in ass_subs.styles else "Default"
        ev = SSAEvent(
            start=seg.start_ms,
            end=max(seg.start_ms + 300, seg.end_ms),
            style=style_name,
            text=seg.sdh_ass_text,
        )
        ass_subs.events.append(ev)

    ass_path = out_dir / f"{stem}.ass"
    ass_subs.save(str(ass_path), format_="ass", encoding="utf-8")

    return {"srt": srt_path, "vtt": vtt_path, "ass": ass_path}


def build_srt_line(
    seg: SegmentMoS,
    override_body: str | None = None,
    include_mild_emotion: bool = True,
) -> str:
    """Render standard Korean SDH bracketed descriptor + emphasis/CAPE-corrected dialogue."""
    spk_label = (
        f"[{seg.speaker_id}(화면 밖)]"
        if seg.visual.on_screen_speaker is False
        else f"[{seg.speaker_id}]"
    )
    prefix_tags: list[str] = [spk_label]

    tone_notes: list[str] = []
    if seg.emotion_irony.irony_detected:
        tone_notes.append("비꼬는 어조로")

    if seg.honorific.consistency_status == "dramatic_shift":
        if seg.honorific.is_polite_group is False:
            tone_notes.append("격앙되어 반말로")
        else:
            tone_notes.append("어조를 바꾸어 존댓말로")

    if not tone_notes and include_mild_emotion:
        dom_emo = seg.emotion_irony.dominant_emotion
        if dom_emo in _EMOTION_TONE_KO:
            tone_notes.append(_EMOTION_TONE_KO[dom_emo])
        elif seg.arousal_zscore >= 1.5:
            tone_notes.append("힘주어")

    leading_cues = [c.label_ko for c in seg.cues if c.cue_code in {"b/", "l/", "c/"}]
    trailing_sfx = [c.label_ko for c in seg.cues if c.cue_code == "sfx"]

    if tone_notes:
        prefix_tags.append(f"({', '.join(tone_notes)})")

    if override_body is not None:
        body = override_body
    elif seg.honorific.consistency_status == "mismatch_error" and seg.honorific.suggested_correction:
        body = seg.honorific.suggested_correction
    elif not seg.disfluency.preserve_disfluency and seg.disfluency.cleaned_text:
        body = seg.disfluency.cleaned_text
    elif seg.words:
        parts: list[str] = []
        for w in seg.words:
            if w.emphasis_level >= 3:
                parts.append(f"'{w.text}'")
            else:
                parts.append(w.text)
        body = " ".join(parts)
    else:
        body = seg.existing_subtitle or seg.raw_transcript

    suffix_parts: list[str] = []
    for lbl in leading_cues:
        suffix_parts.append(f"({lbl})")
    for sfx in trailing_sfx:
        suffix_parts.append(f"[{sfx}]")

    pieces = [" ".join(prefix_tags), body.strip()]
    if suffix_parts:
        pieces.append(" ".join(suffix_parts))
    return " ".join(p for p in pieces if p).strip()


def build_ass_line(
    seg: SegmentMoS,
    override_body: str | None = None,
    include_mild_emotion: bool = True,
) -> str:
    """Render ASS override tags: spatial zone + OCR collision avoidance, emphasis scaling, and styled tone cues."""
    zone = seg.visual.speaker_zone
    align_tag = (
        _ASS_TOP_ALIGN.get(zone, r"{\an8}")
        if seg.visual.avoid_bottom_region
        else _ASS_BOTTOM_ALIGN.get(zone, "")
    )

    spk_label = (
        f"[{seg.speaker_id}(화면 밖)]"
        if seg.visual.on_screen_speaker is False
        else f"[{seg.speaker_id}]"
    )
    header = f"{align_tag}{spk_label} "

    tone_parts: list[str] = []
    if seg.emotion_irony.irony_detected:
        tone_parts.append(r"{\c&HFFCC66&\i1}(비꼬는 어조로){\i0\r}")
    if seg.honorific.consistency_status == "dramatic_shift":
        note = "격앙되어 반말로" if seg.honorific.is_polite_group is False else "어조 전환"
        tone_parts.append(rf"{{\c&H00D7FF&\i1}}({note}){{\i0\r}}")
    if not tone_parts and include_mild_emotion:
        dom_emo = seg.emotion_irony.dominant_emotion
        if dom_emo in _EMOTION_TONE_KO:
            tone_parts.append(rf"{{\i1}}({_EMOTION_TONE_KO[dom_emo]}){{\i0}}")

    tone_prefix = (" ".join(tone_parts) + " ") if tone_parts else ""

    if override_body is not None:
        body = override_body.replace("'", "")
    elif seg.honorific.consistency_status == "mismatch_error" and seg.honorific.suggested_correction:
        body = seg.honorific.suggested_correction
    elif seg.words:
        styled_words: list[str] = []
        for w in seg.words:
            lvl = w.emphasis_level
            if lvl in _EMPH_SCALE_PCT:
                pct = _EMPH_SCALE_PCT[lvl]
                extra_color = r"\c&H66FFFF&" if lvl == 4 else ""
                styled_words.append(rf"{{\b1\fscx{pct}\fscy{pct}{extra_color}}}{w.text}{{\r}}")
            else:
                styled_words.append(w.text)
        body = " ".join(styled_words)
    else:
        body = seg.existing_subtitle or seg.raw_transcript

    cue_tail = ""
    if seg.cues:
        labels = ", ".join(c.label_ko for c in seg.cues)
        cue_tail = rf" {{\i1}}({labels}){{\i0}}"

    return f"{header}{tone_prefix}{body.strip()}{cue_tail}".strip()
