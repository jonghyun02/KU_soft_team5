"""Positional Tag-and-Generate (Madaan et al., 2020) + Augmented Zero-Shot Local LLM (Reif et al., 2022)."""

from __future__ import annotations

import re

from mos_sdh.config import AnalysisConfig
from mos_sdh.core.vram import StageVRAMReport, gpu_stage
from mos_sdh.schemas.document import SegmentMoS, SubtitleCandidate
from mos_sdh.stage4_mos.korean_honorifics import strip_sdh_brackets

# Reif et al. (2022) Figure 1c / Table 7: Fixed rewrite demonstrations with curly-brace { } delimiters
_AUGMENTED_ZEROSHOT_DEMOS = """다음은 원문 대사의 핵심 의미를 그대로 보존하면서 목표 화법(어조·경어·강조) 조건만 반영하여 중괄호 { } 안에 재작성하는 예시입니다.

원문 대사: {지금 당장 서류 가져와}
목표 조건: 공손한 합쇼체로 완화
재작성: {지금 바로 서류를 가져다주십시오.}

원문 대사: {그 사람 말은 믿을 수 없어}
목표 조건: '절대' 강조 부사를 넣고 단호한 해요체로 변환
재작성: {그 사람 말은 절대 믿을 수 없어요.}

원문 대사: {참 일 처리 잘한다}
목표 조건: 차가운 반어법 뉘앙스를 살린 합쇼체로 변환
재작성: {참, 일 처리 한 번 깔끔하게 잘하시는군요.}
"""

_EMOTION_DESCRIPTOR_KO = {
    "분노": "분노하며",
    "슬픔": "슬픈 목소리로",
    "기쁨": "밝은 목소리로",
    "공포": "떨리는 목소리로",
    "냉소": "비꼬는 어조로",
}

_INTENSIFIER_BY_LEVEL = {
    3: "정말",
    4: "절대",
}


def generate_sdh_candidates(
    segments: list[SegmentMoS],
    config: AnalysisConfig,
     render_srt_fn: object,
    render_ass_fn: object,
    telemetry_sink: list[StageVRAMReport] | None = None,
) -> list[SegmentMoS]:
    """Generate up to 4 SDH subtitle candidates per segment (Positional Tag variants + Local 4-bit LLM)."""
    if not segments:
        return segments

    # Step 1: Deterministic interpretable Positional Tag candidates (Madaan et al., 2020)
    for seg in segments:
        seg.candidates = _build_positional_tag_candidates(seg, render_srt_fn, render_ass_fn)

    # Step 2: Optional 4-bit Local LLM Augmented Zero-Shot candidate for salient MoS segments
    salient_segments = [
        seg
        for seg in segments
        if seg.honorific.consistency_status in {"mismatch_error", "dramatic_shift"}
        or seg.emotion_irony.irony_detected
        or any(w.emphasis_level >= 3 for w in seg.words)
    ]

    if not config.enable_llm_stage or not salient_segments:
        return segments

    with gpu_stage("Stage 5-A: Local 4-bit LLM Augmented Zero-Shot", telemetry_sink=telemetry_sink) as device:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        try:
            quant_cfg = (
                BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
                if device == "cuda"
                else None
            )
            tokenizer = AutoTokenizer.from_pretrained(config.llm_model)
            model = AutoModelForCausalLM.from_pretrained(
                config.llm_model,
                quantization_config=quant_cfg,
                device_map="auto" if device == "cuda" else None,
                dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            model.eval()

            for seg in salient_segments:
                base_text = strip_sdh_brackets(seg.existing_subtitle or seg.raw_transcript)
                target_cond = _describe_target_mos_condition(seg)
                user_prompt = (
                    f"{_AUGMENTED_ZEROSHOT_DEMOS}\n"
                    f"원문 대사: {{{base_text}}}\n"
                    f"목표 조건: {target_cond}\n"
                    "재작성: {"
                )

                inputs = tokenizer(user_prompt, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                # Reif et al. (2022) probabilistic decoding hyperparameters: top_k=40, top_p=0.6
                with torch.no_grad():
                    out_ids = model.generate(
                        **inputs,
                        max_new_tokens=64,
                        do_sample=True,
                        top_k=40,
                        top_p=0.6,
                        temperature=0.65,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                gen_tokens = out_ids[0][inputs["input_ids"].shape[1] :]
                decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                rewritten = _extract_curly_brace_rewrite(decoded)

                if rewritten and len(rewritten) >= 2:
                    srt_line = render_srt_fn(seg, override_body=rewritten, include_mild_emotion=True)
                    ass_line = render_ass_fn(seg, override_body=rewritten, include_mild_emotion=True)
                    seg.candidates.append(
                        SubtitleCandidate(
                            candidate_id=len(seg.candidates) + 1,
                            strategy="llm_augmented_zeroshot",
                            srt_text=srt_line,
                            ass_text=ass_line,
                            plain_dialogue=strip_sdh_brackets(rewritten),
                        )
                    )

            del model, tokenizer
        except Exception:
            # Keep deterministic positional-tag candidates if local LLM is offline or skipped
            pass

    return segments


def _build_positional_tag_candidates(
    seg: SegmentMoS,
    render_srt_fn: object,
    render_ass_fn: object,
) -> list[SubtitleCandidate]:
    """Build 3 structured Positional-Tag candidates (Standard, Lexical-Intensified, Concise Minimal)."""
    base_plain = strip_sdh_brackets(seg.existing_subtitle or seg.raw_transcript)
    corrected_plain = (
        seg.honorific.suggested_correction
        if seg.honorific.consistency_status == "mismatch_error" and seg.honorific.suggested_correction
        else base_plain
    )

    # Candidate 1: Balanced Standard SDH (CAPE honorific fix + tone/irony cue + emphasis quotes)
    c1_srt = render_srt_fn(seg, override_body=None, include_mild_emotion=True)
    c1_ass = render_ass_fn(seg, override_body=None, include_mild_emotion=True)

    candidates: list[SubtitleCandidate] = [
        SubtitleCandidate(
            candidate_id=1,
            strategy="positional_tag_standard",
            srt_text=c1_srt,
            ass_text=c1_ass,
            plain_dialogue=corrected_plain,
        )
    ]

    # Candidate 2: Tokuyama et al. NMT-Insertion / Madaan [TAG_t] degree adverb insertion on strong emphasis
    strong_words = [w for w in seg.words if w.emphasis_level >= 3]
    if strong_words:
        intensified_tokens: list[str] = []
        inserted = False
        for w in seg.words:
            if not inserted and w.emphasis_level >= 3 and w.text not in {"정말", "진짜", "절대", "너무", "무척"}:
                adv = "정말" if w.emphasis_level == 3 else "진짜"
                intensified_tokens.append(f"{adv} '{w.text}'")
                inserted = True
            else:
                intensified_tokens.append(w.text)
        intensified_body = " ".join(intensified_tokens)
        if seg.honorific.consistency_status == "mismatch_error" and seg.honorific.suggested_correction:
            intensified_body = f"정말 {seg.honorific.suggested_correction}"

        candidates.append(
            SubtitleCandidate(
                candidate_id=2,
                strategy="tokuyama_intensifier_insertion",
                srt_text=render_srt_fn(seg, override_body=intensified_body, include_mild_emotion=True),
                ass_text=render_ass_fn(seg, override_body=intensified_body, include_mild_emotion=True),
                plain_dialogue=intensified_body.replace("'", ""),
            )
        )

    # Candidate 3: Concise High-Readability SDH (low CPS, omits mild stage directions)
    concise_body = (
        seg.disfluency.cleaned_text
        if seg.disfluency.cleaned_text
        else corrected_plain
    )
    candidates.append(
        SubtitleCandidate(
            candidate_id=len(candidates) + 1,
            strategy="concise_low_cps",
            srt_text=render_srt_fn(seg, override_body=concise_body, include_mild_emotion=False),
            ass_text=render_ass_fn(seg, override_body=concise_body, include_mild_emotion=False),
            plain_dialogue=concise_body,
        )
    )
    return candidates


def _describe_target_mos_condition(seg: SegmentMoS) -> str:
    parts: list[str] = []
    if seg.honorific.consistency_status == "mismatch_error" and seg.honorific.expected_speech_level:
        parts.append(f"문맥에 맞게 '{seg.honorific.expected_speech_level}' 종결 어미로 자연스럽게 교정")
    elif seg.honorific.consistency_status == "dramatic_shift":
        parts.append(f"격앙된 감정의 '{seg.honorific.speech_level}' 어조 유지")
    else:
        parts.append(f"'{seg.honorific.speech_level}' 화계 유지")

    if seg.emotion_irony.irony_detected:
        parts.append("비꼬는 반어법 어감을 살림")
    elif seg.emotion_irony.dominant_emotion in _EMOTION_DESCRIPTOR_KO:
        parts.append(f"{seg.emotion_irony.dominant_emotion} 정서 반영")

    emph_words = [w.text for w in seg.words if w.emphasis_level >= 3]
    if emph_words:
        parts.append(f"강조 단어({', '.join(emph_words[:2])})의 힘준 느낌 보존")
    parts.append("원문 핵심 단어와 사실 정보 절대 누락 금지")
    return ", ".join(parts)


def _extract_curly_brace_rewrite(decoded: str) -> str:
    """Parse text up to closing '}' delimiter (Reif et al., 2022 parsing failure suppression)."""
    if "}" in decoded:
        candidate = decoded.split("}", 1)[0]
    else:
        candidate = decoded.splitlines()[0] if decoded.splitlines() else decoded
    candidate = candidate.strip().strip("{}'\" ")
    # Reject chatty meta responses
    if any(bad in candidate for bad in ("재작성:", "원문:", "목표 조건:", "예시")):
        return ""
    candidate = re.sub(r"\s+", " ", candidate)
    return candidate
