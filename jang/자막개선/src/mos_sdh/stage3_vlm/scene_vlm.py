"""4-bit local VLM (Qwen2.5-VL) scene description and 6-speech-level social relationship graph builder."""

from __future__ import annotations

import json
import re
from pathlib import Path

from mos_sdh.config import AnalysisConfig
from mos_sdh.core.vram import StageVRAMReport, gpu_stage
from mos_sdh.schemas.document import (
    SceneContext,
    SegmentMoS,
    SpeakerRelation,
    SpeechLevelType,
)

_VALID_SPEECH_LEVELS: set[str] = {"합쇼체", "해요체", "하오체", "하게체", "반말체", "해라체"}

_SUPERIOR_VOCATIVES = ("부장", "과장", "대표", "사장", "팀장", "교수", "선생", "박사", "회장", "선배", "어르신", "아버지", "어머니")
_INTIMATE_VOCATIVES = ("야 ", "너 ", "얘야", "자네")


def infer_scene_contexts_and_relationships(
    scenes: list[SceneContext],
    keyframe_paths: dict[int, Path],
    segments: list[SegmentMoS],
    config: AnalysisConfig,
    telemetry_sink: list[StageVRAMReport] | None = None,
) -> list[SceneContext]:
    """Infer spatial scene descriptions and speaker-pair speech-level priors via local 4-bit VLM + dialogue cues."""
    if not scenes:
        return scenes

    # Always compute fast linguistic/OCR baseline priors first (Hwang et al., 2021 Section 3 vocative clues)
    for sc in scenes:
        sc_segs = [s for s in segments if s.visual.scene_id == sc.scene_id]
        sc.inferred_relations = _heuristic_relations_from_dialogue(sc_segs)
        if not sc.description_ko:
            ocr_hint = f" (화면 텍스트: {', '.join(sc.ocr_summary[:3])})" if sc.ocr_summary else ""
            sc.description_ko = f"씬 {sc.scene_id} 대화 장면{ocr_hint}"

    if not config.enable_vlm_stage or not keyframe_paths:
        return scenes

    with gpu_stage("Stage 3: Local 4-bit VLM Scene & Relationship Graph", telemetry_sink=telemetry_sink) as device:
        import torch
        from PIL import Image
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

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
            processor = AutoProcessor.from_pretrained(
                config.vlm_model,
                min_pixels=256 * 28 * 28,
                max_pixels=512 * 28 * 28,
            )
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config.vlm_model,
                quantization_config=quant_cfg,
                device_map="auto" if device == "cuda" else None,
                dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            model.eval()

            for sc in scenes:
                kf_path = keyframe_paths.get(sc.scene_id)
                if kf_path is None or not kf_path.exists():
                    continue

                sc_segs = [s for s in segments if s.visual.scene_id == sc.scene_id][:6]
                dialogue_preview = "\n".join(
                    f"- {s.speaker_id}: {s.existing_subtitle or s.raw_transcript}" for s in sc_segs
                )
                ocr_preview = ", ".join(sc.ocr_summary[:4]) if sc.ocr_summary else "없음"

                prompt_text = (
                    "다음 드라마/영화 장면 이미지와 대사 초안을 보고 배리어프리 자막 제작을 위한 상황과 인물 관계를 JSON으로만 답하세요.\n"
                    f"[화면 OCR 텍스트]: {ocr_preview}\n"
                    f"[대사 초안]:\n{dialogue_preview or '(대사 없음)'}\n\n"
                    '출력 JSON 형식: {"description_ko": "장소·분위기·상황 한 줄 요약", '
                    '"relations": [{"from_speaker": "화자_1", "to_speaker": "화자_2", '
                    '"relation_ko": "직장 부하->상사 등", "prior_speech_level": "합쇼체 또는 해요체 또는 반말체"}]}'
                )

                img = Image.open(kf_path).convert("RGB")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]
                chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[chat_text], images=[img], padding=True, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    out_ids = model.generate(**inputs, max_new_tokens=180, do_sample=False)

                gen_ids = out_ids[0][inputs["input_ids"].shape[1] :]
                resp_text = processor.decode(gen_ids, skip_special_tokens=True).strip()
                _apply_vlm_json_to_scene(sc, resp_text)

            del model, processor
        except Exception:
            # Keep clean heuristic relations & OCR scene summaries if VLM load/inference fails
            pass

    return scenes


def _apply_vlm_json_to_scene(sc: SceneContext, raw_text: str) -> None:
    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if not match:
        return
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return

    desc = str(payload.get("description_ko", "")).strip()
    if desc:
        sc.description_ko = desc

    rels_raw = payload.get("relations")
    if isinstance(rels_raw, list) and rels_raw:
        parsed_rels: list[SpeakerRelation] = []
        for item in rels_raw:
            if not isinstance(item, dict):
                continue
            lvl = str(item.get("prior_speech_level", "해요체")).strip()
            prior_lvl: SpeechLevelType = lvl if lvl in _VALID_SPEECH_LEVELS else "해요체"  # type: ignore[assignment]
            parsed_rels.append(
                SpeakerRelation(
                    from_speaker=str(item.get("from_speaker", "화자_1")),
                    to_speaker=str(item.get("to_speaker", "상대방")),
                    relation_ko=str(item.get("relation_ko", "대화 상대")),
                    prior_speech_level=prior_lvl,
                    confidence=0.82,
                )
            )
        if parsed_rels:
            sc.inferred_relations = parsed_rels


def _heuristic_relations_from_dialogue(sc_segs: list[SegmentMoS]) -> list[SpeakerRelation]:
    """Infer initial speaker relationship priors from Korean vocatives (Hwang et al., 2021 Figure 2)."""
    relations: list[SpeakerRelation] = []
    seen_pairs: set[tuple[str, str]] = set()
    speakers = sorted({s.speaker_id for s in sc_segs})

    for idx, seg in enumerate(sc_segs):
        txt = seg.existing_subtitle or seg.raw_transcript
        other_spk = "상대방"
        for s in speakers:
            if s != seg.speaker_id:
                other_spk = s
                break

        pair = (seg.speaker_id, other_spk)
        if pair in seen_pairs:
            continue

        if any(voc in txt for voc in _SUPERIOR_VOCATIVES):
            relations.append(
                SpeakerRelation(
                    from_speaker=seg.speaker_id,
                    to_speaker=other_spk,
                    relation_ko="하위자/후배 -> 상위자/연장자 (호칭 단서)",
                    prior_speech_level="합쇼체",
                    confidence=0.80,
                )
            )
            seen_pairs.add(pair)
        elif any(voc in txt for voc in _INTIMATE_VOCATIVES):
            relations.append(
                SpeakerRelation(
                    from_speaker=seg.speaker_id,
                    to_speaker=other_spk,
                    relation_ko="동년배/친밀 또는 상위자 -> 하위자 (호칭 단서)",
                    prior_speech_level="반말체",
                    confidence=0.75,
                )
            )
            seen_pairs.add(pair)
    return relations
