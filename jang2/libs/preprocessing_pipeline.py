"""AI-Hub 71693(국제학술대회) 및 71379(방송콘텐츠) 영-한 데이터셋 통합 전처리 파이프라인 모듈."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
import importlib
import inspect
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch

from libs.alignment import (
    KoAlignFn,
    build_hf_ctc_forced_align_fn,
    build_hf_wavlm_sim_bin_fn,
    build_hf_whisper_align_fn,
    compute_aligned_token_placement,
)
from libs.codec import (
    ExternalMimiCodec,
    FRAME_RATE,
    MiniMimiCodec,
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
    load_mono_wav_24k,
)
from libs.config import EDarchConfig
from libs.env import setup_device
from libs.preprocessing_common import (
    DEFAULT_CONTEXT_FRAMES,
    ZERO_TOKEN_ID,
    BroadcastUtterance71379,
    UtteranceSpan,
    assemble_multistream_example,
    build_edge_korean_tts_fn,
    build_mms_korean_tts_fn,
    classify_71379_accent,
    clean_transcript_text,
    count_completed_window_tree,
    encode_mimi_timeline,
    extract_10s_speaker_prompt,
    extract_json_records,
    extract_prompt_from_clip_paths,
    index_existing_window_trees,
    index_wav_files,
    is_71693_record,
    mark_71379_segment_breaks,
    mark_71693_segment_breaks,
    normalize_record_keys,
    normalize_tts_output,
    parse_71379_scene_and_seq,
    parse_hhmmss,
    resolve_default_sim_bin,
    resolve_wav_path,
    save_training_example_pt,
    slice_71379_windows,
    slice_71693_windows,
    validate_ko_token_ids,
    validate_sim_bin,
)
from libs.tokenizer import (
    TEXT_EOS_ID,
    TEXT_PAD_ID,
    load_korean_tokenizer,
    resolve_spm_model_path,
    train_korean_sentencepiece,
)

logger = logging.getLogger(__name__)

# 하위 호환 별칭
slice_60s_windows = slice_71693_windows
_encode_and_align_mimi = encode_mimi_timeline


def _call_tts_fn(
    tts_fn: Callable[..., Union[torch.Tensor, Tuple[torch.Tensor, int]]],
    ko_text: str,
    speaker_prompt_24k: torch.Tensor,
    gender: str = "",
) -> torch.Tensor:
    """2-인자(제로샷 훅) 및 3-인자(성별 지원 TTS) 함수를 모두 호환 호출하여 24kHz 파형으로 반환합니다."""
    try:
        sig = inspect.signature(tts_fn)
        params = [
            p
            for p in sig.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
        if has_varargs or len(params) >= 3:
            raw_out = tts_fn(ko_text, speaker_prompt_24k, gender)
        else:
            raw_out = tts_fn(ko_text, speaker_prompt_24k)
    except (TypeError, ValueError):
        raw_out = tts_fn(ko_text, speaker_prompt_24k)
    return normalize_tts_output(raw_out)


def load_and_group_71693_json(
    json_dir: Path,
    use_mt_fallback: bool = False,
    include_ko2en: bool = False,
) -> dict[str, List[UtteranceSpan]]:
    """AI-Hub 71693 JSON 라벨들을 파싱하고 미번역 구간 단절을 표시하여 파일별로 시간순 정렬합니다."""
    grouped: dict[str, List[UtteranceSpan]] = defaultdict(list)
    skipped_intervals: dict[str, List[Tuple[float, float]]] = defaultdict(list)
    seen_records: set[tuple[str, str]] = set()

    for jp in sorted(json_dir.rglob("*.json")):
        try:
            payload = json.loads(jp.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("JSON 파싱 실패로 건너뜁니다 (%s): %s", jp, exc)
            continue

        for raw_item in extract_json_records(payload):
            _, item = normalize_record_keys(raw_item)
            if not is_71693_record(item):
                continue

            src_lang = str(item.get("source_language") or "").strip().lower()
            tgt_lang = str(item.get("target_language") or "").strip().lower()
            is_en2ko = src_lang.startswith("en") and tgt_lang.startswith("ko")
            is_ko2en = include_ko2en and src_lang.startswith("ko") and tgt_lang.startswith("en")
            if not (is_en2ko or is_ko2en):
                continue

            if not all(k in item for k in ("sn", "file_name", "audio_start", "audio_end")):
                continue

            file_name = str(item["file_name"]).strip()
            sn = str(item["sn"]).strip()
            record_key = (file_name, sn)
            if record_key in seen_records:
                continue

            try:
                start_sec = parse_hhmmss(item["audio_start"])
                end_sec = parse_hhmmss(item["audio_end"])
            except (ValueError, AttributeError, TypeError) as exc:
                logger.warning("타임스탬프 파싱 실패로 레코드 건너뜀 (sn=%s): %s", sn, exc)
                continue
            if end_sec <= start_sec:
                continue
            seen_records.add(record_key)

            tgt_trans = clean_transcript_text(item.get("mtpe"))
            if not tgt_trans and use_mt_fallback:
                tgt_trans = clean_transcript_text(item.get("mt"))
            src_trans = clean_transcript_text(item.get("source_cleaned")) or clean_transcript_text(
                item.get("source_original")
            )

            if is_en2ko:
                en_text, ko_text, direction = src_trans, tgt_trans, "en2ko"
            else:
                ko_text, en_text, direction = src_trans, tgt_trans, "ko2en"

            if not ko_text or (is_ko2en and not en_text):
                skipped_intervals[file_name].append((start_sec, end_sec))
                continue

            grouped[file_name].append(
                UtteranceSpan(
                    sn=sn,
                    file_name=file_name,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    en_text=en_text,
                    ko_mtpe=ko_text,
                    gender=clean_transcript_text(item.get("info_gender")),
                    direction=direction,
                )
            )

    for fname, spans in grouped.items():
        mark_71693_segment_breaks(spans, skipped_intervals.get(fname))
    return grouped


def _assemble_virtual_lecture_from_sn_clips(
    spans: List[UtteranceSpan],
    wav_dir: Path,
    wav_index: dict[str, Path],
) -> Optional[torch.Tensor]:
    """전체 강의 WAV 대신 발화별 분할 클립(`{sn}.wav`)만 존재할 때 가상 타임라인 WAV를 재조립합니다."""
    max_end_sec = max((sp.end_sec for sp in spans), default=0.0)
    if max_end_sec <= 0.0:
        return None
    resolved_clips: List[Tuple[UtteranceSpan, Path]] = []
    for sp in spans:
        clip_path = resolve_wav_path(wav_dir, f"{sp.sn}.wav", wav_index)
        if clip_path is not None:
            resolved_clips.append((sp, clip_path))
    if not resolved_clips:
        return None

    total_samples = int(max_end_sec * SAMPLE_RATE) + SAMPLE_RATE
    canvas = torch.zeros(total_samples, dtype=torch.float32)
    for sp, clip_path in resolved_clips:
        clip_wav = load_mono_wav_24k(clip_path)
        s_idx = max(0, int(sp.start_sec * SAMPLE_RATE))
        max_len = min(clip_wav.numel(), total_samples - s_idx)
        if max_len > 0:
            canvas[s_idx : s_idx + max_len] = clip_wav[:max_len]
    return canvas


def build_EDarch_training_example(
    lecture_wav_path: Path,
    window_spans: List[UtteranceSpan],
    all_lecture_spans: List[UtteranceSpan],
    zero_shot_ko_tts_fn: Optional[
        Callable[..., Union[torch.Tensor, Tuple[torch.Tensor, int]]]
    ],
    ko_sp_encode_fn: Callable[[str], List[int]],
    mimi_encode_fn: Callable[[torch.Tensor], torch.Tensor],
    wavlm_sim_bin_fn: Callable[[torch.Tensor, torch.Tensor], int],
    preloaded_wav_24k: Optional[torch.Tensor] = None,
    speaker_prompt_24k: Optional[torch.Tensor] = None,
    expected_q: Optional[int] = None,
    max_context_frames: int = DEFAULT_CONTEXT_FRAMES,
    text_card: Optional[int] = None,
    num_speaker_sim_bins: int = 5,
    align_mode: str = "auto",
    ko_align_fn: Optional[KoAlignFn] = None,
    ko_sp_pieces_fn: Optional[Callable[[str], List[str]]] = None,
    monologue_lead_frames: int = 1,
    mask_target_audio: bool = False,
    hard_max_frames: Optional[int] = None,
    _out_metadata: Optional[dict[str, Any]] = None,
    _tts_tok_cache: Optional[dict[str, Tuple[torch.Tensor, List[int]]]] = None,
    _suppress_length_warning: bool = False,
) -> Tuple[torch.Tensor, int]:
    """AI-Hub 71693 발화 윈도우를 인과적 정밀 정렬이 적용된 `[1 + 2Q, T]` 코드북 텐서로 변환합니다."""
    if not window_spans:
        raise ValueError("window_spans는 비어 있을 수 없습니다.")
    if not mask_target_audio and zero_shot_ko_tts_fn is None:
        raise ValueError("mask_target_audio=False일 때 zero_shot_ko_tts_fn이 필요합니다.")

    with torch.no_grad():
        wav = (
            preloaded_wav_24k
            if preloaded_wav_24k is not None
            else load_mono_wav_24k(lecture_wav_path)
        )
        wav = wav.detach().cpu().float().reshape(-1)

        if speaker_prompt_24k is None:
            speaker_prompt_24k = extract_10s_speaker_prompt(
                wav, all_lecture_spans, prompt_sec=10.0
            )
        else:
            speaker_prompt_24k = speaker_prompt_24k.detach().cpu().float().reshape(-1)

        win_start_sec = min(sp.start_sec for sp in window_spans)
        win_end_sec = max(sp.end_sec for sp in window_spans)
        start_idx = max(0, min(int(win_start_sec * SAMPLE_RATE), wav.numel()))
        end_idx = max(start_idx, min(int(win_end_sec * SAMPLE_RATE), wav.numel()))
        en_span_samples = end_idx - start_idx

        en_placements: List[Tuple[int, torch.Tensor]] = []
        for sp in window_spans:
            sp_s = max(start_idx, min(int(sp.start_sec * SAMPLE_RATE), wav.numel()))
            sp_e = max(sp_s, min(int(sp.end_sec * SAMPLE_RATE), wav.numel()))
            if sp_e > sp_s:
                en_placements.append((sp_s - start_idx, wav[sp_s:sp_e]))

        ko_placements: List[Tuple[int, torch.Tensor, List[Tuple[int, int]]]] = []
        src_start_frames: List[int] = []
        src_end_frames: List[int] = []
        ref_token_groups: List[List[int]] = []
        ko_cursor_sample = 0
        last_token_frame = -1

        for sp in window_spans:
            cached = _tts_tok_cache.get(sp.sn) if _tts_tok_cache is not None else None
            if cached is not None:
                ko_sent_wav, ko_token_ids = cached
            else:
                ko_token_ids = list(ko_sp_encode_fn(sp.ko_mtpe))
                if mask_target_audio:
                    est_frames = max(len(ko_token_ids) + 2, int(max(1, len(sp.ko_mtpe)) * 1.6))
                    ko_sent_wav = torch.zeros(est_frames * SAMPLES_PER_FRAME, dtype=torch.float32)
                else:
                    assert zero_shot_ko_tts_fn is not None
                    ko_sent_wav = _call_tts_fn(
                        zero_shot_ko_tts_fn, sp.ko_mtpe, speaker_prompt_24k, sp.gender
                    )
                if _tts_tok_cache is not None:
                    _tts_tok_cache[sp.sn] = (ko_sent_wav, ko_token_ids)

            validate_ko_token_ids(ko_token_ids, text_card)
            tok_pieces = (
                list(ko_sp_pieces_fn(sp.ko_mtpe)) if ko_sp_pieces_fn is not None else None
            )

            rel_en_start_sample = max(0, int((sp.start_sec - win_start_sec) * SAMPLE_RATE))
            rel_en_end_sample = max(0, int((sp.end_sec - win_start_sec) * SAMPLE_RATE))
            en_start_frame = rel_en_start_sample // SAMPLES_PER_FRAME
            en_end_frame = (rel_en_end_sample + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME
            prev_audio_end_frame = (ko_cursor_sample + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME

            eff_align = "uniform" if mask_target_audio else align_mode
            audio_start_sample, token_positions, last_token_frame = compute_aligned_token_placement(
                ko_text=sp.ko_mtpe,
                ko_wav_24k=ko_sent_wav,
                ko_token_ids=ko_token_ids,
                en_end_frame=en_end_frame,
                prev_audio_end_frame=prev_audio_end_frame,
                last_token_frame=last_token_frame,
                align_mode=eff_align,
                ko_align_fn=None if mask_target_audio else ko_align_fn,
                token_pieces=tok_pieces,
                monologue_lead_frames=monologue_lead_frames,
            )

            ko_placements.append((audio_start_sample, ko_sent_wav, token_positions))
            src_start_frames.append(int(en_start_frame))
            src_end_frames.append(int(en_end_frame))
            ref_token_groups.append([int(tid) for tid in ko_token_ids])
            ko_cursor_sample = audio_start_sample + ko_sent_wav.numel()

        codes, ko_speech_only, built_meta = assemble_multistream_example(
            en_placements=en_placements,
            ko_placements=ko_placements,
            en_span_end_sample=en_span_samples,
            ko_end_sample=ko_cursor_sample,
            last_token_frame=last_token_frame,
            src_start_frames=src_start_frames,
            src_end_frames=src_end_frames,
            ref_token_groups=ref_token_groups,
            mimi_encode_fn=mimi_encode_fn,
            expected_q=expected_q,
            align_mode=align_mode,
            mask_target_audio=mask_target_audio,
            hard_max_frames=hard_max_frames,
        )

        if (
            not _suppress_length_warning
            and max_context_frames > 0
            and codes.shape[1] > max_context_frames
        ):
            logger.warning(
                "생성된 샘플 길이(%d 프레임)가 max_context_frames(%d)를 초과합니다 (%s). "
                "학습 시 EOS 절단을 피하려면 target_window_sec를 줄이세요.",
                codes.shape[1],
                max_context_frames,
                lecture_wav_path.name,
            )

        if mask_target_audio:
            sim_bin = resolve_default_sim_bin("text-only", num_speaker_sim_bins)
        else:
            sim_bin = validate_sim_bin(
                wavlm_sim_bin_fn(speaker_prompt_24k, ko_speech_only),
                num_speaker_sim_bins,
            )

        if _out_metadata is not None:
            _out_metadata.clear()
            _out_metadata.update(built_meta)
        return codes, sim_bin


@dataclass
class _SttMeta71379:
    """71379 원천 한국어 음성전사 JSON에서 추출한 부가 타임스탬프 및 비식별화 정보입니다."""

    ko_text: str
    raw_start_sec: Optional[float]
    raw_end_sec: Optional[float]
    deidentified: bool


def load_and_group_71379_json(
    json_dir: Path,
    en_accent: str = "all",
    use_mt_fallback: bool = False,
) -> dict[str, List[BroadcastUtterance71379]]:
    """AI-Hub 71379 라벨링·전사 JSON을 결합하여 씬 및 영어 억양 단위로 발화들을 정렬·그룹화합니다."""
    if en_accent not in ("all", "us", "uk"):
        raise ValueError(f"en_accent는 'all', 'us', 'uk' 중 하나여야 합니다: {en_accent}")

    stt_by_stem: dict[str, _SttMeta71379] = {}
    deidentified_stems: set[str] = set()
    raw_trans_items: List[Tuple[Path, dict[str, Any]]] = []

    for jp in sorted(json_dir.rglob("*.json")):
        lower_name = jp.name.lower()
        if lower_name.endswith(("_es.json", "_ru.json")):
            continue
        try:
            payload = json.loads(jp.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("JSON 파싱 실패로 건너뜁니다 (%s): %s", jp, exc)
            continue

        for raw_rec in extract_json_records(payload):
            _, item = normalize_record_keys(raw_rec)
            if is_71693_record(item):
                continue

            if "음성파일명" in item and "t-fname" not in item and "t-code" not in item:
                ko_fname = clean_transcript_text(item.get("음성파일명"))
                if not ko_fname:
                    continue
                stem_key = Path(ko_fname).stem.lower()
                is_deid = str(item.get("비식별화") or "").strip().upper() == "O"
                if is_deid:
                    deidentified_stems.add(stem_key)

                ko_stt_text = (
                    clean_transcript_text(item.get("번역본"))
                    or clean_transcript_text(item.get("철자전사"))
                    or clean_transcript_text(item.get("발음전사"))
                    or clean_transcript_text(item.get("전사작업본"))
                )
                raw_s: Optional[float] = None
                raw_e: Optional[float] = None
                try:
                    if item.get("원시시작") is not None and item.get("원시끝") is not None:
                        raw_s = parse_hhmmss(item["원시시작"])
                        raw_e = parse_hhmmss(item["원시끝"])
                except (ValueError, TypeError):
                    raw_s, raw_e = None, None

                stt_by_stem[stem_key] = _SttMeta71379(
                    ko_text=ko_stt_text,
                    raw_start_sec=raw_s,
                    raw_end_sec=raw_e,
                    deidentified=is_deid,
                )
                continue

            if any(k in item for k in ("t-fname", "s-fname", "t-code", "최종번역문", "최종번역")):
                raw_trans_items.append((jp, item))

    grouped: dict[str, List[BroadcastUtterance71379]] = defaultdict(list)
    skipped_seqs_by_group: dict[str, set[int]] = defaultdict(set)
    seen_ids: set[str] = set()

    for jp, item in raw_trans_items:
        s_code = str(item.get("s-code") or "ko-KR").strip().lower()
        if s_code and not s_code.startswith("ko"):
            continue

        ko_wav_name = clean_transcript_text(item.get("s-fname"))
        en_wav_name = clean_transcript_text(item.get("t-fname"))
        t_code = str(item.get("t-code") or "").strip()

        variant = classify_71379_accent(t_code, en_wav_name, jp.name)
        if variant is None:
            continue
        if en_accent != "all" and variant != en_accent:
            continue

        if not en_wav_name and ko_wav_name:
            sfx = "_uk.wav" if variant == "uk" else "_en.wav"
            en_wav_name = f"{Path(ko_wav_name).stem}{sfx}"
        if not en_wav_name:
            continue

        ko_stem = Path(ko_wav_name).stem.lower() if ko_wav_name else ""
        stt_meta = stt_by_stem.get(ko_stem) if ko_stem else None

        scene_key, seq_num = parse_71379_scene_and_seq(ko_wav_name, en_wav_name, item)
        group_key = f"{scene_key}_{variant}"
        utt_id = Path(en_wav_name).stem

        if utt_id.lower() in seen_ids:
            continue
        seen_ids.add(utt_id.lower())

        if (stt_meta is not None and stt_meta.deidentified) or (ko_stem in deidentified_stems):
            skipped_seqs_by_group[group_key].add(seq_num)
            continue

        ko_text = clean_transcript_text(item.get("수정원문")) or clean_transcript_text(item.get("원문"))
        if not ko_text and stt_meta is not None:
            ko_text = stt_meta.ko_text

        en_text = (
            clean_transcript_text(item.get("수정번역문"))
            or clean_transcript_text(item.get("최종번역문"))
            or clean_transcript_text(item.get("최종번역"))
            or clean_transcript_text(item.get("2차수정"))
            or clean_transcript_text(item.get("1차수정"))
        )
        if not en_text and use_mt_fallback:
            en_text = clean_transcript_text(item.get("mt"))

        if not ko_text or not en_text:
            skipped_seqs_by_group[group_key].add(seq_num)
            continue

        try:
            s_ulen = float(item.get("s-ulength") or 0.0)
        except (TypeError, ValueError):
            s_ulen = 0.0
        try:
            t_ulen = float(item.get("t-ulength") or 0.0)
        except (TypeError, ValueError):
            t_ulen = 0.0

        grouped[group_key].append(
            BroadcastUtterance71379(
                utt_id=utt_id,
                scene_key=scene_key,
                seq_num=seq_num,
                lang_variant=variant,
                ko_wav_name=ko_wav_name,
                en_wav_name=en_wav_name,
                ko_text=ko_text,
                en_text=en_text,
                speaker_id=clean_transcript_text(item.get("발화자id")),
                s_sex=clean_transcript_text(item.get("s-usex")).lower(),
                t_sex=clean_transcript_text(item.get("t-usex")).lower(),
                s_ulength=max(0.0, s_ulen),
                t_ulength=max(0.0, t_ulen),
                raw_start_sec=stt_meta.raw_start_sec if stt_meta else None,
                raw_end_sec=stt_meta.raw_end_sec if stt_meta else None,
            )
        )

    for group_key, recs in grouped.items():
        mark_71379_segment_breaks(recs, skipped_seqs_by_group.get(group_key))
    return grouped


def build_EDarch_71379_training_example(
    window_records: List[BroadcastUtterance71379],
    en_clip_paths: List[Path],
    ko_clip_paths: List[Optional[Path]],
    ko_sp_encode_fn: Callable[[str], List[int]],
    mimi_encode_fn: Callable[[torch.Tensor], torch.Tensor],
    wavlm_sim_bin_fn: Optional[Callable[[torch.Tensor, torch.Tensor], int]] = None,
    zero_shot_ko_tts_fn: Optional[
        Callable[..., Union[torch.Tensor, Tuple[torch.Tensor, int]]]
    ] = None,
    speaker_prompt_24k: Optional[torch.Tensor] = None,
    ko_audio_mode: str = "auto",
    inter_utt_pause_sec: float = 0.5,
    max_gap_sec: float = 8.0,
    max_overlap_sec: float = 4.0,
    expected_q: Optional[int] = None,
    max_context_frames: int = DEFAULT_CONTEXT_FRAMES,
    text_card: Optional[int] = None,
    num_speaker_sim_bins: int = 5,
    fallback_real_sim_bin: Optional[int] = None,
    fallback_tts_sim_bin: Optional[int] = None,
    align_mode: str = "auto",
    ko_align_fn: Optional[KoAlignFn] = None,
    ko_sp_pieces_fn: Optional[Callable[[str], List[str]]] = None,
    monologue_lead_frames: int = 1,
    hard_max_frames: Optional[int] = None,
    _out_metadata: Optional[dict[str, Any]] = None,
    _clip_cache: Optional[dict[str, Tuple[torch.Tensor, torch.Tensor, List[int], bool]]] = None,
    _suppress_length_warning: bool = False,
) -> Tuple[torch.Tensor, int]:
    """AI-Hub 71379 발화 윈도우를 인과적 정밀 정렬이 적용된 `[1 + 2Q, T]` 코드북 텐서로 변환합니다."""
    if not window_records:
        raise ValueError("window_records는 비어 있을 수 없습니다.")
    if len(window_records) != len(en_clip_paths) or len(window_records) != len(ko_clip_paths):
        raise ValueError("window_records와 WAV 경로 리스트의 길이가 일치해야 합니다.")
    if ko_audio_mode not in ("auto", "real", "tts"):
        raise ValueError(f"ko_audio_mode는 'auto', 'real', 'tts' 중 하나여야 합니다: {ko_audio_mode}")

    with torch.no_grad():
        if speaker_prompt_24k is None:
            speaker_prompt_24k = extract_prompt_from_clip_paths(en_clip_paths, prompt_sec=10.0)
        else:
            speaker_prompt_24k = speaker_prompt_24k.detach().cpu().float().reshape(-1)

        en_placements: List[Tuple[int, torch.Tensor]] = []
        ko_placements: List[Tuple[int, torch.Tensor, List[Tuple[int, int]]]] = []
        src_start_frames: List[int] = []
        src_end_frames: List[int] = []
        ref_token_groups: List[List[int]] = []
        used_real_ko = False

        prev_en_end_sample = 0
        ko_cursor_sample = 0
        last_token_frame = -1
        max_overlap_samples = max(0, int(max_overlap_sec * SAMPLE_RATE))

        for i, (rec, en_path, ko_path) in enumerate(
            zip(window_records, en_clip_paths, ko_clip_paths)
        ):
            cache_key = f"{rec.utt_id}:{ko_audio_mode}"
            cached = _clip_cache.get(cache_key) if _clip_cache is not None else None
            if cached is not None:
                en_wav, ko_wav, ko_token_ids, is_real_ko = cached
            else:
                en_wav = load_mono_wav_24k(en_path).detach().cpu().float().reshape(-1)
                if en_wav.numel() == 0:
                    raise ValueError(f"비어 있는 영어 WAV 파일입니다: {en_path}")

                use_real = ko_audio_mode in ("auto", "real") and ko_path is not None
                if ko_audio_mode == "real" and ko_path is None:
                    raise FileNotFoundError(
                        f"ko_audio_mode='real'이지만 한국어 원천 WAV를 찾을 수 없습니다: {rec.ko_wav_name}"
                    )
                if use_real:
                    assert ko_path is not None
                    ko_wav = load_mono_wav_24k(ko_path).detach().cpu().float().reshape(-1)
                    if ko_wav.numel() == 0:
                        raise ValueError(f"비어 있는 한국어 WAV 파일입니다: {ko_path}")
                    is_real_ko = True
                else:
                    if zero_shot_ko_tts_fn is None:
                        raise ValueError(
                            f"한국어 원천 WAV({rec.ko_wav_name})가 없거나 ko_audio_mode='tts'인데 "
                            "zero_shot_ko_tts_fn이 지정되지 않았습니다."
                        )
                    ko_wav = _call_tts_fn(
                        zero_shot_ko_tts_fn, rec.ko_text, speaker_prompt_24k, rec.t_sex or rec.s_sex
                    )
                    is_real_ko = False

                ko_token_ids = list(ko_sp_encode_fn(rec.ko_text))
                if _clip_cache is not None:
                    _clip_cache[cache_key] = (en_wav, ko_wav, ko_token_ids, is_real_ko)

            validate_ko_token_ids(ko_token_ids, text_card)
            tok_pieces = (
                list(ko_sp_pieces_fn(rec.ko_text)) if ko_sp_pieces_fn is not None else None
            )
            if is_real_ko:
                used_real_ko = True

            if i == 0:
                en_start_sample = 0
            else:
                prev_rec = window_records[i - 1]
                pause_sec = inter_utt_pause_sec
                if prev_rec.raw_end_sec is not None and rec.raw_start_sec is not None:
                    raw_gap = rec.raw_start_sec - prev_rec.raw_end_sec
                    if raw_gap >= 0.0:
                        pause_sec = min(max(raw_gap, inter_utt_pause_sec), max_gap_sec)
                pause_samples = int(pause_sec * SAMPLE_RATE)
                en_start_sample = max(
                    prev_en_end_sample + pause_samples,
                    ko_cursor_sample - max_overlap_samples,
                )

            en_end_sample = en_start_sample + en_wav.numel()
            en_placements.append((en_start_sample, en_wav))
            prev_en_end_sample = en_end_sample

            en_start_frame = en_start_sample // SAMPLES_PER_FRAME
            en_end_frame = (en_end_sample + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME
            prev_ko_end_frame = (ko_cursor_sample + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME

            ko_audio_start_sample, token_positions, last_token_frame = compute_aligned_token_placement(
                ko_text=rec.ko_text,
                ko_wav_24k=ko_wav,
                ko_token_ids=ko_token_ids,
                en_end_frame=en_end_frame,
                prev_audio_end_frame=prev_ko_end_frame,
                last_token_frame=last_token_frame,
                align_mode=align_mode,
                ko_align_fn=ko_align_fn,
                token_pieces=tok_pieces,
                monologue_lead_frames=monologue_lead_frames,
            )

            ko_placements.append((ko_audio_start_sample, ko_wav, token_positions))
            src_start_frames.append(int(en_start_frame))
            src_end_frames.append(int(en_end_frame))
            ref_token_groups.append([int(tid) for tid in ko_token_ids])
            ko_cursor_sample = ko_audio_start_sample + ko_wav.numel()

        codes, ko_speech_only, built_meta = assemble_multistream_example(
            en_placements=en_placements,
            ko_placements=ko_placements,
            en_span_end_sample=prev_en_end_sample,
            ko_end_sample=ko_cursor_sample,
            last_token_frame=last_token_frame,
            src_start_frames=src_start_frames,
            src_end_frames=src_end_frames,
            ref_token_groups=ref_token_groups,
            mimi_encode_fn=mimi_encode_fn,
            expected_q=expected_q,
            align_mode=align_mode,
            mask_target_audio=False,
            hard_max_frames=hard_max_frames,
        )

        if (
            not _suppress_length_warning
            and max_context_frames > 0
            and codes.shape[1] > max_context_frames
        ):
            logger.warning(
                "71379 샘플 길이(%d 프레임)가 max_context_frames(%d)를 초과합니다 (%s).",
                codes.shape[1],
                max_context_frames,
                window_records[0].utt_id,
            )

        if wavlm_sim_bin_fn is not None:
            raw_sim = wavlm_sim_bin_fn(speaker_prompt_24k, ko_speech_only)
            sim_bin = validate_sim_bin(raw_sim, num_speaker_sim_bins)
        elif used_real_ko:
            first_rec = window_records[0]
            same_sex = bool(
                first_rec.s_sex and first_rec.t_sex and first_rec.s_sex == first_rec.t_sex
            )
            sim_bin = resolve_default_sim_bin(
                "real_pair",
                num_speaker_sim_bins=num_speaker_sim_bins,
                same_sex=same_sex,
                explicit_sim_bin=fallback_real_sim_bin,
            )
        else:
            sim_bin = resolve_default_sim_bin(
                "cloned_tts",
                num_speaker_sim_bins=num_speaker_sim_bins,
                explicit_sim_bin=fallback_tts_sim_bin,
            )

        if _out_metadata is not None:
            _out_metadata.clear()
            _out_metadata.update(built_meta)
        return codes, sim_bin


def preprocess_71693_dataset(
    json_dir: Path,
    wav_dir: Path,
    output_dir: Path,
    zero_shot_ko_tts_fn: Optional[
        Callable[..., Union[torch.Tensor, Tuple[torch.Tensor, int]]]
    ],
    ko_sp_encode_fn: Callable[[str], List[int]],
    mimi_encode_fn: Callable[[torch.Tensor], torch.Tensor],
    wavlm_sim_bin_fn: Callable[[torch.Tensor, torch.Tensor], int],
    target_window_sec: float = 45.0,
    max_gap_sec: float = 8.0,
    expected_q: Optional[int] = None,
    max_context_frames: int = DEFAULT_CONTEXT_FRAMES,
    text_card: Optional[int] = None,
    num_speaker_sim_bins: int = 5,
    use_mt_fallback: bool = False,
    file_prefix: str = "",
    skip_existing: bool = False,
    align_mode: str = "auto",
    ko_align_fn: Optional[KoAlignFn] = None,
    ko_sp_pieces_fn: Optional[Callable[[str], List[str]]] = None,
    monologue_lead_frames: int = 1,
    mask_target_audio: bool = False,
) -> int:
    """AI-Hub 71693 전체 디렉터리를 순회하며 적응형 윈도우 분할 및 `.pt` 배치 저장을 수행합니다."""
    if max_context_frames > 0 and target_window_sec * FRAME_RATE >= max_context_frames:
        max_win_sec = max_context_frames / FRAME_RATE
        logger.warning(
            "target_window_sec(%.1fs)가 max_context_frames(%d = %.1fs) 이상입니다. "
            "초과 윈도우는 발화 단위로 자동 분할되지만 target_window_sec를 %.1fs 이하로 설정하는 것을 권장합니다.",
            target_window_sec,
            max_context_frames,
            max_win_sec,
            max_win_sec * 0.6,
        )

    grouped = load_and_group_71693_json(json_dir, use_mt_fallback=use_mt_fallback)
    if not grouped:
        raise ValueError(f"{json_dir}에서 유효한 71693 en->ko 발화 레코드를 찾지 못했습니다.")

    wav_index, _ = index_wav_files(wav_dir)
    existing_stems, split_ancestors = (
        index_existing_window_trees(output_dir) if skip_existing else (set(), set())
    )
    saved_count = 0

    for file_name, spans in sorted(grouped.items()):
        windows = slice_71693_windows(
            spans, target_window_sec=target_window_sec, max_gap_sec=max_gap_sec
        )
        stem = Path(file_name).stem
        if skip_existing and windows:
            lecture_leaf_counts = [
                count_completed_window_tree(
                    f"{file_prefix}{stem}_win{w_idx:04d}",
                    len(w_spans),
                    existing_stems,
                    branch_tags=("_a", "_b"),
                )
                for w_idx, w_spans in enumerate(windows)
            ]
            if all(cnt is not None for cnt in lecture_leaf_counts):
                saved_count += sum(cnt for cnt in lecture_leaf_counts if cnt is not None)
                continue

        wav_path = resolve_wav_path(wav_dir, file_name, wav_index)
        wav_24k: Optional[torch.Tensor] = None
        if wav_path is not None:
            wav_24k = load_mono_wav_24k(wav_path)
        else:
            wav_24k = _assemble_virtual_lecture_from_sn_clips(spans, wav_dir, wav_index)
            if wav_24k is not None:
                wav_path = wav_dir / file_name

        if wav_path is None or wav_24k is None:
            logger.warning("71693 원본 WAV 파일을 찾지 못해 건너뜁니다: %s", file_name)
            continue

        speaker_prompt = extract_10s_speaker_prompt(wav_24k, spans, prompt_sec=10.0)
        tts_cache: dict[str, Tuple[torch.Tensor, List[int]]] = {}
        work_queue: deque[Tuple[str, List[UtteranceSpan]]] = deque(
            (f"{file_prefix}{stem}_win{w_idx:04d}", w_spans)
            for w_idx, w_spans in enumerate(windows)
        )

        while work_queue:
            cur_stem, win_spans = work_queue.popleft()
            if skip_existing:
                done_leaves = count_completed_window_tree(
                    cur_stem, len(win_spans), existing_stems, branch_tags=("_a", "_b")
                )
                if done_leaves is not None:
                    saved_count += done_leaves
                    continue
                if len(win_spans) > 1 and cur_stem in split_ancestors:
                    mid = len(win_spans) // 2
                    work_queue.appendleft((f"{cur_stem}_b", win_spans[mid:]))
                    work_queue.appendleft((f"{cur_stem}_a", win_spans[:mid]))
                    continue

            can_split = max_context_frames > 0 and len(win_spans) > 1
            leaf_hard_cap = (
                max_context_frames if (max_context_frames > 0 and not can_split) else None
            )
            meta_dict: dict[str, Any] = {}
            codes, sim_bin = build_EDarch_training_example(
                lecture_wav_path=wav_path,
                window_spans=win_spans,
                all_lecture_spans=spans,
                zero_shot_ko_tts_fn=zero_shot_ko_tts_fn,
                ko_sp_encode_fn=ko_sp_encode_fn,
                mimi_encode_fn=mimi_encode_fn,
                wavlm_sim_bin_fn=wavlm_sim_bin_fn,
                preloaded_wav_24k=wav_24k,
                speaker_prompt_24k=speaker_prompt,
                expected_q=expected_q,
                max_context_frames=max_context_frames,
                text_card=text_card,
                num_speaker_sim_bins=num_speaker_sim_bins,
                align_mode=align_mode,
                ko_align_fn=ko_align_fn,
                ko_sp_pieces_fn=ko_sp_pieces_fn,
                monologue_lead_frames=monologue_lead_frames,
                mask_target_audio=mask_target_audio,
                hard_max_frames=leaf_hard_cap,
                _out_metadata=meta_dict,
                _tts_tok_cache=tts_cache,
                _suppress_length_warning=can_split,
            )
            if can_split and codes.shape[1] > max_context_frames:
                mid = len(win_spans) // 2
                logger.info(
                    "71693 윈도우 길이(%d 프레임)가 max_context_frames(%d)를 초과하여 %d+%d개 발화로 분할합니다 (%s).",
                    codes.shape[1],
                    max_context_frames,
                    mid,
                    len(win_spans) - mid,
                    wav_path.name,
                )
                work_queue.appendleft((f"{cur_stem}_b", win_spans[mid:]))
                work_queue.appendleft((f"{cur_stem}_a", win_spans[:mid]))
                continue

            out_pt = output_dir / f"{cur_stem}.pt"
            if not (skip_existing and out_pt.exists()):
                save_training_example_pt(out_pt, codes, sim_bin, metadata=meta_dict)
                existing_stems.add(cur_stem)
            saved_count += 1

    return saved_count


def preprocess_71379_dataset(
    json_dir: Path,
    output_dir: Path,
    ko_sp_encode_fn: Callable[[str], List[int]],
    mimi_encode_fn: Callable[[torch.Tensor], torch.Tensor],
    wav_dir: Optional[Path] = None,
    en_wav_dir: Optional[Path] = None,
    ko_wav_dir: Optional[Path] = None,
    wavlm_sim_bin_fn: Optional[Callable[[torch.Tensor, torch.Tensor], int]] = None,
    zero_shot_ko_tts_fn: Optional[
        Callable[..., Union[torch.Tensor, Tuple[torch.Tensor, int]]]
    ] = None,
    ko_audio_mode: str = "auto",
    en_accent: str = "all",
    target_window_sec: float = 45.0,
    max_gap_sec: float = 8.0,
    inter_utt_pause_sec: float = 0.5,
    max_overlap_sec: float = 4.0,
    expected_q: Optional[int] = None,
    max_context_frames: int = DEFAULT_CONTEXT_FRAMES,
    text_card: Optional[int] = None,
    num_speaker_sim_bins: int = 5,
    fallback_real_sim_bin: Optional[int] = None,
    use_mt_fallback: bool = False,
    file_prefix: str = "71379_",
    skip_existing: bool = False,
    align_mode: str = "auto",
    ko_align_fn: Optional[KoAlignFn] = None,
    ko_sp_pieces_fn: Optional[Callable[[str], List[str]]] = None,
    monologue_lead_frames: int = 1,
) -> int:
    """AI-Hub 71379 방송콘텐츠 데이터셋 전체를 순회하며 윈도우 분할 및 `.pt` 배치 저장을 수행합니다."""
    if en_wav_dir is None and wav_dir is None:
        raise ValueError("71379 전처리를 위해 wav_dir 또는 en_wav_dir 중 하나는 지정되어야 합니다.")

    grouped = load_and_group_71379_json(
        json_dir=json_dir,
        en_accent=en_accent,
        use_mt_fallback=use_mt_fallback,
    )
    if not grouped:
        raise ValueError(f"{json_dir}에서 유효한 71379 한-영 통번역 레코드를 찾지 못했습니다.")

    en_search_dir = en_wav_dir or wav_dir
    ko_search_dir = ko_wav_dir or wav_dir
    en_wav_index, _ = index_wav_files(
        en_search_dir,
        wav_dir,
        prefer_keywords=("원어민", "발화", "영어", "영국", "en", "eng", "uk", "target"),
    )
    ko_wav_index, _ = index_wav_files(
        ko_search_dir,
        wav_dir,
        prefer_keywords=("원천", "한국어", "한글", "ko", "kor", "source"),
    )

    speaker_en_paths: dict[str, List[Path]] = defaultdict(list)
    resolved_groups: dict[str, List[Tuple[BroadcastUtterance71379, Path, Optional[Path]]]] = {}

    for group_key, recs in sorted(grouped.items()):
        valid_triples: List[Tuple[BroadcastUtterance71379, Path, Optional[Path]]] = []
        for idx, rec in enumerate(recs):
            en_path = resolve_wav_path(en_search_dir, rec.en_wav_name, en_wav_index)
            if en_path is None:
                logger.warning("71379 원어민 영어 WAV 누락으로 건너뜁니다: %s", rec.en_wav_name)
                if idx + 1 < len(recs):
                    recs[idx + 1].segment_break = True
                continue

            ko_path = (
                resolve_wav_path(ko_search_dir, rec.ko_wav_name, ko_wav_index)
                if rec.ko_wav_name
                else None
            )
            if ko_audio_mode == "real" and ko_path is None:
                logger.warning("71379 한국어 원천 WAV 누락으로 건너뜁니다: %s", rec.ko_wav_name)
                if idx + 1 < len(recs):
                    recs[idx + 1].segment_break = True
                continue
            if ko_audio_mode == "auto" and ko_path is None and zero_shot_ko_tts_fn is None:
                logger.warning(
                    "71379 한국어 원천 WAV(%s)가 없고 TTS 훅도 없어 건너뜁니다.",
                    rec.ko_wav_name,
                )
                if idx + 1 < len(recs):
                    recs[idx + 1].segment_break = True
                continue

            valid_triples.append((rec, en_path, ko_path))
            spk_cache_key = f"{rec.speaker_id}:{rec.t_sex}" if rec.speaker_id else ""
            if spk_cache_key and len(speaker_en_paths[spk_cache_key]) < 8:
                speaker_en_paths[spk_cache_key].append(en_path)

        if valid_triples:
            resolved_groups[group_key] = valid_triples

    if not resolved_groups:
        raise ValueError("71379 JSON에 매칭되는 유효한 WAV 파일을 하나도 찾지 못했습니다.")

    speaker_prompt_cache: dict[str, torch.Tensor] = {}
    existing_stems, split_ancestors = (
        index_existing_window_trees(output_dir) if skip_existing else (set(), set())
    )
    saved_count = 0

    for group_key, triples in sorted(resolved_groups.items()):
        recs_only = [t[0] for t in triples]
        path_by_id = {t[0].utt_id: (t[1], t[2]) for t in triples}
        windows = slice_71379_windows(
            recs_only,
            target_window_sec=target_window_sec,
            max_gap_sec=max_gap_sec,
            inter_utt_pause_sec=inter_utt_pause_sec,
            min_est_dur_sec=3.0,
        )
        if skip_existing and windows:
            group_leaf_counts = [
                count_completed_window_tree(
                    f"{file_prefix}{group_key}_win{w_idx:04d}",
                    len(w_recs),
                    existing_stems,
                    branch_tags=("_a", "_b"),
                )
                for w_idx, w_recs in enumerate(windows)
            ]
            if all(cnt is not None for cnt in group_leaf_counts):
                saved_count += sum(cnt for cnt in group_leaf_counts if cnt is not None)
                continue

        clip_cache: dict[str, Tuple[torch.Tensor, torch.Tensor, List[int], bool]] = {}
        work_queue: deque[Tuple[str, List[BroadcastUtterance71379]]] = deque(
            (f"{file_prefix}{group_key}_win{w_idx:04d}", w_recs)
            for w_idx, w_recs in enumerate(windows)
        )

        while work_queue:
            cur_stem, win_recs = work_queue.popleft()
            if skip_existing:
                done_leaves = count_completed_window_tree(
                    cur_stem, len(win_recs), existing_stems, branch_tags=("_a", "_b")
                )
                if done_leaves is not None:
                    saved_count += done_leaves
                    continue
                if len(win_recs) > 1 and cur_stem in split_ancestors:
                    mid = len(win_recs) // 2
                    work_queue.appendleft((f"{cur_stem}_b", win_recs[mid:]))
                    work_queue.appendleft((f"{cur_stem}_a", win_recs[:mid]))
                    continue

            win_en_paths = [path_by_id[r.utt_id][0] for r in win_recs]
            win_ko_paths = [path_by_id[r.utt_id][1] for r in win_recs]

            first_r = win_recs[0]
            spk_cache_key = f"{first_r.speaker_id}:{first_r.t_sex}" if first_r.speaker_id else ""
            prompt_24k: Optional[torch.Tensor] = None
            if spk_cache_key:
                prompt_24k = speaker_prompt_cache.get(spk_cache_key)
                if prompt_24k is None and spk_cache_key in speaker_en_paths:
                    prompt_24k = extract_prompt_from_clip_paths(
                        speaker_en_paths[spk_cache_key], prompt_sec=10.0
                    )
                    speaker_prompt_cache[spk_cache_key] = prompt_24k
            if prompt_24k is None:
                prompt_24k = extract_prompt_from_clip_paths(win_en_paths, prompt_sec=10.0)

            can_split = max_context_frames > 0 and len(win_recs) > 1
            leaf_hard_cap = (
                max_context_frames if (max_context_frames > 0 and not can_split) else None
            )
            meta_dict: dict[str, Any] = {}
            try:
                codes, sim_bin = build_EDarch_71379_training_example(
                    window_records=win_recs,
                    en_clip_paths=win_en_paths,
                    ko_clip_paths=win_ko_paths,
                    ko_sp_encode_fn=ko_sp_encode_fn,
                    mimi_encode_fn=mimi_encode_fn,
                    wavlm_sim_bin_fn=wavlm_sim_bin_fn,
                    zero_shot_ko_tts_fn=zero_shot_ko_tts_fn,
                    speaker_prompt_24k=prompt_24k,
                    ko_audio_mode=ko_audio_mode,
                    inter_utt_pause_sec=inter_utt_pause_sec,
                    max_gap_sec=max_gap_sec,
                    max_overlap_sec=max_overlap_sec,
                    expected_q=expected_q,
                    max_context_frames=max_context_frames,
                    text_card=text_card,
                    num_speaker_sim_bins=num_speaker_sim_bins,
                    fallback_real_sim_bin=fallback_real_sim_bin,
                    align_mode=align_mode,
                    ko_align_fn=ko_align_fn,
                    ko_sp_pieces_fn=ko_sp_pieces_fn,
                    monologue_lead_frames=monologue_lead_frames,
                    hard_max_frames=leaf_hard_cap,
                    _out_metadata=meta_dict,
                    _clip_cache=clip_cache,
                    _suppress_length_warning=can_split,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                if len(win_recs) > 1:
                    mid = len(win_recs) // 2
                    work_queue.appendleft((f"{cur_stem}_b", win_recs[mid:]))
                    work_queue.appendleft((f"{cur_stem}_a", win_recs[:mid]))
                else:
                    logger.warning("71379 발화 변환 실패로 건너뜁니다 (%s): %s", win_recs[0].utt_id, exc)
                continue

            if can_split and codes.shape[1] > max_context_frames:
                mid = len(win_recs) // 2
                logger.info(
                    "71379 윈도우 길이(%d 프레임)가 max_context_frames(%d)를 초과하여 %d+%d개 발화로 분할합니다 (%s).",
                    codes.shape[1],
                    max_context_frames,
                    mid,
                    len(win_recs) - mid,
                    group_key,
                )
                work_queue.appendleft((f"{cur_stem}_b", win_recs[mid:]))
                work_queue.appendleft((f"{cur_stem}_a", win_recs[:mid]))
                continue

            out_pt = output_dir / f"{cur_stem}.pt"
            if not (skip_existing and out_pt.exists()):
                save_training_example_pt(out_pt, codes, sim_bin, metadata=meta_dict)
                existing_stems.add(cur_stem)
            saved_count += 1

    return saved_count


def detect_datasets_in_json_dir(json_dir: Path) -> Tuple[bool, bool]:
    """디렉터리 내 JSON 파일들을 샘플링하여 71693 및 71379 스키마 포함 여부를 자동 감지합니다."""
    has_71693 = False
    has_71379 = False
    for jp in sorted(json_dir.rglob("*.json")):
        try:
            payload = json.loads(jp.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        for raw_rec in extract_json_records(payload)[:5]:
            _, item = normalize_record_keys(raw_rec)
            if is_71693_record(item):
                has_71693 = True
            elif any(
                k in item for k in ("t-fname", "s-fname", "t-code", "최종번역문", "최종번역", "음성파일명")
            ):
                has_71379 = True
            if has_71693 and has_71379:
                return True, True
    return has_71693, has_71379


def collect_korean_corpus(
    json_dirs_71693: Iterable[Path] = (),
    json_dirs_71379: Iterable[Path] = (),
    use_mt_fallback: bool = False,
    en_accent_71379: str = "all",
) -> List[str]:
    """71693 및 71379 라벨 디렉터리들에서 토크나이저 구축용 고유 한국어 문장 리스트를 수집합니다."""
    sentences: List[str] = []
    seen: set[str] = set()

    for jd in json_dirs_71693:
        grouped_93 = load_and_group_71693_json(jd, use_mt_fallback=use_mt_fallback)
        for spans in grouped_93.values():
            for sp in spans:
                if sp.ko_mtpe and sp.ko_mtpe not in seen:
                    seen.add(sp.ko_mtpe)
                    sentences.append(sp.ko_mtpe)

    for jd in json_dirs_71379:
        grouped_79 = load_and_group_71379_json(
            jd, en_accent=en_accent_71379, use_mt_fallback=use_mt_fallback
        )
        for recs in grouped_79.values():
            for rec in recs:
                if rec.ko_text and rec.ko_text not in seen:
                    seen.add(rec.ko_text)
                    sentences.append(rec.ko_text)

    return sentences


def _resolve_cli_tasks(
    args: argparse.Namespace,
) -> Tuple[
    Optional[Tuple[Path, Path]],
    Optional[Tuple[Path, Optional[Path], Optional[Path], Optional[Path]]],
]:
    """입력 경로 CLI 인자들을 해석하여 71693 및 71379 전처리 작업 대상 경로를 결정합니다."""
    task_71693: Optional[Tuple[Path, Path]] = None
    task_71379: Optional[Tuple[Path, Optional[Path], Optional[Path], Optional[Path]]] = None

    j93 = args.json_dir_71693 or args.root_71693
    w93 = args.wav_dir_71693 or args.root_71693
    if j93 is not None or w93 is not None:
        if j93 is None or w93 is None:
            raise ValueError(
                "71693 경로 지정 시 --root-71693 또는 (--json-dir-71693 및 --wav-dir-71693)가 필요합니다."
            )
        task_71693 = (j93, w93)

    j79 = args.json_dir_71379 or args.root_71379
    w79 = args.wav_dir_71379 or args.root_71379
    en_w79 = args.en_wav_dir_71379
    ko_w79 = args.ko_wav_dir_71379
    if any(x is not None for x in (j79, w79, en_w79, ko_w79)):
        if j79 is None or (w79 is None and en_w79 is None):
            raise ValueError(
                "71379 경로 지정 시 --root-71379 또는 (--json-dir-71379 및 --wav-dir-71379/--en-wav-dir-71379)가 필요합니다."
            )
        task_71379 = (j79, w79, en_w79, ko_w79)

    if args.json_dir is not None or args.wav_dir is not None:
        if args.json_dir is None or args.wav_dir is None:
            raise ValueError("--json-dir와 --wav-dir는 함께 지정해야 합니다.")
        if args.dataset == "71693":
            task_71693 = task_71693 or (args.json_dir, args.wav_dir)
        elif args.dataset == "71379":
            task_71379 = task_71379 or (args.json_dir, args.wav_dir, None, None)
        else:
            has_93, has_79 = detect_datasets_in_json_dir(args.json_dir)
            if has_93:
                task_71693 = task_71693 or (args.json_dir, args.wav_dir)
            if has_79:
                task_71379 = task_71379 or (args.json_dir, args.wav_dir, None, None)
            if not has_93 and not has_79:
                raise ValueError(
                    f"{args.json_dir}에서 71693 또는 71379 형식의 JSON 파일을 감지하지 못했습니다."
                )

    if task_71693 is None and task_71379 is None:
        raise ValueError(
            "전처리할 입력 경로가 지정되지 않았습니다. "
            "--root-71693 / --root-71379 또는 --json-dir / --wav-dir를 지정하세요."
        )
    return task_71693, task_71379


def run_preprocessing(args: argparse.Namespace) -> None:
    """토크나이저·정렬기·코덱·TTS 훅을 준비하고 AI-Hub 71693 및 71379 데이터셋 전처리를 통합 실행합니다."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    task_71693, task_71379 = _resolve_cli_tasks(args)

    device = setup_device(cpu_only=args.cpu, seed=args.seed, tag="Preprocess")
    cfg = EDarchConfig.from_preset(args.preset)
    if args.ko_vocab_size is not None:
        cfg.text_card = args.ko_vocab_size

    hooks_mod = importlib.import_module(args.hooks_module) if args.hooks_module else None
    tts_fn = getattr(hooks_mod, "zero_shot_ko_tts_fn", None) if hooks_mod else None
    sp_fn = getattr(hooks_mod, "ko_sp_encode_fn", None) if hooks_mod else None
    sp_pieces_fn = getattr(hooks_mod, "ko_sp_pieces_fn", None) if hooks_mod else None
    align_fn = getattr(hooks_mod, "ko_align_fn", None) if hooks_mod else None
    mimi_fn = getattr(hooks_mod, "mimi_encode_fn", None) if hooks_mod else None
    sim_fn = getattr(hooks_mod, "wavlm_sim_bin_fn", None) if hooks_mod else None

    ko_mode_93 = getattr(args, "ko_mode_71693", "hooks")
    mask_target_93 = ko_mode_93 == "text-only"
    if tts_fn is None and not mask_target_93:
        if ko_mode_93 == "mms":
            mms_id = getattr(args, "mms_tts_model", "facebook/mms-tts-kor")
            tts_fn = build_mms_korean_tts_fn(device=device, model_id=mms_id)
        elif ko_mode_93 == "edge-tts":
            tts_fn = build_edge_korean_tts_fn(cache_dir=args.output_dir / ".edge_tts_cache")

    if sp_fn is None and (args.sp_model is not None or args.train_sp_if_missing):
        raw_tok_path = (
            args.sp_model
            if args.sp_model is not None
            else (args.output_dir / "ko_spm.model")
        )
        tok_path = (
            raw_tok_path
            if raw_tok_path.exists()
            else resolve_spm_model_path(raw_tok_path)
        )
        if not tok_path.exists():
            if args.train_sp_if_missing:
                corpus = collect_korean_corpus(
                    json_dirs_71693=[task_71693[0]] if task_71693 else (),
                    json_dirs_71379=[task_71379[0]] if task_71379 else (),
                    use_mt_fallback=args.use_mt_fallback,
                    en_accent_71379=args.en_accent_71379,
                )
                logger.info("SentencePiece 모델(%s) 자동 학습을 시작합니다...", tok_path)
                tok_path = train_korean_sentencepiece(
                    sentences=corpus,
                    output_model_path=tok_path,
                    vocab_size=cfg.text_card,
                )
                trained_proc = load_korean_tokenizer(tok_path)
                logger.info(
                    "SentencePiece 학습 완료: %s (어휘 크기=%d)",
                    tok_path,
                    trained_proc.GetPieceSize(),
                )
            else:
                raise FileNotFoundError(
                    f"SentencePiece 토크나이저 파일을 찾을 수 없습니다: {tok_path} "
                    "(자동 생성하려면 --train-sp-if-missing 옵션을 추가하세요.)"
                )

        sp_proc = load_korean_tokenizer(tok_path)
        sp_eos = sp_proc.eos_id()
        sp_pad = sp_proc.pad_id()
        sp_vocab = sp_proc.GetPieceSize()
        if sp_eos != TEXT_EOS_ID:
            raise ValueError(
                f"토크나이저 eos_id({sp_eos})가 EDarch TEXT_EOS_ID({TEXT_EOS_ID})와 일치하지 않습니다."
            )
        if sp_pad != TEXT_PAD_ID:
            raise ValueError(
                f"토크나이저 pad_id({sp_pad})가 EDarch TEXT_PAD_ID({TEXT_PAD_ID})와 일치하지 않습니다."
            )
        if args.ko_vocab_size is None and sp_vocab != cfg.text_card:
            logger.info(
                "토크나이저 어휘 크기(%d)에 맞춰 text_card를 %d -> %d로 설정합니다.",
                sp_vocab,
                cfg.text_card,
                sp_vocab,
            )
            cfg.text_card = sp_vocab
        elif sp_vocab > cfg.text_card:
            raise ValueError(
                f"토크나이저 어휘 크기({sp_vocab})가 --ko-vocab-size({cfg.text_card})보다 큽니다."
            )
        sp_fn = lambda text: list(sp_proc.EncodeAsIds(text))
        if sp_pieces_fn is None:
            sp_pieces_fn = lambda text: list(sp_proc.EncodeAsPieces(text))

    if align_fn is None:
        if args.ctc_align_model is not None and args.align_mode in ("auto", "ctc"):
            logger.info("CTC 강제 정렬 모델 로드 중: %s", args.ctc_align_model)
            align_fn = build_hf_ctc_forced_align_fn(args.ctc_align_model, device=device)
        elif args.whisper_align_model is not None and args.align_mode in ("auto", "whisper"):
            logger.info("Whisper 단어 타임스탬프 모델 로드 중: %s", args.whisper_align_model)
            align_fn = build_hf_whisper_align_fn(args.whisper_align_model, device=device)
        elif args.align_mode == "whisper":
            raise ValueError(
                "--align-mode whisper 사용 시 --whisper-align-model 또는 ko_align_fn이 필요합니다."
            )
        elif args.align_mode == "ctc":
            raise ValueError(
                "--align-mode ctc 사용 시 --ctc-align-model 또는 ko_align_fn이 필요합니다."
            )

    if mimi_fn is None and args.mimi_model is not None:
        ext_codec = ExternalMimiCodec(cfg, model_id=args.mimi_model).to(device).eval()
        mimi_fn = lambda wav: ext_codec.encode(wav.to(device))

    if sim_fn is None and args.wavlm_model is not None:
        sim_fn = build_hf_wavlm_sim_bin_fn(
            model_id=args.wavlm_model,
            device=device,
            num_speaker_sim_bins=cfg.num_speaker_sim_bins,
        )

    if args.demo_hooks:
        if tts_fn is None and not mask_target_93:
            tts_fn = lambda text, prompt: torch.zeros(
                max(SAMPLES_PER_FRAME, len(text) * (SAMPLES_PER_FRAME // 2)),
                dtype=torch.float32,
            )
        if sp_fn is None:
            usable = max(1, cfg.text_card - 4)
            sp_fn = lambda text: [4 + (ord(ch) % usable) for ch in text if not ch.isspace()]
            if sp_pieces_fn is None:
                sp_pieces_fn = lambda text: [ch for ch in text if not ch.isspace()]
        if mimi_fn is None:
            demo_codec = MiniMimiCodec(cfg).to(device).eval()
            mimi_fn = lambda wav: demo_codec.encode(wav.to(device))

    needs_tts = (task_71693 is not None and not mask_target_93) or (
        task_71379 is not None and args.ko_audio_mode_71379 == "tts"
    )
    missing = []
    if needs_tts and tts_fn is None:
        missing.append("zero_shot_ko_tts_fn (또는 --ko-mode-71693 mms/edge-tts/text-only)")
    if sp_fn is None:
        missing.append("ko_sp_encode_fn")
    if mimi_fn is None:
        missing.append("mimi_encode_fn")
    if missing:
        raise ValueError(
            f"필수 전처리 훅이 지정되지 않았습니다: {', '.join(missing)}. "
            "--hooks-module(--tokenizer, --mimi-model)을 지정하거나 데모 검증 시 --demo-hooks를 사용하세요."
        )

    has_user_sim_fn = sim_fn is not None
    audio_src_tag = (
        ko_mode_93
        if ko_mode_93 in ("mms", "edge-tts", "text-only")
        else "cloned_tts"
    )
    default_93_bin = resolve_default_sim_bin(audio_src_tag, cfg.num_speaker_sim_bins)
    sim_fn_for_71693 = (
        sim_fn if sim_fn is not None else (lambda prompt, gen: default_93_bin)
    )

    standalone_71379 = task_71379 is not None and task_71693 is None
    fallback_real_sim_bin = args.real_pair_sim_bin
    if not has_user_sim_fn:
        if standalone_71379 and fallback_real_sim_bin is None:
            fallback_real_sim_bin = resolve_default_sim_bin(
                "real_pair", cfg.num_speaker_sim_bins, standalone_real=True
            )
            logger.warning(
                "71379 단독 전처리 시 WavLM이 지정되지 않아 추론 기본 조건과 정합되도록 sim_bin=%d로 설정합니다.",
                fallback_real_sim_bin,
            )
        elif task_71693 is not None and task_71379 is not None:
            logger.info(
                "WavLM 미지정: 71693(%s) 샘플에는 sim_bin=%d, "
                "71379(이종 화자 원천 WAV) 샘플에는 성별 일치 여부에 따라 sim_bin=0..1을 부여합니다.",
                audio_src_tag,
                default_93_bin,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    both_active = task_71693 is not None and task_71379 is not None
    saved_71693 = 0
    saved_71379 = 0

    if task_71693 is not None:
        assert sp_fn is not None and mimi_fn is not None
        out_93 = (args.output_dir / "71693") if both_active else args.output_dir
        logger.info(
            "AI-Hub 71693 전처리 시작: json=%s, wav=%s -> %s",
            task_71693[0],
            task_71693[1],
            out_93,
        )
        saved_71693 = preprocess_71693_dataset(
            json_dir=task_71693[0],
            wav_dir=task_71693[1],
            output_dir=out_93,
            zero_shot_ko_tts_fn=tts_fn,
            ko_sp_encode_fn=sp_fn,
            mimi_encode_fn=mimi_fn,
            wavlm_sim_bin_fn=sim_fn_for_71693,
            target_window_sec=args.target_window_sec,
            max_gap_sec=args.max_gap_sec,
            expected_q=cfg.num_codebooks_per_stream,
            max_context_frames=cfg.context_frames,
            text_card=cfg.text_card,
            num_speaker_sim_bins=cfg.num_speaker_sim_bins,
            use_mt_fallback=args.use_mt_fallback,
            file_prefix="71693_" if both_active else "",
            skip_existing=args.skip_existing,
            align_mode=args.align_mode,
            ko_align_fn=align_fn,
            ko_sp_pieces_fn=sp_pieces_fn,
            monologue_lead_frames=args.monologue_lead_frames,
            mask_target_audio=mask_target_93,
        )
        logger.info("AI-Hub 71693 전처리 완료: %d개 샘플 저장됨", saved_71693)

    if task_71379 is not None:
        assert sp_fn is not None and mimi_fn is not None
        out_79 = (args.output_dir / "71379") if both_active else args.output_dir
        logger.info("AI-Hub 71379 전처리 시작: json=%s -> %s", task_71379[0], out_79)
        saved_71379 = preprocess_71379_dataset(
            json_dir=task_71379[0],
            wav_dir=task_71379[1],
            en_wav_dir=task_71379[2],
            ko_wav_dir=task_71379[3],
            output_dir=out_79,
            ko_sp_encode_fn=sp_fn,
            mimi_encode_fn=mimi_fn,
            wavlm_sim_bin_fn=sim_fn,
            zero_shot_ko_tts_fn=tts_fn,
            ko_audio_mode=args.ko_audio_mode_71379,
            en_accent=args.en_accent_71379,
            target_window_sec=args.target_window_sec,
            max_gap_sec=args.max_gap_sec,
            inter_utt_pause_sec=args.inter_utt_pause_sec,
            max_overlap_sec=args.max_overlap_sec,
            expected_q=cfg.num_codebooks_per_stream,
            max_context_frames=cfg.context_frames,
            text_card=cfg.text_card,
            num_speaker_sim_bins=cfg.num_speaker_sim_bins,
            fallback_real_sim_bin=fallback_real_sim_bin,
            use_mt_fallback=args.use_mt_fallback,
            file_prefix="71379_",
            skip_existing=args.skip_existing,
            align_mode=args.align_mode,
            ko_align_fn=align_fn,
            ko_sp_pieces_fn=sp_pieces_fn,
            monologue_lead_frames=args.monologue_lead_frames,
        )
        logger.info("AI-Hub 71379 전처리 완료: %d개 샘플 저장됨", saved_71379)

    effective_align = (
        args.align_mode
        if args.align_mode != "auto"
        else ("external_timestamps" if align_fn is not None else "energy")
    )
    total_saved = saved_71693 + saved_71379
    manifest = {
        "preset": args.preset,
        "total_streams": cfg.total_streams,
        "num_codebooks_per_stream": cfg.num_codebooks_per_stream,
        "text_card": cfg.text_card,
        "audio_card": cfg.audio_card,
        "context_frames": cfg.context_frames,
        "align_mode": effective_align,
        "monologue_lead_frames": args.monologue_lead_frames,
        "samples_71693": saved_71693,
        "samples_71379": saved_71379,
        "total_samples": total_saved,
    }
    manifest_path = args.output_dir / "preprocessing_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"[Done] 전처리 완료: 총 {total_saved}개 .pt 샘플 저장됨 "
        f"(71693={saved_71693}, 71379={saved_71379}, align={effective_align}, text_card={cfg.text_card}) -> {args.output_dir}"
    )
    print(
        f"  -> Stage 1 SFT 학습 예시: python train.py --preset {args.preset} "
        f"--ko-vocab-size {cfg.text_card} --data-dir \"{args.output_dir}\""
    )
    print(
        f"  -> Stage 2 GRPO 강화학습 예시: python train_grpo.py "
        f"--data-dir \"{args.output_dir}\""
    )
