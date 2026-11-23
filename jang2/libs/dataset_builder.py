"""AI-Hub 71693·71379 대규모 분할(Training/Validation) 저메모리 병렬 `.pt` 변환 및 전처리 실행 모듈."""

from __future__ import annotations

import argparse
from collections import deque
import gc
import importlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, List, Optional, Sequence, Set, Tuple

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
    MiniMimiCodec,
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
    load_mono_wav_24k,
)
from libs.config import EDarchConfig
from libs.dataset_parsing import (
    SplitLayout71379,
    SplitLayout71693,
    discover_71379_splits,
    discover_71693_splits,
    extract_sublabel_korean_sentences,
    parse_71379_split,
    parse_71693_split,
    parse_single_split,
)
from libs.env import setup_device
from libs.preprocessing_common import (
    BroadcastUtterance71379,
    UtteranceSpan,
    assemble_multistream_example,
    build_edge_korean_tts_fn,
    build_mms_korean_tts_fn,
    count_completed_window_tree,
    extract_prompt_24k_from_file,
    fast_scandir_files,
    index_existing_window_trees,
    normalize_tts_output,
    resolve_default_sim_bin,
    save_training_example_pt,
    slice_71379_windows,
    slice_71693_windows,
    validate_ko_token_ids,
    validate_sim_bin,
)
from libs.tokenizer import (
    load_korean_tokenizer,
    normalize_korean_text,
    resolve_spm_model_path,
    train_korean_sentencepiece,
)

logger = logging.getLogger(__name__)


def ensure_preprocess_optional_packages(args: argparse.Namespace) -> None:
    """선택된 코덱·TTS·정렬 옵션에 필요한 추가 파이썬 패키지가 누락된 경우 자동 설치합니다."""
    if getattr(args, "no_auto_install", False):
        return

    needed: List[str] = []
    needs_hf = (
        args.codec == "mimi"
        or (not args.only_71379 and args.ko_mode_71693 == "mms" and args.hooks_module is None)
        or bool(args.ctc_align_model or args.whisper_align_model or args.wavlm_model)
    )
    if needs_hf:
        if importlib.util.find_spec("transformers") is None:
            needed.append("transformers>=4.46.0")
        if importlib.util.find_spec("huggingface_hub") is None:
            needed.append("huggingface_hub>=0.26.0")

    if (
        not args.only_71379
        and args.ko_mode_71693 == "edge-tts"
        and importlib.util.find_spec("edge_tts") is None
    ):
        needed.append("edge-tts>=6.1.0")

    if needed:
        print(f"[Setup] 전처리 필수 추가 패키지 설치 중: {', '.join(needed)}")
        subprocess.run([sys.executable, "-m", "pip", "install", *needed], check=True)


def _maybe_release_memory(saved_count: int, device: torch.device) -> None:
    """일정 샘플 저장 주기마다 파이썬 가비지 컬렉션 및 CUDA 캐시 해제를 수행합니다."""
    if saved_count > 0 and saved_count % 100 == 0:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def process_71693_to_pt(
    grouped: dict[str, List[UtteranceSpan]],
    wav_dirs: Sequence[Path],
    output_dir: Path,
    cfg: EDarchConfig,
    device: torch.device,
    sp_encode_fn: Callable[[str], List[int]],
    sp_pieces_fn: Callable[[str], List[str]],
    mimi_encode_fn: Callable[[torch.Tensor], torch.Tensor],
    ko_mode_71693: str,
    ko_tts_fn: Optional[Callable[[str, torch.Tensor, str], torch.Tensor]],
    wavlm_sim_fn: Optional[Callable[[torch.Tensor, torch.Tensor], int]],
    align_mode: str = "energy",
    ko_align_fn: Optional[KoAlignFn] = None,
    monologue_lead_frames: int = 1,
    target_window_sec: float = 35.0,
    max_gap_sec: float = 8.0,
    skip_existing: bool = True,
    max_samples: Optional[int] = None,
) -> int:
    """전체 강의 WAV를 통째로 메모리에 올리지 않고 윈도우 구간만 부분 Seek 로드하여 `.pt`로 저장합니다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_index, _ = fast_scandir_files(wav_dirs, ".wav")
    existing_stems, split_ancestors = (
        index_existing_window_trees(output_dir) if skip_existing else (set(), set())
    )

    saved_count = 0
    t0 = time.perf_counter()
    total_lectures = len(grouped)
    hard_max_frames = cfg.context_frames if cfg.context_frames > 0 else 1000
    mask_target = ko_mode_71693 == "text-only"
    audio_src_tag = (
        ko_mode_71693 if ko_mode_71693 in ("mms", "edge-tts", "text-only") else "cloned_tts"
    )

    for lec_idx, (fname, spans) in enumerate(grouped.items(), start=1):
        if max_samples is not None and saved_count >= max_samples:
            break

        base_name_lower = os.path.basename(fname).lower()
        stem_lower = os.path.splitext(base_name_lower)[0]
        wav_path = wav_index.get(base_name_lower) or wav_index.get(f"{stem_lower}.wav")
        if wav_path is None:
            logger.warning("[71693] 원천 영어 WAV 누락으로 건너뜀: %s", fname)
            continue

        windows = slice_71693_windows(
            spans, target_window_sec=target_window_sec, max_gap_sec=max_gap_sec
        )
        lec_stem = os.path.splitext(os.path.basename(fname))[0]

        if skip_existing and windows:
            existing_hits = [
                count_completed_window_tree(
                    f"71693_{lec_stem}_w{w_i:04d}",
                    len(w_spans),
                    existing_stems,
                    branch_tags=("_p0", "_p1"),
                )
                for w_i, w_spans in enumerate(windows)
            ]
            if all(h is not None for h in existing_hits):
                saved_count += sum(h for h in existing_hits if h is not None)
                continue

        need_prompt = wavlm_sim_fn is not None or (
            ko_mode_71693 not in ("text-only", "mms", "edge-tts")
        )
        speaker_prompt = (
            extract_prompt_24k_from_file(wav_path, spans)
            if need_prompt
            else torch.zeros(SAMPLES_PER_FRAME, dtype=torch.float32)
        )

        for w_i, base_win in enumerate(windows):
            if max_samples is not None and saved_count >= max_samples:
                break

            base_stem = f"71693_{lec_stem}_w{w_i:04d}"
            tts_cache: dict[str, Tuple[torch.Tensor, List[int], List[str]]] = {}
            queue: deque[Tuple[str, List[UtteranceSpan]]] = deque([(base_stem, base_win)])
            while queue:
                if max_samples is not None and saved_count >= max_samples:
                    break
                cur_stem, win_spans = queue.popleft()

                if skip_existing:
                    done_leaves = count_completed_window_tree(
                        cur_stem, len(win_spans), existing_stems, branch_tags=("_p0", "_p1")
                    )
                    if done_leaves is not None:
                        saved_count += done_leaves
                        continue
                    if len(win_spans) > 1 and cur_stem in split_ancestors:
                        mid = len(win_spans) // 2
                        queue.appendleft((f"{cur_stem}_p1", win_spans[mid:]))
                        queue.appendleft((f"{cur_stem}_p0", win_spans[:mid]))
                        continue

                can_split = cfg.context_frames > 0 and len(win_spans) > 1
                win_start_sec = min(sp.start_sec for sp in win_spans)
                win_end_sec = min(
                    max(sp.end_sec for sp in win_spans),
                    win_start_sec + (hard_max_frames / 12.5),
                )
                en_span_samples = max(0, int((win_end_sec - win_start_sec) * SAMPLE_RATE))

                ko_placements: List[Tuple[int, torch.Tensor, List[Tuple[int, int]]]] = []
                src_start_frames: List[int] = []
                src_end_frames: List[int] = []
                ref_token_groups: List[List[int]] = []
                ko_cursor_sample = 0
                last_token_frame = -1

                for sp in win_spans:
                    cached = tts_cache.get(sp.sn)
                    if cached is not None:
                        ko_wav, ko_ids, ko_pieces = cached
                    else:
                        ko_ids = list(sp_encode_fn(sp.ko_mtpe))
                        validate_ko_token_ids(ko_ids, cfg.text_card)
                        ko_pieces = list(sp_pieces_fn(sp.ko_mtpe))
                        if mask_target:
                            est_frames = max(len(ko_ids) + 2, int(max(1, len(sp.ko_mtpe)) * 1.6))
                            ko_wav = torch.zeros(est_frames * SAMPLES_PER_FRAME, dtype=torch.float32)
                        else:
                            assert ko_tts_fn is not None
                            ko_wav = normalize_tts_output(
                                ko_tts_fn(sp.ko_mtpe, speaker_prompt, sp.gender)
                            )
                        max_ko_samples = max(
                            SAMPLES_PER_FRAME * 4, (hard_max_frames - 8) * SAMPLES_PER_FRAME
                        )
                        if ko_wav.numel() > max_ko_samples:
                            ko_wav = ko_wav[:max_ko_samples]
                        tts_cache[sp.sn] = (ko_wav, ko_ids, ko_pieces)

                    clamped_end_sec = min(sp.end_sec, win_end_sec)
                    rel_en_s = max(0, int((sp.start_sec - win_start_sec) * SAMPLE_RATE))
                    rel_en_e = max(rel_en_s, int((clamped_end_sec - win_start_sec) * SAMPLE_RATE))
                    en_s_frame = rel_en_s // SAMPLES_PER_FRAME
                    en_e_frame = (rel_en_e + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME
                    prev_ko_end_frame = (ko_cursor_sample + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME

                    eff_align = "uniform" if mask_target else align_mode
                    audio_start_sample, tok_pos, last_token_frame = compute_aligned_token_placement(
                        ko_text=sp.ko_mtpe,
                        ko_wav_24k=ko_wav,
                        ko_token_ids=ko_ids,
                        en_end_frame=en_e_frame,
                        prev_audio_end_frame=prev_ko_end_frame,
                        last_token_frame=last_token_frame,
                        align_mode=eff_align,
                        ko_align_fn=None if mask_target else ko_align_fn,
                        token_pieces=ko_pieces,
                        monologue_lead_frames=monologue_lead_frames,
                    )
                    ko_placements.append((audio_start_sample, ko_wav, tok_pos))
                    src_start_frames.append(int(en_s_frame))
                    src_end_frames.append(int(en_e_frame))
                    ref_token_groups.append([int(tid) for tid in ko_ids])
                    ko_cursor_sample = audio_start_sample + ko_wav.numel()

                audio_end_frames = (
                    max(en_span_samples, ko_cursor_sample) + SAMPLES_PER_FRAME - 1
                ) // SAMPLES_PER_FRAME
                est_total_frames = max(
                    audio_end_frames + 4, max(last_token_frame + 1, audio_end_frames) + 1
                )
                if can_split and est_total_frames > cfg.context_frames:
                    mid = len(win_spans) // 2
                    queue.appendleft((f"{cur_stem}_p1", win_spans[mid:]))
                    queue.appendleft((f"{cur_stem}_p0", win_spans[:mid]))
                    continue

                win_wav_slice = load_mono_wav_24k(
                    wav_path, start_sec=win_start_sec, end_sec=win_end_sec
                )
                if win_wav_slice.numel() == 0:
                    continue

                en_placements: List[Tuple[int, torch.Tensor]] = []
                for sp in win_spans:
                    rel_s = max(
                        0, min(int((sp.start_sec - win_start_sec) * SAMPLE_RATE), win_wav_slice.numel())
                    )
                    rel_e = max(
                        rel_s,
                        min(int((sp.end_sec - win_start_sec) * SAMPLE_RATE), win_wav_slice.numel()),
                    )
                    if rel_e > rel_s:
                        en_placements.append((rel_s, win_wav_slice[rel_s:rel_e]))
                del win_wav_slice

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
                    expected_q=cfg.num_codebooks_per_stream,
                    align_mode=align_mode,
                    mask_target_audio=mask_target,
                    hard_max_frames=hard_max_frames,
                )

                if not mask_target and wavlm_sim_fn is not None and ko_speech_only.numel() > 0:
                    sim_bin = validate_sim_bin(
                        wavlm_sim_fn(speaker_prompt, ko_speech_only[: 10 * SAMPLE_RATE]),
                        cfg.num_speaker_sim_bins,
                    )
                else:
                    sim_bin = resolve_default_sim_bin(audio_src_tag, cfg.num_speaker_sim_bins)

                built_meta["dataset"] = "71693"
                built_meta["ko_audio_source"] = ko_mode_71693
                out_pt = output_dir / f"{cur_stem}.pt"
                save_training_example_pt(out_pt, codes, sim_bin, metadata=built_meta)
                existing_stems.add(cur_stem)
                saved_count += 1
                _maybe_release_memory(saved_count, device)

            tts_cache.clear()

        if lec_idx == 1 or lec_idx % 10 == 0 or lec_idx == total_lectures:
            logger.info(
                "[71693] 진행: 강의 %d/%d | 누적 저장 샘플=%d (%.1fs)",
                lec_idx,
                total_lectures,
                saved_count,
                time.perf_counter() - t0,
            )

    return saved_count


def process_71379_to_pt(
    grouped: dict[str, List[BroadcastUtterance71379]],
    ko_wav_dirs: Sequence[Path],
    en_wav_dirs: Sequence[Path],
    output_dir: Path,
    cfg: EDarchConfig,
    device: torch.device,
    sp_encode_fn: Callable[[str], List[int]],
    sp_pieces_fn: Callable[[str], List[str]],
    mimi_encode_fn: Callable[[torch.Tensor], torch.Tensor],
    wavlm_sim_fn: Optional[Callable[[torch.Tensor, torch.Tensor], int]],
    align_mode: str = "energy",
    ko_align_fn: Optional[KoAlignFn] = None,
    monologue_lead_frames: int = 1,
    target_window_sec: float = 35.0,
    max_gap_sec: float = 8.0,
    inter_utt_pause_sec: float = 0.5,
    max_overlap_sec: float = 4.0,
    skip_existing: bool = True,
    max_samples: Optional[int] = None,
    standalone_real: bool = False,
    explicit_real_sim_bin: Optional[int] = None,
) -> int:
    """파싱된 71379 씬 그룹들을 저메모리·정밀 정렬 `.pt` 파일로 변환하여 저장합니다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[71379] 한/영 원천·발화 WAV 파일 고속 인덱싱 중...")
    t_idx = time.perf_counter()
    ko_wav_index, _ = fast_scandir_files(ko_wav_dirs, ".wav")
    en_wav_index, _ = fast_scandir_files(en_wav_dirs, ".wav")
    logger.info(
        "[71379] WAV 인덱싱 완료: KO=%d개, EN=%d개 (%.1fs)",
        len(ko_wav_index),
        len(en_wav_index),
        time.perf_counter() - t_idx,
    )

    existing_stems, split_ancestors = (
        index_existing_window_trees(output_dir) if skip_existing else (set(), set())
    )

    saved_count = 0
    t0 = time.perf_counter()
    ordered_groups = list(grouped.items())
    total_groups = len(ordered_groups)
    max_overlap_samples = max(0, int(max_overlap_sec * SAMPLE_RATE))
    hard_max_frames = cfg.context_frames if cfg.context_frames > 0 else 1000

    for g_idx, (group_key, recs) in enumerate(ordered_groups, start=1):
        if max_samples is not None and saved_count >= max_samples:
            break

        valid_triples: List[Tuple[BroadcastUtterance71379, str, str]] = []
        for idx, rec in enumerate(recs):
            en_base = os.path.basename(rec.en_wav_name).lower()
            en_stem = os.path.splitext(en_base)[0]
            ko_base = os.path.basename(rec.ko_wav_name).lower()
            ko_stem = os.path.splitext(ko_base)[0]
            en_p = en_wav_index.get(en_base) or en_wav_index.get(f"{en_stem}.wav")
            ko_p = ko_wav_index.get(ko_base) or ko_wav_index.get(f"{ko_stem}.wav")
            if en_p is None or ko_p is None:
                if idx + 1 < len(recs):
                    recs[idx + 1].segment_break = True
                continue
            valid_triples.append((rec, en_p, ko_p))

        if not valid_triples:
            continue

        recs_only = [t[0] for t in valid_triples]
        path_map = {t[0].utt_id: (t[1], t[2]) for t in valid_triples}
        windows = slice_71379_windows(
            recs_only,
            target_window_sec=target_window_sec,
            max_gap_sec=max_gap_sec,
            inter_utt_pause_sec=inter_utt_pause_sec,
            min_est_dur_sec=2.5,
        )

        if skip_existing and windows:
            hits = [
                count_completed_window_tree(
                    f"71379_{group_key}_w{w_i:04d}",
                    len(w_recs),
                    existing_stems,
                    branch_tags=("_p0", "_p1"),
                )
                for w_i, w_recs in enumerate(windows)
            ]
            if all(h is not None for h in hits):
                saved_count += sum(h for h in hits if h is not None)
                continue

        for w_i, base_win in enumerate(windows):
            if max_samples is not None and saved_count >= max_samples:
                break

            base_stem = f"71379_{group_key}_w{w_i:04d}"
            clip_cache: dict[str, Tuple[torch.Tensor, torch.Tensor, List[int], List[str]]] = {}
            queue: deque[Tuple[str, List[BroadcastUtterance71379]]] = deque([(base_stem, base_win)])
            while queue:
                if max_samples is not None and saved_count >= max_samples:
                    break
                cur_stem, win_recs = queue.popleft()

                if skip_existing:
                    done_leaves = count_completed_window_tree(
                        cur_stem, len(win_recs), existing_stems, branch_tags=("_p0", "_p1")
                    )
                    if done_leaves is not None:
                        saved_count += done_leaves
                        continue
                    if len(win_recs) > 1 and cur_stem in split_ancestors:
                        mid = len(win_recs) // 2
                        queue.appendleft((f"{cur_stem}_p1", win_recs[mid:]))
                        queue.appendleft((f"{cur_stem}_p0", win_recs[:mid]))
                        continue

                can_split = cfg.context_frames > 0 and len(win_recs) > 1

                en_placements: List[Tuple[int, torch.Tensor]] = []
                ko_placements: List[Tuple[int, torch.Tensor, List[Tuple[int, int]]]] = []
                src_start_frames: List[int] = []
                src_end_frames: List[int] = []
                ref_token_groups: List[List[int]] = []

                prev_en_end_sample = 0
                ko_cursor_sample = 0
                last_token_frame = -1
                failed_utt = False

                for i, rec in enumerate(win_recs):
                    cached = clip_cache.get(rec.utt_id)
                    if cached is not None:
                        en_wav, ko_wav, ko_ids, ko_pieces = cached
                    else:
                        en_p, ko_p = path_map[rec.utt_id]
                        en_wav = load_mono_wav_24k(en_p)
                        ko_wav = load_mono_wav_24k(ko_p)
                        if en_wav.numel() == 0 or ko_wav.numel() == 0:
                            failed_utt = True
                            break
                        ko_ids = list(sp_encode_fn(rec.ko_text))
                        try:
                            validate_ko_token_ids(ko_ids, cfg.text_card)
                        except ValueError:
                            failed_utt = True
                            break
                        ko_pieces = list(sp_pieces_fn(rec.ko_text))
                        clip_cache[rec.utt_id] = (en_wav, ko_wav, ko_ids, ko_pieces)

                    if i == 0:
                        en_s_sample = 0
                    else:
                        prev_rec = win_recs[i - 1]
                        pause_sec = inter_utt_pause_sec
                        if prev_rec.raw_end_sec is not None and rec.raw_start_sec is not None:
                            rg = rec.raw_start_sec - prev_rec.raw_end_sec
                            if rg >= 0.0:
                                pause_sec = min(max(rg, inter_utt_pause_sec), max_gap_sec)
                        en_s_sample = max(
                            prev_en_end_sample + int(pause_sec * SAMPLE_RATE),
                            ko_cursor_sample - max_overlap_samples,
                        )

                    en_e_sample = en_s_sample + en_wav.numel()
                    en_placements.append((en_s_sample, en_wav))
                    prev_en_end_sample = en_e_sample

                    en_s_frame = en_s_sample // SAMPLES_PER_FRAME
                    en_e_frame = (en_e_sample + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME
                    prev_ko_end_frame = (ko_cursor_sample + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME

                    ko_start_sample, tok_pos, last_token_frame = compute_aligned_token_placement(
                        ko_text=rec.ko_text,
                        ko_wav_24k=ko_wav,
                        ko_token_ids=ko_ids,
                        en_end_frame=en_e_frame,
                        prev_audio_end_frame=prev_ko_end_frame,
                        last_token_frame=last_token_frame,
                        align_mode=align_mode,
                        ko_align_fn=ko_align_fn,
                        token_pieces=ko_pieces,
                        monologue_lead_frames=monologue_lead_frames,
                    )
                    ko_placements.append((ko_start_sample, ko_wav, tok_pos))
                    src_start_frames.append(int(en_s_frame))
                    src_end_frames.append(int(en_e_frame))
                    ref_token_groups.append([int(tid) for tid in ko_ids])
                    ko_cursor_sample = ko_start_sample + ko_wav.numel()

                if failed_utt:
                    if len(win_recs) > 1:
                        mid = len(win_recs) // 2
                        queue.appendleft((f"{cur_stem}_p1", win_recs[mid:]))
                        queue.appendleft((f"{cur_stem}_p0", win_recs[:mid]))
                    continue

                audio_end_frames = (
                    max(prev_en_end_sample, ko_cursor_sample) + SAMPLES_PER_FRAME - 1
                ) // SAMPLES_PER_FRAME
                est_total_frames = max(
                    audio_end_frames + 4, max(last_token_frame + 1, audio_end_frames) + 1
                )
                if can_split and est_total_frames > cfg.context_frames:
                    mid = len(win_recs) // 2
                    queue.appendleft((f"{cur_stem}_p1", win_recs[mid:]))
                    queue.appendleft((f"{cur_stem}_p0", win_recs[:mid]))
                    continue

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
                    expected_q=cfg.num_codebooks_per_stream,
                    align_mode=align_mode,
                    mask_target_audio=False,
                    hard_max_frames=hard_max_frames,
                )

                if wavlm_sim_fn is not None:
                    en_only = torch.cat([w for _, w in en_placements], dim=0)[: 10 * SAMPLE_RATE]
                    sim_bin = validate_sim_bin(
                        wavlm_sim_fn(en_only, ko_speech_only[: 10 * SAMPLE_RATE]),
                        cfg.num_speaker_sim_bins,
                    )
                else:
                    first_rec = win_recs[0]
                    same_sex = bool(
                        first_rec.s_sex and first_rec.t_sex and first_rec.s_sex == first_rec.t_sex
                    )
                    sim_bin = resolve_default_sim_bin(
                        "real_pair",
                        num_speaker_sim_bins=cfg.num_speaker_sim_bins,
                        same_sex=same_sex,
                        standalone_real=standalone_real,
                        explicit_sim_bin=explicit_real_sim_bin,
                    )

                built_meta["dataset"] = "71379"
                built_meta["ko_audio_source"] = "real"
                out_pt = output_dir / f"{cur_stem}.pt"
                save_training_example_pt(out_pt, codes, sim_bin, metadata=built_meta)
                existing_stems.add(cur_stem)
                saved_count += 1
                _maybe_release_memory(saved_count, device)

            clip_cache.clear()

        if g_idx == 1 or g_idx % 200 == 0 or g_idx == total_groups:
            logger.info(
                "[71379] 진행: 씬 그룹 %d/%d | 누적 저장 샘플=%d (%.1fs)",
                g_idx,
                total_groups,
                saved_count,
                time.perf_counter() - t0,
            )

    del ko_wav_index, en_wav_index
    return saved_count


def verify_preprocessed_split(
    split_dir: Path,
    cfg: EDarchConfig,
) -> Tuple[int, int]:
    """생성된 `.pt` 디렉터리를 Stage 1(`PreprocessedPtDataset`)과 Stage 2(`GRPOPromptDataset`)로 검증합니다."""
    from libs.grpo_rollout import GRPOPromptDataset
    from libs.training import PreprocessedPtDataset

    pt_files = list(split_dir.rglob("*.pt"))
    if not pt_files:
        return 0, 0

    sft_ds = PreprocessedPtDataset(
        data_dir=split_dir,
        crop_frames=cfg.context_frames,
        expected_total_streams=cfg.total_streams,
        text_card=cfg.text_card,
        audio_card=cfg.audio_card,
        default_sim_bin=cfg.num_speaker_sim_bins - 1,
        text_eos_id=cfg.text_eos_id,
        num_speaker_sim_bins=cfg.num_speaker_sim_bins,
    )
    grpo_ds = GRPOPromptDataset(
        data_dir=split_dir,
        crop_src_frames=min(450, cfg.context_frames),
        cfg=cfg,
    )
    for idx in {0, len(sft_ds) // 2, len(sft_ds) - 1}:
        _ = sft_ds[idx]
        _ = grpo_ds[idx]
    return len(sft_ds), len(grpo_ds)


def run_preprocess_datasets(args: argparse.Namespace) -> None:
    """대용량 AI-Hub 71693·71379 데이터셋을 저메모리 순차 파이프라인으로 전처리하여 train/val로 분리 저장합니다."""
    ensure_preprocess_optional_packages(args)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    device = setup_device(cpu_only=args.cpu, seed=args.seed, tag="PreprocessAll")

    cfg = EDarchConfig.from_preset(args.preset)
    if args.ko_vocab_size is not None:
        cfg.text_card = args.ko_vocab_size

    requested_splits: Set[str] = (
        {"train", "val"} if args.splits == "both" else {args.splits}
    )
    cat_filter = {c.strip() for c in args.categories if c.strip()} if args.categories else None

    use_71693 = (
        args.dir_71693 is not None
        and args.dir_71693.exists()
        and args.ko_mode_71693 != "skip"
        and not args.only_71379
    )
    use_71379 = (
        args.dir_71379 is not None
        and args.dir_71379.exists()
        and not args.only_71693
    )

    if not use_71693 and not use_71379:
        raise FileNotFoundError(
            f"전처리할 활성 데이터셋이 없습니다 (dir_71693={args.dir_71693}, dir_71379={args.dir_71379})."
        )

    layouts_93, sublabel_ko_dirs = (
        discover_71693_splits(args.dir_71693, requested_splits)
        if use_71693
        else ([], [])
    )
    layouts_79 = (
        discover_71379_splits(args.dir_71379, requested_splits, categories=cat_filter)
        if use_71379
        else []
    )
    ly93_map = {ly.split_name: ly for ly in layouts_93}
    ly79_map = {ly.split_name: ly for ly in layouts_79}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tok_path = resolve_spm_model_path(
        args.tokenizer if args.tokenizer is not None else (args.output_dir / "ko_spm.model")
    )

    ordered_splits = [s for s in ("train", "val") if s in requested_splits]
    first_split = ordered_splits[0]
    first_g93, first_g79 = parse_single_split(first_split, ly93_map, ly79_map, args)

    if not tok_path.exists():
        logger.info("SentencePiece 토크나이저(%s) 자동 학습용 코퍼스 수집 중...", tok_path)
        corpus_seen: Set[str] = set()
        corpus_list: List[str] = []

        def _add_sent(s: str) -> None:
            ns = normalize_korean_text(s)
            if ns and ns not in corpus_seen:
                corpus_seen.add(ns)
                corpus_list.append(ns)

        if first_g93 is not None:
            for spans in first_g93.values():
                for s_item in spans:
                    _add_sent(s_item.ko_mtpe)
        if first_g79 is not None:
            for recs in first_g79.values():
                for r_item in recs:
                    _add_sent(r_item.ko_text)

        if not args.no_sublabel_corpus and sublabel_ko_dirs:
            sub_sents = extract_sublabel_korean_sentences(
                sublabel_ko_dirs, num_io_threads=args.io_threads
            )
            for s_txt in sub_sents:
                _add_sent(s_txt)
            logger.info("[Sublabel] 한국어 학술 문장 %d개 코퍼스 병합 완료", len(sub_sents))
            del sub_sents

        if not corpus_list:
            raise ValueError("SentencePiece 학습용 한국어 문장이 하나도 수집되지 않았습니다.")

        logger.info(
            "SentencePiece 학습 시작: 고유 문장 %d개 -> %s (목표 vocab=%d)",
            len(corpus_list),
            tok_path,
            cfg.text_card,
        )
        tok_path = train_korean_sentencepiece(
            sentences=corpus_list,
            output_model_path=tok_path,
            vocab_size=cfg.text_card,
            model_type="unigram",
            byte_fallback=True,
        )
        del corpus_list, corpus_seen
        gc.collect()

    sp_proc = load_korean_tokenizer(tok_path)
    sp_vocab = int(sp_proc.GetPieceSize())
    if args.ko_vocab_size is None and sp_vocab != cfg.text_card:
        logger.info("SentencePiece 실제 어휘 크기(%d)에 맞춰 text_card를 조정합니다.", sp_vocab)
        cfg.text_card = sp_vocab
    elif sp_vocab > cfg.text_card:
        raise ValueError(
            f"토크나이저 어휘 크기({sp_vocab})가 설정된 text_card({cfg.text_card})보다 큽니다."
        )
    logger.info("한국어 토크나이저 준비 완료: %s (vocab_size=%d)", tok_path, sp_vocab)

    sp_encode_fn = lambda txt: list(sp_proc.EncodeAsIds(txt))
    sp_pieces_fn = lambda txt: list(sp_proc.EncodeAsPieces(txt))

    hooks_mod = importlib.import_module(args.hooks_module) if args.hooks_module else None
    custom_tts = getattr(hooks_mod, "zero_shot_ko_tts_fn", None) if hooks_mod else None

    if args.codec == "mimi":
        logger.info("사전학습 Mimi 코덱 로드 중: %s", args.mimi_model)
        codec_mod: torch.nn.Module = ExternalMimiCodec(cfg, model_id=args.mimi_model).to(device).eval()
    else:
        logger.warning(
            "--codec mini(데모용 미학습 MiniMimiCodec)가 선택되었습니다. "
            "실제 음성 합성 품질 학습 시에는 기본값(--codec mimi)을 사용하세요."
        )
        codec_mod = MiniMimiCodec(cfg).to(device).eval()

    @torch.inference_mode()
    def mimi_encode_fn(wav: torch.Tensor) -> torch.Tensor:
        return codec_mod.encode(wav.to(device))

    ko_align_fn: Optional[KoAlignFn] = None
    if args.ctc_align_model and args.align_mode in ("auto", "ctc"):
        ko_align_fn = build_hf_ctc_forced_align_fn(args.ctc_align_model, device=device)
    elif args.whisper_align_model and args.align_mode in ("auto", "whisper"):
        ko_align_fn = build_hf_whisper_align_fn(args.whisper_align_model, device=device)

    wavlm_sim_fn = (
        build_hf_wavlm_sim_bin_fn(
            model_id=args.wavlm_model,
            device=device,
            num_speaker_sim_bins=cfg.num_speaker_sim_bins,
        )
        if args.wavlm_model
        else None
    )

    ko_tts_fn: Optional[Callable[[str, torch.Tensor, str], torch.Tensor]] = None
    effective_93_mode = args.ko_mode_71693
    if use_71693 and args.ko_mode_71693 != "text-only":
        if custom_tts is not None:
            ko_tts_fn = lambda txt, prompt, gender="": normalize_tts_output(custom_tts(txt, prompt))
            effective_93_mode = "cloned_tts"
        elif args.ko_mode_71693 == "mms":
            ko_tts_fn = build_mms_korean_tts_fn(device=device, model_id=args.mms_tts_model)
        elif args.ko_mode_71693 == "edge-tts":
            ko_tts_fn = build_edge_korean_tts_fn(cache_dir=args.output_dir / ".edge_tts_cache")

    effective_align = (
        "energy"
        if (args.align_mode == "auto" and ko_align_fn is None)
        else args.align_mode
    )
    standalone_71379 = use_71379 and not use_71693
    explicit_real_sim_bin = getattr(args, "real_pair_sim_bin", None)

    stats: dict[str, dict[str, int]] = {"train": {}, "val": {}}
    for idx_sp, sp_name in enumerate(ordered_splits):
        if idx_sp == 0:
            cur_g93, cur_g79 = first_g93, first_g79
            first_g93, first_g79 = None, None
        else:
            cur_g93, cur_g79 = parse_single_split(sp_name, ly93_map, ly79_map, args)

        val_cap_93 = (
            max(1, args.max_samples_71693 // 5)
            if (sp_name == "val" and args.max_samples_71693 is not None and args.max_val_samples is None)
            else (args.max_val_samples if sp_name == "val" else args.max_samples_71693)
        )
        val_cap_79 = (
            max(1, args.max_samples_71379 // 5)
            if (sp_name == "val" and args.max_samples_71379 is not None and args.max_val_samples is None)
            else (args.max_val_samples if sp_name == "val" else args.max_samples_71379)
        )

        if cur_g79 is not None and sp_name in ly79_map:
            ly79 = ly79_map[sp_name]
            out_79 = args.output_dir / sp_name / "71379"
            logger.info("=== [71379 / %s] 전처리 시작 -> %s ===", sp_name, out_79)
            cnt_79 = process_71379_to_pt(
                grouped=cur_g79,
                ko_wav_dirs=ly79.ko_wav_dirs,
                en_wav_dirs=ly79.en_wav_dirs,
                output_dir=out_79,
                cfg=cfg,
                device=device,
                sp_encode_fn=sp_encode_fn,
                sp_pieces_fn=sp_pieces_fn,
                mimi_encode_fn=mimi_encode_fn,
                wavlm_sim_fn=wavlm_sim_fn,
                align_mode=effective_align,
                ko_align_fn=ko_align_fn,
                monologue_lead_frames=args.monologue_lead_frames,
                target_window_sec=args.target_window_sec,
                max_gap_sec=args.max_gap_sec,
                inter_utt_pause_sec=args.inter_utt_pause_sec,
                max_overlap_sec=args.max_overlap_sec,
                skip_existing=not args.overwrite,
                max_samples=val_cap_79,
                standalone_real=standalone_71379,
                explicit_real_sim_bin=explicit_real_sim_bin,
            )
            stats[sp_name]["71379"] = cnt_79
            cur_g79 = None
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if cur_g93 is not None and sp_name in ly93_map:
            ly93 = ly93_map[sp_name]
            out_93 = args.output_dir / sp_name / "71693"
            logger.info(
                "=== [71693 / %s] 전처리 시작 (ko_mode=%s) -> %s ===",
                sp_name,
                effective_93_mode,
                out_93,
            )
            cnt_93 = process_71693_to_pt(
                grouped=cur_g93,
                wav_dirs=ly93.wav_dirs,
                output_dir=out_93,
                cfg=cfg,
                device=device,
                sp_encode_fn=sp_encode_fn,
                sp_pieces_fn=sp_pieces_fn,
                mimi_encode_fn=mimi_encode_fn,
                ko_mode_71693=effective_93_mode,
                ko_tts_fn=ko_tts_fn,
                wavlm_sim_fn=wavlm_sim_fn,
                align_mode=effective_align,
                ko_align_fn=ko_align_fn,
                monologue_lead_frames=args.monologue_lead_frames,
                target_window_sec=args.target_window_sec,
                max_gap_sec=args.max_gap_sec,
                skip_existing=not args.overwrite,
                max_samples=val_cap_93,
            )
            stats[sp_name]["71693"] = cnt_93
            cur_g93 = None
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    verified_counts: dict[str, dict[str, int]] = {}
    if not args.no_verify:
        for sp_name in ("train", "val"):
            sp_dir = args.output_dir / sp_name
            if sp_dir.exists() and any(sp_dir.rglob("*.pt")):
                sft_n, grpo_n = verify_preprocessed_split(sp_dir, cfg)
                verified_counts[sp_name] = {"sft_samples": sft_n, "grpo_prompts": grpo_n}
                logger.info(
                    "[Verify/%s] Stage 1 SFT(%d개) & Stage 2 GRPO(%d개) 데이터셋 검증 통과!",
                    sp_name,
                    sft_n,
                    grpo_n,
                )

    train_dir = args.output_dir / "train"
    val_dir = args.output_dir / "val"
    manifest = {
        "preset": args.preset,
        "codec": args.mimi_model if args.codec == "mimi" else "mini_demo",
        "tokenizer_path": str(tok_path),
        "text_card": cfg.text_card,
        "audio_card": cfg.audio_card,
        "total_streams": cfg.total_streams,
        "num_codebooks_per_stream": cfg.num_codebooks_per_stream,
        "context_frames": cfg.context_frames,
        "target_window_sec": args.target_window_sec,
        "align_mode": effective_align,
        "ko_mode_71693": effective_93_mode if use_71693 else "disabled",
        "saved_samples": stats,
        "verified": verified_counts,
        "commands": {
            "stage1_sft": (
                f'python train.py --preset {args.preset} --ko-vocab-size {cfg.text_card} '
                f'--data-dir "{train_dir}" --val-data-dir "{val_dir}" '
                f'--grad-checkpointing --batch-size 1 --grad-accum-steps 16 --epochs 10'
            ),
            "stage2_grpo": (
                f'python train_grpo.py --preset {args.preset} --ko-vocab-size {cfg.text_card} '
                f'--data-dir "{train_dir}" --val-data-dir "{val_dir}" '
                f'--tokenizer "{tok_path}" --grad-checkpointing --group-size 4'
            ),
            "inference": (
                f'python run.py --mimi-model {args.mimi_model} --tokenizer "{tok_path}" '
                f'--input-wav sample_en.wav'
            ),
        },
    }
    manifest_path = args.output_dir / "preprocessing_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("[Done] AI-Hub 71693 & 71379 학습 데이터 분할 전처리 완료!")
    print(f"  - 출력 루트 디렉터리 : {args.output_dir}")
    print(
        f"  - 한국어 토크나이저  : {tok_path} "
        f"(spm_vocab={sp_vocab}, model_text_card={cfg.text_card})"
    )
    print(f"  - Train 샘플 수     : {stats.get('train', {})}")
    print(f"  - Val 샘플 수       : {stats.get('val', {})} (경로: {val_dir})")
    print("-" * 78)
    print("  [1] Stage 1 지도학습(SFT) 실행 명령어:")
    print(f"      {manifest['commands']['stage1_sft']}")
    print("  [2] Stage 2 GRPO 강화학습 실행 명령어:")
    print(f"      {manifest['commands']['stage2_grpo']}")
    print("=" * 78)
