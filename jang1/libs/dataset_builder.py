"""arXiv:2309.11093 노래 섹션 데이터셋의 이진 분할 `.pt` 변환 및 통합 전처리 실행 모듈."""

from __future__ import annotations

import argparse
from collections import deque
import gc
import json
import logging
import os
from pathlib import Path
import time
from typing import Callable, List, Optional, Sequence, Set, Tuple

import torch

from libs.codec import ExternalMimiCodec, MiniMimiCodec
from libs.config import LyricTransConfig
from libs.dataset_parsing import (
    build_synthetic_kpop_sections,
    discover_lyric_splits,
    parse_lyric_split,
)
from libs.env import setup_device
from libs.preprocessing_common import (
    LyricSectionUnit,
    assemble_lyric_section_example,
    build_lyric_section_unit,
    count_completed_window_tree,
    fast_scandir_files,
    index_existing_window_trees,
    resolve_default_sim_bin,
    save_training_example_pt,
)
from libs.tokenizer import (
    load_lyric_tokenizer,
    resolve_spm_model_path,
    train_lyric_sentencepiece,
)

logger = logging.getLogger(__name__)


def _maybe_release_memory(saved_count: int, device: torch.device) -> None:
    """일정 샘플 저장 주기마다 파이썬 가비지 컬렉션 및 CUDA 캐시 해제를 수행합니다."""
    if saved_count > 0 and saved_count % 100 == 0:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def process_lyric_sections_to_pt(
    sections: Sequence[LyricSectionUnit],
    wav_dirs: Sequence[Path],
    output_dir: Path,
    cfg: LyricTransConfig,
    device: torch.device,
    sp_encode_fn: Callable[[str], List[int]],
    mimi_encode_fn: Callable[[torch.Tensor], torch.Tensor],
    vocal_mode: str = "auto",
    lead_frames: int = 1,
    skip_existing: bool = True,
    max_samples: Optional[int] = None,
) -> int:
    """파싱된 노래 섹션(LyricSectionUnit)들을 음절·음소·SSM·보컬 코드북이 결합된 `.pt`로 원자적 저장합니다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_index, _ = fast_scandir_files(wav_dirs, ".wav")
    existing_stems, split_ancestors = (
        index_existing_window_trees(output_dir) if skip_existing else (set(), set())
    )

    saved_count = 0
    t0 = time.perf_counter()
    total_sections = len(sections)
    hard_max_frames = cfg.context_frames if cfg.context_frames > 0 else 1000

    for sec_i, base_unit in enumerate(sections, start=1):
        if max_samples is not None and saved_count >= max_samples:
            break

        safe_song = "".join(
            ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in base_unit.song_id
        )
        base_stem = f"lyric_{safe_song}_s{base_unit.section_idx:04d}"

        queue: deque[Tuple[str, LyricSectionUnit]] = deque([(base_stem, base_unit)])
        while queue:
            if max_samples is not None and saved_count >= max_samples:
                break
            cur_stem, cur_unit = queue.popleft()

            if skip_existing:
                done_leaves = count_completed_window_tree(
                    cur_stem, len(cur_unit.lines), existing_stems, branch_tags=("_p0", "_p1")
                )
                if done_leaves is not None:
                    saved_count += done_leaves
                    continue
                if len(cur_unit.lines) > 1 and cur_stem in split_ancestors:
                    mid = len(cur_unit.lines) // 2
                    left_u = build_lyric_section_unit(
                        song_id=cur_unit.song_id,
                        section_idx=cur_unit.section_idx,
                        section_type=cur_unit.section_type,
                        src_lines=[ln.src_text for ln in cur_unit.lines[:mid]],
                        tgt_lines=[ln.tgt_text for ln in cur_unit.lines[:mid]],
                        direction=cur_unit.direction,
                        src_wav_name=cur_unit.src_wav_name,
                        tgt_wav_name=cur_unit.tgt_wav_name,
                    )
                    right_u = build_lyric_section_unit(
                        song_id=cur_unit.song_id,
                        section_idx=cur_unit.section_idx,
                        section_type=cur_unit.section_type,
                        src_lines=[ln.src_text for ln in cur_unit.lines[mid:]],
                        tgt_lines=[ln.tgt_text for ln in cur_unit.lines[mid:]],
                        direction=cur_unit.direction,
                        src_wav_name=cur_unit.src_wav_name,
                        tgt_wav_name=cur_unit.tgt_wav_name,
                    )
                    if right_u is not None:
                        queue.appendleft((f"{cur_stem}_p1", right_u))
                    if left_u is not None:
                        queue.appendleft((f"{cur_stem}_p0", left_u))
                    continue

            src_wav_p = (
                wav_index.get(os.path.basename(cur_unit.src_wav_name).lower())
                if cur_unit.src_wav_name
                else None
            )
            tgt_wav_p = (
                wav_index.get(os.path.basename(cur_unit.tgt_wav_name).lower())
                if cur_unit.tgt_wav_name
                else None
            )

            codes, built_meta = assemble_lyric_section_example(
                unit=cur_unit,
                sp_encode_fn=sp_encode_fn,
                mimi_encode_fn=mimi_encode_fn,
                expected_q=cfg.num_codebooks_per_stream,
                text_card=cfg.text_card,
                vocal_mode=vocal_mode,
                src_wav_path=src_wav_p,
                tgt_wav_path=tgt_wav_p,
                lead_frames=lead_frames,
                hard_max_frames=hard_max_frames + 64,
            )

            if cfg.context_frames > 0 and codes.shape[1] > cfg.context_frames and len(cur_unit.lines) > 1:
                mid = len(cur_unit.lines) // 2
                left_u = build_lyric_section_unit(
                    song_id=cur_unit.song_id,
                    section_idx=cur_unit.section_idx,
                    section_type=cur_unit.section_type,
                    src_lines=[ln.src_text for ln in cur_unit.lines[:mid]],
                    tgt_lines=[ln.tgt_text for ln in cur_unit.lines[:mid]],
                    direction=cur_unit.direction,
                    src_wav_name=cur_unit.src_wav_name,
                    tgt_wav_name=cur_unit.tgt_wav_name,
                )
                right_u = build_lyric_section_unit(
                    song_id=cur_unit.song_id,
                    section_idx=cur_unit.section_idx,
                    section_type=cur_unit.section_type,
                    src_lines=[ln.src_text for ln in cur_unit.lines[mid:]],
                    tgt_lines=[ln.tgt_text for ln in cur_unit.lines[mid:]],
                    direction=cur_unit.direction,
                    src_wav_name=cur_unit.src_wav_name,
                    tgt_wav_name=cur_unit.tgt_wav_name,
                )
                if right_u is not None:
                    queue.appendleft((f"{cur_stem}_p1", right_u))
                if left_u is not None:
                    queue.appendleft((f"{cur_stem}_p0", left_u))
                continue

            if cfg.context_frames > 0 and codes.shape[1] > cfg.context_frames:
                codes = codes[:, : cfg.context_frames].contiguous()
                codes[0, -1] = cfg.text_eos_id
                if isinstance(built_meta.get("frame_target_syllables"), torch.Tensor):
                    built_meta["frame_target_syllables"] = built_meta["frame_target_syllables"][
                        : cfg.context_frames
                    ].contiguous()
                if isinstance(built_meta.get("frame_repeat_group_ids"), torch.Tensor):
                    built_meta["frame_repeat_group_ids"] = built_meta["frame_repeat_group_ids"][
                        : cfg.context_frames
                    ].contiguous()
                built_meta["src_start_frames"] = [
                    min(int(f), cfg.context_frames - 1)
                    for f in built_meta.get("src_start_frames", [])
                ]
                built_meta["src_end_frames"] = [
                    min(int(f), cfg.context_frames)
                    for f in built_meta.get("src_end_frames", [])
                ]

            sim_bin = resolve_default_sim_bin(
                vocal_source="real_vocal" if (src_wav_p and tgt_wav_p) else vocal_mode,
                num_speaker_sim_bins=cfg.num_speaker_sim_bins,
            )
            out_pt = output_dir / f"{cur_stem}.pt"
            save_training_example_pt(out_pt, codes, sim_bin, metadata=built_meta)
            existing_stems.add(cur_stem)
            saved_count += 1
            _maybe_release_memory(saved_count, device)

        if sec_i == 1 or sec_i % 50 == 0 or sec_i == total_sections:
            logger.info(
                "[LyricPreprocess] 섹션 %d/%d | 누적 저장 샘플=%d (%.1fs)",
                sec_i,
                total_sections,
                saved_count,
                time.perf_counter() - t0,
            )

    return saved_count


def verify_preprocessed_split(
    split_dir: Path,
    cfg: LyricTransConfig,
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
        crop_src_frames=min(300, cfg.context_frames),
        cfg=cfg,
    )
    for idx in {0, len(sft_ds) // 2, len(sft_ds) - 1}:
        _ = sft_ds[idx]
        _ = grpo_ds[idx]
    return len(sft_ds), len(grpo_ds)


def run_preprocess_datasets(args: argparse.Namespace) -> None:
    """한-영 병렬 노래 가사(Section/Line) 및 보컬 데이터를 전처리하여 train/val `.pt`로 분리 저장합니다."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    device = setup_device(cpu_only=args.cpu, seed=args.seed, tag="PreprocessLyrics")

    src_lang = "en" if args.direction == "en2ko" else "ko"
    tgt_lang = "ko" if args.direction == "en2ko" else "en"
    cfg = LyricTransConfig.from_preset(
        args.preset, source_lang=src_lang, target_lang=tgt_lang
    )
    if args.vocab_size is not None:
        cfg.text_card = args.vocab_size

    requested_splits: Set[str] = (
        {"train", "val"} if args.splits == "both" else {args.splits}
    )

    layouts = discover_lyric_splits(args.data_dir, requested_splits)
    split_units_map: dict[str, Tuple[List[LyricSectionUnit], List[Path]]] = {}

    if layouts:
        shared_source = (
            len(layouts) == 2
            and layouts[0].data_files == layouts[1].data_files
        )
        for ly in layouts:
            cap = args.max_val_samples if ly.split_name == "val" else args.max_samples
            parsed = parse_lyric_split(
                layout=ly,
                direction=args.direction,
                max_lines_per_section=args.max_lines_per_section,
                max_samples=cap,
                num_io_threads=args.io_threads,
                shared_single_source=shared_source,
            )
            if parsed:
                split_units_map[ly.split_name] = (parsed, ly.wav_dirs)

    if not split_units_map:
        logger.info(
            "외부 가사 데이터셋 미지정 또는 미발견 -> 내장 K-pop 싱어블 가사 합성 데모 코퍼스를 생성합니다."
        )
        syn_train, syn_val = build_synthetic_kpop_sections(
            num_songs=args.demo_num_songs,
            direction=args.direction,
            max_lines_per_section=args.max_lines_per_section,
        )
        if "train" in requested_splits:
            split_units_map["train"] = (syn_train, [])
        if "val" in requested_splits:
            split_units_map["val"] = (syn_val, [])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tok_path = resolve_spm_model_path(
        args.tokenizer if args.tokenizer is not None else (args.output_dir / "lyric_spm.model")
    )

    if not tok_path.exists():
        logger.info("SentencePiece 가사 토크나이저(%s) 자동 학습용 코퍼스 수집 중...", tok_path)
        corpus_seen: Set[str] = set()
        corpus_list: List[str] = []
        for units, _ in split_units_map.values():
            for u in units:
                for ln in u.lines:
                    for txt in (ln.src_text, ln.tgt_text):
                        if txt and txt not in corpus_seen:
                            corpus_seen.add(txt)
                            corpus_list.append(txt)
        target_vocab = args.vocab_size if args.vocab_size is not None else min(cfg.text_card, 2000)
        tok_path = train_lyric_sentencepiece(
            sentences=corpus_list,
            output_model_path=tok_path,
            vocab_size=target_vocab,
            model_type="unigram",
            byte_fallback=True,
        )

    sp_proc = load_lyric_tokenizer(tok_path)
    sp_vocab = int(sp_proc.GetPieceSize())
    if args.vocab_size is None and sp_vocab != cfg.text_card:
        cfg.text_card = sp_vocab
    elif sp_vocab > cfg.text_card:
        raise ValueError(
            f"토크나이저 어휘 크기({sp_vocab})가 설정된 text_card({cfg.text_card})보다 큽니다."
        )
    logger.info("노래 가사 토크나이저 준비 완료: %s (vocab_size=%d)", tok_path, sp_vocab)

    sp_encode_fn = lambda txt: list(sp_proc.EncodeAsIds(txt))

    if args.codec == "mimi":
        codec_mod: torch.nn.Module = ExternalMimiCodec(cfg, model_id=args.mimi_model).to(device).eval()
    else:
        codec_mod = MiniMimiCodec(cfg).to(device).eval()

    @torch.inference_mode()
    def mimi_encode_fn(wav: torch.Tensor) -> torch.Tensor:
        return codec_mod.encode(wav.to(device))

    stats: dict[str, int] = {}
    for sp_name in ("train", "val"):
        if sp_name not in split_units_map:
            continue
        units, wav_dirs = split_units_map[sp_name]
        out_sp_dir = args.output_dir / sp_name
        cap = args.max_val_samples if sp_name == "val" else args.max_samples
        cnt = process_lyric_sections_to_pt(
            sections=units,
            wav_dirs=wav_dirs,
            output_dir=out_sp_dir,
            cfg=cfg,
            device=device,
            sp_encode_fn=sp_encode_fn,
            mimi_encode_fn=mimi_encode_fn,
            vocal_mode=args.vocal_mode,
            lead_frames=args.monologue_lead_frames,
            skip_existing=not args.overwrite,
            max_samples=cap,
        )
        stats[sp_name] = cnt

    verified_counts: dict[str, dict[str, int]] = {}
    if not args.no_verify:
        for sp_name in ("train", "val"):
            sp_dir = args.output_dir / sp_name
            if sp_dir.exists() and any(sp_dir.rglob("*.pt")):
                sft_n, grpo_n = verify_preprocessed_split(sp_dir, cfg)
                verified_counts[sp_name] = {"sft_samples": sft_n, "grpo_prompts": grpo_n}
                logger.info(
                    "[Verify/%s] Stage 1 SFT(%d개) & Stage 2 Singability GRPO(%d개) 데이터셋 검증 통과!",
                    sp_name,
                    sft_n,
                    grpo_n,
                )

    train_dir = args.output_dir / "train"
    val_dir = args.output_dir / "val"
    manifest = {
        "paper": "arXiv:2309.11093 (K-pop Lyric Translation)",
        "preset": args.preset,
        "direction": args.direction,
        "codec": args.mimi_model if args.codec == "mimi" else "mini_demo",
        "tokenizer_path": str(tok_path),
        "text_card": cfg.text_card,
        "audio_card": cfg.audio_card,
        "total_streams": cfg.total_streams,
        "saved_samples": stats,
        "verified": verified_counts,
        "commands": {
            "stage1_sft": (
                f'python train.py --preset {args.preset} --vocab-size {cfg.text_card} '
                f'--data-dir "{train_dir}" --val-data-dir "{val_dir}"'
            ),
            "stage2_grpo": (
                f'python train_grpo.py --preset {args.preset} --vocab-size {cfg.text_card} '
                f'--data-dir "{train_dir}" --val-data-dir "{val_dir}" --tokenizer "{tok_path}"'
            ),
            "inference": (
                f'python run.py --preset {args.preset} --tokenizer "{tok_path}"'
            ),
        },
    }
    manifest_path = args.output_dir / "preprocessing_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print("[Done] arXiv:2309.11093 노래 가사·음절·음소·보컬 데이터 분할 전처리 완료!")
    print(f"  - 출력 루트 디렉터리 : {args.output_dir}")
    print(f"  - 가사 토크나이저    : {tok_path} (vocab_size={sp_vocab})")
    print(f"  - 저장된 섹션 샘플   : {stats}")
    print("=" * 78)
