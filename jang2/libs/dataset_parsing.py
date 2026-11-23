"""AI-Hub 71693·71379 데이터셋의 Training/Validation 디렉터리 탐색 및 저메모리 병렬 JSON 파서 모듈."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, List, Optional, Sequence, Set, Tuple

from libs.preprocessing_common import (
    BroadcastUtterance71379,
    UtteranceSpan,
    batched_pool_map,
    fast_scandir_files,
    interleave_lists,
    mark_71379_segment_breaks,
    mark_71693_segment_breaks,
    parse_71379_scene_and_seq,
)
from libs.tokenizer import (
    classify_71379_accent,
    clean_transcript_text,
    extract_json_records,
    is_71693_record,
    normalize_record_keys,
    parse_hhmmss,
)

logger = logging.getLogger(__name__)

_SlimTransTuple = Tuple[str, str, str, str, str, str, str, str, float, float]


@dataclass
class SplitLayout71693:
    """AI-Hub 71693 단일 분할의 원천 WAV 및 라벨 JSON 경로 묶음입니다."""

    split_name: str
    wav_dirs: List[Path]
    json_dirs: List[Path]


@dataclass
class SplitLayout71379:
    """AI-Hub 71379 단일 분할의 한/영 WAV 및 전사/통번역 JSON 경로 묶음입니다."""

    split_name: str
    ko_wav_dirs: List[Path]
    en_wav_dirs: List[Path]
    stt_json_dirs: List[Path]
    trans_json_dirs: List[Path]


def discover_71693_splits(
    root_dir: Path,
    requested_splits: Set[str],
) -> Tuple[List[SplitLayout71693], List[Path]]:
    """AI-Hub 71693 루트 구조에서 Training/Validation 경로와 Sublabel 한국어 코퍼스 경로를 탐색합니다."""
    layouts: List[SplitLayout71693] = []
    sublabel_ko_dirs: List[Path] = []
    if not root_dir.exists():
        return layouts, sublabel_ko_dirs

    sub_ko = root_dir / "Sublabel" / "SbL" / "1.한국어"
    if sub_ko.exists():
        sublabel_ko_dirs.append(sub_ko)

    split_map = [("Training", "train"), ("Validation", "val")]
    found_standard = False
    for folder_name, split_tag in split_map:
        sp_root = root_dir / folder_name
        if not sp_root.exists():
            continue
        found_standard = True
        if split_tag not in requested_splits:
            continue
        raw_dir = sp_root / "01.원천데이터"
        lbl_dir = sp_root / "02.라벨링데이터"
        wav_dirs = [raw_dir] if raw_dir.exists() else [sp_root]
        json_dirs = [lbl_dir] if lbl_dir.exists() else [sp_root]
        layouts.append(SplitLayout71693(split_name=split_tag, wav_dirs=wav_dirs, json_dirs=json_dirs))

    if not found_standard and "train" in requested_splits:
        layouts.append(SplitLayout71693(split_name="train", wav_dirs=[root_dir], json_dirs=[root_dir]))

    return layouts, sublabel_ko_dirs


def discover_71379_splits(
    root_dir: Path,
    requested_splits: Set[str],
    categories: Optional[Set[str]] = None,
) -> List[SplitLayout71379]:
    """AI-Hub 71379 루트 구조에서 Training/Validation의 한국어음성·발화녹음·음성전사·통번역 폴더를 탐색합니다."""
    layouts: List[SplitLayout71379] = []
    if not root_dir.exists():
        return layouts

    def _cat_ok(folder_name: str) -> bool:
        if not categories:
            return True
        return any(cat in folder_name for cat in categories)

    split_map = [("Training", "train"), ("Validation", "val")]
    found_standard = False
    for folder_name, split_tag in split_map:
        sp_root = root_dir / folder_name
        if not sp_root.exists():
            continue
        found_standard = True
        if split_tag not in requested_splits:
            continue

        raw_root = sp_root / "01.원천데이터"
        lbl_root = sp_root / "02.라벨링데이터"
        ko_wav_dirs: List[Path] = []
        en_wav_dirs: List[Path] = []
        stt_json_dirs: List[Path] = []
        trans_json_dirs: List[Path] = []

        if raw_root.exists():
            for d in sorted(raw_root.iterdir()):
                if d.is_dir() and _cat_ok(d.name):
                    ko_wav_dirs.append(d)

        if lbl_root.exists():
            for d in sorted(lbl_root.iterdir()):
                if not d.is_dir() or not _cat_ok(d.name):
                    continue
                if "발화녹음" in d.name:
                    en_wav_dirs.append(d)
                elif "음성전사" in d.name:
                    stt_json_dirs.append(d)
                elif "통번역" in d.name:
                    trans_json_dirs.append(d)
                else:
                    en_wav_dirs.append(d)
                    trans_json_dirs.append(d)

        if trans_json_dirs and en_wav_dirs:
            layouts.append(
                SplitLayout71379(
                    split_name=split_tag,
                    ko_wav_dirs=ko_wav_dirs,
                    en_wav_dirs=en_wav_dirs,
                    stt_json_dirs=stt_json_dirs,
                    trans_json_dirs=trans_json_dirs,
                )
            )

    if not found_standard and "train" in requested_splits:
        layouts.append(
            SplitLayout71379(
                split_name="train",
                ko_wav_dirs=[root_dir],
                en_wav_dirs=[root_dir],
                stt_json_dirs=[root_dir],
                trans_json_dirs=[root_dir],
            )
        )

    return layouts


def _read_json_str(path_str: str) -> Optional[Any]:
    """UTF-8(BOM 포함) JSON 파일을 읽어 파싱된 객체를 반환하고 실패 시 None을 반환합니다."""
    try:
        with open(path_str, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def parse_71693_split(
    layout: SplitLayout71693,
    use_mt_fallback: bool = True,
    max_lectures: Optional[int] = None,
    num_io_threads: int = 16,
) -> dict[str, List[UtteranceSpan]]:
    """71693 라벨 JSON들을 도메인별 균등 순서로 저메모리 배치 파싱합니다."""
    domain_dirs: List[Path] = []
    for jd in layout.json_dirs:
        if jd.exists():
            subs = sorted(d for d in jd.iterdir() if d.is_dir())
            domain_dirs.extend(subs if subs else [jd])
    per_domain_jsons = [fast_scandir_files([dd], ".json")[1] for dd in domain_dirs]
    json_paths: List[str] = interleave_lists(per_domain_jsons)
    if max_lectures is not None and max_lectures > 0:
        json_paths = json_paths[: max_lectures * 2]

    grouped: dict[str, List[UtteranceSpan]] = defaultdict(list)
    skipped_intervals: dict[str, List[Tuple[float, float]]] = defaultdict(list)
    seen_keys: Set[Tuple[str, str]] = set()

    for payload in batched_pool_map(
        _read_json_str, json_paths, num_workers=num_io_threads, batch_size=256
    ):
        if payload is None:
            continue
        for raw in extract_json_records(payload):
            _, item = normalize_record_keys(raw)
            if not is_71693_record(item):
                continue
            src_lang = str(item.get("source_language") or "").strip().lower()
            tgt_lang = str(item.get("target_language") or "").strip().lower()
            if not (src_lang.startswith("en") and tgt_lang.startswith("ko")):
                continue
            fname = str(item.get("file_name") or "").strip()
            sn = str(item.get("sn") or "").strip()
            if not fname or not sn or (fname, sn) in seen_keys:
                continue
            try:
                s_sec = parse_hhmmss(item["audio_start"])
                e_sec = parse_hhmmss(item["audio_end"])
            except (KeyError, ValueError, TypeError):
                continue
            if e_sec <= s_sec:
                continue
            seen_keys.add((fname, sn))

            ko_text = clean_transcript_text(item.get("mtpe"))
            if not ko_text and use_mt_fallback:
                ko_text = clean_transcript_text(item.get("mt"))
            en_text = clean_transcript_text(item.get("source_cleaned")) or clean_transcript_text(
                item.get("source_original")
            )
            if not ko_text:
                skipped_intervals[fname].append((s_sec, e_sec))
                continue

            grouped[fname].append(
                UtteranceSpan(
                    sn=sn,
                    file_name=fname,
                    start_sec=s_sec,
                    end_sec=e_sec,
                    en_text=en_text,
                    ko_mtpe=ko_text,
                    gender=clean_transcript_text(item.get("info_gender")),
                )
            )

    ordered_fnames = list(grouped.keys())
    if max_lectures is not None and max_lectures > 0:
        ordered_fnames = ordered_fnames[:max_lectures]

    final_grouped: dict[str, List[UtteranceSpan]] = {}
    for fname in ordered_fnames:
        spans = grouped[fname]
        mark_71693_segment_breaks(spans, skipped_intervals.get(fname))
        final_grouped[fname] = spans

    return final_grouped


def extract_sublabel_korean_sentences(
    sublabel_dirs: Sequence[Path],
    num_io_threads: int = 16,
) -> List[str]:
    """Sublabel 한국어 학술 강연 문장을 저메모리로 추출합니다."""
    if not sublabel_dirs:
        return []
    _, json_paths = fast_scandir_files(sublabel_dirs, ".json")
    sentences: List[str] = []
    for payload in batched_pool_map(
        _read_json_str, json_paths, num_workers=num_io_threads, batch_size=512
    ):
        if not isinstance(payload, list):
            continue
        for rec in payload:
            if isinstance(rec, dict):
                s = clean_transcript_text(rec.get("source_sentence"))
                if s:
                    sentences.append(s)
    return sentences


def parse_71379_split(
    layout: SplitLayout71379,
    en_accent: str = "all",
    use_mt_fallback: bool = True,
    max_utterances: Optional[int] = None,
    num_io_threads: int = 16,
) -> dict[str, List[BroadcastUtterance71379]]:
    """71379 통번역·음성전사 JSON을 고정 크기 배치 스트리밍으로 읽어 시스템 RAM 스파이크를 방지합니다."""
    t0 = time.perf_counter()
    per_cat_trans: List[List[str]] = []
    for td in layout.trans_json_dirs:
        _, t_list = fast_scandir_files([td], ".json")
        t_filtered = [
            p for p in t_list if not p.lower().endswith(("_es.json", "_ru.json"))
        ]
        if max_utterances is not None and max_utterances > 0:
            per_cat_cap = max(
                16, int((max_utterances * 1.3) / max(1, len(layout.trans_json_dirs))) + 16
            )
            t_filtered = t_filtered[:per_cat_cap]
        per_cat_trans.append(t_filtered)

    block_sz = 24
    interleaved_blocks = interleave_lists(
        [
            [grp[i : i + block_sz] for i in range(0, len(grp), block_sz)]
            for grp in per_cat_trans
        ]
    )
    trans_paths: List[str] = [p for blk in interleaved_blocks for p in blk]
    del per_cat_trans, interleaved_blocks

    stt_index: dict[str, str] = {}
    if layout.stt_json_dirs and layout.stt_json_dirs != layout.trans_json_dirs:
        stt_by_name, _ = fast_scandir_files(layout.stt_json_dirs, ".json")
        stt_index = {os.path.splitext(k)[0]: v for k, v in stt_by_name.items()}
        del stt_by_name

    logger.info(
        "[71379/%s] 통번역 JSON %d개 저메모리 배치 파싱 시작 (threads=%d)...",
        layout.split_name,
        len(trans_paths),
        num_io_threads,
    )

    def _read_trans_slim(jpath: str) -> Optional[_SlimTransTuple]:
        payload = _read_json_str(jpath)
        if not isinstance(payload, dict):
            return None
        _, item = normalize_record_keys(payload)
        if "t-fname" not in item and "최종번역문" not in item and "원문" not in item:
            return None
        s_code = str(item.get("s-code") or "ko-KR").strip().lower()
        if s_code and not s_code.startswith("ko"):
            return None
        if str(item.get("비식별화") or "").strip().upper() == "O":
            return None

        t_code = str(item.get("t-code") or "").strip()
        en_wav_name = clean_transcript_text(item.get("t-fname"))
        variant = classify_71379_accent(t_code, en_wav_name, os.path.basename(jpath))
        if variant is None:
            return None
        if en_accent != "all" and variant != en_accent:
            return None

        ko_wav_name = clean_transcript_text(item.get("s-fname"))
        if not en_wav_name and ko_wav_name:
            stem0 = os.path.splitext(ko_wav_name)[0]
            en_wav_name = f"{stem0}{'_uk.wav' if variant == 'uk' else '_en.wav'}"
        if not en_wav_name:
            return None

        ko_text = clean_transcript_text(item.get("수정원문")) or clean_transcript_text(item.get("원문"))
        en_text = (
            clean_transcript_text(item.get("수정번역문"))
            or clean_transcript_text(item.get("최종번역문"))
            or clean_transcript_text(item.get("최종번역"))
            or clean_transcript_text(item.get("2차수정"))
            or clean_transcript_text(item.get("1차수정"))
        )
        if not en_text and use_mt_fallback:
            en_text = clean_transcript_text(item.get("mt"))

        try:
            s_ulen = max(0.0, float(item.get("s-ulength") or 0.0))
        except (TypeError, ValueError):
            s_ulen = 0.0
        try:
            t_ulen = max(0.0, float(item.get("t-ulength") or 0.0))
        except (TypeError, ValueError):
            t_ulen = 0.0

        return (
            variant,
            ko_wav_name,
            en_wav_name,
            ko_text,
            en_text,
            clean_transcript_text(item.get("발화자id")),
            clean_transcript_text(item.get("s-usex")).lower(),
            clean_transcript_text(item.get("t-usex")).lower(),
            s_ulen,
            t_ulen,
        )

    slim_items: List[_SlimTransTuple] = []
    needed_stt_stems: Set[str] = set()

    for idx, res in enumerate(
        batched_pool_map(_read_trans_slim, trans_paths, num_workers=num_io_threads, batch_size=4096),
        start=1,
    ):
        if idx % 100_000 == 0:
            logger.info(
                "[71379/%s] 통번역 JSON 파싱 진행: %d / %d (%.1fs)",
                layout.split_name,
                idx,
                len(trans_paths),
                time.perf_counter() - t0,
            )
        if res is None:
            continue
        slim_items.append(res)
        if res[1]:
            needed_stt_stems.add(os.path.splitext(os.path.basename(res[1]))[0].lower())
        if max_utterances is not None and len(slim_items) >= int(max_utterances * 1.15) + 32:
            break

    del trans_paths

    stt_meta_map: dict[str, Tuple[str, Optional[float], Optional[float], bool]] = {}
    stt_targets = [(stem, stt_index[stem]) for stem in needed_stt_stems if stem in stt_index]
    del stt_index, needed_stt_stems

    if stt_targets:
        logger.info(
            "[71379/%s] 매칭 음성전사 JSON %d개 저메모리 배치 로드 중...",
            layout.split_name,
            len(stt_targets),
        )

        def _read_stt_item(
            pair: Tuple[str, str],
        ) -> Optional[Tuple[str, str, Optional[float], Optional[float], bool]]:
            stem, sp = pair
            payload = _read_json_str(sp)
            if not isinstance(payload, dict):
                return None
            _, item = normalize_record_keys(payload)
            is_deid = str(item.get("비식별화") or "").strip().upper() == "O"
            ko_stt = (
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
            return stem, ko_stt, raw_s, raw_e, is_deid

        for stt_res in batched_pool_map(
            _read_stt_item, stt_targets, num_workers=num_io_threads, batch_size=4096
        ):
            if stt_res is not None:
                stem, ko_stt, raw_s, raw_e, is_deid = stt_res
                stt_meta_map[stem] = (ko_stt, raw_s, raw_e, is_deid)
    del stt_targets

    grouped: dict[str, List[BroadcastUtterance71379]] = defaultdict(list)
    skipped_seqs: dict[str, Set[int]] = defaultdict(set)
    seen_utt_ids: Set[str] = set()
    kept_utterances = 0

    for (
        variant,
        ko_wav_name,
        en_wav_name,
        ko_text,
        en_text,
        speaker_id,
        s_sex,
        t_sex,
        s_ulen,
        t_ulen,
    ) in slim_items:
        utt_id = os.path.splitext(os.path.basename(en_wav_name))[0]
        if utt_id.lower() in seen_utt_ids:
            continue
        seen_utt_ids.add(utt_id.lower())

        scene_key, seq_num = parse_71379_scene_and_seq(ko_wav_name, en_wav_name)
        group_key = f"{scene_key}_{variant}"

        ko_stem = os.path.splitext(os.path.basename(ko_wav_name))[0].lower() if ko_wav_name else ""
        stt_info = stt_meta_map.get(ko_stem)
        if stt_info is not None and stt_info[3]:
            skipped_seqs[group_key].add(seq_num)
            continue

        if not ko_text and stt_info is not None:
            ko_text = stt_info[0]
        if not ko_text or not en_text:
            skipped_seqs[group_key].add(seq_num)
            continue

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
                speaker_id=speaker_id,
                s_sex=s_sex,
                t_sex=t_sex,
                s_ulength=s_ulen,
                t_ulength=t_ulen,
                raw_start_sec=stt_info[1] if stt_info else None,
                raw_end_sec=stt_info[2] if stt_info else None,
            )
        )
        kept_utterances += 1
        if max_utterances is not None and kept_utterances >= max_utterances:
            break

    del slim_items, stt_meta_map, seen_utt_ids

    for group_key, recs in grouped.items():
        mark_71379_segment_breaks(recs, skipped_seqs.get(group_key))

    logger.info(
        "[71379/%s] 파싱 완료: %d개 씬 그룹, %d개 유효 발화 (%.1fs)",
        layout.split_name,
        len(grouped),
        kept_utterances,
        time.perf_counter() - t0,
    )
    return grouped


def parse_single_split(
    sp_name: str,
    ly93_map: dict[str, SplitLayout71693],
    ly79_map: dict[str, SplitLayout71379],
    args: argparse.Namespace,
) -> Tuple[Optional[dict[str, List[UtteranceSpan]]], Optional[dict[str, List[BroadcastUtterance71379]]]]:
    """메모리 중복 점유를 피하기 위해 요청된 단일 분할(`train` 또는 `val`)만 파싱합니다."""
    g93: Optional[dict[str, List[UtteranceSpan]]] = None
    g79: Optional[dict[str, List[BroadcastUtterance71379]]] = None

    if sp_name in ly93_map:
        max_lec = args.max_lectures_71693
        cap_93 = args.max_val_samples if sp_name == "val" else args.max_samples_71693
        if sp_name == "val" and cap_93 is None and args.max_samples_71693 is not None:
            cap_93 = max(1, args.max_samples_71693 // 5)
        if max_lec is None and cap_93 is not None:
            max_lec = max(2, (cap_93 // 10) + 2)
        logger.info("[71693/%s] 라벨 JSON 파싱 중...", sp_name)
        g93 = parse_71693_split(
            ly93_map[sp_name],
            use_mt_fallback=not args.no_mt_fallback,
            max_lectures=max_lec,
            num_io_threads=args.io_threads,
        )
        logger.info(
            "[71693/%s] 파싱 완료: 강의 %d개, 발화 %d개",
            sp_name,
            len(g93),
            sum(len(v) for v in g93.values()),
        )

    if sp_name in ly79_map:
        cap_79 = args.max_val_samples if sp_name == "val" else args.max_samples_71379
        if sp_name == "val" and cap_79 is None and args.max_samples_71379 is not None:
            cap_79 = max(1, args.max_samples_71379 // 5)
        max_utts = max(32, cap_79 * 10) if cap_79 is not None else None
        g79 = parse_71379_split(
            ly79_map[sp_name],
            en_accent=args.en_accent_71379,
            use_mt_fallback=not args.no_mt_fallback,
            max_utterances=max_utts,
            num_io_threads=args.io_threads,
        )

    return g93, g79
