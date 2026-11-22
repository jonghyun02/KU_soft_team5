"""AI-Hub 71693(국제학술대회) 및 71379(방송콘텐츠) 영-한 데이터셋 통합 전처리 파이프라인 모듈."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import operator
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

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
    resample_wav,
)
from libs.config import EDarchConfig
from libs.env import atomic_torch_save, setup_device
from libs.tokenizer import (
    TEXT_BOS_ID,
    TEXT_EOS_ID,
    TEXT_PAD_ID,
    TEXT_UNK_ID,
    load_korean_tokenizer,
    resolve_spm_model_path,
    train_korean_sentencepiece,
)

logger = logging.getLogger(__name__)

ZERO_TOKEN_ID = -1
DEFAULT_CONTEXT_FRAMES = 1000

_EMPTY_TEXT_MARKERS = {"", "-", "n/a", "na", "null", "none", "x"}
_LEADING_TRANSCRIPT_NOISE_RE = re.compile(r"^(?:##+|\*\*+|//+)\s*")
_SPLIT_STEM_RE = re.compile(r"^(.*_win\d{4})(_[ab]+)$")

MappingOrDict = Dict[Any, Any]


def _index_existing_window_trees(output_dir: Path) -> Tuple[set[str], set[str]]:
    """출력 디렉터리 내 기존 `.pt` 스템 집합과 이진 분할 상위 윈도우 스템 집합을 인덱싱합니다."""
    existing_stems: set[str] = set()
    split_ancestors: set[str] = set()
    if not output_dir.exists():
        return existing_stems, split_ancestors
    for p in output_dir.glob("*.pt"):
        if not p.is_file():
            continue
        stem = p.stem
        existing_stems.add(stem)
        m = _SPLIT_STEM_RE.match(stem)
        if m is not None:
            root_base, branch_sfx = m.group(1), m.group(2)
            split_ancestors.add(root_base)
            for k in range(2, len(branch_sfx)):
                split_ancestors.add(f"{root_base}{branch_sfx[:k]}")
    return existing_stems, split_ancestors


def _count_completed_window_tree(
    base_stem: str, num_items: int, existing_stems: set[str]
) -> Optional[int]:
    """윈도우 또는 이진 분할 하위 트리가 모두 저장되었으면 리프 파일 개수를, 미완료면 None을 반환합니다."""
    if base_stem in existing_stems:
        return 1
    if num_items <= 1:
        return None
    mid = num_items // 2
    left_cnt = _count_completed_window_tree(f"{base_stem}_a", mid, existing_stems)
    if left_cnt is None:
        return None
    right_cnt = _count_completed_window_tree(
        f"{base_stem}_b", num_items - mid, existing_stems
    )
    if right_cnt is None:
        return None
    return left_cnt + right_cnt


def parse_hhmmss(ts: Union[str, int, float]) -> float:
    """시:분:초 문자열 또는 숫자 타임스탬프를 초 단위 부동소수점 값으로 변환합니다."""
    if isinstance(ts, (int, float)) and not isinstance(ts, bool):
        return float(ts)
    if not isinstance(ts, str):
        raise ValueError(f"타임스탬프 문자열이 아닙니다: {ts!r}")
    cleaned = ts.strip().replace(",", ".")
    if not cleaned:
        raise ValueError("빈 타임스탬프입니다.")
    parts = cleaned.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    if len(parts) == 1:
        return float(parts[0])
    raise ValueError(f"지원하지 않는 타임스탬프 형식입니다: {ts!r}")


def _clean_text_field(val: Any) -> str:
    """전사·번역 문자열 필드에서 불필요한 노이즈 기호를 제거하고 빈 마커를 걸러냅니다."""
    if val is None or isinstance(val, bool):
        return ""
    text = str(val).strip()
    text = _LEADING_TRANSCRIPT_NOISE_RE.sub("", text).strip()
    if text.lower() in _EMPTY_TEXT_MARKERS:
        return ""
    return text


def _normalize_record_keys(raw: MappingOrDict) -> Tuple[dict[str, Any], dict[str, Any]]:
    """JSON 레코드의 키에서 BOM과 공백을 제거한 원본 사전 및 소문자 키 사전을 생성합니다."""
    norm: dict[str, Any] = {}
    lower_norm: dict[str, Any] = {}
    for k, v in raw.items():
        ck = str(k).lstrip("\ufeff").strip()
        norm[ck] = v
        lower_norm[ck.lower()] = v
    return norm, lower_norm


def _extract_json_records(payload: Any) -> List[dict[str, Any]]:
    """다양한 AI-Hub JSON 최상위 구조(리스트 또는 딕셔너리)에서 발화 레코드 리스트를 추출합니다."""
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        norm, lower_norm = _normalize_record_keys(payload)
        for list_key in ("data", "utterances", "dialogue", "dialogs", "sentences"):
            cand = lower_norm.get(list_key)
            if isinstance(cand, list):
                return [x for x in cand if isinstance(x, dict)]
        return [norm]
    return []


def _index_wav_files(
    *dirs: Optional[Path],
    prefer_keywords: Tuple[str, ...] = (),
) -> Tuple[dict[str, Path], List[Path]]:
    """지정된 디렉터리들에서 WAV 파일을 대소문자 무관하게 인덱싱하고 경로 키워드 우선순위를 적용합니다."""
    by_lower_name: dict[str, Path] = {}
    best_score: dict[str, int] = {}
    all_paths: List[Path] = []
    seen_resolved: set[Path] = set()

    def _path_score(p: Path) -> int:
        """부모 디렉터리 경로에 선호 키워드가 포함된 개수를 점수로 계산합니다."""
        if not prefer_keywords:
            return 0
        parent_str = str(p.parent).lower()
        return sum(1 for kw in prefer_keywords if kw in parent_str)

    for d in dirs:
        if d is None or not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if not p.is_file() or p.suffix.lower() != ".wav":
                continue
            resolved = p.resolve()
            if resolved in seen_resolved:
                continue
            seen_resolved.add(resolved)
            all_paths.append(p)
            lname = p.name.lower()
            sc = _path_score(p)
            if lname not in by_lower_name or sc > best_score[lname]:
                by_lower_name[lname] = p
                best_score[lname] = sc
    return by_lower_name, all_paths


def _resolve_wav_path(
    wav_dir: Optional[Path],
    file_name: str,
    wav_index: dict[str, Path],
) -> Optional[Path]:
    """파일명 또는 스템 이름을 이용해 인덱싱된 WAV 파일의 실제 경로를 탐색합니다."""
    if not file_name:
        return None
    if wav_dir is not None:
        direct = wav_dir / file_name
        if direct.exists():
            return direct
    name_only = Path(file_name).name.lower()
    if name_only in wav_index:
        return wav_index[name_only]
    stem_wav = f"{Path(file_name).stem.lower()}.wav"
    return wav_index.get(stem_wav)


def _validate_ko_token_ids(ko_token_ids: Sequence[int], text_card: Optional[int]) -> None:
    """인코딩된 한국어 토큰 ID들이 허용 어휘 범위에 속하며 특수 예약 토큰을 침범하지 않는지 검증합니다."""
    if any(
        tid < 0
        or tid in (TEXT_EOS_ID, TEXT_PAD_ID)
        or (text_card is not None and tid >= text_card)
        for tid in ko_token_ids
    ):
        upper_msg = f", < {text_card}" if text_card is not None else ""
        raise ValueError(
            f"ko_sp_encode_fn 출력에 허용 범위를 벗어난 토큰(0 이상{upper_msg}) 또는 "
            f"예약 특수 토큰(EOS={TEXT_EOS_ID}, PAD={TEXT_PAD_ID})이 포함되어 있습니다."
        )


def _validate_sim_bin(raw_sim_bin: Any, num_speaker_sim_bins: int) -> int:
    """화자 유사도 반환값이 유효한 정수 구간 인덱스인지 검증합니다."""
    if isinstance(raw_sim_bin, bool):
        raise ValueError(
            f"wavlm_sim_bin_fn 반환값은 정수 bin 인덱스(0..{num_speaker_sim_bins - 1})여야 합니다: {raw_sim_bin!r}"
        )
    try:
        sim_bin = operator.index(raw_sim_bin)
    except TypeError as exc:
        raise ValueError(
            f"wavlm_sim_bin_fn 반환값은 정수 bin 인덱스(0..{num_speaker_sim_bins - 1})여야 합니다: {raw_sim_bin!r}"
        ) from exc
    if sim_bin < 0 or sim_bin >= num_speaker_sim_bins:
        raise ValueError(
            f"wavlm_sim_bin_fn 반환값({sim_bin})이 허용 범위 [0, {num_speaker_sim_bins - 1}]를 벗어났습니다."
        )
    return sim_bin


def _encode_and_align_mimi(
    timeline: torch.Tensor,
    mimi_encode_fn: Callable[[torch.Tensor], torch.Tensor],
    total_frames: int,
    expected_q: Optional[int],
) -> torch.Tensor:
    """오디오 타임라인을 Mimi 코덱으로 인코딩하고 목표 프레임 길이에 맞춰 정렬합니다."""
    raw_c = mimi_encode_fn(timeline.view(1, 1, -1)).detach().cpu().long()
    c = raw_c.squeeze(0) if (raw_c.ndim == 3 and raw_c.shape[0] == 1) else raw_c
    if c.ndim != 2:
        raise ValueError(
            f"mimi_encode_fn 출력은 [1, Q, T] 또는 [Q, T]여야 합니다. 실제 shape={tuple(raw_c.shape)}"
        )
    if expected_q is not None and c.shape[0] != expected_q:
        raise ValueError(f"Mimi 코드북 수 불일치: 기대 Q={expected_q}, 실제 {c.shape[0]}")
    if c.shape[1] >= total_frames:
        return c[:, :total_frames]
    pad_len = total_frames - c.shape[1]
    if c.shape[1] > 0:
        pad = c[:, -1:].expand(c.shape[0], pad_len).contiguous()
    else:
        pad = torch.full((c.shape[0], pad_len), ZERO_TOKEN_ID, dtype=torch.long)
    return torch.cat([c, pad], dim=1)


def _normalize_tts_output(
    tts_out: Union[torch.Tensor, Tuple[torch.Tensor, int]],
) -> torch.Tensor:
    """제로샷 TTS 함수 출력을 24kHz 1차원 CPU Float32 파형 텐서로 정규화합니다."""
    if isinstance(tts_out, tuple):
        wav_out, tts_sr = tts_out
        wav_out = wav_out.detach().cpu().float().reshape(-1)
        if tts_sr != SAMPLE_RATE:
            wav_out = resample_wav(wav_out, int(tts_sr), SAMPLE_RATE)
        return wav_out
    return tts_out.detach().cpu().float().reshape(-1)


@dataclass
class UtteranceSpan:
    """AI-Hub 71693 학술대회 발화 단위의 시간 구간 및 영-한 병렬 텍스트 정보를 담는 데이터클래스입니다."""

    sn: str
    file_name: str
    start_sec: float
    end_sec: float
    en_text: str
    ko_mtpe: str
    segment_break: bool = False
    direction: str = "en2ko"


def _is_71693_record(lower_item: dict[str, Any]) -> bool:
    """JSON 레코드가 AI-Hub 71693 스키마에 해당하는지 판별합니다."""
    return (
        "source_language" in lower_item
        and "target_language" in lower_item
        and ("audio_start" in lower_item or "mtpe" in lower_item)
    )


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

        for raw_item in _extract_json_records(payload):
            _, item = _normalize_record_keys(raw_item)
            if not _is_71693_record(item):
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

            tgt_trans = _clean_text_field(item.get("mtpe"))
            if not tgt_trans and use_mt_fallback:
                tgt_trans = _clean_text_field(item.get("mt"))
            src_trans = _clean_text_field(item.get("source_cleaned")) or _clean_text_field(
                item.get("source_original")
            )

            if is_en2ko:
                en_text = src_trans
                ko_text = tgt_trans
                direction = "en2ko"
            else:
                ko_text = src_trans
                en_text = tgt_trans
                direction = "ko2en"

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
                    direction=direction,
                )
            )

    for fname, spans in grouped.items():
        spans.sort(key=lambda x: (x.start_sec, x.end_sec))
        sk_list = skipped_intervals.get(fname)
        for idx in range(1, len(spans)):
            prev_end = spans[idx - 1].end_sec
            cur_start = spans[idx].start_sec
            if spans[idx].direction != spans[idx - 1].direction:
                spans[idx].segment_break = True
            elif sk_list and cur_start > prev_end and any(
                sk_e > prev_end and sk_s < cur_start for sk_s, sk_e in sk_list
            ):
                spans[idx].segment_break = True
    return grouped


def slice_60s_windows(
    spans: List[UtteranceSpan],
    target_window_sec: float = 45.0,
    max_gap_sec: float = 8.0,
) -> List[List[UtteranceSpan]]:
    """연속된 71693 발화들을 최대 휴지기와 목표 윈도우 길이에 맞춰 에피소드 단위로 분할합니다."""
    windows: List[List[UtteranceSpan]] = []
    cur: List[UtteranceSpan] = []
    cur_max_end = 0.0
    for sp in spans:
        if not cur:
            cur = [sp]
            cur_max_end = sp.end_sec
            continue
        gap = sp.start_sec - cur_max_end
        win_len = max(cur_max_end, sp.end_sec) - cur[0].start_sec
        if sp.segment_break or gap > max_gap_sec or win_len > target_window_sec:
            windows.append(cur)
            cur = [sp]
            cur_max_end = sp.end_sec
        else:
            cur.append(sp)
            cur_max_end = max(cur_max_end, sp.end_sec)
    if cur:
        windows.append(cur)
    return windows


def extract_10s_speaker_prompt(
    lecture_wav_24k: torch.Tensor,
    spans: List[UtteranceSpan],
    prompt_sec: float = 10.0,
) -> torch.Tensor:
    """강의 음성에서 제로샷 TTS 및 화자 유사도 평가에 사용할 목표 길이의 참조 음성을 추출합니다."""
    target_samples = int(prompt_sec * SAMPLE_RATE)
    wav_len = lecture_wav_24k.numel()
    chunks = []
    fallback_chunks = []
    collected = 0
    for sp in spans:
        s_idx = max(0, min(int(sp.start_sec * SAMPLE_RATE), wav_len))
        e_idx = max(s_idx, min(int(sp.end_sec * SAMPLE_RATE), wav_len))
        clip = lecture_wav_24k[s_idx:e_idx]
        if clip.numel() > SAMPLE_RATE:
            chunks.append(clip)
            collected += clip.numel()
            if collected >= target_samples:
                break
        elif clip.numel() > 0:
            fallback_chunks.append(clip)

    if chunks:
        if collected < target_samples and fallback_chunks:
            chunks.extend(fallback_chunks)
        return torch.cat(chunks, dim=0)[:target_samples]
    if fallback_chunks:
        return torch.cat(fallback_chunks, dim=0)[:target_samples]
    if wav_len > 0:
        return lecture_wav_24k[:target_samples]
    return torch.zeros(target_samples, dtype=lecture_wav_24k.dtype, device=lecture_wav_24k.device)


def _assemble_virtual_lecture_from_sn_clips(
    spans: List[UtteranceSpan],
    wav_dir: Path,
    wav_index: dict[str, Path],
) -> Optional[torch.Tensor]:
    """전체 강의 WAV 대신 발화별 분할 클립(`{sn}.wav`)만 존재할 때 가상 타임라인 WAV를 재조립합니다."""
    max_end_sec = max((sp.end_sec for sp in spans), default=0.0)
    if max_end_sec <= 0.0:
        return None
    found_any = False
    resolved_clips: List[Tuple[UtteranceSpan, Path]] = []
    for sp in spans:
        clip_path = _resolve_wav_path(wav_dir, f"{sp.sn}.wav", wav_index)
        if clip_path is not None:
            found_any = True
            resolved_clips.append((sp, clip_path))
    if not found_any:
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
    zero_shot_ko_tts_fn: Callable[
        [str, torch.Tensor], Union[torch.Tensor, Tuple[torch.Tensor, int]]
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
    _out_metadata: Optional[dict[str, Any]] = None,
    _tts_tok_cache: Optional[dict[str, Tuple[torch.Tensor, List[int]]]] = None,
    _suppress_length_warning: bool = False,
) -> Tuple[torch.Tensor, int]:
    """AI-Hub 71693 발화 윈도우를 인과적 정밀 정렬이 적용된 `[1 + 2Q, T]` 코드북 텐서로 변환합니다."""
    if not window_spans:
        raise ValueError("window_spans는 비어 있을 수 없습니다.")

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
                ko_sent_wav = _normalize_tts_output(
                    zero_shot_ko_tts_fn(sp.ko_mtpe, speaker_prompt_24k)
                )
                ko_token_ids = list(ko_sp_encode_fn(sp.ko_mtpe))
                if _tts_tok_cache is not None:
                    _tts_tok_cache[sp.sn] = (ko_sent_wav, ko_token_ids)

            _validate_ko_token_ids(ko_token_ids, text_card)
            tok_pieces = (
                list(ko_sp_pieces_fn(sp.ko_mtpe)) if ko_sp_pieces_fn is not None else None
            )

            rel_en_start_sample = max(0, int((sp.start_sec - win_start_sec) * SAMPLE_RATE))
            rel_en_end_sample = max(0, int((sp.end_sec - win_start_sec) * SAMPLE_RATE))
            en_start_frame = rel_en_start_sample // SAMPLES_PER_FRAME
            en_end_frame = (rel_en_end_sample + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME
            prev_audio_end_frame = (ko_cursor_sample + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME

            audio_start_sample, token_positions, last_token_frame = compute_aligned_token_placement(
                ko_text=sp.ko_mtpe,
                ko_wav_24k=ko_sent_wav,
                ko_token_ids=ko_token_ids,
                en_end_frame=en_end_frame,
                prev_audio_end_frame=prev_audio_end_frame,
                last_token_frame=last_token_frame,
                align_mode=align_mode,
                ko_align_fn=ko_align_fn,
                token_pieces=tok_pieces,
                monologue_lead_frames=monologue_lead_frames,
            )

            ko_placements.append((audio_start_sample, ko_sent_wav, token_positions))
            src_start_frames.append(int(en_start_frame))
            src_end_frames.append(int(en_end_frame))
            ref_token_groups.append([int(tid) for tid in ko_token_ids])
            ko_cursor_sample = audio_start_sample + ko_sent_wav.numel()

        audio_end_frames = (
            max(en_span_samples, ko_cursor_sample) + SAMPLES_PER_FRAME - 1
        ) // SAMPLES_PER_FRAME
        eos_frame = max(last_token_frame + 1, audio_end_frames)
        total_frames = max(audio_end_frames + 4, eos_frame + 1)
        if (
            not _suppress_length_warning
            and max_context_frames > 0
            and total_frames > max_context_frames
        ):
            logger.warning(
                "생성된 샘플 길이(%d 프레임)가 max_context_frames(%d)를 초과합니다 (%s). "
                "학습 시 EOS 절단을 피하려면 target_window_sec를 줄이세요.",
                total_frames,
                max_context_frames,
                lecture_wav_path.name,
            )
        padded_samples = (total_frames + 2) * SAMPLES_PER_FRAME

        en_timeline = torch.zeros(padded_samples, dtype=torch.float32)
        ko_timeline = torch.zeros(padded_samples, dtype=torch.float32)
        for sp in window_spans:
            sp_s = max(start_idx, min(int(sp.start_sec * SAMPLE_RATE), wav.numel()))
            sp_e = max(sp_s, min(int(sp.end_sec * SAMPLE_RATE), wav.numel()))
            if sp_e > sp_s:
                en_timeline[sp_s - start_idx : sp_e - start_idx] = wav[sp_s:sp_e]

        ko_text_stream = torch.full((1, total_frames), TEXT_PAD_ID, dtype=torch.long)

        for start_sample, ko_sent_wav, token_positions in ko_placements:
            end_sample = start_sample + ko_sent_wav.numel()
            ko_timeline[start_sample:end_sample] = ko_sent_wav
            for f_pos, tid in token_positions:
                ko_text_stream[0, f_pos] = tid

        ko_text_stream[0, eos_frame] = TEXT_EOS_ID

        ko_codes = _encode_and_align_mimi(ko_timeline, mimi_encode_fn, total_frames, expected_q)
        en_codes = _encode_and_align_mimi(en_timeline, mimi_encode_fn, total_frames, expected_q)
        if ko_codes.shape[0] != en_codes.shape[0]:
            raise ValueError(
                f"타깃/소스 코드북 수 불일치: ko={ko_codes.shape[0]}, en={en_codes.shape[0]}"
            )

        active_ko_clips = [w for _, w, _ in ko_placements if w.numel() > 0]
        ko_speech_only = (
            torch.cat(active_ko_clips, dim=0)
            if active_ko_clips
            else ko_timeline[: total_frames * SAMPLES_PER_FRAME]
        )
        sim_bin = _validate_sim_bin(
            wavlm_sim_bin_fn(speaker_prompt_24k, ko_speech_only),
            num_speaker_sim_bins,
        )
        if _out_metadata is not None:
            _out_metadata.clear()
            _out_metadata.update(
                {
                    "src_start_frames": src_start_frames,
                    "src_end_frames": src_end_frames,
                    "ref_token_groups": ref_token_groups,
                    "align_mode": align_mode,
                }
            )
        codes = torch.cat([ko_text_stream, ko_codes, en_codes], dim=0)
        return codes, sim_bin


def save_training_example_pt(
    output_path: Path,
    codes: torch.Tensor,
    sim_bin: int,
    metadata: Optional[Dict[str, Any]] = None,
    max_retries: int = 5,
) -> None:
    """멀티스트림 코드 텐서, 화자 유사도 구간 및 Stage 2 메타데이터를 원자적으로 `.pt`에 저장합니다."""
    payload: Dict[str, Any] = {"codes": codes.detach().cpu(), "sim_bin": int(sim_bin)}
    if metadata:
        for k, v in metadata.items():
            if k not in payload:
                payload[k] = v
    atomic_torch_save(payload, output_path, max_retries=max_retries)


@dataclass
class BroadcastUtterance71379:
    """AI-Hub 71379 방송콘텐츠 통번역 발화 단위의 메타데이터와 파일 정보를 담는 데이터클래스입니다."""

    utt_id: str
    scene_key: str
    seq_num: int
    lang_variant: str
    ko_wav_name: str
    en_wav_name: str
    ko_text: str
    en_text: str
    speaker_id: str = ""
    s_sex: str = ""
    t_sex: str = ""
    s_ulength: float = 0.0
    t_ulength: float = 0.0
    raw_start_sec: Optional[float] = None
    raw_end_sec: Optional[float] = None
    segment_break: bool = False


@dataclass
class _SttMeta71379:
    """71379 원천 한국어 음성전사 JSON에서 추출한 부가 타임스탬프 및 비식별화 정보입니다."""

    ko_text: str
    raw_start_sec: Optional[float]
    raw_end_sec: Optional[float]
    deidentified: bool


def _parse_71379_scene_and_seq(
    ko_wav_name: str,
    en_wav_name: str,
    lower_item: dict[str, Any],
) -> Tuple[str, int]:
    """71379 음성 파일명 또는 분류 필드로부터 방송 씬 식별 키와 발화 순번을 추출합니다."""
    base_name = ko_wav_name or en_wav_name
    stem = Path(base_name).stem
    for suffix in ("_en-us", "_en-gb", "_en_us", "_en_gb", "_en", "_uk"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    parts = stem.split("_")
    if len(parts) >= 6 and parts[5].isdigit():
        return "_".join(parts[:5]), int(parts[5])
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1]), int(parts[-1])

    cat = _clean_text_field(lower_item.get("대분류")) or "bc"
    broadcaster = _clean_text_field(lower_item.get("중분류")) or "ch"
    prog = _clean_text_field(lower_item.get("소분류")) or "prog"
    ts_num = _clean_text_field(lower_item.get("tsnum")) or "0"
    seg_raw = _clean_text_field(lower_item.get("segnum")) or _clean_text_field(
        lower_item.get("순서")
    )
    seq_digits = "".join(ch for ch in seg_raw if ch.isdigit())
    seq_num = int(seq_digits) if seq_digits else 0
    return f"{cat}_{broadcaster}_{prog}_{ts_num}", seq_num


def _classify_71379_t_code(
    t_code_raw: str, en_wav_name: str, json_name: str
) -> Optional[str]:
    """71379 타깃 언어 코드 및 파일명으로부터 미국식('us') 또는 영국식('uk') 영어 레코드만 판별합니다."""
    tc = t_code_raw.strip().lower()
    if tc in ("es", "es-es", "ru", "ru-ru"):
        return None
    if tc in ("en-gb", "uk", "en_gb"):
        return "uk"
    if tc in ("en-us", "en", "en_us", "eng"):
        return "us"

    lower_en = en_wav_name.lower()
    lower_jp = json_name.lower()
    for sfx in ("_es.wav", "_ru.wav", "_es.json", "_ru.json"):
        if lower_en.endswith(sfx) or lower_jp.endswith(sfx):
            return None
    if lower_en.endswith("_uk.wav") or lower_jp.endswith("_uk.json"):
        return "uk"
    if lower_en.endswith("_en.wav") or lower_jp.endswith("_en.json"):
        return "us"
    return None


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

        for raw_rec in _extract_json_records(payload):
            _, item = _normalize_record_keys(raw_rec)
            if _is_71693_record(item):
                continue

            if "음성파일명" in item and "t-fname" not in item and "t-code" not in item:
                ko_fname = _clean_text_field(item.get("음성파일명"))
                if not ko_fname:
                    continue
                stem_key = Path(ko_fname).stem.lower()
                deid_raw = str(item.get("비식별화") or "").strip().upper()
                is_deid = deid_raw == "O"
                if is_deid:
                    deidentified_stems.add(stem_key)

                ko_stt_text = (
                    _clean_text_field(item.get("번역본"))
                    or _clean_text_field(item.get("철자전사"))
                    or _clean_text_field(item.get("발음전사"))
                    or _clean_text_field(item.get("전사작업본"))
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

        ko_wav_name = _clean_text_field(item.get("s-fname"))
        en_wav_name = _clean_text_field(item.get("t-fname"))
        t_code = str(item.get("t-code") or "").strip()

        variant = _classify_71379_t_code(t_code, en_wav_name, jp.name)
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

        scene_key, seq_num = _parse_71379_scene_and_seq(ko_wav_name, en_wav_name, item)
        group_key = f"{scene_key}_{variant}"
        utt_id = Path(en_wav_name).stem

        if utt_id.lower() in seen_ids:
            continue
        seen_ids.add(utt_id.lower())

        if (stt_meta is not None and stt_meta.deidentified) or (ko_stem in deidentified_stems):
            skipped_seqs_by_group[group_key].add(seq_num)
            continue

        ko_text = _clean_text_field(item.get("수정원문")) or _clean_text_field(item.get("원문"))
        if not ko_text and stt_meta is not None:
            ko_text = stt_meta.ko_text

        en_text = (
            _clean_text_field(item.get("수정번역문"))
            or _clean_text_field(item.get("최종번역문"))
            or _clean_text_field(item.get("최종번역"))
            or _clean_text_field(item.get("2차수정"))
            or _clean_text_field(item.get("1차수정"))
        )
        if not en_text and use_mt_fallback:
            en_text = _clean_text_field(item.get("mt"))

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
                speaker_id=_clean_text_field(item.get("발화자id")),
                s_sex=_clean_text_field(item.get("s-usex")).lower(),
                t_sex=_clean_text_field(item.get("t-usex")).lower(),
                s_ulength=max(0.0, s_ulen),
                t_ulength=max(0.0, t_ulen),
                raw_start_sec=stt_meta.raw_start_sec if stt_meta else None,
                raw_end_sec=stt_meta.raw_end_sec if stt_meta else None,
            )
        )

    for group_key, recs in grouped.items():
        recs.sort(key=lambda r: (r.seq_num, r.utt_id))
        sk_seqs = skipped_seqs_by_group.get(group_key, set())
        for idx in range(1, len(recs)):
            prev_r = recs[idx - 1]
            cur_r = recs[idx]
            non_consecutive = (
                cur_r.seq_num <= prev_r.seq_num
                or (cur_r.seq_num - prev_r.seq_num > 1)
                or any(prev_r.seq_num < s < cur_r.seq_num for s in sk_seqs)
            )
            speaker_changed = bool(
                (cur_r.speaker_id and prev_r.speaker_id and cur_r.speaker_id != prev_r.speaker_id)
                or (cur_r.s_sex and prev_r.s_sex and cur_r.s_sex != prev_r.s_sex)
            )
            if non_consecutive or speaker_changed:
                cur_r.segment_break = True

    return grouped


def slice_71379_windows(
    records: List[BroadcastUtterance71379],
    target_window_sec: float = 45.0,
    max_gap_sec: float = 8.0,
    inter_utt_pause_sec: float = 0.5,
) -> List[List[BroadcastUtterance71379]]:
    """71379 발화 시퀀스를 화자 변경, 순번 단절 및 목표 길이에 따라 스트리밍 윈도우로 분할합니다."""
    windows: List[List[BroadcastUtterance71379]] = []
    cur: List[BroadcastUtterance71379] = []
    cur_est_sec = 0.0

    for rec in records:
        est_dur = max(rec.t_ulength, rec.s_ulength, 3.0)
        if not cur:
            cur = [rec]
            cur_est_sec = est_dur
            continue

        prev = cur[-1]
        raw_gap: Optional[float] = None
        if prev.raw_end_sec is not None and rec.raw_start_sec is not None:
            raw_gap = rec.raw_start_sec - prev.raw_end_sec

        gap_exceeded = raw_gap is not None and (raw_gap < -0.5 or raw_gap > max_gap_sec)
        pause = (
            min(max(raw_gap, inter_utt_pause_sec), max_gap_sec)
            if (raw_gap is not None and raw_gap >= 0.0)
            else inter_utt_pause_sec
        )
        next_est_sec = cur_est_sec + pause + est_dur

        if rec.segment_break or gap_exceeded or next_est_sec > target_window_sec:
            windows.append(cur)
            cur = [rec]
            cur_est_sec = est_dur
        else:
            cur.append(rec)
            cur_est_sec = next_est_sec

    if cur:
        windows.append(cur)
    return windows


def _extract_prompt_from_clip_paths(
    clip_paths: Sequence[Path],
    prompt_sec: float = 10.0,
) -> torch.Tensor:
    """개별 발화 WAV 클립 경로들을 이어 붙여 목표 길이의 참조 화자 프롬프트를 생성합니다."""
    target_samples = int(prompt_sec * SAMPLE_RATE)
    chunks: List[torch.Tensor] = []
    collected = 0
    for p in clip_paths:
        wav = load_mono_wav_24k(p).detach().cpu().float().reshape(-1)
        if wav.numel() > 0:
            chunks.append(wav)
            collected += wav.numel()
            if collected >= target_samples:
                break
    if chunks:
        return torch.cat(chunks, dim=0)[:target_samples]
    return torch.zeros(target_samples, dtype=torch.float32)


def build_EDarch_71379_training_example(
    window_records: List[BroadcastUtterance71379],
    en_clip_paths: List[Path],
    ko_clip_paths: List[Optional[Path]],
    ko_sp_encode_fn: Callable[[str], List[int]],
    mimi_encode_fn: Callable[[torch.Tensor], torch.Tensor],
    wavlm_sim_bin_fn: Optional[Callable[[torch.Tensor, torch.Tensor], int]] = None,
    zero_shot_ko_tts_fn: Optional[
        Callable[[str, torch.Tensor], Union[torch.Tensor, Tuple[torch.Tensor, int]]]
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
            speaker_prompt_24k = _extract_prompt_from_clip_paths(en_clip_paths, prompt_sec=10.0)
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
                    ko_wav = _normalize_tts_output(
                        zero_shot_ko_tts_fn(rec.ko_text, speaker_prompt_24k)
                    )
                    is_real_ko = False

                ko_token_ids = list(ko_sp_encode_fn(rec.ko_text))
                if _clip_cache is not None:
                    _clip_cache[cache_key] = (en_wav, ko_wav, ko_token_ids, is_real_ko)

            _validate_ko_token_ids(ko_token_ids, text_card)
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

        audio_end_frames = (
            max(prev_en_end_sample, ko_cursor_sample) + SAMPLES_PER_FRAME - 1
        ) // SAMPLES_PER_FRAME
        eos_frame = max(last_token_frame + 1, audio_end_frames)
        total_frames = max(audio_end_frames + 4, eos_frame + 1)

        if (
            not _suppress_length_warning
            and max_context_frames > 0
            and total_frames > max_context_frames
        ):
            logger.warning(
                "71379 샘플 길이(%d 프레임)가 max_context_frames(%d)를 초과합니다 (%s).",
                total_frames,
                max_context_frames,
                window_records[0].utt_id,
            )

        padded_samples = (total_frames + 2) * SAMPLES_PER_FRAME
        en_timeline = torch.zeros(padded_samples, dtype=torch.float32)
        ko_timeline = torch.zeros(padded_samples, dtype=torch.float32)
        ko_text_stream = torch.full((1, total_frames), TEXT_PAD_ID, dtype=torch.long)

        for start_sample, en_wav in en_placements:
            en_timeline[start_sample : start_sample + en_wav.numel()] = en_wav

        for start_sample, ko_wav, token_positions in ko_placements:
            ko_timeline[start_sample : start_sample + ko_wav.numel()] = ko_wav
            for f_pos, tid in token_positions:
                ko_text_stream[0, f_pos] = tid

        ko_text_stream[0, eos_frame] = TEXT_EOS_ID

        ko_codes = _encode_and_align_mimi(ko_timeline, mimi_encode_fn, total_frames, expected_q)
        en_codes = _encode_and_align_mimi(en_timeline, mimi_encode_fn, total_frames, expected_q)
        if ko_codes.shape[0] != en_codes.shape[0]:
            raise ValueError(
                f"타깃/소스 코드북 수 불일치: ko={ko_codes.shape[0]}, en={en_codes.shape[0]}"
            )

        active_ko_clips = [w for _, w, _ in ko_placements if w.numel() > 0]
        ko_speech_only = (
            torch.cat(active_ko_clips, dim=0)
            if active_ko_clips
            else ko_timeline[: total_frames * SAMPLES_PER_FRAME]
        )

        if wavlm_sim_bin_fn is not None:
            raw_sim = wavlm_sim_bin_fn(speaker_prompt_24k, ko_speech_only)
        elif used_real_ko:
            if fallback_real_sim_bin is not None:
                raw_sim = fallback_real_sim_bin
            else:
                first_rec = window_records[0]
                same_sex = bool(
                    first_rec.s_sex and first_rec.t_sex and first_rec.s_sex == first_rec.t_sex
                )
                raw_sim = min(1 if same_sex else 0, num_speaker_sim_bins - 1)
        else:
            raw_sim = (
                fallback_tts_sim_bin
                if fallback_tts_sim_bin is not None
                else num_speaker_sim_bins - 1
            )

        sim_bin = _validate_sim_bin(raw_sim, num_speaker_sim_bins)
        if _out_metadata is not None:
            _out_metadata.clear()
            _out_metadata.update(
                {
                    "src_start_frames": src_start_frames,
                    "src_end_frames": src_end_frames,
                    "ref_token_groups": ref_token_groups,
                    "align_mode": align_mode,
                }
            )
        codes = torch.cat([ko_text_stream, ko_codes, en_codes], dim=0)
        return codes, sim_bin


def preprocess_71693_dataset(
    json_dir: Path,
    wav_dir: Path,
    output_dir: Path,
    zero_shot_ko_tts_fn: Callable[
        [str, torch.Tensor], Union[torch.Tensor, Tuple[torch.Tensor, int]]
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

    wav_index, _ = _index_wav_files(wav_dir)
    existing_stems, split_ancestors = (
        _index_existing_window_trees(output_dir)
        if skip_existing
        else (set(), set())
    )
    saved_count = 0

    for file_name, spans in sorted(grouped.items()):
        windows = slice_60s_windows(
            spans, target_window_sec=target_window_sec, max_gap_sec=max_gap_sec
        )
        stem = Path(file_name).stem
        if skip_existing and windows:
            lecture_leaf_counts = [
                _count_completed_window_tree(
                    f"{file_prefix}{stem}_win{w_idx:04d}",
                    len(w_spans),
                    existing_stems,
                )
                for w_idx, w_spans in enumerate(windows)
            ]
            if all(cnt is not None for cnt in lecture_leaf_counts):
                saved_count += sum(cnt for cnt in lecture_leaf_counts if cnt is not None)
                continue

        wav_path = _resolve_wav_path(wav_dir, file_name, wav_index)
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
                done_leaves = _count_completed_window_tree(
                    cur_stem, len(win_spans), existing_stems
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
        Callable[[str, torch.Tensor], Union[torch.Tensor, Tuple[torch.Tensor, int]]]
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
    en_wav_index, _ = _index_wav_files(
        en_search_dir,
        wav_dir,
        prefer_keywords=("원어민", "발화", "영어", "영국", "en", "eng", "uk", "target"),
    )
    ko_wav_index, _ = _index_wav_files(
        ko_search_dir,
        wav_dir,
        prefer_keywords=("원천", "한국어", "한글", "ko", "kor", "source"),
    )

    speaker_en_paths: dict[str, List[Path]] = defaultdict(list)
    resolved_groups: dict[str, List[Tuple[BroadcastUtterance71379, Path, Optional[Path]]]] = {}

    for group_key, recs in sorted(grouped.items()):
        valid_triples: List[Tuple[BroadcastUtterance71379, Path, Optional[Path]]] = []
        for idx, rec in enumerate(recs):
            en_path = _resolve_wav_path(en_search_dir, rec.en_wav_name, en_wav_index)
            if en_path is None:
                logger.warning("71379 원어민 영어 WAV 누락으로 건너뜁니다: %s", rec.en_wav_name)
                if idx + 1 < len(recs):
                    recs[idx + 1].segment_break = True
                continue

            ko_path = (
                _resolve_wav_path(ko_search_dir, rec.ko_wav_name, ko_wav_index)
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
            if rec.speaker_id and len(speaker_en_paths[rec.speaker_id]) < 8:
                speaker_en_paths[rec.speaker_id].append(en_path)

        if valid_triples:
            resolved_groups[group_key] = valid_triples

    if not resolved_groups:
        raise ValueError("71379 JSON에 매칭되는 유효한 WAV 파일을 하나도 찾지 못했습니다.")

    speaker_prompt_cache: dict[str, torch.Tensor] = {}
    existing_stems, split_ancestors = (
        _index_existing_window_trees(output_dir)
        if skip_existing
        else (set(), set())
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
        )
        if skip_existing and windows:
            group_leaf_counts = [
                _count_completed_window_tree(
                    f"{file_prefix}{group_key}_win{w_idx:04d}",
                    len(w_recs),
                    existing_stems,
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
                done_leaves = _count_completed_window_tree(
                    cur_stem, len(win_recs), existing_stems
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

            spk_id = win_recs[0].speaker_id
            prompt_24k: Optional[torch.Tensor] = None
            if spk_id:
                prompt_24k = speaker_prompt_cache.get(spk_id)
                if prompt_24k is None and spk_id in speaker_en_paths:
                    prompt_24k = _extract_prompt_from_clip_paths(
                        speaker_en_paths[spk_id], prompt_sec=10.0
                    )
                    speaker_prompt_cache[spk_id] = prompt_24k
            if prompt_24k is None:
                prompt_24k = _extract_prompt_from_clip_paths(win_en_paths, prompt_sec=10.0)

            can_split = max_context_frames > 0 and len(win_recs) > 1
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
        for raw_rec in _extract_json_records(payload)[:5]:
            _, item = _normalize_record_keys(raw_rec)
            if _is_71693_record(item):
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
    """토크나이저·정렬기·코덱 훅을 준비하고 AI-Hub 71693 및 71379 데이터셋 전처리를 통합 실행합니다."""
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
                logger.info(
                    "SentencePiece 모델(%s) 자동 학습을 시작합니다...",
                    tok_path,
                )
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
        if tts_fn is None:
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

    needs_tts = (task_71693 is not None) or (
        task_71379 is not None and args.ko_audio_mode_71379 == "tts"
    )
    missing = []
    if needs_tts and tts_fn is None:
        missing.append("zero_shot_ko_tts_fn")
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
    sim_fn_for_71693 = (
        sim_fn if sim_fn is not None else (lambda prompt, gen: cfg.num_speaker_sim_bins - 1)
    )

    fallback_real_sim_bin = args.real_pair_sim_bin
    if not has_user_sim_fn:
        if task_71693 is not None and task_71379 is not None:
            logger.info(
                "WavLM 미지정: 71693(음성복제 TTS) 샘플에는 sim_bin=%d(very_good), "
                "71379(실제 이종 화자 원천 WAV) 샘플에는 성별 일치 여부에 따라 sim_bin=0..1을 부여합니다.",
                cfg.num_speaker_sim_bins - 1,
            )
        elif task_71379 is not None and task_71693 is None and fallback_real_sim_bin is None:
            fallback_real_sim_bin = cfg.num_speaker_sim_bins - 1
            logger.warning(
                "71379 단독 전처리 시 WavLM이 지정되지 않아 추론(run.py)의 very_good_id(%d) 조건과 "
                "정합되도록 sim_bin=%d로 설정합니다.",
                fallback_real_sim_bin,
                fallback_real_sim_bin,
            )
        else:
            logger.warning(
                "wavlm_sim_bin_fn이 지정되지 않아 모든 샘플의 sim_bin이 상수(%d = very_good)로 고정됩니다.",
                cfg.num_speaker_sim_bins - 1,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    both_active = task_71693 is not None and task_71379 is not None
    saved_71693 = 0
    saved_71379 = 0

    if task_71693 is not None:
        assert tts_fn is not None and sp_fn is not None and mimi_fn is not None
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
