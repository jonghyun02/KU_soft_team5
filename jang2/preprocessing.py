from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import operator
import os
import re
import tempfile
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:
    import torch
    import torchaudio
except ImportError:
    torch = None
    torchaudio = None

if TYPE_CHECKING:
    import torch as torch_t

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24_000
FRAME_RATE = 12.5
SAMPLES_PER_FRAME = int(SAMPLE_RATE / FRAME_RATE)
TEXT_UNK_ID = 0
TEXT_BOS_ID_SPM = 1
TEXT_EOS_ID = 2
TEXT_PAD_ID = 3
ZERO_TOKEN_ID = -1
DEFAULT_CONTEXT_FRAMES = 1000

_EMPTY_TEXT_MARKERS = {"", "-", "n/a", "na", "null", "none", "x"}
_LEADING_TRANSCRIPT_NOISE_RE = re.compile(r"^(?:##+|\*\*+|//+)\s*")
_RESAMPLER_CACHE: dict[tuple[int, int, str, str], object] = {}


def _resample_wav(wav: "torch_t.Tensor", orig_sr: int, target_sr: int) -> "torch_t.Tensor":
    if orig_sr == target_sr:
        return wav
    key = (orig_sr, target_sr, str(wav.dtype), str(wav.device))
    resampler = _RESAMPLER_CACHE.get(key)
    if resampler is None:
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sr, new_freq=target_sr, dtype=wav.dtype
        ).to(wav.device)
        _RESAMPLER_CACHE[key] = resampler
    return resampler(wav)


def parse_hhmmss(ts: Union[str, int, float]) -> float:
    """'HH:MM:SS.mmm', 'HH:MM:SS,mmm', 'MM:SS.mmm', 'SS.mmm' 또는 숫자(초)를 float 초로 변환합니다."""
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
    if val is None or isinstance(val, bool):
        return ""
    text = str(val).strip()
    text = _LEADING_TRANSCRIPT_NOISE_RE.sub("", text).strip()
    if text.lower() in _EMPTY_TEXT_MARKERS:
        return ""
    return text


def _normalize_record_keys(raw: Mapping_or_Dict) -> Tuple[dict[str, Any], dict[str, Any]]:
    norm: dict[str, Any] = {}
    lower_norm: dict[str, Any] = {}
    for k, v in raw.items():
        ck = str(k).lstrip("\ufeff").strip()
        norm[ck] = v
        lower_norm[ck.lower()] = v
    return norm, lower_norm


Mapping_or_Dict = Dict[Any, Any]


def _extract_json_records(payload: Any) -> List[dict[str, Any]]:
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
    """디렉터리들에서 WAV 파일을 대소문자 무관하게 인덱싱하며, 동명 파일 충돌 시 경로 키워드 점수로 우선순위를 결정합니다."""
    by_lower_name: dict[str, Path] = {}
    best_score: dict[str, int] = {}
    all_paths: List[Path] = []
    seen_resolved: set[Path] = set()

    def _path_score(p: Path) -> int:
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
    if any(
        tid < 0
        or tid in (TEXT_EOS_ID, TEXT_PAD_ID)
        or (text_card is not None and tid >= text_card)
        for tid in ko_token_ids
    ):
        upper_msg = f", < {text_card}" if text_card is not None else ""
        raise ValueError(
            f"ko_sp_encode_fn 출력에 허용 범위를 벗어난 토큰(0 이상{upper_msg}) 또는 "
            f"예약 특수 토큰(EOS={TEXT_EOS_ID}, PAD={TEXT_PAD_ID})이 포함되어 있습니다. "
            "토크나이저에서 add_eos=False, --eos_id=2, --pad_id=3 및 어휘 크기를 확인하세요."
        )


def _validate_sim_bin(raw_sim_bin: Any, num_speaker_sim_bins: int) -> int:
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
    timeline: "torch_t.Tensor",
    mimi_encode_fn: Callable[["torch_t.Tensor"], "torch_t.Tensor"],
    total_frames: int,
    expected_q: Optional[int],
) -> "torch_t.Tensor":
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
    tts_out: Union["torch_t.Tensor", Tuple["torch_t.Tensor", int]],
) -> "torch_t.Tensor":
    if isinstance(tts_out, tuple):
        wav_out, tts_sr = tts_out
        wav_out = wav_out.detach().cpu().float().reshape(-1)
        if tts_sr != SAMPLE_RATE:
            wav_out = _resample_wav(wav_out, int(tts_sr), SAMPLE_RATE)
        return wav_out
    return tts_out.detach().cpu().float().reshape(-1)


# ==============================================================================
# 0. Inner Monologue 정밀 정렬 엔진 (Acoustic Energy / Whisper / CTC Forced Align)
# ==============================================================================


@dataclass
class WordTimestamp:
    text: str
    start_sec: float
    end_sec: float


AlignmentTimestampItem = Union[
    float,
    int,
    Tuple[str, float, float],
    Tuple[float, float],
    WordTimestamp,
    Dict[str, Any],
]
KoAlignFn = Callable[[str, "torch_t.Tensor", Sequence[int]], Sequence[AlignmentTimestampItem]]


def _compute_subword_weights(
    ko_token_ids: Sequence[int],
    token_pieces: Optional[Sequence[str]] = None,
) -> List[float]:
    if token_pieces is not None and len(token_pieces) == len(ko_token_ids):
        weights: List[float] = []
        for piece in token_pieces:
            cleaned = piece.replace("\u2581", "").strip()
            weights.append(float(max(1, len(cleaned))))
        return weights
    return [1.0] * len(ko_token_ids)


def align_tokens_by_acoustic_energy(
    ko_wav_24k: "torch_t.Tensor",
    ko_token_ids: Sequence[int],
    token_pieces: Optional[Sequence[str]] = None,
    silence_db: float = -35.0,
    energy_gamma: float = 0.6,
) -> List[int]:
    """
    외부 ASR 없이 오디오 파형의 12.5Hz 프레임별 RMS 에너지 포락선과 VAD(음성 활성 구간)를 분석하여
    앞뒤 무음 및 문장 내 휴지기(쉼표 구간)를 회피하고 실제 유성음 발화 프레임에 토큰을 정렬합니다.
    """
    num_tokens = len(ko_token_ids)
    if num_tokens == 0:
        return []

    wav = ko_wav_24k.detach().cpu().float().reshape(-1)
    dur_frames = max(1, (wav.numel() + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME)
    if dur_frames == 1:
        return [0] * num_tokens

    padded_len = dur_frames * SAMPLES_PER_FRAME
    if wav.numel() < padded_len:
        wav_padded = torch.zeros(padded_len, dtype=torch.float32)
        wav_padded[: wav.numel()] = wav
    else:
        wav_padded = wav[:padded_len]

    frames = wav_padded.view(dur_frames, SAMPLES_PER_FRAME)
    rms = torch.sqrt(frames.pow(2).mean(dim=-1) + 1e-12)
    peak_rms = float(rms.max().item())

    if peak_rms <= 1e-6:
        return [(idx * dur_frames) // num_tokens for idx in range(num_tokens)]

    vad_thresh = peak_rms * (10.0 ** (silence_db / 20.0))
    voiced_indices = torch.nonzero(rms >= vad_thresh, as_tuple=False).reshape(-1)
    if voiced_indices.numel() > 0:
        onset_frame = int(voiced_indices[0].item())
        offset_frame = int(voiced_indices[-1].item())
    else:
        onset_frame = 0
        offset_frame = dur_frames - 1

    active_rms = rms[onset_frame : offset_frame + 1] / peak_rms
    # 무음 골짜기(휴지기)의 누적 가중치를 낮춰 토큰이 실제 발음 봉우리에 배치되도록 유도
    activity = torch.where(
        active_rms >= (10.0 ** (silence_db / 20.0)),
        active_rms.clamp_min(1e-4).pow(energy_gamma),
        torch.full_like(active_rms, 0.015),
    )
    total_act = float(activity.sum().item())
    if total_act <= 1e-8:
        active_len = offset_frame - onset_frame + 1
        return [onset_frame + (idx * active_len) // num_tokens for idx in range(num_tokens)]

    cum_end = torch.cumsum(activity, dim=0) / total_act
    cum_start = torch.cat([torch.zeros(1, dtype=cum_end.dtype), cum_end[:-1]], dim=0)

    tok_weights = _compute_subword_weights(ko_token_ids, token_pieces)
    total_tok_w = sum(tok_weights)
    rel_frames: List[int] = []
    acc_w = 0.0
    for w in tok_weights:
        target_u = acc_w / max(1e-8, total_tok_w)
        # 누적 발화 에너지가 target_u를 넘어서는 첫 프레임 선택
        cand = int(torch.searchsorted(cum_end, torch.tensor(target_u, dtype=cum_end.dtype), right=True).item())
        cand = max(0, min(cand, int(activity.numel()) - 1))
        # 아주 짧은 시작점에서 cum_start 기준으로 보정
        if cand > 0 and float(cum_start[cand].item()) > target_u + 1e-6:
            cand -= 1
        rel_frames.append(onset_frame + cand)
        acc_w += w
    return rel_frames


def _parse_timestamp_items(
    raw_items: Sequence[AlignmentTimestampItem],
    max_sec: float,
) -> Tuple[Optional[List[float]], List[WordTimestamp]]:
    if not raw_items:
        return None, []

    if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in raw_items):
        onsets = [max(0.0, min(float(x), max_sec)) for x in raw_items]  # type: ignore[arg-type]
        return onsets, []

    spans: List[WordTimestamp] = []
    for item in raw_items:
        w_text = ""
        s_sec: Optional[float] = None
        e_sec: Optional[float] = None
        if isinstance(item, WordTimestamp):
            w_text, s_sec, e_sec = item.text, item.start_sec, item.end_sec
        elif isinstance(item, dict):
            w_text = str(item.get("text") or item.get("word") or "")
            ts = item.get("timestamp") or item.get("timestamps")
            if isinstance(ts, (tuple, list)) and len(ts) >= 2:
                s_sec = float(ts[0]) if ts[0] is not None else None
                e_sec = float(ts[1]) if ts[1] is not None else None
            else:
                if item.get("start") is not None:
                    s_sec = float(item["start"])
                if item.get("end") is not None:
                    e_sec = float(item["end"])
        elif isinstance(item, (tuple, list)):
            if len(item) >= 3:
                w_text = str(item[0])
                s_sec = float(item[1]) if item[1] is not None else None
                e_sec = float(item[2]) if item[2] is not None else None
            elif len(item) == 2:
                s_sec = float(item[0]) if item[0] is not None else None
                e_sec = float(item[1]) if item[1] is not None else None

        if s_sec is None and e_sec is None:
            continue
        if s_sec is None:
            s_sec = max(0.0, float(e_sec) - 0.08)  # type: ignore[arg-type]
        if e_sec is None or e_sec < s_sec:
            e_sec = min(max_sec, s_sec + 0.08)
        s_clamped = max(0.0, min(s_sec, max_sec))
        e_clamped = max(s_clamped, min(e_sec, max_sec))
        spans.append(WordTimestamp(text=w_text.strip(), start_sec=s_clamped, end_sec=e_clamped))

    return None, spans


def align_tokens_with_timestamps(
    raw_timestamps: Sequence[AlignmentTimestampItem],
    ko_wav_24k: "torch_t.Tensor",
    ko_token_ids: Sequence[int],
    token_pieces: Optional[Sequence[str]] = None,
) -> List[int]:
    """
    Whisper 단어 타임스탬프 또는 CTC 강제 정렬(Forced Alignment) 구간을 받아
    각 한국어 SentencePiece 서브워드 토큰의 발화 시작 프레임(0-based)을 정밀 매핑합니다.
    """
    num_tokens = len(ko_token_ids)
    if num_tokens == 0:
        return []

    dur_samples = int(ko_wav_24k.numel())
    dur_frames = max(1, (dur_samples + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME)
    max_sec = max(dur_samples / float(SAMPLE_RATE), dur_frames / FRAME_RATE)

    per_tok_onsets, word_spans = _parse_timestamp_items(raw_timestamps, max_sec=max_sec)
    if per_tok_onsets is not None and len(per_tok_onsets) == num_tokens:
        return [
            max(0, min(int(sec * FRAME_RATE), dur_frames - 1))
            for sec in per_tok_onsets
        ]

    if not word_spans:
        return align_tokens_by_acoustic_energy(ko_wav_24k, ko_token_ids, token_pieces=token_pieces)

    tok_weights = _compute_subword_weights(ko_token_ids, token_pieces)
    total_tok_w = max(1e-8, sum(tok_weights))

    span_char_weights = [float(max(1, len("".join(sp.text.split())))) for sp in word_spans]
    total_span_w = max(1e-8, sum(span_char_weights))

    span_cum_end: List[float] = []
    acc_sw = 0.0
    for sw in span_char_weights:
        acc_sw += sw
        span_cum_end.append(acc_sw / total_span_w)

    rel_frames: List[int] = []
    acc_tw = 0.0
    span_idx = 0
    for tw in tok_weights:
        u = acc_tw / total_tok_w
        while span_idx + 1 < len(word_spans) and u >= span_cum_end[span_idx] - 1e-7:
            span_idx += 1
        prev_u = 0.0 if span_idx == 0 else span_cum_end[span_idx - 1]
        span_u_width = max(1e-6, span_cum_end[span_idx] - prev_u)
        local_ratio = max(0.0, min(1.0, (u - prev_u) / span_u_width))
        sp = word_spans[span_idx]
        onset_sec = sp.start_sec + local_ratio * max(0.0, sp.end_sec - sp.start_sec)
        f_idx = max(0, min(int(onset_sec * FRAME_RATE), dur_frames - 1))
        rel_frames.append(f_idx)
        acc_tw += tw

    return rel_frames


def compute_aligned_token_placement(
    ko_text: str,
    ko_wav_24k: "torch_t.Tensor",
    ko_token_ids: Sequence[int],
    en_end_frame: int,
    prev_audio_end_frame: int,
    last_token_frame: int,
    align_mode: str = "auto",
    ko_align_fn: Optional[KoAlignFn] = None,
    token_pieces: Optional[Sequence[str]] = None,
    monologue_lead_frames: int = 1,
) -> Tuple[int, List[Tuple[int, int]], int]:
    """
    타깃 한국어 음성과 텍스트 토큰(Inner Monologue)의 인과적·음향적 정렬 위치를 계산합니다.

    Returns:
        (audio_start_sample, [(global_frame_pos, token_id), ...], updated_last_token_frame)
    """
    if align_mode not in ("auto", "energy", "whisper", "ctc", "timestamps", "uniform"):
        raise ValueError(
            f"지원하지 않는 align_mode입니다: {align_mode} "
            "(지원: 'auto', 'energy', 'whisper', 'ctc', 'timestamps', 'uniform')"
        )

    audio_floor = max(en_end_frame, prev_audio_end_frame)
    text_start_frame = max(audio_floor, last_token_frame + 1)
    dur_frames = max(1, (int(ko_wav_24k.numel()) + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME)
    num_tokens = len(ko_token_ids)

    if num_tokens == 0:
        return audio_floor * SAMPLES_PER_FRAME, [], last_token_frame

    if align_mode == "uniform":
        lead_frames = max(0, num_tokens - dur_frames)
        audio_start_frame = text_start_frame + lead_frames
        token_positions: List[Tuple[int, int]] = []
        cur_last = last_token_frame
        for idx, tid in enumerate(ko_token_ids):
            if num_tokens <= dur_frames:
                ideal_pos = audio_start_frame + (idx * dur_frames) // max(1, num_tokens)
                f_pos = max(ideal_pos, cur_last + 1)
            else:
                f_pos = text_start_frame + idx
            token_positions.append((f_pos, int(tid)))
            cur_last = f_pos
        return audio_start_frame * SAMPLES_PER_FRAME, token_positions, cur_last

    use_ts = ko_align_fn is not None and align_mode in ("auto", "whisper", "ctc", "timestamps")
    rel_acoustic: Optional[List[int]] = None
    if use_ts:
        assert ko_align_fn is not None
        try:
            raw_ts = ko_align_fn(ko_text, ko_wav_24k, ko_token_ids)
            rel_acoustic = align_tokens_with_timestamps(
                raw_ts, ko_wav_24k, ko_token_ids, token_pieces=token_pieces
            )
        except Exception as exc:
            logger.warning("외부 타임스탬프 정렬 실패로 에너지 정렬로 폴백합니다: %s", exc)
            rel_acoustic = None

    if rel_acoustic is None:
        rel_acoustic = align_tokens_by_acoustic_energy(
            ko_wav_24k, ko_token_ids, token_pieces=token_pieces
        )

    lead = max(0, int(monologue_lead_frames))
    rel_tok: List[int] = []
    for idx, r_f in enumerate(rel_acoustic):
        desired = int(r_f) - lead
        if idx == 0:
            rel_tok.append(desired)
        else:
            rel_tok.append(max(desired, rel_tok[-1] + 1))

    # 토큰 밀도가 높아 오디오 끝을 초과한 경우 좌측 여유 범위 내에서 당겨 정렬
    max_allowed_rel = max(dur_frames - 1, num_tokens - 1 - lead)
    overflow = rel_tok[-1] - max_allowed_rel
    if overflow > 0:
        rel_tok[-1] -= overflow
        for idx in range(num_tokens - 2, -1, -1):
            if rel_tok[idx] >= rel_tok[idx + 1]:
                rel_tok[idx] = rel_tok[idx + 1] - 1

    # 토큰 수가 오디오 프레임보다 많은 경우 선행 lead_frames 보장
    if num_tokens > dur_frames and rel_tok[0] > -(num_tokens - dur_frames):
        shift = rel_tok[0] + (num_tokens - dur_frames)
        rel_tok = [rt - shift for rt in rel_tok]

    audio_start_frame = max(audio_floor, text_start_frame - rel_tok[0])
    token_positions = []
    cur_last = last_token_frame
    for rt, tid in zip(rel_tok, ko_token_ids):
        f_pos = max(audio_start_frame + rt, cur_last + 1)
        token_positions.append((f_pos, int(tid)))
        cur_last = f_pos

    return audio_start_frame * SAMPLES_PER_FRAME, token_positions, cur_last


# ==============================================================================
# 1. AI-Hub 71693: 국제 학술대회용 전문분야 한영/영한 통번역 데이터
# ==============================================================================


@dataclass
class UtteranceSpan:
    sn: str
    file_name: str
    start_sec: float
    end_sec: float
    en_text: str
    ko_mtpe: str
    segment_break: bool = False
    direction: str = "en2ko"


def _is_71693_record(lower_item: dict[str, Any]) -> bool:
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
    lecture_wav_24k: "torch_t.Tensor",
    spans: List[UtteranceSpan],
    prompt_sec: float = 10.0,
) -> "torch_t.Tensor":
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


def load_lecture_wav_24k(lecture_wav_path: Path) -> "torch_t.Tensor":
    wav, sr = torchaudio.load(str(lecture_wav_path))
    wav = wav.mean(dim=0).float()
    return _resample_wav(wav, sr, SAMPLE_RATE)


def _assemble_virtual_lecture_from_sn_clips(
    spans: List[UtteranceSpan],
    wav_dir: Path,
    wav_index: dict[str, Path],
) -> Optional["torch_t.Tensor"]:
    """전체 강의 WAV 대신 발화별 {sn}.wav 분할본만 제공된 경우 타임라인 WAV를 재조립합니다."""
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
        clip_wav = load_lecture_wav_24k(clip_path)
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
        [str, "torch_t.Tensor"], Union["torch_t.Tensor", Tuple["torch_t.Tensor", int]]
    ],
    ko_sp_encode_fn: Callable[[str], List[int]],
    mimi_encode_fn: Callable[["torch_t.Tensor"], "torch_t.Tensor"],
    wavlm_sim_bin_fn: Callable[["torch_t.Tensor", "torch_t.Tensor"], int],
    preloaded_wav_24k: Optional["torch_t.Tensor"] = None,
    speaker_prompt_24k: Optional["torch_t.Tensor"] = None,
    expected_q: Optional[int] = None,
    max_context_frames: int = DEFAULT_CONTEXT_FRAMES,
    text_card: Optional[int] = None,
    num_speaker_sim_bins: int = 5,
    align_mode: str = "auto",
    ko_align_fn: Optional[KoAlignFn] = None,
    ko_sp_pieces_fn: Optional[Callable[[str], List[str]]] = None,
    monologue_lead_frames: int = 1,
    _out_metadata: Optional[dict[str, Any]] = None,
    _tts_tok_cache: Optional[dict[str, Tuple["torch_t.Tensor", List[int]]]] = None,
    _suppress_length_warning: bool = False,
) -> Tuple["torch_t.Tensor", int]:
    if not window_spans:
        raise ValueError("window_spans는 비어 있을 수 없습니다.")

    with torch.no_grad():
        wav = (
            preloaded_wav_24k
            if preloaded_wav_24k is not None
            else load_lecture_wav_24k(lecture_wav_path)
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

        ko_placements: List[Tuple[int, "torch_t.Tensor", List[Tuple[int, int]]]] = []
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
            tok_pieces = list(ko_sp_pieces_fn(sp.ko_mtpe)) if ko_sp_pieces_fn is not None else None

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
    codes: "torch_t.Tensor",
    sim_bin: int,
    metadata: Optional[Dict[str, Any]] = None,
    max_retries: int = 5,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    payload: Dict[str, Any] = {"codes": codes.detach().cpu(), "sim_bin": int(sim_bin)}
    if metadata:
        for k, v in metadata.items():
            if k not in payload:
                payload[k] = v
    torch.save(payload, tmp_path)
    for attempt in range(max_retries):
        try:
            os.replace(tmp_path, output_path)
            return
        except PermissionError:
            if attempt == max_retries - 1:
                raise
            time.sleep(0.1 * (2**attempt))


# ==============================================================================
# 2. AI-Hub 71379: 방송콘텐츠 한국어-영어 통번역 음성 데이터 (1-009-031)
# ==============================================================================


@dataclass
class BroadcastUtterance71379:
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
    ko_text: str
    raw_start_sec: Optional[float]
    raw_end_sec: Optional[float]
    deidentified: bool


def _parse_71379_scene_and_seq(
    ko_wav_name: str,
    en_wav_name: str,
    lower_item: dict[str, Any],
) -> Tuple[str, int]:
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


def _classify_71379_t_code(t_code_raw: str, en_wav_name: str, json_name: str) -> Optional[str]:
    """영어(미국/영국) 레코드만 판별하여 'us' 또는 'uk'를 반환하고, 스페인어/러시아어 등은 None을 반환합니다."""
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
    """
    AI-Hub 71379(방송콘텐츠 한국어-영어 통번역 음성 데이터) 라벨링 JSON을 파싱하고
    (씬, 영어 억양 변종) 단위로 정렬·그룹핑합니다.
    음성전사 JSON이 함께 존재하면 원시 타임스탬프 및 비식별화 정보를 자동 결합합니다.
    """
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
) -> "torch_t.Tensor":
    target_samples = int(prompt_sec * SAMPLE_RATE)
    chunks: List["torch_t.Tensor"] = []
    collected = 0
    for p in clip_paths:
        wav = load_lecture_wav_24k(p).detach().cpu().float().reshape(-1)
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
    mimi_encode_fn: Callable[["torch_t.Tensor"], "torch_t.Tensor"],
    wavlm_sim_bin_fn: Optional[Callable[["torch_t.Tensor", "torch_t.Tensor"], int]] = None,
    zero_shot_ko_tts_fn: Optional[
        Callable[[str, "torch_t.Tensor"], Union["torch_t.Tensor", Tuple["torch_t.Tensor", int]]]
    ] = None,
    speaker_prompt_24k: Optional["torch_t.Tensor"] = None,
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
    _clip_cache: Optional[dict[str, Tuple["torch_t.Tensor", "torch_t.Tensor", List[int], bool]]] = None,
    _suppress_length_warning: bool = False,
) -> Tuple["torch_t.Tensor", int]:
    """
    AI-Hub 71379 발화 윈도우(원어민 영어 발화 WAV + 한국어 방송 원천 WAV 또는 TTS + 한국어 텍스트)를
    EDarch 멀티스트림 코드북 텐서 [1 + 2Q, T]로 인과 정렬·인코딩합니다.
    """
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

        en_placements: List[Tuple[int, "torch_t.Tensor"]] = []
        ko_placements: List[Tuple[int, "torch_t.Tensor", List[Tuple[int, int]]]] = []
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
                en_wav = load_lecture_wav_24k(en_path).detach().cpu().float().reshape(-1)
                if en_wav.numel() == 0:
                    raise ValueError(f"비어 있는 영어 WAV 파일입니다: {en_path}")

                use_real = ko_audio_mode in ("auto", "real") and ko_path is not None
                if ko_audio_mode == "real" and ko_path is None:
                    raise FileNotFoundError(
                        f"ko_audio_mode='real'이지만 한국어 원천 WAV를 찾을 수 없습니다: {rec.ko_wav_name}"
                    )
                if use_real:
                    assert ko_path is not None
                    ko_wav = load_lecture_wav_24k(ko_path).detach().cpu().float().reshape(-1)
                    if ko_wav.numel() == 0:
                        raise ValueError(f"비어 있는 한국어 WAV 파일입니다: {ko_path}")
                    is_real_ko = True
                else:
                    if zero_shot_ko_tts_fn is None:
                        raise ValueError(
                            f"한국어 원천 WAV({rec.ko_wav_name})가 없거나 ko_audio_mode='tts'인데 "
                            "zero_shot_ko_tts_fn이 지정되지 않았습니다."
                        )
                    ko_wav = _normalize_tts_output(zero_shot_ko_tts_fn(rec.ko_text, speaker_prompt_24k))
                    is_real_ko = False

                ko_token_ids = list(ko_sp_encode_fn(rec.ko_text))
                if _clip_cache is not None:
                    _clip_cache[cache_key] = (en_wav, ko_wav, ko_token_ids, is_real_ko)

            _validate_ko_token_ids(ko_token_ids, text_card)
            tok_pieces = list(ko_sp_pieces_fn(rec.ko_text)) if ko_sp_pieces_fn is not None else None
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


# ==============================================================================
# 3. 데이터셋별 배치 전처리 엔트리 함수 (71693 & 71379)
# ==============================================================================


def preprocess_71693_dataset(
    json_dir: Path,
    wav_dir: Path,
    output_dir: Path,
    zero_shot_ko_tts_fn: Callable[
        [str, "torch_t.Tensor"], Union["torch_t.Tensor", Tuple["torch_t.Tensor", int]]
    ],
    ko_sp_encode_fn: Callable[[str], List[int]],
    mimi_encode_fn: Callable[["torch_t.Tensor"], "torch_t.Tensor"],
    wavlm_sim_bin_fn: Callable[["torch_t.Tensor", "torch_t.Tensor"], int],
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
    saved_count = 0

    for file_name, spans in sorted(grouped.items()):
        wav_path = _resolve_wav_path(wav_dir, file_name, wav_index)
        wav_24k: Optional["torch_t.Tensor"] = None
        if wav_path is not None:
            wav_24k = load_lecture_wav_24k(wav_path)
        else:
            wav_24k = _assemble_virtual_lecture_from_sn_clips(spans, wav_dir, wav_index)
            if wav_24k is not None:
                wav_path = wav_dir / file_name

        if wav_path is None or wav_24k is None:
            logger.warning("71693 원본 WAV 파일을 찾지 못해 건너뜁니다: %s", file_name)
            continue

        speaker_prompt = extract_10s_speaker_prompt(wav_24k, spans, prompt_sec=10.0)
        windows = slice_60s_windows(
            spans, target_window_sec=target_window_sec, max_gap_sec=max_gap_sec
        )
        stem = Path(file_name).stem
        tts_cache: dict[str, Tuple["torch_t.Tensor", List[int]]] = {}
        work_queue = deque(windows)
        win_idx = 0

        while work_queue:
            win_spans = work_queue.popleft()
            out_pt = output_dir / f"{file_prefix}{stem}_win{win_idx:04d}.pt"
            if skip_existing and out_pt.exists() and len(win_spans) == 1:
                win_idx += 1
                saved_count += 1
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
                work_queue.appendleft(win_spans[mid:])
                work_queue.appendleft(win_spans[:mid])
                continue

            if not (skip_existing and out_pt.exists()):
                save_training_example_pt(out_pt, codes, sim_bin, metadata=meta_dict)
            win_idx += 1
            saved_count += 1

    return saved_count


def preprocess_71379_dataset(
    json_dir: Path,
    output_dir: Path,
    ko_sp_encode_fn: Callable[[str], List[int]],
    mimi_encode_fn: Callable[["torch_t.Tensor"], "torch_t.Tensor"],
    wav_dir: Optional[Path] = None,
    en_wav_dir: Optional[Path] = None,
    ko_wav_dir: Optional[Path] = None,
    wavlm_sim_bin_fn: Optional[Callable[["torch_t.Tensor", "torch_t.Tensor"], int]] = None,
    zero_shot_ko_tts_fn: Optional[
        Callable[[str, "torch_t.Tensor"], Union["torch_t.Tensor", Tuple["torch_t.Tensor", int]]]
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
    """
    AI-Hub 71379 데이터셋(방송콘텐츠 한국어-영어 통번역 음성 데이터)을 전처리하여
    train.py(PreprocessedPtDataset)에서 바로 학습 가능한 .pt 파일들을 생성합니다.
    """
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

    speaker_prompt_cache: dict[str, "torch_t.Tensor"] = {}
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

        clip_cache: dict[str, Tuple["torch_t.Tensor", "torch_t.Tensor", List[int], bool]] = {}
        work_queue = deque(windows)
        win_idx = 0

        while work_queue:
            win_recs = work_queue.popleft()
            out_pt = output_dir / f"{file_prefix}{group_key}_win{win_idx:04d}.pt"
            if skip_existing and out_pt.exists() and len(win_recs) == 1:
                win_idx += 1
                saved_count += 1
                continue

            win_en_paths = [path_by_id[r.utt_id][0] for r in win_recs]
            win_ko_paths = [path_by_id[r.utt_id][1] for r in win_recs]

            spk_id = win_recs[0].speaker_id
            prompt_24k: Optional["torch_t.Tensor"] = None
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
                    work_queue.appendleft(win_recs[mid:])
                    work_queue.appendleft(win_recs[:mid])
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
                work_queue.appendleft(win_recs[mid:])
                work_queue.appendleft(win_recs[:mid])
                continue

            if not (skip_existing and out_pt.exists()):
                save_training_example_pt(out_pt, codes, sim_bin, metadata=meta_dict)
            win_idx += 1
            saved_count += 1

    return saved_count


# ==============================================================================
# 4. SentencePiece 자동 학습 및 HuggingFace WavLM 화자 유사도 빌더
# ==============================================================================


def collect_korean_corpus(
    json_dirs_71693: Iterable[Path] = (),
    json_dirs_71379: Iterable[Path] = (),
    use_mt_fallback: bool = False,
    en_accent_71379: str = "all",
) -> List[str]:
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


def train_korean_sentencepiece(
    sentences: Sequence[str],
    output_model_path: Path,
    vocab_size: int = 32000,
    character_coverage: float = 0.9995,
    model_type: str = "unigram",
) -> int:
    """
    EDarch 특수 토큰 규격(unk=0, bos=1, eos=2, pad=3)을 준수하는 한국어 SentencePiece 모델을 학습합니다.
    소규모 샘플 코퍼스에서도 실패하지 않도록 실제 가능한 어휘 크기로 자동 조정 후 반환합니다.
    """
    if not sentences:
        raise ValueError("SentencePiece 학습을 위한 한국어 문장이 비어 있습니다.")
    import sentencepiece as spm

    output_model_path = output_model_path.resolve()
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    model_prefix = str(
        output_model_path.with_suffix("")
        if output_model_path.suffix == ".model"
        else output_model_path
    )

    unique_chars = len({ch for s in sentences for ch in s if not ch.isspace()})
    effective_vocab = min(vocab_size, max(32, unique_chars * 4, len(sentences) * 2))
    effective_vocab = max(effective_vocab, max(TEXT_EOS_ID, TEXT_PAD_ID) + 8)

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".txt", delete=False
    ) as tmp_f:
        tmp_corpus_path = Path(tmp_f.name)
        for line in sentences:
            clean_line = " ".join(line.splitlines()).strip()
            if clean_line:
                tmp_f.write(clean_line + "\n")

    try:
        cur_vocab = effective_vocab
        while True:
            try:
                spm.SentencePieceTrainer.Train(
                    input=str(tmp_corpus_path),
                    model_prefix=model_prefix,
                    vocab_size=cur_vocab,
                    character_coverage=character_coverage,
                    model_type=model_type,
                    unk_id=TEXT_UNK_ID,
                    bos_id=TEXT_BOS_ID_SPM,
                    eos_id=TEXT_EOS_ID,
                    pad_id=TEXT_PAD_ID,
                    hard_vocab_limit=False,
                )
                break
            except RuntimeError as exc:
                msg = str(exc)
                if "Vocabulary size too high" in msg and cur_vocab > 32:
                    cur_vocab = max(32, cur_vocab // 2)
                    continue
                raise
    finally:
        try:
            tmp_corpus_path.unlink(missing_ok=True)
        except OSError:
            pass

    proc = spm.SentencePieceProcessor()
    proc.Load(f"{model_prefix}.model")
    return int(proc.GetPieceSize())


def build_hf_wavlm_sim_bin_fn(
    model_id: str,
    device: "torch_t.device",
    num_speaker_sim_bins: int = 5,
) -> Callable[["torch_t.Tensor", "torch_t.Tensor"], int]:
    """HuggingFace WavLM X-Vector 모델로 화자 코사인 유사도를 0..num_speaker_sim_bins-1 구간으로 양자화합니다."""
    import torch.nn.functional as F
    from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

    feat_ext = AutoFeatureExtractor.from_pretrained(model_id)
    sv_model = AutoModelForAudioXVector.from_pretrained(model_id).to(device).eval()
    target_sr = int(getattr(feat_ext, "sampling_rate", 16000))

    def _embed(wav_24k: "torch_t.Tensor") -> "torch_t.Tensor":
        w = wav_24k.detach().cpu().float().reshape(-1)
        if w.numel() < SAMPLE_RATE // 4:
            w = F.pad(w, (0, SAMPLE_RATE // 4 - w.numel()))
        w_16k = _resample_wav(w, SAMPLE_RATE, target_sr)
        inputs = feat_ext(
            w_16k.numpy(),
            sampling_rate=target_sr,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            emb = sv_model(**inputs).embeddings[0]
        return F.normalize(emb.float(), dim=-1)

    def _score_bin(prompt_24k: "torch_t.Tensor", gen_24k: "torch_t.Tensor") -> int:
        e1 = _embed(prompt_24k)
        e2 = _embed(gen_24k)
        cos_sim = float(torch.sum(e1 * e2).item())
        norm_score = (cos_sim - 0.50) / 0.40
        bin_idx = int(norm_score * num_speaker_sim_bins)
        return max(0, min(num_speaker_sim_bins - 1, bin_idx))

    return _score_bin


def build_hf_whisper_align_fn(
    model_id: str,
    device: "torch_t.device",
) -> KoAlignFn:
    """
    HuggingFace Whisper ASR 파이프라인(return_timestamps='word')을 사용하여
    타깃 한국어 음성에서 단어 단위 발화 시작·종료 타임스탬프를 추출합니다.
    """
    from transformers import pipeline

    pipe_device = 0 if device.type == "cuda" else -1
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=pipe_device,
    )

    def _align_fn(
        ko_text: str,
        ko_wav_24k: "torch_t.Tensor",
        ko_token_ids: Sequence[int],
    ) -> Sequence[AlignmentTimestampItem]:
        del ko_text, ko_token_ids
        w_16k = _resample_wav(ko_wav_24k.detach().cpu().float().reshape(-1), SAMPLE_RATE, 16000)
        result = asr_pipe(
            w_16k.numpy(),
            return_timestamps="word",
            generate_kwargs={"language": "korean", "task": "transcribe"},
        )
        chunks = result.get("chunks", []) if isinstance(result, dict) else []
        out: List[WordTimestamp] = []
        max_sec = float(ko_wav_24k.numel()) / float(SAMPLE_RATE)
        for ch in chunks:
            ts = ch.get("timestamp")
            txt = str(ch.get("text") or "").strip()
            if not isinstance(ts, (tuple, list)) or len(ts) < 2:
                continue
            s_sec = float(ts[0]) if ts[0] is not None else 0.0
            e_sec = float(ts[1]) if ts[1] is not None else max_sec
            out.append(WordTimestamp(text=txt, start_sec=s_sec, end_sec=max(s_sec, e_sec)))
        return out

    return _align_fn


def _viterbi_ctc_forced_align(
    log_probs: "torch_t.Tensor",
    target_ids: List[int],
    blank_id: int,
) -> List[Tuple[int, int]]:
    """
    외부 C++ 확장 없이 순수 PyTorch 동적 계획법(Viterbi CTC Trellis)으로
    각 target_id 토큰의 프레임(start_frame, end_frame) 강제 정렬 구간을 계산합니다.
    """
    num_frames = int(log_probs.shape[0])
    num_targets = len(target_ids)
    if num_targets == 0 or num_frames == 0:
        return []

    ext_tokens: List[int] = [blank_id]
    for tid in target_ids:
        ext_tokens.extend([tid, blank_id])
    num_states = len(ext_tokens)

    neg_inf = -1e30
    dp = torch.full((num_frames, num_states), neg_inf, dtype=torch.float32)
    backptr = torch.zeros((num_frames, num_states), dtype=torch.long)

    dp[0, 0] = float(log_probs[0, ext_tokens[0]].item())
    if num_states > 1:
        dp[0, 1] = float(log_probs[0, ext_tokens[1]].item())

    lp_cpu = log_probs.detach().cpu().float()
    for t in range(1, num_frames):
        prev = dp[t - 1]
        for s in range(num_states):
            best_val = float(prev[s].item())
            best_src = s
            if s >= 1 and float(prev[s - 1].item()) > best_val:
                best_val = float(prev[s - 1].item())
                best_src = s - 1
            if (
                s >= 2
                and ext_tokens[s] != blank_id
                and ext_tokens[s] != ext_tokens[s - 2]
                and float(prev[s - 2].item()) > best_val
            ):
                best_val = float(prev[s - 2].item())
                best_src = s - 2
            dp[t, s] = best_val + float(lp_cpu[t, ext_tokens[s]].item())
            backptr[t, s] = best_src

    last_s = num_states - 1
    if num_states >= 2 and float(dp[-1, num_states - 2].item()) > float(dp[-1, num_states - 1].item()):
        last_s = num_states - 2

    state_path = [0] * num_frames
    cur_s = last_s
    for t in range(num_frames - 1, -1, -1):
        state_path[t] = cur_s
        cur_s = int(backptr[t, cur_s].item())

    spans: List[Tuple[int, int]] = []
    for tok_i in range(num_targets):
        target_state = 2 * tok_i + 1
        frames_for_tok = [t for t, s in enumerate(state_path) if s == target_state]
        if frames_for_tok:
            spans.append((frames_for_tok[0], frames_for_tok[-1] + 1))
        else:
            approx = (tok_i * num_frames) // max(1, num_targets)
            spans.append((approx, min(num_frames, approx + 1)))
    return spans


def build_hf_ctc_forced_align_fn(
    model_id: str,
    device: "torch_t.device",
) -> KoAlignFn:
    """
    HuggingFace 한국어 Wav2Vec2/MMS CTC 모델(예: kresnik/wav2vec2-large-xlsr-korean)과
    Viterbi 강제 정렬을 사용하여 정답 한국어 텍스트를 오디오 프레임에 직접 정렬합니다.
    """
    import torch.nn.functional as F
    from transformers import AutoModelForCTC, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    ctc_model = AutoModelForCTC.from_pretrained(model_id).to(device).eval()
    target_sr = int(getattr(processor.feature_extractor, "sampling_rate", 16000))
    vocab = processor.tokenizer.get_vocab()
    blank_id = int(getattr(ctc_model.config, "pad_token_id", 0))

    def _align_fn(
        ko_text: str,
        ko_wav_24k: "torch_t.Tensor",
        ko_token_ids: Sequence[int],
    ) -> Sequence[AlignmentTimestampItem]:
        del ko_token_ids
        w_16k = _resample_wav(ko_wav_24k.detach().cpu().float().reshape(-1), SAMPLE_RATE, target_sr)
        if w_16k.numel() < target_sr // 10:
            return []
        inputs = processor(w_16k.numpy(), sampling_rate=target_sr, return_tensors="pt")
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            logits = ctc_model(input_values).logits[0]
            log_probs = F.log_softmax(logits.float(), dim=-1)

        total_dur_sec = float(ko_wav_24k.numel()) / float(SAMPLE_RATE)
        num_ctc_frames = max(1, int(log_probs.shape[0]))
        sec_per_frame = total_dur_sec / float(num_ctc_frames)

        encoded_chars: List[int] = []
        valid_chars: List[str] = []
        for ch in ko_text:
            if ch.isspace():
                continue
            tok_ids = processor.tokenizer.encode(ch, add_special_tokens=False)
            tok_ids = [tid for tid in tok_ids if tid != blank_id and tid in vocab.values()]
            if tok_ids:
                encoded_chars.extend(tok_ids)
                valid_chars.extend([ch] * len(tok_ids))

        if not encoded_chars or num_ctc_frames < len(encoded_chars):
            return []

        spans = _viterbi_ctc_forced_align(log_probs, encoded_chars, blank_id=blank_id)
        return [
            WordTimestamp(
                text=ch,
                start_sec=s_f * sec_per_frame,
                end_sec=max(s_f * sec_per_frame, e_f * sec_per_frame),
            )
            for ch, (s_f, e_f) in zip(valid_chars, spans)
        ]

    return _align_fn


# ==============================================================================
# 5. CLI 실행 엔트리포인트 (71693 + 71379 통합 및 단독 실행 지원)
# ==============================================================================


def _detect_datasets_in_json_dir(json_dir: Path) -> Tuple[bool, bool]:
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


def _parse_args() -> argparse.Namespace:
    from env_setup import DEFAULT_CUDA_INDEX_URL, get_default_venv_dir

    parser = argparse.ArgumentParser(
        description="EDarch AI-Hub 71693(국제학술대회) & 71379(방송콘텐츠) 통합 전처리(.pt 생성) 스크립트"
    )
    parser.add_argument("--venv-dir", type=Path, default=get_default_venv_dir())
    parser.add_argument("--no-venv", action="store_true")
    parser.add_argument("--torch-index-url", type=str, default=DEFAULT_CUDA_INDEX_URL)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--reinstall", action="store_true")

    # 기존 단일 디렉터리 호환 인자 (--dataset auto로 71693/71379 자동 판별)
    parser.add_argument(
        "--dataset",
        choices=("auto", "71693", "71379"),
        default="auto",
        help="--json-dir/--wav-dir 사용 시 데이터셋 종류 (기본값: auto 자동 감지)",
    )
    parser.add_argument("--json-dir", type=Path, default=None, help="공통 또는 단일 데이터셋 JSON 디렉터리")
    parser.add_argument("--wav-dir", type=Path, default=None, help="공통 또는 단일 데이터셋 WAV 디렉터리")

    # AI-Hub 71693 전용 경로 인자
    parser.add_argument(
        "--root-71693",
        type=Path,
        default=None,
        help="71693 데이터셋 루트 디렉터리 (내부 JSON/WAV 하위 폴더 자동 탐색)",
    )
    parser.add_argument("--json-dir-71693", type=Path, default=None, help="71693 JSON 라벨 디렉터리")
    parser.add_argument("--wav-dir-71693", type=Path, default=None, help="71693 원본 영어 강의 WAV 디렉터리")

    # AI-Hub 71379 전용 경로 인자
    parser.add_argument(
        "--root-71379",
        type=Path,
        default=None,
        help="71379 데이터셋 루트 디렉터리 (원천 한국어 WAV / 원어민 영어 WAV / 라벨 JSON 자동 탐색)",
    )
    parser.add_argument("--json-dir-71379", type=Path, default=None, help="71379 JSON 라벨 디렉터리")
    parser.add_argument("--wav-dir-71379", type=Path, default=None, help="71379 통합 WAV 디렉터리")
    parser.add_argument(
        "--en-wav-dir-71379",
        type=Path,
        default=None,
        help="71379 원어민 영어 발화 WAV 디렉터리 (미지정 시 --wav-dir-71379 또는 --root-71379 사용)",
    )
    parser.add_argument(
        "--ko-wav-dir-71379",
        type=Path,
        default=None,
        help="71379 한국어 원천 음성 WAV 디렉터리 (미지정 시 --wav-dir-71379 또는 --root-71379 사용)",
    )
    parser.add_argument(
        "--ko-audio-mode-71379",
        choices=("auto", "real", "tts"),
        default="auto",
        help="71379 타깃 한국어 오디오 선택: auto(원천 WAV 우선, 없으면 TTS), real(원천 WAV만), tts(TTS 합성)",
    )
    parser.add_argument(
        "--en-accent-71379",
        choices=("all", "us", "uk"),
        default="all",
        help="71379 영어 발화 억양 필터 (all=미국+영국, us=en-US, uk=en-GB)",
    )
    parser.add_argument(
        "--inter-utt-pause-sec",
        type=float,
        default=0.5,
        help="71379 연속 발화 결합 시 기본 발화 간 휴지기(초)",
    )
    parser.add_argument(
        "--max-overlap-sec",
        type=float,
        default=4.0,
        help="71379 연속 발화 결합 시 이전 한국어 통역과 다음 영어 발화의 최대 동시 중첩 허용 길이(초)",
    )
    parser.add_argument(
        "--real-pair-sim-bin",
        type=int,
        default=None,
        help="WavLM 미사용 시 71379 실제 이종 화자(원어민 영어 vs 한국어 원천) 쌍에 부여할 sim_bin(0..4)",
    )

    # 공통 출력 및 모델 설정
    parser.add_argument("--output-dir", type=Path, required=True, help="전처리된 .pt 저장 디렉터리 (train.py --data-dir)")
    parser.add_argument(
        "--preset",
        choices=("1b", "2b"),
        default="1b",
        help="모델 프리셋 (기본값: 1b -> K=17 스트림; 2b -> K=33 스트림)",
    )
    parser.add_argument("--ko-vocab-size", type=int, default=None, help="한국어 어휘 크기 (예: 32000)")
    parser.add_argument("--target-window-sec", type=float, default=45.0, help="목표 스트리밍 윈도우 길이(초)")
    parser.add_argument("--max-gap-sec", type=float, default=8.0, help="윈도우 분할 기준 최대 공백(초)")
    parser.add_argument(
        "--use-mt-fallback",
        action="store_true",
        help="검수 번역문(MTPE/최종번역문)이 비어 있을 때 기계번역(MT)을 대체 사용",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="이미 생성된 .pt 파일이 있으면 건너뛰기 (대용량 전처리 재개용)",
    )

    # Inner Monologue 정밀 정렬 옵션
    parser.add_argument(
        "--align-mode",
        choices=("auto", "energy", "whisper", "ctc", "uniform"),
        default="auto",
        help="타깃 텍스트-오디오 정렬 방식: auto(외부 정렬기 있으면 사용, 없으면 energy), energy(음향 RMS/VAD 에너지 정렬), whisper(단어 타임스탬프), ctc(Viterbi 강제 정렬), uniform(단순 등간격)",
    )
    parser.add_argument(
        "--whisper-align-model",
        type=str,
        default=None,
        help="Whisper 단어 타임스탬프 정렬용 HuggingFace 모델 ID (예: openai/whisper-small)",
    )
    parser.add_argument(
        "--ctc-align-model",
        type=str,
        default=None,
        help="한국어 CTC 강제 정렬(Forced Alignment)용 HuggingFace 모델 ID (예: kresnik/wav2vec2-large-xlsr-korean)",
    )
    parser.add_argument(
        "--monologue-lead-frames",
        type=int,
        default=1,
        help="타깃 발화 음향 프레임 대비 Inner Monologue 텍스트 토큰 선행 프레임 수 (기본값: 1 = 80ms 선행)",
    )

    # 외부 모델 / 훅 설정
    parser.add_argument(
        "--hooks-module",
        type=str,
        default=None,
        help="zero_shot_ko_tts_fn, ko_sp_encode_fn, ko_align_fn, mimi_encode_fn, wavlm_sim_bin_fn 제공 파이썬 모듈명",
    )
    parser.add_argument("--sp-model", type=Path, default=None, help="한국어 SentencePiece .model 경로")
    parser.add_argument(
        "--train-sp-if-missing",
        action="store_true",
        help="--sp-model 파일이 없을 경우 71693/71379 한국어 문장으로 SentencePiece(eos=2, pad=3) 자동 학습",
    )
    parser.add_argument(
        "--mimi-model",
        type=str,
        default=None,
        help="사전학습 Mimi 코덱 ID 또는 로컬 경로 (예: kyutai/mimi)",
    )
    parser.add_argument(
        "--wavlm-model",
        type=str,
        default=None,
        help="화자 유사도(sim_bin) 계산용 HuggingFace WavLM 모델 ID (예: microsoft/wavlm-base-plus-sv)",
    )
    parser.add_argument(
        "--demo-hooks",
        action="store_true",
        help="외부 TTS/토크나이저/Mimi 없이 파이프라인 및 train.py 연동 검증용 합성 훅으로 실행",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _resolve_cli_tasks(
    args: argparse.Namespace,
) -> Tuple[Optional[Tuple[Path, Path]], Optional[Tuple[Path, Optional[Path], Optional[Path], Optional[Path]]]]:
    task_71693: Optional[Tuple[Path, Path]] = None
    task_71379: Optional[Tuple[Path, Optional[Path], Optional[Path], Optional[Path]]] = None

    j93 = args.json_dir_71693 or args.root_71693
    w93 = args.wav_dir_71693 or args.root_71693
    if j93 is not None or w93 is not None:
        if j93 is None or w93 is None:
            raise ValueError("71693 경로 지정 시 --root-71693 또는 (--json-dir-71693 및 --wav-dir-71693)가 필요합니다.")
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
            has_93, has_79 = _detect_datasets_in_json_dir(args.json_dir)
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


def _run_cli(args: argparse.Namespace) -> None:
    global torch, torchaudio
    import torch as _torch
    import torchaudio as _torchaudio

    torch = _torch
    torchaudio = _torchaudio

    from EDarch_from_scratch import EDarchConfig, ExternalMimiCodec, MiniMimiCodec
    from env_setup import setup_device

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

    if sp_fn is None and args.sp_model is not None:
        if not args.sp_model.exists():
            if args.train_sp_if_missing:
                logger.info(
                    "SentencePiece 모델(%s)이 없어 데이터셋 한국어 문장으로 자동 학습을 시작합니다...",
                    args.sp_model,
                )
                corpus = collect_korean_corpus(
                    json_dirs_71693=[task_71693[0]] if task_71693 else (),
                    json_dirs_71379=[task_71379[0]] if task_71379 else (),
                    use_mt_fallback=args.use_mt_fallback,
                    en_accent_71379=args.en_accent_71379,
                )
                trained_vocab = train_korean_sentencepiece(
                    sentences=corpus,
                    output_model_path=args.sp_model,
                    vocab_size=cfg.text_card,
                )
                logger.info(
                    "SentencePiece 학습 완료: %s (어휘 크기=%d)", args.sp_model, trained_vocab
                )
            else:
                raise FileNotFoundError(
                    f"SentencePiece 모델 파일을 찾을 수 없습니다: {args.sp_model} "
                    "(자동 생성하려면 --train-sp-if-missing 옵션을 추가하세요.)"
                )

        import sentencepiece as spm

        sp_proc = spm.SentencePieceProcessor()
        sp_proc.Load(str(args.sp_model))
        sp_eos = sp_proc.eos_id()
        sp_pad = sp_proc.pad_id()
        sp_vocab = sp_proc.GetPieceSize()
        if sp_eos != TEXT_EOS_ID:
            raise ValueError(
                f"SentencePiece eos_id({sp_eos})가 EDarch TEXT_EOS_ID({TEXT_EOS_ID})와 일치하지 않습니다. "
                "SentencePiece 학습 시 --eos_id=2 설정이 필요합니다."
            )
        if sp_pad != TEXT_PAD_ID:
            raise ValueError(
                f"SentencePiece pad_id({sp_pad})가 EDarch TEXT_PAD_ID({TEXT_PAD_ID})와 일치하지 않습니다. "
                "SentencePiece 학습 시 --pad_id=3 설정이 필요합니다."
            )
        if args.ko_vocab_size is None and sp_vocab != cfg.text_card:
            logger.info(
                "SentencePiece 어휘 크기(%d)에 맞춰 text_card를 %d -> %d로 설정합니다.",
                sp_vocab,
                cfg.text_card,
                sp_vocab,
            )
            cfg.text_card = sp_vocab
        elif sp_vocab > cfg.text_card:
            raise ValueError(
                f"SentencePiece 어휘 크기({sp_vocab})가 --ko-vocab-size({cfg.text_card})보다 큽니다."
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
            raise ValueError("--align-mode whisper 사용 시 --whisper-align-model 또는 ko_align_fn이 필요합니다.")
        elif args.align_mode == "ctc":
            raise ValueError("--align-mode ctc 사용 시 --ctc-align-model 또는 ko_align_fn이 필요합니다.")

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
            "--hooks-module(--sp-model, --mimi-model)을 지정하거나 데모 검증 시 --demo-hooks를 사용하세요."
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
                "71379 단독 전처리 시 WavLM이 지정되지 않아 추론(try.py)의 very_good_id(%d) 조건과 "
                "정합되도록 sim_bin=%d로 설정합니다. 화자 유사도 분별 학습을 원하면 --wavlm-model을 지정하거나 "
                "71693과 함께 전처리하세요.",
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
        logger.info("AI-Hub 71693 전처리 시작: json=%s, wav=%s -> %s", task_71693[0], task_71693[1], out_93)
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
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

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


def main() -> None:
    from env_setup import ensure_venv_and_reexec

    args = _parse_args()
    ensure_venv_and_reexec(
        script_path=Path(__file__),
        venv_dir=args.venv_dir,
        torch_index_url=args.torch_index_url,
        cpu_only=args.cpu,
        reinstall=args.reinstall,
        no_venv=args.no_venv,
    )
    _run_cli(args)


if __name__ == "__main__":
    main()
