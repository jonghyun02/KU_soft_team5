"""D:\\data(AI-Hub 71693) 및 D:\\data2(AI-Hub 71379) 저메모리·대용량 병렬 전처리 엔진 모듈."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import gc
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, TypeVar

import soundfile as sf
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
    resample_wav,
)
from libs.config import EDarchConfig
from libs.env import atomic_torch_save, setup_device
from libs.tokenizer import (
    TEXT_EOS_ID,
    TEXT_PAD_ID,
    load_korean_tokenizer,
    normalize_korean_text,
    resolve_spm_model_path,
    train_korean_sentencepiece,
)

logger = logging.getLogger(__name__)

ZERO_TOKEN_ID = -1
_EMPTY_MARKERS = {"", "-", "n/a", "na", "null", "none", "x"}
_LEADING_NOISE_RE = re.compile(r"^(?:##+|\*\*+|//+)\s*")
_SENT_SPLIT_RE = re.compile(r"(?<=[.?!,;])\s+")

T_In = TypeVar("T_In")
T_Out = TypeVar("T_Out")

# ==============================================================================
# 1. 한국어 로마자 변환기 및 OOM 방지 청크형 TTS 빌더
# ==============================================================================

_CHOSEONG = (
    "g", "kk", "n", "d", "tt", "r", "m", "b", "pp",
    "s", "ss", "", "j", "jj", "ch", "k", "t", "p", "h",
)
_JUNGSEONG = (
    "a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae",
    "oe", "yo", "u", "wo", "we", "wi", "yu", "eu", "ui", "i",
)
_JONGSEONG = (
    "", "k", "k", "ks", "n", "nj", "nh", "t", "l", "lk", "lm", "lb",
    "ls", "lt", "lp", "lh", "m", "p", "ps", "t", "t", "ng", "t", "t",
    "k", "t", "p", "t",
)
_LIAISON_NEXT_CHO: dict[int, Tuple[str, str]] = {
    1: ("", "g"),
    2: ("", "kk"),
    3: ("k", "s"),
    4: ("", "n"),
    5: ("n", "j"),
    6: ("n", "h"),
    7: ("", "d"),
    8: ("", "r"),
    9: ("l", "g"),
    10: ("l", "m"),
    11: ("l", "b"),
    12: ("l", "s"),
    13: ("l", "t"),
    14: ("l", "p"),
    15: ("l", "h"),
    16: ("", "m"),
    17: ("", "b"),
    18: ("b", "s"),
    19: ("", "s"),
    20: ("", "ss"),
    22: ("", "j"),
    23: ("", "ch"),
    24: ("", "k"),
    25: ("", "t"),
    26: ("", "p"),
}
_DIGIT_KO = {
    "0": "yeong", "1": "il", "2": "i", "3": "sam", "4": "sa",
    "5": "o", "6": "yuk", "7": "chil", "8": "pal", "9": "gu",
}
_LATIN_FALLBACK = {"f": "p", "q": "k", "v": "b", "x": "ks", "z": "j"}
_MMS_ALLOWED_CHARS = set(" '-_abcdeghijklmnoprstuwy")


def romanize_korean_for_mms(text: str) -> str:
    """한국어 문장을 `facebook/mms-tts-kor`의 25자 uroman 어휘에 맞춰 로마자로 변환합니다."""
    norm = normalize_korean_text(text)
    if not norm:
        return "a"

    syllables: List[Optional[Tuple[int, int, int]]] = []
    raw_chars: List[str] = []
    for ch in norm:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            idx = code - 0xAC00
            syllables.append((idx // 588, (idx % 588) // 28, idx % 28))
        else:
            syllables.append(None)
        raw_chars.append(ch)

    n = len(raw_chars)
    override_cho: dict[int, str] = {}
    out_parts: List[str] = []

    for i in range(n):
        syl = syllables[i]
        if syl is not None:
            cho_i, jung_i, jong_i = syl
            cho_str = override_cho.get(i, _CHOSEONG[cho_i])
            jung_str = _JUNGSEONG[jung_i]
            jong_str = _JONGSEONG[jong_i]

            if jong_i in _LIAISON_NEXT_CHO and i + 1 < n and syllables[i + 1] is not None:
                next_cho, _, _ = syllables[i + 1]
                if next_cho == 11:
                    rem_jong, moved_cho = _LIAISON_NEXT_CHO[jong_i]
                    jong_str = rem_jong
                    override_cho[i + 1] = moved_cho

            out_parts.append(cho_str + jung_str + jong_str)
            continue

        ch = raw_chars[i]
        if ch in _DIGIT_KO:
            out_parts.append(f" {_DIGIT_KO[ch]} ")
        elif ch == "%":
            out_parts.append(" peosenteu ")
        elif "a" <= ch.lower() <= "z":
            lc = ch.lower()
            out_parts.append(_LATIN_FALLBACK.get(lc, lc))
        elif ch in (",", ".", "?", "!", ";", ":", "/", "(", ")"):
            out_parts.append(" ")
        elif ch in ("'", "-"):
            out_parts.append(ch)
        elif ch.isspace():
            out_parts.append(" ")

    joined = "".join(out_parts)
    filtered = "".join(c for c in joined if c in _MMS_ALLOWED_CHARS)
    cleaned = " ".join(filtered.split())
    return cleaned if cleaned else "a"


def _split_long_korean_text(text: str, max_chars: int = 140) -> List[str]:
    """장문 한국어 발화로 인한 VITS HiFi-GAN VRAM 스파이크를 막기 위해 절 단위로 분할합니다."""
    norm = normalize_korean_text(text)
    if len(norm) <= max_chars:
        return [norm] if norm else ["아"]

    raw_clauses = _SENT_SPLIT_RE.split(norm)
    chunks: List[str] = []
    cur = ""
    for cl in raw_clauses:
        for word in (cl.split() if len(cl) > max_chars else [cl]):
            cand = f"{cur} {word}".strip() if cur else word
            if len(cand) <= max_chars:
                cur = cand
            else:
                if cur:
                    chunks.append(cur)
                cur = word[:max_chars]
    if cur:
        chunks.append(cur)
    return chunks if chunks else [norm[:max_chars]]


def build_mms_korean_tts_fn(
    device: torch.device,
    model_id: str = "facebook/mms-tts-kor",
) -> Callable[[str, torch.Tensor, str], torch.Tensor]:
    """장문 자동 청킹이 적용된 `VitsModel`(`facebook/mms-tts-kor`) 로컬 GPU 한국어 TTS 함수를 반환합니다."""
    from transformers import AutoTokenizer, VitsModel

    logger.info("로컬 GPU 한국어 VITS TTS 모델 로드 중: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    vits = VitsModel.from_pretrained(model_id).to(device).eval()
    tts_sr = int(getattr(vits.config, "sampling_rate", 16000))

    @torch.inference_mode()
    def _synthesize(ko_text: str, speaker_prompt_24k: torch.Tensor, gender: str = "") -> torch.Tensor:
        del speaker_prompt_24k, gender
        wav_pieces: List[torch.Tensor] = []
        for sub_txt in _split_long_korean_text(ko_text, max_chars=140):
            roman = romanize_korean_for_mms(sub_txt)
            inputs = tokenizer(roman, return_tensors="pt")
            input_ids = inputs["input_ids"][:, :512].to(device)
            if input_ids.shape[-1] == 0:
                continue
            w = vits(input_ids=input_ids).waveform[0].detach().cpu().float().reshape(-1)
            if w.numel() > 0:
                wav_pieces.append(w)

        if not wav_pieces:
            return torch.zeros(SAMPLES_PER_FRAME * 2, dtype=torch.float32)

        wav = torch.cat(wav_pieces, dim=0) if len(wav_pieces) > 1 else wav_pieces[0]
        if tts_sr != SAMPLE_RATE:
            wav = resample_wav(wav, tts_sr, SAMPLE_RATE)
        peak = float(wav.abs().max().item())
        if peak > 1e-4:
            wav = wav * min(1.0, 0.9 / peak)
        return wav

    return _synthesize


def build_edge_korean_tts_fn(
    cache_dir: Path,
    male_voice: str = "ko-KR-InJoonNeural",
    female_voice: str = "ko-KR-SunHiNeural",
) -> Callable[[str, torch.Tensor, str], torch.Tensor]:
    """Microsoft Edge 신경망 한국어 TTS(성별 매칭 + 디스크 캐시) 함수를 생성합니다."""
    import asyncio
    import edge_tts

    cache_dir.mkdir(parents=True, exist_ok=True)

    def _synthesize(ko_text: str, speaker_prompt_24k: torch.Tensor, gender: str = "") -> torch.Tensor:
        del speaker_prompt_24k
        voice = female_voice if gender.lower().startswith("f") else male_voice
        digest = hashlib.sha1(f"{voice}:{ko_text}".encode("utf-8")).hexdigest()
        cached_mp3 = cache_dir / f"{digest[:2]}" / f"{digest}.mp3"
        cached_mp3.parent.mkdir(parents=True, exist_ok=True)

        if not cached_mp3.exists() or cached_mp3.stat().st_size == 0:
            async def _run_tts() -> None:
                comm = edge_tts.Communicate(ko_text, voice)
                await comm.save(str(cached_mp3))

            for attempt in range(3):
                try:
                    asyncio.run(_run_tts())
                    break
                except Exception as exc:
                    if attempt == 2:
                        raise RuntimeError(f"edge-tts 합성 실패 ({ko_text[:30]}...): {exc}") from exc
                    time.sleep(0.5 * (attempt + 1))

        return load_mono_wav_24k(cached_mp3)

    return _synthesize


# ==============================================================================
# 2. 저메모리 배치 스레드풀·오디오 슬라이스 로더·디렉터리 탐색
# ==============================================================================


def _batched_pool_map(
    fn: Callable[[T_In], T_Out],
    items: Sequence[T_In],
    num_workers: int = 16,
    batch_size: int = 4096,
) -> Iterator[T_Out]:
    """수십만 개 작업의 `Future`가 한꺼번에 RAM에 쌓이지 않도록 고정 배치 단위로 스트리밍 실행합니다."""
    total = len(items)
    if total == 0:
        return
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
        for start in range(0, total, batch_size):
            chunk = items[start : start + batch_size]
            for out in pool.map(fn, chunk):
                yield out


def fast_load_mono_wav_24k(
    wav_path: Path | str,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> torch.Tensor:
    """`soundfile`을 이용해 전체 또는 필요한 시간 구간만 부분 로드하여 RAM 사용량을 최소화합니다."""
    path_str = str(wav_path)
    try:
        if start_sec is not None and end_sec is not None:
            with sf.SoundFile(path_str, mode="r") as sfile:
                sr = int(sfile.samplerate)
                total_frames = len(sfile)
                s_frame = max(0, min(int(start_sec * sr), total_frames))
                e_frame = max(s_frame, min(int(end_sec * sr) + 1, total_frames))
                if e_frame <= s_frame:
                    return torch.zeros(0, dtype=torch.float32)
                sfile.seek(s_frame)
                data = sfile.read(e_frame - s_frame, dtype="float32", always_2d=True)
        else:
            data, sr = sf.read(path_str, dtype="float32", always_2d=True)

        if data.size == 0:
            return torch.zeros(0, dtype=torch.float32)
        mono = torch.from_numpy(data.mean(axis=1) if data.shape[1] > 1 else data[:, 0])
        if int(sr) != SAMPLE_RATE:
            mono = resample_wav(mono, int(sr), SAMPLE_RATE)
        return mono.contiguous()
    except Exception:
        full = load_mono_wav_24k(Path(path_str)).detach().cpu().float().reshape(-1)
        if start_sec is not None and end_sec is not None:
            s_i = max(0, min(int(start_sec * SAMPLE_RATE), full.numel()))
            e_i = max(s_i, min(int(end_sec * SAMPLE_RATE), full.numel()))
            return full[s_i:e_i].contiguous()
        return full


def _fast_scandir_files(
    dirs: Iterable[Path],
    suffix: str,
) -> Tuple[dict[str, str], List[str]]:
    """`Path` 객체 오버헤드 없이 가벼운 문자열(`str`) 경로로 대용량 디렉터리를 고속 인덱싱합니다."""
    by_lower_name: dict[str, str] = {}
    ordered_paths: List[str] = []
    sfx = suffix.lower()

    stack = list(reversed([str(d) for d in dirs if d is not None and d.exists()]))
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                sub_dirs: List[str] = []
                files: List[Tuple[str, str]] = []
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        sub_dirs.append(entry.path)
                    elif entry.is_file(follow_symlinks=False) and entry.name.lower().endswith(sfx):
                        files.append((entry.name, entry.path))
                for sd in sorted(sub_dirs, reverse=True):
                    stack.append(sd)
                files.sort(key=lambda pair: pair[0])
                for fname, fpath in files:
                    ordered_paths.append(fpath)
                    lname = fname.lower()
                    if lname not in by_lower_name:
                        by_lower_name[lname] = fpath
        except OSError as exc:
            logger.warning("디렉터리 스캔 건너뜀 (%s): %s", cur, exc)

    return by_lower_name, ordered_paths


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
    """`D:\\data` 구조에서 Training/Validation 경로와 Sublabel 한국어 코퍼스 경로를 탐색합니다."""
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
    """`D:\\data2` 구조에서 Training/Validation의 한국어음성·발화녹음·음성전사·통번역 폴더를 분리 탐색합니다."""
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


# ==============================================================================
# 3. 슬림 레코드(`__slots__`) 및 저메모리 병렬 JSON 파서
# ==============================================================================


def _clean_str(val: Any) -> str:
    if val is None or isinstance(val, bool):
        return ""
    text = _LEADING_NOISE_RE.sub("", str(val).strip()).strip()
    return "" if text.lower() in _EMPTY_MARKERS else text


def _parse_ts(ts: Any) -> float:
    if isinstance(ts, (int, float)) and not isinstance(ts, bool):
        return float(ts)
    cleaned = str(ts).strip().replace(",", ".")
    parts = cleaned.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


@dataclass(slots=True)
class Span71693:
    """`slots=True`로 메모리 오버헤드를 최소화한 AI-Hub 71693 발화 구간 데이터클래스입니다."""

    sn: str
    file_name: str
    start_sec: float
    end_sec: float
    en_text: str
    ko_text: str
    gender: str = ""
    segment_break: bool = False


@dataclass(slots=True)
class Utterance71379:
    """`slots=True`로 40만 개 적재 시에도 수백 MB 이하만 사용하는 AI-Hub 71379 발화 레코드입니다."""

    utt_id: str
    scene_key: str
    seq_num: int
    variant: str
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


def _read_json_str(path_str: str) -> Optional[Any]:
    try:
        with open(path_str, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _interleave_lists(groups: Sequence[Sequence[Any]]) -> List[Any]:
    """여러 도메인·카테고리 목록을 라운드로빈으로 고르게 교차 배치합니다."""
    non_empty = [list(g) for g in groups if g]
    if not non_empty:
        return []
    max_len = max(len(g) for g in non_empty)
    out: List[Any] = []
    for i in range(max_len):
        for g in non_empty:
            if i < len(g):
                out.append(g[i])
    return out


def parse_71693_split(
    layout: SplitLayout71693,
    use_mt_fallback: bool = True,
    max_lectures: Optional[int] = None,
    num_io_threads: int = 16,
) -> dict[str, List[Span71693]]:
    """71693 라벨 JSON들을 도메인별 균등 순서로 저메모리 배치 파싱합니다."""
    domain_dirs: List[Path] = []
    for jd in layout.json_dirs:
        if jd.exists():
            subs = sorted(d for d in jd.iterdir() if d.is_dir())
            domain_dirs.extend(subs if subs else [jd])
    per_domain_jsons = [_fast_scandir_files([dd], ".json")[1] for dd in domain_dirs]
    json_paths: List[str] = _interleave_lists(per_domain_jsons)
    if max_lectures is not None and max_lectures > 0:
        json_paths = json_paths[: max_lectures * 2]

    grouped: dict[str, List[Span71693]] = defaultdict(list)
    skipped_intervals: dict[str, List[Tuple[float, float]]] = defaultdict(list)
    seen_keys: Set[Tuple[str, str]] = set()

    for payload in _batched_pool_map(_read_json_str, json_paths, num_workers=num_io_threads, batch_size=256):
        if payload is None:
            continue
        records = payload.get("data", []) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            continue
        for raw in records:
            if not isinstance(raw, dict):
                continue
            item = {str(k).lstrip("\ufeff").strip().lower(): v for k, v in raw.items()}
            src_lang = str(item.get("source_language") or "").strip().lower()
            tgt_lang = str(item.get("target_language") or "").strip().lower()
            if not (src_lang.startswith("en") and tgt_lang.startswith("ko")):
                continue
            fname = str(item.get("file_name") or "").strip()
            sn = str(item.get("sn") or "").strip()
            if not fname or not sn or (fname, sn) in seen_keys:
                continue
            try:
                s_sec = _parse_ts(item["audio_start"])
                e_sec = _parse_ts(item["audio_end"])
            except (KeyError, ValueError, TypeError):
                continue
            if e_sec <= s_sec:
                continue
            seen_keys.add((fname, sn))

            ko_text = _clean_str(item.get("mtpe"))
            if not ko_text and use_mt_fallback:
                ko_text = _clean_str(item.get("mt"))
            en_text = _clean_str(item.get("source_cleaned")) or _clean_str(
                item.get("source_original")
            )
            if not ko_text:
                skipped_intervals[fname].append((s_sec, e_sec))
                continue

            grouped[fname].append(
                Span71693(
                    sn=sn,
                    file_name=fname,
                    start_sec=s_sec,
                    end_sec=e_sec,
                    en_text=en_text,
                    ko_text=ko_text,
                    gender=_clean_str(item.get("info_gender")),
                )
            )

    ordered_fnames = list(grouped.keys())
    if max_lectures is not None and max_lectures > 0:
        ordered_fnames = ordered_fnames[:max_lectures]

    final_grouped: dict[str, List[Span71693]] = {}
    for fname in ordered_fnames:
        spans = grouped[fname]
        spans.sort(key=lambda x: (x.start_sec, x.end_sec))
        sk_list = skipped_intervals.get(fname)
        for idx in range(1, len(spans)):
            p_end = spans[idx - 1].end_sec
            c_start = spans[idx].start_sec
            if sk_list and c_start > p_end and any(
                sk_e > p_end and sk_s < c_start for sk_s, sk_e in sk_list
            ):
                spans[idx].segment_break = True
        final_grouped[fname] = spans

    return final_grouped


def extract_sublabel_korean_sentences(
    sublabel_dirs: Sequence[Path],
    num_io_threads: int = 16,
) -> List[str]:
    """`D:\\data\\Sublabel\\SbL\\1.한국어` 학술 강연 한국어 문장을 저메모리로 추출합니다."""
    if not sublabel_dirs:
        return []
    _, json_paths = _fast_scandir_files(sublabel_dirs, ".json")
    sentences: List[str] = []
    for payload in _batched_pool_map(_read_json_str, json_paths, num_workers=num_io_threads, batch_size=512):
        if not isinstance(payload, list):
            continue
        for rec in payload:
            if isinstance(rec, dict):
                s = _clean_str(rec.get("source_sentence"))
                if s:
                    sentences.append(s)
    return sentences


def _parse_scene_and_seq(ko_wav_name: str, en_wav_name: str) -> Tuple[str, int]:
    base = ko_wav_name or en_wav_name
    stem = os.path.splitext(os.path.basename(base))[0]
    for sfx in ("_en-us", "_en-gb", "_en_us", "_en_gb", "_en", "_uk"):
        if stem.lower().endswith(sfx):
            stem = stem[: -len(sfx)]
            break
    parts = stem.split("_")
    if len(parts) >= 6 and parts[5].isdigit():
        return "_".join(parts[:5]), int(parts[5])
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1]), int(parts[-1])
    return stem, 0


# 워커 스레드에서 반환할 10-필드 경량 튜플 타입 (원본 30키 dict 보관 대비 RAM 90% 절약)
_SlimTransTuple = Tuple[str, str, str, str, str, str, str, str, float, float]


def parse_71379_split(
    layout: SplitLayout71379,
    en_accent: str = "all",
    use_mt_fallback: bool = True,
    max_utterances: Optional[int] = None,
    num_io_threads: int = 16,
) -> dict[str, List[Utterance71379]]:
    """71379 통번역·음성전사 JSON을 고정 크기 배치 스트리밍으로 읽어 시스템 RAM 스파이크를 원천 차단합니다."""
    t0 = time.perf_counter()
    per_cat_trans: List[List[str]] = []
    for td in layout.trans_json_dirs:
        _, t_list = _fast_scandir_files([td], ".json")
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
    interleaved_blocks = _interleave_lists(
        [
            [grp[i : i + block_sz] for i in range(0, len(grp), block_sz)]
            for grp in per_cat_trans
        ]
    )
    trans_paths: List[str] = [p for blk in interleaved_blocks for p in blk]
    del per_cat_trans, interleaved_blocks

    stt_index: dict[str, str] = {}
    if layout.stt_json_dirs and layout.stt_json_dirs != layout.trans_json_dirs:
        stt_by_name, _ = _fast_scandir_files(layout.stt_json_dirs, ".json")
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
        item = {str(k).lstrip("\ufeff").strip().lower(): v for k, v in payload.items()}
        if "t-fname" not in item and "최종번역문" not in item and "원문" not in item:
            return None
        s_code = str(item.get("s-code") or "ko-KR").strip().lower()
        if s_code and not s_code.startswith("ko"):
            return None

        t_code = str(item.get("t-code") or "").strip().lower()
        en_wav_name = _clean_str(item.get("t-fname"))
        jname_lower = os.path.basename(jpath).lower()
        if t_code in ("es", "es-es", "ru", "ru-ru"):
            return None
        if t_code in ("en-gb", "uk", "en_gb") or en_wav_name.lower().endswith("_uk.wav"):
            variant = "uk"
        elif (
            t_code in ("en-us", "en", "en_us", "eng")
            or en_wav_name.lower().endswith("_en.wav")
            or jname_lower.endswith("_en.json")
        ):
            variant = "us"
        else:
            return None

        if en_accent != "all" and variant != en_accent:
            return None

        ko_wav_name = _clean_str(item.get("s-fname"))
        if not en_wav_name and ko_wav_name:
            stem0 = os.path.splitext(ko_wav_name)[0]
            en_wav_name = f"{stem0}{'_uk.wav' if variant == 'uk' else '_en.wav'}"
        if not en_wav_name:
            return None

        ko_text = _clean_str(item.get("수정원문")) or _clean_str(item.get("원문"))
        en_text = (
            _clean_str(item.get("수정번역문"))
            or _clean_str(item.get("최종번역문"))
            or _clean_str(item.get("최종번역"))
            or _clean_str(item.get("2차수정"))
            or _clean_str(item.get("1차수정"))
        )
        if not en_text and use_mt_fallback:
            en_text = _clean_str(item.get("mt"))

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
            _clean_str(item.get("발화자id")),
            _clean_str(item.get("s-usex")).lower(),
            _clean_str(item.get("t-usex")).lower(),
            s_ulen,
            t_ulen,
        )

    slim_items: List[_SlimTransTuple] = []
    needed_stt_stems: Set[str] = set()

    for idx, res in enumerate(
        _batched_pool_map(_read_trans_slim, trans_paths, num_workers=num_io_threads, batch_size=4096),
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
            item = {str(k).lstrip("\ufeff").strip().lower(): v for k, v in payload.items()}
            is_deid = str(item.get("비식별화") or "").strip().upper() == "O"
            ko_stt = (
                _clean_str(item.get("번역본"))
                or _clean_str(item.get("철자전사"))
                or _clean_str(item.get("전사작업본"))
            )
            raw_s: Optional[float] = None
            raw_e: Optional[float] = None
            try:
                if item.get("원시시작") is not None and item.get("원시끝") is not None:
                    raw_s = _parse_ts(item["원시시작"])
                    raw_e = _parse_ts(item["원시끝"])
            except (ValueError, TypeError):
                raw_s, raw_e = None, None
            return stem, ko_stt, raw_s, raw_e, is_deid

        for stt_res in _batched_pool_map(
            _read_stt_item, stt_targets, num_workers=num_io_threads, batch_size=4096
        ):
            if stt_res is not None:
                stem, ko_stt, raw_s, raw_e, is_deid = stt_res
                stt_meta_map[stem] = (ko_stt, raw_s, raw_e, is_deid)
    del stt_targets

    grouped: dict[str, List[Utterance71379]] = defaultdict(list)
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

        scene_key, seq_num = _parse_scene_and_seq(ko_wav_name, en_wav_name)
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
            Utterance71379(
                utt_id=utt_id,
                scene_key=scene_key,
                seq_num=seq_num,
                variant=variant,
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
        recs.sort(key=lambda r: (r.seq_num, r.utt_id))
        sk = skipped_seqs.get(group_key, set())
        for idx in range(1, len(recs)):
            prev_r, cur_r = recs[idx - 1], recs[idx]
            non_consec = (
                cur_r.seq_num <= prev_r.seq_num
                or (cur_r.seq_num - prev_r.seq_num > 1)
                or any(prev_r.seq_num < s < cur_r.seq_num for s in sk)
            )
            spk_changed = bool(
                (cur_r.speaker_id and prev_r.speaker_id and cur_r.speaker_id != prev_r.speaker_id)
                or (cur_r.s_sex and prev_r.s_sex and cur_r.s_sex != prev_r.s_sex)
            )
            if non_consec or spk_changed:
                cur_r.segment_break = True

    logger.info(
        "[71379/%s] 파싱 완료: %d개 씬 그룹, %d개 유효 발화 (%.1fs)",
        layout.split_name,
        len(grouped),
        kept_utterances,
        time.perf_counter() - t0,
    )
    return grouped


# ==============================================================================
# 4. VRAM 안전 가드가 포함된 멀티스트림 텐서 생성 및 저장
# ==============================================================================


def _encode_mimi_aligned(
    timeline_24k: torch.Tensor,
    mimi_encode_fn: Callable[[torch.Tensor], torch.Tensor],
    total_frames: int,
    expected_q: int,
) -> torch.Tensor:
    """장시간 파형에 대해서도 VRAM 피크가 일정하도록 최대 45초 단위로 분할 인코딩한 뒤 정렬합니다."""
    max_chunk_frames = 560  # 약 44.8초 단위로 인코딩하여 Conv1d 활성화 VRAM 상한 고정
    max_chunk_samples = max_chunk_frames * SAMPLES_PER_FRAME
    needed_samples = total_frames * SAMPLES_PER_FRAME
    wav_in = timeline_24k[:needed_samples]

    if wav_in.numel() <= max_chunk_samples:
        raw_c = mimi_encode_fn(wav_in.view(1, 1, -1)).detach().cpu().long()
        c = raw_c.squeeze(0) if (raw_c.ndim == 3 and raw_c.shape[0] == 1) else raw_c
    else:
        encoded_parts: List[torch.Tensor] = []
        for s_off in range(0, wav_in.numel(), max_chunk_samples):
            piece = wav_in[s_off : s_off + max_chunk_samples]
            part_c = mimi_encode_fn(piece.view(1, 1, -1)).detach().cpu().long()
            if part_c.ndim == 3 and part_c.shape[0] == 1:
                part_c = part_c.squeeze(0)
            encoded_parts.append(part_c)
        c = torch.cat(encoded_parts, dim=1)

    if c.ndim != 2 or c.shape[0] != expected_q:
        raise ValueError(f"Mimi 코드북 차원 불일치: 기대 [{expected_q}, T], 실제 {tuple(c.shape)}")
    if c.shape[1] >= total_frames:
        return c[:, :total_frames].contiguous()
    pad_len = total_frames - c.shape[1]
    pad = (
        c[:, -1:].expand(expected_q, pad_len).contiguous()
        if c.shape[1] > 0
        else torch.full((expected_q, pad_len), ZERO_TOKEN_ID, dtype=torch.long)
    )
    return torch.cat([c, pad], dim=1)


def _slice_71693_windows(
    spans: List[Span71693],
    target_window_sec: float = 35.0,
    max_gap_sec: float = 8.0,
) -> List[List[Span71693]]:
    windows: List[List[Span71693]] = []
    cur: List[Span71693] = []
    cur_end = 0.0
    for sp in spans:
        if not cur:
            cur = [sp]
            cur_end = sp.end_sec
            continue
        gap = sp.start_sec - cur_end
        win_len = max(cur_end, sp.end_sec) - cur[0].start_sec
        if sp.segment_break or gap > max_gap_sec or win_len > target_window_sec:
            windows.append(cur)
            cur = [sp]
            cur_end = sp.end_sec
        else:
            cur.append(sp)
            cur_end = max(cur_end, sp.end_sec)
    if cur:
        windows.append(cur)
    return windows


def _slice_71379_windows(
    records: List[Utterance71379],
    target_window_sec: float = 35.0,
    max_gap_sec: float = 8.0,
    inter_utt_pause_sec: float = 0.5,
) -> List[List[Utterance71379]]:
    windows: List[List[Utterance71379]] = []
    cur: List[Utterance71379] = []
    cur_est = 0.0
    for rec in records:
        est_dur = max(rec.t_ulength, rec.s_ulength, 2.5)
        if not cur:
            cur = [rec]
            cur_est = est_dur
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
        if rec.segment_break or gap_exceeded or (cur_est + pause + est_dur) > target_window_sec:
            windows.append(cur)
            cur = [rec]
            cur_est = est_dur
        else:
            cur.append(rec)
            cur_est += pause + est_dur
    if cur:
        windows.append(cur)
    return windows


def _extract_prompt_24k_from_file(wav_path: str, spans: List[Span71693]) -> torch.Tensor:
    target = 10 * SAMPLE_RATE
    chunks: List[torch.Tensor] = []
    acc = 0
    for sp in spans[:12]:
        if sp.end_sec - sp.start_sec > 0.5:
            clip = fast_load_mono_wav_24k(wav_path, start_sec=sp.start_sec, end_sec=sp.end_sec)
            if clip.numel() > 0:
                chunks.append(clip)
                acc += clip.numel()
                if acc >= target:
                    break
    if chunks:
        return torch.cat(chunks, dim=0)[:target]
    fallback = fast_load_mono_wav_24k(wav_path, start_sec=0.0, end_sec=10.0)
    return fallback[:target] if fallback.numel() > 0 else torch.zeros(target, dtype=torch.float32)


def _window_already_saved(existing_names: Set[str], stem_prefix: str) -> int:
    exact = f"{stem_prefix}.pt"
    if exact in existing_names:
        return 1
    part_prefix = f"{stem_prefix}_p"
    return sum(1 for name in existing_names if name.startswith(part_prefix) and name.endswith(".pt"))


def _maybe_release_memory(saved_count: int, device: torch.device) -> None:
    if saved_count > 0 and saved_count % 100 == 0:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def process_71693_to_pt(
    grouped: dict[str, List[Span71693]],
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
    """145MB 전체 강의 WAV를 통째로 올리지 않고 윈도우 구간만 부분 Seek 로드하여 RAM을 절약합니다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_index, _ = _fast_scandir_files(wav_dirs, ".wav")
    existing_names = {p.name for p in output_dir.glob("*.pt")} if skip_existing else set()

    saved_count = 0
    t0 = time.perf_counter()
    total_lectures = len(grouped)
    hard_max_frames = cfg.context_frames if cfg.context_frames > 0 else 1000

    for lec_idx, (fname, spans) in enumerate(grouped.items(), start=1):
        if max_samples is not None and saved_count >= max_samples:
            break

        base_name_lower = os.path.basename(fname).lower()
        stem_lower = os.path.splitext(base_name_lower)[0]
        wav_path = wav_index.get(base_name_lower) or wav_index.get(f"{stem_lower}.wav")
        if wav_path is None:
            logger.warning("[71693] 원천 영어 WAV 누락으로 건너뜀: %s", fname)
            continue

        windows = _slice_71693_windows(
            spans, target_window_sec=target_window_sec, max_gap_sec=max_gap_sec
        )
        lec_stem = os.path.splitext(os.path.basename(fname))[0]

        if skip_existing and windows:
            existing_hits = [
                _window_already_saved(existing_names, f"71693_{lec_stem}_w{w_i:04d}")
                for w_i in range(len(windows))
            ]
            if all(h > 0 for h in existing_hits):
                saved_count += sum(existing_hits)
                continue

        need_prompt = wavlm_sim_fn is not None or (
            ko_mode_71693 not in ("text-only", "mms", "edge-tts")
        )
        speaker_prompt = (
            _extract_prompt_24k_from_file(wav_path, spans)
            if need_prompt
            else torch.zeros(SAMPLES_PER_FRAME, dtype=torch.float32)
        )

        for w_i, base_win in enumerate(windows):
            if max_samples is not None and saved_count >= max_samples:
                break

            base_stem = f"71693_{lec_stem}_w{w_i:04d}"
            if skip_existing:
                already = _window_already_saved(existing_names, base_stem)
                if already > 0:
                    saved_count += already
                    continue

            tts_cache: dict[str, Tuple[torch.Tensor, List[int], List[str]]] = {}
            queue: deque[Tuple[str, List[Span71693]]] = deque([(base_stem, base_win)])
            while queue:
                if max_samples is not None and saved_count >= max_samples:
                    break
                cur_stem, win_spans = queue.popleft()
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
                        ko_ids = sp_encode_fn(sp.ko_text)
                        ko_pieces = sp_pieces_fn(sp.ko_text)
                        if ko_mode_71693 == "text-only":
                            est_frames = max(len(ko_ids) + 2, int(max(1, len(sp.ko_text)) * 1.6))
                            ko_wav = torch.zeros(est_frames * SAMPLES_PER_FRAME, dtype=torch.float32)
                        else:
                            assert ko_tts_fn is not None
                            ko_wav = ko_tts_fn(sp.ko_text, speaker_prompt, sp.gender)
                        max_ko_samples = max(SAMPLES_PER_FRAME * 4, (hard_max_frames - 8) * SAMPLES_PER_FRAME)
                        if ko_wav.numel() > max_ko_samples:
                            ko_wav = ko_wav[:max_ko_samples]
                        tts_cache[sp.sn] = (ko_wav, ko_ids, ko_pieces)

                    clamped_end_sec = min(sp.end_sec, win_end_sec)
                    rel_en_s = max(0, int((sp.start_sec - win_start_sec) * SAMPLE_RATE))
                    rel_en_e = max(rel_en_s, int((clamped_end_sec - win_start_sec) * SAMPLE_RATE))
                    en_s_frame = rel_en_s // SAMPLES_PER_FRAME
                    en_e_frame = (rel_en_e + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME
                    prev_ko_end_frame = (ko_cursor_sample + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME

                    eff_align = "uniform" if ko_mode_71693 == "text-only" else align_mode
                    audio_start_sample, tok_pos, last_token_frame = compute_aligned_token_placement(
                        ko_text=sp.ko_text,
                        ko_wav_24k=ko_wav,
                        ko_token_ids=ko_ids,
                        en_end_frame=en_e_frame,
                        prev_audio_end_frame=prev_ko_end_frame,
                        last_token_frame=last_token_frame,
                        align_mode=eff_align,
                        ko_align_fn=None if ko_mode_71693 == "text-only" else ko_align_fn,
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
                eos_frame = max(last_token_frame + 1, audio_end_frames)
                total_frames = max(audio_end_frames + 4, eos_frame + 1)

                if can_split and total_frames > cfg.context_frames:
                    mid = len(win_spans) // 2
                    queue.appendleft((f"{cur_stem}_p1", win_spans[mid:]))
                    queue.appendleft((f"{cur_stem}_p0", win_spans[:mid]))
                    continue

                if total_frames > hard_max_frames:
                    total_frames = hard_max_frames
                    eos_frame = total_frames - 1

                # 해당 윈도우 구간만 디스크에서 부분 Seek 로드 (145MB 전체 강의 로드 회피)
                win_wav_slice = fast_load_mono_wav_24k(
                    wav_path, start_sec=win_start_sec, end_sec=win_end_sec
                )
                if win_wav_slice.numel() == 0:
                    continue

                padded_samples = (total_frames + 2) * SAMPLES_PER_FRAME
                en_timeline = torch.zeros(padded_samples, dtype=torch.float32)
                for sp in win_spans:
                    rel_s = max(0, min(int((sp.start_sec - win_start_sec) * SAMPLE_RATE), win_wav_slice.numel()))
                    rel_e = max(rel_s, min(int((sp.end_sec - win_start_sec) * SAMPLE_RATE), win_wav_slice.numel()))
                    if rel_e > rel_s and rel_e <= padded_samples:
                        en_timeline[rel_s:rel_e] = win_wav_slice[rel_s:rel_e]
                del win_wav_slice

                ko_text_stream = torch.full((1, total_frames), TEXT_PAD_ID, dtype=torch.long)
                for _, _, tok_pos in ko_placements:
                    for f_pos, tid in tok_pos:
                        if 0 <= f_pos < eos_frame:
                            ko_text_stream[0, f_pos] = tid
                ko_text_stream[0, eos_frame] = TEXT_EOS_ID

                en_codes = _encode_mimi_aligned(
                    en_timeline, mimi_encode_fn, total_frames, cfg.num_codebooks_per_stream
                )
                del en_timeline

                if ko_mode_71693 == "text-only":
                    ko_codes = torch.full(
                        (cfg.num_codebooks_per_stream, total_frames),
                        ZERO_TOKEN_ID,
                        dtype=torch.long,
                    )
                    sim_bin = cfg.num_speaker_sim_bins - 1
                else:
                    ko_timeline = torch.zeros(padded_samples, dtype=torch.float32)
                    active_clips: List[torch.Tensor] = []
                    for start_sample, ko_wav, _ in ko_placements:
                        end_s = min(padded_samples, start_sample + ko_wav.numel())
                        if end_s > start_sample:
                            ko_timeline[start_sample:end_s] = ko_wav[: end_s - start_sample]
                            active_clips.append(ko_wav[: end_s - start_sample])
                    ko_codes = _encode_mimi_aligned(
                        ko_timeline, mimi_encode_fn, total_frames, cfg.num_codebooks_per_stream
                    )
                    del ko_timeline
                    if wavlm_sim_fn is not None and active_clips:
                        sim_bin = int(
                            wavlm_sim_fn(speaker_prompt, torch.cat(active_clips, dim=0)[: 10 * SAMPLE_RATE])
                        )
                    elif ko_mode_71693 == "mms":
                        sim_bin = min(1, cfg.num_speaker_sim_bins - 1)
                    else:
                        sim_bin = min(2, cfg.num_speaker_sim_bins - 1)

                codes = torch.cat([ko_text_stream, ko_codes, en_codes], dim=0)
                out_pt = output_dir / f"{cur_stem}.pt"
                atomic_torch_save(
                    {
                        "codes": codes,
                        "sim_bin": int(sim_bin),
                        "src_start_frames": src_start_frames,
                        "src_end_frames": src_end_frames,
                        "ref_token_groups": ref_token_groups,
                        "align_mode": "uniform" if ko_mode_71693 == "text-only" else align_mode,
                        "dataset": "71693",
                        "ko_audio_source": ko_mode_71693,
                    },
                    out_pt,
                )
                existing_names.add(out_pt.name)
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
    grouped: dict[str, List[Utterance71379]],
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
) -> int:
    """파싱된 71379 씬 그룹들을 저메모리·정밀 정렬 `.pt` 파일로 변환하여 저장합니다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[71379] 한/영 원천·발화 WAV 파일 고속 인덱싱 중...")
    t_idx = time.perf_counter()
    ko_wav_index, _ = _fast_scandir_files(ko_wav_dirs, ".wav")
    en_wav_index, _ = _fast_scandir_files(en_wav_dirs, ".wav")
    logger.info(
        "[71379] WAV 인덱싱 완료: KO=%d개, EN=%d개 (%.1fs)",
        len(ko_wav_index),
        len(en_wav_index),
        time.perf_counter() - t_idx,
    )

    existing_names = {p.name for p in output_dir.glob("*.pt")} if skip_existing else set()

    saved_count = 0
    t0 = time.perf_counter()
    ordered_groups = list(grouped.items())
    total_groups = len(ordered_groups)
    max_overlap_samples = max(0, int(max_overlap_sec * SAMPLE_RATE))
    hard_max_frames = cfg.context_frames if cfg.context_frames > 0 else 1000

    for g_idx, (group_key, recs) in enumerate(ordered_groups, start=1):
        if max_samples is not None and saved_count >= max_samples:
            break

        valid_triples: List[Tuple[Utterance71379, str, str]] = []
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
        windows = _slice_71379_windows(
            recs_only,
            target_window_sec=target_window_sec,
            max_gap_sec=max_gap_sec,
            inter_utt_pause_sec=inter_utt_pause_sec,
        )

        if skip_existing and windows:
            hits = [
                _window_already_saved(existing_names, f"71379_{group_key}_w{w_i:04d}")
                for w_i in range(len(windows))
            ]
            if all(h > 0 for h in hits):
                saved_count += sum(hits)
                continue

        for w_i, base_win in enumerate(windows):
            if max_samples is not None and saved_count >= max_samples:
                break

            base_stem = f"71379_{group_key}_w{w_i:04d}"
            if skip_existing:
                already = _window_already_saved(existing_names, base_stem)
                if already > 0:
                    saved_count += already
                    continue

            clip_cache: dict[str, Tuple[torch.Tensor, torch.Tensor, List[int], List[str]]] = {}
            queue: deque[Tuple[str, List[Utterance71379]]] = deque([(base_stem, base_win)])
            while queue:
                if max_samples is not None and saved_count >= max_samples:
                    break
                cur_stem, win_recs = queue.popleft()
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
                        en_wav = fast_load_mono_wav_24k(en_p)
                        ko_wav = fast_load_mono_wav_24k(ko_p)
                        if en_wav.numel() == 0 or ko_wav.numel() == 0:
                            failed_utt = True
                            break
                        ko_ids = sp_encode_fn(rec.ko_text)
                        ko_pieces = sp_pieces_fn(rec.ko_text)
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
                eos_frame = max(last_token_frame + 1, audio_end_frames)
                total_frames = max(audio_end_frames + 4, eos_frame + 1)

                if can_split and total_frames > cfg.context_frames:
                    mid = len(win_recs) // 2
                    queue.appendleft((f"{cur_stem}_p1", win_recs[mid:]))
                    queue.appendleft((f"{cur_stem}_p0", win_recs[:mid]))
                    continue

                if total_frames > hard_max_frames:
                    total_frames = hard_max_frames
                    eos_frame = total_frames - 1

                padded_samples = (total_frames + 2) * SAMPLES_PER_FRAME
                en_timeline = torch.zeros(padded_samples, dtype=torch.float32)
                ko_timeline = torch.zeros(padded_samples, dtype=torch.float32)
                ko_text_stream = torch.full((1, total_frames), TEXT_PAD_ID, dtype=torch.long)

                for start_s, en_w in en_placements:
                    end_s = min(padded_samples, start_s + en_w.numel())
                    if end_s > start_s:
                        en_timeline[start_s:end_s] = en_w[: end_s - start_s]
                for start_s, ko_w, tok_pos in ko_placements:
                    end_s = min(padded_samples, start_s + ko_w.numel())
                    if end_s > start_s:
                        ko_timeline[start_s:end_s] = ko_w[: end_s - start_s]
                    for f_pos, tid in tok_pos:
                        if 0 <= f_pos < eos_frame:
                            ko_text_stream[0, f_pos] = tid
                ko_text_stream[0, eos_frame] = TEXT_EOS_ID

                ko_codes = _encode_mimi_aligned(
                    ko_timeline, mimi_encode_fn, total_frames, cfg.num_codebooks_per_stream
                )
                del ko_timeline
                en_codes = _encode_mimi_aligned(
                    en_timeline, mimi_encode_fn, total_frames, cfg.num_codebooks_per_stream
                )
                del en_timeline

                if wavlm_sim_fn is not None:
                    en_only = torch.cat([w for _, w in en_placements], dim=0)[: 10 * SAMPLE_RATE]
                    ko_only = torch.cat([w for _, w, _ in ko_placements], dim=0)[: 10 * SAMPLE_RATE]
                    sim_bin = int(wavlm_sim_fn(en_only, ko_only))
                else:
                    first_rec = win_recs[0]
                    same_sex = bool(
                        first_rec.s_sex and first_rec.t_sex and first_rec.s_sex == first_rec.t_sex
                    )
                    sim_bin = (
                        cfg.num_speaker_sim_bins - 1
                        if same_sex
                        else max(0, cfg.num_speaker_sim_bins - 2)
                    )

                codes = torch.cat([ko_text_stream, ko_codes, en_codes], dim=0)
                out_pt = output_dir / f"{cur_stem}.pt"
                atomic_torch_save(
                    {
                        "codes": codes,
                        "sim_bin": int(sim_bin),
                        "src_start_frames": src_start_frames,
                        "src_end_frames": src_end_frames,
                        "ref_token_groups": ref_token_groups,
                        "align_mode": align_mode,
                        "dataset": "71379",
                        "ko_audio_source": "real",
                    },
                    out_pt,
                )
                existing_names.add(out_pt.name)
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


# ==============================================================================
# 5. 순차 분할 오케스트레이터 (train 해제 후 val 적재)
# ==============================================================================


def verify_preprocessed_split(
    split_dir: Path,
    cfg: EDarchConfig,
) -> Tuple[int, int]:
    """생성된 `.pt` 디렉터리를 Stage 1(`PreprocessedPtDataset`)과 Stage 2(`GRPOPromptDataset`)로 검증합니다."""
    from libs.grpo import GRPOPromptDataset
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


def _parse_single_split(
    sp_name: str,
    ly93_map: dict[str, SplitLayout71693],
    ly79_map: dict[str, SplitLayout71379],
    args: argparse.Namespace,
) -> Tuple[Optional[dict[str, List[Span71693]]], Optional[dict[str, List[Utterance71379]]]]:
    """메모리 중복 점유를 피하기 위해 요청된 단일 분할(`train` 또는 `val`)만 파싱합니다."""
    g93: Optional[dict[str, List[Span71693]]] = None
    g79: Optional[dict[str, List[Utterance71379]]] = None

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


def run_d_drive_preprocessing(args: argparse.Namespace) -> None:
    """`D:\\data`와 `D:\\data2`를 저메모리 순차 파이프라인으로 전처리하여 `E:\\preprocessed_edarch`에 저장합니다."""
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

    # 첫 번째 분할(train 우선)만 먼저 파싱하여 토크나이저 구축 후 즉시 재사용 (train+val 동시 상주 방지)
    ordered_splits = [s for s in ("train", "val") if s in requested_splits]
    first_split = ordered_splits[0]
    first_g93, first_g79 = _parse_single_split(first_split, ly93_map, ly79_map, args)

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
                    _add_sent(s_item.ko_text)
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
    if use_71693 and args.ko_mode_71693 != "text-only":
        if custom_tts is not None:
            ko_tts_fn = lambda txt, prompt, gender="": custom_tts(txt, prompt)
        elif args.ko_mode_71693 == "mms":
            ko_tts_fn = build_mms_korean_tts_fn(device=device, model_id=args.mms_tts_model)
        elif args.ko_mode_71693 == "edge-tts":
            ko_tts_fn = build_edge_korean_tts_fn(cache_dir=args.output_dir / ".edge_tts_cache")

    effective_align = (
        "energy"
        if (args.align_mode == "auto" and ko_align_fn is None)
        else args.align_mode
    )

    stats: dict[str, dict[str, int]] = {"train": {}, "val": {}}
    for idx_sp, sp_name in enumerate(ordered_splits):
        if idx_sp == 0:
            cur_g93, cur_g79 = first_g93, first_g79
            first_g93, first_g79 = None, None
        else:
            cur_g93, cur_g79 = _parse_single_split(sp_name, ly93_map, ly79_map, args)

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
                args.ko_mode_71693,
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
                ko_mode_71693=args.ko_mode_71693,
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
        "ko_mode_71693": args.ko_mode_71693 if use_71693 else "disabled",
        "saved_samples": stats,
        "verified": verified_counts,
        "commands": {
            "stage1_sft": (
                f'python train.py --preset {args.preset} --ko-vocab-size {cfg.text_card} '
                f'--data-dir "{train_dir}" --grad-checkpointing --batch-size 1 --grad-accum-steps 16 --epochs 10'
            ),
            "stage2_grpo": (
                f'python train_grpo.py --preset {args.preset} --ko-vocab-size {cfg.text_card} '
                f'--data-dir "{train_dir}" --tokenizer "{tok_path}" --grad-checkpointing --group-size 4'
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
    print("[Done] D:\\data & D:\\data2 학습 데이터 전처리 완료!")
    print(f"  - 출력 루트 디렉터리 : {args.output_dir}")
    print(f"  - 한국어 토크나이저  : {tok_path} (vocab_size={cfg.text_card})")
    print(f"  - Train 샘플 수     : {stats.get('train', {})}")
    print(f"  - Val 샘플 수       : {stats.get('val', {})} (경로: {val_dir})")
    print("-" * 78)
    print("  [1] Stage 1 지도학습(SFT) 실행 명령어:")
    print(f"      {manifest['commands']['stage1_sft']}")
    print("  [2] Stage 2 GRPO 강화학습 실행 명령어:")
    print(f"      {manifest['commands']['stage2_grpo']}")
    print("=" * 78)
