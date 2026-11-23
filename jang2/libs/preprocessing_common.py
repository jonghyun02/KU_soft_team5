"""AI-Hub 71693·71379 전처리 파이프라인들이 공유하는 데이터 구조, 오디오/JSON I/O, TTS 및 타임라인 조립 모듈."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import logging
import operator
import os
from pathlib import Path
import re
import time
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import torch

from libs.codec import (
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
    load_mono_wav_24k,
    resample_wav,
)
from libs.env import atomic_torch_save
from libs.tokenizer import (
    TEXT_EOS_ID,
    TEXT_PAD_ID,
    clean_transcript_text,
    normalize_korean_text,
)

logger = logging.getLogger(__name__)

ZERO_TOKEN_ID = -1
DEFAULT_CONTEXT_FRAMES = 1000

_SENT_SPLIT_RE = re.compile(r"(?<=[.?!,;])\s+")
_SPLIT_TREE_STEM_RE = re.compile(r"^(.*_w\d{4})((?:_(?:p0|p1))+)$")
_BRANCH_TOKEN_RE = re.compile(r"_(?:p0|p1)")

T_In = TypeVar("T_In")
T_Out = TypeVar("T_Out")


def batched_pool_map(
    fn: Callable[[T_In], T_Out],
    items: Sequence[T_In],
    num_workers: int = 16,
    batch_size: int = 4096,
) -> Iterator[T_Out]:
    """대량 작업의 Future가 한꺼번에 메모리에 쌓이지 않도록 고정 배치 단위로 스트리밍 실행합니다."""
    total = len(items)
    if total == 0:
        return
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
        for start in range(0, total, batch_size):
            chunk = items[start : start + batch_size]
            for out in pool.map(fn, chunk):
                yield out


def interleave_lists(groups: Sequence[Sequence[Any]]) -> List[Any]:
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


def fast_scandir_files(
    dirs: Iterable[Optional[Path]],
    suffix: str,
) -> Tuple[dict[str, str], List[str]]:
    """Path 객체 오버헤드를 줄이고 문자열 경로로 대용량 디렉터리를 고속 인덱싱합니다."""
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


def split_long_korean_text(text: str, max_chars: int = 140) -> List[str]:
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


def normalize_tts_output(
    tts_out: Union[torch.Tensor, Tuple[torch.Tensor, int]],
) -> torch.Tensor:
    """TTS 함수 출력을 24kHz 1차원 CPU Float32 파형 텐서로 정규화합니다."""
    if isinstance(tts_out, tuple):
        wav_out, tts_sr = tts_out
        wav_out = wav_out.detach().cpu().float().reshape(-1)
        if tts_sr != SAMPLE_RATE:
            wav_out = resample_wav(wav_out, int(tts_sr), SAMPLE_RATE)
        return wav_out
    return tts_out.detach().cpu().float().reshape(-1)


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
        for sub_txt in split_long_korean_text(ko_text, max_chars=140):
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


@dataclass(slots=True)
class UtteranceSpan:
    """AI-Hub 71693 학술대회 발화 단위의 시간 구간 및 영-한 병렬 텍스트 데이터클래스입니다."""

    sn: str
    file_name: str
    start_sec: float
    end_sec: float
    en_text: str
    ko_mtpe: str
    gender: str = ""
    segment_break: bool = False
    direction: str = "en2ko"


@dataclass(slots=True)
class BroadcastUtterance71379:
    """AI-Hub 71379 방송콘텐츠 통번역 발화 단위의 메타데이터와 파일 정보 데이터클래스입니다."""

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


def parse_71379_scene_and_seq(
    ko_wav_name: str,
    en_wav_name: str,
    lower_item: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, int]:
    """71379 음성 파일명 또는 분류 필드로부터 방송 씬 식별 키와 발화 순번을 추출합니다."""
    base_name = ko_wav_name or en_wav_name
    stem = os.path.splitext(os.path.basename(base_name))[0]
    for suffix in ("_en-us", "_en-gb", "_en_us", "_en_gb", "_en", "_uk"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    parts = stem.split("_")
    if len(parts) >= 6 and parts[5].isdigit():
        return "_".join(parts[:5]), int(parts[5])
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1]), int(parts[-1])

    if lower_item is not None:
        cat = clean_transcript_text(lower_item.get("대분류")) or "bc"
        broadcaster = clean_transcript_text(lower_item.get("중분류")) or "ch"
        prog = clean_transcript_text(lower_item.get("소분류")) or "prog"
        ts_num = clean_transcript_text(lower_item.get("tsnum")) or "0"
        seg_raw = clean_transcript_text(lower_item.get("segnum")) or clean_transcript_text(
            lower_item.get("순서")
        )
        seq_digits = "".join(ch for ch in seg_raw if ch.isdigit())
        seq_num = int(seq_digits) if seq_digits else 0
        return f"{cat}_{broadcaster}_{prog}_{ts_num}", seq_num

    return stem, 0


def mark_71693_segment_breaks(
    spans: List[UtteranceSpan],
    skipped_intervals: Optional[Sequence[Tuple[float, float]]] = None,
) -> None:
    """71693 발화 목록을 시간순 정렬하고 미번역 누락 구간이나 언어 방향 전환 지점에 단절 플래그를 설정합니다."""
    spans.sort(key=lambda x: (x.start_sec, x.end_sec))
    for idx in range(1, len(spans)):
        prev_end = spans[idx - 1].end_sec
        cur_start = spans[idx].start_sec
        if spans[idx].direction != spans[idx - 1].direction:
            spans[idx].segment_break = True
        elif skipped_intervals and cur_start > prev_end and any(
            sk_e > prev_end and sk_s < cur_start for sk_s, sk_e in skipped_intervals
        ):
            spans[idx].segment_break = True


def mark_71379_segment_breaks(
    recs: List[BroadcastUtterance71379],
    skipped_seqs: Optional[Set[int]] = None,
) -> None:
    """71379 발화 목록을 순번 정렬하고 누락 순번·원화자 변경·영어 성우 성별 변경 시 단절 플래그를 설정합니다."""
    recs.sort(key=lambda r: (r.seq_num, r.utt_id))
    sk = skipped_seqs or set()
    for idx in range(1, len(recs)):
        prev_r = recs[idx - 1]
        cur_r = recs[idx]
        non_consecutive = (
            cur_r.seq_num <= prev_r.seq_num
            or (cur_r.seq_num - prev_r.seq_num > 1)
            or any(prev_r.seq_num < s < cur_r.seq_num for s in sk)
        )
        speaker_changed = bool(
            (cur_r.speaker_id and prev_r.speaker_id and cur_r.speaker_id != prev_r.speaker_id)
            or (cur_r.s_sex and prev_r.s_sex and cur_r.s_sex != prev_r.s_sex)
            or (cur_r.t_sex and prev_r.t_sex and cur_r.t_sex != prev_r.t_sex)
        )
        if non_consecutive or speaker_changed:
            cur_r.segment_break = True


def slice_71693_windows(
    spans: Sequence[UtteranceSpan],
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


def slice_71379_windows(
    records: Sequence[BroadcastUtterance71379],
    target_window_sec: float = 45.0,
    max_gap_sec: float = 8.0,
    inter_utt_pause_sec: float = 0.5,
    min_est_dur_sec: float = 2.5,
) -> List[List[BroadcastUtterance71379]]:
    """71379 발화 시퀀스를 화자 변경, 순번 단절 및 목표 길이에 따라 스트리밍 윈도우로 분할합니다."""
    windows: List[List[BroadcastUtterance71379]] = []
    cur: List[BroadcastUtterance71379] = []
    cur_est_sec = 0.0

    for rec in records:
        est_dur = max(rec.t_ulength, rec.s_ulength, min_est_dur_sec)
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


def extract_prompt_24k_from_file(
    wav_path: Path | str,
    spans: Sequence[UtteranceSpan],
    prompt_sec: float = 10.0,
) -> torch.Tensor:
    """대용량 강의 WAV 전체를 읽지 않고 앞부분 발화 구간만 부분 Seek하여 참조 화자 음성을 추출합니다."""
    target = int(prompt_sec * SAMPLE_RATE)
    chunks: List[torch.Tensor] = []
    acc = 0
    for sp in spans[:12]:
        if sp.end_sec - sp.start_sec > 0.5:
            clip = load_mono_wav_24k(wav_path, start_sec=sp.start_sec, end_sec=sp.end_sec)
            if clip.numel() > 0:
                chunks.append(clip)
                acc += clip.numel()
                if acc >= target:
                    break
    if chunks:
        return torch.cat(chunks, dim=0)[:target]
    fallback = load_mono_wav_24k(wav_path, start_sec=0.0, end_sec=prompt_sec)
    return fallback[:target] if fallback.numel() > 0 else torch.zeros(target, dtype=torch.float32)


def validate_ko_token_ids(ko_token_ids: Sequence[int], text_card: Optional[int]) -> None:
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


def validate_sim_bin(raw_sim_bin: Any, num_speaker_sim_bins: int) -> int:
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


def resolve_default_sim_bin(
    audio_source: str,
    num_speaker_sim_bins: int = 5,
    same_sex: bool = False,
    standalone_real: bool = False,
    explicit_sim_bin: Optional[int] = None,
) -> int:
    """WavLM 미사용 환경에서 타깃 음성 생성 방식에 따른 일관된 화자 유사도 구간(0..num_bins-1)을 결정합니다.

    - `cloned_tts` / `text-only`: 최상위 구간 (`num_bins - 1`, 예: 4 = very_good)
    - `edge-tts`: 성별 매칭 신경망 TTS (`min(2, num_bins - 1)`, neutral)
    - `mms`: 고정 단일 화자 VITS (`min(1, num_bins - 1)`, bad)
    - `real_pair`: 71379 이종 화자 원천 쌍. 단독 학습(`standalone_real=True`) 시 추론 기본값 정합을 위해
      `num_bins - 1`을 부여하고, 혼합 학습 시 성별 일치 여부에 따라 `1`(bad) 또는 `0`(very_bad)을 부여합니다.
    """
    if explicit_sim_bin is not None:
        return validate_sim_bin(explicit_sim_bin, num_speaker_sim_bins)

    top_bin = max(0, num_speaker_sim_bins - 1)
    if audio_source in ("cloned_tts", "text-only"):
        return top_bin
    if audio_source == "edge-tts":
        return min(2, top_bin)
    if audio_source == "mms":
        return min(1, top_bin)
    if audio_source == "real_pair":
        if standalone_real:
            return top_bin
        return min(1 if same_sex else 0, top_bin)
    return top_bin


def encode_mimi_timeline(
    timeline_24k: torch.Tensor,
    mimi_encode_fn: Callable[[torch.Tensor], torch.Tensor],
    total_frames: int,
    expected_q: Optional[int] = None,
    max_chunk_frames: int = 1024,
    overlap_frames: int = 64,
) -> torch.Tensor:
    """오디오 타임라인을 Mimi 코덱으로 인코딩하되, 초장문 파형은 인과 오버랩을 두고 안전하게 분할 인코딩합니다."""
    wav_flat = timeline_24k.reshape(-1)
    max_needed_samples = (total_frames + 2) * SAMPLES_PER_FRAME
    wav_in = wav_flat[:max_needed_samples]
    max_chunk_samples = max(1, max_chunk_frames) * SAMPLES_PER_FRAME

    if wav_in.numel() <= max_chunk_samples:
        raw_c = mimi_encode_fn(wav_in.view(1, 1, -1)).detach().cpu().long()
        c = raw_c.squeeze(0) if (raw_c.ndim == 3 and raw_c.shape[0] == 1) else raw_c
    else:
        encoded_parts: List[torch.Tensor] = []
        in_frames_ceil = (wav_in.numel() + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME
        for frame_off in range(0, in_frames_ceil, max_chunk_frames):
            ctx_frames = min(max(0, overlap_frames), frame_off)
            ctx_start_sample = (frame_off - ctx_frames) * SAMPLES_PER_FRAME
            chunk_end_sample = min(
                wav_in.numel(), (frame_off + max_chunk_frames) * SAMPLES_PER_FRAME
            )
            if chunk_end_sample <= ctx_start_sample:
                break
            piece = wav_in[ctx_start_sample:chunk_end_sample]
            part_raw = mimi_encode_fn(piece.view(1, 1, -1)).detach().cpu().long()
            part_c = (
                part_raw.squeeze(0)
                if (part_raw.ndim == 3 and part_raw.shape[0] == 1)
                else part_raw
            )
            if ctx_frames > 0:
                part_c = part_c[:, min(ctx_frames, part_c.shape[1]) :]
            expected_chunk_len = min(max_chunk_frames, in_frames_ceil - frame_off)
            encoded_parts.append(part_c[:, :expected_chunk_len])
        c = torch.cat(encoded_parts, dim=1)

    if c.ndim != 2:
        raise ValueError(
            f"mimi_encode_fn 출력은 [1, Q, T] 또는 [Q, T]여야 합니다. 실제 shape={tuple(c.shape)}"
        )
    if expected_q is not None and c.shape[0] != expected_q:
        raise ValueError(f"Mimi 코드북 수 불일치: 기대 Q={expected_q}, 실제 {c.shape[0]}")

    if c.shape[1] >= total_frames:
        return c[:, :total_frames].contiguous()
    pad_len = total_frames - c.shape[1]
    if c.shape[1] > 0:
        pad = c[:, -1:].expand(c.shape[0], pad_len).contiguous()
    else:
        pad = torch.full((c.shape[0], pad_len), ZERO_TOKEN_ID, dtype=torch.long)
    return torch.cat([c, pad], dim=1)


def assemble_multistream_example(
    en_placements: Sequence[Tuple[int, torch.Tensor]],
    ko_placements: Sequence[Tuple[int, torch.Tensor, Sequence[Tuple[int, int]]]],
    en_span_end_sample: int,
    ko_end_sample: int,
    last_token_frame: int,
    src_start_frames: Sequence[int],
    src_end_frames: Sequence[int],
    ref_token_groups: Sequence[Sequence[int]],
    mimi_encode_fn: Callable[[torch.Tensor], torch.Tensor],
    expected_q: Optional[int],
    align_mode: str,
    mask_target_audio: bool = False,
    hard_max_frames: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """배치된 한/영 오디오 조각과 토큰 타임스탬프로부터 `[1 + 2Q, T]` 코드 텐서와 정합된 메타데이터를 생성합니다."""
    audio_end_frames = (
        max(en_span_end_sample, ko_end_sample) + SAMPLES_PER_FRAME - 1
    ) // SAMPLES_PER_FRAME
    eos_frame = max(last_token_frame + 1, audio_end_frames)
    total_frames = max(audio_end_frames + 4, eos_frame + 1)

    was_clamped = False
    if hard_max_frames is not None and hard_max_frames > 0 and total_frames > hard_max_frames:
        total_frames = hard_max_frames
        eos_frame = total_frames - 1
        was_clamped = True

    if was_clamped:
        final_starts: List[int] = []
        final_ends: List[int] = []
        final_groups: List[List[int]] = []
        for idx, (s_f, e_f) in enumerate(zip(src_start_frames, src_end_frames)):
            if s_f >= total_frames:
                break
            final_starts.append(int(s_f))
            final_ends.append(int(min(e_f, total_frames)))
            if idx < len(ko_placements):
                kept_toks = [
                    int(tid)
                    for f_pos, tid in ko_placements[idx][2]
                    if 0 <= f_pos < eos_frame
                ]
                final_groups.append(kept_toks)
            elif idx < len(ref_token_groups):
                final_groups.append([int(t) for t in ref_token_groups[idx]])
    else:
        final_starts = [int(x) for x in src_start_frames]
        final_ends = [int(x) for x in src_end_frames]
        final_groups = [[int(t) for t in grp] for grp in ref_token_groups]

    padded_samples = (total_frames + 2) * SAMPLES_PER_FRAME
    en_timeline = torch.zeros(padded_samples, dtype=torch.float32)
    for start_s, en_w in en_placements:
        if start_s < 0 or start_s >= padded_samples:
            continue
        end_s = min(padded_samples, start_s + en_w.numel())
        if end_s > start_s:
            en_timeline[start_s:end_s] = en_w[: end_s - start_s]

    ko_text_stream = torch.full((1, total_frames), TEXT_PAD_ID, dtype=torch.long)
    for _, _, tok_pos in ko_placements:
        for f_pos, tid in tok_pos:
            if 0 <= f_pos < eos_frame:
                ko_text_stream[0, f_pos] = int(tid)
    ko_text_stream[0, eos_frame] = TEXT_EOS_ID

    en_codes = encode_mimi_timeline(en_timeline, mimi_encode_fn, total_frames, expected_q)
    del en_timeline

    if mask_target_audio:
        q_out = expected_q if expected_q is not None else int(en_codes.shape[0])
        ko_codes = torch.full((q_out, total_frames), ZERO_TOKEN_ID, dtype=torch.long)
        ko_speech_only = torch.zeros(0, dtype=torch.float32)
    else:
        ko_timeline = torch.zeros(padded_samples, dtype=torch.float32)
        active_ko_clips: List[torch.Tensor] = []
        for start_s, ko_w, _ in ko_placements:
            if start_s < 0 or start_s >= padded_samples:
                continue
            end_s = min(padded_samples, start_s + ko_w.numel())
            if end_s > start_s:
                clipped = ko_w[: end_s - start_s]
                ko_timeline[start_s:end_s] = clipped
                if clipped.numel() > 0:
                    active_ko_clips.append(clipped)
        ko_codes = encode_mimi_timeline(ko_timeline, mimi_encode_fn, total_frames, expected_q)
        ko_speech_only = (
            torch.cat(active_ko_clips, dim=0)
            if active_ko_clips
            else ko_timeline[: total_frames * SAMPLES_PER_FRAME].clone()
        )
        del ko_timeline
        if ko_codes.shape[0] != en_codes.shape[0]:
            raise ValueError(
                f"타깃/소스 코드북 수 불일치: ko={ko_codes.shape[0]}, en={en_codes.shape[0]}"
            )

    codes = torch.cat([ko_text_stream, ko_codes, en_codes], dim=0)
    metadata: Dict[str, Any] = {
        "src_start_frames": final_starts,
        "src_end_frames": final_ends,
        "ref_token_groups": final_groups,
        "align_mode": "uniform" if mask_target_audio else align_mode,
    }
    return codes, ko_speech_only, metadata


def save_training_example_pt(
    output_path: Path,
    codes: torch.Tensor,
    sim_bin: int,
    metadata: Optional[Mapping[str, Any]] = None,
    max_retries: int = 5,
) -> None:
    """멀티스트림 코드 텐서, 화자 유사도 구간 및 Stage 2 메타데이터를 원자적으로 `.pt`에 저장합니다."""
    payload: Dict[str, Any] = {"codes": codes.detach().cpu(), "sim_bin": int(sim_bin)}
    if metadata:
        for k, v in metadata.items():
            if k not in payload:
                payload[k] = v
    atomic_torch_save(payload, output_path, max_retries=max_retries)


def index_existing_window_trees(output_dir: Path) -> Tuple[Set[str], Set[str]]:
    """출력 디렉터리 내 기존 `.pt` 스템 집합과 이진 분할 상위 조상 스템 집합을 인덱싱합니다."""
    existing_stems: Set[str] = set()
    split_ancestors: Set[str] = set()
    if not output_dir.exists():
        return existing_stems, split_ancestors
    for p in output_dir.glob("*.pt"):
        if not p.is_file():
            continue
        stem = p.stem
        existing_stems.add(stem)
        m = _SPLIT_TREE_STEM_RE.match(stem)
        if m is not None:
            root_base, branch_sfx = m.group(1), m.group(2)
            split_ancestors.add(root_base)
            tokens = _BRANCH_TOKEN_RE.findall(branch_sfx)
            acc = root_base
            for tok in tokens[:-1]:
                acc = f"{acc}{tok}"
                split_ancestors.add(acc)
    return existing_stems, split_ancestors


def count_completed_window_tree(
    base_stem: str,
    num_items: int,
    existing_stems: Set[str],
    branch_tags: Tuple[str, str] = ("_p0", "_p1"),
) -> Optional[int]:
    """윈도우 또는 이진 분할 하위 트리가 모두 저장되었으면 리프 개수를, 일부라도 누락되었으면 None을 반환합니다."""
    if base_stem in existing_stems:
        return 1
    if num_items <= 1:
        return None
    mid = num_items // 2
    left_tag, right_tag = branch_tags
    left_cnt = count_completed_window_tree(
        f"{base_stem}{left_tag}", mid, existing_stems, branch_tags=branch_tags
    )
    if left_cnt is None:
        return None
    right_cnt = count_completed_window_tree(
        f"{base_stem}{right_tag}", num_items - mid, existing_stems, branch_tags=branch_tags
    )
    if right_cnt is None:
        return None
    return left_cnt + right_cnt


__all__ = [
    "ZERO_TOKEN_ID",
    "DEFAULT_CONTEXT_FRAMES",
    "UtteranceSpan",
    "BroadcastUtterance71379",
    "batched_pool_map",
    "interleave_lists",
    "fast_scandir_files",
    "romanize_korean_for_mms",
    "split_long_korean_text",
    "normalize_tts_output",
    "build_mms_korean_tts_fn",
    "build_edge_korean_tts_fn",
    "parse_71379_scene_and_seq",
    "mark_71693_segment_breaks",
    "mark_71379_segment_breaks",
    "slice_71693_windows",
    "slice_71379_windows",
    "extract_prompt_24k_from_file",
    "validate_ko_token_ids",
    "validate_sim_bin",
    "resolve_default_sim_bin",
    "encode_mimi_timeline",
    "assemble_multistream_example",
    "save_training_example_pt",
    "index_existing_window_trees",
    "count_completed_window_tree",
]
