from __future__ import annotations

import argparse
import importlib
import json
import logging
import operator
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

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
TEXT_EOS_ID = 2
TEXT_PAD_ID = 3
ZERO_TOKEN_ID = -1
DEFAULT_CONTEXT_FRAMES = 1000

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


def parse_hhmmss(ts: str) -> float:
    h, m, s = ts.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


@dataclass
class UtteranceSpan:
    sn: str
    file_name: str
    start_sec: float
    end_sec: float
    en_text: str
    ko_mtpe: str
    segment_break: bool = False


def load_and_group_71693_json(json_dir: Path) -> dict[str, List[UtteranceSpan]]:
    grouped: dict[str, List[UtteranceSpan]] = defaultdict(list)
    skipped_intervals: dict[str, List[Tuple[float, float]]] = defaultdict(list)
    seen_records: set[tuple[str, str]] = set()
    for jp in sorted(json_dir.rglob("*.json")):
        try:
            payload = json.loads(jp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("JSON 파싱 실패로 건너뜁니다 (%s): %s", jp, exc)
            continue
        for item in payload.get("data", []):
            if item.get("source_language") != "en" or item.get("target_language") != "ko":
                continue
            if not all(k in item for k in ("sn", "file_name", "audio_start", "audio_end")):
                continue
            record_key = (str(item["file_name"]), str(item["sn"]))
            if record_key in seen_records:
                continue
            try:
                start_sec = parse_hhmmss(item["audio_start"])
                end_sec = parse_hhmmss(item["audio_end"])
            except (ValueError, AttributeError) as exc:
                logger.warning("타임스탬프 파싱 실패로 레코드 건너뜀 (sn=%s): %s", item.get("sn"), exc)
                continue
            if end_sec <= start_sec:
                continue
            seen_records.add(record_key)
            ko_text = (item.get("MTPE") or "").strip()
            if not ko_text or ko_text == "-":
                skipped_intervals[item["file_name"]].append((start_sec, end_sec))
                continue
            grouped[item["file_name"]].append(
                UtteranceSpan(
                    sn=str(item["sn"]),
                    file_name=item["file_name"],
                    start_sec=start_sec,
                    end_sec=end_sec,
                    en_text=(item.get("source_cleaned") or "").strip(),
                    ko_mtpe=ko_text,
                )
            )
    for fname, spans in grouped.items():
        spans.sort(key=lambda x: (x.start_sec, x.end_sec))
        sk_list = skipped_intervals.get(fname)
        if sk_list:
            for idx in range(1, len(spans)):
                prev_end = spans[idx - 1].end_sec
                cur_start = spans[idx].start_sec
                if cur_start > prev_end and any(
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
    lecture_wav_24k: torch.Tensor,
    spans: List[UtteranceSpan],
    prompt_sec: float = 10.0,
) -> torch.Tensor:
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
        ko_cursor_sample = 0
        last_token_frame = -1

        for sp in window_spans:
            cached = _tts_tok_cache.get(sp.sn) if _tts_tok_cache is not None else None
            if cached is not None:
                ko_sent_wav, ko_token_ids = cached
            else:
                tts_out = zero_shot_ko_tts_fn(sp.ko_mtpe, speaker_prompt_24k)
                if isinstance(tts_out, tuple):
                    ko_sent_wav, tts_sr = tts_out
                    ko_sent_wav = ko_sent_wav.detach().cpu().float().reshape(-1)
                    if tts_sr != SAMPLE_RATE:
                        ko_sent_wav = _resample_wav(ko_sent_wav, tts_sr, SAMPLE_RATE)
                else:
                    ko_sent_wav = tts_out.detach().cpu().float().reshape(-1)

                ko_token_ids = list(ko_sp_encode_fn(sp.ko_mtpe))
                if _tts_tok_cache is not None:
                    _tts_tok_cache[sp.sn] = (ko_sent_wav, ko_token_ids)

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

            rel_en_end_sample = max(0, int((sp.end_sec - win_start_sec) * SAMPLE_RATE))
            en_end_frame = (rel_en_end_sample + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME
            prev_audio_end_frame = (ko_cursor_sample + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME
            text_start_frame = max(en_end_frame, prev_audio_end_frame, last_token_frame + 1)

            dur_frames = max(1, (ko_sent_wav.numel() + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME)
            num_tokens = len(ko_token_ids)
            lead_frames = max(0, num_tokens - dur_frames)
            audio_start_frame = text_start_frame + lead_frames
            audio_start_sample = audio_start_frame * SAMPLES_PER_FRAME

            token_positions: List[Tuple[int, int]] = []
            for idx, tid in enumerate(ko_token_ids):
                if num_tokens <= dur_frames:
                    ideal_pos = audio_start_frame + (idx * dur_frames) // max(1, num_tokens)
                    f_pos = max(ideal_pos, last_token_frame + 1)
                else:
                    f_pos = text_start_frame + idx
                token_positions.append((f_pos, tid))
                last_token_frame = f_pos

            ko_placements.append((audio_start_sample, ko_sent_wav, token_positions))
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

        def _encode_and_align(timeline: "torch_t.Tensor") -> "torch_t.Tensor":
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

        ko_codes = _encode_and_align(ko_timeline)
        en_codes = _encode_and_align(en_timeline)
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
        raw_sim_bin = wavlm_sim_bin_fn(speaker_prompt_24k, ko_speech_only)
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
        codes = torch.cat([ko_text_stream, ko_codes, en_codes], dim=0)
        return codes, sim_bin


def save_training_example_pt(
    output_path: Path,
    codes: "torch_t.Tensor",
    sim_bin: int,
    max_retries: int = 5,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save({"codes": codes.detach().cpu(), "sim_bin": int(sim_bin)}, tmp_path)
    for attempt in range(max_retries):
        try:
            os.replace(tmp_path, output_path)
            return
        except PermissionError:
            if attempt == max_retries - 1:
                raise
            time.sleep(0.1 * (2**attempt))


def _resolve_wav_path(wav_dir: Path, file_name: str, wav_index: dict[str, Path]) -> Optional[Path]:
    direct = wav_dir / file_name
    if direct.exists():
        return direct
    name_only = Path(file_name).name
    if name_only in wav_index:
        return wav_index[name_only]
    stem_wav = f"{Path(file_name).stem}.wav"
    return wav_index.get(stem_wav)


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

    grouped = load_and_group_71693_json(json_dir)
    if not grouped:
        raise ValueError(f"{json_dir}에서 유효한 en->ko 발화 레코드를 찾지 못했습니다.")

    wav_index = {p.name: p for p in wav_dir.rglob("*.wav")}
    saved_count = 0

    for file_name, spans in sorted(grouped.items()):
        wav_path = _resolve_wav_path(wav_dir, file_name, wav_index)
        if wav_path is None:
            logger.warning("원본 WAV 파일을 찾지 못해 건너뜁니다: %s", file_name)
            continue

        wav_24k = load_lecture_wav_24k(wav_path)
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
            can_split = max_context_frames > 0 and len(win_spans) > 1
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
                _tts_tok_cache=tts_cache,
                _suppress_length_warning=can_split,
            )
            if can_split and codes.shape[1] > max_context_frames:
                mid = len(win_spans) // 2
                logger.info(
                    "윈도우 길이(%d 프레임)가 max_context_frames(%d)를 초과하여 %d+%d개 발화로 분할합니다 (%s).",
                    codes.shape[1],
                    max_context_frames,
                    mid,
                    len(win_spans) - mid,
                    wav_path.name,
                )
                work_queue.appendleft(win_spans[mid:])
                work_queue.appendleft(win_spans[:mid])
                continue

            out_pt = output_dir / f"{stem}_win{win_idx:04d}.pt"
            save_training_example_pt(out_pt, codes, sim_bin)
            win_idx += 1
            saved_count += 1

    return saved_count


def _parse_args() -> argparse.Namespace:
    from env_setup import DEFAULT_CUDA_INDEX_URL, get_default_venv_dir

    parser = argparse.ArgumentParser(description="EDarch 71693 데이터셋 전처리(.pt 생성) 스크립트")
    parser.add_argument("--venv-dir", type=Path, default=get_default_venv_dir())
    parser.add_argument("--no-venv", action="store_true")
    parser.add_argument("--torch-index-url", type=str, default=DEFAULT_CUDA_INDEX_URL)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--reinstall", action="store_true")

    parser.add_argument("--json-dir", type=Path, required=True, help="71693 JSON 라벨 디렉터리")
    parser.add_argument("--wav-dir", type=Path, required=True, help="원본 영어 강의 WAV 디렉터리")
    parser.add_argument("--output-dir", type=Path, required=True, help="전처리된 .pt 저장 디렉터리")
    parser.add_argument(
        "--preset",
        choices=("1b", "2b"),
        default="1b",
        help="모델 프리셋 (기본값: 1b; audio_card=2048, context_frames=1000)",
    )
    parser.add_argument("--ko-vocab-size", type=int, default=None)
    parser.add_argument("--target-window-sec", type=float, default=45.0)
    parser.add_argument("--max-gap-sec", type=float, default=8.0)
    parser.add_argument(
        "--hooks-module",
        type=str,
        default=None,
        help="zero_shot_ko_tts_fn, ko_sp_encode_fn 등을 제공하는 파이썬 모듈 이름",
    )
    parser.add_argument("--sp-model", type=Path, default=None, help="SentencePiece .model 파일 경로")
    parser.add_argument(
        "--mimi-model",
        type=str,
        default=None,
        help="사전학습 Mimi 코덱 ID 또는 로컬 경로 (예: kyutai/mimi)",
    )
    parser.add_argument(
        "--demo-hooks",
        action="store_true",
        help="외부 TTS/토크나이저 없이 파이프라인 검증용 합성 훅으로 실행",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _run_cli(args: argparse.Namespace) -> None:
    global torch, torchaudio
    import torch as _torch
    import torchaudio as _torchaudio

    torch = _torch
    torchaudio = _torchaudio

    from env_setup import setup_device
    from EDarch_from_scratch import EDarchConfig, ExternalMimiCodec, MiniMimiCodec

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    device = setup_device(cpu_only=args.cpu, seed=args.seed, tag="Preprocess")
    cfg = EDarchConfig.from_preset(args.preset)
    if args.ko_vocab_size is not None:
        cfg.text_card = args.ko_vocab_size

    hooks_mod = importlib.import_module(args.hooks_module) if args.hooks_module else None

    tts_fn = getattr(hooks_mod, "zero_shot_ko_tts_fn", None) if hooks_mod else None
    sp_fn = getattr(hooks_mod, "ko_sp_encode_fn", None) if hooks_mod else None
    mimi_fn = getattr(hooks_mod, "mimi_encode_fn", None) if hooks_mod else None
    sim_fn = getattr(hooks_mod, "wavlm_sim_bin_fn", None) if hooks_mod else None

    if sp_fn is None and args.sp_model is not None:
        if not args.sp_model.exists():
            raise FileNotFoundError(f"SentencePiece 모델 파일을 찾을 수 없습니다: {args.sp_model}")
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

    if mimi_fn is None and args.mimi_model is not None:
        ext_codec = ExternalMimiCodec(cfg, model_id=args.mimi_model).to(device).eval()
        mimi_fn = lambda wav: ext_codec.encode(wav.to(device))

    if args.demo_hooks:
        if tts_fn is None:
            tts_fn = lambda text, prompt: torch.zeros(
                max(SAMPLES_PER_FRAME, len(text) * (SAMPLES_PER_FRAME // 2)),
                dtype=torch.float32,
            )
        if sp_fn is None:
            usable = max(1, cfg.text_card - 4)
            sp_fn = lambda text: [4 + (ord(ch) % usable) for ch in text if not ch.isspace()]
        if mimi_fn is None:
            demo_codec = MiniMimiCodec(cfg).to(device).eval()
            mimi_fn = lambda wav: demo_codec.encode(wav.to(device))
        if sim_fn is None:
            sim_fn = lambda prompt, gen: cfg.num_speaker_sim_bins - 1

    if sim_fn is None:
        logger.warning(
            "wavlm_sim_bin_fn이 지정되지 않아 모든 샘플의 sim_bin이 상수(%d = very_good)로 고정됩니다. "
            "이 데이터로 학습 시 화자 유사도 CFG 조건 분별력이 비활성화되므로 "
            "--hooks-module에 wavlm_sim_bin_fn을 제공하거나 학습/추론 시 --cfg-dropout 0.0 / --cfg-coef 1.0 사용을 권장합니다.",
            cfg.num_speaker_sim_bins - 1,
        )
        sim_fn = lambda prompt, gen: cfg.num_speaker_sim_bins - 1

    missing = [
        name
        for name, fn in (
            ("zero_shot_ko_tts_fn", tts_fn),
            ("ko_sp_encode_fn", sp_fn),
            ("mimi_encode_fn", mimi_fn),
        )
        if fn is None
    ]
    if missing:
        raise ValueError(
            f"필수 전처리 훅이 지정되지 않았습니다: {', '.join(missing)}. "
            "--hooks-module(--sp-model, --mimi-model)을 지정하거나 데모 검증 시 --demo-hooks를 사용하세요."
        )

    total_saved = preprocess_71693_dataset(
        json_dir=args.json_dir,
        wav_dir=args.wav_dir,
        output_dir=args.output_dir,
        zero_shot_ko_tts_fn=tts_fn,
        ko_sp_encode_fn=sp_fn,
        mimi_encode_fn=mimi_fn,
        wavlm_sim_bin_fn=sim_fn,
        target_window_sec=args.target_window_sec,
        max_gap_sec=args.max_gap_sec,
        expected_q=cfg.num_codebooks_per_stream,
        max_context_frames=cfg.context_frames,
        text_card=cfg.text_card,
        num_speaker_sim_bins=cfg.num_speaker_sim_bins,
    )
    print(f"[Done] 전처리 완료: 총 {total_saved}개 .pt 샘플 저장됨 -> {args.output_dir}")


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
