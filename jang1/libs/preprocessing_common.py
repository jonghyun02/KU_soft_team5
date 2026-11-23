"""arXiv:2309.11093 노래 번역 전처리 파이프라인이 공유하는 섹션·행 구조체, 타임라인 조립 및 원자적 저장 모듈."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
import operator
import os
from pathlib import Path
import re
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
)

import torch

from libs.codec import (
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
    load_mono_wav_24k,
)
from libs.env import atomic_torch_save
from libs.phonetics import (
    align_lyric_tokens_to_frames,
    compute_phoneme_ssm,
    compute_repetition_group_ids,
    compute_semantic_ssm,
    count_line_syllables,
    detect_vocables_and_codeswitch,
    extract_end_rhyme_phonemes,
    normalize_lyric_text,
    resolve_section_type_id,
    synthesize_rhythm_guide_waveform,
)
from libs.tokenizer import (
    TEXT_EOS_ID,
    TEXT_PAD_ID,
    render_section_with_controls,
)

logger = logging.getLogger(__name__)

ZERO_TOKEN_ID = -1
DEFAULT_CONTEXT_FRAMES = 1000

_SPLIT_TREE_STEM_RE = re.compile(r"^(.*_s\d{4})((?:_(?:p0|p1))+)$")
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
    """여러 도메인·장르 목록을 라운드로빈으로 고르게 교차 배치합니다."""
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


@dataclass(slots=True)
class LyricLinePair:
    """노래 한 행(Line)의 원문·번역 가사, 음절 수, 반복 그룹 ID, 각운 및 타임스탬프 구조체입니다."""

    src_text: str
    tgt_text: str
    src_syllables: int
    tgt_syllables: int
    repeat_group_id: int = 0
    src_rhyme: Tuple[str, ...] = ()
    tgt_rhyme: Tuple[str, ...] = ()
    vocables: Tuple[str, ...] = ()
    is_code_switched: bool = False
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None


@dataclass
class LyricSectionUnit:
    """노래 한 섹션(Verse, Chorus, Bridge 등)의 계층 정렬된 행 목록과 자기유사도 행렬(SSM) 구조체입니다."""

    song_id: str
    section_idx: int
    section_type: str
    section_type_id: int
    direction: str
    lines: List[LyricLinePair]
    src_semantic_ssm: List[List[float]] = field(default_factory=list)
    src_phoneme_ssm: List[List[float]] = field(default_factory=list)
    tgt_phoneme_ssm: List[List[float]] = field(default_factory=list)
    src_wav_name: str = ""
    tgt_wav_name: str = ""


def build_lyric_section_unit(
    song_id: str,
    section_idx: int,
    section_type: str,
    src_lines: Sequence[str],
    tgt_lines: Sequence[str],
    direction: str = "ko2en",
    src_wav_name: str = "",
    tgt_wav_name: str = "",
    line_timestamps: Optional[Sequence[Tuple[float, float]]] = None,
) -> Optional[LyricSectionUnit]:
    """원문·번역 행 리스트로부터 음절 수, 반복 그룹, 각운 및 의미·음소 SSM이 계산된 LyricSectionUnit을 생성합니다."""
    paired_src: List[str] = []
    paired_tgt: List[str] = []
    paired_ts: List[Optional[Tuple[float, float]]] = []

    for idx, (s_raw, t_raw) in enumerate(zip(src_lines, tgt_lines)):
        s_norm = normalize_lyric_text(s_raw)
        t_norm = normalize_lyric_text(t_raw)
        if not s_norm or not t_norm:
            continue
        paired_src.append(s_norm)
        paired_tgt.append(t_norm)
        if line_timestamps is not None and idx < len(line_timestamps):
            paired_ts.append(line_timestamps[idx])
        else:
            paired_ts.append(None)

    if not paired_src:
        return None

    rep_ids = compute_repetition_group_ids(paired_src)
    line_objs: List[LyricLinePair] = []
    for idx, (s_txt, t_txt) in enumerate(zip(paired_src, paired_tgt)):
        s_syl = max(1, count_line_syllables(s_txt))
        t_syl = max(1, count_line_syllables(t_txt))
        cs_info = detect_vocables_and_codeswitch(s_txt)
        ts_pair = paired_ts[idx]
        line_objs.append(
            LyricLinePair(
                src_text=s_txt,
                tgt_text=t_txt,
                src_syllables=s_syl,
                tgt_syllables=t_syl,
                repeat_group_id=rep_ids[idx],
                src_rhyme=extract_end_rhyme_phonemes(s_txt),
                tgt_rhyme=extract_end_rhyme_phonemes(t_txt),
                vocables=tuple(cs_info.vocables),
                is_code_switched=cs_info.is_code_switched,
                start_sec=ts_pair[0] if ts_pair is not None else None,
                end_sec=ts_pair[1] if ts_pair is not None else None,
            )
        )

    sec_type_id = resolve_section_type_id(section_type)
    return LyricSectionUnit(
        song_id=song_id,
        section_idx=section_idx,
        section_type=section_type,
        section_type_id=sec_type_id,
        direction=direction,
        lines=line_objs,
        src_semantic_ssm=compute_semantic_ssm(paired_src),
        src_phoneme_ssm=compute_phoneme_ssm(paired_src),
        tgt_phoneme_ssm=compute_phoneme_ssm(paired_tgt),
        src_wav_name=src_wav_name,
        tgt_wav_name=tgt_wav_name,
    )


def slice_long_section_unit(
    unit: LyricSectionUnit,
    max_lines: int = 12,
) -> List[LyricSectionUnit]:
    """단일 섹션의 행 수가 max_lines를 초과할 경우 문맥을 유지하는 하위 섹션 단위들로 분할합니다."""
    if max_lines <= 0 or len(unit.lines) <= max_lines:
        return [unit]
    chunks: List[LyricSectionUnit] = []
    for start in range(0, len(unit.lines), max_lines):
        sub_lines = unit.lines[start : start + max_lines]
        s_txts = [ln.src_text for ln in sub_lines]
        t_txts = [ln.tgt_text for ln in sub_lines]
        ts_list = [
            (ln.start_sec, ln.end_sec)
            if (ln.start_sec is not None and ln.end_sec is not None)
            else (0.0, 0.0)
            for ln in sub_lines
        ]
        has_ts = all(ln.start_sec is not None and ln.end_sec is not None for ln in sub_lines)
        sub_unit = build_lyric_section_unit(
            song_id=unit.song_id,
            section_idx=unit.section_idx * 100 + (start // max_lines),
            section_type=unit.section_type,
            src_lines=s_txts,
            tgt_lines=t_txts,
            direction=unit.direction,
            src_wav_name=unit.src_wav_name,
            tgt_wav_name=unit.tgt_wav_name,
            line_timestamps=ts_list if has_ts else None,
        )
        if sub_unit is not None:
            chunks.append(sub_unit)
    return chunks


def validate_token_ids(token_ids: Sequence[int], text_card: Optional[int]) -> None:
    """인코딩된 가사 토큰 ID들이 허용 어휘 범위에 속하며 특수 예약 토큰을 침범하지 않는지 검증합니다."""
    if any(
        tid < 0
        or tid in (TEXT_EOS_ID, TEXT_PAD_ID)
        or (text_card is not None and tid >= text_card)
        for tid in token_ids
    ):
        upper_msg = f", < {text_card}" if text_card is not None else ""
        raise ValueError(
            f"sp_encode_fn 출력에 허용 범위를 벗어난 토큰(0 이상{upper_msg}) 또는 "
            f"예약 특수 토큰(EOS={TEXT_EOS_ID}, PAD={TEXT_PAD_ID})이 포함되어 있습니다."
        )


def validate_sim_bin(raw_sim_bin: Any, num_speaker_sim_bins: int) -> int:
    """보컬 음색 유사도 반환값이 유효한 정수 구간 인덱스인지 검증합니다."""
    if isinstance(raw_sim_bin, bool):
        raise ValueError(f"sim_bin은 정수 인덱스여야 합니다: {raw_sim_bin!r}")
    try:
        sim_bin = operator.index(raw_sim_bin)
    except TypeError as exc:
        raise ValueError(f"sim_bin은 정수 인덱스여야 합니다: {raw_sim_bin!r}") from exc
    if sim_bin < 0 or sim_bin >= num_speaker_sim_bins:
        raise ValueError(
            f"sim_bin({sim_bin})이 허용 범위 [0, {num_speaker_sim_bins - 1}]를 벗어났습니다."
        )
    return sim_bin


def resolve_default_sim_bin(
    vocal_source: str,
    num_speaker_sim_bins: int = 5,
    explicit_sim_bin: Optional[int] = None,
) -> int:
    """보컬 오디오 소스 종류에 따른 기본 보컬 음색 유사도 구간(0..num_bins-1)을 결정합니다."""
    if explicit_sim_bin is not None:
        return validate_sim_bin(explicit_sim_bin, num_speaker_sim_bins)
    top_bin = max(0, num_speaker_sim_bins - 1)
    if vocal_source in ("real_vocal", "text-only", "synth-guide", "auto"):
        return top_bin
    return min(2, top_bin)


def encode_mimi_timeline(
    timeline_24k: torch.Tensor,
    mimi_encode_fn: Callable[[torch.Tensor], torch.Tensor],
    total_frames: int,
    expected_q: Optional[int] = None,
    max_chunk_frames: int = 1024,
    overlap_frames: int = 64,
) -> torch.Tensor:
    """보컬 오디오 타임라인을 코덱으로 양자화하되 초장문 파형은 인과 오버랩을 두고 분할 인코딩합니다."""
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
        raise ValueError(f"코덱 코드북 수 불일치: 기대 Q={expected_q}, 실제 {c.shape[0]}")

    if c.shape[1] >= total_frames:
        return c[:, :total_frames].contiguous()
    pad_len = total_frames - c.shape[1]
    if c.shape[1] > 0:
        pad = c[:, -1:].expand(c.shape[0], pad_len).contiguous()
    else:
        pad = torch.full((c.shape[0], pad_len), ZERO_TOKEN_ID, dtype=torch.long)
    return torch.cat([c, pad], dim=1)


def assemble_lyric_section_example(
    unit: LyricSectionUnit,
    sp_encode_fn: Callable[[str], List[int]],
    mimi_encode_fn: Callable[[torch.Tensor], torch.Tensor],
    expected_q: int,
    text_card: int,
    vocal_mode: str = "auto",
    src_wav_path: Optional[str] = None,
    tgt_wav_path: Optional[str] = None,
    lead_frames: int = 1,
    hard_max_frames: int = DEFAULT_CONTEXT_FRAMES,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """단일 노래 섹션(LyricSectionUnit)으로부터 `[1 + 2Q, T]` 코드 텐서와 음절·음소·SSM 메타데이터를 조립합니다."""
    src_lines = [ln.src_text for ln in unit.lines]
    tgt_lines = [ln.tgt_text for ln in unit.lines]
    src_syllables = [ln.src_syllables for ln in unit.lines]
    target_syllables = [ln.tgt_syllables for ln in unit.lines]
    repeat_group_ids = [ln.repeat_group_id for ln in unit.lines]

    per_line_token_groups: List[List[int]] = []
    for idx, ln in enumerate(unit.lines):
        rendered_line = render_section_with_controls(
            lines=[ln.tgt_text],
            target_syllables=[ln.tgt_syllables],
            repeat_group_ids=[ln.repeat_group_id],
            section_type_id=unit.section_type_id if idx == 0 else None,
            include_syllable_tokens=True,
        )
        if idx < len(unit.lines) - 1:
            rendered_line = f"{rendered_line} <SEP>"
        tok_ids = list(sp_encode_fn(rendered_line))
        validate_token_ids(tok_ids, text_card)
        per_line_token_groups.append(tok_ids)

    src_wav_24k: Optional[torch.Tensor] = None
    tgt_wav_24k: Optional[torch.Tensor] = None
    if vocal_mode != "synth-guide":
        if src_wav_path and os.path.exists(src_wav_path):
            src_wav_24k = load_mono_wav_24k(src_wav_path)
        if tgt_wav_path and os.path.exists(tgt_wav_path):
            tgt_wav_24k = load_mono_wav_24k(tgt_wav_path)

    if src_wav_24k is None or src_wav_24k.numel() == 0:
        src_wav_24k, src_start_frames, src_end_frames = synthesize_rhythm_guide_waveform(
            syllable_counts=src_syllables,
            sample_rate=SAMPLE_RATE,
            base_freq_hz=220.0,
        )
    else:
        total_src_f = max(1, (int(src_wav_24k.numel()) + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME)
        total_syl = max(1, sum(src_syllables))
        src_start_frames = []
        src_end_frames = []
        acc_s = 0
        for s_cnt in src_syllables:
            sf_i = (acc_s * total_src_f) // total_syl
            acc_s += s_cnt
            ef_i = max(sf_i + 1, (acc_s * total_src_f) // total_syl)
            src_start_frames.append(sf_i)
            src_end_frames.append(ef_i)

    if vocal_mode != "text-only" and (tgt_wav_24k is None or tgt_wav_24k.numel() == 0):
        tgt_wav_24k, _, _ = synthesize_rhythm_guide_waveform(
            syllable_counts=target_syllables,
            sample_rate=SAMPLE_RATE,
            base_freq_hz=261.63,
        )

    token_placements: List[Tuple[int, int]] = []
    last_tok_frame = -1
    for s_f, e_f, tok_grp in zip(src_start_frames, src_end_frames, per_line_token_groups):
        line_pos, last_tok_frame = align_lyric_tokens_to_frames(
            token_ids=tok_grp,
            start_frame=s_f,
            end_frame=e_f,
            last_token_frame=last_tok_frame,
            lead_frames=lead_frames,
        )
        token_placements.extend(line_pos)

    src_audio_frames = (int(src_wav_24k.numel()) + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME
    tgt_audio_frames = (
        (int(tgt_wav_24k.numel()) + SAMPLES_PER_FRAME - 1) // SAMPLES_PER_FRAME
        if (tgt_wav_24k is not None and vocal_mode != "text-only")
        else src_audio_frames
    )
    audio_end_frames = max(src_audio_frames, tgt_audio_frames, src_end_frames[-1] if src_end_frames else 1)
    eos_frame = max(last_tok_frame + 1, audio_end_frames)
    total_frames = max(audio_end_frames + 4, eos_frame + 1)

    if hard_max_frames > 0 and total_frames > hard_max_frames:
        total_frames = hard_max_frames
        eos_frame = total_frames - 1

    text_stream = torch.full((1, total_frames), TEXT_PAD_ID, dtype=torch.long)
    for f_pos, tid in token_placements:
        if 0 <= f_pos < eos_frame:
            text_stream[0, f_pos] = int(tid)
    text_stream[0, eos_frame] = TEXT_EOS_ID

    src_codes = encode_mimi_timeline(src_wav_24k, mimi_encode_fn, total_frames, expected_q)
    if vocal_mode == "text-only" or tgt_wav_24k is None:
        tgt_codes = torch.full((expected_q, total_frames), ZERO_TOKEN_ID, dtype=torch.long)
    else:
        tgt_codes = encode_mimi_timeline(tgt_wav_24k, mimi_encode_fn, total_frames, expected_q)

    frame_target_syllables = torch.zeros(total_frames, dtype=torch.long)
    frame_repeat_group_ids = torch.zeros(total_frames, dtype=torch.long)
    for idx, (s_f, e_f) in enumerate(zip(src_start_frames, src_end_frames)):
        clamped_s = max(0, min(s_f, total_frames))
        clamped_e = max(clamped_s, min(e_f, total_frames))
        if idx == len(src_start_frames) - 1:
            clamped_e = total_frames
        frame_target_syllables[clamped_s:clamped_e] = int(target_syllables[idx])
        frame_repeat_group_ids[clamped_s:clamped_e] = int(repeat_group_ids[idx])

    codes = torch.cat([text_stream, tgt_codes, src_codes], dim=0)
    full_rendered_tgt = render_section_with_controls(
        lines=tgt_lines,
        target_syllables=target_syllables,
        repeat_group_ids=repeat_group_ids,
        section_type_id=unit.section_type_id,
        include_syllable_tokens=True,
    )

    metadata: Dict[str, Any] = {
        "song_id": unit.song_id,
        "section_idx": int(unit.section_idx),
        "section_type": unit.section_type,
        "section_type_id": int(unit.section_type_id),
        "direction": unit.direction,
        "src_lines": src_lines,
        "tgt_lines": tgt_lines,
        "rendered_tgt_text": full_rendered_tgt,
        "src_syllables": [int(x) for x in src_syllables],
        "target_syllables": [int(x) for x in target_syllables],
        "repeat_group_ids": [int(x) for x in repeat_group_ids],
        "frame_target_syllables": frame_target_syllables,
        "frame_repeat_group_ids": frame_repeat_group_ids,
        "src_semantic_ssm": unit.src_semantic_ssm,
        "src_phoneme_ssm": unit.src_phoneme_ssm,
        "tgt_phoneme_ssm": unit.tgt_phoneme_ssm,
        "src_start_frames": [int(min(f, total_frames - 1)) for f in src_start_frames],
        "src_end_frames": [int(min(f, total_frames)) for f in src_end_frames],
        "ref_token_groups": per_line_token_groups,
        "vocal_mode": vocal_mode,
    }
    return codes, metadata


def save_training_example_pt(
    output_path: Path,
    codes: torch.Tensor,
    sim_bin: int,
    metadata: Optional[Mapping[str, Any]] = None,
    max_retries: int = 5,
) -> None:
    """멀티스트림 코드 텐서, 보컬 유사도 구간 및 싱어빌리티 메타데이터를 원자적으로 `.pt`에 저장합니다."""
    payload: Dict[str, Any] = {"codes": codes.detach().cpu(), "sim_bin": int(sim_bin)}
    if metadata:
        for k, v in metadata.items():
            if k not in payload:
                payload[k] = v.detach().cpu() if isinstance(v, torch.Tensor) else v
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
    """섹션 또는 이진 분할 하위 트리가 모두 저장되었으면 리프 개수를, 일부라도 누락되었으면 None을 반환합니다."""
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
    "LyricLinePair",
    "LyricSectionUnit",
    "batched_pool_map",
    "interleave_lists",
    "fast_scandir_files",
    "build_lyric_section_unit",
    "slice_long_section_unit",
    "validate_token_ids",
    "validate_sim_bin",
    "resolve_default_sim_bin",
    "encode_mimi_timeline",
    "assemble_lyric_section_example",
    "save_training_example_pt",
    "index_existing_window_trees",
    "count_completed_window_tree",
]
