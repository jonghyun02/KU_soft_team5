"""arXiv:2309.11093 Stage 2 Singability GRPO 프롬프트 데이터셋, 병렬 롤아웃 및 복합 싱어빌리티 보상 모듈."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch

from libs.config import LyricTransConfig
from libs.model import LyricTransModel
from libs.phonetics import (
    compute_phoneme_ssm,
    compute_repetition_group_ids,
    compute_section_rhyme_and_consistency_score,
    compute_section_scd,
    compute_semantic_ssm,
    compute_ssm_alignment_score,
    count_line_syllables,
)
from libs.streaming import LyricStreamingGenerator, build_frame_condition_schedules
from libs.tokenizer import split_rendered_section_lines
from libs.training import list_preprocessed_pt_files, load_and_validate_pt_item


@dataclass
class GRPOPromptSample:
    """단일 노래 섹션 프롬프트와 목표 음절 수·반복 행 구조·원문 자기유사도 행렬(SSM)을 담는 데이터클래스입니다."""

    name: str
    src_codes: torch.Tensor
    silence_code: torch.Tensor
    ref_tokens: List[int]
    src_start_frames: List[int]
    src_end_frames: List[int]
    ref_token_groups: List[List[int]]
    sim_bin: int
    section_type_id: int = 2
    src_lines: List[str] = field(default_factory=list)
    tgt_lines: List[str] = field(default_factory=list)
    target_syllables: List[int] = field(default_factory=list)
    repeat_group_ids: List[int] = field(default_factory=list)
    src_semantic_ssm: List[List[float]] = field(default_factory=list)
    src_phoneme_ssm: List[List[float]] = field(default_factory=list)
    frame_target_syllables: Optional[torch.Tensor] = None
    frame_repeat_group_ids: Optional[torch.Tensor] = None


@dataclass
class RolloutBatch:
    """단일 노래 섹션 프롬프트에 대해 병렬 샘플링된 G개 싱어블 번역 궤적 묶음 데이터클래스입니다."""

    rollout_codes: torch.Tensor
    pred_token_lists: List[List[int]]
    pred_token_frames: List[List[int]]
    eos_reached: List[bool]
    valid_lengths: List[int]
    eos_suppress_steps: int = 0
    section_type_id: int = 2
    frame_target_syllables: Optional[torch.Tensor] = None
    frame_repeat_group_ids: Optional[torch.Tensor] = None


def _infer_source_span(
    src_codes: torch.Tensor,
    raw_item: dict[str, Any],
    crop_src_frames: int,
) -> Tuple[List[int], List[int], int]:
    """메타데이터 또는 무음 꼬리 역추적을 통해 원곡 섹션 보컬의 행별 시작·종료 프레임 구간을 결정합니다."""
    total_t = int(src_codes.shape[1])
    raw_starts = raw_item.get("src_start_frames")
    raw_ends = raw_item.get("src_end_frames")

    if (
        isinstance(raw_ends, list)
        and len(raw_ends) > 0
        and all(isinstance(x, int) for x in raw_ends)
    ):
        starts = (
            [max(0, int(x)) for x in raw_starts]
            if (isinstance(raw_starts, list) and len(raw_starts) == len(raw_ends))
            else [0] * len(raw_ends)
        )
        ends = [max(1, min(total_t, int(x))) for x in raw_ends]
        src_len = min(crop_src_frames, max(ends))
        clipped_starts: List[int] = []
        clipped_ends: List[int] = []
        for s_f, e_f in zip(starts, ends):
            if s_f >= src_len:
                break
            clipped_starts.append(s_f)
            clipped_ends.append(min(e_f, src_len))
        if clipped_ends:
            return clipped_starts, clipped_ends, max(1, src_len)

    tail_sem = int(src_codes[0, -1].item())
    active_end = total_t
    while active_end > 1 and int(src_codes[0, active_end - 1].item()) == tail_sem:
        active_end -= 1
    if active_end <= 1:
        active_end = max(1, min(total_t, int(total_t * 0.75)))
    src_len = max(1, min(crop_src_frames, active_end))
    return [0], [src_len], src_len


class GRPOPromptDataset:
    """전처리된 `.pt` 파일들에서 GRPO 롤아웃용 섹션 프롬프트와 싱어빌리티 정답 메타데이터를 로드하는 데이터셋입니다."""

    def __init__(
        self,
        data_dir: Path,
        crop_src_frames: int,
        cfg: LyricTransConfig,
        default_sim_bin: Optional[int] = None,
    ):
        """프롬프트 `.pt` 파일 목록을 수집하고 첫 번째 섹션 프롬프트의 유효성을 검증합니다."""
        if crop_src_frames < 1:
            raise ValueError(f"crop_src_frames({crop_src_frames})는 1 이상이어야 합니다.")
        self.files = list_preprocessed_pt_files(data_dir)
        self.crop_src_frames = crop_src_frames
        self.cfg = cfg
        self.default_sim_bin = (
            cfg.num_speaker_sim_bins - 1 if default_sim_bin is None else int(default_sim_bin)
        )

        _ = self[0]

    def __len__(self) -> int:
        """전체 섹션 프롬프트 개수를 반환합니다."""
        return len(self.files)

    def __getitem__(self, idx: int) -> GRPOPromptSample:
        """지정된 인덱스의 `.pt` 샘플에서 소스 보컬 스트림과 싱어빌리티 참조 정보를 반환합니다."""
        path = self.files[idx]
        item, codes, raw_sim_bin = load_and_validate_pt_item(
            path=path,
            expected_total_streams=self.cfg.total_streams,
            default_sim_bin=self.default_sim_bin,
            num_speaker_sim_bins=self.cfg.num_speaker_sim_bins,
            validate_ranges=False,
        )
        q = self.cfg.dep_q
        text_stream = codes[0]
        raw_src_codes = codes[1 + q : 1 + 2 * q]
        silence_code = raw_src_codes[:, -1:].clone()

        src_starts, src_ends, src_len = _infer_source_span(
            raw_src_codes, item, crop_src_frames=self.crop_src_frames
        )
        src_codes = raw_src_codes[:, :src_len].contiguous()

        raw_groups = item.get("ref_token_groups")
        if isinstance(raw_groups, list) and raw_groups:
            kept_groups = raw_groups[: len(src_ends)]
            ref_groups = [
                [
                    int(t)
                    for t in grp
                    if 0 <= int(t) < self.cfg.text_card
                    and int(t) not in (self.cfg.text_pad_id, self.cfg.text_eos_id)
                ]
                for grp in kept_groups
            ]
            ref_tokens = [t for grp in ref_groups for t in grp]
        else:
            ref_tokens = [
                int(t)
                for t in text_stream.tolist()
                if 0 <= int(t) < self.cfg.text_card
                and int(t) not in (self.cfg.text_pad_id, self.cfg.text_eos_id)
            ]
            ref_groups = [ref_tokens] if ref_tokens else [[]]

        num_lines = max(1, len(src_ends))
        src_lines = [str(x) for x in item.get("src_lines", [])][:num_lines]
        tgt_lines = [str(x) for x in item.get("tgt_lines", [])][:num_lines]

        raw_tgt_syl = item.get("target_syllables")
        if isinstance(raw_tgt_syl, list) and raw_tgt_syl:
            target_syllables = [max(1, int(x)) for x in raw_tgt_syl[:num_lines]]
        elif tgt_lines:
            target_syllables = [max(1, count_line_syllables(ln)) for ln in tgt_lines]
        else:
            target_syllables = [8] * num_lines

        raw_rep_ids = item.get("repeat_group_ids")
        if isinstance(raw_rep_ids, list) and raw_rep_ids:
            repeat_group_ids = [int(x) for x in raw_rep_ids[:num_lines]]
        elif src_lines:
            repeat_group_ids = compute_repetition_group_ids(src_lines)
        else:
            repeat_group_ids = list(range(num_lines))

        src_sem_ssm = item.get("src_semantic_ssm")
        if not isinstance(src_sem_ssm, list) or not src_sem_ssm:
            src_sem_ssm = (
                compute_semantic_ssm(src_lines)
                if src_lines
                else [[1.0 if i == j else 0.0 for j in range(num_lines)] for i in range(num_lines)]
            )

        src_pho_ssm = item.get("src_phoneme_ssm")
        if not isinstance(src_pho_ssm, list) or not src_pho_ssm:
            src_pho_ssm = (
                compute_phoneme_ssm(src_lines)
                if src_lines
                else src_sem_ssm
            )

        raw_f_syl = item.get("frame_target_syllables")
        f_syl = (
            raw_f_syl[:src_len].long().contiguous()
            if isinstance(raw_f_syl, torch.Tensor) and raw_f_syl.numel() > 0
            else None
        )
        raw_f_rep = item.get("frame_repeat_group_ids")
        f_rep = (
            raw_f_rep[:src_len].long().contiguous()
            if isinstance(raw_f_rep, torch.Tensor) and raw_f_rep.numel() > 0
            else None
        )

        sim_bin = max(0, min(self.cfg.num_speaker_sim_bins - 1, raw_sim_bin))
        section_type_id = int(item.get("section_type_id", 2))

        return GRPOPromptSample(
            name=path.name,
            src_codes=src_codes,
            silence_code=silence_code,
            ref_tokens=ref_tokens,
            src_start_frames=src_starts,
            src_end_frames=src_ends,
            ref_token_groups=ref_groups,
            sim_bin=sim_bin,
            section_type_id=section_type_id,
            src_lines=src_lines,
            tgt_lines=tgt_lines,
            target_syllables=target_syllables,
            repeat_group_ids=repeat_group_ids,
            src_semantic_ssm=src_sem_ssm,
            src_phoneme_ssm=src_pho_ssm,
            frame_target_syllables=f_syl,
            frame_repeat_group_ids=f_rep,
        )


def compute_smoothed_bleu(
    pred_seq: Sequence[int | str],
    ref_seq: Sequence[int | str],
    max_order: int = 4,
) -> float:
    """외부 패키지 없이 토큰 또는 문자 시퀀스에 대한 스무딩 BLEU 점수(`0.0 ~ 1.0`)를 계산합니다."""
    pred_len = len(pred_seq)
    ref_len = len(ref_seq)
    if pred_len == 0 or ref_len == 0:
        return 0.0

    eff_order = min(max_order, pred_len, ref_len)
    if eff_order < 1:
        return 0.0

    log_prec_sum = 0.0
    smooth_eps = 0.1
    for n in range(1, eff_order + 1):
        pred_ngrams = Counter(tuple(pred_seq[i : i + n]) for i in range(pred_len - n + 1))
        ref_ngrams = Counter(tuple(ref_seq[i : i + n]) for i in range(ref_len - n + 1))
        overlap = sum(min(cnt, ref_ngrams[ng]) for ng, cnt in pred_ngrams.items())
        total = max(1, pred_len - n + 1)
        if overlap == 0:
            if n == 1:
                return 0.0
            prec = smooth_eps / float(total)
            smooth_eps *= 0.5
        else:
            prec = float(overlap) / float(total)
        log_prec_sum += math.log(prec) / float(eff_order)

    bp = 1.0 if pred_len >= ref_len else math.exp(1.0 - float(ref_len) / float(pred_len))
    return float(max(0.0, min(1.0, bp * math.exp(log_prec_sum))))


def rollout_group_trajectories(
    model: LyricTransModel,
    prompt: GRPOPromptSample,
    group_size: int,
    max_tail_sec: float,
    rollout_cfg_coef: float,
    temp_text: float,
    temp_audio: float,
    top_k_text: int,
    top_k_audio: int,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> RolloutBatch:
    """단일 노래 섹션 프롬프트에 대해 G개의 싱어블 가사·보컬 번역 궤적을 배치 병렬로 샘플링합니다."""
    cfg = model.cfg
    src_len = int(prompt.src_codes.shape[1])
    src_batch = prompt.src_codes.unsqueeze(0).expand(group_size, -1, -1).to(device)
    sil_batch = prompt.silence_code.unsqueeze(0).expand(group_size, -1, -1).to(device)

    max_allowed_total = max(src_len + 2, cfg.context_frames)
    max_tail_frames = min(
        max(2, int(max_tail_sec * cfg.frame_rate)),
        max(2, max_allowed_total - src_len),
    )

    avg_target_syl = (
        int(round(sum(prompt.target_syllables) / len(prompt.target_syllables)))
        if prompt.target_syllables
        else 8
    )
    full_sched_len = src_len + max_tail_frames + max(cfg.delays) + 4
    frame_syl_sched, frame_rep_sched = build_frame_condition_schedules(
        total_frames=full_sched_len,
        target_syllables=prompt.target_syllables,
        repeat_group_ids=prompt.repeat_group_ids,
        src_start_frames=prompt.src_start_frames,
        src_end_frames=prompt.src_end_frames,
        frame_target_syllables=prompt.frame_target_syllables,
        frame_repeat_group_ids=prompt.frame_repeat_group_ids,
        default_target_syllables=avg_target_syl,
    )

    was_training = model.training
    model.eval()
    try:
        streamer = LyricStreamingGenerator(
            model,
            batch_size=group_size,
            cfg_coef=rollout_cfg_coef,
            temp_audio=temp_audio,
            temp_text=temp_text,
            top_k_audio=top_k_audio,
            top_k_text=top_k_text,
            sim_bin=prompt.sim_bin,
            section_type_id=prompt.section_type_id,
            default_target_syllables=avg_target_syl,
            frame_target_syllables=frame_syl_sched,
            frame_repeat_group_ids=frame_rep_sched,
        )

        fed_src_list: List[torch.Tensor] = []
        emitted_text_list: List[torch.Tensor] = []
        emitted_audio_list: List[torch.Tensor] = []
        eos_frame_idx = [-1] * group_size

        with torch.no_grad():
            for t in range(src_len):
                chunk = src_batch[:, :, t : t + 1]
                fed_src_list.append(chunk)
                with torch.amp.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=use_amp
                ):
                    step_out = streamer.step(chunk, suppress_eos=True)
                if step_out is not None:
                    t_tok, a_tok = step_out
                    emitted_text_list.append(t_tok.unsqueeze(1))
                    emitted_audio_list.append(a_tok)

            for _ in range(max_tail_frames):
                if all(idx >= 0 for idx in eos_frame_idx):
                    break
                fed_src_list.append(sil_batch)
                with torch.amp.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=use_amp
                ):
                    step_out = streamer.step(sil_batch, suppress_eos=False)
                if step_out is not None:
                    t_tok, a_tok = step_out
                    cur_f = len(emitted_text_list)
                    emitted_text_list.append(t_tok.unsqueeze(1))
                    emitted_audio_list.append(a_tok)
                    for g in range(group_size):
                        if eos_frame_idx[g] < 0 and int(t_tok[g, 0].item()) == cfg.text_eos_id:
                            eos_frame_idx[g] = cur_f

            while len(emitted_text_list) < len(fed_src_list):
                with torch.amp.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=use_amp
                ):
                    step_out = streamer.step(sil_batch, suppress_eos=False)
                if step_out is None:
                    break
                t_tok, a_tok = step_out
                cur_f = len(emitted_text_list)
                emitted_text_list.append(t_tok.unsqueeze(1))
                emitted_audio_list.append(a_tok)
                for g in range(group_size):
                    if eos_frame_idx[g] < 0 and int(t_tok[g, 0].item()) == cfg.text_eos_id:
                        eos_frame_idx[g] = cur_f

        total_roll_t = min(len(emitted_text_list), len(fed_src_list), cfg.context_frames)
        pred_text = torch.cat(emitted_text_list[:total_roll_t], dim=-1)
        pred_audio = torch.cat(emitted_audio_list[:total_roll_t], dim=-1)
        fed_src = torch.cat(fed_src_list[:total_roll_t], dim=-1)

        rollout_codes = torch.cat([pred_text, pred_audio, fed_src], dim=1)

        pred_token_lists: List[List[int]] = []
        pred_token_frames: List[List[int]] = []
        eos_reached_flags: List[bool] = []
        valid_lengths: List[int] = []

        for g in range(group_size):
            e_idx = eos_frame_idx[g]
            if 0 <= e_idx < total_roll_t:
                eos_reached_flags.append(True)
                valid_len = e_idx + 1
                if valid_len < total_roll_t:
                    rollout_codes[g, : 1 + cfg.dep_q, valid_len:] = cfg.zero_token_id
            else:
                eos_reached_flags.append(False)
                valid_len = total_roll_t
            valid_lengths.append(valid_len)

            seq_t = pred_text[g, 0, :valid_len].tolist()
            toks: List[int] = []
            frames: List[int] = []
            for f_i, tid in enumerate(seq_t):
                tid_int = int(tid)
                if tid_int in (cfg.text_pad_id, cfg.text_eos_id) or tid_int < 0:
                    continue
                toks.append(tid_int)
                frames.append(f_i)
            pred_token_lists.append(toks)
            pred_token_frames.append(frames)

        f_syl = frame_syl_sched[:total_roll_t].clone()
        f_rep = frame_rep_sched[:total_roll_t].clone()

        return RolloutBatch(
            rollout_codes=rollout_codes,
            pred_token_lists=pred_token_lists,
            pred_token_frames=pred_token_frames,
            eos_reached=eos_reached_flags,
            valid_lengths=valid_lengths,
            eos_suppress_steps=min(src_len, total_roll_t),
            section_type_id=prompt.section_type_id,
            frame_target_syllables=f_syl,
            frame_repeat_group_ids=f_rep,
        )
    finally:
        if was_training:
            model.train()


def _extract_predicted_lines_and_syllables(
    pred_toks: List[int],
    pred_frames: List[int],
    prompt: GRPOPromptSample,
    sp_decode_fn: Optional[Callable[[List[int]], str]],
) -> Tuple[List[str], List[int]]:
    """예측 토큰 시퀀스를 토크나이저 디코딩 또는 프레임 구간 분할을 통해 행별 문자열과 음절 수로 변환합니다."""
    num_lines = max(1, len(prompt.target_syllables) if prompt.target_syllables else len(prompt.src_end_frames))
    if sp_decode_fn is not None and pred_toks:
        decoded_str = sp_decode_fn(pred_toks)
        pred_lines = split_rendered_section_lines(decoded_str, expected_num_lines=num_lines)
        pred_syls = [count_line_syllables(ln) for ln in pred_lines]
        return pred_lines, pred_syls

    if not pred_toks:
        return [""] * num_lines, [0] * num_lines

    line_tok_groups: List[List[int]] = [[] for _ in range(num_lines)]
    if len(prompt.src_end_frames) == num_lines:
        for tid, f_pos in zip(pred_toks, pred_frames):
            placed = False
            for l_i, e_f in enumerate(prompt.src_end_frames):
                if f_pos <= e_f + 2:
                    line_tok_groups[l_i].append(tid)
                    placed = True
                    break
            if not placed:
                line_tok_groups[-1].append(tid)
    else:
        for idx, tid in enumerate(pred_toks):
            l_i = min(num_lines - 1, (idx * num_lines) // max(1, len(pred_toks)))
            line_tok_groups[l_i].append(tid)

    pseudo_lines = [" ".join(f"w{t}" for t in grp) for grp in line_tok_groups]
    pseudo_syls = [len(grp) for grp in line_tok_groups]
    return pseudo_lines, pseudo_syls


def evaluate_rollout_group_rewards(
    prompt: GRPOPromptSample,
    rollout: RolloutBatch,
    cfg: LyricTransConfig,
    bleu_weight: float = 1.0,
    syllable_weight: float = 1.2,
    ssm_weight: float = 0.8,
    rhyme_weight: float = 0.6,
    audio_weight: float = 0.10,
    sp_decode_fn: Optional[Callable[[List[int]], str]] = None,
    custom_reward_fn: Optional[Callable[[GRPOPromptSample, RolloutBatch, int], float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """arXiv:2309.11093의 4대 싱어빌리티 지표(BLEU, 음절 정합 1-SCD, 반복 행렬 SSM, 각운·반복 일관성)를 결합해 그룹 상대 우위를 산출합니다."""
    del cfg
    group_size = len(rollout.pred_token_lists)
    if prompt.tgt_lines:
        ref_clean_str: Optional[str] = " ".join(prompt.tgt_lines)
    elif sp_decode_fn is not None and prompt.ref_tokens:
        ref_clean_str = " ".join(split_rendered_section_lines(sp_decode_fn(prompt.ref_tokens)))
    else:
        ref_clean_str = None
    target_syllables = (
        prompt.target_syllables
        if prompt.target_syllables
        else [max(1, len(grp)) for grp in prompt.ref_token_groups]
    )

    rewards: List[float] = []
    bleu_scores: List[float] = []
    syllable_scores: List[float] = []
    scd_values: List[float] = []
    ssm_scores: List[float] = []
    rhyme_scores: List[float] = []
    audio_scores: List[float] = []

    for g in range(group_size):
        pred_toks = rollout.pred_token_lists[g]
        pred_frames = rollout.pred_token_frames[g]
        v_len = rollout.valid_lengths[g]

        pred_lines, pred_syls = _extract_predicted_lines_and_syllables(
            pred_toks=pred_toks,
            pred_frames=pred_frames,
            prompt=prompt,
            sp_decode_fn=sp_decode_fn,
        )

        tok_bleu = compute_smoothed_bleu(pred_toks, prompt.ref_tokens)
        if sp_decode_fn is not None and ref_clean_str is not None and pred_toks:
            pred_clean_str = " ".join(pred_lines)
            char_bleu = compute_smoothed_bleu(
                [ch for ch in pred_clean_str if not ch.isspace()],
                [ch for ch in ref_clean_str if not ch.isspace()],
            )
            bleu_val = 0.6 * tok_bleu + 0.4 * char_bleu
        else:
            bleu_val = tok_bleu

        scd_val, _ = compute_section_scd(target_syllables, pred_syls)
        syl_score = max(0.0, min(1.0, 1.0 - scd_val))

        if any(ln.strip() for ln in pred_lines) and len(pred_lines) > 1:
            pred_pho_ssm = compute_phoneme_ssm(pred_lines)
            ref_ssm = (
                prompt.src_semantic_ssm
                if prompt.src_semantic_ssm
                else prompt.src_phoneme_ssm
            )
            ssm_score = compute_ssm_alignment_score(ref_ssm, pred_pho_ssm)
        else:
            ssm_score = 1.0 if len(target_syllables) <= 1 else 0.0

        src_for_rhyme = prompt.src_lines if prompt.src_lines else pred_lines
        rhyme_score = compute_section_rhyme_and_consistency_score(
            src_lines=src_for_rhyme,
            pred_lines=pred_lines,
            repeat_group_ids=prompt.repeat_group_ids,
        )

        penalty = 0.0
        if len(prompt.ref_tokens) > 0 and len(pred_toks) == 0:
            penalty += 0.50
        if not rollout.eos_reached[g]:
            penalty += 0.10

        sem_audio = rollout.rollout_codes[g, 1, :v_len]
        valid_sem = sem_audio[sem_audio >= 0]
        if valid_sem.numel() > 2:
            transitions = float((valid_sem[1:] != valid_sem[:-1]).float().mean().item())
            audio_health = min(1.0, transitions / 0.25)
        else:
            audio_health = 0.0

        ext_bonus = (
            float(custom_reward_fn(prompt, rollout, g))
            if custom_reward_fn is not None
            else 0.0
        )
        a_score = audio_health + ext_bonus

        total_r = (
            bleu_weight * bleu_val
            + syllable_weight * syl_score
            + ssm_weight * ssm_score
            + rhyme_weight * rhyme_score
            + audio_weight * a_score
            - penalty
        )
        rewards.append(total_r)
        bleu_scores.append(bleu_val)
        syllable_scores.append(syl_score)
        scd_values.append(scd_val)
        ssm_scores.append(ssm_score)
        rhyme_scores.append(rhyme_score)
        audio_scores.append(a_score)

    r_tensor = torch.tensor(rewards, dtype=torch.float32)
    r_mean = r_tensor.mean()
    r_std = r_tensor.std(correction=0)
    if float(r_std.item()) < 1e-6:
        advantages = torch.zeros_like(r_tensor)
    else:
        advantages = (r_tensor - r_mean) / (r_std + 1e-5)

    denom = float(max(1, group_size))
    stats = {
        "reward_mean": float(r_mean.item()),
        "reward_std": float(r_std.item()),
        "bleu_mean": float(sum(bleu_scores) / denom),
        "syllable_score_mean": float(sum(syllable_scores) / denom),
        "scd_mean": float(sum(scd_values) / denom),
        "ssm_score_mean": float(sum(ssm_scores) / denom),
        "rhyme_score_mean": float(sum(rhyme_scores) / denom),
        "audio_score_mean": float(sum(audio_scores) / denom),
        "eos_rate": float(sum(1.0 for flag in rollout.eos_reached if flag) / denom),
    }
    return r_tensor, advantages, stats
