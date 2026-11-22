from __future__ import annotations

import argparse
import copy
import importlib
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Tuple

from env_setup import (
    DEFAULT_CUDA_INDEX_URL,
    ensure_venv_and_reexec,
    get_default_checkpoint_path,
    get_default_grpo_checkpoint_path,
    get_default_venv_dir,
    setup_device,
)

if TYPE_CHECKING:
    import torch
    from EDarch_from_scratch import EDarchConfig, EDarchModel, EDarchSequenceLogProbs


@dataclass
class GRPOPromptSample:
    name: str
    src_codes: "torch.Tensor"  # [Q, T_src]
    silence_code: "torch.Tensor"  # [Q, 1]
    ref_tokens: List[int]
    src_start_frames: List[int]
    src_end_frames: List[int]
    ref_token_groups: List[List[int]]
    sim_bin: int


@dataclass
class RolloutBatch:
    rollout_codes: "torch.Tensor"  # [G, 1 + 2Q, T_roll]
    pred_token_lists: List[List[int]]
    pred_token_frames: List[List[int]]
    eos_reached: List[bool]
    valid_lengths: List[int]


def _infer_source_span(
    en_codes: "torch.Tensor",
    raw_item: dict[str, Any],
    crop_src_frames: int,
) -> Tuple[List[int], List[int], int]:
    total_t = int(en_codes.shape[1])
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

    # 구버전 .pt 호환: 우측 무음 패딩 구간을 역추적하여 실제 영어 발화 종료 프레임 추정
    tail_sem = int(en_codes[0, -1].item())
    active_end = total_t
    while active_end > 1 and int(en_codes[0, active_end - 1].item()) == tail_sem:
        active_end -= 1
    if active_end <= 1:
        active_end = max(1, min(total_t, int(total_t * 0.6)))
    src_len = max(1, min(crop_src_frames, active_end))
    return [0], [src_len], src_len


class GRPOPromptDataset:
    """전처리된 .pt 파일에서 Stage 2 GRPO 롤아웃용 소스 오디오 프롬프트와 정답 시퀀스를 추출합니다."""

    def __init__(
        self,
        data_dir: Path,
        crop_src_frames: int,
        cfg: "EDarchConfig",
    ):
        if crop_src_frames < 1:
            raise ValueError(f"crop_src_frames({crop_src_frames})는 1 이상이어야 합니다.")
        self.files = sorted(data_dir.rglob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"{data_dir}에서 .pt 파일을 찾지 못했습니다.")
        self.crop_src_frames = crop_src_frames
        self.cfg = cfg

        _ = self[0]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> GRPOPromptSample:
        import torch

        path = self.files[idx]
        item = torch.load(path, map_location="cpu", weights_only=True)
        codes = item["codes"].long()
        q = self.cfg.dep_q
        if codes.ndim != 2 or codes.shape[0] != self.cfg.total_streams:
            raise ValueError(
                f"스트림 수 불일치 ({path.name}): 기대 {self.cfg.total_streams}, 실제 {tuple(codes.shape)}"
            )

        text_stream = codes[0]
        en_codes = codes[1 + q : 1 + 2 * q]
        silence_code = en_codes[:, -1:].clone()

        src_starts, src_ends, src_len = _infer_source_span(
            en_codes, item, crop_src_frames=self.crop_src_frames
        )
        src_codes = en_codes[:, :src_len].contiguous()

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

        sim_bin = int(item.get("sim_bin", self.cfg.num_speaker_sim_bins - 1))
        sim_bin = max(0, min(self.cfg.num_speaker_sim_bins - 1, sim_bin))

        return GRPOPromptSample(
            name=path.name,
            src_codes=src_codes,
            silence_code=silence_code,
            ref_tokens=ref_tokens,
            src_start_frames=src_starts,
            src_end_frames=src_ends,
            ref_token_groups=ref_groups,
            sim_bin=sim_bin,
        )


def compute_smoothed_bleu(
    pred_seq: Sequence[int | str],
    ref_seq: Sequence[int | str],
    max_order: int = 4,
) -> float:
    """외부 패키지 의존성 없이 토큰/단어 시퀀스에 대한 Chen & Cherry Smoothed BLEU(0..1)를 계산합니다."""
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


def compute_laal_latency_sec(
    tok_frames: Sequence[int],
    src_start_frame: int,
    src_end_frame: int,
    ref_len: int,
    frame_rate: float = 12.5,
) -> float:
    """
    동시통역 표준 지연 지표인 LAAL(Length-Adaptive Average Lagging, 초 단위)을 계산합니다.
    모델이 원문 종료 전 적절한 시점에 번역을 일찍 시작할수록 지연시간이 감소합니다.
    """
    src_dur = max(1, src_end_frame - src_start_frame)
    if not tok_frames:
        return float(src_dur) / frame_rate

    pred_len = len(tok_frames)
    gamma = float(max(pred_len, max(1, ref_len))) / float(src_dur)
    lags: List[float] = []

    for i, f_pos in enumerate(tok_frames):
        rel_read = min(float(max(0, f_pos - src_start_frame + 1)), float(src_dur))
        ideal_read = float(i) / max(1e-6, gamma)
        lags.append(max(0.0, rel_read - ideal_read))
        if f_pos + 1 >= src_end_frame:
            break

    if not lags:
        return float(src_dur) / frame_rate
    return float(sum(lags) / len(lags)) / frame_rate


def rollout_group_trajectories(
    model: "EDarchModel",
    prompt: GRPOPromptSample,
    group_size: int,
    max_tail_sec: float,
    rollout_cfg_coef: float,
    temp_text: float,
    temp_audio: float,
    top_k_text: int,
    top_k_audio: int,
    device: "torch.device",
    use_amp: bool,
    amp_dtype: "torch.dtype",
) -> RolloutBatch:
    import torch
    from EDarch_from_scratch import EDarchStreamingGenerator

    cfg = model.cfg
    src_len = int(prompt.src_codes.shape[1])
    src_batch = prompt.src_codes.unsqueeze(0).expand(group_size, -1, -1).to(device)
    sil_batch = prompt.silence_code.unsqueeze(0).expand(group_size, -1, -1).to(device)

    max_allowed_total = max(src_len + 2, cfg.context_frames)
    max_tail_frames = min(
        max(2, int(max_tail_sec * cfg.frame_rate)),
        max(2, max_allowed_total - src_len),
    )

    was_training = model.training
    model.eval()
    try:
        streamer = EDarchStreamingGenerator(
            model,
            batch_size=group_size,
            cfg_coef=rollout_cfg_coef,
            temp_audio=temp_audio,
            temp_text=temp_text,
            top_k_audio=top_k_audio,
            top_k_text=top_k_text,
        )

        fed_src_list: List[torch.Tensor] = []
        emitted_text_list: List[torch.Tensor] = []
        emitted_audio_list: List[torch.Tensor] = []
        eos_frame_idx = [-1] * group_size

        with torch.no_grad():
            for t in range(src_len):
                chunk = src_batch[:, :, t : t + 1]
                fed_src_list.append(chunk)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    step_out = streamer.step(chunk, suppress_eos=True)
                if step_out is not None:
                    t_tok, a_tok = step_out
                    emitted_text_list.append(t_tok.unsqueeze(1))
                    emitted_audio_list.append(a_tok)

            for _ in range(max_tail_frames):
                if all(idx >= 0 for idx in eos_frame_idx):
                    break
                fed_src_list.append(sil_batch)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    step_out = streamer.step(sil_batch, suppress_eos=False)
                if step_out is not None:
                    t_tok, a_tok = step_out
                    cur_f = len(emitted_text_list)
                    emitted_text_list.append(t_tok.unsqueeze(1))
                    emitted_audio_list.append(a_tok)
                    for g in range(group_size):
                        if eos_frame_idx[g] < 0 and int(t_tok[g, 0].item()) == cfg.text_eos_id:
                            eos_frame_idx[g] = cur_f

            # 링 버퍼에 지연되어 남아 있는 마지막 max_delay 프레임을 마저 배출
            while len(emitted_text_list) < len(fed_src_list):
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
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
        pred_text = torch.cat(emitted_text_list[:total_roll_t], dim=-1)  # [G, 1, T]
        pred_audio = torch.cat(emitted_audio_list[:total_roll_t], dim=-1)  # [G, Q, T]
        fed_src = torch.cat(fed_src_list[:total_roll_t], dim=-1)  # [G, Q, T]

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

        return RolloutBatch(
            rollout_codes=rollout_codes,
            pred_token_lists=pred_token_lists,
            pred_token_frames=pred_token_frames,
            eos_reached=eos_reached_flags,
            valid_lengths=valid_lengths,
        )
    finally:
        if was_training:
            model.train()


def evaluate_rollout_group_rewards(
    prompt: GRPOPromptSample,
    rollout: RolloutBatch,
    cfg: "EDarchConfig",
    bleu_weight: float = 1.0,
    latency_weight: float = 0.15,
    audio_weight: float = 0.10,
    min_wait_frames: int = 3,
    sp_decode_fn: Optional[Callable[[List[int]], str]] = None,
    custom_reward_fn: Optional[Callable[[GRPOPromptSample, RolloutBatch, int], float]] = None,
) -> Tuple["torch.Tensor", "torch.Tensor", dict[str, float]]:
    """
    Hibiki-Zero 철학에 따라 단어 정렬 데이터 없이:
      1) 최종 + 발화 경계 중간(Process) BLEU 번역 정확도 보상
      2) LAAL 동시통역 지연시간 페널티 (인과성 조기 추측 방지 가드 포함)
      3) 타깃 오디오 코드북 건강도 및 사용자 정의 보상(예: WavLM/COMET)
    을 결합하고 그룹 상대 정규화(Group Relative Advantage)를 계산합니다.
    """
    import torch

    group_size = len(rollout.pred_token_lists)
    src_onset = prompt.src_start_frames[0] if prompt.src_start_frames else 0
    src_end = prompt.src_end_frames[-1] if prompt.src_end_frames else int(prompt.src_codes.shape[1])
    ref_text_str = (
        sp_decode_fn(prompt.ref_tokens)
        if (sp_decode_fn is not None and prompt.ref_tokens)
        else None
    )

    rewards: List[float] = []
    bleu_scores: List[float] = []
    laal_secs: List[float] = []
    audio_scores: List[float] = []

    for g in range(group_size):
        pred_toks = rollout.pred_token_lists[g]
        pred_frames = rollout.pred_token_frames[g]
        v_len = rollout.valid_lengths[g]

        # 1. 번역 품질 보상 (Token BLEU + 선택적 디코딩 문자열 문자 n-gram BLEU + 다중 발화 Process 보상)
        tok_bleu = compute_smoothed_bleu(pred_toks, prompt.ref_tokens)
        if sp_decode_fn is not None and ref_text_str is not None and pred_toks:
            pred_str = sp_decode_fn(pred_toks)
            char_bleu = compute_smoothed_bleu(
                [ch for ch in pred_str if not ch.isspace()],
                [ch for ch in ref_text_str if not ch.isspace()],
            )
            final_bleu = 0.6 * tok_bleu + 0.4 * char_bleu
        else:
            final_bleu = tok_bleu

        if len(prompt.src_end_frames) > 1 and len(prompt.ref_token_groups) == len(prompt.src_end_frames):
            milestone_bleus: List[float] = []
            cum_ref: List[int] = []
            for u_end, ref_grp in zip(prompt.src_end_frames[:-1], prompt.ref_token_groups[:-1]):
                cum_ref.extend(ref_grp)
                if not cum_ref:
                    continue
                horizon = u_end + int(1.5 * cfg.frame_rate)
                prefix_pred = [t for t, f in zip(pred_toks, pred_frames) if f <= horizon]
                milestone_bleus.append(compute_smoothed_bleu(prefix_pred, cum_ref))
            if milestone_bleus:
                qual_score = 0.7 * final_bleu + 0.3 * (sum(milestone_bleus) / len(milestone_bleus))
            else:
                qual_score = final_bleu
        else:
            qual_score = final_bleu

        # 2. 동시통역 지연시간(LAAL) 계산 및 보상 해킹(빈 출력 / 발화 전 추측) 방지 페널티
        laal_sec = compute_laal_latency_sec(
            tok_frames=pred_frames,
            src_start_frame=src_onset,
            src_end_frame=src_end,
            ref_len=len(prompt.ref_tokens),
            frame_rate=cfg.frame_rate,
        )

        anti_cheat_penalty = 0.0
        min_allowed_frame = src_onset + max(0, min_wait_frames)
        for f_pos in pred_frames:
            if f_pos < min_allowed_frame:
                anti_cheat_penalty += 0.15 * float(min_allowed_frame - f_pos) / cfg.frame_rate

        if len(prompt.ref_tokens) > 0 and len(pred_toks) == 0:
            anti_cheat_penalty += 0.50
        if not rollout.eos_reached[g]:
            anti_cheat_penalty += 0.10

        # 3. 타깃 오디오 의미 코드북(cb=0) 붕괴 방지 점수 + 선택적 외부 보상 훅
        sem_audio = rollout.rollout_codes[g, 1, :v_len]
        valid_sem = sem_audio[sem_audio >= 0]
        if valid_sem.numel() > 2:
            transitions = float((valid_sem[1:] != valid_sem[:-1]).float().mean().item())
            # 단일 토큰 무한 반복(0.0) 억제
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
            bleu_weight * qual_score
            - latency_weight * laal_sec
            + audio_weight * a_score
            - anti_cheat_penalty
        )
        rewards.append(total_r)
        bleu_scores.append(qual_score)
        laal_secs.append(laal_sec)
        audio_scores.append(a_score)

    r_tensor = torch.tensor(rewards, dtype=torch.float32)
    r_mean = r_tensor.mean()
    r_std = r_tensor.std(correction=0)
    if float(r_std.item()) < 1e-6:
        advantages = torch.zeros_like(r_tensor)
    else:
        advantages = (r_tensor - r_mean) / (r_std + 1e-5)

    stats = {
        "reward_mean": float(r_mean.item()),
        "reward_std": float(r_std.item()),
        "bleu_mean": float(sum(bleu_scores) / max(1, len(bleu_scores))),
        "laal_sec_mean": float(sum(laal_secs) / max(1, len(laal_secs))),
        "audio_score_mean": float(sum(audio_scores) / max(1, len(audio_scores))),
        "eos_rate": float(sum(1.0 for flag in rollout.eos_reached if flag) / max(1, group_size)),
    }
    return r_tensor, advantages, stats


def compute_grpo_loss(
    policy_model: "EDarchModel",
    rollout_codes: "torch.Tensor",
    sim_bins: "torch.Tensor",
    advantages: "torch.Tensor",
    old_lp: "EDarchSequenceLogProbs",
    ref_lp: Optional["EDarchSequenceLogProbs"],
    clip_eps: float = 0.2,
    kl_coef: float = 0.02,
    audio_pg_weight: float = 0.15,
) -> Tuple["torch.Tensor", dict[str, float]]:
    import torch

    new_lp = policy_model.compute_token_log_probs(
        rollout_codes, speaker_sim_bins=sim_bins
    )
    bsz = rollout_codes.shape[0]
    adv_t = advantages.to(device=rollout_codes.device, dtype=torch.float32).view(bsz, 1, 1)

    # 1. 텍스트(Inner Monologue) 스트림 Clipped Policy Gradient + Schulman k3 KL
    t_mask_f = new_lp.text_mask.float()
    t_denom = t_mask_f.sum(dim=(1, 2)).clamp_min(1.0)
    t_log_ratio = (new_lp.text_log_probs - old_lp.text_log_probs.detach()).clamp(-20.0, 20.0)
    t_ratio = torch.exp(t_log_ratio)
    t_surr1 = t_ratio * adv_t
    t_surr2 = torch.clamp(t_ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t
    t_pg_per_tok = -torch.minimum(t_surr1, t_surr2) * t_mask_f
    text_pg_loss = (t_pg_per_tok.sum(dim=(1, 2)) / t_denom).mean()

    if ref_lp is not None and kl_coef > 0.0:
        t_ref_diff = (ref_lp.text_log_probs.to(new_lp.text_log_probs.device) - new_lp.text_log_probs).clamp(
            -20.0, 20.0
        )
        t_kl_per_tok = (torch.exp(t_ref_diff) - t_ref_diff - 1.0) * t_mask_f
        text_kl = (t_kl_per_tok.sum(dim=(1, 2)) / t_denom).mean()
    else:
        text_kl = text_pg_loss * 0.0

    # 2. 타깃 오디오 코드북(Depformer) 스트림 (의미 코드북 cb=0 중심 가중치)
    dep_q = policy_model.train_dep_q
    cb_w = torch.ones(dep_q, device=rollout_codes.device, dtype=torch.float32)
    cb_w[0] = 5.0
    cb_w = cb_w / cb_w.mean()
    a_weight_mask = new_lp.audio_mask.float() * cb_w.view(1, dep_q, 1)
    a_denom = a_weight_mask.sum(dim=(1, 2)).clamp_min(1.0)

    if audio_pg_weight > 0.0:
        a_log_ratio = (new_lp.audio_log_probs - old_lp.audio_log_probs.detach()).clamp(-20.0, 20.0)
        a_ratio = torch.exp(a_log_ratio)
        a_surr1 = a_ratio * adv_t
        a_surr2 = torch.clamp(a_ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t
        a_pg_per_tok = -torch.minimum(a_surr1, a_surr2) * a_weight_mask
        audio_pg_loss = (a_pg_per_tok.sum(dim=(1, 2)) / a_denom).mean()
    else:
        audio_pg_loss = text_pg_loss * 0.0

    if ref_lp is not None and kl_coef > 0.0:
        a_ref_diff = (
            ref_lp.audio_log_probs.to(new_lp.audio_log_probs.device) - new_lp.audio_log_probs
        ).clamp(-20.0, 20.0)
        a_kl_per_tok = (torch.exp(a_ref_diff) - a_ref_diff - 1.0) * a_weight_mask
        audio_kl = (a_kl_per_tok.sum(dim=(1, 2)) / a_denom).mean()
    else:
        audio_kl = text_pg_loss * 0.0

    audio_kl_scale = kl_coef * max(audio_pg_weight, 0.1) if kl_coef > 0.0 else 0.0
    total_loss = (
        text_pg_loss
        + kl_coef * text_kl
        + audio_pg_weight * audio_pg_loss
        + audio_kl_scale * audio_kl
    )

    metrics = {
        "loss": float(total_loss.detach().cpu().item()),
        "text_pg": float(text_pg_loss.detach().cpu().item()),
        "audio_pg": float(audio_pg_loss.detach().cpu().item()),
        "text_kl": float(text_kl.detach().cpu().item()),
        "audio_kl": float(audio_kl.detach().cpu().item()),
    }
    return total_loss, metrics


def _atomic_save_checkpoint(checkpoint: dict, save_path: Path, max_retries: int = 5) -> None:
    import torch

    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
    torch.save(checkpoint, tmp_path)
    for attempt in range(max_retries):
        try:
            os.replace(tmp_path, save_path)
            return
        except PermissionError:
            if attempt == max_retries - 1:
                raise
            time.sleep(0.1 * (2**attempt))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EDarch Stage 2 GRPO(Group Relative Policy Optimization) 동시통역 강화학습 스크립트"
    )
    parser.add_argument("--venv-dir", type=Path, default=get_default_venv_dir())
    parser.add_argument("--no-venv", action="store_true")
    parser.add_argument("--torch-index-url", type=str, default=DEFAULT_CUDA_INDEX_URL)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--reinstall", action="store_true")

    parser.add_argument(
        "--init-from",
        type=Path,
        default=get_default_checkpoint_path(),
        help="Stage 1 지도학습 체크포인트 경로 (미존재 시 --preset으로 초기화)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="중단된 Stage 2 GRPO 체크포인트에서 이어서 학습 재개",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=get_default_grpo_checkpoint_path(),
        help="Stage 2 GRPO 체크포인트 저장 경로 (기본값: EDarch_grpo_ckpt.pt)",
    )
    parser.add_argument("--preset", choices=("1b", "2b"), default="1b")
    parser.add_argument("--ko-vocab-size", type=int, default=None)
    parser.add_argument(
        "--grad-checkpointing",
        action="store_true",
        help="정책 업데이트 역전파 시 Activation Checkpointing 활성화 (VRAM 절감)",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="전처리된 .pt 폴더 경로 (미지정 시 합성 프롬프트로 데모 GRPO 루프 검증)",
    )
    parser.add_argument("--crop-src-frames", type=int, default=450, help="프롬프트 최대 소스 프레임 수")
    parser.add_argument("--max-tail-sec", type=float, default=6.0, help="롤아웃 시 원문 종료 후 최대 무음 꼬리(초)")
    parser.add_argument("--sp-model", type=Path, default=None, help="문자열 BLEU 및 로그 출력용 SentencePiece .model")
    parser.add_argument(
        "--hooks-module",
        type=str,
        default=None,
        help="사용자 정의 custom_reward_fn(prompt, rollout, group_idx) 제공 파이썬 모듈명",
    )

    # GRPO 핵심 하이퍼파라미터
    parser.add_argument("--group-size", type=int, default=4, help="프롬프트당 샘플링할 롤아웃 수 G (>= 2)")
    parser.add_argument("--ppo-epochs", type=int, default=1, help="롤아웃 그룹당 내부 정책 업데이트 횟수")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO/GRPO 중요도 비율 클리핑 범위")
    parser.add_argument("--kl-coef", type=float, default=0.02, help="Stage 1 기준 모델 대비 KL 페널티 계수")
    parser.add_argument(
        "--ref-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="KL 계산용 기준 모델(ref_model) 배치 디바이스 (기본값: cpu로 GPU VRAM 절약)",
    )
    parser.add_argument("--audio-pg-weight", type=float, default=0.15, help="타깃 오디오 코드북 정책 경사 가중치")
    parser.add_argument("--rollout-cfg-coef", type=float, default=1.0, help="롤아웃 샘플링 CFG 계수 (Hibiki-Zero 기본: 1.0)")
    parser.add_argument("--temp-text", type=float, default=0.7, help="롤아웃 텍스트 탐색 온도")
    parser.add_argument("--temp-audio", type=float, default=0.8, help="롤아웃 오디오 탐색 온도")
    parser.add_argument("--top-k-text", type=int, default=50, help="롤아웃 텍스트 Top-k")
    parser.add_argument("--top-k-audio", type=int, default=250, help="롤아웃 오디오 Top-k")

    # 동시통역 보상 가중치
    parser.add_argument("--reward-bleu-weight", type=float, default=1.0, help="BLEU 번역 품질 보상 가중치")
    parser.add_argument("--reward-latency-weight", type=float, default=0.15, help="초당 LAAL 지연시간 페널티 가중치")
    parser.add_argument("--reward-audio-weight", type=float, default=0.10, help="오디오 연속성/외부 음성 보상 가중치")
    parser.add_argument("--min-wait-frames", type=int, default=3, help="발화 시작 후 최소 청취 프레임 수(조기 추측 페널티 기준)")

    parser.add_argument("--epochs", type=int, default=1, help="GRPO 학습 에폭 수")
    parser.add_argument("--steps-per-epoch", type=int, default=5, help="데모 모드 시 에폭당 프롬프트 수")
    parser.add_argument("--demo-src-frames", type=int, default=16, help="데모 모드 시 소스 오디오 프레임 수")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation 프롬프트 수")
    parser.add_argument("--lr", type=float, default=1e-5, help="GRPO 학습률 (기본값: 1e-5)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_grpo_training(args: argparse.Namespace) -> None:
    import torch

    from EDarch_from_scratch import EDarchConfig, EDarchModel, MiniMimiCodec
    from train import _build_adamw_param_groups, _build_warmup_cosine_scheduler

    if args.group_size < 2:
        raise ValueError(f"--group-size({args.group_size})는 그룹 상대 우위 계산을 위해 2 이상이어야 합니다.")
    if args.ppo_epochs < 1:
        raise ValueError(f"--ppo-epochs({args.ppo_epochs})는 1 이상이어야 합니다.")
    if args.grad_accum_steps < 1:
        raise ValueError(f"--grad-accum-steps({args.grad_accum_steps})는 1 이상이어야 합니다.")

    device = setup_device(cpu_only=args.cpu, seed=args.seed, tag="GRPO")

    start_epoch = 1
    global_step = 0
    resume_ckpt = None
    codec_type = "external_mimi" if args.data_dir is not None else "mini_demo"
    saved_codec_state = None

    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(f"--resume 체크포인트를 찾을 수 없습니다: {args.resume}")
        print(f"[GRPO] Stage 2 체크포인트에서 재개: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        cfg = EDarchConfig.from_dict(resume_ckpt["config"])
        cfg.train_on_source_audio = False
        codec_type = resume_ckpt.get("codec_type", codec_type)
        saved_codec_state = resume_ckpt.get("codec_state_dict")
        policy_model = EDarchModel(cfg)
        policy_model.load_state_dict_for_inference(resume_ckpt["model_state_dict"])
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        global_step = int(resume_ckpt.get("global_step", 0))
    elif args.init_from is not None and args.init_from.exists():
        print(f"[GRPO] Stage 1 초기 정책 로드: {args.init_from}")
        init_ckpt = torch.load(args.init_from, map_location="cpu", weights_only=True)
        cfg = EDarchConfig.from_dict(init_ckpt["config"])
        cfg.train_on_source_audio = False
        codec_type = init_ckpt.get("codec_type", codec_type)
        saved_codec_state = init_ckpt.get("codec_state_dict")
        policy_model = EDarchModel(cfg)
        policy_model.load_state_dict_for_inference(init_ckpt["model_state_dict"])
        del init_ckpt
    else:
        print(
            f"[GRPO] Stage 1 체크포인트({args.init_from}) 없음 -> '{args.preset}' 프리셋으로 새 정책 초기화"
        )
        cfg = EDarchConfig.from_preset(args.preset)
        cfg.train_on_source_audio = False
        if args.ko_vocab_size is not None:
            cfg.text_card = args.ko_vocab_size
        policy_model = EDarchModel(cfg)

    ref_model: Optional[EDarchModel] = None
    ref_device = torch.device("cuda" if (args.ref_device == "cuda" and device.type == "cuda") else "cpu")
    if args.kl_coef > 0.0:
        ref_source_path = (
            args.init_from
            if (args.resume is not None and args.init_from is not None and args.init_from.exists())
            else None
        )
        if ref_source_path is not None:
            ref_ckpt = torch.load(ref_source_path, map_location="cpu", weights_only=True)
            ref_model = EDarchModel(copy.deepcopy(cfg))
            ref_model.load_state_dict_for_inference(ref_ckpt["model_state_dict"])
            del ref_ckpt
        else:
            ref_model = copy.deepcopy(policy_model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        if ref_device.type == "cuda":
            ref_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            ref_model = ref_model.to(device=ref_device, dtype=ref_dtype)
        else:
            ref_model = ref_model.to(device=ref_device)
        print(f"[GRPO] 기준 모델(ref_model) 준비 완료 (device={ref_device}, kl_coef={args.kl_coef})")

    policy_model = policy_model.to(device)
    if args.grad_checkpointing:
        policy_model.set_gradient_checkpointing(True)
        print("[GRPO] 정책 모델 Gradient Checkpointing 활성화됨")

    sp_decode_fn: Optional[Callable[[List[int]], str]] = None
    if args.sp_model is not None:
        if not args.sp_model.exists():
            raise FileNotFoundError(f"SentencePiece 모델을 찾을 수 없습니다: {args.sp_model}")
        import sentencepiece as spm

        sp_proc = spm.SentencePieceProcessor()
        sp_proc.Load(str(args.sp_model))
        piece_limit = int(sp_proc.GetPieceSize())
        sp_decode_fn = lambda ids: sp_proc.DecodeIds([i for i in ids if 0 <= i < piece_limit])

    hooks_mod = importlib.import_module(args.hooks_module) if args.hooks_module else None
    custom_reward_fn = getattr(hooks_mod, "custom_reward_fn", None) if hooks_mod else None

    demo_codec: Optional[MiniMimiCodec] = None
    dataset: Optional[GRPOPromptDataset] = None
    if args.data_dir is not None:
        dataset = GRPOPromptDataset(
            data_dir=args.data_dir,
            crop_src_frames=args.crop_src_frames,
            cfg=cfg,
        )
        prompts_per_epoch = len(dataset)
        print(f"[GRPO] 프롬프트 데이터셋 로드 완료: {prompts_per_epoch}개 샘플 (G={args.group_size})")
    else:
        demo_codec = MiniMimiCodec(cfg).to(device).eval()
        if saved_codec_state is not None:
            demo_codec.load_state_dict(saved_codec_state)
        prompts_per_epoch = args.steps_per_epoch
        print(
            f"[GRPO] --data-dir 미지정 -> 합성 데모 프롬프트 모드 "
            f"(epochs={args.epochs}, steps/epoch={prompts_per_epoch}, G={args.group_size})"
        )

    updates_per_epoch = max(1, math.ceil(prompts_per_epoch / args.grad_accum_steps)) * args.ppo_epochs
    total_updates = updates_per_epoch * max(1, args.epochs)
    warmup_steps = (
        args.warmup_steps if args.warmup_steps is not None else max(1, int(total_updates * 0.05))
    )

    param_groups = _build_adamw_param_groups(policy_model, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    scheduler = _build_warmup_cosine_scheduler(
        optimizer, total_updates=total_updates, warmup_steps=warmup_steps
    )
    use_amp = (not args.no_amp) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    if resume_ckpt is not None:
        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if resume_ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        if resume_ckpt.get("scaler_state_dict") is not None and scaler.is_enabled():
            scaler.load_state_dict(resume_ckpt["scaler_state_dict"])
        del resume_ckpt

    if start_epoch > args.epochs:
        print(f"[GRPO] 이미 epoch {start_epoch - 1}까지 완료되어 있습니다 (--epochs={args.epochs}).")
        return

    very_good_id = cfg.num_speaker_sim_bins - 1
    cond_sim_bins = torch.full((args.group_size,), very_good_id, dtype=torch.long, device=device)
    last_metrics: dict[str, float] = {}

    for epoch in range(start_epoch, args.epochs + 1):
        if dataset is not None:
            perm = torch.randperm(len(dataset)).tolist()
            prompt_iter = (dataset[i] for i in perm)
        else:
            assert demo_codec is not None

            def _demo_prompts():
                num_samples = args.demo_src_frames * demo_codec.hop_length
                with torch.no_grad():
                    sil_wav = torch.zeros(1, 1, 4 * demo_codec.hop_length, device=device)
                    sil_c = demo_codec.encode(sil_wav)[0, :, -1:].cpu()
                for s_i in range(prompts_per_epoch):
                    with torch.no_grad():
                        wav = torch.randn(1, 1, num_samples, device=device) * 0.1
                        src_c = demo_codec.encode(wav)[0].cpu()
                    ref_toks = torch.randint(4, min(64, cfg.text_card), (6,)).tolist()
                    yield GRPOPromptSample(
                        name=f"demo_{s_i:04d}",
                        src_codes=src_c,
                        silence_code=sil_c,
                        ref_tokens=ref_toks,
                        src_start_frames=[0],
                        src_end_frames=[args.demo_src_frames],
                        ref_token_groups=[ref_toks],
                        sim_bin=very_good_id,
                    )

            prompt_iter = _demo_prompts()

        epoch_reward_sum = 0.0
        epoch_laal_sum = 0.0
        epoch_bleu_sum = 0.0
        rollout_buffer: List[
            Tuple["torch.Tensor", "torch.Tensor", "EDarchSequenceLogProbs", Optional["EDarchSequenceLogProbs"]]
        ] = []
        optimizer.zero_grad(set_to_none=True)

        for step, prompt in enumerate(prompt_iter, start=1):
            rollout = rollout_group_trajectories(
                model=policy_model,
                prompt=prompt,
                group_size=args.group_size,
                max_tail_sec=args.max_tail_sec,
                rollout_cfg_coef=args.rollout_cfg_coef,
                temp_text=args.temp_text,
                temp_audio=args.temp_audio,
                top_k_text=args.top_k_text,
                top_k_audio=args.top_k_audio,
                device=device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
            )

            _, advantages, r_stats = evaluate_rollout_group_rewards(
                prompt=prompt,
                rollout=rollout,
                cfg=cfg,
                bleu_weight=args.reward_bleu_weight,
                latency_weight=args.reward_latency_weight,
                audio_weight=args.reward_audio_weight,
                min_wait_frames=args.min_wait_frames,
                sp_decode_fn=sp_decode_fn,
                custom_reward_fn=custom_reward_fn,
            )

            policy_model.eval()
            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    old_lp = policy_model.compute_token_log_probs(
                        rollout.rollout_codes, speaker_sim_bins=cond_sim_bins
                    )

                ref_lp = None
                if ref_model is not None and args.kl_coef > 0.0:
                    ref_codes = rollout.rollout_codes.to(ref_device)
                    ref_bins = cond_sim_bins.to(ref_device)
                    use_ref_amp = use_amp and ref_device.type == "cuda"
                    with torch.amp.autocast(
                        device_type=ref_device.type, dtype=amp_dtype, enabled=use_ref_amp
                    ):
                        ref_lp = ref_model.compute_token_log_probs(
                            ref_codes, speaker_sim_bins=ref_bins
                        )

            rollout_buffer.append((rollout.rollout_codes, advantages, old_lp, ref_lp))
            epoch_reward_sum += r_stats["reward_mean"]
            epoch_bleu_sum += r_stats["bleu_mean"]
            epoch_laal_sum += r_stats["laal_sec_mean"]

            is_last_prompt = step == prompts_per_epoch
            should_step_opt = (len(rollout_buffer) >= args.grad_accum_steps) or is_last_prompt

            step_loss_metrics: dict[str, float] = {}
            if should_step_opt:
                policy_model.train()
                accum_denom = float(len(rollout_buffer))
                for _ in range(args.ppo_epochs):
                    optimizer.zero_grad(set_to_none=True)
                    for buf_codes, buf_adv, buf_old_lp, buf_ref_lp in rollout_buffer:
                        with torch.amp.autocast(
                            device_type=device.type, dtype=amp_dtype, enabled=use_amp
                        ):
                            loss, loss_metrics = compute_grpo_loss(
                                policy_model=policy_model,
                                rollout_codes=buf_codes,
                                sim_bins=cond_sim_bins,
                                advantages=buf_adv,
                                old_lp=buf_old_lp,
                                ref_lp=buf_ref_lp,
                                clip_eps=args.clip_eps,
                                kl_coef=args.kl_coef,
                                audio_pg_weight=args.audio_pg_weight,
                            )
                            scaled_loss = loss / accum_denom

                        if scaler.is_enabled():
                            scaler.scale(scaled_loss).backward()
                        else:
                            scaled_loss.backward()
                        step_loss_metrics = loss_metrics

                    if scaler.is_enabled():
                        old_scale = scaler.get_scale()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                        stepped = scaler.get_scale() >= old_scale
                    else:
                        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.grad_clip)
                        optimizer.step()
                        stepped = True
                    if stepped:
                        scheduler.step()
                        global_step += 1
                    optimizer.zero_grad(set_to_none=True)

                rollout_buffer.clear()
                last_metrics = {**r_stats, **step_loss_metrics}

                if step == 1 or step % 5 == 0 or is_last_prompt:
                    cur_lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"  [Epoch {epoch}/{args.epochs} | Prompt {step}/{prompts_per_epoch} (opt={global_step})] "
                        f"R={r_stats['reward_mean']:.3f}±{r_stats['reward_std']:.3f} | "
                        f"BLEU={r_stats['bleu_mean']:.3f} | LAAL={r_stats['laal_sec_mean']:.2f}s | "
                        f"KL={step_loss_metrics['text_kl']:.4f} | loss={step_loss_metrics['loss']:.4f} | lr={cur_lr:.2e}"
                    )

        n_p = float(max(1, prompts_per_epoch))
        print(
            f"[Epoch {epoch}/{args.epochs} 완료] "
            f"avg_reward={epoch_reward_sum / n_p:.4f} | "
            f"avg_BLEU={epoch_bleu_sum / n_p:.4f} | "
            f"avg_LAAL={epoch_laal_sum / n_p:.2f}s"
        )

        checkpoint = {
            "config": asdict(cfg),
            "stage": "grpo_stage2",
            "codec_type": codec_type,
            "model_state_dict": policy_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
            "epoch": epoch,
            "global_step": global_step,
            "last_metrics": last_metrics,
        }
        if demo_codec is not None:
            checkpoint["codec_state_dict"] = demo_codec.state_dict()
        elif saved_codec_state is not None:
            checkpoint["codec_state_dict"] = saved_codec_state
        _atomic_save_checkpoint(checkpoint, args.save_path)
        print(f"  -> Stage 2 GRPO 체크포인트 저장됨: {args.save_path}")

    print(f"\n[Done] Stage 2 GRPO 강화학습 완료 (최종 체크포인트: {args.save_path})")


def main() -> None:
    args = parse_args()
    ensure_venv_and_reexec(
        script_path=Path(__file__),
        venv_dir=args.venv_dir,
        torch_index_url=args.torch_index_url,
        cpu_only=args.cpu,
        reinstall=args.reinstall,
        no_venv=args.no_venv,
    )
    run_grpo_training(args)


if __name__ == "__main__":
    main()
