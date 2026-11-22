"""Stage 2 Hibiki-Zero 방식 GRPO(Group Relative Policy Optimization) 동시통역 강화학습 모듈."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict, dataclass
import importlib
import math
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch

from libs.codec import MiniMimiCodec
from libs.config import EDarchConfig
from libs.env import atomic_torch_save, setup_device
from libs.model import EDarchModel, EDarchSequenceLogProbs
from libs.optim import (
    AMPContext,
    build_adamw_param_groups,
    build_warmup_cosine_scheduler,
    resolve_amp_context,
    step_optimizer_with_clip,
)
from libs.streaming import EDarchStreamingGenerator
from libs.tokenizer import load_korean_tokenizer


@dataclass
class GRPOPromptSample:
    """단일 소스 오디오 프롬프트와 발화 경계 메타데이터 및 정답 토큰을 담는 데이터클래스입니다."""

    name: str
    src_codes: torch.Tensor
    silence_code: torch.Tensor
    ref_tokens: List[int]
    src_start_frames: List[int]
    src_end_frames: List[int]
    ref_token_groups: List[List[int]]
    sim_bin: int


@dataclass
class RolloutBatch:
    """단일 프롬프트에 대해 병렬 샘플링된 G개 스트리밍 통역 궤적 묶음 데이터클래스입니다."""

    rollout_codes: torch.Tensor
    pred_token_lists: List[List[int]]
    pred_token_frames: List[List[int]]
    eos_reached: List[bool]
    valid_lengths: List[int]
    eos_suppress_steps: int = 0


def _move_seq_lp(
    lp: EDarchSequenceLogProbs, target_device: torch.device
) -> EDarchSequenceLogProbs:
    """시퀀스 로그 확률 데이터클래스 내부 텐서들을 지정된 디바이스로 분리·이동합니다."""
    return EDarchSequenceLogProbs(
        text_log_probs=lp.text_log_probs.detach().to(target_device, non_blocking=True),
        text_mask=lp.text_mask.detach().to(target_device, non_blocking=True),
        audio_log_probs=lp.audio_log_probs.detach().to(target_device, non_blocking=True),
        audio_mask=lp.audio_mask.detach().to(target_device, non_blocking=True),
    )


def _average_metric_dicts(dicts: Sequence[dict[str, float]]) -> dict[str, float]:
    """여러 스텝에서 수집된 스칼라 메트릭 딕셔너리들의 키별 산술 평균을 반환합니다."""
    if not dicts:
        return {}
    keys = dicts[0].keys()
    denom = float(len(dicts))
    return {k: sum(float(d.get(k, 0.0)) for d in dicts) / denom for k in keys}


def _infer_source_span(
    en_codes: torch.Tensor,
    raw_item: dict[str, Any],
    crop_src_frames: int,
) -> Tuple[List[int], List[int], int]:
    """메타데이터 또는 무음 꼬리 역추적을 통해 영어 원문 발화의 시작·종료 프레임 구간을 결정합니다."""
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
            if e_f > crop_src_frames and len(clipped_ends) >= 1:
                src_len = clipped_ends[-1]
                break
            clipped_starts.append(s_f)
            clipped_ends.append(min(e_f, src_len))
        if clipped_ends:
            return clipped_starts, clipped_ends, max(1, src_len)

    tail_sem = int(en_codes[0, -1].item())
    active_end = total_t
    while active_end > 1 and int(en_codes[0, active_end - 1].item()) == tail_sem:
        active_end -= 1
    if active_end <= 1:
        active_end = max(1, min(total_t, int(total_t * 0.6)))
    src_len = max(1, min(crop_src_frames, active_end))
    return [0], [src_len], src_len


class GRPOPromptDataset:
    """전처리된 `.pt` 파일들에서 GRPO 롤아웃용 소스 오디오 코드와 정답 토큰 그룹을 추출하는 데이터셋입니다."""

    def __init__(
        self,
        data_dir: Path,
        crop_src_frames: int,
        cfg: EDarchConfig,
        default_sim_bin: Optional[int] = None,
    ):
        """프롬프트 `.pt` 파일 목록을 수집하고 첫 번째 프롬프트의 유효성을 검증합니다."""
        if crop_src_frames < 1:
            raise ValueError(f"crop_src_frames({crop_src_frames})는 1 이상이어야 합니다.")
        self.files = sorted(data_dir.rglob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"{data_dir}에서 .pt 파일을 찾지 못했습니다.")
        self.crop_src_frames = crop_src_frames
        self.cfg = cfg
        self.default_sim_bin = (
            cfg.num_speaker_sim_bins - 1 if default_sim_bin is None else int(default_sim_bin)
        )

        _ = self[0]

    def __len__(self) -> int:
        """전체 프롬프트 개수를 반환합니다."""
        return len(self.files)

    def __getitem__(self, idx: int) -> GRPOPromptSample:
        """지정된 인덱스의 `.pt` 샘플에서 소스 오디오 스트림과 정답 한국어 토큰 시퀀스를 반환합니다."""
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

        sim_bin = int(item.get("sim_bin", self.default_sim_bin))
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


def compute_laal_latency_sec(
    tok_frames: Sequence[int],
    src_start_frame: int,
    src_end_frame: int,
    ref_len: int,
    frame_rate: float = 12.5,
    horizon_frames: Optional[int] = None,
) -> float:
    """방출된 번역 토큰 타임스탬프로부터 원문 종료 후 침묵 지연까지 반영한 실시간 LAAL(초)을 계산합니다."""
    src_dur = max(1, src_end_frame - src_start_frame)
    empty_frames = (
        max(src_dur, horizon_frames - src_start_frame)
        if horizon_frames is not None
        else src_dur
    )
    if not tok_frames:
        return float(empty_frames) / frame_rate

    pred_len = len(tok_frames)
    gamma = float(max(pred_len, max(1, ref_len))) / float(src_dur)
    lags: List[float] = []

    for i, f_pos in enumerate(tok_frames):
        elapsed_frames = float(max(0, f_pos - src_start_frame + 1))
        ideal_read = float(i) / max(1e-6, gamma)
        lags.append(max(0.0, elapsed_frames - ideal_read))
        if f_pos + 1 >= src_end_frame:
            break

    if not lags:
        return float(empty_frames) / frame_rate
    return float(sum(lags) / len(lags)) / frame_rate


def rollout_group_trajectories(
    model: EDarchModel,
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
    """단일 소스 프롬프트에 대해 G개의 독립적인 스트리밍 동시통역 궤적을 배치 병렬로 생성합니다."""
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
            sim_bin=prompt.sim_bin,
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

        return RolloutBatch(
            rollout_codes=rollout_codes,
            pred_token_lists=pred_token_lists,
            pred_token_frames=pred_token_frames,
            eos_reached=eos_reached_flags,
            valid_lengths=valid_lengths,
            eos_suppress_steps=min(src_len, total_roll_t),
        )
    finally:
        if was_training:
            model.train()


def evaluate_rollout_group_rewards(
    prompt: GRPOPromptSample,
    rollout: RolloutBatch,
    cfg: EDarchConfig,
    bleu_weight: float = 1.0,
    latency_weight: float = 0.15,
    audio_weight: float = 0.10,
    min_wait_frames: int = 3,
    sp_decode_fn: Optional[Callable[[List[int]], str]] = None,
    custom_reward_fn: Optional[Callable[[GRPOPromptSample, RolloutBatch, int], float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """번역 정확도(BLEU), 동시통역 지연시간(LAAL), 조기 추측 억제 페널티를 결합하여 그룹 정규화 우위를 산출합니다."""
    group_size = len(rollout.pred_token_lists)
    src_onset = prompt.src_start_frames[0] if prompt.src_start_frames else 0
    src_end = (
        prompt.src_end_frames[-1]
        if prompt.src_end_frames
        else int(prompt.src_codes.shape[1])
    )
    rollout_horizon = int(rollout.rollout_codes.shape[2])
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

        if len(prompt.src_end_frames) > 1 and len(prompt.ref_token_groups) == len(
            prompt.src_end_frames
        ):
            milestone_bleus: List[float] = []
            cum_ref: List[int] = []
            for u_end, ref_grp in zip(
                prompt.src_end_frames[:-1], prompt.ref_token_groups[:-1]
            ):
                cum_ref.extend(ref_grp)
                if not cum_ref:
                    continue
                horizon = u_end + int(1.5 * cfg.frame_rate)
                prefix_pred = [t for t, f in zip(pred_toks, pred_frames) if f <= horizon]
                milestone_bleus.append(compute_smoothed_bleu(prefix_pred, cum_ref))
            if milestone_bleus:
                qual_score = 0.7 * final_bleu + 0.3 * (
                    sum(milestone_bleus) / len(milestone_bleus)
                )
            else:
                qual_score = final_bleu
        else:
            qual_score = final_bleu

        laal_sec = compute_laal_latency_sec(
            tok_frames=pred_frames,
            src_start_frame=src_onset,
            src_end_frame=src_end,
            ref_len=len(prompt.ref_tokens),
            frame_rate=cfg.frame_rate,
            horizon_frames=rollout_horizon,
        )

        anti_cheat_penalty = 0.0
        min_allowed_frame = src_onset + max(0, min_wait_frames)
        for f_pos in pred_frames:
            if f_pos < min_allowed_frame:
                anti_cheat_penalty += (
                    0.15 * float(min_allowed_frame - f_pos) / cfg.frame_rate
                )

        if len(prompt.ref_tokens) > 0 and len(pred_toks) == 0:
            anti_cheat_penalty += 0.50
        if not rollout.eos_reached[g]:
            anti_cheat_penalty += 0.10

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
        "eos_rate": float(
            sum(1.0 for flag in rollout.eos_reached if flag) / max(1, group_size)
        ),
    }
    return r_tensor, advantages, stats


def compute_grpo_loss(
    policy_model: EDarchModel,
    rollout_codes: torch.Tensor,
    sim_bins: torch.Tensor,
    advantages: torch.Tensor,
    old_lp: EDarchSequenceLogProbs,
    ref_lp: Optional[EDarchSequenceLogProbs],
    clip_eps: float = 0.2,
    kl_coef: float = 0.02,
    audio_pg_weight: float = 0.15,
    temp_text: float = 1.0,
    temp_audio: float = 1.0,
    top_k_text: int = 0,
    top_k_audio: int = 0,
    eos_suppress_steps: int = 0,
    cfg_coef: float = 1.0,
) -> Tuple[torch.Tensor, dict[str, float]]:
    """텍스트 및 오디오 코드북에 대한 Clipped Surrogate 정책 손실과 Schulman k3 KL 페널티를 계산합니다."""
    new_lp = policy_model.compute_token_log_probs(
        rollout_codes,
        speaker_sim_bins=sim_bins,
        temp_text=temp_text,
        temp_audio=temp_audio,
        top_k_text=top_k_text,
        top_k_audio=top_k_audio,
        eos_suppress_steps=eos_suppress_steps,
        cfg_coef=cfg_coef,
    )
    dev = rollout_codes.device
    bsz = rollout_codes.shape[0]
    adv_t = advantages.to(device=dev, dtype=torch.float32).view(bsz, 1, 1)

    t_mask_f = new_lp.text_mask.float()
    t_denom = t_mask_f.sum(dim=(1, 2)).clamp_min(1.0)
    old_t_lp = old_lp.text_log_probs.to(dev, non_blocking=True).detach()
    t_log_ratio = (new_lp.text_log_probs - old_t_lp).clamp(-20.0, 20.0)
    t_ratio = torch.exp(t_log_ratio)
    t_surr1 = t_ratio * adv_t
    t_surr2 = torch.clamp(t_ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t
    t_pg_per_tok = -torch.minimum(t_surr1, t_surr2) * t_mask_f
    text_pg_loss = (t_pg_per_tok.sum(dim=(1, 2)) / t_denom).mean()

    if ref_lp is not None and kl_coef > 0.0:
        t_ref_diff = (
            ref_lp.text_log_probs.to(dev, non_blocking=True) - new_lp.text_log_probs
        ).clamp(-20.0, 20.0)
        t_kl_per_tok = (torch.exp(t_ref_diff) - t_ref_diff - 1.0) * t_mask_f
        text_kl = (t_kl_per_tok.sum(dim=(1, 2)) / t_denom).mean()
    else:
        text_kl = text_pg_loss * 0.0

    dep_q = policy_model.train_dep_q
    cb_w = torch.ones(dep_q, device=dev, dtype=torch.float32)
    cb_w[0] = 5.0
    cb_w = cb_w / cb_w.mean()
    a_weight_mask = new_lp.audio_mask.float() * cb_w.view(1, dep_q, 1)
    a_denom = a_weight_mask.sum(dim=(1, 2)).clamp_min(1.0)

    if audio_pg_weight > 0.0:
        old_a_lp = old_lp.audio_log_probs.to(dev, non_blocking=True).detach()
        a_log_ratio = (new_lp.audio_log_probs - old_a_lp).clamp(-20.0, 20.0)
        a_ratio = torch.exp(a_log_ratio)
        a_surr1 = a_ratio * adv_t
        a_surr2 = torch.clamp(a_ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t
        a_pg_per_tok = -torch.minimum(a_surr1, a_surr2) * a_weight_mask
        audio_pg_loss = (a_pg_per_tok.sum(dim=(1, 2)) / a_denom).mean()
    else:
        audio_pg_loss = text_pg_loss * 0.0

    if ref_lp is not None and kl_coef > 0.0:
        a_ref_diff = (
            ref_lp.audio_log_probs.to(dev, non_blocking=True) - new_lp.audio_log_probs
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


@torch.no_grad()
def _evaluate_grpo_validation(
    policy_model: EDarchModel,
    val_dataset: GRPOPromptDataset,
    max_prompts: int,
    args: argparse.Namespace,
    cfg: EDarchConfig,
    device: torch.device,
    amp: AMPContext,
    sp_decode_fn: Optional[Callable[[List[int]], str]],
    custom_reward_fn: Optional[Callable[[GRPOPromptSample, RolloutBatch, int], float]],
) -> dict[str, float]:
    """검증 프롬프트들에 대해 스트리밍 롤아웃 보상·BLEU·LAAL 평균을 평가합니다."""
    was_training = policy_model.training
    policy_model.eval()
    eval_n = min(len(val_dataset), max(1, max_prompts)) if max_prompts > 0 else len(val_dataset)
    collected: List[dict[str, float]] = []
    try:
        for idx in range(eval_n):
            prompt = val_dataset[idx]
            rollout = rollout_group_trajectories(
                model=policy_model,
                prompt=prompt,
                group_size=max(1, min(2, args.group_size)),
                max_tail_sec=args.max_tail_sec,
                rollout_cfg_coef=args.rollout_cfg_coef,
                temp_text=max(0.2, args.temp_text * 0.5),
                temp_audio=args.temp_audio,
                top_k_text=args.top_k_text,
                top_k_audio=args.top_k_audio,
                device=device,
                use_amp=amp.enabled,
                amp_dtype=amp.dtype,
            )
            _, _, r_stats = evaluate_rollout_group_rewards(
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
            collected.append(r_stats)
    finally:
        if was_training:
            policy_model.train()

    avg = _average_metric_dicts(collected)
    return {
        "val_reward_mean": avg.get("reward_mean", 0.0),
        "val_bleu_mean": avg.get("bleu_mean", 0.0),
        "val_laal_sec_mean": avg.get("laal_sec_mean", 0.0),
    }


def run_grpo_training(args: argparse.Namespace) -> None:
    """Stage 2 GRPO 동시통역 강화학습 전체 롤아웃·보상·정책 최적화 루프를 실행합니다."""
    if args.group_size < 2:
        raise ValueError(
            f"--group-size({args.group_size})는 그룹 상대 우위 계산을 위해 2 이상이어야 합니다."
        )
    if args.ppo_epochs < 1:
        raise ValueError(f"--ppo-epochs({args.ppo_epochs})는 1 이상이어야 합니다.")
    if args.grad_accum_steps < 1:
        raise ValueError(f"--grad-accum-steps({args.grad_accum_steps})는 1 이상이어야 합니다.")

    device = setup_device(cpu_only=args.cpu, seed=args.seed, tag="GRPO")
    cpu_dev = torch.device("cpu")

    start_epoch = 1
    global_step = 0
    best_val_reward = float("-inf")
    resume_ckpt = None
    codec_type = "external_mimi" if args.data_dir is not None else "mini_demo"
    saved_codec_state = None
    default_sim_bin: Optional[int] = None

    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(f"--resume 체크포인트를 찾을 수 없습니다: {args.resume}")
        print(f"[GRPO] Stage 2 체크포인트에서 재개: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        cfg = EDarchConfig.from_dict(resume_ckpt["config"])
        cfg.train_on_source_audio = False
        codec_type = resume_ckpt.get("codec_type", codec_type)
        saved_codec_state = resume_ckpt.get("codec_state_dict")
        if resume_ckpt.get("default_sim_bin") is not None:
            default_sim_bin = int(resume_ckpt["default_sim_bin"])
        best_val_reward = float(resume_ckpt.get("best_val_reward", float("-inf")))
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
        if init_ckpt.get("default_sim_bin") is not None:
            default_sim_bin = int(init_ckpt["default_sim_bin"])
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

    if default_sim_bin is None:
        default_sim_bin = cfg.num_speaker_sim_bins - 1

    ref_model: Optional[EDarchModel] = None
    ref_device = torch.device(
        "cuda" if (args.ref_device == "cuda" and device.type == "cuda") else "cpu"
    )
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
        sp_proc = load_korean_tokenizer(args.sp_model)
        piece_limit = int(sp_proc.GetPieceSize())
        sp_decode_fn = lambda ids: sp_proc.DecodeIds([i for i in ids if 0 <= i < piece_limit])

    hooks_mod = importlib.import_module(args.hooks_module) if args.hooks_module else None
    custom_reward_fn = getattr(hooks_mod, "custom_reward_fn", None) if hooks_mod else None

    demo_codec: Optional[MiniMimiCodec] = None
    dataset: Optional[GRPOPromptDataset] = None
    val_dataset: Optional[GRPOPromptDataset] = None
    if args.data_dir is not None:
        dataset = GRPOPromptDataset(
            data_dir=args.data_dir,
            crop_src_frames=args.crop_src_frames,
            cfg=cfg,
            default_sim_bin=default_sim_bin,
        )
        prompts_per_epoch = len(dataset)
        print(f"[GRPO] 프롬프트 데이터셋 로드 완료: {prompts_per_epoch}개 샘플 (G={args.group_size})")

        val_dir = getattr(args, "val_data_dir", None)
        if val_dir is not None and val_dir.exists() and any(val_dir.rglob("*.pt")):
            val_dataset = GRPOPromptDataset(
                data_dir=val_dir,
                crop_src_frames=args.crop_src_frames,
                cfg=cfg,
                default_sim_bin=default_sim_bin,
            )
            print(f"[GRPO] 검증(val) 프롬프트 로드 완료: {len(val_dataset)}개 샘플 ({val_dir})")
    else:
        demo_codec = MiniMimiCodec(cfg).to(device).eval()
        if saved_codec_state is not None:
            demo_codec.load_state_dict(saved_codec_state)
        prompts_per_epoch = args.steps_per_epoch
        print(
            f"[GRPO] --data-dir 미지정 -> 합성 데모 프롬프트 모드 "
            f"(epochs={args.epochs}, steps/epoch={prompts_per_epoch}, G={args.group_size})"
        )

    updates_per_epoch = (
        max(1, math.ceil(prompts_per_epoch / args.grad_accum_steps)) * args.ppo_epochs
    )
    total_updates = updates_per_epoch * max(1, args.epochs)
    warmup_steps = (
        args.warmup_steps
        if args.warmup_steps is not None
        else max(1, int(total_updates * 0.05))
    )

    param_groups = build_adamw_param_groups(policy_model, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    scheduler = build_warmup_cosine_scheduler(
        optimizer, total_updates=total_updates, warmup_steps=warmup_steps
    )
    amp = resolve_amp_context(device=device, no_amp=args.no_amp)

    if resume_ckpt is not None:
        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if resume_ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        if resume_ckpt.get("scaler_state_dict") is not None and amp.scaler.is_enabled():
            amp.scaler.load_state_dict(resume_ckpt["scaler_state_dict"])
        del resume_ckpt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if start_epoch > args.epochs:
        print(f"[GRPO] 이미 epoch {start_epoch - 1}까지 완료되어 있습니다 (--epochs={args.epochs}).")
        return

    last_metrics: dict[str, float] = {}
    best_save_path = args.save_path.with_name(f"{args.save_path.stem}_best{args.save_path.suffix}")

    for epoch in range(start_epoch, args.epochs + 1):
        if dataset is not None:
            perm = torch.randperm(len(dataset)).tolist()
            prompt_iter = (dataset[i] for i in perm)
        else:
            assert demo_codec is not None

            def _demo_prompts():
                """외부 데이터셋 없이 GRPO 루프를 검증하기 위한 합성 프롬프트를 생성합니다."""
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
                        sim_bin=int(default_sim_bin),
                    )

            prompt_iter = _demo_prompts()

        epoch_reward_sum = 0.0
        epoch_laal_sum = 0.0
        epoch_bleu_sum = 0.0
        rollout_buffer: List[
            Tuple[
                torch.Tensor,
                torch.Tensor,
                int,
                torch.Tensor,
                EDarchSequenceLogProbs,
                Optional[EDarchSequenceLogProbs],
            ]
        ] = []
        accum_r_stats: List[dict[str, float]] = []
        optimizer.zero_grad(set_to_none=True)

        for step, prompt in enumerate(prompt_iter, start=1):
            prompt_sim_bins = torch.full(
                (args.group_size,), prompt.sim_bin, dtype=torch.long, device=device
            )
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
                use_amp=amp.enabled,
                amp_dtype=amp.dtype,
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
                with torch.amp.autocast(
                    device_type=device.type, dtype=amp.dtype, enabled=amp.enabled
                ):
                    old_lp_gpu = policy_model.compute_token_log_probs(
                        rollout.rollout_codes,
                        speaker_sim_bins=prompt_sim_bins,
                        temp_text=args.temp_text,
                        temp_audio=args.temp_audio,
                        top_k_text=args.top_k_text,
                        top_k_audio=args.top_k_audio,
                        eos_suppress_steps=rollout.eos_suppress_steps,
                        cfg_coef=args.rollout_cfg_coef,
                    )
                old_lp_cpu = _move_seq_lp(old_lp_gpu, cpu_dev)
                del old_lp_gpu

                ref_lp_cpu: Optional[EDarchSequenceLogProbs] = None
                if ref_model is not None and args.kl_coef > 0.0:
                    ref_codes = rollout.rollout_codes.to(ref_device)
                    ref_bins = prompt_sim_bins.to(ref_device)
                    use_ref_amp = amp.enabled and ref_device.type == "cuda"
                    with torch.amp.autocast(
                        device_type=ref_device.type, dtype=amp.dtype, enabled=use_ref_amp
                    ):
                        ref_lp_raw = ref_model.compute_token_log_probs(
                            ref_codes,
                            speaker_sim_bins=ref_bins,
                            temp_text=args.temp_text,
                            temp_audio=args.temp_audio,
                            top_k_text=args.top_k_text,
                            top_k_audio=args.top_k_audio,
                            eos_suppress_steps=rollout.eos_suppress_steps,
                            cfg_coef=args.rollout_cfg_coef,
                        )
                    ref_lp_cpu = _move_seq_lp(ref_lp_raw, cpu_dev)
                    del ref_lp_raw, ref_codes, ref_bins

            rollout_buffer.append(
                (
                    rollout.rollout_codes.detach().cpu(),
                    prompt_sim_bins.detach().cpu(),
                    rollout.eos_suppress_steps,
                    advantages.detach().cpu(),
                    old_lp_cpu,
                    ref_lp_cpu,
                )
            )
            accum_r_stats.append(r_stats)
            epoch_reward_sum += r_stats["reward_mean"]
            epoch_bleu_sum += r_stats["bleu_mean"]
            epoch_laal_sum += r_stats["laal_sec_mean"]

            is_last_prompt = step == prompts_per_epoch
            should_step_opt = (len(rollout_buffer) >= args.grad_accum_steps) or is_last_prompt

            if should_step_opt:
                all_zero_adv = args.kl_coef <= 0.0 and all(
                    float(b_adv.abs().max().item()) < 1e-8
                    for _, _, _, b_adv, _, _ in rollout_buffer
                )
                avg_r_stats = _average_metric_dicts(accum_r_stats)

                if all_zero_adv:
                    step_loss_metrics = {
                        "loss": 0.0,
                        "text_pg": 0.0,
                        "audio_pg": 0.0,
                        "text_kl": 0.0,
                        "audio_kl": 0.0,
                    }
                else:
                    policy_model.train()
                    accum_denom = float(len(rollout_buffer))
                    collected_losses: List[dict[str, float]] = []
                    for _ in range(args.ppo_epochs):
                        optimizer.zero_grad(set_to_none=True)
                        collected_losses.clear()
                        for (
                            buf_codes_cpu,
                            buf_bins_cpu,
                            buf_eos_sup,
                            buf_adv_cpu,
                            buf_old_lp_cpu,
                            buf_ref_lp_cpu,
                        ) in rollout_buffer:
                            buf_codes = buf_codes_cpu.to(device, non_blocking=True)
                            buf_bins = buf_bins_cpu.to(device, non_blocking=True)
                            with torch.amp.autocast(
                                device_type=device.type, dtype=amp.dtype, enabled=amp.enabled
                            ):
                                loss, loss_metrics = compute_grpo_loss(
                                    policy_model=policy_model,
                                    rollout_codes=buf_codes,
                                    sim_bins=buf_bins,
                                    advantages=buf_adv_cpu,
                                    old_lp=buf_old_lp_cpu,
                                    ref_lp=buf_ref_lp_cpu,
                                    clip_eps=args.clip_eps,
                                    kl_coef=args.kl_coef,
                                    audio_pg_weight=args.audio_pg_weight,
                                    temp_text=args.temp_text,
                                    temp_audio=args.temp_audio,
                                    top_k_text=args.top_k_text,
                                    top_k_audio=args.top_k_audio,
                                    eos_suppress_steps=buf_eos_sup,
                                    cfg_coef=args.rollout_cfg_coef,
                                )
                                scaled_loss = loss / accum_denom

                            if amp.scaler.is_enabled():
                                amp.scaler.scale(scaled_loss).backward()
                            else:
                                scaled_loss.backward()
                            collected_losses.append(loss_metrics)
                            del buf_codes, buf_bins

                        if step_optimizer_with_clip(
                            model=policy_model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=amp.scaler,
                            grad_clip=args.grad_clip,
                        ):
                            global_step += 1

                    step_loss_metrics = _average_metric_dicts(collected_losses)

                rollout_buffer.clear()
                accum_r_stats.clear()
                last_metrics = {**avg_r_stats, **step_loss_metrics}

                if step == 1 or step % 5 == 0 or is_last_prompt:
                    cur_lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"  [Epoch {epoch}/{args.epochs} | Prompt {step}/{prompts_per_epoch} (opt={global_step})] "
                        f"R={avg_r_stats['reward_mean']:.3f}±{avg_r_stats['reward_std']:.3f} | "
                        f"BLEU={avg_r_stats['bleu_mean']:.3f} | LAAL={avg_r_stats['laal_sec_mean']:.2f}s | "
                        f"KL={step_loss_metrics['text_kl']:.4f} | loss={step_loss_metrics['loss']:.4f} | lr={cur_lr:.2e}"
                    )

        n_p = float(max(1, prompts_per_epoch))
        val_msg = ""
        improved_val = False
        if val_dataset is not None:
            val_stats = _evaluate_grpo_validation(
                policy_model=policy_model,
                val_dataset=val_dataset,
                max_prompts=getattr(args, "max_val_prompts", 32),
                args=args,
                cfg=cfg,
                device=device,
                amp=amp,
                sp_decode_fn=sp_decode_fn,
                custom_reward_fn=custom_reward_fn,
            )
            last_metrics.update(val_stats)
            val_r = val_stats["val_reward_mean"]
            val_msg = (
                f" | val_R={val_r:.4f} (val_BLEU={val_stats['val_bleu_mean']:.4f}, "
                f"val_LAAL={val_stats['val_laal_sec_mean']:.2f}s)"
            )
            if val_r > best_val_reward:
                best_val_reward = val_r
                improved_val = True

        print(
            f"[Epoch {epoch}/{args.epochs} 완료] "
            f"avg_reward={epoch_reward_sum / n_p:.4f} | "
            f"avg_BLEU={epoch_bleu_sum / n_p:.4f} | "
            f"avg_LAAL={epoch_laal_sum / n_p:.2f}s{val_msg}"
        )

        checkpoint = {
            "config": asdict(cfg),
            "stage": "grpo_stage2",
            "codec_type": codec_type,
            "rollout_cfg_coef": float(args.rollout_cfg_coef),
            "default_sim_bin": int(default_sim_bin),
            "best_val_reward": best_val_reward,
            "model_state_dict": policy_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": amp.scaler.state_dict() if amp.scaler.is_enabled() else None,
            "epoch": epoch,
            "global_step": global_step,
            "last_metrics": last_metrics,
        }
        if demo_codec is not None:
            checkpoint["codec_state_dict"] = demo_codec.state_dict()
        elif saved_codec_state is not None:
            checkpoint["codec_state_dict"] = saved_codec_state
        atomic_torch_save(checkpoint, args.save_path)
        print(f"  -> Stage 2 GRPO 체크포인트 저장됨: {args.save_path}")
        if improved_val:
            atomic_torch_save(checkpoint, best_save_path)
            print(f"  -> 최고 검증 성능(val_R={best_val_reward:.4f}) 갱신 저장됨: {best_save_path}")

    print(f"\n[Done] Stage 2 GRPO 강화학습 완료 (최종 체크포인트: {args.save_path})")
