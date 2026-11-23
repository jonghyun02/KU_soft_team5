"""Stage 2 GRPO 프롬프트 데이터셋, 배치 병렬 스트리밍 롤아웃 및 BLEU·LAAL 복합 보상 평가 모듈."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch

from libs.config import EDarchConfig
from libs.model import EDarchModel
from libs.streaming import EDarchStreamingGenerator
from libs.training import list_preprocessed_pt_files, load_and_validate_pt_item


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
        self.files = list_preprocessed_pt_files(data_dir)
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
        item, codes, raw_sim_bin = load_and_validate_pt_item(
            path=path,
            expected_total_streams=self.cfg.total_streams,
            default_sim_bin=self.default_sim_bin,
            num_speaker_sim_bins=self.cfg.num_speaker_sim_bins,
            validate_ranges=False,
        )
        q = self.cfg.dep_q

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

        sim_bin = max(0, min(self.cfg.num_speaker_sim_bins - 1, raw_sim_bin))

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
