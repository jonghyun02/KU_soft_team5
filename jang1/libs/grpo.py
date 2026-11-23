"""arXiv:2309.11093 Stage 2 Singability GRPO(Clipped Surrogate PG + Schulman k3 KL) 손실 및 학습 루프 모듈."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import importlib
import math
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from libs.codec import MiniMimiCodec
from libs.config import LyricTransConfig
from libs.env import atomic_torch_save, setup_device
from libs.grpo_rollout import (
    GRPOPromptDataset,
    GRPOPromptSample,
    RolloutBatch,
    compute_smoothed_bleu,
    evaluate_rollout_group_rewards,
    rollout_group_trajectories,
)
from libs.model import LyricSequenceLogProbs, LyricTransModel
from libs.optim import (
    AMPContext,
    build_adamw_param_groups,
    build_warmup_cosine_scheduler,
    resolve_amp_context,
    step_optimizer_with_clip,
)
from libs.tokenizer import load_lyric_tokenizer


def _move_seq_lp(
    lp: LyricSequenceLogProbs, target_device: torch.device
) -> LyricSequenceLogProbs:
    """시퀀스 로그 확률 데이터클래스 내부 텐서들을 지정된 디바이스로 분리·이동합니다."""
    return LyricSequenceLogProbs(
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


def compute_grpo_loss(
    policy_model: LyricTransModel,
    rollout_codes: torch.Tensor,
    sim_bins: torch.Tensor,
    advantages: torch.Tensor,
    old_lp: LyricSequenceLogProbs,
    ref_lp: Optional[LyricSequenceLogProbs],
    target_syllables: Optional[torch.Tensor] = None,
    section_types: Optional[torch.Tensor] = None,
    repeat_group_ids: Optional[torch.Tensor] = None,
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
    """가사 텍스트 및 보컬 코드북에 대한 Clipped Surrogate 정책 손실과 Schulman k3 KL 페널티를 계산합니다."""
    new_lp = policy_model.compute_token_log_probs(
        rollout_codes,
        speaker_sim_bins=sim_bins,
        target_syllables=target_syllables,
        section_types=section_types,
        repeat_group_ids=repeat_group_ids,
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
    policy_model: LyricTransModel,
    val_dataset: GRPOPromptDataset,
    max_prompts: int,
    args: argparse.Namespace,
    cfg: LyricTransConfig,
    device: torch.device,
    amp: AMPContext,
    sp_decode_fn: Optional[Callable[[List[int]], str]],
    custom_reward_fn: Optional[Callable[[GRPOPromptSample, RolloutBatch, int], float]],
) -> dict[str, float]:
    """검증 섹션 프롬프트들에 대해 싱어빌리티 복합 보상·BLEU·SCD·SSM·Rhyme 평균을 평가합니다."""
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
                group_size=max(2, min(2, args.group_size)),
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
                syllable_weight=args.reward_syllable_weight,
                ssm_weight=args.reward_ssm_weight,
                rhyme_weight=args.reward_rhyme_weight,
                audio_weight=args.reward_audio_weight,
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
        "val_scd_mean": avg.get("scd_mean", 0.0),
        "val_ssm_mean": avg.get("ssm_score_mean", 0.0),
        "val_rhyme_mean": avg.get("rhyme_score_mean", 0.0),
    }


def run_grpo_training(args: argparse.Namespace) -> None:
    """Stage 2 Singability GRPO 강화학습 전체 롤아웃·복합 보상·정책 최적화 루프를 실행합니다."""
    if args.group_size < 2:
        raise ValueError(
            f"--group-size({args.group_size})는 그룹 상대 우위 계산을 위해 2 이상이어야 합니다."
        )
    if args.ppo_epochs < 1:
        raise ValueError(f"--ppo-epochs({args.ppo_epochs})는 1 이상이어야 합니다.")
    if args.grad_accum_steps < 1:
        raise ValueError(f"--grad-accum-steps({args.grad_accum_steps})는 1 이상이어야 합니다.")

    device = setup_device(cpu_only=args.cpu, seed=args.seed, tag="SingableGRPO")
    cpu_dev = torch.device("cpu")

    start_epoch = 1
    global_step = 0
    best_val_reward = float("-inf")
    resume_ckpt = None
    codec_type = "mini_demo"
    if args.data_dir is not None:
        manifest_p = args.data_dir.parent / "preprocessing_manifest.json"
        if manifest_p.exists():
            import json

            m_info = json.loads(manifest_p.read_text(encoding="utf-8"))
            if str(m_info.get("codec", "mini_demo")) != "mini_demo":
                codec_type = "external_mimi"
    saved_codec_state = None
    default_sim_bin: Optional[int] = None

    vocab_arg = getattr(args, "vocab_size", None)
    direction = getattr(args, "direction", "ko2en")
    src_lang = "en" if direction == "en2ko" else "ko"
    tgt_lang = "ko" if direction == "en2ko" else "en"

    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(f"--resume 체크포인트를 찾을 수 없습니다: {args.resume}")
        print(f"[GRPO] Stage 2 체크포인트에서 재개: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        cfg = LyricTransConfig.from_dict(resume_ckpt["config"])
        cfg.train_on_source_audio = False
        codec_type = resume_ckpt.get("codec_type", codec_type)
        saved_codec_state = resume_ckpt.get("codec_state_dict")
        if resume_ckpt.get("default_sim_bin") is not None:
            default_sim_bin = int(resume_ckpt["default_sim_bin"])
        best_val_reward = float(resume_ckpt.get("best_val_reward", float("-inf")))
        policy_model = LyricTransModel(cfg)
        policy_model.load_state_dict_for_inference(resume_ckpt["model_state_dict"])
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        global_step = int(resume_ckpt.get("global_step", 0))
    elif args.init_from is not None and args.init_from.exists():
        print(f"[GRPO] Stage 1 초기 정책 로드: {args.init_from}")
        init_ckpt = torch.load(args.init_from, map_location="cpu", weights_only=True)
        cfg = LyricTransConfig.from_dict(init_ckpt["config"])
        cfg.train_on_source_audio = False
        codec_type = init_ckpt.get("codec_type", codec_type)
        saved_codec_state = init_ckpt.get("codec_state_dict")
        if init_ckpt.get("default_sim_bin") is not None:
            default_sim_bin = int(init_ckpt["default_sim_bin"])
        policy_model = LyricTransModel(cfg)
        policy_model.load_state_dict_for_inference(init_ckpt["model_state_dict"])
        del init_ckpt
    else:
        print(
            f"[GRPO] Stage 1 체크포인트({args.init_from}) 없음 -> '{args.preset}' 프리셋으로 새 정책 초기화"
        )
        cfg = LyricTransConfig.from_preset(
            args.preset, source_lang=src_lang, target_lang=tgt_lang
        )
        cfg.train_on_source_audio = False
        if vocab_arg is not None:
            cfg.text_card = vocab_arg
        elif args.sp_model is not None and args.sp_model.exists():
            sp_tmp = load_lyric_tokenizer(args.sp_model)
            cfg.text_card = int(sp_tmp.GetPieceSize())
        policy_model = LyricTransModel(cfg)

    if default_sim_bin is None:
        default_sim_bin = cfg.num_speaker_sim_bins - 1

    ref_model: Optional[LyricTransModel] = None
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
            ref_model = LyricTransModel(copy.deepcopy(cfg))
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

    sp_decode_fn: Optional[Callable[[List[int]], str]] = None
    if args.sp_model is not None and args.sp_model.exists():
        sp_proc = load_lyric_tokenizer(args.sp_model)
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
        print(f"[GRPO] 섹션 프롬프트 데이터셋 로드 완료: {prompts_per_epoch}개 샘플 (G={args.group_size})")

        val_dir = getattr(args, "val_data_dir", None)
        if val_dir is not None and val_dir.exists() and any(val_dir.rglob("*.pt")):
            val_dataset = GRPOPromptDataset(
                data_dir=val_dir,
                crop_src_frames=args.crop_src_frames,
                cfg=cfg,
                default_sim_bin=default_sim_bin,
            )
            print(f"[GRPO] 검증(val) 섹션 프롬프트 로드 완료: {len(val_dataset)}개 샘플")
    else:
        demo_codec = MiniMimiCodec(cfg).to(device).eval()
        if saved_codec_state is not None:
            demo_codec.load_state_dict(saved_codec_state)
        prompts_per_epoch = args.steps_per_epoch
        print(
            f"[GRPO] --data-dir 미지정 -> K-pop 섹션 합성 데모 프롬프트 모드 "
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

    if start_epoch > args.epochs:
        print(f"[GRPO] 이미 epoch {start_epoch - 1}까지 완료되어 있습니다.")
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
                """외부 데이터셋 없이 Singability GRPO 루프를 검증하기 위한 합성 섹션 프롬프트를 생성합니다."""
                num_samples = args.demo_src_frames * demo_codec.hop_length
                with torch.no_grad():
                    sil_wav = torch.zeros(1, 1, 4 * demo_codec.hop_length, device=device)
                    sil_c = demo_codec.encode(sil_wav)[0, :, -1:].cpu()
                demo_src_lines = [
                    "빛나는 별빛 아래 너와 나",
                    "영원히 멈추지 않을 이 노래",
                    "빛나는 별빛 아래 너와 나",
                ]
                demo_tgt_lines = [
                    "Under the shining starlight you and me",
                    "Singing this melody forever free",
                    "Under the shining starlight you and me",
                ]
                demo_ssm = [
                    [1.0, 0.1, 1.0],
                    [0.1, 1.0, 0.1],
                    [1.0, 0.1, 1.0],
                ]
                for s_i in range(prompts_per_epoch):
                    with torch.no_grad():
                        wav = torch.randn(1, 1, num_samples, device=device) * 0.1
                        src_c = demo_codec.encode(wav)[0].cpu()
                    ref_toks = torch.randint(4, min(64, cfg.text_card), (9,)).tolist()
                    f1 = max(1, args.demo_src_frames // 3)
                    f2 = max(f1 + 1, (2 * args.demo_src_frames) // 3)
                    yield GRPOPromptSample(
                        name=f"demo_chorus_{s_i:04d}",
                        src_codes=src_c,
                        silence_code=sil_c,
                        ref_tokens=ref_toks,
                        src_start_frames=[0, f1, f2],
                        src_end_frames=[f1, f2, args.demo_src_frames],
                        ref_token_groups=[ref_toks[:3], ref_toks[3:6], ref_toks[6:]],
                        sim_bin=int(default_sim_bin),
                        section_type_id=2,
                        src_lines=demo_src_lines,
                        tgt_lines=demo_tgt_lines,
                        target_syllables=[10, 10, 10],
                        repeat_group_ids=[0, 1, 0],
                        src_semantic_ssm=demo_ssm,
                        src_phoneme_ssm=demo_ssm,
                    )

            prompt_iter = _demo_prompts()

        epoch_reward_sum = 0.0
        epoch_bleu_sum = 0.0
        epoch_scd_sum = 0.0
        epoch_ssm_sum = 0.0
        epoch_rhyme_sum = 0.0
        rollout_buffer: List[
            Tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                Optional[torch.Tensor],
                Optional[torch.Tensor],
                int,
                torch.Tensor,
                LyricSequenceLogProbs,
                Optional[LyricSequenceLogProbs],
            ]
        ] = []
        accum_r_stats: List[dict[str, float]] = []
        optimizer.zero_grad(set_to_none=True)

        for step, prompt in enumerate(prompt_iter, start=1):
            prompt_sim_bins = torch.full(
                (args.group_size,), prompt.sim_bin, dtype=torch.long, device=device
            )
            prompt_sec_types = torch.full(
                (args.group_size,), prompt.section_type_id, dtype=torch.long, device=device
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

            roll_t = rollout.rollout_codes.shape[2]
            avg_syl = (
                int(round(sum(prompt.target_syllables) / len(prompt.target_syllables)))
                if prompt.target_syllables
                else 8
            )
            if (
                rollout.frame_target_syllables is not None
                and rollout.frame_target_syllables.numel() >= roll_t
            ):
                batch_syl = (
                    rollout.frame_target_syllables[:roll_t]
                    .unsqueeze(0)
                    .expand(args.group_size, -1)
                    .contiguous()
                    .to(device)
                )
            else:
                batch_syl = torch.full(
                    (args.group_size, roll_t), avg_syl, dtype=torch.long, device=device
                )

            if (
                rollout.frame_repeat_group_ids is not None
                and rollout.frame_repeat_group_ids.numel() >= roll_t
            ):
                batch_rep = (
                    rollout.frame_repeat_group_ids[:roll_t]
                    .unsqueeze(0)
                    .expand(args.group_size, -1)
                    .contiguous()
                    .to(device)
                )
            else:
                batch_rep = torch.zeros(
                    (args.group_size, roll_t), dtype=torch.long, device=device
                )

            _, advantages, r_stats = evaluate_rollout_group_rewards(
                prompt=prompt,
                rollout=rollout,
                cfg=cfg,
                bleu_weight=args.reward_bleu_weight,
                syllable_weight=args.reward_syllable_weight,
                ssm_weight=args.reward_ssm_weight,
                rhyme_weight=args.reward_rhyme_weight,
                audio_weight=args.reward_audio_weight,
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
                        target_syllables=batch_syl,
                        section_types=prompt_sec_types,
                        repeat_group_ids=batch_rep,
                        temp_text=args.temp_text,
                        temp_audio=args.temp_audio,
                        top_k_text=args.top_k_text,
                        top_k_audio=args.top_k_audio,
                        eos_suppress_steps=rollout.eos_suppress_steps,
                        cfg_coef=args.rollout_cfg_coef,
                    )
                old_lp_cpu = _move_seq_lp(old_lp_gpu, cpu_dev)
                del old_lp_gpu

                ref_lp_cpu: Optional[LyricSequenceLogProbs] = None
                if ref_model is not None and args.kl_coef > 0.0:
                    ref_codes = rollout.rollout_codes.to(ref_device)
                    ref_bins = prompt_sim_bins.to(ref_device)
                    ref_secs = prompt_sec_types.to(ref_device)
                    ref_syl = batch_syl.to(ref_device)
                    ref_rep = batch_rep.to(ref_device)
                    use_ref_amp = amp.enabled and ref_device.type == "cuda"
                    with torch.amp.autocast(
                        device_type=ref_device.type, dtype=amp.dtype, enabled=use_ref_amp
                    ):
                        ref_lp_raw = ref_model.compute_token_log_probs(
                            ref_codes,
                            speaker_sim_bins=ref_bins,
                            target_syllables=ref_syl,
                            section_types=ref_secs,
                            repeat_group_ids=ref_rep,
                            temp_text=args.temp_text,
                            temp_audio=args.temp_audio,
                            top_k_text=args.top_k_text,
                            top_k_audio=args.top_k_audio,
                            eos_suppress_steps=rollout.eos_suppress_steps,
                            cfg_coef=args.rollout_cfg_coef,
                        )
                    ref_lp_cpu = _move_seq_lp(ref_lp_raw, cpu_dev)
                    del ref_lp_raw, ref_codes, ref_bins, ref_secs, ref_syl, ref_rep

            rollout_buffer.append(
                (
                    rollout.rollout_codes.detach().cpu(),
                    prompt_sim_bins.detach().cpu(),
                    prompt_sec_types.detach().cpu(),
                    batch_syl.detach().cpu(),
                    batch_rep.detach().cpu(),
                    rollout.eos_suppress_steps,
                    advantages.detach().cpu(),
                    old_lp_cpu,
                    ref_lp_cpu,
                )
            )
            accum_r_stats.append(r_stats)
            epoch_reward_sum += r_stats["reward_mean"]
            epoch_bleu_sum += r_stats["bleu_mean"]
            epoch_scd_sum += r_stats["scd_mean"]
            epoch_ssm_sum += r_stats["ssm_score_mean"]
            epoch_rhyme_sum += r_stats["rhyme_score_mean"]

            is_last_prompt = step == prompts_per_epoch
            should_step_opt = (len(rollout_buffer) >= args.grad_accum_steps) or is_last_prompt

            if should_step_opt:
                all_zero_adv = args.kl_coef <= 0.0 and all(
                    float(b_adv.abs().max().item()) < 1e-8
                    for _, _, _, _, _, _, b_adv, _, _ in rollout_buffer
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
                            buf_secs_cpu,
                            buf_syl_cpu,
                            buf_rep_cpu,
                            buf_eos_sup,
                            buf_adv_cpu,
                            buf_old_lp_cpu,
                            buf_ref_lp_cpu,
                        ) in rollout_buffer:
                            buf_codes = buf_codes_cpu.to(device, non_blocking=True)
                            buf_bins = buf_bins_cpu.to(device, non_blocking=True)
                            buf_secs = buf_secs_cpu.to(device, non_blocking=True)
                            buf_syl = buf_syl_cpu.to(device, non_blocking=True) if buf_syl_cpu is not None else None
                            buf_rep = buf_rep_cpu.to(device, non_blocking=True) if buf_rep_cpu is not None else None
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
                                    target_syllables=buf_syl,
                                    section_types=buf_secs,
                                    repeat_group_ids=buf_rep,
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
                            del buf_codes, buf_bins, buf_secs, buf_syl, buf_rep

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
                        f"BLEU={avg_r_stats['bleu_mean']:.3f} | SCD={avg_r_stats['scd_mean']:.3f} | "
                        f"SSM={avg_r_stats['ssm_score_mean']:.3f} | Rhyme={avg_r_stats['rhyme_score_mean']:.3f} | "
                        f"KL={step_loss_metrics['text_kl']:.4f} | loss={step_loss_metrics['loss']:.4f} | lr={cur_lr:.2e}"
                    )

        n_p = float(max(1, prompts_per_epoch))
        val_msg = ""
        improved_val = False
        if val_dataset is not None:
            val_stats = _evaluate_grpo_validation(
                policy_model=policy_model,
                val_dataset=val_dataset,
                max_prompts=getattr(args, "max_val_prompts", 16),
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
                f" | val_R={val_r:.4f} (BLEU={val_stats['val_bleu_mean']:.3f}, "
                f"SCD={val_stats['val_scd_mean']:.3f}, SSM={val_stats['val_ssm_mean']:.3f})"
            )
            if val_r > best_val_reward:
                best_val_reward = val_r
                improved_val = True

        print(
            f"[Epoch {epoch}/{args.epochs} 완료] "
            f"avg_R={epoch_reward_sum / n_p:.4f} | "
            f"BLEU={epoch_bleu_sum / n_p:.3f} | "
            f"SCD={epoch_scd_sum / n_p:.3f} | "
            f"SSM={epoch_ssm_sum / n_p:.3f} | "
            f"Rhyme={epoch_rhyme_sum / n_p:.3f}{val_msg}"
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
        print(f"  -> Stage 2 Singability GRPO 체크포인트 저장됨: {args.save_path}")
        if improved_val:
            atomic_torch_save(checkpoint, best_save_path)
            print(f"  -> 최고 검증 성능(val_R={best_val_reward:.4f}) 갱신 저장됨: {best_save_path}")

    print(f"\n[Done] Stage 2 Singability GRPO 강화학습 완료 (최종 체크포인트: {args.save_path})")


__all__ = [
    "GRPOPromptSample",
    "RolloutBatch",
    "GRPOPromptDataset",
    "compute_smoothed_bleu",
    "rollout_group_trajectories",
    "evaluate_rollout_group_rewards",
    "compute_grpo_loss",
    "run_grpo_training",
]
