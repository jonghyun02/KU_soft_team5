"""Stage 2 Hibiki-Zero 방식 GRPO(Group Relative Policy Optimization) 정책 손실 및 학습 루프 모듈."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import importlib
import math
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from libs.codec import MiniMimiCodec
from libs.config import EDarchConfig
from libs.env import atomic_torch_save, setup_device
from libs.grpo_rollout import (
    GRPOPromptDataset,
    GRPOPromptSample,
    RolloutBatch,
    compute_laal_latency_sec,
    compute_smoothed_bleu,
    evaluate_rollout_group_rewards,
    rollout_group_trajectories,
)
from libs.model import EDarchModel, EDarchSequenceLogProbs
from libs.optim import (
    AMPContext,
    apply_bf16_master_weights_if_supported,
    build_optimizer,
    build_warmup_cosine_scheduler,
    load_optimizer_state_safely,
    resolve_amp_context,
    resolve_optim_type,
    sanitize_optimizer_state_for_save,
    step_optimizer_with_clip,
)
from libs.tokenizer import load_korean_tokenizer


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

    policy_model = apply_bf16_master_weights_if_supported(
        policy_model,
        device=device,
        enabled=bool(getattr(args, "bf16_weights", False)),
        tag="GRPO",
    )
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

    optim_type = resolve_optim_type(
        optim_arg=getattr(args, "optim", "adamw"),
        use_8bit_adam=bool(getattr(args, "use_8bit_adam", False)),
        device=device,
        tag="GRPO",
    )
    optimizer = build_optimizer(
        model=policy_model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optim_type=optim_type,
        tag="GRPO",
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_updates=total_updates,
        warmup_steps=warmup_steps,
        min_total_ratio=float(getattr(args, "lr_min_ratio", 0.02)),
        mode="max",
        patience=int(getattr(args, "lr_patience", 1000)),
        epoch_patience=int(getattr(args, "lr_epoch_patience", 1)),
        factor=float(getattr(args, "lr_factor", 0.5)),
        threshold=float(getattr(args, "lr_threshold", 1e-3)),
        cooldown=int(getattr(args, "lr_cooldown", 200)),
    )
    amp = resolve_amp_context(device=device, no_amp=args.no_amp)

    if resume_ckpt is not None:
        if "optimizer_state_dict" in resume_ckpt:
            load_optimizer_state_safely(
                optimizer=optimizer,
                state_dict=resume_ckpt["optimizer_state_dict"],
                saved_optim_type=resume_ckpt.get("optim_type"),
                current_optim_type=optim_type,
                tag="GRPO",
            )
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
                            step_metric=avg_r_stats["reward_mean"],
                        ):
                            global_step += 1
                            save_every = int(getattr(args, "save_every", 0))
                            if (
                                save_every > 0
                                and global_step % save_every == 0
                                and not is_last_prompt
                            ):
                                mid_ckpt = {
                                    "config": asdict(cfg),
                                    "stage": "grpo_stage2",
                                    "codec_type": codec_type,
                                    "optim_type": optim_type,
                                    "rollout_cfg_coef": float(args.rollout_cfg_coef),
                                    "default_sim_bin": int(default_sim_bin),
                                    "best_val_reward": best_val_reward,
                                    "model_state_dict": policy_model.state_dict(),
                                    "optimizer_state_dict": sanitize_optimizer_state_for_save(
                                        optimizer.state_dict()
                                    ),
                                    "scheduler_state_dict": scheduler.state_dict(),
                                    "scaler_state_dict": (
                                        amp.scaler.state_dict()
                                        if amp.scaler.is_enabled()
                                        else None
                                    ),
                                    "epoch": max(0, epoch - 1),
                                    "global_step": global_step,
                                    "last_metrics": dict(avg_r_stats),
                                }
                                if demo_codec is not None:
                                    mid_ckpt["codec_state_dict"] = demo_codec.state_dict()
                                elif saved_codec_state is not None:
                                    mid_ckpt["codec_state_dict"] = saved_codec_state
                                atomic_torch_save(mid_ckpt, args.save_path)
                                print(
                                    f"  [Checkpoint] Stage 2 중간 체크포인트 자동 저장 완료 (opt={global_step}): {args.save_path}"
                                )

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
        epoch_avg_reward = epoch_reward_sum / n_p
        val_msg = ""
        improved_val = False
        epoch_monitor_reward = epoch_avg_reward
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
            epoch_monitor_reward = val_r
            val_msg = (
                f" | val_R={val_r:.4f} (val_BLEU={val_stats['val_bleu_mean']:.4f}, "
                f"val_LAAL={val_stats['val_laal_sec_mean']:.2f}s)"
            )
            if val_r > best_val_reward:
                best_val_reward = val_r
                improved_val = True

        scheduler.observe_epoch_metric(epoch_monitor_reward)
        print(
            f"[Epoch {epoch}/{args.epochs} 완료] "
            f"avg_reward={epoch_avg_reward:.4f} | "
            f"avg_BLEU={epoch_bleu_sum / n_p:.4f} | "
            f"avg_LAAL={epoch_laal_sum / n_p:.2f}s{val_msg}"
        )

        checkpoint = {
            "config": asdict(cfg),
            "stage": "grpo_stage2",
            "codec_type": codec_type,
            "optim_type": optim_type,
            "rollout_cfg_coef": float(args.rollout_cfg_coef),
            "default_sim_bin": int(default_sim_bin),
            "best_val_reward": best_val_reward,
            "model_state_dict": policy_model.state_dict(),
            "optimizer_state_dict": sanitize_optimizer_state_for_save(optimizer.state_dict()),
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


__all__ = [
    "GRPOPromptSample",
    "RolloutBatch",
    "GRPOPromptDataset",
    "compute_smoothed_bleu",
    "compute_laal_latency_sec",
    "rollout_group_trajectories",
    "evaluate_rollout_group_rewards",
    "compute_grpo_loss",
    "run_grpo_training",
]
