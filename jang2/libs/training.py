"""Stage 1 문장 단위 지도학습(SFT) 데이터셋 로더, 검증 평가기 및 학습 루프 모듈."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import math
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from libs.codec import MiniMimiCodec
from libs.config import EDarchConfig
from libs.env import atomic_torch_save, setup_device
from libs.model import EDarchModel, adapt_EDarch_for_korean_vocab
from libs.optim import (
    AMPContext,
    build_adamw_param_groups,
    build_warmup_cosine_scheduler,
    resolve_amp_context,
    step_optimizer_with_clip,
)


def list_preprocessed_pt_files(data_dir: Path) -> list[Path]:
    """지정된 디렉터리 하위에서 정렬된 `.pt` 학습 파일 목록을 수집해 반환합니다."""
    files = sorted(data_dir.rglob("*.pt"))
    if not files:
        raise FileNotFoundError(f"{data_dir}에서 .pt 학습 파일을 찾지 못했습니다.")
    return files


def load_and_validate_pt_item(
    path: Path,
    expected_total_streams: int,
    default_sim_bin: int,
    num_speaker_sim_bins: int = 5,
    text_card: Optional[int] = None,
    audio_card: Optional[int] = None,
    validate_ranges: bool = True,
) -> tuple[dict[str, Any], torch.Tensor, int]:
    """단일 `.pt` 파일을 로드하고 스트림 차원, 토큰 범위 및 화자 유사도 구간을 검증합니다."""
    item = torch.load(path, map_location="cpu", weights_only=True)
    codes = item["codes"].long()
    sim_bin = int(item.get("sim_bin", default_sim_bin))
    if codes.ndim != 2 or codes.shape[0] != expected_total_streams:
        raise ValueError(
            f"데이터 스트림 수 불일치 ({path.name}): 기대 [K={expected_total_streams}, T], "
            f"실제 {tuple(codes.shape)}. --preset 설정(1b는 17, 2b는 33)을 확인하세요."
        )
    if validate_ranges:
        if text_card is not None:
            t_min, t_max = int(codes[0].min().item()), int(codes[0].max().item())
            if t_min < -1 or t_max >= text_card:
                raise ValueError(
                    f"텍스트 토큰 범위 초과 ({path.name}): [{t_min}, {t_max}], "
                    f"허용 [-1, {text_card - 1}]. --ko-vocab-size 옵션을 확인하세요."
                )
        if audio_card is not None:
            a_min, a_max = int(codes[1:].min().item()), int(codes[1:].max().item())
            if a_min < -1 or a_max >= audio_card:
                raise ValueError(
                    f"오디오 코드북 범위 초과 ({path.name}): [{a_min}, {a_max}], "
                    f"허용 [-1, {audio_card - 1}]. --preset(1b/2b는 audio_card=2048)을 확인하세요."
                )
        if sim_bin < 0 or sim_bin >= num_speaker_sim_bins:
            raise ValueError(
                f"화자 유사도 sim_bin 범위 초과 ({path.name}): {sim_bin}, "
                f"허용 [0, {num_speaker_sim_bins - 1}]."
            )
    return item, codes, sim_bin


class PreprocessedPtDataset:
    """전처리된 `.pt` 멀티스트림 코드북 샘플들을 검증·로드하고 최대 프레임 길이로 자르는 데이터셋입니다."""

    def __init__(
        self,
        data_dir: Path,
        crop_frames: int,
        expected_total_streams: int,
        text_card: int,
        audio_card: int,
        default_sim_bin: int,
        text_eos_id: int = 2,
        num_speaker_sim_bins: int = 5,
    ):
        """디렉터리 내 `.pt` 파일 목록을 수집하고 첫 샘플의 차원·토큰 범위 정합성을 검증합니다."""
        if crop_frames < 1:
            raise ValueError(f"crop_frames({crop_frames})는 1 이상이어야 합니다.")
        self.files = list_preprocessed_pt_files(data_dir)
        self.crop_frames = crop_frames
        self.expected_total_streams = expected_total_streams
        self.text_card = text_card
        self.audio_card = audio_card
        self.default_sim_bin = default_sim_bin
        self.text_eos_id = text_eos_id
        self.num_speaker_sim_bins = num_speaker_sim_bins
        self._validated_indices: set[int] = set()
        self._warned_crop_eos = False

        _ = self._load_and_validate(0)

    def __len__(self) -> int:
        """전체 학습 샘플 개수를 반환합니다."""
        return len(self.files)

    def _load_and_validate(self, idx: int) -> tuple[torch.Tensor, int]:
        """지정된 인덱스의 `.pt` 샘플을 로드하고 최초 접근 시 범위 정합성을 검증합니다."""
        need_val = idx not in self._validated_indices
        _, codes, sim_bin = load_and_validate_pt_item(
            path=self.files[idx],
            expected_total_streams=self.expected_total_streams,
            default_sim_bin=self.default_sim_bin,
            num_speaker_sim_bins=self.num_speaker_sim_bins,
            text_card=self.text_card,
            audio_card=self.audio_card,
            validate_ranges=need_val,
        )
        if need_val:
            self._validated_indices.add(idx)
        return codes, sim_bin

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """인덱스에 해당하는 멀티스트림 시퀀스와 화자 유사도 구간을 반환합니다."""
        codes, sim_bin = self._load_and_validate(idx)
        if codes.shape[1] > self.crop_frames:
            if (
                not self._warned_crop_eos
                and bool((codes[0, self.crop_frames :] == self.text_eos_id).any().item())
            ):
                self._warned_crop_eos = True
                print(
                    f"[Train] WARNING: {self.files[idx].name}의 길이({codes.shape[1]})가 "
                    f"crop_frames({self.crop_frames})를 초과하여 끝부분의 EOS 토큰이 잘렸습니다. "
                    "전처리 윈도우 길이를 줄이거나 --crop-frames를 늘리는 것을 권장합니다."
                )
            codes = codes[:, : self.crop_frames]
        return codes, sim_bin


class CollatePad:
    """가변 길이 멀티스트림 샘플들을 배치 내 최대 프레임 길이로 패딩하는 Collate 호출 객체입니다."""

    def __init__(self, pad_value: int = -1):
        """패딩에 사용할 토큰 값(`zero_token_id`)을 초기화합니다."""
        self.pad_value = pad_value

    def __call__(
        self, batch: list[tuple[torch.Tensor, int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """샘플 리스트를 `[B, K, max_T]` 텐서와 `[B]` 화자 유사도 텐서로 묶어 반환합니다."""
        max_t = max(c.shape[1] for c, _ in batch)
        k = batch[0][0].shape[0]
        padded = torch.full((len(batch), k, max_t), self.pad_value, dtype=torch.long)
        sim_bins = torch.empty(len(batch), dtype=torch.long)
        for i, (c, sb) in enumerate(batch):
            padded[i, :, : c.shape[1]] = c
            sim_bins[i] = sb
        return padded, sim_bins


@torch.no_grad()
def _evaluate_sft_validation(
    model: EDarchModel,
    val_loader: DataLoader,
    device: torch.device,
    amp: AMPContext,
) -> dict[str, float]:
    """검증 데이터셋에 대해 평균 SFT 손실(총합·텍스트·오디오)을 평가합니다."""
    was_training = model.training
    model.eval()
    total_sum = 0.0
    text_sum = 0.0
    audio_sum = 0.0
    count = 0
    try:
        for codes, sim_bins in val_loader:
            codes = codes.to(device, non_blocking=True)
            sim_bins = sim_bins.to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device.type, dtype=amp.dtype, enabled=amp.enabled
            ):
                _, metrics = model.compute_loss(
                    codes,
                    speaker_sim_bins=sim_bins,
                    cfg_dropout_prob=0.0,
                )
            total_sum += metrics["loss"]
            text_sum += metrics["text_loss"]
            audio_sum += metrics["audio_loss"]
            count += 1
    finally:
        if was_training:
            model.train()

    denom = float(max(1, count))
    return {
        "val_loss": total_sum / denom,
        "val_text_loss": text_sum / denom,
        "val_audio_loss": audio_sum / denom,
    }


def run_sft_training(args: argparse.Namespace) -> None:
    """Stage 1 문장 단위 지도학습(SFT) 또는 파인튜닝·재개 학습 루프를 실행합니다."""
    if args.grad_accum_steps < 1:
        raise ValueError(f"--grad-accum-steps({args.grad_accum_steps})는 1 이상이어야 합니다.")

    device = setup_device(cpu_only=args.cpu, seed=args.seed, tag="Train")

    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")
    seen_sim_bins: set[int] = set()
    saved_codec_state = None
    resume_opt_state = None
    resume_sched_state = None
    resume_scaler_state = None

    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(f"--resume 체크포인트를 찾을 수 없습니다: {args.resume}")
        print(f"[Train] 체크포인트에서 학습 재개(CPU 로드 후 디바이스 이동): {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        cfg = EDarchConfig.from_dict(resume_ckpt["config"])
        model = EDarchModel(cfg)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        model = model.to(device)
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        global_step = int(resume_ckpt.get("global_step", 0))
        best_val_loss = float(resume_ckpt.get("best_val_loss", float("inf")))
        for sb_val in resume_ckpt.get("seen_sim_bins", []):
            seen_sim_bins.add(int(sb_val))
        if "default_sim_bin" in resume_ckpt:
            seen_sim_bins.add(int(resume_ckpt["default_sim_bin"]))

        saved_codec_type = resume_ckpt.get("codec_type")
        if saved_codec_type == "external_mimi" and args.data_dir is None:
            raise ValueError(
                "--resume 체크포인트는 외부 Mimi 데이터(--data-dir)로 학습되었습니다. "
                "데모 모드로 덮어쓰지 않도록 --data-dir를 함께 지정하세요."
            )
        if args.ko_vocab_size is not None and args.ko_vocab_size != cfg.text_card:
            print(
                f"[Train] WARNING: --resume 모드에서는 체크포인트의 text_card={cfg.text_card}가 "
                f"유지되며 --ko-vocab-size={args.ko_vocab_size}는 무시됩니다."
            )
        saved_codec_state = resume_ckpt.get("codec_state_dict")
        resume_opt_state = resume_ckpt.get("optimizer_state_dict")
        resume_sched_state = resume_ckpt.get("scheduler_state_dict")
        resume_scaler_state = resume_ckpt.get("scaler_state_dict")
        del resume_ckpt
    elif args.init_from is not None:
        if not args.init_from.exists():
            raise FileNotFoundError(f"--init-from 체크포인트를 찾을 수 없습니다: {args.init_from}")
        print(f"[Train] 사전학습 체크포인트 로드(CPU): {args.init_from}")
        init_ckpt = torch.load(args.init_from, map_location="cpu", weights_only=True)
        cfg = EDarchConfig.from_dict(init_ckpt["config"])
        if args.no_source_audio_loss:
            cfg.train_on_source_audio = False
        model = EDarchModel(cfg)
        if not cfg.train_on_source_audio:
            model.load_state_dict_for_inference(init_ckpt["model_state_dict"])
        else:
            model.load_state_dict(init_ckpt["model_state_dict"])
        del init_ckpt
        if args.ko_vocab_size is not None and args.ko_vocab_size != cfg.text_card:
            model = adapt_EDarch_for_korean_vocab(model, new_ko_vocab_size=args.ko_vocab_size)
            print(f"[Train] 사전학습 모델 한국어 어휘 어댑테이션 완료: text_card={cfg.text_card}")
        model = model.to(device)
    else:
        cfg = EDarchConfig.from_preset(args.preset)
        if args.no_source_audio_loss:
            cfg.train_on_source_audio = False
        if args.ko_vocab_size is not None:
            cfg.text_card = args.ko_vocab_size
        model = EDarchModel(cfg).to(device)

    if args.grad_checkpointing:
        model.set_gradient_checkpointing(True)
        print("[Train] Gradient Checkpointing 활성화됨")

    use_demo_codec = args.data_dir is None
    codec = MiniMimiCodec(cfg).to(device) if use_demo_codec else None
    if codec is not None and saved_codec_state is not None:
        codec.load_state_dict(saved_codec_state)
    saved_codec_state = None
    codec_type = "mini_demo" if use_demo_codec else "external_mimi"

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] EDarchModel 파라미터 수: {total_params:,} | codec_type={codec_type}")

    crop_frames = args.crop_frames if args.crop_frames is not None else cfg.context_frames
    dataloader: Optional[DataLoader] = None
    val_dataloader: Optional[DataLoader] = None
    if args.data_dir is not None:
        dataset = PreprocessedPtDataset(
            data_dir=args.data_dir,
            crop_frames=crop_frames,
            expected_total_streams=cfg.total_streams,
            text_card=cfg.text_card,
            audio_card=cfg.audio_card,
            default_sim_bin=cfg.num_speaker_sim_bins - 1,
            text_eos_id=cfg.text_eos_id,
            num_speaker_sim_bins=cfg.num_speaker_sim_bins,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=CollatePad(pad_value=cfg.zero_token_id),
        )
        batches_per_epoch = len(dataloader)
        print(f"[Train] 전처리 데이터셋 로드 완료: {len(dataset)}개 샘플 (crop_frames={crop_frames})")

        val_dir = getattr(args, "val_data_dir", None)
        if val_dir is not None and val_dir.exists() and any(val_dir.rglob("*.pt")):
            val_dataset = PreprocessedPtDataset(
                data_dir=val_dir,
                crop_frames=crop_frames,
                expected_total_streams=cfg.total_streams,
                text_card=cfg.text_card,
                audio_card=cfg.audio_card,
                default_sim_bin=cfg.num_speaker_sim_bins - 1,
                text_eos_id=cfg.text_eos_id,
                num_speaker_sim_bins=cfg.num_speaker_sim_bins,
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
                collate_fn=CollatePad(pad_value=cfg.zero_token_id),
            )
            print(f"[Train] 검증(val) 데이터셋 로드 완료: {len(val_dataset)}개 샘플 ({val_dir})")
    else:
        batches_per_epoch = args.steps_per_epoch
        print(
            f"[Train] --data-dir 미지정 -> MiniMimiCodec 합성 데이터 데모 학습 모드 "
            f"(epochs={args.epochs}, steps/epoch={args.steps_per_epoch}, batch={args.batch_size})"
        )

    updates_per_epoch = max(1, math.ceil(batches_per_epoch / args.grad_accum_steps))
    total_updates = updates_per_epoch * max(1, args.epochs)
    warmup_steps = (
        args.warmup_steps if args.warmup_steps is not None else max(1, int(total_updates * 0.05))
    )

    param_groups = build_adamw_param_groups(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    scheduler = build_warmup_cosine_scheduler(
        optimizer, total_updates=total_updates, warmup_steps=warmup_steps
    )
    amp = resolve_amp_context(device=device, no_amp=args.no_amp)

    if resume_opt_state is not None:
        optimizer.load_state_dict(resume_opt_state)
        del resume_opt_state
    if resume_sched_state is not None:
        scheduler.load_state_dict(resume_sched_state)
        del resume_sched_state
    if resume_scaler_state is not None and amp.scaler.is_enabled():
        amp.scaler.load_state_dict(resume_scaler_state)
    del resume_scaler_state
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if start_epoch > args.epochs:
        print(f"[Train] 이미 epoch {start_epoch - 1}까지 완료되어 있습니다 (--epochs={args.epochs}).")
        return

    model.train()
    if codec is not None:
        codec.eval()
    last_metrics: dict[str, float] = {}
    best_save_path = args.save_path.with_name(f"{args.save_path.stem}_best{args.save_path.suffix}")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_loss = 0.0
        num_steps = 0

        if dataloader is not None:
            batch_iter = dataloader
            num_batches = len(dataloader)
        else:
            assert codec is not None
            num_batches = args.steps_per_epoch

            def _synthetic_batches():
                """외부 데이터셋 없이 학습 루프를 검증하기 위한 합성 멀티스트림 배치를 생성합니다."""
                num_samples = args.num_frames * codec.hop_length
                for _ in range(args.steps_per_epoch):
                    with torch.no_grad():
                        src_wav = torch.randn(args.batch_size, 1, num_samples, device=device)
                        tgt_wav = torch.randn(args.batch_size, 1, num_samples, device=device)
                        src_codes = codec.encode(src_wav)
                        tgt_codes = codec.encode(tgt_wav)
                    tgt_text = torch.randint(
                        4, cfg.text_card, (args.batch_size, 1, args.num_frames), device=device
                    )
                    tgt_text[:, :, : min(3, args.num_frames)] = cfg.text_pad_id
                    if args.num_frames > 4:
                        tgt_text[:, :, -1] = cfg.text_eos_id
                    codes = torch.cat([tgt_text, tgt_codes, src_codes], dim=1)
                    sim_bins = torch.randint(
                        0, cfg.num_speaker_sim_bins, (args.batch_size,), device=device
                    )
                    yield codes, sim_bins

            batch_iter = _synthetic_batches()

        optimizer.zero_grad(set_to_none=True)
        full_accum_batches = (
            num_batches // args.grad_accum_steps
        ) * args.grad_accum_steps

        for step, (codes, sim_bins) in enumerate(batch_iter, start=1):
            seen_sim_bins.update(int(x) for x in sim_bins.detach().cpu().tolist())
            codes = codes.to(device, non_blocking=True)
            sim_bins = sim_bins.to(device, non_blocking=True)

            is_last_batch = step == num_batches
            accum_denom = (
                (num_batches - full_accum_batches)
                if step > full_accum_batches
                else args.grad_accum_steps
            )

            with torch.amp.autocast(
                device_type=device.type, dtype=amp.dtype, enabled=amp.enabled
            ):
                loss, metrics = model.compute_loss(
                    codes,
                    speaker_sim_bins=sim_bins,
                    cfg_dropout_prob=args.cfg_dropout,
                )
                scaled_loss = loss / accum_denom

            if amp.scaler.is_enabled():
                amp.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if step % args.grad_accum_steps == 0 or is_last_batch:
                if step_optimizer_with_clip(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=amp.scaler,
                    grad_clip=args.grad_clip,
                ):
                    global_step += 1

            epoch_loss += metrics["loss"]
            num_steps += 1
            last_metrics = dict(metrics)
            if step == 1 or step % 5 == 0:
                cur_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  [Epoch {epoch}/{args.epochs} | Step {step} (opt_step={global_step})] "
                    f"loss={metrics['loss']:.4f} "
                    f"(text={metrics['text_loss']:.4f}, audio={metrics['audio_loss']:.4f}) "
                    f"lr={cur_lr:.2e}"
                )

        avg_loss = epoch_loss / max(1, num_steps)
        val_msg = ""
        improved_val = False
        if val_dataloader is not None:
            val_metrics = _evaluate_sft_validation(model, val_dataloader, device, amp)
            last_metrics.update(val_metrics)
            val_loss = val_metrics["val_loss"]
            val_msg = (
                f" | val_loss={val_loss:.4f} "
                f"(text={val_metrics['val_text_loss']:.4f}, audio={val_metrics['val_audio_loss']:.4f})"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                improved_val = True

        print(f"[Epoch {epoch}/{args.epochs} 완료] avg_loss={avg_loss:.4f}{val_msg}")

        default_sim_bin = (
            max(seen_sim_bins) if seen_sim_bins else (cfg.num_speaker_sim_bins - 1)
        )
        checkpoint = {
            "config": asdict(cfg),
            "stage": "sft_stage1",
            "codec_type": codec_type,
            "default_sim_bin": int(default_sim_bin),
            "seen_sim_bins": sorted(seen_sim_bins),
            "best_val_loss": best_val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": amp.scaler.state_dict() if amp.scaler.is_enabled() else None,
            "epoch": epoch,
            "global_step": global_step,
            "last_metrics": last_metrics,
        }
        if codec is not None:
            checkpoint["codec_state_dict"] = codec.state_dict()
        atomic_torch_save(checkpoint, args.save_path)
        print(f"  -> 체크포인트 저장됨: {args.save_path} (default_sim_bin={default_sim_bin})")
        if improved_val:
            atomic_torch_save(checkpoint, best_save_path)
            print(f"  -> 최고 검증 성능(val_loss={best_val_loss:.4f}) 갱신 저장됨: {best_save_path}")

    print(f"\n[Done] 학습 완료 (최종 체크포인트: {args.save_path})")
