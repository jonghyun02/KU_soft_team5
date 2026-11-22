from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from env_setup import (
    DEFAULT_CUDA_INDEX_URL,
    ensure_venv_and_reexec,
    get_default_checkpoint_path,
    get_default_venv_dir,
    setup_device,
)

if TYPE_CHECKING:
    import torch


class PreprocessedPtDataset:
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
        if crop_frames < 1:
            raise ValueError(f"crop_frames({crop_frames})는 1 이상이어야 합니다.")
        self.files = sorted(data_dir.rglob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"{data_dir}에서 .pt 학습 파일을 찾지 못했습니다.")
        self.crop_frames = crop_frames
        self.expected_total_streams = expected_total_streams
        self.text_card = text_card
        self.audio_card = audio_card
        self.default_sim_bin = default_sim_bin
        self.text_eos_id = text_eos_id
        self.num_speaker_sim_bins = num_speaker_sim_bins
        self._validated_indices: set[int] = set()
        self._warned_crop_eos = False

        first_codes, first_sim_bin = self._load_raw(0)
        self._validate_once(0, first_codes, first_sim_bin)

    def __len__(self) -> int:
        return len(self.files)

    def _load_raw(self, idx: int) -> tuple["torch.Tensor", int]:
        import torch

        item = torch.load(self.files[idx], map_location="cpu", weights_only=True)
        codes = item["codes"].long()
        sim_bin = int(item.get("sim_bin", self.default_sim_bin))
        return codes, sim_bin

    def _validate_once(self, idx: int, codes: "torch.Tensor", sim_bin: int) -> None:
        if idx in self._validated_indices:
            return
        path = self.files[idx]
        if codes.ndim != 2 or codes.shape[0] != self.expected_total_streams:
            raise ValueError(
                f"데이터 스트림 수 불일치 ({path.name}): 기대 [K={self.expected_total_streams}, T], "
                f"실제 {tuple(codes.shape)}. --preset 설정(1b는 17, 2b는 33)을 확인하세요."
            )
        t_min, t_max = int(codes[0].min().item()), int(codes[0].max().item())
        if t_min < -1 or t_max >= self.text_card:
            raise ValueError(
                f"텍스트 토큰 범위 초과 ({path.name}): [{t_min}, {t_max}], "
                f"허용 [-1, {self.text_card - 1}]. --ko-vocab-size 옵션을 확인하세요."
            )
        a_min, a_max = int(codes[1:].min().item()), int(codes[1:].max().item())
        if a_min < -1 or a_max >= self.audio_card:
            raise ValueError(
                f"오디오 코드북 범위 초과 ({path.name}): [{a_min}, {a_max}], "
                f"허용 [-1, {self.audio_card - 1}]. --preset(1b/2b는 audio_card=2048)을 확인하세요."
            )
        if sim_bin < 0 or sim_bin >= self.num_speaker_sim_bins:
            raise ValueError(
                f"화자 유사도 sim_bin 범위 초과 ({path.name}): {sim_bin}, "
                f"허용 [0, {self.num_speaker_sim_bins - 1}]."
            )
        self._validated_indices.add(idx)

    def __getitem__(self, idx: int) -> tuple["torch.Tensor", int]:
        codes, sim_bin = self._load_raw(idx)
        self._validate_once(idx, codes, sim_bin)
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
    def __init__(self, pad_value: int = -1):
        self.pad_value = pad_value

    def __call__(self, batch: list[tuple["torch.Tensor", int]]) -> tuple["torch.Tensor", "torch.Tensor"]:
        import torch

        max_t = max(c.shape[1] for c, _ in batch)
        k = batch[0][0].shape[0]
        padded = torch.full((len(batch), k, max_t), self.pad_value, dtype=torch.long)
        sim_bins = torch.empty(len(batch), dtype=torch.long)
        for i, (c, sb) in enumerate(batch):
            padded[i, :, : c.shape[1]] = c
            sim_bins[i] = sb
        return padded, sim_bins


def _build_adamw_param_groups(model: "torch.nn.Module", weight_decay: float) -> list[dict]:
    import torch.nn as nn

    no_decay_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            for p in module.parameters(recurse=False):
                no_decay_ids.add(id(p))

    decay_params = []
    no_decay_params = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or id(p) in no_decay_ids:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def _build_warmup_cosine_scheduler(
    optimizer: "torch.optim.Optimizer",
    total_updates: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.1,
) -> "torch.optim.lr_scheduler.LambdaLR":
    import torch

    total_updates = max(1, total_updates)
    warmup_steps = max(0, min(warmup_steps, total_updates - 1))

    def _lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_updates <= warmup_steps:
            return 1.0
        progress = min(1.0, float(step - warmup_steps) / float(total_updates - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


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
    parser = argparse.ArgumentParser(description="EDarch 가상환경 구축 및 CUDA 12.8 학습 스크립트")
    parser.add_argument(
        "--venv-dir",
        type=Path,
        default=get_default_venv_dir(),
        help="가상환경 경로 (기본값: OneDrive 밖 %%LOCALAPPDATA%%\\EDarch_venv)",
    )
    parser.add_argument(
        "--no-venv",
        action="store_true",
        help="가상환경 자동 생성을 건너뛰고 현재 활성 파이썬 환경에서 바로 실행",
    )
    parser.add_argument(
        "--torch-index-url",
        type=str,
        default=DEFAULT_CUDA_INDEX_URL,
        help=f"PyTorch CUDA 인덱스 URL (기본값: {DEFAULT_CUDA_INDEX_URL})",
    )
    parser.add_argument("--cpu", action="store_true", help="CUDA 대신 CPU 전용으로 설치 및 실행")
    parser.add_argument("--reinstall", action="store_true", help="필수 패키지 강제 재설치")

    parser.add_argument(
        "--preset",
        choices=("1b", "2b"),
        default="1b",
        help="EDarch 모델 프리셋 (기본값: 1b; VRAM 절감 시 --grad-checkpointing 함께 사용 권장)",
    )
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="사전학습 체크포인트에서 가중치만 불러와 파인튜닝 시작 (--ko-vocab-size와 함께 사용 가능)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="중단된 체크포인트(모델+옵티마이저+에폭)에서 학습 재개",
    )
    parser.add_argument(
        "--ko-vocab-size",
        type=int,
        default=None,
        help="한국어 어휘 크기 (예: 32000). --init-from 시 어댑테이션 적용, 신규 학습 시 text_card 설정",
    )
    parser.add_argument(
        "--grad-checkpointing",
        action="store_true",
        help="Activation(Gradient) Checkpointing 활성화 (1b/2b 학습 시 VRAM 절감)",
    )
    parser.add_argument(
        "--no-source-audio-loss",
        action="store_true",
        help="소스 오디오 코드북 보조 손실(train_on_source_audio) 비활성화 (타깃 스트림만 학습)",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="전처리된 .pt 샘플 폴더 경로 (미지정 시 MiniMimiCodec 합성 데이터 데모 학습)",
    )
    parser.add_argument(
        "--crop-frames",
        type=int,
        default=None,
        help="학습 시 최대 시퀀스 프레임 수 (초과 시 t=0부터 절단, 미지정 시 cfg.context_frames 사용)",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 워커 수")
    parser.add_argument("--epochs", type=int, default=3, help="총 학습 에폭 수")
    parser.add_argument("--steps-per-epoch", type=int, default=10, help="데모 모드 시 에폭당 스텝 수")
    parser.add_argument("--batch-size", type=int, default=2, help="배치 크기")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation 스텝 수")
    parser.add_argument("--num-frames", type=int, default=16, help="데모 모드 시 샘플 프레임 수 (12.5Hz)")
    parser.add_argument("--lr", type=float, default=1e-4, help="최대 학습률")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="LR 워밍업 옵티마이저 스텝 수 (미지정 시 전체 업데이트의 5%% 자동 설정)",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--cfg-dropout", type=float, default=0.1, help="CFG 조건 드롭아웃 확률")
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="CUDA 혼합 정밀도(BF16/FP16) 비활성화 (기본값: CUDA에서 자동 활성)",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=get_default_checkpoint_path(),
        help="체크포인트 저장 경로 (에폭마다 갱신, 기본값: OneDrive 밖 %%LOCALAPPDATA%%\\EDarch_checkpoints)",
    )
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    return parser.parse_args()


def run_training(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader

    from EDarch_from_scratch import EDarchConfig, EDarchModel, MiniMimiCodec
    from lm import adapt_EDarch_for_korean_vocab

    if args.grad_accum_steps < 1:
        raise ValueError(f"--grad-accum-steps({args.grad_accum_steps})는 1 이상이어야 합니다.")

    device = setup_device(cpu_only=args.cpu, seed=args.seed, tag="Train")

    start_epoch = 1
    global_step = 0
    resume_ckpt = None

    if args.resume is not None:
        if not args.resume.exists():
            raise FileNotFoundError(f"--resume 체크포인트를 찾을 수 없습니다: {args.resume}")
        print(f"[Train] 체크포인트에서 학습 재개: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        cfg = EDarchConfig.from_dict(resume_ckpt["config"])
        model = EDarchModel(cfg).to(device)
        model.load_state_dict(resume_ckpt["model_state_dict"])
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        global_step = int(resume_ckpt.get("global_step", 0))

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
    if codec is not None and resume_ckpt is not None and "codec_state_dict" in resume_ckpt:
        codec.load_state_dict(resume_ckpt["codec_state_dict"])
    codec_type = "mini_demo" if use_demo_codec else "external_mimi"

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] EDarchModel 파라미터 수: {total_params:,} | codec_type={codec_type}")

    crop_frames = args.crop_frames if args.crop_frames is not None else cfg.context_frames
    dataloader = None
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

    param_groups = _build_adamw_param_groups(model, weight_decay=args.weight_decay)
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

    if start_epoch > args.epochs:
        print(f"[Train] 이미 epoch {start_epoch - 1}까지 완료되어 있습니다 (--epochs={args.epochs}).")
        return

    model.train()
    if codec is not None:
        codec.eval()
    last_metrics: dict[str, float] = {}

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

        for step, (codes, sim_bins) in enumerate(batch_iter, start=1):
            codes = codes.to(device, non_blocking=True)
            sim_bins = sim_bins.to(device, non_blocking=True)

            is_last_batch = step == num_batches
            accum_denom = (
                ((num_batches - 1) % args.grad_accum_steps) + 1
                if (is_last_batch and num_batches % args.grad_accum_steps != 0)
                else args.grad_accum_steps
            )

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss, metrics = model.compute_loss(
                    codes,
                    speaker_sim_bins=sim_bins,
                    cfg_dropout_prob=args.cfg_dropout,
                )
                scaled_loss = loss / accum_denom

            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if step % args.grad_accum_steps == 0 or is_last_batch:
                if scaler.is_enabled():
                    old_scale = scaler.get_scale()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    stepped = scaler.get_scale() >= old_scale
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    stepped = True
                if stepped:
                    scheduler.step()
                    global_step += 1
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += metrics["loss"]
            num_steps += 1
            last_metrics = metrics
            if step == 1 or step % 5 == 0:
                cur_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  [Epoch {epoch}/{args.epochs} | Step {step} (opt_step={global_step})] "
                    f"loss={metrics['loss']:.4f} "
                    f"(text={metrics['text_loss']:.4f}, audio={metrics['audio_loss']:.4f}) "
                    f"lr={cur_lr:.2e}"
                )

        avg_loss = epoch_loss / max(1, num_steps)
        print(f"[Epoch {epoch}/{args.epochs} 완료] avg_loss={avg_loss:.4f}")

        checkpoint = {
            "config": asdict(cfg),
            "codec_type": codec_type,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
            "epoch": epoch,
            "global_step": global_step,
            "last_metrics": last_metrics,
        }
        if codec is not None:
            checkpoint["codec_state_dict"] = codec.state_dict()
        _atomic_save_checkpoint(checkpoint, args.save_path)
        print(f"  -> 체크포인트 저장됨: {args.save_path}")

    print(f"\n[Done] 학습 완료 (최종 체크포인트: {args.save_path})")


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
    run_training(args)


if __name__ == "__main__":
    main()
