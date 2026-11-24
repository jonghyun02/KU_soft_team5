"""28만 개급 대용량 전처리 데이터셋에서 에폭별 균등 서브샘플링과 길이 버킷 배칭으로 96GB VRAM을 풀 활용해 초고속 학습하는 스크립트."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import math
import os
from pathlib import Path
import random
import time
from typing import Any, List, Optional, Sequence, Tuple

from libs.env import (
    DEFAULT_CUDA_INDEX_URL,
    add_env_cli_args,
    atomic_torch_save,
    ensure_venv_and_reexec,
    setup_device,
)


def build_sampling_train_parser() -> argparse.ArgumentParser:
    """대용량 데이터셋 서브샘플링 및 96GB VRAM 고속 학습용 CLI 인자 파서를 생성합니다."""
    parser = argparse.ArgumentParser(
        description="EDarch 28만 개 대용량 데이터셋 서브샘플링 & 96GB VRAM 초고속 학습 스크립트"
    )
    add_env_cli_args(parser, include_seed=True)

    parser.add_argument(
        "--src_loc",
        "--src-loc",
        "--data-dir",
        dest="src_loc",
        type=Path,
        required=True,
        help="전처리된 .pt 데이터셋 루트 경로 (약 280,000개 샘플 디렉터리)",
    )
    parser.add_argument(
        "--cp_loc",
        "--cp-loc",
        dest="cp_loc",
        type=Path,
        required=True,
        help="에폭마다 가중치 체크포인트를 저장할 디렉터리(또는 파일) 경로",
    )
    parser.add_argument(
        "--save_loc",
        "--save-loc",
        "--save-path",
        dest="save_loc",
        type=Path,
        required=True,
        help="최종 및 최고 성능 가중치를 저장할 파일(또는 디렉터리) 경로",
    )
    parser.add_argument(
        "--val_loc",
        "--val-loc",
        "--val-data-dir",
        dest="val_loc",
        type=Path,
        default=None,
        help="검증(.pt) 데이터 경로 (미지정 시 src_loc/val 자동 탐색 또는 src_loc에서 자동 분할)",
    )

    parser.add_argument("--preset", choices=("1b", "2b"), default="1b", help="EDarch 모델 프리셋 (기본값: 1b)")
    parser.add_argument("--ko-vocab-size", type=int, default=32000, help="한국어 어휘 사전 크기 (기본값: 32000)")
    parser.add_argument("--init-from", type=Path, default=None, help="초기 가중치로 사용할 사전학습 체크포인트 경로")
    parser.add_argument("--fresh", action="store_true", help="cp_loc의 기존 체크포인트 자동 이어하기를 무시하고 처음부터 학습")

    parser.add_argument(
        "--samples_per_epoch",
        "--samples-per-epoch",
        dest="samples_per_epoch",
        type=int,
        default=12000,
        help="28만 개 데이터 중 1에폭당 추출하여 학습할 샘플 수 (기본값: 12000)",
    )
    parser.add_argument(
        "--val_samples",
        "--val-samples",
        dest="val_samples",
        type=int,
        default=512,
        help="에폭 종료 시 고속 검증에 사용할 고정 대표 샘플 수 (기본값: 512)",
    )
    parser.add_argument("--epochs", type=int, default=10, help="총 학습 에폭 수 (기본값: 10)")
    parser.add_argument("--batch-size", type=int, default=28, help="96GB VRAM 기준 배치 크기 (기본값: 28)")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation 스텝 수 (기본값: 1)")
    parser.add_argument("--crop-frames", type=int, default=750, help="배치 최대 시퀀스 프레임 길이 (기본값: 750 = 60초)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader 병렬 프리페치 워커 수 (기본값: 4)")

    parser.add_argument("--lr", type=float, default=1.5e-4, help="초기 최대 학습률 (기본값: 1.5e-4)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW Weight Decay (기본값: 0.01)")
    parser.add_argument("--warmup-steps", type=int, default=None, help="워밍업 옵티마이저 스텝 수")
    parser.add_argument("--lr-patience", type=int, default=40, help="손실 EMA 정체 감지 스텝 인내 횟수 (기본값: 40)")
    parser.add_argument("--lr-epoch-patience", type=int, default=1, help="에폭 검증 손실 정체 감지 인내 횟수 (기본값: 1)")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="정체 감지 시 학습률 감쇠 배율 (기본값: 0.5)")
    parser.add_argument("--lr-min-ratio", type=float, default=0.02, help="초기 학습률 대비 최소 학습률 하한 비율 (기본값: 0.02)")
    parser.add_argument("--lr-threshold", type=float, default=1e-3, help="손실 개선 인정 상대 임계치 (기본값: 1e-3)")
    parser.add_argument("--lr-cooldown", type=int, default=10, help="감쇠 직후 유예 스텝 수 (기본값: 10)")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="그래디언트 클리핑 노름 (기본값: 1.0)")
    parser.add_argument("--cfg-dropout", type=float, default=0.1, help="화자 유사도 CFG 드롭아웃 확률 (기본값: 0.1)")

    parser.add_argument("--grad-checkpointing", action="store_true", help="VRAM 부족 시에만 켜는 Activation Checkpointing")
    parser.add_argument("--bf16-weights", action="store_true", help="마스터 가중치를 BF16으로 유지")
    parser.add_argument("--enable-source-audio-loss", action="store_true", help="소스 영어 오디오 보조 손실 활성화 (기본값은 비활성으로 속도 2배 가속)")
    parser.add_argument("--save-every", type=int, default=200, help="에폭 도중 중간 체크포인트 저장 스텝 주기 (기본값: 200)")
    return parser


def _resolve_save_file_paths(cp_loc: Path, save_loc: Path) -> Tuple[Path, Path, Path, Path]:
    """체크포인트 디렉터리 경로와 최종 가중치 파일 경로들을 정규화하여 반환합니다."""
    if cp_loc.suffix.lower() == ".pt":
        cp_dir = cp_loc.parent
        latest_cp_path = cp_loc
    else:
        cp_dir = cp_loc
        latest_cp_path = cp_dir / "EDarch_latest_ckpt.pt"
    cp_dir.mkdir(parents=True, exist_ok=True)

    if save_loc.suffix.lower() == ".pt":
        final_save_path = save_loc
    else:
        save_loc.mkdir(parents=True, exist_ok=True)
        final_save_path = save_loc / "EDarch_ckpt.pt"
    final_save_path.parent.mkdir(parents=True, exist_ok=True)
    best_save_path = final_save_path.with_name(
        f"{final_save_path.stem}_best{final_save_path.suffix}"
    )
    return cp_dir, latest_cp_path, final_save_path, best_save_path


def _fast_discover_train_val_pt(
    src_loc: Path,
    val_loc: Optional[Path],
    val_samples: int,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """os.scandir 기반 고속 스캔으로 28만 개급 .pt 경로를 수집하고 train/val 목록을 분리합니다."""
    from libs.preprocessing_common import fast_scandir_files

    if not src_loc.exists():
        raise FileNotFoundError(f"--src_loc 경로를 찾을 수 없습니다: {src_loc}")

    t0 = time.perf_counter()
    train_sub = src_loc / "train"
    val_sub = val_loc if val_loc is not None else (src_loc / "val")

    if train_sub.exists() and train_sub.is_dir():
        _, train_paths = fast_scandir_files([train_sub], ".pt")
    else:
        _, all_paths = fast_scandir_files([src_loc], ".pt")
        if val_sub.exists() and val_sub.is_dir() and val_sub != src_loc:
            val_prefix = os.path.normcase(os.path.abspath(str(val_sub))) + os.sep
            train_paths = [
                p
                for p in all_paths
                if not os.path.normcase(os.path.abspath(p)).startswith(val_prefix)
            ]
        else:
            train_paths = all_paths

    if not train_paths:
        raise FileNotFoundError(f"{src_loc}에서 학습용 .pt 파일을 찾지 못했습니다.")

    val_paths: List[str] = []
    if val_sub is not None and val_sub.exists() and val_sub.is_dir() and val_sub != src_loc:
        _, val_paths = fast_scandir_files([val_sub], ".pt")

    if not val_paths and len(train_paths) > max(64, val_samples * 2):
        rng = random.Random(seed)
        indices = list(range(len(train_paths)))
        rng.shuffle(indices)
        holdout_n = min(max(64, val_samples), len(train_paths) // 10)
        val_idx_set = set(indices[:holdout_n])
        val_paths = [train_paths[i] for i in sorted(val_idx_set)]
        train_paths = [p for i, p in enumerate(train_paths) if i not in val_idx_set]
    elif len(val_paths) > val_samples > 0:
        rng = random.Random(seed)
        val_paths = rng.sample(val_paths, val_samples)

    elapsed = time.perf_counter() - t0
    print(
        f"[Dataset] 고속 인덱싱 완료 ({elapsed:.2f}s): "
        f"전체 학습 풀={len(train_paths):,}개 | 고정 검증 풀={len(val_paths):,}개"
    )
    return train_paths, val_paths


def _select_epoch_subset_with_length_buckets(
    all_train_paths: Sequence[str],
    samples_per_epoch: int,
    batch_size: int,
    epoch: int,
    seed: int,
) -> List[str]:
    """전체 데이터 풀에서 에폭별로 겹치지 않게 도메인 균등 샘플을 뽑고 파일 크기 기반 길이 버킷으로 정렬합니다."""
    total_n = len(all_train_paths)
    target_n = total_n if (samples_per_epoch <= 0 or samples_per_epoch >= total_n) else samples_per_epoch

    paths_71379: List[str] = []
    paths_71693: List[str] = []
    for p in all_train_paths:
        bname = os.path.basename(p)
        if bname.startswith("71379") or (os.sep + "71379" + os.sep) in p:
            paths_71379.append(p)
        else:
            paths_71693.append(p)

    def _slice_cyclic(pool: List[str], k: int, rng_salt: int) -> List[str]:
        if not pool or k <= 0:
            return []
        pool_n = len(pool)
        if k >= pool_n:
            shuffled = list(pool)
            random.Random(seed + rng_salt + epoch).shuffle(shuffled)
            return shuffled
        cycle_idx = ((epoch - 1) * k) // pool_n
        start_off = ((epoch - 1) * k) % pool_n
        shuffled = list(pool)
        random.Random(seed + rng_salt + cycle_idx).shuffle(shuffled)
        if start_off + k <= pool_n:
            return shuffled[start_off : start_off + k]
        first_part = shuffled[start_off:]
        rem = k - len(first_part)
        shuffled_next = list(pool)
        random.Random(seed + rng_salt + cycle_idx + 1).shuffle(shuffled_next)
        return first_part + shuffled_next[:rem]

    if paths_71379 and paths_71693:
        ratio_79 = len(paths_71379) / float(total_n)
        k_79 = min(len(paths_71379), max(1, int(round(target_n * ratio_79))))
        k_93 = min(len(paths_71693), max(0, target_n - k_79))
        if k_79 + k_93 < target_n:
            k_79 = min(len(paths_71379), target_n - k_93)
            k_93 = min(len(paths_71693), target_n - k_79)
        selected = _slice_cyclic(paths_71379, k_79, 100) + _slice_cyclic(paths_71693, k_93, 200)
    else:
        selected = _slice_cyclic(list(all_train_paths), target_n, 300)

    sized_paths: List[Tuple[int, str]] = []
    for p in selected:
        try:
            sz = os.path.getsize(p)
        except OSError:
            sz = 0
        sized_paths.append((sz, p))

    sized_paths.sort(key=lambda item: item[0])
    ordered = [p for _, p in sized_paths]

    batches: List[List[str]] = [
        ordered[i : i + batch_size] for i in range(0, len(ordered), batch_size)
    ]
    random.Random(seed + 997 * epoch).shuffle(batches)
    return [p for batch in batches for p in batch]


class PathListPtDataset:
    """지정된 파일 경로 리스트에서 멀티스트림 `.pt` 텐서를 고속 로드하고 시퀀스 길이를 제한하는 데이터셋입니다."""

    def __init__(
        self,
        file_paths: Sequence[str],
        crop_frames: int,
        expected_total_streams: int,
        text_card: int,
        audio_card: int,
        default_sim_bin: int,
        num_speaker_sim_bins: int = 5,
    ):
        """경로 리스트를 보관하고 첫 번째 샘플로 스트림 차원 및 어휘 범위 정합성을 검증합니다."""
        from libs.training import load_and_validate_pt_item

        self.file_paths = [Path(p) for p in file_paths]
        self.crop_frames = max(1, int(crop_frames))
        self.expected_total_streams = expected_total_streams
        self.text_card = text_card
        self.audio_card = audio_card
        self.default_sim_bin = default_sim_bin
        self.num_speaker_sim_bins = num_speaker_sim_bins
        self._load_fn = load_and_validate_pt_item

        if self.file_paths:
            self._load_fn(
                path=self.file_paths[0],
                expected_total_streams=self.expected_total_streams,
                default_sim_bin=self.default_sim_bin,
                num_speaker_sim_bins=self.num_speaker_sim_bins,
                text_card=self.text_card,
                audio_card=self.audio_card,
                validate_ranges=True,
            )

    def __len__(self) -> int:
        """현재 에폭 서브셋의 샘플 개수를 반환합니다."""
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        """인덱스의 `.pt` 샘플을 로드하고 `crop_frames` 길이로 슬라이싱해 반환합니다."""
        _, codes, sim_bin = self._load_fn(
            path=self.file_paths[idx],
            expected_total_streams=self.expected_total_streams,
            default_sim_bin=self.default_sim_bin,
            num_speaker_sim_bins=self.num_speaker_sim_bins,
            validate_ranges=False,
        )
        if codes.shape[1] > self.crop_frames:
            codes = codes[:, : self.crop_frames]
        sim_bin = max(0, min(self.num_speaker_sim_bins - 1, sim_bin))
        return codes, sim_bin


def _build_fast_fused_adamw(
    model,
    lr: float,
    weight_decay: float,
    device,
):
    """CUDA 환경에서 단일 커널 Fused AdamW를 우선 활성화하여 옵티마이저 스텝 병목을 제거합니다."""
    import torch
    from libs.optim import build_adamw_param_groups

    param_groups = build_adamw_param_groups(model, weight_decay=weight_decay)
    if device.type == "cuda":
        try:
            opt = torch.optim.AdamW(param_groups, lr=lr, fused=True)
            print(f"[Optim] CUDA Fused AdamW 활성화 완료 (lr={lr:.2e})")
            return opt
        except Exception:
            pass
    return torch.optim.AdamW(param_groups, lr=lr)


def run_sampling_training(args: argparse.Namespace) -> None:
    """28만 개 데이터셋에서 에폭별 길이 버킷 서브샘플링을 수행하며 96GB GPU로 고속 학습합니다."""
    import torch
    from torch.utils.data import DataLoader
    from libs.config import EDarchConfig
    from libs.model import EDarchModel, adapt_EDarch_for_korean_vocab
    from libs.optim import (
        apply_bf16_master_weights_if_supported,
        build_warmup_cosine_scheduler,
        resolve_amp_context,
        sanitize_optimizer_state_for_save,
        step_optimizer_with_clip,
    )
    from libs.training import CollatePad, _evaluate_sft_validation

    device = setup_device(cpu_only=args.cpu, seed=args.seed, tag="SamplingTrain")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    cp_dir, latest_cp_path, final_save_path, best_save_path = _resolve_save_file_paths(
        args.cp_loc, args.save_loc
    )
    all_train_paths, val_paths = _fast_discover_train_val_pt(
        src_loc=args.src_loc,
        val_loc=args.val_loc,
        val_samples=args.val_samples,
        seed=args.seed,
    )

    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")
    seen_sim_bins: set[int] = set()
    resume_opt_state: Optional[dict[str, Any]] = None
    resume_sched_state: Optional[dict[str, Any]] = None
    resume_scaler_state: Optional[dict[str, Any]] = None

    auto_resume_path: Optional[Path] = None
    if not args.fresh:
        if latest_cp_path.exists():
            auto_resume_path = latest_cp_path
        else:
            epoch_ckpts = sorted(cp_dir.glob("EDarch_epoch_*.pt"))
            if epoch_ckpts:
                auto_resume_path = epoch_ckpts[-1]
            elif final_save_path.exists():
                auto_resume_path = final_save_path

    if auto_resume_path is not None:
        print(f"[Resume] 기존 체크포인트 자동 감지 -> 이어서 학습 재개: {auto_resume_path}")
        ckpt = torch.load(auto_resume_path, map_location="cpu", weights_only=True)
        cfg = EDarchConfig.from_dict(ckpt["config"])
        if not args.enable_source_audio_loss:
            cfg.train_on_source_audio = False
        model = EDarchModel(cfg)
        if not cfg.train_on_source_audio:
            model.load_state_dict_for_inference(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt["model_state_dict"])
        model = apply_bf16_master_weights_if_supported(
            model, device=device, enabled=args.bf16_weights, tag="SamplingTrain"
        )
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        for sb in ckpt.get("seen_sim_bins", []):
            seen_sim_bins.add(int(sb))
        resume_opt_state = ckpt.get("optimizer_state_dict")
        resume_sched_state = ckpt.get("scheduler_state_dict")
        resume_scaler_state = ckpt.get("scaler_state_dict")
        del ckpt
    elif args.init_from is not None and args.init_from.exists():
        print(f"[Init] 초기 사전학습 가중치 로드: {args.init_from}")
        init_ckpt = torch.load(args.init_from, map_location="cpu", weights_only=True)
        cfg = EDarchConfig.from_dict(init_ckpt["config"])
        cfg.train_on_source_audio = bool(args.enable_source_audio_loss)
        model = EDarchModel(cfg)
        if not cfg.train_on_source_audio:
            model.load_state_dict_for_inference(init_ckpt["model_state_dict"])
        else:
            model.load_state_dict(init_ckpt["model_state_dict"])
        del init_ckpt
        if args.ko_vocab_size is not None and args.ko_vocab_size != cfg.text_card:
            model = adapt_EDarch_for_korean_vocab(model, new_ko_vocab_size=args.ko_vocab_size)
        model = apply_bf16_master_weights_if_supported(
            model, device=device, enabled=args.bf16_weights, tag="SamplingTrain"
        )
    else:
        cfg = EDarchConfig.from_preset(args.preset)
        cfg.train_on_source_audio = bool(args.enable_source_audio_loss)
        if args.ko_vocab_size is not None:
            cfg.text_card = args.ko_vocab_size
        model = apply_bf16_master_weights_if_supported(
            EDarchModel(cfg), device=device, enabled=args.bf16_weights, tag="SamplingTrain"
        )

    if args.grad_checkpointing:
        model.set_gradient_checkpointing(True)
        print("[SamplingTrain] Gradient Checkpointing 활성화됨")

    eff_samples_per_epoch = (
        len(all_train_paths)
        if (args.samples_per_epoch <= 0 or args.samples_per_epoch >= len(all_train_paths))
        else args.samples_per_epoch
    )
    batches_per_epoch = max(1, math.ceil(eff_samples_per_epoch / args.batch_size))
    updates_per_epoch = max(1, math.ceil(batches_per_epoch / args.grad_accum_steps))
    total_updates = updates_per_epoch * max(1, args.epochs)
    warmup_steps = (
        args.warmup_steps if args.warmup_steps is not None else max(1, int(total_updates * 0.04))
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"[SamplingTrain] 모델 파라미터={total_params:,} | "
        f"train_on_source_audio={cfg.train_on_source_audio} | "
        f"에폭당 샘플={eff_samples_per_epoch:,}/{len(all_train_paths):,} "
        f"(batch={args.batch_size}, steps/epoch={batches_per_epoch}, crop_frames={args.crop_frames})"
    )

    optimizer = _build_fast_fused_adamw(
        model=model, lr=args.lr, weight_decay=args.weight_decay, device=device
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_updates=total_updates,
        warmup_steps=warmup_steps,
        min_total_ratio=args.lr_min_ratio,
        mode="min",
        patience=args.lr_patience,
        epoch_patience=args.lr_epoch_patience,
        factor=args.lr_factor,
        threshold=args.lr_threshold,
        cooldown=args.lr_cooldown,
    )
    amp = resolve_amp_context(device=device, no_amp=False)

    if resume_opt_state is not None:
        try:
            optimizer.load_state_dict(resume_opt_state)
        except Exception as exc:
            print(f"[Resume] WARNING: 옵티마이저 상태 복원 생략 ({exc})")
        del resume_opt_state
    if resume_sched_state is not None:
        scheduler.load_state_dict(resume_sched_state)
        del resume_sched_state
    if resume_scaler_state is not None and amp.scaler.is_enabled():
        amp.scaler.load_state_dict(resume_scaler_state)
    del resume_scaler_state

    val_dataloader: Optional[DataLoader] = None
    if val_paths:
        val_dataset = PathListPtDataset(
            file_paths=val_paths,
            crop_frames=args.crop_frames,
            expected_total_streams=cfg.total_streams,
            text_card=cfg.text_card,
            audio_card=cfg.audio_card,
            default_sim_bin=cfg.num_speaker_sim_bins - 1,
            num_speaker_sim_bins=cfg.num_speaker_sim_bins,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=max(0, min(2, args.num_workers)),
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
            collate_fn=CollatePad(pad_value=cfg.zero_token_id),
        )

    if start_epoch > args.epochs:
        print(f"[Done] 이미 목표 에폭({args.epochs})까지 학습이 완료되어 있습니다.")
        return

    def _save_all_checkpoints(
        ep_num: int,
        is_epoch_end: bool,
        is_best: bool,
        metrics_dict: dict[str, float],
    ) -> None:
        default_sim_bin = (
            max(seen_sim_bins) if seen_sim_bins else (cfg.num_speaker_sim_bins - 1)
        )
        payload = {
            "config": asdict(cfg),
            "stage": "sft_stage1",
            "codec_type": "external_mimi",
            "optim_type": "adamw",
            "default_sim_bin": int(default_sim_bin),
            "seen_sim_bins": sorted(seen_sim_bins),
            "best_val_loss": best_val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": sanitize_optimizer_state_for_save(optimizer.state_dict()),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": amp.scaler.state_dict() if amp.scaler.is_enabled() else None,
            "epoch": ep_num,
            "global_step": global_step,
            "last_metrics": metrics_dict,
        }
        atomic_torch_save(payload, latest_cp_path)
        atomic_torch_save(payload, final_save_path)
        if is_epoch_end:
            epoch_cp_file = cp_dir / f"EDarch_epoch_{ep_num:03d}.pt"
            atomic_torch_save(payload, epoch_cp_file)
            print(f"  -> [cp_loc] 에폭 {ep_num} 체크포인트 저장됨: {epoch_cp_file}")
            print(f"  -> [save_loc] 최신 가중치 동기화됨: {final_save_path}")
        if is_best:
            atomic_torch_save(payload, best_save_path)
            print(f"  -> [Best] 최고 검증 가중치(val_loss={best_val_loss:.4f}) 갱신됨: {best_save_path}")

    for epoch in range(start_epoch, args.epochs + 1):
        ep_t0 = time.perf_counter()
        epoch_paths = _select_epoch_subset_with_length_buckets(
            all_train_paths=all_train_paths,
            samples_per_epoch=eff_samples_per_epoch,
            batch_size=args.batch_size,
            epoch=epoch,
            seed=args.seed,
        )
        epoch_ds = PathListPtDataset(
            file_paths=epoch_paths,
            crop_frames=args.crop_frames,
            expected_total_streams=cfg.total_streams,
            text_card=cfg.text_card,
            audio_card=cfg.audio_card,
            default_sim_bin=cfg.num_speaker_sim_bins - 1,
            num_speaker_sim_bins=cfg.num_speaker_sim_bins,
        )
        loader_kwargs: dict[str, Any] = {
            "batch_size": args.batch_size,
            "shuffle": False,
            "num_workers": args.num_workers,
            "pin_memory": (device.type == "cuda"),
            "collate_fn": CollatePad(pad_value=cfg.zero_token_id),
        }
        if args.num_workers > 0:
            loader_kwargs["prefetch_factor"] = 4
        train_loader = DataLoader(epoch_ds, **loader_kwargs)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        num_batches = len(train_loader)
        full_accum_batches = (num_batches // args.grad_accum_steps) * args.grad_accum_steps

        epoch_loss = 0.0
        num_steps = 0
        accum_loss_sum = 0.0
        accum_loss_count = 0
        last_metrics: dict[str, float] = {}

        for step, (codes, sim_bins) in enumerate(train_loader, start=1):
            seen_sim_bins.update(int(x) for x in sim_bins.tolist())
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

            accum_loss_sum += metrics["loss"]
            accum_loss_count += 1
            epoch_loss += metrics["loss"]
            num_steps += 1
            last_metrics = dict(metrics)

            if step % args.grad_accum_steps == 0 or is_last_batch:
                step_avg_loss = accum_loss_sum / max(1, accum_loss_count)
                accum_loss_sum = 0.0
                accum_loss_count = 0
                if step_optimizer_with_clip(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=amp.scaler,
                    grad_clip=args.grad_clip,
                    step_metric=step_avg_loss,
                ):
                    global_step += 1
                    if (
                        args.save_every > 0
                        and global_step % args.save_every == 0
                        and not is_last_batch
                    ):
                        _save_all_checkpoints(
                            ep_num=max(0, epoch - 1),
                            is_epoch_end=False,
                            is_best=False,
                            metrics_dict=last_metrics,
                        )
                        print(
                            f"  [Mid-Save] 스텝 {global_step} 중간 체크포인트 저장 완료 -> {latest_cp_path}"
                        )

            if step == 1 or step % 10 == 0 or is_last_batch:
                cur_lr = optimizer.param_groups[0]["lr"]
                elapsed_min = (time.perf_counter() - ep_t0) / 60.0
                eta_min = (elapsed_min / step) * (num_batches - step) if step > 0 else 0.0
                print(
                    f"  [Epoch {epoch}/{args.epochs} | Step {step}/{num_batches} (opt={global_step})] "
                    f"loss={metrics['loss']:.4f} (text={metrics['text_loss']:.4f}, audio={metrics['audio_loss']:.4f}) "
                    f"lr={cur_lr:.2e} | 경과={elapsed_min:.1f}m, 남은시간={eta_min:.1f}m"
                )

        avg_loss = epoch_loss / max(1, num_steps)
        epoch_monitor_loss = avg_loss
        val_msg = ""
        improved_val = False
        if val_dataloader is not None:
            val_metrics = _evaluate_sft_validation(model, val_dataloader, device, amp)
            last_metrics.update(val_metrics)
            val_loss = val_metrics["val_loss"]
            epoch_monitor_loss = val_loss
            val_msg = (
                f" | val_loss={val_loss:.4f} "
                f"(text={val_metrics['val_text_loss']:.4f}, audio={val_metrics['audio_loss'] if 'audio_loss' in val_metrics else val_metrics['val_audio_loss']:.4f})"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                improved_val = True
        else:
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                improved_val = True

        scheduler.observe_epoch_metric(epoch_monitor_loss)
        ep_min = (time.perf_counter() - ep_t0) / 60.0
        print(
            f"[Epoch {epoch}/{args.epochs} 완료 ({ep_min:.1f}분)] avg_loss={avg_loss:.4f}{val_msg}"
        )
        _save_all_checkpoints(
            ep_num=epoch,
            is_epoch_end=True,
            is_best=improved_val,
            metrics_dict=last_metrics,
        )

    print(
        f"\n[Done] 전체 서브샘플링 고속 학습 완료!\n"
        f"  - 에폭별 체크포인트 폴더: {cp_dir}\n"
        f"  - 최종 가중치 경로      : {final_save_path}\n"
        f"  - 최고 성능 가중치 경로 : {best_save_path}"
    )


def main() -> None:
    """CLI 인자를 파싱하고 현재 환경에 PyTorch가 있으면 즉시, 없으면 가상환경을 준비한 뒤 실행합니다."""
    parser = build_sampling_train_parser()
    args = parser.parse_args()
    has_torch = importlib.util.find_spec("torch") is not None
    if not has_torch and not args.no_venv:
        ensure_venv_and_reexec(
            script_path=Path(__file__),
            venv_dir=args.venv_dir,
            torch_index_url=args.torch_index_url or DEFAULT_CUDA_INDEX_URL,
            cpu_only=args.cpu,
            reinstall=args.reinstall,
            no_venv=args.no_venv,
        )
    run_sampling_training(args)


if __name__ == "__main__":
    main()
