"""28만 개급 대용량 데이터셋에서 서브샘플링으로 Stage 1(SFT)부터 Stage 2(GRPO 강화학습)까지 96GB VRAM을 풀 활용해 연속 학습하는 통합 스크립트."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import importlib.util
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Callable, List, Optional, Sequence, Tuple

from libs.env import (
    DEFAULT_CUDA_INDEX_URL,
    add_env_cli_args,
    atomic_torch_save,
    ensure_venv_and_reexec,
    setup_device,
)


def build_sampling_train_parser() -> argparse.ArgumentParser:
    """28만 개 데이터셋 서브샘플링 기반 Stage 1 SFT + Stage 2 GRPO 자동 연속 학습용 CLI 파서를 생성합니다."""
    parser = argparse.ArgumentParser(
        description="EDarch 28만 개 대용량 데이터셋 서브샘플링 & 96GB VRAM Stage 1(SFT) + Stage 2(GRPO) 통합 고속 학습 스크립트"
    )
    add_env_cli_args(parser, include_seed=True)

    parser.add_argument("--src_loc", "--src-loc", "--data-dir", dest="src_loc", type=Path, required=True, help="전처리된 .pt 데이터셋 루트 경로")
    parser.add_argument("--cp_loc", "--cp-loc", dest="cp_loc", type=Path, required=True, help="에폭마다 Stage 1 및 Stage 2 체크포인트를 저장할 디렉터리 경로")
    parser.add_argument("--save_loc", "--save-loc", "--save-path", dest="save_loc", type=Path, required=True, help="최종 및 최고 성능 가중치를 저장할 파일(또는 디렉터리) 경로")
    parser.add_argument("--val_loc", "--val-loc", "--val-data-dir", dest="val_loc", type=Path, default=None, help="검증(.pt) 데이터 경로 (미지정 시 자동 탐색/분할)")
    parser.add_argument("--tokenizer", "--sp-model", dest="sp_model", type=Path, default=None, help="한국어 SentencePiece 토크나이저(.model) 경로 (미지정 시 자동 탐색)")

    parser.add_argument("--preset", choices=("1b", "2b"), default="1b", help="EDarch 모델 프리셋 (기본값: 1b)")
    parser.add_argument("--ko-vocab-size", type=int, default=32000, help="한국어 어휘 사전 크기 (기본값: 32000)")
    parser.add_argument("--init-from", type=Path, default=None, help="초기 가중치로 사용할 사전학습 체크포인트 경로")
    parser.add_argument("--fresh", action="store_true", help="기존 체크포인트 자동 이어하기를 무시하고 처음부터 새로 학습")
    parser.add_argument("--skip-sft", action="store_true", help="Stage 1 SFT를 건너뛰고 즉시 Stage 2 GRPO 강화학습만 수행")
    parser.add_argument("--skip-grpo", action="store_true", help="Stage 1 SFT만 수행하고 Stage 2 GRPO 강화학습은 생략")

    parser.add_argument("--samples_per_epoch", "--samples-per-epoch", dest="samples_per_epoch", type=int, default=12000, help="Stage 1 에폭당 샘플 수 (기본값: 12000)")
    parser.add_argument("--val_samples", "--val-samples", dest="val_samples", type=int, default=512, help="Stage 1 검증 샘플 수 (기본값: 512)")
    parser.add_argument("--epochs", type=int, default=10, help="Stage 1 SFT 총 에폭 수 (기본값: 10)")
    parser.add_argument("--batch-size", type=int, default=28, help="Stage 1 96GB VRAM 배치 크기 (기본값: 28)")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Stage 1 Gradient accumulation 스텝 수 (기본값: 1)")
    parser.add_argument("--crop-frames", type=int, default=750, help="Stage 1 최대 시퀀스 프레임 수 (기본값: 750 = 60초)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader 병렬 워커 수 (기본값: 4)")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Stage 1 초기 최대 학습률 (기본값: 1.5e-4)")

    parser.add_argument("--grpo_prompts_per_epoch", "--grpo-prompts-per-epoch", dest="grpo_prompts_per_epoch", type=int, default=300, help="Stage 2 에폭당 프롬프트 수 (기본값: 300)")
    parser.add_argument("--grpo_val_prompts", "--grpo-val-prompts", dest="grpo_val_prompts", type=int, default=24, help="Stage 2 검증 프롬프트 수 (기본값: 24)")
    parser.add_argument("--grpo-epochs", type=int, default=2, help="Stage 2 GRPO 총 에폭 수 (기본값: 2)")
    parser.add_argument("--grpo-group-size", type=int, default=12, help="Stage 2 프롬프트당 병렬 롤아웃 그룹 크기 G (기본값: 12)")
    parser.add_argument("--grpo-ppo-epochs", type=int, default=2, help="Stage 2 그룹당 내부 정책 업데이트 횟수 (기본값: 2)")
    parser.add_argument("--grpo-grad-accum", type=int, default=2, help="Stage 2 Gradient accumulation 프롬프트 수 (기본값: 2)")
    parser.add_argument("--grpo-lr", type=float, default=1e-5, help="Stage 2 GRPO 초기 학습률 (기본값: 1e-5)")
    parser.add_argument("--grpo-crop-src-frames", type=int, default=400, help="Stage 2 프롬프트 최대 소스 프레임 수 (기본값: 400)")
    parser.add_argument("--grpo-max-tail-sec", type=float, default=5.0, help="Stage 2 롤아웃 무음 꼬리 최대 길이(초) (기본값: 5.0)")
    parser.add_argument("--kl-coef", type=float, default=0.02, help="Stage 2 기준 모델 대비 KL 페널티 계수 (기본값: 0.02)")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="Stage 2 PPO 클리핑 범위 (기본값: 0.2)")
    parser.add_argument("--audio-pg-weight", type=float, default=0.15, help="Stage 2 오디오 정책 경사 가중치 (기본값: 0.15)")
    parser.add_argument("--rollout-cfg-coef", type=float, default=1.0, help="Stage 2 롤아웃 CFG 계수 (기본값: 1.0)")
    parser.add_argument("--temp-text", type=float, default=0.7, help="Stage 2 텍스트 샘플링 온도 (기본값: 0.7)")
    parser.add_argument("--temp-audio", type=float, default=0.8, help="Stage 2 오디오 샘플링 온도 (기본값: 0.8)")
    parser.add_argument("--top-k-text", type=int, default=50, help="Stage 2 텍스트 Top-k (기본값: 50)")
    parser.add_argument("--top-k-audio", type=int, default=250, help="Stage 2 오디오 Top-k (기본값: 250)")
    parser.add_argument("--reward-bleu-weight", type=float, default=1.0, help="Stage 2 BLEU 보상 가중치 (기본값: 1.0)")
    parser.add_argument("--reward-latency-weight", type=float, default=0.15, help="Stage 2 LAAL 지연 페널티 가중치 (기본값: 0.15)")
    parser.add_argument("--reward-audio-weight", type=float, default=0.10, help="Stage 2 오디오 건강도 가중치 (기본값: 0.10)")
    parser.add_argument("--min-wait-frames", type=int, default=3, help="Stage 2 최소 청취 프레임 수 (기본값: 3)")

    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW Weight Decay (기본값: 0.01)")
    parser.add_argument("--warmup-steps", type=int, default=None, help="워밍업 옵티마이저 스텝 수")
    parser.add_argument("--lr-patience", type=int, default=40, help="Stage 1 손실 EMA 정체 감지 스텝 인내 횟수 (기본값: 40)")
    parser.add_argument("--grpo-lr-patience", type=int, default=15, help="Stage 2 보상 EMA 정체 감지 스텝 인내 횟수 (기본값: 15)")
    parser.add_argument("--lr-epoch-patience", type=int, default=1, help="에폭 검증 정체 감지 인내 횟수 (기본값: 1)")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="정체 감지 시 학습률 감쇠 배율 (기본값: 0.5)")
    parser.add_argument("--lr-min-ratio", type=float, default=0.02, help="초기 학습률 대비 최소 학습률 하한 비율 (기본값: 0.02)")
    parser.add_argument("--lr-threshold", type=float, default=1e-3, help="지표 개선 인정 상대 임계치 (기본값: 1e-3)")
    parser.add_argument("--lr-cooldown", type=int, default=10, help="감쇠 직후 유예 스텝 수 (기본값: 10)")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="그래디언트 클리핑 노름 (기본값: 1.0)")
    parser.add_argument("--cfg-dropout", type=float, default=0.1, help="Stage 1 화자 유사도 CFG 드롭아웃 확률 (기본값: 0.1)")

    parser.add_argument("--grad-checkpointing", action="store_true", help="VRAM 절약용 Activation Checkpointing")
    parser.add_argument("--bf16-weights", action="store_true", help="마스터 가중치를 BF16으로 유지")
    parser.add_argument("--enable-source-audio-loss", action="store_true", help="Stage 1 소스 오디오 보조 손실 활성화")
    parser.add_argument("--save-every", type=int, default=200, help="Stage 1 중간 체크포인트 저장 스텝 주기 (기본값: 200)")
    parser.add_argument("--grpo-save-every", type=int, default=40, help="Stage 2 중간 체크포인트 저장 스텝 주기 (기본값: 40)")
    return parser


def _resolve_all_paths(cp_loc: Path, save_loc: Path) -> dict[str, Path]:
    """Stage 1 및 Stage 2의 에폭별 체크포인트 경로와 최종 저장 경로들을 생성해 반환합니다."""
    cp_dir = cp_loc.parent if cp_loc.suffix.lower() == ".pt" else cp_loc
    cp_dir.mkdir(parents=True, exist_ok=True)
    if save_loc.suffix.lower() == ".pt":
        final_main = save_loc
    else:
        save_loc.mkdir(parents=True, exist_ok=True)
        final_main = save_loc / "EDarch_ckpt.pt"
    final_main.parent.mkdir(parents=True, exist_ok=True)
    stem, sfx, parent = final_main.stem, final_main.suffix, final_main.parent
    return {
        "cp_dir": cp_dir,
        "sft_latest_cp": cp_dir / "EDarch_sft_latest_ckpt.pt",
        "grpo_latest_cp": cp_dir / "EDarch_grpo_latest_ckpt.pt",
        "sft_save": parent / f"{stem}_sft{sfx}",
        "sft_best": parent / f"{stem}_sft_best{sfx}",
        "final_save": final_main,
        "final_best": parent / f"{stem}_best{sfx}",
    }


def _auto_find_tokenizer(src_loc: Path, explicit_path: Optional[Path]) -> Optional[Path]:
    """명시된 경로 또는 데이터셋 폴더 인근에서 ko_spm.model 토크나이저 파일을 자동 탐색합니다."""
    if explicit_path is not None and explicit_path.exists():
        return explicit_path
    for cand in (src_loc / "ko_spm.model", src_loc.parent / "ko_spm.model", Path("ko_spm.model")):
        if cand.exists() and cand.is_file():
            return cand
    return None


def _fast_discover_train_val_pt(
    src_loc: Path, val_loc: Optional[Path], val_samples: int, seed: int
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
            train_paths = [p for p in all_paths if not os.path.normcase(os.path.abspath(p)).startswith(val_prefix)]
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
        val_paths = random.Random(seed).sample(val_paths, val_samples)

    print(
        f"[Dataset] 고속 인덱싱 완료 ({time.perf_counter() - t0:.2f}s): "
        f"전체 학습 풀={len(train_paths):,}개 | 고정 검증 풀={len(val_paths):,}개"
    )
    return train_paths, val_paths


def _select_domain_balanced_subset(
    all_paths: Sequence[str],
    target_count: int,
    epoch: int,
    seed: int,
    sort_by_size_buckets: bool = True,
    batch_size: int = 28,
) -> List[str]:
    """전체 풀에서 71379/71693 비율을 유지하며 에폭별 비중복 순환 샘플을 추출합니다."""
    total_n = len(all_paths)
    target_n = total_n if (target_count <= 0 or target_count >= total_n) else target_count
    paths_71379: List[str] = []
    paths_71693: List[str] = []
    for p in all_paths:
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
        selected = _slice_cyclic(list(all_paths), target_n, 300)

    if not sort_by_size_buckets:
        random.Random(seed + 777 * epoch).shuffle(selected)
        return selected

    sized_paths: List[Tuple[int, str]] = []
    for p in selected:
        try:
            sz = os.path.getsize(p)
        except OSError:
            sz = 0
        sized_paths.append((sz, p))
    sized_paths.sort(key=lambda item: item[0])
    ordered = [p for _, p in sized_paths]
    batches = [ordered[i : i + batch_size] for i in range(0, len(ordered), batch_size)]
    random.Random(seed + 997 * epoch).shuffle(batches)
    return [p for batch in batches for p in batch]


class PathListPtDataset:
    """지정된 파일 경로 리스트에서 Stage 1 멀티스트림 `.pt` 텐서를 고속 로드하는 데이터셋입니다."""

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
        """현재 서브셋의 샘플 개수를 반환합니다."""
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
        return codes, max(0, min(self.num_speaker_sim_bins - 1, sim_bin))


def _load_grpo_prompt_from_path(path_str: str, crop_src_frames: int, cfg, default_sim_bin: int):
    """단일 `.pt` 경로에서 Stage 2 GRPO 롤아웃용 GRPOPromptSample 인스턴스를 생성합니다."""
    from libs.grpo_rollout import GRPOPromptSample, _infer_source_span
    from libs.training import load_and_validate_pt_item

    p_obj = Path(path_str)
    item, codes, raw_sim_bin = load_and_validate_pt_item(
        path=p_obj,
        expected_total_streams=cfg.total_streams,
        default_sim_bin=default_sim_bin,
        num_speaker_sim_bins=cfg.num_speaker_sim_bins,
        validate_ranges=False,
    )
    q = cfg.dep_q
    text_stream, en_codes = codes[0], codes[1 + q : 1 + 2 * q]
    silence_code = en_codes[:, -1:].clone()
    src_starts, src_ends, src_len = _infer_source_span(en_codes, item, crop_src_frames=crop_src_frames)
    src_codes = en_codes[:, :src_len].contiguous()

    raw_groups = item.get("ref_token_groups")
    if isinstance(raw_groups, list) and raw_groups:
        ref_groups = [
            [int(t) for t in grp if 0 <= int(t) < cfg.text_card and int(t) not in (cfg.text_pad_id, cfg.text_eos_id)]
            for grp in raw_groups[: len(src_ends)]
        ]
        ref_tokens = [t for grp in ref_groups for t in grp]
    else:
        ref_tokens = [
            int(t) for t in text_stream.tolist()
            if 0 <= int(t) < cfg.text_card and int(t) not in (cfg.text_pad_id, cfg.text_eos_id)
        ]
        ref_groups = [ref_tokens] if ref_tokens else [[]]

    return GRPOPromptSample(
        name=p_obj.name,
        src_codes=src_codes,
        silence_code=silence_code,
        ref_tokens=ref_tokens,
        src_start_frames=src_starts,
        src_end_frames=src_ends,
        ref_token_groups=ref_groups,
        sim_bin=max(0, min(cfg.num_speaker_sim_bins - 1, raw_sim_bin)),
    )


def _build_fast_fused_adamw(model, lr: float, weight_decay: float, device):
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


def _run_stage1_sft(
    args: argparse.Namespace,
    device,
    paths: dict[str, Path],
    all_train_paths: Sequence[str],
    val_paths: Sequence[str],
) -> Path:
    """Stage 1 문장 단위 서브샘플링 지도학습(SFT)을 수행하고 Stage 2 초기 가중치 경로를 반환합니다."""
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

    cp_dir, sft_latest_cp = paths["cp_dir"], paths["sft_latest_cp"]
    sft_save, sft_best = paths["sft_save"], paths["sft_best"]
    final_save, final_best = paths["final_save"], paths["final_best"]

    start_epoch, global_step, best_val_loss = 1, 0, float("inf")
    seen_sim_bins: set[int] = set()
    resume_opt_state: Optional[dict[str, Any]] = None
    resume_sched_state: Optional[dict[str, Any]] = None
    resume_scaler_state: Optional[dict[str, Any]] = None

    auto_resume_path: Optional[Path] = None
    if not args.fresh:
        if sft_latest_cp.exists():
            auto_resume_path = sft_latest_cp
        else:
            sft_ep_ckpts = sorted(cp_dir.glob("EDarch_sft_epoch_*.pt"))
            if sft_ep_ckpts:
                auto_resume_path = sft_ep_ckpts[-1]
            elif sft_save.exists():
                auto_resume_path = sft_save

    if auto_resume_path is not None:
        print(f"[Stage 1 / Resume] 기존 SFT 체크포인트 감지 -> 이어서 재개: {auto_resume_path}")
        ckpt = torch.load(auto_resume_path, map_location="cpu", weights_only=True)
        cfg = EDarchConfig.from_dict(ckpt["config"])
        if not args.enable_source_audio_loss:
            cfg.train_on_source_audio = False
        model = EDarchModel(cfg)
        if not cfg.train_on_source_audio:
            model.load_state_dict_for_inference(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt["model_state_dict"])
        model = apply_bf16_master_weights_if_supported(model, device=device, enabled=args.bf16_weights, tag="Stage1-SFT")
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        seen_sim_bins.update(int(sb) for sb in ckpt.get("seen_sim_bins", []))
        resume_opt_state = ckpt.get("optimizer_state_dict")
        resume_sched_state = ckpt.get("scheduler_state_dict")
        resume_scaler_state = ckpt.get("scaler_state_dict")
        del ckpt
    elif args.init_from is not None and args.init_from.exists():
        print(f"[Stage 1 / Init] 초기 사전학습 가중치 로드: {args.init_from}")
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
        model = apply_bf16_master_weights_if_supported(model, device=device, enabled=args.bf16_weights, tag="Stage1-SFT")
    else:
        cfg = EDarchConfig.from_preset(args.preset)
        cfg.train_on_source_audio = bool(args.enable_source_audio_loss)
        if args.ko_vocab_size is not None:
            cfg.text_card = args.ko_vocab_size
        model = apply_bf16_master_weights_if_supported(EDarchModel(cfg), device=device, enabled=args.bf16_weights, tag="Stage1-SFT")

    if start_epoch > args.epochs:
        chosen = sft_best if sft_best.exists() else (sft_save if sft_save.exists() else auto_resume_path)
        print(f"[Stage 1] 이미 {args.epochs}에폭 학습 완료됨 -> Stage 2로 전환 ({chosen})")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        assert chosen is not None
        return chosen

    if args.grad_checkpointing:
        model.set_gradient_checkpointing(True)

    eff_samples = len(all_train_paths) if (args.samples_per_epoch <= 0 or args.samples_per_epoch >= len(all_train_paths)) else args.samples_per_epoch
    batches_per_epoch = max(1, math.ceil(eff_samples / args.batch_size))
    updates_per_epoch = max(1, math.ceil(batches_per_epoch / args.grad_accum_steps))
    total_updates = updates_per_epoch * max(1, args.epochs)
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else max(1, int(total_updates * 0.04))

    print(
        f"\n==============================================================================\n"
        f"[Stage 1: SFT 고속 지도학습 시작] 에폭당 샘플={eff_samples:,}/{len(all_train_paths):,} "
        f"(batch={args.batch_size}, steps/epoch={batches_per_epoch}, epochs={start_epoch}..{args.epochs})\n"
        f"=============================================================================="
    )

    optimizer = _build_fast_fused_adamw(model, args.lr, args.weight_decay, device)
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
        except Exception:
            pass
        del resume_opt_state
    if resume_sched_state is not None:
        scheduler.load_state_dict(resume_sched_state)
        del resume_sched_state
    if resume_scaler_state is not None and amp.scaler.is_enabled():
        amp.scaler.load_state_dict(resume_scaler_state)
    del resume_scaler_state

    val_dataloader: Optional[DataLoader] = None
    if val_paths:
        val_ds = PathListPtDataset(val_paths, args.crop_frames, cfg.total_streams, cfg.text_card, cfg.audio_card, cfg.num_speaker_sim_bins - 1, cfg.num_speaker_sim_bins)
        val_dataloader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=max(0, min(2, args.num_workers)),
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
            collate_fn=CollatePad(pad_value=cfg.zero_token_id),
        )

    def _save_sft(ep_num: int, is_epoch_end: bool, is_best: bool, metrics_dict: dict[str, float]) -> None:
        default_sim_bin = max(seen_sim_bins) if seen_sim_bins else (cfg.num_speaker_sim_bins - 1)
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
        atomic_torch_save(payload, sft_latest_cp)
        atomic_torch_save(payload, sft_save)
        if args.skip_grpo:
            atomic_torch_save(payload, final_save)
        if is_epoch_end:
            ep_file = cp_dir / f"EDarch_sft_epoch_{ep_num:03d}.pt"
            atomic_torch_save(payload, ep_file)
            print(f"  -> [cp_loc / Stage 1] 에폭 {ep_num} 체크포인트 저장됨: {ep_file}")
        if is_best:
            atomic_torch_save(payload, sft_best)
            if args.skip_grpo:
                atomic_torch_save(payload, final_best)
            print(f"  -> [Stage 1 Best] 최고 검증 가중치(val_loss={best_val_loss:.4f}) 갱신됨: {sft_best}")

    for epoch in range(start_epoch, args.epochs + 1):
        ep_t0 = time.perf_counter()
        epoch_paths = _select_domain_balanced_subset(all_train_paths, eff_samples, epoch, args.seed, True, args.batch_size)
        epoch_ds = PathListPtDataset(epoch_paths, args.crop_frames, cfg.total_streams, cfg.text_card, cfg.audio_card, cfg.num_speaker_sim_bins - 1, cfg.num_speaker_sim_bins)
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
        epoch_loss, num_steps, accum_loss_sum, accum_loss_count = 0.0, 0, 0.0, 0
        last_metrics: dict[str, float] = {}

        for step, (codes, sim_bins) in enumerate(train_loader, start=1):
            seen_sim_bins.update(int(x) for x in sim_bins.tolist())
            codes = codes.to(device, non_blocking=True)
            sim_bins = sim_bins.to(device, non_blocking=True)
            is_last_batch = step == num_batches
            accum_denom = (num_batches - full_accum_batches) if step > full_accum_batches else args.grad_accum_steps

            with torch.amp.autocast(device_type=device.type, dtype=amp.dtype, enabled=amp.enabled):
                loss, metrics = model.compute_loss(codes, speaker_sim_bins=sim_bins, cfg_dropout_prob=args.cfg_dropout)
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
                accum_loss_sum, accum_loss_count = 0.0, 0
                if step_optimizer_with_clip(model, optimizer, scheduler, amp.scaler, args.grad_clip, step_metric=step_avg_loss):
                    global_step += 1
                    if args.save_every > 0 and global_step % args.save_every == 0 and not is_last_batch:
                        _save_sft(max(0, epoch - 1), False, False, last_metrics)

            if step == 1 or step % 10 == 0 or is_last_batch:
                cur_lr = optimizer.param_groups[0]["lr"]
                elapsed_min = (time.perf_counter() - ep_t0) / 60.0
                eta_min = (elapsed_min / step) * (num_batches - step) if step > 0 else 0.0
                print(
                    f"  [SFT Epoch {epoch}/{args.epochs} | Step {step}/{num_batches} (opt={global_step})] "
                    f"loss={metrics['loss']:.4f} (text={metrics['text_loss']:.4f}, audio={metrics['audio_loss']:.4f}) "
                    f"lr={cur_lr:.2e} | 경과={elapsed_min:.1f}m, 남은시간={eta_min:.1f}m"
                )

        avg_loss = epoch_loss / max(1, num_steps)
        epoch_monitor_loss, val_msg, improved_val = avg_loss, "", False
        if val_dataloader is not None:
            val_metrics = _evaluate_sft_validation(model, val_dataloader, device, amp)
            last_metrics.update(val_metrics)
            val_loss = val_metrics["val_loss"]
            epoch_monitor_loss = val_loss
            val_msg = f" | val_loss={val_loss:.4f} (text={val_metrics['val_text_loss']:.4f}, audio={val_metrics['val_audio_loss']:.4f})"
            if val_loss < best_val_loss:
                best_val_loss, improved_val = val_loss, True
        elif avg_loss < best_val_loss:
            best_val_loss, improved_val = avg_loss, True

        scheduler.observe_epoch_metric(epoch_monitor_loss)
        print(f"[Stage 1 Epoch {epoch}/{args.epochs} 완료 ({(time.perf_counter() - ep_t0) / 60.0:.1f}분)] avg_loss={avg_loss:.4f}{val_msg}")
        _save_sft(epoch, True, improved_val, last_metrics)

    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return sft_best if sft_best.exists() else sft_save


def _run_stage2_grpo(
    args: argparse.Namespace,
    device,
    paths: dict[str, Path],
    stage1_ckpt_path: Path,
    all_train_paths: Sequence[str],
    val_paths: Sequence[str],
) -> None:
    """Stage 1 가중치로부터 서브샘플링된 프롬프트에 대해 Stage 2 GRPO 동시통역 강화학습을 자동 수행합니다."""
    import torch
    from libs.config import EDarchConfig
    from libs.grpo import _average_metric_dicts, _move_seq_lp, compute_grpo_loss
    from libs.grpo_rollout import evaluate_rollout_group_rewards, rollout_group_trajectories
    from libs.model import EDarchModel, EDarchSequenceLogProbs
    from libs.optim import (
        apply_bf16_master_weights_if_supported,
        build_warmup_cosine_scheduler,
        resolve_amp_context,
        sanitize_optimizer_state_for_save,
        step_optimizer_with_clip,
    )
    from libs.tokenizer import load_korean_tokenizer

    cp_dir, grpo_latest_cp = paths["cp_dir"], paths["grpo_latest_cp"]
    final_save, final_best = paths["final_save"], paths["final_best"]
    cpu_dev = torch.device("cpu")

    start_epoch, global_step, best_val_reward = 1, 0, float("-inf")
    default_sim_bin: Optional[int] = None
    resume_opt_state: Optional[dict[str, Any]] = None
    resume_sched_state: Optional[dict[str, Any]] = None
    resume_scaler_state: Optional[dict[str, Any]] = None

    auto_resume_grpo: Optional[Path] = None
    if not args.fresh:
        if grpo_latest_cp.exists():
            auto_resume_grpo = grpo_latest_cp
        else:
            grpo_ep_ckpts = sorted(cp_dir.glob("EDarch_grpo_epoch_*.pt"))
            if grpo_ep_ckpts:
                auto_resume_grpo = grpo_ep_ckpts[-1]

    if auto_resume_grpo is not None:
        print(f"[Stage 2 / Resume] 기존 GRPO 체크포인트 감지 -> 이어서 재개: {auto_resume_grpo}")
        r_ckpt = torch.load(auto_resume_grpo, map_location="cpu", weights_only=True)
        cfg = EDarchConfig.from_dict(r_ckpt["config"])
        cfg.train_on_source_audio = False
        if r_ckpt.get("default_sim_bin") is not None:
            default_sim_bin = int(r_ckpt["default_sim_bin"])
        best_val_reward = float(r_ckpt.get("best_val_reward", float("-inf")))
        policy_model = EDarchModel(cfg)
        policy_model.load_state_dict_for_inference(r_ckpt["model_state_dict"])
        start_epoch = int(r_ckpt.get("epoch", 0)) + 1
        global_step = int(r_ckpt.get("global_step", 0))
        resume_opt_state = r_ckpt.get("optimizer_state_dict")
        resume_sched_state = r_ckpt.get("scheduler_state_dict")
        resume_scaler_state = r_ckpt.get("scaler_state_dict")
        del r_ckpt
    elif stage1_ckpt_path.exists():
        print(f"[Stage 2 / Init] Stage 1 완료 가중치 로드: {stage1_ckpt_path}")
        s1_ckpt = torch.load(stage1_ckpt_path, map_location="cpu", weights_only=True)
        cfg = EDarchConfig.from_dict(s1_ckpt["config"])
        cfg.train_on_source_audio = False
        if s1_ckpt.get("default_sim_bin") is not None:
            default_sim_bin = int(s1_ckpt["default_sim_bin"])
        policy_model = EDarchModel(cfg)
        policy_model.load_state_dict_for_inference(s1_ckpt["model_state_dict"])
        del s1_ckpt
    else:
        cfg = EDarchConfig.from_preset(args.preset)
        cfg.train_on_source_audio = False
        if args.ko_vocab_size is not None:
            cfg.text_card = args.ko_vocab_size
        policy_model = EDarchModel(cfg)

    if default_sim_bin is None:
        default_sim_bin = cfg.num_speaker_sim_bins - 1
    if start_epoch > args.grpo_epochs:
        print(f"[Stage 2] 이미 GRPO {args.grpo_epochs}에폭 학습이 완료되어 있습니다.")
        return

    ref_model: Optional[EDarchModel] = None
    if args.kl_coef > 0.0:
        if stage1_ckpt_path.exists():
            ref_ckpt = torch.load(stage1_ckpt_path, map_location="cpu", weights_only=True)
            ref_model = EDarchModel(copy.deepcopy(cfg))
            ref_model.load_state_dict_for_inference(ref_ckpt["model_state_dict"])
            del ref_ckpt
        else:
            ref_model = copy.deepcopy(policy_model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        if device.type == "cuda":
            ref_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            ref_model = ref_model.to(device=device, dtype=ref_dtype)
        else:
            ref_model = ref_model.to(device=device)

    policy_model = apply_bf16_master_weights_if_supported(policy_model, device=device, enabled=args.bf16_weights, tag="Stage2-GRPO")
    if args.grad_checkpointing:
        policy_model.set_gradient_checkpointing(True)

    sp_decode_fn: Optional[Callable[[List[int]], str]] = None
    tok_path = _auto_find_tokenizer(args.src_loc, args.sp_model)
    if tok_path is not None:
        sp_proc = load_korean_tokenizer(tok_path)
        piece_limit = int(sp_proc.GetPieceSize())
        sp_decode_fn = lambda ids: sp_proc.DecodeIds([i for i in ids if 0 <= i < piece_limit])
        print(f"[Stage 2] 한국어 토크나이저 자동 연결 완료 (문자+토큰 혼합 BLEU 보상): {tok_path}")

    prompts_per_epoch = min(len(all_train_paths), max(1, args.grpo_prompts_per_epoch))
    updates_per_epoch = max(1, math.ceil(prompts_per_epoch / args.grpo_grad_accum)) * args.grpo_ppo_epochs
    total_updates = updates_per_epoch * max(1, args.grpo_epochs)
    warmup_steps = max(1, int(total_updates * 0.05))

    print(
        f"\n==============================================================================\n"
        f"[Stage 2: GRPO 동시통역 강화학습 시작] 에폭당 프롬프트={prompts_per_epoch:,} "
        f"(G={args.grpo_group_size}, ppo_epochs={args.grpo_ppo_epochs}, epochs={start_epoch}..{args.grpo_epochs})\n"
        f"=============================================================================="
    )

    optimizer = _build_fast_fused_adamw(policy_model, args.grpo_lr, args.weight_decay, device)
    scheduler = build_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_updates=total_updates,
        warmup_steps=warmup_steps,
        min_total_ratio=args.lr_min_ratio,
        mode="max",
        patience=args.grpo_lr_patience,
        epoch_patience=args.lr_epoch_patience,
        factor=args.lr_factor,
        threshold=args.lr_threshold,
        cooldown=5,
    )
    amp = resolve_amp_context(device=device, no_amp=False)

    if resume_opt_state is not None:
        try:
            optimizer.load_state_dict(resume_opt_state)
        except Exception:
            pass
        del resume_opt_state
    if resume_sched_state is not None:
        scheduler.load_state_dict(resume_sched_state)
        del resume_sched_state
    if resume_scaler_state is not None and amp.scaler.is_enabled():
        amp.scaler.load_state_dict(resume_scaler_state)
    del resume_scaler_state

    val_prompt_paths = list(val_paths[: max(1, args.grpo_val_prompts)]) if val_paths else []

    def _save_grpo(ep_num: int, is_epoch_end: bool, is_best: bool, metrics_dict: dict[str, float]) -> None:
        payload = {
            "config": asdict(cfg),
            "stage": "grpo_stage2",
            "codec_type": "external_mimi",
            "optim_type": "adamw",
            "rollout_cfg_coef": float(args.rollout_cfg_coef),
            "default_sim_bin": int(default_sim_bin),
            "best_val_reward": best_val_reward,
            "model_state_dict": policy_model.state_dict(),
            "optimizer_state_dict": sanitize_optimizer_state_for_save(optimizer.state_dict()),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": amp.scaler.state_dict() if amp.scaler.is_enabled() else None,
            "epoch": ep_num,
            "global_step": global_step,
            "last_metrics": metrics_dict,
        }
        atomic_torch_save(payload, grpo_latest_cp)
        atomic_torch_save(payload, final_save)
        if is_epoch_end:
            ep_file = cp_dir / f"EDarch_grpo_epoch_{ep_num:03d}.pt"
            atomic_torch_save(payload, ep_file)
            print(f"  -> [cp_loc / Stage 2] GRPO 에폭 {ep_num} 체크포인트 저장됨: {ep_file}")
            print(f"  -> [save_loc] 최종 가중치 업데이트됨: {final_save}")
        if is_best:
            atomic_torch_save(payload, final_best)
            print(f"  -> [Stage 2 Best] 최고 보상 가중치(val_R={best_val_reward:.4f}) 저장됨: {final_best}")

    for epoch in range(start_epoch, args.grpo_epochs + 1):
        ep_t0 = time.perf_counter()
        epoch_prompt_paths = _select_domain_balanced_subset(all_train_paths, prompts_per_epoch, epoch + 100, args.seed, False)
        epoch_reward_sum, epoch_bleu_sum, epoch_laal_sum = 0.0, 0.0, 0.0
        rollout_buffer: List[Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, EDarchSequenceLogProbs, Optional[EDarchSequenceLogProbs]]] = []
        accum_r_stats: List[dict[str, float]] = []
        last_metrics: dict[str, float] = {}
        optimizer.zero_grad(set_to_none=True)

        for step, p_str in enumerate(epoch_prompt_paths, start=1):
            prompt = _load_grpo_prompt_from_path(p_str, args.grpo_crop_src_frames, cfg, default_sim_bin)
            prompt_sim_bins = torch.full((args.grpo_group_size,), prompt.sim_bin, dtype=torch.long, device=device)
            rollout = rollout_group_trajectories(
                policy_model, prompt, args.grpo_group_size, args.grpo_max_tail_sec,
                args.rollout_cfg_coef, args.temp_text, args.temp_audio,
                args.top_k_text, args.top_k_audio, device, amp.enabled, amp.dtype,
            )
            _, advantages, r_stats = evaluate_rollout_group_rewards(
                prompt, rollout, cfg, args.reward_bleu_weight,
                args.reward_latency_weight, args.reward_audio_weight,
                args.min_wait_frames, sp_decode_fn,
            )

            policy_model.eval()
            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type, dtype=amp.dtype, enabled=amp.enabled):
                    old_lp_gpu = policy_model.compute_token_log_probs(
                        rollout.rollout_codes, prompt_sim_bins, args.temp_text, args.temp_audio,
                        args.top_k_text, args.top_k_audio, rollout.eos_suppress_steps, args.rollout_cfg_coef,
                    )
                old_lp_cpu = _move_seq_lp(old_lp_gpu, cpu_dev)
                del old_lp_gpu

                ref_lp_cpu: Optional[EDarchSequenceLogProbs] = None
                if ref_model is not None and args.kl_coef > 0.0:
                    with torch.amp.autocast(device_type=device.type, dtype=amp.dtype, enabled=amp.enabled):
                        ref_lp_raw = ref_model.compute_token_log_probs(
                            rollout.rollout_codes, prompt_sim_bins, args.temp_text, args.temp_audio,
                            args.top_k_text, args.top_k_audio, rollout.eos_suppress_steps, args.rollout_cfg_coef,
                        )
                    ref_lp_cpu = _move_seq_lp(ref_lp_raw, cpu_dev)
                    del ref_lp_raw

            rollout_buffer.append((
                rollout.rollout_codes.detach().cpu(),
                prompt_sim_bins.detach().cpu(),
                rollout.eos_suppress_steps,
                advantages.detach().cpu(),
                old_lp_cpu,
                ref_lp_cpu,
            ))
            accum_r_stats.append(r_stats)
            epoch_reward_sum += r_stats["reward_mean"]
            epoch_bleu_sum += r_stats["bleu_mean"]
            epoch_laal_sum += r_stats["laal_sec_mean"]

            is_last_prompt = step == len(epoch_prompt_paths)
            if len(rollout_buffer) >= args.grpo_grad_accum or is_last_prompt:
                avg_r_stats = _average_metric_dicts(accum_r_stats)
                policy_model.train()
                accum_denom = float(len(rollout_buffer))
                collected_losses: List[dict[str, float]] = []
                for _ in range(args.grpo_ppo_epochs):
                    optimizer.zero_grad(set_to_none=True)
                    collected_losses.clear()
                    for b_codes, b_bins, b_eos, b_adv, b_old_lp, b_ref_lp in rollout_buffer:
                        buf_codes = b_codes.to(device, non_blocking=True)
                        buf_bins = b_bins.to(device, non_blocking=True)
                        with torch.amp.autocast(device_type=device.type, dtype=amp.dtype, enabled=amp.enabled):
                            loss, loss_metrics = compute_grpo_loss(
                                policy_model, buf_codes, buf_bins, b_adv, b_old_lp, b_ref_lp,
                                args.clip_eps, args.kl_coef, args.audio_pg_weight,
                                args.temp_text, args.temp_audio, args.top_k_text, args.top_k_audio,
                                b_eos, args.rollout_cfg_coef,
                            )
                            scaled_loss = loss / accum_denom
                        if amp.scaler.is_enabled():
                            amp.scaler.scale(scaled_loss).backward()
                        else:
                            scaled_loss.backward()
                        collected_losses.append(loss_metrics)
                        del buf_codes, buf_bins

                    if step_optimizer_with_clip(policy_model, optimizer, scheduler, amp.scaler, args.grad_clip, step_metric=avg_r_stats["reward_mean"]):
                        global_step += 1
                        if args.grpo_save_every > 0 and global_step % args.grpo_save_every == 0 and not is_last_prompt:
                            _save_grpo(max(0, epoch - 1), False, False, avg_r_stats)

                step_loss_metrics = _average_metric_dicts(collected_losses)
                rollout_buffer.clear()
                accum_r_stats.clear()
                last_metrics = {**avg_r_stats, **step_loss_metrics}

                if step == 1 or step % 5 == 0 or is_last_prompt:
                    cur_lr = optimizer.param_groups[0]["lr"]
                    elapsed_min = (time.perf_counter() - ep_t0) / 60.0
                    eta_min = (elapsed_min / step) * (len(epoch_prompt_paths) - step) if step > 0 else 0.0
                    print(
                        f"  [GRPO Epoch {epoch}/{args.grpo_epochs} | Prompt {step}/{len(epoch_prompt_paths)} (opt={global_step})] "
                        f"R={avg_r_stats['reward_mean']:.3f} | BLEU={avg_r_stats['bleu_mean']:.3f} | "
                        f"LAAL={avg_r_stats['laal_sec_mean']:.2f}s | KL={step_loss_metrics['text_kl']:.4f} | "
                        f"lr={cur_lr:.2e} | 남은시간={eta_min:.1f}m"
                    )

        n_p = float(max(1, len(epoch_prompt_paths)))
        epoch_avg_r = epoch_reward_sum / n_p
        val_msg, improved_val, monitor_r = "", False, epoch_avg_r

        if val_prompt_paths:
            policy_model.eval()
            val_collected: List[dict[str, float]] = []
            with torch.no_grad():
                for vp_str in val_prompt_paths:
                    vp = _load_grpo_prompt_from_path(vp_str, args.grpo_crop_src_frames, cfg, default_sim_bin)
                    v_roll = rollout_group_trajectories(
                        policy_model, vp, 2, args.grpo_max_tail_sec, args.rollout_cfg_coef,
                        max(0.2, args.temp_text * 0.5), args.temp_audio,
                        args.top_k_text, args.top_k_audio, device, amp.enabled, amp.dtype,
                    )
                    _, _, vr_stats = evaluate_rollout_group_rewards(
                        vp, v_roll, cfg, args.reward_bleu_weight,
                        args.reward_latency_weight, args.reward_audio_weight,
                        args.min_wait_frames, sp_decode_fn,
                    )
                    val_collected.append(vr_stats)
            val_avg = _average_metric_dicts(val_collected)
            val_r = val_avg.get("reward_mean", epoch_avg_r)
            monitor_r = val_r
            val_msg = f" | val_R={val_r:.4f} (val_BLEU={val_avg.get('bleu_mean', 0.0):.4f}, val_LAAL={val_avg.get('laal_sec_mean', 0.0):.2f}s)"
            if val_r > best_val_reward:
                best_val_reward, improved_val = val_r, True
        elif epoch_avg_r > best_val_reward:
            best_val_reward, improved_val = epoch_avg_r, True

        scheduler.observe_epoch_metric(monitor_r)
        print(
            f"[Stage 2 GRPO Epoch {epoch}/{args.grpo_epochs} 완료 ({(time.perf_counter() - ep_t0) / 60.0:.1f}분)] "
            f"avg_R={epoch_avg_r:.4f} | avg_BLEU={epoch_bleu_sum / n_p:.4f} | avg_LAAL={epoch_laal_sum / n_p:.2f}s{val_msg}"
        )
        _save_grpo(epoch, True, improved_val, last_metrics)


def run_sampling_training(args: argparse.Namespace) -> None:
    """28만 개 데이터셋 서브샘플링으로 Stage 1(SFT)과 Stage 2(GRPO)를 순차 자동 실행합니다."""
    import torch

    device = setup_device(cpu_only=args.cpu, seed=args.seed, tag="SamplingTrain")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    paths = _resolve_all_paths(args.cp_loc, args.save_loc)
    all_train_paths, val_paths = _fast_discover_train_val_pt(args.src_loc, args.val_loc, args.val_samples, args.seed)

    if args.skip_sft:
        stage1_out = (
            args.init_from
            if (args.init_from is not None and args.init_from.exists())
            else (paths["sft_best"] if paths["sft_best"].exists() else paths["sft_save"])
        )
    else:
        stage1_out = _run_stage1_sft(args, device, paths, all_train_paths, val_paths)

    if not args.skip_grpo:
        _run_stage2_grpo(args, device, paths, stage1_out, all_train_paths, val_paths)

    print(
        f"\n==============================================================================\n"
        f"[Done] Stage 1(SFT) + Stage 2(GRPO) 전 과정 고속 학습 완료!\n"
        f"  - 에폭별 체크포인트 디렉터리: {paths['cp_dir']}\n"
        f"  - Stage 1 최고 가중치       : {paths['sft_best']}\n"
        f"  - 최종(Stage 2) 가중치      : {paths['final_save']}\n"
        f"  - 최고 성능(Stage 2) 가중치 : {paths['final_best']}\n"
        f"=============================================================================="
    )


def main() -> None:
    """CLI 인자를 파싱하고 현재 환경에 PyTorch가 있으면 즉시, 없으면 가상환경을 준비한 뒤 실행합니다."""
    parser = build_sampling_train_parser()
    args = parser.parse_args()
    if importlib.util.find_spec("torch") is None and not args.no_venv:
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
