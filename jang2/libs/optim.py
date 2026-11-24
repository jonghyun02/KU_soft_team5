"""Stage 1 지도학습과 Stage 2 GRPO 강화학습이 공유하는 최적화기·8-bit AdamW·스케줄러·AMP 유틸리티 모듈."""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn


@dataclass
class AMPContext:
    """자동 혼합 정밀도(AMP) 활성 여부, 연산 데이터 타입 및 GradScaler를 묶어 관리하는 데이터클래스입니다."""

    enabled: bool
    dtype: torch.dtype
    scaler: torch.amp.GradScaler


def resolve_amp_context(device: torch.device, no_amp: bool = False) -> AMPContext:
    """디바이스 종류와 BF16 지원 여부를 확인하여 최적의 AMP 설정과 GradScaler를 생성합니다."""
    use_amp = (not no_amp) and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(use_amp and amp_dtype == torch.float16)
    )
    return AMPContext(enabled=use_amp, dtype=amp_dtype, scaler=scaler)


def apply_bf16_master_weights_if_supported(
    model: nn.Module,
    device: torch.device,
    enabled: bool = False,
    tag: str = "Train",
) -> nn.Module:
    """BF16 지원 CUDA GPU에서 요청 시 모델 마스터 가중치를 bfloat16으로 전환해 VRAM을 절감합니다."""
    if not enabled:
        return model.to(device)
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        print(f"[{tag}] BF16 마스터 가중치 모드 활성화 (가중치·그래디언트 VRAM 50% 절감)")
        return model.to(device=device, dtype=torch.bfloat16)
    print(
        f"[{tag}] WARNING: 현재 디바이스({device})가 네이티브 BF16을 지원하지 않아 "
        "수치 안정성을 위해 FP32 마스터 가중치를 유지합니다."
    )
    return model.to(device)


def resolve_optim_type(
    optim_arg: str,
    use_8bit_adam: bool,
    device: torch.device,
    tag: str = "Train",
) -> str:
    """CLI 옵티마이저 인자와 디바이스 가용성을 조합하여 최종 옵티마이저 종류를 결정합니다."""
    resolved = "paged_adamw8bit" if (use_8bit_adam and optim_arg == "adamw") else optim_arg
    if resolved in ("adamw8bit", "paged_adamw8bit") and device.type != "cuda":
        print(
            f"[{tag}] WARNING: 8-bit AdamW({resolved})는 CUDA 전용이므로 CPU에서는 표준 AdamW로 대체합니다."
        )
        return "adamw"
    return resolved


def build_adamw_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """임베딩 및 1차원 정규화 가중치를 Weight Decay에서 제외한 AdamW 파라미터 그룹을 구성합니다."""
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


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    optim_type: str = "adamw",
    auto_install_bnb: bool = True,
    tag: str = "Train",
) -> torch.optim.Optimizer:
    """지정된 종류('adamw', 'adamw8bit', 'paged_adamw8bit')에 맞는 최적화기를 생성해 반환합니다."""
    param_groups = build_adamw_param_groups(model, weight_decay=weight_decay)
    if optim_type == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr)

    if optim_type not in ("adamw8bit", "paged_adamw8bit"):
        raise ValueError(
            f"지원하지 않는 optim_type입니다: {optim_type} "
            "(지원: 'adamw', 'adamw8bit', 'paged_adamw8bit')"
        )

    if importlib.util.find_spec("bitsandbytes") is None:
        if not auto_install_bnb:
            raise RuntimeError(
                f"{optim_type} 사용을 위해 bitsandbytes 패키지가 필요합니다: pip install bitsandbytes>=0.43.0"
            )
        print(f"[{tag}] bitsandbytes 미설치 감지 -> 자동 설치 중 (bitsandbytes>=0.43.0)...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "bitsandbytes>=0.43.0"],
            check=True,
        )

    import bitsandbytes as bnb

    opt_cls = (
        bnb.optim.PagedAdamW8bit
        if optim_type == "paged_adamw8bit"
        else bnb.optim.AdamW8bit
    )
    print(f"[{tag}] VRAM 절약형 8-bit 옵티마이저 활성화: {opt_cls.__name__} (lr={lr:.2e})")
    return opt_cls(param_groups, lr=lr)


def _to_cpu_serializable(obj: Any) -> Any:
    """페이징된 CUDA 텐서를 포함한 중첩 객체 내 모든 텐서를 표준 CPU 텐서로 복제합니다."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _to_cpu_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu_serializable(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu_serializable(x) for x in obj)
    return obj


def sanitize_optimizer_state_for_save(state_dict: dict[str, Any]) -> dict[str, Any]:
    """`torch.load(weights_only=True)` 및 PagedAdamW8bit 호환을 위해 상태를 표준 CPU 텐서로 직렬화합니다."""
    return _to_cpu_serializable(state_dict)


def load_optimizer_state_safely(
    optimizer: torch.optim.Optimizer,
    state_dict: Optional[dict[str, Any]],
    saved_optim_type: Optional[str],
    current_optim_type: str,
    tag: str = "Train",
) -> None:
    """체크포인트 재개 시 32-bit/8-bit 옵티마이저 포맷 호환성을 확인하고 상태를 안전하게 복원합니다."""
    if state_dict is None:
        return
    is_cur_8bit = "8bit" in current_optim_type
    if saved_optim_type is not None:
        is_saved_8bit = "8bit" in saved_optim_type
    else:
        sample_vals = next(iter(state_dict.get("state", {}).values()), {})
        is_saved_8bit = isinstance(sample_vals, dict) and ("state1" in sample_vals)

    if is_cur_8bit != is_saved_8bit:
        print(
            f"[{tag}] WARNING: 체크포인트 옵티마이저 포맷({'8-bit' if is_saved_8bit else '32-bit'})과 "
            f"현재 설정({current_optim_type})이 달라 모멘텀 상태 복원을 건너뛰고 새 버퍼로 재개합니다."
        )
        return

    try:
        optimizer.load_state_dict(state_dict)
    except Exception as exc:
        print(
            f"[{tag}] WARNING: 옵티마이저 상태 복원 실패({exc}) -> 새 옵티마이저 버퍼로 재개합니다."
        )


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_updates: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """선형 워밍업 이후 코사인 감쇠를 수행하는 학습률 스케줄러를 생성합니다."""
    total_updates = max(1, total_updates)
    warmup_steps = max(0, min(warmup_steps, total_updates - 1))

    def _lr_lambda(step: int) -> float:
        """현재 옵티마이저 스텝에 대한 학습률 배율을 계산합니다."""
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_updates <= warmup_steps:
            return 1.0
        progress = min(
            1.0, float(step - warmup_steps) / float(total_updates - warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


def step_optimizer_with_clip(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    grad_clip: float,
) -> bool:
    """GradScaler 언스케일링과 그래디언트 클리핑 후 옵티마이저 및 스케줄러 스텝을 안전하게 수행합니다."""
    if scaler.is_enabled():
        old_scale = scaler.get_scale()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        stepped = scaler.get_scale() >= old_scale
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        stepped = True

    if stepped:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return stepped
