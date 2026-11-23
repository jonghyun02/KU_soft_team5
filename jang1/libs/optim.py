"""Stage 1 지도학습과 Stage 2 Singability GRPO 강화학습이 공유하는 최적화기·스케줄러·AMP 유틸리티 모듈."""

from __future__ import annotations

import math
from dataclasses import dataclass

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
