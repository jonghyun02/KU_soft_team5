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


class WarmupCosinePlateauScheduler(torch.optim.lr_scheduler.LRScheduler):
    """선형 워밍업 + 코사인 감쇠에 손실/보상 정체(Plateau) 감지 자동 학습률 감쇠를 결합한 스케줄러입니다."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_updates: int,
        warmup_steps: int,
        min_lr_ratio: float = 0.1,
        min_total_ratio: float = 0.02,
        mode: str = "min",
        patience: int = 2500,
        epoch_patience: int = 1,
        factor: float = 0.5,
        threshold: float = 1e-3,
        cooldown: int = 500,
        ema_alpha: float = 0.02,
    ):
        """워밍업·코사인 주기와 스텝/에폭 단위 정체 감지 하이퍼파라미터를 초기화합니다."""
        if mode not in ("min", "max"):
            raise ValueError(f"mode는 'min' 또는 'max'여야 합니다: {mode}")
        if not (0.0 < factor < 1.0):
            raise ValueError(f"factor({factor})는 0과 1 사이여야 합니다.")

        self.total_updates = max(1, int(total_updates))
        self.warmup_steps = max(0, min(int(warmup_steps), self.total_updates - 1))
        self.min_lr_ratio = float(min_lr_ratio)
        self.min_total_ratio = max(1e-5, min(float(min_total_ratio), self.min_lr_ratio))
        self.mode = mode
        self.patience = max(0, int(patience))
        self.epoch_patience = max(0, int(epoch_patience))
        self.factor = float(factor)
        self.threshold = max(0.0, float(threshold))
        self.cooldown = max(0, int(cooldown))
        self.ema_alpha = max(0.01, min(1.0, float(ema_alpha)))

        self.plateau_scale: float = 1.0
        self.ema_metric: Optional[float] = None
        self.best_metric: Optional[float] = None
        self.num_bad_steps: int = 0
        self.cooldown_counter: int = 0
        self.best_epoch_metric: Optional[float] = None
        self.num_bad_epochs: int = 0
        self.num_reductions: int = 0

        super().__init__(optimizer)

    def _base_schedule_ratio(self, step: int) -> float:
        """현재 스텝의 기본 워밍업-코사인 배율(0..1)을 계산합니다."""
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return float(step + 1) / float(self.warmup_steps)
        if self.total_updates <= self.warmup_steps:
            return 1.0
        progress = min(
            1.0,
            float(step - self.warmup_steps)
            / float(self.total_updates - self.warmup_steps),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def get_lr(self) -> list[float]:
        """기본 워밍업-코사인 배율과 정체 감쇠 배율(`plateau_scale`)을 결합한 학습률 리스트를 반환합니다."""
        step = max(0, self.last_epoch)
        base_ratio = self._base_schedule_ratio(step)
        if self.warmup_steps > 0 and step < self.warmup_steps:
            eff_ratio = base_ratio * self.plateau_scale
        else:
            eff_ratio = max(self.min_total_ratio, base_ratio * self.plateau_scale)
        return [base_lr * eff_ratio for base_lr in self.base_lrs]

    def _is_improved(self, current: float, best: float) -> bool:
        """현재 지표가 최고 지표 대비 유의미한 임계치(`threshold`) 이상 개선되었는지 판별합니다."""
        margin = max(1e-6, abs(best) * self.threshold)
        if self.mode == "min":
            return current < (best - margin)
        return current > (best + margin)

    def _apply_current_lrs(self) -> None:
        """갱신된 학습률을 옵티마이저 파라미터 그룹에 즉시 반영합니다."""
        new_lrs = self.get_lr()
        for group, lr_val in zip(self.optimizer.param_groups, new_lrs):
            group["lr"] = lr_val
        self._last_lr = new_lrs

    def _reduce_lr(self, reason: str, cur_val: float, best_val: float) -> bool:
        """정체 감지 시 `plateau_scale`을 `factor`만큼 축소하고 학습률을 갱신합니다."""
        old_lr = float(self.optimizer.param_groups[0]["lr"])
        new_scale = max(self.min_total_ratio, self.plateau_scale * self.factor)
        if new_scale >= self.plateau_scale - 1e-9:
            return False
        self.plateau_scale = new_scale
        self.num_reductions += 1
        self.num_bad_steps = 0
        self.cooldown_counter = self.cooldown
        self._apply_current_lrs()
        new_lr = float(self.optimizer.param_groups[0]["lr"])
        if new_lr < old_lr - 1e-12:
            print(
                f"  [LR Plateau] {reason} (현재={cur_val:.4f}, 최고={best_val:.4f}) -> "
                f"학습률 자동 감쇠: {old_lr:.2e} -> {new_lr:.2e} (누적 {self.num_reductions}회, scale={self.plateau_scale:.3f})"
            )
            return True
        return False

    def observe_step_metric(self, metric: float) -> bool:
        """옵티마이저 스텝의 손실(또는 보상)을 지수이동평균(EMA)으로 추적하고 정체 시 학습률을 감쇠합니다."""
        if self.patience <= 0 or not math.isfinite(metric):
            return False
        if self.last_epoch < self.warmup_steps:
            return False

        if self.ema_metric is None:
            self.ema_metric = float(metric)
        else:
            self.ema_metric = (
                1.0 - self.ema_alpha
            ) * self.ema_metric + self.ema_alpha * float(metric)

        if self.best_metric is None:
            self.best_metric = self.ema_metric
            return False

        if self._is_improved(self.ema_metric, self.best_metric):
            self.best_metric = self.ema_metric
            self.num_bad_steps = 0
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
            return False

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False

        self.num_bad_steps += 1
        if self.num_bad_steps >= self.patience:
            reduced = self._reduce_lr(
                reason=f"{self.patience} 옵티마이저 스텝 동안 EMA 지표 미개선",
                cur_val=self.ema_metric,
                best_val=self.best_metric,
            )
            self.best_metric = self.ema_metric
            return reduced
        return False

    def observe_epoch_metric(self, metric: float) -> bool:
        """에폭 종료 시 검증(또는 평균) 지표를 평가하여 미개선 시 학습률을 감쇠합니다."""
        if self.epoch_patience <= 0 or not math.isfinite(metric):
            return False

        cur_val = float(metric)
        if self.best_epoch_metric is None:
            self.best_epoch_metric = cur_val
            return False

        if self._is_improved(cur_val, self.best_epoch_metric):
            self.best_epoch_metric = cur_val
            self.num_bad_epochs = 0
            return False

        self.num_bad_epochs += 1
        if self.num_bad_epochs >= self.epoch_patience:
            self.num_bad_epochs = 0
            return self._reduce_lr(
                reason=f"{self.epoch_patience} 에폭 연속 지표 미개선",
                cur_val=cur_val,
                best_val=self.best_epoch_metric,
            )
        return False

    def state_dict(self) -> dict[str, Any]:
        """스케줄러의 기본 상태와 정체 감지기 내부 상태를 딕셔너리로 직렬화합니다."""
        return {
            "last_epoch": self.last_epoch,
            "plateau_scale": self.plateau_scale,
            "ema_metric": self.ema_metric,
            "best_metric": self.best_metric,
            "num_bad_steps": self.num_bad_steps,
            "cooldown_counter": self.cooldown_counter,
            "best_epoch_metric": self.best_epoch_metric,
            "num_bad_epochs": self.num_bad_epochs,
            "num_reductions": self.num_reductions,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """체크포인트에서 스케줄러 상태와 정체 감쇠 배율을 복원합니다."""
        self.last_epoch = int(state_dict.get("last_epoch", self.last_epoch))
        if "plateau_scale" in state_dict:
            self.plateau_scale = float(state_dict["plateau_scale"])
        self.ema_metric = state_dict.get("ema_metric")
        self.best_metric = state_dict.get("best_metric")
        self.num_bad_steps = int(state_dict.get("num_bad_steps", 0))
        self.cooldown_counter = int(state_dict.get("cooldown_counter", 0))
        self.best_epoch_metric = state_dict.get("best_epoch_metric")
        self.num_bad_epochs = int(state_dict.get("num_bad_epochs", 0))
        self.num_reductions = int(state_dict.get("num_reductions", 0))
        self._apply_current_lrs()


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_updates: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.1,
    min_total_ratio: float = 0.02,
    mode: str = "min",
    patience: int = 2500,
    epoch_patience: int = 1,
    factor: float = 0.5,
    threshold: float = 1e-3,
    cooldown: int = 500,
) -> WarmupCosinePlateauScheduler:
    """선형 워밍업 이후 코사인 감쇠 및 손실/보상 정체 시 자동 학습률 감쇠를 수행하는 스케줄러를 생성합니다."""
    return WarmupCosinePlateauScheduler(
        optimizer=optimizer,
        total_updates=total_updates,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
        min_total_ratio=min_total_ratio,
        mode=mode,
        patience=patience,
        epoch_patience=epoch_patience,
        factor=factor,
        threshold=threshold,
        cooldown=cooldown,
    )


def step_optimizer_with_clip(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    grad_clip: float,
    step_metric: Optional[float] = None,
) -> bool:
    """GradScaler 언스케일링과 그래디언트 클리핑 후 옵티마이저 및 정체 감지 스케줄러 스텝을 안전하게 수행합니다."""
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
        if step_metric is not None and isinstance(
            scheduler, WarmupCosinePlateauScheduler
        ):
            scheduler.observe_step_metric(step_metric)
    optimizer.zero_grad(set_to_none=True)
    return stepped
