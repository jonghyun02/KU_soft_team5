"""Windows CUDA DLL binding and sequential GPU stage VRAM lifecycle management."""

from __future__ import annotations

import gc
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

_DLL_BOUND = False
_DLL_HANDLES: list[object] = []


def bind_windows_cuda_dlls() -> list[str]:
    """Register NVIDIA cuBLAS/cuDNN and PyTorch lib DLL folders on Windows for CTranslate2/CUDA."""
    global _DLL_BOUND
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    registered: list[str] = []
    if sys.platform != "win32":
        return registered
    if _DLL_BOUND:
        return registered

    candidate_rel_paths = [
        Path("nvidia") / "cublas" / "bin",
        Path("nvidia") / "cudnn" / "bin",
        Path("nvidia") / "cuda_nvrtc" / "bin",
        Path("torch") / "lib",
    ]

    seen: set[str] = set()
    for sp in list(sys.path):
        if not sp:
            continue
        base = Path(sp)
        if not base.is_dir():
            continue
        for rel in candidate_rel_paths:
            dll_dir = base / rel
            if dll_dir.is_dir():
                resolved = str(dll_dir.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    registered.append(resolved)
                    if hasattr(os, "add_dll_directory"):
                        try:
                            _DLL_HANDLES.append(os.add_dll_directory(resolved))
                        except OSError:
                            pass

    if registered:
        current_path = os.environ.get("PATH", "")
        os.environ["PATH"] = os.pathsep.join(registered + [current_path])

    _DLL_BOUND = True
    return registered


@dataclass(slots=True)
class StageVRAMReport:
    """Telemetry captured for a single sequential GPU execution stage."""

    stage_name: str
    device: str
    gpu_name: str
    allocated_before_mb: float
    peak_allocated_mb: float
    allocated_after_mb: float


@contextmanager
def gpu_stage(stage_name: str, prefer_cuda: bool = True, telemetry_sink: list[StageVRAMReport] | None = None) -> Iterator[str]:
    """Run a pipeline stage on GPU and aggressively reclaim VRAM upon exit."""
    bind_windows_cuda_dlls()
    import torch

    use_cuda = bool(prefer_cuda and torch.cuda.is_available())
    device = "cuda" if use_cuda else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    before_mb = 0.0

    if use_cuda:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before_mb = torch.cuda.memory_allocated() / (1024 * 1024)

    try:
        yield device
    finally:
        gc.collect()
        if use_cuda:
            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            after_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            peak_mb = 0.0
            after_mb = 0.0

        if telemetry_sink is not None:
            telemetry_sink.append(
                StageVRAMReport(
                    stage_name=stage_name,
                    device=device,
                    gpu_name=gpu_name,
                    allocated_before_mb=round(before_mb, 2),
                    peak_allocated_mb=round(peak_mb, 2),
                    allocated_after_mb=round(after_mb, 2),
                )
            )
