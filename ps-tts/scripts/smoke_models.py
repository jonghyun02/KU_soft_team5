"""모델 세 개(모델 1 enko-nmt, LaBSE, COMET)를 차례로 올려 보는 점검 스크립트.

모델 1 로 영어 3줄을 번역하고 (영어, 한국어) 짝의 LaBSE·COMET 점수를 찍는다. 사람 번역(해요체, 반말),
뜻이 어긋난 번역(숫자·부정), 번역하지 않은 영어, 뜻이 전혀 다른 번역을 같이 넣어 점수 차이를 본다.
올리는 시간, 도는 시간, 프로세스 최대 메모리를 찍는다.
처음 실행 때 LaBSE(약 1.9GB)와 COMET(약 2.3GB)를 Hugging Face 기본 캐시로 내려받는다(올리기 시간에 포함).

예) .venv\\Scripts\\python.exe scripts\\smoke_models.py --device cpu
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pstts.config import Config  # noqa: E402
from pstts.nmt import Model1Drafter  # noqa: E402
from pstts.scorers import CometScorer, LabseScorer  # noqa: E402

LINES = ["I didn't order two tickets.", "Where were you last night?", "Thank you for coming all this way."]
# (영어, 한국어, 종류): 점수가 무엇에 반응하는지 보려고 넣은 짝
EXTRA = [
    ("I didn't order two tickets.", "저는 표를 두 장 주문하지 않았어요.", "사람 번역"),
    ("I didn't order two tickets.", "나 표 두 장 주문 안 했어.", "사람 번역(반말)"),
    ("I didn't order two tickets.", "저는 표를 세 장 주문했어요.", "뜻 어긋남(숫자·부정)"),
    ("I didn't order two tickets.", "I didn't order two tickets.", "번역 안 함(영어 그대로)"),
    ("I didn't order two tickets.", "오늘 저녁은 비빔밥을 먹었다.", "틀린 번역"),
]


def peak_memory_mb() -> float:
    """이 프로세스의 최대 메모리(MB). Windows 는 PeakWorkingSetSize, 그 밖은 ru_maxrss."""
    if os.name != "nt":
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    kernel32 = ctypes.WinDLL("kernel32")
    psapi = ctypes.WinDLL("psapi")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise OSError("GetProcessMemoryInfo 호출이 실패했습니다.")
    return counters.PeakWorkingSetSize / 2**20


class Stopwatch:
    """단계별 걸린 시간과 그때까지의 최대 메모리를 모은다."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, float, float]] = []

    def run(self, name: str, fn, *args, **kwargs):
        t = time.perf_counter()
        out = fn(*args, **kwargs)
        self.rows.append((name, time.perf_counter() - t, peak_memory_mb()))
        print(f"  {name}: {self.rows[-1][1]:.1f}초", file=sys.stderr, flush=True)
        return out


def main() -> None:
    cfg = Config()
    ap = argparse.ArgumentParser(description="모델 1, LaBSE, COMET 을 올려 보는 점검")
    ap.add_argument("--device", default=cfg.device, help="cuda 또는 cpu(기본 cuda). GPU 가 바쁘면 cpu 를 명시")
    ap.add_argument("--nmt-dir", default=cfg.nmt_dir)
    ap.add_argument("--nmt-ckpt", default=cfg.nmt_ckpt)
    ap.add_argument("--beam", type=int, default=cfg.nmt_beam)
    ap.add_argument("--labse-model", default=cfg.labse_model)
    ap.add_argument("--comet-model", default=cfg.comet_model)
    ap.add_argument("--cpu-threads", type=int, default=cfg.cpu_threads)
    args = ap.parse_args()
    watch = Stopwatch()
    print(f"장치 {args.device}, 시작 메모리 {peak_memory_mb():.0f}MB", file=sys.stderr, flush=True)

    drafter = watch.run("모델 1 올리기", Model1Drafter, args.nmt_dir, args.nmt_ckpt, device=args.device,
                        beam=args.beam, cpu_threads=args.cpu_threads)
    drafts = watch.run(f"모델 1 번역 {len(LINES)}줄(beam {args.beam})", drafter.translate, LINES)
    print(f"[모델 1] {drafter.ckpt} (epoch {drafter.info['epoch']}, 평균 {drafter.info['averaged']})")
    drafter.close()

    pairs = [(en, ko, "모델 1 초벌") for en, ko in zip(LINES, drafts)] + EXTRA
    srcs, tgts = [p[0] for p in pairs], [p[1] for p in pairs]

    labse = LabseScorer(args.labse_model, device=args.device, cpu_threads=args.cpu_threads)
    watch.run("LaBSE 올리기", labse.load)
    sims = watch.run(f"LaBSE 점수 {len(pairs)}짝", labse.similarity, srcs, tgts)
    labse.close()

    comet = CometScorer(args.comet_model, device=args.device, batch_size=cfg.comet_batch,
                        cpu_threads=args.cpu_threads)
    watch.run("COMET 올리기", comet.load)
    referenceless = comet.referenceless
    scores = watch.run(f"COMET 점수 {len(pairs)}짝", comet.score, srcs, tgts)
    print(f"[COMET] {args.comet_model} 참조 없음={referenceless} ({comet.checkpoint_path})")
    comet.close()

    print("\n종류\tLaBSE\tCOMET\t영어\t한국어")
    for (en, ko, kind), sim, score in zip(pairs, sims, scores):
        print(f"{kind}\t{sim:.3f}\t{score:.4f}\t{en}\t{ko}")
    print("\n단계\t초\t그때까지 최대 메모리(MB)")
    for name, sec, mem in watch.rows:
        print(f"{name}\t{sec:.1f}\t{mem:.0f}")
    import torch
    print(f"\nCUDA 초기화됨: {torch.cuda.is_initialized()}")


if __name__ == "__main__":
    main()
