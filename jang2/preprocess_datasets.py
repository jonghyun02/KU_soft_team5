"""AI-Hub 71693·71379 Training/Validation 분할 일괄 전처리 실행 진입점 스크립트."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys

from libs.cli import build_d_drive_parser
from libs.env import ensure_venv_and_reexec


def _ensure_optional_packages(args: argparse.Namespace) -> None:
    """선택된 코덱·TTS·정렬 옵션에 필요한 추가 파이썬 패키지가 누락된 경우 자동 설치합니다."""
    if args.no_auto_install:
        return

    needed: list[str] = []
    needs_hf = (
        args.codec == "mimi"
        or (not args.only_71379 and args.ko_mode_71693 == "mms" and args.hooks_module is None)
        or bool(args.ctc_align_model or args.whisper_align_model or args.wavlm_model)
    )
    if needs_hf:
        if importlib.util.find_spec("transformers") is None:
            needed.append("transformers>=4.46.0")
        if importlib.util.find_spec("huggingface_hub") is None:
            needed.append("huggingface_hub>=0.26.0")

    if (
        not args.only_71379
        and args.ko_mode_71693 == "edge-tts"
        and importlib.util.find_spec("edge_tts") is None
    ):
        needed.append("edge-tts>=6.1.0")

    if needed:
        print(f"[Setup] 전처리 필수 추가 패키지 설치 중: {', '.join(needed)}")
        subprocess.run([sys.executable, "-m", "pip", "install", *needed], check=True)


def main() -> None:
    """가상환경 및 의존성을 구성한 뒤 AI-Hub 71693·71379 분할 전처리 빌더를 실행합니다."""
    parser = build_d_drive_parser()
    args = parser.parse_args()

    ensure_venv_and_reexec(
        script_path=Path(__file__),
        venv_dir=args.venv_dir,
        torch_index_url=args.torch_index_url,
        cpu_only=args.cpu,
        reinstall=args.reinstall,
        no_venv=args.no_venv,
    )
    _ensure_optional_packages(args)

    from libs.dataset_builder import run_d_drive_preprocessing

    run_d_drive_preprocessing(args)


if __name__ == "__main__":
    main()
