"""AI-Hub 71693 및 71379 데이터셋 통합 전처리 실행 진입점 스크립트."""

from __future__ import annotations

from pathlib import Path

from libs.cli import build_preprocess_parser
from libs.env import ensure_venv_and_reexec


def main() -> None:
    """CLI 인자를 파싱하고 가상환경을 준비한 뒤 통합 전처리 파이프라인을 실행합니다."""
    parser = build_preprocess_parser()
    args = parser.parse_args()
    ensure_venv_and_reexec(
        script_path=Path(__file__),
        venv_dir=args.venv_dir,
        torch_index_url=args.torch_index_url,
        cpu_only=args.cpu,
        reinstall=args.reinstall,
        no_venv=args.no_venv,
    )
    from libs.preprocessing_pipeline import run_preprocessing

    run_preprocessing(args)


if __name__ == "__main__":
    main()
