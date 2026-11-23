"""arXiv:2309.11093 한-영 병렬 노래 가사(Section/Line)·음절·음소·보컬 분할 일괄 전처리 실행 진입점 스크립트."""

from __future__ import annotations

from pathlib import Path

from libs.cli import build_preprocess_datasets_parser
from libs.env import ensure_venv_and_reexec


def main() -> None:
    """CLI 인자를 파싱하고 가상환경을 준비한 뒤 노래 가사·보컬 분할 전처리를 실행합니다."""
    parser = build_preprocess_datasets_parser()
    args = parser.parse_args()
    ensure_venv_and_reexec(
        script_path=Path(__file__),
        venv_dir=args.venv_dir,
        torch_index_url=args.torch_index_url,
        cpu_only=args.cpu,
        reinstall=args.reinstall,
        no_venv=args.no_venv,
    )
    from libs.dataset_builder import run_preprocess_datasets

    run_preprocess_datasets(args)


if __name__ == "__main__":
    main()
