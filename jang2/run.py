"""EDarch 실시간 스트리밍 동시통역 추론 실행 진입점 스크립트."""

from __future__ import annotations

from pathlib import Path

from libs.cli import build_infer_parser
from libs.env import ensure_venv_and_reexec


def main() -> None:
    """CLI 인자를 파싱하고 가상환경을 준비한 뒤 실시간 스트리밍 추론을 실행합니다."""
    parser = build_infer_parser()
    args = parser.parse_args()
    ensure_venv_and_reexec(
        script_path=Path(__file__),
        venv_dir=args.venv_dir,
        torch_index_url=args.torch_index_url,
        cpu_only=args.cpu,
        reinstall=args.reinstall,
        no_venv=args.no_venv,
    )
    from libs.streaming import run_streaming_inference

    run_streaming_inference(args)


if __name__ == "__main__":
    main()
