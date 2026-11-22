"""EDarch 코어 신경망·토크나이저 자체 검증 및 토크나이저 단독 학습 실행 스크립트."""

from __future__ import annotations

from pathlib import Path

from libs.cli import build_verify_parser
from libs.env import ensure_venv_and_reexec


def main() -> None:
    """하위 명령어에 따라 토크나이저 검증·학습 또는 신경망 아키텍처 통합 검증을 실행합니다."""
    parser = build_verify_parser()
    args = parser.parse_args()
    cmd = args.cmd or "all"

    if cmd == "tokenizer":
        from libs.tokenizer import run_tokenizer_self_test

        run_tokenizer_self_test()
        return

    if cmd == "train-tokenizer":
        from libs.tokenizer import train_bpe_tokenizer_from_cli

        train_bpe_tokenizer_from_cli(
            input_path=args.input,
            output_path=args.output,
            vocab_size=args.vocab_size,
            min_freq=args.min_freq,
            byte_fallback=not args.no_byte_fallback,
        )
        return

    ensure_venv_and_reexec(
        script_path=Path(__file__),
        venv_dir=args.venv_dir,
        torch_index_url=args.torch_index_url,
        cpu_only=args.cpu,
        reinstall=args.reinstall,
        no_venv=args.no_venv,
    )

    if cmd in ("all", "tokenizer"):
        from libs.tokenizer import run_tokenizer_self_test

        run_tokenizer_self_test()

    if cmd in ("all", "model"):
        from libs.streaming import run_bottom_up_verification

        preset = getattr(args, "preset", "1b")
        run_bottom_up_verification(preset=preset)


if __name__ == "__main__":
    main()
