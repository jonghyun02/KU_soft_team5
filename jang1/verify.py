"""arXiv:2309.11093 음절·음소·SSM 분석기, SentencePiece 토크나이저 및 코어 모델 자체 검증 실행 스크립트."""

from __future__ import annotations

from pathlib import Path

from libs.cli import build_verify_parser
from libs.env import ensure_venv_and_reexec


def main() -> None:
    """가상환경을 준비한 뒤 하위 명령어에 따라 음절/음소 분석기·토크나이저·신경망 통합 검증을 실행합니다."""
    parser = build_verify_parser()
    args = parser.parse_args()
    cmd = args.cmd or "all"

    ensure_venv_and_reexec(
        script_path=Path(__file__),
        venv_dir=args.venv_dir,
        torch_index_url=args.torch_index_url,
        cpu_only=args.cpu,
        reinstall=args.reinstall,
        no_venv=args.no_venv,
    )

    if cmd == "train-tokenizer":
        from libs.tokenizer import train_spm_tokenizer_from_cli

        train_spm_tokenizer_from_cli(
            input_path=args.input,
            output_path=args.output,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            character_coverage=args.character_coverage,
            byte_fallback=not args.no_byte_fallback,
        )
        return

    if cmd in ("all", "phonetics"):
        from libs.phonetics import run_phonetics_self_test

        run_phonetics_self_test()

    if cmd in ("all", "tokenizer"):
        from libs.tokenizer import run_tokenizer_self_test

        run_tokenizer_self_test()

    if cmd in ("all", "model"):
        from libs.streaming import run_bottom_up_verification

        preset = getattr(args, "preset", "tiny")
        run_bottom_up_verification(preset=preset)


if __name__ == "__main__":
    main()
