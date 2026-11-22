"""D:\\data(AI-Hub 71693) 및 D:\\data2(AI-Hub 71379) 통합 전처리 실행 스크립트."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys

from libs.env import add_env_cli_args, ensure_venv_and_reexec


def build_d_drive_parser() -> argparse.ArgumentParser:
    """`D:\\data` 및 `D:\\data2` 전용 전처리 CLI ArgumentParser를 생성합니다."""
    parser = argparse.ArgumentParser(
        description="D:\\data(AI-Hub 71693 학술대회) & D:\\data2(AI-Hub 71379 방송콘텐츠) 학습용 .pt 전처리 스크립트"
    )
    add_env_cli_args(parser, include_seed=True)

    parser.add_argument(
        "--dir-71693",
        type=Path,
        default=Path(r"D:\data"),
        help="AI-Hub 71693 데이터셋 루트 경로 (기본값: D:\\data)",
    )
    parser.add_argument(
        "--dir-71379",
        type=Path,
        default=Path(r"D:\data2"),
        help="AI-Hub 71379 데이터셋 루트 경로 (기본값: D:\\data2)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"E:\preprocessed_edarch"),
        help="전처리된 train/val .pt 및 ko_spm.model 저장 루트 경로 (기본값: E:\\preprocessed_edarch)",
    )
    parser.add_argument(
        "--splits",
        choices=("both", "train", "val"),
        default="both",
        help="전처리할 데이터 분할 선택 (기본값: both -> train 및 val 분리 저장)",
    )
    parser.add_argument(
        "--only-71379",
        action="store_true",
        help="D:\\data2(71379 실제 한/영 음성 쌍)만 단독 전처리",
    )
    parser.add_argument(
        "--only-71693",
        action="store_true",
        help="D:\\data(71693 국제학술대회)만 단독 전처리",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="71379 방송 카테고리 필터 (예: 교양 다큐 인터뷰 연예공연 영화드라마 예능오락)",
    )
    parser.add_argument(
        "--en-accent-71379",
        choices=("all", "us", "uk"),
        default="all",
        help="71379 영어 억양 필터 (기본값: all)",
    )

    parser.add_argument(
        "--preset",
        choices=("1b", "2b"),
        default="1b",
        help="EDarch 모델 프리셋 (1b: 17스트림, 2b: 33스트림, 기본값: 1b)",
    )
    parser.add_argument(
        "--ko-vocab-size",
        type=int,
        default=None,
        help="한국어 SentencePiece 목표 어휘 크기 (미지정 시 프리셋 기본값 32000)",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="기존 SentencePiece .model 경로 (미지정 시 <output-dir>/ko_spm.model 자동 생성/로드)",
    )
    parser.add_argument(
        "--no-sublabel-corpus",
        action="store_true",
        help="토크나이저 학습 시 D:\\data\\Sublabel\\SbL\\1.한국어 문장 병합 비활성화",
    )
    parser.add_argument(
        "--no-mt-fallback",
        action="store_true",
        help="검수 번역문(MTPE) 누락 시 기계번역(MT) 대체 사용 비활성화",
    )

    parser.add_argument(
        "--codec",
        choices=("mimi", "mini"),
        default="mimi",
        help="오디오 코덱 선택: mimi(사전학습 kyutai/mimi, 기본값) 또는 mini(경량 데모 코덱)",
    )
    parser.add_argument(
        "--mimi-model",
        type=str,
        default="kyutai/mimi",
        help="사전학습 Mimi 코덱 HuggingFace ID 또는 로컬 경로 (기본값: kyutai/mimi)",
    )
    parser.add_argument(
        "--ko-mode-71693",
        choices=("mms", "text-only", "edge-tts", "skip"),
        default="mms",
        help=(
            "한국어 원천 음성이 없는 D:\\data(71693) 처리 모드: "
            "mms(로컬 GPU 한국어 VITS TTS, 기본값), "
            "text-only(타깃 오디오 -1 마스킹 고속 모드), "
            "edge-tts(신경망 성별 매칭 TTS), "
            "skip(71693 건너뜀)"
        ),
    )
    parser.add_argument(
        "--mms-tts-model",
        type=str,
        default="facebook/mms-tts-kor",
        help="--ko-mode-71693 mms 사용 시 HuggingFace VITS 모델 ID",
    )
    parser.add_argument(
        "--hooks-module",
        type=str,
        default=None,
        help="커스텀 zero_shot_ko_tts_fn 훅 파이썬 모듈명",
    )

    parser.add_argument(
        "--align-mode",
        choices=("auto", "energy", "ctc", "whisper", "uniform"),
        default="energy",
        help="Inner Monologue 한국어 텍스트-오디오 정렬 방식 (기본값: energy)",
    )
    parser.add_argument("--ctc-align-model", type=str, default=None, help="한국어 CTC 정렬 모델 ID")
    parser.add_argument("--whisper-align-model", type=str, default=None, help="Whisper 정렬 모델 ID")
    parser.add_argument("--wavlm-model", type=str, default=None, help="WavLM 화자 유사도 모델 ID")
    parser.add_argument("--monologue-lead-frames", type=int, default=1, help="텍스트 선행 프레임 수")

    parser.add_argument(
        "--target-window-sec",
        type=float,
        default=35.0,
        help="에피소드 목표 윈도우 길이(초) — Stage 2 crop_src_frames(36초) 호환 기본값 35.0",
    )
    parser.add_argument("--max-gap-sec", type=float, default=8.0, help="윈도우 분리 기준 최대 휴지기(초)")
    parser.add_argument("--inter-utt-pause-sec", type=float, default=0.5, help="71379 발화 간 기본 휴지기(초)")
    parser.add_argument("--max-overlap-sec", type=float, default=4.0, help="71379 최대 발화 중첩 허용치(초)")

    parser.add_argument(
        "--max-samples-71693",
        type=int,
        default=None,
        help="71693 train 최대 생성 .pt 샘플 수 제한 (빠른 검증용, 미지정 시 전체)",
    )
    parser.add_argument(
        "--max-lectures-71693",
        type=int,
        default=None,
        help="71693 분할당 최대 로드 강의 수 제한",
    )
    parser.add_argument(
        "--max-samples-71379",
        type=int,
        default=None,
        help="71379 train 최대 생성 .pt 샘플 수 제한 (빠른 검증용, 미지정 시 전체)",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="val 분할 데이터셋당 최대 생성 .pt 샘플 수 제한",
    )
    parser.add_argument("--io-threads", type=int, default=16, help="JSON 병렬 파싱 스레드 수")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 .pt 파일을 건너뛰지 않고 덮어쓰기 (기본값은 이어하기 활성)",
    )
    parser.add_argument("--no-verify", action="store_true", help="완료 후 데이터셋 로더 검증 생략")
    parser.add_argument(
        "--no-auto-install",
        action="store_true",
        help="transformers / huggingface_hub / edge-tts 자동 설치 비활성화",
    )
    return parser


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
    """가상환경 및 의존성을 구성한 뒤 D:\\data 및 D:\\data2 통합 전처리 엔진을 실행합니다."""
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
