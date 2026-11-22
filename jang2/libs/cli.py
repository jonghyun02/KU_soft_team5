"""가상환경 부트스트랩 이전에 동작하는 순수 표준 라이브러리 기반 CLI 인자 파서 빌더 모듈."""

from __future__ import annotations

import argparse
from pathlib import Path

from libs.env import (
    PROJECT_DIR,
    add_env_cli_args,
    get_default_checkpoint_path,
    get_default_grpo_checkpoint_path,
)


def build_preprocess_parser() -> argparse.ArgumentParser:
    """AI-Hub 71693 및 71379 데이터셋 전처리 실행용 ArgumentParser를 구성해 반환합니다."""
    parser = argparse.ArgumentParser(
        description="EDarch AI-Hub 71693(국제학술대회) & 71379(방송콘텐츠) 통합 전처리(.pt 생성) 스크립트"
    )
    add_env_cli_args(parser, include_seed=False)

    parser.add_argument(
        "--dataset",
        choices=("auto", "71693", "71379"),
        default="auto",
        help="--json-dir/--wav-dir 사용 시 데이터셋 종류 (기본값: auto 자동 감지)",
    )
    parser.add_argument("--json-dir", type=Path, default=None, help="공통 또는 단일 데이터셋 JSON 디렉터리")
    parser.add_argument("--wav-dir", type=Path, default=None, help="공통 또는 단일 데이터셋 WAV 디렉터리")

    parser.add_argument("--root-71693", type=Path, default=None, help="71693 데이터셋 루트 디렉터리")
    parser.add_argument("--json-dir-71693", type=Path, default=None, help="71693 JSON 라벨 디렉터리")
    parser.add_argument("--wav-dir-71693", type=Path, default=None, help="71693 원본 영어 강의 WAV 디렉터리")

    parser.add_argument("--root-71379", type=Path, default=None, help="71379 데이터셋 루트 디렉터리")
    parser.add_argument("--json-dir-71379", type=Path, default=None, help="71379 JSON 라벨 디렉터리")
    parser.add_argument("--wav-dir-71379", type=Path, default=None, help="71379 통합 WAV 디렉터리")
    parser.add_argument("--en-wav-dir-71379", type=Path, default=None, help="71379 원어민 영어 발화 WAV 디렉터리")
    parser.add_argument("--ko-wav-dir-71379", type=Path, default=None, help="71379 한국어 원천 음성 WAV 디렉터리")
    parser.add_argument(
        "--ko-audio-mode-71379",
        choices=("auto", "real", "tts"),
        default="auto",
        help="71379 타깃 한국어 오디오 선택: auto, real, tts",
    )
    parser.add_argument(
        "--en-accent-71379",
        choices=("all", "us", "uk"),
        default="all",
        help="71379 영어 발화 억양 필터 (all, us, uk)",
    )
    parser.add_argument("--inter-utt-pause-sec", type=float, default=0.5, help="71379 기본 발화 간 휴지기(초)")
    parser.add_argument("--max-overlap-sec", type=float, default=4.0, help="71379 최대 동시 중첩 허용 길이(초)")
    parser.add_argument("--real-pair-sim-bin", type=int, default=None, help="71379 이종 화자 쌍 기본 sim_bin(0..4)")

    parser.add_argument("--output-dir", type=Path, required=True, help="전처리된 .pt 저장 디렉터리")
    parser.add_argument("--preset", choices=("1b", "2b"), default="1b", help="모델 프리셋 (1b 또는 2b)")
    parser.add_argument("--ko-vocab-size", type=int, default=None, help="한국어 어휘 크기 (예: 32000)")
    parser.add_argument("--target-window-sec", type=float, default=45.0, help="목표 스트리밍 윈도우 길이(초)")
    parser.add_argument("--max-gap-sec", type=float, default=8.0, help="윈도우 분할 기준 최대 공백(초)")
    parser.add_argument("--use-mt-fallback", action="store_true", help="검수 번역문 누락 시 기계번역(MT) 대체 사용")
    parser.add_argument("--skip-existing", action="store_true", help="이미 존재하는 .pt 파일 건너뛰기")

    parser.add_argument(
        "--align-mode",
        choices=("auto", "energy", "whisper", "ctc", "uniform"),
        default="auto",
        help="타깃 텍스트-오디오 정렬 방식 (auto, energy, whisper, ctc, uniform)",
    )
    parser.add_argument("--whisper-align-model", type=str, default=None, help="Whisper 단어 타임스탬프 모델 ID")
    parser.add_argument("--ctc-align-model", type=str, default=None, help="한국어 CTC 강제 정렬 모델 ID")
    parser.add_argument("--monologue-lead-frames", type=int, default=1, help="Inner Monologue 선행 프레임 수")

    parser.add_argument("--hooks-module", type=str, default=None, help="사용자 정의 전처리 훅 파이썬 모듈명")
    parser.add_argument(
        "--sp-model",
        "--tokenizer",
        dest="sp_model",
        type=Path,
        default=None,
        help="한국어 SentencePiece 토크나이저 모델 경로 (.model)",
    )
    parser.add_argument("--train-sp-if-missing", action="store_true", help="SentencePiece 모델 미존재 시 코퍼스에서 자동 생성")
    parser.add_argument("--mimi-model", type=str, default=None, help="사전학습 Mimi 코덱 ID 또는 로컬 경로")
    parser.add_argument("--wavlm-model", type=str, default=None, help="화자 유사도 계산용 WavLM 모델 ID")
    parser.add_argument("--demo-hooks", action="store_true", help="외부 모델 없이 합성 데모 훅으로 파이프라인 검증")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    return parser


def build_train_parser() -> argparse.ArgumentParser:
    """Stage 1 문장 단위 지도학습(SFT) 실행용 ArgumentParser를 구성해 반환합니다."""
    parser = argparse.ArgumentParser(description="EDarch Stage 1 지도학습(SFT) 스크립트")
    add_env_cli_args(parser, include_seed=False)

    parser.add_argument("--preset", choices=("1b", "2b"), default="1b", help="EDarch 모델 프리셋")
    parser.add_argument("--init-from", type=Path, default=None, help="사전학습 체크포인트 경로")
    parser.add_argument("--resume", type=Path, default=None, help="중단된 체크포인트에서 학습 재개")
    parser.add_argument("--ko-vocab-size", type=int, default=None, help="한국어 어휘 크기")
    parser.add_argument("--grad-checkpointing", action="store_true", help="Gradient Checkpointing 활성화")
    parser.add_argument("--no-source-audio-loss", action="store_true", help="소스 오디오 보조 손실 비활성화")

    parser.add_argument("--data-dir", type=Path, default=None, help="전처리된 .pt 폴더 경로")
    parser.add_argument("--crop-frames", type=int, default=None, help="최대 시퀀스 프레임 수")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 워커 수")
    parser.add_argument("--epochs", type=int, default=3, help="총 학습 에폭 수")
    parser.add_argument("--steps-per-epoch", type=int, default=10, help="데모 모드 시 에폭당 스텝 수")
    parser.add_argument("--batch-size", type=int, default=2, help="배치 크기")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation 스텝 수")
    parser.add_argument("--num-frames", type=int, default=16, help="데모 모드 시 샘플 프레임 수")
    parser.add_argument("--lr", type=float, default=1e-4, help="최대 학습률")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--warmup-steps", type=int, default=None, help="LR 워밍업 옵티마이저 스텝 수")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--cfg-dropout", type=float, default=0.1, help="CFG 조건 드롭아웃 확률")
    parser.add_argument("--no-amp", action="store_true", help="CUDA 혼합 정밀도 비활성화")
    parser.add_argument("--save-path", type=Path, default=get_default_checkpoint_path(), help="체크포인트 저장 경로")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    return parser


def build_grpo_parser() -> argparse.ArgumentParser:
    """Stage 2 GRPO 동시통역 강화학습 실행용 ArgumentParser를 구성해 반환합니다."""
    parser = argparse.ArgumentParser(
        description="EDarch Stage 2 GRPO(Group Relative Policy Optimization) 동시통역 강화학습 스크립트"
    )
    add_env_cli_args(parser, include_seed=False)

    parser.add_argument("--init-from", type=Path, default=get_default_checkpoint_path(), help="Stage 1 체크포인트 경로")
    parser.add_argument("--resume", type=Path, default=None, help="중단된 Stage 2 체크포인트에서 재개")
    parser.add_argument("--save-path", type=Path, default=get_default_grpo_checkpoint_path(), help="저장 경로")
    parser.add_argument("--preset", choices=("1b", "2b"), default="1b", help="모델 프리셋")
    parser.add_argument("--ko-vocab-size", type=int, default=None, help="한국어 어휘 크기")
    parser.add_argument("--grad-checkpointing", action="store_true", help="Gradient Checkpointing 활성화")

    parser.add_argument("--data-dir", type=Path, default=None, help="전처리된 .pt 폴더 경로")
    parser.add_argument("--crop-src-frames", type=int, default=450, help="프롬프트 최대 소스 프레임 수")
    parser.add_argument("--max-tail-sec", type=float, default=6.0, help="롤아웃 무음 꼬리 최대 길이(초)")
    parser.add_argument(
        "--sp-model",
        "--tokenizer",
        dest="sp_model",
        type=Path,
        default=None,
        help="문자열 BLEU 평가용 한국어 SentencePiece 토크나이저 경로 (.model)",
    )
    parser.add_argument("--hooks-module", type=str, default=None, help="custom_reward_fn 제공 모듈명")

    parser.add_argument("--group-size", type=int, default=4, help="프롬프트당 롤아웃 그룹 크기 G")
    parser.add_argument("--ppo-epochs", type=int, default=1, help="그룹당 내부 정책 업데이트 횟수")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="중요도 비율 클리핑 범위")
    parser.add_argument("--kl-coef", type=float, default=0.02, help="기준 모델 대비 KL 페널티 계수")
    parser.add_argument("--ref-device", choices=("cpu", "cuda"), default="cpu", help="기준 모델 디바이스")
    parser.add_argument("--audio-pg-weight", type=float, default=0.15, help="타깃 오디오 정책 경사 가중치")
    parser.add_argument("--rollout-cfg-coef", type=float, default=1.0, help="롤아웃 CFG 계수")
    parser.add_argument("--temp-text", type=float, default=0.7, help="롤아웃 텍스트 샘플링 온도")
    parser.add_argument("--temp-audio", type=float, default=0.8, help="롤아웃 오디오 샘플링 온도")
    parser.add_argument("--top-k-text", type=int, default=50, help="롤아웃 텍스트 Top-k")
    parser.add_argument("--top-k-audio", type=int, default=250, help="롤아웃 오디오 Top-k")

    parser.add_argument("--reward-bleu-weight", type=float, default=1.0, help="BLEU 보상 가중치")
    parser.add_argument("--reward-latency-weight", type=float, default=0.15, help="초당 LAAL 지연 페널티 가중치")
    parser.add_argument("--reward-audio-weight", type=float, default=0.10, help="오디오 건강도 보상 가중치")
    parser.add_argument("--min-wait-frames", type=int, default=3, help="최소 청취 프레임 수")

    parser.add_argument("--epochs", type=int, default=1, help="GRPO 학습 에폭 수")
    parser.add_argument("--steps-per-epoch", type=int, default=5, help="데모 모드 에폭당 프롬프트 수")
    parser.add_argument("--demo-src-frames", type=int, default=16, help="데모 모드 소스 프레임 수")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation 프롬프트 수")
    parser.add_argument("--lr", type=float, default=1e-5, help="학습률")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--warmup-steps", type=int, default=None, help="워밍업 스텝 수")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--no-amp", action="store_true", help="CUDA 혼합 정밀도 비활성화")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    return parser


def build_infer_parser() -> argparse.ArgumentParser:
    """실시간 스트리밍 음성·텍스트 추론 실행용 ArgumentParser를 구성해 반환합니다."""
    parser = argparse.ArgumentParser(description="EDarch 실시간 스트리밍 동시통역 추론 스크립트")
    add_env_cli_args(parser, include_seed=False)

    parser.add_argument("--checkpoint", type=Path, default=get_default_checkpoint_path(), help="체크포인트 경로")
    parser.add_argument("--preset", choices=("1b", "2b"), default="1b", help="체크포인트 부재 시 사용할 프리셋")
    parser.add_argument("--mimi-model", type=str, default=None, help="외부 사전학습 Mimi 코덱 ID 또는 경로")
    parser.add_argument(
        "--sp-model",
        "--tokenizer",
        dest="sp_model",
        type=Path,
        default=None,
        help="한국어 텍스트 디코딩용 SentencePiece 토크나이저 경로 (.model)",
    )
    parser.add_argument("--input-wav", type=Path, default=None, help="입력 영어 음성 WAV 경로")
    parser.add_argument(
        "--output-wav",
        type=Path,
        default=PROJECT_DIR / "outputs" / "EDarch_out.wav",
        help="번역된 한국어 음성 WAV 저장 경로",
    )
    parser.add_argument("--demo-frames", type=int, default=20, help="합성 데모 입력 프레임 수")
    parser.add_argument("--max-tail-sec", type=float, default=10.0, help="원문 종료 후 무음 꼬리 최대 길이(초)")
    parser.add_argument("--cfg-coef", type=float, default=3.0, help="Classifier-Free Guidance 계수")
    parser.add_argument("--temp-audio", type=float, default=0.8, help="오디오 샘플링 온도")
    parser.add_argument("--temp-text", type=float, default=0.4, help="텍스트 샘플링 온도")
    parser.add_argument("--top-k-audio", type=int, default=250, help="오디오 Top-k")
    parser.add_argument("--top-k-text", type=int, default=50, help="텍스트 Top-k")
    parser.add_argument("--no-amp", action="store_true", help="CUDA 혼합 정밀도 비활성화")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    return parser


def build_verify_parser() -> argparse.ArgumentParser:
    """코어 아키텍처·SentencePiece 토크나이저 검증 및 단독 토크나이저 학습용 ArgumentParser를 반환합니다."""
    parser = argparse.ArgumentParser(description="EDarch 코어 모델 및 한국어 SentencePiece 토크나이저 검증·관리 도구")
    add_env_cli_args(parser, include_seed=True)

    sub = parser.add_subparsers(dest="cmd")

    p_all = sub.add_parser("all", help="SentencePiece 토크나이저와 EDarch 신경망 아키텍처를 모두 검증")
    p_all.add_argument("--preset", choices=("1b", "2b"), default="1b")

    sub.add_parser("tokenizer", help="한국어 SentencePiece 토크나이저만 단독 검증")

    p_model = sub.add_parser("model", help="EDarchModel 초기화·역전파·스트리밍 추론 무결성 검증")
    p_model.add_argument("--preset", choices=("1b", "2b"), default="1b")

    p_train_tok = sub.add_parser("train-tokenizer", help="코퍼스 또는 AI-Hub JSON에서 한국어 SentencePiece 모델(.model) 생성")
    p_train_tok.add_argument("--input", type=Path, required=True, help=".txt 코퍼스 파일 또는 AI-Hub JSON 파일/디렉터리")
    p_train_tok.add_argument("--output", type=Path, required=True, help="출력 SentencePiece .model 경로")
    p_train_tok.add_argument("--vocab-size", type=int, default=32000)
    p_train_tok.add_argument("--model-type", choices=("unigram", "bpe"), default="unigram")
    p_train_tok.add_argument("--character-coverage", type=float, default=0.9995)
    p_train_tok.add_argument("--no-byte-fallback", action="store_true")
    p_train_tok.add_argument("--use-mt-fallback", action="store_true", help="71693 검수 번역문 누락 시 기계번역(MT) 포함")

    return parser
