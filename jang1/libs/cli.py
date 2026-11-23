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


def build_preprocess_datasets_parser() -> argparse.ArgumentParser:
    """한-영 병렬 노래 가사(Section/Line) 및 보컬 오디오 전처리용 ArgumentParser를 반환합니다."""
    parser = argparse.ArgumentParser(
        description="arXiv:2309.11093 기반 노래 가사(Section/Line)·음절·음소·보컬 .pt 일괄 전처리 스크립트"
    )
    add_env_cli_args(parser, include_seed=True)

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="원천 병렬 노래 가사(JSON/JSONL/CSV 및 선택적 WAV) 디렉터리 (미지정 시 내장 K-pop 합성 데모 코퍼스 생성)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "outputs" / "preprocessed_lyrics",
        help="전처리된 train/val .pt 및 lyric_spm.model 저장 루트 경로",
    )
    parser.add_argument(
        "--direction",
        choices=("ko2en", "en2ko", "both"),
        default="ko2en",
        help="번역 방향 (ko2en: 한->영, en2ko: 영->한, both: 양방향)",
    )
    parser.add_argument(
        "--splits",
        choices=("both", "train", "val"),
        default="both",
        help="전처리할 데이터 분할 선택 (기본값: both -> train 및 val 분리 저장)",
    )
    parser.add_argument(
        "--preset",
        choices=("tiny", "1b", "2b"),
        default="tiny",
        help="모델 프리셋 (tiny: 고속 검증용, 1b: 17스트림 표준, 2b: 33스트림 대형)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="한-영 통합 SentencePiece 목표 어휘 크기",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="기존 SentencePiece .model 경로 (미지정 시 <output-dir>/lyric_spm.model 자동 학습)",
    )
    parser.add_argument(
        "--codec",
        choices=("mini", "mimi"),
        default="mini",
        help="보컬 오디오 코덱 선택: mini(내장 경량 코덱, 기본값) 또는 mimi(사전학습 kyutai/mimi)",
    )
    parser.add_argument(
        "--mimi-model",
        type=str,
        default="kyutai/mimi",
        help="사전학습 Mimi 코덱 ID 또는 로컬 경로",
    )
    parser.add_argument(
        "--vocal-mode",
        choices=("auto", "synth-guide", "text-only"),
        default="auto",
        help="보컬 오디오 처리 모드: auto(WAV 존재 시 인코딩, 없으면 리듬 가이드 생성), synth-guide, text-only",
    )
    parser.add_argument(
        "--align-mode",
        choices=("syllable", "energy", "uniform"),
        default="syllable",
        help="가사 토큰과 보컬 프레임 간 리듬 정렬 방식 (기본값: syllable)",
    )
    parser.add_argument(
        "--monologue-lead-frames",
        type=int,
        default=1,
        help="보컬 발화 대비 가사 토큰 선행 프레임 수",
    )
    parser.add_argument(
        "--max-lines-per-section",
        type=int,
        default=12,
        help="단일 섹션 샘플당 최대 가사 행 수 (초과 시 하위 섹션으로 분할)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="train 분할 최대 생성 섹션 샘플 수",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="val 분할 최대 생성 섹션 샘플 수",
    )
    parser.add_argument(
        "--demo-num-songs",
        type=int,
        default=12,
        help="--data-dir 미지정 시 생성할 합성 K-pop 노래 수",
    )
    parser.add_argument(
        "--io-threads",
        type=int,
        default=8,
        help="병렬 JSON/CSV 파싱 스레드 수",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 .pt 덮어쓰기 (기본값은 이어하기 활성)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="완료 후 데이터셋 로더 검증 생략",
    )
    return parser


def build_train_parser() -> argparse.ArgumentParser:
    """Stage 1 섹션·음절 조건부 지도학습(SFT) 실행용 ArgumentParser를 구성해 반환합니다."""
    parser = argparse.ArgumentParser(
        description="arXiv:2309.11093 Stage 1 섹션 문맥 및 목표 음절 수 조건부 지도학습(SFT) 스크립트"
    )
    add_env_cli_args(parser, include_seed=False)

    parser.add_argument(
        "--preset",
        choices=("tiny", "1b", "2b"),
        default="tiny",
        help="모델 프리셋 (기본값: tiny)",
    )
    parser.add_argument(
        "--direction",
        choices=("ko2en", "en2ko"),
        default="ko2en",
        help="번역 방향 (기본값: ko2en)",
    )
    parser.add_argument("--init-from", type=Path, default=None, help="사전학습 체크포인트 경로")
    parser.add_argument("--resume", type=Path, default=None, help="중단된 체크포인트에서 학습 재개")
    parser.add_argument("--vocab-size", type=int, default=None, help="토크나이저 어휘 크기")
    parser.add_argument("--grad-checkpointing", action="store_true", help="Gradient Checkpointing 활성화")
    parser.add_argument("--no-source-audio-loss", action="store_true", help="소스 보컬 보조 손실 비활성화")

    parser.add_argument("--data-dir", type=Path, default=None, help="전처리된 학습 .pt 폴더 경로")
    parser.add_argument("--val-data-dir", type=Path, default=None, help="전처리된 검증(val) .pt 폴더 경로")
    parser.add_argument("--crop-frames", type=int, default=None, help="최대 시퀀스 프레임 수")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 워커 수")
    parser.add_argument("--epochs", type=int, default=2, help="총 학습 에폭 수")
    parser.add_argument("--steps-per-epoch", type=int, default=6, help="합성 데모 모드 시 에폭당 스텝 수")
    parser.add_argument("--batch-size", type=int, default=2, help="배치 크기")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation 스텝 수")
    parser.add_argument("--num-frames", type=int, default=24, help="데모 모드 시 샘플 프레임 수")
    parser.add_argument("--lr", type=float, default=1e-4, help="최대 학습률")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--warmup-steps", type=int, default=None, help="LR 워밍업 옵티마이저 스텝 수")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--cfg-dropout", type=float, default=0.1, help="CFG 조건 드롭아웃 확률")
    parser.add_argument("--no-amp", action="store_true", help="CUDA 혼합 정밀도 비활성화")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=get_default_checkpoint_path(),
        help="Stage 1 체크포인트 저장 경로",
    )
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    return parser


def build_grpo_parser() -> argparse.ArgumentParser:
    """Stage 2 Singability GRPO 강화학습 실행용 ArgumentParser를 구성해 반환합니다."""
    parser = argparse.ArgumentParser(
        description="arXiv:2309.11093 Stage 2 Singability GRPO(음절 정합·음소 반복 SSM·각운·의미) 강화학습 스크립트"
    )
    add_env_cli_args(parser, include_seed=False)

    parser.add_argument(
        "--init-from",
        type=Path,
        default=get_default_checkpoint_path(),
        help="Stage 1 체크포인트 경로",
    )
    parser.add_argument("--resume", type=Path, default=None, help="중단된 Stage 2 체크포인트에서 재개")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=get_default_grpo_checkpoint_path(),
        help="Stage 2 체크포인트 저장 경로",
    )
    parser.add_argument("--preset", choices=("tiny", "1b", "2b"), default="tiny", help="모델 프리셋")
    parser.add_argument("--direction", choices=("ko2en", "en2ko"), default="ko2en", help="번역 방향")
    parser.add_argument("--vocab-size", type=int, default=None, help="어휘 크기")
    parser.add_argument("--grad-checkpointing", action="store_true", help="Gradient Checkpointing 활성화")

    parser.add_argument("--data-dir", type=Path, default=None, help="전처리된 학습 .pt 폴더 경로")
    parser.add_argument("--val-data-dir", type=Path, default=None, help="전처리된 검증(val) .pt 폴더 경로")
    parser.add_argument("--max-val-prompts", type=int, default=16, help="에폭당 검증 최대 프롬프트 수")
    parser.add_argument("--crop-src-frames", type=int, default=300, help="프롬프트 최대 소스 프레임 수")
    parser.add_argument("--max-tail-sec", type=float, default=4.0, help="롤아웃 무음 꼬리 최대 길이(초)")
    parser.add_argument(
        "--sp-model",
        "--tokenizer",
        dest="sp_model",
        type=Path,
        default=None,
        help="가사 디코딩 및 음절·음소·라임 평가용 SentencePiece 토크나이저 경로 (.model)",
    )
    parser.add_argument("--hooks-module", type=str, default=None, help="custom_reward_fn 제공 모듈명")

    parser.add_argument("--group-size", type=int, default=4, help="섹션 프롬프트당 롤아웃 그룹 크기 G")
    parser.add_argument("--ppo-epochs", type=int, default=1, help="그룹당 내부 정책 업데이트 횟수")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="중요도 비율 클리핑 범위")
    parser.add_argument("--kl-coef", type=float, default=0.02, help="기준 모델 대비 Schulman k3 KL 계수")
    parser.add_argument("--ref-device", choices=("cpu", "cuda"), default="cpu", help="기준 모델 디바이스")
    parser.add_argument("--audio-pg-weight", type=float, default=0.15, help="타깃 보컬 오디오 정책 경사 가중치")
    parser.add_argument("--rollout-cfg-coef", type=float, default=1.0, help="롤아웃 CFG 계수")
    parser.add_argument("--temp-text", type=float, default=0.7, help="롤아웃 텍스트 샘플링 온도")
    parser.add_argument("--temp-audio", type=float, default=0.8, help="롤아웃 오디오 샘플링 온도")
    parser.add_argument("--top-k-text", type=int, default=50, help="롤아웃 텍스트 Top-k")
    parser.add_argument("--top-k-audio", type=int, default=250, help="롤아웃 오디오 Top-k")

    parser.add_argument("--reward-bleu-weight", type=float, default=1.0, help="(1) 의미 정확도(BLEU) 보상 가중치")
    parser.add_argument(
        "--reward-syllable-weight",
        type=float,
        default=1.2,
        help="(2) 음절 수 정합(1 - SCD) 보상 가중치",
    )
    parser.add_argument(
        "--reward-ssm-weight",
        type=float,
        default=0.8,
        help="(3) 섹션 내 의미·음소 반복 행렬(SSM) 정합 보상 가중치",
    )
    parser.add_argument(
        "--reward-rhyme-weight",
        type=float,
        default=0.6,
        help="(4) 행 끝 라임(End-Rhyme)·반복 행 일관성·코드스위칭 보존 보상 가중치",
    )
    parser.add_argument("--reward-audio-weight", type=float, default=0.10, help="보컬 오디오 건강도 보상 가중치")

    parser.add_argument("--epochs", type=int, default=1, help="GRPO 학습 에폭 수")
    parser.add_argument("--steps-per-epoch", type=int, default=4, help="데모 모드 에폭당 섹션 프롬프트 수")
    parser.add_argument("--demo-src-frames", type=int, default=20, help="데모 모드 소스 프레임 수")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation 프롬프트 수")
    parser.add_argument("--lr", type=float, default=1e-5, help="학습률")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--warmup-steps", type=int, default=None, help="워밍업 스텝 수")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--no-amp", action="store_true", help="CUDA 혼합 정밀도 비활성화")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    return parser


def build_infer_parser() -> argparse.ArgumentParser:
    """노래 섹션 가사 및 보컬 스트리밍 추론 실행용 ArgumentParser를 구성해 반환합니다."""
    parser = argparse.ArgumentParser(
        description="arXiv:2309.11093 기반 싱어블 노래 가사(Section/Line) 및 보컬 스트리밍 번역 추론 스크립트"
    )
    add_env_cli_args(parser, include_seed=False)

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=get_default_grpo_checkpoint_path(),
        help="Stage 2 또는 Stage 1 체크포인트 경로",
    )
    parser.add_argument("--preset", choices=("tiny", "1b", "2b"), default="tiny", help="체크포인트 부재 시 프리셋")
    parser.add_argument("--direction", choices=("ko2en", "en2ko"), default="ko2en", help="번역 방향")
    parser.add_argument("--mimi-model", type=str, default=None, help="외부 사전학습 Mimi 코덱 ID 또는 경로")
    parser.add_argument(
        "--sp-model",
        "--tokenizer",
        dest="sp_model",
        type=Path,
        default=None,
        help="가사 텍스트 인코딩·디코딩용 SentencePiece 토크나이저 경로 (.model)",
    )
    parser.add_argument(
        "--input-lyrics",
        type=Path,
        default=None,
        help="입력 노래 가사 파일(.txt 또는 .json) 경로 (미지정 시 내장 K-pop 섹션 데모 가사 사용)",
    )
    parser.add_argument(
        "--section-type",
        type=str,
        default="chorus",
        help="입력 가사의 기본 섹션 종류 (verse, prechorus, chorus, bridge, hook, outro)",
    )
    parser.add_argument("--input-wav", type=Path, default=None, help="선택적 입력 원곡 보컬 WAV 경로")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_DIR / "outputs" / "translated_lyrics.json",
        help="번역된 싱어블 가사 및 음절·라임·SSM 분석 리포트 JSON 저장 경로",
    )
    parser.add_argument(
        "--output-wav",
        type=Path,
        default=PROJECT_DIR / "outputs" / "translated_vocal.wav",
        help="번역된 보컬 오디오 WAV 저장 경로",
    )
    parser.add_argument("--demo-frames", type=int, default=24, help="합성 데모 입력 프레임 수")
    parser.add_argument("--max-tail-sec", type=float, default=6.0, help="원곡 종료 후 무음 꼬리 최대 길이(초)")
    parser.add_argument("--eos-tail-frames", type=int, default=2, help="EOS 감지 후 추가 플러시할 프레임 수")
    parser.add_argument(
        "--cfg-coef",
        type=float,
        default=None,
        help="Classifier-Free Guidance 계수 (미지정 시 Stage 2는 1.0, Stage 1은 2.0 자동 선택)",
    )
    parser.add_argument("--sim-bin", type=int, default=None, help="보컬 음색 유사도 조건 구간 ID (0..4)")
    parser.add_argument("--temp-audio", type=float, default=0.8, help="오디오 샘플링 온도")
    parser.add_argument("--temp-text", type=float, default=0.4, help="가사 텍스트 샘플링 온도")
    parser.add_argument("--top-k-audio", type=int, default=250, help="오디오 Top-k")
    parser.add_argument("--top-k-text", type=int, default=50, help="가사 텍스트 Top-k")
    parser.add_argument("--no-amp", action="store_true", help="CUDA 혼합 정밀도 비활성화")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    return parser


def build_verify_parser() -> argparse.ArgumentParser:
    """음절·음소 분석기, SentencePiece 토크나이저 및 코어 모델 통합 검증용 ArgumentParser를 반환합니다."""
    parser = argparse.ArgumentParser(
        description="arXiv:2309.11093 싱어블 노래 번역 시스템(음절·음소·SSM·토크나이저·모델·GRPO) 통합 검증 도구"
    )
    add_env_cli_args(parser, include_seed=True)

    sub = parser.add_subparsers(dest="cmd")

    p_all = sub.add_parser("all", help="음절/음소 분석기, 토크나이저, 신경망 SFT/GRPO/추론을 모두 검증")
    p_all.add_argument("--preset", choices=("tiny", "1b", "2b"), default="tiny")

    sub.add_parser("phonetics", help="한-영 음절 카운팅, 연음 음소 추출, 라임, SSM 및 SCD 지표 단독 검증")

    sub.add_parser("tokenizer", help="한-영 노래 제어 토큰 내장 SentencePiece 토크나이저 단독 검증")

    p_model = sub.add_parser("model", help="LyricTransModel 초기화·SFT·GRPO·스트리밍 추론 무결성 검증")
    p_model.add_argument("--preset", choices=("tiny", "1b", "2b"), default="tiny")

    p_train_tok = sub.add_parser("train-tokenizer", help="가사 코퍼스에서 한-영 SentencePiece 모델(.model) 생성")
    p_train_tok.add_argument("--input", type=Path, required=True, help="가사 .txt/.json/.jsonl 파일 또는 디렉터리")
    p_train_tok.add_argument("--output", type=Path, required=True, help="출력 SentencePiece .model 경로")
    p_train_tok.add_argument("--vocab-size", type=int, default=8000)
    p_train_tok.add_argument("--model-type", choices=("unigram", "bpe"), default="unigram")
    p_train_tok.add_argument("--character-coverage", type=float, default=0.9995)
    p_train_tok.add_argument("--no-byte-fallback", action="store_true")

    return parser
