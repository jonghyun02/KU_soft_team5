"""EDarch 영-한 동시통역 시스템의 코어 모델, 코덱, 토크나이저, 전처리 및 학습 라이브러리 패키지."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORT_MAP: dict[str, tuple[str, str]] = {
    "EDarchConfig": ("libs.config", "EDarchConfig"),
    "EDarchModel": ("libs.model", "EDarchModel"),
    "EDarchForwardOutput": ("libs.model", "EDarchForwardOutput"),
    "EDarchSequenceLogProbs": ("libs.model", "EDarchSequenceLogProbs"),
    "adapt_EDarch_for_korean_vocab": ("libs.model", "adapt_EDarch_for_korean_vocab"),
    "MiniMimiCodec": ("libs.codec", "MiniMimiCodec"),
    "ExternalMimiCodec": ("libs.codec", "ExternalMimiCodec"),
    "EDarchStreamingGenerator": ("libs.streaming", "EDarchStreamingGenerator"),
    "train_korean_sentencepiece": ("libs.tokenizer", "train_korean_sentencepiece"),
    "load_korean_tokenizer": ("libs.tokenizer", "load_korean_tokenizer"),
    "UtteranceSpan": ("libs.preprocessing_common", "UtteranceSpan"),
    "BroadcastUtterance71379": ("libs.preprocessing_common", "BroadcastUtterance71379"),
    "assemble_multistream_example": ("libs.preprocessing_common", "assemble_multistream_example"),
    "encode_mimi_timeline": ("libs.preprocessing_common", "encode_mimi_timeline"),
    "resolve_default_sim_bin": ("libs.preprocessing_common", "resolve_default_sim_bin"),
    "SplitLayout71693": ("libs.dataset_parsing", "SplitLayout71693"),
    "SplitLayout71379": ("libs.dataset_parsing", "SplitLayout71379"),
    "discover_71693_splits": ("libs.dataset_parsing", "discover_71693_splits"),
    "discover_71379_splits": ("libs.dataset_parsing", "discover_71379_splits"),
    "parse_71693_split": ("libs.dataset_parsing", "parse_71693_split"),
    "parse_71379_split": ("libs.dataset_parsing", "parse_71379_split"),
    "run_preprocess_datasets": ("libs.dataset_builder", "run_preprocess_datasets"),
    "GRPOPromptSample": ("libs.grpo_rollout", "GRPOPromptSample"),
    "RolloutBatch": ("libs.grpo_rollout", "RolloutBatch"),
    "GRPOPromptDataset": ("libs.grpo_rollout", "GRPOPromptDataset"),
    "rollout_group_trajectories": ("libs.grpo_rollout", "rollout_group_trajectories"),
    "evaluate_rollout_group_rewards": ("libs.grpo_rollout", "evaluate_rollout_group_rewards"),
    "compute_grpo_loss": ("libs.grpo", "compute_grpo_loss"),
    "run_grpo_training": ("libs.grpo", "run_grpo_training"),
}

__all__ = list(_EXPORT_MAP.keys())


def __getattr__(name: str) -> Any:
    """패키지 최상위 심볼 접근 시 해당 하위 모듈을 지연 로드하여 불필요한 무거운 임포트를 방지합니다."""
    target = _EXPORT_MAP.get(name)
    if target is None:
        raise AttributeError(f"module 'libs' has no attribute {name!r}")
    mod_name, attr_name = target
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr_name)
