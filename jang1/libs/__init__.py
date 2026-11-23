"""arXiv:2309.11093 기반 한-영/영-한 싱어블 노래(가사·보컬) 번역 시스템 코어 라이브러리 패키지."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORT_MAP: dict[str, tuple[str, str]] = {
    "LyricTransConfig": ("libs.config", "LyricTransConfig"),
    "LyricTransModel": ("libs.model", "LyricTransModel"),
    "LyricForwardOutput": ("libs.model", "LyricForwardOutput"),
    "LyricSequenceLogProbs": ("libs.model", "LyricSequenceLogProbs"),
    "adapt_model_for_lyric_vocab": ("libs.model", "adapt_model_for_lyric_vocab"),
    "MiniMimiCodec": ("libs.codec", "MiniMimiCodec"),
    "ExternalMimiCodec": ("libs.codec", "ExternalMimiCodec"),
    "LyricStreamingGenerator": ("libs.streaming", "LyricStreamingGenerator"),
    "train_lyric_sentencepiece": ("libs.tokenizer", "train_lyric_sentencepiece"),
    "load_lyric_tokenizer": ("libs.tokenizer", "load_lyric_tokenizer"),
    "render_section_with_controls": ("libs.tokenizer", "render_section_with_controls"),
    "split_rendered_section_lines": ("libs.tokenizer", "split_rendered_section_lines"),
    "count_korean_syllables": ("libs.phonetics", "count_korean_syllables"),
    "count_english_syllables": ("libs.phonetics", "count_english_syllables"),
    "count_line_syllables": ("libs.phonetics", "count_line_syllables"),
    "extract_korean_phonemes": ("libs.phonetics", "extract_korean_phonemes"),
    "extract_english_phonemes": ("libs.phonetics", "extract_english_phonemes"),
    "extract_line_phonemes": ("libs.phonetics", "extract_line_phonemes"),
    "extract_end_rhyme_phonemes": ("libs.phonetics", "extract_end_rhyme_phonemes"),
    "compute_phoneme_ssm": ("libs.phonetics", "compute_phoneme_ssm"),
    "compute_semantic_ssm": ("libs.phonetics", "compute_semantic_ssm"),
    "compute_ssm_alignment_score": ("libs.phonetics", "compute_ssm_alignment_score"),
    "compute_line_scd": ("libs.phonetics", "compute_line_scd"),
    "compute_section_scd": ("libs.phonetics", "compute_section_scd"),
    "detect_vocables_and_codeswitch": ("libs.phonetics", "detect_vocables_and_codeswitch"),
    "LyricLinePair": ("libs.preprocessing_common", "LyricLinePair"),
    "LyricSectionUnit": ("libs.preprocessing_common", "LyricSectionUnit"),
    "assemble_lyric_section_example": ("libs.preprocessing_common", "assemble_lyric_section_example"),
    "encode_mimi_timeline": ("libs.preprocessing_common", "encode_mimi_timeline"),
    "resolve_default_sim_bin": ("libs.preprocessing_common", "resolve_default_sim_bin"),
    "LyricSplitLayout": ("libs.dataset_parsing", "LyricSplitLayout"),
    "discover_lyric_splits": ("libs.dataset_parsing", "discover_lyric_splits"),
    "parse_lyric_split": ("libs.dataset_parsing", "parse_lyric_split"),
    "build_synthetic_kpop_sections": ("libs.dataset_parsing", "build_synthetic_kpop_sections"),
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
