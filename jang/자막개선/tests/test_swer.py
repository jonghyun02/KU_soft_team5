"""Unit tests for Bang et al. (2020) Appendix B Algorithm A1 & A2 (CER, WER, sWER)."""

from __future__ import annotations

from mos_sdh.stage4_mos.swer_evaluator import (
    char_tokenizing,
    evaluate_cer_wer_swer,
    get_norm_text,
)


def test_char_tokenizing_binds_leading_space() -> None:
    tokens = char_tokenizing("오늘 날씨가 좋네요")
    assert tokens == ["오", "늘", "_날", "씨", "가", "_좋", "네", "요"]


def test_swer_ignores_pure_spacing_differences() -> None:
    ref = ["오늘 날씨가 참 좋네요"]
    hyp = ["오늘날씨가 참좋네요"]

    res = evaluate_cer_wer_swer(ref, hyp)
    assert res.cer_percent == 0.0
    assert res.wer_percent == 100.0  # Raw WER is heavily inflated by spacing
    assert res.swer_percent == 0.0   # Space-normalized sWER recognizes 100% syllable match


def test_swer_penalizes_real_syllable_substitution_without_spacing_inflation() -> None:
    ref = ["오늘 날씨가 참 좋네요"]  # 4 words
    hyp = ["오늘날씨가 참 좋내요"]   # spacing merge on word 1+2, real typo '내' on word 4

    ref_wn, hyp_wn = get_norm_text(ref[0], hyp[0])
    assert ref_wn == "오늘 날씨가 참 좋네요"
    assert hyp_wn == "오늘 날씨가 참 좋내요"

    res = evaluate_cer_wer_swer(ref, hyp)
    assert res.swer_percent == 25.0  # Exactly 1 of 4 words wrong


def test_swer_strips_sdh_stage_descriptors() -> None:
    ref = ["잠시만 기다려 주세요"]
    hyp = ["[화자_1] (한숨) 잠시만기다려 주세요."]

    res = evaluate_cer_wer_swer(ref, hyp)
    assert res.cer_percent == 0.0
    assert res.swer_percent == 0.0
