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
    "KoreanBPETokenizer": ("libs.tokenizer", "KoreanBPETokenizer"),
    "load_korean_tokenizer": ("libs.tokenizer", "load_korean_tokenizer"),
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
