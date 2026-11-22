"""토크나이저, 모델 만들기, 단위(unit)를 글로 펼치기."""
from __future__ import annotations

import random
import re

from ..config import (D_MODEL, EXTRA_TOKENS, FFN_DIM, MAX_POS, N_HEADS, N_LAYERS, SEP_TOKEN, SYL_MAX,
                      TOKENIZER, VOCAB_SIZE)
from ..text import en_syllables, ko_syllables

SPLIT_RE = re.compile(r"\s*(?:<SYL\d+>|<SEP>)\s*")


def get_tokenizer():
    from transformers import MarianTokenizer
    tok = MarianTokenizer.from_pretrained(TOKENIZER)
    tok.add_tokens(EXTRA_TOKENS)
    assert len(tok) == VOCAB_SIZE, len(tok)
    return tok


def new_model(tok):
    """4.1절: Marian 구조, 무작위 초기화. 값은 표 4 (해석은 config.py)."""
    from transformers import MarianConfig, MarianMTModel
    base = MarianConfig.from_pretrained(TOKENIZER)
    cfg = MarianConfig(
        vocab_size=VOCAB_SIZE, decoder_vocab_size=VOCAB_SIZE, max_position_embeddings=MAX_POS,
        d_model=D_MODEL, encoder_layers=N_LAYERS, decoder_layers=N_LAYERS,
        encoder_attention_heads=N_HEADS, decoder_attention_heads=N_HEADS,
        encoder_ffn_dim=FFN_DIM, decoder_ffn_dim=FFN_DIM,
        activation_function=base.activation_function, dropout=base.dropout,
        attention_dropout=base.attention_dropout, activation_dropout=base.activation_dropout,
        scale_embedding=base.scale_embedding, static_position_embeddings=True,
        share_encoder_decoder_embeddings=True,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
        decoder_start_token_id=tok.pad_token_id, forced_eos_token_id=tok.eos_token_id,
        bad_words_ids=[[tok.pad_token_id]],
    )
    return MarianMTModel(cfg)


def syl_token(n: int) -> str:
    return f"<SYL{min(max(n, 1), SYL_MAX)}>"


def render(src: list[str], tgt: list[str] | None, syl: bool, syl_counts: list[int] | None = None):
    """조각 목록을 모델 입력/정답 글로.

    syl=True : 각 조각 앞에 <SYLs>. s 는 학습 때 영어 정답 조각의 음절 수 (4.2절),
               추론 때는 syl_counts (원문 한국어 줄 음절 수, 4.4절).
    syl=False: 조각 사이에 <SEP> (4.2절 "sections are split using <SEP> tokens"). 조각이 하나면 그대로.
    """
    if syl:
        if syl_counts is None:
            syl_counts = [en_syllables(t) for t in tgt]
        toks = [syl_token(s) for s in syl_counts]
        s = " ".join(f"{k} {x}" for k, x in zip(toks, src))
        t = None if tgt is None else " ".join(f"{k} {x}" for k, x in zip(toks, tgt))
    else:
        s = f" {SEP_TOKEN} ".join(src)
        t = None if tgt is None else f" {SEP_TOKEN} ".join(tgt)
    return s, t


def render_unit(unit: dict, syl: bool, rng: random.Random | None):
    src, tgt = list(unit["src"]), list(unit["tgt"])
    if unit.get("shuffle") and rng is not None and len(src) > 1:
        order = list(range(len(src)))
        rng.shuffle(order)
        src, tgt = [src[i] for i in order], [tgt[i] for i in order]
    return render(src, tgt, syl)


def source_syllables(lines: list[str]) -> list[int]:
    return [ko_syllables(x) for x in lines]


def split_output(text: str) -> list[str]:
    text = text.replace("</s>", "").replace("<pad>", "").strip()
    parts = [p.strip() for p in SPLIT_RE.split(text)]
    if parts and parts[0] == "":
        parts = parts[1:]
    return parts


def fit_lines(parts: list[str], n: int) -> list[str]:
    """[해석] 마디 단위 출력의 줄 수가 원문과 다르면: 모자라면 빈 줄을 채우고, 넘치면 남는 줄을 마지막 줄에 붙인다."""
    if len(parts) >= n:
        return parts[:n - 1] + [" ".join(parts[n - 1:])] if n > 0 else []
    return parts + [""] * (n - len(parts))
