"""Space-normalized Word Error Rate (sWER) DP & traceback algorithm (Bang et al., 2020 Appendix B Alg. A1 & A2)."""

from __future__ import annotations

import re

from mos_sdh.schemas.document import SWERComparison

_NORM_RE = re.compile(r"[^\w가-힣\s]+")
_PAREN_RE = re.compile(r"(\([^)]*\)|\[[^\]]*\])")


def normalize_for_eval(text: str) -> str:
    """Strip SDH stage directions and punctuation while preserving words and spaces."""
    no_cues = _PAREN_RE.sub(" ", text)
    cleaned = _NORM_RE.sub(" ", no_cues)
    return " ".join(cleaned.split()).strip()


def char_tokenizing(text: str) -> list[str]:
    """Bind leading space '_' to the immediately following syllable character (Bang et al., 2020 Alg. A2).

    Example:
        '오늘 날씨가 좋네요' -> ['오', '늘', '_날', '씨', '가', '_좋', '네', '요']
    """
    norm = " ".join(text.strip().split())
    if not norm:
        return []

    tokens: list[str] = []
    pending_space = False
    for ch in norm:
        if ch == " ":
            pending_space = True
        else:
            tokens.append(f"_{ch}" if pending_space else ch)
            pending_space = False
    return tokens


def remove_space_symbol(token: str) -> str:
    """Strip leading '_' space marker to expose pure syllable character."""
    return token[1:] if token.startswith("_") else token


def get_norm_text(ref: str, hyp: str) -> tuple[str, str]:
    """Implement Bang et al. (2020) Algorithm A2: space-agnostic syllable DP alignment & space sync traceback."""
    refs = char_tokenizing(ref)
    hyps = char_tokenizing(hyp)
    rlen = len(refs)
    hlen = len(hyps)

    if rlen == 0 or hlen == 0:
        return " ".join(ref.split()), " ".join(hyp.split())

    # Cumulative edit distance matrix scores[(hlen+1) x (rlen+1)] and backpointers
    # 0 = diag (sub_or_cor), 1 = ins (from h-1, r), 2 = del (from h, r-1)
    scores = [[0] * (rlen + 1) for _ in range(hlen + 1)]
    backptr = [[0] * (rlen + 1) for _ in range(hlen + 1)]

    for r in range(1, rlen + 1):
        scores[0][r] = r
        backptr[0][r] = 2
    for h in range(1, hlen + 1):
        scores[h][0] = scores[h - 1][0] + 1
        backptr[h][0] = 1

    for h in range(1, hlen + 1):
        hyp_nosp = remove_space_symbol(hyps[h - 1])
        row_scores = scores[h]
        prev_scores = scores[h - 1]
        row_bp = backptr[h]

        for r in range(1, rlen + 1):
            ref_nosp = remove_space_symbol(refs[r - 1])
            sub_or_cor = prev_scores[r - 1] + (0 if hyp_nosp == ref_nosp else 1)
            ins = prev_scores[r] + 1
            del_cost = row_scores[r - 1] + 1

            if sub_or_cor <= ins and sub_or_cor <= del_cost:
                row_scores[r] = sub_or_cor
                row_bp[r] = 0
            elif ins <= del_cost:
                row_scores[r] = ins
                row_bp[r] = 1
            else:
                row_scores[r] = del_cost
                row_bp[r] = 2

    # Traceback from (hlen, rlen) to (0, 0) synchronizing space symbols on matching syllables
    synced_hyps = list(hyps)
    h, r = hlen, rlen
    while h > 0 or r > 0:
        move = backptr[h][r]
        if h > 0 and r > 0 and move == 0:
            c_hyp = synced_hyps[h - 1]
            c_ref = refs[r - 1]
            if remove_space_symbol(c_hyp) == remove_space_symbol(c_ref):
                synced_hyps[h - 1] = c_ref
            h -= 1
            r -= 1
        elif h > 0 and (r == 0 or move == 1):
            h -= 1
        else:
            r -= 1

    ref_wordnorm = "".join(refs).replace("_", " ").strip()
    hyp_wordnorm = "".join(synced_hyps).replace("_", " ").strip()
    return ref_wordnorm, hyp_wordnorm


def levenshtein_distance(seq_ref: list[str], seq_hyp: list[str]) -> int:
    """Compute standard Levenshtein edit distance (S + D + I)."""
    n = len(seq_ref)
    m = len(seq_hyp)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        h_tok = seq_hyp[i - 1]
        for j in range(1, n + 1):
            cost = 0 if h_tok == seq_ref[j - 1] else 1
            curr[j] = min(prev[j - 1] + cost, prev[j] + 1, curr[j - 1] + 1)
        prev, curr = curr, prev
    return prev[n]


def evaluate_cer_wer_swer(ref_texts: list[str], hyp_texts: list[str]) -> SWERComparison:
    """Implement Bang et al. (2020) Algorithm A1 across utterance pairs: CER, WER, and sWER (%)."""
    if len(ref_texts) != len(hyp_texts):
        raise ValueError("ref_texts와 hyp_texts의 문장 수가 일치해야 합니다.")

    total_char_err = 0
    total_char_ref = 0
    total_word_err = 0
    total_word_ref = 0
    total_sword_err = 0
    total_sword_ref = 0
    valid_pairs = 0

    for raw_ref, raw_hyp in zip(ref_texts, hyp_texts, strict=True):
        ref = normalize_for_eval(raw_ref)
        hyp = normalize_for_eval(raw_hyp)
        if not ref and not hyp:
            continue
        valid_pairs += 1

        ref_chars = list(ref.replace(" ", ""))
        hyp_chars = list(hyp.replace(" ", ""))
        total_char_err += levenshtein_distance(ref_chars, hyp_chars)
        total_char_ref += max(1, len(ref_chars))

        ref_words = ref.split()
        hyp_words = hyp.split()
        total_word_err += levenshtein_distance(ref_words, hyp_words)
        total_word_ref += max(1, len(ref_words))

        ref_wn, hyp_wn = get_norm_text(ref, hyp)
        ref_swords = ref_wn.split()
        hyp_swords = hyp_wn.split()
        total_sword_err += levenshtein_distance(ref_swords, hyp_swords)
        total_sword_ref += max(1, len(ref_swords))

    cer = round((total_char_err / max(1, total_char_ref)) * 100.0, 2)
    wer = round((total_word_err / max(1, total_word_ref)) * 100.0, 2)
    swer = round((total_sword_err / max(1, total_sword_ref)) * 100.0, 2)

    return SWERComparison(
        cer_percent=cer,
        wer_percent=wer,
        swer_percent=swer,
        compared_utterances=valid_pairs,
    )
