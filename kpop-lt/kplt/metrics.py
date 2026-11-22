"""평가 지표: sts, Semline (식 1), Semsec (식 2), SCD (식 4), error rate, NSP 점수.

한 곡 = 마디 목록, 마디 = 줄 목록. 원문(한국어)과 번역(영어)은 줄 단위로 맞춰져 있어야 한다.
"""
from __future__ import annotations

from statistics import mean

import numpy as np

from .config import NSP_BASE, STS_MODEL
from .text import syllables


# ---------------- 음절 ----------------
def line_scd(s: int, t: int) -> float:
    """식 4 의 한 줄 몫 (|s-t|/s + |s-t|/t) / 2.

    [해석] 한쪽이 0 음절이면 식이 정의되지 않는다. 그때는 정의되는 쪽 항 하나만 쓴다
    (t=0 이면 |s-t|/s = 1). 둘 다 0 이면 0.
    """
    d = abs(s - t)
    if s > 0 and t > 0:
        return (d / s + d / t) / 2
    if s == 0 and t == 0:
        return 0.0
    return 1.0


def scd(src_syl: list[int], tgt_syl: list[int]) -> float:
    """식 4: 1/(2n) * sum(|s-t|/s + |s-t|/t) = 줄 몫의 평균."""
    n = len(src_syl)
    return sum(line_scd(s, t) for s, t in zip(src_syl, tgt_syl)) / n


def song_scd(src: list[list[str]], tgt: list[list[str]], src_lang="ko", tgt_lang="en") -> float:
    s = [syllables(x, src_lang) for sec in src for x in sec]
    t = [syllables(x, tgt_lang) for sec in tgt for x in sec]
    return scd(s, t)


def error_lines(src: list[list[str]], tgt: list[list[str]], src_lang="ko", tgt_lang="en") -> tuple[int, int]:
    """(음절 수가 틀린 줄 수, 전체 줄 수). error rate = 전체 합으로 나눈다 (4.3절)."""
    s = [syllables(x, src_lang) for sec in src for x in sec]
    t = [syllables(x, tgt_lang) for sec in tgt for x in sec]
    return sum(a != b for a, b in zip(s, t)), len(s)


# ---------------- 의미 ----------------
class STS:
    """all-MiniLM-L6-v2 코사인 유사도. 한국어는 먼저 구글 번역으로 영어로 바꾼다 (3.1절).

    [해석] 한국어는 줄마다 번역하고, 마디 글은 번역된 줄을 공백으로 이은 것.
    """

    def __init__(self, translator=None, device=None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(STS_MODEL, device=device)
        self.translator = translator
        self._emb: dict[str, np.ndarray] = {}

    def to_en(self, lines: list[str], lang: str) -> list[str]:
        if lang == "en":
            return lines
        return self.translator.translate_lines(lines, "ko", "en")

    def embed(self, texts: list[str]) -> np.ndarray:
        todo = list(dict.fromkeys(t for t in texts if t not in self._emb))
        if todo:
            vecs = self.model.encode(todo, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
            for t, v in zip(todo, vecs):
                self._emb[t] = v
        return np.stack([self._emb[t] for t in texts]) if texts else np.zeros((0, 384))

    def pairs(self, a: list[str], b: list[str]) -> list[float]:
        ea, eb = self.embed(a), self.embed(b)
        return [float(x) for x in (ea * eb).sum(1)]

    def song(self, src: list[list[str]], tgt: list[list[str]], src_lang="ko", tgt_lang="en") -> dict:
        """한 곡의 Semline, Semsec 와 줄별 sts."""
        src_en = [self.to_en(sec, src_lang) for sec in src]
        tgt_en = [self.to_en(sec, tgt_lang) for sec in tgt]
        line_a = [x for sec in src_en for x in sec]
        line_b = [x for sec in tgt_en for x in sec]
        line_sts = self.pairs(line_a, line_b)
        n = len(line_sts)
        sec_a = [" ".join(sec) for sec in src_en]
        sec_b = [" ".join(sec) for sec in tgt_en]
        sec_sts = self.pairs(sec_a, sec_b)
        semline = sum(line_sts) / n                                     # 식 1
        semsec = sum(len(sec) / n * v for sec, v in zip(src, sec_sts))  # 식 2
        return {"semline": semline, "semsec": semsec, "line_sts": line_sts}


# ---------------- NSP ----------------
class NSPScorer:
    """미세조정한 BERT NSP 로 이웃한 두 줄이 이어지는 확률의 평균 (4.3절)."""

    def __init__(self, model_dir, device="cuda", batch_size=64):
        import torch
        from transformers import BertForNextSentencePrediction, BertTokenizerFast
        self.torch = torch
        self.tok = BertTokenizerFast.from_pretrained(model_dir)
        self.model = BertForNextSentencePrediction.from_pretrained(model_dir).to(device).eval()
        self.device = device
        self.bs = batch_size

    def probs(self, pairs: list[tuple[str, str]]) -> list[float]:
        out: list[float] = []
        for i in range(0, len(pairs), self.bs):
            chunk = pairs[i:i + self.bs]
            enc = self.tok([a for a, _ in chunk], [b for _, b in chunk], truncation=True, max_length=128,
                           padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                logits = self.model(**enc).logits
            # label 0 = "B 가 A 다음 문장"
            out.extend(self.torch.softmax(logits.float(), -1)[:, 0].tolist())
        return out

    def song(self, tgt: list[list[str]]) -> float | None:
        """[해석] 한 곡의 모든 줄을 순서대로 이어 놓고 이웃한 두 줄씩 (마디 경계 포함)."""
        lines = [x for sec in tgt for x in sec]
        pairs = list(zip(lines, lines[1:]))
        if not pairs:
            return None
        return mean(self.probs(pairs))


def empty_lines(tgt: list[list[str]]) -> int:
    """번역이 빈 줄 수 (마디 단위 모델이 줄을 덜 낸 경우, common.fit_lines 가 채운 줄)."""
    return sum(1 for sec in tgt for x in sec if not x.strip())


def mean_std(xs: list[float]) -> tuple[float, float]:
    a = np.asarray([x for x in xs if x is not None], dtype=float)
    return float(a.mean()), float(a.std())
