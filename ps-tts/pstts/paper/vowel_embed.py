"""식 (8), (9) 모음 거리표 (Your-TTS 판).

논문 3.2절 'Estimation of Vowel Distance' 를 그대로 따른다.
1. TTS 학습 자료의 문장(한국어 6,000, 영어 5,000)을 3.1절 G2P 로 음소화한다(front.tokens, 빈 토큰 포함).
2. 문장마다 텍스트 인코더 출력에 선형 투영을 씌운 평균 벡터를 얻는다. Coqui VITS TextEncoder 의 stats = proj(x) 를
   둘로 나눈 앞 절반 m_p 다(front.encoder_means. 해석: 'the mean vector of each vowel ... by applying the linear
   projection to the embedding vectors from the transformer-based text encoder').
3. 문장 안 같은 모음 토큰 자리의 m_p 를 평균해 (문장, 모음)마다 벡터 하나를 만든다.
4. 모음마다 그 벡터들을 K-means 로 5개 중심에 묶고 5개 중심을 평균해 mu_i 로 둔다(식 8, N = 5).
   sklearn KMeans(n_clusters=5, n_init=10, random_state=seed). 벡터가 5개보다 적으면 그 개수만큼 묶는다(해석).
   논문은 'A group of mean vectors for a given vowel was projected to calculate its centroids' 라고 적었지만 무엇으로
   투영했는지 없고, 투영이 나오는 Fig. 5(a), (b) 는 2차원 t-SNE 그림이다. 그래서 투영은 그림용으로 보고 K-means 와 식 (9) 는
   m_p 공간 그대로에서 잰다(해석, README_paper).
5. 거리 = ||mu(KR, m) - mu(EN, n)|| (식 9, 유클리드).
6. 확인: 한국어 ㅜ 와 가장 가까운 영어 모음이 u 인지 본다(논문 Fig. 5c 의 예).

표에 드는 모음은 학습한 모델의 글자 목록에 있고 문장에서 한 번 이상 나온 한국어 모음 자모(호환 자모)와 영어 IPA 모음
글자다(논문: 'all Korean and English vowels available in the text decoder of the trained cross-lingual TTS model').
DTW 국소 비용(EmbedVowelTable.cost): 모음끼리는 식 (9) 거리, null 끼리 0, null 과 모음 사이는 표의 최대 거리(해석).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

from ..vowels import ordered_symbols
from . import g2p

METHOD = ("식 (8), (9): Your-TTS 텍스트 인코더 m_p 를 문장마다 모음별로 평균하고, 모음마다 KMeans 중심 k 개의 평균(mu)을 "
          "구해 한국어 x 영어 유클리드 거리를 잰다")
KO_U = "ㅜ"     # 호환 모음 자모 ㅜ(U+315C). Fig. 5c 의 확인 예
EN_U = "u"


class EncoderFrontend(Protocol):
    """모음표를 만들 때 쓰는 프런트엔드 부분(YourTTSFrontend). 구조로만 맞춘다."""
    def tokens(self, text: str, lang: str) -> list[str]: ...
    def encoder_means(self, toks: list[str], lang: str) -> np.ndarray: ...


# ---------------------------------------------------------------------------- 식 (8)
def sentence_vowel_means(toks: Sequence[str], means: np.ndarray, lang: str) -> dict[str, np.ndarray]:
    """한 문장 안에서 모음 토큰마다 그 자리들의 m_p 평균. means 꼴은 (C, 토큰 수)."""
    m = np.asarray(means, dtype=np.float64)
    if m.ndim != 2 or m.shape[1] != len(toks):
        raise ValueError(f"m_p 꼴 {m.shape} 가 토큰 {len(toks)}개와 맞지 않습니다.")
    where: dict[str, list[int]] = {}
    for i, tok in enumerate(toks):
        if g2p.is_vowel(tok, lang):
            where.setdefault(tok, []).append(i)
    return {v: m[:, idx].mean(axis=1) for v, idx in where.items()}


def centroid_mean(vectors: Sequence[np.ndarray] | np.ndarray, k: int = 5, seed: int = 0) -> tuple[np.ndarray, int]:
    """식 (8): 벡터들을 KMeans 로 k 개 중심에 묶고 중심들의 평균을 돌려준다. (mu, 실제 묶음 수).

    벡터가 k 개보다 적으면 그 개수만큼 묶는다(해석). 그때 중심은 벡터 자신이라 mu 는 벡터 평균과 같다.
    """
    from sklearn.cluster import KMeans

    x = np.asarray(vectors, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError(f"벡터가 없거나 꼴이 잘못됐습니다: {x.shape}")
    if k < 1:
        raise ValueError(f"묶음 수는 1 이상이어야 합니다: {k}")
    n = min(int(k), x.shape[0])
    km = KMeans(n_clusters=n, n_init=10, random_state=seed).fit(x)
    return km.cluster_centers_.mean(axis=0), n


# ---------------------------------------------------------------------------- 만들기
def _collect(front: EncoderFrontend, sentences: Sequence[str], lang: str) -> tuple[dict[str, list[np.ndarray]], dict]:
    """문장마다 (모음 -> 문장 평균 벡터). 빈 문장과 모델 글자 목록에 없는 토큰이 있는 문장은 건너뛰고 센다."""
    per_vowel: dict[str, list[np.ndarray]] = {}
    used = empty = oov = 0
    oov_tokens: Counter[str] = Counter()
    unknown = getattr(front, "unknown_tokens", None)
    for text in sentences:
        if not str(text).strip():
            empty += 1
            continue
        toks = list(front.tokens(text, lang))
        if unknown is not None:
            bad = unknown(toks)
            if bad:
                oov += 1
                oov_tokens.update(bad)
                continue
        means = front.encoder_means(toks, lang)
        for v, vec in sentence_vowel_means(toks, means, lang).items():
            per_vowel.setdefault(v, []).append(vec)
        used += 1
    stats = {"given": len(sentences), "used": used, "skipped_empty": empty, "skipped_unknown_tokens": oov,
             "unknown_tokens": dict(oov_tokens.most_common())}
    return per_vowel, stats


def _order(symbols: Sequence[str], lang: str) -> list[str]:
    if lang == "ko":
        return sorted(set(symbols))         # 호환 모음 자모 순서(U+314F ㅏ ... U+3163 ㅣ, Fig. 5c 의 한국어 축과 같음)
    return ordered_symbols(symbols)         # 기존 모음표와 같은 혀 앞->뒤, 높음->낮음 순서


def build_vowel_table(front: EncoderFrontend, ko_sents: Sequence[str], en_sents: Sequence[str], k: int = 5,
                      seed: int = 0) -> dict[str, Any]:
    """식 (8), (9) 모음표 문서(JSON 으로 쓸 수 있는 dict)를 만든다. 행 = 한국어 모음, 열 = 영어 모음."""
    ko_vecs, ko_stats = _collect(front, ko_sents, "ko")
    en_vecs, en_stats = _collect(front, en_sents, "en")
    if not ko_vecs or not en_vecs:
        raise ValueError(f"모음 벡터가 없습니다(한국어 모음 {len(ko_vecs)}개, 영어 모음 {len(en_vecs)}개). "
                         "문장과 모델 글자 목록을 확인하세요.")
    ko = _order(ko_vecs, "ko")
    en = _order(en_vecs, "en")
    mu: dict[str, np.ndarray] = {}
    clusters: dict[str, int] = {}
    for sym, vecs in [*((s, ko_vecs[s]) for s in ko), *((s, en_vecs[s]) for s in en)]:
        mu[sym], clusters[sym] = centroid_mean(vecs, k, seed)
    mk = np.stack([mu[s] for s in ko])
    me = np.stack([mu[s] for s in en])
    matrix = np.sqrt(((mk[:, None, :] - me[None, :, :]) ** 2).sum(axis=-1))     # 식 (9)

    doc: dict[str, Any] = {
        "method": METHOD,
        "k": int(k),
        "seed": int(seed),
        "ko": ko,
        "en": en,
        "matrix": matrix.tolist(),
        "max_distance": float(matrix.max()),
        "mu": {s: mu[s].tolist() for s in [*ko, *en]},
        "dim": int(mk.shape[1]),
        "n_clusters": clusters,
        "n_sentences_per_vowel": {s: len(ko_vecs[s]) for s in ko} | {s: len(en_vecs[s]) for s in en},
        "sentences": {"ko": ko_stats, "en": en_stats},
        "checkpoint": {"path": str(getattr(front, "checkpoint_path", "") or ""),
                       "sha1": getattr(front, "checkpoint_sha1", None)},
    }
    vocab = getattr(front, "vocab_vowels", None)
    if callable(vocab):
        doc["vocab_vowels_without_data"] = {"ko": sorted(set(vocab("ko")) - set(ko)),
                                            "en": sorted(set(vocab("en")) - set(en))}
    doc["nearest"] = nearest_english(doc)
    doc["check_u"] = check_u(doc)
    return doc


def nearest_english(doc: dict[str, Any]) -> dict[str, list]:
    """한국어 모음마다 가장 가까운 영어 모음과 거리. 같으면 열 순서가 앞선 것."""
    m = np.asarray(doc["matrix"], dtype=np.float64)
    out: dict[str, list] = {}
    for i, k in enumerate(doc["ko"]):
        j = int(np.argmin(m[i]))
        out[k] = [doc["en"][j], float(m[i, j])]
    return out


def check_u(doc: dict[str, Any]) -> dict[str, Any]:
    """Fig. 5c 확인: ㅜ 와 가장 가까운 영어 모음이 u 인지. 둘 중 하나가 표에 없으면 판단하지 않는다(None)."""
    if KO_U not in doc["ko"] or EN_U not in doc["en"]:
        return {"ko": KO_U, "nearest_en": None, "is_u": None,
                "note": "ㅜ 나 u 가 표에 없어 확인하지 못했습니다."}
    en, dist = nearest_english(doc)[KO_U]
    i = doc["ko"].index(KO_U)
    du = float(np.asarray(doc["matrix"], dtype=np.float64)[i, doc["en"].index(EN_U)])
    return {"ko": KO_U, "nearest_en": en, "distance": dist, "distance_to_u": du, "is_u": en == EN_U}


# ---------------------------------------------------------------------------- 표 쓰기와 읽기
def _write_atomic(path: Path, text: str, encoding: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(text, encoding=encoding, newline="")
    os.replace(tmp, path)


def save_vowel_table(doc: dict[str, Any], json_path: str | os.PathLike, csv_path: str | os.PathLike | None = None
                     ) -> tuple[Path, Path]:
    """JSON(문서 전체)과 같은 이름 CSV(한국어 행 x 영어 열 거리)를 쓴다. 행렬과 벡터는 한 줄에 하나씩 적는다."""
    jp = Path(json_path)
    cp = Path(csv_path) if csv_path is not None else jp.with_suffix(".csv")

    def one(obj: object) -> str:
        return json.dumps(obj, ensure_ascii=False)

    parts = []
    for key, val in doc.items():
        if key == "matrix":
            rows = ",\n".join(f"  {one([round(float(v), 6) for v in row])}" for row in val)
            parts.append(f' "matrix": [\n{rows}\n ]')
        elif key == "mu":
            vecs = ",\n".join(f"  {one(s)}: {one([round(float(v), 6) for v in vec])}" for s, vec in val.items())
            parts.append(f' "mu": {{\n{vecs}\n }}')
        else:
            parts.append(f" {one(key)}: {one(val)}")
    text = "{\n" + ",\n".join(parts) + "\n}\n"
    json.loads(text)                                    # 쓰기 전에 형식 확인
    _write_atomic(jp, text, "utf-8")

    buf = StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["ko", *doc["en"]])
    for k, row in zip(doc["ko"], doc["matrix"]):
        w.writerow([k, *(f"{float(v):.6f}" for v in row)])
    _write_atomic(cp, buf.getvalue(), "utf-8-sig")
    return jp, cp


# (절대 경로, 크기, 고친 시각 ns) -> sha1. 파이프라인이 모델 1 앞에서 한 번, 프런트엔드가 한 번 체크포인트를 재므로 큰 파일은
# 다시 읽지 않는다. 작은 파일(모음표 등)은 Windows 시각 눈금(수 ms) 안에 같은 크기로 다시 쓰일 수 있어 늘 새로 잰다.
_SHA1_SEEN: dict[tuple[str, int, int], str] = {}
_SHA1_CACHE_MIN_BYTES = 1 << 20


def file_sha1(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    """파일 내용의 sha1(체크포인트와 모음표 파일을 가리는 데 쓴다). 1 MiB 이상이고 크기와 고친 시각이 같으면 다시 읽지 않는다."""
    p = Path(path).resolve()
    st = p.stat()
    key = (str(p), st.st_size, st.st_mtime_ns)
    big = st.st_size >= _SHA1_CACHE_MIN_BYTES
    if big and key in _SHA1_SEEN:
        return _SHA1_SEEN[key]
    h = hashlib.sha1()
    with open(p, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    digest = h.hexdigest()
    if big:
        _SHA1_SEEN[key] = digest
    return digest


@dataclass(eq=False)
class EmbedVowelTable:
    """식 (9) 거리표. 행 = 한국어 모음(ko), 열 = 영어 모음(en)."""
    ko: list[str]
    en: list[str]
    matrix: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=np.float64)
        if self.matrix.shape != (len(self.ko), len(self.en)) or self.matrix.size == 0:
            raise ValueError(f"모음표 행렬 꼴 {self.matrix.shape} 가 한국어 {len(self.ko)} x 영어 {len(self.en)} 와 "
                             "맞지 않습니다.")
        if not np.all(np.isfinite(self.matrix)) or np.any(self.matrix < 0):
            raise ValueError("모음표 거리에 음수나 NaN 이 있습니다.")
        self.ko_index = {s: i for i, s in enumerate(self.ko)}
        self.en_index = {s: i for i, s in enumerate(self.en)}

    @property
    def max_distance(self) -> float:
        """null 과 모음 사이 비용(해석: 표의 최대 거리)."""
        return float(self.matrix.max())

    def distance(self, ko: str, en: str) -> float:
        return float(self.matrix[self.ko_index[ko], self.en_index[en]])

    def symbols(self, lang: str) -> list[str]:
        return list(self.en if g2p.check_lang(lang) == "en" else self.ko)

    def cost(self, src_labels: Sequence[str | None], tgt_labels: Sequence[str | None]) -> np.ndarray:
        """DTW 국소 비용 (len(src), len(tgt)). src = 영어 원문 라벨, tgt = 한국어 후보 라벨.

        모음끼리 식 (9) 거리, null(None) 끼리 0, null 과 모음 사이 max_distance(해석).
        """
        n_en, n_ko = len(self.en), len(self.ko)
        ext = np.zeros((n_en + 1, n_ko + 1), dtype=np.float64)
        ext[:n_en, :n_ko] = self.matrix.T
        ext[n_en, :n_ko] = self.max_distance
        ext[:n_en, n_ko] = self.max_distance

        def index(labels: Sequence[str | None], table: dict[str, int], null: int, side: str) -> np.ndarray:
            try:
                return np.fromiter((null if lab is None else table[lab] for lab in labels),
                                   dtype=np.intp, count=len(labels))
            except KeyError as e:
                raise KeyError(f"모음표의 {side} 모음이 아닙니다: {e.args[0]!r}") from None

        rows = index(src_labels, self.en_index, n_en, "영어")
        cols = index(tgt_labels, self.ko_index, n_ko, "한국어")
        return ext[np.ix_(rows, cols)]


def table_from_doc(doc: dict[str, Any]) -> EmbedVowelTable:
    return EmbedVowelTable(ko=list(doc["ko"]), en=list(doc["en"]), matrix=np.asarray(doc["matrix"], dtype=np.float64),
                           meta={k: v for k, v in doc.items() if k not in ("ko", "en", "matrix", "mu")})


def load_vowel_table(path: str | os.PathLike) -> EmbedVowelTable:
    """save_vowel_table 이 쓴 JSON 을 읽는다. meta 에 파일 경로와 sha1 을 더한다."""
    p = Path(path)
    doc = json.loads(p.read_text(encoding="utf-8"))
    table = table_from_doc(doc)
    table.meta["file"] = str(p.resolve())
    table.meta["file_sha1"] = file_sha1(p)
    return table
